from collections import defaultdict
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import repeat
from torchtyping import TensorType

from .action_models import (
    PigletActionApplyModel,
    PigletAnnotatedActionEncoder,
    PigletSymbolicActionEncoder,
)
from .analysis import (
    calculate_average_accuracy,
    calculate_diff_accuracy,
    log_action_accuracies,
    log_attribute_level_error_rates,
    log_average_attribute_accuracy,
    log_confusion_matrices,
    plot_images,
)
from .image_models import PigletImageEncoder
from .loss import calculate_avg_object_loss
from .mappings import get_actions_mapper, get_objects_mapper
from .object_models import PigletObjectDecoder, PigletObjectEncoder


class Piglet(pl.LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        hidden_size=256,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        object_embedding_size=329,
        num_attributes=38,
        none_object_index=102,
        action_embedding_size=10,
        data_dir_path="data",
        output_dir_path="output",
        pretrain=True,
        bert_model_name="roberta-base",
        encode_images=False,
    ):
        """
        Args:
          hidden_size: dimension of the hidden state of all layers
          num_layers: number of layers in the transformer encoder / decoder and action encoder
          num_heads: number of heads in the transformers
          dropout: dropout probability
          object_embedding_size: max embedding index of the object attributes
          num_attributes: number of attributes per object
          none_object_index: index of the none object
          action_embedding_size: max embedding index the action
          data_dir_path: path to the data directory
          output_dir_path: path to the output directory
          pretrain: whether in pretraining mode (no language model)
          bert_model_name: name of the bert model to use (must have pretrain = False)
          encode_images: whether to encode images
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.object_embedding_size = object_embedding_size
        self.none_object_index = none_object_index  # mask object tensor with zeroes if object = none_object_index
        self.action_embedding_size = action_embedding_size
        self.pretrain = pretrain
        self.encode_images = encode_images
        self.plotted_images = False

        # Image encoder
        if self.encode_images:
            self.object_embedding_layer = nn.Embedding(
                object_embedding_size, hidden_size, padding_idx=none_object_index
            )
            self.image_encoder = PigletImageEncoder(
                hidden_size=hidden_size, output_dir_path=output_dir_path
            )

        else:
            # Object Encoder Model
            self.object_encoder = PigletObjectEncoder(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                object_embedding_size=object_embedding_size,
                num_attributes=num_attributes,
                none_object_index=none_object_index,
            )

        # Action Encoder Model
        if self.pretrain:
            self.action_encoder = PigletSymbolicActionEncoder(
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                action_embedding_size=action_embedding_size,
            )
        else:
            self.action_encoder = PigletAnnotatedActionEncoder(
                hidden_size=hidden_size,
                bert_model_name=bert_model_name,
                output_dir_path=output_dir_path,
            )

        # Action Apply Model
        self.apply_action = PigletActionApplyModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Object Decoder
        self.object_decoder = PigletObjectDecoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            object_embedding_size=object_embedding_size,
            num_attributes=num_attributes,
            none_object_index=none_object_index,
            data_dir_path=data_dir_path,
        )

        self.action_idx_to_name = get_actions_mapper(data_dir_path)
        self.object_attributes_idx_to_mapper = get_objects_mapper(data_dir_path)

    def forward(
        self,
        object_inputs,
        action_inputs,
        action_text_inputs=None,
        image_inputs=None,
        training=True,
    ):
        # check that image inputs are not passed if no image encoder and vice versa
        assert (image_inputs is None) == (not self.encode_images)

        if self.pretrain:
            # sum the embedding of object targeted and its receptacle
            if self.encode_images:
                action_args_embeddings = self.object_embedding_layer(
                        action_inputs[:, 1:]
                ).sum(1)
            else:
                action_args_embeddings = self.object_encoder.object_embedding_layer(
                    action_inputs[:, 1:]
                ).sum(1)
            h_a = self.action_encoder(action_inputs[:, 0], action_args_embeddings)
        else:
            h_a = self.action_encoder(
                action_text_inputs["input_ids"], action_text_inputs["attention_mask"]
            )

        image_model_outputs = None
        if self.encode_images:
            # even if we only use images we still need an object_embedding layer for the object names
            object_names = self.object_embedding_layer(object_inputs[:, :, 0])
            conditional_vector = torch.cat(
                (repeat(h_a, "b h -> b 4 h"), object_names), dim=2
            )
            image_model_outputs = self.image_encoder(
                image_inputs, conditional_vector, training=training
            )
            h_o = image_model_outputs["h_i_o"]
        else:
            # embed object vector
            h_o = self.object_encoder(object_inputs)

        # select only the pre action objects
        h_o_pre = h_o[:, [0, 1], :]

        # apply action to object hidden representations
        h_o_a = self.apply_action(h_o_pre, h_a)

        # use transformer decoder and pass pre_object as src vector and h_o_a as the memory vector
        h_o_post_pred = self.object_decoder(h_o_a, h_o_pre)

        h_o_a_init = torch.zeros_like(h_o_a)
        h_o_pre_pred = self.object_decoder(h_o_a_init, h_o_pre)

        # returns predicted sequence of object attributes
        # torch.Size([batch_size*2, num_attributes, object_embedding_size])
        if self.encode_images and not training and image_model_outputs is not None:
            return h_o_post_pred, h_o_pre_pred, image_model_outputs

        return h_o_post_pred, h_o_pre_pred

    def calculate_object_attribute_loss_and_predictions(
        self,
        h_out: TensorType["batch_objects", "num_attributes", "object_embedding_size"],
        labels: TensorType["batch", 2, "num_attributes"],
    ) -> Tuple[torch.Tensor, TensorType["batch_objects", "num_attributes"]]:
        # mask the embedding layer output to only allow predictions for valid attributes at each position
        h_out = self.object_decoder.mask_output_probabilities(h_out)
        selection_mask = labels[:, :, 0] != self.none_object_index
        # calculate loss for each object (since it is a set we evaluate over both possibilities object 0 -> object label 0)
        avg_loss = calculate_avg_object_loss(h_out, labels, selection_mask)
        # get top 1 prediction for each object attribute
        predictions = h_out.argmax(-1)
        return avg_loss, predictions

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        objects = batch["objects"]
        actions = batch["actions"]
        images = batch.get("images", None)
        action_text_inputs = batch.get("action_text", None)

        # the objects vector contains four dimensions (pre_0, pre_1, post_0, post_1)
        # here we select the labels for both objects post action
        objects_labels_pre = objects[:, [0, 1], :]
        objects_labels_post = objects[:, [2, 3], :]

        # encode objects and actions, apply action, and predict resulting object attributes
        h_o_post_ouput, h_o_pre_ouput = self(
            objects, actions, action_text_inputs=action_text_inputs, image_inputs=images
        )
        avg_loss_post, _ = self.calculate_object_attribute_loss_and_predictions(
            h_o_post_ouput, objects_labels_post
        )
        avg_loss_pre, _ = self.calculate_object_attribute_loss_and_predictions(
            h_o_pre_ouput, objects_labels_pre
        )
        avg_loss = avg_loss_post + avg_loss_pre

        self.log("train/loss", avg_loss, batch_size=len(batch["objects"]))
        self.log("train/loss_post", avg_loss_post, batch_size=len(batch["objects"]))
        self.log("train/loss_pre", avg_loss_pre, batch_size=len(batch["objects"]))

        return avg_loss

    def process_inference_batch(
        self, batch, batch_idx, split="val"
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of data for inference.
        Returns a tuple of:
            predictions: predicted object attributes post-action
            objects_labels_pre: symbolic representation of the objects before action
            objects_labels_post: symbolic representation of the objects post-action
            action_inputs: symbolic representation of actions
        """
        objects = batch["objects"]
        actions = batch["actions"]
        images = batch.get("images", None)
        action_text_inputs = batch.get("action_text", None)

        # the objects vector containst four dimensions (pre_0, pre_1, post_0, post_1)
        objects_labels_pre = objects[:, [0, 1], :]
        # here we select the labels for both objects post action
        objects_labels_post = objects[:, [2, 3], :]

        # encode objects and actions, apply action, and predict resulting object attributes
        image_outputs = None
        if images is not None:
            h_o_post_ouput, h_o_pre_ouput, image_outputs = self(
                objects,
                actions,
                action_text_inputs=action_text_inputs,
                image_inputs=images,
                training=False,
            )
        else:
            h_o_post_ouput, h_o_pre_ouput = self(
                objects, actions, action_text_inputs=action_text_inputs
            )

        (
            avg_loss_post,
            predictions,
        ) = self.calculate_object_attribute_loss_and_predictions(
            h_o_post_ouput, objects_labels_post
        )
        avg_loss_pre, _ = self.calculate_object_attribute_loss_and_predictions(
            h_o_pre_ouput, objects_labels_pre
        )
        avg_loss = avg_loss_post + avg_loss_pre

        self.log(f"{split}/loss", avg_loss, batch_size=len(batch["objects"]))
        self.log(f"{split}/loss_post", avg_loss_post, batch_size=len(batch["objects"]))
        self.log(f"{split}/loss_pre", avg_loss_pre, batch_size=len(batch["objects"]))

        results: Dict[str, torch.Tensor] = {
            "predictions": predictions.cpu(),
            "objects_labels_pre": objects_labels_pre.cpu(),
            "objects_labels_post": objects_labels_post.cpu(),
            "actions": actions.cpu(),
        }

        if batch_idx == 0 and images is not None and image_outputs is not None:
            results["images"] = images.cpu()
            results["image_bboxes"] = image_outputs["bboxes"].cpu()
            results["image_bbox_scores"] = image_outputs["bbox_scores"].cpu()

        return results

    def calculate_epoch_end_statistics(self, step_outputs, split="val") -> None:

        """
        Calculate epoch statistics (accuracy) over entire validation set
        Multiple types of accuracy are calculated:
            - average accuracy over all objects and attributes
            - average accuracy over changing attributes only
            - average accuracy per attribute
            - average accuracy per action
        For all accuracies we exclude none objects from the calculation
        """

        outputs = defaultdict(list)
        for step_output in step_outputs:
            for output_name, step_output in step_output.items():
                outputs[output_name] += step_output

        for output_name in outputs.keys():
            if "image" in output_name:
                # only care about a single batch of images
                outputs[output_name] = step_outputs[0][output_name]
            else:
                # concat batches into tensors of shape (num_batches*num_objects, num_attributes)
                outputs[output_name] = torch.concat(
                    outputs[output_name], dim=0
                ).reshape(-1, outputs[output_name][0].shape[-1])

        # ignore None objects - ignore where label of object is none_object_index
        selection_mask = outputs["objects_labels_pre"][:, 0] != self.none_object_index

        # accuracy when every attribute is guessed correctly
        average_accuracy = calculate_average_accuracy(
            outputs["predictions"], outputs["objects_labels_post"], selection_mask
        )
        self.log(f"{split}/accuracy/average_accuracy", average_accuracy, prog_bar=True)

        # log accuracy per attribute
        log_average_attribute_accuracy(
            self.log,
            outputs["predictions"],
            outputs["objects_labels_post"],
            selection_mask,
            split=split,
        )

        # log accuracy of attributes that have changed
        diff_accuracy = calculate_diff_accuracy(
            outputs["predictions"],
            outputs["objects_labels_pre"],
            outputs["objects_labels_post"],
        )
        self.log(f"{split}/accuracy/diff_accuracy", diff_accuracy)

        # log accuracy of actions
        log_action_accuracies(
            self.log,
            self.action_idx_to_name,
            outputs["actions"],
            outputs["predictions"],
            outputs["objects_labels_post"],
            selection_mask,
            split=split,
        )

        # log error rate of attributes
        log_attribute_level_error_rates(
            self.logger.experiment.log,
            outputs["predictions"],
            outputs["objects_labels_post"],
            selection_mask,
            split=split,
        )

        # log confusion matrices for attributes
        log_confusion_matrices(
            self.logger.experiment.log,
            self.object_attributes_idx_to_mapper,
            outputs["predictions"],
            outputs["objects_labels_post"],
            selection_mask,
            split=split,
        )

        if "images" in outputs and not self.plotted_images:
            images_to_log, boxes_to_log, captions_to_log = plot_images(
                outputs,
                self.action_idx_to_name,
                self.object_attributes_idx_to_mapper[0],
            )
            self.logger.log_image(
                f"images",
                images_to_log,
                boxes=boxes_to_log,
                caption=captions_to_log,
            )

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        return self.process_inference_batch(batch, batch_idx, split="val")

    def validation_epoch_end(self, validation_step_outputs) -> None:
        self.calculate_epoch_end_statistics(validation_step_outputs, split="val")

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        return self.process_inference_batch(batch, batch_idx, split="test")

    def test_epoch_end(self, test_step_outputs) -> None:
        self.plotted_images = False
        self.calculate_epoch_end_statistics(test_step_outputs, split="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
