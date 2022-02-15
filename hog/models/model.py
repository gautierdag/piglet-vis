from typing import Tuple
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange

from .action_models import (
    PigletAnnotatedActionEncoder,
    PigletSymbolicActionEncoder,
    PigletActionApplyModel,
)
from .object_models import PigletObjectEncoder, PigletObjectDecoder
from .image_models import PigletImageEncoder
from .mappings import OBJECT_ATTRIBUTES, ACTIONS_MAPPER


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
        reverse_object_mapping_dir="data",
        symbolic_action=True,
        bert_model_name="roberta-base",
        bert_model_dir="output/bert-models",
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
          action_embedding_size: max embedding index the action
          none_object_index: index of the none object
          reverse_object_mapping_dir: path to the reverse object mapping
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.object_embedding_size = object_embedding_size
        self.num_attributes = num_attributes
        self.none_object_index = none_object_index  # mask object tensor with zeroes if object = none_object_index
        self.action_embedding_size = action_embedding_size
        self.symbolic_action = symbolic_action
        self.bert_model_name = bert_model_name
        self.bert_model_dir = bert_model_dir
        self.encode_images = encode_images

        assert len(OBJECT_ATTRIBUTES) == self.num_attributes

        # Image encoder
        self.image_encoder = PigletImageEncoder(hidden_size=hidden_size)

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
        if self.symbolic_action:
            self.action_encoder = PigletSymbolicActionEncoder(
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                action_embedding_size=action_embedding_size,
            )
        else:
            self.action_encoder = PigletAnnotatedActionEncoder(
                hidden_size=hidden_size,
                bert_model_name="roberta-base",
                bert_model_dir="output/bert-models",
            )

        # Action Apply Model
        self.apply_action = PigletActionApplyModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            input_fuse_size=hidden_size * 3,
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
            reverse_object_mapping_dir=reverse_object_mapping_dir,
        )

    def forward(
        self, object_inputs, action_inputs, action_text_inputs=None, image_inputs=None
    ):
        # check that image inputs are not passed if no image encoder and vice versa
        assert (image_inputs is None) == (not self.encode_images)

        batch_size = object_inputs.shape[0]

        # embed object vector
        h_o, object_embeddings = self.object_encoder(object_inputs)

        if self.encode_images:
            h_o = self.image_encoder(image_inputs)

        if self.symbolic_action:
            # sum the embedding of object targeted and its receptacle
            action_args_embeddings = self.object_encoder.object_embedding_layer(
                action_inputs[:, 1:]
            ).sum(1)
            h_a = self.action_encoder(action_inputs[:, 0], action_args_embeddings)
        else:
            h_a = self.action_encoder(
                action_text_inputs["input_ids"], action_text_inputs["attention_mask"]
            )

        # apply action to object hidden representations
        h_o_a = self.apply_action(h_o.reshape(batch_size, -1), h_a)

        # use transformer decoder and pass object_embeddings as src vector and h_o_a as the memory vector
        h_o_post = self.object_decoder(h_o_a, object_embeddings)

        # returns predicted sequence of object attributes
        # torch.Size([batch_size*2, num_attributes, object_embedding_size])
        return h_o_post

    def calculate_object_attribute_loss_and_predictions(
        self, h_o_post, objects_labels_post
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # mask the embedding layer output to only allow predictions for valid attributes at each position
        h_o_post = self.object_decoder.mask_output_probabilities(h_o_post)

        loss = F.cross_entropy(
            h_o_post.reshape(-1, self.object_embedding_size),
            objects_labels_post.flatten(),
            reduction="none",
        )

        none_mask = torch.ones_like(objects_labels_post)
        # first column of the object vector is the indexed (name) of the object
        # if index == self.none_object_index then the object is not present
        none_mask *= rearrange(
            objects_labels_post[:, :, 0] != self.none_object_index, "b o -> b o 1"
        )

        # average over actual objects (not none objects)
        avg_loss = (loss * none_mask.flatten()).sum() / (none_mask.sum() + 1e-12)

        # get top 1 prediction for each object attribute
        predictions = h_o_post.argmax(2)

        return avg_loss, predictions

    def training_step(self, batch, batch_idx) -> torch.tensor:
        objects = batch["objects"]
        actions = batch["actions"]
        images = batch.get("images", None)
        action_text_inputs = batch.get("action_text", None)

        # the objects vector contains four dimensions (pre_0, pre_1, post_0, post_1)
        # here we select the labels for both objects post action
        objects_labels_post = objects[:, [2, 3], :]

        # encode objects and actions, apply action, and predict resulting object attributes
        h_o_post_ouput = self(
            objects, actions, action_text_inputs=action_text_inputs, image_inputs=images
        )

        avg_loss, _ = self.calculate_object_attribute_loss_and_predictions(
            h_o_post_ouput, objects_labels_post
        )
        self.log("train/loss", avg_loss)
        return avg_loss

    def process_inference_batch(self, batch, split="val") -> None:
        objects = batch["objects"]
        actions = batch["actions"]
        images = batch.get("images", None)
        action_text_inputs = batch.get("action_text", None)

        # the objects vector containst four dimensions (pre_0, pre_1, post_0, post_1)
        objects_labels_pre = objects[:, [0, 1], :]

        # here we select the labels for both objects post action
        objects_labels_post = objects[:, [2, 3], :]

        # encode objects and actions, apply action, and predict resulting object attributes
        h_o_post_ouput = self(
            objects, actions, action_text_inputs=action_text_inputs, image_inputs=images
        )

        avg_loss, predictions = self.calculate_object_attribute_loss_and_predictions(
            h_o_post_ouput, objects_labels_post
        )
        self.log(f"{split}/loss", avg_loss)
        return (
            predictions.cpu(),
            objects_labels_pre.cpu(),
            objects_labels_post.cpu(),
            actions.cpu(),
        )

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
        output_names = [
            "predictions",
            "objects_labels_pre",
            "objects_labels_post",
            "actions",
        ]
        outputs = defaultdict(list)
        for step_output in step_outputs:
            for output_name, step_output in zip(output_names, step_output):
                outputs[output_name] += step_output

        # concat batches into tensors of shape (num_batches*num_objects, num_attributes)
        for output_name in output_names:
            outputs[output_name] = torch.concat(outputs[output_name], dim=0).reshape(
                -1, outputs[output_name][0].shape[-1]
            )
        # ignore None objects - ignore where label of object is none_object_index
        ignore_mask = outputs["objects_labels_pre"][:, 0] != self.none_object_index

        # accuracy when every attribute is guessed correctly
        num_objects = ignore_mask.shape[0]
        average_accuracy = (
            (outputs["predictions"] == outputs["objects_labels_post"]).sum(1)[
                ignore_mask
            ]
            == self.num_attributes
        ).sum() / num_objects
        self.log(
            f"{split}/accuracy/average_accuracy",
            average_accuracy,
        )

        # calculate accuracy per attribute
        average_attribute_accuracy = (
            outputs["predictions"] == outputs["objects_labels_post"]
        )[ignore_mask].sum(0) / num_objects
        # log accuracy per attribute
        for acc, object_attribute_name in zip(
            average_attribute_accuracy, OBJECT_ATTRIBUTES
        ):
            self.log(f"{split}/accuracy/attribute_level/{object_attribute_name}", acc)
        # log average overall accuracy
        self.log(
            f"{split}/accuracy/average_attribute_accuracy",
            average_attribute_accuracy.mean(),
        )

        # log accuracy of attributes that have changed
        diff = outputs["objects_labels_post"] != outputs["objects_labels_pre"]
        diff_accuracy = 0
        if diff.sum() > 0:
            diff_accuracy = (
                outputs["predictions"][diff] == outputs["objects_labels_post"][diff]
            ).sum() / diff.sum()
        self.log(f"{split}/accuracy/diff_accuracy", diff_accuracy)

        # log accuracy of actions
        num_examples = outputs["actions"].shape[0]
        for action_index, action_name in ACTIONS_MAPPER.items():
            action_mask = outputs["actions"][:, 0] == action_index
            num_examples_with_action = action_mask.sum()
            if num_examples_with_action > 0:
                preds = (
                    outputs["predictions"]
                    .reshape(num_examples, 2, -1)[action_mask]
                    .reshape(num_examples_with_action * 2, -1)
                )
                labels = (
                    outputs["objects_labels_post"]
                    .reshape(num_examples, 2, -1)[action_mask]
                    .reshape(num_examples_with_action * 2, -1)
                )
                ignore_mask = labels[:, 0] != self.none_object_index
                n = ignore_mask.sum()
                action_accuracy = (
                    (preds[ignore_mask] == labels[ignore_mask]).sum(1)
                    == self.num_attributes
                ).sum() / n
            else:
                action_accuracy = torch.tensor(0, dtype=float)

            self.log(
                f"{split}/accuracy/action_level/{action_name}_accuracy", action_accuracy
            )

    def validation_step(self, batch, batch_idx) -> None:
        return self.process_inference_batch(batch, split="val")

    def validation_epoch_end(self, validation_step_outputs) -> None:
        self.calculate_epoch_end_statistics(validation_step_outputs, split="val")

    def test_step(self, batch, batch_idx) -> None:
        return self.process_inference_batch(batch, split="test")

    def test_epoch_end(self, test_step_outputs) -> None:
        self.calculate_epoch_end_statistics(test_step_outputs, split="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
