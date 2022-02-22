from collections import defaultdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from dataset import denormalize_image
from einops import rearrange
from PIL import Image

from .action_models import (
    PigletActionApplyModel,
    PigletAnnotatedActionEncoder,
    PigletSymbolicActionEncoder,
)
from .image_models import PigletImageEncoder
from .mappings import OBJECT_ATTRIBUTES, get_actions_mapper, get_objects_mapper
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
        self.pretrain = pretrain
        self.encode_images = encode_images
        self.plotted_images = False

        assert len(OBJECT_ATTRIBUTES) == self.num_attributes

        # Image encoder
        if self.encode_images:
            self.image_encoder = PigletImageEncoder(
                hidden_size=hidden_size, output_dir_path=output_dir_path
            )

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
            data_dir_path=data_dir_path,
        )

        self.action_idx_to_name = get_actions_mapper(data_dir_path)
        self.object_idx_to_name = get_objects_mapper(data_dir_path)

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

        batch_size = object_inputs.shape[0]

        # embed object vector
        h_o, object_embeddings = self.object_encoder(object_inputs)

        if self.encode_images:
            image_model_outputs = self.image_encoder(image_inputs, training=training)
            h_o = image_model_outputs["h_i_pre_action"]

        if self.pretrain:
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
        if self.encode_images and not training:
            return h_o_post, image_model_outputs

        return h_o_post

    def calculate_avg_object_loss(
        self, objects: torch.Tensor, labels: torch.Tensor, eps=1e-12
    ) -> torch.Tensor:
        """
        Args:
            objects: float tensor of dims [batch_size, num_attributes, object_embedding_size]
            labels: long tensor of dims [batch_size, num_attributes]
        """
        loss = F.cross_entropy(
            objects.reshape(-1, self.object_embedding_size),
            labels.flatten(),
            reduction="none",
        )
        # first column of the object vector is the indexed (name) of the object
        # if index == self.none_object_index then the object is not present
        none_mask = torch.ones_like(labels)
        none_mask *= (labels[:, 0] != self.none_object_index).unsqueeze(-1)
        avg_loss = (loss * none_mask.flatten()).sum() / (none_mask.sum() + eps)
        return avg_loss

    def calculate_object_attribute_loss_and_predictions(
        self, h_o_post, objects_labels_post
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # mask the embedding layer output to only allow predictions for valid attributes at each position
        h_o_post = self.object_decoder.mask_output_probabilities(h_o_post)

        # select the two objects of each example separately
        h_o_post = rearrange(h_o_post, "(b n) a o -> b n a o", n=2)
        objects_0 = h_o_post[:, 0, :, :]
        objects_1 = h_o_post[:, 1, :, :]
        objects_0_labels = objects_labels_post[:, 0, :]
        objects_1_labels = objects_labels_post[:, 1, :]

        # calculate loss for each object (since it is a set we evaluate over both possibilities object 0 -> object label 0)
        loss_0_1 = (
            self.calculate_avg_object_loss(objects_0, objects_0_labels)
            + self.calculate_avg_object_loss(objects_1, objects_1_labels)
        ) / 2.0
        loss_1_0 = (
            self.calculate_avg_object_loss(objects_1, objects_0_labels)
            + self.calculate_avg_object_loss(objects_0, objects_1_labels)
        ) / 2.0

        if loss_0_1 < loss_1_0:
            avg_loss = loss_0_1
            # get top 1 prediction for each object attribute
            predictions = h_o_post.argmax(-1)
        else:
            avg_loss = loss_1_0
            # predictions should be reversed for each object
            predictions = torch.stack([objects_1, objects_0], dim=1).argmax(-1)

        predictions = rearrange(predictions, "b n p -> (b n) p", n=2)

        return avg_loss, predictions

    def training_step(self, batch, batch_idx) -> torch.Tensor:
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
        self.log("train/loss", avg_loss, batch_size=len(batch["objects"]))
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
        if images is not None:
            h_o_post_ouput, image_outputs = self(
                objects,
                actions,
                action_text_inputs=action_text_inputs,
                image_inputs=images,
                training=False,
            )
        else:
            h_o_post_ouput = self(
                objects, actions, action_text_inputs=action_text_inputs
            )

        avg_loss, predictions = self.calculate_object_attribute_loss_and_predictions(
            h_o_post_ouput, objects_labels_post
        )
        self.log(f"{split}/loss", avg_loss, batch_size=len(batch["objects"]))
        results: Dict[str, torch.Tensor] = {
            "predictions": predictions.cpu(),
            "objects_labels_pre": objects_labels_pre.cpu(),
            "objects_labels_post": objects_labels_post.cpu(),
            "actions": actions.cpu(),
        }

        if batch_idx == 0 and images is not None and image_outputs is not None:
            results["images"] = images.cpu()
            results["image_probas"] = image_outputs["probas"].cpu()
            results["image_bboxes"] = image_outputs["bboxes"].cpu()
            results["image_indices"] = image_outputs["indices"].cpu()
            results["image_diffs"] = image_outputs["diffs"].cpu()
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
        ignore_mask = outputs["objects_labels_pre"][:, 0] != self.none_object_index

        # accuracy when every attribute is guessed correctly
        num_objects = ignore_mask.shape[0]
        average_accuracy = (
            (outputs["predictions"] == outputs["objects_labels_post"]).sum(1)[
                ignore_mask
            ]
            == self.num_attributes
        ).sum() / num_objects
        self.log(f"{split}/accuracy/average_accuracy", average_accuracy, prog_bar=True)

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
        for action_index, action_name in self.action_idx_to_name.items():
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
                action_accuracy = torch.tensor(0, dtype=torch.float)

            self.log(
                f"{split}/accuracy/action_level/{action_name}_accuracy", action_accuracy
            )

            if "images" in outputs and not self.plotted_images:
                self.plot_images(outputs)
                self.plotted_images = True

    def plot_images(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        images = outputs["images"]
        bboxes = outputs["image_bboxes"]
        indices = outputs["image_indices"]
        probas = outputs["image_probas"]
        diff = outputs["image_diffs"]
        scores = outputs["image_bbox_scores"]

        # Get the color map by name for mapping diff images
        cm = plt.get_cmap("hot", lut=8)
        for img_idx in range(0, images.shape[0], 2):

            # format bounding boxes for before image
            image_before = denormalize_image(images[img_idx]).permute(1, 2, 0).numpy()
            image_before_boxes = {"predictions": {"box_data": []}}
            for bbox_idx, ((x_min, y_min, x_max, y_max), proba) in enumerate(
                zip(bboxes[img_idx], probas[img_idx])
            ):
                box_id = proba.argmax().item()
                box = {
                    "position": {
                        "minX": x_min.item(),
                        "maxX": x_max.item(),
                        "minY": y_min.item(),
                        "maxY": y_max.item(),
                    },
                    "domain": "pixel",
                    "class_id": box_id,
                    "box_caption": self.image_encoder.image_model.config.id2label[
                        box_id
                    ],
                    "scores": {
                        "chosen": 1.0 if bbox_idx in indices[img_idx] else 0.0,
                        "confidence": proba.max().item(),
                        "overlap": scores[img_idx][bbox_idx].item(),
                    },
                }
                image_before_boxes["predictions"]["box_data"].append(box)

            # draw image diff using matplotlib for nice 'hot' gradient
            d = diff[img_idx].permute(1, 2, 0).squeeze(-1)  # permute to (W,H,C)
            d = (d - d.min()) / (d.max() - d.min())  # normalize
            colored_image = cm(d.numpy())
            # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
            # But we want to convert to RGB in uint8 and save it:
            image_diff = Image.fromarray(
                (colored_image[:, :, :3] * 255).astype(np.uint8)
            )

            # format bounding boxes for after image
            image_after = (
                denormalize_image(images[img_idx + 1]).permute(1, 2, 0).numpy()
            )
            image_after_boxes = {"predictions": {"box_data": []}}
            for bbox_idx, ((x_min, y_min, x_max, y_max), proba) in enumerate(
                zip(bboxes[img_idx + 1], probas[img_idx + 1])
            ):
                box_id = proba.argmax().item()
                box = {
                    "position": {
                        "minX": x_min.item(),
                        "maxX": x_max.item(),
                        "minY": y_min.item(),
                        "maxY": y_max.item(),
                    },
                    "domain": "pixel",
                    "class_id": box_id,
                    "box_caption": self.image_encoder.image_model.config.id2label[
                        box_id
                    ],
                    "scores": {
                        "chosen": 1.0 if bbox_idx in indices[img_idx + 1] else 0.0,
                        "confidence": proba.max().item(),
                        "overlap": scores[img_idx + 1][bbox_idx].item(),
                    },
                }
                image_after_boxes["predictions"]["box_data"].append(box)

            title = self.get_example_title_from_actions_object(
                outputs["actions"], outputs["objects_labels_pre"], img_idx
            )

            self.logger.log_image(
                f"images",
                [image_before, image_diff, image_after],
                boxes=[
                    image_before_boxes,
                    {"predictions": {"box_data": []}},
                    image_after_boxes,
                ],
                caption=["before", title, "after"],
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

    def get_example_title_from_actions_object(
        self, actions, objects, index: int
    ) -> str:
        """
        Actions: [B x 3]
        Object: [B x 2, num_attributes]
        """
        assert 0 <= index < objects.shape[0] - 1

        action_text = self.action_idx_to_name[actions[int(index / 2)][0].item()]
        action_object = self.object_idx_to_name[actions[int(index / 2)][1].item()]
        action_receptacle = self.object_idx_to_name[actions[int(index / 2)][2].item()]
        object_1 = self.object_idx_to_name[objects[index][0].item()]
        object_2 = self.object_idx_to_name[objects[index + 1][0].item()]
        return f"Effects of {action_text}({action_object}, {action_receptacle}) on ({object_1},{object_2})"
