import pickle
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from object_attributes import OBJECT_ATTRIBUTES


class Piglet(pl.LightningModule):
    def __init__(
        self,
        hidden_size=256,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        object_embedding_size=329,
        num_attributes=38,
        action_embedding_size=10,
        none_object_index=102,
        reverse_object_mapping_dir="data",
        tie_weights=True,
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
          tie_weights: if true, the weights of the object decoder are tied to the weights of the object encoder
        """
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.object_embedding_size = object_embedding_size
        self.num_attributes = num_attributes
        self.action_embedding_size = action_embedding_size
        self.none_object_index = none_object_index  # mask object tensor with zeroes if object = none_object_index
        self.tie_weights = tie_weights
        self.activation = nn.Tanh()

        assert len(OBJECT_ATTRIBUTES) == self.num_attributes

        # mask for embedding layer output -> based on position
        # mask probability to 0 for other attributes when looking at a specific attribute
        # This is needed because we are using a single embedding layer for all object attributes
        reverse_object_mapper_path = (
            f"{reverse_object_mapping_dir}/reverse_object_mapping.pkl"
        )
        with open(reverse_object_mapper_path, "rb") as f:
            self.reverse_object_mapper = pickle.load(f)
        indexes = torch.tensor(
            [
                (pos, index)
                for pos, index_mapper in self.reverse_object_mapper.items()
                for index, _ in index_mapper.items()
            ]
        )
        mask_embedding_layer = torch.zeros((38, 329))
        mask_embedding_layer[indexes[:, 0], indexes[:, 1]] = 1
        self.register_buffer("mask_embedding_layer", mask_embedding_layer)

        # Object Encoder Model
        self.object_embedding_layer = nn.Embedding(
            object_embedding_size, hidden_size, padding_idx=none_object_index
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.object_encoder_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Action Encoder Model
        self.action_embedding_layer = nn.Embedding(action_embedding_size, hidden_size)
        action_encoder_layers = OrderedDict()
        for l in range(num_layers):
            action_encoder_layers[f"dropout_{l}"] = nn.Dropout(dropout)
            action_encoder_layers[f"linear_{l}"] = nn.Linear(hidden_size, hidden_size)
            action_encoder_layers[f"activation_{l}"] = nn.Tanh()
        self.action_encoder = nn.Sequential(action_encoder_layers)

        # Action Apply Model
        self.fuse_action_object = nn.Linear(hidden_size * 3, hidden_size)
        action_apply_layers = OrderedDict()
        for l in range(num_layers):
            action_apply_layers[f"dropout_{l}"] = nn.Dropout(dropout)
            action_apply_layers[f"linear_{l}"] = nn.Linear(hidden_size, hidden_size)
            action_apply_layers[f"activation_{l}"] = nn.Tanh()
        self.action_apply_layers = nn.Sequential(action_encoder_layers)

        # Object Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.object_decoder_transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.object_decoder_output_layer = nn.Linear(hidden_size, object_embedding_size)

        if tie_weights:
            self.object_decoder_output_layer.weight = self.object_embedding_layer.weight

    def forward(self, object_inputs, action_inputs):
        # always four "objects" per example 2 before and 2 after
        assert object_inputs.shape[1] == 4
        # consitent number of object attributes
        assert object_inputs.shape[2] == self.num_attributes

        batch_size = object_inputs.shape[0]

        # select only the pre action objects
        object_inputs = object_inputs[:, [0, 1], :]

        # embed object vector
        object_embeddings = self.object_embedding_layer(object_inputs).reshape(
            -1, self.num_attributes, self.hidden_size
        )

        # process representation of objects
        h_o = self.object_encoder_transformer(object_embeddings)

        # we "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        h_o = self.activation(h_o[:, 0])

        # embed the action vector
        action_embedding = self.action_embedding_layer(action_inputs[:, 0])

        # sum the embedding of object targeted and its receptacle
        action_args_embeddings = self.object_embedding_layer(action_inputs[:, 1:]).sum(
            1
        )
        h_a = self.action_encoder(action_embedding + action_args_embeddings)

        # concat the two encoded objects with encoded action
        h_o_a = torch.concat((h_o.reshape(batch_size, -1), h_a), dim=1)

        # apply action to objects
        h_o_a = self.fuse_action_object(h_o_a)
        h_o_a = self.action_apply_layers(h_o_a)
        h_o_a = self.activation(h_o_a)

        # expand the fused action/object_pre representation to apply to both objects
        h_o_a = torch.repeat_interleave(h_o_a, (2), dim=0)

        # use transformer decoder and pass object_embeddings as src vector and h_o_a as the memory vector
        h_o_post = self.object_decoder_transformer(
            object_embeddings, h_o_a.reshape(batch_size * 2, 1, -1)
        )
        h_o_post = self.activation(h_o_post)

        # map sequence to embedding dimension
        h_o_post = self.object_decoder_output_layer(h_o_post)

        # returns predicted sequence of object attributes
        # torch.Size([batch_size*2, num_attributes, object_embedding_size])
        return h_o_post

    def calculate_object_attribute_loss_and_predictions(
        self, h_o_post, objects_labels_post
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # mask the embedding layer output to only allow predictions for valid attributes at each position
        h_o_post = torch.where(
            self.mask_embedding_layer == 1,
            h_o_post,
            torch.tensor(float("-inf")),
        )

        loss = F.cross_entropy(
            h_o_post.reshape(-1, self.object_embedding_size),
            objects_labels_post.flatten(),
            reduction="none",
        )

        none_mask = torch.ones_like(objects_labels_post)
        # first column of the object vector is the indexed (name) of the object
        # if index == self.none_object_index then the object is not present
        none_mask *= (objects_labels_post[:, :, 0] != self.none_object_index).unsqueeze(
            2
        )

        # average over actual objects (not none objects)
        avg_loss = (loss * none_mask.flatten()).sum() / (none_mask.sum() + 1e-12)

        # get top 1 prediction for each object attribute
        predictions = h_o_post.argmax(2)

        return avg_loss, predictions

    def training_step(self, batch, batch_idx) -> torch.tensor:
        objects = batch["objects"]
        actions = batch["actions"]

        # the objects vector containst four dimensions (pre_0, pre_1, post_0, post_1)
        # here we select the labels for both objects post action
        objects_labels_post = objects[:, [2, 3], :]

        # encode objects and actions, apply action, and predict resulting object attributes
        h_o_post_ouput = self(objects, actions)

        avg_loss, _ = self.calculate_object_attribute_loss_and_predictions(
            h_o_post_ouput, objects_labels_post
        )
        self.log("train/loss", avg_loss)
        return avg_loss

    def validation_step(self, batch, batch_idx) -> None:
        objects = batch["objects"]
        actions = batch["actions"]

        # the objects vector containst four dimensions (pre_0, pre_1, post_0, post_1)
        objects_labels_pre = objects[:, [0, 1], :]

        # here we select the labels for both objects post action
        objects_labels_post = objects[:, [2, 3], :]

        # encode objects and actions, apply action, and predict resulting object attributes
        h_o_post_ouput = self(objects, actions)

        avg_loss, predictions = self.calculate_object_attribute_loss_and_predictions(
            h_o_post_ouput, objects_labels_post
        )
        self.log("val/loss", avg_loss)
        return (predictions.cpu(), objects_labels_pre.cpu(), objects_labels_post.cpu())

    def validation_epoch_end(self, validation_step_outputs) -> None:
        object_attribute_predictions_post = []
        object_attribute_labels_pre = []
        object_attribute_labels_post = []
        for (preds, labels_pre, labels_pro) in validation_step_outputs:
            object_attribute_predictions_post.append(preds)
            object_attribute_labels_pre.append(labels_pre)
            object_attribute_labels_post.append(labels_pro)

        # concat all predictions and labels into 2D tensor
        object_attribute_predictions_post = torch.stack(
            object_attribute_predictions_post
        ).reshape(-1, self.num_attributes)
        object_attribute_labels_pre = torch.stack(object_attribute_labels_pre).reshape(
            -1, self.num_attributes
        )
        object_attribute_labels_post = torch.stack(
            object_attribute_labels_post
        ).reshape(-1, self.num_attributes)

        # ignore where label of object is none_object_index
        object_attribute_predictions_post = object_attribute_predictions_post[
            object_attribute_labels_pre[:, 0] != self.none_object_index
        ]
        object_attribute_labels_pre = object_attribute_labels_pre[
            object_attribute_labels_pre[:, 0] != self.none_object_index
        ]
        object_attribute_labels_post = object_attribute_labels_post[
            object_attribute_labels_post[:, 0] != self.none_object_index
        ]

        # calculate accuracy per attribute
        average_accuracy = (
            object_attribute_predictions_post == object_attribute_labels_post
        ).sum(0) / object_attribute_predictions_post.shape[0]
        # log accuracy per attribute
        for acc, object_attribute_name in zip(average_accuracy, OBJECT_ATTRIBUTES):
            self.log(f"val/accuracy/attribute_level/{object_attribute_name}", acc)
        # log average overall accuracy
        self.log(f"val/accuracy/average_accuracy", average_accuracy.mean())
        # log accuracy of attributes that have changed
        diff = object_attribute_labels_post != object_attribute_labels_pre
        diff_accuracy = 0
        if len(diff) > 0:
            diff_accuracy = (
                object_attribute_predictions_post[diff]
                == object_attribute_labels_post[diff]
            ).sum() / len(diff)
        self.log(f"val/accuracy/diff_accuracy", diff_accuracy)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.001)
