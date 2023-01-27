import pickle

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchtyping import TensorType

from .mappings import get_objects_mapper


class PigletObjectEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        num_attributes=38,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_attributes = num_attributes

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.object_encoder_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.activation = nn.ReLU()

    def forward(
        self,
        object_embeddings: TensorType[
            "batch_size", "num_objects", "num_attributes", "hidden_size"
        ],
    ) -> TensorType["batch_size", "num_objects", "hidden_size"]:
        assert self.num_attributes == object_embeddings.shape[2]
        num_objects = object_embeddings.shape[1]

        # reshape object_embeddings to [batch_size*2, num_attributes, hidden_size]
        object_embeddings = rearrange(
            object_embeddings,
            "b o a h -> (b o) a h",
            a=self.num_attributes,
            h=self.hidden_size,
        )

        # process representation of objects
        h_o = self.object_encoder_transformer(object_embeddings)

        # we "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        h_o = self.activation(h_o[:, 0])

        h_o = rearrange(
            h_o,
            "(b o) h -> b o h",
            o=num_objects,
            h=self.hidden_size,
        )

        return h_o


class PigletObjectDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        object_embedding_size=329,
        num_attributes=38,
        none_object_index=102,
        use_full=False,
        data_dir_path="data",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.object_embedding_size = object_embedding_size
        self.num_attributes = num_attributes
        self.none_object_index = none_object_index  # mask object tensor with zeroes if object = none_object_index

        # mask for embedding layer output -> based on position
        # mask probability to 0 for other attributes when looking at a specific attribute
        # This is needed because we are using a single embedding layer for all object attributes
        self.reverse_object_mapper = get_objects_mapper(
            data_dir_path, use_full=use_full
        )
        indexes = torch.tensor(
            [
                (pos, index)
                for pos, index_mapper in self.reverse_object_mapper.items()
                for index, _ in index_mapper.items()
            ]
        )
        mask_embedding_layer = torch.zeros((38, object_embedding_size))
        mask_embedding_layer[indexes[:, 0], indexes[:, 1]] = 1
        self.register_buffer("mask_embedding_layer", mask_embedding_layer)

        self.activation = nn.ReLU()

        self.object_h_to_attributes = nn.Sequential(
            nn.Linear(hidden_size, self.num_attributes * hidden_size),
            Rearrange(
                "b o (a h) -> (b o) a h", a=self.num_attributes, h=hidden_size, o=2
            ),
            self.activation,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.object_decoder_transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.object_decoder_output_layer = nn.Linear(hidden_size, object_embedding_size)

    def forward(
        self,
        h_o_a: TensorType["batch_size", "hidden_size"],
        h_o: TensorType["batch_size", 2, "hidden_size"],
    ) -> TensorType["batch_object_size", "num_attributes", "embedding_dim"]:

        # extract an object embedding for each object
        object_embeddings = self.object_h_to_attributes(h_o)

        # expand the fused action/object_pre representation to apply to both objects
        h_o_a = repeat(h_o_a, "b h -> (b 2) 1 h")

        # use transformer decoder and pass object_embeddings as src vector and h_o_a as the memory vector
        h_o_post = self.object_decoder_transformer(object_embeddings, h_o_a)
        h_o_post = self.activation(h_o_post)

        # map sequence to embedding dimension
        h_o_post = self.object_decoder_output_layer(h_o_post)

        return h_o_post

    def mask_output_probabilities(
        self,
        h_o_post: TensorType["batch_object_size", "num_attributes", "embedding_dim"],
    ) -> TensorType["batch_object_size", "num_attributes", "embedding_dim"]:
        """
        Since we use a single embedding matrix for all attributes we need to mask the output probabilities
        of the embedding layer to -inf for all unnattainable attributes at given position.
        """

        # mask the embedding layer output to only allow predictions for valid attributes at each position
        h_o_post = torch.where(
            self.mask_embedding_layer == 1,
            h_o_post,
            torch.tensor(float("-inf"), device=self.mask_embedding_layer.device),
        )
        return h_o_post
