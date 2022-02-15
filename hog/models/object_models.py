from typing import Tuple
import pickle

import torch
import torch.nn as nn
from einops import repeat, rearrange


class PigletObjectEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        object_embedding_size=329,
        num_attributes=38,
        none_object_index=102,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.object_embedding_size = object_embedding_size
        self.num_attributes = num_attributes
        self.none_object_index = none_object_index  # mask object tensor with zeroes if object = none_object_index

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
        self.activation = nn.Tanh()

    def forward(self, object_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            object_inputs: [batch_size, 4, num_attributes]
        Returns:
            h_o: [batch_size*2, hidden_size]
            object_embeddings: [batch_size*2, num_attributes, hidden_size]
        """
        # always four "objects" per example 2 before and 2 after
        assert object_inputs.shape[1] == 4
        # consitent number of object attributes
        assert object_inputs.shape[2] == self.num_attributes

        # select only the pre action objects
        object_inputs = object_inputs[:, [0, 1], :]

        # embed object vector
        object_embeddings = self.object_embedding_layer(object_inputs)

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

        return h_o, object_embeddings


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
        reverse_object_mapping_dir="data",
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

        # Object Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.object_decoder_transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.activation = nn.Tanh()
        self.object_decoder_output_layer = nn.Linear(hidden_size, object_embedding_size)

    def forward(
        self, h_o_a: torch.Tensor, object_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_o_a: [batch_size, hidden_size]
            object_embeddings: [batch_size*2, num_attributes, hidden_size]
        Returns:
            h_a: [batch_size, hidden_size]
        """
        # expand the fused action/object_pre representation to apply to both objects
        h_o_a = repeat(h_o_a, "b h -> (b 2) 1 h")

        # use transformer decoder and pass object_embeddings as src vector and h_o_a as the memory vector
        h_o_post = self.object_decoder_transformer(object_embeddings, h_o_a)
        h_o_post = self.activation(h_o_post)

        # map sequence to embedding dimension
        h_o_post = self.object_decoder_output_layer(h_o_post)

        return h_o_post

    def mask_output_probabilities(self, h_o_post: torch.Tensor) -> torch.Tensor:
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
