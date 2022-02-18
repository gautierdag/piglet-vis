from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class PigletSymbolicActionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        num_layers=3,
        dropout=0.1,
        action_embedding_size=10,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Action Encoder Model
        self.action_embedding_layer = nn.Embedding(action_embedding_size, hidden_size)
        action_encoder_layers = OrderedDict()
        for l in range(num_layers):
            action_encoder_layers[f"dropout_{l}"] = nn.Dropout(dropout)
            action_encoder_layers[f"linear_{l}"] = nn.Linear(hidden_size, hidden_size)
            action_encoder_layers[f"activation_{l}"] = nn.Tanh()
        self.action_encoder = nn.Sequential(action_encoder_layers)

    def forward(
        self, action: torch.Tensor, action_args_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            action: [batch_size, 1]
            action_args_embedding: [batch_size, hidden_size]
        Returns:
            h_a: [batch_size, hidden_size]
        """

        # embed the action vector
        action_embedding = self.action_embedding_layer(action)
        # combine with object representation of arguments
        h_a = self.action_encoder(action_embedding + action_args_embeddings)

        return h_a


class PigletAnnotatedActionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        bert_model_name="roberta-base",
        bert_model_dir="output/bert-models",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bert_model = AutoModel.from_pretrained(
            bert_model_name,
            cache_dir=f"{bert_model_dir}/{bert_model_name}",
            add_pooling_layer=False,
        )
        self.output_layer = nn.Linear(self.bert_model.config.hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self,
        action_text_input_ids: torch.Tensor,
        action_text_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            action_text_input_ids: [batch_size, N]
            action_text_attention_mask: [batch_size, N]
        Returns:
            h_a: [batch_size, hidden_size]
        """
        bert_outputs = self.bert_model(
            action_text_input_ids, attention_mask=action_text_attention_mask
        )
        # pass first token (cls token) of each action to output layer
        h_a = self.output_layer(bert_outputs.last_hidden_state[:, 0, :])
        h_a = self.activation(h_a)
        return h_a


class PigletActionApplyModel(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        num_layers=3,
        dropout=0.1,
        input_fuse_size=768,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.input_fuse_size = input_fuse_size

        # Action Apply Model
        self.fuse_action_object = nn.Linear(input_fuse_size, hidden_size)
        action_apply_layers = OrderedDict()
        for l in range(num_layers):
            action_apply_layers[f"dropout_{l}"] = nn.Dropout(dropout)
            action_apply_layers[f"linear_{l}"] = nn.Linear(hidden_size, hidden_size)
            action_apply_layers[f"activation_{l}"] = nn.Tanh()
        self.action_apply_layers = nn.Sequential(action_apply_layers)

    def forward(
        self, h_o: torch.Tensor, h_a: torch.Tensor, h_i: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h_o: [batch_size, hidden_size*2]
            h_a: [batch_size, hidden_size]
            h_i: [batch_size, hidden_size]
        Returns:
            h_o_a: [batch_size, hidden_size]

        """
        # concat the two encoded objects with encoded action
        h_o_a = torch.concat((h_o, h_a), dim=1)

        # apply action to objects
        h_o_a = self.fuse_action_object(h_o_a)
        h_o_a = self.action_apply_layers(h_o_a)
        return h_o_a
