from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange
from torchtyping import TensorType
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
            action_encoder_layers[f"activation_{l}"] = nn.ReLU()
        self.action_encoder = nn.Sequential(action_encoder_layers)

    def forward(
        self,
        action: TensorType["batch_size", 1],
        action_args_embeddings: TensorType["batch_size", "hidden_size"],
    ) -> TensorType["batch_size", "hidden_size"]:
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
        output_dir_path="output",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bert_model = AutoModel.from_pretrained(
            bert_model_name,
            cache_dir=f"{output_dir_path}/bert-models/{bert_model_name}",
            add_pooling_layer=False,
        )
        self.output_layer = nn.Linear(self.bert_model.config.hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(
        self,
        action_text_input_ids: TensorType["batch_size", "max_action_length"],
        action_text_attention_mask: TensorType["batch_size", "max_action_length"],
    ) -> TensorType["batch_size", "hidden_size"]:
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers

        # Action Apply Model
        self.fuse_action_object = nn.Linear(hidden_size * 3, hidden_size)
        action_apply_layers = OrderedDict()
        for l in range(num_layers):
            action_apply_layers[f"dropout_{l}"] = nn.Dropout(dropout)
            action_apply_layers[f"linear_{l}"] = nn.Linear(hidden_size, hidden_size)
            action_apply_layers[f"activation_{l}"] = nn.ReLU()
        self.action_apply_layers = nn.Sequential(action_apply_layers)

    def forward(
        self,
        h_o: TensorType["batch_size", "num_objects", "hidden_size"],
        h_a: TensorType["batch_size", "hidden_size"],
    ) -> TensorType["batch_size", "hidden_size"]:

        # concat h_o_0 and h_o_1
        h_o = rearrange(h_o, "b o h -> b (o h)")

        # concat the two encoded objects with encoded action
        h_o_a = torch.concat((h_o, h_a), dim=1)

        # apply action to objects
        h_o_a = self.fuse_action_object(h_o_a)
        h_o_a = self.action_apply_layers(h_o_a)
        return h_o_a
