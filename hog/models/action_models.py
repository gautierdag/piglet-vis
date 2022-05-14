from collections import OrderedDict
from typing import Union

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
        label_name_embeddings=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.label_name_embeddings = label_name_embeddings

        # Action Encoder Model
        if not self.label_name_embeddings:
            self.action_embedding_layer = nn.Embedding(
                action_embedding_size, hidden_size
            )

        action_encoder_layers = OrderedDict()
        for l in range(num_layers):
            action_encoder_layers[f"dropout_{l}"] = nn.Dropout(dropout)
            action_encoder_layers[f"linear_{l}"] = nn.Linear(hidden_size, hidden_size)
            action_encoder_layers[f"activation_{l}"] = nn.ReLU()
        self.action_encoder = nn.Sequential(action_encoder_layers)

    def forward(
        self,
        action: Union[
            TensorType["batch_size", 1], TensorType["batch_size", "hidden_size"]
        ],
        action_args_embeddings: TensorType["batch_size", "hidden_size"],
    ) -> TensorType["batch_size", "hidden_size"]:

        # if we are using the true symbolic representation of the action
        if not self.label_name_embeddings:
            action = self.action_embedding_layer(action)

        # combine with object representation of arguments
        h_a = self.action_encoder(action + action_args_embeddings)
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


class PigletActionEncoder(nn.Module):
    def __init__(
        self,
        pretrain=False,
        hidden_size=256,
        bert_model_name="roberta-base",
        output_dir_path="output",
        num_layers=3,
        dropout=0.1,
        action_embedding_size=10,
        label_name_embeddings=False,
    ):
        super().__init__()
        self.pretrain = pretrain
        if pretrain:
            self.action_encoder = PigletSymbolicActionEncoder(
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                action_embedding_size=action_embedding_size,
                label_name_embeddings=label_name_embeddings,
            )
        else:
            self.action_encoder = PigletAnnotatedActionEncoder(
                hidden_size=hidden_size,
                bert_model_name=bert_model_name,
                output_dir_path=output_dir_path,
            )

    def forward(
        self,
        actions=None,
        action_args_embeddings=None,
        action_text=None,
    ):
        if self.pretrain:
            h_a = self.action_encoder(actions, action_args_embeddings)
        else:
            h_a = self.action_encoder(
                action_text["input_ids"], action_text["attention_mask"]
            )
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
