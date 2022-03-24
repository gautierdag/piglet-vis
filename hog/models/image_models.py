from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from torchtyping import TensorType
from transformers import DetrFeatureExtractor, DetrForObjectDetection


def get_image_feature_extractor(
    output_dir_path: str,
    resize=False,
    normalize=True,
    vision_model="detr",
) -> DetrFeatureExtractor:
    if vision_model == "detr":
        image_feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50",
            cache_dir=f"{output_dir_path}/vision_model/detr",
            do_resize=resize,
            do_normalize=normalize,
        )
        return image_feature_extractor
    raise NotImplemented(f"Image model {self.vision_model} not implemented")


class BoundingBoxImageModel(nn.Module):
    def __init__(
        self,
        image_model_name="detr",
        output_dir_path="output",
        width=640,
        height=384,
    ):
        super().__init__()

        self.image_model_name = image_model_name

        # note all the images are the same width and height so
        # for simplicity we save these dimensions for rescaling purposes
        self.width = width
        self.heigth = height

        if image_model_name == "detr":
            self.image_model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                cache_dir=f"{output_dir_path}/vision_model/detr",
            )
        else:
            raise NotImplementedError(
                "Only facebook/detr-resnet-50 is currently supported"
            )

        self.register_buffer(
            "img_dims",
            torch.tensor([width, height, width, height], dtype=torch.int),
        )
        # freeze image model
        for param in self.image_model.parameters():
            param.requires_grad = False

    def forward(
        self, images: TensorType["batch_images", "channel", "height", "width"]
    ) -> Tuple[
        TensorType["batch_images", "num_boxes", 4],
        TensorType["batch_images", "num_boxes", "hidden"],
    ]:
        outputs = self.image_model(pixel_values=images)

        # convert bounding boxes from cx cy w h to x y x y
        bboxes = torchvision.ops.box_convert(outputs.pred_boxes, "cxcywh", "xyxy")
        bboxes = (bboxes * self.img_dims).int()

        return bboxes, outputs.last_hidden_state


class PigletImageEncoder(nn.Module):
    def __init__(self, hidden_size=256, hidden_input_size=256):
        super().__init__()
        self.conditional_fc = nn.Linear(hidden_size * 2, hidden_input_size)
        # map to smaller dimensionality
        self.fc = nn.Linear(hidden_input_size, hidden_size)

    def forward(
        self,
        images_hidden_states: TensorType[
            "batch", "num_images", "num_boxes", "hidden_input_size"
        ],
        conditional_vector: TensorType["batch", "num_objects", "hidden_size"],
    ) -> Tuple[
        TensorType["batch", "num_objects_images", "hidden_size"],
        TensorType["batch_images", "num_objects", "num_boxes"],
    ]:
        """
        Args:
            images_hidden_states: [batch_size*2, C, W, H]
            training: (default: True), if False we also return the bboxes, indices, and probas to plot
        Returns:
            h_i_o: [batch_size, 4, hidden_size]
            bboxes: [batch_size*2, N, 4]
            bbox_scores: [batch_size*2, 2, N]
        """
        h_c = self.conditional_fc(conditional_vector)
        h_c = rearrange(h_c, "b (i o) h -> b i o 1 h", i=2, o=2)
        h_i = rearrange(images_hidden_states, "b i n h -> b i 1 n h", i=2, n=100)

        attention_scores = (
            (h_c * h_i).sum(-1).softmax(-1).unsqueeze(-1)
        )  # [b, i, o, n, 1]
         
        h_i_o = (h_i * attention_scores).sum(3) # [b, i, o, h]

        # map to lower dimension
        h_i_o = self.fc(h_i_o)
        h_i_o = rearrange(h_i_o, "b i o h -> b (i o) h ", i=2, o=2)

        bbox_scores = rearrange(attention_scores, "b i o n 1 -> b i o n", i=2, o=2)
        return h_i_o, bbox_scores
