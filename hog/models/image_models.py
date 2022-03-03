from typing import Dict

import torch
import torch.nn as nn
import torchvision
from einops import rearrange, repeat
from torchtyping import TensorType
from transformers import DetrForObjectDetection


class PigletImageEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        image_model_name="detr",
        output_dir_path="output",
        width=640,
        height=384,
    ):
        super().__init__()
        # load backbone model
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
            "img_dims", torch.tensor([width, height, width, height], dtype=torch.int)
        )

        # freeze image model
        for param in self.image_model.parameters():
            param.requires_grad = False

        # encode conditional action and object names
        self.conditional_fc = nn.Linear(
            hidden_size * 2, self.image_model.config.d_model
        )
        # map to smaller dimensionality
        self.fc = nn.Linear(self.image_model.config.d_model, hidden_size)

    @staticmethod
    def bboxes_to_mask(
        bboxes: TensorType["batch_size", "N", 4], height=384, width=640
    ) -> TensorType["batch_size", "N", "H", "W"]:
        """
        Args:
            bboxes: [batch_size, N, 4] in [x_min y_min x_max y_max] format
            height: image height
            width: image width
        Returns:
            masks: [batch_size, N, H, W]
        """
        batch_size, N, _ = bboxes.shape

        x = torch.arange(0, height, dtype=torch.float, device=bboxes.device)
        y = torch.arange(0, width, dtype=torch.float, device=bboxes.device)
        y, x = torch.meshgrid(
            x, y, indexing="ij"
        )  # this looks like a mistake but this is not a mistake

        y = repeat(y, "h w -> h w b n", n=N, b=batch_size)
        x = repeat(x, "h w -> h w b n", n=N, b=batch_size)

        masks = (
            (bboxes[:, :, 0] <= x)
            & (x <= bboxes[:, :, 2])
            & (bboxes[:, :, 1] <= y)
            & (y <= bboxes[:, :, 3])
        )
        masks = rearrange(masks, "h w b n -> b n h w")
        return masks

    def forward(
        self,
        images: TensorType["batch_images", "channel", "height", "width"],
        conditional_vector: TensorType["batch", "num_objects", "hidden_size"],
        training=True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [batch_size*2, C, W, H]
            training: (default: True), if False we also return the bboxes, indices, and probas to plot
        Returns:
            h_i_o: [batch_size, 4, hidden_size]
            bboxes: [batch_size*2, N, 4]
            bbox_scores: [batch_size*2, 2, N]
        """
        outputs = self.image_model(pixel_values=images)

        # convert bounding boxes from cx cy w h to x y x y
        bboxes = torchvision.ops.box_convert(outputs.pred_boxes, "cxcywh", "xyxy")
        bboxes = (bboxes * self.img_dims).int()

        h_c = self.conditional_fc(conditional_vector)
        h_c = rearrange(h_c, "b (i o) h -> b i o 1 h", i=2, o=2)
        h_i = rearrange(outputs.last_hidden_state, "(b i) n h -> b i 1 n h", i=2, n=100)

        attention_scores = (
            (h_c * h_i).sum(-1).softmax(-1).unsqueeze(-1)
        )  # [b, i, o, n, 1]
        h_i_o = (h_i * attention_scores).sum(3)  # [b, i, o, h]

        # map to lower dimension
        h_i_o = self.fc(h_i_o)
        h_i_o = rearrange(h_i_o, "b i o h -> b (i o) h ", i=2, o=2)

        output = {"h_i_o": h_i_o}
        if training:
            return output

        output["bboxes"] = bboxes
        output["bbox_scores"] = rearrange(
            attention_scores, "b i o n 1 -> (b i) o n", i=2, o=2
        )
        return output
