import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from einops import rearrange, reduce, repeat
from transformers import DetrForObjectDetection


class PigletImageEncoder(pl.LightningModule):
    def __init__(
        self,
        hidden_size=256,
        image_model_name="detr",
        width=640,
        height=384,
        K=1000,
        num_objects=2,
    ):
        super().__init__()
        # load backbone model
        self.image_model_name = image_model_name

        # note all the images are the same width and height so
        # for simplicity we save these dimensions for rescaling purposes
        self.width = width
        self.heigth = height

        # K is a hyperparameter that controls the impact of the diff average pixel wise present in a bounding box
        # The lower K is the less the model will focus on the bounding boxes which contain the
        # average pixel wise difference between the pre-action and post-action images and
        # instead will focus more on ones it has confidently labeled
        self.K = K
        self.num_objects = num_objects

        if image_model_name == "detr":
            self.image_model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50", cache_dir=f"output/vision_model/detr"
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

        # change last layer
        self.fc = nn.Linear(self.image_model.config.d_model, hidden_size)

    @staticmethod
    def bboxes_to_mask(bboxes: torch.Tensor, height=384, width=640) -> torch.Tensor:
        """
        Args:
            bboxes: [batch_size, N, 4]
            height: image height
            width: image width
        Returns:
            masks: [batch_size, N, H, W]
        """
        batch_size, N, _ = bboxes.shape

        x = torch.arange(0, height, dtype=torch.float)
        y = torch.arange(0, width, dtype=torch.float)
        x, y = torch.meshgrid(x, y)

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size*2, C, W, H]
        Returns:
            h_i: [batch_size*2, hidden_size]
        """
        outputs = self.image_model(pixel_values=images)
        # softmax over labels to get probabilities per class for each bounding box
        probas = outputs.logits.softmax(-1)[:, :, :-1]

        # convert bounding boxes from cx cy w h to x y x y
        bboxes = (
            torchvision.ops.box_convert(outputs.pred_boxes, "cxcywh", "xyxy")
            * self.img_dims
        ).int()

        # convert bboxes to img masks
        masks = self.bboxes_to_mask(bboxes)

        # calculate pixel wise difference between pre-action and post action image
        diff_images = (
            reduce(images, "(b i) c h w -> b i h w", "sum", i=2, c=3).diff(dim=1).abs()
        )
        diff_images = repeat(diff_images, "b i h w -> (b i 2) 1 h w", i=1)

        # calculate average pixel wise difference between pre-action and post action image for each bounding box
        bbox_scores = reduce(masks * diff_images, "b n h w -> b n", "mean")

        # filter out low confidence bounding boxes based on threshold
        scores = (bbox_scores * self.K).softmax(-1) + probas.max(-1).values
        indices = torch.topk(scores, k=self.num_objects)[1]
        # select hidden states for the top k bounding boxes
        h_i = outputs.last_hidden_state[torch.arange(32).unsqueeze(1), indices, :]

        # map to lower dimension
        h_i = self.fc(h_i)

        return h_i
