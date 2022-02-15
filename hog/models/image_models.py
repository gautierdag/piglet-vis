import torch
import torch.nn as nn
import torchvision
from einops import rearrange


class PigletImageEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        backbone="resnet50",
    ):
        super().__init__()
        # load backbone model
        self.backbone_model = getattr(torchvision.models, backbone)(pretrained=True)
        # freeze backbone model
        for param in self.backbone_model.parameters():
            param.requires_grad = False

        # change last layer
        self.backbone_model.fc = nn.Linear(
            self.backbone_model.fc.in_features, hidden_size
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 2, C, W, H]
        Returns:
            h_i: [batch_size*2, hidden_size]
        """
        # reshape images to [batch_size*2, C, W, H]
        image_inputs = rearrange(images, "b n c w h -> (b n) c w h", c=3, n=2)

        return self.backbone_model(image_inputs)
