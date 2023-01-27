import torch
import torch.nn.functional as F
from torchtyping import TensorType


def calculate_avg_object_loss(
    h_out: TensorType["batch_objects", "num_attributes", "object_embedding_size"],
    labels: TensorType["batch", "num_objects", "num_attributes"],
    selection_mask: TensorType["batch", "num_objects"],
) -> torch.Tensor:
    """
    Calculate the average cross entropy loss for each object attribute in the selection mask
    """
    object_embedding_size = h_out.shape[2]
    loss = F.cross_entropy(
        h_out[selection_mask.flatten()].reshape(-1, object_embedding_size),
        labels[selection_mask].flatten(),
        reduction="none",
    )
    return loss.mean()
