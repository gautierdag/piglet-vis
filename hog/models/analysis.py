from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import rearrange, repeat
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from torchtyping import TensorType

from .mappings import OBJECT_ATTRIBUTES


def get_cmap(cm="jet", N=256):
    colors = getattr(plt.cm, cm)(np.linspace(0, 1, N))
    transparent_hot_colors = []
    for c, t in zip(colors, np.linspace(0, 1, N)):
        transparent_hot_colors.append(c)
        transparent_hot_colors[-1][-1] = t
    return mcolors.LinearSegmentedColormap.from_list("hot", transparent_hot_colors, N=N)


def calculate_attention_weights_per_pixel(bboxes, scores):
    attention_weights = torch.zeros(384, 640, 1)
    for box, score in zip(bboxes, scores):
        x_min, y_min, x_max, y_max = box
        attention_weights[y_min:y_max, x_min:x_max] += score
    return attention_weights


def imshow_with_attention_overlay(ax, image, bboxes, scores):
    attention_weights = calculate_attention_weights_per_pixel(bboxes, scores)
    ax.imshow(image)
    ax.imshow(attention_weights, cmap=get_cmap())


def turn_off_axis(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_effect_grid(images, bboxes, scores, objects) -> Image.Image:
    dpi = 100
    sizes = np.shape(images[0])
    fig, axs = plt.subplots(
        2, 2, figsize=((sizes[1] / dpi) * 2, (sizes[0] / dpi) * 2), dpi=dpi
    )
    canvas = FigureCanvas(fig)
    caption_dict = {0: "Before", 1: "After"}
    for i in range(2):
        for o in range(2):
            if i == 0:
                axs[i][o].set_title(f"{objects[o]}")
            if o == 0:
                axs[i][o].set_ylabel(caption_dict[i])
            turn_off_axis(axs[i][o])
            imshow_with_attention_overlay(axs[i][o], images[i], bboxes[i], scores[i][o])

    plt.subplots_adjust(wspace=0, hspace=0)

    canvas.draw()

    image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return Image.fromarray(image_from_plot)


def get_example_title_from_actions_object(
    actions, objects, index: int, action_idx_to_name: dict, object_idx_to_name: dict
) -> str:
    """
    Actions: [B x 3]
    Object: [B x 2, num_attributes]
    """
    assert 0 <= index < objects.shape[0] - 1

    action_text = action_idx_to_name[actions[index, 0].item()]
    action_object = object_idx_to_name[actions[index, 1].item()]
    action_receptacle = object_idx_to_name[actions[index, 2].item()]
    object_1 = object_idx_to_name[objects[index, 0, 0].item()]
    object_2 = object_idx_to_name[objects[index, 1, 0].item()]
    return f"Effects of {action_text}({action_object}, {action_receptacle}) on ({object_1},{object_2})"


def plot_images(
    dataset,
    outputs: Dict[str, torch.Tensor],
    action_idx_to_name: dict,
    object_idx_to_name: dict,
    num_images=16,
) -> Tuple[List[Image.Image], List[str]]:
    scores = outputs["image_bbox_scores"]
    objects = rearrange(outputs["objects_labels_pre"], "(b o) n -> b o n", o=2)

    indices = outputs["indices"][:num_images]
    scores = scores[:num_images]

    images_to_log = []
    captions_to_log = []
    for i in range(len(indices)):
        image_idx = indices[i]
        images, bboxes = dataset.get_images_and_bounding_boxes(image_idx)

        title = get_example_title_from_actions_object(
            outputs["actions"],
            objects,
            i,
            action_idx_to_name,
            object_idx_to_name,
        )
        object_names = [
            object_idx_to_name[objects[i, 0, 0].item()],
            object_idx_to_name[objects[i, 1, 0].item()],
        ]
        img = plot_effect_grid(images, bboxes, scores[i], object_names)
        images_to_log.append(img)
        captions_to_log.append(f"{title}")

    return images_to_log, captions_to_log


def calculate_average_accuracy(
    predictions: TensorType["batch", "num_attributes"],
    labels: TensorType["batch", "num_attributes"],
    selection_mask: TensorType["batch"],
) -> torch.Tensor:
    num_attributes = predictions.shape[1]
    average_accuracy = (
        ((predictions == labels).sum(1)[selection_mask] == num_attributes)
        .float()
        .mean()
    )
    return average_accuracy


def calculate_average_attribute_accuracy(
    predictions: TensorType["batch", "num_attributes"],
    labels: TensorType["batch", "num_attributes"],
    selection_mask: TensorType["batch"],
) -> TensorType["num_attributes"]:
    average_attribute_accuracy = (predictions == labels)[selection_mask].sum(
        0
    ) / selection_mask.sum()
    return average_attribute_accuracy


def calculate_diff_accuracy(
    predictions: TensorType["batch", "num_attributes"],
    labels_pre: TensorType["batch", "num_attributes"],
    labels_post: TensorType["batch", "num_attributes"],
) -> torch.Tensor:
    diff_accuracy = torch.tensor(0, dtype=torch.float)
    diff = labels_pre != labels_post
    if diff.sum() > 0:
        diff_accuracy = (predictions[diff] == labels_post[diff]).sum() / diff.sum()
    return diff_accuracy


def log_action_accuracies(
    logger,
    action_idx_to_name: dict,
    actions: TensorType["batch_examples", "action_args"],
    predictions: TensorType["batch", "num_attributes"],
    labels: TensorType["batch", "num_attributes"],
    selection_mask: TensorType["batch"],
    split="val",
) -> None:
    for action_index, action_name in action_idx_to_name.items():
        action_mask = actions[:, 0] == action_index
        action_mask = repeat(action_mask, "b -> (b 2)") & selection_mask
        action_accuracy = calculate_average_accuracy(predictions, labels, action_mask)
        logger(f"Action Accuracy/{split}_{action_name}_accuracy", action_accuracy)


def log_average_attribute_accuracy(
    logger,
    predictions: TensorType["batch", "num_attributes"],
    labels: TensorType["batch", "num_attributes"],
    selection_mask: TensorType["batch"],
    split="val",
) -> None:
    average_attribute_accuracy = calculate_average_attribute_accuracy(
        predictions, labels, selection_mask
    )
    for acc, object_attribute_name in zip(
        average_attribute_accuracy, OBJECT_ATTRIBUTES
    ):
        logger(f"Attribute Accuracy/{split}_{object_attribute_name}_accuracy", acc)
    logger(
        f"Attribute Accuracy/{split}_average_attribute_accuracy",
        average_attribute_accuracy.mean(),
    )


def log_confusion_matrices(
    logger,
    object_mapper,
    predictions: TensorType["batch", "num_attributes"],
    labels_post: TensorType["batch", "num_attributes"],
    selection_mask: TensorType["batch"],
    split="val",
):
    preds = predictions[selection_mask]
    labels = labels_post[selection_mask]
    for idx, object_attribute_name in enumerate(OBJECT_ATTRIBUTES):
        # skip long confusion matrices
        if len(list(object_mapper[idx].values())) > 10:
            continue

        min_idx = list(object_mapper[idx].keys())[0]
        class_names = list(object_mapper[idx].values())

        print(f"object_attribute_name: {object_attribute_name}")
        print(f"idx: {idx}")
        print(f"min_idx: {min_idx}")
        print(f"class_names: {class_names}")
        print(f"preds: {preds[:, idx].numpy()}")
        print(f"y_true: {labels[:, idx].numpy()}")

        confusion_matrix = wandb.plot.confusion_matrix(
            title=f"Confusion Matrix for {object_attribute_name}",
            preds=preds[:, idx].numpy() - min_idx,
            y_true=labels[:, idx].numpy() - min_idx,
            class_names=class_names,
        )
        try:
            logger(
                {
                    f"Confusion Matrices/{split}_{object_attribute_name}": confusion_matrix
                }
            )
        except FileNotFoundError as e:
            print(f"{e}: Error saving wandb confusion matrix to /tmp/")


def log_attribute_level_error_rates(
    logger,
    predictions: TensorType["batch", "num_attributes"],
    labels_post: TensorType["batch", "num_attributes"],
    selection_mask: TensorType["batch"],
    split="val",
):
    preds = predictions[selection_mask]
    labels = labels_post[selection_mask]
    values = (preds != labels).sum(0) / preds.shape[0]
    data = [[att, val] for (att, val) in zip(OBJECT_ATTRIBUTES, values)]
    table = wandb.Table(data=data, columns=["attribute", "error_rate"])
    bar_chart = wandb.plot.bar(
        table, "attribute", "error_rate", title="Attribute Level Error Rates"
    )
    try:
        logger({f"Attribute Errors/{split}_attribute_level_error_rate": bar_chart})
    except FileNotFoundError as e:
        print(f"{e}: Error saving wandb table to /tmp/")
