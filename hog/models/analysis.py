from typing import Dict, Tuple

import torch
import wandb
from dataset import denormalize_image
from einops import repeat
from torchtyping import TensorType

from .mappings import OBJECT_ATTRIBUTES


def generate_wandb_bounding_boxes(
    bboxes: TensorType["N", "box"], scores: TensorType["obj", "N"]
) -> dict:
    """
    Format the bboxes and scores for wandb logger
    """
    image_before_boxes = {"predictions": {"box_data": []}}
    for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes):
        obj1_score = scores[0][i]
        obj2_score = scores[1][i]
        box = {
            "position": {
                "minX": x_min.item(),
                "maxX": x_max.item(),
                "minY": y_min.item(),
                "maxY": y_max.item(),
            },
            "domain": "pixel",
            "class_id": i,
            "scores": {
                "obj_1_score": obj1_score.item(),
                "obj_2_score": obj2_score.item(),
            },
        }
        image_before_boxes["predictions"]["box_data"].append(box)
    return image_before_boxes


def get_example_title_from_actions_object(
    actions, objects, index: int, action_idx_to_name: dict, object_idx_to_name: dict
) -> str:
    """
    Actions: [B x 3]
    Object: [B x 2, num_attributes]
    """
    assert 0 <= index < objects.shape[0] - 1

    action_text = action_idx_to_name[actions[int(index / 2)][0].item()]
    action_object = object_idx_to_name[actions[int(index / 2)][1].item()]
    action_receptacle = object_idx_to_name[actions[int(index / 2)][2].item()]
    object_1 = object_idx_to_name[objects[index][0].item()]
    object_2 = object_idx_to_name[objects[index + 1][0].item()]
    return f"Effects of {action_text}({action_object}, {action_receptacle}) on ({object_1},{object_2})"


def plot_images(
    outputs: Dict[str, torch.Tensor], action_idx_to_name: dict, object_idx_to_name: dict
) -> Tuple[list, list, list]:
    images = outputs["images"]
    bboxes = outputs["image_bboxes"]
    scores = outputs["image_bbox_scores"]

    images_to_log = []
    boxes_to_log = []
    captions_to_log = []
    for img_idx in range(0, images.shape[0], 2):
        # format bounding boxes for before image
        image_before = denormalize_image(images[img_idx]).permute(1, 2, 0).numpy()
        image_before_boxes = generate_wandb_bounding_boxes(
            bboxes[img_idx], scores[img_idx]
        )
        # format bounding boxes for after image
        image_after = denormalize_image(images[img_idx + 1]).permute(1, 2, 0).numpy()
        image_after_boxes = generate_wandb_bounding_boxes(
            bboxes[img_idx + 1], scores[img_idx + 1]
        )
        title = get_example_title_from_actions_object(
            outputs["actions"],
            outputs["objects_labels_pre"],
            img_idx,
            action_idx_to_name,
            object_idx_to_name,
        )
        images_to_log += [image_before, image_after]
        boxes_to_log += [image_before_boxes, image_after_boxes]
        captions_to_log += [f"Before: {title}", f"After: {title}"]

    return images_to_log, boxes_to_log, captions_to_log


def calculate_average_accuracy(
    predictions: TensorType["batch", "num_attributes"],
    labels: TensorType["batch", "num_attributes"],
    selection_mask: TensorType["batch"],
) -> torch.Tensor:
    num_objects, num_attributes = predictions.shape
    average_accuracy = (
        (predictions == labels).sum(1)[selection_mask] == num_attributes
    ).sum() / num_objects
    return average_accuracy


def calculate_average_attribute_accuracy(
    predictions: TensorType["batch", "num_attributes"],
    labels: TensorType["batch", "num_attributes"],
    selection_mask: TensorType["batch"],
) -> TensorType["num_attributes"]:
    num_objects = predictions.shape[0]
    average_attribute_accuracy = (predictions == labels)[selection_mask].sum(
        0
    ) / num_objects
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
        logger(f"{split}/accuracy/action_level/{action_name}_accuracy", action_accuracy)


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
        logger(f"{split}/accuracy/attribute_level/{object_attribute_name}", acc)
    logger(
        f"{split}/accuracy/average_attribute_accuracy",
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
        confusion_matrix = wandb.plot.confusion_matrix(
            title=f"Confusion Matrix for {object_attribute_name}",
            preds=preds[:, idx].numpy() - min_idx,
            y_true=labels[:, idx].numpy() - min_idx,
            class_names=class_names,
        )
        logger({f"{split}/{object_attribute_name}": confusion_matrix})


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
    logger({f"{split}/attribute_level_error_rate": bar_chart})
