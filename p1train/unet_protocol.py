"""Shared training and tiled-evaluation routines for the frozen U-Net protocol."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import Tensor

from p1model.losses import taxonomy_partial_label_nll


@dataclass(frozen=True)
class UnetSegmentationResult:
    mean_iou: float
    per_class_iou: dict[str, float | None]
    valid_pixels: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stack_examples(examples: list[dict]) -> tuple[Tensor, Tensor]:
    return (
        torch.stack([example["image"] for example in examples]),
        torch.stack([example["target"] for example in examples]),
    )


def train_one_update(
    model,
    optimizer: torch.optim.Optimizer,
    images: Tensor,
    target: Tensor,
    descendant_leaf_mask: Tensor,
) -> Tensor:
    logits = model(images)
    loss = taxonomy_partial_label_nll(logits, target, descendant_leaf_mask)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


def target_nodes_to_leaf_ids(target: Tensor, leaf_node_ids: Tensor) -> Tensor:
    lookup = torch.full((int(leaf_node_ids.max()) + 1,), -1, device=target.device, dtype=torch.long)
    lookup[leaf_node_ids] = torch.arange(len(leaf_node_ids), device=target.device)
    result = torch.full_like(target, -1)
    valid = target >= 0
    result[valid] = lookup[target[valid]]
    return result


def accumulate_confusion(confusion: Tensor, target: Tensor, prediction: Tensor) -> None:
    valid = target >= 0
    if not valid.any():
        return
    class_count = confusion.shape[0]
    encoded = target[valid] * class_count + prediction[valid]
    confusion += torch.bincount(encoded, minlength=class_count * class_count).reshape(class_count, class_count)


def iou_from_confusion(confusion: Tensor, leaf_names: tuple[str, ...]) -> tuple[float, dict[str, float | None]]:
    values: list[float] = []
    per_class: dict[str, float | None] = {}
    for index, name in enumerate(leaf_names):
        target_support = int(confusion[index, :].sum())
        intersection = int(confusion[index, index])
        union = int(target_support + confusion[:, index].sum() - intersection)
        value = None if union == 0 else intersection / union
        per_class[name] = value
        if value is not None:
            values.append(value)
    return float(sum(values) / len(values)) if values else float("nan"), per_class


@torch.inference_mode()
def evaluate_unet_tiled_by_identifier(
    dataset,
    model,
    leaf_node_ids: Tensor,
    leaf_names: tuple[str, ...],
    device: torch.device,
    tile_size: int,
    prediction_callback=None,
) -> tuple[UnetSegmentationResult, dict[str, Tensor]]:
    if tile_size < 32 or tile_size % 32:
        raise ValueError("tile_size must be a positive multiple of 32")
    model.eval()
    class_count = len(leaf_names)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long, device=device)
    per_identifier: dict[str, Tensor] = {}
    for example in dataset:
        image = example["image"]
        target = example["target"]
        identifier = str(example["identifier"])
        if identifier in per_identifier:
            raise ValueError(f"duplicate evaluation identifier: {identifier}")
        height, width = target.shape
        prediction = torch.empty((height, width), dtype=torch.long, device=device)
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                tile = torch.zeros((1, image.shape[0], tile_size, tile_size), dtype=image.dtype, device=device)
                tile[0, :, : bottom - top, : right - left] = image[:, top:bottom, left:right].to(device)
                prediction[top:bottom, left:right] = model(tile).argmax(dim=1)[0, : bottom - top, : right - left]
        target_leaf = target_nodes_to_leaf_ids(target.to(device), leaf_node_ids)
        matrix = torch.zeros_like(confusion)
        accumulate_confusion(matrix, target_leaf, prediction)
        confusion += matrix
        per_identifier[identifier] = matrix.cpu()
        if prediction_callback is not None:
            prediction_callback(identifier, target_leaf.detach().cpu(), prediction.detach().cpu())
    mean_iou, per_class_iou = iou_from_confusion(confusion, leaf_names)
    return UnetSegmentationResult(mean_iou, per_class_iou, int(confusion.sum())), per_identifier
