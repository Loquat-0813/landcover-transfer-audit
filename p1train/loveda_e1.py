"""E0/E1 training and tiled evaluation for the LoveDA frozen-CLIP protocol."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from p1model.calibration import LowRankSemanticCalibration
from p1model.losses import taxonomy_partial_label_nll


def source_supervised_leaf_mask(mapping, leaf_names: tuple[str, ...]) -> Tensor:
    """Return leaves with exact supervision in the current source dataset."""
    supervised_nodes = {mapping.raw_id_to_node[raw_id] for raw_id in mapping.exact_raw_ids}
    unknown_nodes = supervised_nodes.difference(leaf_names)
    if unknown_nodes:
        raise ValueError(f"source exact labels must map to leaves, got {sorted(unknown_nodes)}")
    return torch.tensor([name in supervised_nodes for name in leaf_names], dtype=torch.bool)


@dataclass(frozen=True)
class SegmentationResult:
    mean_iou: float
    per_class_iou: dict[str, float | None]
    valid_pixels: int


def base_leaf_logits(pixel_features: Tensor, text_features: Tensor, logit_scale: float = 1 / 0.07) -> Tensor:
    """Frozen E0 text-cost baseline without any trainable calibration term."""
    image = F.normalize(pixel_features, dim=1)
    text = F.normalize(text_features, dim=1)
    return logit_scale * torch.einsum("bdhw,ld->blhw", image, text)


def train_calibration_epoch(
    loader,
    backbone,
    adapter: LowRankSemanticCalibration,
    text_features: Tensor,
    descendant_leaf_mask: Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train only the E1 low-rank adapter for one epoch."""
    adapter.train()
    backbone.eval()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        features = backbone.encode_image(images)
        logits = adapter(features, text_features)
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        loss = taxonomy_partial_label_nll(logits, target, descendant_leaf_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        batches += 1
    if batches == 0:
        raise ValueError("training loader produced no batches")
    return total_loss / batches


def _target_nodes_to_leaf_ids(target: Tensor, leaf_node_ids: Tensor) -> Tensor:
    lookup = torch.full((int(leaf_node_ids.max()) + 1,), -1, device=target.device, dtype=torch.long)
    lookup[leaf_node_ids] = torch.arange(len(leaf_node_ids), device=target.device)
    result = torch.full_like(target, -1)
    valid = target >= 0
    result[valid] = lookup[target[valid]]
    return result


def _accumulate_confusion(confusion: Tensor, target: Tensor, prediction: Tensor) -> None:
    valid = target >= 0
    if not valid.any():
        return
    class_count = confusion.shape[0]
    encoded = target[valid] * class_count + prediction[valid]
    confusion += torch.bincount(encoded, minlength=class_count * class_count).reshape(class_count, class_count)


def _iou_from_loveda_confusion(confusion: Tensor, leaf_names: tuple[str, ...]) -> tuple[float, dict[str, float | None]]:
    """Score only leaves represented by LoveDA ground truth.

    The shared taxonomy includes ``herbaceous_vegetation`` for external datasets,
    but LoveDA has no corresponding source label. Predictions of that unsupported
    leaf still lower the IoU of the true LoveDA classes, but the leaf itself must
    not add an artificial zero to the LoveDA macro average.
    """
    per_class_iou: dict[str, float | None] = {}
    values: list[float] = []
    for index, name in enumerate(leaf_names):
        target_support = int(confusion[index, :].sum())
        if target_support == 0:
            per_class_iou[name] = None
            continue
        intersection = int(confusion[index, index])
        union = int(target_support + confusion[:, index].sum() - intersection)
        value = intersection / union
        per_class_iou[name] = value
        values.append(value)
    return float(sum(values) / len(values)) if values else float("nan"), per_class_iou


@torch.inference_mode()
def evaluate_taxonomy_tiled(
    dataset,
    backbone,
    adapter: LowRankSemanticCalibration | None,
    text_features: Tensor,
    descendant_leaf_mask: Tensor,
    leaf_node_ids: Tensor,
    leaf_names: tuple[str, ...],
    device: torch.device,
    tile_size: int,
) -> SegmentationResult:
    """Evaluate exact taxonomy leaves on padded, non-overlapping RGB tiles."""
    result, _ = evaluate_taxonomy_tiled_by_identifier(
        dataset, backbone, adapter, text_features, descendant_leaf_mask, leaf_node_ids, leaf_names, device, tile_size
    )
    return result


@torch.inference_mode()
def evaluate_taxonomy_tiled_by_identifier(
    dataset,
    backbone,
    adapter: LowRankSemanticCalibration | None,
    text_features: Tensor,
    descendant_leaf_mask: Tensor,
    leaf_node_ids: Tensor,
    leaf_names: tuple[str, ...],
    device: torch.device,
    tile_size: int,
    prediction_callback=None,
) -> tuple[SegmentationResult, dict[str, Tensor]]:
    """Evaluate taxonomy leaves and retain one exact-label confusion matrix per raster."""
    if adapter is not None:
        adapter.eval()
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    class_count = len(leaf_names)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long, device=device)
    per_identifier_confusion: dict[str, Tensor] = {}
    for example in dataset:
        image = example["image"]
        target = example["target"]
        identifier = str(example["identifier"])
        if identifier in per_identifier_confusion:
            raise ValueError(f"duplicate evaluation identifier: {identifier}")
        height, width = target.shape
        predicted = torch.empty((height, width), dtype=torch.long, device=device)
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                tile = torch.zeros((image.shape[0], tile_size, tile_size), dtype=image.dtype)
                tile[:, : bottom - top, : right - left] = image[:, top:bottom, left:right]
                tile = tile.unsqueeze(0).to(device)
                features = backbone.encode_image(tile)
                logits = base_leaf_logits(features, text_features) if adapter is None else adapter(features, text_features)
                logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                predicted[top:bottom, left:right] = logits.argmax(dim=1)[0, : bottom - top, : right - left]
        target_leaf = _target_nodes_to_leaf_ids(target.to(device), leaf_node_ids)
        per_raster_confusion = torch.zeros_like(confusion)
        _accumulate_confusion(per_raster_confusion, target_leaf, predicted)
        confusion += per_raster_confusion
        per_identifier_confusion[identifier] = per_raster_confusion.cpu()
        if prediction_callback is not None:
            prediction_callback(identifier, target_leaf.detach().cpu(), predicted.detach().cpu())
    mean_iou, per_class_iou = _iou_from_loveda_confusion(confusion, leaf_names)
    return (
        SegmentationResult(
            mean_iou=mean_iou,
            per_class_iou=per_class_iou,
            valid_pixels=int(confusion.sum()),
        ),
        per_identifier_confusion,
    )


evaluate_loveda_tiled = evaluate_taxonomy_tiled
