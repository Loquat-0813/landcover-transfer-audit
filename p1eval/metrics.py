"""Taxonomy-aware semantic-segmentation metrics for P1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .taxonomy import Taxonomy


@dataclass(frozen=True)
class IoUResult:
    class_names: tuple[str, ...]
    confusion: np.ndarray
    per_class_iou: dict[str, float | None]
    macro_iou: float | None
    valid_pixels: int

    def as_dict(self) -> dict:
        return {
            "class_names": list(self.class_names),
            "confusion": self.confusion.tolist(),
            "per_class_iou": self.per_class_iou,
            "macro_iou": self.macro_iou,
            "valid_pixels": self.valid_pixels,
        }


def unknown_rejection_auroc(unknown_target: np.ndarray, unknown_score: np.ndarray, valid_mask: np.ndarray) -> float | None:
    """Compute AUROC for an unknown score, excluding void and ambiguous pixels.

    Returns ``None`` when the eligible pixels do not contain both known and genuinely
    unknown labels. This makes an unavailable metric explicit rather than inventing
    a number from background or void pixels.
    """
    if unknown_target.shape != unknown_score.shape or unknown_target.shape != valid_mask.shape:
        raise ValueError("unknown target, score, and valid mask must have identical shapes")
    labels = unknown_target[valid_mask].astype(bool, copy=False)
    scores = unknown_score[valid_mask].astype(np.float64, copy=False)
    if labels.size == 0 or not np.isfinite(scores).all():
        raise ValueError("unknown score has no eligible finite values")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def stable_node_index(nodes: tuple[str, ...]) -> dict[str, int]:
    """Use the YAML node order as the stable serialization order."""
    return {node: index for index, node in enumerate(nodes)}


def _confusion_matrix(target: np.ndarray, prediction: np.ndarray, class_count: int, ignore_index: int) -> tuple[np.ndarray, int]:
    if target.shape != prediction.shape:
        raise ValueError(f"target shape {target.shape} does not match prediction shape {prediction.shape}")
    valid = target != ignore_index
    target = target[valid].astype(np.int64, copy=False)
    prediction = prediction[valid].astype(np.int64, copy=False)
    if target.size == 0:
        return np.zeros((class_count, class_count), dtype=np.int64), 0
    if target.min() < 0 or target.max() >= class_count:
        raise ValueError("target contains a class outside the scoring taxonomy")
    if prediction.min() < 0 or prediction.max() >= class_count:
        raise ValueError("prediction contains a class outside the scoring taxonomy")
    encoded = target * class_count + prediction
    return np.bincount(encoded, minlength=class_count * class_count).reshape(class_count, class_count), int(target.size)


def iou_from_taxonomy_ids(
    target: np.ndarray,
    prediction: np.ndarray,
    class_names: tuple[str, ...],
    ignore_index: int = -1,
) -> IoUResult:
    """Compute class IoU where rows are ground truth and columns predictions."""
    confusion, valid_pixels = _confusion_matrix(target, prediction, len(class_names), ignore_index)
    per_class_iou: dict[str, float | None] = {}
    finite_values: list[float] = []
    for index, name in enumerate(class_names):
        intersection = int(confusion[index, index])
        union = int(confusion[index, :].sum() + confusion[:, index].sum() - intersection)
        iou = None if union == 0 else intersection / union
        per_class_iou[name] = iou
        if iou is not None:
            finite_values.append(iou)
    macro = None if not finite_values else float(np.mean(finite_values))
    return IoUResult(class_names, confusion, per_class_iou, macro, valid_pixels)


def remap_to_ancestor_level(
    labels: np.ndarray,
    taxonomy: Taxonomy,
    levels: dict[str, int],
    node_to_id: dict[str, int],
    level: int,
    ignore_index: int = -1,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Collapse taxonomy IDs to all nodes at a requested taxonomy level."""
    if level < 0:
        raise ValueError("taxonomy level must be non-negative")
    names = tuple(name for name in taxonomy.nodes if levels[name] == level)
    if not names:
        raise ValueError(f"taxonomy has no nodes at level {level}")
    output_index = {name: index for index, name in enumerate(names)}
    lookup = np.full(len(node_to_id), ignore_index, dtype=np.int64)
    for name, old_id in node_to_id.items():
        if levels[name] >= level:
            lookup[old_id] = output_index[taxonomy.ancestor_at_level(name, level, levels)]
    result = np.full(labels.shape, ignore_index, dtype=np.int64)
    valid = labels != ignore_index
    if valid.any():
        values = labels[valid]
        if values.min() < 0 or values.max() >= len(lookup):
            raise ValueError("labels contain a class outside the taxonomy")
        result[valid] = lookup[values]
    return result, names


def hierarchy_iou(
    target: np.ndarray,
    prediction: np.ndarray,
    taxonomy: Taxonomy,
    levels: dict[str, int],
    node_to_id: dict[str, int],
    level: int,
    ignore_index: int = -1,
) -> IoUResult:
    """Compute IoU after collapsing target and predictions to an ancestor level."""
    remapped_target, names = remap_to_ancestor_level(target, taxonomy, levels, node_to_id, level, ignore_index)
    remapped_prediction, _ = remap_to_ancestor_level(prediction, taxonomy, levels, node_to_id, level, ignore_index)
    return iou_from_taxonomy_ids(remapped_target, remapped_prediction, names, ignore_index)
