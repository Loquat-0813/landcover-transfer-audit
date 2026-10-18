"""Metrics for exact and coarse labels in external P1 transfer evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import IoUResult, hierarchy_iou, iou_from_taxonomy_ids
from .taxonomy import DatasetMapping, Taxonomy


def _target_supported_iou(target: np.ndarray, prediction: np.ndarray, class_names: tuple[str, ...]) -> IoUResult:
    """Score only labels represented by ground truth, retaining false positives in their unions."""
    result = iou_from_taxonomy_ids(target, prediction, class_names)
    per_class_iou = dict(result.per_class_iou)
    supported: list[float] = []
    for index, name in enumerate(class_names):
        if int(result.confusion[index, :].sum()) == 0:
            per_class_iou[name] = None
        elif per_class_iou[name] is not None:
            supported.append(per_class_iou[name])
    return IoUResult(
        class_names=result.class_names,
        confusion=result.confusion,
        per_class_iou=per_class_iou,
        macro_iou=None if not supported else float(np.mean(supported)),
        valid_pixels=result.valid_pixels,
    )


def _leaf_targets(
    raw_mask: np.ndarray,
    prediction_leaf_ids: np.ndarray,
    mapping: DatasetMapping,
    node_to_id: dict[str, int],
    leaf_node_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if raw_mask.shape != prediction_leaf_ids.shape:
        raise ValueError("raw mask and predicted leaf IDs must have identical shapes")
    target_nodes = mapping.adapt(raw_mask, node_to_id)
    node_to_leaf = np.full(len(node_to_id), -1, dtype=np.int64)
    node_to_leaf[leaf_node_ids] = np.arange(len(leaf_node_ids), dtype=np.int64)
    target = np.full(raw_mask.shape, -1, dtype=np.int64)
    valid = mapping.exact_valid_mask(raw_mask)
    target[valid] = node_to_leaf[target_nodes[valid]]
    if np.any(target[valid] < 0):
        raise ValueError(f"{mapping.dataset}: an exact label did not map to a taxonomy leaf")
    return target, prediction_leaf_ids.astype(np.int64, copy=False)


def exact_leaf_iou(
    raw_mask: np.ndarray,
    prediction_leaf_ids: np.ndarray,
    mapping: DatasetMapping,
    node_to_id: dict[str, int],
    leaf_node_ids: np.ndarray,
    leaf_names: tuple[str, ...],
) -> IoUResult:
    """Compute leaf IoU only on source labels declared exact in the frozen mapping."""
    target, prediction = _leaf_targets(raw_mask, prediction_leaf_ids, mapping, node_to_id, leaf_node_ids)
    return _target_supported_iou(target, prediction, leaf_names)


def coarse_ancestor_iou(
    raw_mask: np.ndarray,
    prediction_leaf_ids: np.ndarray,
    mapping: DatasetMapping,
    taxonomy: Taxonomy,
    levels: dict[str, int],
    node_to_id: dict[str, int],
    leaf_node_ids: np.ndarray,
    level: int = 1,
) -> IoUResult | None:
    """Score coarse source annotations only after collapsing predictions to an ancestor level."""
    if not mapping.coarse_raw_ids:
        return None
    if raw_mask.shape != prediction_leaf_ids.shape:
        raise ValueError("raw mask and predicted leaf IDs must have identical shapes")
    target = mapping.adapt(raw_mask, node_to_id)
    target[~np.isin(raw_mask, tuple(mapping.coarse_raw_ids))] = -1
    prediction = leaf_node_ids[prediction_leaf_ids]
    result = hierarchy_iou(target, prediction, taxonomy, levels, node_to_id, level=level)
    return _target_supported_iou(
        # hierarchy_iou has already validated and collapsed the arrays, but its public
        # result retains all level names. Recompute on its confusion-equivalent labels.
        _collapse(target, taxonomy, levels, node_to_id, level),
        _collapse(prediction, taxonomy, levels, node_to_id, level),
        result.class_names,
    )


def _collapse(
    labels: np.ndarray,
    taxonomy: Taxonomy,
    levels: dict[str, int],
    node_to_id: dict[str, int],
    level: int,
) -> np.ndarray:
    names = tuple(name for name in taxonomy.nodes if levels[name] == level)
    output_index = {name: index for index, name in enumerate(names)}
    result = np.full(labels.shape, -1, dtype=np.int64)
    for name, node_id in node_to_id.items():
        valid = labels == node_id
        if valid.any():
            result[valid] = output_index[taxonomy.ancestor_at_level(name, level, levels)]
    return result


@dataclass(frozen=True)
class ExternalTransferResult:
    exact_leaf: IoUResult
    coarse_ancestor: IoUResult | None

    def as_dict(self) -> dict:
        return {
            "exact_leaf": self.exact_leaf.as_dict(),
            "coarse_ancestor": None if self.coarse_ancestor is None else self.coarse_ancestor.as_dict(),
        }


class ExternalTransferAccumulator:
    """Accumulate exact and coarse confusion matrices across full external rasters."""

    def __init__(
        self,
        mapping: DatasetMapping,
        taxonomy: Taxonomy,
        levels: dict[str, int],
        node_to_id: dict[str, int],
        leaf_node_ids: np.ndarray,
        leaf_names: tuple[str, ...],
    ) -> None:
        self.mapping = mapping
        self.taxonomy = taxonomy
        self.levels = levels
        self.node_to_id = node_to_id
        self.leaf_node_ids = leaf_node_ids
        self.leaf_names = leaf_names
        self.exact_confusion = np.zeros((len(leaf_names), len(leaf_names)), dtype=np.int64)
        self.coarse_confusion: np.ndarray | None = None
        self.coarse_names: tuple[str, ...] | None = None

    def update(self, raw_mask: np.ndarray, prediction_leaf_ids: np.ndarray) -> None:
        exact = exact_leaf_iou(
            raw_mask, prediction_leaf_ids, self.mapping, self.node_to_id, self.leaf_node_ids, self.leaf_names
        )
        self.exact_confusion += exact.confusion
        coarse = coarse_ancestor_iou(
            raw_mask,
            prediction_leaf_ids,
            self.mapping,
            self.taxonomy,
            self.levels,
            self.node_to_id,
            self.leaf_node_ids,
        )
        if coarse is not None:
            if self.coarse_confusion is None:
                self.coarse_confusion = np.zeros_like(coarse.confusion)
                self.coarse_names = coarse.class_names
            self.coarse_confusion += coarse.confusion

    def result(self) -> ExternalTransferResult:
        exact = _target_supported_iou_from_confusion(self.exact_confusion, self.leaf_names)
        coarse = None
        if self.coarse_confusion is not None and self.coarse_names is not None:
            coarse = _target_supported_iou_from_confusion(self.coarse_confusion, self.coarse_names)
        return ExternalTransferResult(exact_leaf=exact, coarse_ancestor=coarse)


def _target_supported_iou_from_confusion(confusion: np.ndarray, class_names: tuple[str, ...]) -> IoUResult:
    per_class_iou: dict[str, float | None] = {}
    values: list[float] = []
    for index, name in enumerate(class_names):
        support = int(confusion[index, :].sum())
        if support == 0:
            per_class_iou[name] = None
            continue
        intersection = int(confusion[index, index])
        union = support + int(confusion[:, index].sum()) - intersection
        value = intersection / union
        per_class_iou[name] = value
        values.append(value)
    return IoUResult(
        class_names=class_names,
        confusion=confusion,
        per_class_iou=per_class_iou,
        macro_iou=None if not values else float(np.mean(values)),
        valid_pixels=int(confusion.sum()),
    )
