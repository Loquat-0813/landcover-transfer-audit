"""Accumulate protocol-primary metrics for coverage-aware hierarchy outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p1eval.selective_hierarchy import SelectiveHierarchyMetrics, selective_hierarchy_metrics
from p1eval.taxonomy import DatasetMapping, Taxonomy


@dataclass(frozen=True)
class CoveredLeafMetrics:
    class_names: tuple[str, ...]
    per_class_iou: dict[str, float | None]
    macro_iou: float | None
    valid_pixels: int

    def as_dict(self) -> dict:
        return {
            "class_names": list(self.class_names),
            "per_class_iou": self.per_class_iou,
            "macro_iou": self.macro_iou,
            "valid_pixels": self.valid_pixels,
        }


@dataclass(frozen=True)
class CoverageAwareResult:
    covered_leaf: CoveredLeafMetrics
    selective_hierarchy: SelectiveHierarchyMetrics
    routed_valid_pixels: int
    route_rate: float | None
    mean_conformal_set_size: float | None

    def as_dict(self) -> dict:
        return {
            "covered_leaf": self.covered_leaf.as_dict(),
            "selective_hierarchy": self.selective_hierarchy.as_dict(),
            "routed_valid_pixels": self.routed_valid_pixels,
            "route_rate": self.route_rate,
            "mean_conformal_set_size": self.mean_conformal_set_size,
        }


class CoverageAwareAccumulator:
    """Accumulate actual taxonomy-node outputs without leaf-level coercion."""

    def __init__(
        self,
        mapping: DatasetMapping,
        taxonomy: Taxonomy,
        levels: dict[str, int],
        node_to_id: dict[str, int],
        leaf_names: tuple[str, ...],
        covered_leaf_mask: np.ndarray,
    ) -> None:
        if covered_leaf_mask.dtype != np.bool_ or covered_leaf_mask.shape != (len(leaf_names),):
            raise ValueError("covered_leaf_mask must be a bool vector aligned to leaf_names")
        self.mapping = mapping
        self.taxonomy = taxonomy
        self.levels = levels
        self.node_to_id = node_to_id
        self.leaf_names = leaf_names
        self.covered_leaf_mask = covered_leaf_mask
        self.leaf_node_ids = np.asarray([node_to_id[name] for name in leaf_names], dtype=np.int64)
        self.covered_leaf_node_ids = set(self.leaf_node_ids[covered_leaf_mask].tolist())
        self.covered_truth = np.zeros(len(leaf_names), dtype=np.int64)
        self.covered_prediction = np.zeros(len(leaf_names), dtype=np.int64)
        self.covered_intersection = np.zeros(len(leaf_names), dtype=np.int64)
        self.valid_pixels = 0
        self.covered_pixels = 0
        self.uncovered_pixels = 0
        self.uncovered_ancestor_correct = 0.0
        self.uncovered_leaf_violations = 0.0
        self.uncovered_utility = 0.0
        self.depth_counts = {str(level): 0 for level in sorted(set(levels.values()))}
        self.covered_depth_counts = self.depth_counts.copy()
        self.uncovered_depth_counts = self.depth_counts.copy()
        self.routed_valid_pixels = 0
        self.conformal_set_size_sum = 0
        self.conformal_set_size_pixels = 0

    def update(
        self,
        raw_mask: np.ndarray,
        node_prediction: np.ndarray,
        routed_to_base: np.ndarray | None = None,
        conformal_set_size: np.ndarray | None = None,
    ) -> None:
        if raw_mask.shape != node_prediction.shape:
            raise ValueError("raw_mask and node_prediction must have identical shapes")
        if routed_to_base is not None and routed_to_base.shape != raw_mask.shape:
            raise ValueError("routed_to_base must align to raw_mask")
        if conformal_set_size is not None and conformal_set_size.shape != raw_mask.shape:
            raise ValueError("conformal_set_size must align to raw_mask")
        target = self.mapping.adapt(raw_mask, self.node_to_id)
        target[~self.mapping.exact_valid_mask(raw_mask)] = -1
        hierarchy = selective_hierarchy_metrics(
            target,
            node_prediction,
            self.covered_leaf_node_ids,
            self.taxonomy,
            self.levels,
            self.node_to_id,
        )
        self.valid_pixels += hierarchy.valid_pixels
        self.covered_pixels += hierarchy.covered_pixels
        self.uncovered_pixels += hierarchy.uncovered_pixels
        if hierarchy.uncovered_pixels:
            self.uncovered_ancestor_correct += hierarchy.uncovered_pixels * float(hierarchy.uncovered_ancestor_correctness)
            self.uncovered_leaf_violations += hierarchy.uncovered_pixels * float(hierarchy.violation_leaf_rate)
            self.uncovered_utility += hierarchy.uncovered_pixels * float(hierarchy.uncovered_ancestor_specific_utility)
        for name, count in hierarchy.prediction_depth_counts.items():
            self.depth_counts[name] += count
        for name, count in hierarchy.covered_prediction_depth_counts.items():
            self.covered_depth_counts[name] += count
        for name, count in hierarchy.uncovered_prediction_depth_counts.items():
            self.uncovered_depth_counts[name] += count
        valid = target >= 0
        covered_target = valid & np.isin(target, self.leaf_node_ids[self.covered_leaf_mask])
        for index, leaf_node_id in enumerate(self.leaf_node_ids):
            if not self.covered_leaf_mask[index]:
                continue
            truth = covered_target & (target == leaf_node_id)
            predicted = covered_target & (node_prediction == leaf_node_id)
            self.covered_truth[index] += int(truth.sum())
            self.covered_prediction[index] += int(predicted.sum())
            self.covered_intersection[index] += int((truth & predicted).sum())
        if routed_to_base is not None:
            self.routed_valid_pixels += int(routed_to_base[valid].sum())
        if conformal_set_size is not None:
            self.conformal_set_size_sum += int(conformal_set_size[valid].sum())
            self.conformal_set_size_pixels += int(valid.sum())

    def result(self) -> CoverageAwareResult:
        per_class_iou: dict[str, float | None] = {}
        values: list[float] = []
        covered_names: list[str] = []
        for index, name in enumerate(self.leaf_names):
            if not self.covered_leaf_mask[index]:
                continue
            covered_names.append(name)
            support = int(self.covered_truth[index])
            if support == 0:
                per_class_iou[name] = None
                continue
            intersection = int(self.covered_intersection[index])
            union = support + int(self.covered_prediction[index]) - intersection
            value = intersection / union
            per_class_iou[name] = value
            values.append(value)
        hierarchy = SelectiveHierarchyMetrics(
            valid_pixels=self.valid_pixels,
            covered_pixels=self.covered_pixels,
            uncovered_pixels=self.uncovered_pixels,
            uncovered_ancestor_correctness=None
            if not self.uncovered_pixels
            else self.uncovered_ancestor_correct / self.uncovered_pixels,
            violation_leaf_rate=None if not self.uncovered_pixels else self.uncovered_leaf_violations / self.uncovered_pixels,
            uncovered_ancestor_specific_utility=None
            if not self.uncovered_pixels
            else self.uncovered_utility / self.uncovered_pixels,
            prediction_depth_counts=self.depth_counts,
            covered_prediction_depth_counts=self.covered_depth_counts,
            uncovered_prediction_depth_counts=self.uncovered_depth_counts,
        )
        return CoverageAwareResult(
            covered_leaf=CoveredLeafMetrics(
                class_names=tuple(covered_names),
                per_class_iou=per_class_iou,
                macro_iou=None if not values else float(np.mean(values)),
                valid_pixels=self.covered_pixels,
            ),
            selective_hierarchy=hierarchy,
            routed_valid_pixels=self.routed_valid_pixels,
            route_rate=None if not self.valid_pixels else self.routed_valid_pixels / self.valid_pixels,
            mean_conformal_set_size=None
            if not self.conformal_set_size_pixels
            else self.conformal_set_size_sum / self.conformal_set_size_pixels,
        )
