"""Coverage-aware hierarchy helpers for selective land-cover predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p1eval.taxonomy import Taxonomy


def _lowest_common_ancestor(nodes: tuple[str, ...], taxonomy: Taxonomy, levels: dict[str, int]) -> str:
    if not nodes:
        raise ValueError("lowest common ancestor needs at least one node")
    common: set[str] | None = None
    for node in nodes:
        ancestors: set[str] = set()
        current: str | None = node
        while current is not None:
            ancestors.add(current)
            current = taxonomy.parents[current]
        common = ancestors if common is None else common.intersection(ancestors)
    assert common is not None
    return max(common, key=lambda name: levels[name])


def build_coverage_node_lookup(
    taxonomy: Taxonomy,
    levels: dict[str, int],
    node_to_id: dict[str, int],
    leaf_names: tuple[str, ...],
    covered_leaf_mask: np.ndarray,
) -> np.ndarray:
    """Map every non-empty conformal leaf set to its permitted emitted node.

    A singleton source-covered leaf may be emitted at leaf level. A singleton
    source-uncovered leaf is promoted to its parent. Multi-leaf conformal sets
    are emitted at their lowest common ancestor.
    """
    if covered_leaf_mask.dtype != np.bool_ or covered_leaf_mask.shape != (len(leaf_names),):
        raise ValueError("covered_leaf_mask must be a bool vector aligned to leaf_names")
    if len(leaf_names) > 20:
        raise ValueError("bitset lookup is restricted to at most 20 leaves")
    if len(set(leaf_names)) != len(leaf_names) or any(name not in taxonomy.parents for name in leaf_names):
        raise ValueError("leaf names must be unique taxonomy nodes")
    lookup = np.full(1 << len(leaf_names), -1, dtype=np.int64)
    for code in range(1, len(lookup)):
        selected = tuple(name for index, name in enumerate(leaf_names) if code & (1 << index))
        if len(selected) == 1 and covered_leaf_mask[leaf_names.index(selected[0])]:
            emitted = selected[0]
        elif len(selected) == 1:
            parent = taxonomy.parents[selected[0]]
            if parent is None:
                raise ValueError("an uncovered root leaf cannot be promoted")
            emitted = parent
        else:
            emitted = _lowest_common_ancestor(selected, taxonomy, levels)
        lookup[code] = node_to_id[emitted]
    return lookup


def is_ancestor_or_same(predicted_node_id: int, target_node_id: int, taxonomy: Taxonomy, node_to_id: dict[str, int]) -> bool:
    """Return whether the emitted node is a semantically valid ancestor answer."""
    id_to_node = {node_id: name for name, node_id in node_to_id.items()}
    if predicted_node_id not in id_to_node or target_node_id not in id_to_node:
        raise ValueError("predicted and target nodes must belong to the taxonomy")
    current: str | None = id_to_node[target_node_id]
    while current is not None:
        if current == id_to_node[predicted_node_id]:
            return True
        current = taxonomy.parents[current]
    return False


@dataclass(frozen=True)
class SelectiveHierarchyMetrics:
    valid_pixels: int
    covered_pixels: int
    uncovered_pixels: int
    uncovered_ancestor_correctness: float | None
    violation_leaf_rate: float | None
    uncovered_ancestor_specific_utility: float | None
    prediction_depth_counts: dict[str, int]
    covered_prediction_depth_counts: dict[str, int]
    uncovered_prediction_depth_counts: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "valid_pixels": self.valid_pixels,
            "covered_pixels": self.covered_pixels,
            "uncovered_pixels": self.uncovered_pixels,
            "uncovered_ancestor_correctness": self.uncovered_ancestor_correctness,
            "violation_leaf_rate": self.violation_leaf_rate,
            "uncovered_ancestor_specific_utility": self.uncovered_ancestor_specific_utility,
            "prediction_depth_counts": self.prediction_depth_counts,
            "covered_prediction_depth_counts": self.covered_prediction_depth_counts,
            "uncovered_prediction_depth_counts": self.uncovered_prediction_depth_counts,
        }


def selective_hierarchy_metrics(
    target_node_ids: np.ndarray,
    predicted_node_ids: np.ndarray,
    covered_leaf_node_ids: set[int],
    taxonomy: Taxonomy,
    levels: dict[str, int],
    node_to_id: dict[str, int],
    ignore_index: int = -1,
) -> SelectiveHierarchyMetrics:
    """Evaluate whether source-uncovered targets receive warranted coarse answers."""
    if target_node_ids.shape != predicted_node_ids.shape:
        raise ValueError("target and prediction arrays must have identical shapes")
    id_to_node = {node_id: name for name, node_id in node_to_id.items()}
    valid = target_node_ids != ignore_index
    targets = target_node_ids[valid].astype(np.int64, copy=False)
    predictions = predicted_node_ids[valid].astype(np.int64, copy=False)
    if targets.size and (set(np.unique(targets)).difference(id_to_node) or set(np.unique(predictions)).difference(id_to_node)):
        raise ValueError("target or prediction contains a node outside the taxonomy")
    level_values = tuple(sorted(set(levels.values())))
    if not targets.size:
        empty_depths = {str(level): 0 for level in level_values}
        return SelectiveHierarchyMetrics(0, 0, 0, None, None, None, empty_depths, empty_depths.copy(), empty_depths.copy())
    max_node_id = max(node_to_id.values())
    depth_by_node_id = np.full(max_node_id + 1, -1, dtype=np.int64)
    covered_by_node_id = np.zeros(max_node_id + 1, dtype=bool)
    ancestor_relation = np.zeros((max_node_id + 1, max_node_id + 1), dtype=bool)
    for name, node_id in node_to_id.items():
        depth_by_node_id[node_id] = levels[name]
        covered_by_node_id[node_id] = node_id in covered_leaf_node_ids
        current: str | None = name
        while current is not None:
            ancestor_relation[node_to_id[current], node_id] = True
            current = taxonomy.parents[current]
    prediction_depth = depth_by_node_id[predictions]

    def depth_counts(mask: np.ndarray) -> dict[str, int]:
        return {str(level): int((prediction_depth[mask] == level).sum()) for level in level_values}

    depths = depth_counts(np.ones(targets.shape, dtype=bool))
    uncovered = ~covered_by_node_id[targets]
    covered = ~uncovered
    if not uncovered.any():
        return SelectiveHierarchyMetrics(
            int(targets.size),
            int(covered.sum()),
            0,
            None,
            None,
            None,
            depths,
            depth_counts(covered),
            depth_counts(uncovered),
        )
    selected_targets = targets[uncovered]
    selected_predictions = predictions[uncovered]
    ancestor = ancestor_relation[selected_predictions, selected_targets]
    leaf_depth = max(levels.values())
    violation = depth_by_node_id[selected_predictions] == leaf_depth
    utility = np.where(
        ancestor,
        depth_by_node_id[selected_predictions] / depth_by_node_id[selected_targets],
        0.0,
    )
    return SelectiveHierarchyMetrics(
        valid_pixels=int(targets.size),
        covered_pixels=int(covered.sum()),
        uncovered_pixels=int(uncovered.sum()),
        uncovered_ancestor_correctness=float(ancestor.mean()),
        violation_leaf_rate=float(violation.mean()),
        uncovered_ancestor_specific_utility=float(utility.mean()),
        prediction_depth_counts=depths,
        covered_prediction_depth_counts=depth_counts(covered),
        uncovered_prediction_depth_counts=depth_counts(uncovered),
    )
