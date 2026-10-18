"""Paired city bootstrap for a declared subset of exact ontology leaves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _indices(class_names: Sequence[str], supported_leaves: Sequence[str]) -> tuple[int, ...]:
    names = tuple(class_names)
    leaves = tuple(supported_leaves)
    if not leaves or len(set(leaves)) != len(leaves):
        raise ValueError("supported leaves must be non-empty and unique")
    unknown = [leaf for leaf in leaves if leaf not in names]
    if unknown:
        raise ValueError(f"supported leaves are absent from class names: {unknown}")
    return tuple(names.index(leaf) for leaf in leaves)


def supported_leaf_mean_iou(
    confusion: np.ndarray,
    class_names: Sequence[str],
    supported_leaves: Sequence[str],
) -> float:
    """Return mean IoU for selected leaves without removing their prediction errors."""
    matrix = np.asarray(confusion, dtype=np.int64)
    names = tuple(class_names)
    if matrix.shape != (len(names), len(names)) or np.any(matrix < 0):
        raise ValueError("confusion shape or values are invalid")
    values: list[float] = []
    for index in _indices(names, supported_leaves):
        target_support = int(matrix[index, :].sum())
        if target_support == 0:
            continue
        intersection = int(matrix[index, index])
        union = target_support + int(matrix[:, index].sum()) - intersection
        values.append(intersection / union)
    if not values:
        raise ValueError("selected supported leaves have no target support")
    return float(np.mean(values))


def _supported_leaf_mean_iou_or_nan(
    confusion: np.ndarray,
    class_names: Sequence[str],
    supported_leaves: Sequence[str],
) -> float:
    names = tuple(class_names)
    matrix = np.asarray(confusion, dtype=np.int64)
    values: list[float] = []
    for index in _indices(names, supported_leaves):
        target_support = int(matrix[index, :].sum())
        if target_support == 0:
            continue
        intersection = int(matrix[index, index])
        union = target_support + int(matrix[:, index].sum()) - intersection
        values.append(intersection / union)
    return float(np.mean(values)) if values else float("nan")


def paired_city_bootstrap_mean_supported_leaf_difference(
    baseline_city_confusions: Sequence[Mapping[str, np.ndarray]],
    model_city_confusions: Sequence[Mapping[str, np.ndarray]],
    class_names: Sequence[str],
    supported_leaves: Sequence[str],
    *,
    replicates: int = 2000,
    seed: int = 20260803,
) -> dict:
    """Bootstrap mean model-minus-baseline supported-leaf mIoU with shared city draws."""
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not baseline_city_confusions or not model_city_confusions:
        raise ValueError("at least one baseline and model seed are required")
    names = tuple(class_names)
    _indices(names, supported_leaves)
    cities = tuple(sorted(baseline_city_confusions[0]))
    if not cities or any(tuple(sorted(item)) != cities for item in (*baseline_city_confusions, *model_city_confusions)):
        raise ValueError("baseline and model city sets must match exactly")
    baselines = np.stack(
        [np.stack([np.asarray(confusions[city], dtype=np.int64) for city in cities]) for confusions in baseline_city_confusions]
    )
    models = np.stack(
        [np.stack([np.asarray(confusions[city], dtype=np.int64) for city in cities]) for confusions in model_city_confusions]
    )
    expected_shape = (len(names), len(names))
    if baselines.shape[2:] != expected_shape or models.shape[2:] != expected_shape:
        raise ValueError("city confusion shape does not match class names")

    def mean_seed_metric(values: np.ndarray) -> float:
        return float(np.mean([supported_leaf_mean_iou(seed_counts, names, supported_leaves) for seed_counts in values]))

    baseline_point = mean_seed_metric(baselines.sum(axis=1))
    model_point = mean_seed_metric(models.sum(axis=1))
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(cities), size=(replicates, len(cities)))
    baseline_samples = np.nanmean(
        [
            np.asarray(
                [_supported_leaf_mean_iou_or_nan(counts, names, supported_leaves) for counts in seed_counts[draws].sum(axis=1)]
            )
            for seed_counts in baselines
        ],
        axis=0,
    )
    model_samples = np.nanmean(
        [
            np.asarray(
                [_supported_leaf_mean_iou_or_nan(counts, names, supported_leaves) for counts in seed_counts[draws].sum(axis=1)]
            )
            for seed_counts in models
        ],
        axis=0,
    )
    differences = model_samples - baseline_samples
    finite = differences[np.isfinite(differences)]
    if not finite.size:
        raise ValueError("bootstrap produced no finite supported-leaf differences")
    lower, upper = np.quantile(finite, (0.025, 0.975))
    return {
        "comparison": "mean_model_minus_mean_baseline",
        "resampling_unit": "openearthmap_city",
        "city_names": list(cities),
        "city_count": len(cities),
        "model_seed_count": len(model_city_confusions),
        "baseline_seed_count": len(baseline_city_confusions),
        "supported_leaves": list(supported_leaves),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
        "baseline_supported_leaf_mean_iou": baseline_point,
        "model_supported_leaf_mean_iou": model_point,
        "point_difference": model_point - baseline_point,
        "confidence_interval": {"lower": float(lower), "upper": float(upper)},
    }
