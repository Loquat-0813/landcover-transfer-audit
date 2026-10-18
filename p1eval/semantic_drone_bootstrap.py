"""Paired raster bootstrap for the narrow Semantic Drone grass score."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from p1data.splits import identifier_set_sha256

from p1eval.semantic_drone_metrics import grass_iou


_COUNT_KEYS = ("true_positive", "false_positive", "false_negative")


def exclude_identifiers(
    per_identifier: Mapping[str, Mapping[str, object]], identifiers: Sequence[str]
) -> dict[str, Mapping[str, object]]:
    """Return a validated subset after removing explicitly named rasters."""
    excluded = tuple(sorted(set(identifiers)))
    missing = [identifier for identifier in excluded if identifier not in per_identifier]
    if missing:
        raise ValueError(f"excluded identifiers are absent from results: {missing}")
    retained = {identifier: value for identifier, value in per_identifier.items() if identifier not in excluded}
    if not retained:
        raise ValueError("excluding identifiers leaves no Semantic Drone rasters")
    return retained


def _validated_counts(per_identifier: Mapping[str, Mapping[str, object]]) -> tuple[tuple[str, ...], np.ndarray]:
    """Return sorted per-raster confusion counts after strict validation."""
    if not per_identifier:
        raise ValueError("at least one Semantic Drone raster is required")
    identifiers = tuple(sorted(per_identifier))
    rows: list[tuple[int, int, int]] = []
    for identifier in identifiers:
        value = per_identifier[identifier]
        if not isinstance(value, Mapping):
            raise ValueError(f"{identifier}: per-raster result must be a mapping")
        row: list[int] = []
        for key in _COUNT_KEYS:
            count = value.get(key)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{identifier}: {key} must be a non-negative integer")
            row.append(count)
        rows.append(tuple(row))
    return identifiers, np.asarray(rows, dtype=np.int64)


def aggregate_grass_counts(per_identifier: Mapping[str, Mapping[str, object]]) -> dict[str, int | float | None]:
    """Aggregate stored per-raster counts using the evaluation's global-IoU convention."""
    _, counts = _validated_counts(per_identifier)
    true_positive, false_positive, false_negative = (int(value) for value in counts.sum(axis=0))
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "grass_iou": grass_iou(true_positive, false_positive, false_negative),
    }


def _iou_samples(counts: np.ndarray) -> np.ndarray:
    if counts.ndim != 2 or counts.shape[1] != len(_COUNT_KEYS):
        raise ValueError("bootstrap counts must have shape (replicates, 3)")
    true_positive = counts[:, 0]
    unions = counts.sum(axis=1)
    values = np.full(len(counts), np.nan, dtype=np.float64)
    present = unions > 0
    values[present] = true_positive[present] / unions[present]
    return values


def paired_raster_bootstrap_difference(
    baseline_per_identifier: Mapping[str, Mapping[str, object]],
    model_per_identifier: Mapping[str, Mapping[str, object]],
    *,
    replicates: int = 2000,
    seed: int = 20260804,
) -> dict:
    """Estimate a paired percentile CI for model-minus-baseline global grass IoU.

    Each draw resamples the same complete Semantic Drone rasters for both arms,
    sums TP/FP/FN, and then computes global IoU. It is deliberately a raster
    bootstrap: no unsupported geographic independence claim is made.
    """
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    baseline_identifiers, baseline_counts = _validated_counts(baseline_per_identifier)
    model_identifiers, model_counts = _validated_counts(model_per_identifier)
    if model_identifiers != baseline_identifiers:
        raise ValueError("baseline and model raster identifier sets must match exactly")
    baseline_point = aggregate_grass_counts(baseline_per_identifier)
    model_point = aggregate_grass_counts(model_per_identifier)
    if baseline_point["grass_iou"] is None or model_point["grass_iou"] is None:
        raise ValueError("global grass IoU is unavailable for a comparison arm")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(baseline_identifiers), size=(replicates, len(baseline_identifiers)))
    baseline_samples = _iou_samples(baseline_counts[draws].sum(axis=1))
    model_samples = _iou_samples(model_counts[draws].sum(axis=1))
    differences = model_samples - baseline_samples
    finite = differences[np.isfinite(differences)]
    if not finite.size:
        raise ValueError("bootstrap produced no finite global grass IoU differences")
    lower, upper = np.quantile(finite, (0.025, 0.975))
    return {
        "comparison": "model_minus_baseline",
        "resampling_unit": "semantic_drone_raster",
        "raster_count": len(baseline_identifiers),
        "raster_identifier_set_sha256": identifier_set_sha256(baseline_identifiers),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
        "baseline": baseline_point,
        "model": model_point,
        "point_difference": float(model_point["grass_iou"] - baseline_point["grass_iou"]),
        "confidence_interval": {"lower": float(lower), "upper": float(upper)},
    }


def paired_raster_bootstrap_mean_difference(
    baseline_per_identifier: Sequence[Mapping[str, Mapping[str, object]]],
    model_per_identifier: Sequence[Mapping[str, Mapping[str, object]]],
    *,
    replicates: int = 2000,
    seed: int = 20260804,
) -> dict:
    """Bootstrap the mean model-minus-baseline difference across paired seeds."""
    if len(baseline_per_identifier) != len(model_per_identifier) or not baseline_per_identifier:
        raise ValueError("baseline and model must contain the same non-empty seed count")
    validated_baselines = []
    validated_models = []
    identifiers = None
    for baseline, model in zip(baseline_per_identifier, model_per_identifier):
        baseline_ids, baseline_counts = _validated_counts(baseline)
        model_ids, model_counts = _validated_counts(model)
        if baseline_ids != model_ids:
            raise ValueError("baseline and model raster identifier sets must match exactly")
        if identifiers is None:
            identifiers = baseline_ids
        elif identifiers != baseline_ids:
            raise ValueError("all seeds must share one raster identifier set")
        validated_baselines.append(baseline_counts)
        validated_models.append(model_counts)
    assert identifiers is not None
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    baseline_points = [aggregate_grass_counts(values) for values in baseline_per_identifier]
    model_points = [aggregate_grass_counts(values) for values in model_per_identifier]
    point_difference = float(np.mean([item["grass_iou"] for item in model_points])) - float(
        np.mean([item["grass_iou"] for item in baseline_points])
    )
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(identifiers), size=(replicates, len(identifiers)))
    baseline_samples = np.mean(
        [_iou_samples(counts[draws].sum(axis=1)) for counts in validated_baselines], axis=0
    )
    model_samples = np.mean(
        [_iou_samples(counts[draws].sum(axis=1)) for counts in validated_models], axis=0
    )
    differences = model_samples - baseline_samples
    finite = differences[np.isfinite(differences)]
    if not finite.size:
        raise ValueError("bootstrap produced no finite global grass IoU differences")
    lower, upper = np.quantile(finite, (0.025, 0.975))
    return {
        "comparison": "mean_model_minus_mean_baseline",
        "resampling_unit": "semantic_drone_raster",
        "raster_count": len(identifiers),
        "model_seed_count": len(model_per_identifier),
        "baseline_seed_count": len(baseline_per_identifier),
        "raster_identifier_set_sha256": identifier_set_sha256(identifiers),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
        "baseline_per_seed": baseline_points,
        "model_per_seed": model_points,
        "point_difference": point_difference,
        "confidence_interval": {"lower": float(lower), "upper": float(upper)},
    }
