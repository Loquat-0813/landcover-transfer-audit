"""Deterministic city-block bootstrap summaries for the OpenEarthMap B2 control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from p1data.openearthmap_city_split import city_from_openearthmap_identifier


def target_supported_metrics(confusion: np.ndarray, class_names: Sequence[str]) -> dict:
    """Match B2's exact-label IoU convention from an aggregated confusion matrix."""
    matrix = np.asarray(confusion, dtype=np.int64)
    class_names = tuple(class_names)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("confusion shape does not match class names")
    if np.any(matrix < 0):
        raise ValueError("confusion counts must be non-negative")
    per_class_iou: dict[str, float | None] = {}
    values: list[float] = []
    for index, name in enumerate(class_names):
        target_support = int(matrix[index, :].sum())
        if target_support == 0:
            per_class_iou[name] = None
            continue
        intersection = int(matrix[index, index])
        union = target_support + int(matrix[:, index].sum()) - intersection
        value = intersection / union
        per_class_iou[name] = value
        values.append(value)
    return {
        "mean_iou": float(np.mean(values)) if values else None,
        "per_class_iou": per_class_iou,
        "valid_pixels": int(matrix.sum()),
    }


def city_confusions_from_identifier_confusions(
    identifier_confusions: Mapping[str, Sequence[Sequence[int]]],
    class_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Aggregate retained per-raster confusion matrices into city spatial blocks."""
    class_count = len(tuple(class_names))
    city_confusions: dict[str, np.ndarray] = {}
    for identifier, raw_confusion in identifier_confusions.items():
        city = city_from_openearthmap_identifier(identifier)
        confusion = np.asarray(raw_confusion, dtype=np.int64)
        if confusion.shape != (class_count, class_count) or np.any(confusion < 0):
            raise ValueError(f"{identifier}: invalid confusion matrix")
        if city not in city_confusions:
            city_confusions[city] = np.zeros_like(confusion)
        city_confusions[city] += confusion
    if not city_confusions:
        raise ValueError("at least one city confusion is required")
    return city_confusions


def _bootstrap_metric_samples(confusions: np.ndarray, class_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Vectorize supported-leaf IoU over bootstrap confusion matrices."""
    if confusions.ndim != 3 or confusions.shape[1:] != (len(class_names), len(class_names)):
        raise ValueError("bootstrap confusions have an invalid shape")
    target_support = confusions.sum(axis=2)
    prediction_support = confusions.sum(axis=1)
    intersections = np.diagonal(confusions, axis1=1, axis2=2)
    unions = target_support + prediction_support - intersections
    with np.errstate(divide="ignore", invalid="ignore"):
        ious = intersections / unions
    ious[target_support == 0] = np.nan
    return {
        "mean_iou": np.nanmean(ious, axis=1),
        **{name: ious[:, index] for index, name in enumerate(class_names)},
    }


def _confidence_intervals(samples: dict[str, np.ndarray], point: dict, class_names: tuple[str, ...]) -> dict:
    def interval(values: np.ndarray, point_value: float | None) -> dict[str, float] | None:
        if point_value is None:
            return None
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        lower, upper = np.quantile(finite, (0.025, 0.975))
        return {"lower": float(lower), "upper": float(upper)}

    return {
        "mean_iou": interval(samples["mean_iou"], point["mean_iou"]),
        "per_class_iou": {
            name: interval(samples[name], point["per_class_iou"][name]) for name in class_names
        },
    }


def city_block_bootstrap(
    city_confusions: Mapping[str, np.ndarray],
    class_names: Sequence[str],
    *,
    replicates: int = 2000,
    seed: int = 20260803,
) -> dict:
    """Bootstrap complete cities with replacement for a deterministic 95% percentile CI."""
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    class_names = tuple(class_names)
    cities = tuple(sorted(city_confusions))
    matrices = np.stack([np.asarray(city_confusions[city], dtype=np.int64) for city in cities])
    if matrices.shape[1:] != (len(class_names), len(class_names)):
        raise ValueError("city confusion shape does not match class names")
    point_confusion = matrices.sum(axis=0)
    point = target_supported_metrics(point_confusion, class_names)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(cities), size=(replicates, len(cities)))
    samples = _bootstrap_metric_samples(matrices[draws].sum(axis=1), class_names)
    return {
        "resampling_unit": "openearthmap_city",
        "city_names": list(cities),
        "city_count": len(cities),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
        "point": point,
        "confidence_interval": _confidence_intervals(samples, point, class_names),
    }


def paired_city_bootstrap_mean_difference(
    baseline_city_confusions: Mapping[str, np.ndarray] | Sequence[Mapping[str, np.ndarray]],
    model_city_confusions: Sequence[Mapping[str, np.ndarray]],
    class_names: Sequence[str],
    *,
    replicates: int = 2000,
    seed: int = 20260803,
) -> dict:
    """Bootstrap the mean per-seed metric difference using shared city draws."""
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not model_city_confusions:
        raise ValueError("at least one model seed is required")
    class_names = tuple(class_names)
    baseline_arms = (
        (baseline_city_confusions,)
        if isinstance(baseline_city_confusions, Mapping)
        else tuple(baseline_city_confusions)
    )
    if not baseline_arms:
        raise ValueError("at least one baseline seed is required")
    cities = tuple(sorted(baseline_arms[0]))
    if not cities or any(tuple(sorted(confusions)) != cities for confusions in (*baseline_arms, *model_city_confusions)):
        raise ValueError("baseline and model city sets must match exactly")
    baselines = np.stack(
        [np.stack([np.asarray(confusions[city], dtype=np.int64) for city in cities]) for confusions in baseline_arms]
    )
    models = np.stack(
        [np.stack([np.asarray(confusions[city], dtype=np.int64) for city in cities]) for confusions in model_city_confusions]
    )
    if baselines.shape[2:] != (len(class_names), len(class_names)) or models.shape[2:] != baselines.shape[2:]:
        raise ValueError("city confusion shape does not match class names")
    baseline_points = [target_supported_metrics(baseline.sum(axis=0), class_names) for baseline in baselines]
    model_points = [target_supported_metrics(model.sum(axis=0), class_names) for model in models]
    point = {
        "mean_iou": float(np.mean([metrics["mean_iou"] for metrics in model_points]))
        - float(np.mean([metrics["mean_iou"] for metrics in baseline_points])),
        "per_class_iou": {
            name: float(np.mean([metrics["per_class_iou"][name] for metrics in model_points]))
            - float(np.mean([metrics["per_class_iou"][name] for metrics in baseline_points]))
            for name in class_names
        },
    }
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(cities), size=(replicates, len(cities)))
    baseline_samples = [
        _bootstrap_metric_samples(baseline[draws].sum(axis=1), class_names) for baseline in baselines
    ]
    model_samples = [
        _bootstrap_metric_samples(model[draws].sum(axis=1), class_names) for model in models
    ]
    samples = {
        metric: np.mean([sample[metric] for sample in model_samples], axis=0)
        - np.mean([sample[metric] for sample in baseline_samples], axis=0)
        for metric in ("mean_iou", *class_names)
    }
    ci_point = {"mean_iou": point["mean_iou"], "per_class_iou": point["per_class_iou"]}
    return {
        "comparison": "mean_model_minus_mean_baseline",
        "resampling_unit": "openearthmap_city",
        "city_names": list(cities),
        "city_count": len(cities),
        "model_seed_count": len(model_city_confusions),
        "baseline_seed_count": len(baseline_arms),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
        "point_difference": point,
        "confidence_interval": _confidence_intervals(samples, ci_point, class_names),
    }
