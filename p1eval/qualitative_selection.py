"""Deterministic post-hoc selection of illustrative OEM qualitative examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _matrix(value: Sequence[Sequence[int]], class_count: int, identifier: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.int64)
    if matrix.shape != (class_count, class_count) or np.any(matrix < 0):
        raise ValueError(f"{identifier}: expected a non-negative {class_count}x{class_count} confusion matrix")
    return matrix


def _iou(matrix: np.ndarray, class_index: int) -> float:
    true_positive = int(matrix[class_index, class_index])
    reference = int(matrix[class_index, :].sum())
    predicted = int(matrix[:, class_index].sum())
    union = reference + predicted - true_positive
    return true_positive / union if union else float("nan")


def _rank(
    rows: list[dict[str, object]],
    *,
    exclude: set[str],
) -> dict[str, object]:
    eligible = [row for row in rows if str(row["identifier"]) not in exclude]
    if not eligible:
        raise ValueError("no eligible qualitative identifier remains after enforcing unique selections")
    return sorted(eligible, key=lambda row: (-float(row["score"]), str(row["identifier"])))[0]


def select_oem_qualitative_identifiers(
    b0_confusions: Mapping[str, Sequence[Sequence[int]]],
    b1_confusions: Mapping[str, Sequence[Sequence[int]]],
    class_names: Sequence[str],
    *,
    min_reference_pixels: int = 10_000,
) -> dict[str, object]:
    """Select two illustration-only rasters from frozen per-raster confusions.

    The first maximizes per-raster rangeland IoU gain and the second maximizes
    cropland IoU decrease. This outcome-aware selection is explicitly for a
    qualitative supplement; it is not used in any point estimate or interval.
    """
    names = tuple(class_names)
    if len(names) != len(set(names)):
        raise ValueError("class names must be unique")
    try:
        herbaceous_index = names.index("herbaceous_vegetation")
        cropland_index = names.index("cropland")
    except ValueError as exc:
        raise ValueError("class names must include herbaceous_vegetation and cropland") from exc
    if min_reference_pixels < 1:
        raise ValueError("min_reference_pixels must be positive")
    if set(b0_confusions) != set(b1_confusions):
        raise ValueError("B0 and B1 qualitative inputs must contain the same identifiers")

    recovery_rows: list[dict[str, object]] = []
    cropland_rows: list[dict[str, object]] = []
    for identifier in sorted(b0_confusions):
        b0 = _matrix(b0_confusions[identifier], len(names), identifier)
        b1 = _matrix(b1_confusions[identifier], len(names), identifier)
        herbaceous_reference = int(b0[herbaceous_index, :].sum())
        cropland_reference = int(b0[cropland_index, :].sum())
        if herbaceous_reference >= min_reference_pixels:
            b0_iou = _iou(b0, herbaceous_index)
            b1_iou = _iou(b1, herbaceous_index)
            recovery_rows.append(
                {
                    "identifier": identifier,
                    "score": b1_iou - b0_iou,
                    "b0_iou": b0_iou,
                    "b1_iou": b1_iou,
                    "reference_pixels": herbaceous_reference,
                }
            )
        if cropland_reference >= min_reference_pixels:
            b0_iou = _iou(b0, cropland_index)
            b1_iou = _iou(b1, cropland_index)
            cropland_rows.append(
                {
                    "identifier": identifier,
                    "score": b0_iou - b1_iou,
                    "b0_iou": b0_iou,
                    "b1_iou": b1_iou,
                    "reference_pixels": cropland_reference,
                }
            )

    recovery = _rank(recovery_rows, exclude=set())
    cropland = _rank(cropland_rows, exclude={str(recovery["identifier"])})
    return {
        "selection_status": "post_hoc_illustrative_only",
        "selection_rule": {
            "minimum_reference_pixels": min_reference_pixels,
            "rangeland_recovery": "maximum per-raster B1 minus B0 herbaceous IoU",
            "cropland_tradeoff": "maximum per-raster B0 minus B1 cropland IoU, excluding the recovery raster",
        },
        "selected": {"rangeland_recovery": recovery, "cropland_tradeoff": cropland},
    }
