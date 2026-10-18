"""Summarize the frozen U-Net OpenEarthMap replay family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1eval.b2_bootstrap import paired_city_bootstrap_mean_difference
from p1eval.supported_leaf_bootstrap import (
    paired_city_bootstrap_mean_supported_leaf_difference,
    supported_leaf_mean_iou,
)


ARMS = ("unet_b0", "unet_b0_half", "unet_b1", "unet_b1_shuffle")
SEEDS = (19, 37, 73)
TARGET_LEAF = "herbaceous_vegetation"
SUPPORTED_LEAVES = (
    "built_structure",
    "transport_surface",
    "woody_vegetation",
    "cropland",
    "bare_surface",
    "surface_water",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    return parser.parse_args()


def _load(output_root: Path, arm: str, seed: int) -> tuple[dict, dict, dict[str, np.ndarray]]:
    directory = output_root / f"unet_oem_{arm}_seed{seed}"
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    city_payload = json.loads((directory / "city_confusions.json").read_text(encoding="utf-8"))
    class_names = tuple(city_payload["class_names"])
    if TARGET_LEAF not in class_names or any(name not in class_names for name in SUPPORTED_LEAVES):
        raise ValueError(f"{directory}: unexpected class names {class_names}")
    city_confusions = {
        city: np.asarray(matrix, dtype=np.int64)
        for city, matrix in city_payload["city_confusions"].items()
    }
    if len(city_confusions) != 74:
        raise ValueError(f"{directory}: expected 74 retained cities, found {len(city_confusions)}")
    return metrics, {"class_names": class_names}, city_confusions


def _point_metrics(metrics: dict, city_confusions: dict[str, np.ndarray], class_names: tuple[str, ...]) -> dict[str, float]:
    exact = metrics["exact_leaf"]
    supported = supported_leaf_mean_iou(
        np.sum(np.stack(tuple(city_confusions.values())), axis=0), class_names, SUPPORTED_LEAVES
    )
    return {
        "rangeland_iou": float(exact["per_class_iou"][TARGET_LEAF]),
        "supported_leaf_miou": float(supported),
        "overall_miou": float(exact["mean_iou"]),
    }


def _mean_sd(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(statistics.mean(row[metric] for row in rows)),
            "sd": float(statistics.stdev(row[metric] for row in rows)),
        }
        for metric in ("rangeland_iou", "supported_leaf_miou", "overall_miou")
    }


def _contrast(
    baseline: list[dict[str, np.ndarray]],
    model: list[dict[str, np.ndarray]],
    class_names: tuple[str, ...],
    replicates: int,
    seed: int,
) -> dict:
    overall = paired_city_bootstrap_mean_difference(
        baseline, model, class_names, replicates=replicates, seed=seed
    )
    supported = paired_city_bootstrap_mean_supported_leaf_difference(
        baseline, model, class_names, SUPPORTED_LEAVES, replicates=replicates, seed=seed
    )
    target_interval = overall["confidence_interval"]["per_class_iou"][TARGET_LEAF]
    return {
        "rangeland_iou": {
            "estimate": float(overall["point_difference"]["per_class_iou"][TARGET_LEAF]),
            "checkpoint_conditional_city_bootstrap_interval": target_interval,
        },
        "supported_leaf_miou": {
            "estimate": float(supported["point_difference"]),
            "checkpoint_conditional_city_bootstrap_interval": supported["confidence_interval"],
        },
        "overall_miou": {
            "estimate": float(overall["point_difference"]["mean_iou"]),
            "checkpoint_conditional_city_bootstrap_interval": overall["confidence_interval"]["mean_iou"],
        },
        "resampling_unit": "openearthmap_city",
        "checkpoint_conditional": True,
        "seed_count": 3,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def main() -> int:
    args = parse_args()
    loaded: dict[str, dict] = {}
    class_names: tuple[str, ...] | None = None
    for arm in ARMS:
        loaded[arm] = {"rows": [], "cities": []}
        for seed in SEEDS:
            metrics, metadata, cities = _load(args.output_root, arm, seed)
            current_names = metadata["class_names"]
            if class_names is None:
                class_names = current_names
            if current_names != class_names:
                raise ValueError("class names differ across replay outputs")
            loaded[arm]["rows"].append({"seed": seed, **_point_metrics(metrics, cities, class_names)})
            loaded[arm]["cities"].append(cities)

    assert class_names is not None
    contrasts = {
        "unet_b1_minus_unet_b0": _contrast(
            loaded["unet_b0"]["cities"], loaded["unet_b1"]["cities"], class_names, args.bootstrap_replicates, args.bootstrap_seed
        ),
        "unet_b1_minus_unet_b0_half": _contrast(
            loaded["unet_b0_half"]["cities"], loaded["unet_b1"]["cities"], class_names, args.bootstrap_replicates, args.bootstrap_seed
        ),
        "unet_b1_shuffle_minus_unet_b0_half": _contrast(
            loaded["unet_b0_half"]["cities"], loaded["unet_b1_shuffle"]["cities"], class_names, args.bootstrap_replicates, args.bootstrap_seed
        ),
        "unet_b1_minus_unet_b1_shuffle": _contrast(
            loaded["unet_b1_shuffle"]["cities"], loaded["unet_b1"]["cities"], class_names, args.bootstrap_replicates, args.bootstrap_seed
        ),
    }
    report = {
        "protocol_id": "p1-unet-cross-family-v1",
        "target_population": "OpenEarthMap Paris-excluded 74-city replay",
        "target_leaf": TARGET_LEAF,
        "supported_leaves": list(SUPPORTED_LEAVES),
        "class_names": list(class_names),
        "seed_metrics": {arm: loaded[arm]["rows"] for arm in ARMS},
        "mean_sd": {arm: _mean_sd(loaded[arm]["rows"]) for arm in ARMS},
        "contrasts": contrasts,
        "inference_note": "Intervals are checkpoint-conditional paired city-bootstrap intervals and do not quantify training-run randomness.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"unet_cross_family_summary_complete: output={args.output}")
    for name, contrast in contrasts.items():
        target = contrast["rangeland_iou"]
        print(f"{name}: rangeland={target['estimate']:.6f} ci=[{target['checkpoint_conditional_city_bootstrap_interval']['lower']:.6f},{target['checkpoint_conditional_city_bootstrap_interval']['upper']:.6f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
