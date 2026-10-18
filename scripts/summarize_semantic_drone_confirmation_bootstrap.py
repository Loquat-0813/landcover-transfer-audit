"""Summarize the frozen E0/B0/B1 Semantic Drone confirmation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1eval.semantic_drone_bootstrap import (
    aggregate_grass_counts,
    exclude_identifiers,
    paired_raster_bootstrap_mean_difference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0-result-dir", type=Path, required=True)
    parser.add_argument("--b0-result-dir", type=Path, action="append", required=True)
    parser.add_argument("--b1-result-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--exclude-identifier", action="append", default=[])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    return parser.parse_args()


def _load(directory: Path, arm: str) -> tuple[dict, dict[str, dict[str, object]]]:
    report_path = directory / "metrics.json"
    per_identifier_path = directory / "per_identifier.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_identifier = json.loads(per_identifier_path.read_text(encoding="utf-8"))
    config = report.get("run_config")
    if not isinstance(config, dict) or config.get("experiment") != f"semantic_drone_confirmation_{arm}":
        raise ValueError(f"{directory}: expected Semantic Drone {arm} confirmation output")
    if config.get("target_leaf") != "herbaceous_vegetation" or config.get("target_raw_class") != "grass":
        raise ValueError(f"{directory}: target mapping is not the frozen grass-to-herbaceous evaluation")
    if not isinstance(per_identifier, dict):
        raise ValueError(f"{directory}: per_identifier.json must contain an object")
    aggregate = aggregate_grass_counts(per_identifier)
    reported = report.get("grass")
    if not isinstance(reported, dict) or any(reported.get(key) != aggregate[key] for key in aggregate):
        raise ValueError(f"{directory}: aggregate per-raster counts do not match metrics.json")
    if config.get("evaluated_pairs") != len(per_identifier):
        raise ValueError(f"{directory}: evaluated pair count does not match per-raster results")
    return report, per_identifier


def _seed_summary(values: list[dict[str, int | float | None]]) -> dict:
    """Keep global-count results per checkpoint and make their descriptive mean explicit."""
    return {
        "per_seed": values,
        "mean_grass_iou": statistics.mean(float(value["grass_iou"]) for value in values),
        "grass_iou_sample_std": statistics.stdev(float(value["grass_iou"]) for value in values),
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    if args.output_json.exists():
        raise SystemExit(f"output already exists: {args.output_json}")
    if len(args.b0_result_dir) != 3 or len(args.b1_result_dir) != 3:
        raise SystemExit("Semantic Drone bootstrap requires exactly three B0 and three B1 result directories")
    e0_report, e0 = _load(args.e0_result_dir, "e0")
    b0_loaded = [_load(directory, "b0") for directory in args.b0_result_dir]
    b1_loaded = [_load(directory, "b1") for directory in args.b1_result_dir]
    b0 = [item[1] for item in b0_loaded]
    b1 = [item[1] for item in b1_loaded]
    manifests = {e0_report["run_config"].get("manifest_sha256")}
    manifests.update(report["run_config"].get("manifest_sha256") for report, _ in (*b0_loaded, *b1_loaded))
    if len(manifests) != 1 or None in manifests:
        raise ValueError("E0/B0/B1 outputs do not share one frozen target manifest")
    pair_counts = {e0_report["run_config"].get("evaluated_pairs")}
    pair_counts.update(report["run_config"].get("evaluated_pairs") for report, _ in (*b0_loaded, *b1_loaded))
    if len(pair_counts) != 1:
        raise ValueError("E0/B0/B1 evaluated pair counts do not match")
    excluded = tuple(sorted(set(args.exclude_identifier)))
    if excluded:
        e0 = exclude_identifiers(e0, excluded)
        b0 = [exclude_identifiers(value, excluded) for value in b0]
        b1 = [exclude_identifiers(value, excluded) for value in b1]
    comparisons = {
        "paired_raster_bootstrap_b0_minus_e0": paired_raster_bootstrap_mean_difference(
            [e0] * len(b0), b0, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
        ),
        "paired_raster_bootstrap_b1_minus_b0": paired_raster_bootstrap_mean_difference(
            b0, b1, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
        ),
        "paired_raster_bootstrap_b1_minus_e0": paired_raster_bootstrap_mean_difference(
            [e0] * len(b1), b1, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
        ),
    }
    summary = {
        "version": "semantic-drone-confirmation-bootstrap-v2",
        "target": {
            "raw_class": "grass",
            "p1_leaf": "herbaceous_vegetation",
            "manifest_sha256": manifests.pop(),
            "manifest_evaluated_pairs": pair_counts.pop(),
            "evaluated_pairs": len(e0),
            "excluded_identifiers": list(excluded),
        },
        "arms": {
            "e0": aggregate_grass_counts(e0),
            "b0": _seed_summary([aggregate_grass_counts(value) for value in b0]),
            "b1": _seed_summary([aggregate_grass_counts(value) for value in b1]),
        },
        "comparisons": comparisons,
        "source_result_directories": {
            "e0": str(args.e0_result_dir),
            "b0": [str(value) for value in args.b0_result_dir],
            "b1": [str(value) for value in args.b1_result_dir],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    b0_e0 = comparisons["paired_raster_bootstrap_b0_minus_e0"]
    b1_b0 = comparisons["paired_raster_bootstrap_b1_minus_b0"]
    print(
        "semantic_drone_confirmation_bootstrap_complete:",
        f"e0_grass_iou={summary['arms']['e0']['grass_iou']:.6f}",
        f"b0_mean_grass_iou={summary['arms']['b0']['mean_grass_iou']:.6f}",
        f"b1_mean_grass_iou={summary['arms']['b1']['mean_grass_iou']:.6f}",
        f"b0_minus_e0_ci=[{b0_e0['confidence_interval']['lower']:.6f},{b0_e0['confidence_interval']['upper']:.6f}]",
        f"b1_minus_b0_ci=[{b1_b0['confidence_interval']['lower']:.6f},{b1_b0['confidence_interval']['upper']:.6f}]",
        f"output={args.output_json}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
