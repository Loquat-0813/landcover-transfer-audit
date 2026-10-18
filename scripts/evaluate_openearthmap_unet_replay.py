"""Replay one frozen U-Net checkpoint on the Paris-excluded OpenEarthMap target."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

# PyTorch 2.6 serializes ``torch.__version__`` as TorchVersion when an older
# run manifest is embedded in the checkpoint. These checkpoints are generated
# by this trusted local protocol; allowlist that metadata class while keeping
# weights_only=True for all other globals.
try:
    from torch.torch_version import TorchVersion

    torch.serialization.add_safe_globals([TorchVersion])
except (ImportError, AttributeError):  # pragma: no cover - older torch fallback.
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1data.external import discover_external_pairs
from p1data.openearthmap_city_split import city_from_openearthmap_identifier, exclude_openearthmap_cities
from p1data.splits import identifier_set_sha256
from p1data.torch_dataset import TaxonomyRasterDataset
from p1eval.b2_bootstrap import city_block_bootstrap, city_confusions_from_identifier_confusions
from p1eval.metrics import stable_node_index
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.taxonomy import build_taxonomy_leaf_index
from p1model.unet import ResNet34UNet
from p1train.unet_protocol import evaluate_unet_tiled_by_identifier


REQUIRED_EXCLUDED_CITIES = ("paris",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("unet_b0", "unet_b0_half", "unet_b1", "unet_b1_shuffle"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "openearthmap")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--exclude-city", action="append", default=[])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--export-predictions",
        action="store_true",
        help="Save per-raster taxonomy predictions and targets as compressed NPZ files for spatial-null audits.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_config(path: Path, arm: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config")
    if not isinstance(config, dict) or config.get("experiment") != f"unet_coverage_{arm}":
        raise ValueError("checkpoint does not belong to the requested U-Net arm")
    if config.get("protocol_id") != "p1-unet-cross-family-v1":
        raise ValueError("checkpoint was not created under the frozen U-Net protocol")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint has no U-Net state dictionary")
    return config


def main() -> int:
    args = parse_args()
    if args.tile_size < 32 or args.tile_size % 32:
        raise SystemExit("--tile-size must be a positive multiple of 32")
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    if tuple(sorted(set(args.exclude_city))) != REQUIRED_EXCLUDED_CITIES:
        raise SystemExit("U-Net replay requires exactly --exclude-city paris")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")
    config = _checkpoint_config(args.checkpoint, args.arm)
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "openearthmap", taxonomy)
    discovery = discover_external_pairs("openearthmap", args.data_root)
    pairs = exclude_openearthmap_cities(discovery.pairs, REQUIRED_EXCLUDED_CITIES)
    pairs = pairs if args.max_samples is None else pairs[: args.max_samples]
    target_hash = identifier_set_sha256(pair.identifier for pair in pairs)
    run_config = {
        "experiment": f"unet_openearthmap_{args.arm}_replay",
        "protocol_id": "p1-unet-cross-family-v1",
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "source_checkpoint_run_config": config,
        "excluded_cities": list(REQUIRED_EXCLUDED_CITIES),
        "paired_raster_count": len(discovery.pairs),
        "evaluated_pairs": len(pairs),
        "evaluated_pair_identifier_set_sha256": target_hash,
        "evaluated_city_count": len({city_from_openearthmap_identifier(pair.identifier) for pair in pairs}),
        "tile_size": args.tile_size,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "prediction_export": bool(args.export_predictions),
    }
    if args.dry_run:
        print("unet_openearthmap_replay_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("full OpenEarthMap U-Net replay is restricted to CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset = TaxonomyRasterDataset(pairs, mapping, node_to_id, crop_size=None, seed=int(config["seed"]))
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    model = ResNet34UNet(class_count=len(leaves.leaf_names), pretrained_encoder=False).to("cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    prediction_dir = args.output_dir / "predictions"
    prediction_manifest: dict[str, dict[str, str]] = {}
    if args.export_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=False)

    def export_prediction(identifier: str, target_leaf: torch.Tensor, prediction: torch.Tensor) -> None:
        if not args.export_predictions:
            return
        # A hash avoids path traversal and keeps identifiers reversible through the manifest.
        token = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:24]
        filename = f"{token}.npz"
        target_np = target_leaf.numpy().astype(np.int8, copy=False)
        prediction_np = prediction.numpy().astype(np.int8, copy=False)
        valid_np = target_np >= 0
        np.savez_compressed(prediction_dir / filename, target_leaf=target_np, prediction=prediction_np, valid=valid_np)
        prediction_manifest[identifier] = {"file": filename, "city": city_from_openearthmap_identifier(identifier)}

    result, per_identifier = evaluate_unet_tiled_by_identifier(
        dataset,
        model,
        torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], device="cuda"),
        leaves.leaf_names,
        torch.device("cuda"),
        args.tile_size,
        prediction_callback=export_prediction if args.export_predictions else None,
    )
    identifier_confusions = {identifier: matrix.tolist() for identifier, matrix in per_identifier.items()}
    city_confusions = city_confusions_from_identifier_confusions(identifier_confusions, leaves.leaf_names)
    bootstrap = city_block_bootstrap(city_confusions, leaves.leaf_names, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed)
    if not np.isclose(result.mean_iou, bootstrap["point"]["mean_iou"], rtol=0.0, atol=1e-12):
        raise RuntimeError("city aggregation does not reproduce all-raster mIoU")
    if args.export_predictions:
        (prediction_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "p1-unet-prediction-export-v1",
                    "class_names": list(leaves.leaf_names),
                    "identifiers": prediction_manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.output_dir / "per_raster_confusions.json").write_text(
        json.dumps({"class_names": list(leaves.leaf_names), "confusions": identifier_confusions}), encoding="utf-8"
    )
    (args.output_dir / "city_confusions.json").write_text(
        json.dumps({"class_names": list(leaves.leaf_names), "city_confusions": {city: matrix.tolist() for city, matrix in city_confusions.items()}}),
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"run_config": run_config, "exact_leaf": {"mean_iou": result.mean_iou, "per_class_iou": result.per_class_iou, "valid_pixels": result.valid_pixels}, "city_bootstrap": bootstrap}, indent=2),
        encoding="utf-8",
    )
    print(f"unet_openearthmap_replay_complete: arm={args.arm} rangeland_iou={result.per_class_iou['herbaceous_vegetation']} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
