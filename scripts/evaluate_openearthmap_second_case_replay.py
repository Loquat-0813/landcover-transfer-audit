"""Evaluate second-case checkpoints on the frozen Paris-excluded OEM target."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1data.external import discover_external_pairs
from p1data.openearthmap_city_split import city_from_openearthmap_identifier, exclude_openearthmap_cities
from p1data.splits import identifier_set_sha256
from p1data.torch_dataset import TaxonomyRasterDataset
from p1eval.b2_bootstrap import city_block_bootstrap, city_confusions_from_identifier_confusions
from p1eval.metrics import stable_node_index
from p1eval.prompts import load_leaf_prompts
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.loveda_e1 import evaluate_taxonomy_tiled_by_identifier


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("sd_b0", "sd_b0_half", "sd_b1", "sd_b1_shuffle"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    parser.add_argument("--exclude-city", action="append", default=[])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("second-case replay is restricted to CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")
    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if source_audit.get("training_admission") != "source_audit_ready":
        raise SystemExit("source audit has not passed geographic-overlap admission")
    required_exclusions = tuple(sorted(source_audit.get("geographic_overlap_review", {}).get("target_exclusions", ())))
    requested_exclusions = tuple(sorted(set(args.exclude_city)))
    if not required_exclusions:
        raise SystemExit("approved source audit has no frozen target-exclusion set")
    if requested_exclusions != required_exclusions:
        raise SystemExit(f"second-case replay requires exactly --exclude-city values {required_exclusions}")
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config", {})
    if config.get("case") != "semantic_drone_loveda_to_openearthmap_cropland" or config.get("arm") != args.arm:
        raise SystemExit("checkpoint is not a second-case checkpoint for the requested arm")
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "openearthmap", taxonomy)
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(ROOT / "configs" / "taxonomy_prompts_v0.yaml", leaves.leaf_names)
    if tuple(config.get("prompts", ())) != prompts:
        raise SystemExit("checkpoint prompts differ from frozen prompts")
    discovery = discover_external_pairs("openearthmap", args.data_root)
    pairs = exclude_openearthmap_cities(discovery.pairs, required_exclusions)
    if not pairs:
        raise SystemExit("city exclusion removed all target rasters")
    run_config = {
        "experiment": f"semantic_drone_loveda_openearthmap_{args.arm}_replay",
        "case": config["case"],
        "arm": args.arm,
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "adapter_checkpoint_sha256": _sha256(args.adapter_checkpoint),
        "source_audit": str(args.source_audit),
        "source_audit_sha256": _sha256(args.source_audit),
        "excluded_cities": list(required_exclusions),
        "evaluated_pairs": len(pairs),
        "evaluated_pair_identifier_set_sha256": identifier_set_sha256(pair.identifier for pair in pairs),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    dataset = TaxonomyRasterDataset(tuple(pairs), mapping, node_to_id, crop_size=None, seed=19)
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    descendant_mask = torch.from_numpy(leaves.descendant_leaf_mask).to(device=device)
    leaf_node_ids = torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], device=device)
    adapter = LowRankSemanticCalibration(backbone.embedding_dim, rank=int(config["rank"])).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    result, per_identifier = evaluate_taxonomy_tiled_by_identifier(dataset, backbone, adapter, text_features, descendant_mask, leaf_node_ids, leaves.leaf_names, device, args.tile_size)
    identifier_confusions = {identifier: matrix.tolist() for identifier, matrix in per_identifier.items()}
    city_confusions = city_confusions_from_identifier_confusions(identifier_confusions, leaves.leaf_names)
    bootstrap = city_block_bootstrap(city_confusions, leaves.leaf_names, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed)
    report = {"run_config": run_config, "exact_leaf": {"mean_iou": result.mean_iou, "per_class_iou": result.per_class_iou, "valid_pixels": result.valid_pixels}, "city_bootstrap": bootstrap, "city_count": len(city_confusions), "cities": sorted(city_confusions)}
    (args.output_dir / "per_raster_confusions.json").write_text(json.dumps({"class_names": list(leaves.leaf_names), "confusions": identifier_confusions}), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("second_case_replay_complete:", f"arm={args.arm}", f"cropland_iou={result.per_class_iou.get('cropland')}", f"output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
