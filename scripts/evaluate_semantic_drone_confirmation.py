"""Evaluate fixed E0/B0/B1 checkpoints on the frozen Semantic Drone target."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1eval.prompts import load_leaf_prompts
from p1eval.semantic_drone_metrics import grass_iou
from p1eval.taxonomy import load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.loveda_e1 import base_leaf_logits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path, arm: str, prompts: tuple[str, ...]) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config")
    if not isinstance(config, dict) or config.get("experiment") != f"flair_coverage_{arm}":
        raise ValueError(f"checkpoint is not a completed FLAIR {arm} run: {path}")
    if config.get("protocol_mode") != "fixed_budget_primary":
        raise ValueError("confirmation requires a primary fixed-budget checkpoint")
    if tuple(config.get("prompts", ())) != prompts:
        raise ValueError("checkpoint prompts do not match the frozen prompt configuration")
    if "adapter_state_dict" not in checkpoint:
        raise ValueError("checkpoint has no adapter state dictionary")
    return config, checkpoint


def _load_example(image_path: Path, mask_path: Path, grass_rgb: tuple[int, int, int]) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(image_path) as image_file:
        image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as mask_file:
        mask = np.asarray(mask_file.convert("RGB"), dtype=np.uint8)
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"image/mask shape mismatch: {image_path} {image.shape} vs {mask_path} {mask.shape}")
    target = np.all(mask == np.asarray(grass_rgb, dtype=np.uint8), axis=2)
    image_tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True, order="C")).permute(2, 0, 1).float().div_(255.0)
    return image_tensor, target


@torch.inference_mode()
def _evaluate_pairs(
    manifest: dict,
    extracted_root: Path,
    backbone: FrozenCLIPPatchEncoder,
    adapter: LowRankSemanticCalibration | None,
    text_features: torch.Tensor,
    grass_leaf_index: int,
    device: torch.device,
    tile_size: int,
    max_samples: int | None,
) -> tuple[dict, dict[str, dict[str, int | float | None]]]:
    if tile_size < 1 or tile_size % backbone.patch_size:
        raise ValueError("tile_size must be a positive multiple of the CLIP patch size")
    records = manifest["pairs"] if max_samples is None else manifest["pairs"][:max_samples]
    if not records:
        raise ValueError("manifest selected no Semantic Drone records")
    grass_rgb = tuple(int(value) for value in manifest["mapping"]["grass_rgb"])
    total = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    per_identifier: dict[str, dict[str, int | float | None]] = {}
    for record in records:
        image, target = _load_example(
            extracted_root / record["image"],
            extracted_root / record["mask"],
            grass_rgb,
        )
        height, width = target.shape
        predicted_grass = np.zeros((height, width), dtype=bool)
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                tile = torch.zeros((1, 3, tile_size, tile_size), dtype=image.dtype)
                tile[0, :, : bottom - top, : right - left] = image[:, top:bottom, left:right]
                features = backbone.encode_image(tile.to(device))
                logits = base_leaf_logits(features, text_features) if adapter is None else adapter(features, text_features)
                logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                tile_prediction = logits.argmax(dim=1)[0].eq(grass_leaf_index).cpu().numpy()
                predicted_grass[top:bottom, left:right] = tile_prediction[: bottom - top, : right - left]
        true_positive = int(np.logical_and(target, predicted_grass).sum())
        false_positive = int(np.logical_and(~target, predicted_grass).sum())
        false_negative = int(np.logical_and(target, ~predicted_grass).sum())
        total["true_positive"] += true_positive
        total["false_positive"] += false_positive
        total["false_negative"] += false_negative
        per_identifier[record["identifier"]] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "grass_iou": grass_iou(true_positive, false_positive, false_negative),
        }
        print(f"evaluated={len(per_identifier)}/{len(records)} identifier={record['identifier']}", flush=True)
    total["grass_iou"] = grass_iou(total["true_positive"], total["false_positive"], total["false_negative"])
    return total, per_identifier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e0", "b0", "b1"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--prompt-config", type=Path, default=ROOT / "configs" / "taxonomy_prompts_v0.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.experiment == "e0" and args.adapter_checkpoint is not None:
        raise SystemExit("E0 does not accept --adapter-checkpoint")
    if args.experiment in {"b0", "b1"} and args.adapter_checkpoint is None:
        raise SystemExit("B0/B1 require --adapter-checkpoint")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("version") != "semantic-drone-confirmation-manifest-v1":
        raise ValueError("manifest is not the frozen Semantic Drone confirmation manifest")
    if not manifest.get("frozen_before_inference") or manifest.get("target_metrics_inspected"):
        raise ValueError("manifest does not prove that target metrics were unopened before inference")
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    leaves = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(args.prompt_config, leaves.leaf_names)
    grass_leaf_index = leaves.leaf_names.index("herbaceous_vegetation")
    checkpoint_config = None
    checkpoint = None
    if args.experiment in {"b0", "b1"}:
        checkpoint_config, checkpoint = _load_checkpoint(args.adapter_checkpoint, args.experiment, prompts)
    run_config = {
        "experiment": f"semantic_drone_confirmation_{args.experiment}",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "model_id": args.model_id,
        "adapter_checkpoint": None if args.adapter_checkpoint is None else str(args.adapter_checkpoint),
        "adapter_checkpoint_sha256": None if args.adapter_checkpoint is None else _sha256(args.adapter_checkpoint),
        "checkpoint_run_config": checkpoint_config,
        "extracted_root": str(args.extracted_root),
        "tile_size": args.tile_size,
        "evaluated_pairs": manifest["pair_count"] if args.max_samples is None else min(args.max_samples, manifest["pair_count"]),
        "target_leaf": "herbaceous_vegetation",
        "target_raw_class": "grass",
        "target_metrics_used_for_method_selection": False,
    }
    if not torch.cuda.is_available():
        raise SystemExit("Semantic Drone confirmation inference is restricted to CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    adapter = None
    if checkpoint is not None:
        adapter = LowRankSemanticCalibration(backbone.embedding_dim, rank=int(checkpoint_config["rank"])).to(device)
        adapter.load_state_dict(checkpoint["adapter_state_dict"])
        adapter.eval()
        print(f"loaded_adapter_checkpoint={args.adapter_checkpoint}", flush=True)
    total, per_identifier = _evaluate_pairs(
        manifest,
        args.extracted_root,
        backbone,
        adapter,
        text_features,
        grass_leaf_index,
        device,
        args.tile_size,
        args.max_samples,
    )
    report = {"run_config": run_config, "grass": total}
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.output_dir / "per_identifier.json").write_text(json.dumps(per_identifier), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "semantic_drone_confirmation_complete:",
        f"experiment={args.experiment}",
        f"grass_iou={total['grass_iou']}",
        f"evaluated_pairs={run_config['evaluated_pairs']}",
        f"output={args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
