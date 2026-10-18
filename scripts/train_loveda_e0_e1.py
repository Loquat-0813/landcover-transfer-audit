"""Run frozen-CLIP E0 or low-rank-calibrated E1 on the LoveDA protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1data.torch_dataset import LoveDATaxonomyDataset
from p1eval.metrics import stable_node_index
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.loveda_e1 import evaluate_loveda_tiled, source_supervised_leaf_mask, train_calibration_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e0", "e1", "e1_gated", "e2_fixedscale_gated"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "loveda")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adapter-checkpoint", type=Path, help="Evaluate an existing E1 or E1-gated adapter without retraining it.")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Validate data/protocol without downloading CLIP weights or training.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_leaf_prompts(leaf_names: tuple[str, ...]) -> tuple[str, ...]:
    document = yaml.safe_load((ROOT / "configs" / "taxonomy_prompts_v0.yaml").read_text(encoding="utf-8"))
    configured = document["leaf_prompts"]
    if set(configured) != set(leaf_names):
        raise ValueError("prompt configuration must contain every and only taxonomy leaf")
    return tuple(configured[name] for name in leaf_names)


def metric_dict(result) -> dict:
    return {
        "mean_iou": result.mean_iou,
        "per_class_iou": result.per_class_iou,
        "valid_pixels": result.valid_pixels,
    }


def main() -> int:
    args = parse_args()
    if args.crop_size < 1 or args.tile_size < 1:
        raise SystemExit("crop and tile sizes must be positive")
    if args.batch_size < 1 or args.epochs < 1:
        raise SystemExit("batch size and epochs must be positive")
    if args.adapter_checkpoint is not None and args.experiment not in {"e1", "e1_gated", "e2_fixedscale_gated"}:
        raise SystemExit("--adapter-checkpoint is valid only for E1, E1-gated, or E2-fixedscale-gated")
    if args.experiment == "e0" and args.epochs != 20:
        print("note: --epochs is ignored by E0 because it has no trainable parameters")

    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "loveda", taxonomy)
    node_to_id = stable_node_index(taxonomy.nodes)
    leaf_index = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(leaf_index.leaf_names)
    gated_mask = (
        source_supervised_leaf_mask(mapping, leaf_index.leaf_names)
        if args.experiment in {"e1_gated", "e2_fixedscale_gated"}
        else None
    )
    freeze_logit_scale = args.experiment == "e2_fixedscale_gated"
    adapted_leaf_names = list(leaf_index.leaf_names) if gated_mask is None else [
        name for name, adapted in zip(leaf_index.leaf_names, gated_mask.tolist()) if adapted
    ]
    frozen_leaf_names = [] if gated_mask is None else [
        name for name, adapted in zip(leaf_index.leaf_names, gated_mask.tolist()) if not adapted
    ]
    train_dataset = LoveDATaxonomyDataset(
        args.data_root, "train", mapping, node_to_id, args.crop_size, args.seed, args.max_train_samples
    )
    val_dataset = LoveDATaxonomyDataset(
        args.data_root, "val", mapping, node_to_id, None, args.seed, args.max_val_samples
    )
    run_config = {
        "experiment": args.experiment,
        "seed": args.seed,
        "model_id": args.model_id,
        "taxonomy": taxonomy.version,
        "mapping": "dataset-label-mappings-v0",
        "train_pairs": len(train_dataset),
        "validation_pairs": len(val_dataset),
        "crop_size": args.crop_size,
        "tile_size": args.tile_size,
        "rank": args.rank,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adapter_checkpoint": None if args.adapter_checkpoint is None else str(args.adapter_checkpoint),
        "leaf_names": list(leaf_index.leaf_names),
        "prompts": list(prompts),
        "calibration_leaf_mode": (
            "source_exact_leaves_fixed_temperature"
            if freeze_logit_scale
            else "source_exact_leaves_only"
            if gated_mask is not None
            else "all_leaves"
        ),
        "freeze_logit_scale": freeze_logit_scale,
        "adapted_leaf_names": adapted_leaf_names,
        "frozen_leaf_names": frozen_leaf_names,
    }
    if args.dry_run:
        print("loveda_e0_e1_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("E0/E1 is restricted to CUDA because the selected CLIP feature extraction is expensive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory already contains files: {args.output_dir}; use --overwrite only intentionally")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    seed_everything(args.seed)
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    leaf_mask = torch.from_numpy(leaf_index.descendant_leaf_mask).to(device=device)
    leaf_node_ids = torch.tensor([leaf_index.node_to_id[name] for name in leaf_index.leaf_names], device=device)

    adapter = None
    history: list[dict] = []
    if args.experiment in {"e1", "e1_gated", "e2_fixedscale_gated"}:
        adapter = LowRankSemanticCalibration(
            backbone.embedding_dim,
            args.rank,
            adapted_leaf_mask=gated_mask,
            freeze_logit_scale=freeze_logit_scale,
        ).to(device)
        if args.adapter_checkpoint is not None:
            checkpoint = torch.load(args.adapter_checkpoint, map_location=device, weights_only=True)
            adapter.load_state_dict(checkpoint["adapter_state_dict"])
            print(f"loaded_adapter_checkpoint={args.adapter_checkpoint}", flush=True)
        else:
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                # Workers are recreated each epoch so dataset.set_epoch reaches them.
                persistent_workers=False,
            )
            for epoch in range(args.epochs):
                train_dataset.set_epoch(epoch)
                loss = train_calibration_epoch(loader, backbone, adapter, text_features, leaf_mask, optimizer, device)
                history.append({"epoch": epoch + 1, "train_loss": loss})
                print(f"epoch={epoch + 1}/{args.epochs} train_loss={loss:.6f}", flush=True)
            torch.save(
                {
                    "adapter_state_dict": adapter.state_dict(),
                    "run_config": run_config,
                    "trainable_parameters": adapter.trainable_parameter_count,
                },
                args.output_dir / "adapter_final.pt",
            )

    result = evaluate_loveda_tiled(
        val_dataset,
        backbone,
        adapter,
        text_features,
        leaf_mask,
        leaf_node_ids,
        leaf_index.leaf_names,
        device,
        args.tile_size,
    )
    report = {"run_config": run_config, "history": history, "validation": metric_dict(result)}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "loveda_e0_e1_complete:",
        f"experiment={args.experiment}",
        f"mean_iou={result.mean_iou:.6f}",
        f"valid_pixels={result.valid_pixels}",
        f"output={args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
