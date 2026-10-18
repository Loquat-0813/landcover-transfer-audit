"""Train the preregistered Semantic Drone + LoveDA second source mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1data.loveda import discover_loveda_pairs
from p1data.semantic_drone import discover_semantic_drone_pairs, load_semantic_drone_classes
from p1data.torch_dataset import LoveDATaxonomyDataset, SemanticDroneTaxonomyDataset
from p1eval.metrics import stable_node_index
from p1eval.prompts import load_leaf_prompts
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.losses import taxonomy_partial_label_nll
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.flair_shuffle import FlairCropRecord, build_density_matched_derangement
from p1train.fixed_budget import CyclingPermutation

DEFAULT_UPDATES = 25_220


class LoveDACroplandDataset:
    """LoveDA crops with only agriculture/cropland pixels supervised."""

    def __init__(self, root, mapping, node_to_id, crop_size, seed, max_samples=None):
        self.base = LoveDATaxonomyDataset(root, "train", mapping, node_to_id, crop_size, seed, max_samples)
        self.cropland_id = node_to_id["cropland"]

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __getitem__(self, index: int):
        item = self.base[index]
        target = item["target"].clone()
        target[target != self.cropland_id] = -1
        item["target"] = target
        return item


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("sd_b0", "sd_b0_half", "sd_b1", "sd_b1_shuffle"), required=True)
    parser.add_argument("--semantic-drone-root", type=Path, required=True)
    parser.add_argument("--loveda-root", type=Path, default=ROOT / "data" / "raw" / "loveda")
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(19, 37, 73), required=True)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--shuffle-density-bin-width", type=float, default=0.01)
    parser.add_argument("--max-semantic-samples", type=int)
    parser.add_argument("--max-loveda-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_audit(path: Path, dry_run: bool) -> dict:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("case") != "semantic_drone_loveda_to_openearthmap_cropland":
        raise ValueError("source audit is not the second-case audit")
    if not dry_run and audit.get("training_admission") != "source_audit_ready":
        raise ValueError("second-case training is blocked until geographic overlap review is recorded")
    return audit


def _materialize_aux_records(aux, updates: int, seed: int, cropland_id: int):
    order = CyclingPermutation(len(aux), seed, stream=1)
    records = []
    for update in range(updates):
        index = order.at(update)
        aux.set_epoch(update)
        item = aux[index]
        target = item["target"]
        records.append(
            FlairCropRecord(
                update=update,
                source_index=index,
                source_identifier=str(item["identifier"]),
                domain="loveda_aux",
                id10_pixels=int((target == cropland_id).sum().item()),
                crop_pixels=int(target.numel()),
            )
        )
    return tuple(records)


def _train(args, sd, aux, plan, records, backbone, adapter, text_features, descendant_mask, device, leaf_names):
    sd_order = CyclingPermutation(len(sd), args.seed, stream=0)
    aux_order = CyclingPermutation(len(aux), args.seed, stream=1) if aux is not None else None
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    adapter.train()
    backbone.eval()
    for update in range(args.updates):
        sd.set_epoch(update)
        samples = [sd[sd_order.at(2 * update + offset)] for offset in range(2)] if args.arm == "sd_b0" else [sd[sd_order.at(update)]]
        if args.arm == "sd_b1":
            aux.set_epoch(update)
            samples.append(aux[aux_order.at(update)])
        elif args.arm == "sd_b1_shuffle":
            image_record = records[update]
            mask_record = records[plan.permutation[update]]
            aux.set_epoch(image_record.update)
            image_item = aux[image_record.source_index]
            aux.set_epoch(mask_record.update)
            mask_item = aux[mask_record.source_index]
            samples.append({"image": image_item["image"], "target": mask_item["target"], "identifier": f"{image_record.source_identifier}<-{mask_record.source_identifier}"})
        images = torch.stack([item["image"] for item in samples]).to(device, non_blocking=True)
        targets = torch.stack([item["target"] for item in samples]).to(device, non_blocking=True)
        features = backbone.encode_image(images)
        logits = adapter(features, text_features)
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        loss = taxonomy_partial_label_nll(logits, targets, descendant_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (update + 1) % 100 == 0 or update + 1 == args.updates:
            history.append({"update": update + 1, "loss": float(loss.detach())})
            print(f"update={update + 1}/{args.updates} loss={float(loss.detach()):.6f}", flush=True)
    return history


def main() -> int:
    args = _parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")
    audit = _load_audit(args.source_audit, args.dry_run)
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(ROOT / "configs" / "taxonomy_prompts_v0.yaml", leaves.leaf_names)
    sd_classes = load_semantic_drone_classes(args.semantic_drone_root / "semantic_drone_dataset" / "training_set" / "gt" / "semantic" / "class_dict.csv")
    class_by_name = {item.name: item for item in sd_classes}
    rgb_to_node = {class_by_name[name].packed_rgb: node_to_id[target] for name, target in (("grass", "herbaceous_vegetation"), ("tree", "woody_vegetation"), ("water", "surface_water"))}
    sd_pairs = discover_semantic_drone_pairs(args.semantic_drone_root)
    sd = SemanticDroneTaxonomyDataset(sd_pairs, rgb_to_node, args.crop_size, args.seed, args.max_semantic_samples)
    loveda_mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "loveda", taxonomy)
    aux = LoveDACroplandDataset(args.loveda_root, loveda_mapping, node_to_id, args.crop_size, args.seed, args.max_loveda_samples) if args.arm in {"sd_b1", "sd_b1_shuffle"} else None
    records = _materialize_aux_records(aux, args.updates, args.seed, node_to_id["cropland"]) if aux is not None and args.arm == "sd_b1_shuffle" else None
    plan = build_density_matched_derangement(records, args.seed, args.shuffle_density_bin_width) if records is not None else None
    run_config = {
        "experiment": f"semantic_drone_loveda_second_case_{args.arm}",
        "protocol_mode": "second_case_fixed_budget" if args.updates == DEFAULT_UPDATES and not args.dry_run else "smoke_or_debug_not_target_evaluation",
        "case": "semantic_drone_loveda_to_openearthmap_cropland",
        "arm": args.arm,
        "seed": args.seed,
        "updates": args.updates,
        "rank": args.rank,
        "crop_size": args.crop_size,
        "model_id": args.model_id,
        "source_audit": str(args.source_audit),
        "source_audit_sha256": _sha256(args.source_audit),
        "semantic_drone_pairs": len(sd),
        "loveda_aux_pairs": len(aux) if aux is not None else 0,
        "prompts": list(prompts),
        "supervised_semantic_drone_classes": ["grass", "tree", "water"],
        "auxiliary_label": "LoveDA agriculture raw value 7 -> cropland",
    }
    if plan is not None:
        run_config["shuffle_schedule"] = {"record_count": len(records), "permutation": list(plan.permutation), "max_abs_density_difference": plan.max_abs_density_difference, "mean_abs_density_difference": plan.mean_abs_density_difference, "merged_singleton_bins": list(plan.merged_singleton_bins)}
    if args.dry_run:
        print("second_case_train_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("second-case optimization is restricted to CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    if plan is not None:
        (args.output_dir / "shuffle_schedule.json").write_text(json.dumps(run_config["shuffle_schedule"], indent=2) + "\n", encoding="utf-8")
    _seed_everything(args.seed)
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    descendant_mask = torch.from_numpy(leaves.descendant_leaf_mask).to(device=device)
    adapter = LowRankSemanticCalibration(backbone.embedding_dim, args.rank).to(device)
    history = _train(args, sd, aux, plan, records, backbone, adapter, text_features, descendant_mask, device, leaves.leaf_names)
    torch.save({"adapter_state_dict": adapter.state_dict(), "run_config": run_config, "trainable_parameters": adapter.trainable_parameter_count}, args.output_dir / "adapter_final.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps({"run_config": run_config, "history": history}, indent=2) + "\n", encoding="utf-8")
    print("second_case_train_complete:", f"arm={args.arm}", f"output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
