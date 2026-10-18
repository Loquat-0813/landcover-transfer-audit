"""Train one frozen-protocol U-Net source-mixture arm without opening target metrics."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1data.flair import FLAIR_TRAIN_DOMAINS, activate_flair_b1_mapping, discover_flair_domain_pairs
from p1data.torch_dataset import FlairTaxonomyDataset, LoveDATaxonomyDataset
from p1eval.metrics import stable_node_index
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.taxonomy import build_taxonomy_leaf_index
from p1model.unet import ResNet34UNet
from p1train.fixed_budget import CyclingPermutation, loveda_positions_for_update
from p1train.flair_shuffle import FlairCropRecord, FlairShufflePlan, build_density_matched_derangement, load_density_matched_derangement
from p1train.unet_protocol import evaluate_unet_tiled_by_identifier, seed_everything, stack_examples, train_one_update


DEFAULT_UPDATES = 25_220
SCHEDULE_MEAN_DENSITY_LIMIT = 0.005
SCHEDULE_P95_DENSITY_LIMIT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("unet_b0", "unet_b0_half", "unet_b1", "unet_b1_shuffle"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "loveda")
    parser.add_argument("--flair-extracted-root", type=Path)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(19, 37, 73), required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--encoder-weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--max-love-train-samples", type=int)
    parser.add_argument("--max-flair-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--shuffle-schedule", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict_sha256(state_dict: dict) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _load_ready_source_audit(path: Path, allow_pending: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"FLAIR source audit is missing: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("herbaceous_raw_id") != 10 or audit.get("herbaceous_pixel_count", 0) <= 0:
        raise ValueError("FLAIR source audit does not validate raw ID 10 herbaceous supervision")
    if audit.get("train", {}).get("paired_raster_count", 0) <= 0:
        raise ValueError("FLAIR source audit lacks paired train-domain rasters")
    if not allow_pending and audit.get("source_mapping_status") != "source_audit_ready":
        raise ValueError("FLAIR source audit is not ready for U-Net optimization")
    return audit


def _base_arm_name(arm: str) -> str:
    return {"unet_b0": "b0", "unet_b0_half": "b0_half", "unet_b1": "b1", "unet_b1_shuffle": "b1_shuffle"}[arm]


def _materialize_flair_crop_records(dataset: FlairTaxonomyDataset, updates: int, seed: int) -> tuple[FlairCropRecord, ...]:
    order = CyclingPermutation(len(dataset), seed, stream=1)
    records: list[FlairCropRecord] = []
    for update in range(updates):
        source_index = order.at(update)
        example = dataset.sample_at(source_index, update)
        target = example["target"]
        identifier = str(example["identifier"])
        domain, separator, _ = identifier.partition("/")
        if not separator or not isinstance(target, torch.Tensor):
            raise ValueError("FLAIR crop record is malformed")
        records.append(
            FlairCropRecord(
                update=update,
                source_index=source_index,
                source_identifier=identifier,
                domain=domain,
                id10_pixels=int(target.ne(-1).sum().item()),
                crop_pixels=int(target.numel()),
            )
        )
    return tuple(records)


def _schedule_audit(records: tuple[FlairCropRecord, ...], plan: FlairShufflePlan) -> dict:
    differences = np.asarray(
        [abs(records[index].id10_density - records[mask_index].id10_density) for index, mask_index in enumerate(plan.permutation)],
        dtype=np.float64,
    )
    native_pairs = sum(
        records[index].source_identifier == records[mask_index].source_identifier
        for index, mask_index in enumerate(plan.permutation)
    )
    same_region = all(records[index].domain == records[mask_index].domain for index, mask_index in enumerate(plan.permutation))
    audit = {
        "native_pair_count": native_pairs,
        "same_region": same_region,
        "mask_density_multiset_preserved": sorted(record.id10_pixels for record in records)
        == sorted(records[index].id10_pixels for index in plan.permutation),
        "mean_abs_density_difference": float(differences.mean()),
        "p95_abs_density_difference": float(np.quantile(differences, 0.95)),
        "max_abs_density_difference": float(differences.max()),
        "mean_limit": SCHEDULE_MEAN_DENSITY_LIMIT,
        "p95_limit": SCHEDULE_P95_DENSITY_LIMIT,
    }
    audit["eligible_for_pairing_interpretation"] = bool(
        native_pairs == 0
        and same_region
        and audit["mask_density_multiset_preserved"]
        and audit["mean_abs_density_difference"] <= SCHEDULE_MEAN_DENSITY_LIMIT
        and audit["p95_abs_density_difference"] <= SCHEDULE_P95_DENSITY_LIMIT
    )
    return audit


def _train(
    arm: str,
    love_dataset: LoveDATaxonomyDataset,
    flair_dataset: FlairTaxonomyDataset | None,
    model: ResNet34UNet,
    optimizer: torch.optim.Optimizer,
    descendant_leaf_mask: torch.Tensor,
    device: torch.device,
    updates: int,
    seed: int,
    records: tuple[FlairCropRecord, ...] | None,
    plan: FlairShufflePlan | None,
) -> list[dict]:
    base_arm = _base_arm_name(arm)
    love_order = CyclingPermutation(len(love_dataset), seed, stream=0)
    flair_order = CyclingPermutation(len(flair_dataset), seed, stream=1) if flair_dataset is not None else None
    history: list[dict] = []
    interval_loss = 0.0
    model.train()
    for update in range(updates):
        love_dataset.set_epoch(update)
        examples = [love_dataset[love_order.at(position)] for position in loveda_positions_for_update(base_arm, update)]
        if arm == "unet_b1":
            if flair_dataset is None or flair_order is None:
                raise RuntimeError("unet_b1 requires FLAIR data")
            flair_dataset.set_epoch(update)
            examples.append(flair_dataset[flair_order.at(update)])
        elif arm == "unet_b1_shuffle":
            if flair_dataset is None or records is None or plan is None:
                raise RuntimeError("unet_b1_shuffle requires a frozen FLAIR shuffle schedule")
            image_record = records[update]
            mask_record = records[plan.permutation[update]]
            image_example = flair_dataset.sample_at(image_record.source_index, image_record.update)
            mask_example = flair_dataset.sample_at(mask_record.source_index, mask_record.update)
            examples.append(
                {
                    "image": image_example["image"],
                    "target": mask_example["target"],
                    "identifier": f"{image_record.source_identifier}<-{mask_record.source_identifier}",
                }
            )
        images, target = stack_examples(examples)
        loss = train_one_update(model, optimizer, images.to(device), target.to(device), descendant_leaf_mask)
        interval_loss += float(loss.cpu())
        if (update + 1) % 100 == 0 or update + 1 == updates:
            record = {"update": update + 1, "mean_loss_last_interval": interval_loss / ((update % 100) + 1)}
            history.append(record)
            print(f"update={record['update']}/{updates} train_loss={record['mean_loss_last_interval']:.6f}", flush=True)
            interval_loss = 0.0
    return history


def main() -> int:
    args = parse_args()
    if min(args.crop_size, args.tile_size, args.updates) < 1 or args.crop_size % 32 or args.tile_size % 32:
        raise SystemExit("crop size, tile size, and updates must be positive; tiles must be divisible by 32")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")
    needs_flair = args.arm in {"unet_b1", "unet_b1_shuffle"}
    if needs_flair and args.flair_extracted_root is None:
        raise SystemExit("FLAIR arms require --flair-extracted-root")
    source_audit = _load_ready_source_audit(args.source_audit, allow_pending=args.dry_run)
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    love_mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "loveda", taxonomy)
    flair_mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_flair_b1_v1.yaml", "flair_b1", taxonomy)
    if needs_flair:
        flair_mapping = activate_flair_b1_mapping(flair_mapping, source_audit)
    love_train = LoveDATaxonomyDataset(args.data_root, "train", love_mapping, node_to_id, args.crop_size, args.seed, args.max_love_train_samples)
    love_validation = LoveDATaxonomyDataset(args.data_root, "val", love_mapping, node_to_id, None, args.seed, args.max_validation_samples)
    flair_train = None
    if needs_flair:
        flair_pairs = discover_flair_domain_pairs(args.flair_extracted_root, FLAIR_TRAIN_DOMAINS)
        flair_train = FlairTaxonomyDataset(flair_pairs, flair_mapping, node_to_id, args.crop_size, args.seed, args.max_flair_train_samples)
    full_budget = args.updates == DEFAULT_UPDATES and args.max_love_train_samples is None and args.max_flair_train_samples is None
    protocol_mode = "p1_unet_cross_family_v1" if full_budget else "p1_unet_smoke_or_debug_not_target_evaluation"
    if args.dry_run:
        print(
            "unet_train_dry_run_valid:",
            json.dumps({"arm": args.arm, "protocol_mode": protocol_mode, "love_train_pairs": len(love_train), "flair_train_pairs": len(flair_train) if flair_train else 0}, sort_keys=True),
        )
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("full U-Net optimization is restricted to CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    model = ResNet34UNet(class_count=len(leaves.leaf_names), pretrained_encoder=args.encoder_weights == "imagenet").to(device)
    initial_encoder_sha256 = _state_dict_sha256(model.encoder.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    records = None
    plan = None
    schedule_audit = None
    if args.arm == "unet_b1_shuffle":
        if flair_train is None:
            raise RuntimeError("unet_b1_shuffle requires FLAIR training data")
        records = _materialize_flair_crop_records(flair_train, args.updates, args.seed)
        if full_budget and args.shuffle_schedule is None:
            raise SystemExit("full unet_b1_shuffle requires a precomputed --shuffle-schedule")
        plan = (
            load_density_matched_derangement(args.shuffle_schedule, records)
            if args.shuffle_schedule is not None
            else build_density_matched_derangement(records, args.seed, density_bin_width=0.01)
        )
        schedule_audit = _schedule_audit(records, plan)
        schedule_audit["gate_enforced"] = full_budget
        schedule_audit["schedule_status"] = (
            "eligible_for_pairing_interpretation"
            if schedule_audit["eligible_for_pairing_interpretation"]
            else "ineligible_for_pairing_interpretation"
        )
        if full_budget and not schedule_audit["eligible_for_pairing_interpretation"]:
            raise SystemExit(f"frozen shuffle schedule failed pre-training gates: {json.dumps(schedule_audit, sort_keys=True)}")
    run_config = {
        "experiment": f"unet_coverage_{args.arm}",
        "protocol_mode": protocol_mode,
        "protocol_id": "p1-unet-cross-family-v1",
        "arm": args.arm,
        "seed": args.seed,
        "model": "resnet34_unet_groupnorm_decoder",
        "encoder_weight_id": ResNet34UNet.encoder_weight_id if args.encoder_weights == "imagenet" else "none",
        "initial_encoder_state_sha256": initial_encoder_sha256,
        "source_audit": str(args.source_audit),
        "source_audit_sha256": _sha256(args.source_audit),
        "source_audit_identifier_set_sha256": source_audit.get("identifier_set_sha256"),
        "love_train_pairs": len(love_train),
        "flair_train_pairs": len(flair_train) if flair_train else 0,
        "love_validation_pairs": len(love_validation),
        "flair_train_domains": list(FLAIR_TRAIN_DOMAINS) if needs_flair else [],
        "crop_size": args.crop_size,
        "tile_size": args.tile_size,
        "updates": args.updates,
        "batch_composition": {"unet_b0": ["loveda", "loveda"], "unet_b0_half": ["loveda"], "unet_b1": ["loveda", "flair_raw_id_10_only"], "unet_b1_shuffle": ["loveda", "flair_rgb_with_deranged_raw_id_10_mask"]}[args.arm],
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "taxonomy": taxonomy.version,
        "loveda_mapping": "dataset-label-mappings-v0",
        "flair_mapping": "dataset-label-mappings-flair-b1-v1",
        "torch_version": str(torch.__version__),
        "torchvision_version": __import__("torchvision").__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(device),
        "shuffle_schedule_audit": schedule_audit,
        "input_shuffle_schedule": str(args.shuffle_schedule) if args.shuffle_schedule is not None else None,
        "input_shuffle_schedule_sha256": _sha256(args.shuffle_schedule) if args.shuffle_schedule is not None else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if records is not None and plan is not None:
        schedule_path = args.output_dir / "flair_shuffle_schedule.json"
        plan.write_json(schedule_path, records)
        run_config["shuffle_schedule_sha256"] = _sha256(schedule_path)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    history = _train(args.arm, love_train, flair_train, model, optimizer, torch.from_numpy(leaves.descendant_leaf_mask).to(device), device, args.updates, args.seed, records, plan)
    torch.save({"model_state_dict": model.state_dict(), "run_config": run_config, "trainable_parameters": model.trainable_parameter_count}, args.output_dir / "model_final.pt")
    validation, _ = evaluate_unet_tiled_by_identifier(
        love_validation,
        model,
        torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], device=device),
        leaves.leaf_names,
        device,
        args.tile_size,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"run_config": run_config, "history": history, "loveda_validation": {"mean_iou": validation.mean_iou, "per_class_iou": validation.per_class_iou, "valid_pixels": validation.valid_pixels}}, indent=2),
        encoding="utf-8",
    )
    print(f"unet_train_complete: arm={args.arm} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
