"""Run the preregistered fixed-budget LoveDA/FLAIR B0 or B1 source experiment."""

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
from p1data.flair import FLAIR_TRAIN_DOMAINS, activate_flair_b1_mapping, discover_flair_domain_pairs
from p1data.torch_dataset import FlairTaxonomyDataset, LoveDATaxonomyDataset
from p1eval.metrics import stable_node_index
from p1eval.prompts import load_leaf_prompts
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.losses import taxonomy_partial_label_nll
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.fixed_budget import CyclingPermutation, loveda_positions_for_update
from p1train.flair_shuffle import FlairCropRecord, FlairShufflePlan, build_density_matched_derangement
from p1train.loveda_e1 import evaluate_loveda_tiled
from p1train.source_balance import SourceBalancer, two_source_losses
from p1train.retention import old_class_distillation


DEFAULT_UPDATES = 25_220


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("b0", "b0_half", "b0_half_batch2", "b1", "b1_shuffle"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "loveda")
    parser.add_argument("--flair-extracted-root", type=Path)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(19, 37, 73), default=19)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--shuffle-density-bin-width", type=float, default=0.01)
    parser.add_argument("--max-love-train-samples", type=int)
    parser.add_argument("--max-flair-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loss-balance", choices=("legacy", "fixed_source", "gradnorm"), default="legacy")
    parser.add_argument("--auxiliary-weight", type=float, default=1.0)
    parser.add_argument("--gradnorm-alpha", type=float, default=1.5)
    parser.add_argument("--gradnorm-weight-lr", type=float, default=0.025)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_ready_source_audit(path: Path, allow_pending: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"FLAIR source audit is missing: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("herbaceous_raw_id") != 10 or audit.get("herbaceous_pixel_count", 0) <= 0:
        raise ValueError("FLAIR source audit does not validate raw ID 10 herbaceous supervision")
    train = audit.get("train", {})
    if train.get("paired_raster_count", 0) <= 0 or train.get("herbaceous_pixel_count", 0) <= 0:
        raise ValueError("FLAIR source audit lacks nonempty train-domain herbaceous support")
    if not allow_pending and audit.get("source_mapping_status") != "source_audit_ready":
        raise ValueError(
            "FLAIR source audit is not ready for optimization; complete the geographic-overlap review before GPU training"
        )
    return audit


def _stack_examples(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.stack([example["image"] for example in examples]), torch.stack(
        [example["target"] for example in examples]
    )


def _materialize_flair_crop_records(
    flair_dataset: FlairTaxonomyDataset,
    updates: int,
    seed: int,
) -> tuple[FlairCropRecord, ...]:
    """Freeze the B1 FLAIR crop schedule before shuffled-pairing training."""
    order = CyclingPermutation(len(flair_dataset), seed, stream=1)
    records: list[FlairCropRecord] = []
    for update in range(updates):
        source_index = order.at(update)
        example = flair_dataset.sample_at(source_index, update)
        target = example["target"]
        if not isinstance(target, torch.Tensor):
            raise TypeError("FLAIR dataset returned a non-tensor target")
        identifier = str(example["identifier"])
        domain, separator, _ = identifier.partition("/")
        if not separator:
            raise ValueError(f"FLAIR identifier lacks a domain prefix: {identifier}")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_fixed_budget(
    love_dataset: LoveDATaxonomyDataset,
    flair_dataset: FlairTaxonomyDataset | None,
    arm: str,
    updates: int,
    seed: int,
    backbone: FrozenCLIPPatchEncoder,
    adapter: LowRankSemanticCalibration,
    text_features: torch.Tensor,
    descendant_leaf_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    flair_shuffle_records: tuple[FlairCropRecord, ...] | None = None,
    flair_shuffle_plan: FlairShufflePlan | None = None,
    source_balancer: SourceBalancer | None = None,
    balance_trace=None,
    retention_teacher: LowRankSemanticCalibration | None = None,
    retention_weight: float = 1.0,
    retention_temperature: float = 2.0,
    retention_node_ids: tuple[int, ...] = (),
    retention_trace=None,
    verify_finite: bool = False,
    coverage_gate=None,
) -> list[dict]:
    if coverage_gate is not None and (retention_teacher is None or arm != 'b1' or source_balancer is not None):
        raise ValueError('coverage gate requires the matched teacher B1 protocol')
    if source_balancer is not None and arm != "b1":
        raise ValueError("source balancing is a new native B1 experiment only")
    if retention_teacher is not None:
        if arm != "b1" or source_balancer is not None:
            raise ValueError("retention requires native B1 with unchanged pooled-pixel loss")
        if retention_teacher.training or any(p.requires_grad for p in retention_teacher.parameters()):
            raise ValueError("the retention teacher must be frozen and in evaluation mode")
        if not np.isfinite(retention_weight) or retention_weight <= 0:
            raise ValueError("retention weight must be finite and positive")
    love_order = CyclingPermutation(len(love_dataset), seed, stream=0)
    flair_order = CyclingPermutation(len(flair_dataset), seed, stream=1) if flair_dataset is not None else None
    history: list[dict] = []
    loss_sum = 0.0
    adapter.train()
    backbone.eval()
    for update in range(updates):
        # A unique update nonce changes the deterministic crop even if a source
        # raster appears in a later cycling pass.
        love_dataset.set_epoch(update)
        examples = [love_dataset[love_order.at(position)] for position in loveda_positions_for_update(arm, update)]
        if arm == "b0_half_batch2":
            # Duplicate the deterministic crop to match B1's two-sample batch
            # without increasing LoveDA exposure beyond the half-control.
            examples.append(examples[0].copy())
        if arm == "b1":
            if flair_dataset is None or flair_order is None:
                raise RuntimeError("B1 requires the FLAIR training dataset")
            flair_dataset.set_epoch(update)
            examples.append(flair_dataset[flair_order.at(update)])
        if arm == "b1_shuffle":
            if flair_dataset is None or flair_shuffle_records is None or flair_shuffle_plan is None:
                raise RuntimeError("B1-shuffle requires frozen FLAIR crop records and a shuffle plan")
            image_record = flair_shuffle_records[update]
            mask_record = flair_shuffle_records[flair_shuffle_plan.permutation[update]]
            image_example = flair_dataset.sample_at(image_record.source_index, image_record.update)
            mask_example = flair_dataset.sample_at(mask_record.source_index, mask_record.update)
            mask_target = mask_example["target"]
            if not isinstance(mask_target, torch.Tensor):
                raise TypeError("FLAIR dataset returned a non-tensor shuffled target")
            if int(mask_target.ne(-1).sum().item()) != mask_record.id10_pixels:
                raise RuntimeError("materialized shuffled mask density differs from the frozen schedule")
            examples.append(
                {
                    "image": image_example["image"],
                    "target": mask_target,
                    "identifier": f"{image_record.source_identifier}<-{mask_record.source_identifier}",
                }
            )
        images, target = _stack_examples(examples)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        features = backbone.encode_image(images)
        logits = adapter(features, text_features)
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        balance_record = None
        if source_balancer is None:
            loss = taxonomy_partial_label_nll(logits, target, descendant_leaf_mask)
        else:
            source_losses, valid_counts = two_source_losses(logits, target, descendant_leaf_mask)
            loss, balance_record = source_balancer.objective(
                source_losses, valid_counts, (adapter.image_projection.weight,)
            )
        retention_record = None
        if retention_teacher is not None:
            with torch.no_grad():
                teacher_logits = retention_teacher(features[:1], text_features)
                teacher_logits = F.interpolate(teacher_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            supervised_loss = loss
            retention_loss, retention_pixels = old_class_distillation(
                logits[:1], teacher_logits, target[:1], retention_node_ids, retention_temperature
            )
            loss = supervised_loss + retention_weight * retention_loss
            retention_record = {"update": update + 1, "supervised_pooled_loss": float(supervised_loss.detach()),
                                "old_class_kl": float(retention_loss.detach()), "distilled_pixels": retention_pixels,
                                "total_loss": float(loss.detach()), "weight": retention_weight}
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite retention objective")
        if coverage_gate is None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if source_balancer is not None or retention_teacher is not None or verify_finite:
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in adapter.parameters()):
                    raise FloatingPointError("nonfinite adapter gradient; experiment stopped")
            optimizer.step()
        else:
            old_nll = taxonomy_partial_label_nll(logits[:1], target[:1], descendant_leaf_mask)
            aux_nll = taxonomy_partial_label_nll(logits[1:], target[1:], descendant_leaf_mask)
            def score_proposal():
                proposed = adapter(features, text_features)
                proposed = F.interpolate(proposed, size=target.shape[-2:], mode='bilinear', align_corners=False)
                old = taxonomy_partial_label_nll(proposed[:1], target[:1], descendant_leaf_mask)
                aux = taxonomy_partial_label_nll(proposed[1:], target[1:], descendant_leaf_mask)
                kd,_ = old_class_distillation(proposed[:1], teacher_logits, target[:1], retention_node_ids, retention_temperature)
                return old + retention_weight*kd, aux
            gate_record = coverage_gate.step(adapter,optimizer,supervised_loss,old_nll,aux_nll,retention_loss,
                retention_weight,(retention_pixels,int(target[1:].ne(-1).sum())),score_proposal)
            retention_record['coverage_gate'] = gate_record
        if retention_record is not None and retention_trace is not None:
            retention_trace.write(json.dumps(retention_record, allow_nan=False) + '\n')
        if source_balancer is not None:
            source_balancer.step()
            balance_record.update(update=update + 1, weights_next=source_balancer.weights.detach().tolist(),
                                  positivity_projection_steps=int(source_balancer.projection_steps))
            if balance_trace is not None:
                balance_trace.write(json.dumps(balance_record, allow_nan=False) + "\n")
        loss_sum += float(loss.detach())
        if (update + 1) % 100 == 0 or update + 1 == updates:
            record = {"update": update + 1, "mean_loss_last_interval": loss_sum / ((update % 100) + 1)}
            if balance_record is not None:
                record["source_balance"] = balance_record
                if balance_trace is not None:
                    balance_trace.flush()
            if retention_record is not None:
                record["retention"] = retention_record
                if retention_trace is not None:
                    retention_trace.flush()
            history.append(record)
            print(f"update={record['update']}/{updates} train_loss={record['mean_loss_last_interval']:.6f}", flush=True)
            loss_sum = 0.0
    return history


def main() -> int:
    args = parse_args()
    if args.loss_balance != "legacy":
        if args.arm != "b1":
            raise SystemExit("--loss-balance is supported only for native B1")
        # Validate settings even on dry runs, before accessing any datasets.
        SourceBalancer(args.loss_balance, args.auxiliary_weight, args.gradnorm_alpha, args.gradnorm_weight_lr)
    elif args.auxiliary_weight != 1.0:
        raise SystemExit("--auxiliary-weight requires --loss-balance fixed_source")
    if min(args.crop_size, args.tile_size, args.rank, args.updates) < 1:
        raise SystemExit("crop size, tile size, rank, and updates must be positive")
    if args.arm in {"b1", "b1_shuffle"} and args.flair_extracted_root is None:
        raise SystemExit("B1 and B1-shuffle require --flair-extracted-root")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")

    source_audit = _load_ready_source_audit(args.source_audit, allow_pending=args.dry_run)
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    loveda_mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "loveda", taxonomy)
    flair_mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_flair_b1_v1.yaml", "flair_b1", taxonomy)
    if flair_mapping.raw_id_to_node != {10: "herbaceous_vegetation"}:
        raise ValueError("B1 requires the narrow, raw-ID-10-only FLAIR mapping")
    if args.arm in {"b1", "b1_shuffle"} and not args.dry_run:
        flair_mapping = activate_flair_b1_mapping(flair_mapping, source_audit)
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(ROOT / "configs" / "taxonomy_prompts_v0.yaml", leaves.leaf_names)
    love_train = LoveDATaxonomyDataset(
        args.data_root, "train", loveda_mapping, node_to_id, args.crop_size, args.seed, args.max_love_train_samples
    )
    love_validation = LoveDATaxonomyDataset(
        args.data_root, "val", loveda_mapping, node_to_id, None, args.seed, args.max_validation_samples
    )
    flair_train = None
    flair_train_pairs = 0
    if args.arm in {"b1", "b1_shuffle"}:
        flair_pairs = discover_flair_domain_pairs(args.flair_extracted_root, FLAIR_TRAIN_DOMAINS)
        flair_train = FlairTaxonomyDataset(
            flair_pairs, flair_mapping, node_to_id, args.crop_size, args.seed, args.max_flair_train_samples
        )
        flair_train_pairs = len(flair_train)
    flair_shuffle_records = None
    flair_shuffle_plan = None
    if args.arm == "b1_shuffle":
        flair_shuffle_records = _materialize_flair_crop_records(flair_train, args.updates, args.seed)
        flair_shuffle_plan = build_density_matched_derangement(
            flair_shuffle_records,
            args.seed,
            args.shuffle_density_bin_width,
        )
    is_full_budget = args.updates == DEFAULT_UPDATES and args.max_love_train_samples is None and args.max_flair_train_samples is None
    protocol_mode = (
        "fixed_budget_primary"
        if is_full_budget and args.arm in {"b0", "b1"}
        else "post_primary_reviewer_driven_shuffled_pairing_robustness_control"
        if is_full_budget and args.arm == "b1_shuffle"
        else "post_primary_loveda_exposure_matched_batch_size_control"
        if is_full_budget and args.arm == "b0_half_batch2"
        else "post_primary_loveda_exposure_control"
        if is_full_budget and args.arm == "b0_half"
        else "smoke_or_debug_not_target_evaluation"
    )
    batch_composition = {
        "b0": ["loveda", "loveda"],
        "b0_half": ["loveda"],
        "b0_half_batch2": ["loveda", "loveda_duplicate"],
        "b1": ["loveda", "flair_raw_id_10_only"],
        "b1_shuffle": ["loveda", "flair_rgb_with_density_matched_deranged_raw_id_10_mask"],
    }[args.arm]
    if args.loss_balance != "legacy":
        protocol_mode = "exploratory_source_balance_full_budget" if is_full_budget else "exploratory_source_balance_pilot"
    run_config = {
        "experiment": f"flair_coverage_{args.arm}",
        "protocol_mode": protocol_mode,
        "arm": args.arm,
        "seed": args.seed,
        "model_id": args.model_id,
        "source_audit": str(args.source_audit),
        "source_audit_identifier_set_sha256": source_audit.get("identifier_set_sha256"),
        "love_train_pairs": len(love_train),
        "flair_train_pairs": flair_train_pairs,
        "love_validation_pairs": len(love_validation),
        "flair_train_domains": list(FLAIR_TRAIN_DOMAINS) if args.arm in {"b1", "b1_shuffle"} else [],
        "crop_size": args.crop_size,
        "tile_size": args.tile_size,
        "rank": args.rank,
        "updates": args.updates,
        "batch_composition": batch_composition,
        "loveda_crop_exposures_per_update": len(loveda_positions_for_update(args.arm, 0)),
        "loveda_crop_exposures_total": args.updates * len(loveda_positions_for_update(args.arm, 0)),
        "scientific_role": {
            "b0": "primary_fixed_budget_loveda_only",
            "b0_half": "post_primary_loveda_exposure_matched_control",
            "b0_half_batch2": "post_primary_loveda_exposure_matched_batch_size_control",
            "b1": "primary_fixed_budget_independent_herbaceous_completion",
            "b1_shuffle": "post_primary_density_matched_shuffled_pairing_robustness_control",
        }[args.arm],
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "prompts": list(prompts),
        "taxonomy": taxonomy.version,
        "loveda_mapping": "dataset-label-mappings-v0",
        "flair_mapping": "dataset-label-mappings-flair-b1-v1",
        "flair_mapping_runtime_status": flair_mapping.scoring_status,
        "flair_rgb_channels": [0, 1, 2] if args.arm in {"b1", "b1_shuffle"} else [],
    }
    if args.loss_balance != "legacy":
        run_config.update(experiment=f"flair_coverage_b1_{args.loss_balance}",
                          scientific_role="exploratory_source_balance_not_frozen_primary")
        run_config["source_balance"] = {
            "mode": args.loss_balance, "auxiliary_weight": args.auxiliary_weight,
            "alpha": args.gradnorm_alpha, "weight_lr": args.gradnorm_weight_lr,
            "weight_sum": 2.0, "reference_parameters": ["image_projection.weight"],
            "normalization": "per_source_valid_pixel_mean",
            "empty_source_policy": "zero_loss_keep_weights_skip_controller",
            "initial_loss_policy": "first_joint_valid_nonzero_gradient_step",
            "weight_optimizer": "SGD_without_momentum", "positive_floor_before_normalization": 1e-6,
            "performance_fallback": False,
            "citation": "https://proceedings.mlr.press/v80/chen18a.html",
        }
        run_config["implementation_sha256"] = {
            str(path.relative_to(ROOT)): _sha256(path) for path in
            (Path(__file__).resolve(), ROOT / "p1train/source_balance.py", ROOT / "p1model/losses.py", ROOT / "p1model/calibration.py")
        }
        run_config["local_inputs"] = {"loveda_root": str(args.data_root),
                                      "flair_root": str(args.flair_extracted_root)}
        run_config["runtime"] = {"python": sys.version, "torch": str(torch.__version__),
                                 "cuda_build": torch.version.cuda}
    if args.dry_run:
        print("flair_b1_train_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("B1 optimization is restricted to CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if flair_shuffle_records is not None and flair_shuffle_plan is not None:
        schedule_path = args.output_dir / "flair_shuffle_schedule.json"
        flair_shuffle_plan.write_json(schedule_path, flair_shuffle_records)
        run_config["flair_shuffle_schedule"] = {
            "path": schedule_path.name,
            "sha256": _sha256(schedule_path),
            "record_count": len(flair_shuffle_records),
            "density_bin_width": args.shuffle_density_bin_width,
            "merged_singleton_bins": len(flair_shuffle_plan.merged_singleton_bins),
            "max_abs_density_difference": flair_shuffle_plan.max_abs_density_difference,
            "mean_abs_density_difference": flair_shuffle_plan.mean_abs_density_difference,
        }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    seed_everything(args.seed)
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    descendant_leaf_mask = torch.from_numpy(leaves.descendant_leaf_mask).to(device=device)
    leaf_node_ids = torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], device=device)
    adapter = LowRankSemanticCalibration(backbone.embedding_dim, args.rank).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    source_balancer = (SourceBalancer(args.loss_balance, args.auxiliary_weight, args.gradnorm_alpha,
                                     args.gradnorm_weight_lr).to(device) if args.loss_balance != "legacy" else None)
    balance_trace = (args.output_dir / "source_balance.jsonl").open("w", encoding="utf-8") if source_balancer is not None else None
    try:
        history = _train_fixed_budget(
            love_train,
            flair_train,
            args.arm,
            args.updates,
            args.seed,
            backbone,
            adapter,
            text_features,
            descendant_leaf_mask,
            optimizer,
            device,
            flair_shuffle_records,
            flair_shuffle_plan,
            source_balancer,
            balance_trace,
        )
    finally:
        if balance_trace is not None:
            balance_trace.close()
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "run_config": run_config,
            "trainable_parameters": adapter.trainable_parameter_count,
            "source_balance_state_dict": source_balancer.state_dict() if source_balancer is not None else None,
        },
        args.output_dir / "adapter_final.pt",
    )
    validation = evaluate_loveda_tiled(
        love_validation,
        backbone,
        adapter,
        text_features,
        descendant_leaf_mask,
        leaf_node_ids,
        leaves.leaf_names,
        device,
        args.tile_size,
    )
    report = {
        "run_config": run_config,
        "history": history,
        "loveda_validation": {
            "mean_iou": validation.mean_iou,
            "per_class_iou": validation.per_class_iou,
            "valid_pixels": validation.valid_pixels,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "flair_b1_train_complete:",
        f"arm={args.arm}",
        f"mean_iou={validation.mean_iou:.6f}",
        f"updates={args.updates}",
        f"output={args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
