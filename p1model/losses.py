"""Partial-label likelihood for taxonomy-aware open-vocabulary segmentation."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def taxonomy_partial_label_nll(
    leaf_logits: Tensor,
    target_node_ids: Tensor,
    descendant_leaf_mask: Tensor,
    ignore_index: int = -1,
    reduction: str = "mean",
) -> Tensor:
    """Negative log likelihood for exact or ancestor taxonomy labels.

    ``descendant_leaf_mask[node_id, leaf_id]`` determines the valid leaves for a
    target node. A leaf target reduces to ordinary cross entropy. An ancestor
    target instead optimizes the summed probability of all of its leaves, so a
    coarse label cannot fabricate an arbitrary leaf annotation.
    """
    if leaf_logits.ndim != 4:
        raise ValueError(f"leaf_logits must have shape [B, L, H, W], got {tuple(leaf_logits.shape)}")
    if target_node_ids.shape != (leaf_logits.shape[0], leaf_logits.shape[2], leaf_logits.shape[3]):
        raise ValueError("target_node_ids must have shape [B, H, W] aligned with leaf_logits")
    if descendant_leaf_mask.ndim != 2 or descendant_leaf_mask.shape[1] != leaf_logits.shape[1]:
        raise ValueError("descendant_leaf_mask must have shape [node_count, leaf_class_count]")
    if descendant_leaf_mask.dtype != torch.bool:
        raise TypeError("descendant_leaf_mask must be a bool tensor")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"unsupported reduction: {reduction}")

    valid = target_node_ids != ignore_index
    if not valid.any():
        return leaf_logits.sum() * 0.0 if reduction != "none" else leaf_logits.new_empty((0,))
    targets = target_node_ids[valid].to(dtype=torch.long)
    if targets.min() < 0 or targets.max() >= descendant_leaf_mask.shape[0]:
        raise ValueError("target contains a taxonomy node ID outside descendant_leaf_mask")
    valid_leaves = descendant_leaf_mask.to(device=leaf_logits.device)[targets]
    if not valid_leaves.any(dim=1).all():
        raise ValueError("a target taxonomy node has no valid leaf descendants")

    log_probabilities = F.log_softmax(leaf_logits, dim=1).permute(0, 2, 3, 1)[valid]
    masked = log_probabilities.masked_fill(~valid_leaves, -torch.inf)
    losses = -torch.logsumexp(masked, dim=1)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def uncertainty_unknown_score(leaf_logits: Tensor) -> Tensor:
    """Return a conservative candidate unknown score from leaf uncertainty.

    This is only a model output. It becomes a reported unknown-rejection metric
    when and only when the target dataset declares genuine `unknown` labels.
    """
    if leaf_logits.ndim != 4:
        raise ValueError("leaf_logits must have shape [B, L, H, W]")
    return 1.0 - leaf_logits.softmax(dim=1).amax(dim=1)
