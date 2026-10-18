"""Split-conformal routing that protects source-uncovered leaf evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def split_conformal_probability_threshold_from_scores(nonconformity_scores: Tensor, alpha: float) -> float:
    """Convert valid true-label nonconformity scores into a set threshold."""
    if nonconformity_scores.ndim != 1 or nonconformity_scores.numel() == 0:
        raise ValueError("conformal calibration needs a non-empty one-dimensional score tensor")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not torch.isfinite(nonconformity_scores).all() or nonconformity_scores.min() < 0 or nonconformity_scores.max() > 1:
        raise ValueError("nonconformity scores must be finite values in [0, 1]")
    rank = min(nonconformity_scores.numel(), math.ceil((nonconformity_scores.numel() + 1) * (1.0 - alpha)))
    quantile = nonconformity_scores.sort().values[rank - 1]
    return float((1.0 - quantile).clamp(min=0.0, max=1.0).item())


def split_conformal_probability_threshold(base_logits: Tensor, target_leaf_ids: Tensor, alpha: float) -> float:
    """Fit a split-conformal probability threshold from labeled source pixels.

    The guarantee requires exchangeability between these calibration pixels and
    the deployment population. Cross-dataset transfer uses this threshold as a
    routing mechanism, not as a target-domain coverage guarantee.
    """
    if base_logits.ndim != 4 or target_leaf_ids.shape != base_logits.shape[:1] + base_logits.shape[2:]:
        raise ValueError("base logits and target leaf IDs must align as [B, L, H, W] and [B, H, W]")
    valid = target_leaf_ids >= 0
    if not valid.any():
        raise ValueError("conformal calibration needs at least one valid target pixel")
    probabilities = base_logits.softmax(dim=1).permute(0, 2, 3, 1)[valid]
    targets = target_leaf_ids[valid].to(dtype=torch.long)
    if targets.min() < 0 or targets.max() >= base_logits.shape[1]:
        raise ValueError("calibration target leaf ID is outside logits")
    scores = 1.0 - probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    return split_conformal_probability_threshold_from_scores(scores, alpha)


def conformal_membership(base_logits: Tensor, probability_threshold: float) -> Tensor:
    """Return a non-empty leaf set for every pixel from frozen base logits."""
    if base_logits.ndim != 4:
        raise ValueError("base logits must have shape [B, L, H, W]")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability threshold must lie in [0, 1]")
    probabilities = base_logits.softmax(dim=1)
    membership = probabilities >= probability_threshold
    # Split-conformal thresholds can yield an empty finite-precision set. Adding
    # the top leaf retains a valid taxonomy answer without reducing set coverage.
    top = probabilities.argmax(dim=1, keepdim=True)
    return membership.scatter(1, top, True)


@dataclass(frozen=True)
class CoverageGuardOutput:
    leaf_prediction: Tensor
    node_prediction: Tensor
    routed_to_base: Tensor
    conformal_membership: Tensor


class CoverageGuard(nn.Module):
    """Route potential source-uncovered concepts around a trained flat adapter."""

    def __init__(
        self,
        covered_leaf_mask: Tensor,
        leaf_node_ids: Tensor,
        coverage_node_lookup: Tensor,
    ) -> None:
        super().__init__()
        if covered_leaf_mask.ndim != 1 or covered_leaf_mask.dtype != torch.bool:
            raise ValueError("covered_leaf_mask must be a one-dimensional bool tensor")
        if leaf_node_ids.ndim != 1 or leaf_node_ids.shape != covered_leaf_mask.shape:
            raise ValueError("leaf_node_ids must align to covered_leaf_mask")
        expected_lookup = 1 << covered_leaf_mask.numel()
        if coverage_node_lookup.ndim != 1 or coverage_node_lookup.numel() != expected_lookup:
            raise ValueError("coverage_node_lookup must contain every leaf-set bit pattern")
        self.register_buffer("covered_leaf_mask", covered_leaf_mask.detach().clone(), persistent=False)
        self.register_buffer("leaf_node_ids", leaf_node_ids.detach().to(dtype=torch.long).clone(), persistent=False)
        self.register_buffer("coverage_node_lookup", coverage_node_lookup.detach().to(dtype=torch.long).clone(), persistent=False)

    def forward(self, base_logits: Tensor, adapted_logits: Tensor, probability_threshold: float) -> CoverageGuardOutput:
        if base_logits.shape != adapted_logits.shape or base_logits.ndim != 4:
            raise ValueError("base and adapted logits must share shape [B, L, H, W]")
        if base_logits.shape[1] != self.covered_leaf_mask.numel():
            raise ValueError("logit leaf count does not match coverage mask")
        membership = conformal_membership(base_logits, probability_threshold)
        uncovered_membership = membership & ~self.covered_leaf_mask.view(1, -1, 1, 1)
        routed_to_base = uncovered_membership.any(dim=1)
        base_leaf_prediction = base_logits.argmax(dim=1)
        adapted_leaf_prediction = adapted_logits.argmax(dim=1)
        leaf_prediction = torch.where(routed_to_base, base_leaf_prediction, adapted_leaf_prediction)
        weights = (1 << torch.arange(base_logits.shape[1], device=base_logits.device, dtype=torch.long)).view(1, -1, 1, 1)
        set_code = (membership.to(dtype=torch.long) * weights).sum(dim=1)
        guarded_node_prediction = self.coverage_node_lookup[set_code]
        adapted_node_prediction = self.leaf_node_ids[adapted_leaf_prediction]
        node_prediction = torch.where(routed_to_base, guarded_node_prediction, adapted_node_prediction)
        return CoverageGuardOutput(
            leaf_prediction=leaf_prediction,
            node_prediction=node_prediction,
            routed_to_base=routed_to_base,
            conformal_membership=membership,
        )
