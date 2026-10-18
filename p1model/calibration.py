"""Low-rank text-conditioned calibration over frozen vision-language cost maps."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LowRankSemanticCalibration(nn.Module):
    """Adapt frozen pixel/text features without replacing open-vocabulary logits.

    Inputs are deliberately detached in ``forward``. The image and text encoders
    must remain frozen; only the two low-rank projections and scalar logit terms
    are trainable. New text prompts remain usable because their embeddings are
    projected with the same shared text projection.
    """

    def __init__(
        self,
        embedding_dim: int,
        rank: int = 32,
        base_logit_scale: float = 1 / 0.07,
        residual_scale: float = 0.05,
        adapted_leaf_mask: Tensor | None = None,
        freeze_logit_scale: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or rank < 1:
            raise ValueError("embedding_dim and rank must be positive")
        if base_logit_scale <= 0 or residual_scale <= 0:
            raise ValueError("logit scales must be positive")
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.base_logit_scale = float(base_logit_scale)
        self.freeze_logit_scale = bool(freeze_logit_scale)
        if adapted_leaf_mask is None:
            adapted_leaf_mask = torch.empty(0, dtype=torch.bool)
        if adapted_leaf_mask.ndim != 1:
            raise ValueError("adapted_leaf_mask must be one-dimensional when provided")
        # This protocol setting is reconstructed from the run config at evaluation,
        # so old E1 checkpoints remain compatible with the model state dictionary.
        self.register_buffer("adapted_leaf_mask", adapted_leaf_mask.detach().to(dtype=torch.bool).clone(), persistent=False)
        self.image_projection = nn.Linear(embedding_dim, rank, bias=False)
        self.text_projection = nn.Linear(embedding_dim, rank, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(base_logit_scale), dtype=torch.float32))
        if self.freeze_logit_scale:
            self.logit_scale.requires_grad_(False)
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale, dtype=torch.float32))
        nn.init.normal_(self.image_projection.weight, std=0.01)
        nn.init.normal_(self.text_projection.weight, std=0.01)

    def forward(self, pixel_features: Tensor, text_features: Tensor) -> Tensor:
        """Return leaf logits with shape ``[batch, leaf_class, height, width]``.

        ``pixel_features`` must be `[B, D, H, W]` and ``text_features`` `[L, D]`.
        Both can be taken directly from a frozen VLM or loaded from a feature
        cache. Their gradients are stopped at this module boundary.
        """
        if pixel_features.ndim != 4:
            raise ValueError(f"pixel_features must have shape [B, D, H, W], got {tuple(pixel_features.shape)}")
        if text_features.ndim != 2:
            raise ValueError(f"text_features must have shape [L, D], got {tuple(text_features.shape)}")
        if pixel_features.shape[1] != self.embedding_dim or text_features.shape[1] != self.embedding_dim:
            raise ValueError("pixel/text embedding dimension does not match calibration module")
        image = F.normalize(pixel_features.detach(), dim=1)
        text = F.normalize(text_features.detach(), dim=1)
        base_cost = torch.einsum("bdhw,ld->blhw", image, text)
        image_low_rank = self.image_projection(image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        text_low_rank = self.text_projection(text)
        residual = torch.einsum("brhw,lr->blhw", image_low_rank, text_low_rank) / math.sqrt(self.rank)
        if self.freeze_logit_scale:
            calibrated_scale = self.base_logit_scale
        else:
            calibrated_scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        calibrated = calibrated_scale * base_cost + self.residual_scale * residual
        if self.adapted_leaf_mask.numel() == 0:
            return calibrated
        if self.adapted_leaf_mask.numel() != text_features.shape[0]:
            raise ValueError("adapted_leaf_mask length must equal the text leaf count")
        frozen = self.base_logit_scale * base_cost
        mask = self.adapted_leaf_mask.view(1, -1, 1, 1)
        # Unseen leaves receive exactly the immutable E0 logits. This prevents
        # shared low-rank updates and the learned global temperature from erasing
        # source-uncovered text concepts.
        return torch.where(mask, calibrated, frozen)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
