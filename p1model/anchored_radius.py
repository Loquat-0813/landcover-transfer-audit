"""Anchor class-vector magnitudes while allowing calibration directions to learn."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from p1model.calibration import LowRankSemanticCalibration


class AnchoredRadiusCalibration(nn.Module):
    """A stage-two wrapper with checkpoint-anchored per-class radial gains.

    Cosine classifiers motivate controlling class-vector magnitude. Here the
    anchor is the actual initial calibrated vector norm, rather than a common
    radius. This preserves the initial forward exactly when gains are one.
    Protected vectors may still rotate; protected logits, probabilities and
    IoU are not invariant. Unprotected new-class scores remain the native
    calibration output. This is not a pixel gate or a reproduced LUCIR model.
    """

    def __init__(self, embedding_dim: int, rank: int, protected_leaf_mask: Tensor) -> None:
        super().__init__()
        if protected_leaf_mask.ndim != 1 or protected_leaf_mask.dtype != torch.bool:
            raise ValueError('protected_leaf_mask must be a one-dimensional bool tensor')
        if not protected_leaf_mask.numel():
            raise ValueError('an explicit nonempty leaf mask is required')
        self.adapter=LowRankSemanticCalibration(embedding_dim,rank)
        self.register_buffer('protected_leaf_mask',protected_leaf_mask.detach().clone())
        self.register_buffer('anchor_radii',torch.zeros(protected_leaf_mask.numel()))
        self.register_buffer('anchor_text_features',torch.zeros(protected_leaf_mask.numel(),embedding_dim))
        self.register_buffer('anchor_ready',torch.tensor(False))

    def effective_classifier_weights(self,text_features: Tensor) -> Tensor:
        if text_features.ndim!=2 or text_features.shape!=(self.anchor_radii.numel(),self.adapter.embedding_dim):
            raise ValueError('text bank does not match the explicit leaf/dimension contract')
        if self.adapter.adapted_leaf_mask.numel()!=0 or self.adapter.freeze_logit_scale:
            raise ValueError('this wrapper requires the documented native unmasked learned-scale adapter')
        text=F.normalize(text_features.detach(),dim=1)
        scale=self.adapter.logit_scale.clamp(max=math.log(100.)).exp()
        residual=(text@self.adapter.text_projection.weight.T)@self.adapter.image_projection.weight
        return scale*text+self.adapter.residual_scale/math.sqrt(self.adapter.rank)*residual

    @torch.no_grad()
    def set_anchor_from_current(self,text_features: Tensor) -> None:
        if bool(self.anchor_ready):
            raise RuntimeError('anchors may be initialized exactly once, before training')
        radii=self.effective_classifier_weights(text_features).norm(dim=1)
        if not torch.isfinite(radii).all() or (radii<=1e-12).any():
            raise ValueError('nonfinite or degenerate initial class radius')
        self.anchor_radii.copy_(radii)
        self.anchor_text_features.copy_(text_features.detach())
        self.anchor_ready.fill_(True)

    def radial_gains(self,text_features: Tensor) -> Tensor:
        if not bool(self.anchor_ready):
            raise RuntimeError('initial checkpoint/text anchors must be captured before forward')
        if not torch.equal(text_features.detach(),self.anchor_text_features):
            raise ValueError('text bank changed after radius anchoring')
        radii=self.effective_classifier_weights(text_features).norm(dim=1)
        if not torch.isfinite(radii).all() or (radii<=1e-12).any():
            raise ValueError('nonfinite/degenerate current class radius; do not silently clip')
        return torch.where(self.protected_leaf_mask,self.anchor_radii/radii,torch.ones_like(radii))

    def forward(self,pixel_features: Tensor,text_features: Tensor) -> Tensor:
        raw=self.adapter(pixel_features,text_features)
        return raw*self.radial_gains(text_features).view(1,-1,1,1)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
