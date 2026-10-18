"""Shared bounded residual using frozen auxiliary class-local visual scores.

ClearCLIP-inspired dense readout is mature. Zero initialization gives initial
native identity only, not a learned performance or fallback guarantee.
"""
import torch
from torch import nn
import torch.nn.functional as F
from p1model.calibration import LowRankSemanticCalibration

ARMS = ('native_info', 'dense_info')
RESIDUAL_SCALE = 1 / 0.07


class DenseInformationResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, logits, auxiliary_features, texts):
        if logits.ndim != 4 or texts.ndim != 2 or logits.shape[1] < 2:
            raise ValueError('requires BCHW logits and at least two class texts')
        if auxiliary_features.ndim != 4 or auxiliary_features.shape[1] != texts.shape[1]:
            raise ValueError('auxiliary/text dimension mismatch')
        if logits.shape[1] != texts.shape[0] or logits.shape[0] != auxiliary_features.shape[0] or logits.shape[2:] != auxiliary_features.shape[2:]:
            raise ValueError('class/batch/grid mismatch')
        visual = F.normalize(auxiliary_features.detach(), dim=1)
        text = F.normalize(texts.detach(), dim=1)
        scores = torch.einsum('bdhw,cd->bchw', visual, text)
        count = scores.shape[1]
        # Each class margin uses its strongest OTHER class; all seven competitors
        # participate, without GT, water/crop IDs, class thresholds or a gate target.
        competitors = scores[:, None].expand(-1, count, -1, -1, -1)
        diagonal = torch.eye(count, dtype=torch.bool, device=scores.device)[None, :, :, None, None]
        other = competitors.masked_fill(diagonal, -float('inf')).max(2).values
        margin = scores - other
        similarity = text @ text.T
        mean_other = (similarity.sum(1) - similarity.diagonal()) / (count - 1)
        semantic = mean_other[None, :, None, None].expand_as(scores)
        inputs = torch.stack((logits, RESIDUAL_SCALE * scores,
                              RESIDUAL_SCALE * margin, semantic), dim=-1)
        gate = self.network(inputs).squeeze(-1).tanh()
        return logits + RESIDUAL_SCALE * gate


class DenseResidualCalibration(nn.Module):
    def __init__(self, embedding_dim, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown information arm')
        self.embedding_dim = embedding_dim
        self.arm = arm
        self.adapter = LowRankSemanticCalibration(embedding_dim, 32)
        self.residual = DenseInformationResidual()

    def forward(self, packed_features, texts):
        if packed_features.ndim != 4 or packed_features.shape[1] != 2 * self.embedding_dim:
            raise ValueError('requires concatenated native and auxiliary features')
        native, auxiliary = packed_features.split(self.embedding_dim, dim=1)
        return self.residual(self.adapter(native, texts), auxiliary, texts)

    @property
    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
