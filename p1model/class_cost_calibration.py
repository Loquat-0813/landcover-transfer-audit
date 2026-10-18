"""Native frozen-feature calibration followed by a trainable patch-cost residual."""
from torch import nn
from p1model.calibration import LowRankSemanticCalibration
from p1model.class_cost_residual import ClassCostResidual


class ClassCostCalibration(nn.Module):
    def __init__(self, embedding_dim, arm):
        super().__init__()
        self.adapter = LowRankSemanticCalibration(embedding_dim, 32)
        self.residual = ClassCostResidual(arm)

    def forward(self, pixel_features, text_features):
        return self.residual(self.adapter(pixel_features, text_features), text_features)

    @property
    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
