"""Standard ResNet-34 U-Net used by the frozen cross-family replication protocol."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return groups


class _ConvNormReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = _ConvNormReLU(in_channels + skip_channels, out_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((value, skip), dim=1))


class ResNet34UNet(nn.Module):
    """A seven-logit U-Net with an ImageNet-initialized ResNet-34 encoder.

    Decoder normalization uses GroupNorm. Encoder BatchNorm running statistics
    stay in evaluation mode during fine tuning so the fixed B0/B0-half/B1
    batch compositions cannot introduce batch-statistics confounding.
    """

    encoder_weight_id = "ResNet34_Weights.IMAGENET1K_V1"

    def __init__(self, class_count: int, pretrained_encoder: bool = True) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must be at least two")
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as exc:  # pragma: no cover - dependency is verified in the environment manifest.
            raise RuntimeError("ResNet34UNet requires torchvision") from exc
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained_encoder else None
        self.encoder = resnet34(weights=weights)
        self.pretrained_encoder = bool(pretrained_encoder)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)
        self.decode4 = _DecoderBlock(512, 256, 256)
        self.decode3 = _DecoderBlock(256, 128, 128)
        self.decode2 = _DecoderBlock(128, 64, 64)
        self.decode1 = _DecoderBlock(64, 64, 64)
        self.head = nn.Conv2d(64, class_count, kernel_size=1)
        self._freeze_batch_norm_statistics()

    def _freeze_batch_norm_statistics(self) -> None:
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True) -> "ResNet34UNet":
        super().train(mode)
        # Keep affine BatchNorm parameters trainable while fixing their running
        # statistics. This preserves ImageNet initialization without a batch-size arm confound.
        self._freeze_batch_norm_statistics()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")
        height, width = images.shape[-2:]
        if height % 32 or width % 32:
            raise ValueError("U-Net input height and width must be divisible by 32")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("images must be finite RGB tensors in [0, 1]")
        value = (images - self.image_mean) / self.image_std
        stem = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(value)))
        layer1 = self.encoder.layer1(self.encoder.maxpool(stem))
        layer2 = self.encoder.layer2(layer1)
        layer3 = self.encoder.layer3(layer2)
        layer4 = self.encoder.layer4(layer3)
        value = self.decode4(layer4, layer3)
        value = self.decode3(value, layer2)
        value = self.decode2(value, layer1)
        value = self.decode1(value, stem)
        value = F.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)
        return self.head(value)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
