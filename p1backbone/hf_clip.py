"""Frozen Hugging Face CLIP patch encoder used by the E0/E1 baseline."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FrozenCLIPPatchEncoder(nn.Module):
    """Expose CLIP patch and text embeddings while preventing backbone gradients.

    The dependency is intentionally imported only during construction so protocol
    and data-audit scripts remain usable before Transformers is installed.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__()
        try:
            from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError(
                "FrozenCLIPPatchEncoder requires transformers. Install the pinned "
                "experiment dependencies before running E0/E1."
            ) from exc
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = CLIPImageProcessor.from_pretrained(model_id)
        self.register_buffer("image_mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(processor.image_std).view(1, 3, 1, 1), persistent=False)
        self.patch_size = int(self.model.config.vision_config.patch_size)
        self.embedding_dim = int(self.model.projection_dim)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenCLIPPatchEncoder":
        """Keep the frozen backbone in eval mode even when the outer model trains."""
        super().train(False)
        self.model.eval()
        return self

    def _normalise(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError(f"image dimensions must be divisible by CLIP patch size {self.patch_size}")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("images must be finite RGB values in [0, 1]")
        return (images - self.image_mean) / self.image_std

    @torch.inference_mode()
    def encode_image(self, images: Tensor) -> Tensor:
        """Return projected patch features as `[B, D, H/patch, W/patch]`."""
        pixel_values = self._normalise(images)
        output = self.model.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=True)
        patch_tokens = output.last_hidden_state[:, 1:, :]
        side = math.isqrt(patch_tokens.shape[1])
        if side * side != patch_tokens.shape[1]:
            raise RuntimeError("CLIP patch sequence is not square")
        projected = self.model.visual_projection(patch_tokens)
        return projected.transpose(1, 2).reshape(images.shape[0], self.embedding_dim, side, side)

    @torch.inference_mode()
    def encode_text(self, prompts: tuple[str, ...] | list[str]) -> Tensor:
        """Return projected frozen CLIP text features in the same dimension as patches."""
        if not prompts:
            raise ValueError("at least one prompt is required")
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(list(prompts), padding=True, truncation=True, return_tensors="pt")
        tokens = {name: value.to(device) for name, value in tokens.items()}
        return self.model.get_text_features(**tokens)
