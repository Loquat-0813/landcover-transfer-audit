"""Frozen last-layer readout diagnostic inspired by ClearCLIP (ECCV2024).

Independent HF implementation of paper Eq6, not a full-system reproduction.
Historical hf_clip.py is unmodified. No residual training is admitted here.
"""
from __future__ import annotations

import torch
from torch import Tensor


def attention_readout(layer, hidden: Tensor, kind: str) -> Tensor:
    """Attention output only; keep CLS during attention, omit final RC/FFN.

    qk is a correctness/decomposition control; qq is the fixed paper default.
    No masks/dropout allowed: frozen CLIP vision attention in eval mode.
    """
    if kind not in ('qk', 'qq'):
        raise ValueError('kind must be qk or qq')
    attn = layer.self_attn
    if layer.training or attn.training:
        raise ValueError('requires eval mode')
    x = layer.layer_norm1(hidden)
    batch, length, width = x.shape
    heads = attn.num_heads
    if width % heads:
        raise ValueError('width must be divisible by heads')
    dim = width // heads
    def split(z):
        return z.reshape(batch, length, heads, dim).transpose(1, 2)
    q = split(attn.q_proj(x))
    keys = split(attn.k_proj(x)) if kind == 'qk' else q
    v = split(attn.v_proj(x))
    probabilities = ((q * (dim ** -0.5)) @ keys.transpose(-1, -2)).softmax(-1)
    values = (probabilities @ v).transpose(1, 2).reshape(batch, length, width)
    return attn.out_proj(values)


@torch.inference_mode()
def encode_readouts(backbone, images: Tensor) -> dict[str, Tensor]:
    """One untouched vision forward plus fixed extra frozen readout branches.

    native: historical unnormalized last-token projection, unchanged.
    norm_control: same final tokens with vision post-LN before projection.
    dense_qq: final QQ attention output, post-LN and visual projection.
    An added branch has extra compute even though it has no trainable weights.
    """
    if backbone.training or any(p.requires_grad for p in backbone.model.parameters()):
        raise ValueError('backbone must be frozen and eval')
    height, width = images.shape[-2:]
    vision = backbone.model.vision_model
    output = vision(pixel_values=backbone._normalise(images),
                    interpolate_pos_encoding=True, output_hidden_states=True)
    final = output.last_hidden_state
    penultimate = output.hidden_states[-2]
    dense = attention_readout(vision.encoder.layers[-1], penultimate, 'qq')
    tokens = {'native': final, 'norm_control': vision.post_layernorm(final),
              'dense_qq': vision.post_layernorm(dense)}
    rows, columns = height // backbone.patch_size, width // backbone.patch_size
    result = {}
    for name, value in tokens.items():
        patches = backbone.model.visual_projection(value[:, 1:, :])
        if patches.shape[1] != rows * columns:
            raise ValueError('patch grid mismatch')
        result[name] = patches.transpose(1, 2).reshape(images.shape[0], backbone.embedding_dim, rows, columns)
    return result
