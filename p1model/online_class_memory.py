"""Nontrainable exact-label memory substrate; not wired into a trainer.

Read before the batch loss; commit only after that batch's successful optimizer
step. The caller owns that ordering. Missing scores require availability masking.
"""
import torch
from torch import nn


class OnlineClassMemory(nn.Module):
    def __init__(self, embedding_dim, class_names):
        super().__init__()
        names = tuple(class_names)
        if embedding_dim < 1 or not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
            raise ValueError('positive dimension and unique nonempty class names required')
        self.embedding_dim = embedding_dim
        self.class_names = names
        self.register_buffer('sums', torch.zeros(len(names), embedding_dim, dtype=torch.float64))
        self.register_buffer('counts', torch.zeros(len(names), dtype=torch.float64))
        self.register_buffer('commits', torch.zeros((), dtype=torch.int64))

    def get_extra_state(self):
        return {'version': 1, 'embedding_dim': self.embedding_dim, 'class_names': self.class_names}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('memory class ordering/dimension/version mismatch')

    def load_state_dict(self, state_dict, strict=True):
        # Check metadata before copying any buffers; plain shape checks miss a
        # class-axis permutation. This substrate requires complete state loading.
        if not strict:
            raise ValueError('memory requires strict checkpoint loading')
        self.set_extra_state(state_dict.get('_extra_state'))
        if state_dict['sums'].dtype != torch.float64 or state_dict['counts'].dtype != torch.float64 or state_dict['commits'].dtype != torch.int64:
            raise ValueError('memory checkpoint dtype mismatch')
        if not torch.isfinite(state_dict['sums']).all() or not torch.isfinite(state_dict['counts']).all() or (state_dict['counts'] < 0).any() or (state_dict['commits'] < 0).any():
            raise ValueError('invalid memory checkpoint')
        return super().load_state_dict(state_dict, strict=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Parent Module.load_state_dict bypasses a child's public loader.
        # Validate semantic identity and storage before copying child buffers.
        self.set_extra_state(state_dict.get(prefix + '_extra_state'))
        for name, dtype, shape in [('sums', torch.float64, self.sums.shape),
                                   ('counts', torch.float64, self.counts.shape),
                                   ('commits', torch.int64, self.commits.shape)]:
            value = state_dict.get(prefix + name)
            if not torch.is_tensor(value) or value.dtype != dtype or value.shape != shape or not torch.isfinite(value).all():
                raise ValueError('invalid memory checkpoint buffer ' + name)
            if name != 'sums' and (value < 0).any():
                raise ValueError('negative memory count')
        if (state_dict[prefix + 'sums'][state_dict[prefix + 'counts'] == 0] != 0).any():
            raise ValueError('unsupported class has nonzero memory')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _check(self, features):
        if self.sums.dtype != torch.float64 or self.counts.dtype != torch.float64 or self.commits.dtype != torch.int64:
            raise RuntimeError('memory was dtype-converted and is invalid; reload a fresh instance')
        if features.ndim != 4 or features.shape[1] != self.embedding_dim or min(features.shape) < 1 or not features.is_floating_point():
            raise ValueError('nonempty floating BDHW features required')
        if features.device != self.sums.device or not torch.isfinite(features).all():
            raise ValueError('feature device mismatch or nonfinite features')

    @staticmethod
    def _unit(x, dim):
        # Scale first, so finite large inputs cannot overflow the norm.
        scale = x.abs().amax(dim=dim, keepdim=True)
        scaled = x / torch.where(scale > 0, scale, torch.ones_like(scale))
        norm = scaled.square().sum(dim=dim, keepdim=True).sqrt()
        return scaled / torch.where(norm > 0, norm, torch.ones_like(norm))

    @torch.no_grad()
    def forward(self, features):
        self._check(features)
        x = self._unit(features.detach().to(torch.float64), 1)
        available = (self.counts > 0) & self.sums.ne(0).any(dim=1)
        prototypes = self._unit(self.sums, 1)
        scores = torch.einsum('bdhw,cd->bchw', x, prototypes)
        scores = scores.masked_fill(~available[None, :, None, None], 0)
        return scores.to(features.dtype), available.clone()

    @torch.no_grad()
    def commit(self, features, exact_leaf_ids, step):
        if not self.training:
            raise RuntimeError('evaluation memory is read-only')
        self._check(features)
        if type(step) is not int or step > torch.iinfo(torch.int64).max or step != int(self.commits) + 1:
            raise ValueError('commit must be sequential and unique')
        y = exact_leaf_ids.detach()
        if y.dtype != torch.int64 or y.device != features.device or y.ndim != 3 or y.shape[0] != features.shape[0]:
            raise ValueError('exact leaf IDs must be aligned int64 BHW')
        b, _, h, w = features.shape
        height, width = y.shape[-2:]
        if height < h or width < w or height % h or width % w:
            raise ValueError('label dimensions must be positive exact multiples of feature grid')
        if (y < -1).any() or (y >= len(self.class_names)).any():
            raise ValueError('only exact leaf indices or -1 may enter memory')
        # Full patch area is the denominator, including ignored pixels.
        q = torch.stack([(y == c).reshape(b, h, height//h, w, width//w).to(torch.float64).mean(dim=(2, 4)) for c in range(len(self.class_names))], dim=1)
        x = self._unit(features.detach().to(torch.float64), 1)
        sums = self.sums + torch.einsum('bchw,bdhw->cd', q, x)
        counts = self.counts + q.sum(dim=(0, 2, 3))
        if not torch.isfinite(sums).all() or not torch.isfinite(counts).all():
            raise FloatingPointError('nonfinite proposed memory update')
        # All validation and arithmetic finish before state mutation.
        self.sums.copy_(sums)
        self.counts.copy_(counts)
        self.commits.fill_(step)
