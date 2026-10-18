"""Local prototype for fixed-taxonomy visual-reference competition.

No existing trainer uses this module. Memory is explicitly committed after the
caller's optimizer step; scoring has no access to labels.
"""
import torch
from torch import nn
from p1model.calibration import LowRankSemanticCalibration
from p1model.online_class_memory import OnlineClassMemory

ARMS = ('visual_reference', 'text_reference')
SCALE = 1 / .07


def exact_leaf_targets(target_nodes, leaf_node_ids, node_count):
    """Coarse/ignore nodes remain -1; never reinterpret node IDs as leaf IDs."""
    ids = tuple(leaf_node_ids)
    if target_nodes.dtype != torch.int64 or target_nodes.ndim != 3:
        raise ValueError('int64 BHW node targets required')
    if len(set(ids)) != len(ids) or not ids or any(type(v) is not int or v < 0 or v >= node_count for v in ids):
        raise ValueError('invalid exact leaf node axis')
    if (target_nodes < -1).any() or (target_nodes >= node_count).any():
        raise ValueError('invalid taxonomy target')
    result = torch.full_like(target_nodes, -1)
    for c, node in enumerate(ids):
        result[target_nodes == node] = c
    return result


class VisualMemoryCalibration(nn.Module):
    def __init__(self, embedding_dim, class_names, arm):
        super().__init__()
        if arm not in ARMS or len(class_names) < 2:
            raise ValueError('known arm and at least two classes required')
        self.arm = arm
        self.embedding_dim = embedding_dim
        self.class_names = tuple(class_names)
        self.adapter = LowRankSemanticCalibration(embedding_dim, 32)
        self.memory = OnlineClassMemory(embedding_dim, self.class_names)
        self.network = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def get_extra_state(self):
        return {'version': 1, 'arm': self.arm, 'class_names': self.class_names,
                'embedding_dim': self.embedding_dim, 'scale': SCALE}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('visual-memory model identity mismatch')

    def reference_inputs(self, features, texts):
        self.memory._check(features)
        if texts.shape != (len(self.class_names), self.embedding_dim) or not texts.is_floating_point() or texts.device != features.device or not torch.isfinite(texts).all():
            raise ValueError('finite class-aligned text matrix required')
        with torch.no_grad():
            x = self.memory._unit(features.detach().to(torch.float64), 1)
            t = self.memory._unit(texts.detach().to(torch.float64), 1)
            if not t.ne(0).any(1).all():
                raise ValueError('zero text vector')
            visual, available = self.memory(features)
            text_scores = torch.einsum('bdhw,cd->bchw', x, t).to(features.dtype)
            prototypes = self.memory._unit(self.memory.sums, 1)
            visual_alignment = (prototypes*t).sum(1).to(features.dtype)
            text_alignment = (t*t).sum(1).to(features.dtype)
            # Both arms compute both routes; class coverage is identical.
            u = visual if self.arm == 'visual_reference' else text_scores
            alignment = visual_alignment if self.arm == 'visual_reference' else text_alignment
            u = u.masked_fill(~available[None, :, None, None], 0)
            alignment = alignment.masked_fill(~available, 0)
            c = len(self.class_names)
            other_mask = available[None, :].expand(c, c) & ~torch.eye(c, device=u.device, dtype=torch.bool)
            competitors = u[:, None].expand(-1, c, -1, -1, -1)
            other = competitors.masked_fill(~other_mask[None, :, :, None, None], -torch.inf).amax(2)
            has_other = other_mask.any(1)[None, :, None, None]
            # Replace the all-missing reduction before subtracting/MLP.
            safe_other = torch.where(has_other, other, u)
            margin = u - safe_other
            mask = available & (available.sum() >= 2)
            return u, margin, alignment, mask

    def forward(self, features, texts):
        u, margin, alignment, mask = self.reference_inputs(features, texts)
        z = self.adapter(features, texts)
        c = z.shape[1]
        other = z[:, None].expand(-1, c, -1, -1, -1).masked_fill(
            torch.eye(c, device=z.device, dtype=torch.bool)[None, :, :, None, None], -torch.inf).amax(2)
        native_margin = z-other  # Deliberately retain native adapter gradients.
        a = alignment[None, :, None, None].expand_as(z)
        inputs = torch.stack((native_margin, SCALE*u, SCALE*margin, a), dim=-1)
        correction = SCALE*self.network(inputs).squeeze(-1).tanh()
        # Always execute the head: even empty memory yields zero grad tensors,
        # with ordinary AdamW state/decay semantics, not optimizer freezing.
        return z + correction*mask[None, :, None, None].to(z.dtype)

    def commit_memory(self, features, exact_leaf_ids, step):
        self.memory.commit(features, exact_leaf_ids, step)

    @property
    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
