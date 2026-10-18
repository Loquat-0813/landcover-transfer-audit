"""Bounded training-only crop retries; source raster order is preserved."""
from __future__ import annotations
import hashlib

class CoverageCropDataset:
    def __init__(self, base, indices=None, min_pixels=1024, attempts=8):
        if min_pixels < 1 or attempts < 1:
            raise ValueError('positive coverage threshold and retry bound required')
        self.base = base
        self.indices = tuple(range(len(base))) if indices is None else tuple(indices)
        if not self.indices or len(set(self.indices)) != len(self.indices) or any(i < 0 or i >= len(base) for i in self.indices):
            raise ValueError('invalid source index population')
        self.min_pixels, self.attempts, self.epoch = min_pixels, attempts, 0

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('negative sampling nonce')
        self.epoch = epoch

    def __getitem__(self, index):
        source_index = self.indices[index]
        best, best_count, used = None, -1, 0
        try:
            for attempt in range(self.attempts):
                # Attempt zero exactly preserves the original crop schedule.
                nonce = self.epoch if attempt == 0 else int.from_bytes(hashlib.sha256(
                    f'coverage-v2:{self.epoch}:{source_index}:{attempt}'.encode()).digest()[:4], 'big') + 2**32
                self.base.set_epoch(nonce)
                item = self.base[source_index]
                count = int(item['target'].ne(-1).sum())
                used += 1
                if count > best_count:
                    best, best_count = item, count
                if count >= self.min_pixels:
                    break
        finally:
            self.base.set_epoch(self.epoch)
        return {**best, 'coverage_sampling': {'source_index': source_index, 'attempts': used,
                    'valid_pixels': best_count, 'coverage_met': best_count >= self.min_pixels}}

def stage_two_source_split(identifiers, seed=19, holdout_count=40):
    """Predeclared identifier split; teacher may already have seen all images."""
    if len(set(identifiers)) != len(identifiers) or not 0 < holdout_count < len(identifiers):
        raise ValueError('unique identifiers and nonempty train/holdout required')
    order = sorted(range(len(identifiers)), key=lambda i: hashlib.sha256(
        f'p1-source-v2:{seed}:{identifiers[i]}'.encode()).hexdigest())
    heldout = frozenset(order[:holdout_count])
    return tuple(i for i in range(len(identifiers)) if i not in heldout), tuple(sorted(heldout))
