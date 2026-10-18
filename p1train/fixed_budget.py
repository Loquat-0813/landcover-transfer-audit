"""Deterministic source sampling for preregistered fixed-update experiments."""

from __future__ import annotations

import numpy as np


def loveda_positions_for_update(arm: str, update: int) -> tuple[int, ...]:
    """Return deterministic LoveDA stream positions for one source-mixture update."""
    if update < 0:
        raise ValueError("update must be non-negative")
    if arm == "b0":
        return (2 * update, 2 * update + 1)
    if arm in {"b0_half", "b0_half_batch2", "b1", "b1_shuffle"}:
        return (update,)
    raise ValueError(f"unsupported fixed-budget arm: {arm}")


class CyclingPermutation:
    """Return a reproducible shuffled source order, cycling without replacement."""

    def __init__(self, length: int, seed: int, stream: int) -> None:
        if length < 1:
            raise ValueError("cycling permutation length must be positive")
        if stream < 0:
            raise ValueError("cycling permutation stream must be non-negative")
        self.length = length
        self.seed = seed
        self.stream = stream
        self._cycle = -1
        self._permutation = np.empty(0, dtype=np.int64)

    def at(self, position: int) -> int:
        if position < 0:
            raise ValueError("cycling permutation position must be non-negative")
        cycle, offset = divmod(position, self.length)
        if cycle != self._cycle:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.stream, cycle]))
            self._permutation = rng.permutation(self.length)
            self._cycle = cycle
        return int(self._permutation[offset])
