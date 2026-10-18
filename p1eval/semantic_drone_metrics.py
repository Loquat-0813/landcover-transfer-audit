"""Small dependency-free metrics used by Semantic Drone confirmation."""

from __future__ import annotations


def grass_iou(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    union = true_positive + false_positive + false_negative
    return None if union == 0 else true_positive / union
