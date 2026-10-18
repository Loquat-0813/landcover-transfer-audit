"""Versioned taxonomy prompt loading with explicit leaf-order validation."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_leaf_prompts(path: Path, leaf_names: tuple[str, ...]) -> tuple[str, ...]:
    """Load exactly one configured text prompt for every taxonomy leaf."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    configured = document["leaf_prompts"]
    if set(configured) != set(leaf_names):
        raise ValueError("prompt configuration must contain every and only taxonomy leaf")
    prompts = tuple(str(configured[name]) for name in leaf_names)
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("taxonomy prompts must be non-empty")
    return prompts
