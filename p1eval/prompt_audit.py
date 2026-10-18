"""Frozen configuration helpers for the preregistered herbaceous prompt audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptAuditPlan:
    """Prompt variants that differ at exactly one taxonomy leaf."""

    version: str
    target_leaf: str
    leaf_names: tuple[str, ...]
    candidates: dict[str, str]
    ensemble_members: tuple[str, ...]
    base_prompts: tuple[str, ...]

    @property
    def individual_variant_names(self) -> tuple[str, ...]:
        return tuple(self.candidates)

    @property
    def ensemble_variant_name(self) -> str:
        return "normalized_text_feature_mean"

    @property
    def variant_names(self) -> tuple[str, ...]:
        return self.individual_variant_names + (self.ensemble_variant_name,)

    def prompts_for_candidate(self, candidate_name: str) -> tuple[str, ...]:
        if candidate_name not in self.candidates:
            raise KeyError(f"unknown prompt audit candidate: {candidate_name}")
        target_index = self.leaf_names.index(self.target_leaf)
        prompts = list(self.base_prompts)
        prompts[target_index] = self.candidates[candidate_name]
        return tuple(prompts)


def load_prompt_audit_plan(
    path: Path,
    leaf_names: tuple[str, ...],
    base_prompts: tuple[str, ...],
) -> PromptAuditPlan:
    """Load a validated prompt audit without silently changing other leaf prompts."""
    if len(leaf_names) != len(base_prompts):
        raise ValueError("leaf names and base prompts must have equal length")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    target_leaf = str(document["target_leaf"])
    if target_leaf not in leaf_names:
        raise ValueError(f"prompt audit target leaf is absent from taxonomy: {target_leaf}")
    candidates = {str(name): str(prompt) for name, prompt in document["candidates"].items()}
    if len(candidates) < 2 or any(not prompt.strip() for prompt in candidates.values()):
        raise ValueError("prompt audit needs at least two non-empty candidate prompts")
    ensemble_members = tuple(str(name) for name in document["ensemble_members"])
    if len(ensemble_members) < 2 or len(set(ensemble_members)) != len(ensemble_members):
        raise ValueError("prompt audit ensemble members must be unique and contain at least two candidates")
    missing = set(ensemble_members).difference(candidates)
    if missing:
        raise ValueError(f"prompt audit ensemble has unknown candidates: {sorted(missing)}")
    return PromptAuditPlan(
        version=str(document["version"]),
        target_leaf=target_leaf,
        leaf_names=leaf_names,
        candidates=candidates,
        ensemble_members=ensemble_members,
        base_prompts=base_prompts,
    )
