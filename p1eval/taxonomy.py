"""Load and validate the frozen P1 taxonomy and raw-label mappings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


SEMANTIC_RELATIONS = frozenset({"exact", "coarser_than"})
NON_SEMANTIC_RELATIONS = frozenset({"ambiguous", "ignore", "unknown"})


@dataclass(frozen=True)
class Taxonomy:
    """A rooted, named taxonomy with stable node ordering."""

    version: str
    root: str
    parents: dict[str, str | None]

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self.parents)

    def ancestor_at_level(self, node: str, level: int, levels: dict[str, int]) -> str:
        """Return the ancestor of *node* at ``level`` or fail on an invalid request."""
        if node not in self.parents:
            raise KeyError(f"unknown taxonomy node: {node}")
        if level < 0 or level > levels[node]:
            raise ValueError(f"invalid level {level} for node {node}")
        current = node
        while levels[current] > level:
            parent = self.parents[current]
            if parent is None:
                raise RuntimeError(f"taxonomy node {current} unexpectedly has no parent")
            current = parent
        return current


@dataclass(frozen=True)
class DatasetMapping:
    """A raw-label mapping from one dataset into taxonomy nodes."""

    dataset: str
    scoring_status: str
    raw_id_to_node: dict[int, str]
    exact_raw_ids: frozenset[int]
    coarse_raw_ids: frozenset[int]
    ignored_raw_ids: frozenset[int]
    unknown_raw_ids: frozenset[int]
    unresolved_labels: tuple[str, ...]

    @property
    def ready_for_scoring(self) -> bool:
        return self.scoring_status == "ready" and not self.unresolved_labels

    def adapt(self, raw_mask: np.ndarray, node_to_id: dict[str, int], ignore_index: int = -1) -> np.ndarray:
        """Map a raw label mask to taxonomy IDs, preserving ignored pixels as ``ignore_index``."""
        if raw_mask.ndim != 2:
            raise ValueError(f"expected a 2D raw label mask, got shape {raw_mask.shape}")
        if not self.ready_for_scoring:
            detail = ", ".join(self.unresolved_labels) if self.unresolved_labels else self.scoring_status
            raise ValueError(f"{self.dataset} mapping is not ready for scoring: {detail}")
        result = np.full(raw_mask.shape, ignore_index, dtype=np.int64)
        for raw_id, node in self.raw_id_to_node.items():
            result[raw_mask == raw_id] = node_to_id[node]
        return result

    def unknown_target_and_valid_mask(self, raw_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return unknown targets and the only pixels eligible for unknown rejection."""
        if raw_mask.ndim != 2:
            raise ValueError(f"expected a 2D raw label mask, got shape {raw_mask.shape}")
        known_ids = tuple(self.raw_id_to_node)
        unknown_ids = tuple(self.unknown_raw_ids)
        eligible_ids = known_ids + unknown_ids
        valid = np.isin(raw_mask, eligible_ids)
        unknown = np.isin(raw_mask, unknown_ids)
        return unknown, valid

    def exact_valid_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """Return pixels eligible for leaf-level IoU, excluding coarse annotations."""
        if raw_mask.ndim != 2:
            raise ValueError(f"expected a 2D raw label mask, got shape {raw_mask.shape}")
        return np.isin(raw_mask, tuple(self.exact_raw_ids))


def load_taxonomy(path: Path) -> tuple[Taxonomy, dict[str, int]]:
    """Load taxonomy YAML and return its node levels for efficient aggregation."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    nodes = document["nodes"]
    parents = {name: node["parent"] for name, node in nodes.items()}
    levels = {name: int(node["level"]) for name, node in nodes.items()}
    taxonomy = Taxonomy(version=document["version"], root=document["root"], parents=parents)
    if taxonomy.root not in parents or parents[taxonomy.root] is not None:
        raise ValueError("taxonomy root must exist and have no parent")
    for name, parent in parents.items():
        if parent is not None and parent not in parents:
            raise ValueError(f"{name}: unknown parent {parent}")
        if parent is not None and levels[parent] >= levels[name]:
            raise ValueError(f"{name}: parent level must be lower than child level")
    return taxonomy, levels


def load_dataset_mapping(path: Path, dataset: str, taxonomy: Taxonomy) -> DatasetMapping:
    """Load one dataset's mapping, retaining unresolved labels as an explicit block."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        labels = document["datasets"][dataset]["labels"]
    except KeyError as exc:
        raise KeyError(f"dataset {dataset!r} is not defined in {path}") from exc

    raw_id_to_node: dict[int, str] = {}
    exact_raw_ids: set[int] = set()
    coarse_raw_ids: set[int] = set()
    scoring_status = document["datasets"][dataset].get("scoring_status")
    if scoring_status not in {"ready", "source_audit_pending"}:
        raise ValueError(f"{dataset}: missing or invalid scoring_status")
    ignored_raw_ids: set[int] = set()
    unknown_raw_ids: set[int] = set()
    unresolved: list[str] = []
    for label in labels:
        source_id = label["source_id"]
        relation = label["relation"]
        name = label["source_name"]
        target = label["target"]
        if source_id is None:
            unresolved.append(name)
            continue
        source_id = int(source_id)
        if relation in SEMANTIC_RELATIONS:
            if target not in taxonomy.parents:
                raise ValueError(f"{dataset}/{name}: unknown taxonomy target {target}")
            if source_id in raw_id_to_node:
                raise ValueError(f"{dataset}: duplicate raw ID {source_id}")
            raw_id_to_node[source_id] = target
            if relation == "exact":
                exact_raw_ids.add(source_id)
            else:
                coarse_raw_ids.add(source_id)
        elif relation == "unknown":
            unknown_raw_ids.add(source_id)
        elif relation in {"ambiguous", "ignore"}:
            ignored_raw_ids.add(source_id)
        else:
            raise ValueError(f"{dataset}/{name}: invalid relation {relation}")
    return DatasetMapping(
        dataset=dataset,
        scoring_status=scoring_status,
        raw_id_to_node=raw_id_to_node,
        exact_raw_ids=frozenset(exact_raw_ids),
        coarse_raw_ids=frozenset(coarse_raw_ids),
        ignored_raw_ids=frozenset(ignored_raw_ids),
        unknown_raw_ids=frozenset(unknown_raw_ids),
        unresolved_labels=tuple(unresolved),
    )
