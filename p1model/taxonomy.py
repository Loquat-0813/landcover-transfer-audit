"""Translate the versioned P1 taxonomy into leaf-class supervision masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p1eval.taxonomy import Taxonomy


@dataclass(frozen=True)
class TaxonomyLeafIndex:
    """Stable taxonomy IDs and descendant-leaf memberships for model supervision."""

    node_names: tuple[str, ...]
    leaf_names: tuple[str, ...]
    node_to_id: dict[str, int]
    descendant_leaf_mask: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.node_names), len(self.leaf_names))
        if self.descendant_leaf_mask.shape != expected:
            raise ValueError(f"descendant mask shape must be {expected}, got {self.descendant_leaf_mask.shape}")
        if not self.descendant_leaf_mask.dtype == np.bool_:
            raise TypeError("descendant leaf mask must be boolean")


def build_taxonomy_leaf_index(taxonomy: Taxonomy) -> TaxonomyLeafIndex:
    """Build leaf memberships where each node is supervised by its descendant leaves.

    The leaf ordering follows the versioned YAML node ordering. This keeps prompt
    embeddings, logits, checkpoints, and reported class names aligned.
    """
    node_names = taxonomy.nodes
    node_to_id = {name: index for index, name in enumerate(node_names)}
    children = {name: [] for name in node_names}
    for child, parent in taxonomy.parents.items():
        if parent is not None:
            children[parent].append(child)
    leaf_names = tuple(name for name in node_names if not children[name])
    leaf_to_id = {name: index for index, name in enumerate(leaf_names)}
    mask = np.zeros((len(node_names), len(leaf_names)), dtype=bool)
    for leaf_name, leaf_id in leaf_to_id.items():
        current: str | None = leaf_name
        while current is not None:
            mask[node_to_id[current], leaf_id] = True
            current = taxonomy.parents[current]
    if not mask.any(axis=1).all():
        empty = [node_names[index] for index, values in enumerate(mask) if not values.any()]
        raise ValueError(f"taxonomy has nodes without descendant leaves: {empty}")
    return TaxonomyLeafIndex(
        node_names=node_names,
        leaf_names=leaf_names,
        node_to_id=node_to_id,
        descendant_leaf_mask=mask,
    )
