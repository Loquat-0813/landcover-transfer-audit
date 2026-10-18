"""Model components for P1 taxonomy-aware segmentation experiments."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .taxonomy import TaxonomyLeafIndex, build_taxonomy_leaf_index

__all__ = ["TaxonomyLeafIndex", "build_taxonomy_leaf_index"]


def __getattr__(name: str):
    """Keep model-only utilities importable without eagerly loading YAML support."""
    if name in __all__:
        from .taxonomy import TaxonomyLeafIndex, build_taxonomy_leaf_index

        return {"TaxonomyLeafIndex": TaxonomyLeafIndex, "build_taxonomy_leaf_index": build_taxonomy_leaf_index}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
