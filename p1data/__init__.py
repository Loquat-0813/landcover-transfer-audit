"""Dataset adapters for the active P1 taxonomy-aware land-cover track."""

# Keep package import light: several audit utilities need only path and split
# helpers, whereas the LoveDA adapter also requires the YAML taxonomy runtime.
from .splits import split_identifiers_for_calibration, stable_identifier_bucket

__all__ = [
    "LoveDAPair",
    "center_aligned_crop",
    "discover_loveda_pairs",
    "load_loveda_example",
    "random_aligned_crop",
    "split_identifiers_for_calibration",
    "stable_identifier_bucket",
]


def __getattr__(name: str):
    if name in {
        "LoveDAPair",
        "center_aligned_crop",
        "discover_loveda_pairs",
        "load_loveda_example",
        "random_aligned_crop",
    }:
        from . import loveda

        return getattr(loveda, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
