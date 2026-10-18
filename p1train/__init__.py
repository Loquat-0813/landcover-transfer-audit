"""Training and evaluation helpers for registered P1 experiment protocols."""

__all__ = ["evaluate_loveda_tiled", "train_calibration_epoch"]


def __getattr__(name: str):
    """Avoid importing PyTorch for small protocol-only utilities."""
    if name in __all__:
        from .loveda_e1 import evaluate_loveda_tiled, train_calibration_epoch

        return {"evaluate_loveda_tiled": evaluate_loveda_tiled, "train_calibration_epoch": train_calibration_epoch}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
