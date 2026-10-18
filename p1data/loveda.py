"""Read LoveDA image/mask pairs without embedding model-specific preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from p1eval.taxonomy import DatasetMapping


SPLIT_DIRECTORIES = {
    "train": ("train", "Train"),
    "val": ("val", "Val"),
}


@dataclass(frozen=True)
class LoveDAPair:
    """A verified LoveDA image and its matching raw semantic mask."""

    identifier: str
    image_path: Path
    mask_path: Path


def discover_loveda_pairs(root: Path, split: str) -> tuple[LoveDAPair, ...]:
    """Return a deterministic, complete pairing for an extracted LoveDA split."""
    if split not in SPLIT_DIRECTORIES:
        raise ValueError(f"split must be one of {sorted(SPLIT_DIRECTORIES)}, got {split!r}")
    split_root = root.joinpath(*SPLIT_DIRECTORIES[split])
    if not split_root.is_dir():
        raise FileNotFoundError(f"LoveDA split directory not found: {split_root}")
    pairs: list[LoveDAPair] = []
    for region in sorted(path for path in split_root.iterdir() if path.is_dir()):
        image_directory = region / "images_png"
        mask_directory = region / "masks_png"
        if not image_directory.is_dir() or not mask_directory.is_dir():
            raise FileNotFoundError(f"{region}: expected images_png and masks_png directories")
        images = {path.stem: path for path in image_directory.glob("*.png")}
        masks = {path.stem: path for path in mask_directory.glob("*.png")}
        missing = sorted(images.keys() - masks.keys())
        extra = sorted(masks.keys() - images.keys())
        if missing or extra:
            raise ValueError(f"{region}: missing masks={missing[:3]}, extra masks={extra[:3]}")
        pairs.extend(
            LoveDAPair(
                identifier=f"{region.name}/{stem}",
                image_path=images[stem],
                mask_path=masks[stem],
            )
            for stem in sorted(images)
        )
    if not pairs:
        raise ValueError(f"no paired LoveDA examples found below {split_root}")
    return tuple(pairs)


def load_loveda_example(pair: LoveDAPair, mapping: DatasetMapping, node_to_id: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB uint8 imagery and an int64 taxonomy mask for one paired example."""
    with Image.open(pair.image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(pair.mask_path) as mask:
        raw_mask = np.asarray(mask.convert("L"), dtype=np.uint8)
    if rgb.shape[:2] != raw_mask.shape:
        raise ValueError(f"{pair.identifier}: image shape {rgb.shape[:2]} does not match mask shape {raw_mask.shape}")
    return rgb, mapping.adapt(raw_mask, node_to_id)


def random_aligned_crop(
    image: np.ndarray,
    target: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and mask with one shared random origin; no resize is performed."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be RGB [H, W, 3], got {image.shape}")
    if target.ndim != 2 or image.shape[:2] != target.shape:
        raise ValueError("target must be [H, W] and aligned with image")
    height, width = target.shape
    if crop_size < 1 or crop_size > height or crop_size > width:
        raise ValueError(f"crop_size {crop_size} is incompatible with image size {(height, width)}")
    top = int(rng.integers(0, height - crop_size + 1))
    left = int(rng.integers(0, width - crop_size + 1))
    return image[top : top + crop_size, left : left + crop_size], target[top : top + crop_size, left : left + crop_size]


def center_aligned_crop(image: np.ndarray, target: np.ndarray, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one centered, aligned crop for deterministic validation previews."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be RGB [H, W, 3], got {image.shape}")
    if target.ndim != 2 or image.shape[:2] != target.shape:
        raise ValueError("target must be [H, W] and aligned with image")
    height, width = target.shape
    if crop_size < 1 or crop_size > height or crop_size > width:
        raise ValueError(f"crop_size {crop_size} is incompatible with image size {(height, width)}")
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[top : top + crop_size, left : left + crop_size], target[top : top + crop_size, left : left + crop_size]
