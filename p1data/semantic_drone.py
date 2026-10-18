"""Semantic Drone Dataset pairing and raw RGB-mask auditing."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from p1data.external import RasterPair
from p1data.splits import identifier_set_sha256


SEMANTIC_DRONE_ROOT_NAME = "semantic_drone_dataset"
SEMANTIC_DRONE_IMAGE_RELATIVE = Path("training_set/images")
SEMANTIC_DRONE_MASK_RELATIVE = Path("training_set/gt/semantic/label_images")
SEMANTIC_DRONE_CLASS_DICT_RELATIVE = Path("training_set/gt/semantic/class_dict.csv")


@dataclass(frozen=True)
class SemanticDroneClass:
    name: str
    rgb: tuple[int, int, int]

    @property
    def packed_rgb(self) -> int:
        red, green, blue = self.rgb
        return (red << 16) | (green << 8) | blue


def _dataset_root(extracted_root: Path) -> Path:
    direct = extracted_root / SEMANTIC_DRONE_ROOT_NAME
    if direct.is_dir():
        return direct
    if extracted_root.name == SEMANTIC_DRONE_ROOT_NAME and extracted_root.is_dir():
        return extracted_root
    raise FileNotFoundError(f"Semantic Drone dataset root is missing: {direct}")


def load_semantic_drone_classes(class_dict_path: Path) -> tuple[SemanticDroneClass, ...]:
    if not class_dict_path.is_file():
        raise FileNotFoundError(f"Semantic Drone class dictionary is missing: {class_dict_path}")
    classes: list[SemanticDroneClass] = []
    seen_names: set[str] = set()
    seen_colors: set[tuple[int, int, int]] = set()
    with class_dict_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        expected_fields = {"name", "r", "g", "b"}
        if set(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"unexpected Semantic Drone class dictionary fields: {reader.fieldnames}")
        for row in reader:
            name = row["name"].strip()
            rgb = tuple(int(row[channel]) for channel in ("r", "g", "b"))
            if not name or name in seen_names:
                raise ValueError(f"duplicate or empty Semantic Drone class name: {name!r}")
            if any(channel < 0 or channel > 255 for channel in rgb):
                raise ValueError(f"invalid Semantic Drone RGB color for {name}: {rgb}")
            if rgb in seen_colors:
                raise ValueError(f"duplicate Semantic Drone RGB color: {rgb}")
            seen_names.add(name)
            seen_colors.add(rgb)
            classes.append(SemanticDroneClass(name=name, rgb=rgb))
    if not classes:
        raise ValueError("Semantic Drone class dictionary is empty")
    return tuple(classes)


def _stem_paths(root: Path, suffix: str) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Semantic Drone directory is missing: {root}")
    paths = tuple(sorted(path for path in root.glob(f"*{suffix}") if path.is_file()))
    if not paths:
        raise FileNotFoundError(f"Semantic Drone directory has no {suffix} files: {root}")
    output = {path.stem: path for path in paths}
    if len(output) != len(paths):
        raise ValueError(f"Semantic Drone directory has duplicate stems: {root}")
    return output


def discover_semantic_drone_pairs(extracted_root: Path) -> tuple[RasterPair, ...]:
    """Pair official training JPGs with same-stem semantic PNG masks."""
    dataset_root = _dataset_root(extracted_root)
    image_paths = _stem_paths(dataset_root / SEMANTIC_DRONE_IMAGE_RELATIVE, ".jpg")
    mask_paths = _stem_paths(dataset_root / SEMANTIC_DRONE_MASK_RELATIVE, ".png")
    missing_masks = sorted(image_paths.keys() - mask_paths.keys())
    extra_masks = sorted(mask_paths.keys() - image_paths.keys())
    if missing_masks or extra_masks:
        raise ValueError(
            "Semantic Drone image/mask pairing mismatch: "
            f"missing_masks={missing_masks[:5]}, extra_masks={extra_masks[:5]}"
        )
    return tuple(
        RasterPair(identifier=identifier, image_path=image_paths[identifier], mask_path=mask_paths[identifier])
        for identifier in sorted(image_paths)
    )


def _packed_rgb_array(rgb_image: np.ndarray) -> np.ndarray:
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"expected RGB image array, got {rgb_image.shape}")
    return (
        (rgb_image[..., 0].astype(np.uint32) << 16)
        | (rgb_image[..., 1].astype(np.uint32) << 8)
        | rgb_image[..., 2].astype(np.uint32)
    )


def audit_semantic_drone_pairs(
    pairs: tuple[RasterPair, ...],
    classes: tuple[SemanticDroneClass, ...],
    *,
    max_samples: int | None = None,
) -> dict:
    """Audit image/mask extent alignment and every registered mask color."""
    if not pairs:
        raise ValueError("Semantic Drone audit needs at least one paired image")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when set")
    selected = pairs if max_samples is None else pairs[:max_samples]
    if not selected:
        raise ValueError("max_samples selected no Semantic Drone pairs")
    class_by_packed = {item.packed_rgb: item for item in classes}
    raw_counts: Counter[int] = Counter()
    image_shapes: set[tuple[int, ...]] = set()
    mask_shapes: set[tuple[int, ...]] = set()
    image_modes: set[str] = set()
    mask_modes: set[str] = set()
    for pair in selected:
        with Image.open(pair.image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"))
            image_shapes.add(tuple(int(value) for value in image.shape))
            image_modes.add(image_file.mode)
        with Image.open(pair.mask_path) as mask_file:
            mask = np.asarray(mask_file.convert("RGB"))
            mask_shapes.add(tuple(int(value) for value in mask.shape))
            mask_modes.add(mask_file.mode)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"{pair.identifier}: image shape {image.shape} does not match mask shape {mask.shape}")
        packed = _packed_rgb_array(mask)
        values, counts = np.unique(packed, return_counts=True)
        raw_counts.update({int(value): int(count) for value, count in zip(values, counts)})
    unknown = sorted(set(raw_counts).difference(class_by_packed))
    if unknown:
        unknown_rgb = [((value >> 16) & 255, (value >> 8) & 255, value & 255) for value in unknown]
        raise ValueError(f"Semantic Drone masks contain unregistered RGB colors: {unknown_rgb[:10]}")
    class_counts = {
        item.name: int(raw_counts.get(item.packed_rgb, 0)) for item in classes
    }
    grass = next((item for item in classes if item.name == "grass"), None)
    if grass is None:
        raise ValueError("Semantic Drone class dictionary has no exact grass class")
    total_pixels = sum(class_counts.values())
    return {
        "version": "semantic-drone-raw-audit-v1",
        "paired_raster_count": len(selected),
        "total_available_pairs": len(pairs),
        "identifier_set_sha256": identifier_set_sha256(pair.identifier for pair in selected),
        "image_shapes": [list(shape) for shape in sorted(image_shapes)],
        "mask_shapes": [list(shape) for shape in sorted(mask_shapes)],
        "image_modes": sorted(image_modes),
        "mask_modes": sorted(mask_modes),
        "class_names": [item.name for item in classes],
        "class_rgb": {item.name: list(item.rgb) for item in classes},
        "raw_rgb_pixel_counts": class_counts,
        "grass_rgb": list(grass.rgb),
        "grass_pixel_count": class_counts["grass"],
        "total_pixels": total_pixels,
        "grass_pixel_fraction": class_counts["grass"] / total_pixels,
        "source_mapping_status": "source_audit_pending_geographic_overlap_review",
    }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
