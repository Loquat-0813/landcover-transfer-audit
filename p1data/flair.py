"""FLAIR #1 raster pairing and raw-label auditing for the narrow B1 intervention."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import tifffile

from p1data.external import RasterPair
from p1eval.taxonomy import DatasetMapping
from p1data.splits import identifier_set_sha256


# These official ``train-val`` domains are frozen by the B1 amendment.  Keep
# them in code as well as in the protocol so a command cannot silently use an
# archive from an unregistered partition.
FLAIR_TRAIN_DOMAINS = (
    "D006_2020", "D007_2020", "D008_2019", "D009_2019", "D013_2020", "D016_2020", "D017_2018",
    "D021_2020", "D023_2020", "D030_2021", "D032_2019", "D033_2021", "D034_2021", "D035_2020",
    "D038_2021", "D041_2021", "D044_2020", "D046_2019", "D049_2020", "D051_2019", "D052_2019",
    "D055_2018", "D060_2021", "D063_2019", "D070_2020", "D072_2019", "D074_2020", "D078_2021",
    "D080_2021", "D081_2020", "D086_2020", "D091_2021",
)
FLAIR_VALIDATION_DOMAINS = (
    "D004_2021", "D014_2020", "D029_2021", "D031_2019", "D058_2020", "D066_2021", "D067_2021",
    "D077_2021",
)
FLAIR_ALL_DOMAINS = tuple(sorted(FLAIR_TRAIN_DOMAINS + FLAIR_VALIDATION_DOMAINS))


def activate_flair_b1_mapping(mapping: DatasetMapping, source_audit: dict) -> DatasetMapping:
    """Enable the narrow mapping only after the separately persisted audit passes.

    The YAML deliberately remains ``source_audit_pending`` so arbitrary code
    cannot consume FLAIR labels.  This local state transition is allowed only
    for the already-audited raw-ID-10 mapping and leaves every ignored source
    class untouched.
    """
    if mapping.dataset != "flair_b1" or mapping.scoring_status != "source_audit_pending":
        raise ValueError("expected the provisional flair_b1 mapping")
    if mapping.raw_id_to_node != {10: "herbaceous_vegetation"} or mapping.exact_raw_ids != frozenset({10}):
        raise ValueError("FLAIR B1 mapping must remain limited to exact raw ID 10")
    if source_audit.get("source_mapping_status") != "source_audit_ready":
        raise ValueError("FLAIR B1 mapping needs a completed source readiness audit")
    if source_audit.get("herbaceous_raw_id") != 10 or source_audit.get("train", {}).get("herbaceous_pixel_count", 0) <= 0:
        raise ValueError("source readiness audit does not support raw ID 10")
    return replace(mapping, scoring_status="ready")


def _relative_stem(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if not relative.suffix:
        raise ValueError(f"{path}: expected a file extension")
    return relative.with_suffix("").as_posix()


def _discover_files(root: Path, pattern: str, role: str) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"FLAIR {role} directory is missing: {root}")
    paths = tuple(sorted(path for path in root.glob(pattern) if path.is_file()))
    if not paths:
        raise FileNotFoundError(f"FLAIR {role} directory has no files matching {pattern!r}: {root}")
    output = {_relative_stem(path, root): path for path in paths}
    if len(output) != len(paths):
        raise ValueError(f"FLAIR {role} paths have duplicate relative stems")
    return output


def _official_pair_identifier(identifier: str, role: str) -> str:
    """Normalize the official FLAIR ``IMG_<id>``/``MSK_<id>`` filenames.

    FLAIR's own loader discovers ``IMG*.tif`` and ``MSK*.tif`` in each
    domain, then aligns them using the suffix after the final underscore.
    Preserve all relative directories while replacing only that role-specific
    filename prefix.  Non-FLAIR-style fixture names remain unchanged so that
    mismatch checks remain useful for arbitrary explicit glob inputs.
    """
    relative = Path(identifier)
    prefix = "IMG_" if role == "image" else "MSK_"
    stem = relative.name
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return relative.with_name(stem[len(prefix) :]).as_posix()
    return relative.as_posix()


def discover_flair_pairs(
    image_root: Path,
    mask_root: Path,
    *,
    image_glob: str = "**/*",
    mask_glob: str = "**/*",
) -> tuple[RasterPair, ...]:
    """Pair FLAIR imagery and masks by their official patch identifiers.

    The official archive layout is intentionally supplied at the command line
    after toy-subset inspection. This avoids baking an unverified vendor layout
    into the B1 protocol.
    """
    raw_images = _discover_files(image_root, image_glob, "image")
    raw_masks = _discover_files(mask_root, mask_glob, "mask")
    images = {_official_pair_identifier(identifier, "image"): path for identifier, path in raw_images.items()}
    masks = {_official_pair_identifier(identifier, "mask"): path for identifier, path in raw_masks.items()}
    if len(images) != len(raw_images) or len(masks) != len(raw_masks):
        raise ValueError("FLAIR filenames have duplicate normalized patch identifiers")
    missing_masks = sorted(images.keys() - masks.keys())
    extra_masks = sorted(masks.keys() - images.keys())
    if missing_masks or extra_masks:
        raise ValueError(
            "FLAIR image/mask pairing mismatch: "
            f"missing_masks={missing_masks[:5]}, extra_masks={extra_masks[:5]}"
        )
    return tuple(
        RasterPair(identifier, images[identifier], masks[identifier]) for identifier in sorted(images)
    )


def discover_flair_domain_pairs(extracted_root: Path, domains: tuple[str, ...]) -> tuple[RasterPair, ...]:
    """Discover official FLAIR pairs while retaining an archive-domain prefix.

    Each official domain archive contains sibling ``aerial/`` and ``labels/``
    directories.  Extracting every archive into its own directory is both
    resumable and prevents patches with the same zone/tile name from different
    domains being conflated.  The returned identifiers begin with the domain,
    which makes the frozen train/validation split auditable downstream.
    """
    if not domains:
        raise ValueError("FLAIR domain discovery needs at least one domain")
    if len(domains) != len(set(domains)):
        raise ValueError("FLAIR domain list contains duplicate names")
    unknown = sorted(set(domains).difference(FLAIR_ALL_DOMAINS))
    if unknown:
        raise ValueError(f"FLAIR domain list contains unregistered domains: {unknown}")
    pairs: list[RasterPair] = []
    for domain in domains:
        domain_root = extracted_root / domain
        domain_pairs = discover_flair_pairs(
            domain_root / "aerial",
            domain_root / "labels",
            image_glob="**/*.tif",
            mask_glob="**/*.tif",
        )
        pairs.extend(
            RasterPair(
                identifier=f"{domain}/{pair.identifier}",
                image_path=pair.image_path,
                mask_path=pair.mask_path,
            )
            for pair in domain_pairs
        )
    identifiers = [pair.identifier for pair in pairs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("FLAIR domains contain duplicate normalized patch identifiers")
    return tuple(sorted(pairs, key=lambda pair: pair.identifier))


def _read_flair_image_metadata(path: Path) -> tuple[tuple[int, ...], str]:
    """Read GeoTIFF shape and dtype without decoding the aerial pixel data."""
    with tifffile.TiffFile(path) as image_file:
        if not image_file.series:
            raise ValueError(f"{path}: GeoTIFF has no image series")
        series = image_file.series[0]
        return tuple(int(dimension) for dimension in series.shape), str(np.dtype(series.dtype))


def _read_flair_mask(path: Path) -> np.ndarray:
    """Decode the single-band raw FLAIR label raster without remapping it."""
    return np.asarray(tifffile.imread(path))


def load_flair_example(
    pair: RasterPair,
    mapping: DatasetMapping,
    node_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the preregistered RGB subset and the narrowly mapped FLAIR target.

    The official FLAIR card defines aerial band order as Red, Green, Blue,
    NIR, and nDSM.  The frozen CLIP image encoder accepts RGB, so B1 uses only
    source channels zero through two.  The mapping itself remains responsible
    for blocking use before its full-source audit has made it ready.
    """
    image = np.asarray(tifffile.imread(pair.image_path))
    if image.ndim != 3 or image.shape[2] != 5:
        raise ValueError(f"{pair.identifier}: expected 5-channel HWC aerial raster, got {image.shape}")
    raw_mask = _read_flair_mask(pair.mask_path)
    if raw_mask.ndim != 2:
        raise ValueError(f"{pair.identifier}: expected one-channel mask, got {raw_mask.shape}")
    if image.shape[:2] != raw_mask.shape:
        raise ValueError(f"{pair.identifier}: image shape {image.shape} does not match mask shape {raw_mask.shape}")
    rgb = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
    return rgb, mapping.adapt(raw_mask, node_to_id)


def audit_flair_pairs(pairs: tuple[RasterPair, ...], herbaceous_raw_id: int = 10) -> dict:
    """Audit every paired FLAIR raster without converting or remapping raw labels."""
    if not pairs:
        raise ValueError("FLAIR audit needs at least one paired raster")
    if herbaceous_raw_id < 0:
        raise ValueError("herbaceous raw ID must be non-negative")
    image_shapes: set[tuple[int, ...]] = set()
    image_dtypes: set[str] = set()
    image_channel_counts: set[int] = set()
    mask_shapes: set[tuple[int, int]] = set()
    mask_dtypes: set[str] = set()
    raw_counts: Counter[int] = Counter()
    for pair in pairs:
        image_shape, image_dtype = _read_flair_image_metadata(pair.image_path)
        raw_mask = _read_flair_mask(pair.mask_path)
        if len(image_shape) != 3:
            raise ValueError(f"{pair.identifier}: expected HWC aerial raster, got {image_shape}")
        if raw_mask.ndim != 2:
            raise ValueError(f"{pair.identifier}: expected one-channel mask, got {raw_mask.shape}")
        if image_shape[:2] != raw_mask.shape:
            raise ValueError(
                f"{pair.identifier}: image shape {image_shape} does not match mask shape {raw_mask.shape}"
            )
        raw_ids, counts = np.unique(raw_mask, return_counts=True)
        raw_counts.update({int(raw_id): int(count) for raw_id, count in zip(raw_ids, counts)})
        image_shapes.add(image_shape)
        image_dtypes.add(image_dtype)
        image_channel_counts.add(image_shape[2])
        mask_shapes.add(tuple(int(dimension) for dimension in raw_mask.shape))
        mask_dtypes.add(str(raw_mask.dtype))
    herbaceous_pixels = raw_counts[herbaceous_raw_id]
    return {
        "version": "flair-b1-raw-audit-v1",
        "paired_raster_count": len(pairs),
        "identifier_set_sha256": identifier_set_sha256(pair.identifier for pair in pairs),
        "image_shapes": [list(shape) for shape in sorted(image_shapes)],
        "image_dtypes": sorted(image_dtypes),
        "image_channel_counts": sorted(image_channel_counts),
        "mask_shapes": [list(shape) for shape in sorted(mask_shapes)],
        "mask_dtypes": sorted(mask_dtypes),
        "mask_values_pixel_counts": {str(raw_id): count for raw_id, count in sorted(raw_counts.items())},
        "herbaceous_raw_id": herbaceous_raw_id,
        "herbaceous_pixel_count": herbaceous_pixels,
        "herbaceous_pixel_fraction": herbaceous_pixels / sum(raw_counts.values()),
    }
