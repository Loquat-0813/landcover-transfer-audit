"""Pair audited RGB rasters and label masks for external P1 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RasterPair:
    """One RGB raster and its single-band segmentation mask."""

    identifier: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class ExternalRasterDiscovery:
    """Paired external rasters and label-only files excluded from inference."""

    dataset: str
    pairs: tuple[RasterPair, ...]
    audited_mask_count: int
    unpaired_mask_identifiers: tuple[str, ...]

    @property
    def paired_raster_count(self) -> int:
        return len(self.pairs)


def _build_discovery(
    pairs: list[RasterPair], dataset: str, audited_mask_count: int, unpaired_mask_identifiers: list[str]
) -> ExternalRasterDiscovery:
    if not pairs:
        raise ValueError(f"{dataset}: no audited image/mask pairs were discovered")
    identifiers = [pair.identifier for pair in pairs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{dataset}: duplicate raster identifiers")
    if len(pairs) + len(unpaired_mask_identifiers) != audited_mask_count:
        raise RuntimeError(f"{dataset}: paired and unpaired mask counts do not reconcile")
    return ExternalRasterDiscovery(
        dataset=dataset,
        pairs=tuple(pairs),
        audited_mask_count=audited_mask_count,
        unpaired_mask_identifiers=tuple(unpaired_mask_identifiers),
    )


def discover_openearthmap_pairs(root: Path) -> ExternalRasterDiscovery:
    """Discover the non-xBD OpenEarthMap layout distributed by Zenodo."""
    dataset_root = root / "OpenEarthMap_wo_xBD"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"OpenEarthMap root is missing: {dataset_root}")
    masks = sorted(dataset_root.glob("**/labels/*.tif"))
    pairs: list[RasterPair] = []
    unpaired: list[str] = []
    for mask_path in masks:
        image_directory = mask_path.parent.parent / "images"
        candidates = sorted(path for path in image_directory.glob(f"{mask_path.stem}.*") if path.is_file())
        identifier = str(mask_path.relative_to(dataset_root))
        if not candidates:
            # The official archive distributes labels but not RGB for xBD regions.
            unpaired.append(identifier)
            continue
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"OpenEarthMap needs one RGB image for {mask_path}; found {len(candidates)} in {image_directory}"
            )
        image_path = candidates[0]
        pairs.append(
            RasterPair(
                identifier=identifier,
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return _build_discovery(pairs, "openearthmap", len(masks), unpaired)


def _landcover_root(root: Path) -> Path:
    direct = root / "masks"
    if direct.is_dir():
        return root
    candidates = sorted(path.parent for path in root.glob("**/masks") if path.is_dir())
    if len(candidates) != 1:
        raise FileNotFoundError(f"LandCoverAI needs exactly one masks directory below {root}, found {candidates}")
    return candidates[0]


def discover_landcoverai_pairs(root: Path) -> ExternalRasterDiscovery:
    """Discover the official LandCoverAI v1 ``images/`` and ``masks/`` layout."""
    dataset_root = _landcover_root(root)
    masks = sorted((dataset_root / "masks").glob("*.tif"))
    pairs: list[RasterPair] = []
    for mask_path in masks:
        image_path = dataset_root / "images" / mask_path.name
        if not image_path.is_file():
            raise FileNotFoundError(f"LandCoverAI image is missing for {mask_path}: {image_path}")
        pairs.append(
            RasterPair(
                identifier=str(mask_path.relative_to(dataset_root)),
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return _build_discovery(pairs, "landcoverai", len(masks), [])


def discover_external_pairs(dataset: str, root: Path) -> ExternalRasterDiscovery:
    """Dispatch discovery for one frozen external-evaluation dataset."""
    if dataset == "openearthmap":
        return discover_openearthmap_pairs(root)
    if dataset == "landcoverai":
        return discover_landcoverai_pairs(root)
    raise ValueError(f"unsupported external dataset: {dataset}")
