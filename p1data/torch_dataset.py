"""Torch dataset adapter for the frozen-VLM LoveDA E0/E1 experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from p1data.external import RasterPair
from p1data.flair import load_flair_example
from p1data.loveda import LoveDAPair, discover_loveda_pairs, load_loveda_example, random_aligned_crop
from p1data.semantic_drone import _packed_rgb_array
from p1eval.taxonomy import DatasetMapping
from PIL import Image


class LoveDATaxonomyDataset(Dataset["dict[str, Tensor | str]"]):
    """Deterministic LoveDA samples with taxonomy-node targets.

    Images are emitted as RGB float tensors in `[0, 1]`. Normalization belongs
    to the selected frozen backbone, so this dataset never applies CLIP-specific
    normalization itself.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        mapping: DatasetMapping,
        node_to_id: dict[str, int],
        crop_size: int | None,
        seed: int,
        max_samples: int | None = None,
        include_identifiers: frozenset[str] | None = None,
    ) -> None:
        self.pairs: tuple[LoveDAPair, ...] = discover_loveda_pairs(root, split)
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive when set")
            self.pairs = self.pairs[:max_samples]
        if include_identifiers is not None:
            self.pairs = tuple(pair for pair in self.pairs if pair.identifier in include_identifiers)
            if not self.pairs:
                raise ValueError("identifier filter selected no LoveDA pairs")
        self.mapping = mapping
        self.node_to_id = node_to_id
        self.crop_size = crop_size
        self.seed = seed
        self.split = split
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        pair = self.pairs[index]
        image, target = load_loveda_example(pair, self.mapping, self.node_to_id)
        if self.crop_size is not None:
            # The crop is stable across worker order but changes across epochs.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            image, target = random_aligned_crop(image, target, self.crop_size, rng)
        # PIL-backed NumPy views can be read-only; PyTorch tensors need writable storage.
        image_tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True, order="C")).permute(2, 0, 1).float().div_(255.0)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target)).long()
        return {"image": image_tensor, "target": target_tensor, "identifier": pair.identifier}


class TaxonomyRasterDataset(Dataset["dict[str, Tensor | str]"]):
    """Deterministic taxonomy-labelled RGB rasters for a prepaired dataset split."""

    def __init__(
        self,
        pairs: tuple[RasterPair, ...],
        mapping: DatasetMapping,
        node_to_id: dict[str, int],
        crop_size: int | None,
        seed: int,
        max_samples: int | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("taxonomy raster dataset needs at least one paired raster")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive when set")
        self.pairs = pairs if max_samples is None else pairs[:max_samples]
        if not self.pairs:
            raise ValueError("max_samples selected no taxonomy rasters")
        self.mapping = mapping
        self.node_to_id = node_to_id
        self.crop_size = crop_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        pair = self.pairs[index]
        with Image.open(pair.image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
        with Image.open(pair.mask_path) as mask_file:
            raw_mask = np.asarray(mask_file)
        if raw_mask.ndim != 2:
            raise ValueError(f"{pair.identifier}: expected one-channel mask, got {raw_mask.shape}")
        if image.shape[:2] != raw_mask.shape:
            raise ValueError(f"{pair.identifier}: image shape {image.shape[:2]} does not match mask shape {raw_mask.shape}")
        target = self.mapping.adapt(raw_mask, self.node_to_id)
        if self.crop_size is not None:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            image, target = random_aligned_crop(image, target, self.crop_size, rng)
        image_tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True, order="C")).permute(2, 0, 1).float().div_(255.0)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target)).long()
        return {"image": image_tensor, "target": target_tensor, "identifier": pair.identifier}


class SemanticDroneTaxonomyDataset(Dataset["dict[str, Tensor | str]"]):
    """Semantic Drone RGB-mask pairs with an explicit RGB-to-taxonomy mapping.

    The official masks are color-coded PNGs rather than integer label rasters.
    Only colors listed in ``packed_rgb_to_node`` receive supervision; every
    other registered class is intentionally ignored by the second-case audit.
    """

    def __init__(
        self,
        pairs: tuple[RasterPair, ...],
        packed_rgb_to_node: dict[int, int],
        crop_size: int | None,
        seed: int,
        max_samples: int | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("Semantic Drone taxonomy dataset needs at least one paired raster")
        if not packed_rgb_to_node:
            raise ValueError("Semantic Drone RGB mapping must contain at least one supervised color")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive when set")
        self.pairs = pairs if max_samples is None else pairs[:max_samples]
        if not self.pairs:
            raise ValueError("max_samples selected no Semantic Drone pairs")
        self.packed_rgb_to_node = dict(packed_rgb_to_node)
        self.crop_size = crop_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        if not 0 <= index < len(self.pairs):
            raise IndexError("Semantic Drone sample index is out of range")
        pair = self.pairs[index]
        with Image.open(pair.image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
        with Image.open(pair.mask_path) as mask_file:
            mask = np.asarray(mask_file.convert("RGB"), dtype=np.uint8)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"{pair.identifier}: image and RGB mask shapes differ")
        packed = _packed_rgb_array(mask)
        target = np.full(mask.shape[:2], -1, dtype=np.int64)
        for color, node_id in self.packed_rgb_to_node.items():
            target[packed == color] = node_id
        if self.crop_size is not None:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            image, target = random_aligned_crop(image, target, self.crop_size, rng)
        image_tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True, order="C")).permute(2, 0, 1).float().div_(255.0)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target)).long()
        return {"image": image_tensor, "target": target_tensor, "identifier": pair.identifier}


class FlairTaxonomyDataset(Dataset["dict[str, Tensor | str]"]):
    """RGB-only FLAIR samples with a deliberately narrow B1 label mapping.

    FLAIR aerial GeoTIFFs have five bands.  ``load_flair_example`` validates
    that layout and discards NIR/nDSM before the frozen RGB CLIP encoder sees
    the raster.  Unlike generic raster datasets, this class must therefore not
    use PIL's implicit colour conversion.
    """

    def __init__(
        self,
        pairs: tuple[RasterPair, ...],
        mapping: DatasetMapping,
        node_to_id: dict[str, int],
        crop_size: int | None,
        seed: int,
        max_samples: int | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("FLAIR taxonomy dataset needs at least one paired raster")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive when set")
        self.pairs = pairs if max_samples is None else pairs[:max_samples]
        if not self.pairs:
            raise ValueError("max_samples selected no FLAIR rasters")
        self.mapping = mapping
        self.node_to_id = node_to_id
        self.crop_size = crop_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        return self.sample_at(index, self.epoch)

    def sample_at(self, index: int, epoch: int) -> dict[str, Tensor | str]:
        """Return a deterministic FLAIR crop without mutating dataset state."""
        if not 0 <= index < len(self.pairs) or epoch < 0:
            raise IndexError("FLAIR sample index or epoch is out of range")
        pair = self.pairs[index]
        image, target = load_flair_example(pair, self.mapping, self.node_to_id)
        if self.crop_size is not None:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
            image, target = random_aligned_crop(image, target, self.crop_size, rng)
        image_tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True, order="C")).permute(2, 0, 1).float().div_(255.0)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target)).long()
        return {"image": image_tensor, "target": target_tensor, "identifier": pair.identifier}
