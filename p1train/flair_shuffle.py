"""Deterministic, density-matched FLAIR image/mask derangements."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FlairCropRecord:
    """One materialized B1 FLAIR crop used by the shuffle schedule."""

    update: int
    source_index: int
    source_identifier: str
    domain: str
    id10_pixels: int
    crop_pixels: int

    @property
    def id10_density(self) -> float:
        if self.crop_pixels < 1:
            raise ValueError("crop_pixels must be positive")
        return float(self.id10_pixels) / float(self.crop_pixels)

    def to_dict(self) -> dict[str, int | str | float]:
        return {
            "update": self.update,
            "source_index": self.source_index,
            "source_identifier": self.source_identifier,
            "domain": self.domain,
            "id10_pixels": self.id10_pixels,
            "crop_pixels": self.crop_pixels,
            "id10_density": self.id10_density,
        }


@dataclass(frozen=True)
class FlairShufflePlan:
    """A permutation mapping each B1 RGB crop row to a different mask row."""

    permutation: tuple[int, ...]
    blocks: tuple[tuple[int, ...], ...]
    merged_singleton_bins: tuple[tuple[str, int], ...]
    seed: int
    density_bin_width: float
    max_abs_density_difference: float
    mean_abs_density_difference: float

    def to_dict(self, records: tuple[FlairCropRecord, ...]) -> dict:
        if len(records) != len(self.permutation):
            raise ValueError("records and permutation lengths differ")
        return {
            "version": "flair-shuffle-plan-v1",
            "seed": self.seed,
            "density_bin_width": self.density_bin_width,
            "records": [record.to_dict() for record in records],
            "permutation": list(self.permutation),
            "blocks": [list(block) for block in self.blocks],
            "merged_singleton_bins": [list(item) for item in self.merged_singleton_bins],
            "max_abs_density_difference": self.max_abs_density_difference,
            "mean_abs_density_difference": self.mean_abs_density_difference,
        }

    def write_json(self, path: Path, records: tuple[FlairCropRecord, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(records), indent=2) + "\n", encoding="utf-8")


def load_density_matched_derangement(path: Path, records: tuple[FlairCropRecord, ...]) -> FlairShufflePlan:
    """Load a precomputed schedule and bind it to the current crop records."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "flair-shuffle-plan-v1":
        raise ValueError("shuffle schedule has an unknown version")
    expected = [record.to_dict() for record in records]
    if payload.get("records") != expected:
        raise ValueError("shuffle schedule records differ from the materialized training crop schedule")
    permutation = tuple(int(index) for index in payload.get("permutation", ()))
    if len(permutation) != len(records) or sorted(permutation) != list(range(len(records))):
        raise ValueError("shuffle schedule is not a complete permutation")
    if any(records[index].source_identifier == records[mask_index].source_identifier for index, mask_index in enumerate(permutation)):
        raise ValueError("shuffle schedule contains a native image-mask pairing")
    differences = np.asarray(
        [abs(records[index].id10_density - records[mask_index].id10_density) for index, mask_index in enumerate(permutation)],
        dtype=np.float64,
    )
    return FlairShufflePlan(
        permutation=permutation,
        blocks=tuple(tuple(int(index) for index in block) for block in payload.get("blocks", ())),
        merged_singleton_bins=tuple((str(domain), int(density_bin)) for domain, density_bin in payload.get("merged_singleton_bins", ())),
        seed=int(payload["seed"]),
        density_bin_width=float(payload["density_bin_width"]),
        max_abs_density_difference=float(differences.max()),
        mean_abs_density_difference=float(differences.mean()),
    )


def _domain_derangement(
    positions: tuple[int, ...],
    records: tuple[FlairCropRecord, ...],
    rng: np.random.Generator,
) -> dict[int, int]:
    """Match adjacent density-sorted masks while forbidding native identifiers.

    A one-position cyclic shift is tried first, so all but one assignment per
    domain receive their nearest density neighbour. The wrap assignment is the
    unavoidable cost of making the assignment a permutation. Repeated source
    identifiers are repaired with the minimum-density-cost swap available.
    """
    if len(positions) < 2:
        raise ValueError("domain has fewer than two scheduled crops")
    ordered = tuple(sorted(positions, key=lambda position: (records[position].id10_density, records[position].update)))
    # Do not randomize this order. A large cyclic offset makes a valid
    # derangement but defeats density matching. The seed remains relevant only
    # when several collision-repair swaps have equal cost.
    offsets = sorted(range(1, len(ordered)), key=lambda offset: (min(offset, len(ordered) - offset), offset))
    for offset in offsets:
        candidate = {position: ordered[(index + offset) % len(ordered)] for index, position in enumerate(ordered)}
        for _ in range(max(4, len(ordered) * 2)):
            conflicts = [
                position
                for position in ordered
                if records[position].source_identifier == records[candidate[position]].source_identifier
            ]
            if not conflicts:
                return candidate
            image_position = conflicts[0]
            choices: list[tuple[float, int]] = []
            for other_position in ordered:
                if other_position == image_position:
                    continue
                if records[image_position].source_identifier == records[candidate[other_position]].source_identifier:
                    continue
                if records[other_position].source_identifier == records[candidate[image_position]].source_identifier:
                    continue
                score = abs(
                    records[image_position].id10_density - records[candidate[other_position]].id10_density
                ) + abs(records[other_position].id10_density - records[candidate[image_position]].id10_density)
                choices.append((score, other_position))
            if not choices:
                break
            best_score = min(score for score, _ in choices)
            best = [position for score, position in choices if np.isclose(score, best_score)]
            other_position = int(best[int(rng.integers(0, len(best)))])
            candidate[image_position], candidate[other_position] = candidate[other_position], candidate[image_position]
    raise ValueError("cannot construct a non-self source-identifier derangement for the domain")


def build_density_matched_derangement(
    records: tuple[FlairCropRecord, ...],
    seed: int,
    density_bin_width: float = 0.01,
) -> FlairShufflePlan:
    """Build a deterministic same-domain derangement ordered by crop density.

    Adjacent records in each domain's density ordering form pairs. An odd
    domain count uses a final three-record block. This is equivalent to
    merging singleton density bins into their nearest adjacent bin, while
    retaining the exact multiset of crop-level ID-10 pixel counts.
    """
    if not records:
        raise ValueError("records must be non-empty")
    if seed < 0 or not 0 < density_bin_width <= 1:
        raise ValueError("seed must be non-negative and density_bin_width must be in (0, 1]")
    if any(record.update != position for position, record in enumerate(records)):
        raise ValueError("records must be ordered consecutively by update")

    by_domain: dict[str, list[int]] = defaultdict(list)
    for position, record in enumerate(records):
        if record.crop_pixels < 1 or record.id10_pixels < 0 or record.id10_pixels > record.crop_pixels:
            raise ValueError(f"invalid ID-10 crop count at update {record.update}")
        by_domain[record.domain].append(position)

    permutation = [-1] * len(records)
    blocks: list[tuple[int, ...]] = []
    singleton_bins: list[tuple[str, int]] = []
    for domain_index, domain in enumerate(sorted(by_domain)):
        positions = sorted(
            by_domain[domain],
            key=lambda position: (records[position].id10_density, records[position].update),
        )
        if len(positions) < 2:
            raise ValueError(f"domain {domain!r} has fewer than two scheduled crops")
        bin_counts = Counter(
            int(records[position].id10_density / density_bin_width)
            for position in positions
        )
        singleton_bins.extend(
            (domain, density_bin)
            for density_bin, count in sorted(bin_counts.items())
            if count == 1
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, domain_index, len(positions)]))
        domain_assignment = _domain_derangement(tuple(positions), records, rng)
        cursor = 0
        while cursor < len(positions):
            remaining = len(positions) - cursor
            block_size = 3 if remaining == 3 else 2
            block = tuple(positions[cursor : cursor + block_size])
            for image_position in block:
                permutation[image_position] = domain_assignment[image_position]
            blocks.append(block)
            cursor += block_size

    if any(position < 0 for position in permutation):
        raise RuntimeError("shuffle plan did not assign every record")
    if sorted(permutation) != list(range(len(records))):
        raise RuntimeError("shuffle plan is not a permutation")
    if any(records[position].source_identifier == records[mask_position].source_identifier for position, mask_position in enumerate(permutation)):
        raise RuntimeError("shuffle plan contains a native image-mask pairing")
    if sorted(record.id10_pixels for record in records) != sorted(records[position].id10_pixels for position in permutation):
        raise RuntimeError("shuffle plan changed the ID-10 pixel-count multiset")
    density_differences = np.asarray(
        [
            abs(records[position].id10_density - records[mask_position].id10_density)
            for position, mask_position in enumerate(permutation)
        ],
        dtype=np.float64,
    )
    return FlairShufflePlan(
        permutation=tuple(permutation),
        blocks=tuple(blocks),
        merged_singleton_bins=tuple(singleton_bins),
        seed=seed,
        density_bin_width=density_bin_width,
        max_abs_density_difference=float(density_differences.max()),
        mean_abs_density_difference=float(density_differences.mean()),
    )
