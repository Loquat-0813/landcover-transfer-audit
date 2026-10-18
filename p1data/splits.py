"""Deterministic source-data partitions recorded by stable sample identifiers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def stable_identifier_bucket(identifier: str, bucket_count: int = 100) -> int:
    """Return a process-independent hash bucket for one dataset identifier."""
    if not identifier:
        raise ValueError("identifier must be non-empty")
    if bucket_count < 2:
        raise ValueError("bucket_count must be at least two")
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % bucket_count


def split_identifiers_for_calibration(
    identifiers: Iterable[str],
    calibration_percent: int = 15,
    bucket_count: int = 100,
) -> tuple[frozenset[str], frozenset[str]]:
    """Split IDs into non-empty optimization and conformal-calibration sets."""
    if not 1 <= calibration_percent < bucket_count:
        raise ValueError("calibration_percent must lie between one and bucket_count - 1")
    ordered = tuple(identifiers)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("identifiers must be non-empty and unique")
    calibration = frozenset(identifier for identifier in ordered if stable_identifier_bucket(identifier, bucket_count) < calibration_percent)
    optimization = frozenset(ordered).difference(calibration)
    if not calibration or not optimization:
        raise ValueError("deterministic split produced an empty partition")
    return optimization, calibration


def identifier_set_sha256(identifiers: Iterable[str]) -> str:
    """Fingerprint an unordered identifier set for a reproducibility record."""
    ordered = tuple(sorted(identifiers))
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("identifiers must be non-empty and unique")
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
