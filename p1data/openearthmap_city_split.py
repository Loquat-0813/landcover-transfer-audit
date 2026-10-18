"""Deterministic city-disjoint OpenEarthMap partitions for the B2 control."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from p1data.external import RasterPair
from p1data.splits import identifier_set_sha256, stable_identifier_bucket


@dataclass(frozen=True)
class OpenEarthMapCitySplit:
    train_pairs: tuple[RasterPair, ...]
    validation_pairs: tuple[RasterPair, ...]
    test_pairs: tuple[RasterPair, ...]
    train_cities: tuple[str, ...]
    validation_cities: tuple[str, ...]
    test_cities: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "train_pairs": len(self.train_pairs),
            "validation_pairs": len(self.validation_pairs),
            "test_pairs": len(self.test_pairs),
            "train_cities": len(self.train_cities),
            "validation_cities": len(self.validation_cities),
            "test_cities": len(self.test_cities),
            "train_city_set_sha256": identifier_set_sha256(self.train_cities),
            "validation_city_set_sha256": identifier_set_sha256(self.validation_cities),
            "test_city_set_sha256": identifier_set_sha256(self.test_cities),
            "train_pair_identifier_set_sha256": identifier_set_sha256(pair.identifier for pair in self.train_pairs),
            "validation_pair_identifier_set_sha256": identifier_set_sha256(
                pair.identifier for pair in self.validation_pairs
            ),
            "test_pair_identifier_set_sha256": identifier_set_sha256(pair.identifier for pair in self.test_pairs),
        }


@dataclass(frozen=True)
class OpenEarthMapTestCityManifest:
    """Validated B2 test-city manifest for inference-only target alignment."""

    path: Path
    test_cities: tuple[str, ...]
    test_city_set_sha256: str
    declared_test_pairs: int
    declared_test_pair_identifier_set_sha256: str


def city_from_openearthmap_identifier(identifier: str) -> str:
    """Extract the official city directory from an OpenEarthMap pair ID."""
    components = identifier.split("/")
    if len(components) < 3 or components[1] != "labels" or not components[0]:
        raise ValueError(f"invalid OpenEarthMap identifier: {identifier!r}")
    return components[0]


def exclude_openearthmap_cities(pairs: tuple[RasterPair, ...], excluded_cities: tuple[str, ...]) -> tuple[RasterPair, ...]:
    """Remove complete named cities while preserving deterministic pair order."""
    normalized = frozenset(excluded_cities)
    if not normalized:
        return pairs
    if any(not city for city in normalized):
        raise ValueError("excluded OpenEarthMap city names must be nonempty")
    result = tuple(pair for pair in pairs if city_from_openearthmap_identifier(pair.identifier) not in normalized)
    if not result:
        raise ValueError("city exclusion removed every OpenEarthMap pair")
    return result


def select_openearthmap_cities(
    pairs: tuple[RasterPair, ...], included_cities: tuple[str, ...]
) -> tuple[RasterPair, ...]:
    """Select complete named cities and reject manifests absent from this release."""
    if not included_cities or any(not city for city in included_cities):
        raise ValueError("included OpenEarthMap city names must be nonempty")
    if len(set(included_cities)) != len(included_cities):
        raise ValueError("included OpenEarthMap city names must be unique")
    requested = frozenset(included_cities)
    available = {city_from_openearthmap_identifier(pair.identifier) for pair in pairs}
    missing = sorted(requested.difference(available))
    if missing:
        raise ValueError(f"requested OpenEarthMap cities are absent from this release: {missing}")
    result = tuple(pair for pair in pairs if city_from_openearthmap_identifier(pair.identifier) in requested)
    if not result:
        raise ValueError("city selection removed every OpenEarthMap pair")
    return result


def load_openearthmap_test_city_manifest(path: Path) -> OpenEarthMapTestCityManifest:
    """Load the frozen B2 test cities while checking its self-reported fingerprints."""
    if not path.is_file():
        raise FileNotFoundError(f"OpenEarthMap test-city manifest is missing: {path}")
    return parse_openearthmap_test_city_manifest(json.loads(path.read_text(encoding="utf-8")), path)


def parse_openearthmap_test_city_manifest(payload: object, path: Path) -> OpenEarthMapTestCityManifest:
    """Validate a decoded B2 city manifest without depending on filesystem state."""
    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
        raise ValueError("OpenEarthMap test-city manifest needs an object-valued summary")
    raw_cities = payload.get("test_cities")
    if not isinstance(raw_cities, list) or not raw_cities or any(not isinstance(city, str) or not city for city in raw_cities):
        raise ValueError("OpenEarthMap test-city manifest needs nonempty string test_cities")
    test_cities = tuple(raw_cities)
    if tuple(sorted(test_cities)) != test_cities or len(set(test_cities)) != len(test_cities):
        raise ValueError("OpenEarthMap test-city manifest test_cities must be sorted and unique")
    summary = payload["summary"]
    observed_hash = identifier_set_sha256(test_cities)
    declared_hash = summary.get("test_city_set_sha256")
    if declared_hash != observed_hash:
        raise ValueError("OpenEarthMap test-city manifest city fingerprint does not match test_cities")
    declared_count = summary.get("test_cities")
    if not isinstance(declared_count, int) or declared_count != len(test_cities):
        raise ValueError("OpenEarthMap test-city manifest city count does not match test_cities")
    declared_pairs = summary.get("test_pairs")
    declared_pair_hash = summary.get("test_pair_identifier_set_sha256")
    if not isinstance(declared_pairs, int) or declared_pairs < 1:
        raise ValueError("OpenEarthMap test-city manifest needs a positive test_pairs count")
    if not isinstance(declared_pair_hash, str) or len(declared_pair_hash) != 64:
        raise ValueError("OpenEarthMap test-city manifest needs a SHA-256 test-pair fingerprint")
    return OpenEarthMapTestCityManifest(
        path=path,
        test_cities=test_cities,
        test_city_set_sha256=declared_hash,
        declared_test_pairs=declared_pairs,
        declared_test_pair_identifier_set_sha256=declared_pair_hash,
    )


def split_openearthmap_pairs_by_city(
    pairs: tuple[RasterPair, ...],
    train_percent: int = 70,
    validation_percent: int = 15,
) -> OpenEarthMapCitySplit:
    """Partition full cities using a stable SHA-256 bucket, never individual tiles."""
    if not pairs:
        raise ValueError("OpenEarthMap city split needs at least one paired raster")
    if not 1 <= train_percent < 100 or not 1 <= validation_percent < 100 or train_percent + validation_percent >= 100:
        raise ValueError("train and validation percentages must be positive and leave a test partition")
    cities = tuple(sorted({city_from_openearthmap_identifier(pair.identifier) for pair in pairs}))
    memberships: dict[str, str] = {}
    for city in cities:
        bucket = stable_identifier_bucket(f"openearthmap-city:{city}")
        memberships[city] = "train" if bucket < train_percent else "validation" if bucket < train_percent + validation_percent else "test"
    train_cities = tuple(city for city in cities if memberships[city] == "train")
    validation_cities = tuple(city for city in cities if memberships[city] == "validation")
    test_cities = tuple(city for city in cities if memberships[city] == "test")
    if not train_cities or not validation_cities or not test_cities:
        raise ValueError("city hash partition produced an empty train, validation, or test split")

    def select(partition: str) -> tuple[RasterPair, ...]:
        selected = tuple(pair for pair in pairs if memberships[city_from_openearthmap_identifier(pair.identifier)] == partition)
        if not selected:
            raise ValueError(f"OpenEarthMap {partition} city split selected no paired rasters")
        return selected

    return OpenEarthMapCitySplit(
        train_pairs=select("train"),
        validation_pairs=select("validation"),
        test_pairs=select("test"),
        train_cities=train_cities,
        validation_cities=validation_cities,
        test_cities=test_cities,
    )
