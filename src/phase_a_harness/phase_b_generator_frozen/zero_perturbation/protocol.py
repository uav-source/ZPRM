"""Immutable protocol loader and Development-only seed firewall."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from capture_range.day2_development_protocol import canonical_seed


PROTOCOL_RELATIVE = Path("configs/zero_perturbation/development_v1.yaml")
PROTOCOL_SHA256 = "8fe4bcfabfb8492d003b9f690162e106e9fa5745b962ed2c4edfe8666acb291e"
SEED_SCHEDULE_RELATIVE = Path("configs/zero_perturbation/seed_schedule_v1.json")
SEED_SCHEDULE_SHA256 = "1006b76427b19151389f35a82b04e2b93abca339310bf8d56a5b92a27131f050"

EXPECTED_SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
EXPECTED_CONDITIONS = (
    "IDEAL_MATCHED",
    "INDEPENDENT_NOISE_FREE",
    "SCAN_NOISE_ONLY",
    "MAP_NOISE_ONLY",
    "DROPOUT_ONLY",
    "FULL_NOISE",
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze(value: Any) -> Any:
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("protocol mapping keys must be strings")
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if type(value) is list:
        return tuple(_freeze(child) for child in value)
    return value


def _extract_json_object(text: str, marker: str, start: int = 0) -> str:
    marker_index = text.index(marker, start)
    object_start = text.index("{", marker_index + len(marker))
    depth = 0
    in_string = False
    escaped = False
    for index in range(object_start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[object_start : index + 1]
    raise ValueError(f"unterminated JSON object after {marker!r}")


def _load_development_seed_labels(path: Path) -> Mapping[str, Mapping[str, int]]:
    """Parse only labels.development, never the Confirmatory JSON subtree."""

    if file_sha256(path) != SEED_SCHEDULE_SHA256:
        raise ValueError("frozen zero-perturbation seed schedule hash changed")
    text = path.read_text(encoding="utf-8")
    labels_start = text.index('"labels"')
    payload = _extract_json_object(text, '"development"', labels_start)
    value = json.loads(payload)
    expected_domains = {"geometry", "measurement", "bootstrap", "backend"}
    if type(value) is not dict or set(value) != expected_domains:
        raise ValueError("Development seed label domains changed")
    frozen: dict[str, Mapping[str, int]] = {}
    for domain, labels in value.items():
        if type(labels) is not dict or not labels:
            raise ValueError(f"invalid Development seed labels: {domain}")
        parsed = {str(label): int(seed) for label, seed in labels.items()}
        if any(seed <= 0 for seed in parsed.values()):
            raise ValueError("Development seeds must be positive")
        frozen[str(domain)] = MappingProxyType(parsed)
    all_values = [seed for labels in frozen.values() for seed in labels.values()]
    if len(all_values) != len(set(all_values)):
        raise ValueError("Development seed values are not unique")
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class ZeroPerturbationProtocol:
    root: Path
    data: Mapping[str, Any]
    source_sha256: str
    development_seeds: Mapping[str, Mapping[str, int]]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.data.get(str(name))
        if not isinstance(value, Mapping):
            raise KeyError(f"unknown zero-perturbation protocol section: {name}")
        return value

    @property
    def scenes(self) -> tuple[str, ...]:
        return tuple(self.section("scene_generation")["variants_in_order"])

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(str(row["name"]) for row in self.data["noise_conditions"])

    def condition(self, name: str) -> Mapping[str, Any]:
        for row in self.data["noise_conditions"]:
            if row["name"] == name:
                return row
        raise KeyError(f"unknown noise condition: {name}")


def _validate_protocol(data: Mapping[str, Any], root: Path) -> None:
    if data.get("schema_version") != "zero_perturbation_development_v1":
        raise ValueError("zero-perturbation Development schema version changed")
    authority = data.get("protocol")
    expected_authority = {
        "protocol_type": "zero_perturbation_development",
        "confirmatory_claim_authorized": False,
        "real_data_authorized": False,
        "measurement_paper_authorized": False,
        "confirmatory_run_authorized": False,
    }
    if type(authority) is not dict:
        raise ValueError("protocol authority is missing")
    for key, expected in expected_authority.items():
        if authority.get(key) != expected:
            raise ValueError(f"protocol authority changed: {key}")
    scenes = tuple(data["scene_generation"]["variants_in_order"])
    conditions = tuple(row["name"] for row in data["noise_conditions"])
    if scenes != EXPECTED_SCENES or conditions != EXPECTED_CONDITIONS:
        raise ValueError("frozen scene or condition order changed")
    if data["snapshot_contract"]["expected_snapshot_count"] != 1260:
        raise ValueError("Development snapshot count changed")
    if data["snapshot_contract"]["expected_trial_count"] != 3780:
        raise ValueError("Development trial count changed")
    for declaration in data["immutable_inputs"].values():
        candidate = root / declaration["path"]
        if file_sha256(candidate) != declaration["sha256"]:
            raise ValueError(f"immutable input hash changed: {declaration['path']}")


def load_protocol(root: str | Path) -> ZeroPerturbationProtocol:
    repository = Path(root).resolve()
    source = repository / PROTOCOL_RELATIVE
    actual = file_sha256(source)
    if actual != PROTOCOL_SHA256:
        raise ValueError(f"Development protocol byte hash changed: {actual}")
    try:
        raw = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("invalid zero-perturbation Development YAML") from exc
    if type(raw) is not dict:
        raise ValueError("Development YAML root must be a mapping")
    _validate_protocol(raw, repository)
    seeds = _load_development_seed_labels(repository / SEED_SCHEDULE_RELATIVE)
    return ZeroPerturbationProtocol(repository, _freeze(raw), actual, seeds)


class DevelopmentSeedFirewall:
    """Reject non-Development seeds before any hash or RNG construction."""

    def __init__(self, protocol: ZeroPerturbationProtocol) -> None:
        self.protocol = protocol
        self.confirmatory_seed_instantiation_count = 0
        self.old_capture_range_test_seed_access_count = 0
        self.gt_optimization_leakage_count = 0
        self.rng_construction_count = 0
        self.snapshot_access_count = 0
        seeds = protocol.development_seeds
        self._geometry = frozenset(seeds["geometry"].values())
        self._measurement = frozenset(seeds["measurement"].values())
        self._repeats = frozenset(protocol.section("seed_firewall")["repeat_indices"])

    @property
    def global_seed(self) -> int:
        return int(self.protocol.section("scene_generation")["global_scene_seed"])

    def assert_access(self, geometry_seed: int, measurement_seed: int, repeat_index: int) -> None:
        if int(geometry_seed) not in self._geometry or int(measurement_seed) not in self._measurement:
            self.confirmatory_seed_instantiation_count += 1
            raise PermissionError("only scheduled Development geometry/measurement seeds are allowed")
        if int(repeat_index) not in self._repeats:
            raise PermissionError("repeat index is outside the frozen Development schedule")
        self.snapshot_access_count += 1

    def seed(self, domain: str, label: str) -> int:
        try:
            return int(self.protocol.development_seeds[str(domain)][str(label)])
        except KeyError as exc:
            self.confirmatory_seed_instantiation_count += 1
            raise PermissionError("only scheduled Development seed labels are allowed") from exc

    def rng(
        self,
        scene_variant: str,
        geometry_seed: int,
        measurement_seed: int,
        repeat_index: int,
        noise_condition: str,
        stream_role: str,
    ) -> np.random.Generator:
        self.assert_access(geometry_seed, measurement_seed, repeat_index)
        if scene_variant not in self.protocol.scenes:
            raise ValueError("unknown Development scene")
        if noise_condition not in self.protocol.conditions:
            raise ValueError("unknown Development condition")
        if stream_role not in tuple(
            self.protocol.section("measurement_realization")["stream_roles_in_order"]
        ):
            raise ValueError("unknown measurement stream role")
        derived = canonical_seed(
            {
                "development_measurement_seed": int(measurement_seed),
                "scene_variant": str(scene_variant),
                "geometry_seed": int(geometry_seed),
                "repeat_index": int(repeat_index),
                "noise_condition": str(noise_condition),
                "stream_role": str(stream_role),
            }
        )
        self.rng_construction_count += 1
        return np.random.Generator(np.random.PCG64(derived))

    def record_old_capture_range_test_seed_access(self) -> None:
        self.old_capture_range_test_seed_access_count += 1
        raise PermissionError("old capture-range Test seeds are outside this route")

    def report(self) -> dict[str, int]:
        return {
            "CONFIRMATORY_SEED_INSTANTIATION_COUNT": self.confirmatory_seed_instantiation_count,
            "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT": self.old_capture_range_test_seed_access_count,
            "GT_OPTIMIZATION_LEAKAGE_COUNT": self.gt_optimization_leakage_count,
            "RNG_CONSTRUCTION_COUNT": self.rng_construction_count,
            "SNAPSHOT_ACCESS_COUNT": self.snapshot_access_count,
        }


__all__ = [
    "DevelopmentSeedFirewall",
    "EXPECTED_CONDITIONS",
    "EXPECTED_SCENES",
    "PROTOCOL_RELATIVE",
    "PROTOCOL_SHA256",
    "ZeroPerturbationProtocol",
    "file_sha256",
    "load_protocol",
]
