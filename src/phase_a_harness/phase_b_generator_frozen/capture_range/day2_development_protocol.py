"""Frozen core helpers for the exploratory Day 2 Development protocol.

Only ``day2_development_feasibility.yaml`` is parsed.  The archived Day 2
v1.0/v1.1 files are opened solely to verify their byte hashes; they never
become runtime configuration inputs for Development.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from .types import PerturbationSpec


DEVELOPMENT_YAML_RELATIVE = Path(
    "configs/capture_range/day2_development_feasibility.yaml"
)
DEVELOPMENT_YAML_SHA256 = (
    "efbc316542fd32f801d7df84a33f943d4f4ea2eabc5e9c95a5a0770b0383edec"
)

PARENT_FILE_HASHES: Mapping[Path, str] = MappingProxyType(
    {
        Path("configs/capture_range/day2_synthetic_locked.yaml"): (
            "3b2f007c1d68d2399493ce5e15775120c4936495e4c08ef3fd0df1a88c35db65"
        ),
        Path("docs/directional_capture_range_day2_protocol.md"): (
            "765f3eb755db18559fa0a83a1729512f2123b9ce658169542491faecc59e7bd4"
        ),
        Path("configs/capture_range/day2_protocol_amendment_v1_1.yaml"): (
            "f2d68cbe467d617e013c19b6436bda8fc78744e0c8a3e26a76cf3bec654360d5"
        ),
        Path("docs/directional_capture_range_day2_protocol_amendment_v1_1.md"): (
            "8121e3d73110e7f367a0252c6f4654ba47a3073a784c19af3fd626d75d0d1c22"
        ),
    }
)

EXPECTED_DIRECTION_IDS = (
    "pos_x",
    "neg_x",
    "pos_y",
    "neg_y",
    "pos_z",
    "neg_z",
    "ico_00",
    "ico_01",
    "ico_02",
    "ico_03",
    "ico_04",
    "ico_05",
    "ico_06",
    "ico_07",
    "ico_08",
    "ico_09",
    "ico_10",
    "ico_11",
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of the file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_unique_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.load(
            path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
        )
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid Day 2 Development YAML: {path}") from exc
    if type(value) is not dict:
        raise ValueError(f"Day 2 Development YAML root must be a mapping: {path}")
    return value


def _freeze(value: Any) -> Any:
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("Day 2 Development YAML mapping keys must be strings")
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if type(value) is list:
        return tuple(_freeze(child) for child in value)
    return value


@dataclass(frozen=True)
class Day2DevelopmentProtocol:
    """Immutable, independently loaded exploratory Development authority."""

    data: Mapping[str, Any]
    source_sha256: str
    parent_file_hashes: Mapping[Path, str]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.data.get(str(name))
        if not isinstance(value, Mapping):
            raise KeyError(f"unknown Day 2 Development section: {name}")
        return value


def _validate_parent_declarations(data: Mapping[str, Any]) -> None:
    parent = data.get("immutable_parent_files")
    if type(parent) is not dict:
        raise ValueError("immutable_parent_files must be a mapping")
    if parent.get("runtime_use") != "byte_hash_validation_only":
        raise ValueError("parent runtime use must remain byte_hash_validation_only")
    if parent.get("parse_parent_test_split_during_development") is not False:
        raise ValueError("parent Test split parsing must remain disabled")
    entries = parent.get("files")
    if type(entries) is not list:
        raise ValueError("immutable parent file inventory must be a list")
    declared: dict[Path, str] = {}
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"path", "sha256"}:
            raise ValueError("invalid immutable parent file declaration")
        path = Path(str(entry["path"]))
        if path in declared:
            raise ValueError(f"duplicate immutable parent declaration: {path}")
        declared[path] = str(entry["sha256"])
    if declared != dict(PARENT_FILE_HASHES):
        raise ValueError("immutable parent file declarations changed")


def _validate_authority(data: Mapping[str, Any]) -> None:
    if data.get("schema_version") != (
        "directional_capture_range_day2_development_feasibility_v1"
    ):
        raise ValueError("Day 2 Development schema version changed")
    protocol = data.get("protocol")
    if type(protocol) is not dict:
        raise ValueError("Day 2 Development protocol metadata missing")
    expected = {
        "protocol_type": "exploratory_development",
        "status": "prospectively_frozen_before_any_development_seed_run",
        "scientific_claim_authorized": False,
        "confirmatory_test_seed_access": "forbidden",
        "formal_scientific_pass_fail_authorized": False,
        "day3_authorization": "forbidden",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Day 2 Development authority changed: protocol.{key}")

    canonical = data.get("canonical_hashing")
    if type(canonical) is not dict:
        raise ValueError("canonical_hashing section missing")
    canonical_expected = {
        "text_encoding": "UTF-8",
        "mapping_key_order": "sorted",
        "json_separators": [",", ":"],
        "ensure_ascii": False,
        "float_encoding": "python_float_hex",
        "negative_zero_normalized_to_positive_zero": True,
        "digest": "SHA-256",
        "digest_to_seed": "first_16_digest_bytes_unsigned_big_endian",
        "digest_to_unit_interval": (
            "first_8_digest_bytes_unsigned_big_endian_divided_by_2_pow_64"
        ),
        "cross_process_identity_required": True,
    }
    for key, value in canonical_expected.items():
        if canonical.get(key) != value:
            raise ValueError(f"canonical hashing rule changed: {key}")

    _validate_parent_declarations(data)


def load_day2_development_protocol(
    root: str | Path,
) -> Day2DevelopmentProtocol:
    """Byte-validate the lock and parse only the new Development YAML."""

    repository = Path(root).resolve()
    source = repository / DEVELOPMENT_YAML_RELATIVE
    source_hash = file_sha256(source)
    if source_hash != DEVELOPMENT_YAML_SHA256:
        raise ValueError(
            "Day 2 Development protocol byte hash changed for "
            f"{DEVELOPMENT_YAML_RELATIVE}: {source_hash}"
        )

    for relative_path, expected in PARENT_FILE_HASHES.items():
        actual = file_sha256(repository / relative_path)
        if actual != expected:
            raise ValueError(
                f"archived Day 2 parent byte hash changed for {relative_path}: {actual}"
            )

    raw = _load_unique_yaml(source)
    _validate_authority(raw)
    return Day2DevelopmentProtocol(
        data=_freeze(raw),
        source_sha256=source_hash,
        parent_file_hashes=PARENT_FILE_HASHES,
    )


def _canonical_float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("canonical floating values must be finite")
    if number == 0.0:
        number = 0.0
    return number.hex()


def canonical_array_payload(value: Any) -> dict[str, Any]:
    """Encode an array as dtype, shape, and C-order scalar values."""

    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.fields is not None:
        raise TypeError("canonical arrays cannot use object or structured dtypes")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("canonical arrays cannot use complex dtypes")
    contiguous = np.ascontiguousarray(array)
    values = [item.item() for item in contiguous.reshape(-1, order="C")]
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "values": values,
    }


def _canonicalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _canonicalize(canonical_array_payload(value))
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return _canonical_float(value)
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return {key: _canonicalize(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_canonicalize(child) for child in value]
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Return the frozen recursive UTF-8 JSON representation."""

    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: Any) -> bytes:
    return hashlib.sha256(canonical_bytes(value)).digest()


def canonical_sha256(value: Any) -> str:
    return _canonical_digest(value).hex()


def canonical_seed(value: Any) -> int:
    """Map a canonical payload to the frozen unsigned 128-bit seed."""

    return int.from_bytes(_canonical_digest(value)[:16], "big", signed=False)


def canonical_unit_interval(value: Any) -> float:
    """Map a canonical payload into the half-open interval [0, 1)."""

    numerator = int.from_bytes(_canonical_digest(value)[:8], "big", signed=False)
    return numerator / float(1 << 64)


def canonical_array_sha256(value: Any) -> str:
    return canonical_sha256(canonical_array_payload(value))


class DevelopmentSeedAccessError(ValueError):
    """Raised before hashing/RNG when a seed is outside Development authority."""


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(integer)


@dataclass(frozen=True)
class DevelopmentSeedIdentity:
    geometry_seed: int
    measurement_seed: int
    repeat_index: int


@dataclass
class _SeedFirewallAudit:
    access_attempt_count: int = 0
    allowed_access_count: int = 0
    confirmatory_access_attempt_count: int = 0
    unknown_access_attempt_count: int = 0
    seed_hash_materialization_count: int = 0
    rng_materialization_count: int = 0


@dataclass(frozen=True)
class DevelopmentSeedFirewall:
    """Equality-first confirmatory guard for every Development seed stream."""

    global_seed: int
    allowed_geometry_seeds: tuple[int, ...]
    allowed_measurement_seeds: tuple[int, ...]
    repeats: int
    forbidden_geometry_seed_sentinels: tuple[int, ...]
    forbidden_measurement_seed_sentinels: tuple[int, ...]
    scene_variants: tuple[str, ...] = ()
    stream_roles: tuple[str, ...] = ()
    _audit: _SeedFirewallAudit = field(
        default_factory=_SeedFirewallAudit, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        global_seed = _strict_nonnegative_int(self.global_seed, name="global_seed")
        repeats = _strict_nonnegative_int(self.repeats, name="repeats")
        if repeats == 0:
            raise ValueError("repeats must be positive")

        def normalized(values: Sequence[Any], name: str) -> tuple[int, ...]:
            result = tuple(
                _strict_nonnegative_int(value, name=name) for value in values
            )
            if not result or len(set(result)) != len(result):
                raise ValueError(f"{name} must be non-empty and unique")
            return result

        allowed_geometry = normalized(
            self.allowed_geometry_seeds, "allowed_geometry_seed"
        )
        allowed_measurement = normalized(
            self.allowed_measurement_seeds, "allowed_measurement_seed"
        )
        forbidden_geometry = normalized(
            self.forbidden_geometry_seed_sentinels,
            "forbidden_geometry_seed_sentinel",
        )
        forbidden_measurement = normalized(
            self.forbidden_measurement_seed_sentinels,
            "forbidden_measurement_seed_sentinel",
        )
        if set(allowed_geometry) & set(forbidden_geometry):
            raise ValueError("allowed and forbidden geometry seeds overlap")
        if set(allowed_measurement) & set(forbidden_measurement):
            raise ValueError("allowed and forbidden measurement seeds overlap")

        object.__setattr__(self, "global_seed", global_seed)
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "allowed_geometry_seeds", allowed_geometry)
        object.__setattr__(self, "allowed_measurement_seeds", allowed_measurement)
        object.__setattr__(
            self, "forbidden_geometry_seed_sentinels", forbidden_geometry
        )
        object.__setattr__(
            self, "forbidden_measurement_seed_sentinels", forbidden_measurement
        )
        object.__setattr__(
            self, "scene_variants", tuple(str(value) for value in self.scene_variants)
        )
        object.__setattr__(
            self, "stream_roles", tuple(str(value) for value in self.stream_roles)
        )

    def _validated_pair(
        self, geometry_seed: int, measurement_seed: int
    ) -> tuple[int, int]:
        geometry = _strict_nonnegative_int(geometry_seed, name="geometry_seed")
        measurement = _strict_nonnegative_int(
            measurement_seed, name="measurement_seed"
        )
        if geometry in self.forbidden_geometry_seed_sentinels:
            self._audit.confirmatory_access_attempt_count += 1
            raise DevelopmentSeedAccessError(
                f"confirmatory geometry seed access forbidden: {geometry}"
            )
        if measurement in self.forbidden_measurement_seed_sentinels:
            self._audit.confirmatory_access_attempt_count += 1
            raise DevelopmentSeedAccessError(
                f"confirmatory measurement seed access forbidden: {measurement}"
            )
        if geometry not in self.allowed_geometry_seeds:
            self._audit.unknown_access_attempt_count += 1
            raise DevelopmentSeedAccessError(
                f"unknown Development geometry seed: {geometry}"
            )
        if measurement not in self.allowed_measurement_seeds:
            self._audit.unknown_access_attempt_count += 1
            raise DevelopmentSeedAccessError(
                f"unknown Development measurement seed: {measurement}"
            )
        return geometry, measurement

    def assert_geometry_allowed(self, geometry_seed: int) -> int:
        """Guard a geometry-only phase before its first hash."""

        self._audit.access_attempt_count += 1
        geometry = _strict_nonnegative_int(geometry_seed, name="geometry_seed")
        if geometry in self.forbidden_geometry_seed_sentinels:
            self._audit.confirmatory_access_attempt_count += 1
            raise DevelopmentSeedAccessError(
                f"confirmatory geometry seed access forbidden: {geometry}"
            )
        if geometry not in self.allowed_geometry_seeds:
            self._audit.unknown_access_attempt_count += 1
            raise DevelopmentSeedAccessError(
                f"unknown Development geometry seed: {geometry}"
            )
        self._audit.allowed_access_count += 1
        return geometry

    def assert_allowed(
        self, geometry_seed: int, measurement_seed: int
    ) -> tuple[int, int]:
        """Guard a Development seed pair before hashing or RNG construction."""

        self._audit.access_attempt_count += 1
        pair = self._validated_pair(geometry_seed, measurement_seed)
        self._audit.allowed_access_count += 1
        return pair

    def assert_access(
        self,
        geometry_seed: int,
        measurement_seed: int,
        repeat_index: int,
    ) -> DevelopmentSeedIdentity:
        self._audit.access_attempt_count += 1
        geometry, measurement = self._validated_pair(
            geometry_seed, measurement_seed
        )
        repeat = _strict_nonnegative_int(repeat_index, name="repeat_index")
        if repeat >= self.repeats:
            self._audit.unknown_access_attempt_count += 1
            raise DevelopmentSeedAccessError(
                f"unknown Development repeat index: {repeat}"
            )
        self._audit.allowed_access_count += 1
        return DevelopmentSeedIdentity(geometry, measurement, repeat)

    @property
    def audit_counts(self) -> Mapping[str, int]:
        """Return an immutable point-in-time manifest counter snapshot."""

        return MappingProxyType(
            {
                "seed_access_attempt_count": self._audit.access_attempt_count,
                "allowed_seed_access_count": self._audit.allowed_access_count,
                "confirmatory_seed_access_attempt_count": (
                    self._audit.confirmatory_access_attempt_count
                ),
                "unknown_seed_access_attempt_count": (
                    self._audit.unknown_access_attempt_count
                ),
                "seed_hash_materialization_count": (
                    self._audit.seed_hash_materialization_count
                ),
                "rng_materialization_count": self._audit.rng_materialization_count,
                "confirmatory_seed_materialization_count": 0,
            }
        )

    def stream_seed(
        self,
        scene_variant: str,
        geometry_seed: int,
        measurement_seed: int,
        repeat_index: int,
        stream_role: str,
    ) -> int:
        identity = self.assert_access(
            geometry_seed, measurement_seed, repeat_index
        )
        scene = str(scene_variant)
        role = str(stream_role)
        if self.scene_variants and scene not in self.scene_variants:
            raise ValueError(f"unknown Development scene variant: {scene}")
        if self.stream_roles and role not in self.stream_roles:
            raise ValueError(f"unknown Development stream role: {role}")
        result = canonical_seed(
            {
                "global_seed": self.global_seed,
                "scene_variant": scene,
                "geometry_seed": identity.geometry_seed,
                "measurement_seed": identity.measurement_seed,
                "repeat_index": identity.repeat_index,
                "stream_role": role,
            }
        )
        self._audit.seed_hash_materialization_count += 1
        return result

    def rng(
        self,
        scene_variant: str,
        geometry_seed: int,
        measurement_seed: int,
        repeat_index: int,
        stream_role: str,
    ) -> np.random.Generator:
        seed = self.stream_seed(
            scene_variant,
            geometry_seed,
            measurement_seed,
            repeat_index,
            stream_role,
        )
        generator = np.random.Generator(np.random.PCG64(seed))
        self._audit.rng_materialization_count += 1
        return generator


def development_seed_firewall(
    protocol: Day2DevelopmentProtocol,
) -> DevelopmentSeedFirewall:
    seed = protocol.section("seed_firewall")
    measurement = protocol.section("measurement_realization")
    scenes = protocol.section("scene_generation")
    return DevelopmentSeedFirewall(
        global_seed=seed["global_seed"],
        allowed_geometry_seeds=tuple(seed["allowed_geometry_seeds"]),
        allowed_measurement_seeds=tuple(seed["allowed_measurement_seeds"]),
        repeats=seed["repeats"],
        forbidden_geometry_seed_sentinels=tuple(
            seed["forbidden_confirmatory_geometry_seed_sentinels"]
        ),
        forbidden_measurement_seed_sentinels=tuple(
            seed["forbidden_confirmatory_measurement_seed_sentinels"]
        ),
        scene_variants=tuple(scenes["variants_in_order"]),
        stream_roles=tuple(measurement["stream_roles_in_order"]),
    )


@dataclass(frozen=True)
class DevelopmentDirection:
    """One physical directed vector and its PerturbationSpec representation."""

    direction_id: str
    physical_vector: tuple[float, float, float]
    canonical_basis: tuple[float, float, float]
    signed_side: int
    order_index: int
    source: str

    @property
    def vector(self) -> tuple[float, float, float]:
        """Backward-compatible spelling for the physical directed vector."""

        return self.physical_vector


def _direction(
    direction_id: Any,
    vector: Any,
    order_index: int,
    source: str,
) -> DevelopmentDirection:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid direction vector for {direction_id}")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        raise ValueError(f"zero direction vector for {direction_id}")
    physical = np.asarray(array / norm, dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(physical) > 1.0e-12)
    if nonzero.size == 0:
        raise ValueError(f"zero direction vector for {direction_id}")
    side = 1 if float(physical[int(nonzero[0])]) > 0.0 else -1
    basis = np.asarray(physical * side, dtype=np.float64)
    return DevelopmentDirection(
        direction_id=str(direction_id),
        physical_vector=tuple(float(value) for value in physical),
        canonical_basis=tuple(float(value) for value in basis),
        signed_side=side,
        order_index=int(order_index),
        source=str(source),
    )


def development_directions(
    protocol: Day2DevelopmentProtocol,
) -> tuple[DevelopmentDirection, ...]:
    """Resolve the exact 18 physical directed rows, without runtime aliases."""

    section = protocol.section("directions")
    if section.get("directed") is not True:
        raise ValueError("Development directions must remain directed")
    if section.get("expected_per_scene") != 18:
        raise ValueError("Development direction count must remain 18")
    if section.get("aliases_in_runtime_inventory") is not False:
        raise ValueError("runtime direction aliases must remain disabled")
    if section.get("role_declarations_change_runtime_count") is not False:
        raise ValueError("role declarations cannot change runtime count")

    rows: list[DevelopmentDirection] = []
    for entry in section["cartesian"]:
        rows.append(
            _direction(
                entry["direction_id"], entry["vector"], len(rows), "cartesian"
            )
        )
    for entry in section["normalized_icosahedron"]:
        rows.append(
            _direction(
                entry["direction_id"],
                entry["raw_vector"],
                len(rows),
                "normalized_icosahedron",
            )
        )

    if tuple(row.direction_id for row in rows) != EXPECTED_DIRECTION_IDS:
        raise ValueError("Development direction identity or order changed")
    vectors = np.asarray([row.physical_vector for row in rows], dtype=np.float64)
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1.0e-12):
        raise ValueError("Development direction normalization failed")
    dot = vectors @ vectors.T
    distinct = ~np.eye(len(rows), dtype=bool)
    if np.any(dot[distinct] >= 0.9999999999):
        raise ValueError("Development physical directed rows are not unique")
    return tuple(rows)


def direction_for_id(
    protocol: Day2DevelopmentProtocol,
    direction_id: str,
) -> DevelopmentDirection:
    matches = [
        row
        for row in development_directions(protocol)
        if row.direction_id == str(direction_id)
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown Development direction: {direction_id}")
    return matches[0]


def make_perturbation_spec(
    direction: DevelopmentDirection,
    perturbation_type: str,
    amplitude: float,
    repeat_index: int,
    seed: int,
) -> PerturbationSpec:
    """Adapt a physical directed row to canonical basis + signed side.

    ``amplitude`` is an unsigned native-unit magnitude: metres for translation
    and radians for rotation.  A caller holding the protocol's displayed
    rotation degrees must call ``math.radians`` first.  The returned
    ``PerturbationSpec.perturbation_vector`` equals
    ``direction.physical_vector * amplitude`` for both positive and antipodal
    rows.
    """

    magnitude = float(amplitude)
    if not math.isfinite(magnitude) or magnitude < 0.0:
        raise ValueError("Development perturbation amplitude must be finite and non-negative")
    signed_amplitude = 0.0 if magnitude == 0.0 else direction.signed_side * magnitude
    return PerturbationSpec(
        perturbation_type=str(perturbation_type),
        direction=np.asarray(direction.canonical_basis, dtype=np.float64),
        signed_amplitude=signed_amplitude,
        repeat_index=repeat_index,
        seed=seed,
        direction_id=direction.direction_id,
        signed_side=direction.signed_side,
    )
