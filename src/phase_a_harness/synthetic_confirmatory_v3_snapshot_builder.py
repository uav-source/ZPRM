"""External, immutable snapshot cache for Synthetic Confirmatory v3.

This module is intentionally a lifecycle adapter, not a new scientific
generator.  IDEAL construction delegates to the frozen scene and Phase-A
parent-lineage primitives; non-IDEAL construction delegates to the frozen
``phase_b_generator.build_snapshot`` implementation.  Importing this module
does not import NumPy, construct an RNG, read a seed schedule, or touch the
formal runtime root.

The formal construction surface is :func:`prepare_v3_snapshots`.  Snapshot
reads are checksum-strict and RNG-free.  Existing entries are never replaced:
resume either authenticates them exactly or fails closed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SNAPSHOT_BUILDER_CONTRACT_VERSION = "synthetic_confirmatory_snapshot_builder_v3"
NON_RNG_MEASUREMENT_SENTINEL = 0
PARENT_INDEX_FILENAME = "source_parent_target_indices.npy"
BASE_ARRAY_FILENAMES = (
    "source_points.npy",
    "target_points.npy",
    "reference_pose.npy",
)
FULL_NOISE_RNG_STREAM_ROLES = frozenset(
    {"scan_dropout", "map_dropout", "scan_noise", "map_noise"}
)

_CLOSURE_FIELDS = (
    "reconstruction_error_median_m",
    "reconstruction_error_q95_m",
    "reconstruction_error_max_m",
    "predicted_quantization_median_m",
    "predicted_quantization_q95_m",
    "predicted_quantization_max_m",
    "closure_residual_max_m",
    "float64_guard_max_m",
    "max_normalized_closure_ratio",
    "closure_residual_violation_count",
    "actual_error_bound_violation_count",
    "quantization_closure_pass",
)

METADATA_FIELDS = frozenset(
    {
        "array_file_sha256",
        "condition",
        "confirmatory_rng_instantiation_count",
        "development_protocol_sha256",
        "dropout_parameters",
        "formal_manifest_payload_sha256",
        "generator_record_checksums",
        "generator_sha256",
        "geometry_seed",
        "independent_sampling",
        "initial_pose",
        "lineage_closure_violation_count",
        "lineage_schema_version",
        "lineage_validation_method",
        "measurement_seed",
        "metadata_payload_sha256",
        "noise_parameters",
        "parent_index_count",
        "parent_index_duplicate_count",
        "parent_index_out_of_range_count",
        "parent_index_unique_count",
        "parent_points_map_f64_sha256",
        "planned_snapshot_id",
        "reference_pose_checksum",
        "repeat_index",
        "scene_variant",
        "schema_version",
        "seed_namespace",
        "snapshot_builder_contract_version",
        "snapshot_checksum",
        "snapshot_id",
        "snapshot_schema_version",
        "source_checksum",
        "source_has_target_parent_lineage",
        "source_is_target_subset",
        "source_parent_row_count_match",
        "source_parent_target_indices_path",
        "source_parent_target_indices_sha256",
        "source_point_count",
        "target_checksum",
        "target_point_count",
    }
) | frozenset(_CLOSURE_FIELDS)

_LOCK_ENTRY_FIELDS = frozenset(
    {
        "condition",
        "confirmatory_rng_instantiation_count",
        "file_sha256",
        "geometry_seed",
        "measurement_seed",
        "metadata_payload_sha256",
        "reference_pose_checksum",
        "repeat_index",
        "scene_variant",
        "snapshot_checksum",
        "snapshot_id",
        "source_checksum",
        "source_parent_target_indices_sha256",
        "target_checksum",
    }
)


class V3SnapshotContractError(RuntimeError):
    """A v3 snapshot identity, cache, lock, or seed firewall was violated."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raw_sha256(value: Any) -> str:
    if not getattr(value, "flags", None) or not value.flags.c_contiguous:
        raise V3SnapshotContractError("raw checksum input must be C-contiguous")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise V3SnapshotContractError(
                    f"duplicate JSON key in {label}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                V3SnapshotContractError(
                    f"non-finite JSON constant in {label}: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V3SnapshotContractError(f"cannot read {label}: {path}") from error
    if type(value) is not dict:
        raise V3SnapshotContractError(f"{label} must be an object")
    return value


def _symlink_components(path: Path) -> tuple[Path, ...]:
    current = Path(path.anchor)
    result: list[Path] = []
    for component in path.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                result.append(current)
        except OSError:
            result.append(current)
    return tuple(result)


def _canonical_absolute(path: str | Path, label: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    if not candidate.is_absolute() or candidate == Path(candidate.anchor):
        raise V3SnapshotContractError(f"{label} must be a non-root absolute path")
    links = _symlink_components(candidate)
    if links:
        raise V3SnapshotContractError(
            f"{label} contains symbolic link component: {links[0]}"
        )
    if candidate.resolve(strict=False) != candidate:
        raise V3SnapshotContractError(f"{label} is not canonical")
    return candidate


def _contract_module() -> Any:
    return importlib.import_module(
        "phase_a_harness.synthetic_confirmatory_v3_contract"
    )


def _load_manifest(repository: Path) -> tuple[Any, dict[str, Any], Path]:
    contract = _contract_module()
    relative = getattr(
        contract,
        "MANIFEST_RELATIVE",
        Path("frozen_assets/synthetic_confirmatory_formal_manifest_v3.json"),
    )
    path = repository / Path(relative)
    direct = _strict_object(path, "v3 formal manifest")
    loader = getattr(contract, "load_v3_contract", None)
    if loader is None:
        loader = getattr(contract, "load_contract", None)
    if callable(loader):
        loaded = loader(path)
        if type(loaded) is not dict:
            raise V3SnapshotContractError("v3 contract loader did not return a dict")
        if type(loaded.get("manifest")) is dict:
            loaded = loaded["manifest"]
        if loaded != direct:
            raise V3SnapshotContractError(
                "v3 contract loader differs from the manifest bytes"
            )
        manifest = dict(loaded)
    else:
        stored = direct.get("manifest_payload_sha256")
        unsigned = {
            name: value
            for name, value in direct.items()
            if name != "manifest_payload_sha256"
        }
        if stored != _canonical_sha256(unsigned):
            raise V3SnapshotContractError("v3 manifest payload SHA mismatch")
        manifest = direct
    return contract, manifest, path


def _normalize_plans(
    repository: Path,
    contract: Any,
    manifest: Mapping[str, Any],
    plans: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validator = getattr(contract, "validate_snapshot_plan_row", None)
    reader = getattr(contract, "typed_snapshot_rows", None)
    if not callable(validator) or not callable(reader):
        raise V3SnapshotContractError(
            "v3 contract lacks the frozen snapshot-plan API"
        )
    normalized = [validator(row) for row in plans]
    plan_path = repository / str(manifest["planned_snapshots_path"])
    canonical = [validator(row) for row in reader(plan_path)]
    if normalized != canonical:
        raise V3SnapshotContractError(
            "snapshot plans differ from the exact frozen v3 plan"
        )
    if (
        len(normalized) != int(manifest["planned_snapshot_count"])
        or len({row["planned_snapshot_id"] for row in normalized})
        != len(normalized)
    ):
        raise V3SnapshotContractError("v3 snapshot plan cardinality changed")
    return normalized


def _formal_paths(
    contract: Any, manifest: Mapping[str, Any], cache_root: str | Path,
    lock_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    cache = _canonical_absolute(cache_root, "v3 snapshot cache")
    expected_cache = _canonical_absolute(
        str(manifest["snapshot_cache_path"]), "manifest snapshot cache"
    )
    if cache != expected_cache:
        raise V3SnapshotContractError(
            "v3 snapshot cache differs from the frozen external path"
        )
    lock: Path | None = None
    if lock_path is not None:
        lock = _canonical_absolute(lock_path, "v3 snapshot lock")
        expected_lock = _canonical_absolute(
            str(manifest["snapshot_lock_path"]), "manifest snapshot lock"
        )
        if lock != expected_lock:
            raise V3SnapshotContractError(
                "v3 snapshot lock differs from the frozen external path"
            )
    repository = Path(getattr(contract, "SOURCE_REPOSITORY", "/home/lj/Degen-LIO"))
    archive = Path(
        getattr(
            contract,
            "RUNTIME_ARCHIVE_ROOT",
            "/home/lj/zero_perturbation_runtime_archive",
        )
    )
    for protected in (repository.resolve(strict=False), archive.resolve(strict=False)):
        if cache == protected or protected in cache.parents or cache in protected.parents:
            raise V3SnapshotContractError(
                f"v3 snapshot cache overlaps protected root: {protected}"
            )
    return cache, lock


def _formal_authorized(repository: Path, contract: Any) -> None:
    relative = getattr(
        contract,
        "PRE_RUN_FINAL_DECISION_RELATIVE",
        Path("artifacts/synthetic_confirmatory_v3_prerun/final_decision.json"),
    )
    decision = _strict_object(repository / Path(relative), "v3 pre-run decision")
    required = {
        "SYNTHETIC_CONFIRMATORY_V3_PRE_RUN_QUALIFICATION_PASS": True,
        "CONFIRMATORY_V3_RUN_AUTHORIZED": True,
        "SYNTHETIC_CONFIRMATORY_V3_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V3_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_V3_PASS": "NOT_EVALUATED",
    }
    if any(decision.get(name) != value for name, value in required.items()):
        raise PermissionError("v3 formal snapshot construction is not authorized")


class ConfirmatoryV3SeedFirewall:
    """The only v3 formal RNG construction point.

    The constructor records no access and creates no generator.  It accepts
    only values already declared by the v3 metadata contract.  The noiseless
    conditions use a non-RNG sentinel internally and cannot be repeated.
    """

    def __init__(self, protocol: Any, contract: Any) -> None:
        self.protocol = protocol
        self._scenes = frozenset(str(value) for value in contract.SCENES)
        self._geometry = frozenset(int(value) for value in contract.GEOMETRY_SEEDS)
        self._measurement = frozenset(
            int(value) for value in contract.MEASUREMENT_SEEDS
        )
        self.rng_instantiation_count = 0
        self.snapshot_access_count = 0

    @property
    def global_seed(self) -> int:
        return int(self.protocol.section("scene_generation")["global_scene_seed"])

    def assert_access(
        self, geometry_seed: int, measurement_seed: int, repeat_index: int
    ) -> None:
        geometry = int(geometry_seed)
        measurement = int(measurement_seed)
        repeat = int(repeat_index)
        if geometry not in self._geometry:
            raise PermissionError("geometry seed is outside Confirmatory v3")
        if measurement == NON_RNG_MEASUREMENT_SENTINEL:
            if repeat != 0:
                raise PermissionError("noiseless v3 input cannot be repeated")
        elif measurement not in self._measurement or repeat not in range(5):
            raise PermissionError(
                "measurement realization is outside Confirmatory v3"
            )
        self.snapshot_access_count += 1

    def rng(
        self,
        scene_variant: str,
        geometry_seed: int,
        measurement_seed: int,
        repeat_index: int,
        noise_condition: str,
        stream_role: str,
    ) -> Any:
        self.assert_access(geometry_seed, measurement_seed, repeat_index)
        if (
            str(scene_variant) not in self._scenes
            or str(noise_condition) != "FULL_NOISE"
            or int(measurement_seed) not in self._measurement
            or str(stream_role) not in FULL_NOISE_RNG_STREAM_ROLES
        ):
            raise PermissionError("RNG request is outside v3 FULL_NOISE")
        import numpy as np
        from .phase_b_generator_frozen.zero_perturbation.protocol import (
            canonical_seed,
        )

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
        self.rng_instantiation_count += 1
        return np.random.Generator(np.random.PCG64(derived))

    def report(self) -> dict[str, int]:
        return {
            "V3_RNG_INSTANTIATION_COUNT": self.rng_instantiation_count,
            "V3_SNAPSHOT_SEED_ACCESS_COUNT": self.snapshot_access_count,
        }


def _lineage_arrays(
    *, protocol: Any, firewall: ConfirmatoryV3SeedFirewall,
    scene: str, geometry_seed: int,
) -> dict[str, Any]:
    """Delegate IDEAL geometry and lineage math to the frozen authorities."""

    import numpy as np
    # Importing this wrapper installs the harness-local frozen capture_range
    # alias required by the copied scientific modules.
    from . import phase_b_generator as _local_generator  # noqa: F401
    from .phase_b_generator_frozen.capture_range.day2_development_scene import (
        build_scene_geometry,
    )
    from .phase_b_generator_frozen.zero_perturbation.backend_phase_a_v1_2 import (
        canonical_target,
        eligible_parent_indices,
        quantization_closure,
        reference_pose_from_development,
        source_from_parent_indices,
    )

    geometry = build_scene_geometry(
        protocol,
        firewall,
        scene,
        int(geometry_seed),
        NON_RNG_MEASUREMENT_SENTINEL,
        0,
        "map",
    )
    target = canonical_target(geometry.points_world)
    reference = reference_pose_from_development(protocol)
    indices = eligible_parent_indices(target, reference)
    source, source_f64, parents = source_from_parent_indices(
        target, indices, reference
    )
    closure = quantization_closure(
        source_points=source,
        source_float64=source_f64,
        parent_points_map_float64=parents,
        reference_pose=reference,
    )
    duplicate_count = int(len(indices) - len(np.unique(indices)))
    out_of_range = int(
        np.count_nonzero((indices < 0) | (indices >= len(target)))
    )
    closure_violations = int(
        closure["closure_residual_violation_count"]
        + closure["actual_error_bound_violation_count"]
    )
    valid = bool(
        indices.dtype == np.dtype("<i8")
        and indices.ndim == 1
        and indices.flags.c_contiguous
        and len(source) == len(indices)
        and duplicate_count == 0
        and out_of_range == 0
        and closure["quantization_closure_pass"] is True
        and closure_violations == 0
    )
    if not valid:
        raise V3SnapshotContractError("v3 IDEAL parent lineage validation failed")
    return {
        "closure": closure,
        "indices": indices,
        "lineage_closure_violation_count": closure_violations,
        "parent_points": parents,
        "reference": reference,
        "source": source,
        "target": target,
    }


def _nonideal_arrays(
    *, protocol: Any, firewall: ConfirmatoryV3SeedFirewall,
    scene: str, geometry_seed: int, measurement_seed: int,
    repeat_index: int, condition: str,
) -> dict[str, Any]:
    """Delegate the unchanged non-IDEAL numerical path to the frozen builder."""

    import numpy as np
    from .phase_b_generator import SnapshotKey, build_snapshot, reference_matrix

    if condition == "IDEAL_MATCHED":
        raise V3SnapshotContractError("IDEAL must use the parent-lineage path")
    bundle = build_snapshot(
        protocol,
        firewall,
        SnapshotKey(
            scene_variant=scene,
            geometry_seed=int(geometry_seed),
            measurement_seed=int(measurement_seed),
            repeat_index=int(repeat_index),
            noise_condition=condition,
        ),
    )
    return {
        "record_checksums": dict(bundle.checksums),
        "reference": np.ascontiguousarray(
            reference_matrix(bundle.reference_pose), dtype="<f8"
        ),
        "source": np.ascontiguousarray(bundle.scan_points, dtype="<f4"),
        "target": np.ascontiguousarray(bundle.map_points, dtype="<f4"),
    }


def _lineage_fields(
    *, lineage: Mapping[str, Any] | None, source: Any, target: Any,
) -> dict[str, Any]:
    if lineage is None:
        return {
            "actual_error_bound_violation_count": None,
            "closure_residual_max_m": None,
            "closure_residual_violation_count": None,
            "float64_guard_max_m": None,
            "lineage_closure_violation_count": 0,
            "lineage_validation_method": "NOT_APPLICABLE_NONIDEAL",
            "max_normalized_closure_ratio": None,
            "parent_index_count": 0,
            "parent_index_duplicate_count": 0,
            "parent_index_out_of_range_count": 0,
            "parent_index_unique_count": 0,
            "parent_points_map_f64_sha256": None,
            "predicted_quantization_max_m": None,
            "predicted_quantization_median_m": None,
            "predicted_quantization_q95_m": None,
            "quantization_closure_pass": False,
            "reconstruction_error_max_m": None,
            "reconstruction_error_median_m": None,
            "reconstruction_error_q95_m": None,
            "source_has_target_parent_lineage": False,
            "source_is_target_subset": False,
            "source_parent_row_count_match": False,
            "source_parent_target_indices_path": None,
            "source_parent_target_indices_sha256": None,
        }
    import numpy as np

    indices = lineage["indices"]
    closure = dict(lineage["closure"])
    count = int(len(indices))
    unique = int(len(np.unique(indices)))
    duplicate = count - unique
    out_of_range = int(np.count_nonzero((indices < 0) | (indices >= len(target))))
    row_match = bool(len(source) == count)
    violations = int(lineage["lineage_closure_violation_count"])
    valid = bool(
        indices.dtype == np.dtype("<i8")
        and indices.ndim == 1
        and indices.flags.c_contiguous
        and row_match
        and duplicate == 0
        and out_of_range == 0
        and closure["quantization_closure_pass"] is True
        and violations == 0
    )
    if not valid:
        raise V3SnapshotContractError("invalid v3 IDEAL lineage metadata")
    return {
        **closure,
        "lineage_closure_violation_count": violations,
        "lineage_validation_method": (
            "PARENT_INDEX_ROW_CORRESPONDENCE_PLUS_PHASE_A_QUANTIZATION_CLOSURE"
        ),
        "parent_index_count": count,
        "parent_index_duplicate_count": duplicate,
        "parent_index_out_of_range_count": out_of_range,
        "parent_index_unique_count": unique,
        "parent_points_map_f64_sha256": _raw_sha256(
            np.ascontiguousarray(lineage["parent_points"], dtype="<f8")
        ),
        "source_has_target_parent_lineage": True,
        "source_is_target_subset": True,
        "source_parent_row_count_match": row_match,
        "source_parent_target_indices_path": PARENT_INDEX_FILENAME,
        "source_parent_target_indices_sha256": _raw_sha256(indices),
    }


def _metadata(
    *, contract: Any, manifest: Mapping[str, Any], row: Mapping[str, Any],
    source: Any, target: Any, reference: Any, rng_count: int,
    lineage: Mapping[str, Any] | None, record_checksums: Mapping[str, Any],
) -> dict[str, Any]:
    from .full_synthetic_development_protocol import CONDITION_PARAMETERS
    from .phase_b_generator import DEVELOPMENT_PROTOCOL_SHA256, GENERATOR_SHA256

    raw = {
        "source_checksum": _raw_sha256(source),
        "target_checksum": _raw_sha256(target),
        "reference_pose_checksum": _raw_sha256(reference),
    }
    parent_sha = None if lineage is None else _raw_sha256(lineage["indices"])
    snapshot_checksum = _canonical_sha256(
        {
            "snapshot_id": row["planned_snapshot_id"],
            **raw,
            "source_parent_target_indices_sha256": parent_sha,
        }
    )
    parameters = CONDITION_PARAMETERS[row["condition"]]
    return {
        "array_file_sha256": {},
        "condition": row["condition"],
        "confirmatory_rng_instantiation_count": int(rng_count),
        "development_protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
        "dropout_parameters": {
            "map_dropout_fraction": float(parameters["map_dropout_fraction"]),
            "scan_dropout_fraction": float(parameters["scan_dropout_fraction"]),
        },
        "formal_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "generator_record_checksums": dict(record_checksums),
        "generator_sha256": GENERATOR_SHA256,
        "geometry_seed": int(row["geometry_seed"]),
        "independent_sampling": bool(parameters["independent_sampling"]),
        "initial_pose": "reference_pose_exact",
        "lineage_schema_version": contract.LINEAGE_SCHEMA,
        "measurement_seed": row["measurement_seed"],
        "metadata_payload_sha256": "",
        "noise_parameters": {
            "map_noise_sigma_m": float(parameters["map_noise_sigma_m"]),
            "scan_noise_sigma_m": float(parameters["scan_noise_sigma_m"]),
        },
        "planned_snapshot_id": row["planned_snapshot_id"],
        "reference_pose_checksum": raw["reference_pose_checksum"],
        "repeat_index": int(row["repeat_index"]),
        "scene_variant": row["scene_variant"],
        "schema_version": contract.METADATA_SCHEMA,
        "seed_namespace": contract.NAMESPACE,
        "snapshot_builder_contract_version": SNAPSHOT_BUILDER_CONTRACT_VERSION,
        "snapshot_checksum": snapshot_checksum,
        "snapshot_id": row["planned_snapshot_id"],
        "snapshot_schema_version": contract.SNAPSHOT_SCHEMA,
        "source_checksum": raw["source_checksum"],
        "source_point_count": int(len(source)),
        "target_checksum": raw["target_checksum"],
        "target_point_count": int(len(target)),
        **_lineage_fields(lineage=lineage, source=source, target=target),
    }


def _build_v3_snapshot(
    repository: Path, contract: Any, manifest: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    from .full_synthetic_snapshot_builder import (
        load_full_synthetic_generator_protocol,
    )

    protocol = load_full_synthetic_generator_protocol(repository)
    firewall = ConfirmatoryV3SeedFirewall(protocol, contract)
    condition = row["condition"]
    if condition == "IDEAL_MATCHED":
        lineage = _lineage_arrays(
            protocol=protocol,
            firewall=firewall,
            scene=row["scene_variant"],
            geometry_seed=row["geometry_seed"],
        )
        source = lineage["source"]
        target = lineage["target"]
        reference = lineage["reference"]
        record_checksums: dict[str, Any] = {}
    else:
        internal_measurement = (
            NON_RNG_MEASUREMENT_SENTINEL
            if row["measurement_seed"] is None
            else int(row["measurement_seed"])
        )
        generated = _nonideal_arrays(
            protocol=protocol,
            firewall=firewall,
            scene=row["scene_variant"],
            geometry_seed=row["geometry_seed"],
            measurement_seed=internal_measurement,
            repeat_index=row["repeat_index"],
            condition=condition,
        )
        source = generated["source"]
        target = generated["target"]
        reference = generated["reference"]
        record_checksums = dict(generated["record_checksums"])
        lineage = None
    metadata = _metadata(
        contract=contract,
        manifest=manifest,
        row=row,
        source=source,
        target=target,
        reference=reference,
        rng_count=firewall.rng_instantiation_count,
        lineage=lineage,
        record_checksums=record_checksums,
    )
    expected_rng = 3 if condition == "FULL_NOISE" else 0
    if firewall.rng_instantiation_count != expected_rng:
        raise V3SnapshotContractError(
            "v3 snapshot used an unexpected number of RNG streams"
        )
    return {
        "firewall_audit": firewall.report(),
        "metadata": metadata,
        "parent_indices": None if lineage is None else lineage["indices"],
        "reference": reference,
        "source": source,
        "target": target,
    }


def _regular_file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise V3SnapshotContractError(f"regular snapshot file required: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
    finally:
        os.close(descriptor)


def _load_npy(path: Path) -> Any:
    import numpy as np

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise V3SnapshotContractError(f"regular NumPy file required: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            return np.load(stream, allow_pickle=False)
    finally:
        os.close(descriptor)


def _snapshot_inventory(directory: Path, expected: set[str]) -> None:
    try:
        metadata = os.lstat(directory)
    except FileNotFoundError as error:
        raise V3SnapshotContractError(
            f"v3 snapshot directory is missing: {directory}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise V3SnapshotContractError("v3 snapshot entry is not a real directory")
    names: set[str] = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            item = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(item.st_mode) or stat.S_ISLNK(item.st_mode):
                raise V3SnapshotContractError(
                    f"v3 snapshot contains non-regular entry: {entry.name}"
                )
            names.add(entry.name)
    if names != expected:
        raise V3SnapshotContractError("v3 snapshot file inventory mismatch")


def _lineage_recomputed(
    source: Any, target: Any, reference: Any, indices: Any
) -> dict[str, Any]:
    import numpy as np
    from . import phase_b_generator as _local_generator  # noqa: F401
    from .phase_b_generator_frozen.zero_perturbation.backend_phase_a_v1_2 import (
        quantization_closure,
        source_from_parent_indices,
    )

    expected, source_f64, parents = source_from_parent_indices(
        target, indices, reference
    )
    closure = quantization_closure(
        source_points=source,
        source_float64=source_f64,
        parent_points_map_float64=parents,
        reference_pose=reference,
    )
    return {
        "closure": closure,
        "lineage_closure_violation_count": int(
            closure["closure_residual_violation_count"]
            + closure["actual_error_bound_violation_count"]
        ),
        "parent_points_map_f64_sha256": _raw_sha256(
            np.ascontiguousarray(parents, dtype="<f8")
        ),
        "row_correspondence_pass": bool(np.array_equal(source, expected)),
    }


def _validate_array_contract(source: Any, target: Any, reference: Any) -> None:
    import numpy as np

    if (
        source.dtype != np.dtype("<f4")
        or target.dtype != np.dtype("<f4")
        or reference.dtype != np.dtype("<f8")
        or source.ndim != 2
        or source.shape[1:] != (3,)
        or target.ndim != 2
        or target.shape[1:] != (3,)
        or reference.shape != (4, 4)
        or not all(value.flags.c_contiguous for value in (source, target, reference))
        or not all(np.all(np.isfinite(value)) for value in (source, target, reference))
    ):
        raise V3SnapshotContractError("v3 snapshot array contract mismatch")


def _write_v3_snapshot_atomic(cache_root: Path, value: Mapping[str, Any]) -> Path:
    import numpy as np
    from .runtime_lifecycle_io import atomic_publish_directory, canonical_json_bytes

    metadata = dict(value["metadata"])
    snapshot_id = str(metadata["snapshot_id"])
    destination = cache_root / snapshot_id
    if destination.parent != cache_root or destination.exists():
        raise FileExistsError(f"unsafe or existing v3 snapshot: {destination}")
    _validate_array_contract(value["source"], value["target"], value["reference"])

    def populate(staging: Path) -> None:
        arrays = [
            ("source_points.npy", value["source"]),
            ("target_points.npy", value["target"]),
            ("reference_pose.npy", value["reference"]),
        ]
        if value.get("parent_indices") is not None:
            arrays.append((PARENT_INDEX_FILENAME, value["parent_indices"]))
        for name, array in arrays:
            with (staging / name).open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
        metadata["array_file_sha256"] = {
            name: _regular_file_sha256(staging / name) for name, _array in arrays
        }
        unsigned = dict(metadata)
        unsigned.pop("metadata_payload_sha256")
        metadata["metadata_payload_sha256"] = _canonical_sha256(unsigned)
        with (staging / "metadata.json").open("xb") as stream:
            stream.write(canonical_json_bytes(metadata))

    return atomic_publish_directory(destination, populate)


def read_v3_snapshot(
    cache_root: str | Path,
    plan: Mapping[str, Any],
    *,
    execution_context: Any,
    expected_lock_entry: Mapping[str, Any] | None,
    arrays: bool = True,
) -> dict[str, Any]:
    """Authenticate one snapshot under one explicit execution context."""

    import numpy as np
    from .runtime_lifecycle_io import read_canonical_json

    from .execution_context import ExecutionContext, ExecutionContextError

    if type(execution_context) is not ExecutionContext:
        raise ExecutionContextError(
            "EXECUTION_CONTEXT_CONTRACT_MISMATCH",
            field="execution_context",
            actual=type(execution_context).__name__,
            expected="ExecutionContext",
            detail="snapshot reader requires an explicit execution context",
        )
    row = execution_context.validate_snapshot_row(plan)
    policy = execution_context.snapshot_reader_policy
    expected_cache = execution_context.cache_root
    cache = _canonical_absolute(cache_root, "v3 snapshot cache")
    if cache != _canonical_absolute(expected_cache, "contract snapshot cache"):
        raise ExecutionContextError(
            "EXECUTION_CONTEXT_CACHE_MISMATCH",
            field="cache_root",
            actual=cache,
            expected=expected_cache,
            detail="snapshot read escaped the injected cache",
        )
    directory = cache / row["planned_snapshot_id"]
    if directory.parent != cache:
        raise V3SnapshotContractError("v3 snapshot ID is not a single path segment")
    expected_files = set(BASE_ARRAY_FILENAMES) | {"metadata.json"}
    lineage_required = row["condition"] in set(policy.lineage_required_conditions)
    if lineage_required:
        expected_files.add(PARENT_INDEX_FILENAME)
    _snapshot_inventory(directory, expected_files)
    metadata = read_canonical_json(directory / "metadata.json")
    if type(metadata) is not dict or set(metadata) != set(policy.metadata_fields):
        raise V3SnapshotContractError("v3 snapshot metadata fields changed")
    unsigned = dict(metadata)
    stored_metadata_sha = unsigned.pop("metadata_payload_sha256")
    if stored_metadata_sha != _canonical_sha256(unsigned):
        raise V3SnapshotContractError("v3 snapshot metadata payload SHA mismatch")
    expected_identity = {
        "condition": row["condition"],
        "geometry_seed": row["geometry_seed"],
        "measurement_seed": row["measurement_seed"],
        "planned_snapshot_id": row["planned_snapshot_id"],
        "repeat_index": row["repeat_index"],
        "scene_variant": row["scene_variant"],
        "snapshot_id": row["planned_snapshot_id"],
    }
    if (
        any(metadata.get(name) != value for name, value in expected_identity.items())
        or metadata.get("schema_version") != policy.metadata_schema
        or metadata.get("snapshot_schema_version") != policy.snapshot_schema_version
        or metadata.get("lineage_schema_version") != policy.lineage_schema_version
        or metadata.get("seed_namespace") != policy.seed_namespace
        or metadata.get("snapshot_builder_contract_version")
        != policy.snapshot_builder_contract_version
    ):
        raise V3SnapshotContractError("v3 snapshot metadata identity mismatch")
    file_sha = {
        name: _regular_file_sha256(directory / name)
        for name in sorted(expected_files)
    }
    array_sha = {
        name: file_sha[name] for name in expected_files if name.endswith(".npy")
    }
    if metadata["array_file_sha256"] != array_sha:
        raise V3SnapshotContractError("v3 snapshot array-file SHA mismatch")
    if expected_lock_entry is not None:
        if (
            type(expected_lock_entry) is not dict
            or set(expected_lock_entry) != set(policy.lock_entry_fields)
        ):
            raise V3SnapshotContractError("v3 snapshot lock entry fields changed")
        if (
            expected_lock_entry.get("snapshot_id") != row["planned_snapshot_id"]
            or expected_lock_entry.get("file_sha256") != file_sha
        ):
            raise V3SnapshotContractError("v3 snapshot lock file binding mismatch")

    source = _load_npy(directory / "source_points.npy")
    target = _load_npy(directory / "target_points.npy")
    reference = _load_npy(directory / "reference_pose.npy")
    _validate_array_contract(source, target, reference)
    raw = {
        "source_checksum": _raw_sha256(source),
        "target_checksum": _raw_sha256(target),
        "reference_pose_checksum": _raw_sha256(reference),
    }
    parent_indices = None
    parent_sha = None
    if lineage_required:
        parent_indices = _load_npy(directory / PARENT_INDEX_FILENAME)
        if (
            parent_indices.dtype != np.dtype("<i8")
            or parent_indices.ndim != 1
            or not parent_indices.flags.c_contiguous
        ):
            raise V3SnapshotContractError("v3 parent-index array contract mismatch")
        parent_sha = _raw_sha256(parent_indices)
        recomputed = _lineage_recomputed(
            source, target, reference, parent_indices
        )
        duplicate = int(len(parent_indices) - len(np.unique(parent_indices)))
        out_of_range = int(
            np.count_nonzero(
                (parent_indices < 0) | (parent_indices >= len(target))
            )
        )
        expected_lineage = {
            "parent_index_count": len(parent_indices),
            "parent_index_duplicate_count": duplicate,
            "parent_index_out_of_range_count": out_of_range,
            "parent_index_unique_count": len(np.unique(parent_indices)),
            "parent_points_map_f64_sha256": recomputed[
                "parent_points_map_f64_sha256"
            ],
            "source_has_target_parent_lineage": True,
            "source_is_target_subset": True,
            "source_parent_row_count_match": len(source) == len(parent_indices),
            "source_parent_target_indices_path": PARENT_INDEX_FILENAME,
            "source_parent_target_indices_sha256": parent_sha,
            "lineage_closure_violation_count": recomputed[
                "lineage_closure_violation_count"
            ],
            **recomputed["closure"],
        }
        if (
            any(metadata.get(name) != value for name, value in expected_lineage.items())
            or not recomputed["row_correspondence_pass"]
            or duplicate != 0
            or out_of_range != 0
            or recomputed["lineage_closure_violation_count"] != 0
        ):
            raise V3SnapshotContractError("v3 IDEAL lineage validation failed")
    else:
        expected_nonideal = {
            "source_has_target_parent_lineage": False,
            "source_is_target_subset": False,
            "source_parent_target_indices_path": None,
            "source_parent_target_indices_sha256": None,
            "parent_index_count": 0,
            "parent_index_unique_count": 0,
            "parent_index_duplicate_count": 0,
            "parent_index_out_of_range_count": 0,
            "source_parent_row_count_match": False,
            "parent_points_map_f64_sha256": None,
            "quantization_closure_pass": False,
        }
        if any(
            metadata.get(name) != value
            for name, value in expected_nonideal.items()
        ):
            raise V3SnapshotContractError(
                "v3 non-IDEAL snapshot forged parent lineage"
            )
    snapshot_checksum = _canonical_sha256(
        {
            "snapshot_id": row["planned_snapshot_id"],
            **raw,
            "source_parent_target_indices_sha256": parent_sha,
        }
    )
    expected_rng = policy.expected_rng_counts[row["condition"]]
    if (
        any(metadata.get(name) != value for name, value in raw.items())
        or metadata.get("snapshot_checksum") != snapshot_checksum
        or metadata.get("source_point_count") != len(source)
        or metadata.get("target_point_count") != len(target)
        or metadata.get("confirmatory_rng_instantiation_count") != expected_rng
    ):
        raise V3SnapshotContractError("v3 snapshot checksum or count mismatch")
    if expected_lock_entry is not None:
        expected_values = {
            **raw,
            "metadata_payload_sha256": stored_metadata_sha,
            "snapshot_checksum": snapshot_checksum,
            "source_parent_target_indices_sha256": parent_sha,
        }
        if any(
            expected_lock_entry.get(name) != value
            for name, value in expected_values.items()
        ):
            raise V3SnapshotContractError("v3 snapshot lock payload mismatch")
    result: dict[str, Any] = {
        "directory": directory,
        "file_sha256": file_sha,
        "metadata": metadata,
        "reference_pose_checksum": raw["reference_pose_checksum"],
        "snapshot_checksum": snapshot_checksum,
        "source_checksum": raw["source_checksum"],
        "target_checksum": raw["target_checksum"],
    }
    if arrays:
        result.update(
            source=source,
            target=target,
            reference=reference,
            parent_indices=parent_indices,
        )
    return result


def read_formal_v3_snapshot(
    cache_root: str | Path,
    plan: Mapping[str, Any],
    *,
    expected_lock_entry: Mapping[str, Any] | None,
    arrays: bool = True,
) -> dict[str, Any]:
    """Explicit formal compatibility wrapper over the generic reader."""

    contract = _contract_module()
    context_factory = getattr(contract, "formal_execution_context", None)
    if not callable(context_factory):
        raise V3SnapshotContractError("formal execution context is unavailable")
    return read_v3_snapshot(
        cache_root,
        plan,
        execution_context=context_factory(),
        expected_lock_entry=expected_lock_entry,
        arrays=arrays,
    )


def _lock_entry(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    metadata = snapshot["metadata"]
    return {
        "condition": metadata["condition"],
        "confirmatory_rng_instantiation_count": metadata[
            "confirmatory_rng_instantiation_count"
        ],
        "file_sha256": dict(snapshot["file_sha256"]),
        "geometry_seed": metadata["geometry_seed"],
        "measurement_seed": metadata["measurement_seed"],
        "metadata_payload_sha256": metadata["metadata_payload_sha256"],
        "reference_pose_checksum": snapshot["reference_pose_checksum"],
        "repeat_index": metadata["repeat_index"],
        "scene_variant": metadata["scene_variant"],
        "snapshot_checksum": snapshot["snapshot_checksum"],
        "snapshot_id": metadata["snapshot_id"],
        "source_checksum": snapshot["source_checksum"],
        "source_parent_target_indices_sha256": metadata[
            "source_parent_target_indices_sha256"
        ],
        "target_checksum": snapshot["target_checksum"],
    }


def _build_snapshot_lock(
    cache: Path,
    plans: Sequence[Mapping[str, Any]],
    *,
    contract: Any,
    manifest: Mapping[str, Any],
    execution_context: Any,
) -> dict[str, Any]:
    entries = [
        _lock_entry(
            read_v3_snapshot(
                cache,
                plan,
                execution_context=execution_context,
                expected_lock_entry=None,
                arrays=False,
            )
        )
        for plan in plans
    ]
    identity_hasher = getattr(contract, "canonical_identity_sha256", _canonical_sha256)
    core = {
        "condition_snapshot_counts": dict(
            sorted(Counter(row["condition"] for row in entries).items())
        ),
        "confirmatory_rng_instantiation_count": sum(
            int(row["confirmatory_rng_instantiation_count"]) for row in entries
        ),
        "formal_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "lineage_schema_version": contract.LINEAGE_SCHEMA,
        "planned_snapshot_count": len(entries),
        "planned_snapshot_identity_sha256": identity_hasher(list(plans)),
        "schema_version": contract.SNAPSHOT_LOCK_SCHEMA,
        "seed_namespace": contract.NAMESPACE,
        "snapshot_builder_contract_version": SNAPSHOT_BUILDER_CONTRACT_VERSION,
        "snapshot_schema_version": contract.SNAPSHOT_SCHEMA,
        "snapshots": entries,
    }
    return {**core, "snapshot_lock_payload_sha256": _canonical_sha256(core)}


def _validate_snapshot_lock(
    lock_path: Path,
    cache: Path,
    plans: Sequence[Mapping[str, Any]],
    *,
    contract: Any,
    manifest: Mapping[str, Any],
    execution_context: Any,
) -> dict[str, Any]:
    from .runtime_lifecycle_io import read_canonical_json

    value = read_canonical_json(lock_path)
    if type(value) is not dict:
        raise V3SnapshotContractError("v3 snapshot lock must be an object")
    unsigned = {
        name: item
        for name, item in value.items()
        if name != "snapshot_lock_payload_sha256"
    }
    entries = value.get("snapshots")
    expected_counts = dict(
        sorted(Counter(row["condition"] for row in plans).items())
    )
    identity_hasher = getattr(contract, "canonical_identity_sha256", _canonical_sha256)
    if (
        value.get("snapshot_lock_payload_sha256") != _canonical_sha256(unsigned)
        or value.get("schema_version") != contract.SNAPSHOT_LOCK_SCHEMA
        or value.get("snapshot_schema_version") != contract.SNAPSHOT_SCHEMA
        or value.get("lineage_schema_version") != contract.LINEAGE_SCHEMA
        or value.get("snapshot_builder_contract_version")
        != SNAPSHOT_BUILDER_CONTRACT_VERSION
        or value.get("seed_namespace") != contract.NAMESPACE
        or value.get("formal_manifest_payload_sha256")
        != manifest["manifest_payload_sha256"]
        or value.get("planned_snapshot_count") != len(plans)
        or value.get("planned_snapshot_identity_sha256")
        != identity_hasher(list(plans))
        or value.get("condition_snapshot_counts") != expected_counts
        or value.get("confirmatory_rng_instantiation_count")
        != 3 * expected_counts.get("FULL_NOISE", 0)
        or type(entries) is not list
        or len(entries) != len(plans)
        or [entry.get("snapshot_id") for entry in entries]
        != [row["planned_snapshot_id"] for row in plans]
    ):
        raise V3SnapshotContractError("v3 snapshot lock identity mismatch")
    for plan, entry in zip(plans, entries):
        read_v3_snapshot(
            cache,
            plan,
            execution_context=execution_context,
            expected_lock_entry=entry,
            arrays=False,
        )
    return value


def _cache_inventory(cache: Path, expected_ids: set[str]) -> None:
    if not cache.is_dir() or cache.is_symlink():
        raise V3SnapshotContractError("v3 snapshot cache is not a real directory")
    actual: set[str] = set()
    with os.scandir(cache) as entries:
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise V3SnapshotContractError(
                    f"v3 snapshot cache contains non-directory entry: {entry.name}"
                )
            actual.add(entry.name)
    if actual != expected_ids:
        raise V3SnapshotContractError("v3 snapshot cache inventory mismatch")


def prepare_v3_snapshots(
    repository: str | Path,
    cache_root: str | Path,
    plans: Sequence[Mapping[str, Any]],
    *,
    lock_path: str | Path,
    resume: bool,
) -> dict[str, Any]:
    """Construct or resume the exact 595-snapshot v3 external cache."""

    from .runtime_lifecycle_io import atomic_create_canonical_json

    if type(resume) is not bool:
        raise TypeError("v3 snapshot resume state must be bool")
    root = _canonical_absolute(repository, "harness repository")
    contract, manifest, _manifest_path = _load_manifest(root)
    context_factory = getattr(contract, "formal_execution_context", None)
    if not callable(context_factory):
        raise V3SnapshotContractError("formal execution context is unavailable")
    execution_context = context_factory()
    normalized = _normalize_plans(root, contract, manifest, plans)
    cache, lock = _formal_paths(contract, manifest, cache_root, lock_path)
    assert lock is not None
    _formal_authorized(root, contract)

    if lock.exists():
        if not resume:
            raise PermissionError(
                "fresh v3 snapshot preparation found an existing lock"
            )
        value = _validate_snapshot_lock(
            lock,
            cache,
            normalized,
            contract=contract,
            manifest=manifest,
            execution_context=execution_context,
        )
        return {
            "confirmatory_rng_instantiation_count_this_invocation": 0,
            "generated_snapshot_count": 0,
            "lock": value,
            "lock_path": lock,
            "resumed_snapshot_count": len(normalized),
        }

    if cache.exists() and not resume:
        raise PermissionError(
            "fresh v3 snapshot preparation found an existing cache"
        )

    generated = 0
    resumed_count = 0
    rng_count = 0
    for row in normalized:
        destination = cache / row["planned_snapshot_id"]
        if destination.exists() or destination.is_symlink():
            read_v3_snapshot(
                cache,
                row,
                execution_context=execution_context,
                expected_lock_entry=None,
                arrays=False,
            )
            resumed_count += 1
            continue
        value = _build_v3_snapshot(root, contract, manifest, row)
        rng_count += int(
            value["metadata"]["confirmatory_rng_instantiation_count"]
        )
        _write_v3_snapshot_atomic(cache, value)
        read_v3_snapshot(
            cache,
            row,
            execution_context=execution_context,
            expected_lock_entry=None,
            arrays=False,
        )
        generated += 1

    _cache_inventory(
        cache, {row["planned_snapshot_id"] for row in normalized}
    )
    value = _build_snapshot_lock(
        cache,
        normalized,
        contract=contract,
        manifest=manifest,
        execution_context=execution_context,
    )
    atomic_create_canonical_json(lock, value)
    authenticated = _validate_snapshot_lock(
        lock,
        cache,
        normalized,
        contract=contract,
        manifest=manifest,
        execution_context=execution_context,
    )
    return {
        "confirmatory_rng_instantiation_count_this_invocation": rng_count,
        "generated_snapshot_count": generated,
        "lock": authenticated,
        "lock_path": lock,
        "resumed_snapshot_count": resumed_count,
    }


__all__ = [
    "BASE_ARRAY_FILENAMES",
    "ConfirmatoryV3SeedFirewall",
    "METADATA_FIELDS",
    "PARENT_INDEX_FILENAME",
    "SNAPSHOT_BUILDER_CONTRACT_VERSION",
    "V3SnapshotContractError",
    "prepare_v3_snapshots",
    "read_formal_v3_snapshot",
    "read_v3_snapshot",
]
