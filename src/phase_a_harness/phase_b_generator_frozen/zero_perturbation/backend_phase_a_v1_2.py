"""Phase A v1.2 protocol, indexed lineage, and Stage-0 cache primitives.

This module is deliberately registration-backend free.  Stage 0 creates only
canonical snapshot inputs and provenance evidence; it cannot create a trial.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from capture_range.day2_development_scene import build_scene_geometry

from .backend_phase_a_protocol import (
    PROTOCOL_RELATIVE as V1_PROTOCOL_RELATIVE,
    PROTOCOL_SHA256 as V1_PROTOCOL_SHA256,
    BackendPhaseAProtocol,
    canonical_json_sha256,
    file_sha256,
    load_backend_phase_a_protocol,
)
from .protocol import DevelopmentSeedFirewall, ZeroPerturbationProtocol, load_protocol


V1_2_PROTOCOL_RELATIVE = Path("configs/zero_perturbation/backend_phase_a_v1_2.yaml")
V1_2_PROTOCOL_SHA256 = "d412c9bb74fd4e3828c830d17144e1fb4440a27936941ce000e9daa87afabf81"
V1_2_DOCUMENT_RELATIVE = Path(
    "docs/zero_perturbation_backend_phase_a_v1_2_protocol.md"
)
V1_2_DOCUMENT_SHA256 = "5a386b211b72f72e9e4e3fc1f5ee0b4c0f12e8c45d32dba3dab53fbae349d2b4"
V1_2_PROTOCOL_LOCK_TAG = (
    "archive/zero-perturbation-backend-phase-a-v1.2-protocol-lock"
)
V1_2_LOCK_RELATIVE = Path(
    "artifacts/current/zero_perturbation_backend_phase_a_v1_2_lock"
)
V1_2_STAGE0_ARTIFACT_RELATIVE = Path(
    "artifacts/current/zero_perturbation_backend_phase_a_v1_2_stage0"
)
V1_2_INVALIDATION_RELATIVE = Path(
    "artifacts/current/zero_perturbation_backend_phase_a_v1_1_invalidation"
)
V1_2_CACHE_RELATIVE = Path(
    "data/zero_perturbation/backend_phase_a_v1_2_stage0"
)
SNAPSHOT_LOCK_SCHEMA = "backend_phase_a_v1_2_snapshot_lock_v1"
PROTOCOL_LOCK_SCHEMA = "backend_phase_a_v1_2_protocol_lock_v1"
STAGE0_METADATA_SCHEMA = "backend_phase_a_v1_2_stage0_snapshot_v1"
PARENT_SELECTION_PREFIX = "phase-a-v1.2-parent-selection"
ARRAY_FILENAMES = (
    "source_points.npy",
    "target_points.npy",
    "reference_pose.npy",
    "source_parent_target_indices.npy",
)


class Stage0ContractError(RuntimeError):
    """A frozen Stage-0 contract or cache invariant was violated."""


class CorruptSnapshotCache(Stage0ContractError):
    """An existing cache entry is incomplete, mismatched, or corrupt."""


@dataclass(frozen=True)
class CanonicalSnapshot:
    source_points: np.ndarray
    target_points: np.ndarray
    reference_pose: np.ndarray
    parent_indices: np.ndarray
    source_float64: np.ndarray
    metadata: Mapping[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorruptSnapshotCache(f"cannot parse JSON: {path}") from error
    if type(value) is not dict:
        raise CorruptSnapshotCache(f"JSON root is not an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as error:
        raise Stage0ContractError(f"cannot parse planned CSV: {path}") from error


def raw_array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise Stage0ContractError("raw checksum requires a C-contiguous array")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_v1_2_protocol(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    source = repository / V1_2_PROTOCOL_RELATIVE
    document = repository / V1_2_DOCUMENT_RELATIVE
    if file_sha256(source) != V1_2_PROTOCOL_SHA256:
        raise Stage0ContractError("Phase A v1.2 protocol SHA changed")
    if file_sha256(document) != V1_2_DOCUMENT_SHA256:
        raise Stage0ContractError("Phase A v1.2 protocol document SHA changed")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise Stage0ContractError("invalid Phase A v1.2 YAML") from error
    if type(value) is not dict:
        raise Stage0ContractError("Phase A v1.2 root is not a mapping")
    identity = value.get("protocol", {})
    if (
        identity.get("protocol_type")
        != "dual_independent_backend_phase_a_qualification"
        or float(identity.get("protocol_version", 0.0)) != 1.2
        or identity.get("amendment_type")
        != "provenance_contract_and_execution_staging_only"
    ):
        raise Stage0ContractError("Phase A v1.2 amendment identity changed")
    scientific = value.get("scientific_contract", {})
    if (
        scientific.get("authoritative_base_path")
        != V1_PROTOCOL_RELATIVE.as_posix()
        or scientific.get("authoritative_base_sha256") != V1_PROTOCOL_SHA256
    ):
        raise Stage0ContractError("Phase A v1.2 scientific base changed")
    load_backend_phase_a_protocol(repository)
    return value


def scientific_contract_diff(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    amendment = load_v1_2_protocol(repository)
    base = load_backend_phase_a_protocol(repository)
    required = amendment["scientific_contract"]["required_difference_counts"]
    inherited = amendment["scientific_contract"]["inherited_byte_exact_sections"]
    counters = {
        name: int(value)
        for name, value in required.items()
        if name.endswith("_difference_count")
    }
    output = {
        "schema_version": "backend_phase_a_v1_1_to_v1_2_diff_v1",
        "scientific_base_protocol_path": V1_PROTOCOL_RELATIVE.as_posix(),
        "scientific_base_protocol_sha256": V1_PROTOCOL_SHA256,
        "v1_2_protocol_path": V1_2_PROTOCOL_RELATIVE.as_posix(),
        "v1_2_protocol_sha256": V1_2_PROTOCOL_SHA256,
        "inherited_section_sha256": {
            name: canonical_json_sha256(base.data[name]) for name in inherited
        },
        **counters,
        "provenance_implementation_difference_count": 5,
        "execution_staging_difference_count": 2,
    }
    zero_names = tuple(
        name for name in counters if not name.startswith(("provenance_", "execution_"))
    )
    output["scientific_contract_unchanged"] = all(
        output[name] == 0 for name in zero_names
    )
    return output


def planned_rows(root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = load_backend_phase_a_protocol(root)
    return (
        [row.row() for row in base.planned_snapshots()],
        [row.row() for row in base.planned_trials()],
    )


def _canonical_array(value: Any, dtype: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, dtype=np.dtype(dtype), order="C", copy=True)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise Stage0ContractError(f"invalid canonical array shape: {array.shape}")
    if not np.isfinite(array).all() or not array.flags.c_contiguous:
        raise Stage0ContractError("canonical coordinates must be finite and C-contiguous")
    return array


def canonical_target(points: Any) -> np.ndarray:
    """Quantize target coordinates exactly once to the canonical representation."""

    return _canonical_array(points, "<f4", (3,))


def reference_pose_from_development(protocol: ZeroPerturbationProtocol) -> np.ndarray:
    common = protocol.section("scene_generation")["common"]
    roll, pitch, yaw = np.deg2rad(
        np.asarray(common["reference_pose_rpy_deg"], dtype=np.float64)
    )
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    pose = np.eye(4, dtype="<f8", order="C")
    pose[:3, :3] = rz @ ry @ rx
    pose[:3, 3] = np.asarray(
        common["reference_pose_translation_world"], dtype=np.float64
    )
    return np.ascontiguousarray(pose, dtype="<f8")


def eligible_parent_indices(target: np.ndarray, reference_pose: np.ndarray) -> np.ndarray:
    points = np.asarray(target, dtype=np.float64)
    translation = np.asarray(reference_pose, dtype=np.float64)[:3, 3]
    ranges = np.linalg.norm(points - translation, axis=1)
    return np.flatnonzero((ranges >= 0.30) & (ranges <= 20.0)).astype("<i8")


def select_parent_indices(
    *,
    target: np.ndarray,
    reference_pose: np.ndarray,
    scene_variant: str,
    geometry_seed: int,
    measurement_seed: int,
    repeat_index: int,
) -> np.ndarray:
    """Apply the prospectively frozen deterministic parent-subset rule."""

    eligible = eligible_parent_indices(target, reference_pose)
    selected: list[int] = []
    for parent_index in eligible.tolist():
        payload = (
            f"{PARENT_SELECTION_PREFIX}|scene={scene_variant}"
            f"|geometry_seed={int(geometry_seed)}"
            f"|measurement_seed={int(measurement_seed)}"
            f"|repeat={int(repeat_index)}|parent_index={int(parent_index)}"
        ).encode("utf-8")
        if hashlib.sha256(payload).digest()[0] % 4 != 0:
            selected.append(int(parent_index))
    if not selected:
        raise Stage0ContractError("deterministic source subset is empty")
    return np.ascontiguousarray(selected, dtype="<i8")


def source_from_parent_indices(
    target: np.ndarray, parent_indices: np.ndarray, reference_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_f32 = np.asarray(target)
    indices = np.asarray(parent_indices)
    if target_f32.dtype != np.dtype("<f4") or indices.dtype != np.dtype("<i8"):
        raise Stage0ContractError("source lineage inputs have noncanonical dtype")
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise Stage0ContractError("parent indices must be a unique vector")
    if len(indices) == 0 or int(indices.min()) < 0 or int(indices.max()) >= len(target_f32):
        raise Stage0ContractError("parent index is outside canonical target")
    parents = target_f32[indices].astype(np.float64)
    transform = np.asarray(reference_pose, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    source_f64 = (rotation.T @ (parents - translation).T).T
    source_f32 = np.ascontiguousarray(source_f64, dtype="<f4")
    return source_f32, np.ascontiguousarray(source_f64), np.ascontiguousarray(parents)


def quantization_closure(
    *,
    source_points: np.ndarray,
    source_float64: np.ndarray,
    parent_points_map_float64: np.ndarray,
    reference_pose: np.ndarray,
) -> dict[str, Any]:
    source_q = np.asarray(source_points, dtype=np.float64)
    source_f64 = np.asarray(source_float64, dtype=np.float64)
    parents = np.asarray(parent_points_map_float64, dtype=np.float64)
    transform = np.asarray(reference_pose, dtype=np.float64)
    if source_q.shape != source_f64.shape or source_q.shape != parents.shape:
        raise Stage0ContractError("closure arrays have different shapes")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    reconstructed = (rotation @ source_q.T).T + translation
    actual = reconstructed - parents
    source_quantization = source_q - source_f64
    predicted = (rotation @ source_quantization.T).T
    residual = actual - predicted
    actual_norm = np.linalg.norm(actual, axis=1)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    scale = np.maximum.reduce(
        (
            np.ones(len(parents), dtype=np.float64),
            np.linalg.norm(parents, axis=1),
            np.linalg.norm(source_f64, axis=1),
            np.full(len(parents), np.linalg.norm(translation), dtype=np.float64),
        )
    )
    guard = 256.0 * np.finfo(np.float64).eps * scale
    normalized = residual_norm / guard
    residual_gate = residual_norm <= guard
    actual_gate = actual_norm <= predicted_norm + guard
    quantile = lambda values, q: float(np.quantile(values, q, method="linear"))
    return {
        "reconstruction_error_median_m": float(np.median(actual_norm)),
        "reconstruction_error_q95_m": quantile(actual_norm, 0.95),
        "reconstruction_error_max_m": float(np.max(actual_norm)),
        "predicted_quantization_median_m": float(np.median(predicted_norm)),
        "predicted_quantization_q95_m": quantile(predicted_norm, 0.95),
        "predicted_quantization_max_m": float(np.max(predicted_norm)),
        "closure_residual_max_m": float(np.max(residual_norm)),
        "float64_guard_max_m": float(np.max(guard)),
        "max_normalized_closure_ratio": float(np.max(normalized)),
        "closure_residual_violation_count": int(np.count_nonzero(~residual_gate)),
        "actual_error_bound_violation_count": int(np.count_nonzero(~actual_gate)),
        "quantization_closure_pass": bool(residual_gate.all() and actual_gate.all()),
    }


def _rotation_quality(reference_pose: np.ndarray) -> dict[str, Any]:
    rotation = np.asarray(reference_pose, dtype=np.float64)[:3, :3]
    determinant = float(np.linalg.det(rotation))
    defect = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    return {
        "reference_rotation_determinant": determinant,
        "reference_rotation_orthogonality_defect_fro": defect,
        "reference_rotation_quality_pass": bool(
            np.isfinite(rotation).all()
            and determinant > 0.0
            and defect <= 1.0e-12
            and abs(determinant - 1.0) <= 1.0e-12
        ),
    }


def build_canonical_snapshot(
    *,
    development: ZeroPerturbationProtocol,
    firewall: DevelopmentSeedFirewall,
    plan: Mapping[str, Any],
) -> CanonicalSnapshot:
    scene = str(plan["scene_variant"])
    geometry_seed = int(plan["geometry_seed_value"])
    measurement_seed = int(plan["measurement_seed_value"])
    repeat_index = int(plan["repeat_index"])
    if str(plan["condition"]) != "IDEAL_MATCHED":
        raise Stage0ContractError("Stage 0 accepts IDEAL_MATCHED only")
    geometry = build_scene_geometry(
        development,
        firewall,
        scene,
        geometry_seed,
        measurement_seed,
        repeat_index,
        "map",
    )
    target = canonical_target(geometry.points_world)
    reference = reference_pose_from_development(development)
    indices = select_parent_indices(
        target=target,
        reference_pose=reference,
        scene_variant=scene,
        geometry_seed=geometry_seed,
        measurement_seed=measurement_seed,
        repeat_index=repeat_index,
    )
    source, source_f64, parents = source_from_parent_indices(
        target, indices, reference
    )
    closure = quantization_closure(
        source_points=source,
        source_float64=source_f64,
        parent_points_map_float64=parents,
        reference_pose=reference,
    )
    checksums = {
        "source_raw_checksum": raw_array_sha256(source),
        "target_raw_checksum": raw_array_sha256(target),
        "reference_pose_raw_checksum": raw_array_sha256(reference),
        "parent_index_raw_checksum": raw_array_sha256(indices),
        "parent_point_checksum": raw_array_sha256(np.ascontiguousarray(parents, dtype="<f8")),
    }
    snapshot_checksum = canonical_json_sha256(
        {"snapshot_id": str(plan["snapshot_id"]), **checksums}
    )
    metadata = {
        "schema_version": STAGE0_METADATA_SCHEMA,
        "snapshot_id": str(plan["snapshot_id"]),
        "scene_variant": scene,
        "geometry_seed": geometry_seed,
        "geometry_seed_index": int(plan["geometry_seed_index"]),
        "measurement_seed": measurement_seed,
        "measurement_seed_index": int(plan["measurement_seed_index"]),
        "repeat_index": repeat_index,
        "condition": "IDEAL_MATCHED",
        "source_point_count": int(len(source)),
        "target_point_count": int(len(target)),
        "parent_index_checksum": checksums["parent_index_raw_checksum"],
        "parent_index_unique_count": int(len(np.unique(indices))),
        "parent_index_out_of_range_count": int(
            np.count_nonzero((indices < 0) | (indices >= len(target)))
        ),
        "parent_index_duplicate_count": int(len(indices) - len(np.unique(indices))),
        "source_parent_row_count_match": bool(len(source) == len(indices)),
        "source_parent_target_points_map_f64_checksum": checksums[
            "parent_point_checksum"
        ],
        "source_points_all_finite": bool(np.isfinite(source).all()),
        "target_points_all_finite": bool(np.isfinite(target).all()),
        "reference_pose_all_finite": bool(np.isfinite(reference).all()),
        **checksums,
        "snapshot_checksum": snapshot_checksum,
        **closure,
        **_rotation_quality(reference),
    }
    return CanonicalSnapshot(source, target, reference, indices, source_f64, metadata)


def snapshot_directory(cache_root: str | Path, snapshot_id: str) -> Path:
    parts = Path(str(snapshot_id)).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise Stage0ContractError("unsafe snapshot ID")
    return Path(cache_root).joinpath(*parts)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_npy(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _save_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_snapshot_atomic(
    cache_root: str | Path, snapshot: CanonicalSnapshot, *, resume: bool
) -> tuple[dict[str, Any], bool]:
    root = Path(cache_root)
    destination = snapshot_directory(root, str(snapshot.metadata["snapshot_id"]))
    if destination.exists():
        if not resume:
            raise FileExistsError(f"snapshot exists without --resume: {destination}")
        existing = validate_snapshot_directory(destination, snapshot.metadata)
        return existing, True
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir()
    try:
        arrays = {
            "source_points.npy": snapshot.source_points,
            "target_points.npy": snapshot.target_points,
            "reference_pose.npy": snapshot.reference_pose,
            "source_parent_target_indices.npy": snapshot.parent_indices,
        }
        for filename, array in arrays.items():
            _save_npy(temporary / filename, array)
        metadata = dict(snapshot.metadata)
        metadata["array_file_sha256"] = {
            filename: file_sha256(temporary / filename) for filename in ARRAY_FILENAMES
        }
        payload = dict(metadata)
        metadata["metadata_payload_sha256"] = canonical_json_sha256(payload)
        _save_json(temporary / "metadata.json", metadata)
        _fsync_directory(temporary)
        _fsync_directory(destination.parent)
        if destination.exists():
            raise CorruptSnapshotCache("snapshot appeared before atomic rename")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_snapshot_directory(destination, snapshot.metadata), False


def _load_array(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise CorruptSnapshotCache(f"cannot load array: {path}") from error


def validate_snapshot_directory(
    directory: str | Path, expected_metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    path = Path(directory)
    expected_files = {*ARRAY_FILENAMES, "metadata.json"}
    actual_files = {child.name for child in path.iterdir() if child.is_file()}
    if actual_files != expected_files or any(child.is_dir() for child in path.iterdir()):
        raise CorruptSnapshotCache(f"snapshot file set changed: {path}")
    metadata = _read_json(path / "metadata.json")
    payload = {
        key: value for key, value in metadata.items() if key != "metadata_payload_sha256"
    }
    if canonical_json_sha256(payload) != metadata.get("metadata_payload_sha256"):
        raise CorruptSnapshotCache("metadata payload SHA mismatch")
    file_hashes = metadata.get("array_file_sha256", {})
    if any(file_sha256(path / name) != file_hashes.get(name) for name in ARRAY_FILENAMES):
        raise CorruptSnapshotCache("array file SHA mismatch")
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise CorruptSnapshotCache(f"existing snapshot metadata changed: {key}")
    arrays = {
        "source": _load_array(path / "source_points.npy"),
        "target": _load_array(path / "target_points.npy"),
        "reference": _load_array(path / "reference_pose.npy"),
        "indices": _load_array(path / "source_parent_target_indices.npy"),
    }
    expected_specs = {
        "source": (np.dtype("<f4"), 2, (3,)),
        "target": (np.dtype("<f4"), 2, (3,)),
        "reference": (np.dtype("<f8"), 2, (4,)),
        "indices": (np.dtype("<i8"), 1, ()),
    }
    for name, (dtype, ndim, tail) in expected_specs.items():
        array = arrays[name]
        if (
            array.dtype != dtype
            or array.ndim != ndim
            or array.shape[1:] != tail
            or not array.flags.c_contiguous
        ):
            raise CorruptSnapshotCache(f"noncanonical cached array: {name}")
    raw_expected = {
        "source": "source_raw_checksum",
        "target": "target_raw_checksum",
        "reference": "reference_pose_raw_checksum",
        "indices": "parent_index_raw_checksum",
    }
    if any(
        raw_array_sha256(arrays[name]) != metadata.get(field)
        for name, field in raw_expected.items()
    ):
        raise CorruptSnapshotCache("cached raw checksum mismatch")
    return metadata


def validate_protocol_lock(path: str | Path, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    lock = _read_json(Path(path).resolve())
    if lock.get("schema_version") != PROTOCOL_LOCK_SCHEMA:
        raise Stage0ContractError("unknown or old Phase A protocol lock")
    stored = lock.get("lock_payload_sha256")
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if canonical_json_sha256(payload) != stored:
        raise Stage0ContractError("v1.2 protocol lock payload SHA mismatch")
    load_v1_2_protocol(repository)
    if (
        lock.get("protocol_sha256") != V1_2_PROTOCOL_SHA256
        or lock.get("PHASE_A_V1_2_PROTOCOL_LOCK_PASS") is not True
        or lock.get("stage0_snapshot_build_authorized") is not True
    ):
        raise Stage0ContractError("v1.2 protocol lock does not authorize Stage 0")
    for item in lock.get("implementation", {}).get("files", {}).values():
        candidate = repository / str(item.get("path", ""))
        if not candidate.is_file() or file_sha256(candidate) != item.get("sha256"):
            raise Stage0ContractError("v1.2 implementation SHA mismatch")
    plan_path = repository / str(lock.get("planned_snapshots_path", ""))
    if not plan_path.is_file() or file_sha256(plan_path) != lock.get(
        "planned_snapshots_sha256"
    ):
        raise Stage0ContractError("v1.2 planned snapshot manifest SHA mismatch")
    expected, _ = planned_rows(repository)
    actual = _read_csv(plan_path)
    normalized = [{key: str(value) for key, value in row.items()} for row in expected]
    if actual != normalized or len(actual) != 210:
        raise Stage0ContractError("v1.2 planned snapshot manifest changed")
    return lock


def validate_snapshot_lock(path: str | Path, root: str | Path) -> dict[str, Any]:
    lock_path = Path(path).resolve()
    if not lock_path.is_file():
        raise FileNotFoundError("--snapshot-lock file does not exist")
    lock = _read_json(lock_path)
    if lock.get("schema_version") != SNAPSHOT_LOCK_SCHEMA:
        raise Stage0ContractError("unknown or invalid Stage-0 snapshot lock")
    stored = lock.get("lock_payload_sha256")
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if canonical_json_sha256(payload) != stored:
        raise Stage0ContractError("Stage-0 snapshot lock payload SHA mismatch")
    if (
        lock.get("protocol_sha256") != V1_2_PROTOCOL_SHA256
        or lock.get("PHASE_A_V1_2_STAGE0_PASS") is not True
        or lock.get("PHASE_A_STAGE1_BACKEND_RUN_AUTHORIZED") is not True
        or lock.get("FORMAL_STAGE0_SNAPSHOT_BUILD_COUNT") != 210
        or lock.get("FORMAL_BACKEND_EXECUTION_COUNT") != 0
        or lock.get("FORMAL_TRIAL_RESULT_COUNT") != 0
    ):
        raise Stage0ContractError("Stage-0 snapshot lock does not authorize Stage 1")
    repository = Path(root).resolve()
    cache_root = repository / str(lock.get("snapshot_cache_root", ""))
    inventory = repository / str(lock.get("snapshot_inventory_path", ""))
    if not cache_root.is_dir() or not inventory.is_file():
        raise Stage0ContractError("locked Stage-0 cache or inventory is missing")
    if file_sha256(inventory) != lock.get("snapshot_inventory_sha256"):
        raise Stage0ContractError("locked snapshot inventory SHA mismatch")
    return lock


def implementation_hashes(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    paths = {
        "protocol_yaml": V1_2_PROTOCOL_RELATIVE.as_posix(),
        "protocol_markdown": V1_2_DOCUMENT_RELATIVE.as_posix(),
        "scene_generator": "src/capture_range/day2_development_scene.py",
        "development_protocol_loader": "src/zero_perturbation/protocol.py",
        "v1_plan_loader": "src/zero_perturbation/backend_phase_a_protocol.py",
        "stage0_engine": "src/zero_perturbation/backend_phase_a_v1_2.py",
        "stage0_independent_verifier": "src/zero_perturbation/backend_phase_a_stage0_verification.py",
        "stage0_builder_script": "scripts/169_build_backend_phase_a_stage0.py",
        "stage0_verifier_script": "scripts/170_verify_backend_phase_a_stage0.py",
        "protocol_lock_publisher": "scripts/172_prepare_backend_phase_a_v1_2_lock.py",
        "stage0_artifact_engine": "src/zero_perturbation/backend_phase_a_stage0_artifact.py",
        "stage0_artifact_publisher": "scripts/173_publish_backend_phase_a_v1_2_stage0.py",
        "stage0_artifact_verifier": "scripts/174_verify_backend_phase_a_v1_2_stage0.py",
        "stage1_engine": "src/zero_perturbation/backend_phase_a_stage1.py",
        "stage1_runner": "scripts/168_run_backend_phase_a.py",
        "open3d_backend": "src/zero_perturbation/open3d_backend.py",
        "pcl_python_adapter": "src/zero_perturbation/pcl_backend.py",
        "pcl_cli_source": "tools/pcl_point_to_plane/pcl_point_to_plane_cli.cpp",
        "rotation_metric": "src/zero_perturbation/rotation_metrics.py",
    }
    files = {
        label: {"path": path, "sha256": file_sha256(repository / path)}
        for label, path in paths.items()
    }
    result = {"schema_version": "backend_phase_a_v1_2_implementation_hashes_v1", "files": files}
    result["implementation_sha256"] = canonical_json_sha256(result)
    return result


def build_stage0_cache(
    *,
    root: str | Path,
    protocol_lock: str | Path,
    run_id: str,
    output_dir: str | Path,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    """Build the complete canonical cache without importing a registration backend."""

    repository = Path(root).resolve()
    lock = validate_protocol_lock(protocol_lock, repository)
    if not run_id or int(workers) < 1:
        raise ValueError("run ID and positive worker count are required")
    destination = Path(output_dir).resolve()
    development = load_protocol(repository)
    firewall = DevelopmentSeedFirewall(development)
    plans, _ = planned_rows(repository)
    destination.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    reused_count = 0
    # Scene construction is deterministic but the seed firewall counters are
    # intentionally single-threaded. Workers are accepted and locked for the
    # later I/O/execution interface; Stage 0 preserves plan order here.
    for plan in plans:
        snapshot = build_canonical_snapshot(
            development=development, firewall=firewall, plan=plan
        )
        metadata, reused = write_snapshot_atomic(
            destination, snapshot, resume=resume
        )
        completed.append(metadata)
        reused_count += int(reused)
    if len(completed) != 210:
        raise Stage0ContractError("formal Stage-0 snapshot count changed")
    return {
        "schema_version": "backend_phase_a_v1_2_stage0_build_manifest_v1",
        "run_id": str(run_id),
        "protocol_sha256": V1_2_PROTOCOL_SHA256,
        "implementation_sha256": lock["implementation_sha256"],
        "snapshot_cache_root": str(destination),
        "workers": int(workers),
        "resume": bool(resume),
        "planned_snapshot_count": len(plans),
        "FORMAL_STAGE0_SNAPSHOT_BUILD_COUNT": len(completed),
        "new_snapshot_write_count": len(completed) - reused_count,
        "resumed_snapshot_count": reused_count,
        "FORMAL_BACKEND_EXECUTION_COUNT": 0,
        "FORMAL_TRIAL_RESULT_COUNT": 0,
        "STAGE0_BACKEND_IMPORT_COUNT": 0,
        "STAGE0_BACKEND_EXECUTION_COUNT": 0,
        "NATIVE_EXECUTION_COUNT": 0,
        **firewall.report(),
        "snapshot_ids": [str(row["snapshot_id"]) for row in completed],
    }


__all__ = [
    "ARRAY_FILENAMES",
    "CanonicalSnapshot",
    "CorruptSnapshotCache",
    "PROTOCOL_LOCK_SCHEMA",
    "SNAPSHOT_LOCK_SCHEMA",
    "STAGE0_METADATA_SCHEMA",
    "Stage0ContractError",
    "V1_2_CACHE_RELATIVE",
    "V1_2_DOCUMENT_RELATIVE",
    "V1_2_DOCUMENT_SHA256",
    "V1_2_LOCK_RELATIVE",
    "V1_2_PROTOCOL_LOCK_TAG",
    "V1_2_PROTOCOL_RELATIVE",
    "V1_2_PROTOCOL_SHA256",
    "V1_2_STAGE0_ARTIFACT_RELATIVE",
    "build_canonical_snapshot",
    "build_stage0_cache",
    "canonical_target",
    "eligible_parent_indices",
    "implementation_hashes",
    "load_v1_2_protocol",
    "planned_rows",
    "quantization_closure",
    "raw_array_sha256",
    "reference_pose_from_development",
    "scientific_contract_diff",
    "select_parent_indices",
    "snapshot_directory",
    "source_from_parent_indices",
    "validate_protocol_lock",
    "validate_snapshot_directory",
    "validate_snapshot_lock",
    "write_snapshot_atomic",
]
