"""Independent evidence verifier for Synthetic Confirmatory v2.

The parent-index reconstruction and float32 quantization closure in this file
are deliberately implemented from the stored arrays.  This module does not
import the production snapshot reader, its lineage helper, or the primary
H1--H6 analysis implementation.  The unchanged v1 independent H1--H6 core is
reused only after v2 evidence has been independently authenticated.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .synthetic_confirmatory_independent_verifier import (
    _compare,
    _confirmatory_trial,
    _independent_prepare_common,
    _independent_safe_common_record,
    independently_recompute_synthetic_confirmatory,
)
from .synthetic_confirmatory_v2_contract import (
    ARTIFACT_INVENTORY_VERSION,
    ARTIFACT_SCHEMA,
    BACKENDS,
    BOOTSTRAP_SEED,
    BOUND_FILE_PATHS,
    CONDITIONS,
    FORMAL_BRANCH,
    FORMAL_OUTPUT_DIR,
    FORMAL_PRERUN_TAG,
    FORMAL_RUN_ID,
    FORMAL_WORKERS,
    GATE_RELATIVE,
    GATE_SCHEMA,
    GEOMETRY_SEEDS,
    INDEPENDENT_SCHEMA,
    LINEAGE_SCHEMA,
    MANIFEST_RELATIVE,
    MANIFEST_SCHEMA,
    MEASUREMENT_SEEDS,
    METADATA_SCHEMA,
    NAMESPACE,
    PROTOCOL_RELATIVE,
    PROTOCOL_SCHEMA,
    RAW_RESULT_MANIFEST_SCHEMA,
    SCENES,
    SEED_SCHEDULE_RELATIVE,
    SNAPSHOT_CACHE_ROOT,
    SNAPSHOT_COUNT,
    SNAPSHOT_FIELDS,
    SNAPSHOT_LOCK_RELATIVE,
    SNAPSHOT_LOCK_SCHEMA,
    SNAPSHOT_SCHEMA,
    TRIAL_COUNT,
    TRIAL_FIELDS,
)


_SNAPSHOT_BUILDER_CONTRACT_VERSION = "synthetic_confirmatory_snapshot_builder_v2"
_PARENT_INDEX_FILENAME = "source_parent_target_indices.npy"
_BASE_ARRAY_FILENAMES = frozenset(
    {"source_points.npy", "target_points.npy", "reference_pose.npy"}
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
_METADATA_FIELDS = frozenset(
    {
        "array_file_sha256",
        "condition",
        "confirmatory_rng_instantiation_count",
        "development_protocol_sha256",
        "dropout_parameters",
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
        *_CLOSURE_FIELDS,
    }
)
_LOCK_FIELDS = frozenset(
    {
        "condition_snapshot_counts",
        "confirmatory_rng_instantiation_count",
        "lineage_schema_version",
        "planned_snapshot_count",
        "schema_version",
        "snapshot_builder_contract_version",
        "snapshot_lock_payload_sha256",
        "snapshot_schema_version",
        "snapshots",
    }
)
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
_CONDITION_METADATA: Mapping[str, Mapping[str, Any]] = {
    "IDEAL_MATCHED": {
        "independent_sampling": False,
        "dropout_parameters": {
            "map_dropout_fraction": 0.0,
            "scan_dropout_fraction": 0.0,
        },
        "noise_parameters": {
            "map_noise_sigma_m": 0.0,
            "scan_noise_sigma_m": 0.0,
        },
        "rng_count": 0,
    },
    "INDEPENDENT_NOISE_FREE": {
        "independent_sampling": True,
        "dropout_parameters": {
            "map_dropout_fraction": 0.0,
            "scan_dropout_fraction": 0.0,
        },
        "noise_parameters": {
            "map_noise_sigma_m": 0.0,
            "scan_noise_sigma_m": 0.0,
        },
        "rng_count": 0,
    },
    "FULL_NOISE": {
        "independent_sampling": True,
        "dropout_parameters": {
            "map_dropout_fraction": 0.0,
            "scan_dropout_fraction": 0.01,
        },
        "noise_parameters": {
            "map_noise_sigma_m": 0.001,
            "scan_noise_sigma_m": 0.003,
        },
        "rng_count": 3,
    },
}
_ALIAS_NAMES = (
    "scientific_protocol",
    "scientific_protocol_document",
    "gate_contract",
    "planned_snapshots",
    "planned_trials",
    "seed_schedule",
    "frozen_model",
)
_PROTOCOL_FIELDS = frozenset(
    {
        "CONFIRMATORY_RUN_AUTHORIZED",
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED",
        "REAL_DATA_RUN_AUTHORIZED",
        "SYNTHETIC_CONFIRMATORY_PROTOCOL_READY",
        "backend_count",
        "backends",
        "bootstrap_seed",
        "condition_count",
        "conditions",
        "confirmatory_seed_provenance_pass",
        "development_model_lock_sha256",
        "development_model_weighting",
        "gate_contract_payload_sha256",
        "geometry_seeds",
        "history",
        "lineage_schema_version",
        "measurement_seeds",
        "native_trial_count",
        "planned_snapshot_count",
        "planned_snapshot_identity_sha256",
        "planned_trial_count",
        "planned_trial_identity_sha256",
        "protocol_payload_sha256",
        "registration_execution_count",
        "scene_count",
        "scenes",
        "schema_version",
        "scientific_survival_audit_pass",
        "seed_namespace",
        "seed_schedule_payload_sha256",
        "snapshot_generation_count",
    }
)
_GATE_FIELDS = frozenset(
    {
        "all_hypotheses_required",
        "gate_contract_payload_sha256",
        "hypotheses",
        "hypothesis_count",
        "schema_version",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "bootstrap_seed",
        "derivation",
        "geometry_seeds",
        "measurement_seeds",
        "namespace",
        "records",
        "schema_version",
        "seed_schedule_payload_sha256",
    }
)


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid v2 independent JSON input: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"v2 independent JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raw_array_sha256(value: np.ndarray) -> str:
    if not value.flags.c_contiguous:
        raise ValueError("independent checksum input is not C-contiguous")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("v2 independent path escaped the standalone repository")
    return candidate


def _linear_summary(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.95, method="linear")),
        float(np.max(values)),
    )


def independently_recompute_v2_lineage(
    source_points: Any,
    target_points: Any,
    reference_pose: Any,
    parent_indices: Any,
) -> dict[str, Any]:
    """Reconstruct IDEAL lineage and Phase-A quantization closure from raw arrays.

    Invalid index inventories return a failing diagnostic without indexing the
    target.  Shape, dtype, finite, transform, duplicate, ordering, and closure
    checks are all independent of the production construction path.
    """

    source = np.asarray(source_points)
    target = np.asarray(target_points)
    reference = np.asarray(reference_pose)
    indices = np.asarray(parent_indices)
    array_contract_pass = bool(
        source.dtype == np.dtype("<f4")
        and target.dtype == np.dtype("<f4")
        and reference.dtype == np.dtype("<f8")
        and indices.dtype == np.dtype("<i8")
        and source.ndim == 2
        and source.shape[1:] == (3,)
        and target.ndim == 2
        and target.shape[1:] == (3,)
        and reference.shape == (4, 4)
        and indices.ndim == 1
        and all(value.flags.c_contiguous for value in (source, target, reference, indices))
        and all(np.all(np.isfinite(value)) for value in (source, target, reference))
        and source.shape[0] > 0
        and target.shape[0] > 0
        and indices.size > 0
    )
    if not array_contract_pass:
        raise ValueError("v2 independent lineage array contract mismatch")

    rotation = reference[:3, :3]
    homogeneous_row_pass = bool(
        np.allclose(reference[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12, rtol=0.0)
    )
    rotation_orthogonality_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3, dtype=np.float64)))
    )
    rotation_determinant = float(np.linalg.det(rotation))
    reference_transform_pass = bool(
        homogeneous_row_pass
        and rotation_orthogonality_error <= 1.0e-12
        and abs(rotation_determinant - 1.0) <= 1.0e-12
    )
    duplicate_count = int(indices.size - np.unique(indices).size)
    out_of_range_count = int(
        np.count_nonzero((indices < 0) | (indices >= target.shape[0]))
    )
    row_count_match = bool(source.shape[0] == indices.size)
    base = {
        "array_contract_pass": array_contract_pass,
        "homogeneous_row_pass": homogeneous_row_pass,
        "parent_index_count": int(indices.size),
        "parent_index_duplicate_count": duplicate_count,
        "parent_index_out_of_range_count": out_of_range_count,
        "parent_index_unique_count": int(np.unique(indices).size),
        "reference_rotation_determinant": rotation_determinant,
        "reference_rotation_orthogonality_error": rotation_orthogonality_error,
        "reference_transform_pass": reference_transform_pass,
        "source_parent_row_count_match": row_count_match,
    }
    if duplicate_count or out_of_range_count or not row_count_match or not reference_transform_pass:
        return {
            **base,
            "actual_error_bound_violation_count": None,
            "closure_residual_max_m": None,
            "closure_residual_violation_count": None,
            "float64_guard_max_m": None,
            "lineage_closure_violation_count": None,
            "max_normalized_closure_ratio": None,
            "parent_points_map_f64_sha256": None,
            "predicted_quantization_max_m": None,
            "predicted_quantization_median_m": None,
            "predicted_quantization_q95_m": None,
            "quantization_closure_pass": False,
            "reconstruction_error_max_m": None,
            "reconstruction_error_median_m": None,
            "reconstruction_error_q95_m": None,
            "row_correspondence_pass": False,
        }

    parents = np.ascontiguousarray(target[indices].astype(np.float64), dtype="<f8")
    translation = reference[:3, 3]
    source_float64 = np.ascontiguousarray(
        (rotation.T @ (parents - translation).T).T,
        dtype="<f8",
    )
    expected_source = np.ascontiguousarray(source_float64, dtype="<f4")
    source_quantized = source.astype(np.float64)
    reconstructed = (rotation @ source_quantized.T).T + translation
    actual_error = reconstructed - parents
    source_quantization = source_quantized - source_float64
    predicted_error = (rotation @ source_quantization.T).T
    closure_residual = actual_error - predicted_error
    actual_norm = np.linalg.norm(actual_error, axis=1)
    predicted_norm = np.linalg.norm(predicted_error, axis=1)
    residual_norm = np.linalg.norm(closure_residual, axis=1)
    scale = np.maximum.reduce(
        (
            np.ones(indices.size, dtype=np.float64),
            np.linalg.norm(parents, axis=1),
            np.linalg.norm(source_float64, axis=1),
            np.full(indices.size, np.linalg.norm(translation), dtype=np.float64),
        )
    )
    guard = 256.0 * np.finfo(np.float64).eps * scale
    normalized = residual_norm / guard
    closure_violations = int(np.count_nonzero(residual_norm > guard))
    bound_violations = int(np.count_nonzero(actual_norm > predicted_norm + guard))
    reconstruction = _linear_summary(actual_norm)
    predicted = _linear_summary(predicted_norm)
    row_pass = bool(np.array_equal(source, expected_source))
    quantization_pass = bool(
        row_pass and closure_violations == 0 and bound_violations == 0
    )
    return {
        **base,
        "actual_error_bound_violation_count": bound_violations,
        "closure_residual_max_m": float(np.max(residual_norm)),
        "closure_residual_violation_count": closure_violations,
        "float64_guard_max_m": float(np.max(guard)),
        "lineage_closure_violation_count": closure_violations + bound_violations,
        "max_normalized_closure_ratio": float(np.max(normalized)),
        "parent_points_map_f64_sha256": _raw_array_sha256(parents),
        "predicted_quantization_max_m": predicted[2],
        "predicted_quantization_median_m": predicted[0],
        "predicted_quantization_q95_m": predicted[1],
        "quantization_closure_pass": quantization_pass,
        "reconstruction_error_max_m": reconstruction[2],
        "reconstruction_error_median_m": reconstruction[0],
        "reconstruction_error_q95_m": reconstruction[1],
        "row_correspondence_pass": row_pass,
    }


def _canonical_integer(value: str, label: str) -> int:
    if not value or value.strip() != value:
        raise ValueError(f"{label} is not canonical")
    result = int(value)
    if str(result) != value:
        raise ValueError(f"{label} is not canonical")
    return result


def _read_plan_csv(path: Path, fields: Sequence[str]) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise ValueError("v2 independent plan CSV schema changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"invalid v2 independent plan CSV: {path}") from error
    output: list[dict[str, Any]] = []
    for raw in rows:
        if None in raw:
            raise ValueError("v2 independent plan CSV has excess columns")
        row: dict[str, Any] = dict(raw)
        for name in ("geometry_seed", "repeat_index"):
            row[name] = _canonical_integer(raw[name], name)
        row["measurement_seed"] = (
            None
            if raw["measurement_seed"] == ""
            else _canonical_integer(raw["measurement_seed"], "measurement_seed")
        )
        if "planned_backend_count" in row:
            row["planned_backend_count"] = _canonical_integer(
                raw["planned_backend_count"], "planned_backend_count"
            )
        output.append(row)
    return output


def _validate_snapshot_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    if not set(SNAPSHOT_FIELDS).issubset(row):
        raise ValueError("v2 independent snapshot plan row is incomplete")
    if (
        type(row.get("geometry_seed")) is not int
        or type(row.get("repeat_index")) is not int
        or type(row.get("planned_backend_count")) is not int
        or (
            row.get("measurement_seed") not in (None, "")
            and type(row.get("measurement_seed")) is not int
        )
    ):
        raise ValueError("v2 independent snapshot plan scalar type mismatch")
    value = {
        "planned_snapshot_id": str(row["planned_snapshot_id"]),
        "scene_variant": str(row["scene_variant"]),
        "condition": str(row["condition"]),
        "geometry_seed": int(row["geometry_seed"]),
        "measurement_seed": (
            None if row.get("measurement_seed") in (None, "")
            else int(row["measurement_seed"])
        ),
        "repeat_index": int(row["repeat_index"]),
        "planned_backend_count": int(row["planned_backend_count"]),
        "replicate_semantics": str(row["replicate_semantics"]),
    }
    identity = {
        name: value[name]
        for name in (
            "condition",
            "geometry_seed",
            "measurement_seed",
            "repeat_index",
            "scene_variant",
        )
    }
    expected_id = f"synthetic-confirmatory-v2::{_identity_sha256(identity)}"
    condition = value["condition"]
    if condition == "IDEAL_MATCHED":
        semantics = "ONE_CONTROL_INPUT"
        replicate_pass = value["measurement_seed"] is None and value["repeat_index"] == 0
    elif condition == "INDEPENDENT_NOISE_FREE":
        semantics = "ONE_DETERMINISTIC_INDEPENDENT_INPUT"
        replicate_pass = value["measurement_seed"] is None and value["repeat_index"] == 0
    elif condition == "FULL_NOISE":
        semantics = "FIFTEEN_STOCHASTIC_INPUTS_PER_SCENE_GEOMETRY"
        replicate_pass = (
            value["measurement_seed"] in MEASUREMENT_SEEDS
            and value["repeat_index"] in range(5)
        )
    else:
        semantics = ""
        replicate_pass = False
    if (
        value["scene_variant"] not in SCENES
        or value["geometry_seed"] not in GEOMETRY_SEEDS
        or value["planned_backend_count"] != 2
        or value["planned_snapshot_id"] != expected_id
        or not replicate_pass
        or value["replicate_semantics"] != semantics
    ):
        raise ValueError("v2 independent snapshot identity is outside the frozen plan")
    return value


def _validate_plans(
    snapshots: Sequence[Mapping[str, Any]], trials: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checked_snapshots = [_validate_snapshot_plan(row) for row in snapshots]
    if (
        len(checked_snapshots) != SNAPSHOT_COUNT
        or len({row["planned_snapshot_id"] for row in checked_snapshots}) != SNAPSHOT_COUNT
        or Counter(row["condition"] for row in checked_snapshots)
        != Counter({"IDEAL_MATCHED": 35, "INDEPENDENT_NOISE_FREE": 35, "FULL_NOISE": 525})
    ):
        raise ValueError("v2 independent snapshot plan inventory mismatch")
    by_snapshot = {row["planned_snapshot_id"]: row for row in checked_snapshots}
    checked_trials: list[dict[str, Any]] = []
    for raw in trials:
        if not set(TRIAL_FIELDS).issubset(raw):
            raise ValueError("v2 independent trial plan row is incomplete")
        if (
            type(raw.get("geometry_seed")) is not int
            or type(raw.get("repeat_index")) is not int
            or (
                raw.get("measurement_seed") not in (None, "")
                and type(raw.get("measurement_seed")) is not int
            )
        ):
            raise ValueError("v2 independent trial plan scalar type mismatch")
        row = {
            "planned_trial_id": str(raw["planned_trial_id"]),
            "planned_snapshot_id": str(raw["planned_snapshot_id"]),
            "scene_variant": str(raw["scene_variant"]),
            "condition": str(raw["condition"]),
            "geometry_seed": int(raw["geometry_seed"]),
            "measurement_seed": (
                None if raw.get("measurement_seed") in (None, "")
                else int(raw["measurement_seed"])
            ),
            "repeat_index": int(raw["repeat_index"]),
            "backend": str(raw["backend"]),
        }
        parent = by_snapshot.get(row["planned_snapshot_id"])
        if (
            parent is None
            or row["backend"] not in BACKENDS
            or row["planned_trial_id"]
            != f"{row['planned_snapshot_id']}::{row['backend']}"
            or any(
                row[name] != parent[name]
                for name in (
                    "scene_variant",
                    "condition",
                    "geometry_seed",
                    "measurement_seed",
                    "repeat_index",
                )
            )
        ):
            raise ValueError("v2 independent trial/parent plan mismatch")
        checked_trials.append(row)
    groups = Counter(row["planned_snapshot_id"] for row in checked_trials)
    if (
        len(checked_trials) != TRIAL_COUNT
        or len({row["planned_trial_id"] for row in checked_trials}) != TRIAL_COUNT
        or set(groups) != set(by_snapshot)
        or any(count != 2 for count in groups.values())
        or Counter(row["backend"] for row in checked_trials)
        != Counter({BACKENDS[0]: SNAPSHOT_COUNT, BACKENDS[1]: SNAPSHOT_COUNT})
    ):
        raise ValueError("v2 independent trial plan inventory/pairing mismatch")
    return checked_snapshots, checked_trials


def independently_authenticate_v2_manifest(
    manifest_path: str | Path, *, require_authorized: bool = True
) -> tuple[Path, dict[str, Any]]:
    """Authenticate the exact v2 manifest without calling its implementation."""

    path = Path(manifest_path).resolve()
    repository = path.parent.parent.resolve()
    if path != (repository / MANIFEST_RELATIVE).resolve() or not path.is_file():
        raise ValueError("v2 independent verifier requires the exact formal manifest")
    manifest = _strict_object(path)
    expected_keys = {
        "artifact_inventory_version",
        "artifact_schema",
        "backend_count",
        "bootstrap_seed",
        "bound_files",
        "formal_branch",
        "formal_execution_authorized",
        "formal_output_dir",
        "formal_pre_run_tag",
        "formal_run_id",
        "formal_workers",
        "geometry_seeds",
        "manifest_payload_sha256",
        "manifest_schema",
        "manifest_version",
        "measurement_seeds",
        "native_trial_count",
        "open3d_parameter_sha256",
        "open3d_version",
        "pcl_parameter_sha256",
        "pcl_version",
        "planned_snapshot_count",
        "planned_trial_count",
        "raw_result_manifest_schema",
        "schema_version",
        "seed_namespace",
        "seed_schedule_payload_sha256",
        "snapshot_cache_root",
        "snapshot_lock_path",
        *(f"{name}_path" for name in _ALIAS_NAMES),
        *(f"{name}_sha256" for name in _ALIAS_NAMES),
    }
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    identity_pass = bool(
        set(manifest) == expected_keys
        and manifest.get("manifest_payload_sha256") == _canonical_sha256(unsigned)
        and all(
            type(manifest.get(name)) is int
            for name in (
                "backend_count",
                "bootstrap_seed",
                "formal_workers",
                "native_trial_count",
                "planned_snapshot_count",
                "planned_trial_count",
            )
        )
        and manifest.get("artifact_inventory_version") == ARTIFACT_INVENTORY_VERSION
        and manifest.get("artifact_schema") == ARTIFACT_SCHEMA
        and manifest.get("backend_count") == 2
        and manifest.get("bootstrap_seed") == BOOTSTRAP_SEED
        and manifest.get("formal_branch") == FORMAL_BRANCH
        and type(manifest.get("formal_execution_authorized")) is bool
        and manifest.get("formal_output_dir") == FORMAL_OUTPUT_DIR
        and manifest.get("formal_pre_run_tag") == FORMAL_PRERUN_TAG
        and manifest.get("formal_run_id") == FORMAL_RUN_ID
        and manifest.get("formal_workers") == FORMAL_WORKERS
        and manifest.get("geometry_seeds") == list(GEOMETRY_SEEDS)
        and manifest.get("manifest_schema") == MANIFEST_SCHEMA
        and manifest.get("schema_version") == MANIFEST_SCHEMA
        and manifest.get("manifest_version") == "2"
        and manifest.get("measurement_seeds") == list(MEASUREMENT_SEEDS)
        and manifest.get("native_trial_count") == 0
        and manifest.get("planned_snapshot_count") == SNAPSHOT_COUNT
        and manifest.get("planned_trial_count") == TRIAL_COUNT
        and manifest.get("raw_result_manifest_schema") == RAW_RESULT_MANIFEST_SCHEMA
        and manifest.get("seed_namespace") == NAMESPACE
        and manifest.get("snapshot_cache_root") == SNAPSHOT_CACHE_ROOT
        and manifest.get("snapshot_lock_path") == SNAPSHOT_LOCK_RELATIVE.as_posix()
    )
    if not identity_pass:
        raise ValueError("v2 independent formal manifest identity mismatch")
    if require_authorized and manifest["formal_execution_authorized"] is not True:
        raise PermissionError("v2 formal manifest is not authorized")
    bound = manifest.get("bound_files")
    if type(bound) is not dict or set(bound) != set(BOUND_FILE_PATHS):
        raise ValueError("v2 independent manifest binding inventory changed")
    for name, relative in BOUND_FILE_PATHS.items():
        row = bound.get(name)
        candidate = _inside(repository, relative)
        if (
            type(row) is not dict
            or set(row) != {"path", "sha256"}
            or row.get("path") != relative
            or not _is_sha256(row.get("sha256"))
            or not candidate.is_file()
            or _sha256(candidate) != row["sha256"]
        ):
            raise ValueError(f"v2 independent manifest binding mismatch: {name}")
    for name in _ALIAS_NAMES:
        if (
            manifest.get(f"{name}_path") != bound[name]["path"]
            or manifest.get(f"{name}_sha256") != bound[name]["sha256"]
        ):
            raise ValueError("v2 independent manifest flat alias mismatch")

    protocol = _strict_object(repository / PROTOCOL_RELATIVE)
    gate = _strict_object(repository / GATE_RELATIVE)
    schedule = _strict_object(repository / SEED_SCHEDULE_RELATIVE)
    for value, signature_name, schema, label in (
        (protocol, "protocol_payload_sha256", PROTOCOL_SCHEMA, "protocol"),
        (gate, "gate_contract_payload_sha256", GATE_SCHEMA, "gate"),
        (schedule, "seed_schedule_payload_sha256", "synthetic_confirmatory_v2_seed_schedule_v1", "seed schedule"),
    ):
        unsigned_value = {key: item for key, item in value.items() if key != signature_name}
        if (
            set(value)
            != {
                "protocol": _PROTOCOL_FIELDS,
                "gate": _GATE_FIELDS,
                "seed schedule": _SCHEDULE_FIELDS,
            }[label]
            or value.get("schema_version") != schema
            or value.get(signature_name) != _identity_sha256(unsigned_value)
        ):
            raise ValueError(f"v2 independent {label} schema/SHA mismatch")
    if (
        any(
            type(protocol.get(name)) is not int
            for name in (
                "backend_count",
                "bootstrap_seed",
                "condition_count",
                "native_trial_count",
                "planned_snapshot_count",
                "planned_trial_count",
                "registration_execution_count",
                "scene_count",
                "snapshot_generation_count",
            )
        )
        or protocol.get("SYNTHETIC_CONFIRMATORY_PROTOCOL_READY") is not True
        or protocol.get("CONFIRMATORY_RUN_AUTHORIZED") is not False
        or protocol.get("REAL_DATA_RUN_AUTHORIZED") is not False
        or protocol.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is not False
        or protocol.get("confirmatory_seed_provenance_pass") is not True
        or protocol.get("scientific_survival_audit_pass") is not True
        or protocol.get("registration_execution_count") != 0
        or protocol.get("snapshot_generation_count") != 0
        or protocol.get("backend_count") != 2
        or protocol.get("backends") != list(BACKENDS)
        or protocol.get("condition_count") != 3
        or protocol.get("conditions") != list(CONDITIONS)
        or protocol.get("scene_count") != 7
        or protocol.get("scenes") != list(SCENES)
        or protocol.get("native_trial_count") != 0
        or protocol.get("development_model_weighting")
        != "UNIQUE_INPUT_CONDITION_BALANCED"
        or protocol.get("planned_snapshot_count") != SNAPSHOT_COUNT
        or protocol.get("planned_trial_count") != TRIAL_COUNT
        or protocol.get("geometry_seeds") != list(GEOMETRY_SEEDS)
        or protocol.get("measurement_seeds") != list(MEASUREMENT_SEEDS)
        or protocol.get("bootstrap_seed") != BOOTSTRAP_SEED
        or protocol.get("seed_namespace") != NAMESPACE
        or protocol.get("lineage_schema_version") != LINEAGE_SCHEMA
        or protocol.get("gate_contract_payload_sha256")
        != gate.get("gate_contract_payload_sha256")
        or protocol.get("development_model_lock_sha256")
        != manifest.get("frozen_model_sha256")
        or protocol.get("seed_schedule_payload_sha256")
        != schedule.get("seed_schedule_payload_sha256")
        or manifest.get("seed_schedule_payload_sha256")
        != schedule.get("seed_schedule_payload_sha256")
        or gate.get("hypothesis_count") != 6
        or gate.get("all_hypotheses_required") is not True
        or type(gate.get("hypotheses")) is not dict
    ):
        raise ValueError("v2 independent frozen protocol semantics changed")
    expected_schedule_records = [
        {
            "domain": domain,
            "index": index,
            "payload_sha256": hashlib.sha256(
                f"{NAMESPACE}|{domain}|{index}".encode("utf-8")
            ).hexdigest(),
            "seed": seed,
        }
        for domain, values in (
            ("geometry", GEOMETRY_SEEDS),
            ("measurement", MEASUREMENT_SEEDS),
            ("bootstrap", (BOOTSTRAP_SEED,)),
        )
        for index, seed in enumerate(values)
    ]
    if (
        schedule.get("namespace") != NAMESPACE
        or schedule.get("geometry_seeds") != list(GEOMETRY_SEEDS)
        or schedule.get("measurement_seeds") != list(MEASUREMENT_SEEDS)
        or schedule.get("bootstrap_seed") != BOOTSTRAP_SEED
        or schedule.get("records") != expected_schedule_records
        or schedule.get("derivation")
        != {
            "digest": "SHA256(UTF-8(namespace|domain|index))",
            "integer": "int.from_bytes(digest[0:8], byteorder=big, signed=false) % 2147483647",
            "zero_remap": "0 -> 1",
        }
    ):
        raise ValueError("v2 independent seed schedule semantics changed")
    parameters = _strict_object(_inside(repository, BOUND_FILE_PATHS["backend_parameter_contract"]))
    if (
        type(parameters.get("open3d")) is not dict
        or type(parameters.get("pcl")) is not dict
        or type(parameters["open3d"].get("parameters")) is not dict
        or type(parameters["pcl"].get("parameters")) is not dict
    ):
        raise ValueError("v2 independent backend parameter schema mismatch")
    if (
        manifest.get("open3d_parameter_sha256")
        != parameters.get("open3d", {}).get("canonical_sha256")
        or manifest.get("open3d_version")
        != parameters.get("open3d", {}).get("parameters", {}).get("version")
        or manifest.get("pcl_parameter_sha256")
        != parameters.get("pcl", {}).get("canonical_sha256")
        or manifest.get("pcl_version")
        != parameters.get("pcl", {}).get("parameters", {}).get("version")
    ):
        raise ValueError("v2 independent backend parameter binding mismatch")
    return repository, manifest


def independently_read_v2_snapshot(
    cache_root: str | Path,
    plan: Mapping[str, Any],
    *,
    expected_lock_entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly authenticate one v2 snapshot and its conditional sidecar."""

    row = _validate_snapshot_plan(plan)
    root = Path(cache_root).resolve()
    directory = (root / row["planned_snapshot_id"]).resolve()
    expected_files = set(_BASE_ARRAY_FILENAMES) | {"metadata.json"}
    if row["condition"] == "IDEAL_MATCHED":
        expected_files.add(_PARENT_INDEX_FILENAME)
    if (
        root not in directory.parents
        or not directory.is_dir()
        or {candidate.name for candidate in directory.iterdir()} != expected_files
        or not all((directory / name).is_file() for name in expected_files)
    ):
        raise ValueError("v2 independent snapshot conditional file inventory mismatch")
    metadata = _strict_object(directory / "metadata.json")
    if set(metadata) != _METADATA_FIELDS:
        raise ValueError("v2 independent snapshot metadata fields changed")
    unsigned = dict(metadata)
    metadata_sha = unsigned.pop("metadata_payload_sha256")
    condition_contract = _CONDITION_METADATA[row["condition"]]
    exact = {
        "condition": row["condition"],
        "geometry_seed": row["geometry_seed"],
        "measurement_seed": row["measurement_seed"],
        "planned_snapshot_id": row["planned_snapshot_id"],
        "repeat_index": row["repeat_index"],
        "scene_variant": row["scene_variant"],
        "snapshot_id": row["planned_snapshot_id"],
    }
    if (
        not _is_sha256(metadata_sha)
        or metadata_sha != _canonical_sha256(unsigned)
        or any(metadata.get(name) != expected for name, expected in exact.items())
        or metadata.get("schema_version") != METADATA_SCHEMA
        or metadata.get("snapshot_schema_version") != SNAPSHOT_SCHEMA
        or metadata.get("lineage_schema_version") != LINEAGE_SCHEMA
        or metadata.get("snapshot_builder_contract_version")
        != _SNAPSHOT_BUILDER_CONTRACT_VERSION
        or metadata.get("initial_pose") != "reference_pose_exact"
        or metadata.get("confirmatory_rng_instantiation_count")
        != condition_contract["rng_count"]
        or metadata.get("independent_sampling")
        is not condition_contract["independent_sampling"]
        or metadata.get("dropout_parameters")
        != condition_contract["dropout_parameters"]
        or metadata.get("noise_parameters") != condition_contract["noise_parameters"]
        or not _is_sha256(metadata.get("development_protocol_sha256"))
        or not _is_sha256(metadata.get("generator_sha256"))
        or (
            metadata.get("measurement_seed") is not None
            and type(metadata.get("measurement_seed")) is not int
        )
        or type(metadata.get("source_point_count")) is not int
        or type(metadata.get("target_point_count")) is not int
        or metadata["source_point_count"] <= 0
        or metadata["target_point_count"] <= 0
        or any(
            type(metadata.get(name)) is not bool
            for name in (
                "independent_sampling",
                "quantization_closure_pass",
                "source_has_target_parent_lineage",
                "source_is_target_subset",
                "source_parent_row_count_match",
            )
        )
        or any(
            type(metadata.get(name)) is not int
            for name in (
                "confirmatory_rng_instantiation_count",
                "geometry_seed",
                "lineage_closure_violation_count",
                "parent_index_count",
                "parent_index_duplicate_count",
                "parent_index_out_of_range_count",
                "parent_index_unique_count",
                "repeat_index",
            )
        )
    ):
        raise ValueError("v2 independent snapshot metadata identity mismatch")

    file_sha = {name: _sha256(directory / name) for name in sorted(expected_files)}
    array_file_sha = {name: digest for name, digest in file_sha.items() if name.endswith(".npy")}
    if metadata.get("array_file_sha256") != array_file_sha:
        raise ValueError("v2 independent snapshot array file SHA mismatch")
    if expected_lock_entry is not None and (
        type(expected_lock_entry) is not dict
        or set(expected_lock_entry) != _LOCK_ENTRY_FIELDS
        or expected_lock_entry.get("snapshot_id") != row["planned_snapshot_id"]
        or expected_lock_entry.get("file_sha256") != file_sha
    ):
        raise ValueError("v2 independent snapshot lock file binding mismatch")
    try:
        source = np.load(directory / "source_points.npy", allow_pickle=False)
        target = np.load(directory / "target_points.npy", allow_pickle=False)
        reference = np.load(directory / "reference_pose.npy", allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError("v2 independent snapshot array load failed") from error
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
        or metadata.get("source_point_count") != len(source)
        or metadata.get("target_point_count") != len(target)
    ):
        raise ValueError("v2 independent snapshot array contract mismatch")
    rotation = reference[:3, :3]
    if (
        not np.allclose(
            reference[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12, rtol=0.0
        )
        or float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))) > 1.0e-12
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-12
    ):
        raise ValueError("v2 independent snapshot reference transform is invalid")
    raw = {
        "source_checksum": _raw_array_sha256(source),
        "target_checksum": _raw_array_sha256(target),
        "reference_pose_checksum": _raw_array_sha256(reference),
    }
    parent_indices: np.ndarray | None = None
    parent_sha: str | None = None
    lineage: dict[str, Any] | None = None
    if row["condition"] == "IDEAL_MATCHED":
        try:
            parent_indices = np.load(directory / _PARENT_INDEX_FILENAME, allow_pickle=False)
        except (OSError, ValueError, TypeError) as error:
            raise ValueError("v2 independent parent-index load failed") from error
        lineage = independently_recompute_v2_lineage(
            source, target, reference, parent_indices
        )
        parent_sha = _raw_array_sha256(parent_indices)
        expected_lineage = {
            "actual_error_bound_violation_count": lineage["actual_error_bound_violation_count"],
            "closure_residual_max_m": lineage["closure_residual_max_m"],
            "closure_residual_violation_count": lineage["closure_residual_violation_count"],
            "float64_guard_max_m": lineage["float64_guard_max_m"],
            "lineage_closure_violation_count": lineage["lineage_closure_violation_count"],
            "lineage_validation_method": "PARENT_INDEX_ROW_CORRESPONDENCE_PLUS_PHASE_A_QUANTIZATION_CLOSURE",
            "max_normalized_closure_ratio": lineage["max_normalized_closure_ratio"],
            "parent_index_count": lineage["parent_index_count"],
            "parent_index_duplicate_count": lineage["parent_index_duplicate_count"],
            "parent_index_out_of_range_count": lineage["parent_index_out_of_range_count"],
            "parent_index_unique_count": lineage["parent_index_unique_count"],
            "parent_points_map_f64_sha256": lineage["parent_points_map_f64_sha256"],
            "predicted_quantization_max_m": lineage["predicted_quantization_max_m"],
            "predicted_quantization_median_m": lineage["predicted_quantization_median_m"],
            "predicted_quantization_q95_m": lineage["predicted_quantization_q95_m"],
            "quantization_closure_pass": True,
            "reconstruction_error_max_m": lineage["reconstruction_error_max_m"],
            "reconstruction_error_median_m": lineage["reconstruction_error_median_m"],
            "reconstruction_error_q95_m": lineage["reconstruction_error_q95_m"],
            "source_has_target_parent_lineage": True,
            "source_is_target_subset": True,
            "source_parent_row_count_match": True,
            "source_parent_target_indices_path": _PARENT_INDEX_FILENAME,
            "source_parent_target_indices_sha256": parent_sha,
        }
        if (
            any(metadata.get(name) != expected for name, expected in expected_lineage.items())
            or any(
                type(metadata.get(name)) is not int
                for name in (
                    "actual_error_bound_violation_count",
                    "closure_residual_violation_count",
                    "lineage_closure_violation_count",
                    "parent_index_count",
                    "parent_index_duplicate_count",
                    "parent_index_out_of_range_count",
                    "parent_index_unique_count",
                )
            )
            or any(
                type(metadata.get(name)) is not float
                for name in (
                    "closure_residual_max_m",
                    "float64_guard_max_m",
                    "max_normalized_closure_ratio",
                    "predicted_quantization_max_m",
                    "predicted_quantization_median_m",
                    "predicted_quantization_q95_m",
                    "reconstruction_error_max_m",
                    "reconstruction_error_median_m",
                    "reconstruction_error_q95_m",
                )
            )
            or lineage["row_correspondence_pass"] is not True
            or lineage["reference_transform_pass"] is not True
            or lineage["parent_index_duplicate_count"] != 0
            or lineage["parent_index_out_of_range_count"] != 0
            or lineage["lineage_closure_violation_count"] != 0
            or lineage["quantization_closure_pass"] is not True
        ):
            raise ValueError("v2 independent IDEAL parent-lineage validation failed")
    else:
        expected_nonideal = {
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
        if any(metadata.get(name) != expected for name, expected in expected_nonideal.items()):
            raise ValueError("v2 independent non-IDEAL lineage sidecar contract mismatch")

    snapshot_checksum = _canonical_sha256(
        {
            "snapshot_id": row["planned_snapshot_id"],
            **raw,
            "source_parent_target_indices_sha256": parent_sha,
        }
    )
    if (
        any(metadata.get(name) != digest for name, digest in raw.items())
        or metadata.get("snapshot_checksum") != snapshot_checksum
    ):
        raise ValueError("v2 independent snapshot raw/aggregate checksum mismatch")
    if expected_lock_entry is not None:
        expected_lock_values = {
            **raw,
            "metadata_payload_sha256": metadata_sha,
            "snapshot_checksum": snapshot_checksum,
            "source_parent_target_indices_sha256": parent_sha,
        }
        if any(
            expected_lock_entry.get(name) != expected
            for name, expected in expected_lock_values.items()
        ):
            raise ValueError("v2 independent snapshot lock payload mismatch")
    return {
        "directory": directory,
        "file_sha256": file_sha,
        "lineage": lineage,
        "metadata": metadata,
        "parent_indices": parent_indices,
        "reference": reference,
        "snapshot_checksum": snapshot_checksum,
        "source": source,
        "target": target,
        **raw,
    }


def independently_validate_v2_snapshot_lock(
    lock_path: str | Path,
    plans: Sequence[Mapping[str, Any]],
    *,
    cache_root: str | Path | None = None,
    require_formal_inventory: bool | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Authenticate a v2 lock and optionally every bound snapshot."""

    path = Path(lock_path).resolve()
    value = _strict_object(path)
    if set(value) != _LOCK_FIELDS:
        raise ValueError("v2 independent snapshot lock fields changed")
    checked_plans = [_validate_snapshot_plan(row) for row in plans]
    formal = len(checked_plans) == SNAPSHOT_COUNT if require_formal_inventory is None else require_formal_inventory
    if formal and len(checked_plans) != SNAPSHOT_COUNT:
        raise ValueError("v2 independent formal snapshot lock plan count mismatch")
    expected_ids = [row["planned_snapshot_id"] for row in checked_plans]
    expected_counts = dict(sorted(Counter(row["condition"] for row in checked_plans).items()))
    expected_rng = sum(int(_CONDITION_METADATA[row["condition"]]["rng_count"]) for row in checked_plans)
    unsigned = {key: item for key, item in value.items() if key != "snapshot_lock_payload_sha256"}
    entries = value.get("snapshots")
    if (
        value.get("snapshot_lock_payload_sha256") != _canonical_sha256(unsigned)
        or value.get("schema_version") != SNAPSHOT_LOCK_SCHEMA
        or value.get("snapshot_schema_version") != SNAPSHOT_SCHEMA
        or value.get("lineage_schema_version") != LINEAGE_SCHEMA
        or value.get("snapshot_builder_contract_version")
        != _SNAPSHOT_BUILDER_CONTRACT_VERSION
        or type(value.get("planned_snapshot_count")) is not int
        or type(value.get("confirmatory_rng_instantiation_count")) is not int
        or value.get("planned_snapshot_count") != len(checked_plans)
        or value.get("condition_snapshot_counts") != expected_counts
        or value.get("confirmatory_rng_instantiation_count") != expected_rng
        or type(entries) is not list
        or len(entries) != len(checked_plans)
        or [entry.get("snapshot_id") if type(entry) is dict else None for entry in entries]
        != expected_ids
        or len(set(expected_ids)) != len(expected_ids)
    ):
        raise ValueError("v2 independent snapshot lock identity/inventory mismatch")
    if formal and (
        expected_counts
        != {"FULL_NOISE": 525, "IDEAL_MATCHED": 35, "INDEPENDENT_NOISE_FREE": 35}
        or expected_rng != 1575
    ):
        raise ValueError("v2 independent formal snapshot lock cardinality changed")
    by_id: dict[str, dict[str, Any]] = {}
    for plan, entry in zip(checked_plans, entries):
        if type(entry) is not dict or set(entry) != _LOCK_ENTRY_FIELDS:
            raise ValueError("v2 independent snapshot lock entry fields changed")
        expected_files = set(_BASE_ARRAY_FILENAMES) | {"metadata.json"}
        if plan["condition"] == "IDEAL_MATCHED":
            expected_files.add(_PARENT_INDEX_FILENAME)
        exact = {
            "condition": plan["condition"],
            "confirmatory_rng_instantiation_count": _CONDITION_METADATA[plan["condition"]]["rng_count"],
            "geometry_seed": plan["geometry_seed"],
            "measurement_seed": plan["measurement_seed"],
            "repeat_index": plan["repeat_index"],
            "scene_variant": plan["scene_variant"],
            "snapshot_id": plan["planned_snapshot_id"],
        }
        if (
            any(entry.get(name) != expected for name, expected in exact.items())
            or type(entry.get("confirmatory_rng_instantiation_count")) is not int
            or type(entry.get("geometry_seed")) is not int
            or type(entry.get("repeat_index")) is not int
            or (
                entry.get("measurement_seed") is not None
                and type(entry.get("measurement_seed")) is not int
            )
            or type(entry.get("file_sha256")) is not dict
            or set(entry["file_sha256"]) != expected_files
            or not all(_is_sha256(item) for item in entry["file_sha256"].values())
            or not all(
                _is_sha256(entry.get(name))
                for name in (
                    "metadata_payload_sha256",
                    "reference_pose_checksum",
                    "snapshot_checksum",
                    "source_checksum",
                    "target_checksum",
                )
            )
            or (
                plan["condition"] == "IDEAL_MATCHED"
                and not _is_sha256(entry.get("source_parent_target_indices_sha256"))
            )
            or (
                plan["condition"] != "IDEAL_MATCHED"
                and entry.get("source_parent_target_indices_sha256") is not None
            )
        ):
            raise ValueError("v2 independent snapshot lock entry mismatch")
        by_id[plan["planned_snapshot_id"]] = dict(entry)
    if cache_root is not None:
        for plan in checked_plans:
            independently_read_v2_snapshot(
                cache_root,
                plan,
                expected_lock_entry=by_id[plan["planned_snapshot_id"]],
            )
    return value, by_id


def independently_validate_v2_lineage_inventory(
    *,
    cache_root: str | Path,
    plans: Sequence[Mapping[str, Any]],
    lock_path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Read every snapshot and enforce the 35/560 conditional lineage split."""

    lock, by_lock = independently_validate_v2_snapshot_lock(lock_path, plans)
    del lock
    ideal = nonideal = violations = 0
    snapshots: dict[str, dict[str, Any]] = {}
    for raw_plan in plans:
        plan = _validate_snapshot_plan(raw_plan)
        item = independently_read_v2_snapshot(
            cache_root,
            plan,
            expected_lock_entry=by_lock[plan["planned_snapshot_id"]],
        )
        if plan["condition"] == "IDEAL_MATCHED":
            ideal += 1
            violations += int(item["lineage"] is None)
        else:
            nonideal += 1
            violations += int(item["parent_indices"] is not None or item["lineage"] is not None)
        snapshots[plan["planned_snapshot_id"]] = item
    report = {
        "IDEAL_PARENT_LINEAGE_COUNT": ideal,
        "LINEAGE_INTEGRITY_PASS": bool(
            ideal == 35 and nonideal == 560 and violations == 0
        ),
        "LINEAGE_VIOLATION_COUNT": violations,
        "NONIDEAL_NO_LINEAGE_COUNT": nonideal,
        "schema_version": "synthetic_confirmatory_v2_independent_lineage_inventory_v1",
    }
    if report["LINEAGE_INTEGRITY_PASS"] is not True:
        raise ValueError("v2 independent lineage inventory failed")
    return report, snapshots


def independently_read_v2_evidence(
    *, manifest_path: str | Path, run_dir: str | Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Authenticate v2 manifest, plans, raw trials, lock, arrays, and lineage."""

    from .rotation_metrics import rotation_metric_audit

    repository, manifest = independently_authenticate_v2_manifest(manifest_path)
    snapshot_plan_path = _inside(repository, manifest["planned_snapshots_path"])
    trial_plan_path = _inside(repository, manifest["planned_trials_path"])
    if (
        _sha256(snapshot_plan_path) != manifest["planned_snapshots_sha256"]
        or _sha256(trial_plan_path) != manifest["planned_trials_sha256"]
    ):
        raise ValueError("v2 independent plan file SHA mismatch")
    snapshots, trials = _validate_plans(
        _read_plan_csv(snapshot_plan_path, SNAPSHOT_FIELDS),
        _read_plan_csv(trial_plan_path, TRIAL_FIELDS),
    )
    protocol = _strict_object(_inside(repository, manifest["scientific_protocol_path"]))
    if (
        protocol.get("planned_snapshot_identity_sha256")
        != _identity_sha256(snapshots)
        or protocol.get("planned_trial_identity_sha256") != _identity_sha256(trials)
    ):
        raise ValueError("v2 independent plan semantic identity SHA mismatch")
    directory = Path(run_dir).resolve()
    if directory != _inside(repository, manifest["formal_output_dir"]):
        raise ValueError("v2 independent result directory differs from manifest")
    raw_path = directory / "raw_result_manifest.json"
    raw = _strict_object(raw_path)
    if (
        set(raw) != {"schema_version", "run_id", "results"}
        or raw.get("schema_version") != RAW_RESULT_MANIFEST_SCHEMA
        or raw.get("run_id") != FORMAL_RUN_ID
        or type(raw.get("results")) is not dict
    ):
        raise ValueError("v2 independent raw-result manifest schema mismatch")
    trial_by_id = {row["planned_trial_id"]: row for row in trials}
    if set(raw["results"]) != set(trial_by_id):
        raise ValueError("v2 independent raw-result inventory incomplete or has extras")
    raw_root = (directory / "raw_results").resolve()
    trials_by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial_id in sorted(trial_by_id):
        entry = raw["results"][trial_id]
        if (
            type(entry) is not dict
            or set(entry) != {"planned_trial_id", "path", "sha256"}
            or entry.get("planned_trial_id") != trial_id
            or not isinstance(entry.get("path"), str)
            or Path(entry["path"]).name != entry["path"]
        ):
            raise ValueError("v2 independent raw-result entry mismatch")
        result_path = (raw_root / entry["path"]).resolve()
        if (
            result_path.parent != raw_root
            or not result_path.is_file()
            or _sha256(result_path) != entry.get("sha256")
        ):
            raise ValueError("v2 independent raw-result file/SHA mismatch")
        result = _confirmatory_trial(_strict_object(result_path))
        plan = trial_by_id[trial_id]
        exact = {
            "planned_trial_id": trial_id,
            "snapshot_id": plan["planned_snapshot_id"],
            "scene_variant": plan["scene_variant"],
            "condition": plan["condition"],
            "backend": plan["backend"],
        }
        if any(result.get(name) != expected for name, expected in exact.items()):
            raise ValueError("v2 independent raw result differs from frozen plan")
        trials_by_snapshot[plan["planned_snapshot_id"]].append(
            {
                **result,
                "backend_schema_name": result["backend"],
                "geometry_seed": plan["geometry_seed"],
                "measurement_seed": plan["measurement_seed"],
                "planned_snapshot_id": plan["planned_snapshot_id"],
                "repeat_index": plan["repeat_index"],
            }
        )

    cache = _inside(repository, manifest["snapshot_cache_root"])
    lock_path = _inside(repository, manifest["snapshot_lock_path"])
    lineage_report, snapshot_by_id = independently_validate_v2_lineage_inventory(
        cache_root=cache, plans=snapshots, lock_path=lock_path
    )
    lock_sha = _sha256(lock_path)
    if set(trials_by_snapshot) != set(snapshot_by_id):
        raise ValueError("v2 independent trial/snapshot pairing mismatch")
    normalized: list[dict[str, Any]] = []
    common: list[dict[str, Any]] = []
    for snapshot_id in sorted(snapshot_by_id):
        snapshot = snapshot_by_id[snapshot_id]
        chosen = trials_by_snapshot[snapshot_id]
        if len(chosen) != 2 or {row["backend"] for row in chosen} != set(BACKENDS):
            raise ValueError("v2 independent backend pairing mismatch")
        context = None
        if chosen[0]["condition"] != "IDEAL_MATCHED":
            context = _independent_prepare_common(
                snapshot["source"],
                snapshot["target"],
                snapshot["reference"],
                snapshot_id=snapshot_id,
            )
        for row in chosen:
            if any(
                row.get(name) != snapshot[name]
                for name in (
                    "snapshot_checksum",
                    "source_checksum",
                    "target_checksum",
                    "reference_pose_checksum",
                )
            ):
                raise ValueError("v2 independent trial/snapshot checksum mismatch")
            if row.get("snapshot_lock_sha256") != lock_sha:
                raise ValueError("v2 independent trial/snapshot-lock SHA mismatch")
            updated = dict(row)
            transform = row.get("final_transform_4x4")
            if transform is not None and not row["solver_failure"] and row["finite_output"]:
                estimate = np.asarray(transform, dtype=np.float64)
                vector = estimate[:3, 3] - snapshot["reference"][:3, 3]
                translation_error = float(np.linalg.norm(vector))
                rotation_error = rotation_metric_audit(
                    estimate[:3, :3], snapshot["reference"][:3, :3]
                )["rotation_error_rad"]
                if rotation_error is None:
                    raise ValueError("v2 independent rotation recomputation failed")
                if (
                    abs(float(row["translation_update_m"]) - translation_error) > 1.0e-12
                    or abs(float(row["rotation_update_rad"]) - float(rotation_error)) > 1.0e-12
                ):
                    raise ValueError("v2 independent stored metric recomputation mismatch")
                updated.update(
                    translation_error_m=translation_error,
                    translation_vector=vector.astype(float).tolist(),
                    rotation_error_rad=float(rotation_error),
                )
            normalized.append(updated)
            if (
                row["condition"] != "IDEAL_MATCHED"
                and not row["solver_failure"]
                and row["finite_output"]
            ):
                if context is None or transform is None:
                    raise ValueError("v2 independent successful nonideal trial lacks common input")
                common.append(
                    _independent_safe_common_record(
                        context,
                        np.asarray(transform, dtype=np.float64),
                        identifiers={
                            "planned_trial_id": row["planned_trial_id"],
                            "backend_schema_name": row["backend_schema_name"],
                            "scene_variant": row["scene_variant"],
                            "condition": row["condition"],
                            "geometry_seed": row["geometry_seed"],
                            "measurement_seed": row["measurement_seed"],
                            "repeat_index": row["repeat_index"],
                        },
                    )
                )
    return (
        sorted(normalized, key=lambda row: row["planned_trial_id"]),
        sorted(common, key=lambda row: row["planned_trial_id"]),
        manifest,
        raw,
        lineage_report,
    )


def _v2_decision(report: dict[str, Any]) -> dict[str, Any]:
    decision = dict(report["final_decision"])
    decision.update(
        {
            "CONFIRMATORY_V2_RUN_AUTHORIZED": False,
            "SYNTHETIC_CONFIRMATORY_V2_COMPLETE": decision.pop(
                "SYNTHETIC_CONFIRMATORY_COMPLETE"
            ),
            "SYNTHETIC_CONFIRMATORY_V2_EXECUTED": decision.pop(
                "SYNTHETIC_CONFIRMATORY_EXECUTED"
            ),
            "SYNTHETIC_CONFIRMATORY_V2_PASS": decision.pop(
                "SYNTHETIC_CONFIRMATORY_PASS"
            ),
        }
    )
    decision.pop("CONFIRMATORY_RUN_AUTHORIZED", None)
    projection = report["verification_projection"]
    projection["final_decision"] = decision
    report["final_decision"] = decision
    return report


def independently_verify_v2(
    *, manifest_path: str | Path, run_dir: str | Path
) -> dict[str, Any]:
    """Independently verify a complete formal v2 evidence tree."""

    trials, common, manifest, _raw, lineage = independently_read_v2_evidence(
        manifest_path=manifest_path, run_dir=run_dir
    )
    repository = Path(manifest_path).resolve().parent.parent
    protocol = _strict_object(_inside(repository, manifest["scientific_protocol_path"]))
    report = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_strict_object(_inside(repository, manifest["frozen_model_path"])),
        gate_contract=_strict_object(_inside(repository, manifest["gate_contract_path"])),
        expected_geometry_seeds=protocol["geometry_seeds"],
    )
    report = _v2_decision(report)
    report["schema_version"] = INDEPENDENT_SCHEMA
    report["lineage_integrity"] = lineage
    report["run_id"] = FORMAL_RUN_ID
    report["raw_result_manifest_sha256"] = _sha256(
        Path(run_dir).resolve() / "raw_result_manifest.json"
    )
    report["verification_projection"]["lineage_integrity"] = {
        name: lineage[name]
        for name in (
            "IDEAL_PARENT_LINEAGE_COUNT",
            "LINEAGE_INTEGRITY_PASS",
            "LINEAGE_VIOLATION_COUNT",
            "NONIDEAL_NO_LINEAGE_COUNT",
        )
    }
    return report


def independently_analyze_v2_fixture_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently evaluate the fixed three-snapshot/six-trial fixture."""

    rows = [dict(row) for row in results]
    inventory = Counter(
        (row.get("backend"), row.get("condition"), row.get("failure_classification"))
        for row in rows
    )
    expected = Counter(
        {
            (backend, condition, classification): 1
            for backend in BACKENDS
            for condition, classification in (
                ("FIXTURE_IDENTITY", "NONE"),
                ("FIXTURE_NONIDENTITY_REFERENCE", "NONE"),
                ("FIXTURE_NO_CORRESPONDENCE", "NO_CORRESPONDENCES"),
            )
        }
    )
    passed = bool(
        len(rows) == 6
        and inventory == expected
        and all(type(row.get("finite_output")) is bool for row in rows)
    )
    return {
        "decision": {
            "FIXTURE_EXECUTION_CHAIN_PASS": passed,
            "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED": False,
        },
        "failure_inventory": [
            {
                "backend": key[0],
                "condition": key[1],
                "failure_classification": key[2],
                "count": count,
            }
            for key, count in sorted(inventory.items())
        ],
        "fixture_snapshot_count": 3,
        "fixture_trial_count": len(rows),
        "results": sorted(rows, key=lambda row: str(row.get("planned_trial_id"))),
        "schema_version": "synthetic_confirmatory_v2_fixture_independent_verification_v1",
    }


def v2_primary_verification_projection(primary: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "final_decision",
        "gate_summary",
        "integrity",
        "h1_ideal_control",
        "h2_scene_effect",
        "h3_cross_backend_ranking",
        "h4_reassociation",
        "h5_frozen_models",
        "h6_systematic_groups",
        "h6_systematic_backend",
    )
    projection = {name: primary[name] for name in names}
    lineage = primary["lineage_integrity"]
    projection["lineage_integrity"] = {
        name: lineage[name]
        for name in (
            "IDEAL_PARENT_LINEAGE_COUNT",
            "LINEAGE_INTEGRITY_PASS",
            "LINEAGE_VIOLATION_COUNT",
            "NONIDEAL_NO_LINEAGE_COUNT",
        )
    }
    return projection


def compare_v2_primary_and_independent(
    primary: Mapping[str, Any], independent: Mapping[str, Any]
) -> dict[str, Any]:
    expected = v2_primary_verification_projection(primary)
    actual = independent.get("verification_projection")
    if type(actual) is not dict:
        raise ValueError("v2 independent verification projection is missing")
    section_count = leaf_count = 0
    maximum = 0.0
    names: list[str] = []
    for name in expected:
        count, difference = _compare(expected[name], actual.get(name))
        if count:
            section_count += 1
            names.append(name)
        leaf_count += count
        maximum = max(maximum, difference)
    return {
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.0,
        "section_difference_count": section_count,
        "leaf_difference_count": leaf_count,
        "maximum_absolute_numeric_difference": maximum,
        "exact_match_pass": leaf_count == 0 and maximum == 0.0,
        "differing_sections": names,
    }


def compare_v2_fixture_primary_and_independent(
    primary: Mapping[str, Any], independent: Mapping[str, Any]
) -> dict[str, Any]:
    names = (
        "decision",
        "failure_inventory",
        "fixture_snapshot_count",
        "fixture_trial_count",
        "results",
    )
    section_count = leaf_count = 0
    maximum = 0.0
    differing: list[str] = []
    for name in names:
        count, difference = _compare(primary.get(name), independent.get(name))
        if count:
            section_count += 1
            differing.append(name)
        leaf_count += count
        maximum = max(maximum, difference)
    return {
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.0,
        "section_difference_count": section_count,
        "leaf_difference_count": leaf_count,
        "maximum_absolute_numeric_difference": maximum,
        "exact_match_pass": leaf_count == 0 and maximum == 0.0,
        "differing_sections": differing,
    }


# Descriptive aliases retained for callers that spell out the experiment name.
independently_verify_synthetic_confirmatory_v2 = independently_verify_v2
independently_read_synthetic_confirmatory_v2_evidence = independently_read_v2_evidence
independently_recompute_synthetic_confirmatory_v2_lineage = independently_recompute_v2_lineage
independently_verify_v2_fixture_results = independently_analyze_v2_fixture_results
compare_primary_and_independent_v2 = compare_v2_primary_and_independent


__all__ = [
    "compare_v2_fixture_primary_and_independent",
    "compare_v2_primary_and_independent",
    "compare_primary_and_independent_v2",
    "independently_analyze_v2_fixture_results",
    "independently_authenticate_v2_manifest",
    "independently_read_synthetic_confirmatory_v2_evidence",
    "independently_read_v2_evidence",
    "independently_read_v2_snapshot",
    "independently_recompute_synthetic_confirmatory_v2_lineage",
    "independently_recompute_v2_lineage",
    "independently_validate_v2_lineage_inventory",
    "independently_validate_v2_snapshot_lock",
    "independently_verify_synthetic_confirmatory_v2",
    "independently_verify_v2",
    "independently_verify_v2_fixture_results",
    "v2_primary_verification_projection",
]
