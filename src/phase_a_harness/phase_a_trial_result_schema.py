"""Strict, versioned Phase A trial-result contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "phase_a_trial_result_v1"
SCHEMA_RELATIVE_PATH = Path("schemas/phase_a_trial_result_v1.schema.json")
OPEN3D_BACKEND = "open3d_point_to_plane"
PCL_BACKEND = "pcl_point_to_plane"
BACKENDS = frozenset({OPEN3D_BACKEND, PCL_BACKEND})
CONDITIONS = frozenset(
    {
        "FIXTURE_IDENTITY",
        "FIXTURE_NONIDENTITY_REFERENCE",
        "FIXTURE_NO_CORRESPONDENCE",
        "IDEAL_MATCHED",
    }
)
FAILURE_CLASSIFICATIONS = frozenset(
    {
        "NONE",
        "SCIENTIFIC_SOLVER_FAILURE",
        "NONFINITE_OUTPUT",
        "INPUT_PAIRING_VIOLATION",
        "ROTATION_MATRIX_QUALITY_FAILURE",
        "INVALID_NORMALS",
        "NO_CORRESPONDENCES",
        "BACKEND_EXCEPTION",
        "LOCK_MISMATCH",
        "CORRUPT_EXISTING_RESULT",
        "SCHEMA_VALIDATION_FAILURE",
    }
)
REQUIRED_FIELDS = (
    "schema_version",
    "planned_trial_id",
    "snapshot_id",
    "scene_variant",
    "condition",
    "backend",
    "protocol_sha256",
    "snapshot_lock_sha256",
    "snapshot_checksum",
    "source_checksum",
    "target_checksum",
    "reference_pose_checksum",
    "implementation_sha256",
    "solver_failure",
    "failure_classification",
    "failure_detail",
    "final_transform_4x4",
    "translation_update_m",
    "rotation_update_rad",
    "raw_rotation_finite",
    "raw_rotation_determinant",
    "orthogonality_defect_fro",
    "projection_correction_fro",
    "finite_output",
    "runtime_ms",
    "backend_diagnostics",
)
OLD_TOP_LEVEL_ALIASES = frozenset(
    {"solver_failed", "failure_classifications", "cli_exit_code"}
)
OPEN3D_DIAGNOSTIC_FIELDS = frozenset(
    {"fitness", "inlier_rmse", "correspondence_set_size"}
)
PCL_DIAGNOSTIC_FIELDS = frozenset(
    {
        "pcl_version",
        "pcl_cli_sha256",
        "exit_code",
        "has_converged_raw",
        "fitness_score",
        "iteration_count",
        "correspondence_count",
        "source_normal_statistics",
        "target_normal_statistics",
    }
)
NORMAL_STATISTIC_FIELDS = frozenset(
    {"finite_count", "zero_count", "nan_count", "norm_min", "norm_median", "norm_max"}
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TrialResultValidationError(ValueError):
    """Raised when a trial result violates the exact v1 contract."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    compact = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(compact).hexdigest()


def load_json_strict(path: str | Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise TrialResultValidationError(f"non-finite JSON constant: {token}")

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrialResultValidationError(f"invalid JSON: {path}") from error
    if type(value) is not dict:
        raise TrialResultValidationError("trial result root must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str) -> None:
    keys = set(value)
    missing = set(expected) - keys
    unknown = keys - set(expected)
    if missing:
        raise TrialResultValidationError(f"{label} missing fields: {sorted(missing)}")
    if unknown:
        raise TrialResultValidationError(f"{label} unknown fields: {sorted(unknown)}")


def _finite_number(value: Any, label: str, *, nullable: bool = False, nonnegative: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrialResultValidationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise TrialResultValidationError(f"{label} must be finite")
    if nonnegative and result < 0.0:
        raise TrialResultValidationError(f"{label} must be nonnegative")
    return result


def _integer(value: Any, label: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrialResultValidationError(f"{label} must be an integer")
    if nonnegative and value < 0:
        raise TrialResultValidationError(f"{label} must be nonnegative")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise TrialResultValidationError(f"{label} must be 64 lowercase hex characters")
    return value


def _normal_statistics(value: Any, label: str) -> None:
    if type(value) is not dict:
        raise TrialResultValidationError(f"{label} must be an object")
    _require_exact_keys(value, NORMAL_STATISTIC_FIELDS, label)
    for name in ("finite_count", "zero_count", "nan_count"):
        _integer(value[name], f"{label}.{name}", nonnegative=True)
    norms = [
        _finite_number(value[name], f"{label}.{name}", nullable=True, nonnegative=True)
        for name in ("norm_min", "norm_median", "norm_max")
    ]
    present = [item for item in norms if item is not None]
    if present and len(present) != 3:
        raise TrialResultValidationError(f"{label} norm statistics must all be present or null")
    if len(present) == 3 and not (present[0] <= present[1] <= present[2]):
        raise TrialResultValidationError(f"{label} norm statistics are not ordered")


def _validate_diagnostics(backend: str, value: Any) -> None:
    if type(value) is not dict:
        raise TrialResultValidationError("backend_diagnostics must be an object")
    if backend == OPEN3D_BACKEND:
        _require_exact_keys(value, OPEN3D_DIAGNOSTIC_FIELDS, "Open3D diagnostics")
        if "correspondence_count" in value:
            raise TrialResultValidationError("Open3D correspondence_count alias is forbidden")
        _finite_number(value["fitness"], "fitness", nullable=True, nonnegative=True)
        _finite_number(value["inlier_rmse"], "inlier_rmse", nullable=True, nonnegative=True)
        _integer(value["correspondence_set_size"], "correspondence_set_size", nonnegative=True)
        return
    _require_exact_keys(value, PCL_DIAGNOSTIC_FIELDS, "PCL diagnostics")
    if "cli_exit_code" in value:
        raise TrialResultValidationError("PCL cli_exit_code alias is forbidden")
    if not isinstance(value["pcl_version"], str) or not value["pcl_version"]:
        raise TrialResultValidationError("pcl_version must be a non-empty string")
    _sha(value["pcl_cli_sha256"], "pcl_cli_sha256")
    _integer(value["exit_code"], "exit_code")
    if type(value["has_converged_raw"]) is not bool:
        raise TrialResultValidationError("has_converged_raw must be bool")
    _finite_number(value["fitness_score"], "fitness_score", nullable=True, nonnegative=True)
    _integer(value["iteration_count"], "iteration_count", nonnegative=True)
    _integer(value["correspondence_count"], "correspondence_count", nonnegative=True)
    _normal_statistics(value["source_normal_statistics"], "source_normal_statistics")
    _normal_statistics(value["target_normal_statistics"], "target_normal_statistics")


def validate_phase_a_trial_result_strict(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate without migration, alias support, or unknown-field tolerance."""

    if type(value) is not dict:
        raise TrialResultValidationError("trial result must be a plain object")
    aliases = OLD_TOP_LEVEL_ALIASES & set(value)
    if aliases:
        raise TrialResultValidationError(f"old aliases are forbidden: {sorted(aliases)}")
    _require_exact_keys(value, set(REQUIRED_FIELDS), "trial result")
    if value["schema_version"] != SCHEMA_VERSION:
        raise TrialResultValidationError("schema_version mismatch")
    for name in ("planned_trial_id", "snapshot_id", "scene_variant"):
        if not isinstance(value[name], str) or not value[name]:
            raise TrialResultValidationError(f"{name} must be a non-empty string")
    if value["condition"] not in CONDITIONS:
        raise TrialResultValidationError("condition is not authorized")
    backend = value["backend"]
    if backend not in BACKENDS:
        raise TrialResultValidationError("backend is not authorized")
    for name in (
        "protocol_sha256",
        "snapshot_lock_sha256",
        "snapshot_checksum",
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "implementation_sha256",
    ):
        _sha(value[name], name)
    if type(value["solver_failure"]) is not bool:
        raise TrialResultValidationError("solver_failure must be bool")
    if value["failure_classification"] not in FAILURE_CLASSIFICATIONS:
        raise TrialResultValidationError("failure_classification is not authorized")
    if type(value["raw_rotation_finite"]) is not bool or type(value["finite_output"]) is not bool:
        raise TrialResultValidationError("finite flags must be bool")
    _finite_number(value["runtime_ms"], "runtime_ms", nonnegative=True)

    failure = value["solver_failure"]
    if failure:
        if value["failure_classification"] == "NONE":
            raise TrialResultValidationError("failed result cannot have NONE classification")
        if not isinstance(value["failure_detail"], str) or not value["failure_detail"].strip():
            raise TrialResultValidationError("failed result requires failure_detail")
    else:
        if value["failure_classification"] != "NONE" or value["failure_detail"] is not None:
            raise TrialResultValidationError("successful result must have NONE and null detail")
        if value["finite_output"] is not True:
            raise TrialResultValidationError("successful result must have finite_output=true")

    transform = value["final_transform_4x4"]
    if transform is None:
        if not failure:
            raise TrialResultValidationError("successful result requires final_transform_4x4")
        if value["raw_rotation_finite"]:
            raise TrialResultValidationError("null transform cannot have finite rotation")
    else:
        try:
            matrix = np.asarray(transform, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TrialResultValidationError("final_transform_4x4 must contain numbers") from error
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise TrialResultValidationError("final_transform_4x4 must be finite 4x4")
        if any(isinstance(item, (str, bool)) for row in transform for item in row):
            raise TrialResultValidationError("final_transform_4x4 cannot contain string/bool values")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12, rtol=0.0):
            raise TrialResultValidationError("final_transform_4x4 has invalid homogeneous row")
        if not value["raw_rotation_finite"]:
            raise TrialResultValidationError("finite transform requires raw_rotation_finite=true")

    translation = _finite_number(
        value["translation_update_m"], "translation_update_m", nullable=failure, nonnegative=True
    )
    rotation = _finite_number(
        value["rotation_update_rad"], "rotation_update_rad", nullable=failure, nonnegative=True
    )
    for name in ("raw_rotation_determinant", "orthogonality_defect_fro", "projection_correction_fro"):
        _finite_number(
            value[name],
            name,
            nullable=failure,
            nonnegative=name != "raw_rotation_determinant",
        )
    if not failure and (translation is None or rotation is None):
        raise TrialResultValidationError("successful result requires update metrics")
    if value["finite_output"] and transform is None:
        raise TrialResultValidationError("finite_output cannot be true without a transform")
    _validate_diagnostics(backend, value["backend_diagnostics"])

    # Prove that canonical serialization cannot emit NaN/Infinity.
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise TrialResultValidationError("result is not canonical-JSON serializable") from error
    return dict(value)


__all__ = [
    "BACKENDS",
    "CONDITIONS",
    "FAILURE_CLASSIFICATIONS",
    "NORMAL_STATISTIC_FIELDS",
    "OPEN3D_BACKEND",
    "OPEN3D_DIAGNOSTIC_FIELDS",
    "OLD_TOP_LEVEL_ALIASES",
    "PCL_BACKEND",
    "PCL_DIAGNOSTIC_FIELDS",
    "REQUIRED_FIELDS",
    "SCHEMA_RELATIVE_PATH",
    "SCHEMA_VERSION",
    "TrialResultValidationError",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "file_sha256",
    "load_json_strict",
    "validate_phase_a_trial_result_strict",
]
