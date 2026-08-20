"""Pure-Python validator for future FMB1 formal trial-result rows.

This module deliberately imports no registration backend.  A row is checked
against both ``formal_trial_result_schema_v1.json`` and a caller-supplied,
already-authenticated lock context.  The lock context has this minimal shape::

    {
      "FORMAL_AUTHORITY": bool,
      "FORMAL_REGISTRATION_AUTHORIZED": bool,
      "formal_lock_sha256": "<64 lowercase hex>",
      "backend_parameter_contract_sha256": "<frozen contract sha>",
      "FIXTURE_ONLY": bool,        # required true for fixture contexts
      "NOT_REAL_FMB1": bool,       # required true for fixture contexts
      "expected_trials": {
        "<trial id>": {
          "scene_id": "FMB1_R01",
          "station_id": "S01",
          "snapshot_id": "FMB1_R01_S01_Q01",
          "track": "ZERO_PERTURBATION_MAINLINE" | "CAPTURE_BASIN",
          "execution_kind": "FORMAL" | "FIXTURE",
          "backend": "open3d" | "pcl",
          "backend_version": "...",
          "source_sha256": "...",
          "target_sha256": "...",
          "T0": [[...], [...], [...], [...]]
        }
      }
    }

The validator does not create, authenticate, or activate a formal lock.  In
particular, the repository's current ``FORMAL_REGISTRATION_AUTHORIZED=false``
state is unchanged.  Synthetic fixture rows can be validated only as
``FIXTURE`` and are rejected whenever publication is requested.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "mid360_fmb1_formal_trial_result_v1"
SCHEMA_RELATIVE_PATH = Path(__file__).with_name("formal_trial_result_schema_v1.json")
BACKEND_PARAMETER_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
BACKEND_VERSIONS = {
    "open3d": "0.19.0+b012259",
    "pcl": "1.15.1",
}
TRACKS = frozenset({"ZERO_PERTURBATION_MAINLINE", "CAPTURE_BASIN"})
TRACK_CLASSIFICATIONS = frozenset(
    {"ZERO_PERTURBATION_TRACK", "CAPTURE_RADIUS_TRACK", "FIXTURE_ONLY"}
)
EXECUTION_KINDS = frozenset({"FORMAL", "FIXTURE"})
TRIAL_STATUSES = frozenset({"COMPLETED", "BACKEND_FAILURE"})
IDENTITY_4X4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SCENE_PATTERN = re.compile(r"^FMB1_[RW][0-9]{2}$")
SNAPSHOT_PATTERN = re.compile(r"^(FMB1_[RW][0-9]{2})_(S0[1-3])_Q(0[1-9]|10)$")

REQUIRED_FIELDS = (
    "schema",
    "trial_id",
    "batch_id",
    "track",
    "track_classification",
    "execution_kind",
    "fixture_only",
    "publish_eligible",
    "formal_authority",
    "formal_lock_sha256",
    "scene_id",
    "station_id",
    "snapshot_id",
    "backend",
    "backend_version",
    "backend_parameter_contract_sha256",
    "source_sha256",
    "target_sha256",
    "source_point_count",
    "target_point_count",
    "T0",
    "T_est",
    "trial_status",
    "solver_status",
    "solver_success",
    "finite_transform",
    "finite_result",
    "pose_recovered",
    "translation_vector_m",
    "translation_norm_m",
    "rotation_vector_deg",
    "rotation_angle_deg",
    "initial_correspondence_count",
    "initial_correspondence_fraction",
    "LOW_INITIAL_OVERLAP",
    "final_correspondence_count",
    "correspondence_turnover",
    "fitness",
    "final_residual_rmse",
    "final_translation_error_m",
    "final_rotation_error_deg",
    "iteration_count",
    "runtime_ms",
    "failure_reason",
    "created_at_utc",
    "MEASUREMENT_FINAL_RESULT",
)

LOCK_BINDING_FIELDS = (
    "scene_id",
    "station_id",
    "snapshot_id",
    "track",
    "execution_kind",
    "backend",
    "backend_version",
    "source_sha256",
    "target_sha256",
)


class FormalTrialResultValidationError(ValueError):
    """Raised when a result violates schema, lock, or publication policy."""


def _require_plain_object(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise FormalTrialResultValidationError(f"{label} must be a plain object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise FormalTrialResultValidationError(f"{label} missing fields: {sorted(missing)}")
    if unknown:
        raise FormalTrialResultValidationError(f"{label} unknown fields: {sorted(unknown)}")


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise FormalTrialResultValidationError(f"{label} must be bool")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormalTrialResultValidationError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise FormalTrialResultValidationError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_integer(
    value: Any, label: str, *, minimum: int = 0, nullable: bool = False
) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormalTrialResultValidationError(f"{label} must be an integer")
    if value < minimum:
        raise FormalTrialResultValidationError(f"{label} must be >= {minimum}")
    return value


def _require_finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    nullable: bool = False,
) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalTrialResultValidationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise FormalTrialResultValidationError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise FormalTrialResultValidationError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise FormalTrialResultValidationError(f"{label} must be <= {maximum}")
    return result


def _determinant_3x3(matrix: tuple[tuple[float, ...], ...]) -> float:
    a = matrix
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def _require_rigid_matrix(value: Any, label: str) -> tuple[tuple[float, ...], ...]:
    if type(value) is not list or len(value) != 4:
        raise FormalTrialResultValidationError(f"{label} must be a 4x4 list")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        if type(row) is not list or len(row) != 4:
            raise FormalTrialResultValidationError(f"{label}[{row_index}] must have 4 values")
        rows.append(
            tuple(
                _require_finite_number(item, f"{label}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )
        )
    matrix = tuple(rows)
    if any(abs(matrix[3][index] - expected) > 1.0e-12 for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise FormalTrialResultValidationError(f"{label} has an invalid homogeneous row")
    rotation = tuple(tuple(matrix[row][column] for column in range(3)) for row in range(3))
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[index][left] * rotation[index][right] for index in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > 1.0e-6:
                raise FormalTrialResultValidationError(f"{label} rotation is not orthonormal")
    if abs(_determinant_3x3(rotation) - 1.0) > 1.0e-6:
        raise FormalTrialResultValidationError(f"{label} rotation determinant is not +1")
    return matrix


def _require_vector3(value: Any, label: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise FormalTrialResultValidationError(f"{label} must be a three-value list")
    return tuple(
        _require_finite_number(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _relative_displacement(
    initial: tuple[tuple[float, ...], ...],
    estimate: tuple[tuple[float, ...], ...],
) -> tuple[
    tuple[float, float, float],
    float,
    tuple[float, float, float],
    float,
]:
    translation_difference = tuple(estimate[row][3] - initial[row][3] for row in range(3))
    translation = tuple(
        sum(initial[row][column] * translation_difference[row] for row in range(3))
        for column in range(3)
    )
    norm = math.sqrt(sum(component * component for component in translation))
    relative_rotation = tuple(
        tuple(
            sum(initial[index][row] * estimate[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    cosine = (sum(relative_rotation[index][index] for index in range(3)) - 1.0) / 2.0
    angle_rad = math.acos(max(-1.0, min(1.0, cosine)))
    angle_deg = math.degrees(angle_rad)
    if angle_rad <= 1.0e-12:
        rotation_vector = (0.0, 0.0, 0.0)
    elif abs(math.pi - angle_rad) <= 1.0e-7:
        # A pi rotation has a sign-ambiguous axis.  Pick a deterministic axis
        # from the diagonal; exact sign is not scientifically identifiable.
        components = [
            math.sqrt(max(0.0, (relative_rotation[index][index] + 1.0) / 2.0))
            for index in range(3)
        ]
        axis_norm = math.sqrt(sum(component * component for component in components))
        axis = tuple(component / axis_norm for component in components)
        rotation_vector = tuple(component * angle_deg for component in axis)
    else:
        denominator = 2.0 * math.sin(angle_rad)
        axis = (
            (relative_rotation[2][1] - relative_rotation[1][2]) / denominator,
            (relative_rotation[0][2] - relative_rotation[2][0]) / denominator,
            (relative_rotation[1][0] - relative_rotation[0][1]) / denominator,
        )
        rotation_vector = tuple(component * angle_deg for component in axis)
    return translation, norm, rotation_vector, angle_deg


def _require_close(actual: float, expected: float, label: str) -> None:
    tolerance = 1.0e-8 + 1.0e-8 * max(abs(actual), abs(expected))
    if abs(actual - expected) > tolerance:
        raise FormalTrialResultValidationError(f"{label} is inconsistent with T0/T_est")


def _require_utc_timestamp(value: Any, label: str) -> str:
    text = _require_nonempty_string(value, label)
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise FormalTrialResultValidationError(f"{label} must be ISO-8601") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise FormalTrialResultValidationError(f"{label} must carry an explicit UTC offset")
    return text


def load_formal_trial_result_json(path: str | Path) -> dict[str, Any]:
    """Load JSON while rejecting the non-standard NaN/Infinity constants."""

    def reject_constant(token: str) -> None:
        raise FormalTrialResultValidationError(f"non-finite JSON constant: {token}")

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    except FormalTrialResultValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalTrialResultValidationError(f"invalid JSON: {path}") from error
    return dict(_require_plain_object(payload, "trial result"))


def _validate_lock_and_binding(
    payload: Mapping[str, Any], lock: Mapping[str, Any]
) -> Mapping[str, Any]:
    lock = _require_plain_object(lock, "formal lock context")
    for field in (
        "FORMAL_AUTHORITY",
        "FORMAL_REGISTRATION_AUTHORIZED",
        "formal_lock_sha256",
        "backend_parameter_contract_sha256",
        "expected_trials",
    ):
        if field not in lock:
            raise FormalTrialResultValidationError(f"formal lock context missing field: {field}")
    _require_bool(lock["FORMAL_AUTHORITY"], "formal lock context.FORMAL_AUTHORITY")
    _require_bool(
        lock["FORMAL_REGISTRATION_AUTHORIZED"],
        "formal lock context.FORMAL_REGISTRATION_AUTHORIZED",
    )
    lock_sha = _require_sha256(lock["formal_lock_sha256"], "formal lock context.formal_lock_sha256")
    contract_sha = _require_sha256(
        lock["backend_parameter_contract_sha256"],
        "formal lock context.backend_parameter_contract_sha256",
    )
    if contract_sha != BACKEND_PARAMETER_CONTRACT_SHA256:
        raise FormalTrialResultValidationError("formal lock context backend contract SHA mismatch")
    if payload["formal_lock_sha256"] != lock_sha:
        raise FormalTrialResultValidationError("formal_lock_sha256 does not match lock context")

    trials = _require_plain_object(lock["expected_trials"], "formal lock context.expected_trials")
    trial_id = payload["trial_id"]
    if trial_id not in trials:
        raise FormalTrialResultValidationError(f"trial_id is outside the lock: {trial_id}")
    expected = _require_plain_object(trials[trial_id], f"expected trial {trial_id}")
    missing = (set(LOCK_BINDING_FIELDS) | {"T0"}) - set(expected)
    if missing:
        raise FormalTrialResultValidationError(
            f"expected trial {trial_id} missing fields: {sorted(missing)}"
        )
    for field in LOCK_BINDING_FIELDS:
        if payload[field] != expected[field]:
            raise FormalTrialResultValidationError(
                f"{field} does not match locked trial {trial_id}"
            )
    expected_t0 = _require_rigid_matrix(expected["T0"], f"expected trial {trial_id}.T0")
    if _require_rigid_matrix(payload["T0"], "T0") != expected_t0:
        raise FormalTrialResultValidationError(f"T0 does not match locked trial {trial_id}")
    return expected


def validate_formal_trial_result(
    payload: Mapping[str, Any],
    formal_lock: Mapping[str, Any],
    *,
    for_publication: bool = False,
) -> dict[str, Any]:
    """Validate one exact result row and return a detached JSON-compatible copy.

    ``for_publication`` adds the publication gate; it never upgrades a row's
    authority.  Formal rows require an authorization context that already has
    both authority flags true.  Fixture rows require both flags false and are
    unconditionally rejected for publication.
    """

    payload = _require_plain_object(payload, "trial result")
    _require_exact_keys(payload, set(REQUIRED_FIELDS), "trial result")
    if payload["schema"] != SCHEMA_NAME:
        raise FormalTrialResultValidationError("schema mismatch")
    _require_nonempty_string(payload["trial_id"], "trial_id")
    if payload["batch_id"] != "FMB1":
        raise FormalTrialResultValidationError("batch_id must be FMB1")
    if payload["track"] not in TRACKS:
        raise FormalTrialResultValidationError("track is not allowed by schema")
    if payload["track_classification"] not in TRACK_CLASSIFICATIONS:
        raise FormalTrialResultValidationError("track_classification is invalid")
    if payload["execution_kind"] not in EXECUTION_KINDS:
        raise FormalTrialResultValidationError("execution_kind is invalid")
    for field in (
        "fixture_only",
        "publish_eligible",
        "formal_authority",
        "solver_success",
        "finite_transform",
        "finite_result",
        "LOW_INITIAL_OVERLAP",
        "MEASUREMENT_FINAL_RESULT",
    ):
        _require_bool(payload[field], field)
    if payload["MEASUREMENT_FINAL_RESULT"] is not False:
        raise FormalTrialResultValidationError("a trial row cannot be a final measurement result")

    _require_sha256(payload["formal_lock_sha256"], "formal_lock_sha256")
    if not isinstance(payload["scene_id"], str) or SCENE_PATTERN.fullmatch(payload["scene_id"]) is None:
        raise FormalTrialResultValidationError("scene_id has invalid FMB1 form")
    if payload["station_id"] not in {"S01", "S02", "S03"}:
        raise FormalTrialResultValidationError("station_id must be S01, S02, or S03")
    snapshot_match = (
        SNAPSHOT_PATTERN.fullmatch(payload["snapshot_id"])
        if isinstance(payload["snapshot_id"], str)
        else None
    )
    if snapshot_match is None:
        raise FormalTrialResultValidationError("snapshot_id has invalid FMB1 form")
    if snapshot_match.group(1) != payload["scene_id"] or snapshot_match.group(2) != payload["station_id"]:
        raise FormalTrialResultValidationError("snapshot_id is inconsistent with scene/station")

    backend = payload["backend"]
    if backend not in BACKEND_VERSIONS:
        raise FormalTrialResultValidationError("backend is not allowed")
    if payload["backend_version"] != BACKEND_VERSIONS[backend]:
        raise FormalTrialResultValidationError("backend_version mismatch")
    contract_sha = _require_sha256(
        payload["backend_parameter_contract_sha256"],
        "backend_parameter_contract_sha256",
    )
    if contract_sha != BACKEND_PARAMETER_CONTRACT_SHA256:
        raise FormalTrialResultValidationError("backend parameter contract SHA mismatch")
    _require_sha256(payload["source_sha256"], "source_sha256")
    _require_sha256(payload["target_sha256"], "target_sha256")
    _require_integer(payload["source_point_count"], "source_point_count", minimum=1)
    _require_integer(payload["target_point_count"], "target_point_count", minimum=1)

    t0 = _require_rigid_matrix(payload["T0"], "T0")
    if payload["track"] == "ZERO_PERTURBATION_MAINLINE" and t0 != IDENTITY_4X4:
        raise FormalTrialResultValidationError(
            "ZERO_PERTURBATION_MAINLINE requires exact identity T0"
        )
    _validate_lock_and_binding(payload, formal_lock)

    if payload["trial_status"] not in TRIAL_STATUSES:
        raise FormalTrialResultValidationError("trial_status is invalid")
    if payload["solver_status"] not in {"CONVERGED", "NOT_CONVERGED", "BACKEND_ERROR"}:
        raise FormalTrialResultValidationError("solver_status is invalid")
    _require_integer(
        payload["initial_correspondence_count"],
        "initial_correspondence_count",
    )
    _require_finite_number(
        payload["initial_correspondence_fraction"],
        "initial_correspondence_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    _require_finite_number(payload["runtime_ms"], "runtime_ms", minimum=0.0)
    _require_utc_timestamp(payload["created_at_utc"], "created_at_utc")

    if payload["trial_status"] == "COMPLETED":
        if payload["finite_transform"] is not True or payload["finite_result"] is not True:
            raise FormalTrialResultValidationError(
                "COMPLETED requires finite_transform=true and finite_result=true"
            )
        if (payload["solver_status"] == "CONVERGED") != payload["solver_success"]:
            raise FormalTrialResultValidationError(
                "solver_status and solver_success are inconsistent"
            )
        if payload["solver_status"] not in {"CONVERGED", "NOT_CONVERGED"}:
            raise FormalTrialResultValidationError("COMPLETED solver_status is invalid")
        estimate_matrix = _require_rigid_matrix(payload["T_est"], "T_est")
        _require_bool(payload["pose_recovered"], "pose_recovered")
        translation_vector = _require_vector3(
            payload["translation_vector_m"], "translation_vector_m"
        )
        translation_norm = _require_finite_number(
            payload["translation_norm_m"], "translation_norm_m", minimum=0.0
        )
        rotation_vector = _require_vector3(
            payload["rotation_vector_deg"], "rotation_vector_deg"
        )
        rotation_angle = _require_finite_number(
            payload["rotation_angle_deg"],
            "rotation_angle_deg",
            minimum=0.0,
            maximum=180.0,
        )
        (
            expected_vector,
            expected_norm,
            expected_rotation_vector,
            expected_angle,
        ) = _relative_displacement(t0, estimate_matrix)
        for index, component in enumerate(translation_vector):
            _require_close(component, expected_vector[index], f"translation_vector_m[{index}]")
        _require_close(translation_norm, expected_norm, "translation_norm_m")
        for index, component in enumerate(rotation_vector):
            _require_close(
                component,
                expected_rotation_vector[index],
                f"rotation_vector_deg[{index}]",
            )
        _require_close(rotation_angle, expected_angle, "rotation_angle_deg")
        _require_integer(payload["final_correspondence_count"], "final_correspondence_count")
        _require_finite_number(
            payload["correspondence_turnover"],
            "correspondence_turnover",
            minimum=0.0,
            maximum=1.0,
        )
        for field in (
            "fitness",
            "final_residual_rmse",
            "final_translation_error_m",
            "final_rotation_error_deg",
        ):
            _require_finite_number(payload[field], field, minimum=0.0)
        _require_integer(payload["iteration_count"], "iteration_count")
        if payload["failure_reason"] is not None:
            raise FormalTrialResultValidationError("COMPLETED requires failure_reason=null")
    else:
        if (
            payload["solver_status"] != "BACKEND_ERROR"
            or payload["solver_success"] is not False
            or payload["finite_transform"] is not False
            or payload["finite_result"] is not False
        ):
            raise FormalTrialResultValidationError(
                "BACKEND_FAILURE requires BACKEND_ERROR and false solver/finite flags"
            )
        for field in (
            "T_est",
            "pose_recovered",
            "translation_vector_m",
            "translation_norm_m",
            "rotation_vector_deg",
            "rotation_angle_deg",
            "final_correspondence_count",
            "correspondence_turnover",
            "fitness",
            "final_residual_rmse",
            "final_translation_error_m",
            "final_rotation_error_deg",
            "iteration_count",
        ):
            if payload[field] is not None:
                raise FormalTrialResultValidationError(
                    f"BACKEND_FAILURE requires {field}=null"
                )
        _require_nonempty_string(payload["failure_reason"], "failure_reason")

    lock_authority = _require_bool(
        formal_lock["FORMAL_AUTHORITY"], "formal lock context.FORMAL_AUTHORITY"
    )
    lock_authorized = _require_bool(
        formal_lock["FORMAL_REGISTRATION_AUTHORIZED"],
        "formal lock context.FORMAL_REGISTRATION_AUTHORIZED",
    )
    if payload["execution_kind"] == "FIXTURE":
        if payload["track_classification"] != "FIXTURE_ONLY":
            raise FormalTrialResultValidationError(
                "FIXTURE requires track_classification=FIXTURE_ONLY"
            )
        if (
            payload["fixture_only"] is not True
            or payload["publish_eligible"] is not False
            or payload["formal_authority"] is not False
        ):
            raise FormalTrialResultValidationError(
                "FIXTURE requires fixture_only=true, publish_eligible=false, formal_authority=false"
            )
        if lock_authority or lock_authorized:
            raise FormalTrialResultValidationError("fixture lock context cannot carry formal authority")
        if (
            formal_lock.get("FIXTURE_ONLY") is not True
            or formal_lock.get("NOT_REAL_FMB1") is not True
        ):
            raise FormalTrialResultValidationError(
                "fixture lock context requires FIXTURE_ONLY=true and NOT_REAL_FMB1=true"
            )
        if for_publication:
            raise FormalTrialResultValidationError("fixture results cannot be published")
    else:
        expected_classification = (
            "ZERO_PERTURBATION_TRACK"
            if payload["track"] == "ZERO_PERTURBATION_MAINLINE"
            else "CAPTURE_RADIUS_TRACK"
        )
        if payload["track_classification"] != expected_classification:
            raise FormalTrialResultValidationError(
                "formal track_classification is inconsistent with track"
            )
        if (
            payload["fixture_only"] is not False
            or payload["publish_eligible"] is not True
            or payload["formal_authority"] is not True
        ):
            raise FormalTrialResultValidationError(
                "FORMAL requires fixture_only=false, publish_eligible=true, formal_authority=true"
            )
        if not lock_authority or not lock_authorized:
            raise FormalTrialResultValidationError(
                "FORMAL result requires an already-authorized formal lock context"
            )

    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise FormalTrialResultValidationError(
            "trial result is not finite canonical-JSON serializable"
        ) from error
    return json.loads(canonical)


def validate_trial_result(
    payload: Mapping[str, Any],
    formal_lock: Mapping[str, Any],
    *,
    for_publication: bool = False,
) -> dict[str, Any]:
    """Compatibility alias for :func:`validate_formal_trial_result`."""

    return validate_formal_trial_result(
        payload, formal_lock, for_publication=for_publication
    )


__all__ = [
    "BACKEND_PARAMETER_CONTRACT_SHA256",
    "BACKEND_VERSIONS",
    "FormalTrialResultValidationError",
    "IDENTITY_4X4",
    "REQUIRED_FIELDS",
    "SCHEMA_NAME",
    "load_formal_trial_result_json",
    "validate_formal_trial_result",
    "validate_trial_result",
]
