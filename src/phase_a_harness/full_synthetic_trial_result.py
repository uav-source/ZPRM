"""Five-condition extension of the frozen 26-field Phase A trial contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .full_synthetic_development_protocol import NEW_CONDITIONS
from .phase_a_trial_result_schema import (
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    TrialResultValidationError,
    canonical_json_bytes,
    file_sha256,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .phase_a_trial_result_writer import atomic_write_bytes, result_filename
from .phase_a_trial_resume import EXPECTED_IDENTITY_FIELDS, CorruptExistingResult


FULL_SYNTHETIC_CONDITIONS = frozenset(NEW_CONDITIONS)
_NORMALIZED_CONDITION = "IDEAL_MATCHED"


def validate_full_synthetic_trial_result_strict(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TrialResultValidationError("trial result must be a plain object")
    condition = value.get("condition")
    if condition not in FULL_SYNTHETIC_CONDITIONS:
        raise TrialResultValidationError("Full Synthetic condition is not authorized")
    normalized = dict(value)
    normalized["condition"] = _NORMALIZED_CONDITION
    validated = validate_phase_a_trial_result_strict(normalized)
    validated["condition"] = condition
    if tuple(validated) != tuple(value):
        raise TrialResultValidationError("frozen validator changed result field order")
    if set(validated) != set(REQUIRED_FIELDS) or len(validated) != 26:
        raise TrialResultValidationError("result must retain all 26 Phase A fields")
    if validated["schema_version"] != SCHEMA_VERSION:
        raise TrialResultValidationError("schema version changed")
    return validated


def write_full_synthetic_trial_result(
    directory: str | Path, result: Mapping[str, Any]
) -> tuple[Path, str]:
    validated = validate_full_synthetic_trial_result_strict(result)
    destination = Path(directory) / result_filename(validated["planned_trial_id"])
    atomic_write_bytes(destination, canonical_json_bytes(validated), replace=False)
    return destination, file_sha256(destination)


def validate_existing_full_synthetic_trial_result_for_resume(
    path: str | Path,
    *,
    manifest_entry: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = Path(path)
    if manifest_entry is None or set(manifest_entry) != {
        "planned_trial_id",
        "path",
        "sha256",
    }:
        raise CorruptExistingResult("existing result manifest entry is invalid")
    if (
        manifest_entry.get("planned_trial_id") != expected.get("planned_trial_id")
        or manifest_entry.get("path") != result_path.name
    ):
        raise CorruptExistingResult("existing result manifest identity mismatch")
    try:
        if file_sha256(result_path) != manifest_entry.get("sha256"):
            raise CorruptExistingResult("existing result SHA mismatch")
        value = validate_full_synthetic_trial_result_strict(load_json_strict(result_path))
    except (OSError, ValueError) as error:
        if isinstance(error, CorruptExistingResult):
            raise
        raise CorruptExistingResult("existing result failed strict schema") from error
    if set(EXPECTED_IDENTITY_FIELDS) - set(expected):
        raise ValueError("resume expectation is incomplete")
    if expected.get("condition") not in FULL_SYNTHETIC_CONDITIONS:
        raise ValueError("resume expectation condition is unauthorized")
    for name in EXPECTED_IDENTITY_FIELDS:
        if value.get(name) != expected.get(name):
            raise CorruptExistingResult(f"existing result mismatch: {name}")
    return value


__all__ = [
    "FULL_SYNTHETIC_CONDITIONS",
    "validate_existing_full_synthetic_trial_result_for_resume",
    "validate_full_synthetic_trial_result_strict",
    "write_full_synthetic_trial_result",
]
