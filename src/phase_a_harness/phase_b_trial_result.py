"""Phase B condition extension over the frozen Phase A trial-result contract.

The Phase A validator remains the single authority for the 26 result fields,
schema version, backend diagnostics, metrics, and failure classifications.  A
temporary copy is validated with the frozen ``IDEAL_MATCHED`` condition; the
stored and returned payload always retains its real Phase B condition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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
from .phase_a_trial_resume import (
    EXPECTED_IDENTITY_FIELDS,
    CorruptExistingResult,
)


PHASE_B_CONDITIONS = frozenset({"INDEPENDENT_NOISE_FREE", "FULL_NOISE"})
_PHASE_A_VALIDATION_CONDITION = "IDEAL_MATCHED"


def validate_phase_b_trial_result_strict(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate Phase B while preserving the byte-frozen Phase A contract.

    No migration, aliases, additional fields, or alternate schema version are
    accepted.  ``value`` is never mutated.
    """

    if type(value) is not dict:
        raise TrialResultValidationError("trial result must be a plain object")
    condition = value.get("condition")
    if condition not in PHASE_B_CONDITIONS:
        raise TrialResultValidationError("Phase B condition is not authorized")
    normalized = dict(value)
    normalized["condition"] = _PHASE_A_VALIDATION_CONDITION
    validated = validate_phase_a_trial_result_strict(normalized)
    validated["condition"] = condition
    # These checks document the extension boundary and fail closed if the
    # frozen validator's public contract is ever changed.
    if tuple(validated) != tuple(value):
        raise TrialResultValidationError("frozen validator changed result field order")
    if set(validated) != set(REQUIRED_FIELDS) or len(validated) != 26:
        raise TrialResultValidationError("Phase B result must retain all 26 Phase A fields")
    if validated["schema_version"] != SCHEMA_VERSION:
        raise TrialResultValidationError("schema_version mismatch")
    return validated


def write_phase_b_trial_result(
    directory: str | Path, result: Mapping[str, Any]
) -> tuple[Path, str]:
    """Validate and atomically create one canonical Phase B result file."""

    validated = validate_phase_b_trial_result_strict(result)
    destination = Path(directory) / result_filename(validated["planned_trial_id"])
    atomic_write_bytes(destination, canonical_json_bytes(validated), replace=False)
    return destination, file_sha256(destination)


def validate_existing_phase_b_trial_result_for_resume(
    path: str | Path,
    *,
    manifest_entry: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept a completed Phase B result only after SHA, schema, and identity checks."""

    result_path = Path(path)
    if manifest_entry is None:
        raise CorruptExistingResult("existing result has no manifest entry")
    expected_manifest_keys = {"planned_trial_id", "path", "sha256"}
    if set(manifest_entry) != expected_manifest_keys:
        raise CorruptExistingResult("result manifest entry schema mismatch")
    if manifest_entry.get("planned_trial_id") != expected.get("planned_trial_id"):
        raise CorruptExistingResult("manifest trial identity mismatch")
    if manifest_entry.get("path") != result_path.name:
        raise CorruptExistingResult("manifest result path mismatch")
    try:
        actual_sha = file_sha256(result_path)
    except OSError as error:
        raise CorruptExistingResult("existing result cannot be read") from error
    if manifest_entry.get("sha256") != actual_sha:
        raise CorruptExistingResult("existing result SHA mismatch")
    try:
        value = validate_phase_b_trial_result_strict(load_json_strict(result_path))
    except (ValueError, OSError) as error:
        raise CorruptExistingResult("existing result failed strict Phase B schema") from error
    missing_expected = set(EXPECTED_IDENTITY_FIELDS) - set(expected)
    if missing_expected:
        raise ValueError(f"resume expectation is incomplete: {sorted(missing_expected)}")
    if expected.get("condition") not in PHASE_B_CONDITIONS:
        raise ValueError("resume expectation has an unauthorized Phase B condition")
    for name in EXPECTED_IDENTITY_FIELDS:
        if value.get(name) != expected.get(name):
            raise CorruptExistingResult(f"existing result mismatch: {name}")
    return value


__all__ = [
    "PHASE_B_CONDITIONS",
    "validate_existing_phase_b_trial_result_for_resume",
    "validate_phase_b_trial_result_strict",
    "write_phase_b_trial_result",
]
