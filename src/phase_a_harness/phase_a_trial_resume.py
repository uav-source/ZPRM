"""Strict resume validation for complete Phase A trial results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .phase_a_trial_result_schema import (
    file_sha256,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)


class CorruptExistingResult(RuntimeError):
    classification = "CORRUPT_EXISTING_RESULT"


EXPECTED_IDENTITY_FIELDS = (
    "planned_trial_id",
    "snapshot_id",
    "backend",
    "scene_variant",
    "condition",
    "protocol_sha256",
    "snapshot_lock_sha256",
    "snapshot_checksum",
    "source_checksum",
    "target_checksum",
    "reference_pose_checksum",
    "implementation_sha256",
)


def validate_existing_trial_result_for_resume(
    path: str | Path,
    *,
    manifest_entry: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
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
        value = validate_phase_a_trial_result_strict(load_json_strict(result_path))
    except (ValueError, OSError) as error:
        raise CorruptExistingResult("existing result failed strict schema") from error
    missing_expected = set(EXPECTED_IDENTITY_FIELDS) - set(expected)
    if missing_expected:
        raise ValueError(f"resume expectation is incomplete: {sorted(missing_expected)}")
    for name in EXPECTED_IDENTITY_FIELDS:
        if value.get(name) != expected.get(name):
            raise CorruptExistingResult(f"existing result mismatch: {name}")
    return value


__all__ = [
    "CorruptExistingResult",
    "EXPECTED_IDENTITY_FIELDS",
    "validate_existing_trial_result_for_resume",
]
