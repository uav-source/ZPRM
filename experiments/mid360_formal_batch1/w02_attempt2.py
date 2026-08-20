"""Administrative reclassification of the 2026-08-20 capture as W02 attempt 2.

The underlying acquisition audit is byte-for-byte the same registration-free
reader used for the previously named W04 candidate.  This module changes only
the pre-backend scene/attempt identity after the operator confirmed that the
first W02 capture had been made at the wrong location.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Union

from .preregistration_acquisition import to_json_serializable
from .w04_replacement import run_w04_acquisition


SCENE_ID = "FMB1_W02"
ATTEMPT = 2
CORRECTION_REASON = "OPERATOR_CONFIRMED_WRONG_SCENE_LOCATION"
INVALID_ATTEMPT_REASON = "WRONG_SCENE_LOCATION / OPERATOR_SCENE_SELECTION_ERROR"


def _scene_relabel(value: Any, *, source: str, destination: str) -> Any:
    """Relabel exact scientific identifiers without altering filesystem paths."""

    if isinstance(value, Mapping):
        return {
            key: _scene_relabel(child, source=source, destination=destination)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _scene_relabel(child, source=source, destination=destination)
            for child in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _scene_relabel(child, source=source, destination=destination)
            for child in value
        )
    if isinstance(value, str):
        if value == source:
            return destination
        if value.startswith(source + "_") and "/" not in value:
            return destination + value[len(source) :]
    return copy.deepcopy(value)


def as_w02_attempt2(value: Any) -> Any:
    """Convert exact W04 identifiers to W02 while preserving file paths."""

    return _scene_relabel(value, source="FMB1_W04", destination=SCENE_ID)


def as_w04_compatibility(value: Any) -> Any:
    """Create an in-memory compatibility view for the older strict composer."""

    return _scene_relabel(value, source=SCENE_ID, destination="FMB1_W04")


def run_w02_attempt2_acquisition(
    repository: Union[Path, str], config: Union[Mapping[str, Any], Path, str]
) -> dict[str, Any]:
    """Re-read and audit the exact six new bags as W02 attempt 2."""

    base = run_w04_acquisition(repository, config)
    payload = as_w02_attempt2(base)
    payload["schema"] = "mid360_fmb1_w02_attempt2_acquisition_v1"
    payload["scene_id"] = SCENE_ID
    payload["attempt"] = ATTEMPT
    payload["W02_ATTEMPT2_INPUT_INVENTORY_FAIL"] = payload.pop(
        "W04_INPUT_INVENTORY_FAIL"
    )
    payload["W02_ATTEMPT2_ACQUISITION_PASS"] = payload.pop(
        "W04_ACQUISITION_PASS"
    )
    payload["W02_ATTEMPT2_GEOMETRY_ADMISSION_NOT_RUN"] = payload.pop(
        "W04_GEOMETRY_ADMISSION_NOT_RUN"
    )
    payload["administrative_correction"] = {
        "correction_reason": CORRECTION_REASON,
        "correction_before_any_icp": True,
        "formal_trial_count_at_correction": 0,
        "old_attempt": 1,
        "new_attempt": ATTEMPT,
        "failed_attempts_retained": True,
        "failed_raw_data_retained": True,
    }
    for collection in ("raw_bags", "mapping", "stations"):
        rows = payload.get(collection, [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    row["attempt"] = ATTEMPT
                    row["attempt_status"] = "VALID_ACQUISITION"
    normalized = to_json_serializable(payload)
    json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    return normalized


__all__ = [
    "ATTEMPT",
    "CORRECTION_REASON",
    "INVALID_ATTEMPT_REASON",
    "SCENE_ID",
    "as_w02_attempt2",
    "as_w04_compatibility",
    "run_w02_attempt2_acquisition",
]
