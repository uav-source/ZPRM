"""Durable attempt events, deliberately separate from trial results."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .phase_a_trial_result_schema import BACKENDS


EVENT_VERSION = "phase_a_attempt_event_v1"
EVENT_TYPES = frozenset(
    {
        "STARTED",
        "COMPLETED",
        "INFRASTRUCTURE_INTERRUPTION",
        "RESUMED",
        "SKIPPED_VALID_RESULT",
        "REJECTED_CORRUPT_RESULT",
    }
)


def append_attempt_event(
    path: str | Path,
    *,
    planned_trial_id: str,
    snapshot_id: str,
    backend: str,
    event_type: str,
    detail: str | None,
) -> dict[str, Any]:
    if backend not in BACKENDS or event_type not in EVENT_TYPES:
        raise ValueError("attempt event backend/type is not authorized")
    event = {
        "backend": backend,
        "detail": detail,
        "event_type": event_type,
        "event_version": EVENT_VERSION,
        "planned_trial_id": planned_trial_id,
        "snapshot_id": snapshot_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with destination.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return event


def read_attempt_events(path: str | Path) -> list[dict[str, Any]]:
    destination = Path(path)
    if not destination.exists():
        return []
    events: list[dict[str, Any]] = []
    for index, line in enumerate(destination.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid attempt event line {index}") from error
        if type(event) is not dict or set(event) != {
            "event_version",
            "planned_trial_id",
            "snapshot_id",
            "backend",
            "event_type",
            "timestamp_utc",
            "detail",
        }:
            raise ValueError(f"attempt event schema mismatch at line {index}")
        if event["event_version"] != EVENT_VERSION or event["backend"] not in BACKENDS or event["event_type"] not in EVENT_TYPES:
            raise ValueError(f"attempt event value mismatch at line {index}")
        events.append(event)
    return events


__all__ = ["EVENT_TYPES", "EVENT_VERSION", "append_attempt_event", "read_attempt_events"]
