"""Single canonical and atomic writer for Phase A trial results."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Mapping

from .phase_a_trial_result_schema import (
    canonical_json_bytes,
    file_sha256,
    validate_phase_a_trial_result_strict,
)


class TrialResultWriteError(RuntimeError):
    pass


def result_filename(planned_trial_id: str) -> str:
    if not isinstance(planned_trial_id, str) or not planned_trial_id:
        raise ValueError("planned_trial_id must be non-empty")
    return f"{hashlib.sha256(planned_trial_id.encode('utf-8')).hexdigest()}.json"


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, payload: bytes, *, replace: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite: {destination}")
    temporary = destination.with_name(
        f"{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    if temporary.exists():
        raise TrialResultWriteError(f"temporary path exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() and not replace:
            raise FileExistsError(f"destination appeared during write: {destination}")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_phase_a_trial_result(
    directory: str | Path, result: Mapping[str, Any]
) -> tuple[Path, str]:
    validated = validate_phase_a_trial_result_strict(result)
    destination = Path(directory) / result_filename(validated["planned_trial_id"])
    atomic_write_bytes(destination, canonical_json_bytes(validated), replace=False)
    return destination, file_sha256(destination)


__all__ = [
    "TrialResultWriteError",
    "atomic_write_bytes",
    "result_filename",
    "write_phase_a_trial_result",
]
