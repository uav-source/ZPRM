"""Fail-closed deterministic fixture replay for the Stage-2 full test gate.

This module exists for one narrow reason: the repository's historical lifecycle
tests exercise the qualified Open3D/PCL fixture adapters.  During Boreas v2
Stage-2 data preparation those tests must retain their identities and assertions
without executing either registration backend.  The adapters therefore replay a
small, deterministic result materialization only when all three formal gates are
authenticated:

* ``ZPRM_REAL_DATA_PREP_NO_REGISTRATION=1``;
* ``ZPRM_STAGE2_FORMAL_FULL_PYTEST=1``; and
* an immutable marker file whose exact SHA-256 is carried in the environment.

Normal and partially configured environments never silently select replay.  A
partial Stage-2 marker raises :class:`Stage2SafeFixtureReplayError`; the lone
NO-registration variable remains compatible with the older preparation guard.
Every replay is recorded in one process-safe, hash-chained JSONL journal shared
by the pytest child and copied-repository Python descendants.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    validate_phase_a_trial_result_strict,
)


NO_REGISTRATION_ENV = "ZPRM_REAL_DATA_PREP_NO_REGISTRATION"
FORMAL_FULL_PYTEST_ENV = "ZPRM_STAGE2_FORMAL_FULL_PYTEST"
MARKER_PATH_ENV = "ZPRM_STAGE2_FORMAL_FULL_PYTEST_MARKER"
MARKER_SHA256_ENV = "ZPRM_STAGE2_FORMAL_FULL_PYTEST_MARKER_SHA256"

MARKER_SCHEMA = "zprm.boreas.v2.stage2.formal_full_pytest_marker.v1"
MARKER_PURPOSE = (
    "BOREAS_V2_STAGE2_DATA_PREPARATION_FULL_TEST_NO_REGISTRATION"
)
REPLAY_CATALOG_SCHEMA = (
    "zprm.boreas.v2.stage2.safe_fixture_replay_catalog.v1"
)
REPLAY_EVENT_SCHEMA = (
    "zprm.boreas.v2.stage2.safe_fixture_replay_event.v1"
)
EMPTY_EVENT_CHAIN_SHA256 = "0" * 64
PCL_CLI_SHA256 = (
    "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff"
)

AUTHORIZED_CONDITIONS = (
    "FIXTURE_IDENTITY",
    "FIXTURE_NONIDENTITY_REFERENCE",
    "FIXTURE_NO_CORRESPONDENCE",
)
AUTHORIZED_BACKENDS = (OPEN3D_BACKEND, PCL_BACKEND)

REPLAY_CATALOG: dict[str, Any] = {
    "schema": REPLAY_CATALOG_SCHEMA,
    "authorized_backends": list(AUTHORIZED_BACKENDS),
    "authorized_conditions": list(AUTHORIZED_CONDITIONS),
    "authorized_parameter_sha256": {
        OPEN3D_BACKEND: (
            "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413"
        ),
        PCL_BACKEND: (
            "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd"
        ),
    },
    "backend_execution_policy": "NEVER_EXECUTE_REGISTRATION_BACKEND",
    "diagnostic_policy": "FROZEN_SEED_FREE_FIXTURE_DIAGNOSTICS",
    "failure_policy": {
        "FIXTURE_IDENTITY": "NONE",
        "FIXTURE_NONIDENTITY_REFERENCE": "NONE",
        "FIXTURE_NO_CORRESPONDENCE": "NO_CORRESPONDENCES",
    },
    "pcl_cli_sha256": PCL_CLI_SHA256,
    "runtime_ms": 0.0,
    "transform_policy": "AUTHENTICATED_FIXTURE_REFERENCE_POSE",
}

# Deliberately independent of Python formatting and insertion order.  A test
# freezes this literal; import also fails if the catalog changes without review.
REPLAY_CATALOG_SHA256 = (
    "5402064dcfa87d489541685f57723bca5fd2b8128ec13762590af00637ba5fb9"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MARKER_FIELDS = frozenset(
    {
        "event_log_path",
        "formal_full_pytest",
        "formal_python_executable",
        "git_head",
        "marker_payload_sha256",
        "no_registration",
        "purpose",
        "registration_execution_count",
        "repository_root",
        "runtime_lifecycle_fixture_sha256",
        "safe_fixture_replay_sha256",
        "schema",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "actual_registration_execution_count",
        "backend",
        "condition",
        "event_sha256",
        "event_version",
        "planned_trial_id",
        "previous_event_sha256",
        "process_id",
        "replay_catalog_sha256",
        "replay_result_sha256",
        "safe_simulated_materialization_count",
        "sequence",
        "snapshot_id",
    }
)
_COMMON_FIELDS = frozenset(
    {
        "backend",
        "condition",
        "implementation_sha256",
        "planned_trial_id",
        "protocol_sha256",
        "reference_pose_checksum",
        "scene_variant",
        "schema_version",
        "snapshot_checksum",
        "snapshot_id",
        "snapshot_lock_sha256",
        "source_checksum",
        "target_checksum",
    }
)


class Stage2SafeFixtureReplayError(RuntimeError):
    """The formal safe-replay contract is absent, partial, or corrupted."""


def _compact_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_bytes(value: Any) -> bytes:
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


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_compact_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise Stage2SafeFixtureReplayError(
                    f"duplicate key in {label}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Stage2SafeFixtureReplayError(f"invalid {label}") from error
    if type(value) is not dict:
        raise Stage2SafeFixtureReplayError(f"{label} must be a JSON object")
    return value


def _canonical_regular(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise Stage2SafeFixtureReplayError(f"{label} is not absolute")
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise Stage2SafeFixtureReplayError(f"{label} is absent") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or path.resolve(strict=True) != path
    ):
        raise Stage2SafeFixtureReplayError(f"{label} is unsafe")
    return path


def _canonical_directory(path: Path, *, label: str) -> Path:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
        or path.resolve(strict=True) != path
    ):
        raise Stage2SafeFixtureReplayError(f"{label} is unsafe")
    return path


def _catalog_sha256() -> str:
    actual = _value_sha256(REPLAY_CATALOG)
    if actual != REPLAY_CATALOG_SHA256:
        raise Stage2SafeFixtureReplayError("safe fixture replay catalog changed")
    return actual


def build_formal_stage2_marker(
    *,
    repository_root: str | Path,
    git_head: str,
    formal_python_executable: str | Path,
    event_log_path: str | Path,
) -> dict[str, Any]:
    """Build, but never write, the exact authenticated marker value."""

    repository = _canonical_directory(
        Path(repository_root), label="formal marker repository"
    )
    python = _canonical_regular(
        Path(formal_python_executable), label="formal marker Python"
    )
    current_python = Path(sys.executable).resolve(strict=True)
    if python != current_python:
        raise Stage2SafeFixtureReplayError(
            "marker-bound Python differs from the running interpreter"
        )
    event_log = Path(event_log_path)
    _canonical_directory(event_log.parent, label="safe replay event parent")
    if (
        not event_log.is_absolute()
        or event_log.parent.resolve(strict=True) != event_log.parent
        or event_log.exists()
        or event_log.is_symlink()
    ):
        raise Stage2SafeFixtureReplayError(
            "safe replay event log destination is unsafe"
        )
    if not _GIT_COMMIT.fullmatch(git_head):
        raise Stage2SafeFixtureReplayError("formal marker Git HEAD is invalid")
    replay_path = (
        repository
        / "src/phase_a_harness/real_data_preparation/"
        "stage2_safe_fixture_replay.py"
    )
    lifecycle_path = repository / "src/phase_a_harness/runtime_lifecycle_fixture.py"
    _canonical_regular(replay_path, label="marker safe replay implementation")
    _canonical_regular(lifecycle_path, label="marker runtime lifecycle implementation")
    unsigned = {
        "event_log_path": str(event_log),
        "formal_full_pytest": True,
        "formal_python_executable": str(python),
        "git_head": git_head,
        "no_registration": True,
        "purpose": MARKER_PURPOSE,
        "registration_execution_count": 0,
        "repository_root": str(repository),
        "runtime_lifecycle_fixture_sha256": _file_sha256(lifecycle_path),
        "safe_fixture_replay_sha256": _file_sha256(replay_path),
        "schema": MARKER_SCHEMA,
    }
    return {**unsigned, "marker_payload_sha256": _value_sha256(unsigned)}


def formal_stage2_environment(
    *, marker_path: str | Path, marker_sha256: str
) -> dict[str, str]:
    """Return the four exact variables descendants must inherit unchanged."""

    return {
        NO_REGISTRATION_ENV: "1",
        FORMAL_FULL_PYTEST_ENV: "1",
        MARKER_PATH_ENV: str(Path(marker_path)),
        MARKER_SHA256_ENV: marker_sha256,
    }


def authenticate_formal_stage2_environment(
    environment: Mapping[str, str] | None = None,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    """Authenticate the triple gate, returning marker identity when active."""

    values: Mapping[str, str] = os.environ if environment is None else environment
    formal = values.get(FORMAL_FULL_PYTEST_ENV)
    marker_raw = values.get(MARKER_PATH_ENV)
    expected_sha = values.get(MARKER_SHA256_ENV)
    staged = formal is not None or marker_raw is not None or expected_sha is not None
    if not staged:
        if required:
            raise Stage2SafeFixtureReplayError(
                "formal Stage-2 full-pytest marker is absent"
            )
        # NO_REGISTRATION alone is the established guard API, not replay.
        return None
    if (
        formal != "1"
        or values.get(NO_REGISTRATION_ENV) != "1"
        or marker_raw is None
        or expected_sha is None
        or not _SHA256.fullmatch(expected_sha)
    ):
        raise Stage2SafeFixtureReplayError(
            "formal Stage-2 safe replay environment is partial"
        )
    marker_path = _canonical_regular(
        Path(marker_raw), label="formal Stage-2 marker"
    )
    payload = marker_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise Stage2SafeFixtureReplayError("formal Stage-2 marker SHA mismatch")
    value = _strict_object_bytes(payload, label="formal Stage-2 marker")
    if payload != _canonical_bytes(value):
        raise Stage2SafeFixtureReplayError(
            "formal Stage-2 marker is not canonical JSON"
        )
    unsigned = dict(value)
    claim = unsigned.pop("marker_payload_sha256", None)
    if (
        set(value) != _MARKER_FIELDS
        or value.get("schema") != MARKER_SCHEMA
        or value.get("purpose") != MARKER_PURPOSE
        or value.get("formal_full_pytest") is not True
        or value.get("no_registration") is not True
        or value.get("registration_execution_count") != 0
        or claim != _value_sha256(unsigned)
        or not _GIT_COMMIT.fullmatch(str(value.get("git_head", "")))
    ):
        raise Stage2SafeFixtureReplayError(
            "formal Stage-2 marker content differs"
        )
    repository = _canonical_directory(
        Path(str(value.get("repository_root", ""))),
        label="marker-bound repository",
    )
    python = _canonical_regular(
        Path(str(value.get("formal_python_executable", ""))),
        label="marker-bound Python",
    )
    current_python = Path(sys.executable).resolve(strict=True)
    if python != current_python:
        raise Stage2SafeFixtureReplayError(
            "marker-bound Python differs from the running interpreter"
        )
    expected_module = (
        repository
        / "src/phase_a_harness/real_data_preparation/"
        "stage2_safe_fixture_replay.py"
    )
    runtime_module = repository / "src/phase_a_harness/runtime_lifecycle_fixture.py"
    current_module = Path(__file__).resolve(strict=True)
    current_runtime_module = current_module.parents[1] / "runtime_lifecycle_fixture.py"
    try:
        implementations_match = (
            _file_sha256(current_module)
            == value.get("safe_fixture_replay_sha256")
            and _file_sha256(expected_module)
            == value.get("safe_fixture_replay_sha256")
            and _file_sha256(current_runtime_module)
            == value.get("runtime_lifecycle_fixture_sha256")
            and _file_sha256(runtime_module)
            == value.get("runtime_lifecycle_fixture_sha256")
        )
    except OSError as error:
        raise Stage2SafeFixtureReplayError(
            "safe replay copied-repository implementation is absent"
        ) from error
    if not implementations_match:
        raise Stage2SafeFixtureReplayError(
            "safe replay copied-repository implementation SHA differs"
        )
    event_log = Path(str(value.get("event_log_path", "")))
    if (
        not event_log.is_absolute()
        or event_log == marker_path
        or event_log.parent != marker_path.parent
        or event_log.parent.resolve(strict=True) != event_log.parent
        or event_log.is_symlink()
    ):
        raise Stage2SafeFixtureReplayError(
            "marker-bound safe replay event log is unsafe"
        )
    return {
        "event_log_path": event_log,
        "formal_python_executable": python,
        "git_head": value["git_head"],
        "marker_path": marker_path,
        "marker_sha256": expected_sha,
        "repository_root": repository,
        "value": value,
    }


def safe_fixture_replay_active() -> bool:
    return authenticate_formal_stage2_environment(required=False) is not None


def initialize_safe_replay_event_log() -> Path:
    """Create the one empty formal journal before test execution starts."""

    binding = authenticate_formal_stage2_environment(required=True)
    assert binding is not None
    path = binding["event_log_path"]
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise Stage2SafeFixtureReplayError(
            "safe replay event log was not fresh"
        ) from error
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    _canonical_regular(path, label="safe replay event log")
    return path


def _lock_path(event_log: Path) -> Path:
    return event_log.with_name(f".{event_log.name}.lock")


def _event_unsigned(
    *,
    sequence: int,
    previous: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "actual_registration_execution_count": 0,
        "backend": result["backend"],
        "condition": result["condition"],
        "event_version": REPLAY_EVENT_SCHEMA,
        "planned_trial_id": result["planned_trial_id"],
        "previous_event_sha256": previous,
        "process_id": os.getpid(),
        "replay_catalog_sha256": _catalog_sha256(),
        "replay_result_sha256": _value_sha256(result),
        "safe_simulated_materialization_count": 1,
        "sequence": sequence,
        "snapshot_id": result["snapshot_id"],
    }


def _read_events_locked(event_log: Path) -> list[dict[str, Any]]:
    _canonical_regular(event_log, label="safe replay event log")
    try:
        payload = event_log.read_bytes()
    except OSError as error:
        raise Stage2SafeFixtureReplayError(
            "safe replay event log cannot be read"
        ) from error
    if payload and not payload.endswith(b"\n"):
        raise Stage2SafeFixtureReplayError(
            "safe replay event log has a partial line"
        )
    events: list[dict[str, Any]] = []
    previous = EMPTY_EVENT_CHAIN_SHA256
    for sequence, raw in enumerate(payload.splitlines(), start=1):
        value = _strict_object_bytes(raw, label="safe replay event")
        if raw != _compact_bytes(value):
            raise Stage2SafeFixtureReplayError(
                "safe replay event is not compact canonical JSON"
            )
        unsigned = dict(value)
        claim = unsigned.pop("event_sha256", None)
        if (
            set(value) != _EVENT_FIELDS
            or value.get("event_version") != REPLAY_EVENT_SCHEMA
            or value.get("sequence") != sequence
            or value.get("previous_event_sha256") != previous
            or value.get("actual_registration_execution_count") != 0
            or value.get("safe_simulated_materialization_count") != 1
            or value.get("backend") not in AUTHORIZED_BACKENDS
            or value.get("condition") not in AUTHORIZED_CONDITIONS
            or value.get("replay_catalog_sha256") != _catalog_sha256()
            or not isinstance(value.get("process_id"), int)
            or value.get("process_id", 0) <= 0
            or not _SHA256.fullmatch(str(value.get("replay_result_sha256", "")))
            or claim != _value_sha256(unsigned)
        ):
            raise Stage2SafeFixtureReplayError(
                "safe replay event chain differs"
            )
        previous = str(claim)
        events.append(value)
    return events


def _with_event_lock(event_log: Path, operation: Any) -> Any:
    lock_path = _lock_path(event_log)
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise Stage2SafeFixtureReplayError(
            "safe replay event lock cannot be opened"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Stage2SafeFixtureReplayError(
                "safe replay event lock is unsafe"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return operation()
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _append_replay_event(result: Mapping[str, Any]) -> None:
    binding = authenticate_formal_stage2_environment(required=True)
    assert binding is not None
    event_log: Path = binding["event_log_path"]

    def append() -> None:
        events = _read_events_locked(event_log)
        previous = (
            events[-1]["event_sha256"]
            if events
            else EMPTY_EVENT_CHAIN_SHA256
        )
        unsigned = _event_unsigned(
            sequence=len(events) + 1,
            previous=str(previous),
            result=result,
        )
        event = {**unsigned, "event_sha256": _value_sha256(unsigned)}
        flags = os.O_WRONLY | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(event_log, flags)
        try:
            payload = _compact_bytes(event) + b"\n"
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise Stage2SafeFixtureReplayError(
                        "safe replay event append made no progress"
                    )
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    _with_event_lock(event_log, append)


def summarize_safe_replay_event_log() -> dict[str, Any]:
    """Verify and project the complete cross-process replay journal."""

    binding = authenticate_formal_stage2_environment(required=True)
    assert binding is not None
    event_log: Path = binding["event_log_path"]
    events = _with_event_lock(
        event_log, lambda: _read_events_locked(event_log)
    )
    counts = Counter(str(event["backend"]) for event in events)
    return {
        "safe_fixture_replay_catalog_sha256": _catalog_sha256(),
        "safe_fixture_replay_event_chain_head_sha256": (
            events[-1]["event_sha256"]
            if events
            else EMPTY_EVENT_CHAIN_SHA256
        ),
        "safe_fixture_replay_event_count": len(events),
        "safe_fixture_replay_event_log_path": str(event_log),
        "safe_fixture_replay_event_log_sha256": _file_sha256(event_log),
        "safe_fixture_replay_event_schema": REPLAY_EVENT_SCHEMA,
        "safe_fixture_replay_marker_path": str(binding["marker_path"]),
        "safe_fixture_replay_marker_sha256": binding["marker_sha256"],
        "safe_simulated_materialization_count": len(events),
        "safe_simulated_open3d_materialization_count": counts[OPEN3D_BACKEND],
        "safe_simulated_pcl_materialization_count": counts[PCL_BACKEND],
    }


def remove_safe_replay_event_lock() -> None:
    """Remove only the exact, empty-of-lockers sidecar after session finish."""

    binding = authenticate_formal_stage2_environment(required=True)
    assert binding is not None
    event_log: Path = binding["event_log_path"]
    lock_path = _lock_path(event_log)
    if not lock_path.exists():
        return
    _canonical_regular(lock_path, label="safe replay event lock")
    descriptor = os.open(
        lock_path,
        os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_path.unlink()
    except BlockingIOError as error:
        raise Stage2SafeFixtureReplayError(
            "safe replay event lock is still held"
        ) from error
    finally:
        os.close(descriptor)


def _rotation_fields(reference: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(reference, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise Stage2SafeFixtureReplayError(
            "safe replay fixture reference is not finite 4x4"
        )
    if not np.allclose(
        matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12, rtol=0.0
    ):
        raise Stage2SafeFixtureReplayError(
            "safe replay fixture reference homogeneous row differs"
        )
    rotation = matrix[:3, :3]
    left, _singular, right = np.linalg.svd(rotation)
    projected = left @ right
    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right
    determinant = float(np.linalg.det(rotation))
    orthogonality = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    )
    correction = float(np.linalg.norm(rotation - projected, ord="fro"))
    if not all(math.isfinite(value) for value in (determinant, orthogonality, correction)):
        raise Stage2SafeFixtureReplayError(
            "safe replay rotation audit is non-finite"
        )
    return {
        "final_transform_4x4": matrix.tolist(),
        "orthogonality_defect_fro": orthogonality,
        "projection_correction_fro": correction,
        "raw_rotation_determinant": determinant,
        "raw_rotation_finite": True,
        "rotation_update_rad": 0.0,
        "translation_update_m": 0.0,
    }


def _normal_statistics(point_count: int) -> dict[str, Any]:
    return {
        "finite_count": point_count,
        "nan_count": 0,
        "norm_max": 1.0,
        "norm_median": 1.0,
        "norm_min": 1.0,
        "zero_count": 0,
    }


def replay_fixture_result(
    *,
    fixture: Any,
    common: Mapping[str, Any],
    parameters: Mapping[str, Any],
    pcl_cli: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize one strict result without importing or calling a backend."""

    authenticate_formal_stage2_environment(required=True)
    _catalog_sha256()
    if set(common) != _COMMON_FIELDS:
        raise Stage2SafeFixtureReplayError(
            "safe replay common-record field set differs"
        )
    backend = common.get("backend")
    condition = common.get("condition")
    if (
        backend not in AUTHORIZED_BACKENDS
        or condition not in AUTHORIZED_CONDITIONS
        or getattr(fixture, "condition", None) != condition
        or getattr(fixture, "snapshot_id", None) != common.get("snapshot_id")
        or not isinstance(parameters, Mapping)
        or _value_sha256(parameters)
        != REPLAY_CATALOG["authorized_parameter_sha256"][backend]
    ):
        raise Stage2SafeFixtureReplayError(
            "safe replay fixture/backend binding differs"
        )
    checksums = getattr(fixture, "checksums", None)
    if not isinstance(checksums, Mapping) or any(
        common.get(field) != checksums.get(field)
        for field in (
            "reference_pose_checksum",
            "snapshot_checksum",
            "source_checksum",
            "target_checksum",
        )
    ):
        raise Stage2SafeFixtureReplayError(
            "safe replay fixture checksum binding differs"
        )
    source = np.asarray(getattr(fixture, "source", None))
    target = np.asarray(getattr(fixture, "target", None))
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1:] != (3,)
        or target.shape[1:] != (3,)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
    ):
        raise Stage2SafeFixtureReplayError(
            "safe replay fixture point arrays differ"
        )
    if backend == OPEN3D_BACKEND:
        if pcl_cli is not None:
            raise Stage2SafeFixtureReplayError(
                "Open3D safe replay received a PCL executable"
            )
        diagnostics: dict[str, Any] = {
            "correspondence_set_size": (
                0 if condition == "FIXTURE_NO_CORRESPONDENCE" else len(source)
            ),
            "fitness": 0.0 if condition == "FIXTURE_NO_CORRESPONDENCE" else 1.0,
            "inlier_rmse": 0.0,
        }
    else:
        if pcl_cli is None:
            raise Stage2SafeFixtureReplayError(
                "PCL safe replay lacks its frozen executable binding"
            )
        authenticated_pcl = _canonical_regular(
            Path(pcl_cli), label="safe replay frozen PCL executable"
        )
        binding = authenticate_formal_stage2_environment(required=True)
        assert binding is not None
        bound_pcl = _canonical_regular(
            binding["repository_root"] / "bin/pcl_point_to_plane_cli",
            label="marker-bound frozen PCL executable",
        )
        if (
            _file_sha256(authenticated_pcl) != PCL_CLI_SHA256
            or _file_sha256(bound_pcl) != PCL_CLI_SHA256
        ):
            raise Stage2SafeFixtureReplayError(
                "safe replay frozen PCL executable SHA differs"
            )
        diagnostics = {
            "correspondence_count": (
                0 if condition == "FIXTURE_NO_CORRESPONDENCE" else len(source)
            ),
            "exit_code": 0,
            "fitness_score": (
                1.7976931348623157e308
                if condition == "FIXTURE_NO_CORRESPONDENCE"
                else 0.0
            ),
            "has_converged_raw": condition != "FIXTURE_NO_CORRESPONDENCE",
            "iteration_count": 0 if condition == "FIXTURE_NO_CORRESPONDENCE" else 1,
            "pcl_cli_sha256": PCL_CLI_SHA256,
            "pcl_version": "1.15.1",
            "source_normal_statistics": _normal_statistics(len(source)),
            "target_normal_statistics": _normal_statistics(len(target)),
        }
    failure = condition == "FIXTURE_NO_CORRESPONDENCE"
    result = validate_phase_a_trial_result_strict(
        {
            **dict(common),
            **_rotation_fields(np.asarray(fixture.reference)),
            "backend_diagnostics": diagnostics,
            "failure_classification": "NO_CORRESPONDENCES" if failure else "NONE",
            "failure_detail": (
                f"{('Open3D' if backend == OPEN3D_BACKEND else 'PCL')} returned zero correspondences"
                if failure
                else None
            ),
            "finite_output": True,
            "runtime_ms": 0.0,
            "solver_failure": failure,
        }
    )
    _append_replay_event(result)
    return result


def record_authenticated_test_double_result(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Attest the one frozen pytest double without weakening production routing.

    The caller authenticates the exact pytest callable identity.  This function
    then applies the same strict result schema and durable event-chain projection
    used by deterministic replay.
    """

    authenticate_formal_stage2_environment(required=True)
    result = validate_phase_a_trial_result_strict(value)
    if (
        result["backend"] not in AUTHORIZED_BACKENDS
        or result["condition"] not in AUTHORIZED_CONDITIONS
    ):
        raise Stage2SafeFixtureReplayError(
            "authenticated pytest double result identity differs"
        )
    _append_replay_event(result)
    return result


__all__ = [
    "EMPTY_EVENT_CHAIN_SHA256",
    "FORMAL_FULL_PYTEST_ENV",
    "MARKER_PATH_ENV",
    "MARKER_PURPOSE",
    "MARKER_SCHEMA",
    "MARKER_SHA256_ENV",
    "NO_REGISTRATION_ENV",
    "REPLAY_CATALOG",
    "REPLAY_CATALOG_SCHEMA",
    "REPLAY_CATALOG_SHA256",
    "REPLAY_EVENT_SCHEMA",
    "Stage2SafeFixtureReplayError",
    "authenticate_formal_stage2_environment",
    "build_formal_stage2_marker",
    "formal_stage2_environment",
    "initialize_safe_replay_event_log",
    "remove_safe_replay_event_lock",
    "record_authenticated_test_double_result",
    "replay_fixture_result",
    "safe_fixture_replay_active",
    "summarize_safe_replay_event_log",
]
