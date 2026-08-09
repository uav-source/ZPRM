"""Seed-agnostic four-state bootstrap for an external formal runtime.

The state machine exists to keep command provenance outside the scientific
execution path without confusing a pre-created runtime root with a resumable
run.  It performs no seed derivation, snapshot construction, backend import,
or scientific analysis.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import stat
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .runtime_lifecycle_io import (
    FORMAL_BOOTSTRAP_BINDING_VERSION,
    IMMUTABLE_RUN_LOCK_VERSION,
    SingleWriterLease,
    atomic_create_bytes,
    atomic_publish_directory,
    assert_no_symlink_path,
    bind_formal_bootstrap_to_run_contract,
    canonical_json_sha256,
    canonical_json_bytes,
    read_canonical_json,
    read_regular_bytes,
    resume_immutable_run_lock,
    write_once_formal_bootstrap_run_lock,
)


FORMAL_COMMAND_LOG_NAME = "formal_command.log"
FORMAL_COMMAND_SHA256_NAME = "formal_command.log.sha256"
IMMUTABLE_RUN_LOCK_NAME = "immutable_run_lock.json"
SINGLE_WRITER_LEASE_NAME = "single_writer.lease"
FORMAL_COMMAND_SHA256_SCHEMA = "formal_command_log_sha256_v1"


class FormalRuntimeState(str, Enum):
    ABSENT = "FORMAL_RUNTIME_ABSENT"
    BOOTSTRAP_ONLY = "FORMAL_RUNTIME_BOOTSTRAP_ONLY"
    RESUMABLE = "FORMAL_RUNTIME_RESUMABLE"
    INVALID = "FORMAL_RUNTIME_INVALID"


FORMAL_RUNTIME_ABSENT = FormalRuntimeState.ABSENT
FORMAL_RUNTIME_BOOTSTRAP_ONLY = FormalRuntimeState.BOOTSTRAP_ONLY
FORMAL_RUNTIME_RESUMABLE = FormalRuntimeState.RESUMABLE
FORMAL_RUNTIME_INVALID = FormalRuntimeState.INVALID


class FormalRuntimeStateError(RuntimeError):
    """Base error for a forbidden or ambiguous runtime transition."""


class InvalidFormalRuntimeError(FormalRuntimeStateError):
    """The external root contains an unknown, unsafe, or corrupt inventory."""


class FormalBootstrapError(FormalRuntimeStateError):
    """A requested fresh/resume bootstrap transition is not authorized."""


class FormalCommandBindingError(FormalRuntimeStateError):
    """The canonical formal command and its SHA sidecar do not bind."""


_BOOTSTRAP_TOP_LEVEL = frozenset(
    {
        FORMAL_COMMAND_LOG_NAME,
        FORMAL_COMMAND_SHA256_NAME,
    }
)
_RESUMABLE_TOP_LEVEL = frozenset(
    {
        *_BOOTSTRAP_TOP_LEVEL,
        IMMUTABLE_RUN_LOCK_NAME,
        # Version-specific lifecycle adapters may persist audit-only evidence
        # beside the version-neutral runtime payload.  These names are files,
        # never directories, and cannot replace any core lock/result path.
        "completeness_report.json",
        "environment_report.json",
        "preflight_report.json",
        "run_summary.json",
        "run_summary.md",
        "smoke_qualification_report.json",
        "source_asset_hashes_after.json",
        "source_asset_hashes_before.json",
        "analysis",
        "artifact_staging",
        "backend_tmp",
        "event_logs",
        "publisher_staging",
        "raw_result_manifest.json",
        "raw_results",
        "run_manifest.json",
        "snapshot_cache",
        "snapshot_lock.json",
        "verification",
        "working_inventory",
    }
)
_EXPECTED_DIRECTORY_NAMES = frozenset(
    {
        "analysis",
        "artifact_staging",
        "backend_tmp",
        "event_logs",
        "publisher_staging",
        "raw_results",
        "snapshot_cache",
        "verification",
        "working_inventory",
    }
)


def _mode(value: str) -> str:
    if value not in {"fresh", "resume"}:
        raise ValueError("formal runtime mode must be 'fresh' or 'resume'")
    return value


def canonical_formal_command(command: str | Sequence[str]) -> bytes:
    """Return the sole accepted one-line UTF-8 command-log representation."""

    if isinstance(command, str):
        value = command
    elif isinstance(command, Sequence) and not isinstance(command, (bytes, bytearray)):
        if not command or any(type(item) is not str or not item for item in command):
            raise TypeError("formal command argv must contain non-empty strings")
        value = shlex.join(list(command))
    else:
        raise TypeError("formal command must be a string or argv sequence")
    if (
        not value
        or value.strip() != value
        or "\x00" in value
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError("formal command must be one canonical non-empty line")
    payload = value.encode("utf-8", errors="strict") + b"\n"
    if payload.decode("utf-8", errors="strict").encode("utf-8") != payload:
        raise ValueError("formal command UTF-8 round trip failed")
    return payload


def formal_command_sha256(command: str | Sequence[str]) -> str:
    return hashlib.sha256(canonical_formal_command(command)).hexdigest()


def bootstrap_lease_path(root: str | Path) -> Path:
    runtime = Path(os.path.abspath(os.fspath(root)))
    return runtime.parent / f".{runtime.name}.bootstrap.lease"


def formal_runtime_path_contract(root: str | Path) -> dict[str, Path]:
    """Return the state machine's complete version-neutral runtime layout.

    The lifecycle spec still supplies every path explicitly.  This projection
    lets the generic orchestrator authenticate that explicit layout against
    the bootstrap state machine without duplicating any filename policy in
    the orchestrator itself.
    """

    runtime = Path(os.path.abspath(os.fspath(root)))
    return {
        "runtime_root": runtime,
        "snapshot_cache": runtime / "snapshot_cache",
        "snapshot_lock": runtime / "snapshot_lock.json",
        "raw_results": runtime / "raw_results",
        "raw_manifest": runtime / "raw_result_manifest.json",
        "event_log": runtime / "event_logs" / "attempt_events.ndjson",
        "backend_temporary": runtime / "backend_tmp",
        "analysis": runtime / "analysis",
        "verification": runtime / "verification",
        "publisher_staging": runtime / "publisher_staging",
        "artifact_staging": runtime / "artifact_staging",
        "temporary_inventory": runtime / "working_inventory",
        "run_manifest": runtime / "run_manifest.json",
        "formal_command_log": runtime / FORMAL_COMMAND_LOG_NAME,
        "formal_command_sha256": runtime / FORMAL_COMMAND_SHA256_NAME,
        "immutable_run_lock": runtime / IMMUTABLE_RUN_LOCK_NAME,
        "single_writer_lease": bootstrap_lease_path(runtime),
    }


def build_formal_runner_command(
    *,
    repository: str | Path,
    manifest_path: str | Path,
    run_id: str,
    runtime_root: str | Path,
    workers: int,
    mode: str,
    entry_script: str,
) -> str:
    """Build a canonical, explicitly version-bound formal runner command."""

    selected_mode = _mode(mode)
    root = Path(repository).resolve()
    manifest_raw = Path(manifest_path)
    manifest = (
        manifest_raw.resolve()
        if manifest_raw.is_absolute()
        else (root / manifest_raw).resolve()
    )
    if root not in manifest.parents:
        raise ValueError("formal manifest must remain inside the harness repository")
    manifest_relative = manifest.relative_to(root).as_posix()
    runtime = Path(os.path.abspath(os.fspath(runtime_root)))
    if not runtime.is_absolute():
        raise ValueError("formal runtime root must be absolute")
    if type(run_id) is not str or not run_id or "\x00" in run_id:
        raise ValueError("formal run ID is invalid")
    if type(workers) is not int or type(workers) is bool or workers <= 0:
        raise ValueError("formal workers must be a positive integer")
    if (
        type(entry_script) is not str
        or not entry_script
        or entry_script.startswith("/")
        or "\x00" in entry_script
        or Path(entry_script).parts[:1] != ("scripts",)
        or any(part in {"", ".", ".."} for part in Path(entry_script).parts)
    ):
        raise ValueError("formal entry script must be an explicit scripts/ path")
    argv = [
        "env",
        "-u",
        "PYTHONPATH",
        "PYTHONNOUSERSITE=1",
        "MAMBA_ROOT_PREFIX=/home/lj/.local/share/degen-lio-micromamba",
        "/home/lj/.local/bin/micromamba",
        "run",
        "-n",
        "degen-lio-zprm-py311",
        "python",
        entry_script,
        "--manifest",
        manifest_relative,
        "--run-id",
        run_id,
        "--runtime-root",
        os.fspath(runtime),
        "--workers",
        str(workers),
        "--mode",
        selected_mode,
    ]
    return canonical_formal_command(argv).decode("utf-8").removesuffix("\n")


def _command_sidecar(payload: bytes) -> dict[str, Any]:
    return {
        "command_log_path": FORMAL_COMMAND_LOG_NAME,
        "command_log_sha256": hashlib.sha256(payload).hexdigest(),
        "command_log_size_bytes": len(payload),
        "schema_version": FORMAL_COMMAND_SHA256_SCHEMA,
    }


def _recursive_inventory(root: Path) -> tuple[list[str], list[str]]:
    top_level: list[str] = []
    violations: list[str] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            violations.append(f"cannot scan {directory}: {error.__class__.__name__}")
            return
        for entry in entries:
            path = directory / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                violations.append(
                    f"cannot stat {path}: {error.__class__.__name__}"
                )
                continue
            if stat.S_ISLNK(metadata.st_mode):
                violations.append(f"symbolic link is forbidden: {path}")
            elif stat.S_ISDIR(metadata.st_mode):
                visit(path)
            elif not stat.S_ISREG(metadata.st_mode):
                violations.append(f"special filesystem entry is forbidden: {path}")

    try:
        with os.scandir(root) as iterator:
            top_level = sorted(entry.name for entry in iterator)
    except OSError as error:
        return [], [f"cannot scan runtime root: {error.__class__.__name__}"]
    visit(root)
    return top_level, violations


def verify_formal_command_binding(
    root: str | Path,
    expected_command: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    runtime = assert_no_symlink_path(root)
    log_path = runtime / FORMAL_COMMAND_LOG_NAME
    sha_path = runtime / FORMAL_COMMAND_SHA256_NAME
    try:
        payload = read_regular_bytes(log_path)
        text = payload.decode("utf-8", errors="strict")
        if payload != canonical_formal_command(text.removesuffix("\n")):
            raise FormalCommandBindingError("formal command log is not canonical")
        sidecar = read_canonical_json(sha_path)
    except FormalCommandBindingError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError, RuntimeError) as error:
        raise FormalCommandBindingError("cannot authenticate formal command log") from error
    expected_sidecar = _command_sidecar(payload)
    if sidecar != expected_sidecar:
        raise FormalCommandBindingError("formal command SHA sidecar mismatch")
    if expected_command is not None and payload != canonical_formal_command(expected_command):
        raise FormalCommandBindingError("formal command differs from expected invocation")
    return dict(expected_sidecar)


def _validate_lock_shape(
    path: Path, command_binding: Mapping[str, Any]
) -> dict[str, Any]:
    value = read_canonical_json(path)
    if (
        type(value) is not dict
        or set(value) != {"contract", "payload_sha256", "run_lock_version"}
        or value.get("run_lock_version") != IMMUTABLE_RUN_LOCK_VERSION
        or type(value.get("contract")) is not dict
    ):
        raise InvalidFormalRuntimeError("immutable run lock schema is invalid")
    unsigned = {
        "contract": value["contract"],
        "run_lock_version": value["run_lock_version"],
    }
    if value.get("payload_sha256") != canonical_json_sha256(unsigned):
        raise InvalidFormalRuntimeError("immutable run lock SHA is invalid")
    expected_bootstrap_binding = {
        "binding_version": FORMAL_BOOTSTRAP_BINDING_VERSION,
        "command_log_path": command_binding["command_log_path"],
        "command_log_sha256": command_binding["command_log_sha256"],
        "command_log_size_bytes": command_binding["command_log_size_bytes"],
    }
    contract = value["contract"]
    if (
        contract.get("formal_bootstrap_binding")
        != expected_bootstrap_binding
        or contract.get("formal_command_sha256")
        != command_binding["command_log_sha256"]
    ):
        raise InvalidFormalRuntimeError(
            "immutable run lock does not bind the formal command"
        )
    return value


def inspect_formal_runtime(
    root: str | Path,
    *,
    expected_command: str | Sequence[str] | None = None,
    expected_base_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify the external root without following links or mutating it."""

    raw = Path(os.path.abspath(os.fspath(root)))
    reasons: list[str] = []
    try:
        runtime = assert_no_symlink_path(raw)
    except (OSError, RuntimeError) as error:
        return {
            "state": FormalRuntimeState.INVALID,
            "reasons": [str(error)],
            "top_level_entries": [],
        }
    if not os.path.lexists(runtime):
        return {
            "state": FormalRuntimeState.ABSENT,
            "reasons": [],
            "top_level_entries": [],
        }
    try:
        root_metadata = os.lstat(runtime)
    except OSError as error:
        return {
            "state": FormalRuntimeState.INVALID,
            "reasons": [f"cannot stat runtime root: {error.__class__.__name__}"],
            "top_level_entries": [],
        }
    if not stat.S_ISDIR(root_metadata.st_mode):
        return {
            "state": FormalRuntimeState.INVALID,
            "reasons": ["runtime root must be a real directory"],
            "top_level_entries": [],
        }
    entries, recursive_violations = _recursive_inventory(runtime)
    reasons.extend(recursive_violations)
    names = set(entries)
    lock_present = IMMUTABLE_RUN_LOCK_NAME in names
    allowed = _RESUMABLE_TOP_LEVEL if lock_present else _BOOTSTRAP_TOP_LEVEL
    unknown = sorted(names - allowed)
    if unknown:
        reasons.append(f"unknown top-level entries: {unknown}")
    for name in sorted(names):
        path = runtime / name
        try:
            mode = os.lstat(path).st_mode
        except OSError as error:
            reasons.append(f"cannot stat top-level entry {name}: {error.__class__.__name__}")
            continue
        expected_directory = name in _EXPECTED_DIRECTORY_NAMES
        if expected_directory and not stat.S_ISDIR(mode):
            reasons.append(f"runtime entry must be a directory: {name}")
        if not expected_directory and not stat.S_ISREG(mode):
            reasons.append(f"runtime entry must be a regular file: {name}")
    required = {FORMAL_COMMAND_LOG_NAME, FORMAL_COMMAND_SHA256_NAME}
    if not required.issubset(names):
        reasons.append("formal command log/SHA inventory is incomplete")
    command_binding: dict[str, Any] | None = None
    if not reasons:
        try:
            command_binding = verify_formal_command_binding(
                runtime, expected_command=expected_command
            )
            if lock_present:
                _validate_lock_shape(
                    runtime / IMMUTABLE_RUN_LOCK_NAME, command_binding
                )
                if expected_base_contract is not None:
                    resume_immutable_run_lock(
                        runtime / IMMUTABLE_RUN_LOCK_NAME,
                        _enhanced_contract(
                            expected_base_contract, command_binding
                        ),
                    )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            reasons.append(str(error))
    state = (
        FormalRuntimeState.INVALID
        if reasons
        else (
            FormalRuntimeState.RESUMABLE
            if lock_present
            else FormalRuntimeState.BOOTSTRAP_ONLY
        )
    )
    return {
        "command_binding": command_binding,
        "reasons": reasons,
        "state": state,
        "top_level_entries": entries,
    }


def classify_formal_runtime(
    root: str | Path,
    *,
    expected_command: str | Sequence[str] | None = None,
    expected_base_contract: Mapping[str, Any] | None = None,
) -> FormalRuntimeState:
    return inspect_formal_runtime(
        root,
        expected_command=expected_command,
        expected_base_contract=expected_base_contract,
    )["state"]


def bootstrap_formal_runtime(
    root: str | Path,
    command: str | Sequence[str],
    *,
    mode: str,
) -> dict[str, Any]:
    """Create or authenticate the command-only bootstrap boundary."""

    selected_mode = _mode(mode)
    runtime = Path(os.path.abspath(os.fspath(root)))
    payload = canonical_formal_command(command)
    with SingleWriterLease(bootstrap_lease_path(runtime)):
        before = inspect_formal_runtime(runtime)
        state = before["state"]
        if selected_mode == "resume":
            if state is not FormalRuntimeState.RESUMABLE:
                raise FormalBootstrapError(
                    f"resume bootstrap requires RESUMABLE, observed {state.value}"
                )
            binding = verify_formal_command_binding(
                runtime, expected_command=command
            )
            return {
                "command_binding": binding,
                "created": False,
                "mode": selected_mode,
                "state_after": FormalRuntimeState.RESUMABLE,
                "state_before": state,
            }
        if state is FormalRuntimeState.BOOTSTRAP_ONLY:
            binding = verify_formal_command_binding(runtime, expected_command=command)
            return {
                "command_binding": binding,
                "created": False,
                "mode": selected_mode,
                "state_after": state,
                "state_before": state,
            }
        if state is not FormalRuntimeState.ABSENT:
            raise FormalBootstrapError(
                f"fresh bootstrap requires ABSENT, observed {state.value}"
            )
        sidecar = _command_sidecar(payload)

        def populate(staging: Path) -> None:
            atomic_create_bytes(staging / FORMAL_COMMAND_LOG_NAME, payload)
            atomic_create_bytes(
                staging / FORMAL_COMMAND_SHA256_NAME,
                canonical_json_bytes(sidecar),
            )

        atomic_publish_directory(runtime, populate)
        after = inspect_formal_runtime(runtime)
        if after["state"] is not FormalRuntimeState.BOOTSTRAP_ONLY:
            raise FormalBootstrapError("fresh bootstrap did not reach BOOTSTRAP_ONLY")
        return {
            "command_binding": after["command_binding"],
            "created": True,
            "mode": selected_mode,
            "state_after": after["state"],
            "state_before": state,
        }


def _enhanced_contract(
    contract: Mapping[str, Any], command_binding: Mapping[str, Any]
) -> dict[str, Any]:
    return bind_formal_bootstrap_to_run_contract(
        contract,
        command_log_path=str(command_binding["command_log_path"]),
        command_log_sha256=str(command_binding["command_log_sha256"]),
        command_log_size_bytes=int(command_binding["command_log_size_bytes"]),
    )


def transition_bootstrap_run_lock(
    root: str | Path,
    base_contract: Mapping[str, Any],
    *,
    mode: str,
    before_lock_rename: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Under a writer lease, create or authenticate the enhanced run lock."""

    selected_mode = _mode(mode)
    runtime = Path(os.path.abspath(os.fspath(root)))
    before = inspect_formal_runtime(runtime)
    expected_state = (
        FormalRuntimeState.BOOTSTRAP_ONLY
        if selected_mode == "fresh"
        else FormalRuntimeState.RESUMABLE
    )
    if before["state"] is not expected_state:
        raise InvalidFormalRuntimeError(
            f"{selected_mode} lock transition requires {expected_state.value}; "
            f"observed {before['state'].value}: {before['reasons']}"
        )
    binding = verify_formal_command_binding(runtime)
    enhanced = _enhanced_contract(base_contract, binding)
    lock = write_once_formal_bootstrap_run_lock(
        runtime / IMMUTABLE_RUN_LOCK_NAME,
        enhanced,
        resume=(selected_mode == "resume"),
        before_rename=before_lock_rename,
    )
    if selected_mode == "fresh":
        # Re-open the official inode and validate its complete canonical bytes;
        # successful publication alone is not the seed-entry authorization.
        lock = write_once_formal_bootstrap_run_lock(
            runtime / IMMUTABLE_RUN_LOCK_NAME,
            enhanced,
            resume=True,
        )
    after = inspect_formal_runtime(runtime)
    if after["state"] is not FormalRuntimeState.RESUMABLE:
        raise InvalidFormalRuntimeError(
            f"lock transition did not reach RESUMABLE: {after['reasons']}"
        )
    return {
        "command_binding": binding,
        "enhanced_contract": enhanced,
        "is_resume": selected_mode == "resume",
        "lock": lock,
        "mode": selected_mode,
        "state_after": after["state"],
        "state_before": before["state"],
    }


def assert_seed_entry_lock(
    root: str | Path, base_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Final lock/command gate immediately before any seed-capable import."""

    runtime = Path(os.path.abspath(os.fspath(root)))
    inspection = inspect_formal_runtime(
        runtime, expected_base_contract=base_contract
    )
    if inspection["state"] is not FormalRuntimeState.RESUMABLE:
        raise InvalidFormalRuntimeError(
            f"seed entry requires RESUMABLE: {inspection['reasons']}"
        )
    binding = verify_formal_command_binding(runtime)
    enhanced = _enhanced_contract(base_contract, binding)
    lock = write_once_formal_bootstrap_run_lock(
        runtime / IMMUTABLE_RUN_LOCK_NAME, enhanced, resume=True
    )
    return {
        "command_binding": binding,
        "enhanced_contract": enhanced,
        "lock": lock,
        "seed_entry_lock_gate_pass": True,
        "state": FormalRuntimeState.RESUMABLE,
    }


__all__ = [
    "FORMAL_COMMAND_LOG_NAME",
    "FORMAL_COMMAND_SHA256_NAME",
    "FORMAL_COMMAND_SHA256_SCHEMA",
    "FORMAL_RUNTIME_ABSENT",
    "FORMAL_RUNTIME_BOOTSTRAP_ONLY",
    "FORMAL_RUNTIME_INVALID",
    "FORMAL_RUNTIME_RESUMABLE",
    "IMMUTABLE_RUN_LOCK_NAME",
    "SINGLE_WRITER_LEASE_NAME",
    "FormalBootstrapError",
    "FormalCommandBindingError",
    "FormalRuntimeState",
    "FormalRuntimeStateError",
    "InvalidFormalRuntimeError",
    "assert_seed_entry_lock",
    "bootstrap_formal_runtime",
    "bootstrap_lease_path",
    "build_formal_runner_command",
    "canonical_formal_command",
    "classify_formal_runtime",
    "formal_command_sha256",
    "formal_runtime_path_contract",
    "inspect_formal_runtime",
    "transition_bootstrap_run_lock",
    "verify_formal_command_binding",
]
