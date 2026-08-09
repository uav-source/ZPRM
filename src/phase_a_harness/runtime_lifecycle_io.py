"""Durable, seed-agnostic I/O primitives for runtime lifecycle control.

This module deliberately knows nothing about scientific plans, snapshots, or
backend parameters.  It provides the small set of filesystem primitives needed
by a higher-level fresh/resume runner:

* strict canonical JSON;
* durable create-without-clobber and replace writes;
* all-at-once directory publication;
* a kernel-backed single-writer lease;
* an immutable, content-addressed run lock; and
* a hash-chained append-only lifecycle event journal.

The APIs reject symbolic links in controlled paths.  They are intended for a
local POSIX filesystem; callers must hold :class:`SingleWriterLease` around a
complete run when several files form one logical transaction.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping


IMMUTABLE_RUN_LOCK_VERSION = "runtime_lifecycle_run_lock_v1"
FORMAL_BOOTSTRAP_BINDING_VERSION = "formal_runtime_bootstrap_binding_v1"
EVENT_VERSION_V2 = "runtime_lifecycle_event_v2"
EVENT_TYPES_V2 = frozenset(
    {
        "STARTED",
        "COMPLETED",
        "INFRASTRUCTURE_INTERRUPTION",
        "RESUMED",
        "SKIPPED_VALID_RESULT",
        "REJECTED_CORRUPT_RESULT",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_FIELDS = frozenset(
    {
        "event_version",
        "run_id",
        "invocation_id",
        "sequence",
        "event_type",
        "planned_trial_id",
        "backend",
        "signal_number",
        "result_sha256",
        "previous_event_sha256",
        "event_sha256",
    }
)
_RUN_LOCK_FIELDS = frozenset(
    {"run_lock_version", "contract", "payload_sha256"}
)


class RuntimeLifecycleIOError(RuntimeError):
    """Base class for lifecycle I/O contract failures."""


class SymlinkPathError(RuntimeLifecycleIOError):
    """Raised when a controlled path contains a symbolic link."""


class LeaseUnavailableError(RuntimeLifecycleIOError):
    """Raised when another writer currently holds the run lease."""


class ImmutableRunLockError(RuntimeLifecycleIOError):
    """Raised when an immutable run lock is invalid or does not bind."""


class EventJournalError(RuntimeLifecycleIOError):
    """Raised when an event journal is malformed or hash-chain-invalid."""


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number is forbidden: {token}")
    return value


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def strict_json_loads(payload: str | bytes) -> Any:
    """Decode strict UTF-8 JSON, rejecting duplicate keys and non-finites."""

    if isinstance(payload, bytes):
        text = payload.decode("utf-8", errors="strict")
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError("JSON payload must be str or bytes")
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_float,
    )


def _validate_json_native(value: Any, *, location: str = "$") -> None:
    """Reject Python conveniences that do not have an exact JSON identity."""

    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {location}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_native(item, location=f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"JSON object key is not a string at {location}")
            _validate_json_native(item, location=f"{location}.{key}")
        return
    raise TypeError(f"value is not an exact JSON type at {location}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one accepted UTF-8 representation of a JSON value."""

    _validate_json_native(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _assert_no_symlink_components(path: str | Path) -> Path:
    """Return a lexical absolute path after rejecting each existing symlink."""

    candidate = _lexical_absolute(path)
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise SymlinkPathError(f"symbolic links are forbidden: {current}")
    return candidate


def assert_no_symlink_path(path: str | Path) -> Path:
    """Public fail-closed path normalization for lifecycle state machines."""

    return _assert_no_symlink_components(path)


def _ensure_parent_directory(path: str | Path) -> Path:
    destination = _assert_no_symlink_components(path)
    parent = _assert_no_symlink_components(destination.parent)
    parent.mkdir(parents=True, exist_ok=True)
    parent = _assert_no_symlink_components(parent)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    return destination


def _fsync_directory(directory: str | Path) -> None:
    path = _assert_no_symlink_components(directory)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("zero-byte write while committing lifecycle data")
        view = view[written:]


def _write_temporary_file(destination: Path, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    return temporary


def _reject_existing_symlink(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise SymlinkPathError(f"symbolic-link destination is forbidden: {path}")


def atomic_create_bytes(path: str | Path, payload: bytes) -> Path:
    """Durably create ``path`` without ever replacing an existing inode.

    A same-directory temporary file is fsynced and then hard-linked into place.
    The hard-link operation is an atomic no-clobber commit on POSIX filesystems.
    """

    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    destination = _ensure_parent_directory(path)
    _reject_existing_symlink(destination)
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite: {destination}")
    temporary = _write_temporary_file(destination, payload)
    try:
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            _reject_existing_symlink(destination)
            raise FileExistsError(f"refusing to overwrite: {destination}") from None
        temporary.unlink()
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_create_bytes_rename_noreplace(
    path: str | Path,
    payload: bytes,
    *,
    before_rename: Callable[[Path, Path], None] | None = None,
) -> Path:
    """Durably create a file through same-directory rename-no-replace.

    This specialized transition is used for the formal bootstrap lock.  The
    optional callback runs after the temporary inode has been fully written and
    fsynced but before publication, allowing interruption tests to prove that a
    failed transition leaves no official lock.
    """

    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    if before_rename is not None and not callable(before_rename):
        raise TypeError("before_rename must be callable or None")
    destination = _ensure_parent_directory(path)
    _reject_existing_symlink(destination)
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite: {destination}")
    temporary = _write_temporary_file(destination, payload)
    try:
        if before_rename is not None:
            before_rename(temporary, destination)
        _reject_existing_symlink(destination)
        _rename_noreplace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_replace_bytes(path: str | Path, payload: bytes) -> Path:
    """Durably replace a regular destination through a sibling temporary file."""

    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    destination = _ensure_parent_directory(path)
    _reject_existing_symlink(destination)
    temporary = _write_temporary_file(destination, payload)
    try:
        _reject_existing_symlink(destination)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_create_canonical_json(path: str | Path, value: Any) -> Path:
    return atomic_create_bytes(path, canonical_json_bytes(value))


def atomic_replace_canonical_json(path: str | Path, value: Any) -> Path:
    return atomic_replace_bytes(path, canonical_json_bytes(value))


def _open_regular_readonly(path: str | Path) -> tuple[Path, int]:
    source = _assert_no_symlink_components(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise SymlinkPathError(f"symbolic-link file is forbidden: {source}") from error
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RuntimeLifecycleIOError(f"regular file required: {source}")
    return source, descriptor


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def read_regular_bytes(path: str | Path) -> bytes:
    """Read one regular file without following any symbolic-link component."""

    _source, descriptor = _open_regular_readonly(path)
    try:
        return _read_all(descriptor)
    finally:
        os.close(descriptor)


def read_canonical_json(path: str | Path) -> Any:
    """Read a regular non-symlink file and require canonical bytes exactly."""

    source, descriptor = _open_regular_readonly(path)
    try:
        payload = _read_all(descriptor)
    finally:
        os.close(descriptor)
    try:
        value = strict_json_loads(payload)
        if payload != canonical_json_bytes(value):
            raise ValueError("JSON bytes are not canonical")
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise RuntimeLifecycleIOError(f"invalid canonical JSON: {source}") from error
    return value


def _fsync_tree(directory: Path) -> None:
    """Reject links/special files and fsync all regular files and directories."""

    root = _assert_no_symlink_components(directory)
    metadata = os.lstat(root)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeLifecycleIOError(f"publication staging root is not a directory: {root}")

    def visit(current: Path) -> None:
        with os.scandir(current) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            entry_path = current / entry.name
            entry_metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(entry_metadata.st_mode):
                raise SymlinkPathError(
                    f"publication tree contains symbolic link: {entry_path}"
                )
            if stat.S_ISDIR(entry_metadata.st_mode):
                visit(entry_path)
            elif stat.S_ISREG(entry_metadata.st_mode):
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(entry_path, flags)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            else:
                raise RuntimeLifecycleIOError(
                    f"publication tree contains non-regular entry: {entry_path}"
                )
        _fsync_directory(current)

    visit(root)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Use Linux renameat2 when available, retaining a conservative fallback."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100, os.fsencode(source), -100, os.fsencode(destination), 1
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"refusing to overwrite: {destination}")
        if error_number not in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
            raise OSError(error_number, os.strerror(error_number), destination)

    # This fallback is used only where renameat2 is unavailable.  Callers must
    # hold the run lease to exclude another cooperating writer.
    _reject_existing_symlink(destination)
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite: {destination}")
    os.rename(source, destination)


def atomic_publish_directory(
    path: str | Path, populate: Callable[[Path], None]
) -> Path:
    """Populate a sibling staging tree and publish it as one no-clobber rename."""

    if not callable(populate):
        raise TypeError("populate must be callable")
    destination = _ensure_parent_directory(path)
    _reject_existing_symlink(destination)
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite: {destination}")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )
    staging = temporary / "payload"
    staging.mkdir()
    try:
        populate(staging)
        if not staging.is_dir() or staging.is_symlink():
            raise RuntimeLifecycleIOError("publication callback replaced staging root")
        _fsync_tree(staging)
        _reject_existing_symlink(destination)
        _rename_noreplace(staging, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


class SingleWriterLease:
    """Non-blocking kernel lease for one lifecycle writer.

    The lock file is intentionally persistent.  The kernel releases the lease
    when the descriptor is closed or the process exits, including after SIGKILL.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = _lexical_absolute(path)
        self._descriptor: int | None = None

    @property
    def acquired(self) -> bool:
        return self._descriptor is not None

    def acquire(self) -> "SingleWriterLease":
        if self._descriptor is not None:
            raise RuntimeLifecycleIOError("lease object is already acquired")
        path = _ensure_parent_directory(self.path)
        _reject_existing_symlink(path)
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise SymlinkPathError(f"symbolic-link lease is forbidden: {path}") from error
            raise
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise RuntimeLifecycleIOError(f"lease path must be a regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(descriptor)
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise LeaseUnavailableError(f"run lease is already held: {path}") from error
            raise
        self._descriptor = descriptor
        _fsync_directory(path.parent)
        return self

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "SingleWriterLease":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def _normalize_json_object(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        normalized = strict_json_loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if type(normalized) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return normalized


def build_immutable_run_lock(contract: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_json_object(contract, name="run contract")
    unsigned = {
        "contract": normalized,
        "run_lock_version": IMMUTABLE_RUN_LOCK_VERSION,
    }
    return {**unsigned, "payload_sha256": canonical_json_sha256(unsigned)}


def bind_formal_bootstrap_to_run_contract(
    contract: Mapping[str, Any],
    *,
    command_log_path: str,
    command_log_sha256: str,
    command_log_size_bytes: int,
) -> dict[str, Any]:
    """Add the immutable formal-command identity to a scientific run contract.

    The outer immutable-lock schema remains backward compatible.  The enhanced
    contract is nevertheless cryptographically distinct because the command
    log path, byte length, and SHA-256 become part of the lock payload.
    """

    normalized = _normalize_json_object(contract, name="run contract")
    if (
        "formal_bootstrap_binding" in normalized
        or "formal_command_sha256" in normalized
    ):
        raise ImmutableRunLockError(
            "run contract already contains a formal bootstrap binding"
        )
    if (
        type(command_log_path) is not str
        or not command_log_path
        or Path(command_log_path).name != command_log_path
        or type(command_log_sha256) is not str
        or _SHA256_RE.fullmatch(command_log_sha256) is None
        or type(command_log_size_bytes) is not int
        or type(command_log_size_bytes) is bool
        or command_log_size_bytes <= 0
    ):
        raise ImmutableRunLockError("formal bootstrap command binding is invalid")
    return {
        **normalized,
        "formal_command_sha256": command_log_sha256,
        "formal_bootstrap_binding": {
            "binding_version": FORMAL_BOOTSTRAP_BINDING_VERSION,
            "command_log_path": command_log_path,
            "command_log_sha256": command_log_sha256,
            "command_log_size_bytes": command_log_size_bytes,
        },
    }


def _validate_immutable_run_lock(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _RUN_LOCK_FIELDS:
        raise ImmutableRunLockError("immutable run lock schema mismatch")
    if value.get("run_lock_version") != IMMUTABLE_RUN_LOCK_VERSION:
        raise ImmutableRunLockError("immutable run lock version mismatch")
    if type(value.get("contract")) is not dict:
        raise ImmutableRunLockError("immutable run lock contract must be an object")
    recorded_sha = value.get("payload_sha256")
    if type(recorded_sha) is not str or not _SHA256_RE.fullmatch(recorded_sha):
        raise ImmutableRunLockError("immutable run lock payload SHA is invalid")
    unsigned = {
        "contract": value["contract"],
        "run_lock_version": value["run_lock_version"],
    }
    if canonical_json_sha256(unsigned) != recorded_sha:
        raise ImmutableRunLockError("immutable run lock payload SHA mismatch")
    return value


def create_immutable_run_lock(
    path: str | Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    value = build_immutable_run_lock(contract)
    atomic_create_canonical_json(path, value)
    return value


def create_formal_bootstrap_run_lock(
    path: str | Path,
    contract: Mapping[str, Any],
    *,
    before_rename: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Create the command-bound formal lock by atomic rename-no-replace."""

    value = build_immutable_run_lock(contract)
    atomic_create_bytes_rename_noreplace(
        path,
        canonical_json_bytes(value),
        before_rename=before_rename,
    )
    return value


def resume_immutable_run_lock(
    path: str | Path, expected_contract: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        value = read_canonical_json(path)
    except (RuntimeLifecycleIOError, OSError) as error:
        if isinstance(error, SymlinkPathError):
            raise
        raise ImmutableRunLockError(f"cannot read immutable run lock: {path}") from error
    validated = _validate_immutable_run_lock(value)
    expected = build_immutable_run_lock(expected_contract)
    if canonical_json_bytes(validated) != canonical_json_bytes(expected):
        raise ImmutableRunLockError("immutable run lock binds a different contract")
    return validated


def write_once_immutable_run_lock(
    path: str | Path,
    contract: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    """Create a fresh lock or strictly validate the exact lock on resume."""

    if type(resume) is not bool:
        raise TypeError("resume must be bool")
    if resume:
        return resume_immutable_run_lock(path, contract)
    return create_immutable_run_lock(path, contract)


def write_once_formal_bootstrap_run_lock(
    path: str | Path,
    contract: Mapping[str, Any],
    *,
    resume: bool,
    before_rename: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Create the formal lock by rename, or strictly authenticate it on resume."""

    if type(resume) is not bool:
        raise TypeError("resume must be bool")
    if resume:
        if before_rename is not None:
            raise ValueError("before_rename is forbidden during lock validation")
        return resume_immutable_run_lock(path, contract)
    return create_formal_bootstrap_run_lock(
        path, contract, before_rename=before_rename
    )


def _nonempty_string(value: Any) -> bool:
    return type(value) is str and bool(value) and "\x00" not in value


def _optional_nonempty_string(value: Any) -> bool:
    return value is None or _nonempty_string(value)


def _optional_sha256(value: Any) -> bool:
    return value is None or (type(value) is str and _SHA256_RE.fullmatch(value) is not None)


def _validate_event_semantics(event: Mapping[str, Any], *, line: int) -> None:
    event_type = event["event_type"]
    trial = event["planned_trial_id"]
    backend = event["backend"]
    signal_number = event["signal_number"]
    result_sha = event["result_sha256"]

    if event_type in {"STARTED", "COMPLETED", "SKIPPED_VALID_RESULT", "REJECTED_CORRUPT_RESULT"}:
        if not _nonempty_string(trial) or not _nonempty_string(backend):
            raise EventJournalError(f"trial/backend required at event line {line}")
    elif (trial is None) != (backend is None):
        raise EventJournalError(f"trial/backend must both be set or null at event line {line}")

    if event_type in {"COMPLETED", "SKIPPED_VALID_RESULT"}:
        if not (type(result_sha) is str and _SHA256_RE.fullmatch(result_sha)):
            raise EventJournalError(f"result SHA required at event line {line}")
    elif event_type != "REJECTED_CORRUPT_RESULT" and result_sha is not None:
        raise EventJournalError(f"result SHA is forbidden at event line {line}")

    if event_type == "INFRASTRUCTURE_INTERRUPTION":
        if type(signal_number) is not int or type(signal_number) is bool or signal_number <= 0:
            raise EventJournalError(f"positive signal required at event line {line}")
    elif signal_number is not None:
        raise EventJournalError(f"signal is forbidden at event line {line}")


def _validate_event_shape(event: Any, *, line: int) -> dict[str, Any]:
    if type(event) is not dict or set(event) != _EVENT_FIELDS:
        raise EventJournalError(f"event schema mismatch at line {line}")
    if event["event_version"] != EVENT_VERSION_V2:
        raise EventJournalError(f"event version mismatch at line {line}")
    if not _nonempty_string(event["run_id"]):
        raise EventJournalError(f"run ID is invalid at line {line}")
    if not _nonempty_string(event["invocation_id"]):
        raise EventJournalError(f"invocation ID is invalid at line {line}")
    sequence = event["sequence"]
    if type(sequence) is not int or type(sequence) is bool or sequence <= 0:
        raise EventJournalError(f"sequence is invalid at line {line}")
    if event["event_type"] not in EVENT_TYPES_V2:
        raise EventJournalError(f"event type is invalid at line {line}")
    if not _optional_nonempty_string(event["planned_trial_id"]):
        raise EventJournalError(f"trial ID is invalid at line {line}")
    if not _optional_nonempty_string(event["backend"]):
        raise EventJournalError(f"backend is invalid at line {line}")
    if not _optional_sha256(event["result_sha256"]):
        raise EventJournalError(f"result SHA is invalid at line {line}")
    if not _optional_sha256(event["previous_event_sha256"]):
        raise EventJournalError(f"previous event SHA is invalid at line {line}")
    if type(event["event_sha256"]) is not str or not _SHA256_RE.fullmatch(event["event_sha256"]):
        raise EventJournalError(f"event SHA is invalid at line {line}")
    _validate_event_semantics(event, line=line)
    return event


def _decode_event_journal(payload: bytes, *, source: Path) -> list[dict[str, Any]]:
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise EventJournalError(f"torn final event line: {source}")
    lines = payload.splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    run_id: str | None = None
    previous_sha: str | None = None
    for line_number, raw_line in enumerate(lines, 1):
        if raw_line == b"\n":
            raise EventJournalError(f"blank event line {line_number}")
        try:
            event = strict_json_loads(raw_line)
        except (UnicodeDecodeError, ValueError, TypeError) as error:
            raise EventJournalError(f"invalid event JSON at line {line_number}") from error
        event = _validate_event_shape(event, line=line_number)
        if raw_line != canonical_json_bytes(event):
            raise EventJournalError(f"noncanonical event JSON at line {line_number}")
        if event["sequence"] != line_number:
            raise EventJournalError(f"non-contiguous event sequence at line {line_number}")
        if run_id is None:
            run_id = event["run_id"]
        elif event["run_id"] != run_id:
            raise EventJournalError(f"run ID changed at line {line_number}")
        if event["previous_event_sha256"] != previous_sha:
            raise EventJournalError(f"event hash-chain predecessor mismatch at line {line_number}")
        unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
        computed_sha = canonical_json_sha256(unsigned)
        if event["event_sha256"] != computed_sha:
            raise EventJournalError(f"event SHA mismatch at line {line_number}")
        previous_sha = computed_sha
        events.append(event)
    return events


def read_event_journal_v2(
    path: str | Path, *, expected_run_id: str | None = None
) -> list[dict[str, Any]]:
    source = _assert_no_symlink_components(path)
    if not _lexists(source):
        return []
    source, descriptor = _open_regular_readonly(source)
    try:
        payload = _read_all(descriptor)
    finally:
        os.close(descriptor)
    events = _decode_event_journal(payload, source=source)
    if expected_run_id is not None:
        if not _nonempty_string(expected_run_id):
            raise ValueError("expected_run_id must be a non-empty string")
        if events and events[0]["run_id"] != expected_run_id:
            raise EventJournalError("event journal belongs to a different run")
    return events


def append_event_v2(
    path: str | Path,
    *,
    run_id: str,
    invocation_id: str,
    event_type: str,
    planned_trial_id: str | None = None,
    backend: str | None = None,
    signal_number: int | None = None,
    result_sha256: str | None = None,
    sequence: int | None = None,
    previous_event_sha256: str | None = None,
) -> dict[str, Any]:
    """Append one fsynced v2 event after validating the complete prior chain."""

    if not _nonempty_string(run_id) or not _nonempty_string(invocation_id):
        raise ValueError("run_id and invocation_id must be non-empty strings")
    # Validate all caller-controlled semantic fields before creating a journal
    # inode.  Sequence and predecessor are checked again against the locked tail.
    probe_sequence = sequence if sequence is not None else 1
    probe_unsigned = {
        "backend": backend,
        "event_type": event_type,
        "event_version": EVENT_VERSION_V2,
        "invocation_id": invocation_id,
        "planned_trial_id": planned_trial_id,
        "previous_event_sha256": previous_event_sha256,
        "result_sha256": result_sha256,
        "run_id": run_id,
        "sequence": probe_sequence,
        "signal_number": signal_number,
    }
    _validate_event_shape(
        {**probe_unsigned, "event_sha256": canonical_json_sha256(probe_unsigned)},
        line=probe_sequence if type(probe_sequence) is int else 0,
    )
    destination = _ensure_parent_directory(path)
    _reject_existing_symlink(destination)
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(destination, flags, 0o600)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise SymlinkPathError(
                f"symbolic-link event journal is forbidden: {destination}"
            ) from error
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EventJournalError("event journal must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        events = _decode_event_journal(_read_all(descriptor), source=destination)
        if events and events[0]["run_id"] != run_id:
            raise EventJournalError("refusing to append event for a different run")
        next_sequence = len(events) + 1
        predecessor = events[-1]["event_sha256"] if events else None
        if sequence is not None and sequence != next_sequence:
            raise EventJournalError("requested event sequence does not match journal tail")
        if previous_event_sha256 is not None and previous_event_sha256 != predecessor:
            raise EventJournalError("requested previous event SHA does not match journal tail")
        unsigned = {
            "backend": backend,
            "event_type": event_type,
            "event_version": EVENT_VERSION_V2,
            "invocation_id": invocation_id,
            "planned_trial_id": planned_trial_id,
            "previous_event_sha256": predecessor,
            "result_sha256": result_sha256,
            "run_id": run_id,
            "sequence": next_sequence,
            "signal_number": signal_number,
        }
        event = {**unsigned, "event_sha256": canonical_json_sha256(unsigned)}
        _validate_event_shape(event, line=next_sequence)
        _write_all(descriptor, canonical_json_bytes(event))
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    _fsync_directory(destination.parent)
    return event


__all__ = [
    "EVENT_TYPES_V2",
    "EVENT_VERSION_V2",
    "FORMAL_BOOTSTRAP_BINDING_VERSION",
    "IMMUTABLE_RUN_LOCK_VERSION",
    "EventJournalError",
    "ImmutableRunLockError",
    "LeaseUnavailableError",
    "RuntimeLifecycleIOError",
    "SingleWriterLease",
    "SymlinkPathError",
    "append_event_v2",
    "assert_no_symlink_path",
    "atomic_create_bytes",
    "atomic_create_bytes_rename_noreplace",
    "atomic_create_canonical_json",
    "atomic_publish_directory",
    "atomic_replace_bytes",
    "atomic_replace_canonical_json",
    "build_immutable_run_lock",
    "bind_formal_bootstrap_to_run_contract",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "create_immutable_run_lock",
    "create_formal_bootstrap_run_lock",
    "read_canonical_json",
    "read_regular_bytes",
    "read_event_journal_v2",
    "resume_immutable_run_lock",
    "strict_json_loads",
    "write_once_immutable_run_lock",
    "write_once_formal_bootstrap_run_lock",
]
