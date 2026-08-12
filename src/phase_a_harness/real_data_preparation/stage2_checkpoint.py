"""Strict append-only checkpoints for resumable Boreas Stage-2 streaming.

Checkpoint records bind every completed remote object to its immutable remote
identity, processing/GT/calibration contracts, and the resulting scientific
state transition.  The log is canonical JSONL with a SHA-256 hash chain; it is
never silently replaced or repaired.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .io import atomic_write_json, canonical_json_bytes, sha256_file


CHECKPOINT_SCHEMA = "zprm-boreas-stage2-object-checkpoint-v1"
TEMP_ROOT_SCHEMA = "zprm-boreas-stage2-temporary-root-v1"
ZERO_SHA256 = "0" * 64
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_RECORD_KINDS = frozenset({"MAP", "QUERY", "SELECTED_SOURCE"})
ALLOWED_TEMP_PURPOSES = frozenset({"tmp_download", "tmp_decode", "tmp_pcl"})


class Stage2CheckpointError(RuntimeError):
    """The checkpoint cannot be trusted for resume."""


class ChangedRemoteObjectError(Stage2CheckpointError):
    """A completed S3 object's remote identity changed."""


class OrphanCheckpointError(Stage2CheckpointError):
    """Checkpoint/state data has no authenticated parent in the current plan."""


class DuplicateCheckpointError(Stage2CheckpointError):
    """An object was appended more than once."""


@dataclass(frozen=True)
class CheckpointRecord:
    """Logical fields recorded for one completed Stage-2 object."""

    record_kind: str
    s3_key: str
    remote_size_bytes: int
    etag: str
    last_modified: str
    local_temporary_sha256: str
    processing_contract_sha256: str
    gt_sha256: str
    calibration_sha256: str
    completed_at_utc: str
    result_geometry_row_sha256: str | None = None
    map_state_transition_sha256: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "record_kind": self.record_kind,
            "s3_key": self.s3_key,
            "remote_size_bytes": self.remote_size_bytes,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "local_temporary_sha256": self.local_temporary_sha256,
            "processing_contract_sha256": self.processing_contract_sha256,
            "gt_sha256": self.gt_sha256,
            "calibration_sha256": self.calibration_sha256,
            "result_geometry_row_sha256": self.result_geometry_row_sha256,
            "map_state_transition_sha256": self.map_state_transition_sha256,
            "completed_at_utc": self.completed_at_utc,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckpointRecord":
        allowed = {
            "record_kind",
            "s3_key",
            "remote_size_bytes",
            "etag",
            "last_modified",
            "local_temporary_sha256",
            "processing_contract_sha256",
            "gt_sha256",
            "calibration_sha256",
            "result_geometry_row_sha256",
            "map_state_transition_sha256",
            "completed_at_utc",
        }
        if set(value) != allowed:
            missing = sorted(allowed - set(value))
            extra = sorted(set(value) - allowed)
            raise Stage2CheckpointError(
                f"checkpoint logical field set differs; missing={missing}, extra={extra}"
            )
        record = cls(
            record_kind=str(value["record_kind"]),
            s3_key=str(value["s3_key"]),
            remote_size_bytes=_strict_nonnegative_int(
                value["remote_size_bytes"], "remote_size_bytes"
            ),
            etag=_normalize_etag(value["etag"]),
            last_modified=str(value["last_modified"]),
            local_temporary_sha256=str(value["local_temporary_sha256"]),
            processing_contract_sha256=str(value["processing_contract_sha256"]),
            gt_sha256=str(value["gt_sha256"]),
            calibration_sha256=str(value["calibration_sha256"]),
            result_geometry_row_sha256=_optional_string(
                value["result_geometry_row_sha256"]
            ),
            map_state_transition_sha256=_optional_string(
                value["map_state_transition_sha256"]
            ),
            completed_at_utc=str(value["completed_at_utc"]),
        )
        _validate_logical_record(record)
        return record


def _strict_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stage2CheckpointError(f"{label} must be a nonnegative integer")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_etag(value: Any) -> str:
    etag = str(value).strip()
    if len(etag) >= 2 and etag[0] == etag[-1] == '"':
        etag = etag[1:-1]
    if not etag or any(character.isspace() for character in etag):
        raise Stage2CheckpointError("ETag must be non-empty and contain no whitespace")
    return etag


def _validate_sha256(value: str, label: str) -> None:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise Stage2CheckpointError(f"{label} is not a lowercase SHA-256")


def _validate_timestamp(value: str, label: str) -> None:
    if not value or not (value.endswith("Z") or "+" in value[10:]):
        raise Stage2CheckpointError(f"{label} must be an explicit UTC/offset timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Stage2CheckpointError(f"{label} is not ISO-8601: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage2CheckpointError(f"{label} lacks a timezone")


def _validate_utc_timestamp(value: str, label: str) -> None:
    _validate_timestamp(value, label)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise Stage2CheckpointError(f"{label} must use UTC")


def _validate_s3_key(value: str) -> None:
    key = PurePosixPath(value)
    if key.is_absolute() or not key.parts or any(part in {"", ".", ".."} for part in key.parts):
        raise Stage2CheckpointError(f"unsafe S3 key: {value!r}")
    if "lidar" not in key.parts or key.suffix.lower() != ".bin":
        raise Stage2CheckpointError(f"checkpoint key is not a Boreas lidar object: {value!r}")


def _validate_logical_record(record: CheckpointRecord) -> None:
    if record.record_kind not in ALLOWED_RECORD_KINDS:
        raise Stage2CheckpointError(f"invalid record kind: {record.record_kind!r}")
    _validate_s3_key(record.s3_key)
    _normalize_etag(record.etag)
    _validate_timestamp(record.last_modified, "last_modified")
    _validate_utc_timestamp(record.completed_at_utc, "completed_at_utc")
    for field in (
        "local_temporary_sha256",
        "processing_contract_sha256",
        "gt_sha256",
        "calibration_sha256",
    ):
        _validate_sha256(str(getattr(record, field)), field)
    if record.record_kind == "MAP":
        if record.result_geometry_row_sha256 is not None:
            raise Stage2CheckpointError("MAP checkpoint cannot bind a query geometry row")
        if record.map_state_transition_sha256 is None:
            raise Stage2CheckpointError("MAP checkpoint requires a map-state transition")
        _validate_sha256(record.map_state_transition_sha256, "map_state_transition_sha256")
    else:
        if record.map_state_transition_sha256 is not None:
            raise Stage2CheckpointError(f"{record.record_kind} checkpoint cannot bind map state")
        if record.result_geometry_row_sha256 is None:
            raise Stage2CheckpointError(
                f"{record.record_kind} checkpoint requires a result/canonical row digest"
            )
        _validate_sha256(record.result_geometry_row_sha256, "result_geometry_row_sha256")


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
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


def _record_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_line(value)).hexdigest()


def _logical_from_envelope(value: Mapping[str, Any]) -> CheckpointRecord:
    chain_fields = {
        "schema",
        "sequence_number",
        "previous_record_sha256",
        "record_sha256",
    }
    logical = {key: item for key, item in value.items() if key not in chain_fields}
    return CheckpointRecord.from_mapping(logical)


def _parse_log(payload: bytes) -> list[dict[str, Any]]:
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise Stage2CheckpointError("checkpoint JSONL has a truncated final record")
    records: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    seen_keys: set[str] = set()
    for index, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise Stage2CheckpointError(f"invalid checkpoint JSONL record {index}") from error
        if not isinstance(value, dict):
            raise Stage2CheckpointError(f"checkpoint record {index} is not an object")
        if _canonical_json_line(value) != line:
            raise Stage2CheckpointError(f"checkpoint record {index} is not canonical JSONL")
        if value.get("schema") != CHECKPOINT_SCHEMA:
            raise Stage2CheckpointError(f"checkpoint record {index} schema differs")
        if value.get("sequence_number") != index:
            raise Stage2CheckpointError(f"checkpoint record {index} sequence differs")
        if value.get("previous_record_sha256") != previous:
            raise Stage2CheckpointError(f"checkpoint hash chain breaks at record {index}")
        supplied_digest = value.get("record_sha256")
        if not isinstance(supplied_digest, str):
            raise Stage2CheckpointError(f"checkpoint record {index} lacks its digest")
        digest_payload = dict(value)
        del digest_payload["record_sha256"]
        actual_digest = _record_digest(digest_payload)
        if supplied_digest != actual_digest:
            raise Stage2CheckpointError(f"checkpoint record {index} digest differs")
        logical = _logical_from_envelope(value)
        if logical.s3_key in seen_keys:
            raise DuplicateCheckpointError(f"duplicate checkpoint key: {logical.s3_key}")
        seen_keys.add(logical.s3_key)
        records.append(value)
        previous = actual_digest
    return records


def _ensure_log_parent(path: Path) -> Path:
    if not path.is_absolute():
        raise Stage2CheckpointError(f"checkpoint path must be absolute: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    if parent != path.parent or path.parent.is_symlink():
        raise Stage2CheckpointError(f"checkpoint parent is unsafe: {path.parent}")
    if path.exists() and path.is_symlink():
        raise Stage2CheckpointError(f"checkpoint is a symlink: {path}")
    return path


class Stage2CheckpointLog:
    """Durable append and fail-closed resume for one JSONL checkpoint."""

    def __init__(
        self,
        path: str | Path,
        *,
        processing_contract_sha256: str | None = None,
        gt_sha256: str | None = None,
        calibration_sha256: str | None = None,
        expected_record_kinds: Iterable[str] | None = None,
    ) -> None:
        self.path = _ensure_log_parent(Path(path))
        self.expected_bindings = {
            "processing_contract_sha256": processing_contract_sha256,
            "gt_sha256": gt_sha256,
            "calibration_sha256": calibration_sha256,
        }
        for label, value in self.expected_bindings.items():
            if value is not None:
                _validate_sha256(value, label)
        self.expected_record_kinds = (
            ALLOWED_RECORD_KINDS
            if expected_record_kinds is None
            else frozenset(str(value) for value in expected_record_kinds)
        )
        if not self.expected_record_kinds or not self.expected_record_kinds <= ALLOWED_RECORD_KINDS:
            raise Stage2CheckpointError("invalid expected checkpoint record kinds")

    def _validate_bindings(self, record: CheckpointRecord) -> None:
        if record.record_kind not in self.expected_record_kinds:
            raise Stage2CheckpointError(
                f"checkpoint record kind differs from log role: {record.record_kind}"
            )
        for field, expected in self.expected_bindings.items():
            if expected is not None and getattr(record, field) != expected:
                raise Stage2CheckpointError(f"checkpoint {field} differs from resume contract")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        if not self.path.is_file() or self.path.is_symlink():
            raise Stage2CheckpointError(f"unsafe checkpoint path: {self.path}")
        records = _parse_log(self.path.read_bytes())
        for envelope in records:
            self._validate_bindings(_logical_from_envelope(envelope))
        return records

    def logical_records(self) -> list[CheckpointRecord]:
        return [_logical_from_envelope(value) for value in self.read()]

    def append_completed(
        self, record: CheckpointRecord | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Append one completed object; duplicate keys are always rejected."""

        # Round-trip dataclass instances as well, so quoted ETags and numeric
        # subclasses cannot bypass the canonical logical representation.
        logical = CheckpointRecord.from_mapping(
            record.to_mapping() if isinstance(record, CheckpointRecord) else record
        )
        _validate_logical_record(logical)
        self._validate_bindings(logical)
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as error:
            raise Stage2CheckpointError(f"cannot safely open checkpoint: {self.path}") from error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.lseek(descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            existing = _parse_log(b"".join(chunks))
            existing_logical = [_logical_from_envelope(value) for value in existing]
            for existing_record in existing_logical:
                self._validate_bindings(existing_record)
            if logical.s3_key in {value.s3_key for value in existing_logical}:
                raise DuplicateCheckpointError(
                    f"checkpoint already contains object: {logical.s3_key}"
                )
            previous = existing[-1]["record_sha256"] if existing else ZERO_SHA256
            envelope = {
                "schema": CHECKPOINT_SCHEMA,
                "sequence_number": len(existing) + 1,
                "previous_record_sha256": previous,
                **logical.to_mapping(),
            }
            envelope["record_sha256"] = _record_digest(envelope)
            line = _canonical_json_line(envelope)
            view = memoryview(line)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("zero-byte checkpoint append")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        directory_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return envelope

    def resume_plan(
        self,
        remote_inventory: Sequence[Mapping[str, Any]],
        *,
        current_map_state_sha256: str | None | object = ...,
        temporary_roots: Iterable[str | Path] = (),
    ) -> dict[str, Any]:
        """Clean marked temp roots and validate completed objects against S3 metadata.

        ``current_map_state_sha256`` is optional.  When explicitly provided,
        the latest MAP transition must match it exactly; ``None`` represents
        an absent map state and therefore rejects a checkpoint that claims one.
        """

        cleanup_reports = [cleanup_partial_temporaries(path) for path in temporary_roots]
        inventory: dict[str, dict[str, Any]] = {}
        ordered_keys: list[str] = []
        for raw in remote_inventory:
            if not isinstance(raw, Mapping):
                raise Stage2CheckpointError("remote inventory row is not an object")
            key = str(raw.get("s3_key", raw.get("key", "")))
            _validate_s3_key(key)
            if key in inventory:
                raise Stage2CheckpointError(f"duplicate remote inventory key: {key}")
            size_value = raw.get("remote_size_bytes", raw.get("size_bytes"))
            size = _strict_nonnegative_int(size_value, "remote inventory size")
            etag = _normalize_etag(raw.get("etag", ""))
            last_modified = str(raw.get("last_modified", ""))
            _validate_timestamp(last_modified, "remote inventory last_modified")
            inventory[key] = {
                "s3_key": key,
                "remote_size_bytes": size,
                "etag": etag,
                "last_modified": last_modified,
            }
            ordered_keys.append(key)

        completed = self.logical_records()
        for record in completed:
            current = inventory.get(record.s3_key)
            if current is None:
                raise OrphanCheckpointError(
                    f"checkpoint object is absent from frozen inventory: {record.s3_key}"
                )
            changed = {
                field: (getattr(record, field), current[field])
                for field in ("remote_size_bytes", "etag", "last_modified")
                if getattr(record, field) != current[field]
            }
            if changed:
                raise ChangedRemoteObjectError(
                    f"completed S3 object identity changed: {record.s3_key}: {changed}"
                )

        latest_map = next(
            (record for record in reversed(completed) if record.record_kind == "MAP"), None
        )
        if current_map_state_sha256 is not ...:
            if current_map_state_sha256 is None:
                if latest_map is not None:
                    raise OrphanCheckpointError(
                        "MAP checkpoints exist but the bound accumulator state is absent"
                    )
            else:
                state_sha = str(current_map_state_sha256)
                _validate_sha256(state_sha, "current_map_state_sha256")
                if latest_map is None:
                    raise OrphanCheckpointError(
                        "an accumulator state exists without a completed MAP checkpoint"
                    )
                if latest_map.map_state_transition_sha256 != state_sha:
                    raise OrphanCheckpointError(
                        "accumulator state is not the latest authenticated MAP transition"
                    )

        completed_keys = {record.s3_key for record in completed}
        return {
            "pass": True,
            "completed_object_count": len(completed_keys),
            "pending_object_count": len(inventory) - len(completed_keys),
            "completed_keys": [key for key in ordered_keys if key in completed_keys],
            "pending_keys": [key for key in ordered_keys if key not in completed_keys],
            "checkpoint_sha256": sha256_file(self.path) if self.path.exists() else None,
            "latest_map_state_transition_sha256": (
                latest_map.map_state_transition_sha256 if latest_map is not None else None
            ),
            "temporary_cleanup": cleanup_reports,
        }


def initialize_temporary_root(path: str | Path, *, purpose: str) -> Path:
    """Mark an isolated Stage-2 temp root before any disposable data is written."""

    if purpose not in ALLOWED_TEMP_PURPOSES:
        raise Stage2CheckpointError(f"invalid temporary-root purpose: {purpose!r}")
    root = Path(path)
    if not root.is_absolute():
        raise Stage2CheckpointError(f"temporary root must be absolute: {root}")
    existed = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise Stage2CheckpointError(f"unsafe temporary root: {root}")
    marker = {
        "schema": TEMP_ROOT_SCHEMA,
        "temporary": True,
        "purpose": purpose,
        "canonical_root": str(root),
    }
    marker_path = root / ".stage2_temporary_root.json"
    existing_names = {item.name for item in root.iterdir()}
    if existed and existing_names and existing_names != {marker_path.name}:
        raise Stage2CheckpointError(
            f"refusing to mark a nonempty directory as temporary: {root}"
        )
    atomic_write_json(marker_path, marker, overwrite=False)
    return root


def _assert_cleanup_tree_safe(root: Path) -> list[Path]:
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    marker = root / ".stage2_temporary_root.json"
    for path in entries:
        if path == marker:
            continue
        for candidate in (path, *path.rglob("*")) if path.is_dir() else (path,):
            metadata = os.lstat(candidate)
            if stat.S_ISLNK(metadata.st_mode):
                raise Stage2CheckpointError(
                    f"refusing to clean temporary tree containing a symlink: {candidate}"
                )
            if not (
                stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)
            ):
                raise Stage2CheckpointError(
                    f"refusing to clean special temporary entry: {candidate}"
                )
    return [path for path in entries if path != marker]


def cleanup_partial_temporaries(path: str | Path) -> dict[str, Any]:
    """Clear only an explicitly marked, canonical Stage-2 temporary directory."""

    root = Path(path)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise Stage2CheckpointError(f"unsafe temporary cleanup root: {root}")
    if root.resolve(strict=True) != root:
        raise Stage2CheckpointError(f"temporary cleanup root is not canonical: {root}")
    marker_path = root / ".stage2_temporary_root.json"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise Stage2CheckpointError(f"unmarked temporary cleanup root: {root}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage2CheckpointError(
            f"temporary-root marker is invalid JSON: {root}"
        ) from exc
    expected_marker_keys = {"canonical_root", "purpose", "schema", "temporary"}
    if not isinstance(marker, dict) or set(marker) != expected_marker_keys:
        raise Stage2CheckpointError(f"temporary-root marker field set differs: {root}")
    if canonical_json_bytes(marker) != marker_path.read_bytes():
        raise Stage2CheckpointError(f"temporary-root marker is not canonical: {root}")
    if (
        marker.get("schema") != TEMP_ROOT_SCHEMA
        or marker.get("temporary") is not True
        or marker.get("purpose") not in ALLOWED_TEMP_PURPOSES
        or marker.get("canonical_root") != str(root)
    ):
        raise Stage2CheckpointError(f"temporary-root marker differs: {root}")
    targets = _assert_cleanup_tree_safe(root)
    removed_file_count = sum(
        1
        for target in targets
        for candidate in ((target, *target.rglob("*")) if target.is_dir() else (target,))
        if candidate.is_file()
    )
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "pass": True,
        "temporary_root": str(root),
        "purpose": marker["purpose"],
        "removed_entry_count": len(targets),
        "removed_file_count": removed_file_count,
        "remaining_non_marker_count": 0,
    }


__all__ = [
    "CheckpointRecord",
    "ChangedRemoteObjectError",
    "DuplicateCheckpointError",
    "OrphanCheckpointError",
    "Stage2CheckpointError",
    "Stage2CheckpointLog",
    "cleanup_partial_temporaries",
    "initialize_temporary_root",
]
