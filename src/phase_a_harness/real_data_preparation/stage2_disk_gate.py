"""Live, fail-closed disk gates for Boreas Stage-2 production preparation.

The storage planner computes thresholds; this module enforces them against a
fresh filesystem reading.  Every start/runtime decision is appended to a
canonical, fsynced SHA-256 hash chain.  It deliberately knows nothing about
preprocessing or registration semantics.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .io import canonical_json_bytes, sha256_file


DISK_GATE_EVENT_SCHEMA = "zprm.boreas.stage2.disk_gate_event.v1"
DISK_GATE_CONFIG_SCHEMA = "zprm.boreas.stage2.disk_gate_config.v1"
ZERO_SHA256 = "0" * 64
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_OPERATIONS = frozenset(
    {"BEFORE_DOWNLOAD", "BEFORE_MATERIALIZATION", "BEFORE_CHECKPOINT"}
)


class Stage2DiskGateError(RuntimeError):
    """A disk threshold or its audit chain could not be trusted."""


class InsufficientLiveDiskError(Stage2DiskGateError):
    """A fresh disk reading failed the frozen start or runtime threshold."""


def _strict_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stage2DiskGateError(f"{field} must be a nonnegative integer")
    return value


def _canonical_line(value: Mapping[str, Any]) -> bytes:
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


def _event_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_line(value)).hexdigest()


def _validate_operation_id(value: str) -> str:
    text = str(value)
    if not text or len(text.encode("utf-8")) > 4096 or any(
        character in text for character in ("\x00", "\r", "\n")
    ):
        raise Stage2DiskGateError("operation identifier is empty or unsafe")
    return text


def _parse_event_log(payload: bytes, *, expected_config_sha256: str) -> list[dict[str, Any]]:
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise Stage2DiskGateError("disk-gate audit JSONL is truncated")
    rows: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    required = {
        "current_free_bytes",
        "event_sha256",
        "gate_config_sha256",
        "operation",
        "operation_id",
        "pass",
        "previous_event_sha256",
        "projected_remaining_free_bytes",
        "projected_write_bytes",
        "schema",
        "sequence_number",
        "threshold_bytes",
        "timestamp_utc",
    }
    for index, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise Stage2DiskGateError("disk-gate audit contains invalid JSONL") from exc
        if not isinstance(value, dict) or _canonical_line(value) != line:
            raise Stage2DiskGateError("disk-gate audit is not canonical JSONL")
        if set(value) != required:
            raise Stage2DiskGateError("disk-gate audit event field set differs")
        if value["schema"] != DISK_GATE_EVENT_SCHEMA:
            raise Stage2DiskGateError("disk-gate audit event schema differs")
        if value["sequence_number"] != index:
            raise Stage2DiskGateError("disk-gate audit sequence is non-contiguous")
        if value["previous_event_sha256"] != previous:
            raise Stage2DiskGateError("disk-gate audit hash chain breaks")
        if value["gate_config_sha256"] != expected_config_sha256:
            raise Stage2DiskGateError("disk-gate audit was produced under another config")
        if value["operation"] not in RUNTIME_OPERATIONS | {"START", "RESUME"}:
            raise Stage2DiskGateError("disk-gate audit operation differs")
        _validate_operation_id(value["operation_id"])
        current = _strict_nonnegative_int(
            value["current_free_bytes"], field="audit current_free_bytes"
        )
        projected = _strict_nonnegative_int(
            value["projected_write_bytes"], field="audit projected_write_bytes"
        )
        threshold = _strict_nonnegative_int(
            value["threshold_bytes"], field="audit threshold_bytes"
        )
        remaining = value["projected_remaining_free_bytes"]
        if isinstance(remaining, bool) or not isinstance(remaining, int):
            raise Stage2DiskGateError(
                "audit projected_remaining_free_bytes must be an integer"
            )
        if remaining != current - projected:
            raise Stage2DiskGateError("disk-gate audit remaining-byte arithmetic differs")
        if not isinstance(value["pass"], bool) or value["pass"] != (remaining >= threshold):
            raise Stage2DiskGateError("disk-gate audit decision arithmetic differs")
        try:
            timestamp = datetime.fromisoformat(
                str(value["timestamp_utc"]).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise Stage2DiskGateError("disk-gate audit timestamp is invalid") from exc
        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
            or timestamp.utcoffset().total_seconds() != 0
        ):
            raise Stage2DiskGateError("disk-gate audit timestamp must be UTC")
        digest_payload = dict(value)
        supplied = digest_payload.pop("event_sha256")
        actual = _event_digest(digest_payload)
        if supplied != actual:
            raise Stage2DiskGateError("disk-gate audit event digest differs")
        rows.append(value)
        previous = actual
    return rows


@dataclass(frozen=True)
class DiskGateThresholds:
    minimum_start_free_bytes: int
    runtime_low_disk_watermark_bytes: int
    storage_mode: str
    storage_budget_sha256: str

    def __post_init__(self) -> None:
        _strict_nonnegative_int(
            self.minimum_start_free_bytes, field="minimum_start_free_bytes"
        )
        _strict_nonnegative_int(
            self.runtime_low_disk_watermark_bytes,
            field="runtime_low_disk_watermark_bytes",
        )
        if not self.storage_mode:
            raise Stage2DiskGateError("storage_mode must be nonempty")
        if SHA256_PATTERN.fullmatch(self.storage_budget_sha256) is None:
            raise Stage2DiskGateError("storage_budget_sha256 must be lowercase SHA-256")


FreeBytesProvider = Callable[[Path], int]


def _live_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


class Stage2DiskGate:
    """Mandatory start/runtime gates backed by fresh free-space readings."""

    def __init__(
        self,
        monitored_path: str | Path,
        *,
        thresholds: DiskGateThresholds,
        audit_log_path: str | Path,
        free_bytes_provider: FreeBytesProvider = _live_free_bytes,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        monitored = Path(monitored_path)
        if (
            not monitored.is_absolute()
            or monitored.is_symlink()
            or not monitored.is_dir()
            or monitored.resolve(strict=True) != monitored
        ):
            raise Stage2DiskGateError("monitored disk path must be an existing canonical directory")
        audit = Path(audit_log_path)
        if not audit.is_absolute() or not audit.parent.is_dir():
            raise Stage2DiskGateError("disk-gate audit parent must already exist")
        if audit.parent.is_symlink() or audit.parent.resolve(strict=True) != audit.parent:
            raise Stage2DiskGateError("disk-gate audit parent is unsafe")
        if audit.exists() and (audit.is_symlink() or not audit.is_file()):
            raise Stage2DiskGateError("disk-gate audit path is unsafe")
        if not isinstance(thresholds, DiskGateThresholds):
            raise Stage2DiskGateError("thresholds must be a DiskGateThresholds instance")
        self.monitored_path = monitored
        self.thresholds = thresholds
        self.audit_log_path = audit
        self._free_bytes_provider = free_bytes_provider
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._started = False
        config = {
            "minimum_start_free_bytes": thresholds.minimum_start_free_bytes,
            "monitored_path": str(monitored),
            "runtime_low_disk_watermark_bytes": (
                thresholds.runtime_low_disk_watermark_bytes
            ),
            "schema": DISK_GATE_CONFIG_SCHEMA,
            "storage_budget_sha256": thresholds.storage_budget_sha256,
            "storage_mode": thresholds.storage_mode,
        }
        self.gate_config_sha256 = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
        self.verify_audit()

    @classmethod
    def from_frozen_budget(
        cls,
        monitored_path: str | Path,
        *,
        budget_path: str | Path,
        expected_budget_sha256: str,
        audit_log_path: str | Path,
        storage_mode: str = "RECOMMENDED_OPERATIONAL",
        free_bytes_provider: FreeBytesProvider = _live_free_bytes,
        now: Callable[[], datetime] | None = None,
    ) -> "Stage2DiskGate":
        source = Path(budget_path)
        expected = str(expected_budget_sha256).lower()
        if SHA256_PATTERN.fullmatch(expected) is None:
            raise Stage2DiskGateError("expected budget SHA must be lowercase SHA-256")
        if (
            not source.is_absolute()
            or source.is_symlink()
            or not source.is_file()
            or source.resolve(strict=True) != source
        ):
            raise Stage2DiskGateError("frozen disk budget path is unsafe")
        if sha256_file(source) != expected:
            raise Stage2DiskGateError("frozen disk budget SHA-256 differs")
        try:
            budget = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise Stage2DiskGateError("frozen disk budget is invalid JSON") from exc
        if not isinstance(budget, Mapping) or not isinstance(budget.get("modes"), list):
            raise Stage2DiskGateError("frozen disk budget lacks mode rows")
        matches = [
            row
            for row in budget["modes"]
            if isinstance(row, Mapping) and row.get("mode") == storage_mode
        ]
        if len(matches) != 1:
            raise Stage2DiskGateError(f"frozen disk budget mode is not unique: {storage_mode}")
        row = matches[0]
        start = row.get(
            "minimum_free_disk_required_before_start_bytes",
            row.get("recommended_free_disk_bytes"),
        )
        watermark = row.get("runtime_low_disk_watermark_bytes")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise Stage2DiskGateError("frozen minimum start threshold is invalid")
        if isinstance(watermark, bool) or not isinstance(watermark, int) or watermark < 0:
            raise Stage2DiskGateError("frozen runtime watermark is invalid")
        top_start = budget.get("minimum_free_disk_required_before_start_bytes")
        top_watermark = budget.get("runtime_low_disk_watermark_bytes")
        if storage_mode == "RECOMMENDED_OPERATIONAL" and (
            top_start != start or top_watermark != watermark
        ):
            raise Stage2DiskGateError("top-level disk thresholds disagree with recommended mode")
        return cls(
            monitored_path,
            thresholds=DiskGateThresholds(
                minimum_start_free_bytes=start,
                runtime_low_disk_watermark_bytes=watermark,
                storage_mode=storage_mode,
                storage_budget_sha256=expected,
            ),
            audit_log_path=audit_log_path,
            free_bytes_provider=free_bytes_provider,
            now=now,
        )

    def _fresh_free_bytes(self) -> int:
        value = self._free_bytes_provider(self.monitored_path)
        return _strict_nonnegative_int(value, field="live free bytes")

    def _timestamp(self) -> str:
        value = self._now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise Stage2DiskGateError("disk-gate clock must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")

    def _append_decision(
        self,
        *,
        operation: str,
        operation_id: str,
        current_free_bytes: int,
        projected_write_bytes: int,
        threshold_bytes: int,
    ) -> dict[str, Any]:
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.audit_log_path, flags, 0o600)
        except OSError as exc:
            raise Stage2DiskGateError("cannot safely open disk-gate audit log") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise Stage2DiskGateError("disk-gate audit descriptor is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.lseek(descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            existing = _parse_event_log(
                b"".join(chunks), expected_config_sha256=self.gate_config_sha256
            )
            previous = existing[-1]["event_sha256"] if existing else ZERO_SHA256
            remaining = current_free_bytes - projected_write_bytes
            envelope = {
                "current_free_bytes": current_free_bytes,
                "gate_config_sha256": self.gate_config_sha256,
                "operation": operation,
                "operation_id": operation_id,
                "pass": remaining >= threshold_bytes,
                "previous_event_sha256": previous,
                "projected_remaining_free_bytes": remaining,
                "projected_write_bytes": projected_write_bytes,
                "schema": DISK_GATE_EVENT_SCHEMA,
                "sequence_number": len(existing) + 1,
                "threshold_bytes": threshold_bytes,
                "timestamp_utc": self._timestamp(),
            }
            envelope["event_sha256"] = _event_digest(envelope)
            view = memoryview(_canonical_line(envelope))
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("zero-byte disk-gate audit append")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        directory = os.open(
            self.audit_log_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return envelope

    def assert_start(self, *, operation_id: str = "stage2-start") -> dict[str, Any]:
        operation_id = _validate_operation_id(operation_id)
        current = self._fresh_free_bytes()
        event = self._append_decision(
            operation="START",
            operation_id=operation_id,
            current_free_bytes=current,
            projected_write_bytes=0,
            threshold_bytes=self.thresholds.minimum_start_free_bytes,
        )
        if not event["pass"]:
            raise InsufficientLiveDiskError(
                "BLOCKED_INSUFFICIENT_DISK: live free bytes are below the frozen start threshold"
            )
        self._started = True
        return event

    def assert_resume(self, *, operation_id: str = "stage2-resume") -> dict[str, Any]:
        """Enter runtime mode only after the caller authenticates allocated state.

        This gate deliberately does not decide whether a resume artifact is
        trustworthy.  The production runner must first prove the exact,
        nonsparse replay array and its ledger.  Once proven, only the frozen
        runtime watermark is relevant because the large replay allocation is
        already reflected in the live free-space reading.
        """

        operation_id = _validate_operation_id(operation_id)
        current = self._fresh_free_bytes()
        event = self._append_decision(
            operation="RESUME",
            operation_id=operation_id,
            current_free_bytes=current,
            projected_write_bytes=0,
            threshold_bytes=self.thresholds.runtime_low_disk_watermark_bytes,
        )
        if not event["pass"]:
            raise InsufficientLiveDiskError(
                "BLOCKED_INSUFFICIENT_DISK: live free bytes are below the frozen resume watermark"
            )
        self._started = True
        return event

    def _runtime_gate(
        self, operation: str, projected_write_bytes: int, operation_id: str
    ) -> dict[str, Any]:
        if not self._started:
            raise Stage2DiskGateError("runtime disk gate used before a passing live start gate")
        if operation not in RUNTIME_OPERATIONS:
            raise Stage2DiskGateError(f"unknown runtime disk operation: {operation}")
        projected = _strict_nonnegative_int(
            projected_write_bytes, field="projected_write_bytes"
        )
        operation_id = _validate_operation_id(operation_id)
        current = self._fresh_free_bytes()
        event = self._append_decision(
            operation=operation,
            operation_id=operation_id,
            current_free_bytes=current,
            projected_write_bytes=projected,
            threshold_bytes=self.thresholds.runtime_low_disk_watermark_bytes,
        )
        if not event["pass"]:
            raise InsufficientLiveDiskError(
                f"{operation} would cross the frozen runtime low-disk watermark"
            )
        return event

    def before_download(self, projected_write_bytes: int, *, object_key: str) -> dict[str, Any]:
        return self._runtime_gate("BEFORE_DOWNLOAD", projected_write_bytes, object_key)

    def before_materialization(
        self, projected_write_bytes: int, *, artifact_id: str
    ) -> dict[str, Any]:
        return self._runtime_gate(
            "BEFORE_MATERIALIZATION", projected_write_bytes, artifact_id
        )

    def before_checkpoint(
        self, projected_write_bytes: int, *, checkpoint_id: str
    ) -> dict[str, Any]:
        return self._runtime_gate("BEFORE_CHECKPOINT", projected_write_bytes, checkpoint_id)

    def verify_audit(self) -> tuple[dict[str, Any], ...]:
        if not self.audit_log_path.exists():
            return ()
        if self.audit_log_path.is_symlink() or not self.audit_log_path.is_file():
            raise Stage2DiskGateError("disk-gate audit path is unsafe")
        rows = _parse_event_log(
            self.audit_log_path.read_bytes(),
            expected_config_sha256=self.gate_config_sha256,
        )
        return tuple(json.loads(json.dumps(row)) for row in rows)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return self.verify_audit()


__all__ = [
    "DiskGateThresholds",
    "InsufficientLiveDiskError",
    "Stage2DiskGate",
    "Stage2DiskGateError",
]
