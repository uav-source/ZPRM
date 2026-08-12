"""Strict, allowlist-only Boreas Stage-2 remote object materialization.

The frozen Stage-1 CSV intentionally contains no ETag.  This module therefore
keeps two separate authorities:

* :func:`load_frozen_allowlist` authenticates the immutable scientific scope;
* :func:`reconcile_remote_metadata` adds the current S3 identity and fails if
  size, LastModified, sequence, timestamp, or membership differs.

Only a reconciled :class:`AuthorizedRemoteObject` can reach the downloader.
The downloader uses an S3 ``If-Match`` condition, verifies the response and
local bytes, and materializes into an explicitly marked disposable directory.
It contains no registration, decoding, or geometry code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence

from .io import canonical_json_bytes, sha256_file
from .boreas_v2_stage2_authorization import (
    BoreasStage2AuthorizationError,
    VerifiedStage2Authorization,
)
from .stage2_checkpoint import (
    Stage2CheckpointError,
    TEMP_ROOT_SCHEMA,
    cleanup_partial_temporaries,
    initialize_temporary_root,
)
from .stage2_disk_gate import Stage2DiskGate


ALLOWLIST_FIELDS = (
    "selection_role",
    "sequence_id",
    "key",
    "timestamp_us",
    "last_modified",
    "size_bytes",
    "selection_reason",
)
ALLOWED_SELECTION_ROLES = frozenset({"TARGET_MAP", "QUERY"})
BOREAS_RAW_POINT_STRIDE_BYTES = 24
ETAG_PATTERN = re.compile(r"^[0-9a-f]{32}(?:-[1-9][0-9]*)?$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TIMESTAMP_STEM_PATTERN = re.compile(r"^[0-9]{16}$")
DOWNLOAD_RECEIPT_SCHEMA = "zprm.boreas.stage2.download_receipt.v1"
ZERO_SHA256 = "0" * 64
BOREAS_V2_STAGE2_FROZEN_ALLOWLIST_SHA256 = (
    "26ac211c854472dcb3db2f1cd5b096849bfd27867bac34e75bc8ce0d01bb2787"
)
# Stage-1 streamed LiDAR metadata through ``aws s3 ls``.  That command rendered
# the object clock in the host's Asia/Shanghai timezone, while the Stage-1
# parser appended ``Z`` without converting the local display back to UTC.  A
# live audit of every one of the 20,061 frozen rows found one exact offset and
# no key/size exception.  Preserve the immutable scientific allowlist bytes,
# but interpret this one authenticated file's display clock correctly when
# reconciling against S3's canonical UTC LastModified value.
STAGE1_S3_LS_FROZEN_DISPLAY_OFFSET_SECONDS = 8 * 60 * 60


class BoreasStage2RemoteError(RuntimeError):
    """An allowlist, remote identity, or temporary payload was not trustworthy."""


class RemoteIdentityChangedError(BoreasStage2RemoteError):
    """S3 metadata or a conditional download no longer matches reconciliation."""


class UnauthorizedRemoteObjectError(BoreasStage2RemoteError):
    """A caller attempted to materialize an object outside the reconciled set."""


def _strict_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise BoreasStage2RemoteError(f"{field} must be a positive integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None:
        result = int(value)
    else:
        raise BoreasStage2RemoteError(f"{field} must be a positive integer")
    if result <= 0:
        raise BoreasStage2RemoteError(f"{field} must be a positive integer")
    return result


def _normalize_etag(value: Any) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]
    text = text.lower()
    if ETAG_PATTERN.fullmatch(text) is None:
        raise BoreasStage2RemoteError(f"invalid S3 ETag: {value!r}")
    return text


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError) as exc:
            raise BoreasStage2RemoteError(
                f"{field} is not an explicit timestamp: {value!r}"
            ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BoreasStage2RemoteError(f"{field} lacks a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_utc(value: Any, *, field: str) -> str:
    parsed = _parse_timestamp(value, field=field)
    if parsed.microsecond:
        return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_key(key: Any, *, sequence_id: Any, timestamp_us: Any) -> tuple[str, str, int]:
    text = str(key)
    sequence = str(sequence_id)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or len(path.parts) != 3
        or path.parts[0] != sequence
        or path.parts[1] != "lidar"
        or path.suffix != ".bin"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BoreasStage2RemoteError(
            f"object key is not the expected sequence/lidar payload: {text!r}"
        )
    stem = path.stem
    if TIMESTAMP_STEM_PATTERN.fullmatch(stem) is None:
        raise BoreasStage2RemoteError(f"object key has an invalid timestamp: {text!r}")
    timestamp = _strict_positive_int(timestamp_us, field="timestamp_us")
    if int(stem) != timestamp:
        raise BoreasStage2RemoteError("object key timestamp differs from timestamp_us")
    return text, sequence, timestamp


@dataclass(frozen=True)
class FrozenAllowlistObject:
    ordinal: int
    role_ordinal: int
    selection_role: str
    sequence_id: str
    key: str
    timestamp_us: int
    last_modified: str
    size_bytes: int
    selection_reason: str

    def identity_without_etag(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "last_modified": self.last_modified,
            "sequence_id": self.sequence_id,
            "size_bytes": self.size_bytes,
            "timestamp_us": self.timestamp_us,
        }


@dataclass(frozen=True)
class FrozenAllowlist:
    path: Path
    sha256: str
    objects: tuple[FrozenAllowlistObject, ...]

    @property
    def by_key(self) -> dict[str, FrozenAllowlistObject]:
        return {item.key: item for item in self.objects}

    def for_role(self, role: str) -> tuple[FrozenAllowlistObject, ...]:
        if role not in ALLOWED_SELECTION_ROLES:
            raise BoreasStage2RemoteError(f"unknown allowlist role: {role!r}")
        return tuple(item for item in self.objects if item.selection_role == role)


@dataclass(frozen=True)
class RemoteObjectIdentity:
    key: str
    size_bytes: int
    etag: str
    last_modified: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(
            self,
            "size_bytes",
            _strict_positive_int(self.size_bytes, field="remote size"),
        )
        object.__setattr__(self, "etag", _normalize_etag(self.etag))
        object.__setattr__(
            self,
            "last_modified",
            _canonical_utc(self.last_modified, field="remote LastModified"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RemoteObjectIdentity":
        allowed_key = value.get("key", value.get("Key"))
        allowed_size = value.get(
            "size_bytes", value.get("Size", value.get("ContentLength"))
        )
        allowed_etag = value.get("etag", value.get("ETag"))
        allowed_modified = value.get("last_modified", value.get("LastModified"))
        if any(item is None for item in (allowed_key, allowed_size, allowed_etag, allowed_modified)):
            raise BoreasStage2RemoteError("remote metadata row lacks key/size/ETag/LastModified")
        return cls(
            key=str(allowed_key),
            size_bytes=_strict_positive_int(allowed_size, field="remote size"),
            etag=_normalize_etag(allowed_etag),
            last_modified=_canonical_utc(allowed_modified, field="remote LastModified"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "etag": self.etag,
            "key": self.key,
            "last_modified": self.last_modified,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class AuthorizedRemoteObject:
    frozen: FrozenAllowlistObject
    remote: RemoteObjectIdentity
    frozen_last_modified_display_offset_seconds: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.frozen, FrozenAllowlistObject) or not isinstance(
            self.remote, RemoteObjectIdentity
        ):
            raise BoreasStage2RemoteError(
                "authorized object requires frozen and remote identity records"
            )
        if self.remote.key != self.frozen.key:
            raise BoreasStage2RemoteError("authorized remote key differs from frozen key")
        if self.remote.size_bytes != self.frozen.size_bytes:
            raise RemoteIdentityChangedError(
                "authorized remote size differs from frozen size"
            )
        offset = self.frozen_last_modified_display_offset_seconds
        if isinstance(offset, bool) or offset not in {
            0,
            STAGE1_S3_LS_FROZEN_DISPLAY_OFFSET_SECONDS,
        }:
            raise BoreasStage2RemoteError(
                "frozen LastModified display offset is not an approved value"
            )
        expected_remote = _parse_timestamp(
            self.frozen.last_modified, field="allowlist LastModified"
        ) - timedelta(seconds=offset)
        if _parse_timestamp(
            self.remote.last_modified, field="remote LastModified"
        ) != expected_remote:
            raise RemoteIdentityChangedError(
                "authorized remote LastModified differs from frozen value"
            )

    @property
    def key(self) -> str:
        return self.frozen.key

    @property
    def selection_role(self) -> str:
        return self.frozen.selection_role

    @property
    def size_bytes(self) -> int:
        return self.remote.size_bytes

    @property
    def etag(self) -> str:
        return self.remote.etag

    @property
    def last_modified(self) -> str:
        return self.remote.last_modified

    def identity(self) -> dict[str, Any]:
        return {
            **self.frozen.identity_without_etag(),
            "etag": self.remote.etag,
            "frozen_last_modified_display_offset_seconds": (
                self.frozen_last_modified_display_offset_seconds
            ),
            "selection_role": self.frozen.selection_role,
        }


@dataclass(frozen=True)
class ReconciledRemoteInventory:
    allowlist_sha256: str
    objects: tuple[AuthorizedRemoteObject, ...]
    inventory_sha256: str

    def __post_init__(self) -> None:
        if SHA256_PATTERN.fullmatch(str(self.allowlist_sha256)) is None:
            raise BoreasStage2RemoteError(
                "reconciled allowlist SHA must be lowercase SHA-256"
            )
        if not self.objects or any(
            not isinstance(item, AuthorizedRemoteObject) for item in self.objects
        ):
            raise BoreasStage2RemoteError(
                "reconciled inventory requires authorized object records"
            )
        keys = [item.key for item in self.objects]
        if len(keys) != len(set(keys)):
            raise BoreasStage2RemoteError("reconciled inventory keys are not unique")
        actual = hashlib.sha256(
            canonical_json_bytes([item.identity() for item in self.objects])
        ).hexdigest()
        if self.inventory_sha256 != actual:
            raise BoreasStage2RemoteError("reconciled inventory SHA-256 differs")

    @property
    def by_key(self) -> dict[str, AuthorizedRemoteObject]:
        return {item.key: item for item in self.objects}

    def for_role(self, role: str) -> tuple[AuthorizedRemoteObject, ...]:
        if role not in ALLOWED_SELECTION_ROLES:
            raise BoreasStage2RemoteError(f"unknown allowlist role: {role!r}")
        return tuple(item for item in self.objects if item.selection_role == role)


def load_frozen_allowlist(
    path: str | Path, *, expected_sha256: str
) -> FrozenAllowlist:
    """Authenticate and strictly parse the Stage-1 download allowlist."""

    source = Path(path)
    expected = str(expected_sha256).lower()
    if SHA256_PATTERN.fullmatch(expected) is None:
        raise BoreasStage2RemoteError("expected allowlist SHA must be lowercase SHA-256")
    if not source.is_absolute():
        raise BoreasStage2RemoteError("frozen allowlist path must be absolute")
    if source.is_symlink() or not source.is_file() or source.resolve(strict=True) != source:
        raise BoreasStage2RemoteError("frozen allowlist path is unsafe")
    actual = sha256_file(source)
    if actual != expected:
        raise BoreasStage2RemoteError("frozen allowlist SHA-256 differs")

    objects: list[FrozenAllowlistObject] = []
    role_counts = {role: 0 for role in ALLOWED_SELECTION_ROLES}
    last_timestamp: dict[str, int] = {}
    role_sequences: dict[str, str] = {}
    seen_keys: set[str] = set()
    try:
        with source.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != ALLOWLIST_FIELDS:
                raise BoreasStage2RemoteError("frozen allowlist CSV field order differs")
            for row in reader:
                if None in row or any(row[field] is None for field in ALLOWLIST_FIELDS):
                    raise BoreasStage2RemoteError("frozen allowlist contains a malformed CSV row")
                role = str(row["selection_role"])
                if role not in ALLOWED_SELECTION_ROLES:
                    raise BoreasStage2RemoteError(f"unknown frozen selection role: {role!r}")
                key, sequence, timestamp = _validate_key(
                    row["key"],
                    sequence_id=row["sequence_id"],
                    timestamp_us=row["timestamp_us"],
                )
                if key in seen_keys:
                    raise BoreasStage2RemoteError(f"duplicate frozen object key: {key}")
                if role in role_sequences and role_sequences[role] != sequence:
                    raise BoreasStage2RemoteError("one allowlist role spans multiple sequences")
                if timestamp <= last_timestamp.get(role, -1):
                    raise BoreasStage2RemoteError(
                        f"{role} allowlist timestamps are not strictly increasing"
                    )
                size = _strict_positive_int(row["size_bytes"], field="size_bytes")
                if size % BOREAS_RAW_POINT_STRIDE_BYTES:
                    raise BoreasStage2RemoteError(
                        "Boreas payload size is not divisible by the 24-byte point stride"
                    )
                reason = str(row["selection_reason"])
                if not reason.strip():
                    raise BoreasStage2RemoteError("selection_reason must be nonempty")
                modified = _canonical_utc(
                    row["last_modified"], field="allowlist LastModified"
                )
                objects.append(
                    FrozenAllowlistObject(
                        ordinal=len(objects),
                        role_ordinal=role_counts[role],
                        selection_role=role,
                        sequence_id=sequence,
                        key=key,
                        timestamp_us=timestamp,
                        last_modified=modified,
                        size_bytes=size,
                        selection_reason=reason,
                    )
                )
                role_counts[role] += 1
                last_timestamp[role] = timestamp
                role_sequences[role] = sequence
                seen_keys.add(key)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BoreasStage2RemoteError("cannot parse frozen allowlist CSV") from exc
    if not objects:
        raise BoreasStage2RemoteError("frozen allowlist is empty")
    return FrozenAllowlist(path=source, sha256=actual, objects=tuple(objects))


CommandRunner = Callable[[Sequence[str]], Any]


def _default_command_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _validated_aws_executable(path: str | Path) -> Path:
    executable = Path(path)
    if (
        not executable.is_absolute()
        or executable.is_symlink()
        or not executable.is_file()
        or executable.resolve(strict=True) != executable
        or not os.access(executable, os.X_OK)
    ):
        raise BoreasStage2RemoteError("AWS CLI executable path is unsafe or not executable")
    return executable


def list_remote_metadata(
    *,
    aws_executable: str | Path,
    bucket: str,
    sequence_ids: Sequence[str],
    command_runner: CommandRunner = _default_command_runner,
) -> tuple[RemoteObjectIdentity, ...]:
    """List metadata only for explicit sequence prefixes using AWS CLI pagination."""

    aws = _validated_aws_executable(aws_executable)
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", str(bucket)):
        raise BoreasStage2RemoteError("invalid S3 bucket")
    sequences = tuple(str(value) for value in sequence_ids)
    if not sequences or len(sequences) != len(set(sequences)):
        raise BoreasStage2RemoteError("sequence_ids must be a nonempty unique sequence")
    rows: list[RemoteObjectIdentity] = []
    seen: set[str] = set()
    for sequence in sequences:
        if not sequence or "/" in sequence or sequence in {".", ".."}:
            raise BoreasStage2RemoteError(f"unsafe sequence id: {sequence!r}")
        argv = (
            str(aws),
            "s3api",
            "list-objects-v2",
            "--bucket",
            str(bucket),
            "--prefix",
            f"{sequence}/lidar/",
            "--query",
            "Contents[].{Key:Key,Size:Size,ETag:ETag,LastModified:LastModified}",
            "--output",
            "json",
            "--no-sign-request",
        )
        result = command_runner(argv)
        if result.returncode != 0:
            raise BoreasStage2RemoteError(
                f"S3 metadata listing failed for {sequence}: {result.stderr.strip()}"
            )
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise BoreasStage2RemoteError("S3 metadata listing returned invalid JSON") from exc
        if not isinstance(value, list):
            raise BoreasStage2RemoteError("S3 metadata listing must return one JSON array")
        for raw in value:
            if not isinstance(raw, Mapping):
                raise BoreasStage2RemoteError("S3 metadata listing contains a non-object row")
            identity = RemoteObjectIdentity.from_mapping(raw)
            path = PurePosixPath(identity.key)
            if (
                len(path.parts) != 3
                or path.parts[:2] != (sequence, "lidar")
                or path.suffix != ".bin"
            ):
                raise BoreasStage2RemoteError(
                    f"metadata listing escaped the requested prefix: {identity.key}"
                )
            if identity.key in seen:
                raise BoreasStage2RemoteError(
                    f"duplicate key across metadata listings: {identity.key}"
                )
            rows.append(identity)
            seen.add(identity.key)
    return tuple(sorted(rows, key=lambda item: item.key))


def reconcile_remote_metadata(
    allowlist: FrozenAllowlist,
    remote_rows: Sequence[RemoteObjectIdentity | Mapping[str, Any]],
) -> ReconciledRemoteInventory:
    """Add ETags without broadening or changing the frozen allowlist scope."""

    remote: dict[str, RemoteObjectIdentity] = {}
    for value in remote_rows:
        item = value if isinstance(value, RemoteObjectIdentity) else RemoteObjectIdentity.from_mapping(value)
        if item.key in remote:
            raise BoreasStage2RemoteError(f"duplicate remote metadata key: {item.key}")
        remote[item.key] = item
    display_offset_seconds = (
        STAGE1_S3_LS_FROZEN_DISPLAY_OFFSET_SECONDS
        if allowlist.sha256 == BOREAS_V2_STAGE2_FROZEN_ALLOWLIST_SHA256
        else 0
    )
    authorized: list[AuthorizedRemoteObject] = []
    for frozen in allowlist.objects:
        current = remote.get(frozen.key)
        if current is None:
            raise RemoteIdentityChangedError(
                f"allowlisted S3 object is absent from metadata: {frozen.key}"
            )
        if current.size_bytes != frozen.size_bytes:
            raise RemoteIdentityChangedError(
                f"remote size changed for allowlisted object: {frozen.key}"
            )
        expected_remote = _parse_timestamp(
            frozen.last_modified, field="allowlist LastModified"
        ) - timedelta(seconds=display_offset_seconds)
        if _parse_timestamp(
            current.last_modified, field="remote LastModified"
        ) != expected_remote:
            raise RemoteIdentityChangedError(
                f"remote LastModified changed for allowlisted object: {frozen.key}"
            )
        # Re-parse the key rather than trusting fields inferred by the CSV loader.
        _validate_key(
            current.key,
            sequence_id=frozen.sequence_id,
            timestamp_us=frozen.timestamp_us,
        )
        authorized.append(
            AuthorizedRemoteObject(
                frozen=frozen,
                remote=current,
                frozen_last_modified_display_offset_seconds=display_offset_seconds,
            )
        )
    identity_rows = [item.identity() for item in authorized]
    digest = hashlib.sha256(canonical_json_bytes(identity_rows)).hexdigest()
    return ReconciledRemoteInventory(
        allowlist_sha256=allowlist.sha256,
        objects=tuple(authorized),
        inventory_sha256=digest,
    )


@dataclass(frozen=True)
class DownloadReceipt:
    selection_role: str
    sequence_id: str
    key: str
    timestamp_us: int
    remote_size_bytes: int
    etag: str
    last_modified: str
    local_temporary_sha256: str
    downloaded_at_utc: str

    def payload_identity(self) -> dict[str, Any]:
        """Fields that must be identical across query first/second pass."""

        return {
            "etag": self.etag,
            "key": self.key,
            "last_modified": self.last_modified,
            "local_temporary_sha256": self.local_temporary_sha256,
            "remote_size_bytes": self.remote_size_bytes,
            "selection_role": self.selection_role,
            "sequence_id": self.sequence_id,
            "timestamp_us": self.timestamp_us,
        }

    def core(self) -> dict[str, Any]:
        return {
            "downloaded_at_utc": self.downloaded_at_utc,
            **self.payload_identity(),
            "schema": DOWNLOAD_RECEIPT_SCHEMA,
        }

    def as_dict(self) -> dict[str, Any]:
        core = self.core()
        return {
            **core,
            "receipt_sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest(),
        }

    def csv_row(self, *, execution_stage: str) -> dict[str, Any]:
        """Project one receipt into the frozen preparation-verifier CSV schema."""

        stage = str(execution_stage)
        if not stage or any(character in stage for character in ("\x00", "\r", "\n")):
            raise BoreasStage2RemoteError("execution_stage is empty or unsafe")
        return {
            "execution_stage": stage,
            "selection_role": self.selection_role,
            "sequence_id": self.sequence_id,
            "object_key": self.key,
            "timestamp_us": self.timestamp_us,
            "last_modified": self.last_modified,
            "remote_size_bytes": self.remote_size_bytes,
            "etag": self.etag,
            "payload_sha256": self.local_temporary_sha256,
            "receipt_status": "PASS_SIZE_ETAG_SHA256",
        }


@dataclass(frozen=True)
class TemporaryDownloadedObject:
    authorized_object: AuthorizedRemoteObject
    path: Path
    receipt: DownloadReceipt


class StrictAllowlistDownloader:
    """Materialize only reconciled objects with a mandatory live disk gate."""

    def __init__(
        self,
        inventory: ReconciledRemoteInventory,
        *,
        aws_executable: str | Path,
        bucket: str,
        temporary_root: str | Path,
        disk_gate: Stage2DiskGate,
        authorization: VerifiedStage2Authorization,
        command_runner: CommandRunner = _default_command_runner,
        now: Callable[[], datetime] | None = None,
        preserve_existing_temporary_payload: bool = False,
    ) -> None:
        if not isinstance(disk_gate, Stage2DiskGate):
            raise BoreasStage2RemoteError("strict downloader requires a Stage2DiskGate")
        if not isinstance(authorization, VerifiedStage2Authorization):
            raise BoreasStage2RemoteError(
                "strict downloader requires a VerifiedStage2Authorization capability"
            )
        if not isinstance(inventory, ReconciledRemoteInventory):
            raise BoreasStage2RemoteError(
                "strict downloader requires a reconciled remote inventory"
            )
        try:
            authorization.assert_live()
            authorization.bind_disk_gate(disk_gate)
        except BoreasStage2AuthorizationError as exc:
            raise BoreasStage2RemoteError(
                "strict downloader authorization is not live"
            ) from exc
        if inventory.allowlist_sha256 != authorization.allowlist_sha256:
            raise BoreasStage2RemoteError(
                "reconciled inventory allowlist SHA differs from authorization"
            )
        if len(inventory.objects) != authorization.expected_object_count:
            raise BoreasStage2RemoteError(
                "reconciled inventory object count differs from authorization"
            )
        if sum(item.size_bytes for item in inventory.objects) != authorization.expected_remote_bytes:
            raise BoreasStage2RemoteError(
                "reconciled inventory remote bytes differ from authorization"
            )
        pair = authorization.primary_pair
        expected_roles = {
            "TARGET_MAP": pair["map_sequence_id"],
            "QUERY": pair["query_sequence_id"],
        }
        for role, sequence_id in expected_roles.items():
            rows = inventory.for_role(role)
            if not rows or any(item.frozen.sequence_id != sequence_id for item in rows):
                raise BoreasStage2RemoteError(
                    f"reconciled inventory {role} scope differs from authorization"
                )
        self.inventory = inventory
        self.authorization = authorization
        self.aws_executable = _validated_aws_executable(aws_executable)
        self.bucket = str(bucket)
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", self.bucket):
            raise BoreasStage2RemoteError("invalid S3 bucket")
        if self.bucket != authorization.bucket:
            raise BoreasStage2RemoteError("S3 bucket differs from authorization")
        temporary_path = Path(temporary_root)
        if preserve_existing_temporary_payload:
            marker = temporary_path / ".stage2_temporary_root.json"
            expected_marker = {
                "canonical_root": str(temporary_path),
                "purpose": "tmp_download",
                "schema": TEMP_ROOT_SCHEMA,
                "temporary": True,
            }
            if (
                not temporary_path.is_absolute()
                or not temporary_path.is_dir()
                or temporary_path.is_symlink()
                or temporary_path.resolve(strict=True) != temporary_path
                or marker.is_symlink()
                or not marker.is_file()
                or marker.read_bytes() != canonical_json_bytes(expected_marker)
            ):
                raise BoreasStage2RemoteError(
                    "unsafe retained Stage-2 download temporary root"
                )
            self.temporary_root = temporary_path
        else:
            try:
                self.temporary_root = initialize_temporary_root(
                    temporary_path, purpose="tmp_download"
                )
            except Stage2CheckpointError as exc:
                raise BoreasStage2RemoteError(
                    "unsafe Stage-2 download temporary root"
                ) from exc
        if self.temporary_root != authorization.temporary_root / "tmp_download":
            raise BoreasStage2RemoteError(
                "download temporary root differs from the authorized tmp_download child"
            )
        self.disk_gate = disk_gate
        self.command_runner = command_runner
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._authorized = inventory.by_key
        if preserve_existing_temporary_payload:
            authorized_names = {
                self._temporary_path(item).name for item in inventory.objects
            }
            for path in self.temporary_root.iterdir():
                if path.name == ".stage2_temporary_root.json":
                    continue
                metadata = os.lstat(path)
                if (
                    path.name not in authorized_names
                    or path.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise BoreasStage2RemoteError(
                        "retained Stage-2 payload is outside authorized deterministic paths"
                    )

    def _resolve_authorized(
        self, value: AuthorizedRemoteObject
    ) -> AuthorizedRemoteObject:
        if not isinstance(value, AuthorizedRemoteObject):
            raise UnauthorizedRemoteObjectError(
                "download requires a reconciled AuthorizedRemoteObject"
            )
        expected = self._authorized.get(value.key)
        if expected is None or expected != value:
            raise UnauthorizedRemoteObjectError(
                f"object is not in the reconciled frozen allowlist: {value.key}"
            )
        return expected

    def _temporary_path(self, item: AuthorizedRemoteObject) -> Path:
        key_digest = hashlib.sha256(item.key.encode("utf-8")).hexdigest()[:24]
        return self.temporary_root / f"{item.frozen.ordinal:06d}-{key_digest}.bin.partial"

    def planned_temporary_path(self, value: AuthorizedRemoteObject) -> Path:
        """Return the deterministic authorized path for a durable transfer intent."""

        return self._temporary_path(self._resolve_authorized(value))

    @staticmethod
    def _unlink_regular(path: Path) -> None:
        if not path.exists():
            return
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise BoreasStage2RemoteError(
                f"refusing to clean unsafe temporary payload: {path}"
            )
        path.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def materialize(
        self,
        value: AuthorizedRemoteObject,
        *,
        authenticated_receipt_sink: Callable[[TemporaryDownloadedObject], None]
        | None = None,
    ) -> TemporaryDownloadedObject:
        try:
            self.authorization.assert_operation_live()
        except BoreasStage2AuthorizationError as exc:
            raise BoreasStage2RemoteError("download authorization is no longer live") from exc
        item = self._resolve_authorized(value)
        self.disk_gate.before_download(item.size_bytes, object_key=item.key)
        destination = self._temporary_path(item)
        if destination.exists() or destination.is_symlink():
            raise BoreasStage2RemoteError(
                f"uncommitted temporary payload already exists: {destination.name}"
            )
        argv = (
            str(self.aws_executable),
            "s3api",
            "get-object",
            "--bucket",
            self.bucket,
            "--key",
            item.key,
            "--if-match",
            f'"{item.etag}"',
            "--output",
            "json",
            "--no-sign-request",
            str(destination),
        )
        try:
            try:
                self.authorization.assert_operation_live()
            except BoreasStage2AuthorizationError as exc:
                raise BoreasStage2RemoteError(
                    "download authorization is no longer live"
                ) from exc
            result = self.command_runner(argv)
            if result.returncode != 0:
                raise BoreasStage2RemoteError(
                    f"conditional S3 download failed for {item.key}: {result.stderr.strip()}"
                )
            try:
                response = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise BoreasStage2RemoteError(
                    "conditional S3 download returned invalid metadata JSON"
                ) from exc
            if not isinstance(response, Mapping):
                raise BoreasStage2RemoteError(
                    "conditional S3 download response must be a JSON object"
                )
            response_identity = RemoteObjectIdentity.from_mapping(
                {
                    "key": item.key,
                    "ContentLength": response.get("ContentLength"),
                    "ETag": response.get("ETag"),
                    "LastModified": response.get("LastModified"),
                }
            )
            if response_identity != item.remote:
                raise RemoteIdentityChangedError(
                    f"conditional download identity changed for {item.key}"
                )
            if not destination.is_file() or destination.is_symlink():
                raise BoreasStage2RemoteError("AWS CLI did not create a regular temporary file")
            metadata = os.lstat(destination)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise BoreasStage2RemoteError("temporary payload file identity is unsafe")
            if metadata.st_size != item.size_bytes:
                raise RemoteIdentityChangedError(
                    f"downloaded byte size differs for {item.key}"
                )
            local_sha = sha256_file(destination)
            if SHA256_PATTERN.fullmatch(local_sha) is None:
                raise BoreasStage2RemoteError("local payload SHA-256 is invalid")
            now = self._now()
            if now.tzinfo is None or now.utcoffset() is None:
                raise BoreasStage2RemoteError("download receipt clock must be timezone-aware")
            receipt = DownloadReceipt(
                selection_role=item.selection_role,
                sequence_id=item.frozen.sequence_id,
                key=item.key,
                timestamp_us=item.frozen.timestamp_us,
                remote_size_bytes=item.size_bytes,
                etag=item.etag,
                last_modified=item.last_modified,
                local_temporary_sha256=local_sha,
                downloaded_at_utc=now.astimezone(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z"),
            )
            materialized = TemporaryDownloadedObject(
                authorized_object=item, path=destination, receipt=receipt
            )
            if authenticated_receipt_sink is not None:
                authenticated_receipt_sink(materialized)
            return materialized
        except Exception:
            self._unlink_regular(destination)
            raise

    def recover_completed_transfer(
        self,
        value: AuthorizedRemoteObject,
        *,
        recovered_at_utc: datetime,
    ) -> TemporaryDownloadedObject:
        """Authenticate a full deterministic payload retained by a durable intent."""

        try:
            self.authorization.assert_operation_live()
        except BoreasStage2AuthorizationError as exc:
            raise BoreasStage2RemoteError(
                "download authorization is no longer live"
            ) from exc
        item = self._resolve_authorized(value)
        path = self._temporary_path(item)
        if not path.is_file() or path.is_symlink():
            raise BoreasStage2RemoteError("intended transfer payload is absent or unsafe")
        metadata = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != item.size_bytes
        ):
            raise BoreasStage2RemoteError(
                "intended transfer payload is incomplete or unsafe"
            )
        if recovered_at_utc.tzinfo is None or recovered_at_utc.utcoffset() is None:
            raise BoreasStage2RemoteError("recovery receipt clock must be timezone-aware")
        receipt = DownloadReceipt(
            selection_role=item.selection_role,
            sequence_id=item.frozen.sequence_id,
            key=item.key,
            timestamp_us=item.frozen.timestamp_us,
            remote_size_bytes=item.size_bytes,
            etag=item.etag,
            last_modified=item.last_modified,
            local_temporary_sha256=sha256_file(path),
            downloaded_at_utc=recovered_at_utc.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
        )
        return TemporaryDownloadedObject(item, path, receipt)

    def release(self, value: TemporaryDownloadedObject) -> int:
        """Delete exactly one authenticated temporary payload and return its size."""

        try:
            self.authorization.assert_operation_live()
        except BoreasStage2AuthorizationError as exc:
            raise BoreasStage2RemoteError("download authorization is no longer live") from exc
        if not isinstance(value, TemporaryDownloadedObject):
            raise BoreasStage2RemoteError("release requires a TemporaryDownloadedObject")
        expected = self._temporary_path(value.authorized_object)
        if value.path != expected or value.path.parent != self.temporary_root:
            raise BoreasStage2RemoteError("temporary payload path is outside downloader scope")
        if not value.path.is_file() or value.path.is_symlink():
            raise BoreasStage2RemoteError("temporary payload disappeared before release")
        if value.path.stat().st_size != value.receipt.remote_size_bytes:
            raise BoreasStage2RemoteError("temporary payload size changed before release")
        if sha256_file(value.path) != value.receipt.local_temporary_sha256:
            raise BoreasStage2RemoteError("temporary payload changed before release")
        size = value.path.stat().st_size
        self._unlink_regular(value.path)
        return size

    @contextmanager
    def temporary_payload(
        self, value: AuthorizedRemoteObject
    ) -> Iterator[TemporaryDownloadedObject]:
        materialized = self.materialize(value)
        try:
            yield materialized
        finally:
            if materialized.path.exists():
                self.release(materialized)

    def cleanup_uncommitted(self) -> dict[str, Any]:
        """Fail-closed cleanup for a crash-resume boundary with no active payload."""

        try:
            return cleanup_partial_temporaries(self.temporary_root)
        except Stage2CheckpointError as exc:
            raise BoreasStage2RemoteError("cannot safely clean download temporary root") from exc


__all__ = [
    "AuthorizedRemoteObject",
    "BoreasStage2RemoteError",
    "DownloadReceipt",
    "FrozenAllowlist",
    "FrozenAllowlistObject",
    "ReconciledRemoteInventory",
    "RemoteIdentityChangedError",
    "RemoteObjectIdentity",
    "StrictAllowlistDownloader",
    "TemporaryDownloadedObject",
    "UnauthorizedRemoteObjectError",
    "list_remote_metadata",
    "load_frozen_allowlist",
    "reconcile_remote_metadata",
]
