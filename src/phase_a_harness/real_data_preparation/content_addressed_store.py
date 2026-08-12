"""Immutable, content-addressed Stage-2 array storage.

The store deliberately keeps scientific arrays separate from backend working
formats.  A canonical NPY payload is published once under its SHA-256, while
snapshot manifests contain only content references.  PCL's PCD representation
is derived in a marked temporary directory and is always removed by the
context manager.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from phase_a_harness.pcl_backend import write_binary_xyz_pcd

from .io import canonical_json_bytes, sha256_file


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
OBJECT_SCHEMA = "zprm-stage2-content-addressed-object-v1"
SNAPSHOT_REFERENCE_SCHEMA = "zprm-stage2-snapshot-content-reference-v1"


class ContentAddressedStoreError(RuntimeError):
    """A content-addressed object or reference violated the immutable contract."""


@dataclass(frozen=True)
class StoredObject:
    """One verified physical object in a :class:`ContentAddressedStore`."""

    sha256: str
    object_kind: str
    payload_path: Path
    metadata_path: Path
    size_bytes: int


@dataclass(frozen=True)
class TemporaryPclConversion:
    """Paths and hashes for one temporary, deterministic PCL input bundle."""

    directory: Path
    source_path: Path
    target_path: Path
    source_pcd_sha256: str
    target_pcd_sha256: str
    source_npy_sha256: str
    target_npy_sha256: str


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_existing_directory(path: str | Path, *, create: bool = False) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ContentAddressedStoreError(f"directory must be absolute: {candidate}")
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    if not candidate.is_dir() or candidate.is_symlink():
        raise ContentAddressedStoreError(f"unsafe directory: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ContentAddressedStoreError(
            f"directory must be canonical: {candidate} != {resolved}"
        )
    return resolved


def _safe_name(value: str, label: str) -> str:
    name = str(value)
    if SAFE_NAME_PATTERN.fullmatch(name) is None or name in {".", ".."}:
        raise ContentAddressedStoreError(f"unsafe {label}: {name!r}")
    return name


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _publish_immutable_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish bytes without replacing a concurrent/existing file."""

    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ContentAddressedStoreError(f"immutable reference differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise ContentAddressedStoreError(f"immutable reference differs: {path}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_npy_bytes(value: np.ndarray) -> bytes:
    """Return a stable, little-endian, C-contiguous NPY v1.0 encoding.

    Object arrays and non-numeric values are rejected because their encodings
    can depend on pickle or Python implementation details.  Coordinate values
    are not rounded or otherwise scientifically transformed.
    """

    array = np.asarray(value)
    if array.dtype.hasobject or not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ContentAddressedStoreError("canonical NPY requires a non-object numeric array")
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian and array.dtype.itemsize > 1
    ):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    elif array.dtype.byteorder == "=" and array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    array = np.ascontiguousarray(array)
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def _canonical_npy_from_file(path: Path) -> tuple[np.ndarray, bytes]:
    if not path.is_file() or path.is_symlink():
        raise ContentAddressedStoreError(f"canonical NPY is missing or unsafe: {path}")
    payload = path.read_bytes()
    with io.BytesIO(payload) as stream:
        array = np.load(stream, allow_pickle=False)
        if stream.read(1) != b"":
            raise ContentAddressedStoreError(f"trailing bytes in canonical NPY: {path}")
    canonical = canonical_npy_bytes(array)
    if payload != canonical:
        raise ContentAddressedStoreError(f"NPY is not in canonical storage encoding: {path}")
    return array, payload


class ContentAddressedStore:
    """An immutable directory store with one physical payload per digest."""

    def __init__(self, root: str | Path) -> None:
        self.root = _safe_existing_directory(root, create=True)
        self.references_root = self.root / "snapshot_refs"

    @staticmethod
    def _validate_digest(value: str) -> str:
        digest = str(value)
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise ContentAddressedStoreError(f"invalid SHA-256: {digest!r}")
        return digest

    def _directory(self, digest: str) -> Path:
        return self.root / self._validate_digest(digest)

    def _metadata(
        self,
        *,
        digest: str,
        object_kind: str,
        payload_name: str,
        payload_size: int,
        payload_format: str,
        array: np.ndarray | None,
        user_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "schema": OBJECT_SCHEMA,
            "sha256": digest,
            "object_kind": _safe_name(object_kind, "object kind"),
            "payload_filename": _safe_name(payload_name, "payload filename"),
            "payload_format": payload_format,
            "size_bytes": int(payload_size),
            "user_metadata": dict(user_metadata or {}),
        }
        if array is not None:
            metadata.update(
                {
                    "dtype": array.dtype.str.replace("=", "<"),
                    "shape": [int(item) for item in array.shape],
                    "c_contiguous": True,
                    "allow_pickle": False,
                    "npy_version": "1.0",
                }
            )
        # Reject NaN, non-string dictionary keys, and non-JSON metadata now.
        canonical_json_bytes(metadata)
        return metadata

    def _publish(
        self,
        payload: bytes,
        *,
        object_kind: str,
        payload_name: str,
        payload_format: str,
        array: np.ndarray | None,
        user_metadata: Mapping[str, Any] | None,
    ) -> StoredObject:
        digest = _sha256(payload)
        destination = self._directory(digest)
        metadata = self._metadata(
            digest=digest,
            object_kind=object_kind,
            payload_name=payload_name,
            payload_size=len(payload),
            payload_format=payload_format,
            array=array,
            user_metadata=user_metadata,
        )
        expected_metadata = canonical_json_bytes(metadata)

        if destination.exists():
            return self._verify_existing(
                destination,
                expected_digest=digest,
                expected_payload_name=payload_name,
                expected_payload=payload,
                expected_metadata=expected_metadata,
            )

        temporary = Path(tempfile.mkdtemp(prefix=".cas-partial-", dir=self.root))
        try:
            payload_path = temporary / payload_name
            metadata_path = temporary / "metadata.json"
            with payload_path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            with metadata_path.open("xb") as stream:
                stream.write(expected_metadata)
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(temporary)
            try:
                os.rename(temporary, destination)
                _fsync_directory(self.root)
            except FileExistsError:
                # A concurrent writer may only win with byte-identical content.
                return self._verify_existing(
                    destination,
                    expected_digest=digest,
                    expected_payload_name=payload_name,
                    expected_payload=payload,
                    expected_metadata=expected_metadata,
                )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return self._verify_existing(
            destination,
            expected_digest=digest,
            expected_payload_name=payload_name,
            expected_payload=payload,
            expected_metadata=expected_metadata,
        )

    def _verify_existing(
        self,
        directory: Path,
        *,
        expected_digest: str,
        expected_payload_name: str | None = None,
        expected_payload: bytes | None = None,
        expected_metadata: bytes | None = None,
    ) -> StoredObject:
        if not directory.is_dir() or directory.is_symlink():
            raise ContentAddressedStoreError(f"unsafe object directory: {directory}")
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise ContentAddressedStoreError(f"missing object metadata: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError, OSError) as error:
            raise ContentAddressedStoreError(f"invalid object metadata: {metadata_path}") from error
        if not isinstance(metadata, dict) or metadata.get("schema") != OBJECT_SCHEMA:
            raise ContentAddressedStoreError(f"object metadata schema differs: {metadata_path}")
        if canonical_json_bytes(metadata) != metadata_path.read_bytes():
            raise ContentAddressedStoreError(f"object metadata is not canonical: {metadata_path}")
        if metadata.get("sha256") != expected_digest:
            raise ContentAddressedStoreError(f"object digest binding differs: {directory}")
        payload_name = str(metadata.get("payload_filename", ""))
        _safe_name(payload_name, "payload filename")
        if expected_payload_name is not None and payload_name != expected_payload_name:
            raise ContentAddressedStoreError("same content digest requested with different object contract")
        payload_path = directory / payload_name
        if not payload_path.is_file() or payload_path.is_symlink():
            raise ContentAddressedStoreError(f"missing or unsafe object payload: {payload_path}")
        actual_payload = payload_path.read_bytes()
        if _sha256(actual_payload) != expected_digest:
            raise ContentAddressedStoreError(f"object payload digest differs: {payload_path}")
        if int(metadata.get("size_bytes", -1)) != len(actual_payload):
            raise ContentAddressedStoreError(f"object payload size differs: {payload_path}")
        if expected_payload is not None and actual_payload != expected_payload:
            raise ContentAddressedStoreError("content-address collision or immutable payload mismatch")
        if expected_metadata is not None and metadata_path.read_bytes() != expected_metadata:
            raise ContentAddressedStoreError("immutable metadata differs for existing content")
        files = {path.name for path in directory.iterdir() if path.is_file()}
        if files != {payload_name, "metadata.json"}:
            raise ContentAddressedStoreError(f"unexpected physical files in object: {directory}")
        return StoredObject(
            sha256=expected_digest,
            object_kind=str(metadata["object_kind"]),
            payload_path=payload_path,
            metadata_path=metadata_path,
            size_bytes=len(actual_payload),
        )

    def put_npy(
        self,
        value: np.ndarray,
        *,
        object_kind: str = "canonical_array",
        payload_name: str = "array.npy",
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredObject:
        """Publish a canonical NPY without overwriting an existing object."""

        payload_name = _safe_name(payload_name, "payload filename")
        if not payload_name.endswith(".npy"):
            raise ContentAddressedStoreError("NPY payload filename must end in .npy")
        payload = canonical_npy_bytes(value)
        array, _ = _canonical_npy_from_bytes(payload)
        return self._publish(
            payload,
            object_kind=object_kind,
            payload_name=payload_name,
            payload_format="NPY",
            array=array,
            user_metadata=metadata,
        )

    def put_target_map(
        self, value: np.ndarray, *, metadata: Mapping[str, Any] | None = None
    ) -> StoredObject:
        """Publish the one canonical target representation used by all snapshots."""

        return self.put_npy(
            value,
            object_kind="target_map",
            payload_name="target_points.npy",
            metadata=metadata,
        )

    def put_json(
        self,
        value: Any,
        *,
        object_kind: str = "canonical_json",
        payload_name: str = "value.json",
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredObject:
        payload_name = _safe_name(payload_name, "payload filename")
        if not payload_name.endswith(".json") or payload_name == "metadata.json":
            raise ContentAddressedStoreError("JSON payload filename is unsafe")
        return self._publish(
            canonical_json_bytes(value),
            object_kind=object_kind,
            payload_name=payload_name,
            payload_format="CANONICAL_JSON",
            array=None,
            user_metadata=metadata,
        )

    def get(self, digest: str) -> StoredObject:
        digest = self._validate_digest(digest)
        return self._verify_existing(self._directory(digest), expected_digest=digest)

    def load_npy(self, digest: str) -> np.ndarray:
        stored = self.get(digest)
        if stored.payload_path.suffix != ".npy":
            raise ContentAddressedStoreError("requested object is not NPY")
        array, _payload = _canonical_npy_from_file(stored.payload_path)
        array.setflags(write=False)
        return array

    def load_json(self, digest: str) -> Any:
        stored = self.get(digest)
        if stored.payload_path.suffix != ".json":
            raise ContentAddressedStoreError("requested object is not JSON")
        value = json.loads(stored.payload_path.read_text(encoding="utf-8"))
        if canonical_json_bytes(value) != stored.payload_path.read_bytes():
            raise ContentAddressedStoreError("stored JSON is not canonical")
        return value

    def create_snapshot_reference(
        self,
        snapshot_id: str,
        *,
        target_map_sha256: str,
        canonical_source_sha256: str | None = None,
        extra_bindings: Mapping[str, Any] | None = None,
    ) -> Path:
        """Create an immutable manifest reference without copying target bytes."""

        snapshot = _safe_name(snapshot_id, "snapshot ID")
        target_digest = self._validate_digest(target_map_sha256)
        target = self.get(target_digest)
        if target.object_kind != "target_map" or target.payload_path.name != "target_points.npy":
            raise ContentAddressedStoreError("snapshot target does not name a target-map object")
        source_digest = None
        if canonical_source_sha256 is not None:
            source_digest = self._validate_digest(canonical_source_sha256)
        bindings = dict(extra_bindings or {})
        forbidden = {
            key
            for key in bindings
            if "path" in str(key).lower()
            or "points" in str(key).lower()
            or "payload" in str(key).lower()
        }
        if forbidden:
            raise ContentAddressedStoreError(
                f"snapshot references may not embed/copy target payloads: {sorted(forbidden)}"
            )
        value = {
            "schema": SNAPSHOT_REFERENCE_SCHEMA,
            "snapshot_id": snapshot,
            "target_map_sha256": target_digest,
            "canonical_source_sha256": source_digest,
            "bindings": bindings,
        }
        _safe_existing_directory(self.references_root, create=True)
        path = self.references_root / f"{snapshot}.json"
        _publish_immutable_bytes(path, canonical_json_bytes(value))
        return path

    def audit_snapshot_references(
        self,
        *,
        expected_target_map_sha256: str,
        expected_reference_count: int,
    ) -> dict[str, Any]:
        target_digest = self._validate_digest(expected_target_map_sha256)
        references = (
            sorted(self.references_root.glob("*.json")) if self.references_root.exists() else []
        )
        if len(references) != int(expected_reference_count):
            raise ContentAddressedStoreError("snapshot reference count differs")
        snapshot_ids: set[str] = set()
        for path in references:
            if path.is_symlink() or not path.is_file():
                raise ContentAddressedStoreError(f"unsafe snapshot reference: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json_bytes(value) != path.read_bytes():
                raise ContentAddressedStoreError(f"non-canonical snapshot reference: {path}")
            if value.get("schema") != SNAPSHOT_REFERENCE_SCHEMA:
                raise ContentAddressedStoreError(f"snapshot reference schema differs: {path}")
            if value.get("target_map_sha256") != target_digest:
                raise ContentAddressedStoreError(f"snapshot target digest differs: {path}")
            snapshot_id = str(value.get("snapshot_id", ""))
            if snapshot_id in snapshot_ids or path.name != f"{snapshot_id}.json":
                raise ContentAddressedStoreError(f"duplicate/misnamed snapshot reference: {path}")
            snapshot_ids.add(snapshot_id)
        target_objects: list[StoredObject] = []
        for directory in sorted(self.root.iterdir(), key=lambda item: item.name):
            if SHA256_PATTERN.fullmatch(directory.name) is None:
                continue
            stored = self.get(directory.name)
            if stored.object_kind == "target_map":
                target_objects.append(stored)
        if len(target_objects) != 1 or target_objects[0].sha256 != target_digest:
            raise ContentAddressedStoreError(
                "store must contain exactly one target-map content object"
            )
        physical_count = self.physical_payload_copy_count(target_digest)
        if physical_count != 1:
            raise ContentAddressedStoreError(
                f"target map has {physical_count} physical copies; exactly one is required"
            )
        return {
            "pass": True,
            "unique_target_map_count": len(target_objects),
            "snapshot_target_reference_count": len(references),
            "physical_target_map_copy_count": physical_count,
            "target_map_sha256": target_digest,
        }

    def physical_payload_copy_count(self, digest: str) -> int:
        """Count all physical payload files matching ``digest`` beneath the store."""

        digest = self._validate_digest(digest)
        count = 0
        for path in self.root.rglob("*"):
            if not path.is_file() or path.is_symlink() or path.name == "metadata.json":
                continue
            if sha256_file(path) == digest:
                count += 1
        return count


def _canonical_npy_from_bytes(payload: bytes) -> tuple[np.ndarray, bytes]:
    with io.BytesIO(payload) as stream:
        array = np.load(stream, allow_pickle=False)
        if stream.read(1) != b"":
            raise ContentAddressedStoreError("canonical NPY has trailing bytes")
    if canonical_npy_bytes(array) != payload:
        raise ContentAddressedStoreError("NPY payload is not canonical")
    return array, payload


@contextmanager
def deterministic_temporary_pcl_conversion(
    source_npy: str | Path,
    target_npy: str | Path,
    *,
    temporary_root: str | Path,
) -> Iterator[TemporaryPclConversion]:
    """Derive deterministic PCD files and remove them on every exit path.

    This function only serializes points.  It never invokes the PCL executable
    or any registration entrypoint.
    """

    root = _safe_existing_directory(temporary_root, create=True)
    source_path = Path(source_npy)
    target_path = Path(target_npy)
    source, source_payload = _canonical_npy_from_file(source_path)
    target, target_payload = _canonical_npy_from_file(target_path)
    work = Path(tempfile.mkdtemp(prefix="pcl-derived-", dir=root))
    try:
        source_pcd = work / "source.pcd"
        target_pcd = work / "target.pcd"
        write_binary_xyz_pcd(source_pcd, source)
        write_binary_xyz_pcd(target_pcd, target)
        for path in (source_pcd, target_pcd):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(work)
        yield TemporaryPclConversion(
            directory=work,
            source_path=source_pcd,
            target_path=target_pcd,
            source_pcd_sha256=sha256_file(source_pcd),
            target_pcd_sha256=sha256_file(target_pcd),
            source_npy_sha256=_sha256(source_payload),
            target_npy_sha256=_sha256(target_payload),
        )
    finally:
        if work.exists():
            shutil.rmtree(work)
        _fsync_directory(root)


__all__ = [
    "ContentAddressedStore",
    "ContentAddressedStoreError",
    "StoredObject",
    "TemporaryPclConversion",
    "canonical_npy_bytes",
    "deterministic_temporary_pcl_conversion",
]
