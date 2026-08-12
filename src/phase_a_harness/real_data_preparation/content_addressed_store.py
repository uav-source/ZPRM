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
from .stage2_checkpoint import (
    Stage2CheckpointError,
    cleanup_partial_temporaries,
    initialize_temporary_root,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
OBJECT_SCHEMA = "zprm-stage2-content-addressed-object-v1"
SNAPSHOT_REFERENCE_SCHEMA = "zprm-stage2-snapshot-content-reference-v1"
XYZ_ARRAY_SCHEMA = "zprm-stage2-finite-float64-xyz-v1"


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


def _validated_float64_xyz(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or array.dtype.kind != "f"
        or array.dtype.itemsize != 8
        or not np.all(np.isfinite(array))
    ):
        raise ContentAddressedStoreError(
            f"{label} must be a finite (N, 3) float64 XYZ array"
        )
    return np.ascontiguousarray(array, dtype="<f8")


def _validated_snapshot_bindings(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key, child in value.items():
        key = _safe_name(str(raw_key), "snapshot binding key")
        if isinstance(child, bool) or child is None:
            result[key] = child
        elif isinstance(child, int) and not isinstance(child, bool):
            result[key] = child
        elif isinstance(child, float) and np.isfinite(child):
            result[key] = child
        elif isinstance(child, str) and len(child.encode("utf-8")) <= 512:
            result[key] = child
        else:
            raise ContentAddressedStoreError(
                "snapshot bindings must be small scalar JSON values; containers/payloads are forbidden"
            )
    return result


def _forbidden_reference_binding_paths(value: Any, *, path: str = "bindings") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized = str(key).lower()
            if any(token in normalized for token in ("path", "points", "payload")):
                findings.append(child_path)
            findings.extend(_forbidden_reference_binding_paths(child, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(
                _forbidden_reference_binding_paths(child, path=f"{path}[{index}]")
            )
    return findings


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
        # Validate the reserved scientific roles before creating a temporary
        # directory or publishing anything.  A failing role/name request must
        # never leave an immutable digest directory behind and poison a later
        # correct publication of the same payload.
        reserved = {
            "target_map": ("target_points.npy", "NPY"),
            "canonical_source": ("source_points.npy", "NPY"),
        }
        if object_kind in reserved:
            expected_name, expected_format = reserved[object_kind]
            if (
                payload_name != expected_name
                or payload_format != expected_format
                or array is None
                or user_metadata != {"array_contract_schema": XYZ_ARRAY_SCHEMA}
            ):
                raise ContentAddressedStoreError(
                    f"reserved {object_kind} publication contract differs"
                )
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
        base_fields = {
            "object_kind",
            "payload_filename",
            "payload_format",
            "schema",
            "sha256",
            "size_bytes",
            "user_metadata",
        }
        array_fields = {"allow_pickle", "c_contiguous", "dtype", "npy_version", "shape"}
        expected_fields = (
            base_fields | array_fields
            if metadata.get("payload_format") == "NPY"
            else base_fields
        )
        if set(metadata) != expected_fields:
            raise ContentAddressedStoreError(f"object metadata field set differs: {metadata_path}")
        if canonical_json_bytes(metadata) != metadata_path.read_bytes():
            raise ContentAddressedStoreError(f"object metadata is not canonical: {metadata_path}")
        if metadata.get("sha256") != expected_digest:
            raise ContentAddressedStoreError(f"object digest binding differs: {directory}")
        payload_name = str(metadata.get("payload_filename", ""))
        _safe_name(payload_name, "payload filename")
        if payload_name == "target_points.npy" and metadata.get("object_kind") != "target_map":
            raise ContentAddressedStoreError("target payload object-kind contract differs")
        if (
            payload_name == "source_points.npy"
            and metadata.get("object_kind") != "canonical_source"
        ):
            raise ContentAddressedStoreError("source payload object-kind contract differs")
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
        if metadata.get("object_kind") in {"target_map", "canonical_source"}:
            kind = str(metadata["object_kind"])
            expected_name = (
                "target_points.npy" if kind == "target_map" else "source_points.npy"
            )
            user_metadata = metadata.get("user_metadata")
            if (
                user_metadata != {"array_contract_schema": XYZ_ARRAY_SCHEMA}
                or metadata.get("payload_filename") != expected_name
                or metadata.get("payload_format") != "NPY"
                or metadata.get("dtype") != "<f8"
                or metadata.get("c_contiguous") is not True
                or metadata.get("allow_pickle") is not False
                or metadata.get("npy_version") != "1.0"
            ):
                raise ContentAddressedStoreError("XYZ object array-contract schema differs")
            try:
                xyz_array, canonical_payload = _canonical_npy_from_bytes(actual_payload)
                xyz_array = _validated_float64_xyz(
                    xyz_array, label=str(metadata.get("object_kind"))
                )
            except (ValueError, ContentAddressedStoreError) as exc:
                raise ContentAddressedStoreError("XYZ object payload contract differs") from exc
            if canonical_payload != actual_payload:
                raise ContentAddressedStoreError("XYZ object encoding contract differs")
            if metadata.get("shape") != [int(value) for value in xyz_array.shape]:
                raise ContentAddressedStoreError("XYZ object shape metadata differs")
        entries = list(directory.iterdir())
        if (
            {path.name for path in entries} != {payload_name, "metadata.json"}
            or any(path.is_symlink() or not path.is_file() for path in entries)
        ):
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
        if object_kind in {"target_map", "canonical_source"}:
            value = _validated_float64_xyz(value, label=object_kind)
            # Scientific XYZ object metadata is completely determined by its
            # payload and fixed storage role.  Arbitrary provenance belongs in
            # a separately content-addressed manifest; allowing it here would
            # let a re-signed metadata.json alter scientific bindings while the
            # payload-address directory stayed unchanged.
            if metadata:
                raise ContentAddressedStoreError(
                    "scientific XYZ metadata must be stored in a separate content-addressed manifest"
                )
            metadata = {"array_contract_schema": XYZ_ARRAY_SCHEMA}
            expected_name = (
                "target_points.npy"
                if object_kind == "target_map"
                else "source_points.npy"
            )
            if payload_name != expected_name:
                raise ContentAddressedStoreError(
                    f"reserved {object_kind} payload filename differs"
                )
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
            _validated_float64_xyz(value, label="target_map"),
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
            source = self.get(source_digest)
            if (
                source.object_kind != "canonical_source"
                or source.payload_path.name != "source_points.npy"
            ):
                raise ContentAddressedStoreError(
                    "snapshot source does not name a canonical-source object"
                )
        raw_bindings = dict(extra_bindings or {})
        bindings = _validated_snapshot_bindings(raw_bindings)
        forbidden = _forbidden_reference_binding_paths(raw_bindings)
        if forbidden:
            raise ContentAddressedStoreError(
                f"snapshot references may not contain path/point/payload bindings: {forbidden}"
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
        root_entries = list(self.root.iterdir())
        for entry in root_entries:
            if entry.name == self.references_root.name:
                if entry.is_symlink() or not entry.is_dir():
                    raise ContentAddressedStoreError("snapshot reference root is unsafe")
                continue
            if (
                SHA256_PATTERN.fullmatch(entry.name) is None
                or entry.is_symlink()
                or not entry.is_dir()
            ):
                raise ContentAddressedStoreError(
                    f"unknown entry in content-addressed store root: {entry}"
                )
        if self.references_root.exists():
            if self.references_root.is_symlink() or not self.references_root.is_dir():
                raise ContentAddressedStoreError("snapshot reference root is unsafe")
            entries = sorted(self.references_root.iterdir(), key=lambda item: item.name)
            if any(
                item.is_symlink() or not item.is_file() or item.suffix != ".json"
                for item in entries
            ):
                raise ContentAddressedStoreError("unknown entry in snapshot reference root")
            references = entries
        else:
            references = []
        if len(references) != int(expected_reference_count):
            raise ContentAddressedStoreError("snapshot reference count differs")
        snapshot_ids: set[str] = set()
        referenced_source_digests: set[str] = set()
        for path in references:
            if path.is_symlink() or not path.is_file():
                raise ContentAddressedStoreError(f"unsafe snapshot reference: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json_bytes(value) != path.read_bytes():
                raise ContentAddressedStoreError(f"non-canonical snapshot reference: {path}")
            if value.get("schema") != SNAPSHOT_REFERENCE_SCHEMA:
                raise ContentAddressedStoreError(f"snapshot reference schema differs: {path}")
            if set(value) != {
                "bindings",
                "canonical_source_sha256",
                "schema",
                "snapshot_id",
                "target_map_sha256",
            }:
                raise ContentAddressedStoreError(f"snapshot reference field set differs: {path}")
            if value.get("target_map_sha256") != target_digest:
                raise ContentAddressedStoreError(f"snapshot target digest differs: {path}")
            try:
                bindings = _validated_snapshot_bindings(value.get("bindings", {}))
            except (AttributeError, ContentAddressedStoreError) as exc:
                raise ContentAddressedStoreError(
                    f"snapshot bindings are invalid: {path}"
                ) from exc
            if bindings != value.get("bindings"):
                raise ContentAddressedStoreError(f"snapshot bindings differ: {path}")
            source_digest = value.get("canonical_source_sha256")
            if int(expected_reference_count) == 100 and source_digest is None:
                raise ContentAddressedStoreError(
                    f"published 100-snapshot reference lacks its canonical source: {path}"
                )
            if source_digest is not None:
                source = self.get(self._validate_digest(source_digest))
                if (
                    source.object_kind != "canonical_source"
                    or source.payload_path.name != "source_points.npy"
                    or self.physical_payload_copy_count(source.sha256) != 1
                ):
                    raise ContentAddressedStoreError(
                        f"snapshot source is absent, mis-typed, or duplicated: {path}"
                    )
                referenced_source_digests.add(source.sha256)
            snapshot_id = str(value.get("snapshot_id", ""))
            if snapshot_id in snapshot_ids or path.name != f"{snapshot_id}.json":
                raise ContentAddressedStoreError(f"duplicate/misnamed snapshot reference: {path}")
            snapshot_ids.add(snapshot_id)
        target_objects: list[StoredObject] = []
        source_objects: list[StoredObject] = []
        for directory in sorted(self.root.iterdir(), key=lambda item: item.name):
            if SHA256_PATTERN.fullmatch(directory.name) is None:
                continue
            stored = self.get(directory.name)
            if stored.object_kind == "target_map":
                target_objects.append(stored)
            elif stored.object_kind == "canonical_source":
                source_objects.append(stored)
            else:
                raise ContentAddressedStoreError(
                    f"unexpected content object kind in snapshot store: {stored.object_kind}"
                )
        if len(target_objects) != 1 or target_objects[0].sha256 != target_digest:
            raise ContentAddressedStoreError(
                "store must contain exactly one target-map content object"
            )
        if {stored.sha256 for stored in source_objects} != referenced_source_digests:
            raise ContentAddressedStoreError(
                "snapshot store contains an orphan or missing canonical-source object"
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


def _verify_binary_xyz_pcd_projection(path: Path, canonical: np.ndarray) -> None:
    """Independently reparse the PCD and verify its mandated float32 projection."""

    payload = path.read_bytes()
    marker = b"DATA binary\n"
    split = payload.find(marker)
    if split < 0:
        raise ContentAddressedStoreError("temporary PCD has no binary-data marker")
    header = payload[: split + len(marker)].decode("ascii")
    expected = _validated_float64_xyz(canonical, label="temporary PCD input")
    expected_header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {expected.shape[0]}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {expected.shape[0]}",
        "DATA binary",
    ]
    if header.splitlines() != expected_header_lines:
        raise ContentAddressedStoreError("temporary PCD structural header differs")
    expected_projection = np.ascontiguousarray(expected, dtype="<f4")
    body = payload[split + len(marker) :]
    width_match = re.search(r"(?m)^WIDTH ([0-9]+)$", header)
    points_match = re.search(r"(?m)^POINTS ([0-9]+)$", header)
    if (
        width_match is None
        or points_match is None
        or int(width_match.group(1)) != expected_projection.shape[0]
        or int(points_match.group(1)) != expected_projection.shape[0]
        or len(body) != expected_projection.nbytes
    ):
        raise ContentAddressedStoreError("temporary PCD header/size differs")
    reconstructed = np.frombuffer(body, dtype="<f4").reshape((-1, 3))
    if not np.array_equal(reconstructed, expected_projection):
        raise ContentAddressedStoreError(
            "temporary PCD differs from deterministic float32 projection"
        )


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

    candidate = Path(temporary_root)
    if not candidate.is_absolute():
        raise ContentAddressedStoreError("tmp_pcl root must be absolute")
    try:
        if candidate.exists():
            if candidate.is_symlink() or not candidate.is_dir():
                raise ContentAddressedStoreError("tmp_pcl root is unsafe")
            marker = candidate / ".stage2_temporary_root.json"
            if any(candidate.iterdir()):
                if not marker.is_file() or marker.is_symlink():
                    raise ContentAddressedStoreError("tmp_pcl root has an unknown orphan")
                marker_value = json.loads(marker.read_text(encoding="utf-8"))
                if (
                    set(marker_value)
                    != {"canonical_root", "purpose", "schema", "temporary"}
                    or canonical_json_bytes(marker_value) != marker.read_bytes()
                    or marker_value.get("schema")
                    != "zprm-boreas-stage2-temporary-root-v1"
                    or marker_value.get("temporary") is not True
                    or marker_value.get("purpose") != "tmp_pcl"
                    or marker_value.get("canonical_root") != str(candidate)
                ):
                    raise ContentAddressedStoreError("tmp_pcl marker differs")
                cleanup_partial_temporaries(candidate)
        root = initialize_temporary_root(candidate, purpose="tmp_pcl")
    except (Stage2CheckpointError, json.JSONDecodeError, OSError) as exc:
        raise ContentAddressedStoreError(f"tmp_pcl root is untrusted: {exc}") from exc
    def require_cas_role(
        raw_path: str | Path, *, expected_kind: str, expected_name: str
    ) -> Path:
        path = Path(raw_path)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or path.name != expected_name
            or SHA256_PATTERN.fullmatch(path.parent.name) is None
        ):
            raise ContentAddressedStoreError(
                f"PCL {expected_kind} input is not a canonical CAS payload path"
            )
        stored = ContentAddressedStore(path.parent.parent).get(path.parent.name)
        if (
            stored.object_kind != expected_kind
            or stored.payload_path != path
            or stored.sha256 != _sha256(path.read_bytes())
        ):
            raise ContentAddressedStoreError(
                f"PCL {expected_kind} input role or digest differs"
            )
        return path

    source_path = require_cas_role(
        source_npy,
        expected_kind="canonical_source",
        expected_name="source_points.npy",
    )
    target_path = require_cas_role(
        target_npy,
        expected_kind="target_map",
        expected_name="target_points.npy",
    )
    source, source_payload = _canonical_npy_from_file(source_path)
    target, target_payload = _canonical_npy_from_file(target_path)
    source = _validated_float64_xyz(source, label="PCL canonical source")
    target = _validated_float64_xyz(target, label="PCL canonical target")
    work = Path(tempfile.mkdtemp(prefix="pcl-derived-", dir=root))
    try:
        source_pcd = work / "source.pcd"
        target_pcd = work / "target.pcd"
        write_binary_xyz_pcd(source_pcd, source)
        write_binary_xyz_pcd(target_pcd, target)
        _verify_binary_xyz_pcd_projection(source_pcd, source)
        _verify_binary_xyz_pcd_projection(target_pcd, target)
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
