"""Two-pass, registration-blind query screening with bounded raw storage.

The first pass persists only authenticated object identity and geometry-only
metrics.  Selection is an injected, pre-registered hook: this module records
its output but intentionally defines no weak/rich threshold.  The second pass
materializes byte-deterministic canonical sources for exactly the frozen 100
snapshots.

``STREAMING_LOW_DISK`` and ``FULL_RAW_CACHE`` differ only in raw-object
lifecycle.  Raw-cache paths and execution mode are excluded from every
scientific row so semantic equality can be checked exactly.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Sequence, Tuple

import numpy as np

from .io import atomic_write_bytes, atomic_write_json, canonical_json_bytes
from .selection import (
    FORBIDDEN_SELECTOR_FRAGMENTS,
    GEOMETRY_ONLY_FIELDS,
    geometry_only_view,
)
from .stage2_checkpoint import (
    CheckpointRecord,
    Stage2CheckpointError,
    Stage2CheckpointLog,
    cleanup_partial_temporaries,
    initialize_temporary_root,
)
from .streaming_target_map import canonical_array_sha256


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FIRST_PASS_SCHEMA = "zprm.stage2.streaming_query_first_pass.v1"
SELECTION_SCHEMA = "zprm.stage2.blind_selection_record.v1"
SELECTED_SOURCE_SCHEMA = "zprm.stage2.selected_canonical_sources.v1"
WEAK_LABEL = "CORRIDOR_OR_WEAK_GEOMETRY"
RICH_LABEL = "GEOMETRY_RICH"
EXPECTED_INTERVALS_PER_LABEL = 10
QUERY_SCREEN_AUXILIARY_FIELDS = frozenset(
    {
        "finite_source_point_count",
        "target_map_point_count",
        "reference_interpolation_valid",
        "gt_gap_within_limit",
        "target_map_coverage_valid",
        "deskew_uncertainty_within_limit",
    }
)
EXPECTED_SNAPSHOTS_PER_INTERVAL = 5
EXPECTED_SNAPSHOTS_PER_LABEL = 50
EXPECTED_SNAPSHOT_COUNT = 100


class StreamingQueryError(RuntimeError):
    """Raised when streaming query or blind-freeze semantics are violated."""


class QueryExecutionMode(str, Enum):
    STREAMING_LOW_DISK = "STREAMING_LOW_DISK"
    FULL_RAW_CACHE = "FULL_RAW_CACHE"


def _require_sha256(value: str, *, field: str) -> str:
    text = str(value).lower()
    if SHA256_RE.fullmatch(text) is None:
        raise StreamingQueryError(f"{field} must be a lowercase SHA-256")
    return text


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StreamingQueryError("scientific rows cannot contain NaN or infinity")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise StreamingQueryError(f"value is not canonical JSON data: {type(value).__name__}")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class QueryObject:
    ordinal: int
    object_key: str
    remote_size_bytes: int
    etag: str
    last_modified: str
    object_sha256: str | None = None

    def validated(self) -> "QueryObject":
        ordinal = int(self.ordinal)
        if ordinal < 0:
            raise StreamingQueryError("query object ordinal must be nonnegative")
        if not self.object_key or self.object_key.startswith("/"):
            raise StreamingQueryError("object_key must be a nonempty relative key")
        size = int(self.remote_size_bytes)
        if size < 0:
            raise StreamingQueryError("remote_size_bytes must be nonnegative")
        checksum = self.object_sha256
        if checksum is not None:
            checksum = _require_sha256(checksum, field="object_sha256")
        return QueryObject(
            ordinal=ordinal,
            object_key=str(self.object_key),
            remote_size_bytes=size,
            etag=str(self.etag),
            last_modified=str(self.last_modified),
            object_sha256=checksum,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "etag": self.etag,
            "last_modified": self.last_modified,
            "object_key": self.object_key,
            "ordinal": self.ordinal,
            "remote_size_bytes": self.remote_size_bytes,
        }


PayloadLoader = Callable[[QueryObject], bytes]
PointDecoder = Callable[[Path], np.ndarray]
GeometryMetricFunction = Callable[[np.ndarray, QueryObject], Mapping[str, Any]]
SelectionHook = Callable[[Tuple[Dict[str, Any], ...]], Sequence[Mapping[str, Any]]]
Canonicalizer = Callable[[np.ndarray, QueryObject], np.ndarray]


def _validated_objects(objects: Sequence[QueryObject]) -> list[QueryObject]:
    rows = [item.validated() for item in objects]
    if [row.ordinal for row in rows] != list(range(len(rows))):
        raise StreamingQueryError("query object input order must be explicit ordinals 0..N-1")
    keys = [row.object_key for row in rows]
    if len(keys) != len(set(keys)):
        raise StreamingQueryError("query object keys must be unique")
    return rows


def _validate_workspace(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.exists() and candidate.is_symlink():
        raise StreamingQueryError("query workspace cannot be a symlink")
    candidate.mkdir(parents=True, exist_ok=True)
    root = candidate.resolve(strict=True)
    if root != candidate or root.is_symlink():
        raise StreamingQueryError("query workspace must be canonical and cannot be a symlink")
    return root


def _marked_temporary_root(workspace: Path, purpose: str) -> Path:
    """Initialize or fail-closed clean one disposable production workspace."""

    root = workspace / purpose
    try:
        if root.exists():
            if root.is_symlink() or not root.is_dir():
                raise StreamingQueryError(f"unsafe {purpose} temporary root")
            marker = root / ".stage2_temporary_root.json"
            if any(root.iterdir()):
                if not marker.is_file() or marker.is_symlink():
                    raise StreamingQueryError(
                        f"unknown orphan in unmarked {purpose} temporary root"
                    )
                try:
                    marker_value = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise StreamingQueryError(
                        f"invalid {purpose} temporary-root marker"
                    ) from exc
                expected_marker_keys = {
                    "canonical_root",
                    "purpose",
                    "schema",
                    "temporary",
                }
                if (
                    not isinstance(marker_value, dict)
                    or set(marker_value) != expected_marker_keys
                    or canonical_json_bytes(marker_value) != marker.read_bytes()
                    or marker_value.get("schema")
                    != "zprm-boreas-stage2-temporary-root-v1"
                    or marker_value.get("temporary") is not True
                    or marker_value.get("purpose") != purpose
                    or marker_value.get("canonical_root") != str(root)
                ):
                    raise StreamingQueryError(
                        f"{purpose} temporary-root marker purpose/identity differs"
                    )
                cleanup_partial_temporaries(root)
        return initialize_temporary_root(root, purpose=purpose)
    except Stage2CheckpointError as exc:
        raise StreamingQueryError(f"unsafe {purpose} temporary root: {exc}") from exc


def _initialize_query_temporary_roots(workspace: Path) -> dict[str, Path]:
    return {
        purpose: _marked_temporary_root(workspace, purpose)
        for purpose in ("tmp_download", "tmp_decode")
    }


def _payload_cache_path(workspace: Path, sha256: str) -> Path:
    root = workspace / "raw_cache"
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise StreamingQueryError("raw-cache root is unsafe")
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve(strict=True) != root:
        raise StreamingQueryError("raw-cache root is not canonical")
    prefix = root / sha256[:2]
    if prefix.exists() and (prefix.is_symlink() or not prefix.is_dir()):
        raise StreamingQueryError("raw-cache prefix is unsafe")
    prefix.mkdir(parents=False, exist_ok=True)
    if prefix.resolve(strict=True) != prefix:
        raise StreamingQueryError("raw-cache prefix is not canonical")
    return prefix / f"{sha256}.bin"


def _load_payload(
    query_object: QueryObject,
    *,
    payload_loader: PayloadLoader,
    mode: QueryExecutionMode,
    workspace: Path,
    known_sha256: str | None = None,
) -> tuple[bytes, str]:
    expected_sha = known_sha256 or query_object.object_sha256
    cache_path = (
        _payload_cache_path(workspace, expected_sha)
        if mode is QueryExecutionMode.FULL_RAW_CACHE and expected_sha is not None
        else None
    )
    if mode is QueryExecutionMode.FULL_RAW_CACHE and cache_path is not None and cache_path.exists():
        if cache_path.is_symlink() or not cache_path.is_file():
            raise StreamingQueryError("raw-cache payload path is unsafe")
        payload = cache_path.read_bytes()
    else:
        payload = payload_loader(query_object)
        if not isinstance(payload, bytes):
            raise StreamingQueryError("payload_loader must return bytes")
    if len(payload) != query_object.remote_size_bytes:
        raise StreamingQueryError(
            f"remote size mismatch for {query_object.object_key}: "
            f"{len(payload)} != {query_object.remote_size_bytes}"
        )
    actual_sha = hashlib.sha256(payload).hexdigest()
    if expected_sha is not None and actual_sha != expected_sha:
        raise StreamingQueryError(f"payload SHA changed for {query_object.object_key}")
    if mode is QueryExecutionMode.FULL_RAW_CACHE:
        cache_path = _payload_cache_path(workspace, actual_sha)
        if cache_path.exists() and (cache_path.is_symlink() or not cache_path.is_file()):
            raise StreamingQueryError("raw-cache payload path is unsafe")
        atomic_write_bytes(cache_path, payload)
    return payload, actual_sha


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _decode_from_temporary(
    payload: bytes,
    *,
    decoder: PointDecoder,
    temporary_roots: Mapping[str, Path],
    ordinal: int,
) -> Iterator[np.ndarray]:
    """Keep raw bytes live until the caller commits its persistent output."""

    temp_root = temporary_roots["tmp_download"]
    if temporary_roots["tmp_decode"].name != "tmp_decode":
        raise StreamingQueryError("decoded temporary-root purpose differs")
    temporary = temp_root / f"query-{ordinal:08d}.bin"
    if temporary.exists():
        raise StreamingQueryError("orphan query temporary survived initialization")
    try:
        atomic_write_bytes(temporary, payload)
        points = np.asarray(decoder(temporary), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise StreamingQueryError("decoder must return an (N, 3) point array")
        if not np.all(np.isfinite(points)):
            raise StreamingQueryError("decoded query points must be finite")
        yield np.ascontiguousarray(points, dtype="<f8")
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(temp_root)


def _metric_row(
    metrics: Mapping[str, Any],
    *,
    query_object: QueryObject,
    payload_sha256: str,
    target_map_sha256: str,
    gt_sha256: str,
    calibration_sha256: str,
    processing_contract_sha256: str,
) -> dict[str, Any]:
    # ``geometry_only_view`` both requires the frozen metric vocabulary and
    # rejects fields whose names disclose registration outcomes.
    try:
        approved_metrics = geometry_only_view(metrics)
    except (KeyError, TypeError, ValueError) as exc:
        raise StreamingQueryError(f"invalid geometry-only metric row: {exc}") from exc
    forbidden = sorted(
        str(key)
        for key in metrics
        if any(fragment in str(key).lower() for fragment in FORBIDDEN_SELECTOR_FRAGMENTS)
    )
    if forbidden:
        raise StreamingQueryError(f"registration-derived geometry fields: {forbidden}")
    unexpected = sorted(
        set(metrics) - set(GEOMETRY_ONLY_FIELDS) - QUERY_SCREEN_AUXILIARY_FIELDS
    )
    if unexpected:
        raise StreamingQueryError(f"non-allowlisted geometry fields: {unexpected}")
    # Persist only the explicit scientific allowlist.  Merely checking a few
    # forbidden-name fragments is insufficient to protect blind selection.
    native_metrics = _json_native(
        {
            **approved_metrics,
            **{
                field: metrics[field]
                for field in sorted(QUERY_SCREEN_AUXILIARY_FIELDS)
                if field in metrics
            },
        }
    )
    core = {
        **query_object.identity(),
        "calibration_sha256": calibration_sha256,
        "geometry_metrics": native_metrics,
        "gt_sha256": gt_sha256,
        "local_temporary_sha256": payload_sha256,
        "processing_contract_sha256": processing_contract_sha256,
        "schema": FIRST_PASS_SCHEMA,
        "target_map_sha256": target_map_sha256,
    }
    return {**core, "geometry_row_sha256": _hash_json(core)}


def _validate_existing_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    objects: Sequence[QueryObject],
    *,
    target_map_sha256: str,
    gt_sha256: str,
    calibration_sha256: str,
    processing_contract_sha256: str,
) -> list[dict[str, Any]]:
    if len(existing_rows) > len(objects):
        raise StreamingQueryError("orphan query checkpoint contains unknown objects")
    result: list[dict[str, Any]] = []
    for expected_ordinal, source in enumerate(existing_rows):
        row = _json_native(source)
        if not isinstance(row, dict):
            raise StreamingQueryError("query checkpoint rows must be objects")
        query_object = objects[expected_ordinal]
        if row.get("ordinal") != expected_ordinal:
            raise StreamingQueryError("orphan or non-contiguous query checkpoint")
        expected_identity = query_object.identity()
        actual_identity = {key: row.get(key) for key in expected_identity}
        if actual_identity != expected_identity:
            raise StreamingQueryError(
                f"authenticated query object changed on resume at ordinal {expected_ordinal}"
            )
        for field, expected in (
            ("target_map_sha256", target_map_sha256),
            ("gt_sha256", gt_sha256),
            ("calibration_sha256", calibration_sha256),
            ("processing_contract_sha256", processing_contract_sha256),
        ):
            if row.get(field) != expected:
                raise StreamingQueryError(f"query checkpoint {field} changed")
        claimed = row.get("geometry_row_sha256")
        core = {key: value for key, value in row.items() if key != "geometry_row_sha256"}
        if claimed != _hash_json(core):
            raise StreamingQueryError("query checkpoint geometry-row SHA mismatch")
        try:
            approved = geometry_only_view(row.get("geometry_metrics", {}))
        except (KeyError, TypeError, ValueError) as exc:
            raise StreamingQueryError("invalid geometry metrics in query checkpoint") from exc
        metrics = row.get("geometry_metrics", {})
        if set(metrics) - set(GEOMETRY_ONLY_FIELDS) - QUERY_SCREEN_AUXILIARY_FIELDS:
            raise StreamingQueryError("query checkpoint contains non-allowlisted metrics")
        result.append(row)
    return result


def _query_geometry_root(checkpoint_path: Path) -> Path:
    workspace = (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == "checkpoints"
        else checkpoint_path.parent
    )
    return workspace / "geometry_metrics" / "rows_by_sha256"


def _safe_persistent_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise StreamingQueryError(f"{label} must be absolute")
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir() or cursor.resolve(strict=True) != cursor:
        raise StreamingQueryError(f"{label} ancestor is unsafe")
    for directory in reversed(missing):
        directory.mkdir()
    cursor = path
    while True:
        if cursor.is_symlink() or not cursor.is_dir():
            raise StreamingQueryError(f"{label} contains an unsafe component")
        if cursor == path.anchor or cursor.parent == cursor:
            break
        cursor = cursor.parent
    if path.resolve(strict=True) != path:
        raise StreamingQueryError(f"{label} is not canonical")
    return path


def _checkpoint_path_within_workspace(
    value: str | Path | None,
    *,
    workspace: Path,
    default_name: str,
) -> Path:
    candidate = workspace / "checkpoints" / default_name if value is None else Path(value)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    if candidate.name != default_name:
        raise StreamingQueryError(f"checkpoint filename must be {default_name}")
    checkpoints = _safe_persistent_directory(
        workspace / "checkpoints", label="query checkpoint root"
    )
    if candidate.parent != checkpoints:
        raise StreamingQueryError("query checkpoint must be directly under workspace/checkpoints")
    if candidate.exists() and (candidate.is_symlink() or not candidate.is_file()):
        raise StreamingQueryError("query checkpoint path is unsafe")
    return candidate


def _persist_query_geometry_row(checkpoint_path: Path, row: Mapping[str, Any]) -> Path:
    digest = _require_sha256(str(row.get("geometry_row_sha256", "")), field="geometry_row_sha256")
    root = _safe_persistent_directory(
        _query_geometry_root(checkpoint_path), label="query geometry-row root"
    )
    destination = root / f"{digest}.json"
    atomic_write_json(destination, _json_native(row))
    return destination


def _load_authenticated_query_rows(
    checkpoint: Stage2CheckpointLog,
    objects: Sequence[QueryObject],
) -> list[dict[str, Any]]:
    records = checkpoint.logical_records()
    if len(records) > len(objects):
        raise StreamingQueryError("orphan query checkpoint contains unknown objects")
    geometry_root = _query_geometry_root(checkpoint.path)
    expected_files: set[str] = set()
    rows: list[dict[str, Any]] = []
    for expected_ordinal, record in enumerate(records):
        query_object = objects[expected_ordinal]
        if record.record_kind != "QUERY" or record.s3_key != query_object.object_key:
            raise StreamingQueryError("orphan or non-contiguous query checkpoint")
        normalized_etag = query_object.etag.strip().strip('"')
        if (
            record.remote_size_bytes != query_object.remote_size_bytes
            or record.etag != normalized_etag
            or record.last_modified != query_object.last_modified
        ):
            raise StreamingQueryError(
                f"authenticated query object changed on resume at ordinal {expected_ordinal}"
            )
        digest = _require_sha256(
            str(record.result_geometry_row_sha256), field="result_geometry_row_sha256"
        )
        expected_files.add(f"{digest}.json")
        row_path = geometry_root / f"{digest}.json"
        if not row_path.is_file() or row_path.is_symlink():
            raise StreamingQueryError("query checkpoint geometry row is missing or unsafe")
        try:
            row = json.loads(row_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StreamingQueryError("query checkpoint geometry row is invalid") from exc
        if canonical_json_bytes(row) != row_path.read_bytes():
            raise StreamingQueryError("query checkpoint geometry row is not canonical")
        if not isinstance(row, dict) or row.get("geometry_row_sha256") != digest:
            raise StreamingQueryError("query checkpoint geometry-row binding differs")
        if row.get("local_temporary_sha256") != record.local_temporary_sha256:
            raise StreamingQueryError("query checkpoint payload SHA binding differs")
        rows.append(row)
    if geometry_root.exists():
        if geometry_root.is_symlink() or not geometry_root.is_dir():
            raise StreamingQueryError("query geometry-row root is unsafe")
        actual_files = {item.name for item in geometry_root.iterdir() if item.is_file()}
        if actual_files != expected_files:
            raise StreamingQueryError("orphan or missing query geometry-row object")
        if any(item.is_symlink() or not item.is_file() for item in geometry_root.iterdir()):
            raise StreamingQueryError("unknown entry in query geometry-row root")
    elif records:
        raise StreamingQueryError("query checkpoint geometry-row root is absent")
    return rows


def first_pass_screen(
    objects: Sequence[QueryObject],
    *,
    payload_loader: PayloadLoader,
    decoder: PointDecoder,
    geometry_metric: GeometryMetricFunction,
    target_map_sha256: str,
    gt_sha256: str,
    calibration_sha256: str,
    processing_contract_sha256: str,
    workspace: str | Path,
    mode: QueryExecutionMode | str,
    existing_rows: Sequence[Mapping[str, Any]] = (),
    processed_jsonl_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Commit each blind metric row before deleting its temporary raw object."""

    try:
        execution_mode = QueryExecutionMode(mode)
    except ValueError as exc:
        raise StreamingQueryError(f"unsupported query execution mode: {mode}") from exc
    target_map_sha256 = _require_sha256(target_map_sha256, field="target_map_sha256")
    gt_sha256 = _require_sha256(gt_sha256, field="gt_sha256")
    calibration_sha256 = _require_sha256(
        calibration_sha256, field="calibration_sha256"
    )
    processing_contract_sha256 = _require_sha256(
        processing_contract_sha256, field="processing_contract_sha256"
    )
    validated = _validated_objects(objects)
    root = _validate_workspace(workspace)
    temporary_roots = _initialize_query_temporary_roots(root)
    manifest_path = _checkpoint_path_within_workspace(
        processed_jsonl_path,
        workspace=root,
        default_name="processed_query_objects.jsonl",
    )
    try:
        checkpoint = Stage2CheckpointLog(
            manifest_path,
            processing_contract_sha256=processing_contract_sha256,
            gt_sha256=gt_sha256,
            calibration_sha256=calibration_sha256,
            expected_record_kinds={"QUERY"},
        )
        persisted_rows = _load_authenticated_query_rows(checkpoint, validated)
    except Stage2CheckpointError as exc:
        raise StreamingQueryError(f"query checkpoint is untrusted: {exc}") from exc
    if existing_rows and not persisted_rows:
        raise StreamingQueryError("existing query rows lack an authenticated checkpoint")
    if existing_rows and _json_native(existing_rows) != persisted_rows:
        raise StreamingQueryError("provided and persisted query checkpoints differ")
    rows = _validate_existing_rows(
        persisted_rows,
        validated,
        target_map_sha256=target_map_sha256,
        gt_sha256=gt_sha256,
        calibration_sha256=calibration_sha256,
        processing_contract_sha256=processing_contract_sha256,
    )

    for query_object in validated[len(rows) :]:
        payload, payload_sha = _load_payload(
            query_object,
            payload_loader=payload_loader,
            mode=execution_mode,
            workspace=root,
        )
        with _decode_from_temporary(
            payload,
            decoder=decoder,
            temporary_roots=temporary_roots,
            ordinal=query_object.ordinal,
        ) as points:
            metrics = geometry_metric(points, query_object)
            row = _metric_row(
                metrics,
                query_object=query_object,
                payload_sha256=payload_sha,
                target_map_sha256=target_map_sha256,
                gt_sha256=gt_sha256,
                calibration_sha256=calibration_sha256,
                processing_contract_sha256=processing_contract_sha256,
            )
            # Commit the immutable scientific row first, then append+fsync its
            # authenticated chain record.  Raw bytes remain until both steps
            # succeed.  An orphan row is rejected rather than auto-adopted.
            _persist_query_geometry_row(manifest_path, row)
            checkpoint.append_completed(
                CheckpointRecord(
                    record_kind="QUERY",
                    s3_key=query_object.object_key,
                    remote_size_bytes=query_object.remote_size_bytes,
                    etag=query_object.etag,
                    last_modified=query_object.last_modified,
                    local_temporary_sha256=payload_sha,
                    processing_contract_sha256=processing_contract_sha256,
                    gt_sha256=gt_sha256,
                    calibration_sha256=calibration_sha256,
                    result_geometry_row_sha256=row["geometry_row_sha256"],
                    map_state_transition_sha256=None,
                    completed_at_utc=datetime.now(timezone.utc)
                    .isoformat(timespec="microseconds")
                    .replace("+00:00", "Z"),
                )
            )
            rows.append(row)
    return rows


def assert_100_snapshot_contract(selected: Sequence[Mapping[str, Any]]) -> None:
    """Enforce the already-frozen 10 x 5 weak and 10 x 5 rich design."""

    if len(selected) != EXPECTED_SNAPSHOT_COUNT:
        raise StreamingQueryError(
            f"snapshot count changed: {len(selected)} != {EXPECTED_SNAPSHOT_COUNT}"
        )
    keys = [str(row.get("object_key", "")) for row in selected]
    if not all(keys) or len(keys) != len(set(keys)):
        raise StreamingQueryError("selected snapshot object keys must be nonempty and unique")
    labels = Counter(str(row.get("scene_label", "")) for row in selected)
    expected_labels = {
        WEAK_LABEL: EXPECTED_SNAPSHOTS_PER_LABEL,
        RICH_LABEL: EXPECTED_SNAPSHOTS_PER_LABEL,
    }
    if labels != expected_labels:
        raise StreamingQueryError(f"weak/rich snapshot counts changed: {dict(labels)}")
    for label in (WEAK_LABEL, RICH_LABEL):
        interval_counts = Counter(
            str(row.get("interval_id", ""))
            for row in selected
            if row.get("scene_label") == label
        )
        if "" in interval_counts or len(interval_counts) != EXPECTED_INTERVALS_PER_LABEL:
            raise StreamingQueryError(f"{label} must contain exactly 10 named intervals")
        if set(interval_counts.values()) != {EXPECTED_SNAPSHOTS_PER_INTERVAL}:
            raise StreamingQueryError(f"{label} must contain exactly 5 snapshots per interval")


def freeze_selection_record(
    first_pass_rows: Sequence[Mapping[str, Any]],
    *,
    selector: SelectionHook,
    selection_contract_sha256: str,
) -> dict[str, Any]:
    """Invoke an external blind selector and bind its output without policy invention."""

    selection_contract_sha256 = _require_sha256(
        selection_contract_sha256, field="selection_contract_sha256"
    )
    rows = [_json_native(row) for row in first_pass_rows]
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise StreamingQueryError("first-pass rows must be objects")
        try:
            approved = geometry_only_view(row.get("geometry_metrics", {}))
        except (KeyError, TypeError, ValueError) as exc:
            raise StreamingQueryError("invalid first-pass geometry metrics") from exc
        metrics = row.get("geometry_metrics", {})
        if set(metrics) - set(GEOMETRY_ONLY_FIELDS) - QUERY_SCREEN_AUXILIARY_FIELDS:
            raise StreamingQueryError("first-pass row contains non-allowlisted metrics")
        key = str(row.get("object_key", ""))
        if not key or key in by_key:
            raise StreamingQueryError("first-pass object keys must be nonempty and unique")
        core = {field: value for field, value in row.items() if field != "geometry_row_sha256"}
        if row.get("geometry_row_sha256") != _hash_json(core):
            raise StreamingQueryError("first-pass row SHA mismatch before selection")
        by_key[key] = row

    # JSON round trips provide the hook an isolated, registration-blind view.
    hook_input = tuple(
        {
            "object_key": row["object_key"],
            "ordinal": row["ordinal"],
            **json.loads(json.dumps(row["geometry_metrics"])),
        }
        for row in rows
    )
    raw_selected = selector(hook_input)
    selected: list[dict[str, Any]] = []
    for selection_index, raw in enumerate(raw_selected):
        value = _json_native(raw)
        if not isinstance(value, dict):
            raise StreamingQueryError("selector output rows must be objects")
        key = str(value.get("object_key", ""))
        if key not in by_key:
            raise StreamingQueryError(f"selector returned unknown object: {key}")
        allowed_fields = {"object_key", "scene_label", "interval_id"}
        unexpected = sorted(set(value) - allowed_fields)
        if unexpected:
            raise StreamingQueryError(
                f"selector output contains unregistered fields: {unexpected}"
            )
        source = by_key[key]
        selected.append(
            {
                "calibration_sha256": source["calibration_sha256"],
                "etag": source["etag"],
                "first_pass_geometry_row_sha256": source["geometry_row_sha256"],
                "gt_sha256": source["gt_sha256"],
                "interval_id": str(value.get("interval_id", "")),
                "last_modified": source["last_modified"],
                "local_temporary_sha256": source["local_temporary_sha256"],
                "object_key": key,
                "ordinal": source["ordinal"],
                "remote_size_bytes": source["remote_size_bytes"],
                "scene_label": str(value.get("scene_label", "")),
                "selection_index": selection_index,
                "target_map_sha256": source["target_map_sha256"],
            }
        )
    assert_100_snapshot_contract(selected)
    core = {
        "first_pass_rows_sha256": _hash_json(rows),
        "policy_note": (
            "SELECTION_THRESHOLDS_ARE_NOT_DEFINED_BY_THE_STORAGE_ARCHITECTURE"
        ),
        "schema": SELECTION_SCHEMA,
        "selected_snapshots": selected,
        "selection_contract_sha256": selection_contract_sha256,
        "snapshot_count": EXPECTED_SNAPSHOT_COUNT,
    }
    return {**core, "selection_record_sha256": _hash_json(core)}


def _verify_selection_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = _json_native(record)
    if not isinstance(value, dict) or value.get("schema") != SELECTION_SCHEMA:
        raise StreamingQueryError("invalid blind selection record schema")
    claimed = value.get("selection_record_sha256")
    core = {key: child for key, child in value.items() if key != "selection_record_sha256"}
    if claimed != _hash_json(core):
        raise StreamingQueryError("blind selection record SHA mismatch")
    selected = value.get("selected_snapshots")
    if not isinstance(selected, list):
        raise StreamingQueryError("selection record snapshots must be a list")
    assert_100_snapshot_contract(selected)
    if [row.get("selection_index") for row in selected] != list(
        range(EXPECTED_SNAPSHOT_COUNT)
    ):
        raise StreamingQueryError("selection indices must be contiguous and immutable")
    return selected


def deterministic_npy_bytes(points: Any) -> tuple[np.ndarray, bytes, str]:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
        raise StreamingQueryError("canonical source must be a finite (N, 3) array")
    value = np.ascontiguousarray(value, dtype="<f8")
    value[value == 0.0] = 0.0
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=False)
    payload = stream.getvalue()
    return value, payload, hashlib.sha256(payload).hexdigest()


def _selected_row_root(output_root: Path) -> Path:
    return output_root / "manifests" / "selected_source_rows_by_sha256"


def _persist_selected_source_row(output_root: Path, row: Mapping[str, Any]) -> Path:
    digest = _require_sha256(
        str(row.get("selected_source_row_sha256", "")),
        field="selected_source_row_sha256",
    )
    root = _safe_persistent_directory(
        _selected_row_root(output_root), label="selected-source row root"
    )
    destination = root / f"{digest}.json"
    atomic_write_json(destination, _json_native(row))
    return destination


def _load_authenticated_selected_source_rows(
    checkpoint: Stage2CheckpointLog,
    selected: Sequence[Mapping[str, Any]],
    objects_by_key: Mapping[str, QueryObject],
    *,
    output_root: Path,
    canonicalization_contract_sha256: str,
    selection_record_sha256: str,
) -> list[dict[str, Any]]:
    records = checkpoint.logical_records()
    if len(records) > len(selected):
        raise StreamingQueryError("orphan selected-source checkpoint exceeds selection")
    row_root = _selected_row_root(output_root)
    expected_row_files: set[str] = set()
    expected_source_digests: set[str] = set()
    rows: list[dict[str, Any]] = []
    for selection_index, record in enumerate(records):
        frozen = selected[selection_index]
        key = str(frozen["object_key"])
        query_object = objects_by_key.get(key)
        if query_object is None:
            raise StreamingQueryError("selected-source checkpoint object left inventory")
        if record.record_kind != "SELECTED_SOURCE" or record.s3_key != key:
            raise StreamingQueryError("selected-source checkpoint is not a frozen prefix")
        normalized_etag = query_object.etag.strip().strip('"')
        if (
            record.remote_size_bytes != query_object.remote_size_bytes
            or record.etag != normalized_etag
            or record.last_modified != query_object.last_modified
            or record.local_temporary_sha256 != frozen["local_temporary_sha256"]
        ):
            raise StreamingQueryError(f"selected object identity changed: {key}")
        digest = _require_sha256(
            str(record.result_geometry_row_sha256),
            field="selected_source_result_row_sha256",
        )
        expected_row_files.add(f"{digest}.json")
        row_path = row_root / f"{digest}.json"
        if not row_path.is_file() or row_path.is_symlink():
            raise StreamingQueryError("selected-source checkpoint row is missing or unsafe")
        try:
            row = json.loads(row_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StreamingQueryError("selected-source checkpoint row is invalid") from exc
        if canonical_json_bytes(row) != row_path.read_bytes():
            raise StreamingQueryError("selected-source checkpoint row is not canonical")
        claimed = row.get("selected_source_row_sha256")
        core = {field: value for field, value in row.items() if field != "selected_source_row_sha256"}
        if claimed != digest or _hash_json(core) != digest:
            raise StreamingQueryError("selected-source checkpoint row SHA differs")
        if (
            row.get("selection_index") != selection_index
            or row.get("object_key") != key
            or row.get("payload_sha256") != frozen["local_temporary_sha256"]
            or row.get("canonicalization_contract_sha256")
            != canonicalization_contract_sha256
            or row.get("selection_record_sha256") != selection_record_sha256
        ):
            raise StreamingQueryError("selected-source checkpoint row binding differs")
        source_sha = _require_sha256(
            str(row.get("canonical_source_sha256", "")),
            field="canonical_source_sha256",
        )
        expected_source_digests.add(source_sha)
        relative = Path(str(row.get("relative_path", "")))
        expected_relative = Path("selected_sources") / source_sha / "source_points.npy"
        if relative != expected_relative:
            raise StreamingQueryError("selected-source checkpoint path differs")
        source_path = output_root / relative
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha
        ):
            raise StreamingQueryError("selected-source canonical NPY differs")
        rows.append(row)
    if row_root.exists():
        if row_root.is_symlink() or not row_root.is_dir():
            raise StreamingQueryError("selected-source row root is unsafe")
        entries = list(row_root.iterdir())
        if any(item.is_symlink() or not item.is_file() for item in entries):
            raise StreamingQueryError("unknown entry in selected-source row root")
        if {item.name for item in entries} != expected_row_files:
            raise StreamingQueryError("orphan or missing selected-source row")
    elif records:
        raise StreamingQueryError("selected-source row root is absent")
    sources_root = output_root / "selected_sources"
    if sources_root.exists():
        if sources_root.is_symlink() or not sources_root.is_dir():
            raise StreamingQueryError("selected-source content root is unsafe")
        directories = list(sources_root.iterdir())
        if any(item.is_symlink() or not item.is_dir() for item in directories):
            raise StreamingQueryError("unknown entry in selected-source content root")
        if {item.name for item in directories} != expected_source_digests:
            raise StreamingQueryError("orphan or missing selected-source content object")
        for directory in directories:
            entries = list(directory.iterdir())
            if (
                len(entries) != 1
                or entries[0].name != "source_points.npy"
                or entries[0].is_symlink()
                or not entries[0].is_file()
            ):
                raise StreamingQueryError("selected-source content inventory differs")
    elif records:
        raise StreamingQueryError("selected-source content root is absent")
    return rows


def second_pass_materialize_selected(
    selection_record: Mapping[str, Any],
    objects: Sequence[QueryObject],
    *,
    payload_loader: PayloadLoader,
    decoder: PointDecoder,
    canonicalizer: Canonicalizer,
    canonicalization_contract_sha256: str,
    workspace: str | Path,
    output_root: str | Path,
    mode: QueryExecutionMode | str,
    processed_jsonl_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reacquire/cache-read only the frozen 100 objects and write canonical NPYs."""

    try:
        execution_mode = QueryExecutionMode(mode)
    except ValueError as exc:
        raise StreamingQueryError(f"unsupported query execution mode: {mode}") from exc
    canonicalization_contract_sha256 = _require_sha256(
        canonicalization_contract_sha256, field="canonicalization_contract_sha256"
    )
    selected = _verify_selection_record(selection_record)
    validated = _validated_objects(objects)
    by_key = {item.object_key: item for item in validated}
    workspace_root = _validate_workspace(workspace)
    temporary_roots = _initialize_query_temporary_roots(workspace_root)
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    forbidden_runtime_roots = [
        workspace_root / name
        for name in ("tmp_download", "tmp_decode", "tmp_pcl", "raw_cache")
    ]
    for forbidden in forbidden_runtime_roots:
        if destination == forbidden or forbidden in destination.parents:
            raise StreamingQueryError(
                "selected-source output root overlaps a disposable/cache workspace"
            )
    destination = _safe_persistent_directory(
        destination, label="selected-source output root"
    )

    gt_values = {str(row.get("gt_sha256", "")) for row in selected}
    calibration_values = {str(row.get("calibration_sha256", "")) for row in selected}
    if len(gt_values) != 1 or len(calibration_values) != 1:
        raise StreamingQueryError("selected sources do not share frozen GT/calibration bindings")
    selected_checkpoint_path = _checkpoint_path_within_workspace(
        processed_jsonl_path,
        workspace=workspace_root,
        default_name="processed_selected_sources.jsonl",
    )
    selection_record_sha256 = _require_sha256(
        str(selection_record["selection_record_sha256"]),
        field="selection_record_sha256",
    )
    selected_processing_contract_sha256 = _hash_json(
        {
            "canonicalization_contract_sha256": canonicalization_contract_sha256,
            "selection_record_sha256": selection_record_sha256,
        }
    )
    try:
        selected_checkpoint = Stage2CheckpointLog(
            selected_checkpoint_path,
            processing_contract_sha256=selected_processing_contract_sha256,
            gt_sha256=next(iter(gt_values)),
            calibration_sha256=next(iter(calibration_values)),
            expected_record_kinds={"SELECTED_SOURCE"},
        )
        source_rows = _load_authenticated_selected_source_rows(
            selected_checkpoint,
            selected,
            by_key,
            output_root=destination,
            canonicalization_contract_sha256=canonicalization_contract_sha256,
            selection_record_sha256=selection_record_sha256,
        )
    except Stage2CheckpointError as exc:
        raise StreamingQueryError(f"selected-source checkpoint is untrusted: {exc}") from exc

    for frozen in selected[len(source_rows) :]:
        key = frozen["object_key"]
        if key not in by_key:
            raise StreamingQueryError(f"selected source is absent from object inventory: {key}")
        query_object = by_key[key]
        expected_identity = query_object.identity()
        actual_identity = {field: frozen.get(field) for field in expected_identity}
        if actual_identity != expected_identity:
            raise StreamingQueryError(f"selected object identity changed: {key}")
        expected_payload_sha = _require_sha256(
            frozen["local_temporary_sha256"], field="local_temporary_sha256"
        )
        payload, payload_sha = _load_payload(
            query_object,
            payload_loader=payload_loader,
            mode=execution_mode,
            workspace=workspace_root,
            known_sha256=expected_payload_sha,
        )
        with _decode_from_temporary(
            payload,
            decoder=decoder,
            temporary_roots=temporary_roots,
            ordinal=query_object.ordinal,
        ) as points:
            first = canonicalizer(points.copy(), query_object)
            second = canonicalizer(points.copy(), query_object)
            first_value, first_bytes, source_sha = deterministic_npy_bytes(first)
            second_value, second_bytes, second_sha = deterministic_npy_bytes(second)
            if source_sha != second_sha or first_bytes != second_bytes or not np.array_equal(
                first_value, second_value
            ):
                raise StreamingQueryError(f"canonicalizer is nondeterministic for {key}")
            relative_path = Path("selected_sources") / source_sha / "source_points.npy"
            # The canonical NPY is atomically committed and fsynced before the
            # context removes the only temporary raw materialization.
            source_directory = _safe_persistent_directory(
                destination / "selected_sources" / source_sha,
                label="selected-source content directory",
            )
            source_destination = source_directory / "source_points.npy"
            if source_destination.exists() and (
                source_destination.is_symlink() or not source_destination.is_file()
            ):
                raise StreamingQueryError("selected-source payload path is unsafe")
            atomic_write_bytes(source_destination, first_bytes)
            row_core = {
                    "canonical_source_point_count": int(first_value.shape[0]),
                    "canonical_source_sha256": source_sha,
                    "canonicalization_contract_sha256": canonicalization_contract_sha256,
                    "object_key": key,
                    "payload_sha256": payload_sha,
                    "relative_path": relative_path.as_posix(),
                    "selection_index": frozen["selection_index"],
                    "selection_record_sha256": selection_record_sha256,
                }
            row = {
                **row_core,
                "selected_source_row_sha256": _hash_json(row_core),
            }
            _persist_selected_source_row(destination, row)
            selected_checkpoint.append_completed(
                CheckpointRecord(
                    record_kind="SELECTED_SOURCE",
                    s3_key=query_object.object_key,
                    remote_size_bytes=query_object.remote_size_bytes,
                    etag=query_object.etag,
                    last_modified=query_object.last_modified,
                    local_temporary_sha256=payload_sha,
                    processing_contract_sha256=selected_processing_contract_sha256,
                    gt_sha256=next(iter(gt_values)),
                    calibration_sha256=next(iter(calibration_values)),
                    result_geometry_row_sha256=row["selected_source_row_sha256"],
                    map_state_transition_sha256=None,
                    completed_at_utc=datetime.now(timezone.utc)
                    .isoformat(timespec="microseconds")
                    .replace("+00:00", "Z"),
                )
            )
            source_rows.append(row)

    core = {
        "canonicalization_contract_sha256": canonicalization_contract_sha256,
        "schema": SELECTED_SOURCE_SCHEMA,
        "selected_source_count": len(source_rows),
        "selected_sources": source_rows,
        "selection_record_sha256": selection_record["selection_record_sha256"],
    }
    return {**core, "selected_source_manifest_sha256": _hash_json(core)}


def assert_execution_modes_semantically_equal(
    streaming_first_pass: Sequence[Mapping[str, Any]],
    full_cache_first_pass: Sequence[Mapping[str, Any]],
    streaming_selected_sources: Mapping[str, Any],
    full_cache_selected_sources: Mapping[str, Any],
) -> None:
    if _json_native(streaming_first_pass) != _json_native(full_cache_first_pass):
        raise StreamingQueryError("query geometry metrics differ between execution modes")
    if _json_native(streaming_selected_sources) != _json_native(full_cache_selected_sources):
        raise StreamingQueryError("canonical selected sources differ between execution modes")


# Descriptive aliases used by the Stage-2 planner/contract prose.
stream_query_first_pass = first_pass_screen
reconstruct_selected_sources = second_pass_materialize_selected
