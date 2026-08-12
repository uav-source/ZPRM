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
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .io import atomic_write_bytes, atomic_write_json, canonical_json_bytes
from .selection import FORBIDDEN_SELECTOR_FRAGMENTS, geometry_only_view
from .streaming_target_map import canonical_array_sha256


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FIRST_PASS_SCHEMA = "zprm.stage2.streaming_query_first_pass.v1"
SELECTION_SCHEMA = "zprm.stage2.blind_selection_record.v1"
SELECTED_SOURCE_SCHEMA = "zprm.stage2.selected_canonical_sources.v1"
WEAK_LABEL = "CORRIDOR_OR_WEAK_GEOMETRY"
RICH_LABEL = "GEOMETRY_RICH"
EXPECTED_INTERVALS_PER_LABEL = 10
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
SelectionHook = Callable[[tuple[dict[str, Any], ...]], Sequence[Mapping[str, Any]]]
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
    root = Path(path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise StreamingQueryError("query workspace cannot be a symlink")
    return root


def _payload_cache_path(workspace: Path, sha256: str) -> Path:
    return workspace / "raw_cache" / sha256[:2] / f"{sha256}.bin"


def _load_payload(
    query_object: QueryObject,
    *,
    payload_loader: PayloadLoader,
    mode: QueryExecutionMode,
    workspace: Path,
    known_sha256: str | None = None,
) -> tuple[bytes, str]:
    expected_sha = known_sha256 or query_object.object_sha256
    cache_path = None if expected_sha is None else _payload_cache_path(workspace, expected_sha)
    if mode is QueryExecutionMode.FULL_RAW_CACHE and cache_path is not None and cache_path.exists():
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
        atomic_write_bytes(cache_path, payload)
    return payload, actual_sha


def _decode_from_temporary(
    payload: bytes,
    *,
    decoder: PointDecoder,
    workspace: Path,
    ordinal: int,
) -> np.ndarray:
    temp_root = workspace / "tmp_download"
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary = temp_root / f"query-{ordinal:08d}.bin"
    # A crash residue is not evidence and is never silently adopted.
    if temporary.exists():
        temporary.unlink()
    try:
        atomic_write_bytes(temporary, payload)
        points = np.asarray(decoder(temporary), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise StreamingQueryError("decoder must return an (N, 3) point array")
        if not np.all(np.isfinite(points)):
            raise StreamingQueryError("decoded query points must be finite")
        return np.ascontiguousarray(points, dtype="<f8")
    finally:
        if temporary.exists():
            temporary.unlink()


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
        geometry_only_view(metrics)
    except (KeyError, TypeError, ValueError) as exc:
        raise StreamingQueryError(f"invalid geometry-only metric row: {exc}") from exc
    forbidden = sorted(
        str(key)
        for key in metrics
        if any(fragment in str(key).lower() for fragment in FORBIDDEN_SELECTOR_FRAGMENTS)
    )
    if forbidden:
        raise StreamingQueryError(f"registration-derived geometry fields: {forbidden}")
    native_metrics = _json_native(metrics)
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
            geometry_only_view(row.get("geometry_metrics", {}))
        except (KeyError, TypeError, ValueError) as exc:
            raise StreamingQueryError("invalid geometry metrics in query checkpoint") from exc
        result.append(row)
    return result


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(
            _json_native(row),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    destination = Path(path)
    if destination.exists():
        existing = destination.read_bytes()
        if existing == payload:
            return
        if existing and (not existing.endswith(b"\n") or not payload.startswith(existing)):
            raise StreamingQueryError(
                "processed-query manifest differs and cannot be silently overwritten"
            )
    atomic_write_bytes(destination, payload, overwrite=True)


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
    """Run the blind first pass, deleting each temporary raw object on consume."""

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
    rows = _validate_existing_rows(
        existing_rows,
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
        points = _decode_from_temporary(
            payload, decoder=decoder, workspace=root, ordinal=query_object.ordinal
        )
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
        rows.append(row)
        if processed_jsonl_path is not None:
            _write_jsonl(processed_jsonl_path, rows)

    if processed_jsonl_path is not None and not validated:
        _write_jsonl(processed_jsonl_path, rows)
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
            geometry_only_view(row.get("geometry_metrics", {}))
        except (KeyError, TypeError, ValueError) as exc:
            raise StreamingQueryError("invalid first-pass geometry metrics") from exc
        key = str(row.get("object_key", ""))
        if not key or key in by_key:
            raise StreamingQueryError("first-pass object keys must be nonempty and unique")
        core = {field: value for field, value in row.items() if field != "geometry_row_sha256"}
        if row.get("geometry_row_sha256") != _hash_json(core):
            raise StreamingQueryError("first-pass row SHA mismatch before selection")
        by_key[key] = row

    # JSON round trips provide the hook an isolated, registration-blind view.
    hook_input = tuple(json.loads(json.dumps(row)) for row in rows)
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
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise StreamingQueryError("selected-source output root cannot be a symlink")

    source_rows: list[dict[str, Any]] = []
    for frozen in selected:
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
        points = _decode_from_temporary(
            payload, decoder=decoder, workspace=workspace_root, ordinal=query_object.ordinal
        )
        first = canonicalizer(points.copy(), query_object)
        second = canonicalizer(points.copy(), query_object)
        first_value, first_bytes, source_sha = deterministic_npy_bytes(first)
        second_value, second_bytes, second_sha = deterministic_npy_bytes(second)
        if source_sha != second_sha or first_bytes != second_bytes or not np.array_equal(
            first_value, second_value
        ):
            raise StreamingQueryError(f"canonicalizer is nondeterministic for {key}")
        relative_path = Path("selected_sources") / source_sha / "source_points.npy"
        atomic_write_bytes(destination / relative_path, first_bytes)
        source_rows.append(
            {
                "canonical_source_point_count": int(first_value.shape[0]),
                "canonical_source_sha256": source_sha,
                "canonicalization_contract_sha256": canonicalization_contract_sha256,
                "object_key": key,
                "payload_sha256": payload_sha,
                "relative_path": relative_path.as_posix(),
                "selection_index": frozen["selection_index"],
            }
        )

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
