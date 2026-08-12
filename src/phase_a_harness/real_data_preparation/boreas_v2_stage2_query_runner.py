"""Resumable production query preparation for Boreas v2 Stage-2.

The map producer deliberately stops after freezing one target.  This module
continues from that immutable target and owns only the two query payload
passes:

* all frozen QUERY objects are screened once, in allowlist order, with the
  frozen preprocessing and geometry-only selector;
* only the already frozen 100 snapshots are downloaded a second time and
  serialized as canonical source/T/metadata bundles.

Every successful download is journalled before its raw bytes are consumed.
The scientific result (or the canonical dual-path witness) is then committed
to the same fsynced hash chain before the temporary raw object is deleted.
The journal is the restart authority.  It contains no solver or backend entry
point and cannot enlarge the reconciled allowlist.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import resource
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from phase_a_harness.common_association_analysis import MAX_ASSOCIATION_DISTANCE_M

from .boreas_stage2_execution import (
    QUERY_FIRST_PASS_STAGE,
    QUERY_SECOND_PASS_STAGE,
)
from .boreas_stage2_remote import (
    AuthorizedRemoteObject,
    DownloadReceipt,
    ReconciledRemoteInventory,
    StrictAllowlistDownloader,
    TemporaryDownloadedObject,
)
from .boreas_v2_stage2_authorization import (
    AUTHORIZATION_SCHEMA,
    BoreasStage2AuthorizationError,
    VerifiedStage2Authorization,
)
from .boreas_v2_stage2_canonical_witness import (
    CanonicalSourceWitnessBindings,
    independently_verify_canonical_source,
)
from .boreas_v2_stage2_preprocessing import (
    BoreasLidarPoseIndex,
    PreprocessedBoreasScan,
    canonical_source_npy_bytes,
    canonical_t_reference_npy_bytes,
    decode_authenticated_boreas_velodyne_file,
    preprocess_primary_boreas_scan,
)
from .boreas_v2_stage2_selection import (
    CANDIDATE_INTERVAL_CSV_FIELDS,
    CANDIDATE_SCAN_CSV_FIELDS,
    FIRST_PASS_SCAN_FIELDS,
    GEOMETRY_METRIC_CSV_FIELDS,
    GEOMETRY_ONLY_FIELDS,
    SELECTED_INTERVAL_CSV_FIELDS,
    SELECTED_SNAPSHOT_CSV_FIELDS,
    ReferencePoseSeries,
    SelectionBindings,
    Stage2SelectionContract,
    TargetGeometryContext,
    build_blind_selection_manifest,
    build_candidate_intervals,
    build_candidate_scan_inventory,
    candidate_interval_csv_rows,
    candidate_scan_csv_rows,
    geometry_metric_csv_rows,
    select_frozen_intervals,
    select_interval_quantile_snapshots,
    selected_interval_csv_rows,
    selected_snapshot_csv_rows,
)
from .guard import NoRegistrationGuard
from .io import (
    PreparationIOError,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_bytes,
    csv_bytes,
    sha256_file,
)
from .stage2_disk_gate import Stage2DiskGate


ZERO_SHA256 = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QUERY_JOURNAL_SCHEMA = "zprm.boreas.v2.stage2.query_journal.v1"
SELECTION_FREEZE_SCHEMA = "zprm.boreas.v2.stage2.selection_freeze.v1"
TARGET_FREEZE_SCHEMA = "boreas_v2_stage2_target_map_freeze_v1"
FIRST_PASS_COMMIT = "FIRST_PASS_COMMITTED"
SELECTED_SOURCE_COMMIT = "SELECTED_SOURCE_COMMITTED"
TARGET_FROZEN = "TARGET_FROZEN"
TRANSFER_INTENT = "TRANSFER_INTENT"
TRANSFER_ABORTED = "TRANSFER_ABORTED"
FIRST_PASS_COMPLETE = "QUERY_FIRST_PASS_COMPLETE"
SELECTION_FROZEN = "R14_SELECTION_FROZEN"
GEOMETRY_WITNESS_STATUS = "PASS_DUAL_PATH_GEOMETRY_EXACT_SHARED_FROZEN_PRIMITIVES"

GEOMETRY_METRIC_VERIFICATION_FIELDS = (
    "query_ordinal",
    "object_key",
    "raw_payload_sha256",
    "receipt_sha256",
    "preprocessing_contract_sha256",
    "gt_sha256",
    "extrinsic_sha256",
    "target_map_sha256",
    "producer_source_sha256",
    "independent_source_sha256",
    "producer_T_reference_sha256",
    "independent_T_reference_sha256",
    "producer_geometry_sha256",
    "independent_geometry_sha256",
    "raw_point_count",
    "nonfinite_excluded_count",
    "range_excluded_count",
    "post_filter_point_count",
    "source_voxel_reduced_count",
    "canonical_source_point_count",
    "verification_status",
    "geometry_metric_verification_row_sha256",
)

FIRST_PASS_SELECTION_PROJECTION_FIELDS = (
    "execution_stage",
    "query_ordinal",
    "object_key",
    "first_pass_event_sha256",
    "first_pass_result_sha256",
    "candidate_scan_row_sha256",
    "selection_freeze_event_sha256",
    "projection_row_sha256",
)

RECEIPT_EXPORT_FIELDS = (
    "execution_stage",
    "schema",
    "selection_role",
    "sequence_id",
    "key",
    "timestamp_us",
    "remote_size_bytes",
    "etag",
    "last_modified",
    "local_temporary_sha256",
    "downloaded_at_utc",
    "receipt_sha256",
    "processing_result_sha256",
    "preprocessing_contract_sha256",
    "gt_sha256",
    "extrinsic_sha256",
    "checkpoint_status",
)

CANONICAL_INPUT_FIELDS = (
    "selection_index",
    "snapshot_id",
    "scene_label",
    "interval_id",
    "object_key",
    "source_points_path",
    "source_points_sha256",
    "source_points_size_bytes",
    "source_point_count",
    "T_reference_path",
    "T_reference_sha256",
    "T_reference_size_bytes",
    "snapshot_metadata_path",
    "snapshot_metadata_sha256",
    "target_map_path",
    "target_map_sha256",
    "target_map_size_bytes",
    "target_point_count",
    "preprocessing_contract_sha256",
    "selection_contract_sha256",
    "backend_parameter_contract_sha256",
    "future_open3d_source_sha256",
    "future_pcl_source_sha256",
    "future_open3d_target_sha256",
    "future_pcl_target_sha256",
    "byte_identical_for_both_backends",
    "bundle_sha256",
)

CANONICAL_SOURCE_VERIFICATION_FIELDS = (
    "selection_index",
    "snapshot_id",
    "object_key",
    "raw_payload_sha256",
    "raw_size_bytes",
    "first_pass_receipt_sha256",
    "second_pass_receipt_sha256",
    "preprocessing_contract_sha256",
    "gt_sha256",
    "extrinsic_sha256",
    "witness_implementation_sha256",
    "producer_source_sha256",
    "independent_source_sha256",
    "producer_T_reference_sha256",
    "independent_T_reference_sha256",
    "raw_point_count",
    "nonfinite_excluded_count",
    "range_excluded_count",
    "post_filter_point_count",
    "source_voxel_reduced_count",
    "canonical_source_point_count",
    "verification_status",
    "canonical_source_verification_row_sha256",
)

SELECTION_ARTIFACT_NAMES = frozenset(
    {
        "all_candidate_scans.csv",
        "all_candidate_intervals.csv",
        "excluded_candidate_scans.csv",
        "excluded_intervals.csv",
        "geometry_only_metrics.csv",
        "geometry_metric_verification.csv",
        "first_pass_selection_projection.csv",
        "selected_scene_intervals.csv",
        "selected_snapshots.csv",
        "canonical_input_manifest.csv",
        "canonical_source_verification.csv",
        "map_lineage_manifest.json",
        "target_map_freeze_manifest.json",
        "target_map_reducer_verification.json",
        "boreas_v2_stage2_preprocessing_contract.json",
        "boreas_v2_stage2_uncertainty_budget.csv",
        "boreas_v2_stage2_uncertainty_budget.json",
        "NO_ICP_ATTESTATION.json",
    }
)


class BoreasV2Stage2QueryRunnerError(RuntimeError):
    """A query phase, durable checkpoint, or authority binding is invalid."""


class QueryRuntimeLease:
    """One process-wide owner of the shared Stage-2 runtime lock.

    Production orchestration acquires this lease before capacity evidence,
    temporary recovery, downloader construction, or QueryRunner construction.
    Runner phase locks then reuse the exact live lease instead of opening a
    second flock description and risking a self-conflict.
    """

    def __init__(self, runtime_root: str | Path) -> None:
        self.runtime_root = _canonical_root(Path(runtime_root), label="runtime root")
        self.path = self.runtime_root / ".boreas_v2_stage2.lock"
        self._descriptor: int | None = None

    @property
    def active(self) -> bool:
        return self._descriptor is not None

    def __enter__(self) -> "QueryRuntimeLease":
        if self.active:
            raise BoreasV2Stage2QueryRunnerError("query runtime lease is already active")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise BoreasV2Stage2QueryRunnerError("query runtime lease is unsafe")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BoreasV2Stage2QueryRunnerError(
                    "another Boreas Stage-2 process owns the runtime"
                ) from exc
        except Exception:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def assert_live(self) -> None:
        if self._descriptor is None:
            raise BoreasV2Stage2QueryRunnerError("query runtime lease is not active")
        if not stat.S_ISREG(os.fstat(self._descriptor).st_mode):
            raise BoreasV2Stage2QueryRunnerError("query runtime lease descriptor changed")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _sha(value: Any, field: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise BoreasV2Stage2QueryRunnerError(f"{field} must be lowercase SHA-256")
    return text


def _strict_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BoreasV2Stage2QueryRunnerError(f"{field} must be an integer >= {minimum}")
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


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _event_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_line(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_root(path: Path, *, label: str) -> Path:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
        or path.resolve(strict=True) != path
    ):
        raise BoreasV2Stage2QueryRunnerError(f"{label} must be a canonical directory")
    return path


def _safe_relative_file(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise BoreasV2Stage2QueryRunnerError(f"unsafe {label} path")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise BoreasV2Stage2QueryRunnerError(f"{label} is absent or unsafe")
    if root != path.parent and root not in path.parents:
        raise BoreasV2Stage2QueryRunnerError(f"{label} escapes runtime root")
    return path


def _validated_receipt(value: Mapping[str, Any]) -> DownloadReceipt:
    expected = {
        "downloaded_at_utc",
        "etag",
        "key",
        "last_modified",
        "local_temporary_sha256",
        "remote_size_bytes",
        "selection_role",
        "sequence_id",
        "timestamp_us",
        "schema",
        "receipt_sha256",
    }
    if set(value) != expected:
        raise BoreasV2Stage2QueryRunnerError("download receipt field set differs")
    if value.get("schema") != "zprm.boreas.stage2.download_receipt.v1":
        raise BoreasV2Stage2QueryRunnerError("download receipt schema differs")
    receipt = DownloadReceipt(
        selection_role=str(value["selection_role"]),
        sequence_id=str(value["sequence_id"]),
        key=str(value["key"]),
        timestamp_us=_strict_int(value["timestamp_us"], "receipt timestamp", minimum=1),
        remote_size_bytes=_strict_int(
            value["remote_size_bytes"], "receipt size", minimum=1
        ),
        etag=str(value["etag"]),
        last_modified=str(value["last_modified"]),
        local_temporary_sha256=_sha(
            value["local_temporary_sha256"], "receipt payload SHA"
        ),
        downloaded_at_utc=str(value["downloaded_at_utc"]),
    )
    if receipt.as_dict() != dict(value):
        raise BoreasV2Stage2QueryRunnerError("download receipt self-hash differs")
    return receipt


@dataclass(frozen=True)
class QueryRunnerConfig:
    runtime_root: Path
    preprocessing_contract_sha256: str
    extrinsic_sha256: str
    backend_parameter_contract_sha256: str
    canonical_witness_implementation_sha256: str
    expected_target_freeze_sha256: str
    query_reference_pose_sha256: str
    map_reference_pose_sha256: str
    stage1_manifest_sha256: str
    storage_manifest_sha256: str
    stage1_allowlist_sha256: str
    pair_selection_sha256: str
    target_context_resource_plan_sha256: str
    target_context_measurement_evidence_sha256: str
    target_context_measurement_provenance_sha256: str
    production_mode: bool = True

    def validated(self) -> "QueryRunnerConfig":
        root = _canonical_root(Path(self.runtime_root), label="runtime root")
        values = {
            field: _sha(getattr(self, field), field)
            for field in (
                "preprocessing_contract_sha256",
                "extrinsic_sha256",
                "backend_parameter_contract_sha256",
                "canonical_witness_implementation_sha256",
                "expected_target_freeze_sha256",
                "query_reference_pose_sha256",
                "map_reference_pose_sha256",
                "stage1_manifest_sha256",
                "storage_manifest_sha256",
                "stage1_allowlist_sha256",
                "pair_selection_sha256",
                "target_context_resource_plan_sha256",
                "target_context_measurement_evidence_sha256",
                "target_context_measurement_provenance_sha256",
            )
        }
        if not isinstance(self.production_mode, bool):
            raise BoreasV2Stage2QueryRunnerError("production_mode must be boolean")
        return QueryRunnerConfig(runtime_root=root, production_mode=self.production_mode, **values)


QueryPreprocessor = Callable[
    [TemporaryDownloadedObject, AuthorizedRemoteObject, BoreasLidarPoseIndex],
    PreprocessedBoreasScan,
]
TargetContextFactory = Callable[[Any], TargetGeometryContext]
GeometryComputer = Callable[[PreprocessedBoreasScan, TargetGeometryContext], Mapping[str, Any]]
CanonicalWitness = Callable[..., Mapping[str, Any]]
FaultHook = Callable[[str, Optional[AuthorizedRemoteObject]], None]


def _production_query_preprocessor(
    materialized: TemporaryDownloadedObject,
    item: AuthorizedRemoteObject,
    pose_index: BoreasLidarPoseIndex,
) -> PreprocessedBoreasScan:
    decoded = decode_authenticated_boreas_velodyne_file(
        materialized.path,
        object_key=item.key,
        expected_size_bytes=materialized.receipt.remote_size_bytes,
        expected_sha256=materialized.receipt.local_temporary_sha256,
    )
    return preprocess_primary_boreas_scan(decoded, pose_index=pose_index, role="QUERY")


def _production_geometry(
    scan: PreprocessedBoreasScan, context: TargetGeometryContext
) -> Mapping[str, Any]:
    return context.compute(scan.points_xyz, scan.t_reference)


def _independent_production_geometry(
    scan: PreprocessedBoreasScan, context: TargetGeometryContext
) -> Mapping[str, Any]:
    """Recompute the ten fields without calling the producer geometry method.

    The immutable target tree and normal field are shared to stay inside the
    32 GiB machine envelope. Association masking, Hessian construction,
    eigenspectrum normalization, conditioning, and entropy are independently
    expressed here; no residual, gradient, solver, or estimated transform is
    formed.
    """

    source = np.asarray(scan.points_xyz, dtype="<f8")
    transform = np.asarray(scan.t_reference, dtype="<f8")
    if source.ndim != 2 or source.shape[1] != 3 or not np.all(np.isfinite(source)):
        raise BoreasV2Stage2QueryRunnerError(
            "independent geometry source must be finite N-by-3"
        )
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise BoreasV2Stage2QueryRunnerError(
            "independent geometry transform must be finite 4-by-4"
        )
    reference_points = source @ transform[:3, :3].T + transform[:3, 3]
    distances, indices = context.target_tree.query(
        reference_points,
        k=1,
        workers=1,
    )
    associated = (
        np.isfinite(distances)
        & (distances <= MAX_ASSOCIATION_DISTANCE_M)
        & (indices >= 0)
        & (indices < context.target_points.shape[0])
    )
    association_count = int(np.count_nonzero(associated))
    associated_indices = indices[associated].astype(np.int64, copy=False)
    normals = context.target_normals[
        associated_indices[context.target_normal_valid[associated_indices]]
    ]
    valid_count = int(normals.shape[0])
    result: dict[str, Any] = {
        "initial_correspondence_count": association_count,
        "initial_valid_normal_correspondence_count": valid_count,
    }
    if valid_count == 0:
        result.update({field: None for field in GEOMETRY_ONLY_FIELDS[2:]})
        return result
    hessian = (normals.T @ normals) / float(valid_count)
    eigenvalues = np.clip(np.linalg.eigvalsh(hessian), 0.0, None)
    total = max(float(eigenvalues.sum()), 1.0e-12)
    normalized = eigenvalues / total
    positive = normalized[normalized > 0.0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(3.0)
    result.update(
        {
            "lambda_min_trans": float(eigenvalues[0]),
            "lambda_mid_trans": float(eigenvalues[1]),
            "lambda_max_trans": float(eigenvalues[2]),
            "normalized_lambda_min_trans": float(normalized[0]),
            "normalized_lambda_mid_trans": float(normalized[1]),
            "normalized_lambda_max_trans": float(normalized[2]),
            "condition_number_trans": float(
                eigenvalues[2] / max(float(eigenvalues[0]), 1.0e-12)
            ),
            "spectral_entropy_trans": entropy,
        }
    )
    if set(result) != set(GEOMETRY_ONLY_FIELDS):
        raise AssertionError("independent geometry-only output schema drift")
    return result


def _production_target_context(target: Any) -> TargetGeometryContext:
    """Stable production identity wrapper around the classmethod."""

    return TargetGeometryContext.prepare(target)


@dataclass(frozen=True)
class QueryRunnerDependencies:
    query_preprocessor: QueryPreprocessor = _production_query_preprocessor
    independent_query_preprocessor: QueryPreprocessor = _production_query_preprocessor
    target_context_factory: TargetContextFactory = _production_target_context
    geometry_computer: GeometryComputer = _production_geometry
    independent_geometry_computer: GeometryComputer = _independent_production_geometry
    canonical_witness: CanonicalWitness = independently_verify_canonical_source
    available_memory_provider: Callable[[], int] | None = None
    fault_hook: FaultHook | None = None


@dataclass(frozen=True)
class QueryPhaseSummary:
    completed_object_count: int
    total_object_count: int
    downloaded_bytes_this_run: int
    deleted_temporary_bytes_this_run: int
    final_journal_event_sha256: str


class _QueryJournal:
    """Strict append-only state for both query passes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise BoreasV2Stage2QueryRunnerError("query journal is unsafe")
        self._cache: list[dict[str, Any]] | None = None
        self._cache_size = -1
        self._cache = self._parse_disk()
        self._cache_size = self.path.stat().st_size if self.path.exists() else 0
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        rows = self._cache or []
        self._unresolved_index: dict[str, dict[str, Any]] = {}
        self._downloads_index: dict[str, dict[str, Any]] = {}
        self._pending_transfer_index: dict[str, dict[str, Any]] = {}
        self._committed_index: list[dict[str, Any]] = []
        self._barrier_index: dict[str, dict[str, Any]] = {}
        for row in rows:
            self._index_row(row)

    def _index_row(self, row: dict[str, Any]) -> None:
        kind = row["event_kind"]
        if kind == TRANSFER_INTENT:
            self._pending_transfer_index[row["event_sha256"]] = row
        elif kind == "DOWNLOADED":
            self._pending_transfer_index.pop(
                row["payload"]["transfer_intent_event_sha256"], None
            )
            self._downloads_index[row["event_sha256"]] = row
            self._unresolved_index[row["event_sha256"]] = row
        elif kind == TRANSFER_ABORTED:
            self._pending_transfer_index.pop(
                row["payload"]["transfer_intent_event_sha256"], None
            )
        elif kind in {FIRST_PASS_COMMIT, SELECTED_SOURCE_COMMIT, "ABORTED"}:
            self._unresolved_index.pop(row["payload"]["download_event_sha256"], None)
            if kind != "ABORTED":
                self._committed_index.append(row)
        elif kind in {TARGET_FROZEN, FIRST_PASS_COMPLETE, SELECTION_FROZEN}:
            self._barrier_index[kind] = row

    def _rows(self) -> list[dict[str, Any]]:
        size = self.path.stat().st_size if self.path.exists() else 0
        if self._cache is None or size != self._cache_size:
            self._cache = self._parse_disk()
            self._cache_size = size
            self._rebuild_indexes()
        return self._cache

    @property
    def final_event_sha256(self) -> str:
        rows = self._rows()
        return rows[-1]["event_sha256"] if rows else ZERO_SHA256

    @property
    def event_count(self) -> int:
        return len(self._rows())

    def download_for_event(self, event_sha256: str) -> dict[str, Any] | None:
        self._rows()
        row = self._downloads_index.get(event_sha256)
        return json.loads(json.dumps(row)) if row is not None else None

    @property
    def pending_transfer_intents(self) -> tuple[dict[str, Any], ...]:
        self._rows()
        return tuple(
            json.loads(json.dumps(row))
            for row in self._pending_transfer_index.values()
        )

    def _parse_disk(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = self.path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise BoreasV2Stage2QueryRunnerError("query journal is truncated")
        rows: list[dict[str, Any]] = []
        previous = ZERO_SHA256
        downloads: dict[str, dict[str, Any]] = {}
        intents: dict[str, dict[str, Any]] = {}
        resolved_intents: set[str] = set()
        resolved: set[str] = set()
        committed: set[tuple[str, str]] = set()
        for sequence_number, line in enumerate(payload.splitlines(keepends=True), start=1):
            try:
                row = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise BoreasV2Stage2QueryRunnerError(
                    "query journal contains invalid JSON"
                ) from exc
            if not isinstance(row, dict) or _canonical_line(row) != line:
                raise BoreasV2Stage2QueryRunnerError("query journal JSONL is noncanonical")
            expected_envelope = {
                "event_kind",
                "event_sha256",
                "execution_stage",
                "object_key",
                "payload",
                "previous_event_sha256",
                "schema",
                "sequence_number",
            }
            if set(row) != expected_envelope:
                raise BoreasV2Stage2QueryRunnerError("query journal envelope schema differs")
            if (
                row["schema"] != QUERY_JOURNAL_SCHEMA
                or row["sequence_number"] != sequence_number
                or row["previous_event_sha256"] != previous
                or row["execution_stage"] not in {
                    "QUERY_BARRIER",
                    QUERY_FIRST_PASS_STAGE,
                    QUERY_SECOND_PASS_STAGE,
                }
                or not isinstance(row["payload"], dict)
            ):
                raise BoreasV2Stage2QueryRunnerError("query journal chain metadata differs")
            unsigned = dict(row)
            supplied = _sha(unsigned.pop("event_sha256"), "query event SHA")
            if _event_sha(unsigned) != supplied:
                raise BoreasV2Stage2QueryRunnerError("query journal event SHA differs")
            kind = row["event_kind"]
            payload_row = row["payload"]
            if kind in {TARGET_FROZEN, FIRST_PASS_COMPLETE, SELECTION_FROZEN}:
                if row["execution_stage"] != "QUERY_BARRIER":
                    raise BoreasV2Stage2QueryRunnerError("barrier stage differs")
                if kind == TARGET_FROZEN:
                    expected_barrier_fields = {
                        "target_freeze_file_sha256",
                        "target_map_sha256",
                        "target_context_resource_plan_sha256",
                        "target_context_measurement_evidence_sha256",
                        "target_context_measurement_provenance_sha256",
                    }
                elif kind == FIRST_PASS_COMPLETE:
                    expected_barrier_fields = {
                        "first_pass_count",
                        "first_pass_rows_sha256",
                        "target_freeze_event_sha256",
                    }
                else:
                    expected_barrier_fields = {
                        "artifact_sha256",
                        "first_pass_complete_event_sha256",
                        "selected_snapshot_count",
                        "selection_contract_sha256",
                        "selection_freeze_file_sha256",
                    }
                if set(payload_row) != expected_barrier_fields:
                    raise BoreasV2Stage2QueryRunnerError("query barrier schema differs")
                for name, child in payload_row.items():
                    if name.endswith("sha256"):
                        if name == "artifact_sha256":
                            if not isinstance(child, dict) or not child:
                                raise BoreasV2Stage2QueryRunnerError(
                                    "query barrier artifact SHA mapping differs"
                                )
                            for artifact_name, digest in child.items():
                                if not isinstance(artifact_name, str) or not artifact_name:
                                    raise BoreasV2Stage2QueryRunnerError(
                                        "query barrier artifact name differs"
                                    )
                                _sha(digest, f"query barrier artifact {artifact_name}")
                        else:
                            _sha(child, f"query barrier {name}")
            elif kind == TRANSFER_INTENT:
                expected_intent = {
                    "etag",
                    "last_modified",
                    "remote_size_bytes",
                    "sequence_id",
                    "temporary_relative_path",
                    "timestamp_us",
                }
                if set(payload_row) != expected_intent:
                    raise BoreasV2Stage2QueryRunnerError("transfer intent schema differs")
                _strict_int(payload_row["remote_size_bytes"], "intent size", minimum=1)
                _strict_int(payload_row["timestamp_us"], "intent timestamp", minimum=1)
                if not all(
                    isinstance(payload_row[field], str) and payload_row[field]
                    for field in ("etag", "last_modified", "sequence_id", "temporary_relative_path")
                ):
                    raise BoreasV2Stage2QueryRunnerError("transfer intent identity differs")
                if supplied in intents:
                    raise BoreasV2Stage2QueryRunnerError("duplicate transfer intent")
                intents[supplied] = row
            elif kind == TRANSFER_ABORTED:
                if set(payload_row) != {"reason", "transfer_intent_event_sha256"}:
                    raise BoreasV2Stage2QueryRunnerError("transfer abort schema differs")
                intent_sha = _sha(
                    payload_row["transfer_intent_event_sha256"], "transfer intent SHA"
                )
                intent = intents.get(intent_sha)
                if (
                    intent is None
                    or intent_sha in resolved_intents
                    or intent["execution_stage"] != row["execution_stage"]
                    or intent["object_key"] != row["object_key"]
                    or not isinstance(payload_row["reason"], str)
                    or not payload_row["reason"]
                ):
                    raise BoreasV2Stage2QueryRunnerError("transfer abort identity differs")
                resolved_intents.add(intent_sha)
            elif kind == "DOWNLOADED":
                if set(payload_row) != {
                    "receipt",
                    "temporary_relative_path",
                    "transfer_intent_event_sha256",
                }:
                    raise BoreasV2Stage2QueryRunnerError("download event schema differs")
                intent_sha = _sha(
                    payload_row["transfer_intent_event_sha256"], "transfer intent SHA"
                )
                intent = intents.get(intent_sha)
                if (
                    intent is None
                    or intent_sha in resolved_intents
                    or intent["event_kind"] != TRANSFER_INTENT
                    or intent["execution_stage"] != row["execution_stage"]
                    or intent["object_key"] != row["object_key"]
                ):
                    raise BoreasV2Stage2QueryRunnerError("download lacks transfer intent")
                receipt = _validated_receipt(payload_row["receipt"])
                intent_payload = intent["payload"]
                if (
                    payload_row["temporary_relative_path"]
                    != intent_payload["temporary_relative_path"]
                    or receipt.sequence_id != intent_payload["sequence_id"]
                    or receipt.timestamp_us != intent_payload["timestamp_us"]
                    or receipt.remote_size_bytes != intent_payload["remote_size_bytes"]
                    or receipt.etag != intent_payload["etag"]
                    or receipt.last_modified != intent_payload["last_modified"]
                ):
                    raise BoreasV2Stage2QueryRunnerError(
                        "download/transfer-intent identity differs"
                    )
                if (
                    receipt.selection_role != "QUERY"
                    or receipt.key != row["object_key"]
                    or supplied in downloads
                ):
                    raise BoreasV2Stage2QueryRunnerError("download event identity differs")
                downloads[supplied] = row
                resolved_intents.add(intent_sha)
            elif kind in {FIRST_PASS_COMMIT, SELECTED_SOURCE_COMMIT, "ABORTED"}:
                required = {"download_event_sha256", "receipt_sha256"}
                if not required <= set(payload_row):
                    raise BoreasV2Stage2QueryRunnerError("resolution event schema differs")
                download_sha = _sha(
                    payload_row["download_event_sha256"], "download event SHA"
                )
                download = downloads.get(download_sha)
                if download is None or download_sha in resolved:
                    raise BoreasV2Stage2QueryRunnerError(
                        "resolution lacks one unresolved download"
                    )
                receipt = _validated_receipt(download["payload"]["receipt"])
                if (
                    download["execution_stage"] != row["execution_stage"]
                    or download["object_key"] != row["object_key"]
                    or payload_row["receipt_sha256"]
                    != receipt.as_dict()["receipt_sha256"]
                ):
                    raise BoreasV2Stage2QueryRunnerError("resolution receipt differs")
                expected_kind = (
                    FIRST_PASS_COMMIT
                    if row["execution_stage"] == QUERY_FIRST_PASS_STAGE
                    else SELECTED_SOURCE_COMMIT
                )
                if kind != "ABORTED" and kind != expected_kind:
                    raise BoreasV2Stage2QueryRunnerError("commit kind/stage differs")
                if kind == FIRST_PASS_COMMIT:
                    if set(payload_row) != {
                        "download_event_sha256",
                        "first_pass_row",
                        "geometry_metric_verification_row",
                        "processing_result_sha256",
                        "receipt_sha256",
                    }:
                        raise BoreasV2Stage2QueryRunnerError("first-pass commit schema differs")
                    if set(payload_row["first_pass_row"]) != FIRST_PASS_SCAN_FIELDS:
                        raise BoreasV2Stage2QueryRunnerError("first-pass row schema differs")
                    if (
                        set(payload_row["geometry_metric_verification_row"])
                        != set(GEOMETRY_METRIC_VERIFICATION_FIELDS)
                    ):
                        raise BoreasV2Stage2QueryRunnerError(
                            "geometry witness row schema differs"
                        )
                    _sha(payload_row["processing_result_sha256"], "first-pass result SHA")
                elif kind == SELECTED_SOURCE_COMMIT:
                    if set(payload_row) != {
                        "canonical_input_row",
                        "canonical_source_verification_row",
                        "download_event_sha256",
                        "processing_result_sha256",
                        "receipt_sha256",
                        "selected_snapshot_row_sha256",
                    }:
                        raise BoreasV2Stage2QueryRunnerError("selected-source commit schema differs")
                    if set(payload_row["canonical_input_row"]) != set(CANONICAL_INPUT_FIELDS):
                        raise BoreasV2Stage2QueryRunnerError("canonical manifest row schema differs")
                    if (
                        set(payload_row["canonical_source_verification_row"])
                        != set(CANONICAL_SOURCE_VERIFICATION_FIELDS)
                    ):
                        raise BoreasV2Stage2QueryRunnerError("canonical witness row schema differs")
                    _sha(payload_row["processing_result_sha256"], "selected-source result SHA")
                    _sha(
                        payload_row["selected_snapshot_row_sha256"],
                        "selected snapshot row SHA",
                    )
                elif set(payload_row) != {
                    "download_event_sha256",
                    "reason",
                    "receipt_sha256",
                }:
                    raise BoreasV2Stage2QueryRunnerError("abort event schema differs")
                identity = (row["execution_stage"], row["object_key"])
                if kind != "ABORTED":
                    if identity in committed:
                        raise BoreasV2Stage2QueryRunnerError("duplicate query commit")
                    committed.add(identity)
                resolved.add(download_sha)
            else:
                raise BoreasV2Stage2QueryRunnerError("unknown query journal event kind")
            rows.append(row)
            previous = supplied
        if len(set(intents) - resolved_intents) > 1:
            raise BoreasV2Stage2QueryRunnerError(
                "multiple unresolved transfer intents exist"
            )
        return rows

    def read(self, *, full_rescan: bool = False) -> list[dict[str, Any]]:
        if full_rescan:
            self._cache = self._parse_disk()
            self._cache_size = self.path.stat().st_size if self.path.exists() else 0
            self._rebuild_indexes()
        # Internal rows are never exposed mutably.
        return json.loads(json.dumps(self._rows()))

    def append(
        self,
        *,
        event_kind: str,
        execution_stage: str,
        object_key: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise BoreasV2Stage2QueryRunnerError("query journal is not regular")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            current_size = os.fstat(descriptor).st_size
            if current_size != self._cache_size:
                raise BoreasV2Stage2QueryRunnerError(
                    "query journal changed concurrently"
                )
            rows = self._rows()
            unsigned = {
                "event_kind": event_kind,
                "execution_stage": execution_stage,
                "object_key": str(object_key),
                "payload": json.loads(canonical_json_bytes(payload)),
                "previous_event_sha256": rows[-1]["event_sha256"] if rows else ZERO_SHA256,
                "schema": QUERY_JOURNAL_SCHEMA,
                "sequence_number": len(rows) + 1,
            }
            row = {**unsigned, "event_sha256": _event_sha(unsigned)}
            view = memoryview(_canonical_line(row))
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise OSError("zero-byte query journal append")
                view = view[count:]
            os.fsync(descriptor)
            self._cache.append(row)
            self._cache_size = current_size + len(_canonical_line(row))
            self._index_row(row)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        _fsync_directory(self.path.parent)
        return row

    @property
    def committed(self) -> tuple[dict[str, Any], ...]:
        self._rows()
        return tuple(json.loads(json.dumps(row)) for row in self._committed_index)

    def barrier(self, kind: str) -> dict[str, Any] | None:
        self._rows()
        value = self._barrier_index.get(kind)
        return json.loads(json.dumps(value)) if value is not None else None

    @property
    def unresolved(self) -> tuple[dict[str, Any], ...]:
        self._rows()
        return tuple(
            json.loads(json.dumps(row)) for row in self._unresolved_index.values()
        )


class BoreasV2Stage2QueryRunner:
    """Compose the authorized first and selected-source query passes."""

    def __init__(
        self,
        *,
        config: QueryRunnerConfig,
        dependencies: QueryRunnerDependencies,
        authorization: VerifiedStage2Authorization,
        no_registration_guard: NoRegistrationGuard,
        disk_gate: Stage2DiskGate,
        downloader: StrictAllowlistDownloader,
        inventory: ReconciledRemoteInventory,
        query_pose_index: BoreasLidarPoseIndex,
        map_pose_index: BoreasLidarPoseIndex,
        frozen_windows: Sequence[Mapping[str, Any]],
        selection_contract: Stage2SelectionContract = Stage2SelectionContract(),
        runtime_lease: QueryRuntimeLease | None = None,
    ) -> None:
        self.config = config.validated()
        if self.config.production_mode:
            if (
                not isinstance(runtime_lease, QueryRuntimeLease)
                or runtime_lease.runtime_root != self.config.runtime_root
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "production query requires the exact active runtime lease"
                )
            runtime_lease.assert_live()
        elif runtime_lease is not None:
            if not isinstance(runtime_lease, QueryRuntimeLease):
                raise BoreasV2Stage2QueryRunnerError("runtime lease type differs")
            runtime_lease.assert_live()
        self.runtime_lease = runtime_lease
        if not isinstance(dependencies, QueryRunnerDependencies):
            raise BoreasV2Stage2QueryRunnerError("query dependencies have an invalid type")
        if not isinstance(authorization, VerifiedStage2Authorization):
            raise BoreasV2Stage2QueryRunnerError(
                "query runner requires a VerifiedStage2Authorization capability"
            )
        if not isinstance(no_registration_guard, NoRegistrationGuard):
            raise BoreasV2Stage2QueryRunnerError(
                "query runner requires a NoRegistrationGuard"
            )
        if not no_registration_guard.active:
            raise BoreasV2Stage2QueryRunnerError("NoRegistrationGuard is not active")
        if authorization.no_registration_guard is not no_registration_guard:
            raise BoreasV2Stage2QueryRunnerError(
                "query runner and authorization must share the exact guard"
            )
        if not isinstance(disk_gate, Stage2DiskGate):
            raise BoreasV2Stage2QueryRunnerError("query runner requires a Stage2DiskGate")
        if not isinstance(downloader, StrictAllowlistDownloader):
            raise BoreasV2Stage2QueryRunnerError(
                "query runner requires a StrictAllowlistDownloader"
            )
        if not isinstance(inventory, ReconciledRemoteInventory):
            raise BoreasV2Stage2QueryRunnerError(
                "query runner requires a reconciled remote inventory"
            )
        if (
            downloader.authorization is not authorization
            or downloader.disk_gate is not disk_gate
            or downloader.inventory is not inventory
        ):
            raise BoreasV2Stage2QueryRunnerError(
                "query phases must share the exact authorization/disk/downloader/inventory"
            )
        if self.config.runtime_root != authorization.runtime_root:
            raise BoreasV2Stage2QueryRunnerError(
                "query runtime root differs from authorization"
            )
        try:
            authorization.assert_live()
            authorization.bind_disk_gate(disk_gate)
        except BoreasStage2AuthorizationError as exc:
            raise BoreasV2Stage2QueryRunnerError(
                "query authorization is not live"
            ) from exc
        if not getattr(disk_gate, "_started", False):
            raise BoreasV2Stage2QueryRunnerError(
                "query phases require the already-passed shared live disk gate"
            )
        if inventory.allowlist_sha256 != authorization.allowlist_sha256:
            raise BoreasV2Stage2QueryRunnerError("query allowlist binding differs")
        if self.config.stage1_allowlist_sha256 != inventory.allowlist_sha256:
            raise BoreasV2Stage2QueryRunnerError("query config allowlist binding differs")
        primary_pair = authorization.primary_pair
        document = authorization.document
        if self.config.production_mode and authorization.formally_verified is not True:
            raise BoreasV2Stage2QueryRunnerError(
                "production query requires a formally verified authorization capability"
            )
        if not self.config.production_mode and authorization.formally_verified is True:
            raise BoreasV2Stage2QueryRunnerError(
                "formal authorization capability cannot be downgraded to non-production"
            )
        if (
            document.get("schema_version") == AUTHORIZATION_SCHEMA
            and document.get("authorization_scope")
            == "BOREAS_V2_STAGE2_LIDAR_DATA_PREPARATION_ONLY"
            and not self.config.production_mode
        ):
            raise BoreasV2Stage2QueryRunnerError(
                "a formal Boreas Stage-2 authorization may only be consumed "
                "in production_mode=True"
            )
        expected_document_bindings = {
            "allowlist_sha256": self.config.stage1_allowlist_sha256,
            "preprocessing_contract_sha256": (
                self.config.preprocessing_contract_sha256
            ),
            "stage1_manifest_file_sha256": self.config.stage1_manifest_sha256,
            "storage_manifest_file_sha256": self.config.storage_manifest_sha256,
            "primary_pair_sha256": self.config.pair_selection_sha256,
        }
        for field, expected in expected_document_bindings.items():
            if document.get(field) != expected:
                raise BoreasV2Stage2QueryRunnerError(
                    f"query config {field} differs from live authorization"
                )
        query_objects = inventory.for_role("QUERY")
        if not query_objects or any(
            item.frozen.sequence_id != primary_pair["query_sequence_id"]
            for item in query_objects
        ):
            raise BoreasV2Stage2QueryRunnerError("query inventory differs from PRIMARY pair")
        ordinals = [item.frozen.role_ordinal for item in query_objects]
        if ordinals != list(range(len(query_objects))):
            raise BoreasV2Stage2QueryRunnerError("query allowlist order is not exact")
        if self.config.production_mode and len(query_objects) != 11_859:
            raise BoreasV2Stage2QueryRunnerError(
                "production query pass must contain exactly 11,859 objects"
            )
        if not isinstance(query_pose_index, BoreasLidarPoseIndex):
            raise BoreasV2Stage2QueryRunnerError("query pose index has an invalid type")
        if (
            query_pose_index.sequence_id != primary_pair["query_sequence_id"]
            or query_pose_index.source_sha256
            != self.config.query_reference_pose_sha256
        ):
            raise BoreasV2Stage2QueryRunnerError("query pose authority differs")
        if (
            not isinstance(map_pose_index, BoreasLidarPoseIndex)
            or map_pose_index.sequence_id != primary_pair["map_sequence_id"]
            or map_pose_index.source_sha256 != self.config.map_reference_pose_sha256
        ):
            raise BoreasV2Stage2QueryRunnerError("map pose authority differs")
        query_timestamps = np.asarray(sorted(query_pose_index.poses), dtype=np.int64)
        query_transforms = [
            query_pose_index.poses[int(timestamp)].t_enu_lidar
            for timestamp in query_timestamps
        ]
        reference_pose_series = ReferencePoseSeries(
            timestamps_us=query_timestamps,
            translations_xyz_m=np.asarray(
                [transform[:3, 3] for transform in query_transforms], dtype="<f8"
            ),
            quaternions_xyzw=Rotation.from_matrix(
                np.asarray([transform[:3, :3] for transform in query_transforms])
            ).as_quat(),
        ).validated()
        map_positions = np.asarray(
            [
                map_pose_index.poses[timestamp].t_enu_lidar[:3, 3]
                for timestamp in sorted(map_pose_index.poses)
            ],
            dtype="<f8",
        )
        if self.config.production_mode:
            if selection_contract.parameter_authority != "BOREAS_V2_STAGE2_PREREGISTRATION":
                raise BoreasV2Stage2QueryRunnerError(
                    "production query selection contract is not frozen authority"
                )
            if (
                dependencies.query_preprocessor is not _production_query_preprocessor
                or dependencies.independent_query_preprocessor
                is not _production_query_preprocessor
                or dependencies.target_context_factory is not _production_target_context
                or dependencies.geometry_computer is not _production_geometry
                or dependencies.independent_geometry_computer
                is not _independent_production_geometry
                or dependencies.canonical_witness
                is not independently_verify_canonical_source
                or dependencies.available_memory_provider is not None
                or dependencies.fault_hook is not None
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "production query scientific callbacks cannot be replaced"
                )
        if len(frozen_windows) != selection_contract.expected_candidate_interval_count:
            raise BoreasV2Stage2QueryRunnerError("frozen window count differs")

        self.dependencies = dependencies
        self.authorization = authorization
        self.no_registration_guard = no_registration_guard
        self.disk_gate = disk_gate
        self.downloader = downloader
        self.inventory = inventory
        self.query_objects = tuple(query_objects)
        self.query_pose_index = query_pose_index
        self.map_pose_index = map_pose_index
        self.reference_pose_series = reference_pose_series
        self.map_reference_positions = np.ascontiguousarray(map_positions, dtype="<f8")
        self.map_reference_tree = cKDTree(self.map_reference_positions, copy_data=False)
        self.frozen_windows = tuple(dict(row) for row in frozen_windows)
        self.selection_contract = selection_contract
        self.runtime_root = self.config.runtime_root
        self.evidence_root = self.runtime_root / "evidence"
        self.checkpoint_root = self.runtime_root / "checkpoints"
        self.snapshot_root = self.runtime_root / "snapshots"
        for path in (self.evidence_root, self.checkpoint_root, self.snapshot_root):
            path.mkdir(mode=0o700, exist_ok=True)
            _canonical_root(path, label=f"query {path.name} root")
        self.journal = _QueryJournal(self.checkpoint_root / "query_journal.jsonl")
        self._target_context: TargetGeometryContext | None = None
        self._target_array: np.memmap | None = None
        self.target_freeze = self._load_target_freeze()
        self.selection_bindings = SelectionBindings(
            primary_query_sequence_id=primary_pair["query_sequence_id"],
            gt_sha256=query_pose_index.source_sha256,
            calibration_sha256=self.config.extrinsic_sha256,
            preprocessing_contract_sha256=self.config.preprocessing_contract_sha256,
            target_map_sha256=self.target_freeze["target_map_sha256"],
        ).validated()
        self._cleanup_resolved_temporaries()
        target_barrier = self.journal.barrier(TARGET_FROZEN)
        expected_target_barrier = {
            "target_freeze_file_sha256": self.config.expected_target_freeze_sha256,
            "target_map_sha256": self.target_freeze["target_map_sha256"],
            "target_context_resource_plan_sha256": (
                self.config.target_context_resource_plan_sha256
            ),
            "target_context_measurement_evidence_sha256": (
                self.config.target_context_measurement_evidence_sha256
            ),
            "target_context_measurement_provenance_sha256": (
                self.config.target_context_measurement_provenance_sha256
            ),
        }
        if target_barrier is None:
            if self.journal.event_count:
                raise BoreasV2Stage2QueryRunnerError(
                    "query journal did not start with TARGET_FROZEN"
                )
            self.disk_gate.before_checkpoint(
                4096, checkpoint_id="QUERY_TARGET_FROZEN_BARRIER"
            )
            self.journal.append(
                event_kind=TARGET_FROZEN,
                execution_stage="QUERY_BARRIER",
                object_key="",
                payload=expected_target_barrier,
            )
        elif target_barrier["payload"] != expected_target_barrier:
            raise BoreasV2Stage2QueryRunnerError("target-frozen barrier changed")

    def _assert_live(self) -> None:
        if not self.no_registration_guard.active:
            raise BoreasV2Stage2QueryRunnerError("NoRegistrationGuard became inactive")
        if self.authorization.no_registration_guard is not self.no_registration_guard:
            raise BoreasV2Stage2QueryRunnerError("authorization guard identity changed")
        try:
            self.authorization.assert_operation_live()
        except BoreasStage2AuthorizationError as exc:
            raise BoreasV2Stage2QueryRunnerError(
                "query authorization is no longer live"
            ) from exc

    @contextmanager
    def _phase_lock(self):
        if self.runtime_lease is not None:
            self.runtime_lease.assert_live()
            yield
            return
        path = self.runtime_root / ".boreas_v2_stage2.lock"
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            | (getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise BoreasV2Stage2QueryRunnerError("query runner lock is unsafe")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BoreasV2Stage2QueryRunnerError(
                    "another query runner process is active"
                ) from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _fault(self, point: str, item: AuthorizedRemoteObject | None = None) -> None:
        hook = self.dependencies.fault_hook
        if hook is not None:
            hook(point, item)

    def _load_target_freeze(self) -> dict[str, Any]:
        path = self.evidence_root / "target_map_freeze_manifest.json"
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or sha256_file(path) != self.config.expected_target_freeze_sha256
        ):
            raise BoreasV2Stage2QueryRunnerError(
                "authenticated target-map freeze manifest is absent or changed"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BoreasV2Stage2QueryRunnerError("target freeze JSON is invalid") from exc
        if path.read_bytes() != canonical_json_bytes(value):
            raise BoreasV2Stage2QueryRunnerError("target freeze JSON is noncanonical")
        fields = {
            "schema_version",
            "map_lineage_manifest_sha256",
            "preprocessing_contract_sha256",
            "target_map_reducer_verification_sha256",
            "unique_target_map_count",
            "physical_target_map_copy_count",
            "query_contribution_count",
            "target_map_immutable",
            "target_map_path",
            "target_map_sha256",
            "target_map_size_bytes",
            "target_point_count",
            "voxel_rule_sha256",
            "final_map_state_transition_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise BoreasV2Stage2QueryRunnerError("target freeze exact schema differs")
        expected = {
            "schema_version": TARGET_FREEZE_SCHEMA,
            "preprocessing_contract_sha256": self.config.preprocessing_contract_sha256,
            "unique_target_map_count": 1,
            "physical_target_map_copy_count": 1,
            "query_contribution_count": 0,
            "target_map_immutable": True,
        }
        for field, expected_value in expected.items():
            if value[field] != expected_value:
                raise BoreasV2Stage2QueryRunnerError(f"target freeze {field} differs")
        for field in (
            "map_lineage_manifest_sha256",
            "target_map_reducer_verification_sha256",
            "target_map_sha256",
            "voxel_rule_sha256",
            "final_map_state_transition_sha256",
        ):
            _sha(value[field], f"target freeze {field}")
        target_size = _strict_int(value["target_map_size_bytes"], "target size", minimum=1)
        target_count = _strict_int(value["target_point_count"], "target count", minimum=1)
        target_path = _safe_relative_file(
            self.runtime_root, str(value["target_map_path"]), label="target map"
        )
        if target_path.stat().st_size != target_size or sha256_file(target_path) != value[
            "target_map_sha256"
        ]:
            raise BoreasV2Stage2QueryRunnerError("target-map bytes differ from freeze")
        try:
            target = np.load(target_path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise BoreasV2Stage2QueryRunnerError("target map is not canonical NPY") from exc
        if (
            not isinstance(target, np.memmap)
            or target.shape != (target_count, 3)
            or target.dtype != np.dtype("<f8")
            or not target.flags.c_contiguous
        ):
            raise BoreasV2Stage2QueryRunnerError("target-map dtype/shape/layout differs")
        chunk = max(1, (8 * 1024 * 1024) // 24)
        for start in range(0, target_count, chunk):
            if not np.all(np.isfinite(target[start : start + chunk])):
                raise BoreasV2Stage2QueryRunnerError("target map contains nonfinite values")
        self._target_array = target
        return value

    def _context(self) -> TargetGeometryContext:
        if self._target_context is None:
            assert self._target_array is not None
            plan_path = self.evidence_root / "target_context_resource_plan.json"
            if (
                plan_path.is_symlink()
                or not plan_path.is_file()
                or plan_path.resolve(strict=True) != plan_path
                or sha256_file(plan_path)
                != self.config.target_context_resource_plan_sha256
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "authenticated target-context resource plan is absent or changed"
                )
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise BoreasV2Stage2QueryRunnerError(
                    "target-context resource plan is invalid"
                ) from exc
            if plan_path.read_bytes() != canonical_json_bytes(plan):
                raise BoreasV2Stage2QueryRunnerError(
                    "target-context resource plan is noncanonical"
                )
            expected_fields = {
                "estimated_peak_memory_bytes",
                "measurement_evidence_sha256",
                "minimum_live_available_memory_bytes",
                "numpy_version",
                "production_approved",
                "safety_margin_bytes",
                "schema",
                "scipy_version",
                "target_map_sha256",
                "target_point_count",
            }
            if not isinstance(plan, dict) or set(plan) != expected_fields:
                raise BoreasV2Stage2QueryRunnerError(
                    "target-context resource plan exact schema differs"
                )
            expected_peak = _strict_int(
                plan["estimated_peak_memory_bytes"],
                "target-context estimated peak memory",
                minimum=1,
            )
            margin = _strict_int(
                plan["safety_margin_bytes"], "target-context safety margin", minimum=1
            )
            minimum = _strict_int(
                plan["minimum_live_available_memory_bytes"],
                "target-context live memory threshold",
                minimum=1,
            )
            if (
                plan["schema"] != "zprm.boreas.v2.stage2.target_context_resource_plan.v1"
                or plan["target_map_sha256"] != self.target_freeze["target_map_sha256"]
                or plan["target_point_count"] != self.target_freeze["target_point_count"]
                or plan["numpy_version"] != np.__version__
                or plan["scipy_version"]
                != __import__("scipy", fromlist=["__version__"]).__version__
                or minimum != expected_peak + margin
                or plan["production_approved"] is not True
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "target-context resource authority differs"
                )
            _sha(plan["measurement_evidence_sha256"], "memory measurement evidence SHA")
            if (
                plan["measurement_evidence_sha256"]
                != self.config.target_context_measurement_evidence_sha256
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "target-context measurement binding differs"
                )
            evidence_path = (
                self.evidence_root / "target_context_capacity_measurement.json"
            )
            if (
                evidence_path.is_symlink()
                or not evidence_path.is_file()
                or evidence_path.resolve(strict=True) != evidence_path
                or sha256_file(evidence_path)
                != self.config.target_context_measurement_evidence_sha256
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "QUERY_TARGET_CONTEXT_CAPACITY_NOT_READY: authenticated measurement is absent"
                )
            try:
                measured = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise BoreasV2Stage2QueryRunnerError(
                    "target-context capacity measurement is invalid"
                ) from exc
            measurement_fields = {
                "measured_peak_memory_bytes",
                "measurement_method",
                "numpy_version",
                "schema",
                "scipy_version",
                "target_map_sha256",
                "target_point_count",
            }
            if (
                evidence_path.read_bytes() != canonical_json_bytes(measured)
                or not isinstance(measured, dict)
                or set(measured) != measurement_fields
                or measured["schema"]
                != "zprm.boreas.v2.stage2.target_context_capacity_measurement.v1"
                or measured["measurement_method"]
                != "MAX_RSS_PREPARE_NORMALS_KDTREE_SAME_PINNED_ENVIRONMENT"
                or measured["target_map_sha256"]
                != self.target_freeze["target_map_sha256"]
                or measured["target_point_count"]
                != self.target_freeze["target_point_count"]
                or measured["numpy_version"] != np.__version__
                or measured["scipy_version"] != plan["scipy_version"]
                or _strict_int(
                    measured["measured_peak_memory_bytes"],
                    "measured target-context peak memory",
                    minimum=1,
                )
                != expected_peak
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "target-context capacity measurement semantics differ"
                )
            provider = self.dependencies.available_memory_provider
            if provider is None:
                def provider() -> int:
                    try:
                        for line in Path("/proc/meminfo").read_text(
                            encoding="ascii"
                        ).splitlines():
                            if line.startswith("MemAvailable:"):
                                fields = line.split()
                                if len(fields) == 3 and fields[2] == "kB":
                                    return int(fields[1]) * 1024
                    except (OSError, UnicodeError, ValueError) as exc:
                        raise BoreasV2Stage2QueryRunnerError(
                            "cannot read live available memory"
                        ) from exc
                    raise BoreasV2Stage2QueryRunnerError(
                        "MemAvailable is absent from /proc/meminfo"
                    )
            live = provider()
            if isinstance(live, bool) or not isinstance(live, int) or live < minimum:
                raise BoreasV2Stage2QueryRunnerError(
                    "BLOCKED_INSUFFICIENT_MEMORY: target context would cross the approved threshold"
                )
            context = self.dependencies.target_context_factory(self._target_array)
            if not isinstance(context, TargetGeometryContext):
                raise BoreasV2Stage2QueryRunnerError(
                    "target context factory returned an invalid context"
                )
            observed_peak = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ) * 1024
            if observed_peak <= 0 or (
                self.config.production_mode and observed_peak > minimum
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "BLOCKED_INSUFFICIENT_MEMORY: observed target-context MaxRSS "
                    "exceeds the frozen capacity threshold"
                )
            live_after = provider()
            if (
                isinstance(live_after, bool)
                or not isinstance(live_after, int)
                or live_after < margin
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "BLOCKED_INSUFFICIENT_MEMORY: target context left less than "
                    "the fixed safety reserve"
                )
            self._target_context = context
        return self._target_context

    def preflight_target_context(self) -> TargetGeometryContext:
        """Run the frozen target/RAM authority before any query download."""

        self._assert_live()
        return self._context()

    def _temporary_for_download(
        self, event: Mapping[str, Any], item: AuthorizedRemoteObject
    ) -> TemporaryDownloadedObject | None:
        receipt = _validated_receipt(event["payload"]["receipt"])
        if (
            receipt.key != item.key
            or receipt.sequence_id != item.frozen.sequence_id
            or receipt.timestamp_us != item.frozen.timestamp_us
            or receipt.remote_size_bytes != item.size_bytes
            or receipt.etag != item.etag
            or receipt.last_modified != item.last_modified
        ):
            raise BoreasV2Stage2QueryRunnerError("resumed receipt/allowlist identity differs")
        relative = str(event["payload"]["temporary_relative_path"])
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise BoreasV2Stage2QueryRunnerError("resumed temporary path is unsafe")
        path = self.runtime_root.joinpath(*pure.parts)
        expected_name = (
            f"{item.frozen.ordinal:06d}-"
            f"{hashlib.sha256(item.key.encode('utf-8')).hexdigest()[:24]}.bin.partial"
        )
        if path.parent != self.downloader.temporary_root or path.name != expected_name:
            raise BoreasV2Stage2QueryRunnerError("resumed temporary path differs")
        if not path.exists():
            return None
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or path.stat().st_size != receipt.remote_size_bytes
            or sha256_file(path) != receipt.local_temporary_sha256
        ):
            raise BoreasV2Stage2QueryRunnerError("resumed raw payload is unsafe or changed")
        return TemporaryDownloadedObject(item, path, receipt)

    def _append_abort(self, event: Mapping[str, Any], *, reason: str) -> None:
        receipt = _validated_receipt(event["payload"]["receipt"])
        self.disk_gate.before_checkpoint(
            4096, checkpoint_id=f"QUERY_ABORT:{receipt.key}"
        )
        self.journal.append(
            event_kind="ABORTED",
            execution_stage=str(event["execution_stage"]),
            object_key=receipt.key,
            payload={
                "download_event_sha256": event["event_sha256"],
                "reason": str(reason),
                "receipt_sha256": receipt.as_dict()["receipt_sha256"],
            },
        )

    def _materialize(
        self, stage: str, item: AuthorizedRemoteObject
    ) -> tuple[TemporaryDownloadedObject, dict[str, Any], bool]:
        self._assert_live()
        unresolved = self.journal.unresolved
        if unresolved:
            if len(unresolved) != 1:
                raise BoreasV2Stage2QueryRunnerError(
                    "multiple unresolved query downloads exist"
                )
            event = unresolved[0]
            if event["execution_stage"] != stage or event["object_key"] != item.key:
                raise BoreasV2Stage2QueryRunnerError(
                    "unresolved query download is not the next frozen object"
                )
            materialized = self._temporary_for_download(event, item)
            if materialized is not None:
                return materialized, event, False
            self._append_abort(event, reason="TEMPORARY_PAYLOAD_ABSENT_ON_RESUME")
        pending_intents = self.journal.pending_transfer_intents
        if pending_intents:
            if len(pending_intents) != 1:
                raise BoreasV2Stage2QueryRunnerError(
                    "multiple unresolved query transfer intents exist"
                )
            intent = pending_intents[0]
            if intent["execution_stage"] != stage or intent["object_key"] != item.key:
                raise BoreasV2Stage2QueryRunnerError(
                    "unresolved transfer intent is not the next frozen object"
                )
            relative = PurePosixPath(intent["payload"]["temporary_relative_path"])
            path = self.runtime_root.joinpath(*relative.parts)
            if path.exists():
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.resolve(strict=True) != path
                    or path.parent != self.downloader.temporary_root
                    or os.lstat(path).st_nlink != 1
                ):
                    raise BoreasV2Stage2QueryRunnerError(
                        "transfer-intent payload is unsafe or changed"
                    )
                # Bare intent has no authenticated payload SHA/receipt.  A
                # same-size file may be malicious or incomplete and is never
                # upgraded into a DOWNLOADED event on resume.
                path.unlink()
                _fsync_directory(path.parent)
            self.disk_gate.before_checkpoint(
                4096, checkpoint_id=f"QUERY_TRANSFER_ABORT:{stage}:{item.key}"
            )
            self.journal.append(
                event_kind=TRANSFER_ABORTED,
                execution_stage=stage,
                object_key=item.key,
                payload={
                    "reason": "UNAUTHENTICATED_BARE_INTENT_DISCARDED_ON_RESUME",
                    "transfer_intent_event_sha256": intent["event_sha256"],
                },
            )
        planned_path = self.downloader.planned_temporary_path(item)
        if planned_path.exists() or planned_path.is_symlink():
            raise BoreasV2Stage2QueryRunnerError(
                "unowned planned query temporary payload already exists"
            )
        planned_relative = planned_path.relative_to(self.runtime_root).as_posix()
        self.disk_gate.before_checkpoint(
            4096, checkpoint_id=f"QUERY_TRANSFER_INTENT:{stage}:{item.key}"
        )
        intent = self.journal.append(
            event_kind=TRANSFER_INTENT,
            execution_stage=stage,
            object_key=item.key,
            payload={
                "etag": item.etag,
                "last_modified": item.last_modified,
                "remote_size_bytes": item.size_bytes,
                "sequence_id": item.frozen.sequence_id,
                "temporary_relative_path": planned_relative,
                "timestamp_us": item.frozen.timestamp_us,
            },
        )
        sink_event: dict[str, Any] = {}

        def receipt_sink(value: TemporaryDownloadedObject) -> None:
            relative = value.path.relative_to(self.runtime_root).as_posix()
            self.disk_gate.before_checkpoint(
                8192, checkpoint_id=f"QUERY_DOWNLOADED:{stage}:{item.key}"
            )
            sink_event.update(
                self.journal.append(
                    event_kind="DOWNLOADED",
                    execution_stage=stage,
                    object_key=item.key,
                    payload={
                        "receipt": value.receipt.as_dict(),
                        "temporary_relative_path": relative,
                        "transfer_intent_event_sha256": intent["event_sha256"],
                    },
                )
            )

        materialized = self.downloader.materialize(
            item, authenticated_receipt_sink=receipt_sink
        )
        try:
            relative = materialized.path.relative_to(self.runtime_root).as_posix()
        except ValueError as exc:
            raise BoreasV2Stage2QueryRunnerError(
                "download temporary path escapes runtime root"
            ) from exc
        if materialized.path != planned_path or relative != planned_relative:
            raise BoreasV2Stage2QueryRunnerError(
                "downloader materialized outside the durable transfer intent"
            )
        if not sink_event:
            # Backwards-compatible synthetic DI only.  Production pins the
            # strict downloader implementation and therefore always executes
            # the authenticated sink before materialize returns.
            receipt_sink(materialized)
        event = sink_event
        return materialized, event, True

    def _cleanup_resolved_temporaries(self) -> None:
        rows = self.journal.read(full_rescan=True)
        unresolved = {row["event_sha256"] for row in self.journal.unresolved}
        authorized = self.inventory.by_key
        known_paths: set[Path] = set()
        for row in rows:
            if row["event_kind"] not in {TRANSFER_INTENT, "DOWNLOADED"}:
                continue
            item = authorized.get(row["object_key"])
            if item is None:
                raise BoreasV2Stage2QueryRunnerError(
                    "journal names an object outside the reconciled inventory"
                )
            relative = PurePosixPath(row["payload"]["temporary_relative_path"])
            path = self.runtime_root.joinpath(*relative.parts)
            known_paths.add(path)
            if row["event_kind"] == TRANSFER_INTENT:
                continue
            if row["event_sha256"] in unresolved or not path.exists():
                continue
            materialized = self._temporary_for_download(row, item)
            if materialized is not None:
                self.downloader.release(materialized)
        for path in self.downloader.temporary_root.iterdir():
            if path.name == ".stage2_temporary_root.json":
                continue
            if path not in known_paths:
                raise BoreasV2Stage2QueryRunnerError(
                    "unknown temporary payload exists; refusing broad cleanup"
                )

    def _first_pass_commits(self) -> tuple[dict[str, Any], ...]:
        rows = tuple(
            row for row in self.journal.committed if row["event_kind"] == FIRST_PASS_COMMIT
        )
        if len(rows) > len(self.query_objects):
            raise BoreasV2Stage2QueryRunnerError("too many first-pass commits")
        for expected, (event, item) in enumerate(zip(rows, self.query_objects)):
            first_pass = event["payload"]["first_pass_row"]
            if (
                event["execution_stage"] != QUERY_FIRST_PASS_STAGE
                or event["object_key"] != item.key
                or first_pass["query_ordinal"] != expected
                or first_pass["object_key"] != item.key
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "first-pass journal does not follow frozen query order"
                )
        return rows

    def _first_pass_row(
        self,
        *,
        item: AuthorizedRemoteObject,
        materialized: TemporaryDownloadedObject,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        scan = self.dependencies.query_preprocessor(
            materialized, item, self.query_pose_index
        )
        if not isinstance(scan, PreprocessedBoreasScan):
            raise BoreasV2Stage2QueryRunnerError(
                "query preprocessor returned an invalid scan"
            )
        if (
            scan.role != "QUERY"
            or scan.object_key != item.key
            or scan.sequence_id != item.frozen.sequence_id
            or scan.timestamp_us != item.frozen.timestamp_us
        ):
            raise BoreasV2Stage2QueryRunnerError("preprocessed query identity differs")
        geometry = dict(self.dependencies.geometry_computer(scan, self._context()))
        if set(geometry) != set(GEOMETRY_ONLY_FIELDS):
            raise BoreasV2Stage2QueryRunnerError(
                "geometry callback exposed fields outside the frozen ten"
            )
        distance, _ = self.map_reference_tree.query(
            np.asarray(scan.t_reference[:3, 3], dtype=np.float64), k=1
        )
        if not np.isfinite(distance):
            raise BoreasV2Stage2QueryRunnerError("GT overlap distance is nonfinite")
        row = {
            "query_ordinal": item.frozen.role_ordinal,
            "sequence_id": item.frozen.sequence_id,
            "object_key": item.key,
            "timestamp_us": item.frozen.timestamp_us,
            "remote_size_bytes": item.size_bytes,
            "last_modified": item.last_modified,
            "etag": item.etag,
            "payload_sha256": materialized.receipt.local_temporary_sha256,
            "finite_source_point_count": int(scan.points_xyz.shape[0]),
            "target_map_point_count": int(self.target_freeze["target_point_count"]),
            **geometry,
            "reference_interpolation_valid": True,
            "reference_gap_s": 0.0,
            "gt_overlap_within_5m": bool(
                float(distance) <= self.selection_contract.gt_overlap_radius_m
            ),
            "target_map_frozen_complete": True,
            "deskew_processing_contract_valid": True,
            "gt_sha256": self.selection_bindings.gt_sha256,
            "calibration_sha256": self.selection_bindings.calibration_sha256,
            "preprocessing_contract_sha256": (
                self.selection_bindings.preprocessing_contract_sha256
            ),
            "target_map_sha256": self.selection_bindings.target_map_sha256,
        }
        if set(row) != FIRST_PASS_SCAN_FIELDS:
            raise AssertionError("first-pass producer schema drift")
        # A separate orchestration path starts again from authenticated raw
        # bytes and independently serializes the shared frozen primitives.  It
        # is intentionally not an algorithmically different preprocessing
        # implementation; the exact-byte/metric comparison catches producer
        # orchestration, persistence, and field-wiring faults before deletion.
        independent = self.dependencies.independent_query_preprocessor(
            materialized, item, self.query_pose_index
        )
        independent_geometry = dict(
            self.dependencies.independent_geometry_computer(
                independent, self._context()
            )
        )
        if independent_geometry != geometry:
            raise BoreasV2Stage2QueryRunnerError(
                "producer and dual-path geometry metrics differ"
            )
        producer_source = canonical_source_npy_bytes(scan)
        independent_source = canonical_source_npy_bytes(independent)
        producer_transform = canonical_t_reference_npy_bytes(scan)
        independent_transform = canonical_t_reference_npy_bytes(independent)
        if producer_source != independent_source or producer_transform != independent_transform:
            raise BoreasV2Stage2QueryRunnerError(
                "producer and dual-path first-pass canonical bytes differ"
            )
        witness_core = {
            "query_ordinal": item.frozen.role_ordinal,
            "object_key": item.key,
            "raw_payload_sha256": materialized.receipt.local_temporary_sha256,
            "receipt_sha256": materialized.receipt.as_dict()["receipt_sha256"],
            "preprocessing_contract_sha256": self.config.preprocessing_contract_sha256,
            "gt_sha256": self.selection_bindings.gt_sha256,
            "extrinsic_sha256": self.config.extrinsic_sha256,
            "target_map_sha256": self.target_freeze["target_map_sha256"],
            "producer_source_sha256": hashlib.sha256(producer_source).hexdigest(),
            "independent_source_sha256": hashlib.sha256(independent_source).hexdigest(),
            "producer_T_reference_sha256": hashlib.sha256(producer_transform).hexdigest(),
            "independent_T_reference_sha256": hashlib.sha256(independent_transform).hexdigest(),
            "producer_geometry_sha256": _hash_json(geometry),
            "independent_geometry_sha256": _hash_json(independent_geometry),
            "raw_point_count": independent.raw_point_count,
            "nonfinite_excluded_count": independent.nonfinite_excluded_count,
            "range_excluded_count": independent.range_excluded_count,
            "post_filter_point_count": independent.post_filter_point_count,
            "source_voxel_reduced_count": independent.source_voxel_reduced_count,
            "canonical_source_point_count": int(independent.points_xyz.shape[0]),
            "verification_status": GEOMETRY_WITNESS_STATUS,
        }
        witness = {
            **witness_core,
            "geometry_metric_verification_row_sha256": _hash_json(witness_core),
        }
        return row, witness

    def run_first_pass(self, *, object_limit: int | None = None) -> QueryPhaseSummary:
        """Screen the fixed query stream; ``object_limit`` is synthetic-only."""

        if object_limit is not None:
            if self.config.production_mode:
                raise BoreasV2Stage2QueryRunnerError(
                    "partial query limits are forbidden in production"
                )
            _strict_int(object_limit, "object_limit", minimum=1)
        with self._phase_lock():
            self._assert_live()
            # Must precede the first downloader/materialize call.  This is a
            # production resource authorization, not a lazy scientific step.
            self.preflight_target_context()
            commits = self._first_pass_commits()
            pending = self.query_objects[len(commits) :]
            if object_limit is not None:
                pending = pending[:object_limit]
            downloaded = 0
            deleted = 0
            for item in pending:
                materialized, download_event, is_new = self._materialize(
                    QUERY_FIRST_PASS_STAGE, item
                )
                if is_new:
                    downloaded += item.size_bytes
                try:
                    self._fault("QUERY_FIRST_PASS_AFTER_DOWNLOADED", item)
                    first_pass, geometry_witness = self._first_pass_row(
                        item=item, materialized=materialized
                    )
                    processing_sha = _hash_json(first_pass)
                    self.disk_gate.before_checkpoint(
                        max(8192, len(canonical_json_bytes(first_pass))),
                        checkpoint_id=f"QUERY_FIRST_PASS_COMMIT:{item.key}",
                    )
                    self.journal.append(
                        event_kind=FIRST_PASS_COMMIT,
                        execution_stage=QUERY_FIRST_PASS_STAGE,
                        object_key=item.key,
                        payload={
                            "download_event_sha256": download_event["event_sha256"],
                            "first_pass_row": first_pass,
                            "geometry_metric_verification_row": geometry_witness,
                            "processing_result_sha256": processing_sha,
                            "receipt_sha256": materialized.receipt.as_dict()[
                                "receipt_sha256"
                            ],
                        },
                    )
                    self._fault("QUERY_FIRST_PASS_AFTER_COMMIT", item)
                finally:
                    if materialized.path.exists():
                        deleted += self.downloader.release(materialized)
            commits = self._first_pass_commits()
            if len(commits) == len(self.query_objects):
                barrier = self.journal.barrier(FIRST_PASS_COMPLETE)
                payload = {
                    "first_pass_count": len(commits),
                    "first_pass_rows_sha256": _hash_json(
                        [row["payload"]["first_pass_row"] for row in commits]
                    ),
                    "target_freeze_event_sha256": self.journal.barrier(TARGET_FROZEN)[
                        "event_sha256"
                    ],
                }
                if barrier is None:
                    self.disk_gate.before_checkpoint(
                        4096, checkpoint_id="QUERY_FIRST_PASS_COMPLETE_BARRIER"
                    )
                    self.journal.append(
                        event_kind=FIRST_PASS_COMPLETE,
                        execution_stage="QUERY_BARRIER",
                        object_key="",
                        payload=payload,
                    )
                elif barrier["payload"] != payload:
                    raise BoreasV2Stage2QueryRunnerError(
                        "first-pass-complete barrier changed"
                    )
            return QueryPhaseSummary(
                completed_object_count=len(commits),
                total_object_count=len(self.query_objects),
                downloaded_bytes_this_run=downloaded,
                deleted_temporary_bytes_this_run=deleted,
                final_journal_event_sha256=(
                    self.journal.final_event_sha256
                ),
            )

    def _selection_paths(self) -> dict[str, Path]:
        return {
            name: self.evidence_root / name
            for name in (
                "all_candidate_scans.csv",
                "all_candidate_intervals.csv",
                "excluded_candidate_scans.csv",
                "excluded_intervals.csv",
                "geometry_only_metrics.csv",
                "geometry_metric_verification.csv",
                "first_pass_selection_projection.csv",
                "selected_scene_intervals.csv",
                "selected_snapshots.csv",
                "boreas_v2_stage2_selection_freeze.json",
            )
        }

    def freeze_selection(self) -> dict[str, Any]:
        """Freeze the 246/20/100 R14 result before any selected-source download."""

        with self._phase_lock():
            self._assert_live()
            if any(
                row["event_kind"] == SELECTED_SOURCE_COMMIT
                for row in self.journal.committed
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "cannot (re)freeze selection after second-pass materialization"
                )
            first_barrier = self.journal.barrier(FIRST_PASS_COMPLETE)
            commits = self._first_pass_commits()
            if first_barrier is None or len(commits) != len(self.query_objects):
                raise BoreasV2Stage2QueryRunnerError(
                    "selection requires the complete fixed first pass"
                )
            first_pass_rows = [row["payload"]["first_pass_row"] for row in commits]
            scans, excluded_scans = build_candidate_scan_inventory(
                first_pass_rows,
                self.frozen_windows,
                bindings=self.selection_bindings,
                contract=self.selection_contract,
            )
            intervals, excluded_intervals = build_candidate_intervals(
                scans,
                self.frozen_windows,
                self.reference_pose_series,
                contract=self.selection_contract,
            )
            selected_intervals = select_frozen_intervals(
                intervals, contract=self.selection_contract
            )
            selected_snapshots = select_interval_quantile_snapshots(
                scans, selected_intervals, contract=self.selection_contract
            )
            if (
                len(scans) != len(self.query_objects)
                or len(intervals) != 246
                or len(selected_intervals) != 20
                or len(selected_snapshots) != 100
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "selection result does not close 11,859/246/20/100"
                )
            paths = self._selection_paths()
            artifacts: dict[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]] = {
                "all_candidate_scans.csv": (
                    CANDIDATE_SCAN_CSV_FIELDS,
                    candidate_scan_csv_rows(scans),
                ),
                "all_candidate_intervals.csv": (
                    CANDIDATE_INTERVAL_CSV_FIELDS,
                    candidate_interval_csv_rows(intervals),
                ),
                "excluded_candidate_scans.csv": (
                    CANDIDATE_SCAN_CSV_FIELDS,
                    candidate_scan_csv_rows(excluded_scans),
                ),
                "excluded_intervals.csv": (
                    CANDIDATE_INTERVAL_CSV_FIELDS,
                    candidate_interval_csv_rows(excluded_intervals),
                ),
                "geometry_only_metrics.csv": (
                    GEOMETRY_METRIC_CSV_FIELDS,
                    geometry_metric_csv_rows(scans),
                ),
                "geometry_metric_verification.csv": (
                    GEOMETRY_METRIC_VERIFICATION_FIELDS,
                    [
                        row["payload"]["geometry_metric_verification_row"]
                        for row in commits
                    ],
                ),
                # Written below after the barrier identity is known.  Its
                # provisional binding uses the deterministic would-be event
                # identity to avoid mutating historical first-pass events.
                "selected_scene_intervals.csv": (
                    SELECTED_INTERVAL_CSV_FIELDS,
                    selected_interval_csv_rows(selected_intervals),
                ),
                "selected_snapshots.csv": (
                    SELECTED_SNAPSHOT_CSV_FIELDS,
                    selected_snapshot_csv_rows(selected_snapshots),
                ),
            }
            projected = sum(
                len(csv_bytes(rows, fields)) for fields, rows in artifacts.values()
            ) + 64 * 1024
            self.disk_gate.before_checkpoint(
                projected, checkpoint_id="R14_SELECTION_FREEZE_ARTIFACTS"
            )
            for name, (fields, rows) in artifacts.items():
                atomic_write_csv(paths[name], rows, fields, overwrite=False)
            blind = build_blind_selection_manifest(
                candidate_scans=scans,
                candidate_intervals=intervals,
                selected_intervals=selected_intervals,
                selected_snapshots=selected_snapshots,
                contract=self.selection_contract,
                bindings=self.selection_bindings,
            )
            artifact_sha = {
                name: sha256_file(paths[name]) for name in sorted(artifacts)
            }
            core = {
                "schema": SELECTION_FREEZE_SCHEMA,
                "target_freeze_file_sha256": self.config.expected_target_freeze_sha256,
                "first_pass_complete_event_sha256": first_barrier["event_sha256"],
                "selection_contract_sha256": self.selection_contract.sha256,
                "candidate_scan_count": len(scans),
                "candidate_scan_rows_sha256": blind["candidate_scan_rows_sha256"],
                "candidate_interval_count": len(intervals),
                "candidate_interval_rows_sha256": blind[
                    "candidate_interval_rows_sha256"
                ],
                "selected_interval_count": len(selected_intervals),
                "selected_interval_rows_sha256": blind[
                    "selected_interval_rows_sha256"
                ],
                "selected_snapshot_count": len(selected_snapshots),
                "selected_snapshot_rows_sha256": blind[
                    "selected_snapshot_rows_sha256"
                ],
                "artifact_sha256": artifact_sha,
                "blind_selection": blind,
                "R14_frozen": True,
                "registration_execution_count": 0,
            }
            freeze = {
                **core,
                "selection_freeze_payload_sha256": _hash_json(core),
            }
            atomic_write_json(paths["boreas_v2_stage2_selection_freeze.json"], freeze)
            file_sha = sha256_file(paths["boreas_v2_stage2_selection_freeze.json"])
            barrier_payload = {
                "artifact_sha256": artifact_sha,
                "first_pass_complete_event_sha256": first_barrier["event_sha256"],
                "selected_snapshot_count": 100,
                "selection_contract_sha256": self.selection_contract.sha256,
                "selection_freeze_file_sha256": file_sha,
            }
            barrier = self.journal.barrier(SELECTION_FROZEN)
            if barrier is None:
                self.disk_gate.before_checkpoint(
                    4096, checkpoint_id="R14_SELECTION_FROZEN_BARRIER"
                )
                barrier = self.journal.append(
                    event_kind=SELECTION_FROZEN,
                    execution_stage="QUERY_BARRIER",
                    object_key="",
                    payload=barrier_payload,
                )
            elif barrier["payload"] != barrier_payload:
                raise BoreasV2Stage2QueryRunnerError(
                    "R14 selection barrier changed"
                )
            projection_rows: list[dict[str, Any]] = []
            for commit, scan in zip(commits, scans):
                projection_core = {
                    "execution_stage": QUERY_FIRST_PASS_STAGE,
                    "query_ordinal": int(scan["query_ordinal"]),
                    "object_key": str(scan["object_key"]),
                    "first_pass_event_sha256": commit["event_sha256"],
                    "first_pass_result_sha256": commit["payload"][
                        "processing_result_sha256"
                    ],
                    "candidate_scan_row_sha256": scan[
                        "candidate_scan_row_sha256"
                    ],
                    "selection_freeze_event_sha256": barrier["event_sha256"],
                }
                projection_rows.append(
                    {**projection_core, "projection_row_sha256": _hash_json(projection_core)}
                )
            projection_path = paths["first_pass_selection_projection.csv"]
            self.disk_gate.before_checkpoint(
                len(csv_bytes(projection_rows, FIRST_PASS_SELECTION_PROJECTION_FIELDS)),
                checkpoint_id="FIRST_PASS_SELECTION_PROJECTION",
            )
            atomic_write_csv(
                projection_path,
                projection_rows,
                FIRST_PASS_SELECTION_PROJECTION_FIELDS,
                overwrite=False,
            )
            # The projection is a post-barrier derivative and therefore binds
            # the barrier rather than being recursively included in it.
            return {
                **freeze,
                "selection_frozen_event_sha256": barrier["event_sha256"],
                "first_pass_selection_projection_sha256": sha256_file(projection_path),
            }

    def _load_frozen_selection(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        barrier = self.journal.barrier(SELECTION_FROZEN)
        if barrier is None:
            raise BoreasV2Stage2QueryRunnerError(
                "second pass is forbidden before durable R14_SELECTION_FROZEN"
            )
        paths = self._selection_paths()
        freeze_path = paths["boreas_v2_stage2_selection_freeze.json"]
        if (
            freeze_path.is_symlink()
            or not freeze_path.is_file()
            or sha256_file(freeze_path)
            != barrier["payload"]["selection_freeze_file_sha256"]
        ):
            raise BoreasV2Stage2QueryRunnerError("selection freeze bytes changed")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze_path.read_bytes() != canonical_json_bytes(freeze):
            raise BoreasV2Stage2QueryRunnerError("selection freeze is noncanonical")
        unsigned = dict(freeze)
        claimed = unsigned.pop("selection_freeze_payload_sha256", None)
        if claimed != _hash_json(unsigned):
            raise BoreasV2Stage2QueryRunnerError("selection freeze self-hash differs")
        artifact_sha = barrier["payload"]["artifact_sha256"]
        if freeze.get("artifact_sha256") != artifact_sha:
            raise BoreasV2Stage2QueryRunnerError("selection artifact binding differs")
        for name, expected in artifact_sha.items():
            if name not in paths or sha256_file(paths[name]) != expected:
                raise BoreasV2Stage2QueryRunnerError(
                    f"frozen selection artifact changed: {name}"
                )
        with paths["selected_snapshots.csv"].open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != SELECTED_SNAPSHOT_CSV_FIELDS:
                raise BoreasV2Stage2QueryRunnerError(
                    "selected snapshot CSV header differs"
                )
            rows = [dict(row) for row in reader]
        if len(rows) != 100 or len({row["object_key"] for row in rows}) != 100:
            raise BoreasV2Stage2QueryRunnerError(
                "frozen selected snapshot inventory is not exactly 100 unique objects"
            )
        return freeze, rows

    def _download_events_by_stage_key(self) -> dict[tuple[str, str], dict[str, Any]]:
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.journal.read():
            if row["event_kind"] != "DOWNLOADED":
                continue
            key = (str(row["execution_stage"]), str(row["object_key"]))
            # An ABORTED retry is intentionally superseded by the later
            # successful download.  Commit lookup below selects the event SHA.
            result[key] = row
        return result

    def _first_receipt_for_key(self, key: str) -> DownloadReceipt:
        commits = {
            row["object_key"]: row for row in self._first_pass_commits()
        }
        commit = commits.get(key)
        if commit is None:
            raise BoreasV2Stage2QueryRunnerError(
                "selected source lacks first-pass receipt"
            )
        download_sha = commit["payload"]["download_event_sha256"]
        download = self.journal.download_for_event(download_sha)
        if download is None:
            raise BoreasV2Stage2QueryRunnerError(
                "first-pass download event is absent"
            )
        return _validated_receipt(download["payload"]["receipt"])

    def _selected_commits(self) -> tuple[dict[str, Any], ...]:
        rows = tuple(
            row
            for row in self.journal.committed
            if row["event_kind"] == SELECTED_SOURCE_COMMIT
        )
        _, selected = self._load_frozen_selection()
        if len(rows) > len(selected):
            raise BoreasV2Stage2QueryRunnerError("too many selected-source commits")
        for event, snapshot in zip(rows, selected):
            if (
                event["execution_stage"] != QUERY_SECOND_PASS_STAGE
                or event["object_key"] != snapshot["object_key"]
                or event["payload"]["selected_snapshot_row_sha256"]
                != snapshot["selected_snapshot_row_sha256"]
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "selected-source journal order differs from frozen selection"
                )
        return rows

    def _canonical_bundle(
        self,
        *,
        snapshot: Mapping[str, str],
        item: AuthorizedRemoteObject,
        materialized: TemporaryDownloadedObject,
        first_receipt: DownloadReceipt,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if first_receipt.payload_identity() != materialized.receipt.payload_identity():
            raise BoreasV2Stage2QueryRunnerError(
                "selected-source payload differs from first-pass receipt"
            )
        scan = self.dependencies.query_preprocessor(
            materialized, item, self.query_pose_index
        )
        if (
            not isinstance(scan, PreprocessedBoreasScan)
            or scan.role != "QUERY"
            or scan.object_key != item.key
        ):
            raise BoreasV2Stage2QueryRunnerError(
                "selected-source preprocessing identity differs"
            )
        source_bytes = canonical_source_npy_bytes(scan)
        transform_bytes = canonical_t_reference_npy_bytes(scan)
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        transform_sha = hashlib.sha256(transform_bytes).hexdigest()
        selection_index = int(snapshot["selection_index"])
        snapshot_id = snapshot["snapshot_id"]
        relative_root = PurePosixPath("snapshots") / snapshot_id
        source_relative = (relative_root / "source_points.npy").as_posix()
        transform_relative = (relative_root / "T_reference.npy").as_posix()
        metadata_relative = (relative_root / "metadata.json").as_posix()
        source_path = self.runtime_root / source_relative
        transform_path = self.runtime_root / transform_relative
        metadata_path = self.runtime_root / metadata_relative
        bundle_root = source_path.parent
        snapshots_root = self.snapshot_root
        if snapshots_root.is_symlink() or snapshots_root.resolve(strict=True) != snapshots_root:
            raise BoreasV2Stage2QueryRunnerError("snapshot root became unsafe")
        if bundle_root.exists():
            if (
                bundle_root.is_symlink()
                or not bundle_root.is_dir()
                or bundle_root.resolve(strict=True) != bundle_root
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "snapshot bundle directory is unsafe"
                )
        else:
            bundle_root.mkdir(mode=0o700)
            _fsync_directory(snapshots_root)
        for path in (source_path, transform_path, metadata_path):
            if path.parent != bundle_root or path.is_symlink():
                raise BoreasV2Stage2QueryRunnerError(
                    "snapshot artifact path is unsafe"
                )
        projected = len(source_bytes) + len(transform_bytes) + 16 * 1024
        self.disk_gate.before_materialization(
            projected, artifact_id=f"SELECTED_CANONICAL_BUNDLE:{snapshot_id}"
        )
        atomic_write_bytes(source_path, source_bytes, overwrite=False)
        atomic_write_bytes(transform_path, transform_bytes, overwrite=False)
        metadata = {
            "snapshot_id": snapshot_id,
            "object_key": item.key,
            "source_points_sha256": source_sha,
            "T_reference_sha256": transform_sha,
            "target_map_sha256": self.target_freeze["target_map_sha256"],
        }
        atomic_write_json(metadata_path, metadata, overwrite=False)
        metadata_sha = sha256_file(metadata_path)
        row: dict[str, Any] = {
            "selection_index": selection_index,
            "snapshot_id": snapshot_id,
            "scene_label": snapshot["scene_label"],
            "interval_id": snapshot["interval_id"],
            "object_key": item.key,
            "source_points_path": source_relative,
            "source_points_sha256": source_sha,
            "source_points_size_bytes": len(source_bytes),
            "source_point_count": int(scan.points_xyz.shape[0]),
            "T_reference_path": transform_relative,
            "T_reference_sha256": transform_sha,
            "T_reference_size_bytes": len(transform_bytes),
            "snapshot_metadata_path": metadata_relative,
            "snapshot_metadata_sha256": metadata_sha,
            "target_map_path": self.target_freeze["target_map_path"],
            "target_map_sha256": self.target_freeze["target_map_sha256"],
            "target_map_size_bytes": self.target_freeze["target_map_size_bytes"],
            "target_point_count": self.target_freeze["target_point_count"],
            "preprocessing_contract_sha256": self.config.preprocessing_contract_sha256,
            "selection_contract_sha256": self.selection_contract.sha256,
            "backend_parameter_contract_sha256": (
                self.config.backend_parameter_contract_sha256
            ),
            "future_open3d_source_sha256": source_sha,
            "future_pcl_source_sha256": source_sha,
            "future_open3d_target_sha256": self.target_freeze["target_map_sha256"],
            "future_pcl_target_sha256": self.target_freeze["target_map_sha256"],
            "byte_identical_for_both_backends": True,
        }
        scalar_core = {field: str(value) for field, value in row.items()}
        row["bundle_sha256"] = _hash_json(scalar_core)
        if set(row) != set(CANONICAL_INPUT_FIELDS):
            raise AssertionError("canonical input row schema drift")
        with materialized.path.open("rb") as stream:
            raw_payload = stream.read()
        witness = dict(
            self.dependencies.canonical_witness(
                raw_payload,
                object_key=item.key,
                pose_index=self.query_pose_index,
                producer_source_npy=source_bytes,
                producer_t_reference_npy=transform_bytes,
                bindings=CanonicalSourceWitnessBindings(
                    selection_index=selection_index,
                    snapshot_id=snapshot_id,
                    first_pass_receipt_sha256=first_receipt.as_dict()["receipt_sha256"],
                    second_pass_receipt_sha256=materialized.receipt.as_dict()[
                        "receipt_sha256"
                    ],
                    preprocessing_contract_sha256=(
                        self.config.preprocessing_contract_sha256
                    ),
                    gt_sha256=self.config.query_reference_pose_sha256,
                    extrinsic_sha256=self.config.extrinsic_sha256,
                    witness_implementation_sha256=(
                        self.config.canonical_witness_implementation_sha256
                    ),
                ),
            )
        )
        if set(witness) != set(CANONICAL_SOURCE_VERIFICATION_FIELDS):
            raise BoreasV2Stage2QueryRunnerError(
                "canonical witness exact schema differs"
            )
        return row, witness

    def run_second_pass(self, *, object_limit: int | None = None) -> QueryPhaseSummary:
        """Materialize only the frozen 100 selected objects and witness each bundle."""

        if object_limit is not None:
            if self.config.production_mode:
                raise BoreasV2Stage2QueryRunnerError(
                    "partial second-pass limits are forbidden in production"
                )
            _strict_int(object_limit, "object_limit", minimum=1)
        with self._phase_lock():
            self._assert_live()
            _, selected = self._load_frozen_selection()
            commits = self._selected_commits()
            pending = selected[len(commits) :]
            if object_limit is not None:
                pending = pending[:object_limit]
            downloaded = 0
            deleted = 0
            by_key = self.inventory.by_key
            for snapshot in pending:
                item = by_key.get(snapshot["object_key"])
                if item is None or item.selection_role != "QUERY":
                    raise BoreasV2Stage2QueryRunnerError(
                        "selected object is outside reconciled query allowlist"
                    )
                first_receipt = self._first_receipt_for_key(item.key)
                materialized, download_event, is_new = self._materialize(
                    QUERY_SECOND_PASS_STAGE, item
                )
                if is_new:
                    downloaded += item.size_bytes
                try:
                    self._fault("QUERY_SECOND_PASS_AFTER_DOWNLOADED", item)
                    row, witness = self._canonical_bundle(
                        snapshot=snapshot,
                        item=item,
                        materialized=materialized,
                        first_receipt=first_receipt,
                    )
                    self._fault("QUERY_SECOND_PASS_AFTER_WITNESS", item)
                    self.disk_gate.before_checkpoint(
                        max(16 * 1024, len(canonical_json_bytes(row)) * 2),
                        checkpoint_id=f"SELECTED_SOURCE_COMMIT:{item.key}",
                    )
                    self.journal.append(
                        event_kind=SELECTED_SOURCE_COMMIT,
                        execution_stage=QUERY_SECOND_PASS_STAGE,
                        object_key=item.key,
                        payload={
                            "canonical_input_row": row,
                            "canonical_source_verification_row": witness,
                            "download_event_sha256": download_event["event_sha256"],
                            "processing_result_sha256": row["source_points_sha256"],
                            "receipt_sha256": materialized.receipt.as_dict()[
                                "receipt_sha256"
                            ],
                            "selected_snapshot_row_sha256": snapshot[
                                "selected_snapshot_row_sha256"
                            ],
                        },
                    )
                    self._fault("QUERY_SECOND_PASS_AFTER_COMMIT", item)
                finally:
                    if materialized.path.exists():
                        deleted += self.downloader.release(materialized)
            commits = self._selected_commits()
            if len(commits) == 100:
                self.export_query_evidence()
            return QueryPhaseSummary(
                completed_object_count=len(commits),
                total_object_count=100,
                downloaded_bytes_this_run=downloaded,
                deleted_temporary_bytes_this_run=deleted,
                final_journal_event_sha256=self.journal.final_event_sha256,
            )

    def _receipt_projection(self) -> list[dict[str, Any]]:
        projection_path = self.evidence_root / "first_pass_selection_projection.csv"
        if not projection_path.is_file() or projection_path.is_symlink():
            raise BoreasV2Stage2QueryRunnerError(
                "first-pass selection projection is absent or unsafe"
            )
        with projection_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIRST_PASS_SELECTION_PROJECTION_FIELDS:
                raise BoreasV2Stage2QueryRunnerError(
                    "first-pass selection projection header differs"
                )
            projections = [dict(row) for row in reader]
        projection_by_event: dict[str, dict[str, str]] = {}
        selection_barrier = self.journal.barrier(SELECTION_FROZEN)
        for projection in projections:
            core: dict[str, Any] = {
                "execution_stage": projection["execution_stage"],
                "query_ordinal": int(projection["query_ordinal"]),
                "object_key": projection["object_key"],
                "first_pass_event_sha256": projection["first_pass_event_sha256"],
                "first_pass_result_sha256": projection["first_pass_result_sha256"],
                "candidate_scan_row_sha256": projection[
                    "candidate_scan_row_sha256"
                ],
                "selection_freeze_event_sha256": projection[
                    "selection_freeze_event_sha256"
                ],
            }
            if projection["projection_row_sha256"] != _hash_json(core):
                raise BoreasV2Stage2QueryRunnerError(
                    "first-pass selection projection row SHA differs"
                )
            if (
                projection["execution_stage"] != QUERY_FIRST_PASS_STAGE
                or projection["selection_freeze_event_sha256"]
                != selection_barrier["event_sha256"]
                or projection["first_pass_event_sha256"] in projection_by_event
            ):
                raise BoreasV2Stage2QueryRunnerError(
                    "first-pass selection projection identity differs"
                )
            projection_by_event[projection["first_pass_event_sha256"]] = projection
        output: list[dict[str, Any]] = []
        for commit in self.journal.committed:
            download = self.journal.download_for_event(
                commit["payload"]["download_event_sha256"]
            )
            if download is None:
                raise BoreasV2Stage2QueryRunnerError(
                    "committed receipt lacks download event"
                )
            receipt = _validated_receipt(download["payload"]["receipt"])
            result = commit["payload"]["processing_result_sha256"]
            if commit["event_kind"] == FIRST_PASS_COMMIT:
                projection = projection_by_event.get(commit["event_sha256"])
                if (
                    projection is None
                    or projection["object_key"] != commit["object_key"]
                    or projection["first_pass_result_sha256"] != result
                ):
                    raise BoreasV2Stage2QueryRunnerError(
                        "first-pass event lacks its frozen selection projection"
                    )
                result = projection["candidate_scan_row_sha256"]
            output.append(
                {
                    "execution_stage": commit["execution_stage"],
                    **receipt.as_dict(),
                    "processing_result_sha256": result,
                    "preprocessing_contract_sha256": (
                        self.config.preprocessing_contract_sha256
                    ),
                    "gt_sha256": self.config.query_reference_pose_sha256,
                    "extrinsic_sha256": self.config.extrinsic_sha256,
                    "checkpoint_status": "COMMITTED",
                }
            )
        if len(projection_by_event) != len(self._first_pass_commits()):
            raise BoreasV2Stage2QueryRunnerError(
                "first-pass selection projection count differs"
            )
        return output

    def export_query_evidence(self) -> dict[str, Any]:
        """Write exact verifier-facing query CSVs from durable committed state."""

        if self.config.production_mode:
            assert self.runtime_lease is not None
            self.runtime_lease.assert_live()
        self._assert_live()
        selected = self._selected_commits()
        if len(selected) != 100:
            raise BoreasV2Stage2QueryRunnerError(
                "canonical evidence requires 100 selected-source commits"
            )
        canonical = [row["payload"]["canonical_input_row"] for row in selected]
        witnesses = [
            row["payload"]["canonical_source_verification_row"] for row in selected
        ]
        receipts = self._receipt_projection()
        if len(receipts) != len(self.query_objects) + 100:
            raise BoreasV2Stage2QueryRunnerError(
                "query receipt closure does not equal 11,859 + 100"
            )
        projected = (
            len(csv_bytes(canonical, CANONICAL_INPUT_FIELDS))
            + len(
                csv_bytes(witnesses, CANONICAL_SOURCE_VERIFICATION_FIELDS)
            )
            + len(csv_bytes(receipts, RECEIPT_EXPORT_FIELDS))
            + 64 * 1024
        )
        self.disk_gate.before_checkpoint(
            projected, checkpoint_id="EXPORT_QUERY_EVIDENCE"
        )
        atomic_write_csv(
            self.evidence_root / "canonical_input_manifest.csv",
            canonical,
            CANONICAL_INPUT_FIELDS,
            overwrite=True,
        )
        atomic_write_csv(
            self.evidence_root / "canonical_source_verification.csv",
            witnesses,
            CANONICAL_SOURCE_VERIFICATION_FIELDS,
            overwrite=True,
        )
        # The map runner may have already exported 8,202 map receipts.  Merge
        # by exact (stage,key), preserving map first, query allowlist, selected.
        receipt_path = self.evidence_root / "download_receipts.csv"
        map_receipt_path = self.evidence_root / "map_download_receipts.csv"
        if not map_receipt_path.is_file() or map_receipt_path.is_symlink():
            raise BoreasV2Stage2QueryRunnerError(
                "immutable map receipt projection is absent or unsafe"
            )
        with map_receipt_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != RECEIPT_EXPORT_FIELDS:
                raise BoreasV2Stage2QueryRunnerError("map receipt CSV header differs")
            map_receipts = [dict(row) for row in reader]
        if any(row["execution_stage"] != "MAP_INGEST" for row in map_receipts):
            raise BoreasV2Stage2QueryRunnerError("map receipt stage differs")
        merged = map_receipts + receipts
        identities = [(row["execution_stage"], row["key"]) for row in merged]
        if len(identities) != len(set(identities)):
            raise BoreasV2Stage2QueryRunnerError("receipt export contains duplicates")
        atomic_write_csv(
            receipt_path, merged, RECEIPT_EXPORT_FIELDS, overwrite=True
        )
        # Final export deliberately performs a complete on-disk replay.  The
        # per-object hot path uses the in-memory indices above.
        rows = self.journal.read(full_rescan=True)
        download_rows = [row for row in rows if row["event_kind"] == "DOWNLOADED"]
        aborted = [row for row in rows if row["event_kind"] == "ABORTED"]
        query_committed_bytes = sum(int(row["remote_size_bytes"]) for row in receipts)
        query_download_bytes = sum(
            int(row["payload"]["receipt"]["remote_size_bytes"])
            for row in download_rows
        )
        raw_persistent = sum(
            path.stat().st_size
            for path in self.downloader.temporary_root.iterdir()
            if path.is_file() and path.name != ".stage2_temporary_root.json"
        )
        audit = {
            "schema": "zprm.boreas.v2.stage2.query_download_audit.v1",
            "successful_download_event_count": len(download_rows),
            "successful_payload_bytes": query_download_bytes,
            "committed_payload_event_count": len(receipts),
            "committed_payload_bytes": query_committed_bytes,
            "retry_download_event_count": len(aborted),
            "retry_download_payload_bytes": query_download_bytes
            - query_committed_bytes,
            "raw_payload_persistent_bytes": raw_persistent,
            "first_pass_committed_count": len(self.query_objects),
            "selected_source_committed_count": 100,
            "query_journal_path": self.journal.path.relative_to(
                self.runtime_root
            ).as_posix(),
            "query_journal_sha256": sha256_file(self.journal.path),
            "query_journal_final_event_sha256": rows[-1]["event_sha256"],
            "target_frozen_event_sha256": self.journal.barrier(TARGET_FROZEN)[
                "event_sha256"
            ],
            "first_pass_complete_event_sha256": self.journal.barrier(
                FIRST_PASS_COMPLETE
            )["event_sha256"],
            "R14_selection_frozen_event_sha256": self.journal.barrier(
                SELECTION_FROZEN
            )["event_sha256"],
            "registration_execution_count": 0,
        }
        atomic_write_json(
            self.evidence_root / "QUERY_LIDAR_DOWNLOAD_AUDIT.json",
            audit,
            overwrite=True,
        )
        map_audit_path = self.evidence_root / "MAP_LIDAR_DOWNLOAD_AUDIT.json"
        if not map_audit_path.is_file() or map_audit_path.is_symlink():
            raise BoreasV2Stage2QueryRunnerError(
                "immutable map download audit is absent or unsafe"
            )
        try:
            map_audit = json.loads(map_audit_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BoreasV2Stage2QueryRunnerError("map download audit is invalid") from exc
        required_map = {
            "aborted_download_event_count",
            "canonical_source_witness_count",
            "committed_payload_bytes",
            "committed_payload_event_count",
            "raw_payload_persistent_bytes",
            "receipt_checkpoint_final_chain_sha256",
            "receipt_checkpoint_journal_path",
            "receipt_checkpoint_journal_sha256",
            "replay_committed_event_count",
            "retry_download_event_count",
            "retry_download_payload_bytes",
            "schema",
            "stream_deleted_raw_bytes",
            "successful_download_event_count",
            "successful_payload_bytes",
            "unique_allowlist_object_count",
        }
        if (
            not isinstance(map_audit, dict)
            or set(map_audit) != required_map
            or map_audit["schema"] != "zprm.boreas.v2.stage2.lidar_download_audit.v2"
            or map_audit["committed_payload_event_count"] != len(map_receipts)
            or map_audit["committed_payload_bytes"]
            != sum(int(row["remote_size_bytes"]) for row in map_receipts)
            or map_audit["raw_payload_persistent_bytes"] != 0
        ):
            raise BoreasV2Stage2QueryRunnerError("map download audit semantics differ")
        map_journal = _safe_relative_file(
            self.runtime_root,
            str(map_audit["receipt_checkpoint_journal_path"]),
            label="map receipt journal",
        )
        if sha256_file(map_journal) != map_audit["receipt_checkpoint_journal_sha256"]:
            raise BoreasV2Stage2QueryRunnerError("map receipt journal SHA differs")
        for field in (
            "receipt_checkpoint_final_chain_sha256",
            "receipt_checkpoint_journal_sha256",
        ):
            _sha(map_audit[field], f"map audit {field}")
        map_temp_persistent = int(map_audit["raw_payload_persistent_bytes"])
        map_success = _strict_int(
            map_audit["successful_download_event_count"],
            "map successful downloads",
        )
        map_success_bytes = _strict_int(
            map_audit["successful_payload_bytes"], "map successful bytes"
        )
        aggregate_audit = {
            "schema": "zprm.boreas.v2.stage2.lidar_download_audit.v3",
            "successful_download_event_count": map_success + len(download_rows),
            "successful_payload_bytes": map_success_bytes + query_download_bytes,
            "committed_payload_event_count": len(merged),
            "committed_payload_bytes": sum(
                int(row["remote_size_bytes"]) for row in merged
            ),
            "unique_allowlist_object_count": len(self.inventory.objects),
            "raw_payload_persistent_bytes": map_temp_persistent + raw_persistent,
            "map_receipt_journal_path": map_audit[
                "receipt_checkpoint_journal_path"
            ],
            "map_receipt_journal_sha256": map_audit[
                "receipt_checkpoint_journal_sha256"
            ],
            "map_receipt_final_chain_sha256": map_audit[
                "receipt_checkpoint_final_chain_sha256"
            ],
            "map_retry_download_event_count": map_audit[
                "retry_download_event_count"
            ],
            "map_retry_download_payload_bytes": map_audit[
                "retry_download_payload_bytes"
            ],
            "query_journal_path": audit["query_journal_path"],
            "query_journal_sha256": audit["query_journal_sha256"],
            "query_journal_final_event_sha256": audit[
                "query_journal_final_event_sha256"
            ],
            "query_retry_download_event_count": len(aborted),
            "query_retry_download_payload_bytes": audit[
                "retry_download_payload_bytes"
            ],
            "target_frozen_event_sha256": audit["target_frozen_event_sha256"],
            "query_first_pass_complete_event_sha256": audit[
                "first_pass_complete_event_sha256"
            ],
            "R14_selection_frozen_event_sha256": audit[
                "R14_selection_frozen_event_sha256"
            ],
            "map_receipt_count": sum(
                row["execution_stage"] == "MAP_INGEST" for row in merged
            ),
            "query_first_pass_receipt_count": len(self.query_objects),
            "query_second_pass_receipt_count": 100,
            "registration_execution_count": 0,
        }
        atomic_write_json(
            self.evidence_root / "LIDAR_DOWNLOAD_AUDIT.json",
            aggregate_audit,
            overwrite=True,
        )
        return audit

    def build_selection_manifest(self) -> dict[str, Any]:
        """Assemble the verifier's final selection manifest after closure files exist."""

        if self.config.production_mode:
            assert self.runtime_lease is not None
            self.runtime_lease.assert_live()
        freeze, _ = self._load_frozen_selection()
        missing = [
            name for name in SELECTION_ARTIFACT_NAMES if not (self.evidence_root / name).is_file()
        ]
        if missing:
            raise BoreasV2Stage2QueryRunnerError(
                f"selection closure artifact is absent: {sorted(missing)[0]}"
            )
        core = {
            "schema_version": "boreas_v2_stage2_selection_manifest_v1",
            "authority_bindings": {
                "stage1_manifest_sha256": self.config.stage1_manifest_sha256,
                "storage_manifest_sha256": self.config.storage_manifest_sha256,
                "stage1_allowlist_sha256": self.config.stage1_allowlist_sha256,
                "pair_selection_sha256": self.config.pair_selection_sha256,
                "preprocessing_contract_sha256": (
                    self.config.preprocessing_contract_sha256
                ),
                "primary_pair": self.authorization.primary_pair,
            },
            "artifact_sha256": {
                name: sha256_file(self.evidence_root / name)
                for name in sorted(SELECTION_ARTIFACT_NAMES)
            },
            "blind_selection": freeze["blind_selection"],
            "R14_frozen": True,
            "registration_execution_count": 0,
        }
        result = {**core, "selection_manifest_sha256": _hash_json(core)}
        self.disk_gate.before_checkpoint(
            len(canonical_json_bytes(result)),
            checkpoint_id="FINAL_SELECTION_MANIFEST",
        )
        atomic_write_json(
            self.evidence_root / "boreas_v2_stage2_selection_manifest.json",
            result,
            overwrite=False,
        )
        return result


__all__ = [
    "BoreasV2Stage2QueryRunner",
    "BoreasV2Stage2QueryRunnerError",
    "CANONICAL_INPUT_FIELDS",
    "CANONICAL_SOURCE_VERIFICATION_FIELDS",
    "FIRST_PASS_COMPLETE",
    "FIRST_PASS_SELECTION_PROJECTION_FIELDS",
    "GEOMETRY_METRIC_VERIFICATION_FIELDS",
    "QueryPhaseSummary",
    "QueryRuntimeLease",
    "QueryRunnerConfig",
    "QueryRunnerDependencies",
    "SELECTION_FROZEN",
    "TARGET_FROZEN",
]
