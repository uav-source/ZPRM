"""Resumable production runner for Boreas v2 Stage-2 preparation.

This module composes the already-frozen authorization, remote identity,
preprocessing, disk-gate, replay, and external reducer primitives.  It owns
operational ordering and crash recovery only; it has no registration entry
point and it does not select scientific parameters.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .boreas_stage2_execution import MAP_STAGE
from .boreas_stage2_external_voxel import (
    CAPACITY_LAYOUT_SCHEMA,
    ExternalTargetMapResult,
    build_external_target_map,
    compile_external_voxel_reducer,
    default_reducer_source,
    describe_external_voxel_capacity,
    write_range_descriptor,
)
from .boreas_stage2_remote import (
    AuthorizedRemoteObject,
    DownloadReceipt,
    FrozenAllowlist,
    ReconciledRemoteInventory,
    RemoteObjectIdentity,
    StrictAllowlistDownloader,
    TemporaryDownloadedObject,
    list_remote_metadata,
    load_frozen_allowlist,
    reconcile_remote_metadata,
)
from .boreas_v2_stage2_authorization import (
    VerifiedStage2Authorization,
    verify_boreas_v2_stage2_download_authorization,
)
from .boreas_v2_stage2_preprocessing import (
    BoreasLidarPoseIndex,
    decode_authenticated_boreas_velodyne_file,
    preprocess_primary_boreas_scan,
    target_map_voxel_rule,
    transform_preprocessed_map_scan_to_enu_ref,
)
from .io import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_bytes,
    compact_sha256,
    sha256_file,
)
from .stage2_checkpoint import (
    TEMP_ROOT_SCHEMA,
    cleanup_partial_temporaries,
    initialize_temporary_root,
)
from .stage2_disk_gate import Stage2DiskGate
from .guard import NoRegistrationGuard
from .streaming_target_map import (
    ProductionMapReplayArray,
    ReplayMapObject,
    canonical_array_sha256,
    canonical_float64_npy_bytes,
    production_replay_plan_payload,
    recover_production_replay_allocation_before_gate,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ZERO_SHA256 = "0" * 64
REMOTE_INVENTORY_SCHEMA = "zprm.boreas.v2.stage2.remote_inventory.v1"
RECEIPT_JOURNAL_SCHEMA = "zprm.boreas.v2.stage2.receipt_checkpoint.v1"
REDUCER_RESOURCE_PLAN_SCHEMA = "zprm.boreas.v2.stage2.reducer_resource_plan.v1"
TARGET_COMMIT_INTENT_SCHEMA = "zprm.boreas.v2.stage2.target_commit_intent.v1"
TARGET_BUILD_INTENT_SCHEMA = "zprm.boreas.v2.stage2.target_build_intent.v1"
TARGET_STAGING_SCHEMA = "zprm.boreas.v2.stage2.target_staging.v1"
TRANSFER_INTENT_SCHEMA = "zprm.boreas.v2.stage2.map_transfer_intent.v1"
MAP_LINEAGE_SCHEMA = "boreas_v2_stage2_map_lineage_v1"
TARGET_FREEZE_SCHEMA = "boreas_v2_stage2_target_map_freeze_v1"
TARGET_REDUCER_VERIFICATION_SCHEMA = (
    "boreas_v2_stage2_target_reducer_verification_v1"
)
PRODUCTION_MAP_REPLAY_BYTES = 41_998_817_280
PRODUCTION_MAX_VOXELS = 90_000_000
MINIMUM_REDUCER_SAFETY_MARGIN_BYTES = 5 * 1024**3

# Exact enhanced closure agreed with the independent preparation verifier.
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


class BoreasV2Stage2RunnerError(RuntimeError):
    """A production orchestration, identity, or recovery invariant failed."""


def _require_sha256(value: Any, *, field: str) -> str:
    result = str(value).lower()
    if SHA256_RE.fullmatch(result) is None:
        raise BoreasV2Stage2RunnerError(f"{field} must be lowercase SHA-256")
    return result


def _strict_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BoreasV2Stage2RunnerError(f"{field} must be a positive integer")
    return value


def _json_object(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    try:
        payload = canonical_json_bytes(value)
        result = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BoreasV2Stage2RunnerError(f"{field} is not canonical JSON") from exc
    if not isinstance(result, dict):
        raise BoreasV2Stage2RunnerError(f"{field} must be a JSON object")
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_existing_file(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if (
        not value.is_absolute()
        or value.is_symlink()
        or not value.is_file()
        or value.resolve(strict=True) != value
    ):
        raise BoreasV2Stage2RunnerError(f"{label} is absent or unsafe")
    return value


def _prepare_runtime_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute() or root.is_symlink():
        raise BoreasV2Stage2RunnerError("runtime root must be an absolute non-symlink path")
    parent = root.parent.resolve(strict=True)
    if root.parent != parent:
        raise BoreasV2Stage2RunnerError("runtime-root parent is not canonical")
    if not root.exists():
        root.mkdir(mode=0o700)
        _fsync_directory(parent)
    if not root.is_dir() or root.is_symlink() or root.resolve(strict=True) != root:
        raise BoreasV2Stage2RunnerError("runtime root is unsafe")
    for name in ("checkpoints", "evidence", "map", "target_maps"):
        child = root / name
        child.mkdir(mode=0o700, exist_ok=True)
        if child.is_symlink() or child.resolve(strict=True) != child:
            raise BoreasV2Stage2RunnerError(f"runtime directory is unsafe: {name}")
    return root


def _prepare_disposable_root(path: Path, *, purpose: str) -> Path:
    if path.exists():
        cleanup_partial_temporaries(path)
    return initialize_temporary_root(path, purpose=purpose)


def _compact_line(value: Mapping[str, Any]) -> bytes:
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


def _event_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_compact_line(value)).hexdigest()


def map_source_witness_sha256(
    *,
    receipt_sha256: str,
    raw_payload_sha256: str,
    preprocessing_contract_sha256: str,
    gt_sha256: str,
    extrinsic_sha256: str,
    transformed_xyz: np.ndarray,
    source_point_count: int,
    metadata: Mapping[str, Any],
) -> str:
    """Bind raw identity, exact transformed bytes, and all producer counts."""

    raw_sha = _require_sha256(raw_payload_sha256, field="raw_payload_sha256")
    receipt_sha = _require_sha256(receipt_sha256, field="receipt_sha256")
    contract_sha = _require_sha256(
        preprocessing_contract_sha256, field="preprocessing_contract_sha256"
    )
    gt_digest = _require_sha256(gt_sha256, field="gt_sha256")
    extrinsic_digest = _require_sha256(extrinsic_sha256, field="extrinsic_sha256")
    points = np.ascontiguousarray(np.asarray(transformed_xyz, dtype="<f8"))
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
        raise BoreasV2Stage2RunnerError("map witness transformed XYZ is invalid")
    count = _strict_positive_int(source_point_count, field="source_point_count")
    if count != int(points.shape[0]):
        raise BoreasV2Stage2RunnerError(
            "source_point_count must equal transformed XYZ row count"
        )
    native_metadata = _json_object(metadata, field="map witness metadata")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "extrinsic_sha256": extrinsic_digest,
                "gt_sha256": gt_digest,
                "preprocessing_contract_sha256": contract_sha,
                "raw_payload_sha256": raw_sha,
                "receipt_sha256": receipt_sha,
                "source_point_count": count,
                "source_witness_metadata": native_metadata,
                "transformed_xyz_sha256": hashlib.sha256(
                    points.tobytes(order="C")
                ).hexdigest(),
                "witness_claim": (
                    "RAW_TO_TRANSFORMED_BYTE_IDENTITY_USING_SHARED_FROZEN_"
                    "PREPROCESSING_PRIMITIVES_NOT_AN_INDEPENDENT_ALGORITHM"
                ),
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class BoreasV2Stage2RunnerConfig:
    repository: Path
    data_root: Path
    runtime_root: Path
    temporary_root: Path
    monitored_disk_path: Path
    authorization_path: Path
    allowlist_path: Path
    expected_allowlist_sha256: str
    disk_budget_path: Path
    expected_disk_budget_sha256: str
    preprocessing_contract_path: Path
    aws_executable: Path
    reducer_resource_plan_path: Path | None = None
    bucket: str = "boreas"
    storage_mode: str = "RECOMMENDED_OPERATIONAL"
    production_mode: bool = True


@dataclass(frozen=True)
class MapScanMaterialization:
    """Injected frozen preprocessing output before raw-payload deletion."""

    transformed_xyz: np.ndarray
    canonical_source_witness_sha256: str
    source_point_count: int
    metadata: Mapping[str, Any]

    def validated(self) -> "MapScanMaterialization":
        points = np.asarray(self.transformed_xyz, dtype="<f8")
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or points.shape[0] <= 0
            or not np.all(np.isfinite(points))
        ):
            raise BoreasV2Stage2RunnerError(
                "map preprocessing must return finite nonempty N-by-3 points"
            )
        points = np.ascontiguousarray(points, dtype="<f8")
        witness = _require_sha256(
            self.canonical_source_witness_sha256,
            field="canonical_source_witness_sha256",
        )
        source_count = _strict_positive_int(
            self.source_point_count, field="source_point_count"
        )
        if source_count != int(points.shape[0]):
            raise BoreasV2Stage2RunnerError(
                "source_point_count must equal transformed XYZ row count"
            )
        metadata = _json_object(self.metadata, field="map preprocessing metadata")
        return MapScanMaterialization(points, witness, source_count, metadata)


AuthorizationVerifier = Callable[..., VerifiedStage2Authorization]
RemoteMetadataProvider = Callable[[FrozenAllowlist], Sequence[Any]]
MapPreprocessor = Callable[
    [TemporaryDownloadedObject, AuthorizedRemoteObject], MapScanMaterialization
]
CommandRunner = Callable[[Sequence[str]], Any]
ReducerCompiler = Callable[..., Mapping[str, Any]]
ReducerBuilder = Callable[..., ExternalTargetMapResult]


@dataclass(frozen=True)
class ProductionRemoteMetadataProvider:
    """Pinned metadata-only provider; no command callback is injectable."""

    aws_executable: Path
    bucket: str

    def __call__(self, allowlist: FrozenAllowlist) -> Sequence[RemoteObjectIdentity]:
        sequences = tuple(
            dict.fromkeys(item.sequence_id for item in allowlist.objects)
        )
        return list_remote_metadata(
            aws_executable=self.aws_executable,
            bucket=self.bucket,
            sequence_ids=sequences,
        )


@dataclass(frozen=True)
class ProductionMapPreprocessor:
    """Pinned map callback composed only from frozen public preprocessing APIs."""

    pose_index: BoreasLidarPoseIndex
    preprocessing_contract_sha256: str
    gt_sha256: str
    extrinsic_sha256: str

    @staticmethod
    def _scan_path(
        downloaded: TemporaryDownloadedObject,
        item: AuthorizedRemoteObject,
        pose_index: BoreasLidarPoseIndex,
    ) -> tuple[Any, Any, np.ndarray]:
        decoded = decode_authenticated_boreas_velodyne_file(
            downloaded.path,
            object_key=item.key,
            expected_size_bytes=downloaded.receipt.remote_size_bytes,
            expected_sha256=downloaded.receipt.local_temporary_sha256,
        )
        scan = preprocess_primary_boreas_scan(
            decoded, pose_index=pose_index, role="TARGET_MAP"
        )
        transformed = transform_preprocessed_map_scan_to_enu_ref(scan)
        return decoded, scan, transformed

    def __call__(
        self,
        downloaded: TemporaryDownloadedObject,
        item: AuthorizedRemoteObject,
    ) -> MapScanMaterialization:
        decoded, scan, transformed = self._scan_path(
            downloaded, item, self.pose_index
        )
        verified_decoded, verified_scan, verified_transformed = self._scan_path(
            downloaded, item, self.pose_index
        )
        if (
            not np.array_equal(decoded.fields, verified_decoded.fields)
            or not np.array_equal(scan.points_xyz, verified_scan.points_xyz)
            or not np.array_equal(scan.t_reference, verified_scan.t_reference)
            or not np.array_equal(transformed, verified_transformed)
            or (
                scan.raw_point_count,
                scan.nonfinite_excluded_count,
                scan.range_excluded_count,
                scan.post_filter_point_count,
                scan.source_voxel_reduced_count,
            )
            != (
                verified_scan.raw_point_count,
                verified_scan.nonfinite_excluded_count,
                verified_scan.range_excluded_count,
                verified_scan.post_filter_point_count,
                verified_scan.source_voxel_reduced_count,
            )
        ):
            raise BoreasV2Stage2RunnerError(
                "two frozen map preprocessing orchestrations differ"
            )
        transformed_sha = hashlib.sha256(
            transformed.tobytes(order="C")
        ).hexdigest()
        metadata = {
            "decoded_field_sha256": hashlib.sha256(
                decoded.fields.tobytes(order="C")
            ).hexdigest(),
            "dual_path_exact_match": True,
            "nonfinite_excluded_count": scan.nonfinite_excluded_count,
            "post_filter_point_count": scan.post_filter_point_count,
            "range_excluded_count": scan.range_excluded_count,
            "raw_point_count": scan.raw_point_count,
            "source_voxel_reduced_count": scan.source_voxel_reduced_count,
            "transformed_xyz_sha256": transformed_sha,
            "verification_path_transformed_xyz_sha256": hashlib.sha256(
                verified_transformed.tobytes(order="C")
            ).hexdigest(),
            "witness_claim": (
                "TWO_ORCHESTRATIONS_USING_SHARED_FROZEN_PREPROCESSING_"
                "PRIMITIVES_BYTE_IDENTICAL_NOT_INDEPENDENT_ALGORITHMS"
            ),
        }
        return MapScanMaterialization(
            transformed_xyz=transformed,
            canonical_source_witness_sha256=map_source_witness_sha256(
                receipt_sha256=downloaded.receipt.as_dict()["receipt_sha256"],
                raw_payload_sha256=downloaded.receipt.local_temporary_sha256,
                preprocessing_contract_sha256=self.preprocessing_contract_sha256,
                gt_sha256=self.gt_sha256,
                extrinsic_sha256=self.extrinsic_sha256,
                transformed_xyz=transformed,
                source_point_count=int(transformed.shape[0]),
                metadata=metadata,
            ),
            source_point_count=int(transformed.shape[0]),
            metadata=metadata,
        )


def _default_command_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) * 1024
    except (OSError, UnicodeError, ValueError) as exc:
        raise BoreasV2Stage2RunnerError("cannot read live available memory") from exc
    raise BoreasV2Stage2RunnerError("MemAvailable is absent from /proc/meminfo")


@dataclass(frozen=True)
class BoreasV2Stage2RunnerDependencies:
    metadata_provider: RemoteMetadataProvider
    map_preprocessor: MapPreprocessor
    no_registration_guard: NoRegistrationGuard
    authorization_verifier: AuthorizationVerifier = (
        verify_boreas_v2_stage2_download_authorization
    )
    command_runner: CommandRunner = _default_command_runner
    reducer_compiler: ReducerCompiler = compile_external_voxel_reducer
    reducer_builder: ReducerBuilder = build_external_target_map
    free_bytes_provider: Callable[[Path], int] | None = None
    available_memory_provider: Callable[[], int] = _available_memory_bytes
    now: Callable[[], datetime] = _utc_now
    fault_hook: Callable[[str, AuthorizedRemoteObject | None], None] | None = None
    synthetic_fixture_only: bool = False


@dataclass(frozen=True)
class ReducerResourcePlan:
    """Compiled-layout operational capacity authority; there is no default."""

    replay_plan_sha256: str | None
    reducer_source_sha256: str
    reducer_binary_sha256: str | None
    max_voxels: int
    estimated_peak_memory_bytes: int
    safety_margin_bytes: int
    minimum_live_available_memory_bytes: int
    capacity_probe_kind: str
    capacity_probe_evidence_sha256: str
    reducer_binary_projected_bytes: int
    production_approved: bool
    plan_payload_sha256: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, production_mode: bool
    ) -> "ReducerResourcePlan":
        raw = _json_object(value, field="reducer resource plan")
        supplied = _require_sha256(
            raw.pop("plan_payload_sha256", ""), field="plan_payload_sha256"
        )
        if raw.get("schema") != REDUCER_RESOURCE_PLAN_SCHEMA:
            raise BoreasV2Stage2RunnerError("reducer resource-plan schema differs")
        actual = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
        if supplied != actual:
            raise BoreasV2Stage2RunnerError("reducer resource-plan self-hash differs")
        estimated = _strict_positive_int(
            raw.get("estimated_peak_memory_bytes"),
            field="estimated_peak_memory_bytes",
        )
        margin = _strict_positive_int(
            raw.get("safety_margin_bytes"), field="safety_margin_bytes"
        )
        minimum = _strict_positive_int(
            raw.get("minimum_live_available_memory_bytes"),
            field="minimum_live_available_memory_bytes",
        )
        if minimum != estimated + margin:
            raise BoreasV2Stage2RunnerError(
                "reducer memory threshold must equal estimate plus safety margin"
            )
        approved = raw.get("production_approved") is True
        probe_kind = str(raw.get("capacity_probe_kind", ""))
        if production_mode and (
            not approved
            or probe_kind != "COMPILED_LAYOUT_EXACT_UPPER_BOUND_V1"
            or int(raw.get("max_voxels", 0)) != PRODUCTION_MAX_VOXELS
            or margin < MINIMUM_REDUCER_SAFETY_MARGIN_BYTES
            or raw.get("replay_plan_sha256") is not None
            or SHA256_RE.fullmatch(str(raw.get("reducer_binary_sha256", "")))
            is None
        ):
            raise BoreasV2Stage2RunnerError(
                "production reducer requires the pinned compiled-layout capacity plan"
            )
        return cls(
            replay_plan_sha256=(
                _require_sha256(
                    raw.get("replay_plan_sha256"), field="replay_plan_sha256"
                )
                if raw.get("replay_plan_sha256") is not None
                else None
            ),
            reducer_source_sha256=_require_sha256(
                raw.get("reducer_source_sha256"), field="reducer_source_sha256"
            ),
            reducer_binary_sha256=(
                _require_sha256(
                    raw.get("reducer_binary_sha256"), field="reducer_binary_sha256"
                )
                if raw.get("reducer_binary_sha256") is not None
                else None
            ),
            max_voxels=_strict_positive_int(raw.get("max_voxels"), field="max_voxels"),
            estimated_peak_memory_bytes=estimated,
            safety_margin_bytes=margin,
            minimum_live_available_memory_bytes=minimum,
            capacity_probe_kind=probe_kind,
            capacity_probe_evidence_sha256=_require_sha256(
                raw.get("capacity_probe_evidence_sha256"),
                field="capacity_probe_evidence_sha256",
            ),
            reducer_binary_projected_bytes=_strict_positive_int(
                raw.get("reducer_binary_projected_bytes"),
                field="reducer_binary_projected_bytes",
            ),
            production_approved=approved,
            plan_payload_sha256=supplied,
        )

    @classmethod
    def load(cls, path: str | Path, *, production_mode: bool) -> "ReducerResourcePlan":
        source = _canonical_existing_file(path, label="reducer resource plan")
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BoreasV2Stage2RunnerError("cannot parse reducer resource plan") from exc
        if not isinstance(value, Mapping):
            raise BoreasV2Stage2RunnerError("reducer resource plan is not an object")
        return cls.from_mapping(value, production_mode=production_mode)

    @staticmethod
    def signed_payload(**fields: Any) -> dict[str, Any]:
        """Create a canonical plan payload for an external capacity probe."""

        value = {"schema": REDUCER_RESOURCE_PLAN_SCHEMA, **fields}
        value["plan_payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(value)
        ).hexdigest()
        return value


class _ReceiptJournal:
    """Append-only download/preprocess/replay state with strict crash recovery."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise BoreasV2Stage2RunnerError("receipt journal path is unsafe")
        self._rows = self._read_from_disk()
        self._file_size = path.stat().st_size if path.exists() else 0
        self._downloads_by_sha: dict[str, dict[str, Any]] = {}
        self._preprocessed_by_download: dict[str, dict[str, Any]] = {}
        self._unresolved_by_download: dict[str, dict[str, Any]] = {}
        self._committed_pairs: list[dict[str, Any]] = []
        for row in self._rows:
            self._index_row(row)

    def _index_row(self, row: dict[str, Any]) -> None:
        kind = row["event_kind"]
        if kind == "DOWNLOADED":
            event_sha = row["event_sha256"]
            self._downloads_by_sha[event_sha] = row
            self._unresolved_by_download[event_sha] = row
        elif kind == "PREPROCESSED":
            self._preprocessed_by_download[row["download_event_sha256"]] = row
        elif kind == "REPLAY_COMMITTED":
            download_sha = row["download_event_sha256"]
            self._committed_pairs.append(
                {
                    "download": self._downloads_by_sha[download_sha],
                    "preprocessed": self._preprocessed_by_download[download_sha],
                    "commit": row,
                }
            )
            self._unresolved_by_download.pop(download_sha)
        elif kind == "ABORTED":
            self._unresolved_by_download.pop(row["download_event_sha256"])
        else:  # The disk parser rejects this before indexing.
            raise BoreasV2Stage2RunnerError("unknown receipt journal event kind")

    @staticmethod
    def _validate_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
        receipt = _json_object(value, field="download receipt")
        required = {
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
        if set(receipt) != required:
            raise BoreasV2Stage2RunnerError("download receipt field set differs")
        supplied = _require_sha256(
            receipt["receipt_sha256"], field="receipt_sha256"
        )
        core = {key: child for key, child in receipt.items() if key != "receipt_sha256"}
        if hashlib.sha256(canonical_json_bytes(core)).hexdigest() != supplied:
            raise BoreasV2Stage2RunnerError("download receipt self-hash differs")
        _require_sha256(
            receipt["local_temporary_sha256"], field="local_temporary_sha256"
        )
        return receipt

    def _read_from_disk(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = self.path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise BoreasV2Stage2RunnerError("receipt journal is truncated")
        rows: list[dict[str, Any]] = []
        previous = ZERO_SHA256
        downloads: dict[str, dict[str, Any]] = {}
        preprocessed: dict[str, dict[str, Any]] = {}
        resolved: set[str] = set()
        completed: set[tuple[str, str]] = set()
        for index, line in enumerate(payload.splitlines(keepends=True), start=1):
            try:
                row = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise BoreasV2Stage2RunnerError("receipt journal contains invalid JSON") from exc
            if not isinstance(row, dict) or _compact_line(row) != line:
                raise BoreasV2Stage2RunnerError("receipt journal is not canonical JSONL")
            if (
                row.get("schema") != RECEIPT_JOURNAL_SCHEMA
                or row.get("sequence_number") != index
                or row.get("previous_event_sha256") != previous
            ):
                raise BoreasV2Stage2RunnerError("receipt journal chain metadata differs")
            supplied = _require_sha256(row.get("event_sha256"), field="event_sha256")
            unsigned = {key: child for key, child in row.items() if key != "event_sha256"}
            actual = _event_sha(unsigned)
            if actual != supplied:
                raise BoreasV2Stage2RunnerError("receipt journal event hash differs")
            kind = row.get("event_kind")
            if kind == "DOWNLOADED":
                receipt = self._validate_receipt(row.get("receipt", {}))
                if row.get("execution_stage") != MAP_STAGE:
                    raise BoreasV2Stage2RunnerError("unsupported download execution stage")
                for name in (
                    "preprocessing_contract_sha256",
                    "gt_sha256",
                    "extrinsic_sha256",
                ):
                    _require_sha256(row.get(name), field=name)
                downloads[supplied] = row
            elif kind == "PREPROCESSED":
                download_sha = _require_sha256(
                    row.get("download_event_sha256"), field="download_event_sha256"
                )
                download = downloads.get(download_sha)
                if download is None or download_sha in resolved or download_sha in preprocessed:
                    raise BoreasV2Stage2RunnerError(
                        "preprocessing event lacks one unresolved download"
                    )
                receipt = download["receipt"]
                if (
                    row.get("execution_stage") != download["execution_stage"]
                    or row.get("key") != receipt["key"]
                    or row.get("receipt_sha256") != receipt["receipt_sha256"]
                ):
                    raise BoreasV2Stage2RunnerError("preprocessing receipt identity differs")
                _require_sha256(
                    row.get("canonical_source_witness_sha256"),
                    field="canonical_source_witness_sha256",
                )
                _strict_positive_int(
                    row.get("source_point_count"), field="source_point_count"
                )
                _json_object(
                    row.get("source_witness_metadata", {}),
                    field="source_witness_metadata",
                )
                preprocessed[download_sha] = row
            elif kind in {"REPLAY_COMMITTED", "ABORTED"}:
                download_sha = _require_sha256(
                    row.get("download_event_sha256"), field="download_event_sha256"
                )
                download = downloads.get(download_sha)
                if download is None or download_sha in resolved:
                    raise BoreasV2Stage2RunnerError(
                        "receipt resolution lacks one unresolved download"
                    )
                receipt = download["receipt"]
                if (
                    row.get("execution_stage") != download["execution_stage"]
                    or row.get("key") != receipt["key"]
                    or row.get("receipt_sha256") != receipt["receipt_sha256"]
                ):
                    raise BoreasV2Stage2RunnerError("receipt resolution identity differs")
                if kind == "REPLAY_COMMITTED":
                    preprocessing_sha = _require_sha256(
                        row.get("preprocessed_event_sha256"),
                        field="preprocessed_event_sha256",
                    )
                    preprocessing = preprocessed.get(download_sha)
                    if preprocessing is None or preprocessing["event_sha256"] != preprocessing_sha:
                        raise BoreasV2Stage2RunnerError(
                            "replay commit lacks its preprocessing witness"
                        )
                    if row.get("checkpoint_status") != "COMMITTED":
                        raise BoreasV2Stage2RunnerError("receipt checkpoint status differs")
                    _require_sha256(
                        row.get("processing_result_sha256"),
                        field="processing_result_sha256",
                    )
                    identity = (str(row["execution_stage"]), str(row["key"]))
                    if identity in completed:
                        raise BoreasV2Stage2RunnerError("duplicate committed receipt")
                    completed.add(identity)
                resolved.add(download_sha)
            else:
                raise BoreasV2Stage2RunnerError("unknown receipt journal event kind")
            rows.append(row)
            previous = actual
        return rows

    def _append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise BoreasV2Stage2RunnerError("receipt journal is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            metadata = os.fstat(descriptor)
            if metadata.st_size != self._file_size:
                raise BoreasV2Stage2RunnerError("receipt journal changed concurrently")
            unsigned = {
                **_json_object(event, field="receipt journal event"),
                "previous_event_sha256": (
                    self._rows[-1]["event_sha256"] if self._rows else ZERO_SHA256
                ),
                "schema": RECEIPT_JOURNAL_SCHEMA,
                "sequence_number": len(self._rows) + 1,
            }
            unsigned.pop("event_sha256", None)
            row = {**unsigned, "event_sha256": _event_sha(unsigned)}
            line = _compact_line(row)
            view = memoryview(line)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("zero-byte receipt journal append")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        _fsync_directory(self.path.parent)
        self._rows.append(row)
        self._index_row(row)
        self._file_size += len(line)
        return row

    def record_download(
        self,
        *,
        receipt: DownloadReceipt,
        preprocessing_contract_sha256: str,
        gt_sha256: str,
        extrinsic_sha256: str,
    ) -> dict[str, Any]:
        if self._unresolved_by_download:
            raise BoreasV2Stage2RunnerError("an unresolved download event already exists")
        return self._append(
            {
                "event_kind": "DOWNLOADED",
                "execution_stage": MAP_STAGE,
                "extrinsic_sha256": extrinsic_sha256,
                "gt_sha256": gt_sha256,
                "preprocessing_contract_sha256": preprocessing_contract_sha256,
                "receipt": receipt.as_dict(),
            }
        )

    def record_preprocessed(
        self,
        download: Mapping[str, Any],
        *,
        source: MapScanMaterialization,
    ) -> dict[str, Any]:
        receipt = download["receipt"]
        return self._append(
            {
                "canonical_source_witness_sha256": source.canonical_source_witness_sha256,
                "download_event_sha256": download["event_sha256"],
                "event_kind": "PREPROCESSED",
                "execution_stage": MAP_STAGE,
                "key": receipt["key"],
                "receipt_sha256": receipt["receipt_sha256"],
                "source_point_count": source.source_point_count,
                "source_witness_metadata": source.metadata,
            }
        )

    def commit(
        self,
        download: Mapping[str, Any],
        preprocessed: Mapping[str, Any],
        *,
        processing_result_sha256: str,
        replay_ordinal: int,
    ) -> dict[str, Any]:
        receipt = download["receipt"]
        return self._append(
            {
                "checkpoint_status": "COMMITTED",
                "download_event_sha256": download["event_sha256"],
                "event_kind": "REPLAY_COMMITTED",
                "execution_stage": download["execution_stage"],
                "key": receipt["key"],
                "preprocessed_event_sha256": preprocessed["event_sha256"],
                "processing_result_sha256": processing_result_sha256,
                "receipt_sha256": receipt["receipt_sha256"],
                "replay_ordinal": replay_ordinal,
            }
        )

    def abort(self, download: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
        receipt = download["receipt"]
        return self._append(
            {
                "download_event_sha256": download["event_sha256"],
                "event_kind": "ABORTED",
                "execution_stage": download["execution_stage"],
                "key": receipt["key"],
                "reason": reason,
                "receipt_sha256": receipt["receipt_sha256"],
            }
        )

    @property
    def unresolved_downloads(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._unresolved_by_download.values())

    @property
    def download_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self._rows if row["event_kind"] == "DOWNLOADED")

    @property
    def committed(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._committed_pairs)

    def preprocessing_for(self, download_event_sha256: str) -> dict[str, Any] | None:
        return self._preprocessed_by_download.get(download_event_sha256)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._rows)


@dataclass(frozen=True)
class MapIngestSummary:
    completed_object_count: int
    total_object_count: int
    downloaded_bytes_this_run: int
    deleted_temporary_bytes_this_run: int
    replay_plan_sha256: str
    replay_complete: bool
    receipt_journal_sha256: str | None


class BoreasV2Stage2Runner:
    """High-level, fail-closed Stage-2 map preparation runner."""

    def __init__(
        self,
        config: BoreasV2Stage2RunnerConfig,
        dependencies: BoreasV2Stage2RunnerDependencies,
    ) -> None:
        if not isinstance(config, BoreasV2Stage2RunnerConfig):
            raise BoreasV2Stage2RunnerError("runner config has an invalid type")
        if not isinstance(dependencies, BoreasV2Stage2RunnerDependencies):
            raise BoreasV2Stage2RunnerError("runner dependencies have an invalid type")
        if not callable(dependencies.metadata_provider) or not callable(
            dependencies.map_preprocessor
        ):
            raise BoreasV2Stage2RunnerError("metadata and preprocessing callbacks are mandatory")
        if (
            not isinstance(dependencies.no_registration_guard, NoRegistrationGuard)
            or not dependencies.no_registration_guard.active
        ):
            raise BoreasV2Stage2RunnerError(
                "runner requires one already-active NoRegistrationGuard"
            )
        if config.production_mode:
            self._validate_production_dependencies(config, dependencies)
        elif dependencies.synthetic_fixture_only is not True:
            raise BoreasV2Stage2RunnerError(
                "non-production runner dependencies must be explicit synthetic fixtures"
            )
        self.config = config
        self.dependencies = dependencies
        self.runtime_root: Path | None = None
        self.authorization: VerifiedStage2Authorization | None = None
        self._lock_descriptor: int | None = None
        self.allowlist: FrozenAllowlist | None = None
        self.inventory: ReconciledRemoteInventory | None = None
        self.preprocessing_contract: dict[str, Any] | None = None
        self.preprocessing_contract_sha256: str | None = None
        self.gt_sha256: str | None = None
        self.extrinsic_sha256: str | None = None
        self.disk_gate: Stage2DiskGate | None = None
        self.downloader: StrictAllowlistDownloader | None = None
        self.replay: ProductionMapReplayArray | None = None
        self.receipts: _ReceiptJournal | None = None
        self.reducer_resource_plan: ReducerResourcePlan | None = None
        self.reducer_binary: Path | None = None
        self.reducer_binary_sha256: str | None = None
        self._pre_gate_inventory: ReconciledRemoteInventory | None = None
        self._pre_gate_replay: ProductionMapReplayArray | None = None

    @staticmethod
    def _validate_production_dependencies(
        config: BoreasV2Stage2RunnerConfig,
        dependencies: BoreasV2Stage2RunnerDependencies,
    ) -> None:
        """Reject every production callback or provider not pinned by this module."""

        if type(dependencies.metadata_provider) is not ProductionRemoteMetadataProvider:
            raise BoreasV2Stage2RunnerError(
                "production metadata provider must be ProductionRemoteMetadataProvider"
            )
        metadata = dependencies.metadata_provider
        assert isinstance(metadata, ProductionRemoteMetadataProvider)
        if (
            metadata.aws_executable != config.aws_executable
            or metadata.bucket != config.bucket
        ):
            raise BoreasV2Stage2RunnerError(
                "production metadata provider configuration differs from runner"
            )
        if type(dependencies.map_preprocessor) is not ProductionMapPreprocessor:
            raise BoreasV2Stage2RunnerError(
                "production map preprocessor must be ProductionMapPreprocessor"
            )
        exact_identities = (
            (
                dependencies.authorization_verifier,
                verify_boreas_v2_stage2_download_authorization,
                "authorization verifier",
            ),
            (dependencies.command_runner, _default_command_runner, "command runner"),
            (
                dependencies.reducer_compiler,
                compile_external_voxel_reducer,
                "reducer compiler",
            ),
            (
                dependencies.reducer_builder,
                build_external_target_map,
                "reducer builder",
            ),
            (
                dependencies.available_memory_provider,
                _available_memory_bytes,
                "memory provider",
            ),
            (dependencies.now, _utc_now, "clock"),
        )
        for actual, expected, label in exact_identities:
            if actual is not expected:
                raise BoreasV2Stage2RunnerError(
                    f"production {label} is not the pinned implementation"
                )
        if dependencies.free_bytes_provider is not None:
            raise BoreasV2Stage2RunnerError(
                "production free-space provider cannot be injected"
            )
        if dependencies.fault_hook is not None:
            raise BoreasV2Stage2RunnerError(
                "production fault injection is forbidden"
            )
        if dependencies.synthetic_fixture_only:
            raise BoreasV2Stage2RunnerError(
                "production dependencies cannot carry synthetic fixture authority"
            )

    def _fault(self, label: str, item: AuthorizedRemoteObject | None = None) -> None:
        if self.dependencies.fault_hook is not None:
            self.dependencies.fault_hook(label, item)

    def _acquire_runtime_lock(self) -> None:
        assert self.runtime_root is not None
        path = self.runtime_root / ".boreas_v2_stage2.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise BoreasV2Stage2RunnerError("runtime lock is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BoreasV2Stage2RunnerError(
                    "another Boreas Stage-2 runner owns the runtime root"
                ) from exc
        except Exception:
            os.close(descriptor)
            raise
        self._lock_descriptor = descriptor

    def close(self) -> None:
        descriptor = self._lock_descriptor
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
            self._lock_descriptor = None

    def __enter__(self) -> "BoreasV2Stage2Runner":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _authenticate_authority(self) -> None:
        if (
            not self.config.production_mode
            and (
                not self.dependencies.synthetic_fixture_only
                or self.dependencies.authorization_verifier
                is verify_boreas_v2_stage2_download_authorization
            )
        ):
            raise BoreasV2Stage2RunnerError(
                "VerifiedStage2Authorization may only be consumed in production_mode=True"
            )
        repository = Path(self.config.repository).resolve(strict=True)
        if repository != self.config.repository or repository.is_symlink():
            raise BoreasV2Stage2RunnerError("repository path must be canonical")
        capability = self.dependencies.authorization_verifier(
            repository=repository,
            data_root=self.config.data_root,
            runtime_root=self.config.runtime_root,
            temporary_root=self.config.temporary_root,
            monitored_disk_path=self.config.monitored_disk_path,
            authorization_path=self.config.authorization_path,
            no_registration_guard=self.dependencies.no_registration_guard,
        )
        if not isinstance(capability, VerifiedStage2Authorization):
            raise BoreasV2Stage2RunnerError(
                "authorization verifier must mint VerifiedStage2Authorization"
            )
        if self.config.production_mode and not capability.formally_verified:
            raise BoreasV2Stage2RunnerError(
                "production runner requires a formally verified authorization capability"
            )
        if not self.config.production_mode and capability.formally_verified:
            raise BoreasV2Stage2RunnerError(
                "formal authorization capability cannot be downgraded to non-production"
            )
        if capability.no_registration_guard is not self.dependencies.no_registration_guard:
            raise BoreasV2Stage2RunnerError("authorization guard identity differs")
        capability.assert_live()
        authorization = capability.document
        for field, expected in (
            ("STAGE2_DOWNLOAD_AUTHORIZED", True),
            ("REAL_REGISTRATION_AUTHORIZED", False),
            ("PUBLIC_DATA_V2_RUN_AUTHORIZED", False),
            ("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED", False),
            ("registration_execution_count", 0),
            ("actual_trials", 0),
        ):
            if authorization.get(field) != expected:
                raise BoreasV2Stage2RunnerError(f"authorization {field} differs")
        if capability.allowlist_sha256 != self.config.expected_allowlist_sha256:
            raise BoreasV2Stage2RunnerError("authorization allowlist SHA differs")
        if capability.bucket != self.config.bucket:
            raise BoreasV2Stage2RunnerError("authorization bucket differs")
        self.authorization = capability

    def _authenticate_preprocessing_contract(self) -> None:
        assert self.authorization is not None
        source = _canonical_existing_file(
            self.config.preprocessing_contract_path,
            label="preprocessing contract",
        )
        actual_sha = sha256_file(source)
        authorization = self.authorization.document
        if actual_sha != authorization.get("preprocessing_contract_sha256"):
            raise BoreasV2Stage2RunnerError(
                "preprocessing contract file differs from authorization"
            )
        try:
            contract = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BoreasV2Stage2RunnerError("cannot parse preprocessing contract") from exc
        if not isinstance(contract, dict) or contract.get("contract_status") != "FROZEN":
            raise BoreasV2Stage2RunnerError("preprocessing contract is not frozen")
        supplied = _require_sha256(
            contract.get("contract_payload_sha256"),
            field="preprocessing contract payload SHA",
        )
        unsigned = {key: child for key, child in contract.items() if key != "contract_payload_sha256"}
        if compact_sha256(unsigned) != supplied:
            raise BoreasV2Stage2RunnerError("preprocessing contract self-hash differs")
        if authorization.get("preprocessing_contract_payload_sha256") != supplied:
            raise BoreasV2Stage2RunnerError(
                "preprocessing payload SHA differs from authorization"
            )
        pair = contract.get("primary_pair")
        if not isinstance(pair, Mapping):
            raise BoreasV2Stage2RunnerError("preprocessing contract primary pair is absent")
        self.preprocessing_contract = contract
        self.preprocessing_contract_sha256 = actual_sha
        self.gt_sha256 = _require_sha256(
            pair.get("map_lidar_pose_sha256"), field="map lidar-pose SHA"
        )
        self.extrinsic_sha256 = _require_sha256(
            pair.get("static_t_applanix_lidar_sha256"), field="extrinsic SHA"
        )
        if self.config.production_mode:
            preprocessor = self.dependencies.map_preprocessor
            assert isinstance(preprocessor, ProductionMapPreprocessor)
            if (
                preprocessor.preprocessing_contract_sha256 != actual_sha
                or preprocessor.gt_sha256 != self.gt_sha256
                or preprocessor.pose_index.source_sha256 != self.gt_sha256
                or preprocessor.extrinsic_sha256 != self.extrinsic_sha256
                or preprocessor.pose_index.sequence_id != pair.get("map_sequence_id")
            ):
                raise BoreasV2Stage2RunnerError(
                    "production map preprocessor differs from authenticated contract"
                )

    def _load_production_capacity_plan(self) -> None:
        if not self.config.production_mode:
            return
        path = self.config.reducer_resource_plan_path
        if path is None:
            raise BoreasV2Stage2RunnerError(
                "REDUCER_CAPACITY_NOT_READY: reducer resource plan is required"
            )
        expected_path = (
            self.config.runtime_root / "checkpoints/reducer_resource_plan.json"
        )
        if path != expected_path:
            raise BoreasV2Stage2RunnerError(
                "REDUCER_CAPACITY_NOT_READY: production reducer resource plan "
                "must use runtime/checkpoints/reducer_resource_plan.json"
            )
        self.reducer_resource_plan = ReducerResourcePlan.load(
            path, production_mode=True
        )

    def _freeze_or_verify_inventory(
        self, allowlist: FrozenAllowlist
    ) -> ReconciledRemoteInventory:
        assert self.runtime_root is not None
        assert self.authorization is not None
        self.authorization.assert_operation_live()
        current = reconcile_remote_metadata(
            allowlist, self.dependencies.metadata_provider(allowlist)
        )
        path = self.runtime_root / "checkpoints" / "remote_inventory.json"
        unsigned = {
            "allowlist_sha256": current.allowlist_sha256,
            "inventory_sha256": current.inventory_sha256,
            "objects": [item.remote.as_dict() for item in current.objects],
            "schema": REMOTE_INVENTORY_SCHEMA,
        }
        frozen = {
            **unsigned,
            "inventory_file_payload_sha256": hashlib.sha256(
                canonical_json_bytes(unsigned)
            ).hexdigest(),
        }
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise BoreasV2Stage2RunnerError("frozen remote inventory is unsafe")
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise BoreasV2Stage2RunnerError("frozen remote inventory is invalid") from exc
            if prior != frozen or canonical_json_bytes(prior) != path.read_bytes():
                raise BoreasV2Stage2RunnerError(
                    "current remote inventory differs from frozen resume identity"
                )
        else:
            assert self.disk_gate is not None
            self.disk_gate.before_checkpoint(
                len(canonical_json_bytes(frozen)) + 4096,
                checkpoint_id="FREEZE_REMOTE_INVENTORY",
            )
            atomic_write_json(path, frozen)
        return current

    def _load_frozen_inventory_for_pre_gate(
        self, allowlist: FrozenAllowlist
    ) -> ReconciledRemoteInventory:
        """Authenticate the prior remote identity without making a network call."""

        assert self.runtime_root is not None
        path = self.runtime_root / "checkpoints/remote_inventory.json"
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or os.lstat(path).st_nlink != 1
        ):
            raise BoreasV2Stage2RunnerError(
                "replay resume state lacks its authenticated remote inventory"
            )
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BoreasV2Stage2RunnerError(
                "frozen remote inventory is invalid"
            ) from exc
        if (
            not isinstance(value, dict)
            or raw != canonical_json_bytes(value)
            or set(value)
            != {
                "allowlist_sha256",
                "inventory_file_payload_sha256",
                "inventory_sha256",
                "objects",
                "schema",
            }
            or value.get("schema") != REMOTE_INVENTORY_SCHEMA
            or value.get("allowlist_sha256") != allowlist.sha256
            or not isinstance(value.get("objects"), list)
        ):
            raise BoreasV2Stage2RunnerError(
                "frozen remote inventory exact schema differs"
            )
        unsigned = dict(value)
        claim = unsigned.pop("inventory_file_payload_sha256", None)
        if claim != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
            raise BoreasV2Stage2RunnerError(
                "frozen remote inventory self-hash differs"
            )
        try:
            inventory = reconcile_remote_metadata(allowlist, value["objects"])
        except Exception as exc:
            raise BoreasV2Stage2RunnerError(
                "frozen remote inventory no longer reconciles to the allowlist"
            ) from exc
        if inventory.inventory_sha256 != value.get("inventory_sha256"):
            raise BoreasV2Stage2RunnerError(
                "frozen remote inventory identity differs"
            )
        return inventory

    @staticmethod
    def _recompute_capacity_layout(
        layout: Mapping[str, Any], *, max_voxels: int
    ) -> dict[str, Any]:
        fields = {
            "constructor_capacity",
            "final_table_bytes",
            "fixed_overhead_bytes",
            "input_buffer_bytes",
            "key_bytes",
            "max_voxels",
            "maximum_load_denominator",
            "maximum_load_numerator",
            "output_buffer_bytes",
            "peak_growth_table_bytes",
            "schema",
            "size_t_bytes",
            "slot_bytes",
            "sorted_index_upper_bytes",
            "table_capacity",
            "target_npy_upper_bytes",
            "total_peak_upper_bound_bytes",
        }
        if set(layout) != fields or layout.get("schema") != CAPACITY_LAYOUT_SCHEMA:
            raise BoreasV2Stage2RunnerError("reducer capacity layout field set differs")
        integers: dict[str, int] = {}
        for field in fields - {"schema"}:
            value = layout.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise BoreasV2Stage2RunnerError(
                    f"reducer capacity layout {field} is invalid"
                )
            integers[field] = value
        if (
            integers["max_voxels"] != max_voxels
            or integers["maximum_load_numerator"] != 7
            or integers["maximum_load_denominator"] != 10
            or integers["key_bytes"] != 12
            or integers["slot_bytes"] != 56
            or integers["size_t_bytes"] != 8
        ):
            raise BoreasV2Stage2RunnerError(
                "compiled reducer ABI/load layout differs from pinned production layout"
            )
        usable = lambda capacity: capacity * 7 // 10
        constructor_capacity = 1024
        while usable(constructor_capacity) < max_voxels and constructor_capacity < 2**20:
            constructor_capacity *= 2
        table_capacity = 1024
        while usable(table_capacity) < max_voxels:
            table_capacity *= 2
        final_table = table_capacity * integers["slot_bytes"]
        largest_old = table_capacity // 2 if table_capacity > constructor_capacity else 0
        growth_table = (table_capacity + largest_old) * integers["slot_bytes"]
        sorted_indices = max_voxels * integers["size_t_bytes"]
        input_buffer = (2**18) * 3 * 8
        output_buffer = 4096 * 3 * 8
        target_npy = max_voxels * 3 * 8 + 4096
        fixed = 64 * 1024**2
        total = max(
            growth_table + input_buffer + fixed,
            final_table
            + sorted_indices
            + input_buffer
            + output_buffer
            + target_npy
            + fixed,
        )
        expected = {
            "constructor_capacity": constructor_capacity,
            "final_table_bytes": final_table,
            "fixed_overhead_bytes": fixed,
            "input_buffer_bytes": input_buffer,
            "key_bytes": 12,
            "max_voxels": max_voxels,
            "maximum_load_denominator": 10,
            "maximum_load_numerator": 7,
            "output_buffer_bytes": output_buffer,
            "peak_growth_table_bytes": growth_table,
            "schema": CAPACITY_LAYOUT_SCHEMA,
            "size_t_bytes": 8,
            "slot_bytes": 56,
            "sorted_index_upper_bytes": sorted_indices,
            "table_capacity": table_capacity,
            "target_npy_upper_bytes": target_npy,
            "total_peak_upper_bound_bytes": total,
        }
        if dict(layout) != expected:
            raise BoreasV2Stage2RunnerError(
                "compiled reducer capacity output differs from independent formula"
            )
        return expected

    def _verify_production_reducer_capacity_before_remote(self) -> None:
        if not self.config.production_mode:
            return
        assert self.reducer_resource_plan is not None
        assert self.authorization is not None
        assert self.runtime_root is not None
        plan = self.reducer_resource_plan
        self.authorization.assert_operation_live()
        binary, binary_sha = self._prepare_reducer_binary(plan)
        self.authorization.assert_operation_live()
        raw_layout = describe_external_voxel_capacity(
            reducer_binary=binary,
            reducer_binary_sha256=binary_sha,
            max_voxels=plan.max_voxels,
        )
        layout = self._recompute_capacity_layout(
            raw_layout, max_voxels=plan.max_voxels
        )
        layout_sha = hashlib.sha256(canonical_json_bytes(layout)).hexdigest()
        if layout_sha != plan.capacity_probe_evidence_sha256:
            raise BoreasV2Stage2RunnerError(
                "compiled capacity layout differs from frozen resource plan"
            )
        if layout["total_peak_upper_bound_bytes"] != plan.estimated_peak_memory_bytes:
            raise BoreasV2Stage2RunnerError(
                "resource-plan memory estimate differs from independent formula"
            )
        available = self.dependencies.available_memory_provider()
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or available < plan.minimum_live_available_memory_bytes
        ):
            raise BoreasV2Stage2RunnerError(
                "BLOCKED_INSUFFICIENT_RAM: live memory is below compiled-layout plan"
            )
        source = _canonical_existing_file(
            default_reducer_source(), label="external reducer source"
        )
        unsigned = {
            "binary_path": binary.relative_to(self.runtime_root).as_posix(),
            "binary_sha256": binary_sha,
            "capacity_layout": layout,
            "capacity_layout_sha256": layout_sha,
            "minimum_live_available_memory_bytes": (
                plan.minimum_live_available_memory_bytes
            ),
            "resource_plan_sha256": plan.plan_payload_sha256,
            "schema": "zprm.boreas.v2.stage2.reducer_capacity_layout_verification.v1",
            "source_path": source.relative_to(self.config.repository).as_posix(),
            "source_sha256": sha256_file(source),
            "verification_status": "PASS_COMPILED_LAYOUT_AND_LIVE_MEMORY_GATE",
        }
        evidence = {
            **unsigned,
            "verification_payload_sha256": hashlib.sha256(
                canonical_json_bytes(unsigned)
            ).hexdigest(),
        }
        self.authorization.assert_operation_live()
        assert self.disk_gate is not None
        self.disk_gate.before_checkpoint(
            len(canonical_json_bytes(evidence)) + 4096,
            checkpoint_id="REDUCER_CAPACITY_VERIFICATION",
        )
        atomic_write_json(
            self.runtime_root / "checkpoints/reducer_capacity_layout_verification.json",
            evidence,
        )
        self.reducer_binary = binary
        self.reducer_binary_sha256 = binary_sha

    def _bind_capacity_verification_to_replay(self) -> None:
        if not self.config.production_mode:
            return
        assert self.runtime_root is not None
        assert self.replay is not None
        assert self.reducer_resource_plan is not None
        assert self.authorization is not None
        layout_path = (
            self.runtime_root
            / "checkpoints/reducer_capacity_layout_verification.json"
        )
        layout_file_sha = sha256_file(
            _canonical_existing_file(layout_path, label="capacity layout verification")
        )
        unsigned = {
            "capacity_layout_verification_file_sha256": layout_file_sha,
            "replay_plan_sha256": self.replay.plan_identity["plan_sha256"],
            "resource_plan_sha256": self.reducer_resource_plan.plan_payload_sha256,
            "schema": "zprm.boreas.v2.stage2.reducer_capacity_replay_binding.v1",
            "verification_status": "PASS_CAPACITY_LAYOUT_BOUND_TO_RECONCILED_REPLAY",
        }
        value = {
            **unsigned,
            "binding_payload_sha256": hashlib.sha256(
                canonical_json_bytes(unsigned)
            ).hexdigest(),
        }
        self.authorization.assert_operation_live()
        assert self.disk_gate is not None
        self.disk_gate.before_checkpoint(
            len(canonical_json_bytes(value)) + 4096,
            checkpoint_id="REDUCER_CAPACITY_REPLAY_BINDING",
        )
        atomic_write_json(
            self.runtime_root / "checkpoints/reducer_capacity_replay_binding.json",
            value,
        )

    def _build_replay(self) -> ProductionMapReplayArray:
        assert self.inventory is not None
        assert self.runtime_root is not None
        assert self.disk_gate is not None
        assert self.preprocessing_contract_sha256 is not None
        assert self.gt_sha256 is not None
        assert self.extrinsic_sha256 is not None
        objects = self.inventory.for_role("TARGET_MAP")
        replay_rows = [
            ReplayMapObject(
                ordinal=item.frozen.role_ordinal,
                object_key=item.key,
                remote_size_bytes=item.size_bytes,
                etag=item.etag,
                last_modified=item.last_modified,
            )
            for item in objects
        ]
        replay_path = self.runtime_root / "map" / "transformed_xyz.f64le"
        ledger_path = self.runtime_root / "map" / "replay_ledger.jsonl"
        expected_plan = production_replay_plan_payload(
            replay_rows,
            replay_path=replay_path,
            processing_contract_sha256=self.preprocessing_contract_sha256,
            gt_sha256=self.gt_sha256,
            calibration_sha256=self.extrinsic_sha256,
            voxel_rule_sha256=target_map_voxel_rule(
                preprocessing_contract_sha256=self.preprocessing_contract_sha256
            ).contract_sha256,
            track_python_voxel_state=False,
        )
        if self._pre_gate_replay is not None:
            replay = self._pre_gate_replay
            if replay.plan_identity["plan_sha256"] != hashlib.sha256(
                canonical_json_bytes(expected_plan)
            ).hexdigest():
                raise BoreasV2Stage2RunnerError(
                    "live remote inventory differs from pre-gate authenticated replay"
                )
            self._pre_gate_replay = None
            return replay
        if replay_path.exists() != ledger_path.exists():
            allocation_intent = (
                self.runtime_root / "map/replay_allocation_intent.json"
            )
            if not allocation_intent.is_file() or allocation_intent.is_symlink():
                raise BoreasV2Stage2RunnerError("orphan map replay array or ledger")
        if not replay_path.exists():
            assert self.authorization is not None
            self.authorization.assert_operation_live()
            self.disk_gate.before_materialization(
                sum(item.remote_size_bytes for item in replay_rows),
                artifact_id="MAP_REPLAY_PREALLOCATION",
            )
        replay = ProductionMapReplayArray(
            replay_rows,
            replay_path=replay_path,
            ledger_path=ledger_path,
            processing_contract_sha256=self.preprocessing_contract_sha256,
            gt_sha256=self.gt_sha256,
            calibration_sha256=self.extrinsic_sha256,
            voxel_rule=target_map_voxel_rule(
                preprocessing_contract_sha256=self.preprocessing_contract_sha256
            ),
            track_python_voxel_state=False,
            allocation_fault_hook=(
                None
                if self.dependencies.fault_hook is None
                else lambda label: self._fault(label)
            ),
        )
        if (
            self.config.production_mode
            and self.reducer_resource_plan is not None
            and self.reducer_resource_plan.replay_plan_sha256 is not None
            and self.reducer_resource_plan.replay_plan_sha256
            != replay.plan_identity["plan_sha256"]
        ):
            raise BoreasV2Stage2RunnerError(
                "production reducer resource plan binds another replay plan"
            )
        return replay

    def _has_exact_allocated_replay_resume_evidence(
        self, allowlist: FrozenAllowlist
    ) -> bool:
        """Authenticate the complete replay before selecting the RESUME gate."""

        assert self.runtime_root is not None
        assert self.preprocessing_contract_sha256 is not None
        assert self.gt_sha256 is not None
        assert self.extrinsic_sha256 is not None
        replay_path = self.runtime_root / "map" / "transformed_xyz.f64le"
        ledger_path = self.runtime_root / "map" / "replay_ledger.jsonl"
        intent_path = self.runtime_root / "map/replay_allocation_intent.json"
        replay_exists = replay_path.exists() or replay_path.is_symlink()
        ledger_exists = ledger_path.exists() or ledger_path.is_symlink()
        intent_exists = intent_path.exists() or intent_path.is_symlink()
        if not replay_exists and not ledger_exists and not intent_exists:
            return False
        inventory = self._load_frozen_inventory_for_pre_gate(allowlist)
        replay_rows = [
            ReplayMapObject(
                ordinal=item.frozen.role_ordinal,
                object_key=item.key,
                remote_size_bytes=item.size_bytes,
                etag=item.etag,
                last_modified=item.last_modified,
            )
            for item in inventory.for_role("TARGET_MAP")
        ]
        voxel_rule = target_map_voxel_rule(
            preprocessing_contract_sha256=self.preprocessing_contract_sha256
        )
        plan = production_replay_plan_payload(
            replay_rows,
            replay_path=replay_path,
            processing_contract_sha256=self.preprocessing_contract_sha256,
            gt_sha256=self.gt_sha256,
            calibration_sha256=self.extrinsic_sha256,
            voxel_rule_sha256=voxel_rule.contract_sha256,
            track_python_voxel_state=False,
        )
        expected_bytes = int(plan["total_replay_bytes"])
        if self.config.production_mode and expected_bytes != PRODUCTION_MAP_REPLAY_BYTES:
            raise BoreasV2Stage2RunnerError(
                "production map replay size differs from the frozen 41,998,817,280 bytes"
            )
        try:
            complete = recover_production_replay_allocation_before_gate(
                plan_payload=plan,
                replay_path=replay_path,
                ledger_path=ledger_path,
            )
            if not complete:
                return False
            replay = ProductionMapReplayArray(
                replay_rows,
                replay_path=replay_path,
                ledger_path=ledger_path,
                processing_contract_sha256=self.preprocessing_contract_sha256,
                gt_sha256=self.gt_sha256,
                calibration_sha256=self.extrinsic_sha256,
                voxel_rule=voxel_rule,
                track_python_voxel_state=False,
            )
        except Exception as exc:
            raise BoreasV2Stage2RunnerError(
                "resume rejected: production replay authentication failed"
            ) from exc
        self._pre_gate_inventory = inventory
        self._pre_gate_replay = replay
        return True

    def initialize(self) -> dict[str, Any]:
        """Reverify authority/remote identity and recover every durable prefix."""

        if self.runtime_root is not None:
            raise BoreasV2Stage2RunnerError("runner instance is already initialized")
        self._authenticate_authority()
        self._authenticate_preprocessing_contract()
        self._load_production_capacity_plan()
        self.runtime_root = _prepare_runtime_root(self.config.runtime_root)
        self._acquire_runtime_lock()
        try:
            return self._initialize_under_lock()
        except Exception:
            self.close()
            raise

    def _initialize_under_lock(self) -> dict[str, Any]:
        assert self.runtime_root is not None
        self.allowlist = load_frozen_allowlist(
            self.config.allowlist_path,
            expected_sha256=self.config.expected_allowlist_sha256,
        )
        resume = self._has_exact_allocated_replay_resume_evidence(self.allowlist)
        gate_kwargs: dict[str, Any] = {
            "budget_path": self.config.disk_budget_path,
            "expected_budget_sha256": self.config.expected_disk_budget_sha256,
            "audit_log_path": self.runtime_root / "checkpoints" / "disk_gate_events.jsonl",
            "storage_mode": self.config.storage_mode,
            "now": self.dependencies.now,
        }
        if self.dependencies.free_bytes_provider is not None:
            gate_kwargs["free_bytes_provider"] = self.dependencies.free_bytes_provider
        self.disk_gate = Stage2DiskGate.from_frozen_budget(
            self.config.monitored_disk_path, **gate_kwargs
        )
        assert self.authorization is not None
        self.authorization.bind_disk_gate(self.disk_gate)
        if resume:
            self.disk_gate.assert_resume(operation_id="boreas-v2-stage2-runner-resume")
        else:
            self.disk_gate.assert_start(operation_id="boreas-v2-stage2-runner-fresh")
        self._verify_production_reducer_capacity_before_remote()
        transfer_intent_path = (
            self.runtime_root / "checkpoints/map_transfer_intent.json"
        )
        if transfer_intent_path.exists() or transfer_intent_path.is_symlink():
            if transfer_intent_path.is_symlink() or not transfer_intent_path.is_file():
                raise BoreasV2Stage2RunnerError("map transfer intent is unsafe")
            transfer_intent = self._load_map_transfer_intent()
            assert transfer_intent is not None
            self._validate_retained_download_root(transfer_intent)
        else:
            query_journal = self.runtime_root / "checkpoints/query_journal.jsonl"
            download_root = self.config.temporary_root / "tmp_download"
            retained_names = (
                {
                    path.name
                    for path in download_root.iterdir()
                    if path.name != ".stage2_temporary_root.json"
                }
                if download_root.is_dir() and not download_root.is_symlink()
                else set()
            )
            if retained_names and (query_journal.exists() or query_journal.is_symlink()):
                raise BoreasV2Stage2RunnerError(
                    "retained download may belong to query transfer recovery; "
                    "map runner refuses temporary cleanup"
                )
            _prepare_disposable_root(
                self.config.temporary_root / "tmp_download", purpose="tmp_download"
            )
        _prepare_disposable_root(
            self.config.temporary_root / "tmp_decode", purpose="tmp_decode"
        )
        self.inventory = self._freeze_or_verify_inventory(self.allowlist)
        if (
            self._pre_gate_inventory is not None
            and self._pre_gate_inventory.inventory_sha256
            != self.inventory.inventory_sha256
        ):
            raise BoreasV2Stage2RunnerError(
                "live remote inventory differs from pre-gate replay authority"
            )
        self._pre_gate_inventory = None
        self.downloader = StrictAllowlistDownloader(
            self.inventory,
            aws_executable=self.config.aws_executable,
            bucket=self.config.bucket,
            temporary_root=self.config.temporary_root / "tmp_download",
            disk_gate=self.disk_gate,
            authorization=self.authorization,
            command_runner=self.dependencies.command_runner,
            now=self.dependencies.now,
            preserve_existing_temporary_payload=(
                transfer_intent_path.exists()
            ),
        )
        self.replay = self._build_replay()
        self._bind_capacity_verification_to_replay()
        self.receipts = _ReceiptJournal(
            self.runtime_root / "checkpoints" / "receipt_checkpoints.jsonl"
        )
        recovered_transfer = self._recover_map_transfer_intent()
        self._reconcile_map_checkpoint()
        if recovered_transfer is not None and recovered_transfer.path.exists():
            assert self.authorization is not None
            self.authorization.assert_operation_live()
            self.downloader.release(recovered_transfer)
        if recovered_transfer is not None:
            self._clear_map_transfer_intent()
        self.export_receipt_evidence()
        return {
            "authorization_payload_sha256": self.authorization.document[
                "authorization_payload_sha256"
            ],
            "inventory_sha256": self.inventory.inventory_sha256,
            "map_completed_object_count": self.replay.completed_object_count,
            "map_total_object_count": len(self.replay.allowlist),
            "preprocessing_contract_sha256": self.preprocessing_contract_sha256,
            "replay_plan_identity": self.replay.plan_identity,
            "runtime_root": str(self.runtime_root),
        }

    @property
    def _map_transfer_intent_path(self) -> Path:
        if self.runtime_root is None:
            raise BoreasV2Stage2RunnerError("runtime root is not initialized")
        return self.runtime_root / "checkpoints/map_transfer_intent.json"

    def _validate_retained_download_root(self, intent: Mapping[str, Any]) -> None:
        root = self.config.temporary_root / "tmp_download"
        if (
            not root.is_dir()
            or root.is_symlink()
            or root.resolve(strict=True) != root
        ):
            raise BoreasV2Stage2RunnerError("retained download root is unsafe")
        marker_path = root / ".stage2_temporary_root.json"
        expected_marker = {
            "canonical_root": str(root),
            "purpose": "tmp_download",
            "schema": TEMP_ROOT_SCHEMA,
            "temporary": True,
        }
        if (
            marker_path.is_symlink()
            or not marker_path.is_file()
            or marker_path.read_bytes() != canonical_json_bytes(expected_marker)
        ):
            raise BoreasV2Stage2RunnerError("retained download marker differs")
        relative = Path(str(intent.get("temporary_path", "")))
        if (
            relative.is_absolute()
            or relative.parts[:1] != ("tmp_download",)
            or len(relative.parts) != 2
        ):
            raise BoreasV2Stage2RunnerError("retained transfer path is unsafe")
        expected_payload = self.config.temporary_root / relative
        extras = [path for path in root.iterdir() if path != marker_path]
        if extras and extras != [expected_payload]:
            raise BoreasV2Stage2RunnerError(
                "retained download root contains an unexpected entry"
            )
        if extras:
            metadata = os.lstat(expected_payload)
            if (
                expected_payload.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise BoreasV2Stage2RunnerError("retained transfer payload is unsafe")

    def _write_map_transfer_intent(
        self, item: AuthorizedRemoteObject
    ) -> dict[str, Any]:
        assert self.downloader is not None
        assert self.disk_gate is not None
        assert self.authorization is not None
        path = self._map_transfer_intent_path
        if path.exists() or path.is_symlink():
            raise BoreasV2Stage2RunnerError("another map transfer intent is unresolved")
        now = self.dependencies.now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise BoreasV2Stage2RunnerError("transfer-intent clock is not aware")
        temporary = self.downloader.planned_temporary_path(item)
        unsigned = {
            "etag": item.etag,
            "execution_stage": MAP_STAGE,
            "key": item.key,
            "last_modified": item.last_modified,
            "remote_size_bytes": item.size_bytes,
            "schema": TRANSFER_INTENT_SCHEMA,
            "selection_role": item.selection_role,
            "sequence_id": item.frozen.sequence_id,
            "temporary_path": temporary.relative_to(
                self.config.temporary_root
            ).as_posix(),
            "timestamp_us": item.frozen.timestamp_us,
            "transfer_intent_at_utc": now.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
        }
        value = {
            **unsigned,
            "intent_payload_sha256": hashlib.sha256(
                canonical_json_bytes(unsigned)
            ).hexdigest(),
        }
        self.authorization.assert_operation_live()
        self.disk_gate.before_checkpoint(
            len(canonical_json_bytes(value)) + 4096,
            checkpoint_id=f"MAP_TRANSFER_INTENT:{item.key}",
        )
        atomic_write_json(path, value)
        return value

    def _load_map_transfer_intent(self) -> dict[str, Any] | None:
        path = self._map_transfer_intent_path
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink() or not path.is_file():
            raise BoreasV2Stage2RunnerError("map transfer intent is unsafe")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BoreasV2Stage2RunnerError("map transfer intent is invalid") from exc
        if not isinstance(value, dict) or path.read_bytes() != canonical_json_bytes(value):
            raise BoreasV2Stage2RunnerError("map transfer intent is noncanonical")
        supplied = _require_sha256(
            value.get("intent_payload_sha256"), field="transfer intent SHA"
        )
        unsigned = {
            key: child for key, child in value.items() if key != "intent_payload_sha256"
        }
        if (
            value.get("schema") != TRANSFER_INTENT_SCHEMA
            or value.get("execution_stage") != MAP_STAGE
            or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != supplied
        ):
            raise BoreasV2Stage2RunnerError("map transfer intent binding differs")
        return value

    def _clear_map_transfer_intent(self) -> None:
        path = self._map_transfer_intent_path
        if not path.exists():
            return
        if path.is_symlink() or not path.is_file() or os.lstat(path).st_nlink != 1:
            raise BoreasV2Stage2RunnerError("map transfer intent is unsafe to clear")
        path.unlink()
        _fsync_directory(path.parent)

    def _recover_map_transfer_intent(
        self,
    ) -> TemporaryDownloadedObject | None:
        intent = self._load_map_transfer_intent()
        if intent is None:
            return None
        assert self.inventory is not None
        assert self.downloader is not None
        assert self.receipts is not None
        try:
            item = self.inventory.by_key[str(intent["key"])]
        except KeyError as exc:
            raise BoreasV2Stage2RunnerError(
                "map transfer intent is outside reconciled inventory"
            ) from exc
        planned = self.downloader.planned_temporary_path(item)
        expected_fields = {
            "etag": item.etag,
            "key": item.key,
            "last_modified": item.last_modified,
            "remote_size_bytes": item.size_bytes,
            "selection_role": item.selection_role,
            "sequence_id": item.frozen.sequence_id,
            "temporary_path": planned.relative_to(
                self.config.temporary_root
            ).as_posix(),
            "timestamp_us": item.frozen.timestamp_us,
        }
        if any(intent.get(name) != value for name, value in expected_fields.items()):
            raise BoreasV2Stage2RunnerError(
                "map transfer intent differs from reconciled remote identity"
            )
        unresolved = self.receipts.unresolved_downloads
        if unresolved:
            if len(unresolved) != 1 or unresolved[0]["receipt"]["key"] != item.key:
                raise BoreasV2Stage2RunnerError(
                    "map transfer intent differs from receipt journal"
                )
            if not planned.exists():
                self._clear_map_transfer_intent()
                return None
            materialized = self.downloader.recover_completed_transfer(
                item, recovered_at_utc=self.dependencies.now()
            )
            receipt = unresolved[0]["receipt"]
            if (
                materialized.receipt.local_temporary_sha256
                != receipt["local_temporary_sha256"]
                or materialized.receipt.remote_size_bytes
                != receipt["remote_size_bytes"]
            ):
                raise BoreasV2Stage2RunnerError(
                    "retained transfer differs from durable receipt"
                )
            return materialized
        if not planned.exists():
            self._clear_map_transfer_intent()
            return None
        metadata = os.lstat(planned)
        if (
            planned.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or planned.resolve(strict=True) != planned
            or planned.parent != self.downloader.temporary_root
        ):
            raise BoreasV2Stage2RunnerError(
                "bare-intent map transfer payload is unsafe"
            )
        # A TRANSFER_INTENT carries no durable payload digest.  Never convert
        # matching length into a receipt after a crash; discard and redownload.
        planned.unlink()
        _fsync_directory(planned.parent)
        self._clear_map_transfer_intent()
        return None

    def _require_initialized(self) -> None:
        if any(
            value is None
            for value in (
                self.runtime_root,
                self.authorization,
                self.inventory,
                self.disk_gate,
                self.downloader,
                self.replay,
                self.receipts,
            )
        ):
            raise BoreasV2Stage2RunnerError("runner is not initialized")

    def _assert_live_execution(self) -> None:
        self._require_initialized()
        if (
            self._lock_descriptor is None
            or not self.dependencies.no_registration_guard.active
        ):
            raise BoreasV2Stage2RunnerError(
                "runner lock and NoRegistrationGuard must remain active"
            )
        try:
            metadata = os.fstat(self._lock_descriptor)
        except OSError as exc:
            raise BoreasV2Stage2RunnerError("runner lock descriptor is closed") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise BoreasV2Stage2RunnerError("runner lock descriptor is unsafe")
        assert self.authorization is not None
        self.authorization.assert_operation_live()

    def _reconcile_map_checkpoint(self) -> None:
        assert self.receipts is not None
        assert self.replay is not None
        assert self.disk_gate is not None
        records = self.replay.authenticated_range_records
        committed = self.receipts.committed
        if len(committed) > len(records):
            raise BoreasV2Stage2RunnerError("committed receipts exceed replay prefix")
        for ordinal, pair in enumerate(committed):
            download, commit = pair["download"], pair["commit"]
            record = records[ordinal]
            receipt = download["receipt"]
            if (
                commit["replay_ordinal"] != ordinal
                or receipt["key"] != record["object_key"]
                or receipt["local_temporary_sha256"]
                != record["local_temporary_sha256"]
                or commit["processing_result_sha256"]
                != record["replay_range_sha256"]
            ):
                raise BoreasV2Stage2RunnerError(
                    "committed receipt differs from authenticated replay prefix"
                )
        unresolved = self.receipts.unresolved_downloads
        if len(unresolved) > 1:
            raise BoreasV2Stage2RunnerError("multiple unresolved download events")
        if unresolved:
            download = unresolved[0]
            ordinal = len(committed)
            receipt = download["receipt"]
            preprocessed = self.receipts.preprocessing_for(download["event_sha256"])
            if ordinal < len(records):
                record = records[ordinal]
                if (
                    preprocessed is None
                    or preprocessed["key"] != receipt["key"]
                    or
                    receipt["key"] != record["object_key"]
                    or receipt["local_temporary_sha256"]
                    != record["local_temporary_sha256"]
                ):
                    raise BoreasV2Stage2RunnerError(
                        "pending receipt differs from crash-completed replay range"
                    )
                self.disk_gate.before_checkpoint(
                    4096, checkpoint_id=f"RECOVER_MAP_COMMIT:{receipt['key']}"
                )
                self.receipts.commit(
                    download,
                    preprocessed,
                    processing_result_sha256=record["replay_range_sha256"],
                    replay_ordinal=ordinal,
                )
            else:
                self.disk_gate.before_checkpoint(
                    4096, checkpoint_id=f"ABORT_ORPHAN_PENDING:{receipt['key']}"
                )
                self.receipts.abort(
                    download, reason="CRASH_BEFORE_AUTHENTICATED_REPLAY_COMMIT"
                )
        if len(self.receipts.committed) != len(records):
            raise BoreasV2Stage2RunnerError("receipt/replay prefix recovery did not close")

    def run_map_ingest(self, *, synthetic_object_limit: int | None = None) -> MapIngestSummary:
        """Download, preprocess, witness, replay, checkpoint, and delete map scans."""

        self._assert_live_execution()
        assert self.inventory is not None
        assert self.replay is not None
        assert self.receipts is not None
        assert self.disk_gate is not None
        assert self.downloader is not None
        assert self.preprocessing_contract_sha256 is not None
        assert self.gt_sha256 is not None
        assert self.extrinsic_sha256 is not None
        if synthetic_object_limit is not None:
            if self.config.production_mode:
                raise BoreasV2Stage2RunnerError(
                    "partial object limits are forbidden in production mode"
                )
            _strict_positive_int(synthetic_object_limit, field="synthetic_object_limit")
        pending = list(self.replay.pending_objects)
        if synthetic_object_limit is not None:
            pending = pending[:synthetic_object_limit]
        authorized = self.inventory.by_key
        downloaded = 0
        deleted = 0
        try:
            for replay_object in pending:
                assert self.authorization is not None
                self.authorization.assert_operation_live()
                item = authorized[replay_object.object_key]
                # Treat transfer + native receipt fsync as one operation.  Gate the
                # receipt before the downloader's own fresh raw-byte gate so a
                # post-transfer low-disk decision can never erase network truth.
                self.disk_gate.before_checkpoint(
                    16 * 1024,
                    checkpoint_id=f"MAP_DOWNLOADED:{item.key}",
                )
                self._write_map_transfer_intent(item)
                sink_result: list[dict[str, Any]] = []

                def durable_receipt_sink(
                    authenticated: TemporaryDownloadedObject,
                ) -> None:
                    if sink_result:
                        raise BoreasV2Stage2RunnerError(
                            "download receipt sink was invoked more than once"
                        )
                    sink_result.append(
                        self.receipts.record_download(
                            receipt=authenticated.receipt,
                            preprocessing_contract_sha256=(
                                self.preprocessing_contract_sha256
                            ),
                            gt_sha256=self.gt_sha256,
                            extrinsic_sha256=self.extrinsic_sha256,
                        )
                    )

                try:
                    materialized = self.downloader.materialize(
                        item, authenticated_receipt_sink=durable_receipt_sink
                    )
                except Exception:
                    self._clear_map_transfer_intent()
                    raise
                if len(sink_result) != 1:
                    raise BoreasV2Stage2RunnerError(
                        "downloader returned without a durable receipt"
                    )
                download_event = sink_result[0]
                self._clear_map_transfer_intent()
                downloaded += item.size_bytes
                try:
                    self._fault("AFTER_DOWNLOADED_RECEIPT", item)
                    self.authorization.assert_operation_live()
                    raw_source = self.dependencies.map_preprocessor(materialized, item)
                    if not isinstance(raw_source, MapScanMaterialization):
                        raise BoreasV2Stage2RunnerError(
                            "map preprocessor must return MapScanMaterialization"
                        )
                    source = raw_source.validated()
                    if source.transformed_xyz.shape[0] > replay_object.point_capacity:
                        raise BoreasV2Stage2RunnerError(
                            "map preprocessing output exceeds authenticated raw capacity"
                        )
                    self.authorization.assert_operation_live()
                    self.disk_gate.before_checkpoint(
                        16 * 1024,
                        checkpoint_id=f"MAP_PREPROCESSED:{item.key}",
                    )
                    preprocessed_event = self.receipts.record_preprocessed(
                        download_event,
                        source=source,
                    )
                    self._fault("AFTER_PREPROCESSED_WITNESS", item)
                    self.authorization.assert_operation_live()
                    self.disk_gate.before_materialization(
                        item.size_bytes,
                        artifact_id=f"MAP_REPLAY_RANGE:{item.key}",
                    )
                    self.disk_gate.before_checkpoint(
                        32 * 1024,
                        checkpoint_id=f"MAP_REPLAY_LEDGER:{item.key}",
                    )
                    record = self.replay.append_transformed_scan(
                        replay_object,
                        source.transformed_xyz,
                        local_temporary_sha256=materialized.receipt.local_temporary_sha256,
                    )
                    if record["transformed_xyz_sha256"] != hashlib.sha256(
                        source.transformed_xyz.tobytes(order="C")
                    ).hexdigest():
                        raise BoreasV2Stage2RunnerError(
                            "replay active-range witness differs from preprocessing output"
                        )
                    self._fault("AFTER_REPLAY_COMMIT", item)
                    self.authorization.assert_operation_live()
                    self.disk_gate.before_checkpoint(
                        16 * 1024,
                        checkpoint_id=f"MAP_RECEIPT_COMMIT:{item.key}",
                    )
                    self.receipts.commit(
                        download_event,
                        preprocessed_event,
                        processing_result_sha256=record["replay_range_sha256"],
                        replay_ordinal=replay_object.ordinal,
                    )
                    self._fault("AFTER_MAP_COMMIT", item)
                finally:
                    if materialized.path.exists():
                        self.authorization.assert_operation_live()
                        deleted += self.downloader.release(materialized)
        finally:
            # The journal is the per-transfer fsync boundary.  CSV/audit
            # projections are intentionally rebuilt only at an explicit stage
            # boundary (including exceptional exit), avoiding O(N^2) exports.
            self.export_receipt_evidence()
        journal_path = self.receipts.path
        return MapIngestSummary(
            completed_object_count=self.replay.completed_object_count,
            total_object_count=len(self.replay.allowlist),
            downloaded_bytes_this_run=downloaded,
            deleted_temporary_bytes_this_run=deleted,
            replay_plan_sha256=self.replay.plan_identity["plan_sha256"],
            replay_complete=not self.replay.pending_objects,
            receipt_journal_sha256=(
                sha256_file(journal_path) if journal_path.exists() else None
            ),
        )

    def _receipt_rows(self) -> list[dict[str, Any]]:
        assert self.receipts is not None
        result: list[dict[str, Any]] = []
        for pair in self.receipts.committed:
            download, commit = pair["download"], pair["commit"]
            receipt = download["receipt"]
            result.append(
                {
                    "execution_stage": download["execution_stage"],
                    **receipt,
                    "processing_result_sha256": commit[
                        "processing_result_sha256"
                    ],
                    "preprocessing_contract_sha256": download[
                        "preprocessing_contract_sha256"
                    ],
                    "gt_sha256": download["gt_sha256"],
                    "extrinsic_sha256": download["extrinsic_sha256"],
                    "checkpoint_status": commit["checkpoint_status"],
                }
            )
        return result

    def export_receipt_evidence(self) -> dict[str, Any]:
        """Project durable COMMITTED events; PENDING/ABORTED never enter closure."""

        if self.runtime_root is None or self.receipts is None or self.disk_gate is None:
            raise BoreasV2Stage2RunnerError("runner is not initialized")
        rows = self._receipt_rows()
        projected = max(4096, len(rows) * 1024)
        self.disk_gate.before_checkpoint(
            projected, checkpoint_id="EXPORT_DOWNLOAD_RECEIPTS"
        )
        evidence = self.runtime_root / "evidence"
        atomic_write_csv(
            evidence / "map_download_receipts.csv",
            rows,
            RECEIPT_EXPORT_FIELDS,
            overwrite=True,
        )
        temporary_payload_bytes = sum(
            path.stat().st_size
            for root_name in ("tmp_download", "tmp_decode")
            for path in (self.config.temporary_root / root_name).rglob("*")
            if path.is_file() and path.name != ".stage2_temporary_root.json"
        )
        journal_events = self.receipts.events
        downloads = self.receipts.download_events
        committed = self.receipts.committed
        aborted = [row for row in journal_events if row["event_kind"] == "ABORTED"]
        preprocessed_events = [
            row for row in journal_events if row["event_kind"] == "PREPROCESSED"
        ]
        downloaded_bytes = sum(
            int(row["receipt"]["remote_size_bytes"]) for row in downloads
        )
        committed_bytes = sum(
            int(pair["download"]["receipt"]["remote_size_bytes"])
            for pair in committed
        )
        retry_bytes = downloaded_bytes - committed_bytes
        audit = {
            "aborted_download_event_count": len(aborted),
            "canonical_source_witness_count": len(preprocessed_events),
            "committed_payload_bytes": committed_bytes,
            "committed_payload_event_count": len(committed),
            "raw_payload_persistent_bytes": temporary_payload_bytes,
            "receipt_checkpoint_final_chain_sha256": (
                journal_events[-1]["event_sha256"] if journal_events else ZERO_SHA256
            ),
            "receipt_checkpoint_journal_path": self.receipts.path.relative_to(
                self.runtime_root
            ).as_posix(),
            "receipt_checkpoint_journal_sha256": (
                sha256_file(self.receipts.path) if self.receipts.path.exists() else None
            ),
            "replay_committed_event_count": len(committed),
            "retry_download_event_count": len(downloads) - len(committed),
            "retry_download_payload_bytes": retry_bytes,
            "schema": "zprm.boreas.v2.stage2.lidar_download_audit.v2",
            "stream_deleted_raw_bytes": (
                downloaded_bytes if temporary_payload_bytes == 0 else None
            ),
            "successful_download_event_count": len(downloads),
            "successful_payload_bytes": downloaded_bytes,
            "unique_allowlist_object_count": len(self.allowlist.objects)
            if self.allowlist is not None
            else 0,
        }
        atomic_write_json(
            evidence / "MAP_LIDAR_DOWNLOAD_AUDIT.json", audit, overwrite=True
        )
        witnesses = [
            {
                "canonical_source_witness_sha256": pair["preprocessed"][
                    "canonical_source_witness_sha256"
                ],
                "execution_stage": pair["download"]["execution_stage"],
                "key": pair["download"]["receipt"]["key"],
                "receipt_sha256": pair["download"]["receipt"]["receipt_sha256"],
                "source_point_count": pair["preprocessed"]["source_point_count"],
                "source_witness_metadata": pair["preprocessed"][
                    "source_witness_metadata"
                ],
            }
            for pair in self.receipts.committed
        ]
        witness_audit = {
            "raw_payload_deletion_order": "WITNESS_FSYNC_BEFORE_RAW_DELETE",
            "schema": "zprm.boreas.v2.stage2.canonical_source_witness_audit.v1",
            "witnesses": witnesses,
        }
        witness_audit["witness_audit_payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(witness_audit)
        ).hexdigest()
        atomic_write_json(
            evidence / "canonical_source_witness_audit.json",
            witness_audit,
            overwrite=True,
        )
        return audit

    def freeze_map_lineage(self) -> dict[str, Any]:
        self._assert_live_execution()
        assert self.runtime_root is not None
        assert self.replay is not None
        assert self.receipts is not None
        assert self.preprocessing_contract_sha256 is not None
        if self.replay.pending_objects:
            raise BoreasV2Stage2RunnerError("map lineage requires complete replay ingestion")
        records = self.replay.authenticated_range_records
        committed = self.receipts.committed
        if len(records) != len(committed):
            raise BoreasV2Stage2RunnerError("map lineage receipt/replay count differs")
        rows: list[dict[str, Any]] = []
        ledger_envelopes = self._replay_ledger_envelopes()
        for ordinal, (record, pair) in enumerate(zip(records, committed)):
            receipt = pair["download"]["receipt"]
            commit = pair["commit"]
            envelope = ledger_envelopes[ordinal + 1]
            rows.append(
                {
                    "map_ordinal": ordinal,
                    "object_key": record["object_key"],
                    "receipt_sha256": receipt["receipt_sha256"],
                    "raw_payload_sha256": record["local_temporary_sha256"],
                    "processing_result_sha256": commit[
                        "processing_result_sha256"
                    ],
                    "replay_range_sha256": record["replay_range_sha256"],
                    "replay_active_point_count": record["point_count"],
                    "transformed_xyz_sha256": record["transformed_xyz_sha256"],
                    "previous_map_state_transition_sha256": record[
                        "previous_map_state_transition_sha256"
                    ],
                    "map_state_transition_sha256": record[
                        "map_state_transition_sha256"
                    ],
                    "replay_state_transition_sha256": record[
                        "replay_state_transition_sha256"
                    ],
                    "replay_ledger_record_sha256": envelope["record_sha256"],
                }
            )
        map_objects = self.inventory.for_role("TARGET_MAP") if self.inventory else ()
        lineage = {
            "final_map_state_transition_sha256": rows[-1][
                "map_state_transition_sha256"
            ],
            "map_sequence_id": map_objects[0].frozen.sequence_id,
            "preprocessing_contract_sha256": self.preprocessing_contract_sha256,
            "gt_sha256": self.gt_sha256,
            "extrinsic_sha256": self.extrinsic_sha256,
            "query_contribution_count": 0,
            "replay_array_path": self.replay.replay_path.relative_to(
                self.runtime_root
            ).as_posix(),
            "replay_ledger_path": self.replay.ledger_path.relative_to(
                self.runtime_root
            ).as_posix(),
            "schema_version": MAP_LINEAGE_SCHEMA,
            "source_object_count": len(rows),
            "source_objects": rows,
        }
        lineage["map_lineage_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(lineage)
        ).hexdigest()
        path = self.runtime_root / "evidence" / "map_lineage_manifest.json"
        atomic_write_json(path, lineage)
        return {**lineage, "manifest_file_sha256": sha256_file(path)}

    def _replay_ledger_envelopes(self) -> list[dict[str, Any]]:
        assert self.replay is not None
        try:
            rows = [
                json.loads(line)
                for line in self.replay.ledger_path.read_text(encoding="utf-8").splitlines()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BoreasV2Stage2RunnerError("cannot parse authenticated replay ledger") from exc
        if len(rows) != self.replay.completed_object_count + 1:
            raise BoreasV2Stage2RunnerError("replay ledger envelope count differs")
        return rows

    def _prepare_reducer_binary(
        self, resource_plan: ReducerResourcePlan
    ) -> tuple[Path, str]:
        assert self.runtime_root is not None
        assert self.disk_gate is not None
        source = _canonical_existing_file(
            default_reducer_source(), label="external reducer source"
        )
        if sha256_file(source) != resource_plan.reducer_source_sha256:
            raise BoreasV2Stage2RunnerError("resource plan reducer source SHA differs")
        binary = self.runtime_root / "map" / "boreas_stage2_voxel_reduce"
        record_path = self.runtime_root / "checkpoints" / "reducer_build.json"
        compile_intent_path = (
            self.runtime_root / "checkpoints/reducer_compile_intent.json"
        )
        compile_unsigned = {
            "binary_path": binary.relative_to(self.runtime_root).as_posix(),
            "expected_binary_sha256": resource_plan.reducer_binary_sha256,
            "resource_plan_sha256": resource_plan.plan_payload_sha256,
            "schema": "zprm.boreas.v2.stage2.reducer_compile_intent.v1",
            "source_path": source.relative_to(self.config.repository).as_posix(),
            "source_sha256": resource_plan.reducer_source_sha256,
        }
        compile_intent = {
            **compile_unsigned,
            "intent_payload_sha256": hashlib.sha256(
                canonical_json_bytes(compile_unsigned)
            ).hexdigest(),
        }
        if compile_intent_path.exists() or compile_intent_path.is_symlink():
            if (
                compile_intent_path.is_symlink()
                or not compile_intent_path.is_file()
                or compile_intent_path.read_bytes()
                != canonical_json_bytes(compile_intent)
            ):
                raise BoreasV2Stage2RunnerError("reducer compile intent differs")
        else:
            assert self.authorization is not None
            self.authorization.assert_operation_live()
            self.disk_gate.before_checkpoint(
                len(canonical_json_bytes(compile_intent)) + 4096,
                checkpoint_id="EXTERNAL_REDUCER_COMPILE_INTENT",
            )
            atomic_write_json(compile_intent_path, compile_intent)
        partials = sorted(
            binary.parent.glob(f".{binary.name}.compile-partial-*")
        )
        if partials:
            assert self.authorization is not None
            self.authorization.assert_operation_live()
        for partial in partials:
            if re.fullmatch(
                rf"\.{re.escape(binary.name)}\.compile-partial-[0-9]+",
                partial.name,
            ) is None:
                raise BoreasV2Stage2RunnerError("reducer compile partial name differs")
            metadata = os.lstat(partial)
            if (
                partial.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise BoreasV2Stage2RunnerError("reducer compile partial is unsafe")
            if (
                not binary.exists()
                and resource_plan.reducer_binary_sha256 is not None
                and sha256_file(partial) == resource_plan.reducer_binary_sha256
            ):
                os.replace(partial, binary)
                _fsync_directory(binary.parent)
            else:
                partial.unlink()
                _fsync_directory(partial.parent)
        if record_path.exists():
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise BoreasV2Stage2RunnerError("reducer build checkpoint is invalid") from exc
            if (
                not binary.is_file()
                or binary.is_symlink()
                or record.get("binary_sha256") != sha256_file(binary)
                or record.get("source_sha256") != resource_plan.reducer_source_sha256
                or (
                    resource_plan.reducer_binary_sha256 is not None
                    and record.get("binary_sha256")
                    != resource_plan.reducer_binary_sha256
                )
            ):
                raise BoreasV2Stage2RunnerError("reducer build checkpoint differs")
            return binary, str(record["binary_sha256"])
        if binary.exists() or binary.is_symlink():
            if (
                resource_plan.reducer_binary_sha256 is None
                or binary.is_symlink()
                or not binary.is_file()
                or os.lstat(binary).st_nlink != 1
                or sha256_file(binary) != resource_plan.reducer_binary_sha256
            ):
                raise BoreasV2Stage2RunnerError("orphan reducer binary exists")
            record = {
                "binary_path": str(binary),
                "binary_sha256": resource_plan.reducer_binary_sha256,
                "command": ["PREBUILT_BY_CAPACITY_PLAN_HELPER"],
                "compiler_version_first_line": "RECORDED_BY_CAPACITY_PLAN_HELPER",
                "source_path": str(source),
                "source_sha256": sha256_file(source),
            }
            assert self.authorization is not None
            self.authorization.assert_operation_live()
            self.disk_gate.before_checkpoint(
                len(canonical_json_bytes(record)) + 4096,
                checkpoint_id="EXTERNAL_REDUCER_BUILD",
            )
            atomic_write_json(record_path, record)
            return binary, resource_plan.reducer_binary_sha256
        assert self.authorization is not None
        self.authorization.assert_operation_live()
        self.disk_gate.before_materialization(
            resource_plan.reducer_binary_projected_bytes,
            artifact_id="EXTERNAL_REDUCER_BINARY",
        )
        self.authorization.assert_operation_live()
        raw = self.dependencies.reducer_compiler(source=source, output=binary)
        record = _json_object(raw, field="reducer compiler result")
        binary_path = _canonical_existing_file(
            record.get("binary_path", binary), label="compiled reducer binary"
        )
        if binary_path != binary or record.get("source_sha256") != sha256_file(source):
            raise BoreasV2Stage2RunnerError("compiled reducer binding differs")
        digest = _require_sha256(record.get("binary_sha256"), field="reducer binary SHA")
        if sha256_file(binary) != digest:
            raise BoreasV2Stage2RunnerError("compiled reducer binary SHA differs")
        if (
            resource_plan.reducer_binary_sha256 is not None
            and digest != resource_plan.reducer_binary_sha256
        ):
            raise BoreasV2Stage2RunnerError(
                "compiled reducer binary differs from resource plan"
            )
        self.authorization.assert_operation_live()
        self.disk_gate.before_checkpoint(
            len(canonical_json_bytes(record)) + 4096,
            checkpoint_id="EXTERNAL_REDUCER_BUILD",
        )
        atomic_write_json(record_path, record)
        return binary, digest

    def _prepare_target_staging(self) -> Path:
        """Create/verify the fixed crash-recoverable publication staging root."""

        assert self.runtime_root is not None
        root = self.runtime_root / "map" / "target_materialization_staging"
        if root.exists() or root.is_symlink():
            if (
                root.is_symlink()
                or not root.is_dir()
                or root.resolve(strict=True) != root
            ):
                raise BoreasV2Stage2RunnerError("target staging root is unsafe")
        else:
            root.mkdir(mode=0o700)
            _fsync_directory(root.parent)
        marker = root / ".stage2_target_staging.json"
        expected = {
            "canonical_root": str(root),
            "publication_staging": True,
            "purpose": "TARGET_MAP_CONTENT_ADDRESS_PUBLICATION",
            "schema": TARGET_STAGING_SCHEMA,
        }
        if marker.exists() or marker.is_symlink():
            if marker.is_symlink() or not marker.is_file():
                raise BoreasV2Stage2RunnerError("target staging marker is unsafe")
            try:
                current = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise BoreasV2Stage2RunnerError(
                    "target staging marker is invalid"
                ) from exc
            if current != expected or marker.read_bytes() != canonical_json_bytes(expected):
                raise BoreasV2Stage2RunnerError("target staging marker differs")
        else:
            atomic_write_json(marker, expected)
        return root

    def _target_build_intent(
        self,
        *,
        plan: ReducerResourcePlan,
        descriptor_sha: str,
        binary_sha: str,
        lineage: Mapping[str, Any],
        staging_root: Path,
    ) -> tuple[dict[str, Any], bool]:
        assert self.runtime_root is not None
        assert self.replay is not None
        assert self.disk_gate is not None
        path = self.runtime_root / "checkpoints/target_build_intent.json"
        unsigned = {
            "final_map_state_transition_sha256": lineage[
                "final_map_state_transition_sha256"
            ],
            "map_lineage_manifest_sha256": lineage["manifest_file_sha256"],
            "range_descriptor_sha256": descriptor_sha,
            "reducer_binary_sha256": binary_sha,
            "replay_plan_sha256": self.replay.plan_identity["plan_sha256"],
            "resource_plan_sha256": plan.plan_payload_sha256,
            "schema": TARGET_BUILD_INTENT_SCHEMA,
            "staging_payload_path": (
                staging_root / "target_points.unpublished.npy"
            ).relative_to(self.runtime_root).as_posix(),
            "staging_root_path": staging_root.relative_to(
                self.runtime_root
            ).as_posix(),
        }
        value = {
            **unsigned,
            "intent_payload_sha256": hashlib.sha256(
                canonical_json_bytes(unsigned)
            ).hexdigest(),
        }
        existed = path.exists() or path.is_symlink()
        if existed:
            if path.is_symlink() or not path.is_file():
                raise BoreasV2Stage2RunnerError("target build intent is unsafe")
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise BoreasV2Stage2RunnerError("target build intent is invalid") from exc
            if prior != value or path.read_bytes() != canonical_json_bytes(value):
                raise BoreasV2Stage2RunnerError("target build intent binding differs")
        else:
            assert self.authorization is not None
            self.authorization.assert_operation_live()
            self.disk_gate.before_checkpoint(
                len(canonical_json_bytes(value)) + 4096,
                checkpoint_id="TARGET_MAP_BUILD_INTENT",
            )
            atomic_write_json(path, value)
        return value, existed

    def _clean_interrupted_target_staging(
        self, staging_root: Path, *, build_intent_exists: bool
    ) -> None:
        """Delete only named regular reducer partials authenticated by pre-intent."""

        entries = list(staging_root.iterdir())
        allowed = {".stage2_target_staging.json"}
        payload_names = []
        for path in entries:
            if path.name in allowed:
                continue
            if path.name == "target_points.unpublished.npy" or re.fullmatch(
                r"\.target_points\.unpublished\.npy\.materialize-partial-[0-9]+",
                path.name,
            ):
                payload_names.append(path)
                continue
            raise BoreasV2Stage2RunnerError(
                "target staging contains an unexpected entry"
            )
        if payload_names and not build_intent_exists:
            raise BoreasV2Stage2RunnerError(
                "target staging payload exists without durable build intent"
            )
        if payload_names:
            assert self.authorization is not None
            self.authorization.assert_operation_live()
        for path in payload_names:
            metadata = os.lstat(path)
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise BoreasV2Stage2RunnerError(
                    "interrupted target staging payload is unsafe"
                )
            path.unlink()
        if payload_names:
            _fsync_directory(staging_root)

    def _recover_target_intent(
        self, intent: Mapping[str, Any], *, staging_root: Path
    ) -> dict[str, Any]:
        assert self.runtime_root is not None
        assert self.authorization is not None
        staging = staging_root / "target_points.unpublished.npy"
        expected = _require_sha256(intent.get("target_map_sha256"), field="target map SHA")
        target_store = self.runtime_root / "target_maps"
        if (
            self.runtime_root.is_symlink()
            or self.runtime_root.resolve(strict=True) != self.runtime_root
            or target_store.is_symlink()
            or not target_store.is_dir()
            or target_store.resolve(strict=True) != target_store
            or staging_root.is_symlink()
            or staging_root.resolve(strict=True) != staging_root
        ):
            raise BoreasV2Stage2RunnerError("target publication parent is unsafe")
        object_directory = target_store / expected
        target = object_directory / "target_points.npy"
        metadata_path = object_directory / "metadata.json"
        if metadata_path.is_symlink():
            raise BoreasV2Stage2RunnerError("target metadata path is a symlink")
        target_present = target.exists() or target.is_symlink()
        staging_present = staging.exists() or staging.is_symlink()
        if target_present and staging_present:
            raise BoreasV2Stage2RunnerError("target exists in both staging and final paths")
        current = target if target_present else staging if staging_present else None
        if current is None or current.is_symlink() or not current.is_file():
            raise BoreasV2Stage2RunnerError("target commit intent has no payload")
        metadata = os.lstat(current)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise BoreasV2Stage2RunnerError("target payload is not regular with nlink=1")
        if sha256_file(current) != expected or current.stat().st_size != intent.get(
            "target_map_size_bytes"
        ):
            raise BoreasV2Stage2RunnerError("target payload differs from commit intent")
        if current == staging:
            if object_directory.exists() or object_directory.is_symlink():
                if (
                    object_directory.is_symlink()
                    or not object_directory.is_dir()
                    or any(object_directory.iterdir())
                ):
                    raise BoreasV2Stage2RunnerError(
                        "content-address target directory is nonempty before publish"
                    )
            else:
                object_directory.mkdir(mode=0o700)
                _fsync_directory(object_directory.parent)
            if object_directory.resolve(strict=True) != object_directory:
                raise BoreasV2Stage2RunnerError(
                    "content-address target directory is not canonical"
                )
            self.authorization.assert_operation_live()
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            directory = getattr(os, "O_DIRECTORY", 0)
            source_fd = os.open(staging_root, os.O_RDONLY | directory | nofollow)
            destination_fd = os.open(
                object_directory, os.O_RDONLY | directory | nofollow
            )
            try:
                os.replace(
                    staging.name,
                    target.name,
                    src_dir_fd=source_fd,
                    dst_dir_fd=destination_fd,
                )
            finally:
                os.close(destination_fd)
                os.close(source_fd)
            _fsync_directory(object_directory)
            _fsync_directory(staging.parent)
        check = np.load(target, mmap_mode="r", allow_pickle=False)
        if (
            check.dtype != np.dtype("<f8")
            or check.shape != (int(intent["target_point_count"]), 3)
            or not check.flags.c_contiguous
            or not np.all(np.isfinite(check))
            or canonical_array_sha256(check) != intent["target_array_sha256"]
        ):
            raise BoreasV2Stage2RunnerError("published target NPY contract differs")
        del check
        metadata = {
            "allow_pickle": False,
            "c_contiguous": True,
            "dtype": "<f8",
            "npy_version": "1.0",
            "object_kind": "target_map",
            "payload_filename": "target_points.npy",
            "payload_format": "NPY",
            "schema": "zprm-stage2-content-addressed-object-v1",
            "sha256": expected,
            "shape": [int(intent["target_point_count"]), 3],
            "size_bytes": int(intent["target_map_size_bytes"]),
            "user_metadata": {
                "array_contract_schema": "zprm-stage2-finite-float64-xyz-v1"
            },
        }
        self.authorization.assert_operation_live()
        atomic_write_json(metadata_path, metadata)
        os.chmod(target, 0o444)
        os.chmod(metadata_path, 0o444)
        descriptor = os.open(target, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        metadata_descriptor = os.open(metadata_path, os.O_RDONLY)
        try:
            os.fsync(metadata_descriptor)
        finally:
            os.close(metadata_descriptor)
        _fsync_directory(object_directory)
        for path, label in ((target, "target"), (metadata_path, "target metadata")):
            info = os.lstat(path)
            if (
                path.is_symlink()
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or path.resolve(strict=True) != path
            ):
                raise BoreasV2Stage2RunnerError(f"published {label} is unsafe")
        entries = {path.name for path in object_directory.iterdir()}
        if entries != {"target_points.npy", "metadata.json"}:
            raise BoreasV2Stage2RunnerError("content-address target entries differ")
        return dict(intent)

    def finalize_target_map(
        self, resource_plan: ReducerResourcePlan | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Run the reducer only under a compiled-layout RAM/capacity plan."""

        self._assert_live_execution()
        assert self.runtime_root is not None
        assert self.replay is not None
        assert self.disk_gate is not None
        assert self.preprocessing_contract_sha256 is not None
        if self.replay.pending_objects:
            raise BoreasV2Stage2RunnerError("target reduction requires complete map replay")
        plan = (
            resource_plan
            if isinstance(resource_plan, ReducerResourcePlan)
            else ReducerResourcePlan.from_mapping(
                resource_plan, production_mode=self.config.production_mode
            )
        )
        if (
            plan.replay_plan_sha256 is not None
            and plan.replay_plan_sha256 != self.replay.plan_identity["plan_sha256"]
        ):
            raise BoreasV2Stage2RunnerError("reducer resource plan binds another replay")
        if self.config.production_mode:
            binding_path = _canonical_existing_file(
                self.runtime_root
                / "checkpoints/reducer_capacity_replay_binding.json",
                label="capacity replay binding",
            )
            try:
                binding = json.loads(binding_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise BoreasV2Stage2RunnerError(
                    "capacity replay binding is invalid"
                ) from exc
            unsigned_binding = {
                key: value
                for key, value in binding.items()
                if key != "binding_payload_sha256"
            }
            if (
                binding.get("schema")
                != "zprm.boreas.v2.stage2.reducer_capacity_replay_binding.v1"
                or binding.get("resource_plan_sha256") != plan.plan_payload_sha256
                or binding.get("replay_plan_sha256")
                != self.replay.plan_identity["plan_sha256"]
                or binding.get("binding_payload_sha256")
                != hashlib.sha256(
                    canonical_json_bytes(unsigned_binding)
                ).hexdigest()
                or binding_path.read_bytes() != canonical_json_bytes(binding)
            ):
                raise BoreasV2Stage2RunnerError(
                    "capacity replay binding differs from live replay"
                )
        current_memory = self.dependencies.available_memory_provider()
        if (
            isinstance(current_memory, bool)
            or not isinstance(current_memory, int)
            or current_memory < plan.minimum_live_available_memory_bytes
        ):
            raise BoreasV2Stage2RunnerError(
                "BLOCKED_INSUFFICIENT_RAM: live memory is below measured reducer plan"
            )
        self._assert_live_execution()
        lineage = self.freeze_map_lineage()
        descriptor_path = self.runtime_root / "map" / "authenticated_ranges.tsv"
        self._assert_live_execution()
        _, descriptor_sha = write_range_descriptor(
            descriptor_path, self.replay.authenticated_range_records
        )
        self._assert_live_execution()
        binary, binary_sha = self._prepare_reducer_binary(plan)
        self._assert_live_execution()
        staging_root = self._prepare_target_staging()
        _build_intent, build_intent_existed = self._target_build_intent(
            plan=plan,
            descriptor_sha=descriptor_sha,
            binary_sha=binary_sha,
            lineage=lineage,
            staging_root=staging_root,
        )
        intent_path = self.runtime_root / "checkpoints" / "target_commit_intent.json"
        if intent_path.exists() or intent_path.is_symlink():
            if intent_path.is_symlink() or not intent_path.is_file():
                raise BoreasV2Stage2RunnerError("target commit intent is unsafe")
            try:
                intent = json.loads(intent_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise BoreasV2Stage2RunnerError("target commit intent is invalid") from exc
            unsigned = {key: child for key, child in intent.items() if key != "intent_payload_sha256"}
            if (
                intent.get("schema") != TARGET_COMMIT_INTENT_SCHEMA
                or intent.get("intent_payload_sha256")
                != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
                or intent.get("resource_plan_sha256") != plan.plan_payload_sha256
            ):
                raise BoreasV2Stage2RunnerError("target commit intent binding differs")
            recovered = self._recover_target_intent(
                intent, staging_root=staging_root
            )
        else:
            self._clean_interrupted_target_staging(
                staging_root, build_intent_exists=build_intent_existed
            )
            staging = staging_root / "target_points.unpublished.npy"
            self._assert_live_execution()
            result = self.dependencies.reducer_builder(
                replay_path=self.replay.replay_path,
                range_descriptor_path=descriptor_path,
                range_descriptor_sha256=descriptor_sha,
                reducer_binary=binary,
                reducer_binary_sha256=binary_sha,
                voxel_rule=self.replay.voxel_rule,
                target_points_path=staging,
                max_voxels=plan.max_voxels,
                disk_gate=self.disk_gate,
            )
            self._assert_live_execution()
            if not isinstance(result, ExternalTargetMapResult):
                raise BoreasV2Stage2RunnerError(
                    "reducer builder must return ExternalTargetMapResult"
                )
            if (
                result.target_points_path != staging
                or result.target_points_file_sha256 != sha256_file(staging)
                or result.range_descriptor_sha256 != descriptor_sha
                or result.reducer_binary_sha256 != binary_sha
                or result.voxel_rule_sha256 != self.replay.voxel_rule_sha256
            ):
                raise BoreasV2Stage2RunnerError("external reducer result binding differs")
            intent_unsigned = {
                "final_map_state_transition_sha256": lineage[
                    "final_map_state_transition_sha256"
                ],
                "map_lineage_manifest_sha256": lineage["manifest_file_sha256"],
                "range_descriptor_sha256": descriptor_sha,
                "reducer_binary_sha256": binary_sha,
                "resource_plan_sha256": plan.plan_payload_sha256,
                "schema": TARGET_COMMIT_INTENT_SCHEMA,
                "target_array_sha256": result.target_array_sha256,
                "target_map_sha256": result.target_points_file_sha256,
                "target_map_size_bytes": staging.stat().st_size,
                "target_point_count": result.point_count,
            }
            recovered = {
                **intent_unsigned,
                "intent_payload_sha256": hashlib.sha256(
                    canonical_json_bytes(intent_unsigned)
                ).hexdigest(),
            }
            self._assert_live_execution()
            self.disk_gate.before_checkpoint(
                len(canonical_json_bytes(recovered)) + 4096,
                checkpoint_id="TARGET_MAP_COMMIT_INTENT",
            )
            atomic_write_json(intent_path, recovered)
            self._assert_live_execution()
            self._fault("AFTER_TARGET_COMMIT_INTENT", None)
            recovered = self._recover_target_intent(
                recovered, staging_root=staging_root
            )
        target_path = (
            self.runtime_root
            / "target_maps"
            / recovered["target_map_sha256"]
            / "target_points.npy"
        )
        relative_target = target_path.relative_to(self.runtime_root).as_posix()
        ledger_envelopes = self._replay_ledger_envelopes()
        replay_ranges = self.replay.authenticated_range_records
        implementation = _canonical_existing_file(
            default_reducer_source(), label="external reducer source"
        )
        implementation_relative = implementation.relative_to(
            self.config.repository
        ).as_posix()
        binary_relative = binary.relative_to(self.runtime_root).as_posix()
        if self.config.production_mode:
            layout_evidence_sha = sha256_file(
                self.runtime_root
                / "checkpoints/reducer_capacity_layout_verification.json"
            )
            replay_binding_sha = sha256_file(
                self.runtime_root
                / "checkpoints/reducer_capacity_replay_binding.json"
            )
            plan_file_sha = sha256_file(self.config.reducer_resource_plan_path)
        else:
            # Synthetic execution has no production compiled-layout authority.
            # Bind the explicit synthetic plan itself in all three resource
            # witness slots without pretending a live production probe ran.
            layout_evidence_sha = plan.plan_payload_sha256
            replay_binding_sha = plan.plan_payload_sha256
            plan_file_sha = plan.plan_payload_sha256
        reducer_evidence_unsigned = {
            "final_map_state_transition_sha256": recovered[
                "final_map_state_transition_sha256"
            ],
            "map_lineage_manifest_sha256": recovered[
                "map_lineage_manifest_sha256"
            ],
            "producer_target_map_sha256": recovered["target_map_sha256"],
            "reducer_binary_path": binary_relative,
            "reducer_binary_sha256": binary_sha,
            "reducer_capacity_layout_verification_file_sha256": (
                layout_evidence_sha
            ),
            "reducer_capacity_replay_binding_file_sha256": replay_binding_sha,
            "reducer_resource_plan_file_sha256": plan_file_sha,
            "reducer_implementation_path": implementation_relative,
            "reducer_implementation_sha256": sha256_file(implementation),
            "reducer_output_target_map_sha256": recovered["target_map_sha256"],
            "replay_authenticated_ranges_sha256": hashlib.sha256(
                canonical_json_bytes(
                    [row["replay_range_sha256"] for row in replay_ranges]
                )
            ).hexdigest(),
            "replay_descriptor_count": len(replay_ranges),
            "replay_ledger_path": self.replay.ledger_path.relative_to(
                self.runtime_root
            ).as_posix(),
            "replay_ledger_sha256": sha256_file(self.replay.ledger_path),
            "replay_plan_sha256": hashlib.sha256(
                canonical_json_bytes(ledger_envelopes[0]["payload"])
            ).hexdigest(),
            "replay_total_bytes": self.replay.plan_identity[
                "total_replay_bytes"
            ],
            "schema_version": TARGET_REDUCER_VERIFICATION_SCHEMA,
            "target_map_size_bytes": recovered["target_map_size_bytes"],
            "target_point_count": recovered["target_point_count"],
            "verification_scope": (
                "PRODUCER_REDUCER_LINEAGE_AND_OUTPUT_BINDINGS_"
                "FINAL_VERIFIER_REPLAYS_EXACT_TARGET_BYTES"
            ),
            "verification_status": (
                "PASS_PRODUCER_BINDINGS_AWAITING_FINAL_EXACT_REPLAY"
            ),
        }
        reducer_evidence = {
            **reducer_evidence_unsigned,
            "target_map_reducer_verification_sha256": hashlib.sha256(
                canonical_json_bytes(reducer_evidence_unsigned)
            ).hexdigest(),
        }
        reducer_evidence_path = (
            self.runtime_root / "evidence" / "target_map_reducer_verification.json"
        )
        self._assert_live_execution()
        atomic_write_json(reducer_evidence_path, reducer_evidence)
        freeze = {
            "final_map_state_transition_sha256": recovered[
                "final_map_state_transition_sha256"
            ],
            "map_lineage_manifest_sha256": recovered[
                "map_lineage_manifest_sha256"
            ],
            "physical_target_map_copy_count": 1,
            "preprocessing_contract_sha256": self.preprocessing_contract_sha256,
            "query_contribution_count": 0,
            "schema_version": TARGET_FREEZE_SCHEMA,
            "target_map_reducer_verification_sha256": sha256_file(
                reducer_evidence_path
            ),
            "target_map_immutable": True,
            "target_map_path": relative_target,
            "target_map_sha256": recovered["target_map_sha256"],
            "target_map_size_bytes": recovered["target_map_size_bytes"],
            "target_point_count": recovered["target_point_count"],
            "unique_target_map_count": 1,
            "voxel_rule_sha256": self.replay.voxel_rule_sha256,
        }
        path = self.runtime_root / "evidence" / "target_map_freeze_manifest.json"
        self._assert_live_execution()
        atomic_write_json(path, freeze)
        return {**freeze, "manifest_file_sha256": sha256_file(path)}


def generate_production_reducer_resource_plan(
    *,
    output_plan_path: str | Path,
    reducer_binary_path: str | Path,
    safety_margin_bytes: int = MINIMUM_REDUCER_SAFETY_MARGIN_BYTES,
) -> dict[str, Any]:
    """Compile/probe the pinned reducer and freeze a replay-independent plan."""

    margin = _strict_positive_int(safety_margin_bytes, field="safety_margin_bytes")
    if margin < MINIMUM_REDUCER_SAFETY_MARGIN_BYTES:
        raise BoreasV2Stage2RunnerError("capacity safety margin must be at least 5 GiB")
    source = _canonical_existing_file(
        default_reducer_source(), label="external reducer source"
    )
    binary = Path(reducer_binary_path)
    if not binary.is_absolute():
        raise BoreasV2Stage2RunnerError("reducer binary path must be absolute")
    record = _json_object(
        compile_external_voxel_reducer(source=source, output=binary),
        field="reducer compiler result",
    )
    binary_sha = _require_sha256(
        record.get("binary_sha256"), field="reducer_binary_sha256"
    )
    layout = describe_external_voxel_capacity(
        reducer_binary=binary,
        reducer_binary_sha256=binary_sha,
        max_voxels=PRODUCTION_MAX_VOXELS,
    )
    recomputed = BoreasV2Stage2Runner._recompute_capacity_layout(
        layout, max_voxels=PRODUCTION_MAX_VOXELS
    )
    estimated = int(recomputed["total_peak_upper_bound_bytes"])
    payload = ReducerResourcePlan.signed_payload(
        capacity_probe_evidence_sha256=hashlib.sha256(
            canonical_json_bytes(recomputed)
        ).hexdigest(),
        capacity_probe_kind="COMPILED_LAYOUT_EXACT_UPPER_BOUND_V1",
        estimated_peak_memory_bytes=estimated,
        max_voxels=PRODUCTION_MAX_VOXELS,
        minimum_live_available_memory_bytes=estimated + margin,
        production_approved=True,
        reducer_binary_projected_bytes=os.lstat(binary).st_size,
        reducer_binary_sha256=binary_sha,
        reducer_source_sha256=sha256_file(source),
        safety_margin_bytes=margin,
    )
    output = Path(output_plan_path)
    if not output.is_absolute():
        raise BoreasV2Stage2RunnerError("resource-plan output path must be absolute")
    atomic_write_json(output, payload)
    return {
        "binary_path": str(binary.resolve(strict=True)),
        "binary_sha256": binary_sha,
        "capacity_layout": recomputed,
        "plan_path": str(output.resolve(strict=True)),
        "plan_sha256": sha256_file(output),
        "resource_plan": payload,
    }


__all__ = [
    "BoreasV2Stage2Runner",
    "BoreasV2Stage2RunnerConfig",
    "BoreasV2Stage2RunnerDependencies",
    "BoreasV2Stage2RunnerError",
    "MapIngestSummary",
    "MapScanMaterialization",
    "ProductionMapPreprocessor",
    "ProductionRemoteMetadataProvider",
    "RECEIPT_EXPORT_FIELDS",
    "REDUCER_RESOURCE_PLAN_SCHEMA",
    "ReducerResourcePlan",
    "generate_production_reducer_resource_plan",
    "map_source_witness_sha256",
]
