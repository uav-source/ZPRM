from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from unittest.mock import Mock

import numpy as np
import pytest

from phase_a_harness.real_data_preparation.boreas_stage2_remote import (
    AuthorizedRemoteObject,
    DownloadReceipt,
    FrozenAllowlistObject,
    ReconciledRemoteInventory,
    RemoteObjectIdentity,
    StrictAllowlistDownloader,
    TemporaryDownloadedObject,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_authorization import (
    VerifiedStage2Authorization,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_preprocessing import (
    BoreasLidarPose,
    BoreasLidarPoseIndex,
    PreprocessedBoreasScan,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_query_runner import (
    FIRST_PASS_COMMIT,
    FIRST_PASS_COMPLETE,
    GEOMETRY_METRIC_VERIFICATION_FIELDS,
    GEOMETRY_WITNESS_STATUS,
    QUERY_FIRST_PASS_STAGE,
    QUERY_SECOND_PASS_STAGE,
    SELECTION_FROZEN,
    TARGET_FROZEN,
    BoreasV2Stage2QueryRunner,
    BoreasV2Stage2QueryRunnerError,
    QueryRuntimeLease,
    QueryRunnerConfig,
    QueryRunnerDependencies,
    _canonical_line,
    _event_sha,
    _hash_json,
    _independent_production_geometry,
    _production_geometry,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    FIRST_PASS_SCAN_FIELDS,
    SYNTHETIC_AUTHORITY,
    Stage2SelectionContract,
    TargetGeometryContext,
)
from phase_a_harness.real_data_preparation.guard import NoRegistrationGuard
from phase_a_harness.real_data_preparation.io import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)
from phase_a_harness.real_data_preparation.stage2_disk_gate import (
    DiskGateThresholds,
    Stage2DiskGate,
)


QUERY_SEQUENCE = "boreas-2021-01-26-11-22"
MAP_SEQUENCE = "boreas-2021-11-14-09-47"
LAST_MODIFIED = "2021-11-18T08:37:01Z"
ETAG = "1" * 32
RAW_PAYLOAD = b"x" * 24_000
RAW_SHA = hashlib.sha256(RAW_PAYLOAD).hexdigest()
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _npy_bytes(value: np.ndarray) -> bytes:
    import io

    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(value, dtype="<f8"),
        version=(1, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _pose(timestamp: int, x: float = 0.0) -> BoreasLidarPose:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = x
    return BoreasLidarPose(timestamp, transform, np.zeros(6, dtype=np.float64))


def _pose_index(
    sequence: str, timestamps_and_x: Sequence[tuple[int, float]], digest: str
) -> BoreasLidarPoseIndex:
    poses = {timestamp: _pose(timestamp, x) for timestamp, x in timestamps_and_x}
    return BoreasLidarPoseIndex(
        sequence_id=sequence,
        source_sha256=digest,
        poses=MappingProxyType(poses),
        maximum_native_gap_s=0.1,
    )


def _inventory(count: int, timestamps: Sequence[int]) -> ReconciledRemoteInventory:
    objects: list[AuthorizedRemoteObject] = []
    for ordinal, timestamp in enumerate(timestamps):
        frozen = FrozenAllowlistObject(
            ordinal=ordinal,
            role_ordinal=ordinal,
            selection_role="QUERY",
            sequence_id=QUERY_SEQUENCE,
            key=f"{QUERY_SEQUENCE}/lidar/{timestamp}.bin",
            timestamp_us=timestamp,
            last_modified=LAST_MODIFIED,
            size_bytes=len(RAW_PAYLOAD),
            selection_reason="synthetic query fixture",
        )
        objects.append(
            AuthorizedRemoteObject(
                frozen=frozen,
                remote=RemoteObjectIdentity(
                    frozen.key, len(RAW_PAYLOAD), ETAG, LAST_MODIFIED
                ),
            )
        )
    assert len(objects) == count
    allowlist_sha = SHA_A
    inventory_sha = hashlib.sha256(
        canonical_json_bytes([item.identity() for item in objects])
    ).hexdigest()
    return ReconciledRemoteInventory(allowlist_sha, tuple(objects), inventory_sha)


def _gate(runtime: Path) -> Stage2DiskGate:
    gate = Stage2DiskGate(
        runtime,
        thresholds=DiskGateThresholds(0, 0, "SYNTHETIC_TEST_ONLY", SHA_B),
        audit_log_path=runtime / "checkpoints" / "disk.jsonl",
        free_bytes_provider=lambda _: 10**12,
        now=lambda: datetime(2026, 8, 13, tzinfo=timezone.utc),
    )
    gate.assert_start(operation_id="synthetic-query-test")
    return gate


def _resource_plan(runtime: Path, target_sha: str, target_count: int) -> tuple[Path, Path, Path]:
    measurement = runtime / "evidence" / "target_context_capacity_measurement.json"
    atomic_write_json(
        measurement,
        {
            "measured_peak_memory_bytes": 100,
            "measurement_method": "MAX_RSS_PREPARE_NORMALS_KDTREE_SAME_PINNED_ENVIRONMENT",
            "numpy_version": np.__version__,
            "schema": "zprm.boreas.v2.stage2.target_context_capacity_measurement.v1",
            "scipy_version": __import__("scipy", fromlist=["__version__"]).__version__,
            "target_map_sha256": target_sha,
            "target_point_count": target_count,
        },
    )
    path = runtime / "evidence" / "target_context_resource_plan.json"
    atomic_write_json(
        path,
        {
            "estimated_peak_memory_bytes": 100,
            "measurement_evidence_sha256": sha256_file(measurement),
            "minimum_live_available_memory_bytes": 200,
            "numpy_version": np.__version__,
            "production_approved": True,
            "safety_margin_bytes": 100,
            "schema": "zprm.boreas.v2.stage2.target_context_resource_plan.v1",
            "scipy_version": __import__("scipy", fromlist=["__version__"]).__version__,
            "target_map_sha256": target_sha,
            "target_point_count": target_count,
        },
    )
    provenance = runtime / "evidence" / "target_context_capacity_measurement_provenance.json"
    atomic_write_json(
        provenance,
        {
            "generator_sha256": SHA_E,
            "generator_path": "/synthetic/fixed-generator.py",
            "measurement_evidence_sha256": sha256_file(measurement),
            "measurement_method": "MAX_RSS_PREPARE_NORMALS_KDTREE_SAME_PINNED_ENVIRONMENT",
            "numpy_version": np.__version__,
            "platform": "synthetic",
            "python_executable": "/synthetic/python",
            "python_executable_sha256": SHA_E,
            "python_version": "synthetic",
            "schema": "zprm.boreas.v2.stage2.target_context_measurement_provenance.v1",
            "scipy_version": __import__("scipy", fromlist=["__version__"]).__version__,
            "subprocess_mode": "ISOLATED_FIXED_GENERATOR_BEFORE_ANY_QUERY_PAYLOAD",
            "target_map_sha256": target_sha,
            "target_point_count": target_count,
        },
    )
    return path, measurement, provenance


def _target_freeze(runtime: Path, target_count: int) -> tuple[Path, str]:
    target = np.stack(
        (
            np.linspace(0.0, 100.0, target_count),
            np.sin(np.linspace(0.0, 10.0, target_count)),
            np.cos(np.linspace(0.0, 10.0, target_count)),
        ),
        axis=1,
    )
    target_path = runtime / "target_maps" / "target.npy"
    atomic_write_bytes(target_path, _npy_bytes(target))
    target_sha = sha256_file(target_path)
    freeze = runtime / "evidence" / "target_map_freeze_manifest.json"
    atomic_write_json(
        freeze,
        {
            "schema_version": "boreas_v2_stage2_target_map_freeze_v1",
            "map_lineage_manifest_sha256": SHA_A,
            "preprocessing_contract_sha256": SHA_B,
            "target_map_reducer_verification_sha256": SHA_C,
            "unique_target_map_count": 1,
            "physical_target_map_copy_count": 1,
            "query_contribution_count": 0,
            "target_map_immutable": True,
            "target_map_path": "target_maps/target.npy",
            "target_map_sha256": target_sha,
            "target_map_size_bytes": target_path.stat().st_size,
            "target_point_count": target_count,
            "voxel_rule_sha256": SHA_D,
            "final_map_state_transition_sha256": SHA_E,
        },
    )
    return freeze, target_sha


def _scan(item: AuthorizedRemoteObject, pose_index: BoreasLidarPoseIndex) -> PreprocessedBoreasScan:
    points = np.column_stack(
        (
            np.linspace(1.0, 2.0, 1000),
            np.linspace(0.0, 1.0, 1000),
            np.zeros(1000),
        )
    )
    return PreprocessedBoreasScan(
        role="QUERY",
        object_key=item.key,
        sequence_id=QUERY_SEQUENCE,
        timestamp_us=item.frozen.timestamp_us,
        points_xyz=points,
        t_reference=pose_index.exact_pose(item.frozen.timestamp_us).t_enu_lidar,
        raw_point_count=1000,
        nonfinite_excluded_count=0,
        range_excluded_count=0,
        post_filter_point_count=1000,
        source_voxel_reduced_count=0,
    )


def _geometry(_: PreprocessedBoreasScan, __: TargetGeometryContext) -> Mapping[str, Any]:
    return {
        "initial_correspondence_count": 1000,
        "initial_valid_normal_correspondence_count": 100,
        "lambda_min_trans": 0.1,
        "lambda_mid_trans": 0.3,
        "lambda_max_trans": 0.6,
        "normalized_lambda_min_trans": 0.1,
        "normalized_lambda_mid_trans": 0.3,
        "normalized_lambda_max_trans": 0.6,
        "condition_number_trans": 6.0,
        "spectral_entropy_trans": 0.75,
    }


def _witness(raw_payload: bytes, **kwargs: Any) -> Mapping[str, Any]:
    bindings = kwargs["bindings"].validated()
    source = bytes(kwargs["producer_source_npy"])
    transform = bytes(kwargs["producer_t_reference_npy"])
    source_sha = hashlib.sha256(source).hexdigest()
    transform_sha = hashlib.sha256(transform).hexdigest()
    core = {
        "selection_index": bindings.selection_index,
        "snapshot_id": bindings.snapshot_id,
        "object_key": kwargs["object_key"],
        "raw_payload_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "raw_size_bytes": len(raw_payload),
        "first_pass_receipt_sha256": bindings.first_pass_receipt_sha256,
        "second_pass_receipt_sha256": bindings.second_pass_receipt_sha256,
        "preprocessing_contract_sha256": bindings.preprocessing_contract_sha256,
        "gt_sha256": bindings.gt_sha256,
        "extrinsic_sha256": bindings.extrinsic_sha256,
        "witness_implementation_sha256": bindings.witness_implementation_sha256,
        "producer_source_sha256": source_sha,
        "independent_source_sha256": source_sha,
        "producer_T_reference_sha256": transform_sha,
        "independent_T_reference_sha256": transform_sha,
        "raw_point_count": 1000,
        "nonfinite_excluded_count": 0,
        "range_excluded_count": 0,
        "post_filter_point_count": 1000,
        "source_voxel_reduced_count": 0,
        "canonical_source_point_count": 1000,
        "verification_status": "PASS_DUAL_PATH_BYTE_IDENTITY_SHARED_FROZEN_PRIMITIVES",
    }
    return {**core, "canonical_source_verification_row_sha256": _hash_json(core)}


def _make_runner(
    tmp_path: Path,
    timestamps: Sequence[int],
    windows: Sequence[Mapping[str, Any]],
    contract: Stage2SelectionContract,
    *,
    fault_hook=None,
    formal_authorization: bool = False,
) -> tuple[BoreasV2Stage2QueryRunner, Mock]:
    runtime = tmp_path / "runtime"
    for name in ("checkpoints", "evidence", "target_maps", "tmp_download"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    freeze, target_sha = _target_freeze(runtime, 10_000)
    plan, measurement, provenance = _resource_plan(runtime, target_sha, 10_000)
    inventory = _inventory(len(timestamps), timestamps)
    gate = _gate(runtime)
    guard = NoRegistrationGuard()
    guard.active = True
    authorization = Mock(spec=VerifiedStage2Authorization, unsafe=True)
    authorization.runtime_root = runtime
    authorization.no_registration_guard = guard
    authorization.formally_verified = False
    authorization.allowlist_sha256 = inventory.allowlist_sha256
    authorization.primary_pair = {
        "map_sequence_id": MAP_SEQUENCE,
        "query_sequence_id": QUERY_SEQUENCE,
    }
    authorization.document = {
        "allowlist_sha256": inventory.allowlist_sha256,
        "preprocessing_contract_sha256": SHA_B,
        "stage1_manifest_file_sha256": SHA_A,
        "storage_manifest_file_sha256": SHA_B,
        "primary_pair_sha256": SHA_C,
    }
    if formal_authorization:
        authorization.document.update(
            {
                "schema_version": "boreas_v2_stage2_download_authorization_v2",
                "authorization_scope": (
                    "BOREAS_V2_STAGE2_LIDAR_DATA_PREPARATION_ONLY"
                ),
            }
        )
    downloader = Mock(spec=StrictAllowlistDownloader, unsafe=True)
    downloader.authorization = authorization
    downloader.disk_gate = gate
    downloader.inventory = inventory
    downloader.temporary_root = runtime / "tmp_download"
    download_count = {"value": 0}

    def planned_temporary_path(item: AuthorizedRemoteObject) -> Path:
        name = (
            f"{item.frozen.ordinal:06d}-"
            f"{hashlib.sha256(item.key.encode()).hexdigest()[:24]}.bin.partial"
        )
        return downloader.temporary_root / name

    def materialize(
        item: AuthorizedRemoteObject, *, authenticated_receipt_sink=None
    ) -> TemporaryDownloadedObject:
        download_count["value"] += 1
        path = planned_temporary_path(item)
        path.write_bytes(RAW_PAYLOAD)
        receipt = DownloadReceipt(
            "QUERY",
            QUERY_SEQUENCE,
            item.key,
            item.frozen.timestamp_us,
            len(RAW_PAYLOAD),
            ETAG,
            LAST_MODIFIED,
            RAW_SHA,
            f"2026-08-13T00:00:{download_count['value'] % 60:02d}.000000Z",
        )
        value = TemporaryDownloadedObject(item, path, receipt)
        if authenticated_receipt_sink is not None:
            authenticated_receipt_sink(value)
        return value

    def release(value: TemporaryDownloadedObject) -> int:
        size = value.path.stat().st_size
        value.path.unlink()
        return size

    downloader.materialize.side_effect = materialize
    downloader.planned_temporary_path.side_effect = planned_temporary_path
    downloader.release.side_effect = release
    downloader.synthetic_download_count = download_count
    query_pairs = [(timestamp, float(index // 5) * 2.0) for index, timestamp in enumerate(timestamps)]
    # Include every interval midpoint for deterministic interval-pose authority.
    for window in windows:
        query_pairs.append(
            (
                (int(window["start_time_us"]) + int(window["end_time_us"])) // 2,
                float(window["window_index"]) * 2.0,
            )
        )
    query_pairs = sorted(dict(query_pairs).items())
    map_pairs = [(timestamp, x) for timestamp, x in query_pairs]
    query_pose = _pose_index(QUERY_SEQUENCE, query_pairs, SHA_C)
    map_pose = _pose_index(MAP_SEQUENCE, map_pairs, SHA_D)
    dependencies = QueryRunnerDependencies(
        query_preprocessor=lambda materialized, item, pose: _scan(item, pose),
        independent_query_preprocessor=lambda materialized, item, pose: _scan(item, pose),
        target_context_factory=lambda target: TargetGeometryContext.prepare(target),
        geometry_computer=_geometry,
        independent_geometry_computer=_geometry,
        canonical_witness=_witness,
        available_memory_provider=lambda: 10**12,
        fault_hook=fault_hook,
    )
    config = QueryRunnerConfig(
        runtime_root=runtime,
        preprocessing_contract_sha256=SHA_B,
        extrinsic_sha256=SHA_F,
        backend_parameter_contract_sha256=SHA_A,
        canonical_witness_implementation_sha256=SHA_E,
        expected_target_freeze_sha256=sha256_file(freeze),
        query_reference_pose_sha256=SHA_C,
        map_reference_pose_sha256=SHA_D,
        stage1_manifest_sha256=SHA_A,
        storage_manifest_sha256=SHA_B,
        stage1_allowlist_sha256=inventory.allowlist_sha256,
        pair_selection_sha256=SHA_C,
        target_context_resource_plan_sha256=sha256_file(plan),
        target_context_measurement_evidence_sha256=sha256_file(measurement),
        target_context_measurement_provenance_sha256=sha256_file(provenance),
        production_mode=False,
    )
    runner = BoreasV2Stage2QueryRunner(
        config=config,
        dependencies=dependencies,
        authorization=authorization,
        no_registration_guard=guard,
        disk_gate=gate,
        downloader=downloader,
        inventory=inventory,
        query_pose_index=query_pose,
        map_pose_index=map_pose,
        frozen_windows=windows,
        selection_contract=contract,
    )
    return runner, downloader


def test_formal_authorization_cannot_downgrade_to_synthetic_mode(
    tmp_path: Path,
) -> None:
    base = 1_610_000_000_000_000
    windows = [_window(index, base) for index in range(246)]
    offsets = (500_000, 1_500_000, 2_500_000, 3_500_000, 4_500_000)
    timestamps = [
        int(window["start_time_us"]) + offset
        for window in windows
        for offset in offsets
    ]
    with pytest.raises(
        BoreasV2Stage2QueryRunnerError,
        match="formal Boreas Stage-2 authorization",
    ):
        _make_runner(
            tmp_path,
            timestamps,
            windows,
            Stage2SelectionContract(),
            formal_authorization=True,
        )


def test_independent_geometry_path_matches_frozen_producer_fields_exactly() -> None:
    rng = np.random.default_rng(20260813)
    target = rng.normal(size=(2_000, 3))
    source = target[:1_000] + rng.normal(scale=0.02, size=(1_000, 3))
    context = TargetGeometryContext.prepare(target)
    scan = PreprocessedBoreasScan(
        role="QUERY",
        object_key=f"{QUERY_SEQUENCE}/lidar/1610000000000000.bin",
        sequence_id=QUERY_SEQUENCE,
        timestamp_us=1_610_000_000_000_000,
        points_xyz=source,
        t_reference=np.eye(4),
        raw_point_count=1_000,
        nonfinite_excluded_count=0,
        range_excluded_count=0,
        post_filter_point_count=1_000,
        source_voxel_reduced_count=0,
    )
    assert _independent_production_geometry(scan, context) == _production_geometry(
        scan, context
    )


def _window(index: int, base: int) -> dict[str, int]:
    start = base + index * 5_000_000
    return {
        "window_index": index,
        "interval_index": index,
        "start_time_us": start,
        "end_time_us": start + 5_000_000,
        "duration_us": 5_000_000,
    }


def test_first_pass_crash_records_abort_redownloads_and_deletes_raw(tmp_path: Path) -> None:
    base = 1_610_000_000_000_000
    timestamps = [base + 1_000_000]
    windows = [_window(0, base)]
    contract = Stage2SelectionContract(
        parameter_authority=SYNTHETIC_AUTHORITY,
        expected_candidate_scan_count=1,
        expected_candidate_interval_count=1,
    )
    crashed = {"value": False}

    def fault(point: str, _: AuthorizedRemoteObject | None) -> None:
        if point == "QUERY_FIRST_PASS_AFTER_DOWNLOADED" and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("synthetic crash")

    runner, downloader = _make_runner(
        tmp_path, timestamps, windows, contract, fault_hook=fault
    )
    with pytest.raises(RuntimeError, match="synthetic crash"):
        runner.run_first_pass()
    assert len(runner.journal.unresolved) == 1
    assert not list(downloader.temporary_root.glob("*.partial"))
    summary = runner.run_first_pass()
    assert summary.completed_object_count == 1
    assert downloader.synthetic_download_count["value"] == 2
    assert [row["event_kind"] for row in runner.journal.read()].count("ABORTED") == 1
    assert runner.journal.barrier(TARGET_FROZEN) is not None
    assert runner.journal.barrier(FIRST_PASS_COMPLETE) is not None
    witness = runner._first_pass_commits()[0]["payload"][
        "geometry_metric_verification_row"
    ]
    assert set(witness) == set(GEOMETRY_METRIC_VERIFICATION_FIELDS)
    assert witness["verification_status"] == GEOMETRY_WITNESS_STATUS


def _bulk_first_pass_journal(runner: BoreasV2Stage2QueryRunner) -> None:
    rows = runner.journal.read()
    previous = rows[-1]["event_sha256"]
    sequence = len(rows) + 1
    output = [runner.journal.path.read_bytes()]
    first_rows: list[dict[str, Any]] = []
    for item in runner.query_objects:
        receipt = DownloadReceipt(
            "QUERY",
            QUERY_SEQUENCE,
            item.key,
            item.frozen.timestamp_us,
            len(RAW_PAYLOAD),
            ETAG,
            LAST_MODIFIED,
            RAW_SHA,
            "2026-08-13T00:00:00.000000Z",
        )
        temporary_relative = (
            f"tmp_download/{item.frozen.ordinal:06d}-"
            f"{hashlib.sha256(item.key.encode()).hexdigest()[:24]}.bin.partial"
        )
        intent_unsigned = {
            "event_kind": "TRANSFER_INTENT",
            "execution_stage": QUERY_FIRST_PASS_STAGE,
            "object_key": item.key,
            "payload": {
                "etag": ETAG,
                "last_modified": LAST_MODIFIED,
                "remote_size_bytes": len(RAW_PAYLOAD),
                "sequence_id": QUERY_SEQUENCE,
                "temporary_relative_path": temporary_relative,
                "timestamp_us": item.frozen.timestamp_us,
            },
            "previous_event_sha256": previous,
            "schema": "zprm.boreas.v2.stage2.query_journal.v1",
            "sequence_number": sequence,
        }
        intent = {**intent_unsigned, "event_sha256": _event_sha(intent_unsigned)}
        output.append(_canonical_line(intent))
        previous = intent["event_sha256"]
        sequence += 1
        download_unsigned = {
            "event_kind": "DOWNLOADED",
            "execution_stage": QUERY_FIRST_PASS_STAGE,
            "object_key": item.key,
            "payload": {
                "receipt": receipt.as_dict(),
                "temporary_relative_path": temporary_relative,
                "transfer_intent_event_sha256": intent["event_sha256"],
            },
            "previous_event_sha256": previous,
            "schema": "zprm.boreas.v2.stage2.query_journal.v1",
            "sequence_number": sequence,
        }
        download = {**download_unsigned, "event_sha256": _event_sha(download_unsigned)}
        output.append(_canonical_line(download))
        previous = download["event_sha256"]
        sequence += 1
        window_index = item.frozen.role_ordinal // 5
        score = 0.001 + window_index * 0.0001
        first = {
            "query_ordinal": item.frozen.role_ordinal,
            "sequence_id": QUERY_SEQUENCE,
            "object_key": item.key,
            "timestamp_us": item.frozen.timestamp_us,
            "remote_size_bytes": len(RAW_PAYLOAD),
            "last_modified": LAST_MODIFIED,
            "etag": ETAG,
            "payload_sha256": RAW_SHA,
            "finite_source_point_count": 1000,
            "target_map_point_count": 10_000,
            "initial_correspondence_count": 1000,
            "initial_valid_normal_correspondence_count": 100,
            "lambda_min_trans": score,
            "lambda_mid_trans": 0.3,
            "lambda_max_trans": 0.6,
            "normalized_lambda_min_trans": score,
            "normalized_lambda_mid_trans": 0.3,
            "normalized_lambda_max_trans": 0.6 - score,
            "condition_number_trans": 1000.0 / (window_index + 1),
            "spectral_entropy_trans": 0.1 + window_index / 1000.0,
            "reference_interpolation_valid": True,
            "reference_gap_s": 0.0,
            "gt_overlap_within_5m": True,
            "target_map_frozen_complete": True,
            "deskew_processing_contract_valid": True,
            "gt_sha256": SHA_C,
            "calibration_sha256": SHA_F,
            "preprocessing_contract_sha256": SHA_B,
            "target_map_sha256": runner.target_freeze["target_map_sha256"],
        }
        assert set(first) == FIRST_PASS_SCAN_FIELDS
        first_rows.append(first)
        witness_core = {
            "query_ordinal": item.frozen.role_ordinal,
            "object_key": item.key,
            "raw_payload_sha256": RAW_SHA,
            "receipt_sha256": receipt.as_dict()["receipt_sha256"],
            "preprocessing_contract_sha256": SHA_B,
            "gt_sha256": SHA_C,
            "extrinsic_sha256": SHA_F,
            "target_map_sha256": runner.target_freeze["target_map_sha256"],
            "producer_source_sha256": SHA_A,
            "independent_source_sha256": SHA_A,
            "producer_T_reference_sha256": SHA_B,
            "independent_T_reference_sha256": SHA_B,
            "producer_geometry_sha256": _hash_json(
                {field: first[field] for field in _geometry(None, None)}
            ),
            "independent_geometry_sha256": _hash_json(
                {field: first[field] for field in _geometry(None, None)}
            ),
            "raw_point_count": 1000,
            "nonfinite_excluded_count": 0,
            "range_excluded_count": 0,
            "post_filter_point_count": 1000,
            "source_voxel_reduced_count": 0,
            "canonical_source_point_count": 1000,
            "verification_status": GEOMETRY_WITNESS_STATUS,
        }
        witness = {
            **witness_core,
            "geometry_metric_verification_row_sha256": _hash_json(witness_core),
        }
        commit_unsigned = {
            "event_kind": FIRST_PASS_COMMIT,
            "execution_stage": QUERY_FIRST_PASS_STAGE,
            "object_key": item.key,
            "payload": {
                "download_event_sha256": download["event_sha256"],
                "first_pass_row": first,
                "geometry_metric_verification_row": witness,
                "processing_result_sha256": _hash_json(first),
                "receipt_sha256": receipt.as_dict()["receipt_sha256"],
            },
            "previous_event_sha256": previous,
            "schema": "zprm.boreas.v2.stage2.query_journal.v1",
            "sequence_number": sequence,
        }
        commit = {**commit_unsigned, "event_sha256": _event_sha(commit_unsigned)}
        output.append(_canonical_line(commit))
        previous = commit["event_sha256"]
        sequence += 1
    target = runner.journal.barrier(TARGET_FROZEN)
    complete_unsigned = {
        "event_kind": FIRST_PASS_COMPLETE,
        "execution_stage": "QUERY_BARRIER",
        "object_key": "",
        "payload": {
            "first_pass_count": len(first_rows),
            "first_pass_rows_sha256": _hash_json(first_rows),
            "target_freeze_event_sha256": target["event_sha256"],
        },
        "previous_event_sha256": previous,
        "schema": "zprm.boreas.v2.stage2.query_journal.v1",
        "sequence_number": sequence,
    }
    complete = {**complete_unsigned, "event_sha256": _event_sha(complete_unsigned)}
    output.append(_canonical_line(complete))
    runner.journal.path.write_bytes(b"".join(output))
    runner.journal._cache = None
    runner.journal._cache_size = -1
    assert len(runner.journal.read()) == 3 * len(first_rows) + 2


def test_selection_freeze_tamper_and_exact_100_selected_source_resume(
    tmp_path: Path,
) -> None:
    base = 1_610_000_000_000_000
    windows = [_window(index, base) for index in range(246)]
    offsets = (500_000, 1_500_000, 2_500_000, 3_500_000, 4_500_000)
    timestamps = [
        int(window["start_time_us"]) + offset
        for window in windows
        for offset in offsets
    ]
    contract = Stage2SelectionContract(
        parameter_authority=SYNTHETIC_AUTHORITY,
        expected_candidate_scan_count=len(timestamps),
        expected_candidate_interval_count=246,
    )
    runner, downloader = _make_runner(tmp_path, timestamps, windows, contract)
    _bulk_first_pass_journal(runner)
    frozen = runner.freeze_selection()
    assert frozen["candidate_scan_count"] == 1230
    assert frozen["candidate_interval_count"] == 246
    assert frozen["selected_interval_count"] == 20
    assert frozen["selected_snapshot_count"] == 100
    assert runner.journal.barrier(SELECTION_FROZEN) is not None

    selected_path = runner.evidence_root / "selected_snapshots.csv"
    original = selected_path.read_bytes()
    selected_path.write_bytes(original.replace(b"weak", b"weAk", 1))
    with pytest.raises(BoreasV2Stage2QueryRunnerError, match="artifact changed"):
        runner._load_frozen_selection()
    selected_path.write_bytes(original)

    map_journal = runner.checkpoint_root / "map_receipts.jsonl"
    map_journal.write_bytes(b"synthetic-map-journal\n")
    atomic_write_json(
        runner.evidence_root / "MAP_LIDAR_DOWNLOAD_AUDIT.json",
        {
            "aborted_download_event_count": 0,
            "canonical_source_witness_count": 0,
            "committed_payload_bytes": 0,
            "committed_payload_event_count": 0,
            "raw_payload_persistent_bytes": 0,
            "receipt_checkpoint_final_chain_sha256": SHA_A,
            "receipt_checkpoint_journal_path": "checkpoints/map_receipts.jsonl",
            "receipt_checkpoint_journal_sha256": sha256_file(map_journal),
            "replay_committed_event_count": 0,
            "retry_download_event_count": 0,
            "retry_download_payload_bytes": 0,
            "schema": "zprm.boreas.v2.stage2.lidar_download_audit.v2",
            "stream_deleted_raw_bytes": 0,
            "successful_download_event_count": 0,
            "successful_payload_bytes": 0,
            "unique_allowlist_object_count": 0,
        },
    )
    atomic_write_csv(
        runner.evidence_root / "map_download_receipts.csv",
        [],
        (
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
        ),
    )

    first = runner.run_second_pass(object_limit=37)
    assert first.completed_object_count == 37
    second = runner.run_second_pass(object_limit=100)
    assert second.completed_object_count == 100
    assert downloader.synthetic_download_count["value"] == 100
    assert len(runner._selected_commits()) == 100
    assert len(list(runner.snapshot_root.glob("*/source_points.npy"))) == 100
    assert not list(downloader.temporary_root.glob("*.partial"))
    audit = json.loads(
        (runner.evidence_root / "QUERY_LIDAR_DOWNLOAD_AUDIT.json").read_text()
    )
    assert audit["first_pass_committed_count"] == 1230
    assert audit["selected_source_committed_count"] == 100
    assert audit["R14_selection_frozen_event_sha256"] == runner.journal.barrier(
        SELECTION_FROZEN
    )["event_sha256"]


def test_query_journal_hot_path_does_not_reparse_or_copy_full_history(
    tmp_path: Path,
) -> None:
    base = 1_610_000_000_000_000
    timestamps = [base + 1_000_000]
    windows = [_window(0, base)]
    contract = Stage2SelectionContract(
        parameter_authority=SYNTHETIC_AUTHORITY,
        expected_candidate_scan_count=1,
        expected_candidate_interval_count=1,
    )
    runner, _ = _make_runner(tmp_path, timestamps, windows, contract)
    calls = {"value": 0}
    original = runner.journal._parse_disk

    def counted():
        calls["value"] += 1
        return original()

    runner.journal._parse_disk = counted
    receipt = DownloadReceipt(
        "QUERY",
        QUERY_SEQUENCE,
        runner.query_objects[0].key,
        timestamps[0],
        len(RAW_PAYLOAD),
        ETAG,
        LAST_MODIFIED,
        RAW_SHA,
        "2026-08-13T00:00:00.000000Z",
    )
    for index in range(1000):
        # ABORTED retry pairs exercise the two hot index transitions without
        # introducing duplicate phase barriers.
        intent = runner.journal.append(
            event_kind="TRANSFER_INTENT",
            execution_stage=QUERY_FIRST_PASS_STAGE,
            object_key=runner.query_objects[0].key,
            payload={
                "etag": ETAG,
                "last_modified": LAST_MODIFIED,
                "remote_size_bytes": len(RAW_PAYLOAD),
                "sequence_id": QUERY_SEQUENCE,
                "temporary_relative_path": "tmp_download/000000-deadbeef.bin.partial",
                "timestamp_us": timestamps[0],
            },
        )
        download = runner.journal.append(
            event_kind="DOWNLOADED",
            execution_stage=QUERY_FIRST_PASS_STAGE,
            object_key=runner.query_objects[0].key,
            payload={
                "receipt": receipt.as_dict(),
                "temporary_relative_path": "tmp_download/000000-deadbeef.bin.partial",
                "transfer_intent_event_sha256": intent["event_sha256"],
            },
        )
        runner.journal.append(
            event_kind="ABORTED",
            execution_stage=QUERY_FIRST_PASS_STAGE,
            object_key=runner.query_objects[0].key,
            payload={
                "download_event_sha256": download["event_sha256"],
                "reason": f"synthetic-{index}",
                "receipt_sha256": receipt.as_dict()["receipt_sha256"],
            },
        )
    assert calls["value"] == 0
    assert runner.journal.event_count == 3001


def test_production_defaults_construct_and_all_scientific_injections_fail(
    tmp_path: Path,
) -> None:
    base = 1_610_000_000_000_000
    windows = [_window(index, base) for index in range(246)]
    # Exact production cardinality; values need only satisfy the already
    # authenticated allowlist/pose identities at construction time.
    timestamps = [base + index * 100_000 for index in range(11_859)]
    seed, _ = _make_runner(
        tmp_path,
        timestamps,
        windows,
        Stage2SelectionContract(),
    )
    seed.authorization.formally_verified = True
    seed.authorization.document.update(
        {
            "schema_version": "boreas_v2_stage2_download_authorization_v2",
            "authorization_scope": "BOREAS_V2_STAGE2_LIDAR_DATA_PREPARATION_ONLY",
        }
    )
    config = replace(seed.config, production_mode=True)
    common = {
        "config": config,
        "authorization": seed.authorization,
        "no_registration_guard": seed.no_registration_guard,
        "disk_gate": seed.disk_gate,
        "downloader": seed.downloader,
        "inventory": seed.inventory,
        "query_pose_index": seed.query_pose_index,
        "map_pose_index": seed.map_pose_index,
        "frozen_windows": windows,
        "selection_contract": Stage2SelectionContract(),
    }
    with QueryRuntimeLease(config.runtime_root) as lease:
        common["runtime_lease"] = lease
        production = BoreasV2Stage2QueryRunner(
            dependencies=QueryRunnerDependencies(), **common
        )
        assert production.config.production_mode is True
        bad_dependencies = (
            replace(
                QueryRunnerDependencies(),
                target_context_factory=lambda target: TargetGeometryContext.prepare(target),
            ),
            replace(
                QueryRunnerDependencies(), available_memory_provider=lambda: 10**12
            ),
            replace(QueryRunnerDependencies(), fault_hook=lambda *_: None),
        )
        for dependencies in bad_dependencies:
            with pytest.raises(
                BoreasV2Stage2QueryRunnerError,
                match="scientific callbacks cannot be replaced",
            ):
                BoreasV2Stage2QueryRunner(dependencies=dependencies, **common)
