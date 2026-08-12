"""Small, reproducible Boreas v2 Stage-2 preparation closure.

The target map and the 100 canonical point bundles deliberately remain in the
runtime tree.  This module only assembles the verifier-facing, small evidence
closure, verifies it against those runtime objects, and then publishes the
same bytes below ``frozen_assets``.
"""

from __future__ import annotations

import hashlib
import csv
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .boreas_v2_stage2_authorization import VerifiedStage2Authorization
from .boreas_v2_stage2_preparation_verifier import (
    PAYLOAD_FILES,
    REQUIRED_FILES,
    UNCERTAINTY_FIELDS,
    PreparationVerificationAuthority,
    verify_boreas_v2_stage2_preparation,
)
from .io import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_bytes,
    compact_sha256,
    sha256_file,
)
from .guard import NoRegistrationGuard
from .boreas_v2_stage2_runner import ReducerResourcePlan


UNKNOWN_COMPONENTS = (
    "deskew_uncertainty",
    "extrinsic_rotation_uncertainty",
    "extrinsic_translation_uncertainty",
    "orientation_uncertainty",
    "reference_interpolation_uncertainty",
    "target_map_accumulation_uncertainty",
    "time_synchronization_numeric_bound",
    "voxelization_uncertainty",
)
GENERATED_PAYLOADS = frozenset(
    {
        "boreas_v2_stage2_readiness.json",
        "boreas_v2_stage2_summary.json",
        "boreas_v2_stage2_summary.md",
    }
)
EXTERNAL_PAYLOADS = frozenset(
    {
        "boreas_v2_stage2_download_authorization.json",
        "boreas_v2_stage2_preprocessing_contract.json",
        "boreas_v2_stage2_preprocessing_contract.md",
    }
)
MAX_SINGLE_CLOSURE_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_CLOSURE_BYTES = 512 * 1024 * 1024


class BoreasV2Stage2ClosureError(RuntimeError):
    """The small closure was incomplete, unsafe, or not independently valid."""


@dataclass(frozen=True)
class ClosurePublication:
    destination: Path
    manifest_root_sha256: str
    verification_report: Mapping[str, Any]
    reused_existing_destination: bool


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_directory(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if (
        not value.is_absolute()
        or value.is_symlink()
        or not value.is_dir()
        or value.resolve(strict=True) != value
    ):
        raise BoreasV2Stage2ClosureError(f"{label} must be a canonical directory")
    return value


def _safe_regular(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise BoreasV2Stage2ClosureError(f"{label} is absent or unsafe")
    if path.resolve(strict=True) != path:
        raise BoreasV2Stage2ClosureError(f"{label} is not canonical")
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise BoreasV2Stage2ClosureError(f"{label} is not a single-link regular file")
    if metadata.st_size > MAX_SINGLE_CLOSURE_FILE_BYTES:
        raise BoreasV2Stage2ClosureError(f"{label} exceeds the small-closure limit")
    return path


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise BoreasV2Stage2ClosureError(f"immutable closure path is unsafe: {path}")
    atomic_write_bytes(path, payload, overwrite=False)
    _safe_regular(path, label=path.name)


def write_selection_prerequisites(
    *,
    evidence_root: str | Path,
    preprocessing_contract_path: str | Path,
    authorization: VerifiedStage2Authorization,
    no_registration_guard: NoRegistrationGuard,
    runtime_root: str | Path,
) -> dict[str, str]:
    """Write the UNKNOWN uncertainty and NO-ICP inputs to the selection manifest.

    These artifacts make no numerical uncertainty claim.  Existing bytes are
    treated as immutable, so a resume either reuses the exact projection or
    fails closed.
    """

    if not isinstance(authorization, VerifiedStage2Authorization):
        raise BoreasV2Stage2ClosureError("a verified Stage-2 capability is required")
    if (
        not isinstance(no_registration_guard, NoRegistrationGuard)
        or not no_registration_guard.active
        or authorization.no_registration_guard is not no_registration_guard
    ):
        raise BoreasV2Stage2ClosureError("the live authorization guard identity differs")
    authorization.assert_operation_live()
    root = _canonical_directory(evidence_root, label="runtime evidence root")
    runtime = _canonical_directory(runtime_root, label="runtime root")
    contract = _safe_regular(
        Path(preprocessing_contract_path), label="preprocessing contract"
    )
    rows = [
        {
            "component": component,
            "value": "UNKNOWN",
            "unit": "UNKNOWN",
            "uncertainty_type": "UNKNOWN",
            "status": "UNKNOWN",
            "evidence": "not numerically supported by frozen public evidence",
        }
        for component in UNKNOWN_COMPONENTS
    ]
    csv_path = root / "boreas_v2_stage2_uncertainty_budget.csv"
    json_path = root / "boreas_v2_stage2_uncertainty_budget.json"
    no_icp_path = root / "NO_ICP_ATTESTATION.json"
    contract_copy = root / "boreas_v2_stage2_preprocessing_contract.json"
    atomic_write_csv(csv_path, rows, UNCERTAINTY_FIELDS, overwrite=False)
    atomic_write_json(json_path, {"rows": rows}, overwrite=False)
    attestation = no_registration_guard.attestation(runtime)
    if attestation.get("pass") is not True:
        raise BoreasV2Stage2ClosureError(
            "NO-ICP guard attestation detected an attempt or result artifact"
        )
    atomic_write_json(
        no_icp_path,
        {
            **attestation,
            "actual_open3d_trials": 0,
            "actual_pcl_trials": 0,
            "actual_trials": 0,
            "status": "PASS",
        },
        overwrite=False,
    )
    _write_immutable(contract_copy, contract.read_bytes())
    authorization.assert_operation_live()
    return {
        path.name: sha256_file(path)
        for path in (csv_path, json_path, no_icp_path, contract_copy)
    }


def _readiness() -> dict[str, Any]:
    return {
        "BOREAS_EXTERNAL_V2_STAGE2_READY": True,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
        "READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION": True,
        "REAL_REGISTRATION_AUTHORIZED": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_trials": 0,
        "planned_future_open3d_trials": 100,
        "planned_future_pcl_trials": 100,
        "planned_future_trials": 200,
        "registration_execution_count": 0,
        "rich_snapshot_count": 50,
        "snapshot_count": 100,
        "weak_snapshot_count": 50,
        "schema": "zprm.boreas.v2.stage2.readiness.v1",
    }


def _json_file(path: Path, *, label: str) -> dict[str, Any]:
    source = _safe_regular(path, label=label)
    try:
        payload = source.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoreasV2Stage2ClosureError(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise BoreasV2Stage2ClosureError(f"noncanonical JSON object: {label}")
    return value


def _csv_rows(path: Path) -> list[dict[str, str]]:
    source = _safe_regular(path, label=path.name)
    try:
        with source.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BoreasV2Stage2ClosureError(f"invalid CSV: {path.name}") from exc
    if any(None in row for row in rows):
        raise BoreasV2Stage2ClosureError(f"malformed CSV: {path.name}")
    return rows


def _resource_evidence(
    *, runtime: Path, reducer_resource_plan_path: Path
) -> dict[str, Any]:
    expected_reducer_plan = runtime / "checkpoints/reducer_resource_plan.json"
    if reducer_resource_plan_path != expected_reducer_plan:
        raise BoreasV2Stage2ClosureError(
            "reducer resource plan must use runtime/checkpoints/reducer_resource_plan.json"
        )
    target_freeze = _json_file(
        runtime / "evidence/target_map_freeze_manifest.json",
        label="target map freeze",
    )
    target_relative = PurePosixPath(str(target_freeze.get("target_map_path", "")))
    if target_relative.is_absolute() or any(
        part in {"", ".", ".."} for part in target_relative.parts
    ):
        raise BoreasV2Stage2ClosureError("target path in freeze is unsafe")
    target_path = runtime.joinpath(*target_relative.parts)
    target_path = _safe_regular(target_path, label="runtime target map")
    target_metadata_path = _safe_regular(
        target_path.parent / "metadata.json", label="target-map metadata"
    )
    target_metadata = _json_file(target_metadata_path, label="target-map metadata")
    expected_target_metadata = {
        "allow_pickle": False,
        "c_contiguous": True,
        "dtype": "<f8",
        "npy_version": "1.0",
        "object_kind": "target_map",
        "payload_filename": "target_points.npy",
        "payload_format": "NPY",
        "schema": "zprm-stage2-content-addressed-object-v1",
        "sha256": target_freeze["target_map_sha256"],
        "shape": [target_freeze["target_point_count"], 3],
        "size_bytes": target_freeze["target_map_size_bytes"],
        "user_metadata": {
            "array_contract_schema": "zprm-stage2-finite-float64-xyz-v1"
        },
    }
    if (
        target_metadata != expected_target_metadata
        or sha256_file(target_path) != target_freeze["target_map_sha256"]
        or target_path.stat().st_size != target_freeze["target_map_size_bytes"]
    ):
        raise BoreasV2Stage2ClosureError("target metadata/payload binding differs")
    measurement_path = runtime / "evidence/target_context_capacity_measurement.json"
    context_plan_path = runtime / "evidence/target_context_resource_plan.json"
    provenance_path = (
        runtime / "evidence/target_context_capacity_measurement_provenance.json"
    )
    measurement = _json_file(measurement_path, label="target-context measurement")
    context_plan = _json_file(context_plan_path, label="target-context resource plan")
    provenance = _json_file(provenance_path, label="target-context provenance")
    measurement_sha = sha256_file(measurement_path)
    if (
        set(provenance)
        != {
            "generator_path",
            "generator_sha256",
            "measurement_evidence_sha256",
            "measurement_method",
            "numpy_version",
            "platform",
            "python_executable",
            "python_executable_sha256",
            "python_version",
            "schema",
            "scipy_version",
            "subprocess_mode",
            "target_map_sha256",
            "target_point_count",
        }
        or provenance.get("python_executable_sha256")
        != sha256_file(
            _safe_regular(
                Path(str(provenance.get("python_executable", ""))),
                label="target-context Python executable",
            )
        )
        or
        measurement.get("schema")
        != "zprm.boreas.v2.stage2.target_context_capacity_measurement.v1"
        or measurement.get("measurement_method")
        != "MAX_RSS_PREPARE_NORMALS_KDTREE_SAME_PINNED_ENVIRONMENT"
        or context_plan.get("schema")
        != "zprm.boreas.v2.stage2.target_context_resource_plan.v1"
        or context_plan.get("measurement_evidence_sha256") != measurement_sha
        or context_plan.get("estimated_peak_memory_bytes")
        != measurement.get("measured_peak_memory_bytes")
        or context_plan.get("minimum_live_available_memory_bytes")
        != context_plan.get("estimated_peak_memory_bytes")
        + context_plan.get("safety_margin_bytes")
        or context_plan.get("safety_margin_bytes", 0) < 5 * 1024**3
        or context_plan.get("production_approved") is not True
        or provenance.get("schema")
        != "zprm.boreas.v2.stage2.target_context_measurement_provenance.v1"
        or provenance.get("measurement_evidence_sha256") != measurement_sha
        or provenance.get("target_map_sha256")
        != measurement.get("target_map_sha256")
        or provenance.get("target_point_count")
        != measurement.get("target_point_count")
    ):
        raise BoreasV2Stage2ClosureError("target-context resource evidence differs")

    reducer_source = _safe_regular(
        reducer_resource_plan_path, label="reducer resource plan"
    )
    reducer_plan_value = _json_file(reducer_source, label="reducer resource plan")
    reducer_plan = ReducerResourcePlan.from_mapping(
        reducer_plan_value, production_mode=True
    )
    layout_path = runtime / "checkpoints/reducer_capacity_layout_verification.json"
    binding_path = runtime / "checkpoints/reducer_capacity_replay_binding.json"
    layout = _json_file(layout_path, label="reducer capacity layout verification")
    binding = _json_file(binding_path, label="reducer capacity replay binding")
    layout_unsigned = dict(layout)
    layout_claim = layout_unsigned.pop("verification_payload_sha256", None)
    binding_unsigned = dict(binding)
    binding_claim = binding_unsigned.pop("binding_payload_sha256", None)
    if (
        layout_claim != hashlib.sha256(canonical_json_bytes(layout_unsigned)).hexdigest()
        or binding_claim
        != hashlib.sha256(canonical_json_bytes(binding_unsigned)).hexdigest()
        or layout.get("schema")
        != "zprm.boreas.v2.stage2.reducer_capacity_layout_verification.v1"
        or binding.get("schema")
        != "zprm.boreas.v2.stage2.reducer_capacity_replay_binding.v1"
        or layout.get("resource_plan_sha256") != reducer_plan.plan_payload_sha256
        or binding.get("resource_plan_sha256") != reducer_plan.plan_payload_sha256
        or binding.get("capacity_layout_verification_file_sha256")
        != sha256_file(layout_path)
        or layout.get("capacity_layout_sha256")
        != reducer_plan.capacity_probe_evidence_sha256
        or layout.get("minimum_live_available_memory_bytes")
        != reducer_plan.minimum_live_available_memory_bytes
        or layout.get("verification_status")
        != "PASS_COMPILED_LAYOUT_AND_LIVE_MEMORY_GATE"
        or binding.get("verification_status")
        != "PASS_CAPACITY_LAYOUT_BOUND_TO_RECONCILED_REPLAY"
    ):
        raise BoreasV2Stage2ClosureError("reducer resource evidence differs")
    replay_path = _safe_regular(
        runtime / "map/transformed_xyz.f64le", label="authenticated map replay"
    )
    replay_ledger_path = _safe_regular(
        runtime / "map/replay_ledger.jsonl", label="authenticated replay ledger"
    )
    return {
        "reducer": {
            "capacity_layout_verification": layout,
            "capacity_layout_verification_file_sha256": sha256_file(layout_path),
            "capacity_replay_binding": binding,
            "capacity_replay_binding_file_sha256": sha256_file(binding_path),
            "resource_plan": reducer_plan_value,
            "resource_plan_file_sha256": sha256_file(reducer_source),
        },
        "target_context": {
            "capacity_measurement": measurement,
            "capacity_measurement_file_sha256": measurement_sha,
            "capacity_measurement_provenance": provenance,
            "capacity_measurement_provenance_file_sha256": sha256_file(
                provenance_path
            ),
            "resource_plan": context_plan,
            "resource_plan_file_sha256": sha256_file(context_plan_path),
        },
        "target_map_content_address": {
            "metadata": target_metadata,
            "metadata_file_sha256": sha256_file(target_metadata_path),
            "target_map_path": target_relative.as_posix(),
            "target_map_sha256": sha256_file(target_path),
        },
        "runtime_reverification_dependencies": {
            "authenticated_replay_ledger_path": replay_ledger_path.relative_to(
                runtime
            ).as_posix(),
            "authenticated_replay_ledger_sha256": sha256_file(replay_ledger_path),
            "authenticated_replay_path": replay_path.relative_to(runtime).as_posix(),
            "authenticated_replay_size_bytes": replay_path.stat().st_size,
            "retention_policy": (
                "RETAIN_AUTHENTICATED_REPLAY_AND_LEDGER_AFTER_FREEZE_"
                "BECAUSE_FINAL_VERIFIER_REPLAYS_EXACT_TARGET_BYTES"
            ),
        },
    }


def _summary(
    *,
    evidence: Path,
    runtime: Path,
    preprocessing_contract_path: Path,
    backend_contract_path: Path,
    reducer_resource_plan_path: Path,
    test_status_path: Path,
    disk_gate_events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    readiness = _readiness()
    preprocessing = _json_file(
        preprocessing_contract_path, label="preprocessing contract"
    )
    aggregate = _json_file(evidence / "LIDAR_DOWNLOAD_AUDIT.json", label="aggregate audit")
    map_audit = _json_file(evidence / "MAP_LIDAR_DOWNLOAD_AUDIT.json", label="map audit")
    query_audit = _json_file(evidence / "QUERY_LIDAR_DOWNLOAD_AUDIT.json", label="query audit")
    lineage = _json_file(evidence / "map_lineage_manifest.json", label="map lineage")
    target = _json_file(evidence / "target_map_freeze_manifest.json", label="target freeze")
    selection = _json_file(
        evidence / "boreas_v2_stage2_selection_manifest.json",
        label="selection manifest",
    )
    no_icp = _json_file(evidence / "NO_ICP_ATTESTATION.json", label="NO-ICP attestation")
    uncertainty = _json_file(
        evidence / "boreas_v2_stage2_uncertainty_budget.json",
        label="uncertainty budget",
    )
    test_status = _json_file(test_status_path, label="current full-test status")
    if (
        test_status_path != runtime / "evidence/full_test_status.json"
        or test_status.get("schema")
        != "zprm.boreas.v2.stage2.full_test_status.v1"
        or test_status.get("status") != "PASS"
        or test_status.get("failed") != 0
        or test_status.get("errors") != 0
        or any(
            isinstance(test_status.get(field), bool)
            or not isinstance(test_status.get(field), int)
            or test_status[field] < 0
            for field in ("collected", "passed", "skipped", "failed", "errors")
        )
        or test_status["collected"] != test_status["passed"] + test_status["skipped"]
    ):
        raise BoreasV2Stage2ClosureError("current full-test status is not a clean baseline")
    start_events = [
        dict(row)
        for row in disk_gate_events
        if row.get("operation") == "START" and row.get("pass") is True
    ]
    if not start_events:
        raise BoreasV2Stage2ClosureError("a passing Stage-2 START gate is required")
    # Crashes before the replay allocation may legitimately produce another
    # fresh START.  Report the first authenticated successful attempt and keep
    # the full count visible instead of making a valid resume unclosable.
    start = start_events[0]
    scans = _csv_rows(evidence / "all_candidate_scans.csv")
    intervals = _csv_rows(evidence / "all_candidate_intervals.csv")
    selected_intervals = _csv_rows(evidence / "selected_scene_intervals.csv")
    snapshots = _csv_rows(evidence / "selected_snapshots.csv")
    canonical = _csv_rows(evidence / "canonical_input_manifest.csv")
    unknown_rows = uncertainty.get("rows")
    if not isinstance(unknown_rows, list):
        raise BoreasV2Stage2ClosureError("uncertainty row projection is absent")
    unknown = sorted(
        str(row["component"])
        for row in unknown_rows
        if isinstance(row, Mapping) and row.get("value") == "UNKNOWN"
    )
    expected_unknown = sorted(UNKNOWN_COMPONENTS)
    weak_intervals = sum(row.get("scene_label") == "CORRIDOR_OR_WEAK_GEOMETRY" for row in selected_intervals)
    rich_intervals = sum(row.get("scene_label") == "GEOMETRY_RICH" for row in selected_intervals)
    weak_snapshots = sum(row.get("scene_label") == "CORRIDOR_OR_WEAK_GEOMETRY" for row in snapshots)
    rich_snapshots = sum(row.get("scene_label") == "GEOMETRY_RICH" for row in snapshots)
    backend_sha = sha256_file(backend_contract_path)
    backends_identical = bool(canonical) and all(
        row.get("future_open3d_source_sha256") == row.get("future_pcl_source_sha256")
        and row.get("future_open3d_target_sha256") == row.get("future_pcl_target_sha256")
        and row.get("backend_parameter_contract_sha256") == backend_sha
        and row.get("byte_identical_for_both_backends") == "True"
        for row in canonical
    )
    deleted_bytes = int(map_audit["stream_deleted_raw_bytes"]) + int(
        query_audit["successful_payload_bytes"]
    )
    zero_fields = (
        "actual_open3d_trials",
        "actual_pcl_trials",
        "actual_trials",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pcl_cli_invocation_count",
        "real_trial_result_count",
        "registration_execution_count",
    )
    empirical = {
        "backend_input_sha_equal": backends_identical,
        "backend_parameter_contract_sha256": backend_sha,
        "candidate_interval_count": len(intervals),
        "candidate_scan_count": len(scans),
        "geometry_invalid_scan_count": sum(row.get("geometry_valid") == "False" for row in scans),
        "map_scan_count": int(lineage["source_object_count"]),
        "network_download_event_count": int(aggregate["successful_download_event_count"]),
        "network_download_bytes": int(aggregate["successful_payload_bytes"]),
        "primary_pair": selection["authority_bindings"]["primary_pair"],
        "query_contribution_count": int(lineage["query_contribution_count"]),
        "raw_payload_persistent_bytes": int(aggregate["raw_payload_persistent_bytes"]),
        "raw_payload_stream_deleted_bytes": deleted_bytes,
        "rich_interval_count": rich_intervals,
        "rich_snapshot_count": rich_snapshots,
        "stage2_start_free_bytes": int(start["current_free_bytes"]),
        "successful_start_attempt_count": len(start_events),
        "target_map_physical_copy_count": int(target["physical_target_map_copy_count"]),
        "target_map_size_bytes": int(target["target_map_size_bytes"]),
        "unique_allowlist_object_count": int(aggregate["unique_allowlist_object_count"]),
        "weak_interval_count": weak_intervals,
        "weak_snapshot_count": weak_snapshots,
        "snapshot_count": len(snapshots),
    }
    if (
        empirical["raw_payload_persistent_bytes"] != 0
        or empirical["raw_payload_stream_deleted_bytes"]
        != empirical["network_download_bytes"]
        or empirical["map_scan_count"] != 8202
        or empirical["target_map_physical_copy_count"] != 1
        or empirical["query_contribution_count"] != 0
        or empirical["candidate_scan_count"] != 11859
        or empirical["candidate_interval_count"] != 246
        or (weak_intervals, rich_intervals) != (10, 10)
        or (weak_snapshots, rich_snapshots, len(snapshots)) != (50, 50, 100)
        or len(canonical) != 100
        or not backends_identical
        or unknown != expected_unknown
        or no_icp.get("pass") is not True
        or any(no_icp.get(field) != 0 for field in zero_fields)
        or selection.get("R14_frozen") is not True
    ):
        raise BoreasV2Stage2ClosureError("summary evidence does not meet frozen Stage-2 closure")
    resource_evidence = _resource_evidence(
        runtime=runtime, reducer_resource_plan_path=reducer_resource_plan_path
    )
    questions = [
        (1, "Stage-2 启动时实际可用磁盘多少 GiB？", {"bytes": empirical["stage2_start_free_bytes"], "GiB": empirical["stage2_start_free_bytes"] / 1024**3}),
        (2, "实际下载多少个 LiDAR objects？", {"unique_objects": empirical["unique_allowlist_object_count"], "successful_transfer_events": empirical["network_download_event_count"]}),
        (3, "实际网络下载多少 GB？", {"bytes": empirical["network_download_bytes"], "GB": empirical["network_download_bytes"] / 1e9}),
        (4, "raw payload 最终持久保留多少 GB？", {"bytes": empirical["raw_payload_persistent_bytes"], "GB": empirical["raw_payload_persistent_bytes"] / 1e9}),
        (5, "临时 payload 删除多少 GB？", {"bytes": deleted_bytes, "GB": deleted_bytes / 1e9}),
        (6, "是否始终使用 PRIMARY_PAIR？", {"yes": True, "pair": empirical["primary_pair"]}),
        (7, "preprocessing 参数最终是什么？", {"contract_sha256": sha256_file(preprocessing_contract_path), "filtering": preprocessing["filtering"], "source_downsampling": preprocessing["source_downsampling"], "map_accumulation": preprocessing["map_accumulation"], "target_geometry_analysis": preprocessing["target_geometry_analysis"], "crop_policy": preprocessing["crop_policy"], "dynamic_object_policy": preprocessing["dynamic_object_policy"]}),
        (8, "preprocessing 是否在任何 geometry metric 前冻结？", {"yes": preprocessing["execution_state_at_freeze"]["geometry_metric_execution_count"] == 0, "execution_state_at_freeze": preprocessing["execution_state_at_freeze"]}),
        (9, "deskew 怎么做？", preprocessing["deskew"]),
        (10, "是否使用任何 LiDAR odometry？", {"used": preprocessing["deskew"]["uses_lidar_odometry"]}),
        (11, "target map 使用多少 map scans？", empirical["map_scan_count"]),
        (12, "target map 是否只有一份？", {"yes": True, "physical_copy_count": empirical["target_map_physical_copy_count"]}),
        (13, "target map 大小多少 GiB？", {"bytes": empirical["target_map_size_bytes"], "GiB": empirical["target_map_size_bytes"] / 1024**3}),
        (14, "query 是否 0 contribution to map？", {"yes": empirical["query_contribution_count"] == 0, "count": empirical["query_contribution_count"]}),
        (15, "query screening 处理多少 candidate scans？", empirical["candidate_scan_count"]),
        (16, "geometry-invalid 排除多少？", empirical["geometry_invalid_scan_count"]),
        (17, "candidate 5 s intervals 有多少？", empirical["candidate_interval_count"]),
        (18, "是否成功获得 10 weak intervals？", {"yes": weak_intervals == 10, "count": weak_intervals}),
        (19, "是否成功获得 10 rich intervals？", {"yes": rich_intervals == 10, "count": rich_intervals}),
        (20, "weak 50 是否完成？", {"yes": weak_snapshots == 50, "count": weak_snapshots}),
        (21, "rich 50 是否完成？", {"yes": rich_snapshots == 50, "count": rich_snapshots}),
        (22, "canonical snapshot 是否精确 100 个？", {"yes": len(canonical) == 100, "count": len(canonical)}),
        (23, "future Open3D/PCL 输入 SHA 是否完全相同？", {"yes": backends_identical, "backend_parameter_contract_sha256": backend_sha}),
        (24, "R14 是否成功冻结？", {"yes": selection["R14_frozen"] is True, "selection_manifest_sha256": selection["selection_manifest_sha256"]}),
        (25, "uncertainty 中仍有哪些 UNKNOWN？", unknown),
        (26, "ICP 是否严格为 0？", {"yes": True, "attestation": no_icp}),
        (27, "全量测试是否 0 failed？", {"yes": True, "test_status": test_status, "test_status_file_sha256": sha256_file(test_status_path)}),
        (28, "independent verifier 是否 PASS？", {"status": "PASS_REQUIRED_BEFORE_AND_AFTER_PUBLICATION"}),
        (29, "BOREAS_EXTERNAL_V2_STAGE2_READY 是否 true？", readiness["BOREAS_EXTERNAL_V2_STAGE2_READY"]),
        (30, "是否已经可以单独授权 Stage-3 的 200 次 registration？", {"ready_for_separate_authorization": readiness["READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION"], "planned_future_trials": readiness["planned_future_trials"], "automatically_authorized": False}),
    ]
    answers = [
        {"answer": answer, "item": item, "question": question}
        for item, question, answer in questions
    ]
    value = {
        "answers": answers,
        "conclusion": (
            "BOREAS_EXTERNAL_V2_STAGE2_READY=true; "
            "READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION=true; "
            "Stage-3 was not automatically authorized or executed."
        ),
        "empirical_counts_and_bytes": empirical,
        "large_artifact_policy": (
            "TARGET_CANONICAL_BUNDLES_AND_AUTHENTICATED_REPLAY_REQUIRED_BY_FINAL_VERIFIER_REMAIN_RUNTIME_ONLY"
        ),
        "readiness": readiness,
        "resource_capacity_evidence": resource_evidence,
        "schema": "zprm.boreas.v2.stage2.preparation_summary.v1",
    }
    lines = [
        "# Boreas External Validation v2 — Stage-2 preparation",
        "",
        "独立发布门要求候选、staging 和最终 frozen destination 均通过 verifier；"
        "本文件只在这些门全部启用的发布流程中生成。Stage-3 未获自动授权，也未执行。",
        "",
    ]
    for row in answers:
        rendered = json.dumps(row["answer"], sort_keys=True, ensure_ascii=False)
        lines.extend((f"{row['item']}. {row['question']}", "", f"   `{rendered}`", ""))
    return value, ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _copy_payload(source: Path, destination: Path) -> None:
    source = _safe_regular(source, label=f"closure source {source.name}")
    _write_immutable(destination, source.read_bytes())


def _candidate_intent(candidate: Path, runtime: Path) -> dict[str, Any]:
    unsigned = {
        "candidate_root": str(candidate),
        "runtime_root": str(runtime),
        "schema": "zprm.boreas.v2.stage2.closure_candidate_intent.v1",
    }
    return {**unsigned, "intent_sha256": compact_sha256(unsigned)}


def _candidate_intent_path(candidate: Path) -> Path:
    return candidate.parent / f".{candidate.name}.assembly_intent.json"


def _validate_completed_candidate(
    candidate: Path, *, runtime: Path, authority: PreparationVerificationAuthority
) -> dict[str, Any]:
    if {entry.name for entry in candidate.iterdir()} != set(REQUIRED_FILES):
        raise BoreasV2Stage2ClosureError("candidate root is not complete")
    for name in REQUIRED_FILES:
        _safe_regular(candidate / name, label=f"candidate {name}")
    manifest = _json_file(
        candidate / "boreas_v2_stage2_frozen_manifest.json",
        label="candidate frozen manifest",
    )
    unsigned = dict(manifest)
    claim = unsigned.pop("manifest_root_sha256", None)
    if claim != compact_sha256(unsigned):
        raise BoreasV2Stage2ClosureError("candidate manifest self-hash differs")
    payload = manifest.get("payload")
    if not isinstance(payload, list) or payload != [
        {
            "path": name,
            "sha256": sha256_file(candidate / name),
            "size_bytes": (candidate / name).stat().st_size,
        }
        for name in sorted(PAYLOAD_FILES)
    ]:
        raise BoreasV2Stage2ClosureError("candidate payload manifest differs")
    checksum_names = sorted(REQUIRED_FILES - {"SHA256SUMS"})
    expected_sums = "".join(
        f"{sha256_file(candidate / name)}  {name}\n" for name in checksum_names
    ).encode("utf-8")
    if (candidate / "SHA256SUMS").read_bytes() != expected_sums:
        raise BoreasV2Stage2ClosureError("candidate SHA256SUMS differs")
    _verify_generated_documents(candidate, runtime=runtime, authority=authority)
    return manifest


def _recover_candidate(candidate: Path, *, runtime: Path) -> dict[str, Any] | None:
    """Reuse a complete candidate or remove only marked, regular partial files."""

    intent_path = _candidate_intent_path(candidate)
    expected_intent = _candidate_intent(candidate, runtime)
    if candidate.exists():
        candidate = _canonical_directory(candidate, label="closure candidate root")
        actual_names = {entry.name for entry in candidate.iterdir()}
        if actual_names == set(REQUIRED_FILES):
            return None  # The caller performs authority-aware exact validation.
        if not intent_path.is_file() or intent_path.is_symlink():
            raise BoreasV2Stage2ClosureError(
                "partial closure candidate lacks its authenticated assembly intent"
            )
        if _json_file(intent_path, label="closure candidate intent") != expected_intent:
            raise BoreasV2Stage2ClosureError("closure candidate intent differs")
        for entry in list(candidate.iterdir()):
            _safe_regular(entry, label=f"partial closure candidate {entry.name}")
        for entry in list(candidate.iterdir()):
            entry.unlink()
        candidate.rmdir()
        _fsync_directory(candidate.parent)
    if intent_path.exists():
        if _json_file(intent_path, label="closure candidate intent") != expected_intent:
            raise BoreasV2Stage2ClosureError("closure candidate intent differs")
    else:
        atomic_write_json(intent_path, expected_intent, overwrite=False)
        _fsync_directory(candidate.parent)
    return expected_intent


def assemble_small_closure_candidate(
    *,
    candidate_root: str | Path,
    evidence_root: str | Path,
    authorization_path: str | Path,
    preprocessing_contract_json: str | Path,
    preprocessing_contract_markdown: str | Path,
    backend_contract_path: str | Path,
    reducer_resource_plan_path: str | Path,
    test_status_path: str | Path,
    disk_gate_events: Sequence[Mapping[str, Any]],
    runtime_root: str | Path,
    authority: PreparationVerificationAuthority,
) -> dict[str, Any]:
    """Assemble the exact verifier closure without copying any large arrays."""

    evidence = _canonical_directory(evidence_root, label="runtime evidence root")
    runtime = _canonical_directory(runtime_root, label="runtime root")
    candidate = Path(candidate_root)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise BoreasV2Stage2ClosureError("candidate root must be absolute and non-symlink")
    if not candidate.parent.is_dir() or candidate.parent.is_symlink():
        raise BoreasV2Stage2ClosureError("candidate parent is absent or unsafe")
    intent_path = _candidate_intent_path(candidate)
    if candidate.exists() and {entry.name for entry in candidate.iterdir()} == set(
        REQUIRED_FILES
    ):
        completed = _validate_completed_candidate(
            _canonical_directory(candidate, label="closure candidate root"),
            runtime=runtime,
            authority=authority,
        )
        if intent_path.exists():
            if _json_file(intent_path, label="closure candidate intent") != _candidate_intent(
                candidate, runtime
            ):
                raise BoreasV2Stage2ClosureError("closure candidate intent differs")
            intent_path.unlink()
            _fsync_directory(candidate.parent)
        return completed
    _recover_candidate(candidate, runtime=runtime)
    candidate.mkdir(mode=0o700, parents=False, exist_ok=False)
    candidate = _canonical_directory(candidate, label="closure candidate root")

    source_overrides = {
        "boreas_v2_stage2_download_authorization.json": Path(authorization_path),
        "boreas_v2_stage2_preprocessing_contract.json": Path(
            preprocessing_contract_json
        ),
        "boreas_v2_stage2_preprocessing_contract.md": Path(
            preprocessing_contract_markdown
        ),
    }
    for name in sorted(PAYLOAD_FILES - GENERATED_PAYLOADS):
        _copy_payload(source_overrides.get(name, evidence / name), candidate / name)

    readiness = _readiness()
    summary, summary_markdown = _summary(
        evidence=evidence,
        runtime=runtime,
        preprocessing_contract_path=Path(preprocessing_contract_json),
        backend_contract_path=Path(backend_contract_path),
        reducer_resource_plan_path=Path(reducer_resource_plan_path),
        test_status_path=Path(test_status_path),
        disk_gate_events=disk_gate_events,
    )
    atomic_write_json(candidate / "boreas_v2_stage2_readiness.json", readiness)
    atomic_write_json(candidate / "boreas_v2_stage2_summary.json", summary)
    _write_immutable(candidate / "boreas_v2_stage2_summary.md", summary_markdown)

    payload_rows = [
        {
            "path": name,
            "sha256": sha256_file(candidate / name),
            "size_bytes": (candidate / name).stat().st_size,
        }
        for name in sorted(PAYLOAD_FILES)
    ]
    total = sum(int(row["size_bytes"]) for row in payload_rows)
    if total > MAX_TOTAL_CLOSURE_BYTES:
        raise BoreasV2Stage2ClosureError("candidate exceeds the small-closure total limit")
    manifest_core = {
        "pair_selection_sha256": sha256_file(authority.pair_selection_path),
        "payload": payload_rows,
        "preprocessing_contract_sha256": sha256_file(
            authority.preprocessing_contract_path
        ),
        "schema_version": "boreas_v2_stage2_preparation_frozen_manifest_v1",
        "stage1_allowlist_sha256": sha256_file(authority.stage1_allowlist_path),
        "stage1_manifest_sha256": sha256_file(authority.stage1_manifest_path),
        "storage_manifest_sha256": sha256_file(authority.storage_manifest_path),
    }
    manifest = {
        **manifest_core,
        "manifest_root_sha256": compact_sha256(manifest_core),
    }
    atomic_write_json(candidate / "boreas_v2_stage2_frozen_manifest.json", manifest)
    checksum_names = sorted(REQUIRED_FILES - {"SHA256SUMS"})
    sums = "".join(
        f"{sha256_file(candidate / name)}  {name}\n" for name in checksum_names
    ).encode("utf-8")
    _write_immutable(candidate / "SHA256SUMS", sums)
    actual = {entry.name for entry in candidate.iterdir()}
    if actual != set(REQUIRED_FILES):
        raise BoreasV2Stage2ClosureError("candidate root contains an open file inventory")
    _fsync_directory(candidate)
    intent_path.unlink()
    _fsync_directory(candidate.parent)
    return manifest


def _same_tree(left: Path, right: Path) -> bool:
    left_names = {entry.name for entry in left.iterdir()}
    right_names = {entry.name for entry in right.iterdir()}
    return left_names == right_names and all(
        _safe_regular(left / name, label=f"left {name}").stat().st_size
        == _safe_regular(right / name, label=f"right {name}").stat().st_size
        and sha256_file(left / name) == sha256_file(right / name)
        for name in left_names
    )


def _publication_intent(
    *,
    candidate: Path,
    destination: Path,
    staging: Path,
    runtime: Path,
    manifest_root_sha256: str,
) -> dict[str, Any]:
    unsigned = {
        "candidate_root": str(candidate),
        "destination_root": str(destination),
        "manifest_root_sha256": manifest_root_sha256,
        "payload": [
            {
                "path": name,
                "sha256": sha256_file(candidate / name),
                "size_bytes": (candidate / name).stat().st_size,
            }
            for name in sorted(REQUIRED_FILES)
        ],
        "runtime_root": str(runtime),
        "schema": "zprm.boreas.v2.stage2.closure_publication_intent.v1",
        "staging_root": str(staging),
    }
    return {**unsigned, "intent_sha256": compact_sha256(unsigned)}


def _publication_intent_path(runtime: Path) -> Path:
    checkpoints = _canonical_directory(
        runtime / "checkpoints", label="runtime checkpoints root"
    )
    return checkpoints / "closure_publication_intent.json"


def _clear_publication_intent(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    _safe_regular(path, label="closure publication intent").unlink()
    _fsync_directory(path.parent)


def _recover_or_fill_publication_staging(
    *,
    candidate: Path,
    staging: Path,
    live_check: Callable[[], Any],
) -> None:
    """Authenticate every retained byte and fill only missing closure files."""

    if staging.exists() or staging.is_symlink():
        staging = _canonical_directory(staging, label="frozen closure staging root")
        existing_names = {entry.name for entry in staging.iterdir()}
        extras = existing_names - set(REQUIRED_FILES)
        for name in sorted(extras):
            temporary = staging / name
            if not any(
                re.fullmatch(rf"\.{re.escape(required)}\.[^.\/]+\.partial", name)
                for required in REQUIRED_FILES
            ):
                raise BoreasV2Stage2ClosureError(
                    "publication staging contains an unexpected file"
                )
            _safe_regular(
                temporary, label=f"interrupted publication temporary {name}"
            ).unlink()
        if extras:
            _fsync_directory(staging)
            existing_names -= extras
        for name in sorted(existing_names):
            retained = _safe_regular(
                staging / name, label=f"retained publication staging {name}"
            )
            source = _safe_regular(candidate / name, label=f"candidate {name}")
            if (
                retained.stat().st_size != source.stat().st_size
                or sha256_file(retained) != sha256_file(source)
            ):
                raise BoreasV2Stage2ClosureError(
                    "publication staging contains different bytes"
                )
    else:
        staging.mkdir(mode=0o700, parents=False, exist_ok=False)
        staging = _canonical_directory(staging, label="frozen closure staging root")
        existing_names = set()
        _fsync_directory(staging.parent)
    for name in sorted(set(REQUIRED_FILES) - existing_names):
        live_check()
        _copy_payload(candidate / name, staging / name)
    _fsync_directory(staging)


def recover_closure_publication_staging(
    *,
    candidate_root: str | Path,
    destination_root: str | Path,
    runtime_root: str | Path,
    live_check: Callable[[], Any],
) -> bool:
    """Close an authenticated partial staging tree before the formal test gate."""

    candidate_path = Path(candidate_root)
    destination = Path(destination_root)
    runtime = _canonical_directory(runtime_root, label="runtime root")
    intent_path = _publication_intent_path(runtime)
    if not intent_path.exists() and not intent_path.is_symlink():
        return False
    candidate = _canonical_directory(
        candidate_path, label="closure publication recovery candidate"
    )
    if {entry.name for entry in candidate.iterdir()} != set(REQUIRED_FILES):
        raise BoreasV2Stage2ClosureError(
            "publication recovery candidate inventory is incomplete"
        )
    for name in REQUIRED_FILES:
        _safe_regular(candidate / name, label=f"publication recovery candidate {name}")
    manifest = _json_file(
        candidate / "boreas_v2_stage2_frozen_manifest.json",
        label="publication recovery candidate manifest",
    )
    unsigned_manifest = dict(manifest)
    manifest_root = unsigned_manifest.pop("manifest_root_sha256", None)
    if manifest_root != compact_sha256(unsigned_manifest):
        raise BoreasV2Stage2ClosureError(
            "publication recovery candidate manifest differs"
        )
    if (
        not destination.is_absolute()
        or destination.is_symlink()
        or not destination.parent.is_dir()
        or destination.parent.is_symlink()
        or destination.parent.resolve(strict=True) != destination.parent
    ):
        raise BoreasV2Stage2ClosureError(
            "publication recovery destination is unsafe"
        )
    staging = destination.parent / f".{destination.name}.{manifest_root}.staging"
    expected = _publication_intent(
        candidate=candidate,
        destination=destination,
        staging=staging,
        runtime=runtime,
        manifest_root_sha256=str(manifest_root),
    )
    if _json_file(intent_path, label="closure publication intent") != expected:
        raise BoreasV2Stage2ClosureError("closure publication intent differs")
    if destination.exists() or destination.is_symlink():
        return False  # The formal publisher authenticates and completes this state.
    live_check()
    _recover_or_fill_publication_staging(
        candidate=candidate, staging=staging, live_check=live_check
    )
    return True


def _verify_generated_documents(
    root: Path, *, runtime: Path, authority: PreparationVerificationAuthority
) -> None:
    readiness = _json_file(root / "boreas_v2_stage2_readiness.json", label="readiness")
    if readiness != _readiness():
        raise BoreasV2Stage2ClosureError("readiness exact schema/values differ")
    summary = _json_file(root / "boreas_v2_stage2_summary.json", label="summary")
    if (
        set(summary)
        != {
            "answers",
            "conclusion",
            "empirical_counts_and_bytes",
            "large_artifact_policy",
            "readiness",
            "resource_capacity_evidence",
            "schema",
        }
        or summary.get("schema") != "zprm.boreas.v2.stage2.preparation_summary.v1"
        or summary.get("readiness") != readiness
        or not isinstance(summary.get("answers"), list)
        or [row.get("item") for row in summary["answers"]] != list(range(1, 31))
        or any(
            not isinstance(row, dict)
            or set(row) != {"answer", "item", "question"}
            or not isinstance(row["question"], str)
            or not row["question"]
            for row in summary["answers"]
        )
    ):
        raise BoreasV2Stage2ClosureError("summary exact schema/30-answer closure differs")

    uncertainty = _json_file(
        root / "boreas_v2_stage2_uncertainty_budget.json", label="uncertainty"
    )
    if set(uncertainty) != {"rows"} or not isinstance(uncertainty["rows"], list):
        raise BoreasV2Stage2ClosureError("uncertainty JSON exact schema differs")
    if any(set(row) != set(UNCERTAINTY_FIELDS) for row in uncertainty["rows"]):
        raise BoreasV2Stage2ClosureError("uncertainty row exact schema differs")
    unknown = sorted(
        row["component"]
        for row in uncertainty["rows"]
        if row["value"] == row["uncertainty_type"] == row["status"] == "UNKNOWN"
    )
    if unknown != sorted(UNKNOWN_COMPONENTS):
        raise BoreasV2Stage2ClosureError("uncertainty UNKNOWN inventory differs")

    no_icp = _json_file(root / "NO_ICP_ATTESTATION.json", label="NO-ICP")
    no_icp_fields = {
        "actual_open3d_trials",
        "actual_pcl_trials",
        "actual_trials",
        "estimated_transform_count",
        "estimated_transform_evidence",
        "estimated_transform_file_count",
        "estimated_transform_files",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pass",
        "pcl_cli_invocation_count",
        "real_trial_result_count",
        "registration_execution_count",
        "status",
        "structured_result_scan_error_count",
        "structured_result_scan_error_files",
    }
    zero_fields = {
        "actual_open3d_trials",
        "actual_pcl_trials",
        "actual_trials",
        "estimated_transform_count",
        "estimated_transform_file_count",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pcl_cli_invocation_count",
        "real_trial_result_count",
        "registration_execution_count",
        "structured_result_scan_error_count",
    }
    if (
        set(no_icp) != no_icp_fields
        or no_icp.get("pass") is not True
        or no_icp.get("status") != "PASS"
        or any(no_icp.get(field) != 0 for field in zero_fields)
        or no_icp.get("estimated_transform_evidence") != []
        or no_icp.get("estimated_transform_files") != []
        or no_icp.get("structured_result_scan_error_files") != []
    ):
        raise BoreasV2Stage2ClosureError("NO-ICP exact live-attestation schema differs")

    authoritative_markdown = authority.preprocessing_contract_path.with_suffix(".md")
    if (
        _safe_regular(authoritative_markdown, label="authoritative preprocessing Markdown").read_bytes()
        != _safe_regular(
            root / "boreas_v2_stage2_preprocessing_contract.md",
            label="frozen preprocessing Markdown",
        ).read_bytes()
    ):
        raise BoreasV2Stage2ClosureError("preprocessing Markdown authority bytes differ")

    resources = summary["resource_capacity_evidence"]
    if not isinstance(resources, dict) or set(resources) != {
        "reducer",
        "runtime_reverification_dependencies",
        "target_context",
        "target_map_content_address",
    }:
        raise BoreasV2Stage2ClosureError("resource evidence exact section differs")
    context = resources["target_context"]
    if set(context) != {
        "capacity_measurement",
        "capacity_measurement_file_sha256",
        "capacity_measurement_provenance",
        "capacity_measurement_provenance_file_sha256",
        "resource_plan",
        "resource_plan_file_sha256",
    }:
        raise BoreasV2Stage2ClosureError(
            "target-context resource exact schema differs"
        )
    context_paths = {
        "capacity_measurement": runtime
        / "evidence/target_context_capacity_measurement.json",
        "capacity_measurement_provenance": runtime
        / "evidence/target_context_capacity_measurement_provenance.json",
        "resource_plan": runtime / "evidence/target_context_resource_plan.json",
    }
    for name, path in context_paths.items():
        if context.get(name) != _json_file(path, label=f"runtime target context {name}"):
            raise BoreasV2Stage2ClosureError("embedded target-context evidence changed")
        if context.get(f"{name}_file_sha256") != sha256_file(path):
            raise BoreasV2Stage2ClosureError("target-context evidence SHA differs")
    reducer = resources["reducer"]
    if set(reducer) != {
        "capacity_layout_verification",
        "capacity_layout_verification_file_sha256",
        "capacity_replay_binding",
        "capacity_replay_binding_file_sha256",
        "resource_plan",
        "resource_plan_file_sha256",
    }:
        raise BoreasV2Stage2ClosureError("reducer resource exact schema differs")
    reducer_paths = {
        "capacity_layout_verification": runtime
        / "checkpoints/reducer_capacity_layout_verification.json",
        "capacity_replay_binding": runtime
        / "checkpoints/reducer_capacity_replay_binding.json",
    }
    for name, path in reducer_paths.items():
        if reducer.get(name) != _json_file(path, label=f"runtime reducer {name}"):
            raise BoreasV2Stage2ClosureError("embedded reducer evidence changed")
        if reducer.get(f"{name}_file_sha256") != sha256_file(path):
            raise BoreasV2Stage2ClosureError("reducer evidence SHA differs")
    plan = ReducerResourcePlan.from_mapping(
        reducer.get("resource_plan", {}), production_mode=True
    )
    if plan.plan_payload_sha256 != reducer["capacity_layout_verification"].get(
        "resource_plan_sha256"
    ):
        raise BoreasV2Stage2ClosureError("embedded reducer plan binding differs")
    target_resource = resources["target_map_content_address"]
    if set(target_resource) != {
        "metadata",
        "metadata_file_sha256",
        "target_map_path",
        "target_map_sha256",
    }:
        raise BoreasV2Stage2ClosureError(
            "target-map content-address exact schema differs"
        )
    target_relative = PurePosixPath(str(target_resource.get("target_map_path", "")))
    if target_relative.is_absolute() or any(
        part in {"", ".", ".."} for part in target_relative.parts
    ):
        raise BoreasV2Stage2ClosureError("embedded target path is unsafe")
    target_path = _safe_regular(runtime.joinpath(*target_relative.parts), label="target map")
    metadata_path = _safe_regular(target_path.parent / "metadata.json", label="target metadata")
    if (
        target_resource.get("target_map_sha256") != sha256_file(target_path)
        or target_resource.get("metadata")
        != _json_file(metadata_path, label="runtime target metadata")
        or target_resource.get("metadata_file_sha256") != sha256_file(metadata_path)
    ):
        raise BoreasV2Stage2ClosureError("embedded target content-address evidence differs")
    retention = resources["runtime_reverification_dependencies"]
    if set(retention) != {
        "authenticated_replay_ledger_path",
        "authenticated_replay_ledger_sha256",
        "authenticated_replay_path",
        "authenticated_replay_size_bytes",
        "retention_policy",
    }:
        raise BoreasV2Stage2ClosureError(
            "runtime reverification dependency exact schema differs"
        )
    for prefix in ("authenticated_replay", "authenticated_replay_ledger"):
        relative = PurePosixPath(str(retention.get(f"{prefix}_path", "")))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise BoreasV2Stage2ClosureError("reverification dependency path is unsafe")
        path = _safe_regular(runtime.joinpath(*relative.parts), label=prefix)
        if prefix.endswith("ledger"):
            if retention.get(f"{prefix}_sha256") != sha256_file(path):
                raise BoreasV2Stage2ClosureError("replay ledger SHA differs")
        elif retention.get(f"{prefix}_size_bytes") != path.stat().st_size:
            raise BoreasV2Stage2ClosureError("replay retained byte size differs")
    if retention.get("retention_policy") != (
        "RETAIN_AUTHENTICATED_REPLAY_AND_LEDGER_AFTER_FREEZE_"
        "BECAUSE_FINAL_VERIFIER_REPLAYS_EXACT_TARGET_BYTES"
    ):
        raise BoreasV2Stage2ClosureError("replay retention policy differs")


def verify_and_publish_small_closure(
    *,
    candidate_root: str | Path,
    destination_root: str | Path,
    runtime_root: str | Path,
    authority: PreparationVerificationAuthority,
    live_check: Callable[[], Any],
    verifier: Callable[..., dict[str, Any]] = verify_boreas_v2_stage2_preparation,
    production_mode: bool = True,
) -> ClosurePublication:
    """Verify candidate, copy only its closed inventory, then verify destination."""

    if production_mode and verifier is not verify_boreas_v2_stage2_preparation:
        raise BoreasV2Stage2ClosureError("production verifier dependency cannot be replaced")
    candidate = _canonical_directory(candidate_root, label="closure candidate root")
    runtime = _canonical_directory(runtime_root, label="runtime root")
    destination = Path(destination_root)
    if not destination.is_absolute() or destination.is_symlink():
        raise BoreasV2Stage2ClosureError("destination root must be absolute and non-symlink")
    parent = _canonical_directory(destination.parent, label="frozen-assets parent")
    live_check()
    _verify_generated_documents(candidate, runtime=runtime, authority=authority)
    candidate_report = verifier(root=candidate, runtime_root=runtime, authority=authority)
    if candidate_report.get("BOREAS_EXTERNAL_V2_STAGE2_VERIFICATION_PASS") is not True:
        raise BoreasV2Stage2ClosureError("candidate independent verification did not pass")
    manifest_root = str(candidate_report["manifest_root_sha256"])

    staging = parent / f".{destination.name}.{manifest_root}.staging"
    intent_path = _publication_intent_path(runtime)
    expected_intent = _publication_intent(
        candidate=candidate,
        destination=destination,
        staging=staging,
        runtime=runtime,
        manifest_root_sha256=manifest_root,
    )
    if intent_path.exists() or intent_path.is_symlink():
        if _json_file(intent_path, label="closure publication intent") != expected_intent:
            raise BoreasV2Stage2ClosureError("closure publication intent differs")

    if destination.exists():
        if staging.exists() or staging.is_symlink():
            raise BoreasV2Stage2ClosureError(
                "published closure and publication staging both exist"
            )
        destination = _canonical_directory(destination, label="existing frozen closure")
        if not _same_tree(candidate, destination):
            raise BoreasV2Stage2ClosureError("existing frozen closure differs from candidate")
        live_check()
        _verify_generated_documents(destination, runtime=runtime, authority=authority)
        report = verifier(root=destination, runtime_root=runtime, authority=authority)
        _clear_publication_intent(intent_path)
        return ClosurePublication(destination, manifest_root, report, True)

    if staging.exists() and not intent_path.exists():
        raise BoreasV2Stage2ClosureError(
            "publication staging lacks its authenticated intent"
        )
    if not intent_path.exists():
        live_check()
        atomic_write_json(intent_path, expected_intent, overwrite=False)
        _fsync_directory(intent_path.parent)
    _recover_or_fill_publication_staging(
        candidate=candidate, staging=staging, live_check=live_check
    )
    live_check()
    _verify_generated_documents(staging, runtime=runtime, authority=authority)
    verifier(root=staging, runtime_root=runtime, authority=authority)
    live_check()
    os.rename(staging, destination)
    _fsync_directory(parent)
    live_check()
    _verify_generated_documents(destination, runtime=runtime, authority=authority)
    report = verifier(root=destination, runtime_root=runtime, authority=authority)
    if report.get("manifest_root_sha256") != manifest_root:
        raise BoreasV2Stage2ClosureError("published closure manifest root changed")
    _clear_publication_intent(intent_path)
    return ClosurePublication(destination, manifest_root, report, False)


__all__ = [
    "BoreasV2Stage2ClosureError",
    "ClosurePublication",
    "UNKNOWN_COMPONENTS",
    "assemble_small_closure_candidate",
    "recover_closure_publication_staging",
    "verify_and_publish_small_closure",
    "write_selection_prerequisites",
]
