"""Independent verifier for the Boreas v2 Stage-2 preparation closure.

The verifier intentionally does not import the Stage-2 producer, geometry
entry point, or selector.  Scientific constants and the weak/rich algorithms
are redeclared below.  Consequently a self-consistent producer defect, or a
semantic edit followed by rebuilding the outer SHA manifests, does not become
self-validating.

Large point arrays remain outside Git.  Their relative paths, byte sizes, and
SHA-256 identities are frozen in the small closure and verified against an
explicit runtime root.

The final target-map check independently compiles the pinned reducer source,
replays every authenticated active range, and compares the canonical target
NPY byte-for-byte in place.  It never creates a second target-sized copy.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ZERO_SHA256 = "0" * 64
SAFE_REPLAY_MARKER_SCHEMA = (
    "zprm.boreas.v2.stage2.formal_full_pytest_marker.v1"
)
SAFE_REPLAY_MARKER_PURPOSE = (
    "BOREAS_V2_STAGE2_DATA_PREPARATION_FULL_TEST_NO_REGISTRATION"
)
SAFE_REPLAY_EVENT_SCHEMA = (
    "zprm.boreas.v2.stage2.safe_fixture_replay_event.v1"
)
SAFE_REPLAY_CATALOG_SHA256 = (
    "5402064dcfa87d489541685f57723bca5fd2b8128ec13762590af00637ba5fb9"
)
OPEN3D_BACKEND = "open3d_point_to_plane"
PCL_BACKEND = "pcl_point_to_plane"
WEAK_LABEL = "CORRIDOR_OR_WEAK_GEOMETRY"
RICH_LABEL = "GEOMETRY_RICH"
LABEL_ORDER = (WEAK_LABEL, RICH_LABEL)

EXPECTED_STAGE1_MANIFEST_SHA256 = (
    "71cfd78aa588c67dd28f8f8be87b7514c20ed090e75a8433583362253a97a8be"
)
EXPECTED_STORAGE_MANIFEST_SHA256 = (
    "51ac1448f30461ceb8c0dc93c8c33e084841cd088c8321698a1cdd338cfa4a9c"
)
EXPECTED_STAGE1_ALLOWLIST_SHA256 = (
    "26ac211c854472dcb3db2f1cd5b096849bfd27867bac34e75bc8ce0d01bb2787"
)
STAGE1_S3_LS_FROZEN_DISPLAY_OFFSET_SECONDS = 8 * 60 * 60
EXPECTED_PAIR_SELECTION_SHA256 = (
    "b28a52498a2fddc1b80d43c626b75179358066677ce8f493be22afba7a19639c"
)
EXPECTED_MAP_SEQUENCE = "boreas-2021-11-14-09-47"
EXPECTED_QUERY_SEQUENCE = "boreas-2021-01-26-11-22"
EXPECTED_BACKEND_PARAMETER_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
DOWNLOAD_RECEIPT_SCHEMA = "zprm.boreas.stage2.download_receipt.v1"
EXPECTED_QUERY_REFERENCE_POSE_SHA256 = (
    "516b9e86d050e5ee09b4c048271f2372efeaaede4e176b78e3fa0ea631c490ef"
)
EXPECTED_MAP_REFERENCE_POSE_SHA256 = (
    "734d0a25bb8150fbe7eae8e5a667be926f9af8bd0f9356792d280756d2ae7f17"
)
EXPECTED_EXTRINSIC_SHA256 = (
    "e32772bcc7d617c62db0c3047e896e3626760d3f148fa83e26ad24b0c8f024db"
)
EXPECTED_TRANSFORM_CHAIN_CONSISTENCY_SHA256 = (
    "283d13bc8c276c052dbf3f933b3fadc054e596a5193a94814295c994b26eb697"
)
EXPECTED_TRANSFORM_CHAIN_MANIFEST_SHA256 = (
    "8d9e51e44795a1b960cecbb561ee9b0eeceaabf3bce8cfcde8cdbecdc60b6775"
)

PAYLOAD_FILES = frozenset(
    {
        "LIDAR_DOWNLOAD_AUDIT.json",
        "MAP_LIDAR_DOWNLOAD_AUDIT.json",
        "NO_ICP_ATTESTATION.json",
        "QUERY_LIDAR_DOWNLOAD_AUDIT.json",
        "all_candidate_intervals.csv",
        "all_candidate_scans.csv",
        "boreas_v2_stage2_download_authorization.json",
        "boreas_v2_stage2_preprocessing_contract.json",
        "boreas_v2_stage2_preprocessing_contract.md",
        "boreas_v2_stage2_readiness.json",
        "boreas_v2_stage2_selection_manifest.json",
        "boreas_v2_stage2_summary.json",
        "boreas_v2_stage2_summary.md",
        "boreas_v2_stage2_uncertainty_budget.csv",
        "boreas_v2_stage2_uncertainty_budget.json",
        "canonical_input_manifest.csv",
        "canonical_source_verification.csv",
        "download_receipts.csv",
        "excluded_candidate_scans.csv",
        "excluded_intervals.csv",
        "geometry_only_metrics.csv",
        "geometry_metric_verification.csv",
        "first_pass_selection_projection.csv",
        "map_lineage_manifest.json",
        "map_download_receipts.csv",
        "selected_scene_intervals.csv",
        "selected_snapshots.csv",
        "target_map_freeze_manifest.json",
        "target_map_reducer_verification.json",
        "boreas_v2_stage2_selection_freeze.json",
    }
)
FROZEN_MANIFEST_NAME = "boreas_v2_stage2_frozen_manifest.json"
REQUIRED_FILES = PAYLOAD_FILES | {FROZEN_MANIFEST_NAME, "SHA256SUMS"}

GEOMETRY_ONLY_FIELDS = (
    "initial_correspondence_count",
    "initial_valid_normal_correspondence_count",
    "lambda_min_trans",
    "lambda_mid_trans",
    "lambda_max_trans",
    "normalized_lambda_min_trans",
    "normalized_lambda_mid_trans",
    "normalized_lambda_max_trans",
    "condition_number_trans",
    "spectral_entropy_trans",
)
SPECTRAL_FIELDS = GEOMETRY_ONLY_FIELDS[2:]
INTERVAL_SCORE_FIELDS = (
    "normalized_lambda_min_trans",
    "condition_number_trans",
    "spectral_entropy_trans",
)

CANDIDATE_SCAN_FIELDS = (
    "query_ordinal",
    "sequence_id",
    "object_key",
    "timestamp_us",
    "remote_size_bytes",
    "last_modified",
    "etag",
    "payload_sha256",
    "interval_id",
    "window_index",
    "finite_source_point_count",
    "target_map_point_count",
    *GEOMETRY_ONLY_FIELDS,
    "reference_interpolation_valid",
    "reference_gap_s",
    "gt_gap_within_limit",
    "gt_overlap_within_5m",
    "target_map_frozen_complete",
    "target_map_coverage_valid",
    "deskew_processing_contract_valid",
    "geometry_valid",
    "exclusion_reason",
    "gt_sha256",
    "calibration_sha256",
    "preprocessing_contract_sha256",
    "selection_contract_sha256",
    "target_map_sha256",
    "candidate_scan_row_sha256",
)
GEOMETRY_METRIC_FIELDS = (
    "query_ordinal",
    "object_key",
    "timestamp_us",
    *GEOMETRY_ONLY_FIELDS,
    "geometry_valid",
    "exclusion_reason",
    "candidate_scan_row_sha256",
)
CANDIDATE_INTERVAL_FIELDS = (
    "interval_id",
    "window_index",
    "interval_index",
    "start_time_us",
    "end_time_us",
    "duration_us",
    "center_time_us",
    "center_world_x_m",
    "center_world_y_m",
    "center_world_z_m",
    "center_quaternion_x",
    "center_quaternion_y",
    "center_quaternion_z",
    "center_quaternion_w",
    "center_reference_lower_timestamp_us",
    "center_reference_upper_timestamp_us",
    "center_interpolation_method",
    "candidate_scan_count",
    "geometry_valid_scan_count",
    "geometry_invalid_scan_count",
    "geometry_valid_fraction",
    "minimum_geometry_valid_fraction",
    "interval_valid",
    "exclusion_reason",
    *INTERVAL_SCORE_FIELDS,
    "candidate_scan_rows_sha256",
    "selection_contract_sha256",
    "candidate_interval_row_sha256",
)
SELECTED_INTERVAL_FIELDS = (
    "scene_label",
    "selection_rank_within_label",
    "interval_id",
    "window_index",
    "start_time_us",
    "end_time_us",
    "center_world_x_m",
    "center_world_y_m",
    "center_world_z_m",
    *INTERVAL_SCORE_FIELDS,
    "candidate_interval_row_sha256",
    "selected_interval_row_sha256",
)
SELECTED_SNAPSHOT_FIELDS = (
    "selection_index",
    "snapshot_id",
    "scene_label",
    "interval_id",
    "interval_selection_rank",
    "selected_interval_row_sha256",
    "quantile_index",
    "quantile_probability",
    "quantile_method",
    "quantile_target_timestamp_us",
    "selected_timestamp_us",
    "absolute_quantile_delta_us",
    "object_key",
    "query_ordinal",
    "first_pass_geometry_row_sha256",
    "selection_contract_sha256",
    "selected_snapshot_row_sha256",
)
RECEIPT_FIELDS = (
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
FIRST_PASS_SCAN_FIELDS = frozenset(
    {
        "query_ordinal",
        "sequence_id",
        "object_key",
        "timestamp_us",
        "remote_size_bytes",
        "last_modified",
        "etag",
        "payload_sha256",
        "finite_source_point_count",
        "target_map_point_count",
        "reference_interpolation_valid",
        "reference_gap_s",
        "gt_overlap_within_5m",
        "target_map_frozen_complete",
        "deskew_processing_contract_valid",
        "gt_sha256",
        "calibration_sha256",
        "preprocessing_contract_sha256",
        "target_map_sha256",
        *GEOMETRY_ONLY_FIELDS,
    }
)
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

QUERY_JOURNAL_SCHEMA = "zprm.boreas.v2.stage2.query_journal.v1"
MAP_RECEIPT_JOURNAL_SCHEMA = "zprm.boreas.v2.stage2.receipt_checkpoint.v1"
QUERY_AUDIT_SCHEMA = "zprm.boreas.v2.stage2.query_download_audit.v1"
MAP_AUDIT_SCHEMA = "zprm.boreas.v2.stage2.lidar_download_audit.v2"
AGGREGATE_AUDIT_SCHEMA = "zprm.boreas.v2.stage2.lidar_download_audit.v3"
SELECTION_FREEZE_SCHEMA = "zprm.boreas.v2.stage2.selection_freeze.v1"
GEOMETRY_WITNESS_STATUS = (
    "PASS_DUAL_PATH_GEOMETRY_EXACT_SHARED_FROZEN_PRIMITIVES"
)
AUTHORIZATION_UNSIGNED_FIELDS = frozenset(
    {
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED",
        "PUBLIC_DATA_V2_RUN_AUTHORIZED",
        "REAL_REGISTRATION_AUTHORIZED",
        "STAGE2_DOWNLOAD_AUTHORIZED",
        "actual_trials",
        "allowlist_object_count",
        "allowlist_remote_bytes",
        "allowlist_sha256",
        "authorization_path",
        "authorization_scope",
        "branch",
        "bucket",
        "commit",
        "disk_free_bytes_at_authorization",
        "execution_mode",
        "extrinsic_limitation",
        "minimum_start_free_disk_bytes",
        "no_icp_guard_active",
        "no_registration_environment_active",
        "preprocessing_contract_payload_sha256",
        "preprocessing_contract_sha256",
        "primary_pair",
        "primary_pair_sha256",
        "registration_execution_count",
        "root_bindings",
        "runtime_low_disk_watermark_bytes",
        "schema_version",
        "self_hash_semantics",
        "stage1_manifest_file_sha256",
        "stage1_verification_report_path",
        "stage1_verification_report_sha256",
        "static_source_audit",
        "storage_budget_sha256",
        "storage_contract_sha256",
        "storage_manifest_file_sha256",
        "storage_verification",
        "timestamp_utc",
        "worktree_clean_before_authorization",
    }
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
MAP_LINEAGE_SOURCE_FIELDS = frozenset(
    {
        "map_ordinal",
        "object_key",
        "receipt_sha256",
        "raw_payload_sha256",
        "processing_result_sha256",
        "replay_range_sha256",
        "replay_active_point_count",
        "transformed_xyz_sha256",
        "previous_map_state_transition_sha256",
        "map_state_transition_sha256",
        "replay_state_transition_sha256",
        "replay_ledger_record_sha256",
    }
)
MAP_LINEAGE_FIELDS = frozenset(
    {
        "schema_version",
        "map_sequence_id",
        "source_object_count",
        "query_contribution_count",
        "preprocessing_contract_sha256",
        "gt_sha256",
        "extrinsic_sha256",
        "replay_array_path",
        "replay_ledger_path",
        "source_objects",
        "final_map_state_transition_sha256",
        "map_lineage_manifest_sha256",
    }
)
TARGET_REDUCER_VERIFICATION_FIELDS = frozenset(
    {
        "schema_version",
        "map_lineage_manifest_sha256",
        "replay_ledger_path",
        "replay_ledger_sha256",
        "replay_plan_sha256",
        "replay_descriptor_count",
        "replay_authenticated_ranges_sha256",
        "replay_total_bytes",
        "reducer_implementation_path",
        "reducer_implementation_sha256",
        "reducer_binary_path",
        "reducer_binary_sha256",
        "reducer_capacity_layout_verification_file_sha256",
        "reducer_capacity_replay_binding_file_sha256",
        "reducer_resource_plan_file_sha256",
        "producer_target_map_sha256",
        "reducer_output_target_map_sha256",
        "target_map_size_bytes",
        "target_point_count",
        "final_map_state_transition_sha256",
        "verification_scope",
        "verification_status",
        "target_map_reducer_verification_sha256",
    }
)
UNCERTAINTY_FIELDS = (
    "component",
    "value",
    "unit",
    "uncertainty_type",
    "status",
    "evidence",
)
REQUIRED_UNKNOWN_COMPONENTS = frozenset(
    {
        "orientation_uncertainty",
        "time_synchronization_numeric_bound",
        "extrinsic_translation_uncertainty",
        "extrinsic_rotation_uncertainty",
        "reference_interpolation_uncertainty",
        "deskew_uncertainty",
        "target_map_accumulation_uncertainty",
        "voxelization_uncertainty",
    }
)


class BoreasStage2PreparationVerificationError(RuntimeError):
    """The Stage-2 preparation closure or its scientific meaning differs."""


def _fail(message: str) -> None:
    raise BoreasStage2PreparationVerificationError(message)


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} mismatch: {actual!r} != {expected!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_npy_bytes(value: Any) -> bytes:
    """Serialize one verifier-owned float64 array in the frozen NPY format."""

    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(value, dtype="<f8"),
        version=(1, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _compact_hash(value: Any) -> str:
    return hashlib.sha256(_compact_json_bytes(value)).hexdigest()


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        _fail(f"{label} is not a lowercase SHA-256")
    return text


def _load_json(path: Path, *, canonical: bool = True) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BoreasStage2PreparationVerificationError(f"invalid JSON: {path}") from error
    if canonical and path.read_bytes() != _canonical_json_bytes(value):
        _fail(f"noncanonical JSON: {path}")
    return value


def _safe_root(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink():
        _fail(f"{label} must be an absolute non-symlink path")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate or not resolved.is_dir():
        _fail(f"{label} must be a canonical directory")
    return resolved


def _safe_runtime_file(root: Path, relative: str, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(f"unsafe {label} relative path: {relative!r}")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is absent or unsafe: {relative}")
    resolved = path.resolve(strict=True)
    if root != resolved.parent and root not in resolved.parents:
        _fail(f"{label} escapes runtime root")
    return resolved


def _csv_rows(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                _fail(f"CSV header differs: {path.name}")
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise BoreasStage2PreparationVerificationError(f"invalid CSV: {path}") from error
    if any(None in row or set(row) != set(fields) for row in rows):
        _fail(f"CSV row field set differs: {path.name}")
    return rows


def _parse_int(value: str, label: str) -> int:
    if re.fullmatch(r"0|-?[1-9][0-9]*", value) is None:
        _fail(f"{label} is not a canonical integer: {value!r}")
    return int(value)


def _parse_float(value: str, label: str, *, nullable: bool = False) -> float | None:
    if nullable and value == "":
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise BoreasStage2PreparationVerificationError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _parse_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    _fail(f"{label} is not canonical True/False")


@dataclass(frozen=True)
class PreparationVerificationAuthority:
    """Evidence outside the generated closure against which semantics are checked."""

    stage1_manifest_path: Path
    storage_manifest_path: Path
    stage1_allowlist_path: Path
    pair_selection_path: Path
    preprocessing_contract_path: Path
    reference_pose_path: Path
    expected_reference_pose_sha256: str
    map_reference_pose_path: Path
    expected_map_reference_pose_sha256: str
    extrinsic_path: Path
    expected_extrinsic_sha256: str
    transform_chain_consistency_path: Path
    transform_chain_manifest_path: Path
    canonical_witness_implementation_path: Path
    target_reducer_implementation_path: Path
    expected_map_sequence_id: str
    expected_query_sequence_id: str
    expected_candidate_scan_count: int
    expected_candidate_interval_count: int
    expected_map_object_count: int
    expected_backend_parameter_sha256: str
    selection_parameter_authority: str
    expected_t_reference_sha256_by_object_key: Mapping[str, str] | None = None

    @classmethod
    def production(cls, repository: str | Path) -> "PreparationVerificationAuthority":
        root = Path(repository).resolve(strict=True)
        return cls(
            stage1_manifest_path=root
            / "frozen_assets/public_data_external_validation_v2_boreas_stage1/frozen_manifest.json",
            storage_manifest_path=root
            / "frozen_assets/boreas_v2_stage2_storage_optimization/frozen_manifest.json",
            stage1_allowlist_path=root
            / "frozen_assets/public_data_external_validation_v2_boreas_stage1/boreas_v2_stage2_download_allowlist.csv",
            pair_selection_path=root
            / "frozen_assets/public_data_external_validation_v2_boreas_stage1/boreas_v2_pair_selection.json",
            preprocessing_contract_path=root
            / "protocols/boreas_v2_stage2_preprocessing_contract.json",
            reference_pose_path=Path.home()
            / "zero_perturbation_data/boreas_stage1_v1/stage1_payload"
            / f"{EXPECTED_QUERY_SEQUENCE}/applanix/lidar_poses.csv",
            expected_reference_pose_sha256=EXPECTED_QUERY_REFERENCE_POSE_SHA256,
            map_reference_pose_path=Path.home()
            / "zero_perturbation_data/boreas_stage1_v1/stage1_payload"
            / f"{EXPECTED_MAP_SEQUENCE}/applanix/lidar_poses.csv",
            expected_map_reference_pose_sha256=EXPECTED_MAP_REFERENCE_POSE_SHA256,
            extrinsic_path=Path.home()
            / "zero_perturbation_data/boreas_stage1_v1/stage1_payload"
            / f"{EXPECTED_QUERY_SEQUENCE}/calib/T_applanix_lidar.txt",
            expected_extrinsic_sha256=EXPECTED_EXTRINSIC_SHA256,
            transform_chain_consistency_path=root
            / "frozen_assets/real_data_boreas_stage1_v1/boreas_transform_chain_consistency.json",
            transform_chain_manifest_path=root
            / "frozen_assets/real_data_boreas_stage1_v1/boreas_transform_chain_manifest.json",
            canonical_witness_implementation_path=root
            / "src/phase_a_harness/real_data_preparation/boreas_v2_stage2_canonical_witness.py",
            target_reducer_implementation_path=root
            / "src/phase_a_harness/real_data_preparation/boreas_stage2_voxel_reduce.cpp",
            expected_map_sequence_id=EXPECTED_MAP_SEQUENCE,
            expected_query_sequence_id=EXPECTED_QUERY_SEQUENCE,
            expected_candidate_scan_count=11_859,
            expected_candidate_interval_count=246,
            expected_map_object_count=8_202,
            expected_backend_parameter_sha256=EXPECTED_BACKEND_PARAMETER_SHA256,
            selection_parameter_authority="BOREAS_V2_STAGE2_PREREGISTRATION",
        )

    @classmethod
    def from_profile(cls, path: str | Path) -> "PreparationVerificationAuthority":
        profile_path = Path(path).resolve(strict=True)
        value = _load_json(profile_path)
        base = profile_path.parent

        def authority_path(name: str) -> Path:
            candidate = Path(str(value[name]))
            if not candidate.is_absolute():
                candidate = base / candidate
            return candidate.resolve(strict=True)

        transforms = value.get("expected_t_reference_sha256_by_object_key")
        if transforms is not None and not isinstance(transforms, Mapping):
            _fail("authority T_reference mapping is not an object")
        return cls(
            stage1_manifest_path=authority_path("stage1_manifest_path"),
            storage_manifest_path=authority_path("storage_manifest_path"),
            stage1_allowlist_path=authority_path("stage1_allowlist_path"),
            pair_selection_path=authority_path("pair_selection_path"),
            preprocessing_contract_path=authority_path("preprocessing_contract_path"),
            reference_pose_path=authority_path("reference_pose_path"),
            expected_reference_pose_sha256=_require_sha(
                value["expected_reference_pose_sha256"],
                "authority reference-pose SHA",
            ),
            map_reference_pose_path=authority_path("map_reference_pose_path"),
            expected_map_reference_pose_sha256=_require_sha(
                value["expected_map_reference_pose_sha256"],
                "authority map reference-pose SHA",
            ),
            extrinsic_path=authority_path("extrinsic_path"),
            expected_extrinsic_sha256=_require_sha(
                value["expected_extrinsic_sha256"], "authority extrinsic SHA"
            ),
            transform_chain_consistency_path=authority_path(
                "transform_chain_consistency_path"
            ),
            transform_chain_manifest_path=authority_path(
                "transform_chain_manifest_path"
            ),
            canonical_witness_implementation_path=authority_path(
                "canonical_witness_implementation_path"
            ),
            target_reducer_implementation_path=authority_path(
                "target_reducer_implementation_path"
            ),
            expected_map_sequence_id=str(value["expected_map_sequence_id"]),
            expected_query_sequence_id=str(value["expected_query_sequence_id"]),
            expected_candidate_scan_count=int(value["expected_candidate_scan_count"]),
            expected_candidate_interval_count=int(value["expected_candidate_interval_count"]),
            expected_map_object_count=int(value["expected_map_object_count"]),
            expected_backend_parameter_sha256=_require_sha(
                value["expected_backend_parameter_sha256"],
                "authority backend parameter SHA",
            ),
            selection_parameter_authority=str(value["selection_parameter_authority"]),
            expected_t_reference_sha256_by_object_key=(
                None
                if transforms is None
                else {
                    str(key): _require_sha(child, f"T_reference authority {key}")
                    for key, child in transforms.items()
                }
            ),
        )


def _verify_outer_closure(root: Path) -> Mapping[str, Any]:
    entries = list(root.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        _fail("frozen closure contains a directory, symlink, or non-file entry")
    _same({entry.name for entry in entries}, set(REQUIRED_FILES), "frozen file closure")
    sums_rows: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            _fail("malformed SHA256SUMS row")
        digest, name = match.groups()
        if name in sums_rows or name == "SHA256SUMS":
            _fail("duplicate or recursive SHA256SUMS row")
        sums_rows[name] = digest
    _same(set(sums_rows), set(REQUIRED_FILES) - {"SHA256SUMS"}, "SHA256SUMS inventory")
    for name, digest in sums_rows.items():
        _same(_sha256_file(root / name), digest, f"SHA256SUMS digest {name}")

    manifest = _load_json(root / FROZEN_MANIFEST_NAME)
    if not isinstance(manifest, Mapping):
        _fail("frozen manifest is not an object")
    unsigned = dict(manifest)
    recorded_root = unsigned.pop("manifest_root_sha256", None)
    _same(recorded_root, _compact_hash(unsigned), "frozen manifest root")
    _same(
        manifest.get("schema_version"),
        "boreas_v2_stage2_preparation_frozen_manifest_v1",
        "frozen manifest schema",
    )
    payload = manifest.get("payload")
    if not isinstance(payload, list):
        _fail("frozen manifest payload is not a list")
    _same([row.get("path") for row in payload], sorted(PAYLOAD_FILES), "payload order")
    for row in payload:
        if set(row) != {"path", "sha256", "size_bytes"}:
            _fail("frozen manifest payload row schema differs")
        name = str(row["path"])
        _same(row["sha256"], _sha256_file(root / name), f"manifest SHA {name}")
        _same(row["size_bytes"], (root / name).stat().st_size, f"manifest size {name}")
    return manifest


def _authority_windows(pair: Mapping[str, Any]) -> list[dict[str, int]]:
    raw = pair.get("primary_complete_five_second_windows")
    if not isinstance(raw, list):
        _fail("authority pair lacks frozen five-second windows")
    output: list[dict[str, int]] = []
    for expected, row in enumerate(raw):
        if "start_time_us" in row:
            start = int(row["start_time_us"])
            end = int(row["end_time_us"])
            duration = int(row.get("duration_us", end - start))
        else:
            start = int(round(float(row["start_time_s"]) * 1_000_000))
            end = int(round(float(row["end_time_s"]) * 1_000_000))
            duration = int(round(float(row["duration_s"]) * 1_000_000))
        _same(int(row["window_index"]), expected, "authority window index")
        _same(duration, 5_000_000, "authority window duration")
        _same(end - start, 5_000_000, "authority window endpoints")
        output.append(
            {
                "window_index": expected,
                "interval_index": int(row["interval_index"]),
                "start_time_us": start,
                "end_time_us": end,
                "duration_us": duration,
            }
        )
    return output


def _roll(value: float) -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(value), math.sin(value)],
            [0.0, -math.sin(value), math.cos(value)],
        ],
        dtype=np.float64,
    )


def _pitch(value: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(value), 0.0, -math.sin(value)],
            [0.0, 1.0, 0.0],
            [math.sin(value), 0.0, math.cos(value)],
        ],
        dtype=np.float64,
    )


def _yaw(value: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(value), math.sin(value), 0.0],
            [-math.sin(value), math.cos(value), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _load_pose_csv(
    path: Path, expected_sha256: str, *, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _same(
        _sha256_file(path),
        expected_sha256,
        f"authority {label} reference-pose file SHA",
    )
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = tuple(next(reader, ()))
            expected_prefix = (
                "GPSTime",
                "easting",
                "northing",
                "altitude",
                "vel_east",
                "vel_north",
                "vel_up",
                "roll",
                "pitch",
                "heading",
            )
            if header[: len(expected_prefix)] != expected_prefix:
                _fail(f"authority {label} reference-pose header differs")
            values = np.asarray(
                [[float(value) for value in row] for row in reader if row],
                dtype=np.float64,
            )
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        raise BoreasStage2PreparationVerificationError(
            f"authority {label} reference-pose CSV is invalid"
        ) from error
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 10
        or not np.all(np.isfinite(values))
    ):
        _fail(f"authority {label} reference-pose matrix differs")
    timestamps = np.rint(values[:, 0]).astype(np.int64)
    if not np.all(values[:, 0] == timestamps) or not np.all(np.diff(timestamps) > 0):
        _fail(
            f"authority {label} reference-pose timestamps are not exact increasing microseconds"
        )
    translations = np.ascontiguousarray(values[:, 1:4], dtype=np.float64)
    rotations = np.stack(
        [_roll(row[7]) @ _pitch(row[8]) @ _yaw(row[9]) for row in values]
    )
    quaternions = Rotation.from_matrix(rotations).as_quat()
    return timestamps, translations, np.ascontiguousarray(quaternions)


def _load_reference_poses(
    authority: PreparationVerificationAuthority,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _load_pose_csv(
        authority.reference_pose_path,
        authority.expected_reference_pose_sha256,
        label="query",
    )


def _load_map_reference_positions(
    authority: PreparationVerificationAuthority,
) -> np.ndarray:
    _, translations, _ = _load_pose_csv(
        authority.map_reference_pose_path,
        authority.expected_map_reference_pose_sha256,
        label="map",
    )
    return translations


def _independent_midpoint_pose(
    timestamp_us: int,
    timestamps: np.ndarray,
    translations: np.ndarray,
    quaternions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    exact = int(np.searchsorted(timestamps, timestamp_us, side="left"))
    if exact < timestamps.size and int(timestamps[exact]) == timestamp_us:
        return (
            translations[exact].copy(),
            quaternions[exact].copy(),
            timestamp_us,
            timestamp_us,
        )
    upper = int(np.searchsorted(timestamps, timestamp_us, side="right"))
    if upper == 0 or upper >= timestamps.size:
        _fail("candidate midpoint lies outside frozen reference poses")
    lower = upper - 1
    lower_time = int(timestamps[lower])
    upper_time = int(timestamps[upper])
    if (upper_time - lower_time) > 200_000:
        _fail("candidate midpoint reference-pose gap exceeds 0.2 s")
    weight = (timestamp_us - lower_time) / float(upper_time - lower_time)
    translation = (
        (1.0 - weight) * translations[lower] + weight * translations[upper]
    )
    quaternion = Slerp(
        np.asarray([lower_time, upper_time], dtype=np.float64),
        Rotation.from_quat(quaternions[[lower, upper]]),
    )([float(timestamp_us)]).as_quat()[0]
    return translation, quaternion, lower_time, upper_time


def _independent_exact_pose(
    timestamp_us: int,
    timestamps: np.ndarray,
    translations: np.ndarray,
    quaternions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    index = int(np.searchsorted(timestamps, timestamp_us, side="left"))
    if index >= timestamps.size or int(timestamps[index]) != timestamp_us:
        _fail("selected/candidate timestamp lacks an exact official lidar_poses row")
    return translations[index].copy(), quaternions[index].copy()


def _load_allowlist(authority: PreparationVerificationAuthority) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with authority.stage1_allowlist_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        required = {"selection_role", "sequence_id", "key", "timestamp_us", "last_modified", "size_bytes"}
        if not required <= set(fields):
            _fail("authority allowlist field set is incomplete")
        rows = [dict(row) for row in reader]
    if _sha256_file(authority.stage1_allowlist_path) == EXPECTED_STAGE1_ALLOWLIST_SHA256:
        # The immutable production allowlist recorded the Asia/Shanghai clock
        # emitted by ``aws s3 ls`` and appended ``Z``.  Runtime receipts retain
        # S3's true UTC value; independently apply the one audited 8-hour
        # display correction only for the exact protected allowlist bytes.
        for row in rows:
            try:
                frozen_display = datetime.fromisoformat(
                    row["last_modified"].replace("Z", "+00:00")
                )
            except (KeyError, ValueError) as exc:
                _fail(f"production allowlist LastModified is invalid: {exc}")
            expected_remote = frozen_display - timedelta(
                seconds=STAGE1_S3_LS_FROZEN_DISPLAY_OFFSET_SECONDS
            )
            row["last_modified"] = expected_remote.isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
    map_rows = [row for row in rows if row["selection_role"] == "TARGET_MAP"]
    query_rows = [row for row in rows if row["selection_role"] == "QUERY"]
    _same(len(map_rows), authority.expected_map_object_count, "authority map object count")
    _same(len(query_rows), authority.expected_candidate_scan_count, "authority query object count")
    _same(
        {row["sequence_id"] for row in map_rows},
        {authority.expected_map_sequence_id},
        "authority map sequence",
    )
    _same(
        {row["sequence_id"] for row in query_rows},
        {authority.expected_query_sequence_id},
        "authority query sequence",
    )
    if {row["key"] for row in map_rows} & {row["key"] for row in query_rows}:
        _fail("authority map/query allowlists overlap")
    return map_rows, query_rows


def _expected_selection_contract(authority: PreparationVerificationAuthority) -> dict[str, Any]:
    return {
        "expected_candidate_interval_count": authority.expected_candidate_interval_count,
        "expected_candidate_scan_count": authority.expected_candidate_scan_count,
        "geometry_metric_parameter_binding": {
            "association_distance_m": 0.5,
            "target_normal_pca_k": 50,
            "target_normal_pca_min_neighbors": 10,
        },
        "gt_overlap_radius_m": 5.0,
        "interval_aggregation": "MEDIAN_OF_GEOMETRY_VALID_SCANS_ONLY",
        "interval_boundary": "FROZEN_HALF_OPEN_START_INCLUSIVE_END_EXCLUSIVE",
        "interval_count_per_label": 10,
        "interval_duration_us": 5_000_000,
        "maximum_reference_gap_s": 0.2,
        "minimum_center_position_separation_m": 1.0,
        "minimum_center_time_separation_s": 10.0,
        "minimum_finite_source_point_count": 1000,
        "minimum_interval_valid_denominator": 5,
        "minimum_interval_valid_numerator": 4,
        "minimum_target_map_point_count": 10000,
        "minimum_valid_normal_correspondence_count": 100,
        "minimum_valid_normal_correspondence_fraction": 0.05,
        "parameter_authority": authority.selection_parameter_authority,
        "proximity_rejection_logic": "TIME_LT_THRESHOLD_AND_3D_DISTANCE_LT_THRESHOLD",
        "quantile_method": "linear",
        "quantile_probabilities": [0.1, 0.3, 0.5, 0.7, 0.9],
        "schema": "boreas_v2_stage2_blind_selection_v1",
        "snapshots_per_interval": 5,
        "weak_first": True,
    }


def _verify_authority_bindings(
    root: Path, authority: PreparationVerificationAuthority, manifest: Mapping[str, Any]
) -> tuple[
    Mapping[str, Any],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, int]],
    str,
    str,
    str,
]:
    stage1_sha = _sha256_file(authority.stage1_manifest_path)
    storage_sha = _sha256_file(authority.storage_manifest_path)
    allowlist_sha = _sha256_file(authority.stage1_allowlist_path)
    pair_sha = _sha256_file(authority.pair_selection_path)
    preprocessing_sha = _sha256_file(authority.preprocessing_contract_path)
    reference_pose_sha = _sha256_file(authority.reference_pose_path)
    map_reference_pose_sha = _sha256_file(authority.map_reference_pose_path)
    extrinsic_sha = _sha256_file(authority.extrinsic_path)
    chain_consistency_sha = _sha256_file(authority.transform_chain_consistency_path)
    chain_manifest_sha = _sha256_file(authority.transform_chain_manifest_path)
    witness_sha = _sha256_file(authority.canonical_witness_implementation_path)
    if authority.selection_parameter_authority == "BOREAS_V2_STAGE2_PREREGISTRATION":
        _same(stage1_sha, EXPECTED_STAGE1_MANIFEST_SHA256, "production Stage-1 manifest")
        _same(
            storage_sha,
            EXPECTED_STORAGE_MANIFEST_SHA256,
            "production storage-optimization manifest",
        )
        _same(allowlist_sha, EXPECTED_STAGE1_ALLOWLIST_SHA256, "production allowlist")
        _same(pair_sha, EXPECTED_PAIR_SELECTION_SHA256, "production pair selection")
        _same(
            reference_pose_sha,
            EXPECTED_QUERY_REFERENCE_POSE_SHA256,
            "production query reference-pose CSV",
        )
        _same(
            map_reference_pose_sha,
            EXPECTED_MAP_REFERENCE_POSE_SHA256,
            "production map reference-pose CSV",
        )
        _same(extrinsic_sha, EXPECTED_EXTRINSIC_SHA256, "production extrinsic")
        _same(
            chain_consistency_sha,
            EXPECTED_TRANSFORM_CHAIN_CONSISTENCY_SHA256,
            "production transform-chain consistency",
        )
        _same(
            chain_manifest_sha,
            EXPECTED_TRANSFORM_CHAIN_MANIFEST_SHA256,
            "production transform-chain manifest",
        )
    _same(
        reference_pose_sha,
        authority.expected_reference_pose_sha256,
        "authority reference-pose SHA",
    )
    _same(
        map_reference_pose_sha,
        authority.expected_map_reference_pose_sha256,
        "authority map reference-pose SHA",
    )
    _same(extrinsic_sha, authority.expected_extrinsic_sha256, "authority extrinsic SHA")
    chain = _load_json(authority.transform_chain_manifest_path)
    expected_chain = {
        "composition": "T_ENU_lidar(t)=T_ENU_applanix(t)@T_applanix_lidar",
        "consistency_sha256": chain_consistency_sha,
        "direction_verified_against_official_lidar_poses": True,
        "inverse_used": False,
        "source_sha256": extrinsic_sha,
        "status": "PASS",
    }
    for field, expected in expected_chain.items():
        _same(chain.get(field), expected, f"transform-chain authority {field}")
    preprocessing = _load_json(authority.preprocessing_contract_path)
    _same(
        preprocessing.get("primary_pair"),
        {
            "map_lidar_pose_sha256": authority.expected_map_reference_pose_sha256,
            "map_sequence_id": authority.expected_map_sequence_id,
            "query_lidar_pose_sha256": authority.expected_reference_pose_sha256,
            "query_sequence_id": authority.expected_query_sequence_id,
            "static_t_applanix_lidar_sha256": extrinsic_sha,
        },
        "preprocessing PRIMARY pair bindings",
    )
    witness_binding = preprocessing.get("implementation_bindings", {}).get(
        "independent_canonical_source_witness", {}
    )
    _same(
        witness_binding.get("file_sha256"),
        witness_sha,
        "preprocessing canonical-witness implementation SHA",
    )
    _same(
        (root / "boreas_v2_stage2_preprocessing_contract.json").read_bytes(),
        authority.preprocessing_contract_path.read_bytes(),
        "preprocessing contract authoritative bytes",
    )
    pair = _load_json(authority.pair_selection_path)
    primary = pair.get("PRIMARY_PAIR")
    if not isinstance(primary, Mapping):
        _fail("authority pair selection lacks PRIMARY_PAIR")
    expected_pair = {
        "map_sequence_id": authority.expected_map_sequence_id,
        "query_sequence_id": authority.expected_query_sequence_id,
    }
    _same(
        {key: primary.get(key) for key in expected_pair}, expected_pair, "authority PRIMARY_PAIR"
    )
    map_rows, query_rows = _load_allowlist(authority)
    windows = _authority_windows(pair)
    _same(len(windows), authority.expected_candidate_interval_count, "authority window count")

    selection_manifest = _load_json(root / "boreas_v2_stage2_selection_manifest.json")
    unsigned = dict(selection_manifest)
    claimed = unsigned.pop("selection_manifest_sha256", None)
    _same(claimed, _hash_json(unsigned), "selection manifest self-hash")
    bindings = selection_manifest.get("authority_bindings")
    expected_bindings = {
        "stage1_manifest_sha256": stage1_sha,
        "storage_manifest_sha256": storage_sha,
        "stage1_allowlist_sha256": allowlist_sha,
        "pair_selection_sha256": pair_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "primary_pair": expected_pair,
    }
    _same(bindings, expected_bindings, "selection authority bindings")
    for key, expected in (
        ("stage1_manifest_sha256", stage1_sha),
        ("storage_manifest_sha256", storage_sha),
        ("stage1_allowlist_sha256", allowlist_sha),
        ("pair_selection_sha256", pair_sha),
        ("preprocessing_contract_sha256", preprocessing_sha),
    ):
        _same(manifest.get(key), expected, f"frozen manifest {key}")
    return (
        selection_manifest,
        map_rows,
        query_rows,
        windows,
        preprocessing_sha,
        extrinsic_sha,
        witness_sha,
    )


def _parse_candidate_scans(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    raw = _csv_rows(root / "all_candidate_scans.csv", CANDIDATE_SCAN_FIELDS)
    rows: list[dict[str, Any]] = []
    integer_fields = {
        "query_ordinal",
        "timestamp_us",
        "remote_size_bytes",
        "window_index",
        "finite_source_point_count",
        "target_map_point_count",
        "initial_correspondence_count",
        "initial_valid_normal_correspondence_count",
    }
    boolean_fields = {
        "reference_interpolation_valid",
        "gt_gap_within_limit",
        "gt_overlap_within_5m",
        "target_map_frozen_complete",
        "target_map_coverage_valid",
        "deskew_processing_contract_valid",
        "geometry_valid",
    }
    nullable_floats = set(SPECTRAL_FIELDS)
    ordinary_floats = {"reference_gap_s"}
    for source in raw:
        row: dict[str, Any] = {}
        for field, value in source.items():
            if field in integer_fields:
                row[field] = _parse_int(value, f"candidate {field}")
            elif field in boolean_fields:
                row[field] = _parse_bool(value, f"candidate {field}")
            elif field in nullable_floats:
                row[field] = _parse_float(value, f"candidate {field}", nullable=True)
            elif field in ordinary_floats:
                row[field] = _parse_float(value, f"candidate {field}")
            else:
                row[field] = value
        core = {key: value for key, value in row.items() if key != "candidate_scan_row_sha256"}
        _same(row["candidate_scan_row_sha256"], _hash_json(core), "candidate scan row SHA")
        rows.append(row)
    return rows, raw


def _candidate_validity(row: Mapping[str, Any]) -> tuple[bool, str]:
    source_count = int(row["finite_source_point_count"])
    if source_count < 1000:
        return False, "FINITE_SOURCE_POINT_COUNT_LT_1000"
    if int(row["target_map_point_count"]) < 10000:
        return False, "TARGET_MAP_POINT_COUNT_LT_10000"
    if int(row["initial_valid_normal_correspondence_count"]) < max(
        100, math.ceil(0.05 * source_count)
    ):
        return False, "INSUFFICIENT_VALID_NORMAL_CORRESPONDENCES"
    for field, reason in (
        ("reference_interpolation_valid", "REFERENCE_INTERPOLATION_INVALID"),
        ("gt_gap_within_limit", "GT_GAP_EXCEEDS_0_2_S"),
        ("target_map_coverage_valid", "TARGET_MAP_COVERAGE_INVALID"),
        ("deskew_processing_contract_valid", "DESKEW_PROCESSING_CONTRACT_INVALID"),
    ):
        if row[field] is not True:
            return False, reason
    if not all(row[field] is not None for field in SPECTRAL_FIELDS):
        return False, "GEOMETRY_SPECTRUM_NOT_COMPUTABLE"
    return True, ""


def _verify_geometry_spectrum(row: Mapping[str, Any]) -> None:
    values = [row[field] for field in SPECTRAL_FIELDS]
    if all(value is None for value in values):
        _same(
            int(row["initial_valid_normal_correspondence_count"]),
            0,
            "null spectrum valid-normal count",
        )
        return
    if not all(value is not None and math.isfinite(float(value)) for value in values):
        _fail("candidate spectrum mixes null/nonfinite values")
    eigenvalues = np.asarray(
        [row["lambda_min_trans"], row["lambda_mid_trans"], row["lambda_max_trans"]],
        dtype=np.float64,
    )
    normalized = np.asarray(
        [
            row["normalized_lambda_min_trans"],
            row["normalized_lambda_mid_trans"],
            row["normalized_lambda_max_trans"],
        ],
        dtype=np.float64,
    )
    if np.any(eigenvalues < -1.0e-12) or np.any(np.diff(eigenvalues) < -1.0e-12):
        _fail("translation geometry eigenvalues are negative or unordered")
    # H=N^T N/n for unit normals has trace one.  Recompute every derived field
    # so a re-signed selector score cannot be invented independently of H.
    if not np.isclose(np.sum(eigenvalues), 1.0, rtol=0.0, atol=1.0e-10):
        _fail("translation geometry eigenvalue trace differs from one")
    expected_normalized = eigenvalues / max(float(np.sum(eigenvalues)), 1.0e-12)
    if not np.allclose(normalized, expected_normalized, rtol=0.0, atol=1.0e-12):
        _fail("normalized translation eigenvalues disagree")
    expected_condition = float(
        eigenvalues[2] / max(float(eigenvalues[0]), 1.0e-12)
    )
    if not math.isclose(
        float(row["condition_number_trans"]),
        expected_condition,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        _fail("translation geometry condition number disagrees")
    positive = normalized > 0.0
    expected_entropy = -float(
        np.sum(normalized[positive] * np.log(normalized[positive]))
    ) / math.log(3.0)
    if not math.isclose(
        float(row["spectral_entropy_trans"]),
        expected_entropy,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        _fail("translation geometry spectral entropy disagrees")


def _verify_candidate_scans(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    raw: Sequence[Mapping[str, str]],
    query_allowlist: Sequence[Mapping[str, str]],
    windows: Sequence[Mapping[str, int]],
    *,
    authority: PreparationVerificationAuthority,
    preprocessing_sha: str,
    selection_contract_sha: str,
    target_map_sha: str,
    gt_sha: str,
    calibration_sha: str,
    reference_poses: tuple[np.ndarray, np.ndarray, np.ndarray],
    map_reference_positions: np.ndarray,
) -> None:
    _same(len(rows), authority.expected_candidate_scan_count, "candidate scan count")
    map_reference_tree = cKDTree(map_reference_positions, copy_data=False)
    _same(
        [row["query_ordinal"] for row in rows], list(range(len(rows))), "candidate ordinals"
    )
    _same(
        [row["object_key"] for row in rows],
        [row["key"] for row in query_allowlist],
        "all query allowlist scans retained",
    )
    for row, allow in zip(rows, query_allowlist):
        _same(row["sequence_id"], authority.expected_query_sequence_id, "candidate sequence")
        for field, expected in (
            ("timestamp_us", int(allow["timestamp_us"])),
            ("remote_size_bytes", int(allow["size_bytes"])),
            ("last_modified", allow["last_modified"]),
            ("preprocessing_contract_sha256", preprocessing_sha),
            ("selection_contract_sha256", selection_contract_sha),
            ("target_map_sha256", target_map_sha),
            ("gt_sha256", gt_sha),
            ("calibration_sha256", calibration_sha),
        ):
            _same(row[field], expected, f"candidate binding {field}")
        matching = [
            window
            for window in windows
            if window["start_time_us"] <= row["timestamp_us"] < window["end_time_us"]
        ]
        _same(len(matching), 1, "candidate frozen-window membership")
        expected_window = matching[0]
        _same(row["window_index"], expected_window["window_index"], "candidate window index")
        _same(
            row["interval_id"],
            f"boreas-v2-window-{expected_window['window_index']:03d}",
            "candidate interval ID",
        )
        _same(
            row["gt_gap_within_limit"], row["reference_gap_s"] <= 0.2, "GT gap gate"
        )
        _same(
            row["target_map_coverage_valid"],
            row["gt_overlap_within_5m"] and row["target_map_frozen_complete"],
            "target-map coverage evidence",
        )
        query_position, _ = _independent_exact_pose(
            int(row["timestamp_us"]), *reference_poses
        )
        nearest_map_distance_m, _ = map_reference_tree.query(
            query_position, k=1, workers=1
        )
        if not math.isfinite(float(nearest_map_distance_m)):
            _fail("candidate independent map-reference distance is nonfinite")
        _same(
            row["gt_overlap_within_5m"],
            float(nearest_map_distance_m) <= 5.0,
            "candidate independently recomputed 5 m GT overlap",
        )
        _same(
            row["reference_interpolation_valid"],
            True,
            "candidate independent reference interpolation gate",
        )
        _same(float(row["reference_gap_s"]), 0.0, "candidate exact-pose join gap")
        if row["initial_valid_normal_correspondence_count"] > row["initial_correspondence_count"]:
            _fail("valid-normal correspondence count exceeds initial count")
        if row["initial_correspondence_count"] > row["finite_source_point_count"]:
            _fail("initial correspondence count exceeds finite source count")
        _verify_geometry_spectrum(row)
        expected_valid, expected_reason = _candidate_validity(row)
        _same(row["geometry_valid"], expected_valid, "candidate geometry validity")
        _same(row["exclusion_reason"], expected_reason, "candidate exclusion reason")

    excluded = _csv_rows(root / "excluded_candidate_scans.csv", CANDIDATE_SCAN_FIELDS)
    _same(excluded, [row for row in raw if row["geometry_valid"] == "False"], "excluded scan projection")
    metrics = _csv_rows(root / "geometry_only_metrics.csv", GEOMETRY_METRIC_FIELDS)
    _same(
        metrics,
        [{field: row[field] for field in GEOMETRY_METRIC_FIELDS} for row in raw],
        "geometry-only metric projection",
    )


def _parse_candidate_intervals(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    raw = _csv_rows(root / "all_candidate_intervals.csv", CANDIDATE_INTERVAL_FIELDS)
    rows: list[dict[str, Any]] = []
    integer_fields = {
        "window_index",
        "interval_index",
        "start_time_us",
        "end_time_us",
        "duration_us",
        "center_time_us",
        "center_reference_lower_timestamp_us",
        "center_reference_upper_timestamp_us",
        "candidate_scan_count",
        "geometry_valid_scan_count",
        "geometry_invalid_scan_count",
    }
    float_fields = {
        "center_world_x_m",
        "center_world_y_m",
        "center_world_z_m",
        "center_quaternion_x",
        "center_quaternion_y",
        "center_quaternion_z",
        "center_quaternion_w",
        "geometry_valid_fraction",
        "minimum_geometry_valid_fraction",
    }
    for source in raw:
        values: dict[str, Any] = {}
        for field, value in source.items():
            if field in integer_fields:
                values[field] = _parse_int(value, f"interval {field}")
            elif field in float_fields:
                values[field] = _parse_float(value, f"interval {field}")
            elif field in INTERVAL_SCORE_FIELDS:
                values[field] = _parse_float(value, f"interval {field}", nullable=True)
            elif field == "interval_valid":
                values[field] = _parse_bool(value, "interval valid")
            else:
                values[field] = value
        row = {
            "interval_id": values["interval_id"],
            "window_index": values["window_index"],
            "interval_index": values["interval_index"],
            "start_time_us": values["start_time_us"],
            "end_time_us": values["end_time_us"],
            "duration_us": values["duration_us"],
            "center_time_us": values["center_time_us"],
            "center_world_position": [
                values["center_world_x_m"],
                values["center_world_y_m"],
                values["center_world_z_m"],
            ],
            "center_world_quaternion_xyzw": [
                values["center_quaternion_x"],
                values["center_quaternion_y"],
                values["center_quaternion_z"],
                values["center_quaternion_w"],
            ],
            "center_reference_lower_timestamp_us": values[
                "center_reference_lower_timestamp_us"
            ],
            "center_reference_upper_timestamp_us": values[
                "center_reference_upper_timestamp_us"
            ],
            "center_interpolation_method": values["center_interpolation_method"],
            "candidate_scan_count": values["candidate_scan_count"],
            "geometry_valid_scan_count": values["geometry_valid_scan_count"],
            "geometry_invalid_scan_count": values["geometry_invalid_scan_count"],
            "geometry_valid_fraction": values["geometry_valid_fraction"],
            "minimum_geometry_valid_fraction": values["minimum_geometry_valid_fraction"],
            "interval_valid": values["interval_valid"],
            "exclusion_reason": values["exclusion_reason"],
            **{field: values[field] for field in INTERVAL_SCORE_FIELDS},
            "candidate_scan_rows_sha256": values["candidate_scan_rows_sha256"],
            "selection_contract_sha256": values["selection_contract_sha256"],
            "candidate_interval_row_sha256": values["candidate_interval_row_sha256"],
        }
        core = {key: value for key, value in row.items() if key != "candidate_interval_row_sha256"}
        _same(row["candidate_interval_row_sha256"], _hash_json(core), "candidate interval row SHA")
        rows.append(row)
    return rows, raw


def _verify_candidate_intervals(
    root: Path,
    intervals: Sequence[Mapping[str, Any]],
    raw: Sequence[Mapping[str, str]],
    scans: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, int]],
    *,
    selection_contract_sha: str,
    reference_poses: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _same(len(intervals), len(windows), "candidate interval count")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in scans:
        grouped.setdefault(str(row["interval_id"]), []).append(row)
    for interval, window in zip(intervals, windows):
        interval_id = f"boreas-v2-window-{window['window_index']:03d}"
        for field, expected in (
            ("interval_id", interval_id),
            ("window_index", window["window_index"]),
            ("interval_index", window["interval_index"]),
            ("start_time_us", window["start_time_us"]),
            ("end_time_us", window["end_time_us"]),
            ("duration_us", 5_000_000),
            ("center_time_us", (window["start_time_us"] + window["end_time_us"]) // 2),
            ("center_interpolation_method", "LINEAR_TRANSLATION_SLERP_XYZW"),
            ("selection_contract_sha256", selection_contract_sha),
        ):
            _same(interval[field], expected, f"candidate interval {field}")
        quaternion = np.asarray(interval["center_world_quaternion_xyzw"], dtype=np.float64)
        if not np.isclose(np.linalg.norm(quaternion), 1.0, rtol=0.0, atol=1e-10):
            _fail("interval midpoint quaternion is not unit")
        expected_position, expected_quaternion, lower, upper = _independent_midpoint_pose(
            int(interval["center_time_us"]), *reference_poses
        )
        if not np.allclose(
            interval["center_world_position"],
            expected_position,
            rtol=0.0,
            atol=1.0e-10,
        ):
            _fail("interval midpoint 3D translation differs from frozen LiDAR poses")
        # q and -q encode the same rotation; compare the rotation matrices.
        if not np.allclose(
            Rotation.from_quat(quaternion).as_matrix(),
            Rotation.from_quat(expected_quaternion).as_matrix(),
            rtol=0.0,
            atol=1.0e-10,
        ):
            _fail("interval midpoint SLERP rotation differs from frozen LiDAR poses")
        _same(
            interval["center_reference_lower_timestamp_us"],
            lower,
            "interval midpoint lower reference timestamp",
        )
        _same(
            interval["center_reference_upper_timestamp_us"],
            upper,
            "interval midpoint upper reference timestamp",
        )
        source_rows = grouped.get(interval_id, [])
        valid_rows = [row for row in source_rows if row["geometry_valid"] is True]
        total = len(source_rows)
        valid = len(valid_rows)
        expected_valid = 5 * valid >= 4 * total
        for field, expected in (
            ("candidate_scan_count", total),
            ("geometry_valid_scan_count", valid),
            ("geometry_invalid_scan_count", total - valid),
            ("geometry_valid_fraction", valid / total),
            ("minimum_geometry_valid_fraction", 0.8),
            ("interval_valid", expected_valid),
            ("exclusion_reason", "" if expected_valid else "GEOMETRY_VALID_FRACTION_LT_0_80"),
            (
                "candidate_scan_rows_sha256",
                _hash_json([row["candidate_scan_row_sha256"] for row in source_rows]),
            ),
        ):
            _same(interval[field], expected, f"candidate interval aggregate {field}")
        for field in INTERVAL_SCORE_FIELDS:
            expected_score = (
                float(np.median([float(row[field]) for row in valid_rows]))
                if expected_valid
                else None
            )
            _same(interval[field], expected_score, f"valid-only interval median {field}")
    excluded = _csv_rows(root / "excluded_intervals.csv", CANDIDATE_INTERVAL_FIELDS)
    _same(excluded, [row for row in raw if row["interval_valid"] == "False"], "excluded interval projection")


def _rank_key(row: Mapping[str, Any], label: str) -> tuple[Any, ...]:
    primary = float(row["normalized_lambda_min_trans"])
    condition = float(row["condition_number_trans"])
    entropy = float(row["spectral_entropy_trans"])
    start = int(row["start_time_us"])
    if label == WEAK_LABEL:
        return (primary, -condition, entropy, start, str(row["interval_id"]))
    return (-primary, condition, -entropy, start, str(row["interval_id"]))


def _compatible(candidate: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> bool:
    start = int(candidate["start_time_us"])
    end = int(candidate["end_time_us"])
    center = (start + end) / 2_000_000.0
    position = np.asarray(candidate["center_world_position"], dtype=np.float64)
    for prior in selected:
        prior_start = int(prior["start_time_us"])
        prior_end = int(prior["end_time_us"])
        if max(start, prior_start) < min(end, prior_end):
            return False
        prior_center = (prior_start + prior_end) / 2_000_000.0
        prior_position = np.asarray(prior["center_world_position"], dtype=np.float64)
        if abs(center - prior_center) < 10.0 and np.linalg.norm(position - prior_position) < 1.0:
            return False
    return True


def _independent_selected_intervals(
    intervals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    eligible = [dict(row) for row in intervals if row["interval_valid"] is True]
    selected: list[dict[str, Any]] = []
    occupied: list[dict[str, Any]] = []
    for label in LABEL_ORDER:
        chosen: list[dict[str, Any]] = []
        for row in sorted(eligible, key=lambda value: _rank_key(value, label)):
            if _compatible(row, occupied + chosen):
                core = {
                    **row,
                    "scene_label": label,
                    "selection_rank_within_label": len(chosen) + 1,
                }
                chosen.append({**core, "selected_interval_row_sha256": _hash_json(core)})
            if len(chosen) == 10:
                break
        if len(chosen) != 10:
            _fail(f"STAGE2_SELECTION_FAIL: independently found {len(chosen)} {label}")
        selected.extend(chosen)
        occupied.extend(chosen)
    return selected


def _parse_selected_intervals(root: Path) -> list[dict[str, Any]]:
    raw = _csv_rows(root / "selected_scene_intervals.csv", SELECTED_INTERVAL_FIELDS)
    rows: list[dict[str, Any]] = []
    for source in raw:
        rows.append(
            {
                "scene_label": source["scene_label"],
                "selection_rank_within_label": _parse_int(
                    source["selection_rank_within_label"], "selected interval rank"
                ),
                "interval_id": source["interval_id"],
                "window_index": _parse_int(source["window_index"], "selected window"),
                "start_time_us": _parse_int(source["start_time_us"], "selected start"),
                "end_time_us": _parse_int(source["end_time_us"], "selected end"),
                "center_world_position": [
                    _parse_float(source["center_world_x_m"], "selected center x"),
                    _parse_float(source["center_world_y_m"], "selected center y"),
                    _parse_float(source["center_world_z_m"], "selected center z"),
                ],
                **{
                    field: _parse_float(source[field], f"selected {field}")
                    for field in INTERVAL_SCORE_FIELDS
                },
                "candidate_interval_row_sha256": source["candidate_interval_row_sha256"],
                "selected_interval_row_sha256": source["selected_interval_row_sha256"],
            }
        )
    return rows


def _verify_selected_intervals(
    actual: Sequence[Mapping[str, Any]], intervals: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    expected = _independent_selected_intervals(intervals)
    _same(len(actual), 20, "selected interval count")
    projection = []
    for row in expected:
        projection.append(
            {
                "scene_label": row["scene_label"],
                "selection_rank_within_label": row["selection_rank_within_label"],
                "interval_id": row["interval_id"],
                "window_index": row["window_index"],
                "start_time_us": row["start_time_us"],
                "end_time_us": row["end_time_us"],
                "center_world_position": row["center_world_position"],
                **{field: row[field] for field in INTERVAL_SCORE_FIELDS},
                "candidate_interval_row_sha256": row["candidate_interval_row_sha256"],
                "selected_interval_row_sha256": row["selected_interval_row_sha256"],
            }
        )
    _same(list(actual), projection, "independently recomputed weak/rich intervals")
    return expected


def _parse_selected_snapshots(root: Path) -> list[dict[str, Any]]:
    raw = _csv_rows(root / "selected_snapshots.csv", SELECTED_SNAPSHOT_FIELDS)
    rows: list[dict[str, Any]] = []
    ints = {
        "selection_index",
        "interval_selection_rank",
        "quantile_index",
        "selected_timestamp_us",
        "query_ordinal",
    }
    floats = {
        "quantile_probability",
        "quantile_target_timestamp_us",
        "absolute_quantile_delta_us",
    }
    for source in raw:
        row: dict[str, Any] = {}
        for field, value in source.items():
            if field in ints:
                row[field] = _parse_int(value, f"snapshot {field}")
            elif field in floats:
                row[field] = _parse_float(value, f"snapshot {field}")
            else:
                row[field] = value
        core = {key: value for key, value in row.items() if key != "selected_snapshot_row_sha256"}
        _same(row["selected_snapshot_row_sha256"], _hash_json(core), "selected snapshot row SHA")
        rows.append(row)
    return rows


def _independent_snapshots(
    scans: Sequence[Mapping[str, Any]], selected_intervals: Sequence[Mapping[str, Any]], selection_sha: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in scans:
        if row["geometry_valid"] is True:
            grouped.setdefault(str(row["interval_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for interval in selected_intervals:
        rows = sorted(grouped[str(interval["interval_id"])], key=lambda row: row["timestamp_us"])
        timestamps = [int(row["timestamp_us"]) for row in rows]
        array = np.asarray(timestamps, dtype=np.float64)
        chosen: set[int] = set()
        label = str(interval["scene_label"])
        prefix = "weak" if label == WEAK_LABEL else "rich"
        rank = int(interval["selection_rank_within_label"])
        by_time = {row["timestamp_us"]: row for row in rows}
        for quantile_index, probability in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
            target = float(np.quantile(array, probability, method="linear"))
            available = [timestamp for timestamp in timestamps if timestamp not in chosen]
            timestamp = min(available, key=lambda value: (abs(float(value) - target), value))
            chosen.add(timestamp)
            source = by_time[timestamp]
            core = {
                "selection_index": len(output),
                "snapshot_id": f"boreas-v2-{prefix}-{rank:02d}-q{int(round(probability * 100)):02d}",
                "scene_label": label,
                "interval_id": interval["interval_id"],
                "interval_selection_rank": rank,
                "selected_interval_row_sha256": interval["selected_interval_row_sha256"],
                "quantile_index": quantile_index,
                "quantile_probability": probability,
                "quantile_method": "linear",
                "quantile_target_timestamp_us": target,
                "selected_timestamp_us": timestamp,
                "absolute_quantile_delta_us": abs(float(timestamp) - target),
                "object_key": source["object_key"],
                "query_ordinal": source["query_ordinal"],
                "first_pass_geometry_row_sha256": source["candidate_scan_row_sha256"],
                "selection_contract_sha256": selection_sha,
            }
            output.append({**core, "selected_snapshot_row_sha256": _hash_json(core)})
    return output


def _verify_selected_snapshots(
    actual: Sequence[Mapping[str, Any]],
    scans: Sequence[Mapping[str, Any]],
    selected_intervals: Sequence[Mapping[str, Any]],
    selection_sha: str,
) -> None:
    expected = _independent_snapshots(scans, selected_intervals, selection_sha)
    _same(list(actual), expected, "independently recomputed 10/30/50/70/90 snapshots")
    _same(len(actual), 100, "snapshot count")
    _same(len({row["object_key"] for row in actual}), 100, "unique snapshot object count")
    counts = Counter(row["scene_label"] for row in actual)
    _same(counts, Counter({WEAK_LABEL: 50, RICH_LABEL: 50}), "weak/rich snapshot counts")


def _canonical_npy(
    path: Path, *, shape_tail: tuple[int, ...]
) -> tuple[np.memmap, str, int]:
    """Validate canonical NPY v1.0 without loading its payload into memory."""

    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise BoreasStage2PreparationVerificationError(
            f"invalid canonical NPY: {path}"
        ) from error
    if not isinstance(value, np.memmap):
        _fail(f"canonical NPY is not memory-mappable: {path}")
    if value.ndim != len(shape_tail) + 1 or value.shape[1:] != shape_tail:
        _fail(f"canonical NPY shape differs: {path}")
    if value.dtype != np.dtype("<f8") or not value.flags.c_contiguous:
        _fail(f"canonical NPY dtype/layout contract differs: {path}")

    header_stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header_stream, np.lib.format.header_data_from_array_1_0(value)
    )
    expected_header = header_stream.getvalue()
    expected_size = len(expected_header) + int(value.nbytes)
    if path.stat().st_size != expected_size:
        _fail(f"canonical NPY file size differs: {path}")
    with path.open("rb") as stream:
        if stream.read(len(expected_header)) != expected_header:
            _fail(f"canonical NPY v1.0 header differs: {path}")

    # Avoid a target-sized temporary boolean allocation.  A row-chunked scan
    # touches the mmap sequentially while bounding resident scratch memory.
    row_chunk = max(1, (8 * 1024 * 1024) // max(1, value.shape[1] * value.dtype.itemsize))
    for start in range(0, value.shape[0], row_chunk):
        if not np.all(np.isfinite(value[start : start + row_chunk])):
            _fail(f"canonical NPY contains nonfinite values: {path}")
    return value, _sha256_file(path), expected_size


def _compact_json_line(value: Mapping[str, Any]) -> bytes:
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


def _independent_replay_ledger(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        _fail("production replay ledger is empty or truncated")
    rows: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    expected_envelope_fields = {
        "payload",
        "previous_record_sha256",
        "record_sha256",
        "record_type",
        "schema",
        "sequence_number",
    }
    for sequence_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            envelope = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise BoreasStage2PreparationVerificationError(
                "production replay ledger contains invalid JSONL"
            ) from error
        if not isinstance(envelope, dict) or set(envelope) != expected_envelope_fields:
            _fail("production replay ledger envelope schema differs")
        if _compact_json_line(envelope) != line:
            _fail("production replay ledger JSONL is noncanonical")
        _same(
            envelope["schema"],
            "zprm.stage2.production_map_replay_ledger.v2",
            "replay ledger schema",
        )
        _same(envelope["sequence_number"], sequence_number, "replay ledger sequence")
        _same(
            envelope["previous_record_sha256"], previous, "replay ledger record chain"
        )
        unsigned = dict(envelope)
        claimed = unsigned.pop("record_sha256")
        actual = hashlib.sha256(_compact_json_line(unsigned)).hexdigest()
        _same(claimed, actual, "replay ledger record SHA")
        if not isinstance(envelope["payload"], Mapping):
            _fail("replay ledger payload is not an object")
        rows.append(envelope)
        previous = actual
    if rows[0]["record_type"] != "PLAN" or any(
        row["record_type"] != "SCAN" for row in rows[1:]
    ):
        _fail("replay ledger record ordering differs")
    return rows


def _verify_replay_range(
    replay_path: Path,
    *,
    byte_start: int,
    byte_end: int,
    active_point_count: int,
) -> tuple[str, str]:
    if byte_start < 0 or byte_end <= byte_start or active_point_count < 0:
        _fail("replay range descriptor is invalid")
    active_bytes_remaining = active_point_count * 24
    if active_bytes_remaining > byte_end - byte_start:
        _fail("replay active prefix exceeds fixed range")
    range_digest = hashlib.sha256()
    active_digest = hashlib.sha256()
    remaining = byte_end - byte_start
    with replay_path.open("rb") as stream:
        stream.seek(byte_start)
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                _fail("replay array has a short authenticated range")
            range_digest.update(block)
            active_count = min(len(block), active_bytes_remaining)
            if active_count:
                active_digest.update(block[:active_count])
                active_bytes_remaining -= active_count
            if any(block[active_count:]):
                _fail("replay inactive padding contains nonzero bytes")
            remaining -= len(block)
    if active_bytes_remaining:
        _fail("replay active prefix is truncated")
    return range_digest.hexdigest(), active_digest.hexdigest()


def _live_available_memory_bytes() -> int:
    """Read Linux's immediately usable memory authority, fail closed."""

    try:
        row = next(
            line
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
            if line.startswith("MemAvailable:")
        )
        fields = row.split()
        if len(fields) != 3 or fields[2] != "kB":
            _fail("MemAvailable schema differs")
        value = int(fields[1]) * 1024
    except (OSError, StopIteration, UnicodeError, ValueError) as error:
        raise BoreasStage2PreparationVerificationError(
            "cannot establish live MemAvailable for target replay"
        ) from error
    if value <= 0:
        _fail("live MemAvailable is not positive")
    return value


def _independent_reducer_capacity(
    value: Mapping[str, Any], *, max_voxels: int
) -> dict[str, int | str]:
    """Recompute the pinned C++ ABI/layout envelope without trusting its total."""

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
    _same(set(value), fields, "compiled reducer capacity exact schema")
    if max_voxels <= 0:
        _fail("target replay max_voxels is not positive")
    constructor_capacity = 1024
    while constructor_capacity * 7 // 10 < max_voxels and constructor_capacity < 2**20:
        constructor_capacity *= 2
    table_capacity = 1024
    while table_capacity * 7 // 10 < max_voxels:
        table_capacity *= 2
    # These sizes are part of the pinned production ABI, not self-reported
    # tunables.  A compiler/architecture change therefore fails closed.
    slot_bytes = 56
    size_t_bytes = 8
    key_bytes = 12
    fixed = 64 * 1024 * 1024
    input_buffer = (1 << 18) * 24
    output_buffer = 4096 * 24
    final_table = table_capacity * slot_bytes
    largest_old = table_capacity // 2 if table_capacity > constructor_capacity else 0
    growth_table = (table_capacity + largest_old) * slot_bytes
    sorted_indices = max_voxels * size_t_bytes
    target_npy = max_voxels * 24 + 4096
    growth_peak = growth_table + input_buffer + fixed
    output_peak = (
        final_table
        + sorted_indices
        + input_buffer
        + output_buffer
        + target_npy
        + fixed
    )
    expected: dict[str, int | str] = {
        "constructor_capacity": constructor_capacity,
        "final_table_bytes": final_table,
        "fixed_overhead_bytes": fixed,
        "input_buffer_bytes": input_buffer,
        "key_bytes": key_bytes,
        "max_voxels": max_voxels,
        "maximum_load_denominator": 10,
        "maximum_load_numerator": 7,
        "output_buffer_bytes": output_buffer,
        "peak_growth_table_bytes": growth_table,
        "schema": "zprm-boreas-stage2-reducer-capacity-layout-v1",
        "size_t_bytes": size_t_bytes,
        "slot_bytes": slot_bytes,
        "sorted_index_upper_bytes": sorted_indices,
        "table_capacity": table_capacity,
        "target_npy_upper_bytes": target_npy,
        "total_peak_upper_bound_bytes": max(growth_peak, output_peak),
    }
    _same(dict(value), expected, "compiled reducer capacity independent formula")
    return expected


def _target_voxel_parameters(authority: PreparationVerificationAuthority) -> tuple[float, tuple[float, float, float]]:
    contract = _load_json(authority.preprocessing_contract_path)
    accumulation = contract.get("map_accumulation")
    if isinstance(accumulation, Mapping):
        size_value = accumulation.get("target_voxel_size_m")
        origin_value = accumulation.get("target_voxel_origin_enu_ref_m")
    else:
        # Synthetic authority profile used only by the verifier tests.
        size_value = contract.get("target_map_voxel_size_m")
        origin_value = contract.get("target_voxel_origin_enu_ref_m", [0.0, 0.0, 0.0])
    try:
        size = float(size_value)
        origin = tuple(float(child) for child in origin_value)
    except (TypeError, ValueError) as error:
        raise BoreasStage2PreparationVerificationError(
            "target voxel authority is not numeric"
        ) from error
    if not math.isfinite(size) or size <= 0.0:
        _fail("target voxel size authority is invalid")
    if len(origin) != 3 or not all(math.isfinite(child) for child in origin):
        _fail("target voxel origin authority is invalid")
    return size, (origin[0], origin[1], origin[2])


def _verify_reducer_resource_evidence(
    *,
    runtime: Path,
    reducer: Mapping[str, Any],
    replay_plan_sha256: str,
    reducer_source_sha256: str,
    reducer_binary_sha256: str,
    reducer_binary_size_bytes: int,
    production: bool,
) -> dict[str, Any]:
    """Independently replay the exact plan/layout/replay-binding semantics."""

    plan_path = _safe_runtime_file(
        runtime,
        "checkpoints/reducer_resource_plan.json",
        "reducer resource plan",
    )
    layout_path = _safe_runtime_file(
        runtime,
        "checkpoints/reducer_capacity_layout_verification.json",
        "reducer capacity-layout verification",
    )
    binding_path = _safe_runtime_file(
        runtime,
        "checkpoints/reducer_capacity_replay_binding.json",
        "reducer capacity replay binding",
    )
    for field, path in (
        ("reducer_resource_plan_file_sha256", plan_path),
        ("reducer_capacity_layout_verification_file_sha256", layout_path),
        ("reducer_capacity_replay_binding_file_sha256", binding_path),
    ):
        _same(reducer[field], _sha256_file(path), f"target reducer {field}")

    plan = _load_json(plan_path)
    plan_fields = {
        "capacity_probe_evidence_sha256",
        "capacity_probe_kind",
        "estimated_peak_memory_bytes",
        "max_voxels",
        "minimum_live_available_memory_bytes",
        "plan_payload_sha256",
        "production_approved",
        "reducer_binary_projected_bytes",
        "reducer_binary_sha256",
        "reducer_source_sha256",
        "safety_margin_bytes",
        "schema",
    }
    _same(set(plan), plan_fields, "reducer resource plan exact schema")
    unsigned_plan = dict(plan)
    plan_claim = unsigned_plan.pop("plan_payload_sha256")
    _same(plan_claim, _hash_json(unsigned_plan), "reducer resource plan self-hash")
    max_voxels = plan.get("max_voxels")
    if isinstance(max_voxels, bool) or not isinstance(max_voxels, int):
        _fail("reducer resource plan max_voxels is invalid")
    if production:
        _same(max_voxels, 90_000_000, "production reducer max_voxels")
    estimated = plan.get("estimated_peak_memory_bytes")
    margin = plan.get("safety_margin_bytes")
    minimum = plan.get("minimum_live_available_memory_bytes")
    for field, value in (
        ("estimated_peak_memory_bytes", estimated),
        ("safety_margin_bytes", margin),
        ("minimum_live_available_memory_bytes", minimum),
        ("reducer_binary_projected_bytes", plan.get("reducer_binary_projected_bytes")),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _fail(f"reducer resource plan {field} is not a positive integer")
    _same(minimum, estimated + margin, "reducer resource plan memory threshold")
    if production and margin < 5 * 1024**3:
        _fail("production reducer safety margin is below 5 GiB")
    expected_plan_values = {
        "capacity_probe_kind": "COMPILED_LAYOUT_EXACT_UPPER_BOUND_V1",
        "max_voxels": max_voxels,
        "production_approved": True,
        "reducer_binary_projected_bytes": reducer_binary_size_bytes,
        "reducer_binary_sha256": reducer_binary_sha256,
        "reducer_source_sha256": reducer_source_sha256,
        "schema": "zprm.boreas.v2.stage2.reducer_resource_plan.v1",
    }
    for field, expected in expected_plan_values.items():
        _same(plan.get(field), expected, f"reducer resource plan {field}")

    layout_evidence = _load_json(layout_path)
    layout_fields = {
        "binary_path",
        "binary_sha256",
        "capacity_layout",
        "capacity_layout_sha256",
        "minimum_live_available_memory_bytes",
        "resource_plan_sha256",
        "schema",
        "source_path",
        "source_sha256",
        "verification_payload_sha256",
        "verification_status",
    }
    _same(set(layout_evidence), layout_fields, "reducer layout evidence exact schema")
    unsigned_layout = dict(layout_evidence)
    layout_claim = unsigned_layout.pop("verification_payload_sha256")
    _same(layout_claim, _hash_json(unsigned_layout), "reducer layout evidence self-hash")
    raw_layout = layout_evidence.get("capacity_layout")
    if not isinstance(raw_layout, Mapping):
        _fail("reducer capacity layout is not an object")
    capacity = _independent_reducer_capacity(raw_layout, max_voxels=max_voxels)
    capacity_sha = _hash_json(capacity)
    expected_layout_values = {
        "binary_path": reducer["reducer_binary_path"],
        "binary_sha256": reducer_binary_sha256,
        "capacity_layout_sha256": capacity_sha,
        "minimum_live_available_memory_bytes": minimum,
        "resource_plan_sha256": plan_claim,
        "schema": "zprm.boreas.v2.stage2.reducer_capacity_layout_verification.v1",
        "source_path": reducer["reducer_implementation_path"],
        "source_sha256": reducer_source_sha256,
        "verification_status": "PASS_COMPILED_LAYOUT_AND_LIVE_MEMORY_GATE",
    }
    for field, expected in expected_layout_values.items():
        _same(layout_evidence.get(field), expected, f"reducer layout evidence {field}")
    _same(
        plan.get("capacity_probe_evidence_sha256"),
        capacity_sha,
        "resource-plan/layout capacity SHA",
    )
    _same(estimated, capacity["total_peak_upper_bound_bytes"], "resource-plan capacity")
    binary_label = PurePosixPath(str(layout_evidence.get("binary_path")))
    source_label = PurePosixPath(str(layout_evidence.get("source_path")))
    if (
        binary_label.is_absolute()
        or source_label.is_absolute()
        or ".." in binary_label.parts
        or ".." in source_label.parts
        or not binary_label.parts
        or not source_label.parts
    ):
        _fail("reducer layout evidence path label is unsafe")
    live_available = _live_available_memory_bytes()
    if live_available < minimum:
        _fail(
            "live memory is below frozen production reducer gate: "
            f"available={live_available}, required={minimum}"
        )

    binding = _load_json(binding_path)
    binding_fields = {
        "binding_payload_sha256",
        "capacity_layout_verification_file_sha256",
        "replay_plan_sha256",
        "resource_plan_sha256",
        "schema",
        "verification_status",
    }
    _same(set(binding), binding_fields, "reducer replay binding exact schema")
    unsigned_binding = dict(binding)
    binding_claim = unsigned_binding.pop("binding_payload_sha256")
    _same(binding_claim, _hash_json(unsigned_binding), "reducer replay binding self-hash")
    expected_binding = {
        "capacity_layout_verification_file_sha256": _sha256_file(layout_path),
        "replay_plan_sha256": replay_plan_sha256,
        "resource_plan_sha256": plan_claim,
        "schema": "zprm.boreas.v2.stage2.reducer_capacity_replay_binding.v1",
        "verification_status": "PASS_CAPACITY_LAYOUT_BOUND_TO_RECONCILED_REPLAY",
    }
    for field, expected in expected_binding.items():
        _same(binding.get(field), expected, f"reducer replay binding {field}")
    return {
        "capacity_layout_sha256": capacity_sha,
        "capacity_replay_binding_payload_sha256": binding_claim,
        "live_available_memory_bytes": live_available,
        "minimum_live_available_memory_bytes": minimum,
        "resource_plan_payload_sha256": plan_claim,
    }


def _verify_target_replay_exact(
    *,
    replay_path: Path,
    target_path: Path,
    target_sha256: str,
    target_size_bytes: int,
    target_point_count: int,
    descriptors: Sequence[Mapping[str, Any]],
    lineage_rows: Sequence[Mapping[str, Any]],
    reducer: Mapping[str, Any],
    authority: PreparationVerificationAuthority,
) -> dict[str, Any]:
    """Compile a clean verifier and stream exact replay output against target."""

    if len(descriptors) != len(lineage_rows) or not descriptors:
        _fail("target replay descriptor projection is incomplete")
    production = (
        authority.expected_map_sequence_id == EXPECTED_MAP_SEQUENCE
        and authority.expected_query_sequence_id == EXPECTED_QUERY_SEQUENCE
        and authority.selection_parameter_authority
        == "BOREAS_V2_STAGE2_PREREGISTRATION"
    )
    if production and not (0 < target_point_count <= 90_000_000):
        _fail("production target point count exceeds the frozen 90M voxel bound")
    verification_max_voxels = 90_000_000 if production else target_point_count
    source = authority.target_reducer_implementation_path
    if source.is_symlink() or not source.is_file() or source.resolve(strict=True) != source:
        _fail("target reducer authority source is not canonical")
    source_sha = _sha256_file(source)
    _same(
        reducer["reducer_implementation_sha256"],
        source_sha,
        "target replay pinned source SHA",
    )
    compiler_text = shutil.which("g++")
    if compiler_text is None:
        _fail("independent target replay requires g++")
    compiler = Path(compiler_text).resolve(strict=True)
    if not compiler.is_file() or compiler.stat().st_mode & 0o111 == 0:
        _fail("independent target replay compiler is unsafe")
    compiler_sha = _sha256_file(compiler)
    compiler_version_process = subprocess.run(
        [str(compiler), "--version"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if compiler_version_process.returncode != 0 or not compiler_version_process.stdout:
        _fail("independent target replay compiler version probe failed")
    compiler_version = compiler_version_process.stdout.splitlines()[0]

    descriptor_payload = "byte_offset\tpoint_count\tsha256\n" + "".join(
        f"{int(descriptor['byte_start'])}\t"
        f"{int(lineage['replay_active_point_count'])}\t"
        f"{lineage['transformed_xyz_sha256']}\n"
        for descriptor, lineage in zip(descriptors, lineage_rows)
    )
    try:
        descriptor_bytes = descriptor_payload.encode("ascii")
    except UnicodeEncodeError as error:
        raise BoreasStage2PreparationVerificationError(
            "target replay descriptor is not ASCII"
        ) from error
    descriptor_sha = hashlib.sha256(descriptor_bytes).hexdigest()
    voxel_size, origin = _target_voxel_parameters(authority)
    with tempfile.TemporaryDirectory(prefix="zprm-target-replay-verifier-") as temporary:
        temporary_root = Path(temporary).resolve(strict=True)
        descriptor_path = temporary_root / "authenticated-ranges.tsv"
        with descriptor_path.open("xb") as stream:
            stream.write(descriptor_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        binary_path = temporary_root / "voxel-replay-verifier"
        compile_command = [
            str(compiler),
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-Wall",
            "-Wextra",
            str(source),
            "-lcrypto",
            "-o",
            str(binary_path),
        ]
        compiled = subprocess.run(
            compile_command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if compiled.returncode != 0:
            _fail("independent target replay reducer compilation failed")
        os.chmod(binary_path, 0o500)
        binary_sha = _sha256_file(binary_path)
        _same(
            binary_sha,
            reducer["reducer_binary_sha256"],
            "independently compiled/producer reducer binary SHA",
        )

        capacity_command = [
            str(binary_path),
            "--print-capacity-layout",
            "--max-voxels",
            str(verification_max_voxels),
        ]
        capacity_process = subprocess.run(
            capacity_command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if capacity_process.returncode != 0:
            _fail("target replay capacity probe failed")
        try:
            capacity_raw = json.loads(capacity_process.stdout)
        except json.JSONDecodeError as error:
            raise BoreasStage2PreparationVerificationError(
                "target replay capacity probe is not JSON"
            ) from error
        if not isinstance(capacity_raw, Mapping):
            _fail("target replay capacity probe is not an object")
        _same(
            capacity_process.stdout,
            _compact_json_line(capacity_raw).decode("utf-8"),
            "target replay capacity canonical JSON",
        )
        capacity = _independent_reducer_capacity(
            capacity_raw, max_voxels=verification_max_voxels
        )
        estimated = int(capacity["total_peak_upper_bound_bytes"])
        safety_margin = 5 * 1024**3 if production else 64 * 1024**2
        available = _live_available_memory_bytes()
        required = estimated + safety_margin
        if available < required:
            _fail(
                "insufficient live memory for independently bounded target replay: "
                f"available={available}, required={required}"
            )

        command = [
            str(binary_path),
            "--replay",
            str(replay_path),
            "--descriptor",
            str(descriptor_path),
            "--descriptor-sha256",
            descriptor_sha,
            "--voxel-size-m",
            repr(voxel_size),
            "--origin-x-m",
            repr(origin[0]),
            "--origin-y-m",
            repr(origin[1]),
            "--origin-z-m",
            repr(origin[2]),
            "--max-voxels",
            str(verification_max_voxels),
            "--verify-target-npy",
            str(target_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            _fail(
                "independent target replay byte comparison failed: "
                + completed.stderr[-1000:]
            )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise BoreasStage2PreparationVerificationError(
                "independent target replay output is not JSON"
            ) from error
        report_fields = {
            "compared_npy_bytes",
            "descriptor_sha256",
            "replay_range_count",
            "schema",
            "status",
            "target_npy_sha256",
            "target_point_count",
        }
        if not isinstance(report, Mapping) or set(report) != report_fields:
            _fail("independent target replay report exact schema differs")
        _same(
            completed.stdout,
            _compact_json_line(report).decode("utf-8"),
            "independent target replay canonical JSON",
        )
        expected_report = {
            "compared_npy_bytes": target_size_bytes,
            "descriptor_sha256": descriptor_sha,
            "replay_range_count": len(descriptors),
            "schema": "zprm.boreas.stage2.target_replay_verification.v1",
            "status": "PASS_EXACT_TARGET_NPY_REPLAY",
            "target_npy_sha256": target_sha256,
            "target_point_count": target_point_count,
        }
        _same(dict(report), expected_report, "independent target replay report")
    return {
        "binary_sha256": binary_sha,
        "compared_npy_bytes": target_size_bytes,
        "compiler_path": str(compiler),
        "compiler_sha256": compiler_sha,
        "compiler_version_first_line": compiler_version,
        "estimated_peak_memory_bytes": estimated,
        "live_available_memory_bytes": available,
        "memory_safety_margin_bytes": safety_margin,
        "max_voxels": verification_max_voxels,
        "range_descriptor_sha256": descriptor_sha,
        "replay_range_count": len(descriptors),
        "source_sha256": source_sha,
        "verification_scope": (
            "INDEPENDENT_PINNED_SOURCE_COMPILE_AUTHENTICATED_REPLAY_"
            "EXACT_CANONICAL_TARGET_NPY_BYTES_NO_SECOND_TARGET_COPY"
        ),
        "verification_status": "PASS_EXACT_TARGET_NPY_REPLAY",
    }


def _verify_map(
    root: Path,
    runtime: Path,
    map_allowlist: Sequence[Mapping[str, str]],
    query_allowlist: Sequence[Mapping[str, str]],
    preprocessing_sha: str,
    receipts: Mapping[tuple[str, str], Mapping[str, str]],
    authority: PreparationVerificationAuthority,
    *,
    extrinsic_sha: str,
) -> tuple[str, str, int, int, Mapping[str, Any]]:
    lineage = _load_json(root / "map_lineage_manifest.json")
    _same(set(lineage), MAP_LINEAGE_FIELDS, "map lineage exact schema")
    lineage_unsigned = dict(lineage)
    claimed_lineage_sha = lineage_unsigned.pop("map_lineage_manifest_sha256")
    _same(claimed_lineage_sha, _hash_json(lineage_unsigned), "map lineage self-hash")
    for field, expected in (
        ("schema_version", "boreas_v2_stage2_map_lineage_v1"),
        ("map_sequence_id", map_allowlist[0]["sequence_id"]),
        ("source_object_count", len(map_allowlist)),
        ("query_contribution_count", 0),
        ("preprocessing_contract_sha256", preprocessing_sha),
        ("gt_sha256", authority.expected_map_reference_pose_sha256),
        ("extrinsic_sha256", extrinsic_sha),
    ):
        _same(lineage.get(field), expected, f"map lineage {field}")
    source_objects = lineage.get("source_objects")
    if not isinstance(source_objects, list) or any(
        not isinstance(row, Mapping) or set(row) != MAP_LINEAGE_SOURCE_FIELDS
        for row in source_objects
    ):
        _fail("map lineage source_objects is not a list")
    if {row.get("object_key") for row in source_objects} & {
        row["key"] for row in query_allowlist
    }:
        _fail("query object contaminates map lineage")

    freeze = _load_json(root / "target_map_freeze_manifest.json")
    freeze_fields = {
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
    _same(set(freeze), freeze_fields, "target-map freeze exact schema")
    for field, expected in (
        ("schema_version", "boreas_v2_stage2_target_map_freeze_v1"),
        ("map_lineage_manifest_sha256", _sha256_file(root / "map_lineage_manifest.json")),
        ("preprocessing_contract_sha256", preprocessing_sha),
        (
            "target_map_reducer_verification_sha256",
            _sha256_file(root / "target_map_reducer_verification.json"),
        ),
        ("unique_target_map_count", 1),
        ("physical_target_map_copy_count", 1),
        ("query_contribution_count", 0),
        ("target_map_immutable", True),
    ):
        _same(freeze.get(field), expected, f"target freeze {field}")

    replay_path = _safe_runtime_file(
        runtime, str(lineage["replay_array_path"]), "production map replay array"
    )
    ledger_path = _safe_runtime_file(
        runtime, str(lineage["replay_ledger_path"]), "production map replay ledger"
    )
    ledger = _independent_replay_ledger(ledger_path)
    plan = dict(ledger[0]["payload"])
    expected_plan_fields = {
        "allowlist",
        "allowlist_sha256",
        "boreas_raw_point_stride_bytes",
        "calibration_sha256",
        "gt_sha256",
        "numeric_format",
        "processing_contract_sha256",
        "padding_rule",
        "replay_path",
        "replay_point_stride_bytes",
        "track_python_voxel_state",
        "total_point_capacity",
        "total_replay_bytes",
        "voxel_rule_sha256",
    }
    _same(set(plan), expected_plan_fields, "replay plan exact schema")
    expected_descriptors: list[dict[str, Any]] = []
    byte_start = 0
    for ordinal, allow in enumerate(map_allowlist):
        size = int(allow["size_bytes"])
        if size <= 0 or size % 24:
            _fail("map allowlist size is not a positive multiple of 24")
        capacity = size // 24
        expected_descriptors.append(
            {
                "etag": receipts[("MAP_INGEST", allow["key"])]["etag"],
                "last_modified": allow["last_modified"],
                "object_key": allow["key"],
                "ordinal": ordinal,
                "remote_size_bytes": size,
                "byte_end_exclusive": byte_start + size,
                "byte_start": byte_start,
                "point_capacity": capacity,
            }
        )
        byte_start += size
    expected_plan = {
        "allowlist": expected_descriptors,
        "allowlist_sha256": _hash_json(expected_descriptors),
        "boreas_raw_point_stride_bytes": 24,
        "calibration_sha256": extrinsic_sha,
        "gt_sha256": authority.expected_map_reference_pose_sha256,
        "numeric_format": "LITTLE_ENDIAN_FLOAT64_XYZ_C_ORDER_NO_HEADER",
        "processing_contract_sha256": preprocessing_sha,
        "padding_rule": "ZERO_FLOAT64_XYZ_UNUSED_SUFFIX_V1",
        "replay_path": str(replay_path),
        "replay_point_stride_bytes": 24,
        "track_python_voxel_state": False,
        "total_point_capacity": sum(row["point_capacity"] for row in expected_descriptors),
        "total_replay_bytes": byte_start,
        "voxel_rule_sha256": freeze["voxel_rule_sha256"],
    }
    _same(plan, expected_plan, "replay plan/frozen descriptors")
    _same(replay_path.stat().st_size, byte_start, "replay array fixed byte size")
    _same(len(ledger) - 1, len(map_allowlist), "replay completed map object count")

    scan_payload_fields = {
        "byte_end_exclusive",
        "byte_start",
        "calibration_sha256",
        "completed_at_utc",
        "etag",
        "gt_sha256",
        "last_modified",
        "local_temporary_sha256",
        "map_state_transition_sha256",
        "map_state_transition_kind",
        "object_key",
        "ordinal",
        "padding_point_count",
        "padding_rule",
        "point_count",
        "previous_map_state_transition_sha256",
        "previous_replay_state_transition_sha256",
        "processing_contract_sha256",
        "remote_size_bytes",
        "replay_range_sha256",
        "replay_state_transition_sha256",
        "transformed_xyz_sha256",
    }
    expected_lineage_rows: list[dict[str, Any]] = []
    previous_map_transition = ZERO_SHA256
    previous_replay_transition = ZERO_SHA256
    for ordinal, (envelope, descriptor, allow) in enumerate(
        zip(ledger[1:], expected_descriptors, map_allowlist)
    ):
        payload = dict(envelope["payload"])
        _same(set(payload), scan_payload_fields, "replay scan exact schema")
        receipt = receipts[("MAP_INGEST", allow["key"])]
        for field, expected in (
            ("ordinal", ordinal),
            ("object_key", allow["key"]),
            ("remote_size_bytes", int(allow["size_bytes"])),
            ("etag", receipt["etag"]),
            ("last_modified", allow["last_modified"]),
            ("byte_start", descriptor["byte_start"]),
            ("byte_end_exclusive", descriptor["byte_end_exclusive"]),
            ("calibration_sha256", extrinsic_sha),
            ("gt_sha256", authority.expected_map_reference_pose_sha256),
            ("processing_contract_sha256", preprocessing_sha),
            ("local_temporary_sha256", receipt["local_temporary_sha256"]),
            ("padding_rule", "ZERO_FLOAT64_XYZ_UNUSED_SUFFIX_V1"),
            (
                "map_state_transition_kind",
                "AUTHENTICATED_REPLAY_RANGE_CHAIN_V1",
            ),
            ("previous_map_state_transition_sha256", previous_map_transition),
            (
                "previous_replay_state_transition_sha256",
                previous_replay_transition,
            ),
        ):
            _same(payload[field], expected, f"replay scan {field}")
        point_count = payload["point_count"]
        if isinstance(point_count, bool) or not isinstance(point_count, int) or not (
            0 <= point_count <= descriptor["point_capacity"]
        ):
            _fail("replay active point count is invalid")
        _same(
            payload["padding_point_count"],
            descriptor["point_capacity"] - point_count,
            "replay padding point count",
        )
        range_sha, active_sha = _verify_replay_range(
            replay_path,
            byte_start=descriptor["byte_start"],
            byte_end=descriptor["byte_end_exclusive"],
            active_point_count=point_count,
        )
        _same(payload["replay_range_sha256"], range_sha, "replay range SHA")
        _same(payload["transformed_xyz_sha256"], active_sha, "replay active XYZ SHA")
        _same(receipt["processing_result_sha256"], range_sha, "map receipt processing result")
        try:
            completed = datetime.fromisoformat(
                str(payload["completed_at_utc"]).replace("Z", "+00:00")
            )
        except ValueError as error:
            raise BoreasStage2PreparationVerificationError(
                "replay completion timestamp is invalid"
            ) from error
        if completed.tzinfo is None or completed.utcoffset() != timezone.utc.utcoffset(completed):
            _fail("replay completion timestamp is not UTC")
        map_core = {
            "byte_end_exclusive": payload["byte_end_exclusive"],
            "byte_start": payload["byte_start"],
            "calibration_sha256": payload["calibration_sha256"],
            "etag": payload["etag"],
            "gt_sha256": payload["gt_sha256"],
            "last_modified": payload["last_modified"],
            "local_temporary_sha256": payload["local_temporary_sha256"],
            "map_state_transition_kind": payload["map_state_transition_kind"],
            "object_key": payload["object_key"],
            "ordinal": payload["ordinal"],
            "padding_point_count": payload["padding_point_count"],
            "padding_rule": payload["padding_rule"],
            "point_count": payload["point_count"],
            "previous_map_state_transition_sha256": payload[
                "previous_map_state_transition_sha256"
            ],
            "processing_contract_sha256": payload["processing_contract_sha256"],
            "remote_size_bytes": payload["remote_size_bytes"],
            "replay_range_sha256": payload["replay_range_sha256"],
            "transformed_xyz_sha256": payload["transformed_xyz_sha256"],
        }
        _same(
            payload["map_state_transition_sha256"],
            _hash_json(map_core),
            "independent map-state transition SHA",
        )
        replay_core = dict(payload)
        replay_claim = replay_core.pop("replay_state_transition_sha256")
        _same(replay_claim, _hash_json(replay_core), "replay-state transition SHA")
        expected_lineage_rows.append(
            {
                "map_ordinal": ordinal,
                "object_key": allow["key"],
                "receipt_sha256": receipt["receipt_sha256"],
                "raw_payload_sha256": receipt["local_temporary_sha256"],
                "processing_result_sha256": receipt["processing_result_sha256"],
                "replay_range_sha256": range_sha,
                "replay_active_point_count": point_count,
                "transformed_xyz_sha256": active_sha,
                "previous_map_state_transition_sha256": previous_map_transition,
                "map_state_transition_sha256": payload["map_state_transition_sha256"],
                "replay_state_transition_sha256": replay_claim,
                "replay_ledger_record_sha256": envelope["record_sha256"],
            }
        )
        previous_map_transition = payload["map_state_transition_sha256"]
        previous_replay_transition = replay_claim
    _same(source_objects, expected_lineage_rows, "map lineage/replay/receipt projection")
    _same(
        lineage["final_map_state_transition_sha256"],
        previous_map_transition,
        "map lineage final transition",
    )
    _same(
        freeze["final_map_state_transition_sha256"],
        previous_map_transition,
        "target freeze final transition",
    )

    target_path = _safe_runtime_file(runtime, str(freeze["target_map_path"]), "target map")
    target, target_sha, target_size = _canonical_npy(target_path, shape_tail=(3,))
    _same(freeze.get("target_map_sha256"), target_sha, "target map payload SHA")
    _same(freeze.get("target_map_size_bytes"), target_size, "target map byte size")
    _same(freeze.get("target_point_count"), int(target.shape[0]), "target point count")
    _same(target_path.stat().st_mode & 0o222, 0, "immutable target map write bits")
    physical = 0
    for path in runtime.rglob("*"):
        if path.is_symlink():
            _fail(f"runtime contains a symlink: {path}")
        if (
            path.is_file()
            and path.stat().st_size == target_size
            and _sha256_file(path) == target_sha
        ):
            physical += 1
    _same(physical, 1, "physical target map copy count")

    reducer = _load_json(root / "target_map_reducer_verification.json")
    _same(set(reducer), TARGET_REDUCER_VERIFICATION_FIELDS, "target reducer exact schema")
    reducer_unsigned = dict(reducer)
    reducer_claim = reducer_unsigned.pop("target_map_reducer_verification_sha256")
    _same(reducer_claim, _hash_json(reducer_unsigned), "target reducer self-hash")
    implementation_sha = _sha256_file(authority.target_reducer_implementation_path)
    implementation_label = PurePosixPath(str(reducer["reducer_implementation_path"]))
    if implementation_label.is_absolute() or ".." in implementation_label.parts:
        _fail("target reducer implementation label is unsafe")
    binary_path = _safe_runtime_file(
        runtime, str(reducer["reducer_binary_path"]), "target reducer binary"
    )
    if binary_path.stat().st_mode & 0o111 == 0:
        _fail("target reducer binary is not executable")
    expected_reducer = {
        "schema_version": "boreas_v2_stage2_target_reducer_verification_v1",
        "map_lineage_manifest_sha256": _sha256_file(root / "map_lineage_manifest.json"),
        "replay_ledger_path": str(lineage["replay_ledger_path"]),
        "replay_ledger_sha256": _sha256_file(ledger_path),
        "replay_plan_sha256": _hash_json(plan),
        "replay_descriptor_count": len(expected_descriptors),
        "replay_authenticated_ranges_sha256": _hash_json(
            [row["replay_range_sha256"] for row in expected_lineage_rows]
        ),
        "replay_total_bytes": byte_start,
        "reducer_implementation_path": str(implementation_label),
        "reducer_implementation_sha256": implementation_sha,
        "reducer_binary_path": str(reducer["reducer_binary_path"]),
        "reducer_binary_sha256": _sha256_file(binary_path),
        "reducer_capacity_layout_verification_file_sha256": reducer[
            "reducer_capacity_layout_verification_file_sha256"
        ],
        "reducer_capacity_replay_binding_file_sha256": reducer[
            "reducer_capacity_replay_binding_file_sha256"
        ],
        "reducer_resource_plan_file_sha256": reducer[
            "reducer_resource_plan_file_sha256"
        ],
        "producer_target_map_sha256": target_sha,
        "reducer_output_target_map_sha256": target_sha,
        "target_map_size_bytes": target_size,
        "target_point_count": int(target.shape[0]),
        "final_map_state_transition_sha256": previous_map_transition,
        "verification_scope": (
            "PRODUCER_REDUCER_LINEAGE_AND_OUTPUT_BINDINGS_"
            "FINAL_VERIFIER_REPLAYS_EXACT_TARGET_BYTES"
        ),
        "verification_status": "PASS_PRODUCER_BINDINGS_AWAITING_FINAL_EXACT_REPLAY",
    }
    for field, expected in expected_reducer.items():
        _same(reducer[field], expected, f"target reducer {field}")
    target_count = int(target.shape[0])
    del target
    production = (
        authority.expected_map_sequence_id == EXPECTED_MAP_SEQUENCE
        and authority.expected_query_sequence_id == EXPECTED_QUERY_SEQUENCE
        and authority.selection_parameter_authority
        == "BOREAS_V2_STAGE2_PREREGISTRATION"
    )
    resource_verification = _verify_reducer_resource_evidence(
        runtime=runtime,
        reducer=reducer,
        replay_plan_sha256=_hash_json(plan),
        reducer_source_sha256=implementation_sha,
        reducer_binary_sha256=_sha256_file(binary_path),
        reducer_binary_size_bytes=binary_path.stat().st_size,
        production=production,
    )
    replay_verification = _verify_target_replay_exact(
        replay_path=replay_path,
        target_path=target_path,
        target_sha256=target_sha,
        target_size_bytes=target_size,
        target_point_count=target_count,
        descriptors=expected_descriptors,
        lineage_rows=expected_lineage_rows,
        reducer=reducer,
        authority=authority,
    )
    replay_verification = {
        **replay_verification,
        "resource_evidence": resource_verification,
    }
    return (
        target_sha,
        str(freeze["target_map_path"]),
        target_size,
        target_count,
        replay_verification,
    )


def _validated_native_receipt(value: Any, label: str) -> dict[str, Any]:
    fields = {
        "downloaded_at_utc",
        "etag",
        "key",
        "last_modified",
        "local_temporary_sha256",
        "remote_size_bytes",
        "receipt_sha256",
        "schema",
        "selection_role",
        "sequence_id",
        "timestamp_us",
    }
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{label} exact schema differs")
    receipt = dict(value)
    for field in ("remote_size_bytes", "timestamp_us"):
        if type(receipt[field]) is not int or receipt[field] <= 0:
            _fail(f"{label} {field} is not a positive integer")
    for field in ("etag", "key", "last_modified", "selection_role", "sequence_id"):
        if not isinstance(receipt[field], str) or not receipt[field]:
            _fail(f"{label} {field} is empty")
    _same(receipt["schema"], DOWNLOAD_RECEIPT_SCHEMA, f"{label} schema")
    _require_sha(receipt["local_temporary_sha256"], f"{label} payload SHA")
    supplied = _require_sha(receipt["receipt_sha256"], f"{label} receipt SHA")
    core = {key: child for key, child in receipt.items() if key != "receipt_sha256"}
    _same(supplied, _hash_json(core), f"{label} self-hash")
    try:
        downloaded = datetime.fromisoformat(
            receipt["downloaded_at_utc"].replace("Z", "+00:00")
        )
    except (AttributeError, ValueError) as error:
        raise BoreasStage2PreparationVerificationError(
            f"{label} download timestamp is invalid"
        ) from error
    if (
        not receipt["downloaded_at_utc"].endswith("Z")
        or downloaded.tzinfo is None
        or downloaded.utcoffset() != timezone.utc.utcoffset(downloaded)
    ):
        _fail(f"{label} download timestamp is not canonical UTC")
    return receipt


def _read_canonical_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        _fail(f"{label} is empty or truncated")
    rows: list[dict[str, Any]] = []
    for line in payload.splitlines(keepends=True):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise BoreasStage2PreparationVerificationError(
                f"{label} contains invalid JSONL"
            ) from error
        if not isinstance(row, dict) or _compact_json_line(row) != line:
            _fail(f"{label} JSONL is noncanonical")
        rows.append(row)
    return rows


def _csv_projection(value: Mapping[str, Any], fields: Sequence[str]) -> dict[str, str]:
    if set(value) != set(fields):
        _fail("journal payload/CSV projection field set differs")
    return {field: "" if value[field] is None else str(value[field]) for field in fields}


def _verify_selection_freeze(
    root: Path,
    *,
    target_freeze_sha256: str,
    first_pass_complete_event: Mapping[str, Any],
    selection_event: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    scans: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    selected_intervals: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    selection_contract_sha256: str,
) -> Mapping[str, Any]:
    path = root / "boreas_v2_stage2_selection_freeze.json"
    freeze = _load_json(path)
    expected_fields = {
        "R14_frozen",
        "artifact_sha256",
        "blind_selection",
        "candidate_interval_count",
        "candidate_interval_rows_sha256",
        "candidate_scan_count",
        "candidate_scan_rows_sha256",
        "first_pass_complete_event_sha256",
        "registration_execution_count",
        "schema",
        "selected_interval_count",
        "selected_interval_rows_sha256",
        "selected_snapshot_count",
        "selected_snapshot_rows_sha256",
        "selection_contract_sha256",
        "selection_freeze_payload_sha256",
        "target_freeze_file_sha256",
    }
    _same(set(freeze), expected_fields, "selection freeze exact schema")
    unsigned = dict(freeze)
    supplied = unsigned.pop("selection_freeze_payload_sha256")
    _same(supplied, _hash_json(unsigned), "selection freeze self-hash")
    frozen_artifacts = {
        "all_candidate_scans.csv",
        "all_candidate_intervals.csv",
        "excluded_candidate_scans.csv",
        "excluded_intervals.csv",
        "geometry_only_metrics.csv",
        "geometry_metric_verification.csv",
        "selected_scene_intervals.csv",
        "selected_snapshots.csv",
    }
    expected_artifacts = {
        name: _sha256_file(root / name) for name in sorted(frozen_artifacts)
    }
    expected_values = {
        "schema": SELECTION_FREEZE_SCHEMA,
        "target_freeze_file_sha256": target_freeze_sha256,
        "first_pass_complete_event_sha256": first_pass_complete_event[
            "event_sha256"
        ],
        "selection_contract_sha256": selection_contract_sha256,
        "candidate_scan_count": len(scans),
        "candidate_scan_rows_sha256": _hash_json(
            [row["candidate_scan_row_sha256"] for row in scans]
        ),
        "candidate_interval_count": len(intervals),
        "candidate_interval_rows_sha256": _hash_json(
            [row["candidate_interval_row_sha256"] for row in intervals]
        ),
        "selected_interval_count": len(selected_intervals),
        "selected_interval_rows_sha256": _hash_json(
            [row["selected_interval_row_sha256"] for row in selected_intervals]
        ),
        "selected_snapshot_count": len(selected),
        "selected_snapshot_rows_sha256": _hash_json(
            [row["selected_snapshot_row_sha256"] for row in selected]
        ),
        "artifact_sha256": expected_artifacts,
        "blind_selection": selection_manifest.get("blind_selection"),
        "R14_frozen": True,
        "registration_execution_count": 0,
    }
    for field, expected in expected_values.items():
        _same(freeze[field], expected, f"selection freeze {field}")
    barrier_payload = selection_event["payload"]
    expected_barrier = {
        "artifact_sha256": expected_artifacts,
        "first_pass_complete_event_sha256": first_pass_complete_event[
            "event_sha256"
        ],
        "selected_snapshot_count": len(selected),
        "selection_contract_sha256": selection_contract_sha256,
        "selection_freeze_file_sha256": _sha256_file(path),
    }
    _same(barrier_payload, expected_barrier, "R14 selection barrier payload")
    return freeze


def _verify_query_evidence(
    root: Path,
    runtime: Path,
    *,
    query_allowlist: Sequence[Mapping[str, str]],
    scans: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    selected_intervals: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
    selection_contract_sha256: str,
    preprocessing_sha256: str,
    gt_sha256: str,
    extrinsic_sha256: str,
    target_map_sha256: str,
    reference_poses: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    audit_fields = {
        "R14_selection_frozen_event_sha256",
        "committed_payload_bytes",
        "committed_payload_event_count",
        "first_pass_committed_count",
        "first_pass_complete_event_sha256",
        "query_journal_final_event_sha256",
        "query_journal_path",
        "query_journal_sha256",
        "raw_payload_persistent_bytes",
        "registration_execution_count",
        "retry_download_event_count",
        "retry_download_payload_bytes",
        "schema",
        "selected_source_committed_count",
        "successful_download_event_count",
        "successful_payload_bytes",
        "target_frozen_event_sha256",
    }
    audit = _load_json(root / "QUERY_LIDAR_DOWNLOAD_AUDIT.json")
    _same(set(audit), audit_fields, "query audit exact schema")
    _same(audit["schema"], QUERY_AUDIT_SCHEMA, "query audit schema")
    journal_path = _safe_runtime_file(
        runtime, str(audit["query_journal_path"]), "query journal"
    )
    _same(
        audit["query_journal_sha256"],
        _sha256_file(journal_path),
        "query journal audit SHA",
    )
    events = _read_canonical_jsonl(journal_path, "query journal")
    envelope_fields = {
        "event_kind",
        "event_sha256",
        "execution_stage",
        "object_key",
        "payload",
        "previous_event_sha256",
        "schema",
        "sequence_number",
    }
    stages = {
        "QUERY_BARRIER",
        "QUERY_GEOMETRY_FIRST_PASS",
        "QUERY_CANONICAL_SECOND_PASS",
    }
    barriers: dict[str, dict[str, Any]] = {}
    intents: dict[str, dict[str, Any]] = {}
    resolved_intents: set[str] = set()
    downloads: dict[str, dict[str, Any]] = {}
    resolved_downloads: set[str] = set()
    first_commits: list[dict[str, Any]] = []
    second_commits: list[dict[str, Any]] = []
    aborts: list[dict[str, Any]] = []
    transfer_aborts: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    first_complete_seen = False
    selection_seen = False
    query_by_key = {row["key"]: row for row in query_allowlist}
    selected_keys = {str(row["object_key"]) for row in selected}
    for sequence_number, event in enumerate(events, start=1):
        _same(set(event), envelope_fields, "query journal envelope schema")
        _same(event["schema"], QUERY_JOURNAL_SCHEMA, "query journal schema")
        _same(event["sequence_number"], sequence_number, "query journal sequence")
        _same(event["previous_event_sha256"], previous, "query journal chain")
        if event["execution_stage"] not in stages or not isinstance(event["payload"], dict):
            _fail("query journal stage or payload differs")
        unsigned = dict(event)
        supplied = _require_sha(unsigned.pop("event_sha256"), "query event SHA")
        _same(supplied, hashlib.sha256(_compact_json_line(unsigned)).hexdigest(), "query event SHA")
        previous = supplied
        kind = event["event_kind"]
        stage = event["execution_stage"]
        object_key = event["object_key"]
        payload = event["payload"]
        if stage == "QUERY_CANONICAL_SECOND_PASS" and not selection_seen:
            _fail("query second pass precedes durable R14 selection barrier")
        if stage == "QUERY_GEOMETRY_FIRST_PASS" and first_complete_seen:
            _fail("query first-pass activity occurs after completion barrier")
        if kind in {"TARGET_FROZEN", "QUERY_FIRST_PASS_COMPLETE", "R14_SELECTION_FROZEN"}:
            if stage != "QUERY_BARRIER" or object_key != "" or kind in barriers:
                _fail("query barrier identity or uniqueness differs")
            expected_fields = {
                "TARGET_FROZEN": {
                    "target_freeze_file_sha256",
                    "target_map_sha256",
                    "target_context_resource_plan_sha256",
                    "target_context_measurement_evidence_sha256",
                    "target_context_measurement_provenance_sha256",
                },
                "QUERY_FIRST_PASS_COMPLETE": {
                    "first_pass_count",
                    "first_pass_rows_sha256",
                    "target_freeze_event_sha256",
                },
                "R14_SELECTION_FROZEN": {
                    "artifact_sha256",
                    "first_pass_complete_event_sha256",
                    "selected_snapshot_count",
                    "selection_contract_sha256",
                    "selection_freeze_file_sha256",
                },
            }[kind]
            _same(set(payload), expected_fields, f"{kind} payload schema")
            if kind == "TARGET_FROZEN" and sequence_number != 1:
                _fail("TARGET_FROZEN is not the first query-journal event")
            if kind == "QUERY_FIRST_PASS_COMPLETE":
                if "TARGET_FROZEN" not in barriers:
                    _fail("first-pass-complete barrier precedes target freeze")
                first_complete_seen = True
            if kind == "R14_SELECTION_FROZEN":
                if not first_complete_seen:
                    _fail("R14 selection barrier precedes first-pass completion")
                selection_seen = True
            barriers[kind] = event
            continue
        if stage == "QUERY_BARRIER" or not isinstance(object_key, str) or not object_key:
            _fail("non-barrier query event identity differs")
        allowed_keys = query_by_key if stage == "QUERY_GEOMETRY_FIRST_PASS" else selected_keys
        if object_key not in allowed_keys:
            _fail("query journal names an object outside its frozen phase set")
        if kind == "TRANSFER_INTENT":
            expected = {
                "etag",
                "last_modified",
                "remote_size_bytes",
                "sequence_id",
                "temporary_relative_path",
                "timestamp_us",
            }
            _same(set(payload), expected, "query transfer-intent schema")
            allow = query_by_key[object_key]
            for field, expected_value in (
                ("etag", None),
                ("last_modified", allow["last_modified"]),
                ("remote_size_bytes", int(allow["size_bytes"])),
                ("sequence_id", allow["sequence_id"]),
                ("timestamp_us", int(allow["timestamp_us"])),
            ):
                if expected_value is None:
                    if not isinstance(payload[field], str) or not payload[field]:
                        _fail("query transfer-intent ETag is empty")
                else:
                    _same(payload[field], expected_value, f"query transfer intent {field}")
            relative = PurePosixPath(str(payload["temporary_relative_path"]))
            if relative.is_absolute() or not relative.parts or any(
                part in {"", ".", ".."} for part in relative.parts
            ):
                _fail("query transfer temporary path is unsafe")
            if runtime.joinpath(*relative.parts).exists():
                _fail("query raw temporary payload persists after closure")
            intents[supplied] = event
            continue
        if kind == "TRANSFER_ABORTED":
            _same(set(payload), {"reason", "transfer_intent_event_sha256"}, "query transfer-abort schema")
            intent_sha = _require_sha(payload["transfer_intent_event_sha256"], "transfer-intent event SHA")
            intent = intents.get(intent_sha)
            if (
                intent is None
                or intent_sha in resolved_intents
                or intent["execution_stage"] != stage
                or intent["object_key"] != object_key
                or not isinstance(payload["reason"], str)
                or not payload["reason"]
            ):
                _fail("query transfer-abort identity differs")
            resolved_intents.add(intent_sha)
            transfer_aborts.append(event)
            continue
        if kind == "DOWNLOADED":
            _same(
                set(payload),
                {"receipt", "temporary_relative_path", "transfer_intent_event_sha256"},
                "query download-event schema",
            )
            intent_sha = _require_sha(payload["transfer_intent_event_sha256"], "transfer-intent event SHA")
            intent = intents.get(intent_sha)
            receipt = _validated_native_receipt(payload["receipt"], "query journal receipt")
            if intent is None or intent_sha in resolved_intents:
                _fail("query download lacks one unresolved transfer intent")
            intent_payload = intent["payload"]
            expected_identity = {
                "etag": intent_payload["etag"],
                "last_modified": intent_payload["last_modified"],
                "remote_size_bytes": intent_payload["remote_size_bytes"],
                "sequence_id": intent_payload["sequence_id"],
                "timestamp_us": intent_payload["timestamp_us"],
            }
            for field, expected_value in expected_identity.items():
                _same(receipt[field], expected_value, f"query intent/download {field}")
            _same(payload["temporary_relative_path"], intent_payload["temporary_relative_path"], "query intent/download temporary path")
            _same(receipt["selection_role"], "QUERY", "query receipt role")
            _same(receipt["key"], object_key, "query receipt key")
            resolved_intents.add(intent_sha)
            downloads[supplied] = event
            continue
        if kind not in {"ABORTED", "FIRST_PASS_COMMITTED", "SELECTED_SOURCE_COMMITTED"}:
            _fail("unknown query journal event kind")
        required = {"download_event_sha256", "receipt_sha256"}
        if not required <= set(payload):
            _fail("query resolution schema lacks receipt linkage")
        download_sha = _require_sha(payload["download_event_sha256"], "query download-event SHA")
        download = downloads.get(download_sha)
        if download is None or download_sha in resolved_downloads:
            _fail("query resolution lacks one unresolved download")
        receipt = _validated_native_receipt(download["payload"]["receipt"], "resolved query receipt")
        if (
            download["execution_stage"] != stage
            or download["object_key"] != object_key
            or payload["receipt_sha256"] != receipt["receipt_sha256"]
        ):
            _fail("query resolution/download identity differs")
        if kind == "ABORTED":
            _same(set(payload), {"download_event_sha256", "reason", "receipt_sha256"}, "query abort schema")
            if not isinstance(payload["reason"], str) or not payload["reason"]:
                _fail("query abort reason is empty")
            aborts.append(event)
        elif kind == "FIRST_PASS_COMMITTED":
            _same(stage, "QUERY_GEOMETRY_FIRST_PASS", "first-pass commit stage")
            _same(
                set(payload),
                {
                    "download_event_sha256",
                    "first_pass_row",
                    "geometry_metric_verification_row",
                    "processing_result_sha256",
                    "receipt_sha256",
                },
                "first-pass commit payload schema",
            )
            first_commits.append(event)
        else:
            _same(stage, "QUERY_CANONICAL_SECOND_PASS", "selected-source commit stage")
            _same(
                set(payload),
                {
                    "canonical_input_row",
                    "canonical_source_verification_row",
                    "download_event_sha256",
                    "processing_result_sha256",
                    "receipt_sha256",
                    "selected_snapshot_row_sha256",
                },
                "selected-source commit payload schema",
            )
            second_commits.append(event)
        resolved_downloads.add(download_sha)

    _same(set(barriers), {"TARGET_FROZEN", "QUERY_FIRST_PASS_COMPLETE", "R14_SELECTION_FROZEN"}, "query barrier closure")
    _same(set(intents), resolved_intents, "query transfer-intent closure")
    _same(set(downloads), resolved_downloads, "query download-event closure")
    _same([row["object_key"] for row in first_commits], [row["key"] for row in query_allowlist], "query first-pass committed order")
    _same([row["object_key"] for row in second_commits], [str(row["object_key"]) for row in selected], "query second-pass committed order")
    target_event = barriers["TARGET_FROZEN"]
    _same(
        target_event["payload"],
        {
            "target_freeze_file_sha256": _sha256_file(root / "target_map_freeze_manifest.json"),
            "target_map_sha256": target_map_sha256,
            "target_context_resource_plan_sha256": _sha256_file(
                runtime / "evidence/target_context_resource_plan.json"
            ),
            "target_context_measurement_evidence_sha256": _sha256_file(
                runtime / "evidence/target_context_capacity_measurement.json"
            ),
            "target_context_measurement_provenance_sha256": _sha256_file(
                runtime
                / "evidence/target_context_capacity_measurement_provenance.json"
            ),
        },
        "TARGET_FROZEN payload",
    )

    geometry_csv = _csv_rows(
        root / "geometry_metric_verification.csv",
        GEOMETRY_METRIC_VERIFICATION_FIELDS,
    )
    _same(len(geometry_csv), len(first_commits), "geometry witness count")
    first_rows: list[Mapping[str, Any]] = []
    first_by_event: dict[str, Mapping[str, Any]] = {}
    for ordinal, (commit, scan, witness_csv) in enumerate(
        zip(first_commits, scans, geometry_csv)
    ):
        payload = commit["payload"]
        first = payload["first_pass_row"]
        witness = payload["geometry_metric_verification_row"]
        _same(set(first), FIRST_PASS_SCAN_FIELDS, "first-pass row exact schema")
        expected_first = {field: scan[field] for field in FIRST_PASS_SCAN_FIELDS}
        _same(first, expected_first, "journal/candidate first-pass projection")
        first_result = _hash_json(first)
        _same(payload["processing_result_sha256"], first_result, "first-pass processing-result SHA")
        _same(set(witness), set(GEOMETRY_METRIC_VERIFICATION_FIELDS), "geometry witness exact schema")
        _same(witness_csv, _csv_projection(witness, GEOMETRY_METRIC_VERIFICATION_FIELDS), "geometry witness journal/CSV projection")
        witness_core = dict(witness)
        witness_claim = witness_core.pop("geometry_metric_verification_row_sha256")
        _same(witness_claim, _hash_json(witness_core), "geometry witness row SHA")
        geometry = {field: first[field] for field in GEOMETRY_ONLY_FIELDS}
        geometry_sha = _hash_json(geometry)
        download = downloads[payload["download_event_sha256"]]
        receipt = _validated_native_receipt(download["payload"]["receipt"], "first-pass receipt")
        expected_witness = {
            "query_ordinal": ordinal,
            "object_key": first["object_key"],
            "raw_payload_sha256": receipt["local_temporary_sha256"],
            "receipt_sha256": receipt["receipt_sha256"],
            "preprocessing_contract_sha256": preprocessing_sha256,
            "gt_sha256": gt_sha256,
            "extrinsic_sha256": extrinsic_sha256,
            "target_map_sha256": target_map_sha256,
            "producer_geometry_sha256": geometry_sha,
            "independent_geometry_sha256": geometry_sha,
            "verification_status": GEOMETRY_WITNESS_STATUS,
        }
        for field, expected_value in expected_witness.items():
            _same(witness[field], expected_value, f"geometry witness {field}")
        for left, right, label in (
            ("producer_source_sha256", "independent_source_sha256", "source"),
            ("producer_T_reference_sha256", "independent_T_reference_sha256", "T_reference"),
        ):
            _require_sha(witness[left], f"geometry witness producer {label} SHA")
            _same(witness[left], witness[right], f"geometry dual-path {label} identity")
        integer_fields = (
            "raw_point_count",
            "nonfinite_excluded_count",
            "range_excluded_count",
            "post_filter_point_count",
            "source_voxel_reduced_count",
            "canonical_source_point_count",
        )
        if any(type(witness[field]) is not int or witness[field] < 0 for field in integer_fields):
            _fail("geometry witness counters are not nonnegative integers")
        _same(witness["raw_point_count"] * 24, receipt["remote_size_bytes"], "geometry witness raw byte/count closure")
        _same(
            witness["raw_point_count"],
            witness["nonfinite_excluded_count"] + witness["range_excluded_count"] + witness["post_filter_point_count"],
            "geometry witness raw filtering closure",
        )
        _same(
            witness["post_filter_point_count"],
            witness["source_voxel_reduced_count"] + witness["canonical_source_point_count"],
            "geometry witness voxel closure",
        )
        _same(witness["canonical_source_point_count"], first["finite_source_point_count"], "geometry witness finite source count")
        translation, quaternion = _independent_exact_pose(int(first["timestamp_us"]), *reference_poses)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        transform[:3, 3] = translation
        expected_transform_sha = hashlib.sha256(_canonical_npy_bytes(transform)).hexdigest()
        _same(witness["producer_T_reference_sha256"], expected_transform_sha, "geometry witness independently reconstructed T_reference")
        first_rows.append(first)
        first_by_event[commit["event_sha256"]] = commit

    first_complete = barriers["QUERY_FIRST_PASS_COMPLETE"]
    _same(
        first_complete["payload"],
        {
            "first_pass_count": len(first_commits),
            "first_pass_rows_sha256": _hash_json(first_rows),
            "target_freeze_event_sha256": target_event["event_sha256"],
        },
        "query first-pass-complete payload",
    )
    selection_event = barriers["R14_SELECTION_FROZEN"]
    _verify_selection_freeze(
        root,
        target_freeze_sha256=_sha256_file(root / "target_map_freeze_manifest.json"),
        first_pass_complete_event=first_complete,
        selection_event=selection_event,
        selection_manifest=selection_manifest,
        scans=scans,
        intervals=intervals,
        selected_intervals=selected_intervals,
        selected=selected,
        selection_contract_sha256=selection_contract_sha256,
    )
    projection_rows = _csv_rows(
        root / "first_pass_selection_projection.csv",
        FIRST_PASS_SELECTION_PROJECTION_FIELDS,
    )
    _same(len(projection_rows), len(first_commits), "first-pass projection count")
    projection_by_event: dict[str, Mapping[str, str]] = {}
    for ordinal, (projection, commit, scan) in enumerate(
        zip(projection_rows, first_commits, scans)
    ):
        core = {
            "execution_stage": projection["execution_stage"],
            "query_ordinal": _parse_int(projection["query_ordinal"], "projection ordinal"),
            "object_key": projection["object_key"],
            "first_pass_event_sha256": projection["first_pass_event_sha256"],
            "first_pass_result_sha256": projection["first_pass_result_sha256"],
            "candidate_scan_row_sha256": projection["candidate_scan_row_sha256"],
            "selection_freeze_event_sha256": projection["selection_freeze_event_sha256"],
        }
        _same(projection["projection_row_sha256"], _hash_json(core), "first-pass projection row SHA")
        expected = {
            "execution_stage": "QUERY_GEOMETRY_FIRST_PASS",
            "query_ordinal": ordinal,
            "object_key": commit["object_key"],
            "first_pass_event_sha256": commit["event_sha256"],
            "first_pass_result_sha256": commit["payload"]["processing_result_sha256"],
            "candidate_scan_row_sha256": scan["candidate_scan_row_sha256"],
            "selection_freeze_event_sha256": selection_event["event_sha256"],
        }
        _same(core, expected, "first-pass selection projection linkage")
        projection_by_event[commit["event_sha256"]] = projection

    canonical_csv = _csv_rows(root / "canonical_input_manifest.csv", CANONICAL_INPUT_FIELDS)
    canonical_witness_csv = _csv_rows(
        root / "canonical_source_verification.csv",
        CANONICAL_SOURCE_VERIFICATION_FIELDS,
    )
    _same(len(canonical_csv), len(second_commits), "selected-source canonical count")
    for commit, snapshot, canonical_row, witness_row in zip(
        second_commits, selected, canonical_csv, canonical_witness_csv
    ):
        payload = commit["payload"]
        _same(
            canonical_row,
            _csv_projection(payload["canonical_input_row"], CANONICAL_INPUT_FIELDS),
            "selected-source journal/canonical projection",
        )
        _same(
            witness_row,
            _csv_projection(
                payload["canonical_source_verification_row"],
                CANONICAL_SOURCE_VERIFICATION_FIELDS,
            ),
            "selected-source journal/witness projection",
        )
        _same(payload["selected_snapshot_row_sha256"], snapshot["selected_snapshot_row_sha256"], "selected-source snapshot linkage")
        _same(payload["processing_result_sha256"], payload["canonical_input_row"]["source_points_sha256"], "selected-source processing-result SHA")
        first_event = first_commits[int(snapshot["query_ordinal"])]
        first_witness = first_event["payload"]["geometry_metric_verification_row"]
        _same(first_witness["producer_source_sha256"], payload["canonical_input_row"]["source_points_sha256"], "selected first/second-pass canonical source identity")
        _same(first_witness["producer_T_reference_sha256"], payload["canonical_input_row"]["T_reference_sha256"], "selected first/second-pass T_reference identity")

    download_rows = list(downloads.values())
    committed_count = len(first_commits) + len(second_commits)
    committed_bytes = sum(
        int(downloads[event["payload"]["download_event_sha256"]]["payload"]["receipt"]["remote_size_bytes"])
        for event in first_commits + second_commits
    )
    downloaded_bytes = sum(int(row["payload"]["receipt"]["remote_size_bytes"]) for row in download_rows)
    expected_audit = {
        "schema": QUERY_AUDIT_SCHEMA,
        "successful_download_event_count": len(download_rows),
        "successful_payload_bytes": downloaded_bytes,
        "committed_payload_event_count": committed_count,
        "committed_payload_bytes": committed_bytes,
        "retry_download_event_count": len(aborts),
        "retry_download_payload_bytes": downloaded_bytes - committed_bytes,
        "raw_payload_persistent_bytes": 0,
        "first_pass_committed_count": len(first_commits),
        "selected_source_committed_count": len(second_commits),
        "query_journal_path": str(audit["query_journal_path"]),
        "query_journal_sha256": _sha256_file(journal_path),
        "query_journal_final_event_sha256": events[-1]["event_sha256"],
        "target_frozen_event_sha256": target_event["event_sha256"],
        "first_pass_complete_event_sha256": first_complete["event_sha256"],
        "R14_selection_frozen_event_sha256": selection_event["event_sha256"],
        "registration_execution_count": 0,
    }
    _same(audit, expected_audit, "query audit replay")
    return {
        "audit": audit,
        "events": events,
        "downloads": downloads,
        "first_commits": first_commits,
        "second_commits": second_commits,
        "aborts": aborts,
        "transfer_aborts": transfer_aborts,
        "projection_by_event": projection_by_event,
    }


def _verify_map_receipt_evidence(
    root: Path,
    runtime: Path,
    *,
    map_allowlist: Sequence[Mapping[str, str]],
    preprocessing_sha256: str,
    gt_sha256: str,
    extrinsic_sha256: str,
) -> dict[str, Any]:
    audit_fields = {
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
    audit = _load_json(root / "MAP_LIDAR_DOWNLOAD_AUDIT.json")
    _same(set(audit), audit_fields, "map audit exact schema")
    _same(audit["schema"], MAP_AUDIT_SCHEMA, "map audit schema")
    journal_path = _safe_runtime_file(
        runtime,
        str(audit["receipt_checkpoint_journal_path"]),
        "map receipt journal",
    )
    _same(
        audit["receipt_checkpoint_journal_sha256"],
        _sha256_file(journal_path),
        "map receipt journal audit SHA",
    )
    events = _read_canonical_jsonl(journal_path, "map receipt journal")
    common_fields = {
        "event_kind",
        "event_sha256",
        "execution_stage",
        "previous_event_sha256",
        "schema",
        "sequence_number",
    }
    fields_by_kind = {
        "DOWNLOADED": common_fields
        | {
            "extrinsic_sha256",
            "gt_sha256",
            "preprocessing_contract_sha256",
            "receipt",
        },
        "PREPROCESSED": common_fields
        | {
            "canonical_source_witness_sha256",
            "download_event_sha256",
            "key",
            "receipt_sha256",
            "source_point_count",
            "source_witness_metadata",
        },
        "REPLAY_COMMITTED": common_fields
        | {
            "checkpoint_status",
            "download_event_sha256",
            "key",
            "preprocessed_event_sha256",
            "processing_result_sha256",
            "receipt_sha256",
            "replay_ordinal",
        },
        "ABORTED": common_fields
        | {
            "download_event_sha256",
            "key",
            "reason",
            "receipt_sha256",
        },
    }
    allow_by_key = {row["key"]: row for row in map_allowlist}
    previous = ZERO_SHA256
    downloads: dict[str, dict[str, Any]] = {}
    preprocessed: dict[str, dict[str, Any]] = {}
    resolved: set[str] = set()
    commits: list[dict[str, Any]] = []
    aborts: list[dict[str, Any]] = []
    preprocessed_events: list[dict[str, Any]] = []
    for sequence_number, event in enumerate(events, start=1):
        kind = event.get("event_kind")
        if kind not in fields_by_kind:
            _fail("unknown map receipt-journal event kind")
        _same(set(event), fields_by_kind[kind], "map receipt-journal event schema")
        _same(event["schema"], MAP_RECEIPT_JOURNAL_SCHEMA, "map receipt-journal schema")
        _same(event["sequence_number"], sequence_number, "map receipt-journal sequence")
        _same(event["previous_event_sha256"], previous, "map receipt-journal chain")
        _same(event["execution_stage"], "MAP_INGEST", "map receipt-journal stage")
        unsigned = dict(event)
        supplied = _require_sha(unsigned.pop("event_sha256"), "map journal event SHA")
        _same(supplied, hashlib.sha256(_compact_json_line(unsigned)).hexdigest(), "map journal event SHA")
        previous = supplied
        if kind == "DOWNLOADED":
            receipt = _validated_native_receipt(event["receipt"], "map journal receipt")
            allow = allow_by_key.get(receipt["key"])
            if allow is None:
                _fail("map journal receipt is outside the frozen map allowlist")
            for field, expected in (
                ("selection_role", "TARGET_MAP"),
                ("sequence_id", allow["sequence_id"]),
                ("timestamp_us", int(allow["timestamp_us"])),
                ("remote_size_bytes", int(allow["size_bytes"])),
                ("last_modified", allow["last_modified"]),
            ):
                _same(receipt[field], expected, f"map journal receipt {field}")
            for field, expected in (
                ("preprocessing_contract_sha256", preprocessing_sha256),
                ("gt_sha256", gt_sha256),
                ("extrinsic_sha256", extrinsic_sha256),
            ):
                _same(event[field], expected, f"map journal {field}")
            downloads[supplied] = event
            continue
        download_sha = _require_sha(event["download_event_sha256"], "map download-event SHA")
        download = downloads.get(download_sha)
        if download is None or download_sha in resolved:
            _fail("map journal event lacks one unresolved download")
        receipt = _validated_native_receipt(download["receipt"], "resolved map receipt")
        if event["key"] != receipt["key"] or event["receipt_sha256"] != receipt["receipt_sha256"]:
            _fail("map journal resolution/download identity differs")
        if kind == "PREPROCESSED":
            if download_sha in preprocessed:
                _fail("duplicate map preprocessing event")
            _require_sha(event["canonical_source_witness_sha256"], "map source witness SHA")
            if type(event["source_point_count"]) is not int or event["source_point_count"] <= 0:
                _fail("map source point count is not positive")
            if not isinstance(event["source_witness_metadata"], dict):
                _fail("map source witness metadata is not an object")
            preprocessed[download_sha] = event
            preprocessed_events.append(event)
            continue
        if kind == "REPLAY_COMMITTED":
            preprocessing = preprocessed.get(download_sha)
            if preprocessing is None or event["preprocessed_event_sha256"] != preprocessing["event_sha256"]:
                _fail("map replay commit lacks its preprocessing witness")
            _same(event["checkpoint_status"], "COMMITTED", "map replay checkpoint status")
            _require_sha(event["processing_result_sha256"], "map processing result SHA")
            if type(event["replay_ordinal"]) is not int or event["replay_ordinal"] < 0:
                _fail("map replay ordinal is invalid")
            commits.append(event)
        else:
            if not isinstance(event["reason"], str) or not event["reason"]:
                _fail("map abort reason is empty")
            aborts.append(event)
        resolved.add(download_sha)
    _same(set(downloads), resolved, "map receipt-journal download closure")
    _same([row["key"] for row in commits], [row["key"] for row in map_allowlist], "map committed order")
    _same([row["replay_ordinal"] for row in commits], list(range(len(map_allowlist))), "map replay ordinal order")

    projected_rows = _csv_rows(root / "map_download_receipts.csv", RECEIPT_FIELDS)
    _same(len(projected_rows), len(commits), "map receipt projection count")
    committed_pairs: list[dict[str, Any]] = []
    for exported, commit in zip(projected_rows, commits):
        download = downloads[commit["download_event_sha256"]]
        receipt = _validated_native_receipt(download["receipt"], "committed map receipt")
        expected_native = {
            "execution_stage": "MAP_INGEST",
            **receipt,
            "processing_result_sha256": commit["processing_result_sha256"],
            "preprocessing_contract_sha256": download["preprocessing_contract_sha256"],
            "gt_sha256": download["gt_sha256"],
            "extrinsic_sha256": download["extrinsic_sha256"],
            "checkpoint_status": "COMMITTED",
        }
        _same(exported, _csv_projection(expected_native, RECEIPT_FIELDS), "map journal/receipt CSV projection")
        committed_pairs.append({"download": download, "commit": commit, "exported": exported})
    download_bytes = sum(int(row["receipt"]["remote_size_bytes"]) for row in downloads.values())
    committed_bytes = sum(int(row["remote_size_bytes"]) for row in projected_rows)
    expected_audit = {
        "aborted_download_event_count": len(aborts),
        "canonical_source_witness_count": len(preprocessed_events),
        "committed_payload_bytes": committed_bytes,
        "committed_payload_event_count": len(commits),
        "raw_payload_persistent_bytes": 0,
        "receipt_checkpoint_final_chain_sha256": events[-1]["event_sha256"],
        "receipt_checkpoint_journal_path": str(audit["receipt_checkpoint_journal_path"]),
        "receipt_checkpoint_journal_sha256": _sha256_file(journal_path),
        "replay_committed_event_count": len(commits),
        "retry_download_event_count": len(downloads) - len(commits),
        "retry_download_payload_bytes": download_bytes - committed_bytes,
        "schema": MAP_AUDIT_SCHEMA,
        "stream_deleted_raw_bytes": download_bytes,
        "successful_download_event_count": len(downloads),
        "successful_payload_bytes": download_bytes,
        "unique_allowlist_object_count": len(map_allowlist),
    }
    _same(audit, expected_audit, "map audit replay")
    return {
        "audit": audit,
        "events": events,
        "downloads": downloads,
        "commits": commits,
        "aborts": aborts,
        "projected_rows": projected_rows,
        "committed_pairs": committed_pairs,
    }


def _verify_receipts(
    root: Path,
    runtime: Path,
    map_allowlist: Sequence[Mapping[str, str]],
    query_allowlist: Sequence[Mapping[str, str]],
    scans: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    authority: PreparationVerificationAuthority,
    *,
    preprocessing_sha: str,
    extrinsic_sha: str,
    query_state: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, str]]:
    rows = _csv_rows(root / "download_receipts.csv", RECEIPT_FIELDS)
    expected_identity: dict[tuple[str, str], Mapping[str, str]] = {}
    for phase, role, allow_rows in (
        ("MAP_INGEST", "TARGET_MAP", map_allowlist),
        ("QUERY_GEOMETRY_FIRST_PASS", "QUERY", query_allowlist),
    ):
        for allow in allow_rows:
            expected_identity[(phase, allow["key"])] = allow
    query_by_key = {row["key"]: row for row in query_allowlist}
    for snapshot in selected:
        expected_identity[("QUERY_CANONICAL_SECOND_PASS", snapshot["object_key"])] = query_by_key[
            snapshot["object_key"]
        ]
    _same(len(rows), len(expected_identity), "successful receipt count")
    actual_identity: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (row["execution_stage"], row["key"])
        if key in actual_identity:
            _fail("duplicate successful download receipt")
        actual_identity[key] = row
        if key not in expected_identity:
            _fail("receipt is outside frozen map/query/selected sets")
        allow = expected_identity[key]
        expected_role = (
            "TARGET_MAP" if key[0] == "MAP_INGEST" else "QUERY"
        )
        expected_gt = (
            authority.expected_map_reference_pose_sha256
            if expected_role == "TARGET_MAP"
            else authority.expected_reference_pose_sha256
        )
        for field, expected in (
            ("schema", DOWNLOAD_RECEIPT_SCHEMA),
            ("selection_role", expected_role),
            ("sequence_id", allow["sequence_id"]),
            ("timestamp_us", allow["timestamp_us"]),
            ("last_modified", allow["last_modified"]),
            ("remote_size_bytes", allow["size_bytes"]),
            ("preprocessing_contract_sha256", preprocessing_sha),
            ("gt_sha256", expected_gt),
            ("extrinsic_sha256", extrinsic_sha),
            ("checkpoint_status", "COMMITTED"),
        ):
            _same(row[field], expected, f"download receipt {field}")
        for field in (
            "local_temporary_sha256",
            "receipt_sha256",
            "processing_result_sha256",
            "preprocessing_contract_sha256",
            "gt_sha256",
            "extrinsic_sha256",
        ):
            _require_sha(row[field], f"receipt {field}")
        if not row["etag"]:
            _fail("download receipt ETag is empty")
        try:
            downloaded = datetime.fromisoformat(
                row["downloaded_at_utc"].replace("Z", "+00:00")
            )
        except ValueError as error:
            raise BoreasStage2PreparationVerificationError(
                "receipt download timestamp is invalid"
            ) from error
        if (
            not row["downloaded_at_utc"].endswith("Z")
            or downloaded.tzinfo is None
            or downloaded.utcoffset() != timezone.utc.utcoffset(downloaded)
        ):
            _fail("receipt download timestamp is not canonical UTC")
        native_core = {
            "downloaded_at_utc": row["downloaded_at_utc"],
            "etag": row["etag"],
            "key": row["key"],
            "last_modified": row["last_modified"],
            "local_temporary_sha256": row["local_temporary_sha256"],
            "remote_size_bytes": int(row["remote_size_bytes"]),
            "schema": row["schema"],
            "selection_role": row["selection_role"],
            "sequence_id": row["sequence_id"],
            "timestamp_us": int(row["timestamp_us"]),
        }
        _same(row["receipt_sha256"], _hash_json(native_core), "native receipt self-hash")
    _same(set(actual_identity), set(expected_identity), "exact download receipt identity")
    for snapshot in selected:
        object_key = str(snapshot["object_key"])
        first = actual_identity[("QUERY_GEOMETRY_FIRST_PASS", object_key)]
        second = actual_identity[("QUERY_CANONICAL_SECOND_PASS", object_key)]
        for field in (
            "etag",
            "local_temporary_sha256",
            "remote_size_bytes",
            "last_modified",
        ):
            _same(second[field], first[field], f"two-pass query receipt {field}")

    map_state = _verify_map_receipt_evidence(
        root,
        runtime,
        map_allowlist=map_allowlist,
        preprocessing_sha256=preprocessing_sha,
        gt_sha256=authority.expected_map_reference_pose_sha256,
        extrinsic_sha256=extrinsic_sha,
    )
    _same(
        rows[: len(map_allowlist)],
        map_state["projected_rows"],
        "aggregate/map receipt projection",
    )
    scan_by_key = {str(row["object_key"]): row for row in scans}
    expected_query_rows: list[dict[str, str]] = []
    for commit in list(query_state["first_commits"]) + list(query_state["second_commits"]):
        download = query_state["downloads"][commit["payload"]["download_event_sha256"]]
        native = _validated_native_receipt(download["payload"]["receipt"], "committed query receipt")
        if commit["event_kind"] == "FIRST_PASS_COMMITTED":
            processing_result = scan_by_key[commit["object_key"]][
                "candidate_scan_row_sha256"
            ]
        else:
            processing_result = commit["payload"]["processing_result_sha256"]
        expected_query_rows.append(
            _csv_projection(
                {
                    "execution_stage": commit["execution_stage"],
                    **native,
                    "processing_result_sha256": processing_result,
                    "preprocessing_contract_sha256": preprocessing_sha,
                    "gt_sha256": authority.expected_reference_pose_sha256,
                    "extrinsic_sha256": extrinsic_sha,
                    "checkpoint_status": "COMMITTED",
                },
                RECEIPT_FIELDS,
            )
        )
    _same(
        rows[len(map_allowlist) :],
        expected_query_rows,
        "query journal/aggregate receipt projection",
    )

    aggregate_fields = {
        "R14_selection_frozen_event_sha256",
        "committed_payload_bytes",
        "committed_payload_event_count",
        "map_receipt_count",
        "map_receipt_final_chain_sha256",
        "map_receipt_journal_path",
        "map_receipt_journal_sha256",
        "map_retry_download_event_count",
        "map_retry_download_payload_bytes",
        "query_first_pass_receipt_count",
        "query_first_pass_complete_event_sha256",
        "query_journal_final_event_sha256",
        "query_journal_path",
        "query_journal_sha256",
        "query_retry_download_event_count",
        "query_retry_download_payload_bytes",
        "query_second_pass_receipt_count",
        "raw_payload_persistent_bytes",
        "registration_execution_count",
        "schema",
        "successful_download_event_count",
        "successful_payload_bytes",
        "target_frozen_event_sha256",
        "unique_allowlist_object_count",
    }
    aggregate = _load_json(root / "LIDAR_DOWNLOAD_AUDIT.json")
    _same(set(aggregate), aggregate_fields, "aggregate audit exact schema")
    map_audit = map_state["audit"]
    query_audit = query_state["audit"]
    expected_aggregate = {
        "schema": AGGREGATE_AUDIT_SCHEMA,
        "successful_download_event_count": map_audit[
            "successful_download_event_count"
        ]
        + query_audit["successful_download_event_count"],
        "successful_payload_bytes": map_audit["successful_payload_bytes"]
        + query_audit["successful_payload_bytes"],
        "committed_payload_event_count": len(rows),
        "committed_payload_bytes": sum(int(row["remote_size_bytes"]) for row in rows),
        "unique_allowlist_object_count": len(map_allowlist) + len(query_allowlist),
        "raw_payload_persistent_bytes": 0,
        "map_receipt_journal_path": map_audit["receipt_checkpoint_journal_path"],
        "map_receipt_journal_sha256": map_audit[
            "receipt_checkpoint_journal_sha256"
        ],
        "map_receipt_final_chain_sha256": map_audit[
            "receipt_checkpoint_final_chain_sha256"
        ],
        "map_retry_download_event_count": map_audit["retry_download_event_count"],
        "map_retry_download_payload_bytes": map_audit[
            "retry_download_payload_bytes"
        ],
        "query_journal_path": query_audit["query_journal_path"],
        "query_journal_sha256": query_audit["query_journal_sha256"],
        "query_journal_final_event_sha256": query_audit[
            "query_journal_final_event_sha256"
        ],
        "query_retry_download_event_count": query_audit[
            "retry_download_event_count"
        ],
        "query_retry_download_payload_bytes": query_audit[
            "retry_download_payload_bytes"
        ],
        "target_frozen_event_sha256": query_audit["target_frozen_event_sha256"],
        "query_first_pass_complete_event_sha256": query_audit[
            "first_pass_complete_event_sha256"
        ],
        "R14_selection_frozen_event_sha256": query_audit[
            "R14_selection_frozen_event_sha256"
        ],
        "map_receipt_count": len(map_allowlist),
        "query_first_pass_receipt_count": len(query_allowlist),
        "query_second_pass_receipt_count": len(selected),
        "registration_execution_count": 0,
    }
    _same(aggregate, expected_aggregate, "aggregate audit replay")
    return actual_identity


def _verify_canonical_inputs(
    root: Path,
    runtime: Path,
    selected: Sequence[Mapping[str, Any]],
    authority: PreparationVerificationAuthority,
    candidate_scans: Sequence[Mapping[str, Any]],
    reference_poses: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    target_sha: str,
    target_path: str,
    target_size: int,
    target_count: int,
    preprocessing_sha: str,
    selection_sha: str,
) -> list[dict[str, str]]:
    rows = _csv_rows(root / "canonical_input_manifest.csv", CANONICAL_INPUT_FIELDS)
    _same(len(rows), 100, "canonical input count")
    by_snapshot = {row["snapshot_id"]: row for row in selected}
    if len(by_snapshot) != 100:
        _fail("selected snapshot IDs are not unique")
    expected_t = authority.expected_t_reference_sha256_by_object_key
    candidate_by_key = {str(row["object_key"]): row for row in candidate_scans}
    for expected_index, row in enumerate(rows):
        _same(_parse_int(row["selection_index"], "canonical selection index"), expected_index, "canonical order")
        snapshot = by_snapshot.get(row["snapshot_id"])
        if snapshot is None:
            _fail("canonical row names an unknown snapshot")
        for field in ("scene_label", "interval_id", "object_key"):
            _same(row[field], str(snapshot[field]), f"canonical snapshot {field}")
        source_path = _safe_runtime_file(runtime, row["source_points_path"], "canonical source")
        source, source_sha, source_size = _canonical_npy(source_path, shape_tail=(3,))
        candidate = candidate_by_key.get(row["object_key"])
        if candidate is None:
            _fail("canonical source is absent from candidate inventory")
        for field, expected in (
            ("source_points_sha256", source_sha),
            ("source_points_size_bytes", str(source_size)),
            ("source_point_count", str(source.shape[0])),
            ("target_map_path", target_path),
            ("target_map_sha256", target_sha),
            ("target_map_size_bytes", str(target_size)),
            ("target_point_count", str(target_count)),
            ("preprocessing_contract_sha256", preprocessing_sha),
            ("selection_contract_sha256", selection_sha),
            ("backend_parameter_contract_sha256", authority.expected_backend_parameter_sha256),
            ("future_open3d_source_sha256", source_sha),
            ("future_pcl_source_sha256", source_sha),
            ("future_open3d_target_sha256", target_sha),
            ("future_pcl_target_sha256", target_sha),
            ("byte_identical_for_both_backends", "True"),
        ):
            _same(row[field], expected, f"canonical bundle {field}")
        _same(
            int(source.shape[0]),
            int(candidate["finite_source_point_count"]),
            "canonical/candidate finite source point count",
        )
        transform_path = _safe_runtime_file(runtime, row["T_reference_path"], "T_reference")
        transform, transform_sha, transform_size = _canonical_npy(
            transform_path, shape_tail=(4,)
        )
        if transform.shape != (4, 4):
            _fail("T_reference is not 4x4")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-12):
            _fail("T_reference homogeneous row differs")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-10):
            _fail("T_reference rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-10):
            _fail("T_reference rotation determinant differs")
        expected_translation, expected_quaternion = _independent_exact_pose(
            int(snapshot["selected_timestamp_us"]), *reference_poses
        )
        expected_transform = np.eye(4, dtype=np.float64)
        expected_transform[:3, :3] = Rotation.from_quat(
            expected_quaternion
        ).as_matrix()
        expected_transform[:3, 3] = expected_translation
        if not np.allclose(
            transform, expected_transform, rtol=0.0, atol=1.0e-10
        ):
            _fail("T_reference differs from the exact authoritative lidar_poses row")
        _same(row["T_reference_sha256"], transform_sha, "T_reference SHA")
        _same(row["T_reference_size_bytes"], str(transform_size), "T_reference size")
        if expected_t is not None:
            _same(
                transform_sha,
                expected_t.get(row["object_key"]),
                "independently derived T_reference SHA",
            )
        metadata_path = _safe_runtime_file(runtime, row["snapshot_metadata_path"], "snapshot metadata")
        metadata = _load_json(metadata_path)
        _same(_sha256_file(metadata_path), row["snapshot_metadata_sha256"], "snapshot metadata SHA")
        for field, expected in (
            ("snapshot_id", row["snapshot_id"]),
            ("object_key", row["object_key"]),
            ("source_points_sha256", source_sha),
            ("T_reference_sha256", transform_sha),
            ("target_map_sha256", target_sha),
        ):
            _same(metadata.get(field), expected, f"snapshot metadata {field}")
        bundle_core = {field: value for field, value in row.items() if field != "bundle_sha256"}
        _same(row["bundle_sha256"], _hash_json(bundle_core), "canonical bundle row SHA")
    return rows


def _verify_canonical_source_witness(
    root: Path,
    selected: Sequence[Mapping[str, Any]],
    canonical_rows: Sequence[Mapping[str, str]],
    receipts: Mapping[tuple[str, str], Mapping[str, str]],
    *,
    preprocessing_sha: str,
    gt_sha: str,
    extrinsic_sha: str,
    witness_sha: str,
) -> None:
    rows = _csv_rows(
        root / "canonical_source_verification.csv",
        CANONICAL_SOURCE_VERIFICATION_FIELDS,
    )
    _same(len(rows), 100, "canonical source witness count")
    canonical_by_snapshot = {row["snapshot_id"]: row for row in canonical_rows}
    for expected_index, (row, snapshot) in enumerate(zip(rows, selected)):
        canonical = canonical_by_snapshot.get(str(snapshot["snapshot_id"]))
        if canonical is None:
            _fail("canonical witness names an unknown canonical bundle")
        first = receipts[("QUERY_GEOMETRY_FIRST_PASS", str(snapshot["object_key"]))]
        second = receipts[("QUERY_CANONICAL_SECOND_PASS", str(snapshot["object_key"]))]
        integer_fields = (
            "selection_index",
            "raw_size_bytes",
            "raw_point_count",
            "nonfinite_excluded_count",
            "range_excluded_count",
            "post_filter_point_count",
            "source_voxel_reduced_count",
            "canonical_source_point_count",
        )
        parsed = {
            field: _parse_int(row[field], f"canonical witness {field}")
            for field in integer_fields
        }
        expected_bindings = {
            "selection_index": expected_index,
            "snapshot_id": str(snapshot["snapshot_id"]),
            "object_key": str(snapshot["object_key"]),
            "raw_payload_sha256": second["local_temporary_sha256"],
            "raw_size_bytes": int(second["remote_size_bytes"]),
            "first_pass_receipt_sha256": first["receipt_sha256"],
            "second_pass_receipt_sha256": second["receipt_sha256"],
            "preprocessing_contract_sha256": preprocessing_sha,
            "gt_sha256": gt_sha,
            "extrinsic_sha256": extrinsic_sha,
            "witness_implementation_sha256": witness_sha,
            "producer_source_sha256": canonical["source_points_sha256"],
            "independent_source_sha256": canonical["source_points_sha256"],
            "producer_T_reference_sha256": canonical["T_reference_sha256"],
            "independent_T_reference_sha256": canonical["T_reference_sha256"],
            "canonical_source_point_count": int(canonical["source_point_count"]),
            "verification_status": (
                "PASS_DUAL_PATH_BYTE_IDENTITY_SHARED_FROZEN_PRIMITIVES"
            ),
        }
        for field, expected in expected_bindings.items():
            actual: Any = parsed[field] if field in parsed else row[field]
            _same(actual, expected, f"canonical source witness {field}")
        _same(
            first["local_temporary_sha256"],
            second["local_temporary_sha256"],
            "canonical witness two-pass raw SHA",
        )
        _same(
            second["processing_result_sha256"],
            canonical["source_points_sha256"],
            "second-pass processing result/source SHA",
        )
        if parsed["raw_size_bytes"] != 24 * parsed["raw_point_count"]:
            _fail("canonical witness raw point/byte count does not close")
        if parsed["raw_point_count"] != (
            parsed["nonfinite_excluded_count"]
            + parsed["range_excluded_count"]
            + parsed["post_filter_point_count"]
        ):
            _fail("canonical witness filter counts do not close")
        if parsed["canonical_source_point_count"] != (
            parsed["post_filter_point_count"]
            - parsed["source_voxel_reduced_count"]
        ):
            _fail("canonical witness voxel counts do not close")
        core: dict[str, Any] = {}
        for field in CANONICAL_SOURCE_VERIFICATION_FIELDS:
            if field == "canonical_source_verification_row_sha256":
                continue
            core[field] = parsed[field] if field in parsed else row[field]
        _same(
            row["canonical_source_verification_row_sha256"],
            _hash_json(core),
            "canonical source witness row SHA",
        )


def _verify_uncertainty(root: Path) -> None:
    rows = _csv_rows(root / "boreas_v2_stage2_uncertainty_budget.csv", UNCERTAINTY_FIELDS)
    json_value = _load_json(root / "boreas_v2_stage2_uncertainty_budget.json")
    _same(set(json_value), {"rows"}, "uncertainty JSON exact schema")
    _same(json_value["rows"], rows, "uncertainty CSV/JSON projection")
    by_component = {row["component"]: row for row in rows}
    if len(by_component) != len(rows):
        _fail("uncertainty components are duplicated")
    if not REQUIRED_UNKNOWN_COMPONENTS <= set(by_component):
        _fail("uncertainty inventory lost a required UNKNOWN component")
    for component in REQUIRED_UNKNOWN_COMPONENTS:
        row = by_component[component]
        _same(row["value"], "UNKNOWN", f"uncertainty UNKNOWN value {component}")
        _same(row["uncertainty_type"], "UNKNOWN", f"uncertainty type {component}")
        _same(row["status"], "UNKNOWN", f"uncertainty status {component}")
    if any(
        row["uncertainty_type"] == "UNKNOWN" and row["value"] != "UNKNOWN"
        for row in rows
    ):
        _fail("UNKNOWN uncertainty was replaced by a numeric value")


def _walk_forbidden(value: Any, *, path: str = "root") -> None:
    forbidden_exact = {
        "translation_displacement",
        "translation_displacement_m",
        "rotation_displacement",
        "rotation_displacement_rad",
        "final_transform",
        "final_residual",
        "correspondence_turnover",
        "fitness",
        "solver_status",
        "registration_result",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in forbidden_exact:
                _fail(f"Stage-3 registration result field exists: {path}.{key}")
            _walk_forbidden(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, path=f"{path}[{index}]")


def _verify_no_registration(root: Path, runtime: Path) -> None:
    attestation = _load_json(root / "NO_ICP_ATTESTATION.json")
    expected_fields = {
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
    _same(set(attestation), expected_fields, "NO-ICP exact attestation schema")
    for field in (
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "registration_execution_count",
        "real_trial_result_count",
        "actual_open3d_trials",
        "actual_pcl_trials",
        "actual_trials",
        "estimated_transform_count",
        "estimated_transform_file_count",
        "structured_result_scan_error_count",
    ):
        _same(attestation[field], 0, f"NO-ICP {field}")
    for field in (
        "estimated_transform_evidence",
        "estimated_transform_files",
        "structured_result_scan_error_files",
    ):
        _same(attestation[field], [], f"NO-ICP {field}")
    _same(attestation["pass"], True, "NO-ICP pass")
    _same(attestation["status"], "PASS", "NO-ICP status")
    forbidden_names = re.compile(
        r"(?:registration[_-]?result|open3d[_-]?result|pcl[_-]?result|icp[_-]?result)", re.I
    )
    for tree in (root, runtime):
        for path in tree.rglob("*"):
            if path.is_symlink():
                _fail(f"closure/runtime symlink exists: {path}")
            if forbidden_names.search(path.name):
                _fail(f"Stage-3 result artifact exists: {path}")
            if path.is_file() and path.suffix == ".json":
                _walk_forbidden(_load_json(path))


def _verify_readiness(root: Path) -> Mapping[str, Any]:
    readiness = _load_json(root / "boreas_v2_stage2_readiness.json")
    expected = {
        "BOREAS_EXTERNAL_V2_STAGE2_READY": True,
        "READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION": True,
        "weak_snapshot_count": 50,
        "rich_snapshot_count": 50,
        "snapshot_count": 100,
        "planned_future_open3d_trials": 100,
        "planned_future_pcl_trials": 100,
        "planned_future_trials": 200,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_trials": 0,
        "registration_execution_count": 0,
        "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
        "REAL_REGISTRATION_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "schema": "zprm.boreas.v2.stage2.readiness.v1",
    }
    _same(set(readiness), set(expected), "readiness exact schema")
    for field, value in expected.items():
        _same(readiness.get(field), value, f"readiness {field}")
    return readiness


def _production_authority(authority: PreparationVerificationAuthority) -> bool:
    return (
        authority.expected_map_sequence_id == EXPECTED_MAP_SEQUENCE
        and authority.expected_query_sequence_id == EXPECTED_QUERY_SEQUENCE
        and authority.expected_candidate_scan_count == 11_859
        and authority.expected_candidate_interval_count == 246
        and authority.expected_map_object_count == 8_202
        and authority.selection_parameter_authority
        == "BOREAS_V2_STAGE2_PREREGISTRATION"
    )


def _verify_safe_fixture_replay_evidence(
    *,
    runtime: Path,
    repository: Path,
    status: Mapping[str, Any],
) -> tuple[Path, Path, Mapping[str, Any]]:
    marker_relative = PurePosixPath(
        str(status["safe_fixture_replay_marker_relative_path"])
    )
    event_relative = PurePosixPath(
        str(status["safe_fixture_replay_event_log_relative_path"])
    )
    if marker_relative != PurePosixPath(
        "evidence/full_pytest_safe_fixture_replay_marker.json"
    ) or event_relative != PurePosixPath(
        "evidence/full_pytest_safe_fixture_replay_events.jsonl"
    ):
        _fail("formal safe-fixture replay evidence paths differ")
    marker_path = _safe_runtime_file(
        runtime, marker_relative.as_posix(), "formal safe-fixture replay marker"
    )
    event_path = _safe_runtime_file(
        runtime, event_relative.as_posix(), "formal safe-fixture replay event log"
    )
    for path, label in (
        (marker_path, "formal safe-fixture replay marker"),
        (event_path, "formal safe-fixture replay event log"),
    ):
        metadata = os.lstat(path)
        if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
            _fail(f"{label} is unsafe")
    _same(
        status["safe_fixture_replay_marker_sha256"],
        _sha256_file(marker_path),
        "formal safe-fixture replay marker SHA",
    )
    _same(
        status["safe_fixture_replay_event_log_sha256"],
        _sha256_file(event_path),
        "formal safe-fixture replay event-log SHA",
    )
    marker = _load_json(marker_path)
    marker_fields = {
        "event_log_path",
        "formal_full_pytest",
        "formal_python_executable",
        "git_head",
        "marker_payload_sha256",
        "no_registration",
        "purpose",
        "registration_execution_count",
        "repository_root",
        "runtime_lifecycle_fixture_sha256",
        "safe_fixture_replay_sha256",
        "schema",
    }
    _same(set(marker), marker_fields, "formal safe-fixture replay marker schema")
    unsigned_marker = dict(marker)
    marker_claim = unsigned_marker.pop("marker_payload_sha256", None)
    _same(marker_claim, _compact_hash(unsigned_marker), "formal replay marker self-hash")
    _same(marker["schema"], SAFE_REPLAY_MARKER_SCHEMA, "formal replay marker schema")
    _same(marker["purpose"], SAFE_REPLAY_MARKER_PURPOSE, "formal replay marker purpose")
    if marker["formal_full_pytest"] is not True:
        _fail("formal replay marker full-pytest flag differs")
    if marker["no_registration"] is not True:
        _fail("formal replay marker NO-registration flag differs")
    if (
        type(marker["registration_execution_count"]) is not int
        or marker["registration_execution_count"] != 0
    ):
        _fail("formal replay marker registration count differs")
    _same(marker["repository_root"], str(repository), "formal replay marker repository")
    _same(marker["git_head"], status["git_head"], "formal replay marker Git HEAD")
    _same(
        marker["formal_python_executable"],
        status["python_executable"],
        "formal replay marker Python",
    )
    _same(marker["event_log_path"], str(event_path), "formal replay marker event path")
    implementation_paths = {
        "safe_fixture_replay_sha256": repository
        / "src/phase_a_harness/real_data_preparation/stage2_safe_fixture_replay.py",
        "runtime_lifecycle_fixture_sha256": repository
        / "src/phase_a_harness/runtime_lifecycle_fixture.py",
    }
    for field, path in implementation_paths.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or os.lstat(path).st_nlink != 1
        ):
            _fail(f"formal replay implementation is unsafe: {field}")
        _same(marker[field], _sha256_file(path), f"formal replay implementation {field}")

    payload = event_path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        _fail("formal safe-fixture replay event log is partial")
    event_fields = {
        "actual_registration_execution_count",
        "backend",
        "condition",
        "event_sha256",
        "event_version",
        "planned_trial_id",
        "previous_event_sha256",
        "process_id",
        "replay_catalog_sha256",
        "replay_result_sha256",
        "safe_simulated_materialization_count",
        "sequence",
        "snapshot_id",
    }
    conditions = {
        "FIXTURE_IDENTITY",
        "FIXTURE_NONIDENTITY_REFERENCE",
        "FIXTURE_NO_CORRESPONDENCE",
    }
    previous = ZERO_SHA256
    counts: Counter[str] = Counter()
    event_count = 0
    for event_count, raw in enumerate(payload.splitlines(), start=1):
        try:
            event = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise BoreasStage2PreparationVerificationError(
                "formal safe-fixture replay event is invalid JSON"
            ) from error
        if type(event) is not dict or raw != _compact_json_bytes(event):
            _fail("formal safe-fixture replay event is noncanonical")
        unsigned_event = dict(event)
        event_claim = unsigned_event.pop("event_sha256", None)
        if (
            set(event) != event_fields
            or event.get("event_version") != SAFE_REPLAY_EVENT_SCHEMA
            or type(event.get("sequence")) is not int
            or event.get("sequence") != event_count
            or event.get("previous_event_sha256") != previous
            or type(event.get("actual_registration_execution_count")) is not int
            or event.get("actual_registration_execution_count") != 0
            or type(event.get("safe_simulated_materialization_count")) is not int
            or event.get("safe_simulated_materialization_count") != 1
            or event.get("backend") not in {OPEN3D_BACKEND, PCL_BACKEND}
            or event.get("condition") not in conditions
            or not isinstance(event.get("planned_trial_id"), str)
            or not event.get("planned_trial_id")
            or not isinstance(event.get("snapshot_id"), str)
            or not event.get("snapshot_id")
            or event.get("replay_catalog_sha256")
            != SAFE_REPLAY_CATALOG_SHA256
            or type(event.get("process_id")) is not int
            or event.get("process_id", 0) <= 0
            or SHA256_RE.fullmatch(str(event.get("replay_result_sha256", "")))
            is None
            or event_claim != _compact_hash(unsigned_event)
        ):
            _fail("formal safe-fixture replay event chain differs")
        previous = str(event_claim)
        counts[str(event["backend"])] += 1
    return marker_path, event_path, {
        "safe_fixture_replay_event_chain_head_sha256": previous,
        "safe_fixture_replay_event_count": event_count,
        "safe_fixture_replay_event_log_sha256": hashlib.sha256(payload).hexdigest(),
        "safe_simulated_materialization_count": event_count,
        "safe_simulated_open3d_materialization_count": counts[OPEN3D_BACKEND],
        "safe_simulated_pcl_materialization_count": counts[PCL_BACKEND],
    }


def _verify_full_test_status(
    runtime: Path, repository: Path, closure_root: Path
) -> Mapping[str, Any]:
    path = _safe_runtime_file(
        runtime, "evidence/full_test_status.json", "formal full-test status"
    )
    value = _load_json(path)
    core_fields = {
        "child_no_registration_guard_implementation_path",
        "child_no_registration_guard_implementation_sha256",
        "child_no_registration_plugin_path",
        "child_no_registration_plugin_sha256",
        "command",
        "controlled_environment",
        "git_head",
        "git_tree",
        "pyproject_sha256",
        "python_executable",
        "python_executable_sha256",
        "python_version",
        "safe_fixture_replay_event_log_relative_path",
        "safe_fixture_replay_marker_relative_path",
        "safe_fixture_replay_marker_sha256",
        "source_worktree_status_sha256",
        "tracked_file_inventory_sha256",
    }
    fields = core_fields | {
        "child_no_registration_attestation_relative_path",
        "child_no_registration_attestation_sha256",
        "collected",
        "errors",
        "exit_code",
        "failed",
        "junit_relative_path",
        "junit_sha256",
        "passed",
        "result_receipt_relative_path",
        "result_receipt_sha256",
        "safe_fixture_replay_event_log_sha256",
        "schema",
        "skipped",
        "skipped_tests",
        "status",
        "stderr_sha256",
        "stdout_sha256",
        "test_case_ids_sha256",
    }
    _same(set(value), fields, "formal full-test exact schema")
    python = Path(str(value["python_executable"]))
    expected_python = Path(
        "/home/lj/.local/share/degen-lio-micromamba/envs/"
        "degen-lio-zprm-py311/bin/python3.11"
    )
    if (
        python != expected_python
        or not python.is_file()
        or python.is_symlink()
        or python.resolve(strict=True) != python
    ):
        _fail("formal full-test Python path differs")
    _same(value["python_executable_sha256"], _sha256_file(python), "formal Python SHA")
    version = subprocess.run(
        [str(python), "--version"],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _same(version.returncode, 0, "formal Python version command")
    _same(value["python_version"], version.stdout.decode().strip(), "formal Python version")
    if not str(value["python_version"]).startswith("Python 3.11."):
        _fail("formal full-test interpreter is not Python 3.11")
    junit_relative = PurePosixPath(str(value["junit_relative_path"]))
    if junit_relative != PurePosixPath("evidence/full_pytest_junit.xml"):
        _fail("formal JUnit path differs")
    junit = _safe_runtime_file(runtime, junit_relative.as_posix(), "formal JUnit")
    _same(value["junit_sha256"], _sha256_file(junit), "formal JUnit SHA")
    plugin_relative = PurePosixPath(
        str(value["child_no_registration_plugin_path"])
    )
    guard_relative = PurePosixPath(
        str(value["child_no_registration_guard_implementation_path"])
    )
    if (
        plugin_relative
        != PurePosixPath(
            "src/phase_a_harness/real_data_preparation/"
            "stage2_pytest_no_registration.py"
        )
        or guard_relative
        != PurePosixPath("src/phase_a_harness/real_data_preparation/guard.py")
    ):
        _fail("formal child NO-ICP implementation paths differ")
    plugin_path = repository.joinpath(*plugin_relative.parts)
    guard_path = repository.joinpath(*guard_relative.parts)
    _same(
        value["child_no_registration_plugin_sha256"],
        _sha256_file(plugin_path),
        "formal child NO-ICP plugin SHA",
    )
    _same(
        value["child_no_registration_guard_implementation_sha256"],
        _sha256_file(guard_path),
        "formal child NO-ICP guard SHA",
    )
    marker_path, event_log_path, replay_projection = (
        _verify_safe_fixture_replay_evidence(
            runtime=runtime,
            repository=repository,
            status=value,
        )
    )
    child_relative = PurePosixPath(
        str(value["child_no_registration_attestation_relative_path"])
    )
    if child_relative != PurePosixPath(
        "evidence/full_pytest_no_registration_attestation.json"
    ):
        _fail("formal child NO-ICP attestation path differs")
    child_attestation = _safe_runtime_file(
        runtime, child_relative.as_posix(), "formal child NO-ICP attestation"
    )
    _same(
        value["child_no_registration_attestation_sha256"],
        _sha256_file(child_attestation),
        "formal child NO-ICP attestation SHA",
    )
    child = _load_json(child_attestation)
    child_fields = {
        "actual_open3d_trials",
        "actual_pcl_trials",
        "actual_trials",
        "estimated_transform_count",
        "estimated_transform_evidence",
        "estimated_transform_file_count",
        "estimated_transform_files",
        "guard_implementation_path",
        "guard_implementation_sha256",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pass",
        "pcl_cli_invocation_count",
        "plugin_path",
        "plugin_sha256",
        "pytest_exit_status_at_attestation",
        "real_trial_result_count",
        "registration_execution_count",
        "safe_fixture_replay_catalog_sha256",
        "safe_fixture_replay_event_chain_head_sha256",
        "safe_fixture_replay_event_count",
        "safe_fixture_replay_event_log_path",
        "safe_fixture_replay_event_log_sha256",
        "safe_fixture_replay_event_schema",
        "safe_fixture_replay_marker_path",
        "safe_fixture_replay_marker_sha256",
        "safe_simulated_materialization_count",
        "safe_simulated_open3d_materialization_count",
        "safe_simulated_pcl_materialization_count",
        "schema",
        "status",
        "structured_result_scan_error_count",
        "structured_result_scan_error_files",
    }
    _same(set(child), child_fields, "formal child NO-ICP exact schema")
    for field in (
        "actual_open3d_trials",
        "actual_pcl_trials",
        "actual_trials",
        "estimated_transform_count",
        "estimated_transform_file_count",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pcl_cli_invocation_count",
        "pytest_exit_status_at_attestation",
        "real_trial_result_count",
        "registration_execution_count",
        "structured_result_scan_error_count",
    ):
        if type(child[field]) is not int:
            _fail(f"formal child NO-ICP {field} is not an integer")
        _same(child[field], 0, f"formal child NO-ICP {field}")
    for field in (
        "estimated_transform_evidence",
        "estimated_transform_files",
        "structured_result_scan_error_files",
    ):
        _same(child[field], [], f"formal child NO-ICP {field}")
    if child["pass"] is not True:
        _fail("formal child NO-ICP pass differs")
    _same(child["status"], "PASS", "formal child NO-ICP status")
    _same(
        child["schema"],
        "zprm.boreas.v2.stage2.full_test_child_no_registration.v2",
        "formal child NO-ICP schema",
    )
    _same(child["plugin_path"], str(plugin_path), "formal child plugin path")
    _same(child["plugin_sha256"], _sha256_file(plugin_path), "formal child plugin SHA")
    _same(child["guard_implementation_path"], str(guard_path), "formal child guard path")
    _same(child["guard_implementation_sha256"], _sha256_file(guard_path), "formal child guard SHA")
    _same(
        child["safe_fixture_replay_catalog_sha256"],
        SAFE_REPLAY_CATALOG_SHA256,
        "formal child safe-replay catalog SHA",
    )
    _same(
        child["safe_fixture_replay_event_schema"],
        SAFE_REPLAY_EVENT_SCHEMA,
        "formal child safe-replay event schema",
    )
    _same(
        child["safe_fixture_replay_marker_path"],
        str(marker_path),
        "formal child safe-replay marker path",
    )
    _same(
        child["safe_fixture_replay_marker_sha256"],
        _sha256_file(marker_path),
        "formal child safe-replay marker SHA",
    )
    _same(
        child["safe_fixture_replay_event_log_path"],
        str(event_log_path),
        "formal child safe-replay event path",
    )
    for field, expected in replay_projection.items():
        if field.endswith("_count") and type(child[field]) is not int:
            _fail(f"formal child safe-replay {field} is not an integer")
        _same(child[field], expected, f"formal child safe-replay {field}")
    try:
        xml_root = ET.parse(junit).getroot()
    except (ET.ParseError, OSError) as error:
        raise BoreasStage2PreparationVerificationError(
            "formal JUnit is invalid"
        ) from error
    cases = list(xml_root.iter("testcase"))
    identities = [
        f"{case.get('classname', '')}::{case.get('name', '')}" for case in cases
    ]
    if (
        len(identities) != len(set(identities))
        or any(identity.startswith("::") or identity.endswith("::") for identity in identities)
    ):
        _fail("formal JUnit test identities are empty or duplicated")
    skipped_rows = sorted(
        (
            identities[index],
            str(skipped.get("message", "")),
        )
        for index, case in enumerate(cases)
        if (skipped := case.find("skipped")) is not None
    )
    expected_skips = (
        ("tests.test_formal_semantics_equivalence::test_formal_snapshot_validator_scientific_body_is_baseline_exact", "source-only ZIP excludes the historical Git baseline object"),
        ("tests.test_formal_semantics_equivalence::test_reader_policy_and_authentication_primitives_are_static_baseline_equivalent", "source-only ZIP excludes the historical Git baseline object"),
        ("tests.test_full_synthetic_protocol_assets_runner::test_authorization_rejects_forged_gate_report", "source-only ZIP excludes the historical Phase B raw results"),
        ("tests.test_full_synthetic_protocol_assets_runner::test_phase_a_ideal_import_is_read_only_and_complete", "source-only ZIP excludes the historical formal Phase A results"),
        ("tests.test_full_synthetic_protocol_assets_runner::test_phase_b_overlap_retains_all_published_snapshot_and_trial_ids", "source-only ZIP excludes the historical Phase B tag object"),
        ("tests.test_runtime_lifecycle_qualification::test_fixed_pcl_v3_input_inventory_matches_contract", "/tmp/synthetic_confirmatory_v2_pcl_v3_requalification: source-only package: external historical qualification bundle unavailable"),
        ("tests.test_synthetic_confirmatory_v2::test_frozen_model_and_backend_implementations_are_unchanged", "source-only ZIP excludes the historical v2 Git object"),
        ("tests.test_synthetic_confirmatory_v2::test_v1_and_v2_manifests_are_explicitly_version_selected", "source-only ZIP excludes the historical v2 Git object"),
        ("tests.test_synthetic_confirmatory_v2::test_v1_failure_archive_is_preserved", "source-only ZIP excludes the historical v1 failure bundle"),
        ("tests.test_synthetic_confirmatory_v2::test_v2_static_seed_provenance_audit_is_exact_and_constructor_free", "source-only ZIP excludes the historical v2 Git object"),
    )
    _same(tuple(skipped_rows), expected_skips, "formal test exact skip identities/reasons")
    counts = {
        "collected": len(cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "failed": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
    }
    counts["passed"] = (
        counts["collected"]
        - counts["skipped"]
        - counts["failed"]
        - counts["errors"]
    )
    for field, expected in counts.items():
        _same(value[field], expected, f"formal test {field}")
    _same(
        value["skipped_tests"],
        [{"node_id": node, "reason": reason} for node, reason in skipped_rows],
        "formal skipped-test projection",
    )
    _same(
        value["test_case_ids_sha256"],
        hashlib.sha256(_canonical_json_bytes(identities)).hexdigest(),
        "formal test-case inventory SHA",
    )
    partial = runtime / "checkpoints/full_pytest_junit.xml.partial"
    expected_command = [
        str(python),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "phase_a_harness.real_data_preparation.stage2_pytest_no_registration",
        "-c",
        "pyproject.toml",
        f"--junitxml={partial}",
        "tests",
    ]
    _same(value["command"], expected_command, "formal full-test command")
    for field in ("stderr_sha256", "stdout_sha256"):
        _require_sha(value[field], f"formal test {field}")
    _same(value["schema"], "zprm.boreas.v2.stage2.full_test_status.v1", "formal test schema")
    _same(value["exit_code"], 0, "formal test exit code")
    _same(value["status"], "PASS", "formal test status")
    if (
        counts["collected"] < 934
        or counts["passed"] <= 0
        or counts["failed"]
        or counts["errors"]
    ):
        _fail("formal full-test evidence is not a clean nonempty run")
    controlled_environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTEST_ADDOPTS": "",
        "PYTHONHASHSEED": "0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(repository / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONSTARTUP": "",
        "PYTHONWARNINGS": "",
        "MAMBA_ROOT_PREFIX": "/home/lj/.local/share/degen-lio-micromamba",
        "TZ": "UTC",
        "ZPRM_REAL_DATA_PREP_NO_REGISTRATION": "1",
        "ZPRM_STAGE2_FORMAL_FULL_PYTEST": "1",
        "ZPRM_STAGE2_FORMAL_FULL_PYTEST_MARKER": str(marker_path),
        "ZPRM_STAGE2_FORMAL_FULL_PYTEST_MARKER_SHA256": _sha256_file(
            marker_path
        ),
        "ZPRM_STAGE2_PYTEST_GUARD_ATTESTATION": str(
            runtime / "checkpoints/full_pytest_no_registration_attestation.json.partial"
        ),
        "ZPRM_STAGE2_PYTEST_GUARD_SCAN_ROOT": str(
            runtime / "checkpoints/full_pytest_no_registration_scan_root"
        ),
    }
    _same(
        value["controlled_environment"],
        controlled_environment,
        "formal full-test controlled environment",
    )
    _same(
        value["pyproject_sha256"],
        _sha256_file(repository / "pyproject.toml"),
        "formal pytest configuration SHA",
    )
    _same(
        value["source_worktree_status_sha256"],
        hashlib.sha256(b"").hexdigest(),
        "formal pre-test clean worktree SHA",
    )

    result_relative = PurePosixPath(str(value["result_receipt_relative_path"]))
    if result_relative != PurePosixPath("checkpoints/full_test_gate_result.json"):
        _fail("formal full-test result-receipt path differs")
    result_path = _safe_runtime_file(
        runtime, result_relative.as_posix(), "formal full-test result receipt"
    )
    _same(
        value["result_receipt_sha256"],
        _sha256_file(result_path),
        "formal full-test result-receipt SHA",
    )
    result = _load_json(result_path)
    result_fields = core_fields | {
        "child_no_registration_attestation_sha256",
        "exit_code",
        "junit_sha256",
        "result_sha256",
        "safe_fixture_replay_event_log_sha256",
        "schema",
        "stderr_sha256",
        "stdout_sha256",
    }
    _same(set(result), result_fields, "formal full-test result exact schema")
    unsigned_result = dict(result)
    result_claim = unsigned_result.pop("result_sha256")
    _same(result_claim, _hash_json(unsigned_result), "formal full-test result self-hash")
    _same(result["schema"], "zprm.boreas.v2.stage2.full_test_gate_result.v1", "formal result schema")
    for field in core_fields:
        _same(result[field], value[field], f"formal status/result {field}")
    for field in (
        "child_no_registration_attestation_sha256",
        "exit_code",
        "junit_sha256",
        "safe_fixture_replay_event_log_sha256",
        "stderr_sha256",
        "stdout_sha256",
    ):
        _same(result[field], value[field], f"formal status/result {field}")

    def git(*arguments: str) -> bytes:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            _fail(f"formal test Git evidence failed: {' '.join(arguments)}")
        return result.stdout

    worktree = git("status", "--porcelain=v1", "--untracked-files=all")
    publication_roots: set[Path] = set()
    for line in worktree.decode("utf-8").splitlines():
        if not line.startswith("?? "):
            _fail("formal source worktree has a tracked or non-closure change")
        relative = Path(line[3:])
        parts = relative.parts
        if len(parts) < 3 or parts[0] != "frozen_assets":
            _fail("formal source worktree has an unrelated untracked file")
        directory = parts[1]
        if directory == "boreas_v2_stage2_preparation" or (
            directory.startswith(".boreas_v2_stage2_preparation.")
            and directory.endswith(".staging")
        ):
            publication_roots.add(repository / parts[0] / directory)
        else:
            _fail("formal source worktree has an unrelated frozen-assets file")
    for publication in publication_roots:
        if (
            publication.is_symlink()
            or not publication.is_dir()
            or publication.resolve(strict=True) != publication
            or {entry.name for entry in publication.iterdir()} != set(REQUIRED_FILES)
        ):
            _fail("formal publication exception has an open inventory")
        for name in REQUIRED_FILES:
            published = publication / name
            reference = closure_root / name
            if (
                published.is_symlink()
                or not published.is_file()
                or os.lstat(published).st_nlink != 1
                or _sha256_file(published) != _sha256_file(reference)
            ):
                _fail("formal publication exception differs from verified closure")
    _verify_publication_git_state(
        repository=repository,
        closure_root=closure_root,
        source_head=str(value["git_head"]),
        source_tree=str(value["git_tree"]),
        tracked_file_inventory_sha256=str(
            value["tracked_file_inventory_sha256"]
        ),
    )
    return value


def _verify_publication_git_state(
    *,
    repository: Path,
    closure_root: Path,
    source_head: str,
    source_tree: str,
    tracked_file_inventory_sha256: str,
) -> None:
    """Independent source or publication-only descendant Git proof."""

    def git(*arguments: str) -> bytes:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            _fail(f"publication Git evidence failed: {' '.join(arguments)}")
        return result.stdout

    if re.fullmatch(r"[0-9a-f]{40}", source_head) is None:
        _fail("formal source Git HEAD is invalid")
    _same(
        source_tree,
        git("rev-parse", f"{source_head}^{{tree}}").decode().strip(),
        "formal source Git tree",
    )
    _same(
        tracked_file_inventory_sha256,
        hashlib.sha256(
            git("ls-tree", "-r", "--name-only", "-z", source_head)
        ).hexdigest(),
        "formal source tracked-file inventory",
    )
    current_head = git("rev-parse", "HEAD").decode().strip()
    if current_head == source_head:
        return
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_head, current_head],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ancestor.returncode != 0:
        _fail("current HEAD is not a publication-only descendant")
    diff_lines = git(
        "diff", "--name-status", "--no-renames", source_head, current_head
    ).decode("utf-8").splitlines()
    expected_publication = {
        f"frozen_assets/boreas_v2_stage2_preparation/{name}"
        for name in REQUIRED_FILES
    }
    actual_publication: set[str] = set()
    for line in diff_lines:
        parts = line.split("\t")
        if len(parts) != 2 or parts[0] != "A":
            _fail("publication descendant changed a pre-existing source path")
        actual_publication.add(parts[1])
    _same(
        actual_publication,
        expected_publication,
        "publication-only descendant exact closure diff",
    )
    for name in REQUIRED_FILES:
        tracked = repository / "frozen_assets/boreas_v2_stage2_preparation" / name
        if _sha256_file(tracked) != _sha256_file(closure_root / name):
            _fail("publication descendant closure bytes differ")


def _verify_summary_documents(
    root: Path,
    runtime: Path,
    authority: PreparationVerificationAuthority,
    *,
    readiness: Mapping[str, Any],
    scans: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    selected_intervals: Sequence[Mapping[str, Any]],
    selected_snapshots: Sequence[Mapping[str, Any]],
    canonical_rows: Sequence[Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
    target_sha256: str,
    target_path: str,
    target_size_bytes: int,
    target_count: int,
    target_replay: Mapping[str, Any],
) -> None:
    """Independently recompute the production 30-answer closure."""

    if not _production_authority(authority):
        return
    repository = authority.target_reducer_implementation_path.parents[3]
    summary = _load_json(root / "boreas_v2_stage2_summary.json")
    _same(
        set(summary),
        {
            "answers",
            "conclusion",
            "empirical_counts_and_bytes",
            "large_artifact_policy",
            "readiness",
            "resource_capacity_evidence",
            "schema",
        },
        "summary exact schema",
    )
    _same(summary["schema"], "zprm.boreas.v2.stage2.preparation_summary.v1", "summary schema")
    _same(summary["readiness"], readiness, "summary readiness")
    preprocessing = _load_json(authority.preprocessing_contract_path)
    aggregate = _load_json(root / "LIDAR_DOWNLOAD_AUDIT.json")
    map_audit = _load_json(root / "MAP_LIDAR_DOWNLOAD_AUDIT.json")
    query_audit = _load_json(root / "QUERY_LIDAR_DOWNLOAD_AUDIT.json")
    lineage = _load_json(root / "map_lineage_manifest.json")
    target = _load_json(root / "target_map_freeze_manifest.json")
    no_icp = _load_json(root / "NO_ICP_ATTESTATION.json")
    uncertainty = _load_json(root / "boreas_v2_stage2_uncertainty_budget.json")
    test_status = _verify_full_test_status(runtime, repository, root)

    from .stage2_disk_gate import _parse_event_log

    gate_path = _safe_runtime_file(
        runtime, "checkpoints/disk_gate_events.jsonl", "disk-gate event chain"
    )
    first_line = gate_path.read_bytes().splitlines()
    if not first_line:
        _fail("disk-gate event chain is empty")
    try:
        gate_config_sha = json.loads(first_line[0])["gate_config_sha256"]
        gate_events = _parse_event_log(
            gate_path.read_bytes(), expected_config_sha256=gate_config_sha
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BoreasStage2PreparationVerificationError(
            "disk-gate summary evidence is invalid"
        ) from error
    starts = [
        row for row in gate_events if row["operation"] == "START" and row["pass"] is True
    ]
    if not starts:
        _fail("summary lacks a passing Stage-2 START")
    weak_intervals = sum(row["scene_label"] == WEAK_LABEL for row in selected_intervals)
    rich_intervals = sum(row["scene_label"] == RICH_LABEL for row in selected_intervals)
    weak_snapshots = sum(row["scene_label"] == WEAK_LABEL for row in selected_snapshots)
    rich_snapshots = sum(row["scene_label"] == RICH_LABEL for row in selected_snapshots)
    deleted_bytes = int(map_audit["stream_deleted_raw_bytes"]) + int(
        query_audit["successful_payload_bytes"]
    )
    empirical = {
        "backend_input_sha_equal": all(
            row["future_open3d_source_sha256"] == row["future_pcl_source_sha256"]
            and row["future_open3d_target_sha256"] == row["future_pcl_target_sha256"]
            and row["backend_parameter_contract_sha256"]
            == authority.expected_backend_parameter_sha256
            and row["byte_identical_for_both_backends"] == "True"
            for row in canonical_rows
        ),
        "backend_parameter_contract_sha256": authority.expected_backend_parameter_sha256,
        "candidate_interval_count": len(intervals),
        "candidate_scan_count": len(scans),
        "geometry_invalid_scan_count": sum(row["geometry_valid"] is False for row in scans),
        "map_scan_count": int(lineage["source_object_count"]),
        "network_download_event_count": int(aggregate["successful_download_event_count"]),
        "network_download_bytes": int(aggregate["successful_payload_bytes"]),
        "primary_pair": selection_manifest["authority_bindings"]["primary_pair"],
        "query_contribution_count": int(lineage["query_contribution_count"]),
        "raw_payload_persistent_bytes": int(aggregate["raw_payload_persistent_bytes"]),
        "raw_payload_stream_deleted_bytes": deleted_bytes,
        "rich_interval_count": rich_intervals,
        "rich_snapshot_count": rich_snapshots,
        "stage2_start_free_bytes": int(starts[0]["current_free_bytes"]),
        "successful_start_attempt_count": len(starts),
        "target_map_physical_copy_count": int(target["physical_target_map_copy_count"]),
        "target_map_size_bytes": target_size_bytes,
        "unique_allowlist_object_count": int(aggregate["unique_allowlist_object_count"]),
        "weak_interval_count": weak_intervals,
        "weak_snapshot_count": weak_snapshots,
        "snapshot_count": len(selected_snapshots),
    }
    _same(summary["empirical_counts_and_bytes"], empirical, "summary empirical evidence")

    resources = summary["resource_capacity_evidence"]
    _verify_summary_resource_exact_schema(resources)
    context = resources["target_context"]
    context_paths = {
        "capacity_measurement": runtime / "evidence/target_context_capacity_measurement.json",
        "capacity_measurement_provenance": runtime
        / "evidence/target_context_capacity_measurement_provenance.json",
        "resource_plan": runtime / "evidence/target_context_resource_plan.json",
    }
    for name, path in context_paths.items():
        _same(context[name], _load_json(path), f"summary target-context {name}")
        _same(context[f"{name}_file_sha256"], _sha256_file(path), f"summary target-context {name} SHA")
    measurement = context["capacity_measurement"]
    plan = context["resource_plan"]
    provenance = context["capacity_measurement_provenance"]
    _same(
        set(measurement),
        {"measured_peak_memory_bytes", "measurement_method", "numpy_version", "schema", "scipy_version", "target_map_sha256", "target_point_count"},
        "target-context measurement schema",
    )
    _same(
        set(plan),
        {"estimated_peak_memory_bytes", "measurement_evidence_sha256", "minimum_live_available_memory_bytes", "numpy_version", "production_approved", "safety_margin_bytes", "schema", "scipy_version", "target_map_sha256", "target_point_count"},
        "target-context plan schema",
    )
    _same(
        set(provenance),
        {
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
        },
        "target-context provenance schema",
    )
    _same(measurement["target_map_sha256"], target_sha256, "target-context target SHA")
    _same(measurement["target_point_count"], target_count, "target-context target count")
    _same(plan["measurement_evidence_sha256"], _sha256_file(context_paths["capacity_measurement"]), "target-context measurement binding")
    _same(plan["estimated_peak_memory_bytes"], measurement["measured_peak_memory_bytes"], "target-context peak binding")
    _same(plan["minimum_live_available_memory_bytes"], plan["estimated_peak_memory_bytes"] + plan["safety_margin_bytes"], "target-context memory formula")
    if plan["safety_margin_bytes"] < 5 * 1024**3 or plan["production_approved"] is not True:
        _fail("target-context production capacity plan is not approved")
    expected_generator = repository / "scripts/run_boreas_v2_stage2_query.py"
    _same(provenance["schema"], "zprm.boreas.v2.stage2.target_context_measurement_provenance.v1", "target-context provenance schema version")
    _same(provenance["generator_path"], str(expected_generator), "target-context generator path")
    _same(provenance["generator_sha256"], _sha256_file(expected_generator), "target-context generator SHA")
    _same(provenance["measurement_evidence_sha256"], plan["measurement_evidence_sha256"], "target-context provenance binding")
    _same(provenance["measurement_method"], "MAX_RSS_PREPARE_NORMALS_KDTREE_SAME_PINNED_ENVIRONMENT", "target-context measurement method")
    _same(provenance["numpy_version"], measurement["numpy_version"], "target-context NumPy binding")
    _same(provenance["scipy_version"], measurement["scipy_version"], "target-context SciPy binding")
    _same(provenance["target_map_sha256"], target_sha256, "target-context provenance target SHA")
    _same(provenance["target_point_count"], target_count, "target-context provenance target count")
    if (
        not isinstance(provenance["platform"], str)
        or not provenance["platform"]
        or not isinstance(provenance["python_version"], str)
        or not provenance["python_version"]
    ):
        _fail("target-context platform/Python provenance is empty")
    python = Path(str(provenance["python_executable"]))
    if (
        not python.is_absolute()
        or python.is_symlink()
        or not python.is_file()
        or python.resolve(strict=True) != python
    ):
        _fail("target-context Python executable is unsafe")
    _same(
        provenance["python_executable_sha256"],
        _sha256_file(python),
        "target-context Python executable SHA",
    )
    python_identity = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            (
                "import json,platform;"
                "print(json.dumps({'platform':platform.platform(),"
                "'python_version':platform.python_version()},"
                "sort_keys=True,separators=(',',':')))"
            ),
        ],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        identity = json.loads(python_identity.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BoreasStage2PreparationVerificationError(
            "target-context Python identity is invalid"
        ) from error
    if python_identity.returncode != 0 or set(identity) != {
        "platform",
        "python_version",
    }:
        _fail("target-context Python identity command failed")
    _same(provenance["platform"], identity["platform"], "target-context platform")
    _same(
        provenance["python_version"],
        identity["python_version"],
        "target-context Python version",
    )
    _same(provenance["subprocess_mode"], "ISOLATED_FIXED_GENERATOR_BEFORE_ANY_QUERY_PAYLOAD", "target-context subprocess mode")
    target_for_probe = _safe_runtime_file(
        runtime, target_path, "target-context target"
    )
    if _live_available_memory_bytes() < plan["minimum_live_available_memory_bytes"]:
        _fail("live MemAvailable is below the target-context capacity plan")
    probe = subprocess.run(
        [
            str(python),
            str(expected_generator),
            "--_measure-target-context",
            "--_target-path",
            str(target_for_probe),
            "--_target-sha256",
            target_sha256,
            "--_target-point-count",
            str(target_count),
        ],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if probe.returncode != 0:
        _fail("independent target-context capacity probe failed")
    try:
        observed = json.loads(probe.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BoreasStage2PreparationVerificationError(
            "independent target-context capacity probe returned invalid JSON"
        ) from error
    _same(probe.stdout, _canonical_json_bytes(observed), "independent target-context probe canonical JSON")
    _same(set(observed), set(measurement), "independent target-context probe schema")
    for field in (
        "measurement_method",
        "numpy_version",
        "schema",
        "scipy_version",
        "target_map_sha256",
        "target_point_count",
    ):
        _same(observed[field], measurement[field], f"independent target-context probe {field}")
    observed_peak = observed["measured_peak_memory_bytes"]
    if (
        isinstance(observed_peak, bool)
        or not isinstance(observed_peak, int)
        or observed_peak <= 0
        or observed_peak > plan["estimated_peak_memory_bytes"] + 512 * 1024**2
        or observed_peak > plan["minimum_live_available_memory_bytes"]
    ):
        _fail("independent target-context probe exceeds the frozen capacity plan")

    reducer = resources["reducer"]
    reducer_paths = {
        "capacity_layout_verification": runtime / "checkpoints/reducer_capacity_layout_verification.json",
        "capacity_replay_binding": runtime / "checkpoints/reducer_capacity_replay_binding.json",
        "resource_plan": runtime / "checkpoints/reducer_resource_plan.json",
    }
    for name, path in reducer_paths.items():
        _same(reducer[name], _load_json(path), f"summary reducer {name}")
        _same(reducer[f"{name}_file_sha256"], _sha256_file(path), f"summary reducer {name} SHA")
    _same(
        reducer["resource_plan"]["plan_payload_sha256"],
        target_replay["resource_evidence"]["resource_plan_payload_sha256"],
        "summary independently verified reducer plan",
    )
    target_resource = resources["target_map_content_address"]
    runtime_target = _safe_runtime_file(runtime, target_path, "summary target map")
    metadata_path = runtime_target.parent / "metadata.json"
    _same(target_resource["target_map_path"], target_path, "summary target path")
    _same(target_resource["target_map_sha256"], target_sha256, "summary target SHA")
    _same(target_resource["metadata"], _load_json(metadata_path), "summary target metadata")
    _same(target_resource["metadata_file_sha256"], _sha256_file(metadata_path), "summary target metadata SHA")
    retention = resources["runtime_reverification_dependencies"]
    _same(
        retention["retention_policy"],
        "RETAIN_AUTHENTICATED_REPLAY_AND_LEDGER_AFTER_FREEZE_BECAUSE_FINAL_VERIFIER_REPLAYS_EXACT_TARGET_BYTES",
        "summary replay retention",
    )
    replay_path = _safe_runtime_file(runtime, retention["authenticated_replay_path"], "summary replay")
    ledger_path = _safe_runtime_file(runtime, retention["authenticated_replay_ledger_path"], "summary replay ledger")
    _same(retention["authenticated_replay_size_bytes"], replay_path.stat().st_size, "summary replay size")
    _same(retention["authenticated_replay_ledger_sha256"], _sha256_file(ledger_path), "summary replay ledger SHA")

    unknown = sorted(row["component"] for row in uncertainty["rows"] if row["value"] == "UNKNOWN")
    questions = [
        (1, "Stage-2 启动时实际可用磁盘多少 GiB？", {"bytes": empirical["stage2_start_free_bytes"], "GiB": empirical["stage2_start_free_bytes"] / 1024**3}),
        (2, "实际下载多少个 LiDAR objects？", {"unique_objects": empirical["unique_allowlist_object_count"], "successful_transfer_events": empirical["network_download_event_count"]}),
        (3, "实际网络下载多少 GB？", {"bytes": empirical["network_download_bytes"], "GB": empirical["network_download_bytes"] / 1e9}),
        (4, "raw payload 最终持久保留多少 GB？", {"bytes": empirical["raw_payload_persistent_bytes"], "GB": empirical["raw_payload_persistent_bytes"] / 1e9}),
        (5, "临时 payload 删除多少 GB？", {"bytes": deleted_bytes, "GB": deleted_bytes / 1e9}),
        (6, "是否始终使用 PRIMARY_PAIR？", {"yes": True, "pair": empirical["primary_pair"]}),
        (7, "preprocessing 参数最终是什么？", {"contract_sha256": _sha256_file(authority.preprocessing_contract_path), "filtering": preprocessing["filtering"], "source_downsampling": preprocessing["source_downsampling"], "map_accumulation": preprocessing["map_accumulation"], "target_geometry_analysis": preprocessing["target_geometry_analysis"], "crop_policy": preprocessing["crop_policy"], "dynamic_object_policy": preprocessing["dynamic_object_policy"]}),
        (8, "preprocessing 是否在任何 geometry metric 前冻结？", {"yes": preprocessing["execution_state_at_freeze"]["geometry_metric_execution_count"] == 0, "execution_state_at_freeze": preprocessing["execution_state_at_freeze"]}),
        (9, "deskew 怎么做？", preprocessing["deskew"]),
        (10, "是否使用任何 LiDAR odometry？", {"used": preprocessing["deskew"]["uses_lidar_odometry"]}),
        (11, "target map 使用多少 map scans？", empirical["map_scan_count"]),
        (12, "target map 是否只有一份？", {"yes": True, "physical_copy_count": empirical["target_map_physical_copy_count"]}),
        (13, "target map 大小多少 GiB？", {"bytes": target_size_bytes, "GiB": target_size_bytes / 1024**3}),
        (14, "query 是否 0 contribution to map？", {"yes": empirical["query_contribution_count"] == 0, "count": empirical["query_contribution_count"]}),
        (15, "query screening 处理多少 candidate scans？", len(scans)),
        (16, "geometry-invalid 排除多少？", empirical["geometry_invalid_scan_count"]),
        (17, "candidate 5 s intervals 有多少？", len(intervals)),
        (18, "是否成功获得 10 weak intervals？", {"yes": weak_intervals == 10, "count": weak_intervals}),
        (19, "是否成功获得 10 rich intervals？", {"yes": rich_intervals == 10, "count": rich_intervals}),
        (20, "weak 50 是否完成？", {"yes": weak_snapshots == 50, "count": weak_snapshots}),
        (21, "rich 50 是否完成？", {"yes": rich_snapshots == 50, "count": rich_snapshots}),
        (22, "canonical snapshot 是否精确 100 个？", {"yes": len(canonical_rows) == 100, "count": len(canonical_rows)}),
        (23, "future Open3D/PCL 输入 SHA 是否完全相同？", {"yes": empirical["backend_input_sha_equal"], "backend_parameter_contract_sha256": authority.expected_backend_parameter_sha256}),
        (24, "R14 是否成功冻结？", {"yes": selection_manifest["R14_frozen"] is True, "selection_manifest_sha256": selection_manifest["selection_manifest_sha256"]}),
        (25, "uncertainty 中仍有哪些 UNKNOWN？", unknown),
        (26, "ICP 是否严格为 0？", {"yes": True, "attestation": no_icp}),
        (27, "全量测试是否 0 failed？", {"yes": True, "test_status": test_status, "test_status_file_sha256": _sha256_file(runtime / "evidence/full_test_status.json")}),
        (28, "independent verifier 是否 PASS？", {"status": "PASS_REQUIRED_BEFORE_AND_AFTER_PUBLICATION"}),
        (29, "BOREAS_EXTERNAL_V2_STAGE2_READY 是否 true？", True),
        (30, "是否已经可以单独授权 Stage-3 的 200 次 registration？", {"ready_for_separate_authorization": True, "planned_future_trials": 200, "automatically_authorized": False}),
    ]
    expected_answers = [
        {"answer": answer, "item": item, "question": question}
        for item, question, answer in questions
    ]
    _same(summary["answers"], expected_answers, "summary 30 independently recomputed answers")
    _same(
        summary["conclusion"],
        "BOREAS_EXTERNAL_V2_STAGE2_READY=true; READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION=true; Stage-3 was not automatically authorized or executed.",
        "summary conclusion",
    )
    _same(
        summary["large_artifact_policy"],
        "TARGET_CANONICAL_BUNDLES_AND_AUTHENTICATED_REPLAY_REQUIRED_BY_FINAL_VERIFIER_REMAIN_RUNTIME_ONLY",
        "summary large-artifact policy",
    )
    lines = [
        "# Boreas External Validation v2 — Stage-2 preparation",
        "",
        "独立发布门要求候选、staging 和最终 frozen destination 均通过 verifier；本文件只在这些门全部启用的发布流程中生成。Stage-3 未获自动授权，也未执行。",
        "",
    ]
    for row in expected_answers:
        rendered = json.dumps(row["answer"], sort_keys=True, ensure_ascii=False)
        lines.extend((f"{row['item']}. {row['question']}", "", f"   `{rendered}`", ""))
    expected_markdown = ("\n".join(lines).rstrip() + "\n").encode("utf-8")
    _same(
        (root / "boreas_v2_stage2_summary.md").read_bytes(),
        expected_markdown,
        "summary Markdown projection",
    )


def _verify_summary_resource_exact_schema(resources: Any) -> None:
    """Close every embedded capacity/large-runtime resource field set."""

    if not isinstance(resources, Mapping):
        _fail("summary resource evidence is not an object")
    _same(
        set(resources),
        {
            "reducer",
            "runtime_reverification_dependencies",
            "target_context",
            "target_map_content_address",
        },
        "summary resource section",
    )
    expected = {
        "target_context": {
            "capacity_measurement",
            "capacity_measurement_file_sha256",
            "capacity_measurement_provenance",
            "capacity_measurement_provenance_file_sha256",
            "resource_plan",
            "resource_plan_file_sha256",
        },
        "reducer": {
            "capacity_layout_verification",
            "capacity_layout_verification_file_sha256",
            "capacity_replay_binding",
            "capacity_replay_binding_file_sha256",
            "resource_plan",
            "resource_plan_file_sha256",
        },
        "target_map_content_address": {
            "metadata",
            "metadata_file_sha256",
            "target_map_path",
            "target_map_sha256",
        },
        "runtime_reverification_dependencies": {
            "authenticated_replay_ledger_path",
            "authenticated_replay_ledger_sha256",
            "authenticated_replay_path",
            "authenticated_replay_size_bytes",
            "retention_policy",
        },
    }
    for section, fields in expected.items():
        value = resources[section]
        if not isinstance(value, Mapping):
            _fail(f"summary resource {section} is not an object")
        _same(set(value), fields, f"summary resource {section} exact schema")


def _verify_authorization_git_provenance(
    *,
    repository: Path,
    runtime: Path,
    authorization: Mapping[str, Any],
) -> None:
    _same(
        authorization.get("branch"),
        "run/boreas-v2-stage2-data-preparation",
        "download authorization branch",
    )
    commit = str(authorization.get("commit", ""))
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        _fail("download authorization commit is not an exact Git object ID")
    test_status = _load_json(
        _safe_runtime_file(
            runtime, "evidence/full_test_status.json", "formal full-test status"
        )
    )
    _same(
        commit,
        test_status.get("git_head"),
        "authorization/formal-test Git commit",
    )
    commit_object = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if commit_object.returncode != 0:
        _fail("authorization commit is not a repository commit object")
    tree = subprocess.run(
        ["git", "rev-parse", f"{commit}^{{tree}}"],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tree.returncode != 0:
        _fail("authorization commit tree cannot be resolved")
    _same(
        tree.stdout.decode().strip(),
        test_status.get("git_tree"),
        "authorization/formal-test Git tree",
    )


def _verify_download_authorization(
    root: Path,
    runtime: Path,
    authority: PreparationVerificationAuthority,
    map_allowlist: Sequence[Mapping[str, str]],
    query_allowlist: Sequence[Mapping[str, str]],
    *,
    preprocessing_sha256: str,
) -> None:
    authorization = _load_json(
        root / "boreas_v2_stage2_download_authorization.json"
    )
    _same(
        set(authorization),
        set(AUTHORIZATION_UNSIGNED_FIELDS) | {"authorization_payload_sha256"},
        "download authorization exact schema",
    )
    unsigned = dict(authorization)
    supplied = _require_sha(
        unsigned.pop("authorization_payload_sha256"),
        "download authorization payload SHA",
    )
    _same(supplied, _hash_json(unsigned), "download authorization self-hash")
    allowlist = list(map_allowlist) + list(query_allowlist)
    expected_scalars = {
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
        "REAL_REGISTRATION_AUTHORIZED": False,
        "STAGE2_DOWNLOAD_AUTHORIZED": True,
        "actual_trials": 0,
        "allowlist_object_count": len(allowlist),
        "allowlist_remote_bytes": sum(int(row["size_bytes"]) for row in allowlist),
        "allowlist_sha256": _sha256_file(authority.stage1_allowlist_path),
        "authorization_scope": "BOREAS_V2_STAGE2_LIDAR_DATA_PREPARATION_ONLY",
        "bucket": "boreas",
        "execution_mode": "STREAMING_LOW_DISK",
        "extrinsic_limitation": "PASS_WITH_DOCUMENTED_LIMITATION",
        "no_icp_guard_active": True,
        "no_registration_environment_active": True,
        "preprocessing_contract_sha256": preprocessing_sha256,
        "primary_pair": {
            "map_sequence_id": authority.expected_map_sequence_id,
            "query_sequence_id": authority.expected_query_sequence_id,
        },
        "primary_pair_sha256": _sha256_file(authority.pair_selection_path),
        "registration_execution_count": 0,
        "schema_version": "boreas_v2_stage2_download_authorization_v2",
        "self_hash_semantics": (
            "UNKEYED_SHA256_INTEGRITY_ONLY_LIVE_REAUTHENTICATION_REQUIRED"
        ),
        "stage1_manifest_file_sha256": _sha256_file(authority.stage1_manifest_path),
        "storage_manifest_file_sha256": _sha256_file(authority.storage_manifest_path),
        "worktree_clean_before_authorization": True,
    }
    for field, expected in expected_scalars.items():
        _same(authorization[field], expected, f"download authorization {field}")
    for field in (
        "preprocessing_contract_payload_sha256",
        "stage1_verification_report_sha256",
        "storage_budget_sha256",
        "storage_contract_sha256",
    ):
        _require_sha(authorization[field], f"download authorization {field}")
    if _production_authority(authority):
        repository = authority.target_reducer_implementation_path.parents[3]
        _verify_authorization_git_provenance(
            repository=repository,
            runtime=runtime,
            authorization=authorization,
        )
    else:
        _same(
            authorization["branch"],
            "run/boreas-v2-stage2-data-preparation",
            "download authorization branch",
        )
        if re.fullmatch(r"[0-9a-f]{40}", str(authorization["commit"])) is None:
            _fail("download authorization commit is not an exact Git object ID")
    for field in (
        "disk_free_bytes_at_authorization",
        "minimum_start_free_disk_bytes",
        "runtime_low_disk_watermark_bytes",
    ):
        if type(authorization[field]) is not int or authorization[field] < 0:
            _fail(f"download authorization {field} is invalid")
    if authorization["disk_free_bytes_at_authorization"] < authorization[
        "minimum_start_free_disk_bytes"
    ]:
        _fail("download authorization was minted below its start disk threshold")
    try:
        timestamp = datetime.fromisoformat(
            str(authorization["timestamp_utc"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise BoreasStage2PreparationVerificationError(
            "download authorization timestamp is invalid"
        ) from error
    if (
        not str(authorization["timestamp_utc"]).endswith("Z")
        or timestamp.tzinfo is None
        or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
    ):
        _fail("download authorization timestamp is not canonical UTC")
    roots = authorization["root_bindings"]
    if not isinstance(roots, dict) or set(roots) != {
        "repository",
        "stage1_data",
        "stage2_runtime",
        "stage2_temporary",
        "monitored_disk",
    }:
        _fail("download authorization root bindings differ")
    for name, binding in roots.items():
        if (
            not isinstance(binding, dict)
            or set(binding) != {"access", "path", "st_dev"}
            or binding["access"]
            != (
                "READ_EXECUTE"
                if name in {"repository", "stage1_data"}
                else "READ_WRITE_EXECUTE"
            )
            or not isinstance(binding["path"], str)
            or not Path(binding["path"]).is_absolute()
            or type(binding["st_dev"]) is not int
            or binding["st_dev"] < 0
        ):
            _fail(f"download authorization root binding differs: {name}")
    _same(
        Path(roots["stage2_runtime"]["path"]),
        runtime,
        "download authorization runtime binding",
    )
    temporary_root = Path(roots["stage2_temporary"]["path"])
    if (
        not temporary_root.is_absolute()
        or temporary_root.is_symlink()
        or not temporary_root.is_dir()
        or temporary_root.resolve(strict=True) != temporary_root
        or (temporary_root != runtime and runtime not in temporary_root.parents)
    ):
        _fail("download authorization temporary root is absent or unsafe")
    for path in temporary_root.rglob("*"):
        if path.is_symlink():
            _fail("Stage-2 temporary root contains a symlink")
        if path.is_file() and path.name != ".stage2_temporary_root.json":
            _fail("raw/decoded temporary payload persists after Stage-2 closure")
    authorization_path = Path(str(authorization["authorization_path"]))
    report_path = Path(str(authorization["stage1_verification_report_path"]))
    if not authorization_path.is_absolute() or not report_path.is_absolute():
        _fail("download authorization evidence paths are not absolute")
    if not isinstance(authorization["static_source_audit"], Mapping) or authorization[
        "static_source_audit"
    ].get("pass") is not True:
        _fail("download authorization static source audit is not PASS")
    if not isinstance(authorization["storage_verification"], Mapping):
        _fail("download authorization storage verification is not an object")


def _verify_selection_manifest(
    root: Path,
    selection_manifest: Mapping[str, Any],
    scans: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    selected_intervals: Sequence[Mapping[str, Any]],
    selected_snapshots: Sequence[Mapping[str, Any]],
    expected_contract: Mapping[str, Any],
) -> None:
    _same(
        selection_manifest.get("schema_version"),
        "boreas_v2_stage2_selection_manifest_v1",
        "selection manifest schema",
    )
    artifact_names = {
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
    _same(
        selection_manifest.get("artifact_sha256"),
        {name: _sha256_file(root / name) for name in sorted(artifact_names)},
        "selection artifact SHA bindings",
    )
    blind = selection_manifest.get("blind_selection")
    if not isinstance(blind, Mapping):
        _fail("blind selection manifest is absent")
    blind_unsigned = dict(blind)
    claimed = blind_unsigned.pop("blind_selection_manifest_sha256", None)
    _same(claimed, _hash_json(blind_unsigned), "blind selection self-hash")
    selection_sha = _hash_json(expected_contract)
    expected_counts = {
        "selection_contract": expected_contract,
        "selection_contract_sha256": selection_sha,
        "candidate_scan_count": len(scans),
        "geometry_valid_scan_count": sum(row["geometry_valid"] is True for row in scans),
        "candidate_scan_rows_sha256": _hash_json(
            [row["candidate_scan_row_sha256"] for row in scans]
        ),
        "candidate_interval_count": len(intervals),
        "eligible_interval_count": sum(row["interval_valid"] is True for row in intervals),
        "candidate_interval_rows_sha256": _hash_json(
            [row["candidate_interval_row_sha256"] for row in intervals]
        ),
        "selected_interval_count": len(selected_intervals),
        "selected_interval_rows_sha256": _hash_json(
            [row["selected_interval_row_sha256"] for row in selected_intervals]
        ),
        "weak_interval_count": 10,
        "rich_interval_count": 10,
        "snapshot_count": len(selected_snapshots),
        "weak_snapshot_count": 50,
        "rich_snapshot_count": 50,
        "selected_snapshot_rows_sha256": _hash_json(
            [row["selected_snapshot_row_sha256"] for row in selected_snapshots]
        ),
        "selector_visibility": "GEOMETRY_ONLY_NO_REGISTRATION_FIELDS",
        "registration_execution_count": 0,
        "r14_selection_frozen": True,
    }
    # Bindings contain the four immutable science/data hashes and are verified
    # separately against each candidate row.  Compare every other blind field.
    for field, expected in expected_counts.items():
        _same(blind.get(field), expected, f"blind selection {field}")
    _same(selection_manifest.get("R14_frozen"), True, "R14 frozen")
    _same(selection_manifest.get("registration_execution_count"), 0, "selection registration count")


def verify_boreas_v2_stage2_preparation(
    *,
    root: str | Path,
    runtime_root: str | Path,
    authority: PreparationVerificationAuthority,
) -> dict[str, Any]:
    """Independently verify the small closure and referenced runtime objects."""

    frozen = _safe_root(root, "frozen root")
    runtime = _safe_root(runtime_root, "runtime root")
    frozen_manifest = _verify_outer_closure(frozen)
    (
        selection_manifest,
        map_allowlist,
        query_allowlist,
        windows,
        preprocessing_sha,
        extrinsic_sha,
        witness_sha,
    ) = _verify_authority_bindings(frozen, authority, frozen_manifest)
    expected_contract = _expected_selection_contract(authority)
    selection_contract_sha = _hash_json(expected_contract)
    blind = selection_manifest.get("blind_selection", {})
    bindings = blind.get("bindings", {}) if isinstance(blind, Mapping) else {}
    target_sha_claim = _require_sha(bindings.get("target_map_sha256"), "blind target map SHA")
    primary = _load_json(authority.pair_selection_path)["PRIMARY_PAIR"]
    calibration_sha = _require_sha(
        primary.get("query_calibration_sha256"), "PRIMARY query calibration SHA"
    )
    _same(
        calibration_sha,
        extrinsic_sha,
        "PRIMARY query calibration/authoritative extrinsic SHA",
    )
    expected_blind_bindings = {
        "primary_query_sequence_id": authority.expected_query_sequence_id,
        "gt_sha256": authority.expected_reference_pose_sha256,
        "calibration_sha256": calibration_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "target_map_sha256": target_sha_claim,
    }
    _same(bindings, expected_blind_bindings, "blind immutable data bindings")

    reference_poses = _load_reference_poses(authority)
    map_reference_positions = _load_map_reference_positions(authority)
    selected_snapshots = _parse_selected_snapshots(frozen)
    scans, scan_raw = _parse_candidate_scans(frozen)
    _verify_candidate_scans(
        frozen,
        scans,
        scan_raw,
        query_allowlist,
        windows,
        authority=authority,
        preprocessing_sha=preprocessing_sha,
        selection_contract_sha=selection_contract_sha,
        target_map_sha=target_sha_claim,
        gt_sha=authority.expected_reference_pose_sha256,
        calibration_sha=calibration_sha,
        reference_poses=reference_poses,
        map_reference_positions=map_reference_positions,
    )
    intervals, interval_raw = _parse_candidate_intervals(frozen)
    _verify_candidate_intervals(
        frozen,
        intervals,
        interval_raw,
        scans,
        windows,
        selection_contract_sha=selection_contract_sha,
        reference_poses=reference_poses,
    )
    selected_interval_projection = _parse_selected_intervals(frozen)
    selected_interval_full = _verify_selected_intervals(
        selected_interval_projection, intervals
    )
    _verify_selected_snapshots(
        selected_snapshots,
        scans,
        selected_interval_full,
        selection_contract_sha,
    )

    query_state = _verify_query_evidence(
        frozen,
        runtime,
        query_allowlist=query_allowlist,
        scans=scans,
        intervals=intervals,
        selected_intervals=selected_interval_full,
        selected=selected_snapshots,
        selection_manifest=selection_manifest,
        selection_contract_sha256=selection_contract_sha,
        preprocessing_sha256=preprocessing_sha,
        gt_sha256=authority.expected_reference_pose_sha256,
        extrinsic_sha256=extrinsic_sha,
        target_map_sha256=target_sha_claim,
        reference_poses=reference_poses,
    )
    receipts = _verify_receipts(
        frozen,
        runtime,
        map_allowlist,
        query_allowlist,
        scans,
        selected_snapshots,
        authority,
        preprocessing_sha=preprocessing_sha,
        extrinsic_sha=extrinsic_sha,
        query_state=query_state,
    )
    for row in scans:
        receipt = receipts.get(("QUERY_GEOMETRY_FIRST_PASS", row["object_key"]))
        if receipt is None:
            _fail("candidate scan lacks first-pass receipt")
        for field, expected in (
            ("etag", row["etag"]),
            ("local_temporary_sha256", row["payload_sha256"]),
            ("processing_result_sha256", row["candidate_scan_row_sha256"]),
        ):
            _same(receipt[field], expected, f"first-pass receipt {field}")

    target_sha, target_path, target_size, target_count, target_replay = _verify_map(
        frozen,
        runtime,
        map_allowlist,
        query_allowlist,
        preprocessing_sha,
        receipts,
        authority,
        extrinsic_sha=extrinsic_sha,
    )
    _same(target_sha, target_sha_claim, "selection/target-map shared SHA")
    if any(int(row["target_map_point_count"]) != target_count for row in scans):
        _fail("candidate target point count differs from frozen target map")
    canonical_rows = _verify_canonical_inputs(
        frozen,
        runtime,
        selected_snapshots,
        authority,
        scans,
        reference_poses,
        target_sha=target_sha,
        target_path=target_path,
        target_size=target_size,
        target_count=target_count,
        preprocessing_sha=preprocessing_sha,
        selection_sha=selection_contract_sha,
    )
    _verify_canonical_source_witness(
        frozen,
        selected_snapshots,
        canonical_rows,
        receipts,
        preprocessing_sha=preprocessing_sha,
        gt_sha=authority.expected_reference_pose_sha256,
        extrinsic_sha=extrinsic_sha,
        witness_sha=witness_sha,
    )
    _verify_uncertainty(frozen)
    _verify_selection_manifest(
        frozen,
        selection_manifest,
        scans,
        intervals,
        selected_interval_full,
        selected_snapshots,
        expected_contract,
    )
    _verify_no_registration(frozen, runtime)
    readiness = _verify_readiness(frozen)
    authoritative_preprocessing_markdown = (
        authority.preprocessing_contract_path.with_suffix(".md")
    )
    if (
        not authoritative_preprocessing_markdown.is_file()
        or authoritative_preprocessing_markdown.is_symlink()
        or authoritative_preprocessing_markdown.resolve(strict=True)
        != authoritative_preprocessing_markdown
        or (frozen / "boreas_v2_stage2_preprocessing_contract.md").read_bytes()
        != authoritative_preprocessing_markdown.read_bytes()
    ):
        _fail("preprocessing Markdown authority bytes differ")
    _verify_summary_documents(
        frozen,
        runtime,
        authority,
        readiness=readiness,
        scans=scans,
        intervals=intervals,
        selected_intervals=selected_interval_full,
        selected_snapshots=selected_snapshots,
        canonical_rows=canonical_rows,
        selection_manifest=selection_manifest,
        target_sha256=target_sha,
        target_path=target_path,
        target_size_bytes=target_size,
        target_count=target_count,
        target_replay=target_replay,
    )
    _verify_download_authorization(
        frozen,
        runtime,
        authority,
        map_allowlist,
        query_allowlist,
        preprocessing_sha256=preprocessing_sha,
    )

    return {
        "BOREAS_EXTERNAL_V2_STAGE2_VERIFICATION_PASS": True,
        "BOREAS_EXTERNAL_V2_STAGE2_READY": readiness[
            "BOREAS_EXTERNAL_V2_STAGE2_READY"
        ],
        "READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION": readiness[
            "READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION"
        ],
        "candidate_scan_count": len(scans),
        "candidate_interval_count": len(intervals),
        "weak_interval_count": 10,
        "rich_interval_count": 10,
        "weak_snapshot_count": 50,
        "rich_snapshot_count": 50,
        "snapshot_count": 100,
        "target_map_sha256": target_sha,
        "target_replay_verification_scope": target_replay["verification_scope"],
        "target_replay_verification_status": target_replay["verification_status"],
        "target_replay_compared_npy_bytes": target_replay["compared_npy_bytes"],
        "target_replay_range_count": target_replay["replay_range_count"],
        "target_replay_reducer_source_sha256": target_replay["source_sha256"],
        "target_replay_reducer_binary_sha256": target_replay["binary_sha256"],
        "target_replay_compiler_sha256": target_replay["compiler_sha256"],
        "target_replay_estimated_peak_memory_bytes": target_replay[
            "estimated_peak_memory_bytes"
        ],
        "target_replay_max_voxels": target_replay["max_voxels"],
        "target_replay_live_available_memory_bytes": target_replay[
            "live_available_memory_bytes"
        ],
        "target_replay_memory_safety_margin_bytes": target_replay[
            "memory_safety_margin_bytes"
        ],
        "target_replay_resource_plan_payload_sha256": target_replay[
            "resource_evidence"
        ]["resource_plan_payload_sha256"],
        "target_replay_capacity_layout_sha256": target_replay[
            "resource_evidence"
        ]["capacity_layout_sha256"],
        "target_replay_capacity_replay_binding_payload_sha256": target_replay[
            "resource_evidence"
        ]["capacity_replay_binding_payload_sha256"],
        "manifest_root_sha256": frozen_manifest["manifest_root_sha256"],
        "registration_execution_count": 0,
    }


__all__ = [
    "BoreasStage2PreparationVerificationError",
    "CANONICAL_INPUT_FIELDS",
    "CANONICAL_SOURCE_VERIFICATION_FIELDS",
    "CANDIDATE_INTERVAL_FIELDS",
    "CANDIDATE_SCAN_FIELDS",
    "FROZEN_MANIFEST_NAME",
    "GEOMETRY_METRIC_FIELDS",
    "PAYLOAD_FILES",
    "PreparationVerificationAuthority",
    "RECEIPT_FIELDS",
    "REQUIRED_FILES",
    "SELECTED_INTERVAL_FIELDS",
    "SELECTED_SNAPSHOT_FIELDS",
    "UNCERTAINTY_FIELDS",
    "verify_boreas_v2_stage2_preparation",
]
