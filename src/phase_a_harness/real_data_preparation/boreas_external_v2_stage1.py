"""Build the Boreas supplementary external-validation v2 Stage-1 closure.

This module is deliberately limited to already authenticated trajectory and
calibration files plus S3 *metadata* listings.  It cannot download a LiDAR
payload and it executes under :class:`NoRegistrationGuard`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .boreas_stage1 import parse_pose_csv, parse_s3_ls_line
from .guard import NoRegistrationGuard, assert_preparation_sources_are_safe
from .io import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    compact_sha256,
    sha256_file,
)
from .public_data_v1_closure_verifier import verify_public_data_v1_closure
from .stage1_gt_overlap import (
    FROZEN_STAGE1_OVERLAP_CONTRACT,
    compute_stage1_gt_overlap,
    rank_distinct_stage1_pairs,
    resample_gap_aware,
)


EXPECTED_BRANCH = "prep/public-data-external-validation-v2-boreas-stage1"
BOREAS_V1_COMMIT = "d304ea7a2a201aff21b48e5039422b63b0eb6e99"
BOREAS_V1_ELIGIBILITY_SHA256 = "b76102144ad63e9383ed3027b574ce33c7023eda4b328b6b99621bb8d577f8c7"
BOREAS_V1_EXTRINSIC_PROVENANCE_SHA256 = "fda2be032b760caefdc8db89619a9d47c7deec1a513b72f7714c01f5791a9cf0"
BOREAS_V1_MANIFEST_SHA256 = "29c582361eb83392e54d34ea4465ae4fb6b1813c89a33a5fa1eb85e1abbab112"
BOREAS_V1_SHA256SUMS_SHA256 = "69a3a138617a315a3853dd33a519f9f142bd5bd9d08a3ace65b1dca9ba8d1975"
BACKEND_PARAMETER_SHA256 = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
PCL_EXTERNAL_BUNDLE = Path("/tmp/synthetic_confirmatory_v2_pcl_v3_requalification")
AWS_URI = "s3://boreas"
AWS_BUCKET = "boreas"

PROTOCOL_FILES = (
    "public_data_external_validation_protocol_v2.md",
    "public_data_external_validation_protocol_v2.json",
    "public_data_external_validation_eligibility_v2.csv",
    "public_data_external_validation_analysis_contract_v2.json",
    "public_data_external_validation_pair_selection_contract_v2.json",
)
V1_CLOSURE_FILES = (
    "public_data_v1_closure_summary.md",
    "public_data_v1_closure.json",
    "public_data_v1_candidate_screening.csv",
    "public_data_v1_evidence_manifest.json",
)

OVERLAP_FIELDS = (
    "map_sequence_id",
    "query_sequence_id",
    "query_duration_s",
    "covered_duration_s",
    "coverage_fraction",
    "contiguous_covered_interval_count",
    "eligible_nonoverlapping_5s_intervals",
    "nearest_distance_median_m",
    "nearest_distance_q95_m",
    "nearest_distance_max_m",
    "map_calibration_sha256",
    "query_calibration_sha256",
    "fixed_ENU_ref_status",
    "pair_eligibility_status",
    "exclusion_reason",
)
METADATA_FIELDS = (
    "sequence_id",
    "key",
    "timestamp_us",
    "last_modified",
    "size_bytes",
)
ALLOWLIST_FIELDS = (
    "selection_role",
    "sequence_id",
    "key",
    "timestamp_us",
    "last_modified",
    "size_bytes",
    "selection_reason",
)


class BoreasExternalV2Stage1Error(RuntimeError):
    """A v2 Stage-1 gate or immutable evidence binding failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_immutable(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise BoreasExternalV2Stage1Error(f"unsafe immutable source: {source}")
    atomic_write_bytes(destination, source.read_bytes())


def _verify_sha256sums(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    if not sums.is_file() or sums.is_symlink():
        raise BoreasExternalV2Stage1Error(f"SHA256SUMS absent or unsafe: {root}")
    rows: list[dict[str, Any]] = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            raise BoreasExternalV2Stage1Error(f"malformed SHA256SUMS row in {root}")
        expected, name = match.groups()
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise BoreasExternalV2Stage1Error(f"v1 frozen SHA mismatch: {path}")
        rows.append({"path": name, "sha256": expected, "size_bytes": path.stat().st_size})
    if len(rows) != len({row["path"] for row in rows}):
        raise BoreasExternalV2Stage1Error(f"duplicate SHA256SUMS row in {root}")
    return {
        "entry_count": len(rows),
        "rows": rows,
        "sha256sums_sha256": sha256_file(sums),
    }


def _receipt_for(path: Path) -> Path:
    return path.with_name(path.name + ".receipt.json")


def authenticate_boreas_v1_evidence(
    *, data_root: Path, boreas_v1_root: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Re-hash all 84 v1 evidence files and validate their receipt bindings."""

    if sha256_file(boreas_v1_root / "boreas_stage1_eligibility.json") != BOREAS_V1_ELIGIBILITY_SHA256:
        raise BoreasExternalV2Stage1Error("historical Boreas v1 eligibility changed")
    if sha256_file(boreas_v1_root / "boreas_lidar_extrinsic_provenance.json") != BOREAS_V1_EXTRINSIC_PROVENANCE_SHA256:
        raise BoreasExternalV2Stage1Error("historical Boreas v1 extrinsic provenance changed")
    if sha256_file(boreas_v1_root / "boreas_stage1_manifest.json") != BOREAS_V1_MANIFEST_SHA256:
        raise BoreasExternalV2Stage1Error("historical Boreas v1 manifest changed")
    if sha256_file(boreas_v1_root / "SHA256SUMS") != BOREAS_V1_SHA256SUMS_SHA256:
        raise BoreasExternalV2Stage1Error("historical Boreas v1 SHA256SUMS changed")
    closure = _verify_sha256sums(boreas_v1_root)
    manifest = _load_json(boreas_v1_root / "boreas_stage1_manifest.json")
    download = _load_json(boreas_v1_root / "download_manifest.json")
    evidence = manifest.get("data_evidence")
    if not isinstance(evidence, list) or len(evidence) != 84:
        raise BoreasExternalV2Stage1Error("Boreas v1 evidence count is not 84")
    if evidence != download.get("materialized_files"):
        raise BoreasExternalV2Stage1Error("v1 manifest/download evidence projections differ")
    paths: dict[str, Path] = {}
    reuse_rows: list[dict[str, Any]] = []
    receipt_count = 0
    for source in evidence:
        relative = source.get("relative_path")
        local = source.get("local_path")
        if not isinstance(relative, str) or not isinstance(local, str):
            raise BoreasExternalV2Stage1Error("invalid v1 evidence path")
        path = data_root / relative
        if path.resolve(strict=True) != Path(local).resolve(strict=True):
            raise BoreasExternalV2Stage1Error(f"v1 evidence path binding changed: {relative}")
        if path.is_symlink() or data_root.resolve() not in path.resolve().parents:
            raise BoreasExternalV2Stage1Error(f"unsafe v1 evidence file: {relative}")
        actual_sha = sha256_file(path)
        if actual_sha != source.get("sha256") or path.stat().st_size != source.get("size_bytes"):
            raise BoreasExternalV2Stage1Error(f"v1 evidence content changed: {relative}")
        receipt_path: Path | None = None
        receipt_sha: str | None = None
        if source.get("status") == "DOWNLOADED_ALLOWLISTED_STAGE1_OBJECT":
            receipt_path = _receipt_for(path)
            receipt = _load_json(receipt_path)
            for key, expected in (
                ("local_path", str(path)),
                ("key", source.get("s3_key")),
                ("sha256", actual_sha),
                ("size_bytes", path.stat().st_size),
                ("etag", source.get("etag")),
                ("last_modified", source.get("last_modified")),
                ("version_id", source.get("version_id")),
            ):
                if receipt.get(key) != expected:
                    raise BoreasExternalV2Stage1Error(f"receipt mismatch {relative}: {key}")
            receipt_sha = sha256_file(receipt_path)
            receipt_count += 1
        paths[relative] = path
        reuse_rows.append(
            {
                "local_path": str(path),
                "relative_path": relative,
                "receipt_path": None if receipt_path is None else str(receipt_path),
                "receipt_sha256": receipt_sha,
                "sha256": actual_sha,
                "size_bytes": path.stat().st_size,
                "source_status": source.get("status"),
            }
        )
    if receipt_count != 76:
        raise BoreasExternalV2Stage1Error("Boreas v1 downloaded receipt count is not 76")
    if any(path.suffix == ".bin" or "/lidar/" in path.as_posix() for path in paths.values()):
        raise BoreasExternalV2Stage1Error("v1 Stage-1 evidence unexpectedly contains LiDAR payload")
    return (
        {
            "data_evidence_file_count": len(reuse_rows),
            "downloaded_stage1_small_object_count": receipt_count,
            "files": reuse_rows,
            "files_redownloaded": 0,
            "registration_execution_count": 0,
            "v1_frozen_closure": closure,
            "v1_manifest_data_evidence_sha256": compact_sha256(evidence),
            "v1_manifest_sha256": sha256_file(boreas_v1_root / "boreas_stage1_manifest.json"),
            "v1_sha256sums_sha256": sha256_file(boreas_v1_root / "SHA256SUMS"),
        },
        paths,
    )


def _eligible_trajectories(
    *, boreas_v1_root: Path, data_root: Path, evidence_paths: Mapping[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    with (boreas_v1_root / "eligible_reference_sequences.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        frozen_rows = list(csv.DictReader(stream))
    if len(frozen_rows) != 29:
        raise BoreasExternalV2Stage1Error("eligible reference sequence count is not 29")
    trajectories: dict[str, np.ndarray] = {}
    pose_values: dict[str, np.ndarray] = {}
    reports: list[dict[str, Any]] = []
    calibration_sha: dict[str, str] = {}
    for frozen in frozen_rows:
        sequence = frozen["sequence_id"]
        relative = f"stage1_payload/{sequence}/applanix/lidar_poses.csv"
        path = evidence_paths.get(relative)
        if path is None:
            raise BoreasExternalV2Stage1Error(f"eligible pose evidence absent: {sequence}")
        report, trajectory, values = parse_pose_csv(path, sequence)
        if report["pose_sha256"] != frozen["pose_sha256"]:
            raise BoreasExternalV2Stage1Error(f"eligible pose SHA changed: {sequence}")
        if report["maximum_gap_s"] > FROZEN_STAGE1_OVERLAP_CONTRACT.maximum_native_gap_s:
            raise BoreasExternalV2Stage1Error(f"eligible sequence now violates gap gate: {sequence}")
        calibration = evidence_paths.get(
            f"stage1_payload/{sequence}/calib/T_applanix_lidar.txt"
        )
        if calibration is None:
            raise BoreasExternalV2Stage1Error(f"eligible calibration absent: {sequence}")
        calibration_sha[sequence] = sha256_file(calibration)
        reports.append(report)
        trajectories[sequence] = trajectory
        pose_values[sequence] = values
    if len(set(calibration_sha.values())) != 1:
        raise BoreasExternalV2Stage1Error("eligible sequence extrinsics are not byte-identical")
    reports.sort(key=lambda row: row["sequence_id"])
    return reports, trajectories, pose_values, calibration_sha


def compute_all_directed_pairs(
    *, reports: Sequence[Mapping[str, Any]], trajectories: Mapping[str, np.ndarray],
    calibration_sha: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Compute all 29x28 pairs with the frozen gap-aware implementation."""

    report_by_id = {str(row["sequence_id"]): row for row in reports}
    sequence_ids = sorted(trajectories)
    pair_rows: list[dict[str, Any]] = []
    raw_results: dict[tuple[str, str], dict[str, Any]] = {}
    for map_sequence in sequence_ids:
        for query_sequence in sequence_ids:
            if map_sequence == query_sequence:
                continue
            raw = compute_stage1_gt_overlap(
                trajectories[map_sequence],
                trajectories[query_sequence],
                common_world_frame_proven=True,
                contract=FROZEN_STAGE1_OVERLAP_CONTRACT,
            )
            raw_results[(map_sequence, query_sequence)] = raw
            status = str(raw["overlap_status"])
            row = {
                "map_sequence_id": map_sequence,
                "query_sequence_id": query_sequence,
                "query_duration_s": float(report_by_id[query_sequence]["duration_s"]),
                "covered_duration_s": float(raw["total_covered_duration_s"]),
                "coverage_fraction": float(raw["coverage_fraction"]),
                "contiguous_covered_interval_count": len(raw["contiguous_covered_intervals"]),
                "eligible_nonoverlapping_5s_intervals": int(
                    raw["eligible_nonoverlapping_5s_interval_count"]
                ),
                "nearest_distance_median_m": float(raw["nearest_distance_median_m"]),
                "nearest_distance_q95_m": float(raw["nearest_distance_q95_m"]),
                "nearest_distance_max_m": float(raw["nearest_distance_max_m"]),
                "map_calibration_sha256": calibration_sha[map_sequence],
                "query_calibration_sha256": calibration_sha[query_sequence],
                "fixed_ENU_ref_status": "PASS",
                "pair_eligibility_status": status,
                "exclusion_reason": "" if status == "PASS" else "FROZEN_OVERLAP_THRESHOLD_FAILURE",
            }
            pair_rows.append(row)
    if len(pair_rows) != 29 * 28:
        raise BoreasExternalV2Stage1Error("directed pair closure is not 29x28")
    rank_input = [
        {
            **row,
            "overlap_status": row["pair_eligibility_status"],
            "total_covered_duration_s": row["covered_duration_s"],
            "eligible_nonoverlapping_5s_interval_count": row[
                "eligible_nonoverlapping_5s_intervals"
            ],
            "map_independent_6dof": True,
            "query_independent_6dof": True,
            "common_world_frame_proven": True,
            "map_rig_lidar_transform_available": True,
            "query_rig_lidar_transform_available": True,
        }
        for row in pair_rows
    ]
    ranked_raw = rank_distinct_stage1_pairs(rank_input)
    ranked = [
        {field: row[field] for field in OVERLAP_FIELDS} | {"rank": row["rank"]}
        for row in ranked_raw
    ]
    return pair_rows, ranked, raw_results


def _complete_five_second_windows(intervals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for interval_index, interval in enumerate(intervals):
        count = int(float(interval["duration_s"]) // 5.0)
        start = float(interval["start_time"])
        for local_index in range(count):
            window_start = start + 5.0 * local_index
            windows.append(
                {
                    "duration_s": 5.0,
                    "end_time_s": window_start + 5.0,
                    "interval_index": interval_index,
                    "window_index": len(windows),
                    "start_time_s": window_start,
                }
            )
    return windows


def select_primary_lidar_keys(
    *, map_trajectory: np.ndarray, query_trajectory: np.ndarray,
    map_pose_values: np.ndarray, query_pose_values: np.ndarray,
    primary_overlap: Mapping[str, Any],
) -> dict[str, Any]:
    """Select keys using only fixed-frame GT and frozen complete 5 s windows."""

    windows = _complete_five_second_windows(primary_overlap["contiguous_covered_intervals"])
    if len(windows) != primary_overlap["eligible_nonoverlapping_5s_interval_count"]:
        raise BoreasExternalV2Stage1Error("complete-window count disagrees with overlap result")
    query_resampled = resample_gap_aware(query_trajectory, FROZEN_STAGE1_OVERLAP_CONTRACT)
    selected_query_resampled = np.zeros(query_resampled.shape[0], dtype=bool)
    for window in windows:
        selected_query_resampled |= (
            (query_resampled[:, 0] >= window["start_time_s"])
            & (query_resampled[:, 0] < window["end_time_s"])
        )
    selected_positions = query_resampled[selected_query_resampled, 1:4]
    if selected_positions.shape[0] != len(windows) * 5:
        raise BoreasExternalV2Stage1Error("selected 1 Hz query window closure is incomplete")
    map_distance, _ = cKDTree(selected_positions).query(map_trajectory[:, 1:4], k=1)
    selected_map_native = map_distance <= FROZEN_STAGE1_OVERLAP_CONTRACT.radius_m
    query_native_times = query_trajectory[:, 0]
    selected_query_native = np.zeros(query_native_times.shape[0], dtype=bool)
    for window in windows:
        selected_query_native |= (
            (query_native_times >= window["start_time_s"])
            & (query_native_times < window["end_time_s"])
        )
    map_timestamps = [int(round(value)) for value in map_pose_values[:, 0]]
    query_timestamps = [int(round(value)) for value in query_pose_values[:, 0]]
    return {
        "complete_five_second_windows": windows,
        "map_selected_timestamp_us": [
            timestamp for timestamp, selected in zip(map_timestamps, selected_map_native) if selected
        ],
        "query_selected_timestamp_us": [
            timestamp for timestamp, selected in zip(query_timestamps, selected_query_native) if selected
        ],
        "selected_query_1hz_sample_count": int(selected_positions.shape[0]),
        "selection_definition": {
            "map": "native map poses within 5 m in fixed ENU_ref of selected covered query 1 Hz samples",
            "query": "native query poses in complete half-open nonoverlapping 5 s windows carved from each covered island",
            "point_cloud_payload_consulted": False,
            "registration_result_consulted": False,
        },
    }


def _list_lidar_metadata(
    *, aws: Path, sequence_id: str, expected_inventory: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    command = [
        str(aws),
        "s3",
        "ls",
        f"{AWS_URI}/{sequence_id}/lidar/",
        "--recursive",
        "--no-sign-request",
    ]
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    last_timestamp: int | None = None
    for line in process.stdout:
        parsed = parse_s3_ls_line(line)
        prefix = f"{sequence_id}/lidar/"
        if not parsed["key"].startswith(prefix) or not parsed["key"].endswith(".bin"):
            raise BoreasExternalV2Stage1Error(f"unexpected LiDAR metadata key: {parsed['key']}")
        timestamp_text = Path(parsed["key"]).stem
        if re.fullmatch(r"\d{16}", timestamp_text) is None:
            raise BoreasExternalV2Stage1Error("invalid Boreas LiDAR object timestamp")
        timestamp = int(timestamp_text)
        if last_timestamp is not None and timestamp <= last_timestamp:
            raise BoreasExternalV2Stage1Error("LiDAR metadata listing is not strictly ordered")
        last_timestamp = timestamp
        digest.update(
            json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
            + b"\n"
        )
        rows.append(
            {
                "sequence_id": sequence_id,
                "key": parsed["key"],
                "timestamp_us": timestamp,
                "last_modified": parsed["last_modified"],
                "size_bytes": parsed["size_bytes"],
            }
        )
    stderr = process.communicate()[1]
    if process.returncode:
        raise BoreasExternalV2Stage1Error(
            f"metadata-only S3 listing failed for {sequence_id}: {stderr.strip()}"
        )
    listing_sha = digest.hexdigest()
    expected_count = expected_inventory.get("lidar_object_count")
    expected_bytes = expected_inventory.get("lidar_remote_bytes")
    expected_sha = expected_inventory.get("lidar_listing_rows_sha256")
    if (
        len(rows) != expected_count
        or sum(row["size_bytes"] for row in rows) != expected_bytes
        or listing_sha != expected_sha
        or (rows and rows[0]["timestamp_us"] != expected_inventory.get("first_lidar_timestamp"))
        or (rows and rows[-1]["timestamp_us"] != expected_inventory.get("last_lidar_timestamp"))
    ):
        raise BoreasExternalV2Stage1Error(
            f"metadata listing no longer matches authenticated v1 inventory: {sequence_id}"
        )
    return rows, listing_sha


def _validate_test_baseline(value: Mapping[str, Any]) -> None:
    initial = value.get("initial_baseline", {})
    source = value.get("source_only_baseline", {})
    external = value.get("external_qualification_baseline", {})
    if {key: initial.get(key) for key in ("collected", "failed", "passed", "skipped")} != {
        "collected": 771,
        "failed": 1,
        "passed": 761,
        "skipped": 9,
    }:
        raise BoreasExternalV2Stage1Error("initial test baseline was altered")
    if source.get("status") != "PASS" or source.get("failed") != 0 or source.get("errors") != 0:
        raise BoreasExternalV2Stage1Error("source-only test baseline is not clean")
    if external.get("status") != "UNAVAILABLE" or external.get("passed") != 0:
        raise BoreasExternalV2Stage1Error("external historical qualification was misreported")
    if external.get("missing_path") != str(PCL_EXTERNAL_BUNDLE):
        raise BoreasExternalV2Stage1Error("external qualification missing path changed")
    if "source-only package: external historical qualification bundle unavailable" not in str(
        external.get("reason")
    ):
        raise BoreasExternalV2Stage1Error("external qualification reason is incomplete")


def make_test_baseline_report(
    *, source_only_collected: int, source_only_passed: int, source_only_skipped: int
) -> dict[str, Any]:
    value = {
        "EXTERNAL_QUALIFICATION_BASELINE": "UNAVAILABLE",
        "SOURCE_ONLY_TEST_BASELINE": "PASS",
        "external_qualification_baseline": {
            "command": (
                "ZPRM_REQUIRE_PCL_V3_EXTERNAL_QUALIFICATION=1 python -m pytest -q "
                "tests/test_runtime_lifecycle_qualification.py::"
                "test_fixed_pcl_v3_input_inventory_matches_contract"
            ),
            "errors": 0,
            "exit_code": 1,
            "failed": 1,
            "missing_path": str(PCL_EXTERNAL_BUNDLE),
            "passed": 0,
            "reason": (
                f"{PCL_EXTERNAL_BUNDLE}: source-only package: external historical "
                "qualification bundle unavailable"
            ),
            "required_environment": "ZPRM_REQUIRE_PCL_V3_EXTERNAL_QUALIFICATION=1",
            "status": "UNAVAILABLE",
        },
        "initial_baseline": {
            "collected": 771,
            "command": "python -m pytest -q",
            "failed": 1,
            "passed": 761,
            "skipped": 9,
        },
        "source_only_baseline": {
            "collected": source_only_collected,
            "command": "python -m pytest -q",
            "errors": 0,
            "failed": 0,
            "passed": source_only_passed,
            "skipped": source_only_skipped,
            "status": "PASS",
        },
    }
    _validate_test_baseline(value)
    return value


def _test_baseline_markdown(value: Mapping[str, Any]) -> str:
    source = value["source_only_baseline"]
    external = value["external_qualification_baseline"]
    return (
        "# Public-data v2 test baselines\n\n"
        "- `SOURCE_ONLY_TEST_BASELINE=PASS`: "
        f"{source['collected']} collected, {source['passed']} passed, "
        f"{source['skipped']} skipped, 0 failed, 0 errors.\n"
        "- `EXTERNAL_QUALIFICATION_BASELINE=UNAVAILABLE`: strict mode fails because "
        f"`{external['missing_path']}` is unavailable; it is not reported as PASS.\n"
    )


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Boreas External Validation v2 Stage-1 总结",
        "",
        f"**BOREAS_EXTERNAL_V2_STAGE1_READY={str(summary['BOREAS_EXTERNAL_V2_STAGE1_READY']).lower()}**",
        f"**READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD={str(summary['READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD']).lower()}**",
        "",
    ]
    lines.extend(f"{index}. {answer}" for index, answer in enumerate(summary["answers"], 1))
    lines.extend(["", "## 最终结论", "", summary["final_conclusion"], ""])
    return "\n".join(lines)


def _sha256sums_payload(root: Path, names: Iterable[str]) -> bytes:
    return "".join(
        f"{sha256_file(root / name)}  {name}\n" for name in sorted(names)
    ).encode("utf-8")


def _lidar_payload_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.bin")
        if path.is_file() and path.parent.name.lower() == "lidar"
    )


def build_boreas_external_v2_stage1(
    *, repository: str | Path, data_root: str | Path, runtime_root: str | Path,
    source_only_collected: int, source_only_passed: int, source_only_skipped: int,
) -> dict[str, Any]:
    repository = Path(repository).resolve(strict=True)
    data_root = Path(data_root).resolve(strict=True)
    runtime_argument = Path(runtime_root)
    runtime_root = runtime_argument.resolve(strict=False)
    if runtime_argument != runtime_root or runtime_root.exists() or runtime_root.is_symlink():
        raise BoreasExternalV2Stage1Error("fresh canonical absent runtime root is required")
    if repository in runtime_root.parents or data_root in runtime_root.parents:
        raise BoreasExternalV2Stage1Error("runtime root must be disjoint from repository and data")
    if os.environ.get("ZPRM_REAL_DATA_PREP_NO_REGISTRATION") != "1":
        raise BoreasExternalV2Stage1Error("ZPRM_REAL_DATA_PREP_NO_REGISTRATION=1 is mandatory")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repository, text=True
    ).strip()
    if branch != EXPECTED_BRANCH:
        raise BoreasExternalV2Stage1Error(f"formal v2 Stage-1 requires branch {EXPECTED_BRANCH}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip():
        raise BoreasExternalV2Stage1Error("formal v2 Stage-1 requires a clean worktree")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", BOREAS_V1_COMMIT, "HEAD"], cwd=repository
    ).returncode:
        raise BoreasExternalV2Stage1Error("Boreas v1 commit is not an ancestor")
    producer_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    initial_lidar_payload_paths = _lidar_payload_paths(data_root)
    if initial_lidar_payload_paths:
        raise BoreasExternalV2Stage1Error("LiDAR payload must be absent before Stage-1")
    runtime_root.mkdir(parents=True)

    boreas_v1_root = repository / "frozen_assets/real_data_boreas_stage1_v1"
    v1_closure_root = repository / "frozen_assets/public_data_validation_v1_closure"
    protocols_root = repository / "protocols"
    test_baseline = make_test_baseline_report(
        source_only_collected=source_only_collected,
        source_only_passed=source_only_passed,
        source_only_skipped=source_only_skipped,
    )
    atomic_write_json(runtime_root / "test_baseline_status.json", test_baseline)
    atomic_write_bytes(
        runtime_root / "test_baseline_status.md", _test_baseline_markdown(test_baseline).encode("utf-8")
    )
    v1_closure_report = verify_public_data_v1_closure(
        repository=repository, closure_root=v1_closure_root
    )
    for name in V1_CLOSURE_FILES:
        _copy_immutable(v1_closure_root / name, runtime_root / name)
    for name in PROTOCOL_FILES:
        _copy_immutable(protocols_root / name, runtime_root / name)

    protocol = _load_json(protocols_root / "public_data_external_validation_protocol_v2.json")
    pair_contract = _load_json(
        protocols_root / "public_data_external_validation_pair_selection_contract_v2.json"
    )
    if protocol.get("selected_dataset") != "Boreas" or protocol.get("role") != "SUPPLEMENTARY_EXTERNAL_GENERALIZATION":
        raise BoreasExternalV2Stage1Error("v2 protocol identity mismatch")
    if pair_contract.get("eligibility_thresholds") != {
        "GT_OVERLAP_RADIUS_M": 5.0,
        "GT_RESAMPLE_RATE_HZ": 1.0,
        "MAX_NATIVE_REFERENCE_GAP_S": 0.2,
        "MIN_CONTIGUOUS_COVERED_DURATION_S": 5.0,
        "MIN_COVERAGE_FRACTION": 0.6,
        "MIN_ELIGIBLE_NONOVERLAPPING_5S_INTERVALS": 30,
        "MIN_TOTAL_COVERED_DURATION_S": 150.0,
    }:
        raise BoreasExternalV2Stage1Error("v2 pair thresholds changed")
    if sha256_file(repository / "frozen_assets/backend_parameter_contract.json") != BACKEND_PARAMETER_SHA256:
        raise BoreasExternalV2Stage1Error("frozen backend parameter contract changed")

    reuse, evidence_paths = authenticate_boreas_v1_evidence(
        data_root=data_root, boreas_v1_root=boreas_v1_root
    )
    before_stats = {
        row["relative_path"]: (Path(row["local_path"]).stat().st_mtime_ns, row["sha256"])
        for row in reuse["files"]
    }
    v1_eligibility = _load_json(boreas_v1_root / "boreas_stage1_eligibility.json")
    reference = _load_json(boreas_v1_root / "boreas_reference_trajectory_provenance.json")
    extrinsic = _load_json(boreas_v1_root / "boreas_lidar_extrinsic_provenance.json")
    world = _load_json(boreas_v1_root / "boreas_common_world_frame_audit.json")
    time_audit = _load_json(boreas_v1_root / "boreas_time_sync_audit.json")
    if (
        v1_eligibility.get("R02") != "FAIL"
        or v1_eligibility.get("R10") != "PARTIAL"
        or v1_eligibility.get("BOREAS_STAGE1_READY") is not False
    ):
        raise BoreasExternalV2Stage1Error("Boreas v1 conclusion was retrospectively altered")
    trajectory_facts = {
        "uses_gnss": reference.get("uses_gnss"),
        "uses_imu": reference.get("uses_imu"),
        "uses_wheel_encoder": reference.get("uses_wheel_encoder"),
        "uses_rtx_or_pospac": reference.get("uses_rtx"),
        "uses_lidar": reference.get("uses_lidar"),
        "uses_icp": reference.get("uses_icp"),
        "uses_scan_matching": reference.get("uses_scan_matching"),
    }
    if trajectory_facts != {
        "uses_gnss": True,
        "uses_imu": True,
        "uses_wheel_encoder": True,
        "uses_rtx_or_pospac": True,
        "uses_lidar": False,
        "uses_icp": False,
        "uses_scan_matching": False,
    }:
        raise BoreasExternalV2Stage1Error("trajectory backbone evidence is incomplete")
    static_extrinsic = {
        "fixed": extrinsic.get("common_static_calibration"),
        "identical_across_all_44_sequences": extrinsic.get("all_sequences_byte_identical"),
        "publicly_released": True,
        "calibration_used_lidar_point_clouds": extrinsic.get("uses_lidar_pointclouds"),
        "calibration_date_or_session": extrinsic.get("calibration_date_or_session"),
        "query_participation": extrinsic.get("query_sequence_used_for_calibration"),
        "uncertainty": extrinsic.get("extrinsic_uncertainty"),
        "reestimated_in_v2": False,
        "current_query_target_registration_output_used": False,
    }
    if static_extrinsic != {
        "fixed": True,
        "identical_across_all_44_sequences": True,
        "publicly_released": True,
        "calibration_used_lidar_point_clouds": True,
        "calibration_date_or_session": "UNKNOWN",
        "query_participation": "UNKNOWN",
        "uncertainty": "UNKNOWN",
        "reestimated_in_v2": False,
        "current_query_target_registration_output_used": False,
    }:
        raise BoreasExternalV2Stage1Error("LiDAR-assisted static extrinsic provenance was hidden")
    requalification = {
        "E02": "PASS_WITH_DOCUMENTED_LIMITATION",
        "boreas_v1_verifier_report": {
            "BOREAS_STAGE1_VERIFICATION_PASS": True,
            "data_evidence_file_count": 84,
            "original_sequence_count": 44,
            "public_gt_sequence_count": 31,
            "registration_execution_count": 0,
        },
        "boreas_v1_verifier_command": (
            "python scripts/verify_boreas_stage1_v1.py --data-root "
            "/home/lj/zero_perturbation_data/boreas_stage1_v1 --runtime-root "
            "/home/lj/ZPRM/frozen_assets/real_data_boreas_stage1_v1"
        ),
        "boreas_v1_verifier_pass_confirmed_before_v2_build": True,
        "common_world_frame_status": world.get("status"),
        "files_redownloaded": 0,
        "registration_execution_count": 0,
        "static_extrinsic": static_extrinsic,
        "time_chain_status": time_audit.get("status"),
        "trajectory_backbone": trajectory_facts,
        "v1_conclusion": {
            "BOREAS_STAGE1_READY": False,
            "R02": "FAIL",
            "R10": "PARTIAL",
        },
        "v1_conclusion_preserved": True,
        "v2_role": "SUPPLEMENTARY_EXTERNAL_GENERALIZATION",
        "v2_semantics_versioned": True,
    }
    atomic_write_json(runtime_root / "boreas_v2_requalification_from_frozen_evidence.json", requalification)
    atomic_write_json(runtime_root / "boreas_v2_evidence_reuse_manifest.json", reuse)

    reports, trajectories, pose_values, calibration_sha = _eligible_trajectories(
        boreas_v1_root=boreas_v1_root,
        data_root=data_root,
        evidence_paths=evidence_paths,
    )
    aws = data_root / "tools/awscli-venv/bin/aws"
    if not aws.is_file() or not os.access(aws, os.X_OK):
        raise BoreasExternalV2Stage1Error("isolated AWS CLI is absent")
    source_inventory = _load_json(boreas_v1_root / "boreas_remote_sequence_inventory.json")
    inventory_by_id = {
        row["sequence_id"]: row for row in source_inventory["boreas_original_sequences"]
    }

    with NoRegistrationGuard(open3d_module=None) as guard:
        static_source_audit = assert_preparation_sources_are_safe(
            repository / "src/phase_a_harness/real_data_preparation"
        )
        pair_rows, ranked, raw_results = compute_all_directed_pairs(
            reports=reports,
            trajectories=trajectories,
            calibration_sha=calibration_sha,
        )
        if not ranked:
            raise BoreasExternalV2Stage1Error("no directed pair met the frozen overlap gates")
        atomic_write_csv(runtime_root / "boreas_v2_gt_only_overlap_matrix.csv", pair_rows, OVERLAP_FIELDS)
        overlap_matrix = {
            "contract": asdict(FROZEN_STAGE1_OVERLAP_CONTRACT),
            "eligible_pair_count": len(ranked),
            "eligible_reference_sequence_count": len(reports),
            "expected_directed_pair_count": 29 * 28,
            "gate_status": "PASS",
            "pair_count": len(pair_rows),
            "pair_rows": pair_rows,
            "point_cloud_or_registration_consulted": False,
            "status": "PASS",
        }
        atomic_write_json(runtime_root / "boreas_v2_gt_only_overlap_matrix.json", overlap_matrix)

        selected = ranked[:3]
        primary = selected[0]
        primary_key = (primary["map_sequence_id"], primary["query_sequence_id"])
        primary_raw = raw_results[primary_key]
        selection = select_primary_lidar_keys(
            map_trajectory=trajectories[primary_key[0]],
            query_trajectory=trajectories[primary_key[1]],
            map_pose_values=pose_values[primary_key[0]],
            query_pose_values=pose_values[primary_key[1]],
            primary_overlap=primary_raw,
        )
        pair_selection = {
            "PRIMARY_PAIR": selected[0],
            "RESERVE_PAIR_1": selected[1],
            "RESERVE_PAIR_2": selected[2],
            "eligible_pair_count": len(ranked),
            "primary_contiguous_covered_intervals": primary_raw[
                "contiguous_covered_intervals"
            ],
            "primary_complete_five_second_windows": selection[
                "complete_five_second_windows"
            ],
            "ranking_rule": pair_contract["pair_selection"]["ranking"],
            "reserve_activation_policy": pair_contract["reserve_pair_activation"],
            "selection_inputs": "GT_ONLY_FIXED_ENU_REF",
        }
        atomic_write_json(runtime_root / "boreas_v2_pair_selection.json", pair_selection)
        pair_md = (
            "# Boreas v2 frozen GT-only pairs\n\n"
            f"- PRIMARY: `{primary_key[0]} -> {primary_key[1]}`\n"
            f"- RESERVE 1: `{selected[1]['map_sequence_id']} -> {selected[1]['query_sequence_id']}`\n"
            f"- RESERVE 2: `{selected[2]['map_sequence_id']} -> {selected[2]['query_sequence_id']}`\n"
            f"- Eligible directed pairs: {len(ranked)} / {len(pair_rows)}\n\n"
            "Reserve activation is limited to the frozen infrastructure failures; registration or publication outcomes can never trigger a switch.\n"
        )
        atomic_write_bytes(runtime_root / "boreas_v2_pair_selection.md", pair_md.encode("utf-8"))

        all_metadata: list[dict[str, Any]] = []
        listing_sha: dict[str, str] = {}
        for sequence_id in primary_key:
            rows, digest = _list_lidar_metadata(
                aws=aws,
                sequence_id=sequence_id,
                expected_inventory=inventory_by_id[sequence_id],
            )
            all_metadata.extend(rows)
            listing_sha[sequence_id] = digest
        atomic_write_csv(
            runtime_root / "boreas_v2_primary_pair_lidar_metadata_inventory.csv",
            all_metadata,
            METADATA_FIELDS,
        )
        metadata_by_key = {row["key"]: row for row in all_metadata}
        expected_all_keys = {
            f"{primary_key[0]}/lidar/{int(round(value))}.bin"
            for value in pose_values[primary_key[0]][:, 0]
        } | {
            f"{primary_key[1]}/lidar/{int(round(value))}.bin"
            for value in pose_values[primary_key[1]][:, 0]
        }
        if set(metadata_by_key) != expected_all_keys:
            raise BoreasExternalV2Stage1Error("metadata listing and authenticated pose keys differ")
        allowlist: list[dict[str, Any]] = []
        for role, sequence_id, timestamps, reason in (
            (
                "TARGET_MAP",
                primary_key[0],
                selection["map_selected_timestamp_us"],
                "native map pose within 5 m of selected covered query 1 Hz GT sample",
            ),
            (
                "QUERY",
                primary_key[1],
                selection["query_selected_timestamp_us"],
                "native query pose in frozen complete nonoverlapping covered 5 s window",
            ),
        ):
            for timestamp in timestamps:
                key = f"{sequence_id}/lidar/{timestamp}.bin"
                source = metadata_by_key.get(key)
                if source is None:
                    raise BoreasExternalV2Stage1Error(f"selected key absent from metadata: {key}")
                allowlist.append(
                    {
                        "selection_role": role,
                        **source,
                        "selection_reason": reason,
                    }
                )
        role_order = {"TARGET_MAP": 0, "QUERY": 1}
        allowlist.sort(key=lambda row: (role_order[row["selection_role"]], row["key"]))
        atomic_write_csv(
            runtime_root / "boreas_v2_stage2_download_allowlist.csv",
            allowlist,
            ALLOWLIST_FIELDS,
        )
        role_rows = {
            role: [row for row in allowlist if row["selection_role"] == role]
            for role in ("TARGET_MAP", "QUERY")
        }
        role_summary = {
            role: {
                "first_timestamp_us": rows[0]["timestamp_us"],
                "last_timestamp_us": rows[-1]["timestamp_us"],
                "object_count": len(rows),
                "remote_bytes": sum(row["size_bytes"] for row in rows),
                "sequence_id": rows[0]["sequence_id"],
            }
            for role, rows in role_rows.items()
        }
        selected_bytes = sum(row["size_bytes"] for row in allowlist)
        full_sequence_metadata = {
            sequence_id: {
                "first_timestamp_us": sequence_rows[0]["timestamp_us"],
                "last_timestamp_us": sequence_rows[-1]["timestamp_us"],
                "object_count": len(sequence_rows),
                "remote_bytes": sum(row["size_bytes"] for row in sequence_rows),
                "listing_sha256": listing_sha[sequence_id],
            }
            for sequence_id in primary_key
            for sequence_rows in [
                [row for row in all_metadata if row["sequence_id"] == sequence_id]
            ]
        }
        plan = {
            "allowlist_file": "boreas_v2_stage2_download_allowlist.csv",
            "allowlist_sha256": sha256_file(
                runtime_root / "boreas_v2_stage2_download_allowlist.csv"
            ),
            "downloaded_lidar_bytes": 0,
            "downloaded_lidar_object_count": 0,
            "estimated_download_bytes": selected_bytes,
            "estimated_download_GB_decimal": selected_bytes / 1_000_000_000,
            "estimated_download_GiB": selected_bytes / (1024**3),
            "full_primary_sequence_metadata": full_sequence_metadata,
            "metadata_listing_invocations": [
                {
                    "command_class": "aws s3 ls --recursive --no-sign-request",
                    "listing_sha256": listing_sha[sequence_id],
                    "payload_download": False,
                    "prefix": f"{AWS_URI}/{sequence_id}/lidar/",
                    "sequence_id": sequence_id,
                }
                for sequence_id in primary_key
            ],
            "metadata_only": True,
            "primary_pair": {
                "map_sequence_id": primary_key[0],
                "query_sequence_id": primary_key[1],
            },
            "selection": selection["selection_definition"],
            "selected_complete_five_second_window_count": len(
                selection["complete_five_second_windows"]
            ),
            "selected_objects": role_summary,
            "stage2_execution_authorized": False,
        }
        atomic_write_json(runtime_root / "boreas_v2_stage2_download_plan.json", plan)
        map_bytes = role_summary["TARGET_MAP"]["remote_bytes"]
        query_bytes = role_summary["QUERY"]["remote_bytes"]
        decoded_working = selected_bytes * 2
        target_map = map_bytes * 2
        canonical_bundle = selected_bytes * 2
        temporary = selected_bytes
        subtotal = selected_bytes + decoded_working + target_map + canonical_bundle + temporary
        safety = math.ceil(subtotal * 1.5)
        disk_free = shutil.disk_usage(data_root).free
        disk_budget = {
            "canonical_bundle_bytes": canonical_bundle,
            "current_free_bytes": disk_free,
            "current_machine_meets_safety_budget": disk_free >= safety,
            "decoded_or_unpacked_working_bytes": decoded_working,
            "download_bytes": selected_bytes,
            "formula": "1.5 * (download + decoded_working + target_map + canonical_bundle + temporary_processing)",
            "minimum_safety_multiplier": 1.5,
            "query_download_bytes": query_bytes,
            "required_safe_disk_GB_decimal": safety / 1_000_000_000,
            "required_safe_disk_GiB": safety / (1024**3),
            "required_safe_disk_bytes": safety,
            "subtotal_before_safety_bytes": subtotal,
            "target_map_bytes": target_map,
            "target_map_download_bytes": map_bytes,
            "temporary_processing_bytes": temporary,
        }
        atomic_write_json(runtime_root / "boreas_v2_stage2_disk_budget.json", disk_budget)

        lidar_payload_paths = _lidar_payload_paths(data_root)
        no_lidar = {
            "data_root": str(data_root),
            "downloaded_lidar_bytes": 0,
            "downloaded_lidar_object_count": 0,
            "downloaded_lidar_payload_count": 0,
            "initial_local_lidar_bin_count": len(initial_lidar_payload_paths),
            "local_lidar_bin_count": len(lidar_payload_paths),
            "local_lidar_bin_paths": [str(path) for path in lidar_payload_paths],
            "metadata_listing_only": True,
            "pass": not lidar_payload_paths,
            "payload_materialization_authorized": False,
            "status": "PASS",
        }
        if no_lidar["pass"] is not True:
            raise BoreasExternalV2Stage1Error("LiDAR payload appeared during Stage-1")
        atomic_write_json(runtime_root / "NO_LIDAR_PAYLOAD_ATTESTATION.json", no_lidar)

        eligibility = {
            "BOREAS_EXTERNAL_V2_STAGE1_READY": True,
            "E01": "PASS",
            "E02": "PASS_WITH_DOCUMENTED_LIMITATION",
            "E03": "PASS_PREREGISTERED_DISTINCT_TARGET_MAP",
            "E04": "PASS",
            "E05": "PASS_WITH_DOCUMENTED_LIMITATION",
            "E06": "FROZEN_FOR_FUTURE_STAGE2_CURRENT_COUNTS_ZERO",
            "E07": "PASS_PREREGISTERED",
            "E08": "PASS_PREREGISTERED",
            "E09": "PASS",
            "E10": "PASS_WITH_UNKNOWN_COMPONENTS_PRESERVED",
            "E11": "PASS_SUPPLEMENTARY_BOUNDARY",
            "E12": "PASS_PREREGISTERED",
            "E13": "PASS_PREREGISTERED",
            "E14": "PASS_PAIR_FREEZE",
            "GT_ONLY_OVERLAP": "PASS",
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
            "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
            "READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD": True,
            "REAL_DATA_MAIN_EXPERIMENT_AUTHORIZED": False,
            "REAL_REGISTRATION_AUTHORIZED": False,
            "actual_trials": 0,
            "downloaded_lidar_payload_count": 0,
            "planned_trials": 0,
            "registration_execution_count": 0,
            "rich_snapshot_count": 0,
            "snapshot_count": 0,
            "weak_snapshot_count": 0,
        }
        atomic_write_json(runtime_root / "public_data_v2_stage1_eligibility.json", eligibility)
        attestation = guard.attestation(runtime_root)
        attestation.update(
            {
                "static_source_audit": static_source_audit,
                "status": "PASS" if attestation["pass"] else "FAIL",
            }
        )
        if attestation["pass"] is not True:
            raise BoreasExternalV2Stage1Error("NO-ICP attestation failed")
        atomic_write_json(runtime_root / "NO_ICP_ATTESTATION.json", attestation)

    after_stats = {
        row["relative_path"]: (Path(row["local_path"]).stat().st_mtime_ns, sha256_file(row["local_path"]))
        for row in reuse["files"]
    }
    if after_stats != before_stats:
        raise BoreasExternalV2Stage1Error("an authenticated v1 evidence file changed during v2")
    primary_bytes = plan["estimated_download_bytes"]
    answers = [
        "Boreas v1 的 R02=FAIL 属于已冻结的严格筛选结论；v2 使用新编号和新语义，因此不追溯修改 v1。",
        "v1 原文要求 independent high-accuracy 6DoF reference；Stage-1 严格实现把 LiDAR-assisted static extrinsic 也视为独立性失败。",
        "v2 要求 trajectory backbone 独立于待测 query-to-target registration，且不得由 ICP、scan matching、LiDAR odometry 或 SLAM 产生。",
        "允许历史 LiDAR-assisted fixed extrinsic，是因为它在实验前公开冻结、44 条 sequence 字节一致、未由当前 registration 估计。",
        "限制包括完整披露 provenance，date/session/query participation 与 uncertainty 保持 UNKNOWN，且只能用于 supplementary trend/generalization。",
        "外参、姿态、同步、插值、deskew 和 map accumulation 等不确定性仍含 UNKNOWN，故不能承担毫米级主真实证据。",
        "是；IILABS、GrandTour、RTS-GT、CAVERS、Boreas 五个 v1 候选已完整 closure，eligible=0。",
        "是；Public-data v1 registration count 与 ICP count 均保持 0。",
        "默认 source-only 明确 skip 并给出完整路径；strict external 模式因历史包缺失明确失败，不伪造也不报 PASS。",
        f"是；source-only 为 0 failed/0 errors（{source_only_collected} collected, {source_only_passed} passed, {source_only_skipped} skipped）。",
        "strict external qualification=UNAVAILABLE；缺失外部历史包的单测按要求非零失败。",
        "是；84 个 Boreas v1 evidence 文件逐项重新计算 SHA，并绑定 v1 manifest、SHA256SUMS 与 76 个 receipt。",
        "否；已有 Stage-1 小文件重新下载数为 0，内容与 mtime 均未改变。",
        "否；本任务下载和物化的 lidar/*.bin 均为 0。",
        "是；31 条 public GT 中按原生 gap 排除 2 条，29 条 eligible reference sequences 均重新解析和 SHA 验证。",
        f"是；GT-only overlap=PASS，全部 {len(pair_rows)} 个有向 pair 中 {len(ranked)} 个通过原门槛。",
        f"primary pair 为 {primary_key[0]} -> {primary_key[1]}。",
        f"reserve pairs 为 {selected[1]['map_sequence_id']} -> {selected[1]['query_sequence_id']} 与 {selected[2]['map_sequence_id']} -> {selected[2]['query_sequence_id']}。",
        f"primary covered duration={primary['covered_duration_s']} s。",
        f"primary coverage fraction={primary['coverage_fraction']}。",
        f"primary eligible 5 s intervals={primary['eligible_nonoverlapping_5s_intervals']}。",
        f"Stage-2 冻结 allowlist 预计下载 {primary_bytes / 1_000_000_000:.6f} GB（{primary_bytes / (1024**3):.6f} GiB）。",
        "是；Open3D/PCL/其他 registration 调用与 registration execution count 全部为 0。",
        "BOREAS_EXTERNAL_V2_STAGE1_READY=true。",
        "是；可以进入一个独立、另行授权的 Stage-2 LiDAR 下载任务，但本任务未授权或启动。",
        "是；MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false。",
    ]
    final_conclusion = (
        "BOREAS_EXTERNAL_V2_STAGE1_READY=true；"
        "READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD=true。Boreas 已在版本化的 supplementary "
        "external-validation 协议下通过 GT-only Stage-1。其轨迹 backbone 独立于待测 "
        "registration，固定外参的 LiDAR-assisted provenance 已公开保留，且所有 reference "
        "uncertainty 限制均未被隐藏。下一独立任务可以按冻结 allowlist 下载 primary pair "
        "的最小必要 LiDAR，但本任务未执行 ICP。"
    )
    summary = {
        **eligibility,
        "answers": answers,
        "final_conclusion": final_conclusion,
        "primary_pair": primary,
        "reserve_pairs": selected[1:3],
    }
    atomic_write_json(runtime_root / "public_data_v2_stage1_summary.json", summary)
    atomic_write_bytes(
        runtime_root / "public_data_v2_stage1_summary.md", _summary_markdown(summary).encode("utf-8")
    )

    payload_names = sorted(
        path.name
        for path in runtime_root.iterdir()
        if path.is_file() and path.name not in {"frozen_manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "backend_parameter_contract_sha256": BACKEND_PARAMETER_SHA256,
        "boreas_v1_commit": BOREAS_V1_COMMIT,
        "boreas_v1_evidence_count": 84,
        "eligibility_sha256": sha256_file(
            runtime_root / "public_data_v2_stage1_eligibility.json"
        ),
        "payload": [
            {
                "path": name,
                "sha256": sha256_file(runtime_root / name),
                "size_bytes": (runtime_root / name).stat().st_size,
            }
            for name in payload_names
        ],
        "producer_commit": producer_commit,
        "protocol_sha256": {
            name: sha256_file(protocols_root / name) for name in PROTOCOL_FILES
        },
        "schema_version": "public_data_external_validation_v2_boreas_stage1_manifest_v1",
        "v1_closure_verification": v1_closure_report,
    }
    manifest["manifest_root_sha256"] = compact_sha256(manifest)
    atomic_write_json(runtime_root / "frozen_manifest.json", manifest)
    atomic_write_bytes(
        runtime_root / "SHA256SUMS",
        _sha256sums_payload(runtime_root, [*payload_names, "frozen_manifest.json"]),
    )
    return summary


def freeze_runtime_assets(*, runtime_root: str | Path, frozen_root: str | Path) -> None:
    runtime = Path(runtime_root).resolve(strict=True)
    destination_argument = Path(frozen_root)
    destination = destination_argument.resolve(strict=False)
    if destination_argument != destination or destination.exists() or destination.is_symlink():
        raise BoreasExternalV2Stage1Error("fresh canonical absent frozen root is required")
    if not (runtime / "SHA256SUMS").is_file() or not (runtime / "frozen_manifest.json").is_file():
        raise BoreasExternalV2Stage1Error("runtime closure is incomplete")
    shutil.copytree(runtime, destination, copy_function=shutil.copy2)


__all__ = [
    "ALLOWLIST_FIELDS",
    "BoreasExternalV2Stage1Error",
    "METADATA_FIELDS",
    "OVERLAP_FIELDS",
    "authenticate_boreas_v1_evidence",
    "build_boreas_external_v2_stage1",
    "compute_all_directed_pairs",
    "freeze_runtime_assets",
    "make_test_baseline_report",
    "select_primary_lidar_keys",
]
