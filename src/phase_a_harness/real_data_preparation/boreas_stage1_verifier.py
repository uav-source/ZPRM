"""Independent verifier for the frozen Boreas metadata/GT Stage-1 audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .io import canonical_json_bytes, compact_sha256, sha256_file


EXPECTED_BRANCH = "prep/boreas-single-dataset-stage1-v1"
PYBOREAS_COMMIT = "e968198cd564ccfca5ad256624c80e0e584e7150"
PAPER_DOI = "10.1177/02783649231160195"
AWS_URI = "s3://boreas"
REFERENCE_SEQUENCE = "boreas-2020-11-26-13-58"
MAX_SINGLE = 500_000_000
MAX_TOTAL = 5_000_000_000
EXPECTED_DEVKIT_HASHES = {
    "README.md": "3b75572c0b3b3b81e6387ecbd4ed55b0bf79a898b218925c9df2bfa9580e5f78",
    "DATA_REFERENCE.md": "66b597c4b83ef65b5433d71da909f636ba47be0ad7193d5034e8a55715555d1b",
    "DATA_LICENSE.md": "92368d96fe04291eaa6e56c22faabd9860d34f1d4ce1ecad0f0b3ba9e953a52d",
    "LICENSE": "f40a0817c134eabbd843b1c625380b9cd5e269df92244c822228fc1f08f6ba71",
    "pyboreas/data/splits.py": "e57488ff5f3234f97ba1f321de836920c4434aa1bce32b5569475c7b943517a4",
    "pyboreas/data/calib.py": "4e8c28a7f0fb0494fa0ee0e38b377f7a879a7f2e22809b974d28fdff1b0876e7",
    "pyboreas/utils/utils.py": "f22893877b97dccdc0a97569b3dad204e3f0013a655aff4392eec5c58c8671c5",
}
TRAIN_SEQUENCES = (
    "boreas-2020-11-26-13-58", "boreas-2020-12-01-13-26", "boreas-2020-12-18-13-44",
    "boreas-2021-01-15-12-17", "boreas-2021-01-19-15-08", "boreas-2021-01-26-11-22",
    "boreas-2021-02-02-14-07", "boreas-2021-03-02-13-38", "boreas-2021-03-23-12-43",
    "boreas-2021-03-30-14-23", "boreas-2021-04-08-12-44", "boreas-2021-04-13-14-49",
    "boreas-2021-04-15-18-55", "boreas-2021-04-20-14-11", "boreas-2021-04-29-15-55",
    "boreas-2021-05-06-13-19", "boreas-2021-05-13-16-11", "boreas-2021-06-03-16-00",
    "boreas-2021-06-17-17-52", "boreas-2021-07-20-17-33", "boreas-2021-07-27-14-43",
    "boreas-2021-08-05-13-34", "boreas-2021-09-02-11-42", "boreas-2021-09-07-09-35",
    "boreas-2021-09-14-20-00", "boreas-2021-10-15-12-35", "boreas-2021-10-22-11-36",
    "boreas-2021-11-02-11-16", "boreas-2021-11-14-09-47", "boreas-2021-11-16-14-10",
    "boreas-2021-11-23-14-27",
)
TEST_SEQUENCES = (
    "boreas-2020-12-04-14-00", "boreas-2021-01-26-10-59", "boreas-2021-02-09-12-55",
    "boreas-2021-03-09-14-23", "boreas-2021-04-22-15-00", "boreas-2021-06-29-18-53",
    "boreas-2021-06-29-20-43", "boreas-2021-09-08-21-00", "boreas-2021-09-09-15-28",
    "boreas-2021-10-05-15-35", "boreas-2021-10-26-12-35", "boreas-2021-11-06-18-55",
    "boreas-2021-11-28-09-18",
)
BOREAS_SEQUENCES = TRAIN_SEQUENCES + TEST_SEQUENCES
POSE_HEADER = (
    "GPSTime", "easting", "northing", "altitude", "vel_east", "vel_north", "vel_up",
    "roll", "pitch", "heading", "angvel_z", "angvel_y", "angvel_x",
)
OVERLAP_CONTRACT = {
    "maximum_native_gap_s": 0.2,
    "min_contiguous_covered_duration_s": 5.0,
    "min_coverage_fraction": 0.60,
    "min_eligible_nonoverlapping_5s_intervals": 30,
    "min_total_covered_duration_s": 150.0,
    "radius_m": 5.0,
    "resample_rate_hz": 1.0,
}
REQUIRED_FILES = frozenset({
    "NO_ICP_ATTESTATION.json", "SHA256SUMS", "boreas_cleanup_report.json",
    "boreas_common_world_frame_audit.json", "boreas_extrinsic_independence_risk_assessment.json",
    "boreas_extrinsic_independence_risk_assessment.md", "boreas_gt_only_overlap_matrix.csv",
    "boreas_gt_only_overlap_matrix.json", "boreas_lidar_extrinsic_provenance.json",
    "boreas_pair_selection_v1.json", "boreas_reference_trajectory_provenance.json",
    "boreas_remote_sequence_inventory.csv", "boreas_remote_sequence_inventory.json",
    "boreas_stage1_eligibility.json", "boreas_stage1_manifest.json", "boreas_stage1_summary.json",
    "boreas_stage1_summary.md", "boreas_stage1_uncertainty_feasibility.csv",
    "boreas_stage1_uncertainty_feasibility.json", "boreas_time_sync_audit.json",
    "boreas_transform_chain_consistency.json", "boreas_transform_chain_manifest.json",
    "dataset_citation_report.md", "download_manifest.csv", "download_manifest.json",
    "eligible_reference_sequences.csv", "environment_report.json", "official_source_manifest.json",
    "pyboreas_source_manifest.json",
})


class BoreasStage1VerificationError(RuntimeError):
    """The Boreas evidence closure or its scientific meaning was altered."""


def _fail(message: str) -> None:
    raise BoreasStage1VerificationError(message)


def _load(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    if canonical_json_bytes(value) != path.read_bytes():
        _fail(f"JSON is not canonical: {path.name}")
    return value


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} mismatch: {actual!r} != {expected!r}")


def _csv_rows(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _same(reader.fieldnames, list(fields), f"CSV header {path.name}")
        return list(reader)


def _string_row(row: Mapping[str, Any], fields: Sequence[str]) -> dict[str, str]:
    return {field: "" if row.get(field) is None else str(row.get(field)) for field in fields}


def _parse_sums(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None or match.group(2) in rows:
            _fail("malformed or duplicate SHA256SUMS row")
        rows[match.group(2)] = match.group(1)
    return rows


def _verify_closure(root: Path) -> Mapping[str, Any]:
    if root.is_symlink() or not root.is_dir():
        _fail("runtime root is absent or a symlink")
    children = {path.name for path in root.iterdir() if path.is_file()}
    if children != REQUIRED_FILES or any(path.is_symlink() for path in root.iterdir()):
        _fail(f"runtime file closure mismatch: missing={sorted(REQUIRED_FILES-children)}, extra={sorted(children-REQUIRED_FILES)}")
    sums = _parse_sums(root)
    if set(sums) != children - {"SHA256SUMS"}:
        _fail("SHA256SUMS is not the exact runtime closure")
    for name, expected in sums.items():
        if sha256_file(root / name) != expected:
            _fail(f"SHA256 mismatch: {name}")
    manifest = _load(root / "boreas_stage1_manifest.json")
    payload_hash = manifest.get("manifest_payload_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    _same(payload_hash, compact_sha256(unsigned), "manifest payload SHA")
    rows = manifest.get("payload")
    if not isinstance(rows, list):
        _fail("manifest payload rows are absent")
    expected_payload = children - {"SHA256SUMS", "boreas_stage1_manifest.json"}
    if {row.get("path") for row in rows} != expected_payload or len(rows) != len(expected_payload):
        _fail("manifest payload file closure mismatch")
    for row in rows:
        path = root / row["path"]
        _same(row.get("sha256"), sha256_file(path), f"manifest SHA {path.name}")
        _same(row.get("size_bytes"), path.stat().st_size, f"manifest size {path.name}")
    _same(manifest.get("eligibility_sha256"), sha256_file(root / "boreas_stage1_eligibility.json"), "eligibility SHA")
    _same(manifest.get("schema_version"), "boreas_stage1_manifest_v1", "manifest schema")
    identity = manifest.get("official_source_identity", {})
    _same(identity, {"aws_bucket": AWS_URI, "paper_doi": PAPER_DOI, "pyboreas_commit": PYBOREAS_COMMIT}, "official source identity")
    commit = manifest.get("producer_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        _fail("producer commit is invalid")
    return manifest


def _resolve_evidence(data_root: Path, row: Mapping[str, Any]) -> Path:
    relative = row.get("relative_path")
    local = row.get("local_path")
    if not isinstance(relative, str) or not isinstance(local, str):
        _fail("data evidence path is invalid")
    candidate = data_root / relative
    if candidate.resolve(strict=True) != Path(local).resolve(strict=True):
        _fail(f"data evidence path binding mismatch: {relative}")
    if candidate.is_symlink() or data_root.resolve() not in candidate.resolve().parents:
        _fail(f"unsafe data evidence path: {relative}")
    _same(row.get("size_bytes"), candidate.stat().st_size, f"evidence size {relative}")
    _same(row.get("sha256"), sha256_file(candidate), f"evidence SHA {relative}")
    return candidate


def _read_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if tuple(next(reader, ()))[: len(POSE_HEADER)] != POSE_HEADER:
            _fail(f"pose header mismatch: {path}")
        try:
            values = np.asarray([[float(cell) for cell in row] for row in reader if row], dtype=np.float64)
        except ValueError as error:
            raise BoreasStage1VerificationError(f"nonnumeric pose row: {path}") from error
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 13 or not np.isfinite(values).all():
        _fail(f"invalid pose body: {path}")
    scale = 1e-6 if np.median(values[:, 0]) > 1e12 else 1.0
    times = values[:, 0] * scale
    if not np.all(np.diff(times) > 0.0):
        _fail(f"pose timestamps not strictly monotonic: {path}")
    return times, values


def _roll(value: float) -> np.ndarray:
    return np.array([[1, 0, 0], [0, math.cos(value), math.sin(value)], [0, -math.sin(value), math.cos(value)]])


def _pitch(value: float) -> np.ndarray:
    return np.array([[math.cos(value), 0, -math.sin(value)], [0, 1, 0], [math.sin(value), 0, math.cos(value)]])


def _yaw(value: float) -> np.ndarray:
    return np.array([[math.cos(value), math.sin(value), 0], [-math.sin(value), math.cos(value), 0], [0, 0, 1]])


def _pose(row: Sequence[float]) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = _roll(row[7]) @ _pitch(row[8]) @ _yaw(row[9])
    value[:3, 3] = row[1:4]
    return value


def _matrix(path: Path) -> np.ndarray:
    value = np.loadtxt(path, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        _fail("extrinsic matrix is invalid")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-12, rtol=0.0):
        _fail("extrinsic homogeneous row is invalid")
    if not np.allclose(value[:3, :3] @ value[:3, :3].T, np.eye(3), atol=1e-10, rtol=0.0):
        _fail("extrinsic rotation is invalid")
    return value


def _chain(gps_times: np.ndarray, gps: np.ndarray, lidar_times: np.ndarray, lidar: np.ndarray, extrinsic: np.ndarray) -> dict[str, Any]:
    valid = np.flatnonzero((lidar_times >= gps_times[0]) & (lidar_times <= gps_times[-1]))
    if valid.size < 10:
        _fail("insufficient pose-chain overlap")
    indices = valid[np.linspace(0, valid.size - 1, min(101, valid.size), dtype=int)]
    trans: list[float] = []
    rot: list[float] = []
    gaps: list[float] = []
    for index in indices:
        query = lidar_times[index]
        right = min(max(int(np.searchsorted(gps_times, query, side="right")), 1), len(gps_times) - 1)
        left = right - 1
        gap = float(gps_times[right] - gps_times[left])
        alpha = (query - gps_times[left]) / gap
        predicted = np.eye(4)
        predicted[:3, 3] = (1 - alpha) * gps[left, 1:4] + alpha * gps[right, 1:4]
        rotations = Rotation.from_matrix(np.stack((_pose(gps[left])[:3, :3], _pose(gps[right])[:3, :3])))
        predicted[:3, :3] = Slerp([gps_times[left], gps_times[right]], rotations)([query]).as_matrix()[0]
        predicted = predicted @ extrinsic
        observed = _pose(lidar[index])
        trans.append(float(np.linalg.norm(predicted[:3, 3] - observed[:3, 3])))
        rot.append(float(Rotation.from_matrix(predicted[:3, :3] @ observed[:3, :3].T).magnitude()))
        gaps.append(gap)
    return {
        "maximum_interpolation_gap_s": max(gaps),
        "maximum_rotation_error_rad": max(rot),
        "maximum_translation_error_m": max(trans),
        "median_rotation_error_rad": float(np.median(rot)),
        "median_translation_error_m": float(np.median(trans)),
        "sample_count": len(indices),
    }


def _verify_source_and_inventory(root: Path, data_root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    download = _load(root / "download_manifest.json")
    _same(download.get("maximum_single_download_bytes"), MAX_SINGLE, "single-object ceiling")
    _same(download.get("maximum_total_download_bytes"), MAX_TOTAL, "total ceiling")
    _same(download.get("full_lidar_binary_download_count"), 0, "LiDAR binary download count")
    _same(download.get("lidar_payload_count"), 0, "LiDAR payload count")
    rows = download.get("materialized_files")
    if not isinstance(rows, list) or len(rows) != 83:
        _fail("download evidence row count mismatch")
    paths: dict[str, Path] = {}
    s3_bytes = 0
    s3_count = 0
    for row in rows:
        path = _resolve_evidence(data_root, row)
        relative = row["relative_path"]
        paths[relative] = path
        if relative.endswith(".bin") or "/lidar/" in relative:
            _fail("forbidden lidar payload was materialized")
        if row.get("status") == "DOWNLOADED_ALLOWLISTED_STAGE1_OBJECT":
            if row["size_bytes"] > MAX_SINGLE:
                _fail("downloaded object exceeded frozen ceiling")
            if row.get("s3_key") is None or row.get("etag") is None or row.get("last_modified") is None:
                _fail("downloaded object lacks exact S3 provenance")
            if not str(row["relative_path"]).endswith(str(row["s3_key"])):
                _fail("S3 key/local path binding mismatch")
            s3_bytes += row["size_bytes"]
            s3_count += 1
    _same(s3_count, 76, "selected S3 object count")
    _same(download.get("downloaded_s3_object_count"), s3_count, "download manifest object count")
    _same(download.get("downloaded_s3_bytes"), s3_bytes, "download manifest byte count")
    if s3_bytes > MAX_TOTAL:
        _fail("Stage-1 download total exceeded frozen ceiling")
    download_fields = (
        "relative_path", "local_path", "s3_key", "size_bytes", "sha256", "etag",
        "last_modified", "version_id", "status",
    )
    _same(
        _csv_rows(root / "download_manifest.csv", download_fields),
        [_string_row(row, download_fields) for row in rows],
        "download CSV/JSON projection",
    )
    for relative, expected in EXPECTED_DEVKIT_HASHES.items():
        path = data_root / "pyboreas" / relative
        _same(sha256_file(path), expected, f"pinned devkit hash {relative}")
    source = _load(root / "official_source_manifest.json")
    _same(source.get("aws_bucket"), AWS_URI, "S3 source")
    _same(source.get("paper_doi"), PAPER_DOI, "paper DOI")
    _same(source.get("pyboreas_commit"), PYBOREAS_COMMIT, "pyboreas commit")
    _same(source.get("pinned_source_sha256"), EXPECTED_DEVKIT_HASHES, "official source hashes")
    if source.get("static_no_registration_audit", {}).get("pass") is not True:
        _fail("static no-registration audit did not pass")
    pyboreas = _load(root / "pyboreas_source_manifest.json")
    _same(pyboreas.get("commit"), PYBOREAS_COMMIT, "pyboreas manifest commit")
    if compact_sha256(pyboreas.get("files")) != pyboreas.get("tree_rows_sha256"):
        _fail("pyboreas tree digest mismatch")
    live_devkit_rows = []
    for path in sorted((data_root / "pyboreas").rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            if path.is_symlink():
                _fail("pyboreas checkout contains a symlink")
            live_devkit_rows.append({
                "path": path.relative_to(data_root / "pyboreas").as_posix(),
                "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            })
    _same(pyboreas.get("files"), live_devkit_rows, "live pyboreas tree rows")
    inventory = _load(root / "boreas_remote_sequence_inventory.json")
    sequence_rows = inventory.get("boreas_original_sequences")
    if not isinstance(sequence_rows, list) or [row.get("sequence_id") for row in sequence_rows] != list(BOREAS_SEQUENCES):
        _fail("official original Boreas sequence inventory mismatch")
    _same(inventory.get("boreas_original_sequence_count"), 44, "original sequence count")
    prefixes = inventory.get("all_s3_boreas_named_prefixes")
    if not isinstance(prefixes, list) or prefixes != sorted(set(prefixes)) or not set(BOREAS_SEQUENCES).issubset(prefixes):
        _fail("top-level S3 sequence prefix inventory is invalid")
    _same(inventory.get("boreas_rt_or_non_original_prefix_count"), len(set(prefixes)-set(BOREAS_SEQUENCES)), "non-original prefix count")
    for row in sequence_rows:
        sequence = row["sequence_id"]
        expected_status = "TRAIN_PUBLIC_GT" if sequence in TRAIN_SEQUENCES else "TEST_GT_HIDDEN"
        _same(row.get("test_or_training_status"), expected_status, f"split status {sequence}")
        _same(row.get("ground_truth_candidate"), sequence in TRAIN_SEQUENCES, f"GT eligibility {sequence}")
        if not all(row.get(key) is True for key in ("applanix_available", "calib_available", "T_applanix_lidar_available")):
            _fail(f"required S3 metadata missing: {sequence}")
        if not isinstance(row.get("lidar_object_count"), int) or row["lidar_object_count"] <= 0:
            _fail(f"invalid remote LiDAR count: {sequence}")
        if not isinstance(row.get("lidar_remote_bytes"), int) or row["lidar_remote_bytes"] <= 0:
            _fail(f"invalid remote LiDAR byte count: {sequence}")
        first, last = row.get("first_lidar_timestamp"), row.get("last_lidar_timestamp")
        if not isinstance(first, int) or not isinstance(last, int) or first >= last:
            _fail(f"invalid remote LiDAR timestamp range: {sequence}")
        if re.fullmatch(r"[0-9a-f]{64}", str(row.get("lidar_listing_rows_sha256"))) is None:
            _fail(f"invalid remote LiDAR listing digest: {sequence}")
    inventory_fields = (
        "sequence_id", "applanix_available", "gps_post_process_available", "lidar_poses_available",
        "calib_available", "T_applanix_lidar_available", "lidar_object_count", "lidar_remote_bytes",
        "first_lidar_timestamp", "last_lidar_timestamp", "ground_truth_candidate", "test_or_training_status",
    )
    expected_inventory_csv = [
        _string_row({key: row[key] for key in inventory_fields}, inventory_fields)
        for row in sequence_rows
    ]
    _same(
        _csv_rows(root / "boreas_remote_sequence_inventory.csv", inventory_fields),
        expected_inventory_csv,
        "remote inventory CSV/JSON projection",
    )
    _same(source.get("remote_inventory_sha256"), sha256_file(root / "boreas_remote_sequence_inventory.json"), "remote inventory SHA")
    _same(manifest.get("data_evidence"), rows, "manifest/download data evidence")
    return paths


def _verify_science(root: Path, data_root: Path, paths: Mapping[str, Path]) -> None:
    inventory = _load(root / "boreas_remote_sequence_inventory.json")
    remote = {row["sequence_id"]: row for row in inventory["boreas_original_sequences"]}
    trajectories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    resets: list[str] = []
    maximum_position = 0.0
    for index, sequence in enumerate(TRAIN_SEQUENCES):
        relative = f"stage1_payload/{sequence}/applanix/lidar_poses.csv"
        if relative not in paths:
            _fail(f"public GT trajectory absent: {sequence}")
        times, values = _read_pose(paths[relative])
        trajectories[sequence] = (times, values)
        maximum_position = max(maximum_position, float(np.max(np.linalg.norm(values[:, 1:4], axis=1))))
        if index and np.linalg.norm(values[0, 1:4]) < 1e-6:
            resets.append(sequence)
        _same(values.shape[0], remote[sequence]["lidar_object_count"], f"LiDAR pose/object count {sequence}")
        _same(int(round(times[0]*1e6)), remote[sequence]["first_lidar_timestamp"], f"first LiDAR timestamp {sequence}")
        _same(int(round(times[-1]*1e6)), remote[sequence]["last_lidar_timestamp"], f"last LiDAR timestamp {sequence}")
    eligible_fields = (
        "sequence_id", "status", "gps_post_process_available", "lidar_poses_available",
        "calibration_available", "row_count", "duration_s", "median_rate_hz", "maximum_gap_s",
        "first_timestamp_s", "last_timestamp_s", "reference_frame", "pose_sha256",
    )
    eligible_rows = _csv_rows(root / "eligible_reference_sequences.csv", eligible_fields)
    _same([row["sequence_id"] for row in eligible_rows], list(TRAIN_SEQUENCES), "eligible GT sequence order")
    for row in eligible_rows:
        sequence = row["sequence_id"]
        times, values = trajectories[sequence]
        _same(row["status"], "ELIGIBLE_PUBLIC_GT", f"eligible status {sequence}")
        _same(row["gps_post_process_available"], "True", f"GPS availability {sequence}")
        _same(row["lidar_poses_available"], "True", f"LiDAR pose availability {sequence}")
        _same(row["calibration_available"], "True", f"calibration availability {sequence}")
        _same(int(row["row_count"]), values.shape[0], f"eligible row count {sequence}")
        _same(row["reference_frame"], "ENU_ref", f"eligible frame {sequence}")
        _same(row["pose_sha256"], sha256_file(paths[f"stage1_payload/{sequence}/applanix/lidar_poses.csv"]), f"eligible GT SHA {sequence}")
        if not math.isclose(float(row["duration_s"]), float(times[-1]-times[0]), abs_tol=1e-12, rel_tol=1e-12):
            _fail(f"eligible duration mismatch: {sequence}")
    world = _load(root / "boreas_common_world_frame_audit.json")
    _same(world.get("status"), "PASS", "common-world status")
    _same(world.get("COMMON_WORLD_FRAME"), "PASS", "common-world declaration")
    _same(world.get("sequence_local_reset_detected"), False, "sequence-local reset status")
    _same(world.get("sequence_local_reset_ids"), resets, "sequence-local reset IDs")
    if resets or maximum_position <= 100.0 or world.get("registration_or_trajectory_fitting_used") is not False:
        _fail("actual trajectories do not support fixed ENU_ref without fitting")
    if "first pose of the first sequence" not in world.get("official_definition", ""):
        _fail("fixed ENU_ref definition was altered")

    matrices: dict[str, np.ndarray] = {}
    matrix_hashes: set[str] = set()
    for sequence in BOREAS_SEQUENCES:
        relative = f"stage1_payload/{sequence}/calib/T_applanix_lidar.txt"
        if relative not in paths:
            _fail(f"calibration evidence absent: {sequence}")
        matrices[sequence] = _matrix(paths[relative])
        matrix_hashes.add(sha256_file(paths[relative]))
    extrinsic = _load(root / "boreas_lidar_extrinsic_provenance.json")
    _same(extrinsic.get("uses_lidar_pointclouds"), True, "LiDAR-assisted extrinsic provenance")
    _same(extrinsic.get("uses_post_processed_gps_imu"), True, "GPS/IMU extrinsic provenance")
    _same(extrinsic.get("query_sequence_used_for_calibration"), "UNKNOWN", "query calibration provenance")
    _same(extrinsic.get("extrinsic_uncertainty"), "UNKNOWN", "extrinsic uncertainty")
    _same(extrinsic.get("frozen_R02_independence_violation"), True, "R02 extrinsic violation")
    _same(extrinsic.get("all_sequences_byte_identical"), len(matrix_hashes) == 1, "extrinsic byte identity")
    _same(extrinsic.get("unique_matrix_sha256"), sorted(matrix_hashes), "extrinsic SHA inventory")

    gps_path = paths[f"stage1_payload/{REFERENCE_SEQUENCE}/applanix/gps_post_process.csv"]
    gps_times, gps_values = _read_pose(gps_path)
    lidar_times, lidar_values = trajectories[REFERENCE_SEQUENCE]
    calculated = _chain(gps_times, gps_values, lidar_times, lidar_values, matrices[REFERENCE_SEQUENCE])
    consistency = _load(root / "boreas_transform_chain_consistency.json")
    for key, value in calculated.items():
        if isinstance(value, float):
            if not math.isclose(float(consistency.get(key)), value, abs_tol=1e-12, rel_tol=1e-12):
                _fail(f"transform-chain numeric mismatch: {key}")
        else:
            _same(consistency.get(key), value, f"transform-chain {key}")
    _same(consistency.get("composition"), "T_ENU_lidar(t)=T_ENU_applanix(t)@T_applanix_lidar", "transform composition")
    _same(consistency.get("direction"), "T_applanix_lidar maps lidar coordinates into the Applanix frame", "transform direction")
    _same(consistency.get("status"), "PASS", "transform consistency status")
    if calculated["maximum_translation_error_m"] > 1e-5 or calculated["maximum_rotation_error_rad"] > 1e-7:
        _fail("transform consistency exceeds frozen numerical limit")
    chain = _load(root / "boreas_transform_chain_manifest.json")
    if not np.array_equal(np.asarray(chain.get("matrix")), matrices[REFERENCE_SEQUENCE]):
        _fail("recorded extrinsic matrix was altered")
    _same(chain.get("source_sha256"), sha256_file(paths[f"stage1_payload/{REFERENCE_SEQUENCE}/calib/T_applanix_lidar.txt"]), "transform source SHA")
    _same(chain.get("inverse_used"), False, "transform inverse flag")
    _same(chain.get("status"), "PASS", "transform manifest status")

    reference = _load(root / "boreas_reference_trajectory_provenance.json")
    for key in ("uses_gnss", "uses_imu", "uses_wheel_encoder", "uses_rtx"):
        _same(reference.get(key), True, f"reference provenance {key}")
    for key in ("uses_lidar", "uses_scan_matching", "uses_icp"):
        _same(reference.get(key), False, f"reference provenance {key}")
    claim = reference.get("position_accuracy_claim", {})
    _same(claim.get("nominal_range_m"), [0.02, 0.04], "nominal position RMSE")
    _same(claim.get("urban_canyon_range_m"), [0.20, 0.40], "urban-canyon RMSE")
    _same(claim.get("uncertainty_type"), "RMSE", "position uncertainty type")
    if not math.isclose(float(reference.get("reference_rate_hz")), 200.0, abs_tol=0.1):
        _fail("reference rate is not approximately 200 Hz")

    time = _load(root / "boreas_time_sync_audit.json")
    for key in ("hardware_sync_available", "pps_available", "nmea_available", "per_point_timestamp_documented", "timestamp_monotonicity"):
        _same(time.get(key), True, f"time-chain {key}")
    _same(time.get("timestamp_epoch"), "Unix UTC", "time epoch")
    _same(time.get("time_uncertainty_status"), "UNKNOWN", "time uncertainty status")
    _same(time.get("zero_time_uncertainty_assumed"), False, "time zero assumption")
    _same(time.get("status"), "PASS_WITH_DOCUMENTED_LIMITATION", "time-chain status")

    risk = _load(root / "boreas_extrinsic_independence_risk_assessment.json")
    _same(risk.get("risk_status"), "FAIL_CLOSED_R02", "extrinsic risk status")
    _same(risk.get("query_sequence_participation"), "UNKNOWN", "extrinsic query involvement")
    if "proprietary static extrinsic" not in risk.get("rationale", ""):
        _fail("LiDAR-assisted calibration rationale was hidden")

    uncertainty = _load(root / "boreas_stage1_uncertainty_feasibility.json")
    _same(uncertainty.get("status"), "PARTIAL", "R10 status")
    _same(uncertainty.get("rmse_not_relabelled_as_1sigma"), True, "RMSE semantic guard")
    _same(uncertainty.get("unknown_not_zero"), True, "UNKNOWN semantic guard")
    rows = uncertainty.get("rows")
    if not isinstance(rows, list) or len(rows) != 10:
        _fail("R10 component inventory mismatch")
    components = {row.get("component"): row for row in rows}
    position = components.get("GNSS/RTX position uncertainty", {})
    _same(position.get("uncertainty_type"), "RMSE", "2-4 cm uncertainty type")
    if "0.02-0.04" not in str(position.get("value")):
        _fail("2-4 cm range was altered")
    expected_unknown = {
        "orientation uncertainty", "Applanix time uncertainty", "Velodyne time synchronization uncertainty",
        "T_applanix_lidar translation uncertainty", "T_applanix_lidar rotation uncertainty",
        "GT interpolation uncertainty", "future deskew uncertainty", "future target-map accumulation uncertainty",
    }
    if not expected_unknown.issubset(components):
        _fail("R10 UNKNOWN component missing")
    for component in expected_unknown:
        row = components[component]
        _same(row.get("value"), "UNKNOWN", f"R10 value {component}")
        _same(row.get("uncertainty_type"), "UNKNOWN", f"R10 type {component}")
        _same(row.get("status"), "UNKNOWN", f"R10 status {component}")
    uncertainty_fields = (
        "component", "value", "unit", "uncertainty_type", "evidence_type", "source",
        "source_sha256", "status",
    )
    _same(
        _csv_rows(root / "boreas_stage1_uncertainty_feasibility.csv", uncertainty_fields),
        [_string_row(row, uncertainty_fields) for row in rows],
        "uncertainty CSV/JSON projection",
    )

    overlap = _load(root / "boreas_gt_only_overlap_matrix.json")
    _same(overlap.get("contract"), OVERLAP_CONTRACT, "frozen overlap contract")
    _same(overlap.get("gate_status"), "NOT_COMPUTABLE", "overlap gate")
    _same(overlap.get("pair_rows"), [], "overlap rows")
    _same(overlap.get("reason"), "R02_FAIL_LIDAR_ASSISTED_EXTRINSIC_IN_FINAL_REFERENCE_CHAIN", "overlap gate reason")
    with (root / "boreas_gt_only_overlap_matrix.csv").open("r", encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    _same(csv_rows, [], "overlap CSV rows")
    pair = _load(root / "boreas_pair_selection_v1.json")
    for key in ("PRIMARY_PAIR", "RESERVE_PAIR_1", "RESERVE_PAIR_2"):
        _same(pair.get(key), None, f"pair selection {key}")
    _same(pair.get("eligible_pair_count"), 0, "eligible overlap pair count")


def _verify_decision(root: Path) -> Mapping[str, Any]:
    eligibility = _load(root / "boreas_stage1_eligibility.json")
    expected = {
        "BOREAS_STAGE1_READY": False, "GT_ONLY_OVERLAP": "NOT_COMPUTABLE",
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False, "R01": "PENDING_SECOND_DATASET",
        "R02": "FAIL", "R03": "BLOCKED_STAGE1", "R04": "PASS", "R05": "PASS",
        "R07": "BLOCKED_STAGE1", "R08": "BLOCKED_STAGE1", "R09": "PASS",
        "R10": "PARTIAL", "R14": "BLOCKED_STAGE1", "REAL_DATA_RUN_AUTHORIZED": False,
        "SINGLE_DATASET_PREREGISTRATION_READY": False, "planned_future_trials": 0,
        "registration_execution_count": 0, "rich_snapshot_count": 0, "snapshot_count": 0,
        "weak_snapshot_count": 0,
    }
    for key, value in expected.items():
        _same(eligibility.get(key), value, f"eligibility {key}")
    blockers = eligibility.get("primary_hard_blockers")
    if blockers != ["R02_LIDAR_ASSISTED_EXTRINSIC_INDEPENDENCE", "GT_ONLY_OVERLAP_GATED", "R10_UNCERTAINTY_FEASIBILITY"]:
        _fail("hard-blocker set was altered")
    attestation = _load(root / "NO_ICP_ATTESTATION.json")
    for key in (
        "open3d_registration_call_count", "pcl_cli_invocation_count", "other_registration_process_count",
        "estimated_transform_count", "registration_execution_count", "real_trial_result_count",
        "lidar_payload_count",
    ):
        _same(attestation.get(key), 0, f"NO-ICP {key}")
    _same(attestation.get("pass"), True, "NO-ICP pass")
    _same(attestation.get("status"), "PASS", "NO-ICP status")
    summary = _load(root / "boreas_stage1_summary.json")
    _same(summary.get("boreas_stage1_ready"), False, "summary readiness")
    _same(summary.get("eligibility"), eligibility, "summary eligibility")
    answers = summary.get("answers")
    if not isinstance(answers, list) or len(answers) != 29:
        _fail("summary must contain exactly 29 answers")
    if "R02 independence" not in summary.get("final_conclusion", ""):
        _fail("summary hard failure was altered")
    cleanup = _load(root / "boreas_cleanup_report.json")
    _same(cleanup.get("cavers_data_root_absent"), True, "CAVERS cleanup status")
    if Path(cleanup.get("cavers_data_root", "")).exists():
        _fail("CAVERS data root unexpectedly exists")
    return summary


def verify_boreas_stage1(*, repository: str | Path, data_root: str | Path, runtime_root: str | Path) -> dict[str, Any]:
    repository = Path(repository).resolve(strict=True)
    data_root = Path(data_root).resolve(strict=True)
    runtime_root = Path(runtime_root).resolve(strict=True)
    manifest = _verify_closure(runtime_root)
    producer = manifest["producer_commit"]
    if subprocess.run(["git", "merge-base", "--is-ancestor", producer, "HEAD"], cwd=repository).returncode:
        _fail("producer commit is not an ancestor of current HEAD")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repository, text=True).strip()
    _same(branch, EXPECTED_BRANCH, "verification branch")
    devkit = data_root / "pyboreas"
    _same(subprocess.check_output(["git", "-C", str(devkit), "rev-parse", "HEAD"], text=True).strip(), PYBOREAS_COMMIT, "live pyboreas commit")
    if subprocess.check_output(["git", "-C", str(devkit), "status", "--porcelain"], text=True).strip():
        _fail("live pyboreas checkout is dirty")
    paths = _verify_source_and_inventory(runtime_root, data_root, manifest)
    _verify_science(runtime_root, data_root, paths)
    summary = _verify_decision(runtime_root)
    return {
        "BOREAS_STAGE1_VERIFICATION_PASS": True,
        "data_evidence_file_count": len(paths),
        "original_sequence_count": len(BOREAS_SEQUENCES),
        "public_gt_sequence_count": len(TRAIN_SEQUENCES),
        "registration_execution_count": 0,
        "summary": summary,
        "verification_pass": True,
    }


__all__ = ["BoreasStage1VerificationError", "verify_boreas_stage1"]
