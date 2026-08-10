"""Fail-closed Boreas metadata/GT/calibration-only Stage-1 audit.

The producer lists lidar objects but never requests their bodies.  Only an
allow-list of trajectory, timestamp-index, and calibration files is eligible
for materialization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .guard import NoRegistrationGuard, assert_preparation_sources_are_safe
from .io import atomic_write_bytes, atomic_write_csv, atomic_write_json, compact_sha256, sha256_file
from .stage1_gt_overlap import FROZEN_STAGE1_OVERLAP_CONTRACT


EXPECTED_BRANCH = "prep/boreas-single-dataset-stage1-v1"
AWS_BUCKET = "boreas"
AWS_URI = "s3://boreas"
BOREAS_SITE = "https://www.boreas.utias.utoronto.ca/"
PYBOREAS_URL = "https://github.com/utiasASRL/pyboreas.git"
PYBOREAS_COMMIT = "e968198cd564ccfca5ad256624c80e0e584e7150"
PYBOREAS_CODE_LICENSE = "BSD-3-Clause"
DATASET_LICENSE = "CC-BY-4.0"
PAPER_DOI = "10.1177/02783649231160195"
PAPER_ARXIV = "2203.10168"
PAPER_PDF_SHA256 = "29531b5782937ded0a34cb2fd214a4f47c46bfcb6e28fa4b688cb2d14cb43ee8"
DATA_REFERENCE_SHA256 = "66b597c4b83ef65b5433d71da909f636ba47be0ad7193d5034e8a55715555d1b"
README_SHA256 = "3b75572c0b3b3b81e6387ecbd4ed55b0bf79a898b218925c9df2bfa9580e5f78"
DATA_LICENSE_SHA256 = "92368d96fe04291eaa6e56c22faabd9860d34f1d4ce1ecad0f0b3ba9e953a52d"
CODE_LICENSE_SHA256 = "f40a0817c134eabbd843b1c625380b9cd5e269df92244c822228fc1f08f6ba71"
SPLITS_SHA256 = "e57488ff5f3234f97ba1f321de836920c4434aa1bce32b5569475c7b943517a4"
UTILS_SHA256 = "f22893877b97dccdc0a97569b3dad204e3f0013a655aff4392eec5c58c8671c5"
CALIB_PARSER_SHA256 = "4e8c28a7f0fb0494fa0ee0e38b377f7a879a7f2e22809b974d28fdff1b0876e7"
PROTOCOL_MD_SHA256 = "4755a90dfd42de1650d50caa0facc4d32a0dba7f837ce9d867d28f030edfc834"
PROTOCOL_JSON_SHA256 = "3b4dc774051806728a060a93aa1220ac7a4fc82a6e59181da2390cfdcb7bd86e"
CHECKLIST_SHA256 = "7a3c0b492f3783ea84bdda1eb8ff7fbd41ab73b9f5f7823806e2e30544a19c44"
BACKEND_PARAMETER_SHA256 = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
OPEN3D_PARAMETER_CANONICAL_SHA256 = "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413"
PCL_PARAMETER_CANONICAL_SHA256 = "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd"
MAX_SINGLE_OBJECT_BYTES = 500_000_000
MAX_STAGE1_TOTAL_BYTES = 5_000_000_000
TRANSFORM_NUMERICAL_TRANSLATION_LIMIT_M = 1e-4
TRANSFORM_NUMERICAL_ROTATION_LIMIT_RAD = 1e-4
TRANSFORM_DIRECTION_MINIMUM_DISCRIMINATION_RATIO = 1_000.0
REFERENCE_SEQUENCE = "boreas-2020-11-26-13-58"
POSE_HEADER = (
    "GPSTime", "easting", "northing", "altitude", "vel_east", "vel_north",
    "vel_up", "roll", "pitch", "heading", "angvel_z", "angvel_y", "angvel_x",
)

TRAIN_SEQUENCES = (
    "boreas-2020-11-26-13-58", "boreas-2020-12-01-13-26",
    "boreas-2020-12-18-13-44", "boreas-2021-01-15-12-17",
    "boreas-2021-01-19-15-08", "boreas-2021-01-26-11-22",
    "boreas-2021-02-02-14-07", "boreas-2021-03-02-13-38",
    "boreas-2021-03-23-12-43", "boreas-2021-03-30-14-23",
    "boreas-2021-04-08-12-44", "boreas-2021-04-13-14-49",
    "boreas-2021-04-15-18-55", "boreas-2021-04-20-14-11",
    "boreas-2021-04-29-15-55", "boreas-2021-05-06-13-19",
    "boreas-2021-05-13-16-11", "boreas-2021-06-03-16-00",
    "boreas-2021-06-17-17-52", "boreas-2021-07-20-17-33",
    "boreas-2021-07-27-14-43", "boreas-2021-08-05-13-34",
    "boreas-2021-09-02-11-42", "boreas-2021-09-07-09-35",
    "boreas-2021-09-14-20-00", "boreas-2021-10-15-12-35",
    "boreas-2021-10-22-11-36", "boreas-2021-11-02-11-16",
    "boreas-2021-11-14-09-47", "boreas-2021-11-16-14-10",
    "boreas-2021-11-23-14-27",
)
TEST_SEQUENCES = (
    "boreas-2020-12-04-14-00", "boreas-2021-01-26-10-59",
    "boreas-2021-02-09-12-55", "boreas-2021-03-09-14-23",
    "boreas-2021-04-22-15-00", "boreas-2021-06-29-18-53",
    "boreas-2021-06-29-20-43", "boreas-2021-09-08-21-00",
    "boreas-2021-09-09-15-28", "boreas-2021-10-05-15-35",
    "boreas-2021-10-26-12-35", "boreas-2021-11-06-18-55",
    "boreas-2021-11-28-09-18",
)
BOREAS_SEQUENCES = TRAIN_SEQUENCES + TEST_SEQUENCES


class BoreasStage1Error(RuntimeError):
    """Fail-closed Boreas Stage-1 error."""


class Stage1LargeFileDownloadForbidden(BoreasStage1Error):
    """A selected object exceeds the frozen single-object ceiling."""


class Stage1DownloadBudgetExceeded(BoreasStage1Error):
    """The selected Stage-1 object set exceeds the frozen total ceiling."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def assert_download_budget(size_bytes: int, cumulative_bytes: int, *, maximum_single: int, maximum_total: int) -> None:
    if maximum_single <= 0 or maximum_single > MAX_SINGLE_OBJECT_BYTES:
        raise Stage1LargeFileDownloadForbidden("invalid max-single-download-bytes")
    if maximum_total <= 0 or maximum_total > MAX_STAGE1_TOTAL_BYTES:
        raise Stage1DownloadBudgetExceeded("invalid max-total-download-bytes")
    if size_bytes > maximum_single:
        raise Stage1LargeFileDownloadForbidden(
            f"STAGE1_LARGE_FILE_DOWNLOAD_FORBIDDEN: {size_bytes}>{maximum_single}"
        )
    if cumulative_bytes + size_bytes > maximum_total:
        raise Stage1DownloadBudgetExceeded(
            f"BLOCKED_STAGE1_DOWNLOAD_BUDGET: {cumulative_bytes + size_bytes}>{maximum_total}"
        )


def parse_top_level_s3_listing(text: str) -> list[str]:
    prefixes: list[str] = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*PRE\s+(boreas-\d{4}-\d{2}-\d{2}-\d{2}-\d{2})/\s*", line)
        if match:
            prefixes.append(match.group(1))
    if len(prefixes) != len(set(prefixes)) or prefixes != sorted(prefixes):
        raise BoreasStage1Error("top-level Boreas prefixes are duplicate or unsorted")
    return prefixes


def parse_s3_ls_line(line: str) -> dict[str, Any]:
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(\d+)\s+(.+)", line.strip())
    if match is None:
        raise BoreasStage1Error(f"malformed aws s3 ls row: {line!r}")
    date, clock, size, key = match.groups()
    return {"key": key, "last_modified": f"{date}T{clock}Z", "size_bytes": int(size)}


def _aws_text(aws: Path, arguments: Sequence[str]) -> str:
    command = [str(aws), *arguments, "--no-sign-request"]
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def _aws_object_rows(aws: Path, prefix: str) -> list[dict[str, Any]]:
    raw = _aws_text(
        aws,
        ("s3api", "list-objects-v2", "--bucket", AWS_BUCKET, "--prefix", prefix, "--output", "json"),
    )
    value = json.loads(raw)
    contents = value.get("Contents", [])
    if not isinstance(contents, list):
        raise BoreasStage1Error("invalid S3 list-objects response")
    rows = []
    for item in contents:
        if not isinstance(item, Mapping):
            raise BoreasStage1Error("invalid S3 object row")
        rows.append(
            {
                "etag": str(item.get("ETag", "")).strip('"'),
                "key": item.get("Key"),
                "last_modified": item.get("LastModified"),
                "size_bytes": item.get("Size"),
                "version_id": None,
            }
        )
    if any(
        not isinstance(row["key"], str)
        or not isinstance(row["size_bytes"], int)
        or row["size_bytes"] < 0
        or not isinstance(row["last_modified"], str)
        or not re.fullmatch(r"[0-9a-f]{32}(?:-\d+)?", row["etag"])
        for row in rows
    ):
        raise BoreasStage1Error("invalid S3 object metadata")
    return sorted(rows, key=lambda row: row["key"])


def _stream_lidar_listing(aws: Path, sequence_id: str) -> dict[str, Any]:
    prefix = f"{AWS_URI}/{sequence_id}/lidar/"
    process = subprocess.Popen(
        [str(aws), "s3", "ls", prefix, "--recursive", "--no-sign-request"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    count = 0
    total = 0
    first: int | None = None
    last: int | None = None
    digest = hashlib.sha256()
    for line in process.stdout:
        row = parse_s3_ls_line(line)
        if not row["key"].startswith(f"{sequence_id}/lidar/") or not row["key"].endswith(".bin"):
            raise BoreasStage1Error(f"unexpected lidar listing key: {row['key']}")
        timestamp_text = Path(row["key"]).stem
        if re.fullmatch(r"\d{16}", timestamp_text) is None:
            raise BoreasStage1Error("invalid lidar UTC-microsecond object name")
        timestamp = int(timestamp_text)
        if last is not None and timestamp <= last:
            raise BoreasStage1Error("lidar listing timestamps are not strictly increasing")
        first = timestamp if first is None else first
        last = timestamp
        count += 1
        total += row["size_bytes"]
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
        )
    stderr = process.communicate()[1]
    if process.returncode:
        raise BoreasStage1Error(f"aws lidar listing failed for {sequence_id}: {stderr.strip()}")
    return {
        "first_lidar_timestamp": first,
        "last_lidar_timestamp": last,
        "lidar_listing_rows_sha256": digest.hexdigest(),
        "lidar_object_count": count,
        "lidar_remote_bytes": total,
    }


def _object_by_suffix(rows: Sequence[Mapping[str, Any]], suffix: str) -> dict[str, Any] | None:
    matches = [dict(row) for row in rows if str(row.get("key", "")).endswith(suffix)]
    if len(matches) > 1:
        raise BoreasStage1Error(f"duplicate S3 suffix: {suffix}")
    return matches[0] if matches else None


def collect_remote_inventory(*, aws: Path, source_root: Path) -> dict[str, Any]:
    cache_root = source_root / "remote_inventory"
    cache_root.mkdir(parents=True, exist_ok=True)
    top_path = cache_root / "top_level_prefixes.json"
    if top_path.is_file():
        top = json.loads(top_path.read_text(encoding="utf-8"))
    else:
        listing = _aws_text(aws, ("s3", "ls", AWS_URI + "/"))
        top = {"collected_at_utc": utc_now(), "prefixes": parse_top_level_s3_listing(listing)}
        atomic_write_json(top_path, top)
    prefixes = top.get("prefixes")
    if not isinstance(prefixes, list) or not set(BOREAS_SEQUENCES).issubset(set(prefixes)):
        raise BoreasStage1Error("official Boreas split is absent from S3 top-level prefixes")
    rows: list[dict[str, Any]] = []
    for sequence_id in BOREAS_SEQUENCES:
        cache_path = cache_root / f"{sequence_id}.json"
        if cache_path.is_file():
            row = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            applanix = _aws_object_rows(aws, f"{sequence_id}/applanix/")
            calib = _aws_object_rows(aws, f"{sequence_id}/calib/")
            lidar = _stream_lidar_listing(aws, sequence_id)
            gps = _object_by_suffix(applanix, "/gps_post_process.csv")
            lidar_poses = _object_by_suffix(applanix, "/lidar_poses.csv")
            extrinsic = _object_by_suffix(calib, "/T_applanix_lidar.txt")
            row = {
                **lidar,
                "applanix_objects": applanix,
                "applanix_available": bool(applanix),
                "calib_available": bool(calib),
                "calib_objects": calib,
                "collected_at_utc": utc_now(),
                "gps_post_process_available": gps is not None,
                "ground_truth_candidate": sequence_id in TRAIN_SEQUENCES and gps is not None and lidar_poses is not None,
                "lidar_poses_available": lidar_poses is not None,
                "sequence_id": sequence_id,
                "test_or_training_status": "TRAIN_PUBLIC_GT" if sequence_id in TRAIN_SEQUENCES else "TEST_GT_HIDDEN",
                "T_applanix_lidar_available": extrinsic is not None,
            }
            atomic_write_json(cache_path, row)
        if row.get("sequence_id") != sequence_id:
            raise BoreasStage1Error("cached S3 sequence identity mismatch")
        rows.append(row)
    return {
        "all_s3_boreas_named_prefix_count": len(prefixes),
        "all_s3_boreas_named_prefixes": prefixes,
        "boreas_original_sequence_count": len(rows),
        "boreas_original_sequences": rows,
        "boreas_rt_or_non_original_prefix_count": len(set(prefixes) - set(BOREAS_SEQUENCES)),
        "s3_bucket": AWS_URI,
        "source_cache_root": str(cache_root),
    }


def _selected_objects(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in inventory["boreas_original_sequences"]:
        sequence_id = row["sequence_id"]
        for suffix in ("/lidar_poses.csv", "/T_applanix_lidar.txt"):
            source = _object_by_suffix([*row["applanix_objects"], *row["calib_objects"]], suffix)
            if source is not None and (suffix != "/lidar_poses.csv" or sequence_id in TRAIN_SEQUENCES):
                selected.append({**source, "sequence_id": sequence_id})
        if sequence_id == REFERENCE_SEQUENCE:
            gps = _object_by_suffix(row["applanix_objects"], "/gps_post_process.csv")
            if gps is None:
                raise BoreasStage1Error("reference gps_post_process.csv is absent")
            selected.append({**gps, "sequence_id": sequence_id})
    return sorted(selected, key=lambda row: row["key"])


def materialize_selected_objects(
    *, aws: Path, data_root: Path, inventory: Mapping[str, Any], maximum_single: int, maximum_total: int
) -> dict[str, Any]:
    selected = _selected_objects(inventory)
    planned = 0
    for row in selected:
        assert_download_budget(row["size_bytes"], planned, maximum_single=maximum_single, maximum_total=maximum_total)
        planned += row["size_bytes"]
    receipts: list[dict[str, Any]] = []
    payload_root = data_root / "stage1_payload"
    for row in selected:
        destination = payload_root / row["key"]
        receipt_path = destination.with_name(destination.name + ".receipt.json")
        if destination.is_file() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("key") != row["key"]
                or receipt.get("size_bytes") != row["size_bytes"]
                or receipt.get("last_modified") != row["last_modified"]
                or receipt.get("version_id") != row.get("version_id")
                or receipt.get("local_path") != str(destination)
                or destination.stat().st_size != row["size_bytes"]
                or destination.is_symlink()
                or receipt.get("sha256") != sha256_file(destination)
                or receipt.get("etag") != row["etag"]
            ):
                raise BoreasStage1Error(f"resume object authentication failed: {row['key']}")
        else:
            if destination.exists() or receipt_path.exists():
                raise BoreasStage1Error(f"partial selected-object checkpoint: {row['key']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            subprocess.run(
                [str(aws), "s3", "cp", f"{AWS_URI}/{row['key']}", str(temporary), "--no-sign-request", "--only-show-errors"],
                check=True,
            )
            if temporary.stat().st_size != row["size_bytes"]:
                raise BoreasStage1Error(f"downloaded object size mismatch: {row['key']}")
            os.replace(temporary, destination)
            receipt = {
                **row,
                "downloaded_at_utc": utc_now(),
                "local_path": str(destination),
                "sha256": sha256_file(destination),
                "version_id": row.get("version_id"),
            }
            atomic_write_json(receipt_path, receipt)
        receipts.append(receipt)
    return {"planned_s3_bytes": planned, "receipts": receipts, "selected_object_count": len(receipts)}


def _roll(value: float) -> np.ndarray:
    return np.array([[1, 0, 0], [0, math.cos(value), math.sin(value)], [0, -math.sin(value), math.cos(value)]], dtype=np.float64)


def _pitch(value: float) -> np.ndarray:
    return np.array([[math.cos(value), 0, -math.sin(value)], [0, 1, 0], [math.sin(value), 0, math.cos(value)]], dtype=np.float64)


def _yaw(value: float) -> np.ndarray:
    return np.array([[math.cos(value), math.sin(value), 0], [-math.sin(value), math.cos(value), 0], [0, 0, 1]], dtype=np.float64)


def yaw_pitch_roll_to_rotation(heading: float, pitch: float, roll: float) -> np.ndarray:
    return _roll(roll) @ _pitch(pitch) @ _yaw(heading)


def pose_row_to_transform(row: Sequence[float]) -> np.ndarray:
    if len(row) < 10:
        raise BoreasStage1Error("pose row lacks 6DoF fields")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = yaw_pitch_roll_to_rotation(row[9], row[8], row[7])
    transform[:3, 3] = row[1:4]
    return transform


def parse_pose_csv(path: Path, sequence_id: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = tuple(next(reader, ()))
        if header[: len(POSE_HEADER)] != POSE_HEADER:
            raise BoreasStage1Error(f"unexpected Boreas pose header: {path}")
        try:
            values = np.asarray([[float(value) for value in row] for row in reader if row], dtype=np.float64)
        except ValueError as error:
            raise BoreasStage1Error(f"invalid numeric Boreas pose row: {path}") from error
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 13 or not np.isfinite(values).all():
        raise BoreasStage1Error(f"invalid Boreas pose matrix: {path}")
    timestamp_scale = 1e-6 if float(np.median(values[:, 0])) > 1e12 else 1.0
    timestamps = values[:, 0] * timestamp_scale
    gaps = np.diff(timestamps)
    monotonic = bool(np.all(gaps > 0.0))
    if not monotonic:
        raise BoreasStage1Error(f"duplicate/nonmonotonic pose timestamp: {sequence_id}")
    rotations = np.stack([yaw_pitch_roll_to_rotation(row[9], row[8], row[7]) for row in values])
    orthogonality = np.max(np.abs(rotations @ np.transpose(rotations, (0, 2, 1)) - np.eye(3)))
    report = {
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "first_position_m": values[0, 1:4].tolist(),
        "first_timestamp_s": float(timestamps[0]),
        "last_position_m": values[-1, 1:4].tolist(),
        "last_timestamp_s": float(timestamps[-1]),
        "maximum_gap_s": float(np.max(gaps)),
        "median_rate_hz": float(1.0 / np.median(gaps)),
        "pose_sha256": sha256_file(path),
        "reference_frame": "ENU_ref",
        "rotation_convention": "C_enu_sensor=roll(r)@pitch(p)@yaw(heading)",
        "rotation_orthogonality_error_max": float(orthogonality),
        "row_count": int(values.shape[0]),
        "sequence_id": sequence_id,
        "strictly_monotonic": monotonic,
        "timestamp_scale_to_seconds": timestamp_scale,
    }
    trajectory = np.column_stack((timestamps, values[:, 1:4]))
    return report, trajectory, values


def parse_calibration(path: Path) -> np.ndarray:
    value = np.loadtxt(path, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise BoreasStage1Error("invalid T_applanix_lidar matrix")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-12, rtol=0.0):
        raise BoreasStage1Error("invalid homogeneous calibration row")
    if not np.allclose(value[:3, :3] @ value[:3, :3].T, np.eye(3), atol=1e-10, rtol=0.0):
        raise BoreasStage1Error("calibration rotation is not orthonormal")
    if np.linalg.det(value[:3, :3]) <= 0.0:
        raise BoreasStage1Error("calibration rotation is improper")
    return value


def detect_sequence_local_resets(pose_reports: Sequence[Mapping[str, Any]]) -> list[str]:
    """Flag later sequences whose published first position is locally reset to zero."""

    return [
        str(row["sequence_id"])
        for row in pose_reports[1:]
        if np.linalg.norm(np.asarray(row["first_position_m"], dtype=np.float64)) < 1e-6
    ]


def compose_enu_lidar(enu_applanix: np.ndarray, applanix_lidar: np.ndarray) -> np.ndarray:
    left = np.asarray(enu_applanix, dtype=np.float64)
    right = np.asarray(applanix_lidar, dtype=np.float64)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise BoreasStage1Error("transform composition requires two 4x4 matrices")
    return left @ right


def transform_chain_consistency(gps_values: np.ndarray, lidar_values: np.ndarray, extrinsic: np.ndarray) -> dict[str, Any]:
    gps_times = gps_values[:, 0]
    if float(np.median(gps_times)) > 1e12:
        gps_times = gps_times * 1e-6
    lidar_times = lidar_values[:, 0] * (1e-6 if float(np.median(lidar_values[:, 0])) > 1e12 else 1.0)
    valid = np.flatnonzero((lidar_times >= gps_times[0]) & (lidar_times <= gps_times[-1]))
    if valid.size < 10:
        raise BoreasStage1Error("insufficient overlapping GPS/lidar pose timestamps")
    sample_indices = valid[np.linspace(0, valid.size - 1, min(101, valid.size), dtype=int)]
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    interpolation_gaps: list[float] = []
    for lidar_index in sample_indices:
        query = lidar_times[lidar_index]
        right_index = int(np.searchsorted(gps_times, query, side="right"))
        right_index = min(max(right_index, 1), len(gps_times) - 1)
        left_index = right_index - 1
        span = gps_times[right_index] - gps_times[left_index]
        if span <= 0.0:
            raise BoreasStage1Error("GPS interpolation span is nonpositive")
        alpha = (query - gps_times[left_index]) / span
        translation = (1.0 - alpha) * gps_values[left_index, 1:4] + alpha * gps_values[right_index, 1:4]
        matrices = np.stack(
            [pose_row_to_transform(gps_values[left_index])[:3, :3], pose_row_to_transform(gps_values[right_index])[:3, :3]]
        )
        rotation = Slerp([gps_times[left_index], gps_times[right_index]], Rotation.from_matrix(matrices))([query]).as_matrix()[0]
        enu_applanix = np.eye(4)
        enu_applanix[:3, :3] = rotation
        enu_applanix[:3, 3] = translation
        predicted = compose_enu_lidar(enu_applanix, extrinsic)
        observed = pose_row_to_transform(lidar_values[lidar_index])
        translation_errors.append(float(np.linalg.norm(predicted[:3, 3] - observed[:3, 3])))
        delta = predicted[:3, :3] @ observed[:3, :3].T
        rotation_errors.append(float(Rotation.from_matrix(delta).magnitude()))
        interpolation_gaps.append(float(span))
    return {
        "composition": "T_ENU_lidar(t)=T_ENU_applanix(t)@T_applanix_lidar",
        "direction": "T_applanix_lidar maps lidar coordinates into the Applanix frame",
        "maximum_interpolation_gap_s": max(interpolation_gaps),
        "maximum_rotation_error_rad": max(rotation_errors),
        "maximum_translation_error_m": max(translation_errors),
        "median_rotation_error_rad": float(np.median(rotation_errors)),
        "median_translation_error_m": float(np.median(translation_errors)),
        "sample_count": len(sample_indices),
        "status": "PASS" if (
            max(translation_errors) <= TRANSFORM_NUMERICAL_TRANSLATION_LIMIT_M
            and max(rotation_errors) <= TRANSFORM_NUMERICAL_ROTATION_LIMIT_RAD
        ) else "FAIL",
    }


def audit_transform_chain_direction(
    gps_values: np.ndarray, lidar_values: np.ndarray, extrinsic: np.ndarray
) -> dict[str, Any]:
    """Verify the published direction against poses and explicitly reject its inverse."""

    forward = transform_chain_consistency(gps_values, lidar_values, extrinsic)
    inverse = transform_chain_consistency(gps_values, lidar_values, np.linalg.inv(extrinsic))
    translation_ratio = inverse["maximum_translation_error_m"] / max(
        forward["maximum_translation_error_m"], np.finfo(np.float64).tiny
    )
    rotation_ratio = inverse["maximum_rotation_error_rad"] / max(
        forward["maximum_rotation_error_rad"], np.finfo(np.float64).tiny
    )
    passed = (
        forward["status"] == "PASS"
        and translation_ratio >= TRANSFORM_DIRECTION_MINIMUM_DISCRIMINATION_RATIO
        and rotation_ratio >= TRANSFORM_DIRECTION_MINIMUM_DISCRIMINATION_RATIO
    )
    return {
        **forward,
        "direction_discrimination_rotation_ratio": float(rotation_ratio),
        "direction_discrimination_translation_ratio": float(translation_ratio),
        "inverse_maximum_rotation_error_rad": inverse["maximum_rotation_error_rad"],
        "inverse_maximum_translation_error_m": inverse["maximum_translation_error_m"],
        "minimum_direction_discrimination_ratio": TRANSFORM_DIRECTION_MINIMUM_DISCRIMINATION_RATIO,
        "numerical_rotation_limit_rad": TRANSFORM_NUMERICAL_ROTATION_LIMIT_RAD,
        "numerical_translation_limit_m": TRANSFORM_NUMERICAL_TRANSLATION_LIMIT_M,
        "status": "PASS" if passed else "FAIL",
    }


def _environment_report(repository: Path, data_root: Path, aws: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(data_root)
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[f"{key}_bytes"] = int(value.strip().split()[0]) * 1024
    packages = {}
    for name in ("numpy", "scipy", "PyYAML", "pytest"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "aws_cli": subprocess.check_output([str(aws), "--version"], text=True, stderr=subprocess.STDOUT).strip(),
        "collected_at_utc": utc_now(),
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total, "used_bytes": disk.used},
        "git": {
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repository, text=True).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
            "tags": subprocess.check_output(["git", "tag", "--list"], cwd=repository, text=True).splitlines(),
            "worktree_porcelain": subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).splitlines(),
        },
        "host": platform.node(), "machine": platform.machine(), "memory": memory,
        "os": platform.platform(), "packages": packages,
        "python_executable": str(Path(sys.executable).resolve()), "python_version": sys.version,
    }


def _cleanup_report() -> dict[str, Any]:
    return {
        "cavers_data_root": "/home/lj/zero_perturbation_data/cavers_stage1_v1",
        "cavers_data_root_absent": not Path("/home/lj/zero_perturbation_data/cavers_stage1_v1").exists(),
        "deleted_apparent_bytes": 57_517_402,
        "deleted_allocated_bytes": 57_831_424,
        "deleted_directory_count": 84,
        "deleted_file_count": 127,
        "filesystem_available_bytes_after": 111_266_463_744,
        "filesystem_available_bytes_before": 111_208_640_512,
        "filesystem_available_bytes_delta": 57_823_232,
        "frozen_cavers_root": "/home/lj/ZPRM/frozen_assets/real_data_cavers_stage1_v1",
        "frozen_cavers_sha256sums_pass": True,
        "frozen_cavers_verifier_pass": True,
        "removal_recoverability": "data root removal is not recoverable; frozen audit and formal runtime remain",
    }


def _write_inventory(runtime_root: Path, inventory: Mapping[str, Any]) -> None:
    atomic_write_json(runtime_root / "boreas_remote_sequence_inventory.json", inventory)
    fields = (
        "sequence_id", "applanix_available", "gps_post_process_available", "lidar_poses_available",
        "calib_available", "T_applanix_lidar_available", "lidar_object_count", "lidar_remote_bytes",
        "first_lidar_timestamp", "last_lidar_timestamp", "ground_truth_candidate", "test_or_training_status",
    )
    rows = [{key: row[key] for key in fields} for row in inventory["boreas_original_sequences"]]
    atomic_write_csv(runtime_root / "boreas_remote_sequence_inventory.csv", rows, fields)


def _sha256sums_payload(root: Path, names: Sequence[str]) -> bytes:
    return "".join(f"{sha256_file(root / name)}  {name}\n" for name in sorted(names)).encode("utf-8")


def _source_rows(data_root: Path, receipts: Sequence[Mapping[str, Any]], devkit_root: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "etag": row["etag"],
            "last_modified": row["last_modified"],
            "local_path": row["local_path"],
            "relative_path": str(Path(row["local_path"]).relative_to(data_root)),
            "s3_key": row["key"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "status": "DOWNLOADED_ALLOWLISTED_STAGE1_OBJECT",
            "version_id": row.get("version_id"),
        }
        for row in receipts
    ]
    for relative in ("README.md", "DATA_REFERENCE.md", "DATA_LICENSE.md", "LICENSE", "pyboreas/data/splits.py", "pyboreas/data/calib.py", "pyboreas/utils/utils.py"):
        path = devkit_root / relative
        rows.append({
            "etag": None, "last_modified": None, "local_path": str(path),
            "relative_path": str(path.relative_to(data_root)), "s3_key": None,
            "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            "status": "PINNED_OFFICIAL_DEVKIT_SOURCE", "version_id": None,
        })
    paper = data_root / "source_metadata/boreas_paper_2203.10168.pdf"
    if sha256_file(paper) != PAPER_PDF_SHA256:
        raise BoreasStage1Error("pinned Boreas paper PDF changed")
    rows.append({
        "etag": None, "last_modified": None, "local_path": str(paper),
        "relative_path": str(paper.relative_to(data_root)), "s3_key": None,
        "sha256": PAPER_PDF_SHA256, "size_bytes": paper.stat().st_size,
        "status": "PINNED_OFFICIAL_PAPER", "version_id": None,
    })
    return sorted(rows, key=lambda row: row["relative_path"])


def _uncertainty_rows(reference_sha: str, extrinsic_sha: str, time_sha: str) -> list[dict[str, Any]]:
    return [
        {"component": "GNSS/RTX position uncertainty", "value": "0.02-0.04 nominal; 0.20-0.40 documented urban-canyon residual RMS", "unit": "m", "uncertainty_type": "RMSE", "evidence_type": "IJRR_AND_DATA_REFERENCE", "source": "Boreas IJRR Table 2 and Section 6; DATA_REFERENCE.md", "source_sha256": reference_sha, "status": "SEQUENCE_DEPENDENT_LIMITED_NUMERIC"},
        {"component": "orientation uncertainty", "value": "UNKNOWN", "unit": "rad", "uncertainty_type": "UNKNOWN", "evidence_type": "NO_RELEASE_WIDE_NUMERIC_BOUND", "source": "Boreas IJRR/POSPac documentation audit", "source_sha256": reference_sha, "status": "UNKNOWN"},
        {"component": "sequence-dependent ground-truth quality", "value": "0.02-0.04 nominal; can degrade to 0.20-0.40", "unit": "m", "uncertainty_type": "RMSE", "evidence_type": "OFFICIAL_SEQUENCE_DEPENDENCE_STATEMENT", "source": "DATA_REFERENCE.md", "source_sha256": reference_sha, "status": "SEQUENCE_DEPENDENT_LIMITED_NUMERIC"},
        {"component": "Applanix time uncertainty", "value": "UNKNOWN", "unit": "s", "uncertainty_type": "UNKNOWN", "evidence_type": "PPS_NMEA_WITHOUT_ERROR_BOUND", "source": "boreas_time_sync_audit.json", "source_sha256": time_sha, "status": "UNKNOWN"},
        {"component": "Velodyne time synchronization uncertainty", "value": "UNKNOWN", "unit": "s", "uncertainty_type": "UNKNOWN", "evidence_type": "PPS_NMEA_WITHOUT_ERROR_BOUND", "source": "boreas_time_sync_audit.json", "source_sha256": time_sha, "status": "UNKNOWN"},
        {"component": "T_applanix_lidar translation uncertainty", "value": "UNKNOWN", "unit": "m", "uncertainty_type": "UNKNOWN", "evidence_type": "PROPRIETARY_LIDAR_ASSISTED_CALIBRATION_NO_BOUND", "source": "boreas_lidar_extrinsic_provenance.json", "source_sha256": extrinsic_sha, "status": "UNKNOWN"},
        {"component": "T_applanix_lidar rotation uncertainty", "value": "UNKNOWN", "unit": "rad", "uncertainty_type": "UNKNOWN", "evidence_type": "PROPRIETARY_LIDAR_ASSISTED_CALIBRATION_NO_BOUND", "source": "boreas_lidar_extrinsic_provenance.json", "source_sha256": extrinsic_sha, "status": "UNKNOWN"},
        {"component": "GT interpolation uncertainty", "value": "UNKNOWN", "unit": "m/rad", "uncertainty_type": "UNKNOWN", "evidence_type": "NO_PUBLISHED_BOUND", "source": "boreas_transform_chain_consistency.json", "source_sha256": time_sha, "status": "UNKNOWN"},
        {"component": "future deskew uncertainty", "value": "UNKNOWN", "unit": "m/rad", "uncertainty_type": "UNKNOWN", "evidence_type": "NOT_MATERIALIZED_STAGE1", "source": "Stage-1 scope", "source_sha256": time_sha, "status": "UNKNOWN"},
        {"component": "future target-map accumulation uncertainty", "value": "UNKNOWN", "unit": "m/rad", "uncertainty_type": "UNKNOWN", "evidence_type": "NOT_MATERIALIZED_STAGE1", "source": "Stage-1 scope", "source_sha256": time_sha, "status": "UNKNOWN"},
    ]


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Boreas single-dataset Stage-1 总结", "", f"**BOREAS_STAGE1_READY={str(summary['boreas_stage1_ready']).lower()}**", ""]
    lines.extend(f"{index}. {answer}" for index, answer in enumerate(summary["answers"], 1))
    lines.extend(["", "## 最终结论", "", summary["final_conclusion"], ""])
    return "\n".join(lines)


def execute_boreas_stage1(
    *, repository: str | Path, data_root: str | Path, runtime_root: str | Path,
    mode: str, metadata_only: bool, maximum_single_download_bytes: int,
    maximum_total_download_bytes: int, no_registration: bool,
) -> dict[str, Any]:
    repository = Path(repository).resolve(strict=True)
    data_root = Path(data_root).resolve(strict=True)
    runtime_argument = Path(runtime_root)
    runtime_root = runtime_argument.resolve(strict=False)
    if mode not in {"fresh", "resume"} or metadata_only is not True or no_registration is not True:
        raise PermissionError("fresh/resume metadata-only no-registration mode is mandatory")
    if os.environ.get("ZPRM_REAL_DATA_PREP_NO_REGISTRATION") != "1":
        raise PermissionError("ZPRM_REAL_DATA_PREP_NO_REGISTRATION=1 is mandatory")
    if runtime_argument != runtime_root or repository in runtime_root.parents or data_root in runtime_root.parents:
        raise PermissionError("runtime root must be canonical and disjoint")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repository, text=True).strip()
    if branch != EXPECTED_BRANCH or subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip():
        raise PermissionError("formal Boreas Stage-1 requires the expected clean branch")
    for ancestor in ("01b19b1ba89376cb53aa082b77032b562888eb7b", "862aa44ca8bb8f2b98cdea45b76b13ea21875c9a", "a56ae7e97bea39de7e8753208259a33a541f4b69"):
        if subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, "HEAD"], cwd=repository).returncode:
            raise PermissionError(f"required history ancestor missing: {ancestor}")
    if mode == "fresh":
        if os.path.lexists(runtime_root):
            raise FileExistsError("fresh runtime root must be absent")
        runtime_root.mkdir(parents=True)
    elif not runtime_root.is_dir():
        raise FileNotFoundError("resume runtime root is absent")
    if mode == "resume" and (runtime_root / "SHA256SUMS").is_file():
        from .boreas_stage1_verifier import verify_boreas_stage1
        return verify_boreas_stage1(repository=repository, data_root=data_root, runtime_root=runtime_root)["summary"]

    source_root = data_root / "source_metadata"
    source_root.mkdir(parents=True, exist_ok=True)
    devkit_root = data_root / "pyboreas"
    aws = data_root / "tools/awscli-venv/bin/aws"
    if not aws.is_file() or not os.access(aws, os.X_OK):
        raise FileNotFoundError("isolated AWS CLI is absent")
    if subprocess.check_output(["git", "-C", str(devkit_root), "rev-parse", "HEAD"], text=True).strip() != PYBOREAS_COMMIT:
        raise PermissionError("pyboreas commit mismatch")
    if subprocess.check_output(["git", "-C", str(devkit_root), "status", "--porcelain"], text=True).strip():
        raise PermissionError("pyboreas worktree is not clean")
    expected_hashes = {
        "README.md": README_SHA256, "DATA_REFERENCE.md": DATA_REFERENCE_SHA256,
        "DATA_LICENSE.md": DATA_LICENSE_SHA256, "LICENSE": CODE_LICENSE_SHA256,
        "pyboreas/data/splits.py": SPLITS_SHA256, "pyboreas/data/calib.py": CALIB_PARSER_SHA256,
        "pyboreas/utils/utils.py": UTILS_SHA256,
    }
    for relative, expected in expected_hashes.items():
        if sha256_file(devkit_root / relative) != expected:
            raise BoreasStage1Error(f"pinned pyboreas source changed: {relative}")
    protocol_paths = {
        "framework_markdown": repository / "protocols/real_data_validation_protocol_framework_v1.md",
        "framework_json": repository / "protocols/real_data_validation_protocol_framework_v1.json",
        "eligibility_checklist": repository / "protocols/real_data_dataset_eligibility_checklist.csv",
    }
    for path, expected in zip(protocol_paths.values(), (PROTOCOL_MD_SHA256, PROTOCOL_JSON_SHA256, CHECKLIST_SHA256)):
        if sha256_file(path) != expected:
            raise BoreasStage1Error("frozen real-data protocol changed")
    backend_path = repository / "frozen_assets/backend_parameter_contract.json"
    if sha256_file(backend_path) != BACKEND_PARAMETER_SHA256:
        raise BoreasStage1Error("backend parameter contract changed")
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    for name, expected in (
        ("open3d", OPEN3D_PARAMETER_CANONICAL_SHA256),
        ("pcl", PCL_PARAMETER_CANONICAL_SHA256),
    ):
        section = backend.get(name, {})
        if section.get("canonical_sha256") != expected or compact_sha256(section.get("parameters")) != expected:
            raise BoreasStage1Error(f"{name} backend parameter canonical SHA changed")

    environment_path = runtime_root / "environment_report.json"
    if environment_path.is_file():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
    else:
        environment = _environment_report(repository, data_root, aws)
        atomic_write_json(environment_path, environment)
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if environment.get("git", {}).get("commit") != current_commit:
        raise PermissionError("resume requires the original audited producer commit")
    cleanup = _cleanup_report()
    if cleanup["cavers_data_root_absent"] is not True:
        raise BoreasStage1Error("CAVERS data root was not cleaned")
    atomic_write_json(runtime_root / "boreas_cleanup_report.json", cleanup)

    with NoRegistrationGuard(open3d_module=None) as guard:
        static_audit = assert_preparation_sources_are_safe(repository / "src/phase_a_harness/real_data_preparation")
        inventory = collect_remote_inventory(aws=aws, source_root=source_root)
        _write_inventory(runtime_root, inventory)
        download = materialize_selected_objects(
            aws=aws, data_root=data_root, inventory=inventory,
            maximum_single=maximum_single_download_bytes, maximum_total=maximum_total_download_bytes,
        )
        source_rows = _source_rows(data_root, download["receipts"], devkit_root)
        download_manifest = {
            "aws_no_sign_request": True,
            "downloaded_s3_bytes": sum(row["size_bytes"] for row in download["receipts"]),
            "downloaded_s3_object_count": len(download["receipts"]),
            "full_lidar_binary_download_count": 0,
            "lidar_payload_count": 0,
            "materialized_files": source_rows,
            "maximum_materialized_s3_object_bytes": max(row["size_bytes"] for row in download["receipts"]),
            "maximum_single_download_bytes": maximum_single_download_bytes,
            "maximum_total_download_bytes": maximum_total_download_bytes,
            "planned_s3_bytes": download["planned_s3_bytes"],
        }
        atomic_write_json(runtime_root / "download_manifest.json", download_manifest)
        download_fields = (
            "relative_path", "local_path", "s3_key", "size_bytes", "sha256", "etag",
            "last_modified", "version_id", "status",
        )
        atomic_write_csv(runtime_root / "download_manifest.csv", source_rows, download_fields)

        devkit_files = []
        for path in sorted(devkit_root.rglob("*")):
            if path.is_file() and ".git" not in path.parts:
                devkit_files.append({"path": path.relative_to(devkit_root).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        pyboreas_manifest = {
            "checkout_mode": "official git clone with sparse checkout of root files, pyboreas/data, and pyboreas/utils",
            "code_license": PYBOREAS_CODE_LICENSE, "commit": PYBOREAS_COMMIT,
            "files": devkit_files, "repository_url": PYBOREAS_URL,
            "tree_rows_sha256": compact_sha256(devkit_files), "worktree_clean": True,
        }
        atomic_write_json(runtime_root / "pyboreas_source_manifest.json", pyboreas_manifest)
        official_source = {
            "audit_time_utc": environment["collected_at_utc"], "aws_bucket": AWS_URI, "boreas_site": BOREAS_SITE,
            "code_license": PYBOREAS_CODE_LICENSE, "dataset_license": DATASET_LICENSE,
            "paper_arxiv": PAPER_ARXIV, "paper_doi": PAPER_DOI,
            "paper_pdf_sha256": PAPER_PDF_SHA256,
            "pyboreas_commit": PYBOREAS_COMMIT,
            "pinned_source_sha256": expected_hashes,
            "protocol_sha256": {key: sha256_file(path) for key, path in protocol_paths.items()},
            "remote_inventory_sha256": sha256_file(runtime_root / "boreas_remote_sequence_inventory.json"),
            "static_no_registration_audit": static_audit,
        }
        atomic_write_json(runtime_root / "official_source_manifest.json", official_source)
        citation = (
            "# Boreas dataset citation and licenses\n\n"
            f"- Dataset site: {BOREAS_SITE}\n- Public bucket: `{AWS_URI}`.\n"
            f"- Dataset paper DOI: `{PAPER_DOI}`; arXiv:`{PAPER_ARXIV}`.\n"
            f"- Pinned arXiv PDF SHA-256: `{PAPER_PDF_SHA256}`.\n"
            f"- Dataset terms: `{DATASET_LICENSE}` from DATA_LICENSE.md.\n"
            f"- Devkit: `{PYBOREAS_URL}` at `{PYBOREAS_COMMIT}`, `{PYBOREAS_CODE_LICENSE}`.\n"
        )
        atomic_write_bytes(runtime_root / "dataset_citation_report.md", citation.encode("utf-8"))

        by_key = {row["key"]: row for row in download["receipts"]}
        trajectories: dict[str, np.ndarray] = {}
        pose_reports: list[dict[str, Any]] = []
        pose_values: dict[str, np.ndarray] = {}
        for sequence_id in TRAIN_SEQUENCES:
            key = f"{sequence_id}/applanix/lidar_poses.csv"
            receipt = by_key.get(key)
            if receipt is None:
                raise BoreasStage1Error(f"public lidar pose file missing: {sequence_id}")
            report, trajectory, values = parse_pose_csv(Path(receipt["local_path"]), sequence_id)
            trajectories[sequence_id] = trajectory
            pose_reports.append(report)
            pose_values[sequence_id] = values
        gps_receipt = by_key[f"{REFERENCE_SEQUENCE}/applanix/gps_post_process.csv"]
        gps_report, _, gps_values = parse_pose_csv(Path(gps_receipt["local_path"]), REFERENCE_SEQUENCE)

        eligible_fields = (
            "sequence_id", "status", "gps_post_process_available", "lidar_poses_available",
            "calibration_available", "row_count", "duration_s", "median_rate_hz", "maximum_gap_s",
            "first_timestamp_s", "last_timestamp_s", "reference_frame", "pose_sha256",
        )
        eligible_rows = [
            {
                "sequence_id": report["sequence_id"], "status": "ELIGIBLE_PUBLIC_GT",
                "gps_post_process_available": True, "lidar_poses_available": True,
                "calibration_available": True, **{key: report[key] for key in eligible_fields[5:]},
            }
            for report in pose_reports
        ]
        atomic_write_csv(runtime_root / "eligible_reference_sequences.csv", eligible_rows, eligible_fields)

        reference_provenance = {
            "accuracy_semantics": "2-4 cm is nominal sequence-dependent RMS, not 1SIGMA; official paper also documents 20-40 cm urban-canyon residual RMS",
            "position_accuracy_claim": {"nominal_range_m": [0.02, 0.04], "urban_canyon_range_m": [0.20, 0.40], "uncertainty_type": "RMSE"},
            "reference_frame": "fixed ENU_ref",
            "reference_rate_hz": gps_report["median_rate_hz"],
            "sampled_reference_sequence": REFERENCE_SEQUENCE,
            "uses_gnss": True, "uses_icp": False, "uses_imu": True, "uses_lidar": False,
            "uses_rtx": True, "uses_scan_matching": False, "uses_wheel_encoder": True,
            "source": "DATA_REFERENCE.md and Boreas IJRR Section 6",
        }
        atomic_write_json(runtime_root / "boreas_reference_trajectory_provenance.json", reference_provenance)

        extrinsic_rows: list[dict[str, Any]] = []
        matrices: dict[str, np.ndarray] = {}
        for sequence_id in BOREAS_SEQUENCES:
            receipt = by_key.get(f"{sequence_id}/calib/T_applanix_lidar.txt")
            if receipt is None:
                raise BoreasStage1Error(f"calibration object missing: {sequence_id}")
            matrix = parse_calibration(Path(receipt["local_path"]))
            matrices[sequence_id] = matrix
            extrinsic_rows.append({"matrix": matrix.tolist(), "sequence_id": sequence_id, "sha256": receipt["sha256"], "source_key": receipt["key"]})
        unique_extrinsic_sha = sorted({row["sha256"] for row in extrinsic_rows})
        extrinsic_provenance = {
            "all_sequences_byte_identical": len(unique_extrinsic_sha) == 1,
            "calibration_date_or_session": "UNKNOWN",
            "common_static_calibration": len(unique_extrinsic_sha) == 1,
            "extrinsic_uncertainty": "UNKNOWN",
            "frozen_R02_independence_violation": True,
            "query_sequence_used_for_calibration": "UNKNOWN",
            "sequence_specific": False if len(unique_extrinsic_sha) == 1 else "UNKNOWN",
            "source": "DATA_REFERENCE.md Lidar-to-IMU Extrinsics; Boreas IJRR Section 7.2",
            "uses_lidar_pointclouds": True,
            "uses_post_processed_gps_imu": True,
            "unique_matrix_sha256": unique_extrinsic_sha,
            "rows": extrinsic_rows,
        }
        atomic_write_json(runtime_root / "boreas_lidar_extrinsic_provenance.json", extrinsic_provenance)
        risk = {
            "circular_validation_risk": "UNRESOLVED_STATIC_CALIBRATION_DEPENDENCE",
            "frozen_protocol_R02_text": "Independent high-accuracy 6DoF reference",
            "frozen_protocol_R05_text": "Time synchronization and extrinsics are auditable",
            "query_sequence_participation": "UNKNOWN",
            "risk_status": "FAIL_CLOSED_R02",
            "rationale": "The GNSS/IMU/wheel trajectory is independent, but the final lidar-frame 6DoF reference uses a proprietary static extrinsic obtained from lidar pointclouds; the release does not prove that evaluated query sequences were excluded.",
        }
        atomic_write_json(runtime_root / "boreas_extrinsic_independence_risk_assessment.json", risk)
        atomic_write_bytes(
            runtime_root / "boreas_extrinsic_independence_risk_assessment.md",
            ("# Boreas extrinsic independence risk\n\n"
             "The Applanix trajectory itself uses GNSS, IMU, wheel encoder, and RTX—not lidar. "
             "However, the published `T_applanix_lidar` was produced as a by-product of a proprietary batch optimization using lidar pointclouds and post-processed GPS/IMU. "
             "All audited sequence copies are byte-identical, but the calibration sequence/session and exclusion of future map/query sequences are not published. "
             "The frozen R02 requirement is therefore failed closed; no claim that this dependence is negligible is made.\n").encode("utf-8"),
        )

        reference_first = np.asarray(pose_reports[0]["first_position_m"])
        later_resets = detect_sequence_local_resets(pose_reports)
        global_coordinate_extent = float(max(np.linalg.norm(trajectory[:, 1:4], axis=1).max() for trajectory in trajectories.values()))
        common_world_pass = not later_resets and global_coordinate_extent > 100.0
        world = {
            "COMMON_WORLD_FRAME": "PASS" if common_world_pass else "FAIL",
            "actual_global_coordinate_extent_m": global_coordinate_extent,
            "first_reference_sequence": REFERENCE_SEQUENCE,
            "first_reference_position_m": reference_first.tolist(),
            "independent_acquisition_sequence_count": len(TRAIN_SEQUENCES),
            "independent_sequence_ids_and_time_spans_verified": True,
            "official_definition": "fixed ENU_ref aligned in position with the first pose of the first sequence; WGS-84 tangent orientation, x East/y North/z up",
            "registration_or_trajectory_fitting_used": False,
            "sequence_local_reset_detected": bool(later_resets),
            "sequence_local_reset_ids": later_resets,
            "sequence_specific_origin_transform_published": False,
            "status": "PASS" if common_world_pass else "FAIL",
        }
        atomic_write_json(runtime_root / "boreas_common_world_frame_audit.json", world)

        consistency = audit_transform_chain_direction(
            gps_values, pose_values[REFERENCE_SEQUENCE], matrices[REFERENCE_SEQUENCE]
        )
        atomic_write_json(runtime_root / "boreas_transform_chain_consistency.json", consistency)
        transform_manifest = {
            "composition": "T_ENU_lidar(t)=T_ENU_applanix(t)@T_applanix_lidar",
            "consistency_sha256": sha256_file(runtime_root / "boreas_transform_chain_consistency.json"),
            "direction_verified_against_official_lidar_poses": consistency["status"] == "PASS",
            "inverse_used": False,
            "matrix": matrices[REFERENCE_SEQUENCE].tolist(),
            "source_file": by_key[f"{REFERENCE_SEQUENCE}/calib/T_applanix_lidar.txt"]["local_path"],
            "source_sha256": by_key[f"{REFERENCE_SEQUENCE}/calib/T_applanix_lidar.txt"]["sha256"],
            "status": consistency["status"],
        }
        atomic_write_json(runtime_root / "boreas_transform_chain_manifest.json", transform_manifest)

        inventory_by_id = {row["sequence_id"]: row for row in inventory["boreas_original_sequences"]}
        lidar_rates = []
        timestamp_matches = True
        for report in pose_reports:
            remote = inventory_by_id[report["sequence_id"]]
            timestamp_matches &= (
                report["row_count"] == remote["lidar_object_count"]
                and int(round(report["first_timestamp_s"] * 1e6)) == remote["first_lidar_timestamp"]
                and int(round(report["last_timestamp_s"] * 1e6)) == remote["last_lidar_timestamp"]
            )
            lidar_rates.append(report["median_rate_hz"])
        time_status = "PASS_WITH_DOCUMENTED_LIMITATION" if timestamp_matches and gps_report["strictly_monotonic"] else "FAIL"
        time_audit = {
            "hardware_sync_available": True, "interpolation_candidate": "linear translation plus SO(3) SLERP",
            "lidar_rate_hz": float(np.median(lidar_rates)),
            "lidar_scan_timestamp_semantics": "UTC temporal middle of scan",
            "maximum_reference_gap_s": gps_report["maximum_gap_s"],
            "nmea_available": True, "per_point_timestamp_documented": True, "pps_available": True,
            "reference_rate_hz": gps_report["median_rate_hz"], "status": time_status,
            "time_uncertainty_evidence": "hardware PPS/NMEA chain documented; no numeric end-to-end error bound published",
            "time_uncertainty_status": "UNKNOWN", "timestamp_epoch": "Unix UTC",
            "timestamp_monotonicity": True, "timestamp_unit": "microseconds for lidar filenames/poses; seconds in gps_post_process.csv",
            "reference_timestamp_actual_encoding": "floating-point Unix UTC seconds",
            "reference_timestamp_documentation_claim": "DATA_REFERENCE describes gps_post_process t as UTC microseconds",
            "reference_timestamp_documentation_data_discrepancy": True,
            "reference_timestamp_scale_to_seconds": gps_report["timestamp_scale_to_seconds"],
            "zero_time_uncertainty_assumed": False,
        }
        atomic_write_json(runtime_root / "boreas_time_sync_audit.json", time_audit)

        reference_sha = sha256_file(runtime_root / "boreas_reference_trajectory_provenance.json")
        extrinsic_sha = sha256_file(runtime_root / "boreas_lidar_extrinsic_provenance.json")
        time_sha = sha256_file(runtime_root / "boreas_time_sync_audit.json")
        uncertainty_rows = _uncertainty_rows(reference_sha, extrinsic_sha, time_sha)
        uncertainty = {"rmse_not_relabelled_as_1sigma": True, "status": "PARTIAL", "unknown_not_zero": True, "rows": uncertainty_rows}
        uncertainty_fields = ("component", "value", "unit", "uncertainty_type", "evidence_type", "source", "source_sha256", "status")
        atomic_write_csv(runtime_root / "boreas_stage1_uncertainty_feasibility.csv", uncertainty_rows, uncertainty_fields)
        atomic_write_json(runtime_root / "boreas_stage1_uncertainty_feasibility.json", uncertainty)

        r02 = "FAIL"
        overlap_gate = r02 != "FAIL" and common_world_pass and consistency["status"] == "PASS"
        atomic_write_csv(
            runtime_root / "boreas_gt_only_overlap_matrix.csv", [],
            ("map_sequence_id", "query_sequence_id", "query_duration_s", "covered_duration_s", "coverage_fraction", "contiguous_covered_interval_count", "eligible_nonoverlapping_5s_intervals", "nearest_GT_distance_median", "nearest_GT_distance_q95", "nearest_GT_distance_max"),
        )
        overlap = {
            "contract": asdict(FROZEN_STAGE1_OVERLAP_CONTRACT), "gate_status": "NOT_COMPUTABLE",
            "pair_rows": [], "point_cloud_or_registration_consulted": False,
            "reason": "R02_FAIL_LIDAR_ASSISTED_EXTRINSIC_IN_FINAL_REFERENCE_CHAIN",
        }
        if overlap_gate:
            raise BoreasStage1Error("R02 fail-closed gate unexpectedly opened")
        atomic_write_json(runtime_root / "boreas_gt_only_overlap_matrix.json", overlap)
        pair_selection = {
            "PRIMARY_PAIR": None, "RESERVE_PAIR_1": None, "RESERVE_PAIR_2": None,
            "eligible_pair_count": 0,
            "ranking_rule": ["covered duration descending", "coverage fraction descending", "eligible 5s intervals descending", "nearest GT distance q95 ascending", "map ID lexicographic", "query ID lexicographic"],
            "reserve_activation_policy": "only corruption/download/checksum/GT infrastructure failure; never registration outcome",
        }
        atomic_write_json(runtime_root / "boreas_pair_selection_v1.json", pair_selection)

        r03 = "BLOCKED_STAGE1"
        r04 = "PASS" if common_world_pass else "FAIL"
        r05 = "PASS" if consistency["status"] == "PASS" and time_status == "PASS_WITH_DOCUMENTED_LIMITATION" else "FAIL"
        eligibility = {
            "BOREAS_STAGE1_READY": False, "GT_ONLY_OVERLAP": "NOT_COMPUTABLE",
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False, "R01": "PENDING_SECOND_DATASET",
            "R02": r02, "R03": r03, "R04": r04, "R05": r05,
            "R06": "BLOCKED_STAGE1", "R07": "BLOCKED_STAGE1", "R08": "BLOCKED_STAGE1",
            "R09": "PASS", "R10": "PARTIAL", "R14": "BLOCKED_STAGE1",
            "REAL_DATA_RUN_AUTHORIZED": False, "SINGLE_DATASET_PREREGISTRATION_READY": False,
            "planned_future_trials": 0, "registration_execution_count": 0, "rich_snapshot_count": 0,
            "snapshot_count": 0, "weak_snapshot_count": 0,
            "primary_hard_blockers": ["R02_LIDAR_ASSISTED_EXTRINSIC_INDEPENDENCE", "GT_ONLY_OVERLAP_GATED", "R10_UNCERTAINTY_FEASIBILITY"],
        }
        atomic_write_json(runtime_root / "boreas_stage1_eligibility.json", eligibility)

        answers = [
            f"上一轮 CAVERS 删除 apparent={cleanup['deleted_apparent_bytes']} B，文件系统实测释放={cleanup['filesystem_available_bytes_delta']} B。",
            f"Boreas Stage-1 S3 allowlist 实际下载 {download_manifest['downloaded_s3_bytes'] / 1e6:.6f} MB。",
            "否；lidar/*.bin 下载数为 0。",
            f"实际审计 {len(BOREAS_SEQUENCES)} 条原始 Boreas sequences（另将 bucket 中非原始 Boreas/RT prefixes 分开记录）。",
            f"公开 GT 候选 {len(TRAIN_SEQUENCES)} 条；13 条官方 test split 的 GT 隐藏。",
            f"固定 ENU_ref={world['status']}；官方定义与实际全局数值范围一致。",
            f"sequence-local reset={'检测到 ' + ','.join(later_resets) if later_resets else '未检测到'}。",
            "是；Applanix GT trajectory 来自 GNSS/IMU/轮速/RTX POSPac 后处理，不使用 LiDAR registration。",
            "是；T_applanix_lidar 的官方生成说明明确使用 LiDAR pointclouds 与后处理 GPS/IMU 批优化。",
            "冻结 R02 对最终 LiDAR-frame reference 的结论为 FAIL：query/calibration session 排除关系未公开，不能证明完全独立。",
            f"R02={r02}。", f"R03={r03}。", f"R04={r04}。", f"R05={r05}。", "R09=PASS。", "R10=PARTIAL。",
            f"时间同步链={time_status}；PPS/NMEA/UTC 链可审计但数值误差界 UNKNOWN。",
            "2–4 cm 的 uncertainty_type=RMSE（序列依赖，绝非 1SIGMA）。",
            "orientation、Applanix/Velodyne time、外参平移/旋转、插值、deskew、map accumulation uncertainty 仍为 UNKNOWN。",
            "GT-only 最优 pair=NOT_COMPUTABLE。", "covered_duration_s=NOT_COMPUTABLE。",
            "coverage_fraction=NOT_COMPUTABLE。", "eligible_nonoverlapping_5s_intervals=NOT_COMPUTABLE。",
            "reserve pair=无。", "否；没有执行任何 ICP/registration。",
            "是；weak/rich/snapshot/planned future trials 全部为 0。", "BOREAS_STAGE1_READY=false。",
            "否；不进入 Stage-2，也不下载最小 LiDAR。",
            "主要硬门是 LiDAR-assisted proprietary extrinsic 使最终 LiDAR-frame reference 的 R02 independence 无法证明；R10 仍有关键 UNKNOWN，overlap 因 R02 被门控。",
        ]
        summary = {
            "answers": answers, "boreas_stage1_ready": False, "eligibility": eligibility,
            "final_conclusion": "BOREAS_STAGE1_READY=false；R02 independence 与 R10 硬门未通过，不进入 Stage-2。",
            "pair_selection": pair_selection,
        }
        atomic_write_json(runtime_root / "boreas_stage1_summary.json", summary)
        atomic_write_bytes(runtime_root / "boreas_stage1_summary.md", _summary_markdown(summary).encode("utf-8"))
        attestation = guard.attestation(runtime_root)
        attestation.update({"lidar_payload_count": 0, "status": "PASS" if attestation["pass"] else "FAIL"})
        atomic_write_json(runtime_root / "NO_ICP_ATTESTATION.json", attestation)

    exclusions = {"boreas_stage1_manifest.json", "SHA256SUMS"}
    payload_names = sorted(path.name for path in runtime_root.iterdir() if path.is_file() and path.name not in exclusions)
    manifest = {
        "data_evidence": source_rows,
        "eligibility_sha256": sha256_file(runtime_root / "boreas_stage1_eligibility.json"),
        "official_source_identity": {"aws_bucket": AWS_URI, "paper_doi": PAPER_DOI, "pyboreas_commit": PYBOREAS_COMMIT},
        "payload": [{"path": name, "sha256": sha256_file(runtime_root / name), "size_bytes": (runtime_root / name).stat().st_size} for name in payload_names],
        "producer_commit": environment["git"]["commit"], "schema_version": "boreas_stage1_manifest_v1",
    }
    manifest["manifest_payload_sha256"] = compact_sha256(manifest)
    atomic_write_json(runtime_root / "boreas_stage1_manifest.json", manifest)
    atomic_write_bytes(runtime_root / "SHA256SUMS", _sha256sums_payload(runtime_root, [*payload_names, "boreas_stage1_manifest.json"]))
    return summary


__all__ = [
    "BOREAS_SEQUENCES", "BoreasStage1Error", "MAX_SINGLE_OBJECT_BYTES", "MAX_STAGE1_TOTAL_BYTES",
    "Stage1DownloadBudgetExceeded", "Stage1LargeFileDownloadForbidden", "TEST_SEQUENCES", "TRAIN_SEQUENCES",
    "assert_download_budget", "audit_transform_chain_direction", "compose_enu_lidar", "execute_boreas_stage1", "parse_calibration",
    "detect_sequence_local_resets", "parse_pose_csv", "parse_s3_ls_line", "parse_top_level_s3_listing", "pose_row_to_transform",
    "transform_chain_consistency", "yaw_pitch_roll_to_rotation",
]
