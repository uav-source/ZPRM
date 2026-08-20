"""Small, write-once evidence exporters for the W04 replacement pipeline."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from phase_a_harness.mid360_pilot.bag_reader import IMU_TOPIC, LIDAR_TOPIC
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    GEOMETRY_ONLY_FIELDS,
)

from .preregistration_finalize import (
    _audit,
    _bag_for,
    _flatten_imu,
    _imu,
    _inventory,
    _path,
    _sha,
    _station_pair,
    _station_status,
    _topic,
)


class W04OutputError(RuntimeError):
    """A W04 evidence write or schema gate failed."""


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_once(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise W04OutputError(f"refusing to overwrite different evidence: {path}")
        return path
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return path


def write_json_once(path: Path, payload: Any) -> Path:
    return _write_once(path, _canonical_json(payload))


def write_csv_once(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> Path:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        encoded = {
            field: (
                json.dumps(row.get(field), sort_keys=True, ensure_ascii=False)
                if isinstance(row.get(field), (dict, list))
                else row.get(field)
            )
            for field in fields
        }
        writer.writerow(encoded)
    return _write_once(path, stream.getvalue().encode("utf-8"))


def _raw_rows(acquisition: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = acquisition.get("raw_bags")
    if not isinstance(value, list) or len(value) != 6:
        raise W04OutputError("W04 acquisition must contain exactly six raw bags")
    return sorted(
        value, key=lambda row: (str(row["station_id"]), str(row["role"]))
    )


def _station_rows(acquisition: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = acquisition.get("stations")
    if not isinstance(value, list) or len(value) != 3:
        raise W04OutputError("W04 acquisition must contain exactly three stations")
    return sorted(value, key=lambda row: str(row["station_id"]))


def acquisition_rows(acquisition: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw = _raw_rows(acquisition)
    mapping_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for row in raw:
        path = _path(row)
        stat = path.stat()
        common = {
            "scene_id": "FMB1_W04",
            "station_id": row["station_id"],
            "role": row["role"],
            "raw_filename": path.name,
            "absolute_path": str(path),
            "raw_absolute_path": str(path),
            "canonical_filename": row["canonical_filename"],
            "capture_prefix": row["capture_prefix"],
            "sha256": _sha(row),
            "bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
            "mtime_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "canonical_materialization": "MANIFEST_ONLY_NO_COPY",
        }
        mapping_rows.append(common)
        inventory_rows.append(
            {
                **common,
                "start_timestamp": _inventory(row).get("start_timestamp"),
                "end_timestamp": _inventory(row).get("end_timestamp"),
                "duration_s": _inventory(row).get("duration_seconds"),
                "lidar_count": _topic(_inventory(row), LIDAR_TOPIC).get(
                    "message_count"
                ),
                "imu_count": _topic(_inventory(row), IMU_TOPIC).get(
                    "message_count"
                ),
                "lidar_rate_hz": _topic(_inventory(row), LIDAR_TOPIC).get(
                    "average_bag_record_rate_hz"
                ),
                "imu_rate_hz": _topic(_inventory(row), IMU_TOPIC).get(
                    "average_bag_record_rate_hz"
                ),
                "frame_ids": ";".join(_audit(row).get("frame_ids", [])),
                "pointcloud2_fields": ";".join(
                    _audit(row).get("pointcloud2_fields", [])
                ),
                "motion_status": _audit(row).get("motion_audit_status"),
                "bag_acquisition_pass": _audit(row).get(
                    "ACQUISITION_AUDIT_PASS", False
                ),
            }
        )
    station_rows: list[dict[str, Any]] = []
    for station in _station_rows(acquisition):
        station_id = str(station["station_id"])
        map_row = _bag_for(raw, "FMB1_W04", station_id, "MAP")
        query_row = _bag_for(raw, "FMB1_W04", station_id, "QUERY")
        pair = _station_pair(station)
        failures = station.get("failure_reasons", station.get("failure_reason", []))
        if isinstance(failures, str):
            failures = [failures] if failures else []
        station_rows.append(
            {
                "scene_id": "FMB1_W04",
                "station_id": station_id,
                "capture_prefix": map_row["capture_prefix"],
                "semantic_candidate_label": "WEAK_CANDIDATE",
                "map_sha256": _sha(map_row),
                "query_sha256": _sha(query_row),
                "map_duration_s": _audit(map_row).get("duration_s"),
                "query_duration_s": _audit(query_row).get("duration_s"),
                "map_lidar_count": _topic(_inventory(map_row), LIDAR_TOPIC).get(
                    "message_count"
                ),
                "query_lidar_count": _topic(
                    _inventory(query_row), LIDAR_TOPIC
                ).get("message_count"),
                "map_imu_count": _topic(_inventory(map_row), IMU_TOPIC).get(
                    "message_count"
                ),
                "query_imu_count": _topic(_inventory(query_row), IMU_TOPIC).get(
                    "message_count"
                ),
                "map_lidar_rate_hz": _topic(
                    _inventory(map_row), LIDAR_TOPIC
                ).get("average_bag_record_rate_hz"),
                "query_lidar_rate_hz": _topic(
                    _inventory(query_row), LIDAR_TOPIC
                ).get("average_bag_record_rate_hz"),
                "map_imu_rate_hz": _topic(_inventory(map_row), IMU_TOPIC).get(
                    "average_bag_record_rate_hz"
                ),
                "query_imu_rate_hz": _topic(
                    _inventory(query_row), IMU_TOPIC
                ).get("average_bag_record_rate_hz"),
                "map_motion_status": _audit(map_row).get("motion_audit_status"),
                "query_motion_status": _audit(query_row).get(
                    "motion_audit_status"
                ),
                "gap_s": pair.get("actual_gap_s"),
                "map_query_no_overlap": pair.get("map_query_no_overlap"),
                "gap_pass": pair.get("actual_gap_minimum_pass"),
                "timestamp_basis": "ROSBAG_GLOBAL_MESSAGE_TIME",
                "acquisition_status": _station_status(station),
                "failure_reason": ";".join(str(value) for value in failures),
                "invalidated_at_utc": station.get("invalidated_at_utc"),
            }
        )
    return {
        "inventory": inventory_rows,
        "mapping": mapping_rows,
        "stations": station_rows,
        "imu": [_flatten_imu(row) for row in raw],
    }


INVENTORY_FIELDS = (
    "scene_id", "station_id", "role", "raw_filename", "absolute_path",
    "canonical_filename", "capture_prefix", "sha256", "bytes", "mtime",
    "mtime_utc", "canonical_materialization", "start_timestamp",
    "end_timestamp", "duration_s", "lidar_count", "imu_count",
    "lidar_rate_hz", "imu_rate_hz", "frame_ids", "pointcloud2_fields",
    "motion_status", "bag_acquisition_pass",
)
MAPPING_FIELDS = (
    "scene_id", "station_id", "role", "raw_filename", "raw_absolute_path",
    "canonical_filename", "capture_prefix", "sha256", "bytes", "mtime",
    "mtime_utc", "canonical_materialization",
)
STATION_FIELDS = (
    "scene_id", "station_id", "capture_prefix", "semantic_candidate_label",
    "map_sha256", "query_sha256", "map_duration_s", "query_duration_s",
    "map_lidar_count", "query_lidar_count", "map_imu_count", "query_imu_count",
    "map_lidar_rate_hz", "query_lidar_rate_hz", "map_imu_rate_hz",
    "query_imu_rate_hz", "map_motion_status", "query_motion_status", "gap_s",
    "map_query_no_overlap", "gap_pass", "timestamp_basis",
    "acquisition_status", "failure_reason", "invalidated_at_utc",
)
IMU_FIELDS = (
    "scene_id", "station_id", "role", "raw_filename", "sha256",
    "motion_status", "acceleration_unit", "acceleration_conversion_applied",
    "large_spike_count", "maximum_gyro_step", "maximum_accel_step_raw",
    *tuple(
        f"{channel}_{statistic}"
        for channel in (
            "gyro_x", "gyro_y", "gyro_z", "gyro_norm",
            "accel_x", "accel_y", "accel_z", "accel_norm",
        )
        for statistic in (
            "median", "mad", "q95", "q99", "max", "min", "absolute_max"
        )
    ),
)


def write_acquisition_evidence(
    acquisition: Mapping[str, Any], output_dir: Path
) -> list[Path]:
    rows = acquisition_rows(acquisition)
    inventory_payload = {
        "schema": "mid360_fmb1_w04_raw_bag_inventory_v1",
        "status": acquisition.get("status"),
        "raw_bag_count": len(rows["inventory"]),
        "rows": rows["inventory"],
    }
    station_payload = {
        "schema": "mid360_fmb1_w04_station_acquisition_audit_v1",
        "status": acquisition.get("status"),
        "W04_ACQUISITION_PASS": acquisition.get("W04_ACQUISITION_PASS"),
        "station_count": len(rows["stations"]),
        "station_pass_count": sum(
            row["acquisition_status"] == "ACQUISITION_PASS"
            for row in rows["stations"]
        ),
        "rows": rows["stations"],
    }
    return [
        write_csv_once(
            output_dir / "w04_raw_bag_inventory.csv", rows["inventory"], INVENTORY_FIELDS
        ),
        write_json_once(output_dir / "w04_raw_bag_inventory.json", inventory_payload),
        write_csv_once(
            output_dir / "w04_raw_to_canonical_mapping.csv",
            rows["mapping"],
            MAPPING_FIELDS,
        ),
        write_csv_once(
            output_dir / "w04_station_acquisition_audit.csv",
            rows["stations"],
            STATION_FIELDS,
        ),
        write_json_once(
            output_dir / "w04_station_acquisition_audit.json", station_payload
        ),
        write_csv_once(
            output_dir / "w04_imu_staticity_audit.csv", rows["imu"], IMU_FIELDS
        ),
    ]


TARGET_FIELDS = (
    "scene_id", "station_id", "map_bag_sha256", "map_frame_count",
    "raw_point_count", "filtered_point_count", "target_point_count",
    "target_path", "target_npy_sha256", "target_array_sha256",
    "target_size_bytes", "input_roles", "query_frame_count",
    "query_contribution_to_target", "construction", "registration_called",
    "odometry_called", "scan_matching_called", "minimum_range_m",
    "maximum_range_m", "voxel_size_m", "preprocessing_source_sha256",
)
SNAPSHOT_FIELDS = (
    "scene_id", "station_id", "snapshot_id", "selection_index", "quantile",
    "candidate_count", "target_sequence_rank", "selected_sequence_rank",
    "query_frame_index", "query_timestamp", "query_bag_sha256", "source_path",
    "source_npy_sha256", "source_array_sha256", "source_point_count",
    "source_size_bytes", "target_npy_sha256", "target_array_sha256",
    "selection_method", "selection_frozen_before_registration",
)


def asset_rows(assets: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    targets = assets.get("targets")
    snapshots = assets.get("snapshots")
    if not isinstance(targets, list) or len(targets) != 3:
        raise W04OutputError("W04 must have exactly three targets")
    if not isinstance(snapshots, list) or len(snapshots) != 30:
        raise W04OutputError("W04 must have exactly thirty snapshots")
    target_rows: list[dict[str, Any]] = []
    for row in sorted(targets, key=lambda value: str(value["station_id"])):
        target_rows.append(
            {
                **{field: row.get(field) for field in TARGET_FIELDS},
                "target_size_bytes": Path(str(row["target_path"])).stat().st_size,
                "input_roles": ["MAP"],
                "query_frame_count": 0,
                "query_contribution_to_target": 0,
                "registration_called": False,
                "odometry_called": False,
                "scan_matching_called": False,
            }
        )
    snapshot_rows: list[dict[str, Any]] = []
    for row in sorted(
        snapshots,
        key=lambda value: (str(value["station_id"]), int(value["selection_index"])),
    ):
        snapshot_rows.append(
            {
                **{field: row.get(field) for field in SNAPSHOT_FIELDS},
                "source_size_bytes": Path(str(row["source_path"])).stat().st_size,
            }
        )
    return {"targets": target_rows, "snapshots": snapshot_rows}


def write_asset_evidence(assets: Mapping[str, Any], output_dir: Path) -> list[Path]:
    rows = asset_rows(assets)
    target_payload = {
        "schema": "mid360_fmb1_w04_target_map_manifest_v1",
        "status": "PASS" if not assets.get("failures") else "FAIL",
        "target_count": len(rows["targets"]),
        "query_contribution_to_every_target": 0,
        "rows": rows["targets"],
    }
    return [
        write_csv_once(
            output_dir / "w04_target_map_manifest.csv",
            rows["targets"],
            TARGET_FIELDS,
        ),
        write_json_once(output_dir / "w04_target_map_manifest.json", target_payload),
        write_csv_once(
            output_dir / "w04_snapshot_inventory_pregeometry.csv",
            rows["snapshots"],
            SNAPSHOT_FIELDS,
        ),
    ]


METRIC_FIELDS = (
    "scene_id", "station_id", "snapshot_id", "selection_index",
    *tuple(GEOMETRY_ONLY_FIELDS),
)


def write_geometry_evidence(
    geometry: Mapping[str, Any], output_dir: Path
) -> list[Path]:
    metrics = geometry.get("snapshot_metrics")
    scenes = geometry.get("scene_summaries")
    if not isinstance(metrics, list) or len(metrics) != 30:
        raise W04OutputError("W04 geometry must contain exactly thirty rows")
    if not isinstance(scenes, list) or len(scenes) != 1:
        raise W04OutputError("W04 geometry must contain exactly one scene summary")
    metric_rows = sorted(
        metrics,
        key=lambda row: (str(row["station_id"]), int(row["selection_index"])),
    )
    scene_fields = tuple(sorted({key for row in scenes for key in row}))
    admission = {
        "schema": "mid360_fmb1_w04_geometry_admission_v1",
        "scene": scenes[0],
        "FMB1_W04_FINAL_GEOMETRY_CLASS": scenes[0].get("final_geometry_class"),
        "FMB1_W04_ADMISSION_PASS": bool(
            scenes[0].get("final_geometry_class") == "WEAK"
            and scenes[0].get("geometry_admission_status") == "GEOMETRY_ADMITTED"
        ),
        "registration_execution_count": 0,
        "actual_formal_trials": 0,
    }
    return [
        write_csv_once(
            output_dir / "w04_geometry_metrics.csv", metric_rows, METRIC_FIELDS
        ),
        write_csv_once(
            output_dir / "w04_geometry_scene_summary.csv", scenes, scene_fields
        ),
        write_json_once(output_dir / "w04_geometry_admission.json", admission),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(output_dir: Path) -> Path:
    paths = sorted(
        (
            path
            for path in output_dir.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        key=lambda path: path.name,
    )
    content = "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths)
    return _write_once(output_dir / "SHA256SUMS", content.encode("ascii"))


__all__ = [
    "W04OutputError",
    "acquisition_rows",
    "asset_rows",
    "sha256_file",
    "write_acquisition_evidence",
    "write_asset_evidence",
    "write_geometry_evidence",
    "write_json_once",
    "write_sha256sums",
]
