"""Small, write-once evidence exporters for W02 attempt 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .w04_outputs import (
    write_csv_once,
    write_json_once,
    write_sha256sums,
)


def _bag_rows(acquisition: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = acquisition.get("raw_bags")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("W02 attempt 2 must contain exactly six raw bags")
    return rows


def _topic(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    inventory = row.get("inventory", {})
    topics = inventory.get("topics", []) if isinstance(inventory, Mapping) else []
    if isinstance(topics, list):
        for value in topics:
            if isinstance(value, Mapping) and value.get("topic") == name:
                return value
    return {}


def _summary(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = row.get(name, {})
    return value if isinstance(value, Mapping) else {}


def write_acquisition_evidence(
    acquisition: Mapping[str, Any], output_dir: Path
) -> list[Path]:
    raw = _bag_rows(acquisition)
    inventory_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    imu_rows: list[dict[str, Any]] = []
    for row in raw:
        inventory = row.get("inventory") or {}
        lidar = _summary(row, "lidar_summary")
        imu = _summary(row, "imu_summary")
        lidar_topic = _topic(row, "/livox/lidar")
        imu_topic = _topic(row, "/livox/imu")
        common = {
            "scene_id": row.get("scene_id"),
            "station_id": row.get("station_id"),
            "attempt": row.get("attempt"),
            "role": row.get("role"),
            "capture_prefix": row.get("capture_prefix"),
            "raw_filename": row.get("raw_filename"),
            "raw_absolute_path": row.get("raw_absolute_path"),
            "sha256": row.get("sha256"),
            "bytes": row.get("bytes"),
            "status": row.get("status"),
        }
        inventory_rows.append(
            {
                **common,
                "duration_s": inventory.get("duration_seconds"),
                "lidar_count": lidar_topic.get("message_count"),
                "imu_count": imu_topic.get("message_count"),
                "lidar_rate_hz": lidar_topic.get("average_bag_record_rate_hz"),
                "imu_rate_hz": imu_topic.get("average_bag_record_rate_hz"),
                "lidar_frame_ids": sorted(lidar_topic.get("frame_ids", [])),
                "imu_frame_ids": sorted(imu_topic.get("frame_ids", [])),
                "point_fields": lidar_topic.get("fields"),
            }
        )
        mapping_rows.append(
            {
                **common,
                "canonical_filename": row.get("canonical_filename"),
                "raw_part": row.get("raw_part"),
                "mtime": row.get("mtime"),
                "mtime_utc": row.get("mtime_utc"),
            }
        )
        statistics = imu.get("statistics", {})
        imu_rows.append(
            {
                **common,
                "motion_status": imu.get("STATICITY_SCREEN"),
                "gyro_norm_median": statistics.get("gyro_norm", {}).get("median"),
                "gyro_norm_mad": statistics.get("gyro_norm", {}).get("mad"),
                "gyro_norm_q95": statistics.get("gyro_norm", {}).get("q95"),
                "gyro_norm_q99": statistics.get("gyro_norm", {}).get("q99"),
                "gyro_norm_max": statistics.get("gyro_norm", {}).get("max"),
                "accel_norm_median": statistics.get("accel_norm", {}).get("median"),
                "accel_norm_mad": statistics.get("accel_norm", {}).get("mad"),
                "accel_norm_q95": statistics.get("accel_norm", {}).get("q95"),
                "accel_norm_q99": statistics.get("accel_norm", {}).get("q99"),
                "accel_norm_max": statistics.get("accel_norm", {}).get("max"),
                "large_spike_count": imu.get("large_spike_count"),
                "acceleration_unit": "UNKNOWN",
            }
        )
    station_rows = []
    for row in acquisition.get("stations", []):
        pair = row.get("pair_audit", {})
        station_rows.append(
            {
                "scene_id": row.get("scene_id"),
                "station_id": row.get("station_id"),
                "attempt": row.get("attempt"),
                "capture_prefix": row.get("capture_prefix"),
                "status": row.get("station_acquisition_status"),
                "gap_s": pair.get("actual_gap_s"),
                "gap_pass": pair.get("actual_gap_minimum_pass"),
                "map_query_no_overlap": pair.get("map_query_no_overlap"),
                "failure_reasons": pair.get("exclusion_reasons", []),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = lambda rows: tuple(rows[0].keys())
    paths = [
        write_json_once(output_dir / "w02_attempt2_raw_bag_inventory.json", acquisition),
        write_csv_once(
            output_dir / "w02_attempt2_raw_bag_inventory.csv",
            inventory_rows,
            fields(inventory_rows),
        ),
        write_csv_once(
            output_dir / "w02_attempt2_raw_to_canonical_mapping.csv",
            mapping_rows,
            fields(mapping_rows),
        ),
        write_csv_once(
            output_dir / "w02_attempt2_station_acquisition_audit.csv",
            station_rows,
            fields(station_rows),
        ),
        write_json_once(
            output_dir / "w02_attempt2_station_acquisition_audit.json",
            {
                "schema": "mid360_fmb1_w02_attempt2_station_audit_v1",
                "W02_ATTEMPT2_ACQUISITION_PASS": acquisition.get(
                    "W02_ATTEMPT2_ACQUISITION_PASS"
                ),
                "stations": acquisition.get("stations", []),
            },
        ),
        write_csv_once(
            output_dir / "w02_attempt2_imu_staticity_audit.csv",
            imu_rows,
            fields(imu_rows),
        ),
    ]
    return paths


def write_asset_evidence(assets: Mapping[str, Any], output_dir: Path) -> list[Path]:
    targets = assets.get("targets", [])
    snapshots = assets.get("snapshots", [])
    if not isinstance(targets, list) or len(targets) != 3:
        raise ValueError("W02 attempt 2 must contain exactly three targets")
    if not isinstance(snapshots, list) or len(snapshots) != 30:
        raise ValueError("W02 attempt 2 must contain exactly thirty snapshots")
    return [
        write_json_once(output_dir / "w02_attempt2_target_map_manifest.json", assets),
        write_csv_once(
            output_dir / "w02_attempt2_target_map_manifest.csv",
            targets,
            tuple(targets[0].keys()),
        ),
        write_csv_once(
            output_dir / "w02_attempt2_snapshot_inventory_pregeometry.csv",
            snapshots,
            tuple(snapshots[0].keys()),
        ),
    ]


def write_geometry_evidence(
    geometry: Mapping[str, Any], output_dir: Path
) -> list[Path]:
    metrics = geometry.get("snapshot_metrics", [])
    scenes = geometry.get("scene_summaries", [])
    if not isinstance(metrics, list) or len(metrics) != 30:
        raise ValueError("W02 attempt 2 geometry must contain thirty rows")
    if not isinstance(scenes, list) or len(scenes) != 1:
        raise ValueError("W02 attempt 2 geometry must contain one scene")
    admission = {
        "schema": "mid360_fmb1_w02_attempt2_geometry_admission_v1",
        "attempt": 2,
        "scene": scenes[0],
        "FMB1_W02_ATTEMPT2_FINAL_GEOMETRY_CLASS": scenes[0].get(
            "final_geometry_class"
        ),
        "FMB1_W02_ATTEMPT2_ADMISSION_PASS": bool(
            scenes[0].get("final_geometry_class") == "WEAK"
            and scenes[0].get("geometry_admission_status") == "GEOMETRY_ADMITTED"
        ),
        "actual_formal_trials": 0,
    }
    return [
        write_csv_once(
            output_dir / "w02_attempt2_geometry_metrics.csv",
            metrics,
            tuple(metrics[0].keys()),
        ),
        write_csv_once(
            output_dir / "w02_attempt2_geometry_scene_summary.csv",
            scenes,
            tuple(scenes[0].keys()),
        ),
        write_json_once(
            output_dir / "w02_attempt2_geometry_admission.json", admission
        ),
    ]


__all__ = [
    "write_acquisition_evidence",
    "write_asset_evidence",
    "write_geometry_evidence",
    "write_json_once",
    "write_sha256sums",
]
