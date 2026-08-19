"""Machine-readable FMB1 acquisition, geometry, and analysis-firewall rules.

This module deliberately contains no ICP/backend call.  It reuses the frozen
Mid-360 readers, static target builder, motion screen, query selector, and
geometry-only implementation.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from phase_a_harness.mid360_pilot.bag_reader import (
    IMU_TOPIC,
    LIDAR_TOPIC,
    PilotBagError,
    build_bag_inventory,
    iter_topic_messages,
    sha256_file,
)
from phase_a_harness.mid360_pilot.imu_audit import audit_imu_messages
from phase_a_harness.mid360_pilot.lidar_adapter import (
    audit_lidar_messages,
    lidar_message_to_structured,
    xyz_array,
)
from phase_a_harness.mid360_pilot.pilot_geometry import compute_pilot_geometry
from phase_a_harness.mid360_pilot.split import select_query_frames
from phase_a_harness.mid360_pilot.static_map import (
    build_static_target_map,
    finite_range_filter,
)


BATCH_ID = "FMB1"
INITIAL_SCENE_IDS = (
    "FMB1_R01",
    "FMB1_R02",
    "FMB1_R03",
    "FMB1_W01",
    "FMB1_W02",
    "FMB1_W03",
)
STATION_IDS = ("S01", "S02", "S03")
ROLES = ("MAP", "QUERY")
REQUIRED_TOPICS = {
    LIDAR_TOPIC: "sensor_msgs/PointCloud2",
    IMU_TOPIC: "sensor_msgs/Imu",
}
REQUIRED_FRAME_ID = "livox_frame"
REQUIRED_POINT_FIELDS = frozenset(
    {"x", "y", "z", "intensity", "tag", "line", "timestamp"}
)
FORBIDDEN_BAG_WORDS = (
    "good",
    "bad",
    "best",
    "final",
    "retry_good",
    "weakest",
)
MAP_TARGET_DURATION_S = 20.0
QUERY_TARGET_DURATION_S = 15.0
MAP_MIN_DURATION_S = 19.0
QUERY_MIN_DURATION_S = 14.0
TARGET_MAP_QUERY_GAP_S = 12.0
MINIMUM_MAP_QUERY_GAP_S = 10.0
LIDAR_RATE_RANGE_HZ = (9.5, 10.5)
IMU_RATE_RANGE_HZ = (190.0, 210.0)
QUERY_QUANTILES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
BACKEND_CONTRACT_SHA256 = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
CONTROLLED_PILOT_SHA256SUMS_SHA256 = (
    "51e66ef1b8147add7162116f06a9957de51a075ab72dd9efd1f8afe89a34fe01"
)
CAPTURE_PILOT_SHA256SUMS_SHA256 = (
    "cecbeb3a921e0c29588b4f20a62d5e0b90ec3cf3cb0a97505ac62ff18e0569e8"
)
SCENE_RE = re.compile(r"^FMB1_[RW](\d{2})$")
STATION_RE = re.compile(r"^S0[1-3]$")
BAG_RE = re.compile(
    r"^(FMB1_[RW]\d{2})_(S0[1-3])_(MAP|QUERY)_(\d{8}_\d{6})\.bag$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FormalBatchError(RuntimeError):
    """Raised when a preregistered FMB1 contract would be violated."""


def read_json_yaml(path: Path) -> dict[str, Any]:
    """Read our JSON-form YAML 1.2 files without requiring PyYAML in ROS Python."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FormalBatchError(f"expected object in {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise FormalBatchError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_scene_id(scene_id: str) -> str:
    match = SCENE_RE.fullmatch(scene_id)
    if match is None or int(match.group(1)) == 0:
        raise FormalBatchError(f"invalid FMB1 scene ID: {scene_id}")
    return scene_id


def validate_station_id(station_id: str) -> str:
    if STATION_RE.fullmatch(station_id) is None:
        raise FormalBatchError(f"invalid FMB1 station ID: {station_id}")
    return station_id


def validate_role(role: str) -> str:
    if role not in ROLES:
        raise FormalBatchError(f"invalid bag role: {role}")
    return role


def bag_filename(scene_id: str, station_id: str, role: str, timestamp: str) -> str:
    validate_scene_id(scene_id)
    validate_station_id(station_id)
    validate_role(role)
    try:
        datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    except ValueError as exc:
        raise FormalBatchError(f"invalid bag timestamp: {timestamp}") from exc
    filename = f"{scene_id}_{station_id}_{role}_{timestamp}.bag"
    lowered = filename.lower()
    if any(word in lowered for word in FORBIDDEN_BAG_WORDS):
        raise FormalBatchError("result-leaning bag name is forbidden")
    if BAG_RE.fullmatch(filename) is None:
        raise FormalBatchError(f"generated bag name violates schema: {filename}")
    return filename


def parse_bag_filename(filename: str) -> dict[str, str]:
    match = BAG_RE.fullmatch(Path(filename).name)
    if match is None:
        raise FormalBatchError(f"bag name violates FMB1 schema: {filename}")
    return {
        "scene_id": match.group(1),
        "station_id": match.group(2),
        "role": match.group(3),
        "timestamp": match.group(4),
    }


def build_recording_plan(
    *,
    batch: str,
    scene_id: str,
    station_id: str,
    output_dir: Path,
    timestamp: str,
    attempt: int,
) -> dict[str, Any]:
    if batch != BATCH_ID:
        raise FormalBatchError(f"batch must be {BATCH_ID}")
    if attempt < 1:
        raise FormalBatchError("attempt must be a positive integer")
    validate_scene_id(scene_id)
    validate_station_id(station_id)
    start = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    query_start = start + timedelta(
        seconds=MAP_TARGET_DURATION_S + TARGET_MAP_QUERY_GAP_S
    )
    map_path = output_dir / bag_filename(scene_id, station_id, "MAP", timestamp)
    query_timestamp = query_start.strftime("%Y%m%d_%H%M%S")
    query_path = output_dir / bag_filename(
        scene_id, station_id, "QUERY", query_timestamp
    )
    metadata_path = output_dir / (
        f"{scene_id}_{station_id}_ATTEMPT{attempt:02d}_{timestamp}.pair.json"
    )
    return {
        "schema": "mid360_formal_batch1_recording_plan_v1",
        "batch_id": batch,
        "scene_id": scene_id,
        "station_id": station_id,
        "attempt": attempt,
        "map_path": str(map_path),
        "query_path": str(query_path),
        "pair_metadata_path": str(metadata_path),
        "topics": dict(REQUIRED_TOPICS),
        "map_duration_target_s": MAP_TARGET_DURATION_S,
        "wait_target_s": TARGET_MAP_QUERY_GAP_S,
        "query_duration_target_s": QUERY_TARGET_DURATION_S,
        "steps": [
            "VERIFY_LIVE_TOPICS",
            "RECORD_MAP_20S",
            "WAIT_12S",
            "RECORD_QUERY_15S",
            "COMPUTE_SHA256",
            "RUN_IMMEDIATE_AUDIT",
            "WRITE_PAIR_METADATA",
        ],
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_MEASUREMENT_RESULT": False,
    }


def ensure_paths_do_not_exist(paths: Iterable[Path]) -> None:
    existing = sorted(str(path) for path in paths if path.exists())
    if existing:
        raise FormalBatchError(f"refusing to overwrite prior attempt: {existing}")


def duration_passes(role: str, duration_s: float) -> bool:
    validate_role(role)
    minimum = MAP_MIN_DURATION_S if role == "MAP" else QUERY_MIN_DURATION_S
    return bool(np.isfinite(duration_s) and duration_s >= minimum)


def pair_timing_audit(
    map_start: float, map_end: float, query_start: float, query_end: float
) -> dict[str, Any]:
    values = (map_start, map_end, query_start, query_end)
    finite = all(np.isfinite(value) for value in values)
    map_ordered = finite and map_end >= map_start
    query_ordered = finite and query_end >= query_start
    gap = float(query_start - map_end) if finite else None
    no_overlap = bool(finite and query_start >= map_end)
    gap_pass = bool(no_overlap and gap is not None and gap >= MINIMUM_MAP_QUERY_GAP_S)
    return {
        "map_start_timestamp": map_start,
        "map_end_timestamp": map_end,
        "query_start_timestamp": query_start,
        "query_end_timestamp": query_end,
        "actual_gap_s": gap,
        "map_query_no_overlap": no_overlap,
        "actual_gap_minimum_pass": gap_pass,
        "timing_pass": bool(map_ordered and query_ordered and gap_pass),
    }


def _topic_rows(inventory: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["topic"]): row for row in inventory.get("topics", [])}


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def evaluate_bag_audit(
    inventory: Mapping[str, Any],
    *,
    role: str,
    lidar_summary: Mapping[str, Any],
    imu_summary: Mapping[str, Any],
    finite_point_fraction: float,
) -> dict[str, Any]:
    validate_role(role)
    topics = _topic_rows(inventory)
    lidar = topics.get(LIDAR_TOPIC, {})
    imu = topics.get(IMU_TOPIC, {})
    observed_topics = set(topics)
    lidar_rate = lidar.get("average_bag_record_rate_hz")
    imu_rate = imu.get("average_bag_record_rate_hz")
    observed_fields = {str(field.get("name")) for field in lidar.get("fields", [])}
    frames = {
        str(frame)
        for row in topics.values()
        for frame in row.get("frame_ids", [])
    }
    duration = float(inventory.get("duration_seconds", float("nan")))
    checks = {
        "sha256": _valid_sha(inventory.get("bag_sha256")),
        "exact_topics": observed_topics == set(REQUIRED_TOPICS),
        "topic_types": all(
            topics.get(topic, {}).get("message_type") == expected
            for topic, expected in REQUIRED_TOPICS.items()
        ),
        "frame_id": all(
            set(topics.get(topic, {}).get("frame_ids", [])) == {REQUIRED_FRAME_ID}
            for topic in REQUIRED_TOPICS
        ),
        "duration": duration_passes(role, duration),
        "lidar_rate": isinstance(lidar_rate, (int, float))
        and LIDAR_RATE_RANGE_HZ[0] <= float(lidar_rate) <= LIDAR_RATE_RANGE_HZ[1],
        "imu_rate": isinstance(imu_rate, (int, float))
        and IMU_RATE_RANGE_HZ[0] <= float(imu_rate) <= IMU_RATE_RANGE_HZ[1],
        "pointcloud2_fields": REQUIRED_POINT_FIELDS.issubset(observed_fields),
        "finite_points": np.isfinite(finite_point_fraction)
        and 0.0 < finite_point_fraction <= 1.0,
        "lidar_payload": lidar_summary.get("status") == "PASS",
        "imu_integrity": imu_summary.get("status") == "PASS",
        "obvious_motion": imu_summary.get("STATICITY_SCREEN") == "NO_OBVIOUS_MOTION",
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    review = not checks["frame_id"] or not checks["pointcloud2_fields"]
    return {
        "schema": "mid360_formal_batch1_bag_audit_v1",
        "role": role,
        "bag_path": inventory.get("bag_path"),
        "SHA256": inventory.get("bag_sha256"),
        "start_time": inventory.get("start_timestamp"),
        "end_time": inventory.get("end_timestamp"),
        "duration_s": duration,
        "topics": sorted(observed_topics),
        "message_counts": {
            topic: int(row.get("message_count", 0)) for topic, row in topics.items()
        },
        "frequencies_hz": {LIDAR_TOPIC: lidar_rate, IMU_TOPIC: imu_rate},
        "frame_ids": sorted(frames),
        "pointcloud2_fields": sorted(observed_fields),
        "finite_point_fraction": float(finite_point_fraction),
        "motion_audit_status": imu_summary.get("STATICITY_SCREEN"),
        "large_spike_count": int(imu_summary.get("large_spike_count", 0)),
        "ACCELERATION_UNIT_UNKNOWN": True,
        "acceleration_conversion_applied": False,
        "point_coordinate_unit": "ASSUMED_METERS_FROM_SCALE",
        "checks": checks,
        "failure_reasons": failures,
        "REVIEW_REQUIRED": review,
        "ACQUISITION_AUDIT_PASS": not failures,
    }


def audit_bag(path: Path, role: str, config: Mapping[str, Any]) -> dict[str, Any]:
    inventory = build_bag_inventory(path)
    lidar_rows, lidar_summary = audit_lidar_messages(
        iter_topic_messages(path, LIDAR_TOPIC)
    )
    _, imu_summary = audit_imu_messages(
        iter_topic_messages(path, IMU_TOPIC), config
    )
    point_count = sum(int(row["point_count"]) for row in lidar_rows)
    finite_count = sum(int(row["finite_point_count"]) for row in lidar_rows)
    fraction = finite_count / point_count if point_count else 0.0
    return evaluate_bag_audit(
        inventory,
        role=role,
        lidar_summary=lidar_summary,
        imu_summary=imu_summary,
        finite_point_fraction=fraction,
    )


def audit_pair(
    map_audit: Mapping[str, Any], query_audit: Mapping[str, Any]
) -> dict[str, Any]:
    timing = pair_timing_audit(
        float(map_audit["start_time"]),
        float(map_audit["end_time"]),
        float(query_audit["start_time"]),
        float(query_audit["end_time"]),
    )
    valid = bool(
        map_audit.get("ACQUISITION_AUDIT_PASS")
        and query_audit.get("ACQUISITION_AUDIT_PASS")
        and timing["timing_pass"]
    )
    reasons = []
    if not map_audit.get("ACQUISITION_AUDIT_PASS"):
        reasons.append("MAP_AUDIT_FAIL")
    if not query_audit.get("ACQUISITION_AUDIT_PASS"):
        reasons.append("QUERY_AUDIT_FAIL")
    if not timing["map_query_no_overlap"]:
        reasons.append("MAP_QUERY_OVERLAP")
    elif not timing["actual_gap_minimum_pass"]:
        reasons.append("MAP_QUERY_GAP_LT_10S")
    return {
        "schema": "mid360_formal_batch1_pair_audit_v1",
        **timing,
        "FORMAL_PAIR_VALID": valid,
        "exclusion_reasons": reasons,
        "raw_data_retained": True,
    }


def geometry_class(
    normalized_lambda_min: float, condition_number: float, entropy: float
) -> str:
    if (
        normalized_lambda_min >= 0.18
        and condition_number <= 3.0
        and entropy >= 0.90
    ):
        return "RICH"
    if (
        normalized_lambda_min <= 0.12
        and condition_number >= 6.0
        and entropy <= 0.80
    ):
        return "WEAK"
    return "INTERMEDIATE"


def geometry_admission(
    normalized_lambda_min: float, condition_number: float, entropy: float
) -> dict[str, Any]:
    classification = geometry_class(
        normalized_lambda_min, condition_number, entropy
    )
    return {
        "geometry_lambda_min_median": float(normalized_lambda_min),
        "geometry_condition_median": float(condition_number),
        "geometry_entropy_median": float(entropy),
        "final_geometry_class": classification,
        "admitted": classification in {"RICH", "WEAK"},
        "exclusion_reason": (
            None if classification in {"RICH", "WEAK"} else "GEOMETRY_INTERMEDIATE"
        ),
    }


def replacement_is_allowed(reason: str, *, icp_result_computed: bool) -> bool:
    allowed = {
        "ACQUISITION_QUALITY",
        "BAG_INTEGRITY",
        "GEOMETRY_ONLY_INELIGIBLE",
        "GEOMETRY_INTERMEDIATE",
    }
    return not icp_result_computed and reason in allowed


def build_bag_manifest(
    pair_metadata_paths: Sequence[Path], output_path: Path
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    seen_metadata: set[Path] = set()
    seen_bags: set[str] = set()
    for path in sorted(pair_metadata_paths):
        resolved = path.resolve(strict=True)
        if resolved in seen_metadata:
            raise FormalBatchError(f"duplicate pair metadata: {resolved}")
        seen_metadata.add(resolved)
        row = json.loads(resolved.read_text(encoding="utf-8"))
        validate_scene_id(str(row["scene_id"]))
        validate_station_id(str(row["station_id"]))
        if int(row["attempt"]) < 1:
            raise FormalBatchError("attempt must be positive")
        for role_key in ("map", "query"):
            bag = row[f"{role_key}_audit"]["bag_path"]
            parsed = parse_bag_filename(Path(str(bag)).name)
            if parsed["scene_id"] != row["scene_id"] or parsed["station_id"] != row["station_id"]:
                raise FormalBatchError("pair metadata and bag filename disagree")
            if bag in seen_bags:
                raise FormalBatchError(f"bag reused by multiple attempts: {bag}")
            seen_bags.add(str(bag))
        attempts.append(dict(row, pair_metadata_path=str(resolved)))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        grouped[(str(row["scene_id"]), str(row["station_id"]))].append(row)
    selected: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["attempt"]))
        attempt_numbers = [int(row["attempt"]) for row in ordered]
        if len(attempt_numbers) != len(set(attempt_numbers)):
            raise FormalBatchError(f"duplicate attempt number for {key}")
        passing = [row for row in ordered if row.get("pair_audit", {}).get("FORMAL_PAIR_VALID")]
        if passing:
            chosen = dict(passing[0])
            chosen["selection_rule"] = "FIRST_AUDIT_PASSING_ATTEMPT"
            selected.append(chosen)
    payload = {
        "schema": "mid360_formal_batch1_bag_manifest_v1",
        "batch_id": BATCH_ID,
        "all_attempts_retained": True,
        "selection_uses_icp": False,
        "attempts": attempts,
        "selected_pairs": selected,
        "selected_valid_pair_count": len(selected),
        "FORMAL_MEASUREMENT_RESULT": False,
    }
    write_json(output_path, payload)
    audit_fields = [
        "scene_id", "station_id", "attempt",
        "map_bag_path", "map_sha256", "map_duration_s", "map_lidar_hz", "map_imu_hz",
        "map_frame_id", "map_motion_audit", "map_large_spike_count",
        "query_bag_path", "query_sha256", "query_duration_s", "query_lidar_hz", "query_imu_hz",
        "query_frame_id", "query_motion_audit", "query_large_spike_count",
        "actual_gap_s", "map_query_no_overlap", "formal_pair_valid", "exclusion_reason",
    ]
    audit_rows: list[dict[str, Any]] = []
    for row in attempts:
        map_audit = row.get("map_audit", {})
        query_audit = row.get("query_audit", {})
        pair_audit = row.get("pair_audit", {})
        map_rates = map_audit.get("frequencies_hz", {})
        query_rates = query_audit.get("frequencies_hz", {})
        audit_rows.append(
            {
                "scene_id": row["scene_id"],
                "station_id": row["station_id"],
                "attempt": row["attempt"],
                "map_bag_path": map_audit.get("bag_path"),
                "map_sha256": map_audit.get("SHA256"),
                "map_duration_s": map_audit.get("duration_s"),
                "map_lidar_hz": map_rates.get(LIDAR_TOPIC),
                "map_imu_hz": map_rates.get(IMU_TOPIC),
                "map_frame_id": ";".join(map_audit.get("frame_ids", [])),
                "map_motion_audit": map_audit.get("motion_audit_status"),
                "map_large_spike_count": map_audit.get("large_spike_count"),
                "query_bag_path": query_audit.get("bag_path"),
                "query_sha256": query_audit.get("SHA256"),
                "query_duration_s": query_audit.get("duration_s"),
                "query_lidar_hz": query_rates.get(LIDAR_TOPIC),
                "query_imu_hz": query_rates.get(IMU_TOPIC),
                "query_frame_id": ";".join(query_audit.get("frame_ids", [])),
                "query_motion_audit": query_audit.get("motion_audit_status"),
                "query_large_spike_count": query_audit.get("large_spike_count"),
                "actual_gap_s": pair_audit.get("actual_gap_s"),
                "map_query_no_overlap": pair_audit.get("map_query_no_overlap"),
                "formal_pair_valid": pair_audit.get("FORMAL_PAIR_VALID"),
                "exclusion_reason": ";".join(pair_audit.get("exclusion_reasons", [])),
            }
        )
    write_csv(output_path.parent / "acquisition_audit.csv", audit_rows, audit_fields)
    return payload


def _load_lidar_scans(path: Path) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    scans: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for frame_index, message, _, timestamp in iter_topic_messages(path, LIDAR_TOPIC):
        points = xyz_array(lidar_message_to_structured(message))
        finite_count = int(np.count_nonzero(np.all(np.isfinite(points), axis=1)))
        scans.append(points)
        rows.append(
            {
                "frame_index": int(frame_index),
                "timestamp": float(timestamp),
                "point_count": int(points.shape[0]),
                "finite_point_count": finite_count,
            }
        )
    if not scans:
        raise FormalBatchError(f"LiDAR topic is empty: {path}")
    return scans, rows


def _pair_bag_path(pair: Mapping[str, Any], role: str) -> Path:
    audit = pair[f"{role.lower()}_audit"]
    return Path(str(audit["bag_path"])).resolve(strict=True)


def assert_no_backend_output_before_snapshot_freeze(results_dir: Path) -> None:
    forbidden = (
        "open3d_results.csv",
        "pcl_results.csv",
        "formal_capture_basin_runs.csv",
        "backend_execution_started.json",
    )
    present = [name for name in forbidden if (results_dir / name).exists()]
    if present:
        raise FormalBatchError(
            f"backend outputs exist before snapshot freeze: {sorted(present)}"
        )


def build_geometry_assets(
    *,
    bag_manifest_path: Path,
    results_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build frozen targets/snapshots and registration-blind geometry manifests."""

    assert_no_backend_output_before_snapshot_freeze(results_dir)
    if (results_dir / "formal_batch1_lock.json").exists():
        raise FormalBatchError("geometry assets cannot change after formal batch lock")
    bag_manifest = json.loads(bag_manifest_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in bag_manifest.get("selected_pairs", []):
        if pair.get("pair_audit", {}).get("FORMAL_PAIR_VALID"):
            grouped[str(pair["scene_id"])].append(pair)

    target_path_manifest = results_dir / "target_manifest.json"
    snapshot_path_manifest = results_dir / "snapshot_manifest.json"
    geometry_path_manifest = results_dir / "geometry_manifest.json"
    prior_targets = (
        _load_required_json(target_path_manifest) if target_path_manifest.exists() else {}
    )
    prior_snapshots = (
        _load_required_json(snapshot_path_manifest) if snapshot_path_manifest.exists() else {}
    )
    prior_geometry = (
        _load_required_json(geometry_path_manifest) if geometry_path_manifest.exists() else {}
    )
    target_rows: list[dict[str, Any]] = list(prior_targets.get("targets", []))
    snapshot_rows: list[dict[str, Any]] = list(prior_snapshots.get("snapshots", []))
    geometry_rows: list[dict[str, Any]] = list(prior_geometry.get("snapshot_metrics", []))
    station_summaries: list[dict[str, Any]] = list(
        prior_geometry.get("station_summaries", [])
    )
    prior_scene_rows: list[dict[str, Any]] = list(prior_geometry.get("scenes", []))
    already_processed = {str(row["scene_id"]) for row in prior_scene_rows}
    new_scene_ids: list[str] = []
    for scene_id, pairs in sorted(grouped.items()):
        station_set = {str(pair["station_id"]) for pair in pairs}
        if len(pairs) < 3:
            continue
        if station_set != set(STATION_IDS) or len(pairs) != 3:
            raise FormalBatchError(
                f"geometry precheck requires exactly three passing stations for {scene_id}"
            )
        if scene_id in already_processed:
            continue
        new_scene_ids.append(scene_id)
        for pair in sorted(pairs, key=lambda row: str(row["station_id"])):
            station_id = str(pair["station_id"])
            map_path = _pair_bag_path(pair, "map")
            query_path = _pair_bag_path(pair, "query")
            map_scans, _ = _load_lidar_scans(map_path)
            target, target_metadata = build_static_target_map(map_scans, config)
            target = np.ascontiguousarray(target, dtype="<f8")
            target_dir = results_dir / "targets" / scene_id / station_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / "target_points.npy"
            if target_path.exists():
                raise FormalBatchError(f"refusing to overwrite target: {target_path}")
            np.save(target_path, target, allow_pickle=False)
            target_row = {
                "scene_id": scene_id,
                "station_id": station_id,
                "map_bag_path": str(map_path),
                "map_bag_sha256": pair["map_audit"]["SHA256"],
                "target_path": str(target_path),
                "target_npy_sha256": sha256_file(target_path),
                "target_point_count": int(target.shape[0]),
                "construction": "DIRECT_SAME_SENSOR_FRAME_MERGE_NO_REGISTRATION",
                "registration_called": False,
                "odometry_called": False,
                "scan_matching_called": False,
                "minimum_range_m": float(config["map"]["minimum_range_m"]),
                "maximum_range_m": float(config["map"]["maximum_range_m"]),
                "voxel_size_m": float(config["map"]["voxel_size_m"]),
                "map_frame_count": int(target_metadata["map_scan_count"]),
                "raw_point_count": int(target_metadata["raw_point_count"]),
                "finite_range_filtered_point_count": int(
                    target_metadata["finite_range_filtered_point_count"]
                ),
                "FORMAL_MEASUREMENT_RESULT": False,
            }
            target_rows.append(target_row)

            query_scans, query_rows = _load_lidar_scans(query_path)
            candidates = [
                row
                for row in query_rows
                if int(row["point_count"]) > 0 and int(row["finite_point_count"]) > 0
            ]
            selected = select_query_frames(candidates, QUERY_QUANTILES)
            query_by_index = {
                int(row["frame_index"]): query_scans[index]
                for index, row in enumerate(query_rows)
            }
            selected_points: list[np.ndarray] = []
            snapshot_metadata: list[dict[str, Any]] = []
            for item in selected:
                frame_index = int(item["frame_index"])
                accepted = finite_range_filter(
                    query_by_index[frame_index],
                    minimum_range_m=float(config["map"]["minimum_range_m"]),
                    maximum_range_m=float(config["map"]["maximum_range_m"]),
                )
                snapshot_id = (
                    f"{scene_id}_{station_id}_Q{int(item['selection_index']) + 1:02d}"
                )
                snapshot_dir = results_dir / "snapshots" / scene_id / station_id
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = snapshot_dir / f"{snapshot_id}.npy"
                if snapshot_path.exists():
                    raise FormalBatchError(
                        f"refusing to overwrite snapshot: {snapshot_path}"
                    )
                canonical = np.ascontiguousarray(accepted, dtype="<f8")
                np.save(snapshot_path, canonical, allow_pickle=False)
                snapshot_row = {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "snapshot_id": snapshot_id,
                    "selection_index": int(item["selection_index"]),
                    "quantile": float(item["quantile"]),
                    "target_sequence_rank": float(item["target_sequence_rank"]),
                    "selected_sequence_rank": int(item["selected_sequence_rank"]),
                    "query_frame_index": frame_index,
                    "query_timestamp": float(item["timestamp"]),
                    "query_bag_path": str(query_path),
                    "query_bag_sha256": pair["query_audit"]["SHA256"],
                    "source_path": str(snapshot_path),
                    "source_npy_sha256": sha256_file(snapshot_path),
                    "source_point_count": int(canonical.shape[0]),
                    "target_npy_sha256": target_row["target_npy_sha256"],
                    "selection_frozen_before_icp": True,
                }
                snapshot_rows.append(snapshot_row)
                snapshot_metadata.append(dict(item))
                selected_points.append(canonical)

            metric_rows, geometry_summary = compute_pilot_geometry(
                selected_points, snapshot_metadata, target, config
            )
            if len(metric_rows) != 10:
                raise FormalBatchError("geometry implementation did not return ten rows")
            for snapshot_row, metric_row in zip(snapshot_rows[-10:], metric_rows):
                geometry_rows.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "snapshot_id": snapshot_row["snapshot_id"],
                        **metric_row,
                    }
                )
            station_summaries.append(
                {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "snapshot_count": 10,
                    "median": geometry_summary["median"],
                    "geometry_only": True,
                    "registration_executed": False,
                }
            )

    if not new_scene_ids and not prior_scene_rows:
        raise FormalBatchError(
            "no candidate scene has exactly three acquisition-audit-passing stations"
        )
    scene_rows: list[dict[str, Any]] = list(prior_scene_rows)
    for scene_id in sorted(new_scene_ids):
        rows = [row for row in geometry_rows if row["scene_id"] == scene_id]
        if len(rows) != 30:
            raise FormalBatchError(f"scene {scene_id} must have exactly 30 geometry rows")
        lambda_median = float(
            np.median([float(row["normalized_lambda_min_trans"]) for row in rows])
        )
        condition_median = float(
            np.median([float(row["condition_number_trans"]) for row in rows])
        )
        entropy_median = float(
            np.median([float(row["spectral_entropy_trans"]) for row in rows])
        )
        scene_rows.append(
            {
                "scene_id": scene_id,
                "aggregation": "MEDIAN_OVER_30_FROZEN_STATION_SNAPSHOTS",
                **geometry_admission(lambda_median, condition_median, entropy_median),
            }
        )

    target_manifest = {
        "schema": "mid360_formal_batch1_target_manifest_v1",
        "targets": target_rows,
        "target_count": len(target_rows),
        "registration_executed": False,
    }
    snapshot_manifest = {
        "schema": "mid360_formal_batch1_snapshot_manifest_v1",
        "selection_method": "FIXED_QUANTILE_SEQUENCE_RANK_NEAREST_UNUSED_EARLIER_TIE",
        "quantiles": list(QUERY_QUANTILES),
        "snapshots": snapshot_rows,
        "snapshot_count": len(snapshot_rows),
        "backend_invocation_count_at_freeze": 0,
        "selection_frozen_before_icp": True,
    }
    geometry_manifest = {
        "schema": "mid360_formal_batch1_geometry_manifest_v1",
        "geometry_only": True,
        "registration_executed": False,
        "labels_frozen": True,
        "station_summaries": station_summaries,
        "snapshot_metrics": geometry_rows,
        "scenes": scene_rows,
    }
    write_json(results_dir / "target_manifest.json", target_manifest)
    write_json(results_dir / "snapshot_manifest.json", snapshot_manifest)
    write_json(results_dir / "geometry_manifest.json", geometry_manifest)
    admission_fields = [
        "scene_id", "candidate_label", "geometry_lambda_min_median",
        "geometry_condition_median", "geometry_entropy_median",
        "final_geometry_class", "admitted", "exclusion_reason",
    ]
    admission_rows = [
        {
            **row,
            "candidate_label": (
                "RICH_CANDIDATE" if str(row["scene_id"]).startswith("FMB1_R")
                else "WEAK_CANDIDATE"
            ),
        }
        for row in scene_rows
    ]
    write_csv(results_dir / "geometry_admission.csv", admission_rows, admission_fields)
    candidate_fields = [
        "scene_id", "candidate_label", "candidate_status",
        "geometry_lambda_min_median", "geometry_condition_median",
        "geometry_entropy_median", "final_geometry_class", "admitted",
        "exclusion_reason", "replacement_scene_id",
    ]
    candidate_rows = [
        {
            **row,
            "candidate_label": (
                "RICH_CANDIDATE" if str(row["scene_id"]).startswith("FMB1_R")
                else "WEAK_CANDIDATE"
            ),
            "candidate_status": "ADMITTED" if row["admitted"] else "EXCLUDED",
            "replacement_scene_id": None,
        }
        for row in scene_rows
    ]
    write_csv(results_dir / "candidate_scene_table.csv", candidate_rows, candidate_fields)
    return target_manifest, snapshot_manifest, geometry_manifest


def directory_manifest(directory: Path) -> dict[str, Any]:
    root = directory.resolve(strict=True)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"directory": str(root), "files": files, "file_count": len(files)}


def verify_directory_manifest(directory: Path, anchor: Mapping[str, Any]) -> None:
    current = directory_manifest(directory)
    expected_rows = {
        str(row["path"]): (int(row["size_bytes"]), str(row["sha256"]))
        for row in anchor.get("files", [])
    }
    current_rows = {
        str(row["path"]): (int(row["size_bytes"]), str(row["sha256"]))
        for row in current["files"]
    }
    if current_rows != expected_rows:
        raise FormalBatchError(f"frozen Pilot directory changed: {directory}")


def verify_pilot_anchors(repository: Path, anchors_path: Path) -> None:
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    for entry in anchors["pilot_directories"]:
        verify_directory_manifest(repository / entry["relative_path"], entry)


def verify_backend_contract(repository: Path) -> str:
    path = repository / "frozen_assets/backend_parameter_contract.json"
    actual = sha256_file(path)
    if actual != BACKEND_CONTRACT_SHA256:
        raise FormalBatchError(
            f"backend contract mismatch: expected {BACKEND_CONTRACT_SHA256}, got {actual}"
        )
    return actual


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FormalBatchError(f"required freeze input is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FormalBatchError(f"freeze input must be an object: {path}")
    return payload


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["scene_id"]), str(row["station_id"])


def validate_freeze_inputs(repository: Path, results_dir: Path) -> dict[str, Any]:
    experiment = repository / "experiments/mid360_formal_batch1"
    prereg_path = experiment / "preregistration.yaml"
    anchor_path = results_dir / "preregistration_anchor.json"
    anchor = _load_required_json(anchor_path)
    prereg_sha = sha256_file(prereg_path)
    if prereg_sha != anchor.get("PREREGISTRATION_SHA256"):
        raise FormalBatchError("preregistration SHA changed after preacquisition anchor")
    verify_backend_contract(repository)
    verify_pilot_anchors(repository, results_dir / "pilot_freeze_anchors.json")

    scene_registry = read_json_yaml(experiment / "scene_registry.yaml")
    station_registry = read_json_yaml(experiment / "station_registry.yaml")
    bag_manifest = _load_required_json(results_dir / "bag_manifest.json")
    target_manifest = _load_required_json(results_dir / "target_manifest.json")
    snapshot_manifest = _load_required_json(results_dir / "snapshot_manifest.json")
    geometry_manifest = _load_required_json(results_dir / "geometry_manifest.json")

    if not geometry_manifest.get("labels_frozen"):
        raise FormalBatchError("geometry labels are not frozen")
    if geometry_manifest.get("registration_executed") is not False:
        raise FormalBatchError("geometry admission is contaminated by registration")
    scene_geometry = {
        str(row["scene_id"]): row for row in geometry_manifest.get("scenes", [])
    }
    admitted = {
        scene_id: row for scene_id, row in scene_geometry.items() if row.get("admitted")
    }
    class_counts = Counter(str(row.get("final_geometry_class")) for row in admitted.values())
    if len(admitted) != 6 or class_counts != Counter({"RICH": 3, "WEAK": 3}):
        raise FormalBatchError("freeze requires exactly 3 admitted RICH and 3 admitted WEAK scenes")
    admitted_ids = set(admitted)

    registry_scenes = {str(row["scene_id"]): row for row in scene_registry.get("scenes", [])}
    if not admitted_ids.issubset(registry_scenes):
        raise FormalBatchError("admitted scene is missing from scene registry")
    for scene_id, geometry in admitted.items():
        if registry_scenes[scene_id].get("final_geometry_class") != geometry.get(
            "final_geometry_class"
        ):
            raise FormalBatchError(f"scene registry geometry label is not frozen: {scene_id}")

    station_geometry = [
        row
        for row in geometry_manifest.get("station_summaries", [])
        if str(row.get("scene_id")) in admitted_ids
    ]

    station_rows = [
        row for row in station_registry.get("stations", []) if str(row["scene_id"]) in admitted_ids
    ]
    station_keys = {_pair_key(row) for row in station_rows}
    expected_keys = {(scene, station) for scene in admitted_ids for station in STATION_IDS}
    if station_keys != expected_keys or len(station_rows) != 18:
        raise FormalBatchError("each admitted scene must have exactly S01/S02/S03 in station registry")

    if {_pair_key(row) for row in station_geometry} != expected_keys or len(station_geometry) != 18:
        raise FormalBatchError("geometry manifest must contain all 18 station summaries")
    if any(
        int(row.get("snapshot_count", 0)) != 10
        or row.get("geometry_only") is not True
        or row.get("registration_executed") is not False
        for row in station_geometry
    ):
        raise FormalBatchError("station geometry summary failed geometry-only/count checks")

    snapshot_geometry = [
        row
        for row in geometry_manifest.get("snapshot_metrics", [])
        if str(row.get("scene_id")) in admitted_ids
    ]
    geometry_counts = Counter(_pair_key(row) for row in snapshot_geometry)
    if (
        len(snapshot_geometry) != 180
        or set(geometry_counts) != expected_keys
        or set(geometry_counts.values()) != {10}
    ):
        raise FormalBatchError("geometry manifest must contain 180 frozen snapshot metrics")
    metric_fields = (
        "normalized_lambda_min_trans",
        "condition_number_trans",
        "spectral_entropy_trans",
    )
    if any(
        not all(isinstance(row.get(field), (int, float)) for field in metric_fields)
        for row in snapshot_geometry
    ):
        raise FormalBatchError("geometry snapshot metric is missing")
    for scene_id, scene_row in admitted.items():
        rows = [row for row in snapshot_geometry if str(row["scene_id"]) == scene_id]
        medians = tuple(
            float(np.median([float(row[field]) for row in rows]))
            for field in metric_fields
        )
        recomputed = geometry_class(*medians)
        if recomputed != scene_row.get("final_geometry_class"):
            raise FormalBatchError(f"geometry label does not recompute: {scene_id}")
        stored = (
            scene_row.get("geometry_lambda_min_median"),
            scene_row.get("geometry_condition_median"),
            scene_row.get("geometry_entropy_median"),
        )
        if any(
            not isinstance(value, (int, float))
            or not np.isclose(float(value), median, rtol=0.0, atol=1e-12)
            for value, median in zip(stored, medians)
        ):
            raise FormalBatchError(f"geometry scene medians do not recompute: {scene_id}")

    selected_pairs = [
        row
        for row in bag_manifest.get("selected_pairs", [])
        if str(row.get("scene_id")) in admitted_ids
    ]
    pair_keys = {_pair_key(row) for row in selected_pairs}
    if pair_keys != expected_keys or len(selected_pairs) != 18:
        raise FormalBatchError("each admitted scene must have exactly three valid selected pairs")
    for row in selected_pairs:
        if not row.get("pair_audit", {}).get("FORMAL_PAIR_VALID"):
            raise FormalBatchError(f"selected pair is invalid: {_pair_key(row)}")
        for role in ("map", "query"):
            if not _valid_sha(row.get(f"{role}_audit", {}).get("SHA256")):
                raise FormalBatchError(f"pair SHA is missing: {_pair_key(row)} {role}")

    targets = [
        row
        for row in target_manifest.get("targets", [])
        if str(row.get("scene_id")) in admitted_ids
    ]
    if {_pair_key(row) for row in targets} != expected_keys or len(targets) != 18:
        raise FormalBatchError("target manifest must contain one target per admitted station")
    if any(
        not _valid_sha(row.get("target_npy_sha256"))
        or int(row.get("target_point_count", 0)) <= 0
        or row.get("registration_called") is not False
        for row in targets
    ):
        raise FormalBatchError("target SHA/count/construction firewall failed")

    snapshots = [
        row
        for row in snapshot_manifest.get("snapshots", [])
        if str(row.get("scene_id")) in admitted_ids
    ]
    snapshot_counts = Counter(_pair_key(row) for row in snapshots)
    snapshot_ids = {str(row.get("snapshot_id")) for row in snapshots}
    geometry_snapshot_ids = {str(row.get("snapshot_id")) for row in snapshot_geometry}
    if (
        len(snapshots) != 180
        or len(snapshot_ids) != 180
        or set(snapshot_counts) != expected_keys
        or set(snapshot_counts.values()) != {10}
        or geometry_snapshot_ids != snapshot_ids
    ):
        raise FormalBatchError("snapshot manifest must contain exactly 10 per admitted station (180 total)")
    if snapshot_manifest.get("backend_invocation_count_at_freeze") != 0 or not snapshot_manifest.get(
        "selection_frozen_before_icp"
    ):
        raise FormalBatchError("snapshots were not frozen before ICP")
    if any(not _valid_sha(row.get("source_npy_sha256")) for row in snapshots):
        raise FormalBatchError("snapshot SHA is missing")

    bindings = {
        "preregistration": prereg_sha,
        "scene_registry": sha256_file(experiment / "scene_registry.yaml"),
        "station_registry": sha256_file(experiment / "station_registry.yaml"),
        "bag_manifest": sha256_file(results_dir / "bag_manifest.json"),
        "target_manifest": sha256_file(results_dir / "target_manifest.json"),
        "snapshot_manifest": sha256_file(results_dir / "snapshot_manifest.json"),
        "geometry_manifest": sha256_file(results_dir / "geometry_manifest.json"),
        "backend_parameter_contract": BACKEND_CONTRACT_SHA256,
        "pilot_freeze_anchors": sha256_file(results_dir / "pilot_freeze_anchors.json"),
    }
    return {
        "bindings": bindings,
        "admitted_scene_ids": sorted(admitted_ids),
        "admitted_class_counts": dict(sorted(class_counts.items())),
        "valid_station_count": len(pair_keys),
        "snapshot_count": len(snapshots),
    }


def freeze_batch(repository: Path, results_dir: Path) -> dict[str, Any]:
    lock_path = results_dir / "formal_batch1_lock.json"
    fingerprint_path = results_dir / "formal_batch1_fingerprint.json"
    ensure_paths_do_not_exist((lock_path, fingerprint_path))
    validated = validate_freeze_inputs(repository, results_dir)
    lock = {
        "schema": "mid360_formal_batch1_lock_v1",
        "FORMAL_BATCH_ID": BATCH_ID,
        "FORMAL_BATCH1_FROZEN": True,
        "FORMAL_ICP_UNLOCKED": True,
        "FORMAL_MEASUREMENT_RESULT": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **validated,
    }
    write_json(lock_path, lock, overwrite=False)
    lock_sha = sha256_file(lock_path)
    fingerprint = {
        "schema": "mid360_formal_batch1_fingerprint_v1",
        "PREREGISTRATION_SHA256": validated["bindings"]["preregistration"],
        "FORMAL_BATCH1_LOCK_SHA256": lock_sha,
        "FORMAL_MEASUREMENT_RESULT": False,
    }
    write_json(fingerprint_path, fingerprint, overwrite=False)
    return fingerprint


def formal_icp_authorized(repository: Path, results_dir: Path) -> bool:
    """Future formal runners must call this and fail closed on any drift."""

    lock_path = results_dir / "formal_batch1_lock.json"
    fingerprint_path = results_dir / "formal_batch1_fingerprint.json"
    if not lock_path.is_file() or not fingerprint_path.is_file():
        return False
    try:
        lock = _load_required_json(lock_path)
        fingerprint = _load_required_json(fingerprint_path)
        if not lock.get("FORMAL_BATCH1_FROZEN") or not lock.get("FORMAL_ICP_UNLOCKED"):
            return False
        if sha256_file(lock_path) != fingerprint.get("FORMAL_BATCH1_LOCK_SHA256"):
            return False
        validated = validate_freeze_inputs(repository, results_dir)
        return (
            validated["bindings"] == lock.get("bindings")
            and validated["bindings"]["preregistration"]
            == fingerprint.get("PREREGISTRATION_SHA256")
        )
    except (FormalBatchError, OSError, ValueError, KeyError, TypeError):
        return False


def require_formal_icp_authorization(repository: Path, results_dir: Path) -> None:
    if not formal_icp_authorized(repository, results_dir):
        raise FormalBatchError("FORMAL_ICP_BLOCKED: valid FMB1 lock is absent or stale")


__all__ = [name for name in globals() if not name.startswith("_")]
