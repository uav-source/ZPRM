"""Prepare the independent two-scene Mid-360 pilot without running registration."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phase_a_harness.common_association_analysis import (
    MAX_ASSOCIATION_DISTANCE_M,
    PCA_MIN_NEIGHBOR_COUNT,
    PCA_NEIGHBOR_COUNT,
)
from phase_a_harness.mid360_pilot.bag_reader import (
    IMU_TOPIC,
    LIDAR_TOPIC,
    PilotBagError,
    build_bag_inventory,
    iter_topic_messages,
    sha256_file,
)
from phase_a_harness.mid360_pilot.debug_registration import (
    IDENTITY,
    _parameter_sha,
    array_sha256,
    canonical_points,
    load_canonical_npy,
)
from phase_a_harness.mid360_pilot.imu_audit import audit_imu_messages
from phase_a_harness.mid360_pilot.lidar_adapter import (
    audit_lidar_messages,
    lidar_message_to_structured,
    xyz_array,
)
from phase_a_harness.mid360_pilot.pilot_geometry import compute_pilot_geometry
from phase_a_harness.mid360_pilot.pilot_pipeline import (
    DEFAULT_BACKEND_PARAMETER_CONTRACT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_PCL_EXECUTABLE,
    load_config,
    validate_runtime_root,
)
from phase_a_harness.mid360_pilot.split import select_query_frames
from phase_a_harness.mid360_pilot.static_map import (
    build_static_target_map,
    finite_range_filter,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    GEOMETRY_ONLY_FIELDS,
)

from . import TWO_SCENE_FLAGS


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_ROOT = (
    Path.home()
    / "zero_perturbation_runtime"
    / "real_data"
    / "mid360_two_scene_pilot_v1"
)
DEFAULT_FROZEN_ROOT = REPOSITORY_ROOT / "frozen_assets" / "mid360_two_scene_pilot_v1"
BASE_PROTECTED_COMMIT = "c69f4828844a27e0c6e5d688f881add4bf08c731"
SCENE_SPECS = (
    {
        "scene_id": "R_TEST_01",
        "semantic_scene": "LABORATORY",
        "scene_type": "RICH_CANDIDATE",
        "map_name": "mid360_20260818_203200_part1_20s.bag",
        "query_name": "mid360_20260818_203200_part2_15s.bag",
    },
    {
        "scene_id": "W_TEST_01",
        "semantic_scene": "LONG_CORRIDOR",
        "scene_type": "WEAK_CANDIDATE",
        "map_name": "mid360_20260818_205021_part1_20s.bag",
        "query_name": "mid360_20260818_205021_part2_15s.bag",
    },
)
REQUIRED_BAG_SHA256 = {
    "mid360_20260818_203200_part1_20s.bag": "ec998bc44cd7f6276548cbf7c11cffdfcb7a5a5601335cf2fdddd3c8da998c13",
    "mid360_20260818_203200_part2_15s.bag": "3a96dba7563f9742e8fabe5cbc7a90d77991e1d6d25dc024c1c36073af116e73",
    "mid360_20260818_205021_part1_20s.bag": "d20b50bd4be04229d311717a27021502a74429fa3a143418b18ad70622eb4a6d",
    "mid360_20260818_205021_part2_15s.bag": "2c9893beb56146996eccbe846dc0ee5a4a3ca9aba5b3bb71f7067c598b27c581",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=json_native,
        )
        + "\n",
        encoding="utf-8",
    )


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    return value


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotBagError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
            "q95": None,
        }
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _topic(inventory: Mapping[str, Any], topic: str) -> Mapping[str, Any]:
    return next(row for row in inventory["topics"] if row["topic"] == topic)


def _source_code_sha(relative_path: str) -> str:
    return sha256_file(REPOSITORY_ROOT / relative_path)


def _protected_tracked_files() -> dict[str, str]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "frozen_assets",
            "protocols",
            "configs/zero_perturbation",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output: dict[str, str] = {}
    for relative in completed.stdout.splitlines():
        if relative.startswith("frozen_assets/mid360_two_scene_pilot_v1/"):
            continue
        path = REPOSITORY_ROOT / relative
        output[relative] = sha256_file(path)
    return output


def build_preprocessing_contract(
    config_path: Path,
    parameter_contract_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    backend = read_json(parameter_contract_path)
    for name in ("open3d", "pcl"):
        expected = backend[name]["canonical_sha256"]
        actual = _parameter_sha(backend[name]["parameters"])
        if actual != expected:
            raise PilotBagError(f"frozen {name} parameter contract SHA mismatch")
    map_config = config["map"]
    if float(map_config["voxel_size_m"]) != 0.05:
        raise PilotBagError("prior Mid-360 target voxel contract changed")
    open_distance = float(
        backend["open3d"]["parameters"]["maximum_correspondence_distance_m"]
    )
    pcl_distance = float(
        backend["pcl"]["parameters"]["icp"]["maximum_correspondence_distance_m"]
    )
    if open_distance != pcl_distance or open_distance != MAX_ASSOCIATION_DISTANCE_M:
        raise PilotBagError("backend/common association distance contracts disagree")
    return {
        "schema": "mid360_two_scene_preprocessing_contract_v1",
        **TWO_SCENE_FLAGS,
        "status": "PASS",
        "source_single_bag_config_path": str(config_path.resolve()),
        "source_single_bag_config_sha256": sha256_file(config_path),
        "source_config_scope": "PREPROCESSING_ONLY_LEGACY_EXPERIMENT_FLAGS_NOT_REUSED",
        "finite_filtering": "ALL_XYZ_FINITE",
        "minimum_range_m": float(map_config["minimum_range_m"]),
        "maximum_range_m": float(map_config["maximum_range_m"]),
        "point_dtype": "little_endian_float64",
        "point_ordering": "DETERMINISTIC_SERIALIZED_ORDER_AFTER_STABLE_FILTER",
        "per_point_timestamp_order_assumed": False,
        "target_map_construction": map_config["construction"],
        "target_voxel_size_m": float(map_config["voxel_size_m"]),
        "query_quantiles": list(config["query_quantiles"]),
        "common_association_distance_m": MAX_ASSOCIATION_DISTANCE_M,
        "common_target_normal_pca_k": PCA_NEIGHBOR_COUNT,
        "common_target_normal_pca_min_neighbors": PCA_MIN_NEIGHBOR_COUNT,
        "open3d_target_normal_estimation": backend["open3d"]["parameters"][
            "target_normal_estimation"
        ],
        "pcl_normal_estimation": backend["pcl"]["parameters"]["normal_estimation"],
        "scene_specific_parameters": False,
        "backend_parameter_contract_path": str(parameter_contract_path.resolve()),
        "backend_parameter_contract_sha256": sha256_file(parameter_contract_path),
        "backend_canonical_parameter_sha256": {
            name: backend[name]["canonical_sha256"] for name in ("open3d", "pcl")
        },
        "code_bindings": {
            relative: _source_code_sha(relative)
            for relative in (
                "src/phase_a_harness/mid360_pilot/lidar_adapter.py",
                "src/phase_a_harness/mid360_pilot/static_map.py",
                "src/phase_a_harness/mid360_pilot/split.py",
                "src/phase_a_harness/mid360_pilot/pilot_geometry.py",
                "src/phase_a_harness/common_association_analysis.py",
                "src/phase_a_harness/open3d_backend.py",
                "src/phase_a_harness/pcl_backend.py",
            )
        },
        "bag_preprocessing_python": sys.version.split()[0],
        "bag_preprocessing_runtime": "SYSTEM_ROS_NOETIC_PYTHON_READ_ONLY",
        "backend_runtime": "LOCKED_PYTHON_3_11_OPEN3D_PCL",
        "protected_base_commit": BASE_PROTECTED_COMMIT,
        "protected_tracked_file_sha256": _protected_tracked_files(),
    }


def _bag_paths(bag_root: Path) -> list[dict[str, Any]]:
    resolved_root = bag_root.expanduser().resolve(strict=True)
    output: list[dict[str, Any]] = []
    for scene in SCENE_SPECS:
        for role in ("MAP", "QUERY"):
            key = "map_name" if role == "MAP" else "query_name"
            path = (resolved_root / scene[key]).resolve(strict=True)
            if not path.is_file():
                raise PilotBagError(f"bag is not a regular file: {path}")
            actual_sha = sha256_file(path)
            if actual_sha != REQUIRED_BAG_SHA256[path.name]:
                raise PilotBagError(f"bag SHA authentication failed: {path}")
            output.append({**scene, "role": role, "path": path, "sha256": actual_sha})
    return output


def _audit_bags(
    bag_specs: Sequence[Mapping[str, Any]],
    runtime: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_rows: list[dict[str, Any]] = []
    lidar_rows_all: list[dict[str, Any]] = []
    imu_rows_all: list[dict[str, Any]] = []
    audits: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in bag_specs:
        path = Path(spec["path"])
        inventory = build_bag_inventory(path)
        lidar_rows, lidar_summary = audit_lidar_messages(
            iter_topic_messages(path, LIDAR_TOPIC)
        )
        imu_rows, imu_summary = audit_imu_messages(
            iter_topic_messages(path, IMU_TOPIC), config
        )
        inventory.update(TWO_SCENE_FLAGS)
        lidar_summary.update(TWO_SCENE_FLAGS)
        imu_summary.update(TWO_SCENE_FLAGS)
        lidar_topic = _topic(inventory, LIDAR_TOPIC)
        imu_topic = _topic(inventory, IMU_TOPIC)
        field_names = [field["name"] for field in lidar_topic["fields"]]
        if not {"x", "y", "z"}.issubset(field_names):
            raise PilotBagError(f"required XYZ fields missing: {path}")
        if lidar_summary["status"] != "PASS" or imu_summary["status"] != "PASS":
            raise PilotBagError(f"bag content audit failed: {path}")
        common = {
            "scene_id": spec["scene_id"],
            "semantic_scene": spec["semantic_scene"],
            "scene_type": spec["scene_type"],
            "role": spec["role"],
            "bag_path": str(path),
            "bag_sha256": inventory["bag_sha256"],
        }
        manifest_rows.append(
            {
                **common,
                "file_size_bytes": inventory["bag_size_bytes"],
                "mtime_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "start_time": inventory["start_timestamp"],
                "end_time": inventory["end_timestamp"],
                "duration_seconds": inventory["duration_seconds"],
                "lidar_message_type": lidar_topic["message_type"],
                "lidar_frame_count": lidar_topic["message_count"],
                "lidar_header_rate_hz": lidar_summary["effective_frame_rate_hz"],
                "lidar_frame_ids": lidar_topic["frame_ids"],
                "lidar_point_fields": field_names,
                "imu_message_type": imu_topic["message_type"],
                "imu_message_count": imu_topic["message_count"],
                "imu_header_rate_hz": imu_summary["effective_rate_hz"],
                "imu_frame_ids": imu_topic["frame_ids"],
                "topic_inventory": inventory["topics"],
                "point_coordinate_unit": "ASSUMED_METERS_FROM_SCALE",
                "acceleration_unit": "UNCONFIRMED_RAW_NO_CONVERSION",
                **TWO_SCENE_FLAGS,
            }
        )
        lidar_rows_all.extend(
            {
                **common,
                "message_type": lidar_topic["message_type"],
                "frame_ids": lidar_topic["frame_ids"],
                "point_fields": field_names,
                **row,
                **TWO_SCENE_FLAGS,
            }
            for row in lidar_rows
        )
        imu_rows_all.extend(
            {**common, **row, **TWO_SCENE_FLAGS} for row in imu_rows
        )
        audit_payload = {
            "schema": "mid360_two_scene_bag_audit_v1",
            **TWO_SCENE_FLAGS,
            **common,
            "inventory": inventory,
            "lidar_summary": lidar_summary,
            "imu_summary": imu_summary,
            "STATICITY_SCREEN": imu_summary["STATICITY_SCREEN"],
            "point_coordinate_unit": "ASSUMED_METERS_FROM_SCALE",
            "acceleration_unit_confirmed": False,
            "acceleration_conversion_applied": False,
            "status": "PASS",
        }
        audit_name = f"{spec['scene_id']}_{str(spec['role']).lower()}.json"
        write_json(runtime / "bag_audit" / audit_name, audit_payload)
        audits[(str(spec["scene_id"]), str(spec["role"]))] = audit_payload
    write_csv(runtime / "input_bag_manifest.csv", manifest_rows)
    write_json(
        runtime / "input_bag_manifest.json",
        {
            "schema": "mid360_two_scene_input_bag_manifest_v1",
            **TWO_SCENE_FLAGS,
            "bag_count": len(manifest_rows),
            "bags": manifest_rows,
            "status": "PASS",
        },
    )
    write_csv(runtime / "lidar_frame_audit.csv", lidar_rows_all)
    write_csv(runtime / "imu_staticity_audit.csv", imu_rows_all)
    scene_comparisons: dict[str, Any] = {}
    for scene in SCENE_SPECS:
        scene_id = str(scene["scene_id"])
        map_summary = audits[(scene_id, "MAP")]["imu_summary"]
        query_summary = audits[(scene_id, "QUERY")]["imu_summary"]
        scene_comparisons[scene_id] = {
            "map_gyro_norm_median": map_summary["statistics"]["gyro_norm"]["median"],
            "query_gyro_norm_median": query_summary["statistics"]["gyro_norm"]["median"],
            "map_gyro_norm_q95": map_summary["statistics"]["gyro_norm"]["q95"],
            "query_gyro_norm_q95": query_summary["statistics"]["gyro_norm"]["q95"],
            "map_accel_norm_median_raw": map_summary["statistics"]["accel_norm"]["median"],
            "query_accel_norm_median_raw": query_summary["statistics"]["accel_norm"]["median"],
            "STATICITY_SCREEN": (
                "NO_OBVIOUS_MOTION"
                if map_summary["STATICITY_SCREEN"] == query_summary["STATICITY_SCREEN"] == "NO_OBVIOUS_MOTION"
                else "MOTION_SUSPECTED"
                if "MOTION_SUSPECTED"
                in {map_summary["STATICITY_SCREEN"], query_summary["STATICITY_SCREEN"]}
                else "INCONCLUSIVE"
            ),
        }
    blocked = any(
        payload["imu_summary"]["STATICITY_SCREEN"] != "NO_OBVIOUS_MOTION"
        for payload in audits.values()
    )
    staticity = {
        "schema": "mid360_two_scene_staticity_summary_v1",
        **TWO_SCENE_FLAGS,
        "bag_staticity": {
            f"{scene_id}_{role.lower()}": payload["imu_summary"]
            for (scene_id, role), payload in audits.items()
        },
        "scene_map_query_comparisons": scene_comparisons,
        "PILOT_REGISTRATION_BLOCKED_BY_STATICITY": blocked,
        "imu_interpretation": "OBVIOUS_MOTION_SCREEN_ONLY_NOT_SUBMILLIMETER_TRUTH",
        "imu_position_integration_used": False,
        "acceleration_unit_confirmed": False,
        "status": "FAIL" if blocked else "PASS",
    }
    write_json(runtime / "staticity_summary.json", staticity)
    return {
        "manifest_rows": manifest_rows,
        "audits": audits,
        "staticity": staticity,
    }


def _build_lineage(manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scene_role = {
        (str(row["scene_id"]), str(row["role"])): row for row in manifest_rows
    }
    scenes: dict[str, Any] = {}
    all_pass = True
    for scene in SCENE_SPECS:
        scene_id = str(scene["scene_id"])
        map_row = by_scene_role[(scene_id, "MAP")]
        query_row = by_scene_role[(scene_id, "QUERY")]
        gap = float(query_row["start_time"]) - float(map_row["end_time"])
        path_distinct = map_row["bag_path"] != query_row["bag_path"]
        sha_distinct = map_row["bag_sha256"] != query_row["bag_sha256"]
        nonoverlap = gap > 0.0
        passed = path_distinct and sha_distinct and nonoverlap
        all_pass = all_pass and passed
        scenes[scene_id] = {
            "semantic_scene": scene["semantic_scene"],
            "scene_type": scene["scene_type"],
            "map_source": scene["map_name"],
            "query_source": scene["query_name"],
            "map_path": map_row["bag_path"],
            "query_path": query_row["bag_path"],
            "map_sha256": map_row["bag_sha256"],
            "query_sha256": query_row["bag_sha256"],
            "map_start_time": map_row["start_time"],
            "map_end_time": map_row["end_time"],
            "query_start_time": query_row["start_time"],
            "query_end_time": query_row["end_time"],
            "map_to_query_gap_seconds": gap,
            "MAP_QUERY_GAP_SUSPICIOUS": gap < 3.0,
            "path_distinct": path_distinct,
            "sha256_distinct": sha_distinct,
            "recording_intervals_nonoverlapping": nonoverlap,
            "status": "PASS" if passed else "FAIL",
        }
    return {
        "schema": "mid360_two_scene_map_query_lineage_v1",
        **TWO_SCENE_FLAGS,
        "same_bag_map_query": False,
        "scene_count": 2,
        "scenes": scenes,
        "status": "PASS" if all_pass else "FAIL",
    }


def _load_all_lidar_xyz(path: Path) -> tuple[list[np.ndarray], list[float]]:
    scans: list[np.ndarray] = []
    timestamps: list[float] = []
    for _, message, _, timestamp in iter_topic_messages(path, LIDAR_TOPIC):
        scans.append(xyz_array(lidar_message_to_structured(message)))
        timestamps.append(float(timestamp))
    return scans, timestamps


def _build_targets(
    bag_specs: Sequence[Mapping[str, Any]],
    runtime: Path,
    config: Mapping[str, Any],
    config_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    targets: dict[str, np.ndarray] = {}
    for spec in bag_specs:
        if spec["role"] != "MAP":
            continue
        scene_id = str(spec["scene_id"])
        scans, timestamps = _load_all_lidar_xyz(Path(spec["path"]))
        target, metadata = build_static_target_map(scans, config)
        target = canonical_points(target)
        directory = runtime / "target_maps" / scene_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "target_points.npy"
        np.save(path, target, allow_pickle=False)
        reloaded = load_canonical_npy(path)
        file_sha = sha256_file(path)
        array_sha = array_sha256(reloaded)
        metadata.update(TWO_SCENE_FLAGS)
        metadata.update(
            {
                "schema": "mid360_two_scene_target_map_metadata_v1",
                "scene_id": scene_id,
                "semantic_scene": spec["semantic_scene"],
                "scene_type": spec["scene_type"],
                "map_bag_path": str(spec["path"]),
                "map_bag_sha256": spec["sha256"],
                "map_frame_timestamps": timestamps,
                "target_path": str(path),
                "target_npy_sha256": file_sha,
                "target_array_sha256": array_sha,
                "source_config_sha256": sha256_file(config_path),
                "registration_called": False,
                "scan_matching_called": False,
                "odometry_called": False,
            }
        )
        write_json(directory / "metadata.json", metadata)
        row = {
            "scene_id": scene_id,
            "semantic_scene": spec["semantic_scene"],
            "scene_type": spec["scene_type"],
            "map_bag_path": str(spec["path"]),
            "map_bag_sha256": spec["sha256"],
            "map_frame_count": metadata["map_scan_count"],
            "raw_point_count": metadata["raw_point_count"],
            "filtered_point_count": metadata["finite_range_filtered_point_count"],
            "voxelized_point_count": metadata["target_map_point_count"],
            "target_path": str(path),
            "target_npy_sha256": file_sha,
            "target_array_sha256": array_sha,
            "target_registration_called": False,
            **TWO_SCENE_FLAGS,
        }
        rows.append(row)
        targets[scene_id] = reloaded
    if len(rows) != 2:
        raise PilotBagError("exactly two target maps are required")
    write_csv(runtime / "target_map_manifest.csv", rows)
    return rows, targets


def _freeze_query_selection(
    bag_specs: Sequence[Mapping[str, Any]],
    audits: Mapping[tuple[str, str], Mapping[str, Any]],
    runtime: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forbidden_outputs = (runtime / "open3d_results.csv", runtime / "pcl_results.csv")
    if any(path.exists() for path in forbidden_outputs):
        raise PilotBagError("backend outputs predate query selection freeze")
    rows: list[dict[str, Any]] = []
    for spec in bag_specs:
        if spec["role"] != "QUERY":
            continue
        audit = audits[(str(spec["scene_id"]), "QUERY")]
        candidates = [
            row
            for row in audit["lidar_summary"].get("_unused_rows", [])
        ]
        if candidates:
            raise PilotBagError("unexpected embedded LiDAR rows")
        lidar_rows, _ = audit_lidar_messages(
            iter_topic_messages(Path(spec["path"]), LIDAR_TOPIC)
        )
        valid = [
            row
            for row in lidar_rows
            if int(row["point_count"]) > 0 and int(row["finite_point_count"]) > 0
        ]
        selected = select_query_frames(valid, config["query_quantiles"])
        for item in selected:
            selection_index = int(item["selection_index"])
            rows.append(
                {
                    "scene_id": spec["scene_id"],
                    "semantic_scene": spec["semantic_scene"],
                    "scene_type": spec["scene_type"],
                    "snapshot_id": f"{spec['scene_id']}_Q{selection_index + 1:02d}",
                    "selection_index": selection_index,
                    "quantile": float(item["quantile"]),
                    "query_candidate_count": len(valid),
                    "target_sequence_rank": float(item["target_sequence_rank"]),
                    "selected_sequence_rank": int(item["selected_sequence_rank"]),
                    "query_frame_index": int(item["frame_index"]),
                    "query_timestamp": float(item["timestamp"]),
                    "query_point_count": int(item["point_count"]),
                    "query_finite_point_count": int(item["finite_point_count"]),
                    "query_bag_path": str(spec["path"]),
                    "query_bag_sha256": spec["sha256"],
                    **TWO_SCENE_FLAGS,
                }
            )
    if len(rows) != 20 or len({row["snapshot_id"] for row in rows}) != 20:
        raise PilotBagError("query selection must freeze exactly 20 unique snapshots")
    frozen_at = utc_now()
    write_csv(runtime / "query_selection_frozen.csv", rows)
    payload = {
        "schema": "mid360_two_scene_query_selection_frozen_v1",
        **TWO_SCENE_FLAGS,
        "selection_method": "FIXED_QUANTILE_SEQUENCE_RANK_NEAREST_UNUSED_EARLIER_TIE",
        "quantiles": list(config["query_quantiles"]),
        "selection_frozen_at_utc": frozen_at,
        "backend_invocation_count_at_freeze": 0,
        "rich_snapshot_count": sum(row["scene_id"] == "R_TEST_01" for row in rows),
        "weak_snapshot_count": sum(row["scene_id"] == "W_TEST_01" for row in rows),
        "snapshots": rows,
        "status": "PASS",
    }
    write_json(runtime / "query_selection_frozen.json", payload)
    return rows, payload


def _export_canonical_sources(
    bag_specs: Sequence[Mapping[str, Any]],
    selection: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    runtime: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    target_by_scene = {str(row["scene_id"]): row for row in target_rows}
    selection_by_scene_frame = {
        (str(row["scene_id"]), int(row["query_frame_index"])): row for row in selection
    }
    points_by_snapshot: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    for spec in bag_specs:
        if spec["role"] != "QUERY":
            continue
        scene_id = str(spec["scene_id"])
        loaded: dict[int, tuple[np.ndarray, float]] = {}
        expected_indexes = {
            frame for sid, frame in selection_by_scene_frame if sid == scene_id
        }
        for frame_index, message, _, timestamp in iter_topic_messages(
            Path(spec["path"]), LIDAR_TOPIC
        ):
            if frame_index not in expected_indexes:
                continue
            xyz = xyz_array(lidar_message_to_structured(message))
            accepted = finite_range_filter(
                xyz,
                minimum_range_m=float(config["map"]["minimum_range_m"]),
                maximum_range_m=float(config["map"]["maximum_range_m"]),
            )
            loaded[frame_index] = (canonical_points(accepted), float(timestamp))
        if set(loaded) != expected_indexes:
            raise PilotBagError(f"failed to reload every frozen query for {scene_id}")
        for frame_index in sorted(expected_indexes):
            selection_row = selection_by_scene_frame[(scene_id, frame_index)]
            source, timestamp = loaded[frame_index]
            if timestamp != float(selection_row["query_timestamp"]):
                raise PilotBagError(f"frozen query timestamp drift: {selection_row['snapshot_id']}")
            snapshot_id = str(selection_row["snapshot_id"])
            directory = runtime / "canonical_sources" / snapshot_id
            directory.mkdir(parents=True, exist_ok=True)
            source_path = directory / "source_points.npy"
            np.save(source_path, source, allow_pickle=False)
            reloaded = load_canonical_npy(source_path)
            target_row = target_by_scene[scene_id]
            row = {
                "scene_id": scene_id,
                "semantic_scene": spec["semantic_scene"],
                "scene_type": spec["scene_type"],
                "snapshot_id": snapshot_id,
                "query_frame_index": frame_index,
                "query_timestamp": timestamp,
                "query_bag_path": str(spec["path"]),
                "query_bag_sha256": spec["sha256"],
                "source_path": str(source_path),
                "source_point_count": int(reloaded.shape[0]),
                "source_npy_sha256": sha256_file(source_path),
                "source_array_sha256": array_sha256(reloaded),
                "target_path": target_row["target_path"],
                "target_point_count": int(target_row["voxelized_point_count"]),
                "target_npy_sha256": target_row["target_npy_sha256"],
                "target_array_sha256": target_row["target_array_sha256"],
                "dtype": reloaded.dtype.str,
                "little_endian_float64": reloaded.dtype == np.dtype("<f8"),
                "c_contiguous": bool(reloaded.flags.c_contiguous),
                "finite": bool(np.all(np.isfinite(reloaded))),
                "deterministic_point_ordering": True,
                "T0": IDENTITY.tolist(),
                **TWO_SCENE_FLAGS,
            }
            write_json(
                directory / "metadata.json",
                {"schema": "mid360_two_scene_canonical_source_v1", **row},
            )
            manifest.append(row)
            points_by_snapshot[snapshot_id] = reloaded
    if len(manifest) != 20:
        raise PilotBagError("canonical source manifest must contain 20 snapshots")
    write_csv(runtime / "canonical_input_manifest.csv", manifest)
    return manifest, points_by_snapshot


def _geometry_only(
    selection: Sequence[Mapping[str, Any]],
    canonical_manifest: Sequence[Mapping[str, Any]],
    sources: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    runtime: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection_by_id = {str(row["snapshot_id"]): row for row in selection}
    geometry_rows: list[dict[str, Any]] = []
    scene_summaries: dict[str, Any] = {}
    forbidden_fragments = (
        "t_est",
        "translation",
        "rotation",
        "displacement",
        "final_residual",
        "turnover",
        "fitness",
        "solver",
    )
    for scene in SCENE_SPECS:
        scene_id = str(scene["scene_id"])
        rows = [row for row in canonical_manifest if row["scene_id"] == scene_id]
        query_points = [sources[str(row["snapshot_id"])] for row in rows]
        metadata = [
            {
                "frame_index": selection_by_id[str(row["snapshot_id"])]["query_frame_index"],
                "timestamp": selection_by_id[str(row["snapshot_id"])]["query_timestamp"],
            }
            for row in rows
        ]
        raw_rows, _ = compute_pilot_geometry(
            query_points, metadata, targets[scene_id], config
        )
        for manifest_row, raw in zip(rows, raw_rows):
            metric_payload = {field: raw[field] for field in GEOMETRY_ONLY_FIELDS}
            if any(
                fragment in key.lower()
                for key in metric_payload
                for fragment in forbidden_fragments
            ):
                raise PilotBagError("forbidden backend field escaped geometry-only firewall")
            geometry_rows.append(
                {
                    "scene_id": scene_id,
                    "semantic_scene": scene["semantic_scene"],
                    "scene_type": scene["scene_type"],
                    "snapshot_id": manifest_row["snapshot_id"],
                    "query_frame_index": manifest_row["query_frame_index"],
                    "query_timestamp": manifest_row["query_timestamp"],
                    "T0": IDENTITY.tolist(),
                    **metric_payload,
                    **TWO_SCENE_FLAGS,
                }
            )
        scene_summaries[scene_id] = {
            "semantic_scene": scene["semantic_scene"],
            "scene_type": scene["scene_type"],
            "snapshot_count": len(raw_rows),
            "metrics": {
                field: describe([float(row[field]) for row in raw_rows])
                for field in GEOMETRY_ONLY_FIELDS
            },
        }
    rich = scene_summaries["R_TEST_01"]["metrics"]
    weak = scene_summaries["W_TEST_01"]["metrics"]
    rich_lambda = float(rich["lambda_min_trans"]["median"])
    weak_lambda = float(weak["lambda_min_trans"]["median"])
    rich_condition = float(rich["condition_number_trans"]["median"])
    weak_condition = float(weak["condition_number_trans"]["median"])
    rich_entropy = float(rich["spectral_entropy_trans"]["median"])
    weak_entropy = float(weak["spectral_entropy_trans"]["median"])
    directional_checks = {
        "weak_lambda_min_lower_than_rich": weak_lambda < rich_lambda,
        "weak_condition_number_higher_than_rich": weak_condition > rich_condition,
        "weak_spectral_entropy_lower_than_rich": weak_entropy < rich_entropy,
    }
    weaker = all(directional_checks.values())
    summary = {
        "schema": "mid360_two_scene_geometry_scene_summary_v1",
        **TWO_SCENE_FLAGS,
        "geometry_only": True,
        "T0": "IDENTITY_4X4",
        "metric_fields": list(GEOMETRY_ONLY_FIELDS),
        "scene_summaries": scene_summaries,
        "weak_vs_rich": {
            "lambda_min_median_ratio": weak_lambda / rich_lambda,
            "condition_number_median_ratio": weak_condition / rich_condition,
            "spectral_entropy_median_difference": weak_entropy - rich_entropy,
            "directional_checks": directional_checks,
        },
        "W_TEST_01_GEOMETRICALLY_WEAKER_THAN_R_TEST_01": weaker,
        "PILOT_GEOMETRY_DIRECTIONALLY_CONSISTENT": weaker,
        "SEMANTIC_LABEL_GEOMETRY_MISMATCH": not weaker,
        "formal_threshold_added_posthoc": False,
        "status": "PASS",
    }
    write_csv(runtime / "geometry_only_metrics.csv", geometry_rows)
    write_json(runtime / "geometry_scene_summary.json", summary)
    return geometry_rows, summary


def _authorization(
    runtime: Path,
    canonical_manifest: Sequence[Mapping[str, Any]],
    parameter_contract_path: Path,
) -> dict[str, Any]:
    if (runtime / "open3d_results.csv").exists() or (runtime / "pcl_results.csv").exists():
        raise PilotBagError("cannot authorize after backend result creation")
    selection_csv_sha = sha256_file(runtime / "query_selection_frozen.csv")
    selection_json_sha = sha256_file(runtime / "query_selection_frozen.json")
    snapshots = [
        {
            "scene_id": row["scene_id"],
            "snapshot_id": row["snapshot_id"],
            "query_timestamp": row["query_timestamp"],
            "source_npy_sha256": row["source_npy_sha256"],
            "source_array_sha256": row["source_array_sha256"],
            "target_npy_sha256": row["target_npy_sha256"],
            "target_array_sha256": row["target_array_sha256"],
        }
        for row in canonical_manifest
    ]
    trials = [
        {
            "trial_id": f"{row['snapshot_id']}::{backend}",
            "snapshot_id": row["snapshot_id"],
            "scene_id": row["scene_id"],
            "backend": backend,
            "source_npy_sha256": row["source_npy_sha256"],
            "target_npy_sha256": row["target_npy_sha256"],
            "T0": IDENTITY.tolist(),
        }
        for row in canonical_manifest
        for backend in ("open3d_point_to_plane", "pcl_point_to_plane")
    ]
    if len(trials) != 40 or len({trial["trial_id"] for trial in trials}) != 40:
        raise PilotBagError("authorization must bind exactly 40 unique trials")
    payload = {
        "schema": "mid360_two_scene_debug_registration_authorization_v1",
        **TWO_SCENE_FLAGS,
        "PILOT_ONLY": True,
        "FORMAL_MEASUREMENT_RESULT": False,
        "MAX_ALLOWED_TRIALS": 40,
        "authorized_trial_count": 40,
        "authorized_snapshot_count": 20,
        "authorized_backends": ["open3d_point_to_plane", "pcl_point_to_plane"],
        "forbidden_backends": ["GICP", "NDT", "KISS_ICP", "FAST_LIO", "LIO_SAM", "LOAM"],
        "T0": IDENTITY.tolist(),
        "query_selection_frozen_before_backend": True,
        "query_selection_csv_sha256": selection_csv_sha,
        "query_selection_json_sha256": selection_json_sha,
        "geometry_only_metrics_sha256": sha256_file(runtime / "geometry_only_metrics.csv"),
        "geometry_scene_summary_sha256": sha256_file(
            runtime / "geometry_scene_summary.json"
        ),
        "canonical_input_manifest_sha256": sha256_file(
            runtime / "canonical_input_manifest.csv"
        ),
        "target_map_manifest_sha256": sha256_file(runtime / "target_map_manifest.csv"),
        "preprocessing_contract_sha256": sha256_file(
            runtime / "mid360_two_scene_preprocessing_contract.json"
        ),
        "backend_parameter_contract_path": str(parameter_contract_path.resolve()),
        "backend_parameter_contract_sha256": sha256_file(parameter_contract_path),
        "snapshots": snapshots,
        "authorized_trials": trials,
        "authorization_created_at_utc": utc_now(),
        "formal_multisite_experiment_authorized": False,
        "boreas_authorized": False,
        "status": "AUTHORIZED_PILOT_ONLY",
    }
    write_json(runtime / "mid360_two_scene_debug_registration_authorization.json", payload)
    return payload


def prepare_two_scene_pilot(
    bag_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    parameter_contract_path: Path = DEFAULT_BACKEND_PARAMETER_CONTRACT,
) -> dict[str, Any]:
    runtime = validate_runtime_root(runtime_root)
    if runtime.exists() and any(runtime.iterdir()):
        raise PilotBagError(f"runtime already contains artifacts; refusing overwrite: {runtime}")
    runtime.mkdir(parents=True, exist_ok=True)
    config_path = config_path.resolve(strict=True)
    parameter_contract_path = parameter_contract_path.resolve(strict=True)
    config = load_config(config_path)
    bag_specs = _bag_paths(bag_root)
    preprocessing = build_preprocessing_contract(config_path, parameter_contract_path)
    write_json(runtime / "mid360_two_scene_preprocessing_contract.json", preprocessing)
    audit = _audit_bags(bag_specs, runtime, config)
    lineage = _build_lineage(audit["manifest_rows"])
    write_json(runtime / "mid360_map_query_lineage.json", lineage)
    if lineage["status"] != "PASS":
        raise PilotBagError("independent Map/Query lineage authentication failed")
    if audit["staticity"]["PILOT_REGISTRATION_BLOCKED_BY_STATICITY"]:
        raise PilotBagError("PILOT_REGISTRATION_BLOCKED_BY_STATICITY=true")
    target_rows, targets = _build_targets(
        bag_specs, runtime, config, config_path
    )
    selection_rows, selection_payload = _freeze_query_selection(
        bag_specs, audit["audits"], runtime, config
    )
    canonical_rows, sources = _export_canonical_sources(
        bag_specs, selection_rows, target_rows, runtime, config
    )
    _, geometry_summary = _geometry_only(
        selection_rows,
        canonical_rows,
        sources,
        targets,
        runtime,
        config,
    )
    authorization = _authorization(runtime, canonical_rows, parameter_contract_path)
    return {
        "schema": "mid360_two_scene_prepare_result_v1",
        **TWO_SCENE_FLAGS,
        "status": "READY_FOR_EXACT_40_PILOT_TRIALS",
        "runtime_root": str(runtime),
        "bag_count": len(bag_specs),
        "target_map_count": len(target_rows),
        "snapshot_count": len(selection_rows),
        "rich_snapshot_count": selection_payload["rich_snapshot_count"],
        "weak_snapshot_count": selection_payload["weak_snapshot_count"],
        "geometry_weaker_direction": geometry_summary[
            "W_TEST_01_GEOMETRICALLY_WEAKER_THAN_R_TEST_01"
        ],
        "authorized_trial_count": authorization["authorized_trial_count"],
        "backend_execution_count": 0,
    }


__all__ = [
    "BASE_PROTECTED_COMMIT",
    "DEFAULT_FROZEN_ROOT",
    "DEFAULT_RUNTIME_ROOT",
    "REQUIRED_BAG_SHA256",
    "SCENE_SPECS",
    "build_preprocessing_contract",
    "describe",
    "prepare_two_scene_pilot",
    "read_csv",
    "read_json",
    "utc_now",
    "write_csv",
    "write_json",
]
