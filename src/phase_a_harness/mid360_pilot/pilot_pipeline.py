"""Artifact-producing Mid-360 single-bag PILOT_ONLY pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import PILOT_FLAGS
from .bag_reader import (
    IMU_TOPIC,
    LIDAR_TOPIC,
    PilotBagError,
    build_bag_inventory,
    inventory_markdown,
    iter_topic_messages,
    sha256_file,
)
from .imu_audit import audit_imu_messages
from .lidar_adapter import audit_lidar_messages, lidar_message_to_structured, xyz_array
from .pilot_geometry import compute_pilot_geometry
from .split import build_lineage, build_split_contract, partition_lidar_frames, select_query_frames
from .static_map import build_static_target_map, finite_range_filter


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "mid360_pilot_config.json"
DEFAULT_BACKEND_PARAMETER_CONTRACT = (
    REPOSITORY_ROOT / "frozen_assets" / "backend_parameter_contract.json"
)
DEFAULT_PCL_EXECUTABLE = REPOSITORY_ROOT / "bin" / "pcl_point_to_plane_cli"
FORBIDDEN_RUNTIME_COMPONENTS = frozenset(
    {
        "frozen_assets",
        "synthetic_confirmatory",
        "public_data_validation_v1",
        "public_data_external_validation_v2_boreas_stage1",
        "real_data_boreas_stage1_v1",
        "real_data_cavers_stage1_v1",
    }
)


def _json_native(value: Any) -> Any:
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
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_native,
        )
        + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None
) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("flags") != PILOT_FLAGS:
        raise PilotBagError("pilot config flags are missing or changed")
    return config


def validate_runtime_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if any(component in FORBIDDEN_RUNTIME_COMPONENTS for component in resolved.parts):
        raise PilotBagError(f"pilot output path targets a frozen/formal asset: {resolved}")
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        pass
    else:
        raise PilotBagError("pilot output cannot be written anywhere inside the repository")
    return resolved


def _topic_csv_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "topic": row["topic"],
            "message_type": row["message_type"],
            "message_count": row["message_count"],
            "connections": row["connections"],
            "average_rate_hz": row["average_bag_record_rate_hz"],
            "rosbag_reported_frequency_hz": row["rosbag_reported_frequency_hz"],
            "first_bag_timestamp": row["first_bag_timestamp"],
            "last_bag_timestamp": row["last_bag_timestamp"],
            "first_header_timestamp": row["first_header_timestamp"],
            "last_header_timestamp": row["last_header_timestamp"],
            "frame_ids": row["frame_ids"],
            "fields": row["fields"],
        }
        for row in inventory["topics"]
    ]


def _save_lidar_plots(runtime: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    relative = timestamps - timestamps[0]
    counts = np.asarray([int(row["point_count"]) for row in rows])
    intervals = np.asarray(
        [float(row["inter_frame_interval_seconds"]) for row in rows[1:]]
    )
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(relative, counts, linewidth=1.2)
    axis.set(xlabel="Time from first LiDAR frame (s)", ylabel="Point count", title="PILOT_ONLY LiDAR point count")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(runtime / "lidar_point_count_vs_time.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(relative[1:], intervals, linewidth=1.2)
    axis.set(xlabel="Time from first LiDAR frame (s)", ylabel="Frame interval (s)", title="PILOT_ONLY LiDAR frame interval")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(runtime / "lidar_frame_interval_vs_time.png", dpi=150)
    plt.close(fig)


def _save_imu_plots(runtime: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    relative = timestamps - timestamps[0]
    for prefix, output, ylabel in (
        ("gyro", "imu_gyro_vs_time.png", "Angular velocity (message value)"),
        ("accel", "imu_accel_vs_time.png", "Linear acceleration (raw, unit unconfirmed)"),
    ):
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for component in ("x", "y", "z"):
            axis.plot(
                relative,
                [float(row[f"{prefix}_{component}"]) for row in rows],
                linewidth=0.8,
                label=component,
            )
        axis.set(xlabel="Time from first IMU message (s)", ylabel=ylabel, title=f"PILOT_ONLY IMU {prefix}")
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(runtime / output, dpi=150)
        plt.close(fig)


def _write_data_reference(data_root: Path, inventory: Mapping[str, Any]) -> None:
    resolved = validate_runtime_root(data_root)
    resolved.mkdir(parents=True, exist_ok=True)
    write_json(
        resolved / "bag_reference.json",
        {
            "schema": "mid360_pilot_read_only_bag_reference_v1",
            **PILOT_FLAGS,
            "source_bag_path": inventory["bag_path"],
            "source_bag_sha256": inventory["bag_sha256"],
            "source_bag_size_bytes": inventory["bag_size_bytes"],
            "access_contract": "READ_ONLY_REFERENCE_NO_COPY_NO_MUTATION",
        },
    )


def run_audit(
    bag_path: Path,
    runtime_root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_root: Path | None = None,
) -> dict[str, Any]:
    bag = bag_path.expanduser().resolve(strict=True)
    runtime = validate_runtime_root(runtime_root)
    runtime.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    inventory = build_bag_inventory(bag)
    lidar_rows, lidar_summary = audit_lidar_messages(iter_topic_messages(bag, LIDAR_TOPIC))
    imu_rows, imu_summary = audit_imu_messages(
        iter_topic_messages(bag, IMU_TOPIC), config
    )
    write_json(runtime / "mid360_pilot_config.json", config)
    write_json(runtime / "bag_inventory.json", inventory)
    (runtime / "bag_inventory.md").write_text(
        inventory_markdown(inventory), encoding="utf-8"
    )
    write_csv(runtime / "topic_inventory.csv", _topic_csv_rows(inventory))
    write_csv(runtime / "lidar_frame_audit.csv", lidar_rows)
    write_json(runtime / "lidar_quality_summary.json", lidar_summary)
    write_csv(runtime / "imu_staticity_audit.csv", imu_rows)
    write_json(runtime / "imu_staticity_summary.json", imu_summary)
    _save_lidar_plots(runtime, lidar_rows)
    _save_imu_plots(runtime, imu_rows)
    if data_root is not None:
        _write_data_reference(data_root, inventory)
    audit_pass = (
        inventory["status"] == "PASS"
        and lidar_summary["status"] == "PASS"
        and imu_summary["status"] == "PASS"
    )
    audit_summary = {
        "schema": "mid360_pilot_audit_gate_v1",
        **PILOT_FLAGS,
        "bag_parse": inventory["status"],
        "lidar_qa": lidar_summary["status"],
        "imu_qa": imu_summary["status"],
        "STATICITY_SCREEN": imu_summary["STATICITY_SCREEN"],
        "status": "PASS" if audit_pass else "FAIL",
    }
    write_json(runtime / "pilot_audit_gate.json", audit_summary)
    return {
        "inventory": inventory,
        "lidar_rows": lidar_rows,
        "lidar_summary": lidar_summary,
        "imu_rows": imu_rows,
        "imu_summary": imu_summary,
        "audit_gate": audit_summary,
    }


def _selected_query_csv_rows(
    selected: Sequence[Mapping[str, Any]], candidate_count: int
) -> list[dict[str, Any]]:
    return [
        {
            "selection_index": row["selection_index"],
            "quantile": row["quantile"],
            "query_candidate_count": candidate_count,
            "target_sequence_rank": row["target_sequence_rank"],
            "selected_sequence_rank": row["selected_sequence_rank"],
            "frame_index": row["frame_index"],
            "timestamp": row["timestamp"],
            "relative_time_seconds": row["relative_time_seconds"],
            "point_count": row["point_count"],
            "finite_point_count": row["finite_point_count"],
        }
        for row in selected
    ]


def _write_sha256s(runtime: Path) -> None:
    paths = sorted(
        path
        for path in runtime.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS"}
    )
    text = "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths)
    (runtime / "SHA256SUMS").write_text(text, encoding="utf-8")


def _topic(inventory: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(row for row in inventory["topics"] if row["topic"] == name)


def _pilot_summary_markdown(summary: Mapping[str, Any]) -> str:
    geometry = summary["geometry_median"]
    query_times = ", ".join(f"{value:.9f}" for value in summary["selected_query_timestamps"])
    return f"""# Mid-360 single-bag pilot summary

`PILOT_ONLY=true`, `FORMAL_REAL_DATA=false`, `INDEPENDENT_MAP_QUERY_ACQUISITION=false`, `MEASUREMENT_EVIDENCE=false`.

- Bag: `{summary['bag_path']}`
- SHA256: `{summary['bag_sha256']}`
- Bag duration: `{summary['bag_duration_seconds']:.9f} s`
- LiDAR: `{summary['lidar_message_type']}`, {summary['lidar_frame_count']} frames, {summary['lidar_rate_hz']:.6f} Hz
- IMU: `{summary['imu_message_type']}`, {summary['imu_message_count']} messages, {summary['imu_rate_hz']:.6f} Hz
- Staticity screen: `{summary['STATICITY_SCREEN']}`
- Map frames: {summary['map_frame_count']}; target points: {summary['target_map_point_count']}
- Query candidates: {summary['query_candidate_count']}; selected timestamps: {query_times}
- Map/query disjoint: `{str(summary['map_query_strictly_disjoint']).lower()}`
- Geometry median normalized lambda min: `{geometry['normalized_lambda_min_trans']:.9g}`
- Geometry median condition number: `{geometry['condition_number_trans']:.9g}`
- Geometry median spectral entropy: `{geometry['spectral_entropy_trans']:.9g}`
- Geometry description: `{summary['GEOMETRY_DESCRIPTION']}`
- Registration used to build map: `false`
- Debug registration executed: `false`
- Formal-paper sufficiency: `NO`

This result validates only the single-bag pilot software path. Formal measurement requires independently acquired map/query bags.
"""


def run_prepare(
    bag_path: Path,
    runtime_root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_root: Path | None = None,
) -> dict[str, Any]:
    bag = bag_path.expanduser().resolve(strict=True)
    runtime = validate_runtime_root(runtime_root)
    config = load_config(config_path)
    audit = run_audit(
        bag,
        runtime,
        config_path=config_path,
        data_root=data_root,
    )
    if audit["audit_gate"]["status"] != "PASS":
        raise PilotBagError("mandatory bag/LiDAR/IMU audit gate failed")
    if audit["imu_summary"]["STATICITY_SCREEN"] != "NO_OBVIOUS_MOTION":
        raise PilotBagError(
            "static target-map assumption is not supported: "
            + str(audit["imu_summary"]["STATICITY_SCREEN"])
        )
    valid_frames = [
        row
        for row in audit["lidar_rows"]
        if int(row["finite_point_count"]) > 0 and int(row["point_count"]) > 0
    ]
    split_contract = build_split_contract(
        [float(row["timestamp"]) for row in audit["lidar_rows"]], config
    )
    partition = partition_lidar_frames(valid_frames, split_contract)
    selected = select_query_frames(partition["query"], config["query_quantiles"])
    lineage = build_lineage(partition, selected)
    if lineage["status"] != "PASS":
        raise PilotBagError("map/query lineage disjointness gate failed")
    map_indexes = {int(row["frame_index"]) for row in partition["map"]}
    selected_indexes = {int(row["frame_index"]) for row in selected}
    map_scans: list[np.ndarray] = []
    selected_scan_by_index: dict[int, np.ndarray] = {}
    for frame_index, message, _, _ in iter_topic_messages(bag, LIDAR_TOPIC):
        if frame_index not in map_indexes and frame_index not in selected_indexes:
            continue
        xyz = xyz_array(lidar_message_to_structured(message))
        if frame_index in map_indexes:
            map_scans.append(xyz)
        if frame_index in selected_indexes:
            selected_scan_by_index[frame_index] = xyz
    if len(map_scans) != len(map_indexes) or len(selected_scan_by_index) != 10:
        raise PilotBagError("failed to reload all selected map/query LiDAR frames")
    target_map, target_metadata = build_static_target_map(map_scans, config)
    target_path = runtime / "pilot_target_map.npy"
    np.save(target_path, target_map, allow_pickle=False)
    target_sha = sha256_file(target_path)
    target_metadata.update(
        {
            "bag_sha256": audit["inventory"]["bag_sha256"],
            "config_sha256": sha256_file(config_path),
            "map_frame_indexes": sorted(map_indexes),
            "map_timestamps": [float(row["timestamp"]) for row in partition["map"]],
            "pilot_target_map_npy_sha256": target_sha,
        }
    )
    query_points = []
    for row in selected:
        raw = selected_scan_by_index[int(row["frame_index"])]
        query_points.append(
            finite_range_filter(
                raw,
                minimum_range_m=float(config["map"]["minimum_range_m"]),
                maximum_range_m=float(config["map"]["maximum_range_m"]),
            )
        )
    geometry_rows, geometry_summary = compute_pilot_geometry(
        query_points, selected, target_map, config
    )
    write_json(runtime / "pilot_split_contract.json", split_contract)
    write_json(runtime / "pilot_target_map_metadata.json", target_metadata)
    (runtime / "pilot_target_map_sha256").write_text(target_sha + "\n", encoding="ascii")
    write_csv(
        runtime / "pilot_query_selection.csv",
        _selected_query_csv_rows(selected, len(partition["query"])),
    )
    write_json(runtime / "pilot_map_query_lineage.json", lineage)
    write_csv(runtime / "pilot_geometry_metrics.csv", geometry_rows)
    write_json(runtime / "pilot_geometry_summary.json", geometry_summary)
    lidar_topic = _topic(audit["inventory"], LIDAR_TOPIC)
    imu_topic = _topic(audit["inventory"], IMU_TOPIC)
    mandatory_gates = {
        "bag_parse": audit["inventory"]["status"],
        "lidar_qa": audit["lidar_summary"]["status"],
        "imu_qa": audit["imu_summary"]["status"],
        "staticity_screen": audit["imu_summary"]["STATICITY_SCREEN"],
        "target_map_build": target_metadata["status"],
        "ten_query_selection": "PASS" if len(selected) == 10 else "FAIL",
        "geometry_only": geometry_summary["status"],
        "map_query_lineage": lineage["status"],
    }
    ready = all(
        value in {"PASS", "NO_OBVIOUS_MOTION"} for value in mandatory_gates.values()
    )
    point_fields = audit["lidar_summary"]["field_signatures"][0]
    summary = {
        "schema": "mid360_single_bag_pilot_summary_v1",
        **PILOT_FLAGS,
        "MID360_SINGLE_BAG_PILOT_READY": ready,
        "FORMAL_MEASUREMENT_RESULT": False,
        "formal_paper_conclusion_proven": False,
        "formal_paper_sufficiency": "NO",
        "bag_path": str(bag),
        "bag_sha256": audit["inventory"]["bag_sha256"],
        "bag_duration_seconds": audit["inventory"]["duration_seconds"],
        "lidar_message_type": lidar_topic["message_type"],
        "lidar_frame_count": lidar_topic["message_count"],
        "lidar_rate_hz": lidar_topic["average_bag_record_rate_hz"],
        "imu_message_type": imu_topic["message_type"],
        "imu_message_count": imu_topic["message_count"],
        "imu_rate_hz": imu_topic["average_bag_record_rate_hz"],
        "mean_points_per_lidar_frame": audit["lidar_summary"]["point_count_mean"],
        "point_fields": point_fields,
        "point_coordinate_unit": audit["lidar_summary"]["point_coordinate_unit"],
        "point_timestamp_usable": audit["lidar_summary"]["point_timestamp_all_frames_usable"],
        "point_timestamp_serialized_order_nondecreasing": audit["lidar_summary"][
            "point_timestamp_all_frames_nondecreasing"
        ],
        "point_timestamp_order_note": audit["lidar_summary"][
            "point_timestamp_order_note"
        ],
        "point_timestamp_unit_inferences": audit["lidar_summary"]["point_timestamp_unit_inferences"],
        "STATICITY_SCREEN": audit["imu_summary"]["STATICITY_SCREEN"],
        "acceleration_unit_confirmed": False,
        "acceleration_unit_note": audit["imu_summary"]["acceleration_unit_note"],
        "map_frame_count": len(partition["map"]),
        "query_candidate_count": len(partition["query"]),
        "selected_query_timestamps": [float(row["timestamp"]) for row in selected],
        "map_query_strictly_disjoint": lineage["map_query_strictly_disjoint"],
        "target_map_point_count": int(target_map.shape[0]),
        "target_map_sha256": target_sha,
        "geometry_median": geometry_summary["median"],
        "geometry_minimum": geometry_summary["minimum"],
        "geometry_maximum": geometry_summary["maximum"],
        "GEOMETRY_DESCRIPTION": geometry_summary["GEOMETRY_DESCRIPTION"],
        "target_map_registration_called": False,
        "debug_registration_executed": False,
        "debug_open3d_results": None,
        "debug_pcl_results": None,
        "debug_backend_trend_consistent": None,
        "parsing_or_coordinate_bug_detected": False,
        "parsing_notes": [
            "PointCloud2 layout and per-point timestamp matched the ROS header scale.",
            "Per-point timestamps are not monotonic in serialized point order; consume the values per point or sort explicitly.",
            "Coordinate unit is not encoded by PointCloud2 and remains independently unconfirmed.",
            "Acceleration scale is approximately 1 g in raw values; no SI conversion was applied.",
        ],
        "FORMAL_INDEPENDENT_COLLECTION_SOFTWARE_READY": ready,
        "formal_collection_remaining_actions": [
            "Acquire physically independent map_xxx.bag and query_xxx.bag under the formal protocol.",
            "Confirm and document LiDAR coordinate and IMU acceleration units from the exact driver configuration.",
            "Re-run lineage/frame/unit checks on both independent bags before any formal registration.",
            "Use the frozen Open3D/PCL runtime versions and parameters for any formal backend execution.",
        ],
        "mandatory_gates": mandatory_gates,
    }
    write_json(runtime / "pilot_summary.json", summary)
    (runtime / "pilot_summary.md").write_text(
        _pilot_summary_markdown(summary), encoding="utf-8"
    )
    _write_sha256s(runtime)
    return summary


def run_phase(
    phase: str,
    bag_path: Path,
    runtime_root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_root: Path | None = None,
    backend_parameter_contract: Path = DEFAULT_BACKEND_PARAMETER_CONTRACT,
    pcl_executable: Path = DEFAULT_PCL_EXECUTABLE,
) -> dict[str, Any]:
    if phase == "audit":
        result = run_audit(
            bag_path,
            runtime_root,
            config_path=config_path,
            data_root=data_root,
        )
        return result["audit_gate"]
    if phase == "prepare":
        return run_prepare(
            bag_path,
            runtime_root,
            config_path=config_path,
            data_root=data_root,
        )
    if phase == "debug-inputs":
        from .debug_registration import export_debug_inputs

        return export_debug_inputs(bag_path, runtime_root, config_path)
    if phase == "debug-registration":
        from .debug_registration import execute_debug_registration

        return execute_debug_registration(
            runtime_root,
            backend_parameter_contract,
            pcl_executable,
        )
    if phase == "debug-finalize":
        from .debug_registration import finalize_debug_reporting

        return finalize_debug_reporting(runtime_root)
    if phase == "debug-classify-limitations":
        from .debug_registration import apply_documented_readiness_limitation

        return apply_documented_readiness_limitation(runtime_root)
    raise PilotBagError(f"unsupported phase: {phase}")


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_BACKEND_PARAMETER_CONTRACT",
    "DEFAULT_PCL_EXECUTABLE",
    "load_config",
    "run_audit",
    "run_phase",
    "run_prepare",
    "validate_runtime_root",
    "write_csv",
    "write_json",
]
