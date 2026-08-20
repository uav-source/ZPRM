"""Finalize FMB1 pre-registration evidence without authorizing registration."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from phase_a_harness.mid360_pilot.bag_reader import IMU_TOPIC, LIDAR_TOPIC, sha256_file
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    GEOMETRY_ONLY_FIELDS,
)

from .protocol import (
    BACKEND_CONTRACT_SHA256,
    INITIAL_SCENE_IDS,
    QUERY_QUANTILES,
    REQUIRED_FRAME_ID,
    REQUIRED_POINT_FIELDS,
    STATION_IDS,
    bag_filename,
)


class PreRegistrationFinalizeError(RuntimeError):
    """Raised when the pre-registration evidence cannot be frozen safely."""


REFRESH_ENV = "FMB1_REFRESH_UNAUTHORIZED_PREREGISTRATION_RESULTS"


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_once(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            if os.environ.get(REFRESH_ENV) != "1":
                raise PreRegistrationFinalizeError(
                    f"refusing to overwrite different frozen result: {path}"
                )
            temporary = path.with_suffix(path.suffix + ".refresh.tmp")
            if temporary.exists():
                raise PreRegistrationFinalizeError(
                    f"stale refresh temporary exists: {temporary}"
                )
            temporary.write_bytes(content)
            temporary.replace(path)
        return path
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    return _write_once(path, _canonical_json(payload))


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> Path:
    return _write_once(path, _csv_bytes(rows, fields))


def _raw_rows(acquisition: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = acquisition.get("raw_bags", acquisition.get("bags"))
    if not isinstance(rows, list):
        raise PreRegistrationFinalizeError("acquisition payload lacks raw bag rows")
    return rows


def _stations(acquisition: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = acquisition.get("stations")
    if not isinstance(rows, list):
        raise PreRegistrationFinalizeError("acquisition payload lacks station rows")
    return rows


def _path(row: Mapping[str, Any]) -> Path:
    value = row.get("raw_absolute_path", row.get("raw_path"))
    if value is None and isinstance(row.get("inventory"), Mapping):
        value = row["inventory"].get("bag_path")
    if value is None:
        raise PreRegistrationFinalizeError("bag row lacks raw path")
    return Path(str(value)).resolve(strict=True)


def _inventory(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("inventory", {})
    return value if isinstance(value, Mapping) else {}


def _audit(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("audit", row.get("bag_audit", {}))
    return value if isinstance(value, Mapping) else {}


def _lidar(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("lidar_summary", {})
    return value if isinstance(value, Mapping) else {}


def _imu(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("imu_summary", {})
    return value if isinstance(value, Mapping) else {}


def _sha(row: Mapping[str, Any]) -> str:
    value = row.get("sha256", row.get("SHA256"))
    if value is None:
        value = _inventory(row).get("bag_sha256")
    return str(value)


def _prefix(row: Mapping[str, Any]) -> str:
    value = row.get("capture_prefix", row.get("capture_timestamp"))
    if value is not None:
        return str(value)
    name = _path(row).name
    tokens = name.split("_")
    return f"{tokens[1]}_{tokens[2]}"


def _topic(inventory: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for row in inventory.get("topics", []):
        if row.get("topic") == name:
            return row
    return {}


def _station_status(row: Mapping[str, Any]) -> str:
    return str(
        row.get(
            "acquisition_status",
            row.get(
                "station_status",
                row.get("station_acquisition_status", "ACQUISITION_FAIL"),
            ),
        )
    )


def _station_pair(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("pair_audit", {})
    return value if isinstance(value, Mapping) else {}


def _bag_for(
    rows: Sequence[Mapping[str, Any]], scene_id: str, station_id: str, role: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if str(row.get("scene_id")) == scene_id
        and str(row.get("station_id")) == station_id
        and str(row.get("role")) == role
    ]
    if len(matches) != 1:
        raise PreRegistrationFinalizeError(
            f"expected one {role} bag for {scene_id}/{station_id}"
        )
    return matches[0]


def _flatten_imu(row: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "scene_id": row.get("scene_id"),
        "station_id": row.get("station_id"),
        "role": row.get("role"),
        "raw_filename": _path(row).name,
        "sha256": _sha(row),
        "motion_status": _imu(row).get("STATICITY_SCREEN"),
        "acceleration_unit": "UNKNOWN",
        "acceleration_conversion_applied": False,
        "large_spike_count": _imu(row).get("large_spike_count"),
        "maximum_gyro_step": _imu(row).get("maximum_gyro_step"),
        "maximum_accel_step_raw": _imu(row).get("maximum_accel_step_raw"),
    }
    statistics = _imu(row).get("statistics", {})
    for channel in (
        "gyro_x", "gyro_y", "gyro_z", "gyro_norm",
        "accel_x", "accel_y", "accel_z", "accel_norm",
    ):
        channel_stats = statistics.get(channel, {})
        for statistic in ("median", "mad", "q95", "q99", "max", "min", "absolute_max"):
            output[f"{channel}_{statistic}"] = channel_stats.get(statistic)
    return output


def _aliases(name: str) -> tuple[str, ...]:
    if name.startswith("fmb1_") or name in {"NO_ICP_ATTESTATION.json", "SHA256SUMS"}:
        return (name,)
    return (name, f"fmb1_{name}")


def _record_paths(
    generated: list[Path], writer: Any, results_dir: Path, name: str, *args: Any
) -> None:
    for alias in _aliases(name):
        generated.append(writer(results_dir / alias, *args))


def _summary_markdown(readiness: Mapping[str, Any]) -> str:
    failures = readiness.get("failures", [])
    fail_text = "none" if not failures else "; ".join(
        f"{row.get('scene_id', '')}/{row.get('station_id', '')}: {row.get('failure_reason', row.get('reason'))}"
        for row in failures
    )
    lines = [
        "# Mid-360 Formal Batch-1 预注册数据就绪总结",
        "",
        f"1. 实际 bag 数：{readiness['bag_count']}。",
        f"2. MAP/QUERY 对数：{readiness['pair_count']}，严格配对={str(readiness['inventory_pair_gate_pass']).lower()}。",
        f"3. 3 Rich + 3 Weak 映射完整冻结={str(readiness['mapping_frozen']).lower()}。",
        f"4. SHA 认证：{readiness['authenticated_bag_count']}/{readiness['bag_count']}。",
        f"5. acquisition PASS：{readiness['acquisition_pass_station_count']}/18。",
        f"6. FAIL/REVIEW：{fail_text}。",
        f"7. MAP 时长范围：{readiness.get('map_duration_range_s')} s。",
        f"8. QUERY 时长范围：{readiness.get('query_duration_range_s')} s。",
        f"9. gap 范围：{readiness.get('gap_range_s')} s。",
        f"10. gap <10 s：{readiness.get('gap_lt_10_count', 0)}。",
        f"11. MAP/QUERY overlap：{readiness.get('overlap_count', 0)}。",
        f"12. LiDAR rate 范围：{readiness.get('lidar_rate_range_hz')} Hz。",
        f"13. IMU rate 范围：{readiness.get('imu_rate_range_hz')} Hz。",
        f"14. 全部 livox_frame={str(readiness.get('all_livox_frame')).lower()}。",
        f"15. 全部必需字段齐全={str(readiness.get('all_required_fields')).lower()}。",
        f"16. 明显运动 bag 数：{readiness.get('motion_suspected_bag_count', 0)}。",
        f"17. 需要补采 station 数：{readiness.get('reacquisition_station_count', 0)}。",
        f"18. 无 registration target：{readiness['target_count']}/18。",
        f"19. QUERY 进入 target：{readiness.get('query_contribution_total', 0)}。",
        f"20. 每站 10 query={str(readiness.get('ten_snapshots_per_station')).lower()}。",
        f"21. snapshot 总数：{readiness['snapshot_count']}/180。",
        f"22. R candidates 整体 geometry-rich={str(readiness.get('all_r_candidates_rich')).lower()}。",
        f"23. W candidates 整体 geometry-weak={str(readiness.get('all_w_candidates_weak')).lower()}。",
        f"24. geometry PASS scenes：{', '.join(readiness.get('geometry_admitted_scenes', [])) or 'none'}。",
        f"25. REVIEW/REJECT scenes：{', '.join(readiness.get('geometry_not_admitted_scenes', [])) or 'none'}。",
        f"26. 最终 Rich scenes：{readiness['rich_scene_count']}。",
        f"27. 最终 Weak scenes：{readiness['weak_scene_count']}。",
        "28. ICP 参数修改：否。",
        "29. Open3D/PCL registration 执行：否。",
        f"30. NO_ICP_ATTESTATION PASS={str(readiness['NO_ICP_ATTESTATION_PASS']).lower()}。",
        f"31. Formal Batch-1 verifier PASS={str(readiness.get('formal_batch1_verifier_pass', False)).lower()}（独立 verifier 运行后更新其独立报告，不回写冻结结果）。",
        "32. 全量测试状态见最终执行日志；registration 执行类测试在 NO_FORMAL_REGISTRATION 模式下不得运行。",
        f"33. FMB1_PRE_REGISTRATION_DATA_READY={str(readiness['FMB1_PRE_REGISTRATION_DATA_READY']).lower()}。",
        f"34. 具备后续 360 次独立授权条件={str(readiness['READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION']).lower()}。",
        "",
        "本冻结没有产生 T_est、误差、最终残差、fitness 或 solver result。",
    ]
    return "\n".join(lines) + "\n"


def finalize_pre_registration(
    acquisition: Mapping[str, Any],
    assets: Mapping[str, Any],
    geometry: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    repository: Path,
    results_dir: Path,
    source_bindings: Mapping[str, str],
    deep_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write all small formal evidence while keeping registration unauthorized."""

    backend_path = repository / "frozen_assets/backend_parameter_contract.json"
    if sha256_file(backend_path) != BACKEND_CONTRACT_SHA256:
        raise PreRegistrationFinalizeError("backend parameter contract changed")
    if any(
        int(attestation.get(key, -1)) != 0
        for key in (
            "open3d_registration_call_count",
            "pcl_cli_invocation_count",
            "other_registration_process_count",
            "formal_trial_count",
        )
    ):
        raise PreRegistrationFinalizeError("NO-ICP attestation counters are not zero")

    raw = sorted(
        _raw_rows(acquisition),
        key=lambda row: (str(row["scene_id"]), str(row["station_id"]), str(row["role"])),
    )
    stations = sorted(
        _stations(acquisition),
        key=lambda row: (str(row["scene_id"]), str(row["station_id"])),
    )
    targets = sorted(
        assets.get("targets", []),
        key=lambda row: (str(row["scene_id"]), str(row["station_id"])),
    )
    snapshots = sorted(
        assets.get("snapshots", []),
        key=lambda row: (
            str(row["scene_id"]), str(row["station_id"]), int(row["selection_index"])
        ),
    )
    metric_rows = sorted(
        geometry.get("snapshot_metrics", []),
        key=lambda row: (
            str(row["scene_id"]), str(row["station_id"]), int(row["selection_index"])
        ),
    )
    scene_rows = sorted(geometry.get("scene_summaries", []), key=lambda row: str(row["scene_id"]))

    mapping_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for row in raw:
        path = _path(row)
        stat = path.stat()
        prefix = _prefix(row)
        canonical = bag_filename(
            str(row["scene_id"]), str(row["station_id"]), str(row["role"]), prefix
        )
        common = {
            "scene_id": row["scene_id"],
            "station_id": row["station_id"],
            "role": row["role"],
            "raw_filename": path.name,
            "raw_absolute_path": str(path),
            "canonical_filename": canonical,
            "capture_prefix": prefix,
            "sha256": _sha(row),
            "bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "canonical_materialization": "MANIFEST_ONLY_NO_COPY",
        }
        mapping_rows.append(common)
        inventory_rows.append(
            {
                **common,
                "capture_timestamp": prefix,
                "duration_s": _inventory(row).get("duration_seconds", _audit(row).get("duration_s")),
                "start_timestamp": _inventory(row).get("start_timestamp"),
                "end_timestamp": _inventory(row).get("end_timestamp"),
                "lidar_count": _topic(_inventory(row), LIDAR_TOPIC).get("message_count"),
                "imu_count": _topic(_inventory(row), IMU_TOPIC).get("message_count"),
                "lidar_rate_hz": _topic(_inventory(row), LIDAR_TOPIC).get("average_bag_record_rate_hz"),
                "imu_rate_hz": _topic(_inventory(row), IMU_TOPIC).get("average_bag_record_rate_hz"),
                "frame_ids": ";".join(_audit(row).get("frame_ids", [])),
                "pointcloud2_fields": ";".join(_audit(row).get("pointcloud2_fields", [])),
                "motion_status": _audit(row).get("motion_audit_status"),
                "bag_acquisition_pass": _audit(row).get("ACQUISITION_AUDIT_PASS", False),
            }
        )

    station_audit_rows: list[dict[str, Any]] = []
    for station in stations:
        scene_id = str(station["scene_id"])
        station_id = str(station["station_id"])
        map_row = _bag_for(raw, scene_id, station_id, "MAP")
        query_row = _bag_for(raw, scene_id, station_id, "QUERY")
        pair = _station_pair(station)
        failures = station.get("failure_reasons", station.get("failure_reason", []))
        if isinstance(failures, str):
            failures = [failures] if failures else []
        station_audit_rows.append(
            {
                "scene_id": scene_id,
                "station_id": station_id,
                "capture_prefix": _prefix(map_row),
                "semantic_candidate_label": (
                    "RICH_CANDIDATE" if scene_id.startswith("FMB1_R") else "WEAK_CANDIDATE"
                ),
                "map_sha256": _sha(map_row),
                "query_sha256": _sha(query_row),
                "map_duration_s": _audit(map_row).get("duration_s"),
                "query_duration_s": _audit(query_row).get("duration_s"),
                "map_lidar_count": _topic(_inventory(map_row), LIDAR_TOPIC).get("message_count"),
                "query_lidar_count": _topic(_inventory(query_row), LIDAR_TOPIC).get("message_count"),
                "map_imu_count": _topic(_inventory(map_row), IMU_TOPIC).get("message_count"),
                "query_imu_count": _topic(_inventory(query_row), IMU_TOPIC).get("message_count"),
                "map_lidar_rate_hz": _topic(_inventory(map_row), LIDAR_TOPIC).get("average_bag_record_rate_hz"),
                "query_lidar_rate_hz": _topic(_inventory(query_row), LIDAR_TOPIC).get("average_bag_record_rate_hz"),
                "map_imu_rate_hz": _topic(_inventory(map_row), IMU_TOPIC).get("average_bag_record_rate_hz"),
                "query_imu_rate_hz": _topic(_inventory(query_row), IMU_TOPIC).get("average_bag_record_rate_hz"),
                "map_motion_status": _audit(map_row).get("motion_audit_status"),
                "query_motion_status": _audit(query_row).get("motion_audit_status"),
                "gap_s": pair.get("actual_gap_s"),
                "map_query_no_overlap": pair.get("map_query_no_overlap"),
                "gap_pass": pair.get("actual_gap_minimum_pass"),
                "timestamp_basis": "ROSBAG_GLOBAL_MESSAGE_TIME",
                "acquisition_status": _station_status(station),
                "failure_reason": ";".join(str(value) for value in failures),
                "invalidated_at_utc": station.get("invalidated_at_utc"),
            }
        )

    imu_rows = [_flatten_imu(row) for row in raw]
    target_rows = [
        {
            "scene_id": row["scene_id"],
            "station_id": row["station_id"],
            "map_bag_sha256": row["map_bag_sha256"],
            "map_frame_count": row["map_frame_count"],
            "raw_point_count": row["raw_point_count"],
            "filtered_point_count": row["filtered_point_count"],
            "target_point_count": row["target_point_count"],
            "target_path": row["target_path"],
            "target_npy_sha256": row["target_npy_sha256"],
            "target_array_sha256": row["target_array_sha256"],
            "target_size_bytes": Path(str(row["target_path"])).stat().st_size,
            "input_roles": "MAP",
            "query_frame_count": 0,
            "query_contribution_to_target": 0,
            "construction": row["construction"],
            "registration_called": False,
            "odometry_called": False,
            "scan_matching_called": False,
            "preprocessing_source_sha256": json.dumps(row.get("preprocessing_source_sha256", {}), sort_keys=True),
        }
        for row in targets
    ]
    snapshot_rows = [
        {
            key: row.get(key)
            for key in (
                "scene_id", "station_id", "snapshot_id", "selection_index", "quantile",
                "candidate_count", "target_sequence_rank", "selected_sequence_rank",
                "query_frame_index", "query_timestamp", "query_bag_sha256", "source_path",
                "source_npy_sha256", "source_array_sha256", "source_point_count",
                "target_npy_sha256", "target_array_sha256", "selection_method",
                "selection_frozen_before_registration",
            )
        }
        for row in snapshots
    ]
    for snapshot_row in snapshot_rows:
        snapshot_row["source_size_bytes"] = Path(
            str(snapshot_row["source_path"])
        ).stat().st_size
    allowed_metric_rows = [
        {
            "scene_id": row["scene_id"],
            "station_id": row["station_id"],
            "snapshot_id": row["snapshot_id"],
            "selection_index": row["selection_index"],
            **{field: row[field] for field in GEOMETRY_ONLY_FIELDS},
        }
        for row in metric_rows
    ]

    admitted = [row for row in scene_rows if row.get("geometry_admission_status") == "GEOMETRY_ADMITTED"]
    rich = [row for row in admitted if row.get("final_geometry_class") == "RICH"]
    weak = [row for row in admitted if row.get("final_geometry_class") == "WEAK"]
    all_r_rich = all(
        any(row["scene_id"] == scene and row.get("final_geometry_class") == "RICH" and row.get("geometry_admission_status") == "GEOMETRY_ADMITTED" for row in scene_rows)
        for scene in INITIAL_SCENE_IDS[:3]
    )
    all_w_weak = all(
        any(row["scene_id"] == scene and row.get("final_geometry_class") == "WEAK" and row.get("geometry_admission_status") == "GEOMETRY_ADMITTED" for row in scene_rows)
        for scene in INITIAL_SCENE_IDS[3:]
    )
    pass_stations = [row for row in stations if _station_status(row) == "ACQUISITION_PASS"]
    no_icp_pass = all(
        int(attestation.get(key, -1)) == 0
        for key in (
            "open3d_registration_call_count", "pcl_cli_invocation_count",
            "other_registration_process_count", "formal_trial_count",
        )
    ) and attestation.get("FORMAL_REGISTRATION_AUTHORIZED") is False
    exact_snapshots = all(
        len([row for row in snapshots if row["scene_id"] == scene and row["station_id"] == station]) == 10
        for scene in INITIAL_SCENE_IDS for station in STATION_IDS
    )
    ready = bool(
        len(raw) == 36
        and len(stations) == 18
        and len(pass_stations) == 18
        and len(targets) == 18
        and len(snapshots) == 180
        and len(metric_rows) == 180
        and len(scene_rows) == 6
        and len(rich) == 3
        and len(weak) == 3
        and all_r_rich
        and all_w_weak
        and exact_snapshots
        and not assets.get("failures")
        and not geometry.get("failures")
        and no_icp_pass
    )

    durations_map = [float(row["duration_s"]) for row in inventory_rows if row["role"] == "MAP"]
    durations_query = [float(row["duration_s"]) for row in inventory_rows if row["role"] == "QUERY"]
    lidar_rates = [float(row["lidar_rate_hz"]) for row in inventory_rows]
    imu_rates = [float(row["imu_rate_hz"]) for row in inventory_rows]
    gaps = [float(_station_pair(row)["actual_gap_s"]) for row in stations]
    failure_rows: list[dict[str, Any]] = []
    for row in stations:
        if _station_status(row) != "ACQUISITION_PASS":
            failure_rows.append(
                {
                    "scene_id": row["scene_id"],
                    "station_id": row["station_id"],
                    "failure_reason": row.get("failure_reason", row.get("failure_reasons")),
                    "replacement_allowed_under_preregistration": True,
                }
            )
    for row in scene_rows:
        if row.get("geometry_admission_status") != "GEOMETRY_ADMITTED":
            failure_rows.append(
                {
                    "scene_id": row["scene_id"],
                    "station_id": None,
                    "failure_reason": row.get("failure_reason"),
                    "replacement_allowed_under_preregistration": row.get("replacement_allowed_under_preregistration", True),
                }
            )
    failure_rows.extend(assets.get("failures", []))
    failure_rows.extend(geometry.get("failures", []))
    readiness = {
        "schema": "mid360_fmb1_pre_registration_readiness_v1",
        "FMB1_PRE_REGISTRATION_DATA_READY": ready,
        "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION": ready,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "MEASUREMENT_FINAL_RESULT": False,
        "NO_FORMAL_REGISTRATION": True,
        "bag_count": len(raw),
        "authenticated_bag_count": sum(len(_sha(row)) == 64 for row in raw),
        "pair_count": len(stations),
        "inventory_pair_gate_pass": len(raw) == 36 and len(stations) == 18,
        "mapping_frozen": {str(row["scene_id"]) for row in stations} == set(INITIAL_SCENE_IDS),
        "scene_count": len(scene_rows),
        "rich_scene_count": len(rich),
        "weak_scene_count": len(weak),
        "station_count": len(stations),
        "acquisition_pass_station_count": len(pass_stations),
        "target_count": len(targets),
        "snapshot_count": len(snapshots),
        "rich_snapshot_count": sum(str(row["scene_id"]).startswith("FMB1_R") for row in snapshots),
        "weak_snapshot_count": sum(str(row["scene_id"]).startswith("FMB1_W") for row in snapshots),
        "planned_open3d_trials": 180,
        "planned_pcl_trials": 180,
        "planned_total_trials": 360,
        "actual_trials": 0,
        "actual_registration_trials": 0,
        "registration_execution_count": 0,
        "NO_ICP_ATTESTATION_PASS": no_icp_pass,
        "formal_batch1_verifier_pass": False,
        "map_duration_range_s": [min(durations_map), max(durations_map)] if durations_map else None,
        "query_duration_range_s": [min(durations_query), max(durations_query)] if durations_query else None,
        "gap_range_s": [min(gaps), max(gaps)] if gaps else None,
        "gap_lt_10_count": sum(value < 10.0 for value in gaps),
        "overlap_count": sum(not bool(_station_pair(row).get("map_query_no_overlap")) for row in stations),
        "lidar_rate_range_hz": [min(lidar_rates), max(lidar_rates)] if lidar_rates else None,
        "imu_rate_range_hz": [min(imu_rates), max(imu_rates)] if imu_rates else None,
        "all_livox_frame": all(_audit(row).get("frame_ids") == [REQUIRED_FRAME_ID] for row in raw),
        "all_required_fields": all(set(_audit(row).get("pointcloud2_fields", [])) >= set(REQUIRED_POINT_FIELDS) for row in raw),
        "motion_suspected_bag_count": sum(_audit(row).get("motion_audit_status") != "NO_OBVIOUS_MOTION" for row in raw),
        "reacquisition_station_count": sum(_station_status(row) != "ACQUISITION_PASS" for row in stations),
        "query_contribution_total": sum(int(row.get("query_contribution_to_target", -1)) for row in targets),
        "ten_snapshots_per_station": exact_snapshots,
        "all_r_candidates_rich": all_r_rich,
        "all_w_candidates_weak": all_w_weak,
        "geometry_admitted_scenes": [row["scene_id"] for row in admitted],
        "geometry_not_admitted_scenes": [row["scene_id"] for row in scene_rows if row not in admitted],
        "failures": failure_rows,
    }

    scene_registry = {
        "schema": "mid360_fmb1_scene_registry_frozen_v1",
        "frozen_before_registration": True,
        "scenes": [
            {
                "scene_id": scene,
                "semantic_candidate_label": "RICH_CANDIDATE" if scene.startswith("FMB1_R") else "WEAK_CANDIDATE",
                "geometry": next((row for row in scene_rows if row["scene_id"] == scene), None),
            }
            for scene in INITIAL_SCENE_IDS
        ],
    }
    station_registry = {
        "schema": "mid360_fmb1_station_registry_frozen_v1",
        "frozen_before_registration": True,
        "stations": station_audit_rows,
    }
    admission_manifest = {
        "schema": "mid360_fmb1_geometry_admission_manifest_v1",
        "hierarchy": {"independent_unit": "scene", "nested_repeats": ["station", "snapshot"]},
        "candidate_labels_are_not_geometry_labels": True,
        "scene_summaries": scene_rows,
        "replacement_decisions_use_registration": False,
    }
    identity = json.dumps(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        separators=(",", ":"),
    )
    canonical_inputs = [
        {
            "scene_id": row["scene_id"],
            "station_id": row["station_id"],
            "snapshot_id": row["snapshot_id"],
            "query_timestamp": row["query_timestamp"],
            "source_path": row["source_path"],
            "source_sha256": row["source_npy_sha256"],
            "target_path": next(target["target_path"] for target in targets if target["scene_id"] == row["scene_id"] and target["station_id"] == row["station_id"]),
            "target_sha256": row["target_npy_sha256"],
            "T0": identity,
            "open3d_source_sha256": row["source_npy_sha256"],
            "pcl_source_sha256": row["source_npy_sha256"],
            "open3d_target_sha256": row["target_npy_sha256"],
            "pcl_target_sha256": row["target_npy_sha256"],
            "backend_inputs_byte_identical": True,
        }
        for row in snapshots
    ]
    no_icp = {
        **dict(attestation),
        "schema": "mid360_fmb1_no_icp_attestation_v1",
        "open3d_registration_call_count": 0,
        "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0,
        "formal_trial_count": 0,
        "actual_registration_trials": 0,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "MEASUREMENT_FINAL_RESULT": False,
        "NO_ICP_ATTESTATION_PASS": no_icp_pass,
        "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
    }

    generated: list[Path] = []
    mapping_fields = [
        "scene_id", "station_id", "role", "raw_filename", "raw_absolute_path",
        "canonical_filename", "capture_prefix", "sha256", "bytes", "mtime", "mtime_utc",
        "canonical_materialization",
    ]
    inventory_fields = mapping_fields + [
        "capture_timestamp", "duration_s", "start_timestamp", "end_timestamp",
        "lidar_count", "imu_count", "lidar_rate_hz", "imu_rate_hz", "frame_ids",
        "pointcloud2_fields", "motion_status", "bag_acquisition_pass",
    ]
    _record_paths(generated, _write_csv, results_dir, "raw_to_canonical_mapping.csv", mapping_rows, mapping_fields)
    _record_paths(generated, _write_csv, results_dir, "raw_bag_inventory.csv", inventory_rows, inventory_fields)
    raw_json = {"schema": "mid360_fmb1_raw_bag_inventory_v1", "bag_count": len(raw), "bags": raw}
    _record_paths(generated, _write_json, results_dir, "raw_bag_inventory.json", raw_json)
    station_fields = list(station_audit_rows[0]) if station_audit_rows else []
    _record_paths(generated, _write_csv, results_dir, "station_acquisition_audit.csv", station_audit_rows, station_fields)
    _record_paths(generated, _write_json, results_dir, "station_acquisition_audit.json", {"schema": "mid360_fmb1_station_acquisition_audit_v1", "stations": station_audit_rows})
    imu_fields = list(imu_rows[0]) if imu_rows else []
    _record_paths(generated, _write_csv, results_dir, "imu_staticity_audit.csv", imu_rows, imu_fields)
    target_fields = list(target_rows[0]) if target_rows else []
    _record_paths(generated, _write_csv, results_dir, "target_map_manifest.csv", target_rows, target_fields)
    snapshot_fields = list(snapshot_rows[0]) if snapshot_rows else []
    _record_paths(generated, _write_csv, results_dir, "snapshot_inventory_pregeometry.csv", snapshot_rows, snapshot_fields)
    metric_fields = ["scene_id", "station_id", "snapshot_id", "selection_index", *GEOMETRY_ONLY_FIELDS]
    _record_paths(generated, _write_csv, results_dir, "geometry_metrics.csv", allowed_metric_rows, metric_fields)
    scene_fields = list(scene_rows[0]) if scene_rows else []
    _record_paths(generated, _write_csv, results_dir, "geometry_scene_summary.csv", scene_rows, scene_fields)
    _record_paths(generated, _write_json, results_dir, "scene_registry_frozen.yaml", scene_registry)
    _record_paths(generated, _write_json, results_dir, "station_registry_frozen.yaml", station_registry)
    _record_paths(generated, _write_csv, results_dir, "snapshot_inventory_frozen.csv", snapshot_rows, snapshot_fields)
    _record_paths(generated, _write_json, results_dir, "geometry_admission_manifest.json", admission_manifest)
    canonical_fields = list(canonical_inputs[0]) if canonical_inputs else []
    _record_paths(generated, _write_csv, results_dir, "canonical_input_manifest.csv", canonical_inputs, canonical_fields)
    generated.append(_write_json(results_dir / "NO_ICP_ATTESTATION.json", no_icp))
    generated.append(_write_json(results_dir / "fmb1_pre_registration_readiness.json", readiness))
    generated.append(_write_once(results_dir / "fmb1_pre_registration_summary.md", _summary_markdown(readiness).encode("utf-8")))
    if not ready:
        generated.append(
            _write_json(
                results_dir / "REACQUISITION_REQUIRED.json",
                {
                    "schema": "mid360_fmb1_reacquisition_or_replacement_required_v1",
                    "REACQUISITION_REQUIRED": any(row.get("station_id") for row in failure_rows),
                    "items": failure_rows,
                    "old_attempts_retained": True,
                    "new_capture_timestamp_required": True,
                },
            )
        )

    frozen_manifest = {
        "schema": "mid360_fmb1_complete_frozen_manifest_v1",
        "source_bindings": dict(source_bindings),
        "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
        "mapping": mapping_rows,
        "raw_bags": raw,
        "stations": station_audit_rows,
        "targets": target_rows,
        "snapshots": snapshot_rows,
        "geometry_metrics": allowed_metric_rows,
        "geometry_scenes": scene_rows,
        "canonical_inputs": canonical_inputs,
        "no_icp_attestation": no_icp,
        "readiness": readiness,
        "deep_verification": (
            dict(deep_verification) if deep_verification is not None else None
        ),
    }
    manifest_path = _write_json(results_dir / "fmb1_frozen_manifest.json", frozen_manifest)
    generated.append(manifest_path)
    manifest_digest_path = _write_once(
        results_dir / "fmb1_frozen_manifest.sha256",
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n".encode("ascii"),
    )
    generated.append(manifest_digest_path)
    deep_report_path = results_dir / "fmb1_deep_verification_report.json"
    if deep_report_path.is_file():
        generated.append(deep_report_path)
    checksum_rows = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(set(generated), key=lambda item: item.name)
    ]
    _write_once(results_dir / "SHA256SUMS", ("\n".join(checksum_rows) + "\n").encode("ascii"))
    return readiness


__all__ = ["PreRegistrationFinalizeError", "finalize_pre_registration"]
