from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from experiments.mid360_formal_batch1.protocol import (
    FormalBatchError,
    audit_pair,
    build_bag_manifest,
    build_recording_plan,
    duration_passes,
    ensure_paths_do_not_exist,
    evaluate_bag_audit,
    pair_timing_audit,
)


FIELDS = ["x", "y", "z", "intensity", "tag", "line", "timestamp"]


def inventory(*, lidar_rate: float = 10.0, imu_rate: float = 200.0) -> dict:
    return {
        "bag_path": "/data/FMB1_R01_S01_MAP_20260820_101530.bag",
        "bag_sha256": "a" * 64,
        "start_timestamp": 100.0,
        "end_timestamp": 120.0,
        "duration_seconds": 20.0,
        "topics": [
            {
                "topic": "/livox/lidar",
                "message_type": "sensor_msgs/PointCloud2",
                "message_count": 201,
                "average_bag_record_rate_hz": lidar_rate,
                "frame_ids": ["livox_frame"],
                "fields": [{"name": name} for name in FIELDS],
            },
            {
                "topic": "/livox/imu",
                "message_type": "sensor_msgs/Imu",
                "message_count": 4001,
                "average_bag_record_rate_hz": imu_rate,
                "frame_ids": ["livox_frame"],
                "fields": [],
            },
        ],
    }


def evaluated(source: Optional[dict] = None, **kwargs: object) -> dict:
    return evaluate_bag_audit(
        source or inventory(),
        role=str(kwargs.pop("role", "MAP")),
        lidar_summary=kwargs.pop("lidar_summary", {"status": "PASS"}),
        imu_summary=kwargs.pop(
            "imu_summary",
            {"status": "PASS", "STATICITY_SCREEN": "NO_OBVIOUS_MOTION", "large_spike_count": 0},
        ),
        finite_point_fraction=float(kwargs.pop("finite_point_fraction", 1.0)),
    )


def test_map_and_query_duration_hard_gates_are_inclusive() -> None:
    assert duration_passes("MAP", 19.0)
    assert not duration_passes("MAP", 18.999)
    assert duration_passes("QUERY", 14.0)
    assert not duration_passes("QUERY", 13.999)


def test_gap_ten_seconds_is_inclusive_and_overlap_fails() -> None:
    assert pair_timing_audit(0.0, 20.0, 30.0, 45.0)["timing_pass"]
    assert not pair_timing_audit(0.0, 20.0, 29.999, 45.0)["timing_pass"]
    overlap = pair_timing_audit(0.0, 20.0, 19.0, 34.0)
    assert overlap["map_query_no_overlap"] is False
    assert overlap["timing_pass"] is False


def test_topic_type_and_frame_audit_fail_closed() -> None:
    wrong_type = inventory()
    wrong_type["topics"][0]["message_type"] = "livox_ros_driver/CustomMsg"
    assert not evaluated(wrong_type)["ACQUISITION_AUDIT_PASS"]
    wrong_frame = inventory()
    wrong_frame["topics"][0]["frame_ids"] = ["map"]
    audit = evaluated(wrong_frame)
    assert not audit["ACQUISITION_AUDIT_PASS"]
    assert audit["REVIEW_REQUIRED"] is True


@pytest.mark.parametrize("rate,passes", [(9.5, True), (10.5, True), (9.49, False), (10.51, False)])
def test_lidar_frequency_gate(rate: float, passes: bool) -> None:
    assert evaluated(inventory(lidar_rate=rate))["checks"]["lidar_rate"] is passes


@pytest.mark.parametrize("rate,passes", [(190.0, True), (210.0, True), (189.9, False), (210.1, False)])
def test_imu_frequency_gate(rate: float, passes: bool) -> None:
    assert evaluated(inventory(imu_rate=rate))["checks"]["imu_rate"] is passes


def test_required_pointcloud2_fields_and_finite_fraction() -> None:
    missing = inventory()
    missing["topics"][0]["fields"] = [{"name": name} for name in FIELDS if name != "timestamp"]
    assert not evaluated(missing)["checks"]["pointcloud2_fields"]
    assert not evaluated(finite_point_fraction=0.0)["checks"]["finite_points"]


def test_obvious_motion_audit_invalidates_bag_and_pair() -> None:
    moving = evaluated(
        imu_summary={
            "status": "PASS",
            "STATICITY_SCREEN": "MOTION_SUSPECTED",
            "large_spike_count": 4,
        }
    )
    assert moving["motion_audit_status"] == "MOTION_SUSPECTED"
    assert moving["large_spike_count"] == 4
    assert not moving["ACQUISITION_AUDIT_PASS"]
    query = dict(evaluated(role="QUERY"), start_time=130.0, end_time=145.0)
    pair = audit_pair(moving, query)
    assert pair["FORMAL_PAIR_VALID"] is False
    assert "MAP_AUDIT_FAIL" in pair["exclusion_reasons"]


def test_recording_plan_is_map_wait_query_and_dry_run_safe(tmp_path: Path) -> None:
    plan = build_recording_plan(
        batch="FMB1",
        scene_id="FMB1_R01",
        station_id="S01",
        output_dir=tmp_path,
        timestamp="20260820_101530",
        attempt=1,
    )
    assert Path(plan["map_path"]).name == "FMB1_R01_S01_MAP_20260820_101530.bag"
    assert Path(plan["query_path"]).name == "FMB1_R01_S01_QUERY_20260820_101602.bag"
    assert plan["steps"][:4] == [
        "VERIFY_LIVE_TOPICS", "RECORD_MAP_20S", "WAIT_12S", "RECORD_QUERY_15S"
    ]
    assert list(tmp_path.iterdir()) == []


def test_prior_attempt_is_never_overwritten(tmp_path: Path) -> None:
    prior = tmp_path / "FMB1_R01_S01_MAP_20260820_101530.bag"
    prior.write_bytes(b"retained failure")
    with pytest.raises(FormalBatchError, match="overwrite"):
        ensure_paths_do_not_exist((prior,))


def _pair_metadata(tmp_path: Path, attempt: int, passed: bool) -> Path:
    stamp = f"20260820_10{attempt:02d}30"
    path = tmp_path / f"attempt{attempt}.pair.json"
    payload = {
        "scene_id": "FMB1_R01",
        "station_id": "S01",
        "attempt": attempt,
        "map_audit": {
            "bag_path": str(tmp_path / f"FMB1_R01_S01_MAP_{stamp}.bag"),
            "SHA256": str(attempt) * 64,
        },
        "query_audit": {
            "bag_path": str(tmp_path / f"FMB1_R01_S01_QUERY_{stamp}.bag"),
            "SHA256": str(attempt) * 64,
        },
        "pair_audit": {"FORMAL_PAIR_VALID": passed},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_retains_failures_and_selects_first_audit_pass(tmp_path: Path) -> None:
    paths = [
        _pair_metadata(tmp_path, 1, False),
        _pair_metadata(tmp_path, 2, True),
        _pair_metadata(tmp_path, 3, True),
    ]
    manifest = build_bag_manifest(paths, tmp_path / "bag_manifest.json")
    assert len(manifest["attempts"]) == 3
    assert len(manifest["selected_pairs"]) == 1
    assert manifest["selected_pairs"][0]["attempt"] == 2
    assert manifest["selected_pairs"][0]["selection_rule"] == "FIRST_AUDIT_PASSING_ATTEMPT"
    assert manifest["selection_uses_icp"] is False
    audit_csv = (tmp_path / "acquisition_audit.csv").read_text(encoding="utf-8")
    assert audit_csv.startswith("scene_id,station_id,attempt,")
    assert audit_csv.count("\n") == 4
