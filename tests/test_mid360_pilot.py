from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phase_a_harness.mid360_pilot import NONFORMAL_MARKER
from phase_a_harness.mid360_pilot import bag_reader
from phase_a_harness.mid360_pilot.bag_reader import PilotBagError
from phase_a_harness.mid360_pilot.debug_registration import (
    build_shared_input_trials,
    mark_debug_artifact,
)
from phase_a_harness.mid360_pilot.imu_audit import audit_imu_messages
from phase_a_harness.mid360_pilot.lidar_adapter import (
    audit_lidar_messages,
    lidar_message_to_structured,
    pointcloud2_to_structured,
)
from phase_a_harness.mid360_pilot.pilot_geometry import compute_pilot_geometry
from phase_a_harness.mid360_pilot.pilot_pipeline import (
    DEFAULT_CONFIG_PATH,
    load_config,
    validate_runtime_root,
)
from phase_a_harness.mid360_pilot.split import (
    build_lineage,
    build_split_contract,
    partition_lidar_frames,
    select_query_frames,
)
from phase_a_harness.mid360_pilot.static_map import (
    build_static_target_map,
    deterministic_voxel_downsample,
    finite_range_filter,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    GEOMETRY_ONLY_FIELDS,
)


def _stamp(value: float) -> SimpleNamespace:
    return SimpleNamespace(to_sec=lambda: value)


def _field(name: str, offset: int, datatype: int, count: int = 1) -> SimpleNamespace:
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=count)


def _pointcloud(points: list[tuple[float, float, float, float]], *, padding: int = 0) -> SimpleNamespace:
    packed = b"".join(struct.pack("<ffff", *point) for point in points)
    if padding:
        packed += b"x" * padding
    return SimpleNamespace(
        _type="sensor_msgs/PointCloud2",
        _md5sum="pc2md5",
        header=SimpleNamespace(stamp=_stamp(100.0), frame_id="livox_frame"),
        is_bigendian=False,
        height=1,
        width=len(points),
        point_step=16,
        row_step=16 * len(points) + padding,
        fields=[
            _field("x", 0, 7),
            _field("y", 4, 7),
            _field("z", 8, 7),
            _field("intensity", 12, 7),
        ],
        data=packed,
    )


def _imu(timestamp: float, gyro: tuple[float, float, float], accel: tuple[float, float, float]) -> tuple[int, SimpleNamespace, float, float]:
    index = int(round(timestamp * 100))
    message = SimpleNamespace(
        _type="sensor_msgs/Imu",
        angular_velocity=SimpleNamespace(x=gyro[0], y=gyro[1], z=gyro[2]),
        linear_acceleration=SimpleNamespace(x=accel[0], y=accel[1], z=accel[2]),
    )
    return index, message, timestamp, timestamp


@pytest.fixture
def config() -> dict[str, object]:
    return load_config(DEFAULT_CONFIG_PATH)


def test_ros1_bag_format_detection(tmp_path: Path) -> None:
    bag = tmp_path / "x.bag"
    bag.write_bytes(b"#ROSBAG V2.0\n")
    assert bag_reader.detect_bag_format(bag) == "ROS1_BAG_V2.0"


def test_ros1_bag_topic_inventory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bag_path = tmp_path / "x.bag"
    bag_path.write_bytes(b"#ROSBAG V2.0\ncontent")
    lidar = _pointcloud([(1.0, 2.0, 3.0, 4.0)])
    imu = SimpleNamespace(
        _type="sensor_msgs/Imu",
        _md5sum="imumd5",
        header=SimpleNamespace(stamp=_stamp(100.1), frame_id="livox_frame"),
    )
    messages = {
        "/livox/lidar": [("/livox/lidar", lidar, _stamp(100.01))],
        "/livox/imu": [("/livox/imu", imu, _stamp(100.11))],
    }

    class FakeBag:
        def __init__(self, *_: object) -> None:
            pass

        def __enter__(self) -> "FakeBag":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def get_start_time(self) -> float:
            return 100.0

        def get_end_time(self) -> float:
            return 101.0

        def get_type_and_topic_info(self) -> SimpleNamespace:
            return SimpleNamespace(
                topics={
                    "/livox/lidar": SimpleNamespace(message_count=1, connections=1, frequency=10.0, msg_type="sensor_msgs/PointCloud2"),
                    "/livox/imu": SimpleNamespace(message_count=1, connections=1, frequency=200.0, msg_type="sensor_msgs/Imu"),
                }
            )

        def read_messages(self, topics: list[str]) -> list[tuple[str, object, object]]:
            return messages[topics[0]]

    monkeypatch.setattr(bag_reader, "_rosbag_module", lambda: SimpleNamespace(Bag=FakeBag))
    inventory = bag_reader.build_bag_inventory(bag_path)
    assert inventory["bag_format"] == "ROS1_BAG_V2.0"
    assert {row["topic"] for row in inventory["topics"]} == {"/livox/lidar", "/livox/imu"}


def test_lidar_message_type_detection() -> None:
    assert bag_reader.detect_lidar_message_type(_pointcloud([(1, 2, 3, 4)])) == "sensor_msgs/PointCloud2"
    with pytest.raises(PilotBagError):
        bag_reader.detect_lidar_message_type(SimpleNamespace(_type="unknown/Cloud"))


def test_pointcloud2_parser_preserves_fields_and_padding() -> None:
    message = _pointcloud([(1, 2, 3, 4), (5, 6, 7, 8)], padding=5)
    points = pointcloud2_to_structured(message)
    assert points.dtype.names == ("x", "y", "z", "intensity")
    np.testing.assert_allclose(points["z"], [3, 7])


def test_livox_custom_parser_when_present() -> None:
    point = SimpleNamespace(x=1.0, y=2.0, z=3.0, reflectivity=42, tag=7, line=2, offset_time=100)
    message = SimpleNamespace(_type="livox_ros_driver/CustomMsg", points=[point])
    parsed = lidar_message_to_structured(message)
    assert parsed.dtype.names == ("x", "y", "z", "intensity", "tag", "line", "offset_time")
    assert int(parsed["offset_time"][0]) == 100


def test_lidar_audit_timestamps_monotonic_and_finite() -> None:
    records = []
    for index, timestamp in enumerate((100.0, 100.1, 100.2)):
        message = _pointcloud([(1 + index, 2, 3, 4)])
        records.append((index, message, timestamp, timestamp))
    rows, summary = audit_lidar_messages(records)
    assert summary["timestamp_strictly_monotonic"] is True
    assert sum(row["finite_point_count"] for row in rows) == 3


def test_finite_range_filter_removes_zero_and_nonfinite() -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [np.nan, 0, 0], [300, 0, 0]])
    filtered = finite_range_filter(points, minimum_range_m=0.1, maximum_range_m=200.0)
    np.testing.assert_allclose(filtered, [[1, 0, 0]])


def test_imu_parsing_and_acceleration_is_not_converted(config: dict[str, object]) -> None:
    records = [
        _imu(i * 0.005, (0.01, 0.0, 0.0), (0.0, 0.0, 1.0))
        for i in range(20)
    ]
    rows, summary = audit_imu_messages(records, config)
    assert rows[0]["accel_z"] == 1.0
    assert summary["acceleration_conversion_applied"] is False
    assert summary["acceleration_unit_confirmed"] is False
    assert summary["STATICITY_SCREEN"] == "NO_OBVIOUS_MOTION"


def test_exact_pilot_split(config: dict[str, object]) -> None:
    timestamps = [100.0 + 0.1 * index for index in range(167)]
    contract = build_split_contract(timestamps, config)
    assert contract["map_interval_relative_seconds"] == [1.0, 7.0]
    assert contract["guard_gap_relative_seconds"] == [7.0, 9.0]
    assert contract["query_interval_relative_seconds"] == [9.0, 15.0]


def test_short_bag_split_is_rejected(config: dict[str, object]) -> None:
    with pytest.raises(PilotBagError, match="PILOT_SPLIT_NOT_APPLICABLE"):
        build_split_contract([0.0, 15.9], config)


def test_map_query_disjoint_and_guard_unused(config: dict[str, object]) -> None:
    timestamps = [100.0 + 0.1 * index for index in range(167)]
    contract = build_split_contract(timestamps, config)
    frames = [{"frame_index": index, "timestamp": timestamp} for index, timestamp in enumerate(timestamps)]
    partition = partition_lidar_frames(frames, contract)
    selected = select_query_frames(partition["query"], config["query_quantiles"])
    lineage = build_lineage(partition, selected)
    assert lineage["map_query_strictly_disjoint"] is True
    assert lineage["guard_gap_used"] is False
    assert not set(lineage["map_frame_indexes"]) & set(lineage["query_candidate_frame_indexes"])


def test_deterministic_ten_query_selection(config: dict[str, object]) -> None:
    candidates = [{"frame_index": index, "timestamp": float(index)} for index in range(60)]
    first = select_query_frames(candidates, config["query_quantiles"])
    second = select_query_frames(candidates, config["query_quantiles"])
    assert first == second
    assert len(first) == len({row["frame_index"] for row in first}) == 10


def test_static_map_does_not_call_registration(config: dict[str, object]) -> None:
    target, metadata = build_static_target_map(
        [np.asarray([[1.0, 0, 0], [1.01, 0, 0], [2.0, 0, 0]])], config
    )
    assert target.shape[1] == 3
    assert metadata["registration_called"] is False
    assert metadata["construction"] == "DIRECT_SAME_SENSOR_FRAME_MERGE_NO_REGISTRATION"


def test_target_map_is_deterministic() -> None:
    points = np.asarray([[1.01, 0, 0], [1.02, 0, 0], [-0.1, 1.0, 0], [1.2, 0, 0]])
    first = deterministic_voxel_downsample(points, 0.05)
    second = deterministic_voxel_downsample(points[::-1], 0.05)
    np.testing.assert_array_equal(first, second)


def test_geometry_only_fields_and_t0_identity(monkeypatch: pytest.MonkeyPatch, config: dict[str, object]) -> None:
    import phase_a_harness.mid360_pilot.pilot_geometry as geometry

    captured = []

    class Context:
        @classmethod
        def prepare(cls, _: np.ndarray) -> object:
            return object()

    def fake_compute(source: np.ndarray, transform: np.ndarray, *, context: object) -> dict[str, float]:
        captured.append(transform.copy())
        values = [100, 90, 0.2, 0.3, 0.5, 0.2, 0.3, 0.5, 2.5, 0.95]
        return dict(zip(GEOMETRY_ONLY_FIELDS, values))

    monkeypatch.setattr(geometry, "TargetGeometryContext", Context)
    monkeypatch.setattr(geometry, "compute_geometry_only_initial_metrics", fake_compute)
    queries = [np.ones((20, 3)) for _ in range(10)]
    metadata = [{"frame_index": index, "timestamp": float(index)} for index in range(10)]
    rows, summary = compute_pilot_geometry(queries, metadata, np.ones((100, 3)), config)
    assert all(np.array_equal(matrix, np.eye(4)) for matrix in captured)
    assert set(rows[0]) == {"selection_index", "frame_index", "timestamp", "query_finite_point_count", *GEOMETRY_ONLY_FIELDS}
    assert summary["registration_executed"] is False


def test_geometry_only_rows_do_not_expose_registration_fields(monkeypatch: pytest.MonkeyPatch, config: dict[str, object]) -> None:
    import phase_a_harness.mid360_pilot.pilot_geometry as geometry

    class Context:
        @classmethod
        def prepare(cls, _: np.ndarray) -> object:
            return object()

    metrics = dict(zip(GEOMETRY_ONLY_FIELDS, [100, 90, 0.2, 0.3, 0.5, 0.2, 0.3, 0.5, 2.5, 0.95]))
    monkeypatch.setattr(geometry, "TargetGeometryContext", Context)
    monkeypatch.setattr(geometry, "compute_geometry_only_initial_metrics", lambda *args, **kwargs: metrics)
    rows, _ = compute_pilot_geometry(
        [np.ones((20, 3)) for _ in range(10)],
        [{"frame_index": i, "timestamp": float(i)} for i in range(10)],
        np.ones((100, 3)),
        config,
    )
    forbidden = ("final", "turnover", "fitness", "solver", "displacement", "residual")
    assert not any(fragment in key for row in rows for key in row for fragment in forbidden)


def test_open3d_and_pcl_debug_share_identical_inputs() -> None:
    queries = [np.full((50, 3), float(index)) for index in range(10)]
    trials = build_shared_input_trials(queries, np.ones((100, 3)))
    assert len(trials) == 20
    for index in range(10):
        pair = [row for row in trials if row["query_index"] == index]
        assert len(pair) == 2
        assert pair[0]["source_array_sha256"] == pair[1]["source_array_sha256"]
        assert pair[0]["target_array_sha256"] == pair[1]["target_array_sha256"]


def test_pilot_artifacts_reject_frozen_experiment_directory(tmp_path: Path) -> None:
    with pytest.raises(PilotBagError):
        validate_runtime_root(tmp_path / "frozen_assets" / "pilot")


def test_pilot_artifacts_reject_repository_tree() -> None:
    with pytest.raises(PilotBagError):
        validate_runtime_root(DEFAULT_CONFIG_PATH.parent / "accidental-runtime")


def test_boreas_and_synthetic_names_rejected_for_pilot_output(tmp_path: Path) -> None:
    with pytest.raises(PilotBagError):
        validate_runtime_root(tmp_path / "real_data_boreas_stage1_v1")
    with pytest.raises(PilotBagError):
        validate_runtime_root(tmp_path / "synthetic_confirmatory")


def test_debug_artifacts_have_do_not_cite_marker() -> None:
    artifact = mark_debug_artifact({"status": "PASS"})
    assert artifact[NONFORMAL_MARKER] is True
    assert artifact["PILOT_ONLY"] is True
    assert artifact["MEASUREMENT_EVIDENCE"] is False


def test_config_is_pilot_only_and_no_formal_authority(config: dict[str, object]) -> None:
    assert config["flags"] == {
        "PILOT_ONLY": True,
        "FORMAL_REAL_DATA": False,
        "INDEPENDENT_MAP_QUERY_ACQUISITION": False,
        "MEASUREMENT_EVIDENCE": False,
    }
    serialized = json.dumps(config)
    assert "REAL_DATA_RUN_AUTHORIZED" not in serialized
    assert "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED" not in serialized
