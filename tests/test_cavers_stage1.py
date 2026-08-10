from __future__ import annotations

import inspect
import math
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phase_a_harness.real_data_preparation import cavers_stage1
from phase_a_harness.real_data_preparation.cavers_stage1 import (
    ALLOWED_UNCERTAINTY_TYPES,
    CaversStage1Error,
    MAX_STAGE1_FILE_BYTES,
    RangeClient,
    Stage1LargeFileDownloadForbidden,
    _topic_rows_from_metadata,
    _uncertainty_rows,
    assert_full_download_allowed,
    audit_gt_csv,
    audit_lidar_timestamps,
    parse_tf_csv,
    parse_zenodo_record,
    select_rig_lidar_transform,
)
from phase_a_harness.real_data_preparation.guard import (
    NoRegistrationGuard,
    RegistrationForbiddenError,
)
from phase_a_harness.real_data_preparation.stage1_gt_overlap import (
    FROZEN_STAGE1_OVERLAP_CONTRACT,
    compute_stage1_gt_overlap,
    rank_distinct_stage1_pairs,
    resample_gap_aware,
)
from phase_a_harness.real_data_preparation.transforms import (
    compose_world_sensor,
    transform_from_xyzw,
)


GT_HEADER = (
    "Timestamp,Frame_ID,Child_Frame_ID,PX,PY,PZ,QX,QY,QZ,QW,"
    "VX,VY,VZ,VROLL,VPITCH,VYAW\n"
)
TF_HEADER = "Timestamp,Frame_ID,Child_Frame_ID,TX,TY,TZ,QX,QY,QZ,QW\n"


def _zenodo_record() -> dict[str, Any]:
    sizes = list(range(1, 48))
    sizes.append(cavers_stage1.ZENODO_TOTAL_ARCHIVE_BYTES - sum(sizes))
    files = [
        {
            "key": f"archive_{index:02d}.zip",
            "size": sizes[index],
            "checksum": f"md5:{index:032x}",
            "links": {
                "self": f"https://zenodo.org/api/records/19367714/files/archive_{index:02d}.zip/content"
            },
        }
        for index in reversed(range(48))
    ]
    return {
        "conceptrecid": "19367713",
        "created": "2026-04-15T14:26:49Z",
        "doi": "10.5281/zenodo.19367714",
        "files": files,
        "id": 19367714,
        "metadata": {
            "license": {"id": "cc-by-4.0"},
            "publication_date": "2026-04-01",
            "title": "CAVERS",
            "version": "0.0.1",
        },
        "modified": "2026-04-16T08:49:19Z",
        "revision": 18,
        "state": "done",
        "status": "published",
        "submitted": True,
    }


def _write_gt(path: Path, rows: list[str]) -> Path:
    path.write_text(GT_HEADER + "".join(f"{row}\n" for row in rows), encoding="utf-8")
    return path


def _write_tf(path: Path, rows: list[str]) -> Path:
    path.write_text(TF_HEADER + "".join(f"{row}\n" for row in rows), encoding="utf-8")
    return path


def _trajectory(duration_s: float, *, offset_m: float = 0.0) -> np.ndarray:
    timestamps = np.round(np.arange(0.0, duration_s + 0.05, 0.1), 10)
    return np.column_stack(
        (
            timestamps,
            np.full(timestamps.size, offset_m),
            np.zeros(timestamps.size),
            np.zeros(timestamps.size),
        )
    )


def _eligible_pair(
    map_id: str,
    query_id: str,
    *,
    duration: float = 200.0,
    fraction: float = 0.8,
    intervals: int = 40,
    q95: float = 1.0,
) -> dict[str, Any]:
    return {
        "common_world_frame_proven": True,
        "eligible_nonoverlapping_5s_interval_count": intervals,
        "map_independent_6dof": True,
        "map_rig_lidar_transform_available": True,
        "map_sequence_id": map_id,
        "nearest_distance_q95_m": q95,
        "overlap_status": "PASS",
        "query_independent_6dof": True,
        "query_rig_lidar_transform_available": True,
        "query_sequence_id": query_id,
        "total_covered_duration_s": duration,
        "coverage_fraction": fraction,
    }


def test_zenodo_metadata_parsing_pins_record_and_sorts_inventory() -> None:
    parsed = parse_zenodo_record(_zenodo_record())

    assert parsed["record_id"] == 19367714
    assert parsed["doi"] == "10.5281/zenodo.19367714"
    assert parsed["revision"] == 18
    assert parsed["version"] == "0.0.1"
    assert parsed["license"] == "cc-by-4.0"
    assert parsed["file_count"] == 48
    assert parsed["total_archive_bytes"] == cavers_stage1.ZENODO_TOTAL_ARCHIVE_BYTES
    assert [row["archive_name"] for row in parsed["files"]] == sorted(
        row["archive_name"] for row in parsed["files"]
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("id", 1), ("doi", "10.5281/zenodo.1"), ("revision", 17)),
)
def test_zenodo_metadata_parsing_rejects_unpinned_release(
    field: str,
    replacement: object,
) -> None:
    record = _zenodo_record()
    record[field] = replacement

    with pytest.raises(CaversStage1Error, match=f"Zenodo {field} mismatch"):
        parse_zenodo_record(record)


def test_zenodo_metadata_parsing_rejects_duplicate_inventory_entry() -> None:
    record = _zenodo_record()
    record["files"][1]["key"] = record["files"][0]["key"]

    with pytest.raises(CaversStage1Error, match="invalid or duplicate"):
        parse_zenodo_record(record)


def test_500mb_download_guard_allows_exact_frozen_limit() -> None:
    assert assert_full_download_allowed(MAX_STAGE1_FILE_BYTES, MAX_STAGE1_FILE_BYTES) is None


def test_500mb_download_guard_rejects_one_byte_over_limit() -> None:
    with pytest.raises(
        Stage1LargeFileDownloadForbidden,
        match="STAGE1_LARGE_FILE_DOWNLOAD_FORBIDDEN",
    ):
        assert_full_download_allowed(MAX_STAGE1_FILE_BYTES + 1, MAX_STAGE1_FILE_BYTES)


def test_500mb_download_guard_cannot_be_relaxed_by_cli_limit() -> None:
    with pytest.raises(Stage1LargeFileDownloadForbidden, match="must be in"):
        assert_full_download_allowed(1, MAX_STAGE1_FILE_BYTES + 1)
    with pytest.raises(Stage1LargeFileDownloadForbidden, match="invalid range materialization"):
        RangeClient(maximum_materialized_member_bytes=MAX_STAGE1_FILE_BYTES + 1)


def test_range_client_rejects_ignored_range_before_reading_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[int | None] = []

    class Response:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int | None = None) -> bytes:
            reads.append(size)
            raise AssertionError("ignored Range response body must not be read")

    monkeypatch.setattr(cavers_stage1.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(cavers_stage1.time, "sleep", lambda _: None)
    client = RangeClient(maximum_materialized_member_bytes=1024)

    with pytest.raises(CaversStage1Error, match="did not honor exact 206"):
        client.get(
            "https://zenodo.org/archive.zip/content",
            0,
            9,
            purpose="test",
            expected_total_bytes=1000,
        )

    assert reads == []


def test_small_file_client_reads_only_expected_bytes_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[int | None] = []

    class Response:
        status = 200
        # Zenodo's official /container endpoint is chunked and therefore has
        # no Content-Length; the bounded read still prevents a large body.
        headers: dict[str, str] = {}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int | None = None) -> bytes:
            reads.append(size)
            return b"abc"

    monkeypatch.setattr(cavers_stage1.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    client = RangeClient(maximum_materialized_member_bytes=1024)

    assert client.get_small_file(
        "https://zenodo.org/file/container/member", purpose="test", expected_bytes=3
    ) == b"abc"
    assert reads == [4]


def test_small_file_client_retries_chunked_incomplete_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class Response:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int | None = None) -> bytes:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise cavers_stage1.http.client.IncompleteRead(b"ab")
            return b"abc"

    monkeypatch.setattr(cavers_stage1.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(cavers_stage1.time, "sleep", lambda _: None)
    client = RangeClient(maximum_materialized_member_bytes=1024)

    assert client.get_small_file(
        "https://zenodo.org/file/container/member", purpose="test", expected_bytes=3
    ) == b"abc"
    assert attempts == 2


def test_no_registration_guard_fails_closed_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", raising=False)

    with pytest.raises(RegistrationForbiddenError, match="is required"):
        NoRegistrationGuard().__enter__()


def test_no_registration_guard_blocks_icp_and_never_counts_an_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")

    with NoRegistrationGuard() as guard:
        with pytest.raises(RegistrationForbiddenError, match="forbidden registration process"):
            subprocess.run(["kiss_icp", "--help"], check=True)
        report = guard.attestation(tmp_path)

    assert report["other_registration_process_count"] == 1
    assert report["registration_execution_count"] == 0
    assert report["real_trial_result_count"] == 0


def test_transform_from_xyzw_treats_scalar_as_last_component() -> None:
    half = math.sqrt(0.5)
    transform = transform_from_xyzw(
        np.zeros(3),
        np.array([0.0, 0.0, half, half]),
    )

    np.testing.assert_allclose(transform[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def test_gt_audit_reports_xyzw_and_unit_quaternion(tmp_path: Path) -> None:
    path = _write_gt(
        tmp_path / "gt.csv",
        [
            "0,map,base_link,1,2,3,0,0,0,1,0,0,0,0,0,0",
            "0.01,map,base_link,1,2,3,0,0,0,1,0,0,0,0,0,0",
        ],
    )

    report, _ = audit_gt_csv(path, "loc_diablo_1")

    assert report["quaternion_order"] == "qx,qy,qz,qw (scalar-last)"
    assert report["quaternion_norm_pass"] is True
    assert report["max_quaternion_norm_error"] == 0.0


def test_gt_audit_nonunit_quaternion_fails_reference(tmp_path: Path) -> None:
    path = _write_gt(
        tmp_path / "gt.csv",
        [
            "0,map,base_link,1,2,3,0,0,0,2,0,0,0,0,0,0",
            "0.01,map,base_link,1,2,3,0,0,0,2,0,0,0,0,0,0",
        ],
    )

    report, _ = audit_gt_csv(path, "loc_diablo_1")

    assert report["quaternion_norm_pass"] is False
    assert report["status"] == "FAIL"


def test_gt_audit_requires_strict_timestamp_monotonicity(tmp_path: Path) -> None:
    path = _write_gt(
        tmp_path / "gt.csv",
        [
            "0,map,base_link,1,2,3,0,0,0,1,0,0,0,0,0,0",
            "2,map,base_link,1,2,3,0,0,0,1,0,0,0,0,0,0",
            "1,map,base_link,1,2,3,0,0,0,1,0,0,0,0,0,0",
        ],
    )

    report, _ = audit_gt_csv(path, "loc_diablo_1")

    assert report["strictly_monotonic"] is False
    assert report["status"] == "FAIL"


def test_gt_audit_estimates_rate_and_nanosecond_scale(tmp_path: Path) -> None:
    base = 1_745_929_251_000_000_000
    path = _write_gt(
        tmp_path / "gt.csv",
        [
            f"{base + offset},map,base_link,1,2,3,0,0,0,1,0,0,0,0,0,0"
            for offset in (0, 10_000_000, 20_000_000, 30_000_000)
        ],
    )

    report, trajectory = audit_gt_csv(path, "loc_diablo_1")

    assert report["timestamp_scale_to_seconds"] == 1e-9
    assert report["median_rate_hz"] == pytest.approx(100.0, rel=1e-5)
    assert report["duration_s"] == pytest.approx(0.03, rel=1e-5)
    assert trajectory.shape == (4, 4)


def test_gt_audit_detects_sequence_local_identity_reset(tmp_path: Path) -> None:
    path = _write_gt(
        tmp_path / "gt.csv",
        [
            "0,map,base_link,0,0,0,0,0,0,1,0,0,0,0,0,0",
            "0.01,map,base_link,0.1,0,0,0,0,0,1,0,0,0,0,0,0",
        ],
    )

    report, _ = audit_gt_csv(path, "loc_diablo_1")

    assert report["sequence_local_reset_detected"] is True


def test_gt_audit_does_not_call_nonzero_first_pose_a_reset(tmp_path: Path) -> None:
    path = _write_gt(
        tmp_path / "gt.csv",
        [
            "0,map,base_link,0.1,0,0,0,0,0,1,0,0,0,0,0,0",
            "0.01,map,base_link,0.2,0,0,0,0,0,1,0,0,0,0,0,0",
        ],
    )

    report, _ = audit_gt_csv(path, "loc_diablo_1")

    assert report["sequence_local_reset_detected"] is False


def test_common_world_evidence_does_not_promote_same_named_map_to_proof(tmp_path: Path) -> None:
    reports = []
    for index in range(2):
        path = _write_gt(
            tmp_path / f"gt_{index}.csv",
            [
                f"0,map,base_link,{index + 1},0,0,0,0,0,1,0,0,0,0,0,0",
                f"0.01,map,base_link,{index + 2},0,0,0,0,0,1,0,0,0,0,0,0",
            ],
        )
        reports.append(audit_gt_csv(path, f"sequence_{index}")[0])

    assert {tuple(row["world_frames"]) for row in reports} == {("map",)}
    assert not any(row["sequence_local_reset_detected"] for row in reports)
    producer_source = inspect.getsource(cavers_stage1.execute_cavers_stage1)
    assert "shared_calibration_session_proven = False" in producer_source
    assert (
        "shared_world = same_named_nonreset_frame and shared_calibration_session_proven"
        in producer_source
    )


def test_tf_parser_preserves_rig_to_velodyne_direction(tmp_path: Path) -> None:
    path = _write_tf(
        tmp_path / "tf.csv",
        ["0,base_link,velodyne,0.087,-0.11,0.17295,0,0,0,1"],
    )

    selected = select_rig_lidar_transform(parse_tf_csv(path))

    assert selected is not None
    assert selected["parent_frame"] == "base_link"
    assert selected["child_frame"] == "velodyne"
    assert selected["direction"] == "T_base_link_velodyne"
    assert selected["translation_m"] == [0.087, -0.11, 0.17295]
    assert selected["direction_test_pass"] is True


def test_tf_parser_accepts_only_bounded_export_rounding(tmp_path: Path) -> None:
    path = _write_tf(
        tmp_path / "tf.csv",
        ["0,base_link,velodyne,0.087,-0.11,0.17295,0,0,0,0.99999999"],
    )

    selected = select_rig_lidar_transform(parse_tf_csv(path))

    assert selected is not None
    assert selected["quaternion_raw_norm_error"] == pytest.approx(1e-8)
    assert selected["quaternion_xyzw"] == [0.0, 0.0, 0.0, 1.0]


def test_tf_parser_rejects_unbounded_quaternion_repair(tmp_path: Path) -> None:
    path = _write_tf(
        tmp_path / "tf.csv",
        ["0,base_link,velodyne,0.087,-0.11,0.17295,0,0,0,0.99"],
    )

    with pytest.raises(CaversStage1Error, match="export-rounding tolerance"):
        parse_tf_csv(path)


def test_reverse_velodyne_to_rig_transform_is_not_misused(tmp_path: Path) -> None:
    path = _write_tf(
        tmp_path / "tf.csv",
        ["0,velodyne,base_link,-0.087,0.11,-0.17295,0,0,0,1"],
    )

    assert select_rig_lidar_transform(parse_tf_csv(path)) is None


def test_t_map_lidar_composition_applies_rig_rotation_to_sensor_offset() -> None:
    half = math.sqrt(0.5)
    t_map_rig = transform_from_xyzw(
        np.array([10.0, 0.0, 0.0]),
        np.array([0.0, 0.0, half, half]),
    )
    t_rig_lidar = transform_from_xyzw(np.array([1.0, 0.0, 0.0]), np.array([0, 0, 0, 1]))

    t_map_lidar = compose_world_sensor(t_map_rig, t_rig_lidar)

    np.testing.assert_allclose(t_map_lidar[:3, 3], [10.0, 1.0, 0.0], atol=1e-12)


def test_missing_extrinsic_causes_transform_selection_failure(tmp_path: Path) -> None:
    path = _write_tf(
        tmp_path / "tf.csv",
        ["0,mocap,map,0,0,0,0,0,0,1"],
    )

    assert select_rig_lidar_transform(parse_tf_csv(path)) is None


def test_conflicting_extrinsics_fail_closed(tmp_path: Path) -> None:
    path = _write_tf(
        tmp_path / "tf.csv",
        [
            "0,base_link,velodyne,0.087,-0.11,0.17295,0,0,0,1",
            "1,base_link,velodyne,0.090,-0.11,0.17295,0,0,0,1",
        ],
    )

    assert select_rig_lidar_transform(parse_tf_csv(path)) is None


def test_lidar_timestamp_audit_checks_monotonicity_and_resolution(tmp_path: Path) -> None:
    path = tmp_path / "lidar.csv"
    path.write_text(
        "Timestamp,Filename\n"
        "1745929251000000000,one.pcd\n"
        "1745929251200000000,two.pcd\n",
        encoding="utf-8",
    )

    report = audit_lidar_timestamps(path)

    assert report["strictly_monotonic"] is True
    assert report["timestamp_resolution_s"] == 1e-9
    assert report["row_count"] == 2


def test_rosbag_metadata_parser_reads_gt_tf_and_lidar_topics(tmp_path: Path) -> None:
    path = tmp_path / "metadata.yaml"
    path.write_text(
        "rosbag2_bagfile_information:\n"
        "  topics_with_message_count:\n"
        "    - topic_metadata: {name: /spaceuma/optitrack/odom, type: nav_msgs/msg/Odometry}\n"
        "      message_count: 1200\n"
        "    - topic_metadata: {name: /tf_static, type: tf2_msgs/msg/TFMessage}\n"
        "      message_count: 1\n"
        "    - topic_metadata: {name: /spaceuma/velodyne_points, type: sensor_msgs/msg/PointCloud2}\n"
        "      message_count: 50\n",
        encoding="utf-8",
    )

    rows = _topic_rows_from_metadata(path)

    assert {row["name"] for row in rows} == {
        "/spaceuma/optitrack/odom",
        "/spaceuma/velodyne_points",
        "/tf_static",
    }


def test_time_uncertainty_unknown_is_never_encoded_as_zero() -> None:
    rows = _uncertainty_rows("reference", "extrinsic", "time")
    time_rows = [row for row in rows if "timestamp" in row["component"] or "synchronization" in row["component"]]

    assert time_rows
    assert all(row["value"] == "UNKNOWN" for row in time_rows)
    assert all(row["uncertainty_type"] == "UNKNOWN" for row in time_rows)
    assert all(row["value"] != 0 and row["value"] != "0" for row in time_rows)


def test_mm_accurate_claim_is_not_promoted_to_one_sigma() -> None:
    rows = _uncertainty_rows("reference", "extrinsic", "time")
    optitrack_rows = [row for row in rows if row["component"].startswith("OptiTrack")]

    assert len(optitrack_rows) == 2
    assert all(row["value"] == "UNKNOWN" for row in optitrack_rows)
    assert all(row["uncertainty_type"] == "UNKNOWN" for row in optitrack_rows)
    assert all(row["uncertainty_type"] != "1SIGMA" for row in optitrack_rows)
    assert all(row["uncertainty_type"] in ALLOWED_UNCERTAINTY_TYPES for row in rows)


def test_nominal_cad_transform_is_not_zero_uncertainty() -> None:
    rows = _uncertainty_rows("reference", "extrinsic", "time")
    extrinsic_rows = [row for row in rows if row["component"].startswith("CAD extrinsic")]

    assert len(extrinsic_rows) == 2
    assert all(row["value"] == "UNKNOWN" for row in extrinsic_rows)
    assert all(row["uncertainty_type"] == "UNKNOWN" for row in extrinsic_rows)


def test_gt_only_overlap_is_deterministic() -> None:
    map_trajectory = _trajectory(160.0)
    query_trajectory = _trajectory(160.0, offset_m=1.0)

    first = compute_stage1_gt_overlap(
        map_trajectory,
        query_trajectory,
        common_world_frame_proven=True,
    )
    second = compute_stage1_gt_overlap(
        map_trajectory,
        query_trajectory,
        common_world_frame_proven=True,
    )

    assert first == second
    assert first["overlap_status"] == "PASS"
    assert first["nearest_distance_median_m"] == pytest.approx(1.0)
    assert first["nearest_distance_q95_m"] == pytest.approx(1.0)


def test_gt_only_overlap_is_not_computable_without_proven_common_world() -> None:
    result = compute_stage1_gt_overlap(
        _trajectory(160.0),
        _trajectory(160.0),
        common_world_frame_proven=False,
    )

    assert result == {
        "failure_reason": "UNPROVEN_CROSS_SEQUENCE_FIXED_WORLD_FRAME",
        "overlap_status": "NOT_COMPUTABLE",
    }


def test_gap_aware_resampling_does_not_interpolate_across_recording_gap() -> None:
    first = _trajectory(2.0)
    second = _trajectory(2.0).copy()
    second[:, 0] += 10.0
    rows = resample_gap_aware(np.vstack((first, second)))

    assert set(rows[:, 4].astype(int)) == {0, 1}
    assert not np.any((rows[:, 0] > 2.0) & (rows[:, 0] < 10.0))


def test_same_sequence_pair_is_prohibited_even_if_overlap_row_failed() -> None:
    row = _eligible_pair("loc_diablo_1", "loc_diablo_1")
    row["overlap_status"] = "FAIL"

    with pytest.raises(ValueError, match="same-sequence"):
        rank_distinct_stage1_pairs([row])


def test_frozen_150_second_threshold_is_immutable() -> None:
    assert FROZEN_STAGE1_OVERLAP_CONTRACT.min_total_covered_duration_s == 150.0
    with pytest.raises(FrozenInstanceError):
        FROZEN_STAGE1_OVERLAP_CONTRACT.min_total_covered_duration_s = 149.0  # type: ignore[misc]


def test_frozen_060_coverage_threshold_is_immutable() -> None:
    assert FROZEN_STAGE1_OVERLAP_CONTRACT.min_coverage_fraction == 0.60
    with pytest.raises(FrozenInstanceError):
        FROZEN_STAGE1_OVERLAP_CONTRACT.min_coverage_fraction = 0.50  # type: ignore[misc]


def test_frozen_30_interval_threshold_is_immutable() -> None:
    assert FROZEN_STAGE1_OVERLAP_CONTRACT.min_eligible_nonoverlapping_5s_intervals == 30
    with pytest.raises(FrozenInstanceError):
        FROZEN_STAGE1_OVERLAP_CONTRACT.min_eligible_nonoverlapping_5s_intervals = 29  # type: ignore[misc]


def test_all_frozen_overlap_constants_match_preregistered_contract() -> None:
    contract = FROZEN_STAGE1_OVERLAP_CONTRACT

    assert contract.resample_rate_hz == 1.0
    assert contract.radius_m == 5.0
    assert contract.min_contiguous_covered_duration_s == 5.0
    assert contract.min_total_covered_duration_s == 150.0
    assert contract.min_coverage_fraction == 0.60
    assert contract.min_eligible_nonoverlapping_5s_intervals == 30


def test_overlap_below_frozen_duration_and_interval_gate_fails() -> None:
    result = compute_stage1_gt_overlap(
        _trajectory(148.0),
        _trajectory(148.0),
        common_world_frame_proven=True,
    )

    assert result["total_covered_duration_s"] < 150.0
    assert result["eligible_nonoverlapping_5s_interval_count"] < 30
    assert result["overlap_status"] == "FAIL"


def test_pair_ranking_is_deterministic_and_uses_frozen_precedence() -> None:
    rows = [
        _eligible_pair("map_z", "query_z", duration=201.0, fraction=0.61, intervals=30, q95=4.0),
        _eligible_pair("map_z", "query_b", duration=200.0, fraction=0.80, intervals=42, q95=1.0),
        _eligible_pair("map_a", "query_z", duration=200.0, fraction=0.80, intervals=42, q95=1.0),
        _eligible_pair("map_a", "query_a", duration=200.0, fraction=0.80, intervals=41, q95=0.1),
        _eligible_pair("map_b", "query_a", duration=200.0, fraction=0.79, intervals=99, q95=0.0),
    ]

    first = rank_distinct_stage1_pairs(rows)
    second = rank_distinct_stage1_pairs(reversed(rows))

    assert first == second
    assert [(row["map_sequence_id"], row["query_sequence_id"]) for row in first] == [
        ("map_z", "query_z"),
        ("map_a", "query_z"),
        ("map_z", "query_b"),
        ("map_a", "query_a"),
        ("map_b", "query_a"),
    ]
    assert [row["rank"] for row in first] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "missing_gate",
    (
        "map_independent_6dof",
        "query_independent_6dof",
        "common_world_frame_proven",
        "map_rig_lidar_transform_available",
        "query_rig_lidar_transform_available",
    ),
)
def test_pair_ranking_excludes_any_row_missing_a_hard_gate(missing_gate: str) -> None:
    row = _eligible_pair("map", "query")
    row[missing_gate] = False

    assert rank_distinct_stage1_pairs([row]) == []


def test_registration_attestation_count_stays_zero_on_clean_stage1_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    (tmp_path / "audit.json").write_text(
        '{"point_cloud_or_registration_consulted": false, "registration_execution_count": 0}\n',
        encoding="utf-8",
    )

    with NoRegistrationGuard() as guard:
        report = guard.attestation(tmp_path)

    assert report["open3d_registration_call_count"] == 0
    assert report["pcl_cli_invocation_count"] == 0
    assert report["other_registration_process_count"] == 0
    assert report["estimated_transform_count"] == 0
    assert report["registration_execution_count"] == 0
    assert report["real_trial_result_count"] == 0
    assert report["pass"] is True
