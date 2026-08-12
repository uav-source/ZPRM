from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from phase_a_harness.real_data_preparation.boreas_v2_stage2_preprocessing import (
    BoreasLidarPoseIndex,
    BoreasStage2PreprocessingError,
    MAXIMUM_RANGE_M,
    MINIMUM_RANGE_M,
    MOTION_COMPENSATION_BIN_COUNT,
    POSE_HEADER,
    PRIMARY_MAP_SEQUENCE_ID,
    PRIMARY_QUERY_SEQUENCE_ID,
    SOURCE_VOXEL_SIZE_M,
    TARGET_VOXEL_SIZE_M,
    boreas_microseconds_to_float_seconds,
    canonical_source_npy_bytes,
    canonical_t_reference_npy_bytes,
    compose_t_enu_lidar,
    decode_authenticated_boreas_velodyne_file,
    decode_boreas_velodyne_payload,
    deskew_official_21_bin_constant_body_twist,
    deterministic_voxel_centroids,
    parse_primary_lidar_object_key,
    preprocess_primary_boreas_scan,
    target_map_voxel_rule,
    transform_preprocessed_map_scan_to_enu_ref,
)


REPOSITORY = Path(__file__).resolve().parents[1]
TIMESTAMP_US = 1_600_000_000_000_000
HASH_A = "a" * 64


def _object_key(sequence_id: str, timestamp_us: int = TIMESTAMP_US) -> str:
    return f"{sequence_id}/lidar/{timestamp_us}.bin"


def _payload(rows: list[list[float]]) -> bytes:
    return np.asarray(rows, dtype="<f4").tobytes(order="C")


def _pose_row(
    timestamp_us: int,
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity_enu: tuple[float, float, float] = (0.0, 0.0, 0.0),
    roll: float = 0.0,
    pitch: float = 0.0,
    heading: float = 0.0,
    angular_velocity_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[object]:
    wx, wy, wz = angular_velocity_xyz
    return [
        timestamp_us,
        *translation,
        *velocity_enu,
        roll,
        pitch,
        heading,
        wz,
        wy,
        wx,
    ]


def _pose_index(
    tmp_path: Path,
    *,
    sequence_id: str,
    rows: list[list[object]] | None = None,
) -> BoreasLidarPoseIndex:
    if rows is None:
        rows = [
            _pose_row(TIMESTAMP_US),
            _pose_row(TIMESTAMP_US + 100_000),
        ]
    path = tmp_path / f"{sequence_id}-lidar_poses.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(POSE_HEADER)
        writer.writerows(rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return BoreasLidarPoseIndex.from_csv(
        path, sequence_id=sequence_id, expected_sha256=digest
    )


def test_primary_object_key_is_exact_and_role_bounded() -> None:
    assert parse_primary_lidar_object_key(_object_key(PRIMARY_MAP_SEQUENCE_ID)) == (
        PRIMARY_MAP_SEQUENCE_ID,
        TIMESTAMP_US,
    )
    with pytest.raises(BoreasStage2PreprocessingError, match="PRIMARY_PAIR"):
        parse_primary_lidar_object_key(
            "boreas-2021-01-01-00-00/lidar/1600000000000000.bin"
        )
    with pytest.raises(BoreasStage2PreprocessingError, match="lidar"):
        parse_primary_lidar_object_key(
            f"{PRIMARY_MAP_SEQUENCE_ID}/radar/{TIMESTAMP_US}.bin"
        )


def test_velodyne_decode_uses_six_little_endian_float32_fields() -> None:
    rows = [
        [1.0, 2.0, 3.0, 4.0, 5.0, -0.05],
        [6.0, 7.0, 8.0, 9.0, 10.0, 0.05],
    ]
    decoded = decode_boreas_velodyne_payload(
        _payload(rows), object_key=_object_key(PRIMARY_QUERY_SEQUENCE_ID)
    )
    assert decoded.fields.dtype == np.dtype("<f8")
    assert decoded.fields.flags.c_contiguous
    assert decoded.fields.flags.writeable is False
    assert np.array_equal(decoded.fields[:, :5], np.asarray(rows)[:, :5])
    assert np.allclose(
        decoded.point_timestamps_s - float(TIMESTAMP_US) * 1.0e-6,
        [-0.05, 0.05],
        atol=2.0e-7,
    )


def test_pinned_pyboreas_timestamp_rounding_controls_deskew_bins(
    tmp_path: Path,
) -> None:
    # This real allowlist timestamp is one of the cases where the tempting
    # ``float(us) * 1e-6`` spelling differs from pinned pyboreas micro_to_sec by
    # one float64 ULP at Unix-epoch magnitude.
    timestamp_us = 1_636_901_260_534_912
    reference_time = float(np.round(timestamp_us / 1.0e6, 6))
    old_reference_time = float(timestamp_us) * 1.0e-6
    assert reference_time != old_reference_time
    assert boreas_microseconds_to_float_seconds(timestamp_us) == reference_time

    offsets = np.linspace(-0.05, 0.05, 2001, dtype="<f4")
    rows = np.zeros((offsets.shape[0], 6), dtype="<f4")
    rows[:, 0] = 2.0
    rows[:, 3] = 1.0
    rows[:, 4] = 1.0
    rows[:, 5] = offsets
    decoded = decode_boreas_velodyne_payload(
        rows.tobytes(order="C"),
        object_key=_object_key(PRIMARY_MAP_SEQUENCE_ID, timestamp_us),
    )
    assert np.array_equal(
        decoded.point_timestamps_s,
        reference_time + offsets.astype(np.float64),
    )

    index = _pose_index(
        tmp_path,
        sequence_id=PRIMARY_MAP_SEQUENCE_ID,
        rows=[
            _pose_row(timestamp_us, velocity_enu=(25.0, 0.0, 0.0)),
            _pose_row(timestamp_us + 100_000),
        ],
    )
    result = preprocess_primary_boreas_scan(
        decoded, pose_index=index, role="TARGET_MAP"
    )

    # Independent transcription of the pinned PointCloud.remove_motion sorted
    # path for pure x translation: 21 bin times, 20 left-endpoint intervals,
    # final boundary extended by 1 us.
    official_times = reference_time + offsets.astype(np.float64)
    minimum_time = float(official_times[0])
    maximum_time = float(official_times[-1])
    delta = (maximum_time - minimum_time) / 20.0
    bin_times = np.asarray(
        [minimum_time + index * delta for index in range(21)], dtype=np.float64
    )
    boundaries = bin_times.copy()
    boundaries[-1] += 1.0e-6
    expected = np.array(rows[:, :3], dtype=np.float64, order="C", copy=True)
    for index_value in range(20):
        selected = (official_times >= boundaries[index_value]) & (
            official_times < boundaries[index_value + 1]
        )
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = (bin_times[index_value] - reference_time) * 25.0
        expected[selected] = (
            expected[selected] @ transform[:3, :3].T + transform[:3, 3]
        )
    assert np.array_equal(result.points_xyz, expected)

    old_times = old_reference_time + offsets.astype(np.float64)
    old_result = deskew_official_21_bin_constant_body_twist(
        rows[:, :3],
        old_times,
        scan_reference_timestamp_s=old_reference_time,
        body_rate_lidar=np.asarray([25.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert not np.array_equal(old_result, expected)


@pytest.mark.parametrize("payload", [b"", b"\x00" * 23, b"\x00" * 25])
def test_velodyne_decode_rejects_bad_stride(payload: bytes) -> None:
    with pytest.raises(BoreasStage2PreprocessingError, match="24-byte"):
        decode_boreas_velodyne_payload(
            payload, object_key=_object_key(PRIMARY_QUERY_SEQUENCE_ID)
        )


def test_file_decode_requires_authenticated_size_and_sha(tmp_path: Path) -> None:
    payload = _payload(
        [[2.0, 0.0, 0.0, 1.0, 1.0, -0.05], [2.0, 0.0, 0.0, 1.0, 1.0, 0.05]]
    )
    path = tmp_path / "temporary.bin"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    decoded = decode_authenticated_boreas_velodyne_file(
        path,
        object_key=_object_key(PRIMARY_MAP_SEQUENCE_ID),
        expected_size_bytes=len(payload),
        expected_sha256=digest,
    )
    assert decoded.point_count == 2
    with pytest.raises(BoreasStage2PreprocessingError, match="size"):
        decode_authenticated_boreas_velodyne_file(
            path,
            object_key=_object_key(PRIMARY_MAP_SEQUENCE_ID),
            expected_size_bytes=len(payload) + 1,
            expected_sha256=digest,
        )
    with pytest.raises(BoreasStage2PreprocessingError, match="SHA"):
        decode_authenticated_boreas_velodyne_file(
            path,
            object_key=_object_key(PRIMARY_MAP_SEQUENCE_ID),
            expected_size_bytes=len(payload),
            expected_sha256="0" * 64,
        )


def test_file_decode_rejects_symlink(tmp_path: Path) -> None:
    payload = _payload(
        [[2.0, 0.0, 0.0, 1.0, 1.0, -0.05], [2.0, 0.0, 0.0, 1.0, 1.0, 0.05]]
    )
    target = tmp_path / "target.bin"
    target.write_bytes(payload)
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(BoreasStage2PreprocessingError, match="safely open"):
        decode_authenticated_boreas_velodyne_file(
            link,
            object_key=_object_key(PRIMARY_MAP_SEQUENCE_ID),
            expected_size_bytes=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_exact_pose_index_and_body_rate_are_official_convention(tmp_path: Path) -> None:
    index = _pose_index(
        tmp_path,
        sequence_id=PRIMARY_QUERY_SEQUENCE_ID,
        rows=[
            _pose_row(
                TIMESTAMP_US,
                translation=(10.0, 20.0, 30.0),
                velocity_enu=(1.0, 2.0, 3.0),
                angular_velocity_xyz=(4.0, 5.0, 6.0),
            ),
            _pose_row(TIMESTAMP_US + 100_000),
        ],
    )
    pose = index.exact_pose(TIMESTAMP_US)
    assert np.array_equal(pose.t_enu_lidar[:3, 3], [10.0, 20.0, 30.0])
    assert np.array_equal(pose.body_rate_lidar, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert index.maximum_native_gap_s == pytest.approx(0.1)
    with pytest.raises(BoreasStage2PreprocessingError, match="exact"):
        index.exact_pose(TIMESTAMP_US + 1)


def test_pose_index_rejects_sha_mismatch_and_large_gap(tmp_path: Path) -> None:
    path = tmp_path / "lidar_poses.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(POSE_HEADER)
        writer.writerow(_pose_row(TIMESTAMP_US))
        writer.writerow(_pose_row(TIMESTAMP_US + 300_000))
    with pytest.raises(BoreasStage2PreprocessingError, match="SHA"):
        BoreasLidarPoseIndex.from_csv(
            path,
            sequence_id=PRIMARY_MAP_SEQUENCE_ID,
            expected_sha256="0" * 64,
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(BoreasStage2PreprocessingError, match="0.20"):
        BoreasLidarPoseIndex.from_csv(
            path,
            sequence_id=PRIMARY_MAP_SEQUENCE_ID,
            expected_sha256=digest,
        )


def test_official_21_bin_deskew_uses_left_bin_endpoints() -> None:
    points = np.asarray([[2.0, 0.0, 0.0]] * 3, dtype=np.float64)
    timestamps = np.asarray([99.95, 100.0, 100.05], dtype=np.float64)
    output = deskew_official_21_bin_constant_body_twist(
        points,
        timestamps,
        scan_reference_timestamp_s=100.0,
        body_rate_lidar=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert MOTION_COMPENSATION_BIN_COUNT == 21
    # The sorted pyboreas path assigns the tmax point to interval 19, whose
    # transform is evaluated at +0.045 s, not at the unused 21st transform.
    assert np.allclose(output[:, 0], [1.95, 2.0, 2.045], atol=1.0e-14)
    assert output.dtype == np.dtype("<f8")
    assert output.flags.c_contiguous


def test_official_deskew_rejects_unsorted_and_degenerate_timestamps() -> None:
    points = np.asarray([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(BoreasStage2PreprocessingError, match="nondecreasing"):
        deskew_official_21_bin_constant_body_twist(
            points,
            np.asarray([1.0, 0.0]),
            scan_reference_timestamp_s=0.5,
            body_rate_lidar=np.zeros(6),
        )
    with pytest.raises(BoreasStage2PreprocessingError, match="degenerate"):
        deskew_official_21_bin_constant_body_twist(
            points,
            np.asarray([1.0, 1.0]),
            scan_reference_timestamp_s=1.0,
            body_rate_lidar=np.zeros(6),
        )


def test_voxel_centroids_use_fixed_input_sum_and_lexicographic_keys() -> None:
    points = np.asarray(
        [[0.19, 0.0, 0.0], [-0.01, 0.0, 0.0], [0.11, 0.0, 0.0]],
        dtype=np.float64,
    )
    result = deterministic_voxel_centroids(
        points, voxel_size_m=0.10, origin_xyz_m=(0.0, 0.0, 0.0)
    )
    assert np.allclose(result[:, 0], [-0.01, 0.15])
    assert result.flags.writeable is False


def test_voxel_centroids_do_not_reassociate_nonassociative_sums() -> None:
    # All three points share one deliberately huge voxel.  Sequential raw-point
    # accumulation is (1e16 + -1e16) + 1 == 1; a pairwise/reassociated reducer
    # is allowed to produce a different result and therefore is not acceptable.
    points = np.asarray(
        [[1.0e16, 0.0, 0.0], [-1.0e16, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    result = deterministic_voxel_centroids(
        points, voxel_size_m=1.0e20, origin_xyz_m=(-2.0e16, 0.0, 0.0)
    )
    assert np.array_equal(result, [[1.0 / 3.0, 0.0, 0.0]])


def test_query_preprocessing_filters_then_deskews_then_source_voxelizes(
    tmp_path: Path,
) -> None:
    rows = [
        [np.nan, 0.0, 0.0, 1.0, 1.0, -0.06],
        [0.5, 0.0, 0.0, 1.0, 1.0, -0.05],
        [2.01, 0.0, 0.0, np.nan, np.nan, -0.04],
        [2.09, 0.0, 0.0, 1.0, 1.0, 0.04],
        [81.0, 0.0, 0.0, 1.0, 1.0, 0.05],
    ]
    decoded = decode_boreas_velodyne_payload(
        _payload(rows), object_key=_object_key(PRIMARY_QUERY_SEQUENCE_ID)
    )
    index = _pose_index(tmp_path, sequence_id=PRIMARY_QUERY_SEQUENCE_ID)
    result = preprocess_primary_boreas_scan(decoded, pose_index=index, role="QUERY")
    assert MINIMUM_RANGE_M == 1.0
    assert MAXIMUM_RANGE_M == 80.0
    assert SOURCE_VOXEL_SIZE_M == 0.10
    assert result.raw_point_count == 5
    assert result.nonfinite_excluded_count == 1
    assert result.range_excluded_count == 2
    assert result.post_filter_point_count == 2
    assert result.source_voxel_reduced_count == 1
    assert result.points_xyz.shape == (1, 3)
    # NaN intensity and laser-id are ignored: only xyz and point time gate geometry.
    assert np.allclose(result.points_xyz[0], [2.05, 0.0, 0.0])


def test_range_filter_is_inclusive_at_frozen_boundaries(tmp_path: Path) -> None:
    decoded = decode_boreas_velodyne_payload(
        _payload(
            [
                [MINIMUM_RANGE_M, 0.0, 0.0, 1.0, 1.0, -0.05],
                [MAXIMUM_RANGE_M, 0.0, 0.0, 1.0, 1.0, 0.05],
            ]
        ),
        object_key=_object_key(PRIMARY_QUERY_SEQUENCE_ID),
    )
    result = preprocess_primary_boreas_scan(
        decoded,
        pose_index=_pose_index(tmp_path, sequence_id=PRIMARY_QUERY_SEQUENCE_ID),
        role="QUERY",
    )
    assert result.range_excluded_count == 0
    assert result.points_xyz.shape[0] == 2


def test_map_preprocessing_does_not_apply_source_voxel_and_transforms_to_enu(
    tmp_path: Path,
) -> None:
    decoded = decode_boreas_velodyne_payload(
        _payload(
            [
                [2.01, 0.0, 0.0, 1.0, 1.0, -0.05],
                [2.09, 0.0, 0.0, 1.0, 1.0, 0.05],
            ]
        ),
        object_key=_object_key(PRIMARY_MAP_SEQUENCE_ID),
    )
    index = _pose_index(
        tmp_path,
        sequence_id=PRIMARY_MAP_SEQUENCE_ID,
        rows=[
            _pose_row(TIMESTAMP_US, translation=(10.0, 20.0, 30.0)),
            _pose_row(TIMESTAMP_US + 100_000),
        ],
    )
    result = preprocess_primary_boreas_scan(decoded, pose_index=index, role="TARGET_MAP")
    assert result.points_xyz.shape == (2, 3)
    assert result.source_voxel_reduced_count == 0
    enu = transform_preprocessed_map_scan_to_enu_ref(result)
    assert np.allclose(enu, result.points_xyz + [10.0, 20.0, 30.0])


def test_query_cannot_contribute_to_target_map(tmp_path: Path) -> None:
    decoded = decode_boreas_velodyne_payload(
        _payload(
            [[2.0, 0.0, 0.0, 1.0, 1.0, -0.05], [2.1, 0.0, 0.0, 1.0, 1.0, 0.05]]
        ),
        object_key=_object_key(PRIMARY_QUERY_SEQUENCE_ID),
    )
    query = preprocess_primary_boreas_scan(
        decoded,
        pose_index=_pose_index(tmp_path, sequence_id=PRIMARY_QUERY_SEQUENCE_ID),
        role="QUERY",
    )
    with pytest.raises(BoreasStage2PreprocessingError, match="PRIMARY map"):
        transform_preprocessed_map_scan_to_enu_ref(query)


def test_composition_direction_is_enu_applanix_then_applanix_lidar() -> None:
    enu_applanix = np.eye(4)
    enu_applanix[0, 3] = 10.0
    applanix_lidar = np.eye(4)
    applanix_lidar[1, 3] = 2.0
    result = compose_t_enu_lidar(enu_applanix, applanix_lidar)
    assert np.array_equal(result[:3, 3], [10.0, 2.0, 0.0])


def test_target_voxel_rule_is_stage2_preregistered_global_enu() -> None:
    rule = target_map_voxel_rule(preprocessing_contract_sha256=HASH_A)
    assert TARGET_VOXEL_SIZE_M == 0.10
    assert rule.voxel_size_m == 0.10
    assert rule.origin_xyz_m == (0.0, 0.0, 0.0)
    assert rule.parameter_authority == "STAGE2_PREREGISTRATION"
    assert rule.scientific_contract_sha256 == HASH_A


def test_canonical_source_and_reference_are_deterministic_f64_npy(
    tmp_path: Path,
) -> None:
    decoded = decode_boreas_velodyne_payload(
        _payload(
            [[2.0, 0.0, 0.0, 1.0, 1.0, -0.05], [2.2, 0.0, 0.0, 1.0, 1.0, 0.05]]
        ),
        object_key=_object_key(PRIMARY_QUERY_SEQUENCE_ID),
    )
    query = preprocess_primary_boreas_scan(
        decoded,
        pose_index=_pose_index(tmp_path, sequence_id=PRIMARY_QUERY_SEQUENCE_ID),
        role="QUERY",
    )
    source_first = canonical_source_npy_bytes(query)
    source_second = canonical_source_npy_bytes(query)
    reference_first = canonical_t_reference_npy_bytes(query)
    assert source_first == source_second
    assert hashlib.sha256(source_first).hexdigest() == hashlib.sha256(source_second).hexdigest()
    source = np.load(__import__("io").BytesIO(source_first), allow_pickle=False)
    reference = np.load(__import__("io").BytesIO(reference_first), allow_pickle=False)
    assert source.dtype == np.dtype("<f8") and source.flags.c_contiguous
    assert reference.dtype == np.dtype("<f8") and reference.shape == (4, 4)


def test_preprocessing_contract_self_hash_and_code_bindings() -> None:
    path = REPOSITORY / "protocols/boreas_v2_stage2_preprocessing_contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    supplied = contract["contract_payload_sha256"]
    unsigned = {key: value for key, value in contract.items() if key != "contract_payload_sha256"}
    actual = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert supplied == actual
    assert contract["contract_status"] == "FROZEN"
    implementation = contract["implementation_bindings"]["production_preprocessing"]
    source = REPOSITORY / implementation["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == implementation["file_sha256"]
    assert contract["execution_state_at_freeze"] == {
        "geometry_metric_execution_count": 0,
        "lidar_payload_download_count": 0,
        "registration_execution_count": 0,
    }
    assert contract["uncertainty_policy"]["unknown_imputed_as_zero"] is False
    assert "UNKNOWN" in set(contract["uncertainty_policy"]["numeric_status"].values())
