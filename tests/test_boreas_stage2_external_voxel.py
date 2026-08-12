from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from phase_a_harness.real_data_preparation.boreas_stage2_external_voxel import (
    AuthenticatedReplayRange,
    ExternalVoxelReductionError,
    build_external_target_map,
    compile_external_voxel_reducer,
    default_reducer_source,
    verify_external_target_map_replay,
    write_range_descriptor,
)
from phase_a_harness.real_data_preparation.streaming_target_map import (
    CENTROID_RULE,
    MapScan,
    VoxelRule,
    build_target_map_batch,
)


class RecordingDiskGate:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def before_materialization(
        self, projected_write_bytes: int, *, artifact_id: str
    ) -> None:
        self.calls.append((projected_write_bytes, artifact_id))


@pytest.fixture(scope="module")
def reducer_binary(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    root = tmp_path_factory.mktemp("boreas-stage2-external-voxel")
    report = compile_external_voxel_reducer(
        source=default_reducer_source(), output=root / "voxel-reduce"
    )
    return Path(report["binary_path"]), str(report["binary_sha256"])


def _fixture(tmp_path: Path) -> tuple[Path, list[AuthenticatedReplayRange], list[np.ndarray]]:
    arrays = [
        np.asarray(
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [1.01, 0.0, 0.0]],
            dtype="<f8",
        ),
        np.asarray(
            [[-0.1, 0.0, 0.0], [1.09, 0.0, 0.0], [2.0, 2.0, 2.0]],
            dtype="<f8",
        ),
    ]
    # Fixed capacities deliberately leave authenticated padding/gaps.  Only
    # point_count rows are used by the reducer.
    capacities = [5, 4]
    replay = tmp_path / "replay.bin"
    with replay.open("wb") as stream:
        for array, capacity in zip(arrays, capacities):
            stream.write(array.tobytes(order="C"))
            stream.write(b"\0" * ((capacity - len(array)) * 24))
    ranges: list[AuthenticatedReplayRange] = []
    offset = 0
    for array, capacity in zip(arrays, capacities):
        payload = array.tobytes(order="C")
        ranges.append(
            AuthenticatedReplayRange(
                byte_offset=offset,
                point_count=len(array),
                transformed_xyz_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        offset += capacity * 24
    return replay, ranges, arrays


def _rule() -> VoxelRule:
    return VoxelRule(
        voxel_size_m=0.1,
        representative_rule=CENTROID_RULE,
        origin_xyz_m=(0.0, 0.0, 0.0),
        parameter_authority="STAGE2_PREREGISTRATION",
        scientific_contract_sha256="a" * 64,
    )


def test_external_reducer_is_byte_exact_with_python_specification(
    tmp_path: Path, reducer_binary: tuple[Path, str]
) -> None:
    replay, ranges, arrays = _fixture(tmp_path)
    descriptor, descriptor_sha = write_range_descriptor(tmp_path / "ranges.tsv", ranges)
    binary, binary_sha = reducer_binary
    gate = RecordingDiskGate()
    result = build_external_target_map(
        replay_path=replay,
        range_descriptor_path=descriptor,
        range_descriptor_sha256=descriptor_sha,
        reducer_binary=binary,
        reducer_binary_sha256=binary_sha,
        voxel_rule=_rule(),
        target_points_path=tmp_path / "target.npy",
        max_voxels=100,
        disk_gate=gate,
        io_chunk_rows=2,
    )
    scans = [
        MapScan(
            ordinal=index,
            object_key=f"map/{index}.bin",
            points_xyz=array,
            reference_from_sensor=np.eye(4, dtype="<f8"),
            remote_size_bytes=array.size * 8,
            etag=f"etag-{index}",
            last_modified="2026-01-01T00:00:00Z",
            gt_sha256="1" * 64,
            calibration_sha256="2" * 64,
            object_sha256=str(index + 3) * 64,
        )
        for index, array in enumerate(arrays)
    ]
    expected = build_target_map_batch(scans, _rule())
    actual = np.load(result.target_points_path, allow_pickle=False)
    assert actual.dtype == np.dtype("<f8")
    assert actual.flags.c_contiguous
    assert actual.tobytes(order="C") == expected.points_xyz.tobytes(order="C")
    assert result.target_points_file_sha256 == expected.target_map_sha256
    assert result.point_count == len(expected.points_xyz)
    assert gate.calls == [
        (
            result.point_count * 24 + 4096,
            "TARGET_MAP_CANONICAL_NPY:target_maps/<pending-sha256>/target_points.npy",
        )
    ]

    verified = verify_external_target_map_replay(
        replay_path=replay,
        range_descriptor_path=descriptor,
        range_descriptor_sha256=descriptor_sha,
        reducer_binary=binary,
        reducer_binary_sha256=binary_sha,
        voxel_rule=_rule(),
        target_points_path=result.target_points_path,
        target_points_file_sha256=result.target_points_file_sha256,
        target_point_count=result.point_count,
        max_voxels=100,
        memory_safety_margin_bytes=1024,
        available_memory_provider=lambda: 10**12,
    )
    assert verified.verification_status == "PASS_EXACT_TARGET_NPY_REPLAY"
    assert verified.compared_npy_bytes == result.target_points_path.stat().st_size


def test_external_replay_verifier_rejects_resigned_target_and_memory_shortfall(
    tmp_path: Path, reducer_binary: tuple[Path, str]
) -> None:
    replay, ranges, _ = _fixture(tmp_path)
    descriptor, descriptor_sha = write_range_descriptor(tmp_path / "ranges.tsv", ranges)
    binary, binary_sha = reducer_binary
    built = build_external_target_map(
        replay_path=replay,
        range_descriptor_path=descriptor,
        range_descriptor_sha256=descriptor_sha,
        reducer_binary=binary,
        reducer_binary_sha256=binary_sha,
        voxel_rule=_rule(),
        target_points_path=tmp_path / "target.npy",
        max_voxels=100,
        disk_gate=RecordingDiskGate(),
    )
    target = np.load(built.target_points_path, allow_pickle=False).copy()
    target[0, 0] += 1.0
    resigned = tmp_path / "resigned-target.npy"
    with resigned.open("wb") as stream:
        np.lib.format.write_array(
            stream,
            np.ascontiguousarray(target, dtype="<f8"),
            version=(1, 0),
            allow_pickle=False,
        )
    resigned_sha = hashlib.sha256(resigned.read_bytes()).hexdigest()
    with pytest.raises(ExternalVoxelReductionError, match="target replay verification failed"):
        verify_external_target_map_replay(
            replay_path=replay,
            range_descriptor_path=descriptor,
            range_descriptor_sha256=descriptor_sha,
            reducer_binary=binary,
            reducer_binary_sha256=binary_sha,
            voxel_rule=_rule(),
            target_points_path=resigned,
            target_points_file_sha256=resigned_sha,
            target_point_count=built.point_count,
            max_voxels=100,
            memory_safety_margin_bytes=0,
            available_memory_provider=lambda: 10**12,
        )
    with pytest.raises(ExternalVoxelReductionError, match="insufficient live memory"):
        verify_external_target_map_replay(
            replay_path=replay,
            range_descriptor_path=descriptor,
            range_descriptor_sha256=descriptor_sha,
            reducer_binary=binary,
            reducer_binary_sha256=binary_sha,
            voxel_rule=_rule(),
            target_points_path=built.target_points_path,
            target_points_file_sha256=built.target_points_file_sha256,
            target_point_count=built.point_count,
            max_voxels=100,
            memory_safety_margin_bytes=0,
            available_memory_provider=lambda: 1,
        )


def test_external_reducer_rejects_resigned_range_byte_tamper(
    tmp_path: Path, reducer_binary: tuple[Path, str]
) -> None:
    replay, ranges, _ = _fixture(tmp_path)
    descriptor, descriptor_sha = write_range_descriptor(tmp_path / "ranges.tsv", ranges)
    with replay.open("r+b") as stream:
        stream.seek(0)
        stream.write(np.asarray([99.0], dtype="<f8").tobytes())
    binary, binary_sha = reducer_binary
    with pytest.raises(ExternalVoxelReductionError, match="external reducer failed"):
        build_external_target_map(
            replay_path=replay,
            range_descriptor_path=descriptor,
            range_descriptor_sha256=descriptor_sha,
            reducer_binary=binary,
            reducer_binary_sha256=binary_sha,
            voxel_rule=_rule(),
            target_points_path=tmp_path / "target.npy",
            max_voxels=100,
            disk_gate=RecordingDiskGate(),
        )
    with pytest.raises(ExternalVoxelReductionError, match="range descriptor SHA"):
        build_external_target_map(
            replay_path=replay,
            range_descriptor_path=descriptor,
            range_descriptor_sha256="0" * 64,
            reducer_binary=binary,
            reducer_binary_sha256=binary_sha,
            voxel_rule=_rule(),
            target_points_path=tmp_path / "other.npy",
            max_voxels=100,
            disk_gate=RecordingDiskGate(),
        )


def test_external_reducer_closes_descriptor_and_voxel_capacity(
    tmp_path: Path, reducer_binary: tuple[Path, str]
) -> None:
    replay, ranges, _ = _fixture(tmp_path)
    descriptor, descriptor_sha = write_range_descriptor(tmp_path / "ranges.tsv", ranges)
    binary, binary_sha = reducer_binary
    with pytest.raises(ExternalVoxelReductionError, match="external reducer failed"):
        build_external_target_map(
            replay_path=replay,
            range_descriptor_path=descriptor,
            range_descriptor_sha256=descriptor_sha,
            reducer_binary=binary,
            reducer_binary_sha256=binary_sha,
            voxel_rule=_rule(),
            target_points_path=tmp_path / "target.npy",
            max_voxels=1,
            disk_gate=RecordingDiskGate(),
        )


def test_external_reducer_accepts_authenticated_zero_point_range(
    tmp_path: Path, reducer_binary: tuple[Path, str]
) -> None:
    replay, ranges, _ = _fixture(tmp_path)
    ranges.insert(
        1,
        AuthenticatedReplayRange(
            byte_offset=72,
            point_count=0,
            transformed_xyz_sha256=hashlib.sha256(b"").hexdigest(),
        ),
    )
    # Move the following synthetic range to keep offsets strictly increasing.
    ranges[2] = AuthenticatedReplayRange(
        byte_offset=120,
        point_count=ranges[2].point_count,
        transformed_xyz_sha256=ranges[2].transformed_xyz_sha256,
    )
    descriptor, descriptor_sha = write_range_descriptor(
        tmp_path / "ranges-with-empty.tsv", ranges
    )
    binary, binary_sha = reducer_binary
    result = build_external_target_map(
        replay_path=replay,
        range_descriptor_path=descriptor,
        range_descriptor_sha256=descriptor_sha,
        reducer_binary=binary,
        reducer_binary_sha256=binary_sha,
        voxel_rule=_rule(),
        target_points_path=tmp_path / "target-empty-range.npy",
        max_voxels=100,
        disk_gate=RecordingDiskGate(),
    )
    assert result.point_count > 0
