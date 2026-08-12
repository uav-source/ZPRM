from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from phase_a_harness.real_data_preparation.io import canonical_json_bytes
from phase_a_harness.real_data_preparation.content_addressed_store import (
    ContentAddressedStore,
)
from phase_a_harness.real_data_preparation.stage2_checkpoint import (
    initialize_temporary_root,
)
from phase_a_harness.real_data_preparation.streaming_query_screen import (
    EXPECTED_SNAPSHOT_COUNT,
    QueryExecutionMode,
    QueryObject,
    StreamingQueryError,
    assert_100_snapshot_contract,
    assert_execution_modes_semantically_equal,
    first_pass_screen,
    freeze_selection_record,
    second_pass_materialize_selected,
)
from phase_a_harness.real_data_preparation.streaming_target_map import (
    CENTROID_RULE,
    MapScan,
    ProductionMapReplayArray,
    ReplayMapObject,
    StreamingTargetMapBuilder,
    StreamingTargetMapError,
    VoxelRule,
    assert_target_maps_exact,
    build_target_map_batch,
    build_target_map_incremental,
    canonical_float64_npy_bytes,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _voxel_rule() -> VoxelRule:
    return VoxelRule(
        voxel_size_m=1.0,
        representative_rule=CENTROID_RULE,
        parameter_authority="SYNTHETIC_FIXTURE_ONLY",
        scientific_contract_sha256=HASH_A,
        origin_xyz_m=(0.0, 0.0, 0.0),
    )


def _map_scans() -> list[MapScan]:
    first = np.asarray(
        [[1.25, 0.0, 0.0], [0.25, 0.5, 0.0], [0.75, 0.25, 0.0]],
        dtype=np.float64,
    )
    second = np.asarray(
        [[0.5, 0.25, 0.0], [-0.25, 0.0, 0.0], [1.5, 0.0, 0.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[1, 3] = 0.125
    return [
        MapScan(
            ordinal=0,
            object_key="synthetic/map/000.bin",
            points_xyz=first,
            reference_from_sensor=np.eye(4),
            remote_size_bytes=first.nbytes,
            etag='"etag-0"',
            last_modified="2026-01-01T00:00:00Z",
            gt_sha256=HASH_B,
            calibration_sha256=HASH_C,
            object_sha256=HASH_D,
        ),
        MapScan(
            ordinal=1,
            object_key="synthetic/map/001.bin",
            points_xyz=second,
            reference_from_sensor=transform,
            remote_size_bytes=second.nbytes,
            etag='"etag-1"',
            last_modified="2026-01-01T00:00:01Z",
            gt_sha256=HASH_B,
            calibration_sha256=HASH_C,
            object_sha256="e" * 64,
        ),
    ]


def test_storage_planner_cannot_choose_voxel_size() -> None:
    with pytest.raises(StreamingTargetMapError, match="storage planner"):
        VoxelRule(
            voxel_size_m=9.0,
            representative_rule=CENTROID_RULE,
            parameter_authority="STORAGE_PLANNER",
            scientific_contract_sha256=HASH_A,
            origin_xyz_m=(0.0, 0.0, 0.0),
        )
    with pytest.raises(TypeError):
        VoxelRule(  # type: ignore[call-arg]
            representative_rule=CENTROID_RULE,
            parameter_authority="SYNTHETIC_FIXTURE_ONLY",
            scientific_contract_sha256=HASH_A,
        )


def test_batch_incremental_and_worker_counts_are_byte_exact() -> None:
    scans = _map_scans()
    rule = _voxel_rule()
    batch = build_target_map_batch(scans, rule, worker_count=1)
    incremental = build_target_map_incremental(scans, rule, worker_count=3)
    many_workers = build_target_map_incremental(scans, rule, worker_count=64)
    assert_target_maps_exact(batch, incremental)
    assert_target_maps_exact(batch, many_workers)
    assert batch.final_state_sha256 == incremental.final_state_sha256
    assert batch.final_state_sha256 == many_workers.final_state_sha256
    assert batch.target_map_sha256 == hashlib.sha256(
        canonical_float64_npy_bytes(batch.points_xyz)
    ).hexdigest()
    assert batch.points_xyz.dtype == np.dtype("float64")
    assert batch.voxel_keys.dtype == np.dtype("int64")
    assert np.array_equal(
        batch.voxel_keys,
        np.asarray(sorted(map(tuple, batch.voxel_keys.tolist())), dtype=np.int64),
    )


def test_streaming_result_hash_is_the_physical_target_content_address(
    tmp_path: Path,
) -> None:
    result = build_target_map_incremental(_map_scans(), _voxel_rule())
    stored = ContentAddressedStore(tmp_path / "store").put_target_map(result.points_xyz)
    assert stored.sha256 == result.target_map_sha256
    assert hashlib.sha256(stored.payload_path.read_bytes()).hexdigest() == (
        result.target_map_sha256
    )


def test_input_and_point_order_are_fixed_not_silently_resorted() -> None:
    scans = _map_scans()
    with pytest.raises(StreamingTargetMapError, match="expected 0, got 1"):
        build_target_map_incremental(list(reversed(scans)), _voxel_rule())
    builder = StreamingTargetMapBuilder(_voxel_rule())
    builder.process_scan(scans[0])
    with pytest.raises(StreamingTargetMapError, match="duplicate object key"):
        duplicate_key = MapScan(
            **{
                **scans[1].__dict__,
                "object_key": scans[0].object_key,
            }
        )
        builder.process_scan(duplicate_key)


def test_resume_matches_uninterrupted_and_replay_does_not_double_count() -> None:
    scans = _map_scans()
    rule = _voxel_rule()
    uninterrupted = build_target_map_incremental(scans, rule)
    builder = StreamingTargetMapBuilder(rule)
    first_transition = builder.process_scan(scans[0])
    assert len(first_transition["map_state_transition_sha256"]) == 64
    checkpoint = builder.checkpoint()
    resumed = StreamingTargetMapBuilder.from_checkpoint(
        checkpoint, expected_voxel_rule=rule, worker_count=8
    )
    replay = resumed.process_scan(scans[0])
    assert replay == first_transition
    resumed.process_scan(scans[1])
    assert_target_maps_exact(uninterrupted, resumed.finalize())
    assert uninterrupted.final_state_sha256 == resumed.finalize().final_state_sha256


def test_resume_rejects_changed_etag_and_orphan_transition() -> None:
    scans = _map_scans()
    rule = _voxel_rule()
    builder = StreamingTargetMapBuilder(rule)
    builder.process_scan(scans[0])
    checkpoint = builder.checkpoint()
    resumed = StreamingTargetMapBuilder.from_checkpoint(
        checkpoint, expected_voxel_rule=rule
    )
    changed = MapScan(**{**scans[0].__dict__, "etag": '"changed"'})
    with pytest.raises(StreamingTargetMapError, match="changed on resume"):
        resumed.process_scan(changed)

    orphan = copy.deepcopy(checkpoint)
    orphan["state"]["core"]["next_input_ordinal"] = 2
    # Re-signing the envelope simulates a syntactically valid but semantically
    # orphaned checkpoint rather than a trivial outer-hash corruption.
    from phase_a_harness.real_data_preparation.io import canonical_json_bytes

    orphan["checkpoint_sha256"] = hashlib.sha256(
        canonical_json_bytes(orphan["state"])
    ).hexdigest()
    with pytest.raises(StreamingTargetMapError, match="orphan"):
        StreamingTargetMapBuilder.from_checkpoint(
            orphan, expected_voxel_rule=rule
        )


def test_checkpoint_detects_state_and_transition_tampering() -> None:
    builder = StreamingTargetMapBuilder(_voxel_rule())
    builder.process_scan(_map_scans()[0])
    checkpoint = builder.checkpoint()
    corrupted = copy.deepcopy(checkpoint)
    corrupted["state"]["processed_objects"][0]["object_identity"]["etag"] = "evil"
    with pytest.raises(StreamingTargetMapError, match="checkpoint SHA mismatch"):
        StreamingTargetMapBuilder.from_checkpoint(
            corrupted, expected_voxel_rule=_voxel_rule()
        )


def _production_replay(tmp_path: Path) -> ProductionMapReplayArray:
    scans = _map_scans()
    allowlist = [
        ReplayMapObject(
            ordinal=scan.ordinal,
            object_key=scan.object_key,
            remote_size_bytes=scan.remote_size_bytes,
            etag=scan.etag,
            last_modified=scan.last_modified,
        )
        for scan in scans
    ]
    rule = _voxel_rule()
    return ProductionMapReplayArray(
        allowlist,
        replay_path=tmp_path / "map_transformed_xyz.f64le",
        ledger_path=tmp_path / "processed_map_replay.jsonl",
        processing_contract_sha256=rule.contract_sha256,
        voxel_rule=rule,
        voxel_rule_sha256=rule.contract_sha256,
        gt_sha256=HASH_B,
        calibration_sha256=HASH_C,
    )


def test_production_replay_preallocates_exact_allowlist_size_and_fixed_ranges(
    tmp_path: Path,
) -> None:
    replay = _production_replay(tmp_path)
    scans = _map_scans()
    assert replay.replay_path.stat().st_size == sum(scan.remote_size_bytes for scan in scans)
    assert replay.completed_object_count == 0
    assert replay.pending_objects == tuple(
        ReplayMapObject(
            scan.ordinal,
            scan.object_key,
            scan.remote_size_bytes,
            scan.etag,
            scan.last_modified,
        )
        for scan in scans
    )
    first = replay.append_scan(scans[0])
    assert first["byte_start"] == 0
    assert first["byte_end_exclusive"] == scans[0].remote_size_bytes
    assert first["point_count"] == scans[0].points_xyz.shape[0]
    assert first["completed_at_utc"].endswith("Z")


def test_production_replay_resume_skips_complete_prefix_and_final_is_byte_exact(
    tmp_path: Path,
) -> None:
    scans = _map_scans()
    replay = _production_replay(tmp_path)
    replay.append_scan(scans[0])
    resumed = _production_replay(tmp_path)
    assert resumed.completed_keys == (scans[0].object_key,)
    assert [item.object_key for item in resumed.pending_objects] == [scans[1].object_key]
    resumed.append_scan(scans[1])
    direct = build_target_map_batch(scans, _voxel_rule())
    replay_one_worker = resumed.build_target_map(_voxel_rule(), worker_count=1)
    replay_many_workers = resumed.build_target_map(_voxel_rule(), worker_count=64)
    assert replay_one_worker.target_map_sha256 == direct.target_map_sha256
    assert replay_many_workers.target_map_sha256 == direct.target_map_sha256
    assert replay_one_worker.points_xyz.tobytes() == direct.points_xyz.tobytes()
    assert replay_many_workers.points_xyz.tobytes() == direct.points_xyz.tobytes()
    assert replay_one_worker.final_state_sha256 == replay_many_workers.final_state_sha256


def test_production_replay_rejects_changed_identity_wrong_rule_and_tamper(
    tmp_path: Path,
) -> None:
    scans = _map_scans()
    replay = _production_replay(tmp_path)
    replay.append_scan(scans[0])
    changed = MapScan(**{**scans[0].__dict__, "etag": '"changed"'})
    with pytest.raises(StreamingTargetMapError, match="identity changed"):
        replay.append_scan(changed)
    replay.append_scan(scans[1])
    wrong_rule = VoxelRule(
        voxel_size_m=2.0,
        representative_rule=CENTROID_RULE,
        parameter_authority="SYNTHETIC_FIXTURE_ONLY",
        scientific_contract_sha256=HASH_A,
        origin_xyz_m=(0.0, 0.0, 0.0),
    )
    with pytest.raises(StreamingTargetMapError, match="voxel rule differs"):
        replay.build_target_map(wrong_rule)
    with replay.replay_path.open("r+b") as stream:
        stream.seek(0)
        stream.write(b"X")
    with pytest.raises(StreamingTargetMapError, match="byte range differs"):
        _production_replay(tmp_path)


def test_production_replay_build_reauthenticates_ranges_after_construction(
    tmp_path: Path,
) -> None:
    replay = _production_replay(tmp_path)
    for scan in _map_scans():
        replay.append_scan(scan)
    with replay.replay_path.open("r+b") as stream:
        stream.seek(0)
        original = stream.read(1)
        stream.seek(0)
        stream.write(bytes([original[0] ^ 0x01]))
        stream.flush()

    with pytest.raises(
        StreamingTargetMapError,
        match="file identity changed|byte range differs at use time",
    ):
        replay.read_transformed_scan(0)
    with pytest.raises(
        StreamingTargetMapError,
        match="file identity changed|byte range differs at use time",
    ):
        replay.build_target_map(_voxel_rule())


def test_production_replay_rejects_orphan_and_partial_ledger(tmp_path: Path) -> None:
    replay = _production_replay(tmp_path)
    replay.append_scan(_map_scans()[0])
    ledger_payload = replay.ledger_path.read_bytes()
    replay.ledger_path.write_bytes(ledger_payload[:-1])
    with pytest.raises(StreamingTargetMapError, match="truncated"):
        _production_replay(tmp_path)

    other = tmp_path / "orphan"
    other.mkdir()
    orphan = _production_replay(other)
    orphan.ledger_path.unlink()
    with pytest.raises(StreamingTargetMapError, match="orphan"):
        _production_replay(other)


def test_production_replay_rejects_resigned_bogus_map_transition(
    tmp_path: Path,
) -> None:
    replay = _production_replay(tmp_path)
    replay.append_scan(_map_scans()[0])
    lines = replay.ledger_path.read_bytes().splitlines(keepends=True)
    assert len(lines) == 2
    scan_envelope = json.loads(lines[1].decode("utf-8"))
    scan_payload = scan_envelope["payload"]
    scan_payload["map_state_transition_sha256"] = "f" * 64
    replay_transition_core = {
        key: value
        for key, value in scan_payload.items()
        if key != "replay_state_transition_sha256"
    }
    scan_payload["replay_state_transition_sha256"] = hashlib.sha256(
        canonical_json_bytes(replay_transition_core)
    ).hexdigest()

    def compact_line(value: Mapping[str, Any]) -> bytes:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    digest_payload = {
        key: value
        for key, value in scan_envelope.items()
        if key != "record_sha256"
    }
    scan_envelope["record_sha256"] = hashlib.sha256(
        compact_line(digest_payload)
    ).hexdigest()
    replay.ledger_path.write_bytes(lines[0] + compact_line(scan_envelope))

    with pytest.raises(StreamingTargetMapError, match="map-state transition differs"):
        _production_replay(tmp_path)


def test_resigned_checkpoint_rejects_orphan_voxels_without_processed_objects() -> None:
    from phase_a_harness.real_data_preparation.io import canonical_json_bytes

    checkpoint = StreamingTargetMapBuilder(_voxel_rule()).checkpoint()
    checkpoint["state"]["core"]["voxels"] = [
        {"count": 1, "key": [9, 9, 9], "sum_xyz_float_hex": ["0x1p+0"] * 3}
    ]
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        canonical_json_bytes(checkpoint["state"])
    ).hexdigest()
    with pytest.raises(StreamingTargetMapError, match="orphan voxel"):
        StreamingTargetMapBuilder.from_checkpoint(
            checkpoint, expected_voxel_rule=_voxel_rule()
        )


def _npy_payload(points: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(points, dtype="<f8"),
        version=(1, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _query_fixture(count: int = 100) -> tuple[list[QueryObject], dict[str, bytes]]:
    objects: list[QueryObject] = []
    payloads: dict[str, bytes] = {}
    for index in range(count):
        points = np.asarray(
            [
                [float(index), 0.0, 0.0],
                [float(index), 1.0, 0.0],
                [float(index), 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        payload = _npy_payload(points)
        key = f"synthetic/lidar/query-{index:05d}.bin"
        payloads[key] = payload
        objects.append(
            QueryObject(
                ordinal=index,
                object_key=key,
                remote_size_bytes=len(payload),
                etag=f'"etag-{index:05d}"',
                last_modified=f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}Z",
                object_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return objects, payloads


def _decoder(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        return np.load(stream, allow_pickle=False)


def _metrics(points: np.ndarray, obj: QueryObject) -> dict[str, Any]:
    scale = float(obj.ordinal + 1)
    return {
        "initial_correspondence_count": int(points.shape[0]),
        "initial_valid_normal_correspondence_count": int(points.shape[0]),
        "lambda_min_trans": scale,
        "lambda_mid_trans": scale + 1.0,
        "lambda_max_trans": scale + 2.0,
        "normalized_lambda_min_trans": scale / 1000.0,
        "normalized_lambda_mid_trans": (scale + 1.0) / 1000.0,
        "normalized_lambda_max_trans": (scale + 2.0) / 1000.0,
        "condition_number_trans": scale + 2.0,
        "spectral_entropy_trans": scale / 100.0,
        "finite_source_point_count": int(points.shape[0]),
        "reference_interpolation_valid": True,
    }


def _selector(rows: tuple[dict[str, Any], ...]) -> list[dict[str, str]]:
    assert len(rows) == EXPECTED_SNAPSHOT_COUNT
    result: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        label = (
            "CORRIDOR_OR_WEAK_GEOMETRY"
            if index < 50
            else "GEOMETRY_RICH"
        )
        within_label = index if index < 50 else index - 50
        result.append(
            {
                "object_key": row["object_key"],
                "scene_label": label,
                "interval_id": f"{label}-{within_label // 5:02d}",
            }
        )
    return result


def _loader(payloads: Mapping[str, bytes], calls: list[str]):
    def load(obj: QueryObject) -> bytes:
        calls.append(obj.object_key)
        return payloads[obj.object_key]

    return load


def _assert_marked_temp_empty(workspace: Path) -> None:
    import json

    for purpose in ("tmp_download", "tmp_decode"):
        root = workspace / purpose
        marker = root / ".stage2_temporary_root.json"
        assert root.is_dir()
        assert [path.name for path in root.iterdir()] == [marker.name]
        value = json.loads(marker.read_text(encoding="utf-8"))
        assert value["purpose"] == purpose
        assert value["temporary"] is True


def _first_pass(
    objects: list[QueryObject],
    payloads: dict[str, bytes],
    workspace: Path,
    mode: QueryExecutionMode,
    *,
    calls: list[str] | None = None,
    existing_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    call_log = calls if calls is not None else []
    return first_pass_screen(
        objects,
        payload_loader=_loader(payloads, call_log),
        decoder=_decoder,
        geometry_metric=_metrics,
        target_map_sha256=HASH_A,
        gt_sha256=HASH_B,
        calibration_sha256=HASH_C,
        processing_contract_sha256=HASH_D,
        workspace=workspace,
        mode=mode,
        existing_rows=() if existing_rows is None else existing_rows,
        processed_jsonl_path=workspace / "checkpoints" / "processed_query_objects.jsonl",
    )


def test_streaming_first_pass_persists_metrics_and_deletes_temp_raw(tmp_path: Path) -> None:
    objects, payloads = _query_fixture(3)
    rows = _first_pass(
        objects,
        payloads,
        tmp_path / "streaming",
        QueryExecutionMode.STREAMING_LOW_DISK,
    )
    assert len(rows) == 3
    assert all(row["geometry_metrics"]["reference_interpolation_valid"] for row in rows)
    assert all(len(row["geometry_row_sha256"]) == 64 for row in rows)
    _assert_marked_temp_empty(tmp_path / "streaming")
    assert not (tmp_path / "streaming" / "raw_cache").exists()


def test_query_temp_wrong_purpose_marker_fails_before_cleanup(tmp_path: Path) -> None:
    objects, payloads = _query_fixture(1)
    workspace = tmp_path / "wrong-marker"
    root = initialize_temporary_root(
        workspace / "tmp_download", purpose="tmp_decode"
    )
    victim = root / "must-not-delete.bin"
    victim.write_bytes(b"synthetic")
    with pytest.raises(StreamingQueryError, match="marker purpose"):
        _first_pass(
            objects,
            payloads,
            workspace,
            QueryExecutionMode.STREAMING_LOW_DISK,
        )
    assert victim.read_bytes() == b"synthetic"


def test_query_temp_extra_marker_field_fails_before_cleanup(tmp_path: Path) -> None:
    objects, payloads = _query_fixture(1)
    workspace = tmp_path / "extra-marker"
    root = initialize_temporary_root(
        workspace / "tmp_download", purpose="tmp_download"
    )
    marker = root / ".stage2_temporary_root.json"
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    marker_value["unexpected_authority"] = True
    marker.write_bytes(canonical_json_bytes(marker_value))
    victim = root / "must-not-delete.bin"
    victim.write_bytes(b"synthetic")

    with pytest.raises(StreamingQueryError, match="purpose/identity differs"):
        _first_pass(
            objects,
            payloads,
            workspace,
            QueryExecutionMode.STREAMING_LOW_DISK,
        )
    assert victim.read_bytes() == b"synthetic"


def test_full_cache_rejects_symlink_prefix(tmp_path: Path) -> None:
    objects, payloads = _query_fixture(1)
    workspace = tmp_path / "cache-link"
    outside = tmp_path / "outside"
    outside.mkdir()
    prefix = workspace / "raw_cache" / objects[0].object_sha256[:2]
    prefix.parent.mkdir(parents=True)
    prefix.symlink_to(outside, target_is_directory=True)
    with pytest.raises(StreamingQueryError, match="raw-cache prefix"):
        _first_pass(objects, payloads, workspace, QueryExecutionMode.FULL_RAW_CACHE)
    assert list(outside.iterdir()) == []


def test_first_pass_rejects_registration_derived_metric_and_cleans_temp(
    tmp_path: Path,
) -> None:
    objects, payloads = _query_fixture(1)

    def forbidden(points: np.ndarray, obj: QueryObject) -> dict[str, Any]:
        return {**_metrics(points, obj), "registration_residual_rmse": 0.0}

    workspace = tmp_path / "forbidden"
    with pytest.raises(StreamingQueryError, match="registration-derived"):
        first_pass_screen(
            objects,
            payload_loader=_loader(payloads, []),
            decoder=_decoder,
            geometry_metric=forbidden,
            target_map_sha256=HASH_A,
            gt_sha256=HASH_B,
            calibration_sha256=HASH_C,
            processing_contract_sha256=HASH_D,
            workspace=workspace,
            mode=QueryExecutionMode.STREAMING_LOW_DISK,
        )
    _assert_marked_temp_empty(workspace)


def test_first_pass_rejects_non_allowlisted_metric_even_without_forbidden_name(
    tmp_path: Path,
) -> None:
    objects, payloads = _query_fixture(1)

    def smuggled(points: np.ndarray, obj: QueryObject) -> dict[str, Any]:
        return {**_metrics(points, obj), "alignment_score": 0.99}

    with pytest.raises(StreamingQueryError, match="non-allowlisted"):
        first_pass_screen(
            objects,
            payload_loader=_loader(payloads, []),
            decoder=_decoder,
            geometry_metric=smuggled,
            target_map_sha256=HASH_A,
            gt_sha256=HASH_B,
            calibration_sha256=HASH_C,
            processing_contract_sha256=HASH_D,
            workspace=tmp_path / "smuggled",
            mode=QueryExecutionMode.STREAMING_LOW_DISK,
        )


def test_query_resume_skips_authenticated_rows_and_changed_etag_fails(
    tmp_path: Path,
) -> None:
    objects, payloads = _query_fixture(4)
    prefix_calls: list[str] = []
    workspace = tmp_path / "resume"
    prefix = _first_pass(
        objects[:2],
        {key: payloads[key] for key in [obj.object_key for obj in objects[:2]]},
        workspace,
        QueryExecutionMode.STREAMING_LOW_DISK,
        calls=prefix_calls,
    )
    resume_calls: list[str] = []
    rows = _first_pass(
        objects,
        payloads,
        workspace,
        QueryExecutionMode.STREAMING_LOW_DISK,
        calls=resume_calls,
        existing_rows=prefix,
    )
    assert len(rows) == 4
    assert resume_calls == [objects[2].object_key, objects[3].object_key]

    changed = list(objects)
    changed[0] = QueryObject(**{**objects[0].__dict__, "etag": '"changed"'})
    with pytest.raises(StreamingQueryError, match="changed on resume"):
        _first_pass(
            changed,
            payloads,
            workspace,
            QueryExecutionMode.STREAMING_LOW_DISK,
            existing_rows=prefix,
        )


def test_query_resume_rejects_orphan_and_tampered_rows(tmp_path: Path) -> None:
    objects, payloads = _query_fixture(3)
    rows = _first_pass(
        objects,
        payloads,
        tmp_path / "initial",
        QueryExecutionMode.STREAMING_LOW_DISK,
    )
    geometry_root = tmp_path / "initial" / "geometry_metrics" / "rows_by_sha256"
    orphan_path = geometry_root / f"{'f' * 64}.json"
    orphan_path.write_text("{}", encoding="utf-8")
    with pytest.raises(StreamingQueryError, match="orphan"):
        _first_pass(
            objects,
            payloads,
            tmp_path / "initial",
            QueryExecutionMode.STREAMING_LOW_DISK,
        )
    orphan_path.unlink()
    tampered = copy.deepcopy(rows[:1])
    tampered[0]["geometry_metrics"]["lambda_min_trans"] = 999.0
    first_path = geometry_root / f"{rows[0]['geometry_row_sha256']}.json"
    first_path.write_text(json.dumps(tampered[0], sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(StreamingQueryError, match="geometry-row (?:binding|SHA)|canonical"):
        _first_pass(
            objects,
            payloads,
            tmp_path / "initial",
            QueryExecutionMode.STREAMING_LOW_DISK,
        )


def test_blind_selection_is_external_and_freezes_exact_100_contract(tmp_path: Path) -> None:
    objects, payloads = _query_fixture()
    rows = _first_pass(
        objects,
        payloads,
        tmp_path / "first",
        QueryExecutionMode.STREAMING_LOW_DISK,
    )
    record = freeze_selection_record(
        rows, selector=_selector, selection_contract_sha256=HASH_A
    )
    assert record["snapshot_count"] == 100
    assert len(record["selected_snapshots"]) == 100
    assert "THRESHOLDS_ARE_NOT_DEFINED" in record["policy_note"]
    assert_100_snapshot_contract(record["selected_snapshots"])
    assert all(
        selected["first_pass_geometry_row_sha256"]
        == rows[index]["geometry_row_sha256"]
        for index, selected in enumerate(record["selected_snapshots"])
    )


@pytest.mark.parametrize("count", [99, 101])
def test_snapshot_count_cannot_change(count: int) -> None:
    rows = [
        {
            "object_key": f"key-{index}",
            "scene_label": (
                "CORRIDOR_OR_WEAK_GEOMETRY" if index < 50 else "GEOMETRY_RICH"
            ),
            "interval_id": f"interval-{index // 5}",
        }
        for index in range(count)
    ]
    with pytest.raises(StreamingQueryError, match="snapshot count changed"):
        assert_100_snapshot_contract(rows)


def test_selector_cannot_smuggle_unregistered_threshold_or_result_field(
    tmp_path: Path,
) -> None:
    objects, payloads = _query_fixture()
    rows = _first_pass(
        objects,
        payloads,
        tmp_path,
        QueryExecutionMode.STREAMING_LOW_DISK,
    )

    def smuggle(values: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
        selected = _selector(values)
        selected[0]["weak_threshold"] = 0.1
        return selected

    with pytest.raises(StreamingQueryError, match="unregistered fields"):
        freeze_selection_record(rows, selector=smuggle, selection_contract_sha256=HASH_A)


def _canonicalizer(points: np.ndarray, obj: QueryObject) -> np.ndarray:
    del obj
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return points[order]


def test_second_pass_materializes_deterministic_sources_and_deletes_temp(
    tmp_path: Path,
) -> None:
    objects, payloads = _query_fixture()
    workspace = tmp_path / "workspace"
    rows = _first_pass(
        objects,
        payloads,
        workspace,
        QueryExecutionMode.STREAMING_LOW_DISK,
    )
    selection = freeze_selection_record(
        rows, selector=_selector, selection_contract_sha256=HASH_A
    )
    calls: list[str] = []
    manifest = second_pass_materialize_selected(
        selection,
        objects,
        payload_loader=_loader(payloads, calls),
        decoder=_decoder,
        canonicalizer=_canonicalizer,
        canonicalization_contract_sha256=HASH_B,
        workspace=workspace,
        output_root=tmp_path / "canonical",
        mode=QueryExecutionMode.STREAMING_LOW_DISK,
    )
    assert manifest["selected_source_count"] == 100
    assert len(calls) == 100
    _assert_marked_temp_empty(workspace)
    for row in manifest["selected_sources"]:
        path = tmp_path / "canonical" / row["relative_path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["canonical_source_sha256"]

    resume_calls: list[str] = []
    resumed = second_pass_materialize_selected(
        selection,
        objects,
        payload_loader=_loader(payloads, resume_calls),
        decoder=_decoder,
        canonicalizer=_canonicalizer,
        canonicalization_contract_sha256=HASH_B,
        workspace=workspace,
        output_root=tmp_path / "canonical",
        mode=QueryExecutionMode.STREAMING_LOW_DISK,
    )
    assert resumed == manifest
    assert resume_calls == []

    changed_selection = copy.deepcopy(selection)
    changed_selection["selection_contract_sha256"] = HASH_D
    changed_core = {
        key: value
        for key, value in changed_selection.items()
        if key != "selection_record_sha256"
    }
    changed_selection["selection_record_sha256"] = hashlib.sha256(
        canonical_json_bytes(changed_core)
    ).hexdigest()
    with pytest.raises(StreamingQueryError, match="processing_contract_sha256"):
        second_pass_materialize_selected(
            changed_selection,
            objects,
            payload_loader=_loader(payloads, []),
            decoder=_decoder,
            canonicalizer=_canonicalizer,
            canonicalization_contract_sha256=HASH_B,
            workspace=workspace,
            output_root=tmp_path / "canonical",
            mode=QueryExecutionMode.STREAMING_LOW_DISK,
        )


def test_second_pass_output_rejects_symlink_and_temporary_descendant(
    tmp_path: Path,
) -> None:
    objects, payloads = _query_fixture()
    workspace = tmp_path / "stage2-root"
    rows = _first_pass(objects, payloads, workspace, QueryExecutionMode.STREAMING_LOW_DISK)
    selection = freeze_selection_record(
        rows, selector=_selector, selection_contract_sha256=HASH_A
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked-output"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(StreamingQueryError, match="output root.*unsafe"):
        second_pass_materialize_selected(
            selection,
            objects,
            payload_loader=_loader(payloads, []),
            decoder=_decoder,
            canonicalizer=_canonicalizer,
            canonicalization_contract_sha256=HASH_B,
            workspace=workspace,
            output_root=linked,
            mode=QueryExecutionMode.STREAMING_LOW_DISK,
        )
    with pytest.raises(StreamingQueryError, match="overlaps"):
        second_pass_materialize_selected(
            selection,
            objects,
            payload_loader=_loader(payloads, []),
            decoder=_decoder,
            canonicalizer=_canonicalizer,
            canonicalization_contract_sha256=HASH_B,
            workspace=workspace,
            output_root=workspace / "tmp_download" / "canonical",
            mode=QueryExecutionMode.STREAMING_LOW_DISK,
        )
    assert list(outside.iterdir()) == []


def test_second_pass_natural_stage2_root_layout_is_allowed(tmp_path: Path) -> None:
    objects, payloads = _query_fixture()
    workspace = tmp_path / "stage2-root"
    rows = _first_pass(objects, payloads, workspace, QueryExecutionMode.STREAMING_LOW_DISK)
    selection = freeze_selection_record(
        rows, selector=_selector, selection_contract_sha256=HASH_A
    )
    manifest = second_pass_materialize_selected(
        selection,
        objects,
        payload_loader=_loader(payloads, []),
        decoder=_decoder,
        canonicalizer=_canonicalizer,
        canonicalization_contract_sha256=HASH_B,
        workspace=workspace,
        output_root=workspace,
        mode=QueryExecutionMode.STREAMING_LOW_DISK,
    )
    assert manifest["selected_source_count"] == 100


def test_second_pass_resume_rejects_changed_etag_and_orphan_row(tmp_path: Path) -> None:
    objects, payloads = _query_fixture()
    workspace = tmp_path / "workspace"
    output = tmp_path / "canonical"
    rows = _first_pass(objects, payloads, workspace, QueryExecutionMode.STREAMING_LOW_DISK)
    selection = freeze_selection_record(
        rows, selector=_selector, selection_contract_sha256=HASH_A
    )
    second_pass_materialize_selected(
        selection,
        objects,
        payload_loader=_loader(payloads, []),
        decoder=_decoder,
        canonicalizer=_canonicalizer,
        canonicalization_contract_sha256=HASH_B,
        workspace=workspace,
        output_root=output,
        mode=QueryExecutionMode.STREAMING_LOW_DISK,
    )
    changed = list(objects)
    changed[0] = QueryObject(**{**objects[0].__dict__, "etag": '"changed"'})
    with pytest.raises(StreamingQueryError, match="selected object identity changed"):
        second_pass_materialize_selected(
            selection,
            changed,
            payload_loader=_loader(payloads, []),
            decoder=_decoder,
            canonicalizer=_canonicalizer,
            canonicalization_contract_sha256=HASH_B,
            workspace=workspace,
            output_root=output,
            mode=QueryExecutionMode.STREAMING_LOW_DISK,
        )
    row_root = output / "manifests" / "selected_source_rows_by_sha256"
    orphan = row_root / f"{'f' * 64}.json"
    orphan.write_text("{}", encoding="utf-8")
    with pytest.raises(StreamingQueryError, match="orphan"):
        second_pass_materialize_selected(
            selection,
            objects,
            payload_loader=_loader(payloads, []),
            decoder=_decoder,
            canonicalizer=_canonicalizer,
            canonicalization_contract_sha256=HASH_B,
            workspace=workspace,
            output_root=output,
            mode=QueryExecutionMode.STREAMING_LOW_DISK,
        )


def test_nondeterministic_canonicalizer_fails_closed(tmp_path: Path) -> None:
    objects, payloads = _query_fixture()
    workspace = tmp_path / "workspace"
    rows = _first_pass(
        objects,
        payloads,
        workspace,
        QueryExecutionMode.STREAMING_LOW_DISK,
    )
    selection = freeze_selection_record(
        rows, selector=_selector, selection_contract_sha256=HASH_A
    )
    calls = 0

    def nondeterministic(points: np.ndarray, obj: QueryObject) -> np.ndarray:
        nonlocal calls
        del obj
        calls += 1
        result = points.copy()
        result[0, 0] += calls
        return result

    with pytest.raises(StreamingQueryError, match="nondeterministic"):
        second_pass_materialize_selected(
            selection,
            objects,
            payload_loader=_loader(payloads, []),
            decoder=_decoder,
            canonicalizer=nondeterministic,
            canonicalization_contract_sha256=HASH_B,
            workspace=workspace,
            output_root=tmp_path / "bad-canonical",
            mode=QueryExecutionMode.STREAMING_LOW_DISK,
        )
    _assert_marked_temp_empty(workspace)


def test_full_cache_and_streaming_modes_are_scientifically_identical(
    tmp_path: Path,
) -> None:
    objects, payloads = _query_fixture()
    streaming_workspace = tmp_path / "streaming-work"
    full_workspace = tmp_path / "full-work"
    streaming_calls: list[str] = []
    full_calls: list[str] = []
    streaming_rows = _first_pass(
        objects,
        payloads,
        streaming_workspace,
        QueryExecutionMode.STREAMING_LOW_DISK,
        calls=streaming_calls,
    )
    full_rows = _first_pass(
        objects,
        payloads,
        full_workspace,
        QueryExecutionMode.FULL_RAW_CACHE,
        calls=full_calls,
    )
    assert streaming_rows == full_rows
    assert len(list(full_workspace.glob("raw_cache/*/*.bin"))) == 100
    assert not (streaming_workspace / "raw_cache").exists()
    streaming_selection = freeze_selection_record(
        streaming_rows, selector=_selector, selection_contract_sha256=HASH_A
    )
    full_selection = freeze_selection_record(
        full_rows, selector=_selector, selection_contract_sha256=HASH_A
    )
    assert streaming_selection == full_selection

    streaming_sources = second_pass_materialize_selected(
        streaming_selection,
        objects,
        payload_loader=_loader(payloads, streaming_calls),
        decoder=_decoder,
        canonicalizer=_canonicalizer,
        canonicalization_contract_sha256=HASH_B,
        workspace=streaming_workspace,
        output_root=tmp_path / "streaming-output",
        mode=QueryExecutionMode.STREAMING_LOW_DISK,
    )
    full_sources = second_pass_materialize_selected(
        full_selection,
        objects,
        payload_loader=_loader(payloads, full_calls),
        decoder=_decoder,
        canonicalizer=_canonicalizer,
        canonicalization_contract_sha256=HASH_B,
        workspace=full_workspace,
        output_root=tmp_path / "full-output",
        mode=QueryExecutionMode.FULL_RAW_CACHE,
    )
    assert_execution_modes_semantically_equal(
        streaming_rows, full_rows, streaming_sources, full_sources
    )
    assert len(streaming_calls) == 200
    assert len(full_calls) == 100  # second pass reuses authenticated raw cache
    _assert_marked_temp_empty(streaming_workspace)
    _assert_marked_temp_empty(full_workspace)
