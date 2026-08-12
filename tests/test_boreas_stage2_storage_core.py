from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from phase_a_harness.real_data_preparation.content_addressed_store import (
    ContentAddressedStore,
    ContentAddressedStoreError,
    canonical_npy_bytes,
    deterministic_temporary_pcl_conversion,
)
from phase_a_harness.real_data_preparation.guard import NoRegistrationGuard
from phase_a_harness.real_data_preparation.io import canonical_json_bytes
from phase_a_harness.real_data_preparation.stage2_checkpoint import (
    CheckpointRecord,
    ChangedRemoteObjectError,
    DuplicateCheckpointError,
    OrphanCheckpointError,
    Stage2CheckpointError,
    Stage2CheckpointLog,
    cleanup_partial_temporaries,
    initialize_temporary_root,
)
from phase_a_harness.real_data_preparation.stage2_payload_guard import (
    BoreasLidarPayloadGuard,
    LidarPayloadDownloadForbiddenError,
    command_attempts_boreas_lidar_download,
    url_is_boreas_lidar_payload,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
MAP_KEY = "boreas-2021-11-14-09-47/lidar/1636901260120239.bin"
QUERY_KEY = "boreas-2021-01-26-11-22/lidar/1611678138564575.bin"


def _points(offset: float = 0.0) -> np.ndarray:
    values = np.arange(180, dtype="<f8").reshape(60, 3) / 10.0
    values[:, 0] += offset
    return values


def _record(
    *,
    key: str = MAP_KEY,
    etag: str = "0123456789abcdef",
    kind: str = "MAP",
) -> CheckpointRecord:
    return CheckpointRecord(
        record_kind=kind,
        s3_key=key,
        remote_size_bytes=512,
        etag=etag,
        last_modified="2026-08-12T01:02:03Z",
        local_temporary_sha256=SHA_A,
        processing_contract_sha256=SHA_B,
        gt_sha256=SHA_C,
        calibration_sha256=SHA_D,
        result_geometry_row_sha256=SHA_A if kind != "MAP" else None,
        map_state_transition_sha256=SHA_A if kind == "MAP" else None,
        completed_at_utc="2026-08-12T02:03:04Z",
    )


def _inventory(
    *, key: str = MAP_KEY, etag: str = "0123456789abcdef"
) -> list[dict[str, object]]:
    return [
        {
            "key": key,
            "size_bytes": 512,
            "etag": etag,
            "last_modified": "2026-08-12T01:02:03Z",
        }
    ]


def test_content_addressed_target_is_immutable_and_single_copy(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "target_maps")
    first = store.put_target_map(_points())
    second = store.put_target_map(_points())
    assert first == second
    assert first.payload_path == store.root / first.sha256 / "target_points.npy"
    assert store.physical_payload_copy_count(first.sha256) == 1
    assert np.array_equal(store.load_npy(first.sha256), _points())
    with pytest.raises(ContentAddressedStoreError, match="separate content-addressed manifest"):
        store.put_target_map(_points(), metadata={"sequence": "changed"})


def test_failed_reserved_role_publish_leaves_no_poison_object(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "reserved-role")
    with pytest.raises(ContentAddressedStoreError, match="payload filename differs"):
        store.put_npy(
            _points(), object_kind="target_map", payload_name="array.npy"
        )
    assert list(store.root.iterdir()) == []
    target = store.put_target_map(_points())
    assert target.payload_path.is_file()


def test_content_addressed_payload_tamper_fails_closed(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "target_maps")
    stored = store.put_target_map(_points())
    stored.payload_path.write_bytes(stored.payload_path.read_bytes() + b"tamper")
    with pytest.raises(ContentAddressedStoreError, match="digest differs"):
        store.get(stored.sha256)


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float64),
        np.asarray([[True, False, True]], dtype=np.bool_),
    ],
)
def test_target_and_source_require_finite_float64_xyz(
    tmp_path: Path, value: np.ndarray
) -> None:
    store = ContentAddressedStore(tmp_path / "xyz-contract")
    with pytest.raises(ContentAddressedStoreError, match="finite.*float64 XYZ"):
        store.put_target_map(value)
    with pytest.raises(ContentAddressedStoreError, match="finite.*float64 XYZ"):
        store.put_npy(
            value,
            object_kind="canonical_source",
            payload_name="source_points.npy",
        )


def test_resigned_metadata_kind_and_contract_tamper_fails(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "metadata-contract")
    target = store.put_target_map(_points())
    metadata = json.loads(target.metadata_path.read_text(encoding="utf-8"))
    metadata["object_kind"] = "canonical_array"
    target.metadata_path.write_bytes(canonical_json_bytes(metadata))
    with pytest.raises(ContentAddressedStoreError, match="object-kind"):
        store.get(target.sha256)

    source = store.put_npy(
        _points(1.0),
        object_kind="canonical_source",
        payload_name="source_points.npy",
    )
    metadata = json.loads(source.metadata_path.read_text(encoding="utf-8"))
    metadata["user_metadata"]["array_contract_schema"] = "evil"
    source.metadata_path.write_bytes(canonical_json_bytes(metadata))
    with pytest.raises(ContentAddressedStoreError, match="array-contract"):
        store.get(source.sha256)


def test_content_object_rejects_extra_directory_and_symlink_inventory(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "entry-contract")
    target = store.put_target_map(_points())
    extra = target.payload_path.parent / "extra"
    extra.mkdir()
    with pytest.raises(ContentAddressedStoreError, match="unexpected physical files"):
        store.get(target.sha256)
    extra.rmdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    (target.payload_path.parent / "link").symlink_to(outside)
    with pytest.raises(ContentAddressedStoreError, match="unexpected physical files"):
        store.get(target.sha256)


def test_snapshot_audit_rejects_rogue_or_partial_store_root(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "root-closure")
    target = store.put_target_map(_points())
    store.create_snapshot_reference("snapshot-000", target_map_sha256=target.sha256)
    rogue = store.root / "rogue-target"
    rogue.mkdir()
    (rogue / "target_points.npy").write_bytes(target.payload_path.read_bytes())
    with pytest.raises(ContentAddressedStoreError, match="unknown entry"):
        store.audit_snapshot_references(
            expected_target_map_sha256=target.sha256,
            expected_reference_count=1,
        )
def test_canonical_npy_normalizes_byte_order_without_value_change() -> None:
    little = np.arange(12, dtype="<f8").reshape(4, 3)
    big = little.astype(">f8")
    assert canonical_npy_bytes(little) == canonical_npy_bytes(big)


def test_content_addressed_json_is_canonical_and_immutable(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "json_store")
    stored = store.put_json({"z": 1, "a": [2, 3]}, metadata={"fixture": True})
    assert stored.payload_path.read_bytes() == canonical_json_bytes({"z": 1, "a": [2, 3]})
    assert store.load_json(stored.sha256) == {"a": [2, 3], "z": 1}


def test_snapshot_manifests_reference_one_target_without_copy(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "target_maps")
    target = store.put_target_map(_points())
    sources = [
        store.put_npy(
            _points(float(index + 1)),
            object_kind="canonical_source",
            payload_name="source_points.npy",
        )
        for index in range(100)
    ]
    for index in range(100):
        store.create_snapshot_reference(
            f"snapshot-{index:03d}",
            target_map_sha256=target.sha256,
            canonical_source_sha256=sources[index].sha256,
        )
    audit = store.audit_snapshot_references(
        expected_target_map_sha256=target.sha256,
        expected_reference_count=100,
    )
    assert audit == {
        "pass": True,
        "unique_target_map_count": 1,
        "snapshot_target_reference_count": 100,
        "physical_target_map_copy_count": 1,
        "target_map_sha256": target.sha256,
    }
    assert not list(store.references_root.rglob("*.npy"))


def test_snapshot_reference_rejects_dangling_or_wrong_kind_source(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "target_maps")
    target = store.put_target_map(_points())
    with pytest.raises(ContentAddressedStoreError, match="unsafe object directory"):
        store.create_snapshot_reference(
            "snapshot-000",
            target_map_sha256=target.sha256,
            canonical_source_sha256="a" * 64,
        )
    wrong = store.put_npy(
        _points(1.0), object_kind="other", payload_name="other.npy"
    )
    with pytest.raises(ContentAddressedStoreError, match="canonical-source"):
        store.create_snapshot_reference(
            "snapshot-001",
            target_map_sha256=target.sha256,
            canonical_source_sha256=wrong.sha256,
        )


def test_snapshot_reference_is_immutable_no_clobber(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "target_maps")
    target = store.put_target_map(_points())
    first = store.create_snapshot_reference(
        "snapshot-000", target_map_sha256=target.sha256, extra_bindings={"interval": 1}
    )
    before = first.read_bytes()
    assert (
        store.create_snapshot_reference(
            "snapshot-000", target_map_sha256=target.sha256, extra_bindings={"interval": 1}
        )
        == first
    )
    with pytest.raises(ContentAddressedStoreError, match="immutable reference differs"):
        store.create_snapshot_reference(
            "snapshot-000", target_map_sha256=target.sha256, extra_bindings={"interval": 2}
        )
    assert first.read_bytes() == before
    with pytest.raises(ContentAddressedStoreError, match="containers/payloads"):
        store.create_snapshot_reference(
            "snapshot-nested",
            target_map_sha256=target.sha256,
            extra_bindings={"nested": {"target_points": [[0.0, 0.0, 0.0]]}},
        )
    with pytest.raises(ContentAddressedStoreError, match="containers/payloads"):
        store.create_snapshot_reference(
            "snapshot-vertices",
            target_map_sha256=target.sha256,
            extra_bindings={"vertices": _points().tolist()},
        )
    with pytest.raises(ContentAddressedStoreError, match="path/point/payload"):
        store.create_snapshot_reference(
            "snapshot-path",
            target_map_sha256=target.sha256,
            extra_bindings={"target_path": "/tmp/alternate.npy"},
        )


def test_snapshot_audit_rejects_orphan_canonical_source(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "source-closure")
    target = store.put_target_map(_points())
    store.put_npy(
        _points(1.0),
        object_kind="canonical_source",
        payload_name="source_points.npy",
    )
    store.create_snapshot_reference("snapshot-000", target_map_sha256=target.sha256)
    with pytest.raises(ContentAddressedStoreError, match="orphan or missing"):
        store.audit_snapshot_references(
            expected_target_map_sha256=target.sha256,
            expected_reference_count=1,
        )


def test_duplicate_physical_target_is_detected(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "target_maps")
    target = store.put_target_map(_points())
    store.create_snapshot_reference("snapshot-000", target_map_sha256=target.sha256)
    duplicate = store.references_root / "target_points.npy"
    duplicate.write_bytes(target.payload_path.read_bytes())
    with pytest.raises(ContentAddressedStoreError, match="physical copies|unknown entry"):
        store.audit_snapshot_references(
            expected_target_map_sha256=target.sha256,
            expected_reference_count=1,
        )


def test_second_distinct_target_map_is_detected(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "target_maps")
    target = store.put_target_map(_points())
    store.put_target_map(_points(99.0))
    store.create_snapshot_reference("snapshot-000", target_map_sha256=target.sha256)
    with pytest.raises(ContentAddressedStoreError, match="exactly one target-map"):
        store.audit_snapshot_references(
            expected_target_map_sha256=target.sha256,
            expected_reference_count=1,
        )


def test_pcl_conversion_is_deterministic_and_always_cleaned(tmp_path: Path) -> None:
    sources = ContentAddressedStore(tmp_path / "sources")
    targets = ContentAddressedStore(tmp_path / "targets")
    source = sources.put_npy(
        _points(1.0), object_kind="canonical_source", payload_name="source_points.npy"
    )
    target = targets.put_target_map(_points())
    temporary = tmp_path / "tmp_pcl"
    with deterministic_temporary_pcl_conversion(
        source.payload_path, target.payload_path, temporary_root=temporary
    ) as first:
        first_hashes = (first.source_pcd_sha256, first.target_pcd_sha256)
        assert first.source_npy_sha256 == source.sha256
        assert first.target_npy_sha256 == target.sha256
        assert first.source_path.is_file() and first.target_path.is_file()
    assert [path.name for path in temporary.iterdir()] == [
        ".stage2_temporary_root.json"
    ]

    with pytest.raises(ContentAddressedStoreError, match="canonical CAS payload path|role"):
        with deterministic_temporary_pcl_conversion(
            target.payload_path,
            source.payload_path,
            temporary_root=temporary,
        ):
            pytest.fail("source/target roles must not be interchangeable")
    with deterministic_temporary_pcl_conversion(
        source.payload_path, target.payload_path, temporary_root=temporary
    ) as second:
        assert (second.source_pcd_sha256, second.target_pcd_sha256) == first_hashes
        work = second.directory
        with pytest.raises(RuntimeError, match="synthetic crash"):
            raise RuntimeError("synthetic crash")
    assert not work.exists()
    assert [path.name for path in temporary.iterdir()] == [
        ".stage2_temporary_root.json"
    ]


def test_pcl_cleanup_rejects_extra_marker_field_before_deleting(tmp_path: Path) -> None:
    sources = ContentAddressedStore(tmp_path / "sources")
    targets = ContentAddressedStore(tmp_path / "targets")
    source = sources.put_npy(
        _points(1.0), object_kind="canonical_source", payload_name="source_points.npy"
    )
    target = targets.put_target_map(_points())
    temporary = initialize_temporary_root(tmp_path / "tmp_pcl", purpose="tmp_pcl")
    marker = temporary / ".stage2_temporary_root.json"
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    marker_value["unexpected_authority"] = True
    marker.write_bytes(canonical_json_bytes(marker_value))
    victim = temporary / "must-not-delete.pcd"
    victim.write_bytes(b"synthetic")

    with pytest.raises(ContentAddressedStoreError, match="marker differs"):
        with deterministic_temporary_pcl_conversion(
            source.payload_path,
            target.payload_path,
            temporary_root=temporary,
        ):
            pytest.fail("conversion must not start")
    assert victim.read_bytes() == b"synthetic"


def test_checkpoint_append_is_canonical_chained_and_no_clobber(tmp_path: Path) -> None:
    log = Stage2CheckpointLog(
        tmp_path / "processed_map_objects.jsonl",
        processing_contract_sha256=SHA_B,
        gt_sha256=SHA_C,
        calibration_sha256=SHA_D,
        expected_record_kinds={"MAP", "QUERY"},
    )
    first = log.append_completed(_record())
    second = log.append_completed(_record(key=QUERY_KEY, kind="QUERY"))
    assert first["sequence_number"] == 1
    assert second["previous_record_sha256"] == first["record_sha256"]
    assert len(log.logical_records()) == 2
    with pytest.raises(DuplicateCheckpointError):
        log.append_completed(_record())


def test_checkpoint_append_refuses_existing_mixed_contract(tmp_path: Path) -> None:
    path = tmp_path / "processed_map_objects.jsonl"
    first = Stage2CheckpointLog(
        path,
        processing_contract_sha256=SHA_B,
        gt_sha256=SHA_C,
        calibration_sha256=SHA_D,
    )
    first.append_completed(_record())
    second = Stage2CheckpointLog(
        path,
        processing_contract_sha256=SHA_A,
        gt_sha256=SHA_C,
        calibration_sha256=SHA_D,
        expected_record_kinds={"MAP", "QUERY"},
    )
    replacement = _record(key=QUERY_KEY, kind="QUERY").to_mapping()
    replacement["processing_contract_sha256"] = SHA_A
    before = path.read_bytes()
    with pytest.raises(Stage2CheckpointError, match="differs from resume contract"):
        second.append_completed(replacement)
    assert path.read_bytes() == before


def test_checkpoint_log_role_rejects_wrong_record_kind(tmp_path: Path) -> None:
    log = Stage2CheckpointLog(
        tmp_path / "processed_map_objects.jsonl", expected_record_kinds={"MAP"}
    )
    with pytest.raises(Stage2CheckpointError, match="log role"):
        log.append_completed(_record(key=QUERY_KEY, kind="QUERY"))


def test_checkpoint_normalizes_quoted_etag_and_requires_utc_completion(tmp_path: Path) -> None:
    log = Stage2CheckpointLog(tmp_path / "processed_map_objects.jsonl")
    quoted = _record(etag='"0123456789abcdef"')
    appended = log.append_completed(quoted)
    assert appended["etag"] == "0123456789abcdef"
    assert log.resume_plan(_inventory(), current_map_state_sha256=SHA_A)["pass"] is True

    non_utc = {**_record(key=QUERY_KEY, kind="QUERY").to_mapping()}
    non_utc["completed_at_utc"] = "2026-08-12T10:03:04+08:00"
    with pytest.raises(Stage2CheckpointError, match="must use UTC"):
        log.append_completed(non_utc)


def test_checkpoint_resume_skips_authenticated_and_lists_pending(tmp_path: Path) -> None:
    log = Stage2CheckpointLog(tmp_path / "processed_map_objects.jsonl")
    log.append_completed(_record())
    inventory = _inventory() + _inventory(key=QUERY_KEY, etag="other-etag")
    report = log.resume_plan(inventory, current_map_state_sha256=SHA_A)
    assert report["completed_keys"] == [MAP_KEY]
    assert report["pending_keys"] == [QUERY_KEY]


def test_checkpoint_changed_etag_fails_closed(tmp_path: Path) -> None:
    log = Stage2CheckpointLog(tmp_path / "processed_map_objects.jsonl")
    log.append_completed(_record())
    with pytest.raises(ChangedRemoteObjectError, match="identity changed"):
        log.resume_plan(_inventory(etag="changed-etag"), current_map_state_sha256=SHA_A)


def test_checkpoint_orphan_object_and_orphan_state_are_rejected(tmp_path: Path) -> None:
    log = Stage2CheckpointLog(tmp_path / "processed_map_objects.jsonl")
    log.append_completed(_record())
    with pytest.raises(OrphanCheckpointError, match="absent from frozen inventory"):
        log.resume_plan(_inventory(key=QUERY_KEY, etag="other-etag"))
    with pytest.raises(OrphanCheckpointError, match="accumulator state is absent"):
        log.resume_plan(_inventory(), current_map_state_sha256=None)
    empty = Stage2CheckpointLog(tmp_path / "empty.jsonl")
    with pytest.raises(OrphanCheckpointError, match="without a completed MAP"):
        empty.resume_plan([], current_map_state_sha256=SHA_A)


def test_checkpoint_truncated_or_tampered_log_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "processed_map_objects.jsonl"
    log = Stage2CheckpointLog(path)
    log.append_completed(_record())
    original = path.read_bytes()
    path.write_bytes(original[:-1])
    with pytest.raises(Stage2CheckpointError, match="truncated"):
        log.read()
    path.write_bytes(original.replace(b"512", b"513", 1))
    with pytest.raises(Stage2CheckpointError, match="digest differs"):
        log.read()


def test_partial_temp_cleanup_requires_marker_and_removes_crash_state(tmp_path: Path) -> None:
    root = initialize_temporary_root(tmp_path / "tmp_download", purpose="tmp_download")
    (root / "object.bin.partial").write_bytes(b"synthetic")
    nested = root / "decode-partial"
    nested.mkdir()
    (nested / "points.npy").write_bytes(b"synthetic")
    report = cleanup_partial_temporaries(root)
    assert report["removed_file_count"] == 2
    assert [path.name for path in root.iterdir()] == [".stage2_temporary_root.json"]
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    with pytest.raises(Stage2CheckpointError, match="unmarked"):
        cleanup_partial_temporaries(unmarked)


def test_partial_temp_cleanup_refuses_symlink(tmp_path: Path) -> None:
    root = initialize_temporary_root(tmp_path / "tmp_decode", purpose="tmp_decode")
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    (root / "link").symlink_to(outside)
    with pytest.raises(Stage2CheckpointError, match="symlink"):
        cleanup_partial_temporaries(root)
    assert outside.read_bytes() == b"keep"


def test_partial_temp_cleanup_rejects_extra_marker_field_before_deleting(
    tmp_path: Path,
) -> None:
    root = initialize_temporary_root(tmp_path / "tmp_download", purpose="tmp_download")
    marker = root / ".stage2_temporary_root.json"
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    marker_value["unexpected_authority"] = True
    marker.write_bytes(canonical_json_bytes(marker_value))
    victim = root / "must-not-delete.bin"
    victim.write_bytes(b"synthetic")

    with pytest.raises(Stage2CheckpointError, match="field set differs"):
        cleanup_partial_temporaries(root)
    assert victim.read_bytes() == b"synthetic"


def test_temporary_root_cannot_bless_an_existing_nonempty_directory(tmp_path: Path) -> None:
    root = tmp_path / "persistent"
    root.mkdir()
    (root / "evidence.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Stage2CheckpointError, match="nonempty"):
        initialize_temporary_root(root, purpose="tmp_download")
    assert (root / "evidence.json").is_file()


@pytest.mark.parametrize(
    "command",
    [
        [
            "sh",
            "-c",
            "aws s3api get-object --bucket boreas --key seq/lidar/1.bin /tmp/x",
        ],
        [
            "python",
            "-c",
            "import urllib.request; urllib.request.urlretrieve("
            "'https://boreas.s3.amazonaws.com/seq/lidar/1.bin','/tmp/x')",
        ],
    ],
)
def test_payload_guard_detects_nested_shell_and_quoted_python_urls(command: list[str]) -> None:
    assert command_attempts_boreas_lidar_download(command) is True


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        (["aws", "s3", "ls", "s3://boreas/seq/lidar/"], False),
        (
            [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                "boreas",
                "--key",
                "seq/lidar/1.bin",
            ],
            False,
        ),
        (
            ["aws", "s3api", "list-objects-v2", "--bucket", "boreas", "--prefix", "seq/lidar/"],
            False,
        ),
        (["aws", "s3", "cp", "s3://boreas/seq/lidar/1.bin", "/tmp/x"], True),
        (
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                "boreas",
                "--key",
                "seq/lidar/1.bin",
                "/tmp/x",
            ],
            True,
        ),
        (
            "aws s3api get-object --bucket 'boreas' --key "
            "'boreas-2021-01-26-11-22/lidar/1.bin' /tmp/x",
            True,
        ),
        (
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket=boreas",
                "--key=boreas-2021-01-26-11-22/lidar/1.bin",
                "/tmp/x",
            ],
            True,
        ),
        (["aws", "s3", "sync", "s3://boreas/seq/lidar/", "/tmp/lidar"], True),
        (["curl", "https://boreas.s3.amazonaws.com/seq/lidar/1.bin"], True),
        (["curl", "-i", "https://boreas.s3.amazonaws.com/seq/lidar/1.bin"], True),
        (["curl", "-I", "https://boreas.s3.amazonaws.com/seq/lidar/1.bin"], False),
        (
            ["curl", "https://s3.us-west-2.amazonaws.com/boreas/seq/lidar/1.bin"],
            True,
        ),
        (["curl", "https://example.test/synthetic/lidar/1.bin"], False),
    ],
)
def test_payload_command_classifier(command: object, blocked: bool) -> None:
    assert command_attempts_boreas_lidar_download(command) is blocked


def test_payload_url_classifier_is_narrow() -> None:
    assert url_is_boreas_lidar_payload(
        "https://boreas.s3.amazonaws.com/seq/lidar/123.bin?versionId=x"
    )
    assert not url_is_boreas_lidar_payload(
        "https://boreas.s3.amazonaws.com/seq/applanix/lidar_poses.csv"
    )
    assert not url_is_boreas_lidar_payload("https://example.test/seq/lidar/123.bin")


def test_payload_guard_requires_both_prohibition_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", raising=False)
    monkeypatch.setenv("ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD", "1")
    with pytest.raises(LidarPayloadDownloadForbiddenError, match="NO_REGISTRATION"):
        BoreasLidarPayloadGuard().__enter__()
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    monkeypatch.delenv("ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD", raising=False)
    with pytest.raises(LidarPayloadDownloadForbiddenError, match="NO_LIDAR"):
        BoreasLidarPayloadGuard().__enter__()


def test_payload_attestation_requires_guard_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    monkeypatch.setenv("ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD", "1")
    report = BoreasLidarPayloadGuard().attestation()
    assert report["guard_was_activated"] is False
    assert report["pass"] is False


def test_payload_guard_blocks_download_and_allows_metadata_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    monkeypatch.setenv("ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD", "1")
    calls: list[object] = []

    def fake_run(command: object, *args: object, **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with NoRegistrationGuard(), BoreasLidarPayloadGuard() as guard:
        completed = subprocess.run(
            ["aws", "s3", "ls", "s3://boreas/seq/lidar/"], check=True
        )
        assert completed.returncode == 0
        with pytest.raises(LidarPayloadDownloadForbiddenError):
            subprocess.run(
                ["aws", "s3", "cp", "s3://boreas/seq/lidar/1.bin", "/tmp/x"]
            )
        report = guard.attestation()
    assert len(calls) == 1
    assert report["allowed_metadata_operation_count"] == 1
    assert report["blocked_lidar_payload_attempt_count"] == 1
    assert report["lidar_payload_download_count"] == 0
    assert report["lidar_payload_download_bytes"] == 0
    assert report["pass"] is True


def test_payload_guard_blocks_http_get_but_allows_head_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    monkeypatch.setenv("ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD", "1")
    calls: list[object] = []

    def fake_urlopen(url: object, *args: object, **kwargs: object) -> object:
        calls.append(url)
        return object()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    payload_url = "https://boreas.s3.amazonaws.com/seq/lidar/1.bin"
    with BoreasLidarPayloadGuard() as guard:
        with pytest.raises(LidarPayloadDownloadForbiddenError):
            urllib.request.urlopen(payload_url)
        request = urllib.request.Request(payload_url, method="HEAD")
        urllib.request.urlopen(request)
        report = guard.attestation()
    assert calls == [request]
    assert report["blocked_http_attempt_count"] == 1
    assert report["allowed_metadata_operation_count"] == 1


def test_payload_guard_restores_process_and_http_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    monkeypatch.setenv("ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD", "1")
    original_run = subprocess.run
    original_urlopen = urllib.request.urlopen
    with BoreasLidarPayloadGuard():
        assert subprocess.run is not original_run
        assert urllib.request.urlopen is not original_urlopen
    assert subprocess.run is original_run
    assert urllib.request.urlopen is original_urlopen
