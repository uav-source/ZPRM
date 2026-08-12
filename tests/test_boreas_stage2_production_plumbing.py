from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import Mock

import numpy as np
import pytest

from phase_a_harness.real_data_preparation.boreas_stage2_execution import (
    BoreasStage2Execution,
    BoreasStage2ExecutionError,
    Stage2ObjectProcessingResult,
)
from phase_a_harness.real_data_preparation.boreas_stage2_remote import (
    AuthorizedRemoteObject,
    BoreasStage2RemoteError,
    DownloadReceipt,
    FrozenAllowlist,
    RemoteIdentityChangedError,
    RemoteObjectIdentity,
    StrictAllowlistDownloader,
    UnauthorizedRemoteObjectError,
    list_remote_metadata,
    load_frozen_allowlist,
    reconcile_remote_metadata,
)
from phase_a_harness.real_data_preparation.io import (
    canonical_json_bytes,
    csv_bytes,
    sha256_file,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_authorization import (
    VerifiedStage2Authorization,
)
from phase_a_harness.real_data_preparation.stage2_disk_gate import (
    DiskGateThresholds,
    InsufficientLiveDiskError,
    Stage2DiskGate,
    Stage2DiskGateError,
)
from phase_a_harness.real_data_preparation.streaming_target_map import (
    AUTHENTICATED_RANGE_TRANSITION_KIND,
    CENTROID_RULE,
    MapScan,
    ProductionMapReplayArray,
    ReplayMapObject,
    StreamingTargetMapError,
    VoxelRule,
    build_target_map_batch,
    production_replay_allocation_evidence,
    production_replay_plan_payload,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
ETAG_MAP = "1" * 32
ETAG_QUERY = "2" * 32
MAP_SEQUENCE = "boreas-2021-11-14-09-47"
QUERY_SEQUENCE = "boreas-2021-01-26-11-22"
MAP_KEY = f"{MAP_SEQUENCE}/lidar/1636901260120239.bin"
QUERY_KEY = f"{QUERY_SEQUENCE}/lidar/1611678138564575.bin"
LAST_MODIFIED = "2021-11-18T08:37:01Z"


def _allowlist_path(tmp_path: Path) -> Path:
    rows = [
        {
            "selection_role": "TARGET_MAP",
            "sequence_id": MAP_SEQUENCE,
            "key": MAP_KEY,
            "timestamp_us": "1636901260120239",
            "last_modified": LAST_MODIFIED,
            "size_bytes": "48",
            "selection_reason": "synthetic map fixture",
        },
        {
            "selection_role": "QUERY",
            "sequence_id": QUERY_SEQUENCE,
            "key": QUERY_KEY,
            "timestamp_us": "1611678138564575",
            "last_modified": LAST_MODIFIED,
            "size_bytes": "48",
            "selection_reason": "synthetic query fixture",
        },
    ]
    path = tmp_path / "allowlist.csv"
    path.write_bytes(
        csv_bytes(
            rows,
            (
                "selection_role",
                "sequence_id",
                "key",
                "timestamp_us",
                "last_modified",
                "size_bytes",
                "selection_reason",
            ),
        )
    )
    return path


def _inventory(tmp_path: Path):
    path = _allowlist_path(tmp_path)
    frozen = load_frozen_allowlist(path, expected_sha256=sha256_file(path))
    return reconcile_remote_metadata(
        frozen,
        [
            {
                "key": MAP_KEY,
                "size_bytes": 48,
                "etag": ETAG_MAP,
                "last_modified": "2021-11-18T08:37:01+00:00",
            },
            {
                "key": QUERY_KEY,
                "size_bytes": 48,
                "etag": ETAG_QUERY,
                "last_modified": LAST_MODIFIED,
            },
        ],
    )


def _aws_executable(tmp_path: Path) -> Path:
    path = tmp_path / "aws"
    path.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    path.chmod(0o700)
    return path


class _FakeAws:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.commands: list[tuple[str, ...]] = []
        self.response_etag_override: str | None = None
        self.fail_download = False

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = tuple(argv)
        self.commands.append(command)
        if "list-objects-v2" in command:
            prefix = command[command.index("--prefix") + 1]
            rows = []
            for key, payload in self.payloads.items():
                if key.startswith(prefix):
                    rows.append(
                        {
                            "Key": key,
                            "Size": len(payload),
                            "ETag": ETAG_MAP if key == MAP_KEY else ETAG_QUERY,
                            "LastModified": LAST_MODIFIED,
                        }
                    )
            return subprocess.CompletedProcess(command, 0, json.dumps(rows), "")
        key = command[command.index("--key") + 1]
        destination = Path(command[-1])
        destination.write_bytes(self.payloads[key])
        if self.fail_download:
            return subprocess.CompletedProcess(command, 2, "", "synthetic failure")
        etag = ETAG_MAP if key == MAP_KEY else ETAG_QUERY
        response = {
            "ContentLength": len(self.payloads[key]),
            "ETag": self.response_etag_override or etag,
            "LastModified": LAST_MODIFIED,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")


def _gate(tmp_path: Path, free_values: list[int] | None = None) -> Stage2DiskGate:
    values = iter(free_values) if free_values is not None else None

    def free(_: Path) -> int:
        return next(values) if values is not None else 10_000

    return Stage2DiskGate(
        tmp_path,
        thresholds=DiskGateThresholds(
            minimum_start_free_bytes=100,
            runtime_low_disk_watermark_bytes=50,
            storage_mode="SYNTHETIC_TEST_ONLY",
            storage_budget_sha256=HASH_A,
        ),
        audit_log_path=tmp_path / "disk_gate.jsonl",
        free_bytes_provider=free,
        now=lambda: datetime(2026, 8, 13, tzinfo=timezone.utc),
    )


def _downloader(tmp_path: Path, gate: Stage2DiskGate, fake: _FakeAws):
    inventory = _inventory(tmp_path)
    authorization = Mock(spec=VerifiedStage2Authorization, unsafe=True)
    authorization.allowlist_sha256 = inventory.allowlist_sha256
    authorization.expected_object_count = len(inventory.objects)
    authorization.expected_remote_bytes = sum(
        item.size_bytes for item in inventory.objects
    )
    authorization.primary_pair = {
        "map_sequence_id": MAP_SEQUENCE,
        "query_sequence_id": QUERY_SEQUENCE,
    }
    authorization.bucket = "boreas"
    authorization.temporary_root = tmp_path
    downloader = StrictAllowlistDownloader(
        inventory,
        aws_executable=_aws_executable(tmp_path),
        bucket="boreas",
        temporary_root=tmp_path / "tmp_download",
        disk_gate=gate,
        authorization=authorization,
        command_runner=fake,
        now=lambda: datetime(2026, 8, 13, tzinfo=timezone.utc),
    )
    return inventory, downloader


def test_downloader_and_execution_require_the_exact_capability(
    tmp_path: Path,
) -> None:
    fake = _FakeAws({MAP_KEY: b"m" * 48, QUERY_KEY: b"q" * 48})
    gate = _gate(tmp_path)
    inventory = _inventory(tmp_path)
    common = {
        "aws_executable": _aws_executable(tmp_path),
        "bucket": "boreas",
        "temporary_root": tmp_path / "tmp_download",
        "disk_gate": gate,
        "command_runner": fake,
    }
    with pytest.raises(TypeError, match="authorization"):
        StrictAllowlistDownloader(inventory, **common)
    with pytest.raises(BoreasStage2RemoteError, match="VerifiedStage2Authorization"):
        StrictAllowlistDownloader(inventory, authorization=object(), **common)

    _, downloader = _downloader(tmp_path, gate, fake)
    other = Mock(spec=VerifiedStage2Authorization, unsafe=True)
    with pytest.raises(BoreasStage2ExecutionError, match="exact authorization"):
        BoreasStage2Execution(
            disk_gate=gate,
            downloader=downloader,
            authorization=other,
        )


def test_frozen_allowlist_and_metadata_reconciliation_are_separate(tmp_path: Path) -> None:
    path = _allowlist_path(tmp_path)
    frozen = load_frozen_allowlist(path, expected_sha256=sha256_file(path))
    assert [row.role_ordinal for row in frozen.objects] == [0, 0]
    assert not hasattr(frozen.objects[0], "etag")
    inventory = reconcile_remote_metadata(
        frozen,
        [
            RemoteObjectIdentity(MAP_KEY, 48, ETAG_MAP, LAST_MODIFIED),
            RemoteObjectIdentity(QUERY_KEY, 48, ETAG_QUERY, LAST_MODIFIED),
            RemoteObjectIdentity(
                f"{MAP_SEQUENCE}/lidar/1636901260223877.bin",
                48,
                "3" * 32,
                LAST_MODIFIED,
            ),
        ],
    )
    assert inventory.allowlist_sha256 == sha256_file(path)
    assert len(inventory.objects) == 2
    assert len(inventory.inventory_sha256) == 64


def test_reconcile_fails_closed_on_changed_size_or_last_modified(tmp_path: Path) -> None:
    path = _allowlist_path(tmp_path)
    frozen = load_frozen_allowlist(path, expected_sha256=sha256_file(path))
    rows = [
        {"key": MAP_KEY, "size_bytes": 72, "etag": ETAG_MAP, "last_modified": LAST_MODIFIED},
        {"key": QUERY_KEY, "size_bytes": 48, "etag": ETAG_QUERY, "last_modified": LAST_MODIFIED},
    ]
    with pytest.raises(RemoteIdentityChangedError, match="size changed"):
        reconcile_remote_metadata(frozen, rows)
    rows[0]["size_bytes"] = 48
    rows[0]["last_modified"] = "2021-11-18T08:37:02Z"
    with pytest.raises(RemoteIdentityChangedError, match="LastModified changed"):
        reconcile_remote_metadata(frozen, rows)


def test_exact_production_allowlist_reconciles_audited_s3_ls_display_offset() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / (
        "frozen_assets/public_data_external_validation_v2_boreas_stage1/"
        "boreas_v2_stage2_download_allowlist.csv"
    )
    frozen_all = load_frozen_allowlist(
        path,
        expected_sha256=(
            "26ac211c854472dcb3db2f1cd5b096849bfd27867bac34e75bc8ce0d01bb2787"
        ),
    )
    first = frozen_all.objects[0]
    frozen = FrozenAllowlist(
        path=frozen_all.path,
        sha256=frozen_all.sha256,
        objects=(first,),
    )
    live = {
        "key": first.key,
        "size_bytes": first.size_bytes,
        "etag": "fcdd5678280fd9994f168cf4fefb793a",
        "last_modified": "2021-11-18T00:37:01Z",
    }
    inventory = reconcile_remote_metadata(frozen, [live])
    assert inventory.objects[0].last_modified == live["last_modified"]
    assert (
        inventory.objects[0].frozen_last_modified_display_offset_seconds
        == 8 * 60 * 60
    )
    live["last_modified"] = "2021-11-18T00:37:02Z"
    with pytest.raises(RemoteIdentityChangedError, match="LastModified changed"):
        reconcile_remote_metadata(frozen, [live])


def test_metadata_listing_uses_explicit_prefixes_and_no_payload_command(tmp_path: Path) -> None:
    fake = _FakeAws({MAP_KEY: b"m" * 48, QUERY_KEY: b"q" * 48})
    rows = list_remote_metadata(
        aws_executable=_aws_executable(tmp_path),
        bucket="boreas",
        sequence_ids=(MAP_SEQUENCE, QUERY_SEQUENCE),
        command_runner=fake,
    )
    assert {row.key for row in rows} == {MAP_KEY, QUERY_KEY}
    assert len(fake.commands) == 2
    assert all("list-objects-v2" in command for command in fake.commands)
    assert all("--no-sign-request" in command for command in fake.commands)


def test_conditional_downloader_receipt_and_authenticated_cleanup(tmp_path: Path) -> None:
    fake = _FakeAws({MAP_KEY: b"m" * 48, QUERY_KEY: b"q" * 48})
    gate = _gate(tmp_path)
    gate.assert_start()
    inventory, downloader = _downloader(tmp_path, gate, fake)
    item = inventory.for_role("TARGET_MAP")[0]
    downloaded = downloader.materialize(item)
    command = fake.commands[-1]
    assert command[command.index("--key") + 1] == MAP_KEY
    assert command[command.index("--if-match") + 1] == f'"{ETAG_MAP}"'
    assert downloaded.receipt.remote_size_bytes == 48
    assert downloaded.receipt.local_temporary_sha256 == hashlib.sha256(b"m" * 48).hexdigest()
    assert len(downloaded.receipt.as_dict()["receipt_sha256"]) == 64
    assert downloaded.receipt.csv_row(execution_stage="MAP_INGEST") == {
        "execution_stage": "MAP_INGEST",
        "selection_role": "TARGET_MAP",
        "sequence_id": MAP_SEQUENCE,
        "object_key": MAP_KEY,
        "timestamp_us": 1636901260120239,
        "last_modified": LAST_MODIFIED,
        "remote_size_bytes": 48,
        "etag": ETAG_MAP,
        "payload_sha256": hashlib.sha256(b"m" * 48).hexdigest(),
        "receipt_status": "PASS_SIZE_ETAG_SHA256",
    }
    assert downloader.release(downloaded) == 48
    assert [path.name for path in downloader.temporary_root.iterdir()] == [
        ".stage2_temporary_root.json"
    ]


def test_downloader_rejects_forged_identity_before_command(tmp_path: Path) -> None:
    fake = _FakeAws({MAP_KEY: b"m" * 48, QUERY_KEY: b"q" * 48})
    gate = _gate(tmp_path)
    gate.assert_start()
    inventory, downloader = _downloader(tmp_path, gate, fake)
    original = inventory.objects[0]
    forged = AuthorizedRemoteObject(
        frozen=original.frozen,
        remote=RemoteObjectIdentity(MAP_KEY, 48, "9" * 32, LAST_MODIFIED),
    )
    with pytest.raises(UnauthorizedRemoteObjectError, match="not in"):
        downloader.materialize(forged)
    assert fake.commands == []


def test_download_response_identity_change_removes_partial(tmp_path: Path) -> None:
    fake = _FakeAws({MAP_KEY: b"m" * 48, QUERY_KEY: b"q" * 48})
    fake.response_etag_override = "9" * 32
    gate = _gate(tmp_path)
    gate.assert_start()
    inventory, downloader = _downloader(tmp_path, gate, fake)
    with pytest.raises(RemoteIdentityChangedError, match="identity changed"):
        downloader.materialize(inventory.objects[0])
    assert [path.name for path in downloader.temporary_root.iterdir()] == [
        ".stage2_temporary_root.json"
    ]


def test_disk_gate_uses_fresh_readings_and_hash_chains_all_write_types(tmp_path: Path) -> None:
    gate = _gate(tmp_path, [1000, 900, 800, 700])
    gate.assert_start(operation_id="start")
    gate.before_download(48, object_key=MAP_KEY)
    gate.before_materialization(96, artifact_id="replay-range-0")
    gate.before_checkpoint(10, checkpoint_id="map-ledger-0")
    events = gate.events
    assert [event["operation"] for event in events] == [
        "START",
        "BEFORE_DOWNLOAD",
        "BEFORE_MATERIALIZATION",
        "BEFORE_CHECKPOINT",
    ]
    assert [event["current_free_bytes"] for event in events] == [1000, 900, 800, 700]
    assert events[0]["previous_event_sha256"] == "0" * 64
    assert events[1]["previous_event_sha256"] == events[0]["event_sha256"]


def test_disk_gate_authenticates_and_loads_frozen_budget_mode(tmp_path: Path) -> None:
    budget = {
        "minimum_free_disk_required_before_start_bytes": 100,
        "runtime_low_disk_watermark_bytes": 50,
        "modes": [
            {
                "mode": "RECOMMENDED_OPERATIONAL",
                "minimum_free_disk_required_before_start_bytes": 100,
                "recommended_free_disk_bytes": 100,
                "runtime_low_disk_watermark_bytes": 50,
            }
        ],
    }
    budget_path = tmp_path / "budget.json"
    budget_path.write_bytes(canonical_json_bytes(budget))
    gate = Stage2DiskGate.from_frozen_budget(
        tmp_path,
        budget_path=budget_path,
        expected_budget_sha256=sha256_file(budget_path),
        audit_log_path=tmp_path / "frozen-budget-events.jsonl",
        free_bytes_provider=lambda _: 1000,
    )
    assert gate.thresholds.minimum_start_free_bytes == 100
    assert gate.thresholds.runtime_low_disk_watermark_bytes == 50
    assert gate.assert_start()["pass"] is True


def test_disk_gate_records_and_raises_runtime_watermark_failure(tmp_path: Path) -> None:
    gate = _gate(tmp_path, [1000, 80])
    gate.assert_start()
    with pytest.raises(InsufficientLiveDiskError, match="watermark"):
        gate.before_download(31, object_key=MAP_KEY)
    assert gate.events[-1]["pass"] is False
    assert gate.events[-1]["projected_remaining_free_bytes"] == 49


def test_disk_gate_audit_tamper_is_fail_closed(tmp_path: Path) -> None:
    gate = _gate(tmp_path)
    gate.assert_start()
    path = gate.audit_log_path
    path.write_bytes(path.read_bytes().replace(b'"pass":true', b'"pass":false', 1))
    with pytest.raises(Stage2DiskGateError, match="decision arithmetic|digest differs"):
        gate.verify_audit()


def _voxel_rule() -> VoxelRule:
    return VoxelRule(
        voxel_size_m=1.0,
        representative_rule=CENTROID_RULE,
        parameter_authority="SYNTHETIC_FIXTURE_ONLY",
        scientific_contract_sha256=HASH_A,
        origin_xyz_m=(0.0, 0.0, 0.0),
    )


def _partial_replay(
    tmp_path: Path,
    *,
    track_python_voxel_state: bool = True,
    allocation_fault_hook=None,
) -> ProductionMapReplayArray:
    return ProductionMapReplayArray(
        [ReplayMapObject(0, MAP_KEY, 4 * 24, ETAG_MAP, LAST_MODIFIED)],
        replay_path=tmp_path / "replay.f64le",
        ledger_path=tmp_path / "replay.jsonl",
        processing_contract_sha256=HASH_A,
        gt_sha256=HASH_B,
        calibration_sha256=HASH_C,
        voxel_rule=_voxel_rule(),
        track_python_voxel_state=track_python_voxel_state,
        allocation_fault_hook=allocation_fault_hook,
    )


def _partial_scan() -> MapScan:
    return MapScan(
        ordinal=0,
        object_key=MAP_KEY,
        points_xyz=np.asarray([[0.25, 0.0, 0.0], [1.25, 0.0, 0.0]], dtype=np.float64),
        reference_from_sensor=np.eye(4),
        remote_size_bytes=4 * 24,
        etag=ETAG_MAP,
        last_modified=LAST_MODIFIED,
        gt_sha256=HASH_B,
        calibration_sha256=HASH_C,
        object_sha256=HASH_A,
    )


def test_replay_accepts_active_prefix_below_capacity_and_padding_never_enters_map(
    tmp_path: Path,
) -> None:
    replay = _partial_replay(tmp_path)
    row = replay.append_scan(_partial_scan())
    assert row["point_count"] == 2
    assert row["padding_point_count"] == 2
    assert replay.replay_path.stat().st_size == 4 * 24
    assert np.array_equal(replay.read_transformed_scan(0), _partial_scan().points_xyz)
    direct = build_target_map_batch([_partial_scan()], _voxel_rule())
    rebuilt = replay.build_target_map(_voxel_rule())
    assert rebuilt.target_map_sha256 == direct.target_map_sha256
    assert np.array_equal(rebuilt.points_xyz, direct.points_xyz)


@pytest.mark.parametrize(
    "crash_label",
    ("AFTER_ALLOCATION_INTENT", "AFTER_REPLAY_ALLOCATION", "AFTER_PLAN_LEDGER"),
)
def test_replay_allocation_intent_closes_every_preallocation_crash_window(
    tmp_path: Path, crash_label: str
) -> None:
    def crash(label: str) -> None:
        if label == crash_label:
            raise RuntimeError(f"synthetic crash {label}")

    with pytest.raises(RuntimeError, match=crash_label):
        _partial_replay(tmp_path, allocation_fault_hook=crash)
    assert (tmp_path / "replay_allocation_intent.json").is_file()
    resumed = _partial_replay(tmp_path)
    assert resumed.completed_object_count == 0
    assert resumed.ledger_path.is_file()
    assert resumed.replay_path.stat().st_size == 4 * 24
    assert not (tmp_path / "replay_allocation_intent.json").exists()
    resumed.append_scan(_partial_scan())
    assert _partial_replay(tmp_path).completed_object_count == 1


def test_replay_allocation_intent_never_authenticates_nonzero_or_sparse_orphan(
    tmp_path: Path,
) -> None:
    def crash(label: str) -> None:
        if label == "AFTER_REPLAY_ALLOCATION":
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        _partial_replay(tmp_path, allocation_fault_hook=crash)
    (tmp_path / "replay.f64le").write_bytes(b"X" + b"\0" * 95)
    with pytest.raises(StreamingTargetMapError, match="contains nonzero bytes"):
        _partial_replay(tmp_path)


@pytest.mark.parametrize("allocation_kind", ("short", "sparse"))
def test_pre_gate_recovery_removes_mid_fallocate_zero_state(
    tmp_path: Path, allocation_kind: str
) -> None:
    def crash(label: str) -> None:
        if label == "AFTER_ALLOCATION_INTENT":
            raise RuntimeError("synthetic mid-fallocate crash")

    with pytest.raises(RuntimeError, match="mid-fallocate"):
        _partial_replay(tmp_path, allocation_fault_hook=crash)
    replay_path = tmp_path / "replay.f64le"
    if allocation_kind == "short":
        replay_path.write_bytes(b"\0" * 48)
    else:
        with replay_path.open("wb") as stream:
            stream.truncate(96)
    intent = json.loads((tmp_path / "replay_allocation_intent.json").read_text())
    assert intent["total_replay_bytes"] == 96
    resumed = _partial_replay(tmp_path)
    assert resumed.replay_path.stat().st_size == 96
    assert resumed.completed_object_count == 0
    assert not (tmp_path / "replay_allocation_intent.json").exists()


def test_replay_recovery_accepts_only_exact_truncated_plan_prefix(
    tmp_path: Path,
) -> None:
    def crash(label: str) -> None:
        if label == "AFTER_REPLAY_ALLOCATION":
            raise RuntimeError("synthetic PLAN write crash")

    with pytest.raises(RuntimeError, match="PLAN write"):
        replay = _partial_replay(tmp_path, allocation_fault_hook=crash)
    # Obtain the exact PLAN line through the public evidence API from a clean
    # sibling fixture, then retain only a durable prefix.
    plan = production_replay_plan_payload(
        [ReplayMapObject(0, MAP_KEY, 4 * 24, ETAG_MAP, LAST_MODIFIED)],
        replay_path=tmp_path / "replay.f64le",
        processing_contract_sha256=HASH_A,
        gt_sha256=HASH_B,
        calibration_sha256=HASH_C,
        voxel_rule_sha256=_voxel_rule().contract_sha256,
        track_python_voxel_state=True,
    )
    _, plan_line = production_replay_allocation_evidence(
        plan, ledger_path=tmp_path / "replay.jsonl"
    )
    (tmp_path / "replay.jsonl").write_bytes(plan_line[:37])
    resumed = _partial_replay(tmp_path)
    assert resumed.ledger_path.read_bytes() == plan_line
    assert not (tmp_path / "replay_allocation_intent.json").exists()

    rejected = tmp_path / "rejected"
    rejected.mkdir()
    with pytest.raises(RuntimeError, match="PLAN write"):
        _partial_replay(rejected, allocation_fault_hook=crash)
    (rejected / "replay.jsonl").write_bytes(b"malicious")
    with pytest.raises(StreamingTargetMapError, match="PLAN replay ledger prefix"):
        _partial_replay(rejected)


def test_replay_production_mode_never_constructs_python_voxel_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phase_a_harness.real_data_preparation.streaming_target_map as target_module

    class ForbiddenBuilder:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise AssertionError("Python voxel builder was constructed")

    monkeypatch.setattr(target_module, "StreamingTargetMapBuilder", ForbiddenBuilder)
    replay = _partial_replay(tmp_path, track_python_voxel_state=False)
    row = replay.append_scan(_partial_scan())
    assert replay.tracks_python_voxel_state is False
    assert row["map_state_transition_kind"] == AUTHENTICATED_RANGE_TRANSITION_KIND
    assert len(row["map_state_transition_sha256"]) == 64
    records = replay.authenticated_range_records
    assert records[0]["byte_offset"] == records[0]["byte_start"] == 0
    assert records[0]["transformed_xyz_sha256"] == hashlib.sha256(
        _partial_scan().points_xyz.astype("<f8").tobytes()
    ).hexdigest()
    assert replay.plan_identity["track_python_voxel_state"] is False
    resumed = _partial_replay(tmp_path, track_python_voxel_state=False)
    assert resumed.completed_object_count == 1
    with pytest.raises(StreamingTargetMapError, match="external reducer"):
        resumed.build_target_map(_voxel_rule())


def test_replay_padding_tamper_fails_on_resume(tmp_path: Path) -> None:
    replay = _partial_replay(tmp_path)
    replay.append_scan(_partial_scan())
    with replay.replay_path.open("r+b") as stream:
        stream.seek(3 * 24)
        stream.write(b"X")
    with pytest.raises(StreamingTargetMapError, match="byte range differs"):
        _partial_replay(tmp_path)


def test_execution_orders_gates_callbacks_checkpoints_and_temp_deletion(tmp_path: Path) -> None:
    fake = _FakeAws({MAP_KEY: b"m" * 48, QUERY_KEY: b"q" * 48})
    gate = _gate(tmp_path)
    inventory, downloader = _downloader(tmp_path, gate, fake)
    execution = BoreasStage2Execution(
        disk_gate=gate,
        downloader=downloader,
        authorization=downloader.authorization,
    )
    execution.start()
    calls: list[str] = []

    def process(downloaded, item):
        assert downloaded.path.is_file()
        calls.append(f"process:{item.key}")
        return Stage2ObjectProcessingResult(
            result_sha256=downloaded.receipt.local_temporary_sha256,
            checkpoint_projected_bytes=128,
            metadata={"synthetic": True},
        )

    def checkpoint(stage, item, receipt, result):
        assert stage == "MAP_INGEST"
        assert receipt.key == item.key
        assert result.result_sha256 == receipt.local_temporary_sha256
        calls.append(f"checkpoint:{item.key}")

    summary = execution.run_map(
        inventory.for_role("TARGET_MAP"),
        materialization_projection=lambda item: item.size_bytes,
        processor=process,
        checkpoint_writer=checkpoint,
    )
    assert summary.completed_object_count == 1
    assert summary.downloaded_bytes == summary.deleted_temporary_bytes == 48
    assert calls == [f"process:{MAP_KEY}", f"checkpoint:{MAP_KEY}"]
    assert [row["operation"] for row in gate.events] == [
        "START",
        "BEFORE_DOWNLOAD",
        "BEFORE_MATERIALIZATION",
        "BEFORE_CHECKPOINT",
    ]


def test_execution_callback_failure_still_deletes_raw_temp(tmp_path: Path) -> None:
    fake = _FakeAws({MAP_KEY: b"m" * 48, QUERY_KEY: b"q" * 48})
    gate = _gate(tmp_path)
    inventory, downloader = _downloader(tmp_path, gate, fake)
    execution = BoreasStage2Execution(
        disk_gate=gate,
        downloader=downloader,
        authorization=downloader.authorization,
    )
    execution.start()

    def fail(*_: Any) -> Stage2ObjectProcessingResult:
        raise RuntimeError("synthetic processor crash")

    with pytest.raises(RuntimeError, match="processor crash"):
        execution.run_query_first_pass(
            inventory.for_role("QUERY"),
            materialization_projection=lambda item: item.size_bytes,
            processor=fail,
            checkpoint_writer=lambda *_: None,
        )
    assert [path.name for path in downloader.temporary_root.iterdir()] == [
        ".stage2_temporary_root.json"
    ]


def test_execution_does_not_choose_or_reorder_second_pass_selection(tmp_path: Path) -> None:
    fake = _FakeAws({MAP_KEY: b"m" * 48, QUERY_KEY: b"q" * 48})
    gate = _gate(tmp_path)
    inventory, downloader = _downloader(tmp_path, gate, fake)
    execution = BoreasStage2Execution(
        disk_gate=gate,
        downloader=downloader,
        authorization=downloader.authorization,
    )
    execution.start()
    selected = inventory.for_role("QUERY")
    first_pass_receipts = {
        QUERY_KEY: DownloadReceipt(
            selection_role="QUERY",
            sequence_id=QUERY_SEQUENCE,
            key=QUERY_KEY,
            timestamp_us=1611678138564575,
            remote_size_bytes=48,
            etag=ETAG_QUERY,
            last_modified=LAST_MODIFIED,
            local_temporary_sha256=hashlib.sha256(b"q" * 48).hexdigest(),
            downloaded_at_utc="2026-08-12T00:00:00Z",
        )
    }
    summary = execution.run_query_second_pass(
        selected,
        materialization_projection=lambda _: 64,
        processor=lambda downloaded, _: Stage2ObjectProcessingResult(
            downloaded.receipt.local_temporary_sha256, 64, {}
        ),
        checkpoint_writer=lambda *_: None,
        first_pass_receipts=first_pass_receipts,
    )
    assert summary.completed_object_count == len(selected)
    with pytest.raises(BoreasStage2ExecutionError, match="role must be QUERY"):
        execution.run_query_second_pass(
            inventory.for_role("TARGET_MAP"),
            materialization_projection=lambda _: 64,
            processor=lambda downloaded, _: Stage2ObjectProcessingResult(
                downloaded.receipt.local_temporary_sha256, 64, {}
            ),
            checkpoint_writer=lambda *_: None,
            first_pass_receipts=first_pass_receipts,
        )


def test_execution_second_pass_rejects_payload_sha_that_differs_from_first_pass(
    tmp_path: Path,
) -> None:
    fake = _FakeAws({MAP_KEY: b"m" * 48, QUERY_KEY: b"q" * 48})
    gate = _gate(tmp_path)
    inventory, downloader = _downloader(tmp_path, gate, fake)
    execution = BoreasStage2Execution(
        disk_gate=gate,
        downloader=downloader,
        authorization=downloader.authorization,
    )
    execution.start()
    prior = DownloadReceipt(
        selection_role="QUERY",
        sequence_id=QUERY_SEQUENCE,
        key=QUERY_KEY,
        timestamp_us=1611678138564575,
        remote_size_bytes=48,
        etag=ETAG_QUERY,
        last_modified=LAST_MODIFIED,
        local_temporary_sha256="9" * 64,
        downloaded_at_utc="2026-08-12T00:00:00Z",
    )
    with pytest.raises(BoreasStage2ExecutionError, match="differs from first-pass"):
        execution.run_query_second_pass(
            inventory.for_role("QUERY"),
            materialization_projection=lambda _: 64,
            processor=lambda downloaded, _: Stage2ObjectProcessingResult(
                downloaded.receipt.local_temporary_sha256, 64, {}
            ),
            checkpoint_writer=lambda *_: None,
            first_pass_receipts={QUERY_KEY: prior},
        )
    assert [path.name for path in downloader.temporary_root.iterdir()] == [
        ".stage2_temporary_root.json"
    ]
