from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import phase_a_harness.real_data_preparation.boreas_v2_stage2_authorization as auth
from phase_a_harness.real_data_preparation.boreas_stage2_external_voxel import (
    ExternalTargetMapResult,
    default_reducer_source,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_runner import (
    BoreasV2Stage2Runner,
    BoreasV2Stage2RunnerConfig,
    BoreasV2Stage2RunnerDependencies,
    BoreasV2Stage2RunnerError,
    MapScanMaterialization,
    ProductionMapPreprocessor,
    ProductionRemoteMetadataProvider,
    ReducerResourcePlan,
    generate_production_reducer_resource_plan,
)
from phase_a_harness.real_data_preparation.guard import NoRegistrationGuard
from phase_a_harness.real_data_preparation.io import (
    atomic_write_bytes,
    canonical_json_bytes,
    compact_sha256,
    csv_bytes,
    sha256_file,
)
from phase_a_harness.real_data_preparation.streaming_target_map import (
    canonical_array_sha256,
    canonical_float64_npy_bytes,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MAP_SEQUENCE = "boreas-2021-11-14-09-47"
QUERY_SEQUENCE = "boreas-2021-01-26-11-22"
LAST_MODIFIED = "2021-11-18T08:37:01Z"
MAP_KEYS = (
    f"{MAP_SEQUENCE}/lidar/1636901260120239.bin",
    f"{MAP_SEQUENCE}/lidar/1636901260220239.bin",
)
QUERY_KEY = f"{QUERY_SEQUENCE}/lidar/1611678138564575.bin"
KEYS = (*MAP_KEYS, QUERY_KEY)
ETAGS = {key: f"{index + 1:032x}" for index, key in enumerate(KEYS)}
PAYLOADS = {key: bytes([65 + index]) * 48 for index, key in enumerate(KEYS)}
GT_SHA = "a" * 64
EXTRINSIC_SHA = "b" * 64


class _FakeAws:
    def __init__(self, payloads: Mapping[str, bytes]) -> None:
        self.payloads = dict(payloads)
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = tuple(argv)
        self.commands.append(command)
        key = command[command.index("--key") + 1]
        Path(command[-1]).write_bytes(self.payloads[key])
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "ContentLength": len(self.payloads[key]),
                    "ETag": ETAGS[key],
                    "LastModified": LAST_MODIFIED,
                }
            ),
            "",
        )


def _allowlist(path: Path) -> str:
    rows = []
    for index, key in enumerate(KEYS):
        sequence, _, filename = key.split("/")
        rows.append(
            {
                "selection_role": "TARGET_MAP" if index < 2 else "QUERY",
                "sequence_id": sequence,
                "key": key,
                "timestamp_us": filename[:-4],
                "last_modified": LAST_MODIFIED,
                "size_bytes": "48",
                "selection_reason": "synthetic runner fixture",
            }
        )
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
    return sha256_file(path)


def _preprocessing_contract(path: Path) -> tuple[str, str]:
    unsigned = {
        "contract_status": "FROZEN",
        "primary_pair": {
            "map_lidar_pose_sha256": GT_SHA,
            "map_sequence_id": MAP_SEQUENCE,
            "query_lidar_pose_sha256": "c" * 64,
            "query_sequence_id": QUERY_SEQUENCE,
            "static_t_applanix_lidar_sha256": EXTRINSIC_SHA,
        },
        "schema_version": "synthetic_runner_contract_v1",
    }
    value = {**unsigned, "contract_payload_sha256": compact_sha256(unsigned)}
    path.write_bytes(canonical_json_bytes(value))
    return sha256_file(path), str(value["contract_payload_sha256"])


def _budget(path: Path, *, start_bytes: int = 1, watermark_bytes: int = 1) -> str:
    value = {
        "minimum_free_disk_required_before_start_bytes": start_bytes,
        "runtime_low_disk_watermark_bytes": watermark_bytes,
        "modes": [
            {
                "minimum_free_disk_required_before_start_bytes": start_bytes,
                "mode": "RECOMMENDED_OPERATIONAL",
                "recommended_free_disk_bytes": start_bytes,
                "runtime_low_disk_watermark_bytes": watermark_bytes,
            }
        ],
    }
    path.write_bytes(canonical_json_bytes(value))
    return sha256_file(path)


def _capability(
    *,
    guard: NoRegistrationGuard,
    runtime: Path,
    temporary: Path,
    monitored: Path,
    authorization_path: Path,
    allowlist_sha: str,
    preprocessing_sha: str,
    preprocessing_payload_sha: str,
) -> auth.VerifiedStage2Authorization:
    document = {
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
        "REAL_REGISTRATION_AUTHORIZED": False,
        "STAGE2_DOWNLOAD_AUTHORIZED": True,
        "actual_trials": 0,
        "allowlist_object_count": len(KEYS),
        "allowlist_remote_bytes": sum(len(PAYLOADS[key]) for key in KEYS),
        "allowlist_sha256": allowlist_sha,
        "authorization_payload_sha256": "d" * 64,
        "bucket": "boreas",
        "preprocessing_contract_payload_sha256": preprocessing_payload_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "primary_pair": {
            "map_sequence_id": MAP_SEQUENCE,
            "query_sequence_id": QUERY_SEQUENCE,
        },
        "registration_execution_count": 0,
    }
    authorization_path.write_bytes(canonical_json_bytes(document))
    payload = canonical_json_bytes(document)
    return auth.VerifiedStage2Authorization(
        _token=auth._CAPABILITY_TOKEN,
        _document_bytes=payload,
        repository=REPOSITORY,
        stage1_data_root=runtime.parent / "stage1",
        runtime_root=runtime,
        temporary_root=temporary,
        monitored_disk_path=monitored,
        authorization_path=authorization_path,
        authorization_file_sha256=hashlib.sha256(payload).hexdigest(),
        no_registration_guard=guard,
    )


def _runner_fixture(
    tmp_path: Path,
    guard: NoRegistrationGuard,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fault_hook=None,
    etag_override: Mapping[str, str] | None = None,
    minimum_start_bytes: int = 1,
    runtime_watermark_bytes: int = 1,
    free_bytes_provider=None,
) -> tuple[BoreasV2Stage2Runner, _FakeAws]:
    runtime = tmp_path / "runtime"
    temporary = tmp_path / "temporary"
    stage1 = tmp_path / "stage1"
    for path in (runtime, temporary, stage1):
        path.mkdir(exist_ok=True)
    allowlist_path = tmp_path / "allowlist.csv"
    allowlist_sha = _allowlist(allowlist_path)
    preprocessing_path = tmp_path / "preprocessing.json"
    preprocessing_sha, preprocessing_payload_sha = _preprocessing_contract(
        preprocessing_path
    )
    budget_path = tmp_path / "budget.json"
    budget_sha = _budget(
        budget_path,
        start_bytes=minimum_start_bytes,
        watermark_bytes=runtime_watermark_bytes,
    )
    aws_path = tmp_path / "aws"
    aws_path.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    aws_path.chmod(0o700)
    authorization_path = runtime / "authorization.json"
    capability = _capability(
        guard=guard,
        runtime=runtime,
        temporary=temporary,
        monitored=tmp_path,
        authorization_path=authorization_path,
        allowlist_sha=allowlist_sha,
        preprocessing_sha=preprocessing_sha,
        preprocessing_payload_sha=preprocessing_payload_sha,
    )
    monkeypatch.setattr(
        auth.VerifiedStage2Authorization,
        "assert_live",
        lambda self: self,
    )
    monkeypatch.setattr(
        auth.VerifiedStage2Authorization,
        "assert_operation_live",
        lambda self: self,
    )
    monkeypatch.setattr(
        auth.VerifiedStage2Authorization,
        "bind_disk_gate",
        lambda self, gate: None,
    )
    remote_etags = {**ETAGS, **dict(etag_override or {})}

    def metadata_provider(_allowlist):
        return [
            {
                "etag": remote_etags[key],
                "key": key,
                "last_modified": LAST_MODIFIED,
                "size_bytes": len(PAYLOADS[key]),
            }
            for key in KEYS
        ]

    def preprocessor(downloaded, item) -> MapScanMaterialization:
        ordinal = item.frozen.role_ordinal
        points = np.asarray(
            [[float(ordinal), 0.25, 0.5], [float(ordinal), 1.25, 0.5]],
            dtype="<f8",
        )
        return MapScanMaterialization(
            transformed_xyz=points,
            canonical_source_witness_sha256=hashlib.sha256(
                canonical_float64_npy_bytes(points)
            ).hexdigest(),
            source_point_count=2,
            metadata={"fixture": True, "map_ordinal": ordinal},
        )

    fake = _FakeAws(PAYLOADS)
    config = BoreasV2Stage2RunnerConfig(
        repository=REPOSITORY,
        data_root=stage1,
        runtime_root=runtime,
        temporary_root=temporary,
        monitored_disk_path=tmp_path,
        authorization_path=authorization_path,
        allowlist_path=allowlist_path,
        expected_allowlist_sha256=allowlist_sha,
        disk_budget_path=budget_path,
        expected_disk_budget_sha256=budget_sha,
        preprocessing_contract_path=preprocessing_path,
        aws_executable=aws_path,
        production_mode=False,
    )
    dependencies = BoreasV2Stage2RunnerDependencies(
        authorization_verifier=lambda **_kwargs: capability,
        metadata_provider=metadata_provider,
        map_preprocessor=preprocessor,
        no_registration_guard=guard,
        command_runner=fake,
        free_bytes_provider=free_bytes_provider or (lambda _path: 10**9),
        available_memory_provider=lambda: 10**9,
        now=lambda: datetime(2026, 8, 13, tzinfo=timezone.utc),
        fault_hook=fault_hook,
        synthetic_fixture_only=True,
    )
    return BoreasV2Stage2Runner(config, dependencies), fake


@pytest.mark.parametrize(
    ("fault_label", "expected_download_count", "expected_retry_count"),
    [
        ("AFTER_DOWNLOADED_RECEIPT", 3, 1),
        ("AFTER_REPLAY_COMMIT", 2, 0),
    ],
)
def test_map_crash_resume_preserves_transfer_and_replay_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_label: str,
    expected_download_count: int,
    expected_retry_count: int,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    fired = False

    def fault(label, _item):
        nonlocal fired
        if label == fault_label and not fired:
            fired = True
            raise RuntimeError("synthetic crash")

    with NoRegistrationGuard() as guard:
        runner, first_aws = _runner_fixture(
            tmp_path, guard, monkeypatch, fault_hook=fault
        )
        runner.initialize()
        with pytest.raises(RuntimeError, match="synthetic crash"):
            runner.run_map_ingest()
        runner.close()
        resumed, second_aws = _runner_fixture(tmp_path, guard, monkeypatch)
        state = resumed.initialize()
        summary = resumed.run_map_ingest()
        assert summary.replay_complete is True
        assert state["map_completed_object_count"] == (
            1 if fault_label == "AFTER_REPLAY_COMMIT" else 0
        )
        audit = json.loads(
            (tmp_path / "runtime/evidence/MAP_LIDAR_DOWNLOAD_AUDIT.json").read_text()
        )
        assert audit["successful_download_event_count"] == expected_download_count
        assert audit["retry_download_event_count"] == expected_retry_count
        assert audit["raw_payload_persistent_bytes"] == 0
        assert audit["stream_deleted_raw_bytes"] == expected_download_count * 48
        assert len(first_aws.commands) + len(second_aws.commands) == expected_download_count
        assert resumed.replay.tracks_python_voxel_state is False
        resumed.close()


def test_remote_inventory_resume_change_and_concurrent_runner_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        first, _ = _runner_fixture(tmp_path, guard, monkeypatch)
        first.initialize()
        second, _ = _runner_fixture(tmp_path, guard, monkeypatch)
        with pytest.raises(BoreasV2Stage2RunnerError, match="another Boreas"):
            second.initialize()
        first.close()
        changed, _ = _runner_fixture(
            tmp_path,
            guard,
            monkeypatch,
            etag_override={MAP_KEYS[0]: "f" * 32},
        )
        with pytest.raises(BoreasV2Stage2RunnerError, match="remote inventory differs"):
            changed.initialize()


def test_allocated_replay_resumes_below_start_but_above_watermark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    gib = 1024**3
    live_free = {"bytes": 120 * gib}

    def free(_path: Path) -> int:
        return live_free["bytes"]

    common = {
        "minimum_start_bytes": 100 * gib,
        "runtime_watermark_bytes": 20 * gib,
        "free_bytes_provider": free,
    }
    with NoRegistrationGuard() as guard:
        fresh, _ = _runner_fixture(tmp_path, guard, monkeypatch, **common)
        fresh.initialize()
        fresh.close()
        live_free["bytes"] = 78 * gib
        resumed, _ = _runner_fixture(tmp_path, guard, monkeypatch, **common)
        resumed.initialize()
        assert resumed.disk_gate.events[-2]["operation"] == "RESUME"
        assert resumed.disk_gate.events[-2]["threshold_bytes"] == 20 * gib
        resumed.close()


@pytest.mark.parametrize(
    "crash_label",
    ("AFTER_ALLOCATION_INTENT", "AFTER_REPLAY_ALLOCATION", "AFTER_PLAN_LEDGER"),
)
def test_runner_recovers_replay_preallocation_crash_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_label: str,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    fired = False

    def fault(label, _item):
        nonlocal fired
        if label == crash_label and not fired:
            fired = True
            raise RuntimeError("synthetic allocation crash")

    with NoRegistrationGuard() as guard:
        fresh, _ = _runner_fixture(
            tmp_path, guard, monkeypatch, fault_hook=fault
        )
        with pytest.raises(RuntimeError, match="allocation crash"):
            fresh.initialize()
        resumed, _ = _runner_fixture(tmp_path, guard, monkeypatch)
        state = resumed.initialize()
        assert state["map_completed_object_count"] == 0
        assert not (
            tmp_path / "runtime/map/replay_allocation_intent.json"
        ).exists()
        resumed.close()


@pytest.mark.parametrize("corruption", ["sparse", "wrong-size", "ledger-missing"])
def test_resume_rejects_unauthenticated_replay_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        fresh, _ = _runner_fixture(tmp_path, guard, monkeypatch)
        fresh.initialize()
        fresh.close()
        replay = tmp_path / "runtime/map/transformed_xyz.f64le"
        ledger = tmp_path / "runtime/map/replay_ledger.jsonl"
        if corruption == "sparse":
            replay.write_bytes(b"")
            with replay.open("r+b") as stream:
                stream.truncate(96)
        elif corruption == "wrong-size":
            with replay.open("r+b") as stream:
                stream.truncate(24)
        else:
            ledger.unlink()
        resumed, _ = _runner_fixture(tmp_path, guard, monkeypatch)
        with pytest.raises(BoreasV2Stage2RunnerError, match="resume rejected"):
            resumed.initialize()


def test_complete_map_freezes_one_content_addressed_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        runner, _ = _runner_fixture(tmp_path, guard, monkeypatch)
        runner.initialize()
        assert runner.run_map_ingest().replay_complete is True

        def compiler(*, source: Path, output: Path) -> dict[str, Any]:
            output.write_bytes(b"synthetic reducer binary")
            output.chmod(0o500)
            return {
                "binary_path": str(output),
                "binary_sha256": sha256_file(output),
                "command": ["synthetic-compiler", str(source), str(output)],
                "compiler_version_first_line": "synthetic compiler 1",
                "source_path": str(source),
                "source_sha256": sha256_file(source),
            }

        def builder(**kwargs: Any) -> ExternalTargetMapResult:
            points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype="<f8")
            target = Path(kwargs["target_points_path"])
            kwargs["disk_gate"].before_materialization(
                4096, artifact_id="SYNTHETIC_TARGET"
            )
            atomic_write_bytes(target, canonical_float64_npy_bytes(points))
            return ExternalTargetMapResult(
                target_points_path=target,
                target_points_file_sha256=sha256_file(target),
                target_array_sha256=canonical_array_sha256(points),
                point_count=2,
                voxel_rule_sha256=kwargs["voxel_rule"].contract_sha256,
                reducer_binary_sha256=kwargs["reducer_binary_sha256"],
                range_descriptor_sha256=kwargs["range_descriptor_sha256"],
            )

        runner.dependencies = BoreasV2Stage2RunnerDependencies(
            **{
                **runner.dependencies.__dict__,
                "reducer_compiler": compiler,
                "reducer_builder": builder,
            }
        )
        plan_payload = ReducerResourcePlan.signed_payload(
            replay_plan_sha256=runner.replay.plan_identity["plan_sha256"],
            reducer_source_sha256=sha256_file(default_reducer_source()),
            max_voxels=100,
            estimated_peak_memory_bytes=1000,
            safety_margin_bytes=1000,
            minimum_live_available_memory_bytes=2000,
            capacity_probe_kind="SYNTHETIC_FIXTURE_ONLY",
            capacity_probe_evidence_sha256="e" * 64,
            reducer_binary_projected_bytes=4096,
            production_approved=False,
        )
        plan = ReducerResourcePlan.from_mapping(plan_payload, production_mode=False)
        target_store = tmp_path / "runtime/target_maps"
        outside = tmp_path / "outside-target-store"
        outside.mkdir()
        target_store.rmdir()
        target_store.symlink_to(outside, target_is_directory=True)
        with pytest.raises(
            BoreasV2Stage2RunnerError, match="target publication parent is unsafe"
        ):
            runner.finalize_target_map(plan)
        target_store.unlink()
        target_store.mkdir()
        freeze = runner.finalize_target_map(plan)
        target = tmp_path / "runtime" / freeze["target_map_path"]
        assert target == (
            tmp_path
            / "runtime/target_maps"
            / freeze["target_map_sha256"]
            / "target_points.npy"
        )
        assert target.is_file()
        assert {path.name for path in target.parent.iterdir()} == {
            "metadata.json",
            "target_points.npy",
        }
        copies = [
            path
            for path in (tmp_path / "runtime").rglob("*")
            if path.is_file() and sha256_file(path) == freeze["target_map_sha256"]
        ]
        assert copies == [target]
        assert (tmp_path / "runtime/evidence/map_lineage_manifest.json").is_file()
        assert (
            tmp_path / "runtime/evidence/target_map_reducer_verification.json"
        ).is_file()
        assert runner.finalize_target_map(plan) == freeze
        runner.close()


def test_transfer_intent_recovers_success_before_receipt_sink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bare intent bytes are discarded; only durable DOWNLOADED may resume."""

    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        runner, first_aws = _runner_fixture(tmp_path, guard, monkeypatch)
        runner.initialize()
        item = runner.inventory.for_role("TARGET_MAP")[0]
        runner._write_map_transfer_intent(item)
        retained = runner.downloader.materialize(item)
        assert retained.path.is_file()
        runner.close()

        resumed, second_aws = _runner_fixture(tmp_path, guard, monkeypatch)
        resumed.initialize()
        assert not retained.path.exists()
        assert not (
            tmp_path / "runtime/checkpoints/map_transfer_intent.json"
        ).exists()
        audit = json.loads(
            (tmp_path / "runtime/evidence/MAP_LIDAR_DOWNLOAD_AUDIT.json").read_text()
        )
        assert audit["successful_download_event_count"] == 0
        assert audit["aborted_download_event_count"] == 0
        assert resumed.run_map_ingest().replay_complete is True
        assert len(first_aws.commands) + len(second_aws.commands) == 3
        resumed.close()


def test_finalize_requires_live_lock_and_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    guard = NoRegistrationGuard()
    guard.__enter__()
    runner, _ = _runner_fixture(tmp_path, guard, monkeypatch)
    runner.initialize()
    assert runner.run_map_ingest().replay_complete is True
    plan = ReducerResourcePlan.from_mapping(
        ReducerResourcePlan.signed_payload(
            replay_plan_sha256=runner.replay.plan_identity["plan_sha256"],
            reducer_source_sha256=sha256_file(default_reducer_source()),
            max_voxels=100,
            estimated_peak_memory_bytes=1000,
            safety_margin_bytes=1000,
            minimum_live_available_memory_bytes=2000,
            capacity_probe_kind="SYNTHETIC_FIXTURE_ONLY",
            capacity_probe_evidence_sha256="e" * 64,
            reducer_binary_projected_bytes=4096,
            production_approved=False,
        ),
        production_mode=False,
    )
    guard.__exit__(None, None, None)
    with pytest.raises(BoreasV2Stage2RunnerError, match="lock and NoRegistrationGuard"):
        runner.finalize_target_map(plan)
    runner.close()
    with pytest.raises(BoreasV2Stage2RunnerError, match="lock and NoRegistrationGuard"):
        runner.finalize_target_map(plan)


def test_production_rejects_injected_callbacks_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        fixture, _ = _runner_fixture(tmp_path, guard, monkeypatch)
        production = replace(fixture.config, production_mode=True)
        with pytest.raises(
            BoreasV2Stage2RunnerError, match="ProductionRemoteMetadataProvider"
        ):
            BoreasV2Stage2Runner(production, fixture.dependencies)


def test_capacity_helper_freezes_replay_independent_compiled_layout_plan(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    result = generate_production_reducer_resource_plan(
        output_plan_path=tmp_path / "capacity.json",
        reducer_binary_path=runtime / "map/boreas_stage2_voxel_reduce",
    )
    plan = ReducerResourcePlan.load(
        result["plan_path"], production_mode=True
    )
    assert plan.replay_plan_sha256 is None
    assert plan.reducer_binary_sha256 == result["binary_sha256"]
    assert plan.estimated_peak_memory_bytes == result["capacity_layout"][
        "total_peak_upper_bound_bytes"
    ]
    assert plan.minimum_live_available_memory_bytes > plan.estimated_peak_memory_bytes


def test_formal_authorization_cannot_be_downgraded_by_synthetic_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        runner, _ = _runner_fixture(tmp_path, guard, monkeypatch)
        capability = runner.dependencies.authorization_verifier()
        object.__setattr__(
            capability,
            "_formal_verification_token",
            auth._FORMAL_VERIFICATION_TOKEN,
        )
        with pytest.raises(
            BoreasV2Stage2RunnerError,
            match="cannot be downgraded to non-production",
        ):
            runner.initialize()
