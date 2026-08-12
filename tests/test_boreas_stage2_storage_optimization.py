from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from phase_a_harness.real_data_preparation import (
    boreas_stage2_storage_optimization as producer,
)
from phase_a_harness.real_data_preparation.io import sha256_file


REPOSITORY = Path(__file__).resolve().parents[1]
STAGE1 = REPOSITORY / producer.STAGE1_RELATIVE_ROOT


def _read(name: str) -> dict[str, object]:
    return json.loads((STAGE1 / name).read_text(encoding="utf-8"))


def _fixture_repository_gate() -> dict[str, object]:
    """A closed fixture gate without rerunning the historical branch-only producer gate."""

    return {
        "branch": producer.EXPECTED_BRANCH,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "protected_tags": dict(producer.EXPECTED_TAG_COMMITS),
        "worktree_clean": True,
    }


def test_storage_audit_binds_exact_stage1_science_and_allowlist() -> None:
    value = producer.inspect_frozen_stage1(REPOSITORY)
    assert value["primary_pair"] == {
        "map_sequence_id": producer.PRIMARY_MAP_SEQUENCE,
        "query_sequence_id": producer.PRIMARY_QUERY_SEQUENCE,
    }
    assert value["allowlist_object_count"] == 20_061
    assert value["allowlist_remote_bytes"] == 104_158_637_472
    assert value["map_object_count"] == 8_202
    assert value["query_object_count"] == 11_859


def test_storage_audit_stage1_hashes_are_literal_immutable_bindings() -> None:
    expected = {
        "frozen_manifest.json": producer.STAGE1_FROZEN_MANIFEST_FILE_SHA256,
        "boreas_v2_pair_selection.json": producer.STAGE1_PAIR_SELECTION_SHA256,
        "boreas_v2_stage2_download_allowlist.csv": producer.STAGE1_ALLOWLIST_SHA256,
        "boreas_v2_stage2_download_plan.json": producer.STAGE1_DOWNLOAD_PLAN_SHA256,
        "boreas_v2_stage2_disk_budget.json": producer.STAGE1_OLD_BUDGET_SHA256,
        "NO_ICP_ATTESTATION.json": producer.STAGE1_NO_ICP_SHA256,
        "NO_LIDAR_PAYLOAD_ATTESTATION.json": producer.STAGE1_NO_LIDAR_SHA256,
    }
    assert {name: sha256_file(STAGE1 / name) for name in expected} == expected


def test_old_budget_breakdown_closes_without_inventing_backend_or_target_copies() -> None:
    rows = producer._reasoned_original_breakdown(
        _read("boreas_v2_stage2_disk_budget.json")
    )
    by_name = {row["component"]: row for row in rows}
    assert sum(row["peak_contribution_bytes"] for row in rows) == 1_063_424_189_088
    assert by_name["Open3D copies"]["peak_contribution_bytes"] == 0
    assert by_name["PCL copies"]["peak_contribution_bytes"] == 0
    assert by_name["per-snapshot target copies"]["peak_contribution_bytes"] == 0
    assert "does not encode 100" in by_name["per-snapshot target copies"]["reason"]


def test_contracts_preserve_single_target_and_shared_backend_inputs() -> None:
    stage1 = producer.inspect_frozen_stage1(REPOSITORY)
    canonical = producer._canonical_bundle_contract()
    target = producer._streaming_map_contract(stage1)
    assert canonical["future_open3d_source_sha256"] == canonical[
        "future_pcl_source_sha256"
    ]
    assert canonical["future_open3d_target_sha256"] == canonical[
        "future_pcl_target_sha256"
    ]
    addressing = target["content_addressing"]
    assert addressing == {
        "planned_physical_target_map_copy_count": 1,
        "planned_snapshot_target_reference_count": 100,
        "planned_unique_target_map_count": 1,
        "snapshot_target_storage": "SHA256_REFERENCE_ONLY",
    }


def test_storage_audit_never_assigns_unfrozen_preprocessing_values() -> None:
    value = producer._streaming_map_contract(producer.inspect_frozen_stage1(REPOSITORY))
    assert value["storage_planner_parameter_authority"] == "NONE"
    assert set(value["scientific_preprocessing_parameters"].values()) == {
        producer.PREPROCESSING_UNRESOLVED
    }


def test_two_pass_query_contract_preserves_r07_r08_r14_and_zero_actuals() -> None:
    value = producer._streaming_query_contract(producer.inspect_frozen_stage1(REPOSITORY))
    assert value["actual_geometry_metric_row_count"] == 0
    assert value["actual_selected_source_count"] == 0
    assert value["selection_contract"]["planned_snapshot_count"] == 100
    assert value["selection_contract"]["registration_derived_fields_forbidden"] is True
    assert all(
        str(value["two_pass_adjudication"][name]).startswith("PASS")
        for name in ("R07", "R08", "R14", "reproducibility", "sha_auditability")
    )


def test_checkpoint_and_temp_contracts_are_fail_closed() -> None:
    checkpoint = producer._checkpoint_contract()
    temporary = producer._temp_contract()
    assert checkpoint["strict_resume_implemented"] is True
    assert checkpoint["identity_change_semantics"].startswith("FAIL")
    assert checkpoint["orphan_state"].startswith("REJECT")
    assert "O_APPEND" in checkpoint["append_semantics"]
    assert checkpoint["actual_completed_selected_source_count"] == 0
    assert checkpoint["checkpoint_files"] == [
        "processed_map_objects.jsonl",
        "processed_query_objects.jsonl",
        "processed_selected_sources.jsonl",
    ]
    assert set(temporary["directories"]) == {"tmp_download", "tmp_decode", "tmp_pcl"}
    assert temporary["frozen_manifest_inclusion"] is False


def test_all_readiness_authorization_and_execution_fields_stay_zero() -> None:
    zeros = producer._attestation_zeros()
    assert all(value is False for key, value in zeros.items() if key.endswith("AUTHORIZED"))
    assert all(value == 0 for key, value in zeros.items() if not key.endswith("AUTHORIZED"))


def _junit(path: Path, *, tests: int = 811, failures: int = 0, skipped: int = 10) -> Path:
    cases = "".join(
        (
            f'<testcase classname="synthetic" name="case-{index}">'
            + ("<failure />" if index < failures else "")
            + ("<skipped />" if failures <= index < failures + skipped else "")
            + "</testcase>"
        )
        for index in range(tests)
    )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites><testsuite name="pytest" errors="0" failures="{failures}" '
        f'skipped="{skipped}" tests="{tests}">{cases}</testsuite></testsuites>',
        encoding="utf-8",
    )
    return path


def test_source_only_junit_counts_are_independently_authenticated(tmp_path: Path) -> None:
    report = _junit(tmp_path / "pytest.xml")
    value = producer.authenticate_pytest_junit(
        report_path=report, expected_head="a" * 40
    )
    assert (value["collected"], value["passed"], value["skipped"]) == (811, 801, 10)
    failed = _junit(tmp_path / "failed.xml", failures=1)
    with pytest.raises(producer.BoreasStage2StorageOptimizationError, match="not clean"):
        producer.authenticate_pytest_junit(report_path=failed, expected_head="a" * 40)


def test_environment_gates_are_both_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", raising=False)
    monkeypatch.setenv("ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD", "1")
    with pytest.raises(producer.BoreasStage2StorageOptimizationError, match="environment"):
        producer._assert_environment()


def test_metadata_only_producer_builds_complete_zero_execution_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    monkeypatch.setenv("ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD", "1")
    recorded_gate = _fixture_repository_gate()
    monkeypatch.setattr(
        producer,
        "inspect_repository_gate",
        lambda *_args, **_kwargs: recorded_gate,
    )
    runtime = tmp_path / "runtime"
    summary = producer.build_boreas_stage2_storage_optimization(
        repository=REPOSITORY,
        data_root=Path.home() / "zero_perturbation_data/boreas_stage1_v1",
        runtime_root=runtime,
        pytest_junit_xml=_junit(tmp_path / "pytest.xml"),
        require_clean_worktree=False,
    )
    assert summary["STAGE2_STORAGE_PLAN_READY"] is True
    assert summary["CURRENT_DISK_SUFFICIENT"] is True
    assert summary["answer_count"] == 27
    assert summary["snapshot_count"] == 0
    assert summary["registration_execution_count"] == 0
    assert summary["downloaded_lidar_payload_count"] == 0
    assert summary["real_stage2_execution_runner_in_scope"] is False
    assert summary["runtime_disk_gate_contract_ready"] is True
    assert summary["runtime_disk_gate_runner_integration_required"] is True
    assert len(list(runtime.iterdir())) == 24
    assert len((runtime / "SHA256SUMS").read_text(encoding="utf-8").splitlines()) == 23
    frozen = tmp_path / "frozen"
    producer.freeze_boreas_stage2_storage_optimization(
        repository=REPOSITORY,
        data_root=Path.home() / "zero_perturbation_data/boreas_stage1_v1",
        runtime_root=runtime,
        frozen_root=frozen,
    )
    assert len(list(frozen.iterdir())) == 24
    assert producer._assert_no_local_lidar_payload(
        Path.home() / "zero_perturbation_data/boreas_stage1_v1"
    ) == []
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    monkeypatch.delenv("ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD", raising=False)
    with pytest.raises(producer.BoreasStage2StorageOptimizationError, match="environment"):
        producer._assert_environment()
