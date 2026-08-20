from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import tools.mid360_formal_batch1.preflight_formal_registration as preflight_cli

from experiments.mid360_formal_batch1.registration_firewall import (
    BackendExecutionBlocked,
    FIXTURE_SCOPE,
    REAL_SCOPE,
    RealPreBackendGuard,
    RegistrationFirewallError,
    assert_fixture_only_isolated,
    assert_prebackend_static_safe,
    evaluate_real_batch_preflight,
    preflight_real_batch,
    scan_registration_artifacts,
    write_no_icp_attestation_tonight,
    write_preflight_reports,
)


REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def real_guard_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZPRM_FMB1_NO_FORMAL_REGISTRATION", "1")
    monkeypatch.setenv("NO_FORMAL_REGISTRATION", "true")
    monkeypatch.setenv("FMB1_EXECUTION_SCOPE", REAL_SCOPE)
    for name in (
        "FMB1_FORMAL_REGISTRATION_AUTHORIZED",
        "FORMAL_REGISTRATION_AUTHORIZED",
        "FMB1_FORMAL_ICP_UNLOCKED",
        "FORMAL_ICP_UNLOCKED",
        "FMB1_BACKEND_AUTHORIZED",
        "FMB1_FIXTURE_ONLY",
        "FMB1_SYNTHETIC_FIXTURE_ONLY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_current_real_batch_is_blocked_only_for_missing_w04(
    real_guard_env: None, tmp_path: Path
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    report = preflight_real_batch(REPOSITORY, proc_root=proc_root)
    assert report["status"] == "BLOCKED"
    assert report["CURRENT_REAL_BATCH_BLOCKED"] is True
    assert report["CURRENT_BLOCK_REASON"] == "MISSING_ADMITTED_WEAK_REPLACEMENT_W04"
    assert report["scientific_blocker_codes"] == [
        "MISSING_ADMITTED_WEAK_REPLACEMENT_W04"
    ]
    assert report["integrity_blockers"] == []
    assert report["security_blockers"] == []
    assert report["admitted_weak_scene_count"] == 2
    assert report["required_weak_scene_count"] == 3
    assert report["admitted_weak_scene_ids"] == ["FMB1_W01", "FMB1_W03"]
    qualification_scan = report["qualification_runtime_artifact_scan"]
    assert qualification_scan["status"] == "PASS"
    assert qualification_scan["scanned_file_count"] == 1129
    assert qualification_scan["excluded_nonreal_artifact_count"] == 1128
    assert qualification_scan["real_registration_artifact_count"] == 0
    for field in (
        "FORMAL_RUN_MATRIX_ISSUED",
        "FORMAL_LOCK_ISSUED",
        "TRIAL_MATRIX_ISSUED",
        "FORMAL_BATCH_LOCK_ISSUED",
        "FORMAL_REGISTRATION_AUTHORIZED",
        "FORMAL_ICP_UNLOCKED",
        "backend_invoked",
    ):
        assert report[field] is False
    for field in (
        "real_open3d_registration_call_count",
        "real_pcl_cli_invocation_count",
        "real_other_registration_process_count",
        "real_formal_trial_count",
    ):
        assert report[field] == 0


def test_reports_and_tonight_attestation_issue_no_matrix_or_lock(
    real_guard_env: None, tmp_path: Path
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    report = preflight_real_batch(REPOSITORY, proc_root=proc_root)
    output = tmp_path / "prebackend_qualification_v1"
    written = write_preflight_reports(report, output)
    report["written_reports"] = written
    attestation = write_no_icp_attestation_tonight(report, output)
    assert Path(written["json_path"]).name == "current_real_batch_preflight_report.json"
    assert Path(written["markdown_path"]).name == "current_real_batch_preflight_report.md"
    assert Path(attestation["path"]).name == "NO_ICP_ATTESTATION_TONIGHT.json"
    assert attestation["status"] == "PASS"
    assert attestation["pass"] is True
    assert attestation["real_T_est_file_count"] == 0
    assert attestation["qualification_runtime_artifact_scan_status"] == "PASS"
    assert attestation["qualification_runtime_scanned_file_count"] == 1129
    assert attestation["qualification_runtime_fixture_exclusion_count"] == 1128
    assert attestation["qualification_runtime_real_registration_artifact_count"] == 0
    assert attestation["CURRENT_BLOCK_REASON"] == "MISSING_ADMITTED_WEAK_REPLACEMENT_W04"
    assert not (output / "formal_batch1_lock.json").exists()
    assert not (output / "formal_registration_trial_matrix.json").exists()


def test_cli_without_no_backend_ack_fails_before_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "must_not_exist"
    code = preflight_cli.main(
        [
            "--repository-root",
            str(REPOSITORY),
            "--results-root",
            str(REPOSITORY / "results/mid360_formal_batch1"),
            "--output-dir",
            str(output),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["CURRENT_BLOCK_REASON"] == "EXPLICIT_NO_BACKEND_ACK_REQUIRED"
    assert payload["evidence_written"] is False
    assert not output.exists()


def test_fake_real_result_is_not_accepted(tmp_path: Path) -> None:
    fake_results = tmp_path / "claimed_real"
    fake_runtime = tmp_path / "claimed_runtime"
    fake_results.mkdir()
    fake_runtime.mkdir()
    (fake_results / "fmb1_pre_registration_readiness.json").write_text(
        json.dumps(
            {
                "FMB1_PRE_REGISTRATION_DATA_READY": True,
                "FORMAL_REGISTRATION_AUTHORIZED": True,
                "weak_scene_count": 3,
            }
        ),
        encoding="utf-8",
    )
    report = evaluate_real_batch_preflight(
        REPOSITORY,
        results_dir=fake_results,
        runtime_dir=fake_runtime,
    )
    assert report["pass"] is False
    codes = {row["code"] for row in report["security_blockers"]}
    assert "NONCANONICAL_REAL_RESULTS_DIRECTORY" in codes
    assert "REGISTRATION_OR_AUTHORITY_ARTIFACT_PRESENT" in codes
    assert report["FORMAL_REGISTRATION_AUTHORIZED"] is False


def test_real_artifact_in_qualification_runtime_blocks_preflight(
    tmp_path: Path,
) -> None:
    qualification_runtime = tmp_path / "qualification_runtime"
    qualification_runtime.mkdir()
    (qualification_runtime / "fake_real_result.json").write_text(
        json.dumps({"real_batch": True, "T_est": [[1, 0, 0, 0]] * 4}),
        encoding="utf-8",
    )
    report = evaluate_real_batch_preflight(
        REPOSITORY,
        qualification_runtime_dir=qualification_runtime,
    )
    codes = {row["code"] for row in report["security_blockers"]}
    assert "QUALIFICATION_RUNTIME_REGISTRATION_ARTIFACT_PRESENT" in codes
    assert report["qualification_runtime_artifact_scan"]["pass"] is False
    assert report["qualification_runtime_artifact_scan"][
        "real_registration_artifact_count"
    ] > 0


def test_loaded_open3d_call_is_blocked_before_original(
    real_guard_env: None, tmp_path: Path
) -> None:
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))

    registration = SimpleNamespace(registration_icp=original)
    fake_open3d = SimpleNamespace(
        pipelines=SimpleNamespace(registration=registration)
    )
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    guard = RealPreBackendGuard(proc_root=proc_root, open3d_module=fake_open3d)
    with guard:
        with pytest.raises(BackendExecutionBlocked, match="Open3D backend call blocked"):
            registration.registration_icp("source", "target")
    assert calls == []
    assert registration.registration_icp is original
    report = guard.report()
    assert report["blocked_open3d_attempt_count"] == 1
    assert report["real_open3d_registration_call_count"] == 0


def test_pcl_cli_is_blocked_before_subprocess(
    real_guard_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(subprocess, "run", fake_run)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    guard = RealPreBackendGuard(proc_root=proc_root)
    with guard:
        with pytest.raises(BackendExecutionBlocked, match="backend process blocked"):
            subprocess.run(["pcl_point_to_plane_cli", "source.npy", "target.npy"])
    assert calls == []
    report = guard.report()
    assert report["blocked_pcl_attempt_count"] == 1
    assert report["real_pcl_cli_invocation_count"] == 0


def test_schema_and_explicit_fixture_are_excluded_but_fake_result_is_caught(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe"
    safe.mkdir()
    (safe / "formal_trial_result_schema_v1.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "properties": {
                    "T_est": {"type": "array"},
                    "FORMAL_REGISTRATION_AUTHORIZED": {"const": True},
                },
            }
        ),
        encoding="utf-8",
    )
    (safe / "fixture_result.json").write_text(
        json.dumps(
            {
                "FIXTURE_ONLY": True,
                "FIXTURE_ONLY_DO_NOT_CITE": True,
                "NOT_REAL_FMB1": True,
                "NOT_FORMAL_MEASUREMENT": True,
                "ACTUAL_REGISTRATION_EXECUTION": False,
                "FORMAL_REGISTRATION_AUTHORIZED": False,
                "FORMAL_ICP_UNLOCKED": False,
                "T_est": [[1, 0, 0, 0]] * 4,
            }
        ),
        encoding="utf-8",
    )
    (safe / "formal_batch1_lock_template_UNISSUED.json").write_text(
        json.dumps(
            {
                "TEMPLATE_ONLY": True,
                "LOCK_STATUS": "UNISSUED",
                "FORMAL_LOCK_ISSUED": False,
                "FORMAL_REGISTRATION_AUTHORIZED": False,
                "FORMAL_ICP_UNLOCKED": False,
                "actual_formal_trials": 0,
            }
        ),
        encoding="utf-8",
    )
    safe_report = scan_registration_artifacts([safe])
    assert safe_report["pass"] is True
    assert safe_report["excluded_nonreal_artifact_count"] == 3

    (safe / "fake_real_result.json").write_text(
        json.dumps(
            {
                "real_batch": True,
                "FORMAL_REGISTRATION_AUTHORIZED": False,
                "T_est": [[1, 0, 0, 0]] * 4,
            }
        ),
        encoding="utf-8",
    )
    unsafe_report = scan_registration_artifacts([safe])
    assert unsafe_report["pass"] is False
    assert any(
        row["kind"] == "BACKEND_RESULT_FIELD"
        for row in unsafe_report["artifacts"]
    )


def test_authorization_environment_and_real_lock_are_hard_failures(
    real_guard_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    monkeypatch.setenv("FORMAL_REGISTRATION_AUTHORIZED", "true")
    with pytest.raises(RegistrationFirewallError, match="authorization environment"):
        with RealPreBackendGuard(proc_root=proc_root):
            pass

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "formal_batch1_lock.json").write_text(
        json.dumps(
            {
                "template_unissued": True,
                "issued": False,
                "eligible_for_real_authorization": False,
                "FORMAL_REGISTRATION_AUTHORIZED": False,
            }
        ),
        encoding="utf-8",
    )
    artifact_report = scan_registration_artifacts([artifact_root])
    assert artifact_report["pass"] is False
    assert any(
        row["kind"] == "AUTHORIZATION_MATRIX_OR_LOCK_ARTIFACT"
        for row in artifact_report["artifacts"]
    )


def test_registration_result_used_false_is_exact_negative_evidence(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    path = artifact_root / "proposal_difference_report.json"
    path.write_text(json.dumps({
        "scientific_change_timing": {"registration_result_used": False},
    }))
    assert scan_registration_artifacts([artifact_root])["pass"] is True

    path.write_text(json.dumps({
        "scientific_change_timing": {"registration_result_used": True},
    }))
    positive = scan_registration_artifacts([artifact_root])
    assert positive["pass"] is False
    assert any(
        row["kind"] == "BACKEND_RESULT_FIELD"
        and row.get("json_path")
        == "$.scientific_change_timing.registration_result_used"
        for row in positive["artifacts"]
    )


def test_negative_evidence_does_not_hide_other_result_fields(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "proposal_difference_report.json").write_text(json.dumps({
        "scientific_change_timing": {"registration_result_used": False},
        "nested": {"T_est": [[1, 0, 0, 0]] * 4},
    }))
    report = scan_registration_artifacts([artifact_root])
    assert report["pass"] is False
    assert any(
        row["kind"] == "BACKEND_RESULT_FIELD"
        and row.get("json_path") == "$.nested.T_est"
        for row in report["artifacts"]
    )


def test_primary_no_registration_environment_is_mandatory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZPRM_FMB1_NO_FORMAL_REGISTRATION", raising=False)
    monkeypatch.setenv("NO_FORMAL_REGISTRATION", "true")
    monkeypatch.setenv("FMB1_EXECUTION_SCOPE", REAL_SCOPE)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    with pytest.raises(RegistrationFirewallError, match="ZPRM_FMB1_NO_FORMAL_REGISTRATION"):
        with RealPreBackendGuard(proc_root=proc_root):
            pass


def test_static_scan_rejects_backend_import(tmp_path: Path) -> None:
    for relative in (
        Path("experiments/mid360_formal_batch1/registration_firewall.py"),
        Path("tools/mid360_formal_batch1/preflight_formal_registration.py"),
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((REPOSITORY / relative).read_text(encoding="utf-8"), encoding="utf-8")
    bad = tmp_path / "tools/mid360_formal_batch1/run_formal_registration.py"
    bad.write_text("import open3d\nopen3d.pipelines.registration.registration_icp()\n", encoding="utf-8")
    with pytest.raises(RegistrationFirewallError, match="unsafe real pre-backend source"):
        assert_prebackend_static_safe(tmp_path)


def test_fixture_root_is_explicitly_nonreal(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    report = assert_fixture_only_isolated(REPOSITORY, fixture)
    assert report["execution_scope"] == FIXTURE_SCOPE
    assert report["fixture_only"] is True
    assert report["eligible_for_real_authorization"] is False
    assert report["fixture_counters_contribute_to_real_batch"] is False
    with pytest.raises(RegistrationFirewallError, match="overlaps real FMB1"):
        assert_fixture_only_isolated(
            REPOSITORY, REPOSITORY / "results/mid360_formal_batch1"
        )
