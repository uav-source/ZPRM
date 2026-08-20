from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1 import zero_perturbation_v1_1_r1_runner as runner
from experiments.mid360_formal_batch1.authorization import (
    formal_registration_authorization_verify as authorization_verify,
)
from tests.mid360_formal_batch1.zero_perturbation_v1_1_r1_fixture import (
    COMMIT,
    build_valid_exec_r2_runner_fixture,
    sha,
    write_json,
)


BACKEND_MODULES = {
    "open3d", "phase_a_harness.open3d_backend", "phase_a_harness.pcl_backend",
    "phase_a_harness.common_association_analysis",
}


def _authorization(root: Path, lock_dir: Path) -> Path:
    lock_path = lock_dir / runner.LOCK_FILENAME
    lock = json.loads(lock_path.read_text())
    fingerprint = json.loads((lock_dir / "lock_fingerprint.json").read_text())
    plan = root / lock["bindings"]["trial_plan_json"]["repository_relative_path"]
    runtime = root / lock["authoritative_runtime_root"]
    path = runtime / "authorization" / runner.AUTHORIZATION_FILENAME
    write_json(path, {
        "schema": runner.AUTHORIZATION_SCHEMA,
        "authorization_id": "FMB1-AUTH-" + "1" * 32,
        "status": "ISSUED", "authorization_scope": runner.AUTHORIZATION_SCOPE,
        "track_id": "ZERO_PERTURBATION_TRACK",
        "FORMAL_ICP_UNLOCKED": True, "FORMAL_REGISTRATION_AUTHORIZED": True,
        "lock_fingerprint": fingerprint["lock_fingerprint"],
        "lock_file_sha256": sha(lock_path), "trial_plan_sha256": sha(plan),
        "execution_code_commit": COMMIT, "authorized_trial_count": 360,
        "authorized_backends": list(runner.BACKENDS),
        "issued_at_utc": "2026-08-20T03:00:00+00:00",
    })
    (path.parent / "formal_registration_authorization.sha256").write_text(
        f"{sha(path)}  formal_registration_authorization.json\n", encoding="ascii"
    )
    write_json(path.parent / "authorization_verification_report.json", {
        "status": "PASS", "pass": True,
        "AUTHORIZATION_VERIFICATION_PASS": True,
        "authorization_sha256": sha(path),
        "lock_fingerprint": fingerprint["lock_fingerprint"],
    })
    return path


def _enable_test_authorization_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def verify(*args: object, **kwargs: object) -> dict[str, object]:
        authorization_path = Path(kwargs["authorization_path"])
        if kwargs["requested_mode"] == "resume" and not (
            authorization_path.parent / "authorization_in_use.json"
        ).exists():
            raise authorization_verify.FormalAuthorizationVerificationError(
                "resume requires IN_USE authorization"
            )
        return {
            "status": "PASS", "pass": True,
            "AUTHORIZATION_VERIFICATION_PASS": True,
            "lock_fingerprint": json.loads(
                (Path(kwargs["lock_dir"]) / "lock_fingerprint.json").read_text()
            )["lock_fingerprint"],
        }
    monkeypatch.setattr(
        authorization_verify,
        "verify_formal_registration_authorization",
        verify,
    )


def _identity_core(*args: object, **kwargs: object) -> dict[str, object]:
    return {
        "T_est": runner.IDENTITY, "Delta_T": runner.IDENTITY,
        "translation_x_m": 0.0, "translation_y_m": 0.0,
        "translation_z_m": 0.0, "translation_norm_m": 0.0,
        "rotation_angle_rad": 0.0, "rotation_angle_deg": 0.0,
        "solver_success": True, "finite_result": True, "solver_status": "CONVERGED",
        "initial_correspondence_count": 100,
        "initial_valid_normal_correspondence_count": 95,
        "final_correspondence_count": 90,
        "final_valid_normal_correspondence_count": 85,
        "correspondence_turnover": 0.1, "accepted_source_turnover": 0.2,
        "correspondence_count_change_ratio": -0.1,
        "initial_residual_rmse": 0.01, "final_residual_rmse": 0.009,
        "residual_rmse_change": -0.001,
        "median_normal_angle_change_deg": 0.1,
        "q95_normal_angle_change_deg": 0.2,
        "common_association_valid": True,
        "common_association_invalid_reason": None,
        "common_association_invalid_detail": None,
    }


def _enable_test_execution(
    monkeypatch: pytest.MonkeyPatch, adapter: object,
) -> None:
    """Patch only private post-authorization loaders for lifecycle tests."""

    monkeypatch.setattr(runner, "_verify_execution_code_commit", lambda *args: None)
    monkeypatch.setattr(
        runner, "verify_environment_manifest", lambda *args, **kwargs: {"pass": True}
    )
    monkeypatch.setattr(runner, "_load_authorized_execution_adapter", lambda: adapter)
    _enable_test_authorization_verifier(monkeypatch)


def _runtime_file_bytes(runtime: Path) -> dict[str, bytes]:
    return {
        path.relative_to(runtime).as_posix(): path.read_bytes()
        for path in runtime.rglob("*")
        if path.is_file()
    }


def test_preflight_and_dry_run_never_load_backend_or_write_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    calls = 0
    def denied() -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("backend loader reached")
    monkeypatch.setattr(runner, "_load_authorized_execution_adapter", denied)
    _enable_test_authorization_verifier(monkeypatch)
    before = {name for name in BACKEND_MODULES if name in sys.modules}
    for action in ("preflight", "dry-run"):
        report = runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, runtime_root=runtime, action=action,
            remeasure_environment=False, remeasure_execution_commit=False,
        )
        assert report["REAL_BACKEND_IMPORT_COUNT"] == 0
        assert report["REAL_BACKEND_CALL_COUNT"] == 0
        assert report["actual_formal_trials"] == 0
        assert report["FORMAL_RUN_AUTHORIZATION_VALID"] is False
    assert calls == 0 and not runtime.exists()
    assert {name for name in BACKEND_MODULES if name in sys.modules} == before


def test_execute_without_separate_authorization_fails_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    _enable_test_execution(monkeypatch, _identity_core)
    with pytest.raises(runner.R1RunnerError, match="authorization"):
        runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, runtime_root=runtime, action="execute",
        )
    assert not runtime.exists()


def test_independent_authorization_verifier_unavailable_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    authorization = _authorization(root, lock_dir)
    monkeypatch.setattr(runner, "verify_environment_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        authorization_verify, "verify_formal_registration_authorization", None
    )
    with pytest.raises(runner.R1RunnerError, match="independent authorization"):
        runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, runtime_root=runtime,
            authorization_path=authorization, action="dry-run",
            remeasure_environment=False, remeasure_execution_commit=False,
        )


@pytest.mark.parametrize("injection", ["adapter", "validator", "environment"])
def test_public_execute_rejects_injection_and_disabled_live_environment_before_runtime(
    tmp_path: Path, injection: str,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    kwargs: dict[str, object] = {}
    if injection == "adapter":
        kwargs["execution_adapter"] = _identity_core
    elif injection == "validator":
        kwargs["result_validator"] = lambda *args, **values: values
    else:
        kwargs["remeasure_environment"] = False
    with pytest.raises(
        runner.R1RunnerError,
        match="injected adapter|live environment remeasurement",
    ):
        runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, runtime_root=runtime, action="execute", **kwargs
        )
    assert not runtime.exists()


def test_authorized_fake_execution_fresh_and_completed_resume_are_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    auth = _authorization(root, lock_dir)
    _enable_test_execution(monkeypatch, _identity_core)
    first = runner.preflight_or_dry_run(
        root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
        action="execute", mode="fresh", workers=2,
    )
    assert first["status"] == "COMPLETE"
    assert first["completed_trial_count"] == 360
    assert len(list((runtime / "trial_results").glob("*/attempt-0001.json"))) == 360

    def must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("terminal scientific result was retried")
    monkeypatch.setattr(runner, "_load_authorized_execution_adapter", lambda: must_not_run)
    with pytest.raises(RuntimeError, match="write-once"):
        runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
            action="execute", mode="resume", workers=1,
        )
    assert not list((runtime / "trial_results").glob("*/attempt-0002.json"))


def test_only_infrastructure_failure_is_retried_and_metadata_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    auth = _authorization(root, lock_dir)
    failed_id: str | None = None
    def fail_one(trial: dict[str, object], *args: object) -> dict[str, object]:
        nonlocal failed_id
        if failed_id is None:
            failed_id = str(trial["trial_id"])
            raise runner.AuthorizedBackendInfrastructureError(
                "PROCESS_CRASH", "synthetic infrastructure interruption"
            )
        return _identity_core()
    _enable_test_execution(monkeypatch, fail_one)
    first = runner.preflight_or_dry_run(
        root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
        action="execute", mode="fresh", workers=1,
    )
    assert first["completed_trial_count"] == 359
    assert first["retryable_trial_ids"] == [failed_id]
    monkeypatch.setattr(runner, "_load_authorized_execution_adapter", lambda: _identity_core)
    second = runner.preflight_or_dry_run(
        root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
        action="execute", mode="resume", workers=1,
    )
    assert second["completed_trial_count"] == 360
    assert (runtime / "trial_results" / str(failed_id) / "attempt-0002.json").is_file()

    result = next((runtime / "trial_results").glob("*/attempt-0001.json"))
    payload = json.loads(result.read_text())
    payload["code_commit"] = "b" * 40
    result.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(runner.R1RunnerError, match="execution/environment binding"):
        runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
            action="execute", mode="resume", workers=1,
        )


def test_runtime_root_must_equal_locked_authoritative_location(tmp_path: Path) -> None:
    root, lock_dir, _ = build_valid_exec_r2_runner_fixture(tmp_path)
    with pytest.raises(runner.R1RunnerError, match="authoritative"):
        runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, runtime_root=root / "elsewhere",
            action="dry-run", remeasure_environment=False,
            remeasure_execution_commit=False,
        )


@pytest.mark.parametrize("tamper", ["delete", "authorization", "timestamp"])
def test_resume_rejects_orphan_result_or_invalid_start_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    auth = _authorization(root, lock_dir)
    _enable_test_execution(monkeypatch, _identity_core)
    runner.preflight_or_dry_run(
        root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
        action="execute", mode="fresh", workers=2,
    )
    marker = next((runtime / "inflight").glob("*/attempt-0001.started.json"))
    if tamper == "delete":
        marker.unlink()
    else:
        payload = json.loads(marker.read_text())
        if tamper == "authorization":
            payload["authorization_sha256"] = "0" * 64
        else:
            payload["started_at_utc"] = "not-a-timestamp"
        marker.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(runner.R1RunnerError, match="start marker"):
        runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
            action="execute", mode="resume", workers=1,
        )


@pytest.mark.parametrize("manifest_state", ["missing", "tampered"])
def test_resume_rebuilds_missing_complete_manifest_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_state: str,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    auth = _authorization(root, lock_dir)
    _enable_test_execution(monkeypatch, _identity_core)
    expected = runner.preflight_or_dry_run(
        root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
        action="execute", mode="fresh", workers=2,
    )
    manifest = runtime / "run_manifest.json"
    if manifest_state == "missing":
        manifest.unlink()
        monkeypatch.setattr(
            runner,
            "_load_authorized_execution_adapter",
            lambda: (_ for _ in ()).throw(
                AssertionError("backend reached while rebuilding manifest")
            ),
        )
        with pytest.raises(RuntimeError, match="write-once"):
            runner.preflight_or_dry_run(
                root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
                action="execute", mode="resume", workers=1,
            )
        assert json.loads(manifest.read_text()) == expected
    else:
        payload = json.loads(manifest.read_text())
        payload["trial_plan_sha256"] = "0" * 64
        manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")
        with pytest.raises(runner.R1RunnerError, match="manifest differs"):
            runner.preflight_or_dry_run(
                root, lock_dir=lock_dir, authorization_path=auth,
                runtime_root=runtime, action="execute", mode="resume", workers=1,
            )


def test_terminal_trial_with_extra_orphan_marker_fails_before_any_write_or_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    auth = _authorization(root, lock_dir)
    _enable_test_execution(monkeypatch, _identity_core)
    runner.preflight_or_dry_run(
        root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
        action="execute", mode="fresh", workers=2,
    )
    trial_dir = next((runtime / "inflight").iterdir())
    write_json(trial_dir / "attempt-0002.started.json", {
        "schema": "mid360_fmb1_formal_trial_attempt_start_v1_1_r1",
        "trial_id": trial_dir.name,
        "attempt_number": 2,
        "authorization_sha256": sha(auth),
        "started_at_utc": "2026-08-20T04:00:00+00:00",
    })
    before = _runtime_file_bytes(runtime)
    calls = 0

    def counted_adapter(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _identity_core()

    monkeypatch.setattr(
        runner, "_load_authorized_execution_adapter", lambda: counted_adapter
    )
    with pytest.raises(runner.R1RunnerError, match="terminal scientific outcome"):
        runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
            action="execute", mode="resume", workers=1,
        )
    assert calls == 0
    assert _runtime_file_bytes(runtime) == before


def test_incomplete_results_with_existing_manifest_fail_before_adapter_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    auth = _authorization(root, lock_dir)
    failed = False

    def one_infrastructure_failure(
        trial: dict[str, object], *args: object,
    ) -> dict[str, object]:
        nonlocal failed
        if not failed:
            failed = True
            raise runner.AuthorizedBackendInfrastructureError(
                "PROCESS_CRASH", "synthetic incomplete run"
            )
        return _identity_core()

    _enable_test_execution(monkeypatch, one_infrastructure_failure)
    report = runner.preflight_or_dry_run(
        root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
        action="execute", mode="fresh", workers=1,
    )
    assert report["completed_trial_count"] == 359
    write_json(runtime / "run_manifest.json", {"status": "TAMPERED_PREMATURE"})
    before = _runtime_file_bytes(runtime)
    calls = 0

    def counted_adapter(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _identity_core()

    monkeypatch.setattr(
        runner, "_load_authorized_execution_adapter", lambda: counted_adapter
    )
    with pytest.raises(runner.R1RunnerError, match="manifest exists before"):
        runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
            action="execute", mode="resume", workers=1,
        )
    assert calls == 0
    assert _runtime_file_bytes(runtime) == before


@pytest.mark.parametrize(
    "tamper", ["unexpected", "dangling_symlink", "contract_symlink", "manifest_directory"]
)
def test_resume_runtime_layout_tamper_fails_before_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    root, lock_dir, runtime = build_valid_exec_r2_runner_fixture(tmp_path)
    auth = _authorization(root, lock_dir)
    _enable_test_execution(monkeypatch, _identity_core)
    runner.preflight_or_dry_run(
        root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
        action="execute", mode="fresh", workers=2,
    )
    if tamper == "unexpected":
        (runtime / "unexpected.json").write_text("{}\n")
    elif tamper == "dangling_symlink":
        (runtime / "dangling").symlink_to(runtime / "missing")
    elif tamper == "contract_symlink":
        contract = runtime / "run_contract.json"
        contract.unlink()
        contract.symlink_to(lock_dir / runner.LOCK_FILENAME)
    else:
        manifest = runtime / "run_manifest.json"
        manifest.unlink()
        manifest.mkdir()
    calls = 0

    def counted_adapter(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _identity_core()

    monkeypatch.setattr(
        runner, "_load_authorized_execution_adapter", lambda: counted_adapter
    )
    with pytest.raises(runner.R1RunnerError, match="runtime|run_contract|run_manifest"):
        runner.preflight_or_dry_run(
            root, lock_dir=lock_dir, authorization_path=auth, runtime_root=runtime,
            action="execute", mode="resume", workers=1,
        )
    assert calls == 0
