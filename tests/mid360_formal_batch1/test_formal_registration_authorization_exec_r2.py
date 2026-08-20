from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.authorization import (
    authorization_lifecycle as lifecycle,
    formal_registration_authorization as producer,
    formal_registration_authorization_verify as verifier,
)
from experiments.mid360_formal_batch1 import zero_perturbation_v1_1_exec_r2_verify


SHA = "1" * 64
COMMIT = "2" * 40
FINGERPRINT = "3" * 64
RUNTIME_RELATIVE = verifier.EXPECTED_RUNTIME


def _payload() -> dict[str, object]:
    return {
        "schema": verifier.AUTHORIZATION_SCHEMA,
        "authorization_id": "FMB1-AUTH-" + "4" * 32,
        "authorization_type": "ONE_TIME_FMB1_ZERO_PERTURBATION_FORMAL_EXECUTION",
        "authorization_basis": "EXPLICIT_USER_INSTRUCTION_RECORDED_BY_OPERATOR",
        "issued_at_utc": "2026-08-20T04:00:00+00:00",
        "nonce": "5" * 64,
        "status": "ISSUED",
        "track_id": "ZERO_PERTURBATION_TRACK",
        "active_amendment_id": "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1",
        "lock_revision": 2,
        "lock_fingerprint": FINGERPRINT,
        "lock_file_sha256": SHA,
        "lock_release_commit": COMMIT,
        "trial_plan_sha256": SHA,
        "analysis_contract_sha256": SHA,
        "backend_contract_sha256": SHA,
        "execution_code_commit": COMMIT,
        "environment_manifest_sha256": SHA,
        "pcl_binary_sha256": SHA,
        "planned_trial_count": 360,
        "planned_open3d_count": 180,
        "planned_pcl_count": 180,
        "allowed_backends": list(verifier.BACKENDS),
        "T0_policy": "IDENTITY_ONLY",
        "execution_mode_initial": "fresh",
        "resume_allowed": True,
        "workers": 2,
        "capture_radius_authorized": False,
        "other_backend_authorized": False,
        "parameter_change_authorized": False,
        "trial_reselection_authorized": False,
        "authorization_scope": "ONE_FORMAL_EXECUTION_ATTEMPT_WITH_RESUME_ONLY",
        "authoritative_runtime_root": RUNTIME_RELATIVE,
        "immutable": True,
        "FORMAL_ICP_UNLOCKED": True,
        "FORMAL_REGISTRATION_AUTHORIZED": True,
    }


def _write_authorization(root: Path, payload: dict[str, object] | None = None) -> Path:
    authorization = root / RUNTIME_RELATIVE / "authorization" / verifier.AUTHORIZATION_FILENAME
    authorization.parent.mkdir(parents=True, exist_ok=True)
    content = verifier.canonical_json_bytes(payload or _payload())
    authorization.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    (authorization.parent / verifier.AUTHORIZATION_SHA_FILENAME).write_text(
        f"{digest}  {verifier.AUTHORIZATION_FILENAME}\n", encoding="ascii"
    )
    return authorization


def _enable_isolated_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def verify_lock(*args: object, **kwargs: object) -> tuple[dict[str, object], str]:
        payload = args[2]
        assert isinstance(payload, dict)
        if payload["lock_fingerprint"] != FINGERPRINT:
            verifier._fail("authorization lock fingerprint differs")
        if payload["lock_file_sha256"] != SHA:
            verifier._fail("authorization lock SHA differs")
        if payload["execution_code_commit"] != COMMIT:
            verifier._fail("authorization execution commit differs")
        if payload["lock_release_commit"] != COMMIT:
            verifier._fail("authorization lock-release commit differs")
        return ({"bindings": {
            "analysis_contract": {"sha256": SHA},
            "backend_parameter_contract": {"sha256": SHA},
        }}, FINGERPRINT)

    def verify_plan(
        root: Path, lock: dict[str, object], payload: dict[str, object]
    ) -> None:
        expected = {
            "trial_plan_sha256": SHA,
            "analysis_contract_sha256": SHA,
            "backend_contract_sha256": SHA,
            "environment_manifest_sha256": SHA,
            "pcl_binary_sha256": SHA,
        }
        for key, value in expected.items():
            if payload[key] != value:
                verifier._fail(f"authorization binding differs: {key}")

    monkeypatch.setattr(
        verifier, "_verify_git_and_lock",
        verify_lock,
    )
    monkeypatch.setattr(verifier, "_verify_plan_and_environment", verify_plan)


def _verify(
    root: Path, authorization: Path, *, mode: str = "fresh", workers: int = 2,
) -> dict[str, object]:
    lock_dir = root / "lock"
    lock_dir.mkdir(exist_ok=True)
    return verifier.verify_formal_registration_authorization(
        root,
        lock_dir=lock_dir,
        authorization_path=authorization,
        runtime_root=root / RUNTIME_RELATIVE,
        requested_mode=mode,
        workers=workers,
    )


TAMPERS = [
    ("missing_schema", lambda p: p.pop("schema")),
    ("extra_key", lambda p: p.__setitem__("extra", False)),
    ("schema", lambda p: p.__setitem__("schema", "wrong")),
    ("authorization_id", lambda p: p.__setitem__("authorization_id", "bad")),
    ("authorization_type", lambda p: p.__setitem__("authorization_type", "bad")),
    ("basis", lambda p: p.__setitem__("authorization_basis", "bad")),
    ("timestamp", lambda p: p.__setitem__("issued_at_utc", "not-time")),
    ("nonce", lambda p: p.__setitem__("nonce", "0")),
    ("status", lambda p: p.__setitem__("status", "CONSUMED")),
    ("track", lambda p: p.__setitem__("track_id", "CAPTURE_RADIUS_TRACK")),
    ("amendment", lambda p: p.__setitem__("active_amendment_id", "wrong")),
    ("revision", lambda p: p.__setitem__("lock_revision", 1)),
    ("lock_fingerprint", lambda p: p.__setitem__("lock_fingerprint", "0" * 64)),
    ("lock_sha", lambda p: p.__setitem__("lock_file_sha256", "0" * 64)),
    ("release_commit", lambda p: p.__setitem__("lock_release_commit", "0" * 40)),
    ("plan_sha", lambda p: p.__setitem__("trial_plan_sha256", "0" * 64)),
    ("analysis_sha", lambda p: p.__setitem__("analysis_contract_sha256", "0" * 64)),
    ("backend_sha", lambda p: p.__setitem__("backend_contract_sha256", "0" * 64)),
    ("execution_commit", lambda p: p.__setitem__("execution_code_commit", "0" * 40)),
    ("environment_sha", lambda p: p.__setitem__("environment_manifest_sha256", "0" * 64)),
    ("pcl_sha", lambda p: p.__setitem__("pcl_binary_sha256", "0" * 64)),
    ("total", lambda p: p.__setitem__("planned_trial_count", 359)),
    ("open3d", lambda p: p.__setitem__("planned_open3d_count", 179)),
    ("pcl", lambda p: p.__setitem__("planned_pcl_count", 181)),
    ("backends", lambda p: p.__setitem__("allowed_backends", ["OPEN3D_POINT_TO_PLANE"])),
    ("t0", lambda p: p.__setitem__("T0_policy", "NONZERO")),
    ("resume", lambda p: p.__setitem__("resume_allowed", False)),
    ("workers", lambda p: p.__setitem__("workers", 4)),
    ("capture", lambda p: p.__setitem__("capture_radius_authorized", True)),
    ("other_backend", lambda p: p.__setitem__("other_backend_authorized", True)),
    ("parameter", lambda p: p.__setitem__("parameter_change_authorized", True)),
    ("reselection", lambda p: p.__setitem__("trial_reselection_authorized", True)),
    ("scope", lambda p: p.__setitem__("authorization_scope", "UNBOUNDED")),
    ("runtime", lambda p: p.__setitem__("authoritative_runtime_root", "tmp")),
    ("immutable", lambda p: p.__setitem__("immutable", False)),
    ("unlock", lambda p: p.__setitem__("FORMAL_ICP_UNLOCKED", False)),
    ("authorized", lambda p: p.__setitem__("FORMAL_REGISTRATION_AUTHORIZED", False)),
]


@pytest.mark.parametrize(("name", "mutate"), TAMPERS, ids=[row[0] for row in TAMPERS])
def test_independent_verifier_rejects_authorization_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    name: str, mutate: object,
) -> None:
    _enable_isolated_verifier(monkeypatch)
    payload = _payload()
    mutate(payload)  # type: ignore[operator]
    authorization = _write_authorization(tmp_path, payload)
    with pytest.raises(verifier.FormalAuthorizationVerificationError):
        _verify(tmp_path, authorization)


def test_independent_verifier_accepts_fresh_then_only_same_attempt_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_isolated_verifier(monkeypatch)
    authorization = _write_authorization(tmp_path)
    (tmp_path / "lock").mkdir()
    report = _verify(tmp_path, authorization)
    assert report["AUTHORIZATION_VERIFICATION_PASS"] is True
    marker = lifecycle.mark_authorization_in_use(
        authorization,
        lock_fingerprint=FINGERPRINT,
        runtime_relative=RUNTIME_RELATIVE,
        started_at_utc="2026-08-20T04:01:00+00:00",
    )
    assert marker["state"] == "IN_USE"
    with pytest.raises(verifier.FormalAuthorizationVerificationError, match="resume only"):
        _verify(tmp_path, authorization, mode="fresh")
    resumed = _verify(tmp_path, authorization, mode="resume")
    assert resumed["lifecycle_state"] == "IN_USE_RESUME_SAME_ATTEMPT"


def test_consumption_is_irreversible_and_sha_is_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_isolated_verifier(monkeypatch)
    authorization = _write_authorization(tmp_path)
    lifecycle.mark_authorization_in_use(
        authorization, lock_fingerprint=FINGERPRINT,
        runtime_relative=RUNTIME_RELATIVE,
    )
    receipt = lifecycle.consume_authorization(
        authorization, lock_fingerprint=FINGERPRINT,
        first_backend_invocation_utc="2026-08-20T04:01:00+00:00",
        last_backend_invocation_utc="2026-08-20T04:02:00+00:00",
        actual_open3d_trials=180, actual_pcl_trials=180,
        execution_result_manifest_sha256=SHA,
    )
    assert receipt["state"] == "CONSUMED" and receipt["reusable"] is False
    with pytest.raises(verifier.FormalAuthorizationVerificationError, match="consumed"):
        _verify(tmp_path, authorization, mode="resume")
    with pytest.raises(lifecycle.AuthorizationLifecycleError, match="write-once"):
        lifecycle.consume_authorization(
            authorization, lock_fingerprint=FINGERPRINT,
            first_backend_invocation_utc="2026-08-20T04:01:00+00:00",
            last_backend_invocation_utc="2026-08-20T04:02:00+00:00",
            actual_open3d_trials=180, actual_pcl_trials=180,
            execution_result_manifest_sha256=SHA,
        )


@pytest.mark.parametrize("tamper", ["sidecar", "noncanonical", "duplicate", "wrong_runtime", "wrong_workers"])
def test_independent_verifier_rejects_envelope_and_scope_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    _enable_isolated_verifier(monkeypatch)
    authorization = _write_authorization(tmp_path)
    (tmp_path / "lock").mkdir()
    runtime = tmp_path / RUNTIME_RELATIVE
    workers = 2
    if tamper == "sidecar":
        (authorization.parent / verifier.AUTHORIZATION_SHA_FILENAME).write_text("bad\n")
    elif tamper == "noncanonical":
        authorization.write_text(json.dumps(_payload(), indent=2) + "\n")
        digest = verifier.sha256_file(authorization)
        (authorization.parent / verifier.AUTHORIZATION_SHA_FILENAME).write_text(
            f"{digest}  {verifier.AUTHORIZATION_FILENAME}\n"
        )
    elif tamper == "duplicate":
        duplicate = tmp_path / "zero_perturbation_runtime/duplicate" / verifier.AUTHORIZATION_FILENAME
        duplicate.parent.mkdir(parents=True); duplicate.write_bytes(authorization.read_bytes())
    elif tamper == "wrong_runtime":
        runtime = tmp_path / "zero_perturbation_runtime/wrong"
    else:
        workers = 1
    with pytest.raises(verifier.FormalAuthorizationVerificationError):
        verifier.verify_formal_registration_authorization(
            tmp_path, lock_dir=tmp_path / "lock",
            authorization_path=authorization, runtime_root=runtime,
            requested_mode="fresh", workers=workers,
        )


def test_producer_requires_explicit_confirmation_before_any_write(tmp_path: Path) -> None:
    runtime = tmp_path / RUNTIME_RELATIVE
    with pytest.raises(producer.FormalAuthorizationIssuanceError, match="explicit"):
        producer.issue_formal_registration_authorization(
            tmp_path, lock_dir=tmp_path / "lock",
            expected_lock_fingerprint=FINGERPRINT,
            lock_release_commit=COMMIT, runtime_root=runtime, workers=2,
            explicit_user_confirmation=False,
        )
    assert not runtime.exists()


def _producer_fixture(root: Path) -> tuple[Path, Path, str]:
    lock_dir = root / "lock"
    lock_dir.mkdir(parents=True)
    plan_rows = []
    for index in range(180):
        for backend in verifier.BACKENDS:
            plan_rows.append({
                "trial_id": f"TRIAL-{index:03d}-{backend}",
                "snapshot_id": f"SNAP-{index:03d}",
                "backend": backend,
                "track_id": "ZERO_PERTURBATION_TRACK",
                "scene_id": "FMB1_W02", "attempt": 2,
                "T0": verifier.IDENTITY,
                "translation_perturbation_m": 0.0,
                "rotation_perturbation_deg": 0.0,
            })
    plan = root / "plan.json"
    plan.write_text(json.dumps({"rows": plan_rows}, sort_keys=True) + "\n")
    final = root / "final_verify.json"
    final.write_text('{"pass":true,"status":"PASS"}\n')
    environment = root / "environment.json"
    environment.write_text('{"status":"PASS"}\n')
    pcl = root / "pcl"; pcl.write_bytes(b"fixture-pcl")
    analysis = root / "analysis.json"; analysis.write_text("{}\n")
    backend = root / "backend.json"; backend.write_text("{}\n")
    bindings = {}
    for name, path in {
        "trial_plan_json": plan,
        "final_dataset_independent_verification": final,
        "environment_manifest": environment,
        "pcl_executable": pcl,
        "analysis_contract": analysis,
        "backend_parameter_contract": backend,
    }.items():
        bindings[name] = {
            "repository_relative_path": path.relative_to(root).as_posix(),
            "sha256": verifier.sha256_file(path), "bytes": path.stat().st_size,
        }
    lock = {
        "schema": verifier.LOCK_SCHEMA,
        "execution_lock_revision": 2,
        "status": "ISSUED_AWAITING_SEPARATE_AUTHORIZATION",
        "execution_code_commit": COMMIT,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
        "bindings": bindings,
    }
    lock_path = lock_dir / verifier.LOCK_FILENAME
    lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n")
    inventory = lock_dir / "lock_inventory.csv"; inventory.write_text("fixture\n")
    material = {
        "lock_file_sha256": verifier.sha256_file(lock_path),
        "lock_inventory_file_sha256": verifier.sha256_file(inventory),
        "execution_code_commit": COMMIT,
    }
    fingerprint = hashlib.sha256(verifier.canonical_json_bytes(material)).hexdigest()
    (lock_dir / "lock_fingerprint.json").write_text(
        json.dumps({**material, "lock_fingerprint": fingerprint}) + "\n"
    )
    (lock_dir / "independent_verification.json").write_text(json.dumps({
        "pass": True, "NEW_LOCK_VERIFIER_PASS": True,
        "lock_fingerprint": fingerprint,
    }) + "\n")
    return lock_dir, root / RUNTIME_RELATIVE, fingerprint


def test_producer_issues_canonical_unique_write_once_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_dir, runtime, fingerprint = _producer_fixture(tmp_path)
    monkeypatch.setattr(producer, "_verify_git", lambda *args: None)
    monkeypatch.setattr(
        producer, "verify_environment_manifest", lambda *args, **kwargs: {"pass": True}
    )
    monkeypatch.setattr(
        zero_perturbation_v1_1_exec_r2_verify,
        "verify_exec_r2_lock",
        lambda *args, **kwargs: {
            "NEW_LOCK_VERIFIER_PASS": True,
            "lock_fingerprint": fingerprint,
        },
    )
    report = producer.issue_formal_registration_authorization(
        tmp_path, lock_dir=lock_dir,
        expected_lock_fingerprint=fingerprint,
        lock_release_commit=COMMIT, runtime_root=runtime, workers=2,
        explicit_user_confirmation=True,
        issued_at_utc="2026-08-20T04:00:00+00:00", nonce="6" * 64,
    )
    authorization = Path(str(report["authorization_path"]))
    payload = json.loads(authorization.read_text())
    assert authorization.read_bytes() == producer.canonical_json_bytes(payload)
    assert payload["authorization_id"].startswith("FMB1-AUTH-")
    assert payload["nonce"] == "6" * 64
    assert payload["FORMAL_REGISTRATION_AUTHORIZED"] is True
    with pytest.raises(producer.FormalAuthorizationIssuanceError):
        producer.issue_formal_registration_authorization(
            tmp_path, lock_dir=lock_dir,
            expected_lock_fingerprint=fingerprint,
            lock_release_commit=COMMIT, runtime_root=runtime, workers=2,
            explicit_user_confirmation=True,
        )
