from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.analysis_authorization.contract_v1 import (
    CANDIDATE_NAME,
    FINAL_NAME,
    AuthorizationControlError,
)
from experiments.mid360_formal_batch1.analysis_authorization.formal_analysis_authorization_lifecycle_v1 import (
    publish_verified_authorization,
)
from experiments.mid360_formal_batch1.analysis_authorization.formal_analysis_authorization_producer_v1 import (
    prepare_authorization_candidate,
)
from experiments.mid360_formal_batch1.analysis_authorization.formal_analysis_authorization_verify_v1 import (
    IndependentAuthorizationVerificationError,
    verify_authorization_candidate,
)
from experiments.mid360_formal_batch1.locked_analysis.formal_firewall_v1 import (
    FormalResultFirewallError,
    validate_formal_read_request,
)
from formal_analysis_authorization_fixture_v1 import (
    CONTROL_COMMIT,
    NONCE,
    RELEASE_COMMIT,
    fixture_request,
    prepared_fixture,
    verified_fixture,
)


def _mutate(path: Path, key: str, value) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if value is _MISSING:
        payload.pop(key, None)
    else:
        payload[key] = value
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar = path.with_name("formal_analysis_authorization.candidate.sha256")
    import hashlib
    sidecar.write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )


_MISSING = object()


@pytest.mark.parametrize(("key", "value"), [
    ("analysis_lock_fingerprint", "0" * 64),
    ("analysis_lock_sha256", "0" * 64),
    ("analysis_lock_file_sha256", "0" * 64),
    ("analysis_lock_release_commit", "0" * 40),
    ("locked_analysis_code_commit", "0" * 40),
    ("analysis_code_commit", "0" * 40),
    ("raw_execution_commit", "0" * 40),
    ("postrun_verification_commit", "0" * 40),
    ("c2_clarification_commit", "0" * 40),
    ("r3_lock_fingerprint", "0" * 64),
    ("formal_results_root", "/tmp/wrong-formal-root"),
    ("analysis_output_root", "/tmp/wrong-output-root"),
    ("nonce", _MISSING),
    ("authorization_id", "FMB1_DUPLICATE_AUTHORIZATION_ID"),
    ("allow_registration", True),
    ("allow_capture_radius", True),
    ("allow_change_statistics", True),
    ("reusable", True),
])
def test_candidate_binding_and_permission_tamper_fails(
    tmp_path: Path, key: str, value,
) -> None:
    repository, lock_dir, output_root = prepared_fixture(tmp_path)
    _mutate(lock_dir / CANDIDATE_NAME, key, value)
    with pytest.raises(IndependentAuthorizationVerificationError):
        verify_authorization_candidate(
            repository, lock_dir,
            expected_analysis_control_commit=CONTROL_COMMIT,
            expected_lock_release_commit=RELEASE_COMMIT,
            expected_output_root=output_root,
            fixture_only=True,
        )


def test_output_root_with_artifact_fails(tmp_path: Path) -> None:
    repository, lock_dir, output_root = prepared_fixture(tmp_path)
    output_root.mkdir()
    (output_root / "not-real-fixture.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(IndependentAuthorizationVerificationError, match="not empty"):
        verify_authorization_candidate(
            repository, lock_dir,
            expected_analysis_control_commit=CONTROL_COMMIT,
            expected_lock_release_commit=RELEASE_COMMIT,
            expected_output_root=output_root,
            fixture_only=True,
        )


def test_candidate_changed_after_verification_cannot_publish(tmp_path: Path) -> None:
    _, lock_dir, _ = verified_fixture(tmp_path)
    _mutate(lock_dir / CANDIDATE_NAME, "authorization_basis", "TAMPERED")
    with pytest.raises(AuthorizationControlError, match="not exact PASS"):
        publish_verified_authorization(lock_dir)


def test_publish_without_verifier_fails(tmp_path: Path) -> None:
    _, lock_dir, _ = prepared_fixture(tmp_path)
    with pytest.raises(AuthorizationControlError):
        publish_verified_authorization(lock_dir)


def test_verifier_report_from_another_candidate_fails(tmp_path: Path) -> None:
    _, lock_dir, _ = verified_fixture(tmp_path)
    report_path = lock_dir / "formal_analysis_authorization_verification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["candidate_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with pytest.raises(AuthorizationControlError, match="not exact PASS"):
        publish_verified_authorization(lock_dir)


def test_second_candidate_and_second_live_authorization_fail(tmp_path: Path) -> None:
    repository, lock_dir, output_root = prepared_fixture(tmp_path)
    with pytest.raises(AuthorizationControlError, match="collision"):
        prepare_authorization_candidate(
            repository, lock_dir,
            analysis_control_commit=CONTROL_COMMIT,
            analysis_lock_release_commit=RELEASE_COMMIT,
            nonce="b" * 32,
            confirm_explicit_user_analysis_authorization=True,
            output_root=output_root, fixture_only=True,
        )


def test_candidate_passed_directly_to_firewall_fails(tmp_path: Path) -> None:
    repository, lock_dir, output_root = prepared_fixture(tmp_path)
    with pytest.raises(FormalResultFirewallError):
        validate_formal_read_request(fixture_request(repository, lock_dir, output_root))


def test_missing_explicit_confirmation_fails_before_candidate(tmp_path: Path) -> None:
    repository, lock_dir, output_root = prepared_fixture(tmp_path)
    with pytest.raises(AuthorizationControlError, match="explicit-user"):
        prepare_authorization_candidate(
            repository, lock_dir,
            analysis_control_commit=CONTROL_COMMIT,
            analysis_lock_release_commit=RELEASE_COMMIT,
            nonce=NONCE,
            confirm_explicit_user_analysis_authorization=False,
            output_root=output_root, fixture_only=True,
        )


def test_malformed_candidate_json_fails(tmp_path: Path) -> None:
    repository, lock_dir, output_root = prepared_fixture(tmp_path)
    (lock_dir / CANDIDATE_NAME).write_text("{not-json", encoding="utf-8")
    with pytest.raises(AuthorizationControlError, match="invalid JSON"):
        verify_authorization_candidate(
            repository, lock_dir,
            expected_analysis_control_commit=CONTROL_COMMIT,
            expected_lock_release_commit=RELEASE_COMMIT,
            expected_output_root=output_root, fixture_only=True,
        )


def test_candidate_symlink_is_rejected(tmp_path: Path) -> None:
    repository, lock_dir, output_root = prepared_fixture(tmp_path)
    candidate = lock_dir / CANDIDATE_NAME
    other = lock_dir / "candidate-bytes.json"
    other.write_bytes(candidate.read_bytes())
    candidate.unlink()
    candidate.symlink_to(other.name)
    with pytest.raises(IndependentAuthorizationVerificationError, match="symlink"):
        verify_authorization_candidate(
            repository, lock_dir,
            expected_analysis_control_commit=CONTROL_COMMIT,
            expected_lock_release_commit=RELEASE_COMMIT,
            expected_output_root=output_root, fixture_only=True,
        )


def test_final_publish_is_write_once(tmp_path: Path) -> None:
    _, lock_dir, _ = verified_fixture(tmp_path)
    publish_verified_authorization(lock_dir)
    assert (lock_dir / FINAL_NAME).is_file()
    with pytest.raises(AuthorizationControlError, match="already exists"):
        publish_verified_authorization(lock_dir)
