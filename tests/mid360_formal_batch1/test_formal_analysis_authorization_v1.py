from __future__ import annotations

import ast
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.analysis_authorization.analysis_lock_v2_verify import (
    run_fixture_qualification,
)
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
from experiments.mid360_formal_batch1.locked_analysis.formal_firewall_v1 import (
    FormalReadRequest,
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


REPOSITORY = Path(__file__).resolve().parents[2]


def test_full_fixture_qualification_is_zero_read_pass() -> None:
    report = run_fixture_qualification(REPOSITORY)
    assert report["ANALYSIS_AUTHORIZATION_FIXTURE_QUALIFICATION_PASS"] is True
    assert report["candidate_cannot_directly_authorize"] is True
    assert report["candidate_plus_verifier_cannot_authorize_before_publish"] is True
    assert report["published_authorization_firewall_gate_pass"] is True
    assert report["success_lifecycle_pass"] is True
    assert report["interruption_lifecycle_pass"] is True
    assert report["REAL_FORMAL_RESULT_FILES_READ"] == 0
    assert report["REAL_SCIENTIFIC_VALUES_READ"] == 0
    assert report["REAL_P_VALUE_COUNT"] == 0
    assert report["REGISTRATION_BACKEND_CALLS"] == 0


def test_prepare_requires_explicit_user_confirmation(tmp_path: Path) -> None:
    repository, lock_dir, output_root = prepared_fixture(tmp_path)
    # Use a second empty fixture lock because the first already has a candidate.
    with pytest.raises(AuthorizationControlError, match="explicit-user"):
        prepare_authorization_candidate(
            repository, lock_dir,
            analysis_control_commit=CONTROL_COMMIT,
            analysis_lock_release_commit=RELEASE_COMMIT,
            nonce=NONCE,
            confirm_explicit_user_analysis_authorization=False,
            output_root=output_root,
            fixture_only=True,
        )


def test_candidate_filename_cannot_satisfy_frozen_firewall(tmp_path: Path) -> None:
    repository, lock_dir, output_root = prepared_fixture(tmp_path)
    assert (lock_dir / CANDIDATE_NAME).is_file()
    assert not (lock_dir / FINAL_NAME).exists()
    with pytest.raises(FormalResultFirewallError, match="authorization is absent"):
        validate_formal_read_request(fixture_request(repository, lock_dir, output_root))


def test_verifier_report_still_cannot_authorize_before_publish(tmp_path: Path) -> None:
    repository, lock_dir, output_root = verified_fixture(tmp_path)
    with pytest.raises(FormalResultFirewallError, match="authorization is absent"):
        validate_formal_read_request(fixture_request(repository, lock_dir, output_root))


def test_publish_after_independent_pass_satisfies_metadata_gate(tmp_path: Path) -> None:
    repository, lock_dir, output_root = verified_fixture(tmp_path)
    published = publish_verified_authorization(
        lock_dir, published_at_utc="2026-08-20T00:01:00Z"
    )
    assert published["lifecycle_state"] == "VERIFIED_PUBLISHED"
    boundary = validate_formal_read_request(
        fixture_request(repository, lock_dir, output_root)
    )
    assert boundary["status"] == "PASS_AUTHORIZED_FORMAL_READ_BOUNDARY"


def test_published_file_does_not_replace_frozen_cli_confirmation(tmp_path: Path) -> None:
    repository, lock_dir, output_root = verified_fixture(tmp_path)
    publish_verified_authorization(lock_dir)
    request = fixture_request(repository, lock_dir, output_root)
    denied = FormalReadRequest(
        repository=request.repository,
        formal_results_root=request.formal_results_root,
        postrun_verification_root=request.postrun_verification_root,
        analysis_lock_dir=request.analysis_lock_dir,
        analysis_code_commit=request.analysis_code_commit,
        output_dir=request.output_dir,
        confirm_read_frozen_formal_results=False,
    )
    with pytest.raises(FormalResultFirewallError, match="confirmation is absent"):
        validate_formal_read_request(denied)


def test_independent_verifier_has_no_producer_import() -> None:
    path = REPOSITORY / (
        "experiments/mid360_formal_batch1/analysis_authorization/"
        "formal_analysis_authorization_verify_v1.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.endswith("formal_analysis_authorization_producer_v1") for name in imported)
