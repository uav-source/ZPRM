import hashlib
import json
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.locked_analysis.formal_firewall_v1 import (
    FormalReadRequest, FormalResultFirewallError, reject_ambiguous_mode,
    validate_formal_read_request,
)


def test_default_and_unconfirmed_formal_paths_fail_without_touching_paths():
    with pytest.raises(FormalResultFirewallError):
        reject_ambiguous_mode(fixture_only=False, formal_arguments_present=False, confirmed=False)
    with pytest.raises(FormalResultFirewallError, match="without"):
        reject_ambiguous_mode(fixture_only=False, formal_arguments_present=True, confirmed=False)


def test_fixture_plus_formal_path_or_confirm_fails():
    with pytest.raises(FormalResultFirewallError, match="mutually"):
        reject_ambiguous_mode(fixture_only=True, formal_arguments_present=True, confirmed=False)
    with pytest.raises(FormalResultFirewallError):
        reject_ambiguous_mode(fixture_only=True, formal_arguments_present=False, confirmed=True)


def test_confirm_without_analysis_lock_fails(tmp_path):
    repo = tmp_path / "repo"; result = repo / "results"; postrun = repo / "postrun"; lock = repo / "lock"
    for path in (result, postrun, lock): path.mkdir(parents=True)
    request = FormalReadRequest(repo, result, postrun, lock, "a" * 40,
                                repo / "output", True)
    with pytest.raises(FormalResultFirewallError, match="lock"):
        validate_formal_read_request(request)


def test_output_inside_result_root_fails_before_lock_read(tmp_path):
    repo = tmp_path / "repo"; result = repo / "results"; postrun = repo / "postrun"; lock = repo / "lock"
    for path in (result, postrun, lock): path.mkdir(parents=True)
    request = FormalReadRequest(repo, result, postrun, lock, "a" * 40,
                                result / "analysis", True)
    with pytest.raises(FormalResultFirewallError, match="outside"):
        validate_formal_read_request(request)


def formal_fixture(tmp_path, *, code="a" * 40, postrun_pass=True,
                   raw_commit="059e39533991d929a97ab208ad738643af82d09a",
                   fingerprint="fd601e8daf62c488a3a079beea05f65c9bd283399ae4d7f3acf16e63fc473a6d"):
    repo = tmp_path / "repo"; result = repo / "results"; postrun = repo / "postrun"; lock_dir = repo / "lock"
    for path in (result, postrun, lock_dir): path.mkdir(parents=True)
    lock = {
        "status": "ISSUED_AWAITING_SEPARATE_REAL_ANALYSIS_AUTHORIZATION",
        "READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS": True,
        "LOCKED_ANALYSIS_CODE_COMMIT": code,
        "POSTRUN_VERIFICATION_COMMIT": "18bb94e62761f5193da8cdc5509c470cb4983244",
        "RAW_EXECUTION_COMMIT": raw_commit,
        "R3_LOCK_FINGERPRINT": fingerprint,
        "formal_results_root": str(result.resolve()),
    }
    lock_path = lock_dir / "locked_analysis_lock_v1.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    (lock_dir / "formal_analysis_authorization.json").write_text(json.dumps({
        "REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED": True,
        "analysis_lock_sha256": lock_sha,
        "analysis_code_commit": code,
        "consumed": False,
    }), encoding="utf-8")
    (postrun / "postrun_independent_verification.json").write_text(json.dumps({
        "FMB1_POSTRUN_INDEPENDENT_VERIFICATION_PASS": postrun_pass,
        "pass": postrun_pass, "RAW_EXECUTION_BYTES_UNCHANGED": True,
        "raw_execution_commit": "059e39533991d929a97ab208ad738643af82d09a",
        "r3_lock_fingerprint": "fd601e8daf62c488a3a079beea05f65c9bd283399ae4d7f3acf16e63fc473a6d",
        "VERIFIED_TRIAL_COUNT": 360,
    }), encoding="utf-8")
    return repo, result, postrun, lock_dir


def request_for(parts, *, code="a" * 40, output=None):
    repo, result, postrun, lock = parts
    return FormalReadRequest(repo, result, postrun, lock, code,
                            output or repo / "output", True)


def test_wrong_analysis_code_commit_fails(tmp_path):
    parts = formal_fixture(tmp_path)
    with pytest.raises(FormalResultFirewallError, match="code commit"):
        validate_formal_read_request(request_for(parts, code="b" * 40))


def test_postrun_not_pass_fails(tmp_path):
    parts = formal_fixture(tmp_path, postrun_pass=False)
    with pytest.raises(FormalResultFirewallError, match="Post-run"):
        validate_formal_read_request(request_for(parts))


@pytest.mark.parametrize("field,value,match", [
    ("raw", "0" * 40, "raw execution commit"),
    ("fingerprint", "0" * 64, "fingerprint"),
])
def test_wrong_execution_binding_fails(tmp_path, field, value, match):
    parts = formal_fixture(
        tmp_path, raw_commit=value if field == "raw" else "059e39533991d929a97ab208ad738643af82d09a",
        fingerprint=value if field == "fingerprint" else "fd601e8daf62c488a3a079beea05f65c9bd283399ae4d7f3acf16e63fc473a6d",
    )
    with pytest.raises(FormalResultFirewallError, match=match):
        validate_formal_read_request(request_for(parts))


def test_existing_output_artifact_fails_before_scientific_read(tmp_path):
    parts = formal_fixture(tmp_path); output = parts[0] / "output"; output.mkdir()
    (output / "analysis_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FormalResultFirewallError, match="already contains"):
        validate_formal_read_request(request_for(parts, output=output))
