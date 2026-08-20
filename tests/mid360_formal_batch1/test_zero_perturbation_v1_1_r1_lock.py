from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_lock import (
    AUTHORITATIVE_RUNTIME_ROOT,
    DEFAULT_AUTHORIZATION_PATH,
    DEFAULT_BINDING_PATHS,
    LOCK_FILENAME,
    R1LockError,
    build_lock_payload,
    finalize_lock_directory_checksums,
)
from tests.mid360_formal_batch1.zero_perturbation_v1_1_r1_fixture import (
    COMMIT,
    build_valid_r1_lock,
    sha,
)


def test_lock_is_canonical_issued_but_never_authorizes(tmp_path: Path) -> None:
    root, lock_dir, _ = build_valid_r1_lock(tmp_path)
    assert LOCK_FILENAME == "formal_batch1_zero_perturbation_lock_v1_1.json"
    lock = json.loads((lock_dir / LOCK_FILENAME).read_text())
    assert lock["FORMAL_LOCK_ISSUED"] is True
    assert lock["READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION"] is True
    assert lock["FORMAL_ICP_UNLOCKED"] is False
    assert lock["FORMAL_REGISTRATION_AUTHORIZED"] is False
    assert lock["actual_formal_trials"] == 0
    assert lock["authoritative_runtime_root"] == AUTHORITATIVE_RUNTIME_ROOT
    assert {
        "original_zero_perturbation_proposal_json",
        "original_zero_perturbation_proposal_md",
        "proposal_superseded_sidecar", "proposal_correction_record",
        "proposal_difference_report", "w04_superseded_history",
        "analysis_missingness_clarification",
        "analysis_missingness_clarification_md",
        "analysis_missingness_clarification_transition",
        "analysis_preclarification_history_inventory",
        "protocol_c1_missingness_independent_verification",
        "activation_review",
        "execution_runner_cli", "execution_metrics", "execution_types",
        "execution_experiments_package_init",
        "execution_mid360_formal_batch1_package_init",
        "execution_phase_a_harness_package_init",
    }.issubset(lock["bindings"])


def test_core_and_release_checksums_are_non_self_referential(tmp_path: Path) -> None:
    _, lock_dir, _ = build_valid_r1_lock(tmp_path)
    core = (lock_dir / "LOCK_CORE_SHA256SUMS").read_text().splitlines()
    assert core and all("LOCK_CORE_SHA256SUMS" not in line for line in core)
    release = (lock_dir / "SHA256SUMS").read_text().splitlines()
    assert release and all(not line.endswith("  SHA256SUMS") for line in release)
    assert any(line.endswith("  LOCK_CORE_SHA256SUMS") for line in release)
    before = (lock_dir / "SHA256SUMS").read_bytes()
    finalize_lock_directory_checksums(lock_dir)
    assert (lock_dir / "SHA256SUMS").read_bytes() == before


def test_lock_builder_rejects_unverifiable_execution_commit(tmp_path: Path) -> None:
    root, _, _ = build_valid_r1_lock(tmp_path)
    with pytest.raises(R1LockError, match="commit"):
        build_lock_payload(
            root, execution_code_commit=COMMIT,
            remeasure_environment_versions=False,
            verify_execution_commit=True,
        )


def test_nested_reauthentication_tamper_blocks_lock_rebuild(tmp_path: Path) -> None:
    root, _, _ = build_valid_r1_lock(tmp_path)
    wrapper = root / DEFAULT_BINDING_PATHS["final_dataset_prelock_reauthentication"]
    payload = json.loads(wrapper.read_text())
    payload["source_report_sha256"] = "0" * 64
    wrapper.write_text(json.dumps(payload) + "\n")
    with pytest.raises(R1LockError, match="nested"):
        build_lock_payload(
            root, execution_code_commit=COMMIT,
            remeasure_environment_versions=False,
            verify_execution_commit=False,
        )


def test_r1_no_icp_attestation_is_fresh_and_scoped(tmp_path: Path) -> None:
    root, lock_dir, runtime = build_valid_r1_lock(tmp_path)
    payload = json.loads((lock_dir / "NO_ICP_ATTESTATION.json").read_text())
    assert payload["schema"] == "mid360_fmb1_zero_perturbation_r1_no_icp_attestation_v1"
    assert payload["future_execution_boundary_verified"] is True
    assert payload["execution_lifecycle_file_count"] == 0
    assert payload["real_trial_result_file_count"] == 0
    assert payload["formal_trial_count"] == 0
    assert payload["FORMAL_REGISTRATION_AUTHORIZED"] is False
    assert payload["authoritative_runtime_root"] == runtime.relative_to(root).as_posix()
    for relative, digest in payload["authority_file_sha256"].items():
        assert sha(root / relative) == digest


@pytest.mark.parametrize(
    "relative",
    ("run_contract.json", "inflight/trial-0001.started.json", "run_manifest.json"),
)
def test_lock_builder_rejects_any_execution_lifecycle_file(
    tmp_path: Path, relative: str,
) -> None:
    root, _, runtime = build_valid_r1_lock(tmp_path)
    marker = runtime / relative
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}\n")
    with pytest.raises(R1LockError, match="lifecycle"):
        build_lock_payload(
            root, execution_code_commit=COMMIT,
            remeasure_environment_versions=False,
            verify_execution_commit=False,
        )


def test_lock_builder_rejects_existing_separate_authorization(tmp_path: Path) -> None:
    root, _, _ = build_valid_r1_lock(tmp_path)
    authorization = root / DEFAULT_AUTHORIZATION_PATH
    authorization.write_text("{}\n")
    with pytest.raises(R1LockError, match="authorization"):
        build_lock_payload(
            root, execution_code_commit=COMMIT,
            remeasure_environment_versions=False,
            verify_execution_commit=False,
        )


@pytest.mark.parametrize(
    ("binding_name", "mutator", "message"),
    [
        ("original_zero_perturbation_proposal_json", "bytes", "historical proposal"),
        ("proposal_superseded_sidecar", "status", "supersession"),
        ("proposal_correction_record", "lineage", "correction"),
        ("analysis_missingness_clarification", "clarification", "clarification"),
        ("analysis_missingness_clarification_transition", "transition", "transition"),
        ("protocol_c1_missingness_independent_verification", "c1_report", "R1-C1"),
        ("activation_review", "review", "activation review"),
        ("analysis_preclarification_history_inventory", "archive", "archived"),
    ],
)
def test_lock_builder_semantically_rejects_provenance_or_c1_tamper(
    tmp_path: Path, binding_name: str, mutator: str, message: str,
) -> None:
    root, _, _ = build_valid_r1_lock(tmp_path)
    path = root / DEFAULT_BINDING_PATHS[binding_name]
    if mutator == "bytes":
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutator == "status":
        payload = json.loads(path.read_text())
        payload["status"] = "ACTIVE"
        path.write_text(json.dumps(payload) + "\n")
    elif mutator == "lineage":
        payload = json.loads(path.read_text())
        payload["lineage"]["w04_in_final_dataset"] = True
        path.write_text(json.dumps(payload) + "\n")
    elif mutator == "clarification":
        payload = json.loads(path.read_text())
        payload["clarification_at_formal_trial_count"] = 1
        path.write_text(json.dumps(payload) + "\n")
    elif mutator == "transition":
        payload = json.loads(path.read_text())
        payload["after"]["clarification_json_sha256"] = "0" * 64
        path.write_text(json.dumps(payload) + "\n")
    elif mutator == "c1_report":
        payload = json.loads(path.read_text())
        payload["common_association_status_fields_verified"] = False
        path.write_text(json.dumps(payload) + "\n")
    elif mutator == "review":
        payload = json.loads(path.read_text())
        payload["reviewed_at_utc"] = "2026-08-20T01:00:00+00:00"
        path.write_text(json.dumps(payload) + "\n")
    else:
        payload = json.loads(path.read_text())
        archived = path.parent / payload["files"][0]["name"]
        archived.write_bytes(archived.read_bytes() + b"\n")
    with pytest.raises(R1LockError, match=message):
        build_lock_payload(
            root, execution_code_commit=COMMIT,
            remeasure_environment_versions=False,
            verify_execution_commit=False,
        )


@pytest.mark.parametrize("name", [
    "collect_zero_perturbation_v1_1_r1_environment.py",
    "issue_zero_perturbation_v1_1_r1_lock.py",
    "run_zero_perturbation_v1_1_r1.py",
    "verify_zero_perturbation_v1_1_r1.py",
    "finalize_zero_perturbation_v1_1_r1_checksums.py",
])
def test_formal_cli_help_is_self_contained_without_pythonpath(name: str) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(repository / "tools/mid360_formal_batch1" / name), "--help"],
        cwd="/tmp", env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_mid360_package_import_has_no_protocol_or_backend_side_effects() -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository / "src"), str(repository))
    )
    code = (
        "import sys; import experiments.mid360_formal_batch1; "
        "assert 'experiments.mid360_formal_batch1.protocol' not in sys.modules; "
        "assert 'open3d' not in sys.modules; "
        "assert 'phase_a_harness.open3d_backend' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd="/tmp", env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert completed.returncode == 0, completed.stderr
