"""Issue Analysis Lock v2 without granting real-analysis authority."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.mid360_formal_batch1.locked_analysis.lock_v1 import ANALYSIS_CODE_PATHS

from .contract_v1 import (
    ANALYSIS_OUTPUT_RELATIVE,
    AUTHORIZATION_SCHEMA_RELATIVE,
    CONTROL_CODE_PATHS,
    LOCKED_ANALYSIS_CODE_COMMIT,
    LOCK_V1_FINGERPRINT,
    LOCK_V1_RELEASE_COMMIT,
    LOCK_V1_RELATIVE,
    LOCK_V1_SHA256,
    OUTPUT_SCHEMA_SHA256,
    AuthorizationControlError,
    canonical_fingerprint,
    load_object,
    sha256_file,
)


LOCK_V2_CORE_FILES = (
    "analysis_code_inventory.csv",
    "analysis_control_inventory.csv",
    "analysis_authorization_control_qualification.json",
    "analysis_authorization_contract_audit.json",
    "analysis_authorization_fixture_qualification.json",
    "analysis_no_real_results_attestation.json",
    "analysis_lock_v1_supersession_binding.json",
    "locked_analysis_lock_v1.json",
    "locked_analysis_lock_v1.sha256",
    "locked_analysis_lock_v2.json",
    "locked_analysis_lock_v2.sha256",
)


def _git(repository: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["git", *args], cwd=repository, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if process.returncode:
        raise AuthorizationControlError(
            process.stderr.decode("utf-8", "replace").strip()
        )
    return process.stdout


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _inventory(repository: Path, commit: str, paths: tuple[str, ...]) -> list[dict[str, Any]]:
    if _git(repository, "cat-file", "-t", commit).strip() != b"commit":
        raise AuthorizationControlError(f"commit does not exist: {commit}")
    rows: list[dict[str, Any]] = []
    for relative in paths:
        current = (repository / relative).read_bytes()
        if _git(repository, "show", f"{commit}:{relative}") != current:
            raise AuthorizationControlError(
                f"worktree differs from frozen commit: {relative}"
            )
        rows.append({
            "path": relative,
            "sha256": hashlib.sha256(current).hexdigest(),
            "bytes": len(current),
            "git_blob_exact": True,
        })
    return rows


def _write_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "sha256", "bytes", "git_blob_exact")
        )
        writer.writeheader()
        writer.writerows(rows)


def _require_zero_evidence(
    audit: dict[str, Any], qualification: dict[str, Any], attestation: dict[str, Any],
    supersession: dict[str, Any],
) -> None:
    if not (
        audit.get("EXTERNAL_AUTH_INFRA_CAN_SATISFY_FROZEN_FIREWALL") is True
        and audit.get("ANALYSIS_CODE_REFREEZE_REQUIRED") is False
        and audit.get("real_scientific_values_read") == 0
    ):
        raise AuthorizationControlError("firewall compatibility audit is not exact PASS")
    qualification_true = (
        "ANALYSIS_AUTHORIZATION_FIXTURE_QUALIFICATION_PASS",
        "candidate_cannot_directly_authorize",
        "candidate_plus_verifier_cannot_authorize_before_publish",
        "published_authorization_firewall_gate_pass",
        "success_lifecycle_pass",
        "interruption_lifecycle_pass",
    )
    if any(qualification.get(key) is not True for key in qualification_true):
        raise AuthorizationControlError("authorization fixture qualification is not PASS")
    zero_fields = (
        "REAL_FORMAL_RESULT_FILES_READ", "REAL_SCIENTIFIC_VALUES_READ",
        "REAL_WEAK_RICH_COMPARISON_COUNT", "REAL_P_VALUE_COUNT",
        "REGISTRATION_BACKEND_CALLS",
    )
    if any(qualification.get(key) != 0 for key in zero_fields):
        raise AuthorizationControlError("fixture qualification is not zero-read")
    if not (
        attestation.get("pass") is True
        and attestation.get("REAL_FORMAL_RESULT_FILES_READ") == 0
        and attestation.get("REAL_SCIENTIFIC_VALUES_READ") == 0
        and attestation.get("REAL_AGGREGATIONS") == 0
        and attestation.get("REAL_P_VALUES") == 0
        and attestation.get("REGISTRATION_BACKEND_CALLS") == 0
    ):
        raise AuthorizationControlError("no-real-results attestation is not exact PASS")
    if not (
        supersession.get("status") == "SUPERSEDED_BEFORE_REAL_SCIENTIFIC_ANALYSIS"
        and supersession.get("reason")
        == "MISSING_LOCK_BOUND_REAL_ANALYSIS_AUTHORIZATION_MECHANISM"
        and supersession.get("analysis_lock_v1_sha256") == LOCK_V1_SHA256
        and supersession.get("analysis_lock_v1_fingerprint") == LOCK_V1_FINGERPRINT
        and supersession.get("v1_real_scientific_analysis_count") == 0
        and supersession.get("v1_real_values_read") == 0
        and supersession.get("scientific_analysis_code_defect") is False
        and supersession.get("authorization_infrastructure_defect") is True
    ):
        raise AuthorizationControlError("Analysis Lock v1 supersession mismatch")


def issue_analysis_lock_v2(
    repository: Path,
    lock_dir: Path,
    *,
    analysis_control_commit: str,
    contract_audit_path: Path,
    control_qualification_path: Path,
    fixture_qualification_path: Path,
    no_real_results_attestation_path: Path,
    v1_supersession_path: Path,
    blocked_attempt_audit_path: Path,
    issued_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a non-authorizing v2 lock after the control-code commit."""

    repository = repository.resolve(strict=True)
    lock_dir = lock_dir.resolve(strict=False)
    if lock_dir.exists() and any(lock_dir.iterdir()):
        raise AuthorizationControlError("Analysis Lock v2 directory must be absent or empty")
    v1_dir = (repository / LOCK_V1_RELATIVE).resolve(strict=True)
    v1_path = v1_dir / "locked_analysis_lock_v1.json"
    if sha256_file(v1_path) != LOCK_V1_SHA256:
        raise AuthorizationControlError("Analysis Lock v1 bytes changed")
    if _git(repository, "show", f"{LOCK_V1_RELEASE_COMMIT}:{LOCK_V1_RELATIVE}/locked_analysis_lock_v1.json") != v1_path.read_bytes():
        raise AuthorizationControlError("Analysis Lock v1 release binding mismatch")
    v1_lock = load_object(v1_path)
    if v1_lock.get("analysis_lock_fingerprint") != LOCK_V1_FINGERPRINT:
        raise AuthorizationControlError("Analysis Lock v1 fingerprint changed")

    audit = load_object(contract_audit_path)
    control_qualification = load_object(control_qualification_path)
    qualification = load_object(fixture_qualification_path)
    attestation = load_object(no_real_results_attestation_path)
    supersession = load_object(v1_supersession_path)
    blocked = load_object(blocked_attempt_audit_path)
    _require_zero_evidence(audit, qualification, attestation, supersession)
    if not (
        control_qualification.get("ANALYSIS_AUTH_CONTROL_COMMIT")
        == analysis_control_commit
        and control_qualification.get("LOCKED_ANALYSIS_CODE_COMMIT_UNCHANGED") is True
        and control_qualification.get("LOCKED_ANALYSIS_CODE_COMMIT")
        == LOCKED_ANALYSIS_CODE_COMMIT
        and control_qualification.get("SCIENTIFIC_ANALYSIS_IMPLEMENTATION_CHANGED") is False
        and control_qualification.get("C1_CHANGED") is False
        and control_qualification.get("C2_CHANGED") is False
        and control_qualification.get("RAW_EXECUTION_CHANGED") is False
        and control_qualification.get("REAL_SCIENTIFIC_VALUES_READ") == 0
    ):
        raise AuthorizationControlError("analysis-control qualification mismatch")
    if not (
        blocked.get("blocker")
        == "MISSING_LOCK_BOUND_REAL_ANALYSIS_AUTHORIZATION_MECHANISM"
        and blocked.get("real_scientific_values_read") == 0
        and blocked.get("real_analysis_executed") is False
    ):
        raise AuthorizationControlError("blocked attempt audit mismatch")

    analysis_rows = _inventory(
        repository, LOCKED_ANALYSIS_CODE_COMMIT, tuple(ANALYSIS_CODE_PATHS)
    )
    control_rows = _inventory(
        repository, analysis_control_commit, tuple(CONTROL_CODE_PATHS)
    )
    lock_dir.mkdir(parents=True, exist_ok=False)
    _write_inventory(lock_dir / "analysis_code_inventory.csv", analysis_rows)
    _write_inventory(lock_dir / "analysis_control_inventory.csv", control_rows)
    copies = {
        "analysis_authorization_control_qualification.json": control_qualification_path,
        "analysis_authorization_contract_audit.json": contract_audit_path,
        "analysis_authorization_fixture_qualification.json": fixture_qualification_path,
        "analysis_no_real_results_attestation.json": no_real_results_attestation_path,
        "analysis_lock_v1_supersession_binding.json": v1_supersession_path,
    }
    for name, source in copies.items():
        (lock_dir / name).write_bytes(source.read_bytes())

    schema_path = repository / AUTHORIZATION_SCHEMA_RELATIVE
    output_schema_path = (
        repository / "experiments/mid360_formal_batch1/locked_analysis/output_schema_v1.json"
    )
    blocked_relative = str(blocked_attempt_audit_path.resolve(strict=True).relative_to(repository))
    lock: dict[str, Any] = dict(v1_lock)
    lock.update({
        "schema": "fmb1_zero_perturbation_locked_analysis_lock_v2",
        "lock_id": "FMB1_ZERO_PERTURBATION_LOCKED_ANALYSIS_V2",
        "analysis_lock_revision": 2,
        "supersedes_lock_id": "FMB1_ZERO_PERTURBATION_LOCKED_ANALYSIS_V1",
        "supersedes_lock_v1_release_commit": LOCK_V1_RELEASE_COMMIT,
        "supersedes_lock_v1_sha256": LOCK_V1_SHA256,
        "supersedes_lock_v1_fingerprint": LOCK_V1_FINGERPRINT,
        "supersession_reason": "MISSING_LOCK_BOUND_REAL_ANALYSIS_AUTHORIZATION_MECHANISM",
        "status": "ISSUED_AWAITING_SEPARATE_REAL_ANALYSIS_AUTHORIZATION",
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "LOCKED_ANALYSIS_CODE_COMMIT": LOCKED_ANALYSIS_CODE_COMMIT,
        "analysis_control_commit": analysis_control_commit,
        "analysis_output_schema_sha256": OUTPUT_SCHEMA_SHA256,
        "analysis_authorization_schema_path": AUTHORIZATION_SCHEMA_RELATIVE,
        "analysis_authorization_schema_sha256": sha256_file(schema_path),
        "analysis_authorization_contract_path": (
            "experiments/mid360_formal_batch1/analysis_authorization/"
            "formal_analysis_authorization_contract_v1.md"
        ),
        "analysis_authorization_contract_sha256": sha256_file(
            repository / "experiments/mid360_formal_batch1/analysis_authorization/"
            "formal_analysis_authorization_contract_v1.md"
        ),
        "analysis_code_inventory_sha256": sha256_file(
            lock_dir / "analysis_code_inventory.csv"
        ),
        "analysis_control_inventory_sha256": sha256_file(
            lock_dir / "analysis_control_inventory.csv"
        ),
        "analysis_authorization_contract_audit_sha256": sha256_file(
            lock_dir / "analysis_authorization_contract_audit.json"
        ),
        "analysis_authorization_control_qualification_sha256": sha256_file(
            lock_dir / "analysis_authorization_control_qualification.json"
        ),
        "analysis_authorization_fixture_qualification_sha256": sha256_file(
            lock_dir / "analysis_authorization_fixture_qualification.json"
        ),
        "analysis_no_real_results_attestation_sha256": sha256_file(
            lock_dir / "analysis_no_real_results_attestation.json"
        ),
        "analysis_lock_v1_supersession_binding_sha256": sha256_file(
            lock_dir / "analysis_lock_v1_supersession_binding.json"
        ),
        "blocked_attempt_001_binding": {
            "path": blocked_relative,
            "sha256": sha256_file(blocked_attempt_audit_path),
        },
        "analysis_code_paths": list(ANALYSIS_CODE_PATHS),
        "analysis_control_paths": list(CONTROL_CODE_PATHS),
        "formal_results_root": str(
            (repository / "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1").resolve(strict=True)
        ),
        "postrun_verification_root": str(
            (repository / "results/mid360_formal_batch1/zero_perturbation_v1_1_postrun_verification_v1").resolve(strict=True)
        ),
        "analysis_output_root": str(
            (repository / ANALYSIS_OUTPUT_RELATIVE).resolve(strict=False)
        ),
        "future_analysis_output_root": str(
            (repository / ANALYSIS_OUTPUT_RELATIVE).resolve(strict=False)
        ),
        "READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS": True,
        "AUTHORIZATION_INFRASTRUCTURE_READY": True,
        "REAL_ANALYSIS_AUTHORIZATION_PRODUCER_READY": True,
        "INDEPENDENT_REAL_ANALYSIS_AUTHORIZATION_VERIFIER_READY": True,
        "VERIFIED_AUTHORIZATION_PUBLISHER_READY": True,
        "REAL_ANALYSIS_AUTHORIZATION_LIFECYCLE_QUALIFIED": True,
        "FROZEN_FIREWALL_COMPATIBILITY_PASS": True,
        "FORMAL_ANALYSIS_AUTHORIZATION_PRESENT": False,
        "REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED": False,
        "REAL_SCIENTIFIC_ANALYSIS_EXECUTED": False,
        "REAL_VALUES_ALREADY_UNBLINDED": False,
        "REAL_FORMAL_RESULT_FILES_READ": 0,
        "REAL_P_VALUE_COUNT": 0,
        "SCIENTIFIC_ANALYSIS_IMPLEMENTATION_CHANGED": False,
        "READY_FOR_SEPARATE_REAL_ANALYSIS_AUTHORIZATION": True,
        "analysis_lock_release_commit_intentionally_unbound_to_avoid_self_reference": True,
    })
    if sha256_file(output_schema_path) != OUTPUT_SCHEMA_SHA256:
        raise AuthorizationControlError("frozen analysis output schema changed")
    lock.pop("analysis_lock_fingerprint", None)
    lock["analysis_lock_fingerprint"] = canonical_fingerprint(lock)
    encoded = json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n"
    v2_path = lock_dir / "locked_analysis_lock_v2.json"
    compatibility_path = lock_dir / "locked_analysis_lock_v1.json"
    v2_path.write_text(encoded, encoding="utf-8")
    compatibility_path.write_bytes(v2_path.read_bytes())
    lock_sha = sha256_file(v2_path)
    (lock_dir / "locked_analysis_lock_v2.sha256").write_text(
        f"{lock_sha}  locked_analysis_lock_v2.json\n", encoding="utf-8"
    )
    (lock_dir / "locked_analysis_lock_v1.sha256").write_text(
        f"{lock_sha}  locked_analysis_lock_v1.json\n", encoding="utf-8"
    )
    lines = [
        f"{sha256_file(lock_dir / name)}  {name}\n" for name in LOCK_V2_CORE_FILES
    ]
    (lock_dir / "LOCK_CORE_SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    return lock
