"""Build the frozen analysis lock after the analysis-code commit exists."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .contract_v1 import (
    C2_COMMIT,
    C2_TAG,
    C2_TAG_OBJECT,
    CENTERED_PERMUTATION_DRAW_COUNT,
    CENTERED_PERMUTATION_P_DENOMINATOR,
    CENTERED_PERMUTATION_SEED,
    POSTRUN_VERIFICATION_COMMIT,
    POSTRUN_VERIFICATION_TAG,
    POSTRUN_VERIFIER_CODE_COMMIT,
    PRIMARY_EXACT_ALLOCATION_COUNT,
    PRIMARY_EXACT_P_DENOMINATOR,
    PROTECTED_BINDINGS,
    R3_LOCK_FINGERPRINT,
    RAW_EXECUTION_COMMIT,
    RAW_EXECUTION_TAG,
    sha256_file,
    verify_contract_bindings,
)


ANALYSIS_CODE_PATHS = (
    "experiments/__init__.py",
    "experiments/mid360_formal_batch1/__init__.py",
    "experiments/mid360_formal_batch1/locked_analysis/__init__.py",
    "experiments/mid360_formal_batch1/locked_analysis/contract_v1.py",
    "experiments/mid360_formal_batch1/locked_analysis/authoritative_outcomes_v1.py",
    "experiments/mid360_formal_batch1/locked_analysis/descriptive_summary_v1.py",
    "experiments/mid360_formal_batch1/locked_analysis/exact_scene_permutation_v1.py",
    "experiments/mid360_formal_batch1/locked_analysis/cross_backend_v1.py",
    "experiments/mid360_formal_batch1/locked_analysis/reassociation_v1.py",
    "experiments/mid360_formal_batch1/locked_analysis/systematic_component_v1.py",
    "experiments/mid360_formal_batch1/locked_analysis/formal_firewall_v1.py",
    "experiments/mid360_formal_batch1/locked_analysis/lock_v1.py",
    "experiments/mid360_formal_batch1/locked_analysis/lock_verify_v1.py",
    "experiments/mid360_formal_batch1/locked_analysis/output_schema_v1.json",
    "experiments/mid360_formal_batch1/locked_analysis/locked_analysis_v1.py",
    "tools/mid360_formal_batch1/run_zero_perturbation_locked_analysis_v1.py",
    "tools/mid360_formal_batch1/issue_zero_perturbation_locked_analysis_lock_v1.py",
    "tools/mid360_formal_batch1/verify_zero_perturbation_locked_analysis_lock_v1.py",
)

LOCK_CORE_FILES = (
    "analysis_code_inventory.csv",
    "analysis_contract_binding.json",
    "analysis_fixture_qualification.json",
    "analysis_no_real_results_attestation.json",
    "analysis_determinacy_binding.json",
    "locked_analysis_lock_v1.json",
    "locked_analysis_lock_v1.sha256",
)


class AnalysisLockError(RuntimeError):
    """Raised when the analysis lock cannot be issued exactly."""


def _git(repository: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["git", *args], cwd=repository, check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise AnalysisLockError(process.stderr.decode("utf-8", "replace").strip())
    return process.stdout


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _code_inventory(repository: Path, code_commit: str) -> list[dict[str, Any]]:
    if _git(repository, "cat-file", "-t", code_commit).strip() != b"commit":
        raise AnalysisLockError("analysis code commit does not exist")
    rows = []
    for relative in ANALYSIS_CODE_PATHS:
        path = repository / relative
        current = path.read_bytes()
        committed = _git(repository, "show", f"{code_commit}:{relative}")
        if current != committed:
            raise AnalysisLockError(f"analysis code differs from frozen commit: {relative}")
        rows.append({"path": relative, "sha256": hashlib.sha256(current).hexdigest(),
                     "bytes": len(current), "git_blob_exact": True})
    return rows


def issue_analysis_lock(
    repository: Path,
    lock_dir: Path,
    *,
    analysis_code_commit: str,
    qualification_path: Path,
    attestation_path: Path,
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    lock_dir = lock_dir.resolve(strict=False)
    if lock_dir.exists() and any(lock_dir.iterdir()):
        raise AnalysisLockError("analysis lock directory must be absent or empty")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if not (qualification.get("FIXTURE_ANALYSIS_PASS") is True
            and qualification.get("REAL_FORMAL_RESULT_FILES_READ") == 0):
        raise AnalysisLockError("fixture qualification is not PASS/zero-read")
    zero_fields = (
        "real_formal_result_files_read", "real_translation_values_read",
        "real_rotation_values_read", "real_turnover_values_read",
        "real_scene_aggregations", "real_weak_rich_comparisons",
        "real_exact_p_values", "real_centered_permutation_p_values",
        "registration_backend_calls",
    )
    if attestation.get("pass") is not True or any(attestation.get(key) != 0 for key in zero_fields):
        raise AnalysisLockError("no-real-result attestation is not exact PASS")
    bindings = verify_contract_bindings(repository)
    code_rows = _code_inventory(repository, analysis_code_commit)
    lock_dir.mkdir(parents=True, exist_ok=False)

    with (lock_dir / "analysis_code_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "sha256", "bytes", "git_blob_exact"))
        writer.writeheader(); writer.writerows(code_rows)
    contract_payload = {"schema": "fmb1_locked_analysis_contract_binding_v1", **bindings}
    _write_json(lock_dir / "analysis_contract_binding.json", contract_payload)
    (lock_dir / "analysis_fixture_qualification.json").write_bytes(qualification_path.read_bytes())
    (lock_dir / "analysis_no_real_results_attestation.json").write_bytes(attestation_path.read_bytes())
    determinacy = {
        "schema": "fmb1_locked_analysis_determinacy_binding_v1", "status": "PASS",
        "FMB1_LOCKED_ANALYSIS_CONTRACT_DETERMINATE": True,
        "required_under_specified_count": 0, "unresolved_root_definition_count": 0,
        "primary_exact_allocation_count": PRIMARY_EXACT_ALLOCATION_COUNT,
        "primary_p_value_denominator": PRIMARY_EXACT_P_DENOMINATOR,
        "centered_permutation_draw_count": CENTERED_PERMUTATION_DRAW_COUNT,
        "centered_permutation_seed": CENTERED_PERMUTATION_SEED,
        "centered_permutation_p_value_denominator": CENTERED_PERMUTATION_P_DENOMINATOR,
        "scene_strata_preserved": True, "systematic_formal_p_value_defined": False,
    }
    _write_json(lock_dir / "analysis_determinacy_binding.json", determinacy)
    r3_lock_path = repository / "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r3_lock/formal_batch1_zero_perturbation_lock_v1_1_exec_r3.json"
    lock: dict[str, Any] = {
        "schema": "fmb1_zero_perturbation_locked_analysis_lock_v1",
        "lock_id": "FMB1_ZERO_PERTURBATION_LOCKED_ANALYSIS_V1",
        "status": "ISSUED_AWAITING_SEPARATE_REAL_ANALYSIS_AUTHORIZATION",
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        "RAW_EXECUTION_COMMIT": RAW_EXECUTION_COMMIT,
        "RAW_EXECUTION_TAG": RAW_EXECUTION_TAG,
        "POSTRUN_VERIFIER_CODE_COMMIT": POSTRUN_VERIFIER_CODE_COMMIT,
        "POSTRUN_VERIFICATION_COMMIT": POSTRUN_VERIFICATION_COMMIT,
        "POSTRUN_VERIFICATION_TAG": POSTRUN_VERIFICATION_TAG,
        "POSTRUN_VERIFICATION_REPORT_SHA": PROTECTED_BINDINGS["postrun_verification_report"][1],
        "R3_LOCK_SHA": sha256_file(r3_lock_path),
        "R3_LOCK_FINGERPRINT": R3_LOCK_FINGERPRINT,
        "C2_COMMIT": C2_COMMIT, "C2_TAG": C2_TAG, "C2_TAG_OBJECT": C2_TAG_OBJECT,
        "LOCKED_ANALYSIS_CODE_COMMIT": analysis_code_commit,
        "analysis_output_schema_sha256": sha256_file(
            repository / "experiments/mid360_formal_batch1/locked_analysis/output_schema_v1.json"
        ),
        "fixture_qualification_sha256": sha256_file(lock_dir / "analysis_fixture_qualification.json"),
        "no_real_results_attestation_sha256": sha256_file(lock_dir / "analysis_no_real_results_attestation.json"),
        "analysis_code_inventory_sha256": sha256_file(lock_dir / "analysis_code_inventory.csv"),
        "analysis_contract_binding_sha256": sha256_file(lock_dir / "analysis_contract_binding.json"),
        "analysis_determinacy_binding_sha256": sha256_file(lock_dir / "analysis_determinacy_binding.json"),
        "protected_bindings": {role: {"path": path, "sha256": digest}
                               for role, (path, digest) in PROTECTED_BINDINGS.items()},
        "analysis_code_paths": list(ANALYSIS_CODE_PATHS),
        "formal_results_root": str((repository / "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1").resolve()),
        "future_analysis_output_root": str((repository / "results/mid360_formal_batch1/zero_perturbation_locked_analysis_v1").resolve()),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "primary_exact_allocation_count": 20, "primary_p_value_denominator": 20,
        "centered_permutation_draw_count": 10000,
        "centered_permutation_p_value_denominator": 10001,
        "scene_strata_preserved": True, "systematic_formal_p_value_defined": False,
        "capture_radius_status": "PRESERVED_SUPPLEMENTARY_NOT_EXECUTED",
        "synthetic_transfer_status": "MODEL_TRANSFER_NOT_COMPATIBLE",
        "REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED": False,
        "REAL_SCIENTIFIC_ANALYSIS_EXECUTED": False,
        "REAL_FORMAL_RESULT_FILES_READ_BEFORE_FREEZE": 0,
        "READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS": True,
    }
    lock["analysis_lock_fingerprint"] = _fingerprint(lock)
    lock_path = lock_dir / "locked_analysis_lock_v1.json"
    _write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)
    (lock_dir / "locked_analysis_lock_v1.sha256").write_text(
        f"{lock_sha}  locked_analysis_lock_v1.json\n", encoding="utf-8")
    lines = [f"{sha256_file(lock_dir / name)}  {name}\n" for name in LOCK_CORE_FILES]
    (lock_dir / "LOCK_CORE_SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    return lock
