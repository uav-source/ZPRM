"""Independent structural and SHA verifier for the execution-chain artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .phase_a_trial_result_schema import file_sha256


TABLES = (
    "raw_trial_inventory.csv",
    "backend_trial_inventory.csv",
    "failure_inventory.csv",
    "input_pairing_audit.csv",
    "translation_summary.csv",
    "rotation_summary.csv",
    "runtime_summary.csv",
    "analysis_verifier_comparison.csv",
    "publication_inventory.csv",
    "gate_summary.csv",
)
FIGURES = (
    "fixture_translation_updates.png",
    "fixture_rotation_updates.png",
    "fixture_backend_runtime.png",
    "fixture_result_status.png",
    "fixture_input_pairing.png",
)
ROOT_FILES = (
    "execution_chain_audit_report.md",
    "final_decision.json",
    "run_manifest.json",
    "implementation_manifest.json",
    "schema_contract.json",
    "resume_audit.json",
    "tamper_audit.json",
    "analysis_output.json",
    "independent_verification.json",
    "artifact_verification.json",
    "SHA256SUMS",
)


def required_relative_paths() -> tuple[str, ...]:
    return tuple(f"tables/{name}" for name in TABLES) + tuple(
        f"figures/{name}" for name in FIGURES
    ) + ROOT_FILES


def verify_phase_a_execution_chain_artifact(path: str | Path) -> dict[str, Any]:
    artifact = Path(path).resolve()
    required = required_relative_paths()
    missing = [name for name in required if not (artifact / name).is_file()]
    empty = [
        name
        for name in required
        if (artifact / name).is_file() and (artifact / name).stat().st_size <= 0
    ]
    decision_errors: list[str] = []
    try:
        decision = json.loads((artifact / "final_decision.json").read_text(encoding="utf-8"))
        if decision.get("PHASE_A_STAGE1_BACKEND_RUN_AUTHORIZED") is not False:
            decision_errors.append("formal backend execution authority must remain false")
        if decision.get("DAY1_SCIENTIFIC_VALIDATION_PASS") != "NOT_EVALUATED":
            decision_errors.append("scientific validation must remain NOT_EVALUATED")
    except (OSError, json.JSONDecodeError) as error:
        decision_errors.append(f"invalid final decision: {error}")
    report_label_pass = False
    try:
        report = (artifact / "execution_chain_audit_report.md").read_text(encoding="utf-8")
        report_label_pass = (
            "FIXTURE EXECUTION-CHAIN AUDIT" in report
            and "NOT SCIENTIFIC PHASE-A DATA" in report
        )
    except OSError:
        pass
    sha_errors: list[str] = []
    try:
        lines = (artifact / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        listed: set[str] = set()
        for line in lines:
            digest, relative = line.split("  ", 1)
            listed.add(relative)
            candidate = artifact / relative
            if not candidate.is_file() or file_sha256(candidate) != digest:
                sha_errors.append(relative)
        expected_listed = set(required) - {"SHA256SUMS"}
        if listed != expected_listed:
            sha_errors.append("SHA256SUMS inventory mismatch")
    except (OSError, ValueError) as error:
        sha_errors.append(f"invalid SHA256SUMS: {error}")
    passed = not missing and not empty and not decision_errors and report_label_pass and not sha_errors
    return {
        "PHASE_A_EXECUTION_CHAIN_ARTIFACT_VERIFICATION_PASS": passed,
        "empty_files": empty,
        "decision_errors": decision_errors,
        "fixture_label_pass": report_label_pass,
        "missing_files": missing,
        "required_file_count": len(required),
        "schema_version": "phase_a_execution_chain_artifact_verification_v1",
        "sha256_errors": sha_errors,
        "sha256_validation_pass": not sha_errors,
    }


__all__ = [
    "FIGURES",
    "ROOT_FILES",
    "TABLES",
    "required_relative_paths",
    "verify_phase_a_execution_chain_artifact",
]
