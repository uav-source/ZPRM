"""Publisher for the fixture-only Phase A execution-chain audit."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .phase_a_execution_chain_artifact_verifier import (
    FIGURES,
    ROOT_FILES,
    TABLES,
    required_relative_paths,
    verify_phase_a_execution_chain_artifact,
)
from .phase_a_trial_result_schema import (
    OPEN3D_DIAGNOSTIC_FIELDS,
    PCL_DIAGNOSTIC_FIELDS,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    canonical_json_bytes,
    file_sha256,
)
from .phase_a_trial_result_writer import atomic_write_bytes


FIXTURE_LABEL = "FIXTURE AUDIT — NOT SCIENTIFIC DATA"


def _json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value), replace=False)


def _csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"), replace=False)


def _figure(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(range(len(labels)), values, color="#35618f")
    axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{title}\n{FIXTURE_LABEL}")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def publish_phase_a_execution_chain_audit(
    *,
    artifact_dir: str | Path,
    analysis: Mapping[str, Any],
    verification: Mapping[str, Any],
    fixture_run_manifest: Mapping[str, Any],
    resume_audit: Mapping[str, Any],
    tamper_audit: Mapping[str, Any],
    audit_protocol_lock: str | Path,
) -> dict[str, Any]:
    artifact = Path(artifact_dir).resolve()
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError("publisher requires an empty artifact directory")
    if (
        verification.get("PHASE_A_EXECUTION_CHAIN_INDEPENDENT_VERIFIER_PASS") is not True
        or verification.get("PHASE_A_EXECUTION_CHAIN_ANALYSIS_VERIFIER_MATCH") is not True
        or verification.get("difference_count") != 0
    ):
        raise PermissionError("publisher requires a matching independent verifier PASS")
    (artifact / "tables").mkdir(parents=True, exist_ok=True)
    (artifact / "figures").mkdir(parents=True, exist_ok=True)

    inventory = list(analysis["trial_inventory"])
    _csv(
        artifact / "tables/raw_trial_inventory.csv",
        inventory,
        [
            "planned_trial_id",
            "snapshot_id",
            "condition",
            "backend",
            "solver_failure",
            "failure_classification",
            "finite_output",
            "result_path",
            "result_sha256",
        ],
    )
    _csv(
        artifact / "tables/backend_trial_inventory.csv",
        analysis["backend_inventory"],
        ["backend", "trial_count", "success_count", "failure_count"],
    )
    _csv(
        artifact / "tables/failure_inventory.csv",
        analysis["failure_inventory"],
        ["failure_classification", "count"],
    )
    _csv(
        artifact / "tables/input_pairing_audit.csv",
        analysis["input_pairing_audit"],
        ["planned_trial_id", "snapshot_id", "backend", "checksum_match"],
    )
    for name, key in (
        ("translation_summary.csv", "translation_summary"),
        ("rotation_summary.csv", "rotation_summary"),
        ("runtime_summary.csv", "runtime_summary"),
    ):
        _csv(
            artifact / f"tables/{name}",
            analysis[key],
            ["backend", "count", "median", "q95_linear", "maximum"],
        )
    _csv(
        artifact / "tables/analysis_verifier_comparison.csv",
        [
            {
                "difference": field in verification["differences"],
                "field": field,
            }
            for field in verification["comparison_fields"]
        ],
        ["field", "difference"],
    )
    publication_rows = [
        {"artifact_type": "table", "path": f"tables/{name}", "fixture_only": True}
        for name in TABLES
    ] + [
        {"artifact_type": "figure", "path": f"figures/{name}", "fixture_only": True}
        for name in FIGURES
    ]
    _csv(
        artifact / "tables/publication_inventory.csv",
        publication_rows,
        ["artifact_type", "path", "fixture_only"],
    )

    gates = {
        "PHASE_A_EXECUTION_CHAIN_SCHEMA_PASS": analysis["decision"]["PHASE_A_EXECUTION_CHAIN_SCHEMA_PASS"],
        "PHASE_A_EXECUTION_CHAIN_WRITER_PASS": fixture_run_manifest.get("fixture_completed_trial_count") == 6,
        "PHASE_A_EXECUTION_CHAIN_RESUME_PASS": resume_audit.get("RESUME_SKIPPED_VALID_RESULT_COUNT") == 2 and resume_audit.get("RESUME_REEXECUTED_VALID_RESULT_COUNT") == 0,
        "PHASE_A_EXECUTION_CHAIN_FIXTURE_PASS": analysis["decision"]["PHASE_A_EXECUTION_CHAIN_FIXTURE_PASS"],
        "PHASE_A_EXECUTION_CHAIN_FAILURE_CLASSIFICATION_PASS": analysis["decision"]["PHASE_A_EXECUTION_CHAIN_FAILURE_CLASSIFICATION_PASS"],
        "PHASE_A_EXECUTION_CHAIN_ANALYSIS_PASS": analysis["decision"]["PHASE_A_EXECUTION_CHAIN_ANALYSIS_PASS"],
        "PHASE_A_EXECUTION_CHAIN_INDEPENDENT_VERIFIER_PASS": verification["PHASE_A_EXECUTION_CHAIN_INDEPENDENT_VERIFIER_PASS"],
        "PHASE_A_EXECUTION_CHAIN_ANALYSIS_VERIFIER_MATCH": verification["PHASE_A_EXECUTION_CHAIN_ANALYSIS_VERIFIER_MATCH"],
        "PHASE_A_EXECUTION_CHAIN_PUBLISHER_PASS": True,
        "PHASE_A_EXECUTION_CHAIN_ARTIFACT_VERIFICATION_PASS": True,
        "PHASE_A_EXECUTION_CHAIN_TAMPER_REJECTION_PASS": tamper_audit.get("TAMPERED_RESULT_REJECTED") is True,
        "PHASE_A_EXECUTION_CHAIN_INTERRUPTION_RESUME_PASS": resume_audit.get("RESUMED_AND_FRESH_RESULT_EQUIVALENT") is True and resume_audit.get("FINAL_TRIAL_COUNT_AFTER_RESUME") == 6,
    }
    _csv(
        artifact / "tables/gate_summary.csv",
        [{"gate": name, "value": value} for name, value in gates.items()],
        ["gate", "value"],
    )

    fixture_rows = list(analysis["fixture_summary"])
    trial_labels = [f"{row['condition'].replace('FIXTURE_', '')}\n{row['backend'].split('_')[0]}" for row in fixture_rows]
    _figure(
        artifact / "figures/fixture_translation_updates.png",
        "Fixture translation updates",
        trial_labels,
        [float(row["translation_update_m"] or 0.0) for row in fixture_rows],
        "translation update (m)",
    )
    _figure(
        artifact / "figures/fixture_rotation_updates.png",
        "Fixture rotation updates",
        trial_labels,
        [float(row["rotation_update_rad"] or 0.0) for row in fixture_rows],
        "rotation update (rad)",
    )
    runtime_by_backend = {row["backend"]: row["median"] or 0.0 for row in analysis["runtime_summary"]}
    _figure(
        artifact / "figures/fixture_backend_runtime.png",
        "Fixture backend runtime",
        list(runtime_by_backend),
        list(runtime_by_backend.values()),
        "median runtime (ms)",
    )
    _figure(
        artifact / "figures/fixture_result_status.png",
        "Fixture result status",
        trial_labels,
        [0.0 if row["solver_failure"] else 1.0 for row in fixture_rows],
        "valid success (1) / expected failure (0)",
    )
    _figure(
        artifact / "figures/fixture_input_pairing.png",
        "Fixture input pairing",
        [row["planned_trial_id"].split("/")[-1].split("_")[0] for row in analysis["input_pairing_audit"]],
        [1.0 if row["checksum_match"] else 0.0 for row in analysis["input_pairing_audit"]],
        "checksum match",
    )

    all_gates = all(gates.values())
    zero_boundaries = all(
        fixture_run_manifest.get(name) == 0
        for name in (
            "FORMAL_STAGE0_CACHE_READ_COUNT",
            "FORMAL_PHASE_A_SEED_ACCESS_COUNT",
            "FORMAL_PHASE_A_BACKEND_EXECUTION_COUNT",
            "FORMAL_PHASE_A_TRIAL_RESULT_COUNT",
            "NATIVE_EXECUTION_COUNT",
        )
    )
    audit_pass = bool(all_gates and zero_boundaries)
    decision = {
        **gates,
        "BACKEND_PHASE_A_COMPLETE": False,
        "CONFIRMATORY_AUTHORIZED": False,
        "DAY1_SCIENTIFIC_VALIDATION_PASS": "NOT_EVALUATED",
        "FULL_DEVELOPMENT_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PHASE_A_EXECUTION_CHAIN_AUDIT_PASS": audit_pass,
        "PHASE_A_ROUTE_PAUSED_FOR_CONCENTRATED_CODE_AUDIT": not audit_pass,
        "PHASE_A_STAGE1_BACKEND_EXECUTED": False,
        "PHASE_A_STAGE1_BACKEND_RUN_AUTHORIZED": False,
        "PHASE_A_STAGE1_RELOCK_AUTHORIZED": audit_pass,
        "PHASE_B_AUTHORIZED": False,
        "REAL_DATA_AUTHORIZED": False,
        "TWO_INDEPENDENT_BACKENDS_QUALIFIED": False,
        "schema_version": "phase_a_execution_chain_audit_decision_v1",
    }
    protocol_lock_sha = file_sha256(audit_protocol_lock)
    schema_contract = {
        "open3d_diagnostic_fields": sorted(OPEN3D_DIAGNOSTIC_FIELDS),
        "pcl_diagnostic_fields": sorted(PCL_DIAGNOSTIC_FIELDS),
        "required_common_field_count": len(REQUIRED_FIELDS),
        "required_common_fields": list(REQUIRED_FIELDS),
        "schema_path": "schemas/phase_a_trial_result_v1.schema.json",
        "schema_version": SCHEMA_VERSION,
    }
    run_manifest = {
        "FORMAL_PHASE_A_BACKEND_EXECUTION_COUNT": 0,
        "FORMAL_PHASE_A_SEED_ACCESS_COUNT": 0,
        "FORMAL_PHASE_A_TRIAL_RESULT_COUNT": 0,
        "FORMAL_STAGE0_CACHE_READ_COUNT": 0,
        "NATIVE_EXECUTION_COUNT": 0,
        "audit_protocol_lock_sha256": protocol_lock_sha,
        "fixture_open3d_trial_count": 3,
        "fixture_pcl_trial_count": 3,
        "fixture_snapshot_count": 3,
        "fixture_trial_count": analysis["trial_count"],
        "run_id": fixture_run_manifest["run_id"],
        "schema_version": "phase_a_execution_chain_audit_run_manifest_v1",
    }
    report = f"""# Phase A Execution-Chain Audit

**FIXTURE EXECUTION-CHAIN AUDIT**  
**NOT SCIENTIFIC PHASE-A DATA**

The strict `{SCHEMA_VERSION}` contract was exercised from an empty result
directory through Open3D and PCL, primary analysis, independent verification,
publication, artifact verification, and SHA validation. The three fixture
snapshots and six fixture trials are engineering evidence only.

- Trial count: {analysis['trial_count']}
- Independent-verifier differences: {verification['difference_count']}
- Input checksum pairing violations: {analysis['decision']['input_checksum_pairing_violation_count']}
- Interruption/resume skipped valid results: {resume_audit.get('RESUME_SKIPPED_VALID_RESULT_COUNT')}
- Tampered result rejected: {tamper_audit.get('TAMPERED_RESULT_REJECTED')}
- Formal Stage-0 cache reads: 0
- Formal Phase A backend executions/results: 0/0
- Native executions: 0

`PHASE_A_EXECUTION_CHAIN_AUDIT_PASS = {str(audit_pass).lower()}`  
`PHASE_A_STAGE1_RELOCK_AUTHORIZED = {str(audit_pass).lower()}`  
`PHASE_A_STAGE1_BACKEND_RUN_AUTHORIZED = false`  
`DAY1_SCIENTIFIC_VALIDATION_PASS = NOT_EVALUATED`
"""
    atomic_write_bytes(artifact / "execution_chain_audit_report.md", report.encode("utf-8"))
    _json(artifact / "analysis_output.json", analysis)
    _json(artifact / "independent_verification.json", verification)
    _json(artifact / "resume_audit.json", resume_audit)
    _json(artifact / "tamper_audit.json", tamper_audit)
    _json(artifact / "schema_contract.json", schema_contract)
    _json(artifact / "implementation_manifest.json", fixture_run_manifest["implementation"])
    _json(artifact / "run_manifest.json", run_manifest)
    _json(artifact / "final_decision.json", decision)

    # The final verifier payload is canonicalized first so SHA256SUMS can include it.
    verification_payload = {
        "PHASE_A_EXECUTION_CHAIN_ARTIFACT_VERIFICATION_PASS": True,
        "decision_errors": [],
        "empty_files": [],
        "fixture_label_pass": True,
        "missing_files": [],
        "required_file_count": len(required_relative_paths()),
        "schema_version": "phase_a_execution_chain_artifact_verification_v1",
        "sha256_errors": [],
        "sha256_validation_pass": True,
    }
    verification_bytes = canonical_json_bytes(verification_payload)
    paths_without_sha = set(required_relative_paths()) - {"SHA256SUMS"}
    sha_rows: list[tuple[str, str]] = []
    import hashlib

    for relative in sorted(paths_without_sha):
        if relative == "artifact_verification.json":
            digest = hashlib.sha256(verification_bytes).hexdigest()
        else:
            digest = file_sha256(artifact / relative)
        sha_rows.append((digest, relative))
    atomic_write_bytes(
        artifact / "SHA256SUMS",
        "".join(f"{digest}  {relative}\n" for digest, relative in sha_rows).encode("utf-8"),
    )
    atomic_write_bytes(artifact / "artifact_verification.json", verification_bytes)
    independent_artifact = verify_phase_a_execution_chain_artifact(artifact)
    if independent_artifact != verification_payload:
        raise RuntimeError(f"artifact verifier disagreed: {independent_artifact}")
    return {
        "artifact_dir": str(artifact),
        "artifact_file_count": len(required_relative_paths()),
        "figure_count": len(FIGURES),
        "gate_count": len(gates),
        "publication_pass": True,
        "table_count": len(TABLES),
    }


__all__ = ["FIXTURE_LABEL", "publish_phase_a_execution_chain_audit"]
