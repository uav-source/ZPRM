"""Independent verifier for the seed-free 3/6 fixture publication.

This module does not import a fixture generator, registration backend, primary
analyzer, or publisher.  It reconstructs the fixture evidence contract from
the files inside the published artifact and re-runs strict result and resume
validation from disk.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .contracts import canonical_json_sha256, file_sha256, write_json
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    canonical_json_bytes,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .phase_a_trial_result_writer import result_filename
from .phase_a_trial_resume import validate_existing_trial_result_for_resume


FIXED_PUBLICATION_FILES = (
    "fixture_publication_report.md",
    "primary_analysis.json",
    "independent_verification.json",
    "resume_validation.json",
    "source_binding.json",
    "final_decision.json",
    "evidence/frozen_experiment_manifest.json",
    "evidence/fixture_plan.json",
    "evidence/fixture_snapshot_lock.json",
    "evidence/raw_result_manifest.json",
    "SHA256SUMS",
)
OPTIONAL_VERIFIER_OUTPUT = "artifact_verification.json"
BACKENDS = (OPEN3D_BACKEND, PCL_BACKEND)
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _expected_common(
    *,
    row: Mapping[str, Any],
    plan_row: Mapping[str, Any],
    lock_row: Mapping[str, Any],
    plan_sha256: str,
    lock_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    return {
        "backend": row["backend"],
        "condition": plan_row["condition"],
        "implementation_sha256": implementation_sha256,
        "planned_trial_id": row["planned_trial_id"],
        "protocol_sha256": plan_sha256,
        "reference_pose_checksum": lock_row["reference_pose_checksum"],
        "scene_variant": plan_row["scene_variant"],
        "schema_version": "phase_a_trial_result_v1",
        "snapshot_checksum": lock_row["snapshot_checksum"],
        "snapshot_id": plan_row["snapshot_id"],
        "snapshot_lock_sha256": lock_sha256,
        "source_checksum": lock_row["source_checksum"],
        "target_checksum": lock_row["target_checksum"],
    }


def _semantic_audit(artifact: Path) -> dict[str, Any]:
    manifest_path = artifact / "evidence/frozen_experiment_manifest.json"
    plan_path = artifact / "evidence/fixture_plan.json"
    lock_path = artifact / "evidence/fixture_snapshot_lock.json"
    raw_path = artifact / "evidence/raw_result_manifest.json"
    manifest = _strict_object(manifest_path, "frozen experiment manifest")
    plan = _strict_object(plan_path, "fixture plan")
    lock = _strict_object(lock_path, "fixture snapshot lock")
    raw = _strict_object(raw_path, "fixture raw result manifest")

    stored_manifest_sha = manifest.get("manifest_payload_sha256")
    manifest_payload = {
        name: value
        for name, value in manifest.items()
        if name != "manifest_payload_sha256"
    }
    if (
        type(stored_manifest_sha) is not str
        or stored_manifest_sha != canonical_json_sha256(manifest_payload)
    ):
        raise ValueError("frozen experiment manifest payload SHA mismatch")
    if (
        plan.get("schema_version")
        != "phase_a_execution_chain_fixture_plan_v1"
        or plan.get("fixture_only") is not True
        or plan.get("formal_phase_a") is not False
        or plan.get("random_seed_used") is not False
        or type(plan.get("snapshots")) is not list
        or len(plan["snapshots"]) != 3
    ):
        raise ValueError("fixture plan contract mismatch")
    plan_sha = file_sha256(plan_path)
    lock_sha = file_sha256(lock_path)
    if (
        lock.get("schema_version")
        != "phase_a_execution_chain_fixture_snapshot_lock_v1"
        or lock.get("fixture_only") is not True
        or lock.get("formal_phase_a") is not False
        or lock.get("formal_seed_values_included") is not False
        or lock.get("random_seed_used") is not False
        or lock.get("fixture_plan_sha256") != plan_sha
        or type(lock.get("snapshots")) is not list
        or len(lock["snapshots"]) != 3
    ):
        raise ValueError("fixture snapshot lock contract mismatch")
    plan_by_id = {row["snapshot_id"]: row for row in plan["snapshots"]}
    lock_by_id = {row["snapshot_id"]: row for row in lock["snapshots"]}
    if len(plan_by_id) != 3 or set(plan_by_id) != set(lock_by_id):
        raise ValueError("fixture plan/lock snapshot inventory mismatch")
    for snapshot_id in plan_by_id:
        if any(
            plan_by_id[snapshot_id].get(name) != lock_by_id[snapshot_id].get(name)
            for name in (
                "condition",
                "expected_failure_classifications",
                "scene_variant",
                "snapshot_id",
            )
        ):
            raise ValueError("fixture plan/lock metadata mismatch")

    expected_ids = {
        f"{snapshot_id}/{backend}"
        for snapshot_id in plan_by_id
        for backend in BACKENDS
    }
    if (
        set(raw) != {"results", "run_id", "schema_version"}
        or raw.get("run_id") != "phase-a-minimal-harness-fixture-v1"
        or raw.get("schema_version") != "phase_a_raw_result_manifest_v1"
        or type(raw.get("results")) is not dict
        or set(raw["results"]) != expected_ids
    ):
        raise ValueError("fixture raw manifest inventory mismatch")
    if raw_path.read_bytes() != canonical_json_bytes(raw):
        raise ValueError("fixture raw manifest is not canonical JSON")

    rows: list[dict[str, Any]] = []
    resumed_rows: list[dict[str, Any]] = []
    result_inventory: list[dict[str, str]] = []
    for trial_id in sorted(expected_ids):
        entry = raw["results"][trial_id]
        if (
            type(entry) is not dict
            or set(entry) != {"path", "planned_trial_id", "sha256"}
            or entry.get("planned_trial_id") != trial_id
            or entry.get("path") != result_filename(trial_id)
            or type(entry.get("sha256")) is not str
            or not SHA_PATTERN.fullmatch(entry["sha256"])
        ):
            raise ValueError("fixture raw result entry mismatch")
        result_path = artifact / "evidence/raw_results" / entry["path"]
        if file_sha256(result_path) != entry["sha256"]:
            raise ValueError("fixture raw result SHA mismatch")
        row = validate_phase_a_trial_result_strict(load_json_strict(result_path))
        if result_path.read_bytes() != canonical_json_bytes(row):
            raise ValueError("fixture raw result is not canonical JSON")
        snapshot_id, backend = trial_id.rsplit("/", 1)
        if row["snapshot_id"] != snapshot_id or row["backend"] != backend:
            raise ValueError("fixture raw result identity mismatch")
        expected = _expected_common(
            row=row,
            plan_row=plan_by_id[snapshot_id],
            lock_row=lock_by_id[snapshot_id],
            plan_sha256=plan_sha,
            lock_sha256=lock_sha,
            implementation_sha256=stored_manifest_sha,
        )
        resumed = validate_existing_trial_result_for_resume(
            result_path, manifest_entry=entry, expected=expected
        )
        rows.append(row)
        resumed_rows.append(resumed)
        result_inventory.append(
            {
                "path": f"evidence/raw_results/{entry['path']}",
                "planned_trial_id": trial_id,
                "sha256": entry["sha256"],
            }
        )

    scientific_rows = [
        {name: value for name, value in row.items() if name != "runtime_ms"}
        for row in rows
    ]
    resumed_scientific_rows = [
        {name: value for name, value in row.items() if name != "runtime_ms"}
        for row in resumed_rows
    ]
    resume_difference_count = sum(
        left != right
        for left, right in zip(scientific_rows, resumed_scientific_rows)
    )
    backend_counts = Counter(row["backend"] for row in rows)
    pairing_violation_count = 0
    outcome_violation_count = 0
    for snapshot_id, plan_row in plan_by_id.items():
        selected = [row for row in rows if row["snapshot_id"] == snapshot_id]
        lock_row = lock_by_id[snapshot_id]
        checksum_fields = (
            "snapshot_checksum",
            "source_checksum",
            "target_checksum",
            "reference_pose_checksum",
        )
        if (
            len(selected) != 2
            or {row["backend"] for row in selected} != set(BACKENDS)
            or any(
                any(row[name] != lock_row[name] for name in checksum_fields)
                for row in selected
            )
            or any(selected[0][name] != selected[1][name] for name in checksum_fields)
        ):
            pairing_violation_count += 1
        expected_failures = set(plan_row["expected_failure_classifications"])
        for row in selected:
            successful = row["failure_classification"] == "NONE"
            if (
                row["failure_classification"] not in expected_failures
                or row["solver_failure"] is successful
                or (successful and row["finite_output"] is not True)
            ):
                outcome_violation_count += 1

    primary = _strict_object(artifact / "primary_analysis.json", "primary analysis")
    independent = _strict_object(
        artifact / "independent_verification.json", "independent verification"
    )
    comparison_fields = independent.get("comparison_fields")
    recomputed = independent.get("recomputed")
    if type(comparison_fields) is not list or type(recomputed) is not dict:
        raise ValueError("independent verification comparison contract mismatch")
    exact_differences = [
        name
        for name in comparison_fields
        if primary.get(name) != recomputed.get(name)
    ]
    primary_independent_pass = bool(
        primary.get("trial_count") == 6
        and primary.get("decision", {}).get(
            "PHASE_A_EXECUTION_CHAIN_ANALYSIS_PASS"
        )
        is True
        and primary.get("decision", {}).get(
            "PHASE_A_EXECUTION_CHAIN_FAILURE_CLASSIFICATION_PASS"
        )
        is True
        and primary.get("decision", {}).get(
            "PHASE_A_EXECUTION_CHAIN_FIXTURE_PASS"
        )
        is True
        and primary.get("decision", {}).get(
            "PHASE_A_EXECUTION_CHAIN_SCHEMA_PASS"
        )
        is True
        and independent.get(
            "PHASE_A_EXECUTION_CHAIN_INDEPENDENT_VERIFIER_PASS"
        )
        is True
        and independent.get(
            "PHASE_A_EXECUTION_CHAIN_ANALYSIS_VERIFIER_MATCH"
        )
        is True
        and independent.get("difference_count") == 0
        and independent.get("differences") == []
        and not exact_differences
    )

    resume = _strict_object(
        artifact / "resume_validation.json", "resume validation"
    )
    expected_resume = {
        "RESUME_VALIDATOR_SCIENTIFIC_EQUIVALENCE_PASS": True,
        "backend_execution_count": 0,
        "resume_validated_trial_count": 6,
        "schema_version": "phase_a_fixture_resume_validation_v1",
        "scientific_difference_count": 0,
        "trial_ids": sorted(expected_ids),
    }
    resume_pass = bool(
        resume == expected_resume and resume_difference_count == 0
    )

    expected_binding = {
        "fixture_plan_sha256": plan_sha,
        "fixture_snapshot_lock_sha256": lock_sha,
        "frozen_experiment_manifest_file_sha256": file_sha256(manifest_path),
        "frozen_experiment_manifest_payload_sha256": stored_manifest_sha,
        "raw_result_manifest_sha256": file_sha256(raw_path),
        "result_inventory": result_inventory,
        "run_id": raw["run_id"],
        "schema_version": "phase_a_fixture_publication_source_binding_v1",
    }
    source_binding = _strict_object(
        artifact / "source_binding.json", "fixture source binding"
    )
    source_binding_pass = source_binding == expected_binding

    decision = _strict_object(artifact / "final_decision.json", "fixture decision")
    expected_decision = {
        "FIXTURE_DUAL_BACKEND_OUTCOME_PASS": outcome_violation_count == 0,
        "FIXTURE_INPUT_PAIRING_PASS": pairing_violation_count == 0,
        "FIXTURE_PRIMARY_INDEPENDENT_EXACT_MATCH_PASS": (
            primary_independent_pass
        ),
        "FIXTURE_PUBLICATION_EVIDENCE_COMPLETE": True,
        "FIXTURE_RESUME_SCIENTIFIC_EQUIVALENCE_PASS": resume_pass,
        "analysis_verifier_difference_count": len(exact_differences),
        "backend_execution_count": 0,
        "fixture_snapshot_count": len(plan_by_id),
        "fixture_trial_count": len(rows),
        "input_pairing_violation_count": pairing_violation_count,
        "open3d_trial_count": backend_counts[OPEN3D_BACKEND],
        "pcl_trial_count": backend_counts[PCL_BACKEND],
        "resume_backend_execution_count": 0,
        "resume_scientific_difference_count": resume_difference_count,
        "rng_instantiation_count": 0,
        "schema_version": "phase_a_fixture_publication_decision_v1",
    }
    decision_pass = decision == expected_decision
    report = (artifact / "fixture_publication_report.md").read_text(
        encoding="utf-8"
    )
    label_pass = bool(
        "FIXTURE AUDIT — NOT SCIENTIFIC DATA" in report
        and "No registration backend or random-number generator was executed"
        in report
    )
    return {
        "analysis_verifier_difference_count": len(exact_differences),
        "backend_outcome_pass": outcome_violation_count == 0,
        "decision_semantics_pass": decision_pass,
        "dual_backend_cardinality_pass": backend_counts
        == Counter({OPEN3D_BACKEND: 3, PCL_BACKEND: 3}),
        "fixture_label_pass": label_pass,
        "input_pairing_pass": pairing_violation_count == 0,
        "primary_independent_exact_match_pass": primary_independent_pass,
        "resume_scientific_equivalence_pass": resume_pass,
        "source_binding_pass": source_binding_pass,
    }


def _sha_audit(
    artifact: Path, expected_sha_files: set[str]
) -> dict[str, Any]:
    listed: dict[str, str] = {}
    malformed_count = 0
    unsafe_count = 0
    mismatch_count = 0
    missing_count = 0
    try:
        lines = (artifact / "SHA256SUMS").read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeError):
        lines = []
        missing_count += 1
    for line in lines:
        try:
            digest, relative = line.split("  ", 1)
        except ValueError:
            malformed_count += 1
            continue
        relative_path = Path(relative)
        if (
            not SHA_PATTERN.fullmatch(digest)
            or relative in listed
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            unsafe_count += 1
            continue
        listed[relative] = digest
        candidate = artifact / relative
        if not candidate.is_file():
            missing_count += 1
        elif file_sha256(candidate) != digest:
            mismatch_count += 1
    inventory_pass = set(listed) == expected_sha_files
    passed = bool(
        inventory_pass
        and listed
        and malformed_count == 0
        and unsafe_count == 0
        and mismatch_count == 0
        and missing_count == 0
    )
    return {
        "sha256_entry_count": len(listed),
        "sha256_inventory_pass": inventory_pass,
        "sha256_malformed_count": malformed_count,
        "sha256_mismatch_count": mismatch_count,
        "sha256_missing_count": missing_count,
        "sha256_validation_pass": passed,
        "unsafe_sha256_path_count": unsafe_count,
    }


def verify_fixture_publication_artifact(
    path: str | Path, *, write_report: bool = True
) -> dict[str, Any]:
    """Revalidate published 3/6 evidence, SHA inventory, and semantics."""

    artifact = Path(path).resolve()
    semantic_errors: list[str] = []
    result_files: set[str] = set()
    try:
        raw = _strict_object(
            artifact / "evidence/raw_result_manifest.json",
            "fixture raw result manifest",
        )
        entries = raw.get("results")
        if type(entries) is not dict:
            raise ValueError("fixture raw result inventory is not an object")
        result_files = {
            f"evidence/raw_results/{entry['path']}"
            for entry in entries.values()
            if type(entry) is dict and type(entry.get("path")) is str
        }
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        semantic_errors.append(str(error))

    expected_files = set(FIXED_PUBLICATION_FILES) | result_files
    actual_files = {
        candidate.relative_to(artifact).as_posix()
        for candidate in artifact.rglob("*")
        if candidate.is_file()
        and candidate.relative_to(artifact).as_posix()
        != OPTIONAL_VERIFIER_OUTPUT
    }
    missing_files = sorted(expected_files - actual_files)
    extra_files = sorted(actual_files - expected_files)
    empty_files = sorted(
        relative
        for relative in expected_files & actual_files
        if (artifact / relative).stat().st_size == 0
    )
    inventory_pass = bool(not missing_files and not extra_files and not empty_files)
    sha = _sha_audit(
        artifact,
        expected_files - {"SHA256SUMS"},
    )
    semantics: dict[str, Any] = {
        "analysis_verifier_difference_count": None,
        "backend_outcome_pass": False,
        "decision_semantics_pass": False,
        "dual_backend_cardinality_pass": False,
        "fixture_label_pass": False,
        "input_pairing_pass": False,
        "primary_independent_exact_match_pass": False,
        "resume_scientific_equivalence_pass": False,
        "source_binding_pass": False,
    }
    if inventory_pass and sha["sha256_validation_pass"]:
        try:
            semantics = _semantic_audit(artifact)
        except (
            KeyError,
            OSError,
            UnicodeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            semantic_errors.append(str(error))
    semantic_pass = bool(
        not semantic_errors
        and all(
            semantics[name] is True
            for name in (
                "backend_outcome_pass",
                "decision_semantics_pass",
                "dual_backend_cardinality_pass",
                "fixture_label_pass",
                "input_pairing_pass",
                "primary_independent_exact_match_pass",
                "resume_scientific_equivalence_pass",
                "source_binding_pass",
            )
        )
        and semantics["analysis_verifier_difference_count"] == 0
    )
    passed = bool(
        inventory_pass and sha["sha256_validation_pass"] and semantic_pass
    )
    report: dict[str, Any] = {
        "FIXTURE_ARTIFACT_VERIFICATION_PASS": passed,
        **semantics,
        "artifact_inventory_pass": inventory_pass,
        "empty_file_count": len(empty_files),
        "empty_files": empty_files,
        "extra_file_count": len(extra_files),
        "extra_files": extra_files,
        "missing_file_count": len(missing_files),
        "missing_files": missing_files,
        "schema_version": "phase_a_fixture_publication_artifact_verification_v1",
        "semantic_error_count": len(semantic_errors),
        "semantic_errors": semantic_errors,
        "recorded_verification_match_pass": True,
        **sha,
    }
    recorded_path = artifact / OPTIONAL_VERIFIER_OUTPUT
    if recorded_path.is_file():
        try:
            recorded_match = _strict_object(
                recorded_path, "recorded fixture artifact verification"
            ) == report
        except (
            OSError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ):
            recorded_match = False
        if not recorded_match:
            report["FIXTURE_ARTIFACT_VERIFICATION_PASS"] = False
            report["recorded_verification_match_pass"] = False
    if write_report:
        write_json(recorded_path, report)
    return report


__all__ = [
    "BACKENDS",
    "FIXED_PUBLICATION_FILES",
    "OPTIONAL_VERIFIER_OUTPUT",
    "verify_fixture_publication_artifact",
]
