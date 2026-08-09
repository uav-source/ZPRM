"""Independent disk-first verifier for Phase A execution-chain audit results.

This module deliberately does not import the primary analyzer or publisher.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    file_sha256,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .rotation_metrics import rotation_metric_audit


def _summaries(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row[field] is not None:
            grouped[row["backend"]].append(float(row[field]))
    output = []
    for backend in (OPEN3D_BACKEND, PCL_BACKEND):
        values = np.asarray(grouped.get(backend, []), dtype=np.float64)
        output.append(
            {
                "backend": backend,
                "count": len(values),
                "maximum": float(values.max()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "q95_linear": float(np.quantile(values, 0.95, method="linear")) if len(values) else None,
            }
        )
    return output


def independently_verify_phase_a_stage1_fixture(
    *,
    run_dir: str | Path,
    fixture_plan: str | Path,
    fixture_lock: str | Path,
    analysis_output: Mapping[str, Any],
) -> dict[str, Any]:
    directory = Path(run_dir).resolve()
    plan_path, lock_path = Path(fixture_plan), Path(fixture_lock)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("fixture_plan_sha256") != file_sha256(plan_path):
        raise ValueError("independent verifier fixture plan SHA mismatch")
    expected_ids = {
        f"{snapshot['snapshot_id']}/{backend}"
        for snapshot in plan["snapshots"]
        for backend in (OPEN3D_BACKEND, PCL_BACKEND)
    }
    manifest = json.loads((directory / "raw_result_manifest.json").read_text(encoding="utf-8"))
    if set(manifest.get("results", {})) != expected_ids:
        raise ValueError("independent verifier trial ID set mismatch")
    lock_rows = {row["snapshot_id"]: row for row in lock["snapshots"]}
    rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    metric_mismatches = 0
    for trial_id in sorted(expected_ids):
        entry = manifest["results"][trial_id]
        path = directory / "raw_results" / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("independent verifier raw result SHA mismatch")
        row = validate_phase_a_trial_result_strict(load_json_strict(path))
        if row["planned_trial_id"] != trial_id:
            raise ValueError("independent verifier trial identity mismatch")
        expected = lock_rows[row["snapshot_id"]]
        checksum_match = all(
            row[name] == expected[name]
            for name in ("snapshot_checksum", "source_checksum", "target_checksum", "reference_pose_checksum")
        )
        pairing_rows.append(
            {
                "backend": row["backend"],
                "checksum_match": checksum_match,
                "planned_trial_id": row["planned_trial_id"],
                "snapshot_id": row["snapshot_id"],
            }
        )
        transform = row["final_transform_4x4"]
        if transform is not None:
            reference = np.asarray(expected["reference_pose_4x4"], dtype=np.float64)
            estimate = np.asarray(transform, dtype=np.float64)
            delta = np.linalg.inv(reference) @ estimate
            translation = float(np.linalg.norm(delta[:3, 3]))
            rotation = rotation_metric_audit(delta[:3, :3], np.eye(3))["rotation_error_rad"]
            if not math.isclose(translation, float(row["translation_update_m"]), abs_tol=1.0e-12):
                metric_mismatches += 1
            if not math.isclose(float(rotation), float(row["rotation_update_rad"]), abs_tol=1.0e-12):
                metric_mismatches += 1
            raw_audit = rotation_metric_audit(estimate[:3, :3], reference[:3, :3])
            if not math.isclose(float(np.linalg.det(estimate[:3, :3])), float(row["raw_rotation_determinant"]), abs_tol=1.0e-12):
                metric_mismatches += 1
            if not math.isclose(float(raw_audit["orthogonality_defect_fro"]), float(row["orthogonality_defect_fro"]), abs_tol=1.0e-12):
                metric_mismatches += 1
        rows.append(row)
    backend_inventory = []
    for backend in (OPEN3D_BACKEND, PCL_BACKEND):
        selected = [row for row in rows if row["backend"] == backend]
        backend_inventory.append(
            {
                "backend": backend,
                "failure_count": sum(row["solver_failure"] for row in selected),
                "success_count": sum(not row["solver_failure"] for row in selected),
                "trial_count": len(selected),
            }
        )
    failure_counts = Counter(row["failure_classification"] for row in rows)
    failures = [
        {"count": count, "failure_classification": name}
        for name, count in sorted(failure_counts.items())
    ]
    expected_failure = {
        snapshot["condition"]: set(snapshot["expected_failure_classifications"])
        for snapshot in plan["snapshots"]
    }
    classification_pass = all(
        row["failure_classification"] in expected_failure[row["condition"]]
        and row["solver_failure"] == (row["failure_classification"] != "NONE")
        for row in rows
    )
    fixture_pass = len(rows) == 6 and all(
        (row["finite_output"] and not row["solver_failure"])
        if row["condition"] != "FIXTURE_NO_CORRESPONDENCE"
        else row["solver_failure"]
        for row in rows
    )
    pairing_violations = sum(not row["checksum_match"] for row in pairing_rows)
    decision = {
        "PHASE_A_EXECUTION_CHAIN_ANALYSIS_PASS": len(rows) == 6 and metric_mismatches == 0 and pairing_violations == 0,
        "PHASE_A_EXECUTION_CHAIN_FAILURE_CLASSIFICATION_PASS": classification_pass,
        "PHASE_A_EXECUTION_CHAIN_FIXTURE_PASS": fixture_pass,
        "PHASE_A_EXECUTION_CHAIN_SCHEMA_PASS": len(rows) == 6,
        "input_checksum_pairing_violation_count": pairing_violations,
        "metric_recomputation_mismatch_count": metric_mismatches,
    }
    recomputed = {
        "backend_inventory": backend_inventory,
        "decision": decision,
        "failure_inventory": failures,
        "input_pairing_audit": pairing_rows,
        "rotation_summary": _summaries(rows, "rotation_update_rad"),
        "runtime_summary": _summaries(rows, "runtime_ms"),
        "snapshot_ids": sorted({row["snapshot_id"] for row in rows}),
        "translation_summary": _summaries(rows, "translation_update_m"),
        "trial_count": len(rows),
        "trial_ids": sorted(row["planned_trial_id"] for row in rows),
    }
    comparison_fields = tuple(recomputed)
    differences = [name for name in comparison_fields if analysis_output.get(name) != recomputed[name]]
    return {
        "PHASE_A_EXECUTION_CHAIN_INDEPENDENT_VERIFIER_PASS": len(differences) == 0 and metric_mismatches == 0,
        "PHASE_A_EXECUTION_CHAIN_ANALYSIS_VERIFIER_MATCH": len(differences) == 0,
        "comparison_fields": list(comparison_fields),
        "difference_count": len(differences),
        "differences": differences,
        "recomputed": recomputed,
        "schema_version": "phase_a_execution_chain_independent_verification_v1",
        "trial_count": len(rows),
    }


__all__ = ["independently_verify_phase_a_stage1_fixture"]
