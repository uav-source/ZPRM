"""Strict primary analysis for fixture-only Phase A execution-chain results."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .phase_a_attempt_events import read_attempt_events
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    file_sha256,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)


def _summary(rows: Iterable[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row[field]
        if value is not None:
            grouped[row["backend"]].append(float(value))
    output: list[dict[str, Any]] = []
    for backend in (OPEN3D_BACKEND, PCL_BACKEND):
        values = np.asarray(grouped.get(backend, []), dtype=np.float64)
        output.append(
            {
                "backend": backend,
                "count": int(len(values)),
                "maximum": float(np.max(values)) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "q95_linear": (
                    float(np.quantile(values, 0.95, method="linear"))
                    if len(values)
                    else None
                ),
            }
        )
    return output


def _load_contract(plan_path: Path, lock_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        plan.get("fixture_only") is not True
        or plan.get("formal_phase_a") is not False
        or lock.get("fixture_only") is not True
        or lock.get("formal_phase_a") is not False
        or lock.get("fixture_plan_sha256") != file_sha256(plan_path)
    ):
        raise ValueError("fixture analysis contract is not frozen")
    return plan, lock


def _raw_results(run_dir: Path, expected_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = run_dir / "raw_result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("results")
    if type(entries) is not dict or set(entries) != expected_ids:
        raise ValueError("raw result manifest trial inventory mismatch")
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for trial_id in sorted(expected_ids):
        entry = entries[trial_id]
        if set(entry) != {"path", "planned_trial_id", "sha256"} or entry["planned_trial_id"] != trial_id:
            raise ValueError("raw result manifest entry mismatch")
        path = run_dir / "raw_results" / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("raw result SHA mismatch")
        value = validate_phase_a_trial_result_strict(load_json_strict(path))
        if value["planned_trial_id"] != trial_id:
            raise ValueError("raw result planned_trial_id mismatch")
        rows.append(value)
        inventory.append(
            {
                "backend": value["backend"],
                "condition": value["condition"],
                "failure_classification": value["failure_classification"],
                "finite_output": value["finite_output"],
                "planned_trial_id": trial_id,
                "result_path": f"raw_results/{path.name}",
                "result_sha256": entry["sha256"],
                "snapshot_id": value["snapshot_id"],
                "solver_failure": value["solver_failure"],
            }
        )
    return rows, inventory


def _recompute_update(row: Mapping[str, Any], reference: np.ndarray) -> dict[str, Any]:
    transform = row["final_transform_4x4"]
    if transform is None:
        return {"rotation_update_rad": None, "translation_update_m": None}
    estimate = np.asarray(transform, dtype=np.float64)
    delta = np.linalg.inv(reference) @ estimate
    translation = float(np.linalg.norm(delta[:3, 3]))
    rotation = delta[:3, :3]
    argument_cos = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    angle = float(math.atan2(0.5 * np.linalg.norm(skew), argument_cos))
    return {"rotation_update_rad": angle, "translation_update_m": translation}


def analyze_phase_a_stage1_fixture(
    *,
    run_dir: str | Path,
    fixture_plan: str | Path,
    fixture_lock: str | Path,
) -> dict[str, Any]:
    directory = Path(run_dir).resolve()
    plan, lock = _load_contract(Path(fixture_plan), Path(fixture_lock))
    lock_by_snapshot = {row["snapshot_id"]: row for row in lock["snapshots"]}
    expected_ids = {
        f"{snapshot['snapshot_id']}/{backend}"
        for snapshot in plan["snapshots"]
        for backend in (OPEN3D_BACKEND, PCL_BACKEND)
    }
    rows, trial_inventory = _raw_results(directory, expected_ids)
    metric_mismatches = 0
    pairing_rows: list[dict[str, Any]] = []
    for row in rows:
        expected = lock_by_snapshot[row["snapshot_id"]]
        checksum_match = all(
            row[name] == expected[name]
            for name in (
                "snapshot_checksum",
                "source_checksum",
                "target_checksum",
                "reference_pose_checksum",
            )
        )
        pairing_rows.append(
            {
                "backend": row["backend"],
                "checksum_match": checksum_match,
                "planned_trial_id": row["planned_trial_id"],
                "snapshot_id": row["snapshot_id"],
            }
        )
        recomputed = _recompute_update(
            row, np.asarray(expected["reference_pose_4x4"], dtype=np.float64)
        )
        for field in ("translation_update_m", "rotation_update_rad"):
            left, right = row[field], recomputed[field]
            if (left is None) != (right is None) or (
                left is not None and not math.isclose(float(left), float(right), abs_tol=1.0e-12)
            ):
                metric_mismatches += 1
    by_backend = []
    for backend in (OPEN3D_BACKEND, PCL_BACKEND):
        selected = [row for row in rows if row["backend"] == backend]
        by_backend.append(
            {
                "backend": backend,
                "failure_count": sum(row["solver_failure"] for row in selected),
                "success_count": sum(not row["solver_failure"] for row in selected),
                "trial_count": len(selected),
            }
        )
    failures = Counter(row["failure_classification"] for row in rows)
    fixture_summary = [
        {
            "backend": row["backend"],
            "condition": row["condition"],
            "failure_classification": row["failure_classification"],
            "finite_output": row["finite_output"],
            "rotation_update_rad": row["rotation_update_rad"],
            "solver_failure": row["solver_failure"],
            "translation_update_m": row["translation_update_m"],
        }
        for row in rows
    ]
    expected_failure = {
        snapshot["condition"]: set(snapshot["expected_failure_classifications"])
        for snapshot in plan["snapshots"]
    }
    classification_pass = all(
        row["failure_classification"] in expected_failure[row["condition"]]
        and (
            row["solver_failure"]
            == (row["failure_classification"] != "NONE")
        )
        for row in rows
    )
    fixture_pass = bool(
        len(rows) == 6
        and all(
            (row["finite_output"] and not row["solver_failure"])
            if row["condition"] != "FIXTURE_NO_CORRESPONDENCE"
            else row["solver_failure"]
            for row in rows
        )
    )
    events = read_attempt_events(directory / "attempt_events.ndjson")
    pairing_violations = sum(not row["checksum_match"] for row in pairing_rows)
    decision = {
        "PHASE_A_EXECUTION_CHAIN_ANALYSIS_PASS": bool(
            len(rows) == 6 and metric_mismatches == 0 and pairing_violations == 0
        ),
        "PHASE_A_EXECUTION_CHAIN_FAILURE_CLASSIFICATION_PASS": classification_pass,
        "PHASE_A_EXECUTION_CHAIN_FIXTURE_PASS": fixture_pass,
        "PHASE_A_EXECUTION_CHAIN_SCHEMA_PASS": len(rows) == 6,
        "input_checksum_pairing_violation_count": pairing_violations,
        "metric_recomputation_mismatch_count": metric_mismatches,
    }
    return {
        "attempt_event_count": len(events),
        "backend_inventory": by_backend,
        "decision": decision,
        "failure_inventory": [
            {"count": count, "failure_classification": name}
            for name, count in sorted(failures.items())
        ],
        "fixture_summary": fixture_summary,
        "input_pairing_audit": pairing_rows,
        "rotation_summary": _summary(rows, "rotation_update_rad"),
        "runtime_summary": _summary(rows, "runtime_ms"),
        "schema_version": "phase_a_execution_chain_analysis_v1",
        "snapshot_ids": sorted({row["snapshot_id"] for row in rows}),
        "translation_summary": _summary(rows, "translation_update_m"),
        "trial_count": len(rows),
        "trial_ids": sorted(row["planned_trial_id"] for row in rows),
        "trial_inventory": trial_inventory,
    }


__all__ = ["analyze_phase_a_stage1_fixture"]
