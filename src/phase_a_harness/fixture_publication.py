"""Seed-free publication audit for existing 3/6 fixture results.

The public API in this module is intentionally read-only with respect to the
fixture run.  It cannot generate a fixture, instantiate an RNG, or execute a
registration backend.  Its only writes are a new caller-selected publication
artifact containing byte-for-byte evidence copies and derived audit reports.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .contracts import file_sha256, load_manifest, manifest_root
from .fixture_publication_artifact_verifier import (
    verify_fixture_publication_artifact,
)
from .phase_a_stage1_analysis import analyze_phase_a_stage1_fixture
from .phase_a_stage1_independent_verifier import (
    independently_verify_phase_a_stage1_fixture,
)
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    canonical_json_bytes,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .phase_a_trial_result_writer import atomic_write_bytes, result_filename
from .phase_a_trial_resume import validate_existing_trial_result_for_resume


FIXTURE_PLAN_RELATIVE = Path("frozen_assets/fixtures/fixture_plan.json")
FIXTURE_LOCK_RELATIVE = Path(
    "frozen_assets/fixtures/fixture_snapshot_lock.json"
)
FIXTURE_RUN_ID = "phase-a-minimal-harness-fixture-v1"
BACKENDS = (OPEN3D_BACKEND, PCL_BACKEND)


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
    trial_id: str,
    backend: str,
    plan_row: Mapping[str, Any],
    lock_row: Mapping[str, Any],
    plan_sha256: str,
    lock_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "condition": plan_row["condition"],
        "implementation_sha256": implementation_sha256,
        "planned_trial_id": trial_id,
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


def _read_existing_fixture_evidence(
    *,
    manifest: Mapping[str, Any],
    plan_path: Path,
    lock_path: Path,
    fixture_run_dir: Path,
) -> dict[str, Any]:
    plan = _strict_object(plan_path, "fixture plan")
    lock = _strict_object(lock_path, "fixture snapshot lock")
    raw_path = fixture_run_dir / "raw_result_manifest.json"
    raw = _strict_object(raw_path, "fixture raw result manifest")
    plan_sha = file_sha256(plan_path)
    lock_sha = file_sha256(lock_path)
    if (
        plan.get("fixture_only") is not True
        or plan.get("formal_phase_a") is not False
        or plan.get("random_seed_used") is not False
        or type(plan.get("snapshots")) is not list
        or len(plan["snapshots"]) != 3
        or lock.get("fixture_only") is not True
        or lock.get("formal_phase_a") is not False
        or lock.get("formal_seed_values_included") is not False
        or lock.get("random_seed_used") is not False
        or lock.get("fixture_plan_sha256") != plan_sha
        or type(lock.get("snapshots")) is not list
        or len(lock["snapshots"]) != 3
    ):
        raise ValueError("seed-free fixture plan/lock contract mismatch")
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
        or raw.get("run_id") != FIXTURE_RUN_ID
        or raw.get("schema_version") != "phase_a_raw_result_manifest_v1"
        or type(raw.get("results")) is not dict
        or set(raw["results"]) != expected_ids
    ):
        raise ValueError("fixture raw result inventory mismatch")
    if raw_path.read_bytes() != canonical_json_bytes(raw):
        raise ValueError("fixture raw result manifest is not canonical JSON")

    rows: list[dict[str, Any]] = []
    resumed_rows: list[dict[str, Any]] = []
    result_paths: dict[str, Path] = {}
    result_inventory: list[dict[str, str]] = []
    for trial_id in sorted(expected_ids):
        entry = raw["results"][trial_id]
        if (
            type(entry) is not dict
            or set(entry) != {"path", "planned_trial_id", "sha256"}
            or entry.get("planned_trial_id") != trial_id
            or entry.get("path") != result_filename(trial_id)
        ):
            raise ValueError("fixture raw result manifest entry mismatch")
        source_path = fixture_run_dir / "raw_results" / entry["path"]
        if file_sha256(source_path) != entry.get("sha256"):
            raise ValueError("fixture raw result SHA mismatch")
        row = validate_phase_a_trial_result_strict(load_json_strict(source_path))
        if source_path.read_bytes() != canonical_json_bytes(row):
            raise ValueError("fixture raw result is not canonical JSON")
        snapshot_id, backend = trial_id.rsplit("/", 1)
        if row["snapshot_id"] != snapshot_id or row["backend"] != backend:
            raise ValueError("fixture raw result identity mismatch")
        expected = _expected_common(
            trial_id=trial_id,
            backend=backend,
            plan_row=plan_by_id[snapshot_id],
            lock_row=lock_by_id[snapshot_id],
            plan_sha256=plan_sha,
            lock_sha256=lock_sha,
            implementation_sha256=str(manifest["manifest_payload_sha256"]),
        )
        resumed = validate_existing_trial_result_for_resume(
            source_path, manifest_entry=entry, expected=expected
        )
        rows.append(row)
        resumed_rows.append(resumed)
        result_paths[trial_id] = source_path
        result_inventory.append(
            {
                "path": f"evidence/raw_results/{entry['path']}",
                "planned_trial_id": trial_id,
                "sha256": entry["sha256"],
            }
        )

    fresh_scientific = [
        {name: value for name, value in row.items() if name != "runtime_ms"}
        for row in rows
    ]
    resumed_scientific = [
        {name: value for name, value in row.items() if name != "runtime_ms"}
        for row in resumed_rows
    ]
    resume_difference_count = sum(
        left != right
        for left, right in zip(fresh_scientific, resumed_scientific)
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
    if backend_counts != Counter({OPEN3D_BACKEND: 3, PCL_BACKEND: 3}):
        raise ValueError("fixture dual-backend cardinality mismatch")
    if pairing_violation_count or outcome_violation_count:
        raise ValueError("fixture outcome or input-pairing audit failed")
    if resume_difference_count:
        raise ValueError("fixture resume validator scientific equivalence failed")
    return {
        "backend_counts": backend_counts,
        "lock": lock,
        "lock_sha256": lock_sha,
        "outcome_violation_count": outcome_violation_count,
        "pairing_violation_count": pairing_violation_count,
        "plan": plan,
        "plan_sha256": plan_sha,
        "raw": raw,
        "raw_path": raw_path,
        "result_inventory": result_inventory,
        "result_paths": result_paths,
        "resume_difference_count": resume_difference_count,
        "rows": rows,
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    atomic_write_bytes(path, payload, replace=False)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(path, canonical_json_bytes(value))


def audit_and_publish_existing_fixture_results(
    *,
    manifest_path: str | Path,
    fixture_run_dir: str | Path,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Audit and publish existing strict fixture results without execution."""

    manifest_file, manifest = load_manifest(manifest_path)
    root = manifest_root(manifest_file)
    run_dir = Path(fixture_run_dir).resolve()
    destination = Path(artifact_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError("fixture run directory is missing")
    if destination == run_dir:
        raise ValueError("fixture publication directory must differ from the run")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("fixture publisher requires an empty destination")
    plan_path = root / FIXTURE_PLAN_RELATIVE
    lock_path = root / FIXTURE_LOCK_RELATIVE
    evidence = _read_existing_fixture_evidence(
        manifest=manifest,
        plan_path=plan_path,
        lock_path=lock_path,
        fixture_run_dir=run_dir,
    )

    primary = analyze_phase_a_stage1_fixture(
        run_dir=run_dir,
        fixture_plan=plan_path,
        fixture_lock=lock_path,
    )
    independent = independently_verify_phase_a_stage1_fixture(
        run_dir=run_dir,
        fixture_plan=plan_path,
        fixture_lock=lock_path,
        analysis_output=primary,
    )
    comparison_fields = independent.get("comparison_fields")
    recomputed = independent.get("recomputed")
    if type(comparison_fields) is not list or type(recomputed) is not dict:
        raise ValueError("fixture independent verifier contract mismatch")
    exact_differences = [
        name
        for name in comparison_fields
        if primary.get(name) != recomputed.get(name)
    ]
    if (
        independent.get(
            "PHASE_A_EXECUTION_CHAIN_INDEPENDENT_VERIFIER_PASS"
        )
        is not True
        or independent.get(
            "PHASE_A_EXECUTION_CHAIN_ANALYSIS_VERIFIER_MATCH"
        )
        is not True
        or independent.get("difference_count") != 0
        or independent.get("differences") != []
        or exact_differences
    ):
        raise ValueError("fixture primary/independent exact comparison failed")
    required_primary_gates = (
        "PHASE_A_EXECUTION_CHAIN_ANALYSIS_PASS",
        "PHASE_A_EXECUTION_CHAIN_FAILURE_CLASSIFICATION_PASS",
        "PHASE_A_EXECUTION_CHAIN_FIXTURE_PASS",
        "PHASE_A_EXECUTION_CHAIN_SCHEMA_PASS",
    )
    if primary.get("trial_count") != 6 or any(
        primary.get("decision", {}).get(name) is not True
        for name in required_primary_gates
    ):
        raise ValueError("fixture primary analysis gate failed")

    resume_validation = {
        "RESUME_VALIDATOR_SCIENTIFIC_EQUIVALENCE_PASS": True,
        "backend_execution_count": 0,
        "resume_validated_trial_count": 6,
        "schema_version": "phase_a_fixture_resume_validation_v1",
        "scientific_difference_count": evidence["resume_difference_count"],
        "trial_ids": sorted(evidence["result_paths"]),
    }
    source_binding = {
        "fixture_plan_sha256": evidence["plan_sha256"],
        "fixture_snapshot_lock_sha256": evidence["lock_sha256"],
        "frozen_experiment_manifest_file_sha256": file_sha256(manifest_file),
        "frozen_experiment_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "raw_result_manifest_sha256": file_sha256(evidence["raw_path"]),
        "result_inventory": evidence["result_inventory"],
        "run_id": evidence["raw"]["run_id"],
        "schema_version": "phase_a_fixture_publication_source_binding_v1",
    }
    decision = {
        "FIXTURE_DUAL_BACKEND_OUTCOME_PASS": True,
        "FIXTURE_INPUT_PAIRING_PASS": True,
        "FIXTURE_PRIMARY_INDEPENDENT_EXACT_MATCH_PASS": True,
        "FIXTURE_PUBLICATION_EVIDENCE_COMPLETE": True,
        "FIXTURE_RESUME_SCIENTIFIC_EQUIVALENCE_PASS": True,
        "analysis_verifier_difference_count": len(exact_differences),
        "backend_execution_count": 0,
        "fixture_snapshot_count": 3,
        "fixture_trial_count": 6,
        "input_pairing_violation_count": evidence["pairing_violation_count"],
        "open3d_trial_count": evidence["backend_counts"][OPEN3D_BACKEND],
        "pcl_trial_count": evidence["backend_counts"][PCL_BACKEND],
        "resume_backend_execution_count": 0,
        "resume_scientific_difference_count": evidence[
            "resume_difference_count"
        ],
        "rng_instantiation_count": 0,
        "schema_version": "phase_a_fixture_publication_decision_v1",
    }

    destination.mkdir(parents=True, exist_ok=True)
    evidence_dir = destination / "evidence"
    result_dir = evidence_dir / "raw_results"
    result_dir.mkdir(parents=True)
    for source, relative in (
        (manifest_file, "evidence/frozen_experiment_manifest.json"),
        (plan_path, "evidence/fixture_plan.json"),
        (lock_path, "evidence/fixture_snapshot_lock.json"),
        (evidence["raw_path"], "evidence/raw_result_manifest.json"),
    ):
        _write_bytes(destination / relative, source.read_bytes())
    for trial_id, source in evidence["result_paths"].items():
        entry = evidence["raw"]["results"][trial_id]
        _write_bytes(result_dir / entry["path"], source.read_bytes())
    _write_json(destination / "primary_analysis.json", primary)
    _write_json(destination / "independent_verification.json", independent)
    _write_json(destination / "resume_validation.json", resume_validation)
    _write_json(destination / "source_binding.json", source_binding)
    _write_json(destination / "final_decision.json", decision)
    report = """# Seed-Free Fixture Publication Audit

**FIXTURE AUDIT — NOT SCIENTIFIC DATA**

This artifact republishes three existing fixture snapshots and six existing
strict trial results. No registration backend or random-number generator was executed
by this publication audit. Primary analysis and the disk-first independent
verifier agree exactly, and all six results pass strict resume validation.
"""
    _write_bytes(
        destination / "fixture_publication_report.md", report.encode("utf-8")
    )
    checksum_paths = sorted(
        candidate
        for candidate in destination.rglob("*")
        if candidate.is_file()
        and candidate.name not in {"SHA256SUMS", "artifact_verification.json"}
    )
    _write_bytes(
        destination / "SHA256SUMS",
        "".join(
            f"{file_sha256(candidate)}  "
            f"{candidate.relative_to(destination).as_posix()}\n"
            for candidate in checksum_paths
        ).encode("utf-8"),
    )
    verification = verify_fixture_publication_artifact(
        destination, write_report=True
    )
    live_verification = verify_fixture_publication_artifact(
        destination, write_report=False
    )
    recorded_verification = _strict_object(
        destination / "artifact_verification.json",
        "fixture artifact verification",
    )
    if verification != live_verification or recorded_verification != live_verification:
        raise RuntimeError("fixture artifact verifier result is not stable")
    artifact_pass = bool(
        live_verification.get("FIXTURE_ARTIFACT_VERIFICATION_PASS") is True
    )
    publication_pass = bool(
        artifact_pass
        and len(exact_differences) == 0
        and evidence["pairing_violation_count"] == 0
        and evidence["outcome_violation_count"] == 0
        and evidence["resume_difference_count"] == 0
    )
    if not publication_pass:
        raise RuntimeError("fixture publication or artifact verification failed")
    return {
        "FIXTURE_ARTIFACT_VERIFICATION_PASS": artifact_pass,
        "FIXTURE_PUBLICATION_PASS": publication_pass,
        "analysis_verifier_difference_count": len(exact_differences),
        "artifact_dir": str(destination),
        "artifact_file_count": sum(
            candidate.is_file() for candidate in destination.rglob("*")
        ),
        "artifact_verifier_pass": artifact_pass,
        "backend_execution_count": 0,
        "fixture_snapshot_count": 3,
        "fixture_trial_count": 6,
        "fresh_resume_scientific_equivalence": (
            evidence["resume_difference_count"] == 0
        ),
        "input_pairing_violation_count": evidence[
            "pairing_violation_count"
        ],
        "open3d_trial_count": evidence["backend_counts"][OPEN3D_BACKEND],
        "pcl_trial_count": evidence["backend_counts"][PCL_BACKEND],
        "publisher_pass": publication_pass,
        "resume_backend_execution_count": 0,
        "rng_instantiation_count": 0,
        "schema_version": "phase_a_fixture_publication_audit_v1",
        "sha256_mismatch_count": live_verification[
            "sha256_mismatch_count"
        ],
    }


__all__ = ["audit_and_publish_existing_fixture_results"]
