"""Seed-free 3-snapshot/6-trial standalone fixture qualification."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import file_sha256, load_manifest, manifest_root, write_json
from .fixture_publication import audit_and_publish_existing_fixture_results
from .phase_a_execution_chain_audit import execute_open3d_fixture, execute_pcl_fixture
from .phase_a_execution_chain_fixture import build_fixture_snapshots, materialize_fixture_cache
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    canonical_json_bytes,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .phase_a_trial_result_writer import atomic_write_bytes, write_phase_a_trial_result
from .phase_a_trial_resume import validate_existing_trial_result_for_resume
from .runner import SourceAccessMonitor


def qualify_fixtures(*, manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    manifest_file, manifest = load_manifest(manifest_path)
    root = manifest_root(manifest_file)
    monitor = SourceAccessMonitor()
    monitor.install()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fixtures = build_fixture_snapshots()
    materialize_fixture_cache(destination / "fixture_cache", fixtures)
    parameters = json.loads((root / "frozen_assets/fixtures/fixture_backend_parameter_lock.json").read_text(encoding="utf-8"))
    fixture_lock_sha = file_sha256(root / "frozen_assets/fixtures/fixture_snapshot_lock.json")
    protocol_sha = file_sha256(root / "frozen_assets/fixtures/fixture_plan.json")
    result_dir = destination / "raw_results"
    raw_manifest_path = destination / "raw_result_manifest.json"
    raw_manifest = {"results": {}, "run_id": "phase-a-minimal-harness-fixture-v1", "schema_version": "phase_a_raw_result_manifest_v1"}
    fresh_rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        for backend in (OPEN3D_BACKEND, PCL_BACKEND):
            common = {
                "backend": backend,
                "condition": fixture.condition,
                "implementation_sha256": manifest["manifest_payload_sha256"],
                "planned_trial_id": f"{fixture.snapshot_id}/{backend}",
                "protocol_sha256": protocol_sha,
                "reference_pose_checksum": fixture.checksums["reference_pose_checksum"],
                "scene_variant": fixture.scene_variant,
                "schema_version": "phase_a_trial_result_v1",
                "snapshot_checksum": fixture.checksums["snapshot_checksum"],
                "snapshot_id": fixture.snapshot_id,
                "snapshot_lock_sha256": fixture_lock_sha,
                "source_checksum": fixture.checksums["source_checksum"],
                "target_checksum": fixture.checksums["target_checksum"],
            }
            if backend == OPEN3D_BACKEND:
                payload = execute_open3d_fixture(
                    fixture=fixture,
                    common=common,
                    parameters=parameters["open3d_parameter_contract"]["parameters"],
                )
            else:
                payload = execute_pcl_fixture(
                    fixture=fixture,
                    common=common,
                    parameters=parameters["pcl_parameter_contract"]["parameters"],
                    pcl_cli=root / manifest["pcl_cli_path"],
                )
            path, digest = write_phase_a_trial_result(result_dir, payload)
            raw_manifest["results"][common["planned_trial_id"]] = {
                "path": path.name,
                "planned_trial_id": common["planned_trial_id"],
                "sha256": digest,
            }
            fresh_rows.append(payload)
    atomic_write_bytes(raw_manifest_path, canonical_json_bytes(raw_manifest), replace=False)

    resume_rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        for backend in (OPEN3D_BACKEND, PCL_BACKEND):
            trial_id = f"{fixture.snapshot_id}/{backend}"
            common = {
                key: value
                for key, value in fresh_rows[next(index for index, row in enumerate(fresh_rows) if row["planned_trial_id"] == trial_id)].items()
                if key in {
                    "backend", "condition", "implementation_sha256", "planned_trial_id", "protocol_sha256",
                    "reference_pose_checksum", "scene_variant", "schema_version", "snapshot_checksum",
                    "snapshot_id", "snapshot_lock_sha256", "source_checksum", "target_checksum",
                }
            }
            entry = raw_manifest["results"][trial_id]
            resume_rows.append(
                validate_existing_trial_result_for_resume(
                    result_dir / entry["path"], manifest_entry=entry, expected=common
                )
            )
    fresh_scientific = [
        {key: value for key, value in row.items() if key != "runtime_ms"} for row in fresh_rows
    ]
    resume_scientific = [
        {key: value for key, value in row.items() if key != "runtime_ms"} for row in resume_rows
    ]
    equivalence = fresh_scientific == resume_scientific
    expected = {fixture.condition: set(fixture.expected_failure_classifications) for fixture in fixtures}
    classification_pass = all(row["failure_classification"] in expected[row["condition"]] for row in fresh_rows)
    success_pass = all(
        (not row["solver_failure"] and row["finite_output"])
        if row["condition"] != "FIXTURE_NO_CORRESPONDENCE"
        else row["solver_failure"]
        for row in fresh_rows
    )
    pairing_violations = sum(
        row["source_checksum"] != fixture.checksums["source_checksum"]
        or row["target_checksum"] != fixture.checksums["target_checksum"]
        for fixture in fixtures
        for row in fresh_rows
        if row["snapshot_id"] == fixture.snapshot_id
    )
    primary = Counter((row["backend"], row["condition"], row["failure_classification"]) for row in fresh_rows)
    independent_rows = [
        validate_phase_a_trial_result_strict(load_json_strict(result_dir / raw_manifest["results"][trial_id]["path"]))
        for trial_id in sorted(raw_manifest["results"])
    ]
    independent = Counter((row["backend"], row["condition"], row["failure_classification"]) for row in independent_rows)
    difference_count = int(primary != independent)
    publication = audit_and_publish_existing_fixture_results(
        manifest_path=manifest_file,
        fixture_run_dir=destination,
        artifact_dir=destination / "publication",
    )
    passed = bool(
        len(fixtures) == 3 and len(fresh_rows) == 6 and classification_pass and success_pass
        and pairing_violations == 0 and equivalence and difference_count == 0 and monitor.count == 0
        and publication["FIXTURE_PUBLICATION_PASS"] is True
        and publication["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is True
    )
    result = {
        "FIXTURE_QUALIFICATION_PASS": passed,
        "analysis_verifier_difference_count": difference_count,
        "artifact_verifier_pass": publication["artifact_verifier_pass"],
        "backend_execution_count": 6,
        "failure_inventory": [
            {"count": count, "failure_classification": key[2], "backend": key[0], "condition": key[1]}
            for key, count in sorted(primary.items())
        ],
        "fixture_label": "FIXTURE AUDIT — NOT SCIENTIFIC DATA",
        "fixture_snapshot_count": 3,
        "fixture_trial_count": 6,
        "fresh_resume_scientific_equivalence": equivalence,
        "input_pairing_violation_count": pairing_violations,
        "publisher_pass": publication["publisher_pass"],
        "publication_artifact_dir": publication["artifact_dir"],
        "resume_backend_execution_count": 0,
        "schema_version": "phase_a_minimal_harness_fixture_qualification_v1",
        "source_repository_runtime_file_read_count": monitor.count,
    }
    write_json(destination / "fixture_qualification.json", result)
    (destination / "README.md").write_text(
        "# FIXTURE AUDIT — NOT SCIENTIFIC DATA\n\nThree seed-free snapshots and six qualification trials.\n",
        encoding="utf-8",
    )
    checksum_paths = sorted(path for path in destination.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (destination / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(path)}  {path.relative_to(destination).as_posix()}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    verified = all(
        file_sha256(destination / relative) == digest
        for digest, relative in (line.split("  ", 1) for line in (destination / "SHA256SUMS").read_text(encoding="utf-8").splitlines())
    )
    if not verified:
        raise ValueError("fixture artifact checksum verification failed")
    return result


__all__ = ["qualify_fixtures"]
