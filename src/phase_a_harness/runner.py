"""Standalone dry-run and frozen 420-trial dual-backend execution engine."""

from __future__ import annotations

import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .asset_verifier import source_runtime_import_paths, verify_frozen_assets
from .contracts import (
    OPEN3D_PLAN_BACKEND,
    PCL_PLAN_BACKEND,
    SOURCE_REPOSITORY,
    load_manifest,
    manifest_root,
    write_json,
)
from .phase_a_attempt_events import append_attempt_event
from .phase_a_execution_chain_audit import execute_open3d_fixture, execute_pcl_fixture
from .phase_a_execution_chain_fixture import FixtureSnapshot
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    canonical_json_bytes,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .phase_a_trial_result_writer import atomic_write_bytes, result_filename, write_phase_a_trial_result
from .phase_a_trial_resume import validate_existing_trial_result_for_resume
from .snapshot_reader import read_snapshot


class SourceAccessMonitor:
    """Count and reject post-export reads from the source repository."""

    def __init__(self) -> None:
        self.count = 0
        self.paths: list[str] = []
        self._lock = threading.Lock()
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        source = SOURCE_REPOSITORY.resolve()

        def audit(event: str, args: tuple[Any, ...]) -> None:
            if event != "open" or not args or not isinstance(args[0], (str, bytes)):
                return
            try:
                candidate = Path(args[0]).resolve()
            except (OSError, TypeError):
                return
            if candidate == source or source in candidate.parents:
                with self._lock:
                    self.count += 1
                    self.paths.append(str(candidate))
                raise PermissionError(f"source repository runtime read forbidden: {candidate}")

        sys.addaudithook(audit)
        self._installed = True


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _stack(manifest_path: str | Path, *, require_authorized: bool) -> dict[str, Any]:
    manifest_file, manifest = load_manifest(manifest_path, require_authorized=require_authorized)
    root = manifest_root(manifest_file)
    report = verify_frozen_assets(manifest_file, write_report=False)
    protocol = json.loads((root / manifest["scientific_protocol_path"]).read_text(encoding="utf-8"))
    snapshots = _csv_rows(root / manifest["planned_snapshots_path"])
    trials = _csv_rows(root / manifest["planned_trials_path"])
    return {
        "asset_report": report,
        "manifest": manifest,
        "manifest_file": manifest_file,
        "protocol": protocol,
        "root": root,
        "snapshots": snapshots,
        "trials": trials,
    }


def dry_run_formal(
    *, manifest_path: str | Path, run_id: str, output_dir: str | Path, workers: int
) -> dict[str, Any]:
    if not run_id or int(workers) < 1:
        raise ValueError("run ID and a positive worker count are required")
    stack = _stack(manifest_path, require_authorized=False)
    monitor = SourceAccessMonitor()
    monitor.install()
    root = stack["root"]
    cache_root = root / stack["manifest"]["snapshot_cache_root"]
    for row in stack["snapshots"]:
        read_snapshot(cache_root, row["snapshot_id"], arrays=True)
    open3d_count = sum(row["backend"] == OPEN3D_PLAN_BACKEND for row in stack["trials"])
    pcl_count = sum(row["backend"] == PCL_PLAN_BACKEND for row in stack["trials"])
    native_count = len(stack["trials"]) - open3d_count - pcl_count
    imports = source_runtime_import_paths()
    passed = bool(
        len(stack["snapshots"]) == 210
        and len(stack["trials"]) == 420
        and open3d_count == pcl_count == 210
        and native_count == 0
        and monitor.count == 0
        and not imports
    )
    return {
        "FORMAL_DRY_RUN_PASS": passed,
        "attempt_started_count": 0,
        "backend_execution_count": 0,
        "formal_rng_access_count": 0,
        "native_trial_count": native_count,
        "open3d_trial_count": open3d_count,
        "output_dir": str(Path(output_dir).resolve()),
        "pcl_trial_count": pcl_count,
        "planned_snapshot_count": len(stack["snapshots"]),
        "planned_trial_count": len(stack["trials"]),
        "run_id": run_id,
        "schema_version": "phase_a_minimal_harness_dry_run_v1",
        "snapshot_verification": "PASS",
        "source_repository_runtime_file_read_count": monitor.count,
        "source_repository_runtime_import_count": len(imports),
        "trial_result_count": 0,
        "workers": int(workers),
    }


def _fixture(stack: Mapping[str, Any], snapshot_id: str) -> FixtureSnapshot:
    root = stack["root"]
    item = read_snapshot(root / stack["manifest"]["snapshot_cache_root"], snapshot_id, arrays=True)
    metadata = item["metadata"]
    return FixtureSnapshot(
        snapshot_id=snapshot_id,
        scene_variant=str(metadata["scene_variant"]),
        condition=str(metadata["condition"]),
        source=np.asarray(item["source"]),
        target=np.asarray(item["target"]),
        reference=np.asarray(item["reference"]),
        expected_failure_classifications=("NONE",),
        checksums={
            "source_checksum": str(metadata["source_raw_checksum"]),
            "target_checksum": str(metadata["target_raw_checksum"]),
            "reference_pose_checksum": str(metadata["reference_pose_raw_checksum"]),
            "snapshot_checksum": str(metadata["snapshot_checksum"]),
        },
    )


def _common(stack: Mapping[str, Any], fixture: FixtureSnapshot, row: Mapping[str, str]) -> dict[str, Any]:
    backend = OPEN3D_BACKEND if row["backend"] == OPEN3D_PLAN_BACKEND else PCL_BACKEND
    return {
        "backend": backend,
        "condition": fixture.condition,
        "implementation_sha256": str(stack["manifest"]["manifest_payload_sha256"]),
        "planned_trial_id": row["planned_trial_id"],
        "protocol_sha256": str(stack["manifest"]["scientific_protocol_sha256"]),
        "reference_pose_checksum": fixture.checksums["reference_pose_checksum"],
        "scene_variant": fixture.scene_variant,
        "schema_version": "phase_a_trial_result_v1",
        "snapshot_checksum": fixture.checksums["snapshot_checksum"],
        "snapshot_id": fixture.snapshot_id,
        "snapshot_lock_sha256": str(stack["manifest"]["snapshot_lock_sha256"]),
        "source_checksum": fixture.checksums["source_checksum"],
        "target_checksum": fixture.checksums["target_checksum"],
    }


def _execute_one(stack: Mapping[str, Any], row: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = _fixture(stack, row["snapshot_id"])
    common = _common(stack, fixture, row)
    if row["backend"] == OPEN3D_PLAN_BACKEND:
        result = execute_open3d_fixture(
            fixture=fixture,
            common=common,
            parameters=stack["protocol"]["backend_parameters"][OPEN3D_PLAN_BACKEND],
        )
    elif row["backend"] == PCL_PLAN_BACKEND:
        result = execute_pcl_fixture(
            fixture=fixture,
            common=common,
            parameters=stack["protocol"]["backend_parameters"][PCL_PLAN_BACKEND],
            pcl_cli=stack["root"] / stack["manifest"]["pcl_cli_path"],
        )
    else:
        raise PermissionError(f"unauthorized backend: {row['backend']}")
    return common, validate_phase_a_trial_result_strict(result)


def _read_raw_manifest(path: Path, run_id: str) -> dict[str, Any]:
    if not path.exists():
        return {"results": {}, "run_id": run_id, "schema_version": "phase_a_raw_result_manifest_v1"}
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {"results", "run_id", "schema_version"} or value["run_id"] != run_id:
        raise ValueError("raw result manifest identity mismatch")
    return value


def execute_formal(
    *, manifest_path: str | Path, run_id: str, output_dir: str | Path, workers: int, resume: bool
) -> dict[str, Any]:
    if not resume:
        raise PermissionError("formal execution requires explicit --resume")
    stack = _stack(manifest_path, require_authorized=True)
    monitor = SourceAccessMonitor()
    monitor.install()
    root = stack["root"]
    destination = Path(output_dir).resolve()
    if root not in destination.parents:
        raise ValueError("formal output must reside in the standalone repository")
    destination.mkdir(parents=True, exist_ok=True)
    result_dir = destination / "raw_results"
    raw_manifest_path = destination / "raw_result_manifest.json"
    events = destination / "attempt_events.ndjson"
    raw_manifest = _read_raw_manifest(raw_manifest_path, run_id)
    pending: list[dict[str, str]] = []
    resumed_ids: list[str] = []
    trial_by_id = {row["planned_trial_id"]: row for row in stack["trials"]}
    for trial_id, entry in sorted(raw_manifest["results"].items()):
        row = trial_by_id.get(trial_id)
        if row is None:
            raise ValueError(f"extra existing trial: {trial_id}")
        fixture = _fixture(stack, row["snapshot_id"])
        common = _common(stack, fixture, row)
        validate_existing_trial_result_for_resume(
            result_dir / entry["path"], manifest_entry=entry, expected=common
        )
        resumed_ids.append(trial_id)
    for row in stack["trials"]:
        if row["planned_trial_id"] not in raw_manifest["results"]:
            pending.append(row)

    for row in pending:
        backend = OPEN3D_BACKEND if row["backend"] == OPEN3D_PLAN_BACKEND else PCL_BACKEND
        append_attempt_event(
            events,
            planned_trial_id=row["planned_trial_id"],
            snapshot_id=row["snapshot_id"],
            backend=backend,
            event_type="STARTED",
            detail=None,
        )
    completed_this_invocation = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="phase-a") as executor:
        future_rows = {executor.submit(_execute_one, stack, row): row for row in pending}
        for future in as_completed(future_rows):
            row = future_rows[future]
            common, payload = future.result()
            path, digest = write_phase_a_trial_result(result_dir, payload)
            raw_manifest["results"][row["planned_trial_id"]] = {
                "path": path.name,
                "planned_trial_id": row["planned_trial_id"],
                "sha256": digest,
            }
            atomic_write_bytes(raw_manifest_path, canonical_json_bytes(raw_manifest), replace=True)
            append_attempt_event(
                events,
                planned_trial_id=row["planned_trial_id"],
                snapshot_id=row["snapshot_id"],
                backend=common["backend"],
                event_type="COMPLETED",
                detail=None,
            )
            completed_this_invocation += 1
            if payload["solver_failure"]:
                failures.append(row["planned_trial_id"])

    validated = []
    for trial_id, entry in sorted(raw_manifest["results"].items()):
        payload = validate_phase_a_trial_result_strict(load_json_strict(result_dir / entry["path"]))
        if payload["planned_trial_id"] != trial_id:
            raise ValueError("trial identity mismatch after execution")
        validated.append(payload)
    snapshot_ids = {row["snapshot_id"] for row in validated}
    imports = source_runtime_import_paths()
    output = {
        "backend_execution_count_this_invocation": completed_this_invocation,
        "completed_snapshot_count": len(snapshot_ids),
        "completed_trial_count": len(validated),
        "failed_trial_ids_this_invocation": sorted(failures),
        "formal_rng_access_count": 0,
        "native_execution_count": 0,
        "open3d_trial_count": sum(row["backend"] == OPEN3D_BACKEND for row in validated),
        "pcl_trial_count": sum(row["backend"] == PCL_BACKEND for row in validated),
        "resume_skipped_valid_result_count": len(resumed_ids),
        "run_id": run_id,
        "schema_version": "phase_a_minimal_harness_formal_run_v1",
        "source_repository_runtime_file_read_count": monitor.count,
        "source_repository_runtime_import_count": len(imports),
        "workers": int(workers),
    }
    write_json(destination / "run_manifest.json", output)
    return output


__all__ = ["SourceAccessMonitor", "dry_run_formal", "execute_formal"]

