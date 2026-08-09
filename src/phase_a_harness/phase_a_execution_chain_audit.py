"""Six-trial dual-backend execution chain over fixture-only snapshots."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from .backend_phase_a_metrics import transform_update
from .open3d_backend import run_open3d_full
from .pcl_backend import frozen_parameters, run_pcl_point_to_plane
from .phase_a_attempt_events import append_attempt_event
from .phase_a_execution_chain_fixture import (
    FIXTURE_LOCK_RELATIVE,
    FIXTURE_PARAMETER_LOCK_RELATIVE,
    FORMAL_CACHE_RELATIVE,
    FixtureSnapshot,
    materialize_fixture_cache,
    validate_fixture_lock,
)
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    validate_phase_a_trial_result_strict,
)
from .phase_a_trial_result_writer import (
    atomic_write_bytes,
    result_filename,
    write_phase_a_trial_result,
)
from .phase_a_trial_resume import (
    CorruptExistingResult,
    validate_existing_trial_result_for_resume,
)


AUDIT_PROTOCOL_RELATIVE = Path(
    "configs/zero_perturbation/phase_a_execution_chain_audit_v1.yaml"
)
AUDIT_DOCUMENT_RELATIVE = Path(
    "docs/zero_perturbation_phase_a_execution_chain_audit_v1.md"
)
AUDIT_LOCK_RELATIVE = Path(
    "artifacts/current/zero_perturbation_phase_a_execution_chain_audit_v1_protocol_lock/audit_protocol_lock.json"
)
DEFAULT_PCL_CLI_RELATIVE = Path("build/pcl_point_to_plane_v3/pcl_point_to_plane_cli")


class AuditContractError(RuntimeError):
    pass


class InjectedInfrastructureInterruption(RuntimeError):
    pass


def validate_audit_protocol_lock(root: str | Path, lock_path: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    candidate = Path(lock_path).resolve()
    value = json.loads(candidate.read_text(encoding="utf-8"))
    stored = value.get("lock_payload_sha256")
    payload = {key: item for key, item in value.items() if key != "lock_payload_sha256"}
    if stored != canonical_json_sha256(payload):
        raise AuditContractError("audit protocol lock payload SHA mismatch")
    if (
        value.get("schema_version")
        != "phase_a_execution_chain_audit_protocol_lock_v1"
        or value.get("protocol_type")
        != "phase_a_execution_chain_end_to_end_audit"
        or value.get("protocol_version") != 1
    ):
        raise AuditContractError("audit protocol lock identity mismatch")
    if (
        value.get("formal_phase_a_data_access_authorized") is not False
        or value.get("formal_phase_a_backend_execution_authorized") is not False
        or value.get("formal_seed_access_authorized") is not False
        or value.get("fixture_backend_execution_authorized") is not True
    ):
        raise PermissionError("audit authorization boundary changed")
    for path_key, sha_key in (
        ("audit_protocol_path", "audit_protocol_sha256"),
        ("audit_protocol_document_path", "audit_protocol_document_sha256"),
    ):
        source = repository / str(value[path_key])
        if file_sha256(source) != value[sha_key]:
            raise AuditContractError(f"audit lock file SHA mismatch: {path_key}")
    protocol = yaml.safe_load((repository / AUDIT_PROTOCOL_RELATIVE).read_text(encoding="utf-8"))
    if protocol["protocol"]["scientific_experiment"] is not False:
        raise AuditContractError("audit was reclassified as a scientific experiment")
    return value


def implementation_manifest(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    paths = {
        "trial_schema": "schemas/phase_a_trial_result_v1.schema.json",
        "schema_validator": "src/zero_perturbation/phase_a_trial_result_schema.py",
        "writer": "src/zero_perturbation/phase_a_trial_result_writer.py",
        "resume": "src/zero_perturbation/phase_a_trial_resume.py",
        "attempt_event_schema": "src/zero_perturbation/phase_a_attempt_events.py",
        "fixture_builder": "src/zero_perturbation/phase_a_execution_chain_fixture.py",
        "fixture_parameter_lock": FIXTURE_PARAMETER_LOCK_RELATIVE.as_posix(),
        "execution_chain": "src/zero_perturbation/phase_a_execution_chain_audit.py",
        "analysis": "src/zero_perturbation/phase_a_stage1_analysis.py",
        "independent_verifier": "src/zero_perturbation/phase_a_stage1_independent_verifier.py",
        "publisher": "src/zero_perturbation/phase_a_stage1_publisher.py",
        "artifact_verifier": "src/zero_perturbation/phase_a_execution_chain_artifact_verifier.py",
        "formal_runner": "scripts/168_run_backend_phase_a.py",
        "open3d_adapter": "src/zero_perturbation/open3d_backend.py",
        "pcl_adapter": "src/zero_perturbation/pcl_backend.py",
        "pcl_cli_source": "tools/pcl_point_to_plane/pcl_point_to_plane_cli.cpp",
        "rotation_metric": "src/zero_perturbation/rotation_metrics.py",
    }
    missing = [path for path in paths.values() if not (repository / path).is_file()]
    if missing:
        raise FileNotFoundError(f"implementation manifest inputs missing: {missing}")
    pcl_binary = repository / DEFAULT_PCL_CLI_RELATIVE
    if not pcl_binary.is_file():
        raise FileNotFoundError(f"PCL CLI binary is missing: {pcl_binary}")
    manifest = {
        "files": {
            name: {"path": path, "sha256": file_sha256(repository / path)}
            for name, path in paths.items()
        },
        "pcl_cli_binary": {
            "path": DEFAULT_PCL_CLI_RELATIVE.as_posix(),
            "sha256": file_sha256(pcl_binary),
        },
        "schema_version": "phase_a_execution_chain_implementation_manifest_v1",
    }
    manifest["implementation_sha256"] = canonical_json_sha256(manifest)
    return manifest


def _common_result(
    fixture: FixtureSnapshot,
    *,
    backend: str,
    protocol_sha256: str,
    snapshot_lock_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "condition": fixture.condition,
        "implementation_sha256": implementation_sha256,
        "planned_trial_id": f"{fixture.snapshot_id}/{backend}",
        "protocol_sha256": protocol_sha256,
        "reference_pose_checksum": fixture.checksums["reference_pose_checksum"],
        "scene_variant": fixture.scene_variant,
        "schema_version": "phase_a_trial_result_v1",
        "snapshot_checksum": fixture.checksums["snapshot_checksum"],
        "snapshot_id": fixture.snapshot_id,
        "snapshot_lock_sha256": snapshot_lock_sha256,
        "source_checksum": fixture.checksums["source_checksum"],
        "target_checksum": fixture.checksums["target_checksum"],
    }


def _result_transform_fields(reference: np.ndarray, matrix: np.ndarray | None) -> dict[str, Any]:
    if matrix is None or np.asarray(matrix).shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return {
            "final_transform_4x4": None,
            "orthogonality_defect_fro": None,
            "projection_correction_fro": None,
            "raw_rotation_determinant": None,
            "raw_rotation_finite": False,
            "rotation_update_rad": None,
            "translation_update_m": None,
        }
    update = transform_update(reference, matrix)
    audit = update["rotation_audit"]
    return {
        "final_transform_4x4": np.asarray(matrix, dtype=np.float64).tolist(),
        "orthogonality_defect_fro": audit["orthogonality_defect_fro"],
        "projection_correction_fro": audit["projection_correction_fro"],
        "raw_rotation_determinant": audit["determinant"],
        "raw_rotation_finite": audit["raw_rotation_finite"],
        "rotation_update_rad": update["rotation_update_rad"],
        "translation_update_m": update["translation_update_m"],
    }


def execute_open3d_fixture(
    *,
    fixture: FixtureSnapshot,
    common: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = run_open3d_full(
            fixture.source,
            fixture.target,
            fixture.reference,
            {
                "registration_method": "point_to_plane",
                "maximum_correspondence_distance_m": float(
                    parameters["maximum_correspondence_distance_m"]
                ),
                "target_normal_estimation": dict(parameters["target_normal_estimation"]),
                "icp_convergence": dict(parameters["convergence"]),
            },
            backend_seed=0,
            input_checksum=fixture.checksums["snapshot_checksum"],
        )
        fields = _result_transform_fields(fixture.reference, result.final_pose)
        diagnostics = {
            "correspondence_set_size": int(result.correspondence_count),
            "fitness": float(result.extra["fitness"]),
            "inlier_rmse": float(result.extra["inlier_rmse"]),
        }
        finite = bool(result.finite_result and fields["final_transform_4x4"] is not None)
        if diagnostics["correspondence_set_size"] <= 0:
            classification, detail = "NO_CORRESPONDENCES", "Open3D returned zero correspondences"
        elif not finite:
            classification, detail = "NONFINITE_OUTPUT", "Open3D output was non-finite"
        elif fields["orthogonality_defect_fro"] > 1.0e-5 or abs(fields["raw_rotation_determinant"] - 1.0) > 1.0e-5:
            classification, detail = "ROTATION_MATRIX_QUALITY_FAILURE", "Open3D rotation failed quality limits"
        else:
            classification, detail = "NONE", None
    except Exception as error:
        fields = _result_transform_fields(fixture.reference, None)
        diagnostics = {"correspondence_set_size": 0, "fitness": None, "inlier_rmse": None}
        finite = False
        classification = "BACKEND_EXCEPTION"
        detail = f"{type(error).__name__}: {error}"
    payload = {
        **common,
        **fields,
        "backend_diagnostics": diagnostics,
        "failure_classification": classification,
        "failure_detail": detail,
        "finite_output": finite,
        "runtime_ms": max(0.0, (time.perf_counter() - start) * 1000.0),
        "solver_failure": classification != "NONE",
    }
    return validate_phase_a_trial_result_strict(payload)


def execute_pcl_fixture(
    *,
    fixture: FixtureSnapshot,
    common: Mapping[str, Any],
    parameters: Mapping[str, Any],
    pcl_cli: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = run_pcl_point_to_plane(
            fixture.source,
            fixture.target,
            fixture.reference,
            trial_id=common["planned_trial_id"],
            checksums=dict(fixture.checksums),
            executable=pcl_cli,
            parameters=frozen_parameters(parameters),
        )
        fields = _result_transform_fields(fixture.reference, result.final_transformation)
        diagnostics = {
            "correspondence_count": int(result.correspondence_count),
            "exit_code": int(result.cli_exit_code),
            "fitness_score": result.fitness_score,
            "has_converged_raw": bool(result.has_converged),
            "iteration_count": int(result.iteration_count),
            "pcl_cli_sha256": file_sha256(pcl_cli),
            "pcl_version": result.pcl_version,
            "source_normal_statistics": dict(result.source_normal_statistics),
            "target_normal_statistics": dict(result.target_normal_statistics),
        }
        finite = bool(result.finite_output and fields["final_transform_4x4"] is not None)
        normals_invalid = any(
            int(stat[name]) != 0
            for stat in (result.source_normal_statistics, result.target_normal_statistics)
            for name in ("zero_count", "nan_count")
        )
        if diagnostics["correspondence_count"] <= 0:
            classification, detail = "NO_CORRESPONDENCES", "PCL returned zero correspondences"
        elif normals_invalid:
            classification, detail = "INVALID_NORMALS", "PCL normal statistics contain invalid normals"
        elif not finite:
            classification, detail = "NONFINITE_OUTPUT", "PCL output was non-finite"
        elif not result.has_converged:
            classification, detail = "SCIENTIFIC_SOLVER_FAILURE", "PCL did not report convergence"
        elif fields["orthogonality_defect_fro"] > 1.0e-5 or abs(fields["raw_rotation_determinant"] - 1.0) > 1.0e-5:
            classification, detail = "ROTATION_MATRIX_QUALITY_FAILURE", "PCL rotation failed quality limits"
        else:
            classification, detail = "NONE", None
    except Exception as error:
        fields = _result_transform_fields(fixture.reference, None)
        empty_normals = {
            "finite_count": 0,
            "nan_count": 0,
            "norm_max": None,
            "norm_median": None,
            "norm_min": None,
            "zero_count": 0,
        }
        diagnostics = {
            "correspondence_count": 0,
            "exit_code": -1,
            "fitness_score": None,
            "has_converged_raw": False,
            "iteration_count": 0,
            "pcl_cli_sha256": file_sha256(pcl_cli),
            "pcl_version": "UNKNOWN",
            "source_normal_statistics": dict(empty_normals),
            "target_normal_statistics": dict(empty_normals),
        }
        finite = False
        classification = "BACKEND_EXCEPTION"
        detail = f"{type(error).__name__}: {error}"
    payload = {
        **common,
        **fields,
        "backend_diagnostics": diagnostics,
        "failure_classification": classification,
        "failure_detail": detail,
        "finite_output": finite,
        "runtime_ms": max(0.0, (time.perf_counter() - start) * 1000.0),
        "solver_failure": classification != "NONE",
    }
    return validate_phase_a_trial_result_strict(payload)


def _read_result_manifest(path: Path, run_id: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "results": {},
            "run_id": run_id,
            "schema_version": "phase_a_raw_result_manifest_v1",
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {"schema_version", "run_id", "results"} or value.get("schema_version") != "phase_a_raw_result_manifest_v1" or value.get("run_id") != run_id or type(value.get("results")) is not dict:
        raise CorruptExistingResult("raw result manifest schema mismatch")
    return value


def execute_fixture_audit_trials(
    *,
    root: str | Path,
    protocol_lock: str | Path,
    fixture_lock: str | Path,
    run_id: str,
    output_dir: str | Path,
    resume: bool = False,
    interrupt_after_completed: int | None = None,
    backend_hooks: Mapping[str, Callable[..., dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    destination = Path(output_dir).resolve()
    formal_cache = (repository / FORMAL_CACHE_RELATIVE).resolve()
    if destination == formal_cache or formal_cache in destination.parents:
        raise PermissionError("audit output cannot enter the formal Stage-0 cache")
    lock = validate_audit_protocol_lock(repository, protocol_lock)
    fixture_lock_value, fixtures = validate_fixture_lock(repository, fixture_lock)
    fixture_lock_sha = file_sha256(fixture_lock)
    implementation = implementation_manifest(repository)
    protocol_sha = lock["audit_protocol_sha256"]
    parameter_lock = json.loads(
        (repository / FIXTURE_PARAMETER_LOCK_RELATIVE).read_text(encoding="utf-8")
    )
    if parameter_lock.get("formal_seed_values_included") is not False:
        raise AuditContractError("fixture parameter lock contains formal seed values")
    for name in ("open3d_parameter_contract", "pcl_parameter_contract"):
        section = parameter_lock[name]
        if canonical_json_sha256(section["parameters"]) != section["canonical_sha256"]:
            raise AuditContractError(f"fixture parameter contract SHA mismatch: {name}")
    pcl_cli = repository / DEFAULT_PCL_CLI_RELATIVE
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError("audit output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    materialize_fixture_cache(destination / "fixture_cache", fixtures)
    results_dir = destination / "raw_results"
    events_path = destination / "attempt_events.ndjson"
    manifest_path = destination / "raw_result_manifest.json"
    manifest = _read_result_manifest(manifest_path, run_id)
    hooks = dict(backend_hooks or {})
    completed_ids: list[str] = []
    resumed_ids: list[str] = []
    backend_execution_count = 0
    for fixture in fixtures:
        for backend in (OPEN3D_BACKEND, PCL_BACKEND):
            common = _common_result(
                fixture,
                backend=backend,
                protocol_sha256=protocol_sha,
                snapshot_lock_sha256=fixture_lock_sha,
                implementation_sha256=implementation["implementation_sha256"],
            )
            trial_id = common["planned_trial_id"]
            result_path = results_dir / result_filename(trial_id)
            if result_path.exists():
                if not resume:
                    raise FileExistsError(f"trial result exists without resume: {trial_id}")
                try:
                    validate_existing_trial_result_for_resume(
                        result_path,
                        manifest_entry=manifest["results"].get(trial_id),
                        expected=common,
                    )
                except CorruptExistingResult as error:
                    append_attempt_event(
                        events_path,
                        planned_trial_id=trial_id,
                        snapshot_id=fixture.snapshot_id,
                        backend=backend,
                        event_type="REJECTED_CORRUPT_RESULT",
                        detail=str(error),
                    )
                    raise
                append_attempt_event(
                    events_path,
                    planned_trial_id=trial_id,
                    snapshot_id=fixture.snapshot_id,
                    backend=backend,
                    event_type="RESUMED",
                    detail=None,
                )
                append_attempt_event(
                    events_path,
                    planned_trial_id=trial_id,
                    snapshot_id=fixture.snapshot_id,
                    backend=backend,
                    event_type="SKIPPED_VALID_RESULT",
                    detail=None,
                )
                resumed_ids.append(trial_id)
                completed_ids.append(trial_id)
                continue
            append_attempt_event(
                events_path,
                planned_trial_id=trial_id,
                snapshot_id=fixture.snapshot_id,
                backend=backend,
                event_type="STARTED",
                detail=None,
            )
            if backend in hooks:
                result = hooks[backend](fixture=fixture, common=common)
            elif backend == OPEN3D_BACKEND:
                result = execute_open3d_fixture(
                    fixture=fixture,
                    common=common,
                    parameters=parameter_lock["open3d_parameter_contract"]["parameters"],
                )
            else:
                result = execute_pcl_fixture(
                    fixture=fixture,
                    common=common,
                    parameters=parameter_lock["pcl_parameter_contract"]["parameters"],
                    pcl_cli=pcl_cli,
                )
            result = validate_phase_a_trial_result_strict(result)
            path, sha = write_phase_a_trial_result(results_dir, result)
            manifest["results"][trial_id] = {
                "path": path.name,
                "planned_trial_id": trial_id,
                "sha256": sha,
            }
            atomic_write_bytes(manifest_path, canonical_json_bytes(manifest), replace=True)
            append_attempt_event(
                events_path,
                planned_trial_id=trial_id,
                snapshot_id=fixture.snapshot_id,
                backend=backend,
                event_type="COMPLETED",
                detail=None,
            )
            backend_execution_count += 1
            completed_ids.append(trial_id)
            if interrupt_after_completed is not None and len(completed_ids) == interrupt_after_completed:
                append_attempt_event(
                    events_path,
                    planned_trial_id=trial_id,
                    snapshot_id=fixture.snapshot_id,
                    backend=backend,
                    event_type="INFRASTRUCTURE_INTERRUPTION",
                    detail="test-only injected interruption",
                )
                raise InjectedInfrastructureInterruption("test-only injected interruption")
    result = {
        "FORMAL_PHASE_A_BACKEND_EXECUTION_COUNT": 0,
        "FORMAL_PHASE_A_SEED_ACCESS_COUNT": 0,
        "FORMAL_PHASE_A_TRIAL_RESULT_COUNT": 0,
        "FORMAL_STAGE0_CACHE_READ_COUNT": 0,
        "NATIVE_EXECUTION_COUNT": 0,
        "backend_execution_count_this_invocation": backend_execution_count,
        "completed_trial_ids": completed_ids,
        "fixture_completed_trial_count": len(manifest["results"]),
        "fixture_only": True,
        "fixture_planned_trial_count": 6,
        "fixture_snapshot_count": len(fixture_lock_value["snapshots"]),
        "implementation": implementation,
        "resume_skipped_valid_result_count": len(resumed_ids),
        "resumed_trial_ids": resumed_ids,
        "run_id": run_id,
        "schema_version": "phase_a_execution_chain_fixture_run_v1",
    }
    atomic_write_bytes(
        destination / "fixture_run_manifest.json", canonical_json_bytes(result), replace=resume
    )
    return result


__all__ = [
    "AUDIT_DOCUMENT_RELATIVE",
    "AUDIT_LOCK_RELATIVE",
    "AUDIT_PROTOCOL_RELATIVE",
    "AuditContractError",
    "DEFAULT_PCL_CLI_RELATIVE",
    "InjectedInfrastructureInterruption",
    "execute_fixture_audit_trials",
    "execute_open3d_fixture",
    "execute_pcl_fixture",
    "implementation_manifest",
    "validate_audit_protocol_lock",
]
