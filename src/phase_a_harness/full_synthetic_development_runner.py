"""Shared-backend dry-run, subset gate, and resumable 2100-trial runner."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .asset_verifier import source_runtime_import_paths
from .contracts import (
    OPEN3D_PLAN_BACKEND,
    PCL_PLAN_BACKEND,
    canonical_json_sha256,
    file_sha256,
    load_manifest,
    manifest_root,
    write_json,
)
from .full_synthetic_development_protocol import (
    MANIFEST_RELATIVE,
    NEW_CONDITIONS,
    PHASE_B_SUBSET_EVIDENCE_RELATIVE,
    PHASE_B_SUBSET_REPORT_RELATIVE,
    PROTOCOL_RELATIVE,
    assert_isolated_python_runtime,
    derive_full_synthetic_pre_subset_gate_report,
    expected_full_synthetic_manifest_payload,
    full_synthetic_protocol_payload,
    phase_b_overlap_snapshots,
    planned_new_snapshots,
    planned_trials,
    read_full_synthetic_plans,
    verify_frozen_runtime_environment,
    verify_formal_operational_git_gate,
    verify_phase_a_ideal_import,
)
from .full_synthetic_snapshot_builder import (
    FullSyntheticSourceAccessMonitor,
    read_full_synthetic_snapshot,
    validate_full_synthetic_snapshot_lock,
)
from .phase_a_attempt_events import append_attempt_event, read_attempt_events
from .phase_a_execution_chain_fixture import FixtureSnapshot
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    canonical_json_bytes,
    load_json_strict,
)
from .phase_a_trial_result_writer import atomic_write_bytes
from .phase_b_trial_result import validate_phase_b_trial_result_strict


RAW_MANIFEST_SCHEMA = "full_synthetic_development_raw_result_manifest_v1"
DRY_RUN_SCHEMA = "full_synthetic_development_dry_run_v1"
FORMAL_RUN_SCHEMA = "full_synthetic_development_formal_run_v1"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in pairs:
            if name in value:
                raise ValueError(f"duplicate JSON key in {label}: {name}")
            value[name] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def load_full_synthetic_stack(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file, manifest = load_manifest(manifest_path, require_authorized=False)
    if manifest_file.name != MANIFEST_RELATIVE.name:
        raise ValueError("Full Synthetic must use its one v1 manifest")
    root = manifest_root(manifest_file)
    candidates = sorted(
        path.resolve()
        for path in manifest_file.parent.glob("full_synthetic_development_manifest*.json")
        if path.is_file()
    )
    if candidates != [manifest_file]:
        raise ValueError("multiple or ambiguous Development manifests exist")
    authorization = manifest.get("formal_execution_authorized")
    if type(authorization) is not bool:
        raise ValueError("formal_execution_authorized must be bool")
    expected = expected_full_synthetic_manifest_payload(
        root, authorized=authorization
    )
    if manifest != expected:
        raise ValueError("Development manifest differs from the frozen payload")
    protocol = _load_json(root / PROTOCOL_RELATIVE, "Development protocol")
    if protocol != full_synthetic_protocol_payload():
        raise ValueError("Development protocol payload changed")
    runtime_environment = verify_frozen_runtime_environment(root)
    new_snapshots, new_trials, combined_snapshots, combined_trials = (
        read_full_synthetic_plans(root)
    )
    cache_root = (root / manifest["new_snapshot_cache_root"]).resolve()
    lock_path = (root / manifest["new_snapshot_lock_path"]).resolve()
    lock = validate_full_synthetic_snapshot_lock(lock_path, cache_root, new_snapshots)
    lock_by_id = {entry["snapshot_id"]: entry for entry in lock["snapshots"]}
    if len(lock_by_id) != 1050:
        raise ValueError("Development snapshot lock contains duplicate IDs")
    phase_a_import = verify_phase_a_ideal_import(root, write_report=False)
    if phase_a_import.get("PHASE_A_IDEAL_IMPORT_PASS") is not True:
        raise RuntimeError("Phase A IDEAL import is no longer valid")
    parameter_contract = _load_json(
        root / manifest["backend_parameter_contract_path"],
        "backend parameter contract",
    )
    open3d = parameter_contract["open3d"]
    pcl = parameter_contract["pcl"]
    if (
        canonical_json_sha256(open3d["parameters"])
        != manifest["open3d_parameter_sha256"]
        or canonical_json_sha256(pcl["parameters"])
        != manifest["pcl_parameter_sha256"]
    ):
        raise ValueError("backend parameter contract changed")
    return {
        "cache_root": cache_root,
        "combined_snapshots": combined_snapshots,
        "combined_trials": combined_trials,
        "lock": lock,
        "lock_by_id": lock_by_id,
        "manifest": manifest,
        "manifest_file": manifest_file,
        "new_snapshots": new_snapshots,
        "new_trials": new_trials,
        "parameters": {
            OPEN3D_PLAN_BACKEND: open3d["parameters"],
            PCL_PLAN_BACKEND: pcl["parameters"],
        },
        "pcl_cli": (root / manifest["pcl_cli_path"]).resolve(),
        "phase_a_import": phase_a_import,
        "protocol": protocol,
        "root": root,
        "runtime_environment": runtime_environment,
    }


def dry_run_full_synthetic(
    *, manifest_path: str | Path, run_id: str, output_dir: str | Path, workers: int
) -> dict[str, Any]:
    assert_isolated_python_runtime()
    monitor = FullSyntheticSourceAccessMonitor()
    monitor.install()
    stack = load_full_synthetic_stack(manifest_path)
    manifest = stack["manifest"]
    if manifest.get("formal_execution_authorized") is not False:
        raise PermissionError("Development dry-run must precede formal authorization")
    if (
        run_id != manifest["formal_run_id"]
        or int(workers) != manifest["formal_workers"]
        or Path(output_dir).resolve()
        != (stack["root"] / manifest["formal_output_dir"]).resolve()
    ):
        raise ValueError("Development dry-run invocation differs from manifest")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Development dry-run requires an empty result directory")
    for row in stack["new_snapshots"]:
        read_full_synthetic_snapshot(
            stack["cache_root"],
            row,
            expected_lock_entry=stack["lock_by_id"][row["snapshot_id"]],
            arrays=True,
        )
    backend_counts = Counter(row["backend"] for row in stack["new_trials"])
    condition_counts = Counter(row["condition"] for row in stack["new_trials"])
    subset_trials = planned_trials(phase_b_overlap_snapshots(stack["new_snapshots"]))
    published_manifest = _load_json(
        stack["root"] / stack["manifest"]["phase_b_reference_raw_manifest_path"],
        "published Phase B raw manifest",
    )
    subset_design = {
        row["planned_trial_id"] for row in subset_trials
    } == set(published_manifest.get("results", {}))
    imports = source_runtime_import_paths()
    passed = bool(
        len(stack["new_snapshots"]) == 1050
        and len(stack["new_trials"]) == 2100
        and len(stack["combined_snapshots"]) == 1260
        and len(stack["combined_trials"]) == 2520
        and backend_counts
        == Counter({OPEN3D_PLAN_BACKEND: 1050, PCL_PLAN_BACKEND: 1050})
        and condition_counts == Counter({condition: 420 for condition in NEW_CONDITIONS})
        and subset_design
        and monitor.count == 0
        and not imports
    )
    return {
        "FULL_SYNTHETIC_DRY_RUN_PASS": passed,
        "PHASE_B_SUBSET_REPRODUCTION_DESIGN_PASS": subset_design,
        "attempt_started_count": 0,
        "backend_execution_count": 0,
        "combined_snapshot_count": len(stack["combined_snapshots"]),
        "combined_trial_count": len(stack["combined_trials"]),
        "condition_trial_counts": dict(sorted(condition_counts.items())),
        "formal_rng_access_count": 0,
        "native_trial_count": 0,
        "new_snapshot_count": len(stack["new_snapshots"]),
        "new_trial_count": len(stack["new_trials"]),
        "open3d_trial_count": backend_counts[OPEN3D_PLAN_BACKEND],
        "output_dir": str(output),
        "pcl_trial_count": backend_counts[PCL_PLAN_BACKEND],
        "phase_b_subset_snapshot_count": len(
            phase_b_overlap_snapshots(stack["new_snapshots"])
        ),
        "phase_b_subset_trial_count": len(subset_trials),
        "run_id": run_id,
        "schema_version": DRY_RUN_SCHEMA,
        "snapshot_backend_pairing_mismatch_count": 0,
        "source_repository_runtime_file_read_count": monitor.count,
        "source_repository_runtime_import_count": len(imports),
        "source_repository_runtime_import_paths": imports,
        "started_event_count": 0,
        "trial_result_count": 0,
        "workers": int(workers),
    }


def _fixture(stack: Mapping[str, Any], row: Mapping[str, str]) -> FixtureSnapshot:
    item = read_full_synthetic_snapshot(
        stack["cache_root"],
        row,
        expected_lock_entry=stack["lock_by_id"][row["snapshot_id"]],
        arrays=True,
    )
    metadata = item["metadata"]
    return FixtureSnapshot(
        snapshot_id=row["snapshot_id"],
        scene_variant=row["scene_variant"],
        condition=row["condition"],
        source=np.asarray(item["source"]),
        target=np.asarray(item["target"]),
        reference=np.asarray(item["reference"]),
        expected_failure_classifications=("NONE",),
        checksums={
            "source_checksum": metadata["source_checksum"],
            "target_checksum": metadata["target_checksum"],
            "reference_pose_checksum": metadata["reference_pose_checksum"],
            "snapshot_checksum": metadata["snapshot_checksum"],
        },
    )


def _common(
    stack: Mapping[str, Any], fixture: FixtureSnapshot, row: Mapping[str, str]
) -> dict[str, Any]:
    backend = OPEN3D_BACKEND if row["backend"] == OPEN3D_PLAN_BACKEND else PCL_BACKEND
    return {
        "backend": backend,
        "condition": fixture.condition,
        "implementation_sha256": stack["manifest"]["implementation_contract_sha256"],
        "planned_trial_id": row["planned_trial_id"],
        "protocol_sha256": stack["manifest"]["scientific_protocol_sha256"],
        "reference_pose_checksum": fixture.checksums["reference_pose_checksum"],
        "scene_variant": fixture.scene_variant,
        "schema_version": "phase_a_trial_result_v1",
        "snapshot_checksum": fixture.checksums["snapshot_checksum"],
        "snapshot_id": fixture.snapshot_id,
        "snapshot_lock_sha256": stack["manifest"]["new_snapshot_lock_sha256"],
        "source_checksum": fixture.checksums["source_checksum"],
        "target_checksum": fixture.checksums["target_checksum"],
    }


def _execute_one(
    stack: Mapping[str, Any], row: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .full_synthetic_backend_execution import (
        execute_full_synthetic_open3d_fixture,
        execute_full_synthetic_pcl_fixture,
    )
    from .full_synthetic_trial_result import validate_full_synthetic_trial_result_strict

    fixture = _fixture(stack, row)
    common = _common(stack, fixture, row)
    if row["backend"] == OPEN3D_PLAN_BACKEND:
        result = execute_full_synthetic_open3d_fixture(
            fixture=fixture,
            common=common,
            parameters=stack["parameters"][OPEN3D_PLAN_BACKEND],
        )
    elif row["backend"] == PCL_PLAN_BACKEND:
        result = execute_full_synthetic_pcl_fixture(
            fixture=fixture,
            common=common,
            parameters=stack["parameters"][PCL_PLAN_BACKEND],
            pcl_cli=stack["pcl_cli"],
        )
    else:
        raise PermissionError("Native and unknown backends are forbidden")
    return common, validate_full_synthetic_trial_result_strict(result)


def _read_raw_manifest(path: Path, run_id: str) -> dict[str, Any]:
    if not path.exists():
        return {"results": {}, "run_id": run_id, "schema_version": RAW_MANIFEST_SCHEMA}
    value = _load_json(path, "Development raw result manifest")
    if (
        set(value) != {"results", "run_id", "schema_version"}
        or value["run_id"] != run_id
        or value["schema_version"] != RAW_MANIFEST_SCHEMA
        or type(value["results"]) is not dict
    ):
        raise ValueError("Development raw result manifest identity changed")
    return value


def _audit_raw_result_inventory(
    stack: Mapping[str, Any],
    results_dir: Path,
    raw_manifest: Mapping[str, Any],
    expected_rows: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    from .full_synthetic_trial_result import (
        validate_existing_full_synthetic_trial_result_for_resume,
    )
    from .phase_a_trial_result_writer import result_filename

    results = raw_manifest["results"]
    manifest_ids = set(results)
    expected_ids = set(expected_rows)
    extra_ids = manifest_ids - expected_ids
    missing_ids = expected_ids - manifest_ids
    paths: list[str] = []
    corrupt = 0
    checksum_mismatch = 0
    validated: list[dict[str, Any]] = []
    for trial_id in sorted(manifest_ids & expected_ids):
        entry = results[trial_id]
        try:
            if type(entry) is not dict or set(entry) != {
                "path",
                "planned_trial_id",
                "sha256",
            }:
                raise ValueError("raw result manifest entry schema changed")
            path_name = entry["path"]
            if (
                type(path_name) is not str
                or Path(path_name).name != path_name
                or path_name != result_filename(trial_id)
                or entry["planned_trial_id"] != trial_id
            ):
                raise ValueError("raw result manifest path/identity changed")
            paths.append(path_name)
            result_path = results_dir / path_name
            if file_sha256(result_path) != entry["sha256"]:
                checksum_mismatch += 1
                raise ValueError("raw result checksum mismatch")
            row = expected_rows[trial_id]
            fixture = _fixture(stack, row)
            payload = validate_existing_full_synthetic_trial_result_for_resume(
                result_path,
                manifest_entry=entry,
                expected=_common(stack, fixture, row),
            )
            if payload["planned_trial_id"] != trial_id:
                raise ValueError("raw result payload identity changed")
            validated.append(payload)
        except (KeyError, OSError, TypeError, ValueError, RuntimeError):
            corrupt += 1
    duplicate_path_count = len(paths) - len(set(paths))
    inventory_entries = list(results_dir.iterdir()) if results_dir.exists() else []
    actual_files = {path.name for path in inventory_entries if path.is_file()}
    invalid_entry_count = sum(not path.is_file() for path in inventory_entries)
    manifest_files = {
        entry.get("path")
        for entry in results.values()
        if type(entry) is dict and type(entry.get("path")) is str
    }
    reverse_missing = manifest_files - actual_files
    reverse_extra = actual_files - manifest_files
    reverse_mismatch = len(reverse_missing) + len(reverse_extra) + invalid_entry_count
    audit = {
        "corrupt_trial_count": corrupt,
        "duplicate_trial_count": duplicate_path_count,
        "extra_trial_count": len(extra_ids),
        "missing_trial_count": len(missing_ids),
        "raw_result_path_count": len(paths),
        "reverse_raw_result_inventory_mismatch_count": reverse_mismatch,
        "trial_result_checksum_mismatch_count": checksum_mismatch,
        "validated_payloads": validated,
    }
    if any(
        audit[name] != 0
        for name in (
            "corrupt_trial_count",
            "duplicate_trial_count",
            "extra_trial_count",
            "reverse_raw_result_inventory_mismatch_count",
            "trial_result_checksum_mismatch_count",
        )
    ):
        raise ValueError("Development raw result inventory is corrupt or ambiguous")
    return audit


def _scientific_equal(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if left is None or right is None or isinstance(left, (str, bool)) or isinstance(right, (str, bool)):
        return left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _scientific_equal(left[name], right[name], atol=atol, rtol=rtol)
            for name in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _scientific_equal(a, b, atol=atol, rtol=rtol)
            for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))
    return left == right


def verify_phase_b_trial_subset_reproduction(
    root: str | Path,
    output_root: str | Path,
    *,
    create_immutable_evidence: bool = False,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    output = Path(output_root).resolve()
    new_manifest = _load_json(output / "raw_result_manifest.json", "new raw manifest")
    old_root = repository / "results/phase_b_signal_v1"
    old_manifest = _load_json(old_root / "raw_result_manifest.json", "Phase B raw manifest")
    subset_ids = set(old_manifest["results"])
    planned_subset_ids = {
        row["planned_trial_id"]
        for row in planned_trials(phase_b_overlap_snapshots(planned_new_snapshots()))
    }
    if subset_ids != planned_subset_ids or len(subset_ids) != 84:
        raise ValueError("published Phase B raw trial IDs differ from the frozen overlap")
    full_plan_ids = {
        row["planned_trial_id"] for row in planned_trials(planned_new_snapshots())
    }
    extra = set(new_manifest["results"]) - full_plan_ids
    missing = subset_ids - set(new_manifest["results"])
    discrete_fields = (
        "backend",
        "condition",
        "failure_classification",
        "failure_detail",
        "finite_output",
        "planned_trial_id",
        "raw_rotation_finite",
        "reference_pose_checksum",
        "scene_variant",
        "snapshot_checksum",
        "snapshot_id",
        "solver_failure",
        "source_checksum",
        "target_checksum",
    )
    scientific_fields = (
        "backend_diagnostics",
        "final_transform_4x4",
        "orthogonality_defect_fro",
        "projection_correction_fro",
        "raw_rotation_determinant",
        "rotation_update_rad",
        "translation_update_m",
    )
    mismatch_rows: list[dict[str, str]] = []
    for trial_id in sorted(subset_ids & set(new_manifest["results"])):
        old_entry = old_manifest["results"][trial_id]
        new_entry = new_manifest["results"][trial_id]
        expected_entry_fields = {"path", "planned_trial_id", "sha256"}
        if (
            type(old_entry) is not dict
            or type(new_entry) is not dict
            or set(old_entry) != expected_entry_fields
            or set(new_entry) != expected_entry_fields
            or old_entry.get("planned_trial_id") != trial_id
            or new_entry.get("planned_trial_id") != trial_id
            or Path(str(old_entry.get("path"))).name != old_entry.get("path")
            or Path(str(new_entry.get("path"))).name != new_entry.get("path")
        ):
            mismatch_rows.append(
                {"field": "raw_manifest_entry", "planned_trial_id": trial_id}
            )
            continue
        old_path = old_root / "raw_results" / old_entry["path"]
        new_path = output / "raw_results" / new_entry["path"]
        if file_sha256(old_path) != old_entry["sha256"] or file_sha256(new_path) != new_entry["sha256"]:
            mismatch_rows.append({"field": "raw_result_sha256", "planned_trial_id": trial_id})
            continue
        old = validate_phase_b_trial_result_strict(load_json_strict(old_path))
        from .full_synthetic_trial_result import validate_full_synthetic_trial_result_strict

        new = validate_full_synthetic_trial_result_strict(load_json_strict(new_path))
        for field in discrete_fields:
            if old[field] != new[field]:
                mismatch_rows.append({"field": field, "planned_trial_id": trial_id})
        for field in scientific_fields:
            if not _scientific_equal(old[field], new[field], atol=1e-12, rtol=1e-12):
                mismatch_rows.append({"field": field, "planned_trial_id": trial_id})
    provisional_pass = not missing and not extra and not mismatch_rows
    evidence_path = repository / PHASE_B_SUBSET_EVIDENCE_RELATIVE
    if create_immutable_evidence:
        if set(new_manifest["results"]) != subset_ids or not provisional_pass:
            raise ValueError("immutable subset evidence requires exactly 84 passing trials")
        evidence_core: dict[str, Any] = {
            "planned_trial_count": 84,
            "results": {
                trial_id: dict(new_manifest["results"][trial_id])
                for trial_id in sorted(subset_ids)
            },
            "run_id": "full-synthetic-development-v1",
            "schema_version": "full_synthetic_phase_b_subset_evidence_v1",
        }
        evidence = {
            **evidence_core,
            "evidence_payload_sha256": canonical_json_sha256(evidence_core),
        }
        payload = canonical_json_bytes(evidence)
        if evidence_path.exists():
            if evidence_path.read_bytes() != payload:
                raise ValueError("immutable Phase B subset evidence changed")
        else:
            atomic_write_bytes(evidence_path, payload, replace=False)
    try:
        evidence = _load_json(evidence_path, "immutable Phase B subset evidence")
    except ValueError:
        if create_immutable_evidence:
            raise
        evidence = {}
    evidence_core = {
        name: value
        for name, value in evidence.items()
        if name != "evidence_payload_sha256"
    }
    evidence_results = evidence.get("results")
    evidence_valid = bool(
        evidence.get("schema_version")
        == "full_synthetic_phase_b_subset_evidence_v1"
        and evidence.get("run_id") == "full-synthetic-development-v1"
        and evidence.get("planned_trial_count") == 84
        and type(evidence_results) is dict
        and set(evidence_results) == subset_ids
        and evidence.get("evidence_payload_sha256")
        == canonical_json_sha256(evidence_core)
        and all(
            evidence_results[trial_id] == new_manifest["results"].get(trial_id)
            for trial_id in subset_ids
        )
        and all(
            type(entry) is dict
            and set(entry) == {"path", "planned_trial_id", "sha256"}
            and entry["planned_trial_id"] == trial_id
            and Path(str(entry["path"])).name == entry["path"]
            for trial_id, entry in evidence_results.items()
        )
    )
    if evidence_valid:
        for trial_id, entry in evidence_results.items():
            path = output / "raw_results" / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                evidence_valid = False
                break
    passed = provisional_pass and evidence_valid
    report = {
        "PHASE_B_SUBSET_REPRODUCTION_PASS": passed,
        "absolute_tolerance": 1e-12,
        "compared_trial_count": len(subset_ids & set(new_manifest["results"])),
        "discrete_fields_compared": list(discrete_fields),
        "excluded_descriptive_fields": ["runtime_ms"],
        "excluded_new_protocol_identity_fields": [
            "implementation_sha256",
            "protocol_sha256",
            "snapshot_lock_sha256",
        ],
        "extra_trial_count": len(extra),
        "immutable_subset_evidence_pass": evidence_valid,
        "immutable_subset_evidence_sha256": (
            file_sha256(evidence_path) if evidence_valid else None
        ),
        "mismatch_count": len(mismatch_rows),
        "mismatches": mismatch_rows,
        "missing_trial_count": len(missing),
        "relative_tolerance": 1e-12,
        "schema_version": "full_synthetic_phase_b_subset_reproduction_v1",
        "scientific_continuous_fields_compared": list(scientific_fields),
    }
    write_json(repository / PHASE_B_SUBSET_REPORT_RELATIVE, report)
    return report


def execute_full_synthetic(
    *,
    manifest_path: str | Path,
    run_id: str,
    output_dir: str | Path,
    workers: int,
    resume: bool,
    phase_b_subset_only: bool = False,
) -> dict[str, Any]:
    if not resume:
        raise PermissionError("Development execution requires explicit --resume")
    assert_isolated_python_runtime()
    monitor = FullSyntheticSourceAccessMonitor()
    monitor.install()
    stack = load_full_synthetic_stack(manifest_path)
    manifest = stack["manifest"]
    if (
        run_id != manifest["formal_run_id"]
        or int(workers) != manifest["formal_workers"]
        or Path(output_dir).resolve()
        != (stack["root"] / manifest["formal_output_dir"]).resolve()
    ):
        raise ValueError("Development execution invocation differs from manifest")
    if phase_b_subset_only:
        if (
            manifest.get("formal_execution_authorized") is not False
            or manifest.get("phase_b_subset_execution_authorized") is not True
        ):
            raise PermissionError("Phase B subset execution authorization changed")
        pre_subset = derive_full_synthetic_pre_subset_gate_report(stack["root"])
        if any(
            pre_subset.get(name) is not True
            for name in (
                "ALL_PRE_SUBSET_GATES_PASS",
                "FULL_SYNTHETIC_TEST_PASS",
                "FULL_SYNTHETIC_DRY_RUN_PASS",
                "PHASE_B_SUBSET_REPRODUCTION_DESIGN_PASS",
                "NEW_1050_SNAPSHOTS_COMPLETE",
                "SCIENTIFIC_PROTOCOL_FROZEN_PASS",
            )
        ):
            raise PermissionError("Phase B subset raw-evidence gates are incomplete")
        selected = planned_trials(phase_b_overlap_snapshots(stack["new_snapshots"]))
    else:
        if manifest.get("formal_execution_authorized") is not True:
            raise PermissionError("full Development execution is not authorized")
        verify_formal_operational_git_gate(stack["root"])
        subset_report = _load_json(
            stack["root"] / PHASE_B_SUBSET_REPORT_RELATIVE,
            "Phase B subset reproduction report",
        )
        if subset_report.get("PHASE_B_SUBSET_REPRODUCTION_PASS") is not True:
            raise PermissionError("Phase B subset reproduction gate failed")
        selected = stack["new_trials"]
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results_dir = destination / "raw_results"
    raw_manifest_path = destination / "raw_result_manifest.json"
    events_path = destination / "attempt_events.ndjson"
    raw_manifest = _read_raw_manifest(raw_manifest_path, run_id)

    from .full_synthetic_trial_result import write_full_synthetic_trial_result

    selected_ids = {row["planned_trial_id"] for row in selected}
    if phase_b_subset_only and set(raw_manifest["results"]) - selected_ids:
        raise ValueError("subset-only output already contains non-subset trials")
    selected_by_id = {row["planned_trial_id"]: row for row in selected}
    initial_inventory = _audit_raw_result_inventory(
        stack, results_dir, raw_manifest, selected_by_id
    )
    resumed = [
        payload["planned_trial_id"]
        for payload in initial_inventory["validated_payloads"]
    ]
    pending = [row for row in selected if row["planned_trial_id"] not in raw_manifest["results"]]
    for row in pending:
        append_attempt_event(
            events_path,
            planned_trial_id=row["planned_trial_id"],
            snapshot_id=row["snapshot_id"],
            backend=(OPEN3D_BACKEND if row["backend"] == OPEN3D_PLAN_BACKEND else PCL_BACKEND),
            event_type="STARTED",
            detail=None,
        )
    completed_this_invocation = 0
    with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="full-synthetic") as executor:
        future_rows = {executor.submit(_execute_one, stack, row): row for row in pending}
        for future in as_completed(future_rows):
            row = future_rows[future]
            common, payload = future.result()
            path, digest = write_full_synthetic_trial_result(results_dir, payload)
            raw_manifest["results"][row["planned_trial_id"]] = {
                "path": path.name,
                "planned_trial_id": row["planned_trial_id"],
                "sha256": digest,
            }
            atomic_write_bytes(raw_manifest_path, canonical_json_bytes(raw_manifest), replace=True)
            append_attempt_event(
                events_path,
                planned_trial_id=row["planned_trial_id"],
                snapshot_id=row["snapshot_id"],
                backend=common["backend"],
                event_type="COMPLETED",
                detail=None,
            )
            completed_this_invocation += 1
    final_inventory = _audit_raw_result_inventory(
        stack, results_dir, raw_manifest, selected_by_id
    )
    selected_payloads = final_inventory["validated_payloads"]
    required_count = 84 if phase_b_subset_only else 2100
    if len(selected_payloads) != required_count:
        raise RuntimeError("Development execution ended with an incomplete selected matrix")
    if phase_b_subset_only:
        subset_report = verify_phase_b_trial_subset_reproduction(
            stack["root"], destination, create_immutable_evidence=True
        )
        if subset_report["PHASE_B_SUBSET_REPRODUCTION_PASS"] is not True:
            raise RuntimeError("Phase B trial subset reproduction failed")
    else:
        subset_report = _load_json(
            stack["root"] / PHASE_B_SUBSET_REPORT_RELATIVE,
            "Phase B subset reproduction report",
        )
    imports = source_runtime_import_paths()
    if monitor.count or imports:
        raise PermissionError("source-repository runtime isolation failed")
    backend_counts = Counter(row["backend"] for row in selected_payloads)
    condition_counts = Counter(row["condition"] for row in selected_payloads)
    by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in selected_payloads:
        by_snapshot[payload["snapshot_id"]].append(payload)
    checksum_fields = (
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "snapshot_checksum",
    )
    pairing_mismatches = sum(
        len(rows) != 2
        or {row["backend"] for row in rows} != {OPEN3D_BACKEND, PCL_BACKEND}
        or any(rows[0][name] != rows[1][name] for name in checksum_fields)
        for rows in by_snapshot.values()
    )
    events = read_attempt_events(events_path)
    event_identity_mismatch_count = sum(
        event["planned_trial_id"] not in selected_by_id
        or event["snapshot_id"]
        != selected_by_id.get(event["planned_trial_id"], {}).get("snapshot_id")
        or event["backend"]
        != (
            OPEN3D_BACKEND
            if selected_by_id.get(event["planned_trial_id"], {}).get("backend")
            == OPEN3D_PLAN_BACKEND
            else PCL_BACKEND
        )
        for event in events
    )
    if event_identity_mismatch_count:
        raise ValueError("Development attempt-event identity inventory changed")
    output = {
        "PHASE_B_SUBSET_REPRODUCTION_PASS": subset_report[
            "PHASE_B_SUBSET_REPRODUCTION_PASS"
        ],
        "backend_exception_count": sum(
            row["failure_classification"] == "BACKEND_EXCEPTION"
            for row in selected_payloads
        ),
        "backend_execution_count_this_invocation": completed_this_invocation,
        "backend_input_checksum_mismatch_count": pairing_mismatches,
        "combined_completed_snapshot_count": (
            210 + len(by_snapshot) if not phase_b_subset_only else None
        ),
        "combined_completed_trial_count": (
            420 + len(selected_payloads) if not phase_b_subset_only else None
        ),
        "condition_trial_counts": dict(sorted(condition_counts.items())),
        "corrupt_trial_count": final_inventory["corrupt_trial_count"],
        "duplicate_trial_count": final_inventory["duplicate_trial_count"],
        "event_identity_mismatch_count": event_identity_mismatch_count,
        "extra_trial_count": final_inventory["extra_trial_count"],
        "formal_rng_access_count": 0,
        "missing_trial_count": final_inventory["missing_trial_count"],
        "native_execution_count": 0,
        "native_trial_count": 0,
        "new_completed_snapshot_count": len(by_snapshot),
        "new_completed_trial_count": len(selected_payloads),
        "nonfinite_output_count": sum(
            not row["finite_output"] for row in selected_payloads
        ),
        "open3d_trial_count": backend_counts[OPEN3D_BACKEND],
        "pcl_trial_count": backend_counts[PCL_BACKEND],
        "phase_a_ideal_imported_snapshot_count": 210,
        "phase_a_ideal_imported_trial_count": 420,
        "phase_b_subset_only": phase_b_subset_only,
        "resume_skipped_valid_result_count": len(resumed),
        "run_id": run_id,
        "schema_version": FORMAL_RUN_SCHEMA,
        "scientific_solver_failure_count": sum(
            row["solver_failure"] for row in selected_payloads
        ),
        "snapshot_backend_pairing_mismatch_count": pairing_mismatches,
        "source_repository_runtime_file_read_count": monitor.count,
        "source_repository_runtime_import_count": len(imports),
        "source_repository_runtime_import_paths": imports,
        "reverse_raw_result_inventory_mismatch_count": final_inventory[
            "reverse_raw_result_inventory_mismatch_count"
        ],
        "trial_result_checksum_mismatch_count": final_inventory[
            "trial_result_checksum_mismatch_count"
        ],
        "workers": int(workers),
    }
    write_json(destination / "run_manifest.json", output)
    return output


__all__ = [
    "DRY_RUN_SCHEMA",
    "FORMAL_RUN_SCHEMA",
    "RAW_MANIFEST_SCHEMA",
    "dry_run_full_synthetic",
    "execute_full_synthetic",
    "load_full_synthetic_stack",
    "verify_phase_b_trial_subset_reproduction",
]
