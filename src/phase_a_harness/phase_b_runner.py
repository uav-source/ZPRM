"""Dry-run and resumable 84-trial runner for the frozen Phase B signal test."""

from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .asset_verifier import source_runtime_import_paths
from .contracts import (
    OPEN3D_PLAN_BACKEND,
    PCL_PLAN_BACKEND,
    SOURCE_REPOSITORY,
    canonical_json_sha256,
    file_sha256,
    load_manifest,
    manifest_root,
    write_json,
)
from .phase_a_attempt_events import append_attempt_event
from .phase_a_execution_chain_fixture import FixtureSnapshot
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    canonical_json_bytes,
    load_json_strict,
)
from .phase_a_trial_result_writer import atomic_write_bytes
from .phase_b_generator import CONDITIONS, SCENES, SNAPSHOT_BUILDER_SHA256
from .phase_b_manifest import build_phase_b_manifest_payload
from .phase_b_snapshot_assets import (
    read_phase_b_plans,
    read_phase_b_snapshot,
    validate_phase_b_snapshot_lock,
)


PHASE_B_MANIFEST_NAME = "phase_b_signal_manifest.json"
PHASE_B_DRY_RUN_SCHEMA = "phase_b_scene_signal_dry_run_v1"
PHASE_B_FORMAL_RUN_SCHEMA = "phase_b_scene_signal_formal_run_v1"
PHASE_B_RAW_MANIFEST_SCHEMA = "phase_b_scene_signal_raw_result_manifest_v1"

_OPEN3D_PARAMETER_SHA256 = (
    "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413"
)
_PCL_PARAMETER_SHA256 = (
    "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd"
)
_TRIAL_SCHEMA_SHA256 = (
    "4d1b9da79d11b36b19350bec1c0b9eb84fc64a3699be682a4eecca684b451773"
)
_PCL_CLI_SHA256 = (
    "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff"
)


class PhaseBSourceAccessMonitor:
    """Reject every runtime file read rooted in the source Degen-LIO repository."""

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
                raise PermissionError(
                    f"source repository runtime read forbidden: {candidate}"
                )

        sys.addaudithook(audit)
        self._installed = True


def _safe_repository_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"missing Phase B manifest path: {label}")
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Phase B manifest path escaped repository: {label}")
    return candidate


def _require_file_binding(
    root: Path, manifest: Mapping[str, Any], path_field: str, sha_field: str
) -> Path:
    path = _safe_repository_path(root, manifest.get(path_field), path_field)
    expected = manifest.get(sha_field)
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Phase B frozen binding SHA mismatch: {path_field}")
    return path


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def load_phase_b_stack(
    manifest_path: str | Path, *, require_authorized: bool
) -> dict[str, Any]:
    manifest_file, manifest = load_manifest(
        manifest_path, require_authorized=require_authorized
    )
    if manifest_file.name != PHASE_B_MANIFEST_NAME:
        raise ValueError("formal Phase B must use the one frozen Phase B manifest")
    root = manifest_root(manifest_file)
    authorization = manifest.get("formal_execution_authorized")
    if type(authorization) is not bool:
        raise ValueError("Phase B formal authorization must be bool")
    expected_manifest = build_phase_b_manifest_payload(root)
    expected_manifest["formal_execution_authorized"] = authorization
    expected_manifest["manifest_payload_sha256"] = canonical_json_sha256(
        expected_manifest
    )
    if manifest != expected_manifest:
        raise ValueError("Phase B manifest is not the exact frozen payload")
    if (
        manifest.get("formal_run_id") != "phase-b-signal-v1"
        or manifest.get("formal_output_dir") != "results/phase_b_signal_v1"
        or manifest.get("formal_workers") != 2
    ):
        raise ValueError("Phase B formal run/output/worker contract changed")
    phase_b_manifests = sorted(
        item.resolve()
        for item in manifest_file.parent.glob("phase_b_signal_manifest*.json")
        if item.is_file()
    )
    if phase_b_manifests != [manifest_file]:
        raise ValueError("multiple or ambiguous Phase B manifests exist")

    protocol_path = _require_file_binding(
        root, manifest, "scientific_protocol_path", "scientific_protocol_sha256"
    )
    snapshots_path = _require_file_binding(
        root, manifest, "planned_snapshots_path", "planned_snapshots_sha256"
    )
    trials_path = _require_file_binding(
        root, manifest, "planned_trials_path", "planned_trials_sha256"
    )
    snapshot_lock_path = _require_file_binding(
        root, manifest, "snapshot_lock_path", "snapshot_lock_sha256"
    )
    parameter_contract_path = _require_file_binding(
        root,
        manifest,
        "backend_parameter_contract_path",
        "backend_parameter_contract_sha256",
    )
    trial_schema_path = _require_file_binding(
        root, manifest, "trial_schema_path", "trial_schema_sha256"
    )
    pcl_cli_path = _require_file_binding(
        root, manifest, "pcl_cli_path", "pcl_cli_sha256"
    )
    for path_field, sha_field in (
        ("analysis_path", "analysis_sha256"),
        ("artifact_verifier_path", "artifact_verifier_sha256"),
        ("generator_export_manifest_path", "generator_export_manifest_sha256"),
        (
            "generator_export_verification_path",
            "generator_export_verification_sha256",
        ),
        ("generator_reproduction_path", "generator_reproduction_sha256"),
        ("generator_wrapper_path", "generator_wrapper_sha256"),
        ("independent_verifier_path", "independent_verifier_sha256"),
        ("phase_b_backend_bridge_path", "phase_b_backend_bridge_sha256"),
        ("phase_b_snapshot_assets_path", "phase_b_snapshot_assets_sha256"),
        ("phase_b_trial_bridge_path", "phase_b_trial_bridge_sha256"),
        ("publisher_path", "publisher_sha256"),
        ("preparation_script_path", "preparation_script_sha256"),
        ("runner_path", "runner_sha256"),
        ("runner_script_path", "runner_script_sha256"),
        (
            "scientific_protocol_document_path",
            "scientific_protocol_document_sha256",
        ),
    ):
        _require_file_binding(root, manifest, path_field, sha_field)
    hard_coded_bindings = {
        "open3d_adapter_sha256": "src/phase_a_harness/open3d_backend.py",
        "pcl_adapter_sha256": "src/phase_a_harness/pcl_backend.py",
        "phase_a_trial_validator_sha256": "src/phase_a_harness/phase_a_trial_result_schema.py",
        "rotation_metric_sha256": "src/phase_a_harness/rotation_metrics.py",
        "translation_metric_sha256": "src/phase_a_harness/backend_phase_a_metrics.py",
    }
    if any(
        file_sha256(root / relative) != manifest.get(field)
        for field, relative in hard_coded_bindings.items()
    ):
        raise ValueError("Phase B implementation binding SHA mismatch")
    generator_exported_path = _safe_repository_path(
        root, manifest.get("generator_exported_path"), "generator_exported_path"
    )
    if (
        not generator_exported_path.is_file()
        or file_sha256(generator_exported_path)
        != manifest.get("generator_source_sha256")
    ):
        raise ValueError("Phase B generator source/export SHA mismatch")
    if (
        manifest.get("open3d_parameter_sha256") != _OPEN3D_PARAMETER_SHA256
        or manifest.get("pcl_parameter_sha256") != _PCL_PARAMETER_SHA256
        or manifest.get("trial_schema_sha256") != _TRIAL_SCHEMA_SHA256
        or manifest.get("pcl_cli_sha256") != _PCL_CLI_SHA256
    ):
        raise ValueError("Phase A backend/schema/CLI frozen binding changed")

    protocol = _load_json_object(protocol_path, "Phase B scientific protocol")
    if (
        protocol.get("schema_version") != "phase_b_scene_signal_protocol_v1"
        or protocol.get("planned_snapshot_count") != 42
        or protocol.get("planned_trial_count") != 84
        or tuple(protocol.get("scenes", ())) != SCENES
        or set(protocol.get("conditions", {})) != set(CONDITIONS)
        or protocol.get("allowed_backends")
        != [OPEN3D_PLAN_BACKEND, PCL_PLAN_BACKEND]
        or set(protocol.get("forbidden_backends", ()))
        != {"native_full", "native_frozen"}
    ):
        raise ValueError("Phase B scientific protocol identity changed")

    parameter_contract = _load_json_object(
        parameter_contract_path, "backend parameter contract"
    )
    open3d_contract = parameter_contract.get("open3d", {})
    pcl_contract = parameter_contract.get("pcl", {})
    if (
        open3d_contract.get("canonical_sha256") != _OPEN3D_PARAMETER_SHA256
        or pcl_contract.get("canonical_sha256") != _PCL_PARAMETER_SHA256
        or canonical_json_sha256(open3d_contract.get("parameters", {}))
        != _OPEN3D_PARAMETER_SHA256
        or canonical_json_sha256(pcl_contract.get("parameters", {}))
        != _PCL_PARAMETER_SHA256
    ):
        raise ValueError("frozen backend parameters changed")

    snapshots, trials = read_phase_b_plans(snapshots_path, trials_path)
    cache_root = _safe_repository_path(
        root, manifest.get("snapshot_cache_root"), "snapshot_cache_root"
    )
    snapshot_lock = validate_phase_b_snapshot_lock(
        snapshot_lock_path, cache_root, snapshots
    )
    lock_by_id = {
        entry["snapshot_id"]: entry for entry in snapshot_lock["snapshots"]
    }
    # The strict reader above binds all arrays, metadata, and plan identities.
    if len(lock_by_id) != 42:
        raise ValueError("Phase B snapshot lock contains duplicate identities")
    if (
        manifest.get("planned_snapshot_count") != 42
        or manifest.get("planned_trial_count") != 84
        or manifest.get("native_trial_count") != 0
        or manifest.get("snapshot_builder_sha256") != SNAPSHOT_BUILDER_SHA256
        or manifest.get("snapshot_count") not in (None, 42)
        or manifest.get("trial_count") not in (None, 84)
    ):
        raise ValueError("Phase B manifest planned counts changed")
    return {
        "cache_root": cache_root,
        "lock_by_id": lock_by_id,
        "manifest": manifest,
        "manifest_file": manifest_file,
        "parameters": {
            OPEN3D_PLAN_BACKEND: open3d_contract["parameters"],
            PCL_PLAN_BACKEND: pcl_contract["parameters"],
        },
        "pcl_cli_path": pcl_cli_path,
        "protocol": protocol,
        "root": root,
        "snapshot_lock": snapshot_lock,
        "snapshots": snapshots,
        "trials": trials,
        "trial_schema_path": trial_schema_path,
    }


def dry_run_phase_b(
    *, manifest_path: str | Path, run_id: str, output_dir: str | Path, workers: int
) -> dict[str, Any]:
    if not run_id or int(workers) < 1:
        raise ValueError("run ID and a positive worker count are required")
    if run_id != "phase-b-signal-v1":
        raise ValueError("Phase B formal run ID changed")
    monitor = PhaseBSourceAccessMonitor()
    monitor.install()
    stack = load_phase_b_stack(manifest_path, require_authorized=False)
    if stack["manifest"].get("formal_execution_authorized") is not False:
        raise PermissionError("Phase B dry-run must precede formal authorization")
    if int(workers) != int(stack["manifest"]["formal_workers"]):
        raise ValueError("Phase B dry-run workers changed")
    output = Path(output_dir).resolve()
    expected_output = (stack["root"] / stack["manifest"]["formal_output_dir"]).resolve()
    if output != expected_output:
        raise ValueError("Phase B dry-run output directory changed")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Phase B dry-run requires an empty formal result directory")
    # Re-read all 42 snapshots after the source-access monitor is active.
    for row in stack["snapshots"]:
        read_phase_b_snapshot(
            stack["cache_root"],
            row["snapshot_id"],
            expected_lock_entry=stack["lock_by_id"][row["snapshot_id"]],
            arrays=True,
        )
    backend_counts = Counter(row["backend"] for row in stack["trials"])
    condition_trial_counts = Counter(row["condition"] for row in stack["trials"])
    imports = source_runtime_import_paths()
    passed = bool(
        len(stack["snapshots"]) == 42
        and len(stack["trials"]) == 84
        and backend_counts
        == Counter({OPEN3D_PLAN_BACKEND: 42, PCL_PLAN_BACKEND: 42})
        and condition_trial_counts == Counter({name: 42 for name in CONDITIONS})
        and monitor.count == 0
        and not imports
    )
    return {
        "FORMAL_DRY_RUN_PASS": passed,
        "attempt_started_count": 0,
        "backend_execution_count": 0,
        "condition_trial_counts": dict(sorted(condition_trial_counts.items())),
        "formal_rng_access_count": 0,
        "native_trial_count": 0,
        "open3d_trial_count": backend_counts[OPEN3D_PLAN_BACKEND],
        "output_dir": str(output),
        "pcl_trial_count": backend_counts[PCL_PLAN_BACKEND],
        "planned_snapshot_count": len(stack["snapshots"]),
        "planned_trial_count": len(stack["trials"]),
        "run_id": run_id,
        "schema_version": PHASE_B_DRY_RUN_SCHEMA,
        "snapshot_verification": "PASS",
        "source_repository_runtime_file_read_count": monitor.count,
        "source_repository_runtime_import_count": len(imports),
        "source_repository_runtime_import_paths": imports,
        "started_event_count": 0,
        "trial_result_count": 0,
        "workers": int(workers),
    }


def _fixture(stack: Mapping[str, Any], snapshot_id: str) -> FixtureSnapshot:
    item = read_phase_b_snapshot(
        stack["cache_root"],
        snapshot_id,
        expected_lock_entry=stack["lock_by_id"][snapshot_id],
        arrays=True,
    )
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
            "source_checksum": str(metadata["source_checksum"]),
            "target_checksum": str(metadata["target_checksum"]),
            "reference_pose_checksum": str(metadata["reference_pose_checksum"]),
            "snapshot_checksum": str(metadata["snapshot_checksum"]),
        },
    )


def _common(
    stack: Mapping[str, Any], fixture: FixtureSnapshot, row: Mapping[str, str]
) -> dict[str, Any]:
    backend = OPEN3D_BACKEND if row["backend"] == OPEN3D_PLAN_BACKEND else PCL_BACKEND
    if row["snapshot_id"] != fixture.snapshot_id:
        raise ValueError("Phase B planned trial/snapshot pairing mismatch")
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


def _execute_one(
    stack: Mapping[str, Any], row: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Delayed imports keep dry-run entirely outside both backend adapters.
    from .phase_b_backend_execution import (
        execute_phase_b_open3d_fixture,
        execute_phase_b_pcl_fixture,
    )
    from .phase_b_trial_result import validate_phase_b_trial_result_strict

    fixture = _fixture(stack, row["snapshot_id"])
    common = _common(stack, fixture, row)
    if row["backend"] == OPEN3D_PLAN_BACKEND:
        result = execute_phase_b_open3d_fixture(
            fixture=fixture,
            common=common,
            parameters=stack["parameters"][OPEN3D_PLAN_BACKEND],
        )
    elif row["backend"] == PCL_PLAN_BACKEND:
        result = execute_phase_b_pcl_fixture(
            fixture=fixture,
            common=common,
            parameters=stack["parameters"][PCL_PLAN_BACKEND],
            pcl_cli=stack["pcl_cli_path"],
        )
    else:
        raise PermissionError(f"unauthorized Phase B backend: {row['backend']}")
    return common, validate_phase_b_trial_result_strict(result)


def _read_raw_manifest(path: Path, run_id: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "results": {},
            "run_id": run_id,
            "schema_version": PHASE_B_RAW_MANIFEST_SCHEMA,
        }
    value = _load_json_object(path, "Phase B raw result manifest")
    if (
        set(value) != {"results", "run_id", "schema_version"}
        or value.get("schema_version") != PHASE_B_RAW_MANIFEST_SCHEMA
        or value.get("run_id") != run_id
        or type(value.get("results")) is not dict
    ):
        raise ValueError("Phase B raw result manifest identity mismatch")
    return value


def execute_phase_b(
    *,
    manifest_path: str | Path,
    run_id: str,
    output_dir: str | Path,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    if not resume:
        raise PermissionError("formal Phase B execution requires explicit --resume")
    if run_id != "phase-b-signal-v1" or int(workers) != 2:
        raise ValueError("Phase B run ID/workers changed")
    monitor = PhaseBSourceAccessMonitor()
    monitor.install()
    stack = load_phase_b_stack(manifest_path, require_authorized=True)
    if run_id != stack["manifest"]["formal_run_id"] or int(workers) != int(
        stack["manifest"]["formal_workers"]
    ):
        raise ValueError("Phase B invocation differs from the frozen manifest")
    root = stack["root"]
    destination = Path(output_dir).resolve()
    expected_destination = (root / stack["manifest"]["formal_output_dir"]).resolve()
    if destination != expected_destination:
        raise ValueError("formal Phase B output directory changed")
    destination.mkdir(parents=True, exist_ok=True)
    result_dir = destination / "raw_results"
    raw_manifest_path = destination / "raw_result_manifest.json"
    events_path = destination / "attempt_events.ndjson"
    raw_manifest = _read_raw_manifest(raw_manifest_path, run_id)

    # Delayed import keeps dry-run backend-free and uses the Phase B condition bridge.
    from .phase_b_trial_result import (
        validate_existing_phase_b_trial_result_for_resume,
        validate_phase_b_trial_result_strict,
        write_phase_b_trial_result,
    )

    trial_by_id = {row["planned_trial_id"]: row for row in stack["trials"]}
    if len(trial_by_id) != 84:
        raise ValueError("Phase B planned trial identities are not unique")
    pending: list[dict[str, str]] = []
    resumed_ids: list[str] = []
    for trial_id, entry in sorted(raw_manifest["results"].items()):
        row = trial_by_id.get(trial_id)
        if row is None:
            raise ValueError(f"extra existing Phase B trial: {trial_id}")
        fixture = _fixture(stack, row["snapshot_id"])
        common = _common(stack, fixture, row)
        validate_existing_phase_b_trial_result_for_resume(
            result_dir / entry["path"], manifest_entry=entry, expected=common
        )
        resumed_ids.append(trial_id)
    for row in stack["trials"]:
        if row["planned_trial_id"] not in raw_manifest["results"]:
            pending.append(row)

    for row in pending:
        backend = OPEN3D_BACKEND if row["backend"] == OPEN3D_PLAN_BACKEND else PCL_BACKEND
        append_attempt_event(
            events_path,
            planned_trial_id=row["planned_trial_id"],
            snapshot_id=row["snapshot_id"],
            backend=backend,
            event_type="STARTED",
            detail=None,
        )

    completed_this_invocation = 0
    failed_this_invocation: list[str] = []
    with ThreadPoolExecutor(
        max_workers=int(workers), thread_name_prefix="phase-b"
    ) as executor:
        future_rows = {executor.submit(_execute_one, stack, row): row for row in pending}
        for future in as_completed(future_rows):
            row = future_rows[future]
            common, payload = future.result()
            path, digest = write_phase_b_trial_result(result_dir, payload)
            raw_manifest["results"][row["planned_trial_id"]] = {
                "path": path.name,
                "planned_trial_id": row["planned_trial_id"],
                "sha256": digest,
            }
            atomic_write_bytes(
                raw_manifest_path, canonical_json_bytes(raw_manifest), replace=True
            )
            append_attempt_event(
                events_path,
                planned_trial_id=row["planned_trial_id"],
                snapshot_id=row["snapshot_id"],
                backend=common["backend"],
                event_type="COMPLETED",
                detail=None,
            )
            completed_this_invocation += 1
            if payload["solver_failure"]:
                failed_this_invocation.append(row["planned_trial_id"])

    validated: list[dict[str, Any]] = []
    for trial_id, entry in sorted(raw_manifest["results"].items()):
        payload = validate_phase_b_trial_result_strict(
            load_json_strict(result_dir / entry["path"])
        )
        if payload["planned_trial_id"] != trial_id:
            raise ValueError("Phase B trial identity mismatch after execution")
        validated.append(payload)
    imports = source_runtime_import_paths()
    backend_counts = Counter(row["backend"] for row in validated)
    condition_counts = Counter(row["condition"] for row in validated)
    if monitor.count or imports:
        raise PermissionError("formal Phase B source-repository isolation failed")
    if (
        len(validated) != 84
        or len({row["snapshot_id"] for row in validated}) != 42
        or backend_counts != Counter({OPEN3D_BACKEND: 42, PCL_BACKEND: 42})
        or condition_counts != Counter({name: 42 for name in CONDITIONS})
    ):
        raise RuntimeError("formal Phase B execution ended with an incomplete matrix")
    output = {
        "backend_input_checksum_mismatch_count": 0,
        "backend_execution_count_this_invocation": completed_this_invocation,
        "completed_snapshot_count": len({row["snapshot_id"] for row in validated}),
        "completed_trial_count": len(validated),
        "condition_trial_counts": dict(sorted(condition_counts.items())),
        "failed_trial_ids_this_invocation": sorted(failed_this_invocation),
        "formal_rng_access_count": 0,
        "corrupt_trial_count": 0,
        "duplicate_trial_count": 0,
        "extra_trial_count": 0,
        "missing_trial_count": 84 - len(validated),
        "native_execution_count": 0,
        "native_trial_count": 0,
        "open3d_trial_count": backend_counts[OPEN3D_BACKEND],
        "pcl_trial_count": backend_counts[PCL_BACKEND],
        "resume_skipped_valid_result_count": len(resumed_ids),
        "run_id": run_id,
        "schema_version": PHASE_B_FORMAL_RUN_SCHEMA,
        "source_repository_runtime_file_read_count": monitor.count,
        "source_repository_runtime_import_count": len(imports),
        "source_repository_runtime_import_paths": imports,
        "snapshot_backend_pairing_mismatch_count": 0,
        "trial_result_checksum_mismatch_count": 0,
        "workers": int(workers),
    }
    write_json(destination / "run_manifest.json", output)
    return output


__all__ = [
    "PHASE_B_DRY_RUN_SCHEMA",
    "PHASE_B_FORMAL_RUN_SCHEMA",
    "PHASE_B_MANIFEST_NAME",
    "PhaseBSourceAccessMonitor",
    "dry_run_phase_b",
    "execute_phase_b",
    "load_phase_b_stack",
]
