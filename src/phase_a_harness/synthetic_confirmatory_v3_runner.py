"""External, resumable formal runner and v3 evidence I/O adapters.

Importing this module is side-effect free.  Snapshot/RNG and backend modules
are imported only after the final pre-run artifact, release tag, clean Git,
immutable run-lock, and single-writer lease have all passed.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from copy import deepcopy
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_context import ExecutionContext, ExecutionContextError, ExecutionMode
from .synthetic_confirmatory_v3_prerun import (
    EXPECTED_BACKEND_COUNTS,
    EXPECTED_SNAPSHOT_COUNT,
    EXPECTED_TRIAL_COUNT,
    FORMAL_RUNTIME_ROOT,
    V3ContractError,
    load_v3_plan_rows,
    strict_json_object,
    validate_v3_frozen_contract,
    validate_v3_postrun_entry,
)
from .trial_snapshot_bridge import (
    CanonicalSnapshotIndex,
    TrialSnapshotBindings,
    TrialSnapshotBridgeError,
    bind_trial_to_snapshot,
    build_canonical_snapshot_index,
    build_trial_snapshot_bindings,
)


RAW_RESULT_MANIFEST_SCHEMA = "synthetic_confirmatory_raw_result_manifest_v3"
FORMAL_RUN_SCHEMA = "synthetic_confirmatory_formal_run_v3"
PRIMARY_ANALYSIS_SCHEMA = "synthetic_confirmatory_primary_analysis_v3"
INDEPENDENT_SCHEMA = "synthetic_confirmatory_independent_verification_v3"
FORMAL_ARTIFACT_SCHEMA = "synthetic_confirmatory_formal_artifact_v3"
V3_FINAL_DECISION_SCHEMA = "synthetic_confirmatory_v3_final_decision_v1"

_FORMAL_RESULT_ROUTE = "FORMAL_CONDITION_ROUTING_V1"
_QUALIFICATION_RESULT_ROUTE = "QUALIFICATION_PHASE_A_ROUTING_V1"
_NO_BACKEND_DISPATCH = object()

_DECISION_GATE_FIELDS = (
    "CONFIRMATORY_EVIDENCE_INTEGRITY_PASS",
    "H1_IDEAL_CONTROL_PASS",
    "H2_LONG_CORRIDOR_SCENE_EFFECT_PASS",
    "H3_CROSS_BACKEND_SCENE_RANKING_PASS",
    "H4_REASSOCIATION_MECHANISM_PASS",
    "H5_FROZEN_MODEL_B_INCREMENTAL_VALUE_PASS",
    "H6_FULL_NOISE_SYSTEMATIC_OFFSET_PASS",
)
_GENERIC_TO_V3_DECISION_FIELDS = {
    "SYNTHETIC_CONFIRMATORY_EXECUTED": "SYNTHETIC_CONFIRMATORY_V3_EXECUTED",
    "SYNTHETIC_CONFIRMATORY_COMPLETE": "SYNTHETIC_CONFIRMATORY_V3_COMPLETE",
    "SYNTHETIC_CONFIRMATORY_PASS": "SYNTHETIC_CONFIRMATORY_V3_PASS",
    "CONFIRMATORY_RUN_AUTHORIZED": "CONFIRMATORY_V3_RUN_AUTHORIZED",
    "REAL_DATA_RUN_AUTHORIZED": "REAL_DATA_RUN_AUTHORIZED",
    "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": (
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED"
    ),
}
_GENERIC_DECISION_FIELDS = frozenset(
    (*_DECISION_GATE_FIELDS, *_GENERIC_TO_V3_DECISION_FIELDS)
)
_V3_DECISION_FIELDS = frozenset(
    (
        "schema_version",
        *_DECISION_GATE_FIELDS,
        *_GENERIC_TO_V3_DECISION_FIELDS.values(),
    )
)


def _adapt_v3_final_decision(value: Mapping[str, Any]) -> dict[str, Any]:
    """Version only the decision envelope; never recompute a scientific gate."""

    if type(value) is not dict or set(value) != _GENERIC_DECISION_FIELDS:
        raise V3ContractError("generic scientific final decision schema changed")
    if any(type(value[name]) is not bool for name in _GENERIC_DECISION_FIELDS):
        raise V3ContractError("generic scientific final decision is not boolean")
    if any(
        value[name] is not False
        for name in (
            "CONFIRMATORY_RUN_AUTHORIZED",
            "REAL_DATA_RUN_AUTHORIZED",
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED",
        )
    ):
        raise V3ContractError("generic scientific authorization boundary changed")
    result = {"schema_version": V3_FINAL_DECISION_SCHEMA}
    result.update({name: value[name] for name in _DECISION_GATE_FIELDS})
    result.update(
        {
            v3_name: value[generic_name]
            for generic_name, v3_name in _GENERIC_TO_V3_DECISION_FIELDS.items()
        }
    )
    return result


def _validate_v3_final_decision(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value) != _V3_DECISION_FIELDS
        or value.get("schema_version") != V3_FINAL_DECISION_SCHEMA
        or any(
            type(value[name]) is not bool
            for name in _V3_DECISION_FIELDS - {"schema_version"}
        )
        or value.get("CONFIRMATORY_V3_RUN_AUTHORIZED") is not False
        or value.get("REAL_DATA_RUN_AUTHORIZED") is not False
        or value.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is not False
    ):
        raise V3ContractError("v3 final decision schema or boundary is invalid")
    return dict(value)


def _genericize_v3_final_decision(value: Mapping[str, Any]) -> dict[str, Any]:
    decision = _validate_v3_final_decision(value)
    result = {name: decision[name] for name in _DECISION_GATE_FIELDS}
    result.update(
        {
            generic_name: decision[v3_name]
            for generic_name, v3_name in _GENERIC_TO_V3_DECISION_FIELDS.items()
        }
    )
    return result


def _adapt_v3_analysis(
    value: Mapping[str, Any], *, independent: bool
) -> dict[str, Any]:
    result = deepcopy(dict(value))
    decision = _adapt_v3_final_decision(result.get("final_decision"))
    result["final_decision"] = decision
    if independent:
        projection = result.get("verification_projection")
        if type(projection) is not dict:
            raise V3ContractError("independent scientific projection is absent")
        projection["final_decision"] = deepcopy(decision)
    return result


def _genericize_v3_analysis(
    value: Mapping[str, Any], *, independent: bool
) -> dict[str, Any]:
    result = deepcopy(dict(value))
    decision = _genericize_v3_final_decision(result.get("final_decision"))
    result["final_decision"] = decision
    if independent:
        projection = result.get("verification_projection")
        if type(projection) is not dict:
            raise V3ContractError("independent v3 projection is absent")
        projection["final_decision"] = deepcopy(decision)
    return result


def _bound_path(
    repository: Path,
    manifest: Mapping[str, Any],
    name: str,
) -> Path:
    bound = manifest.get("bound_files")
    entry = bound.get(name) if type(bound) is dict else None
    if type(entry) is not dict or not isinstance(entry.get("path"), str):
        raise V3ContractError(f"v3 bound file is missing: {name}")
    candidate = (repository / entry["path"]).resolve()
    if candidate != repository and repository not in candidate.parents:
        raise V3ContractError(f"v3 bound file escapes repository: {name}")
    if not candidate.is_file():
        raise FileNotFoundError(f"v3 bound file is absent: {candidate}")
    return candidate


def _runtime_paths(manifest: Mapping[str, Any]) -> dict[str, Path]:
    names = {
        "analysis": "analysis_path",
        "artifact_staging": "artifact_staging_path",
        "backend_temporary": "backend_temporary_path",
        "event_log": "event_log_path",
        "publisher_staging": "publisher_staging_path",
        "raw_results": "raw_results_path",
        "snapshot_cache": "snapshot_cache_path",
        "snapshot_lock": "snapshot_lock_path",
        "temporary_inventory": "temporary_inventory_path",
        "verification": "verification_path",
    }
    paths = {name: Path(str(manifest[field])) for name, field in names.items()}
    paths.update(
        {
            "raw_manifest": FORMAL_RUNTIME_ROOT / "raw_result_manifest.json",
            "formal_command_log": FORMAL_RUNTIME_ROOT / "formal_command.log",
            "formal_command_sha256": (
                FORMAL_RUNTIME_ROOT / "formal_command.log.sha256"
            ),
            "run_lock": FORMAL_RUNTIME_ROOT / "immutable_run_lock.json",
            "run_manifest": FORMAL_RUNTIME_ROOT / "run_manifest.json",
        }
    )
    if any(
        not path.is_absolute()
        or (path != FORMAL_RUNTIME_ROOT and FORMAL_RUNTIME_ROOT not in path.parents)
        for path in paths.values()
    ):
        raise V3ContractError("v3 runtime path escaped the formal root")
    return paths


def _parameter_stack(
    repository: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], Path]:
    parameter_path = _bound_path(repository, manifest, "backend_parameter_contract")
    parameter_contract = strict_json_object(parameter_path)
    open3d = parameter_contract.get("open3d")
    pcl = parameter_contract.get("pcl")
    bindings = manifest.get("backend_bindings")
    if not all(type(value) is dict for value in (open3d, pcl, bindings)):
        raise V3ContractError("v3 backend parameter contract is malformed")
    if (
        open3d.get("canonical_sha256")
        != bindings.get("open3d", {}).get("parameter_sha256")
        or pcl.get("canonical_sha256")
        != bindings.get("pcl", {}).get("parameter_sha256")
    ):
        raise V3ContractError("v3 backend parameter binding mismatch")
    pcl_cli = _bound_path(repository, manifest, "pcl_cli")
    if bindings.get("pcl", {}).get("cli_sha256") != _file_sha256(pcl_cli):
        raise V3ContractError("v3 PCL CLI binding mismatch")
    return {
        "open3d_point_to_plane": open3d["parameters"],
        "pcl_point_to_plane": pcl["parameters"],
    }, pcl_cli


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_error(
    classification: str,
    *,
    field: str,
    actual: Any,
    expected: Any,
    detail: str,
) -> None:
    raise ExecutionContextError(
        classification,
        field=field,
        actual=actual,
        expected=expected,
        detail=detail,
    )


def _require_execution_context(stack: Mapping[str, Any]) -> ExecutionContext:
    context = stack.get("execution_context")
    if type(context) is not ExecutionContext:
        _context_error(
            "EXECUTION_CONTEXT_CONTRACT_MISMATCH",
            field="stack.execution_context",
            actual=type(context).__name__,
            expected="ExecutionContext",
            detail="runner stack has no explicit execution context",
        )
    return context


def _require_execution_bindings(
    stack: Mapping[str, Any],
) -> tuple[ExecutionContext, CanonicalSnapshotIndex, TrialSnapshotBindings]:
    context = _require_execution_context(stack)
    index = stack.get("canonical_snapshot_index")
    if type(index) is not CanonicalSnapshotIndex:
        _context_error(
            "EXECUTION_CONTEXT_PLAN_MISMATCH",
            field="stack.canonical_snapshot_index",
            actual=type(index).__name__,
            expected="CanonicalSnapshotIndex",
            detail="runner stack has no canonical snapshot index",
        )
    if index.execution_context is not context:
        _context_error(
            "EXECUTION_CONTEXT_PLAN_MISMATCH",
            field="canonical_snapshot_index.execution_context",
            actual=index.execution_context.report(),
            expected=context.report(),
            detail="canonical snapshot index belongs to another context",
        )
    bindings = stack.get("trial_snapshot_bindings")
    if type(bindings) is not TrialSnapshotBindings:
        _context_error(
            "EXECUTION_CONTEXT_PLAN_MISMATCH",
            field="stack.trial_snapshot_bindings",
            actual=type(bindings).__name__,
            expected="TrialSnapshotBindings",
            detail="runner stack has no fail-fast trial/snapshot bindings",
        )
    return context, index, bindings


def _validate_result_route_context(context: ExecutionContext) -> str:
    route = context.contract.result_route
    expected_mode = {
        _FORMAL_RESULT_ROUTE: ExecutionMode.FORMAL,
        _QUALIFICATION_RESULT_ROUTE: ExecutionMode.QUALIFICATION,
    }.get(route)
    if expected_mode is None or context.mode is not expected_mode:
        _context_error(
            "EXECUTION_CONTEXT_CONTRACT_MISMATCH",
            field="contract.result_route",
            actual={"mode": context.mode.value, "result_route": route},
            expected={
                ExecutionMode.FORMAL.value: _FORMAL_RESULT_ROUTE,
                ExecutionMode.QUALIFICATION.value: _QUALIFICATION_RESULT_ROUTE,
            },
            detail="result route is not authorized for the explicit execution mode",
        )
    return route


def _result_family(context: ExecutionContext, condition: Any) -> str:
    route = _validate_result_route_context(context)
    if type(condition) is not str or condition not in context.allowed_conditions:
        _context_error(
            "EXECUTION_CONTEXT_CONDITION_POLICY_MISMATCH",
            field="result.condition",
            actual=condition,
            expected=context.allowed_conditions,
            detail="result routing condition is outside the explicit context",
        )
    if route == _QUALIFICATION_RESULT_ROUTE or condition == "IDEAL_MATCHED":
        return "PHASE_A"
    return "FULL_SYNTHETIC"


def load_v3_execution_stack(
    *,
    repository: str | Path,
    manifest_path: str | Path,
    run_id: str,
    workers: int,
    require_authorized: bool,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    validated = validate_v3_frozen_contract(
        repository=root,
        manifest_path=manifest_path,
        run_id=run_id,
        runtime_root=FORMAL_RUNTIME_ROOT,
        workers=workers,
        require_authorized=require_authorized,
        require_release_tag=require_authorized,
        require_runtime_absent=False,
        git_checkpoint=(
            "V3_FORMAL_RUNNER_ENTRY_GIT_GATE"
            if require_authorized
            else "V3_FORMAL_METADATA_ENTRY_GIT_GATE"
        ),
    )
    manifest = validated["manifest"]
    snapshots, trials = load_v3_plan_rows(
        validated["planned_snapshots_path"], validated["planned_trials_path"]
    )
    if len(snapshots) != EXPECTED_SNAPSHOT_COUNT or len(trials) != EXPECTED_TRIAL_COUNT:
        raise V3ContractError("v3 execution plan cardinality changed")
    from .synthetic_confirmatory_v3_contract import formal_execution_context

    execution_context = formal_execution_context()
    runtime_paths = _runtime_paths(manifest)
    expected_plan_id = (
        "synthetic-confirmatory-v3-plan:"
        f"{validated['planned_snapshots_sha256']}:"
        f"{validated['planned_trials_sha256']}"
    )
    if execution_context.mode is not ExecutionMode.FORMAL:
        _context_error(
            "EXECUTION_CONTEXT_MODE_MISMATCH",
            field="execution_context.mode",
            actual=execution_context.mode,
            expected=ExecutionMode.FORMAL,
            detail="formal runner received a non-formal execution context",
        )
    if execution_context.plan_id != expected_plan_id:
        _context_error(
            "EXECUTION_CONTEXT_PLAN_MISMATCH",
            field="execution_context.plan_id",
            actual=execution_context.plan_id,
            expected=expected_plan_id,
            detail="formal execution context is not bound to the authenticated plans",
        )
    if (
        execution_context.runtime_root != FORMAL_RUNTIME_ROOT
        or execution_context.cache_root != runtime_paths["snapshot_cache"]
    ):
        _context_error(
            "EXECUTION_CONTEXT_RUNTIME_MISMATCH",
            field="execution_context.runtime_paths",
            actual={
                "runtime_root": execution_context.runtime_root,
                "snapshot_cache": execution_context.cache_root,
            },
            expected={
                "runtime_root": FORMAL_RUNTIME_ROOT,
                "snapshot_cache": runtime_paths["snapshot_cache"],
            },
            detail="formal execution context differs from the authenticated runtime",
        )
    if _validate_result_route_context(execution_context) != _FORMAL_RESULT_ROUTE:
        raise AssertionError("unreachable formal result route")
    canonical_snapshot_index = build_canonical_snapshot_index(
        snapshots, execution_context=execution_context
    )
    trial_snapshot_bindings = build_trial_snapshot_bindings(
        canonical_snapshot_index, trials
    )
    parameters, pcl_cli = _parameter_stack(root, manifest)
    return {
        "canonical_snapshot_index": canonical_snapshot_index,
        "execution_context": execution_context,
        "manifest": manifest,
        "manifest_path": Path(validated["manifest_path"]),
        "manifest_sha256": validated["manifest_sha256"],
        "parameters": parameters,
        "pcl_cli": pcl_cli,
        "repository": root,
        "runtime_paths": runtime_paths,
        "snapshots": snapshots,
        "trials": trials,
        "trial_snapshot_bindings": trial_snapshot_bindings,
        "validated_contract": validated,
    }


def _run_contract(
    stack: Mapping[str, Any], *, run_id: str, workers: int
) -> dict[str, Any]:
    manifest = stack["manifest"]
    validated = stack["validated_contract"]
    bound_files = manifest.get("bound_files")
    bootstrap = manifest.get("formal_bootstrap_contract")
    if type(bound_files) is not dict or type(bootstrap) is not dict:
        raise V3ContractError("v3 bootstrap implementation binding is absent")
    implementation_binding: dict[str, str] = {}
    for name, entry in sorted(bound_files.items()):
        if (
            type(name) is not str
            or type(entry) is not dict
            or type(entry.get("sha256")) is not str
        ):
            raise V3ContractError("v3 bound implementation inventory is malformed")
        implementation_binding[name] = entry["sha256"]
    profile = bound_files.get("execution_profile")
    if type(profile) is not dict or type(profile.get("sha256")) is not str:
        raise V3ContractError("v3 formal execution profile binding is absent")
    runtime_paths = {
        name: str(path) for name, path in sorted(stack["runtime_paths"].items())
    }
    return {
        "schema_version": (
            "synthetic_confirmatory_v3_bootstrap_repair_immutable_run_contract_v1"
        ),
        "backend_parameter_contract_sha256": manifest["bound_files"]
        ["backend_parameter_contract"]["sha256"],
        "branch": manifest["expected_branch"],
        "creation_identity": {
            "implementation_revision": bootstrap["implementation_revision"],
            "state_machine_schema": bootstrap["state_machine_schema"],
        },
        "creation_mode": "FRESH_FROM_BOOTSTRAP_ONLY",
        "expected_commit": validated["git_identity"]["commit"],
        "expected_branch": manifest["expected_branch"],
        "expected_release_tag": manifest["expected_release_tag"],
        "formal_execution_profile_sha256": profile["sha256"],
        "gate_contract_sha256": manifest["gate_contract_sha256"],
        "implementation_binding": implementation_binding,
        "manifest_path": str(stack["manifest_path"]),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "manifest_sha256": stack["manifest_sha256"],
        "pcl_cli_sha256": manifest["bound_files"]["pcl_cli"]["sha256"],
        "planned_snapshot_identity_sha256": validated["plan_audit"]
        ["planned_snapshot_identity_sha256"],
        "planned_snapshots_sha256": validated["planned_snapshots_sha256"],
        "planned_trial_identity_sha256": validated["plan_audit"]
        ["planned_trial_identity_sha256"],
        "planned_trials_sha256": validated["planned_trials_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "run_id": run_id,
        "runtime_root": str(FORMAL_RUNTIME_ROOT),
        "runtime_paths": runtime_paths,
        "seed_schedule_sha256": manifest["seed_schedule_sha256"],
        "tag": manifest["expected_release_tag"],
        "workers": workers,
    }


def _result_filename(planned_trial_id: str) -> str:
    import hashlib

    if not isinstance(planned_trial_id, str) or not planned_trial_id:
        raise ValueError("planned trial ID must be nonempty")
    return hashlib.sha256(planned_trial_id.encode("utf-8")).hexdigest() + ".json"


def _empty_raw_manifest(run_id: str, contract_sha256: str) -> dict[str, Any]:
    return {
        "contract_sha256": contract_sha256,
        "results": {},
        "run_id": run_id,
        "schema_version": RAW_RESULT_MANIFEST_SCHEMA,
    }


def _read_raw_manifest(
    path: Path, *, run_id: str, contract_sha256: str
) -> dict[str, Any]:
    from .runtime_lifecycle_io import read_canonical_json

    if not path.exists():
        return _empty_raw_manifest(run_id, contract_sha256)
    value = read_canonical_json(path)
    if (
        type(value) is not dict
        or set(value) != {"contract_sha256", "results", "run_id", "schema_version"}
        or value["contract_sha256"] != contract_sha256
        or value["run_id"] != run_id
        or value["schema_version"] != RAW_RESULT_MANIFEST_SCHEMA
        or type(value["results"]) is not dict
    ):
        raise V3ContractError("v3 raw result manifest identity changed")
    return value


def _bound_snapshot_row(
    stack: Mapping[str, Any], trial_row: Mapping[str, Any]
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    _context, index, bindings = _require_execution_bindings(stack)
    trial, snapshot = bind_trial_to_snapshot(index, trial_row)
    trial_id = trial["planned_trial_id"]
    bound = bindings.rows.get(trial_id)
    if bound is None:
        raise TrialSnapshotBridgeError(
            "SNAPSHOT_PLAN_ID_NOT_FOUND",
            field="planned_trial_id",
            actual=trial_id,
            expected="one fail-fast trial/snapshot binding",
            detail="trial was not included in the pre-execution binding",
        )
    if bound is not snapshot:
        raise TrialSnapshotBridgeError(
            "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH",
            field="canonical_snapshot_binding",
            actual=dict(bound),
            expected=dict(snapshot),
            detail="trial binding is not the immutable canonical snapshot row",
        )
    return trial, snapshot


def _read_authenticated_snapshot(
    stack: Mapping[str, Any], snapshot_row: Mapping[str, Any]
) -> dict[str, Any]:
    from .synthetic_confirmatory_v3_snapshot_builder import read_v3_snapshot

    context = _require_execution_context(stack)
    snapshot_id = snapshot_row["planned_snapshot_id"]
    lock_by_id = stack.get("lock_by_id")
    if not isinstance(lock_by_id, Mapping) or snapshot_id not in lock_by_id:
        raise TrialSnapshotBridgeError(
            "SNAPSHOT_AUTHENTICATION_FAILURE",
            field="snapshot_lock_entry",
            actual=snapshot_id,
            expected="one authenticated snapshot lock entry",
            detail="canonical snapshot has no lock entry",
        )
    try:
        return read_v3_snapshot(
            stack["runtime_paths"]["snapshot_cache"],
            snapshot_row,
            execution_context=context,
            expected_lock_entry=lock_by_id[snapshot_id],
            arrays=True,
        )
    except (ExecutionContextError, TrialSnapshotBridgeError):
        raise
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise TrialSnapshotBridgeError(
            "SNAPSHOT_AUTHENTICATION_FAILURE",
            field="snapshot_payload",
            actual={
                "exception_type": type(error).__name__,
                "snapshot_id": snapshot_id,
            },
            expected="reader-authenticated snapshot payload",
            detail="canonical snapshot payload or lock authentication failed",
        ) from error


def _fixture(stack: Mapping[str, Any], row: Mapping[str, Any]) -> Any:
    import numpy as np

    from .phase_a_execution_chain_fixture import FixtureSnapshot

    _trial, snapshot_row = _bound_snapshot_row(stack, row)
    snapshot_id = snapshot_row["planned_snapshot_id"]
    item = _read_authenticated_snapshot(stack, snapshot_row)
    metadata = item["metadata"]
    return FixtureSnapshot(
        snapshot_id=snapshot_id,
        scene_variant=snapshot_row["scene_variant"],
        condition=snapshot_row["condition"],
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
    stack: Mapping[str, Any], fixture: Any, row: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = stack["manifest"]
    return {
        "backend": row["backend"],
        "condition": fixture.condition,
        "implementation_sha256": manifest["manifest_payload_sha256"],
        "planned_trial_id": row["planned_trial_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "reference_pose_checksum": fixture.checksums["reference_pose_checksum"],
        "scene_variant": fixture.scene_variant,
        "schema_version": "phase_a_trial_result_v1",
        "snapshot_checksum": fixture.checksums["snapshot_checksum"],
        "snapshot_id": fixture.snapshot_id,
        "snapshot_lock_sha256": stack["snapshot_lock_sha256"],
        "source_checksum": fixture.checksums["source_checksum"],
        "target_checksum": fixture.checksums["target_checksum"],
    }


def _validate_result(
    value: Mapping[str, Any],
    *,
    execution_context: ExecutionContext,
    expected_condition: str,
) -> dict[str, Any]:
    if _result_family(execution_context, expected_condition) == "PHASE_A":
        from .phase_a_trial_result_schema import validate_phase_a_trial_result_strict

        return validate_phase_a_trial_result_strict(value)
    from .full_synthetic_trial_result import validate_full_synthetic_trial_result_strict

    return validate_full_synthetic_trial_result_strict(value)


def _execute_one(
    stack: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    backend_dispatch: Any = _NO_BACKEND_DISPATCH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = _require_execution_context(stack)
    route = _validate_result_route_context(context)
    stack_dispatch_present = "backend_dispatch" in stack
    argument_dispatch_present = backend_dispatch is not _NO_BACKEND_DISPATCH
    if stack_dispatch_present and argument_dispatch_present:
        _context_error(
            "EXECUTION_CONTEXT_BACKEND_POLICY_MISMATCH",
            field="backend_dispatch",
            actual="stack and keyword adapters both present",
            expected="one explicit qualification terminal adapter",
            detail="backend dispatch authority is ambiguous",
        )
    dispatch_present = stack_dispatch_present or argument_dispatch_present
    dispatch = (
        stack.get("backend_dispatch")
        if stack_dispatch_present
        else backend_dispatch
    )
    if context.mode is ExecutionMode.FORMAL and dispatch_present:
        _context_error(
            "EXECUTION_CONTEXT_BACKEND_POLICY_MISMATCH",
            field="stack.backend_dispatch",
            actual="present",
            expected="absent in FORMAL mode",
            detail="formal execution forbids a terminal backend adapter",
        )
    if dispatch_present and (
        context.mode is not ExecutionMode.QUALIFICATION
        or route != _QUALIFICATION_RESULT_ROUTE
        or not callable(dispatch)
    ):
        _context_error(
            "EXECUTION_CONTEXT_BACKEND_POLICY_MISMATCH",
            field="stack.backend_dispatch",
            actual={
                "callable": callable(dispatch),
                "mode": context.mode.value,
                "result_route": route,
            },
            expected={
                "callable": True,
                "mode": ExecutionMode.QUALIFICATION.value,
                "result_route": _QUALIFICATION_RESULT_ROUTE,
            },
            detail=(
                "terminal backend adapter is authorized only for qualification probes"
            ),
        )
    fixture = _fixture(stack, row)
    common = _common(stack, fixture, row)
    backend = row["backend"]
    family = _result_family(context, fixture.condition)
    if dispatch_present:
        result = dispatch(
            fixture=fixture,
            common=common,
            parameters=stack["parameters"][backend],
            backend=backend,
            pcl_cli=stack.get("pcl_cli"),
        )
    else:
        if family == "PHASE_A":
            from .phase_a_execution_chain_audit import (
                execute_open3d_fixture,
                execute_pcl_fixture,
            )
        else:
            from .full_synthetic_backend_execution import (
                execute_full_synthetic_open3d_fixture as execute_open3d_fixture,
            )
            from .full_synthetic_backend_execution import (
                execute_full_synthetic_pcl_fixture as execute_pcl_fixture,
            )
        if backend == "open3d_point_to_plane":
            result = execute_open3d_fixture(
                fixture=fixture,
                common=common,
                parameters=stack["parameters"][backend],
            )
        elif backend == "pcl_point_to_plane":
            result = execute_pcl_fixture(
                fixture=fixture,
                common=common,
                parameters=stack["parameters"][backend],
                pcl_cli=stack["pcl_cli"],
            )
        else:
            raise PermissionError("Native and unknown backends are forbidden")
    return common, _validate_result(
        result,
        execution_context=context,
        expected_condition=fixture.condition,
    )


def _validate_existing_result(
    path: Path,
    *,
    entry: Mapping[str, Any],
    expected: Mapping[str, Any],
    execution_context: ExecutionContext,
) -> dict[str, Any]:
    if _result_family(execution_context, expected["condition"]) == "PHASE_A":
        from .phase_a_trial_resume import validate_existing_trial_result_for_resume

        return validate_existing_trial_result_for_resume(
            path, manifest_entry=entry, expected=expected
        )
    from .full_synthetic_trial_result import (
        validate_existing_full_synthetic_trial_result_for_resume,
    )

    return validate_existing_full_synthetic_trial_result_for_resume(
        path, manifest_entry=entry, expected=expected
    )


def _audit_raw_inventory(
    stack: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    expected_rows: Mapping[str, Mapping[str, Any]],
    *,
    allow_orphans: bool,
) -> dict[str, Any]:
    results_dir = stack["runtime_paths"]["raw_results"]
    results = raw_manifest["results"]
    manifest_ids = set(results)
    expected_ids = set(expected_rows)
    payloads: list[dict[str, Any]] = []
    corrupt = checksum = duplicate_paths = 0
    path_names: list[str] = []
    for trial_id in sorted(manifest_ids & expected_ids):
        entry = results[trial_id]
        try:
            if type(entry) is not dict or set(entry) != {
                "path",
                "planned_trial_id",
                "sha256",
            }:
                raise ValueError("result entry schema")
            expected_name = _result_filename(trial_id)
            if (
                entry["path"] != expected_name
                or entry["planned_trial_id"] != trial_id
                or Path(entry["path"]).name != entry["path"]
            ):
                raise ValueError("result entry identity")
            path_names.append(entry["path"])
            path = results_dir / entry["path"]
            if _file_sha256(path) != entry["sha256"]:
                checksum += 1
                raise ValueError("result checksum")
            fixture = _fixture(stack, expected_rows[trial_id])
            payloads.append(
                _validate_existing_result(
                    path,
                    entry=entry,
                    expected=_common(stack, fixture, expected_rows[trial_id]),
                    execution_context=_require_execution_context(stack),
                )
            )
        except (ExecutionContextError, TrialSnapshotBridgeError):
            raise
        except (KeyError, OSError, TypeError, ValueError, RuntimeError):
            corrupt += 1
    duplicate_paths = len(path_names) - len(set(path_names))
    inventory = list(results_dir.iterdir()) if results_dir.exists() else []
    actual_files = {path.name for path in inventory if path.is_file()}
    invalid_types = sum(not path.is_file() for path in inventory)
    referenced = {
        entry.get("path")
        for entry in results.values()
        if type(entry) is dict and isinstance(entry.get("path"), str)
    }
    orphan_files = actual_files - referenced
    reverse = len(referenced - actual_files) + invalid_types
    if not allow_orphans:
        reverse += len(orphan_files)
    report = {
        "checksum_mismatch_count": checksum,
        "corrupt_trial_count": corrupt,
        "duplicate_trial_count": duplicate_paths,
        "extra_trial_count": len(manifest_ids - expected_ids),
        "missing_trial_count": len(expected_ids - manifest_ids),
        "orphan_result_files": sorted(orphan_files),
        "payloads": payloads,
        "reverse_inventory_mismatch_count": reverse,
    }
    if any(
        report[name]
        for name in (
            "checksum_mismatch_count",
            "corrupt_trial_count",
            "duplicate_trial_count",
            "extra_trial_count",
            "reverse_inventory_mismatch_count",
        )
    ):
        raise V3ContractError("v3 raw result inventory is corrupt or ambiguous")
    return report


def _recover_orphans(
    stack: Mapping[str, Any],
    raw_manifest: dict[str, Any],
    expected_rows: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    from .runtime_lifecycle_io import (
        atomic_replace_canonical_json,
        canonical_json_bytes,
    )

    audit = _audit_raw_inventory(
        stack, raw_manifest, expected_rows, allow_orphans=True
    )
    missing = set(expected_rows) - set(raw_manifest["results"])
    by_filename = {_result_filename(trial_id): trial_id for trial_id in missing}
    recovered: list[str] = []
    for name in audit["orphan_result_files"]:
        trial_id = by_filename.get(name)
        if trial_id is None:
            raise V3ContractError(f"unrecognized v3 raw result orphan: {name}")
        path = stack["runtime_paths"]["raw_results"] / name
        digest = _file_sha256(path)
        entry = {"path": name, "planned_trial_id": trial_id, "sha256": digest}
        fixture = _fixture(stack, expected_rows[trial_id])
        payload = _validate_existing_result(
            path,
            entry=entry,
            expected=_common(stack, fixture, expected_rows[trial_id]),
            execution_context=_require_execution_context(stack),
        )
        if path.read_bytes() != canonical_json_bytes(payload):
            raise V3ContractError("v3 orphan result is not canonical JSON")
        raw_manifest["results"][trial_id] = entry
        recovered.append(trial_id)
    if recovered:
        atomic_replace_canonical_json(stack["runtime_paths"]["raw_manifest"], raw_manifest)
    return recovered


def _git_gate(stack: Mapping[str, Any], checkpoint: str) -> dict[str, Any]:
    from .runtime_git_gate import verify_runtime_git_gate

    manifest = stack["manifest"]
    report = verify_runtime_git_gate(
        stack["repository"],
        expected_commit=stack["validated_contract"]["git_identity"]["commit"],
        expected_branch=manifest["expected_branch"],
        expected_tag=manifest["expected_release_tag"],
        checkpoint=checkpoint,
    )
    from .runtime_lifecycle_io import atomic_create_canonical_json

    directory = stack["runtime_paths"]["temporary_inventory"] / "git_gates"
    index = len(list(directory.glob("*.json"))) if directory.exists() else 0
    atomic_create_canonical_json(directory / f"{index:04d}_{checkpoint}.json", report)
    return report


def build_v3_formal_lifecycle_spec(
    *,
    repository: str | Path,
    manifest_path: str | Path,
    run_id: str,
    runtime_root: str | Path,
    workers: int,
    expected_formal_command: str | None,
) -> Any:
    """Authenticate the historical v3 binding before compatibility delegation.

    The historical v3 run is permanently non-resumable and its frozen
    implementation binding is intentionally not rewritten by this refactor.
    Consequently the current archive fails closed while retaining a single
    compatibility construction boundary for external audit.
    """

    load_v3_execution_stack(
        repository=repository,
        manifest_path=manifest_path,
        run_id=run_id,
        workers=workers,
        require_authorized=True,
    )
    del runtime_root, expected_formal_command
    raise V3ContractError(
        "historical v3 execution remains archived and is not resumable"
    )


def execute_synthetic_confirmatory_v3(
    *,
    repository: str | Path,
    manifest_path: str | Path,
    run_id: str,
    runtime_root: str | Path,
    workers: int,
    resume: bool,
    mode: str | None = None,
    expected_formal_command: str | None = None,
    invocation_id: str = "formal",
) -> dict[str, Any]:
    """Thin historical wrapper around the sole generic lifecycle entry."""

    selected_mode = mode or ("resume" if resume else "fresh")
    if type(resume) is not bool or selected_mode not in {"fresh", "resume"}:
        raise ValueError("historical v3 mode is invalid")
    if resume or selected_mode == "resume":
        raise PermissionError("historical v3 resume is permanently forbidden")
    spec = build_v3_formal_lifecycle_spec(
        repository=repository,
        manifest_path=manifest_path,
        run_id=run_id,
        runtime_root=runtime_root,
        workers=workers,
        expected_formal_command=expected_formal_command,
    )
    from .formal_lifecycle import execute_formal_lifecycle
    from .formal_lifecycle_contract import deep_thaw

    completed = execute_formal_lifecycle(
        spec,
        requested_mode=selected_mode,
        invocation_id=invocation_id,
    )
    return deep_thaw(completed.run_manifest)


def _load_snapshot_lock(stack: dict[str, Any]) -> None:
    from .runtime_lifecycle_io import read_canonical_json

    lock = read_canonical_json(stack["runtime_paths"]["snapshot_lock"])
    entries = lock.get("snapshots") if type(lock) is dict else None
    if type(entries) is not list:
        raise V3ContractError("v3 snapshot lock is malformed")
    stack["lock_by_id"] = {entry["snapshot_id"]: entry for entry in entries}
    if len(stack["lock_by_id"]) != EXPECTED_SNAPSHOT_COUNT:
        raise V3ContractError("v3 snapshot lock inventory changed")
    stack["snapshot_lock_sha256"] = _file_sha256(
        stack["runtime_paths"]["snapshot_lock"]
    )


def load_completed_v3_trials(
    *,
    repository: str | Path,
    manifest_path: str | Path,
    runtime_root: str | Path,
    checkpoint: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Strict v3 external raw-evidence reader, independent of v1/v2 loaders."""

    validated = validate_v3_postrun_entry(
        repository=repository,
        manifest_path=manifest_path,
        runtime_root=runtime_root,
        checkpoint=checkpoint,
    )
    manifest = validated["manifest"]
    stack = load_v3_execution_stack(
        repository=repository,
        manifest_path=manifest_path,
        run_id=manifest["run_id"],
        workers=manifest["workers"],
        require_authorized=True,
    )
    _load_snapshot_lock(stack)
    from .runtime_lifecycle_io import read_canonical_json

    raw = read_canonical_json(stack["runtime_paths"]["raw_manifest"])
    from .formal_runtime_state_machine import assert_seed_entry_lock

    base_contract = _run_contract(
        stack, run_id=manifest["run_id"], workers=manifest["workers"]
    )
    lock_gate = assert_seed_entry_lock(FORMAL_RUNTIME_ROOT, base_contract)
    contract_sha = lock_gate["lock"].get("payload_sha256")
    # The raw manifest binds the enhanced canonical contract object rather
    # than the enclosing run-lock payload.
    from .runtime_lifecycle_io import canonical_json_sha256

    expected_contract_sha = canonical_json_sha256(lock_gate["enhanced_contract"])
    if raw.get("contract_sha256") != expected_contract_sha or not contract_sha:
        raise V3ContractError("v3 raw/run-lock contract binding failed")
    rows_by_id = {row["planned_trial_id"]: row for row in stack["trials"]}
    inventory = _audit_raw_inventory(stack, raw, rows_by_id, allow_orphans=False)
    if inventory["missing_trial_count"] or len(inventory["payloads"]) != EXPECTED_TRIAL_COUNT:
        raise V3ContractError("v3 formal evidence is incomplete")
    enriched = []
    for payload in inventory["payloads"]:
        plan = rows_by_id[payload["planned_trial_id"]]
        enriched.append(
            {
                **payload,
                "backend_schema_name": payload["backend"],
                "geometry_seed": plan["geometry_seed"],
                "measurement_seed": plan["measurement_seed"],
                "planned_snapshot_id": plan["planned_snapshot_id"],
                "repeat_index": plan["repeat_index"],
            }
        )
    return sorted(enriched, key=lambda row: row["planned_trial_id"]), raw, stack


def _recompute_common(
    trials: Sequence[Mapping[str, Any]],
    stack: Mapping[str, Any],
    *,
    independent: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np

    from .rotation_metrics import rotation_metric_audit

    if independent:
        from .synthetic_confirmatory_independent_verifier import (
            _independent_prepare_common as prepare_context,
        )
        from .synthetic_confirmatory_independent_verifier import (
            _independent_safe_common_record as analyze_estimate,
        )
    else:
        from .common_association_analysis import (
            prepare_common_association_context as prepare_context,
        )
        from .common_association_analysis import (
            safe_analyze_estimated_transform as analyze_estimate,
        )

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[str(row["planned_snapshot_id"])].append(row)
    execution_context = _require_execution_context(stack)
    normalized: list[dict[str, Any]] = []
    common: list[dict[str, Any]] = []
    for snapshot_id in sorted(grouped):
        first = grouped[snapshot_id][0]
        first_plan = {
            name: first[name] for name in execution_context.trial_schema.fields
        }
        _first_trial, snapshot_row = _bound_snapshot_row(stack, first_plan)
        if snapshot_row["planned_snapshot_id"] != snapshot_id:
            raise TrialSnapshotBridgeError(
                "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH",
                field="planned_snapshot_id",
                actual=snapshot_id,
                expected=snapshot_row["planned_snapshot_id"],
                detail="analysis group is not bound to its canonical snapshot row",
            )
        for grouped_row in grouped[snapshot_id]:
            grouped_plan = {
                name: grouped_row[name]
                for name in execution_context.trial_schema.fields
            }
            _trial, grouped_snapshot = _bound_snapshot_row(stack, grouped_plan)
            if grouped_snapshot is not snapshot_row:
                raise TrialSnapshotBridgeError(
                    "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH",
                    field="analysis_snapshot_group",
                    actual=grouped_row["planned_trial_id"],
                    expected=snapshot_id,
                    detail="analysis group contains another canonical snapshot",
                )
        snapshot = _read_authenticated_snapshot(stack, snapshot_row)
        association_context = None
        if snapshot_row["condition"] != "IDEAL_MATCHED":
            association_context = prepare_context(
                snapshot["source"],
                snapshot["target"],
                snapshot["reference"],
                snapshot_id=snapshot_id,
            )
        for row in grouped[snapshot_id]:
            if any(
                row[name] != snapshot[name]
                for name in (
                    "snapshot_checksum",
                    "source_checksum",
                    "target_checksum",
                    "reference_pose_checksum",
                )
            ) or row["snapshot_lock_sha256"] != stack["snapshot_lock_sha256"]:
                raise V3ContractError("v3 trial/snapshot binding mismatch")
            updated = dict(row)
            transform = row.get("final_transform_4x4")
            if transform is not None and not row["solver_failure"] and row["finite_output"]:
                estimate = np.asarray(transform, dtype=np.float64)
                vector = estimate[:3, 3] - snapshot["reference"][:3, 3]
                translation = float(np.linalg.norm(vector))
                rotation = rotation_metric_audit(
                    estimate[:3, :3], snapshot["reference"][:3, :3]
                )["rotation_error_rad"]
                if rotation is None or (
                    abs(float(row["translation_update_m"]) - translation) > 1e-12
                    or abs(float(row["rotation_update_rad"]) - float(rotation)) > 1e-12
                ):
                    raise V3ContractError("v3 stored metric recomputation mismatch")
                updated.update(
                    translation_error_m=translation,
                    translation_vector=vector.astype(float).tolist(),
                    rotation_error_rad=float(rotation),
                )
            normalized.append(updated)
            if (
                row["condition"] != "IDEAL_MATCHED"
                and not row["solver_failure"]
                and row["finite_output"]
            ):
                if association_context is None or transform is None:
                    raise V3ContractError("v3 nonideal common input is absent")
                common.append(
                    analyze_estimate(
                        association_context,
                        np.asarray(transform, dtype=np.float64),
                        identifiers={
                            "planned_trial_id": row["planned_trial_id"],
                            "backend_schema_name": row["backend_schema_name"],
                            "scene_variant": row["scene_variant"],
                            "condition": row["condition"],
                            "geometry_seed": row["geometry_seed"],
                            "measurement_seed": row["measurement_seed"],
                            "repeat_index": row["repeat_index"],
                        },
                    )
                )
    return (
        sorted(normalized, key=lambda row: row["planned_trial_id"]),
        sorted(common, key=lambda row: row["planned_trial_id"]),
    )


def analyze_completed_v3(
    *, repository: str | Path, manifest_path: str | Path, runtime_root: str | Path
) -> dict[str, Any]:
    trials, _raw, stack = load_completed_v3_trials(
        repository=repository,
        manifest_path=manifest_path,
        runtime_root=runtime_root,
        checkpoint="V3_PRIMARY_ANALYSIS_INPUT_GIT_GATE",
    )
    normalized, common = _recompute_common(trials, stack, independent=False)
    manifest = stack["manifest"]
    from .synthetic_confirmatory_analysis import analyze_synthetic_confirmatory_records

    protocol = strict_json_object(_bound_path(stack["repository"], manifest, "scientific_protocol"))
    result = analyze_synthetic_confirmatory_records(
        trials=normalized,
        common_records=common,
        model_lock=strict_json_object(_bound_path(stack["repository"], manifest, "frozen_model")),
        gate_contract=strict_json_object(_bound_path(stack["repository"], manifest, "gate_contract")),
        expected_geometry_seeds=protocol["geometry_seeds"],
    )
    result = _adapt_v3_analysis(result, independent=False)
    result["schema_version"] = PRIMARY_ANALYSIS_SCHEMA
    result["run_id"] = manifest["run_id"]
    result["raw_result_manifest_sha256"] = _file_sha256(
        stack["runtime_paths"]["raw_manifest"]
    )
    return result


def independently_verify_completed_v3(
    *, repository: str | Path, manifest_path: str | Path, runtime_root: str | Path
) -> dict[str, Any]:
    trials, _raw, stack = load_completed_v3_trials(
        repository=repository,
        manifest_path=manifest_path,
        runtime_root=runtime_root,
        checkpoint="V3_INDEPENDENT_INPUT_GIT_GATE",
    )
    normalized, common = _recompute_common(trials, stack, independent=True)
    manifest = stack["manifest"]
    from .synthetic_confirmatory_independent_verifier import (
        independently_recompute_synthetic_confirmatory,
    )

    protocol = strict_json_object(_bound_path(stack["repository"], manifest, "scientific_protocol"))
    result = independently_recompute_synthetic_confirmatory(
        trials=normalized,
        common_records=common,
        model_lock=strict_json_object(_bound_path(stack["repository"], manifest, "frozen_model")),
        gate_contract=strict_json_object(_bound_path(stack["repository"], manifest, "gate_contract")),
        expected_geometry_seeds=protocol["geometry_seeds"],
    )
    result = _adapt_v3_analysis(result, independent=True)
    result["schema_version"] = INDEPENDENT_SCHEMA
    result["run_id"] = manifest["run_id"]
    result["raw_result_manifest_sha256"] = _file_sha256(
        stack["runtime_paths"]["raw_manifest"]
    )
    return result


def publish_completed_v3(
    *,
    repository: str | Path,
    manifest_path: str | Path,
    runtime_root: str | Path,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    artifact_dir: str | Path,
) -> dict[str, Any]:
    _trials, _raw, stack = load_completed_v3_trials(
        repository=repository,
        manifest_path=manifest_path,
        runtime_root=runtime_root,
        checkpoint="V3_PUBLISHER_INPUT_GIT_GATE",
    )
    from .synthetic_confirmatory_independent_verifier import compare_primary_and_independent

    comparison = compare_primary_and_independent(primary, independent)
    if not comparison.get("exact_match_pass"):
        raise V3ContractError("v3 primary and independent analyses differ")
    primary_decision = _validate_v3_final_decision(
        primary.get("final_decision")
    )
    independent_decision = _validate_v3_final_decision(
        independent.get("final_decision")
    )
    if (
        primary_decision != independent_decision
        or stack["manifest"].get("final_decision_schema")
        != V3_FINAL_DECISION_SCHEMA
    ):
        raise V3ContractError("v3 final decision identity changed")
    raw_sha = _file_sha256(stack["runtime_paths"]["raw_manifest"])
    if any(
        report.get("run_id") != stack["manifest"]["run_id"]
        or report.get("raw_result_manifest_sha256") != raw_sha
        for report in (primary, independent)
    ):
        raise V3ContractError("v3 analysis input binding mismatch")
    run_manifest = strict_json_object(stack["runtime_paths"]["run_manifest"])
    run_manifest = {**run_manifest, "raw_result_manifest_sha256": raw_sha}
    destination = Path(artifact_dir)
    if destination != stack["runtime_paths"]["artifact_staging"] or destination.exists():
        raise V3ContractError("v3 artifact destination is not the fresh frozen path")
    publisher_parent = stack["runtime_paths"]["publisher_staging"]
    publisher_parent.mkdir(parents=True, exist_ok=False)
    temporary = Path(tempfile.mkdtemp(prefix="formal-", dir=publisher_parent))
    staging = temporary / "artifact"
    try:
        from .synthetic_confirmatory_publisher import _publish_into

        generic_primary = _genericize_v3_analysis(primary, independent=False)
        generic_independent = _genericize_v3_analysis(
            independent, independent=True
        )
        _publish_into(
            staging,
            primary=generic_primary,
            independent=generic_independent,
            run_manifest=run_manifest,
        )
        from .runtime_lifecycle_io import canonical_json_bytes

        for name, value in (
            ("primary_analysis.json", primary),
            ("independent_verification.json", independent),
            ("final_decision.json", primary_decision),
        ):
            (staging / name).write_bytes(canonical_json_bytes(value))
        report_path = staging / "synthetic_confirmatory_report.md"
        report_text = report_path.read_text(encoding="utf-8")
        if not report_text.startswith("# Synthetic Confirmatory v1\n"):
            raise V3ContractError("delegated publisher report identity changed")
        report_text = report_text.replace(
                "# Synthetic Confirmatory v1\n",
                "# Synthetic Confirmatory v3\n",
                1,
            )
        for generic_name, v3_name in _GENERIC_TO_V3_DECISION_FIELDS.items():
            if generic_name.startswith("SYNTHETIC_CONFIRMATORY_"):
                report_text = report_text.replace(generic_name, v3_name)
        report_path.write_text(report_text, encoding="utf-8")
        checksum_paths = sorted(
            candidate
            for candidate in staging.rglob("*")
            if candidate.is_file()
            and candidate.name not in {"SHA256SUMS", "artifact_verification.json"}
        )
        (staging / "SHA256SUMS").write_text(
            "".join(
                f"{_file_sha256(candidate)}  "
                f"{candidate.relative_to(staging).as_posix()}\n"
                for candidate in checksum_paths
            ),
            encoding="utf-8",
        )
        from .synthetic_confirmatory_artifact_verifier import (
            verify_synthetic_confirmatory_artifact,
        )

        recorded = verify_synthetic_confirmatory_artifact(staging, write_report=True)
        live = verify_synthetic_confirmatory_artifact(staging, write_report=False)
        if recorded != live or live.get("ARTIFACT_VERIFICATION_PASS") is not True:
            raise V3ContractError("v3-adapted staged artifact verification failed")
        from .runtime_lifecycle_io import atomic_publish_directory

        atomic_publish_directory(staging, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    verification = verify_v3_formal_artifact(destination)
    if verification["V3_FORMAL_ARTIFACT_VERIFICATION_PASS"] is not True:
        raise V3ContractError("published v3 formal artifact failed verification")
    return {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": comparison["leaf_difference_count"],
        "V3_FORMAL_ARTIFACT_VERIFICATION_PASS": True,
        "artifact_path": str(destination),
        "published_file_count": verification["actual_file_count"],
        "sha256_mismatch_count": len(verification["sha256_mismatch_files"]),
    }


def verify_v3_formal_artifact(path: str | Path) -> dict[str, Any]:
    """Verify the frozen 7-table/3-figure/7-root-file v3 publication."""

    from .synthetic_confirmatory_artifact_verifier import (
        verify_synthetic_confirmatory_artifact,
    )

    root = Path(path)
    generic = verify_synthetic_confirmatory_artifact(root, write_report=False)
    try:
        recorded = strict_json_object(root / "artifact_verification.json")
    except (FileNotFoundError, V3ContractError, OSError, ValueError):
        recorded = None
    primary = strict_json_object(root / "primary_analysis.json") if root.is_dir() else {}
    independent = (
        strict_json_object(root / "independent_verification.json")
        if root.is_dir()
        else {}
    )
    final_decision = (
        strict_json_object(root / "final_decision.json")
        if root.is_dir()
        else {}
    )
    run = strict_json_object(root / "run_manifest.json") if root.is_dir() else {}
    actual = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file()
    } if root.is_dir() else set()
    try:
        authenticated_decision = _validate_v3_final_decision(final_decision)
        decision_identity = bool(
            primary.get("final_decision") == authenticated_decision
            and independent.get("final_decision") == authenticated_decision
            and independent.get("verification_projection", {}).get(
                "final_decision"
            )
            == authenticated_decision
        )
    except (TypeError, V3ContractError):
        decision_identity = False
    report_path = root / "synthetic_confirmatory_report.md"
    report_text = (
        report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    )
    report_identity = bool(
        report_text.startswith("# Synthetic Confirmatory v3\n")
        and "SYNTHETIC_CONFIRMATORY_V3_EXECUTED" in report_text
        and "SYNTHETIC_CONFIRMATORY_V3_COMPLETE" in report_text
        and "SYNTHETIC_CONFIRMATORY_V3_PASS" in report_text
        and "SYNTHETIC_CONFIRMATORY_EXECUTED" not in report_text
        and "SYNTHETIC_CONFIRMATORY_COMPLETE" not in report_text
        and "SYNTHETIC_CONFIRMATORY_PASS" not in report_text
    )
    v3_identity = bool(
        primary.get("schema_version") == PRIMARY_ANALYSIS_SCHEMA
        and independent.get("schema_version") == INDEPENDENT_SCHEMA
        and run.get("schema_version") == FORMAL_RUN_SCHEMA
        and run.get("run_id") == "synthetic-confirmatory-v3"
        and decision_identity
        and report_identity
    )
    result = {
        **generic,
        "schema_version": "synthetic_confirmatory_v3_formal_artifact_verification_v1",
        "actual_file_count": len(actual),
        "publisher_7_table_3_figure_7_root_pass": bool(
            len([name for name in actual if name.startswith("tables/")]) == 7
            and len([name for name in actual if name.startswith("figures/")]) == 3
            and len([name for name in actual if "/" not in name]) == 7
        ),
        "persisted_artifact_verification_match_pass": recorded == generic,
        "v3_final_decision_identity_pass": decision_identity,
        "v3_report_identity_pass": report_identity,
        "v3_identity_pass": v3_identity,
    }
    result["V3_FORMAL_ARTIFACT_VERIFICATION_PASS"] = bool(
        generic.get("ARTIFACT_VERIFICATION_PASS") is True
        and result["publisher_7_table_3_figure_7_root_pass"]
        and result["persisted_artifact_verification_match_pass"]
        and v3_identity
    )
    return result


__all__ = [
    "FORMAL_ARTIFACT_SCHEMA",
    "FORMAL_RUN_SCHEMA",
    "INDEPENDENT_SCHEMA",
    "PRIMARY_ANALYSIS_SCHEMA",
    "RAW_RESULT_MANIFEST_SCHEMA",
    "V3_FINAL_DECISION_SCHEMA",
    "analyze_completed_v3",
    "execute_synthetic_confirmatory_v3",
    "independently_verify_completed_v3",
    "load_completed_v3_trials",
    "load_v3_execution_stack",
    "publish_completed_v3",
    "verify_v3_formal_artifact",
]
