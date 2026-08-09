from __future__ import annotations

import copy
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import pytest

from phase_a_harness import synthetic_confirmatory_v3_contract as formal_contract
from phase_a_harness import synthetic_confirmatory_v3_runner as runner
from phase_a_harness import synthetic_confirmatory_v3_snapshot_builder as snapshot_reader
from phase_a_harness.execution_context import ExecutionContextError, ExecutionMode
from phase_a_harness.trial_snapshot_bridge import (
    TrialSnapshotBridgeError,
    build_canonical_snapshot_index,
    build_trial_snapshot_bindings,
)


REPOSITORY = Path(__file__).resolve().parents[1]
QUALIFIER_PATH = REPOSITORY / "scripts/qualify_formal_execution_context_bridge.py"
BACKEND_MODULES = {
    "phase_a_harness.phase_a_execution_chain_audit",
    "phase_a_harness.full_synthetic_backend_execution",
}


class DispatchBoundaryReached(RuntimeError):
    pass


def _load_qualifier():
    specification = importlib.util.spec_from_file_location(
        "formal_execution_context_bridge_test_helper", QUALIFIER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("qualification helper cannot be imported")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def qualification_stack(tmp_path: Path, monkeypatch):
    import numpy as np

    def forbidden_rng(*_args, **_kwargs):
        raise AssertionError("qualification integration attempted seed/RNG access")

    monkeypatch.setattr(np.random, "default_rng", forbidden_rng)
    monkeypatch.setattr(np.random, "seed", forbidden_rng)
    monkeypatch.setattr(formal_contract, "derive_seed", forbidden_rng)

    qualifier = _load_qualifier()
    runtime_root = (tmp_path / "qualification_runtime").resolve()
    cache_root = runtime_root / "snapshot_cache"
    context = qualifier.build_qualification_execution_context(
        runtime_root=runtime_root,
        cache_root=cache_root,
    )
    snapshots = qualifier.qualification_snapshot_rows()
    trials = qualifier.qualification_trial_rows()
    state = qualifier.materialize_qualification_snapshots(
        execution_context=context,
        snapshot_rows=snapshots,
    )
    index = build_canonical_snapshot_index(
        snapshots, execution_context=context
    )
    bindings = build_trial_snapshot_bindings(index, trials)
    parameters = {
        backend: {"qualification_terminal_binding": backend}
        for backend in qualifier.BACKENDS
    }
    stack = {
        "execution_context": context,
        "canonical_snapshot_index": index,
        "trial_snapshot_bindings": bindings,
        "lock_by_id": state["lock_by_id"],
        "snapshot_lock_sha256": state["snapshot_lock_sha256"],
        "runtime_paths": {
            "snapshot_cache": context.cache_root,
            "raw_results": runtime_root / "raw_results",
            "raw_manifest": runtime_root / "raw_result_manifest.json",
            "backend_temporary": runtime_root / "backend_tmp",
        },
        "parameters": parameters,
        "pcl_cli": REPOSITORY / "bin/pcl_point_to_plane_cli",
        "manifest": {
            "manifest_payload_sha256": "1" * 64,
            "protocol_sha256": "2" * 64,
        },
    }
    assert context.mode is ExecutionMode.QUALIFICATION
    assert context.allowed_seed_policy.confirmatory_seed_allowed is False
    assert state["generated_snapshot_count"] == 3
    assert state["confirmatory_seed_access_count"] == 0
    assert state["confirmatory_rng_instantiation_count"] == 0
    return qualifier, context, snapshots, trials, state, stack


def _probe_six_trials(
    stack: Mapping[str, Any], trials: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def terminal_dispatch(
        *,
        fixture: Any,
        common: Mapping[str, Any],
        parameters: Mapping[str, Any],
        backend: str,
        pcl_cli: Path,
    ) -> dict[str, Any]:
        assert parameters == stack["parameters"][backend]
        assert Path(pcl_cli) == Path(stack["pcl_cli"])
        records.append(
            {
                "backend": backend,
                "condition": fixture.condition,
                "planned_trial_id": common["planned_trial_id"],
                "scene_variant": fixture.scene_variant,
                "snapshot_id": fixture.snapshot_id,
                "snapshot_checksum": fixture.checksums["snapshot_checksum"],
                "source_checksum": fixture.checksums["source_checksum"],
                "target_checksum": fixture.checksums["target_checksum"],
                "reference_pose_checksum": fixture.checksums[
                    "reference_pose_checksum"
                ],
                "source_point_count": int(len(fixture.source)),
                "target_point_count": int(len(fixture.target)),
            }
        )
        raise DispatchBoundaryReached(str(common["planned_trial_id"]))

    probe_stack = dict(stack)
    probe_stack["backend_dispatch"] = terminal_dispatch
    for trial in trials:
        with pytest.raises(DispatchBoundaryReached) as captured:
            runner._execute_one(probe_stack, trial)
        assert str(captured.value) == trial["planned_trial_id"]
    return records


def _authenticated_snapshot_identity(
    context, state: Mapping[str, Any], snapshots: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in snapshots:
        snapshot_id = row["planned_snapshot_id"]
        item = snapshot_reader.read_v3_snapshot(
            context.cache_root,
            row,
            execution_context=context,
            expected_lock_entry=state["lock_by_id"][snapshot_id],
            arrays=False,
        )
        result[snapshot_id] = {
            "snapshot_checksum": item["snapshot_checksum"],
            "source_checksum": item["source_checksum"],
            "target_checksum": item["target_checksum"],
            "reference_pose_checksum": item["reference_pose_checksum"],
            "file_sha256": dict(sorted(item["file_sha256"].items())),
        }
    return result


def test_three_scene_six_trial_real_reader_bridge_runner_path_is_unmocked(
    qualification_stack, monkeypatch
) -> None:
    qualifier, context, snapshots, trials, state, stack = qualification_stack
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name in BACKEND_MODULES:
            raise AssertionError(f"backend implementation import forbidden: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    records = _probe_six_trials(stack, trials)

    assert len(records) == 6
    assert {row["scene_variant"] for row in records} == set(
        qualifier.QUALIFICATION_SCENES
    )
    assert {row["condition"] for row in records} == set(
        qualifier.QUALIFICATION_CONDITIONS
    )
    assert {row["backend"] for row in records} == set(qualifier.BACKENDS)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["snapshot_id"]].append(record)
    assert len(grouped) == 3
    for snapshot_id, paired in grouped.items():
        assert len(paired) == 2
        assert {row["backend"] for row in paired} == set(qualifier.BACKENDS)
        for field in (
            "snapshot_checksum",
            "source_checksum",
            "target_checksum",
            "reference_pose_checksum",
            "source_point_count",
            "target_point_count",
        ):
            assert len({row[field] for row in paired}) == 1, (snapshot_id, field)

    assert runner._execute_one.__module__ == (
        "phase_a_harness.synthetic_confirmatory_v3_runner"
    )
    assert runner._fixture.__module__ == (
        "phase_a_harness.synthetic_confirmatory_v3_runner"
    )
    assert state["confirmatory_seed_access_count"] == 0
    assert state["confirmatory_rng_instantiation_count"] == 0
    assert context.cache_root != formal_contract.RUNTIME_PATHS["snapshot_cache_path"]


def test_seed_free_fresh_resume_reauthentication_is_scientifically_equivalent(
    qualification_stack,
) -> None:
    _qualifier, context, snapshots, trials, state, stack = qualification_stack
    before = _authenticated_snapshot_identity(context, state, snapshots)
    fresh = _probe_six_trials(stack, trials)

    resumed = _probe_six_trials(stack, trials)
    after = _authenticated_snapshot_identity(context, state, snapshots)

    assert fresh == resumed
    assert before == after
    assert len(fresh) == 6
    assert state["generated_snapshot_count"] == 3
    assert state["confirmatory_seed_access_count"] == 0
    assert state["confirmatory_rng_instantiation_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "trial_scene_mismatch",
        "missing_lock_entry",
        "lock_checksum_mismatch",
        "missing_execution_context",
    ),
)
def test_every_invalid_full_path_input_fails_before_terminal_dispatch(
    qualification_stack, mutation: str
) -> None:
    _qualifier, _context, _snapshots, trials, _state, stack = qualification_stack
    calls: list[str] = []

    def forbidden_dispatch(**_kwargs):
        calls.append("backend-dispatch")
        raise AssertionError("invalid input reached terminal backend dispatch")

    bad_stack = dict(stack)
    bad_stack["backend_dispatch"] = forbidden_dispatch
    bad_trial = copy.deepcopy(trials[0])
    if mutation == "trial_scene_mismatch":
        bad_trial["scene_variant"] = "NONIDENTITY_REFERENCE"
        expected = (ExecutionContextError, TrialSnapshotBridgeError)
    elif mutation == "missing_lock_entry":
        bad_stack["lock_by_id"] = {}
        expected = TrialSnapshotBridgeError
    elif mutation == "lock_checksum_mismatch":
        lock_by_id = copy.deepcopy(stack["lock_by_id"])
        snapshot_id = bad_trial["planned_snapshot_id"]
        lock_by_id[snapshot_id]["snapshot_checksum"] = "0" * 64
        bad_stack["lock_by_id"] = lock_by_id
        expected = TrialSnapshotBridgeError
    else:
        bad_stack.pop("execution_context")
        expected = ExecutionContextError

    with pytest.raises(expected):
        runner._execute_one(bad_stack, bad_trial)
    assert calls == []


def test_production_qualification_plan_audit_is_metadata_only() -> None:
    qualifier = _load_qualifier()
    report, context, snapshots, trials = qualifier.audit_formal_plan_bridge()
    assert report["FORMAL_PLAN_BRIDGE_STATIC_AUDIT_PASS"] is True
    assert report["planned_snapshot_count"] == 595
    assert report["unique_snapshot_count"] == 595
    assert report["planned_trial_count"] == 1190
    assert report["unique_trial_count"] == 1190
    assert report["backend_trial_counts"] == {
        "open3d_point_to_plane": 595,
        "pcl_point_to_plane": 595,
    }
    assert report["native_trial_count"] == 0
    assert report["PLAN_TRIAL_WITHOUT_SNAPSHOT_COUNT"] == 0
    assert report["PLAN_DUPLICATE_SNAPSHOT_ID_COUNT"] == 0
    assert report["PLAN_SHARED_IDENTITY_MISMATCH_COUNT"] == 0
    assert report["PLAN_BACKEND_PAIRING_VIOLATION_COUNT"] == 0
    assert report["formal_snapshot_payload_read_count"] == 0
    assert report["formal_backend_execution_count"] == 0
    assert report["confirmatory_seed_schedule_read_count"] == 0
    assert context.mode is ExecutionMode.FORMAL
    assert len(snapshots) == 595
    assert len(trials) == 1190
