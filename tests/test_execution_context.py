from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from phase_a_harness.execution_context import (
    BackendPolicy,
    canonical_plan_rows_sha256,
    ExecutionContext,
    ExecutionContextError,
    ExecutionContract,
    ExecutionMode,
    IdPolicy,
    SchemaBinding,
    SeedPolicy,
    SnapshotReaderPolicy,
)


SNAPSHOT_FIELDS = (
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "planned_backend_count",
    "replicate_semantics",
)
TRIAL_FIELDS = (
    "planned_trial_id",
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "backend",
)
BACKENDS = ("open3d_point_to_plane", "pcl_point_to_plane")


def _identity_validator(row):
    return dict(row)


def _contract(tmp_path: Path, mode: ExecutionMode) -> ExecutionContract:
    label = mode.value.lower()
    runtime = tmp_path / label / "runtime"
    conditions = (("FORMAL_CONDITION",) if mode is ExecutionMode.FORMAL else ("FIXTURE",))
    snapshot_schema = SchemaBinding(mode, f"{label}-snapshot-plan-v1", SNAPSHOT_FIELDS)
    trial_schema = SchemaBinding(mode, f"{label}-trial-plan-v1", TRIAL_FIELDS)
    seed_policy = SeedPolicy(
        mode,
        f"{label}-seed-policy-v1",
        confirmatory_seed_allowed=(mode is ExecutionMode.FORMAL),
    )
    reader_policy = SnapshotReaderPolicy(
        mode=mode,
        policy_id=f"{label}-reader-v1",
        metadata_fields=("schema_version", "snapshot_id"),
        lock_entry_fields=("snapshot_id", "file_sha256"),
        metadata_schema=f"{label}-metadata-v1",
        snapshot_schema_version=f"{label}-snapshot-v1",
        lineage_schema_version=f"{label}-lineage-v1",
        seed_namespace=f"{label}-namespace-v1",
        snapshot_builder_contract_version=f"{label}-builder-v1",
        lineage_required_conditions=conditions,
        expected_rng_counts={condition: 0 for condition in conditions},
    )
    snapshot_plan = [
        {
            "planned_snapshot_id": f"{label}-snapshot-v1",
            "scene_variant": "FORMAL_SCENE" if mode is ExecutionMode.FORMAL else "IDENTITY",
            "condition": conditions[0],
            "planned_backend_count": 2,
            "replicate_semantics": "ONE_INPUT",
        }
    ]
    trial_plan = [
        {
            "planned_trial_id": f"{label}-snapshot-v1::{backend}",
            "planned_snapshot_id": f"{label}-snapshot-v1",
            "scene_variant": snapshot_plan[0]["scene_variant"],
            "condition": conditions[0],
            "backend": backend,
        }
        for backend in BACKENDS
    ]
    return ExecutionContract(
        mode=mode,
        contract_id=f"{label}-contract-v1",
        plan_id=f"{label}-plan-v1",
        expected_snapshot_plan_sha256=canonical_plan_rows_sha256(snapshot_plan),
        expected_trial_plan_sha256=canonical_plan_rows_sha256(trial_plan),
        expected_snapshot_count=len(snapshot_plan),
        expected_trial_count=len(trial_plan),
        expected_cache_root=runtime / "snapshot_cache",
        expected_runtime_root=runtime,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=(("FORMAL_SCENE",) if mode is ExecutionMode.FORMAL else ("IDENTITY",)),
        allowed_conditions=conditions,
        allowed_seed_policy=seed_policy,
        id_policy=IdPolicy(mode, f"{label}-id-policy-v1"),
        backend_policy=BackendPolicy(mode, f"{label}-backend-policy-v1", BACKENDS, 2),
        snapshot_reader_policy=reader_policy,
        snapshot_validator=_identity_validator,
        trial_validator=_identity_validator,
    )


def _context(contract: ExecutionContract, **changes) -> ExecutionContext:
    values = {
        "mode": contract.mode,
        "contract": contract,
        "plan_id": contract.plan_id,
        "cache_root": contract.expected_cache_root,
        "runtime_root": contract.expected_runtime_root,
        "snapshot_schema": contract.snapshot_schema,
        "trial_schema": contract.trial_schema,
        "allowed_scenes": contract.allowed_scenes,
        "allowed_conditions": contract.allowed_conditions,
        "allowed_seed_policy": contract.allowed_seed_policy,
        "id_policy": contract.id_policy,
        "backend_policy": contract.backend_policy,
        "snapshot_reader_policy": contract.snapshot_reader_policy,
    }
    values.update(changes)
    return ExecutionContext(**values)


def _classification(expected: str, operation) -> ExecutionContextError:
    with pytest.raises(ExecutionContextError) as captured:
        operation()
    assert captured.value.classification == expected
    assert captured.value.category == expected
    json.dumps(captured.value.report(), sort_keys=True, allow_nan=False)
    return captured.value


@pytest.mark.parametrize("mode", tuple(ExecutionMode))
def test_valid_context_is_explicit_immutable_and_json_native(
    tmp_path: Path, mode: ExecutionMode
) -> None:
    contract = _contract(tmp_path, mode)
    context = _context(contract)
    report = context.report()
    assert report["EXECUTION_CONTEXT_PARAMETERIZATION_PASS"] is True
    assert report["mode"] == mode.value
    assert report["snapshot_reader_policy"] == contract.snapshot_reader_policy.report()
    assert report["contract"]["snapshot_validator"].endswith("._identity_validator")
    assert json.loads(json.dumps(report, sort_keys=True, allow_nan=False)) == report
    with pytest.raises(AttributeError):
        context.mode = ExecutionMode.QUALIFICATION
    with pytest.raises(TypeError):
        context.snapshot_reader_policy.expected_rng_counts["changed"] = 1


def test_mode_is_never_inferred_or_coerced(tmp_path: Path, monkeypatch) -> None:
    contract = _contract(tmp_path, ExecutionMode.FORMAL)
    monkeypatch.setenv("EXECUTION_MODE", "FORMAL")
    _classification(
        "EXECUTION_CONTEXT_MODE_MISMATCH",
        lambda: _context(contract, mode="FORMAL"),
    )


@pytest.mark.parametrize(
    ("field", "classification"),
    [
        ("plan_id", "EXECUTION_CONTEXT_PLAN_MISMATCH"),
        ("cache_root", "EXECUTION_CONTEXT_CACHE_MISMATCH"),
        ("runtime_root", "EXECUTION_CONTEXT_RUNTIME_MISMATCH"),
        ("snapshot_schema", "EXECUTION_CONTEXT_SCHEMA_MISMATCH"),
        ("trial_schema", "EXECUTION_CONTEXT_SCHEMA_MISMATCH"),
        ("allowed_scenes", "EXECUTION_CONTEXT_SCENE_POLICY_MISMATCH"),
        ("allowed_conditions", "EXECUTION_CONTEXT_CONDITION_POLICY_MISMATCH"),
        ("allowed_seed_policy", "EXECUTION_CONTEXT_SEED_POLICY_MISMATCH"),
        ("id_policy", "EXECUTION_CONTEXT_ID_POLICY_MISMATCH"),
        ("backend_policy", "EXECUTION_CONTEXT_BACKEND_POLICY_MISMATCH"),
        ("snapshot_reader_policy", "EXECUTION_CONTEXT_READER_POLICY_MISMATCH"),
    ],
)
def test_every_injected_dependency_is_exactly_contract_bound(
    tmp_path: Path, field: str, classification: str
) -> None:
    formal = _contract(tmp_path, ExecutionMode.FORMAL)
    qualification = _contract(tmp_path, ExecutionMode.QUALIFICATION)
    replacement = {
        "plan_id": qualification.plan_id,
        "cache_root": qualification.expected_cache_root,
        "runtime_root": qualification.expected_runtime_root,
        "snapshot_schema": replace(formal.snapshot_schema, schema_id="other"),
        "trial_schema": replace(formal.trial_schema, schema_id="other"),
        "allowed_scenes": ("OTHER_SCENE",),
        "allowed_conditions": ("OTHER_CONDITION",),
        "allowed_seed_policy": replace(formal.allowed_seed_policy, policy_id="other"),
        "id_policy": replace(formal.id_policy, policy_id="other"),
        "backend_policy": replace(formal.backend_policy, policy_id="other"),
        "snapshot_reader_policy": replace(
            formal.snapshot_reader_policy, policy_id="other"
        ),
    }[field]
    error = _classification(
        classification, lambda: _context(formal, **{field: replacement})
    )
    assert error.field == field


@pytest.mark.parametrize(
    ("context_mode", "contract_mode"),
    [
        (ExecutionMode.FORMAL, ExecutionMode.QUALIFICATION),
        (ExecutionMode.QUALIFICATION, ExecutionMode.FORMAL),
    ],
)
def test_formal_and_qualification_contracts_cannot_cross_modes(
    tmp_path: Path, context_mode: ExecutionMode, contract_mode: ExecutionMode
) -> None:
    contract = _contract(tmp_path, contract_mode)
    _classification(
        "EXECUTION_CONTEXT_MODE_MISMATCH",
        lambda: _context(contract, mode=context_mode),
    )


def test_contract_rejects_cross_mode_schema_and_reader_policy(tmp_path: Path) -> None:
    formal = _contract(tmp_path, ExecutionMode.FORMAL)
    qualification = _contract(tmp_path, ExecutionMode.QUALIFICATION)
    _classification(
        "EXECUTION_CONTEXT_SCHEMA_MISMATCH",
        lambda: replace(formal, snapshot_schema=qualification.snapshot_schema),
    )
    _classification(
        "EXECUTION_CONTEXT_READER_POLICY_MISMATCH",
        lambda: replace(
            formal, snapshot_reader_policy=qualification.snapshot_reader_policy
        ),
    )


def test_qualification_can_never_authorize_confirmatory_seeds() -> None:
    _classification(
        "EXECUTION_CONTEXT_SEED_POLICY_MISMATCH",
        lambda: SeedPolicy(
            ExecutionMode.QUALIFICATION,
            "invalid-qualification-seed-policy",
            confirmatory_seed_allowed=True,
        ),
    )


def test_reader_policy_must_cover_exact_condition_vocabulary(tmp_path: Path) -> None:
    formal = _contract(tmp_path, ExecutionMode.FORMAL)
    incomplete = replace(formal.snapshot_reader_policy, expected_rng_counts={"OTHER": 0})
    _classification(
        "EXECUTION_CONTEXT_READER_POLICY_MISMATCH",
        lambda: replace(formal, snapshot_reader_policy=incomplete),
    )


def test_snapshot_and_trial_validation_use_only_injected_validators(
    tmp_path: Path,
) -> None:
    context = _context(_contract(tmp_path, ExecutionMode.QUALIFICATION))
    snapshot = {
        "planned_snapshot_id": "fixture-snapshot",
        "scene_variant": "IDENTITY",
        "condition": "FIXTURE",
        "planned_backend_count": 2,
        "replicate_semantics": "ONE_FIXTURE_INPUT",
    }
    trial = {
        "planned_trial_id": "fixture-snapshot::open3d_point_to_plane",
        "planned_snapshot_id": "fixture-snapshot",
        "scene_variant": "IDENTITY",
        "condition": "FIXTURE",
        "backend": "open3d_point_to_plane",
    }
    assert context.validate_snapshot_row(snapshot) == snapshot
    assert context.validate_trial_row(trial) == trial
    _classification(
        "SNAPSHOT_ROW_VALIDATION_FAILURE",
        lambda: context.validate_snapshot_row({**snapshot, "backend": "forbidden"}),
    )
    _classification(
        "TRIAL_ROW_VALIDATION_FAILURE",
        lambda: context.validate_trial_row({key: value for key, value in trial.items() if key != "backend"}),
    )


def test_injected_validator_failure_is_structured(tmp_path: Path) -> None:
    formal = _contract(tmp_path, ExecutionMode.FORMAL)

    def reject(_row):
        raise ValueError("rejected")

    context = _context(replace(formal, snapshot_validator=reject))
    row = {field: "value" for field in SNAPSHOT_FIELDS}
    error = _classification(
        "SNAPSHOT_ROW_VALIDATION_FAILURE",
        lambda: context.validate_snapshot_row(row),
    )
    assert error.actual == "ValueError"


def test_cache_must_be_an_explicit_runtime_descendant(tmp_path: Path) -> None:
    formal = _contract(tmp_path, ExecutionMode.FORMAL)
    _classification(
        "EXECUTION_CONTEXT_CACHE_MISMATCH",
        lambda: replace(formal, expected_cache_root=tmp_path / "outside-cache"),
    )
