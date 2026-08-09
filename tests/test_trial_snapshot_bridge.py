from __future__ import annotations

import json
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
from phase_a_harness.trial_snapshot_bridge import (
    TrialSnapshotBridgeError,
    bind_trial_to_snapshot,
    build_canonical_snapshot_index,
    build_trial_snapshot_bindings,
    derive_shared_identity_contract,
)


SNAPSHOT_FIELDS = (
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
    "planned_backend_count",
    "replicate_semantics",
)
TRIAL_FIELDS = (
    "planned_trial_id",
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
    "backend",
)
SHARED_FIELDS = (
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
)
SCENES = ("IDENTITY", "NONIDENTITY_REFERENCE", "NO_CORRESPONDENCE")
CONDITIONS = ("SEED_FREE_FIXTURE", "SEED_FREE_ALTERNATE")
BACKENDS = ("open3d_point_to_plane", "pcl_point_to_plane")


def _snapshot_validator(row):
    value = {name: row[name] for name in SNAPSHOT_FIELDS}
    if (
        type(value["planned_snapshot_id"]) is not str
        or not value["planned_snapshot_id"]
        or value["scene_variant"] not in SCENES
        or value["condition"] not in CONDITIONS
        or type(value["geometry_seed"]) is not str
        or type(value["measurement_seed"]) is not str
        or type(value["repeat_index"]) is not int
        or value["repeat_index"] not in (0, 1)
        or value["planned_backend_count"] != 2
        or value["replicate_semantics"] != "ONE_SEED_FREE_FIXTURE_INPUT"
    ):
        raise ValueError("invalid seed-free qualification snapshot")
    return value


def _trial_validator(row):
    value = {name: row[name] for name in TRIAL_FIELDS}
    if (
        type(value["planned_trial_id"]) is not str
        or type(value["planned_snapshot_id"]) is not str
        or value["scene_variant"] not in SCENES
        or value["condition"] not in CONDITIONS
        or type(value["geometry_seed"]) is not str
        or type(value["measurement_seed"]) is not str
        or type(value["repeat_index"]) is not int
        or value["repeat_index"] not in (0, 1)
        or value["backend"] not in BACKENDS
        or (
            value["planned_trial_id"]
            != f"{value['planned_snapshot_id']}::{value['backend']}"
        )
    ):
        raise ValueError("invalid seed-free qualification trial")
    return value


def qualification_context(tmp_path: Path) -> ExecutionContext:
    mode = ExecutionMode.QUALIFICATION
    runtime = (tmp_path / "qualification_runtime").resolve()
    snapshot_schema = SchemaBinding(
        mode, "seed-free-qualification-snapshot-plan-v1", SNAPSHOT_FIELDS
    )
    trial_schema = SchemaBinding(
        mode, "seed-free-qualification-trial-plan-v1", TRIAL_FIELDS
    )
    seed_policy = SeedPolicy(
        mode,
        "seed-free-qualification-policy-v1",
        confirmatory_seed_allowed=False,
    )
    id_policy = IdPolicy(mode, "seed-free-fixture-id-policy-v1")
    backend_policy = BackendPolicy(
        mode, "qualification-dual-backend-policy-v1", BACKENDS, 2
    )
    reader_policy = SnapshotReaderPolicy(
        mode=mode,
        policy_id="seed-free-qualification-reader-v1",
        metadata_fields=("schema_version", "snapshot_id"),
        lock_entry_fields=("snapshot_id", "file_sha256"),
        metadata_schema="seed-free-qualification-metadata-v1",
        snapshot_schema_version="seed-free-qualification-snapshot-v1",
        lineage_schema_version="seed-free-qualification-lineage-v1",
        seed_namespace=None,
        snapshot_builder_contract_version="seed-free-fixture-builder-v1",
        lineage_required_conditions=("SEED_FREE_FIXTURE",),
        expected_rng_counts={condition: 0 for condition in CONDITIONS},
    )
    snapshot_plan, trial_plan = qualification_rows()
    contract = ExecutionContract(
        mode=mode,
        contract_id="seed-free-qualification-execution-contract-v1",
        plan_id="seed-free-three-scene-six-trial-plan-v1",
        expected_snapshot_plan_sha256=canonical_plan_rows_sha256(snapshot_plan),
        expected_trial_plan_sha256=canonical_plan_rows_sha256(trial_plan),
        expected_snapshot_count=len(snapshot_plan),
        expected_trial_count=len(trial_plan),
        expected_cache_root=runtime / "snapshot_cache",
        expected_runtime_root=runtime,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=SCENES,
        allowed_conditions=CONDITIONS,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
        snapshot_validator=_snapshot_validator,
        trial_validator=_trial_validator,
        result_route="QUALIFICATION_FIXTURE_ROUTING_V1",
    )
    return ExecutionContext(
        mode=mode,
        contract=contract,
        plan_id=contract.plan_id,
        cache_root=contract.expected_cache_root,
        runtime_root=contract.expected_runtime_root,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=SCENES,
        allowed_conditions=CONDITIONS,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
    )


def qualification_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    snapshots: list[dict[str, object]] = []
    trials: list[dict[str, object]] = []
    for scene in SCENES:
        snapshot_id = f"qualification::{scene}"
        snapshot = {
            "planned_snapshot_id": snapshot_id,
            "scene_variant": scene,
            "condition": "SEED_FREE_FIXTURE",
            "geometry_seed": "NO_GEOMETRY_SEED",
            "measurement_seed": "NO_MEASUREMENT_SEED",
            "repeat_index": 0,
            "planned_backend_count": 2,
            "replicate_semantics": "ONE_SEED_FREE_FIXTURE_INPUT",
        }
        snapshots.append(snapshot)
        for backend in BACKENDS:
            trials.append(
                {
                    "planned_trial_id": f"{snapshot_id}::{backend}",
                    **{name: snapshot[name] for name in SHARED_FIELDS},
                    "backend": backend,
                }
            )
    return snapshots, trials


def _bridge_error(expected: str, operation) -> TrialSnapshotBridgeError:
    with pytest.raises(TrialSnapshotBridgeError) as captured:
        operation()
    assert captured.value.classification == expected
    assert captured.value.category == expected
    json.dumps(captured.value.report(), sort_keys=True, allow_nan=False)
    return captured.value


def test_field_contract_is_the_exact_snapshot_trial_bridge(tmp_path: Path) -> None:
    contract = derive_shared_identity_contract(qualification_context(tmp_path))
    assert contract.snapshot_fields == SNAPSHOT_FIELDS
    assert contract.trial_fields == TRIAL_FIELDS
    assert contract.shared_identity_fields == SHARED_FIELDS
    assert contract.snapshot_only_fields == (
        "planned_backend_count",
        "replicate_semantics",
    )
    assert contract.trial_only_fields == ("planned_trial_id", "backend")


def test_canonical_index_is_an_immutable_defensive_copy(tmp_path: Path) -> None:
    context = qualification_context(tmp_path)
    snapshots, _trials = qualification_rows()
    index = build_canonical_snapshot_index(
        snapshots, execution_context=context
    )
    first_id = str(snapshots[0]["planned_snapshot_id"])
    original_scene = index[first_id]["scene_variant"]

    snapshots[0]["scene_variant"] = "NO_CORRESPONDENCE"
    snapshots.append(dict(snapshots[1]))
    assert len(index) == 3
    assert index.source_row_count == 3
    assert index[first_id]["scene_variant"] == original_scene
    with pytest.raises(TypeError):
        index.rows[first_id] = index[first_id]
    with pytest.raises(TypeError):
        index[first_id]["scene_variant"] = "NO_CORRESPONDENCE"
    with pytest.raises(AttributeError):
        index.source_row_count = 99


def test_three_snapshots_six_trials_bind_once_with_exact_pairing(
    tmp_path: Path,
) -> None:
    context = qualification_context(tmp_path)
    snapshots, trials = qualification_rows()
    index = build_canonical_snapshot_index(snapshots, execution_context=context)
    bindings = build_trial_snapshot_bindings(index, trials)

    assert len(index) == 3
    assert len(bindings.rows) == 6
    assert bindings.audit == {
        "schema_version": "trial_snapshot_binding_audit_v1",
        "canonical_snapshot_count": 3,
        "canonical_snapshot_source_row_count": 3,
        "planned_trial_count": 6,
        "unique_trial_count": 6,
        "backend_trial_counts": {
            "open3d_point_to_plane": 3,
            "pcl_point_to_plane": 3,
        },
        "native_trial_count": 0,
        "pairing_violation_count": 0,
        "CANONICAL_SNAPSHOT_INDEX_PASS": True,
        "TRIAL_TO_SNAPSHOT_BINDING_PASS": True,
        "SHARED_IDENTITY_VALIDATION_PASS": True,
    }
    for snapshot in snapshots:
        snapshot_id = str(snapshot["planned_snapshot_id"])
        open3d = bindings[f"{snapshot_id}::open3d_point_to_plane"]
        pcl = bindings[f"{snapshot_id}::pcl_point_to_plane"]
        assert open3d is index[snapshot_id]
        assert pcl is index[snapshot_id]
        assert open3d is pcl
    with pytest.raises(TypeError):
        bindings.rows["new"] = index[str(snapshots[0]["planned_snapshot_id"])]


@pytest.mark.parametrize("removed", SNAPSHOT_FIELDS)
def test_snapshot_bridge_rejects_every_missing_field(
    tmp_path: Path, removed: str
) -> None:
    context = qualification_context(tmp_path)
    snapshots, _trials = qualification_rows()
    row = {name: value for name, value in snapshots[0].items() if name != removed}
    _bridge_error(
        "SNAPSHOT_ROW_VALIDATION_FAILURE",
        lambda: build_canonical_snapshot_index([row], execution_context=context),
    )


@pytest.mark.parametrize("removed", TRIAL_FIELDS)
def test_trial_bridge_rejects_every_missing_field(
    tmp_path: Path, removed: str
) -> None:
    context = qualification_context(tmp_path)
    snapshots, trials = qualification_rows()
    index = build_canonical_snapshot_index(snapshots, execution_context=context)
    row = {name: value for name, value in trials[0].items() if name != removed}
    _bridge_error(
        "TRIAL_ROW_VALIDATION_FAILURE", lambda: bind_trial_to_snapshot(index, row)
    )


def test_bridge_rejects_extra_and_snapshot_only_trial_fields(tmp_path: Path) -> None:
    context = qualification_context(tmp_path)
    snapshots, trials = qualification_rows()
    index = build_canonical_snapshot_index(snapshots, execution_context=context)

    _bridge_error(
        "SNAPSHOT_ROW_VALIDATION_FAILURE",
        lambda: build_canonical_snapshot_index(
            [{**snapshots[0], "backend": BACKENDS[0]}],
            execution_context=context,
        ),
    )
    error = _bridge_error(
        "TRIAL_ROW_VALIDATION_FAILURE",
        lambda: bind_trial_to_snapshot(
            index, {**trials[0], "planned_backend_count": 2}
        ),
    )
    assert error.field == "planned_backend_count"
    _bridge_error(
        "TRIAL_ROW_VALIDATION_FAILURE",
        lambda: bind_trial_to_snapshot(index, {**trials[0], "unknown": True}),
    )


def test_duplicate_snapshot_and_trial_ids_fail_closed(tmp_path: Path) -> None:
    context = qualification_context(tmp_path)
    snapshots, trials = qualification_rows()
    _bridge_error(
        "DUPLICATE_SNAPSHOT_PLAN_ID",
        lambda: build_canonical_snapshot_index(
            [snapshots[0], dict(snapshots[0])], execution_context=context
        ),
    )
    index = build_canonical_snapshot_index(snapshots, execution_context=context)
    _bridge_error(
        "TRIAL_ROW_VALIDATION_FAILURE",
        lambda: build_trial_snapshot_bindings(index, [trials[0], dict(trials[0])]),
    )


def test_missing_snapshot_lookup_fails_closed(tmp_path: Path) -> None:
    context = qualification_context(tmp_path)
    snapshots, trials = qualification_rows()
    index = build_canonical_snapshot_index(snapshots, execution_context=context)
    missing = dict(trials[0])
    missing["planned_snapshot_id"] = "qualification::MISSING"
    missing["planned_trial_id"] = (
        f"{missing['planned_snapshot_id']}::{missing['backend']}"
    )
    _bridge_error(
        "SNAPSHOT_PLAN_ID_NOT_FOUND", lambda: bind_trial_to_snapshot(index, missing)
    )


@pytest.mark.parametrize(
    ("field", "replacement", "classification"),
    [
        ("scene_variant", "NONIDENTITY_REFERENCE", "TRIAL_SNAPSHOT_SCENE_MISMATCH"),
        ("condition", "SEED_FREE_ALTERNATE", "TRIAL_SNAPSHOT_CONDITION_MISMATCH"),
        (
            "geometry_seed",
            "OTHER_GEOMETRY_FIXTURE",
            "TRIAL_SNAPSHOT_GEOMETRY_SEED_MISMATCH",
        ),
        (
            "measurement_seed",
            "OTHER_MEASUREMENT_FIXTURE",
            "TRIAL_SNAPSHOT_MEASUREMENT_SEED_MISMATCH",
        ),
        ("repeat_index", 1, "TRIAL_SNAPSHOT_REPEAT_MISMATCH"),
    ],
)
def test_each_shared_identity_mismatch_has_a_specific_failure_classification(
    tmp_path: Path, field: str, replacement: object, classification: str
) -> None:
    context = qualification_context(tmp_path)
    snapshots, trials = qualification_rows()
    index = build_canonical_snapshot_index(snapshots, execution_context=context)
    changed = dict(trials[0])
    changed[field] = replacement
    error = _bridge_error(
        classification, lambda: bind_trial_to_snapshot(index, changed)
    )
    assert error.field == field


def test_incomplete_or_wrong_backend_pairing_is_never_accepted(tmp_path: Path) -> None:
    context = qualification_context(tmp_path)
    snapshots, trials = qualification_rows()
    index = build_canonical_snapshot_index(snapshots, execution_context=context)

    _bridge_error(
        "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH",
        lambda: build_trial_snapshot_bindings(index, trials[:-1]),
    )
    with pytest.raises(ExecutionContextError) as snapshot_subset:
        build_canonical_snapshot_index(
            snapshots[:-1], execution_context=context
        )
    assert snapshot_subset.value.classification == "EXECUTION_CONTEXT_PLAN_MISMATCH"
    reordered = [*trials[2:], *trials[:2]]
    with pytest.raises(ExecutionContextError) as trial_reordering:
        build_trial_snapshot_bindings(index, reordered)
    assert trial_reordering.value.classification == "EXECUTION_CONTEXT_PLAN_MISMATCH"
    invalid_backend = dict(trials[0])
    invalid_backend["backend"] = "native"
    invalid_backend["planned_trial_id"] = (
        f"{invalid_backend['planned_snapshot_id']}::native"
    )
    with pytest.raises(ExecutionContextError) as captured:
        bind_trial_to_snapshot(index, invalid_backend)
    assert getattr(captured.value, "classification", None) == (
        "TRIAL_ROW_VALIDATION_FAILURE"
    )


def test_context_binding_cannot_be_swapped_after_index_creation(tmp_path: Path) -> None:
    context = qualification_context(tmp_path)
    snapshots, _trials = qualification_rows()
    index = build_canonical_snapshot_index(snapshots, execution_context=context)
    with pytest.raises(AttributeError):
        index.execution_context = context
