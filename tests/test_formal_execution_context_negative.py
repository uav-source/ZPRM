from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from phase_a_harness import synthetic_confirmatory_v3_contract as formal_contract
from phase_a_harness import synthetic_confirmatory_v3_snapshot_builder as snapshot_reader
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
from phase_a_harness.runtime_lifecycle_io import canonical_json_bytes
from phase_a_harness.trial_snapshot_bridge import (
    TrialSnapshotBridgeError,
    build_canonical_snapshot_index,
    build_trial_snapshot_bindings,
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


def _qualification_context(
    tmp_path: Path, *, extra_shared_field: bool = False
) -> ExecutionContext:
    mode = ExecutionMode.QUALIFICATION
    runtime = (tmp_path / ("qualification_extra" if extra_shared_field else "qualification")).resolve()
    snapshot_fields = SNAPSHOT_FIELDS + (("fixture_revision",) if extra_shared_field else ())
    trial_fields = TRIAL_FIELDS + (("fixture_revision",) if extra_shared_field else ())

    def snapshot_validator(row):
        value = {name: row[name] for name in snapshot_fields}
        if (
            type(value["planned_snapshot_id"]) is not str
            or value["scene_variant"] not in SCENES
            or value["condition"] not in CONDITIONS
            or type(value["geometry_seed"]) is not str
            or type(value["measurement_seed"]) is not str
            or type(value["repeat_index"]) is not int
            or value["planned_backend_count"] != 2
            or value["replicate_semantics"] != "ONE_SEED_FREE_FIXTURE_INPUT"
            or (
                extra_shared_field
                and type(value["fixture_revision"]) is not str
            )
        ):
            raise ValueError("invalid qualification snapshot")
        return value

    def trial_validator(row):
        value = {name: row[name] for name in trial_fields}
        if (
            type(value["planned_trial_id"]) is not str
            or type(value["planned_snapshot_id"]) is not str
            or value["scene_variant"] not in SCENES
            or value["condition"] not in CONDITIONS
            or type(value["geometry_seed"]) is not str
            or type(value["measurement_seed"]) is not str
            or type(value["repeat_index"]) is not int
            or value["backend"] not in BACKENDS
            or value["planned_trial_id"]
            != f"{value['planned_snapshot_id']}::{value['backend']}"
            or (
                extra_shared_field
                and type(value["fixture_revision"]) is not str
            )
        ):
            raise ValueError("invalid qualification trial")
        return value

    snapshot_schema = SchemaBinding(
        mode, "qualification-snapshot-plan-v1", snapshot_fields
    )
    trial_schema = SchemaBinding(mode, "qualification-trial-plan-v1", trial_fields)
    seed_policy = SeedPolicy(mode, "qualification-no-seed-v1", False)
    id_policy = IdPolicy(mode, "qualification-id-v1")
    backend_policy = BackendPolicy(
        mode, "qualification-backends-v1", BACKENDS, 2
    )
    reader_policy = SnapshotReaderPolicy(
        mode=mode,
        policy_id="qualification-reader-v1",
        metadata_fields=tuple(sorted(snapshot_reader.METADATA_FIELDS)),
        lock_entry_fields=tuple(sorted(snapshot_reader._LOCK_ENTRY_FIELDS)),
        metadata_schema="qualification-metadata-v1",
        snapshot_schema_version="qualification-snapshot-v1",
        lineage_schema_version="qualification-lineage-v1",
        seed_namespace=None,
        snapshot_builder_contract_version="qualification-builder-v1",
        lineage_required_conditions=("SEED_FREE_ALTERNATE",),
        expected_rng_counts={condition: 0 for condition in CONDITIONS},
    )
    snapshot_plan, trial_plan = _rows(extra_shared_field=extra_shared_field)
    contract = ExecutionContract(
        mode=mode,
        contract_id="qualification-execution-contract-v1",
        plan_id="qualification-three-scene-plan-v1",
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
        snapshot_validator=snapshot_validator,
        trial_validator=trial_validator,
        result_route="QUALIFICATION_FIXTURE_ROUTING_V1",
    )
    return _context_from_contract(contract)


def _context_from_contract(
    contract: ExecutionContract, **changes
) -> ExecutionContext:
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


def _rows(
    *, extra_shared_field: bool = False
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    snapshots: list[dict[str, object]] = []
    trials: list[dict[str, object]] = []
    for scene in SCENES:
        snapshot_id = f"qualification::{scene}"
        snapshot: dict[str, object] = {
            "planned_snapshot_id": snapshot_id,
            "scene_variant": scene,
            "condition": "SEED_FREE_FIXTURE",
            "geometry_seed": "NO_GEOMETRY_SEED",
            "measurement_seed": "NO_MEASUREMENT_SEED",
            "repeat_index": 0,
            "planned_backend_count": 2,
            "replicate_semantics": "ONE_SEED_FREE_FIXTURE_INPUT",
        }
        if extra_shared_field:
            snapshot["fixture_revision"] = "fixture-v1"
        snapshots.append(snapshot)
        for backend in BACKENDS:
            trial = {
                "planned_trial_id": f"{snapshot_id}::{backend}",
                **{name: snapshot[name] for name in SHARED_FIELDS},
                "backend": backend,
            }
            if extra_shared_field:
                trial["fixture_revision"] = "fixture-v1"
            trials.append(trial)
    return snapshots, trials


def _dispatch_after_all_gates(
    context: ExecutionContext,
    snapshots: list[dict[str, object]],
    trials: list[dict[str, object]],
    backend_invocations: list[str],
    *,
    authenticate=None,
) -> None:
    index = build_canonical_snapshot_index(
        snapshots, execution_context=context
    )
    bindings = build_trial_snapshot_bindings(index, trials)
    authenticated: set[str] = set()
    for trial in trials:
        snapshot = bindings.rows[str(trial["planned_trial_id"])]
        snapshot_id = str(snapshot["planned_snapshot_id"])
        if authenticate is not None and snapshot_id not in authenticated:
            authenticate(snapshot)
            authenticated.add(snapshot_id)
    for trial in trials:
        backend_invocations.append(str(trial["planned_trial_id"]))


def _before_backend(operation, backend_invocations: list[str]) -> None:
    operation()
    backend_invocations.append("FORBIDDEN_BACKEND_INVOCATION")


def _write_checksum_mismatch_snapshot(
    context: ExecutionContext, row: dict[str, object]
) -> None:
    directory = context.cache_root / str(row["planned_snapshot_id"])
    directory.mkdir(parents=True)
    for filename in snapshot_reader.BASE_ARRAY_FILENAMES:
        (directory / filename).write_bytes(b"qualification-not-an-array")
    metadata = {name: None for name in context.snapshot_reader_policy.metadata_fields}
    metadata.update(
        {
            "array_file_sha256": {
                filename: "0" * 64
                for filename in snapshot_reader.BASE_ARRAY_FILENAMES
            },
            "condition": row["condition"],
            "geometry_seed": row["geometry_seed"],
            "measurement_seed": row["measurement_seed"],
            "planned_snapshot_id": row["planned_snapshot_id"],
            "repeat_index": row["repeat_index"],
            "scene_variant": row["scene_variant"],
            "schema_version": context.snapshot_reader_policy.metadata_schema,
            "seed_namespace": context.snapshot_reader_policy.seed_namespace,
            "snapshot_builder_contract_version": (
                context.snapshot_reader_policy.snapshot_builder_contract_version
            ),
            "snapshot_id": row["planned_snapshot_id"],
            "snapshot_schema_version": (
                context.snapshot_reader_policy.snapshot_schema_version
            ),
            "lineage_schema_version": (
                context.snapshot_reader_policy.lineage_schema_version
            ),
        }
    )
    unsigned = dict(metadata)
    unsigned.pop("metadata_payload_sha256")
    metadata["metadata_payload_sha256"] = snapshot_reader._canonical_sha256(unsigned)
    (directory / "metadata.json").write_bytes(canonical_json_bytes(metadata))


def _mutated_trials(
    trials: list[dict[str, object]], field: str, value: object
) -> list[dict[str, object]]:
    changed = copy.deepcopy(trials)
    changed[0][field] = value
    return changed


def test_twenty_negative_context_bridge_and_reader_cases_dispatch_zero_backends(
    tmp_path: Path,
) -> None:
    context = _qualification_context(tmp_path)
    snapshots, trials = _rows()
    backend_invocations: list[str] = []

    missing_snapshot_trials = copy.deepcopy(trials)
    missing_snapshot_trials[0]["planned_snapshot_id"] = "qualification::MISSING"
    missing_snapshot_trials[0]["planned_trial_id"] = (
        "qualification::MISSING::open3d_point_to_plane"
    )
    duplicate_snapshots = [copy.deepcopy(snapshots[0]), copy.deepcopy(snapshots[0])]
    missing_snapshot_only = copy.deepcopy(snapshots)
    missing_snapshot_only[0].pop("planned_backend_count")
    missing_trial_only = copy.deepcopy(trials)
    missing_trial_only[0].pop("backend")
    invalid_backend = copy.deepcopy(trials)
    invalid_backend[0]["backend"] = "native"
    invalid_backend[0]["planned_trial_id"] = (
        f"{invalid_backend[0]['planned_snapshot_id']}::native"
    )

    extra_context = _qualification_context(tmp_path, extra_shared_field=True)
    extra_snapshots, extra_trials = _rows(extra_shared_field=True)
    other_shared_mismatch = copy.deepcopy(extra_trials)
    other_shared_mismatch[0]["fixture_revision"] = "fixture-v2"

    corrupt_context = _qualification_context(tmp_path / "corrupt")
    corrupt_snapshots, corrupt_trials = _rows()
    corrupt_directory = (
        corrupt_context.cache_root
        / str(corrupt_snapshots[0]["planned_snapshot_id"])
    )
    corrupt_directory.mkdir(parents=True)

    checksum_context = _qualification_context(tmp_path / "checksum")
    checksum_snapshots, checksum_trials = _rows()
    _write_checksum_mismatch_snapshot(checksum_context, checksum_snapshots[0])

    split_backend_trials = [copy.deepcopy(trials[0]), copy.deepcopy(trials[3])]
    formal_context = formal_contract.formal_execution_context()
    formal_row = formal_contract.typed_snapshot_rows(
        Path(__file__).resolve().parents[1] / formal_contract.SNAPSHOT_PLAN_RELATIVE
    )[0]

    mismatched_schema = replace(
        context.snapshot_schema, schema_id="qualification-other-schema-v1"
    )
    mismatched_contract = replace(context.contract, snapshot_schema=mismatched_schema)
    tamper_index = build_canonical_snapshot_index(
        snapshots, execution_context=context
    )

    cases = [
        (
            "missing snapshot ID",
            lambda: _dispatch_after_all_gates(
                context, snapshots, missing_snapshot_trials, backend_invocations
            ),
            TrialSnapshotBridgeError,
            "SNAPSHOT_PLAN_ID_NOT_FOUND",
        ),
        (
            "duplicate snapshot ID",
            lambda: _dispatch_after_all_gates(
                context, duplicate_snapshots, [], backend_invocations
            ),
            TrialSnapshotBridgeError,
            "DUPLICATE_SNAPSHOT_PLAN_ID",
        ),
        (
            "scene mismatch",
            lambda: _dispatch_after_all_gates(
                context,
                snapshots,
                _mutated_trials(trials, "scene_variant", "NONIDENTITY_REFERENCE"),
                backend_invocations,
            ),
            TrialSnapshotBridgeError,
            "TRIAL_SNAPSHOT_SCENE_MISMATCH",
        ),
        (
            "condition mismatch",
            lambda: _dispatch_after_all_gates(
                context,
                snapshots,
                _mutated_trials(trials, "condition", "SEED_FREE_ALTERNATE"),
                backend_invocations,
            ),
            TrialSnapshotBridgeError,
            "TRIAL_SNAPSHOT_CONDITION_MISMATCH",
        ),
        (
            "geometry identity mismatch",
            lambda: _dispatch_after_all_gates(
                context,
                snapshots,
                _mutated_trials(trials, "geometry_seed", "OTHER_GEOMETRY_FIXTURE"),
                backend_invocations,
            ),
            TrialSnapshotBridgeError,
            "TRIAL_SNAPSHOT_GEOMETRY_SEED_MISMATCH",
        ),
        (
            "measurement identity mismatch",
            lambda: _dispatch_after_all_gates(
                context,
                snapshots,
                _mutated_trials(
                    trials, "measurement_seed", "OTHER_MEASUREMENT_FIXTURE"
                ),
                backend_invocations,
            ),
            TrialSnapshotBridgeError,
            "TRIAL_SNAPSHOT_MEASUREMENT_SEED_MISMATCH",
        ),
        (
            "repeat mismatch",
            lambda: _dispatch_after_all_gates(
                context,
                snapshots,
                _mutated_trials(trials, "repeat_index", 1),
                backend_invocations,
            ),
            TrialSnapshotBridgeError,
            "TRIAL_SNAPSHOT_REPEAT_MISMATCH",
        ),
        (
            "other shared identity mismatch",
            lambda: _dispatch_after_all_gates(
                extra_context,
                extra_snapshots,
                other_shared_mismatch,
                backend_invocations,
            ),
            TrialSnapshotBridgeError,
            "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH",
        ),
        (
            "missing snapshot-only field",
            lambda: _dispatch_after_all_gates(
                context, missing_snapshot_only, trials, backend_invocations
            ),
            TrialSnapshotBridgeError,
            "SNAPSHOT_ROW_VALIDATION_FAILURE",
        ),
        (
            "missing trial-only field",
            lambda: _dispatch_after_all_gates(
                context, snapshots, missing_trial_only, backend_invocations
            ),
            TrialSnapshotBridgeError,
            "TRIAL_ROW_VALIDATION_FAILURE",
        ),
        (
            "invalid backend",
            lambda: _dispatch_after_all_gates(
                context, snapshots, invalid_backend, backend_invocations
            ),
            ExecutionContextError,
            "TRIAL_ROW_VALIDATION_FAILURE",
        ),
        (
            "corrupt qualification snapshot inventory",
            lambda: _dispatch_after_all_gates(
                    corrupt_context,
                    corrupt_snapshots,
                    corrupt_trials,
                backend_invocations,
                authenticate=lambda row: snapshot_reader.read_v3_snapshot(
                    corrupt_context.cache_root,
                    row,
                    execution_context=corrupt_context,
                    expected_lock_entry=None,
                    arrays=False,
                ),
            ),
            snapshot_reader.V3SnapshotContractError,
            None,
        ),
        (
            "qualification snapshot checksum mismatch",
            lambda: _dispatch_after_all_gates(
                    checksum_context,
                    checksum_snapshots,
                    checksum_trials,
                backend_invocations,
                authenticate=lambda row: snapshot_reader.read_v3_snapshot(
                    checksum_context.cache_root,
                    row,
                    execution_context=checksum_context,
                    expected_lock_entry=None,
                    arrays=False,
                ),
            ),
            snapshot_reader.V3SnapshotContractError,
            None,
        ),
        (
            "Open3D and PCL bind different snapshots",
            lambda: _dispatch_after_all_gates(
                context, snapshots, split_backend_trials, backend_invocations
            ),
            TrialSnapshotBridgeError,
            "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH",
        ),
        (
            "formal mode with qualification contract",
            lambda: _before_backend(
                lambda: _context_from_contract(
                    context.contract, mode=ExecutionMode.FORMAL
                ),
                backend_invocations,
            ),
            ExecutionContextError,
            "EXECUTION_CONTEXT_MODE_MISMATCH",
        ),
        (
            "qualification mode with formal contract",
            lambda: _before_backend(
                lambda: _context_from_contract(
                    formal_context.contract, mode=ExecutionMode.QUALIFICATION
                ),
                backend_invocations,
            ),
            ExecutionContextError,
            "EXECUTION_CONTEXT_MODE_MISMATCH",
        ),
        (
            "qualification context with formal cache",
            lambda: _before_backend(
                lambda: snapshot_reader.read_v3_snapshot(
                    formal_contract.RUNTIME_PATHS["snapshot_cache_path"],
                    snapshots[0],
                    execution_context=context,
                    expected_lock_entry=None,
                    arrays=False,
                ),
                backend_invocations,
            ),
            ExecutionContextError,
            "EXECUTION_CONTEXT_CACHE_MISMATCH",
        ),
        (
            "formal context with qualification cache",
            lambda: _before_backend(
                lambda: snapshot_reader.read_v3_snapshot(
                    context.cache_root,
                    formal_row,
                    execution_context=formal_context,
                    expected_lock_entry=None,
                    arrays=False,
                ),
                backend_invocations,
            ),
            ExecutionContextError,
            "EXECUTION_CONTEXT_CACHE_MISMATCH",
        ),
        (
            "contract and schema mismatch",
            lambda: _before_backend(
                lambda: _context_from_contract(
                    mismatched_contract, snapshot_schema=context.snapshot_schema
                ),
                backend_invocations,
            ),
            ExecutionContextError,
            "EXECUTION_CONTEXT_SCHEMA_MISMATCH",
        ),
        (
            "canonical index post-build tamper",
            lambda: _before_backend(
                lambda: tamper_index.rows.__setitem__(
                    str(snapshots[0]["planned_snapshot_id"]),
                    tamper_index[str(snapshots[0]["planned_snapshot_id"])],
                ),
                backend_invocations,
            ),
            (AttributeError, TypeError),
            None,
        ),
    ]

    assert len(cases) == 20
    observed: list[str] = []
    for name, operation, error_type, classification in cases:
        with pytest.raises(error_type) as captured:
            operation()
        if classification is not None:
            assert getattr(captured.value, "classification", None) == classification
        assert backend_invocations == [], name
        observed.append(name)
    assert len(observed) == len(set(observed)) == 20
