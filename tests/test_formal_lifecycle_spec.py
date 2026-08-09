from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pytest

from phase_a_harness.execution_context import (
    BackendPolicy,
    canonical_plan_rows_sha256,
    ExecutionContext,
    ExecutionContract,
    ExecutionMode,
    IdPolicy,
    SchemaBinding,
    SeedPolicy,
    SnapshotReaderPolicy,
)
from phase_a_harness.formal_lifecycle_components import (
    BACKEND_IDS,
    ComponentBinding,
    FormalLifecycleComponents,
)
from phase_a_harness.formal_lifecycle_contract import (
    CommandProfile,
    FormalLifecycleContractError,
    FormalLifecycleSpec,
    GitIdentityPolicy,
    PublisherInventory,
    deep_thaw,
    validate_formal_lifecycle_spec,
)
from phase_a_harness.formal_lifecycle_paths import FormalLifecyclePaths
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


def _validate_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in SNAPSHOT_FIELDS}


def _validate_trial(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in TRIAL_FIELDS}


def _load_fixture_component(implementation_path: Path) -> Any:
    implementation_path.write_text(
        "def component(**kwargs):\n"
        "    return dict(kwargs)\n",
        encoding="utf-8",
    )
    module_name = (
        "_formal_lifecycle_fixture_"
        + hashlib.sha256(str(implementation_path).encode("utf-8")).hexdigest()[:16]
    )
    module_spec = importlib.util.spec_from_file_location(
        module_name, implementation_path
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module.component


def _runtime_paths(tmp_path: Path) -> FormalLifecyclePaths:
    runtime_root = (tmp_path / "runtime").resolve()
    return FormalLifecyclePaths(
        runtime_root=runtime_root,
        snapshot_cache=runtime_root / "snapshot_cache",
        snapshot_lock=runtime_root / "snapshot_lock.json",
        raw_results=runtime_root / "raw_results",
        raw_manifest=runtime_root / "raw_manifest.json",
        event_log=runtime_root / "event_log.ndjson",
        backend_temporary=runtime_root / "backend_temporary",
        analysis=runtime_root / "analysis",
        verification=runtime_root / "verification",
        publisher_staging=runtime_root / "publisher_staging",
        artifact_staging=runtime_root / "artifact_staging",
        temporary_inventory=runtime_root / "temporary_inventory",
        run_manifest=runtime_root / "run_manifest.json",
        formal_command_log=runtime_root / "formal_command.log",
        formal_command_sha256=runtime_root / "formal_command.log.sha256",
        immutable_run_lock=runtime_root / "immutable_run_lock.json",
        single_writer_lease=(tmp_path / "leases" / "formal.lease").resolve(),
    )


def _execution_context(
    *,
    repository_root: Path,
    runtime_paths: FormalLifecyclePaths,
    snapshot_plan: tuple[Mapping[str, Any], ...],
    trial_plan: tuple[Mapping[str, Any], ...],
) -> ExecutionContext:
    mode = ExecutionMode.QUALIFICATION
    snapshot_schema = SchemaBinding(
        mode=mode,
        schema_id="formal-lifecycle-fixture-snapshot-plan-v1",
        fields=SNAPSHOT_FIELDS,
    )
    trial_schema = SchemaBinding(
        mode=mode,
        schema_id="formal-lifecycle-fixture-trial-plan-v1",
        fields=TRIAL_FIELDS,
    )
    seed_policy = SeedPolicy(
        mode=mode,
        policy_id="formal-lifecycle-fixture-seed-policy-v1",
        confirmatory_seed_allowed=False,
    )
    id_policy = IdPolicy(
        mode=mode,
        policy_id="formal-lifecycle-fixture-id-policy-v1",
    )
    backend_policy = BackendPolicy(
        mode=mode,
        policy_id="formal-lifecycle-fixture-backend-policy-v1",
        allowed_backends=BACKEND_IDS,
        planned_backend_count=2,
    )
    snapshot_reader_policy = SnapshotReaderPolicy(
        mode=mode,
        policy_id="formal-lifecycle-fixture-reader-policy-v1",
        metadata_fields=("schema_version", "snapshot_id"),
        lock_entry_fields=("snapshot_id", "file_sha256"),
        metadata_schema="formal-lifecycle-fixture-metadata-v1",
        snapshot_schema_version="formal-lifecycle-fixture-snapshot-v1",
        lineage_schema_version="formal-lifecycle-fixture-lineage-v1",
        seed_namespace=None,
        snapshot_builder_contract_version="formal-lifecycle-fixture-builder-v1",
        lineage_required_conditions=("FIXTURE_IDENTITY",),
        expected_rng_counts={"FIXTURE_IDENTITY": 0},
    )
    contract = ExecutionContract(
        mode=mode,
        contract_id="formal-lifecycle-fixture-contract-v1",
        plan_id="formal-lifecycle-fixture-plan",
        expected_snapshot_plan_sha256=canonical_plan_rows_sha256(snapshot_plan),
        expected_trial_plan_sha256=canonical_plan_rows_sha256(trial_plan),
        expected_snapshot_count=1,
        expected_trial_count=2,
        expected_cache_root=runtime_paths.snapshot_cache,
        expected_runtime_root=runtime_paths.runtime_root,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=("IDENTITY",),
        allowed_conditions=("FIXTURE_IDENTITY",),
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=snapshot_reader_policy,
        snapshot_validator=_validate_snapshot,
        trial_validator=_validate_trial,
        result_route="QUALIFICATION_FIXTURE_ROUTE",
    )
    return ExecutionContext(
        mode=mode,
        contract=contract,
        plan_id=contract.plan_id,
        cache_root=runtime_paths.snapshot_cache,
        runtime_root=runtime_paths.runtime_root,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=contract.allowed_scenes,
        allowed_conditions=contract.allowed_conditions,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=snapshot_reader_policy,
    )


_COMPONENT_FIELDS = (
    "snapshot_materializer",
    "snapshot_reader",
    "snapshot_validator",
    "trial_validator",
    "snapshot_lock_builder",
    "snapshot_lock_validator",
    "backend_input_builder",
    "common_record_builder",
    "result_validator",
    "resume_result_validator",
    "primary_analyzer",
    "independent_verifier",
    "difference_auditor",
    "publisher",
    "artifact_verifier",
    "git_gate",
    "run_contract_builder",
)


def _component_bundle(
    *,
    binding: ComponentBinding,
    context: ExecutionContext,
    runtime_paths: FormalLifecyclePaths,
    manifest_sha256: str,
    run_id: str,
    publisher_inventory_sha256: str,
) -> FormalLifecycleComponents:
    component_bindings = {field: binding for field in _COMPONENT_FIELDS}
    return FormalLifecycleComponents(
        bundle_id="formal-lifecycle-fixture-components-v1",
        bound_mode=context.mode,
        bound_contract_id=context.contract.contract_id,
        bound_plan_id=context.contract.plan_id,
        bound_manifest_sha256=manifest_sha256,
        bound_run_id=run_id,
        bound_runtime_root=runtime_paths.runtime_root,
        bound_publisher_inventory_sha256=publisher_inventory_sha256,
        backend_registry={backend: binding for backend in BACKEND_IDS},
        **component_bindings,
    )


def _build_spec(
    tmp_path: Path,
    *,
    component_manifest_mutation: tuple[str, str] | None = None,
) -> FormalLifecycleSpec:
    repository_root = (tmp_path / "fixture_repository").resolve()
    repository_root.mkdir()
    entry_script = repository_root / "run_fixture.py"
    entry_script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    implementation_path = repository_root / "fixture_components.py"
    component_callable = _load_fixture_component(implementation_path)
    implementation_sha256 = hashlib.sha256(
        implementation_path.read_bytes()
    ).hexdigest()
    binding = ComponentBinding(
        component_id="fixture_component",
        callable=component_callable,
        implementation_path=implementation_path,
        implementation_sha256=implementation_sha256,
    )

    runtime_paths = _runtime_paths(tmp_path)
    snapshot_plan: tuple[Mapping[str, Any], ...] = (
        {
            "planned_snapshot_id": "snapshot-a",
            "scene_variant": "IDENTITY",
            "condition": "FIXTURE_IDENTITY",
            "planned_backend_count": 2,
            "replicate_semantics": "SEED_FREE",
        },
    )
    trial_plan: tuple[Mapping[str, Any], ...] = tuple(
        {
            "planned_trial_id": f"snapshot-a::{backend}",
            "planned_snapshot_id": "snapshot-a",
            "scene_variant": "IDENTITY",
            "condition": "FIXTURE_IDENTITY",
            "backend": backend,
        }
        for backend in BACKEND_IDS
    )
    context = _execution_context(
        repository_root=repository_root,
        runtime_paths=runtime_paths,
        snapshot_plan=snapshot_plan,
        trial_plan=trial_plan,
    )
    identity_policy = GitIdentityPolicy(
        policy_id="formal-lifecycle-fixture-git-policy-v1",
        expected_branch="fixture/formal-lifecycle",
        expected_commit="a" * 40,
        expected_tag="archive/formal-lifecycle-fixture",
        checkpoint_prefix="checkpoint/formal-lifecycle-fixture",
        require_clean_worktree=True,
    )
    publisher_inventory = PublisherInventory(
        inventory_id="formal-lifecycle-fixture-inventory-v1",
        tables=tuple(f"table_{index}.csv" for index in range(7)),
        figures=tuple(f"figure_{index}.png" for index in range(3)),
        root_files=tuple(f"root_{index}.json" for index in range(7)),
    )
    command_profile = CommandProfile(
        profile_id="formal-lifecycle-fixture-command-v1",
        entry_script=Path("run_fixture.py"),
        environment={"PYTHONNOUSERSITE": "1"},
        argv_prefix=("--manifest", "--run-id", "--output-dir"),
        unset_environment=("PYTHONPATH",),
    )
    run_id = "formal-lifecycle-fixture-run"
    provisional_bundle = _component_bundle(
        binding=binding,
        context=context,
        runtime_paths=runtime_paths,
        manifest_sha256="0" * 64,
        run_id=run_id,
        publisher_inventory_sha256=publisher_inventory.payload_sha256,
    )
    manifest: dict[str, Any] = {
        "execution_mode": context.mode.value,
        "run_id": run_id,
        "plan_id": context.contract.plan_id,
        "snapshot_plan_rows_sha256": canonical_plan_rows_sha256(snapshot_plan),
        "trial_plan_rows_sha256": canonical_plan_rows_sha256(trial_plan),
        "publisher_inventory_sha256": publisher_inventory.payload_sha256,
        "command_profile_sha256": command_profile.payload_sha256,
        "formal_lifecycle_paths": runtime_paths.as_dict(),
        "formal_lifecycle_components": provisional_bundle.manifest_binding(
            repository_root=repository_root
        ),
        "git_identity_policy": identity_policy.manifest_binding(),
        "publisher_inventory": publisher_inventory.report(),
        "command_profile": command_profile.report(),
    }
    if component_manifest_mutation is not None:
        field, value = component_manifest_mutation
        manifest["formal_lifecycle_components"]["components"][
            "snapshot_reader"
        ][field] = value
    manifest_path = repository_root / "formal_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    component_bundle = dataclasses.replace(
        provisional_bundle,
        bound_manifest_sha256=manifest_sha256,
    )
    return FormalLifecycleSpec(
        repository_root=repository_root,
        execution_context=context,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        runtime_paths=runtime_paths,
        snapshot_plan=snapshot_plan,
        trial_plan=trial_plan,
        component_bundle=component_bundle,
        mode=context.mode,
        run_id=run_id,
        workers=1,
        identity_policy=identity_policy,
        publisher_inventory=publisher_inventory,
        command_profile=command_profile,
    )


def test_formal_lifecycle_spec_is_deeply_immutable(tmp_path: Path) -> None:
    spec = _build_spec(tmp_path)

    assert isinstance(spec.manifest, MappingProxyType)
    assert isinstance(spec.snapshot_plan, tuple)
    assert isinstance(spec.snapshot_plan[0], MappingProxyType)
    assert isinstance(spec.trial_plan[0], MappingProxyType)
    assert isinstance(spec.component_bundle.backend_registry, MappingProxyType)

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.workers = 4  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.manifest["run_id"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.snapshot_plan[0]["planned_snapshot_id"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.component_bundle.backend_registry["native"] = object()  # type: ignore[index]

    thawed_manifest = deep_thaw(spec.manifest)
    thawed_manifest["run_id"] = "mutated-copy"
    assert spec.manifest["run_id"] == "formal-lifecycle-fixture-run"


def test_validated_spec_preserves_immutable_canonical_plan(
    tmp_path: Path,
) -> None:
    validated = validate_formal_lifecycle_spec(_build_spec(tmp_path))

    assert isinstance(validated.canonical_snapshot_index.rows, MappingProxyType)
    assert isinstance(validated.trial_snapshot_bindings.rows, MappingProxyType)
    assert isinstance(validated.spec.snapshot_plan[0], MappingProxyType)
    assert isinstance(validated.spec.trial_plan[0], MappingProxyType)
    with pytest.raises(TypeError):
        validated.canonical_snapshot_index.rows["new"] = {}  # type: ignore[index]


def test_spec_rejects_in_memory_manifest_that_differs_from_bound_file(
    tmp_path: Path,
) -> None:
    spec = _build_spec(tmp_path)
    mutated = deep_thaw(spec.manifest)
    mutated["run_id"] = "different-in-memory-run"

    with pytest.raises(
        FormalLifecycleContractError,
        match="in-memory manifest differs",
    ):
        dataclasses.replace(spec, manifest=mutated)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("component_id", "different_component"),
        ("callable_module", "different.module"),
        ("callable_qualname", "different_qualname"),
    ),
)
def test_spec_rejects_component_identity_mismatch_before_entry(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    with pytest.raises(
        FormalLifecycleContractError,
        match="manifest component binding",
    ):
        _build_spec(
            tmp_path,
            component_manifest_mutation=(field, value),
        )
