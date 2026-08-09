from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from phase_a_harness.execution_context import (
    ExecutionContext,
    ExecutionContextError,
    ExecutionMode,
)
from phase_a_harness.formal_lifecycle_components import (
    FormalLifecycleComponentError,
)
from phase_a_harness.formal_lifecycle_contract import (
    FormalLifecycleContractError,
    FormalLifecycleSpec,
    deep_thaw,
)
from phase_a_harness.formal_lifecycle_paths import FormalLifecyclePathError
from phase_a_harness.runtime_lifecycle_fixture import (
    CONTEXT_A,
    CONTEXT_A_RUN_ID,
    CONTEXT_B,
    CONTEXT_B_RUN_ID,
    build_qualification_manifest,
    build_qualification_spec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "a" * 40


def _spec_pair(
    tmp_path: Path,
) -> tuple[FormalLifecycleSpec, FormalLifecycleSpec]:
    context_a_root = (tmp_path / "context_a").resolve()
    context_b_root = (tmp_path / "context_b").resolve()
    manifest_paths: dict[str, Path] = {}
    for name, runtime_root, branch, tag in (
        (
            CONTEXT_A,
            context_a_root,
            "qualification/context-a",
            "archive/qualification-context-a",
        ),
        (
            CONTEXT_B,
            context_b_root,
            "qualification/context-b",
            "archive/qualification-context-b",
        ),
    ):
        value = build_qualification_manifest(
            name,
            repository_root=REPOSITORY_ROOT,
            expected_commit=EXPECTED_COMMIT,
            expected_branch=branch,
            expected_tag=tag,
            runtime_root=runtime_root,
        )
        path = (tmp_path / f"{name}_manifest.json").resolve()
        path.write_text(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_paths[name] = path
    context_a = build_qualification_spec(
        CONTEXT_A,
        repository_root=REPOSITORY_ROOT,
        expected_commit=EXPECTED_COMMIT,
        expected_branch="qualification/context-a",
        expected_tag="archive/qualification-context-a",
        runtime_root=context_a_root,
        manifest_path=manifest_paths[CONTEXT_A],
    )
    context_b = build_qualification_spec(
        CONTEXT_B,
        repository_root=REPOSITORY_ROOT,
        expected_commit=EXPECTED_COMMIT,
        expected_branch="qualification/context-b",
        expected_tag="archive/qualification-context-b",
        runtime_root=context_b_root,
        manifest_path=manifest_paths[CONTEXT_B],
    )
    assert not context_a_root.exists()
    assert not context_b_root.exists()
    return context_a, context_b


def _assert_runtime_absent(*specs: FormalLifecycleSpec) -> None:
    for spec in specs:
        assert not spec.runtime_paths.runtime_root.exists()
        assert not spec.runtime_paths.single_writer_lease.exists()


def _formal_variant(context: ExecutionContext) -> ExecutionContext:
    """Create a local FORMAL-mode contract without loading any formal asset."""

    mode = ExecutionMode.FORMAL
    snapshot_schema = dataclasses.replace(context.snapshot_schema, mode=mode)
    trial_schema = dataclasses.replace(context.trial_schema, mode=mode)
    seed_policy = dataclasses.replace(context.allowed_seed_policy, mode=mode)
    id_policy = dataclasses.replace(context.id_policy, mode=mode)
    backend_policy = dataclasses.replace(context.backend_policy, mode=mode)
    reader_policy = dataclasses.replace(
        context.snapshot_reader_policy,
        mode=mode,
    )
    contract = dataclasses.replace(
        context.contract,
        mode=mode,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
    )
    return dataclasses.replace(
        context,
        mode=mode,
        contract=contract,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
    )


def _component_identity(spec: FormalLifecycleSpec) -> dict[str, tuple[str, ...]]:
    return {
        name: (
            binding.component_id,
            binding.callable_module,
            binding.callable_qualname,
            binding.implementation_sha256,
        )
        for name, binding in spec.components.component_bindings().items()
    }


def test_contexts_are_distinct_but_share_exact_component_implementations(
    tmp_path: Path,
) -> None:
    context_a, context_b = _spec_pair(tmp_path)

    assert context_a.run_id == CONTEXT_A_RUN_ID
    assert context_b.run_id == CONTEXT_B_RUN_ID
    assert context_a.run_id != context_b.run_id
    assert context_a.runtime_paths.runtime_root != context_b.runtime_paths.runtime_root
    assert context_a.execution_context.contract.contract_id != (
        context_b.execution_context.contract.contract_id
    )
    assert context_a.execution_context.plan_id != context_b.execution_context.plan_id
    assert context_a.execution_context.snapshot_schema.schema_id != (
        context_b.execution_context.snapshot_schema.schema_id
    )
    assert context_a.execution_context.trial_schema.schema_id != (
        context_b.execution_context.trial_schema.schema_id
    )
    assert context_a.manifest["protocol_version"] != (
        context_b.manifest["protocol_version"]
    )
    assert context_a.manifest["snapshot_plan_rows_sha256"] != (
        context_b.manifest["snapshot_plan_rows_sha256"]
    )
    assert context_a.manifest["trial_plan_rows_sha256"] != (
        context_b.manifest["trial_plan_rows_sha256"]
    )
    assert context_a.components.bundle_id != context_b.components.bundle_id
    assert context_a.publisher_inventory.inventory_id != (
        context_b.publisher_inventory.inventory_id
    )
    assert _component_identity(context_a) == _component_identity(context_b)
    assert all(row["geometry_seed"] is None for row in context_a.snapshot_plan)
    assert all(row["measurement_seed"] is None for row in context_b.snapshot_plan)
    _assert_runtime_absent(context_a, context_b)


_CROSS_CONTEXT_CASES = (
    "a_manifest_with_b_plans",
    "b_manifest_with_a_plans",
    "a_manifest_with_b_bundle",
    "b_manifest_with_a_bundle",
    "a_with_b_cache",
    "b_with_a_cache",
    "a_with_b_run_id",
    "b_with_a_run_id",
    "a_with_b_runtime_paths",
    "b_with_a_runtime_paths",
    "a_with_b_publisher_inventory",
    "b_with_a_publisher_inventory",
    "a_component_sha_mismatch",
    "b_component_sha_mismatch",
    "formal_mode_with_qualification_contract",
    "qualification_mode_with_formal_contract",
)


def _attempt_invalid_cross_binding(
    case: str,
    context_a: FormalLifecycleSpec,
    context_b: FormalLifecycleSpec,
) -> Any:
    if case == "a_manifest_with_b_plans":
        return dataclasses.replace(
            context_b,
            manifest=context_a.manifest,
        )
    if case == "b_manifest_with_a_plans":
        return dataclasses.replace(
            context_a,
            manifest=context_b.manifest,
        )
    if case == "a_manifest_with_b_bundle":
        return dataclasses.replace(
            context_a,
            component_bundle=context_b.component_bundle,
        )
    if case == "b_manifest_with_a_bundle":
        return dataclasses.replace(
            context_b,
            component_bundle=context_a.component_bundle,
        )
    if case == "a_with_b_cache":
        return dataclasses.replace(
            context_a.runtime_paths,
            snapshot_cache=context_b.runtime_paths.snapshot_cache,
        )
    if case == "b_with_a_cache":
        return dataclasses.replace(
            context_b.runtime_paths,
            snapshot_cache=context_a.runtime_paths.snapshot_cache,
        )
    if case == "a_with_b_run_id":
        return dataclasses.replace(context_a, run_id=context_b.run_id)
    if case == "b_with_a_run_id":
        return dataclasses.replace(context_b, run_id=context_a.run_id)
    if case == "a_with_b_runtime_paths":
        return dataclasses.replace(
            context_a,
            runtime_paths=context_b.runtime_paths,
        )
    if case == "b_with_a_runtime_paths":
        return dataclasses.replace(
            context_b,
            runtime_paths=context_a.runtime_paths,
        )
    if case == "a_with_b_publisher_inventory":
        return dataclasses.replace(
            context_a,
            publisher_inventory=context_b.publisher_inventory,
        )
    if case == "b_with_a_publisher_inventory":
        return dataclasses.replace(
            context_b,
            publisher_inventory=context_a.publisher_inventory,
        )
    if case in {"a_component_sha_mismatch", "b_component_sha_mismatch"}:
        spec = context_a if case.startswith("a_") else context_b
        manifest = deep_thaw(spec.manifest)
        manifest["formal_lifecycle_components"]["components"][
            "snapshot_reader"
        ]["implementation_sha256"] = "0" * 64
        return dataclasses.replace(spec, manifest=manifest)
    if case == "formal_mode_with_qualification_contract":
        return dataclasses.replace(context_a, mode=ExecutionMode.FORMAL)
    if case == "qualification_mode_with_formal_contract":
        return dataclasses.replace(
            context_a,
            execution_context=_formal_variant(context_a.execution_context),
        )
    raise AssertionError(f"unknown cross-context test case: {case}")


@pytest.mark.parametrize("case", _CROSS_CONTEXT_CASES)
def test_cross_context_bindings_fail_before_runtime_creation(
    tmp_path: Path,
    case: str,
) -> None:
    context_a, context_b = _spec_pair(tmp_path)

    with pytest.raises(
        (
            ExecutionContextError,
            FormalLifecycleComponentError,
            FormalLifecycleContractError,
            FormalLifecyclePathError,
        )
    ):
        _attempt_invalid_cross_binding(case, context_a, context_b)

    _assert_runtime_absent(context_a, context_b)
