from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping

import pytest

from phase_a_harness.execution_context import ExecutionContextError
from phase_a_harness import synthetic_confirmatory_v3_contract as contract
from phase_a_harness import synthetic_confirmatory_v3_snapshot_builder as reader
from phase_a_harness.trial_snapshot_bridge import (
    build_canonical_snapshot_index,
    build_trial_snapshot_bindings,
)


REPOSITORY = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "1f8c4254064c90c2c995b0c75d6b9f21a933be88"
CONTRACT_PATH = "src/phase_a_harness/synthetic_confirmatory_v3_contract.py"
READER_PATH = "src/phase_a_harness/synthetic_confirmatory_v3_snapshot_builder.py"
SNAPSHOT_PLAN = REPOSITORY / contract.SNAPSHOT_PLAN_RELATIVE
TRIAL_PLAN = REPOSITORY / contract.TRIAL_PLAN_RELATIVE
SNAPSHOT_PLAN_SHA256 = (
    "dbe75e9df8545b1c61c61bac2baa3ee2c525aeda65b9db10215c61863b3b29ef"
)
TRIAL_PLAN_SHA256 = (
    "f5b2f6fb84e3a64e686eacfc22c8ec90d6bc9c14860116e8b51f4d628a77b8d4"
)
SNAPSHOT_IDENTITY_SHA256 = (
    "190d66851db105c7209054dfbca67cc63e0832b14204cdf410eb9d1e7f6bd1f3"
)
TRIAL_IDENTITY_SHA256 = (
    "ee07eb1dcc3c2ee0a413e581443b200c5fe480cbf9632059244e6a664231db82"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_source(path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:{path}"],
        cwd=REPOSITORY,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        source_state = (REPOSITORY / "SOURCE_STATE.txt").read_text(
            encoding="utf-8"
        )
        if "formal_runtime_artifacts_included=false" in source_state:
            pytest.skip(
                "source-only ZIP excludes the historical Git baseline object"
            )
        result.check_returncode()
    return result.stdout


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _assignment(tree: ast.Module, name: str) -> ast.stmt:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node
    raise AssertionError(f"assignment not found: {name}")


def _dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _scientific_body(function: ast.FunctionDef) -> str:
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return _dump(ast.Module(body=body, type_ignores=[]))


def _baseline_snapshot_validator(row: Mapping[str, Any]) -> dict[str, Any]:
    if not set(contract.SNAPSHOT_FIELDS).issubset(row):
        raise ValueError("v3 snapshot plan row is incomplete")
    value = {
        "planned_snapshot_id": str(row["planned_snapshot_id"]),
        **contract.snapshot_identity(row),
        "planned_backend_count": int(row["planned_backend_count"]),
        "replicate_semantics": str(row["replicate_semantics"]),
    }
    condition = value["condition"]
    measurement = value["measurement_seed"]
    repeat = value["repeat_index"]
    if (
        value["scene_variant"] not in contract.SCENES
        or condition not in contract.CONDITIONS
        or value["geometry_seed"] not in contract.GEOMETRY_SEEDS
        or value["planned_backend_count"] != 2
        or value["planned_snapshot_id"] != contract.expected_snapshot_id(value)
    ):
        raise ValueError("v3 snapshot identity is outside the frozen design")
    if condition == "IDEAL_MATCHED":
        expected_semantics = "ONE_CONTROL_INPUT"
        valid = measurement is None and repeat == 0
    elif condition == "INDEPENDENT_NOISE_FREE":
        expected_semantics = "ONE_DETERMINISTIC_INDEPENDENT_INPUT"
        valid = measurement is None and repeat == 0
    else:
        expected_semantics = "FIFTEEN_STOCHASTIC_INPUTS_PER_SCENE_GEOMETRY"
        valid = measurement in contract.MEASUREMENT_SEEDS and repeat in range(5)
    if not valid or value["replicate_semantics"] != expected_semantics:
        raise ValueError("v3 snapshot replicate semantics changed")
    return value


def _accepted(operation) -> tuple[bool, object]:
    try:
        return True, operation()
    except (ExecutionContextError, KeyError, TypeError, ValueError, RuntimeError) as error:
        return False, type(error).__name__


def test_formal_snapshot_validator_scientific_body_is_baseline_exact() -> None:
    baseline = ast.parse(_git_source(CONTRACT_PATH))
    current = ast.parse((REPOSITORY / CONTRACT_PATH).read_text(encoding="utf-8"))
    assert _scientific_body(
        _function(current, "_validate_formal_snapshot_plan_row")
    ) == _scientific_body(_function(baseline, "validate_snapshot_plan_row"))


def test_formal_snapshot_validator_behavior_matches_baseline_oracle() -> None:
    context = contract.formal_execution_context()
    rows = contract.typed_snapshot_rows(SNAPSHOT_PLAN)

    for row in rows:
        expected = _baseline_snapshot_validator(row)
        assert context.validate_snapshot_row(row) == expected

    representative = dict(rows[0])
    coercible = {
        **representative,
        "geometry_seed": str(representative["geometry_seed"]),
        "measurement_seed": "",
        "repeat_index": str(representative["repeat_index"]),
        "planned_backend_count": "2",
        "baseline_ignored_extra_field": "preserved compatibility",
    }
    assert context.validate_snapshot_row(coercible) == _baseline_snapshot_validator(
        coercible
    )

    invalid_rows = []
    for field, value in (
        ("planned_snapshot_id", "wrong"),
        ("scene_variant", "UNKNOWN_SCENE"),
        ("condition", "UNKNOWN_CONDITION"),
        ("geometry_seed", -1),
        ("measurement_seed", -1),
        ("repeat_index", 9),
        ("planned_backend_count", 1),
        ("replicate_semantics", "UNKNOWN_SEMANTICS"),
    ):
        invalid_rows.append({**representative, field: value})
    invalid_rows.append(
        {
            name: value
            for name, value in representative.items()
            if name != "planned_snapshot_id"
        }
    )
    for row in invalid_rows:
        baseline_accepts, _baseline_value = _accepted(
            lambda row=row: _baseline_snapshot_validator(row)
        )
        candidate_accepts, _candidate_value = _accepted(
            lambda row=row: context.validate_snapshot_row(row)
        )
        assert candidate_accepts is baseline_accepts


def test_reader_policy_and_authentication_primitives_are_static_baseline_equivalent(
) -> None:
    baseline_contract = ast.parse(_git_source(CONTRACT_PATH))
    current_contract = ast.parse(
        (REPOSITORY / CONTRACT_PATH).read_text(encoding="utf-8")
    )
    baseline_reader = ast.parse(_git_source(READER_PATH))
    current_reader = ast.parse(
        (REPOSITORY / READER_PATH).read_text(encoding="utf-8")
    )

    for name in (
        "NAMESPACE",
        "SNAPSHOT_SCHEMA",
        "METADATA_SCHEMA",
        "LINEAGE_SCHEMA",
        "RUNTIME_PATHS",
    ):
        assert _dump(_assignment(current_contract, name)) == _dump(
            _assignment(baseline_contract, name)
        )
    for name in (
        "SNAPSHOT_BUILDER_CONTRACT_VERSION",
        "PARENT_INDEX_FILENAME",
        "BASE_ARRAY_FILENAMES",
        "METADATA_FIELDS",
        "_LOCK_ENTRY_FIELDS",
    ):
        assert _dump(_assignment(current_reader, name)) == _dump(
            _assignment(baseline_reader, name)
        )
    for name in (
        "_canonical_sha256",
        "_raw_sha256",
        "_strict_object",
        "_canonical_absolute",
        "_regular_file_sha256",
        "_load_npy",
        "_snapshot_inventory",
        "_lineage_recomputed",
        "_validate_array_contract",
    ):
        assert _dump(_function(current_reader, name)) == _dump(
            _function(baseline_reader, name)
        )

    policy = contract.formal_execution_context().snapshot_reader_policy
    assert policy.report() == {
        "mode": "FORMAL",
        "policy_id": "synthetic_confirmatory_v3_formal_reader_policy_v1",
        "metadata_fields": sorted(reader.METADATA_FIELDS),
        "lock_entry_fields": sorted(reader._LOCK_ENTRY_FIELDS),
        "metadata_schema": contract.METADATA_SCHEMA,
        "snapshot_schema_version": contract.SNAPSHOT_SCHEMA,
        "lineage_schema_version": contract.LINEAGE_SCHEMA,
        "seed_namespace": contract.NAMESPACE,
        "snapshot_builder_contract_version": (
            reader.SNAPSHOT_BUILDER_CONTRACT_VERSION
        ),
        "lineage_required_conditions": ["IDEAL_MATCHED"],
        "expected_rng_counts": {
            "FULL_NOISE": 3,
            "IDEAL_MATCHED": 0,
            "INDEPENDENT_NOISE_FREE": 0,
        },
    }
    parameterized_reader = _function(current_reader, "read_v3_snapshot")
    loaded_names = {
        node.id
        for node in ast.walk(parameterized_reader)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert loaded_names.isdisjoint(
        {
            "METADATA_FIELDS",
            "_LOCK_ENTRY_FIELDS",
            "METADATA_SCHEMA",
            "SNAPSHOT_SCHEMA",
            "LINEAGE_SCHEMA",
            "NAMESPACE",
            "SNAPSHOT_BUILDER_CONTRACT_VERSION",
        }
    )
    assert {arg.arg for arg in parameterized_reader.args.kwonlyargs} >= {
        "execution_context",
        "expected_lock_entry",
    }


def test_formal_plan_only_bridge_audits_exact_595_1190_without_payload_reads(
) -> None:
    assert _sha256(SNAPSHOT_PLAN) == SNAPSHOT_PLAN_SHA256
    assert _sha256(TRIAL_PLAN) == TRIAL_PLAN_SHA256
    snapshots = contract.typed_snapshot_rows(SNAPSHOT_PLAN)
    trials = contract.typed_trial_rows(TRIAL_PLAN)
    assert len(snapshots) == len({row["planned_snapshot_id"] for row in snapshots}) == 595
    assert len(trials) == len({row["planned_trial_id"] for row in trials}) == 1190
    assert contract.canonical_identity_sha256(snapshots) == SNAPSHOT_IDENTITY_SHA256
    assert contract.canonical_identity_sha256(trials) == TRIAL_IDENTITY_SHA256

    context = contract.formal_execution_context()
    index = build_canonical_snapshot_index(snapshots, execution_context=context)
    bindings = build_trial_snapshot_bindings(index, trials)
    assert len(index) == 595
    assert len(bindings.rows) == 1190
    assert bindings.audit["backend_trial_counts"] == {
        "open3d_point_to_plane": 595,
        "pcl_point_to_plane": 595,
    }
    assert bindings.audit["native_trial_count"] == 0
    assert bindings.audit["pairing_violation_count"] == 0
    assert bindings.audit["CANONICAL_SNAPSHOT_INDEX_PASS"] is True
    assert bindings.audit["TRIAL_TO_SNAPSHOT_BINDING_PASS"] is True
    assert bindings.audit["SHARED_IDENTITY_VALIDATION_PASS"] is True
    for trial in trials:
        snapshot = bindings.rows[trial["planned_trial_id"]]
        assert snapshot is index.rows[trial["planned_snapshot_id"]]
        for field in (
            "scene_variant",
            "condition",
            "geometry_seed",
            "measurement_seed",
            "repeat_index",
        ):
            assert type(trial[field]) is type(snapshot[field])
            assert trial[field] == snapshot[field]

    legacy_audit = contract.audit_v3_plan(SNAPSHOT_PLAN, TRIAL_PLAN)
    assert legacy_audit["V3_PLAN_PASS"] is True
    assert legacy_audit["planned_snapshot_count"] == 595
    assert legacy_audit["planned_trial_count"] == 1190
    assert legacy_audit["planned_snapshot_unique_count"] == 595
    assert legacy_audit["planned_trial_unique_count"] == 1190
    assert legacy_audit["pairing_violation_count"] == 0
    assert legacy_audit["semantic_violation_count"] == 0
    assert legacy_audit["native_trial_count"] == 0
    assert legacy_audit["planned_snapshot_identity_sha256"] == (
        SNAPSHOT_IDENTITY_SHA256
    )
    assert legacy_audit["planned_trial_identity_sha256"] == TRIAL_IDENTITY_SHA256


def test_plan_only_audit_does_not_import_payload_or_backend_modules() -> None:
    current = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(current)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("backend_execution" in name for name in imported)
    assert "numpy" not in imported
    called = {
        node.func.attr
        for node in ast.walk(current)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } | {
        node.func.id
        for node in ast.walk(current)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "derive_seed" not in called
    assert "default_rng" not in called
    assert "read_v3_snapshot" not in called
