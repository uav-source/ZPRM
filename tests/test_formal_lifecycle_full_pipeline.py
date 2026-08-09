from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pytest

from phase_a_harness.formal_lifecycle import (
    CompletedFormalRun,
    PostRunResult,
    execute_postrun_pipeline,
)
from phase_a_harness.formal_lifecycle_contract import (
    FormalLifecycleSpec,
    deep_thaw,
    validate_formal_lifecycle_spec,
)
from phase_a_harness.runtime_lifecycle_io import atomic_create_canonical_json
from phase_a_harness.runtime_lifecycle_fixture import (
    BACKENDS,
    CONDITIONS,
    CONTEXT_A,
    CONTEXT_B,
    EXPECTED_OUTCOMES,
    OPEN3D_BACKEND,
    PCL_BACKEND,
    build_qualification_manifest,
    build_qualification_spec,
    execute_qualification_open3d,
    execute_qualification_pcl,
)
from test_formal_lifecycle_fresh_resume import (
    git_output,
    prepare_qualification_repository,
    run_qualification_worker,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "b" * 40


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_spec(context_name: str, tmp_path: Path) -> FormalLifecycleSpec:
    suffix = "a" if context_name == CONTEXT_A else "b"
    branch = f"qualification/full-pipeline-{suffix}"
    tag = f"archive/qualification-full-pipeline-{suffix}"
    runtime_root = (tmp_path / f"context_{suffix}").resolve()
    manifest = build_qualification_manifest(
        context_name,
        repository_root=REPOSITORY_ROOT,
        expected_commit=EXPECTED_COMMIT,
        expected_branch=branch,
        expected_tag=tag,
        runtime_root=runtime_root,
        workers=1,
    )
    manifest_path = (tmp_path / f"context_{suffix}_manifest.json").resolve()
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return build_qualification_spec(
        context_name,
        repository_root=REPOSITORY_ROOT,
        expected_commit=EXPECTED_COMMIT,
        expected_branch=branch,
        expected_tag=tag,
        runtime_root=runtime_root,
        manifest_path=manifest_path,
        workers=1,
    )


def _call(binding: Any, **kwargs: Any) -> Any:
    return binding.callable(**kwargs)


def _execute_direct_scientific_components(
    spec: FormalLifecycleSpec,
) -> tuple[CompletedFormalRun, dict[str, Any]]:
    """Exercise every scientific component without a Git/runtime bootstrap."""

    validated = validate_formal_lifecycle_spec(spec)
    authenticated_by_id: dict[str, Mapping[str, Any]] = {}
    lock_entries: list[dict[str, Any]] = []
    materialization_receipts: list[dict[str, Any]] = []

    for frozen_snapshot in spec.snapshot_plan:
        snapshot = deep_thaw(frozen_snapshot)
        snapshot_id = str(snapshot["planned_snapshot_id"])
        destination = spec.paths.snapshot_cache / snapshot_id
        receipt = _call(
            spec.components.snapshot_materializer,
            spec=spec,
            snapshot_row=snapshot,
            destination=destination,
        )
        materialization_receipts.append(dict(receipt))
        authenticated = _call(
            spec.components.snapshot_reader,
            spec=spec,
            snapshot_row=snapshot,
            expected_lock_entry=None,
            arrays=True,
        )
        entry = _call(
            spec.components.snapshot_validator,
            spec=spec,
            snapshot_row=snapshot,
            authenticated_snapshot=authenticated,
        )
        authenticated_by_id[snapshot_id] = authenticated
        lock_entries.append(dict(entry))

    lock_value = _call(
        spec.components.snapshot_lock_builder,
        spec=spec,
        authenticated_snapshots=tuple(lock_entries),
    )
    atomic_create_canonical_json(spec.paths.snapshot_lock, lock_value)
    lock_report = _call(
        spec.components.snapshot_lock_validator,
        spec=spec,
        lock_value=lock_value,
    )
    snapshot_lock_sha256 = _file_sha256(spec.paths.snapshot_lock)

    rows: list[dict[str, Any]] = []
    backend_execution_counts: Counter[str] = Counter()
    for frozen_trial in spec.trial_plan:
        trial = deep_thaw(frozen_trial)
        trial_id = str(trial["planned_trial_id"])
        snapshot = deep_thaw(
            validated.trial_snapshot_bindings.rows[trial_id]
        )
        authenticated = authenticated_by_id[
            str(snapshot["planned_snapshot_id"])
        ]
        backend_input = _call(
            spec.components.backend_input_builder,
            spec=spec,
            trial_row=trial,
            snapshot_row=snapshot,
            authenticated_snapshot=authenticated,
        )
        common = _call(
            spec.components.common_record_builder,
            spec=spec,
            trial_row=trial,
            snapshot_row=snapshot,
            backend_input=backend_input,
            snapshot_lock_sha256=snapshot_lock_sha256,
        )
        backend = str(trial["backend"])
        value = _call(
            spec.components.backend_registry[backend],
            backend_input=backend_input,
            common=common,
        )
        row = _call(
            spec.components.result_validator,
            spec=spec,
            trial_row=trial,
            common=common,
            value=value,
        )
        rows.append(dict(row))
        backend_execution_counts[backend] += 1

    run_manifest = {
        "run_id": spec.run_id,
        "planned_snapshot_count": len(spec.snapshot_plan),
        "planned_trial_count": len(spec.trial_plan),
        "completed_snapshot_count": len(authenticated_by_id),
        "completed_trial_count": len(rows),
        "backend_trial_counts": dict(sorted(backend_execution_counts.items())),
        "native_trial_count": 0,
        "complete": True,
    }
    invocation_report = {
        "requested_mode": "fresh",
        "generated_snapshot_count_this_invocation": len(
            materialization_receipts
        ),
        "backend_execution_count_this_invocation": len(rows),
        "confirmatory_seed_access_count": 0,
    }
    completed = CompletedFormalRun(
        spec=spec,
        requested_mode="fresh",
        invocation_id="direct-component-qualification",
        rows=tuple(MappingProxyType(row) for row in rows),
        run_manifest=MappingProxyType(run_manifest),
        invocation_report=MappingProxyType(invocation_report),
        snapshot_lock_sha256=snapshot_lock_sha256,
        raw_manifest_sha256="0" * 64,
        git_gate_reports=(),
    )
    evidence = {
        "authenticated_snapshot_count": len(authenticated_by_id),
        "lock_report": lock_report,
        "backend_execution_counts": dict(sorted(backend_execution_counts.items())),
        "rows": rows,
    }
    return completed, evidence


def _outcome_signature(rows: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                row["condition"],
                row["backend"],
                row["failure_classification"],
                row["solver_failure"],
                row["finite_output"],
            )
            for row in rows
        )
    )


def _assert_expected_fixture_outcomes(rows: list[dict[str, Any]]) -> None:
    assert len(rows) == 6
    assert Counter(row["backend"] for row in rows) == {
        OPEN3D_BACKEND: 3,
        PCL_BACKEND: 3,
    }
    assert Counter(row["condition"] for row in rows) == {
        condition: 2 for condition in CONDITIONS
    }
    for row in rows:
        expected = EXPECTED_OUTCOMES[row["condition"]]
        assert row["failure_classification"] == expected
        assert row["solver_failure"] is (expected != "NONE")


def _assert_postrun(postrun: PostRunResult, spec: FormalLifecycleSpec) -> None:
    assert postrun.difference["exact_match_pass"] is True
    assert postrun.difference["leaf_difference_count"] == 0
    assert postrun.publisher_inventory["publisher_inventory_pass"] is True
    assert postrun.publisher_inventory["table_count"] == 7
    assert postrun.publisher_inventory["figure_count"] == 3
    assert postrun.publisher_inventory["root_file_count"] == 7
    assert postrun.artifact_verification[
        "FIXTURE_ARTIFACT_VERIFICATION_PASS"
    ] is True
    assert len(postrun.artifact_sha256) == 17
    publication = spec.paths.artifact_staging / "formal_publication"
    assert publication.is_dir()
    assert set(postrun.artifact_sha256) == {
        path.relative_to(publication).as_posix()
        for path in publication.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_context_a_and_b_real_three_by_six_full_scientific_pipeline(
    tmp_path: Path,
) -> None:
    """A/B use real readers, bridge, backends, analysis and 7/3/7 publication."""

    outcomes: dict[str, tuple[tuple[Any, ...], ...]] = {}
    for context_name in (CONTEXT_A, CONTEXT_B):
        fixture = prepare_qualification_repository(
            tmp_path,
            label=f"full-pipeline-{context_name}",
            context_name=context_name,
        )
        payload, _process = run_qualification_worker(
            fixture,
            mode="fresh",
            invocation_id=f"fresh-{context_name}",
        )
        lifecycle = payload["lifecycle"]
        invocation = lifecycle["invocation_report"]
        postrun = payload["postrun"]
        rows = postrun["primary"]["results"]
        assert invocation["generated_snapshot_count_this_invocation"] == 3
        assert invocation["backend_execution_count_this_invocation"] == 6
        assert invocation["backend_trial_counts"] == {
            OPEN3D_BACKEND: 3,
            PCL_BACKEND: 3,
        }
        assert invocation["completed_snapshot_count"] == 3
        assert invocation["completed_trial_count"] == 6
        assert invocation["backend_input_checksum_mismatch_count"] == 0
        assert invocation["native_trial_count"] == 0
        assert invocation["all_git_gates_pass"] is True
        _assert_expected_fixture_outcomes(rows)
        outcomes[context_name] = _outcome_signature(rows)
        assert postrun["difference"]["exact_match_pass"] is True
        assert postrun["difference"]["leaf_difference_count"] == 0
        assert postrun["publisher_inventory"]["publisher_inventory_pass"] is True
        assert postrun["publisher_inventory"]["table_count"] == 7
        assert postrun["publisher_inventory"]["figure_count"] == 3
        assert postrun["publisher_inventory"]["root_file_count"] == 7
        assert postrun["artifact_verification"][
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        ] is True
        assert postrun["all_git_gates_pass"] is True
        assert len(postrun["artifact_sha256"]) == 17
        assert git_output(fixture.repository, "status", "--porcelain=v1") == ""

    assert outcomes[CONTEXT_A] == outcomes[CONTEXT_B]
