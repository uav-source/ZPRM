from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import phase_a_harness.formal_lifecycle as lifecycle
from phase_a_harness.formal_lifecycle_contract import (
    FormalLifecycleSpec,
    validate_formal_lifecycle_spec,
)
from phase_a_harness.runtime_lifecycle_fixture import (
    CONTEXT_A,
    build_qualification_manifest,
    build_qualification_spec,
)
from phase_a_harness.runtime_lifecycle_io import (
    atomic_replace_canonical_json,
    read_canonical_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "c" * 40


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_spec(tmp_path: Path, *, workers: int) -> FormalLifecycleSpec:
    runtime_root = (tmp_path / "runtime").resolve()
    branch = "qualification/trial-concurrency"
    tag = "archive/qualification-trial-concurrency"
    manifest = build_qualification_manifest(
        CONTEXT_A,
        repository_root=REPOSITORY_ROOT,
        expected_commit=EXPECTED_COMMIT,
        expected_branch=branch,
        expected_tag=tag,
        runtime_root=runtime_root,
        workers=workers,
    )
    manifest_path = (tmp_path / "manifest.json").resolve()
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
        CONTEXT_A,
        repository_root=REPOSITORY_ROOT,
        expected_commit=EXPECTED_COMMIT,
        expected_branch=branch,
        expected_tag=tag,
        runtime_root=runtime_root,
        manifest_path=manifest_path,
        workers=workers,
    )


def test_workers_bound_backend_calls_while_main_thread_commits_in_plan_order(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    spec = _build_spec(tmp_path, workers=2)
    validated = validate_formal_lifecycle_spec(spec)
    _snapshot_report, authenticated_by_id, _lock_report = (
        lifecycle._execute_snapshots(validated)
    )
    snapshot_lock_sha256 = _file_sha256(spec.paths.snapshot_lock)
    plan_ids = [str(row["planned_trial_id"]) for row in spec.trial_plan]
    filename_to_id = {
        lifecycle._safe_result_filename(trial_id): trial_id
        for trial_id in plan_ids
    }

    main_thread_id = threading.get_ident()
    backend_binding_ids = {
        id(binding) for binding in spec.components.backend_registry.values()
    }
    backend_lock = threading.Lock()
    first_wave = threading.Barrier(spec.workers)
    backend_state = {
        "active": 0,
        "calls": 0,
        "max_active": 0,
        "thread_ids": set(),
    }
    original_call = lifecycle._call

    def observed_call(binding: Any, /, **kwargs: Any) -> Any:
        if id(binding) not in backend_binding_ids:
            return original_call(binding, **kwargs)
        with backend_lock:
            backend_state["active"] += 1
            backend_state["calls"] += 1
            backend_state["max_active"] = max(
                backend_state["max_active"], backend_state["active"]
            )
            backend_state["thread_ids"].add(threading.get_ident())
            call_number = backend_state["calls"]
        try:
            if call_number <= spec.workers:
                first_wave.wait(timeout=10)
            return original_call(binding, **kwargs)
        finally:
            with backend_lock:
                backend_state["active"] -= 1

    monkeypatch.setattr(lifecycle, "_call", observed_call)

    durable_operations: list[tuple[str, str, int]] = []
    original_create_bytes = lifecycle.atomic_create_bytes
    original_replace_json = lifecycle.atomic_replace_canonical_json
    original_append_event = lifecycle.append_event_v2
    original_write_progress = lifecycle._write_progress

    def observed_create_bytes(path: Path, payload: bytes) -> Path:
        durable_operations.append(
            ("raw", filename_to_id[Path(path).name], threading.get_ident())
        )
        return original_create_bytes(path, payload)

    def observed_replace_json(path: Path, value: Any) -> Path:
        if Path(path) == spec.paths.raw_manifest:
            trial_id = next(reversed(value["results"]))
            durable_operations.append(
                ("manifest", trial_id, threading.get_ident())
            )
        return original_replace_json(path, value)

    def observed_append_event(path: Path, **kwargs: Any) -> dict[str, Any]:
        durable_operations.append(
            (
                f"event:{kwargs['event_type']}",
                str(kwargs["planned_trial_id"]),
                threading.get_ident(),
            )
        )
        return original_append_event(path, **kwargs)

    def observed_write_progress(
        target_spec: FormalLifecycleSpec,
        *,
        phase: str,
        committed_id: str,
        committed_count: int,
    ) -> None:
        durable_operations.append(
            ("progress", committed_id, threading.get_ident())
        )
        original_write_progress(
            target_spec,
            phase=phase,
            committed_id=committed_id,
            committed_count=committed_count,
        )

    monkeypatch.setattr(lifecycle, "atomic_create_bytes", observed_create_bytes)
    monkeypatch.setattr(
        lifecycle, "atomic_replace_canonical_json", observed_replace_json
    )
    monkeypatch.setattr(lifecycle, "append_event_v2", observed_append_event)
    monkeypatch.setattr(lifecycle, "_write_progress", observed_write_progress)

    report, rows = lifecycle._execute_trials(
        validated,
        authenticated_by_id=authenticated_by_id,
        snapshot_lock_sha256=snapshot_lock_sha256,
        run_contract_sha256="d" * 64,
        invocation_id="workers-two",
    )

    assert backend_state["calls"] == len(plan_ids)
    assert backend_state["max_active"] == spec.workers
    assert backend_state["thread_ids"]
    assert main_thread_id not in backend_state["thread_ids"]
    assert report["executed_trial_ids"] == plan_ids
    assert len(rows) == len(plan_ids)
    assert all(thread_id == main_thread_id for _, _, thread_id in durable_operations)

    commit_operations = [
        (operation, trial_id)
        for operation, trial_id, _thread_id in durable_operations
        if operation in {"raw", "manifest", "event:COMPLETED", "progress"}
    ]
    assert commit_operations == [
        (operation, trial_id)
        for trial_id in plan_ids
        for operation in ("raw", "manifest", "event:COMPLETED", "progress")
    ]
    for trial_id in plan_ids:
        operations = [
            operation
            for operation, operation_trial_id, _thread_id in durable_operations
            if operation_trial_id == trial_id
        ]
        assert operations.index("event:STARTED") < operations.index("raw")

    orphan_id = plan_ids[0]
    raw_manifest = read_canonical_json(spec.paths.raw_manifest)
    del raw_manifest["results"][orphan_id]
    atomic_replace_canonical_json(spec.paths.raw_manifest, raw_manifest)
    backend_calls_before_resume = backend_state["calls"]

    resume_report, resume_rows = lifecycle._execute_trials(
        validated,
        authenticated_by_id=authenticated_by_id,
        snapshot_lock_sha256=snapshot_lock_sha256,
        run_contract_sha256="d" * 64,
        invocation_id="recover-orphan",
    )

    assert backend_state["calls"] == backend_calls_before_resume
    assert resume_report["backend_execution_count_this_invocation"] == 0
    assert resume_report["recovered_orphan_trial_ids"] == [orphan_id]
    assert resume_report["resume_skipped_valid_result_count"] == len(plan_ids)
    assert len(resume_rows) == len(plan_ids)
