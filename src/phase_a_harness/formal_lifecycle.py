"""Single version-agnostic formal execution and post-run lifecycle.

Every version-specific dependency is supplied through ``FormalLifecycleSpec``.
This module owns the complete control-flow loop, while injected components are
restricted to one scientific or serialization operation at a time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .formal_lifecycle_contract import (
    FormalLifecycleContractError,
    FormalLifecycleSpec,
    ValidatedFormalLifecycleSpec,
    deep_freeze,
    deep_thaw,
    validate_formal_lifecycle_spec,
)
from .formal_runtime_state_machine import (
    FormalRuntimeState,
    assert_seed_entry_lock,
    bootstrap_formal_runtime,
    bootstrap_lease_path,
    formal_runtime_path_contract,
    inspect_formal_runtime,
    transition_bootstrap_run_lock,
)
from .runtime_lifecycle_io import (
    SingleWriterLease,
    append_event_v2,
    atomic_create_bytes,
    atomic_create_canonical_json,
    atomic_replace_canonical_json,
    canonical_json_bytes,
    canonical_json_sha256,
    read_canonical_json,
    read_event_journal_v2,
)


RAW_MANIFEST_SCHEMA = "version_agnostic_formal_raw_manifest_v1"
RUN_MANIFEST_SCHEMA = "version_agnostic_formal_run_manifest_v1"
INVOCATION_REPORT_SCHEMA = "version_agnostic_formal_invocation_report_v1"
POSTRUN_RESULT_SCHEMA = "version_agnostic_formal_postrun_result_v1"
_REQUESTED_MODES = frozenset({"fresh", "resume"})
_PRE_EXECUTION_EVIDENCE_FILES = frozenset(
    {
        "environment_report.json",
        "preflight_report.json",
        "smoke_qualification_report.json",
        "source_asset_hashes_before.json",
    }
)


class FormalLifecycleError(RuntimeError):
    """The authenticated generic lifecycle could not complete safely."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalLifecycleError(f"{label} must be a mapping")
    return {str(key): deep_thaw(child) for key, child in value.items()}


def _safe_result_filename(trial_id: str) -> str:
    if type(trial_id) is not str or not trial_id:
        raise FormalLifecycleError("planned trial ID is invalid")
    token = hashlib.sha256(trial_id.encode("utf-8")).hexdigest()
    return f"trial_{token}.json"


def _safe_invocation_id(value: str) -> str:
    candidate = Path(value)
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError("invocation_id must be one safe canonical name")
    return value


def _persist_pre_execution_evidence(
    spec: FormalLifecycleSpec,
    evidence: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Commit caller-supplied audit evidence before scientific work starts."""

    if evidence is None:
        return
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        _PRE_EXECUTION_EVIDENCE_FILES
    ):
        raise FormalLifecycleError(
            "pre-execution evidence file inventory is invalid"
        )
    for name in sorted(_PRE_EXECUTION_EVIDENCE_FILES):
        value = _mapping(evidence[name], label=f"pre-execution evidence {name}")
        path = spec.paths.runtime_root / name
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise FormalLifecycleError(
                    "pre-execution evidence path is unsafe"
                )
            if read_canonical_json(path) != value:
                raise FormalLifecycleError(
                    "pre-execution evidence changed across resume"
                )
        else:
            atomic_create_canonical_json(path, value)


def _canonical_command(
    spec: FormalLifecycleSpec, *, requested_mode: str
) -> str:
    if requested_mode not in _REQUESTED_MODES:
        raise ValueError("requested_mode must be fresh or resume")
    profile = spec.command_profile
    argv: list[str] = ["env"]
    for name in profile.unset_environment:
        argv.extend(("-u", name))
    argv.extend(
        f"{name}={value}" for name, value in sorted(profile.environment.items())
    )
    argv.extend(profile.argv_prefix)
    argv.extend(
        (
            profile.entry_script.as_posix(),
            "--manifest",
            spec.manifest_path.relative_to(spec.repository_root).as_posix(),
            "--run-id",
            spec.run_id,
            "--runtime-root",
            str(spec.runtime_paths.runtime_root),
            "--workers",
            str(spec.workers),
            "--mode",
            requested_mode,
        )
    )
    return shlex.join(argv)


def _verify_component_files(validated: ValidatedFormalLifecycleSpec) -> None:
    for name, binding in validated.spec.components.component_bindings().items():
        if _file_sha256(binding.implementation_path) != binding.implementation_sha256:
            raise FormalLifecycleContractError(
                f"component implementation changed after binding: {name}"
            )


def _validate_entry(
    spec: FormalLifecycleSpec, *, requested_mode: str
) -> ValidatedFormalLifecycleSpec:
    if requested_mode not in _REQUESTED_MODES:
        raise ValueError("requested_mode must be fresh or resume")
    validated = validate_formal_lifecycle_spec(spec)
    _verify_component_files(validated)
    expected_lease = bootstrap_lease_path(spec.paths.runtime_root)
    if spec.paths.single_writer_lease != expected_lease:
        raise FormalLifecycleContractError(
            "single writer lease differs from the canonical external lease"
        )
    expected_paths = formal_runtime_path_contract(spec.paths.runtime_root)
    mismatches = sorted(
        name
        for name, expected in expected_paths.items()
        if getattr(spec.paths, name) != expected
    )
    if mismatches:
        raise FormalLifecycleContractError(
            "runtime path layout differs from the authenticated state-machine "
            f"contract: {mismatches}"
        )
    delays = deep_thaw(spec.manifest).get("durable_commit_delay_seconds", {})
    if type(delays) is not dict or set(delays) != {"snapshot", "trial"}:
        raise FormalLifecycleContractError(
            "manifest durable commit delay contract is incomplete"
        )
    for phase, value in delays.items():
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or value < 0
            or value > 30
        ):
            raise FormalLifecycleContractError(
                f"invalid durable commit delay for {phase}"
            )
        if value and spec.mode.value != "QUALIFICATION":
            raise FormalLifecycleContractError(
                "durable commit delays are qualification-only"
            )
    return validated


def _call(binding: Any, /, **kwargs: Any) -> Any:
    return binding.callable(**kwargs)


def _git_gate(
    spec: FormalLifecycleSpec, checkpoint: str
) -> dict[str, Any]:
    value = _call(
        spec.components.git_gate,
        spec=spec,
        checkpoint=f"{spec.identity_policy.checkpoint_prefix}_{checkpoint}",
    )
    report = _mapping(value, label="Git gate report")
    passed = report.get(
        "GIT_GATE_PASS", report.get("RUNTIME_GIT_GATE_PASS")
    )
    if passed is not True:
        raise FormalLifecycleError(f"Git gate failed: {checkpoint}")
    report["GIT_GATE_PASS"] = True
    return report


def _record_git_gate(
    spec: FormalLifecycleSpec,
    checkpoint: str,
    report: Mapping[str, Any],
) -> None:
    directory = spec.paths.temporary_inventory / "git_gates"
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("*.json"))
    path = directory / f"{len(existing):04d}_{checkpoint}.json"
    atomic_create_canonical_json(path, dict(report))


def _write_progress(
    spec: FormalLifecycleSpec,
    *,
    phase: str,
    committed_id: str,
    committed_count: int,
) -> None:
    spec.paths.temporary_inventory.mkdir(parents=True, exist_ok=True)
    path = spec.paths.temporary_inventory / "durable_progress.json"
    value = {
        "schema_version": "version_agnostic_durable_progress_v1",
        "phase": phase,
        "committed_id": committed_id,
        "committed_count": committed_count,
        "run_id": spec.run_id,
    }
    if path.exists():
        atomic_replace_canonical_json(path, value)
    else:
        atomic_create_canonical_json(path, value)
    delay = float(
        deep_thaw(spec.manifest)["durable_commit_delay_seconds"][phase]
    )
    if delay:
        time.sleep(delay)


def _read_existing_lock(spec: FormalLifecycleSpec) -> dict[str, Any] | None:
    if not spec.paths.snapshot_lock.exists():
        return None
    value = read_canonical_json(spec.paths.snapshot_lock)
    report = _call(
        spec.components.snapshot_lock_validator,
        spec=spec,
        lock_value=value,
    )
    return _mapping(report, label="snapshot lock validation")


def _snapshot_destination(spec: FormalLifecycleSpec, row: Mapping[str, Any]) -> Path:
    snapshot_id = str(row["planned_snapshot_id"])
    candidate = Path(snapshot_id)
    if (
        not snapshot_id
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or snapshot_id in {".", ".."}
        or "/" in snapshot_id
        or "\\" in snapshot_id
    ):
        raise FormalLifecycleError(
            "planned snapshot ID is not one safe cache-directory name"
        )
    return spec.paths.snapshot_cache / snapshot_id


def _execute_snapshots(
    validated: ValidatedFormalLifecycleSpec,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = validated.spec
    existing_lock_report = _read_existing_lock(spec)
    existing_entries = (
        existing_lock_report.get("entries_by_id", {})
        if existing_lock_report is not None
        else {}
    )
    if not isinstance(existing_entries, Mapping):
        raise FormalLifecycleError("snapshot lock entry index is invalid")
    spec.paths.snapshot_cache.mkdir(parents=True, exist_ok=True)
    expected_ids = {
        str(row["planned_snapshot_id"]) for row in spec.snapshot_plan
    }
    for entry in os.scandir(spec.paths.snapshot_cache):
        metadata = entry.stat(follow_symlinks=False)
        if (
            entry.name not in expected_ids
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise FormalLifecycleError(
                "snapshot cache contains an extra or unsafe entry before execution"
            )
    generated: list[str] = []
    skipped: list[str] = []
    confirmatory_seed_access_count = 0
    confirmatory_rng_instantiation_count = 0
    authenticated_by_id: dict[str, Any] = {}
    lock_entries: list[dict[str, Any]] = []
    for frozen_row in spec.snapshot_plan:
        row = _mapping(frozen_row, label="snapshot row")
        snapshot_id = str(row["planned_snapshot_id"])
        destination = _snapshot_destination(spec, row)
        if destination.exists() or destination.is_symlink():
            skipped.append(snapshot_id)
        else:
            receipt = _mapping(
                _call(
                    spec.components.snapshot_materializer,
                    spec=spec,
                    snapshot_row=row,
                    destination=destination,
                ),
                label="snapshot materializer receipt",
            )
            for field in (
                "confirmatory_seed_access_count",
                "confirmatory_rng_instantiation_count",
            ):
                value = receipt.get(field, 0)
                if type(value) is not int or type(value) is bool or value < 0:
                    raise FormalLifecycleError(
                        f"snapshot materializer returned invalid {field}"
                    )
            confirmatory_seed_access_count += int(
                receipt.get("confirmatory_seed_access_count", 0)
            )
            confirmatory_rng_instantiation_count += int(
                receipt.get("confirmatory_rng_instantiation_count", 0)
            )
            generated.append(snapshot_id)
        authenticated = _call(
            spec.components.snapshot_reader,
            spec=spec,
            snapshot_row=row,
            expected_lock_entry=existing_entries.get(snapshot_id),
            arrays=True,
        )
        entry = _mapping(
            _call(
                spec.components.snapshot_validator,
                spec=spec,
                snapshot_row=row,
                authenticated_snapshot=authenticated,
            ),
            label="snapshot validator entry",
        )
        if entry.get("snapshot_id") != snapshot_id:
            raise FormalLifecycleError("snapshot validator changed the snapshot ID")
        authenticated_by_id[snapshot_id] = authenticated
        lock_entries.append(entry)
        if snapshot_id in generated:
            _write_progress(
                spec,
                phase="snapshot",
                committed_id=snapshot_id,
                committed_count=len(generated),
            )
    lock_value = _mapping(
        _call(
            spec.components.snapshot_lock_builder,
            spec=spec,
            authenticated_snapshots=tuple(lock_entries),
        ),
        label="snapshot lock",
    )
    if spec.paths.snapshot_lock.exists():
        if read_canonical_json(spec.paths.snapshot_lock) != lock_value:
            raise FormalLifecycleError("existing snapshot lock differs from rebuilt lock")
    else:
        atomic_create_canonical_json(spec.paths.snapshot_lock, lock_value)
    lock_report = _mapping(
        _call(
            spec.components.snapshot_lock_validator,
            spec=spec,
            lock_value=read_canonical_json(spec.paths.snapshot_lock),
        ),
        label="snapshot lock validation",
    )
    if lock_report.get("snapshot_lock_pass") is not True:
        raise FormalLifecycleError("snapshot lock validation failed")
    if set(authenticated_by_id) != expected_ids:
        raise FormalLifecycleError("authenticated snapshot inventory is incomplete")
    final_entries = {
        entry.name
        for entry in os.scandir(spec.paths.snapshot_cache)
        if entry.is_dir(follow_symlinks=False)
    }
    if final_entries != expected_ids or any(
        entry.is_symlink() or not entry.is_dir(follow_symlinks=False)
        for entry in os.scandir(spec.paths.snapshot_cache)
    ):
        raise FormalLifecycleError("snapshot cache reverse inventory mismatch")
    return (
        {
            "generated_snapshot_ids": generated,
            "generated_snapshot_count_this_invocation": len(generated),
            "resume_skipped_valid_snapshot_ids": skipped,
            "resumed_snapshot_count_this_invocation": len(skipped),
            "confirmatory_seed_access_count_this_invocation": (
                confirmatory_seed_access_count
            ),
            "confirmatory_rng_instantiation_count_this_invocation": (
                confirmatory_rng_instantiation_count
            ),
        },
        authenticated_by_id,
        lock_report,
    )


def _empty_raw_manifest(
    *, spec: FormalLifecycleSpec, run_contract_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": RAW_MANIFEST_SCHEMA,
        "run_id": spec.run_id,
        "run_contract_sha256": run_contract_sha256,
        "spec_payload_sha256": validate_formal_lifecycle_spec(
            spec
        ).spec_payload_sha256,
        "results": {},
    }


def _load_raw_manifest(
    *,
    spec: FormalLifecycleSpec,
    run_contract_sha256: str,
    create: bool,
) -> dict[str, Any]:
    expected = _empty_raw_manifest(
        spec=spec, run_contract_sha256=run_contract_sha256
    )
    path = spec.paths.raw_manifest
    if not path.exists():
        if not create:
            raise FormalLifecycleError("raw manifest is absent")
        atomic_create_canonical_json(path, expected)
        return expected
    value = read_canonical_json(path)
    if (
        type(value) is not dict
        or set(value) != set(expected)
        or any(value.get(name) != expected[name] for name in expected if name != "results")
        or type(value.get("results")) is not dict
    ):
        raise FormalLifecycleError("raw manifest contract mismatch")
    return value


def _validate_manifest_entry(
    *,
    spec: FormalLifecycleSpec,
    trial_row: Mapping[str, Any],
    common: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    trial_id = str(trial_row["planned_trial_id"])
    expected_name = _safe_result_filename(trial_id)
    if (
        type(entry) is not dict
        or set(entry) != {"path", "planned_trial_id", "sha256"}
        or entry.get("path") != expected_name
        or entry.get("planned_trial_id") != trial_id
    ):
        raise FormalLifecycleError("raw result manifest entry mismatch")
    path = spec.paths.raw_results / expected_name
    if path.is_symlink() or not path.is_file():
        raise FormalLifecycleError("raw result file is absent or unsafe")
    if _file_sha256(path) != entry.get("sha256"):
        raise FormalLifecycleError("raw result file SHA mismatch")
    return _mapping(
        _call(
            spec.components.resume_result_validator,
            spec=spec,
            trial_row=trial_row,
            common=common,
            path=path,
            entry=entry,
        ),
        label="resumed trial result",
    )


@dataclass(frozen=True)
class _PendingBackendTrial:
    trial_id: str
    trial_row: dict[str, Any]
    backend: str
    binding: Any
    backend_input: Any
    common: dict[str, Any]
    result_path: Path


def _call_pending_backend(trial: _PendingBackendTrial) -> Any:
    """Run only the authorized backend operation in an executor thread."""

    return _call(
        trial.binding,
        backend_input=trial.backend_input,
        common=trial.common,
    )


def _execute_trials(
    validated: ValidatedFormalLifecycleSpec,
    *,
    authenticated_by_id: Mapping[str, Any],
    snapshot_lock_sha256: str,
    run_contract_sha256: str,
    invocation_id: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    spec = validated.spec
    spec.paths.raw_results.mkdir(parents=True, exist_ok=True)
    spec.paths.backend_temporary.mkdir(parents=True, exist_ok=True)
    spec.paths.event_log.parent.mkdir(parents=True, exist_ok=True)
    raw_manifest = _load_raw_manifest(
        spec=spec, run_contract_sha256=run_contract_sha256, create=True
    )
    plan_by_id = {
        str(row["planned_trial_id"]): _mapping(row, label="trial row")
        for row in spec.trial_plan
    }
    if len(plan_by_id) != len(spec.trial_plan):
        raise FormalLifecycleError("trial plan contains duplicate IDs")
    expected_files = {
        _safe_result_filename(trial_id) for trial_id in plan_by_id
    }
    for entry in os.scandir(spec.paths.raw_results):
        metadata = entry.stat(follow_symlinks=False)
        if (
            entry.name not in expected_files
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise FormalLifecycleError(
                "raw result inventory contains an extra or unsafe entry "
                "before backend execution"
            )
    extra_manifest = sorted(set(raw_manifest["results"]) - set(plan_by_id))
    if extra_manifest:
        raise FormalLifecycleError("raw manifest contains extra trial IDs")
    completed: dict[str, dict[str, Any]] = {}
    executed: list[str] = []
    skipped: list[str] = []
    recovered: list[str] = []
    pending_trials: list[_PendingBackendTrial] = []
    for trial_id, trial_row in plan_by_id.items():
        validated_trial = _mapping(
            _call(spec.components.trial_validator, row=trial_row),
            label="component-validated trial row",
        )
        if validated_trial != trial_row:
            raise FormalLifecycleError(
                "trial validator changed the frozen trial identity"
            )
        trial_row = validated_trial
        snapshot_row = _mapping(
            validated.trial_snapshot_bindings.rows[trial_id],
            label="bound snapshot row",
        )
        snapshot_id = str(snapshot_row["planned_snapshot_id"])
        authenticated = authenticated_by_id[snapshot_id]
        backend_input = _call(
            spec.components.backend_input_builder,
            spec=spec,
            trial_row=trial_row,
            snapshot_row=snapshot_row,
            authenticated_snapshot=authenticated,
        )
        common = _mapping(
            _call(
                spec.components.common_record_builder,
                spec=spec,
                trial_row=trial_row,
                snapshot_row=snapshot_row,
                backend_input=backend_input,
                snapshot_lock_sha256=snapshot_lock_sha256,
            ),
            label="trial common record",
        )
        entry = raw_manifest["results"].get(trial_id)
        result_path = spec.paths.raw_results / _safe_result_filename(trial_id)
        if entry is None and result_path.exists():
            orphan_entry = {
                "path": result_path.name,
                "planned_trial_id": trial_id,
                "sha256": _file_sha256(result_path),
            }
            payload = _validate_manifest_entry(
                spec=spec,
                trial_row=trial_row,
                common=common,
                entry=orphan_entry,
            )
            raw_manifest["results"][trial_id] = orphan_entry
            atomic_replace_canonical_json(spec.paths.raw_manifest, raw_manifest)
            recovered.append(trial_id)
            entry = orphan_entry
        if entry is not None:
            payload = _validate_manifest_entry(
                spec=spec,
                trial_row=trial_row,
                common=common,
                entry=entry,
            )
            completed[trial_id] = payload
            skipped.append(trial_id)
            continue
        backend = str(trial_row["backend"])
        binding = spec.components.backend_registry.get(backend)
        if binding is None:
            raise PermissionError(f"backend is not authorized: {backend}")
        pending_trials.append(
            _PendingBackendTrial(
                trial_id=trial_id,
                trial_row=trial_row,
                backend=backend,
                binding=binding,
                backend_input=backend_input,
                common=common,
                result_path=result_path,
            )
        )

    # Authenticate the complete resume/orphan inventory before starting any new
    # scientific backend call.  Only the backend operation runs in executor
    # threads; result validation and every durable write remain ordered on this
    # single-writer lifecycle thread.
    remaining = iter(pending_trials)
    in_flight: deque[tuple[_PendingBackendTrial, Future[Any]]] = deque()
    with ThreadPoolExecutor(
        max_workers=spec.workers,
        thread_name_prefix="formal-backend",
    ) as executor:

        def submit_next() -> bool:
            try:
                trial = next(remaining)
            except StopIteration:
                return False
            append_event_v2(
                spec.paths.event_log,
                run_id=spec.run_id,
                invocation_id=invocation_id,
                event_type="STARTED",
                planned_trial_id=trial.trial_id,
                backend=trial.backend,
            )
            in_flight.append(
                (trial, executor.submit(_call_pending_backend, trial))
            )
            return True

        for _index in range(min(spec.workers, len(pending_trials))):
            submit_next()

        while in_flight:
            trial, future = in_flight.popleft()
            value = future.result()
            payload = _mapping(
                _call(
                    spec.components.result_validator,
                    spec=spec,
                    trial_row=trial.trial_row,
                    common=trial.common,
                    value=value,
                ),
                label="trial result",
            )
            atomic_create_bytes(
                trial.result_path,
                canonical_json_bytes(payload),
            )
            entry = {
                "path": trial.result_path.name,
                "planned_trial_id": trial.trial_id,
                "sha256": _file_sha256(trial.result_path),
            }
            raw_manifest["results"][trial.trial_id] = entry
            atomic_replace_canonical_json(spec.paths.raw_manifest, raw_manifest)
            append_event_v2(
                spec.paths.event_log,
                run_id=spec.run_id,
                invocation_id=invocation_id,
                event_type="COMPLETED",
                planned_trial_id=trial.trial_id,
                backend=trial.backend,
                result_sha256=str(entry["sha256"]),
            )
            completed[trial.trial_id] = payload
            executed.append(trial.trial_id)
            _write_progress(
                spec,
                phase="trial",
                committed_id=trial.trial_id,
                committed_count=len(executed),
            )
            submit_next()
    expected_ids = set(plan_by_id)
    if set(raw_manifest["results"]) != expected_ids or set(completed) != expected_ids:
        raise FormalLifecycleError("formal trial matrix is incomplete")
    actual_files: set[str] = set()
    unsafe_entry_count = 0
    for entry in os.scandir(spec.paths.raw_results):
        metadata = entry.stat(follow_symlinks=False)
        actual_files.add(entry.name)
        unsafe_entry_count += int(
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
        )
    if actual_files != expected_files or unsafe_entry_count:
        raise FormalLifecycleError("raw result reverse inventory mismatch")
    rows = tuple(completed[trial_id] for trial_id in sorted(completed))
    events = read_event_journal_v2(
        spec.paths.event_log, expected_run_id=spec.run_id
    )
    return (
        {
            "backend_execution_count_this_invocation": len(executed),
            "executed_trial_ids": executed,
            "resume_skipped_valid_result_count": len(skipped),
            "resumed_trial_ids": skipped,
            "recovered_orphan_result_count": len(recovered),
            "recovered_orphan_trial_ids": recovered,
            "event_count": len(events),
            "raw_result_manifest_sha256": _file_sha256(spec.paths.raw_manifest),
        },
        rows,
    )


def _pairing_audit(
    spec: FormalLifecycleSpec, rows: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["snapshot_id"])].append(row)
    expected_backends = set(spec.execution_context.backend_policy.allowed_backends)
    checksum_fields = (
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "snapshot_checksum",
    )
    mismatch = 0
    for group in groups.values():
        mismatch += int(
            len(group) != len(expected_backends)
            or {str(row["backend"]) for row in group} != expected_backends
            or any(
                group[0].get(field) != row.get(field)
                for field in checksum_fields
                for row in group[1:]
            )
        )
    return {
        "completed_snapshot_count": len(groups),
        "completed_trial_count": len(rows),
        "backend_input_checksum_mismatch_count": mismatch,
        "backend_trial_counts": dict(
            sorted(Counter(str(row["backend"]) for row in rows).items())
        ),
        "native_trial_count": sum(
            1
            for row in rows
            if str(row["backend"]) not in expected_backends
        ),
    }


@dataclass(frozen=True)
class CompletedFormalRun:
    spec: FormalLifecycleSpec
    requested_mode: str
    invocation_id: str
    rows: tuple[Mapping[str, Any], ...]
    run_manifest: Mapping[str, Any]
    invocation_report: Mapping[str, Any]
    snapshot_lock_sha256: str
    raw_manifest_sha256: str
    git_gate_reports: tuple[Mapping[str, Any], ...]

    def report(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "invocation_id": self.invocation_id,
            "row_count": len(self.rows),
            "run_manifest": deep_thaw(self.run_manifest),
            "invocation_report": deep_thaw(self.invocation_report),
            "snapshot_lock_sha256": self.snapshot_lock_sha256,
            "raw_manifest_sha256": self.raw_manifest_sha256,
            "git_gate_count": len(self.git_gate_reports),
        }


@dataclass(frozen=True)
class PostRunResult:
    primary: Mapping[str, Any]
    independent: Mapping[str, Any]
    difference: Mapping[str, Any]
    publication: Mapping[str, Any]
    artifact_verification: Mapping[str, Any]
    publisher_inventory: Mapping[str, Any]
    artifact_sha256: Mapping[str, str]
    git_gate_reports: tuple[Mapping[str, Any], ...]

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": POSTRUN_RESULT_SCHEMA,
            "primary": deep_thaw(self.primary),
            "independent": deep_thaw(self.independent),
            "difference": deep_thaw(self.difference),
            "publication": deep_thaw(self.publication),
            "artifact_verification": deep_thaw(self.artifact_verification),
            "publisher_inventory": deep_thaw(self.publisher_inventory),
            "artifact_sha256": dict(self.artifact_sha256),
            "git_gate_reports": deep_thaw(self.git_gate_reports),
            "git_gate_count": len(self.git_gate_reports),
            "all_git_gates_pass": all(
                report.get("GIT_GATE_PASS") is True
                for report in self.git_gate_reports
            ),
        }


def execute_formal_lifecycle(
    spec: FormalLifecycleSpec,
    *,
    requested_mode: str,
    invocation_id: str | None = None,
    pre_execution_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> CompletedFormalRun:
    """Execute one fresh or resume invocation through the sole lifecycle."""

    validated = _validate_entry(spec, requested_mode=requested_mode)
    invocation = _safe_invocation_id(invocation_id or requested_mode)
    pre_gate = _git_gate(spec, "PRE_RUN_GIT_GATE")
    base_contract = _mapping(
        _call(spec.components.run_contract_builder, spec=spec),
        label="base run contract",
    )
    fresh_command = _canonical_command(spec, requested_mode="fresh")
    initial = inspect_formal_runtime(
        spec.paths.runtime_root,
        expected_command=(fresh_command if spec.paths.runtime_root.exists() else None),
        expected_base_contract=(base_contract if requested_mode == "resume" else None),
    )
    expected_initial = (
        FormalRuntimeState.RESUMABLE
        if requested_mode == "resume"
        else {FormalRuntimeState.ABSENT, FormalRuntimeState.BOOTSTRAP_ONLY}
    )
    if (
        requested_mode == "resume"
        and initial["state"] is not expected_initial
    ) or (
        requested_mode == "fresh"
        and initial["state"] not in expected_initial
    ):
        raise FormalLifecycleError(
            f"{requested_mode} observed invalid initial runtime state: "
            f"{initial['state'].value}"
        )
    bootstrap = bootstrap_formal_runtime(
        spec.paths.runtime_root,
        fresh_command,
        mode=requested_mode,
    )
    gates: list[dict[str, Any]] = [pre_gate]
    with SingleWriterLease(spec.paths.single_writer_lease):
        transition = transition_bootstrap_run_lock(
            spec.paths.runtime_root, base_contract, mode=requested_mode
        )
        seed_gate = assert_seed_entry_lock(
            spec.paths.runtime_root, base_contract
        )
        if seed_gate.get("seed_entry_lock_gate_pass") is not True:
            raise FormalLifecycleError("immutable lifecycle entry lock failed")
        _persist_pre_execution_evidence(spec, pre_execution_evidence)
        spec.paths.temporary_inventory.mkdir(parents=True, exist_ok=True)
        _record_git_gate(spec, "PRE_RUN_GIT_GATE", pre_gate)
        post_lock = _git_gate(spec, "POST_LOCK_GIT_GATE")
        gates.append(post_lock)
        _record_git_gate(spec, "POST_LOCK_GIT_GATE", post_lock)
        snapshots, authenticated, lock_report = _execute_snapshots(validated)
        post_snapshot = _git_gate(spec, "POST_SNAPSHOT_GIT_GATE")
        gates.append(post_snapshot)
        _record_git_gate(spec, "POST_SNAPSHOT_GIT_GATE", post_snapshot)
        snapshot_lock_sha = _file_sha256(spec.paths.snapshot_lock)
        run_contract_sha = canonical_json_sha256(
            transition["enhanced_contract"]
        )
        trials, rows = _execute_trials(
            validated,
            authenticated_by_id=authenticated,
            snapshot_lock_sha256=snapshot_lock_sha,
            run_contract_sha256=run_contract_sha,
            invocation_id=invocation,
        )
        post_trial = _git_gate(spec, "POST_TRIAL_GIT_GATE")
        gates.append(post_trial)
        _record_git_gate(spec, "POST_TRIAL_GIT_GATE", post_trial)
        pairing = _pairing_audit(spec, rows)
        if (
            pairing["backend_input_checksum_mismatch_count"]
            or pairing["native_trial_count"]
            or pairing["completed_snapshot_count"] != len(spec.snapshot_plan)
            or pairing["completed_trial_count"] != len(spec.trial_plan)
        ):
            raise FormalLifecycleError("completed result pairing failed")
        seed_access_count = base_contract.get(
            "confirmatory_seed_access_count"
        )
        contract_rng_instantiation_count = base_contract.get(
            "confirmatory_rng_instantiation_count",
            0,
        )
        lock_value = lock_report.get("lock")
        rng_instantiation_count = (
            lock_value.get("confirmatory_rng_instantiation_count")
            if type(lock_value) is dict
            else None
        )
        if any(
            type(value) is not int or type(value) is bool or value < 0
            for value in (
                seed_access_count,
                contract_rng_instantiation_count,
                rng_instantiation_count,
            )
        ) or contract_rng_instantiation_count != rng_instantiation_count:
            raise FormalLifecycleError(
                "immutable run contract or snapshot lock lacks stable counters"
            )
        run_manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "run_id": spec.run_id,
            "spec_payload_sha256": validated.spec_payload_sha256,
            "manifest_sha256": spec.manifest_sha256,
            "execution_mode": spec.mode.value,
            "planned_snapshot_count": len(spec.snapshot_plan),
            "planned_trial_count": len(spec.trial_plan),
            **pairing,
            "snapshot_lock_sha256": snapshot_lock_sha,
            "raw_result_manifest_sha256": trials[
                "raw_result_manifest_sha256"
            ],
            "immutable_run_lock_payload_sha256": transition["lock"][
                "payload_sha256"
            ],
            "formal_command_log_sha256": transition["command_binding"][
                "command_log_sha256"
            ],
            "formal_confirmatory_seed_access_count": seed_access_count,
            "confirmatory_rng_instantiation_count_this_invocation": (
                rng_instantiation_count
            ),
            "native_execution_count": 0,
            "complete": True,
        }
        if spec.paths.run_manifest.exists():
            if read_canonical_json(spec.paths.run_manifest) != run_manifest:
                raise FormalLifecycleError(
                    "resume run manifest differs from the completed fresh run"
                )
        else:
            atomic_create_canonical_json(spec.paths.run_manifest, run_manifest)
        final_gate = _git_gate(spec, "FINAL_GIT_GATE")
        gates.append(final_gate)
        _record_git_gate(spec, "FINAL_GIT_GATE", final_gate)
        invocation_report = {
            "schema_version": INVOCATION_REPORT_SCHEMA,
            "run_id": spec.run_id,
            "invocation_id": invocation,
            "requested_mode": requested_mode,
            "bootstrap_created": bootstrap["created"],
            "formal_runtime_state_before": initial["state"].value,
            "formal_runtime_state_after": transition["state_after"].value,
            **snapshots,
            **trials,
            **pairing,
            "snapshot_lock_pass": lock_report.get("snapshot_lock_pass"),
            "git_gate_count": len(gates),
            "all_git_gates_pass": all(
                report.get("GIT_GATE_PASS") is True for report in gates
            ),
        }
        report_path = (
            spec.paths.temporary_inventory
            / "invocations"
            / f"{invocation}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            if read_canonical_json(report_path) != invocation_report:
                raise FormalLifecycleError("invocation report identity collision")
        else:
            atomic_create_canonical_json(report_path, invocation_report)
    return CompletedFormalRun(
        spec=spec,
        requested_mode=requested_mode,
        invocation_id=invocation,
        rows=deep_freeze(rows, location="completed_run.rows"),
        run_manifest=deep_freeze(
            run_manifest, location="completed_run.run_manifest"
        ),
        invocation_report=deep_freeze(
            invocation_report, location="completed_run.invocation_report"
        ),
        snapshot_lock_sha256=snapshot_lock_sha,
        raw_manifest_sha256=str(trials["raw_result_manifest_sha256"]),
        git_gate_reports=deep_freeze(
            gates, location="completed_run.git_gate_reports"
        ),
    )


def _artifact_inventory(
    spec: FormalLifecycleSpec, root: Path
) -> dict[str, Any]:
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    if not tables_dir.is_dir() or not figures_dir.is_dir():
        raise FormalLifecycleError("publisher inventory directories are absent")
    root_entries = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    if (
        any(path.is_symlink() for path in root_entries)
        or {path.name for path in root_entries if path.is_dir()}
        != {"tables", "figures"}
        or any(
            not path.is_file() and not path.is_dir()
            for path in root_entries
        )
    ):
        raise FormalLifecycleError(
            "publisher root contains an extra or unsafe entry"
        )
    table_entries = tuple(sorted(tables_dir.iterdir(), key=lambda path: path.name))
    figure_entries = tuple(
        sorted(figures_dir.iterdir(), key=lambda path: path.name)
    )
    if any(
        path.is_symlink() or not path.is_file()
        for path in (*table_entries, *figure_entries)
    ):
        raise FormalLifecycleError(
            "publisher table/figure inventory contains a non-regular entry"
        )
    tables = tuple(path.name for path in table_entries)
    figures = tuple(path.name for path in figure_entries)
    roots = tuple(path.name for path in root_entries if path.is_file())
    expected = spec.publisher_inventory
    result = {
        "tables": list(tables),
        "figures": list(figures),
        "root_files": list(roots),
        "table_count": len(tables),
        "figure_count": len(figures),
        "root_file_count": len(roots),
        "publisher_inventory_pass": bool(
            set(tables) == set(expected.tables)
            and set(figures) == set(expected.figures)
            and set(roots) == set(expected.root_files)
        ),
    }
    if result["publisher_inventory_pass"] is not True:
        raise FormalLifecycleError("publisher inventory differs from frozen 7/3/7")
    return result


def _tree_sha256(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _authenticate_completed_rows(
    spec: FormalLifecycleSpec,
    completed_run: CompletedFormalRun,
) -> None:
    if _file_sha256(spec.paths.snapshot_lock) != completed_run.snapshot_lock_sha256:
        raise FormalLifecycleError("snapshot lock changed before post-run")
    if _file_sha256(spec.paths.raw_manifest) != completed_run.raw_manifest_sha256:
        raise FormalLifecycleError("raw result manifest changed before post-run")
    if read_canonical_json(spec.paths.run_manifest) != deep_thaw(
        completed_run.run_manifest
    ):
        raise FormalLifecycleError("run manifest changed before post-run")
    manifest = read_canonical_json(spec.paths.raw_manifest)
    entries = manifest.get("results") if type(manifest) is dict else None
    if type(entries) is not dict or len(entries) != len(spec.trial_plan):
        raise FormalLifecycleError("post-run raw result inventory is incomplete")
    rows: list[dict[str, Any]] = []
    for trial_id in sorted(entries):
        entry = entries[trial_id]
        if (
            type(entry) is not dict
            or entry.get("planned_trial_id") != trial_id
            or type(entry.get("path")) is not str
            or type(entry.get("sha256")) is not str
        ):
            raise FormalLifecycleError(
                "post-run raw result manifest entry is invalid"
            )
        path = spec.paths.raw_results / entry["path"]
        if (
            path.parent != spec.paths.raw_results
            or path.is_symlink()
            or not path.is_file()
            or _file_sha256(path) != entry["sha256"]
        ):
            raise FormalLifecycleError(
                "post-run raw result file authentication failed"
            )
        value = read_canonical_json(path)
        if canonical_json_bytes(value) != path.read_bytes():
            raise FormalLifecycleError(
                "post-run raw result is not canonical JSON"
            )
        rows.append(_mapping(value, label="post-run raw result"))
    if rows != [deep_thaw(row) for row in completed_run.rows]:
        raise FormalLifecycleError(
            "completed-run rows differ from authenticated raw evidence"
        )


def execute_postrun_pipeline(
    spec: FormalLifecycleSpec,
    completed_run: CompletedFormalRun,
) -> PostRunResult:
    """Execute the sole injected analysis/verifier/publisher pipeline."""

    validated = _validate_entry(spec, requested_mode=completed_run.requested_mode)
    del validated
    if completed_run.spec is not spec:
        raise FormalLifecycleContractError(
            "completed run belongs to another immutable spec"
        )
    if len(completed_run.rows) != len(spec.trial_plan):
        raise FormalLifecycleError("post-run input trial matrix is incomplete")
    _authenticate_completed_rows(spec, completed_run)
    pre_gate = _git_gate(spec, "POSTRUN_PRE_GIT_GATE")
    _record_git_gate(spec, "POSTRUN_PRE_GIT_GATE", pre_gate)
    primary = _mapping(
        _call(
            spec.components.primary_analyzer,
            spec=spec,
            completed_run=completed_run,
        ),
        label="primary analysis",
    )
    independent = _mapping(
        _call(
            spec.components.independent_verifier,
            spec=spec,
            completed_run=completed_run,
        ),
        label="independent verification",
    )
    difference = _mapping(
        _call(
            spec.components.difference_auditor,
            primary=primary,
            independent=independent,
        ),
        label="primary-independent difference",
    )
    if difference.get("exact_match_pass") is not True:
        raise FormalLifecycleError("primary and independent analyses differ")
    publisher_destination = (
        spec.paths.publisher_staging / "formal_publication"
    )
    destination = spec.paths.artifact_staging / "formal_publication"
    if destination.exists():
        publication = {
            "publication_path": str(destination),
            "resume_reused_verified_publication": True,
            "publisher_execution_count_this_invocation": 0,
        }
    else:
        if not publisher_destination.exists():
            publication = _mapping(
                _call(
                    spec.components.publisher,
                    spec=spec,
                    completed_run=completed_run,
                    primary=primary,
                    independent=independent,
                    destination=publisher_destination,
                ),
                label="publication report",
            )
        else:
            publication = {
                "publication_path": str(publisher_destination),
                "resume_reused_verified_publisher_staging": True,
                "publisher_execution_count_this_invocation": 0,
            }
        staging_verification = _mapping(
            _call(
                spec.components.artifact_verifier,
                spec=spec,
                artifact_path=publisher_destination,
                write_report=False,
            ),
            label="publisher staging verification",
        )
        staging_passes = [
            value
            for key, value in staging_verification.items()
            if key.endswith("_VERIFICATION_PASS")
        ]
        if not staging_passes or any(value is not True for value in staging_passes):
            raise FormalLifecycleError(
                "publisher staging artifact verification failed"
            )
        spec.paths.artifact_staging.mkdir(parents=True, exist_ok=True)
        os.replace(publisher_destination, destination)
        if "artifact_path" in publication:
            publication["artifact_path"] = str(destination)
        publication["publication_path"] = str(destination)
        publication["publisher_execution_count_this_invocation"] = int(
            publication.get(
                "publisher_execution_count_this_invocation", 1
            )
        )
    verification = _mapping(
        _call(
            spec.components.artifact_verifier,
            spec=spec,
            artifact_path=destination,
            write_report=False,
        ),
        label="artifact verification",
    )
    verification_passes = [
        value
        for key, value in verification.items()
        if key.endswith("_VERIFICATION_PASS")
    ]
    if not verification_passes or any(
        value is not True for value in verification_passes
    ):
        raise FormalLifecycleError("artifact verifier did not report PASS")
    inventory = _artifact_inventory(spec, destination)
    sha = _tree_sha256(destination)
    spec.paths.analysis.mkdir(parents=True, exist_ok=True)
    spec.paths.verification.mkdir(parents=True, exist_ok=True)
    primary_path = spec.paths.analysis / "primary.json"
    independent_path = spec.paths.verification / "independent.json"
    difference_path = spec.paths.verification / "difference.json"
    for path, value in (
        (primary_path, primary),
        (independent_path, independent),
        (difference_path, difference),
    ):
        if path.exists():
            if read_canonical_json(path) != value:
                raise FormalLifecycleError("post-run resume content changed")
        else:
            atomic_create_canonical_json(path, value)
    final_gate = _git_gate(spec, "POSTRUN_FINAL_GIT_GATE")
    _record_git_gate(spec, "POSTRUN_FINAL_GIT_GATE", final_gate)
    return PostRunResult(
        primary=deep_freeze(primary, location="postrun.primary"),
        independent=deep_freeze(
            independent, location="postrun.independent"
        ),
        difference=deep_freeze(difference, location="postrun.difference"),
        publication=deep_freeze(
            publication, location="postrun.publication"
        ),
        artifact_verification=deep_freeze(
            verification, location="postrun.artifact_verification"
        ),
        publisher_inventory=deep_freeze(
            inventory, location="postrun.publisher_inventory"
        ),
        artifact_sha256=deep_freeze(
            sha, location="postrun.artifact_sha256"
        ),
        git_gate_reports=deep_freeze(
            (pre_gate, final_gate), location="postrun.git_gate_reports"
        ),
    )


__all__ = [
    "CompletedFormalRun",
    "FormalLifecycleError",
    "INVOCATION_REPORT_SCHEMA",
    "POSTRUN_RESULT_SCHEMA",
    "PostRunResult",
    "RAW_MANIFEST_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "execute_formal_lifecycle",
    "execute_postrun_pipeline",
]
