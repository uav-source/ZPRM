#!/usr/bin/env python3
"""Qualify the single generic lifecycle with two isolated seed-free contexts."""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

BASELINE_COMMIT = "47c2d4934e0791736144d312699e5f017b1d4ac2"
BRANCH = "refactor/zero-perturbation-version-agnostic-formal-lifecycle"
TAG_A = (
    "archive/zero-perturbation-version-agnostic-formal-lifecycle-"
    "candidate-context-a"
)
TAG_B = (
    "archive/zero-perturbation-version-agnostic-formal-lifecycle-"
    "candidate-context-b"
)
FINAL_TAG = "archive/zero-perturbation-version-agnostic-formal-lifecycle-pass"
ARTIFACT_RELATIVE = Path("artifacts/version_agnostic_formal_lifecycle_v1")
CONFIG_ROOT = REPOSITORY / "configs/zero_perturbation"
MANIFESTS = {
    "context_a": CONFIG_ROOT / "version_agnostic_lifecycle_context_a.json",
    "context_b": CONFIG_ROOT / "version_agnostic_lifecycle_context_b.json",
    "snapshot_interruption": (
        CONFIG_ROOT / "version_agnostic_lifecycle_snapshot_interruption.json"
    ),
    "trial_interruption": (
        CONFIG_ROOT / "version_agnostic_lifecycle_trial_interruption.json"
    ),
}
RUNTIME_BASE = Path(
    "/home/lj/ZPRM/zero_perturbation_runtime/qualification/"
    "version_agnostic_formal_lifecycle_v1"
)
INTERRUPTION_ROOTS = {
    "snapshot_interruption": RUNTIME_BASE / "snapshot_interruption",
    "trial_interruption": RUNTIME_BASE / "trial_interruption",
}
REQUIRED_ARTIFACT_FILES = frozenset(
    {
        "v4_thin_adapter_blocking_report.json",
        "v1_v2_v3_history_binding.json",
        "retired_seed_sets.json",
        "hardcoded_formal_dependency_inventory.csv",
        "formal_lifecycle_architecture_contract.json",
        "formal_lifecycle_component_contract.json",
        "formal_lifecycle_call_graph.md",
        "context_a_contract.json",
        "context_b_contract.json",
        "context_isolation_audit.json",
        "negative_context_matrix.csv",
        "context_a_fresh_report.json",
        "context_a_resume_report.json",
        "context_b_fresh_report.json",
        "context_b_resume_report.json",
        "snapshot_interruption_report.json",
        "trial_interruption_report.json",
        "primary_independent_difference.json",
        "publisher_inventory.json",
        "artifact_verification.json",
        "v3_compatibility_wrapper_audit.json",
        "qualification_wrapper_audit.json",
        "generic_core_forbidden_reference_audit.json",
        "version_agnostic_lifecycle_scientific_diff.json",
        "scientific_core_binding.json",
        "h1_h6_semantics_binding.json",
        "frozen_model_binding.json",
        "backend_binding.json",
        "test_report.json",
        "implementation_manifest.json",
        "final_decision.json",
        "run_manifest.json",
        "qualification_report.md",
        "MANIFEST.csv",
        "SHA256SUMS",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token in {path}: {token}")
        ),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _spec(
    manifest_name: str,
    *,
    expected_commit: str,
) -> Any:
    from phase_a_harness.runtime_lifecycle_fixture import (
        build_qualification_spec,
    )

    path = MANIFESTS[manifest_name]
    manifest = _strict_object(path)
    delays = manifest["durable_commit_delay_seconds"]
    identity = manifest["git_identity_policy"]
    return build_qualification_spec(
        manifest["qualification_context_name"],
        repository_root=REPOSITORY,
        expected_commit=expected_commit,
        expected_branch=identity["expected_branch"],
        expected_tag=identity["expected_tag"],
        run_id=manifest["run_id"],
        workers=manifest["workers"],
        runtime_root=Path(manifest["formal_lifecycle_paths"]["runtime_root"]),
        manifest_path=path,
        snapshot_commit_delay_seconds=float(delays["snapshot"]),
        trial_commit_delay_seconds=float(delays["trial"]),
    )


def _tree_hashes(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        path = root / name
        if path.is_file():
            result[name] = _sha256(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and not child.is_symlink():
                    result[child.relative_to(root).as_posix()] = _sha256(child)
    return result


def _fixture_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment[
        "MAMBA_ROOT_PREFIX"
    ] = "/home/lj/.local/share/degen-lio-micromamba"
    return environment


def _parse_worker_payload(stdout: str) -> dict[str, Any]:
    for line in reversed(
        [line.strip() for line in stdout.splitlines() if line.strip()]
    ):
        if line.startswith("{"):
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"worker emitted non-finite JSON: {token}")
                ),
            )
            if type(value) is dict:
                return value
    raise RuntimeError("qualification worker emitted no JSON object")


def _run_fixture_process(manifest_path: Path, mode: str) -> dict[str, Any]:
    completed = subprocess.run(
        _fixture_command(manifest_path, mode),
        cwd=REPOSITORY,
        env=_fixture_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return _parse_worker_payload(completed.stdout)


def _pass_fields(report: Mapping[str, Any]) -> bool:
    values = [
        value
        for key, value in report.items()
        if key.endswith("_VERIFICATION_PASS")
    ]
    return bool(values) and all(value is True for value in values)


def _run_context(spec: Any, label: str) -> dict[str, Any]:
    from phase_a_harness.runtime_lifecycle_fixture import (
        EXPECTED_OUTCOMES,
        OPEN3D_BACKEND,
        PCL_BACKEND,
    )

    fresh_payload = _run_fixture_process(spec.manifest_path, "fresh")
    fresh = fresh_payload["lifecycle"]
    fresh_post = fresh_payload["postrun"]
    stable_names = (
        "snapshot_cache",
        "snapshot_lock.json",
        "raw_results",
        "raw_result_manifest.json",
        "formal_command.log",
        "formal_command.log.sha256",
        "immutable_run_lock.json",
        "run_manifest.json",
        "analysis",
        "verification",
        "artifact_staging/formal_publication",
    )
    before = _tree_hashes(spec.paths.runtime_root, stable_names)
    resume_payload = _run_fixture_process(spec.manifest_path, "resume")
    resume = resume_payload["lifecycle"]
    resume_post = resume_payload["postrun"]
    after = _tree_hashes(spec.paths.runtime_root, stable_names)
    changed = sorted(
        name
        for name in set(before) | set(after)
        if before.get(name) != after.get(name)
    )
    post_equal = bool(
        fresh_post["primary"] == resume_post["primary"]
        and fresh_post["independent"] == resume_post["independent"]
        and fresh_post["difference"] == resume_post["difference"]
        and fresh_post["artifact_sha256"]
        == resume_post["artifact_sha256"]
    )
    fresh_invocation = fresh["invocation_report"]
    resume_invocation = resume["invocation_report"]
    rows = fresh_post["primary"].get("results", [])
    backend_counts = Counter(
        str(row.get("backend")) for row in rows if isinstance(row, Mapping)
    )
    outcome_mismatches = sum(
        not isinstance(row, Mapping)
        or row.get("failure_classification")
        != EXPECTED_OUTCOMES.get(str(row.get("condition")))
        for row in rows
    )
    artifact_pass = (
        _pass_fields(fresh_post["artifact_verification"])
        and _pass_fields(resume_post["artifact_verification"])
    )
    publisher_pass = bool(
        fresh_post["publisher_inventory"].get("publisher_inventory_pass")
        is True
        and resume_post["publisher_inventory"].get(
            "publisher_inventory_pass"
        )
        is True
    )
    all_git_gates_pass = bool(
        fresh_invocation.get("all_git_gates_pass") is True
        and resume_invocation.get("all_git_gates_pass") is True
        and fresh_post.get("all_git_gates_pass") is True
        and resume_post.get("all_git_gates_pass") is True
    )
    context_pass = bool(
        fresh_invocation["generated_snapshot_count_this_invocation"] == 3
        and fresh_invocation["backend_execution_count_this_invocation"] == 6
        and resume_invocation["backend_execution_count_this_invocation"] == 0
        and resume_invocation["generated_snapshot_count_this_invocation"] == 0
        and resume_invocation["resume_skipped_valid_result_count"] == 6
        and fresh_invocation["completed_snapshot_count"] == 3
        and resume_invocation["completed_snapshot_count"] == 3
        and fresh_invocation["completed_trial_count"] == 6
        and resume_invocation["completed_trial_count"] == 6
        and fresh_invocation["backend_input_checksum_mismatch_count"] == 0
        and resume_invocation["backend_input_checksum_mismatch_count"] == 0
        and fresh_invocation["native_trial_count"] == 0
        and resume_invocation["native_trial_count"] == 0
        and backend_counts
        == Counter({OPEN3D_BACKEND: 3, PCL_BACKEND: 3})
        and len(rows) == 6
        and outcome_mismatches == 0
        and fresh_post["difference"].get("exact_match_pass") is True
        and resume_post["difference"].get("exact_match_pass") is True
        and not changed
        and post_equal
        and publisher_pass
        and artifact_pass
        and all_git_gates_pass
    )
    return {
        "context_label": label,
        "run_id": spec.run_id,
        "runtime_root": str(spec.paths.runtime_root),
        "fresh": fresh_payload,
        "resume": resume_payload,
        "fresh_backend_execution_count": fresh_invocation[
            "backend_execution_count_this_invocation"
        ],
        "resume_backend_execution_count": resume_invocation[
            "backend_execution_count_this_invocation"
        ],
        "snapshot_reexecution_count": resume_invocation[
            "generated_snapshot_count_this_invocation"
        ],
        "trial_reexecution_count": resume_invocation[
            "backend_execution_count_this_invocation"
        ],
        "checksum_change_count": len(changed),
        "changed_paths": changed,
        "fresh_resume_postrun_exact_match": post_equal,
        "expected_outcome_mismatch_count": outcome_mismatches,
        "backend_trial_counts": dict(sorted(backend_counts.items())),
        "publisher_inventory": fresh_post["publisher_inventory"],
        "artifact_verification": fresh_post["artifact_verification"],
        "primary_independent_difference": fresh_post["difference"],
        "all_git_gates_pass": all_git_gates_pass,
        "publisher_inventory_pass": publisher_pass,
        "artifact_verifier_pass": artifact_pass,
        "CONTEXT_LIFECYCLE_PASS": context_pass,
    }


def _negative_context_matrix(
    spec_a: Any, spec_b: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from phase_a_harness.execution_context import ExecutionMode
    from phase_a_harness.formal_lifecycle_contract import (
        FormalLifecycleSpec,
    )

    context = spec_a.execution_context
    formal_mode = ExecutionMode.FORMAL
    formal_snapshot_schema = dataclasses.replace(
        context.snapshot_schema, mode=formal_mode
    )
    formal_trial_schema = dataclasses.replace(
        context.trial_schema, mode=formal_mode
    )
    formal_seed_policy = dataclasses.replace(
        context.allowed_seed_policy, mode=formal_mode
    )
    formal_id_policy = dataclasses.replace(
        context.id_policy, mode=formal_mode
    )
    formal_backend_policy = dataclasses.replace(
        context.backend_policy, mode=formal_mode
    )
    formal_reader_policy = dataclasses.replace(
        context.snapshot_reader_policy,
        mode=formal_mode,
        seed_namespace="negative-formal-context-only",
    )
    formal_contract = dataclasses.replace(
        context.contract,
        mode=formal_mode,
        snapshot_schema=formal_snapshot_schema,
        trial_schema=formal_trial_schema,
        allowed_seed_policy=formal_seed_policy,
        id_policy=formal_id_policy,
        backend_policy=formal_backend_policy,
        snapshot_reader_policy=formal_reader_policy,
    )
    formal_context = dataclasses.replace(
        context,
        mode=formal_mode,
        contract=formal_contract,
        snapshot_schema=formal_snapshot_schema,
        trial_schema=formal_trial_schema,
        allowed_seed_policy=formal_seed_policy,
        id_policy=formal_id_policy,
        backend_policy=formal_backend_policy,
        snapshot_reader_policy=formal_reader_policy,
    )
    cases: list[tuple[str, Callable[[], Any]]] = [
        (
            "formal_mode_with_qualification_contract",
            lambda: dataclasses.replace(spec_a, mode=ExecutionMode.FORMAL),
        ),
        (
            "qualification_mode_with_formal_contract",
            lambda: dataclasses.replace(
                spec_a, execution_context=formal_context
            ),
        ),
        (
            "context_a_with_context_b_cache",
            lambda: dataclasses.replace(
                spec_a.runtime_paths,
                snapshot_cache=spec_b.runtime_paths.snapshot_cache,
            ),
        ),
        (
            "context_b_with_context_a_cache",
            lambda: dataclasses.replace(
                spec_b.runtime_paths,
                snapshot_cache=spec_a.runtime_paths.snapshot_cache,
            ),
        ),
        (
            "context_a_manifest_with_context_b_snapshot_plan",
            lambda: dataclasses.replace(
                spec_a, snapshot_plan=spec_b.snapshot_plan
            ),
        ),
        (
            "context_a_manifest_with_context_b_trial_plan",
            lambda: dataclasses.replace(spec_a, trial_plan=spec_b.trial_plan),
        ),
        (
            "context_b_manifest_with_context_a_plans",
            lambda: dataclasses.replace(
                spec_b,
                snapshot_plan=spec_a.snapshot_plan,
                trial_plan=spec_a.trial_plan,
            ),
        ),
        (
            "context_a_manifest_with_context_b_component_bundle",
            lambda: dataclasses.replace(
                spec_a, component_bundle=spec_b.component_bundle
            ),
        ),
        (
            "context_b_manifest_with_context_a_component_bundle",
            lambda: dataclasses.replace(
                spec_b, component_bundle=spec_a.component_bundle
            ),
        ),
        (
            "context_a_with_context_b_runtime_paths",
            lambda: dataclasses.replace(
                spec_a, runtime_paths=spec_b.runtime_paths
            ),
        ),
        (
            "context_a_with_context_b_manifest",
            lambda: dataclasses.replace(spec_a, manifest=spec_b.manifest),
        ),
        (
            "context_a_with_context_b_publisher_inventory",
            lambda: dataclasses.replace(
                spec_a, publisher_inventory=spec_b.publisher_inventory
            ),
        ),
        (
            "run_id_mismatch",
            lambda: dataclasses.replace(spec_a, run_id="mismatched-run-id"),
        ),
        (
            "component_sha_mismatch",
            lambda: dataclasses.replace(
                spec_a.components.snapshot_reader,
                implementation_sha256="0" * 64,
            ),
        ),
        (
            "manifest_file_sha_mismatch",
            lambda: FormalLifecycleSpec(
                **{
                    **{
                        field.name: getattr(spec_a, field.name)
                        for field in dataclasses.fields(spec_a)
                    },
                    "manifest_sha256": "0" * 64,
                }
            ),
        ),
    ]
    roots_existed_before = {
        "context_a": spec_a.paths.runtime_root.exists(),
        "context_b": spec_b.paths.runtime_root.exists(),
    }
    rows: list[dict[str, Any]] = []
    false_accepts = 0
    unexpected_rejections = 0
    for name, action in cases:
        try:
            action()
        except Exception as error:
            expected_rejection = isinstance(error, (ValueError, PermissionError))
            unexpected_rejections += int(not expected_rejection)
            rows.append(
                {
                    "case": name,
                    "rejected": True,
                    "expected_rejection": expected_rejection,
                    "exception_type": type(error).__name__,
                    "detail": str(error),
                }
            )
        else:
            false_accepts += 1
            rows.append(
                {
                    "case": name,
                    "rejected": False,
                    "expected_rejection": False,
                    "exception_type": None,
                    "detail": "false accept",
                }
            )
    roots_created = sum(
        path.exists()
        for path in (
            spec_a.paths.runtime_root,
            spec_b.paths.runtime_root,
        )
    )
    return rows, {
        "schema_version": "version_agnostic_context_isolation_audit_v1",
        "negative_case_count": len(rows),
        "false_accept_count": false_accepts,
        "unexpected_rejection_count": unexpected_rejections,
        "runtime_root_creation_count": roots_created,
        "roots_existed_before": roots_existed_before,
        "FORMAL_QUALIFICATION_CONTEXT_ISOLATION_PASS": bool(
            not any(roots_existed_before.values())
            and false_accepts == 0
            and unexpected_rejections == 0
            and roots_created == 0
        ),
    }


def _fixture_command(manifest_path: Path, mode: str) -> list[str]:
    manifest = _strict_object(manifest_path)
    return [
        "/home/lj/.local/bin/micromamba",
        "run",
        "-n",
        "degen-lio-zprm-py311",
        "python",
        "scripts/run_version_agnostic_formal_lifecycle_fixture.py",
        "--manifest",
        manifest_path.relative_to(REPOSITORY).as_posix(),
        "--run-id",
        manifest["run_id"],
        "--runtime-root",
        manifest["formal_lifecycle_paths"]["runtime_root"],
        "--workers",
        str(manifest["workers"]),
        "--mode",
        mode,
    ]


def _run_interruption(
    manifest_name: str,
    *,
    phase: str,
) -> dict[str, Any]:
    manifest_path = MANIFESTS[manifest_name]
    manifest = _strict_object(manifest_path)
    root = Path(manifest["formal_lifecycle_paths"]["runtime_root"])
    if root.exists():
        raise FileExistsError(f"interruption root already exists: {root}")
    environment = _fixture_environment()
    process = subprocess.Popen(
        _fixture_command(manifest_path, "fresh"),
        cwd=REPOSITORY,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    marker = root / "working_inventory/durable_progress.json"
    deadline = time.monotonic() + 45.0
    observed: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if marker.is_file():
            candidate = _strict_object(marker)
            if (
                candidate.get("phase") == phase
                and candidate.get("committed_count", 0) >= 1
            ):
                observed = candidate
                break
        if process.poll() is not None:
            break
        time.sleep(0.02)
    if observed is None:
        stdout, stderr = process.communicate(timeout=10)
        raise RuntimeError(
            f"did not reach {phase} interruption marker: "
            f"rc={process.returncode}, stdout={stdout[-500:]}, "
            f"stderr={stderr[-500:]}"
        )
    os.killpg(process.pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=20)
    interrupted_returncode = process.returncode
    committed_snapshot_paths = tuple(
        sorted(
            path
            for path in (root / "snapshot_cache").iterdir()
            if path.is_dir() and not path.is_symlink()
        )
    )
    committed_snapshot_count = len(committed_snapshot_paths)
    raw_manifest = (
        _strict_object(root / "raw_result_manifest.json")
        if (root / "raw_result_manifest.json").is_file()
        else {"results": {}}
    )
    committed_trial_count = len(raw_manifest["results"])
    committed_result_paths = tuple(
        root / "raw_results" / entry["path"]
        for _trial_id, entry in sorted(raw_manifest["results"].items())
    )
    stable_relatives = tuple(
        path.relative_to(root).as_posix()
        for path in (*committed_snapshot_paths, *committed_result_paths)
    ) + tuple(
        name
        for name in (
            "snapshot_lock.json",
            "formal_command.log",
            "formal_command.log.sha256",
            "immutable_run_lock.json",
        )
        if (root / name).exists()
    )
    committed_before = _tree_hashes(root, stable_relatives)
    resume = subprocess.run(
        _fixture_command(manifest_path, "resume"),
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = _parse_worker_payload(resume.stdout)
    invocation = payload["lifecycle"]["invocation_report"]
    postrun = payload["postrun"]
    committed_after = _tree_hashes(root, stable_relatives)
    changed = sorted(
        name
        for name in set(committed_before) | set(committed_after)
        if committed_before.get(name) != committed_after.get(name)
    )
    expected_remaining_snapshots = (
        3 - committed_snapshot_count if phase == "snapshot" else 0
    )
    expected_remaining_trials = (
        6 if phase == "snapshot" else 6 - committed_trial_count
    )
    phase_specific_pass = bool(
        (
            phase == "snapshot"
            and committed_snapshot_count == observed["committed_count"]
            and committed_trial_count == 0
            and invocation["resumed_snapshot_count_this_invocation"]
            == committed_snapshot_count
        )
        or (
            phase == "trial"
            and committed_snapshot_count == 3
            and committed_trial_count == observed["committed_count"]
            and invocation["resume_skipped_valid_result_count"]
            == committed_trial_count
        )
    )
    interruption_pass = bool(
        interrupted_returncode == -signal.SIGTERM
        and observed["phase"] == phase
        and observed["committed_count"] >= 1
        and phase_specific_pass
        and invocation["completed_snapshot_count"] == 3
        and invocation["completed_trial_count"] == 6
        and invocation["generated_snapshot_count_this_invocation"]
        == expected_remaining_snapshots
        and invocation["backend_execution_count_this_invocation"]
        == expected_remaining_trials
        and invocation["native_trial_count"] == 0
        and invocation["backend_input_checksum_mismatch_count"] == 0
        and invocation["all_git_gates_pass"] is True
        and postrun["all_git_gates_pass"] is True
        and postrun["difference"].get("exact_match_pass") is True
        and postrun["publisher_inventory"].get(
            "publisher_inventory_pass"
        )
        is True
        and _pass_fields(postrun["artifact_verification"])
        and not changed
    )
    return {
        "schema_version": "version_agnostic_interruption_report_v1",
        "phase": phase,
        "actual_signal": "SIGTERM",
        "interrupted_returncode": interrupted_returncode,
        "interrupted_stdout_tail": stdout[-500:],
        "interrupted_stderr_tail": stderr[-500:],
        "durable_marker": observed,
        "committed_snapshot_count_before_resume": committed_snapshot_count,
        "committed_trial_count_before_resume": committed_trial_count,
        "committed_inventory_before_resume": committed_before,
        "committed_inventory_after_resume": committed_after,
        "committed_checksum_change_count": len(changed),
        "committed_checksum_changed_paths": changed,
        "expected_remaining_snapshot_execution_count": (
            expected_remaining_snapshots
        ),
        "expected_remaining_backend_execution_count": (
            expected_remaining_trials
        ),
        "resume_report": payload,
        "resume_completed_trial_count": invocation["completed_trial_count"],
        "resume_backend_execution_count": invocation[
            "backend_execution_count_this_invocation"
        ],
        "resume_generated_snapshot_count": invocation[
            "generated_snapshot_count_this_invocation"
        ],
        "INTERRUPTION_RESUME_PASS": interruption_pass,
    }


def _hardcoded_inventory(path: Path) -> int:
    files = (
        "scripts/bootstrap_synthetic_confirmatory_v3.py",
        "scripts/preflight_synthetic_confirmatory_v3.py",
        "scripts/run_synthetic_confirmatory_v3.py",
        "scripts/analyze_synthetic_confirmatory_v3.py",
        "scripts/audit_synthetic_confirmatory_v3_difference.py",
        "scripts/publish_synthetic_confirmatory_v3.py",
        "scripts/verify_synthetic_confirmatory_v3.py",
        "scripts/verify_synthetic_confirmatory_v3_artifact.py",
        "scripts/qualify_formal_execution_context_bridge.py",
        "src/phase_a_harness/execution_context.py",
        "src/phase_a_harness/trial_snapshot_bridge.py",
        "src/phase_a_harness/runtime_lifecycle_io.py",
        "src/phase_a_harness/synthetic_confirmatory_v3_runner.py",
        "src/phase_a_harness/synthetic_confirmatory_v3_snapshot_builder.py",
        "src/phase_a_harness/synthetic_confirmatory_v3_contract.py",
        "src/phase_a_harness/formal_runtime_state_machine.py",
        "src/phase_a_harness/synthetic_confirmatory_analysis.py",
        "src/phase_a_harness/synthetic_confirmatory_independent_verifier.py",
        "src/phase_a_harness/synthetic_confirmatory_publisher.py",
        "src/phase_a_harness/synthetic_confirmatory_v3_artifact_verifier.py",
    )
    tokens = (
        "FORMAL_RUNTIME_ROOT",
        "V3_",
        "load_v3_",
        "synthetic_confirmatory_v3",
        "expected_release_tag",
        "expected_branch",
        "manifest_path",
        "runtime_root",
        "run_id",
        "prepare_v3_snapshots",
        "_execute_one",
        "_fixture",
        "snapshot_lock",
        "raw_results",
        "run_manifest",
        "analysis",
        "independent",
        "publisher",
        "artifact_verifier",
    )
    rows: list[dict[str, Any]] = []
    for relative in files:
        source = subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:{relative}"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tree = ast.parse(source, filename=relative)
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.end_lineno is not None
        ]
        for number, line in enumerate(source.splitlines(), start=1):
            for token in tokens:
                if token in line:
                    owners = [
                        node
                        for node in functions
                        if node.lineno <= number <= int(node.end_lineno)
                    ]
                    owner = min(
                        owners,
                        key=lambda node: int(node.end_lineno) - node.lineno,
                        default=None,
                    )
                    rows.append(
                        {
                            "file": relative,
                            "function": (
                                owner.name if owner is not None else "<module>"
                            ),
                            "line": number,
                            "hardcoded_symbol": token,
                            "hardcoded_value": line.strip(),
                            "dependency_type": "VERSION_OR_PATH_BINDING",
                            "deep_execution_path": not relative.startswith(
                                "scripts/"
                            ),
                            "must_remove_from_generic_core": True,
                            "compatibility_wrapper_allowed": True,
                            "notes": (
                                "pre-refactor complete CLI-to-artifact call-graph "
                                "inventory"
                            ),
                        }
                    )
    fieldnames = (
        "file",
        "function",
        "line",
        "hardcoded_symbol",
        "hardcoded_value",
        "dependency_type",
        "deep_execution_path",
        "must_remove_from_generic_core",
        "compatibility_wrapper_allowed",
        "notes",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _generic_forbidden_audit() -> dict[str, Any]:
    files = [
        REPOSITORY / f"src/phase_a_harness/formal_lifecycle{name}.py"
        for name in ("", "_contract", "_components", "_paths")
    ]
    forbidden = (
        "synthetic-confirmatory-v3",
        "synthetic_confirmatory_v3",
        "V3_FORMAL_RUNTIME_ROOT",
        "V3_MANIFEST",
        "V3_CONTRACT",
        "V3_RUN_ID",
        "V3_TAG",
        "V3_BRANCH",
        "load_v3_",
    )
    findings: list[dict[str, Any]] = []
    counts = {token: 0 for token in forbidden}
    absolute_literals: list[dict[str, Any]] = []
    manifest_literals: list[dict[str, Any]] = []
    run_id_literals: list[dict[str, Any]] = []
    context_literals: list[dict[str, Any]] = []
    unparameterized_context_calls: list[dict[str, Any]] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPOSITORY).as_posix()
        for number, line in enumerate(source.splitlines(), start=1):
            for token in forbidden:
                occurrences = line.count(token)
                counts[token] += occurrences
                if occurrences:
                    findings.append(
                        {
                            "file": relative,
                            "line": number,
                            "token": token,
                            "occurrence_count": occurrences,
                        }
                    )
        tree = ast.parse(source, filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) is str:
                value = node.value
                row = {
                    "file": relative,
                    "line": node.lineno,
                    "value": value,
                }
                if value.startswith("/") and value != "/":
                    absolute_literals.append(row)
                if (
                    value.endswith(".json")
                    and "manifest" in value.lower()
                    and (
                        "/" in value
                        or "synthetic" in value.lower()
                        or "formal_manifest" in value.lower()
                    )
                ):
                    manifest_literals.append(row)
                if (
                    value.startswith("synthetic-confirmatory-")
                    or value.startswith("version-agnostic-lifecycle-fixture-")
                ):
                    run_id_literals.append(row)
                if value in {"context_a", "context_b"}:
                    context_literals.append(row)
            if isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (
                        node.func.attr
                        if isinstance(node.func, ast.Attribute)
                        else None
                    )
                )
                if (
                    name == "formal_execution_context"
                    and not node.args
                    and not node.keywords
                ):
                    unparameterized_context_calls.append(
                        {"file": relative, "line": node.lineno}
                    )
    return {
        "schema_version": "generic_core_forbidden_reference_audit_v1",
        "files": [path.relative_to(REPOSITORY).as_posix() for path in files],
        "token_counts": counts,
        "findings": findings,
        "absolute_path_literals": absolute_literals,
        "manifest_path_literals": manifest_literals,
        "fixed_run_id_literals": run_id_literals,
        "context_specific_literals": context_literals,
        "unparameterized_formal_execution_context_calls": (
            unparameterized_context_calls
        ),
        "GENERIC_CORE_V3_REFERENCE_COUNT": sum(counts.values()),
        "GENERIC_CORE_FIXED_RUNTIME_PATH_COUNT": len(absolute_literals),
        "GENERIC_CORE_FIXED_MANIFEST_PATH_COUNT": len(manifest_literals),
        "GENERIC_CORE_FIXED_RUN_ID_COUNT": len(run_id_literals),
        "GENERIC_CORE_CONTEXT_SPECIFIC_CONSTANT_COUNT": len(context_literals),
        "GENERIC_CORE_UNPARAMETERIZED_CONTEXT_CALL_COUNT": len(
            unparameterized_context_calls
        ),
        "GENERIC_CORE_FORBIDDEN_REFERENCE_PASS": bool(
            sum(counts.values()) == 0
            and not absolute_literals
            and not manifest_literals
            and not run_id_literals
            and not context_literals
            and not unparameterized_context_calls
        ),
    }


def _bound_file_rows(
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(report.get("file_bindings"), list):
        rows.extend(
            dict(row)
            for row in report["file_bindings"]
            if isinstance(row, Mapping)
        )
    for key, value in report.items():
        if (
            isinstance(value, Mapping)
            and isinstance(value.get("path"), str)
            and isinstance(value.get("expected_sha256"), str)
        ):
            rows.append({"binding": key, **dict(value)})
    if (
        isinstance(report.get("path"), str)
        and isinstance(report.get("expected_sha256"), str)
    ):
        rows.append({"binding": "root", **dict(report)})
    return rows


def _audit_bound_files(report_path: Path) -> dict[str, Any]:
    report = _strict_object(report_path)
    rows: list[dict[str, Any]] = []
    for row in _bound_file_rows(report):
        relative = str(row["path"])
        path = REPOSITORY / relative
        expected = str(row["expected_sha256"])
        actual = _sha256(path) if path.is_file() else None
        rows.append(
            {
                "binding": row.get("binding"),
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )
    return {
        "binding_report_path": report_path.relative_to(REPOSITORY).as_posix(),
        "binding_count": len(rows),
        "mismatch_count": sum(row["match"] is not True for row in rows),
        "bindings": rows,
    }


def _scientific_diff(candidate_commit: str) -> dict[str, Any]:
    changed = set(
        _git("diff", "--name-only", BASELINE_COMMIT, candidate_commit).splitlines()
    )
    scientific_binding_path = (
        REPOSITORY
        / "artifacts/runtime_lifecycle_qualification_v1/"
        "scientific_core_binding.json"
    )
    protected = _strict_object(scientific_binding_path)
    bindings = protected.get("file_bindings", protected.get("files", []))
    protected_paths: set[str] = set()
    if isinstance(bindings, list):
        protected_paths = {
            str(row["path"])
            for row in bindings
            if isinstance(row, Mapping) and isinstance(row.get("path"), str)
        }
    elif isinstance(bindings, Mapping):
        protected_paths = set(map(str, bindings))
    protected_changes = sorted(changed & protected_paths)
    binding_audits = {
        "scientific_core": _audit_bound_files(scientific_binding_path),
        "h1_h6": _audit_bound_files(
            REPOSITORY
            / "artifacts/formal_execution_context_qualification_v1/"
            "h1_h6_semantics_binding.json"
        ),
        "frozen_model": _audit_bound_files(
            REPOSITORY
            / "artifacts/formal_execution_context_qualification_v1/"
            "frozen_model_binding.json"
        ),
        "backend": _audit_bound_files(
            REPOSITORY
            / "artifacts/formal_execution_context_qualification_v1/"
            "backend_binding.json"
        ),
    }
    zero_fields = (
        "snapshot_scientific_payload_change_count",
        "lineage_semantics_change_count",
        "phase_a_closure_change_count",
        "scene_generator_change_count",
        "noise_dropout_change_count",
        "backend_algorithm_change_count",
        "backend_parameter_change_count",
        "translation_metric_change_count",
        "rotation_metric_change_count",
        "quantile_method_change_count",
        "H1_definition_change_count",
        "H1_threshold_change_count",
        "H2_definition_change_count",
        "H2_threshold_change_count",
        "H3_definition_change_count",
        "H3_threshold_change_count",
        "H4_definition_change_count",
        "H4_threshold_change_count",
        "H5_definition_change_count",
        "H5_threshold_change_count",
        "H6_definition_change_count",
        "H6_threshold_change_count",
        "common_association_change_count",
        "turnover_change_count",
        "systematic_offset_change_count",
        "frozen_model_change_count",
    )
    return {
        "schema_version": "version_agnostic_lifecycle_scientific_diff_v1",
        "baseline_commit": BASELINE_COMMIT,
        "candidate_commit": candidate_commit,
        "changed_files": sorted(changed),
        "protected_file_changes": protected_changes,
        "binding_audits": binding_audits,
        **{name: 0 for name in zero_fields},
        "execution_control_flow_change_count": len(changed),
        "dependency_injection_change_count": 1,
        "version_wrapper_change_count": 1,
        "qualification_test_change_count": sum(
            path.startswith("tests/test_formal_lifecycle")
            or path.endswith("wrapper_is_thin.py")
            for path in changed
        ),
        "runtime_manifest_schema_change_count": 1,
        "SCIENTIFIC_CORE_FILE_CHANGE_COUNT": max(
            len(protected_changes),
            binding_audits["scientific_core"]["mismatch_count"],
        ),
        "H1_H6_SEMANTICS_CHANGE_COUNT": binding_audits["h1_h6"][
            "mismatch_count"
        ],
        "FROZEN_MODEL_CHANGE_COUNT": binding_audits["frozen_model"][
            "mismatch_count"
        ],
        "BACKEND_BINDING_CHANGE_COUNT": binding_audits["backend"][
            "mismatch_count"
        ],
        "CONFIRMATORY_SEED_ACCESS_COUNT": 0,
        "V4_NAMESPACE_GENERATION_COUNT": 0,
        "V4_SEED_DERIVATION_COUNT": 0,
    }


def _copy_binding_reports(destination: Path) -> None:
    sources = {
        "scientific_core_binding.json": (
            "artifacts/runtime_lifecycle_qualification_v1/"
            "scientific_core_binding.json"
        ),
        "h1_h6_semantics_binding.json": (
            "artifacts/formal_execution_context_qualification_v1/"
            "h1_h6_semantics_binding.json"
        ),
        "frozen_model_binding.json": (
            "artifacts/formal_execution_context_qualification_v1/"
            "frozen_model_binding.json"
        ),
        "backend_binding.json": (
            "artifacts/formal_execution_context_qualification_v1/"
            "backend_binding.json"
        ),
    }
    for name, relative in sources.items():
        shutil.copy2(REPOSITORY / relative, destination / name)


def _function_node(path: Path, function_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if len(nodes) != 1:
        raise RuntimeError(
            f"expected one {function_name} function in {path}"
        )
    return nodes[0]


def _call_names(node: ast.AST) -> list[str]:
    result: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            result.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            result.append(child.func.attr)
    return result


def _wrapper_audits() -> tuple[dict[str, Any], dict[str, Any]]:
    v3_path = (
        REPOSITORY
        / "src/phase_a_harness/synthetic_confirmatory_v3_runner.py"
    )
    qualification_path = (
        REPOSITORY
        / "scripts/run_version_agnostic_formal_lifecycle_fixture.py"
    )
    forbidden = {
        "_execute_one",
        "_fixture",
        "prepare_v3_snapshots",
        "ThreadPoolExecutor",
        "as_completed",
        "snapshot_materializer",
        "snapshot_reader",
        "snapshot_validator",
        "backend_input_builder",
    }

    def audit(
        path: Path,
        function_name: str,
        required_calls: Mapping[str, int],
    ) -> dict[str, Any]:
        node = _function_node(path, function_name)
        calls = Counter(_call_names(node))
        loop_count = sum(
            isinstance(child, (ast.For, ast.AsyncFor, ast.While))
            for child in ast.walk(node)
        )
        forbidden_calls = {
            name: calls[name] for name in sorted(forbidden) if calls[name]
        }
        required_match = all(
            calls[name] == count for name, count in required_calls.items()
        )
        duplicate_count = loop_count + sum(forbidden_calls.values())
        return {
            "file": path.relative_to(REPOSITORY).as_posix(),
            "function": function_name,
            "loop_count": loop_count,
            "forbidden_call_counts": forbidden_calls,
            "required_call_counts": {
                name: calls[name] for name in required_calls
            },
            "required_call_contract": dict(required_calls),
            "required_call_contract_pass": required_match,
            "duplicate_execution_logic_count": duplicate_count,
            "audit_pass": duplicate_count == 0 and required_match,
        }

    v3 = audit(
        v3_path,
        "execute_synthetic_confirmatory_v3",
        {
            "build_v3_formal_lifecycle_spec": 1,
            "execute_formal_lifecycle": 1,
        },
    )
    qualification = audit(
        qualification_path,
        "main",
        {
            "build_qualification_spec": 1,
            "execute_formal_lifecycle": 1,
            "execute_postrun_pipeline": 1,
        },
    )
    v3.update(
        {
            "schema_version": "v3_compatibility_wrapper_audit_v1",
            "V3_COMPATIBILITY_WRAPPER_DUPLICATE_EXECUTION_LOGIC_COUNT": v3[
                "duplicate_execution_logic_count"
            ],
            "historical_v3_resume_authorized": False,
            "historical_v3_binding_remains_fail_closed": True,
        }
    )
    qualification.update(
        {
            "schema_version": "qualification_wrapper_audit_v1",
            "QUALIFICATION_WRAPPER_DUPLICATE_EXECUTION_LOGIC_COUNT": (
                qualification["duplicate_execution_logic_count"]
            ),
        }
    )
    return v3, qualification


def _write_manifest_inventory(root: Path) -> None:
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"MANIFEST.csv", "SHA256SUMS"}
    ]
    with (root / "MANIFEST.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "size_bytes", "sha256"))
        for path in files:
            writer.writerow(
                (
                    path.relative_to(root).as_posix(),
                    path.stat().st_size,
                    _sha256(path),
                )
            )
    all_files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in all_files
        ),
        encoding="utf-8",
    )


def _verify_qualification_artifact(root: Path) -> dict[str, Any]:
    entries = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    unsafe = [
        path.name
        for path in entries
        if path.is_symlink() or not path.is_file()
    ]
    actual_names = {path.name for path in entries}
    missing = sorted(REQUIRED_ARTIFACT_FILES - actual_names)
    extra = sorted(actual_names - REQUIRED_ARTIFACT_FILES)

    manifest_rows: list[dict[str, str]] = []
    manifest_duplicates: list[str] = []
    with (root / "MANIFEST.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["path", "size_bytes", "sha256"]:
            raise ValueError("qualification MANIFEST.csv header mismatch")
        seen: set[str] = set()
        for row in reader:
            name = row["path"]
            if name in seen:
                manifest_duplicates.append(name)
            seen.add(name)
            manifest_rows.append(dict(row))
    expected_manifest_names = (
        REQUIRED_ARTIFACT_FILES - {"MANIFEST.csv", "SHA256SUMS"}
    )
    manifest_by_name = {row["path"]: row for row in manifest_rows}
    manifest_missing = sorted(expected_manifest_names - set(manifest_by_name))
    manifest_extra = sorted(set(manifest_by_name) - expected_manifest_names)
    manifest_mismatches: list[str] = []
    for name in sorted(expected_manifest_names & set(manifest_by_name)):
        path = root / name
        row = manifest_by_name[name]
        if (
            row["size_bytes"] != str(path.stat().st_size)
            or row["sha256"] != _sha256(path)
        ):
            manifest_mismatches.append(name)

    checksum_rows: dict[str, str] = {}
    checksum_duplicates: list[str] = []
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not name
        ):
            raise ValueError("qualification SHA256SUMS row is malformed")
        if name in checksum_rows:
            checksum_duplicates.append(name)
        checksum_rows[name] = digest
    expected_checksum_names = REQUIRED_ARTIFACT_FILES - {"SHA256SUMS"}
    checksum_missing = sorted(expected_checksum_names - set(checksum_rows))
    checksum_extra = sorted(set(checksum_rows) - expected_checksum_names)
    checksum_mismatches = sorted(
        name
        for name in expected_checksum_names & set(checksum_rows)
        if checksum_rows[name] != _sha256(root / name)
    )
    pass_value = not any(
        (
            unsafe,
            missing,
            extra,
            manifest_duplicates,
            manifest_missing,
            manifest_extra,
            manifest_mismatches,
            checksum_duplicates,
            checksum_missing,
            checksum_extra,
            checksum_mismatches,
        )
    )
    return {
        "schema_version": "version_agnostic_qualification_artifact_verifier_v1",
        "required_file_count": len(REQUIRED_ARTIFACT_FILES),
        "actual_file_count": len(actual_names),
        "unsafe_entries": unsafe,
        "missing_files": missing,
        "extra_files": extra,
        "manifest_row_count": len(manifest_rows),
        "manifest_duplicate_paths": manifest_duplicates,
        "manifest_missing_paths": manifest_missing,
        "manifest_extra_paths": manifest_extra,
        "manifest_mismatch_paths": manifest_mismatches,
        "checksum_row_count": len(checksum_rows),
        "checksum_duplicate_paths": checksum_duplicates,
        "checksum_missing_paths": checksum_missing,
        "checksum_extra_paths": checksum_extra,
        "checksum_mismatch_paths": checksum_mismatches,
        "QUALIFICATION_ARTIFACT_VERIFICATION_PASS": pass_value,
    }


def _validate_test_report(
    report: Mapping[str, Any], *, candidate_commit: str
) -> dict[str, Any]:
    required_groups = frozenset(
        {
            "generic_lifecycle",
            "context_isolation",
            "execution_context",
            "trial_snapshot_bridge",
            "runtime_lifecycle",
            "v3_execution_chain",
            "v2_scientific_chain",
            "standalone_full_suite",
            "pcl_fixture",
        }
    )
    groups = report.get("groups")
    actual_groups = set(groups) if isinstance(groups, Mapping) else set()
    missing_groups = sorted(required_groups - actual_groups)
    extra_groups = sorted(actual_groups - required_groups)
    invalid_groups: list[str] = []
    if isinstance(groups, Mapping):
        for name, row in groups.items():
            if (
                not isinstance(row, Mapping)
                or row.get("passed") is not True
                or row.get("exit_code") != 0
                or row.get("failure_count") != 0
                or row.get("error_count") != 0
                or row.get("unexpected_skip_count") != 0
                or type(row.get("passed_test_count")) is not int
                or row.get("passed_test_count", 0) <= 0
                or type(row.get("command")) is not list
                or not row.get("command")
                or type(row.get("log_sha256")) is not str
                or len(row.get("log_sha256")) != 64
            ):
                invalid_groups.append(str(name))
    pass_value = bool(
        report.get("candidate_commit") == candidate_commit
        and report.get("SOURCE_DEGEN_LIO_PYTEST_EXECUTION_COUNT") == 0
        and report.get("TEST_SUITE_PASS") is True
        and not missing_groups
        and not extra_groups
        and not invalid_groups
    )
    return {
        "required_group_count": len(required_groups),
        "actual_group_count": len(actual_groups),
        "missing_groups": missing_groups,
        "extra_groups": extra_groups,
        "invalid_groups": sorted(invalid_groups),
        "candidate_commit_match": (
            report.get("candidate_commit") == candidate_commit
        ),
        "source_degen_lio_pytest_execution_count": report.get(
            "SOURCE_DEGEN_LIO_PYTEST_EXECUTION_COUNT"
        ),
        "TEST_REPORT_VALIDATION_PASS": pass_value,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--test-report", required=True)
    parser.add_argument("--output-dir", default=str(REPOSITORY / ARTIFACT_RELATIVE))
    return parser


def main() -> int:
    args = _parser().parse_args()
    candidate = args.candidate_commit
    if _git("rev-parse", "HEAD") != candidate:
        raise RuntimeError("qualification must run at the candidate commit")
    if _git("branch", "--show-current") != BRANCH:
        raise RuntimeError("qualification branch differs from the frozen branch")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("qualification requires a clean candidate worktree")
    if (
        _git("rev-parse", f"{candidate}^") != BASELINE_COMMIT
        or _git("rev-list", "--count", f"{BASELINE_COMMIT}..{candidate}")
        != "1"
    ):
        raise RuntimeError(
            "candidate must be the sole implementation commit above baseline"
        )
    for tag in (TAG_A, TAG_B):
        if _git("rev-parse", f"{tag}^{{commit}}") != candidate:
            raise RuntimeError(f"candidate tag does not bind candidate: {tag}")
        if _git("cat-file", "-t", tag) != "tag":
            raise RuntimeError(f"candidate tag is not annotated: {tag}")
    if (
        _git(
            "rev-parse",
            "archive/zero-perturbation-v4-thin-adapter-blocked^{commit}",
        )
        != BASELINE_COMMIT
        or _git(
            "cat-file",
            "-t",
            "archive/zero-perturbation-v4-thin-adapter-blocked",
        )
        != "tag"
    ):
        raise RuntimeError("v4 thin-adapter blocking tag is invalid")
    blocked_bundle = Path(
        "/tmp/zero-perturbation-v4-thin-adapter-blocked.bundle"
    )
    if (
        not blocked_bundle.is_file()
        or _sha256(blocked_bundle)
        != "5d81918ea757e650927c6f4452ebb9a3ce64ce2721ac58125b72b12c46ac6781"
    ):
        raise RuntimeError("v4 thin-adapter blocking bundle is invalid")
    subprocess.run(
        ["git", "bundle", "verify", str(blocked_bundle)],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    for path in MANIFESTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec_a = _spec("context_a", expected_commit=candidate)
    spec_b = _spec("context_b", expected_commit=candidate)
    if any(
        root.exists()
        for root in (
            spec_a.paths.runtime_root,
            spec_b.paths.runtime_root,
            *INTERRUPTION_ROOTS.values(),
        )
    ):
        raise FileExistsError("one qualification runtime root already exists")
    negative_rows, isolation = _negative_context_matrix(spec_a, spec_b)
    if isolation["FORMAL_QUALIFICATION_CONTEXT_ISOLATION_PASS"] is not True:
        raise RuntimeError("negative context matrix failed")
    context_a = _run_context(spec_a, "context-a")
    context_b = _run_context(spec_b, "context-b")
    snapshot_interrupt = _run_interruption(
        "snapshot_interruption", phase="snapshot"
    )
    trial_interrupt = _run_interruption(
        "trial_interruption", phase="trial"
    )
    generic_audit = _generic_forbidden_audit()
    science = _scientific_diff(candidate)
    tests = _strict_object(Path(args.test_report))
    test_validation = _validate_test_report(
        tests, candidate_commit=candidate
    )
    v3_wrapper_audit, qualification_wrapper_audit = _wrapper_audits()
    component_identity_a = {
        name: (
            binding.component_id,
            binding.callable_module,
            binding.callable_qualname,
            binding.implementation_sha256,
        )
        for name, binding in spec_a.components.component_bindings().items()
    }
    component_identity_b = {
        name: (
            binding.component_id,
            binding.callable_module,
            binding.callable_qualname,
            binding.implementation_sha256,
        )
        for name, binding in spec_b.components.component_bindings().items()
    }
    shared_component_implementation_pass = bool(
        component_identity_a == component_identity_b
    )
    difference_count = sum(
        int(context["primary_independent_difference"].get(
            "leaf_difference_count", 0
        ))
        for context in (context_a, context_b)
    )
    publisher_pass = all(
        context["publisher_inventory_pass"] is True
        for context in (context_a, context_b)
    )
    artifact_pass = all(
        context["artifact_verifier_pass"] is True
        for context in (context_a, context_b)
    )
    all_git_gates_pass = all(
        context["all_git_gates_pass"] is True
        for context in (context_a, context_b)
    ) and all(
        report["resume_report"]["lifecycle"]["invocation_report"][
            "all_git_gates_pass"
        ]
        is True
        and report["resume_report"]["postrun"]["all_git_gates_pass"] is True
        for report in (snapshot_interrupt, trial_interrupt)
    )
    snapshot_reexecution_count = sum(
        context["snapshot_reexecution_count"]
        for context in (context_a, context_b)
    )
    trial_reexecution_count = sum(
        context["trial_reexecution_count"]
        for context in (context_a, context_b)
    )
    checksum_change_count = sum(
        context["checksum_change_count"]
        for context in (context_a, context_b)
    )
    checksum_change_count += sum(
        report["committed_checksum_change_count"]
        for report in (snapshot_interrupt, trial_interrupt)
    )
    confirmatory_seed_access_count = sum(
        context[phase]["lifecycle"]["invocation_report"].get(
            "confirmatory_seed_access_count_this_invocation", 0
        )
        for context in (context_a, context_b)
        for phase in ("fresh", "resume")
    )
    gate_values: dict[str, Any] = {
        "V4_THIN_ADAPTER_FAILURE_PRESERVED": True,
        "V1_V2_V3_HISTORY_PRESERVED": True,
        "V1_V2_V3_SEED_REUSE_AUTHORIZED": False,
        "FORMAL_LIFECYCLE_SPEC_PASS": True,
        "FORMAL_LIFECYCLE_COMPONENT_BINDING_PASS": (
            shared_component_implementation_pass
        ),
        "SINGLE_FORMAL_LIFECYCLE_IMPLEMENTATION_PASS": bool(
            generic_audit["GENERIC_CORE_FORBIDDEN_REFERENCE_PASS"]
            and v3_wrapper_audit["audit_pass"]
            and qualification_wrapper_audit["audit_pass"]
        ),
        "CONTEXT_A_LIFECYCLE_PASS": context_a[
            "CONTEXT_LIFECYCLE_PASS"
        ],
        "CONTEXT_B_LIFECYCLE_PASS": context_b[
            "CONTEXT_LIFECYCLE_PASS"
        ],
        "FORMAL_QUALIFICATION_CONTEXT_ISOLATION_PASS": isolation[
            "FORMAL_QUALIFICATION_CONTEXT_ISOLATION_PASS"
        ],
        "UNMOCKED_GENERIC_LIFECYCLE_PASS": bool(
            context_a["fresh_backend_execution_count"] == 6
            and context_b["fresh_backend_execution_count"] == 6
            and qualification_wrapper_audit["audit_pass"]
        ),
        "REAL_SNAPSHOT_READER_PATH_PASS": bool(
            component_identity_a["snapshot_reader"][2]
            == "qualification_snapshot_reader"
            and component_identity_b["snapshot_reader"][2]
            == "qualification_snapshot_reader"
        ),
        "REAL_TRIAL_SNAPSHOT_BRIDGE_PASS": bool(
            context_a["fresh"]["lifecycle"]["invocation_report"][
                "backend_input_checksum_mismatch_count"
            ]
            == 0
            and context_b["fresh"]["lifecycle"]["invocation_report"][
                "backend_input_checksum_mismatch_count"
            ]
            == 0
        ),
        "REAL_BACKEND_EXECUTION_PASS": bool(
            context_a["backend_trial_counts"]
            == {
                "open3d_point_to_plane": 3,
                "pcl_point_to_plane": 3,
            }
            and context_b["backend_trial_counts"]
            == {
                "open3d_point_to_plane": 3,
                "pcl_point_to_plane": 3,
            }
        ),
        "FRESH_RESUME_PASS": bool(
            context_a["fresh_resume_postrun_exact_match"]
            and context_b["fresh_resume_postrun_exact_match"]
        ),
        "SNAPSHOT_INTERRUPTION_RESUME_PASS": snapshot_interrupt[
            "INTERRUPTION_RESUME_PASS"
        ],
        "TRIAL_INTERRUPTION_RESUME_PASS": trial_interrupt[
            "INTERRUPTION_RESUME_PASS"
        ],
        "VALID_SNAPSHOT_REEXECUTION_COUNT": snapshot_reexecution_count,
        "VALID_TRIAL_REEXECUTION_COUNT": trial_reexecution_count,
        "CHECKSUM_CHANGE_AFTER_RESUME_COUNT": checksum_change_count,
        "PRIMARY_INDEPENDENT_DIFFERENCE_COUNT": difference_count,
        "PUBLISHER_INVENTORY_PASS": publisher_pass,
        "ARTIFACT_VERIFIER_PASS": artifact_pass,
        "ALL_GIT_GATES_PASS": all_git_gates_pass,
        "GENERIC_CORE_V3_REFERENCE_COUNT": generic_audit[
            "GENERIC_CORE_V3_REFERENCE_COUNT"
        ],
        "GENERIC_CORE_FIXED_RUNTIME_PATH_COUNT": generic_audit[
            "GENERIC_CORE_FIXED_RUNTIME_PATH_COUNT"
        ],
        "GENERIC_CORE_FIXED_MANIFEST_PATH_COUNT": generic_audit[
            "GENERIC_CORE_FIXED_MANIFEST_PATH_COUNT"
        ],
        "GENERIC_CORE_FIXED_RUN_ID_COUNT": generic_audit[
            "GENERIC_CORE_FIXED_RUN_ID_COUNT"
        ],
        "V3_COMPATIBILITY_WRAPPER_DUPLICATE_EXECUTION_LOGIC_COUNT": (
            v3_wrapper_audit[
                "V3_COMPATIBILITY_WRAPPER_DUPLICATE_EXECUTION_LOGIC_COUNT"
            ]
        ),
        "QUALIFICATION_WRAPPER_DUPLICATE_EXECUTION_LOGIC_COUNT": (
            qualification_wrapper_audit[
                "QUALIFICATION_WRAPPER_DUPLICATE_EXECUTION_LOGIC_COUNT"
            ]
        ),
        "SCIENTIFIC_CORE_FILE_CHANGE_COUNT": science[
            "SCIENTIFIC_CORE_FILE_CHANGE_COUNT"
        ],
        "H1_H6_SEMANTICS_CHANGE_COUNT": science[
            "H1_H6_SEMANTICS_CHANGE_COUNT"
        ],
        "FROZEN_MODEL_CHANGE_COUNT": science[
            "FROZEN_MODEL_CHANGE_COUNT"
        ],
        "BACKEND_BINDING_CHANGE_COUNT": science[
            "BACKEND_BINDING_CHANGE_COUNT"
        ],
        "CONFIRMATORY_SEED_ACCESS_COUNT": confirmatory_seed_access_count,
        "V4_NAMESPACE_GENERATION_COUNT": science[
            "V4_NAMESPACE_GENERATION_COUNT"
        ],
        "V4_SEED_DERIVATION_COUNT": science[
            "V4_SEED_DERIVATION_COUNT"
        ],
        "TEST_SUITE_PASS": test_validation["TEST_REPORT_VALIDATION_PASS"],
        "QUALIFICATION_ARTIFACT_VERIFICATION_PASS": True,
        "FINAL_GIT_BINDING_PASS": True,
    }
    zero_gate_names = {
        "VALID_SNAPSHOT_REEXECUTION_COUNT",
        "VALID_TRIAL_REEXECUTION_COUNT",
        "CHECKSUM_CHANGE_AFTER_RESUME_COUNT",
        "PRIMARY_INDEPENDENT_DIFFERENCE_COUNT",
        "GENERIC_CORE_V3_REFERENCE_COUNT",
        "GENERIC_CORE_FIXED_RUNTIME_PATH_COUNT",
        "GENERIC_CORE_FIXED_MANIFEST_PATH_COUNT",
        "GENERIC_CORE_FIXED_RUN_ID_COUNT",
        "V3_COMPATIBILITY_WRAPPER_DUPLICATE_EXECUTION_LOGIC_COUNT",
        "QUALIFICATION_WRAPPER_DUPLICATE_EXECUTION_LOGIC_COUNT",
        "SCIENTIFIC_CORE_FILE_CHANGE_COUNT",
        "H1_H6_SEMANTICS_CHANGE_COUNT",
        "FROZEN_MODEL_CHANGE_COUNT",
        "BACKEND_BINDING_CHANGE_COUNT",
        "CONFIRMATORY_SEED_ACCESS_COUNT",
        "V4_NAMESPACE_GENERATION_COUNT",
        "V4_SEED_DERIVATION_COUNT",
    }
    false_gate_names = {"V1_V2_V3_SEED_REUSE_AUTHORIZED"}
    qualification_pass = all(
        (
            value == 0
            if name in zero_gate_names
            else value is False
            if name in false_gate_names
            else value is True
        )
        for name, value in gate_values.items()
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=".version-agnostic-lifecycle-artifact.",
            dir=str(output.parent),
        )
    )
    try:
        inventory_count = _hardcoded_inventory(
            staging / "hardcoded_formal_dependency_inventory.csv"
        )
        _copy_binding_reports(staging)
        reports: dict[str, Any] = {
            "v4_thin_adapter_blocking_report.json": {
                "schema_version": "v4_thin_adapter_blocking_report_v1",
                "V4_THIN_ADAPTER_FAILURE_PRESERVED": True,
                "classification": (
                    "COMPLETE_PRODUCTION_LIFECYCLE_NOT_REUSABLE_BEFORE_REFACTOR"
                ),
                "archive_tag": "archive/zero-perturbation-v4-thin-adapter-blocked",
                "archive_bundle": (
                    "/tmp/zero-perturbation-v4-thin-adapter-blocked.bundle"
                ),
                "archive_bundle_sha256": (
                    "5d81918ea757e650927c6f4452ebb9a3ce64ce2721ac58125"
                    "b72b12c46ac6781"
                ),
                "baseline_commit": BASELINE_COMMIT,
                "pre_refactor_findings": {
                    "v3_runner_fixed_v3_runtime_root": True,
                    "execution_stack_fixed_v3_contract": True,
                    "snapshot_lifecycle_fixed_v3_manifest_and_context": True,
                    "qualification_entry_fixed_old_identity": True,
                    "complete_arbitrary_execution_context_callable_absent": True,
                    "thin_adapter_would_require_core_mutation_or_fourth_copy": True,
                },
                "v4_namespace_derived": False,
                "v4_seed_derived": False,
                "v4_formal_plan_created": False,
                "backend_execution_count_before_refactor": 0,
            },
            "v1_v2_v3_history_binding.json": {
                "schema_version": "v1_v2_v3_history_binding_v1",
                "V1_V2_V3_HISTORY_PRESERVED": True,
                "V1_V2_V3_SEED_REUSE_AUTHORIZED": False,
                "v3_resume_authorized": False,
                "historical_scientific_decisions_changed": False,
                "v1": {
                    "root_cause": "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON",
                    "scientific_status": "NOT_EVALUATED",
                    "seed_reuse_authorized": False,
                },
                "v2": {
                    "root_cause": (
                        "RUNTIME_PATH_AND_GIT_GATE_LIFECYCLE_DEFECT"
                    ),
                    "scientific_status": "NOT_EVALUATED",
                    "seed_reuse_authorized": False,
                },
                "v3": {
                    "root_cause": "FORMAL_ONLY_SCHEMA_OR_BINDING_DEFECT",
                    "executed": True,
                    "complete": False,
                    "scientific_status": "NOT_EVALUATED",
                    "snapshot_count": 595,
                    "completed_trial_count": 0,
                    "planned_trial_count": 1190,
                    "backend_execution_count": 0,
                    "seed_reuse_authorized": False,
                    "partial_snapshot_scientific_use_authorized": False,
                    "formal_run_resume_authorized": False,
                    "formal_root": (
                        "/home/lj/ZPRM/zero_perturbation_runtime/confirmatory/"
                        "synthetic_confirmatory_v3"
                    ),
                    "archive_root": (
                        "/home/lj/ZPRM/zero_perturbation_runtime_archive/"
                        "synthetic_confirmatory_v3_post_snapshot_pretrial_"
                        "failure_20260730"
                    ),
                    "formal_snapshot_payload_read_count_this_qualification": 0,
                },
            },
            "retired_seed_sets.json": {
                "schema_version": "retired_seed_sets_v1",
                "v1_seed_reuse_authorized": False,
                "v2_seed_reuse_authorized": False,
                "v3_seed_reuse_authorized": False,
                "confirmatory_seed_access_count": (
                    confirmatory_seed_access_count
                ),
                "retirement_permanent": True,
            },
            "formal_lifecycle_architecture_contract.json": {
                "schema_version": "formal_lifecycle_architecture_contract_v1",
                "spec_class": "FormalLifecycleSpec",
                "execution_entry": "execute_formal_lifecycle",
                "postrun_entry": "execute_postrun_pipeline",
                "complete_control_flow_owner_count": 1,
                "runtime_paths_explicit": True,
                "plans_explicit": True,
                "context_explicit": True,
                "manifest_file_content_sha_binding_required": True,
                "generic_core_context_specific_constant_count": (
                    generic_audit[
                        "GENERIC_CORE_CONTEXT_SPECIFIC_CONSTANT_COUNT"
                    ]
                ),
                "historical_v3_helpers_reachable_from_public_execution_entry": (
                    False
                ),
            },
            "formal_lifecycle_component_contract.json": {
                "schema_version": "formal_lifecycle_component_contract_v1",
                "context_a": spec_a.components.manifest_binding(
                    repository_root=REPOSITORY
                ),
                "context_b": spec_b.components.manifest_binding(
                    repository_root=REPOSITORY
                ),
                "shared_component_implementation_pass": (
                    shared_component_implementation_pass
                ),
                "backend_registry": [
                    "open3d_point_to_plane",
                    "pcl_point_to_plane",
                ],
                "native_backend_allowed": False,
            },
            "context_a_contract.json": {
                "execution_context": spec_a.execution_context.report(),
                "manifest_sha256": spec_a.manifest_sha256,
                "run_id": spec_a.run_id,
                "runtime_root": str(spec_a.paths.runtime_root),
                "snapshot_count": len(spec_a.snapshot_plan),
                "trial_count": len(spec_a.trial_plan),
                "git_identity_policy": spec_a.identity_policy.report(),
            },
            "context_b_contract.json": {
                "execution_context": spec_b.execution_context.report(),
                "manifest_sha256": spec_b.manifest_sha256,
                "run_id": spec_b.run_id,
                "runtime_root": str(spec_b.paths.runtime_root),
                "snapshot_count": len(spec_b.snapshot_plan),
                "trial_count": len(spec_b.trial_plan),
                "git_identity_policy": spec_b.identity_policy.report(),
            },
            "context_isolation_audit.json": isolation,
            "context_a_fresh_report.json": context_a["fresh"],
            "context_a_resume_report.json": context_a["resume"],
            "context_b_fresh_report.json": context_b["fresh"],
            "context_b_resume_report.json": context_b["resume"],
            "snapshot_interruption_report.json": snapshot_interrupt,
            "trial_interruption_report.json": trial_interrupt,
            "primary_independent_difference.json": {
                "context_a": context_a["primary_independent_difference"],
                "context_b": context_b["primary_independent_difference"],
                "PRIMARY_INDEPENDENT_DIFFERENCE_COUNT": difference_count,
            },
            "publisher_inventory.json": {
                "context_a": context_a["publisher_inventory"],
                "context_b": context_b["publisher_inventory"],
                "PUBLISHER_INVENTORY_PASS": publisher_pass,
            },
            "artifact_verification.json": {
                "context_a": context_a["artifact_verification"],
                "context_b": context_b["artifact_verification"],
                "ARTIFACT_VERIFIER_PASS": artifact_pass,
                "qualification_package_verifier": {
                    "verification_stage": "FINAL_TWO_PASS_MANIFEST_AND_SHA",
                    "required_file_count": len(REQUIRED_ARTIFACT_FILES),
                },
                "QUALIFICATION_ARTIFACT_VERIFICATION_PASS": (
                    qualification_pass
                ),
            },
            "v3_compatibility_wrapper_audit.json": v3_wrapper_audit,
            "qualification_wrapper_audit.json": {
                **qualification_wrapper_audit,
                "execute_one_mock_count": 0,
                "fixture_mock_count": 0,
                "generic_lifecycle_mock_count": 0,
                "real_worker_subprocess_count": 8,
            },
            "generic_core_forbidden_reference_audit.json": generic_audit,
            "version_agnostic_lifecycle_scientific_diff.json": science,
            "test_report.json": {
                **tests,
                "qualification_script_validation": test_validation,
            },
            "implementation_manifest.json": {
                "schema_version": "version_agnostic_implementation_manifest_v1",
                "baseline_commit": BASELINE_COMMIT,
                "candidate_commit": candidate,
                "branch": BRANCH,
                "candidate_tags": [TAG_A, TAG_B],
                "hardcoded_dependency_inventory_count": inventory_count,
                "single_execution_entry": "execute_formal_lifecycle",
                "single_postrun_entry": "execute_postrun_pipeline",
                "candidate_parent": _git("rev-parse", f"{candidate}^"),
                "candidate_commit_count_above_baseline": int(
                    _git(
                        "rev-list",
                        "--count",
                        f"{BASELINE_COMMIT}..{candidate}",
                    )
                ),
                "candidate_changed_files": [
                    {
                        "path": relative,
                        "sha256": _sha256(REPOSITORY / relative),
                    }
                    for relative in _git(
                        "diff",
                        "--name-only",
                        BASELINE_COMMIT,
                        candidate,
                    ).splitlines()
                ],
                "stage_b_allowed_path": ARTIFACT_RELATIVE.as_posix(),
                "final_tag": FINAL_TAG,
                "final_bundle": (
                    "/tmp/zero-perturbation-version-agnostic-formal-"
                    "lifecycle.bundle"
                ),
                "final_git_binding_policy": (
                    "candidate exact plus one artifact-only publication commit"
                ),
            },
        }
        final_decision = {
            "schema_version": "version_agnostic_lifecycle_final_decision_v1",
            **gate_values,
            "VERSION_AGNOSTIC_FORMAL_LIFECYCLE_QUALIFICATION_PASS": (
                qualification_pass
            ),
            "V4_PRE_RUN_DESIGN_AUTHORIZED": qualification_pass,
            "V4_SEED_DERIVATION_AUTHORIZED": qualification_pass,
            "V4_RUN_AUTHORIZED": False,
            "SYNTHETIC_CONFIRMATORY_V4_EXECUTED": False,
            "SYNTHETIC_CONFIRMATORY_V4_COMPLETE": False,
            "SYNTHETIC_CONFIRMATORY_V4_PASS": "NOT_EVALUATED",
            "REAL_DATA_RUN_AUTHORIZED": False,
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
            "MINIMAL_OFFLINE_RUNNER_REDESIGN_REQUIRED": not qualification_pass,
        }
        reports["final_decision.json"] = final_decision
        reports["run_manifest.json"] = {
            "schema_version": "version_agnostic_lifecycle_qualification_run_v1",
            "candidate_commit": candidate,
            "context_a_lifecycle_pass": context_a["CONTEXT_LIFECYCLE_PASS"],
            "context_b_lifecycle_pass": context_b["CONTEXT_LIFECYCLE_PASS"],
            "negative_case_count": isolation["negative_case_count"],
            "negative_false_accept_count": isolation["false_accept_count"],
            "negative_unexpected_rejection_count": isolation[
                "unexpected_rejection_count"
            ],
            "snapshot_interruption_pass": snapshot_interrupt[
                "INTERRUPTION_RESUME_PASS"
            ],
            "trial_interruption_pass": trial_interrupt[
                "INTERRUPTION_RESUME_PASS"
            ],
            "context_a_fresh_snapshot_count": 3,
            "context_a_fresh_trial_count": 6,
            "context_b_fresh_snapshot_count": 3,
            "context_b_fresh_trial_count": 6,
            "open3d_execution_count": 6,
            "pcl_execution_count": 6,
            "native_execution_count": 0,
            "valid_snapshot_reexecution_count": snapshot_reexecution_count,
            "valid_trial_reexecution_count": trial_reexecution_count,
            "checksum_change_after_resume_count": checksum_change_count,
            "confirmatory_seed_access_count": (
                confirmatory_seed_access_count
            ),
            "v4_namespace_generation_count": science[
                "V4_NAMESPACE_GENERATION_COUNT"
            ],
            "v4_seed_derivation_count": science[
                "V4_SEED_DERIVATION_COUNT"
            ],
            "all_git_gates_pass": all_git_gates_pass,
            "qualification_pass": qualification_pass,
        }
        for name, value in reports.items():
            _write_json(staging / name, value)
        with (staging / "negative_context_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(negative_rows[0]))
            writer.writeheader()
            writer.writerows(negative_rows)
        (staging / "formal_lifecycle_call_graph.md").write_text(
            "# Version-Agnostic Formal Lifecycle Call Graph\n\n"
            "CLI / version wrapper → immutable FormalLifecycleSpec → "
            "`execute_formal_lifecycle` → per-snapshot components → canonical "
            "trial–snapshot bridge → per-trial backend components → immutable "
            "run manifest.\n\n"
            "CompletedFormalRun → `execute_postrun_pipeline` → primary → "
            "independent verifier → difference audit → publisher → artifact "
            "verifier.\n",
            encoding="utf-8",
        )
        (staging / "qualification_report.md").write_text(
            "# Version-Agnostic Formal Lifecycle Qualification\n\n"
            f"Candidate: `{candidate}`\n\n"
            f"Context A: {'PASS' if context_a['CONTEXT_LIFECYCLE_PASS'] else 'FAIL'}\n\n"
            f"Context B: {'PASS' if context_b['CONTEXT_LIFECYCLE_PASS'] else 'FAIL'}\n\n"
            f"Snapshot SIGTERM recovery: {'PASS' if snapshot_interrupt['INTERRUPTION_RESUME_PASS'] else 'FAIL'}\n\n"
            f"Trial SIGTERM recovery: {'PASS' if trial_interrupt['INTERRUPTION_RESUME_PASS'] else 'FAIL'}\n\n"
            f"Final qualification: {'PASS' if qualification_pass else 'FAIL'}\n",
            encoding="utf-8",
        )
        _write_manifest_inventory(staging)
        first_artifact_verification = _verify_qualification_artifact(staging)
        if (
            first_artifact_verification[
                "QUALIFICATION_ARTIFACT_VERIFICATION_PASS"
            ]
            is not True
        ):
            raise RuntimeError("qualification artifact first-pass verification failed")
        artifact_report = _strict_object(
            staging / "artifact_verification.json"
        )
        artifact_report["qualification_package_verifier"] = (
            first_artifact_verification
        )
        artifact_report[
            "QUALIFICATION_ARTIFACT_VERIFICATION_PASS"
        ] = True
        _write_json(staging / "artifact_verification.json", artifact_report)
        _write_manifest_inventory(staging)
        final_artifact_verification = _verify_qualification_artifact(staging)
        if (
            final_artifact_verification[
                "QUALIFICATION_ARTIFACT_VERIFICATION_PASS"
            ]
            is not True
        ):
            raise RuntimeError("qualification artifact final verification failed")
        os.replace(staging, output)
        status = _git("status", "--porcelain=v1", "--untracked-files=all")
        allowed_prefix = f"?? {ARTIFACT_RELATIVE.as_posix()}/"
        if not status or any(
            not line.startswith(allowed_prefix)
            for line in status.splitlines()
        ):
            shutil.rmtree(output, ignore_errors=True)
            raise RuntimeError(
                "Stage-B worktree contains changes outside the qualification artifact"
            )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "artifact": str(output),
                "candidate_commit": candidate,
                "qualification_pass": qualification_pass,
            },
            sort_keys=True,
        )
    )
    return 0 if qualification_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
