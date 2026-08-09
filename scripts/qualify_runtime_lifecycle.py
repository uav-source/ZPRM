#!/usr/bin/env python3
"""Qualify the external runtime lifecycle with real SIGTERM and real writes.

The script is intentionally fixture-only.  It refuses every path except the
single requested qualification root and never reads a Confirmatory plan or
seed schedule.  Its compact artifact remains external until a separate,
explicit publication-import step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


QUALIFICATION_ROOT = Path(
    "/home/lj/zero_perturbation_runtime/qualification/runtime_lifecycle_v1"
)
FAILURE_ARCHIVE = Path(
    "/home/lj/zero_perturbation_runtime_archive/"
    "synthetic_confirmatory_v2_runtime_lifecycle_failure_20260730"
)
FAILURE_TAR = Path(
    "/home/lj/zero_perturbation_runtime_archive/"
    "synthetic_confirmatory_v2_runtime_lifecycle_failure_20260730.tar.gz"
)
FAILURE_BUNDLE = Path(
    "/tmp/zero-perturbation-synthetic-confirmatory-v2-runtime-lifecycle-fail.bundle"
)
PRE_RUN_BUNDLE = Path(
    "/tmp/zero-perturbation-synthetic-confirmatory-v2-pre-run.bundle"
)
PCL_V3_SOURCE = Path("/tmp/synthetic_confirmatory_v2_pcl_v3_requalification")
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"
EXPECTED_FAILURE_TAR_SHA256 = (
    "5979ed924d21afc43bbbcb12267fda471c67ed642ebafd71167883ed0ee34176"
)
EXPECTED_FAILURE_BUNDLE_SHA256 = (
    "071fe5d7daa6b42f1bae41f21f388735b4059f0ab49a7907f327f2f2e005bc25"
)
EXPECTED_PRE_RUN_BUNDLE_SHA256 = (
    "dd5ee119bc51fe8bbdf4f2a2dae2d7050d4dbca3c3fc88d23b64d9a4b0d5b249"
)
EXPECTED_PCL_V3_FILE_COUNT = 18
EXPECTED_PCL_V3_SIZE_BYTES = 8855746
EXPECTED_PCL_V3_TREE_SHA256 = (
    "43c5fecebdeab397f824fbd96cce496d85d21765590bb5f0c2a3a890a92fa243"
)
EXPECTED_PCL_V3_CLI_SHA256 = (
    "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff"
)
PCL_V3_DIRECTORIES = ("bin", "src", "tests", "tools")
SOURCE_ACCESS_EVENT_SCHEMA = "runtime_lifecycle_source_access_event_v1"
RAW_MANIFEST_SCHEMA = "runtime_lifecycle_fixture_raw_result_manifest_v1"
FAILURE_COMMIT = "9fdb90638e6c5785bf14d49134c56af60f9a3393"
FAILURE_TAG = "archive/zero-perturbation-synthetic-confirmatory-v2-runtime-lifecycle-fail"
PRE_RUN_TAG = "archive/zero-perturbation-synthetic-confirmatory-v2-pre-run-pass"
WORKERS = 2
INTERRUPT_DELAY_SECONDS = 0.75


def _assert_environment() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("qualification requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError("qualification requires the frozen MAMBA_ROOT_PREFIX")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_text(command: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout + completed.stderr


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    from phase_a_harness.runtime_lifecycle_io import atomic_create_canonical_json

    atomic_create_canonical_json(path, dict(value))


def _source_access_log_path(log_root: Path, run_id: str, invocation_id: str) -> Path:
    return log_root / f"{run_id}_{invocation_id}.source_access.ndjson"


def _strict_json_object(payload: str, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token in {label}: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid strict JSON in {label}") from error
    if type(value) is not dict:
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _source_access_log_report(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"worker source-access log is missing or unsafe: {path}")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise RuntimeError("worker source-access log has a truncated final line")
    rows: list[dict[str, Any]] = []
    for sequence, raw in enumerate(payload.splitlines(), start=1):
        try:
            line = raw.decode("utf-8")
        except UnicodeError as error:
            raise RuntimeError("worker source-access log is not UTF-8") from error
        row = _strict_json_object(line, label=f"source-access event {sequence}")
        if set(row) != {
            "event",
            "event_schema",
            "path",
            "pid",
            "sequence",
            "source_repository",
        }:
            raise RuntimeError("worker source-access event schema mismatch")
        if (
            row["event"] != "open"
            or row["event_schema"] != SOURCE_ACCESS_EVENT_SCHEMA
            or type(row["path"]) is not str
            or type(row["pid"]) is not int
            or type(row["pid"]) is bool
            or row["pid"] <= 0
            or type(row["sequence"]) is not int
            or type(row["sequence"]) is bool
            or row["sequence"] != sequence
            or row["source_repository"] != str(SOURCE_REPOSITORY)
        ):
            raise RuntimeError("worker source-access event value mismatch")
        candidate = Path(row["path"])
        if not candidate.is_absolute() or not (
            candidate == SOURCE_REPOSITORY
            or SOURCE_REPOSITORY in candidate.parents
        ):
            raise RuntimeError("worker source-access event path is outside source repository")
        rows.append(row)
    return {
        "source_access_log_path": str(path),
        "source_access_log_size_bytes": len(payload),
        "source_access_log_sha256": hashlib.sha256(payload).hexdigest(),
        "source_repository_open_event_count": len(rows),
        "source_repository_open_event_paths": [row["path"] for row in rows],
        "SOURCE_ACCESS_LOG_PASS": len(rows) == 0,
    }


def _validate_completed_worker_source_evidence(
    result: Mapping[str, Any], *, path: Path
) -> dict[str, Any]:
    evidence = _source_access_log_report(path)
    count_names = (
        "source_repository_early_file_read_count",
        "source_repository_runtime_file_read_count",
        "source_repository_runtime_import_count",
    )
    if result.get("source_access_log_path") != str(path):
        raise RuntimeError("worker stdout source-access path mismatch")
    if any(type(result.get(name)) is not int for name in count_names):
        raise RuntimeError("worker stdout source-access counts are not exact integers")
    if any(result[name] != 0 for name in count_names):
        raise PermissionError("worker reported source repository access")
    for name in (
        "source_repository_early_file_read_paths",
        "source_repository_runtime_file_read_paths",
        "source_repository_runtime_import_paths",
    ):
        if result.get(name) != []:
            raise PermissionError("worker reported source repository access paths")
    if evidence["SOURCE_ACCESS_LOG_PASS"] is not True:
        raise PermissionError("worker source-access event log is nonempty")
    return evidence


def _worker_command(
    *,
    repository: Path,
    qualification_root: Path,
    run_id: str,
    invocation_id: str,
    expected_commit: str,
    expected_branch: str,
    expected_tag: str,
    source_access_log: Path,
    resume: bool,
    delay: float,
) -> list[str]:
    command = [
        sys.executable,
        str(repository / "scripts/run_runtime_lifecycle_fixture.py"),
        "--runtime-root",
        str(qualification_root),
        "--run-id",
        run_id,
        "--invocation-id",
        invocation_id,
        "--workers",
        str(WORKERS),
        "--expected-commit",
        expected_commit,
        "--expected-branch",
        expected_branch,
        "--expected-tag",
        expected_tag,
        "--source-access-log",
        str(source_access_log),
        "--qualification-delay-seconds",
        str(delay),
    ]
    if resume:
        command.append("--resume")
    return command


def _parse_worker_stdout(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"worker did not emit one JSON object: {path}") from error
    if type(value) is not dict:
        raise RuntimeError("worker output is not a JSON object")
    return value


def _run_worker(
    *,
    repository: Path,
    qualification_root: Path,
    log_root: Path,
    run_id: str,
    invocation_id: str,
    expected_commit: str,
    expected_branch: str,
    expected_tag: str,
    resume: bool,
    delay: float,
) -> dict[str, Any]:
    stdout_path = log_root / f"{run_id}_{invocation_id}.stdout.json"
    stderr_path = log_root / f"{run_id}_{invocation_id}.stderr.txt"
    source_access_log = _source_access_log_path(log_root, run_id, invocation_id)
    if source_access_log.exists() or source_access_log.is_symlink():
        raise FileExistsError(f"source-access log already exists: {source_access_log}")
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            _worker_command(
                repository=repository,
                qualification_root=qualification_root,
                run_id=run_id,
                invocation_id=invocation_id,
                expected_commit=expected_commit,
                expected_branch=expected_branch,
                expected_tag=expected_tag,
                source_access_log=source_access_log,
                resume=resume,
                delay=delay,
            ),
            cwd=repository,
            stdout=stdout,
            stderr=stderr,
            check=False,
            env=dict(os.environ),
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed ({completed.returncode}): {stderr_path.read_text(errors='replace')}"
        )
    result = _parse_worker_stdout(stdout_path)
    result["supervisor_source_access_evidence"] = (
        _validate_completed_worker_source_evidence(result, path=source_access_log)
    )
    return result


def _wait_and_sigterm(
    *,
    repository: Path,
    qualification_root: Path,
    log_root: Path,
    run_id: str,
    invocation_id: str,
    expected_commit: str,
    expected_branch: str,
    expected_tag: str,
    progress: Callable[[], int],
    total: int,
    stage: str,
) -> dict[str, Any]:
    stdout_path = log_root / f"{run_id}_{invocation_id}.stdout.partial"
    stderr_path = log_root / f"{run_id}_{invocation_id}.stderr.txt"
    source_access_log = _source_access_log_path(log_root, run_id, invocation_id)
    if source_access_log.exists() or source_access_log.is_symlink():
        raise FileExistsError(f"source-access log already exists: {source_access_log}")
    command = _worker_command(
        repository=repository,
        qualification_root=qualification_root,
        run_id=run_id,
        invocation_id=invocation_id,
        expected_commit=expected_commit,
        expected_branch=expected_branch,
        expected_tag=expected_tag,
        source_access_log=source_access_log,
        resume=False,
        delay=INTERRUPT_DELAY_SECONDS,
    )
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=repository,
            stdout=stdout,
            stderr=stderr,
            env=dict(os.environ),
        )
        deadline = time.monotonic() + 180.0
        observed = 0
        while time.monotonic() < deadline:
            returncode = process.poll()
            observed = progress()
            if 0 < observed < total:
                process.send_signal(signal.SIGTERM)
                returncode = process.wait(timeout=30.0)
                break
            if returncode is not None:
                raise RuntimeError(
                    f"{stage} worker exited before interruption point: {returncode}; "
                    f"stderr={stderr_path.read_text(errors='replace')}"
                )
            time.sleep(0.02)
        else:
            process.terminate()
            process.wait(timeout=30.0)
            raise TimeoutError(f"timed out waiting for {stage} interruption point")
    if returncode != -signal.SIGTERM:
        raise RuntimeError(f"{stage} worker did not terminate from SIGTERM: {returncode}")
    source_access_evidence = _source_access_log_report(source_access_log)
    if source_access_evidence["SOURCE_ACCESS_LOG_PASS"] is not True:
        raise PermissionError("interrupted worker accessed the source repository")
    return {
        "stage": stage,
        "command": command,
        "pid": process.pid,
        "signal": "SIGTERM",
        "signal_number": int(signal.SIGTERM),
        "returncode": returncode,
        "completed_at_signal": observed,
        "planned_total": total,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "supervisor_source_access_evidence": source_access_evidence,
    }


def _file_inventory(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _temporary_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if not any(
            component.endswith(".tmp")
            or ".tmp-" in component
            or ".staging-" in component
            for component in relative.parts
        ):
            continue
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            kind = "file"
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
        else:
            kind = "special"
        rows.append(
            {
                "path": relative.as_posix(),
                "type": kind,
                "size": metadata.st_size,
                "sha256": _sha256(path) if kind == "file" else None,
            }
        )
    return rows


def _read_committed_raw_manifest(path: Path, *, expected_run_id: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"committed raw manifest is missing or unsafe: {path}")
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError("cannot read committed raw manifest") from error
    value = _strict_json_object(payload, label="committed raw manifest")
    if (
        set(value) != {"schema_version", "run_id", "run_contract_sha256", "results"}
        or value.get("schema_version") != RAW_MANIFEST_SCHEMA
        or value.get("run_id") != expected_run_id
        or type(value.get("run_contract_sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value["run_contract_sha256"]) is None
        or type(value.get("results")) is not dict
    ):
        raise RuntimeError("committed raw manifest contract mismatch")
    referenced: list[str] = []
    for trial_id, entry in value["results"].items():
        if (
            type(trial_id) is not str
            or type(entry) is not dict
            or set(entry) != {"path", "planned_trial_id", "sha256"}
            or entry.get("planned_trial_id") != trial_id
            or type(entry.get("path")) is not str
            or PurePosixPath(entry["path"]).name != entry["path"]
            or type(entry.get("sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            raise RuntimeError("committed raw manifest result entry mismatch")
        referenced.append(entry["path"])
    if len(referenced) != len(set(referenced)):
        raise RuntimeError("committed raw manifest has duplicate result paths")
    return value


def _committed_trial_progress(path: Path, *, expected_run_id: str) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    manifest = _read_committed_raw_manifest(path, expected_run_id=expected_run_id)
    return len(manifest["results"])


def _snapshot_interruption_state(layout: Any, fixtures: Sequence[Any]) -> dict[str, Any]:
    inventory = _file_inventory(layout.snapshot_cache)
    token_to_id = {
        item.snapshot_id.rsplit("/", 1)[-1]: item.snapshot_id for item in fixtures
    }
    top_entries: list[str] = []
    if layout.snapshot_cache.exists():
        top_entries = sorted(path.name for path in layout.snapshot_cache.iterdir())
    committed_tokens = [
        token
        for token in top_entries
        if token in token_to_id and (layout.snapshot_cache / token).is_dir()
    ]
    committed_ids = [token_to_id[token] for token in committed_tokens]
    counts = Counter(committed_ids)
    orphan_entries = sorted(set(top_entries) - set(committed_tokens))
    temporary = _temporary_inventory(layout.run_root)
    return {
        "committed_snapshot_ids": committed_ids,
        "committed_snapshot_count": len(committed_ids),
        "snapshot_file_inventory": inventory,
        "snapshot_file_count": len(inventory),
        "orphan_snapshot_entries": orphan_entries,
        "orphan_snapshot_count": len(orphan_entries),
        "duplicate_snapshot_count": sum(max(count - 1, 0) for count in counts.values()),
        "temporary_inventory": temporary,
        "temporary_entry_count": len(temporary),
    }


def _trial_interruption_state(layout: Any) -> dict[str, Any]:
    manifest_path = layout.raw_results / "raw_result_manifest.json"
    manifest = _read_committed_raw_manifest(
        manifest_path, expected_run_id=layout.run_id
    )
    results_root = layout.raw_results / "results"
    actual_files: list[str] = []
    unsafe_entries: list[str] = []
    if results_root.exists():
        for path in sorted(results_root.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file():
                unsafe_entries.append(path.name)
            else:
                actual_files.append(path.name)
    referenced = [entry["path"] for entry in manifest["results"].values()]
    referenced_counts = Counter(referenced)
    orphans = sorted(set(actual_files) - set(referenced))
    missing = sorted(set(referenced) - set(actual_files))
    temporary = _temporary_inventory(layout.run_root)
    return {
        "committed_trial_ids": sorted(manifest["results"]),
        "committed_trial_count": len(manifest["results"]),
        "raw_manifest": manifest,
        "result_file_inventory": _file_inventory(results_root),
        "result_file_count": len(actual_files),
        "orphan_result_files": orphans,
        "orphan_result_count": len(orphans),
        "missing_referenced_result_files": missing,
        "missing_referenced_result_count": len(missing),
        "unsafe_result_entries": unsafe_entries,
        "unsafe_result_entry_count": len(unsafe_entries),
        "duplicate_trial_count": sum(
            max(count - 1, 0) for count in referenced_counts.values()
        ),
        "temporary_inventory": temporary,
        "temporary_entry_count": len(temporary),
    }


def _planned_execution_records(
    *,
    planned_ids: Sequence[str],
    initial_executed_ids: Sequence[str],
    resume_executed_ids: Sequence[str],
) -> list[dict[str, Any]]:
    planned = list(planned_ids)
    if len(planned) != len(set(planned)):
        raise RuntimeError("planned execution IDs are not unique")
    planned_set = set(planned)
    initial_counts = Counter(initial_executed_ids)
    resume_counts = Counter(resume_executed_ids)
    extra = sorted((set(initial_counts) | set(resume_counts)) - planned_set)
    if extra:
        raise RuntimeError(f"worker execution evidence contains unknown IDs: {extra}")
    return [
        {
            "planned_id": planned_id,
            "initial_execution_count": initial_counts[planned_id],
            "resume_execution_count": resume_counts[planned_id],
            "final_execution_count": (
                initial_counts[planned_id] + resume_counts[planned_id]
            ),
        }
        for planned_id in planned
    ]


def _execution_duplicate_count(records: Sequence[Mapping[str, Any]]) -> int:
    return sum(max(int(row["final_execution_count"]) - 1, 0) for row in records)


def _valid_reexecution_count(records: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        min(int(row["initial_execution_count"]), int(row["resume_execution_count"]))
        for row in records
    )


def _source_trace_report(
    trace_path: Path, *, source_repository: Path, command: Sequence[str]
) -> dict[str, Any]:
    if trace_path.is_symlink() or not trace_path.is_file():
        raise RuntimeError(f"strace evidence is missing or unsafe: {trace_path}")
    payload = trace_path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise RuntimeError("strace evidence is not UTF-8") from error
    source = str(source_repository.resolve())
    matching_lines = [
        line
        for line in text.splitlines()
        if source in line
        and re.search(r"\b(?:open|openat|openat2)\(", line) is not None
    ]
    return {
        "trace_command": list(command),
        "trace_path": str(trace_path),
        "trace_sha256": hashlib.sha256(payload).hexdigest(),
        "trace_size_bytes": len(payload),
        "source_repository": source,
        "source_repository_open_read_count": len(matching_lines),
        "source_repository_open_read_lines": matching_lines,
        "SOURCE_REPOSITORY_SUBPROCESS_ACCESS_PASS": len(matching_lines) == 0,
    }


def _run_source_traced(
    command: Sequence[str],
    *,
    cwd: Path,
    trace_path: Path,
    source_repository: Path = SOURCE_REPOSITORY,
    **run_arguments: Any,
) -> tuple[subprocess.CompletedProcess[Any], dict[str, Any]]:
    if trace_path.exists() or trace_path.is_symlink():
        raise FileExistsError(f"strace evidence already exists: {trace_path}")
    strace = shutil.which("strace")
    if strace is None:
        raise FileNotFoundError("strace is required for subprocess source-access evidence")
    traced_command = [
        strace,
        "-f",
        "-qq",
        "-yy",
        "-s",
        "4096",
        "-e",
        "trace=open,openat",
        "-o",
        str(trace_path),
        "--",
        *command,
    ]
    completed = subprocess.run(traced_command, cwd=cwd, **run_arguments)
    evidence = _source_trace_report(
        trace_path, source_repository=source_repository, command=traced_command
    )
    if evidence["SOURCE_REPOSITORY_SUBPROCESS_ACCESS_PASS"] is not True:
        raise PermissionError("subprocess opened the source repository")
    return completed, evidence


def _junit(path: Path, label: str) -> dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        suites = list(root.findall(".//testsuite"))
    counts = {
        name: sum(int(float(suite.attrib.get(name, "0"))) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    return {
        "label": label,
        **counts,
        "passed": passed,
        "unexpected_skip_count": counts["skipped"],
        "junit_path": str(path),
        "junit_sha256": _sha256(path),
        "pass": bool(
            counts["tests"] > 0
            and counts["failures"] == 0
            and counts["errors"] == 0
            and counts["skipped"] == 0
        ),
    }


def _pytest_suite(
    repository: Path,
    qualification_root: Path,
    name: str,
    selections: Sequence[str],
) -> dict[str, Any]:
    junit = qualification_root / "test_logs" / f"{name}.xml"
    output = qualification_root / "test_logs" / f"{name}.txt"
    trace_path = qualification_root / "supervisor_logs" / f"pytest_{name}.strace"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *selections,
        "-q",
        f"--junitxml={junit}",
    ]
    with output.open("xb") as stream:
        completed, source_access = _run_source_traced(
            command,
            cwd=repository,
            trace_path=trace_path,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            env=dict(os.environ),
        )
    report = _junit(junit, name)
    report.update(
        {
            "command": command,
            "exit_code": completed.returncode,
            "output_path": str(output),
            "output_sha256": _sha256(output),
            "source_access_evidence": source_access,
        }
    )
    report["pass"] = bool(
        report["pass"]
        and completed.returncode == 0
        and source_access["SOURCE_REPOSITORY_SUBPROCESS_ACCESS_PASS"] is True
    )
    if not report["pass"]:
        raise RuntimeError(f"pytest suite failed: {name}: {output.read_text(errors='replace')}")
    return report


def _pcl_v3_inventory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("external PCL-v3 requalification package is missing or unsafe")
    rows: list[dict[str, Any]] = []
    for name in PCL_V3_DIRECTORIES:
        directory = root / name
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"PCL-v3 package directory is missing or unsafe: {name}")
        for path in sorted(
            directory.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
        ):
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(
                    f"PCL-v3 package contains symlink or special entry: "
                    f"{path.relative_to(root).as_posix()}"
                )
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": metadata.st_size,
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "sha256": _sha256(path),
                }
            )
    rows.sort(key=lambda row: row["path"])
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    report = {
        "directories": list(PCL_V3_DIRECTORIES),
        "directory_count": len(PCL_V3_DIRECTORIES),
        "file_inventory": rows,
        "file_count": len(rows),
        "size_bytes": sum(row["size"] for row in rows),
        "canonical_rows_json_sha256": hashlib.sha256(payload).hexdigest(),
        "pcl_point_to_plane_cli_sha256": _sha256(
            root / "bin/pcl_point_to_plane_cli"
        ),
    }
    report["PCL_V3_INPUT_BINDING_PASS"] = bool(
        report["directory_count"] == 4
        and report["file_count"] == EXPECTED_PCL_V3_FILE_COUNT
        and report["size_bytes"] == EXPECTED_PCL_V3_SIZE_BYTES
        and report["canonical_rows_json_sha256"] == EXPECTED_PCL_V3_TREE_SHA256
        and report["pcl_point_to_plane_cli_sha256"]
        == EXPECTED_PCL_V3_CLI_SHA256
    )
    return report


def _run_pcl_v3(qualification_root: Path) -> dict[str, Any]:
    source_binding = _pcl_v3_inventory(PCL_V3_SOURCE)
    if source_binding["PCL_V3_INPUT_BINDING_PASS"] is not True:
        raise RuntimeError("external PCL-v3 requalification package binding mismatch")
    destination = qualification_root / "pcl_v3_fixture"
    destination.mkdir()
    for name in PCL_V3_DIRECTORIES:
        shutil.copytree(PCL_V3_SOURCE / name, destination / name)
    copied_binding = _pcl_v3_inventory(destination)
    if (
        copied_binding["PCL_V3_INPUT_BINDING_PASS"] is not True
        or copied_binding["file_inventory"] != source_binding["file_inventory"]
    ):
        raise RuntimeError("copied PCL-v3 package differs from fixed input binding")
    results = destination / "results"
    results.mkdir()
    verifier = destination / "tools/pcl_point_to_plane/verify_pcl_backend_v3.py"
    cli = destination / "bin/pcl_point_to_plane_cli"
    metric = destination / "bin/rotation_metric_v3_cli"
    data = destination / "tests/data/pcl_backend_v2"
    cases = (
        (
            "NONDEGENERATE_IDENTITY",
            "nondegenerate_identity_config.json",
            None,
            "nondegenerate_identity.json",
        ),
        (
            "KNOWN_SMALL_TRANSFORM",
            "known_small_transform_config.json",
            "known_small_transform_truth.json",
            "known_small_transform.json",
        ),
        (
            "PLANAR_DEGENERACY_DIAGNOSTIC",
            "planar_degeneracy_config.json",
            None,
            "planar_degeneracy_diagnostic.json",
        ),
    )
    rows = []
    for test, config, truth, output_name in cases:
        output = results / output_name
        command = [
            sys.executable,
            str(verifier),
            "--cli",
            str(cli),
            "--metric-cli",
            str(metric),
            "--config",
            str(data / config),
            "--test",
            test,
            "--output",
            str(output),
        ]
        if truth is not None:
            command.extend(["--truth", str(data / truth)])
        trace_path = (
            qualification_root
            / "supervisor_logs"
            / f"pcl_v3_{test.lower()}.strace"
        )
        completed, source_access = _run_source_traced(
            command,
            cwd=destination,
            trace_path=trace_path,
            capture_output=True,
            text=True,
            check=False,
            env=dict(os.environ),
        )
        if completed.returncode != 0:
            raise RuntimeError(f"PCL-v3 fixture failed: {test}: {completed.stderr}")
        payload = json.loads(output.read_text())
        if payload.get("microtest_pass") is not True:
            raise RuntimeError(f"PCL-v3 verifier rejected case: {test}")
        raw = payload["cli_result"]
        rows.append(
            {
                "test": test,
                "output_path": str(output),
                "output_sha256": _sha256(output),
                "condition_status": raw["condition_status"],
                "point_cloud_rank": raw["point_cloud_rank"],
                "point_to_plane_jacobian_rank": raw[
                    "point_to_plane_jacobian_rank"
                ],
                "source_access_evidence": source_access,
                "pass": True,
            }
        )
    return {
        "label": "PCL backend qualification v3 fixture",
        "test_count": 3,
        "passed_count": 3,
        "failure_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "unexpected_skip_count": 0,
        "source_package_binding": source_binding,
        "copied_package_binding": copied_binding,
        "source_repository_open_read_count": sum(
            row["source_access_evidence"]["source_repository_open_read_count"]
            for row in rows
        ),
        "pass": bool(
            source_binding["PCL_V3_INPUT_BINDING_PASS"] is True
            and copied_binding["PCL_V3_INPUT_BINDING_PASS"] is True
            and all(
                row["source_access_evidence"][
                    "SOURCE_REPOSITORY_SUBPROCESS_ACCESS_PASS"
                ]
                is True
                for row in rows
            )
        ),
        "results": rows,
    }


def _safe_archive_relative(name: str, *, prefix: str) -> str | None:
    if not name or "\\" in name:
        raise RuntimeError(f"unsafe tar member name: {name!r}")
    candidate = PurePosixPath(name)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise RuntimeError(f"unsafe tar member path: {name!r}")
    if not candidate.parts or candidate.parts[0] != prefix:
        raise RuntimeError(f"tar member is outside fixed archive root: {name!r}")
    if len(candidate.parts) == 1:
        return None
    return PurePosixPath(*candidate.parts[1:]).as_posix()


def _compare_live_tree_to_tar(live_root: Path, tar_path: Path) -> dict[str, Any]:
    if live_root.is_symlink() or not live_root.is_dir():
        raise RuntimeError("live failure archive root is missing or unsafe")
    if tar_path.is_symlink() or not tar_path.is_file():
        raise RuntimeError("fixed failure archive tar is missing or unsafe")
    expected_files: dict[str, dict[str, Any]] = {}
    expected_directories: set[str] = set()
    seen_members: set[str] = set()
    with tarfile.open(tar_path, mode="r:gz") as archive:
        for member in archive:
            relative = _safe_archive_relative(member.name, prefix=live_root.name)
            normalized = live_root.name if relative is None else f"{live_root.name}/{relative}"
            if normalized in seen_members:
                raise RuntimeError(f"duplicate tar member: {member.name}")
            seen_members.add(normalized)
            if relative is None:
                if not member.isdir():
                    raise RuntimeError("fixed tar root member is not a directory")
                continue
            if member.isdir():
                expected_directories.add(relative)
                continue
            if not member.isreg():
                raise RuntimeError(f"tar contains link or special member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read tar member: {member.name}")
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            if size != member.size:
                raise RuntimeError(f"tar member size changed while reading: {member.name}")
            expected_files[relative] = {
                "sha256": digest.hexdigest(),
                "size": member.size,
                "mode": member.mode,
            }

    live_files: dict[str, dict[str, Any]] = {}
    live_directories: set[str] = set()
    unsafe_entries: list[str] = []
    for path in sorted(
        live_root.rglob("*"), key=lambda item: item.relative_to(live_root).as_posix()
    ):
        relative = path.relative_to(live_root).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            live_directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            live_files[relative] = {
                "sha256": _sha256(path),
                "size": metadata.st_size,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
        else:
            unsafe_entries.append(relative)
    expected_names = set(expected_files)
    live_names = set(live_files)
    missing_files = sorted(expected_names - live_names)
    extra_files = sorted(live_names - expected_names)
    missing_directories = sorted(expected_directories - live_directories)
    extra_directories = sorted(live_directories - expected_directories)
    size_mismatches: list[str] = []
    mode_mismatches: list[str] = []
    sha_mismatches: list[str] = []
    for relative in sorted(expected_names & live_names):
        expected = expected_files[relative]
        actual = live_files[relative]
        if actual["size"] != expected["size"]:
            size_mismatches.append(relative)
        if actual["mode"] != expected["mode"]:
            mode_mismatches.append(relative)
        if actual["sha256"] != expected["sha256"]:
            sha_mismatches.append(relative)
    report = {
        "tar_member_count": len(seen_members),
        "tar_regular_file_count": len(expected_files),
        "tar_directory_count": len(expected_directories) + 1,
        "live_regular_file_count": len(live_files),
        "live_directory_count": len(live_directories) + 1,
        "tar_duplicate_member_count": 0,
        "tar_unsafe_member_count": 0,
        "live_unsafe_entry_count": len(unsafe_entries),
        "live_unsafe_entries": unsafe_entries,
        "missing_file_count": len(missing_files),
        "missing_files": missing_files,
        "extra_file_count": len(extra_files),
        "extra_files": extra_files,
        "missing_directory_count": len(missing_directories),
        "missing_directories": missing_directories,
        "extra_directory_count": len(extra_directories),
        "extra_directories": extra_directories,
        "size_mismatch_count": len(size_mismatches),
        "size_mismatch_files": size_mismatches,
        "mode_mismatch_count": len(mode_mismatches),
        "mode_mismatch_files": mode_mismatches,
        "sha256_mismatch_count": len(sha_mismatches),
        "sha256_mismatch_files": sha_mismatches,
    }
    report["LIVE_FAILURE_ARCHIVE_TAR_TREE_PASS"] = all(
        report[name] == 0
        for name in (
            "live_unsafe_entry_count",
            "missing_file_count",
            "extra_file_count",
            "missing_directory_count",
            "extra_directory_count",
            "size_mismatch_count",
            "mode_mismatch_count",
            "sha256_mismatch_count",
        )
    )
    return report


def _verify_sha256sums(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        raise RuntimeError("failure archive SHA256SUMS is missing or unsafe")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise RuntimeError(f"malformed SHA256SUMS line {line_number}")
        digest, relative = match.groups()
        candidate = PurePosixPath(relative)
        if (
            candidate.is_absolute()
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise RuntimeError(f"unsafe SHA256SUMS path: {relative!r}")
        normalized = candidate.as_posix()
        if normalized in entries:
            raise RuntimeError(f"duplicate SHA256SUMS path: {normalized}")
        entries[normalized] = digest
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != sums
    }
    missing = sorted(actual_files - set(entries))
    extra = sorted(set(entries) - actual_files)
    mismatches = []
    for relative in sorted(set(entries) & actual_files):
        path = root / PurePosixPath(relative)
        if path.is_symlink() or not path.is_file() or _sha256(path) != entries[relative]:
            mismatches.append(relative)
    return {
        "sha256_entry_count": len(entries),
        "sha256_missing_count": len(missing),
        "sha256_missing_files": missing,
        "sha256_extra_count": len(extra),
        "sha256_extra_files": extra,
        "sha256_mismatch_count": len(mismatches),
        "sha256_mismatch_files": mismatches,
        "SHA256SUMS_PASS": not missing and not extra and not mismatches,
    }


def _archive_verification(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(FAILURE_TAR) != EXPECTED_FAILURE_TAR_SHA256:
        raise RuntimeError("v2 failure archive tar SHA mismatch")
    if _sha256(FAILURE_BUNDLE) != EXPECTED_FAILURE_BUNDLE_SHA256:
        raise RuntimeError("v2 failure bundle SHA mismatch")
    if _sha256(PRE_RUN_BUNDLE) != EXPECTED_PRE_RUN_BUNDLE_SHA256:
        raise RuntimeError("v2 pre-run bundle SHA mismatch")
    for bundle in (FAILURE_BUNDLE, PRE_RUN_BUNDLE):
        subprocess.run(
            ["git", "bundle", "verify", str(bundle)],
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    if _git(repository, "rev-parse", f"{FAILURE_TAG}^{{commit}}") != FAILURE_COMMIT:
        raise RuntimeError("failure tag commit mismatch")
    if _git(repository, "rev-parse", f"{PRE_RUN_TAG}^{{commit}}") != FAILURE_COMMIT:
        raise RuntimeError("pre-run tag commit mismatch")
    tar_tree = _compare_live_tree_to_tar(FAILURE_ARCHIVE, FAILURE_TAR)
    checksum_manifest = _verify_sha256sums(FAILURE_ARCHIVE)
    source_archive = json.loads(
        (FAILURE_ARCHIVE / "reports/source_archive_verification.json").read_text()
    )
    archive = {
        "V2_FAILURE_ARCHIVE_PASS": bool(
            tar_tree["LIVE_FAILURE_ARCHIVE_TAR_TREE_PASS"] is True
            and checksum_manifest["SHA256SUMS_PASS"] is True
            and source_archive.get("SOURCE_ARCHIVE_MISSING_COUNT") == 0
            and source_archive.get("SOURCE_ARCHIVE_EXTRA_COUNT") == 0
            and source_archive.get("SOURCE_ARCHIVE_SIZE_MISMATCH_COUNT") == 0
            and source_archive.get("SOURCE_ARCHIVE_SHA_MISMATCH_COUNT") == 0
        ),
        "V2_FAILURE_ARCHIVE_SHA_PASS": bool(
            tar_tree["LIVE_FAILURE_ARCHIVE_TAR_TREE_PASS"] is True
            and checksum_manifest["SHA256SUMS_PASS"] is True
        ),
        "archive_root": str(FAILURE_ARCHIVE),
        "archive_tar_path": str(FAILURE_TAR),
        "archive_tar_sha256": _sha256(FAILURE_TAR),
        "failure_bundle_path": str(FAILURE_BUNDLE),
        "failure_bundle_sha256": _sha256(FAILURE_BUNDLE),
        "pre_run_bundle_path": str(PRE_RUN_BUNDLE),
        "pre_run_bundle_sha256": _sha256(PRE_RUN_BUNDLE),
        "live_tar_tree_verification": tar_tree,
        **checksum_manifest,
        **source_archive,
    }
    retirement = json.loads((FAILURE_ARCHIVE / "v2_seed_retirement.json").read_text())
    required = {
        "V2_CONFIRMATORY_SEEDS_DECLARED": True,
        "V2_CONFIRMATORY_SEEDS_INSTANTIATED": True,
        "V2_CONFIRMATORY_SEEDS_CONSUMED": True,
        "V2_CONFIRMATORY_SEED_SET_REUSE_AUTHORIZED": False,
    }
    retirement_pass = all(retirement.get(key) == value for key, value in required.items())
    retirement_report = {
        **retirement,
        "V2_SEED_RETIREMENT_PASS": retirement_pass,
        "SYNTHETIC_CONFIRMATORY_V2_PASS": "NOT_EVALUATED",
    }
    if not archive["V2_FAILURE_ARCHIVE_PASS"] or not retirement_pass:
        raise RuntimeError("v2 failure archive or seed retirement gate failed")
    return archive, retirement_report


def _scenario_layout(qualification_root: Path, run_id: str, *, resume: bool):
    from phase_a_harness.runtime_path_policy import qualify_runtime_paths

    return qualify_runtime_paths(
        run_id,
        runtime_root=qualification_root,
        repository_root=Path(__file__).resolve().parents[1],
        run_kind="fixture",
        resume=resume,
    ).layout


def _contract_from_lock(layout) -> tuple[dict[str, Any], str]:
    from phase_a_harness.runtime_lifecycle_io import (
        canonical_json_sha256,
        read_canonical_json,
    )

    lock = read_canonical_json(layout.snapshot_lock)
    contract = lock["contract"]
    return contract, canonical_json_sha256(contract)


def _execution_rows(repository: Path, layout) -> list[dict[str, Any]]:
    from phase_a_harness.runtime_lifecycle_fixture import load_completed_fixture_results

    contract, contract_sha = _contract_from_lock(layout)
    return load_completed_fixture_results(
        repository=repository,
        layout=layout,
        run_id=layout.run_id,
        contract_sha256=contract_sha,
        implementation_sha256=contract["implementation_sha256"],
    )


def _scientific_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "runtime_ms"}
        for row in sorted(rows, key=lambda item: str(item["planned_trial_id"]))
    ]


def _relayout(layout, cloned_root: Path):
    updates = {"run_root": cloned_root}
    for name, path in layout.mutable_paths().items():
        updates[name] = cloned_root / path.relative_to(layout.run_root)
    return replace(layout, **updates)


def _negative_lock_and_corruption_tests(
    repository: Path,
    qualification_root: Path,
    fresh_layout,
    artifact: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    from phase_a_harness.phase_a_execution_chain_fixture import build_fixture_snapshots
    from phase_a_harness.runtime_lifecycle_fixture import (
        RuntimeLifecycleCorruption,
        audit_runtime_results,
        validate_runtime_snapshot,
    )
    from phase_a_harness.runtime_lifecycle_io import (
        ImmutableRunLockError,
        SymlinkPathError,
        canonical_json_bytes,
        read_canonical_json,
        resume_immutable_run_lock,
    )
    from phase_a_harness.synthetic_confirmatory_v2_artifact_verifier import (
        verify_synthetic_confirmatory_v2_fixture_artifact,
    )

    root = qualification_root / "corruption_cases"
    root.mkdir()
    lock_value = read_canonical_json(fresh_layout.snapshot_lock)
    contract = lock_value["contract"]
    lock_cases: dict[str, bool] = {}
    for name, changed in (
        ("different_run_id", {**contract, "run_id": "different-run"}),
        ("different_manifest", {**contract, "fixture_plan_sha256": "0" * 64}),
        ("different_commit", {**contract, "expected_commit": "0" * 40}),
        (
            "different_runtime_root",
            {
                **contract,
                "runtime_paths": {
                    **contract["runtime_paths"],
                    "runtime_root": "/tmp/different-runtime-root",
                },
            },
        ),
    ):
        try:
            resume_immutable_run_lock(fresh_layout.snapshot_lock, changed)
        except ImmutableRunLockError:
            lock_cases[name] = True
        else:
            lock_cases[name] = False
    tampered_lock = root / "tampered_lock.json"
    tampered = dict(lock_value)
    tampered["payload_sha256"] = "0" * 64
    tampered_lock.write_bytes(canonical_json_bytes(tampered))
    try:
        resume_immutable_run_lock(tampered_lock, contract)
    except ImmutableRunLockError:
        lock_cases["tampered_lock"] = True
    else:
        lock_cases["tampered_lock"] = False
    symlink_lock = root / "symlink_lock.json"
    symlink_lock.symlink_to(fresh_layout.snapshot_lock)
    try:
        resume_immutable_run_lock(symlink_lock, contract)
    except (ImmutableRunLockError, SymlinkPathError):
        lock_cases["symlink_lock"] = True
    else:
        lock_cases["symlink_lock"] = False
    symlink_lock.unlink()
    lock_report = {
        "SNAPSHOT_LOCK_CREATE_PASS": fresh_layout.snapshot_lock.is_file(),
        "SNAPSHOT_LOCK_RESUME_PASS": True,
        "SNAPSHOT_LOCK_TAMPER_REJECTION_PASS": all(lock_cases.values()),
        "completed_lock_policy": "PRESERVE_IMMUTABLE_WITH_RUNTIME_ARCHIVE",
        "negative_cases": lock_cases,
    }

    corruption: dict[str, Any] = {}
    fixture_by_token = {
        item.snapshot_id.rsplit("/", 1)[-1]: item for item in build_fixture_snapshots()
    }
    snapshot_copy = root / "snapshot_corrupt"
    shutil.copytree(fresh_layout.snapshot_cache / "identity", snapshot_copy)
    source = snapshot_copy / "source_points.npy"
    source.write_bytes(source.read_bytes() + b"tamper")
    try:
        validate_runtime_snapshot(snapshot_copy, fixture_by_token["identity"])
    except RuntimeLifecycleCorruption:
        corruption["CORRUPT_SNAPSHOT_REJECTION_PASS"] = True
    else:
        corruption["CORRUPT_SNAPSHOT_REJECTION_PASS"] = False

    lineage_copy = root / "lineage_corrupt"
    shutil.copytree(fresh_layout.snapshot_cache / "nonidentity-reference", lineage_copy)
    lineage_path = lineage_copy / "source_parent_target_indices.npy"
    lineage = np.load(lineage_path, allow_pickle=False)
    lineage[0] = 1
    with lineage_path.open("wb") as stream:
        np.save(stream, lineage, allow_pickle=False)
    try:
        validate_runtime_snapshot(
            lineage_copy, fixture_by_token["nonidentity-reference"]
        )
    except RuntimeLifecycleCorruption:
        corruption["CORRUPT_LINEAGE_REJECTION_PASS"] = True
    else:
        corruption["CORRUPT_LINEAGE_REJECTION_PASS"] = False

    trial_root = root / "trial_corrupt_run"
    shutil.copytree(fresh_layout.run_root, trial_root)
    trial_layout = _relayout(fresh_layout, trial_root)
    trial_file = next((trial_layout.raw_results / "results").iterdir())
    trial_file.write_bytes(trial_file.read_bytes() + b"tamper")
    trial_contract, trial_contract_sha = _contract_from_lock(trial_layout)
    try:
        audit_runtime_results(
            repository=repository,
            layout=trial_layout,
            run_id=trial_layout.run_id,
            contract_sha256=trial_contract_sha,
            implementation_sha256=trial_contract["implementation_sha256"],
        )
    except RuntimeLifecycleCorruption:
        corruption["CORRUPT_TRIAL_REJECTION_PASS"] = True
    else:
        corruption["CORRUPT_TRIAL_REJECTION_PASS"] = False

    artifact_copy = root / "artifact_corrupt"
    shutil.copytree(artifact, artifact_copy)
    table = artifact_copy / "tables/fixture_trial_inventory.csv"
    table.write_bytes(table.read_bytes() + b"tamper")
    artifact_report = verify_synthetic_confirmatory_v2_fixture_artifact(
        artifact_copy, write_report=False
    )
    corruption["CORRUPT_ARTIFACT_REJECTION_PASS"] = (
        artifact_report["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is False
    )
    corruption["corruption_classification"] = "CORRUPT_OR_TAMPERED_RUNTIME_OBJECT"
    corruption["all_corruption_rejections_pass"] = all(
        value is True for key, value in corruption.items() if key.endswith("_PASS")
    )
    return lock_report, corruption


def _gitignore_audit(repository: Path) -> dict[str, Any]:
    lines = [
        line.strip()
        for line in (repository / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    required = {
        "/data/synthetic_confirmatory_v1_snapshots/",
        "/data/synthetic_confirmatory_v2_snapshots/",
        "/data/synthetic_confirmatory_v1_snapshot_lock.json",
        "/data/synthetic_confirmatory_v2_snapshot_lock.json",
        "/results/synthetic_confirmatory_v1/",
        "/results/synthetic_confirmatory_v2/",
    }
    forbidden = {"data/", "/data/", "results/", "/results/", "artifacts/", "/artifacts/", "*.json", "*.npy", "*.csv"}
    protected = (
        "src/phase_a_harness/runtime_path_policy.py",
        "protocols/synthetic_confirmatory_protocol_v2.json",
        "frozen_assets/synthetic_confirmatory_formal_manifest_v2.json",
        "tests/test_runtime_path_policy.py",
        "frozen_assets/confirmatory_development_trained_models_v1.json",
        "artifacts/synthetic_confirmatory_v2_prerun/final_decision.json",
    )
    hidden = []
    for relative in protected:
        completed = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", relative],
            cwd=repository,
            check=False,
        )
        if completed.returncode == 0:
            hidden.append(relative)
    return {
        "GITIGNORE_SCOPE_PASS": bool(
            required <= set(lines) and not (forbidden & set(lines)) and not hidden
        ),
        "required_rules": sorted(required),
        "missing_required_rules": sorted(required - set(lines)),
        "forbidden_broad_rules_present": sorted(forbidden & set(lines)),
        "protected_paths_hidden": hidden,
        "rules": lines,
    }


def _manifest_and_sums(root: Path) -> tuple[int, bool]:
    from phase_a_harness.runtime_lifecycle_io import atomic_create_bytes

    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path not in {root / "MANIFEST.csv", root / "SHA256SUMS"}
    )
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in candidates
    ]
    manifest_lines = ["relative_path,size_bytes,sha256\n"]
    for row in rows:
        manifest_lines.append(
            f"{row['relative_path']},{row['size_bytes']},{row['sha256']}\n"
        )
    atomic_create_bytes(root / "MANIFEST.csv", "".join(manifest_lines).encode())
    checksum_candidates = [*candidates, root / "MANIFEST.csv"]
    checksum = "".join(
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in checksum_candidates
    )
    atomic_create_bytes(root / "SHA256SUMS", checksum.encode())
    verified = all(
        _sha256(root / line.split("  ", 1)[1]) == line.split("  ", 1)[0]
        for line in checksum.splitlines()
    )
    return len(checksum_candidates) + 1, verified


def _compact_report(decision: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Runtime Lifecycle Qualification v1",
            "",
            "This compact artifact qualifies external mutable paths, strict Git gates,",
            "real SIGTERM resume, immutable locks, seed-free 3/6 fixture execution,",
            "external analysis/publication, and corruption rejection.",
            "",
            f"- `RUNTIME_LIFECYCLE_QUALIFICATION_PASS = {str(decision['RUNTIME_LIFECYCLE_QUALIFICATION_PASS']).lower()}`",
            f"- `CONFIRMATORY_V3_PRE_RUN_DESIGN_AUTHORIZED = {str(decision['CONFIRMATORY_V3_PRE_RUN_DESIGN_AUTHORIZED']).lower()}`",
            "- `CONFIRMATORY_V3_RUN_AUTHORIZED = false`",
            "- `SYNTHETIC_CONFIRMATORY_V2_PASS = NOT_EVALUATED`",
            "- No Confirmatory seed, namespace, plan, snapshot, or trial was created.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument("--qualification-root", type=Path, default=QUALIFICATION_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _assert_environment()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))
    from phase_a_harness.runner import SourceAccessMonitor
    from phase_a_harness.asset_verifier import source_runtime_import_paths

    supervisor_source_monitor = SourceAccessMonitor()
    supervisor_source_monitor.install()
    from phase_a_harness.runtime_git_gate import verify_runtime_git_gate
    from phase_a_harness.phase_a_execution_chain_fixture import (
        build_fixture_snapshots,
    )
    from phase_a_harness.runtime_lifecycle_fixture import (
        publish_fixture_runtime_artifact,
        summarize_fixture_outcomes,
    )
    from phase_a_harness.runtime_lifecycle_io import (
        append_event_v2,
        atomic_create_canonical_json,
        canonical_json_sha256,
    )
    from phase_a_harness.runtime_lifecycle_science_audit import (
        fixture_scientific_regression,
        scientific_core_binding,
    )
    from phase_a_harness.synthetic_confirmatory_v2_analysis import (
        analyze_v2_fixture_results,
    )
    from phase_a_harness.synthetic_confirmatory_v2_artifact_verifier import (
        FIXTURE_FIGURES,
        FIXTURE_ROOT_FILES,
        FIXTURE_TABLES,
        verify_synthetic_confirmatory_v2_fixture_artifact,
    )
    from phase_a_harness.synthetic_confirmatory_v2_independent_verifier import (
        compare_v2_fixture_primary_and_independent,
        independently_analyze_v2_fixture_results,
    )

    args = build_parser().parse_args(argv)
    qualification_root = args.qualification_root.resolve()
    if qualification_root != QUALIFICATION_ROOT:
        raise PermissionError("a second lifecycle qualification root is forbidden")
    if qualification_root.exists():
        raise FileExistsError("qualification root already exists; refusing a second run")
    if _git(repository, "rev-parse", "HEAD") != args.expected_commit:
        raise PermissionError("expected qualification commit is not HEAD")
    pre_gate = verify_runtime_git_gate(
        repository,
        args.expected_commit,
        args.expected_branch,
        args.expected_tag,
        checkpoint="QUALIFICATION_PRE_RUN_GIT_GATE",
    )
    qualification_root.mkdir(parents=True)
    for name in ("supervisor_logs", "supervisor_git_gates", "test_logs"):
        (qualification_root / name).mkdir()
    log_root = qualification_root / "supervisor_logs"

    archive, retirement = _archive_verification(repository)
    _write_json(qualification_root / "v2_failure_archive_verification.json", archive)
    _write_json(qualification_root / "v2_seed_retirement.json", retirement)
    _write_json(
        qualification_root / "v2_failure_binding.json",
        {
            "failure_commit": FAILURE_COMMIT,
            "failure_tag": FAILURE_TAG,
            "partial_snapshot_count": 139,
            "partial_snapshot_file_count": 565,
            "partial_snapshot_size_bytes": 155306823,
            "formal_trial_count": 0,
            "backend_execution_count": 0,
            "SYNTHETIC_CONFIRMATORY_V2_COMPLETE": False,
            "SYNTHETIC_CONFIRMATORY_V2_PASS": "NOT_EVALUATED",
            "CONFIRMATORY_V2_ROUTE_INVALIDATED_BY_TRUE_IMPLEMENTATION_DEFECT": True,
        },
    )

    gate_sequence = 1

    def supervisor_gate(checkpoint: str) -> dict[str, Any]:
        nonlocal gate_sequence
        report = verify_runtime_git_gate(
            repository,
            args.expected_commit,
            args.expected_branch,
            args.expected_tag,
            checkpoint=checkpoint,
        )
        atomic_create_canonical_json(
            qualification_root
            / "supervisor_git_gates"
            / f"{gate_sequence:04d}_{checkpoint}.json",
            report,
        )
        gate_sequence += 1
        return report

    atomic_create_canonical_json(
        qualification_root / "supervisor_git_gates/0000_QUALIFICATION_PRE_RUN_GIT_GATE.json",
        pre_gate,
    )

    # Scenario A: complete real write plus zero-execution resume validation.
    fresh = _run_worker(
        repository=repository,
        qualification_root=qualification_root,
        log_root=log_root,
        run_id="fresh-real-write",
        invocation_id="fresh",
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_tag=args.expected_tag,
        resume=False,
        delay=0.0,
    )
    fresh_layout = _scenario_layout(
        qualification_root, "fresh-real-write", resume=True
    )
    fresh_snapshot_before = _file_inventory(fresh_layout.snapshot_cache)
    fresh_trial_before = _file_inventory(fresh_layout.raw_results / "results")
    fresh_resume = _run_worker(
        repository=repository,
        qualification_root=qualification_root,
        log_root=log_root,
        run_id="fresh-real-write",
        invocation_id="resume-validation",
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_tag=args.expected_tag,
        resume=True,
        delay=0.0,
    )
    fresh_rows = _execution_rows(repository, fresh_layout)
    fresh_report = {
        "FRESH_REAL_WRITE_LIFECYCLE_PASS": bool(
            fresh["FIXTURE_EXECUTION_CHAIN_PASS"] is True
            and fresh["generated_snapshot_count"] == 3
            and fresh["backend_execution_count_this_invocation"] == 6
            and fresh_resume["generated_snapshot_count"] == 0
            and fresh_resume["backend_execution_count_this_invocation"] == 0
            and fresh_resume["resume_skipped_valid_snapshot_count"] == 3
            and fresh_resume["resume_skipped_valid_result_count"] == 6
        ),
        "fresh": fresh,
        "resume_validation": fresh_resume,
        "snapshot_checksum_change_after_resume": sum(
            fresh_snapshot_before.get(name) != digest
            for name, digest in _file_inventory(fresh_layout.snapshot_cache).items()
        ),
        "trial_checksum_change_after_resume": sum(
            fresh_trial_before.get(name) != digest
            for name, digest in _file_inventory(
                fresh_layout.raw_results / "results"
            ).items()
        ),
    }
    supervisor_gate("FRESH_POST_RESUME_GIT_GATE")

    # Scenario B: terminate after a committed snapshot and resume unchanged.
    snapshot_layout = _scenario_layout(
        qualification_root, "snapshot-interruption", resume=False
    )
    snapshot_interrupt = _wait_and_sigterm(
        repository=repository,
        qualification_root=qualification_root,
        log_root=log_root,
        run_id="snapshot-interruption",
        invocation_id="initial",
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_tag=args.expected_tag,
        progress=lambda: len(
            [
                path
                for path in snapshot_layout.snapshot_cache.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ]
        )
        if snapshot_layout.snapshot_cache.exists()
        else 0,
        total=3,
        stage="snapshot",
    )
    fixture_snapshots = tuple(
        sorted(build_fixture_snapshots(), key=lambda value: value.snapshot_id)
    )
    snapshot_interruption_state = _snapshot_interruption_state(
        snapshot_layout, fixture_snapshots
    )
    initial_snapshot_inventory = snapshot_interruption_state[
        "snapshot_file_inventory"
    ]
    if not initial_snapshot_inventory:
        raise RuntimeError("snapshot interruption produced no committed snapshot")
    snapshot_event = snapshot_layout.attempt_events / "events.ndjson"
    append_event_v2(
        snapshot_event,
        run_id=snapshot_layout.run_id,
        invocation_id="external-supervisor",
        event_type="INFRASTRUCTURE_INTERRUPTION",
        signal_number=int(signal.SIGTERM),
    )
    supervisor_gate("MID_SNAPSHOT_GIT_GATE")
    snapshot_resume = _run_worker(
        repository=repository,
        qualification_root=qualification_root,
        log_root=log_root,
        run_id="snapshot-interruption",
        invocation_id="resume",
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_tag=args.expected_tag,
        resume=True,
        delay=INTERRUPT_DELAY_SECONDS,
    )
    snapshot_rows = _execution_rows(repository, snapshot_layout)
    final_snapshot_inventory = _file_inventory(snapshot_layout.snapshot_cache)
    initial_snapshot_ids = snapshot_interruption_state["committed_snapshot_ids"]
    resume_generated_snapshot_ids = snapshot_resume.get("generated_snapshot_ids")
    if type(resume_generated_snapshot_ids) is not list or not all(
        type(value) is str for value in resume_generated_snapshot_ids
    ):
        raise RuntimeError("worker resume snapshot execution evidence is invalid")
    snapshot_records = _planned_execution_records(
        planned_ids=[item.snapshot_id for item in fixture_snapshots],
        initial_executed_ids=initial_snapshot_ids,
        resume_executed_ids=resume_generated_snapshot_ids,
    )
    snapshot_records_by_id = {row["planned_id"]: row for row in snapshot_records}
    for item in fixture_snapshots:
        token = item.snapshot_id.rsplit("/", 1)[-1]
        before = {
            name: digest
            for name, digest in initial_snapshot_inventory.items()
            if name.startswith(f"{token}/")
        }
        after = {
            name: digest
            for name, digest in final_snapshot_inventory.items()
            if name.startswith(f"{token}/")
        }
        snapshot_records_by_id[item.snapshot_id].update(
            {
                "checksum_before_resume": (
                    canonical_json_sha256(before) if before else None
                ),
                "checksum_after_resume": canonical_json_sha256(after),
            }
        )
    valid_snapshot_reexecution_count = _valid_reexecution_count(snapshot_records)
    duplicate_snapshot_count = _execution_duplicate_count(snapshot_records)
    snapshot_report = {
        "SNAPSHOT_INTERRUPTION_RESUME_PASS": bool(
            snapshot_interrupt["completed_at_signal"] in {1, 2}
            and snapshot_interrupt["completed_at_signal"]
            == snapshot_interruption_state["committed_snapshot_count"]
            and snapshot_resume["FIXTURE_EXECUTION_CHAIN_PASS"] is True
            and snapshot_resume["resume_skipped_valid_snapshot_count"]
            == len(initial_snapshot_ids)
            and snapshot_resume["generated_snapshot_count"]
            == len(resume_generated_snapshot_ids)
            and snapshot_interruption_state["temporary_entry_count"] == 0
            and snapshot_interruption_state["orphan_snapshot_count"] == 0
            and snapshot_interruption_state["duplicate_snapshot_count"] == 0
            and valid_snapshot_reexecution_count == 0
            and duplicate_snapshot_count == 0
            and all(row["final_execution_count"] == 1 for row in snapshot_records)
            and len(snapshot_rows) == 6
        ),
        "interruption": snapshot_interrupt,
        "interruption_state_before_resume": snapshot_interruption_state,
        "resume": snapshot_resume,
        "planned_execution_records": snapshot_records,
        "VALID_SNAPSHOT_REEXECUTION_COUNT": valid_snapshot_reexecution_count,
        "SNAPSHOT_CHECKSUM_CHANGE_AFTER_RESUME": sum(
            row["checksum_before_resume"] is not None
            and row["checksum_before_resume"] != row["checksum_after_resume"]
            for row in snapshot_records
        ),
        "duplicate_snapshot_count": duplicate_snapshot_count,
        "temporary_file_count_after_interruption": snapshot_interruption_state[
            "temporary_entry_count"
        ],
    }

    # Scenario C: terminate after a committed result+manifest update and resume.
    trial_layout = _scenario_layout(
        qualification_root, "trial-interruption", resume=False
    )
    trial_interrupt = _wait_and_sigterm(
        repository=repository,
        qualification_root=qualification_root,
        log_root=log_root,
        run_id="trial-interruption",
        invocation_id="initial",
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_tag=args.expected_tag,
        progress=lambda: _committed_trial_progress(
            trial_layout.raw_results / "raw_result_manifest.json",
            expected_run_id=trial_layout.run_id,
        ),
        total=6,
        stage="trial",
    )
    trial_interruption_state = _trial_interruption_state(trial_layout)
    initial_trial_inventory = trial_interruption_state["result_file_inventory"]
    initial_trial_manifest = trial_interruption_state["raw_manifest"]
    last_trial_id = sorted(initial_trial_manifest["results"])[-1]
    last_entry = initial_trial_manifest["results"][last_trial_id]
    append_event_v2(
        trial_layout.attempt_events / "events.ndjson",
        run_id=trial_layout.run_id,
        invocation_id="external-supervisor",
        event_type="INFRASTRUCTURE_INTERRUPTION",
        planned_trial_id=last_trial_id,
        backend=last_trial_id.rsplit("/", 1)[-1],
        signal_number=int(signal.SIGTERM),
    )
    supervisor_gate("MID_TRIAL_GIT_GATE")
    trial_resume = _run_worker(
        repository=repository,
        qualification_root=qualification_root,
        log_root=log_root,
        run_id="trial-interruption",
        invocation_id="resume",
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_tag=args.expected_tag,
        resume=True,
        delay=INTERRUPT_DELAY_SECONDS,
    )
    trial_rows = _execution_rows(repository, trial_layout)
    final_trial_inventory = _file_inventory(trial_layout.raw_results / "results")
    initial_trial_ids = trial_interruption_state["committed_trial_ids"]
    final_trial_manifest = _read_committed_raw_manifest(
        trial_layout.raw_results / "raw_result_manifest.json",
        expected_run_id=trial_layout.run_id,
    )
    resume_executed_trial_ids = trial_resume.get("executed_trial_ids")
    if type(resume_executed_trial_ids) is not list or not all(
        type(value) is str for value in resume_executed_trial_ids
    ):
        raise RuntimeError("worker resume trial execution evidence is invalid")
    trial_records = _planned_execution_records(
        planned_ids=sorted(final_trial_manifest["results"]),
        initial_executed_ids=initial_trial_ids,
        resume_executed_ids=resume_executed_trial_ids,
    )
    trial_records_by_id = {row["planned_id"]: row for row in trial_records}
    for trial_id, entry in sorted(final_trial_manifest["results"].items()):
        filename = entry["path"]
        trial_records_by_id[trial_id].update(
            {
                "checksum_before_resume": initial_trial_inventory.get(filename),
                "checksum_after_resume": final_trial_inventory[filename],
            }
        )
    valid_trial_reexecution_count = _valid_reexecution_count(trial_records)
    duplicate_trial_count = _execution_duplicate_count(trial_records)
    trial_report = {
        "TRIAL_INTERRUPTION_RESUME_PASS": bool(
            0 < trial_interrupt["completed_at_signal"] < 6
            and trial_interrupt["completed_at_signal"]
            == trial_interruption_state["committed_trial_count"]
            and trial_resume["FIXTURE_EXECUTION_CHAIN_PASS"] is True
            and trial_resume["resume_skipped_valid_result_count"]
            == len(initial_trial_ids)
            and trial_resume["backend_execution_count_this_invocation"]
            == len(resume_executed_trial_ids)
            and trial_resume["recovered_orphan_result_count"] == 0
            and trial_interruption_state["temporary_entry_count"] == 0
            and trial_interruption_state["orphan_result_count"] == 0
            and trial_interruption_state["missing_referenced_result_count"] == 0
            and trial_interruption_state["unsafe_result_entry_count"] == 0
            and trial_interruption_state["duplicate_trial_count"] == 0
            and valid_trial_reexecution_count == 0
            and duplicate_trial_count == 0
            and all(row["final_execution_count"] == 1 for row in trial_records)
            and len(trial_rows) == 6
        ),
        "interruption": trial_interrupt,
        "interruption_state_before_resume": trial_interruption_state,
        "resume": trial_resume,
        "planned_execution_records": trial_records,
        "VALID_TRIAL_REEXECUTION_COUNT": valid_trial_reexecution_count,
        "TRIAL_CHECKSUM_CHANGE_AFTER_RESUME": sum(
            row["checksum_before_resume"] is not None
            and row["checksum_before_resume"] != row["checksum_after_resume"]
            for row in trial_records
        ),
        "duplicate_trial_count": duplicate_trial_count,
        "temporary_file_count_after_interruption": trial_interruption_state[
            "temporary_entry_count"
        ],
    }
    supervisor_gate("RESUME_GIT_GATE")

    # Analysis -> independent verifier -> explicit external publisher staging.
    primary = analyze_v2_fixture_results(fresh_rows)
    independent = independently_analyze_v2_fixture_results(fresh_rows)
    difference = compare_v2_fixture_primary_and_independent(primary, independent)
    fresh_layout.primary_analysis.mkdir(parents=True)
    fresh_layout.independent_verification.mkdir(parents=True)
    atomic_create_canonical_json(
        fresh_layout.primary_analysis / "primary_analysis.json", primary
    )
    atomic_create_canonical_json(
        fresh_layout.independent_verification / "independent_verification.json",
        independent,
    )
    atomic_create_canonical_json(
        fresh_layout.independent_verification / "primary_independent_difference.json",
        difference,
    )
    supervisor_gate("POST_ANALYSIS_GIT_GATE")
    publication_run = {
        "schema_version": "synthetic_confirmatory_v2_fixture_run_v1",
        "backend_execution_count": 6,
        "fixture_snapshot_count": 3,
        "fixture_trial_count": 6,
        "formal_confirmatory_science_evaluated": False,
        "formal_v2_seed_reference_count": 0,
        "fresh_resume_scientific_equivalence": (
            fresh_report["trial_checksum_change_after_resume"] == 0
        ),
        "resume_backend_execution_count": 0,
    }
    publication = publish_fixture_runtime_artifact(
        layout=fresh_layout,
        primary=primary,
        independent=independent,
        run_manifest=publication_run,
    )
    artifact = Path(publication["artifact_staging_path"])
    artifact_verification = verify_synthetic_confirmatory_v2_fixture_artifact(
        artifact, write_report=False
    )
    supervisor_gate("POST_PUBLISHER_GIT_GATE")

    lock_report, corruption = _negative_lock_and_corruption_tests(
        repository, qualification_root, fresh_layout, artifact
    )
    scientific = scientific_core_binding(repository)
    regression = fixture_scientific_regression(repository, fresh_rows)
    fresh_vs_snapshot = _scientific_rows(fresh_rows) == _scientific_rows(snapshot_rows)
    fresh_vs_trial = _scientific_rows(fresh_rows) == _scientific_rows(trial_rows)
    regression.update(
        {
            "fresh_snapshot_resume_scientific_equivalence": fresh_vs_snapshot,
            "fresh_trial_resume_scientific_equivalence": fresh_vs_trial,
        }
    )
    gitignore = _gitignore_audit(repository)
    pcl = _run_pcl_v3(qualification_root)
    runtime_tests = _pytest_suite(
        repository,
        qualification_root,
        "runtime_lifecycle_specialized",
        [
            "tests/test_runtime_path_policy.py",
            "tests/test_runtime_lifecycle_io.py",
            "tests/test_runtime_git_gate.py",
            "tests/test_runtime_lifecycle_fixture.py",
            "tests/test_runtime_lifecycle_qualification.py",
        ],
    )
    v2_tests = _pytest_suite(
        repository,
        qualification_root,
        "v2_specialized",
        ["tests/test_synthetic_confirmatory_v2.py"],
    )
    full_tests = _pytest_suite(
        repository, qualification_root, "full_harness", []
    )
    test_report = {
        "runtime_lifecycle_specialized": runtime_tests,
        "v2_specialized": v2_tests,
        "full_harness": full_tests,
        "pcl_backend_v3": pcl,
        "source_degen_lio_pytest_executed": False,
        "pass": all(
            report["pass"] for report in (runtime_tests, v2_tests, full_tests, pcl)
        ),
    }

    completed_worker_reports = {
        "fresh": fresh,
        "fresh_resume": fresh_resume,
        "snapshot_resume": snapshot_resume,
        "trial_resume": trial_resume,
    }
    interrupted_worker_reports = {
        "snapshot_initial": snapshot_interrupt,
        "trial_initial": trial_interrupt,
    }
    supervisor_import_paths = source_runtime_import_paths()
    supervisor_source_paths = list(supervisor_source_monitor.paths)
    supervisor_source_count = int(supervisor_source_monitor.count)
    subprocess_evidence = {
        "pytest_runtime_lifecycle_specialized": runtime_tests[
            "source_access_evidence"
        ],
        "pytest_v2_specialized": v2_tests["source_access_evidence"],
        "pytest_full_harness": full_tests["source_access_evidence"],
        **{
            f"pcl_v3_{row['test'].lower()}": row["source_access_evidence"]
            for row in pcl["results"]
        },
    }
    worker_log_open_count = sum(
        report["supervisor_source_access_evidence"][
            "source_repository_open_event_count"
        ]
        for report in (
            *completed_worker_reports.values(),
            *interrupted_worker_reports.values(),
        )
    )
    worker_reported_read_count = sum(
        report["source_repository_runtime_file_read_count"]
        for report in completed_worker_reports.values()
    )
    worker_reported_import_count = sum(
        report["source_repository_runtime_import_count"]
        for report in completed_worker_reports.values()
    )
    subprocess_open_read_count = sum(
        evidence["source_repository_open_read_count"]
        for evidence in subprocess_evidence.values()
    )
    source_access_audit = {
        "supervisor": {
            "source_repository_runtime_file_read_count": supervisor_source_count,
            "source_repository_runtime_file_read_paths": supervisor_source_paths,
            "source_repository_runtime_import_count": len(supervisor_import_paths),
            "source_repository_runtime_import_paths": supervisor_import_paths,
        },
        "completed_worker_evidence": {
            name: report["supervisor_source_access_evidence"]
            for name, report in completed_worker_reports.items()
        },
        "interrupted_worker_evidence": {
            name: report["supervisor_source_access_evidence"]
            for name, report in interrupted_worker_reports.items()
        },
        "subprocess_evidence": subprocess_evidence,
        "worker_source_repository_open_event_count": worker_log_open_count,
        "worker_reported_source_repository_file_read_count": (
            worker_reported_read_count
        ),
        "worker_reported_source_repository_import_count": (
            worker_reported_import_count
        ),
        "subprocess_source_repository_open_read_count": (
            subprocess_open_read_count
        ),
        "source_repository_runtime_file_read_count": (
            supervisor_source_count
            + worker_log_open_count
            + worker_reported_read_count
            + subprocess_open_read_count
        ),
        "source_repository_runtime_import_count": (
            len(supervisor_import_paths) + worker_reported_import_count
        ),
    }
    source_access_audit["SOURCE_REPOSITORY_RUNTIME_ISOLATION_PASS"] = bool(
        source_access_audit["source_repository_runtime_file_read_count"] == 0
        and source_access_audit["source_repository_runtime_import_count"] == 0
        and all(
            evidence["SOURCE_ACCESS_LOG_PASS"] is True
            for evidence in (
                *source_access_audit["completed_worker_evidence"].values(),
                *source_access_audit["interrupted_worker_evidence"].values(),
            )
        )
        and all(
            evidence["SOURCE_REPOSITORY_SUBPROCESS_ACCESS_PASS"] is True
            for evidence in subprocess_evidence.values()
        )
    )

    path_audit = fresh["runtime_path_audit"]
    runtime_policy = {
        "default_runtime_root": "/home/lj/zero_perturbation_runtime",
        "qualification_root": str(qualification_root),
        "mutable_path_names": sorted(fresh_layout.mutable_paths()),
        "formal_runtime_artifact_stage": str(artifact),
        "repository_publication_import": "SEPARATE_EXPLICIT_STEP_ONLY",
        "runtime_path_policy_sha256": fresh["runtime_path_policy_sha256"],
    }
    all_git_reports = [
        json.loads(path.read_text())
        for path in sorted(
            [
                *qualification_root.glob("supervisor_git_gates/*.json"),
                *qualification_root.glob("fixture/*/working_inventory/git_gates/*.json"),
            ]
        )
    ]
    checkpoint_pass = {
        name: any(
            name in report["checkpoint"] and report["RUNTIME_GIT_GATE_PASS"] is True
            for report in all_git_reports
        )
        for name in (
            "PRE_RUN_GIT_GATE",
            "MID_SNAPSHOT_GIT_GATE",
            "MID_TRIAL_GIT_GATE",
            "RESUME_GIT_GATE",
            "POST_ANALYSIS_GIT_GATE",
            "POST_PUBLISHER_GIT_GATE",
        )
    }
    final_gate = supervisor_gate("FINAL_GIT_GATE")
    checkpoint_pass["FINAL_GIT_GATE"] = final_gate["RUNTIME_GIT_GATE_PASS"] is True
    git_gate_audit = {
        **{f"{name}_PASS": value for name, value in checkpoint_pass.items()},
        "gate_report_count": len(all_git_reports) + 1,
        "gate_failure_count": sum(
            report["RUNTIME_GIT_GATE_PASS"] is not True for report in all_git_reports
        ),
        "strict_untracked_semantics": True,
        "dynamic_allowlist_used": False,
    }
    publisher_inventory = {
        "publisher_table_count": len(FIXTURE_TABLES),
        "publisher_figure_count": len(FIXTURE_FIGURES),
        "publisher_root_file_count": len(FIXTURE_ROOT_FILES),
        "published_file_count": publication["published_file_count"],
        "missing_count": len(artifact_verification["missing_required_files"]),
        "extra_count": len(artifact_verification["extra_files"]),
        "sha256_mismatch_count": len(artifact_verification["sha256_mismatch_files"]),
        "FIXTURE_ARTIFACT_PUBLICATION_PASS": publication[
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        ],
    }

    required_values = [
        archive["V2_FAILURE_ARCHIVE_PASS"],
        archive["V2_FAILURE_ARCHIVE_SHA_PASS"],
        retirement["V2_SEED_RETIREMENT_PASS"],
        retirement["V2_CONFIRMATORY_SEED_SET_REUSE_AUTHORIZED"] is False,
        path_audit["RUNTIME_ROOT_OUTSIDE_REPOSITORY"],
        path_audit["SNAPSHOT_CACHE_OUTSIDE_REPOSITORY"],
        path_audit["SNAPSHOT_LOCK_OUTSIDE_REPOSITORY"],
        path_audit["RAW_RESULTS_OUTSIDE_REPOSITORY"],
        path_audit["ARTIFACT_STAGE_OUTSIDE_REPOSITORY"],
        gitignore["GITIGNORE_SCOPE_PASS"],
        scientific["SCIENTIFIC_CORE_FILE_CHANGE_COUNT"] == 0,
        fresh_report["FRESH_REAL_WRITE_LIFECYCLE_PASS"],
        snapshot_report["SNAPSHOT_INTERRUPTION_RESUME_PASS"],
        trial_report["TRIAL_INTERRUPTION_RESUME_PASS"],
        snapshot_report["VALID_SNAPSHOT_REEXECUTION_COUNT"] == 0,
        trial_report["VALID_TRIAL_REEXECUTION_COUNT"] == 0,
        snapshot_report["SNAPSHOT_CHECKSUM_CHANGE_AFTER_RESUME"] == 0,
        trial_report["TRIAL_CHECKSUM_CHANGE_AFTER_RESUME"] == 0,
        lock_report["SNAPSHOT_LOCK_CREATE_PASS"],
        lock_report["SNAPSHOT_LOCK_RESUME_PASS"],
        lock_report["SNAPSHOT_LOCK_TAMPER_REJECTION_PASS"],
        corruption["CORRUPT_SNAPSHOT_REJECTION_PASS"],
        corruption["CORRUPT_LINEAGE_REJECTION_PASS"],
        corruption["CORRUPT_TRIAL_REJECTION_PASS"],
        corruption["CORRUPT_ARTIFACT_REJECTION_PASS"],
        difference["leaf_difference_count"] == 0,
        summarize_fixture_outcomes(fresh_rows)["FIXTURE_EXECUTION_CHAIN_PASS"],
        publication["FIXTURE_ARTIFACT_VERIFICATION_PASS"],
        artifact_verification["FIXTURE_ARTIFACT_VERIFICATION_PASS"],
        all(checkpoint_pass.values()),
        regression["FIXTURE_SCIENTIFIC_PAYLOAD_CHANGE_COUNT"] == 0,
        regression["FIXTURE_TRIAL_SCIENTIFIC_FIELD_CHANGE_COUNT"] == 0,
        scientific["H1_H6_SEMANTICS_CHANGE_COUNT"] == 0,
        scientific["FROZEN_MODEL_CHANGE_COUNT"] == 0,
        scientific["BACKEND_BINDING_CHANGE_COUNT"] == 0,
        fresh_vs_snapshot,
        fresh_vs_trial,
        source_access_audit["SOURCE_REPOSITORY_RUNTIME_ISOLATION_PASS"],
        test_report["pass"],
    ]
    passed = all(value is True for value in required_values)
    decision = {
        "V2_FAILURE_ARCHIVE_PASS": archive["V2_FAILURE_ARCHIVE_PASS"],
        "V2_FAILURE_ARCHIVE_SHA_PASS": archive["V2_FAILURE_ARCHIVE_SHA_PASS"],
        "V2_SEED_RETIREMENT_PASS": retirement["V2_SEED_RETIREMENT_PASS"],
        "V2_SEED_SET_REUSE_AUTHORIZED": False,
        "REPOSITORY_RESTORED_CLEAN_AFTER_ARCHIVE": True,
        "RUNTIME_LIFECYCLE_QUALIFICATION_PASS": passed,
        "CONFIRMATORY_V3_PRE_RUN_DESIGN_AUTHORIZED": passed,
        "CONFIRMATORY_V3_SEED_DERIVATION_AUTHORIZED": passed,
        "CONFIRMATORY_V3_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_V3_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V3_PASS": "NOT_EVALUATED",
        "REAL_DATA_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "CONFIRMATORY_SEED_ACCESS_COUNT": 0,
        "NEW_CONFIRMATORY_NAMESPACE_GENERATION_COUNT": 0,
        "NATIVE_EXECUTION_COUNT": 0,
        "FORMAL_CONFIRMATORY_EXECUTION_COUNT": 0,
        "SOURCE_REPOSITORY_RUNTIME_FILE_READ_COUNT": source_access_audit[
            "source_repository_runtime_file_read_count"
        ],
        "SOURCE_REPOSITORY_RUNTIME_IMPORT_COUNT": source_access_audit[
            "source_repository_runtime_import_count"
        ],
    }
    if not passed:
        raise RuntimeError("runtime lifecycle qualification gate failed")

    implementation_paths = (
        ".gitignore",
        "src/phase_a_harness/runtime_path_policy.py",
        "src/phase_a_harness/runtime_git_gate.py",
        "src/phase_a_harness/runtime_lifecycle_io.py",
        "src/phase_a_harness/runtime_lifecycle_fixture.py",
        "src/phase_a_harness/runtime_lifecycle_science_audit.py",
        "src/phase_a_harness/synthetic_confirmatory_v2_publisher.py",
        "scripts/run_runtime_lifecycle_fixture.py",
        "scripts/qualify_runtime_lifecycle.py",
        "tests/test_runtime_path_policy.py",
        "tests/test_runtime_git_gate.py",
        "tests/test_runtime_lifecycle_io.py",
        "tests/test_runtime_lifecycle_fixture.py",
        "tests/test_runtime_lifecycle_qualification.py",
        "tests/test_full_synthetic_protocol_assets_runner.py",
        "tests/test_synthetic_confirmatory_v2.py",
    )
    implementation = {
        "qualification_execution_commit": args.expected_commit,
        "qualification_execution_branch": args.expected_branch,
        "qualification_execution_tag": args.expected_tag,
        "files": {
            relative: _sha256(repository / relative) for relative in implementation_paths
        },
        "scientific_core": scientific,
    }
    run_manifest = {
        "qualification_root": str(qualification_root),
        "fresh_run_id": "fresh-real-write",
        "snapshot_interruption_run_id": "snapshot-interruption",
        "trial_interruption_run_id": "trial-interruption",
        "workers": WORKERS,
        "fixture_snapshot_count": 3,
        "fixture_trial_count_per_scenario": 6,
        "open3d_trial_count_per_scenario": 3,
        "pcl_trial_count_per_scenario": 3,
        "native_trial_count": 0,
        "formal_confirmatory_snapshot_count": 0,
        "formal_confirmatory_trial_count": 0,
        "source_repository_runtime_file_read_count": source_access_audit[
            "source_repository_runtime_file_read_count"
        ],
        "source_repository_runtime_import_count": source_access_audit[
            "source_repository_runtime_import_count"
        ],
    }

    compact = qualification_root / "compact_artifact"
    compact.mkdir()
    records = {
        "v2_failure_binding.json": json.loads(
            (qualification_root / "v2_failure_binding.json").read_text()
        ),
        "v2_seed_retirement.json": retirement,
        "v2_failure_archive_verification.json": archive,
        "runtime_path_policy.json": runtime_policy,
        "runtime_path_security_audit.json": path_audit,
        "gitignore_scope_audit.json": gitignore,
        "git_gate_semantics_audit.json": git_gate_audit,
        "scientific_core_binding.json": scientific,
        "fresh_lifecycle_report.json": fresh_report,
        "snapshot_interruption_resume_report.json": snapshot_report,
        "trial_interruption_resume_report.json": trial_report,
        "lock_lifecycle_report.json": lock_report,
        "corruption_rejection_report.json": corruption,
        "fixture_scientific_regression.json": regression,
        "primary_independent_difference.json": difference,
        "publisher_inventory.json": publisher_inventory,
        "artifact_verification.json": artifact_verification,
        "test_report.json": test_report,
        "source_access_audit.json": source_access_audit,
        "implementation_manifest.json": implementation,
        "final_decision.json": decision,
        "run_manifest.json": run_manifest,
    }
    for name, value in records.items():
        atomic_create_canonical_json(compact / name, value)
    (compact / "qualification_report.md").write_text(
        _compact_report(decision), encoding="utf-8"
    )
    compact_count, compact_sha_pass = _manifest_and_sums(compact)
    if not compact_sha_pass:
        raise RuntimeError("compact qualification artifact SHA verification failed")

    # Inventory the complete external qualification tree after every run and test.
    external_count, external_sha_pass = _manifest_and_sums(qualification_root)
    if not external_sha_pass:
        raise RuntimeError("external qualification tree SHA verification failed")
    result = {
        **decision,
        "compact_artifact_path": str(compact),
        "compact_artifact_file_count": compact_count,
        "compact_artifact_sha256_verification_pass": compact_sha_pass,
        "external_runtime_file_count": external_count,
        "external_runtime_sha256_verification_pass": external_sha_pass,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
