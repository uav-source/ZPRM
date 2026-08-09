from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from test_formal_lifecycle_fresh_resume import (
    QualificationSubprocessRepository,
    git_output,
    prepare_qualification_repository,
    qualification_worker_command,
    run_qualification_worker,
    runtime_inventory,
    strict_json,
)


def _wait_for_progress(
    fixture: QualificationSubprocessRepository,
    process: subprocess.Popen[str],
    *,
    phase: str,
    timeout: float = 45.0,
) -> dict[str, Any]:
    marker = (
        fixture.runtime_root / "working_inventory/durable_progress.json"
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.is_file():
            value = strict_json(marker)
            if (
                value.get("phase") == phase
                and value.get("committed_count", 0) >= 1
            ):
                return value
        if process.poll() is not None:
            break
        time.sleep(0.01)
    stdout, stderr = process.communicate(timeout=10)
    raise AssertionError(
        f"worker did not reach {phase}: rc={process.returncode}, "
        f"stdout={stdout[-1000:]}, stderr={stderr[-1000:]}"
    )


def _start_fresh_worker(
    fixture: QualificationSubprocessRepository,
    *,
    invocation_id: str,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        qualification_worker_command(
            fixture,
            mode="fresh",
            invocation_id=invocation_id,
        ),
        cwd=fixture.repository,
        env=fixture.environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _interrupt_at_phase(
    fixture: QualificationSubprocessRepository,
    *,
    phase: str,
) -> tuple[dict[str, Any], int, str, str]:
    process = _start_fresh_worker(
        fixture,
        invocation_id=f"interrupt-{phase}",
    )
    marker = _wait_for_progress(fixture, process, phase=phase)
    os.killpg(process.pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == -signal.SIGTERM
    return marker, process.returncode, stdout, stderr


def _snapshot_directories(root: Path) -> list[Path]:
    cache = root / "snapshot_cache"
    if not cache.is_dir():
        return []
    return sorted(
        path for path in cache.iterdir() if path.is_dir() and not path.is_symlink()
    )


def test_real_sigterm_after_snapshot_commit_resumes_without_reexecution(
    tmp_path: Path,
) -> None:
    fixture = prepare_qualification_repository(
        tmp_path,
        label="snapshot-interrupt",
        snapshot_delay_seconds=1.0,
    )

    marker, returncode, _stdout, _stderr = _interrupt_at_phase(
        fixture,
        phase="snapshot",
    )
    committed_snapshots = _snapshot_directories(fixture.runtime_root)
    assert returncode == -signal.SIGTERM
    assert marker["phase"] == "snapshot"
    assert marker["committed_count"] == 1
    assert len(committed_snapshots) == 1
    committed_before = runtime_inventory(
        fixture.runtime_root,
        tuple(
            path.relative_to(fixture.runtime_root).as_posix()
            for path in committed_snapshots
        )
        + (
            "formal_command.log",
            "formal_command.log.sha256",
            "immutable_run_lock.json",
        ),
    )

    resumed, _process = run_qualification_worker(
        fixture,
        mode="resume",
        invocation_id="resume-after-snapshot-sigterm",
    )
    invocation = resumed["lifecycle"]["invocation_report"]
    committed_after = runtime_inventory(
        fixture.runtime_root,
        tuple(
            path.relative_to(fixture.runtime_root).as_posix()
            for path in committed_snapshots
        )
        + (
            "formal_command.log",
            "formal_command.log.sha256",
            "immutable_run_lock.json",
        ),
    )

    assert committed_before == committed_after
    assert invocation["resumed_snapshot_count_this_invocation"] == 1
    assert invocation["generated_snapshot_count_this_invocation"] == 2
    assert invocation["resume_skipped_valid_result_count"] == 0
    assert invocation["backend_execution_count_this_invocation"] == 6
    assert invocation["completed_snapshot_count"] == 3
    assert invocation["completed_trial_count"] == 6
    assert invocation["native_trial_count"] == 0
    assert invocation["all_git_gates_pass"] is True
    assert resumed["postrun"]["difference"]["exact_match_pass"] is True
    assert resumed["postrun"]["artifact_verification"][
        "FIXTURE_ARTIFACT_VERIFICATION_PASS"
    ] is True
    assert git_output(fixture.repository, "status", "--porcelain=v1") == ""
