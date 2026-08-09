from __future__ import annotations

import signal
from pathlib import Path

from test_formal_lifecycle_fresh_resume import (
    git_output,
    prepare_qualification_repository,
    run_qualification_worker,
    runtime_inventory,
    strict_json,
)
from test_formal_lifecycle_snapshot_interrupt import _interrupt_at_phase


def _committed_result_paths(root: Path) -> tuple[Path, ...]:
    manifest_path = root / "raw_result_manifest.json"
    if not manifest_path.is_file():
        return ()
    manifest = strict_json(manifest_path)
    return tuple(
        root / "raw_results" / entry["path"]
        for _trial_id, entry in sorted(manifest["results"].items())
    )


def test_real_sigterm_after_trial_commit_resumes_without_reexecution(
    tmp_path: Path,
) -> None:
    fixture = prepare_qualification_repository(
        tmp_path,
        label="trial-interrupt",
        trial_delay_seconds=1.0,
    )

    marker, returncode, _stdout, _stderr = _interrupt_at_phase(
        fixture,
        phase="trial",
    )
    committed_results = _committed_result_paths(fixture.runtime_root)
    assert returncode == -signal.SIGTERM
    assert marker["phase"] == "trial"
    assert marker["committed_count"] == 1
    assert len(committed_results) == 1
    committed_before = runtime_inventory(
        fixture.runtime_root,
        tuple(
            path.relative_to(fixture.runtime_root).as_posix()
            for path in committed_results
        )
        + (
            "snapshot_cache",
            "snapshot_lock.json",
            "formal_command.log",
            "formal_command.log.sha256",
            "immutable_run_lock.json",
        ),
    )

    resumed, _process = run_qualification_worker(
        fixture,
        mode="resume",
        invocation_id="resume-after-trial-sigterm",
    )
    invocation = resumed["lifecycle"]["invocation_report"]
    committed_after = runtime_inventory(
        fixture.runtime_root,
        tuple(
            path.relative_to(fixture.runtime_root).as_posix()
            for path in committed_results
        )
        + (
            "snapshot_cache",
            "snapshot_lock.json",
            "formal_command.log",
            "formal_command.log.sha256",
            "immutable_run_lock.json",
        ),
    )

    assert committed_before == committed_after
    assert invocation["resumed_snapshot_count_this_invocation"] == 3
    assert invocation["generated_snapshot_count_this_invocation"] == 0
    assert invocation["resume_skipped_valid_result_count"] == 1
    assert invocation["backend_execution_count_this_invocation"] == 5
    assert invocation["completed_snapshot_count"] == 3
    assert invocation["completed_trial_count"] == 6
    assert invocation["native_trial_count"] == 0
    assert invocation["all_git_gates_pass"] is True
    assert resumed["postrun"]["difference"]["exact_match_pass"] is True
    assert resumed["postrun"]["artifact_verification"][
        "FIXTURE_ARTIFACT_VERIFICATION_PASS"
    ] is True
    assert git_output(fixture.repository, "status", "--porcelain=v1") == ""
