from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from phase_a_harness.real_data_preparation.cavers_stage1_verifier import (
    CaversStage1VerificationError,
    RESUME_POLICY,
    _verify_resume_provenance,
)


def _git_commit(repository: Path, message: str) -> str:
    marker = repository / "lineage.txt"
    marker.write_text(marker.read_text(encoding="utf-8") + message + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "lineage.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repository, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()


def _lineage(tmp_path: Path, commit_count: int = 4) -> tuple[Path, Path, list[str]]:
    repository = tmp_path / "repository"
    runtime = tmp_path / "runtime"
    repository.mkdir()
    runtime.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True)
    subprocess.run(
        ["git", "checkout", "-b", "prep/cavers-single-dataset-stage1-v1"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "config", "user.email", "cavers@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "CAVERS verifier test"], cwd=repository, check=True)
    (repository / "lineage.txt").write_text("", encoding="utf-8")
    commits = [_git_commit(repository, f"commit {index}") for index in range(commit_count)]
    return repository, runtime, commits


def _write_report(
    path: Path,
    *,
    commit: str,
    producer: str,
    previous: str | None = None,
) -> None:
    value = {
        "git": {
            "branch": "prep/cavers-single-dataset-stage1-v1",
            "commit": commit,
            "worktree_porcelain": [],
        },
        "resume_origin_commit": producer,
        "resume_policy": RESUME_POLICY,
    }
    if previous is not None:
        value["previous_resume_commit"] = previous
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_accepts_pointer_ordered_multi_resume_chain_independent_of_filename_order(
    tmp_path: Path,
) -> None:
    repository, runtime, commits = _lineage(tmp_path)
    producer, first, second, third = commits
    _write_report(runtime / "resume_environment_report.json", commit=first, producer=producer)
    # Filenames are hashes and therefore not chronological; explicit previous
    # pointers, rather than lexicographic order, define the immutable chain.
    _write_report(
        runtime / f"resume_environment_report_{third}.json",
        commit=third,
        producer=producer,
        previous=second,
    )
    _write_report(
        runtime / f"resume_environment_report_{second}.json",
        commit=second,
        producer=producer,
        previous=first,
    )

    chain = _verify_resume_provenance(
        root=runtime,
        repository=repository,
        producer_commit=producer,
        finalizing_commit=third,
    )

    assert chain == [first, second, third]


def test_rejects_resume_pointer_fork(tmp_path: Path) -> None:
    repository, runtime, commits = _lineage(tmp_path)
    producer, first, second, third = commits
    _write_report(runtime / "resume_environment_report.json", commit=first, producer=producer)
    for commit in (second, third):
        _write_report(
            runtime / f"resume_environment_report_{commit}.json",
            commit=commit,
            producer=producer,
            previous=first,
        )

    with pytest.raises(CaversStage1VerificationError, match="forks"):
        _verify_resume_provenance(
            root=runtime,
            repository=repository,
            producer_commit=producer,
            finalizing_commit=third,
        )


def test_rejects_disconnected_resume_pointer(tmp_path: Path) -> None:
    repository, runtime, commits = _lineage(tmp_path)
    producer, first, second, third = commits
    _write_report(runtime / "resume_environment_report.json", commit=first, producer=producer)
    _write_report(
        runtime / f"resume_environment_report_{second}.json",
        commit=second,
        producer=producer,
        previous=producer,
    )
    _write_report(
        runtime / f"resume_environment_report_{third}.json",
        commit=third,
        producer=producer,
        previous=second,
    )

    with pytest.raises(CaversStage1VerificationError, match="disconnected"):
        _verify_resume_provenance(
            root=runtime,
            repository=repository,
            producer_commit=producer,
            finalizing_commit=third,
        )


def test_rejects_manifest_finalizing_commit_before_pointer_chain_tail(tmp_path: Path) -> None:
    repository, runtime, commits = _lineage(tmp_path, commit_count=3)
    producer, first, second = commits
    _write_report(runtime / "resume_environment_report.json", commit=first, producer=producer)
    _write_report(
        runtime / f"resume_environment_report_{second}.json",
        commit=second,
        producer=producer,
        previous=first,
    )

    with pytest.raises(CaversStage1VerificationError, match="chain tail"):
        _verify_resume_provenance(
            root=runtime,
            repository=repository,
            producer_commit=producer,
            finalizing_commit=first,
        )
