from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from phase_a_harness.runtime_git_gate import (
    RuntimeGitGateError,
    verify_runtime_git_gate,
)


BRANCH = "frozen-runtime"
TAG = "archive/runtime-freeze"


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "checkout", "-b", BRANCH)
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "Runtime Gate Fixture")
    (repository / ".gitignore").write_text("ignored-runtime.log\n", encoding="utf-8")
    (repository / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    nested = repository / "nested"
    nested.mkdir()
    (nested / ".keep").write_text("tracked\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "tracked.txt", "nested/.keep")
    _git(repository, "commit", "-m", "freeze fixture")
    _git(repository, "tag", TAG)
    return repository, _git(repository, "rev-parse", "HEAD^{commit}")


def _failure(
    repository: Path,
    commit: str,
    *,
    branch: str = BRANCH,
    tag: str | None = TAG,
) -> dict[str, Any]:
    with pytest.raises(RuntimeGitGateError) as captured:
        verify_runtime_git_gate(
            repository,
            commit,
            branch,
            tag,
            checkpoint="TEST_CHECKPOINT",
        )
    report = captured.value.report
    assert report["RUNTIME_GIT_GATE_PASS"] is False
    assert report["failure_classification_count"] >= 1
    return report


def test_fresh_repository_passes_with_exact_identity_and_ignored_file(
    tmp_path: Path,
) -> None:
    repository, commit = _repository(tmp_path)
    (repository / "ignored-runtime.log").write_text("ignored\n", encoding="utf-8")

    report = verify_runtime_git_gate(
        repository,
        commit,
        BRANCH,
        TAG,
        checkpoint="PRE_RUN",
    )

    assert report["RUNTIME_GIT_GATE_PASS"] is True
    assert report["checkpoint"] == "PRE_RUN"
    assert report["repository_top_level_match"] is True
    assert report["head_commit_match"] is True
    assert report["branch_match"] is True
    assert report["tag_commit_match"] is True
    assert report["tracked_diff_count"] == 0
    assert report["index_diff_count"] == 0
    assert report["untracked_file_count"] == 0
    assert report["porcelain_status_clean"] is True
    assert report["failure_classifications"] == []


def test_repository_argument_must_be_git_top_level(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)

    report = _failure(repository / "nested", commit)

    assert report["repository_top_level_match"] is False
    assert report["failure_classifications"] == ["REPOSITORY_NOT_TOP_LEVEL"]


def test_tracked_worktree_change_fails_closed(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")

    report = _failure(repository, commit)

    assert report["tracked_diff_count"] == 1
    assert report["index_diff_count"] == 0
    assert report["untracked_file_count"] == 0
    assert report["failure_classifications"] == ["TRACKED_WORKTREE_DIRTY"]


def test_staged_index_change_fails_closed(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    (repository / "tracked.txt").write_text("staged\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")

    report = _failure(repository, commit)

    assert report["tracked_diff_count"] == 0
    assert report["index_diff_count"] == 1
    assert report["untracked_file_count"] == 0
    assert report["failure_classifications"] == ["INDEX_DIRTY"]


def test_every_nonignored_untracked_file_fails_closed(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    (repository / "runtime-output.json").write_text("{}\n", encoding="utf-8")

    report = _failure(repository, commit)

    assert report["tracked_diff_count"] == 0
    assert report["index_diff_count"] == 0
    assert report["untracked_file_count"] == 1
    assert report["untracked_paths"] == ["runtime-output.json"]
    assert report["failure_classifications"] == ["UNTRACKED_FILES_PRESENT"]


def test_wrong_branch_fails_closed(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    _git(repository, "checkout", "-b", "wrong-runtime-branch")

    report = _failure(repository, commit)

    assert report["branch"] == "wrong-runtime-branch"
    assert report["failure_classifications"] == ["BRANCH_MISMATCH"]


def test_wrong_head_fails_closed(tmp_path: Path) -> None:
    repository, frozen_commit = _repository(tmp_path)
    (repository / "tracked.txt").write_text("new committed state\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "move head")

    report = _failure(repository, frozen_commit, tag=None)

    assert report["head_commit"] != frozen_commit
    assert report["failure_classifications"] == ["HEAD_COMMIT_MISMATCH"]


def test_tag_must_resolve_to_expected_commit(tmp_path: Path) -> None:
    repository, old_commit = _repository(tmp_path)
    (repository / "tracked.txt").write_text("new tagged state\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "new expected head")
    new_commit = _git(repository, "rev-parse", "HEAD^{commit}")
    assert old_commit != new_commit

    report = _failure(repository, new_commit)

    assert report["head_commit_match"] is True
    assert report["tag_commit"] == old_commit
    assert report["failure_classifications"] == ["TAG_COMMIT_MISMATCH"]


def test_missing_required_tag_fails_closed(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)

    report = _failure(repository, commit, tag="archive/missing")

    assert report["tag_commit"] is None
    assert report["command_failure_count"] == 1
    assert report["failure_classifications"] == ["TAG_QUERY_FAILED"]
