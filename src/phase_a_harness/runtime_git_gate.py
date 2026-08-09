"""Strict, reusable Git identity and cleanliness gate for runtime checkpoints.

The gate intentionally has no path allowlist.  Every untracked path not hidden
by Git's normal ignore rules fails the gate, regardless of where it lives.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence


_FULL_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")


class RuntimeGitGateError(PermissionError):
    """Fail-closed gate error carrying the complete diagnostic report."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        classifications = report.get("failure_classifications", [])
        checkpoint = report.get("checkpoint", "UNSPECIFIED")
        detail = ", ".join(str(value) for value in classifications)
        super().__init__(f"runtime Git gate failed at {checkpoint}: {detail}")


def _git(
    repository: Path, arguments: Sequence[str]
) -> tuple[int | None, bytes, str]:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except (OSError, ValueError) as error:
        return None, b"", f"{type(error).__name__}: {error}"
    return (
        completed.returncode,
        completed.stdout,
        completed.stderr.decode("utf-8", errors="replace").strip(),
    )


def _text(stdout: bytes) -> str:
    return stdout.decode("utf-8", errors="surrogateescape").strip()


def _nul_paths(stdout: bytes) -> list[str]:
    return [
        os.fsdecode(value)
        for value in stdout.split(b"\0")
        if value
    ]


def verify_runtime_git_gate(
    repository: str | Path,
    expected_commit: str,
    expected_branch: str,
    expected_tag: str | None = None,
    *,
    checkpoint: str = "UNSPECIFIED",
) -> dict[str, Any]:
    """Verify frozen Git identity and a completely clean repository.

    On success, return a structured report.  On any identity, query, or
    cleanliness failure, raise :class:`RuntimeGitGateError`; its ``report``
    attribute contains the same schema with explicit failure classifications
    and counts.  Non-ignored untracked files are never allowlisted.
    """

    failures: list[str] = []
    command_failures: dict[str, dict[str, Any]] = {}

    try:
        repository_path = Path(repository).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        report = {
            "RUNTIME_GIT_GATE_PASS": False,
            "checkpoint": str(checkpoint),
            "failure_classification_count": 1,
            "failure_classifications": ["REPOSITORY_PATH_INVALID"],
            "repository_input": str(repository),
            "repository_path_error": f"{type(error).__name__}: {error}",
            "tracked_diff_count": None,
            "index_diff_count": None,
            "untracked_file_count": None,
        }
        raise RuntimeGitGateError(report) from error

    expected_commit_valid = bool(
        isinstance(expected_commit, str)
        and _FULL_COMMIT_RE.fullmatch(expected_commit)
    )
    normalized_expected_commit = (
        expected_commit.lower() if expected_commit_valid else None
    )
    if not expected_commit_valid:
        failures.append("EXPECTED_COMMIT_INVALID")

    expected_branch_valid = bool(
        isinstance(expected_branch, str)
        and expected_branch
        and "\0" not in expected_branch
        and "\n" not in expected_branch
        and "\r" not in expected_branch
    )
    if not expected_branch_valid:
        failures.append("EXPECTED_BRANCH_INVALID")

    checkpoint_valid = bool(
        isinstance(checkpoint, str)
        and checkpoint
        and "\0" not in checkpoint
        and "\n" not in checkpoint
        and "\r" not in checkpoint
    )
    if not checkpoint_valid:
        failures.append("CHECKPOINT_INVALID")

    expected_tag_valid = expected_tag is None or bool(
        isinstance(expected_tag, str)
        and expected_tag
        and not expected_tag.startswith("refs/")
        and "\0" not in expected_tag
        and "\n" not in expected_tag
        and "\r" not in expected_tag
    )
    if not expected_tag_valid:
        failures.append("EXPECTED_TAG_INVALID")

    if not repository_path.is_dir():
        failures.append("REPOSITORY_NOT_DIRECTORY")

    def query(name: str, *arguments: str) -> bytes | None:
        returncode, stdout, stderr = _git(repository_path, arguments)
        if returncode != 0:
            command_failures[name] = {
                "arguments": list(arguments),
                "returncode": returncode,
                "stderr": stderr,
            }
            failures.append(f"{name.upper()}_QUERY_FAILED")
            return None
        return stdout

    top_stdout = query("top_level", "rev-parse", "--show-toplevel")
    top_level = None
    top_level_match = False
    if top_stdout is not None:
        try:
            top_level = str(Path(_text(top_stdout)).resolve())
            top_level_match = Path(top_level) == repository_path
        except (OSError, RuntimeError, ValueError):
            failures.append("TOP_LEVEL_PATH_INVALID")
        if not top_level_match and "TOP_LEVEL_PATH_INVALID" not in failures:
            failures.append("REPOSITORY_NOT_TOP_LEVEL")

    head_stdout = query("head", "rev-parse", "--verify", "HEAD^{commit}")
    head_commit = None if head_stdout is None else _text(head_stdout).lower()
    head_commit_match = bool(
        normalized_expected_commit is not None
        and head_commit == normalized_expected_commit
    )
    if head_stdout is not None and not head_commit_match:
        failures.append("HEAD_COMMIT_MISMATCH")

    branch_stdout = query("branch", "branch", "--show-current")
    branch = None if branch_stdout is None else _text(branch_stdout)
    branch_match = bool(expected_branch_valid and branch == expected_branch)
    if branch_stdout is not None and not branch_match:
        failures.append("BRANCH_MISMATCH")

    tag_commit: str | None = None
    tag_commit_match = expected_tag is None
    if expected_tag is not None and expected_tag_valid:
        tag_ref = f"refs/tags/{expected_tag}^{{commit}}"
        tag_stdout = query("tag", "rev-parse", "--verify", tag_ref)
        if tag_stdout is not None:
            tag_commit = _text(tag_stdout).lower()
            tag_commit_match = bool(
                normalized_expected_commit is not None
                and tag_commit == normalized_expected_commit
            )
            if not tag_commit_match:
                failures.append("TAG_COMMIT_MISMATCH")

    tracked_stdout = query(
        "tracked_diff", "diff", "--no-ext-diff", "--name-only", "-z", "--"
    )
    tracked_paths = None if tracked_stdout is None else _nul_paths(tracked_stdout)
    tracked_diff_count = None if tracked_paths is None else len(tracked_paths)
    if tracked_diff_count:
        failures.append("TRACKED_WORKTREE_DIRTY")

    index_stdout = query(
        "index_diff",
        "diff",
        "--cached",
        "--no-ext-diff",
        "--name-only",
        "-z",
        "--",
    )
    index_paths = None if index_stdout is None else _nul_paths(index_stdout)
    index_diff_count = None if index_paths is None else len(index_paths)
    if index_diff_count:
        failures.append("INDEX_DIRTY")

    untracked_stdout = query(
        "untracked", "ls-files", "--others", "--exclude-standard", "-z", "--"
    )
    untracked_paths = (
        None if untracked_stdout is None else _nul_paths(untracked_stdout)
    )
    untracked_file_count = None if untracked_paths is None else len(untracked_paths)
    if untracked_file_count:
        failures.append("UNTRACKED_FILES_PRESENT")

    status_stdout = query(
        "porcelain_status",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    porcelain_status_clean = status_stdout == b"" if status_stdout is not None else False
    if (
        status_stdout is not None
        and not porcelain_status_clean
        and not any(
            name in failures
            for name in (
                "TRACKED_WORKTREE_DIRTY",
                "INDEX_DIRTY",
                "UNTRACKED_FILES_PRESENT",
            )
        )
    ):
        failures.append("UNCLASSIFIED_PORCELAIN_STATUS_DIRTY")

    # Preserve deterministic classification ordering while eliminating any
    # duplicate query/input classification.
    failures = list(dict.fromkeys(failures))
    report: dict[str, Any] = {
        "RUNTIME_GIT_GATE_PASS": not failures,
        "branch": branch,
        "branch_match": branch_match,
        "checkpoint": str(checkpoint),
        "command_failure_count": len(command_failures),
        "command_failures": command_failures,
        "expected_branch": expected_branch,
        "expected_commit": expected_commit,
        "expected_tag": expected_tag,
        "failure_classification_count": len(failures),
        "failure_classifications": failures,
        "head_commit": head_commit,
        "head_commit_match": head_commit_match,
        "index_diff_count": index_diff_count,
        "index_diff_paths": index_paths,
        "porcelain_status_clean": porcelain_status_clean,
        "repository_input": str(repository),
        "repository_path": str(repository_path),
        "repository_top_level": top_level,
        "repository_top_level_match": top_level_match,
        "tag_commit": tag_commit,
        "tag_commit_match": tag_commit_match,
        "tracked_diff_count": tracked_diff_count,
        "tracked_diff_paths": tracked_paths,
        "untracked_file_count": untracked_file_count,
        "untracked_paths": untracked_paths,
    }
    if failures:
        raise RuntimeGitGateError(report)
    return report


__all__ = ["RuntimeGitGateError", "verify_runtime_git_gate"]
