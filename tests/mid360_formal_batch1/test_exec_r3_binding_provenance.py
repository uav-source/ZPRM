from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.authorization import (
    formal_registration_authorization as producer,
    formal_registration_authorization_verify as verifier,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def _fixture(root: Path) -> tuple[str, str, dict[str, dict[str, object]]]:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Exec-R3 Fixture")
    code = root / "control/ordinary_name.py"
    code.parent.mkdir()
    code.write_text("EXECUTION = 'A'\n", encoding="utf-8")
    _git(root, "add", "control/ordinary_name.py")
    _git(root, "commit", "-q", "-m", "fixture execution code")
    commit_a = _git(root, "rev-parse", "HEAD")

    release = root / "evidence/release-only.json"
    release.parent.mkdir()
    release.write_text('{"status":"PASS"}\n', encoding="utf-8")
    _git(root, "add", "evidence/release-only.json")
    _git(root, "commit", "-q", "-m", "fixture lock release evidence")
    commit_b = _git(root, "rev-parse", "HEAD")

    def row(
        binding_id: str, path: Path, binding_class: str,
        source: str, role: str,
    ) -> dict[str, object]:
        content = path.read_bytes()
        return {
            "binding_id": binding_id,
            "repository_relative_path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "binding_class": binding_class,
            "verification_source": source,
            "commit_role": role,
        }

    # The names intentionally point the opposite way from the old heuristic:
    # a release binding starts with execution_, while an execution binding does
    # not.  Only explicit class/source/role metadata may select the commit.
    bindings = {
        "ordinary_binding": row(
            "ordinary_binding", code, "EXECUTION_CODE",
            "GIT_BLOB_AT_COMMIT", "EXECUTION_CODE_COMMIT",
        ),
        "execution_control_patch_report": row(
            "execution_control_patch_report", release,
            "LOCK_RELEASE_EVIDENCE", "GIT_BLOB_AT_COMMIT",
            "LOCK_RELEASE_COMMIT",
        ),
    }
    return commit_a, commit_b, bindings


def _select_fixture_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verifier, "EXPECTED_EXECUTION_BINDINGS", {"ordinary_binding"})
    monkeypatch.setattr(
        verifier, "EXPECTED_RELEASE_BINDINGS", {"execution_control_patch_report"}
    )
    monkeypatch.setattr(verifier, "EXPECTED_FROZEN_BINDINGS", set())
    monkeypatch.setattr(verifier, "EXPECTED_ENVIRONMENT_BINDINGS", set())


def _verify_independent(
    root: Path, bindings: dict[str, dict[str, object]],
    commit_a: str, commit_b: str,
) -> None:
    verifier._verify_explicit_binding_provenance(
        root, bindings,
        execution_commit=commit_a,
        lock_release_commit=commit_b,
    )


def test_two_commit_fixture_routes_only_by_explicit_binding_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    commit_a, commit_b, bindings = _fixture(root)
    _select_fixture_registry(monkeypatch)

    # The release-only artifact is absent from A and present in B.
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{commit_a}:evidence/release-only.json"],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0
    assert _git(root, "show", f"{commit_b}:evidence/release-only.json")

    _verify_independent(root, bindings, commit_a, commit_b)
    producer._verify_explicit_binding_provenance(
        root,
        {"execution_code_commit": commit_a, "bindings": bindings},
        lock_release_commit=commit_b,
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_binding", "extra_binding", "missing_metadata",
        "wrong_binding_class", "unknown_binding_class",
        "wrong_verification_source", "wrong_commit_role",
        "wrong_sha", "wrong_bytes", "wrong_execution_commit",
        "wrong_release_commit", "working_tree_only_change",
    ],
)
def test_independent_provenance_dispatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    root = tmp_path / "repo"
    commit_a, commit_b, bindings = _fixture(root)
    _select_fixture_registry(monkeypatch)

    if tamper == "missing_binding":
        bindings.pop("ordinary_binding")
    elif tamper == "extra_binding":
        bindings["extra"] = dict(bindings["ordinary_binding"], binding_id="extra")
    elif tamper == "missing_metadata":
        bindings["ordinary_binding"].pop("commit_role")
    elif tamper == "wrong_binding_class":
        bindings["execution_control_patch_report"].update({
            "binding_class": "EXECUTION_CODE",
            "commit_role": "EXECUTION_CODE_COMMIT",
        })
    elif tamper == "unknown_binding_class":
        bindings["ordinary_binding"]["binding_class"] = "UNKNOWN"
    elif tamper == "wrong_verification_source":
        bindings["ordinary_binding"]["verification_source"] = (
            "INHERITED_EXEC_R2_LOCK_SHA256"
        )
    elif tamper == "wrong_commit_role":
        bindings["ordinary_binding"]["commit_role"] = "LOCK_RELEASE_COMMIT"
    elif tamper == "wrong_sha":
        bindings["ordinary_binding"]["sha256"] = "0" * 64
    elif tamper == "wrong_bytes":
        bindings["ordinary_binding"]["bytes"] = 0
    elif tamper == "wrong_execution_commit":
        commit_a = "0" * 40
    elif tamper == "wrong_release_commit":
        commit_b = commit_a
    else:
        path = root / str(bindings["ordinary_binding"]["repository_relative_path"])
        path.write_text("EXECUTION = 'UNCOMMITTED'\n", encoding="utf-8")

    with pytest.raises(verifier.FormalAuthorizationVerificationError):
        _verify_independent(root, bindings, commit_a, commit_b)
