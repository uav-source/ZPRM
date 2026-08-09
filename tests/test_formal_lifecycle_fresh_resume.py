from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = Path(__file__).resolve().parents[1]
WORKER_RELATIVE = Path(
    "scripts/run_version_agnostic_formal_lifecycle_fixture.py"
)
MANIFEST_RELATIVE = Path(
    "configs/zero_perturbation/version_agnostic_lifecycle_test.json"
)
STABLE_RUNTIME_PATHS = (
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


@dataclass(frozen=True)
class QualificationSubprocessRepository:
    repository: Path
    runtime_root: Path
    manifest_path: Path
    branch: str
    tag: str
    run_id: str
    environment: dict[str, str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in items:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = child
        return value

    result = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )
    assert type(result) is dict
    return result


def git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _copy_qualification_sources(repository: Path) -> None:
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(
        SOURCE_REPOSITORY / "src",
        repository / "src",
        ignore=ignored,
    )
    (repository / "scripts").mkdir(parents=True)
    shutil.copy2(
        SOURCE_REPOSITORY / WORKER_RELATIVE,
        repository / WORKER_RELATIVE,
    )
    parameter_relative = Path(
        "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
    )
    (repository / parameter_relative.parent).mkdir(parents=True)
    shutil.copy2(
        SOURCE_REPOSITORY / parameter_relative,
        repository / parameter_relative,
    )
    pcl_relative = Path("bin/pcl_point_to_plane_cli")
    (repository / pcl_relative.parent).mkdir(parents=True)
    shutil.copy2(
        SOURCE_REPOSITORY / pcl_relative,
        repository / pcl_relative,
    )


def _render_manifest_from_copied_provider(
    *,
    context_name: str,
    repository: Path,
    runtime_root: Path,
    branch: str,
    tag: str,
    run_id: str,
    snapshot_delay_seconds: float,
    trial_delay_seconds: float,
) -> dict[str, Any]:
    code = r"""
import json
import sys
from pathlib import Path

repository, runtime_root, branch, tag, run_id, context_name, snapshot_delay, trial_delay = sys.argv[1:]
sys.path.insert(0, str(Path(repository) / "src"))
from phase_a_harness.runtime_lifecycle_fixture import (
    build_qualification_manifest,
)

value = build_qualification_manifest(
    context_name,
    repository_root=Path(repository),
    expected_commit="0" * 40,
    expected_branch=branch,
    expected_tag=tag,
    run_id=run_id,
    workers=1,
    runtime_root=Path(runtime_root),
    snapshot_commit_delay_seconds=float(snapshot_delay),
    trial_commit_delay_seconds=float(trial_delay),
)
print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(repository),
            str(runtime_root),
            branch,
            tag,
            run_id,
            context_name,
            str(snapshot_delay_seconds),
            str(trial_delay_seconds),
        ],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    assert type(value) is dict
    return value


def prepare_qualification_repository(
    tmp_path: Path,
    *,
    label: str,
    context_name: str = "context_a",
    snapshot_delay_seconds: float = 0.0,
    trial_delay_seconds: float = 0.0,
) -> QualificationSubprocessRepository:
    """Create a committed/tagged repository around the real qualification code."""

    repository = (tmp_path / f"repository_{label}").resolve()
    runtime_root = (tmp_path / f"runtime_{label}").resolve()
    repository.mkdir()
    _copy_qualification_sources(repository)
    assert (
        file_sha256(
            repository
            / "src/phase_a_harness/runtime_lifecycle_fixture.py"
        )
        == file_sha256(
            SOURCE_REPOSITORY
            / "src/phase_a_harness/runtime_lifecycle_fixture.py"
        )
    )

    branch = f"qualification/{label}"
    tag = f"archive/qualification-{label}"
    run_id = f"version-agnostic-lifecycle-{label}"
    manifest = _render_manifest_from_copied_provider(
        context_name=context_name,
        repository=repository,
        runtime_root=runtime_root,
        branch=branch,
        tag=tag,
        run_id=run_id,
        snapshot_delay_seconds=snapshot_delay_seconds,
        trial_delay_seconds=trial_delay_seconds,
    )
    manifest_path = repository / MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (repository / ".gitignore").write_text(
        "__pycache__/\n*.py[cod]\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["git", "init"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    git_output(repository, "config", "user.name", "Lifecycle Test")
    git_output(
        repository,
        "config",
        "user.email",
        "lifecycle-test@example.invalid",
    )
    git_output(repository, "checkout", "-b", branch)
    git_output(
        repository,
        "add",
        "--",
        ".gitignore",
        "src",
        WORKER_RELATIVE.as_posix(),
        "frozen_assets",
        "bin",
        MANIFEST_RELATIVE.as_posix(),
    )
    git_output(
        repository,
        "commit",
        "-m",
        "test: freeze qualification lifecycle fixture",
    )
    git_output(
        repository,
        "tag",
        "-a",
        tag,
        "-m",
        "qualification lifecycle fixture",
    )
    assert git_output(repository, "status", "--porcelain=v1") == ""
    assert not runtime_root.exists()

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "MAMBA_ROOT_PREFIX": (
                "/home/lj/.local/share/degen-lio-micromamba"
            ),
            "MPLCONFIGDIR": str((tmp_path / f"mpl_{label}").resolve()),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return QualificationSubprocessRepository(
        repository=repository,
        runtime_root=runtime_root,
        manifest_path=manifest_path,
        branch=branch,
        tag=tag,
        run_id=run_id,
        environment=environment,
    )


def qualification_worker_command(
    fixture: QualificationSubprocessRepository,
    *,
    mode: str,
    invocation_id: str,
) -> list[str]:
    return [
        sys.executable,
        WORKER_RELATIVE.as_posix(),
        "--manifest",
        fixture.manifest_path.relative_to(fixture.repository).as_posix(),
        "--run-id",
        fixture.run_id,
        "--runtime-root",
        str(fixture.runtime_root),
        "--workers",
        "1",
        "--mode",
        mode,
        "--invocation-id",
        invocation_id,
    ]


def parse_worker_stdout(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{"):
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token: {token}")
                ),
            )
            assert type(value) is dict
            return value
    raise AssertionError(f"worker emitted no JSON object: {stdout[-1000:]}")


def run_qualification_worker(
    fixture: QualificationSubprocessRepository,
    *,
    mode: str,
    invocation_id: str,
    timeout: float = 90.0,
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    completed = subprocess.run(
        qualification_worker_command(
            fixture,
            mode=mode,
            invocation_id=invocation_id,
        ),
        cwd=fixture.repository,
        env=fixture.environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, (
        f"qualification worker failed: rc={completed.returncode}\n"
        f"stdout={completed.stdout[-2000:]}\n"
        f"stderr={completed.stderr[-2000:]}"
    )
    return parse_worker_stdout(completed.stdout), completed


def runtime_inventory(
    root: Path,
    relative_paths: tuple[str, ...] = STABLE_RUNTIME_PATHS,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        if path.is_file() and not path.is_symlink():
            result[relative] = file_sha256(path)
        elif path.is_dir() and not path.is_symlink():
            for child in sorted(path.rglob("*")):
                if child.is_file() and not child.is_symlink():
                    result[child.relative_to(root).as_posix()] = file_sha256(
                        child
                    )
    return result


def test_generic_lifecycle_fresh_resume_has_zero_reexecution_or_checksum_change(
    tmp_path: Path,
) -> None:
    fixture = prepare_qualification_repository(
        tmp_path,
        label="fresh-resume",
    )

    fresh, _fresh_process = run_qualification_worker(
        fixture,
        mode="fresh",
        invocation_id="fresh",
    )
    before = runtime_inventory(fixture.runtime_root)
    resume, _resume_process = run_qualification_worker(
        fixture,
        mode="resume",
        invocation_id="resume",
    )
    after = runtime_inventory(fixture.runtime_root)

    fresh_lifecycle = fresh["lifecycle"]
    resume_lifecycle = resume["lifecycle"]
    fresh_invocation = fresh_lifecycle["invocation_report"]
    resume_invocation = resume_lifecycle["invocation_report"]
    assert fresh_invocation["generated_snapshot_count_this_invocation"] == 3
    assert fresh_invocation["backend_execution_count_this_invocation"] == 6
    assert resume_invocation["generated_snapshot_count_this_invocation"] == 0
    assert resume_invocation["backend_execution_count_this_invocation"] == 0
    assert resume_invocation["resumed_snapshot_count_this_invocation"] == 3
    assert resume_invocation["resume_skipped_valid_result_count"] == 6
    assert resume_invocation["completed_snapshot_count"] == 3
    assert resume_invocation["completed_trial_count"] == 6
    assert resume_invocation["native_trial_count"] == 0
    assert resume_invocation["all_git_gates_pass"] is True
    assert before == after
    assert {
        "snapshot_lock.json",
        "formal_command.log",
        "formal_command.log.sha256",
        "run_manifest.json",
    }.issubset(before)
    assert fresh_lifecycle["snapshot_lock_sha256"] == (
        resume_lifecycle["snapshot_lock_sha256"]
    )
    assert fresh_lifecycle["raw_manifest_sha256"] == (
        resume_lifecycle["raw_manifest_sha256"]
    )
    assert fresh_lifecycle["run_manifest"] == resume_lifecycle["run_manifest"]
    assert fresh["postrun"]["primary"] == resume["postrun"]["primary"]
    assert fresh["postrun"]["independent"] == resume["postrun"]["independent"]
    assert fresh["postrun"]["difference"] == resume["postrun"]["difference"]
    assert fresh["postrun"]["artifact_sha256"] == (
        resume["postrun"]["artifact_sha256"]
    )
    assert resume["postrun"]["publisher_inventory"][
        "publisher_inventory_pass"
    ] is True
    assert resume["postrun"]["artifact_verification"][
        "FIXTURE_ARTIFACT_VERIFICATION_PASS"
    ] is True
    assert fresh["postrun"]["all_git_gates_pass"] is True
    assert resume["postrun"]["all_git_gates_pass"] is True

    repeated, _repeated_process = run_qualification_worker(
        fixture,
        mode="resume",
        invocation_id="resume",
    )
    assert repeated["lifecycle"]["invocation_report"] == resume_invocation
    assert repeated["postrun"]["artifact_sha256"] == (
        resume["postrun"]["artifact_sha256"]
    )
    assert runtime_inventory(fixture.runtime_root) == after
    assert git_output(fixture.repository, "status", "--porcelain=v1") == ""


def test_invocation_id_path_escape_is_rejected_before_runtime_creation(
    tmp_path: Path,
) -> None:
    fixture = prepare_qualification_repository(
        tmp_path,
        label="unsafe-invocation-id",
    )
    escaped = fixture.runtime_root.parent / "escaped.json"
    completed = subprocess.run(
        qualification_worker_command(
            fixture,
            mode="fresh",
            invocation_id="../../escaped",
        ),
        cwd=fixture.repository,
        env=fixture.environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "invocation_id" in completed.stderr
    assert not fixture.runtime_root.exists()
    assert not escaped.exists()
    assert git_output(fixture.repository, "status", "--porcelain=v1") == ""
