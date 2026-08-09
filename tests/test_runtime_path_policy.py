from __future__ import annotations

import json
from pathlib import Path

import pytest

from phase_a_harness.runtime_path_policy import (
    DEFAULT_RUNTIME_ROOT,
    MUTABLE_PATH_NAMES,
    POLICY_SCHEMA,
    RuntimePathPolicyError,
    audit_runtime_paths,
    build_runtime_path_layout,
    qualify_runtime_paths,
)


def _valid(tmp_path: Path, **overrides):
    repository = tmp_path / "standalone_repository"
    runtime = tmp_path / "external_runtime"
    source = tmp_path / "source_repository"
    archive = tmp_path / "runtime_archive"
    development = tmp_path / "development_results" / "raw_results"
    arguments = {
        "runtime_root": runtime,
        "repository_root": repository,
        "source_repository": source,
        "runtime_archive_root": archive,
        "development_raw_result_roots": (development,),
    }
    arguments.update(overrides)
    return qualify_runtime_paths("runtime_lifecycle_v1", **arguments)


def test_default_root_and_qualification_layout_are_exact() -> None:
    assert DEFAULT_RUNTIME_ROOT == Path("/home/lj/zero_perturbation_runtime")
    # Layout identity must remain testable while the one permitted
    # qualification directory exists; explicit resume preserves the strict
    # fresh-run collision check exercised separately below.
    layout = build_runtime_path_layout("runtime_lifecycle_v1", resume=True)
    assert layout.run_root == (
        DEFAULT_RUNTIME_ROOT / "qualification/runtime_lifecycle_v1"
    )
    assert layout.snapshot_cache == layout.run_root / "snapshot_cache"
    assert layout.snapshot_lock == layout.run_root / "snapshot_lock.json"
    assert layout.raw_results == layout.run_root / "raw_results"
    assert layout.artifact_staging == layout.run_root / "artifact_staging"


def test_valid_layout_has_complete_json_serializable_audit(tmp_path: Path) -> None:
    qualified = _valid(tmp_path)
    assert set(qualified.layout.mutable_paths()) == set(MUTABLE_PATH_NAMES)
    audit = qualified.audit
    assert audit["schema_version"] == POLICY_SCHEMA
    assert audit["RUNTIME_PATH_POLICY_PASS"] is True
    assert audit["RUNTIME_ROOT_OUTSIDE_REPOSITORY"] is True
    assert audit["SNAPSHOT_CACHE_OUTSIDE_REPOSITORY"] is True
    assert audit["SNAPSHOT_LOCK_OUTSIDE_REPOSITORY"] is True
    assert audit["RAW_RESULTS_OUTSIDE_REPOSITORY"] is True
    assert audit["ARTIFACT_STAGE_OUTSIDE_REPOSITORY"] is True
    assert audit["all_paths_without_symlink_components"] is True
    assert audit["symlink_component_count"] == 0
    json.dumps(audit, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("value", ["", "relative/path", ".", "..", "/"])
def test_empty_relative_or_root_runtime_path_is_rejected(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(RuntimePathPolicyError):
        _valid(tmp_path, runtime_root=value)


def test_noncanonical_runtime_path_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "outer" / ".." / "external_runtime"
    with pytest.raises(RuntimePathPolicyError, match="canonical"):
        _valid(tmp_path, runtime_root=candidate)


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "a..b", "../escape", "a/b", r"a\b", "a b", "x;touch"],
)
def test_run_id_injection_is_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(RuntimePathPolicyError, match="run ID"):
        qualify_runtime_paths(
            run_id,
            runtime_root=tmp_path / "runtime",
            repository_root=tmp_path / "repository",
            source_repository=tmp_path / "source",
            runtime_archive_root=tmp_path / "archive",
            development_raw_result_roots=(),
        )


def test_runtime_root_inside_repository_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    with pytest.raises(RuntimePathPolicyError, match="repository"):
        _valid(
            tmp_path,
            repository_root=repository,
            runtime_root=repository / "runtime",
        )


def test_runtime_root_containing_repository_is_rejected(tmp_path: Path) -> None:
    container = tmp_path / "container"
    with pytest.raises(RuntimePathPolicyError, match="repository"):
        _valid(
            tmp_path,
            repository_root=container / "repository",
            runtime_root=container,
        )


@pytest.mark.parametrize("protected", ["source", "archive", "development"])
def test_runtime_root_protected_overlap_is_rejected(
    tmp_path: Path, protected: str
) -> None:
    roots = {
        "source": tmp_path / "source_repository",
        "archive": tmp_path / "runtime_archive",
        "development": tmp_path / "development_results" / "raw_results",
    }
    with pytest.raises(RuntimePathPolicyError, match="overlaps protected"):
        _valid(tmp_path, runtime_root=roots[protected] / "nested")


def test_runtime_root_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "external_target"
    target.mkdir()
    link = tmp_path / "runtime_link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimePathPolicyError, match="symlink"):
        _valid(tmp_path, runtime_root=link)


def test_symlink_escape_to_repository_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    link = tmp_path / "escape"
    link.symlink_to(repository, target_is_directory=True)
    with pytest.raises(RuntimePathPolicyError, match="symlink"):
        _valid(tmp_path, runtime_root=link / "runtime")


@pytest.mark.parametrize(
    "name",
    ["snapshot_cache", "snapshot_lock", "raw_results", "artifact_staging"],
)
def test_mutable_path_escape_is_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(RuntimePathPolicyError, match="escapes"):
        _valid(
            tmp_path,
            path_overrides={name: tmp_path / f"escaped_{name}"},
        )


def test_mutable_path_symlink_is_rejected(tmp_path: Path) -> None:
    runtime = tmp_path / "external_runtime"
    run_root = runtime / "qualification/runtime_lifecycle_v1"
    run_root.mkdir(parents=True)
    target = tmp_path / "outside"
    target.mkdir()
    link = run_root / "snapshot_cache"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimePathPolicyError, match="symlink"):
        _valid(tmp_path, runtime_root=runtime, resume=True)


def test_unknown_or_overlapping_mutable_path_is_rejected(tmp_path: Path) -> None:
    run_root = tmp_path / "external_runtime/qualification/runtime_lifecycle_v1"
    with pytest.raises(RuntimePathPolicyError, match="unknown"):
        _valid(tmp_path, path_overrides={"mystery": run_root / "mystery"})
    shared = run_root / "shared"
    with pytest.raises(RuntimePathPolicyError, match="overlap"):
        _valid(
            tmp_path,
            path_overrides={
                "snapshot_cache": shared,
                "raw_results": shared / "raw_results",
            },
        )


def test_different_run_directories_must_not_overlap(tmp_path: Path) -> None:
    runtime = tmp_path / "external_runtime"
    candidate = runtime / "qualification/runtime_lifecycle_v1"
    with pytest.raises(RuntimePathPolicyError, match="different run ID"):
        _valid(
            tmp_path,
            runtime_root=runtime,
            registered_run_directories={"other_run": candidate.parent},
        )


def test_same_run_id_cannot_move_to_a_different_directory(tmp_path: Path) -> None:
    with pytest.raises(RuntimePathPolicyError, match="same run ID"):
        _valid(
            tmp_path,
            registered_run_directories={
                "runtime_lifecycle_v1": tmp_path / "another_runtime/run"
            },
        )


def test_existing_run_requires_explicit_resume(tmp_path: Path) -> None:
    runtime = tmp_path / "external_runtime"
    run_root = runtime / "qualification/runtime_lifecycle_v1"
    run_root.mkdir(parents=True)
    with pytest.raises(RuntimePathPolicyError, match="explicit resume"):
        _valid(tmp_path, runtime_root=runtime)
    resumed = _valid(
        tmp_path,
        runtime_root=runtime,
        resume=True,
        registered_run_directories={"runtime_lifecycle_v1": run_root},
    )
    assert resumed.layout.run_root == run_root
    assert resumed.audit["resume"] is True


def test_existing_runtime_leaf_type_is_checked(tmp_path: Path) -> None:
    runtime = tmp_path / "external_runtime"
    run_root = runtime / "qualification/runtime_lifecycle_v1"
    run_root.mkdir(parents=True)
    (run_root / "raw_results").write_text("not a directory", encoding="utf-8")
    with pytest.raises(RuntimePathPolicyError, match="raw_results"):
        _valid(tmp_path, runtime_root=runtime, resume=True)


def test_existing_runtime_root_must_be_a_directory(tmp_path: Path) -> None:
    runtime = tmp_path / "external_runtime"
    runtime.write_text("not a directory", encoding="utf-8")
    with pytest.raises(RuntimePathPolicyError, match="runtime root"):
        _valid(tmp_path, runtime_root=runtime)


def test_existing_mutable_parent_must_be_a_directory(tmp_path: Path) -> None:
    runtime = tmp_path / "external_runtime"
    run_root = runtime / "qualification/runtime_lifecycle_v1"
    run_root.mkdir(parents=True)
    (run_root / "analysis").write_text("not a directory", encoding="utf-8")
    with pytest.raises(RuntimePathPolicyError, match="non-directory parent"):
        _valid(tmp_path, runtime_root=runtime, resume=True)


def test_audit_only_api_matches_layout_api(tmp_path: Path) -> None:
    arguments = {
        "runtime_root": tmp_path / "external_runtime",
        "repository_root": tmp_path / "repository",
        "source_repository": tmp_path / "source",
        "runtime_archive_root": tmp_path / "archive",
        "development_raw_result_roots": (),
    }
    layout = build_runtime_path_layout("run_a", **arguments)
    audit = audit_runtime_paths("run_a", **arguments)
    assert audit["layout"] == layout.as_dict()
    assert all(
        Path(value).is_absolute()
        for key, value in audit["layout"].items()
        if key not in {"run_id", "run_kind"}
    )
