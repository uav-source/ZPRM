"""Fail-closed external runtime-path policy for mutable experiment state.

The standalone repository is an immutable execution input.  Snapshot caches,
locks, raw results, event logs, analysis work products, and publisher staging
therefore live below a dedicated external runtime root.  This module performs
path validation only: it does not create directories, inspect scientific
assets, construct an RNG, or execute a backend.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RUNTIME_ROOT = Path("/home/lj/zero_perturbation_runtime")
DEFAULT_SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
DEFAULT_RUNTIME_ARCHIVE_ROOT = Path("/home/lj/zero_perturbation_runtime_archive")
DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEVELOPMENT_RAW_RESULT_ROOTS = (
    DEFAULT_REPOSITORY_ROOT / "results/phase_b_signal_v1/raw_results",
    DEFAULT_REPOSITORY_ROOT / "results/full_synthetic_development_v1/raw_results",
)

POLICY_SCHEMA = "zero_perturbation_external_runtime_path_policy_v1"
RUN_KINDS = frozenset({"qualification", "formal", "fixture"})
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")

MUTABLE_PATH_NAMES = (
    "snapshot_cache",
    "snapshot_lock",
    "raw_results",
    "attempt_events",
    "backend_temporary",
    "primary_analysis",
    "independent_verification",
    "publisher_staging",
    "artifact_staging",
    "temporary_inventory",
)
_DIRECTORY_PATH_NAMES = frozenset(MUTABLE_PATH_NAMES) - {"snapshot_lock"}
_DEFAULT_RELATIVE_PATHS = {
    "snapshot_cache": Path("snapshot_cache"),
    "snapshot_lock": Path("snapshot_lock.json"),
    "raw_results": Path("raw_results"),
    "attempt_events": Path("attempt_events"),
    "backend_temporary": Path("backend_tmp"),
    "primary_analysis": Path("analysis/primary"),
    "independent_verification": Path("analysis/independent"),
    "publisher_staging": Path("publisher_staging"),
    "artifact_staging": Path("artifact_staging"),
    "temporary_inventory": Path("working_inventory"),
}


class RuntimePathPolicyError(ValueError):
    """Raised before any write when a runtime path is outside the contract."""


@dataclass(frozen=True)
class RuntimePathLayout:
    """Canonical locations for all mutable state owned by one run ID."""

    runtime_root: Path
    run_kind: str
    run_id: str
    run_root: Path
    snapshot_cache: Path
    snapshot_lock: Path
    raw_results: Path
    attempt_events: Path
    backend_temporary: Path
    primary_analysis: Path
    independent_verification: Path
    publisher_staging: Path
    artifact_staging: Path
    temporary_inventory: Path

    def mutable_paths(self) -> dict[str, Path]:
        return {name: getattr(self, name) for name in MUTABLE_PATH_NAMES}

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "run_root": str(self.run_root),
            "runtime_root": str(self.runtime_root),
            **{name: str(path) for name, path in self.mutable_paths().items()},
        }


@dataclass(frozen=True)
class RuntimePathQualification:
    """One validated layout and its JSON-serializable security audit."""

    layout: RuntimePathLayout
    audit: Mapping[str, Any]


def _trusted_canonical(path: os.PathLike[str] | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _symlink_components(path: Path) -> tuple[str, ...]:
    current = Path(path.anchor)
    result: list[str] = []
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            result.append(str(current))
    return tuple(result)


def _canonical_runtime_path(
    value: os.PathLike[str] | str, *, label: str
) -> Path:
    if isinstance(value, str) and not value.strip():
        raise RuntimePathPolicyError(f"{label} must not be empty")
    try:
        candidate = Path(value)
    except TypeError as error:
        raise RuntimePathPolicyError(f"{label} must be a path") from error
    if not candidate.is_absolute():
        raise RuntimePathPolicyError(f"{label} must be absolute")
    if candidate == Path(candidate.anchor):
        raise RuntimePathPolicyError(f"{label} must not be the filesystem root")
    symlinks = _symlink_components(candidate)
    if symlinks:
        raise RuntimePathPolicyError(
            f"{label} contains symlink component: {symlinks[0]}"
        )
    canonical = candidate.resolve(strict=False)
    if candidate != canonical:
        raise RuntimePathPolicyError(f"{label} must already be canonical")
    return canonical


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _overlap(first: Path, second: Path) -> bool:
    return _inside(first, second) or _inside(second, first)


def _safe_segment(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_SEGMENT.fullmatch(value):
        raise RuntimePathPolicyError(
            f"{label} must match [A-Za-z0-9][A-Za-z0-9_-]{{0,127}}"
        )
    if ".." in value:
        raise RuntimePathPolicyError(f"{label} must not contain '..'")
    return value


def _forbidden_roots(
    *,
    repository_root: Path,
    source_repository: os.PathLike[str] | str,
    runtime_archive_root: os.PathLike[str] | str,
    development_raw_result_roots: Sequence[os.PathLike[str] | str] | None,
) -> dict[str, Path]:
    development = (
        DEFAULT_DEVELOPMENT_RAW_RESULT_ROOTS
        if development_raw_result_roots is None
        else tuple(development_raw_result_roots)
    )
    result = {
        "repository": repository_root,
        "source_repository": _trusted_canonical(source_repository),
        "runtime_archive": _trusted_canonical(runtime_archive_root),
    }
    for index, path in enumerate(development):
        result[f"development_raw_results_{index}"] = _trusted_canonical(path)
    return result


def _assert_no_forbidden_overlap(
    path: Path, *, label: str, forbidden: Mapping[str, Path]
) -> None:
    for name, protected in forbidden.items():
        if _overlap(path, protected):
            raise RuntimePathPolicyError(
                f"{label} overlaps protected {name}: {protected}"
            )


def _layout_paths(
    *,
    run_root: Path,
    path_overrides: Mapping[str, os.PathLike[str] | str] | None,
) -> dict[str, Path]:
    overrides = {} if path_overrides is None else dict(path_overrides)
    unknown = sorted(set(overrides) - set(MUTABLE_PATH_NAMES))
    if unknown:
        raise RuntimePathPolicyError(f"unknown runtime path override(s): {unknown}")
    paths: dict[str, Path] = {}
    for name in MUTABLE_PATH_NAMES:
        proposed = overrides.get(name, run_root / _DEFAULT_RELATIVE_PATHS[name])
        candidate = _canonical_runtime_path(proposed, label=name)
        if candidate == run_root or not _inside(candidate, run_root):
            raise RuntimePathPolicyError(f"{name} escapes the canonical run root")
        paths[name] = candidate
    for index, name in enumerate(MUTABLE_PATH_NAMES):
        for other in MUTABLE_PATH_NAMES[index + 1 :]:
            if _overlap(paths[name], paths[other]):
                raise RuntimePathPolicyError(
                    f"mutable runtime paths overlap: {name} and {other}"
                )
    return paths


def _assert_existing_path_types(paths: Mapping[str, Path]) -> None:
    for name, path in paths.items():
        if not path.exists():
            continue
        if name in _DIRECTORY_PATH_NAMES and not path.is_dir():
            raise RuntimePathPolicyError(f"{name} exists but is not a directory")
        if name == "snapshot_lock" and not path.is_file():
            raise RuntimePathPolicyError("snapshot_lock exists but is not a file")


def _assert_directory_ancestors(path: Path, *, run_root: Path, label: str) -> None:
    for parent in path.parents:
        if parent == run_root:
            break
        if not _inside(parent, run_root):
            raise RuntimePathPolicyError(f"{label} parent escaped run root")
        if parent.exists() and not parent.is_dir():
            raise RuntimePathPolicyError(
                f"{label} has a non-directory parent: {parent}"
            )


def _registered_run_audit(
    *,
    run_id: str,
    run_root: Path,
    registered_run_directories: Mapping[str, os.PathLike[str] | str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for registered_id, raw_path in sorted(
        (registered_run_directories or {}).items(), key=lambda item: str(item[0])
    ):
        other_id = _safe_segment(registered_id, label="registered run ID")
        other_path = _canonical_runtime_path(
            raw_path, label=f"registered run directory {other_id}"
        )
        overlaps = _overlap(run_root, other_path)
        same_identity = other_id == run_id
        if same_identity and other_path != run_root:
            raise RuntimePathPolicyError(
                "the same run ID is registered to a different runtime directory"
            )
        if not same_identity and overlaps:
            raise RuntimePathPolicyError(
                f"run directory overlaps different run ID {other_id}"
            )
        rows.append(
            {
                "registered_run_id": other_id,
                "registered_run_root": str(other_path),
                "same_run_id": same_identity,
                "overlap": overlaps,
            }
        )
    return rows


def qualify_runtime_paths(
    run_id: str,
    *,
    runtime_root: os.PathLike[str] | str = DEFAULT_RUNTIME_ROOT,
    repository_root: os.PathLike[str] | str = DEFAULT_REPOSITORY_ROOT,
    run_kind: str = "qualification",
    source_repository: os.PathLike[str] | str = DEFAULT_SOURCE_REPOSITORY,
    runtime_archive_root: os.PathLike[str] | str = DEFAULT_RUNTIME_ARCHIVE_ROOT,
    development_raw_result_roots: Sequence[os.PathLike[str] | str] | None = None,
    path_overrides: Mapping[str, os.PathLike[str] | str] | None = None,
    registered_run_directories: Mapping[str, os.PathLike[str] | str] | None = None,
    resume: bool = False,
) -> RuntimePathQualification:
    """Validate and return the complete external layout for one run.

    ``resume`` authorizes reuse of the *same* canonical run directory.  It does
    not weaken any containment, symlink, protected-root, or cross-run check.
    The function is deliberately side-effect free.
    """

    if type(resume) is not bool:
        raise RuntimePathPolicyError("resume must be bool")
    identity = _safe_segment(run_id, label="run ID")
    kind = _safe_segment(run_kind, label="run kind")
    if kind not in RUN_KINDS:
        raise RuntimePathPolicyError(f"unsupported run kind: {kind}")
    repository = _trusted_canonical(repository_root)
    root = _canonical_runtime_path(runtime_root, label="runtime root")
    forbidden = _forbidden_roots(
        repository_root=repository,
        source_repository=source_repository,
        runtime_archive_root=runtime_archive_root,
        development_raw_result_roots=development_raw_result_roots,
    )
    _assert_no_forbidden_overlap(root, label="runtime root", forbidden=forbidden)

    if root.exists() and not root.is_dir():
        raise RuntimePathPolicyError("runtime root exists but is not a directory")
    run_kind_root = _canonical_runtime_path(root / kind, label="run-kind directory")
    if run_kind_root.exists() and not run_kind_root.is_dir():
        raise RuntimePathPolicyError(
            "run-kind directory exists but is not a directory"
        )

    run_root = _canonical_runtime_path(
        run_kind_root / identity, label="run directory"
    )
    if not _inside(run_root, root):
        raise RuntimePathPolicyError("run directory escapes runtime root")
    _assert_no_forbidden_overlap(run_root, label="run directory", forbidden=forbidden)
    if run_root.exists() and not run_root.is_dir():
        raise RuntimePathPolicyError("run directory exists but is not a directory")
    if run_root.exists() and not resume:
        raise RuntimePathPolicyError(
            "run directory already exists; explicit resume is required"
        )

    paths = _layout_paths(run_root=run_root, path_overrides=path_overrides)
    for name, path in paths.items():
        _assert_no_forbidden_overlap(path, label=name, forbidden=forbidden)
        _assert_directory_ancestors(path, run_root=run_root, label=name)
    _assert_existing_path_types(paths)
    registered = _registered_run_audit(
        run_id=identity,
        run_root=run_root,
        registered_run_directories=registered_run_directories,
    )

    layout = RuntimePathLayout(
        runtime_root=root,
        run_kind=kind,
        run_id=identity,
        run_root=run_root,
        **paths,
    )
    path_rows = {
        name: {
            "absolute": path.is_absolute(),
            "canonical": path == path.resolve(strict=False),
            "inside_runtime_root": _inside(path, root),
            "inside_run_root": _inside(path, run_root),
            "path": str(path),
            "symlink_component_count": len(_symlink_components(path)),
        }
        for name, path in {"run_root": run_root, **paths}.items()
    }
    audit: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA,
        "RUNTIME_PATH_POLICY_PASS": True,
        "RUNTIME_ROOT_OUTSIDE_REPOSITORY": not _overlap(root, repository),
        "SNAPSHOT_CACHE_OUTSIDE_REPOSITORY": not _overlap(
            paths["snapshot_cache"], repository
        ),
        "SNAPSHOT_LOCK_OUTSIDE_REPOSITORY": not _overlap(
            paths["snapshot_lock"], repository
        ),
        "RAW_RESULTS_OUTSIDE_REPOSITORY": not _overlap(
            paths["raw_results"], repository
        ),
        "ARTIFACT_STAGE_OUTSIDE_REPOSITORY": not _overlap(
            paths["artifact_staging"], repository
        ),
        "all_paths_absolute": all(row["absolute"] for row in path_rows.values()),
        "all_paths_canonical": all(row["canonical"] for row in path_rows.values()),
        "all_paths_inside_run_root": all(
            row["inside_run_root"] for row in path_rows.values()
        ),
        "all_paths_without_symlink_components": all(
            row["symlink_component_count"] == 0 for row in path_rows.values()
        ),
        "different_run_directory_overlap_count": sum(
            bool(row["overlap"] and not row["same_run_id"]) for row in registered
        ),
        "forbidden_roots": {name: str(path) for name, path in forbidden.items()},
        "layout": layout.as_dict(),
        "path_checks": path_rows,
        "registered_run_directories": registered,
        "resume": resume,
        "run_id": identity,
        "run_kind": kind,
        "runtime_root": str(root),
        "symlink_component_count": sum(
            int(row["symlink_component_count"]) for row in path_rows.values()
        ),
    }
    required = (
        "RUNTIME_ROOT_OUTSIDE_REPOSITORY",
        "SNAPSHOT_CACHE_OUTSIDE_REPOSITORY",
        "SNAPSHOT_LOCK_OUTSIDE_REPOSITORY",
        "RAW_RESULTS_OUTSIDE_REPOSITORY",
        "ARTIFACT_STAGE_OUTSIDE_REPOSITORY",
        "all_paths_absolute",
        "all_paths_canonical",
        "all_paths_inside_run_root",
        "all_paths_without_symlink_components",
    )
    if not all(audit[name] is True for name in required):
        raise RuntimePathPolicyError("runtime path audit failed closed")
    return RuntimePathQualification(layout=layout, audit=audit)


def build_runtime_path_layout(*args: Any, **kwargs: Any) -> RuntimePathLayout:
    """Return only the validated layout for callers that store audit elsewhere."""

    return qualify_runtime_paths(*args, **kwargs).layout


def audit_runtime_paths(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return only the JSON-serializable security audit."""

    return dict(qualify_runtime_paths(*args, **kwargs).audit)


__all__ = [
    "DEFAULT_DEVELOPMENT_RAW_RESULT_ROOTS",
    "DEFAULT_REPOSITORY_ROOT",
    "DEFAULT_RUNTIME_ARCHIVE_ROOT",
    "DEFAULT_RUNTIME_ROOT",
    "DEFAULT_SOURCE_REPOSITORY",
    "MUTABLE_PATH_NAMES",
    "POLICY_SCHEMA",
    "RUN_KINDS",
    "RuntimePathLayout",
    "RuntimePathPolicyError",
    "RuntimePathQualification",
    "audit_runtime_paths",
    "build_runtime_path_layout",
    "qualify_runtime_paths",
]
