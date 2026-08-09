"""Immutable, version-agnostic paths for one formal lifecycle.

The lifecycle receives every mutable path explicitly.  This module performs
only lexical/filesystem-policy validation; it never creates a directory or
opens a scientific payload.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


class FormalLifecyclePathError(ValueError):
    """An explicit lifecycle path is unsafe or internally inconsistent."""


def _canonical_absolute_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, Path):
        raise FormalLifecyclePathError(f"{field} must be pathlib.Path")
    if not value.is_absolute() or value == Path(value.anchor):
        raise FormalLifecyclePathError(
            f"{field} must be a non-root absolute path"
        )
    if any(part in {".", ".."} for part in value.parts):
        raise FormalLifecyclePathError(
            f"{field} must be lexically canonical"
        )
    if value.resolve(strict=False) != value:
        raise FormalLifecyclePathError(f"{field} must already be canonical")
    current = Path(value.anchor)
    for part in value.parts[1:]:
        current /= part
        if current.is_symlink():
            raise FormalLifecyclePathError(
                f"{field} contains a symbolic-link component: {current}"
            )
    return value


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _overlap(left: Path, right: Path) -> bool:
    return _inside(left, right) or _inside(right, left)


@dataclass(frozen=True)
class FormalLifecyclePaths:
    """Complete external mutable layout for one run identity."""

    runtime_root: Path
    snapshot_cache: Path
    snapshot_lock: Path
    raw_results: Path
    raw_manifest: Path
    event_log: Path
    backend_temporary: Path
    analysis: Path
    verification: Path
    publisher_staging: Path
    artifact_staging: Path
    temporary_inventory: Path
    run_manifest: Path
    formal_command_log: Path
    formal_command_sha256: Path
    immutable_run_lock: Path
    single_writer_lease: Path

    def __post_init__(self) -> None:
        values = {
            item.name: _canonical_absolute_path(
                getattr(self, item.name), field=item.name
            )
            for item in fields(self)
        }
        root = values["runtime_root"]
        children = {
            name: path
            for name, path in values.items()
            if name not in {"runtime_root", "single_writer_lease"}
        }
        for name, path in children.items():
            if path == root or not _inside(path, root):
                raise FormalLifecyclePathError(
                    f"{name} must be a strict descendant of runtime_root"
                )
        lease = values["single_writer_lease"]
        if _overlap(lease, root):
            raise FormalLifecyclePathError(
                "single_writer_lease must be external to runtime_root"
            )
        pairs = list(children.items())
        for index, (name, path) in enumerate(pairs):
            for other_name, other_path in pairs[index + 1 :]:
                if _overlap(path, other_path):
                    raise FormalLifecyclePathError(
                        "lifecycle paths overlap: "
                        f"{name}={path} and {other_name}={other_path}"
                    )

    @property
    def run_root(self) -> Path:
        """Compatibility spelling for callers that call the per-run root a run root."""

        return self.runtime_root

    def mutable_paths(self) -> dict[str, Path]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "runtime_root"
        }

    def as_dict(self) -> dict[str, str]:
        return {
            item.name: str(getattr(self, item.name))
            for item in fields(self)
        }


RuntimePaths = FormalLifecyclePaths


__all__ = [
    "FormalLifecyclePathError",
    "FormalLifecyclePaths",
    "RuntimePaths",
]
