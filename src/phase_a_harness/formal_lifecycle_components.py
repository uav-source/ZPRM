"""Explicit callable bindings for the version-agnostic formal lifecycle."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .execution_context import ExecutionMode


BACKEND_IDS = ("open3d_point_to_plane", "pcl_point_to_plane")


class FormalLifecycleComponentError(ValueError):
    """A component identity, file digest, or context binding is invalid."""


def _name(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise FormalLifecycleComponentError(
            f"{field} must be a non-empty canonical string"
        )
    return value


def _sha256(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FormalLifecycleComponentError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FormalLifecycleComponentError(
                f"component implementation is not a regular file: {path}"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
    finally:
        os.close(descriptor)


def _implementation_path(value: Any) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise FormalLifecycleComponentError(
            "component implementation_path must be an absolute pathlib.Path"
        )
    if value.resolve(strict=False) != value:
        raise FormalLifecycleComponentError(
            "component implementation_path must already be canonical"
        )
    current = Path(value.anchor)
    for part in value.parts[1:]:
        current /= part
        if current.is_symlink():
            raise FormalLifecycleComponentError(
                f"component implementation path contains a symlink: {current}"
            )
    if not value.is_file():
        raise FormalLifecycleComponentError(
            f"component implementation file is absent: {value}"
        )
    return value


def _callable_identity(value: Callable[..., Any]) -> tuple[str, str]:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module is None or qualname is None:
        target = type(value)
        module = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
    return (
        _name(module, field="component callable module"),
        _name(qualname, field="component callable qualname"),
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ComponentBinding:
    """One direct Python callable bound to one authenticated implementation file."""

    component_id: str
    callable: Callable[..., Any]
    implementation_path: Path
    implementation_sha256: str

    def __post_init__(self) -> None:
        _name(self.component_id, field="component_id")
        if not callable(self.callable):
            raise FormalLifecycleComponentError(
                f"component {self.component_id} is not callable"
            )
        path = _implementation_path(self.implementation_path)
        expected = _sha256(
            self.implementation_sha256,
            field=f"{self.component_id}.implementation_sha256",
        )
        if _file_sha256(path) != expected:
            raise FormalLifecycleComponentError(
                f"component implementation SHA mismatch: {self.component_id}"
            )
        module, _qualname = _callable_identity(self.callable)
        try:
            source = inspect.getsourcefile(self.callable)
        except (OSError, TypeError):
            source = None
        if source is not None:
            source_path = Path(source).resolve(strict=False)
            if source_path != path:
                raise FormalLifecycleComponentError(
                    f"component callable source differs from binding: "
                    f"{self.component_id} ({module})"
                )

    @property
    def callable_module(self) -> str:
        return _callable_identity(self.callable)[0]

    @property
    def callable_qualname(self) -> str:
        return _callable_identity(self.callable)[1]

    def report(self, *, repository_root: Path | None = None) -> dict[str, Any]:
        path = self.implementation_path
        if repository_root is not None:
            root = repository_root.resolve(strict=False)
            if path != root and root not in path.parents:
                raise FormalLifecycleComponentError(
                    f"component escapes repository: {self.component_id}"
                )
            rendered_path = path.relative_to(root).as_posix()
        else:
            rendered_path = str(path)
        return {
            "component_id": self.component_id,
            "callable_module": self.callable_module,
            "callable_qualname": self.callable_qualname,
            "implementation_path": rendered_path,
            "implementation_sha256": self.implementation_sha256,
        }


@dataclass(frozen=True)
class FormalLifecycleComponents:
    """Complete execution and post-run component bundle.

    Context binding fields are intentionally separate from implementation
    bindings.  Context A and B can share the exact callables while a bundle
    constructed for A cannot be accepted with B's manifest or plans.
    """

    bundle_id: str
    bound_mode: ExecutionMode
    bound_contract_id: str
    bound_plan_id: str
    bound_manifest_sha256: str
    bound_run_id: str
    bound_runtime_root: Path
    bound_publisher_inventory_sha256: str
    snapshot_materializer: ComponentBinding
    snapshot_reader: ComponentBinding
    snapshot_validator: ComponentBinding
    trial_validator: ComponentBinding
    snapshot_lock_builder: ComponentBinding
    snapshot_lock_validator: ComponentBinding
    backend_input_builder: ComponentBinding
    common_record_builder: ComponentBinding
    result_validator: ComponentBinding
    resume_result_validator: ComponentBinding
    primary_analyzer: ComponentBinding
    independent_verifier: ComponentBinding
    difference_auditor: ComponentBinding
    publisher: ComponentBinding
    artifact_verifier: ComponentBinding
    git_gate: ComponentBinding
    run_contract_builder: ComponentBinding
    backend_registry: Mapping[str, ComponentBinding]

    def __post_init__(self) -> None:
        _name(self.bundle_id, field="bundle_id")
        if type(self.bound_mode) is not ExecutionMode:
            raise FormalLifecycleComponentError(
                "bound_mode must be ExecutionMode"
            )
        for field in (
            "bound_contract_id",
            "bound_plan_id",
            "bound_run_id",
        ):
            _name(getattr(self, field), field=field)
        for field in (
            "bound_manifest_sha256",
            "bound_publisher_inventory_sha256",
        ):
            _sha256(getattr(self, field), field=field)
        root = self.bound_runtime_root
        if (
            not isinstance(root, Path)
            or not root.is_absolute()
            or root.resolve(strict=False) != root
        ):
            raise FormalLifecycleComponentError(
                "bound_runtime_root must be an explicit canonical absolute Path"
            )
        component_fields = (
            item.name
            for item in fields(self)
            if item.name
            not in {
                "bundle_id",
                "bound_mode",
                "bound_contract_id",
                "bound_plan_id",
                "bound_manifest_sha256",
                "bound_run_id",
                "bound_runtime_root",
                "bound_publisher_inventory_sha256",
                "backend_registry",
            }
        )
        for name in component_fields:
            if type(getattr(self, name)) is not ComponentBinding:
                raise FormalLifecycleComponentError(
                    f"{name} must be ComponentBinding"
                )
        if type(self.backend_registry) not in {dict, MappingProxyType}:
            raise FormalLifecycleComponentError(
                "backend_registry must be an explicit mapping"
            )
        registry = dict(self.backend_registry)
        if set(registry) != set(BACKEND_IDS):
            raise FormalLifecycleComponentError(
                "backend_registry must contain exactly Open3D and PCL"
            )
        if any(type(value) is not ComponentBinding for value in registry.values()):
            raise FormalLifecycleComponentError(
                "every backend registry entry must be ComponentBinding"
            )
        object.__setattr__(
            self,
            "backend_registry",
            MappingProxyType(dict(sorted(registry.items()))),
        )

    def component_bindings(self) -> dict[str, ComponentBinding]:
        result = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if type(getattr(self, item.name)) is ComponentBinding
        }
        result.update(
            {
                f"backend_registry.{name}": binding
                for name, binding in self.backend_registry.items()
            }
        )
        return dict(sorted(result.items()))

    def manifest_binding(self, *, repository_root: Path) -> dict[str, Any]:
        core = {
            "bundle_id": self.bundle_id,
            "bound_mode": self.bound_mode.value,
            "bound_contract_id": self.bound_contract_id,
            "bound_plan_id": self.bound_plan_id,
            "bound_run_id": self.bound_run_id,
            "bound_runtime_root": str(self.bound_runtime_root),
            "bound_publisher_inventory_sha256": (
                self.bound_publisher_inventory_sha256
            ),
            "components": {
                name: binding.report(repository_root=repository_root)
                for name, binding in self.component_bindings().items()
            },
        }
        return {
            **core,
            "bundle_sha256": _canonical_sha256(core),
        }


__all__ = [
    "BACKEND_IDS",
    "ComponentBinding",
    "FormalLifecycleComponentError",
    "FormalLifecycleComponents",
]
