"""Immutable contract for one version-agnostic formal lifecycle invocation.

This module validates only metadata, paths, plans, callable bindings, and Git
identity declarations.  It performs no runtime write, seed access, snapshot
construction, backend dispatch, or scientific analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .execution_context import (
    ExecutionContext,
    ExecutionMode,
    canonical_plan_rows_sha256,
)
from .formal_lifecycle_components import (
    FormalLifecycleComponents,
    FormalLifecycleComponentError,
)
from .formal_lifecycle_paths import FormalLifecyclePaths
from .trial_snapshot_bridge import (
    CanonicalSnapshotIndex,
    TrialSnapshotBindings,
    build_canonical_snapshot_index,
    build_trial_snapshot_bindings,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class FormalLifecycleContractError(ValueError):
    """A lifecycle spec was inconsistent before any mutable action."""


def _require_name(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise FormalLifecycleContractError(
            f"{field} must be a non-empty canonical string"
        )
    if "\x00" in value or "\n" in value or "\r" in value:
        raise FormalLifecycleContractError(
            f"{field} contains a forbidden control character"
        )
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise FormalLifecycleContractError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FormalLifecycleContractError(
                f"regular file required: {path}"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
    finally:
        os.close(descriptor)


def _strict_json_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise FormalLifecycleContractError(
                    f"manifest contains a duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FormalLifecycleContractError(
                    f"manifest contains a non-finite JSON token: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalLifecycleContractError(
            f"manifest is not strict UTF-8 JSON: {path}"
        ) from error
    if type(value) is not dict:
        raise FormalLifecycleContractError(
            "manifest file must contain one JSON object"
        )
    return value


def deep_freeze(value: Any, *, location: str = "$") -> Any:
    """Recursively freeze JSON-native contract data.

    ``Path`` and ``ExecutionMode`` are accepted as explicit Python contract
    values.  Arbitrary objects, sets, bytes, and non-finite numbers are rejected.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str or not key:
                raise FormalLifecycleContractError(
                    f"{location} contains a non-canonical mapping key"
                )
            result[key] = deep_freeze(child, location=f"{location}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (tuple, list)):
        return tuple(
            deep_freeze(child, location=f"{location}[{index}]")
            for index, child in enumerate(value)
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise FormalLifecycleContractError(
            f"{location} contains a non-finite float"
        )
    if isinstance(value, Path):
        return value
    if value is None or type(value) in {bool, int, float, str, ExecutionMode}:
        return value
    raise FormalLifecycleContractError(
        f"{location} is not immutable JSON-native contract data: "
        f"{type(value).__name__}"
    )


def deep_thaw(value: Any) -> Any:
    """Return a JSON-serializable projection of frozen contract data."""

    if isinstance(value, Mapping):
        return {str(key): deep_thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [deep_thaw(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ExecutionMode):
        return value.value
    return value


def _safe_inventory_name(value: Any, *, field: str) -> str:
    name = _require_name(value, field=field)
    path = Path(name)
    if path.is_absolute() or len(path.parts) != 1 or name in {".", ".."}:
        raise FormalLifecycleContractError(
            f"{field} must be one safe filename"
        )
    return name


@dataclass(frozen=True)
class GitIdentityPolicy:
    policy_id: str
    expected_commit: str
    expected_branch: str
    expected_tag: str
    checkpoint_prefix: str
    require_clean_worktree: bool = True
    require_tag_at_commit: bool = True

    def __post_init__(self) -> None:
        _require_name(self.policy_id, field="identity_policy.policy_id")
        if (
            type(self.expected_commit) is not str
            or _GIT_COMMIT.fullmatch(self.expected_commit) is None
        ):
            raise FormalLifecycleContractError(
                "identity_policy.expected_commit must be a lowercase full Git SHA"
            )
        for field in ("expected_branch", "expected_tag", "checkpoint_prefix"):
            _require_name(
                getattr(self, field), field=f"identity_policy.{field}"
            )
        if (
            type(self.require_clean_worktree) is not bool
            or type(self.require_tag_at_commit) is not bool
        ):
            raise FormalLifecycleContractError(
                "Git identity boolean policies must be explicit bools"
            )

    def report(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "expected_commit": self.expected_commit,
            "expected_branch": self.expected_branch,
            "expected_tag": self.expected_tag,
            "checkpoint_prefix": self.checkpoint_prefix,
            "require_clean_worktree": self.require_clean_worktree,
            "require_tag_at_commit": self.require_tag_at_commit,
        }

    def manifest_binding(self) -> dict[str, Any]:
        """Return the non-self-referential policy frozen before candidate commit.

        The exact candidate commit is injected after that commit exists and is
        still checked by every Git gate.  Keeping the digest itself out of a
        file contained by that commit avoids an impossible Git-SHA fixed point.
        """

        return {
            "policy_id": self.policy_id,
            "expected_commit_source": "INJECTED_CANDIDATE_HEAD",
            "expected_branch": self.expected_branch,
            "expected_tag": self.expected_tag,
            "checkpoint_prefix": self.checkpoint_prefix,
            "require_clean_worktree": self.require_clean_worktree,
            "require_tag_at_commit": self.require_tag_at_commit,
        }


@dataclass(frozen=True)
class PublisherInventory:
    inventory_id: str
    tables: tuple[str, ...]
    figures: tuple[str, ...]
    root_files: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_name(self.inventory_id, field="publisher_inventory.inventory_id")
        expected = {"tables": 7, "figures": 3, "root_files": 7}
        all_names: list[str] = []
        for field, count in expected.items():
            values = getattr(self, field)
            if type(values) is not tuple or len(values) != count:
                raise FormalLifecycleContractError(
                    f"publisher_inventory.{field} must contain exactly {count} names"
                )
            normalized = tuple(
                _safe_inventory_name(
                    value, field=f"publisher_inventory.{field}[{index}]"
                )
                for index, value in enumerate(values)
            )
            if len(set(normalized)) != len(normalized):
                raise FormalLifecycleContractError(
                    f"publisher_inventory.{field} contains duplicates"
                )
            all_names.extend(normalized)
        if len(set(all_names)) != len(all_names):
            raise FormalLifecycleContractError(
                "publisher inventory names overlap across locations"
            )

    def report(self) -> dict[str, Any]:
        return {
            "inventory_id": self.inventory_id,
            "tables": list(self.tables),
            "figures": list(self.figures),
            "root_files": list(self.root_files),
            "table_count": len(self.tables),
            "figure_count": len(self.figures),
            "root_file_count": len(self.root_files),
        }

    @property
    def payload_sha256(self) -> str:
        return _canonical_json_sha256(self.report())


@dataclass(frozen=True)
class CommandProfile:
    """Declarative command identity; no command is executed here."""

    profile_id: str
    entry_script: Path
    environment: Mapping[str, str]
    argv_prefix: tuple[str, ...] = ()
    unset_environment: tuple[str, ...] = ("PYTHONPATH",)

    def __post_init__(self) -> None:
        _require_name(self.profile_id, field="command_profile.profile_id")
        script = self.entry_script
        if (
            not isinstance(script, Path)
            or script.is_absolute()
            or not script.parts
            or any(part in {"", ".", ".."} for part in script.parts)
        ):
            raise FormalLifecycleContractError(
                "command_profile.entry_script must be a canonical repository-relative Path"
            )
        if type(self.environment) not in {dict, MappingProxyType}:
            raise FormalLifecycleContractError(
                "command_profile.environment must be an explicit mapping"
            )
        environment = dict(self.environment)
        if any(
            type(name) is not str
            or _ENVIRONMENT_NAME.fullmatch(name) is None
            or type(value) is not str
            or "\x00" in value
            or "\n" in value
            for name, value in environment.items()
        ):
            raise FormalLifecycleContractError(
                "command_profile.environment contains an invalid entry"
            )
        if environment.get("PYTHONNOUSERSITE") != "1":
            raise FormalLifecycleContractError(
                "command profile must set PYTHONNOUSERSITE=1"
            )
        if "PYTHONPATH" in environment:
            raise FormalLifecycleContractError(
                "command profile must not set PYTHONPATH"
            )
        if (
            type(self.argv_prefix) is not tuple
            or any(type(value) is not str or not value for value in self.argv_prefix)
            or type(self.unset_environment) is not tuple
            or any(
                type(value) is not str
                or _ENVIRONMENT_NAME.fullmatch(value) is None
                for value in self.unset_environment
            )
            or "PYTHONPATH" not in self.unset_environment
        ):
            raise FormalLifecycleContractError(
                "command profile argv/environment-unset policy is invalid"
            )
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(dict(sorted(environment.items()))),
        )

    def report(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "entry_script": self.entry_script.as_posix(),
            "environment": dict(self.environment),
            "argv_prefix": list(self.argv_prefix),
            "unset_environment": list(self.unset_environment),
        }

    @property
    def payload_sha256(self) -> str:
        return _canonical_json_sha256(self.report())


@dataclass(frozen=True)
class FormalLifecycleSpec:
    repository_root: Path
    execution_context: ExecutionContext
    manifest: Mapping[str, Any]
    manifest_path: Path
    manifest_sha256: str
    runtime_paths: FormalLifecyclePaths
    snapshot_plan: tuple[Mapping[str, Any], ...]
    trial_plan: tuple[Mapping[str, Any], ...]
    component_bundle: FormalLifecycleComponents
    mode: ExecutionMode
    run_id: str
    workers: int
    identity_policy: GitIdentityPolicy
    publisher_inventory: PublisherInventory
    command_profile: CommandProfile

    def __post_init__(self) -> None:
        root = self.repository_root
        if (
            not isinstance(root, Path)
            or not root.is_absolute()
            or root.resolve(strict=False) != root
            or not root.is_dir()
        ):
            raise FormalLifecycleContractError(
                "repository_root must be an existing canonical absolute directory"
            )
        if type(self.execution_context) is not ExecutionContext:
            raise FormalLifecycleContractError(
                "execution_context must be ExecutionContext"
            )
        if type(self.runtime_paths) is not FormalLifecyclePaths:
            raise FormalLifecycleContractError(
                "runtime_paths must be FormalLifecyclePaths"
            )
        if type(self.component_bundle) is not FormalLifecycleComponents:
            raise FormalLifecycleContractError(
                "component_bundle must be FormalLifecycleComponents"
            )
        if type(self.mode) is not ExecutionMode:
            raise FormalLifecycleContractError("mode must be ExecutionMode")
        _require_name(self.run_id, field="run_id")
        if type(self.workers) is not int or type(self.workers) is bool or self.workers <= 0:
            raise FormalLifecycleContractError(
                "workers must be a positive integer"
            )
        for value, label, expected in (
            (self.identity_policy, "identity_policy", GitIdentityPolicy),
            (self.publisher_inventory, "publisher_inventory", PublisherInventory),
            (self.command_profile, "command_profile", CommandProfile),
        ):
            if type(value) is not expected:
                raise FormalLifecycleContractError(
                    f"{label} must be {expected.__name__}"
                )
        manifest_path = self.manifest_path
        if (
            not isinstance(manifest_path, Path)
            or not manifest_path.is_absolute()
            or manifest_path.resolve(strict=False) != manifest_path
            or not manifest_path.is_file()
        ):
            raise FormalLifecycleContractError(
                "manifest_path must be one canonical regular file"
            )
        _require_sha256(self.manifest_sha256, field="manifest_sha256")
        if _file_sha256(manifest_path) != self.manifest_sha256:
            raise FormalLifecycleContractError("manifest file SHA mismatch")
        if not isinstance(self.manifest, Mapping):
            raise FormalLifecycleContractError("manifest must be a mapping")
        manifest = deep_freeze(self.manifest, location="manifest")
        if deep_thaw(manifest) != _strict_json_object(manifest_path):
            raise FormalLifecycleContractError(
                "in-memory manifest differs from the SHA-bound manifest file"
            )
        snapshots = self._freeze_plan(self.snapshot_plan, label="snapshot_plan")
        trials = self._freeze_plan(self.trial_plan, label="trial_plan")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "snapshot_plan", snapshots)
        object.__setattr__(self, "trial_plan", trials)
        self._validate_cross_bindings()

    @staticmethod
    def _freeze_plan(
        rows: Any, *, label: str
    ) -> tuple[Mapping[str, Any], ...]:
        if type(rows) is not tuple or not rows:
            raise FormalLifecycleContractError(
                f"{label} must be a non-empty tuple"
            )
        frozen = deep_freeze(rows, location=label)
        if any(not isinstance(row, Mapping) for row in frozen):
            raise FormalLifecycleContractError(
                f"{label} must contain only mappings"
            )
        return frozen

    def _validate_cross_bindings(self) -> None:
        context = self.execution_context
        if self.mode is not context.mode:
            raise FormalLifecycleContractError(
                "spec mode differs from execution context mode"
            )
        if (
            context.runtime_root != self.runtime_paths.runtime_root
            or context.cache_root != self.runtime_paths.snapshot_cache
        ):
            raise FormalLifecycleContractError(
                "execution context differs from explicit runtime paths"
            )
        manifest = deep_thaw(self.manifest)
        required_identity = {
            "execution_mode": self.mode.value,
            "run_id": self.run_id,
            "plan_id": context.plan_id,
            "snapshot_plan_rows_sha256": canonical_plan_rows_sha256(
                [deep_thaw(row) for row in self.snapshot_plan]
            ),
            "trial_plan_rows_sha256": canonical_plan_rows_sha256(
                [deep_thaw(row) for row in self.trial_plan]
            ),
            "publisher_inventory_sha256": self.publisher_inventory.payload_sha256,
            "command_profile_sha256": self.command_profile.payload_sha256,
        }
        if any(manifest.get(name) != value for name, value in required_identity.items()):
            raise FormalLifecycleContractError(
                "manifest lifecycle identity differs from the spec"
            )
        if manifest.get("formal_lifecycle_paths") != self.runtime_paths.as_dict():
            raise FormalLifecycleContractError(
                "manifest runtime path binding differs from the spec"
            )
        expected_components = self.component_bundle.manifest_binding(
            repository_root=self.repository_root
        )
        if manifest.get("formal_lifecycle_components") != expected_components:
            raise FormalLifecycleContractError(
                "manifest component binding differs from the direct callables"
            )
        if (
            manifest.get("git_identity_policy")
            != self.identity_policy.manifest_binding()
        ):
            raise FormalLifecycleContractError(
                "manifest Git identity policy differs from the spec"
            )
        if manifest.get("publisher_inventory") != self.publisher_inventory.report():
            raise FormalLifecycleContractError(
                "manifest publisher inventory differs from the spec"
            )
        if manifest.get("command_profile") != self.command_profile.report():
            raise FormalLifecycleContractError(
                "manifest command profile differs from the spec"
            )
        bundle = self.component_bundle
        expected_bundle_identity = (
            bundle.bound_mode is self.mode
            and bundle.bound_contract_id == context.contract.contract_id
            and bundle.bound_plan_id == context.plan_id
            and bundle.bound_manifest_sha256 == self.manifest_sha256
            and bundle.bound_run_id == self.run_id
            and bundle.bound_runtime_root == self.runtime_paths.runtime_root
            and bundle.bound_publisher_inventory_sha256
            == self.publisher_inventory.payload_sha256
        )
        if not expected_bundle_identity:
            raise FormalLifecycleContractError(
                "component bundle belongs to another lifecycle context"
            )
        entry_script = self.repository_root / self.command_profile.entry_script
        if (
            entry_script.resolve(strict=False) != entry_script
            or not entry_script.is_file()
        ):
            raise FormalLifecycleContractError(
                "command profile entry script is absent or non-canonical"
            )
        component_backends = tuple(sorted(bundle.backend_registry))
        context_backends = tuple(sorted(context.backend_policy.allowed_backends))
        if component_backends != context_backends:
            raise FormalLifecycleContractError(
                "backend registry differs from execution context policy"
            )

    @property
    def paths(self) -> FormalLifecyclePaths:
        return self.runtime_paths

    @property
    def components(self) -> FormalLifecycleComponents:
        return self.component_bundle


@dataclass(frozen=True)
class ValidatedFormalLifecycleSpec:
    spec: FormalLifecycleSpec
    canonical_snapshot_index: CanonicalSnapshotIndex
    trial_snapshot_bindings: TrialSnapshotBindings
    spec_payload_sha256: str
    snapshot_plan_rows_sha256: str
    trial_plan_rows_sha256: str


def validate_formal_lifecycle_spec(
    spec: FormalLifecycleSpec,
) -> ValidatedFormalLifecycleSpec:
    """Finish fail-fast plan/bridge validation and return immutable bindings."""

    if type(spec) is not FormalLifecycleSpec:
        raise FormalLifecycleContractError(
            "formal lifecycle entry requires FormalLifecycleSpec"
        )
    snapshots = [deep_thaw(row) for row in spec.snapshot_plan]
    trials = [deep_thaw(row) for row in spec.trial_plan]
    spec.execution_context.validate_snapshot_plan_binding(snapshots)
    spec.execution_context.validate_trial_plan_binding(trials)
    index = build_canonical_snapshot_index(
        snapshots, execution_context=spec.execution_context
    )
    bindings = build_trial_snapshot_bindings(index, trials)
    report = {
        "execution_context": spec.execution_context.report(),
        "manifest_sha256": spec.manifest_sha256,
        "runtime_paths": spec.runtime_paths.as_dict(),
        "run_id": spec.run_id,
        "workers": spec.workers,
        "mode": spec.mode.value,
        "identity_policy": spec.identity_policy.report(),
        "publisher_inventory": spec.publisher_inventory.report(),
        "command_profile": spec.command_profile.report(),
        "component_bundle": spec.component_bundle.manifest_binding(
            repository_root=spec.repository_root
        ),
        "snapshot_plan_rows_sha256": canonical_plan_rows_sha256(snapshots),
        "trial_plan_rows_sha256": canonical_plan_rows_sha256(trials),
    }
    return ValidatedFormalLifecycleSpec(
        spec=spec,
        canonical_snapshot_index=index,
        trial_snapshot_bindings=bindings,
        spec_payload_sha256=_canonical_json_sha256(report),
        snapshot_plan_rows_sha256=report["snapshot_plan_rows_sha256"],
        trial_plan_rows_sha256=report["trial_plan_rows_sha256"],
    )


__all__ = [
    "CommandProfile",
    "FormalLifecycleContractError",
    "FormalLifecycleSpec",
    "GitIdentityPolicy",
    "PublisherInventory",
    "ValidatedFormalLifecycleSpec",
    "deep_freeze",
    "deep_thaw",
    "validate_formal_lifecycle_spec",
]
