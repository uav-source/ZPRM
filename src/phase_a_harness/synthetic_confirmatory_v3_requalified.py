"""Auditable local requalification route for Synthetic Confirmatory v3.

This module deliberately does not relax or reuse the archived v3 authorization
envelope.  It binds the unchanged v3 plans, seed schedule, scientific protocol,
gate contract, frozen models, backend parameters, and backend implementations to
one new version-agnostic :class:`FormalLifecycleSpec`.  The execution environment
is always identified as ``REQUALIFIED_LOCAL_EXECUTION``.

Importing the module is metadata-only: no NumPy RNG, snapshot, backend, or
runtime path is instantiated until a lifecycle component is explicitly called.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_context import ExecutionContext, ExecutionMode, canonical_plan_rows_sha256
from .formal_lifecycle_components import ComponentBinding, FormalLifecycleComponents
from .formal_lifecycle_contract import (
    CommandProfile,
    FormalLifecycleSpec,
    GitIdentityPolicy,
    PublisherInventory,
    deep_thaw,
)
from .formal_lifecycle_paths import FormalLifecyclePaths
from .formal_runtime_state_machine import formal_runtime_path_contract


EXECUTION_CLASSIFICATION = "REQUALIFIED_LOCAL_EXECUTION"
RUN_ID = "synthetic-confirmatory-v3-requalified"
EXPECTED_BRANCH = "run/synthetic-confirmatory-v3-requalified"
EXPECTED_TAG = "execution/synthetic-confirmatory-v3-requalified"
DEFAULT_RUNTIME_ROOT = Path(
    "/home/lj/ZPRM/zero_perturbation_runtime/confirmatory/"
    "synthetic_confirmatory_v3_requalified"
)
DEFAULT_MANIFEST_RELATIVE = Path(
    "configs/zero_perturbation/"
    "synthetic_confirmatory_v3_requalified_manifest.json"
)
ENTRY_SCRIPT = Path("scripts/run_synthetic_confirmatory_v3_requalified.py")
FROZEN_PYTHON = Path(
    "/home/lj/.local/share/degen-lio-micromamba/envs/"
    "degen-lio-zprm-py311/bin/python3.11"
)

SCIENTIFIC_ASSET_SHA256: Mapping[str, str] = {
    "protocols/synthetic_confirmatory_planned_snapshots_v3.csv": (
        "dbe75e9df8545b1c61c61bac2baa3ee2c525aeda65b9db10215c61863b3b29ef"
    ),
    "protocols/synthetic_confirmatory_planned_trials_v3.csv": (
        "f5b2f6fb84e3a64e686eacfc22c8ec90d6bc9c14860116e8b51f4d628a77b8d4"
    ),
    "protocols/synthetic_confirmatory_protocol_v3.json": (
        "216bfe1b0f9d5fef0f9c11235099979b56394d99ea82452d0f476ddc2d777831"
    ),
    "protocols/synthetic_confirmatory_gate_contract_v3.json": (
        "3f66adb54705d2a08803f6916291bc543f14401f22556468576c2f09b66f30e1"
    ),
    "frozen_assets/synthetic_confirmatory_v3_seed_schedule.json": (
        "a0e7921b9c1282d4b12e3e0cccccd3d66ee3cbc10ff5b4666a6c0cb01f521fbc"
    ),
    "frozen_assets/backend_parameter_contract.json": (
        "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
    ),
    "frozen_assets/confirmatory_development_trained_models_v1.json": (
        "99806f83d3a0d2393ee7e36e8c06f14a1a8a44fb8d02b074ebc2d9fe750d0872"
    ),
}
PCL_CLI_RELATIVE = Path("bin/pcl_point_to_plane_cli")
PCL_CLI_SHA256 = (
    "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff"
)


class RequalifiedV3Error(RuntimeError):
    """The local v3 requalification contract failed closed."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_json_object(path: str | Path) -> dict[str, Any]:
    selected = Path(path)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RequalifiedV3Error(
                    f"duplicate JSON key in {selected}: {key}"
                )
            result[key] = value
        return result

    value = json.loads(
        selected.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RequalifiedV3Error(f"non-finite JSON token in {selected}: {token}")
        ),
    )
    if type(value) is not dict:
        raise RequalifiedV3Error(f"JSON object required: {selected}")
    return value


def scientific_asset_hashes(repository_root: str | Path) -> dict[str, Any]:
    """Authenticate every user-declared immutable scientific asset."""

    root = Path(repository_root).resolve()
    rows: dict[str, dict[str, Any]] = {}
    mismatch = 0
    for relative, expected in SCIENTIFIC_ASSET_SHA256.items():
        path = root / relative
        actual = file_sha256(path) if path.is_file() else None
        passed = actual == expected
        mismatch += int(not passed)
        rows[relative] = {
            "absolute_path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "pass": passed,
        }
    pcl = root / PCL_CLI_RELATIVE
    actual_pcl = file_sha256(pcl) if pcl.is_file() else None
    rows[PCL_CLI_RELATIVE.as_posix()] = {
        "absolute_path": str(pcl),
        "expected_sha256": PCL_CLI_SHA256,
        "actual_sha256": actual_pcl,
        "pass": actual_pcl == PCL_CLI_SHA256,
    }
    mismatch += int(actual_pcl != PCL_CLI_SHA256)
    return {
        "schema_version": "synthetic_confirmatory_v3_source_asset_hashes_v1",
        "execution_classification": EXECUTION_CLASSIFICATION,
        "assets": rows,
        "asset_count": len(rows),
        "mismatch_count": mismatch,
        "all_assets_match": mismatch == 0,
    }


def build_requalified_paths(runtime_root: str | Path) -> FormalLifecyclePaths:
    raw = Path(os.path.abspath(os.fspath(runtime_root)))
    if not raw.is_absolute() or raw == Path(raw.anchor):
        raise RequalifiedV3Error("runtime root must be a non-root absolute path")
    if raw.resolve(strict=False) != raw:
        raise RequalifiedV3Error("runtime root must be canonical")
    return FormalLifecyclePaths(**formal_runtime_path_contract(raw))


def _assert_external_runtime(repository_root: Path, runtime_root: Path) -> None:
    if (
        runtime_root == repository_root
        or runtime_root in repository_root.parents
        or repository_root in runtime_root.parents
    ):
        raise RequalifiedV3Error("runtime root overlaps the source repository")
    current = Path(runtime_root.anchor)
    for part in runtime_root.parts[1:]:
        current /= part
        if current.is_symlink():
            raise RequalifiedV3Error(
                f"runtime root contains a symbolic-link component: {current}"
            )


def load_requalified_plans(
    repository_root: str | Path,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    from . import synthetic_confirmatory_v3_contract as contract

    root = Path(repository_root).resolve()
    snapshots = tuple(
        contract.typed_snapshot_rows(
            root / "protocols/synthetic_confirmatory_planned_snapshots_v3.csv"
        )
    )
    trials = tuple(
        contract.typed_trial_rows(
            root / "protocols/synthetic_confirmatory_planned_trials_v3.csv"
        )
    )
    audit = contract.audit_v3_plan(
        root / "protocols/synthetic_confirmatory_planned_snapshots_v3.csv",
        root / "protocols/synthetic_confirmatory_planned_trials_v3.csv",
    )
    if (
        audit.get("V3_PLAN_PASS") is not True
        or len(snapshots) != 595
        or len(trials) != 1190
        or audit.get("backend_trial_counts")
        != {"open3d_point_to_plane": 595, "pcl_point_to_plane": 595}
        or audit.get("native_trial_count") != 0
    ):
        raise RequalifiedV3Error("frozen v3 plan audit failed")
    return snapshots, trials


def build_requalified_execution_context(
    *, runtime_paths: FormalLifecyclePaths
) -> ExecutionContext:
    """Rebind only external paths; retain the exact frozen v3 science."""

    from .synthetic_confirmatory_v3_contract import formal_execution_context

    archived = formal_execution_context()
    contract = dataclasses.replace(
        archived.contract,
        contract_id="synthetic_confirmatory_v3_requalified_execution_contract_v1",
        expected_cache_root=runtime_paths.snapshot_cache,
        expected_runtime_root=runtime_paths.runtime_root,
    )
    return ExecutionContext(
        mode=ExecutionMode.FORMAL,
        contract=contract,
        plan_id=archived.plan_id,
        cache_root=runtime_paths.snapshot_cache,
        runtime_root=runtime_paths.runtime_root,
        snapshot_schema=archived.snapshot_schema,
        trial_schema=archived.trial_schema,
        allowed_scenes=archived.allowed_scenes,
        allowed_conditions=archived.allowed_conditions,
        allowed_seed_policy=archived.allowed_seed_policy,
        id_policy=archived.id_policy,
        backend_policy=archived.backend_policy,
        snapshot_reader_policy=archived.snapshot_reader_policy,
    )


def build_requalified_publisher_inventory() -> PublisherInventory:
    from .synthetic_confirmatory_v3_contract import (
        PUBLISHER_FIGURES,
        PUBLISHER_ROOT_FILES,
        PUBLISHER_TABLES,
    )

    return PublisherInventory(
        inventory_id="synthetic-confirmatory-v3-requalified-7-3-7-v1",
        tables=tuple(PUBLISHER_TABLES),
        figures=tuple(PUBLISHER_FIGURES),
        root_files=tuple(PUBLISHER_ROOT_FILES),
    )


def build_requalified_command_profile(
    *, entry_script: Path = ENTRY_SCRIPT
) -> CommandProfile:
    return CommandProfile(
        profile_id="synthetic-confirmatory-v3-requalified-command-v1",
        entry_script=entry_script,
        environment={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        argv_prefix=(str(FROZEN_PYTHON),),
        unset_environment=("PYTHONPATH",),
    )


@dataclass(frozen=True)
class RequalifiedBackendInput:
    backend: str
    fixture: Any
    parameters: Mapping[str, Any]
    pcl_cli: Path


def requalified_snapshot_materializer(
    *,
    spec: FormalLifecycleSpec,
    snapshot_row: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    from . import synthetic_confirmatory_v3_contract as contract
    from .synthetic_confirmatory_v3_snapshot_builder import (
        _build_v3_snapshot,
        _write_v3_snapshot_atomic,
    )

    expected = spec.paths.snapshot_cache / str(snapshot_row["planned_snapshot_id"])
    if destination != expected or destination.exists() or destination.is_symlink():
        raise RequalifiedV3Error("snapshot materializer destination is unsafe")
    value = _build_v3_snapshot(
        spec.repository_root,
        contract,
        deep_thaw(spec.manifest),
        snapshot_row,
    )
    published = _write_v3_snapshot_atomic(spec.paths.snapshot_cache, value)
    if published != destination:
        raise RequalifiedV3Error("snapshot was published to an unexpected path")
    return {
        "snapshot_id": snapshot_row["planned_snapshot_id"],
        "confirmatory_seed_access_count": int(
            value["firewall_audit"]["V3_SNAPSHOT_SEED_ACCESS_COUNT"]
        ),
        "confirmatory_rng_instantiation_count": int(
            value["metadata"]["confirmatory_rng_instantiation_count"]
        ),
    }


def requalified_snapshot_reader(
    *,
    spec: FormalLifecycleSpec,
    snapshot_row: Mapping[str, Any],
    expected_lock_entry: Mapping[str, Any] | None,
    arrays: bool,
) -> dict[str, Any]:
    from .synthetic_confirmatory_v3_snapshot_builder import read_v3_snapshot

    return read_v3_snapshot(
        spec.paths.snapshot_cache,
        snapshot_row,
        execution_context=spec.execution_context,
        expected_lock_entry=expected_lock_entry,
        arrays=arrays,
    )


def requalified_snapshot_validator(
    *,
    spec: FormalLifecycleSpec,
    snapshot_row: Mapping[str, Any],
    authenticated_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    from .synthetic_confirmatory_v3_snapshot_builder import _lock_entry

    validated = spec.execution_context.validate_snapshot_row(snapshot_row)
    if authenticated_snapshot.get("metadata", {}).get("snapshot_id") != validated[
        "planned_snapshot_id"
    ]:
        raise RequalifiedV3Error("authenticated snapshot identity changed")
    return _lock_entry(authenticated_snapshot)


def requalified_snapshot_lock_builder(
    *,
    spec: FormalLifecycleSpec,
    authenticated_snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from . import synthetic_confirmatory_v3_contract as contract
    from .synthetic_confirmatory_v3_snapshot_builder import _build_snapshot_lock

    expected_ids = [
        str(row["planned_snapshot_id"]) for row in spec.snapshot_plan
    ]
    if [str(row.get("snapshot_id")) for row in authenticated_snapshots] != expected_ids:
        raise RequalifiedV3Error("snapshot lock entry order changed")
    return _build_snapshot_lock(
        spec.paths.snapshot_cache,
        [deep_thaw(row) for row in spec.snapshot_plan],
        contract=contract,
        manifest=deep_thaw(spec.manifest),
        execution_context=spec.execution_context,
    )


def requalified_snapshot_lock_validator(
    *, spec: FormalLifecycleSpec, lock_value: Mapping[str, Any]
) -> dict[str, Any]:
    from . import synthetic_confirmatory_v3_contract as contract
    from .runtime_lifecycle_io import read_canonical_json
    from .synthetic_confirmatory_v3_snapshot_builder import _validate_snapshot_lock

    if read_canonical_json(spec.paths.snapshot_lock) != dict(lock_value):
        raise RequalifiedV3Error("snapshot lock bytes changed before validation")
    value = _validate_snapshot_lock(
        spec.paths.snapshot_lock,
        spec.paths.snapshot_cache,
        [deep_thaw(row) for row in spec.snapshot_plan],
        contract=contract,
        manifest=deep_thaw(spec.manifest),
        execution_context=spec.execution_context,
    )
    entries = value.get("snapshots")
    if type(entries) is not list or len(entries) != len(spec.snapshot_plan):
        raise RequalifiedV3Error("snapshot lock inventory is incomplete")
    return {
        "snapshot_lock_pass": True,
        "entries_by_id": {str(row["snapshot_id"]): row for row in entries},
        "lock": value,
    }


def requalified_backend_input_builder(
    *,
    spec: FormalLifecycleSpec,
    trial_row: Mapping[str, Any],
    snapshot_row: Mapping[str, Any],
    authenticated_snapshot: Mapping[str, Any],
) -> RequalifiedBackendInput:
    import numpy as np

    from .phase_a_execution_chain_fixture import FixtureSnapshot

    if trial_row["planned_snapshot_id"] != snapshot_row["planned_snapshot_id"]:
        raise RequalifiedV3Error("trial/snapshot binding changed")
    metadata = authenticated_snapshot["metadata"]
    fixture = FixtureSnapshot(
        snapshot_id=str(snapshot_row["planned_snapshot_id"]),
        scene_variant=str(snapshot_row["scene_variant"]),
        condition=str(snapshot_row["condition"]),
        source=np.asarray(authenticated_snapshot["source"]),
        target=np.asarray(authenticated_snapshot["target"]),
        reference=np.asarray(authenticated_snapshot["reference"]),
        expected_failure_classifications=("NONE",),
        checksums={
            "source_checksum": metadata["source_checksum"],
            "target_checksum": metadata["target_checksum"],
            "reference_pose_checksum": metadata["reference_pose_checksum"],
            "snapshot_checksum": metadata["snapshot_checksum"],
        },
    )
    parameter_path = spec.repository_root / (
        "frozen_assets/backend_parameter_contract.json"
    )
    if file_sha256(parameter_path) != SCIENTIFIC_ASSET_SHA256[
        "frozen_assets/backend_parameter_contract.json"
    ]:
        raise RequalifiedV3Error("backend parameter contract changed")
    parameters = strict_json_object(parameter_path)
    backend = str(trial_row["backend"])
    selected = (
        parameters["open3d"]["parameters"]
        if backend == "open3d_point_to_plane"
        else parameters["pcl"]["parameters"]
        if backend == "pcl_point_to_plane"
        else None
    )
    if selected is None:
        raise PermissionError("Native and unknown backends are forbidden")
    pcl_cli = spec.repository_root / PCL_CLI_RELATIVE
    if file_sha256(pcl_cli) != PCL_CLI_SHA256:
        raise RequalifiedV3Error("frozen PCL CLI SHA changed")
    if not os.access(pcl_cli, os.X_OK):
        raise PermissionError("frozen PCL CLI is not executable")
    return RequalifiedBackendInput(
        backend=backend,
        fixture=fixture,
        parameters=selected,
        pcl_cli=pcl_cli,
    )


def requalified_common_record_builder(
    *,
    spec: FormalLifecycleSpec,
    trial_row: Mapping[str, Any],
    snapshot_row: Mapping[str, Any],
    backend_input: RequalifiedBackendInput,
    snapshot_lock_sha256: str,
) -> dict[str, Any]:
    fixture = backend_input.fixture
    return {
        "backend": trial_row["backend"],
        "condition": fixture.condition,
        "implementation_sha256": str(spec.manifest["manifest_payload_sha256"]),
        "planned_trial_id": trial_row["planned_trial_id"],
        "protocol_sha256": SCIENTIFIC_ASSET_SHA256[
            "protocols/synthetic_confirmatory_protocol_v3.json"
        ],
        "reference_pose_checksum": fixture.checksums[
            "reference_pose_checksum"
        ],
        "scene_variant": fixture.scene_variant,
        "schema_version": "phase_a_trial_result_v1",
        "snapshot_checksum": fixture.checksums["snapshot_checksum"],
        "snapshot_id": snapshot_row["planned_snapshot_id"],
        "snapshot_lock_sha256": snapshot_lock_sha256,
        "source_checksum": fixture.checksums["source_checksum"],
        "target_checksum": fixture.checksums["target_checksum"],
    }


def requalified_execute_open3d(
    *, backend_input: RequalifiedBackendInput, common: Mapping[str, Any]
) -> dict[str, Any]:
    if backend_input.backend != "open3d_point_to_plane":
        raise RequalifiedV3Error("Open3D component received another backend")
    if backend_input.fixture.condition == "IDEAL_MATCHED":
        from .phase_a_execution_chain_audit import execute_open3d_fixture
    else:
        from .full_synthetic_backend_execution import (
            execute_full_synthetic_open3d_fixture as execute_open3d_fixture,
        )
    return execute_open3d_fixture(
        fixture=backend_input.fixture,
        common=common,
        parameters=backend_input.parameters,
    )


def requalified_execute_pcl(
    *, backend_input: RequalifiedBackendInput, common: Mapping[str, Any]
) -> dict[str, Any]:
    if backend_input.backend != "pcl_point_to_plane":
        raise RequalifiedV3Error("PCL component received another backend")
    if backend_input.fixture.condition == "IDEAL_MATCHED":
        from .phase_a_execution_chain_audit import execute_pcl_fixture
    else:
        from .full_synthetic_backend_execution import (
            execute_full_synthetic_pcl_fixture as execute_pcl_fixture,
        )
    return execute_pcl_fixture(
        fixture=backend_input.fixture,
        common=common,
        parameters=backend_input.parameters,
        pcl_cli=backend_input.pcl_cli,
    )


def _strict_result(
    value: Mapping[str, Any], *, condition: str
) -> dict[str, Any]:
    if condition == "IDEAL_MATCHED":
        from .phase_a_trial_result_schema import validate_phase_a_trial_result_strict

        return validate_phase_a_trial_result_strict(value)
    from .full_synthetic_trial_result import validate_full_synthetic_trial_result_strict

    return validate_full_synthetic_trial_result_strict(value)


def requalified_result_validator(
    *,
    spec: FormalLifecycleSpec,
    trial_row: Mapping[str, Any],
    common: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    del spec
    result = _strict_result(value, condition=str(trial_row["condition"]))
    if any(result.get(name) != expected for name, expected in common.items()):
        raise RequalifiedV3Error("backend result changed a bound common field")
    return result


def requalified_resume_result_validator(
    *,
    spec: FormalLifecycleSpec,
    trial_row: Mapping[str, Any],
    common: Mapping[str, Any],
    path: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    from .runtime_lifecycle_io import read_canonical_json

    manifest = read_canonical_json(spec.paths.raw_manifest)
    recorded = (
        manifest.get("results", {}).get(trial_row["planned_trial_id"])
        if type(manifest) is dict and type(manifest.get("results")) is dict
        else None
    )
    if recorded != dict(entry):
        raise RequalifiedV3Error(
            "uncommitted orphan result is not authenticated by the raw manifest"
        )
    if trial_row["condition"] == "IDEAL_MATCHED":
        from .phase_a_trial_resume import validate_existing_trial_result_for_resume

        value = validate_existing_trial_result_for_resume(
            path, manifest_entry=entry, expected=common
        )
    else:
        from .full_synthetic_trial_result import (
            validate_existing_full_synthetic_trial_result_for_resume,
        )

        value = validate_existing_full_synthetic_trial_result_for_resume(
            path, manifest_entry=entry, expected=common
        )
    from .runtime_lifecycle_io import canonical_json_bytes

    if path.read_bytes() != canonical_json_bytes(value):
        raise RequalifiedV3Error("resumed result is not canonical JSON")
    return value


def requalified_git_gate(
    *, spec: FormalLifecycleSpec, checkpoint: str
) -> dict[str, Any]:
    from .runtime_git_gate import verify_runtime_git_gate

    identity = spec.identity_policy
    return verify_runtime_git_gate(
        spec.repository_root,
        expected_commit=identity.expected_commit,
        expected_branch=identity.expected_branch,
        expected_tag=identity.expected_tag,
        checkpoint=checkpoint,
    )


def requalified_run_contract_builder(
    *, spec: FormalLifecycleSpec
) -> dict[str, Any]:
    assets = scientific_asset_hashes(spec.repository_root)
    if assets["all_assets_match"] is not True:
        raise RequalifiedV3Error("scientific assets changed at lifecycle entry")
    condition_counts = Counter(
        str(row["condition"]) for row in spec.snapshot_plan
    )
    seed_access_count = (
        condition_counts["IDEAL_MATCHED"]
        + 3 * condition_counts["INDEPENDENT_NOISE_FREE"]
        + 6 * condition_counts["FULL_NOISE"]
    )
    rng_instantiation_count = 3 * condition_counts["FULL_NOISE"]
    return {
        "schema_version": "synthetic_confirmatory_v3_requalified_run_contract_v1",
        "execution_classification": EXECUTION_CLASSIFICATION,
        "execution_mode": spec.mode.value,
        "run_id": spec.run_id,
        "runtime_root": str(spec.paths.runtime_root),
        "runtime_paths": spec.paths.as_dict(),
        "manifest_sha256": spec.manifest_sha256,
        "manifest_payload_sha256": spec.manifest["manifest_payload_sha256"],
        "scientific_asset_sha256": dict(SCIENTIFIC_ASSET_SHA256),
        "pcl_cli_sha256": PCL_CLI_SHA256,
        "snapshot_plan_rows_sha256": canonical_plan_rows_sha256(
            [deep_thaw(row) for row in spec.snapshot_plan]
        ),
        "trial_plan_rows_sha256": canonical_plan_rows_sha256(
            [deep_thaw(row) for row in spec.trial_plan]
        ),
        "execution_context": spec.execution_context.report(),
        "component_bundle": spec.component_bundle.manifest_binding(
            repository_root=spec.repository_root
        ),
        "git_identity_policy": spec.identity_policy.report(),
        "publisher_inventory": spec.publisher_inventory.report(),
        "command_profile": spec.command_profile.report(),
        "planned_snapshot_count": 595,
        "planned_trial_count": 1190,
        "backend_trial_counts": {
            "open3d_point_to_plane": 595,
            "pcl_point_to_plane": 595,
        },
        "confirmatory_seed_access_count": seed_access_count,
        "confirmatory_rng_instantiation_count": rng_instantiation_count,
        "native_trial_count": 0,
        "workers": spec.workers,
    }


def _binding(component_id: str, callback: Any) -> ComponentBinding:
    source = inspect.getsourcefile(callback)
    if source is None:
        raise RequalifiedV3Error(f"component source is unavailable: {component_id}")
    path = Path(source).resolve()
    return ComponentBinding(
        component_id=component_id,
        callable=callback,
        implementation_path=path,
        implementation_sha256=file_sha256(path),
    )


def build_requalified_component_bundle(
    *,
    execution_context: ExecutionContext,
    manifest_sha256: str,
    run_id: str,
    runtime_root: Path,
    publisher_inventory: PublisherInventory,
) -> FormalLifecycleComponents:
    from .synthetic_confirmatory_v3_contract import _validate_formal_trial_plan_row
    from .synthetic_confirmatory_v3_requalified_postrun import (
        requalified_artifact_verifier,
        requalified_difference_auditor,
        requalified_independent_verifier,
        requalified_primary_analyzer,
        requalified_publisher,
    )

    return FormalLifecycleComponents(
        bundle_id="synthetic-confirmatory-v3-requalified-components-v1",
        bound_mode=ExecutionMode.FORMAL,
        bound_contract_id=execution_context.contract.contract_id,
        bound_plan_id=execution_context.plan_id,
        bound_manifest_sha256=manifest_sha256,
        bound_run_id=run_id,
        bound_runtime_root=runtime_root,
        bound_publisher_inventory_sha256=publisher_inventory.payload_sha256,
        snapshot_materializer=_binding(
            "v3-requalified-snapshot-materializer-v1",
            requalified_snapshot_materializer,
        ),
        snapshot_reader=_binding(
            "v3-requalified-authenticated-snapshot-reader-v1",
            requalified_snapshot_reader,
        ),
        snapshot_validator=_binding(
            "v3-requalified-snapshot-validator-v1",
            requalified_snapshot_validator,
        ),
        trial_validator=_binding(
            "v3-frozen-trial-validator-v1",
            _validate_formal_trial_plan_row,
        ),
        snapshot_lock_builder=_binding(
            "v3-requalified-snapshot-lock-builder-v1",
            requalified_snapshot_lock_builder,
        ),
        snapshot_lock_validator=_binding(
            "v3-requalified-snapshot-lock-validator-v1",
            requalified_snapshot_lock_validator,
        ),
        backend_input_builder=_binding(
            "v3-requalified-backend-input-builder-v1",
            requalified_backend_input_builder,
        ),
        common_record_builder=_binding(
            "v3-requalified-common-record-builder-v1",
            requalified_common_record_builder,
        ),
        result_validator=_binding(
            "v3-requalified-strict-result-validator-v1",
            requalified_result_validator,
        ),
        resume_result_validator=_binding(
            "v3-requalified-resume-result-validator-v1",
            requalified_resume_result_validator,
        ),
        primary_analyzer=_binding(
            "v3-requalified-primary-analysis-adapter-v1",
            requalified_primary_analyzer,
        ),
        independent_verifier=_binding(
            "v3-requalified-independent-verifier-adapter-v1",
            requalified_independent_verifier,
        ),
        difference_auditor=_binding(
            "v3-requalified-exact-difference-auditor-v1",
            requalified_difference_auditor,
        ),
        publisher=_binding(
            "v3-requalified-publisher-adapter-v1",
            requalified_publisher,
        ),
        artifact_verifier=_binding(
            "v3-requalified-artifact-verifier-adapter-v1",
            requalified_artifact_verifier,
        ),
        git_gate=_binding(
            "v3-requalified-git-gate-v1",
            requalified_git_gate,
        ),
        run_contract_builder=_binding(
            "v3-requalified-run-contract-builder-v1",
            requalified_run_contract_builder,
        ),
        backend_registry={
            "open3d_point_to_plane": _binding(
                "v3-requalified-real-open3d-backend-v1",
                requalified_execute_open3d,
            ),
            "pcl_point_to_plane": _binding(
                "v3-requalified-real-pcl-backend-v1",
                requalified_execute_pcl,
            ),
        },
    )


def build_requalified_manifest(
    *,
    repository_root: str | Path,
    runtime_root: str | Path = DEFAULT_RUNTIME_ROOT,
    expected_branch: str = EXPECTED_BRANCH,
    expected_tag: str = EXPECTED_TAG,
    run_id: str = RUN_ID,
    workers: int = 2,
    entry_script: Path = ENTRY_SCRIPT,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    paths = build_requalified_paths(runtime_root)
    _assert_external_runtime(root, paths.runtime_root)
    assets = scientific_asset_hashes(root)
    if assets["all_assets_match"] is not True:
        raise RequalifiedV3Error("scientific asset hash mismatch")
    snapshots, trials = load_requalified_plans(root)
    context = build_requalified_execution_context(runtime_paths=paths)
    inventory = build_requalified_publisher_inventory()
    command = build_requalified_command_profile(entry_script=entry_script)
    identity = GitIdentityPolicy(
        policy_id="synthetic-confirmatory-v3-requalified-git-identity-v1",
        expected_commit="0" * 40,
        expected_branch=expected_branch,
        expected_tag=expected_tag,
        checkpoint_prefix="SYNTHETIC_CONFIRMATORY_V3_REQUALIFIED",
    )
    provisional = build_requalified_component_bundle(
        execution_context=context,
        manifest_sha256="0" * 64,
        run_id=run_id,
        runtime_root=paths.runtime_root,
        publisher_inventory=inventory,
    )
    core: dict[str, Any] = {
        "schema_version": "synthetic_confirmatory_v3_requalified_manifest_v1",
        "execution_classification": EXECUTION_CLASSIFICATION,
        "execution_mode": ExecutionMode.FORMAL.value,
        "run_id": run_id,
        "workers": workers,
        "plan_id": context.plan_id,
        "snapshot_plan_rows_sha256": canonical_plan_rows_sha256(list(snapshots)),
        "trial_plan_rows_sha256": canonical_plan_rows_sha256(list(trials)),
        "planned_snapshot_count": len(snapshots),
        "planned_trial_count": len(trials),
        "backend_trial_counts": {
            "open3d_point_to_plane": 595,
            "pcl_point_to_plane": 595,
        },
        "native_trial_count": 0,
        "scientific_asset_sha256": dict(SCIENTIFIC_ASSET_SHA256),
        "pcl_cli_path": PCL_CLI_RELATIVE.as_posix(),
        "pcl_cli_sha256": PCL_CLI_SHA256,
        "formal_lifecycle_paths": paths.as_dict(),
        "formal_lifecycle_components": provisional.manifest_binding(
            repository_root=root
        ),
        "git_identity_policy": identity.manifest_binding(),
        "publisher_inventory": inventory.report(),
        "publisher_inventory_sha256": inventory.payload_sha256,
        "command_profile": command.report(),
        "command_profile_sha256": command.payload_sha256,
        "durable_commit_delay_seconds": {"snapshot": 0.0, "trial": 0.0},
        "scientific_contract_mutation_allowed": False,
        "historical_formal_run_impersonated": False,
        "fresh_runtime_must_be_absent": True,
        "resume_requires_immutable_authentication": True,
    }
    return {**core, "manifest_payload_sha256": canonical_json_sha256(core)}


def build_requalified_formal_lifecycle_spec(
    *,
    repository: str | Path,
    manifest_path: str | Path,
    run_id: str,
    runtime_root: str | Path,
    workers: int,
    expected_commit: str,
    expected_branch: str = EXPECTED_BRANCH,
    expected_tag: str = EXPECTED_TAG,
) -> FormalLifecycleSpec:
    root = Path(repository).resolve()
    paths = build_requalified_paths(runtime_root)
    _assert_external_runtime(root, paths.runtime_root)
    selected_manifest = Path(manifest_path)
    if not selected_manifest.is_absolute():
        selected_manifest = root / selected_manifest
    selected_manifest = selected_manifest.resolve(strict=False)
    if root not in selected_manifest.parents or not selected_manifest.is_file():
        raise RequalifiedV3Error("requalified manifest must be a repository file")
    expected = build_requalified_manifest(
        repository_root=root,
        runtime_root=paths.runtime_root,
        expected_branch=expected_branch,
        expected_tag=expected_tag,
        run_id=run_id,
        workers=workers,
        entry_script=ENTRY_SCRIPT,
    )
    manifest = strict_json_object(selected_manifest)
    if manifest != expected:
        raise RequalifiedV3Error(
            "committed requalified manifest differs from the constructed contract"
        )
    manifest_sha = file_sha256(selected_manifest)
    snapshots, trials = load_requalified_plans(root)
    context = build_requalified_execution_context(runtime_paths=paths)
    inventory = build_requalified_publisher_inventory()
    command = build_requalified_command_profile()
    identity = GitIdentityPolicy(
        policy_id="synthetic-confirmatory-v3-requalified-git-identity-v1",
        expected_commit=expected_commit,
        expected_branch=expected_branch,
        expected_tag=expected_tag,
        checkpoint_prefix="SYNTHETIC_CONFIRMATORY_V3_REQUALIFIED",
    )
    bundle = build_requalified_component_bundle(
        execution_context=context,
        manifest_sha256=manifest_sha,
        run_id=run_id,
        runtime_root=paths.runtime_root,
        publisher_inventory=inventory,
    )
    return FormalLifecycleSpec(
        repository_root=root,
        execution_context=context,
        manifest=manifest,
        manifest_path=selected_manifest,
        manifest_sha256=manifest_sha,
        runtime_paths=paths,
        snapshot_plan=snapshots,
        trial_plan=trials,
        component_bundle=bundle,
        mode=ExecutionMode.FORMAL,
        run_id=run_id,
        workers=workers,
        identity_policy=identity,
        publisher_inventory=inventory,
        command_profile=command,
    )


__all__ = [
    "DEFAULT_MANIFEST_RELATIVE",
    "DEFAULT_RUNTIME_ROOT",
    "EXECUTION_CLASSIFICATION",
    "EXPECTED_BRANCH",
    "EXPECTED_TAG",
    "PCL_CLI_SHA256",
    "RUN_ID",
    "RequalifiedV3Error",
    "build_requalified_formal_lifecycle_spec",
    "build_requalified_manifest",
    "file_sha256",
    "load_requalified_plans",
    "scientific_asset_hashes",
]
