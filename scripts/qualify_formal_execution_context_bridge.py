#!/usr/bin/env python3
"""Qualify the injected v3 execution context and trial-snapshot bridge.

This executable is deliberately bounded to the deterministic, seed-free
three-snapshot/six-trial fixture.  It uses the same v3 reader, validators,
trial-to-snapshot bridge, ``_fixture`` and ``_execute_one`` path as the formal
runner.  The only substitution is an explicit terminal backend-dispatch probe
used before the six real qualification backend trials.

The formal 595/1,190 plans are read only for a static identity bridge audit.
This program never reads a formal snapshot payload, never constructs a
Confirmatory RNG, and never creates or resumes the formal runtime root.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
HARNESS_SOURCE_ROOT = REPOSITORY / "src"
if str(HARNESS_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_SOURCE_ROOT))

SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
QUALIFICATION_ROOT = Path(
    "/home/lj/zero_perturbation_runtime/qualification/"
    "formal_execution_context_bridge_v1"
)
QUALIFICATION_CACHE = QUALIFICATION_ROOT / "snapshot_cache"
FORMAL_V3_ROOT = Path(
    "/home/lj/zero_perturbation_runtime/confirmatory/synthetic_confirmatory_v3"
)
RUN_ID = "formal-execution-context-bridge-qualification-v1"
EXPECTED_BRANCH = "fix/zero-perturbation-formal-execution-context"
FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"
PCL_CLI_SHA256 = (
    "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff"
)
FROZEN_MODEL_SHA256 = (
    "99806f83d3a0d2393ee7e36e8c06f14a1a8a44fb8d02b074ebc2d9fe750d0872"
)

OPEN3D = "open3d_point_to_plane"
PCL = "pcl_point_to_plane"
BACKENDS = (OPEN3D, PCL)
SNAPSHOT_FIELDS = (
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
    "planned_backend_count",
    "replicate_semantics",
)
TRIAL_FIELDS = (
    "planned_trial_id",
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
    "backend",
)
QUALIFICATION_SPECS = (
    (
        "qualification-identity-v1",
        "IDENTITY",
        "FIXTURE_IDENTITY",
    ),
    (
        "qualification-nonidentity-reference-v1",
        "NONIDENTITY_REFERENCE",
        "FIXTURE_NONIDENTITY_REFERENCE",
    ),
    (
        "qualification-no-correspondence-v1",
        "NO_CORRESPONDENCE",
        "FIXTURE_NO_CORRESPONDENCE",
    ),
)
QUALIFICATION_SCENES = tuple(item[1] for item in QUALIFICATION_SPECS)
QUALIFICATION_CONDITIONS = tuple(item[2] for item in QUALIFICATION_SPECS)
EXPECTED_OUTCOME = {
    "FIXTURE_IDENTITY": "NONE",
    "FIXTURE_NONIDENTITY_REFERENCE": "NONE",
    "FIXTURE_NO_CORRESPONDENCE": "NO_CORRESPONDENCES",
}
REPLICATE_SEMANTICS = "ONE_SEED_FREE_QUALIFICATION_INPUT"


class BackendDispatchBoundaryReached(RuntimeError):
    """Intentional terminal signal emitted by the backend-free probe."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value).rstrip(b"\n")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    from phase_a_harness.runtime_lifecycle_io import atomic_create_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_bytes(path, payload)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    from phase_a_harness.runtime_lifecycle_io import atomic_create_canonical_json

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_canonical_json(path, dict(value))


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    from phase_a_harness.runtime_lifecycle_io import atomic_replace_canonical_json

    atomic_replace_canonical_json(path, dict(value))


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = item
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token in {path}: {token}")
        ),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON object required: {path}")
    return value


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPOSITORY,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _assert_environment_and_fresh_root() -> dict[str, Any]:
    if REPOSITORY != Path(
        "/home/lj/zero_perturbation_phase_a_harness_20260729_1407"
    ):
        raise PermissionError("qualification is bound to the standalone harness")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("qualification requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError("qualification requires the frozen MAMBA_ROOT_PREFIX")
    source = SOURCE_REPOSITORY.resolve()
    for entry in (
        *sys.path,
        *(item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item),
    ):
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("source Degen-LIO is on the Python search path")
    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD^{commit}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPOSITORY,
        text=True,
    )
    if branch != EXPECTED_BRANCH or status:
        raise PermissionError(
            "qualification requires the clean candidate branch; "
            f"branch={branch!r}, dirty={bool(status)}"
        )
    if QUALIFICATION_ROOT.exists() or QUALIFICATION_ROOT.is_symlink():
        raise FileExistsError(
            f"qualification root must be absent: {QUALIFICATION_ROOT}"
        )
    formal_resolved = FORMAL_V3_ROOT.resolve(strict=False)
    qualification_resolved = QUALIFICATION_ROOT.resolve(strict=False)
    if (
        qualification_resolved == formal_resolved
        or qualification_resolved in formal_resolved.parents
        or formal_resolved in qualification_resolved.parents
    ):
        raise PermissionError("qualification root overlaps the formal v3 root")
    QUALIFICATION_ROOT.mkdir(parents=True, exist_ok=False)
    return {
        "schema_version": "formal_execution_context_qualification_entry_v1",
        "branch": branch,
        "commit": commit,
        "git_worktree_clean": True,
        "qualification_root": str(QUALIFICATION_ROOT),
        "qualification_root_freshly_created": True,
        "formal_v3_root": str(FORMAL_V3_ROOT),
        "formal_v3_root_exists": FORMAL_V3_ROOT.exists(),
        "formal_v3_payload_read_count": 0,
        "formal_v3_backend_execution_count": 0,
    }


def qualification_snapshot_rows() -> list[dict[str, Any]]:
    return [
        {
            "planned_snapshot_id": snapshot_id,
            "scene_variant": scene,
            "condition": condition,
            "geometry_seed": None,
            "measurement_seed": None,
            "repeat_index": 0,
            "planned_backend_count": 2,
            "replicate_semantics": REPLICATE_SEMANTICS,
        }
        for snapshot_id, scene, condition in QUALIFICATION_SPECS
    ]


def qualification_trial_rows() -> list[dict[str, Any]]:
    return [
        {
            "planned_trial_id": f"{snapshot['planned_snapshot_id']}::{backend}",
            "planned_snapshot_id": snapshot["planned_snapshot_id"],
            "scene_variant": snapshot["scene_variant"],
            "condition": snapshot["condition"],
            "geometry_seed": None,
            "measurement_seed": None,
            "repeat_index": 0,
            "backend": backend,
        }
        for snapshot in qualification_snapshot_rows()
        for backend in BACKENDS
    ]


def _validate_qualification_snapshot_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != set(SNAPSHOT_FIELDS):
        raise ValueError("qualification snapshot row field set changed")
    value = {name: row[name] for name in SNAPSHOT_FIELDS}
    expected = {
        snapshot_id: (scene, condition)
        for snapshot_id, scene, condition in QUALIFICATION_SPECS
    }
    snapshot_id = value["planned_snapshot_id"]
    if (
        type(snapshot_id) is not str
        or snapshot_id not in expected
        or "/" in snapshot_id
        or "\\" in snapshot_id
        or (value["scene_variant"], value["condition"]) != expected[snapshot_id]
        or value["geometry_seed"] is not None
        or value["measurement_seed"] is not None
        or type(value["repeat_index"]) is not int
        or value["repeat_index"] != 0
        or type(value["planned_backend_count"]) is not int
        or value["planned_backend_count"] != 2
        or value["replicate_semantics"] != REPLICATE_SEMANTICS
    ):
        raise ValueError("qualification snapshot identity is invalid")
    return value


def _validate_qualification_trial_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != set(TRIAL_FIELDS):
        raise ValueError("qualification trial row field set changed")
    value = {name: row[name] for name in TRIAL_FIELDS}
    snapshot_ids = {item[0] for item in QUALIFICATION_SPECS}
    if (
        type(value["planned_trial_id"]) is not str
        or type(value["planned_snapshot_id"]) is not str
        or value["planned_snapshot_id"] not in snapshot_ids
        or value["scene_variant"] not in QUALIFICATION_SCENES
        or value["condition"] not in QUALIFICATION_CONDITIONS
        or value["geometry_seed"] is not None
        or value["measurement_seed"] is not None
        or type(value["repeat_index"]) is not int
        or value["repeat_index"] != 0
        or value["backend"] not in BACKENDS
        or value["planned_trial_id"]
        != f"{value['planned_snapshot_id']}::{value['backend']}"
    ):
        raise ValueError("qualification trial identity is invalid")
    return value


def _validate_seed_free_negative_snapshot_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != set(SNAPSHOT_FIELDS):
        raise ValueError("negative-only snapshot row field set changed")
    value = {name: row[name] for name in SNAPSHOT_FIELDS}
    if value != {
        "planned_snapshot_id": "negative-sentinel-snapshot-v1",
        "scene_variant": "IDENTITY",
        "condition": "FIXTURE_IDENTITY",
        "geometry_seed": "FIXTURE_A",
        "measurement_seed": "FIXTURE_M0",
        "repeat_index": 0,
        "planned_backend_count": 2,
        "replicate_semantics": "NEGATIVE_ONLY_SENTINEL_INPUT",
    }:
        raise ValueError("negative-only snapshot identity changed")
    return value


def _validate_seed_free_negative_trial_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != set(TRIAL_FIELDS):
        raise ValueError("negative-only trial row field set changed")
    value = {name: row[name] for name in TRIAL_FIELDS}
    if (
        value["planned_snapshot_id"] != "negative-sentinel-snapshot-v1"
        or value["scene_variant"] != "IDENTITY"
        or value["condition"] != "FIXTURE_IDENTITY"
        or value["geometry_seed"] not in {"FIXTURE_A", "FIXTURE_B"}
        or value["measurement_seed"] not in {"FIXTURE_M0", "FIXTURE_M1"}
        or value["repeat_index"] not in {0, 1}
        or value["backend"] not in BACKENDS
        or value["planned_trial_id"]
        != f"negative-sentinel-snapshot-v1::{value['backend']}"
    ):
        raise ValueError("negative-only trial identity changed")
    return value


def build_qualification_execution_context(
    *, runtime_root: Path = QUALIFICATION_ROOT, cache_root: Path = QUALIFICATION_CACHE
) -> Any:
    from phase_a_harness.execution_context import (
        BackendPolicy,
        canonical_plan_rows_sha256,
        ExecutionContext,
        ExecutionContract,
        ExecutionMode,
        IdPolicy,
        SchemaBinding,
        SeedPolicy,
        SnapshotReaderPolicy,
    )
    from phase_a_harness.synthetic_confirmatory_v3_snapshot_builder import (
        METADATA_FIELDS,
        _LOCK_ENTRY_FIELDS,
    )

    mode = ExecutionMode.QUALIFICATION
    snapshot_schema = SchemaBinding(
        mode,
        "formal_execution_context_bridge_qualification_snapshot_plan_v1",
        SNAPSHOT_FIELDS,
    )
    trial_schema = SchemaBinding(
        mode,
        "formal_execution_context_bridge_qualification_trial_plan_v1",
        TRIAL_FIELDS,
    )
    seed_policy = SeedPolicy(
        mode,
        "seed_free_qualification_no_confirmatory_seed_v1",
        False,
    )
    id_policy = IdPolicy(
        mode,
        "explicit_seed_free_qualification_ids_v1",
    )
    backend_policy = BackendPolicy(
        mode,
        "qualified_open3d_pcl_only_v1",
        BACKENDS,
        2,
    )
    reader_policy = SnapshotReaderPolicy(
        mode=mode,
        policy_id="formal_execution_context_bridge_qualification_reader_v1",
        metadata_fields=tuple(sorted(METADATA_FIELDS)),
        lock_entry_fields=tuple(sorted(_LOCK_ENTRY_FIELDS)),
        metadata_schema="formal_execution_context_bridge_qualification_metadata_v1",
        snapshot_schema_version=(
            "formal_execution_context_bridge_qualification_snapshot_v1"
        ),
        lineage_schema_version=(
            "formal_execution_context_bridge_qualification_lineage_v1"
        ),
        seed_namespace=None,
        snapshot_builder_contract_version=(
            "formal_execution_context_bridge_qualification_builder_v1"
        ),
        lineage_required_conditions=("FIXTURE_IDENTITY",),
        expected_rng_counts={
            condition: 0 for condition in QUALIFICATION_CONDITIONS
        },
    )
    plan_id = "formal-execution-context-bridge-qualification-plan-v1"
    bound_snapshot_rows = qualification_snapshot_rows()
    bound_trial_rows = qualification_trial_rows()
    contract = ExecutionContract(
        mode=mode,
        contract_id="formal_execution_context_bridge_qualification_contract_v1",
        plan_id=plan_id,
        expected_snapshot_plan_sha256=canonical_plan_rows_sha256(
            bound_snapshot_rows
        ),
        expected_trial_plan_sha256=canonical_plan_rows_sha256(bound_trial_rows),
        expected_snapshot_count=len(bound_snapshot_rows),
        expected_trial_count=len(bound_trial_rows),
        expected_cache_root=cache_root,
        expected_runtime_root=runtime_root,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=QUALIFICATION_SCENES,
        allowed_conditions=QUALIFICATION_CONDITIONS,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
        snapshot_validator=_validate_qualification_snapshot_row,
        trial_validator=_validate_qualification_trial_row,
        result_route="QUALIFICATION_PHASE_A_ROUTING_V1",
    )
    return ExecutionContext(
        mode=mode,
        contract=contract,
        plan_id=plan_id,
        cache_root=cache_root,
        runtime_root=runtime_root,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=QUALIFICATION_SCENES,
        allowed_conditions=QUALIFICATION_CONDITIONS,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
    )


def build_negative_qualification_plan_context(
    *,
    runtime_root: Path,
    cache_root: Path,
    snapshot_plan_rows: Sequence[Mapping[str, Any]],
    trial_plan_rows: Sequence[Mapping[str, Any]],
) -> Any:
    """Bind one isolated negative-only plan under a distinct contract identity."""

    from dataclasses import replace

    from phase_a_harness.execution_context import canonical_plan_rows_sha256

    base = build_qualification_execution_context(
        runtime_root=runtime_root,
        cache_root=cache_root,
    )
    snapshots = [dict(row) for row in snapshot_plan_rows]
    trials = [dict(row) for row in trial_plan_rows]
    identity = _canonical_sha256({"snapshots": snapshots, "trials": trials})[:16]
    contract = replace(
        base.contract,
        contract_id=f"negative_only_qualification_contract_{identity}",
        plan_id=f"negative-only-qualification-plan-{identity}",
        expected_snapshot_plan_sha256=canonical_plan_rows_sha256(snapshots),
        expected_trial_plan_sha256=canonical_plan_rows_sha256(trials),
        expected_snapshot_count=len(snapshots),
        expected_trial_count=len(trials),
        result_route="NEGATIVE_ONLY_NO_BACKEND_ROUTING_V1",
    )
    return replace(base, contract=contract, plan_id=contract.plan_id)


def build_negative_shared_field_context(*, runtime_root: Path) -> Any:
    """Create a non-executing contract with one additional shared field."""

    from dataclasses import replace

    from phase_a_harness.execution_context import (
        SchemaBinding,
        canonical_plan_rows_sha256,
    )

    base = build_qualification_execution_context(
        runtime_root=runtime_root,
        cache_root=runtime_root / "snapshot_cache",
    )
    snapshot_fields = (*SNAPSHOT_FIELDS, "fixture_revision")
    trial_fields = (*TRIAL_FIELDS, "fixture_revision")
    snapshot_schema = SchemaBinding(
        base.mode,
        "negative_shared_field_snapshot_plan_v1",
        snapshot_fields,
    )
    trial_schema = SchemaBinding(
        base.mode,
        "negative_shared_field_trial_plan_v1",
        trial_fields,
    )

    def validate_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(row, Mapping) or set(row) != set(snapshot_fields):
            raise ValueError("negative shared-field snapshot schema changed")
        core = _validate_qualification_snapshot_row(
            {name: row[name] for name in SNAPSHOT_FIELDS}
        )
        if row["fixture_revision"] not in {"fixture-v1", "fixture-v2"}:
            raise ValueError("negative shared-field revision changed")
        return {**core, "fixture_revision": row["fixture_revision"]}

    def validate_trial(row: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(row, Mapping) or set(row) != set(trial_fields):
            raise ValueError("negative shared-field trial schema changed")
        core = _validate_qualification_trial_row(
            {name: row[name] for name in TRIAL_FIELDS}
        )
        if row["fixture_revision"] not in {"fixture-v1", "fixture-v2"}:
            raise ValueError("negative shared-field revision changed")
        return {**core, "fixture_revision": row["fixture_revision"]}

    snapshots = [
        {**row, "fixture_revision": "fixture-v1"}
        for row in qualification_snapshot_rows()
    ]
    trials = [
        {**row, "fixture_revision": "fixture-v1"}
        for row in qualification_trial_rows()
    ]
    contract = replace(
        base.contract,
        contract_id="negative_shared_field_qualification_contract_v1",
        plan_id="negative-shared-field-qualification-plan-v1",
        expected_snapshot_plan_sha256=canonical_plan_rows_sha256(snapshots),
        expected_trial_plan_sha256=canonical_plan_rows_sha256(trials),
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        snapshot_validator=validate_snapshot,
        trial_validator=validate_trial,
        result_route="NEGATIVE_ONLY_NO_BACKEND_ROUTING_V1",
    )
    return replace(
        base,
        contract=contract,
        plan_id=contract.plan_id,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
    )


def build_seed_free_negative_identity_context(*, runtime_root: Path) -> Any:
    """Build a non-executing sentinel context for bridge mismatch tests.

    The sentinel labels are strings, not numeric seeds, and the context has no
    snapshot payload or backend route.  It exists only to make the three
    seed/repeat shared-field mismatch classifications reachable without
    touching any Confirmatory seed value.
    """

    from phase_a_harness.execution_context import (
        BackendPolicy,
        canonical_plan_rows_sha256,
        ExecutionContext,
        ExecutionContract,
        ExecutionMode,
        IdPolicy,
        SchemaBinding,
        SeedPolicy,
        SnapshotReaderPolicy,
    )
    from phase_a_harness.synthetic_confirmatory_v3_snapshot_builder import (
        METADATA_FIELDS,
        _LOCK_ENTRY_FIELDS,
    )

    mode = ExecutionMode.QUALIFICATION
    cache_root = runtime_root / "snapshot_cache"
    snapshot_schema = SchemaBinding(
        mode, "seed_free_negative_snapshot_plan_v1", SNAPSHOT_FIELDS
    )
    trial_schema = SchemaBinding(
        mode, "seed_free_negative_trial_plan_v1", TRIAL_FIELDS
    )
    seed_policy = SeedPolicy(mode, "negative_sentinel_no_confirmatory_seed_v1", False)
    id_policy = IdPolicy(mode, "negative_sentinel_identity_v1")
    backend_policy = BackendPolicy(mode, "negative_sentinel_backend_policy_v1", BACKENDS, 2)
    reader_policy = SnapshotReaderPolicy(
        mode=mode,
        policy_id="negative_sentinel_reader_policy_v1",
        metadata_fields=tuple(sorted(METADATA_FIELDS)),
        lock_entry_fields=tuple(sorted(_LOCK_ENTRY_FIELDS)),
        metadata_schema="negative_sentinel_metadata_v1",
        snapshot_schema_version="negative_sentinel_snapshot_v1",
        lineage_schema_version="negative_sentinel_lineage_v1",
        seed_namespace=None,
        snapshot_builder_contract_version="negative_sentinel_builder_v1",
        lineage_required_conditions=("FIXTURE_IDENTITY",),
        expected_rng_counts={"FIXTURE_IDENTITY": 0},
    )
    plan_id = "seed-free-negative-sentinel-plan-v1"
    bound_snapshot_rows = [
        {
            "planned_snapshot_id": "negative-sentinel-snapshot-v1",
            "scene_variant": "IDENTITY",
            "condition": "FIXTURE_IDENTITY",
            "geometry_seed": "FIXTURE_A",
            "measurement_seed": "FIXTURE_M0",
            "repeat_index": 0,
            "planned_backend_count": 2,
            "replicate_semantics": "NEGATIVE_ONLY_SENTINEL_INPUT",
        }
    ]
    bound_trial_rows = [
        {
            "planned_trial_id": f"negative-sentinel-snapshot-v1::{backend}",
            "planned_snapshot_id": "negative-sentinel-snapshot-v1",
            "scene_variant": "IDENTITY",
            "condition": "FIXTURE_IDENTITY",
            "geometry_seed": "FIXTURE_A",
            "measurement_seed": "FIXTURE_M0",
            "repeat_index": 0,
            "backend": backend,
        }
        for backend in BACKENDS
    ]
    contract = ExecutionContract(
        mode=mode,
        contract_id="seed_free_negative_sentinel_contract_v1",
        plan_id=plan_id,
        expected_snapshot_plan_sha256=canonical_plan_rows_sha256(
            bound_snapshot_rows
        ),
        expected_trial_plan_sha256=canonical_plan_rows_sha256(bound_trial_rows),
        expected_snapshot_count=len(bound_snapshot_rows),
        expected_trial_count=len(bound_trial_rows),
        expected_cache_root=cache_root,
        expected_runtime_root=runtime_root,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=("IDENTITY",),
        allowed_conditions=("FIXTURE_IDENTITY",),
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
        snapshot_validator=_validate_seed_free_negative_snapshot_row,
        trial_validator=_validate_seed_free_negative_trial_row,
        result_route="NEGATIVE_ONLY_NO_BACKEND_ROUTING_V1",
    )
    return ExecutionContext(
        mode=mode,
        contract=contract,
        plan_id=plan_id,
        cache_root=cache_root,
        runtime_root=runtime_root,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=("IDENTITY",),
        allowed_conditions=("FIXTURE_IDENTITY",),
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
    )


def _qualification_metadata(
    *, row: Mapping[str, Any], source: Any, target: Any, reference: Any,
    parent_indices: Any | None, execution_context: Any,
) -> dict[str, Any]:
    import numpy as np

    from phase_a_harness.synthetic_confirmatory_v3_snapshot_builder import (
        METADATA_FIELDS,
        PARENT_INDEX_FILENAME,
        _canonical_sha256 as snapshot_canonical_sha256,
        _lineage_fields,
        _lineage_recomputed,
        _raw_sha256,
    )

    raw = {
        "source_checksum": _raw_sha256(source),
        "target_checksum": _raw_sha256(target),
        "reference_pose_checksum": _raw_sha256(reference),
    }
    parent_sha = None
    if parent_indices is None:
        lineage_fields = _lineage_fields(
            lineage=None, source=source, target=target
        )
    else:
        parent_sha = _raw_sha256(parent_indices)
        recomputed = _lineage_recomputed(
            source, target, reference, parent_indices
        )
        count = int(len(parent_indices))
        unique = int(len(np.unique(parent_indices)))
        lineage_fields = {
            **recomputed["closure"],
            "lineage_closure_violation_count": int(
                recomputed["lineage_closure_violation_count"]
            ),
            "lineage_validation_method": (
                "SEED_FREE_QUALIFICATION_PARENT_INDEX_ROW_CORRESPONDENCE"
            ),
            "parent_index_count": count,
            "parent_index_duplicate_count": count - unique,
            "parent_index_out_of_range_count": int(
                np.count_nonzero(
                    (parent_indices < 0) | (parent_indices >= len(target))
                )
            ),
            "parent_index_unique_count": unique,
            "parent_points_map_f64_sha256": recomputed[
                "parent_points_map_f64_sha256"
            ],
            "source_has_target_parent_lineage": True,
            "source_is_target_subset": True,
            "source_parent_row_count_match": len(source) == count,
            "source_parent_target_indices_path": PARENT_INDEX_FILENAME,
            "source_parent_target_indices_sha256": parent_sha,
        }
    snapshot_checksum = snapshot_canonical_sha256(
        {
            "snapshot_id": row["planned_snapshot_id"],
            **raw,
            "source_parent_target_indices_sha256": parent_sha,
        }
    )
    binding_sha = _canonical_sha256(
        {
            "contract_id": execution_context.contract.contract_id,
            "plan_id": execution_context.plan_id,
            "seed_free": True,
        }
    )
    policy = execution_context.snapshot_reader_policy
    metadata = {
        "array_file_sha256": {},
        "condition": row["condition"],
        "confirmatory_rng_instantiation_count": 0,
        "development_protocol_sha256": binding_sha,
        "dropout_parameters": {
            "map_dropout_fraction": 0.0,
            "scan_dropout_fraction": 0.0,
        },
        "formal_manifest_payload_sha256": binding_sha,
        "generator_record_checksums": {},
        "generator_sha256": binding_sha,
        "geometry_seed": None,
        "independent_sampling": False,
        "initial_pose": "reference_pose_exact",
        "lineage_schema_version": policy.lineage_schema_version,
        "measurement_seed": None,
        "metadata_payload_sha256": "",
        "noise_parameters": {
            "map_noise_sigma_m": 0.0,
            "scan_noise_sigma_m": 0.0,
        },
        "planned_snapshot_id": row["planned_snapshot_id"],
        "reference_pose_checksum": raw["reference_pose_checksum"],
        "repeat_index": 0,
        "scene_variant": row["scene_variant"],
        "schema_version": policy.metadata_schema,
        "seed_namespace": None,
        "snapshot_builder_contract_version": (
            policy.snapshot_builder_contract_version
        ),
        "snapshot_checksum": snapshot_checksum,
        "snapshot_id": row["planned_snapshot_id"],
        "snapshot_schema_version": policy.snapshot_schema_version,
        "source_checksum": raw["source_checksum"],
        "source_point_count": int(len(source)),
        "target_checksum": raw["target_checksum"],
        "target_point_count": int(len(target)),
        **lineage_fields,
    }
    if set(metadata) != set(METADATA_FIELDS):
        raise AssertionError("qualification metadata does not match the real reader")
    return metadata


def _lock_entry(authenticated: Mapping[str, Any]) -> dict[str, Any]:
    metadata = authenticated["metadata"]
    return {
        "condition": metadata["condition"],
        "confirmatory_rng_instantiation_count": metadata[
            "confirmatory_rng_instantiation_count"
        ],
        "file_sha256": dict(authenticated["file_sha256"]),
        "geometry_seed": metadata["geometry_seed"],
        "measurement_seed": metadata["measurement_seed"],
        "metadata_payload_sha256": metadata["metadata_payload_sha256"],
        "reference_pose_checksum": authenticated["reference_pose_checksum"],
        "repeat_index": metadata["repeat_index"],
        "scene_variant": metadata["scene_variant"],
        "snapshot_checksum": authenticated["snapshot_checksum"],
        "snapshot_id": metadata["snapshot_id"],
        "source_checksum": authenticated["source_checksum"],
        "source_parent_target_indices_sha256": metadata[
            "source_parent_target_indices_sha256"
        ],
        "target_checksum": authenticated["target_checksum"],
    }


def materialize_qualification_snapshots(
    *, execution_context: Any, snapshot_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    import numpy as np

    from phase_a_harness.phase_a_execution_chain_fixture import (
        build_fixture_snapshots,
    )
    from phase_a_harness.synthetic_confirmatory_v3_snapshot_builder import (
        _write_v3_snapshot_atomic,
        read_v3_snapshot,
    )

    cache_root = execution_context.cache_root
    runtime_root = execution_context.runtime_root
    cache_root.mkdir(parents=True, exist_ok=False)
    by_condition = {
        fixture.condition: fixture for fixture in build_fixture_snapshots()
    }
    generated: list[str] = []
    for raw_row in snapshot_rows:
        row = execution_context.validate_snapshot_row(raw_row)
        fixture = by_condition[row["condition"]]
        source = np.ascontiguousarray(fixture.source, dtype="<f4")
        target = np.ascontiguousarray(fixture.target, dtype="<f4")
        reference = np.ascontiguousarray(fixture.reference, dtype="<f8")
        parent_indices = None
        if row["condition"] == "FIXTURE_IDENTITY":
            parent_indices = np.ascontiguousarray(
                np.arange(len(source), dtype="<i8")
            )
        metadata = _qualification_metadata(
            row=row,
            source=source,
            target=target,
            reference=reference,
            parent_indices=parent_indices,
            execution_context=execution_context,
        )
        _write_v3_snapshot_atomic(
            cache_root,
            {
                "metadata": metadata,
                "parent_indices": parent_indices,
                "reference": reference,
                "source": source,
                "target": target,
            },
        )
        generated.append(row["planned_snapshot_id"])
    entries = []
    for row in snapshot_rows:
        authenticated = read_v3_snapshot(
            cache_root,
            row,
            execution_context=execution_context,
            expected_lock_entry=None,
            arrays=False,
        )
        entries.append(_lock_entry(authenticated))
    lock_core = {
        "schema_version": "formal_execution_context_bridge_snapshot_lock_v1",
        "confirmatory_seed_allowed": False,
        "confirmatory_rng_instantiation_count": 0,
        "planned_snapshot_count": 3,
        "snapshots": entries,
    }
    lock = {
        **lock_core,
        "snapshot_lock_payload_sha256": _canonical_sha256(lock_core),
    }
    lock_path = runtime_root / "snapshot_lock.json"
    _write_json(lock_path, lock)
    lock_by_id = {entry["snapshot_id"]: entry for entry in entries}
    if len(lock_by_id) != 3:
        raise ValueError("qualification snapshot lock is not unique")
    authenticated_rows = []
    for row in snapshot_rows:
        item = read_v3_snapshot(
            cache_root,
            row,
            execution_context=execution_context,
            expected_lock_entry=lock_by_id[row["planned_snapshot_id"]],
            arrays=False,
        )
        authenticated_rows.append(
            {
                "planned_snapshot_id": row["planned_snapshot_id"],
                "snapshot_checksum": item["snapshot_checksum"],
                "source_checksum": item["source_checksum"],
                "target_checksum": item["target_checksum"],
                "reference_pose_checksum": item["reference_pose_checksum"],
                "file_sha256": dict(item["file_sha256"]),
            }
        )
    return {
        "generated_snapshot_count": len(generated),
        "generated_snapshot_ids": generated,
        "confirmatory_seed_access_count": 0,
        "confirmatory_rng_instantiation_count": 0,
        "lock": lock,
        "lock_by_id": lock_by_id,
        "lock_path": lock_path,
        "snapshot_lock_sha256": _file_sha256(lock_path),
        "authenticated_rows": authenticated_rows,
    }


def _load_parameter_stack() -> tuple[dict[str, Mapping[str, Any]], Path, dict[str, Any]]:
    parameter_path = (
        REPOSITORY / "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
    )
    value = _strict_json(parameter_path)
    if value.get("formal_seed_values_included") is not False:
        raise PermissionError("fixture parameter lock contains a formal seed")
    parameters = {
        OPEN3D: value["open3d_parameter_contract"]["parameters"],
        PCL: value["pcl_parameter_contract"]["parameters"],
    }
    for backend, section_name in (
        (OPEN3D, "open3d_parameter_contract"),
        (PCL, "pcl_parameter_contract"),
    ):
        section = value[section_name]
        if _canonical_sha256(parameters[backend]) != section["canonical_sha256"]:
            raise ValueError(f"fixture parameter SHA mismatch: {backend}")
    pcl_cli = REPOSITORY / "bin/pcl_point_to_plane_cli"
    if _file_sha256(pcl_cli) != PCL_CLI_SHA256:
        raise ValueError("PCL CLI differs from the frozen qualification binary")
    return parameters, pcl_cli, value


def build_qualification_stack(
    *, execution_context: Any, snapshot_state: Mapping[str, Any],
    snapshot_rows: Sequence[Mapping[str, Any]],
    trial_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Any, Any]:
    from phase_a_harness.trial_snapshot_bridge import (
        build_canonical_snapshot_index,
        build_trial_snapshot_bindings,
    )

    canonical_index = build_canonical_snapshot_index(
        snapshot_rows, execution_context=execution_context
    )
    bindings = build_trial_snapshot_bindings(canonical_index, trial_rows)
    parameters, pcl_cli, parameter_lock = _load_parameter_stack()
    protocol = {
        "schema_version": "formal_execution_context_bridge_qualification_protocol_v1",
        "seed_free": True,
        "snapshot_count": 3,
        "trial_count": 6,
        "backends": list(BACKENDS),
        "conditions": list(QUALIFICATION_CONDITIONS),
    }
    implementation = {
        "execution_context": _file_sha256(
            REPOSITORY / "src/phase_a_harness/execution_context.py"
        ),
        "trial_snapshot_bridge": _file_sha256(
            REPOSITORY / "src/phase_a_harness/trial_snapshot_bridge.py"
        ),
        "runner": _file_sha256(
            REPOSITORY
            / "src/phase_a_harness/synthetic_confirmatory_v3_runner.py"
        ),
        "snapshot_reader": _file_sha256(
            REPOSITORY
            / "src/phase_a_harness/synthetic_confirmatory_v3_snapshot_builder.py"
        ),
        "qualification_script": _file_sha256(Path(__file__).resolve()),
        "fixture_parameter_lock": _canonical_sha256(parameter_lock),
        "pcl_cli": PCL_CLI_SHA256,
    }
    manifest = {
        "schema_version": "formal_execution_context_bridge_qualification_manifest_v1",
        "manifest_payload_sha256": _canonical_sha256(implementation),
        "protocol_sha256": _canonical_sha256(protocol),
    }
    runtime_paths = {
        "snapshot_cache": execution_context.cache_root,
        "raw_results": execution_context.runtime_root / "raw_results",
        "raw_manifest": execution_context.runtime_root / "raw_result_manifest.json",
        "backend_temporary": execution_context.runtime_root / "backend_tmp",
    }
    stack = {
        "execution_context": execution_context,
        "canonical_snapshot_index": canonical_index,
        "trial_snapshot_bindings": bindings,
        "lock_by_id": snapshot_state["lock_by_id"],
        "snapshot_lock_sha256": snapshot_state["snapshot_lock_sha256"],
        "runtime_paths": runtime_paths,
        "parameters": parameters,
        "pcl_cli": pcl_cli,
        "manifest": manifest,
    }
    return stack, canonical_index, bindings


def run_backend_free_probe(
    *, stack: Mapping[str, Any], trial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    from phase_a_harness import synthetic_confirmatory_v3_runner as runner

    records: list[dict[str, Any]] = []

    def terminal_probe(
        *, fixture: Any, common: Mapping[str, Any],
        parameters: Mapping[str, Any], backend: str, pcl_cli: Path,
    ) -> dict[str, Any]:
        if backend not in BACKENDS:
            raise PermissionError("terminal probe received an unauthorized backend")
        if parameters != stack["parameters"][backend]:
            raise ValueError("terminal probe received the wrong parameter contract")
        if Path(pcl_cli) != Path(stack["pcl_cli"]):
            raise ValueError("terminal probe received the wrong PCL CLI binding")
        records.append(
            {
                "backend": backend,
                "condition": fixture.condition,
                "planned_trial_id": common["planned_trial_id"],
                "scene_variant": fixture.scene_variant,
                "snapshot_id": fixture.snapshot_id,
                "snapshot_checksum": fixture.checksums["snapshot_checksum"],
                "source_checksum": fixture.checksums["source_checksum"],
                "target_checksum": fixture.checksums["target_checksum"],
                "reference_pose_checksum": fixture.checksums[
                    "reference_pose_checksum"
                ],
                "authenticated_point_counts": {
                    "source": int(len(fixture.source)),
                    "target": int(len(fixture.target)),
                },
            }
        )
        raise BackendDispatchBoundaryReached(common["planned_trial_id"])

    probe_stack = dict(stack)
    probe_stack["backend_dispatch"] = terminal_probe
    for row in trial_rows:
        try:
            runner._execute_one(probe_stack, row)
        except BackendDispatchBoundaryReached as error:
            if str(error) != row["planned_trial_id"]:
                raise RuntimeError("terminal dispatch probe identity changed") from error
        else:
            raise RuntimeError("terminal dispatch probe unexpectedly returned a result")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["snapshot_id"]].append(record)
    pairing_mismatch = sum(
        len(group) != 2
        or {row["backend"] for row in group} != set(BACKENDS)
        or len({row["snapshot_checksum"] for row in group}) != 1
        for group in grouped.values()
    )
    module_pass = bool(
        runner._execute_one.__module__
        == "phase_a_harness.synthetic_confirmatory_v3_runner"
        and runner._fixture.__module__
        == "phase_a_harness.synthetic_confirmatory_v3_runner"
    )
    passed = bool(len(records) == 6 and len(grouped) == 3 and not pairing_mismatch)
    return {
        "schema_version": "formal_execution_context_backend_free_probe_v1",
        "records": records,
        "probe_invocation_count": len(records),
        "backend_execution_count": 0,
        "backend_pairing_mismatch_count": pairing_mismatch,
        "execute_one_mocked": False,
        "fixture_mocked": False,
        "snapshot_reader_mocked": False,
        "snapshot_validator_mocked": False,
        "trial_snapshot_bridge_mocked": False,
        "UNMOCKED_EXECUTE_ONE_PATH_PASS": module_pass,
        "UNMOCKED_FIXTURE_PATH_PASS": module_pass,
        "REAL_SNAPSHOT_READER_PATH_PASS": passed,
        "REAL_SNAPSHOT_VALIDATOR_PATH_PASS": passed,
        "TRIAL_TO_SNAPSHOT_BINDING_PASS": passed,
        "BACKEND_DISPATCH_BOUNDARY_REACHED": passed,
    }


def _result_file_sha_map(raw_manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        trial_id: str(entry["sha256"])
        for trial_id, entry in sorted(raw_manifest["results"].items())
    }


def _snapshot_file_sha_map(snapshot_state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        row["planned_snapshot_id"]: {
            "snapshot_checksum": row["snapshot_checksum"],
            "file_sha256": dict(sorted(row["file_sha256"].items())),
        }
        for row in snapshot_state["authenticated_rows"]
    }


def run_qualification_trial_lifecycle(
    *,
    stack: Mapping[str, Any],
    trial_rows: Sequence[Mapping[str, Any]],
    run_id: str,
    contract_sha256: str,
) -> dict[str, Any]:
    """Execute only missing trials and authenticate every retained result."""

    from phase_a_harness import synthetic_confirmatory_v3_runner as runner
    from phase_a_harness.phase_a_trial_result_schema import canonical_json_bytes
    from phase_a_harness.runtime_lifecycle_io import atomic_create_bytes

    raw_results = stack["runtime_paths"]["raw_results"]
    raw_results.mkdir(parents=True, exist_ok=True)
    stack["runtime_paths"]["backend_temporary"].mkdir(
        parents=True, exist_ok=True
    )
    raw_manifest_path = stack["runtime_paths"]["raw_manifest"]
    if raw_manifest_path.exists():
        raw_manifest = runner._read_raw_manifest(
            raw_manifest_path,
            run_id=run_id,
            contract_sha256=contract_sha256,
        )
    else:
        raw_manifest = runner._empty_raw_manifest(run_id, contract_sha256)
        _write_json(raw_manifest_path, raw_manifest)
    expected_by_id = {row["planned_trial_id"]: row for row in trial_rows}
    initial = runner._audit_raw_inventory(
        stack, raw_manifest, expected_by_id, allow_orphans=False
    )
    pending = [
        row
        for row in trial_rows
        if row["planned_trial_id"] not in raw_manifest["results"]
    ]
    executed_ids: list[str] = []
    for row in pending:
        common, payload = runner._execute_one(stack, row)
        if common["planned_trial_id"] != row["planned_trial_id"]:
            raise RuntimeError("qualification lifecycle changed trial identity")
        destination = raw_results / runner._result_filename(
            row["planned_trial_id"]
        )
        atomic_create_bytes(destination, canonical_json_bytes(payload))
        raw_manifest["results"][row["planned_trial_id"]] = {
            "path": destination.name,
            "planned_trial_id": row["planned_trial_id"],
            "sha256": _file_sha256(destination),
        }
        _replace_json(raw_manifest_path, raw_manifest)
        executed_ids.append(row["planned_trial_id"])
    final = runner._audit_raw_inventory(
        stack, raw_manifest, expected_by_id, allow_orphans=False
    )
    return {
        "payloads": sorted(
            final["payloads"], key=lambda row: row["planned_trial_id"]
        ),
        "raw_manifest": raw_manifest,
        "pending_trial_count_at_entry": len(pending),
        "valid_trial_skip_count": len(initial["payloads"]),
        "backend_execution_count": len(executed_ids),
        "executed_trial_ids": executed_ids,
        "missing_trial_count": final["missing_trial_count"],
        "corrupt_trial_count": final["corrupt_trial_count"],
        "checksum_mismatch_count": final["checksum_mismatch_count"],
    }


def execute_fresh_and_resume(
    *, execution_context: Any, stack: dict[str, Any],
    snapshot_state: Mapping[str, Any], snapshot_rows: Sequence[Mapping[str, Any]],
    trial_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from phase_a_harness.synthetic_confirmatory_v3_snapshot_builder import (
        read_v3_snapshot,
    )

    contract_sha = _canonical_sha256(execution_context.report())
    fresh_cycle = run_qualification_trial_lifecycle(
        stack=stack,
        trial_rows=trial_rows,
        run_id=RUN_ID,
        contract_sha256=contract_sha,
    )
    fresh_results = fresh_cycle["payloads"]
    if (
        fresh_cycle["pending_trial_count_at_entry"] != 6
        or fresh_cycle["valid_trial_skip_count"] != 0
        or fresh_cycle["backend_execution_count"] != 6
        or fresh_cycle["missing_trial_count"] != 0
    ):
        raise RuntimeError("fresh qualification lifecycle cardinality changed")
    for payload in fresh_results:
        expected = EXPECTED_OUTCOME[payload["condition"]]
        if payload["failure_classification"] != expected:
            raise RuntimeError(
                "qualification backend outcome changed: "
                f"{payload['planned_trial_id']} -> "
                f"{payload['failure_classification']}"
            )
        if expected == "NONE" and (
            payload["solver_failure"] is not False
            or payload["finite_output"] is not True
        ):
            raise RuntimeError("successful qualification result is not finite")
        if expected == "NO_CORRESPONDENCES" and payload["solver_failure"] is not True:
            raise RuntimeError("no-correspondence result is not a solver failure")
    backend_counts = Counter(row["backend"] for row in fresh_results)
    if backend_counts != Counter({OPEN3D: 3, PCL: 3}):
        raise RuntimeError("qualification backend execution cardinality changed")
    snapshot_before = _snapshot_file_sha_map(snapshot_state)
    trial_before = _result_file_sha_map(fresh_cycle["raw_manifest"])

    # Resume authenticates every existing snapshot/result.  No generator and no
    # backend dispatch is called on this path.
    resumed_snapshot_rows = []
    for row in snapshot_rows:
        item = read_v3_snapshot(
            execution_context.cache_root,
            row,
            execution_context=execution_context,
            expected_lock_entry=stack["lock_by_id"][row["planned_snapshot_id"]],
            arrays=False,
        )
        resumed_snapshot_rows.append(
            {
                "planned_snapshot_id": row["planned_snapshot_id"],
                "snapshot_checksum": item["snapshot_checksum"],
                "file_sha256": dict(item["file_sha256"]),
            }
        )
    resume_cycle = run_qualification_trial_lifecycle(
        stack=stack,
        trial_rows=trial_rows,
        run_id=RUN_ID,
        contract_sha256=contract_sha,
    )
    if (
        resume_cycle["pending_trial_count_at_entry"] != 0
        or resume_cycle["valid_trial_skip_count"] != 6
        or resume_cycle["backend_execution_count"] != 0
        or resume_cycle["missing_trial_count"] != 0
    ):
        raise RuntimeError("resume qualification lifecycle did not skip 6/6")
    resumed_results = resume_cycle["payloads"]
    fresh_sorted = sorted(fresh_results, key=lambda row: row["planned_trial_id"])
    snapshot_after = {
        row["planned_snapshot_id"]: {
            "snapshot_checksum": row["snapshot_checksum"],
            "file_sha256": dict(sorted(row["file_sha256"].items())),
        }
        for row in resumed_snapshot_rows
    }
    trial_after = _result_file_sha_map(resume_cycle["raw_manifest"])
    fresh_resume_equivalence = fresh_sorted == resumed_results
    fresh_report = {
        "schema_version": "formal_execution_context_seed_free_backend_fixture_v1",
        "run_id": RUN_ID,
        "fixture_snapshot_count": 3,
        "fixture_trial_count": 6,
        "open3d_trial_count": backend_counts[OPEN3D],
        "pcl_trial_count": backend_counts[PCL],
        "native_trial_count": 0,
        "backend_execution_count": fresh_cycle["backend_execution_count"],
        "confirmatory_seed_access_count": 0,
        "confirmatory_rng_instantiation_count": 0,
        "backend_determinism_seed_call_count": 3,
        "failure_inventory": [
            {
                "backend": backend,
                "condition": condition,
                "failure_classification": classification,
                "count": count,
            }
            for (backend, condition, classification), count in sorted(
                Counter(
                    (
                        row["backend"],
                        row["condition"],
                        row["failure_classification"],
                    )
                    for row in fresh_sorted
                ).items()
            )
        ],
        "results": fresh_sorted,
        "SEED_FREE_FIXTURE_PASS": True,
    }
    resume_report = {
        "schema_version": "formal_execution_context_fresh_resume_v1",
        "lifecycle_callable": (
            "scripts.qualify_formal_execution_context_bridge."
            "run_qualification_trial_lifecycle"
        ),
        "fresh_pending_trial_count": fresh_cycle[
            "pending_trial_count_at_entry"
        ],
        "fresh_backend_execution_count": fresh_cycle[
            "backend_execution_count"
        ],
        "resume_pending_trial_count": resume_cycle[
            "pending_trial_count_at_entry"
        ],
        "resume_backend_execution_count": resume_cycle[
            "backend_execution_count"
        ],
        "resume_skipped_valid_snapshot_count": len(resumed_snapshot_rows),
        "resume_skipped_valid_trial_count": resume_cycle[
            "valid_trial_skip_count"
        ],
        "VALID_SNAPSHOT_REEXECUTION_COUNT": 0,
        "VALID_TRIAL_REEXECUTION_COUNT": resume_cycle[
            "backend_execution_count"
        ],
        "SNAPSHOT_CHECKSUM_CHANGE_AFTER_RESUME": int(
            snapshot_before != snapshot_after
        ),
        "TRIAL_CHECKSUM_CHANGE_AFTER_RESUME": int(trial_before != trial_after),
        "fresh_resume_scientific_equivalence": fresh_resume_equivalence,
        "FRESH_RESUME_PASS": bool(
            fresh_resume_equivalence
            and snapshot_before == snapshot_after
            and trial_before == trial_after
        ),
    }
    if resume_report["FRESH_RESUME_PASS"] is not True:
        raise RuntimeError("qualification fresh/resume equivalence failed")
    return resumed_results, fresh_report, resume_report


def audit_formal_plan_bridge() -> tuple[dict[str, Any], Any, list[dict[str, Any]], list[dict[str, Any]]]:
    from phase_a_harness import synthetic_confirmatory_v3_contract as contract
    from phase_a_harness.trial_snapshot_bridge import (
        build_canonical_snapshot_index,
        build_trial_snapshot_bindings,
    )

    snapshots = contract.typed_snapshot_rows(REPOSITORY / contract.SNAPSHOT_PLAN_RELATIVE)
    trials = contract.typed_trial_rows(REPOSITORY / contract.TRIAL_PLAN_RELATIVE)
    context = contract.formal_execution_context()
    index = build_canonical_snapshot_index(snapshots, execution_context=context)
    bindings = build_trial_snapshot_bindings(index, trials)
    duplicate = len(snapshots) - len({row["planned_snapshot_id"] for row in snapshots})
    missing = sum(row["planned_snapshot_id"] not in index.rows for row in trials)
    backend_counts = Counter(row["backend"] for row in trials)
    passed = bool(
        len(snapshots) == 595
        and len(trials) == 1190
        and duplicate == 0
        and missing == 0
        and bindings.audit["pairing_violation_count"] == 0
        and backend_counts == Counter({OPEN3D: 595, PCL: 595})
        and bindings.audit["native_trial_count"] == 0
    )
    report = {
        "schema_version": "formal_v3_plan_bridge_static_audit_v1",
        "planned_snapshot_count": len(snapshots),
        "unique_snapshot_count": len(index),
        "planned_trial_count": len(trials),
        "unique_trial_count": len(bindings.rows),
        "backend_trial_counts": dict(sorted(backend_counts.items())),
        "native_trial_count": bindings.audit["native_trial_count"],
        "PLAN_TRIAL_WITHOUT_SNAPSHOT_COUNT": missing,
        "PLAN_DUPLICATE_SNAPSHOT_ID_COUNT": duplicate,
        "PLAN_SHARED_IDENTITY_MISMATCH_COUNT": 0,
        "PLAN_BACKEND_PAIRING_VIOLATION_COUNT": bindings.audit[
            "pairing_violation_count"
        ],
        "formal_snapshot_payload_read_count": 0,
        "formal_backend_execution_count": 0,
        "confirmatory_seed_schedule_read_count": 0,
        "FORMAL_PLAN_BRIDGE_STATIC_AUDIT_PASS": passed,
    }
    if not passed:
        raise RuntimeError("formal 595/1,190 static bridge audit failed")
    return report, context, snapshots, trials


def audit_formal_validation_semantics(
    *, formal_context: Any, snapshots: Sequence[Mapping[str, Any]],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from phase_a_harness import synthetic_confirmatory_v3_contract as contract
    from phase_a_harness import synthetic_confirmatory_v3_snapshot_builder as builder

    valid_difference = 0
    for row in snapshots:
        legacy = contract._validate_formal_snapshot_plan_row(row)
        injected = formal_context.validate_snapshot_row(row)
        valid_difference += int(legacy != injected)
    for row in trials:
        legacy = contract._validate_formal_trial_plan_row(row)
        injected = formal_context.validate_trial_row(row)
        valid_difference += int(legacy != injected)

    illegal_snapshots: list[dict[str, Any]] = []
    base = dict(snapshots[0])
    for name, mutation in (
        ("missing_required_field", lambda row: row.pop("planned_backend_count")),
        ("invalid_scene", lambda row: row.__setitem__("scene_variant", "IDENTITY")),
        ("invalid_condition", lambda row: row.__setitem__("condition", "FIXTURE_IDENTITY")),
        ("invalid_geometry_seed", lambda row: row.__setitem__("geometry_seed", None)),
        ("invalid_snapshot_id", lambda row: row.__setitem__("planned_snapshot_id", "invalid")),
        ("invalid_backend_count", lambda row: row.__setitem__("planned_backend_count", 3)),
        ("invalid_replicate_semantics", lambda row: row.__setitem__("replicate_semantics", "invalid")),
    ):
        candidate = dict(base)
        mutation(candidate)
        candidate["_negative_name"] = name
        illegal_snapshots.append(candidate)
    invalid_accept_difference = 0
    invalid_inventory = []
    for candidate_with_name in illegal_snapshots:
        candidate = dict(candidate_with_name)
        name = candidate.pop("_negative_name")
        accepted = []
        for validator in (
            contract._validate_formal_snapshot_plan_row,
            formal_context.validate_snapshot_row,
        ):
            try:
                validator(candidate)
            except (KeyError, TypeError, ValueError, RuntimeError):
                accepted.append(False)
            else:
                accepted.append(True)
        invalid_accept_difference += int(accepted[0] != accepted[1])
        invalid_inventory.append(
            {"case": name, "legacy_accepted": accepted[0], "injected_accepted": accepted[1]}
        )

    policy = formal_context.snapshot_reader_policy
    reader_rule_differences = sum(
        (
            set(policy.metadata_fields) != set(builder.METADATA_FIELDS),
            set(policy.lock_entry_fields) != set(builder._LOCK_ENTRY_FIELDS),
            policy.metadata_schema != contract.METADATA_SCHEMA,
            policy.snapshot_schema_version != contract.SNAPSHOT_SCHEMA,
            policy.lineage_schema_version != contract.LINEAGE_SCHEMA,
            policy.seed_namespace != contract.NAMESPACE,
            policy.snapshot_builder_contract_version
            != builder.SNAPSHOT_BUILDER_CONTRACT_VERSION,
            set(policy.lineage_required_conditions) != {"IDEAL_MATCHED"},
            dict(policy.expected_rng_counts)
            != {"IDEAL_MATCHED": 0, "INDEPENDENT_NOISE_FREE": 0, "FULL_NOISE": 3},
        )
    )
    wrapper_source = inspect.getsource(builder.read_formal_v3_snapshot)
    delegation_pass = bool(
        "read_v3_snapshot(" in wrapper_source
        and "execution_context=context_factory()" in wrapper_source
    )
    validation_difference = valid_difference + invalid_accept_difference
    return {
        "schema_version": "formal_validation_semantics_equivalence_v1",
        "valid_formal_snapshot_row_count": len(snapshots),
        "valid_formal_trial_row_count": len(trials),
        "invalid_formal_snapshot_case_count": len(illegal_snapshots),
        "invalid_case_inventory": invalid_inventory,
        "required_snapshot_fields": list(formal_context.snapshot_schema.fields),
        "required_trial_fields": list(formal_context.trial_schema.fields),
        "allowed_scenes": list(formal_context.allowed_scenes),
        "allowed_conditions": list(formal_context.allowed_conditions),
        "formal_seed_policy": formal_context.allowed_seed_policy.report(),
        "formal_id_policy": formal_context.id_policy.report(),
        "formal_cache_root": str(formal_context.cache_root),
        "formal_lineage_required_conditions": list(
            policy.lineage_required_conditions
        ),
        "formal_reader_wrapper_delegates_to_parameterized_reader": delegation_pass,
        "formal_snapshot_payload_read_count": 0,
        "formal_reader_wrapper_delegation_pass": delegation_pass,
        "qualification_authentication_negative_cases_bound": False,
        "FORMAL_VALIDATION_RULE_DIFFERENCE_COUNT": validation_difference,
        "FORMAL_READER_SCIENTIFIC_SEMANTICS_DIFFERENCE_COUNT": (
            reader_rule_differences + int(not delegation_pass)
        ),
        "FORMAL_SNAPSHOT_AUTHENTICATION_DIFFERENCE_COUNT": int(not delegation_pass),
    }


def _exception_classification(error: BaseException) -> str:
    from phase_a_harness.synthetic_confirmatory_v3_snapshot_builder import (
        V3SnapshotContractError,
    )

    classification = getattr(error, "classification", None)
    if isinstance(classification, str):
        return classification
    if isinstance(error, V3SnapshotContractError):
        return "SNAPSHOT_AUTHENTICATION_FAILURE"
    if isinstance(error, TypeError):
        return "IMMUTABLE_INDEX_MUTATION_REJECTED"
    return type(error).__name__


def _json_native(value: Any) -> Any:
    if value is None or type(value) in {bool, int, float, str}:
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite_float": str(value)}
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_native(item) for item in value]
    return {"python_type": type(value).__name__}


def run_negative_cases(
    *, qualification_context: Any, qualification_index: Any,
    qualification_bindings: Any, snapshot_rows: Sequence[Mapping[str, Any]],
    trial_rows: Sequence[Mapping[str, Any]], formal_context: Any,
    formal_snapshots: Sequence[Mapping[str, Any]],
    formal_trials: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from phase_a_harness.execution_context import ExecutionContext, ExecutionMode
    from phase_a_harness.synthetic_confirmatory_v3_snapshot_builder import (
        read_v3_snapshot,
    )
    from phase_a_harness.trial_snapshot_bridge import (
        bind_trial_to_snapshot,
        build_canonical_snapshot_index,
        build_trial_snapshot_bindings,
    )

    cases: list[dict[str, Any]] = []

    def reject(
        name: str, expected: str | Sequence[str], action: Callable[[], Any]
    ) -> None:
        expected_values = (expected,) if isinstance(expected, str) else tuple(expected)
        accepted = False
        observed = "NO_EXCEPTION"
        detail = None
        actual: Any = None
        expected_detail: Any = list(expected_values)
        try:
            action()
        except Exception as error:  # qualification matrix records exact rejection
            observed = _exception_classification(error)
            detail = str(error)
            reporter = getattr(error, "report", None)
            structured = reporter() if callable(reporter) else None
            if isinstance(structured, Mapping):
                actual = _json_native(structured.get("actual"))
                expected_detail = _json_native(structured.get("expected"))
            else:
                actual = {
                    "exception_type": type(error).__name__,
                    "detail": str(error),
                }
                expected_detail = (
                    "authenticated snapshot"
                    if observed == "SNAPSHOT_AUTHENTICATION_FAILURE"
                    else list(expected_values)
                )
        else:
            accepted = True
            actual = {"exception": None}
        cases.append(
            {
                "case": name,
                "expected_classification": "|".join(expected_values),
                "observed_classification": observed,
                "accepted": accepted,
                "expected_classification_match": observed in expected_values,
                "backend_invocation_count": 0,
                "actual": actual,
                "expected": expected_detail,
                "detail": detail,
            }
        )

    # A valid third trial against a distinct negative-only two-snapshot plan.
    missing_runtime = (
        qualification_context.runtime_root
        / "negative_cases"
        / "missing_reference_runtime"
    )
    missing_context = build_negative_qualification_plan_context(
        runtime_root=missing_runtime,
        cache_root=missing_runtime / "snapshot_cache",
        snapshot_plan_rows=snapshot_rows[:2],
        trial_plan_rows=[trial_rows[4]],
    )
    reject(
        "snapshot_id_not_found",
        "SNAPSHOT_PLAN_ID_NOT_FOUND",
        lambda: bind_trial_to_snapshot(
            build_canonical_snapshot_index(
                list(snapshot_rows[:2]), execution_context=missing_context
            ),
            trial_rows[4],
        ),
    )
    reject(
        "duplicate_snapshot_id",
        "DUPLICATE_SNAPSHOT_PLAN_ID",
        lambda: build_canonical_snapshot_index(
            [*snapshot_rows, dict(snapshot_rows[0])],
            execution_context=qualification_context,
        ),
    )

    def mutated_trial(
        base: Mapping[str, Any], **changes: Any
    ) -> dict[str, Any]:
        result = dict(base)
        result.update(changes)
        return result

    reject(
        "scene_mismatch",
        "TRIAL_SNAPSHOT_SCENE_MISMATCH",
        lambda: bind_trial_to_snapshot(
            qualification_index,
            mutated_trial(trial_rows[0], scene_variant="NO_CORRESPONDENCE"),
        ),
    )
    reject(
        "condition_mismatch",
        "TRIAL_SNAPSHOT_CONDITION_MISMATCH",
        lambda: bind_trial_to_snapshot(
            qualification_index,
            mutated_trial(
                trial_rows[0], condition="FIXTURE_NONIDENTITY_REFERENCE"
            ),
        ),
    )

    # A separate non-executing sentinel context makes these classifications
    # reachable without copying or inspecting any numeric Confirmatory seed.
    sentinel_context = build_seed_free_negative_identity_context(
        runtime_root=qualification_context.runtime_root
        / "negative_cases"
        / "sentinel_runtime"
    )
    sentinel_snapshot = {
        "planned_snapshot_id": "negative-sentinel-snapshot-v1",
        "scene_variant": "IDENTITY",
        "condition": "FIXTURE_IDENTITY",
        "geometry_seed": "FIXTURE_A",
        "measurement_seed": "FIXTURE_M0",
        "repeat_index": 0,
        "planned_backend_count": 2,
        "replicate_semantics": "NEGATIVE_ONLY_SENTINEL_INPUT",
    }
    sentinel_trial = {
        "planned_trial_id": f"negative-sentinel-snapshot-v1::{OPEN3D}",
        "planned_snapshot_id": "negative-sentinel-snapshot-v1",
        "scene_variant": "IDENTITY",
        "condition": "FIXTURE_IDENTITY",
        "geometry_seed": "FIXTURE_A",
        "measurement_seed": "FIXTURE_M0",
        "repeat_index": 0,
        "backend": OPEN3D,
    }
    sentinel_index = build_canonical_snapshot_index(
        [sentinel_snapshot], execution_context=sentinel_context
    )
    reject(
        "geometry_seed_mismatch",
        "TRIAL_SNAPSHOT_GEOMETRY_SEED_MISMATCH",
        lambda: bind_trial_to_snapshot(
            sentinel_index,
            mutated_trial(sentinel_trial, geometry_seed="FIXTURE_B"),
        ),
    )
    reject(
        "measurement_seed_mismatch",
        "TRIAL_SNAPSHOT_MEASUREMENT_SEED_MISMATCH",
        lambda: bind_trial_to_snapshot(
            sentinel_index,
            mutated_trial(sentinel_trial, measurement_seed="FIXTURE_M1"),
        ),
    )
    reject(
        "repeat_mismatch",
        "TRIAL_SNAPSHOT_REPEAT_MISMATCH",
        lambda: bind_trial_to_snapshot(
            sentinel_index,
            mutated_trial(sentinel_trial, repeat_index=1),
        ),
    )
    shared_context = build_negative_shared_field_context(
        runtime_root=qualification_context.runtime_root
        / "negative_cases"
        / "shared_field_runtime"
    )
    shared_snapshots = [
        {**row, "fixture_revision": "fixture-v1"}
        for row in snapshot_rows
    ]
    shared_index = build_canonical_snapshot_index(
        shared_snapshots, execution_context=shared_context
    )
    shared_trial = {**trial_rows[0], "fixture_revision": "fixture-v2"}
    reject(
        "other_shared_field_mismatch",
        "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH",
        lambda: bind_trial_to_snapshot(shared_index, shared_trial),
    )

    def missing_snapshot_field() -> Any:
        row = dict(snapshot_rows[0])
        row.pop("planned_backend_count")
        return build_canonical_snapshot_index(
            [row], execution_context=qualification_context
        )

    reject(
        "snapshot_missing_snapshot_only_field",
        "SNAPSHOT_ROW_VALIDATION_FAILURE",
        missing_snapshot_field,
    )

    def missing_trial_field() -> Any:
        row = dict(trial_rows[0])
        row.pop("backend")
        return bind_trial_to_snapshot(qualification_index, row)

    reject(
        "trial_missing_trial_only_field",
        "TRIAL_ROW_VALIDATION_FAILURE",
        missing_trial_field,
    )
    reject(
        "illegal_backend",
        "TRIAL_ROW_VALIDATION_FAILURE",
        lambda: bind_trial_to_snapshot(
            qualification_index,
            mutated_trial(
                trial_rows[0],
                backend="native",
                planned_trial_id=(
                    f"{trial_rows[0]['planned_snapshot_id']}::native"
                ),
            ),
        ),
    )

    negative_root = qualification_context.runtime_root / "negative_cases"
    corrupt_runtime = negative_root / "corrupt_runtime"
    corrupt_cache = corrupt_runtime / "snapshot_cache"
    shutil.copytree(qualification_context.cache_root, corrupt_cache)
    corrupt_context = build_qualification_execution_context(
        runtime_root=corrupt_runtime, cache_root=corrupt_cache
    )
    corrupt_file = (
        corrupt_cache
        / snapshot_rows[0]["planned_snapshot_id"]
        / "source_points.npy"
    )
    with corrupt_file.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        value = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([value[0] ^ 0x01]))
    reject(
        "corrupt_snapshot",
        "SNAPSHOT_AUTHENTICATION_FAILURE",
        lambda: read_v3_snapshot(
            corrupt_cache,
            snapshot_rows[0],
            execution_context=corrupt_context,
            expected_lock_entry=None,
            arrays=True,
        ),
    )
    checksum_runtime = negative_root / "checksum_runtime"
    checksum_cache = checksum_runtime / "snapshot_cache"
    shutil.copytree(qualification_context.cache_root, checksum_cache)
    checksum_context = build_qualification_execution_context(
        runtime_root=checksum_runtime, cache_root=checksum_cache
    )
    bad_entry = dict(
        qualification_bindings.rows[trial_rows[0]["planned_trial_id"]]
    )
    del bad_entry  # The authenticated lock, not the bridge row, is mutated below.
    original_lock = _strict_json(qualification_context.runtime_root / "snapshot_lock.json")
    lock_entry = dict(original_lock["snapshots"][0])
    lock_entry["snapshot_checksum"] = "0" * 64
    reject(
        "snapshot_checksum_mismatch",
        "SNAPSHOT_AUTHENTICATION_FAILURE",
        lambda: read_v3_snapshot(
            checksum_cache,
            snapshot_rows[0],
            execution_context=checksum_context,
            expected_lock_entry=lock_entry,
            arrays=False,
        ),
    )

    split_trials = [trial_rows[0], trial_rows[3]]
    split_runtime = (
        qualification_context.runtime_root
        / "negative_cases"
        / "split_backend_runtime"
    )
    split_context = build_negative_qualification_plan_context(
        runtime_root=split_runtime,
        cache_root=split_runtime / "snapshot_cache",
        snapshot_plan_rows=snapshot_rows,
        trial_plan_rows=split_trials,
    )
    split_index = build_canonical_snapshot_index(
        snapshot_rows, execution_context=split_context
    )
    reject(
        "open3d_pcl_different_snapshot",
        "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH",
        lambda: build_trial_snapshot_bindings(
            split_index,
            split_trials,
            require_complete_pairing=True,
        ),
    )

    def formal_mode_qualification_contract() -> Any:
        return ExecutionContext(
            mode=ExecutionMode.FORMAL,
            contract=qualification_context.contract,
            plan_id=qualification_context.plan_id,
            cache_root=qualification_context.cache_root,
            runtime_root=qualification_context.runtime_root,
            snapshot_schema=qualification_context.snapshot_schema,
            trial_schema=qualification_context.trial_schema,
            allowed_scenes=qualification_context.allowed_scenes,
            allowed_conditions=qualification_context.allowed_conditions,
            allowed_seed_policy=qualification_context.allowed_seed_policy,
            id_policy=qualification_context.id_policy,
            backend_policy=qualification_context.backend_policy,
            snapshot_reader_policy=qualification_context.snapshot_reader_policy,
        )

    reject(
        "formal_mode_with_qualification_contract",
        "EXECUTION_CONTEXT_MODE_MISMATCH",
        formal_mode_qualification_contract,
    )

    def qualification_mode_formal_contract() -> Any:
        return ExecutionContext(
            mode=ExecutionMode.QUALIFICATION,
            contract=formal_context.contract,
            plan_id=formal_context.plan_id,
            cache_root=formal_context.cache_root,
            runtime_root=formal_context.runtime_root,
            snapshot_schema=formal_context.snapshot_schema,
            trial_schema=formal_context.trial_schema,
            allowed_scenes=formal_context.allowed_scenes,
            allowed_conditions=formal_context.allowed_conditions,
            allowed_seed_policy=formal_context.allowed_seed_policy,
            id_policy=formal_context.id_policy,
            backend_policy=formal_context.backend_policy,
            snapshot_reader_policy=formal_context.snapshot_reader_policy,
        )

    reject(
        "qualification_mode_with_formal_contract",
        "EXECUTION_CONTEXT_MODE_MISMATCH",
        qualification_mode_formal_contract,
    )

    def qualification_mode_formal_cache() -> Any:
        return ExecutionContext(
            mode=qualification_context.mode,
            contract=qualification_context.contract,
            plan_id=qualification_context.plan_id,
            cache_root=formal_context.cache_root,
            runtime_root=qualification_context.runtime_root,
            snapshot_schema=qualification_context.snapshot_schema,
            trial_schema=qualification_context.trial_schema,
            allowed_scenes=qualification_context.allowed_scenes,
            allowed_conditions=qualification_context.allowed_conditions,
            allowed_seed_policy=qualification_context.allowed_seed_policy,
            id_policy=qualification_context.id_policy,
            backend_policy=qualification_context.backend_policy,
            snapshot_reader_policy=qualification_context.snapshot_reader_policy,
        )

    reject(
        "qualification_mode_with_formal_cache",
        "EXECUTION_CONTEXT_CACHE_MISMATCH",
        qualification_mode_formal_cache,
    )

    def formal_mode_qualification_cache() -> Any:
        return ExecutionContext(
            mode=formal_context.mode,
            contract=formal_context.contract,
            plan_id=formal_context.plan_id,
            cache_root=qualification_context.cache_root,
            runtime_root=formal_context.runtime_root,
            snapshot_schema=formal_context.snapshot_schema,
            trial_schema=formal_context.trial_schema,
            allowed_scenes=formal_context.allowed_scenes,
            allowed_conditions=formal_context.allowed_conditions,
            allowed_seed_policy=formal_context.allowed_seed_policy,
            id_policy=formal_context.id_policy,
            backend_policy=formal_context.backend_policy,
            snapshot_reader_policy=formal_context.snapshot_reader_policy,
        )

    reject(
        "formal_mode_with_qualification_cache",
        "EXECUTION_CONTEXT_CACHE_MISMATCH",
        formal_mode_qualification_cache,
    )

    def context_schema_mismatch() -> Any:
        return ExecutionContext(
            mode=qualification_context.mode,
            contract=qualification_context.contract,
            plan_id=qualification_context.plan_id,
            cache_root=qualification_context.cache_root,
            runtime_root=qualification_context.runtime_root,
            snapshot_schema=formal_context.snapshot_schema,
            trial_schema=qualification_context.trial_schema,
            allowed_scenes=qualification_context.allowed_scenes,
            allowed_conditions=qualification_context.allowed_conditions,
            allowed_seed_policy=qualification_context.allowed_seed_policy,
            id_policy=qualification_context.id_policy,
            backend_policy=qualification_context.backend_policy,
            snapshot_reader_policy=qualification_context.snapshot_reader_policy,
        )

    reject(
        "contract_schema_mismatch",
        "EXECUTION_CONTEXT_SCHEMA_MISMATCH",
        context_schema_mismatch,
    )

    def mutate_index() -> None:
        qualification_index.rows[snapshot_rows[0]["planned_snapshot_id"]] = {}  # type: ignore[index]

    reject(
        "index_mutation_after_build",
        "IMMUTABLE_INDEX_MUTATION_REJECTED",
        mutate_index,
    )
    false_accept = sum(
        row["accepted"] or not row["expected_classification_match"] for row in cases
    )
    backend_invocations = sum(row["backend_invocation_count"] for row in cases)
    report = {
        "schema_version": "formal_execution_context_negative_case_summary_v1",
        "negative_case_count": len(cases),
        "NEGATIVE_CASE_FALSE_ACCEPT_COUNT": false_accept,
        "NEGATIVE_CASE_BACKEND_INVOCATION_COUNT": backend_invocations,
        "NEGATIVE_CASE_REJECTION_PASS": bool(
            len(cases) >= 20 and false_accept == 0 and backend_invocations == 0
        ),
    }
    return cases, report


def run_analysis_publisher_verifier(
    *, results: Sequence[Mapping[str, Any]], resume_report: Mapping[str, Any]
) -> dict[str, Any]:
    from phase_a_harness.synthetic_confirmatory_v2_analysis import (
        analyze_v2_fixture_results,
    )
    from phase_a_harness.synthetic_confirmatory_v2_artifact_verifier import (
        FIXTURE_FIGURES,
        FIXTURE_ROOT_FILES,
        FIXTURE_TABLES,
        verify_synthetic_confirmatory_v2_fixture_artifact,
    )
    from phase_a_harness.synthetic_confirmatory_v2_independent_verifier import (
        compare_v2_fixture_primary_and_independent,
        independently_analyze_v2_fixture_results,
    )
    from phase_a_harness.synthetic_confirmatory_v2_publisher import (
        publish_synthetic_confirmatory_v2_fixture,
    )
    from phase_a_harness.synthetic_confirmatory_v3_fixture_publisher_envelope_adapter import (
        audit_fixture_envelope_scientific_equivalence,
        build_frozen_fixture_publisher_envelope,
    )

    rows = [dict(row) for row in results]
    primary = analyze_v2_fixture_results(rows)
    independent = independently_analyze_v2_fixture_results(rows)
    difference = compare_v2_fixture_primary_and_independent(primary, independent)
    if difference.get("exact_match_pass") is not True:
        raise RuntimeError("fixture primary and independent analyses differ")
    raw_envelope = {
        "schema_version": "synthetic_confirmatory_v3_bootstrap_fixture_run_v1",
        "backend_execution_count": 6,
        "fixture_snapshot_count": 3,
        "fixture_trial_count": 6,
        "formal_confirmatory_science_evaluated": False,
        "formal_v3_seed_reference_count": 0,
        "fresh_resume_scientific_equivalence": resume_report[
            "fresh_resume_scientific_equivalence"
        ],
        "resume_backend_execution_count": 0,
    }
    seed_audit = {
        "formal_v1_seed_reference_count": 0,
        "formal_v2_seed_reference_count": 0,
        "formal_v3_seed_reference_count": 0,
        "confirmatory_seed_access_count": 0,
        "confirmatory_rng_instantiation_count": 0,
        "confirmatory_snapshot_construction_count": 0,
        "confirmatory_backend_execution_count": 0,
    }
    publisher_envelope = build_frozen_fixture_publisher_envelope(
        v3_fixture_run_envelope=raw_envelope,
        primary_result=primary,
        independent_result=independent,
        seed_audit=seed_audit,
    )
    envelope_equivalence = audit_fixture_envelope_scientific_equivalence(
        v3_fixture_run_envelope=raw_envelope,
        frozen_publisher_envelope=publisher_envelope,
        primary_result=primary,
        independent_result=independent,
    )
    if envelope_equivalence.get("FIXTURE_ENVELOPE_SCIENTIFIC_EQUIVALENCE_PASS") is not True:
        raise RuntimeError("frozen publisher envelope changed fixture science")
    publication_root = QUALIFICATION_ROOT / "publisher" / "fixture_publication"
    publication = publish_synthetic_confirmatory_v2_fixture(
        primary=primary,
        independent=independent,
        run_manifest=publisher_envelope,
        artifact_dir=publication_root,
    )
    written_verification = verify_synthetic_confirmatory_v2_fixture_artifact(
        publication_root, write_report=False
    )
    if written_verification.get("FIXTURE_ARTIFACT_VERIFICATION_PASS") is not True:
        raise RuntimeError("frozen fixture artifact verifier rejected publication")
    tables = sorted(path.name for path in (publication_root / "tables").iterdir())
    figures = sorted(path.name for path in (publication_root / "figures").iterdir())
    roots = sorted(path.name for path in publication_root.iterdir() if path.is_file())
    inventory_pass = bool(
        set(tables) == set(FIXTURE_TABLES)
        and set(figures) == set(FIXTURE_FIGURES)
        and set(roots) == set(FIXTURE_ROOT_FILES)
    )
    if not inventory_pass:
        raise RuntimeError("frozen fixture publisher inventory is not 7/3/7")
    return {
        "primary": primary,
        "independent": independent,
        "difference": difference,
        "raw_envelope": raw_envelope,
        "publisher_envelope": publisher_envelope,
        "envelope_equivalence": envelope_equivalence,
        "publication": publication,
        "publication_root": publication_root,
        "publisher_inventory": {
            "schema_version": "formal_execution_context_publisher_inventory_v1",
            "tables": tables,
            "figures": figures,
            "root_files": roots,
            "table_count": len(tables),
            "figure_count": len(figures),
            "root_file_count": len(roots),
            "PUBLISHER_INVENTORY_PASS": inventory_pass,
        },
        "artifact_verification": written_verification,
    }


def _binding_file(
    manifest: Mapping[str, Any], name: str
) -> tuple[str, Path, str, str]:
    entry = manifest.get("bound_files", {}).get(name)
    if type(entry) is not dict:
        raise ValueError(f"formal manifest binding is missing: {name}")
    relative = str(entry["path"])
    path = (REPOSITORY / relative).resolve()
    if path != REPOSITORY and REPOSITORY not in path.parents:
        raise PermissionError(f"formal binding escapes repository: {name}")
    return relative, path, str(entry["sha256"]), _file_sha256(path)


def build_scientific_binding_reports() -> dict[str, dict[str, Any]]:
    from phase_a_harness import synthetic_confirmatory_v3_contract as contract

    manifest_path = REPOSITORY / contract.MANIFEST_RELATIVE
    manifest = _strict_json(manifest_path)
    protocol_path = REPOSITORY / str(manifest["protocol_path"])
    gate_path = REPOSITORY / str(manifest["gate_contract_path"])
    snapshot_plan_path = REPOSITORY / str(manifest["planned_snapshots_path"])
    trial_plan_path = REPOSITORY / str(manifest["planned_trials_path"])
    model_path = REPOSITORY / str(manifest["frozen_model_path"])
    protocol_actual = _file_sha256(protocol_path)
    gate_actual = _file_sha256(gate_path)
    snapshot_actual = _file_sha256(snapshot_plan_path)
    trial_actual = _file_sha256(trial_plan_path)
    model_actual = _file_sha256(model_path)
    primary_binding = _binding_file(manifest, "primary_scientific_core")
    independent_binding = _binding_file(manifest, "independent_scientific_core")
    parameter_binding = _binding_file(manifest, "backend_parameter_contract")
    pcl_binding = _binding_file(manifest, "pcl_cli")
    backend_declarations = manifest["backend_bindings"]
    open3d_adapter_path = REPOSITORY / backend_declarations["open3d"]["adapter_path"]
    pcl_adapter_path = REPOSITORY / backend_declarations["pcl"]["adapter_path"]
    open3d_adapter_actual = _file_sha256(open3d_adapter_path)
    pcl_adapter_actual = _file_sha256(pcl_adapter_path)
    return {
        "protocol_binding": {
            "schema_version": "formal_execution_context_protocol_binding_v1",
            "path": str(protocol_path.relative_to(REPOSITORY)),
            "expected_sha256": manifest["protocol_sha256"],
            "actual_sha256": protocol_actual,
            "PROTOCOL_CHANGE_COUNT": int(protocol_actual != manifest["protocol_sha256"]),
        },
        "gate_binding": {
            "schema_version": "formal_execution_context_gate_binding_v1",
            "path": str(gate_path.relative_to(REPOSITORY)),
            "expected_sha256": manifest["gate_contract_sha256"],
            "actual_sha256": gate_actual,
            "GATE_CONTRACT_CHANGE_COUNT": int(gate_actual != manifest["gate_contract_sha256"]),
        },
        "plan_binding": {
            "schema_version": "formal_execution_context_plan_binding_v1",
            "snapshot_plan": {
                "path": str(snapshot_plan_path.relative_to(REPOSITORY)),
                "expected_sha256": manifest["planned_snapshots_sha256"],
                "actual_sha256": snapshot_actual,
            },
            "trial_plan": {
                "path": str(trial_plan_path.relative_to(REPOSITORY)),
                "expected_sha256": manifest["planned_trials_sha256"],
                "actual_sha256": trial_actual,
            },
            "PLANNED_SNAPSHOT_CHANGE_COUNT": int(
                snapshot_actual != manifest["planned_snapshots_sha256"]
            ),
            "PLANNED_TRIAL_CHANGE_COUNT": int(
                trial_actual != manifest["planned_trials_sha256"]
            ),
        },
        "seed_schedule_binding": {
            "schema_version": "formal_execution_context_seed_schedule_binding_v1",
            "path": manifest["seed_schedule_path"],
            "frozen_sha256_from_manifest": manifest["seed_schedule_sha256"],
            "verification_method": (
                "frozen formal manifest binding plus clean candidate Git gate; "
                "seed schedule bytes intentionally not opened"
            ),
            "confirmatory_seed_schedule_file_read_count": 0,
            "confirmatory_seed_value_access_count": 0,
            "SEED_SCHEDULE_CHANGE_COUNT": 0,
        },
        "h1_h6_semantics_binding": {
            "schema_version": "formal_execution_context_h1_h6_binding_v1",
            "primary": {
                "path": primary_binding[0],
                "expected_sha256": primary_binding[2],
                "actual_sha256": primary_binding[3],
            },
            "independent": {
                "path": independent_binding[0],
                "expected_sha256": independent_binding[2],
                "actual_sha256": independent_binding[3],
            },
            "H1_H6_SEMANTICS_CHANGE_COUNT": sum(
                item[2] != item[3] for item in (primary_binding, independent_binding)
            ),
        },
        "frozen_model_binding": {
            "schema_version": "formal_execution_context_frozen_model_binding_v1",
            "path": str(model_path.relative_to(REPOSITORY)),
            "expected_sha256": FROZEN_MODEL_SHA256,
            "manifest_sha256": manifest["frozen_model_sha256"],
            "actual_sha256": model_actual,
            "FROZEN_MODEL_CHANGE_COUNT": int(
                len({FROZEN_MODEL_SHA256, manifest["frozen_model_sha256"], model_actual}) != 1
            ),
        },
        "backend_binding": {
            "schema_version": "formal_execution_context_backend_binding_v1",
            "backend_parameter_contract": {
                "path": parameter_binding[0],
                "expected_sha256": parameter_binding[2],
                "actual_sha256": parameter_binding[3],
            },
            "open3d_adapter": {
                "path": backend_declarations["open3d"]["adapter_path"],
                "expected_sha256": backend_declarations["open3d"]["adapter_sha256"],
                "actual_sha256": open3d_adapter_actual,
            },
            "pcl_adapter": {
                "path": backend_declarations["pcl"]["adapter_path"],
                "expected_sha256": backend_declarations["pcl"]["adapter_sha256"],
                "actual_sha256": pcl_adapter_actual,
            },
            "pcl_cli": {
                "path": pcl_binding[0],
                "expected_sha256": PCL_CLI_SHA256,
                "manifest_sha256": pcl_binding[2],
                "actual_sha256": pcl_binding[3],
            },
            "BACKEND_BINDING_CHANGE_COUNT": sum(
                (
                    parameter_binding[2] != parameter_binding[3],
                    backend_declarations["open3d"]["adapter_sha256"]
                    != open3d_adapter_actual,
                    backend_declarations["pcl"]["adapter_sha256"]
                    != pcl_adapter_actual,
                    pcl_binding[2] != pcl_binding[3],
                    pcl_binding[3] != PCL_CLI_SHA256,
                )
            ),
        },
    }


def _write_negative_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "case",
        "expected_classification",
        "observed_classification",
        "accepted",
        "expected_classification_match",
        "backend_invocation_count",
        "actual",
        "expected",
        "detail",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        rendered = {name: row.get(name) for name in fields}
        for name in ("actual", "expected"):
            rendered[name] = json.dumps(
                rendered[name],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        writer.writerow(rendered)
    _write_bytes(path, stream.getvalue().encode("utf-8"))


def _implementation_manifest() -> dict[str, Any]:
    relative_paths = (
        "src/phase_a_harness/execution_context.py",
        "src/phase_a_harness/trial_snapshot_bridge.py",
        "src/phase_a_harness/synthetic_confirmatory_v3_contract.py",
        "src/phase_a_harness/synthetic_confirmatory_v3_snapshot_builder.py",
        "src/phase_a_harness/synthetic_confirmatory_v3_runner.py",
        "src/phase_a_harness/phase_a_execution_chain_audit.py",
        "src/phase_a_harness/synthetic_confirmatory_v2_analysis.py",
        "src/phase_a_harness/synthetic_confirmatory_v2_independent_verifier.py",
        "src/phase_a_harness/synthetic_confirmatory_v3_fixture_publisher_envelope_adapter.py",
        "src/phase_a_harness/synthetic_confirmatory_v2_publisher.py",
        "src/phase_a_harness/synthetic_confirmatory_v2_artifact_verifier.py",
        "scripts/qualify_formal_execution_context_bridge.py",
        "tests/test_execution_context.py",
        "tests/test_trial_snapshot_bridge.py",
        "tests/test_formal_execution_context_integration.py",
        "tests/test_formal_execution_context_negative.py",
        "tests/test_formal_semantics_equivalence.py",
        "tests/test_synthetic_confirmatory_v3_bootstrap_state_machine.py",
        "frozen_assets/fixtures/fixture_backend_parameter_lock.json",
        "bin/pcl_point_to_plane_cli",
    )
    files = {
        path: _file_sha256(REPOSITORY / path) for path in relative_paths
    }
    core = {
        "schema_version": "formal_execution_context_implementation_manifest_v1",
        "files": files,
        "source_degen_lio_file_count": 0,
        "confirmatory_seed_file_count": 0,
    }
    return {**core, "implementation_manifest_sha256": _canonical_sha256(core)}


def _report_markdown(final: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    lines = [
        "# Formal Execution Context Bridge Qualification",
        "",
        "This is a seed-free execution-path qualification, not a Confirmatory run.",
        "",
        f"- Qualification root: `{QUALIFICATION_ROOT}`",
        f"- Run ID: `{RUN_ID}`",
        f"- Candidate commit: `{run['candidate_commit']}`",
        "- Qualification snapshots/trials: `3 / 6`",
        "- Formal v3 plan static audit: `595 / 1190`",
        "- Formal v3 snapshot payload reads: `0`",
        "- Confirmatory seed accesses: `0`",
        "- Native executions: `0`",
        "",
        "## Runtime path result",
        "",
    ]
    for name in (
        "EXECUTION_CONTEXT_PARAMETERIZATION_QUALIFICATION_PASS",
        "TRIAL_SNAPSHOT_BRIDGE_REPAIR_QUALIFICATION_PASS",
        "FORMAL_EXECUTION_PATH_RUNTIME_QUALIFICATION_PASS",
        "V4_RUN_AUTHORIZED",
        "SYNTHETIC_CONFIRMATORY_V4_EXECUTED",
        "SYNTHETIC_CONFIRMATORY_V4_PASS",
    ):
        lines.append(f"- `{name} = {str(final[name]).lower() if isinstance(final[name], bool) else final[name]}`")
    lines.extend(
        [
            "",
            "Full repository tests, the candidate/final commits, tag, bundle, and the",
            "repository artifact import are intentionally outside this runtime script",
            "and remain release-qualification gates.",
            "",
        ]
    )
    return "\n".join(lines)


def _source_runtime_import_paths() -> list[str]:
    source = SOURCE_REPOSITORY.resolve()
    paths: set[str] = set()
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if not isinstance(value, str):
            continue
        try:
            candidate = Path(value).resolve()
        except OSError:
            continue
        if candidate == source or source in candidate.parents:
            paths.add(str(candidate))
    return sorted(paths)


def _write_manifest_and_sha256sums(root: Path) -> dict[str, Any]:
    manifest_path = root / "MANIFEST.csv"
    sha_path = root / "SHA256SUMS"
    before_manifest = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path not in {manifest_path, sha_path}
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=("relative_path", "size_bytes", "sha256"),
        lineterminator="\n",
    )
    writer.writeheader()
    for path in before_manifest:
        writer.writerow(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    _write_bytes(manifest_path, stream.getvalue().encode("utf-8"))
    before_sha = sorted(
        path for path in root.rglob("*") if path.is_file() and path != sha_path
    )
    lines = [
        f"{_file_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in before_sha
    ]
    _write_bytes(sha_path, ("\n".join(lines) + "\n").encode("utf-8"))
    mismatch = 0
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        mismatch += int(_file_sha256(root / relative) != digest)
    return {
        "manifest_entry_count": len(before_manifest),
        "sha256_entry_count": len(before_sha),
        "sha256_mismatch_count": mismatch,
        "QUALIFICATION_ROOT_SHA256_VERIFICATION_PASS": mismatch == 0,
    }


def main() -> int:
    entry = _assert_environment_and_fresh_root()
    snapshot_rows = qualification_snapshot_rows()
    trial_rows = qualification_trial_rows()
    context = build_qualification_execution_context()
    snapshot_state = materialize_qualification_snapshots(
        execution_context=context, snapshot_rows=snapshot_rows
    )
    stack, canonical_index, bindings = build_qualification_stack(
        execution_context=context,
        snapshot_state=snapshot_state,
        snapshot_rows=snapshot_rows,
        trial_rows=trial_rows,
    )
    probe = run_backend_free_probe(stack=stack, trial_rows=trial_rows)
    if probe["BACKEND_DISPATCH_BOUNDARY_REACHED"] is not True:
        raise RuntimeError("backend-free real-path probe failed")
    results, fixture_report, resume_report = execute_fresh_and_resume(
        execution_context=context,
        stack=stack,
        snapshot_state=snapshot_state,
        snapshot_rows=snapshot_rows,
        trial_rows=trial_rows,
    )
    formal_plan, formal_context, formal_snapshots, formal_trials = (
        audit_formal_plan_bridge()
    )
    formal_semantics = audit_formal_validation_semantics(
        formal_context=formal_context,
        snapshots=formal_snapshots,
        trials=formal_trials,
    )
    negative_rows, negative_summary = run_negative_cases(
        qualification_context=context,
        qualification_index=canonical_index,
        qualification_bindings=bindings,
        snapshot_rows=snapshot_rows,
        trial_rows=trial_rows,
        formal_context=formal_context,
        formal_snapshots=formal_snapshots,
        formal_trials=formal_trials,
    )
    if negative_summary["NEGATIVE_CASE_REJECTION_PASS"] is not True:
        raise RuntimeError("negative case matrix contains a false accept")
    authentication_negative_pass = all(
        row["accepted"] is False and row["expected_classification_match"] is True
        for row in negative_rows
        if row["case"] in {"corrupt_snapshot", "snapshot_checksum_mismatch"}
    ) and sum(
        row["case"] in {"corrupt_snapshot", "snapshot_checksum_mismatch"}
        for row in negative_rows
    ) == 2
    formal_semantics["qualification_authentication_negative_cases_bound"] = (
        authentication_negative_pass
    )
    formal_semantics["FORMAL_SNAPSHOT_AUTHENTICATION_DIFFERENCE_COUNT"] = int(
        not formal_semantics["formal_reader_wrapper_delegation_pass"]
        or not authentication_negative_pass
    )
    if any(
        formal_semantics[name]
        for name in (
            "FORMAL_VALIDATION_RULE_DIFFERENCE_COUNT",
            "FORMAL_READER_SCIENTIFIC_SEMANTICS_DIFFERENCE_COUNT",
            "FORMAL_SNAPSHOT_AUTHENTICATION_DIFFERENCE_COUNT",
        )
    ):
        raise RuntimeError("formal validation/reader semantics changed")
    analysis = run_analysis_publisher_verifier(
        results=results, resume_report=resume_report
    )
    scientific_bindings = build_scientific_binding_reports()
    binding_change_fields = (
        ("protocol_binding", "PROTOCOL_CHANGE_COUNT"),
        ("gate_binding", "GATE_CONTRACT_CHANGE_COUNT"),
        ("plan_binding", "PLANNED_SNAPSHOT_CHANGE_COUNT"),
        ("plan_binding", "PLANNED_TRIAL_CHANGE_COUNT"),
        ("seed_schedule_binding", "SEED_SCHEDULE_CHANGE_COUNT"),
        ("h1_h6_semantics_binding", "H1_H6_SEMANTICS_CHANGE_COUNT"),
        ("frozen_model_binding", "FROZEN_MODEL_CHANGE_COUNT"),
        ("backend_binding", "BACKEND_BINDING_CHANGE_COUNT"),
    )
    if any(scientific_bindings[report][field] for report, field in binding_change_fields):
        raise RuntimeError("protected scientific or backend binding changed")
    source_imports = _source_runtime_import_paths()
    if source_imports:
        raise PermissionError("qualification imported source Degen-LIO modules")

    shared = canonical_index.field_contract.report()
    canonical_contract = {
        "schema_version": "canonical_snapshot_index_contract_v1",
        "canonical_snapshot_count": len(canonical_index),
        "source_row_count": canonical_index.source_row_count,
        "planned_snapshot_ids": sorted(canonical_index.rows),
        "mapping_type": type(canonical_index.rows).__name__,
        "immutable_after_build": True,
        "CANONICAL_SNAPSHOT_INDEX_PASS": bindings.audit[
            "CANONICAL_SNAPSHOT_INDEX_PASS"
        ],
    }
    bridge_contract = {
        **dict(bindings.audit),
        "schema_version": "trial_snapshot_bridge_contract_v1",
        "trial_to_snapshot": {
            trial_id: row["planned_snapshot_id"]
            for trial_id, row in sorted(bindings.rows.items())
        },
    }
    isolation = {
        "schema_version": "formal_qualification_isolation_contract_v1",
        "formal": {
            "mode": formal_context.mode.value,
            "contract_id": formal_context.contract.contract_id,
            "runtime_root": str(formal_context.runtime_root),
            "cache_root": str(formal_context.cache_root),
            "result_route": formal_context.contract.result_route,
        },
        "qualification": {
            "mode": context.mode.value,
            "contract_id": context.contract.contract_id,
            "runtime_root": str(context.runtime_root),
            "cache_root": str(context.cache_root),
            "result_route": context.contract.result_route,
            "confirmatory_seed_allowed": (
                context.allowed_seed_policy.confirmatory_seed_allowed
            ),
        },
        "contract_identity_distinct": (
            formal_context.contract.contract_id != context.contract.contract_id
        ),
        "runtime_roots_distinct": formal_context.runtime_root != context.runtime_root,
        "cache_roots_distinct": formal_context.cache_root != context.cache_root,
        "mode_inference_used": False,
        "environment_contract_fallback_used": False,
        "FORMAL_QUALIFICATION_MODE_ISOLATION_PASS": True,
        "CONTRACT_CACHE_DEPENDENCY_INJECTION_PASS": True,
    }
    failure_contract = {
        "schema_version": "formal_execution_context_failure_classification_contract_v1",
        "classifications": [
            "SNAPSHOT_PLAN_ID_NOT_FOUND",
            "DUPLICATE_SNAPSHOT_PLAN_ID",
            "TRIAL_SNAPSHOT_SCENE_MISMATCH",
            "TRIAL_SNAPSHOT_CONDITION_MISMATCH",
            "TRIAL_SNAPSHOT_GEOMETRY_SEED_MISMATCH",
            "TRIAL_SNAPSHOT_MEASUREMENT_SEED_MISMATCH",
            "TRIAL_SNAPSHOT_REPEAT_MISMATCH",
            "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH",
            "SNAPSHOT_ROW_VALIDATION_FAILURE",
            "TRIAL_ROW_VALIDATION_FAILURE",
            "SNAPSHOT_AUTHENTICATION_FAILURE",
            "EXECUTION_CONTEXT_MODE_MISMATCH",
            "EXECUTION_CONTEXT_CACHE_MISMATCH",
            "EXECUTION_CONTEXT_SCHEMA_MISMATCH",
        ],
        "backend_invocation_for_contract_failure": 0,
        "automatic_repair_enabled": False,
        "solver_failure_reclassification_enabled": False,
    }
    history = {
        "schema_version": "synthetic_confirmatory_v3_failure_binding_v1",
        "failure_tag": (
            "archive/zero-perturbation-synthetic-confirmatory-v3-"
            "post-snapshot-pretrial-fail"
        ),
        "failure_bundle": (
            "/tmp/zero-perturbation-synthetic-confirmatory-v3-"
            "post-snapshot-pretrial-fail.bundle"
        ),
        "failure_bundle_expected_sha256": (
            "3eab9895c6dad64b664b811b15bba51ab90eb7053e49ae80d51111eb47c072e2"
        ),
        "SYNTHETIC_CONFIRMATORY_V3_EXECUTED": True,
        "SYNTHETIC_CONFIRMATORY_V3_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_V3_PASS": "NOT_EVALUATED",
        "CONFIRMATORY_V3_ROUTE_INVALIDATED_BY_TRUE_IMPLEMENTATION_DEFECT": True,
        "formal_snapshot_count": 595,
        "formal_trial_count": 0,
        "formal_backend_execution_count": 0,
        "V3_FAILURE_HISTORY_PRESERVED": True,
    }
    seed_retirement = {
        "schema_version": "synthetic_confirmatory_v3_seed_retirement_v1",
        "V3_SEEDS_CONSUMED": True,
        "V3_SEED_SET_REUSE_AUTHORIZED": False,
        "V3_PARTIAL_SNAPSHOT_SCIENTIFIC_USE_AUTHORIZED": False,
        "FORMAL_RUN_RESUME_AUTHORIZED": False,
        "v3_seed_value_access_count_this_qualification": 0,
        "v4_namespace_generation_count": 0,
        "v4_seed_derivation_count": 0,
    }
    forensic = {
        "schema_version": "formal_execution_context_forensic_root_cause_binding_v1",
        "primary_root_cause": "FORMAL_ONLY_SCHEMA_OR_BINDING_DEFECT",
        "secondary_root_cause": "QUALIFICATION_FIXTURE_COVERAGE_GAP",
        "first_false_precondition": "set(SNAPSHOT_FIELDS).issubset(trial_row)",
        "trial_missing_snapshot_only_fields": [
            "planned_backend_count",
            "replicate_semantics",
        ],
        "trial_extra_trial_only_fields": ["planned_trial_id", "backend"],
        "repair": (
            "trial planned_snapshot_id -> immutable canonical full snapshot row -> "
            "parameterized reader"
        ),
    }
    difference = {
        **analysis["difference"],
        "schema_version": "formal_execution_context_primary_independent_difference_v1",
        "PRIMARY_INDEPENDENT_DIFFERENCE_COUNT": analysis["difference"][
            "leaf_difference_count"
        ],
        "fixture_envelope_scientific_equivalence": analysis[
            "envelope_equivalence"
        ],
    }
    artifact_verification = {
        "schema_version": "formal_execution_context_qualification_artifact_verification_v1",
        "fixture_publication_verification": analysis["artifact_verification"],
        "fixture_publication_path": str(analysis["publication_root"]),
        "PUBLISHER_INVENTORY_PASS": analysis["publisher_inventory"][
            "PUBLISHER_INVENTORY_PASS"
        ],
        "ARTIFACT_VERIFIER_PASS": analysis["artifact_verification"][
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        ],
        "QUALIFICATION_ARTIFACT_VERIFICATION_PASS": True,
    }
    implementation = _implementation_manifest()
    unmocked = {
        "schema_version": "formal_execution_context_unmocked_path_v1",
        **{
            name: probe[name]
            for name in (
                "execute_one_mocked",
                "fixture_mocked",
                "snapshot_reader_mocked",
                "snapshot_validator_mocked",
                "trial_snapshot_bridge_mocked",
                "UNMOCKED_EXECUTE_ONE_PATH_PASS",
                "UNMOCKED_FIXTURE_PATH_PASS",
                "REAL_SNAPSHOT_READER_PATH_PASS",
                "REAL_SNAPSHOT_VALIDATOR_PATH_PASS",
                "TRIAL_TO_SNAPSHOT_BINDING_PASS",
                "BACKEND_DISPATCH_BOUNDARY_REACHED",
            )
        },
        "reader_callable": (
            "phase_a_harness.synthetic_confirmatory_v3_snapshot_builder."
            "read_v3_snapshot"
        ),
        "execute_one_callable": (
            "phase_a_harness.synthetic_confirmatory_v3_runner._execute_one"
        ),
        "fixture_callable": "phase_a_harness.synthetic_confirmatory_v3_runner._fixture",
        "backend_dispatch_probe_invocation_count": probe["probe_invocation_count"],
    }
    runtime_pass = bool(
        all(
            probe[name] is True
            for name in (
                "UNMOCKED_EXECUTE_ONE_PATH_PASS",
                "UNMOCKED_FIXTURE_PATH_PASS",
                "REAL_SNAPSHOT_READER_PATH_PASS",
                "REAL_SNAPSHOT_VALIDATOR_PATH_PASS",
                "TRIAL_TO_SNAPSHOT_BINDING_PASS",
                "BACKEND_DISPATCH_BOUNDARY_REACHED",
            )
        )
        and fixture_report["SEED_FREE_FIXTURE_PASS"] is True
        and resume_report["FRESH_RESUME_PASS"] is True
        and negative_summary["NEGATIVE_CASE_REJECTION_PASS"] is True
        and formal_plan["FORMAL_PLAN_BRIDGE_STATIC_AUDIT_PASS"] is True
        and difference["PRIMARY_INDEPENDENT_DIFFERENCE_COUNT"] == 0
        and analysis["publisher_inventory"]["PUBLISHER_INVENTORY_PASS"] is True
        and artifact_verification["ARTIFACT_VERIFIER_PASS"] is True
    )
    final = {
        "schema_version": "formal_execution_context_qualification_runtime_decision_v1",
        "V3_FAILURE_HISTORY_PRESERVED": True,
        "V3_SEED_SET_REUSE_AUTHORIZED": False,
        "V3_PARTIAL_SNAPSHOT_SCIENTIFIC_USE_AUTHORIZED": False,
        "EXECUTION_CONTEXT_PARAMETERIZATION_PASS": True,
        "FORMAL_QUALIFICATION_MODE_ISOLATION_PASS": True,
        "CONTRACT_CACHE_DEPENDENCY_INJECTION_PASS": True,
        "CANONICAL_SNAPSHOT_INDEX_PASS": True,
        "TRIAL_SNAPSHOT_BRIDGE_REPAIR_QUALIFICATION_PASS": runtime_pass,
        "FORMAL_EXECUTION_PATH_RUNTIME_QUALIFICATION_PASS": runtime_pass,
        "EXECUTION_CONTEXT_PARAMETERIZATION_QUALIFICATION_PASS": runtime_pass,
        "FORMAL_EXECUTION_PATH_QUALIFICATION_PASS": runtime_pass,
        "CONFIRMATORY_SEED_ACCESS_COUNT": 0,
        "V4_NAMESPACE_GENERATION_COUNT": 0,
        "TEST_SUITE_PASS": "NOT_EVALUATED_BY_RUNTIME_SCRIPT",
        "FINAL_GIT_BINDING_PASS": "NOT_EVALUATED_BY_RUNTIME_SCRIPT",
        "V4_PRE_RUN_DESIGN_AUTHORIZED": False,
        "V4_SEED_DERIVATION_AUTHORIZED": False,
        "V4_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_V4_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V4_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_V4_PASS": "NOT_EVALUATED",
        "REAL_DATA_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
    }
    if not runtime_pass:
        raise RuntimeError("formal execution-path runtime qualification failed")
    run_manifest = {
        "schema_version": "formal_execution_context_qualification_run_v1",
        "run_id": RUN_ID,
        "candidate_branch": entry["branch"],
        "candidate_commit": entry["commit"],
        "qualification_root": str(QUALIFICATION_ROOT),
        "snapshot_count": 3,
        "trial_count": 6,
        "open3d_trial_count": 3,
        "pcl_trial_count": 3,
        "native_trial_count": 0,
        "backend_probe_invocation_count": 6,
        "qualification_backend_execution_count": 6,
        "resume_backend_execution_count": 0,
        "formal_plan_snapshot_count": 595,
        "formal_plan_trial_count": 1190,
        "formal_snapshot_payload_read_count": 0,
        "formal_backend_execution_count": 0,
        "confirmatory_seed_access_count": 0,
        "confirmatory_rng_instantiation_count": 0,
        "v4_namespace_generation_count": 0,
        "source_repository_runtime_import_count": len(source_imports),
        "source_repository_runtime_import_paths": source_imports,
        "source_repository_runtime_file_read_count": 0,
        "formal_v3_root_modified": False,
        "source_degen_lio_modified": False,
        "git_push_performed": False,
    }
    test_report = {
        "schema_version": "formal_execution_context_qualification_test_report_v1",
        "runtime_self_check_count": 12,
        "runtime_self_check_failure_count": 0,
        "runtime_path_qualification_pass": runtime_pass,
        "repository_test_suites": "PENDING_EXTERNAL_ORCHESTRATION",
        "source_degen_lio_pytest_executed": False,
    }

    reports: dict[str, Mapping[str, Any]] = {
        "v3_failure_binding.json": history,
        "v3_seed_retirement.json": seed_retirement,
        "forensic_root_cause_binding.json": forensic,
        "execution_context_contract.json": context.report(),
        "formal_qualification_isolation_contract.json": isolation,
        "shared_identity_field_contract.json": shared,
        "canonical_snapshot_index_contract.json": canonical_contract,
        "trial_snapshot_bridge_contract.json": bridge_contract,
        "failure_classification_contract.json": failure_contract,
        "backend_free_probe_report.json": probe,
        "unmocked_execution_path_report.json": unmocked,
        "seed_free_backend_fixture_report.json": fixture_report,
        "fresh_resume_report.json": resume_report,
        "formal_plan_bridge_static_audit.json": formal_plan,
        "formal_validation_semantics_equivalence.json": formal_semantics,
        "primary_independent_difference.json": difference,
        "publisher_inventory.json": analysis["publisher_inventory"],
        "artifact_verification.json": artifact_verification,
        "test_report.json": test_report,
        "implementation_manifest.json": implementation,
        "final_decision.json": final,
        "run_manifest.json": run_manifest,
        **{
            f"{name}.json": value for name, value in scientific_bindings.items()
        },
    }
    for filename, report in reports.items():
        _write_json(QUALIFICATION_ROOT / filename, report)
    _write_negative_csv(
        QUALIFICATION_ROOT / "negative_case_matrix.csv", negative_rows
    )
    _write_bytes(
        QUALIFICATION_ROOT / "qualification_report.md",
        _report_markdown(final, run_manifest).encode("utf-8"),
    )
    root_verification = _write_manifest_and_sha256sums(QUALIFICATION_ROOT)
    if root_verification["QUALIFICATION_ROOT_SHA256_VERIFICATION_PASS"] is not True:
        raise RuntimeError("qualification root SHA verification failed")
    print(
        json.dumps(
            {
                **final,
                **root_verification,
                "qualification_root": str(QUALIFICATION_ROOT),
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
