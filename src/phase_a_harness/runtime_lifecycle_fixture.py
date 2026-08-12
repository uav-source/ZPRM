"""Seed-free, real-write runtime lifecycle qualification fixture.

This is an execution-contract qualification, not a Confirmatory experiment.
It materializes only the three long-standing deterministic fixture snapshots,
executes the two frozen qualified backends, and keeps every mutable byte below
an externally validated runtime root.  No Confirmatory seed or plan is read.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import file_sha256
from .phase_a_execution_chain_audit import (
    execute_open3d_fixture,
    execute_pcl_fixture,
)
from .phase_a_execution_chain_fixture import FixtureSnapshot, build_fixture_snapshots
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    canonical_json_bytes as trial_json_bytes,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .phase_a_trial_result_writer import result_filename
from .phase_a_trial_resume import (
    CorruptExistingResult,
    validate_existing_trial_result_for_resume,
)
from .runtime_lifecycle_io import (
    SingleWriterLease,
    append_event_v2,
    atomic_create_bytes,
    atomic_create_canonical_json,
    atomic_publish_directory,
    atomic_replace_canonical_json,
    canonical_json_sha256,
    read_canonical_json,
    read_event_journal_v2,
    write_once_immutable_run_lock,
)
from .runtime_path_policy import RuntimePathLayout
from .real_data_preparation.stage2_safe_fixture_replay import (
    Stage2SafeFixtureReplayError,
    record_authenticated_test_double_result,
    replay_fixture_result as replay_stage2_safe_fixture_result,
    safe_fixture_replay_active as stage2_safe_fixture_replay_active,
)


RUN_CONTRACT_SCHEMA = "runtime_lifecycle_fixture_run_contract_v1"
SNAPSHOT_METADATA_SCHEMA = "runtime_lifecycle_fixture_snapshot_v1"
RAW_MANIFEST_SCHEMA = "runtime_lifecycle_fixture_raw_result_manifest_v1"
RUN_REPORT_SCHEMA = "runtime_lifecycle_fixture_execution_report_v1"
BACKENDS = (OPEN3D_BACKEND, PCL_BACKEND)
EXPECTED_SNAPSHOT_COUNT = 3
EXPECTED_TRIAL_COUNT = 6
_ORIGINAL_EXECUTE_OPEN3D_FIXTURE = execute_open3d_fixture
_ORIGINAL_EXECUTE_PCL_FIXTURE = execute_pcl_fixture
_AUTHORIZED_PYTEST_DOUBLE_MODULES = frozenset(
    {"test_runtime_lifecycle_fixture", "tests.test_runtime_lifecycle_fixture"}
)
_AUTHORIZED_PYTEST_DOUBLE_QUALNAME = (
    "_install_fake_backends.<locals>.fake"
)


class RuntimeLifecycleCorruption(RuntimeError):
    """A pre-existing runtime object failed its immutable binding."""

    classification = "CORRUPT_OR_TAMPERED_RUNTIME_OBJECT"


def _execute_or_replay_stage2_fixture(
    *,
    executor: Callable[..., Mapping[str, Any]],
    original: Callable[..., Mapping[str, Any]],
    fixture: FixtureSnapshot,
    common: Mapping[str, Any],
    parameters: Mapping[str, Any],
    pcl_cli: Path | None = None,
) -> dict[str, Any]:
    """Preserve the one frozen pytest double; replay every real adapter."""

    if not stage2_safe_fixture_replay_active():
        arguments: dict[str, Any] = {
            "fixture": fixture,
            "common": common,
            "parameters": parameters,
        }
        if pcl_cli is not None:
            arguments["pcl_cli"] = pcl_cli
        return dict(executor(**arguments))
    if executor is original:
        return replay_stage2_safe_fixture_result(
            fixture=fixture,
            common=common,
            parameters=parameters,
            pcl_cli=pcl_cli,
        )
    if (
        getattr(executor, "__module__", None) not in _AUTHORIZED_PYTEST_DOUBLE_MODULES
        or getattr(executor, "__qualname__", None)
        != _AUTHORIZED_PYTEST_DOUBLE_QUALNAME
    ):
        raise Stage2SafeFixtureReplayError(
            "formal Stage-2 backend adapter was replaced by an unauthorized callable"
        )
    arguments = {
        "fixture": fixture,
        "common": common,
        "parameters": parameters,
    }
    if pcl_cli is not None:
        arguments["pcl_cli"] = pcl_cli
    return record_authenticated_test_double_result(executor(**arguments))


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = read_canonical_json(path)
    except (OSError, ValueError) as error:
        raise RuntimeLifecycleCorruption(f"invalid {label}: {path}") from error
    if type(value) is not dict:
        raise RuntimeLifecycleCorruption(f"{label} must be a JSON object")
    return value


def _raw_array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise RuntimeLifecycleCorruption("snapshot array is not C-contiguous")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _snapshot_token(snapshot_id: str) -> str:
    token = snapshot_id.rsplit("/", 1)[-1]
    if not token or token in {".", ".."} or "/" in token:
        raise ValueError("unsafe fixture snapshot ID")
    return token


def _lineage_expected(fixture: FixtureSnapshot) -> bool:
    return fixture.condition in {
        "FIXTURE_IDENTITY",
        "FIXTURE_NONIDENTITY_REFERENCE",
    }


def _planned_trials(fixtures: Sequence[FixtureSnapshot]) -> list[dict[str, str]]:
    return [
        {
            "backend": backend,
            "condition": fixture.condition,
            "planned_trial_id": f"{fixture.snapshot_id}/{backend}",
            "snapshot_id": fixture.snapshot_id,
        }
        for fixture in fixtures
        for backend in BACKENDS
    ]


def build_fixture_run_contract(
    *,
    repository: str | Path,
    layout: RuntimePathLayout,
    run_id: str,
    workers: int,
    expected_commit: str,
    expected_branch: str,
    expected_tag: str,
    runtime_path_policy_sha256: str,
    qualification_delay_seconds: float,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    if run_id != layout.run_id:
        raise ValueError("run ID differs from runtime path layout")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if qualification_delay_seconds < 0.0:
        raise ValueError("qualification delay must be nonnegative")
    fixtures = build_fixture_snapshots()
    manifest = json.loads(
        (root / "frozen_assets/frozen_experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "fixture_only": True,
        "formal_confirmatory_science_evaluated": False,
        "formal_seed_values_included": False,
        "fixture_generation_rng_count": 0,
        "run_id": run_id,
        "workers": workers,
        "expected_commit": expected_commit,
        "expected_branch": expected_branch,
        "expected_tag": expected_tag,
        "runtime_paths": layout.as_dict(),
        "runtime_path_policy_sha256": runtime_path_policy_sha256,
        "qualification_delay_seconds": float(qualification_delay_seconds),
        "fixture_plan_path": "frozen_assets/fixtures/fixture_plan.json",
        "fixture_plan_sha256": file_sha256(
            root / "frozen_assets/fixtures/fixture_plan.json"
        ),
        "fixture_snapshot_lock_path": (
            "frozen_assets/fixtures/fixture_snapshot_lock.json"
        ),
        "fixture_snapshot_lock_sha256": file_sha256(
            root / "frozen_assets/fixtures/fixture_snapshot_lock.json"
        ),
        "fixture_backend_parameter_lock_path": (
            "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
        ),
        "fixture_backend_parameter_lock_sha256": file_sha256(
            root / "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
        ),
        "implementation_sha256": manifest["manifest_payload_sha256"],
        "pcl_cli_path": "bin/pcl_point_to_plane_cli",
        "pcl_cli_sha256": file_sha256(root / "bin/pcl_point_to_plane_cli"),
        "planned_snapshots": [
            {
                "condition": item.condition,
                "expected_failure_classifications": list(
                    item.expected_failure_classifications
                ),
                "lineage_sidecar_required": _lineage_expected(item),
                "reference_pose_checksum": item.checksums[
                    "reference_pose_checksum"
                ],
                "scene_variant": item.scene_variant,
                "snapshot_checksum": item.checksums["snapshot_checksum"],
                "snapshot_id": item.snapshot_id,
                "source_checksum": item.checksums["source_checksum"],
                "target_checksum": item.checksums["target_checksum"],
            }
            for item in fixtures
        ],
        "planned_trials": _planned_trials(fixtures),
    }


def _snapshot_metadata(
    fixture: FixtureSnapshot, file_hashes: Mapping[str, str]
) -> dict[str, Any]:
    unsigned = {
        "schema_version": SNAPSHOT_METADATA_SCHEMA,
        "fixture_only": True,
        "formal_confirmatory_science_evaluated": False,
        "random_seed_used": False,
        "condition": fixture.condition,
        "scene_variant": fixture.scene_variant,
        "snapshot_id": fixture.snapshot_id,
        "checksums": dict(fixture.checksums),
        "file_sha256": dict(sorted(file_hashes.items())),
        "lineage_sidecar_required": _lineage_expected(fixture),
    }
    return {**unsigned, "metadata_payload_sha256": canonical_json_sha256(unsigned)}


def _populate_snapshot(directory: Path, fixture: FixtureSnapshot) -> None:
    paths = {
        "source_points.npy": fixture.source,
        "target_points.npy": fixture.target,
        "reference_pose.npy": fixture.reference,
    }
    for name, value in paths.items():
        _write_npy(directory / name, value)
    if _lineage_expected(fixture):
        indices = np.arange(len(fixture.source), dtype="<i8")
        _write_npy(directory / "source_parent_target_indices.npy", indices)
    hashes = {
        path.name: file_sha256(path)
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
    }
    atomic_create_canonical_json(
        directory / "metadata.json", _snapshot_metadata(fixture, hashes)
    )


def _expected_snapshot_files(fixture: FixtureSnapshot) -> set[str]:
    result = {
        "metadata.json",
        "reference_pose.npy",
        "source_points.npy",
        "target_points.npy",
    }
    if _lineage_expected(fixture):
        result.add("source_parent_target_indices.npy")
    return result


def validate_runtime_snapshot(
    snapshot_dir: str | Path, fixture: FixtureSnapshot
) -> FixtureSnapshot:
    root = Path(snapshot_dir)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeLifecycleCorruption("snapshot directory is missing or a symlink")
    inventory = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in inventory):
        raise RuntimeLifecycleCorruption("snapshot inventory contains unsafe entries")
    actual = {path.name for path in inventory}
    expected = _expected_snapshot_files(fixture)
    if actual != expected:
        raise RuntimeLifecycleCorruption(
            f"snapshot inventory mismatch: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    metadata = _strict_object(root / "metadata.json", "snapshot metadata")
    recorded_payload = metadata.get("metadata_payload_sha256")
    unsigned = {
        key: value for key, value in metadata.items() if key != "metadata_payload_sha256"
    }
    if recorded_payload != canonical_json_sha256(unsigned):
        raise RuntimeLifecycleCorruption("snapshot metadata payload SHA mismatch")
    expected_identity = {
        "schema_version": SNAPSHOT_METADATA_SCHEMA,
        "fixture_only": True,
        "formal_confirmatory_science_evaluated": False,
        "random_seed_used": False,
        "condition": fixture.condition,
        "scene_variant": fixture.scene_variant,
        "snapshot_id": fixture.snapshot_id,
        "checksums": dict(fixture.checksums),
        "lineage_sidecar_required": _lineage_expected(fixture),
    }
    if any(metadata.get(name) != value for name, value in expected_identity.items()):
        raise RuntimeLifecycleCorruption("snapshot metadata identity mismatch")
    file_hashes = metadata.get("file_sha256")
    expected_hash_names = expected - {"metadata.json"}
    if type(file_hashes) is not dict or set(file_hashes) != expected_hash_names:
        raise RuntimeLifecycleCorruption("snapshot file-SHA inventory mismatch")
    for name, expected_sha in file_hashes.items():
        if file_sha256(root / name) != expected_sha:
            raise RuntimeLifecycleCorruption(f"snapshot file SHA mismatch: {name}")
    try:
        source = np.load(root / "source_points.npy", allow_pickle=False)
        target = np.load(root / "target_points.npy", allow_pickle=False)
        reference = np.load(root / "reference_pose.npy", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeLifecycleCorruption("snapshot array load failed") from error
    arrays = {
        "source_checksum": source,
        "target_checksum": target,
        "reference_pose_checksum": reference,
    }
    for name, value in arrays.items():
        if _raw_array_sha256(value) != fixture.checksums[name]:
            raise RuntimeLifecycleCorruption(f"snapshot scientific checksum mismatch: {name}")
    if not (
        np.array_equal(source, fixture.source)
        and np.array_equal(target, fixture.target)
        and np.array_equal(reference, fixture.reference)
    ):
        raise RuntimeLifecycleCorruption("snapshot scientific array changed")
    if _lineage_expected(fixture):
        try:
            lineage = np.load(
                root / "source_parent_target_indices.npy", allow_pickle=False
            )
        except (OSError, ValueError) as error:
            raise RuntimeLifecycleCorruption("lineage sidecar load failed") from error
        if (
            lineage.dtype != np.dtype("<i8")
            or lineage.shape != (len(source),)
            or not np.array_equal(lineage, np.arange(len(source), dtype="<i8"))
        ):
            raise RuntimeLifecycleCorruption("lineage sidecar identity mismatch")
        reconstructed = (
            (reference[:3, :3] @ source.astype(np.float64).T).T
            + reference[:3, 3]
        ).astype("<f4")
        if not np.array_equal(reconstructed, target[lineage]):
            raise RuntimeLifecycleCorruption("lineage reconstruction mismatch")
    return FixtureSnapshot(
        snapshot_id=fixture.snapshot_id,
        scene_variant=fixture.scene_variant,
        condition=fixture.condition,
        source=source,
        target=target,
        reference=reference,
        expected_failure_classifications=fixture.expected_failure_classifications,
        checksums=dict(fixture.checksums),
    )


def prepare_runtime_snapshots(
    *,
    layout: RuntimePathLayout,
    after_commit: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    layout.snapshot_cache.mkdir(parents=True, exist_ok=True)
    fixtures = build_fixture_snapshots()
    generated: list[str] = []
    skipped: list[str] = []
    for fixture in fixtures:
        destination = layout.snapshot_cache / _snapshot_token(fixture.snapshot_id)
        if destination.exists() or destination.is_symlink():
            validate_runtime_snapshot(destination, fixture)
            skipped.append(fixture.snapshot_id)
            continue
        atomic_publish_directory(
            destination,
            lambda staging, item=fixture: _populate_snapshot(staging, item),
        )
        validate_runtime_snapshot(destination, fixture)
        generated.append(fixture.snapshot_id)
        if after_commit is not None:
            after_commit(fixture.snapshot_id, len(generated))
    validate_all_runtime_snapshots(layout)
    return {
        "generated_snapshot_ids": generated,
        "generated_snapshot_count": len(generated),
        "resume_skipped_valid_snapshot_ids": skipped,
        "resume_skipped_valid_snapshot_count": len(skipped),
    }


def validate_all_runtime_snapshots(
    layout: RuntimePathLayout,
) -> tuple[FixtureSnapshot, ...]:
    fixtures = build_fixture_snapshots()
    expected_names = {_snapshot_token(item.snapshot_id) for item in fixtures}
    if not layout.snapshot_cache.is_dir() or layout.snapshot_cache.is_symlink():
        raise RuntimeLifecycleCorruption("snapshot cache is missing or unsafe")
    actual = {path.name for path in layout.snapshot_cache.iterdir()}
    if actual != expected_names:
        raise RuntimeLifecycleCorruption("snapshot cache reverse inventory mismatch")
    return tuple(
        validate_runtime_snapshot(
            layout.snapshot_cache / _snapshot_token(item.snapshot_id), item
        )
        for item in fixtures
    )


def _raw_paths(layout: RuntimePathLayout) -> tuple[Path, Path]:
    return layout.raw_results / "results", layout.raw_results / "raw_result_manifest.json"


def _empty_raw_manifest(run_id: str, contract_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": RAW_MANIFEST_SCHEMA,
        "run_id": run_id,
        "run_contract_sha256": contract_sha256,
        "results": {},
    }


def _read_raw_manifest(
    path: Path, *, run_id: str, contract_sha256: str
) -> dict[str, Any]:
    value = _strict_object(path, "raw result manifest")
    if (
        set(value)
        != {"schema_version", "run_id", "run_contract_sha256", "results"}
        or value.get("schema_version") != RAW_MANIFEST_SCHEMA
        or value.get("run_id") != run_id
        or value.get("run_contract_sha256") != contract_sha256
        or type(value.get("results")) is not dict
    ):
        raise RuntimeLifecycleCorruption("raw result manifest contract mismatch")
    return value


def _trial_common(
    repository: Path,
    fixture: FixtureSnapshot,
    backend: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "condition": fixture.condition,
        "implementation_sha256": implementation_sha256,
        "planned_trial_id": f"{fixture.snapshot_id}/{backend}",
        "protocol_sha256": file_sha256(
            repository / "frozen_assets/fixtures/fixture_plan.json"
        ),
        "reference_pose_checksum": fixture.checksums["reference_pose_checksum"],
        "scene_variant": fixture.scene_variant,
        "schema_version": "phase_a_trial_result_v1",
        "snapshot_checksum": fixture.checksums["snapshot_checksum"],
        "snapshot_id": fixture.snapshot_id,
        "snapshot_lock_sha256": file_sha256(
            repository / "frozen_assets/fixtures/fixture_snapshot_lock.json"
        ),
        "source_checksum": fixture.checksums["source_checksum"],
        "target_checksum": fixture.checksums["target_checksum"],
    }


def _expected_trials(
    repository: Path,
    fixtures: Sequence[FixtureSnapshot],
    implementation_sha256: str,
) -> dict[str, tuple[FixtureSnapshot, str, dict[str, Any]]]:
    return {
        common["planned_trial_id"]: (fixture, backend, common)
        for fixture in fixtures
        for backend in BACKENDS
        for common in (
            _trial_common(repository, fixture, backend, implementation_sha256),
        )
    }


def _validate_result_entry(
    *,
    path: Path,
    entry: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        value = validate_existing_trial_result_for_resume(
            path, manifest_entry=entry, expected=expected
        )
    except (OSError, ValueError, CorruptExistingResult) as error:
        raise RuntimeLifecycleCorruption(
            f"trial result failed strict resume validation: {path.name}"
        ) from error
    if path.read_bytes() != trial_json_bytes(value):
        raise RuntimeLifecycleCorruption("trial result JSON is not canonical")
    return value


def audit_runtime_results(
    *,
    repository: str | Path,
    layout: RuntimePathLayout,
    run_id: str,
    contract_sha256: str,
    implementation_sha256: str,
    allow_orphans: bool = False,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    fixtures = validate_all_runtime_snapshots(layout)
    expected = _expected_trials(root, fixtures, implementation_sha256)
    results_dir, manifest_path = _raw_paths(layout)
    manifest = _read_raw_manifest(
        manifest_path, run_id=run_id, contract_sha256=contract_sha256
    )
    manifest_ids = set(manifest["results"])
    expected_ids = set(expected)
    rows: list[dict[str, Any]] = []
    referenced: set[str] = set()
    for trial_id in sorted(manifest_ids & expected_ids):
        entry = manifest["results"][trial_id]
        if type(entry) is not dict or set(entry) != {
            "path",
            "planned_trial_id",
            "sha256",
        }:
            raise RuntimeLifecycleCorruption("raw result entry schema mismatch")
        filename = result_filename(trial_id)
        if (
            entry.get("planned_trial_id") != trial_id
            or entry.get("path") != filename
        ):
            raise RuntimeLifecycleCorruption("raw result entry identity mismatch")
        path = results_dir / filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeLifecycleCorruption("raw result path is missing or unsafe")
        if file_sha256(path) != entry.get("sha256"):
            raise RuntimeLifecycleCorruption("raw result SHA mismatch")
        rows.append(
            _validate_result_entry(path=path, entry=entry, expected=expected[trial_id][2])
        )
        referenced.add(filename)
    actual_files: set[str] = set()
    if results_dir.exists():
        if results_dir.is_symlink() or not results_dir.is_dir():
            raise RuntimeLifecycleCorruption("raw results directory is unsafe")
        for path in results_dir.iterdir():
            if path.is_symlink() or not path.is_file():
                raise RuntimeLifecycleCorruption("raw results inventory is unsafe")
            actual_files.add(path.name)
    orphans = sorted(actual_files - referenced)
    extra_ids = sorted(manifest_ids - expected_ids)
    missing_ids = sorted(expected_ids - manifest_ids)
    if extra_ids or (orphans and not allow_orphans):
        raise RuntimeLifecycleCorruption("raw result reverse inventory mismatch")
    return {
        "manifest": manifest,
        "rows": rows,
        "missing_trial_ids": missing_ids,
        "extra_trial_ids": extra_ids,
        "orphan_result_files": orphans,
        "duplicate_trial_count": len(rows) - len({row["planned_trial_id"] for row in rows}),
        "checksum_mismatch_count": 0,
        "corrupt_trial_count": 0,
    }


def recover_canonical_result_orphans(
    *,
    repository: str | Path,
    layout: RuntimePathLayout,
    run_id: str,
    contract_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    fixtures = validate_all_runtime_snapshots(layout)
    expected = _expected_trials(root, fixtures, implementation_sha256)
    report = audit_runtime_results(
        repository=root,
        layout=layout,
        run_id=run_id,
        contract_sha256=contract_sha256,
        implementation_sha256=implementation_sha256,
        allow_orphans=True,
    )
    results_dir, manifest_path = _raw_paths(layout)
    manifest = report["manifest"]
    missing = set(report["missing_trial_ids"])
    filename_to_id = {result_filename(trial_id): trial_id for trial_id in missing}
    recovered: list[str] = []
    for filename in report["orphan_result_files"]:
        trial_id = filename_to_id.get(filename)
        if trial_id is None:
            raise RuntimeLifecycleCorruption(f"unrecognized result orphan: {filename}")
        path = results_dir / filename
        entry = {
            "path": filename,
            "planned_trial_id": trial_id,
            "sha256": file_sha256(path),
        }
        _validate_result_entry(path=path, entry=entry, expected=expected[trial_id][2])
        manifest["results"][trial_id] = entry
        recovered.append(trial_id)
    if recovered:
        atomic_replace_canonical_json(manifest_path, manifest)
    return {
        "recovered_orphan_result_count": len(recovered),
        "recovered_orphan_trial_ids": sorted(recovered),
    }


def execute_runtime_trials(
    *,
    repository: str | Path,
    layout: RuntimePathLayout,
    run_id: str,
    invocation_id: str,
    contract_sha256: str,
    implementation_sha256: str,
    resume: bool,
    after_commit: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    fixtures = validate_all_runtime_snapshots(layout)
    expected = _expected_trials(root, fixtures, implementation_sha256)
    results_dir, manifest_path = _raw_paths(layout)
    layout.raw_results.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        atomic_create_canonical_json(
            manifest_path, _empty_raw_manifest(run_id, contract_sha256)
        )
    elif not resume:
        raise FileExistsError("raw result manifest exists without resume")
    recovered = recover_canonical_result_orphans(
        repository=root,
        layout=layout,
        run_id=run_id,
        contract_sha256=contract_sha256,
        implementation_sha256=implementation_sha256,
    )
    inventory = audit_runtime_results(
        repository=root,
        layout=layout,
        run_id=run_id,
        contract_sha256=contract_sha256,
        implementation_sha256=implementation_sha256,
    )
    manifest = inventory["manifest"]
    parameter_lock = json.loads(
        (root / "frozen_assets/fixtures/fixture_backend_parameter_lock.json").read_text(
            encoding="utf-8"
        )
    )
    pcl_cli = root / "bin/pcl_point_to_plane_cli"
    layout.backend_temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(layout.backend_temporary)
    layout.attempt_events.mkdir(parents=True, exist_ok=True)
    events_path = layout.attempt_events / "events.ndjson"
    completed: list[str] = []
    executed: list[str] = []
    skipped: list[str] = []
    executions = 0
    open3d_seed_calls = 0
    for fixture in fixtures:
        for backend in BACKENDS:
            trial_id = f"{fixture.snapshot_id}/{backend}"
            common = expected[trial_id][2]
            existing = manifest["results"].get(trial_id)
            path = results_dir / result_filename(trial_id)
            if existing is not None:
                _validate_result_entry(path=path, entry=existing, expected=common)
                append_event_v2(
                    events_path,
                    run_id=run_id,
                    invocation_id=invocation_id,
                    event_type="SKIPPED_VALID_RESULT",
                    planned_trial_id=trial_id,
                    backend=backend,
                    result_sha256=existing["sha256"],
                )
                skipped.append(trial_id)
                completed.append(trial_id)
                continue
            append_event_v2(
                events_path,
                run_id=run_id,
                invocation_id=invocation_id,
                event_type="STARTED",
                planned_trial_id=trial_id,
                backend=backend,
            )
            if backend == OPEN3D_BACKEND:
                parameters = parameter_lock["open3d_parameter_contract"]["parameters"]
                result = _execute_or_replay_stage2_fixture(
                    executor=execute_open3d_fixture,
                    original=_ORIGINAL_EXECUTE_OPEN3D_FIXTURE,
                    fixture=fixture,
                    common=common,
                    parameters=parameters,
                )
                open3d_seed_calls += 1
            elif backend == PCL_BACKEND:
                parameters = parameter_lock["pcl_parameter_contract"]["parameters"]
                result = _execute_or_replay_stage2_fixture(
                    executor=execute_pcl_fixture,
                    original=_ORIGINAL_EXECUTE_PCL_FIXTURE,
                    fixture=fixture,
                    common=common,
                    parameters=parameters,
                    pcl_cli=pcl_cli,
                )
            else:  # pragma: no cover - frozen tuple protects this branch
                raise PermissionError("Native and unknown backends are forbidden")
            payload = validate_phase_a_trial_result_strict(result)
            atomic_create_bytes(path, trial_json_bytes(payload))
            digest = file_sha256(path)
            manifest["results"][trial_id] = {
                "path": path.name,
                "planned_trial_id": trial_id,
                "sha256": digest,
            }
            atomic_replace_canonical_json(manifest_path, manifest)
            append_event_v2(
                events_path,
                run_id=run_id,
                invocation_id=invocation_id,
                event_type="COMPLETED",
                planned_trial_id=trial_id,
                backend=backend,
                result_sha256=digest,
            )
            executions += 1
            executed.append(trial_id)
            completed.append(trial_id)
            if after_commit is not None:
                after_commit(trial_id, executions)
    final = audit_runtime_results(
        repository=root,
        layout=layout,
        run_id=run_id,
        contract_sha256=contract_sha256,
        implementation_sha256=implementation_sha256,
    )
    if final["missing_trial_ids"] or len(final["rows"]) != EXPECTED_TRIAL_COUNT:
        raise RuntimeError("fixture trial matrix is incomplete")
    events = read_event_journal_v2(events_path, expected_run_id=run_id)
    return {
        "backend_execution_count_this_invocation": executions,
        "backend_determinism_seed_call_count_this_invocation": open3d_seed_calls,
        "executed_trial_ids": executed,
        "completed_trial_ids": completed,
        "resume_skipped_valid_result_count": len(skipped),
        "resumed_trial_ids": skipped,
        "recovered_orphan_result_count": recovered[
            "recovered_orphan_result_count"
        ],
        "event_count": len(events),
        "rows": final["rows"],
        "raw_result_manifest_sha256": file_sha256(manifest_path),
    }


def load_completed_fixture_results(
    *,
    repository: str | Path,
    layout: RuntimePathLayout,
    run_id: str,
    contract_sha256: str,
    implementation_sha256: str,
) -> list[dict[str, Any]]:
    report = audit_runtime_results(
        repository=repository,
        layout=layout,
        run_id=run_id,
        contract_sha256=contract_sha256,
        implementation_sha256=implementation_sha256,
    )
    if report["missing_trial_ids"] or len(report["rows"]) != EXPECTED_TRIAL_COUNT:
        raise RuntimeError("fixture results are incomplete")
    return sorted(report["rows"], key=lambda row: row["planned_trial_id"])


def summarize_fixture_outcomes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {
        fixture.condition: set(fixture.expected_failure_classifications)
        for fixture in build_fixture_snapshots()
    }
    pairing: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        pairing.setdefault(str(row["snapshot_id"]), []).append(row)
    checksum_fields = (
        "snapshot_checksum",
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
    )
    pairing_mismatch = sum(
        len(group) != 2
        or {row["backend"] for row in group} != set(BACKENDS)
        or any(group[0][name] != group[1][name] for name in checksum_fields)
        for group in pairing.values()
    )
    outcome_mismatch = sum(
        row["failure_classification"] not in expected[row["condition"]]
        for row in rows
    )
    return {
        "fixture_snapshot_count": len(pairing),
        "fixture_trial_count": len(rows),
        "backend_trial_counts": dict(
            sorted(Counter(row["backend"] for row in rows).items())
        ),
        "pairing_mismatch_count": pairing_mismatch,
        "outcome_mismatch_count": outcome_mismatch,
        "FIXTURE_EXECUTION_CHAIN_PASS": bool(
            len(pairing) == EXPECTED_SNAPSHOT_COUNT
            and len(rows) == EXPECTED_TRIAL_COUNT
            and pairing_mismatch == 0
            and outcome_mismatch == 0
        ),
    }


def publish_fixture_runtime_artifact(
    *,
    layout: RuntimePathLayout,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish in the external publisher area, then atomically stage a copy.

    The frozen v2 publisher remains byte-for-byte unchanged.  This lifecycle
    wrapper supplies the two external path boundaries required by the repaired
    contract without changing publication semantics or repository state.
    """

    from .synthetic_confirmatory_v2_artifact_verifier import (
        verify_synthetic_confirmatory_v2_fixture_artifact,
    )
    from .synthetic_confirmatory_v2_publisher import (
        publish_synthetic_confirmatory_v2_fixture,
    )

    publisher_output = layout.publisher_staging / "fixture_publication"
    final_artifact = layout.artifact_staging / "fixture_publication"
    publication = publish_synthetic_confirmatory_v2_fixture(
        primary=primary,
        independent=independent,
        run_manifest=run_manifest,
        artifact_dir=publisher_output,
    )
    publisher_verification = verify_synthetic_confirmatory_v2_fixture_artifact(
        publisher_output, write_report=False
    )
    if publisher_verification.get("FIXTURE_ARTIFACT_VERIFICATION_PASS") is not True:
        raise RuntimeLifecycleCorruption("publisher staging verification failed")

    def populate(staging: Path) -> None:
        shutil.copytree(publisher_output, staging, dirs_exist_ok=True)

    atomic_publish_directory(final_artifact, populate)
    final_verification = verify_synthetic_confirmatory_v2_fixture_artifact(
        final_artifact, write_report=False
    )
    if final_verification != publisher_verification:
        raise RuntimeLifecycleCorruption(
            "artifact stage differs from verified publisher staging"
        )
    return {
        **publication,
        "publisher_staging_path": str(publisher_output),
        "artifact_staging_path": str(final_artifact),
        "publisher_staging_verification": publisher_verification,
        "artifact_staging_verification": final_verification,
    }


def run_fixture_lifecycle(
    *,
    repository: str | Path,
    layout: RuntimePathLayout,
    run_id: str,
    invocation_id: str,
    workers: int,
    resume: bool,
    expected_commit: str,
    expected_branch: str,
    expected_tag: str,
    runtime_path_policy_sha256: str,
    qualification_delay_seconds: float = 0.0,
    git_gate: Callable[[str], Mapping[str, Any]] | None = None,
    prebootstrapped_root: bool = False,
    single_writer_lease_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute snapshots and trials under one immutable external run contract."""

    if type(prebootstrapped_root) is not bool:
        raise TypeError("prebootstrapped_root must be bool")
    if single_writer_lease_path is not None and not prebootstrapped_root:
        raise ValueError(
            "an external single-writer lease is reserved for prebootstrapped runs"
        )
    if single_writer_lease_path is not None:
        expected_external_lease = (
            layout.run_root.parent / f".{layout.run_root.name}.bootstrap.lease"
        )
        supplied_external_lease = Path(
            os.path.abspath(os.fspath(single_writer_lease_path))
        )
        if supplied_external_lease != expected_external_lease:
            raise ValueError("prebootstrapped runs require the exact external lease")
    root = Path(repository).resolve()
    contract = build_fixture_run_contract(
        repository=root,
        layout=layout,
        run_id=run_id,
        workers=workers,
        expected_commit=expected_commit,
        expected_branch=expected_branch,
        expected_tag=expected_tag,
        runtime_path_policy_sha256=runtime_path_policy_sha256,
        qualification_delay_seconds=qualification_delay_seconds,
    )
    contract_sha = canonical_json_sha256(contract)
    if git_gate is not None:
        git_gate("RESUME_GIT_GATE" if resume else "PRE_RUN_GIT_GATE")
    layout.run_root.mkdir(
        parents=True, exist_ok=(resume or prebootstrapped_root)
    )
    lease_path = (
        Path(single_writer_lease_path)
        if single_writer_lease_path is not None
        else layout.run_root / "single_writer.lease"
    )
    with SingleWriterLease(lease_path):
        if git_gate is not None:
            git_gate("POST_LEASE_GIT_GATE")
        lock = write_once_immutable_run_lock(
            layout.snapshot_lock, contract, resume=resume
        )

        def snapshot_commit(_snapshot_id: str, _count: int) -> None:
            if git_gate is not None:
                git_gate("MID_SNAPSHOT_GIT_GATE")
            if qualification_delay_seconds:
                time.sleep(qualification_delay_seconds)

        snapshots = prepare_runtime_snapshots(
            layout=layout, after_commit=snapshot_commit
        )
        if git_gate is not None:
            git_gate("POST_SNAPSHOT_GIT_GATE")

        def trial_commit(_trial_id: str, _count: int) -> None:
            if git_gate is not None:
                git_gate("MID_TRIAL_GIT_GATE")
            if qualification_delay_seconds:
                time.sleep(qualification_delay_seconds)

        trials = execute_runtime_trials(
            repository=root,
            layout=layout,
            run_id=run_id,
            invocation_id=invocation_id,
            contract_sha256=contract_sha,
            implementation_sha256=contract["implementation_sha256"],
            resume=resume,
            after_commit=trial_commit,
        )
        outcomes = summarize_fixture_outcomes(trials["rows"])
        report = {
            "schema_version": RUN_REPORT_SCHEMA,
            "run_id": run_id,
            "invocation_id": invocation_id,
            "resume": resume,
            "workers": workers,
            "runtime_paths": layout.as_dict(),
            "run_contract_sha256": contract_sha,
            "immutable_run_lock_sha256": file_sha256(layout.snapshot_lock),
            "fixture_generation_rng_count": 0,
            "formal_confirmatory_seed_access_count": 0,
            "new_confirmatory_namespace_generation_count": 0,
            "native_execution_count": 0,
            **snapshots,
            **{key: value for key, value in trials.items() if key != "rows"},
            **outcomes,
        }
        layout.temporary_inventory.mkdir(parents=True, exist_ok=True)
        report_path = layout.temporary_inventory / f"{invocation_id}.json"
        atomic_create_canonical_json(report_path, report)
        if git_gate is not None:
            git_gate("POST_TRIAL_GIT_GATE")
        return report


__all__ = [
    "BACKENDS",
    "EXPECTED_SNAPSHOT_COUNT",
    "EXPECTED_TRIAL_COUNT",
    "RAW_MANIFEST_SCHEMA",
    "RUN_CONTRACT_SCHEMA",
    "RUN_REPORT_SCHEMA",
    "RuntimeLifecycleCorruption",
    "audit_runtime_results",
    "build_fixture_run_contract",
    "execute_runtime_trials",
    "load_completed_fixture_results",
    "prepare_runtime_snapshots",
    "publish_fixture_runtime_artifact",
    "recover_canonical_result_orphans",
    "run_fixture_lifecycle",
    "summarize_fixture_outcomes",
    "validate_all_runtime_snapshots",
    "validate_runtime_snapshot",
]
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .execution_context import (
    BackendPolicy,
    ExecutionContext,
    ExecutionContract,
    ExecutionMode,
    IdPolicy,
    SchemaBinding,
    SeedPolicy,
    SnapshotReaderPolicy,
    canonical_plan_rows_sha256,
)
from .formal_lifecycle_components import (
    ComponentBinding,
    FormalLifecycleComponents,
)
from .formal_lifecycle_contract import (
    CommandProfile,
    FormalLifecycleSpec,
    GitIdentityPolicy,
    PublisherInventory,
    deep_thaw,
)
from .formal_lifecycle_paths import FormalLifecyclePaths


OPEN3D_BACKEND = "open3d_point_to_plane"
PCL_BACKEND = "pcl_point_to_plane"
BACKENDS = (OPEN3D_BACKEND, PCL_BACKEND)

CONTEXT_A = "context_a"
CONTEXT_B = "context_b"
CONTEXT_NAMES = (CONTEXT_A, CONTEXT_B)

CONTEXT_A_RUN_ID = "version-agnostic-lifecycle-fixture-a"
CONTEXT_B_RUN_ID = "version-agnostic-lifecycle-fixture-b"

QUALIFICATION_RUNTIME_BASE = Path(
    "/home/lj/zero_perturbation_runtime/qualification/"
    "version_agnostic_formal_lifecycle_v1"
)
CONTEXT_A_RUNTIME_ROOT = QUALIFICATION_RUNTIME_BASE / CONTEXT_A
CONTEXT_B_RUNTIME_ROOT = QUALIFICATION_RUNTIME_BASE / CONTEXT_B

SCENES = (
    "IDENTITY",
    "NONIDENTITY_REFERENCE",
    "NO_CORRESPONDENCE",
)
CONDITIONS = (
    "FIXTURE_IDENTITY",
    "FIXTURE_NONIDENTITY_REFERENCE",
    "FIXTURE_NO_CORRESPONDENCE",
)
SCENE_CONDITIONS = tuple(zip(SCENES, CONDITIONS))

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
REPLICATE_SEMANTICS = "ONE_SEED_FREE_QUALIFICATION_INPUT"

EXPECTED_OUTCOMES = MappingProxyType(
    {
        "FIXTURE_IDENTITY": "NONE",
        "FIXTURE_NONIDENTITY_REFERENCE": "NONE",
        "FIXTURE_NO_CORRESPONDENCE": "NO_CORRESPONDENCES",
    }
)

FIXTURE_PARAMETER_LOCK_RELATIVE = Path(
    "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
)
PCL_CLI_RELATIVE = Path("bin/pcl_point_to_plane_cli")
PCL_CLI_SHA256 = (
    "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff"
)


@dataclass(frozen=True)
class _ContextIdentity:
    name: str
    run_id: str
    runtime_root: Path
    id_prefix: str
    protocol_schema: str
    protocol_version: str
    manifest_schema: str
    plan_id: str
    contract_id: str
    snapshot_plan_schema: str
    trial_plan_schema: str
    metadata_schema: str
    snapshot_schema: str
    lineage_schema: str
    lock_schema: str
    builder_contract: str
    bundle_id: str
    inventory_id: str
    command_profile_id: str


_CONTEXTS = MappingProxyType(
    {
        CONTEXT_A: _ContextIdentity(
            name=CONTEXT_A,
            run_id=CONTEXT_A_RUN_ID,
            runtime_root=CONTEXT_A_RUNTIME_ROOT,
            id_prefix="version-agnostic-lifecycle-a",
            protocol_schema="version_agnostic_lifecycle_fixture_protocol_a_v1",
            protocol_version="qualification-a-protocol-v1",
            manifest_schema="version_agnostic_lifecycle_fixture_manifest_a_v1",
            plan_id="version-agnostic-lifecycle-qualification-plan-a-v1",
            contract_id="version-agnostic-lifecycle-qualification-contract-a-v1",
            snapshot_plan_schema="version_agnostic_lifecycle_snapshot_plan_a_v1",
            trial_plan_schema="version_agnostic_lifecycle_trial_plan_a_v1",
            metadata_schema="version_agnostic_lifecycle_metadata_a_v1",
            snapshot_schema="version_agnostic_lifecycle_snapshot_a_v1",
            lineage_schema="version_agnostic_lifecycle_lineage_a_v1",
            lock_schema="version_agnostic_lifecycle_snapshot_lock_a_v1",
            builder_contract="version_agnostic_lifecycle_fixture_builder_a_v1",
            bundle_id="version-agnostic-lifecycle-component-bundle-a-v1",
            inventory_id="version-agnostic-lifecycle-publisher-inventory-a-v1",
            command_profile_id="version-agnostic-lifecycle-command-profile-a-v1",
        ),
        CONTEXT_B: _ContextIdentity(
            name=CONTEXT_B,
            run_id=CONTEXT_B_RUN_ID,
            runtime_root=CONTEXT_B_RUNTIME_ROOT,
            id_prefix="version-agnostic-lifecycle-b",
            protocol_schema="version_agnostic_lifecycle_fixture_protocol_b_v2",
            protocol_version="qualification-b-protocol-v2",
            manifest_schema="version_agnostic_lifecycle_fixture_manifest_b_v2",
            plan_id="version-agnostic-lifecycle-qualification-plan-b-v2",
            contract_id="version-agnostic-lifecycle-qualification-contract-b-v2",
            snapshot_plan_schema="version_agnostic_lifecycle_snapshot_plan_b_v2",
            trial_plan_schema="version_agnostic_lifecycle_trial_plan_b_v2",
            metadata_schema="version_agnostic_lifecycle_metadata_b_v2",
            snapshot_schema="version_agnostic_lifecycle_snapshot_b_v2",
            lineage_schema="version_agnostic_lifecycle_lineage_b_v2",
            lock_schema="version_agnostic_lifecycle_snapshot_lock_b_v2",
            builder_contract="version_agnostic_lifecycle_fixture_builder_b_v2",
            bundle_id="version-agnostic-lifecycle-component-bundle-b-v2",
            inventory_id="version-agnostic-lifecycle-publisher-inventory-b-v2",
            command_profile_id="version-agnostic-lifecycle-command-profile-b-v2",
        ),
    }
)


@dataclass(frozen=True)
class QualificationBackendInput:
    """Authenticated per-trial input passed to one normalized backend callable."""

    backend: str
    fixture: Any
    parameters: Mapping[str, Any]
    pcl_cli: Path | None


def _identity(context_name: str) -> _ContextIdentity:
    try:
        return _CONTEXTS[context_name]
    except KeyError as error:
        raise ValueError(
            f"unknown qualification context: {context_name!r}"
        ) from error


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qualification_strict_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
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


def qualification_snapshot_plan(
    context_name: str,
) -> tuple[dict[str, Any], ...]:
    """Return the exact ordered, seed-free 3-snapshot plan for A or B."""

    identity = _identity(context_name)
    return tuple(
        {
            "planned_snapshot_id": f"{identity.id_prefix}-{scene.lower().replace('_', '-')}",
            "scene_variant": scene,
            "condition": condition,
            "geometry_seed": None,
            "measurement_seed": None,
            "repeat_index": 0,
            "planned_backend_count": 2,
            "replicate_semantics": REPLICATE_SEMANTICS,
        }
        for scene, condition in SCENE_CONDITIONS
    )


def qualification_trial_plan(
    context_name: str,
) -> tuple[dict[str, Any], ...]:
    """Return the exact ordered, seed-free 6-trial Open3D/PCL plan."""

    return tuple(
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
        for snapshot in qualification_snapshot_plan(context_name)
        for backend in BACKENDS
    )


def _snapshot_identity_by_id() -> dict[str, tuple[str, str]]:
    return {
        row["planned_snapshot_id"]: (
            str(row["scene_variant"]),
            str(row["condition"]),
        )
        for name in CONTEXT_NAMES
        for row in qualification_snapshot_plan(name)
    }


def validate_qualification_snapshot_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate either A or B without selecting a context from ambient state."""

    if not isinstance(row, Mapping) or set(row) != set(SNAPSHOT_FIELDS):
        raise ValueError("qualification snapshot row field set changed")
    value = {name: row[name] for name in SNAPSHOT_FIELDS}
    expected = _snapshot_identity_by_id()
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


def validate_qualification_trial_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate either A or B trial row under the shared backend vocabulary."""

    if not isinstance(row, Mapping) or set(row) != set(TRIAL_FIELDS):
        raise ValueError("qualification trial row field set changed")
    value = {name: row[name] for name in TRIAL_FIELDS}
    expected = _snapshot_identity_by_id()
    snapshot_id = value["planned_snapshot_id"]
    if (
        type(snapshot_id) is not str
        or snapshot_id not in expected
        or (value["scene_variant"], value["condition"]) != expected[snapshot_id]
        or value["geometry_seed"] is not None
        or value["measurement_seed"] is not None
        or type(value["repeat_index"]) is not int
        or value["repeat_index"] != 0
        or value["backend"] not in BACKENDS
        or value["planned_trial_id"]
        != f"{snapshot_id}::{value['backend']}"
    ):
        raise ValueError("qualification trial identity is invalid")
    return value


def build_qualification_paths(
    context_name: str,
    *,
    runtime_root: Path | None = None,
) -> FormalLifecyclePaths:
    """Build the exact external layout without creating any path."""

    identity = _identity(context_name)
    root = identity.runtime_root if runtime_root is None else runtime_root
    root = Path(os.path.abspath(os.fspath(root)))
    return FormalLifecyclePaths(
        runtime_root=root,
        snapshot_cache=root / "snapshot_cache",
        snapshot_lock=root / "snapshot_lock.json",
        raw_results=root / "raw_results",
        raw_manifest=root / "raw_result_manifest.json",
        event_log=root / "event_logs" / "attempt_events.ndjson",
        backend_temporary=root / "backend_tmp",
        analysis=root / "analysis",
        verification=root / "verification",
        publisher_staging=root / "publisher_staging",
        artifact_staging=root / "artifact_staging",
        temporary_inventory=root / "working_inventory",
        run_manifest=root / "run_manifest.json",
        formal_command_log=root / "formal_command.log",
        formal_command_sha256=root / "formal_command.log.sha256",
        immutable_run_lock=root / "immutable_run_lock.json",
        single_writer_lease=root.parent / f".{root.name}.bootstrap.lease",
    )


def build_qualification_execution_context(
    context_name: str,
    *,
    runtime_root: Path | None = None,
) -> ExecutionContext:
    """Construct one explicit A/B context with no seed permission."""

    from .synthetic_confirmatory_v3_snapshot_builder import (
        METADATA_FIELDS,
        _LOCK_ENTRY_FIELDS,
    )

    identity = _identity(context_name)
    paths = build_qualification_paths(
        context_name, runtime_root=runtime_root
    )
    mode = ExecutionMode.QUALIFICATION
    snapshot_schema = SchemaBinding(
        mode,
        identity.snapshot_plan_schema,
        SNAPSHOT_FIELDS,
    )
    trial_schema = SchemaBinding(
        mode,
        identity.trial_plan_schema,
        TRIAL_FIELDS,
    )
    seed_policy = SeedPolicy(
        mode,
        f"{identity.id_prefix}-no-confirmatory-seed",
        False,
    )
    id_policy = IdPolicy(mode, f"{identity.id_prefix}-explicit-id-policy")
    backend_policy = BackendPolicy(
        mode,
        f"{identity.id_prefix}-open3d-pcl-only",
        BACKENDS,
        2,
    )
    reader_policy = SnapshotReaderPolicy(
        mode=mode,
        policy_id=f"{identity.id_prefix}-reader-policy",
        metadata_fields=tuple(sorted(METADATA_FIELDS)),
        lock_entry_fields=tuple(sorted(_LOCK_ENTRY_FIELDS)),
        metadata_schema=identity.metadata_schema,
        snapshot_schema_version=identity.snapshot_schema,
        lineage_schema_version=identity.lineage_schema,
        seed_namespace=None,
        snapshot_builder_contract_version=identity.builder_contract,
        lineage_required_conditions=("FIXTURE_IDENTITY",),
        expected_rng_counts={condition: 0 for condition in CONDITIONS},
    )
    snapshots = list(qualification_snapshot_plan(context_name))
    trials = list(qualification_trial_plan(context_name))
    contract = ExecutionContract(
        mode=mode,
        contract_id=identity.contract_id,
        plan_id=identity.plan_id,
        expected_snapshot_plan_sha256=canonical_plan_rows_sha256(snapshots),
        expected_trial_plan_sha256=canonical_plan_rows_sha256(trials),
        expected_snapshot_count=3,
        expected_trial_count=6,
        expected_cache_root=paths.snapshot_cache,
        expected_runtime_root=paths.runtime_root,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=SCENES,
        allowed_conditions=CONDITIONS,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
        snapshot_validator=validate_qualification_snapshot_row,
        trial_validator=validate_qualification_trial_row,
        result_route="QUALIFICATION_PHASE_A_ROUTING_V1",
    )
    return ExecutionContext(
        mode=mode,
        contract=contract,
        plan_id=identity.plan_id,
        cache_root=paths.snapshot_cache,
        runtime_root=paths.runtime_root,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=SCENES,
        allowed_conditions=CONDITIONS,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
    )


def build_qualification_publisher_inventory(
    context_name: str,
) -> PublisherInventory:
    from .synthetic_confirmatory_v2_artifact_verifier import (
        FIXTURE_FIGURES,
        FIXTURE_ROOT_FILES,
        FIXTURE_TABLES,
    )

    identity = _identity(context_name)
    return PublisherInventory(
        inventory_id=identity.inventory_id,
        tables=tuple(FIXTURE_TABLES),
        figures=tuple(FIXTURE_FIGURES),
        root_files=tuple(FIXTURE_ROOT_FILES),
    )


def build_qualification_command_profile(
    context_name: str,
    *,
    entry_script: Path = Path(
        "scripts/run_version_agnostic_formal_lifecycle_fixture.py"
    ),
) -> CommandProfile:
    identity = _identity(context_name)
    return CommandProfile(
        profile_id=identity.command_profile_id,
        entry_script=entry_script,
        environment={
            "MAMBA_ROOT_PREFIX": (
                "/home/lj/.local/share/degen-lio-micromamba"
            ),
            "PYTHONNOUSERSITE": "1",
        },
        argv_prefix=(
            "/home/lj/.local/bin/micromamba",
            "run",
            "-n",
            "degen-lio-zprm-py311",
            "python",
        ),
        unset_environment=("PYTHONPATH",),
    )


def _qualification_metadata(
    *,
    spec: FormalLifecycleSpec,
    row: Mapping[str, Any],
    source: Any,
    target: Any,
    reference: Any,
    parent_indices: Any | None,
) -> dict[str, Any]:
    import numpy as np

    from .synthetic_confirmatory_v3_snapshot_builder import (
        METADATA_FIELDS,
        PARENT_INDEX_FILENAME,
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
    snapshot_checksum = _canonical_sha256(
        {
            "snapshot_id": row["planned_snapshot_id"],
            **raw,
            "source_parent_target_indices_sha256": parent_sha,
        }
    )
    policy = spec.execution_context.snapshot_reader_policy
    metadata = {
        "array_file_sha256": {},
        "condition": row["condition"],
        "confirmatory_rng_instantiation_count": 0,
        "development_protocol_sha256": spec.manifest[
            "qualification_protocol_sha256"
        ],
        "dropout_parameters": {
            "map_dropout_fraction": 0.0,
            "scan_dropout_fraction": 0.0,
        },
        "formal_manifest_payload_sha256": spec.manifest_sha256,
        "generator_record_checksums": {},
        "generator_sha256": spec.manifest[
            "qualification_implementation_sha256"
        ],
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
        raise AssertionError(
            "qualification metadata does not match the authenticated reader"
        )
    return metadata


def qualification_snapshot_materializer(
    *,
    spec: FormalLifecycleSpec,
    snapshot_row: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Materialize exactly one deterministic fixture snapshot atomically."""

    import numpy as np

    from .phase_a_execution_chain_fixture import build_fixture_snapshots
    from .synthetic_confirmatory_v3_snapshot_builder import (
        _write_v3_snapshot_atomic,
    )

    row = spec.execution_context.validate_snapshot_row(snapshot_row)
    expected_destination = (
        spec.runtime_paths.snapshot_cache / row["planned_snapshot_id"]
    )
    if destination != expected_destination:
        raise ValueError("snapshot materializer destination differs from spec")
    fixtures = {
        fixture.condition: fixture
        for fixture in build_fixture_snapshots()
    }
    fixture = fixtures[row["condition"]]
    source = np.ascontiguousarray(fixture.source, dtype="<f4")
    target = np.ascontiguousarray(fixture.target, dtype="<f4")
    reference = np.ascontiguousarray(fixture.reference, dtype="<f8")
    parent_indices = None
    if row["condition"] == "FIXTURE_IDENTITY":
        parent_indices = np.ascontiguousarray(
            np.arange(len(source), dtype="<i8")
        )
    metadata = _qualification_metadata(
        spec=spec,
        row=row,
        source=source,
        target=target,
        reference=reference,
        parent_indices=parent_indices,
    )
    written = _write_v3_snapshot_atomic(
        spec.runtime_paths.snapshot_cache,
        {
            "metadata": metadata,
            "parent_indices": parent_indices,
            "reference": reference,
            "source": source,
            "target": target,
        },
    )
    if written != destination:
        raise RuntimeError("snapshot writer returned another destination")
    return {
        "snapshot_id": row["planned_snapshot_id"],
        "destination": str(written),
        "confirmatory_seed_access_count": 0,
        "confirmatory_rng_instantiation_count": 0,
    }


def qualification_snapshot_reader(
    *,
    spec: FormalLifecycleSpec,
    snapshot_row: Mapping[str, Any],
    expected_lock_entry: Mapping[str, Any] | None,
    arrays: bool,
) -> dict[str, Any]:
    """Authenticate exactly one snapshot through the established real reader."""

    from .synthetic_confirmatory_v3_snapshot_builder import read_v3_snapshot

    return read_v3_snapshot(
        spec.runtime_paths.snapshot_cache,
        snapshot_row,
        execution_context=spec.execution_context,
        expected_lock_entry=expected_lock_entry,
        arrays=arrays,
    )


def qualification_snapshot_validator(
    *,
    spec: FormalLifecycleSpec,
    snapshot_row: Mapping[str, Any],
    authenticated_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one authenticated payload and project its immutable lock entry."""

    row = spec.execution_context.validate_snapshot_row(snapshot_row)
    metadata = authenticated_snapshot.get("metadata")
    if (
        type(metadata) is not dict
        or metadata.get("snapshot_id") != row["planned_snapshot_id"]
        or metadata.get("planned_snapshot_id") != row["planned_snapshot_id"]
        or metadata.get("scene_variant") != row["scene_variant"]
        or metadata.get("condition") != row["condition"]
        or metadata.get("geometry_seed") is not None
        or metadata.get("measurement_seed") is not None
        or metadata.get("confirmatory_rng_instantiation_count") != 0
    ):
        raise ValueError("authenticated qualification snapshot identity mismatch")
    return _snapshot_lock_entry(authenticated_snapshot)


def _snapshot_lock_entry(
    authenticated: Mapping[str, Any],
) -> dict[str, Any]:
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


def qualification_snapshot_lock_builder(
    *,
    spec: FormalLifecycleSpec,
    authenticated_snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the context-specific immutable lock from authenticated receipts."""

    lock_fields = set(
        spec.execution_context.snapshot_reader_policy.lock_entry_fields
    )
    projected: dict[str, dict[str, Any]] = {}
    for snapshot in authenticated_snapshots:
        entry = (
            dict(snapshot)
            if set(snapshot) == lock_fields
            else _snapshot_lock_entry(snapshot)
        )
        snapshot_id = entry.get("snapshot_id")
        if (
            type(snapshot_id) is not str
            or snapshot_id in projected
        ):
            raise ValueError(
                "snapshot lock input has an invalid or duplicate identity"
            )
        projected[snapshot_id] = entry
    entries = [
        projected[deep_thaw(row)["planned_snapshot_id"]]
        for row in spec.snapshot_plan
    ]
    core = {
        "condition_snapshot_counts": dict(
            sorted(Counter(entry["condition"] for entry in entries).items())
        ),
        "confirmatory_rng_instantiation_count": 0,
        "formal_manifest_payload_sha256": spec.manifest_sha256,
        "lineage_schema_version": (
            spec.execution_context.snapshot_reader_policy.lineage_schema_version
        ),
        "planned_snapshot_count": len(spec.snapshot_plan),
        "planned_snapshot_identity_sha256": canonical_plan_rows_sha256(
            [deep_thaw(row) for row in spec.snapshot_plan]
        ),
        "schema_version": spec.manifest["qualification_snapshot_lock_schema"],
        "seed_namespace": None,
        "snapshot_builder_contract_version": (
            spec.execution_context.snapshot_reader_policy
            .snapshot_builder_contract_version
        ),
        "snapshot_schema_version": (
            spec.execution_context.snapshot_reader_policy.snapshot_schema_version
        ),
        "snapshots": entries,
    }
    return {
        **core,
        "snapshot_lock_payload_sha256": _canonical_sha256(core),
    }


def qualification_snapshot_lock_validator(
    *,
    spec: FormalLifecycleSpec,
    lock_value: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on a lock from another context, plan, or manifest."""

    value = dict(lock_value)
    unsigned = {
        name: item
        for name, item in value.items()
        if name != "snapshot_lock_payload_sha256"
    }
    entries = value.get("snapshots")
    plans = [deep_thaw(row) for row in spec.snapshot_plan]
    expected_counts = dict(
        sorted(Counter(row["condition"] for row in plans).items())
    )
    if (
        value.get("snapshot_lock_payload_sha256")
        != _canonical_sha256(unsigned)
        or value.get("schema_version")
        != spec.manifest["qualification_snapshot_lock_schema"]
        or value.get("formal_manifest_payload_sha256")
        != spec.manifest_sha256
        or value.get("planned_snapshot_count") != len(plans)
        or value.get("planned_snapshot_identity_sha256")
        != canonical_plan_rows_sha256(plans)
        or value.get("condition_snapshot_counts") != expected_counts
        or value.get("confirmatory_rng_instantiation_count") != 0
        or value.get("seed_namespace") is not None
        or value.get("snapshot_schema_version")
        != spec.execution_context.snapshot_reader_policy.snapshot_schema_version
        or value.get("lineage_schema_version")
        != spec.execution_context.snapshot_reader_policy.lineage_schema_version
        or value.get("snapshot_builder_contract_version")
        != spec.execution_context.snapshot_reader_policy
        .snapshot_builder_contract_version
        or type(entries) is not list
        or [entry.get("snapshot_id") for entry in entries]
        != [row["planned_snapshot_id"] for row in plans]
    ):
        raise ValueError("qualification snapshot lock identity mismatch")
    entries_by_id = {
        entry["snapshot_id"]: entry
        for entry in entries
    }
    if len(entries_by_id) != len(plans):
        raise ValueError("qualification snapshot lock contains duplicate IDs")
    for plan in plans:
        qualification_snapshot_reader(
            spec=spec,
            snapshot_row=plan,
            expected_lock_entry=entries_by_id[plan["planned_snapshot_id"]],
            arrays=False,
        )
    return {
        "lock": value,
        "entries_by_id": entries_by_id,
        "snapshot_lock_pass": True,
    }


def _load_backend_parameter_contract(
    spec: FormalLifecycleSpec,
) -> dict[str, Any]:
    relative = Path(str(spec.manifest["backend_parameter_contract_path"]))
    path = spec.repository_root / relative
    if (
        path.resolve(strict=False) != path
        or spec.repository_root not in path.parents
        or _file_sha256(path)
        != spec.manifest["backend_parameter_contract_sha256"]
    ):
        raise ValueError("fixture backend parameter contract binding mismatch")
    value = _qualification_strict_object(path)
    if value.get("formal_seed_values_included") is not False:
        raise PermissionError(
            "fixture backend parameter contract contains a formal seed"
        )
    for name in ("open3d_parameter_contract", "pcl_parameter_contract"):
        section = value.get(name)
        if (
            type(section) is not dict
            or _canonical_sha256(section.get("parameters"))
            != section.get("canonical_sha256")
        ):
            raise ValueError(f"fixture parameter identity mismatch: {name}")
    return value


def qualification_backend_input_builder(
    *,
    spec: FormalLifecycleSpec,
    trial_row: Mapping[str, Any],
    snapshot_row: Mapping[str, Any],
    authenticated_snapshot: Mapping[str, Any],
) -> QualificationBackendInput:
    """Convert one authenticated snapshot into the established fixture type."""

    import numpy as np

    from .phase_a_execution_chain_fixture import FixtureSnapshot

    trial = spec.execution_context.validate_trial_row(trial_row)
    snapshot = spec.execution_context.validate_snapshot_row(snapshot_row)
    if trial["planned_snapshot_id"] != snapshot["planned_snapshot_id"]:
        raise ValueError("backend input received an unbound trial/snapshot pair")
    metadata = authenticated_snapshot["metadata"]
    fixture = FixtureSnapshot(
        snapshot_id=snapshot["planned_snapshot_id"],
        scene_variant=snapshot["scene_variant"],
        condition=snapshot["condition"],
        source=np.asarray(authenticated_snapshot["source"]),
        target=np.asarray(authenticated_snapshot["target"]),
        reference=np.asarray(authenticated_snapshot["reference"]),
        expected_failure_classifications=(
            EXPECTED_OUTCOMES[snapshot["condition"]],
        ),
        checksums={
            "source_checksum": metadata["source_checksum"],
            "target_checksum": metadata["target_checksum"],
            "reference_pose_checksum": metadata["reference_pose_checksum"],
            "snapshot_checksum": metadata["snapshot_checksum"],
        },
    )
    parameter_contract = _load_backend_parameter_contract(spec)
    section = {
        OPEN3D_BACKEND: "open3d_parameter_contract",
        PCL_BACKEND: "pcl_parameter_contract",
    }[trial["backend"]]
    pcl_cli = None
    if trial["backend"] == PCL_BACKEND:
        pcl_cli = spec.repository_root / Path(
            str(spec.manifest["pcl_cli_path"])
        )
        if (
            pcl_cli.resolve(strict=False) != pcl_cli
            or spec.repository_root not in pcl_cli.parents
            or _file_sha256(pcl_cli) != spec.manifest["pcl_cli_sha256"]
            or spec.manifest["pcl_cli_sha256"] != PCL_CLI_SHA256
        ):
            raise ValueError("qualification PCL CLI binding mismatch")
    return QualificationBackendInput(
        backend=trial["backend"],
        fixture=fixture,
        parameters=parameter_contract[section]["parameters"],
        pcl_cli=pcl_cli,
    )


def qualification_common_record_builder(
    *,
    spec: FormalLifecycleSpec,
    trial_row: Mapping[str, Any],
    snapshot_row: Mapping[str, Any],
    backend_input: QualificationBackendInput,
    snapshot_lock_sha256: str,
) -> dict[str, Any]:
    trial = spec.execution_context.validate_trial_row(trial_row)
    snapshot = spec.execution_context.validate_snapshot_row(snapshot_row)
    fixture = backend_input.fixture
    if (
        backend_input.backend != trial["backend"]
        or fixture.snapshot_id != snapshot["planned_snapshot_id"]
    ):
        raise ValueError("common record input identity mismatch")
    return {
        "backend": trial["backend"],
        "condition": fixture.condition,
        "implementation_sha256": spec.manifest[
            "qualification_implementation_sha256"
        ],
        "planned_trial_id": trial["planned_trial_id"],
        "protocol_sha256": spec.manifest["qualification_protocol_sha256"],
        "reference_pose_checksum": fixture.checksums[
            "reference_pose_checksum"
        ],
        "scene_variant": fixture.scene_variant,
        "schema_version": "phase_a_trial_result_v1",
        "snapshot_checksum": fixture.checksums["snapshot_checksum"],
        "snapshot_id": fixture.snapshot_id,
        "snapshot_lock_sha256": snapshot_lock_sha256,
        "source_checksum": fixture.checksums["source_checksum"],
        "target_checksum": fixture.checksums["target_checksum"],
    }


def execute_qualification_open3d(
    *,
    backend_input: QualificationBackendInput,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    if backend_input.backend != OPEN3D_BACKEND:
        raise ValueError("Open3D component received another backend")
    return _execute_or_replay_stage2_fixture(
        executor=execute_open3d_fixture,
        original=_ORIGINAL_EXECUTE_OPEN3D_FIXTURE,
        fixture=backend_input.fixture,
        common=common,
        parameters=backend_input.parameters,
    )


def execute_qualification_pcl(
    *,
    backend_input: QualificationBackendInput,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        backend_input.backend != PCL_BACKEND
        or backend_input.pcl_cli is None
    ):
        raise ValueError("PCL component input binding is incomplete")
    return _execute_or_replay_stage2_fixture(
        executor=execute_pcl_fixture,
        original=_ORIGINAL_EXECUTE_PCL_FIXTURE,
        fixture=backend_input.fixture,
        common=common,
        parameters=backend_input.parameters,
        pcl_cli=backend_input.pcl_cli,
    )


def validate_qualification_result(
    *,
    spec: FormalLifecycleSpec,
    trial_row: Mapping[str, Any],
    common: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    from .phase_a_trial_result_schema import validate_phase_a_trial_result_strict

    trial = spec.execution_context.validate_trial_row(trial_row)
    result = validate_phase_a_trial_result_strict(value)
    if (
        result["planned_trial_id"] != trial["planned_trial_id"]
        or any(result.get(name) != expected for name, expected in common.items())
    ):
        raise ValueError("qualification backend result identity mismatch")
    return result


def validate_qualification_resume_result(
    *,
    spec: FormalLifecycleSpec,
    trial_row: Mapping[str, Any],
    common: Mapping[str, Any],
    path: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    from .phase_a_trial_resume import (
        validate_existing_trial_result_for_resume,
    )
    from .runtime_lifecycle_io import canonical_json_bytes

    spec.execution_context.validate_trial_row(trial_row)
    value = validate_existing_trial_result_for_resume(
        path,
        manifest_entry=entry,
        expected=common,
    )
    if path.read_bytes() != canonical_json_bytes(value):
        raise ValueError("qualification resume result is not canonical JSON")
    return value


def _completed_rows(completed_run: Any) -> list[dict[str, Any]]:
    rows = getattr(completed_run, "rows", None)
    if type(rows) not in {list, tuple} or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise TypeError("completed_run.rows must contain authenticated rows")
    return [deep_thaw(row) for row in rows]


def qualification_primary_analyzer(
    *,
    spec: FormalLifecycleSpec,
    completed_run: Any,
) -> dict[str, Any]:
    from .synthetic_confirmatory_v2_analysis import (
        analyze_v2_fixture_results,
    )

    del spec
    return analyze_v2_fixture_results(_completed_rows(completed_run))


def qualification_independent_verifier(
    *,
    spec: FormalLifecycleSpec,
    completed_run: Any,
) -> dict[str, Any]:
    from .synthetic_confirmatory_v2_independent_verifier import (
        independently_analyze_v2_fixture_results,
    )

    del spec
    return independently_analyze_v2_fixture_results(
        _completed_rows(completed_run)
    )


def qualification_difference_auditor(
    *,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
) -> dict[str, Any]:
    from .synthetic_confirmatory_v2_independent_verifier import (
        compare_v2_fixture_primary_and_independent,
    )

    return compare_v2_fixture_primary_and_independent(primary, independent)


def qualification_publisher(
    *,
    spec: FormalLifecycleSpec,
    completed_run: Any,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Adapt completed runtime evidence to the frozen 7/3/7 fixture publisher."""

    from .synthetic_confirmatory_v2_publisher import (
        publish_synthetic_confirmatory_v2_fixture,
    )

    rows = _completed_rows(completed_run)
    if len(rows) != 6:
        raise ValueError("fixture publisher requires six authenticated trials")
    run_manifest = {
        "schema_version": "synthetic_confirmatory_v2_fixture_run_v1",
        "backend_execution_count": 6,
        "fixture_snapshot_count": 3,
        "fixture_trial_count": 6,
        "formal_confirmatory_science_evaluated": False,
        "formal_v2_seed_reference_count": 0,
        "fresh_resume_scientific_equivalence": True,
        "resume_backend_execution_count": 0,
        "source_run_id": spec.run_id,
    }
    # The frozen v2 publisher requires an exact scientific envelope but allows
    # no extra source-run field.  Preserve source identity in the lifecycle
    # evidence, not inside the scientific publisher envelope.
    run_manifest.pop("source_run_id")
    return publish_synthetic_confirmatory_v2_fixture(
        primary=primary,
        independent=independent,
        run_manifest=run_manifest,
        artifact_dir=destination,
    )


def qualification_artifact_verifier(
    *,
    spec: FormalLifecycleSpec,
    artifact_path: Path,
    write_report: bool,
) -> dict[str, Any]:
    from .synthetic_confirmatory_v2_artifact_verifier import (
        verify_synthetic_confirmatory_v2_fixture_artifact,
    )

    del spec
    return verify_synthetic_confirmatory_v2_fixture_artifact(
        artifact_path,
        write_report=write_report,
    )


def qualification_git_gate(
    *,
    spec: FormalLifecycleSpec,
    checkpoint: str,
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


def qualification_run_contract_builder(
    *,
    spec: FormalLifecycleSpec,
) -> dict[str, Any]:
    """Build the seed-free immutable runtime contract from explicit spec data."""

    return {
        "schema_version": spec.manifest["qualification_run_contract_schema"],
        "execution_mode": spec.mode.value,
        "run_id": spec.run_id,
        "runtime_root": str(spec.runtime_paths.runtime_root),
        "runtime_paths": spec.runtime_paths.as_dict(),
        "manifest_sha256": spec.manifest_sha256,
        "protocol_sha256": spec.manifest["qualification_protocol_sha256"],
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
        "fixture_only": True,
        "formal_confirmatory_science_evaluated": False,
        "confirmatory_seed_allowed": False,
        "confirmatory_seed_access_count": 0,
        "planned_snapshot_count": 3,
        "planned_trial_count": 6,
        "backend_trial_counts": {
            OPEN3D_BACKEND: 3,
            PCL_BACKEND: 3,
        },
        "native_trial_count": 0,
        "workers": spec.workers,
    }


def _binding(
    component_id: str,
    callback: Any,
) -> ComponentBinding:
    implementation = Path(__file__).resolve()
    return ComponentBinding(
        component_id=component_id,
        callable=callback,
        implementation_path=implementation,
        implementation_sha256=_file_sha256(implementation),
    )


def build_qualification_component_bundle(
    context_name: str,
    *,
    execution_context: ExecutionContext,
    manifest_sha256: str,
    run_id: str,
    runtime_root: Path,
    publisher_inventory: PublisherInventory,
) -> FormalLifecycleComponents:
    """Bind the shared direct callables to one non-interchangeable context."""

    identity = _identity(context_name)
    return FormalLifecycleComponents(
        bundle_id=identity.bundle_id,
        bound_mode=ExecutionMode.QUALIFICATION,
        bound_contract_id=execution_context.contract.contract_id,
        bound_plan_id=execution_context.plan_id,
        bound_manifest_sha256=manifest_sha256,
        bound_run_id=run_id,
        bound_runtime_root=runtime_root,
        bound_publisher_inventory_sha256=publisher_inventory.payload_sha256,
        snapshot_materializer=_binding(
            "seed-free-snapshot-materializer-v1",
            qualification_snapshot_materializer,
        ),
        snapshot_reader=_binding(
            "authenticated-snapshot-reader-v1",
            qualification_snapshot_reader,
        ),
        snapshot_validator=_binding(
            "seed-free-authenticated-snapshot-validator-v1",
            qualification_snapshot_validator,
        ),
        trial_validator=_binding(
            "seed-free-trial-row-validator-v1",
            validate_qualification_trial_row,
        ),
        snapshot_lock_builder=_binding(
            "seed-free-snapshot-lock-builder-v1",
            qualification_snapshot_lock_builder,
        ),
        snapshot_lock_validator=_binding(
            "seed-free-snapshot-lock-validator-v1",
            qualification_snapshot_lock_validator,
        ),
        backend_input_builder=_binding(
            "authenticated-fixture-input-builder-v1",
            qualification_backend_input_builder,
        ),
        common_record_builder=_binding(
            "phase-a-common-record-builder-v1",
            qualification_common_record_builder,
        ),
        result_validator=_binding(
            "phase-a-result-validator-v1",
            validate_qualification_result,
        ),
        resume_result_validator=_binding(
            "phase-a-resume-result-validator-v1",
            validate_qualification_resume_result,
        ),
        primary_analyzer=_binding(
            "v2-fixture-primary-adapter-v1",
            qualification_primary_analyzer,
        ),
        independent_verifier=_binding(
            "v2-fixture-independent-adapter-v1",
            qualification_independent_verifier,
        ),
        difference_auditor=_binding(
            "v2-fixture-difference-adapter-v1",
            qualification_difference_auditor,
        ),
        publisher=_binding(
            "v2-fixture-publisher-adapter-v1",
            qualification_publisher,
        ),
        artifact_verifier=_binding(
            "v2-fixture-artifact-verifier-adapter-v1",
            qualification_artifact_verifier,
        ),
        git_gate=_binding(
            "runtime-git-gate-adapter-v1",
            qualification_git_gate,
        ),
        run_contract_builder=_binding(
            "seed-free-run-contract-builder-v1",
            qualification_run_contract_builder,
        ),
        backend_registry={
            OPEN3D_BACKEND: _binding(
                "real-open3d-fixture-backend-v1",
                execute_qualification_open3d,
            ),
            PCL_BACKEND: _binding(
                "real-pcl-fixture-backend-v1",
                execute_qualification_pcl,
            ),
        },
    )


def build_qualification_manifest(
    context_name: str,
    *,
    repository_root: Path,
    expected_commit: str,
    expected_branch: str,
    expected_tag: str,
    run_id: str | None = None,
    workers: int = 2,
    runtime_root: Path | None = None,
    snapshot_commit_delay_seconds: float = 0.0,
    trial_commit_delay_seconds: float = 0.0,
    entry_script: Path = Path(
        "scripts/run_version_agnostic_formal_lifecycle_fixture.py"
    ),
) -> dict[str, Any]:
    """Build the in-memory qualification manifest projection.

    Qualification manifests intentionally do not bind Confirmatory seeds or a
    Confirmatory namespace.  The raw file binding used by the spec factory is
    this committed provider module, avoiding a Git commit self-reference while
    the Git gate independently pins the candidate commit and annotated tag.
    """

    identity = _identity(context_name)
    selected_run_id = identity.run_id if run_id is None else run_id
    if (
        type(selected_run_id) is not str
        or not selected_run_id
        or selected_run_id.strip() != selected_run_id
    ):
        raise ValueError("qualification run ID is invalid")
    paths = build_qualification_paths(
        context_name, runtime_root=runtime_root
    )
    context = build_qualification_execution_context(
        context_name, runtime_root=paths.runtime_root
    )
    inventory = build_qualification_publisher_inventory(context_name)
    command = build_qualification_command_profile(
        context_name, entry_script=entry_script
    )
    identity_policy = GitIdentityPolicy(
        policy_id=f"{identity.id_prefix}-git-identity",
        expected_commit=expected_commit,
        expected_branch=expected_branch,
        expected_tag=expected_tag,
        checkpoint_prefix=identity.id_prefix.upper().replace("-", "_"),
    )
    provider_sha = _file_sha256(Path(__file__).resolve())
    provisional_bundle = build_qualification_component_bundle(
        context_name,
        execution_context=context,
        manifest_sha256=provider_sha,
        run_id=selected_run_id,
        runtime_root=paths.runtime_root,
        publisher_inventory=inventory,
    )
    snapshots = list(qualification_snapshot_plan(context_name))
    trials = list(qualification_trial_plan(context_name))
    protocol = {
        "schema_version": identity.protocol_schema,
        "protocol_version": identity.protocol_version,
        "fixture_only": True,
        "formal_confirmatory_science_evaluated": False,
        "confirmatory_seed_allowed": False,
        "confirmatory_seed_value_count": 0,
        "scenes": list(SCENES),
        "conditions": list(CONDITIONS),
        "backends": list(BACKENDS),
        "planned_snapshot_count": 3,
        "planned_trial_count": 6,
    }
    parameter_path = repository_root / FIXTURE_PARAMETER_LOCK_RELATIVE
    pcl_cli = repository_root / PCL_CLI_RELATIVE
    if _file_sha256(pcl_cli) != PCL_CLI_SHA256:
        raise ValueError("frozen qualification PCL CLI SHA changed")
    return {
        "schema_version": identity.manifest_schema,
        "qualification_context_name": context_name,
        "execution_mode": ExecutionMode.QUALIFICATION.value,
        "run_id": selected_run_id,
        "workers": workers,
        "plan_id": identity.plan_id,
        "protocol_version": identity.protocol_version,
        "qualification_protocol": protocol,
        "qualification_protocol_sha256": _canonical_sha256(protocol),
        "qualification_snapshot_lock_schema": identity.lock_schema,
        "qualification_run_contract_schema": (
            f"{identity.id_prefix.replace('-', '_')}_run_contract_v1"
        ),
        "qualification_implementation_sha256": provider_sha,
        "snapshot_plan_rows_sha256": canonical_plan_rows_sha256(snapshots),
        "trial_plan_rows_sha256": canonical_plan_rows_sha256(trials),
        "planned_snapshot_count": 3,
        "planned_trial_count": 6,
        "formal_lifecycle_paths": paths.as_dict(),
        "formal_lifecycle_components": (
            provisional_bundle.manifest_binding(
                repository_root=repository_root
            )
        ),
        "git_identity_policy": identity_policy.manifest_binding(),
        "publisher_inventory": inventory.report(),
        "publisher_inventory_sha256": inventory.payload_sha256,
        "command_profile": command.report(),
        "command_profile_sha256": command.payload_sha256,
        "backend_parameter_contract_path": (
            FIXTURE_PARAMETER_LOCK_RELATIVE.as_posix()
        ),
        "backend_parameter_contract_sha256": _file_sha256(parameter_path),
        "pcl_cli_path": PCL_CLI_RELATIVE.as_posix(),
        "pcl_cli_sha256": PCL_CLI_SHA256,
        "fixture_only": True,
        "formal_confirmatory_science_evaluated": False,
        "confirmatory_seed_values_included": False,
        "confirmatory_seed_access_count": 0,
        "confirmatory_rng_instantiation_count": 0,
        "native_backend_allowed": False,
        "durable_commit_delay_seconds": {
            "snapshot": float(snapshot_commit_delay_seconds),
            "trial": float(trial_commit_delay_seconds),
        },
    }


def build_qualification_spec(
    context_name: str,
    *,
    repository_root: str | Path,
    expected_commit: str,
    expected_branch: str,
    expected_tag: str,
    run_id: str | None = None,
    workers: int = 2,
    runtime_root: Path | None = None,
    manifest_path: Path | None = None,
    snapshot_commit_delay_seconds: float = 0.0,
    trial_commit_delay_seconds: float = 0.0,
    entry_script: Path = Path(
        "scripts/run_version_agnostic_formal_lifecycle_fixture.py"
    ),
) -> FormalLifecycleSpec:
    """Construct a fully bound A/B spec without touching its runtime root."""

    root = Path(repository_root).resolve()
    identity = _identity(context_name)
    selected_run_id = identity.run_id if run_id is None else run_id
    paths = build_qualification_paths(
        context_name, runtime_root=runtime_root
    )
    context = build_qualification_execution_context(
        context_name, runtime_root=paths.runtime_root
    )
    inventory = build_qualification_publisher_inventory(context_name)
    command = build_qualification_command_profile(
        context_name, entry_script=entry_script
    )
    identity_policy = GitIdentityPolicy(
        policy_id=f"{identity.id_prefix}-git-identity",
        expected_commit=expected_commit,
        expected_branch=expected_branch,
        expected_tag=expected_tag,
        checkpoint_prefix=identity.id_prefix.upper().replace("-", "_"),
    )
    provider_path = Path(__file__).resolve()
    provider_sha = _file_sha256(provider_path)
    expected_manifest = build_qualification_manifest(
        context_name,
        repository_root=root,
        expected_commit=expected_commit,
        expected_branch=expected_branch,
        expected_tag=expected_tag,
        run_id=selected_run_id,
        workers=workers,
        runtime_root=paths.runtime_root,
        snapshot_commit_delay_seconds=snapshot_commit_delay_seconds,
        trial_commit_delay_seconds=trial_commit_delay_seconds,
        entry_script=entry_script,
    )
    if manifest_path is None:
        raise ValueError(
            "qualification spec requires an explicit SHA-bound JSON manifest"
        )
    selected_manifest_path = Path(manifest_path).resolve(strict=False)
    if not selected_manifest_path.is_file():
        raise ValueError(
            "qualification manifest must be a canonical regular file"
        )
    manifest = _qualification_strict_object(selected_manifest_path)
    if manifest != expected_manifest:
        raise ValueError(
            "qualification manifest differs from the constructed contract"
        )
    manifest_sha = _file_sha256(selected_manifest_path)
    bundle = build_qualification_component_bundle(
        context_name,
        execution_context=context,
        manifest_sha256=manifest_sha,
        run_id=selected_run_id,
        runtime_root=paths.runtime_root,
        publisher_inventory=inventory,
    )
    return FormalLifecycleSpec(
        repository_root=root,
        execution_context=context,
        manifest=manifest,
        manifest_path=selected_manifest_path,
        manifest_sha256=manifest_sha,
        runtime_paths=paths,
        snapshot_plan=qualification_snapshot_plan(context_name),
        trial_plan=qualification_trial_plan(context_name),
        component_bundle=bundle,
        mode=ExecutionMode.QUALIFICATION,
        run_id=selected_run_id,
        workers=workers,
        identity_policy=identity_policy,
        publisher_inventory=inventory,
        command_profile=command,
    )


def build_context_a_spec(**kwargs: Any) -> FormalLifecycleSpec:
    return build_qualification_spec(CONTEXT_A, **kwargs)


def build_context_b_spec(**kwargs: Any) -> FormalLifecycleSpec:
    return build_qualification_spec(CONTEXT_B, **kwargs)


def qualification_context_report(context_name: str) -> dict[str, Any]:
    """Return a seed-free, JSON-native declaration without building a spec."""

    identity = _identity(context_name)
    snapshots = qualification_snapshot_plan(context_name)
    trials = qualification_trial_plan(context_name)
    return {
        "context_name": context_name,
        "run_id": identity.run_id,
        "runtime_root": str(identity.runtime_root),
        "protocol_version": identity.protocol_version,
        "plan_id": identity.plan_id,
        "contract_id": identity.contract_id,
        "bundle_id": identity.bundle_id,
        "snapshot_plan_schema": identity.snapshot_plan_schema,
        "trial_plan_schema": identity.trial_plan_schema,
        "metadata_schema": identity.metadata_schema,
        "snapshot_schema": identity.snapshot_schema,
        "lineage_schema": identity.lineage_schema,
        "snapshot_lock_schema": identity.lock_schema,
        "planned_snapshot_count": len(snapshots),
        "planned_trial_count": len(trials),
        "snapshot_plan_rows_sha256": canonical_plan_rows_sha256(
            list(snapshots)
        ),
        "trial_plan_rows_sha256": canonical_plan_rows_sha256(list(trials)),
        "confirmatory_seed_allowed": False,
        "confirmatory_seed_value_count": 0,
        "backend_trial_counts": {
            OPEN3D_BACKEND: 3,
            PCL_BACKEND: 3,
        },
        "native_trial_count": 0,
    }


__all__ += [
    "BACKENDS",
    "CONDITIONS",
    "CONTEXT_A",
    "CONTEXT_A_RUN_ID",
    "CONTEXT_A_RUNTIME_ROOT",
    "CONTEXT_B",
    "CONTEXT_B_RUN_ID",
    "CONTEXT_B_RUNTIME_ROOT",
    "CONTEXT_NAMES",
    "EXPECTED_OUTCOMES",
    "OPEN3D_BACKEND",
    "PCL_BACKEND",
    "QUALIFICATION_RUNTIME_BASE",
    "QualificationBackendInput",
    "SCENES",
    "SNAPSHOT_FIELDS",
    "TRIAL_FIELDS",
    "build_context_a_spec",
    "build_context_b_spec",
    "build_qualification_command_profile",
    "build_qualification_component_bundle",
    "build_qualification_execution_context",
    "build_qualification_manifest",
    "build_qualification_paths",
    "build_qualification_publisher_inventory",
    "build_qualification_spec",
    "execute_qualification_open3d",
    "execute_qualification_pcl",
    "qualification_artifact_verifier",
    "qualification_backend_input_builder",
    "qualification_common_record_builder",
    "qualification_context_report",
    "qualification_difference_auditor",
    "qualification_git_gate",
    "qualification_independent_verifier",
    "qualification_primary_analyzer",
    "qualification_publisher",
    "qualification_run_contract_builder",
    "qualification_snapshot_lock_builder",
    "qualification_snapshot_lock_validator",
    "qualification_snapshot_materializer",
    "qualification_snapshot_plan",
    "qualification_snapshot_reader",
    "qualification_snapshot_validator",
    "qualification_trial_plan",
    "validate_qualification_result",
    "validate_qualification_resume_result",
    "validate_qualification_snapshot_row",
    "validate_qualification_trial_row",
]
