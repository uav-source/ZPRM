#!/usr/bin/env python3
"""Run the production Boreas v2 Stage-2 query passes (never registration)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import stat
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import scipy


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_stage2_remote import (  # noqa: E402
    AuthorizedRemoteObject,
    ReconciledRemoteInventory,
    RemoteObjectIdentity,
    StrictAllowlistDownloader,
    load_frozen_allowlist,
    reconcile_remote_metadata,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_authorization import (  # noqa: E402
    EXPECTED_ALLOWLIST_SHA256,
    EXPECTED_STORAGE_BUDGET_SHA256,
    VerifiedStage2Authorization,
    verify_boreas_v2_stage2_download_authorization,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_closure import (  # noqa: E402
    write_selection_prerequisites,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_preprocessing import (  # noqa: E402
    BoreasLidarPoseIndex,
    load_t_applanix_lidar,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_query_runner import (  # noqa: E402
    BoreasV2Stage2QueryRunner,
    QueryRuntimeLease,
    QueryRunnerConfig,
    QueryRunnerDependencies,
    _QueryJournal,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (  # noqa: E402
    TargetGeometryContext,
)
from phase_a_harness.real_data_preparation.guard import NoRegistrationGuard  # noqa: E402
from phase_a_harness.real_data_preparation.io import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)
from phase_a_harness.real_data_preparation.stage2_checkpoint import (  # noqa: E402
    cleanup_partial_temporaries,
)
from phase_a_harness.real_data_preparation.stage2_disk_gate import (  # noqa: E402
    Stage2DiskGate,
)

try:  # Bind installed Open3D registration entrypoints for the guard lifetime.
    import open3d as _open3d  # type: ignore[import-not-found]  # noqa: E402
except ImportError:  # pragma: no cover - source-only test environments
    _open3d = None


DEFAULT_AWS = Path(
    "/home/lj/zero_perturbation_data/boreas_stage1_v1/"
    "tools/awscli-venv/bin/aws"
)
REMOTE_INVENTORY_SCHEMA = "zprm.boreas.v2.stage2.remote_inventory.v1"
MEASUREMENT_SCHEMA = "zprm.boreas.v2.stage2.target_context_capacity_measurement.v1"
PLAN_SCHEMA = "zprm.boreas.v2.stage2.target_context_resource_plan.v1"
PROVENANCE_SCHEMA = "zprm.boreas.v2.stage2.target_context_measurement_provenance.v1"
MEASUREMENT_METHOD = "MAX_RSS_PREPARE_NORMALS_KDTREE_SAME_PINNED_ENVIRONMENT"
DEFAULT_MEMORY_SAFETY_MARGIN_BYTES = 5 * 1024**3
MAX_TARGET_CONTEXT_PROBE_JITTER_BYTES = 512 * 1024**2
CAPACITY_INTENT_SCHEMA = "zprm.boreas.v2.stage2.target_context_capacity_intent.v1"


class BoreasV2Stage2QueryCLIError(RuntimeError):
    """Production query orchestration could not authenticate its inputs."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--monitored-disk-path", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--aws-executable", type=Path, default=DEFAULT_AWS)
    parser.add_argument(
        "--phase",
        choices=(
            "target-context",
            "first-pass",
            "freeze-selection",
            "second-pass",
            "export",
            "manifest",
            "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--memory-safety-margin-bytes",
        type=int,
        default=DEFAULT_MEMORY_SAFETY_MARGIN_BYTES,
    )
    # Private subprocess mode.  It is deliberately not a documented phase.
    parser.add_argument("--_measure-target-context", action="store_true")
    parser.add_argument("--_target-path", type=Path)
    parser.add_argument("--_target-sha256")
    parser.add_argument("--_target-point-count", type=int)
    return parser


def _canonical_directory(path: Path, *, label: str) -> Path:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
        or path.resolve(strict=True) != path
    ):
        raise BoreasV2Stage2QueryCLIError(f"{label} must be a canonical directory")
    return path


def _regular_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise BoreasV2Stage2QueryCLIError(f"{label} is absent or unsafe")
    if path.resolve(strict=True) != path:
        raise BoreasV2Stage2QueryCLIError(f"{label} is not canonical")
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise BoreasV2Stage2QueryCLIError(f"{label} must be regular with nlink=1")
    return path


def _json_object(path: Path, *, canonical: bool = True) -> dict[str, Any]:
    source = _regular_file(path, label=path.name)
    try:
        payload = source.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoreasV2Stage2QueryCLIError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BoreasV2Stage2QueryCLIError(f"JSON is not an object: {path}")
    if canonical and payload != canonical_json_bytes(value):
        raise BoreasV2Stage2QueryCLIError(f"JSON is noncanonical: {path}")
    return value


def load_authenticated_remote_inventory(
    *, allowlist_path: Path, expected_allowlist_sha256: str, inventory_path: Path
) -> ReconciledRemoteInventory:
    """Rebuild the exact reconciled inventory from the immutable runtime freeze."""

    allowlist = load_frozen_allowlist(
        allowlist_path, expected_sha256=expected_allowlist_sha256
    )
    value = _json_object(inventory_path)
    expected_fields = {
        "allowlist_sha256",
        "inventory_file_payload_sha256",
        "inventory_sha256",
        "objects",
        "schema",
    }
    if set(value) != expected_fields or value["schema"] != REMOTE_INVENTORY_SCHEMA:
        raise BoreasV2Stage2QueryCLIError("remote inventory exact schema differs")
    unsigned = {key: child for key, child in value.items() if key != "inventory_file_payload_sha256"}
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != value[
        "inventory_file_payload_sha256"
    ]:
        raise BoreasV2Stage2QueryCLIError("remote inventory payload self-hash differs")
    if value["allowlist_sha256"] != allowlist.sha256 or not isinstance(
        value["objects"], list
    ):
        raise BoreasV2Stage2QueryCLIError("remote inventory allowlist binding differs")
    rows = [RemoteObjectIdentity.from_mapping(row) for row in value["objects"]]
    inventory = reconcile_remote_metadata(allowlist, rows)
    if inventory.inventory_sha256 != value["inventory_sha256"]:
        raise BoreasV2Stage2QueryCLIError("remote inventory identity SHA differs")
    return inventory


def frozen_windows_from_pair(pair_path: Path) -> tuple[dict[str, int], ...]:
    value = _json_object(pair_path)
    rows = value.get("primary_complete_five_second_windows")
    if not isinstance(rows, list):
        raise BoreasV2Stage2QueryCLIError("pair authority lacks frozen windows")
    output: list[dict[str, int]] = []
    previous_end: int | None = None
    for expected, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise BoreasV2Stage2QueryCLIError("frozen window row is not an object")
        try:
            start = int(Decimal(str(row["start_time_s"])) * 1_000_000)
            end = int(Decimal(str(row["end_time_s"])) * 1_000_000)
            duration = int(Decimal(str(row["duration_s"])) * 1_000_000)
        except (KeyError, InvalidOperation, ValueError) as exc:
            raise BoreasV2Stage2QueryCLIError("frozen window time is invalid") from exc
        if (
            row.get("window_index") != expected
            or isinstance(row.get("interval_index"), bool)
            or not isinstance(row.get("interval_index"), int)
            or duration != 5_000_000
            or end - start != duration
            or (previous_end is not None and start < previous_end)
        ):
            raise BoreasV2Stage2QueryCLIError("frozen window semantics differ")
        output.append(
            {
                "duration_us": duration,
                "end_time_us": end,
                "interval_index": int(row["interval_index"]),
                "start_time_us": start,
                "window_index": expected,
            }
        )
        previous_end = end
    if len(output) != 246:
        raise BoreasV2Stage2QueryCLIError("production window count is not 246")
    return tuple(output)


def _target_freeze(runtime_root: Path) -> tuple[dict[str, Any], Path]:
    value = _json_object(runtime_root / "evidence/target_map_freeze_manifest.json")
    target_relative = PurePosixPath(str(value.get("target_map_path", "")))
    if target_relative.is_absolute() or any(
        part in {"", ".", ".."} for part in target_relative.parts
    ):
        raise BoreasV2Stage2QueryCLIError("target-map relative path is unsafe")
    target = runtime_root.joinpath(*target_relative.parts)
    _regular_file(target, label="target map")
    if (
        sha256_file(target) != value.get("target_map_sha256")
        or target.stat().st_size != value.get("target_map_size_bytes")
        or not isinstance(value.get("target_point_count"), int)
        or isinstance(value.get("target_point_count"), bool)
        or value["target_point_count"] <= 0
    ):
        raise BoreasV2Stage2QueryCLIError("target map/freeze identity differs")
    return value, target


def _measure_target_context_child(arguments: argparse.Namespace) -> int:
    if sys.platform != "linux":
        raise BoreasV2Stage2QueryCLIError("target-context MaxRSS units are pinned to Linux")
    if None in (
        arguments._target_path,
        arguments._target_sha256,
        arguments._target_point_count,
    ):
        raise BoreasV2Stage2QueryCLIError("internal measurement arguments are incomplete")
    target_path = _regular_file(arguments._target_path, label="measurement target map")
    if sha256_file(target_path) != arguments._target_sha256:
        raise BoreasV2Stage2QueryCLIError("measurement target SHA differs")
    target = np.load(target_path, mmap_mode="r", allow_pickle=False)
    expected_count = int(arguments._target_point_count)
    if (
        not isinstance(target, np.memmap)
        or target.dtype != np.dtype("<f8")
        or target.shape != (expected_count, 3)
        or not target.flags.c_contiguous
    ):
        raise BoreasV2Stage2QueryCLIError("measurement target layout differs")
    context = TargetGeometryContext.prepare(target)
    if context.target_points.shape != (expected_count, 3):
        raise BoreasV2Stage2QueryCLIError("prepared target context count differs")
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    if peak <= 0:
        raise BoreasV2Stage2QueryCLIError("target-context MaxRSS is invalid")
    value = {
        "measured_peak_memory_bytes": peak,
        "measurement_method": MEASUREMENT_METHOD,
        "numpy_version": np.__version__,
        "schema": MEASUREMENT_SCHEMA,
        "scipy_version": scipy.__version__,
        "target_map_sha256": arguments._target_sha256,
        "target_point_count": expected_count,
    }
    sys.stdout.buffer.write(canonical_json_bytes(value))
    return 0


def ensure_target_context_capacity_evidence(
    *,
    runtime_root: Path,
    target_path: Path,
    target_sha256: str,
    target_point_count: int,
    generator_path: Path,
    safety_margin_bytes: int,
    command_runner: Any = None,
) -> tuple[Path, Path, Path]:
    """Run one fixed isolated MaxRSS probe, or authenticate its immutable resume."""

    if (
        isinstance(safety_margin_bytes, bool)
        or not isinstance(safety_margin_bytes, int)
        or safety_margin_bytes < DEFAULT_MEMORY_SAFETY_MARGIN_BYTES
    ):
        raise BoreasV2Stage2QueryCLIError("memory safety margin must be at least 5 GiB")
    generator = _regular_file(generator_path, label="target-context generator")
    target = _regular_file(target_path, label="target-context target map")
    if sha256_file(target) != target_sha256:
        raise BoreasV2Stage2QueryCLIError("target-context target SHA differs")
    try:
        target_array = np.load(target, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise BoreasV2Stage2QueryCLIError(
            "target-context target is not canonical NPY"
        ) from exc
    if (
        not isinstance(target_array, np.memmap)
        or target_array.dtype != np.dtype("<f8")
        or target_array.shape != (target_point_count, 3)
        or not target_array.flags.c_contiguous
    ):
        raise BoreasV2Stage2QueryCLIError("target-context target layout differs")
    del target_array
    evidence_root = _canonical_directory(
        runtime_root / "evidence", label="runtime evidence root"
    )
    measurement_path = evidence_root / "target_context_capacity_measurement.json"
    plan_path = evidence_root / "target_context_resource_plan.json"
    provenance_path = evidence_root / "target_context_capacity_measurement_provenance.json"
    intent_path = runtime_root / "checkpoints/target_context_capacity_intent.json"
    intent_unsigned = {
        "generator_sha256": sha256_file(generator),
        "measurement_path": measurement_path.relative_to(runtime_root).as_posix(),
        "plan_path": plan_path.relative_to(runtime_root).as_posix(),
        "provenance_path": provenance_path.relative_to(runtime_root).as_posix(),
        "safety_margin_bytes": safety_margin_bytes,
        "schema": CAPACITY_INTENT_SCHEMA,
        "target_map_sha256": target_sha256,
        "target_point_count": target_point_count,
    }
    intent = {
        **intent_unsigned,
        "intent_sha256": hashlib.sha256(
            canonical_json_bytes(intent_unsigned)
        ).hexdigest(),
    }
    existing = [path.exists() or path.is_symlink() for path in (measurement_path, plan_path, provenance_path)]
    journal = _QueryJournal(runtime_root / "checkpoints/query_journal.jsonl")
    barrier = journal.barrier("TARGET_FROZEN")
    if barrier is not None and all(existing):
        expected_hashes = {
            "target_context_resource_plan_sha256": sha256_file(plan_path),
            "target_context_measurement_evidence_sha256": sha256_file(
                measurement_path
            ),
            "target_context_measurement_provenance_sha256": sha256_file(
                provenance_path
            ),
        }
        if any(barrier["payload"].get(key) != value for key, value in expected_hashes.items()):
            raise BoreasV2Stage2QueryCLIError(
                "target-context evidence differs from the first durable query barrier"
            )
    elif any(existing):
        # Without the first query barrier these bytes have no durable authority.
        # Only a matching fixed intent permits safe deletion/remeasurement.
        if (
            not intent_path.is_file()
            or intent_path.is_symlink()
            or _json_object(intent_path) != intent
        ):
            raise BoreasV2Stage2QueryCLIError(
                "unbound/preseeded target-context evidence is forbidden"
            )
        for path in (measurement_path, plan_path, provenance_path):
            if path.exists() or path.is_symlink():
                _regular_file(path, label=f"partial target-context {path.name}").unlink()
        existing = [False, False, False]

    if not all(existing) and not intent_path.exists():
        atomic_write_json(intent_path, intent)
    elif not all(existing) and (
        intent_path.is_symlink()
        or not intent_path.is_file()
        or _json_object(intent_path) != intent
    ):
        raise BoreasV2Stage2QueryCLIError("target-context capacity intent differs")

    runner = subprocess.run if command_runner is None else command_runner

    def run_probe() -> dict[str, Any]:
        argv = (
            sys.executable,
            str(generator),
            "--_measure-target-context",
            "--_target-path",
            str(target_path),
            "--_target-sha256",
            target_sha256,
            "--_target-point-count",
            str(target_point_count),
        )
        result = runner(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", "replace") if isinstance(result.stderr, bytes) else str(result.stderr)
            raise BoreasV2Stage2QueryCLIError(
                f"isolated target-context measurement failed: {stderr.strip()}"
            )
        payload = result.stdout if isinstance(result.stdout, bytes) else str(result.stdout).encode()
        try:
            measurement = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise BoreasV2Stage2QueryCLIError("measurement child returned invalid JSON") from exc
        if payload != canonical_json_bytes(measurement):
            raise BoreasV2Stage2QueryCLIError("measurement child output is noncanonical")
        return measurement

    generated_now = not all(existing)
    if generated_now:
        measurement = run_probe()
        atomic_write_json(measurement_path, measurement)
        measurement_sha = sha256_file(measurement_path)
        provenance = {
            "generator_sha256": sha256_file(generator),
            "generator_path": str(generator),
            "measurement_evidence_sha256": measurement_sha,
            "measurement_method": MEASUREMENT_METHOD,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "python_executable": str(Path(sys.executable).resolve(strict=True)),
            "python_executable_sha256": sha256_file(
                Path(sys.executable).resolve(strict=True)
            ),
            "python_version": platform.python_version(),
            "schema": PROVENANCE_SCHEMA,
            "scipy_version": scipy.__version__,
            "subprocess_mode": "ISOLATED_FIXED_GENERATOR_BEFORE_ANY_QUERY_PAYLOAD",
            "target_map_sha256": target_sha256,
            "target_point_count": target_point_count,
        }
        atomic_write_json(provenance_path, provenance)
        peak = measurement.get("measured_peak_memory_bytes")
        if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
            raise BoreasV2Stage2QueryCLIError("measurement peak is invalid")
        plan = {
            "estimated_peak_memory_bytes": peak,
            "measurement_evidence_sha256": measurement_sha,
            "minimum_live_available_memory_bytes": peak + safety_margin_bytes,
            "numpy_version": np.__version__,
            "production_approved": True,
            "safety_margin_bytes": safety_margin_bytes,
            "schema": PLAN_SCHEMA,
            "scipy_version": scipy.__version__,
            "target_map_sha256": target_sha256,
            "target_point_count": target_point_count,
        }
        atomic_write_json(plan_path, plan)

    measurement = _json_object(measurement_path)
    plan = _json_object(plan_path)
    provenance = _json_object(provenance_path)
    expected_measurement = {
        "measured_peak_memory_bytes",
        "measurement_method",
        "numpy_version",
        "schema",
        "scipy_version",
        "target_map_sha256",
        "target_point_count",
    }
    if set(measurement) != expected_measurement or any(
        (
            measurement["schema"] != MEASUREMENT_SCHEMA,
            measurement["measurement_method"] != MEASUREMENT_METHOD,
            measurement["target_map_sha256"] != target_sha256,
            measurement["target_point_count"] != target_point_count,
            measurement["numpy_version"] != np.__version__,
            measurement["scipy_version"] != scipy.__version__,
        )
    ):
        raise BoreasV2Stage2QueryCLIError("target-context measurement semantics differ")
    peak = measurement["measured_peak_memory_bytes"]
    if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
        raise BoreasV2Stage2QueryCLIError("target-context measurement peak differs")
    measurement_sha = sha256_file(measurement_path)
    expected_plan_fields = {
        "estimated_peak_memory_bytes",
        "measurement_evidence_sha256",
        "minimum_live_available_memory_bytes",
        "numpy_version",
        "production_approved",
        "safety_margin_bytes",
        "schema",
        "scipy_version",
        "target_map_sha256",
        "target_point_count",
    }
    expected_provenance_fields = {
        "generator_path",
        "generator_sha256",
        "measurement_evidence_sha256",
        "measurement_method",
        "numpy_version",
        "platform",
        "python_executable",
        "python_executable_sha256",
        "python_version",
        "schema",
        "scipy_version",
        "subprocess_mode",
        "target_map_sha256",
        "target_point_count",
    }
    if (
        set(plan) != expected_plan_fields
        or set(provenance) != expected_provenance_fields
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("measurement_evidence_sha256") != measurement_sha
        or plan.get("target_map_sha256") != target_sha256
        or plan.get("target_point_count") != target_point_count
        or plan.get("estimated_peak_memory_bytes") != peak
        or plan.get("safety_margin_bytes") != safety_margin_bytes
        or plan.get("minimum_live_available_memory_bytes")
        != peak + safety_margin_bytes
        or plan.get("production_approved") is not True
        or provenance.get("schema") != PROVENANCE_SCHEMA
        or provenance.get("generator_path") != str(generator)
        or provenance.get("generator_sha256") != sha256_file(generator)
        or provenance.get("measurement_evidence_sha256") != measurement_sha
        or provenance.get("measurement_method") != MEASUREMENT_METHOD
        or provenance.get("numpy_version") != np.__version__
        or provenance.get("scipy_version") != scipy.__version__
        or provenance.get("platform") != platform.platform()
        or provenance.get("python_executable")
        != str(Path(sys.executable).resolve(strict=True))
        or provenance.get("python_executable_sha256")
        != sha256_file(Path(sys.executable).resolve(strict=True))
        or provenance.get("python_version") != platform.python_version()
        or provenance.get("target_map_sha256") != target_sha256
        or provenance.get("target_point_count") != target_point_count
        or provenance.get("subprocess_mode")
        != "ISOLATED_FIXED_GENERATOR_BEFORE_ANY_QUERY_PAYLOAD"
    ):
        raise BoreasV2Stage2QueryCLIError("target-context plan/provenance differs")
    # Every production process repeats the fixed isolated target build before
    # any query payload.  The durable trio is therefore an upper-bound claim,
    # not a self-authorizing preseed: a coherently re-signed lowered peak fails.
    observed = measurement if generated_now else run_probe()
    observed_peak = observed.get("measured_peak_memory_bytes")
    if (
        set(observed) != expected_measurement
        or observed.get("schema") != MEASUREMENT_SCHEMA
        or observed.get("measurement_method") != MEASUREMENT_METHOD
        or observed.get("target_map_sha256") != target_sha256
        or observed.get("target_point_count") != target_point_count
        or observed.get("numpy_version") != np.__version__
        or observed.get("scipy_version") != scipy.__version__
        or isinstance(observed_peak, bool)
        or not isinstance(observed_peak, int)
        or observed_peak <= 0
        or observed_peak > peak + MAX_TARGET_CONTEXT_PROBE_JITTER_BYTES
        or observed_peak > plan["minimum_live_available_memory_bytes"]
    ):
        raise BoreasV2Stage2QueryCLIError(
            "fresh isolated target-context probe exceeds or differs from the "
            "approved capacity evidence"
        )
    return measurement_path, plan_path, provenance_path


def commit_target_context_capacity_evidence(runtime_root: Path) -> None:
    """Clear the fixed intent only after TARGET_FROZEN binds all three hashes."""

    intent_path = runtime_root / "checkpoints/target_context_capacity_intent.json"
    journal = _QueryJournal(runtime_root / "checkpoints/query_journal.jsonl")
    barrier = journal.barrier("TARGET_FROZEN")
    if barrier is None:
        raise BoreasV2Stage2QueryCLIError(
            "target-context evidence was not durably bound before intent commit"
        )
    paths = {
        "target_context_resource_plan_sha256": runtime_root
        / "evidence/target_context_resource_plan.json",
        "target_context_measurement_evidence_sha256": runtime_root
        / "evidence/target_context_capacity_measurement.json",
        "target_context_measurement_provenance_sha256": runtime_root
        / "evidence/target_context_capacity_measurement_provenance.json",
    }
    if any(
        barrier["payload"].get(field) != sha256_file(path)
        for field, path in paths.items()
    ):
        raise BoreasV2Stage2QueryCLIError(
            "target-context commit barrier does not bind exact evidence"
        )
    if intent_path.exists():
        _regular_file(intent_path, label="target-context capacity intent").unlink()
        directory = os.open(
            intent_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _authenticate_replay_resume(runtime_root: Path, inventory: ReconciledRemoteInventory) -> None:
    replay = _regular_file(runtime_root / "map/transformed_xyz.f64le", label="map replay")
    ledger = _regular_file(runtime_root / "map/replay_ledger.jsonl", label="map replay ledger")
    expected = sum(item.size_bytes for item in inventory.for_role("TARGET_MAP"))
    metadata = os.stat(replay, follow_symlinks=False)
    if metadata.st_size != expected or int(metadata.st_blocks) * 512 < expected:
        raise BoreasV2Stage2QueryCLIError("query RESUME lacks exact nonsparse map replay")
    if ledger.stat().st_size <= 0:
        raise BoreasV2Stage2QueryCLIError("query RESUME map ledger is empty")


def prepare_authenticated_pending_transfer(
    *,
    runtime_root: Path,
    temporary_download_root: Path,
    inventory: ReconciledRemoteInventory,
) -> bool:
    """Discard bare-intent bytes; only a durable DOWNLOADED event is adoptable.

    A TRANSFER_INTENT authenticates the planned path and remote identity, not the
    bytes at that path.  In particular, matching length is never sufficient to
    manufacture a receipt after a crash.
    """

    journal_path = runtime_root / "checkpoints/query_journal.jsonl"
    journal = _QueryJournal(journal_path)
    pending = journal.pending_transfer_intents
    marker_name = ".stage2_temporary_root.json"
    if not temporary_download_root.exists():
        return False
    root = _canonical_directory(temporary_download_root, label="download temporary root")
    entries = [entry for entry in root.iterdir() if entry.name != marker_name]
    if not pending:
        if entries:
            cleanup_partial_temporaries(root)
        return False
    if len(pending) != 1:
        raise BoreasV2Stage2QueryCLIError("multiple pending query transfers exist")
    intent = pending[0]
    item = inventory.by_key.get(intent["object_key"])
    if item is None:
        raise BoreasV2Stage2QueryCLIError("pending transfer is outside inventory")
    digest = hashlib.sha256(item.key.encode("utf-8")).hexdigest()[:24]
    expected_path = root / f"{item.frozen.ordinal:06d}-{digest}.bin.partial"
    expected_relative = expected_path.relative_to(runtime_root).as_posix()
    expected_identity = {
        "etag": item.etag,
        "last_modified": item.last_modified,
        "remote_size_bytes": item.size_bytes,
        "sequence_id": item.frozen.sequence_id,
        "temporary_relative_path": expected_relative,
        "timestamp_us": item.frozen.timestamp_us,
    }
    if intent["payload"] != expected_identity:
        raise BoreasV2Stage2QueryCLIError("pending transfer identity differs")
    if any(entry != expected_path for entry in entries):
        raise BoreasV2Stage2QueryCLIError("temporary root contains an unowned payload")
    if not entries:
        return False
    metadata = os.lstat(expected_path)
    if (
        expected_path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise BoreasV2Stage2QueryCLIError("pending transfer payload is unsafe")
    expected_path.unlink()
    directory = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return False


def execute_query_phases(
    runner: BoreasV2Stage2QueryRunner,
    *,
    phase: str,
    prerequisite_writer: Any,
) -> dict[str, Any]:
    """Execute the stable query runner API in its only production ordering."""

    result: dict[str, Any] = {"phase": phase, "registration_execution_count": 0}
    if phase in {"target-context", "first-pass", "all"}:
        context = runner.preflight_target_context()
        result["target_context_point_count"] = int(context.target_points.shape[0])
    if phase in {"first-pass", "all"}:
        result["first_pass"] = runner.run_first_pass().__dict__
    if phase in {"freeze-selection", "all"}:
        result["selection_freeze"] = runner.freeze_selection()
    if phase in {"second-pass", "all"}:
        result["second_pass"] = runner.run_second_pass().__dict__
    if phase in {"export", "all"}:
        result["query_audit"] = runner.export_query_evidence()
    if phase in {"manifest", "all"}:
        result["selection_prerequisites"] = prerequisite_writer()
        result["selection_manifest"] = runner.build_selection_manifest()
    return result


def _production_main(arguments: argparse.Namespace) -> int:
    required = {
        "--data-root": arguments.data_root,
        "--runtime-root": arguments.runtime_root,
        "--temporary-root": arguments.temporary_root,
        "--monitored-disk-path": arguments.monitored_disk_path,
        "--authorization": arguments.authorization,
    }
    absent = [name for name, value in required.items() if value is None]
    if absent:
        raise BoreasV2Stage2QueryCLIError(
            f"production query arguments are required: {', '.join(absent)}"
        )
    repository = _canonical_directory(arguments.repository.resolve(strict=True), label="repository")
    data_root = _canonical_directory(arguments.data_root.resolve(strict=True), label="data root")
    runtime_root = _canonical_directory(arguments.runtime_root.resolve(strict=True), label="runtime root")
    temporary_root = _canonical_directory(arguments.temporary_root.resolve(strict=True), label="temporary root")
    monitored = _canonical_directory(arguments.monitored_disk_path.resolve(strict=True), label="monitored disk")
    authorization_path = arguments.authorization.resolve(strict=True)
    aws = _regular_file(arguments.aws_executable.resolve(strict=True), label="AWS executable")
    if not os.access(aws, os.X_OK):
        raise BoreasV2Stage2QueryCLIError("AWS executable is not executable")

    allowlist_path = repository / (
        "frozen_assets/public_data_external_validation_v2_boreas_stage1/"
        "boreas_v2_stage2_download_allowlist.csv"
    )
    pair_path = repository / (
        "frozen_assets/public_data_external_validation_v2_boreas_stage1/"
        "boreas_v2_pair_selection.json"
    )
    stage1_manifest = repository / (
        "frozen_assets/public_data_external_validation_v2_boreas_stage1/frozen_manifest.json"
    )
    storage_manifest = repository / (
        "frozen_assets/boreas_v2_stage2_storage_optimization/frozen_manifest.json"
    )
    budget_path = repository / (
        "frozen_assets/boreas_v2_stage2_storage_optimization/"
        "boreas_v2_stage2_disk_budget_optimized.json"
    )
    contract_path = repository / "protocols/boreas_v2_stage2_preprocessing_contract.json"
    backend_path = repository / "frozen_assets/backend_parameter_contract.json"
    witness_path = repository / (
        "src/phase_a_harness/real_data_preparation/"
        "boreas_v2_stage2_canonical_witness.py"
    )

    os.environ.setdefault("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard(open3d_module=_open3d) as guard:
      with QueryRuntimeLease(runtime_root) as runtime_lease:
        authorization = verify_boreas_v2_stage2_download_authorization(
            repository=repository,
            data_root=data_root,
            runtime_root=runtime_root,
            temporary_root=temporary_root,
            monitored_disk_path=monitored,
            authorization_path=authorization_path,
            no_registration_guard=guard,
        )
        if not isinstance(authorization, VerifiedStage2Authorization) or not authorization.formally_verified:
            raise BoreasV2Stage2QueryCLIError("formal production authorization was not minted")
        inventory = load_authenticated_remote_inventory(
            allowlist_path=allowlist_path,
            expected_allowlist_sha256=authorization.allowlist_sha256,
            inventory_path=runtime_root / "checkpoints/remote_inventory.json",
        )
        if inventory.allowlist_sha256 != EXPECTED_ALLOWLIST_SHA256:
            raise BoreasV2Stage2QueryCLIError("production allowlist constant differs")
        _authenticate_replay_resume(runtime_root, inventory)
        disk_gate = Stage2DiskGate.from_frozen_budget(
            monitored,
            budget_path=budget_path,
            expected_budget_sha256=EXPECTED_STORAGE_BUDGET_SHA256,
            audit_log_path=runtime_root / "checkpoints/disk_gate_events.jsonl",
            storage_mode="RECOMMENDED_OPERATIONAL",
        )
        authorization.bind_disk_gate(disk_gate)
        disk_gate.assert_resume(operation_id="boreas-v2-stage2-query-resume")

        freeze, target_path = _target_freeze(runtime_root)
        measurement_path, resource_plan_path, provenance_path = (
            ensure_target_context_capacity_evidence(
                runtime_root=runtime_root,
                target_path=target_path,
                target_sha256=str(freeze["target_map_sha256"]),
                target_point_count=int(freeze["target_point_count"]),
                generator_path=Path(__file__).resolve(strict=True),
                safety_margin_bytes=arguments.memory_safety_margin_bytes,
            )
        )
        if arguments.phase == "target-context":
            # Construction below performs the independent live-MemAvailable gate.
            pass

        contract = _json_object(contract_path)
        primary = contract.get("primary_pair")
        if not isinstance(primary, Mapping):
            raise BoreasV2Stage2QueryCLIError("preprocessing PRIMARY pair is absent")
        pair = _json_object(pair_path).get("PRIMARY_PAIR")
        if not isinstance(pair, Mapping) or {
            "map_sequence_id": primary.get("map_sequence_id"),
            "query_sequence_id": primary.get("query_sequence_id"),
        } != {
            "map_sequence_id": pair.get("map_sequence_id"),
            "query_sequence_id": pair.get("query_sequence_id"),
        }:
            raise BoreasV2Stage2QueryCLIError("pair/preprocessing authority differs")
        map_sequence = str(primary["map_sequence_id"])
        query_sequence = str(primary["query_sequence_id"])
        payload_root = data_root / "stage1_payload"
        map_pose_path = payload_root / map_sequence / "applanix/lidar_poses.csv"
        query_pose_path = payload_root / query_sequence / "applanix/lidar_poses.csv"
        map_pose = BoreasLidarPoseIndex.from_csv(
            map_pose_path,
            sequence_id=map_sequence,
            expected_sha256=str(primary["map_lidar_pose_sha256"]),
        )
        query_pose = BoreasLidarPoseIndex.from_csv(
            query_pose_path,
            sequence_id=query_sequence,
            expected_sha256=str(primary["query_lidar_pose_sha256"]),
        )
        # Both sequences publish the same frozen static calibration identity.
        load_t_applanix_lidar(
            payload_root / query_sequence / "calib/T_applanix_lidar.txt",
            expected_sha256=str(primary["static_t_applanix_lidar_sha256"]),
        )

        preserve = prepare_authenticated_pending_transfer(
            runtime_root=runtime_root,
            temporary_download_root=temporary_root / "tmp_download",
            inventory=inventory,
        )
        downloader = StrictAllowlistDownloader(
            inventory,
            aws_executable=aws,
            bucket=authorization.bucket,
            temporary_root=temporary_root / "tmp_download",
            disk_gate=disk_gate,
            authorization=authorization,
            preserve_existing_temporary_payload=preserve,
        )
        witness_binding = contract.get("implementation_bindings", {}).get(
            "independent_canonical_source_witness", {}
        )
        config = QueryRunnerConfig(
            runtime_root=runtime_root,
            preprocessing_contract_sha256=sha256_file(contract_path),
            extrinsic_sha256=str(primary["static_t_applanix_lidar_sha256"]),
            backend_parameter_contract_sha256=sha256_file(backend_path),
            canonical_witness_implementation_sha256=sha256_file(witness_path),
            expected_target_freeze_sha256=sha256_file(
                runtime_root / "evidence/target_map_freeze_manifest.json"
            ),
            query_reference_pose_sha256=query_pose.source_sha256,
            map_reference_pose_sha256=map_pose.source_sha256,
            stage1_manifest_sha256=sha256_file(stage1_manifest),
            storage_manifest_sha256=sha256_file(storage_manifest),
            stage1_allowlist_sha256=inventory.allowlist_sha256,
            pair_selection_sha256=sha256_file(pair_path),
            target_context_resource_plan_sha256=sha256_file(resource_plan_path),
            target_context_measurement_evidence_sha256=sha256_file(measurement_path),
            target_context_measurement_provenance_sha256=sha256_file(
                provenance_path
            ),
            production_mode=True,
        )
        if witness_binding.get("file_sha256") != config.canonical_witness_implementation_sha256:
            raise BoreasV2Stage2QueryCLIError("canonical witness implementation differs")
        runner = BoreasV2Stage2QueryRunner(
            config=config,
            dependencies=QueryRunnerDependencies(),
            authorization=authorization,
            no_registration_guard=guard,
            disk_gate=disk_gate,
            downloader=downloader,
            inventory=inventory,
            query_pose_index=query_pose,
            map_pose_index=map_pose,
            frozen_windows=frozen_windows_from_pair(pair_path),
            runtime_lease=runtime_lease,
        )
        commit_target_context_capacity_evidence(runtime_root)

        def prerequisites() -> dict[str, str]:
            return write_selection_prerequisites(
                evidence_root=runtime_root / "evidence",
                preprocessing_contract_path=contract_path,
                authorization=authorization,
                no_registration_guard=guard,
                runtime_root=runtime_root,
            )

        result = execute_query_phases(
            runner, phase=arguments.phase, prerequisite_writer=prerequisites
        )
        result["target_context_capacity_measurement_sha256"] = sha256_file(
            measurement_path
        )
        result["target_context_resource_plan_sha256"] = sha256_file(
            resource_plan_path
        )
        result["target_context_measurement_provenance_sha256"] = sha256_file(
            provenance_path
        )
        print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


def main() -> int:
    arguments = _parser().parse_args()
    if arguments._measure_target_context:
        return _measure_target_context_child(arguments)
    return _production_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
