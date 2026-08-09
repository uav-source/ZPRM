"""Deterministic planning, materialization, and verification for Phase B snapshots.

This module deliberately has no backend dependency.  It is the only Phase B
component allowed to materialize the frozen 42-snapshot cache, and it refuses
to add to or replace a non-empty cache.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    OPEN3D_PLAN_BACKEND,
    PCL_PLAN_BACKEND,
    SOURCE_REPOSITORY,
    canonical_json_sha256,
    file_sha256,
    write_json,
)
from .phase_b_generator import (
    CONDITIONS,
    GENERATOR_SHA256,
    GEOMETRY_SEEDS,
    MEASUREMENT_SEED,
    SCENES,
    SNAPSHOT_BUILDER_SHA256,
    build_phase_b_snapshot,
    reproduce_all_phase_a_anchors,
    verify_generator_export,
    write_phase_b_snapshot_atomic,
)


PHASE_B_CACHE_RELATIVE = Path("data/phase_b_signal_snapshots")
PHASE_B_PLANNED_SNAPSHOTS_RELATIVE = Path(
    "frozen_assets/phase_b_planned_snapshots.csv"
)
PHASE_B_PLANNED_TRIALS_RELATIVE = Path("frozen_assets/phase_b_planned_trials.csv")
PHASE_B_SNAPSHOT_LOCK_RELATIVE = Path("frozen_assets/phase_b_snapshot_lock.json")
PHASE_B_PREPARATION_REPORT_RELATIVE = Path(
    "artifacts/phase_b_snapshot_preparation_report.json"
)

SNAPSHOT_FILES = frozenset(
    {"metadata.json", "source_points.npy", "target_points.npy", "reference_pose.npy"}
)
METADATA_FIELDS = frozenset(
    {
        "condition",
        "dropout_parameters",
        "generator_sha256",
        "generator_firewall_audit",
        "geometry_seed",
        "geometry_seed_index",
        "independent_sampling",
        "initial_pose",
        "measurement_seed",
        "metadata_payload_sha256",
        "noise_parameters",
        "reference_pose_checksum",
        "repeat_index",
        "scene_variant",
        "snapshot_checksum",
        "snapshot_builder_sha256",
        "snapshot_id",
        "source_checksum",
        "source_is_target_subset",
        "source_point_count",
        "target_checksum",
        "target_point_count",
        "array_file_sha256",
    }
)
SNAPSHOT_COLUMNS = (
    "snapshot_id",
    "scene_variant",
    "geometry_seed_index",
    "geometry_seed_value",
    "measurement_seed_index",
    "measurement_seed_value",
    "repeat_index",
    "condition",
)
TRIAL_COLUMNS = SNAPSHOT_COLUMNS + ("backend", "planned_trial_id")

_CONDITION_PARAMETERS = {
    "INDEPENDENT_NOISE_FREE": {
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.0,
        "scan_dropout_fraction": 0.0,
        "scan_noise_sigma_m": 0.0,
    },
    "FULL_NOISE": {
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.001,
        "scan_dropout_fraction": 0.01,
        "scan_noise_sigma_m": 0.003,
    },
}


class PreparationSourceAccessMonitor:
    """Fail closed if preparation reaches back into the source repository."""

    def __init__(self) -> None:
        self.count = 0
        self.paths: list[str] = []
        self._lock = threading.Lock()

    def install(self) -> None:
        source = SOURCE_REPOSITORY.resolve()

        def audit(event: str, args: tuple[Any, ...]) -> None:
            if event != "open" or not args or not isinstance(args[0], (str, bytes)):
                return
            try:
                candidate = Path(args[0]).resolve()
            except (OSError, TypeError):
                return
            if candidate == source or source in candidate.parents:
                with self._lock:
                    self.count += 1
                    self.paths.append(str(candidate))
                raise PermissionError(
                    f"source repository runtime read forbidden: {candidate}"
                )

        sys.addaudithook(audit)


def _raw_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("raw checksum input must be C-contiguous")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def phase_b_snapshot_id(scene: str, geometry_seed_index: int, condition: str) -> str:
    if scene not in SCENES or condition not in CONDITIONS:
        raise ValueError("snapshot identity is outside the frozen Phase B protocol")
    if geometry_seed_index not in range(len(GEOMETRY_SEEDS)):
        raise ValueError("geometry seed index is outside the frozen Phase B protocol")
    return f"phase-b-signal-v1/{scene}/g{geometry_seed_index}/{condition}"


def planned_phase_b_snapshots() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scene in SCENES:
        for geometry_index, geometry_seed in enumerate(GEOMETRY_SEEDS):
            for condition in CONDITIONS:
                rows.append(
                    {
                        "snapshot_id": phase_b_snapshot_id(
                            scene, geometry_index, condition
                        ),
                        "scene_variant": scene,
                        "geometry_seed_index": str(geometry_index),
                        "geometry_seed_value": str(geometry_seed),
                        "measurement_seed_index": "0",
                        "measurement_seed_value": str(MEASUREMENT_SEED),
                        "repeat_index": "0",
                        "condition": condition,
                    }
                )
    validate_phase_b_plans(rows, planned_phase_b_trials(rows))
    return rows


def planned_phase_b_trials(
    snapshots: Sequence[Mapping[str, str]] | None = None,
) -> list[dict[str, str]]:
    snapshot_rows = list(snapshots) if snapshots is not None else planned_phase_b_snapshots()
    rows: list[dict[str, str]] = []
    for snapshot in snapshot_rows:
        for backend in (OPEN3D_PLAN_BACKEND, PCL_PLAN_BACKEND):
            row = {name: str(snapshot[name]) for name in SNAPSHOT_COLUMNS}
            row["backend"] = backend
            row["planned_trial_id"] = f"{snapshot['snapshot_id']}/{backend}"
            rows.append(row)
    return rows


def validate_phase_b_plans(
    snapshots: Sequence[Mapping[str, str]], trials: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    expected_snapshots: list[dict[str, str]] = []
    for scene in SCENES:
        for geometry_index, geometry_seed in enumerate(GEOMETRY_SEEDS):
            for condition in CONDITIONS:
                expected_snapshots.append(
                    {
                        "snapshot_id": phase_b_snapshot_id(scene, geometry_index, condition),
                        "scene_variant": scene,
                        "geometry_seed_index": str(geometry_index),
                        "geometry_seed_value": str(geometry_seed),
                        "measurement_seed_index": "0",
                        "measurement_seed_value": str(MEASUREMENT_SEED),
                        "repeat_index": "0",
                        "condition": condition,
                    }
                )
    normalized_snapshots = [
        {name: str(row[name]) for name in SNAPSHOT_COLUMNS} for row in snapshots
    ]
    if normalized_snapshots != expected_snapshots:
        raise ValueError("Phase B planned snapshot rows/order changed")
    if len({row["snapshot_id"] for row in normalized_snapshots}) != 42:
        raise ValueError("Phase B snapshot identities are not exactly 42 unique values")

    expected_trials: list[dict[str, str]] = []
    for snapshot in expected_snapshots:
        for backend in (OPEN3D_PLAN_BACKEND, PCL_PLAN_BACKEND):
            expected_trials.append(
                {
                    **snapshot,
                    "backend": backend,
                    "planned_trial_id": f"{snapshot['snapshot_id']}/{backend}",
                }
            )
    normalized_trials = [
        {name: str(row[name]) for name in TRIAL_COLUMNS} for row in trials
    ]
    if normalized_trials != expected_trials:
        raise ValueError("Phase B planned trial rows/order changed")
    if len({row["planned_trial_id"] for row in normalized_trials}) != 84:
        raise ValueError("Phase B trial identities are not exactly 84 unique values")

    snapshot_condition_counts = Counter(row["condition"] for row in normalized_snapshots)
    trial_condition_counts = Counter(row["condition"] for row in normalized_trials)
    scene_counts = Counter(row["scene_variant"] for row in normalized_snapshots)
    backend_counts = Counter(row["backend"] for row in normalized_trials)
    pairing_counts = Counter(row["snapshot_id"] for row in normalized_trials)
    if snapshot_condition_counts != Counter({name: 21 for name in CONDITIONS}):
        raise ValueError("Phase B snapshot condition counts changed")
    if trial_condition_counts != Counter({name: 42 for name in CONDITIONS}):
        raise ValueError("Phase B trial condition counts changed")
    if scene_counts != Counter({scene: 6 for scene in SCENES}):
        raise ValueError("each Phase B scene must contain six snapshots")
    if backend_counts != Counter(
        {OPEN3D_PLAN_BACKEND: 42, PCL_PLAN_BACKEND: 42}
    ):
        raise ValueError("Phase B backend counts changed")
    if set(pairing_counts.values()) != {2}:
        raise ValueError("each Phase B snapshot must pair to exactly two backends")
    return {
        "backend_counts": dict(sorted(backend_counts.items())),
        "condition_snapshot_counts": dict(sorted(snapshot_condition_counts.items())),
        "condition_trial_counts": dict(sorted(trial_condition_counts.items())),
        "native_trial_count": 0,
        "planned_snapshot_count": len(normalized_snapshots),
        "planned_trial_count": len(normalized_trials),
        "scene_snapshot_counts": dict(sorted(scene_counts.items())),
        "snapshot_backend_pairing_mismatch_count": 0,
    }


def _read_csv(path: str | Path, columns: Sequence[str]) -> list[dict[str, str]]:
    candidate = Path(path)
    with candidate.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"CSV field order mismatch: {candidate}")
        return list(reader)


def read_phase_b_plans(
    snapshots_path: str | Path, trials_path: str | Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    snapshots = _read_csv(snapshots_path, SNAPSHOT_COLUMNS)
    trials = _read_csv(trials_path, TRIAL_COLUMNS)
    validate_phase_b_plans(snapshots, trials)
    return snapshots, trials


def _csv_bytes(columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> bytes:
    # csv.writer requires a text stream; StringIO keeps newline handling explicit.
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: str(row[name]) for name in columns})
    return stream.getvalue().encode("utf-8")


def _write_once_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"existing frozen asset differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_phase_b_plans(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    snapshots = planned_phase_b_snapshots()
    trials = planned_phase_b_trials(snapshots)
    report = validate_phase_b_plans(snapshots, trials)
    snapshots_path = repository / PHASE_B_PLANNED_SNAPSHOTS_RELATIVE
    trials_path = repository / PHASE_B_PLANNED_TRIALS_RELATIVE
    _write_once_or_verify(snapshots_path, _csv_bytes(SNAPSHOT_COLUMNS, snapshots))
    _write_once_or_verify(trials_path, _csv_bytes(TRIAL_COLUMNS, trials))
    return {
        **report,
        "planned_snapshots_path": PHASE_B_PLANNED_SNAPSHOTS_RELATIVE.as_posix(),
        "planned_snapshots_sha256": file_sha256(snapshots_path),
        "planned_trials_path": PHASE_B_PLANNED_TRIALS_RELATIVE.as_posix(),
        "planned_trials_sha256": file_sha256(trials_path),
    }


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Phase B snapshot metadata: {path}") from error
    if type(value) is not dict:
        raise ValueError("Phase B snapshot metadata must be an object")
    if set(value) != METADATA_FIELDS:
        raise ValueError("Phase B snapshot metadata fields changed")
    payload = dict(value)
    stored = payload.pop("metadata_payload_sha256", None)
    if stored != canonical_json_sha256(payload):
        raise ValueError("Phase B snapshot metadata checksum mismatch")
    return value


def read_phase_b_snapshot(
    cache_root: str | Path,
    snapshot_id: str,
    *,
    expected_lock_entry: Mapping[str, Any] | None = None,
    arrays: bool = True,
) -> dict[str, Any]:
    root = Path(cache_root).resolve()
    directory = (root / snapshot_id).resolve()
    if root not in directory.parents:
        raise ValueError("Phase B snapshot path escaped cache root")
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != SNAPSHOT_FILES:
        raise ValueError(f"Phase B snapshot file inventory mismatch: {snapshot_id}")
    metadata_path = directory / "metadata.json"
    metadata = _load_metadata(metadata_path)
    if metadata.get("snapshot_id") != snapshot_id:
        raise ValueError("Phase B snapshot metadata identity mismatch")
    scene = str(metadata.get("scene_variant"))
    condition = str(metadata.get("condition"))
    geometry_index = metadata.get("geometry_seed_index")
    if (
        scene not in SCENES
        or condition not in CONDITIONS
        or type(geometry_index) is not int
        or phase_b_snapshot_id(scene, geometry_index, condition) != snapshot_id
        or metadata.get("geometry_seed") != GEOMETRY_SEEDS[geometry_index]
        or metadata.get("measurement_seed") != MEASUREMENT_SEED
        or metadata.get("repeat_index") != 0
    ):
        raise ValueError("Phase B snapshot protocol identity mismatch")
    if (
        metadata.get("generator_sha256") != GENERATOR_SHA256
        or metadata.get("snapshot_builder_sha256") != SNAPSHOT_BUILDER_SHA256
        or metadata.get("independent_sampling") is not True
        or metadata.get("initial_pose") != "reference_pose_exact"
        or metadata.get("source_is_target_subset") is not False
    ):
        raise ValueError("Phase B snapshot generator/sampling contract mismatch")
    firewall = metadata.get("generator_firewall_audit")
    expected_firewall_fields = {
        "CONFIRMATORY_SEED_INSTANTIATION_COUNT",
        "GT_OPTIMIZATION_LEAKAGE_COUNT",
        "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT",
        "RNG_CONSTRUCTION_COUNT",
        "SNAPSHOT_ACCESS_COUNT",
    }
    if type(firewall) is not dict or set(firewall) != expected_firewall_fields:
        raise ValueError("Phase B snapshot generator firewall audit changed")
    if any(type(value) is not int or value < 0 for value in firewall.values()):
        raise ValueError("Phase B snapshot generator firewall count is invalid")
    if any(
        firewall[name] != 0
        for name in (
            "CONFIRMATORY_SEED_INSTANTIATION_COUNT",
            "GT_OPTIMIZATION_LEAKAGE_COUNT",
            "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT",
        )
    ):
        raise PermissionError("Phase B snapshot crossed a frozen seed firewall")
    expected_parameters = _CONDITION_PARAMETERS[condition]
    actual_parameters = {
        **dict(metadata.get("noise_parameters", {})),
        **dict(metadata.get("dropout_parameters", {})),
    }
    if actual_parameters != expected_parameters:
        raise ValueError("Phase B snapshot noise/dropout parameters changed")

    file_digests = {
        name: file_sha256(directory / name) for name in sorted(SNAPSHOT_FILES)
    }
    if metadata.get("array_file_sha256") != {
        name: file_digests[name]
        for name in ("reference_pose.npy", "source_points.npy", "target_points.npy")
    }:
        raise ValueError("Phase B snapshot metadata array-file SHA mismatch")
    if expected_lock_entry is not None:
        if expected_lock_entry.get("snapshot_id") != snapshot_id:
            raise ValueError("Phase B snapshot lock identity mismatch")
        if expected_lock_entry.get("file_sha256") != file_digests:
            raise ValueError("Phase B snapshot file SHA mismatch")

    result: dict[str, Any] = {
        "directory": directory,
        "file_sha256": file_digests,
        "metadata": metadata,
        "metadata_file_sha256": file_digests["metadata.json"],
    }
    if not arrays:
        return result
    source = np.load(directory / "source_points.npy", allow_pickle=False)
    target = np.load(directory / "target_points.npy", allow_pickle=False)
    reference = np.load(directory / "reference_pose.npy", allow_pickle=False)
    if source.dtype != np.dtype("<f4") or target.dtype != np.dtype("<f4"):
        raise ValueError("Phase B point arrays must be little-endian float32")
    if reference.dtype != np.dtype("<f8"):
        raise ValueError("Phase B reference must be little-endian float64")
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or target.ndim != 2
        or target.shape[1] != 3
        or reference.shape != (4, 4)
    ):
        raise ValueError("Phase B snapshot array shape mismatch")
    if not all(item.flags.c_contiguous for item in (source, target, reference)):
        raise ValueError("Phase B snapshot arrays must be C-contiguous")
    if not all(np.all(np.isfinite(item)) for item in (source, target, reference)):
        raise ValueError("Phase B snapshot contains non-finite values")
    if metadata.get("source_point_count") != source.shape[0] or metadata.get(
        "target_point_count"
    ) != target.shape[0]:
        raise ValueError("Phase B snapshot point count mismatch")
    raw_digests = {
        "source_checksum": _raw_sha256(source),
        "target_checksum": _raw_sha256(target),
        "reference_pose_checksum": _raw_sha256(reference),
    }
    if any(metadata.get(name) != digest for name, digest in raw_digests.items()):
        raise ValueError("Phase B snapshot raw array checksum mismatch")
    expected_snapshot_checksum = canonical_json_sha256(
        {"snapshot_id": snapshot_id, **raw_digests}
    )
    if metadata.get("snapshot_checksum") != expected_snapshot_checksum:
        raise ValueError("Phase B aggregate snapshot checksum mismatch")
    source_world = np.ascontiguousarray(
        source.astype(np.float64) @ reference[:3, :3].T + reference[:3, 3],
        dtype="<f4",
    )
    target_rows = {row.tobytes() for row in target}
    if all(row.tobytes() in target_rows for row in source_world):
        raise ValueError("Phase B source is an exact target subset")
    if expected_lock_entry is not None:
        exact = {
            **raw_digests,
            "snapshot_checksum": expected_snapshot_checksum,
            "metadata_payload_sha256": metadata["metadata_payload_sha256"],
        }
        if any(expected_lock_entry.get(name) != value for name, value in exact.items()):
            raise ValueError("Phase B snapshot lock checksum payload mismatch")
    result.update(
        source=source,
        target=target,
        reference=reference,
        **raw_digests,
        snapshot_checksum=expected_snapshot_checksum,
    )
    return result


def _lock_entry(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    metadata = snapshot["metadata"]
    return {
        "condition": metadata["condition"],
        "file_sha256": dict(snapshot["file_sha256"]),
        "geometry_seed": metadata["geometry_seed"],
        "geometry_seed_index": metadata["geometry_seed_index"],
        "measurement_seed": metadata["measurement_seed"],
        "metadata_payload_sha256": metadata["metadata_payload_sha256"],
        "reference_pose_checksum": snapshot["reference_pose_checksum"],
        "repeat_index": metadata["repeat_index"],
        "scene_variant": metadata["scene_variant"],
        "snapshot_checksum": snapshot["snapshot_checksum"],
        "snapshot_id": metadata["snapshot_id"],
        "source_checksum": snapshot["source_checksum"],
        "target_checksum": snapshot["target_checksum"],
    }


def build_phase_b_snapshot_lock(
    cache_root: str | Path, snapshots: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    entries = [
        _lock_entry(read_phase_b_snapshot(cache_root, row["snapshot_id"], arrays=True))
        for row in snapshots
    ]
    payload: dict[str, Any] = {
        "condition_snapshot_counts": {name: 21 for name in sorted(CONDITIONS)},
        "generator_sha256": GENERATOR_SHA256,
        "planned_snapshot_count": 42,
        "schema_version": "phase_b_signal_snapshot_lock_v1",
        "snapshot_builder_sha256": SNAPSHOT_BUILDER_SHA256,
        "snapshots": entries,
    }
    payload["snapshot_lock_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_phase_b_snapshot_lock(
    path: str | Path,
    cache_root: str | Path,
    snapshots: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    candidate = Path(path)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Phase B snapshot lock") from error
    if type(value) is not dict:
        raise ValueError("Phase B snapshot lock must be an object")
    stored = value.get("snapshot_lock_payload_sha256")
    payload = {name: item for name, item in value.items() if name != "snapshot_lock_payload_sha256"}
    if stored != canonical_json_sha256(payload):
        raise ValueError("Phase B snapshot lock payload SHA mismatch")
    if (
        value.get("schema_version") != "phase_b_signal_snapshot_lock_v1"
        or value.get("planned_snapshot_count") != 42
        or value.get("generator_sha256") != GENERATOR_SHA256
        or value.get("snapshot_builder_sha256") != SNAPSHOT_BUILDER_SHA256
        or type(value.get("snapshots")) is not list
        or len(value["snapshots"]) != 42
    ):
        raise ValueError("Phase B snapshot lock identity mismatch")
    entries = value["snapshots"]
    if [entry.get("snapshot_id") for entry in entries] != [
        row["snapshot_id"] for row in snapshots
    ]:
        raise ValueError("Phase B snapshot lock inventory/order mismatch")
    for row, entry in zip(snapshots, entries):
        item = read_phase_b_snapshot(
            cache_root, row["snapshot_id"], expected_lock_entry=entry, arrays=True
        )
        metadata = item["metadata"]
        exact = {
            "condition": row["condition"],
            "geometry_seed": int(row["geometry_seed_value"]),
            "geometry_seed_index": int(row["geometry_seed_index"]),
            "measurement_seed": int(row["measurement_seed_value"]),
            "repeat_index": int(row["repeat_index"]),
            "scene_variant": row["scene_variant"],
        }
        if any(metadata.get(name) != expected for name, expected in exact.items()):
            raise ValueError("Phase B snapshot plan/metadata mismatch")
    return value


def _cache_leaf_ids(cache_root: Path) -> list[str]:
    if not cache_root.exists():
        return []
    if not cache_root.is_dir():
        raise ValueError("Phase B cache root is not a directory")
    leaf_ids = []
    for path in cache_root.rglob("*"):
        if path.is_dir() and any(child.is_file() for child in path.iterdir()):
            leaf_ids.append(path.relative_to(cache_root).as_posix())
    return sorted(leaf_ids)


def _expected_cache_inventory(snapshot_ids: Sequence[str]) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for snapshot_id in snapshot_ids:
        snapshot_path = Path(snapshot_id)
        for parent in (snapshot_path, *snapshot_path.parents):
            if parent != Path("."):
                directories.add(parent.as_posix())
        for name in SNAPSHOT_FILES:
            files.add((snapshot_path / name).as_posix())
    return files, directories


def _assert_exact_cache_inventory(cache_root: Path, snapshot_ids: Sequence[str]) -> None:
    expected_files, expected_directories = _expected_cache_inventory(snapshot_ids)
    actual_files = {
        path.relative_to(cache_root).as_posix()
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    actual_directories = {
        path.relative_to(cache_root).as_posix()
        for path in cache_root.rglob("*")
        if path.is_dir()
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise FileExistsError(
            "non-empty Phase B cache is incomplete or contains extra paths"
        )


def materialize_or_verify_phase_b_snapshots(
    root: str | Path, snapshots: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    repository = Path(root).resolve()
    cache_root = repository / PHASE_B_CACHE_RELATIVE
    expected_ids = [row["snapshot_id"] for row in snapshots]
    generated = False
    firewall_totals = {
        "CONFIRMATORY_SEED_INSTANTIATION_COUNT": 0,
        "GT_OPTIMIZATION_LEAKAGE_COUNT": 0,
        "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT": 0,
        "RNG_CONSTRUCTION_COUNT": 0,
        "SNAPSHOT_ACCESS_COUNT": 0,
    }
    if cache_root.exists() and any(cache_root.iterdir()):
        _assert_exact_cache_inventory(cache_root, expected_ids)
    else:
        cache_root.mkdir(parents=True, exist_ok=True)
        generated = True
        for row in snapshots:
            value = build_phase_b_snapshot(
                repository,
                scene=row["scene_variant"],
                geometry_seed=int(row["geometry_seed_value"]),
                condition=row["condition"],
            )
            audit = value.get("firewall_audit")
            if type(audit) is not dict or set(audit) != set(firewall_totals):
                raise RuntimeError("Phase B generator firewall audit schema changed")
            for name in firewall_totals:
                firewall_totals[name] += int(audit[name])
            if any(
                int(audit[name]) != 0
                for name in (
                    "CONFIRMATORY_SEED_INSTANTIATION_COUNT",
                    "GT_OPTIMIZATION_LEAKAGE_COUNT",
                    "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT",
                )
            ):
                raise PermissionError("Phase B generator crossed a frozen seed firewall")
            write_phase_b_snapshot_atomic(cache_root, value)
    verified = [
        read_phase_b_snapshot(cache_root, snapshot_id, arrays=True)
        for snapshot_id in expected_ids
    ]
    _assert_exact_cache_inventory(cache_root, expected_ids)
    return verified, generated, firewall_totals


def prepare_phase_b_signal_assets(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    monitor = PreparationSourceAccessMonitor()
    monitor.install()
    export_verification = verify_generator_export(repository)
    if (
        export_verification.get("GENERATOR_EXPORT_EQUIVALENCE_PASS") is not True
        or export_verification.get("source_destination_sha_mismatch_count") != 0
        or export_verification.get("destination_file_sha_mismatch_count") != 0
    ):
        raise RuntimeError("frozen Phase B generator export equivalence gate failed")
    plan_report = write_phase_b_plans(repository)
    snapshots, trials = read_phase_b_plans(
        repository / PHASE_B_PLANNED_SNAPSHOTS_RELATIVE,
        repository / PHASE_B_PLANNED_TRIALS_RELATIVE,
    )

    # The seven byte-exact Phase A anchors are a hard gate before any new scene
    # materialization.  The called wrapper reads only this standalone repository.
    reproduction = reproduce_all_phase_a_anchors(repository)
    if reproduction.get("GENERATOR_REPRODUCTION_PASS") is not True:
        raise RuntimeError("Phase A generator reproduction gate failed")
    reproduction_firewall = reproduction.get("firewall_totals")
    if type(reproduction_firewall) is not dict or any(
        int(reproduction_firewall.get(name, -1)) != 0
        for name in (
            "CONFIRMATORY_SEED_INSTANTIATION_COUNT",
            "GT_OPTIMIZATION_LEAKAGE_COUNT",
            "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT",
        )
    ):
        raise PermissionError("Phase A reproduction crossed a frozen seed firewall")

    verified, generated, generation_firewall = materialize_or_verify_phase_b_snapshots(
        repository, snapshots
    )
    lock_value = build_phase_b_snapshot_lock(
        repository / PHASE_B_CACHE_RELATIVE, snapshots
    )
    lock_path = repository / PHASE_B_SNAPSHOT_LOCK_RELATIVE
    lock_bytes = (
        json.dumps(
            lock_value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_once_or_verify(lock_path, lock_bytes)
    validate_phase_b_snapshot_lock(
        lock_path, repository / PHASE_B_CACHE_RELATIVE, snapshots
    )
    from .asset_verifier import source_runtime_import_paths

    source_imports = source_runtime_import_paths()
    confirmatory_total = int(
        reproduction_firewall["CONFIRMATORY_SEED_INSTANTIATION_COUNT"]
    ) + int(generation_firewall["CONFIRMATORY_SEED_INSTANTIATION_COUNT"])
    if monitor.count or source_imports or confirmatory_total:
        raise PermissionError("Phase B preparation runtime isolation failed")
    report = {
        "PHASE_A_GENERATOR_REPRODUCTION_PASS": True,
        "GENERATOR_EXPORT_EQUIVALENCE_PASS": True,
        "PHASE_B_SNAPSHOT_ASSET_PASS": True,
        "actual_snapshot_count": len(verified),
        "cache_was_generated": generated,
        "generator_reproduction_checksum_mismatch_count": reproduction[
            "GENERATOR_REPRODUCTION_CHECKSUM_MISMATCH_COUNT"
        ],
        "generator_export_destination_file_sha_mismatch_count": export_verification[
            "destination_file_sha_mismatch_count"
        ],
        "generator_export_source_destination_sha_mismatch_count": export_verification[
            "source_destination_sha_mismatch_count"
        ],
        "generator_exported_file_count": export_verification["exported_file_count"],
        "confirmatory_seed_instantiation_count": confirmatory_total,
        "phase_a_reproduction_firewall_totals": reproduction_firewall,
        "phase_b_generation_firewall_totals": generation_firewall,
        "metadata_checksum_mismatch_count": 0,
        "planned_snapshot_count": len(snapshots),
        "planned_trial_count": len(trials),
        "schema_version": "phase_b_signal_snapshot_preparation_v1",
        "snapshot_checksum_mismatch_count": 0,
        "snapshot_corrupt_count": 0,
        "snapshot_duplicate_count": 0,
        "snapshot_extra_count": 0,
        "snapshot_file_sha_mismatch_count": 0,
        "snapshot_missing_count": 0,
        "snapshot_lock_path": PHASE_B_SNAPSHOT_LOCK_RELATIVE.as_posix(),
        "snapshot_lock_sha256": file_sha256(lock_path),
        "source_repository_runtime_file_read_count": monitor.count,
        "source_repository_runtime_import_count": len(source_imports),
        "source_repository_runtime_import_paths": source_imports,
        "verified_snapshot_count": len(verified),
        **plan_report,
    }
    write_json(repository / PHASE_B_PREPARATION_REPORT_RELATIVE, report)
    return report


__all__ = [
    "PHASE_B_CACHE_RELATIVE",
    "PHASE_B_PLANNED_SNAPSHOTS_RELATIVE",
    "PHASE_B_PLANNED_TRIALS_RELATIVE",
    "PHASE_B_SNAPSHOT_LOCK_RELATIVE",
    "SNAPSHOT_COLUMNS",
    "TRIAL_COLUMNS",
    "build_phase_b_snapshot_lock",
    "materialize_or_verify_phase_b_snapshots",
    "phase_b_snapshot_id",
    "planned_phase_b_snapshots",
    "planned_phase_b_trials",
    "prepare_phase_b_signal_assets",
    "read_phase_b_plans",
    "read_phase_b_snapshot",
    "validate_phase_b_plans",
    "validate_phase_b_snapshot_lock",
    "write_phase_b_plans",
]
