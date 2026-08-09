"""Resumable, checksum-strict snapshot assets for Full Synthetic Development."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import threading
import uuid
import weakref
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .asset_verifier import source_runtime_import_paths
from .contracts import SOURCE_REPOSITORY, canonical_json_sha256, file_sha256, write_json
from .full_synthetic_development_protocol import (
    CONDITION_PARAMETERS,
    GEOMETRY_SEEDS,
    MEASUREMENT_SEEDS,
    NEW_CONDITIONS,
    PREPARATION_REPORT_RELATIVE,
    SCENES,
    SNAPSHOT_CACHE_RELATIVE,
    SNAPSHOT_LOCK_RELATIVE,
    assert_isolated_python_runtime,
    phase_b_overlap_snapshots,
    read_full_synthetic_plans,
    verify_full_synthetic_protocol_file,
    verify_phase_a_ideal_import,
)
from .phase_b_generator import (
    DEVELOPMENT_PROTOCOL_SHA256,
    GENERATOR_SHA256,
    SNAPSHOT_BUILDER_SHA256,
    DevelopmentSeedFirewall,
    SnapshotKey,
    ZeroPerturbationProtocol,
    _freeze,
    build_snapshot,
    reference_matrix,
    reproduce_phase_a_snapshot,
)
from .phase_b_snapshot_assets import read_phase_b_snapshot


SNAPSHOT_FILES = frozenset(
    {"metadata.json", "source_points.npy", "target_points.npy", "reference_pose.npy"}
)
METADATA_FIELDS = frozenset(
    {
        "array_file_sha256",
        "condition",
        "development_protocol_sha256",
        "dropout_parameters",
        "generator_firewall_audit",
        "generator_sha256",
        "geometry_seed",
        "geometry_seed_index",
        "independent_sampling",
        "initial_pose",
        "measurement_seed",
        "measurement_seed_index",
        "metadata_payload_sha256",
        "noise_parameters",
        "reference_pose_checksum",
        "repeat_index",
        "scene_variant",
        "snapshot_builder_sha256",
        "snapshot_checksum",
        "snapshot_id",
        "source_checksum",
        "source_is_target_subset",
        "source_point_count",
        "target_checksum",
        "target_point_count",
    }
)


class FullSyntheticSourceAccessMonitor:
    def __init__(self) -> None:
        self.count = 0
        self.paths: list[str] = []
        self._lock = threading.Lock()

    def install(self) -> None:
        source = SOURCE_REPOSITORY.resolve()
        monitor_reference = weakref.ref(self)

        def audit(event: str, args: tuple[Any, ...]) -> None:
            monitor = monitor_reference()
            # Python audit hooks cannot be removed.  Keep only a weak reference so
            # the hook becomes inert when the monitored operation returns; this
            # prevents a dry-run in a shared pytest process from contaminating
            # later, unrelated read-only source-repository audits.
            if monitor is None:
                return
            if event != "open" or not args or not isinstance(args[0], (str, bytes)):
                return
            try:
                candidate = Path(args[0]).resolve()
            except (OSError, TypeError):
                return
            if candidate == source or source in candidate.parents:
                with monitor._lock:
                    monitor.count += 1
                    monitor.paths.append(str(candidate))
                raise PermissionError(
                    f"source repository runtime read forbidden: {candidate}"
                )

        sys.addaudithook(audit)


def _raw_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("raw checksum input must be C-contiguous")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_full_synthetic_generator_protocol(
    root: str | Path,
) -> ZeroPerturbationProtocol:
    repository = Path(root).resolve()
    protocol_path = repository / "configs/zero_perturbation/development_v1.yaml"
    generator_path = (
        repository
        / "src/phase_a_harness/phase_b_generator_frozen/capture_range/day2_development_scene.py"
    )
    builder_path = (
        repository
        / "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/snapshot_builder.py"
    )
    if file_sha256(protocol_path) != DEVELOPMENT_PROTOCOL_SHA256:
        raise ValueError("frozen Development protocol SHA mismatch")
    if file_sha256(generator_path) != GENERATOR_SHA256:
        raise ValueError("frozen scene generator SHA mismatch")
    if file_sha256(builder_path) != SNAPSHOT_BUILDER_SHA256:
        raise ValueError("frozen snapshot builder SHA mismatch")
    raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if type(raw) is not dict:
        raise ValueError("Development protocol root must be a mapping")
    if tuple(raw["scene_generation"]["variants_in_order"]) != SCENES:
        raise ValueError("Development scene geometry contract changed")
    declared = {row["name"]: row for row in raw["noise_conditions"]}
    for name in NEW_CONDITIONS:
        expected = CONDITION_PARAMETERS[name]
        actual = {
            key: declared[name][key]
            for key in (
                "independent_sampling",
                "map_dropout_fraction",
                "map_noise_sigma_m",
                "scan_dropout_fraction",
                "scan_noise_sigma_m",
            )
        }
        if actual != expected:
            raise ValueError(f"Development condition changed: {name}")
    seeds = MappingProxyType(
        {
            "geometry": MappingProxyType(
                {f"geometry_{index}": seed for index, seed in enumerate(GEOMETRY_SEEDS)}
            ),
            "measurement": MappingProxyType(
                {
                    f"measurement_{index}": seed
                    for index, seed in enumerate(MEASUREMENT_SEEDS)
                }
            ),
        }
    )
    return ZeroPerturbationProtocol(
        root=repository,
        data=_freeze(raw),
        source_sha256=DEVELOPMENT_PROTOCOL_SHA256,
        development_seeds=seeds,
    )


def build_full_synthetic_snapshot(
    root: str | Path,
    plan: Mapping[str, str],
    *,
    protocol: ZeroPerturbationProtocol | None = None,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    development = protocol or load_full_synthetic_generator_protocol(repository)
    scene = str(plan["scene_variant"])
    condition = str(plan["condition"])
    geometry_index = int(plan["geometry_seed_index"])
    measurement_index = int(plan["measurement_seed_index"])
    repeat_index = int(plan["repeat_index"])
    geometry_seed = int(plan["geometry_seed_value"])
    measurement_seed = int(plan["measurement_seed_value"])
    if (
        scene not in SCENES
        or condition not in NEW_CONDITIONS
        or geometry_seed != GEOMETRY_SEEDS[geometry_index]
        or measurement_seed != MEASUREMENT_SEEDS[measurement_index]
        or repeat_index not in range(5)
    ):
        raise ValueError("snapshot plan is outside Full Synthetic Development v1")
    firewall = DevelopmentSeedFirewall(development)
    bundle = build_snapshot(
        development,
        firewall,
        SnapshotKey(
            scene_variant=scene,
            geometry_seed=geometry_seed,
            measurement_seed=measurement_seed,
            repeat_index=repeat_index,
            noise_condition=condition,
        ),
    )
    source = np.ascontiguousarray(bundle.scan_points, dtype="<f4")
    target = np.ascontiguousarray(bundle.map_points, dtype="<f4")
    reference = np.ascontiguousarray(reference_matrix(bundle.reference_pose), dtype="<f8")
    source_world = np.ascontiguousarray(
        source.astype(np.float64) @ reference[:3, :3].T + reference[:3, 3],
        dtype="<f4",
    )
    target_rows = {row.tobytes() for row in target}
    if all(row.tobytes() in target_rows for row in source_world):
        raise ValueError("independently sampled source became an exact target subset")
    checksums = {
        "source_checksum": _raw_sha256(source),
        "target_checksum": _raw_sha256(target),
        "reference_pose_checksum": _raw_sha256(reference),
    }
    snapshot_id = str(plan["snapshot_id"])
    metadata: dict[str, Any] = {
        "condition": condition,
        "development_protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
        "dropout_parameters": {
            "map_dropout_fraction": float(
                CONDITION_PARAMETERS[condition]["map_dropout_fraction"]
            ),
            "scan_dropout_fraction": float(
                CONDITION_PARAMETERS[condition]["scan_dropout_fraction"]
            ),
        },
        "generator_firewall_audit": firewall.report(),
        "generator_sha256": GENERATOR_SHA256,
        "geometry_seed": geometry_seed,
        "geometry_seed_index": geometry_index,
        "independent_sampling": True,
        "initial_pose": "reference_pose_exact",
        "measurement_seed": measurement_seed,
        "measurement_seed_index": measurement_index,
        "noise_parameters": {
            "map_noise_sigma_m": float(
                CONDITION_PARAMETERS[condition]["map_noise_sigma_m"]
            ),
            "scan_noise_sigma_m": float(
                CONDITION_PARAMETERS[condition]["scan_noise_sigma_m"]
            ),
        },
        "reference_pose_checksum": checksums["reference_pose_checksum"],
        "repeat_index": repeat_index,
        "scene_variant": scene,
        "snapshot_builder_sha256": SNAPSHOT_BUILDER_SHA256,
        "snapshot_checksum": canonical_json_sha256(
            {"snapshot_id": snapshot_id, **checksums}
        ),
        "snapshot_id": snapshot_id,
        "source_checksum": checksums["source_checksum"],
        "source_is_target_subset": False,
        "source_point_count": int(source.shape[0]),
        "target_checksum": checksums["target_checksum"],
        "target_point_count": int(target.shape[0]),
    }
    return {
        "firewall_audit": firewall.report(),
        "metadata": metadata,
        "reference": reference,
        "source": source,
        "target": target,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_full_synthetic_snapshot_atomic(
    cache_root: str | Path, value: Mapping[str, Any]
) -> Path:
    root = Path(cache_root).resolve()
    metadata = dict(value["metadata"])
    destination = (root / metadata["snapshot_id"]).resolve()
    if root not in destination.parents:
        raise ValueError("snapshot path escaped Full Synthetic cache")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        arrays = (
            ("source_points.npy", value["source"]),
            ("target_points.npy", value["target"]),
            ("reference_pose.npy", value["reference"]),
        )
        for name, array in arrays:
            with (temporary / name).open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
        metadata["array_file_sha256"] = {
            name: file_sha256(temporary / name)
            for name in ("reference_pose.npy", "source_points.npy", "target_points.npy")
        }
        metadata["metadata_payload_sha256"] = canonical_json_sha256(metadata)
        with (temporary / "metadata.json").open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _load_metadata(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {token}")
        ),
    )
    if type(value) is not dict or set(value) != METADATA_FIELDS:
        raise ValueError("Full Synthetic snapshot metadata fields changed")
    payload = dict(value)
    stored = payload.pop("metadata_payload_sha256")
    if stored != canonical_json_sha256(payload):
        raise ValueError("Full Synthetic snapshot metadata payload SHA mismatch")
    return value


def read_full_synthetic_snapshot(
    cache_root: str | Path,
    plan: Mapping[str, str],
    *,
    expected_lock_entry: Mapping[str, Any] | None = None,
    arrays: bool = True,
) -> dict[str, Any]:
    root = Path(cache_root).resolve()
    snapshot_id = str(plan["snapshot_id"])
    directory = (root / snapshot_id).resolve()
    if root not in directory.parents:
        raise ValueError("snapshot path escaped Full Synthetic cache")
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != SNAPSHOT_FILES:
        raise ValueError(f"snapshot file inventory mismatch: {snapshot_id}")
    metadata = _load_metadata(directory / "metadata.json")
    exact_metadata: dict[str, Any] = {
        "condition": str(plan["condition"]),
        "geometry_seed": int(plan["geometry_seed_value"]),
        "geometry_seed_index": int(plan["geometry_seed_index"]),
        "measurement_seed": int(plan["measurement_seed_value"]),
        "measurement_seed_index": int(plan["measurement_seed_index"]),
        "repeat_index": int(plan["repeat_index"]),
        "scene_variant": str(plan["scene_variant"]),
        "snapshot_id": snapshot_id,
    }
    if any(metadata.get(name) != expected for name, expected in exact_metadata.items()):
        raise ValueError("snapshot plan/metadata identity mismatch")
    if (
        metadata["generator_sha256"] != GENERATOR_SHA256
        or metadata["snapshot_builder_sha256"] != SNAPSHOT_BUILDER_SHA256
        or metadata["development_protocol_sha256"] != DEVELOPMENT_PROTOCOL_SHA256
        or metadata["independent_sampling"] is not True
        or metadata["initial_pose"] != "reference_pose_exact"
        or metadata["source_is_target_subset"] is not False
    ):
        raise ValueError("snapshot generator contract changed")
    expected_parameters = CONDITION_PARAMETERS[str(plan["condition"])]
    actual_parameters = {
        **metadata["noise_parameters"],
        **metadata["dropout_parameters"],
        "independent_sampling": metadata["independent_sampling"],
    }
    if actual_parameters != expected_parameters:
        raise ValueError("snapshot noise/dropout parameters changed")
    firewall = metadata["generator_firewall_audit"]
    if type(firewall) is not dict or any(
        firewall.get(name) != 0
        for name in (
            "CONFIRMATORY_SEED_INSTANTIATION_COUNT",
            "GT_OPTIMIZATION_LEAKAGE_COUNT",
            "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT",
        )
    ):
        raise PermissionError("snapshot crossed the Development seed firewall")
    file_digests = {
        name: file_sha256(directory / name) for name in sorted(SNAPSHOT_FILES)
    }
    if metadata["array_file_sha256"] != {
        name: file_digests[name]
        for name in ("reference_pose.npy", "source_points.npy", "target_points.npy")
    }:
        raise ValueError("snapshot array file SHA mismatch")
    if expected_lock_entry is not None and (
        expected_lock_entry.get("snapshot_id") != snapshot_id
        or expected_lock_entry.get("file_sha256") != file_digests
    ):
        raise ValueError("snapshot lock file binding mismatch")
    result: dict[str, Any] = {
        "directory": directory,
        "file_sha256": file_digests,
        "metadata": metadata,
    }
    if not arrays:
        return result
    source = np.load(directory / "source_points.npy", allow_pickle=False)
    target = np.load(directory / "target_points.npy", allow_pickle=False)
    reference = np.load(directory / "reference_pose.npy", allow_pickle=False)
    if (
        source.dtype != np.dtype("<f4")
        or target.dtype != np.dtype("<f4")
        or reference.dtype != np.dtype("<f8")
        or source.ndim != 2
        or source.shape[1:] != (3,)
        or target.ndim != 2
        or target.shape[1:] != (3,)
        or reference.shape != (4, 4)
    ):
        raise ValueError("snapshot array dtype/shape mismatch")
    if not all(array.flags.c_contiguous for array in (source, target, reference)):
        raise ValueError("snapshot arrays are not C-contiguous")
    if not all(np.all(np.isfinite(array)) for array in (source, target, reference)):
        raise ValueError("snapshot contains non-finite values")
    if metadata["source_point_count"] != len(source) or metadata["target_point_count"] != len(target):
        raise ValueError("snapshot point count mismatch")
    raw = {
        "source_checksum": _raw_sha256(source),
        "target_checksum": _raw_sha256(target),
        "reference_pose_checksum": _raw_sha256(reference),
    }
    if any(metadata[name] != digest for name, digest in raw.items()):
        raise ValueError("snapshot raw checksum mismatch")
    snapshot_checksum = canonical_json_sha256({"snapshot_id": snapshot_id, **raw})
    if metadata["snapshot_checksum"] != snapshot_checksum:
        raise ValueError("snapshot aggregate checksum mismatch")
    source_world = np.ascontiguousarray(
        source.astype(np.float64) @ reference[:3, :3].T + reference[:3, 3],
        dtype="<f4",
    )
    target_rows = {row.tobytes() for row in target}
    if all(row.tobytes() in target_rows for row in source_world):
        raise ValueError("independently sampled source is an exact target subset")
    if expected_lock_entry is not None:
        exact_lock = {
            **raw,
            "metadata_payload_sha256": metadata["metadata_payload_sha256"],
            "snapshot_checksum": snapshot_checksum,
        }
        if any(expected_lock_entry.get(name) != value for name, value in exact_lock.items()):
            raise ValueError("snapshot lock payload checksum mismatch")
    result.update(
        source=source,
        target=target,
        reference=reference,
        snapshot_checksum=snapshot_checksum,
        **raw,
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
        "measurement_seed_index": metadata["measurement_seed_index"],
        "metadata_payload_sha256": metadata["metadata_payload_sha256"],
        "reference_pose_checksum": snapshot["reference_pose_checksum"],
        "repeat_index": metadata["repeat_index"],
        "scene_variant": metadata["scene_variant"],
        "snapshot_checksum": snapshot["snapshot_checksum"],
        "snapshot_id": metadata["snapshot_id"],
        "source_checksum": snapshot["source_checksum"],
        "target_checksum": snapshot["target_checksum"],
    }


def build_full_synthetic_snapshot_lock(
    cache_root: str | Path, plans: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    entries = [
        _lock_entry(read_full_synthetic_snapshot(cache_root, plan, arrays=True))
        for plan in plans
    ]
    payload: dict[str, Any] = {
        "condition_snapshot_counts": {name: 210 for name in sorted(NEW_CONDITIONS)},
        "development_protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
        "generator_sha256": GENERATOR_SHA256,
        "planned_snapshot_count": 1050,
        "schema_version": "full_synthetic_development_snapshot_lock_v1",
        "snapshot_builder_sha256": SNAPSHOT_BUILDER_SHA256,
        "snapshots": entries,
    }
    payload["snapshot_lock_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_full_synthetic_snapshot_lock(
    lock_path: str | Path,
    cache_root: str | Path,
    plans: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    value = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("snapshot lock must be an object")
    stored = value.get("snapshot_lock_payload_sha256")
    payload = {
        name: item for name, item in value.items() if name != "snapshot_lock_payload_sha256"
    }
    if stored != canonical_json_sha256(payload):
        raise ValueError("snapshot lock payload SHA mismatch")
    if (
        value.get("schema_version") != "full_synthetic_development_snapshot_lock_v1"
        or value.get("planned_snapshot_count") != 1050
        or value.get("generator_sha256") != GENERATOR_SHA256
        or value.get("snapshot_builder_sha256") != SNAPSHOT_BUILDER_SHA256
        or type(value.get("snapshots")) is not list
        or len(value["snapshots"]) != 1050
        or [row.get("snapshot_id") for row in value["snapshots"]]
        != [str(plan["snapshot_id"]) for plan in plans]
    ):
        raise ValueError("snapshot lock identity/inventory mismatch")
    for plan, entry in zip(plans, value["snapshots"]):
        read_full_synthetic_snapshot(
            cache_root, plan, expected_lock_entry=entry, arrays=True
        )
    return value


def verify_phase_b_snapshot_subset(
    root: str | Path,
    new_cache_root: str | Path,
    new_plans: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    repository = Path(root).resolve()
    rows = phase_b_overlap_snapshots(new_plans)
    mismatches = 0
    fields = (
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "snapshot_checksum",
    )
    for plan in rows:
        rebuilt = read_full_synthetic_snapshot(new_cache_root, plan, arrays=True)
        published = read_phase_b_snapshot(
            repository / "data/phase_b_signal_snapshots",
            str(plan["snapshot_id"]),
            arrays=True,
        )
        if any(rebuilt[name] != published[name] for name in fields):
            mismatches += 1
        elif not all(
            np.array_equal(rebuilt[name], published[name])
            for name in ("source", "target", "reference")
        ):
            mismatches += 1
    return {
        "PHASE_B_SNAPSHOT_SUBSET_REPRODUCTION_PASS": mismatches == 0 and len(rows) == 42,
        "phase_b_overlap_snapshot_count": len(rows),
        "phase_b_snapshot_checksum_mismatch_count": mismatches,
    }


def _verify_generator_export_read_only(root: Path) -> dict[str, Any]:
    with (root / "frozen_assets/phase_b_generator_export_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    recorded = 0
    destination = 0
    for row in rows:
        if row["copied_exactly"] != "true" or row["source_sha256"] != row["destination_sha256"]:
            recorded += 1
        local = root / row["destination_path"]
        if not local.is_file() or file_sha256(local) != row["destination_sha256"]:
            destination += 1
    return {
        "GENERATOR_EXPORT_EQUIVALENCE_PASS": bool(rows and recorded == destination == 0),
        "destination_file_sha_mismatch_count": destination,
        "exported_file_count": len(rows),
        "source_destination_sha_mismatch_count": recorded,
    }


def _existing_snapshot_ids(cache_root: Path) -> set[str]:
    if not cache_root.exists():
        return set()
    identifiers: set[str] = set()
    for metadata_path in cache_root.rglob("metadata.json"):
        identifiers.add(metadata_path.parent.relative_to(cache_root).as_posix())
    all_files = {path for path in cache_root.rglob("*") if path.is_file()}
    expected_files = {
        cache_root / identifier / name
        for identifier in identifiers
        for name in SNAPSHOT_FILES
    }
    if all_files != expected_files:
        raise ValueError("existing snapshot cache contains partial or extra files")
    return identifiers


def verify_existing_snapshot_rebuild_equivalence(
    root: str | Path,
    cache_root: str | Path,
    plan: Mapping[str, str],
    *,
    protocol: ZeroPerturbationProtocol,
) -> dict[str, Any]:
    """Rebuild a resumable snapshot and reject any scientific divergence."""

    actual = read_full_synthetic_snapshot(cache_root, plan, arrays=True)
    rebuilt = build_full_synthetic_snapshot(root, plan, protocol=protocol)
    array_mismatches = [
        name
        for name in ("source", "target", "reference")
        if not np.array_equal(actual[name], rebuilt[name])
    ]
    actual_metadata = {
        name: value
        for name, value in actual["metadata"].items()
        if name not in {"array_file_sha256", "metadata_payload_sha256"}
    }
    metadata_match = actual_metadata == rebuilt["metadata"]
    checksum_fields = (
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "snapshot_checksum",
    )
    checksum_match = all(
        actual[name] == rebuilt["metadata"][name] for name in checksum_fields
    )
    if array_mismatches or not metadata_match or not checksum_match:
        raise ValueError(
            "existing snapshot does not reproduce exactly; refusing to overwrite"
        )
    return {
        "array_mismatch_count": 0,
        "checksum_equivalence": True,
        "firewall_audit": rebuilt["firewall_audit"],
        "scientific_metadata_equivalence": True,
        "snapshot_id": str(plan["snapshot_id"]),
    }


def prepare_full_synthetic_snapshots(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    assert_isolated_python_runtime()
    frozen_protocol = verify_full_synthetic_protocol_file(repository)
    monitor = FullSyntheticSourceAccessMonitor()
    monitor.install()
    phase_a = verify_phase_a_ideal_import(repository, write_report=True)
    if phase_a["PHASE_A_IDEAL_IMPORT_PASS"] is not True:
        raise RuntimeError("Phase A IDEAL import gate failed")
    export = _verify_generator_export_read_only(repository)
    if export["GENERATOR_EXPORT_EQUIVALENCE_PASS"] is not True:
        raise RuntimeError("frozen generator export gate failed")
    reproduction = [
        reproduce_phase_a_snapshot(repository, scene=scene) for scene in SCENES
    ]
    reproduction_mismatches = sum(int(row["mismatch_count"]) for row in reproduction)
    reproduction_firewall_totals = {
        name: sum(int(row["firewall_audit"][name]) for row in reproduction)
        for name in reproduction[0]["firewall_audit"]
    }
    if any(
        reproduction_firewall_totals[name] != 0
        for name in (
            "CONFIRMATORY_SEED_INSTANTIATION_COUNT",
            "GT_OPTIMIZATION_LEAKAGE_COUNT",
            "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT",
        )
    ):
        raise PermissionError("Phase A reproduction crossed a frozen seed firewall")
    if reproduction_mismatches:
        raise RuntimeError("frozen generator regression failed")
    new_plans, new_trials, combined_plans, combined_trials = read_full_synthetic_plans(
        repository
    )
    cache_root = repository / SNAPSHOT_CACHE_RELATIVE
    cache_root.mkdir(parents=True, exist_ok=True)
    existing = _existing_snapshot_ids(cache_root)
    expected = {row["snapshot_id"] for row in new_plans}
    if not existing <= expected:
        raise ValueError("snapshot cache contains an extra snapshot")
    protocol = load_full_synthetic_generator_protocol(repository)
    generated_count = 0
    resumed_count = 0
    firewall_totals = {
        "CONFIRMATORY_SEED_INSTANTIATION_COUNT": 0,
        "GT_OPTIMIZATION_LEAKAGE_COUNT": 0,
        "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT": 0,
        "RNG_CONSTRUCTION_COUNT": 0,
        "SNAPSHOT_ACCESS_COUNT": 0,
    }
    for plan in new_plans:
        if plan["snapshot_id"] in existing:
            equivalence = verify_existing_snapshot_rebuild_equivalence(
                repository, cache_root, plan, protocol=protocol
            )
            audit = equivalence["firewall_audit"]
            if set(audit) != set(firewall_totals):
                raise RuntimeError("resume generator firewall schema changed")
            for name in firewall_totals:
                firewall_totals[name] += int(audit[name])
            if any(
                audit[name] != 0
                for name in (
                    "CONFIRMATORY_SEED_INSTANTIATION_COUNT",
                    "GT_OPTIMIZATION_LEAKAGE_COUNT",
                    "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT",
                )
            ):
                raise PermissionError("resume rebuild crossed a frozen seed firewall")
            resumed_count += 1
            continue
        value = build_full_synthetic_snapshot(repository, plan, protocol=protocol)
        audit = value["firewall_audit"]
        if set(audit) != set(firewall_totals):
            raise RuntimeError("generator firewall schema changed")
        for name in firewall_totals:
            firewall_totals[name] += int(audit[name])
        if any(
            audit[name] != 0
            for name in (
                "CONFIRMATORY_SEED_INSTANTIATION_COUNT",
                "GT_OPTIMIZATION_LEAKAGE_COUNT",
                "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT",
            )
        ):
            raise PermissionError("generator crossed a frozen seed firewall")
        write_full_synthetic_snapshot_atomic(cache_root, value)
        generated_count += 1
    verified = [
        read_full_synthetic_snapshot(cache_root, plan, arrays=True) for plan in new_plans
    ]
    if _existing_snapshot_ids(cache_root) != expected:
        raise ValueError("snapshot cache is incomplete after materialization")
    subset = verify_phase_b_snapshot_subset(repository, cache_root, new_plans)
    if subset["PHASE_B_SNAPSHOT_SUBSET_REPRODUCTION_PASS"] is not True:
        raise RuntimeError("Phase B snapshot overlap reproduction failed")
    lock = build_full_synthetic_snapshot_lock(cache_root, new_plans)
    lock_path = repository / SNAPSHOT_LOCK_RELATIVE
    lock_bytes = (
        json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if lock_path.exists():
        if lock_path.read_bytes() != lock_bytes:
            raise FileExistsError("existing Development snapshot lock differs")
    else:
        temporary = lock_path.with_name(f".{lock_path.name}.tmp")
        temporary.write_bytes(lock_bytes)
        os.replace(temporary, lock_path)
    validate_full_synthetic_snapshot_lock(lock_path, cache_root, new_plans)
    imports = source_runtime_import_paths()
    if monitor.count or imports:
        raise PermissionError("source-repository runtime isolation failed")
    report = {
        "COMBINED_1260_SNAPSHOTS_PLANNED": len(combined_plans) == 1260,
        "COMBINED_2520_TRIALS_PLANNED": len(combined_trials) == 2520,
        "GENERATOR_REGRESSION_PASS": reproduction_mismatches == 0,
        "NEW_1050_SNAPSHOTS_COMPLETE": len(verified) == 1050,
        "NEW_2100_TRIALS_PLANNED": len(new_trials) == 2100,
        "PHASE_A_IDEAL_IMPORT_PASS": True,
        **subset,
        "confirmatory_seed_instantiation_count": firewall_totals[
            "CONFIRMATORY_SEED_INSTANTIATION_COUNT"
        ],
        "generator_exported_file_count": export["exported_file_count"],
        "generator_reproduction_checksum_mismatch_count": reproduction_mismatches,
        "generator_reproduction_confirmatory_seed_instantiation_count": (
            reproduction_firewall_totals["CONFIRMATORY_SEED_INSTANTIATION_COUNT"]
        ),
        "generator_reproduction_gt_optimization_leakage_count": (
            reproduction_firewall_totals["GT_OPTIMIZATION_LEAKAGE_COUNT"]
        ),
        "generator_reproduction_old_capture_range_seed_access_count": (
            reproduction_firewall_totals["OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT"]
        ),
        "new_snapshot_count": len(verified),
        "new_trial_count": len(new_trials),
        "resumed_snapshot_count": resumed_count,
        "schema_version": "full_synthetic_snapshot_preparation_v1",
        "scientific_protocol_sha256": frozen_protocol["protocol_sha256"],
        "snapshot_checksum_mismatch_count": 0,
        "snapshot_corrupt_count": 0,
        "snapshot_duplicate_count": 0,
        "snapshot_extra_count": 0,
        "snapshot_file_sha_mismatch_count": 0,
        "snapshot_generated_this_invocation": generated_count,
        "snapshot_lock_sha256": file_sha256(lock_path),
        "snapshot_missing_count": 0,
        "source_repository_runtime_file_read_count": monitor.count,
        "source_repository_runtime_import_count": len(imports),
    }
    write_json(repository / PREPARATION_REPORT_RELATIVE, report)
    return report


__all__ = [
    "METADATA_FIELDS",
    "SNAPSHOT_FILES",
    "FullSyntheticSourceAccessMonitor",
    "build_full_synthetic_snapshot",
    "build_full_synthetic_snapshot_lock",
    "load_full_synthetic_generator_protocol",
    "prepare_full_synthetic_snapshots",
    "read_full_synthetic_snapshot",
    "validate_full_synthetic_snapshot_lock",
    "verify_phase_b_snapshot_subset",
    "verify_existing_snapshot_rebuild_equivalence",
    "write_full_synthetic_snapshot_atomic",
]
