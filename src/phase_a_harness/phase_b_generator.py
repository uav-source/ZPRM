"""Thin, auditable wrapper around the byte-exact frozen scene/snapshot generator."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import yaml

from .contracts import canonical_json_sha256, file_sha256, write_json
from .phase_b_generator_frozen import capture_range as _frozen_capture_range

# The copied zero-perturbation modules retain their original absolute import.
# This alias points that import exclusively at the local byte-exact package.  A
# preloaded package is rejected instead of being silently reused.
_existing_capture_range = sys.modules.get("capture_range")
if _existing_capture_range is not None and _existing_capture_range is not _frozen_capture_range:
    raise ImportError("non-frozen capture_range module was already imported")
sys.modules["capture_range"] = _frozen_capture_range

from .phase_b_generator_frozen.zero_perturbation.backend_phase_a_v1_2 import (
    build_canonical_snapshot,
)
from .phase_b_generator_frozen.zero_perturbation.protocol import (
    DevelopmentSeedFirewall,
    ZeroPerturbationProtocol,
    _freeze,
)
from .phase_b_generator_frozen.zero_perturbation.snapshot_builder import build_snapshot
from .phase_b_generator_frozen.zero_perturbation.types import SnapshotKey


SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
GEOMETRY_SEEDS = (1850310744, 1957656152, 1334931069)
MEASUREMENT_SEED = 217775206
CONDITIONS = ("INDEPENDENT_NOISE_FREE", "FULL_NOISE")
GENERATOR_SHA256 = "f1632095ab6c433e1e917cf0c9a49551683fce2f106761d41c3c3a5e6517b922"
SNAPSHOT_BUILDER_SHA256 = "6718fc442439e52327e01622df6356a0167456a6d7d8070f25d6e684ef69b72e"
DEVELOPMENT_PROTOCOL_SHA256 = "8fe4bcfabfb8492d003b9f690162e106e9fa5745b962ed2c4edfe8666acb291e"


def _raw_sha(array: np.ndarray) -> str:
    value = np.asarray(array)
    if not value.flags.c_contiguous:
        raise ValueError("raw checksum input must be C-contiguous")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_generator_export(root: str | Path) -> dict[str, Any]:
    """Verify recorded source/destination equivalence without reading the source repo."""

    repository = Path(root).resolve()
    manifest = repository / "frozen_assets/phase_b_generator_export_manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    recorded_mismatch = 0
    destination_mismatch = 0
    for row in rows:
        if row["copied_exactly"] != "true" or row["source_sha256"] != row["destination_sha256"]:
            recorded_mismatch += 1
        destination = repository / row["destination_path"]
        if not destination.is_file() or file_sha256(destination) != row["destination_sha256"]:
            destination_mismatch += 1
    report = {
        "GENERATOR_EXPORT_EQUIVALENCE_PASS": bool(
            rows and recorded_mismatch == 0 and destination_mismatch == 0
        ),
        "destination_file_sha_mismatch_count": destination_mismatch,
        "exported_file_count": len(rows),
        "schema_version": "phase_b_generator_export_verification_v1",
        "source_destination_sha_mismatch_count": recorded_mismatch,
        "source_repository_runtime_file_read_count": 0,
    }
    write_json(repository / "artifacts/phase_b_generator_export_verification.json", report)
    return report


def load_frozen_generator_protocol(root: str | Path) -> tuple[ZeroPerturbationProtocol, DevelopmentSeedFirewall]:
    repository = Path(root).resolve()
    source = repository / "configs/zero_perturbation/development_v1.yaml"
    if file_sha256(source) != DEVELOPMENT_PROTOCOL_SHA256:
        raise ValueError("frozen Development protocol SHA mismatch")
    generator = repository / "src/phase_a_harness/phase_b_generator_frozen/capture_range/day2_development_scene.py"
    if file_sha256(generator) != GENERATOR_SHA256:
        raise ValueError("frozen scene generator SHA mismatch")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if type(raw) is not dict:
        raise ValueError("Development protocol root must be a mapping")
    if tuple(raw["scene_generation"]["variants_in_order"]) != SCENES:
        raise ValueError("scene inventory changed")
    declared = {row["name"]: row for row in raw["noise_conditions"]}
    if any(name not in declared for name in CONDITIONS):
        raise ValueError("Phase B conditions are absent from frozen Development protocol")
    development_seeds = MappingProxyType(
        {
            "geometry": MappingProxyType({f"geometry_{index}": value for index, value in enumerate(GEOMETRY_SEEDS)}),
            "measurement": MappingProxyType({"measurement_0": MEASUREMENT_SEED}),
        }
    )
    protocol = ZeroPerturbationProtocol(
        root=repository,
        data=_freeze(raw),
        source_sha256=DEVELOPMENT_PROTOCOL_SHA256,
        development_seeds=development_seeds,
    )
    return protocol, DevelopmentSeedFirewall(protocol)


def reference_matrix(reference_pose: np.ndarray) -> np.ndarray:
    value = np.asarray(reference_pose, dtype=np.float64)
    if value.shape != (8,) or not np.allclose(value[4:7], 0.0) or value[7] != 1.0:
        raise ValueError("frozen Phase B reference must be the identity quaternion TUM row")
    result = np.eye(4, dtype="<f8", order="C")
    result[:3, 3] = value[1:4]
    return result


def reproduce_phase_a_snapshot(
    root: str | Path, *, scene: str, geometry_seed: int = GEOMETRY_SEEDS[0],
    measurement_seed: int = MEASUREMENT_SEED, repeat_index: int = 0,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    protocol, firewall = load_frozen_generator_protocol(repository)
    geometry_index = GEOMETRY_SEEDS.index(int(geometry_seed))
    plan = {
        "condition": "IDEAL_MATCHED",
        "geometry_seed_index": geometry_index,
        "geometry_seed_value": int(geometry_seed),
        "measurement_seed_index": 0,
        "measurement_seed_value": int(measurement_seed),
        "repeat_index": int(repeat_index),
        "scene_variant": str(scene),
        "snapshot_id": f"phase-a-v1/{scene}/{geometry_index}/0/{repeat_index}",
    }
    rebuilt = build_canonical_snapshot(development=protocol, firewall=firewall, plan=plan)
    cached = repository / "data/frozen_snapshots" / plan["snapshot_id"]
    metadata = json.loads((cached / "metadata.json").read_text(encoding="utf-8"))
    expected_parent = np.load(cached / "source_parent_target_indices.npy", allow_pickle=False)
    comparisons = {
        "source_checksum_match": rebuilt.metadata["source_raw_checksum"] == metadata["source_raw_checksum"],
        "target_checksum_match": rebuilt.metadata["target_raw_checksum"] == metadata["target_raw_checksum"],
        "reference_pose_checksum_match": rebuilt.metadata["reference_pose_raw_checksum"] == metadata["reference_pose_raw_checksum"],
        "parent_index_checksum_match": rebuilt.metadata["parent_index_raw_checksum"] == metadata["parent_index_raw_checksum"],
        "parent_point_checksum_match": rebuilt.metadata["source_parent_target_points_map_f64_checksum"] == metadata["source_parent_target_points_map_f64_checksum"],
        "parent_indices_exact_match": bool(np.array_equal(rebuilt.parent_indices, expected_parent)),
        "snapshot_checksum_match": rebuilt.metadata["snapshot_checksum"] == metadata["snapshot_checksum"],
    }
    return {
        **comparisons,
        "firewall_audit": firewall.report(),
        "mismatch_count": sum(not value for value in comparisons.values()),
        "scene_variant": scene,
        "snapshot_id": plan["snapshot_id"],
    }


def reproduce_all_phase_a_anchors(root: str | Path) -> dict[str, Any]:
    rows = [reproduce_phase_a_snapshot(root, scene=scene) for scene in SCENES]
    mismatches = sum(row["mismatch_count"] for row in rows)
    firewall_totals = {
        key: sum(int(row["firewall_audit"][key]) for row in rows)
        for key in rows[0]["firewall_audit"]
    }
    report = {
        "GENERATOR_REPRODUCTION_CHECKSUM_MISMATCH_COUNT": mismatches,
        "GENERATOR_REPRODUCTION_PASS": mismatches == 0 and len(rows) == 7,
        "firewall_totals": firewall_totals,
        "reproduction_snapshot_count": len(rows),
        "rows": rows,
        "schema_version": "phase_b_generator_reproduction_v1",
    }
    write_json(Path(root) / "artifacts/phase_b_generator_reproduction.json", report)
    return report


def build_phase_b_snapshot(
    root: str | Path, *, scene: str, geometry_seed: int, condition: str,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    if scene not in SCENES or geometry_seed not in GEOMETRY_SEEDS or condition not in CONDITIONS:
        raise ValueError("Phase B snapshot identity is outside the frozen protocol")
    protocol, firewall = load_frozen_generator_protocol(repository)
    key = SnapshotKey(
        scene_variant=scene,
        geometry_seed=geometry_seed,
        measurement_seed=MEASUREMENT_SEED,
        repeat_index=0,
        noise_condition=condition,
    )
    bundle = build_snapshot(protocol, firewall, key)
    source = np.ascontiguousarray(bundle.scan_points, dtype="<f4")
    target = np.ascontiguousarray(bundle.map_points, dtype="<f4")
    reference = np.ascontiguousarray(reference_matrix(bundle.reference_pose), dtype="<f8")
    source_world = np.ascontiguousarray(
        (reference[:3, :3] @ source.astype(np.float64).T).T + reference[:3, 3],
        dtype="<f4",
    )
    target_rows = {row.tobytes() for row in target}
    source_is_target_subset = all(row.tobytes() in target_rows for row in source_world)
    if source_is_target_subset:
        raise ValueError("Phase B independent source unexpectedly became a target subset")
    checksums = {
        "source_checksum": _raw_sha(source),
        "target_checksum": _raw_sha(target),
        "reference_pose_checksum": _raw_sha(reference),
    }
    geometry_index = GEOMETRY_SEEDS.index(geometry_seed)
    snapshot_id = f"phase-b-signal-v1/{scene}/g{geometry_index}/{condition}"
    snapshot_checksum = canonical_json_sha256({"snapshot_id": snapshot_id, **checksums})
    parameters = dict(protocol.condition(condition))
    metadata = {
        "condition": condition,
        "dropout_parameters": {
            "map_dropout_fraction": float(parameters["map_dropout_fraction"]),
            "scan_dropout_fraction": float(parameters["scan_dropout_fraction"]),
        },
        "generator_sha256": GENERATOR_SHA256,
        "generator_firewall_audit": firewall.report(),
        "snapshot_builder_sha256": SNAPSHOT_BUILDER_SHA256,
        "geometry_seed": geometry_seed,
        "geometry_seed_index": geometry_index,
        "independent_sampling": True,
        "initial_pose": "reference_pose_exact",
        "measurement_seed": MEASUREMENT_SEED,
        "noise_parameters": {
            "map_noise_sigma_m": float(parameters["map_noise_sigma_m"]),
            "scan_noise_sigma_m": float(parameters["scan_noise_sigma_m"]),
        },
        "reference_pose_checksum": checksums["reference_pose_checksum"],
        "repeat_index": 0,
        "scene_variant": scene,
        "snapshot_checksum": snapshot_checksum,
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


def write_phase_b_snapshot_atomic(cache_root: str | Path, value: Mapping[str, Any]) -> Path:
    root = Path(cache_root).resolve()
    metadata = dict(value["metadata"])
    destination = root / metadata["snapshot_id"]
    if destination.exists():
        raise FileExistsError(f"Phase B snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        for name, array in (
            ("source_points.npy", value["source"]),
            ("target_points.npy", value["target"]),
            ("reference_pose.npy", value["reference"]),
        ):
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
            stream.write(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


__all__ = [
    "CONDITIONS",
    "GENERATOR_SHA256",
    "SNAPSHOT_BUILDER_SHA256",
    "GEOMETRY_SEEDS",
    "MEASUREMENT_SEED",
    "SCENES",
    "build_phase_b_snapshot",
    "load_frozen_generator_protocol",
    "reproduce_all_phase_a_anchors",
    "reproduce_phase_a_snapshot",
    "write_phase_b_snapshot_atomic",
    "verify_generator_export",
]
