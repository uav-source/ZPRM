"""Frozen snapshot-cache implementation for Synthetic Confirmatory v1.

The module import and every plan-audit/read path are RNG-free.  The only code
that can construct a Confirmatory RNG is ``ConfirmatorySeedFirewall.rng``;
that class is imported and instantiated lazily by the formal snapshot builder.
Dry-run code must never call any function whose name starts with ``build_`` or
``prepare_`` in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import canonical_json_sha256, file_sha256


SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
CONDITIONS = ("IDEAL_MATCHED", "INDEPENDENT_NOISE_FREE", "FULL_NOISE")
GEOMETRY_SEEDS = (248284635, 376488233, 198112089, 229684695, 226655024)
MEASUREMENT_SEEDS = (469989467, 1088311622, 916609326)
BACKENDS = ("open3d_point_to_plane", "pcl_point_to_plane")
NON_RNG_MEASUREMENT_SENTINEL = 0
FULL_NOISE_RNG_STREAM_ROLES = (
    "scan_dropout",
    "scan_noise",
    "map_noise",
)
SNAPSHOT_COUNT = 595
TRIAL_COUNT = 1190
SNAPSHOT_LOCK_RELATIVE = Path(
    "data/synthetic_confirmatory_v1_snapshot_lock.json"
)
SNAPSHOT_FILES = frozenset(
    {"metadata.json", "source_points.npy", "target_points.npy", "reference_pose.npy"}
)
SNAPSHOT_BUILDER_CONTRACT_VERSION = "synthetic_confirmatory_snapshot_builder_v1"

_METADATA_FIELDS = frozenset(
    {
        "array_file_sha256",
        "condition",
        "confirmatory_rng_instantiation_count",
        "development_protocol_sha256",
        "dropout_parameters",
        "generator_sha256",
        "geometry_seed",
        "independent_sampling",
        "initial_pose",
        "measurement_seed",
        "metadata_payload_sha256",
        "noise_parameters",
        "planned_snapshot_id",
        "reference_pose_checksum",
        "repeat_index",
        "scene_variant",
        "snapshot_builder_contract_version",
        "snapshot_checksum",
        "snapshot_id",
        "source_checksum",
        "source_is_target_subset",
        "source_point_count",
        "target_checksum",
        "target_point_count",
    }
)


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON key in {label}: {name}")
            result[name] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _plan_identity_sha256(value: Any) -> str:
    """Use the protocol's frozen canonical JSON, including its trailing LF."""

    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_plan_row(plan: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "planned_snapshot_id",
        "scene_variant",
        "condition",
        "geometry_seed",
        "measurement_seed",
        "repeat_index",
    }
    if not required.issubset(plan):
        raise ValueError("Confirmatory snapshot plan row is incomplete")
    measurement = plan["measurement_seed"]
    if measurement in (None, ""):
        measurement_value: int | None = None
    elif isinstance(measurement, bool):
        raise TypeError("measurement_seed cannot be bool")
    else:
        measurement_value = int(measurement)
    row = {
        "planned_snapshot_id": str(plan["planned_snapshot_id"]),
        "scene_variant": str(plan["scene_variant"]),
        "condition": str(plan["condition"]),
        "geometry_seed": int(plan["geometry_seed"]),
        "measurement_seed": measurement_value,
        "repeat_index": int(plan["repeat_index"]),
    }
    scene = row["scene_variant"]
    condition = row["condition"]
    geometry = row["geometry_seed"]
    repeat = row["repeat_index"]
    if scene not in SCENES or condition not in CONDITIONS or geometry not in GEOMETRY_SEEDS:
        raise ValueError("snapshot row is outside Synthetic Confirmatory v1")
    if condition == "FULL_NOISE":
        if measurement_value not in MEASUREMENT_SEEDS or repeat not in range(5):
            raise ValueError("FULL_NOISE realization is outside the frozen design")
    elif measurement_value is not None or repeat != 0:
        raise ValueError("noiseless Confirmatory conditions cannot have pseudo-replicates")
    identity = {
        "condition": condition,
        "geometry_seed": geometry,
        "measurement_seed": measurement_value,
        "repeat_index": repeat,
        "scene_variant": scene,
    }
    expected_id = f"synthetic-confirmatory-v1::{_plan_identity_sha256(identity)}"
    if row["planned_snapshot_id"] != expected_id:
        raise ValueError("Confirmatory snapshot ID formula mismatch")
    return row


def _raw_sha256(value: Any) -> str:
    array = value
    if not getattr(array, "flags", None).c_contiguous:
        raise ValueError("raw checksum input must be C-contiguous")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ConfirmatorySeedFirewall:
    """The sole formal RNG construction point for the frozen Confirmatory plan."""

    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol
        self.rng_instantiation_count = 0
        self.snapshot_access_count = 0

    @property
    def global_seed(self) -> int:
        return int(self.protocol.section("scene_generation")["global_scene_seed"])

    def assert_access(
        self, geometry_seed: int, measurement_seed: int, repeat_index: int
    ) -> None:
        if int(geometry_seed) not in GEOMETRY_SEEDS:
            raise PermissionError("geometry seed is outside Confirmatory v1")
        measurement = int(measurement_seed)
        repeat = int(repeat_index)
        if measurement == NON_RNG_MEASUREMENT_SENTINEL:
            if repeat != 0:
                raise PermissionError("noiseless seed sentinel cannot be repeated")
        elif measurement not in MEASUREMENT_SEEDS or repeat not in range(5):
            raise PermissionError("measurement realization is outside Confirmatory v1")
        self.snapshot_access_count += 1

    def rng(
        self,
        scene_variant: str,
        geometry_seed: int,
        measurement_seed: int,
        repeat_index: int,
        noise_condition: str,
        stream_role: str,
    ) -> Any:
        """Construct one PCG64 stream only for an authorized FULL_NOISE asset."""

        self.assert_access(geometry_seed, measurement_seed, repeat_index)
        if (
            scene_variant not in SCENES
            or noise_condition != "FULL_NOISE"
            or int(measurement_seed) not in MEASUREMENT_SEEDS
            or stream_role not in FULL_NOISE_RNG_STREAM_ROLES
        ):
            raise PermissionError("RNG request is outside the frozen FULL_NOISE contract")
        # Lazy imports are an intentional execution boundary: importing the
        # snapshot module during pre-run cannot initialize NumPy RNG state.
        import numpy as np
        from .phase_b_generator_frozen.zero_perturbation.protocol import canonical_seed

        derived = canonical_seed(
            {
                "development_measurement_seed": int(measurement_seed),
                "scene_variant": str(scene_variant),
                "geometry_seed": int(geometry_seed),
                "repeat_index": int(repeat_index),
                "noise_condition": str(noise_condition),
                "stream_role": str(stream_role),
            }
        )
        self.rng_instantiation_count += 1
        return np.random.Generator(np.random.PCG64(derived))

    def report(self) -> dict[str, int]:
        return {
            "CONFIRMATORY_RNG_INSTANTIATION_COUNT": self.rng_instantiation_count,
            "SNAPSHOT_ACCESS_COUNT": self.snapshot_access_count,
        }


def build_synthetic_confirmatory_snapshot(
    root: str | Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Generate exactly one formal snapshot; never call this during pre-run."""

    import numpy as np

    from .full_synthetic_snapshot_builder import (
        CONDITION_PARAMETERS,
        DEVELOPMENT_PROTOCOL_SHA256,
        GENERATOR_SHA256,
        load_full_synthetic_generator_protocol,
    )
    from .phase_b_generator import SnapshotKey, build_snapshot, reference_matrix

    repository = Path(root).resolve()
    row = _canonical_plan_row(plan)
    protocol = load_full_synthetic_generator_protocol(repository)
    firewall = ConfirmatorySeedFirewall(protocol)
    condition = row["condition"]
    # Zero is a non-scientific internal sentinel.  The two noiseless conditions
    # never call rng(); measurement_seed remains null in metadata and identity.
    internal_measurement_seed = (
        int(row["measurement_seed"])
        if row["measurement_seed"] is not None
        else NON_RNG_MEASUREMENT_SENTINEL
    )
    bundle = build_snapshot(
        protocol,
        firewall,
        SnapshotKey(
            scene_variant=row["scene_variant"],
            geometry_seed=row["geometry_seed"],
            measurement_seed=internal_measurement_seed,
            repeat_index=row["repeat_index"],
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
    target_rows = {point.tobytes() for point in target}
    source_is_target_subset = all(point.tobytes() in target_rows for point in source_world)
    if source_is_target_subset != (condition == "IDEAL_MATCHED"):
        raise ValueError("Confirmatory sampling relationship changed")
    checksums = {
        "source_checksum": _raw_sha256(source),
        "target_checksum": _raw_sha256(target),
        "reference_pose_checksum": _raw_sha256(reference),
    }
    snapshot_id = row["planned_snapshot_id"]
    parameters = CONDITION_PARAMETERS[condition]
    metadata: dict[str, Any] = {
        "condition": condition,
        "confirmatory_rng_instantiation_count": firewall.rng_instantiation_count,
        "development_protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
        "dropout_parameters": {
            "map_dropout_fraction": float(parameters["map_dropout_fraction"]),
            "scan_dropout_fraction": float(parameters["scan_dropout_fraction"]),
        },
        "generator_sha256": GENERATOR_SHA256,
        "geometry_seed": row["geometry_seed"],
        "independent_sampling": bool(parameters["independent_sampling"]),
        "initial_pose": "reference_pose_exact",
        "measurement_seed": row["measurement_seed"],
        "noise_parameters": {
            "map_noise_sigma_m": float(parameters["map_noise_sigma_m"]),
            "scan_noise_sigma_m": float(parameters["scan_noise_sigma_m"]),
        },
        "planned_snapshot_id": snapshot_id,
        "reference_pose_checksum": checksums["reference_pose_checksum"],
        "repeat_index": row["repeat_index"],
        "scene_variant": row["scene_variant"],
        "snapshot_builder_contract_version": SNAPSHOT_BUILDER_CONTRACT_VERSION,
        "snapshot_checksum": canonical_json_sha256(
            {"snapshot_id": snapshot_id, **checksums}
        ),
        "snapshot_id": snapshot_id,
        "source_checksum": checksums["source_checksum"],
        "source_is_target_subset": source_is_target_subset,
        "source_point_count": int(source.shape[0]),
        "target_checksum": checksums["target_checksum"],
        "target_point_count": int(target.shape[0]),
    }
    return {
        "metadata": metadata,
        "reference": reference,
        "source": source,
        "target": target,
    }


def write_synthetic_confirmatory_snapshot_atomic(
    cache_root: str | Path, value: Mapping[str, Any]
) -> Path:
    """Atomically create a snapshot directory; replacement is forbidden."""

    import numpy as np

    root = Path(cache_root).resolve()
    metadata = dict(value["metadata"])
    destination = (root / str(metadata["snapshot_id"])).resolve()
    if root not in destination.parents:
        raise ValueError("snapshot path escaped Confirmatory cache")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite snapshot: {destination}")
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


def read_synthetic_confirmatory_snapshot(
    cache_root: str | Path,
    plan: Mapping[str, Any],
    *,
    expected_lock_entry: Mapping[str, Any] | None = None,
    arrays: bool = True,
) -> dict[str, Any]:
    """Strictly validate one cached snapshot without constructing any RNG."""

    import numpy as np

    row = _canonical_plan_row(plan)
    root = Path(cache_root).resolve()
    directory = (root / row["planned_snapshot_id"]).resolve()
    if root not in directory.parents:
        raise ValueError("snapshot path escaped Confirmatory cache")
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != SNAPSHOT_FILES:
        raise ValueError(f"snapshot file inventory mismatch: {row['planned_snapshot_id']}")
    metadata = _strict_json_object(directory / "metadata.json", "snapshot metadata")
    if set(metadata) != _METADATA_FIELDS:
        raise ValueError("Confirmatory snapshot metadata fields changed")
    unsigned = dict(metadata)
    stored_payload_sha = unsigned.pop("metadata_payload_sha256")
    if stored_payload_sha != canonical_json_sha256(unsigned):
        raise ValueError("snapshot metadata payload SHA mismatch")
    exact = {
        "condition": row["condition"],
        "geometry_seed": row["geometry_seed"],
        "measurement_seed": row["measurement_seed"],
        "planned_snapshot_id": row["planned_snapshot_id"],
        "repeat_index": row["repeat_index"],
        "scene_variant": row["scene_variant"],
        "snapshot_id": row["planned_snapshot_id"],
    }
    if any(metadata.get(name) != expected for name, expected in exact.items()):
        raise ValueError("snapshot plan/metadata identity mismatch")
    expected_rng_count = 3 if row["condition"] == "FULL_NOISE" else 0
    if (
        metadata["initial_pose"] != "reference_pose_exact"
        or metadata["snapshot_builder_contract_version"]
        != SNAPSHOT_BUILDER_CONTRACT_VERSION
        or metadata["confirmatory_rng_instantiation_count"] != expected_rng_count
    ):
        raise ValueError("snapshot execution contract changed")
    from .full_synthetic_snapshot_builder import (
        CONDITION_PARAMETERS,
        DEVELOPMENT_PROTOCOL_SHA256,
        GENERATOR_SHA256,
    )

    parameters = CONDITION_PARAMETERS[row["condition"]]
    if (
        metadata["development_protocol_sha256"] != DEVELOPMENT_PROTOCOL_SHA256
        or metadata["generator_sha256"] != GENERATOR_SHA256
        or metadata["independent_sampling"]
        is not bool(parameters["independent_sampling"])
        or metadata["dropout_parameters"]
        != {
            "map_dropout_fraction": float(parameters["map_dropout_fraction"]),
            "scan_dropout_fraction": float(parameters["scan_dropout_fraction"]),
        }
        or metadata["noise_parameters"]
        != {
            "map_noise_sigma_m": float(parameters["map_noise_sigma_m"]),
            "scan_noise_sigma_m": float(parameters["scan_noise_sigma_m"]),
        }
    ):
        raise ValueError("snapshot frozen generator/condition contract changed")
    digests = {name: file_sha256(directory / name) for name in sorted(SNAPSHOT_FILES)}
    if metadata["array_file_sha256"] != {
        name: digests[name]
        for name in ("reference_pose.npy", "source_points.npy", "target_points.npy")
    }:
        raise ValueError("snapshot array file SHA mismatch")
    if expected_lock_entry is not None and (
        expected_lock_entry.get("snapshot_id") != row["planned_snapshot_id"]
        or expected_lock_entry.get("file_sha256") != digests
    ):
        raise ValueError("snapshot lock file binding mismatch")
    result: dict[str, Any] = {
        "directory": directory,
        "file_sha256": digests,
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
        or not all(item.flags.c_contiguous for item in (source, target, reference))
        or not all(np.all(np.isfinite(item)) for item in (source, target, reference))
    ):
        raise ValueError("snapshot array dtype/shape/finite contract changed")
    if metadata["source_point_count"] != len(source) or metadata["target_point_count"] != len(target):
        raise ValueError("snapshot point count mismatch")
    raw = {
        "source_checksum": _raw_sha256(source),
        "target_checksum": _raw_sha256(target),
        "reference_pose_checksum": _raw_sha256(reference),
    }
    if any(metadata[name] != digest for name, digest in raw.items()):
        raise ValueError("snapshot raw checksum mismatch")
    snapshot_checksum = canonical_json_sha256(
        {"snapshot_id": row["planned_snapshot_id"], **raw}
    )
    if metadata["snapshot_checksum"] != snapshot_checksum:
        raise ValueError("snapshot aggregate checksum mismatch")
    source_world = np.ascontiguousarray(
        source.astype(np.float64) @ reference[:3, :3].T + reference[:3, 3],
        dtype="<f4",
    )
    target_rows = {point.tobytes() for point in target}
    is_subset = all(point.tobytes() in target_rows for point in source_world)
    if is_subset != (row["condition"] == "IDEAL_MATCHED") or metadata["source_is_target_subset"] != is_subset:
        raise ValueError("snapshot sampling relationship mismatch")
    if expected_lock_entry is not None:
        lock_exact = {
            **raw,
            "metadata_payload_sha256": metadata["metadata_payload_sha256"],
            "snapshot_checksum": snapshot_checksum,
        }
        if any(expected_lock_entry.get(name) != value for name, value in lock_exact.items()):
            raise ValueError("snapshot lock checksum payload mismatch")
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
        "confirmatory_rng_instantiation_count": metadata[
            "confirmatory_rng_instantiation_count"
        ],
        "file_sha256": dict(snapshot["file_sha256"]),
        "geometry_seed": metadata["geometry_seed"],
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


def build_synthetic_confirmatory_snapshot_lock(
    cache_root: str | Path, plans: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    entries = [
        _lock_entry(read_synthetic_confirmatory_snapshot(cache_root, plan, arrays=True))
        for plan in plans
    ]
    condition_counts = Counter(entry["condition"] for entry in entries)
    core: dict[str, Any] = {
        "condition_snapshot_counts": dict(sorted(condition_counts.items())),
        "confirmatory_rng_instantiation_count": sum(
            int(entry["confirmatory_rng_instantiation_count"]) for entry in entries
        ),
        "planned_snapshot_count": len(entries),
        "schema_version": "synthetic_confirmatory_snapshot_lock_v1",
        "snapshot_builder_contract_version": SNAPSHOT_BUILDER_CONTRACT_VERSION,
        "snapshots": entries,
    }
    return {**core, "snapshot_lock_payload_sha256": canonical_json_sha256(core)}


def validate_synthetic_confirmatory_snapshot_lock(
    lock_path: str | Path,
    cache_root: str | Path,
    plans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = _strict_json_object(Path(lock_path), "Confirmatory snapshot lock")
    unsigned = {
        name: item
        for name, item in value.items()
        if name != "snapshot_lock_payload_sha256"
    }
    entries = value.get("snapshots")
    expected_ids = [_canonical_plan_row(plan)["planned_snapshot_id"] for plan in plans]
    if (
        value.get("snapshot_lock_payload_sha256") != canonical_json_sha256(unsigned)
        or value.get("schema_version") != "synthetic_confirmatory_snapshot_lock_v1"
        or value.get("snapshot_builder_contract_version")
        != SNAPSHOT_BUILDER_CONTRACT_VERSION
        or value.get("planned_snapshot_count") != SNAPSHOT_COUNT
        or value.get("condition_snapshot_counts")
        != {
            "FULL_NOISE": 525,
            "IDEAL_MATCHED": 35,
            "INDEPENDENT_NOISE_FREE": 35,
        }
        or value.get("confirmatory_rng_instantiation_count") != 1575
        or type(entries) is not list
        or len(entries) != SNAPSHOT_COUNT
        or [entry.get("snapshot_id") for entry in entries] != expected_ids
    ):
        raise ValueError("Confirmatory snapshot lock identity/inventory mismatch")
    for plan, entry in zip(plans, entries):
        read_synthetic_confirmatory_snapshot(
            cache_root, plan, expected_lock_entry=entry, arrays=True
        )
    return value


def prepare_synthetic_confirmatory_snapshots(
    root: str | Path,
    cache_root: str | Path,
    plans: Sequence[Mapping[str, Any]],
    *,
    lock_path: str | Path,
    resume: bool,
) -> dict[str, Any]:
    """Materialize the formal cache once, or strictly validate it on resume."""

    if not resume:
        raise PermissionError("Confirmatory snapshot preparation requires --resume")
    if len(plans) != SNAPSHOT_COUNT:
        raise ValueError("Confirmatory snapshot plan count changed")
    cache = Path(cache_root).resolve()
    generated = 0
    resumed = 0
    rng_instantiations = 0
    for plan in plans:
        row = _canonical_plan_row(plan)
        directory = (cache / row["planned_snapshot_id"]).resolve()
        if directory.exists():
            read_synthetic_confirmatory_snapshot(cache, row, arrays=True)
            resumed += 1
        else:
            value = build_synthetic_confirmatory_snapshot(root, row)
            rng_instantiations += int(
                value["metadata"]["confirmatory_rng_instantiation_count"]
            )
            write_synthetic_confirmatory_snapshot_atomic(cache, value)
            read_synthetic_confirmatory_snapshot(cache, row, arrays=True)
            generated += 1
    lock_file = Path(lock_path).resolve()
    if lock_file.exists():
        lock = validate_synthetic_confirmatory_snapshot_lock(
            lock_file, cache, plans
        )
    else:
        lock = build_synthetic_confirmatory_snapshot_lock(cache, plans)
        from .phase_a_trial_result_writer import atomic_write_bytes

        atomic_write_bytes(
            lock_file,
            (json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
            replace=False,
        )
    return {
        "confirmatory_rng_instantiation_count_this_invocation": rng_instantiations,
        "generated_snapshot_count": generated,
        "lock": lock,
        "lock_path": lock_file,
        "resumed_snapshot_count": resumed,
    }


__all__ = [
    "BACKENDS",
    "CONDITIONS",
    "ConfirmatorySeedFirewall",
    "GEOMETRY_SEEDS",
    "FULL_NOISE_RNG_STREAM_ROLES",
    "MEASUREMENT_SEEDS",
    "NON_RNG_MEASUREMENT_SENTINEL",
    "SCENES",
    "SNAPSHOT_BUILDER_CONTRACT_VERSION",
    "SNAPSHOT_COUNT",
    "SNAPSHOT_LOCK_RELATIVE",
    "TRIAL_COUNT",
    "build_synthetic_confirmatory_snapshot",
    "build_synthetic_confirmatory_snapshot_lock",
    "prepare_synthetic_confirmatory_snapshots",
    "read_synthetic_confirmatory_snapshot",
    "validate_synthetic_confirmatory_snapshot_lock",
    "write_synthetic_confirmatory_snapshot_atomic",
]
