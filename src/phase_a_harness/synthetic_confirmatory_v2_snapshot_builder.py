"""Auditable parent-lineage snapshot cache for Synthetic Confirmatory v2.

Formal v2 construction is reachable only through ``build_v2_snapshot``.  The
Development qualification entry points use a geometry-only firewall and reject
all v1 and v2 Confirmatory seeds.  Import and read paths are RNG-free.
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
from .synthetic_confirmatory_v2_contract import (
    CONDITIONS,
    GEOMETRY_SEEDS,
    LINEAGE_SCHEMA,
    MEASUREMENT_SEEDS,
    METADATA_SCHEMA,
    OLD_V1_BOOTSTRAP_SEED,
    OLD_V1_GEOMETRY_SEEDS,
    OLD_V1_MEASUREMENT_SEEDS,
    SCENES,
    SNAPSHOT_COUNT,
    SNAPSHOT_LOCK_SCHEMA,
    SNAPSHOT_SCHEMA,
    validate_snapshot_plan_row,
)


NON_RNG_MEASUREMENT_SENTINEL = 0
FULL_NOISE_RNG_STREAM_ROLES = ("scan_dropout", "scan_noise", "map_noise")
SNAPSHOT_BUILDER_CONTRACT_VERSION = "synthetic_confirmatory_snapshot_builder_v2"
PARENT_INDEX_FILENAME = "source_parent_target_indices.npy"
BASE_ARRAY_FILENAMES = (
    "source_points.npy",
    "target_points.npy",
    "reference_pose.npy",
)
DEVELOPMENT_GEOMETRY_SEEDS = (1850310744, 1957656152, 1334931069)
DEVELOPMENT_MEASUREMENT_SEEDS = (217775206, 1664898153)

_CLOSURE_FIELDS = (
    "reconstruction_error_median_m",
    "reconstruction_error_q95_m",
    "reconstruction_error_max_m",
    "predicted_quantization_median_m",
    "predicted_quantization_q95_m",
    "predicted_quantization_max_m",
    "closure_residual_max_m",
    "float64_guard_max_m",
    "max_normalized_closure_ratio",
    "closure_residual_violation_count",
    "actual_error_bound_violation_count",
    "quantization_closure_pass",
)
METADATA_FIELDS = frozenset(
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
        "lineage_closure_violation_count",
        "lineage_schema_version",
        "lineage_validation_method",
        "measurement_seed",
        "metadata_payload_sha256",
        "noise_parameters",
        "parent_index_count",
        "parent_index_duplicate_count",
        "parent_index_out_of_range_count",
        "parent_index_unique_count",
        "parent_points_map_f64_sha256",
        "planned_snapshot_id",
        "reference_pose_checksum",
        "repeat_index",
        "scene_variant",
        "schema_version",
        "snapshot_builder_contract_version",
        "snapshot_checksum",
        "snapshot_id",
        "snapshot_schema_version",
        "source_checksum",
        "source_has_target_parent_lineage",
        "source_is_target_subset",
        "source_parent_row_count_match",
        "source_parent_target_indices_path",
        "source_parent_target_indices_sha256",
        "source_point_count",
        "target_checksum",
        "target_point_count",
        *_CLOSURE_FIELDS,
    }
)


def _raw_sha256(value: Any) -> str:
    array = value
    if not getattr(array, "flags", None).c_contiguous:
        raise ValueError("raw checksum input must be C-contiguous")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ConfirmatoryV2SeedFirewall:
    """The only formal v2 RNG construction point."""

    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol
        self.rng_instantiation_count = 0
        self.snapshot_access_count = 0

    @property
    def global_seed(self) -> int:
        return int(self.protocol.section("scene_generation")["global_scene_seed"])

    def assert_access(self, geometry_seed: int, measurement_seed: int, repeat_index: int) -> None:
        geometry = int(geometry_seed)
        measurement = int(measurement_seed)
        repeat = int(repeat_index)
        if geometry in OLD_V1_GEOMETRY_SEEDS or measurement in (
            *OLD_V1_MEASUREMENT_SEEDS, OLD_V1_BOOTSTRAP_SEED
        ):
            raise PermissionError("retired v1 Confirmatory seed is forbidden")
        if geometry not in GEOMETRY_SEEDS:
            raise PermissionError("geometry seed is outside Confirmatory v2")
        if measurement == NON_RNG_MEASUREMENT_SENTINEL:
            if repeat != 0:
                raise PermissionError("noiseless v2 identity cannot be repeated")
        elif measurement not in MEASUREMENT_SEEDS or repeat not in range(5):
            raise PermissionError("measurement realization is outside Confirmatory v2")
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
        self.assert_access(geometry_seed, measurement_seed, repeat_index)
        if (
            scene_variant not in SCENES
            or noise_condition != "FULL_NOISE"
            or int(measurement_seed) not in MEASUREMENT_SEEDS
            or stream_role not in FULL_NOISE_RNG_STREAM_ROLES
        ):
            raise PermissionError("RNG request is outside formal v2 FULL_NOISE")
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
            "NEW_V2_RNG_INSTANTIATION_COUNT": self.rng_instantiation_count,
            "SNAPSHOT_ACCESS_COUNT": self.snapshot_access_count,
        }


class GeometryOnlyQualificationFirewall:
    """Development-only geometry access with no measurement or RNG surface."""

    def __init__(self, protocol: Any, allowed_geometry_seeds: Sequence[int]) -> None:
        allowed = tuple(int(value) for value in allowed_geometry_seeds)
        if set(allowed) != set(DEVELOPMENT_GEOMETRY_SEEDS):
            raise PermissionError("qualification requires the exact Development geometry set")
        self.protocol = protocol
        self.allowed = frozenset(allowed)
        self.geometry_access_count = 0
        self.measurement_seed_access_count = 0
        self.repeat_randomness_count = 0
        self.rng_instantiation_count = 0

    @property
    def global_seed(self) -> int:
        return int(self.protocol.section("scene_generation")["global_scene_seed"])

    def assert_access(self, geometry_seed: int, measurement_seed: int, repeat_index: int) -> None:
        geometry = int(geometry_seed)
        if geometry not in self.allowed:
            raise PermissionError("non-Development geometry seed reached qualification")
        if geometry in GEOMETRY_SEEDS or geometry in OLD_V1_GEOMETRY_SEEDS:
            raise PermissionError("Confirmatory seed reached Development qualification")
        if int(measurement_seed) != NON_RNG_MEASUREMENT_SENTINEL:
            self.measurement_seed_access_count += 1
            raise PermissionError("IDEAL qualification cannot access a measurement seed")
        if int(repeat_index) != 0:
            self.repeat_randomness_count += 1
            raise PermissionError("IDEAL qualification cannot use repeat randomness")
        self.geometry_access_count += 1

    def rng(self, *args: Any, **kwargs: Any) -> Any:
        self.rng_instantiation_count += 1
        raise PermissionError("geometry-only qualification cannot construct RNG")

    def report(self) -> dict[str, int]:
        return {
            "geometry_access_count": self.geometry_access_count,
            "measurement_seed_access_count": self.measurement_seed_access_count,
            "repeat_randomness_count": self.repeat_randomness_count,
            "rng_instantiation_count": self.rng_instantiation_count,
        }


def _lineage_arrays(
    *, protocol: Any, firewall: Any, scene: str, geometry_seed: int
) -> dict[str, Any]:
    import numpy as np
    # Importing the wrapper first installs the local frozen capture_range alias.
    from . import phase_b_generator as _local_generator  # noqa: F401
    from .phase_b_generator_frozen.capture_range.day2_development_scene import (
        build_scene_geometry,
    )
    from .phase_b_generator_frozen.zero_perturbation.backend_phase_a_v1_2 import (
        canonical_target,
        eligible_parent_indices,
        quantization_closure,
        reference_pose_from_development,
        source_from_parent_indices,
    )

    geometry = build_scene_geometry(
        protocol,
        firewall,
        scene,
        int(geometry_seed),
        NON_RNG_MEASUREMENT_SENTINEL,
        0,
        "map",
    )
    target = canonical_target(geometry.points_world)
    reference = reference_pose_from_development(protocol)
    indices = eligible_parent_indices(target, reference)
    source, source_f64, parents = source_from_parent_indices(target, indices, reference)
    closure = quantization_closure(
        source_points=source,
        source_float64=source_f64,
        parent_points_map_float64=parents,
        reference_pose=reference,
    )
    duplicate_count = int(len(indices) - len(np.unique(indices)))
    out_of_range = int(np.count_nonzero((indices < 0) | (indices >= len(target))))
    row_match = bool(len(source) == len(indices))
    closure_violations = int(
        closure["closure_residual_violation_count"]
        + closure["actual_error_bound_violation_count"]
    )
    lineage = bool(
        indices.dtype == np.dtype("<i8")
        and indices.ndim == 1
        and indices.flags.c_contiguous
        and row_match
        and duplicate_count == 0
        and out_of_range == 0
        and closure["quantization_closure_pass"] is True
        and closure_violations == 0
    )
    if not lineage:
        raise ValueError("v2 IDEAL parent lineage validation failed")
    return {
        "closure": closure,
        "indices": indices,
        "lineage_closure_violation_count": closure_violations,
        "parent_points": parents,
        "reference": reference,
        "source": source,
        "source_f64": source_f64,
        "target": target,
    }


def _metadata(
    *,
    snapshot_id: str,
    scene: str,
    condition: str,
    geometry_seed: int,
    measurement_seed: int | None,
    repeat_index: int,
    source: Any,
    target: Any,
    reference: Any,
    rng_count: int,
    lineage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from .full_synthetic_development_protocol import CONDITION_PARAMETERS
    from .full_synthetic_snapshot_builder import (
        DEVELOPMENT_PROTOCOL_SHA256,
        GENERATOR_SHA256,
    )

    checksums = {
        "source_checksum": _raw_sha256(source),
        "target_checksum": _raw_sha256(target),
        "reference_pose_checksum": _raw_sha256(reference),
    }
    parent_sha = None if lineage is None else _raw_sha256(lineage["indices"])
    snapshot_checksum = canonical_json_sha256(
        {"snapshot_id": snapshot_id, **checksums, "source_parent_target_indices_sha256": parent_sha}
    )
    parameters = CONDITION_PARAMETERS[condition]
    if lineage is None:
        lineage_fields: dict[str, Any] = {
            "actual_error_bound_violation_count": None,
            "closure_residual_max_m": None,
            "closure_residual_violation_count": None,
            "float64_guard_max_m": None,
            "lineage_closure_violation_count": 0,
            "lineage_schema_version": LINEAGE_SCHEMA,
            "lineage_validation_method": "NOT_APPLICABLE_NONIDEAL",
            "max_normalized_closure_ratio": None,
            "parent_index_count": 0,
            "parent_index_duplicate_count": 0,
            "parent_index_out_of_range_count": 0,
            "parent_index_unique_count": 0,
            "parent_points_map_f64_sha256": None,
            "predicted_quantization_max_m": None,
            "predicted_quantization_median_m": None,
            "predicted_quantization_q95_m": None,
            "quantization_closure_pass": False,
            "reconstruction_error_max_m": None,
            "reconstruction_error_median_m": None,
            "reconstruction_error_q95_m": None,
            "source_has_target_parent_lineage": False,
            "source_is_target_subset": False,
            "source_parent_row_count_match": False,
            "source_parent_target_indices_path": None,
            "source_parent_target_indices_sha256": None,
        }
    else:
        import numpy as np

        indices = lineage["indices"]
        closure = dict(lineage["closure"])
        parent_index_count = int(len(indices))
        parent_index_unique_count = int(len(np.unique(indices)))
        parent_index_duplicate_count = int(
            parent_index_count - parent_index_unique_count
        )
        parent_index_out_of_range_count = int(
            np.count_nonzero((indices < 0) | (indices >= len(target)))
        )
        source_parent_row_count_match = bool(len(source) == parent_index_count)
        lineage_closure_violation_count = int(
            lineage["lineage_closure_violation_count"]
        )
        lineage_valid = bool(
            indices.dtype == np.dtype("<i8")
            and indices.ndim == 1
            and indices.flags.c_contiguous
            and source_parent_row_count_match
            and parent_index_duplicate_count == 0
            and parent_index_out_of_range_count == 0
            and closure["quantization_closure_pass"] is True
            and lineage_closure_violation_count == 0
        )
        if not lineage_valid:
            raise ValueError("v2 IDEAL metadata cannot claim invalid parent lineage")
        lineage_fields = {
            **closure,
            "lineage_closure_violation_count": lineage_closure_violation_count,
            "lineage_schema_version": LINEAGE_SCHEMA,
            "lineage_validation_method": (
                "PARENT_INDEX_ROW_CORRESPONDENCE_PLUS_PHASE_A_QUANTIZATION_CLOSURE"
            ),
            "parent_index_count": parent_index_count,
            "parent_index_duplicate_count": parent_index_duplicate_count,
            "parent_index_out_of_range_count": parent_index_out_of_range_count,
            "parent_index_unique_count": parent_index_unique_count,
            "parent_points_map_f64_sha256": _raw_sha256(
                np.ascontiguousarray(lineage["parent_points"], dtype="<f8")
            ),
            "source_has_target_parent_lineage": lineage_valid,
            "source_is_target_subset": lineage_valid,
            "source_parent_row_count_match": source_parent_row_count_match,
            "source_parent_target_indices_path": PARENT_INDEX_FILENAME,
            "source_parent_target_indices_sha256": parent_sha,
        }
    return {
        "array_file_sha256": {},
        "condition": condition,
        "confirmatory_rng_instantiation_count": int(rng_count),
        "development_protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
        "dropout_parameters": {
            "map_dropout_fraction": float(parameters["map_dropout_fraction"]),
            "scan_dropout_fraction": float(parameters["scan_dropout_fraction"]),
        },
        "generator_sha256": GENERATOR_SHA256,
        "geometry_seed": int(geometry_seed),
        "independent_sampling": bool(parameters["independent_sampling"]),
        "initial_pose": "reference_pose_exact",
        "measurement_seed": measurement_seed,
        "metadata_payload_sha256": "",
        "noise_parameters": {
            "map_noise_sigma_m": float(parameters["map_noise_sigma_m"]),
            "scan_noise_sigma_m": float(parameters["scan_noise_sigma_m"]),
        },
        "planned_snapshot_id": snapshot_id,
        "reference_pose_checksum": checksums["reference_pose_checksum"],
        "repeat_index": int(repeat_index),
        "scene_variant": scene,
        "schema_version": METADATA_SCHEMA,
        "snapshot_builder_contract_version": SNAPSHOT_BUILDER_CONTRACT_VERSION,
        "snapshot_checksum": snapshot_checksum,
        "snapshot_id": snapshot_id,
        "snapshot_schema_version": SNAPSHOT_SCHEMA,
        "source_checksum": checksums["source_checksum"],
        "source_point_count": int(len(source)),
        "target_checksum": checksums["target_checksum"],
        "target_point_count": int(len(target)),
        **lineage_fields,
    }


def _nonideal_arrays(
    *,
    protocol: Any,
    firewall: Any,
    scene: str,
    geometry_seed: int,
    measurement_seed: int,
    repeat_index: int,
    condition: str,
) -> dict[str, Any]:
    """One shared numerical path for formal-v2 and Development regression."""

    import numpy as np
    from .phase_b_generator import SnapshotKey, build_snapshot, reference_matrix

    if condition == "IDEAL_MATCHED":
        raise ValueError("IDEAL must use the explicit parent-lineage path")
    bundle = build_snapshot(
        protocol,
        firewall,
        SnapshotKey(
            scene_variant=scene,
            geometry_seed=int(geometry_seed),
            measurement_seed=int(measurement_seed),
            repeat_index=int(repeat_index),
            noise_condition=condition,
        ),
    )
    return {
        "map_point_count_before_dropout": int(
            bundle.map_point_count_before_dropout
        ),
        "reference": np.ascontiguousarray(
            reference_matrix(bundle.reference_pose), dtype="<f8"
        ),
        "record_checksums": dict(bundle.checksums),
        "scan_point_count_before_dropout": int(
            bundle.scan_point_count_before_dropout
        ),
        "source": np.ascontiguousarray(bundle.scan_points, dtype="<f4"),
        "target": np.ascontiguousarray(bundle.map_points, dtype="<f4"),
    }


def build_v2_snapshot(root: str | Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Construct one formal v2 snapshot. Never call during pre-run qualification."""

    from .full_synthetic_snapshot_builder import load_full_synthetic_generator_protocol

    repository = Path(root).resolve()
    row = validate_snapshot_plan_row(plan)
    protocol = load_full_synthetic_generator_protocol(repository)
    firewall = ConfirmatoryV2SeedFirewall(protocol)
    condition = row["condition"]
    if condition == "IDEAL_MATCHED":
        lineage = _lineage_arrays(
            protocol=protocol,
            firewall=firewall,
            scene=row["scene_variant"],
            geometry_seed=row["geometry_seed"],
        )
        source, target, reference = (
            lineage["source"], lineage["target"], lineage["reference"]
        )
    else:
        internal_measurement = (
            NON_RNG_MEASUREMENT_SENTINEL
            if row["measurement_seed"] is None else int(row["measurement_seed"])
        )
        arrays = _nonideal_arrays(
            protocol=protocol, firewall=firewall,
            scene=row["scene_variant"], geometry_seed=row["geometry_seed"],
            measurement_seed=internal_measurement,
            repeat_index=row["repeat_index"], condition=condition,
        )
        source, target, reference = (
            arrays["source"], arrays["target"], arrays["reference"]
        )
        lineage = None
    metadata = _metadata(
        snapshot_id=row["planned_snapshot_id"], scene=row["scene_variant"],
        condition=condition, geometry_seed=row["geometry_seed"],
        measurement_seed=row["measurement_seed"], repeat_index=row["repeat_index"],
        source=source, target=target, reference=reference,
        rng_count=firewall.rng_instantiation_count, lineage=lineage,
    )
    return {
        "firewall_audit": firewall.report(),
        "metadata": metadata,
        "parent_indices": None if lineage is None else lineage["indices"],
        "reference": reference,
        "source": source,
        "target": target,
    }


def build_development_nonideal_regression_snapshot(
    root: str | Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Exercise the v2 non-IDEAL path using only frozen Development seeds."""

    from .full_synthetic_development_protocol import (
        GEOMETRY_SEEDS as FROZEN_DEVELOPMENT_GEOMETRY_SEEDS,
        MEASUREMENT_SEEDS as FROZEN_DEVELOPMENT_MEASUREMENT_SEEDS,
        NEW_CONDITIONS,
    )
    from .full_synthetic_snapshot_builder import (
        load_full_synthetic_generator_protocol,
    )
    from .phase_b_generator import DevelopmentSeedFirewall

    required = {
        "snapshot_id", "scene_variant", "geometry_seed_index",
        "geometry_seed_value", "measurement_seed_index",
        "measurement_seed_value", "repeat_index", "condition",
    }
    if not required.issubset(plan):
        raise ValueError("Development non-IDEAL regression plan is incomplete")
    scene = str(plan["scene_variant"])
    condition = str(plan["condition"])
    geometry_index = int(plan["geometry_seed_index"])
    measurement_index = int(plan["measurement_seed_index"])
    geometry_seed = int(plan["geometry_seed_value"])
    measurement_seed = int(plan["measurement_seed_value"])
    repeat_index = int(plan["repeat_index"])
    if (
        scene not in SCENES
        or condition not in NEW_CONDITIONS
        or geometry_index not in range(len(FROZEN_DEVELOPMENT_GEOMETRY_SEEDS))
        or measurement_index not in range(len(FROZEN_DEVELOPMENT_MEASUREMENT_SEEDS))
        or geometry_seed != FROZEN_DEVELOPMENT_GEOMETRY_SEEDS[geometry_index]
        or measurement_seed
        != FROZEN_DEVELOPMENT_MEASUREMENT_SEEDS[measurement_index]
        or repeat_index not in range(5)
        or geometry_seed in GEOMETRY_SEEDS
        or measurement_seed in MEASUREMENT_SEEDS
        or geometry_seed in OLD_V1_GEOMETRY_SEEDS
        or measurement_seed in OLD_V1_MEASUREMENT_SEEDS
    ):
        raise PermissionError("regression identity is outside frozen Development")
    protocol = load_full_synthetic_generator_protocol(root)
    firewall = DevelopmentSeedFirewall(protocol)
    arrays = _nonideal_arrays(
        protocol=protocol, firewall=firewall, scene=scene,
        geometry_seed=geometry_seed, measurement_seed=measurement_seed,
        repeat_index=repeat_index, condition=condition,
    )
    metadata = _metadata(
        snapshot_id=str(plan["snapshot_id"]), scene=scene, condition=condition,
        geometry_seed=geometry_seed, measurement_seed=measurement_seed,
        repeat_index=repeat_index, source=arrays["source"],
        target=arrays["target"], reference=arrays["reference"],
        rng_count=0, lineage=None,
    )
    return {
        **arrays,
        "firewall_audit": firewall.report(),
        "metadata": metadata,
        "parent_indices": None,
    }


def build_development_ideal_qualification_snapshot(
    root: str | Path, *, scene: str, geometry_seed: int
) -> dict[str, Any]:
    from .full_synthetic_snapshot_builder import load_full_synthetic_generator_protocol

    if scene not in SCENES or int(geometry_seed) not in DEVELOPMENT_GEOMETRY_SEEDS:
        raise PermissionError("IDEAL qualification identity is outside Development")
    protocol = load_full_synthetic_generator_protocol(root)
    firewall = GeometryOnlyQualificationFirewall(protocol, DEVELOPMENT_GEOMETRY_SEEDS)
    lineage = _lineage_arrays(
        protocol=protocol, firewall=firewall, scene=scene,
        geometry_seed=int(geometry_seed),
    )
    snapshot_id = f"synthetic-confirmatory-v2-qualification/ideal/{scene}/{int(geometry_seed)}"
    metadata = _metadata(
        snapshot_id=snapshot_id, scene=scene, condition="IDEAL_MATCHED",
        geometry_seed=int(geometry_seed), measurement_seed=None, repeat_index=0,
        source=lineage["source"], target=lineage["target"],
        reference=lineage["reference"], rng_count=0, lineage=lineage,
    )
    return {
        "firewall_audit": firewall.report(), "metadata": metadata,
        "parent_indices": lineage["indices"], "reference": lineage["reference"],
        "source": lineage["source"], "target": lineage["target"],
    }


def build_development_independent_qualification_snapshot(
    root: str | Path, *, scene: str, geometry_seed: int
) -> dict[str, Any]:
    import numpy as np
    from .full_synthetic_snapshot_builder import load_full_synthetic_generator_protocol
    from .phase_b_generator import SnapshotKey, build_snapshot, reference_matrix

    if scene not in SCENES or int(geometry_seed) not in DEVELOPMENT_GEOMETRY_SEEDS:
        raise PermissionError("INDEPENDENT qualification identity is outside Development")
    protocol = load_full_synthetic_generator_protocol(root)
    firewall = GeometryOnlyQualificationFirewall(protocol, DEVELOPMENT_GEOMETRY_SEEDS)
    bundle = build_snapshot(
        protocol, firewall,
        SnapshotKey(
            scene_variant=scene, geometry_seed=int(geometry_seed),
            measurement_seed=NON_RNG_MEASUREMENT_SENTINEL, repeat_index=0,
            noise_condition="INDEPENDENT_NOISE_FREE",
        ),
    )
    source = np.ascontiguousarray(bundle.scan_points, dtype="<f4")
    target = np.ascontiguousarray(bundle.map_points, dtype="<f4")
    reference = np.ascontiguousarray(reference_matrix(bundle.reference_pose), dtype="<f8")
    snapshot_id = f"synthetic-confirmatory-v2-qualification/independent/{scene}/{int(geometry_seed)}"
    metadata = _metadata(
        snapshot_id=snapshot_id, scene=scene, condition="INDEPENDENT_NOISE_FREE",
        geometry_seed=int(geometry_seed), measurement_seed=None, repeat_index=0,
        source=source, target=target, reference=reference, rng_count=0, lineage=None,
    )
    return {
        "firewall_audit": firewall.report(), "metadata": metadata,
        "parent_indices": None, "reference": reference, "source": source,
        "target": target,
    }


def write_v2_snapshot_atomic(cache_root: str | Path, value: Mapping[str, Any]) -> Path:
    import numpy as np

    root = Path(cache_root).resolve()
    metadata = dict(value["metadata"])
    destination = (root / str(metadata["snapshot_id"])).resolve()
    if root not in destination.parents or destination.exists():
        raise FileExistsError(f"unsafe or existing v2 snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        arrays = [
            ("source_points.npy", value["source"]),
            ("target_points.npy", value["target"]),
            ("reference_pose.npy", value["reference"]),
        ]
        if value.get("parent_indices") is not None:
            arrays.append((PARENT_INDEX_FILENAME, value["parent_indices"]))
        for name, array in arrays:
            with (temporary / name).open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
        metadata["array_file_sha256"] = {
            name: file_sha256(temporary / name) for name, _array in arrays
        }
        unsigned = dict(metadata)
        unsigned.pop("metadata_payload_sha256")
        metadata["metadata_payload_sha256"] = canonical_json_sha256(unsigned)
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


def _lineage_recomputed(source: Any, target: Any, reference: Any, indices: Any) -> dict[str, Any]:
    import numpy as np
    from .phase_b_generator_frozen.zero_perturbation.backend_phase_a_v1_2 import (
        quantization_closure,
        source_from_parent_indices,
    )

    expected, source_f64, parents = source_from_parent_indices(target, indices, reference)
    closure = quantization_closure(
        source_points=source, source_float64=source_f64,
        parent_points_map_float64=parents, reference_pose=reference,
    )
    return {
        "closure": closure,
        "lineage_closure_violation_count": int(
            closure["closure_residual_violation_count"]
            + closure["actual_error_bound_violation_count"]
        ),
        "parent_points_map_f64_sha256": _raw_sha256(
            np.ascontiguousarray(parents, dtype="<f8")
        ),
        "row_correspondence_pass": bool(np.array_equal(source, expected)),
    }


def read_v2_snapshot(
    cache_root: str | Path,
    plan: Mapping[str, Any],
    *,
    expected_lock_entry: Mapping[str, Any] | None = None,
    arrays: bool = True,
) -> dict[str, Any]:
    import numpy as np

    row = validate_snapshot_plan_row(plan)
    root = Path(cache_root).resolve()
    directory = (root / row["planned_snapshot_id"]).resolve()
    expected_files = set(BASE_ARRAY_FILENAMES) | {"metadata.json"}
    if row["condition"] == "IDEAL_MATCHED":
        expected_files.add(PARENT_INDEX_FILENAME)
    if (
        root not in directory.parents or not directory.is_dir()
        or {path.name for path in directory.iterdir()} != expected_files
    ):
        raise ValueError("v2 snapshot file inventory mismatch")
    metadata = _strict_object(directory / "metadata.json", "v2 snapshot metadata")
    if set(metadata) != METADATA_FIELDS:
        raise ValueError("v2 snapshot metadata fields changed")
    unsigned = dict(metadata)
    stored = unsigned.pop("metadata_payload_sha256")
    if stored != canonical_json_sha256(unsigned):
        raise ValueError("v2 snapshot metadata payload SHA mismatch")
    exact = {
        "condition": row["condition"], "geometry_seed": row["geometry_seed"],
        "measurement_seed": row["measurement_seed"],
        "planned_snapshot_id": row["planned_snapshot_id"],
        "repeat_index": row["repeat_index"], "scene_variant": row["scene_variant"],
        "snapshot_id": row["planned_snapshot_id"],
    }
    if (
        any(metadata.get(name) != value for name, value in exact.items())
        or metadata.get("schema_version") != METADATA_SCHEMA
        or metadata.get("snapshot_schema_version") != SNAPSHOT_SCHEMA
        or metadata.get("lineage_schema_version") != LINEAGE_SCHEMA
        or metadata.get("snapshot_builder_contract_version")
        != SNAPSHOT_BUILDER_CONTRACT_VERSION
    ):
        raise ValueError("v2 snapshot metadata identity mismatch")
    digests = {name: file_sha256(directory / name) for name in sorted(expected_files)}
    array_digests = {name: digests[name] for name in expected_files if name.endswith(".npy")}
    if metadata["array_file_sha256"] != array_digests:
        raise ValueError("v2 snapshot array file SHA mismatch")
    if expected_lock_entry is not None and (
        expected_lock_entry.get("snapshot_id") != row["planned_snapshot_id"]
        or expected_lock_entry.get("file_sha256") != digests
    ):
        raise ValueError("v2 snapshot lock file binding mismatch")
    result: dict[str, Any] = {
        "directory": directory, "file_sha256": digests, "metadata": metadata,
    }
    if not arrays:
        return result
    source = np.load(directory / "source_points.npy", allow_pickle=False)
    target = np.load(directory / "target_points.npy", allow_pickle=False)
    reference = np.load(directory / "reference_pose.npy", allow_pickle=False)
    if (
        source.dtype != np.dtype("<f4") or target.dtype != np.dtype("<f4")
        or reference.dtype != np.dtype("<f8") or source.ndim != 2
        or source.shape[1:] != (3,) or target.ndim != 2
        or target.shape[1:] != (3,) or reference.shape != (4, 4)
        or not all(value.flags.c_contiguous for value in (source, target, reference))
        or not all(np.all(np.isfinite(value)) for value in (source, target, reference))
    ):
        raise ValueError("v2 snapshot array contract mismatch")
    raw = {
        "source_checksum": _raw_sha256(source),
        "target_checksum": _raw_sha256(target),
        "reference_pose_checksum": _raw_sha256(reference),
    }
    parent_indices = None
    parent_sha = None
    if row["condition"] == "IDEAL_MATCHED":
        parent_indices = np.load(directory / PARENT_INDEX_FILENAME, allow_pickle=False)
        if (
            parent_indices.dtype != np.dtype("<i8") or parent_indices.ndim != 1
            or not parent_indices.flags.c_contiguous
        ):
            raise ValueError("v2 parent-index array contract mismatch")
        parent_sha = _raw_sha256(parent_indices)
        recomputed = _lineage_recomputed(source, target, reference, parent_indices)
        duplicate = int(len(parent_indices) - len(np.unique(parent_indices)))
        out_of_range = int(
            np.count_nonzero((parent_indices < 0) | (parent_indices >= len(target)))
        )
        expected_lineage = {
            "parent_index_count": len(parent_indices),
            "parent_index_duplicate_count": duplicate,
            "parent_index_out_of_range_count": out_of_range,
            "parent_index_unique_count": len(np.unique(parent_indices)),
            "parent_points_map_f64_sha256": recomputed["parent_points_map_f64_sha256"],
            "source_has_target_parent_lineage": True,
            "source_is_target_subset": True,
            "source_parent_row_count_match": len(source) == len(parent_indices),
            "source_parent_target_indices_path": PARENT_INDEX_FILENAME,
            "source_parent_target_indices_sha256": parent_sha,
            "lineage_closure_violation_count": recomputed["lineage_closure_violation_count"],
            **recomputed["closure"],
        }
        if (
            any(metadata.get(name) != value for name, value in expected_lineage.items())
            or not recomputed["row_correspondence_pass"]
            or duplicate != 0 or out_of_range != 0
            or recomputed["lineage_closure_violation_count"] != 0
        ):
            raise ValueError("v2 IDEAL lineage validation failed")
    else:
        expected_nonideal = {
            "source_has_target_parent_lineage": False,
            "source_is_target_subset": False,
            "source_parent_target_indices_path": None,
            "source_parent_target_indices_sha256": None,
            "parent_index_count": 0,
            "parent_index_unique_count": 0,
            "parent_index_duplicate_count": 0,
            "parent_index_out_of_range_count": 0,
            "source_parent_row_count_match": False,
            "parent_points_map_f64_sha256": None,
            "quantization_closure_pass": False,
        }
        if any(metadata.get(name) != value for name, value in expected_nonideal.items()):
            raise ValueError("v2 non-IDEAL snapshot forged parent lineage")
    snapshot_checksum = canonical_json_sha256(
        {"snapshot_id": row["planned_snapshot_id"], **raw,
         "source_parent_target_indices_sha256": parent_sha}
    )
    if (
        any(metadata[name] != value for name, value in raw.items())
        or metadata["snapshot_checksum"] != snapshot_checksum
        or metadata["source_point_count"] != len(source)
        or metadata["target_point_count"] != len(target)
    ):
        raise ValueError("v2 snapshot checksum or count mismatch")
    if expected_lock_entry is not None:
        lock_values = {
            **raw, "metadata_payload_sha256": stored,
            "snapshot_checksum": snapshot_checksum,
            "source_parent_target_indices_sha256": parent_sha,
        }
        if any(expected_lock_entry.get(name) != value for name, value in lock_values.items()):
            raise ValueError("v2 snapshot lock payload mismatch")
    result.update(
        source=source, target=target, reference=reference,
        parent_indices=parent_indices, snapshot_checksum=snapshot_checksum, **raw,
    )
    return result


def _lock_entry(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    metadata = snapshot["metadata"]
    return {
        "condition": metadata["condition"],
        "confirmatory_rng_instantiation_count": metadata["confirmatory_rng_instantiation_count"],
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
        "source_parent_target_indices_sha256": metadata[
            "source_parent_target_indices_sha256"
        ],
        "target_checksum": snapshot["target_checksum"],
    }


def build_v2_snapshot_lock(
    cache_root: str | Path, plans: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    entries = [_lock_entry(read_v2_snapshot(cache_root, plan)) for plan in plans]
    core = {
        "condition_snapshot_counts": dict(sorted(Counter(row["condition"] for row in entries).items())),
        "confirmatory_rng_instantiation_count": sum(
            int(row["confirmatory_rng_instantiation_count"]) for row in entries
        ),
        "lineage_schema_version": LINEAGE_SCHEMA,
        "planned_snapshot_count": len(entries),
        "schema_version": SNAPSHOT_LOCK_SCHEMA,
        "snapshot_builder_contract_version": SNAPSHOT_BUILDER_CONTRACT_VERSION,
        "snapshot_schema_version": SNAPSHOT_SCHEMA,
        "snapshots": entries,
    }
    return {**core, "snapshot_lock_payload_sha256": canonical_json_sha256(core)}


def validate_v2_snapshot_lock(
    lock_path: str | Path, cache_root: str | Path,
    plans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = _strict_object(Path(lock_path), "v2 snapshot lock")
    unsigned = {key: item for key, item in value.items() if key != "snapshot_lock_payload_sha256"}
    entries = value.get("snapshots")
    expected_ids = [validate_snapshot_plan_row(row)["planned_snapshot_id"] for row in plans]
    if (
        value.get("snapshot_lock_payload_sha256") != canonical_json_sha256(unsigned)
        or value.get("schema_version") != SNAPSHOT_LOCK_SCHEMA
        or value.get("snapshot_schema_version") != SNAPSHOT_SCHEMA
        or value.get("lineage_schema_version") != LINEAGE_SCHEMA
        or value.get("planned_snapshot_count") != SNAPSHOT_COUNT
        or value.get("condition_snapshot_counts")
        != {"FULL_NOISE": 525, "IDEAL_MATCHED": 35, "INDEPENDENT_NOISE_FREE": 35}
        or value.get("confirmatory_rng_instantiation_count") != 1575
        or type(entries) is not list or len(entries) != SNAPSHOT_COUNT
        or [row.get("snapshot_id") for row in entries] != expected_ids
    ):
        raise ValueError("v2 snapshot lock identity or inventory mismatch")
    for plan, entry in zip(plans, entries):
        read_v2_snapshot(cache_root, plan, expected_lock_entry=entry)
    return value


def prepare_v2_snapshots(
    root: str | Path, cache_root: str | Path, plans: Sequence[Mapping[str, Any]],
    *, lock_path: str | Path, resume: bool,
) -> dict[str, Any]:
    if not resume or len(plans) != SNAPSHOT_COUNT:
        raise PermissionError("v2 formal snapshot preparation requires exact --resume plan")
    cache = Path(cache_root).resolve()
    generated = resumed = rng_count = 0
    for plan in plans:
        row = validate_snapshot_plan_row(plan)
        directory = (cache / row["planned_snapshot_id"]).resolve()
        if directory.exists():
            read_v2_snapshot(cache, row)
            resumed += 1
        else:
            value = build_v2_snapshot(root, row)
            rng_count += int(value["metadata"]["confirmatory_rng_instantiation_count"])
            write_v2_snapshot_atomic(cache, value)
            read_v2_snapshot(cache, row)
            generated += 1
    lock_file = Path(lock_path).resolve()
    if lock_file.exists():
        lock = validate_v2_snapshot_lock(lock_file, cache, plans)
    else:
        from .phase_a_trial_result_schema import canonical_json_bytes
        from .phase_a_trial_result_writer import atomic_write_bytes

        lock = build_v2_snapshot_lock(cache, plans)
        atomic_write_bytes(lock_file, canonical_json_bytes(lock), replace=False)
    return {
        "confirmatory_rng_instantiation_count_this_invocation": rng_count,
        "generated_snapshot_count": generated,
        "lock": lock,
        "lock_path": lock_file,
        "resumed_snapshot_count": resumed,
    }


__all__ = [
    "BASE_ARRAY_FILENAMES", "ConfirmatoryV2SeedFirewall",
    "DEVELOPMENT_GEOMETRY_SEEDS", "DEVELOPMENT_MEASUREMENT_SEEDS",
    "GeometryOnlyQualificationFirewall", "METADATA_FIELDS",
    "PARENT_INDEX_FILENAME", "SNAPSHOT_BUILDER_CONTRACT_VERSION",
    "build_development_nonideal_regression_snapshot",
    "build_development_ideal_qualification_snapshot",
    "build_development_independent_qualification_snapshot", "build_v2_snapshot",
    "build_v2_snapshot_lock", "prepare_v2_snapshots", "read_v2_snapshot",
    "validate_v2_snapshot_lock", "write_v2_snapshot_atomic",
]
