"""Build each frozen Development snapshot exactly once for all backends."""

from __future__ import annotations

from typing import Any

import numpy as np

from capture_range.day2_development_protocol import canonical_array_sha256, canonical_sha256
from capture_range.day2_development_scene import build_scene_geometry

from .protocol import DevelopmentSeedFirewall, ZeroPerturbationProtocol
from .types import SnapshotBundle, SnapshotKey


def _plain(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def build_snapshot(
    protocol: ZeroPerturbationProtocol,
    firewall: DevelopmentSeedFirewall,
    key: SnapshotKey,
) -> SnapshotBundle:
    firewall.assert_access(key.geometry_seed, key.measurement_seed, key.repeat_index)
    if key.scene_variant not in protocol.scenes:
        raise ValueError("unknown Development scene")
    condition = protocol.condition(key.noise_condition)
    scene = protocol.section("scene_generation")
    common = scene["common"]
    reference_translation = np.asarray(
        common["reference_pose_translation_world"], dtype=np.float64
    )
    reference_pose = np.asarray(
        [0.0, *reference_translation.tolist(), 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    map_geometry = build_scene_geometry(
        protocol,
        firewall,
        key.scene_variant,
        key.geometry_seed,
        key.measurement_seed,
        key.repeat_index,
        "map",
    )
    if bool(condition["scan_is_deterministic_map_subset"]):
        ranges = np.linalg.norm(map_geometry.points_world - reference_translation, axis=1)
        keep = (ranges >= 0.30) & (ranges <= 20.0)
        scan_world = np.asarray(map_geometry.points_world[keep], dtype=np.float64)
        scan_normal_checksum = canonical_array_sha256(map_geometry.normals_world[keep])
        primitive_ids = tuple(
            primitive for primitive, selected in zip(map_geometry.primitive_ids, keep) if selected
        )
        scan_geometry_checksum = canonical_array_sha256(scan_world)
    else:
        scan_geometry = build_scene_geometry(
            protocol,
            firewall,
            key.scene_variant,
            key.geometry_seed,
            key.measurement_seed,
            key.repeat_index,
            "scan",
        )
        scan_world = np.asarray(scan_geometry.points_world, dtype=np.float64)
        scan_normal_checksum = scan_geometry.normal_checksum
        primitive_ids = scan_geometry.primitive_ids
        scan_geometry_checksum = scan_geometry.point_checksum

    scan_count_before = int(scan_world.shape[0])
    map_count_before = int(map_geometry.points_world.shape[0])
    scan_dropout = float(condition["scan_dropout_fraction"])
    map_dropout = float(condition["map_dropout_fraction"])
    if scan_dropout > 0.0:
        scan_mask = firewall.rng(
            key.scene_variant,
            key.geometry_seed,
            key.measurement_seed,
            key.repeat_index,
            key.noise_condition,
            "scan_dropout",
        ).random(scan_count_before) >= scan_dropout
    else:
        scan_mask = np.ones(scan_count_before, dtype=bool)
    if map_dropout > 0.0:
        map_mask = firewall.rng(
            key.scene_variant,
            key.geometry_seed,
            key.measurement_seed,
            key.repeat_index,
            key.noise_condition,
            "map_dropout",
        ).random(map_count_before) >= map_dropout
    else:
        map_mask = np.ones(map_count_before, dtype=bool)

    scan_sigma = float(condition["scan_noise_sigma_m"])
    map_sigma = float(condition["map_noise_sigma_m"])
    if scan_sigma > 0.0:
        scan_noise = firewall.rng(
            key.scene_variant,
            key.geometry_seed,
            key.measurement_seed,
            key.repeat_index,
            key.noise_condition,
            "scan_noise",
        ).normal(0.0, scan_sigma, size=scan_world.shape)
    else:
        scan_noise = np.zeros_like(scan_world)
    if map_sigma > 0.0:
        map_noise = firewall.rng(
            key.scene_variant,
            key.geometry_seed,
            key.measurement_seed,
            key.repeat_index,
            key.noise_condition,
            "map_noise",
        ).normal(0.0, map_sigma, size=map_geometry.points_world.shape)
    else:
        map_noise = np.zeros_like(map_geometry.points_world)

    scan_sensor = (scan_world - reference_translation + scan_noise)[scan_mask]
    noisy_map = (np.asarray(map_geometry.points_world) + map_noise)[map_mask]
    scan_checksum = canonical_array_sha256(scan_sensor)
    map_checksum = canonical_array_sha256(noisy_map)
    scene_checksum = canonical_sha256(
        {
            "scene_variant": key.scene_variant,
            "map_points": map_geometry.point_checksum,
            "map_normals": map_geometry.normal_checksum,
            "scan_points": scan_geometry_checksum,
            "scan_normals": scan_normal_checksum,
        }
    )
    noise_checksum = canonical_sha256(
        {
            "scan_noise": canonical_array_sha256(scan_noise),
            "map_noise": canonical_array_sha256(map_noise),
        }
    )
    dropout_checksum = canonical_sha256(
        {
            "scan_dropout_mask": canonical_array_sha256(scan_mask),
            "map_dropout_mask": canonical_array_sha256(map_mask),
        }
    )
    reference_checksum = canonical_array_sha256(reference_pose)
    backend_input_checksum = canonical_sha256(
        {
            "scan_checksum": scan_checksum,
            "map_checksum": map_checksum,
            "reference_pose_checksum": reference_checksum,
        }
    )
    registration_config = {"registration": _plain(protocol.section("native_registration"))}
    for key_to_remove in (
        "full_reassociation_rebuilds_each_nonlinear_evaluation",
        "frozen_jacobian_prepared_once_per_snapshot",
        "optimizer_inputs_exclude_scene_label_and_gt_weak_direction",
    ):
        registration_config["registration"].pop(key_to_remove, None)
    return SnapshotBundle(
        key=key,
        scan_points=scan_sensor,
        map_points=noisy_map,
        reference_pose=reference_pose,
        registration_config=registration_config,
        checksums={
            "scene_checksum": scene_checksum,
            "scan_checksum": scan_checksum,
            "map_checksum": map_checksum,
            "noise_checksum": noise_checksum,
            "dropout_checksum": dropout_checksum,
            "reference_pose_checksum": reference_checksum,
            "backend_input_checksum": backend_input_checksum,
        },
        scan_point_count_before_dropout=scan_count_before,
        map_point_count_before_dropout=map_count_before,
        primitive_count=len(set(map_geometry.primitive_ids) | set(primitive_ids)),
    )


__all__ = ["build_snapshot"]
