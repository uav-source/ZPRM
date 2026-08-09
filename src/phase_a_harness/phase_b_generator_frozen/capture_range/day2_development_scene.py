"""Deterministic seven-scene generator for Day 2 Development only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .day2_development_protocol import (
    Day2DevelopmentProtocol,
    DevelopmentSeedFirewall,
    canonical_array_sha256,
    canonical_sha256,
    canonical_unit_interval,
)
from .types import RegistrationSnapshot


@dataclass(frozen=True)
class SceneGeometry:
    scene_variant: str
    geometry_seed: int
    role: str
    points_world: np.ndarray
    normals_world: np.ndarray
    point_checksum: str
    normal_checksum: str
    primitive_ids: tuple[str, ...]


@dataclass(frozen=True)
class DevelopmentSnapshotBundle:
    snapshot: RegistrationSnapshot
    scene_variant: str
    geometry_seed: int
    measurement_seed: int
    repeat_index: int
    block_id: str
    base_snapshot_id: str
    base_snapshot_checksum: str
    scan_checksum: str
    map_checksum: str
    dropout_checksum: str
    scan_pre_noise_checksum: str
    map_pre_noise_checksum: str
    scan_normals_checksum: str
    map_normals_checksum: str
    scan_dropout_mask_checksum: str
    map_dropout_mask_checksum: str
    scan_point_count_before_dropout: int
    scan_point_count: int
    map_point_count_before_dropout: int
    map_point_count: int
    primitive_count: int


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _readonly(value: Any, dtype=np.float64) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _quantize(values: np.ndarray, resolution: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / float(resolution)
    result = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return result.astype(np.int64)


def _grid(
    lower: float,
    upper: float,
    spacing: float,
    phase: float,
    tolerance: float,
) -> np.ndarray:
    origin = float(lower) + float(phase)
    first = int(math.ceil((float(lower) - tolerance - origin) / spacing))
    last = int(math.floor((float(upper) + tolerance - origin) / spacing))
    if last < first:
        return np.empty(0, dtype=np.float64)
    values = origin + np.arange(first, last + 1, dtype=np.float64) * spacing
    return values[(values >= lower - tolerance) & (values <= upper + tolerance)]


def _phase(
    global_seed: int,
    geometry_seed: int,
    scene_variant: str,
    primitive_id: str,
    role: str,
    axis_name: str,
    spacing: float,
) -> float:
    return spacing * canonical_unit_interval(
        {
            "global_seed": int(global_seed),
            "geometry_seed": int(geometry_seed),
            "scene_variant": str(scene_variant),
            "primitive_id": str(primitive_id),
            "map_or_scan": str(role),
            "surface_axis_name": str(axis_name),
        }
    )


def _face(
    *,
    global_seed: int,
    geometry_seed: int,
    scene_variant: str,
    role: str,
    primitive_id: str,
    fixed_axis: int,
    fixed_value: float,
    first_axis: int,
    first_bounds: Sequence[float],
    second_axis: int,
    second_bounds: Sequence[float],
    normal: Sequence[float],
    spacing: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    axis_names = ("x", "y", "z")
    first = _grid(
        float(first_bounds[0]),
        float(first_bounds[1]),
        spacing,
        _phase(global_seed, geometry_seed, scene_variant, primitive_id, role, axis_names[first_axis], spacing),
        tolerance,
    )
    second = _grid(
        float(second_bounds[0]),
        float(second_bounds[1]),
        spacing,
        _phase(global_seed, geometry_seed, scene_variant, primitive_id, role, axis_names[second_axis], spacing),
        tolerance,
    )
    if first.size == 0 or second.size == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            [],
            np.empty(0, dtype=np.int64),
        )
    a, b = np.meshgrid(first, second, indexing="ij")
    points = np.empty((a.size, 3), dtype=np.float64)
    points[:, fixed_axis] = fixed_value
    points[:, first_axis] = a.ravel()
    points[:, second_axis] = b.ravel()
    normals = np.tile(np.asarray(normal, dtype=np.float64), (points.shape[0], 1))
    return points, normals, [primitive_id] * points.shape[0], np.arange(points.shape[0], dtype=np.int64)


def _box_faces(
    *,
    global_seed: int,
    geometry_seed: int,
    scene_variant: str,
    role: str,
    primitive_id: str,
    center: Sequence[float],
    size: Sequence[float],
    spacing: float,
    tolerance: float,
) -> list[tuple[np.ndarray, np.ndarray, list[str], np.ndarray]]:
    center_v = np.asarray(center, dtype=np.float64)
    half = 0.5 * np.asarray(size, dtype=np.float64)
    lo, hi = center_v - half, center_v + half
    specs = (
        ("neg_x", 0, lo[0], 1, (lo[1], hi[1]), 2, (lo[2], hi[2]), (-1.0, 0.0, 0.0)),
        ("pos_x", 0, hi[0], 1, (lo[1], hi[1]), 2, (lo[2], hi[2]), (1.0, 0.0, 0.0)),
        ("neg_y", 1, lo[1], 0, (lo[0], hi[0]), 2, (lo[2], hi[2]), (0.0, -1.0, 0.0)),
        ("pos_y", 1, hi[1], 0, (lo[0], hi[0]), 2, (lo[2], hi[2]), (0.0, 1.0, 0.0)),
        ("bottom", 2, lo[2], 0, (lo[0], hi[0]), 1, (lo[1], hi[1]), (0.0, 0.0, -1.0)),
        ("top", 2, hi[2], 0, (lo[0], hi[0]), 1, (lo[1], hi[1]), (0.0, 0.0, 1.0)),
    )
    return [
        _face(
            global_seed=global_seed,
            geometry_seed=geometry_seed,
            scene_variant=scene_variant,
            role=role,
            primitive_id=f"{primitive_id}::{token}",
            fixed_axis=fixed_axis,
            fixed_value=float(value),
            first_axis=first_axis,
            first_bounds=first_bounds,
            second_axis=second_axis,
            second_bounds=second_bounds,
            normal=normal,
            spacing=spacing,
            tolerance=tolerance,
        )
        for token, fixed_axis, value, first_axis, first_bounds, second_axis, second_bounds, normal in specs
    ]


def _boundary_faces(
    bounds: np.ndarray,
    tokens: Sequence[str],
    **common: Any,
) -> list[tuple[np.ndarray, np.ndarray, list[str], np.ndarray]]:
    x, y, z = bounds
    definitions = {
        "floor": (2, z[0], 0, x, 1, y, (0.0, 0.0, 1.0)),
        "ceiling": (2, z[1], 0, x, 1, y, (0.0, 0.0, -1.0)),
        "wall_neg_x": (0, x[0], 1, y, 2, z, (1.0, 0.0, 0.0)),
        "wall_pos_x": (0, x[1], 1, y, 2, z, (-1.0, 0.0, 0.0)),
        "wall_neg_y": (1, y[0], 0, x, 2, z, (0.0, 1.0, 0.0)),
        "wall_pos_y": (1, y[1], 0, x, 2, z, (0.0, -1.0, 0.0)),
    }
    rows = []
    for token in tokens:
        fixed_axis, value, first_axis, first_bounds, second_axis, second_bounds, normal = definitions[token]
        rows.append(
            _face(
                primitive_id=token,
                fixed_axis=fixed_axis,
                fixed_value=float(value),
                first_axis=first_axis,
                first_bounds=first_bounds,
                second_axis=second_axis,
                second_bounds=second_bounds,
                normal=normal,
                **common,
            )
        )
    return rows


def _deduplicate_sort(
    points: np.ndarray,
    normals: np.ndarray,
    primitive_ids: Sequence[str],
    source_indices: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    qp = _quantize(points, resolution)
    qn = _quantize(normals, resolution)
    winners: dict[tuple[int, ...], tuple[str, int, int]] = {}
    for index in range(points.shape[0]):
        key = tuple(int(value) for value in np.concatenate((qp[index], qn[index])))
        candidate = (str(primitive_ids[index]), int(source_indices[index]), index)
        if key not in winners or candidate[:2] < winners[key][:2]:
            winners[key] = candidate
    ordered = sorted(
        winners.items(), key=lambda item: (*item[0], item[1][0], item[1][1])
    )
    indices = np.asarray([item[1][2] for item in ordered], dtype=np.int64)
    return points[indices], normals[indices], tuple(str(primitive_ids[i]) for i in indices)


def build_scene_geometry(
    protocol: Day2DevelopmentProtocol,
    firewall: DevelopmentSeedFirewall,
    scene_variant: str,
    geometry_seed: int,
    measurement_seed: int,
    repeat_index: int,
    role: str,
) -> SceneGeometry:
    """Build one noiseless map or scan geometry after the seed firewall."""

    firewall.assert_access(geometry_seed, measurement_seed, repeat_index)
    scene = protocol.section("scene_generation")
    variant = str(scene_variant)
    if variant not in tuple(scene["variants_in_order"]):
        raise ValueError(f"unknown Development scene variant: {variant}")
    if role not in {"map", "scan"}:
        raise ValueError("role must be map or scan")
    common_config = scene["common"]
    spacing = float(common_config[f"{role}_point_spacing_m"])
    tolerance = float(scene["planar_grid"]["bound_tolerance_m"])
    global_seed = int(firewall.global_seed)

    if variant == "GEOMETRY_RICH_ROOM":
        spec = scene["geometry_rich_room"]
        bounds = np.asarray(spec["bounds_xyz_m"], dtype=np.float64)
        parts = _boundary_faces(
            bounds,
            spec["primitive_order"][:6],
            global_seed=global_seed,
            geometry_seed=geometry_seed,
            scene_variant=variant,
            role=role,
            spacing=spacing,
            tolerance=tolerance,
        )
        for cuboid in spec["asymmetric_cuboids"]:
            parts.extend(
                _box_faces(
                    global_seed=global_seed,
                    geometry_seed=geometry_seed,
                    scene_variant=variant,
                    role=role,
                    primitive_id=str(cuboid["primitive_id"]),
                    center=cuboid["center"],
                    size=cuboid["size"],
                    spacing=spacing,
                    tolerance=tolerance,
                )
            )
    elif variant == "LONG_CORRIDOR":
        spec = scene["long_corridor"]
        bounds = np.asarray(spec["bounds_xyz_m"], dtype=np.float64)
        parts = _boundary_faces(bounds, spec["primitive_order"], global_seed=global_seed, geometry_seed=geometry_seed, scene_variant=variant, role=role, spacing=spacing, tolerance=tolerance)
    elif variant == "PARALLEL_WALLS":
        spec = scene["parallel_walls"]
        bounds = np.asarray(spec["bounds_xyz_m"], dtype=np.float64)
        parts = _boundary_faces(bounds, spec["primitive_order"], global_seed=global_seed, geometry_seed=geometry_seed, scene_variant=variant, role=role, spacing=spacing, tolerance=tolerance)
    else:
        key = "repeated_structure" if variant == "REPEATED_STRUCTURE" else "end_face_transition"
        spec = scene[key]
        bounds = np.asarray(spec["bounds_xyz_m"], dtype=np.float64)
        parts = _boundary_faces(bounds, spec["base_primitive_order"], global_seed=global_seed, geometry_seed=geometry_seed, scene_variant=variant, role=role, spacing=spacing, tolerance=tolerance)
        if variant.startswith("END_FACE_TRANSITION_"):
            face = spec["end_face"]
            end_part = _face(global_seed=global_seed, geometry_seed=geometry_seed, scene_variant=variant, role=role, primitive_id=str(face["primitive_id"]), fixed_axis=0, fixed_value=float(face["x_m"]), first_axis=1, first_bounds=bounds[1], second_axis=2, second_bounds=bounds[2], normal=face["normal"], spacing=spacing, tolerance=tolerance)
            fraction = float(spec["retention_fraction"][variant])
            if 0.0 < fraction < 1.0 and end_part[0].shape[0]:
                q = _quantize(end_part[0], float(scene["quantization"]["resolution_m"]))
                keep = np.asarray([
                    canonical_unit_interval({"global_seed": global_seed, "geometry_seed": int(geometry_seed), "scene_variant": variant, "primitive_id": str(face["primitive_id"]), "quantized_position": [int(v) for v in row]}) < fraction
                    for row in q
                ], dtype=bool)
                end_part = (end_part[0][keep], end_part[1][keep], [p for p, flag in zip(end_part[2], keep) if flag], end_part[3][keep])
            elif fraction <= 0.0:
                end_part = (np.empty((0, 3)), np.empty((0, 3)), [], np.empty(0, dtype=np.int64))
            parts.append(end_part)
        else:
            period = float(spec["period_x_m"])
            phase = period * canonical_unit_interval({"global_seed": global_seed, "geometry_seed": int(geometry_seed), "scene_variant": variant, "repeated_rib_longitudinal_phase": True})
            x_min, x_max = bounds[0]
            first = int(math.ceil((x_min - (x_min + phase)) / period))
            last = int(math.floor((x_max - (x_min + phase)) / period))
            rib_index = 0
            for n in range(first, last + 1):
                x = float(x_min + phase + n * period)
                for side in (-1, 1):
                    y = float(side * (abs(bounds[1, 1]) - float(spec["depth_y_m"]) / 2.0))
                    parts.extend(_box_faces(global_seed=global_seed, geometry_seed=geometry_seed, scene_variant=variant, role=role, primitive_id=f"rib_{rib_index:03d}_{'pos' if side > 0 else 'neg'}_y", center=(x, y, float(spec["height_z_m"]) / 2.0), size=(float(spec["thickness_x_m"]), float(spec["depth_y_m"]), float(spec["height_z_m"])), spacing=spacing, tolerance=tolerance))
                rib_index += 1

    points = np.vstack([part[0] for part in parts if part[0].size])
    normals = np.vstack([part[1] for part in parts if part[1].size])
    primitive_ids = [value for part in parts for value in part[2]]
    source_indices = np.concatenate([part[3] for part in parts if part[3].size])
    reference = np.asarray(common_config["reference_pose_translation_world"], dtype=np.float64)
    maximum = float(common_config["maximum_map_range_from_reference_m"] if role == "map" else common_config["maximum_scan_range_m"])
    minimum = 0.0 if role == "map" else float(common_config["minimum_scan_range_m"])
    ranges = np.linalg.norm(points - reference, axis=1)
    keep = (ranges >= minimum) & (ranges <= maximum)
    points, normals = points[keep], normals[keep]
    primitive_ids = [p for p, flag in zip(primitive_ids, keep) if flag]
    source_indices = source_indices[keep]
    points, normals, primitive_tuple = _deduplicate_sort(points, normals, primitive_ids, source_indices, float(scene["quantization"]["resolution_m"]))
    return SceneGeometry(variant, int(geometry_seed), role, _readonly(points), _readonly(normals), canonical_array_sha256(points), canonical_array_sha256(normals), primitive_tuple)


def build_development_base_snapshot(
    protocol: Day2DevelopmentProtocol,
    firewall: DevelopmentSeedFirewall,
    scene_variant: str,
    geometry_seed: int,
    measurement_seed: int,
    repeat_index: int,
) -> DevelopmentSnapshotBundle:
    identity = firewall.assert_access(geometry_seed, measurement_seed, repeat_index)
    map_geometry = build_scene_geometry(protocol, firewall, scene_variant, geometry_seed, measurement_seed, repeat_index, "map")
    scan_geometry = build_scene_geometry(protocol, firewall, scene_variant, geometry_seed, measurement_seed, repeat_index, "scan")
    measurement = protocol.section("measurement_realization")
    scan_mask = firewall.rng(scene_variant, geometry_seed, measurement_seed, repeat_index, "scan_dropout").random(scan_geometry.points_world.shape[0]) >= float(measurement["scan_dropout_fraction"])
    map_mask = firewall.rng(scene_variant, geometry_seed, measurement_seed, repeat_index, "map_dropout").random(map_geometry.points_world.shape[0]) >= float(measurement["map_dropout_fraction"])
    scan_noise = firewall.rng(scene_variant, geometry_seed, measurement_seed, repeat_index, "scan_noise").normal(0.0, float(measurement["scan_point_gaussian_sigma_m"]), size=scan_geometry.points_world.shape)
    map_noise = firewall.rng(scene_variant, geometry_seed, measurement_seed, repeat_index, "map_noise").normal(0.0, float(measurement["map_point_gaussian_sigma_m"]), size=map_geometry.points_world.shape)
    reference_translation = np.asarray(protocol.section("scene_generation")["common"]["reference_pose_translation_world"], dtype=np.float64)
    scan_sensor = (scan_geometry.points_world - reference_translation + scan_noise)[scan_mask]
    noisy_map = (map_geometry.points_world + map_noise)[map_mask]
    scan_checksum = canonical_array_sha256(scan_sensor)
    map_checksum = canonical_array_sha256(noisy_map)
    scan_mask_checksum = canonical_array_sha256(scan_mask)
    map_mask_checksum = canonical_array_sha256(map_mask)
    dropout_checksum = canonical_sha256({"scan_dropout_mask": scan_mask_checksum, "map_dropout_mask": map_mask_checksum})
    base_snapshot_id = f"dev::{scene_variant}::g{identity.geometry_seed}::m{identity.measurement_seed}::r{identity.repeat_index:02d}"
    block_id = f"dev::{scene_variant}::g{identity.geometry_seed}::m{identity.measurement_seed}"
    reference_pose = np.array([0.0, *reference_translation.tolist(), 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    registration_config = {"registration": _plain(protocol.section("registration")), "success": _plain(protocol.section("success"))}
    for internal in ("backend", "correspondence_tie_breaking", "full_reassociation_rebuilds_each_nonlinear_evaluation", "fixed_local_map_kdtree_reuse_within_base_snapshot_allowed", "frozen_jacobian_prepared_once_per_base_snapshot", "optimizer_inputs_exclude_direction_roles_and_theoretical_weak_direction", "execution_matrix", "expected_trial_rows"):
        registration_config["registration"].pop(internal, None)
    base_checksum = canonical_sha256({"base_snapshot_id": base_snapshot_id, "scan_checksum": scan_checksum, "map_checksum": map_checksum, "dropout_checksum": dropout_checksum, "reference_pose": reference_pose, "registration_configuration": registration_config})
    snapshot = RegistrationSnapshot(snapshot_id=base_snapshot_id, scan_points=scan_sensor, local_map_points=noisy_map, reference_pose=reference_pose, registration_config=registration_config, metadata={"scene_variant": str(scene_variant), "geometry_seed": int(geometry_seed), "measurement_seed": int(measurement_seed), "repeat_index": int(repeat_index), "base_snapshot_checksum": base_checksum})
    return DevelopmentSnapshotBundle(snapshot=snapshot, scene_variant=str(scene_variant), geometry_seed=int(geometry_seed), measurement_seed=int(measurement_seed), repeat_index=int(repeat_index), block_id=block_id, base_snapshot_id=base_snapshot_id, base_snapshot_checksum=base_checksum, scan_checksum=scan_checksum, map_checksum=map_checksum, dropout_checksum=dropout_checksum, scan_pre_noise_checksum=scan_geometry.point_checksum, map_pre_noise_checksum=map_geometry.point_checksum, scan_normals_checksum=scan_geometry.normal_checksum, map_normals_checksum=map_geometry.normal_checksum, scan_dropout_mask_checksum=scan_mask_checksum, map_dropout_mask_checksum=map_mask_checksum, scan_point_count_before_dropout=int(scan_geometry.points_world.shape[0]), scan_point_count=int(scan_sensor.shape[0]), map_point_count_before_dropout=int(map_geometry.points_world.shape[0]), map_point_count=int(noisy_map.shape[0]), primitive_count=len(set(scan_geometry.primitive_ids) | set(map_geometry.primitive_ids)))


__all__ = ["DevelopmentSnapshotBundle", "SceneGeometry", "build_development_base_snapshot", "build_scene_geometry"]
