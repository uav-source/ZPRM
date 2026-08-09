"""Typed records for zero-perturbation Development measurements."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _readonly_array(value: Any, *, shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if shape_tail and (array.ndim < len(shape_tail) or array.shape[-len(shape_tail) :] != shape_tail):
        raise ValueError(f"array must end with shape {shape_tail}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("array contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, order=True)
class SnapshotKey:
    scene_variant: str
    geometry_seed: int
    measurement_seed: int
    repeat_index: int
    noise_condition: str

    def __post_init__(self) -> None:
        if not self.scene_variant or not self.noise_condition:
            raise ValueError("snapshot key labels must be non-empty")
        if self.geometry_seed < 0 or self.measurement_seed < 0:
            raise ValueError("snapshot seeds must be non-negative")
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")

    @property
    def snapshot_id(self) -> str:
        return (
            f"dev::{self.scene_variant}::g{self.geometry_seed}::"
            f"m{self.measurement_seed}::r{self.repeat_index:02d}::"
            f"{self.noise_condition}"
        )


@dataclass(frozen=True, eq=False)
class SnapshotBundle:
    key: SnapshotKey
    scan_points: np.ndarray
    map_points: np.ndarray
    reference_pose: np.ndarray
    registration_config: Mapping[str, Any]
    checksums: Mapping[str, str]
    scan_point_count_before_dropout: int
    map_point_count_before_dropout: int
    primitive_count: int

    def __post_init__(self) -> None:
        scan = _readonly_array(self.scan_points, shape_tail=(3,))
        local_map = _readonly_array(self.map_points, shape_tail=(3,))
        pose = _readonly_array(self.reference_pose)
        if scan.ndim != 2 or local_map.ndim != 2:
            raise ValueError("scan_points and map_points must be Nx3")
        if scan.shape[0] == 0 or local_map.shape[0] == 0:
            raise ValueError("snapshot point clouds must be non-empty")
        if pose.shape not in {(8,), (4, 4)}:
            raise ValueError("reference_pose must be a TUM row or 4x4 matrix")
        required = {
            "scene_checksum",
            "scan_checksum",
            "map_checksum",
            "noise_checksum",
            "dropout_checksum",
            "reference_pose_checksum",
            "backend_input_checksum",
        }
        if set(self.checksums) != required:
            raise ValueError("snapshot checksum contract changed")
        object.__setattr__(self, "scan_points", scan)
        object.__setattr__(self, "map_points", local_map)
        object.__setattr__(self, "reference_pose", pose)
        object.__setattr__(self, "registration_config", MappingProxyType(dict(self.registration_config)))
        object.__setattr__(self, "checksums", MappingProxyType(dict(self.checksums)))

    @property
    def snapshot_id(self) -> str:
        return self.key.snapshot_id

    def inventory_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "snapshot_id": self.snapshot_id,
            "scene_variant": self.key.scene_variant,
            "geometry_seed": self.key.geometry_seed,
            "measurement_seed": self.key.measurement_seed,
            "repeat_index": self.key.repeat_index,
            "noise_condition": self.key.noise_condition,
            "scan_point_count_before_dropout": self.scan_point_count_before_dropout,
            "scan_point_count": int(self.scan_points.shape[0]),
            "map_point_count_before_dropout": self.map_point_count_before_dropout,
            "map_point_count": int(self.map_points.shape[0]),
            "primitive_count": self.primitive_count,
        }
        row.update(self.checksums)
        return row


@dataclass(frozen=True)
class BackendResult:
    backend: str
    final_pose: np.ndarray
    runtime_ms: float
    solver_converged: bool
    finite_result: bool
    iteration_count: int
    correspondence_count: int
    initial_cost: float | None
    final_cost: float | None
    termination_reason: str
    failure_reason: str
    extra: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.backend not in {"native_full", "native_frozen", "open3d_full"}:
            raise ValueError(f"unknown backend: {self.backend}")
        pose = _readonly_array(self.final_pose)
        if pose.shape not in {(8,), (4, 4)}:
            raise ValueError("final_pose must be a TUM row or 4x4 matrix")
        object.__setattr__(self, "final_pose", pose)
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    @property
    def solver_failed(self) -> bool:
        return not (self.solver_converged and self.finite_result)


__all__ = ["BackendResult", "SnapshotBundle", "SnapshotKey"]
