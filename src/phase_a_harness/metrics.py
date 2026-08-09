"""Minimal pose helpers required by the byte-exact frozen backend adapters."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose: Any) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("pose must be a finite 4x4 matrix")
    return np.array(value, dtype=np.float64, order="C", copy=True)


def zero_initialization_error(reference_pose: Any, estimated_pose: Any) -> dict[str, float]:
    reference = pose_matrix(reference_pose)
    estimated = pose_matrix(estimated_pose)
    delta = np.linalg.inv(reference) @ estimated
    rho = delta[:3, 3]
    phi = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
    return {
        "translation_x_m": float(rho[0]),
        "translation_y_m": float(rho[1]),
        "translation_z_m": float(rho[2]),
        "rotation_x_rad": float(phi[0]),
        "rotation_y_rad": float(phi[1]),
        "rotation_z_rad": float(phi[2]),
        "translation_error_m": float(np.linalg.norm(rho)),
        "rotation_error_rad": float(np.linalg.norm(phi)),
        "rotation_error_deg": float(math.degrees(np.linalg.norm(phi))),
    }


__all__ = ["pose_matrix", "zero_initialization_error"]

