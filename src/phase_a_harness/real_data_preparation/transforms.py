"""Independent-reference transform, interpolation, and deskew helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform_from_xyzw(translation: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("translation must be (3,), quaternion must be xyzw (4,)")
    if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
        raise ValueError("transform inputs must be finite")
    if not np.isclose(np.linalg.norm(quaternion), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("quaternion must be unit xyzw")
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    value[:3, 3] = translation
    return value


def compose_world_sensor(t_world_base: np.ndarray, t_base_sensor: np.ndarray) -> np.ndarray:
    left = np.asarray(t_world_base, dtype=np.float64)
    right = np.asarray(t_base_sensor, dtype=np.float64)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise ValueError("transforms must be 4x4")
    value = left @ right
    if not np.isfinite(value).all() or not np.allclose(value[3], [0, 0, 0, 1]):
        raise ValueError("invalid homogeneous transform")
    return value


def interpolate_pose(
    timestamps: np.ndarray,
    translations: np.ndarray,
    quaternions_xyzw: np.ndarray,
    query_timestamp: float,
    *,
    max_gap_s: float,
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    quaternions = np.asarray(quaternions_xyzw, dtype=np.float64)
    if timestamps.ndim != 1 or translations.shape != (timestamps.size, 3) or quaternions.shape != (timestamps.size, 4):
        raise ValueError("reference arrays have incompatible shapes")
    if timestamps.size < 2 or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("reference timestamps must be strictly increasing")
    index = int(np.searchsorted(timestamps, query_timestamp, side="right"))
    if index == 0 or index == timestamps.size:
        raise ValueError("query timestamp is outside reference coverage")
    lower, upper = index - 1, index
    gap = float(timestamps[upper] - timestamps[lower])
    if gap > max_gap_s:
        raise ValueError("reference interpolation gap exceeds frozen maximum")
    weight = float((query_timestamp - timestamps[lower]) / gap)
    translation = (1.0 - weight) * translations[lower] + weight * translations[upper]
    rotation = Slerp(timestamps[[lower, upper]], Rotation.from_quat(quaternions[[lower, upper]]))(
        [query_timestamp]
    ).as_quat()[0]
    return transform_from_xyzw(translation, rotation)


def deskew_with_independent_reference(
    points: np.ndarray,
    point_timestamps: np.ndarray,
    reference: Mapping[str, Any],
    scan_reference_timestamp: float,
    *,
    max_gap_s: float,
) -> np.ndarray:
    if reference.get("lineage") != "INDEPENDENT_NON_LIDAR_6DOF_REFERENCE":
        raise PermissionError("deskew reference is not proven independent of LiDAR")
    points = np.asarray(points, dtype=np.float64)
    point_timestamps = np.asarray(point_timestamps, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or point_timestamps.shape != (points.shape[0],):
        raise ValueError("points/timestamps shape mismatch")
    timestamps = np.asarray(reference["timestamps"], dtype=np.float64)
    translations = np.asarray(reference["translations"], dtype=np.float64)
    quaternions = np.asarray(reference["quaternions_xyzw"], dtype=np.float64)
    anchor = interpolate_pose(
        timestamps, translations, quaternions, scan_reference_timestamp, max_gap_s=max_gap_s
    )
    anchor_inverse = np.linalg.inv(anchor)
    output = np.empty_like(points)
    for index, timestamp in enumerate(point_timestamps):
        pose = interpolate_pose(
            timestamps, translations, quaternions, float(timestamp), max_gap_s=max_gap_s
        )
        homogeneous = np.append(points[index], 1.0)
        output[index] = (anchor_inverse @ pose @ homogeneous)[:3]
    if not np.isfinite(output).all():
        raise ValueError("deskew generated nonfinite points")
    return np.ascontiguousarray(output, dtype="<f8")
