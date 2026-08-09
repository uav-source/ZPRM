"""Reflection-safe SO(3) projection and robust rotation-error metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


ORTHOGONALITY_DEFECT_MAX = 1.0e-5
DETERMINANT_ERROR_MAX = 1.0e-5
PROJECTION_CORRECTION_MAX = 1.0e-5
TRUTH_ORTHOGONALITY_TOLERANCE = 1.0e-10
TRUTH_DETERMINANT_TOLERANCE = 1.0e-10


def _matrix3(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    return matrix


def project_to_so3(raw_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest proper rotation and singular values, handling reflection."""

    raw = _matrix3(raw_rotation)
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw rotation is not finite")
    left, singular_values, right_transpose = np.linalg.svd(raw)
    sign = 1.0 if np.linalg.det(left @ right_transpose) >= 0.0 else -1.0
    correction = np.diag([1.0, 1.0, sign])
    projected = left @ correction @ right_transpose
    if not np.linalg.det(projected) > 0.0:
        raise ValueError("reflection-safe projection did not produce positive determinant")
    return projected, singular_values


def rotation_error_atan2(truth_rotation: np.ndarray, projected_rotation: np.ndarray) -> dict[str, float]:
    truth = _matrix3(truth_rotation)
    projected = _matrix3(projected_rotation)
    error = truth.T @ projected
    cos_theta = float(np.clip((np.trace(error) - 1.0) * 0.5, -1.0, 1.0))
    skew_vector = np.asarray(
        [
            error[2, 1] - error[1, 2],
            error[0, 2] - error[2, 0],
            error[1, 0] - error[0, 1],
        ],
        dtype=np.float64,
    )
    sin_theta = float(0.5 * np.linalg.norm(skew_vector))
    return {
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "rotation_error_rad": float(math.atan2(sin_theta, cos_theta)),
    }


def rotation_metric_audit(raw_rotation: np.ndarray, truth_rotation: np.ndarray) -> dict[str, Any]:
    raw = _matrix3(raw_rotation)
    truth = _matrix3(truth_rotation)
    raw_finite = bool(np.all(np.isfinite(raw)))
    truth_finite = bool(np.all(np.isfinite(truth)))
    if not raw_finite or not truth_finite:
        return {
            "raw_rotation_finite": raw_finite,
            "truth_rotation_finite": truth_finite,
            "raw_rotation_determinant_positive": False,
            "truth_rotation_valid": False,
            "rotation_matrix_quality_pass": False,
            "rotation_error_rad": None,
        }

    raw_transpose_raw = raw.T @ raw
    raw_determinant = float(np.linalg.det(raw))
    orthogonality_defect = float(np.linalg.norm(raw_transpose_raw - np.eye(3), ord="fro"))
    projected, singular_values = project_to_so3(raw)
    projected_determinant = float(np.linalg.det(projected))
    projection_correction = float(np.linalg.norm(projected - raw, ord="fro"))

    truth_orthogonality_defect = float(
        np.linalg.norm(truth.T @ truth - np.eye(3), ord="fro")
    )
    truth_determinant = float(np.linalg.det(truth))
    truth_valid = bool(
        truth_finite
        and truth_determinant > 0.0
        and truth_orthogonality_defect <= TRUTH_ORTHOGONALITY_TOLERANCE
        and abs(truth_determinant - 1.0) <= TRUTH_DETERMINANT_TOLERANCE
    )
    quality_pass = bool(
        raw_finite
        and raw_determinant > 0.0
        and orthogonality_defect <= ORTHOGONALITY_DEFECT_MAX
        and abs(raw_determinant - 1.0) <= DETERMINANT_ERROR_MAX
        and projection_correction <= PROJECTION_CORRECTION_MAX
        and projected_determinant > 0.0
        and truth_valid
    )
    raw_error = truth.T @ raw
    raw_argument = float((np.trace(raw_error) - 1.0) * 0.5)
    raw_acos = float(math.acos(float(np.clip(raw_argument, -1.0, 1.0))))
    formal = rotation_error_atan2(truth, projected)
    return {
        "raw_rotation_finite": raw_finite,
        "raw_rotation_determinant_positive": raw_determinant > 0.0,
        "raw_rotation_3x3": raw.tolist(),
        "R_est_transpose_R_est": raw_transpose_raw.tolist(),
        "orthogonality_defect_fro": orthogonality_defect,
        "determinant": raw_determinant,
        "raw_trace_acos_argument": raw_argument,
        "raw_trace_acos_rotation_error_rad": raw_acos,
        "nearest_so3_projection": projected.tolist(),
        "projected_determinant": projected_determinant,
        "projection_correction_fro": projection_correction,
        "singular_values": singular_values.tolist(),
        "truth_rotation_finite": truth_finite,
        "truth_rotation_determinant": truth_determinant,
        "truth_rotation_orthogonality_defect_fro": truth_orthogonality_defect,
        "truth_rotation_valid": truth_valid,
        "maximum_elementwise_error_to_truth": float(np.max(np.abs(raw - truth))),
        "cos_theta": formal["cos_theta"],
        "sin_theta": formal["sin_theta"],
        "rotation_error_rad": formal["rotation_error_rad"],
        "rotation_matrix_quality_pass": quality_pass,
    }


__all__ = [
    "DETERMINANT_ERROR_MAX",
    "ORTHOGONALITY_DEFECT_MAX",
    "PROJECTION_CORRECTION_MAX",
    "project_to_so3",
    "rotation_error_atan2",
    "rotation_metric_audit",
]
