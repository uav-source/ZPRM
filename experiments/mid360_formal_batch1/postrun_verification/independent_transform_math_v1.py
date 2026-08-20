"""Independent transform and reflection-safe rotation arithmetic.

This module intentionally depends only on NumPy and the standard library.
It does not import any execution or scientific helper from ZPRM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


IDENTITY_4X4 = np.eye(4, dtype=np.float64)
HOMOGENEOUS_ATOL = 1.0e-12
ORTHOGONALITY_DEFECT_MAX = 1.0e-5
DETERMINANT_ERROR_MAX = 1.0e-5
PROJECTION_CORRECTION_MAX = 1.0e-5


class IndependentTransformError(ValueError):
    """A transform violates the frozen independent verifier contract."""


@dataclass(frozen=True)
class RotationAudit:
    angle_rad: float
    angle_deg: float
    orthogonality_defect_fro: float
    determinant: float
    projection_correction_fro: float
    projected_determinant: float


@dataclass(frozen=True)
class TransformAudit:
    delta: np.ndarray
    translation: np.ndarray
    translation_norm_m: float
    rotation: RotationAudit


def finite_matrix4(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise IndependentTransformError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(
        matrix[3], np.array([0.0, 0.0, 0.0, 1.0]),
        atol=HOMOGENEOUS_ATOL, rtol=0.0,
    ):
        raise IndependentTransformError(f"{label} has an invalid homogeneous row")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def require_identity(value: Any, label: str = "T0") -> np.ndarray:
    matrix = finite_matrix4(value, label)
    if not np.array_equal(matrix, IDENTITY_4X4):
        raise IndependentTransformError(f"{label} is not exact Identity")
    return matrix


def independent_rotation_audit(raw_rotation: Any) -> RotationAudit:
    raw = np.asarray(raw_rotation, dtype=np.float64)
    if raw.shape != (3, 3) or not np.all(np.isfinite(raw)):
        raise IndependentTransformError("rotation must be a finite 3x3 matrix")
    orthogonality = float(np.linalg.norm(raw.T @ raw - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(raw))
    left, _, right_transpose = np.linalg.svd(raw)
    sign = 1.0 if float(np.linalg.det(left @ right_transpose)) >= 0.0 else -1.0
    projected = left @ np.diag([1.0, 1.0, sign]) @ right_transpose
    projected_determinant = float(np.linalg.det(projected))
    correction = float(np.linalg.norm(projected - raw, ord="fro"))
    quality = (
        determinant > 0.0
        and orthogonality <= ORTHOGONALITY_DEFECT_MAX
        and abs(determinant - 1.0) <= DETERMINANT_ERROR_MAX
        and correction <= PROJECTION_CORRECTION_MAX
        and projected_determinant > 0.0
    )
    if not quality:
        raise IndependentTransformError(
            "rotation failed frozen SO(3) quality gates"
        )
    cosine = float(np.clip((np.trace(projected) - 1.0) * 0.5, -1.0, 1.0))
    skew = np.asarray(
        [
            projected[2, 1] - projected[1, 2],
            projected[0, 2] - projected[2, 0],
            projected[1, 0] - projected[0, 1],
        ],
        dtype=np.float64,
    )
    sine = float(0.5 * np.linalg.norm(skew))
    angle = float(math.atan2(sine, cosine))
    return RotationAudit(
        angle_rad=angle,
        angle_deg=float(math.degrees(angle)),
        orthogonality_defect_fro=orthogonality,
        determinant=determinant,
        projection_correction_fro=correction,
        projected_determinant=projected_determinant,
    )


def independent_transform_audit(initial: Any, estimated: Any) -> TransformAudit:
    t0 = require_identity(initial)
    estimate = finite_matrix4(estimated, "stored estimate")
    delta = np.linalg.inv(t0) @ estimate
    delta = finite_matrix4(delta, "Delta_T_recomputed")
    translation = np.asarray(delta[:3, 3], dtype=np.float64)
    return TransformAudit(
        delta=delta,
        translation=translation,
        translation_norm_m=float(np.linalg.norm(translation)),
        rotation=independent_rotation_audit(delta[:3, :3]),
    )


def absolute_relative_error(observed: Any, expected: Any) -> tuple[float, float]:
    left = np.asarray(observed, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise IndependentTransformError("comparison operands differ in shape or finiteness")
    absolute = np.abs(left - right)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), np.finfo(np.float64).tiny)
    return float(np.max(absolute, initial=0.0)), float(np.max(absolute / scale, initial=0.0))


def close(observed: Any, expected: Any, *, atol: float, rtol: float) -> bool:
    try:
        left = np.asarray(observed, dtype=np.float64)
        right = np.asarray(expected, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        left.shape == right.shape
        and np.all(np.isfinite(left))
        and np.all(np.isfinite(right))
        and np.allclose(left, right, atol=atol, rtol=rtol)
    )


__all__ = [
    "IndependentTransformError",
    "RotationAudit",
    "TransformAudit",
    "absolute_relative_error",
    "close",
    "finite_matrix4",
    "independent_rotation_audit",
    "independent_transform_audit",
    "require_identity",
]
