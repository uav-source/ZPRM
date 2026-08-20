"""Independent point-to-plane association and reassociation arithmetic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from .independent_transform_math_v1 import finite_matrix4


ASSOCIATION_DISTANCE_LIMIT_M = 0.50
PCA_K = 50
PCA_MIN_NEIGHBORS = 10
PCA_CHUNK_SIZE = 2048
NORMAL_NORM_EPSILON = 1.0e-12
QUANTILE_METHOD = "linear"
INVALID_REASONS = frozenset(
    {
        "NO_INITIAL_CORRESPONDENCE",
        "NO_FINAL_CORRESPONDENCE",
        "INSUFFICIENT_VALID_NORMALS",
        "NONFINITE_COMMON_METRICS",
        "OTHER",
    }
)
PAIR_TURNOVER_FIELD = "correspondence_" + "turnover"
SOURCE_TURNOVER_FIELD = "accepted_source_" + "turnover"


class IndependentAssociationError(ValueError):
    """Association inputs or calculations violate the frozen contract."""


def finite_points(value: Any, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise IndependentAssociationError(f"{label} must be a non-empty Nx3 array")
    if not np.all(np.isfinite(points)):
        raise IndependentAssociationError(f"{label} must be finite")
    return np.ascontiguousarray(points, dtype=np.float64)


def transform_points(points: Any, transform: Any) -> np.ndarray:
    source = finite_points(points, "source")
    matrix = finite_matrix4(transform, "source_to_target")
    return np.ascontiguousarray(source @ matrix[:3, :3].T + matrix[:3, 3])


@dataclass(frozen=True)
class Association:
    source_indices: np.ndarray
    target_indices: np.ndarray
    transformed_points: np.ndarray

    @property
    def count(self) -> int:
        return int(self.source_indices.size)


@dataclass(frozen=True)
class TargetContext:
    points: np.ndarray
    tree: cKDTree
    normals: np.ndarray
    normal_valid: np.ndarray


@dataclass(frozen=True)
class InitialContext:
    source_points: np.ndarray
    target: TargetContext
    association: Association
    signed_residuals: np.ndarray
    valid_mask: np.ndarray
    metrics: Mapping[str, Any]


def estimate_target_normals(points: Any, *, chunk_size: int = PCA_CHUNK_SIZE) -> tuple[np.ndarray, np.ndarray, cKDTree]:
    target = finite_points(points, "target")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise IndependentAssociationError("PCA chunk size must be a positive integer")
    tree = cKDTree(target)
    count = int(target.shape[0])
    normals = np.full((count, 3), np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    neighbors = min(PCA_K, count)
    if neighbors < PCA_MIN_NEIGHBORS:
        return normals, valid, tree
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        _, indices = tree.query(target[start:stop], k=neighbors, workers=1)
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices[:, None]
        neighborhoods = target[indices]
        centered = neighborhoods - np.mean(neighborhoods, axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True) / float(neighbors)
        _, eigenvectors = np.linalg.eigh(covariance)
        candidate = eigenvectors[:, :, 0]
        norm = np.linalg.norm(candidate, axis=1)
        okay = np.all(np.isfinite(candidate), axis=1) & np.isfinite(norm) & (norm > NORMAL_NORM_EPSILON)
        candidate[okay] /= norm[okay, None]
        view = normals[start:stop]
        view[okay] = candidate[okay]
        valid[start:stop] = okay
    return normals, valid, tree


def prepare_target(points: Any) -> TargetContext:
    target = finite_points(points, "target")
    normals, valid, tree = estimate_target_normals(target)
    return TargetContext(points=target, tree=tree, normals=normals, normal_valid=valid)


def associate(source_points: Any, transform: Any, target: TargetContext) -> Association:
    transformed = transform_points(source_points, transform)
    distances, indices = target.tree.query(transformed, k=1, workers=1)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    accepted = (
        np.isfinite(distances)
        & (distances <= ASSOCIATION_DISTANCE_LIMIT_M)
        & (indices >= 0)
        & (indices < target.points.shape[0])
    )
    source_indices = np.flatnonzero(accepted).astype(np.int64, copy=False)
    if source_indices.size > 1 and np.any(source_indices[1:] <= source_indices[:-1]):
        raise IndependentAssociationError("accepted source indices are not strictly increasing")
    return Association(
        source_indices=source_indices,
        target_indices=np.ascontiguousarray(indices[accepted], dtype=np.int64),
        transformed_points=np.ascontiguousarray(transformed[accepted], dtype=np.float64),
    )


def signed_residuals(target: TargetContext, state: Association) -> tuple[np.ndarray, np.ndarray]:
    valid = target.normal_valid[state.target_indices]
    if not np.any(valid):
        return np.empty(0, dtype=np.float64), valid
    target_points = target.points[state.target_indices[valid]]
    normals = target.normals[state.target_indices[valid]]
    difference = state.transformed_points[valid] - target_points
    return np.asarray(np.einsum("ij,ij->i", normals, difference), dtype=np.float64), valid


def residual_summary(values: np.ndarray, prefix: str) -> dict[str, float | None]:
    if values.size == 0:
        return {f"{prefix}_residual_rmse": None, f"{prefix}_residual_median": None, f"{prefix}_residual_q95": None}
    absolute = np.abs(values)
    return {
        f"{prefix}_residual_rmse": float(np.sqrt(np.mean(np.square(values)))),
        f"{prefix}_residual_median": float(np.median(absolute)),
        f"{prefix}_residual_q95": float(np.quantile(absolute, 0.95, method=QUANTILE_METHOD)),
    }


def translation_geometry(normals: np.ndarray, signed: np.ndarray) -> dict[str, float | None]:
    if signed.size == 0:
        return {name: None for name in (
            "lambda_min_trans", "lambda_mid_trans", "lambda_max_trans",
            "normalized_lambda_min_trans", "normalized_lambda_mid_trans", "normalized_lambda_max_trans",
            "condition_number_trans", "spectral_entropy_trans", "initial_translation_gradient_norm",
        )}
    hessian = (normals.T @ normals) / float(normals.shape[0])
    eigenvalues = np.maximum(np.linalg.eigvalsh(hessian), 0.0)
    total = max(float(np.sum(eigenvalues)), 1.0e-12)
    normalized = eigenvalues / total
    positive = normalized > 0.0
    entropy = -float(np.sum(normalized[positive] * np.log(normalized[positive]))) / math.log(3.0)
    gradient = np.mean(normals * signed[:, None], axis=0)
    return {
        "lambda_min_trans": float(eigenvalues[0]),
        "lambda_mid_trans": float(eigenvalues[1]),
        "lambda_max_trans": float(eigenvalues[2]),
        "normalized_lambda_min_trans": float(normalized[0]),
        "normalized_lambda_mid_trans": float(normalized[1]),
        "normalized_lambda_max_trans": float(normalized[2]),
        "condition_number_trans": float(eigenvalues[2] / max(float(eigenvalues[0]), 1.0e-12)),
        "spectral_entropy_trans": entropy,
        "initial_translation_gradient_norm": float(np.linalg.norm(gradient)),
    }


def prepare_initial(source_points: Any, target: TargetContext, initial_transform: Any) -> InitialContext:
    source = finite_points(source_points, "source")
    initial = associate(source, initial_transform, target)
    signed, valid = signed_residuals(target, initial)
    selected_normals = target.normals[initial.target_indices[valid]]
    metrics: dict[str, Any] = {
        "initial_correspondence_count": initial.count,
        "initial_valid_normal_correspondence_count": int(signed.size),
        **residual_summary(signed, "initial"),
        **translation_geometry(selected_normals, signed),
    }
    return InitialContext(source, target, initial, signed, valid, metrics)


def turnover(initial: Association, final: Association) -> tuple[float | None, float | None]:
    common, initial_pos, final_pos = np.intersect1d(
        initial.source_indices, final.source_indices, assume_unique=True, return_indices=True
    )
    same_pairs = int(np.count_nonzero(initial.target_indices[initial_pos] == final.target_indices[final_pos]))
    pair_union = initial.count + final.count - same_pairs
    source_union = initial.count + final.count - int(common.size)
    return (
        None if pair_union == 0 else float(1.0 - same_pairs / pair_union),
        None if source_union == 0 else float(1.0 - common.size / source_union),
    )


def normal_angle_summary(initial: InitialContext, final: Association) -> tuple[float | None, float | None, int]:
    _, initial_pos, final_pos = np.intersect1d(
        initial.association.source_indices, final.source_indices,
        assume_unique=True, return_indices=True,
    )
    initial_targets = initial.association.target_indices[initial_pos]
    final_targets = final.target_indices[final_pos]
    valid = initial.target.normal_valid[initial_targets] & initial.target.normal_valid[final_targets]
    if not np.any(valid):
        return None, None, 0
    left = initial.target.normals[initial_targets[valid]]
    right = initial.target.normals[final_targets[valid]]
    cosine = np.clip(np.abs(np.einsum("ij,ij->i", left, right)), 0.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    return float(np.median(angles)), float(np.quantile(angles, 0.95, method=QUANTILE_METHOD)), int(angles.size)


_VALIDITY_FIELDS = (
    "initial_correspondence_count", "final_correspondence_count",
    "initial_valid_normal_correspondence_count", "final_valid_normal_correspondence_count",
    "correspondence_turnover", "accepted_source_turnover", "correspondence_count_change_ratio",
    "initial_residual_rmse", "initial_residual_median", "initial_residual_q95",
    "final_residual_rmse", "final_residual_median", "final_residual_q95", "residual_rmse_change",
    "median_normal_angle_change_deg", "q95_normal_angle_change_deg",
    "lambda_min_trans", "lambda_mid_trans", "lambda_max_trans",
    "normalized_lambda_min_trans", "normalized_lambda_mid_trans", "normalized_lambda_max_trans",
    "condition_number_trans", "spectral_entropy_trans", "initial_translation_gradient_norm",
)


def invalid_reason(metrics: Mapping[str, Any]) -> str | None:
    if metrics["initial_correspondence_count"] == 0:
        return "NO_INITIAL_CORRESPONDENCE"
    if metrics["final_correspondence_count"] == 0:
        return "NO_FINAL_CORRESPONDENCE"
    if (
        metrics["initial_valid_normal_correspondence_count"] == 0
        or metrics["final_valid_normal_correspondence_count"] == 0
        or metrics["common_valid_normal_source_count"] == 0
    ):
        return "INSUFFICIENT_VALID_NORMALS"
    for name in _VALIDITY_FIELDS:
        value = metrics.get(name)
        if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
            return "NONFINITE_COMMON_METRICS"
    return None


def analyze_final(initial: InitialContext, pose_matrix: Any) -> dict[str, Any]:
    final = associate(initial.source_points, pose_matrix, initial.target)
    final_signed, _ = signed_residuals(initial.target, final)
    pair_turnover, source_turnover = turnover(initial.association, final)
    median_angle, q95_angle, common_normal_count = normal_angle_summary(initial, final)
    initial_count = initial.association.count
    change_ratio = None if initial_count == 0 else float((final.count - initial_count) / initial_count)
    initial_rmse = initial.metrics["initial_residual_rmse"]
    final_summary = residual_summary(final_signed, "final")
    final_rmse = final_summary["final_residual_rmse"]
    metrics: dict[str, Any] = {
        **dict(initial.metrics),
        "final_correspondence_count": final.count,
        "final_valid_normal_correspondence_count": int(final_signed.size),
        "common_valid_normal_source_count": common_normal_count,
        PAIR_TURNOVER_FIELD: pair_turnover,
        SOURCE_TURNOVER_FIELD: source_turnover,
        "correspondence_count_change_ratio": change_ratio,
        **final_summary,
        "residual_rmse_change": None if initial_rmse is None or final_rmse is None else float(final_rmse - initial_rmse),
        "median_normal_angle_change_deg": median_angle,
        "q95_normal_angle_change_deg": q95_angle,
    }
    reason = invalid_reason(metrics)
    metrics.update(
        common_association_valid=reason is None,
        common_association_invalid_reason=reason,
        common_association_invalid_detail=None,
    )
    return metrics


__all__ = [
    "ASSOCIATION_DISTANCE_LIMIT_M", "INVALID_REASONS", "IndependentAssociationError",
    "InitialContext", "TargetContext", "analyze_final", "associate", "estimate_target_normals",
    "finite_points", "invalid_reason", "normal_angle_summary", "prepare_initial", "prepare_target",
    "residual_summary", "signed_residuals", "transform_points", "translation_geometry", "turnover",
]
