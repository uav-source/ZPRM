"""Backend-independent offline correspondence diagnostics for Development.

This module never reads backend correspondence diagnostics.  One immutable
``CommonAssociationContext`` is prepared per snapshot and shared by both
backends.  Target PCA normals, the target cKDTree, reference-pose matches,
initial residuals, and translation geometry are therefore computed once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree


MAX_ASSOCIATION_DISTANCE_M = 0.50
PCA_NEIGHBOR_COUNT = 50
PCA_MIN_NEIGHBOR_COUNT = 10
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

_METRIC_FIELDS = (
    "initial_correspondence_count",
    "final_correspondence_count",
    "initial_valid_normal_correspondence_count",
    "final_valid_normal_correspondence_count",
    "correspondence_turnover",
    "accepted_source_turnover",
    "correspondence_count_change_ratio",
    "initial_residual_rmse",
    "initial_residual_median",
    "initial_residual_q95",
    "final_residual_rmse",
    "final_residual_median",
    "final_residual_q95",
    "residual_rmse_change",
    "median_normal_angle_change_deg",
    "q95_normal_angle_change_deg",
    "lambda_min_trans",
    "lambda_mid_trans",
    "lambda_max_trans",
    "normalized_lambda_min_trans",
    "normalized_lambda_mid_trans",
    "normalized_lambda_max_trans",
    "condition_number_trans",
    "spectral_entropy_trans",
    "initial_translation_gradient_norm",
)


def _readonly(value: Any, dtype: np.dtype[Any] | type = np.float64) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.base is not None or array.flags.writeable:
        array = np.array(array, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _points(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(f"{label} must be a non-empty Nx3 array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return np.ascontiguousarray(array)


def _transform(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12, rtol=0.0):
        raise ValueError(f"{label} has an invalid homogeneous row")
    return np.ascontiguousarray(matrix)


def transform_source_points(source_points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform source points into target/map coordinates."""

    points = _points(source_points, "source_points")
    matrix = _transform(transform, "source_to_target_transform")
    return np.ascontiguousarray(
        points @ matrix[:3, :3].T + matrix[:3, 3], dtype=np.float64
    )


@dataclass(frozen=True)
class AssociationState:
    """One nearest-target association per accepted source index."""

    source_indices: np.ndarray
    target_indices: np.ndarray
    source_points_target: np.ndarray
    distances_m: np.ndarray

    def __post_init__(self) -> None:
        source = _readonly(self.source_indices, np.int64)
        target = _readonly(self.target_indices, np.int64)
        points = _readonly(self.source_points_target, np.float64)
        distances = _readonly(self.distances_m, np.float64)
        count = source.size
        if source.ndim != 1 or target.shape != (count,) or distances.shape != (count,):
            raise ValueError("association index/distance shapes disagree")
        if points.shape != (count, 3):
            raise ValueError("association transformed-point shape disagrees")
        if count and (
            np.any(source < 0)
            or np.any(target < 0)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(distances))
            or np.any(distances < 0.0)
            or np.any(distances > MAX_ASSOCIATION_DISTANCE_M)
        ):
            raise ValueError("association payload is invalid")
        if count > 1 and np.any(source[1:] <= source[:-1]):
            raise ValueError("accepted source indices must be strictly increasing")
        object.__setattr__(self, "source_indices", source)
        object.__setattr__(self, "target_indices", target)
        object.__setattr__(self, "source_points_target", points)
        object.__setattr__(self, "distances_m", distances)

    @property
    def count(self) -> int:
        return int(self.source_indices.size)


def associate_source_points(
    source_points: np.ndarray,
    transform: np.ndarray,
    *,
    target_tree: cKDTree,
    target_point_count: int,
) -> AssociationState:
    """Apply the fixed 0.50 m one-nearest-neighbor association contract."""

    transformed = transform_source_points(source_points, transform)
    distances, targets = target_tree.query(transformed, k=1, workers=1)
    distances = np.asarray(distances, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    accepted = (
        np.isfinite(distances)
        & (distances <= MAX_ASSOCIATION_DISTANCE_M)
        & (targets >= 0)
        & (targets < int(target_point_count))
    )
    sources = np.flatnonzero(accepted).astype(np.int64, copy=False)
    return AssociationState(
        source_indices=sources,
        target_indices=targets[accepted],
        source_points_target=transformed[accepted],
        distances_m=distances[accepted],
    )


def _estimate_target_normals_with_tree(
    target_points: np.ndarray,
    tree: cKDTree,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = _points(target_points, "target_points")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    count = target.shape[0]
    normals = np.full((count, 3), np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    neighbor_count = min(PCA_NEIGHBOR_COUNT, count)
    if neighbor_count < PCA_MIN_NEIGHBOR_COUNT:
        normals.setflags(write=False)
        valid.setflags(write=False)
        return normals, valid

    # Chunking avoids materializing N x 50 x 3 neighborhoods for large maps.
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        _, indices = tree.query(
            target[start:stop], k=neighbor_count, workers=1
        )
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices[:, None]
        neighborhoods = target[indices]
        centered = neighborhoods - np.mean(neighborhoods, axis=1, keepdims=True)
        covariance = np.einsum(
            "nki,nkj->nij", centered, centered, optimize=True
        ) / float(neighbor_count)
        _, eigenvectors = np.linalg.eigh(covariance)
        candidate = eigenvectors[:, :, 0]
        norms = np.linalg.norm(candidate, axis=1)
        chunk_valid = np.all(np.isfinite(candidate), axis=1) & np.isfinite(norms) & (
            norms > NORMAL_NORM_EPSILON
        )
        candidate[chunk_valid] /= norms[chunk_valid, None]
        normals[start:stop][chunk_valid] = candidate[chunk_valid]
        valid[start:stop] = chunk_valid
    normals.setflags(write=False)
    valid.setflags(write=False)
    return normals, valid


def estimate_target_normals_pca(
    target_points: np.ndarray, *, chunk_size: int = PCA_CHUNK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate unoriented PCA normals with fixed k=50/minimum=10."""

    target = _points(target_points, "target_points")
    return _estimate_target_normals_with_tree(
        target, cKDTree(target), chunk_size=chunk_size
    )


def _residuals(
    target_points: np.ndarray,
    normals: np.ndarray,
    normal_valid: np.ndarray,
    state: AssociationState,
) -> tuple[np.ndarray, np.ndarray]:
    valid = normal_valid[state.target_indices]
    if not np.any(valid):
        return np.empty(0, dtype=np.float64), valid
    targets = target_points[state.target_indices[valid]]
    selected_normals = normals[state.target_indices[valid]]
    difference = state.source_points_target[valid] - targets
    signed = np.einsum("ij,ij->i", selected_normals, difference)
    return np.asarray(signed, dtype=np.float64), valid


def _residual_summary(signed: np.ndarray, prefix: str) -> dict[str, float | None]:
    if signed.size == 0:
        return {
            f"{prefix}_residual_rmse": None,
            f"{prefix}_residual_median": None,
            f"{prefix}_residual_q95": None,
        }
    absolute = np.abs(signed)
    return {
        f"{prefix}_residual_rmse": float(np.sqrt(np.mean(np.square(signed)))),
        f"{prefix}_residual_median": float(np.median(absolute)),
        f"{prefix}_residual_q95": float(
            np.quantile(absolute, 0.95, method=QUANTILE_METHOD)
        ),
    }


def _translation_geometry(normals: np.ndarray, signed: np.ndarray) -> dict[str, Any]:
    if signed.size == 0:
        return {
            "lambda_min_trans": None,
            "lambda_mid_trans": None,
            "lambda_max_trans": None,
            "normalized_lambda_min_trans": None,
            "normalized_lambda_mid_trans": None,
            "normalized_lambda_max_trans": None,
            "condition_number_trans": None,
            "spectral_entropy_trans": None,
            "initial_translation_gradient_norm": None,
        }
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
        "condition_number_trans": float(
            eigenvalues[2] / max(float(eigenvalues[0]), 1.0e-12)
        ),
        "spectral_entropy_trans": entropy,
        "initial_translation_gradient_norm": float(np.linalg.norm(gradient)),
    }


@dataclass(frozen=True)
class CommonAssociationContext:
    """Snapshot-level cache reusable across all estimated backend transforms."""

    source_points: np.ndarray
    target_points: np.ndarray
    reference_transform: np.ndarray
    target_tree: cKDTree
    target_normals: np.ndarray
    target_normal_valid: np.ndarray
    initial_association: AssociationState
    initial_metrics: Mapping[str, Any]
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_points", _readonly(self.source_points))
        object.__setattr__(self, "target_points", _readonly(self.target_points))
        object.__setattr__(self, "reference_transform", _readonly(self.reference_transform))
        object.__setattr__(self, "target_normals", _readonly(self.target_normals))
        object.__setattr__(
            self, "target_normal_valid", _readonly(self.target_normal_valid, np.bool_)
        )
        object.__setattr__(self, "initial_metrics", MappingProxyType(dict(self.initial_metrics)))


def prepare_common_association_context(
    source_points: np.ndarray,
    target_points: np.ndarray,
    reference_transform: np.ndarray,
    *,
    snapshot_id: str | None = None,
    pca_chunk_size: int = PCA_CHUNK_SIZE,
) -> CommonAssociationContext:
    """Prepare all snapshot-shared common-association state exactly once."""

    # Own the arrays before constructing the tree.  ``cKDTree`` otherwise
    # aliases an already-contiguous float64 caller array, which would let a
    # later caller mutation silently change a cached snapshot context.
    source = _readonly(_points(source_points, "source_points"))
    target = _readonly(_points(target_points, "target_points"))
    reference = _readonly(_transform(reference_transform, "reference_transform"))
    tree = cKDTree(target)
    normals, normal_valid = _estimate_target_normals_with_tree(
        target, tree, chunk_size=pca_chunk_size
    )
    initial = associate_source_points(
        source,
        reference,
        target_tree=tree,
        target_point_count=target.shape[0],
    )
    signed, valid_matches = _residuals(target, normals, normal_valid, initial)
    selected_normals = normals[initial.target_indices[valid_matches]]
    initial_metrics: dict[str, Any] = {
        "initial_correspondence_count": initial.count,
        "initial_valid_normal_correspondence_count": int(signed.size),
        "target_normal_valid_count": int(np.count_nonzero(normal_valid)),
        "target_normal_invalid_count": int(normal_valid.size - np.count_nonzero(normal_valid)),
        **_residual_summary(signed, "initial"),
        **_translation_geometry(selected_normals, signed),
    }
    return CommonAssociationContext(
        source_points=source,
        target_points=target,
        reference_transform=reference,
        target_tree=tree,
        target_normals=normals,
        target_normal_valid=normal_valid,
        initial_association=initial,
        initial_metrics=initial_metrics,
        snapshot_id=snapshot_id,
    )


def turnover_from_associations(
    initial: AssociationState, final: AssociationState
) -> tuple[float | None, float | None]:
    """Return pair-Jaccard and accepted-source Jaccard turnover."""

    common_sources, initial_positions, final_positions = np.intersect1d(
        initial.source_indices,
        final.source_indices,
        assume_unique=True,
        return_indices=True,
    )
    same_pair_count = int(
        np.count_nonzero(
            initial.target_indices[initial_positions]
            == final.target_indices[final_positions]
        )
    )
    pair_union = initial.count + final.count - same_pair_count
    source_intersection = int(common_sources.size)
    source_union = initial.count + final.count - source_intersection
    pair_turnover = None if pair_union == 0 else 1.0 - same_pair_count / pair_union
    source_turnover = (
        None if source_union == 0 else 1.0 - source_intersection / source_union
    )
    return pair_turnover, source_turnover


def _normal_angle_summary(
    context: CommonAssociationContext, final: AssociationState
) -> tuple[float | None, float | None, int]:
    _, initial_positions, final_positions = np.intersect1d(
        context.initial_association.source_indices,
        final.source_indices,
        assume_unique=True,
        return_indices=True,
    )
    initial_targets = context.initial_association.target_indices[initial_positions]
    final_targets = final.target_indices[final_positions]
    valid = context.target_normal_valid[initial_targets] & context.target_normal_valid[
        final_targets
    ]
    if not np.any(valid):
        return None, None, 0
    initial_normals = context.target_normals[initial_targets[valid]]
    final_normals = context.target_normals[final_targets[valid]]
    cosine = np.clip(
        np.abs(np.einsum("ij,ij->i", initial_normals, final_normals)), 0.0, 1.0
    )
    angles = np.degrees(np.arccos(cosine))
    return (
        float(np.median(angles)),
        float(np.quantile(angles, 0.95, method=QUANTILE_METHOD)),
        int(angles.size),
    )


def _invalid_reason(metrics: Mapping[str, Any]) -> str | None:
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
    for name in _METRIC_FIELDS:
        value = metrics.get(name)
        if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
            return "NONFINITE_COMMON_METRICS"
    return None


def analyze_estimated_transform(
    context: CommonAssociationContext,
    estimated_transform: np.ndarray,
    *,
    identifiers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute one trial record while reusing all snapshot-level preparation."""

    final = associate_source_points(
        context.source_points,
        _transform(estimated_transform, "estimated_transform"),
        target_tree=context.target_tree,
        target_point_count=context.target_points.shape[0],
    )
    final_signed, final_valid = _residuals(
        context.target_points,
        context.target_normals,
        context.target_normal_valid,
        final,
    )
    final_residual = _residual_summary(final_signed, "final")
    pair_turnover, source_turnover = turnover_from_associations(
        context.initial_association, final
    )
    median_angle, q95_angle, common_normal_count = _normal_angle_summary(
        context, final
    )
    initial_count = context.initial_association.count
    change_ratio = (
        None
        if initial_count == 0
        else float((final.count - initial_count) / initial_count)
    )
    initial_rmse = context.initial_metrics["initial_residual_rmse"]
    final_rmse = final_residual["final_residual_rmse"]
    metrics: dict[str, Any] = {
        **dict(context.initial_metrics),
        "final_correspondence_count": final.count,
        "final_valid_normal_correspondence_count": int(final_signed.size),
        "common_valid_normal_source_count": common_normal_count,
        "correspondence_turnover": pair_turnover,
        "accepted_source_turnover": source_turnover,
        "correspondence_count_change_ratio": change_ratio,
        **final_residual,
        "residual_rmse_change": (
            None
            if initial_rmse is None or final_rmse is None
            else float(final_rmse - initial_rmse)
        ),
        "median_normal_angle_change_deg": median_angle,
        "q95_normal_angle_change_deg": q95_angle,
    }
    reason = _invalid_reason(metrics)
    identity = dict(identifiers or {})
    collisions = set(identity) & (
        set(metrics)
        | {
            "common_association_valid",
            "common_association_invalid_reason",
            "common_association_invalid_detail",
        }
    )
    if collisions:
        raise ValueError(f"identifier fields collide with common metrics: {sorted(collisions)}")
    return {
        **identity,
        "snapshot_id": identity.get("snapshot_id", context.snapshot_id),
        "common_association_valid": reason is None,
        "common_association_invalid_reason": reason,
        "common_association_is_backend_internal": False,
        "association_distance_limit_m": MAX_ASSOCIATION_DISTANCE_M,
        "target_normal_pca_k": PCA_NEIGHBOR_COUNT,
        "target_normal_pca_min_neighbors": PCA_MIN_NEIGHBOR_COUNT,
        **metrics,
    }


def safe_analyze_estimated_transform(
    context: CommonAssociationContext,
    estimated_transform: np.ndarray,
    *,
    identifiers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Always emit a record; unexpected analysis failures become ``OTHER``."""

    try:
        return analyze_estimated_transform(
            context, estimated_transform, identifiers=identifiers
        )
    except Exception as error:
        identity = dict(identifiers or {})
        return {
            **identity,
            "snapshot_id": identity.get("snapshot_id", context.snapshot_id),
            "common_association_valid": False,
            "common_association_invalid_reason": "OTHER",
            "common_association_invalid_detail": f"{type(error).__name__}: {error}",
            "common_association_is_backend_internal": False,
            "association_distance_limit_m": MAX_ASSOCIATION_DISTANCE_M,
            "target_normal_pca_k": PCA_NEIGHBOR_COUNT,
            "target_normal_pca_min_neighbors": PCA_MIN_NEIGHBOR_COUNT,
            "target_normal_valid_count": int(
                np.count_nonzero(context.target_normal_valid)
            ),
            "target_normal_invalid_count": int(
                context.target_normal_valid.size
                - np.count_nonzero(context.target_normal_valid)
            ),
            "common_valid_normal_source_count": 0,
            **{name: None for name in _METRIC_FIELDS},
        }


def common_association_valid_fraction(records: list[Mapping[str, Any]]) -> float:
    if not records:
        raise ValueError("at least one common-association record is required")
    return float(
        sum(row.get("common_association_valid") is True for row in records)
        / len(records)
    )


__all__ = [
    "AssociationState",
    "CommonAssociationContext",
    "INVALID_REASONS",
    "MAX_ASSOCIATION_DISTANCE_M",
    "NORMAL_NORM_EPSILON",
    "PCA_CHUNK_SIZE",
    "PCA_MIN_NEIGHBOR_COUNT",
    "PCA_NEIGHBOR_COUNT",
    "analyze_estimated_transform",
    "associate_source_points",
    "common_association_valid_fraction",
    "estimate_target_normals_pca",
    "prepare_common_association_context",
    "safe_analyze_estimated_transform",
    "transform_source_points",
    "turnover_from_associations",
]
