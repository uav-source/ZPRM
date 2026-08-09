"""Independent Open3D point-to-plane backend with one frozen parameter set."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import open3d as o3d

from .metrics import pose_matrix
from .types import BackendResult


REQUIRED_OPEN3D_VERSION = "0.19.0+b012259"


def validate_open3d_version() -> str:
    version = str(o3d.__version__)
    if version != REQUIRED_OPEN3D_VERSION:
        raise RuntimeError(
            f"Open3D version must be {REQUIRED_OPEN3D_VERSION}, got {version}"
        )
    return version


def run_open3d_full(
    scan_points: np.ndarray,
    map_points: np.ndarray,
    initial_pose: np.ndarray,
    open3d_config: Mapping[str, Any],
    backend_seed: int,
    input_checksum: str,
) -> BackendResult:
    """Run Open3D without accepting native correspondences or scene labels."""

    validate_open3d_version()
    # Open3D's pybind conversion requires a writeable owner even though ICP
    # does not mutate the logical snapshot.  Copy only at this backend edge.
    source_array = np.array(scan_points, dtype=np.float64, order="C", copy=True)
    target_array = np.array(map_points, dtype=np.float64, order="C", copy=True)
    if source_array.ndim != 2 or source_array.shape[1] != 3:
        raise ValueError("scan_points must be Nx3")
    if target_array.ndim != 2 or target_array.shape[1] != 3:
        raise ValueError("map_points must be Nx3")
    if not np.all(np.isfinite(source_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("Open3D inputs must be finite")
    if str(open3d_config["registration_method"]) != "point_to_plane":
        raise ValueError("Open3D method must remain point_to_plane")

    o3d.utility.random.seed(int(backend_seed))
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_array)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_array)
    normal = open3d_config["target_normal_estimation"]
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal["radius_m"]), max_nn=int(normal["max_nn"])
        )
    )
    convergence = open3d_config["icp_convergence"]
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=float(convergence["relative_fitness"]),
        relative_rmse=float(convergence["relative_rmse"]),
        max_iteration=int(convergence["max_iteration"]),
    )
    initial = pose_matrix(initial_pose)
    start = time.perf_counter()
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(open3d_config["maximum_correspondence_distance_m"]),
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )
    runtime_ms = (time.perf_counter() - start) * 1000.0
    transformation = np.asarray(result.transformation, dtype=np.float64)
    correspondence_count = len(result.correspondence_set)
    finite = bool(
        np.all(np.isfinite(transformation))
        and np.isfinite(result.fitness)
        and np.isfinite(result.inlier_rmse)
    )
    converged = bool(finite and correspondence_count > 0)
    reason = "completed_finite_correspondences" if converged else "open3d_invalid_result"
    return BackendResult(
        backend="open3d_full",
        final_pose=transformation,
        runtime_ms=runtime_ms,
        solver_converged=converged,
        finite_result=finite,
        iteration_count=-1,
        correspondence_count=int(correspondence_count),
        initial_cost=None,
        final_cost=float(result.inlier_rmse),
        termination_reason=reason,
        failure_reason="" if converged else reason,
        extra={
            "backend_input_checksum": str(input_checksum),
            "fitness": float(result.fitness),
            "inlier_rmse": float(result.inlier_rmse),
            "open3d_version": str(o3d.__version__),
        },
    )


__all__ = [
    "REQUIRED_OPEN3D_VERSION",
    "run_open3d_full",
    "validate_open3d_version",
]
