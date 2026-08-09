"""Canonical input, robust update, failure, and Gate contracts for Phase A."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .rotation_metrics import rotation_metric_audit


NOT_EVALUATED = "NOT_EVALUATED"
OPEN3D_BACKEND = "open3d_point_to_plane"
PCL_BACKEND = "pcl_iterative_closest_point_with_normals"


def canonical_float32_points(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(f"{label} must be non-empty Nx3")
    canonical = np.array(array, dtype="<f4", order="C", copy=True)
    if not np.all(np.isfinite(canonical)):
        raise ValueError(f"{label} contains non-finite values")
    if canonical.dtype.str != "<f4" or not canonical.flags.c_contiguous:
        raise ValueError(f"{label} canonicalization failed")
    canonical.setflags(write=False)
    return canonical


def canonical_float64_transform(value: Any, label: str = "reference_pose") -> np.ndarray:
    matrix = np.array(value, dtype="<f8", order="C", copy=True)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if matrix.dtype.str != "<f8" or not matrix.flags.c_contiguous:
        raise ValueError(f"{label} canonicalization failed")
    matrix.setflags(write=False)
    return matrix


def raw_bytes_sha256(array: np.ndarray) -> str:
    if not array.flags.c_contiguous:
        raise ValueError("checksum input must be C-contiguous")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def canonical_backend_inputs(
    source_points: Any, target_points: Any, reference_pose: Any
) -> dict[str, Any]:
    source = canonical_float32_points(source_points, "source_points")
    target = canonical_float32_points(target_points, "target_points")
    reference = canonical_float64_transform(reference_pose)
    return {
        "source_points": source,
        "target_points": target,
        "reference_pose": reference,
        "source_checksum": raw_bytes_sha256(source),
        "target_checksum": raw_bytes_sha256(target),
        "reference_pose_checksum": raw_bytes_sha256(reference),
    }


def source_is_exact_target_subset(source: np.ndarray, target: np.ndarray) -> bool:
    canonical_source = canonical_float32_points(source, "source_points")
    canonical_target = canonical_float32_points(target, "target_points")
    target_rows = {row.tobytes() for row in canonical_target}
    return all(row.tobytes() in target_rows for row in canonical_source)


def transform_update(reference_pose: Any, estimated_pose: Any) -> dict[str, Any]:
    reference = canonical_float64_transform(reference_pose)
    estimate = canonical_float64_transform(estimated_pose, "estimated_pose")
    delta = np.linalg.inv(reference) @ estimate
    if not np.all(np.isfinite(delta)):
        raise ValueError("T_delta is non-finite")
    rotation = rotation_metric_audit(delta[:3, :3], np.eye(3, dtype=np.float64))
    return {
        "T_delta": delta,
        "translation_update_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_update_rad": rotation.get("rotation_error_rad"),
        "rotation_matrix_quality_pass": rotation["rotation_matrix_quality_pass"],
        "rotation_audit": rotation,
    }


def _checksum_mismatch(
    payload: Mapping[str, Any], expected_checksums: Mapping[str, str]
) -> bool:
    return any(
        str(payload.get(name, "")) != str(expected)
        for name, expected in expected_checksums.items()
    )


def solver_failure_reasons(
    backend: str,
    payload: Mapping[str, Any] | None,
    expected_checksums: Mapping[str, str],
) -> tuple[str, ...]:
    """Classify an executed result; callers must not pass an unexecuted trial."""

    if payload is None:
        raise ValueError("unexecuted trial is NOT_EVALUATED, not zero failures")
    reasons: list[str] = []
    if backend == OPEN3D_BACKEND:
        required = {
            "exception",
            "final_transform_finite",
            "fitness_finite",
            "inlier_rmse_finite",
            "correspondence_count",
            "rotation_matrix_quality_pass",
            *expected_checksums,
        }
        if required - set(payload):
            reasons.append("REQUIRED_OUTPUT_FIELD_MISSING")
        if payload.get("exception"):
            reasons.append("PYTHON_OR_CPP_EXCEPTION")
        if payload.get("final_transform_finite") is not True:
            reasons.append("NONFINITE_TRANSFORM")
        if payload.get("fitness_finite") is not True:
            reasons.append("NONFINITE_FITNESS")
        if payload.get("inlier_rmse_finite") is not True:
            reasons.append("NONFINITE_INLIER_RMSE")
        if int(payload.get("correspondence_count", 0)) <= 0:
            reasons.append("NO_CORRESPONDENCES")
        if payload.get("rotation_matrix_quality_pass") is not True:
            reasons.append("ROTATION_MATRIX_QUALITY_FAILED")
    elif backend == PCL_BACKEND:
        required = {
            "cli_exit_code",
            "exception",
            "has_converged_raw",
            "final_transform_finite",
            "fitness_finite",
            "correspondence_count",
            "source_normal_nan_count",
            "source_normal_zero_count",
            "target_normal_nan_count",
            "target_normal_zero_count",
            "rotation_matrix_quality_pass",
            *expected_checksums,
        }
        if required - set(payload):
            reasons.append("REQUIRED_OUTPUT_FIELD_MISSING")
        if payload.get("exception") or int(payload.get("cli_exit_code", -1)) != 0:
            reasons.append("CLI_EXCEPTION_OR_NONZERO_EXIT")
        if payload.get("has_converged_raw") is not True:
            reasons.append("PCL_NOT_CONVERGED")
        if payload.get("final_transform_finite") is not True:
            reasons.append("NONFINITE_TRANSFORM")
        if payload.get("fitness_finite") is not True:
            reasons.append("NONFINITE_FITNESS")
        if int(payload.get("correspondence_count", 0)) <= 0:
            reasons.append("NO_CORRESPONDENCES")
        if any(
            int(payload.get(name, -1)) != 0
            for name in (
                "source_normal_nan_count",
                "source_normal_zero_count",
                "target_normal_nan_count",
                "target_normal_zero_count",
            )
        ):
            reasons.append("INVALID_NORMALS")
        if payload.get("rotation_matrix_quality_pass") is not True:
            reasons.append("ROTATION_MATRIX_QUALITY_FAILED")
    else:
        raise ValueError(f"backend is not authorized for Phase A: {backend}")
    if _checksum_mismatch(payload, expected_checksums):
        reasons.append("SNAPSHOT_OR_INPUT_CHECKSUM_MISMATCH")
    return tuple(dict.fromkeys(reasons))


def phase_a_backend_gate(
    rows: Iterable[Mapping[str, Any]], *, expected_count: int = 210
) -> dict[str, Any]:
    results = tuple(rows)
    if len(results) != expected_count:
        return {
            "status": NOT_EVALUATED,
            "complete": False,
            "executed_trial_count": len(results),
            "required_trial_count": expected_count,
            "solver_failure_count": None,
            "nonfinite_output_count": None,
            "gate_pass": False,
        }
    translations = np.asarray(
        [float(row["translation_update_m"]) for row in results], dtype=np.float64
    )
    rotations = np.asarray(
        [float(row["rotation_update_rad"]) for row in results], dtype=np.float64
    )
    if not np.all(np.isfinite(translations)) or not np.all(np.isfinite(rotations)):
        raise ValueError("Phase A aggregate contains non-finite update metrics")
    solver_failures = sum(bool(row["solver_failed"]) for row in results)
    nonfinite = sum(not bool(row["finite_output"]) for row in results)
    q95_translation = float(np.quantile(translations, 0.95, method="linear"))
    q95_rotation = float(np.quantile(rotations, 0.95, method="linear"))
    fraction = float(np.mean(translations <= 0.001))
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in results:
        grouped[str(row["scene_variant"])].append(float(row["translation_update_m"]))
    scene_medians = {
        scene: float(np.median(values)) for scene, values in sorted(grouped.items())
    }
    scene_gate = len(scene_medians) == 7 and all(
        len(grouped[scene]) == 30 and median <= 0.001
        for scene, median in scene_medians.items()
    )
    passed = bool(
        solver_failures == 0
        and nonfinite == 0
        and q95_translation <= 0.001
        and q95_rotation <= 0.00017453292519943296
        and fraction >= 0.95
        and scene_gate
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "complete": True,
        "executed_trial_count": len(results),
        "required_trial_count": expected_count,
        "solver_failure_count": solver_failures,
        "nonfinite_output_count": nonfinite,
        "q95_translation_update_m": q95_translation,
        "q95_rotation_update_rad": q95_rotation,
        "translation_within_0_001_m_fraction": fraction,
        "scene_median_translation_update_m": scene_medians,
        "per_scene_median_gate_pass": scene_gate,
        "gate_pass": passed,
    }


def snapshot_diversity_gate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    snapshots = tuple(rows)
    if len(snapshots) != 210:
        return {
            "status": NOT_EVALUATED,
            "complete": False,
            "IDEAL_MATCHED_SNAPSHOT_DIVERSITY_PASS": False,
            "scenes": {},
        }
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in snapshots:
        by_scene[str(row["scene_variant"])].append(row)
    scene_results: dict[str, Any] = {}
    for scene, scene_rows in sorted(by_scene.items()):
        sources = [str(row["source_checksum"]) for row in scene_rows]
        targets = [str(row["target_checksum"]) for row in scene_rows]
        source_counts = Counter(sources)
        scene_results[scene] = {
            "snapshot_count": len(scene_rows),
            "unique_target_checksum_count": len(set(targets)),
            "unique_source_checksum_count": len(source_counts),
            "duplicate_source_checksum_count": sum(
                count - 1 for count in source_counts.values() if count > 1
            ),
            "pass": len(scene_rows) == 30 and len(source_counts) >= 10,
        }
    passed = len(scene_results) == 7 and all(
        value["pass"] for value in scene_results.values()
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "complete": True,
        "IDEAL_MATCHED_SNAPSHOT_DIVERSITY_PASS": passed,
        "scenes": scene_results,
    }


__all__ = [
    "NOT_EVALUATED",
    "OPEN3D_BACKEND",
    "PCL_BACKEND",
    "canonical_backend_inputs",
    "canonical_float32_points",
    "canonical_float64_transform",
    "phase_a_backend_gate",
    "raw_bytes_sha256",
    "snapshot_diversity_gate",
    "solver_failure_reasons",
    "source_is_exact_target_subset",
    "transform_update",
]
