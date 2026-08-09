"""Isolated adapter for the frozen PCL point-to-plane CLI."""

from __future__ import annotations

import json
import math
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .metrics import pose_matrix, zero_initialization_error


BACKEND_NAME = "pcl_point_to_plane"
REQUIRED_RESULT_FIELDS = frozenset(
    {
        "trial_id",
        "backend_name",
        "pcl_version",
        "has_converged_raw",
        "has_converged",
        "final_transform_finite",
        "fitness_finite",
        "fitness_score",
        "final_transformation_4x4",
        "translation_update_norm_m",
        "rotation_update_norm_rad",
        "source_point_count",
        "target_point_count",
        "finite_output",
        "qualification_pass",
        "iteration_count",
        "correspondence_count",
        "source_normal_finite_count",
        "source_normal_zero_count",
        "source_normal_nan_count",
        "source_normal_norm_min",
        "source_normal_norm_median",
        "source_normal_norm_max",
        "target_normal_finite_count",
        "target_normal_zero_count",
        "target_normal_nan_count",
        "target_normal_norm_min",
        "target_normal_norm_median",
        "target_normal_norm_max",
        "runtime_ms",
        "failure_reason",
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "snapshot_checksum",
    }
)


@dataclass(frozen=True)
class PclBackendResult:
    trial_id: str
    final_transformation: np.ndarray | None
    has_converged: bool
    final_transform_finite: bool
    fitness_finite: bool
    finite_output: bool
    qualification_pass: bool
    fitness_score: float | None
    translation_update_norm_m: float | None
    rotation_update_norm_rad: float | None
    source_point_count: int
    target_point_count: int
    iteration_count: int
    correspondence_count: int
    source_normal_statistics: Mapping[str, float | int | None]
    target_normal_statistics: Mapping[str, float | int | None]
    runtime_ms: float
    failure_reason: str
    pcl_version: str
    checksums: Mapping[str, str]
    cli_exit_code: int

    def __post_init__(self) -> None:
        if self.final_transformation is not None:
            matrix = np.array(self.final_transformation, dtype=np.float64, copy=True)
            if matrix.shape != (4, 4):
                raise ValueError("PCL final transformation must be 4x4")
            matrix.setflags(write=False)
            object.__setattr__(self, "final_transformation", matrix)
        object.__setattr__(self, "checksums", MappingProxyType(dict(self.checksums)))
        object.__setattr__(
            self,
            "source_normal_statistics",
            MappingProxyType(dict(self.source_normal_statistics)),
        )
        object.__setattr__(
            self,
            "target_normal_statistics",
            MappingProxyType(dict(self.target_normal_statistics)),
        )

    @property
    def solver_failed(self) -> bool:
        return not (self.has_converged and self.finite_output)


def frozen_parameters(protocol_section: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the CLI-visible frozen PCL parameter contract."""

    normal = protocol_section["normal_estimation"]
    icp = protocol_section["icp"]
    return {
        "normal_estimation": {
            "method": str(normal["method"]),
            "k": int(normal["k"]),
        },
        "icp": {
            "maximum_correspondence_distance_m": float(
                icp["maximum_correspondence_distance_m"]
            ),
            "maximum_iterations": int(icp["maximum_iterations"]),
            "transformation_epsilon": float(icp["transformation_epsilon"]),
            "euclidean_fitness_epsilon": float(icp["euclidean_fitness_epsilon"]),
            "use_reciprocal_correspondences": bool(
                icp["use_reciprocal_correspondences"]
            ),
            "use_symmetric_objective": bool(icp["use_symmetric_objective"]),
            "enforce_same_direction_normals": bool(
                icp["enforce_same_direction_normals"]
            ),
        },
    }


def _points(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 50:
        raise ValueError(f"{label} must be Nx3 with at least 50 points")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite coordinates")
    return np.ascontiguousarray(array)


def write_binary_xyz_pcd(path: str | Path, points: np.ndarray) -> None:
    """Serialize one unmodified ordered coordinate array at the PCL float boundary."""

    array = _points(points, "PCD points").astype("<f4", copy=True)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {array.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {array.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")
    destination = Path(path)
    with destination.open("wb") as stream:
        stream.write(header)
        stream.write(array.tobytes(order="C"))


def _finite_optional(value: Any, label: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"PCL result {label} is not finite")
    return number


def validate_result_payload(
    payload: Mapping[str, Any],
    *,
    trial_id: str,
    source_count: int,
    target_count: int,
    checksums: Mapping[str, str],
    initial_transformation: np.ndarray,
    cli_exit_code: int,
) -> PclBackendResult:
    missing = REQUIRED_RESULT_FIELDS - set(payload)
    if missing:
        raise ValueError(f"PCL result fields missing: {sorted(missing)}")
    if payload["trial_id"] != trial_id or payload["backend_name"] != BACKEND_NAME:
        raise ValueError("PCL result identity mismatch")
    if int(payload["source_point_count"]) != source_count:
        raise ValueError("PCL source point count mismatch")
    if int(payload["target_point_count"]) != target_count:
        raise ValueError("PCL target point count mismatch")
    for name in (
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "snapshot_checksum",
    ):
        if str(payload[name]) != str(checksums[name]):
            raise ValueError(f"PCL checksum echo mismatch: {name}")

    finite_output = bool(payload["finite_output"])
    if bool(payload["has_converged"]) != bool(payload["has_converged_raw"]):
        raise ValueError("PCL raw and compatibility convergence fields disagree")
    if bool(payload["final_transform_finite"]) != (
        payload["final_transformation_4x4"] is not None
    ):
        raise ValueError("PCL transform finiteness field disagrees with payload")
    raw_matrix = payload["final_transformation_4x4"]
    matrix: np.ndarray | None
    if raw_matrix is None:
        matrix = None
        if finite_output:
            raise ValueError("PCL claimed finite output without a transformation")
    else:
        matrix = np.asarray(raw_matrix, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("PCL returned a non-finite transformation array")
        independent = zero_initialization_error(initial_transformation, matrix)
        translation = _finite_optional(payload["translation_update_norm_m"], "translation")
        rotation = _finite_optional(payload["rotation_update_norm_rad"], "rotation")
        if translation is None or rotation is None:
            raise ValueError("PCL omitted update metrics for a finite transformation")
        if not math.isclose(
            translation, float(independent["translation_error_m"]), abs_tol=2.0e-6
        ):
            raise ValueError("PCL translation update failed independent recomputation")
        if not math.isclose(
            rotation, float(independent["rotation_error_rad"]), abs_tol=2.0e-6
        ):
            raise ValueError("PCL rotation update failed independent recomputation")

    return PclBackendResult(
        trial_id=trial_id,
        final_transformation=matrix,
        has_converged=bool(payload["has_converged_raw"]),
        final_transform_finite=bool(payload["final_transform_finite"]),
        fitness_finite=bool(payload["fitness_finite"]),
        finite_output=finite_output,
        qualification_pass=bool(payload["qualification_pass"]),
        fitness_score=_finite_optional(payload["fitness_score"], "fitness"),
        translation_update_norm_m=_finite_optional(
            payload["translation_update_norm_m"], "translation"
        ),
        rotation_update_norm_rad=_finite_optional(
            payload["rotation_update_norm_rad"], "rotation"
        ),
        source_point_count=int(payload["source_point_count"]),
        target_point_count=int(payload["target_point_count"]),
        iteration_count=int(payload["iteration_count"]),
        correspondence_count=int(payload["correspondence_count"]),
        source_normal_statistics={
            name: payload[f"source_normal_{name}"]
            for name in (
                "finite_count",
                "zero_count",
                "nan_count",
                "norm_min",
                "norm_median",
                "norm_max",
            )
        },
        target_normal_statistics={
            name: payload[f"target_normal_{name}"]
            for name in (
                "finite_count",
                "zero_count",
                "nan_count",
                "norm_min",
                "norm_median",
                "norm_max",
            )
        },
        runtime_ms=float(payload["runtime_ms"]),
        failure_reason=str(payload["failure_reason"]),
        pcl_version=str(payload["pcl_version"]),
        checksums=checksums,
        cli_exit_code=int(cli_exit_code),
    )


def run_pcl_point_to_plane(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_transformation: np.ndarray,
    *,
    trial_id: str,
    checksums: Mapping[str, str],
    executable: str | Path,
    parameters: Mapping[str, Any],
    timeout_seconds: float = 300.0,
) -> PclBackendResult:
    """Run PCL without accepting a scene label, GT, or another backend result."""

    source = _points(source_points, "source_points")
    target = _points(target_points, "target_points")
    initial = pose_matrix(initial_transformation)
    cli = Path(executable).resolve()
    if not cli.is_file():
        raise FileNotFoundError(f"PCL CLI does not exist: {cli}")
    required_checksums = {
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "snapshot_checksum",
    }
    if set(checksums) != required_checksums:
        raise ValueError("PCL checksum contract changed")

    with tempfile.TemporaryDirectory(prefix="degen-lio-pcl-") as temporary:
        work = Path(temporary)
        source_path = work / "source.pcd"
        target_path = work / "target.pcd"
        config_path = work / "config.json"
        write_binary_xyz_pcd(source_path, source)
        write_binary_xyz_pcd(target_path, target)
        config = {
            "trial_id": str(trial_id),
            "source_path": source_path.name,
            "target_path": target_path.name,
            "initial_transformation_4x4": initial.reshape(-1).tolist(),
            "parameters": json.loads(json.dumps(parameters, allow_nan=False)),
            **{key: str(value) for key, value in checksums.items()},
        }
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [str(cli), "--config", str(config_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"PCL CLI emitted invalid JSON; stderr={completed.stderr[-1000:]!r}"
        ) from exc
    return validate_result_payload(
        payload,
        trial_id=str(trial_id),
        source_count=len(source),
        target_count=len(target),
        checksums=checksums,
        initial_transformation=initial,
        cli_exit_code=completed.returncode,
    )


__all__ = [
    "BACKEND_NAME",
    "PclBackendResult",
    "REQUIRED_RESULT_FIELDS",
    "frozen_parameters",
    "run_pcl_point_to_plane",
    "validate_result_payload",
    "write_binary_xyz_pcd",
]
