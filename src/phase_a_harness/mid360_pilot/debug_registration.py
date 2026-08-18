"""Nonformal Open3D/PCL debug registration for the frozen Mid-360 pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from phase_a_harness.common_association_analysis import (
    prepare_common_association_context,
    safe_analyze_estimated_transform,
)
from phase_a_harness.open3d_backend import run_open3d_full, validate_open3d_version
from phase_a_harness.pcl_backend import frozen_parameters, run_pcl_point_to_plane
from phase_a_harness.rotation_metrics import rotation_metric_audit

from . import NONFORMAL_MARKER, PILOT_FLAGS
from .bag_reader import LIDAR_TOPIC, PilotBagError, iter_topic_messages, sha256_file
from .lidar_adapter import lidar_message_to_structured, xyz_array
from .static_map import finite_range_filter


EXPECTED_BAG_SHA256 = "a0c3488e8e8f920e15ac55ce8ea6b4faa77cea3e9b8cfeb4e64e1c69494b5508"
EXPECTED_TARGET_NPY_SHA256 = "7f01864633c56f175ece27f5996ab6df6f30457409ac5ebeb1a759311670f60f"
EXPECTED_QUERY_TIMESTAMPS = (
    1786887982.8384867,
    1786887983.4380908,
    1786887984.0381424,
    1786887984.6381790,
    1786887985.2381873,
    1786887985.7382357,
    1786887986.3382711,
    1786887986.9383562,
    1786887987.5382230,
    1786887988.1382859,
)
DEBUG_FLAGS = {
    **PILOT_FLAGS,
    NONFORMAL_MARKER: True,
    "FORMAL_MEASUREMENT_RESULT": False,
    "SAME_BAG_MAP_QUERY": True,
    "INDEPENDENT_ACQUISITION": False,
}
IDENTITY = np.eye(4, dtype=np.float64)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotBagError(f"expected JSON object: {path}")
    return value


def array_sha256(points: np.ndarray) -> str:
    array = np.ascontiguousarray(points, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def canonical_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != 3:
        raise PilotBagError("canonical points must have shape Nx3")
    canonical = np.ascontiguousarray(array, dtype="<f8")
    if canonical.shape[0] == 0 or not np.all(np.isfinite(canonical)):
        raise PilotBagError("canonical points must be non-empty and finite")
    return canonical


def load_canonical_npy(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.dtype("<f8"):
        raise PilotBagError(f"canonical array is not little-endian float64: {path}")
    if not array.flags.c_contiguous:
        raise PilotBagError(f"canonical array is not C-contiguous: {path}")
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise PilotBagError(f"canonical array is not finite Nx3: {path}")
    return array


def build_shared_input_trials(
    queries: Sequence[np.ndarray], target_map: np.ndarray
) -> list[dict[str, Any]]:
    if len(queries) != 10:
        raise PilotBagError("debug input contract requires exactly 10 query scans")
    target = canonical_points(target_map)
    target_sha = array_sha256(target)
    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        source = canonical_points(query)
        source_sha = array_sha256(source)
        for backend in ("open3d_point_to_plane", "pcl_point_to_plane"):
            rows.append(
                {
                    "query_index": query_index,
                    "backend": backend,
                    "source_array_sha256": source_sha,
                    "target_array_sha256": target_sha,
                    "T0": IDENTITY.tolist(),
                    **DEBUG_FLAGS,
                }
            )
    return rows


def mark_debug_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**DEBUG_FLAGS, **dict(payload)}


def _authenticate_prior_pilot(bag: Path, runtime: Path) -> dict[str, Any]:
    target_path = runtime / "pilot_target_map.npy"
    target_metadata = _read_json(runtime / "pilot_target_map_metadata.json")
    summary = _read_json(runtime / "pilot_summary.json")
    split = _read_json(runtime / "pilot_split_contract.json")
    lineage = _read_json(runtime / "pilot_map_query_lineage.json")
    selection = _read_csv(runtime / "pilot_query_selection.csv")
    actual_bag_sha = sha256_file(bag)
    actual_target_sha = sha256_file(target_path)
    selection_timestamps = tuple(float(row["timestamp"]) for row in selection)
    lineage_timestamps = tuple(float(value) for value in lineage["selected_query_timestamps"])
    summary_timestamps = tuple(float(value) for value in summary["selected_query_timestamps"])
    checks = {
        "bag_sha256_fixed": actual_bag_sha == EXPECTED_BAG_SHA256 == summary["bag_sha256"],
        "target_sha256_fixed": actual_target_sha
        == EXPECTED_TARGET_NPY_SHA256
        == target_metadata["pilot_target_map_npy_sha256"]
        == summary["target_map_sha256"],
        "query_count_exactly_10": len(selection) == 10,
        "query_timestamps_fixed": selection_timestamps
        == EXPECTED_QUERY_TIMESTAMPS
        == lineage_timestamps
        == summary_timestamps,
        "query_frame_indexes_fixed": [int(row["frame_index"]) for row in selection]
        == [int(value) for value in lineage["selected_query_frame_indexes"]],
        "split_contract_pass": split.get("status") == "PASS"
        and split.get("map_interval_relative_seconds") == [1.0, 7.0]
        and split.get("guard_gap_relative_seconds") == [7.0, 9.0]
        and split.get("query_interval_relative_seconds") == [9.0, 15.0],
        "lineage_pass": lineage.get("status") == "PASS"
        and lineage.get("map_query_message_index_intersection") == []
        and lineage.get("map_query_timestamp_intersection") == []
        and lineage.get("guard_gap_used") is False,
        "prior_pilot_ready": summary.get("MID360_SINGLE_BAG_PILOT_READY") is True,
        "formal_measurement_remains_false": summary.get("FORMAL_MEASUREMENT_RESULT") is False,
    }
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise PilotBagError(f"PILOT_DEBUG_REGISTRATION_READY=false: input authentication failed: {failed}")
    target = load_canonical_npy(target_path)
    return mark_debug_artifact(
        {
            "schema": "mid360_pilot_debug_input_authentication_v1",
            "status": "PASS",
            "checks": checks,
            "bag_path": str(bag),
            "bag_sha256": actual_bag_sha,
            "target_path": str(target_path),
            "target_npy_sha256": actual_target_sha,
            "target_array_sha256": array_sha256(target),
            "target_point_count": int(target.shape[0]),
            "query_selection_sha256": sha256_file(runtime / "pilot_query_selection.csv"),
            "split_contract_sha256": sha256_file(runtime / "pilot_split_contract.json"),
            "lineage_sha256": sha256_file(runtime / "pilot_map_query_lineage.json"),
            "geometry_metrics_sha256": sha256_file(runtime / "pilot_geometry_metrics.csv"),
            "query_timestamps": list(selection_timestamps),
            "query_frame_indexes": [int(row["frame_index"]) for row in selection],
        }
    )


def export_debug_inputs(
    bag_path: Path, runtime_root: Path, config_path: Path
) -> dict[str, Any]:
    bag = bag_path.expanduser().resolve(strict=True)
    runtime = runtime_root.expanduser().resolve(strict=True)
    authentication = _authenticate_prior_pilot(bag, runtime)
    debug = runtime / "debug_registration"
    if debug.exists():
        raise PilotBagError(f"debug directory already exists; refusing overwrite: {debug}")
    debug.mkdir(parents=True)
    config = _read_json(config_path)
    selection = _read_csv(runtime / "pilot_query_selection.csv")
    selected = {int(row["frame_index"]): row for row in selection}
    loaded: dict[int, np.ndarray] = {}
    for frame_index, message, _, timestamp in iter_topic_messages(bag, LIDAR_TOPIC):
        if frame_index not in selected:
            continue
        expected = float(selected[frame_index]["timestamp"])
        if timestamp != expected:
            raise PilotBagError(
                f"query timestamp drift for frame {frame_index}: {timestamp} != {expected}"
            )
        xyz = xyz_array(lidar_message_to_structured(message))
        loaded[frame_index] = canonical_points(
            finite_range_filter(
                xyz,
                minimum_range_m=float(config["map"]["minimum_range_m"]),
                maximum_range_m=float(config["map"]["maximum_range_m"]),
            )
        )
    if set(loaded) != set(selected):
        raise PilotBagError("failed to export every frozen query frame")
    target_path = runtime / "pilot_target_map.npy"
    target = load_canonical_npy(target_path)
    target_npy_sha = sha256_file(target_path)
    target_array_sha = array_sha256(target)
    manifest: list[dict[str, Any]] = []
    for row in selection:
        selection_index = int(row["selection_index"])
        query_id = f"Q{selection_index + 1:02d}"
        source = loaded[int(row["frame_index"])]
        directory = debug / "inputs" / query_id
        directory.mkdir(parents=True)
        source_path = directory / "source_points.npy"
        np.save(source_path, source, allow_pickle=False)
        reloaded = load_canonical_npy(source_path)
        manifest.append(
            {
                "query_id": query_id,
                "selection_index": selection_index,
                "query_timestamp": float(row["timestamp"]),
                "frame_index": int(row["frame_index"]),
                "source_path": str(source_path),
                "source_point_count": int(reloaded.shape[0]),
                "source_dtype": reloaded.dtype.str,
                "source_c_contiguous": bool(reloaded.flags.c_contiguous),
                "source_finite": bool(np.all(np.isfinite(reloaded))),
                "source_sha256": sha256_file(source_path),
                "source_array_sha256": array_sha256(reloaded),
                "target_path": str(target_path),
                "target_point_count": int(target.shape[0]),
                "target_dtype": target.dtype.str,
                "target_c_contiguous": bool(target.flags.c_contiguous),
                "target_finite": bool(np.all(np.isfinite(target))),
                "target_sha256": target_npy_sha,
                "target_array_sha256": target_array_sha,
                "T0": IDENTITY.tolist(),
                **DEBUG_FLAGS,
            }
        )
    _write_json(debug / "pilot_debug_input_authentication.json", authentication)
    _write_csv(debug / "pilot_debug_input_manifest.csv", manifest)
    return mark_debug_artifact(
        {
            "schema": "mid360_pilot_debug_input_export_v1",
            "status": "PASS",
            "debug_root": str(debug),
            "query_count": len(manifest),
            "target_sha256": target_npy_sha,
        }
    )


def transform_update(t0: np.ndarray, estimated: np.ndarray) -> dict[str, Any]:
    initial = np.asarray(t0, dtype=np.float64)
    result = np.asarray(estimated, dtype=np.float64)
    if initial.shape != (4, 4) or result.shape != (4, 4):
        raise PilotBagError("T0 and T_est must be 4x4")
    delta = np.linalg.inv(initial) @ result
    rotation_audit = rotation_metric_audit(delta[:3, :3], np.eye(3))
    if not rotation_audit["rotation_matrix_quality_pass"]:
        raise PilotBagError("estimated transform failed reflection-safe rotation quality")
    translation = delta[:3, 3]
    rotation_rad = float(rotation_audit["rotation_error_rad"])
    return {
        "Delta_T": delta.tolist(),
        "translation_x_m": float(translation[0]),
        "translation_y_m": float(translation[1]),
        "translation_z_m": float(translation[2]),
        "translation_norm_m": float(np.linalg.norm(translation)),
        "rotation_angle_rad": rotation_rad,
        "rotation_angle_deg": float(math.degrees(rotation_rad)),
        "rotation_matrix_quality_pass": True,
        "rotation_determinant": float(rotation_audit["determinant"]),
        "rotation_orthogonality_defect_fro": float(
            rotation_audit["orthogonality_defect_fro"]
        ),
    }


def _pcl_checksums(source: np.ndarray, target: np.ndarray, query_id: str) -> dict[str, str]:
    source_sha = array_sha256(source)
    target_sha = array_sha256(target)
    reference_sha = hashlib.sha256(IDENTITY.astype("<f8").tobytes()).hexdigest()
    snapshot_sha = hashlib.sha256(
        f"mid360-debug|{query_id}|{source_sha}|{target_sha}|{reference_sha}".encode()
    ).hexdigest()
    return {
        "source_checksum": source_sha,
        "target_checksum": target_sha,
        "reference_pose_checksum": reference_sha,
        "snapshot_checksum": snapshot_sha,
    }


def _open3d_config(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "registration_method": "point_to_plane",
        "maximum_correspondence_distance_m": float(
            parameters["maximum_correspondence_distance_m"]
        ),
        "target_normal_estimation": dict(parameters["target_normal_estimation"]),
        "icp_convergence": dict(parameters["convergence"]),
    }


def _tiny_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.linspace(-1.0, 1.0, 20)
    first, second = np.meshgrid(values, values)
    pairs = np.column_stack((first.ravel(), second.ravel()))
    faces: list[np.ndarray] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            face = np.empty((pairs.shape[0], 3), dtype=np.float64)
            face[:, axis] = sign
            other = [index for index in range(3) if index != axis]
            face[:, other] = pairs
            faces.append(face)
    target = canonical_points(np.concatenate(faces, axis=0))
    known = np.eye(4, dtype=np.float64)
    known[:3, :3] = Rotation.from_rotvec([0.01, -0.008, 0.006]).as_matrix()
    known[:3, 3] = [0.03, -0.02, 0.015]
    source = canonical_points((target - known[:3, 3]) @ known[:3, :3])
    return source, target, known


def verify_transform_convention(
    parameter_contract: Mapping[str, Any], pcl_executable: Path
) -> dict[str, Any]:
    source, target, known = _tiny_fixture()
    checksums = _pcl_checksums(source, target, "TINY_CONVENTION_FIXTURE")
    open3d = run_open3d_full(
        source,
        target,
        IDENTITY,
        _open3d_config(parameter_contract["open3d"]["parameters"]),
        backend_seed=0,
        input_checksum=checksums["snapshot_checksum"],
    )
    pcl = run_pcl_point_to_plane(
        source,
        target,
        IDENTITY,
        trial_id="mid360-pilot-tiny-transform-convention",
        checksums=checksums,
        executable=pcl_executable,
        parameters=frozen_parameters(parameter_contract["pcl"]["parameters"]),
    )
    estimates = {
        "open3d": np.asarray(open3d.final_pose, dtype=np.float64),
        "pcl": np.asarray(pcl.final_transformation, dtype=np.float64),
    }
    backend_rows: dict[str, Any] = {}
    for backend, estimate in estimates.items():
        known_error = float(np.linalg.norm(estimate - known, ord="fro"))
        inverse_error = float(np.linalg.norm(estimate - np.linalg.inv(known), ord="fro"))
        backend_rows[backend] = {
            "solver_success": bool(
                open3d.solver_converged if backend == "open3d" else pcl.has_converged
            ),
            "finite_result": bool(np.all(np.isfinite(estimate))),
            "estimated_transform": estimate.tolist(),
            "frobenius_error_to_source_to_target": known_error,
            "frobenius_error_to_inverse": inverse_error,
            "source_to_target_verified": known_error < 1.0e-5 and known_error < inverse_error,
        }
    verified = all(row["source_to_target_verified"] for row in backend_rows.values())
    return mark_debug_artifact(
        {
            "schema": "mid360_pilot_debug_transform_convention_v1",
            "status": "PASS" if verified else "FAIL",
            "definition": "T_est maps source/query points into target/map frame",
            "delta_definition": "Delta_T = inv(T0) @ T_est",
            "known_source_to_target_transform": known.tolist(),
            "backends": backend_rows,
            "transform_convention_verified": verified,
        }
    )


def _base_result(manifest: Mapping[str, str], backend: str) -> dict[str, Any]:
    return {
        "query_id": manifest["query_id"],
        "query_timestamp": float(manifest["query_timestamp"]),
        "backend": backend,
        "source_sha256": manifest["source_sha256"],
        "source_array_sha256": manifest["source_array_sha256"],
        "target_sha256": manifest["target_sha256"],
        "target_array_sha256": manifest["target_array_sha256"],
        "source_point_count": int(manifest["source_point_count"]),
        "target_point_count": int(manifest["target_point_count"]),
        "T0": IDENTITY.tolist(),
        **DEBUG_FLAGS,
    }


def _failure_result(base: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    return {
        **base,
        "T_est": None,
        "Delta_T": None,
        "translation_x_m": None,
        "translation_y_m": None,
        "translation_z_m": None,
        "translation_norm_m": None,
        "rotation_angle_rad": None,
        "rotation_angle_deg": None,
        "rotation_matrix_quality_pass": False,
        "rotation_determinant": None,
        "rotation_orthogonality_defect_fro": None,
        "solver_success": False,
        "finite_result": False,
        "iteration_count": None,
        "correspondence_count": None,
        "fitness": None,
        "inlier_rmse": None,
        "failure_reason": f"{type(error).__name__}: {error}",
    }


def _run_open3d_trial(
    source: np.ndarray,
    target: np.ndarray,
    manifest: Mapping[str, str],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    base = _base_result(manifest, "open3d_point_to_plane")
    try:
        result = run_open3d_full(
            source,
            target,
            IDENTITY,
            _open3d_config(parameters),
            backend_seed=0,
            input_checksum=str(manifest["source_array_sha256"]),
        )
        estimate = np.asarray(result.final_pose, dtype=np.float64)
        return {
            **base,
            "T_est": estimate.tolist(),
            **transform_update(IDENTITY, estimate),
            "solver_success": bool(result.solver_converged),
            "finite_result": bool(result.finite_result and np.all(np.isfinite(estimate))),
            "iteration_count": int(result.iteration_count),
            "correspondence_count": int(result.correspondence_count),
            "fitness": float(result.extra["fitness"]),
            "inlier_rmse": float(result.extra["inlier_rmse"]),
            "open3d_version": str(result.extra["open3d_version"]),
            "failure_reason": str(result.failure_reason),
        }
    except Exception as error:
        return _failure_result(base, error)


def _run_pcl_trial(
    source: np.ndarray,
    target: np.ndarray,
    manifest: Mapping[str, str],
    parameters: Mapping[str, Any],
    pcl_executable: Path,
) -> dict[str, Any]:
    base = _base_result(manifest, "pcl_point_to_plane")
    try:
        result = run_pcl_point_to_plane(
            source,
            target,
            IDENTITY,
            trial_id=f"mid360-pilot-debug-{manifest['query_id']}-pcl",
            checksums=_pcl_checksums(source, target, manifest["query_id"]),
            executable=pcl_executable,
            parameters=frozen_parameters(parameters),
        )
        if result.final_transformation is None:
            raise PilotBagError("PCL returned no finite transform")
        estimate = np.asarray(result.final_transformation, dtype=np.float64)
        return {
            **base,
            "T_est": estimate.tolist(),
            **transform_update(IDENTITY, estimate),
            "solver_success": bool(result.has_converged),
            "finite_result": bool(result.finite_output and np.all(np.isfinite(estimate))),
            "iteration_count": int(result.iteration_count),
            "correspondence_count": int(result.correspondence_count),
            "fitness": result.fitness_score,
            "inlier_rmse": None,
            "pcl_version": str(result.pcl_version),
            "pcl_executable_path": str(pcl_executable.resolve()),
            "pcl_executable_sha256": sha256_file(pcl_executable.resolve()),
            "failure_reason": str(result.failure_reason),
        }
    except Exception as error:
        return _failure_result(base, error)


def _parameter_sha(parameters: Mapping[str, Any]) -> str:
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _input_audit(
    manifest: Sequence[Mapping[str, str]],
    open3d_rows: Sequence[Mapping[str, Any]],
    pcl_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    open_by_id = {row["query_id"]: row for row in open3d_rows}
    pcl_by_id = {row["query_id"]: row for row in pcl_rows}
    output: list[dict[str, Any]] = []
    for source in manifest:
        query_id = source["query_id"]
        open_row = open_by_id[query_id]
        pcl_row = pcl_by_id[query_id]
        source_match = open_row["source_sha256"] == pcl_row["source_sha256"] == source["source_sha256"]
        target_match = open_row["target_sha256"] == pcl_row["target_sha256"] == source["target_sha256"]
        array_source_match = open_row["source_array_sha256"] == pcl_row["source_array_sha256"] == source["source_array_sha256"]
        array_target_match = open_row["target_array_sha256"] == pcl_row["target_array_sha256"] == source["target_array_sha256"]
        output.append(
            {
                "query_id": query_id,
                "query_timestamp": float(source["query_timestamp"]),
                "open3d_source_sha256": open_row["source_sha256"],
                "pcl_source_sha256": pcl_row["source_sha256"],
                "open3d_target_sha256": open_row["target_sha256"],
                "pcl_target_sha256": pcl_row["target_sha256"],
                "open3d_source_array_sha256": open_row["source_array_sha256"],
                "pcl_source_array_sha256": pcl_row["source_array_sha256"],
                "open3d_target_array_sha256": open_row["target_array_sha256"],
                "pcl_target_array_sha256": pcl_row["target_array_sha256"],
                "source_npy_sha_identical": source_match,
                "target_npy_sha_identical": target_match,
                "source_array_sha_identical": array_source_match,
                "target_array_sha_identical": array_target_match,
                "input_identity_pass": source_match
                and target_match
                and array_source_match
                and array_target_match,
                **DEBUG_FLAGS,
            }
        )
    return output


def _reassociation_rows(
    manifest: Sequence[Mapping[str, str]],
    target: np.ndarray,
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_query: dict[str, list[Mapping[str, Any]]] = {}
    for row in results:
        by_query.setdefault(str(row["query_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for source_row in manifest:
        query_id = source_row["query_id"]
        source = load_canonical_npy(Path(source_row["source_path"]))
        context = prepare_common_association_context(
            source, target, IDENTITY, snapshot_id=query_id
        )
        for result in by_query[query_id]:
            if result["T_est"] is None:
                record = {
                    "snapshot_id": query_id,
                    "common_association_valid": False,
                    "common_association_invalid_reason": "OTHER",
                }
            else:
                record = safe_analyze_estimated_transform(
                    context,
                    np.asarray(result["T_est"], dtype=np.float64),
                    identifiers={
                        "query_id": query_id,
                        "query_timestamp": float(source_row["query_timestamp"]),
                        "backend": result["backend"],
                    },
                )
            output.append(
                {
                    "query_id": query_id,
                    "query_timestamp": float(source_row["query_timestamp"]),
                    "backend": result["backend"],
                    **record,
                    **DEBUG_FLAGS,
                }
            )
    return output


def _describe(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "min": None, "max": None, "q95": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _rho(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def _statistics(
    open3d_rows: Sequence[Mapping[str, Any]],
    pcl_rows: Sequence[Mapping[str, Any]],
    reassociation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    backend_rows = {"open3d": list(open3d_rows), "pcl": list(pcl_rows)}
    backend_stats: dict[str, Any] = {}
    for backend, rows in backend_rows.items():
        finite = [row for row in rows if row["finite_result"] is True]
        backend_reassoc = [
            row for row in reassociation if str(row["backend"]).startswith(backend)
        ]
        translations = [float(row["translation_norm_m"]) for row in finite]
        rotations = [float(row["rotation_angle_rad"]) for row in finite]
        turnovers = [
            float(row["correspondence_turnover"])
            for row in backend_reassoc
            if row.get("correspondence_turnover") is not None
        ]
        paired_translation = [
            float(row["translation_norm_m"])
            for row in finite
            if next(
                item for item in backend_reassoc if item["query_id"] == row["query_id"]
            ).get("correspondence_turnover")
            is not None
        ]
        backend_stats[backend] = {
            "translation_norm_m": _describe(translations),
            "rotation_angle_rad": _describe(rotations),
            "rotation_angle_deg": _describe([math.degrees(value) for value in rotations]),
            "correspondence_turnover": _describe(turnovers),
            "turnover_translation_spearman_rho": _rho(turnovers, paired_translation),
            "solver_success_count": sum(row["solver_success"] is True for row in rows),
            "finite_result_count": sum(row["finite_result"] is True for row in rows),
        }
    paired = [
        (open_row, pcl_row)
        for open_row, pcl_row in zip(open3d_rows, pcl_rows)
        if open_row["finite_result"] and pcl_row["finite_result"]
    ]
    open_translation = [float(pair[0]["translation_norm_m"]) for pair in paired]
    pcl_translation = [float(pair[1]["translation_norm_m"]) for pair in paired]
    direction_rows: list[dict[str, Any]] = []
    for open_row, pcl_row in paired:
        first = np.asarray(
            [open_row["translation_x_m"], open_row["translation_y_m"], open_row["translation_z_m"]],
            dtype=np.float64,
        )
        second = np.asarray(
            [pcl_row["translation_x_m"], pcl_row["translation_y_m"], pcl_row["translation_z_m"]],
            dtype=np.float64,
        )
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        cosine = None if denominator <= 1.0e-15 else float(np.dot(first, second) / denominator)
        direction_rows.append(
            {"query_id": open_row["query_id"], "translation_direction_cosine": cosine}
        )
    cosine_values = [
        row["translation_direction_cosine"]
        for row in direction_rows
        if row["translation_direction_cosine"] is not None
    ]
    rho = _rho(open_translation, pcl_translation)
    return mark_debug_artifact(
        {
            "schema": "mid360_pilot_debug_statistics_v1",
            "backend_statistics": backend_stats,
            "open3d_pcl_translation_spearman_rho": rho,
            "per_query_direction_consistency": direction_rows,
            "translation_direction_cosine": _describe(cosine_values),
            "rough_displacement_trend_consistent": bool(
                rho is not None
                and rho >= 0.5
                and cosine_values
                and float(np.median(cosine_values)) >= 0.5
            ),
        }
    )


def _statistics_markdown(statistics: Mapping[str, Any]) -> str:
    lines = [
        "# Mid-360 Open3D/PCL debug statistics",
        "",
        "`PILOT_NONFORMAL_DO_NOT_CITE=true`; these are software-chain diagnostics only.",
        "",
    ]
    for backend in ("open3d", "pcl"):
        stats = statistics["backend_statistics"][backend]
        translation = stats["translation_norm_m"]
        rotation = stats["rotation_angle_deg"]
        turnover = stats["correspondence_turnover"]
        lines.extend(
            [
                f"## {backend}",
                "",
                f"- Translation median/min/max/q95 (m): {translation['median']}, {translation['min']}, {translation['max']}, {translation['q95']}",
                f"- Rotation median/max (deg): {rotation['median']}, {rotation['max']}",
                f"- Correspondence turnover median/min/max: {turnover['median']}, {turnover['min']}, {turnover['max']}",
                "",
            ]
        )
    lines.append(
        f"Open3D/PCL translation Spearman rho: {statistics['open3d_pcl_translation_spearman_rho']}"
    )
    return "\n".join(lines) + "\n"


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    return f"""# Mid-360 nonformal debug registration summary

`PILOT_NONFORMAL_DO_NOT_CITE=true`, `FORMAL_MEASUREMENT_RESULT=false`, `SAME_BAG_MAP_QUERY=true`, `INDEPENDENT_ACQUISITION=false`.

- Open3D success: {summary['open3d_solver_success_count']}/10
- PCL success: {summary['pcl_solver_success_count']}/10
- Finite results: {summary['finite_result_count']}/20
- Backend input identity: {summary['backend_input_identity_pass']}
- Transform convention verified: {summary['transform_convention_verified']}
- Obvious unit/frame bug detected: {summary['obvious_unit_error_detected'] or summary['frame_error_detected']}
- Readiness class: {summary['readiness_class']}
- MID360_OPEN3D_PCL_DEBUG_READY: {summary['MID360_OPEN3D_PCL_DEBUG_READY']}

These 20 same-bag trials are software-chain debugging and cannot be used as formal measurement results.
"""


def _write_sha256s(debug: Path) -> None:
    paths = sorted(
        path for path in debug.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(debug).as_posix()}\n" for path in paths
    ]
    (debug / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def execute_debug_registration(
    runtime_root: Path,
    parameter_contract_path: Path,
    pcl_executable: Path,
) -> dict[str, Any]:
    runtime = runtime_root.expanduser().resolve(strict=True)
    debug = (runtime / "debug_registration").resolve(strict=True)
    protected_outputs = (
        "pilot_debug_transform_convention.json",
        "pilot_debug_open3d_results.csv",
        "pilot_debug_pcl_results.csv",
        "pilot_debug_backend_input_audit.csv",
        "pilot_debug_reassociation.csv",
        "pilot_debug_statistics.json",
        "pilot_debug_readiness.json",
        "pilot_debug_summary.json",
        "SHA256SUMS",
    )
    existing_outputs = [name for name in protected_outputs if (debug / name).exists()]
    if existing_outputs:
        raise PilotBagError(
            "debug backend outputs already exist; refusing retry/overwrite: "
            + ", ".join(existing_outputs)
        )
    manifest = _read_csv(debug / "pilot_debug_input_manifest.csv")
    if len(manifest) != 10 or tuple(float(row["query_timestamp"]) for row in manifest) != EXPECTED_QUERY_TIMESTAMPS:
        raise PilotBagError("frozen debug input manifest is incomplete or changed")
    target_path = runtime / "pilot_target_map.npy"
    target = load_canonical_npy(target_path)
    if sha256_file(target_path) != EXPECTED_TARGET_NPY_SHA256:
        raise PilotBagError("target map SHA drift before backend execution")
    for row in manifest:
        source_path = Path(row["source_path"])
        source = load_canonical_npy(source_path)
        if (
            sha256_file(source_path) != row["source_sha256"]
            or array_sha256(source) != row["source_array_sha256"]
            or row["target_sha256"] != EXPECTED_TARGET_NPY_SHA256
            or json.loads(row["T0"]) != IDENTITY.tolist()
        ):
            raise PilotBagError(f"canonical input drift: {row['query_id']}")
    parameter_contract = _read_json(parameter_contract_path)
    for backend in ("open3d", "pcl"):
        if _parameter_sha(parameter_contract[backend]["parameters"]) != parameter_contract[backend]["canonical_sha256"]:
            raise PilotBagError(f"frozen {backend} parameter SHA mismatch")
    if validate_open3d_version() != parameter_contract["open3d"]["parameters"]["version"]:
        raise PilotBagError("Open3D version does not match frozen parameter contract")
    pcl_cli = pcl_executable.resolve(strict=True)
    convention = verify_transform_convention(parameter_contract, pcl_cli)
    _write_json(debug / "pilot_debug_transform_convention.json", convention)
    if convention["status"] != "PASS":
        raise PilotBagError("transform convention verification failed")
    open3d_rows: list[dict[str, Any]] = []
    pcl_rows: list[dict[str, Any]] = []
    sources = {
        row["query_id"]: load_canonical_npy(Path(row["source_path"])) for row in manifest
    }
    for row in manifest:
        open3d_rows.append(
            _run_open3d_trial(
                sources[row["query_id"]],
                target,
                row,
                parameter_contract["open3d"]["parameters"],
            )
        )
    for row in manifest:
        pcl_rows.append(
            _run_pcl_trial(
                sources[row["query_id"]],
                target,
                row,
                parameter_contract["pcl"]["parameters"],
                pcl_cli,
            )
        )
    _write_csv(debug / "pilot_debug_open3d_results.csv", open3d_rows)
    _write_csv(debug / "pilot_debug_pcl_results.csv", pcl_rows)
    input_audit = _input_audit(manifest, open3d_rows, pcl_rows)
    _write_csv(debug / "pilot_debug_backend_input_audit.csv", input_audit)
    reassociation = _reassociation_rows(manifest, target, [*open3d_rows, *pcl_rows])
    _write_csv(debug / "pilot_debug_reassociation.csv", reassociation)
    statistics = _statistics(open3d_rows, pcl_rows, reassociation)
    _write_json(debug / "pilot_debug_statistics.json", statistics)
    (debug / "pilot_debug_statistics.md").write_text(
        _statistics_markdown(statistics), encoding="utf-8"
    )
    all_results = [*open3d_rows, *pcl_rows]
    finite_count = sum(row["finite_result"] is True for row in all_results)
    solver_failures = [
        f"{row['query_id']}:{row['backend']}" for row in all_results if not row["solver_success"]
    ]
    translation_suspects = [
        f"{row['query_id']}:{row['backend']}"
        for row in all_results
        if row["translation_norm_m"] is not None and float(row["translation_norm_m"]) > 1.0
    ]
    rotation_suspects = [
        f"{row['query_id']}:{row['backend']}"
        for row in all_results
        if row["rotation_angle_deg"] is not None and float(row["rotation_angle_deg"]) > 10.0
    ]
    opposite_large = [
        row["query_id"]
        for row in statistics["per_query_direction_consistency"]
        if row["translation_direction_cosine"] is not None
        and float(row["translation_direction_cosine"]) < -0.5
        and next(float(item["translation_norm_m"]) for item in open3d_rows if item["query_id"] == row["query_id"]) > 0.01
        and next(float(item["translation_norm_m"]) for item in pcl_rows if item["query_id"] == row["query_id"]) > 0.01
    ]
    input_identity = all(row["input_identity_pass"] is True for row in input_audit)
    obvious_bug = bool(translation_suspects or rotation_suspects or opposite_large)
    hard_ready = bool(
        len(open3d_rows) == len(pcl_rows) == 10
        and finite_count == 20
        and input_identity
        and convention["transform_convention_verified"] is True
        and not obvious_bug
    )
    documented_limitations = ["ACCELERATION_UNIT_UNCONFIRMED"]
    readiness_class = "READY_WITH_LIMITATION" if hard_ready else "FAIL"
    readiness = mark_debug_artifact(
        {
            "schema": "mid360_pilot_debug_readiness_v1",
            "readiness_class": readiness_class,
            "MID360_OPEN3D_PCL_DEBUG_READY": hard_ready,
            "PILOT_DEBUG_REGISTRATION_READY": hard_ready,
            "open3d_input_valid_count": len(open3d_rows),
            "pcl_input_valid_count": len(pcl_rows),
            "finite_result_count": finite_count,
            "solver_failures": solver_failures,
            "backend_input_identity_pass": input_identity,
            "transform_convention_verified": convention["transform_convention_verified"],
            "translation_over_1m_suspects": translation_suspects,
            "rotation_over_10deg_suspects": rotation_suspects,
            "opposite_large_direction_suspects": opposite_large,
            "obvious_unit_or_frame_bug_detected": obvious_bug,
            "acceleration_unit_still_unknown": True,
            "documented_limitations": documented_limitations,
            "status": "PASS" if hard_ready else "FAIL",
        }
    )
    _write_json(debug / "pilot_debug_readiness.json", readiness)
    largest_translation = max(
        (row for row in all_results if row["translation_norm_m"] is not None),
        key=lambda row: float(row["translation_norm_m"]),
    )
    largest_rotation = max(
        (row for row in all_results if row["rotation_angle_rad"] is not None),
        key=lambda row: float(row["rotation_angle_rad"]),
    )
    summary = mark_debug_artifact(
        {
            "schema": "mid360_pilot_debug_summary_v1",
            "readiness_class": readiness_class,
            "MID360_OPEN3D_PCL_DEBUG_READY": hard_ready,
            "open3d_solver_success_count": sum(row["solver_success"] is True for row in open3d_rows),
            "pcl_solver_success_count": sum(row["solver_success"] is True for row in pcl_rows),
            "solver_failures": solver_failures,
            "documented_limitations": documented_limitations,
            "finite_result_count": finite_count,
            "nonfinite_result_count": 20 - finite_count,
            "backend_input_identity_pass": input_identity,
            "transform_convention_verified": convention["transform_convention_verified"],
            "statistics": statistics,
            "largest_translation": {
                "query_id": largest_translation["query_id"],
                "backend": largest_translation["backend"],
                "value_m": largest_translation["translation_norm_m"],
            },
            "largest_rotation": {
                "query_id": largest_rotation["query_id"],
                "backend": largest_rotation["backend"],
                "value_rad": largest_rotation["rotation_angle_rad"],
                "value_deg": largest_rotation["rotation_angle_deg"],
            },
            "obvious_unit_error_detected": bool(translation_suspects),
            "obvious_unit_or_frame_bug_detected": bool(
                translation_suspects or rotation_suspects or opposite_large
            ),
            "transform_direction_error_detected": False,
            "frame_error_detected": bool(rotation_suspects or opposite_large),
            "backend_input_mismatch_detected": not input_identity,
            "formal_mid360_capture_program_change_required": False,
            "formal_recording_software_ready": hard_ready,
            "formal_result_eligibility": "NO",
            "pcl_executable_path": str(pcl_cli),
            "pcl_executable_sha256": sha256_file(pcl_cli),
            "pcl_version": next(
                (
                    row["pcl_version"]
                    for row in pcl_rows
                    if row.get("pcl_version") is not None
                ),
                None,
            ),
            "open3d_version": validate_open3d_version(),
            "backend_parameter_contract_path": str(parameter_contract_path.resolve()),
            "backend_parameter_contract_sha256": sha256_file(parameter_contract_path),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        }
    )
    _write_json(debug / "pilot_debug_summary.json", summary)
    (debug / "pilot_debug_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_sha256s(debug)
    return summary


def finalize_debug_reporting(runtime_root: Path) -> dict[str, Any]:
    """Finish report packaging after completed trials without rerunning a backend."""

    runtime = runtime_root.expanduser().resolve(strict=True)
    debug = (runtime / "debug_registration").resolve(strict=True)
    if (debug / "pilot_debug_summary.md").exists() or (debug / "SHA256SUMS").exists():
        raise PilotBagError("debug report packaging already completed; refusing overwrite")
    open3d_rows = _read_csv(debug / "pilot_debug_open3d_results.csv")
    pcl_rows = _read_csv(debug / "pilot_debug_pcl_results.csv")
    reassociation = _read_csv(debug / "pilot_debug_reassociation.csv")
    input_audit = _read_csv(debug / "pilot_debug_backend_input_audit.csv")
    summary = _read_json(debug / "pilot_debug_summary.json")
    if not (
        len(open3d_rows) == 10
        and len(pcl_rows) == 10
        and len(reassociation) == 20
        and len(input_audit) == 10
        and summary.get("finite_result_count") == 20
        and summary.get("MID360_OPEN3D_PCL_DEBUG_READY") is True
    ):
        raise PilotBagError("persisted debug results are incomplete; report-only finalize rejected")
    summary["obvious_unit_or_frame_bug_detected"] = bool(
        summary.get("obvious_unit_error_detected") or summary.get("frame_error_detected")
    )
    summary["report_finalize_mode"] = "POSTPROCESS_ONLY_NO_BACKEND_RERUN"
    _write_json(debug / "pilot_debug_summary.json", summary)
    (debug / "pilot_debug_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_sha256s(debug)
    return summary


def apply_documented_readiness_limitation(runtime_root: Path) -> dict[str, Any]:
    """Apply the pre-existing unknown-acceleration limitation without rerunning ICP."""

    runtime = runtime_root.expanduser().resolve(strict=True)
    debug = (runtime / "debug_registration").resolve(strict=True)
    readiness = _read_json(debug / "pilot_debug_readiness.json")
    summary = _read_json(debug / "pilot_debug_summary.json")
    if not (
        readiness.get("MID360_OPEN3D_PCL_DEBUG_READY") is True
        and readiness.get("acceleration_unit_still_unknown") is True
        and summary.get("MID360_OPEN3D_PCL_DEBUG_READY") is True
        and summary.get("finite_result_count") == 20
    ):
        raise PilotBagError("readiness limitation amendment preconditions failed")
    limitations = ["ACCELERATION_UNIT_UNCONFIRMED"]
    readiness["readiness_class"] = "READY_WITH_LIMITATION"
    readiness["documented_limitations"] = limitations
    readiness["classification_amendment_mode"] = "REPORT_ONLY_NO_BACKEND_RERUN"
    summary["readiness_class"] = "READY_WITH_LIMITATION"
    summary["documented_limitations"] = limitations
    summary["classification_amendment_mode"] = "REPORT_ONLY_NO_BACKEND_RERUN"
    _write_json(debug / "pilot_debug_readiness.json", readiness)
    _write_json(debug / "pilot_debug_summary.json", summary)
    (debug / "pilot_debug_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_sha256s(debug)
    return summary


__all__ = [
    "DEBUG_FLAGS",
    "EXPECTED_BAG_SHA256",
    "EXPECTED_QUERY_TIMESTAMPS",
    "EXPECTED_TARGET_NPY_SHA256",
    "array_sha256",
    "apply_documented_readiness_limitation",
    "build_shared_input_trials",
    "canonical_points",
    "execute_debug_registration",
    "export_debug_inputs",
    "finalize_debug_reporting",
    "load_canonical_npy",
    "mark_debug_artifact",
    "transform_update",
    "verify_transform_convention",
]
