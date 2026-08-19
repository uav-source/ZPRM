"""Controlled initial-translation perturbations on the frozen Mid-360 Pilot.

This module is deliberately outside production registration code.  It reuses
the frozen point arrays, the existing geometry association/normal machinery,
and the existing Open3D/PCL wrappers without changing their algorithms or
parameters.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from phase_a_harness.common_association_analysis import associate_source_points
from phase_a_harness.mid360_pilot.bag_reader import PilotBagError, sha256_file
from phase_a_harness.mid360_pilot.debug_registration import (
    IDENTITY,
    _open3d_config,
    _parameter_sha,
    array_sha256,
    load_canonical_npy,
)
from phase_a_harness.mid360_two_scene_pilot.pipeline import read_csv, read_json
from phase_a_harness.open3d_backend import run_open3d_full, validate_open3d_version
from phase_a_harness.pcl_backend import frozen_parameters, run_pcl_point_to_plane
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    TargetGeometryContext,
)
from phase_a_harness.rotation_metrics import rotation_metric_audit


SCHEMA = "mid360_controlled_perturbation_pilot_v1"
BACKEND_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
PCL_EXECUTABLE_SHA256 = (
    "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff"
)
ZERO_RUNTIME = Path(
    "/home/lj/ZPRM/zero_perturbation_runtime/real_data/mid360_two_scene_pilot_v1"
)
BAG_ROOT = Path("/home/lj/ZPRM/bag")
MAGNITUDES_M = (0.01, 0.03, 0.05)
SIGNS = (1, -1)
DIRECTION_CLASSES = ("weak", "strong")
BACKENDS = ("open3d", "pcl")
RECOVERY_TRANSLATION_M = 0.005
RECOVERY_ROTATION_DEG = 0.2
GEOMETRY_EIGENVALUE_TOLERANCE = 1.0e-12
PERTURBATION_TOLERANCE_M = 1.0e-9
EXPECTED_SNAPSHOT_COUNT = 20
EXPECTED_RUN_COUNT = 480

FLAGS = {
    "PILOT_ONLY": True,
    "PILOT_NONFORMAL_DO_NOT_CITE": True,
    "FORMAL_MEASUREMENT_RESULT": False,
    "MEASUREMENT_EVIDENCE": False,
    "NO_OBVIOUS_MOTION": True,
    "ACCELERATION_UNIT_UNKNOWN": True,
    "POINT_COORDINATE_UNIT_STATUS": "ASSUMED_METERS_FROM_SCALE",
}

EXPECTED_BAGS = {
    "mid360_20260818_203200_part1_20s.bag": (
        "ec998bc44cd7f6276548cbf7c11cffdfcb7a5a5601335cf2fdddd3c8da998c13"
    ),
    "mid360_20260818_203200_part2_15s.bag": (
        "3a96dba7563f9742e8fabe5cbc7a90d77991e1d6d25dc024c1c36073af116e73"
    ),
    "mid360_20260818_205021_part1_20s.bag": (
        "d20b50bd4be04229d311717a27021502a74429fa3a143418b18ad70622eb4a6d"
    ),
    "mid360_20260818_205021_part2_15s.bag": (
        "2c9893beb56146996eccbe846dc0ee5a4a3ca9aba5b3bb71f7067c598b27c581"
    ),
}

EXPECTED_TARGETS = {
    "R_TEST_01": {
        "scene_class": "Rich",
        "scene_type": "RICH_CANDIDATE",
        "semantic_scene": "LABORATORY",
        "point_count": 136443,
        "npy_sha256": (
            "43c6f299b5b24d220e6ee1a5ad8395ebb1380f992ff4fe207f822e25c92a65f7"
        ),
    },
    "W_TEST_01": {
        "scene_class": "Weak",
        "scene_type": "WEAK_CANDIDATE",
        "semantic_scene": "LONG_CORRIDOR",
        "point_count": 158862,
        "npy_sha256": (
            "d4438aba190de47a85fb6188ddc3b39dbfd6e39d3fdb5fa264bf425912e6c02e"
        ),
    },
}

REQUIRED_OUTPUTS = (
    "manifest.json",
    "directions.csv",
    "trial_plan.csv",
    "runs.csv",
    "summary.csv",
    "rich_vs_weak.csv",
    "weak_vs_strong.csv",
    "paired_differences.csv",
    "backend_agreement.csv",
    "decision_gate.json",
    "verification_report.txt",
    "README.md",
    "01_translation_error_vs_perturbation.png",
    "02_recovery_rate_vs_perturbation.png",
    "03_weak_vs_strong_direction.png",
    "04_rich_vs_weak.png",
    "05_open3d_vs_pcl_translation.png",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise PilotBagError(f"refusing to write headerless empty CSV: {path}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def ensure_fresh_output(output: Path) -> Path:
    destination = output.expanduser().resolve()
    if destination.exists():
        raise PilotBagError(f"output already exists; refusing overwrite: {destination}")
    return destination


def _matrix(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise PilotBagError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise PilotBagError(f"{label} homogeneous row is invalid")
    return np.ascontiguousarray(result)


def _pose_sha256(pose: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pose, dtype="<f8").tobytes()).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_sha256_file(path: Path, expected: str) -> str:
    actual = sha256_file(path.resolve(strict=True))
    if actual != expected:
        raise PilotBagError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def _read_sha256s(root: Path) -> dict[str, str]:
    checksum_path = root / "SHA256SUMS"
    rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip(" *")
        if relative in rows:
            raise PilotBagError(f"duplicate runtime checksum entry: {relative}")
        rows[relative] = digest
    if not rows:
        raise PilotBagError("zero-perturbation SHA256SUMS is empty")
    return rows


def authenticate_zero_runtime(root: Path = ZERO_RUNTIME) -> dict[str, str]:
    runtime = root.resolve(strict=True)
    expected = _read_sha256s(runtime)
    actual: dict[str, str] = {}
    for relative, digest in expected.items():
        path = (runtime / relative).resolve(strict=True)
        try:
            path.relative_to(runtime)
        except ValueError as exc:
            raise PilotBagError(f"runtime checksum path escaped root: {relative}") from exc
        actual[relative] = verify_sha256_file(path, digest)
    actual["SHA256SUMS"] = sha256_file(runtime / "SHA256SUMS")
    return actual


def authenticate_bags(bag_root: Path = BAG_ROOT) -> dict[str, str]:
    root = bag_root.resolve(strict=True)
    return {
        name: verify_sha256_file(root / name, expected)
        for name, expected in EXPECTED_BAGS.items()
    }


def authenticate_contract(repository: Path) -> tuple[Path, dict[str, Any]]:
    path = repository.resolve(strict=True) / "frozen_assets/backend_parameter_contract.json"
    verify_sha256_file(path, BACKEND_CONTRACT_SHA256)
    contract = read_json(path)
    if contract.get("backend_parameter_difference_count") != 0:
        raise PilotBagError("backend parameter difference count is not zero")
    for backend in BACKENDS:
        section = contract[backend]
        if _parameter_sha(section["parameters"]) != section["canonical_sha256"]:
            raise PilotBagError(f"{backend} canonical parameter SHA mismatch")
        if section["parameters"].get("scene_specific_parameters") is not False:
            raise PilotBagError(f"{backend} scene-specific parameters are not false")
    return path, contract


def load_and_authenticate_inputs(
    runtime: Path = ZERO_RUNTIME,
) -> tuple[list[dict[str, str]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    root = runtime.resolve(strict=True)
    manifest = read_csv(root / "canonical_input_manifest.csv")
    if len(manifest) != EXPECTED_SNAPSHOT_COUNT:
        raise PilotBagError(f"expected 20 frozen snapshots, got {len(manifest)}")
    if len({row["snapshot_id"] for row in manifest}) != EXPECTED_SNAPSHOT_COUNT:
        raise PilotBagError("frozen snapshot IDs are not unique")
    if {row["scene_id"] for row in manifest} != set(EXPECTED_TARGETS):
        raise PilotBagError("frozen scene set changed")
    sources: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    per_scene = defaultdict(int)
    for row in manifest:
        scene_id = row["scene_id"]
        expected_scene = EXPECTED_TARGETS[scene_id]
        per_scene[scene_id] += 1
        if (
            row["scene_type"] != expected_scene["scene_type"]
            or row["semantic_scene"] != expected_scene["semantic_scene"]
            or json.loads(row["T0"]) != IDENTITY.tolist()
            or row["little_endian_float64"] != "true"
            or row["c_contiguous"] != "true"
            or row["finite"] != "true"
        ):
            raise PilotBagError(f"frozen manifest metadata mismatch: {row['snapshot_id']}")
        source_path = Path(row["source_path"]).resolve(strict=True)
        target_path = Path(row["target_path"]).resolve(strict=True)
        source = load_canonical_npy(source_path)
        target = targets.get(scene_id)
        if target is None:
            target = load_canonical_npy(target_path)
            targets[scene_id] = target
        checks = (
            sha256_file(source_path) == row["source_npy_sha256"],
            array_sha256(source) == row["source_array_sha256"],
            sha256_file(target_path)
            == row["target_npy_sha256"]
            == expected_scene["npy_sha256"],
            array_sha256(target) == row["target_array_sha256"],
            int(row["source_point_count"]) == source.shape[0],
            int(row["target_point_count"])
            == target.shape[0]
            == expected_scene["point_count"],
        )
        if not all(checks):
            raise PilotBagError(f"canonical input authentication failed: {row['snapshot_id']}")
        sources[row["snapshot_id"]] = source
    if dict(per_scene) != {"R_TEST_01": 10, "W_TEST_01": 10}:
        raise PilotBagError(f"frozen per-scene snapshot counts changed: {dict(per_scene)}")
    return manifest, sources, targets


def _canonicalize_eigenvector_sign(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def deterministic_eigendirections(
    information_matrix: np.ndarray,
) -> dict[str, Any]:
    """Return sorted weak/strong unit vectors for the existing 3x3 matrix."""

    matrix = np.asarray(information_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise PilotBagError("translation information matrix must be finite 3x3")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise PilotBagError("translation information matrix is not symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if np.any(eigenvalues[1:] < eigenvalues[:-1]):
        raise PilotBagError("eigenvalues are not sorted ascending")
    weak = _canonicalize_eigenvector_sign(eigenvectors[:, 0])
    strong = _canonicalize_eigenvector_sign(eigenvectors[:, 2])
    weak_norm = float(np.linalg.norm(weak))
    strong_norm = float(np.linalg.norm(strong))
    if weak_norm <= 1e-12 or strong_norm <= 1e-12:
        raise PilotBagError("weak/strong translation eigenvector norm is too small")
    weak /= weak_norm
    strong /= strong_norm
    return {
        "eigenvalues": eigenvalues,
        "weak": weak,
        "strong": strong,
    }


def construct_translation_perturbation(
    t_star: np.ndarray,
    direction: np.ndarray,
    magnitude_m: float,
    sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct target-frame translation by left multiplication: T0=Delta*T*."""

    reference = _matrix(t_star, "T_star")
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise PilotBagError("perturbation direction must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise PilotBagError("perturbation direction norm is too small")
    if sign not in (-1, 1):
        raise PilotBagError("perturbation sign must be -1 or +1")
    if magnitude_m <= 0.0 or not math.isfinite(float(magnitude_m)):
        raise PilotBagError("perturbation magnitude must be positive and finite")
    unit = vector / norm
    perturbation_vector = float(sign) * float(magnitude_m) * unit
    delta = IDENTITY.copy()
    delta[:3, 3] = perturbation_vector
    initial = delta @ reference
    actual = initial @ np.linalg.inv(reference)
    actual_magnitude = float(np.linalg.norm(actual[:3, 3]))
    if abs(actual_magnitude - float(magnitude_m)) >= PERTURBATION_TOLERANCE_M:
        raise PilotBagError("constructed perturbation magnitude is not exact")
    return delta, initial, perturbation_vector


def pose_error(estimate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Measure target-frame left pose error E=T_est*inv(T_star)."""

    estimated = _matrix(estimate, "estimated pose")
    truth = _matrix(reference, "reference pose")
    error = estimated @ np.linalg.inv(truth)
    audit = rotation_metric_audit(error[:3, :3], np.eye(3))
    if audit["rotation_matrix_quality_pass"] is not True:
        raise PilotBagError("pose rotation failed reflection-safe quality checks")
    translation = np.asarray(error[:3, 3], dtype=np.float64)
    rotation_deg = math.degrees(float(audit["rotation_error_rad"]))
    return {
        "error_transform": error,
        "translation_vector": translation,
        "translation_error_m": float(np.linalg.norm(translation)),
        "rotation_error_deg": float(rotation_deg),
        "rotation_quality_pass": True,
    }


def pose_recovered(translation_error_m: float, rotation_error_deg: float) -> bool:
    return bool(
        math.isfinite(float(translation_error_m))
        and math.isfinite(float(rotation_error_deg))
        and float(translation_error_m) <= RECOVERY_TRANSLATION_M
        and float(rotation_error_deg) <= RECOVERY_ROTATION_DEG
    )


def build_run_id(
    snapshot_id: str,
    direction_class: str,
    magnitude_m: float,
    sign: int,
    backend: str,
) -> str:
    if direction_class not in DIRECTION_CLASSES or backend not in BACKENDS:
        raise PilotBagError("invalid run ID dimension")
    if magnitude_m not in MAGNITUDES_M or sign not in SIGNS:
        raise PilotBagError("invalid run ID perturbation")
    mm = int(round(float(magnitude_m) * 1000.0))
    direction = "W" if direction_class == "weak" else "S"
    sign_label = "POS" if sign > 0 else "NEG"
    backend_label = "O3D" if backend == "open3d" else "PCL"
    return f"CP-{snapshot_id}-{direction}-{mm:02d}MM-{sign_label}-{backend_label}"


def _geometry_directions(
    manifest: Sequence[Mapping[str, str]],
    sources: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    runtime: Path,
) -> list[dict[str, Any]]:
    frozen_metrics = {
        row["snapshot_id"]: row
        for row in read_csv(runtime / "geometry_only_metrics.csv")
    }
    contexts = {
        scene_id: TargetGeometryContext.prepare(target)
        for scene_id, target in targets.items()
    }
    output: list[dict[str, Any]] = []
    for row in manifest:
        snapshot_id = row["snapshot_id"]
        scene_id = row["scene_id"]
        context = contexts[scene_id]
        source = sources[snapshot_id]
        # Reuse the same public geometry-only computation for an independent
        # eigenvalue comparison, then extract vectors from its cached normals.
        public_metrics = context.compute(source, IDENTITY)
        association = associate_source_points(
            source,
            IDENTITY,
            target_tree=context.target_tree,
            target_point_count=context.target_points.shape[0],
        )
        valid = context.target_normal_valid[association.target_indices]
        selected_normals = context.target_normals[association.target_indices[valid]]
        if selected_normals.shape[0] == 0:
            raise PilotBagError(f"no valid translation normals: {snapshot_id}")
        information = (selected_normals.T @ selected_normals) / float(
            selected_normals.shape[0]
        )
        decomposition = deterministic_eigendirections(information)
        eigenvalues = decomposition["eigenvalues"]
        frozen = frozen_metrics[snapshot_id]
        for index, name in enumerate(
            ("lambda_min_trans", "lambda_mid_trans", "lambda_max_trans")
        ):
            if not math.isclose(
                float(eigenvalues[index]),
                float(public_metrics[name]),
                rel_tol=0.0,
                abs_tol=GEOMETRY_EIGENVALUE_TOLERANCE,
            ) or not math.isclose(
                float(eigenvalues[index]),
                float(frozen[name]),
                rel_tol=0.0,
                abs_tol=GEOMETRY_EIGENVALUE_TOLERANCE,
            ):
                raise PilotBagError(
                    f"geometry eigenvalue drift for {snapshot_id} {name}"
                )
        total = max(float(np.sum(eigenvalues)), 1e-12)
        normalized = eigenvalues / total
        positive = normalized > 0.0
        entropy = -float(
            np.sum(normalized[positive] * np.log(normalized[positive]))
        ) / math.log(3.0)
        for direction_class, source_name in (
            ("weak", "lambda_min"),
            ("strong", "lambda_max"),
        ):
            vector = decomposition[direction_class]
            output.append(
                {
                    "scene_id": scene_id,
                    "scene_class": EXPECTED_TARGETS[scene_id]["scene_class"],
                    "snapshot_id": snapshot_id,
                    "geometry_matrix_definition": "H_trans=(N.T@N)/valid_normal_count",
                    "geometry_matrix_source": (
                        "phase_a_harness.real_data_preparation."
                        "boreas_v2_stage2_selection.TargetGeometryContext"
                    ),
                    "valid_normal_count": int(selected_normals.shape[0]),
                    "information_h00": float(information[0, 0]),
                    "information_h01": float(information[0, 1]),
                    "information_h02": float(information[0, 2]),
                    "information_h10": float(information[1, 0]),
                    "information_h11": float(information[1, 1]),
                    "information_h12": float(information[1, 2]),
                    "information_h20": float(information[2, 0]),
                    "information_h21": float(information[2, 1]),
                    "information_h22": float(information[2, 2]),
                    "geometry_lambda_min": float(eigenvalues[0]),
                    "geometry_lambda_mid": float(eigenvalues[1]),
                    "geometry_lambda_max": float(eigenvalues[2]),
                    "geometry_condition_number": float(
                        eigenvalues[2] / max(float(eigenvalues[0]), 1e-12)
                    ),
                    "geometry_entropy": entropy,
                    "direction_class": direction_class,
                    "direction_source": source_name,
                    "direction_vector_x": float(vector[0]),
                    "direction_vector_y": float(vector[1]),
                    "direction_vector_z": float(vector[2]),
                    "direction_vector_norm": float(np.linalg.norm(vector)),
                    "direction_sign_canonicalization": (
                        "largest_absolute_component_nonnegative"
                    ),
                    "direction_valid": True,
                    "invalid_reason": "",
                    **FLAGS,
                }
            )
    if len(output) != 40:
        raise PilotBagError(f"expected 40 direction rows, got {len(output)}")
    return output


def _build_trial_plan(
    manifest: Sequence[Mapping[str, str]],
    directions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (row["snapshot_id"], row["direction_class"]): row for row in directions
    }
    output: list[dict[str, Any]] = []
    for source in manifest:
        t_star = _matrix(json.loads(source["T0"]), "T_star")
        scene_id = source["scene_id"]
        for direction_class in DIRECTION_CLASSES:
            direction_row = by_key[(source["snapshot_id"], direction_class)]
            direction = np.asarray(
                [
                    direction_row["direction_vector_x"],
                    direction_row["direction_vector_y"],
                    direction_row["direction_vector_z"],
                ],
                dtype=np.float64,
            )
            for magnitude in MAGNITUDES_M:
                for sign in SIGNS:
                    delta, initial, perturbation = construct_translation_perturbation(
                        t_star, direction, magnitude, sign
                    )
                    for backend in BACKENDS:
                        output.append(
                            {
                                "execution_index": len(output) + 1,
                                "run_id": build_run_id(
                                    source["snapshot_id"],
                                    direction_class,
                                    magnitude,
                                    sign,
                                    backend,
                                ),
                                "scene_id": scene_id,
                                "scene_class": EXPECTED_TARGETS[scene_id]["scene_class"],
                                "snapshot_id": source["snapshot_id"],
                                "backend": backend,
                                "direction_class": direction_class,
                                "direction_source": direction_row["direction_source"],
                                "direction_vector_x": float(direction[0]),
                                "direction_vector_y": float(direction[1]),
                                "direction_vector_z": float(direction[2]),
                                "perturbation_sign": sign,
                                "perturbation_magnitude_m": magnitude,
                                "perturbation_vector_x_m": float(perturbation[0]),
                                "perturbation_vector_y_m": float(perturbation[1]),
                                "perturbation_vector_z_m": float(perturbation[2]),
                                "Delta_T": delta.tolist(),
                                "T_star": t_star.tolist(),
                                "T_initial": initial.tolist(),
                            }
                        )
    if len(output) != EXPECTED_RUN_COUNT:
        raise PilotBagError(f"trial plan is not exactly 480 rows: {len(output)}")
    if len({row["run_id"] for row in output}) != EXPECTED_RUN_COUNT:
        raise PilotBagError("trial plan run IDs are not unique")
    return output


def _pcl_checksums(
    source: np.ndarray,
    target: np.ndarray,
    t_initial: np.ndarray,
    run_id: str,
) -> dict[str, str]:
    source_sha = array_sha256(source)
    target_sha = array_sha256(target)
    pose_sha = _pose_sha256(t_initial)
    snapshot_sha = hashlib.sha256(
        f"mid360-controlled|{run_id}|{source_sha}|{target_sha}|{pose_sha}".encode()
    ).hexdigest()
    return {
        "source_checksum": source_sha,
        "target_checksum": target_sha,
        "reference_pose_checksum": pose_sha,
        "snapshot_checksum": snapshot_sha,
    }


def _run_base(
    plan: Mapping[str, Any],
    source_row: Mapping[str, str],
    direction: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": plan["run_id"],
        "scene_id": plan["scene_id"],
        "scene_class": plan["scene_class"],
        "snapshot_id": plan["snapshot_id"],
        "backend": plan["backend"],
        "source_file": source_row["source_path"],
        "source_file_sha256": source_row["source_npy_sha256"],
        "source_array_sha256": source_row["source_array_sha256"],
        "target_file": source_row["target_path"],
        "target_file_sha256": source_row["target_npy_sha256"],
        "target_array_sha256": source_row["target_array_sha256"],
        "backend_contract_sha256": BACKEND_CONTRACT_SHA256,
        "geometry_lambda_min": direction["geometry_lambda_min"],
        "geometry_lambda_mid": direction["geometry_lambda_mid"],
        "geometry_lambda_max": direction["geometry_lambda_max"],
        "geometry_condition_number": direction["geometry_condition_number"],
        "geometry_entropy": direction["geometry_entropy"],
        "direction_class": plan["direction_class"],
        "direction_vector_x": plan["direction_vector_x"],
        "direction_vector_y": plan["direction_vector_y"],
        "direction_vector_z": plan["direction_vector_z"],
        "direction_source": plan["direction_source"],
        "direction_frame": "TARGET_FRAME",
        "perturbation_sign": plan["perturbation_sign"],
        "perturbation_magnitude_m": plan["perturbation_magnitude_m"],
        "perturbation_vector_x_m": plan["perturbation_vector_x_m"],
        "perturbation_vector_y_m": plan["perturbation_vector_y_m"],
        "perturbation_vector_z_m": plan["perturbation_vector_z_m"],
        "T_star": plan["T_star"],
        "T_initial": plan["T_initial"],
        "transform_convention": "source/query_to_target/map",
        "perturbation_composition": "LEFT: T_initial=Delta_T@T_star",
        **FLAGS,
    }


def _failure_result(base: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    return {
        **base,
        "T_est": None,
        "initial_translation_error_m": float(base["perturbation_magnitude_m"]),
        "final_translation_error_m": None,
        "initial_rotation_error_deg": 0.0,
        "final_rotation_error_deg": None,
        "final_translation_error_vector_x_m": None,
        "final_translation_error_vector_y_m": None,
        "final_translation_error_vector_z_m": None,
        "translation_error_reduction_ratio": None,
        "solver_success": False,
        "finite_transform": False,
        "pose_recovered": False,
        "fitness": None,
        "rmse": None,
        "correspondence_count": None,
        "iteration_count": None,
        "runtime_ms": None,
        "backend_version": None,
        "failure_reason": f"{type(error).__name__}: {error}",
    }


def _complete_result(
    base: Mapping[str, Any],
    t_star: np.ndarray,
    t_initial: np.ndarray,
    estimate: np.ndarray,
    *,
    solver_success: bool,
    finite_transform: bool,
    fitness: float | None,
    rmse: float | None,
    correspondence_count: int,
    iteration_count: int | None,
    runtime_ms: float,
    backend_version: str,
    failure_reason: str,
) -> dict[str, Any]:
    initial_error = pose_error(t_initial, t_star)
    final_error = pose_error(estimate, t_star)
    initial_translation = float(initial_error["translation_error_m"])
    final_translation = float(final_error["translation_error_m"])
    vector = final_error["translation_vector"]
    return {
        **base,
        "T_est": estimate.tolist(),
        "initial_translation_error_m": initial_translation,
        "final_translation_error_m": final_translation,
        "initial_rotation_error_deg": float(initial_error["rotation_error_deg"]),
        "final_rotation_error_deg": float(final_error["rotation_error_deg"]),
        "final_translation_error_vector_x_m": float(vector[0]),
        "final_translation_error_vector_y_m": float(vector[1]),
        "final_translation_error_vector_z_m": float(vector[2]),
        "translation_error_reduction_ratio": float(
            1.0 - final_translation / initial_translation
        ),
        "solver_success": bool(solver_success),
        "finite_transform": bool(finite_transform),
        "pose_recovered": pose_recovered(
            final_translation, float(final_error["rotation_error_deg"])
        ),
        "fitness": fitness,
        "rmse": rmse,
        "correspondence_count": int(correspondence_count),
        "iteration_count": iteration_count,
        "runtime_ms": float(runtime_ms),
        "backend_version": backend_version,
        "failure_reason": failure_reason,
    }


def _execute_one(
    plan: Mapping[str, Any],
    source_row: Mapping[str, str],
    direction: Mapping[str, Any],
    source: np.ndarray,
    target: np.ndarray,
    contract: Mapping[str, Any],
    pcl_executable: Path,
) -> dict[str, Any]:
    base = _run_base(plan, source_row, direction)
    t_star = _matrix(plan["T_star"], "T_star")
    t_initial = _matrix(plan["T_initial"], "T_initial")
    checksums = _pcl_checksums(source, target, t_initial, str(plan["run_id"]))
    if (
        checksums["source_checksum"] != source_row["source_array_sha256"]
        or checksums["target_checksum"] != source_row["target_array_sha256"]
    ):
        raise PilotBagError(f"pre-backend array SHA mismatch: {plan['run_id']}")
    try:
        if plan["backend"] == "open3d":
            result = run_open3d_full(
                source,
                target,
                t_initial,
                _open3d_config(contract["open3d"]["parameters"]),
                backend_seed=0,
                input_checksum=checksums["snapshot_checksum"],
            )
            estimate = np.asarray(result.final_pose, dtype=np.float64)
            return _complete_result(
                base,
                t_star,
                t_initial,
                estimate,
                solver_success=result.solver_converged,
                finite_transform=bool(
                    result.finite_result and np.all(np.isfinite(estimate))
                ),
                fitness=float(result.extra["fitness"]),
                rmse=float(result.extra["inlier_rmse"]),
                correspondence_count=result.correspondence_count,
                # Open3D's registration result does not expose this count.
                iteration_count=None,
                runtime_ms=result.runtime_ms,
                backend_version=str(result.extra["open3d_version"]),
                failure_reason=str(result.failure_reason),
            )
        if plan["backend"] == "pcl":
            result = run_pcl_point_to_plane(
                source,
                target,
                t_initial,
                trial_id=str(plan["run_id"]),
                checksums=checksums,
                executable=pcl_executable,
                parameters=frozen_parameters(contract["pcl"]["parameters"]),
            )
            if result.final_transformation is None:
                raise PilotBagError("PCL returned no finite transformation")
            estimate = np.asarray(result.final_transformation, dtype=np.float64)
            return _complete_result(
                base,
                t_star,
                t_initial,
                estimate,
                solver_success=result.has_converged,
                finite_transform=bool(
                    result.finite_output and np.all(np.isfinite(estimate))
                ),
                fitness=result.fitness_score,
                rmse=None,
                correspondence_count=result.correspondence_count,
                iteration_count=int(result.iteration_count),
                runtime_ms=result.runtime_ms,
                backend_version=str(result.pcl_version),
                failure_reason=str(result.failure_reason),
            )
        raise PilotBagError(f"unknown backend: {plan['backend']}")
    except Exception as error:
        return _failure_result(base, error)


def _float_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if row.get(field) is not None and math.isfinite(float(row[field]))
    ]


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _median(values: Sequence[float]) -> float | None:
    return _quantile(values, 0.5)


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(sum(row.get(field) is True for row in rows) / len(rows))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-15:
        return None
    return float(numerator / denominator)


def build_summary(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[
            (
                str(row["scene_class"]),
                str(row["backend"]),
                str(row["direction_class"]),
                float(row["perturbation_magnitude_m"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        translation = _float_values(rows, "final_translation_error_m")
        rotation = _float_values(rows, "final_rotation_error_deg")
        reduction = _float_values(rows, "translation_error_reduction_ratio")
        fitness = _float_values(rows, "fitness")
        rmse = _float_values(rows, "rmse")
        correspondence = _float_values(rows, "correspondence_count")
        output.append(
            {
                "scene_class": key[0],
                "backend": key[1],
                "direction_class": key[2],
                "perturbation_magnitude_m": key[3],
                "n": len(rows),
                "solver_success_rate": _rate(rows, "solver_success"),
                "finite_rate": _rate(rows, "finite_transform"),
                "pose_recovery_rate": _rate(rows, "pose_recovered"),
                "final_translation_error_m_median": _median(translation),
                "final_translation_error_m_q25": _quantile(translation, 0.25),
                "final_translation_error_m_q75": _quantile(translation, 0.75),
                "final_translation_error_m_q95": _quantile(translation, 0.95),
                "final_translation_error_m_max": max(translation, default=None),
                "final_rotation_error_deg_median": _median(rotation),
                "final_rotation_error_deg_q25": _quantile(rotation, 0.25),
                "final_rotation_error_deg_q75": _quantile(rotation, 0.75),
                "final_rotation_error_deg_q95": _quantile(rotation, 0.95),
                "final_rotation_error_deg_max": max(rotation, default=None),
                "translation_error_reduction_ratio_median": _median(reduction),
                "translation_error_reduction_ratio_q25": _quantile(reduction, 0.25),
                "translation_error_reduction_ratio_q75": _quantile(reduction, 0.75),
                "fitness_median": _median(fitness),
                "rmse_median": _median(rmse),
                "correspondence_count_median": _median(correspondence),
                **FLAGS,
            }
        )
    if len(output) != 24 or any(row["n"] != 20 for row in output):
        raise PilotBagError("summary groups do not close to 24 groups of 20")
    return output


def build_rich_vs_weak(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for backend in BACKENDS:
        for direction in DIRECTION_CLASSES:
            for magnitude in MAGNITUDES_M:
                subsets = {
                    scene: [
                        row
                        for row in runs
                        if row["backend"] == backend
                        and row["direction_class"] == direction
                        and float(row["perturbation_magnitude_m"]) == magnitude
                        and row["scene_class"] == scene
                    ]
                    for scene in ("Rich", "Weak")
                }
                rich_median = _median(
                    _float_values(subsets["Rich"], "final_translation_error_m")
                )
                weak_median = _median(
                    _float_values(subsets["Weak"], "final_translation_error_m")
                )
                rich_recovery = _rate(subsets["Rich"], "pose_recovered")
                weak_recovery = _rate(subsets["Weak"], "pose_recovered")
                output.append(
                    {
                        "backend": backend,
                        "direction_class": direction,
                        "perturbation_magnitude_m": magnitude,
                        "rich_n": len(subsets["Rich"]),
                        "weak_n": len(subsets["Weak"]),
                        "rich_median_final_translation_error_m": rich_median,
                        "weak_median_final_translation_error_m": weak_median,
                        "weak_rich_median_ratio": _safe_ratio(
                            weak_median, rich_median
                        ),
                        "rich_recovery_rate": rich_recovery,
                        "weak_recovery_rate": weak_recovery,
                        "weak_minus_rich_recovery_rate": (
                            weak_recovery - rich_recovery
                        ),
                        **FLAGS,
                    }
                )
    return output


def build_weak_vs_strong(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scene in ("Rich", "Weak"):
        for backend in BACKENDS:
            for magnitude in MAGNITUDES_M:
                subsets = {
                    direction: [
                        row
                        for row in runs
                        if row["scene_class"] == scene
                        and row["backend"] == backend
                        and row["direction_class"] == direction
                        and float(row["perturbation_magnitude_m"]) == magnitude
                    ]
                    for direction in DIRECTION_CLASSES
                }
                weak_median = _median(
                    _float_values(subsets["weak"], "final_translation_error_m")
                )
                strong_median = _median(
                    _float_values(subsets["strong"], "final_translation_error_m")
                )
                output.append(
                    {
                        "scene_class": scene,
                        "backend": backend,
                        "perturbation_magnitude_m": magnitude,
                        "weak_direction_n": len(subsets["weak"]),
                        "strong_direction_n": len(subsets["strong"]),
                        "weak_direction_median_final_translation_error_m": weak_median,
                        "strong_direction_median_final_translation_error_m": strong_median,
                        "weak_strong_median_ratio": _safe_ratio(
                            weak_median, strong_median
                        ),
                        "weak_direction_recovery_rate": _rate(
                            subsets["weak"], "pose_recovered"
                        ),
                        "strong_direction_recovery_rate": _rate(
                            subsets["strong"], "pose_recovered"
                        ),
                        **FLAGS,
                    }
                )
    return output


def build_paired_differences(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, float, int, str], Mapping[str, Any]] = {}
    for row in runs:
        key = (
            str(row["snapshot_id"]),
            str(row["backend"]),
            float(row["perturbation_magnitude_m"]),
            int(row["perturbation_sign"]),
            str(row["direction_class"]),
        )
        by_key[key] = row
    output: list[dict[str, Any]] = []
    bases = sorted({key[:4] for key in by_key})
    for snapshot_id, backend, magnitude, sign in bases:
        weak = by_key[(snapshot_id, backend, magnitude, sign, "weak")]
        strong = by_key[(snapshot_id, backend, magnitude, sign, "strong")]
        weak_error = weak.get("final_translation_error_m")
        strong_error = strong.get("final_translation_error_m")
        output.append(
            {
                "scene_id": weak["scene_id"],
                "scene_class": weak["scene_class"],
                "snapshot_id": snapshot_id,
                "backend": backend,
                "perturbation_magnitude_m": magnitude,
                "perturbation_sign": sign,
                "weak_run_id": weak["run_id"],
                "strong_run_id": strong["run_id"],
                "weak_final_translation_error_m": weak_error,
                "strong_final_translation_error_m": strong_error,
                "weak_minus_strong_final_translation_error_m": (
                    None
                    if weak_error is None or strong_error is None
                    else float(weak_error) - float(strong_error)
                ),
                "weak_strong_error_ratio": _safe_ratio(
                    None if weak_error is None else float(weak_error),
                    None if strong_error is None else float(strong_error),
                ),
                "weak_pose_recovered": weak["pose_recovered"],
                "strong_pose_recovered": strong["pose_recovered"],
                **FLAGS,
            }
        )
    if len(output) != 240:
        raise PilotBagError(f"paired difference count is not 240: {len(output)}")
    return output


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def _direction_cosine(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    fields = (
        "final_translation_error_vector_x_m",
        "final_translation_error_vector_y_m",
        "final_translation_error_vector_z_m",
    )
    if any(left.get(field) is None or right.get(field) is None for field in fields):
        return None
    first = np.asarray([left[field] for field in fields], dtype=np.float64)
    second = np.asarray([right[field] for field in fields], dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-15:
        return None
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def identical_backend_input_pair(
    open3d_row: Mapping[str, Any], pcl_row: Mapping[str, Any]
) -> bool:
    fields = (
        "scene_id",
        "snapshot_id",
        "source_file_sha256",
        "source_array_sha256",
        "target_file_sha256",
        "target_array_sha256",
        "backend_contract_sha256",
        "direction_class",
        "direction_vector_x",
        "direction_vector_y",
        "direction_vector_z",
        "perturbation_sign",
        "perturbation_magnitude_m",
        "perturbation_vector_x_m",
        "perturbation_vector_y_m",
        "perturbation_vector_z_m",
        "T_star",
        "T_initial",
    )
    return all(open3d_row[field] == pcl_row[field] for field in fields)


def build_backend_agreement(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    grouped: dict[tuple[str, str, float, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in runs:
        key = (
            str(row["snapshot_id"]),
            str(row["direction_class"]),
            float(row["perturbation_magnitude_m"]),
            int(row["perturbation_sign"]),
        )
        grouped[key][str(row["backend"])] = row
    for key in sorted(grouped):
        backend_rows = grouped[key]
        if set(backend_rows) != set(BACKENDS):
            raise PilotBagError(f"backend pair incomplete: {key}")
        pair = (backend_rows["open3d"], backend_rows["pcl"])
        if not identical_backend_input_pair(*pair):
            raise PilotBagError(f"Open3D/PCL input mismatch: {key}")
        pairs.append(pair)
    if len(pairs) != 240:
        raise PilotBagError(f"backend pair count is not 240: {len(pairs)}")

    definitions: list[tuple[str, str, float | str, list[Any]]] = [
        ("ALL", "ALL", "ALL", pairs)
    ]
    for scene in ("Rich", "Weak"):
        for direction in DIRECTION_CLASSES:
            for magnitude in MAGNITUDES_M:
                subset = [
                    pair
                    for pair in pairs
                    if pair[0]["scene_class"] == scene
                    and pair[0]["direction_class"] == direction
                    and float(pair[0]["perturbation_magnitude_m"]) == magnitude
                ]
                definitions.append((scene, direction, magnitude, subset))
    output: list[dict[str, Any]] = []
    for scene, direction, magnitude, subset in definitions:
        translation_pairs = [
            (float(left["final_translation_error_m"]), float(right["final_translation_error_m"]))
            for left, right in subset
            if left.get("final_translation_error_m") is not None
            and right.get("final_translation_error_m") is not None
        ]
        rotation_pairs = [
            (float(left["final_rotation_error_deg"]), float(right["final_rotation_error_deg"]))
            for left, right in subset
            if left.get("final_rotation_error_deg") is not None
            and right.get("final_rotation_error_deg") is not None
        ]
        cosines = [
            value
            for value in (_direction_cosine(left, right) for left, right in subset)
            if value is not None
        ]
        output.append(
            {
                "scene_class": scene,
                "direction_class": direction,
                "perturbation_magnitude_m": magnitude,
                "n_pairs": len(subset),
                "identical_input_pair_rate": float(
                    sum(identical_backend_input_pair(*pair) for pair in subset)
                    / len(subset)
                ),
                "final_translation_error_spearman": _spearman(
                    [pair[0] for pair in translation_pairs],
                    [pair[1] for pair in translation_pairs],
                ),
                "final_rotation_error_spearman": _spearman(
                    [pair[0] for pair in rotation_pairs],
                    [pair[1] for pair in rotation_pairs],
                ),
                "error_direction_cosine_median": _median(cosines),
                "error_direction_cosine_q25": _quantile(cosines, 0.25),
                "error_direction_cosine_q75": _quantile(cosines, 0.75),
                "recovery_agreement_rate": float(
                    sum(
                        left["pose_recovered"] == right["pose_recovered"]
                        for left, right in subset
                    )
                    / len(subset)
                ),
                **FLAGS,
            }
        )
    return output


def _comparison_lookup(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {tuple(row[field] for field in fields): row for row in rows}


def build_decision_gate(
    runs: Sequence[Mapping[str, Any]],
    rich_vs_weak: Sequence[Mapping[str, Any]],
    weak_vs_strong: Sequence[Mapping[str, Any]],
    *,
    verification_pass: bool,
) -> dict[str, Any]:
    rich_lookup = _comparison_lookup(
        rich_vs_weak,
        ("backend", "direction_class", "perturbation_magnitude_m"),
    )
    direction_lookup = _comparison_lookup(
        weak_vs_strong,
        ("scene_class", "backend", "perturbation_magnitude_m"),
    )
    large = (0.03, 0.05)

    def direction_more_sensitive(scene: str, backend: str) -> bool:
        return all(
            float(
                direction_lookup[(scene, backend, magnitude)][
                    "weak_direction_median_final_translation_error_m"
                ]
            )
            > float(
                direction_lookup[(scene, backend, magnitude)][
                    "strong_direction_median_final_translation_error_m"
                ]
            )
            for magnitude in large
        )

    def weak_scene_more_sensitive(backend: str) -> bool:
        return all(
            float(
                rich_lookup[(backend, "weak", magnitude)][
                    "weak_median_final_translation_error_m"
                ]
            )
            > float(
                rich_lookup[(backend, "weak", magnitude)][
                    "rich_median_final_translation_error_m"
                ]
            )
            for magnitude in large
        )

    base_by_magnitude: dict[str, bool] = {}
    alternative_a_by_magnitude: dict[str, bool] = {}
    alternative_b_by_magnitude: dict[str, bool] = {}
    for magnitude in large:
        label = f"{int(magnitude * 1000)}mm"
        base_by_magnitude[label] = all(
            float(
                rich_lookup[(backend, "weak", magnitude)][
                    "weak_median_final_translation_error_m"
                ]
            )
            > float(
                rich_lookup[(backend, "weak", magnitude)][
                    "rich_median_final_translation_error_m"
                ]
            )
            for backend in BACKENDS
        )
        alternative_a_by_magnitude[label] = all(
            float(
                rich_lookup[(backend, "weak", magnitude)]["weak_recovery_rate"]
            )
            < float(
                rich_lookup[(backend, "weak", magnitude)]["rich_recovery_rate"]
            )
            for backend in BACKENDS
        )
        alternative_b_by_magnitude[label] = all(
            float(
                direction_lookup[(scene, backend, magnitude)][
                    "weak_direction_median_final_translation_error_m"
                ]
            )
            > float(
                direction_lookup[(scene, backend, magnitude)][
                    "strong_direction_median_final_translation_error_m"
                ]
            )
            for scene in ("Rich", "Weak")
            for backend in BACKENDS
        )

    contrasts = []
    for magnitude in large:
        for scene in ("Rich", "Weak"):
            signs = []
            for backend in BACKENDS:
                row = direction_lookup[(scene, backend, magnitude)]
                signs.append(
                    math.copysign(
                        1.0,
                        float(row["weak_direction_median_final_translation_error_m"])
                        - float(row["strong_direction_median_final_translation_error_m"]),
                    )
                )
            contrasts.append(signs[0] == signs[1])
        scene_signs = []
        for backend in BACKENDS:
            row = rich_lookup[(backend, "weak", magnitude)]
            scene_signs.append(
                math.copysign(
                    1.0,
                    float(row["weak_median_final_translation_error_m"])
                    - float(row["rich_median_final_translation_error_m"]),
                )
            )
        contrasts.append(scene_signs[0] == scene_signs[1])

    expansion = bool(
        all(base_by_magnitude.values())
        and all(
            alternative_a_by_magnitude[label]
            or alternative_b_by_magnitude[label]
            for label in base_by_magnitude
        )
    )
    finite_count = sum(row["finite_transform"] is True for row in runs)
    ready = bool(
        verification_pass
        and len(runs) == EXPECTED_RUN_COUNT
        and len({row["run_id"] for row in runs}) == EXPECTED_RUN_COUNT
        and finite_count == EXPECTED_RUN_COUNT
    )
    return {
        "schema": f"{SCHEMA}_decision_gate",
        "gate_preregistered_before_backend_execution": True,
        "gate_definition": {
            "large_perturbation_levels_m": list(large),
            "base": (
                "At both 30 and 50 mm, Weak scene weak-direction median final "
                "translation error exceeds Rich for both backends."
            ),
            "alternative_a": (
                "At a level, Weak scene weak-direction recovery is below Rich "
                "for both backends."
            ),
            "alternative_b": (
                "At a level, weak-direction median error exceeds strong-direction "
                "for both scenes and both backends."
            ),
            "support": "base AND (A OR B) at both 30 and 50 mm",
        },
        "GEOMETRY_SEPARATION_CONFIRMED": True,
        "WEAK_DIRECTION_MORE_SENSITIVE_RICH_OPEN3D": direction_more_sensitive(
            "Rich", "open3d"
        ),
        "WEAK_DIRECTION_MORE_SENSITIVE_RICH_PCL": direction_more_sensitive(
            "Rich", "pcl"
        ),
        "WEAK_DIRECTION_MORE_SENSITIVE_WEAK_OPEN3D": direction_more_sensitive(
            "Weak", "open3d"
        ),
        "WEAK_DIRECTION_MORE_SENSITIVE_WEAK_PCL": direction_more_sensitive(
            "Weak", "pcl"
        ),
        "WEAK_SCENE_MORE_SENSITIVE_AT_WEAK_DIRECTION_OPEN3D": weak_scene_more_sensitive(
            "open3d"
        ),
        "WEAK_SCENE_MORE_SENSITIVE_AT_WEAK_DIRECTION_PCL": weak_scene_more_sensitive(
            "pcl"
        ),
        "BACKEND_DIRECTIONALLY_CONSISTENT": all(contrasts),
        "base_by_magnitude": base_by_magnitude,
        "alternative_a_by_magnitude": alternative_a_by_magnitude,
        "alternative_b_by_magnitude": alternative_b_by_magnitude,
        "CONTROLLED_PERTURBATION_PILOT_READY": ready,
        "CONTROLLED_PERTURBATION_PILOT_SUPPORTS_EXPANSION": expansion,
        **FLAGS,
    }


def _plot_grouped_lines(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    value_field: str,
    ylabel: str,
    scale: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {"Rich": "#0072B2", "Weak": "#D55E00"}
    styles = {"weak": "-", "strong": "--"}
    for axis, backend in zip(axes, BACKENDS):
        for scene in ("Rich", "Weak"):
            for direction in DIRECTION_CLASSES:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["scene_class"] == scene
                        and row["backend"] == backend
                        and row["direction_class"] == direction
                    ),
                    key=lambda row: float(row["perturbation_magnitude_m"]),
                )
                axis.plot(
                    [float(row["perturbation_magnitude_m"]) * 1000.0 for row in selected],
                    [float(row[value_field]) * scale for row in selected],
                    color=colors[scene],
                    linestyle=styles[direction],
                    marker="o",
                    label=f"{scene} / {direction}",
                )
        axis.set_title(backend.upper())
        axis.set_xlabel("Initial translation perturbation [mm]")
        axis.set_xticks([10, 30, 50])
        axis.grid(alpha=0.25)
    plotted = [float(row[value_field]) * scale for row in rows]
    maximum = max(plotted, default=1.0)
    axes[0].set_ylim(0.0, max(maximum * 1.08, 1.0e-12))
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Mid-360 controlled perturbation Pilot (descriptive, nonformal)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    output: Path,
    runs: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    rich_vs_weak: Sequence[Mapping[str, Any]],
    weak_vs_strong: Sequence[Mapping[str, Any]],
) -> None:
    _plot_grouped_lines(
        summary,
        output / "01_translation_error_vs_perturbation.png",
        value_field="final_translation_error_m_median",
        ylabel="Median final translation error [mm]",
        scale=1000.0,
    )
    _plot_grouped_lines(
        summary,
        output / "02_recovery_rate_vs_perturbation.png",
        value_field="pose_recovery_rate",
        ylabel="Pose recovery rate",
        scale=1.0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {"open3d": "#009E73", "pcl": "#CC79A7"}
    for axis, scene in zip(axes, ("Rich", "Weak")):
        for backend in BACKENDS:
            selected = sorted(
                (
                    row
                    for row in weak_vs_strong
                    if row["scene_class"] == scene and row["backend"] == backend
                ),
                key=lambda row: float(row["perturbation_magnitude_m"]),
            )
            x = [float(row["perturbation_magnitude_m"]) * 1000.0 for row in selected]
            axis.plot(
                x,
                [float(row["weak_direction_median_final_translation_error_m"]) * 1000.0 for row in selected],
                color=colors[backend],
                linestyle="-",
                marker="o",
                label=f"{backend} / weak dir",
            )
            axis.plot(
                x,
                [float(row["strong_direction_median_final_translation_error_m"]) * 1000.0 for row in selected],
                color=colors[backend],
                linestyle="--",
                marker="s",
                label=f"{backend} / strong dir",
            )
        axis.set_title(scene)
        axis.set_xlabel("Initial translation perturbation [mm]")
        axis.set_xticks([10, 30, 50])
        axis.grid(alpha=0.25)
    direction_maximum = max(
        max(
            float(row["weak_direction_median_final_translation_error_m"]),
            float(row["strong_direction_median_final_translation_error_m"]),
        )
        * 1000.0
        for row in weak_vs_strong
    )
    axes[0].set_ylim(0.0, direction_maximum * 1.08)
    axes[0].set_ylabel("Median final translation error [mm]")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Weak vs strong geometry direction (Pilot, nonformal)")
    fig.tight_layout()
    fig.savefig(output / "03_weak_vs_strong_direction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, backend in zip(axes, BACKENDS):
        for scene, field, color in (
            ("Rich", "rich_median_final_translation_error_m", "#0072B2"),
            ("Weak", "weak_median_final_translation_error_m", "#D55E00"),
        ):
            selected = sorted(
                (
                    row
                    for row in rich_vs_weak
                    if row["backend"] == backend and row["direction_class"] == "weak"
                ),
                key=lambda row: float(row["perturbation_magnitude_m"]),
            )
            axis.plot(
                [float(row["perturbation_magnitude_m"]) * 1000.0 for row in selected],
                [float(row[field]) * 1000.0 for row in selected],
                marker="o",
                color=color,
                label=scene,
            )
        axis.set_title(backend.upper())
        axis.set_xlabel("Weak-direction perturbation [mm]")
        axis.set_xticks([10, 30, 50])
        axis.grid(alpha=0.25)
    scene_maximum = max(
        max(
            float(row["rich_median_final_translation_error_m"]),
            float(row["weak_median_final_translation_error_m"]),
        )
        * 1000.0
        for row in rich_vs_weak
        if row["direction_class"] == "weak"
    )
    axes[0].set_ylim(0.0, scene_maximum * 1.08)
    axes[0].set_ylabel("Median final translation error [mm]")
    axes[1].legend(loc="best")
    fig.suptitle("Rich vs Weak scene (Pilot, nonformal)")
    fig.tight_layout()
    fig.savefig(output / "04_rich_vs_weak.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    grouped: dict[tuple[str, str, float, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in runs:
        key = (
            str(row["snapshot_id"]),
            str(row["direction_class"]),
            float(row["perturbation_magnitude_m"]),
            int(row["perturbation_sign"]),
        )
        grouped[key][str(row["backend"])] = row
    points = [
        (
            float(pair["open3d"]["final_translation_error_m"]) * 1000.0,
            float(pair["pcl"]["final_translation_error_m"]) * 1000.0,
            pair["open3d"]["scene_class"],
        )
        for pair in grouped.values()
        if pair["open3d"].get("final_translation_error_m") is not None
        and pair["pcl"].get("final_translation_error_m") is not None
    ]
    fig, axis = plt.subplots(figsize=(6.2, 5.6))
    for scene, color in (("Rich", "#0072B2"), ("Weak", "#D55E00")):
        selected = [point for point in points if point[2] == scene]
        axis.scatter(
            [point[0] for point in selected],
            [point[1] for point in selected],
            s=18,
            alpha=0.65,
            color=color,
            label=scene,
        )
    maximum = max(max(point[0], point[1]) for point in points) * 1.05
    axis.plot([0.0, maximum], [0.0, maximum], color="black", linestyle="--", linewidth=1)
    axis.set_xlim(0.0, maximum)
    axis.set_ylim(0.0, maximum)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Open3D final translation error [mm]")
    axis.set_ylabel("PCL final translation error [mm]")
    axis.set_title("Matched backend comparison (Pilot, nonformal)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / "05_open3d_vs_pcl_translation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_csv_value(row: Mapping[str, str], field: str) -> Any:
    value = row[field]
    if value == "":
        return None
    if field in {
        "T_star",
        "T_initial",
        "T_est",
        "Delta_T",
    }:
        return json.loads(value)
    if field in {
        "solver_success",
        "finite_transform",
        "pose_recovered",
        *FLAGS.keys(),
    }:
        if value in {"true", "True"}:
            return True
        if value in {"false", "False"}:
            return False
        return value
    return value


def load_runs(path: Path) -> list[dict[str, Any]]:
    raw = read_csv(path)
    return [
        {field: _parse_csv_value(row, field) for field in row}
        for row in raw
    ]


def _same_numeric(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def verify_run_rows(
    runs: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
    baseline_before: Mapping[str, str],
    baseline_after: Mapping[str, str],
    repository: Path,
    runtime: Path,
) -> dict[str, bool]:
    plan_ids = {str(row["run_id"]) for row in plan}
    run_ids = {str(row["run_id"]) for row in runs}
    checks: dict[str, bool] = {
        "exactly_480_rows": len(runs) == EXPECTED_RUN_COUNT,
        "exactly_480_unique_run_ids": len(run_ids) == EXPECTED_RUN_COUNT,
        "run_id_set_matches_frozen_plan": run_ids == plan_ids,
        "exactly_240_open3d": sum(row["backend"] == "open3d" for row in runs) == 240,
        "exactly_240_pcl": sum(row["backend"] == "pcl" for row in runs) == 240,
        "all_finite_transforms": all(row["finite_transform"] is True for row in runs),
        "backend_contract_sha_fixed": all(
            row["backend_contract_sha256"] == BACKEND_CONTRACT_SHA256 for row in runs
        ),
        "zero_perturbation_runtime_unchanged": dict(baseline_before)
        == dict(baseline_after),
        "formal_measurement_false": all(
            row["FORMAL_MEASUREMENT_RESULT"] is False for row in runs
        ),
        "repository_contract_sha_fixed": sha256_file(
            repository / "frozen_assets/backend_parameter_contract.json"
        )
        == BACKEND_CONTRACT_SHA256,
        "frozen_transform_convention_authenticated": (
            read_json(runtime / "transform_convention.json").get(
                "transform_convention_verified"
            )
            is True
        ),
    }
    plan_by_id = {str(row["run_id"]): row for row in plan}
    row_math = True
    recovery_math = True
    source_hashes = True
    authenticated_files: dict[str, str] = {}
    for row in runs:
        expected = plan_by_id[str(row["run_id"])]
        t_star = _matrix(row["T_star"], "verified T_star")
        t_initial = _matrix(row["T_initial"], "verified T_initial")
        initial_error = pose_error(t_initial, t_star)
        vector = np.asarray(
            [
                float(row["perturbation_vector_x_m"]),
                float(row["perturbation_vector_y_m"]),
                float(row["perturbation_vector_z_m"]),
            ]
        )
        row_math &= bool(
            _same_numeric(
                initial_error["translation_error_m"],
                float(row["perturbation_magnitude_m"]),
                PERTURBATION_TOLERANCE_M,
            )
            and np.allclose(
                (t_initial @ np.linalg.inv(t_star))[:3, 3],
                vector,
                atol=PERTURBATION_TOLERANCE_M,
                rtol=0.0,
            )
            and row["T_initial"] == expected["T_initial"]
        )
        if row.get("T_est") is not None:
            final_error = pose_error(_matrix(row["T_est"], "verified T_est"), t_star)
            final_translation = float(final_error["translation_error_m"])
            final_rotation = float(final_error["rotation_error_deg"])
            expected_reduction = 1.0 - final_translation / float(
                initial_error["translation_error_m"]
            )
            row_math &= bool(
                _same_numeric(final_translation, row["final_translation_error_m"])
                and _same_numeric(final_rotation, row["final_rotation_error_deg"])
                and _same_numeric(
                    expected_reduction, row["translation_error_reduction_ratio"]
                )
            )
            recovery_math &= row["pose_recovered"] is pose_recovered(
                final_translation, final_rotation
            )
        for path_field, sha_field in (
            ("source_file", "source_file_sha256"),
            ("target_file", "target_file_sha256"),
        ):
            path = str(row[path_field])
            actual = authenticated_files.get(path)
            if actual is None:
                actual = sha256_file(Path(path))
                authenticated_files[path] = actual
            source_hashes &= actual == row[sha_field]
    checks["all_run_math_recomputed"] = row_math
    checks["all_recovery_thresholds_recomputed"] = recovery_math
    checks["all_run_file_hashes_authenticated"] = source_hashes

    grouped: dict[tuple[str, str, float, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in runs:
        key = (
            str(row["snapshot_id"]),
            str(row["direction_class"]),
            float(row["perturbation_magnitude_m"]),
            int(row["perturbation_sign"]),
        )
        grouped[key][str(row["backend"])] = row
    checks["exactly_240_complete_backend_pairs"] = len(grouped) == 240 and all(
        set(pair) == set(BACKENDS) for pair in grouped.values()
    )
    checks["all_backend_pairs_have_identical_inputs"] = all(
        identical_backend_input_pair(pair["open3d"], pair["pcl"])
        for pair in grouped.values()
    )
    return checks


def _write_output_sha256s(output: Path) -> None:
    lines = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verification_report(
    checks: Mapping[str, bool],
    runs: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    baseline_before: Mapping[str, str],
) -> str:
    lines = [
        "MID-360 CONTROLLED INITIAL-POSE PERTURBATION PILOT VERIFICATION",
        f"schema={SCHEMA}",
        f"generated_at_utc={utc_now()}",
        "scope=pilot descriptive / sensitivity analysis; nonformal",
        "transform_convention=T maps source/query into target/map",
        "perturbation=T_initial=Delta_T@T_star; target-frame translation",
        f"recovery_threshold_translation_m={RECOVERY_TRANSLATION_M}",
        f"recovery_threshold_rotation_deg={RECOVERY_ROTATION_DEG}",
        "translation_error_reduction_ratio=1-final_translation_error/initial_translation_error",
        f"run_count={len(runs)}",
        f"open3d_count={sum(row['backend'] == 'open3d' for row in runs)}",
        f"pcl_count={sum(row['backend'] == 'pcl' for row in runs)}",
        f"solver_success_count={sum(row['solver_success'] is True for row in runs)}",
        f"finite_transform_count={sum(row['finite_transform'] is True for row in runs)}",
        f"zero_runtime_authenticated_file_count={len(baseline_before)}",
        f"backend_contract_sha256={BACKEND_CONTRACT_SHA256}",
        "formal_measurement_result=false",
        "",
        "CHECKS",
    ]
    lines.extend(
        f"{'PASS' if passed else 'FAIL'} {name}" for name, passed in checks.items()
    )
    lines.extend(
        [
            "",
            f"CONTROLLED_PERTURBATION_PILOT_READY={str(decision['CONTROLLED_PERTURBATION_PILOT_READY']).lower()}",
            f"CONTROLLED_PERTURBATION_PILOT_SUPPORTS_EXPANSION={str(decision['CONTROLLED_PERTURBATION_PILOT_SUPPORTS_EXPANSION']).lower()}",
            "FORMAL_MEASUREMENT_RESULT=false",
            "",
            "LIMITATIONS",
            "NO_OBVIOUS_MOTION is retained; perfect/static ground truth is not claimed.",
            "ACCELERATION_UNIT_UNKNOWN=true; no 9.81 scaling or IMU integration was used.",
            "Point coordinates remain ASSUMED_METERS_FROM_SCALE.",
            "Snapshot observations within a scene/station are not independent scene replicates.",
        ]
    )
    return "\n".join(lines) + "\n"


def _readme(
    decision: Mapping[str, Any],
    checks: Mapping[str, bool],
    runs: Sequence[Mapping[str, Any]],
) -> str:
    ready = str(decision["CONTROLLED_PERTURBATION_PILOT_READY"]).lower()
    expansion = str(
        decision["CONTROLLED_PERTURBATION_PILOT_SUPPORTS_EXPANSION"]
    ).lower()
    return f"""# Mid-360 controlled perturbation Pilot

`CONTROLLED_PERTURBATION_PILOT_READY={ready}`  
`CONTROLLED_PERTURBATION_PILOT_SUPPORTS_EXPANSION={expansion}`  
`FORMAL_MEASUREMENT_RESULT=false`

This is a nonformal, descriptive sensitivity/debug experiment over the frozen
Rich and Weak Mid-360 Pilot inputs. It contains {len(runs)} new ICP runs and
does not include the preserved zero-perturbation runs in that count.

The geometry matrix is the existing translation-only matrix
`H_trans=(N.T@N)/valid_normal_count`, using the existing frozen 0.50 m
association and target PCA-normal context. Weak and strong directions are the
unit eigenvectors for `lambda_min` and `lambda_max`. Eigenvector signs are made
deterministic by requiring the largest-absolute component to be nonnegative;
both positive and negative perturbations are then tested.

Transforms map source/query points into target/map coordinates. The initial
pose is `T_initial = Delta_T @ T_star`, so the perturbation vector is expressed
in the target frame. Translation errors use `T_est @ inv(T_star)`. Recovery was
pre-registered as final translation error <= 0.005 m and final rotation error
<= 0.2 degree. The reduction ratio is `1 - final/initial`.

The Open3D/PCL algorithms and the backend parameter contract were not changed.
The backend contract SHA256 is `{BACKEND_CONTRACT_SHA256}`. Open3D iteration
count is blank because its API does not expose the executed iteration count.

Current zero-perturbation baseline retained in the report:

- Rich translation median/q95: Open3D 0.000535890/0.000789956 m; PCL 0.000580512/0.000834444 m.
- Weak translation median/q95: Open3D 0.000588254/0.000779236 m; PCL 0.000655950/0.000739324 m.
- Weak/Rich translation median ratio: Open3D 1.097713; PCL 1.129950.
- Rotation was opposite to a general degradation claim (Open3D Rich/Weak median 0.016014/0.009592 deg; PCL 0.018061/0.010979 deg), and turnover also did not support it.

Therefore: geometry separation is confirmed; zero-perturbation translation
showed a weak directionally consistent pilot signal; rotation and reassociation
did not support a general degradation claim.

`NO_OBVIOUS_MOTION` is retained and is not a claim of perfect, sub-millimeter,
or ground-truth static conditions. `ACCELERATION_UNIT_UNKNOWN=true`; no 9.81
conversion, IMU position integration, or IMU displacement ground truth was
used. Point coordinates remain `ASSUMED_METERS_FROM_SCALE`.

Snapshot-level observations within the same scene/station are not independent scene-level replicates.
All summaries are pilot descriptive / sensitivity analysis only and must not be
used for paper-level population inference.

Verification: `{'PASS' if all(checks.values()) else 'FAIL'}`. See
`verification_report.txt`, `manifest.json`, and `decision_gate.json` for the
authenticated inputs and pre-registered decision rule.
"""


def _manifest(
    repository: Path,
    output: Path,
    runtime: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    baseline_before: Mapping[str, str],
    bag_hashes: Mapping[str, str],
    directions: Sequence[Mapping[str, Any]],
    trial_plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    transform_convention_path = runtime / "transform_convention.json"
    geometry_path = runtime / "geometry_only_metrics.csv"
    return {
        "schema": f"{SCHEMA}_manifest",
        "created_before_backend_execution": True,
        "created_at_utc": utc_now(),
        "repository": str(repository),
        "output_directory": str(output),
        "zero_perturbation_runtime": str(runtime),
        "zero_perturbation_runtime_file_sha256": dict(baseline_before),
        "bags": {
            str(BAG_ROOT / name): digest for name, digest in bag_hashes.items()
        },
        "scene_contract": EXPECTED_TARGETS,
        "snapshot_count": EXPECTED_SNAPSHOT_COUNT,
        "direction_row_count": len(directions),
        "planned_run_count": len(trial_plan),
        "planned_backend_counts": {"open3d": 240, "pcl": 240},
        "magnitudes_m": list(MAGNITUDES_M),
        "signs": list(SIGNS),
        "direction_classes": list(DIRECTION_CLASSES),
        "direction_definition": {
            "matrix_dimension": "3x3_translation_only",
            "matrix": "H_trans=(N.T@N)/valid_normal_count",
            "association_and_normal_source": (
                "existing TargetGeometryContext and associate_source_points"
            ),
            "weak": "unit eigenvector(lambda_min)",
            "strong": "unit eigenvector(lambda_max)",
            "eigenvalue_order": "lambda_min <= lambda_mid <= lambda_max",
            "sign_canonicalization": "largest_absolute_component_nonnegative",
            "eigenvalue_match_absolute_tolerance": GEOMETRY_EIGENVALUE_TOLERANCE,
            "geometry_only_metrics_path": str(geometry_path),
            "geometry_only_metrics_sha256": sha256_file(geometry_path),
        },
        "pose_contract": {
            "transform_convention": "source/query_to_target/map",
            "direction_frame": "TARGET_FRAME",
            "perturbation_left_or_right_multiplication": "LEFT",
            "construction": "T_initial=Delta_T@T_star",
            "error_transform": "T_est@inv(T_star)",
            "perturbation_magnitude_tolerance_m": PERTURBATION_TOLERANCE_M,
            "frozen_transform_convention_path": str(transform_convention_path),
            "frozen_transform_convention_sha256": sha256_file(
                transform_convention_path
            ),
        },
        "recovery_contract": {
            "final_translation_error_m_max_inclusive": RECOVERY_TRANSLATION_M,
            "final_rotation_error_deg_max_inclusive": RECOVERY_ROTATION_DEG,
            "definition": "translation<=0.005 AND rotation<=0.2",
            "translation_error_reduction_ratio": "1-final/initial",
        },
        "decision_gate_contract": {
            "large_perturbation_levels_m": [0.03, 0.05],
            "base": (
                "Weak weak-direction median error > Rich at both levels and both backends"
            ),
            "alternative_a": (
                "Weak weak-direction recovery < Rich at a level for both backends"
            ),
            "alternative_b": (
                "weak-direction median error > strong-direction at a level for both scenes and backends"
            ),
            "support": "base AND (A OR B) at each of 30 and 50 mm",
        },
        "backend_contract_path": str(contract_path),
        "backend_contract_sha256": BACKEND_CONTRACT_SHA256,
        "backend_parameters": {
            backend: contract[backend]["parameters"] for backend in BACKENDS
        },
        "backend_canonical_parameter_sha256": {
            backend: contract[backend]["canonical_sha256"] for backend in BACKENDS
        },
        "pcl_executable": str(repository / "bin/pcl_point_to_plane_cli"),
        "pcl_executable_sha256": PCL_EXECUTABLE_SHA256,
        "directions_rows_sha256": _canonical_json_sha256(list(directions)),
        "trial_plan_rows_sha256": _canonical_json_sha256(list(trial_plan)),
        "zero_perturbation_reference_included_in_480": False,
        "rotation_perturbations_included": False,
        "parameter_modification_allowed": False,
        "outlier_query_exclusion_allowed": False,
        "scene_specific_parameters_allowed": False,
        "statistics_scope": "pilot descriptive / sensitivity analysis",
        "independence_warning": (
            "snapshot-level observations within the same scene/station are not independent scene-level replicates."
        ),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        **FLAGS,
    }


def run_benchmark(
    repository: Path,
    output: Path,
    *,
    runtime: Path = ZERO_RUNTIME,
) -> dict[str, Any]:
    repo = repository.expanduser().resolve(strict=True)
    destination = ensure_fresh_output(output)
    zero_runtime = runtime.expanduser().resolve(strict=True)
    baseline_before = authenticate_zero_runtime(zero_runtime)
    bag_hashes = authenticate_bags(BAG_ROOT)
    contract_path, contract = authenticate_contract(repo)
    pcl_executable = repo / "bin/pcl_point_to_plane_cli"
    verify_sha256_file(pcl_executable, PCL_EXECUTABLE_SHA256)
    validate_open3d_version()
    manifest_rows, sources, targets = load_and_authenticate_inputs(zero_runtime)
    geometry_summary = read_json(zero_runtime / "geometry_scene_summary.json")
    if geometry_summary.get("W_TEST_01_GEOMETRICALLY_WEAKER_THAN_R_TEST_01") is not True:
        raise PilotBagError("frozen geometry separation is not confirmed")

    directions = _geometry_directions(
        manifest_rows, sources, targets, zero_runtime
    )
    trial_plan = _build_trial_plan(manifest_rows, directions)
    destination.mkdir(parents=True)
    write_csv(destination / "directions.csv", directions)
    write_csv(destination / "trial_plan.csv", trial_plan)
    manifest_payload = _manifest(
        repo,
        destination,
        zero_runtime,
        contract_path,
        contract,
        baseline_before,
        bag_hashes,
        directions,
        trial_plan,
    )
    write_json(destination / "manifest.json", manifest_payload)
    print(
        f"PREEXECUTION_FROZEN directions={len(directions)} trials={len(trial_plan)} "
        f"manifest={destination / 'manifest.json'}",
        flush=True,
    )

    source_rows = {row["snapshot_id"]: row for row in manifest_rows}
    direction_rows = {
        (row["snapshot_id"], row["direction_class"]): row for row in directions
    }
    runs: list[dict[str, Any]] = []
    for plan in trial_plan:
        source_row = source_rows[str(plan["snapshot_id"])]
        result = _execute_one(
            plan,
            source_row,
            direction_rows[(str(plan["snapshot_id"]), str(plan["direction_class"]))],
            sources[str(plan["snapshot_id"])],
            targets[str(plan["scene_id"])],
            contract,
            pcl_executable,
        )
        runs.append(result)
        if len(runs) % 20 == 0 or len(runs) == EXPECTED_RUN_COUNT:
            write_csv(destination / "runs.csv", runs)
            print(
                f"PROGRESS {len(runs)}/{EXPECTED_RUN_COUNT} "
                f"solver_success={sum(row['solver_success'] is True for row in runs)} "
                f"finite={sum(row['finite_transform'] is True for row in runs)}",
                flush=True,
            )
    if len(runs) != EXPECTED_RUN_COUNT:
        raise PilotBagError("backend execution did not complete exactly 480 attempts")

    summary = build_summary(runs)
    rich_vs_weak = build_rich_vs_weak(runs)
    weak_vs_strong = build_weak_vs_strong(runs)
    paired = build_paired_differences(runs)
    agreement = build_backend_agreement(runs)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "rich_vs_weak.csv", rich_vs_weak)
    write_csv(destination / "weak_vs_strong.csv", weak_vs_strong)
    write_csv(destination / "paired_differences.csv", paired)
    write_csv(destination / "backend_agreement.csv", agreement)
    make_plots(destination, runs, summary, rich_vs_weak, weak_vs_strong)

    baseline_after = authenticate_zero_runtime(zero_runtime)
    checks = verify_run_rows(
        runs,
        trial_plan,
        baseline_before,
        baseline_after,
        repo,
        zero_runtime,
    )
    checks.update(
        {
            "summary_has_24_groups": len(summary) == 24,
            "rich_vs_weak_has_12_rows": len(rich_vs_weak) == 12,
            "weak_vs_strong_has_12_rows": len(weak_vs_strong) == 12,
            "paired_differences_has_240_rows": len(paired) == 240,
            "backend_agreement_has_13_rows": len(agreement) == 13,
            "geometry_direction_rows_40": len(directions) == 40,
            "all_direction_vectors_unit_norm": all(
                math.isclose(
                    float(row["direction_vector_norm"]), 1.0, rel_tol=0.0, abs_tol=1e-12
                )
                for row in directions
            ),
        }
    )
    verification_pass = all(checks.values())
    decision = build_decision_gate(
        runs,
        rich_vs_weak,
        weak_vs_strong,
        verification_pass=verification_pass,
    )
    write_json(destination / "decision_gate.json", decision)
    (destination / "verification_report.txt").write_text(
        _verification_report(checks, runs, decision, baseline_before),
        encoding="utf-8",
    )
    (destination / "README.md").write_text(
        _readme(decision, checks, runs), encoding="utf-8"
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (destination / name).is_file()]
    if missing:
        raise PilotBagError(f"required outputs missing: {missing}")
    _write_output_sha256s(destination)
    return {
        "output": str(destination),
        "checks": checks,
        "decision": decision,
        "run_count": len(runs),
        "solver_success_count": sum(row["solver_success"] is True for row in runs),
        "finite_count": sum(row["finite_transform"] is True for row in runs),
    }


def verify_completed_output(
    repository: Path,
    output: Path,
    *,
    runtime: Path = ZERO_RUNTIME,
) -> dict[str, Any]:
    """Read-only verification of a completed benchmark output."""

    repo = repository.expanduser().resolve(strict=True)
    destination = output.expanduser().resolve(strict=True)
    zero_runtime = runtime.expanduser().resolve(strict=True)
    manifest = read_json(destination / "manifest.json")
    baseline_before = manifest["zero_perturbation_runtime_file_sha256"]
    baseline_after = authenticate_zero_runtime(zero_runtime)
    runs = load_runs(destination / "runs.csv")
    plan_raw = read_csv(destination / "trial_plan.csv")
    plan: list[dict[str, Any]] = []
    for row in plan_raw:
        parsed: dict[str, Any] = dict(row)
        for field in ("T_star", "T_initial", "Delta_T"):
            parsed[field] = json.loads(row[field])
        parsed["perturbation_magnitude_m"] = float(row["perturbation_magnitude_m"])
        parsed["perturbation_sign"] = int(row["perturbation_sign"])
        plan.append(parsed)
    checks = verify_run_rows(
        runs, plan, baseline_before, baseline_after, repo, zero_runtime
    )
    checksum_rows = _read_sha256s(destination)
    checks["output_sha256s_all_match"] = all(
        sha256_file(destination / relative) == digest
        for relative, digest in checksum_rows.items()
    )
    checks["all_required_outputs_present"] = all(
        (destination / name).is_file() for name in REQUIRED_OUTPUTS
    )
    decision = read_json(destination / "decision_gate.json")
    checks["decision_formal_false"] = decision["FORMAL_MEASUREMENT_RESULT"] is False
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "run_count": len(runs),
        "decision": decision,
    }


__all__ = [
    "BACKEND_CONTRACT_SHA256",
    "EXPECTED_BAGS",
    "EXPECTED_TARGETS",
    "MAGNITUDES_M",
    "RECOVERY_ROTATION_DEG",
    "RECOVERY_TRANSLATION_M",
    "authenticate_bags",
    "authenticate_contract",
    "authenticate_zero_runtime",
    "build_run_id",
    "construct_translation_perturbation",
    "deterministic_eigendirections",
    "ensure_fresh_output",
    "identical_backend_input_pair",
    "pose_error",
    "pose_recovered",
    "run_benchmark",
    "verify_completed_output",
    "verify_sha256_file",
]
