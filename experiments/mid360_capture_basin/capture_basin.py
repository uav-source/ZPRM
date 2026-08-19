"""Measure the translation capture basin on frozen Mid-360 Pilot inputs.

All registration and geometry operations are delegated to the already frozen
controlled-perturbation implementation and production backend adapters.  This
module adds only experiment scheduling, a shared pre-ICP overlap proxy,
censoring/non-monotonicity handling, descriptive summaries, and verification.
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
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from experiments.mid360_controlled_perturbation import benchmark as controlled
from phase_a_harness.common_association_analysis import associate_source_points
from phase_a_harness.mid360_pilot.bag_reader import PilotBagError, sha256_file
from phase_a_harness.open3d_backend import validate_open3d_version


SCHEMA = "mid360_capture_basin_pilot_v1"
PREVIOUS_RESULT_RELATIVE = Path("results/mid360_controlled_perturbation_pilot")
PREVIOUS_SHA256SUMS_SHA256 = (
    "51e66ef1b8147add7162116f06a9957de51a075ab72dd9efd1f8afe89a34fe01"
)
PREVIOUS_MANIFEST_SHA256 = (
    "a51ecde8825ab4fd9ceeae1055075ad783a11be970ad21b7e4d18ca0618d541e"
)
PREVIOUS_RUNS_SHA256 = (
    "377bfc2e84ed268e1c8a0b20ed8c7f37aa90dac979eb4cc387d10242aae7b8e5"
)
COARSE_MAGNITUDES_M = (0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60)
AUTO_EXTENSION_MAGNITUDES_M = (2.40, 3.20, 4.80, 6.40)
MAX_TRANSLATION_PERTURBATION_M = 6.40
DIRECTION_CLASSES = ("weak", "strong")
SIGNS = (1, -1)
BACKENDS = ("open3d", "pcl")
DIRECTION_STABILITY_THRESHOLD = 0.90
LOW_INITIAL_OVERLAP_FRACTION_THRESHOLD = 0.01
BISECTION_RESOLUTION_M = 0.025
MAX_BISECTION_ITERATIONS = 8
EXPECTED_BASE_COARSE_ROWS = 1120
EXPECTED_REUSED_50MM_ROWS = 160
EXPECTED_NEW_BASE_COARSE_EXECUTIONS = 960
RIGHT_CENSORED_FRACTION_MAX_FOR_EXPANSION = 0.25
OVERLAP_CONFOUNDED_FRACTION = 0.50
MIN_BOUNDED_FRACTION_FOR_MEDIAN_COMPARISON = 0.50
SPATIAL_AXIS_INTERPRETATION = "UNRESOLVED"

FLAGS = {
    "PILOT_ONLY": True,
    "PILOT_NONFORMAL_DO_NOT_CITE": True,
    "FORMAL_MEASUREMENT_RESULT": False,
    "MEASUREMENT_EVIDENCE": False,
    "SPATIAL_AXIS_INTERPRETATION": SPATIAL_AXIS_INTERPRETATION,
}

REQUIRED_OUTPUTS = (
    "README.md",
    "manifest.json",
    "direction_audit.csv",
    "direction_audit.txt",
    "coarse_runs.csv",
    "boundary_runs.csv",
    "all_runs.csv",
    "capture_radius_per_snapshot.csv",
    "capture_radius_summary.csv",
    "rich_vs_weak_capture.csv",
    "weak_vs_strong_capture.csv",
    "sign_asymmetry.csv",
    "backend_agreement.csv",
    "decision_gate.json",
    "verification_report.txt",
    "01_recovery_rate_vs_translation_perturbation.png",
    "02_capture_radius_rich_vs_weak.png",
    "03_capture_radius_weak_vs_strong.png",
    "04_capture_radius_open3d_vs_pcl.png",
    "05_sign_asymmetry.png",
    "06_initial_overlap_vs_perturbation.png",
    "07_direction_stability.png",
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
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
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
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
    return value


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    if fields is None:
        if not rows:
            raise PilotBagError(f"fields required for empty CSV: {path}")
        fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def ensure_fresh_output(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise PilotBagError(f"output already exists; refusing overwrite: {destination}")
    return destination


def directory_sha256_inventory(root: Path) -> dict[str, str]:
    directory = root.resolve(strict=True)
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def authenticate_previous_pilot(repository: Path) -> dict[str, str]:
    previous = repository.resolve(strict=True) / PREVIOUS_RESULT_RELATIVE
    if sha256_file(previous / "SHA256SUMS") != PREVIOUS_SHA256SUMS_SHA256:
        raise PilotBagError("previous Pilot SHA256SUMS changed")
    if sha256_file(previous / "manifest.json") != PREVIOUS_MANIFEST_SHA256:
        raise PilotBagError("previous Pilot manifest changed")
    if sha256_file(previous / "runs.csv") != PREVIOUS_RUNS_SHA256:
        raise PilotBagError("previous Pilot runs.csv changed")
    listed: dict[str, str] = {}
    for line in (previous / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        listed[relative.lstrip(" *")] = digest
    for relative, digest in listed.items():
        if sha256_file(previous / relative) != digest:
            raise PilotBagError(f"previous Pilot artifact changed: {relative}")
    inventory = directory_sha256_inventory(previous)
    if set(inventory) != set(listed) | {"SHA256SUMS"}:
        raise PilotBagError("previous Pilot file inventory changed")
    decision = controlled.read_json(previous / "decision_gate.json")
    if not (
        decision.get("CONTROLLED_PERTURBATION_PILOT_READY") is True
        and decision.get("CONTROLLED_PERTURBATION_PILOT_SUPPORTS_EXPANSION") is False
        and decision.get("FORMAL_MEASUREMENT_RESULT") is False
    ):
        raise PilotBagError("previous Pilot decision changed")
    return inventory


def canonicalize_direction_sign(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise PilotBagError("direction must be a finite 3-vector")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise PilotBagError("direction norm is too small")
    result /= norm
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def _describe(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _direction_stability(vectors: Sequence[np.ndarray]) -> dict[str, Any]:
    cosines = [
        abs(float(np.dot(vectors[left], vectors[right])))
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    summary = _describe(cosines)
    mean = canonicalize_direction_sign(np.mean(np.asarray(vectors), axis=0))
    return {
        "pair_count": len(cosines),
        "absolute_cosine_median": summary["median"],
        "absolute_cosine_q25": summary["q25"],
        "absolute_cosine_q75": summary["q75"],
        "absolute_cosine_min": summary["min"],
        "representative_direction_x": float(mean[0]),
        "representative_direction_y": float(mean[1]),
        "representative_direction_z": float(mean[2]),
    }


def build_direction_audit(
    directions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], bool]:
    by_snapshot: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    vectors: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for row in directions:
        by_snapshot[str(row["snapshot_id"])][str(row["direction_class"])] = row
    output: list[dict[str, Any]] = []
    for snapshot_id in sorted(by_snapshot):
        pair = by_snapshot[snapshot_id]
        weak_row, strong_row = pair["weak"], pair["strong"]
        matrix = np.asarray(
            [
                [weak_row[f"information_h{r}{c}"] for c in range(3)]
                for r in range(3)
            ],
            dtype=np.float64,
        )
        eigenvalues, raw = np.linalg.eigh(matrix)
        weak_norm = float(np.linalg.norm(raw[:, 0]))
        strong_norm = float(np.linalg.norm(raw[:, 2]))
        weak = canonicalize_direction_sign(raw[:, 0])
        strong = canonicalize_direction_sign(raw[:, 2])
        vectors[(str(weak_row["scene_id"]), "weak")].append(weak)
        vectors[(str(weak_row["scene_id"]), "strong")].append(strong)
        output.append(
            {
                "scene_id": weak_row["scene_id"],
                "scene_class": weak_row["scene_class"],
                "snapshot_id": snapshot_id,
                "lambda_min": float(eigenvalues[0]),
                "lambda_mid": float(eigenvalues[1]),
                "lambda_max": float(eigenvalues[2]),
                "weak_direction_x": float(weak[0]),
                "weak_direction_y": float(weak[1]),
                "weak_direction_z": float(weak[2]),
                "strong_direction_x": float(strong[0]),
                "strong_direction_y": float(strong[1]),
                "strong_direction_z": float(strong[2]),
                "weak_translation_norm_before_normalization": weak_norm,
                "strong_translation_norm_before_normalization": strong_norm,
                "deterministic_sign_rule": "largest_absolute_component_nonnegative",
                **FLAGS,
            }
        )
    stability: dict[str, dict[str, Any]] = {}
    for scene_id in ("R_TEST_01", "W_TEST_01"):
        for direction in DIRECTION_CLASSES:
            key = f"{scene_id}:{direction}"
            stability[key] = {
                "scene_id": scene_id,
                "scene_class": controlled.EXPECTED_TARGETS[scene_id]["scene_class"],
                "direction_class": direction,
                **_direction_stability(vectors[(scene_id, direction)]),
            }
    valid = len(output) == 20 and all(
        math.isclose(
            float(row[field]), 1.0, rel_tol=0.0, abs_tol=1e-12
        )
        for row in output
        for field in (
            "weak_translation_norm_before_normalization",
            "strong_translation_norm_before_normalization",
        )
    )
    audit_pass = bool(
        valid
        and float(stability["W_TEST_01:weak"]["absolute_cosine_median"])
        >= DIRECTION_STABILITY_THRESHOLD
    )
    return output, stability, audit_pass


def direction_audit_text(
    stability: Mapping[str, Mapping[str, Any]], audit_pass: bool
) -> str:
    lines = [
        "MID-360 CAPTURE-BASIN DIRECTION AUDIT",
        f"DIRECTION_AUDIT_PASS={str(audit_pass).lower()}",
        f"threshold_absolute_cosine_median={DIRECTION_STABILITY_THRESHOLD}",
        "sign_rule=largest absolute component is nonnegative",
        "SPATIAL_AXIS_INTERPRETATION=UNRESOLVED",
        "No reliable Mid-360 mounting/scene-axis extrinsic was found; numerical directions only.",
        "",
    ]
    for key in sorted(stability):
        row = stability[key]
        lines.append(
            f"{key} median={row['absolute_cosine_median']:.12f} "
            f"q25={row['absolute_cosine_q25']:.12f} "
            f"q75={row['absolute_cosine_q75']:.12f} "
            f"min={row['absolute_cosine_min']:.12f} "
            f"representative=[{row['representative_direction_x']:.12f},"
            f"{row['representative_direction_y']:.12f},"
            f"{row['representative_direction_z']:.12f}]"
        )
    return "\n".join(lines) + "\n"


def magnitude_token(magnitude_m: float) -> str:
    magnitude = float(magnitude_m)
    if not (0.0 < magnitude <= MAX_TRANSLATION_PERTURBATION_M):
        raise PilotBagError("translation magnitude exceeds frozen cap")
    return f"M{int(round(magnitude * 1_000_000.0)):07d}UM"


def build_capture_run_id(
    phase: str,
    snapshot_id: str,
    direction_class: str,
    magnitude_m: float,
    sign: int,
    backend: str,
    iteration: int | None = None,
) -> str:
    if phase not in {"COARSE", "BOUNDARY"}:
        raise PilotBagError("invalid capture phase")
    if direction_class not in DIRECTION_CLASSES or backend not in BACKENDS:
        raise PilotBagError("invalid capture dimension")
    if sign not in SIGNS:
        raise PilotBagError("invalid sign")
    suffix = "O3D" if backend == "open3d" else "PCL"
    direction = "W" if direction_class == "weak" else "S"
    sign_name = "POS" if sign > 0 else "NEG"
    iteration_name = "" if iteration is None else f"-I{iteration:02d}"
    return (
        f"CB-{phase}-{snapshot_id}-{direction}-{magnitude_token(magnitude_m)}-"
        f"{sign_name}-{suffix}{iteration_name}"
    )


def detect_non_monotonic_recovery(recovery: Sequence[bool]) -> bool:
    failed = False
    for value in recovery:
        if not value:
            failed = True
        elif failed:
            return True
    return False


def classify_recovery_profile(
    profile: Sequence[tuple[float, bool]],
) -> dict[str, Any]:
    ordered = sorted((float(magnitude), bool(value)) for magnitude, value in profile)
    if not ordered or len({magnitude for magnitude, _ in ordered}) != len(ordered):
        raise PilotBagError("recovery profile is empty or has duplicate magnitudes")
    values = [value for _, value in ordered]
    non_monotonic = detect_non_monotonic_recovery(values)
    recovered = [magnitude for magnitude, value in ordered if value]
    failed = [magnitude for magnitude, value in ordered if not value]
    left = not values[0]
    right = bool(
        not non_monotonic
        and all(values)
        and math.isclose(ordered[-1][0], MAX_TRANSLATION_PERTURBATION_M)
    )
    bracket: tuple[float, float] | None = None
    if not non_monotonic and not left and failed:
        first_failed = min(failed)
        candidates = [value for value in recovered if value < first_failed]
        if candidates:
            bracket = (max(candidates), first_failed)
    return {
        "NON_MONOTONIC_RECOVERY_PATTERN": non_monotonic,
        "CAPTURE_RADIUS_AMBIGUOUS": non_monotonic,
        "LEFT_CENSORED": left,
        "RIGHT_CENSORED": right,
        "largest_recovered_coarse_magnitude": max(recovered, default=None),
        "smallest_failed_coarse_magnitude": min(failed, default=None),
        "bracket": bracket,
    }


def bisection_step(lower: float, upper: float) -> dict[str, Any]:
    a, b = float(lower), float(upper)
    if not (0.0 <= a < b <= MAX_TRANSLATION_PERTURBATION_M):
        raise PilotBagError("invalid bisection bracket")
    width = b - a
    return {
        "stop": width <= BISECTION_RESOLUTION_M + 1e-15,
        "mid": (a + b) / 2.0,
        "resolution_m": width,
    }


class OverlapCalculator:
    """One backend-independent 0.50 m nearest-target overlap proxy."""

    def __init__(self, targets: Mapping[str, np.ndarray]) -> None:
        self.targets = targets
        self.trees = {scene: cKDTree(target) for scene, target in targets.items()}

    def calculate(
        self, scene_id: str, source: np.ndarray, t_initial: np.ndarray
    ) -> dict[str, Any]:
        target = self.targets[scene_id]
        state = associate_source_points(
            source,
            t_initial,
            target_tree=self.trees[scene_id],
            target_point_count=target.shape[0],
        )
        fraction = float(state.count / source.shape[0])
        return {
            "initial_correspondence_count": state.count,
            "initial_correspondence_fraction": fraction,
            "LOW_INITIAL_OVERLAP": fraction
            <= LOW_INITIAL_OVERLAP_FRACTION_THRESHOLD,
            "initial_overlap_definition": (
                "shared 0.50m one-nearest-target association fraction"
            ),
        }


def _plan_row(
    source_row: Mapping[str, str],
    direction_row: Mapping[str, Any],
    magnitude_m: float,
    sign: int,
    backend: str,
    *,
    phase: str,
    iteration: int | None = None,
) -> dict[str, Any]:
    t_star = np.asarray(json.loads(source_row["T0"]), dtype=np.float64)
    direction = np.asarray(
        [direction_row[f"direction_vector_{axis}"] for axis in "xyz"],
        dtype=np.float64,
    )
    delta, initial, perturbation = controlled.construct_translation_perturbation(
        t_star, direction, magnitude_m, sign
    )
    return {
        "run_id": build_capture_run_id(
            phase,
            source_row["snapshot_id"],
            str(direction_row["direction_class"]),
            magnitude_m,
            sign,
            backend,
            iteration,
        ),
        "scene_id": source_row["scene_id"],
        "scene_class": controlled.EXPECTED_TARGETS[source_row["scene_id"]][
            "scene_class"
        ],
        "snapshot_id": source_row["snapshot_id"],
        "backend": backend,
        "direction_class": direction_row["direction_class"],
        "direction_source": direction_row["direction_source"],
        "direction_vector_x": float(direction[0]),
        "direction_vector_y": float(direction[1]),
        "direction_vector_z": float(direction[2]),
        "perturbation_sign": sign,
        "perturbation_magnitude_m": float(magnitude_m),
        "perturbation_vector_x_m": float(perturbation[0]),
        "perturbation_vector_y_m": float(perturbation[1]),
        "perturbation_vector_z_m": float(perturbation[2]),
        "Delta_T": delta.tolist(),
        "T_star": t_star.tolist(),
        "T_initial": initial.tolist(),
    }


_RESULT_FIELDS = (
    "T_est",
    "initial_translation_error_m",
    "final_translation_error_m",
    "initial_rotation_error_deg",
    "final_rotation_error_deg",
    "final_translation_error_vector_x_m",
    "final_translation_error_vector_y_m",
    "final_translation_error_vector_z_m",
    "translation_error_reduction_ratio",
    "solver_success",
    "finite_transform",
    "pose_recovered",
    "fitness",
    "rmse",
    "correspondence_count",
    "iteration_count",
    "runtime_ms",
    "backend_version",
    "failure_reason",
)


def _decorate_capture_run(
    result: Mapping[str, Any],
    overlap: Mapping[str, Any],
    *,
    phase: str,
    origin: str,
    new_icp_executed: bool,
    source_run_id: str | None = None,
    boundary_iteration: int | None = None,
) -> dict[str, Any]:
    return {
        **result,
        "capture_phase": phase,
        "execution_origin": origin,
        "new_icp_executed": new_icp_executed,
        "source_run_id": source_run_id,
        "boundary_iteration": boundary_iteration,
        **overlap,
        "converged_but_not_recovered": bool(
            result["solver_success"] is True
            and result["finite_transform"] is True
            and result["pose_recovered"] is False
        ),
        **FLAGS,
    }


def _typed_previous_runs(path: Path) -> list[dict[str, Any]]:
    return controlled.load_runs(path)


def _reused_numeric_input_matches(left: Any, right: Any) -> bool:
    """Accept JSON/CSV round-trip noise, but no meaningful input change."""

    return bool(
        np.allclose(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            rtol=0.0,
            atol=1e-15,
        )
    )


def _reuse_previous_50mm(
    previous_runs: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, str]],
    directions: Mapping[tuple[str, str], Mapping[str, Any]],
    sources: Mapping[str, np.ndarray],
    overlap_calculator: OverlapCalculator,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in previous_runs
        if math.isclose(float(row["perturbation_magnitude_m"]), 0.05)
    ]
    if len(selected) != EXPECTED_REUSED_50MM_ROWS:
        raise PilotBagError("previous Pilot does not contain exactly 160 50 mm rows")
    output: list[dict[str, Any]] = []
    for old in selected:
        source_row = source_rows[str(old["snapshot_id"])]
        direction_row = directions[
            (str(old["snapshot_id"]), str(old["direction_class"]))
        ]
        plan = _plan_row(
            source_row,
            direction_row,
            0.05,
            int(old["perturbation_sign"]),
            str(old["backend"]),
            phase="COARSE",
        )
        reconstructed_base = controlled._run_base(plan, source_row, direction_row)
        copied = {field: old[field] for field in _RESULT_FIELDS}
        if not all(
            _reused_numeric_input_matches(plan[field], old[field])
            for field in (
                "T_initial",
                "T_star",
            )
        ) or not _reused_numeric_input_matches(
            [plan[f"direction_vector_{axis}"] for axis in "xyz"],
            [old[f"direction_vector_{axis}"] for axis in "xyz"],
        ):
            raise PilotBagError(f"reused 50 mm pose mismatch: {old['run_id']}")
        # Preserve the authenticated previous input representation byte-for-byte
        # at the row-value level.  Only the capture-basin run ID is new.
        base = {**old, "run_id": reconstructed_base["run_id"]}
        overlap = overlap_calculator.calculate(
            str(plan["scene_id"]),
            sources[str(plan["snapshot_id"])],
            np.asarray(plan["T_initial"], dtype=np.float64),
        )
        output.append(
            _decorate_capture_run(
                {**base, **copied},
                overlap,
                phase="COARSE",
                origin="REUSED_PREVIOUS_CONTROLLED_PILOT_50MM",
                new_icp_executed=False,
                source_run_id=str(old["run_id"]),
            )
        )
    return output


def _execute_capture_run(
    source_row: Mapping[str, str],
    direction_row: Mapping[str, Any],
    source: np.ndarray,
    target: np.ndarray,
    contract: Mapping[str, Any],
    pcl_executable: Path,
    overlap_calculator: OverlapCalculator,
    magnitude_m: float,
    sign: int,
    backend: str,
    *,
    phase: str,
    origin: str,
    boundary_iteration: int | None = None,
) -> dict[str, Any]:
    plan = _plan_row(
        source_row,
        direction_row,
        magnitude_m,
        sign,
        backend,
        phase=phase,
        iteration=boundary_iteration,
    )
    overlap = overlap_calculator.calculate(
        str(plan["scene_id"]),
        source,
        np.asarray(plan["T_initial"], dtype=np.float64),
    )
    result = controlled._execute_one(
        plan,
        source_row,
        direction_row,
        source,
        target,
        contract,
        pcl_executable,
    )
    return _decorate_capture_run(
        result,
        overlap,
        phase=phase,
        origin=origin,
        new_icp_executed=True,
        boundary_iteration=boundary_iteration,
    )


def _physical_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["snapshot_id"]),
        str(row["direction_class"]),
        int(row["perturbation_sign"]),
    )


def _backend_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["snapshot_id"]),
        str(row["backend"]),
        str(row["direction_class"]),
        int(row["perturbation_sign"]),
    )


def _profile_rows(
    rows: Sequence[Mapping[str, Any]], key: tuple[str, str, str, int]
) -> list[Mapping[str, Any]]:
    return sorted(
        (row for row in rows if _backend_key(row) == key),
        key=lambda row: float(row["perturbation_magnitude_m"]),
    )


def _run_base_coarse(
    manifest_rows: Sequence[Mapping[str, str]],
    source_rows: Mapping[str, Mapping[str, str]],
    directions: Mapping[tuple[str, str], Mapping[str, Any]],
    sources: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    pcl_executable: Path,
    overlap_calculator: OverlapCalculator,
    reused: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    rows = list(reused)
    for source_row in manifest_rows:
        snapshot_id = source_row["snapshot_id"]
        for direction_class in DIRECTION_CLASSES:
            direction_row = directions[(snapshot_id, direction_class)]
            for magnitude in COARSE_MAGNITUDES_M[1:]:
                for sign in SIGNS:
                    for backend in BACKENDS:
                        rows.append(
                            _execute_capture_run(
                                source_row,
                                direction_row,
                                sources[snapshot_id],
                                targets[source_row["scene_id"]],
                                contract,
                                pcl_executable,
                                overlap_calculator,
                                magnitude,
                                sign,
                                backend,
                                phase="COARSE",
                                origin="NEW_BASE_COARSE_EXECUTION",
                            )
                        )
                        new_count = sum(row["new_icp_executed"] is True for row in rows)
                        if new_count % 20 == 0:
                            write_csv(output / "coarse_runs.csv", rows)
                            print(
                                f"BASE_COARSE_PROGRESS new={new_count}/"
                                f"{EXPECTED_NEW_BASE_COARSE_EXECUTIONS} total={len(rows)}",
                                flush=True,
                            )
    if len(rows) != EXPECTED_BASE_COARSE_ROWS:
        raise PilotBagError(f"base coarse grid is not 1120 rows: {len(rows)}")
    return rows


def _run_extensions(
    rows: list[dict[str, Any]],
    manifest_rows: Sequence[Mapping[str, str]],
    source_rows: Mapping[str, Mapping[str, str]],
    directions: Mapping[tuple[str, str], Mapping[str, Any]],
    sources: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    pcl_executable: Path,
    overlap_calculator: OverlapCalculator,
    output: Path,
) -> int:
    by_physical: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if math.isclose(float(row["perturbation_magnitude_m"]), 1.60):
            by_physical[_physical_key(row)][str(row["backend"])] = row
    triggers = {
        key
        for key, pair in by_physical.items()
        if any(row["pose_recovered"] is True for row in pair.values())
    }
    executed = 0
    for snapshot_id, direction_class, sign in sorted(triggers):
        source_row = source_rows[snapshot_id]
        direction_row = directions[(snapshot_id, direction_class)]
        for magnitude in AUTO_EXTENSION_MAGNITUDES_M:
            for backend in BACKENDS:
                rows.append(
                    _execute_capture_run(
                        source_row,
                        direction_row,
                        sources[snapshot_id],
                        targets[source_row["scene_id"]],
                        contract,
                        pcl_executable,
                        overlap_calculator,
                        magnitude,
                        sign,
                        backend,
                        phase="COARSE",
                        origin="NEW_AUTO_EXTENSION_EXECUTION",
                    )
                )
                executed += 1
                if executed % 20 == 0:
                    write_csv(output / "coarse_runs.csv", rows)
                    print(
                        f"EXTENSION_PROGRESS new={executed} physical_triggers={len(triggers)}",
                        flush=True,
                    )
    return executed


def _run_boundaries(
    coarse_rows: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, str]],
    directions: Mapping[tuple[str, str], Mapping[str, Any]],
    sources: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    pcl_executable: Path,
    overlap_calculator: OverlapCalculator,
    output: Path,
) -> list[dict[str, Any]]:
    keys = sorted({_backend_key(row) for row in coarse_rows})
    boundary: list[dict[str, Any]] = []
    for snapshot_id, backend, direction_class, sign in keys:
        profile = _profile_rows(coarse_rows, (snapshot_id, backend, direction_class, sign))
        classified = classify_recovery_profile(
            [
                (float(row["perturbation_magnitude_m"]), row["pose_recovered"] is True)
                for row in profile
            ]
        )
        bracket = classified["bracket"]
        if bracket is None or classified["CAPTURE_RADIUS_AMBIGUOUS"]:
            continue
        lower, upper = bracket
        source_row = source_rows[snapshot_id]
        direction_row = directions[(snapshot_id, direction_class)]
        for iteration in range(1, MAX_BISECTION_ITERATIONS + 1):
            step = bisection_step(lower, upper)
            if step["stop"]:
                break
            mid = float(step["mid"])
            result = _execute_capture_run(
                source_row,
                direction_row,
                sources[snapshot_id],
                targets[source_row["scene_id"]],
                contract,
                pcl_executable,
                overlap_calculator,
                mid,
                sign,
                backend,
                phase="BOUNDARY",
                origin="NEW_BISECTION_BOUNDARY_EXECUTION",
                boundary_iteration=iteration,
            )
            result["bracket_lower_before_m"] = lower
            result["bracket_upper_before_m"] = upper
            if result["pose_recovered"] is True:
                lower = mid
            else:
                upper = mid
            result["bracket_lower_after_m"] = lower
            result["bracket_upper_after_m"] = upper
            boundary.append(result)
            if len(boundary) % 20 == 0:
                write_csv(output / "boundary_runs.csv", boundary)
                print(f"BOUNDARY_PROGRESS new={len(boundary)}", flush=True)
    return boundary


def _row_at_magnitude(
    rows: Sequence[Mapping[str, Any]], magnitude: float
) -> Mapping[str, Any] | None:
    matches = [
        row
        for row in rows
        if math.isclose(
            float(row["perturbation_magnitude_m"]),
            float(magnitude),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    return matches[-1] if matches else None


def build_capture_radius_records(
    coarse_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted({_backend_key(row) for row in coarse_rows})
    output: list[dict[str, Any]] = []
    for key in keys:
        snapshot_id, backend, direction_class, sign = key
        coarse = _profile_rows(coarse_rows, key)
        boundary = sorted(
            (row for row in boundary_rows if _backend_key(row) == key),
            key=lambda row: int(row["boundary_iteration"]),
        )
        classified = classify_recovery_profile(
            [
                (float(row["perturbation_magnitude_m"]), row["pose_recovered"] is True)
                for row in coarse
            ]
        )
        lower: float | None = None
        upper: float | None = None
        if not classified["CAPTURE_RADIUS_AMBIGUOUS"]:
            if boundary:
                lower = float(boundary[-1]["bracket_lower_after_m"])
                upper = float(boundary[-1]["bracket_upper_after_m"])
            elif classified["bracket"] is not None:
                lower, upper = classified["bracket"]
            elif classified["RIGHT_CENSORED"]:
                lower = float(classified["largest_recovered_coarse_magnitude"])
            elif classified["LEFT_CENSORED"]:
                upper = float(coarse[0]["perturbation_magnitude_m"])
        mid = None if lower is None or upper is None else (lower + upper) / 2.0
        resolution = None if lower is None or upper is None else upper - lower
        lower_row = None if lower is None else _row_at_magnitude([*coarse, *boundary], lower)
        upper_row = None if upper is None else _row_at_magnitude([*coarse, *boundary], upper)
        first = coarse[0]
        output.append(
            {
                "scene_id": first["scene_id"],
                "scene_class": first["scene_class"],
                "snapshot_id": snapshot_id,
                "backend": backend,
                "direction_class": direction_class,
                "perturbation_sign": sign,
                "largest_recovered_coarse_magnitude": classified[
                    "largest_recovered_coarse_magnitude"
                ],
                "smallest_failed_coarse_magnitude": classified[
                    "smallest_failed_coarse_magnitude"
                ],
                "capture_radius_lower_m": lower,
                "capture_radius_upper_m": upper,
                "capture_radius_mid_m": mid,
                "capture_radius_resolution_m": resolution,
                "boundary_refinement_run_count": len(boundary),
                "LEFT_CENSORED": classified["LEFT_CENSORED"],
                "RIGHT_CENSORED": classified["RIGHT_CENSORED"],
                "BOUND_NOT_FOUND": classified["RIGHT_CENSORED"],
                "NON_MONOTONIC_RECOVERY_PATTERN": classified[
                    "NON_MONOTONIC_RECOVERY_PATTERN"
                ],
                "CAPTURE_RADIUS_AMBIGUOUS": classified[
                    "CAPTURE_RADIUS_AMBIGUOUS"
                ],
                "lower_initial_correspondence_fraction": (
                    None
                    if lower_row is None
                    else float(lower_row["initial_correspondence_fraction"])
                ),
                "upper_initial_correspondence_fraction": (
                    None
                    if upper_row is None
                    else float(upper_row["initial_correspondence_fraction"])
                ),
                "upper_bound_low_initial_overlap": (
                    False
                    if upper_row is None
                    else upper_row["LOW_INITIAL_OVERLAP"] is True
                ),
                "coarse_profile_json": [
                    {
                        "magnitude_m": float(row["perturbation_magnitude_m"]),
                        "pose_recovered": row["pose_recovered"] is True,
                        "initial_correspondence_fraction": float(
                            row["initial_correspondence_fraction"]
                        ),
                    }
                    for row in coarse
                ],
                **FLAGS,
            }
        )
    if len(output) != 160:
        raise PilotBagError(f"capture radius record count is not 160: {len(output)}")
    return output


def _numbers(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if row.get(field) is not None
        and row.get(field) != ""
        and math.isfinite(float(row[field]))
    ]


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(sum(row[field] is True for row in rows) / len(rows))


def build_capture_radius_summary(
    captures: Sequence[Mapping[str, Any]],
    coarse_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    all_magnitudes = (*COARSE_MAGNITUDES_M[1:], *AUTO_EXTENSION_MAGNITUDES_M)
    for scene in ("Rich", "Weak"):
        for backend in BACKENDS:
            for direction in DIRECTION_CLASSES:
                for sign in SIGNS:
                    selected = [
                        row
                        for row in captures
                        if row["scene_class"] == scene
                        and row["backend"] == backend
                        and row["direction_class"] == direction
                        and int(row["perturbation_sign"]) == sign
                    ]
                    lower = _describe(_numbers(selected, "capture_radius_lower_m"))
                    mid = _describe(_numbers(selected, "capture_radius_mid_m"))
                    row: dict[str, Any] = {
                        "scene_class": scene,
                        "backend": backend,
                        "direction_class": direction,
                        "sign": sign,
                        "n": len(selected),
                        "capture_radius_lower_m_median": lower["median"],
                        "capture_radius_lower_m_q25": lower["q25"],
                        "capture_radius_lower_m_q75": lower["q75"],
                        "capture_radius_lower_m_min": lower["min"],
                        "capture_radius_lower_m_max": lower["max"],
                        "capture_radius_mid_m_median": mid["median"],
                        "capture_radius_mid_m_q25": mid["q25"],
                        "capture_radius_mid_m_q75": mid["q75"],
                        "bounded_mid_count": mid["count"],
                        "right_censored_count": sum(
                            item["RIGHT_CENSORED"] is True for item in selected
                        ),
                        "left_censored_count": sum(
                            item["LEFT_CENSORED"] is True for item in selected
                        ),
                        "ambiguous_count": sum(
                            item["CAPTURE_RADIUS_AMBIGUOUS"] is True
                            for item in selected
                        ),
                    }
                    for magnitude in all_magnitudes:
                        magnitude_rows = [
                            item
                            for item in coarse_rows
                            if item["scene_class"] == scene
                            and item["backend"] == backend
                            and item["direction_class"] == direction
                            and int(item["perturbation_sign"]) == sign
                            and math.isclose(
                                float(item["perturbation_magnitude_m"]), magnitude
                            )
                        ]
                        label = f"{magnitude:.2f}".replace(".", "p")
                        row[f"n_at_{label}m"] = len(magnitude_rows)
                        row[f"recovery_rate_at_{label}m"] = (
                            None
                            if not magnitude_rows
                            else _rate(magnitude_rows, "pose_recovered")
                        )
                    output.append({**row, **FLAGS})
    return output


def _capture_mids(
    rows: Sequence[Mapping[str, Any]],
) -> list[float]:
    return _numbers(
        [
            row
            for row in rows
            if row["CAPTURE_RADIUS_AMBIGUOUS"] is False
            and row["LEFT_CENSORED"] is False
            and row["RIGHT_CENSORED"] is False
        ],
        "capture_radius_mid_m",
    )


def build_rich_vs_weak_capture(
    captures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for backend in BACKENDS:
        for direction in DIRECTION_CLASSES:
            for sign_label, sign in (("ALL", None), ("+", 1), ("-", -1)):
                groups = {}
                for scene in ("Rich", "Weak"):
                    selected = [
                        row
                        for row in captures
                        if row["scene_class"] == scene
                        and row["backend"] == backend
                        and row["direction_class"] == direction
                        and (sign is None or int(row["perturbation_sign"]) == sign)
                    ]
                    groups[scene] = selected
                rich = _describe(_capture_mids(groups["Rich"]))
                weak = _describe(_capture_mids(groups["Weak"]))
                output.append(
                    {
                        "backend": backend,
                        "direction_class": direction,
                        "sign": sign_label,
                        "rich_n": len(groups["Rich"]),
                        "weak_n": len(groups["Weak"]),
                        "rich_bounded_n": rich["count"],
                        "weak_bounded_n": weak["count"],
                        "rich_median_capture_radius_mid_m": rich["median"],
                        "weak_median_capture_radius_mid_m": weak["median"],
                        "weak_rich_capture_ratio": (
                            None
                            if rich["median"] in (None, 0.0) or weak["median"] is None
                            else float(weak["median"]) / float(rich["median"])
                        ),
                        "weak_smaller_than_rich": bool(
                            rich["median"] is not None
                            and weak["median"] is not None
                            and float(weak["median"]) < float(rich["median"])
                        ),
                        **FLAGS,
                    }
                )
    return output


def build_weak_vs_strong_capture(
    captures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scene in ("Rich", "Weak"):
        for backend in BACKENDS:
            for sign_label, sign in (("ALL", None), ("+", 1), ("-", -1)):
                groups = {}
                for direction in DIRECTION_CLASSES:
                    selected = [
                        row
                        for row in captures
                        if row["scene_class"] == scene
                        and row["backend"] == backend
                        and row["direction_class"] == direction
                        and (sign is None or int(row["perturbation_sign"]) == sign)
                    ]
                    groups[direction] = selected
                weak = _describe(_capture_mids(groups["weak"]))
                strong = _describe(_capture_mids(groups["strong"]))
                output.append(
                    {
                        "scene_class": scene,
                        "backend": backend,
                        "sign": sign_label,
                        "weak_bounded_n": weak["count"],
                        "strong_bounded_n": strong["count"],
                        "weak_median_capture_radius_mid_m": weak["median"],
                        "strong_median_capture_radius_mid_m": strong["median"],
                        "weak_strong_capture_ratio": (
                            None
                            if strong["median"] in (None, 0.0)
                            or weak["median"] is None
                            else float(weak["median"]) / float(strong["median"])
                        ),
                        "weak_smaller_than_strong": bool(
                            weak["median"] is not None
                            and strong["median"] is not None
                            and float(weak["median"]) < float(strong["median"])
                        ),
                        **FLAGS,
                    }
                )
    return output


def build_sign_asymmetry(
    captures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in captures:
        grouped[
            (
                str(row["snapshot_id"]),
                str(row["backend"]),
                str(row["direction_class"]),
            )
        ][int(row["perturbation_sign"])] = row
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        pair = grouped[key]
        plus, minus = pair[1], pair[-1]
        plus_radius = plus.get("capture_radius_mid_m")
        minus_radius = minus.get("capture_radius_mid_m")
        ratio = None
        absolute = None
        if plus_radius is not None and minus_radius is not None:
            lo = min(float(plus_radius), float(minus_radius))
            hi = max(float(plus_radius), float(minus_radius))
            ratio = None if hi <= 0.0 else lo / hi
            absolute = abs(float(plus_radius) - float(minus_radius))
        output.append(
            {
                "scene_id": plus["scene_id"],
                "scene_class": plus["scene_class"],
                "snapshot_id": key[0],
                "backend": key[1],
                "direction_class": key[2],
                "capture_radius_plus_m": plus_radius,
                "capture_radius_minus_m": minus_radius,
                "paired_absolute_difference_m": absolute,
                "sign_asymmetry_ratio": ratio,
                "comparable_bounded_pair": ratio is not None,
                **FLAGS,
            }
        )
    return output


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def build_backend_agreement_capture(
    captures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    paired: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in captures:
        paired[
            (
                str(row["snapshot_id"]),
                str(row["direction_class"]),
                int(row["perturbation_sign"]),
            )
        ][str(row["backend"])] = row
    detail: list[dict[str, Any]] = []
    for key in sorted(paired):
        left, right = paired[key]["open3d"], paired[key]["pcl"]
        open_radius = left.get("capture_radius_mid_m")
        pcl_radius = right.get("capture_radius_mid_m")
        detail.append(
            {
                "row_type": "PAIR",
                "scene_class": left["scene_class"],
                "snapshot_id": key[0],
                "direction_class": key[1],
                "sign": key[2],
                "open3d_capture_radius_mid_m": open_radius,
                "pcl_capture_radius_mid_m": pcl_radius,
                "paired_absolute_difference_m": (
                    None
                    if open_radius is None or pcl_radius is None
                    else abs(float(open_radius) - float(pcl_radius))
                ),
                "both_bounded": open_radius is not None and pcl_radius is not None,
                "capture_radius_spearman": None,
                "median_paired_absolute_difference_m": None,
                "directional_conclusion_agreement_rate": None,
                **FLAGS,
            }
        )
    bounded = [row for row in detail if row["both_bounded"]]
    directional: list[bool] = []
    for scene in ("Rich", "Weak"):
        for snapshot_id in sorted(
            {row["snapshot_id"] for row in captures if row["scene_class"] == scene}
        ):
            for sign in SIGNS:
                conclusions = []
                for backend in BACKENDS:
                    weak = next(
                        row
                        for row in captures
                        if row["snapshot_id"] == snapshot_id
                        and row["backend"] == backend
                        and row["direction_class"] == "weak"
                        and int(row["perturbation_sign"]) == sign
                    )
                    strong = next(
                        row
                        for row in captures
                        if row["snapshot_id"] == snapshot_id
                        and row["backend"] == backend
                        and row["direction_class"] == "strong"
                        and int(row["perturbation_sign"]) == sign
                    )
                    if (
                        weak["capture_radius_mid_m"] is not None
                        and strong["capture_radius_mid_m"] is not None
                    ):
                        conclusions.append(
                            float(weak["capture_radius_mid_m"])
                            < float(strong["capture_radius_mid_m"])
                        )
                if len(conclusions) == 2:
                    directional.append(conclusions[0] == conclusions[1])
    summary = {
        "row_type": "SUMMARY",
        "scene_class": "ALL",
        "snapshot_id": "ALL",
        "direction_class": "ALL",
        "sign": "ALL",
        "open3d_capture_radius_mid_m": None,
        "pcl_capture_radius_mid_m": None,
        "paired_absolute_difference_m": None,
        "both_bounded": None,
        "capture_radius_spearman": _spearman(
            [float(row["open3d_capture_radius_mid_m"]) for row in bounded],
            [float(row["pcl_capture_radius_mid_m"]) for row in bounded],
        ),
        "median_paired_absolute_difference_m": (
            _describe(_numbers(bounded, "paired_absolute_difference_m"))["median"]
        ),
        "directional_conclusion_agreement_rate": (
            None if not directional else float(sum(directional) / len(directional))
        ),
        **FLAGS,
    }
    return [summary, *detail]


def _all_sign_row(
    rows: Sequence[Mapping[str, Any]], **criteria: Any
) -> Mapping[str, Any]:
    return next(
        row
        for row in rows
        if row.get("sign") == "ALL"
        and all(row.get(field) == value for field, value in criteria.items())
    )


def build_decision_gate(
    captures: Sequence[Mapping[str, Any]],
    rich_vs_weak: Sequence[Mapping[str, Any]],
    weak_vs_strong: Sequence[Mapping[str, Any]],
    *,
    direction_audit_pass: bool,
    previous_unchanged: bool,
    verification_pass: bool,
) -> dict[str, Any]:
    def rich_weak(backend: str) -> bool:
        row = _all_sign_row(
            rich_vs_weak, backend=backend, direction_class="weak"
        )
        enough = (
            int(row["rich_bounded_n"]) / int(row["rich_n"])
            >= MIN_BOUNDED_FRACTION_FOR_MEDIAN_COMPARISON
            and int(row["weak_bounded_n"]) / int(row["weak_n"])
            >= MIN_BOUNDED_FRACTION_FOR_MEDIAN_COMPARISON
        )
        return bool(enough and row["weak_smaller_than_rich"] is True)

    def weak_strong(scene: str, backend: str) -> bool:
        row = _all_sign_row(
            weak_vs_strong, scene_class=scene, backend=backend
        )
        return bool(row["weak_smaller_than_strong"] is True)

    conclusions = {
        "WEAK_SCENE_SMALLER_CAPTURE_BASIN_OPEN3D": rich_weak("open3d"),
        "WEAK_SCENE_SMALLER_CAPTURE_BASIN_PCL": rich_weak("pcl"),
        "WEAK_DIRECTION_SMALLER_CAPTURE_BASIN_RICH_OPEN3D": weak_strong(
            "Rich", "open3d"
        ),
        "WEAK_DIRECTION_SMALLER_CAPTURE_BASIN_RICH_PCL": weak_strong(
            "Rich", "pcl"
        ),
        "WEAK_DIRECTION_SMALLER_CAPTURE_BASIN_WEAK_OPEN3D": weak_strong(
            "Weak", "open3d"
        ),
        "WEAK_DIRECTION_SMALLER_CAPTURE_BASIN_WEAK_PCL": weak_strong(
            "Weak", "pcl"
        ),
    }
    backend_consistent = bool(
        conclusions["WEAK_SCENE_SMALLER_CAPTURE_BASIN_OPEN3D"]
        == conclusions["WEAK_SCENE_SMALLER_CAPTURE_BASIN_PCL"]
        and conclusions["WEAK_DIRECTION_SMALLER_CAPTURE_BASIN_RICH_OPEN3D"]
        == conclusions["WEAK_DIRECTION_SMALLER_CAPTURE_BASIN_RICH_PCL"]
        and conclusions["WEAK_DIRECTION_SMALLER_CAPTURE_BASIN_WEAK_OPEN3D"]
        == conclusions["WEAK_DIRECTION_SMALLER_CAPTURE_BASIN_WEAK_PCL"]
    )
    weak_failures = [
        row
        for row in captures
        if row["scene_class"] == "Weak"
        and row["direction_class"] == "weak"
        and row["capture_radius_upper_m"] is not None
    ]
    low_overlap_failure_fraction = (
        None
        if not weak_failures
        else float(
            sum(row["upper_bound_low_initial_overlap"] is True for row in weak_failures)
            / len(weak_failures)
        )
    )
    overlap_confounded = bool(
        low_overlap_failure_fraction is not None
        and low_overlap_failure_fraction >= OVERLAP_CONFOUNDED_FRACTION
    )
    right_fraction = float(
        sum(row["RIGHT_CENSORED"] is True for row in captures) / len(captures)
    )
    boundary_observed = any(
        row["capture_radius_lower_m"] is not None
        and row["capture_radius_upper_m"] is not None
        and row["CAPTURE_RADIUS_AMBIGUOUS"] is False
        for row in captures
    )
    supports = bool(
        direction_audit_pass
        and conclusions["WEAK_SCENE_SMALLER_CAPTURE_BASIN_OPEN3D"]
        and conclusions["WEAK_SCENE_SMALLER_CAPTURE_BASIN_PCL"]
        and conclusions["WEAK_DIRECTION_SMALLER_CAPTURE_BASIN_WEAK_OPEN3D"]
        and conclusions["WEAK_DIRECTION_SMALLER_CAPTURE_BASIN_WEAK_PCL"]
        and not overlap_confounded
        and right_fraction <= RIGHT_CENSORED_FRACTION_MAX_FOR_EXPANSION
        and boundary_observed
    )
    return {
        "schema": f"{SCHEMA}_decision_gate",
        "gate_preregistered_before_backend_execution": True,
        "CAPTURE_BASIN_PILOT_READY": bool(
            verification_pass and previous_unchanged and direction_audit_pass
        ),
        "PREVIOUS_PILOT_UNCHANGED": previous_unchanged,
        "DIRECTION_AUDIT_PASS": direction_audit_pass,
        "CAPTURE_BOUNDARY_OBSERVED": boundary_observed,
        **conclusions,
        "BACKEND_CAPTURE_DIRECTIONALLY_CONSISTENT": backend_consistent,
        "LOW_OVERLAP_FAILURE_FRACTION_WEAK_WEAK_DIRECTION": (
            low_overlap_failure_fraction
        ),
        "EFFECT_CONFOUNDED_BY_INITIAL_OVERLAP_GEOMETRY": overlap_confounded,
        "RIGHT_CENSORED_FRACTION": right_fraction,
        "CAPTURE_BASIN_PILOT_SUPPORTS_FORMAL_EXPANSION": supports,
        "FORMAL_MEASUREMENT_RESULT": False,
        **FLAGS,
    }


def _capture_plot_values(
    captures: Sequence[Mapping[str, Any]],
    **criteria: Any,
) -> list[float]:
    return _capture_mids(
        [
            row
            for row in captures
            if all(row.get(field) == value for field, value in criteria.items())
        ]
    )


def make_plots(
    output: Path,
    coarse: Sequence[Mapping[str, Any]],
    captures: Sequence[Mapping[str, Any]],
    asymmetry: Sequence[Mapping[str, Any]],
    agreement: Sequence[Mapping[str, Any]],
    stability: Mapping[str, Mapping[str, Any]],
) -> None:
    # 1: four panels keep scene and backend separate; direction is line style.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for row_index, scene in enumerate(("Rich", "Weak")):
        for col_index, backend in enumerate(BACKENDS):
            axis = axes[row_index, col_index]
            for direction, color in (("weak", "#D55E00"), ("strong", "#0072B2")):
                xs, ys = [], []
                for magnitude in (*COARSE_MAGNITUDES_M, *AUTO_EXTENSION_MAGNITUDES_M):
                    selected = [
                        row
                        for row in coarse
                        if row["scene_class"] == scene
                        and row["backend"] == backend
                        and row["direction_class"] == direction
                        and math.isclose(float(row["perturbation_magnitude_m"]), magnitude)
                    ]
                    if selected:
                        xs.append(magnitude)
                        ys.append(_rate(selected, "pose_recovered"))
                axis.plot(xs, ys, marker="o", color=color, label=direction)
            axis.set_title(f"{scene} / {backend.upper()}")
            axis.set_ylim(0.0, 1.05)
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("Initial translation perturbation [m]")
            if col_index == 0:
                axis.set_ylabel("Recovery rate")
            axis.legend(loc="best")
    fig.suptitle("Capture-basin recovery profiles (Pilot, nonformal)")
    fig.tight_layout()
    fig.savefig(output / "01_recovery_rate_vs_translation_perturbation.png", dpi=180)
    plt.close(fig)
    _make_remaining_plots(output, coarse, captures, asymmetry, agreement, stability)


def _union_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def _manifest(
    repository: Path,
    output: Path,
    previous_inventory: Mapping[str, str],
    zero_inventory: Mapping[str, str],
    direction_stability: Mapping[str, Mapping[str, Any]],
    direction_audit_pass: bool,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}_manifest",
        "created_before_backend_execution": True,
        "created_at_utc": utc_now(),
        "repository": str(repository),
        "output_directory": str(output),
        "previous_pilot_directory": str(repository / PREVIOUS_RESULT_RELATIVE),
        "previous_pilot_inventory_sha256": dict(previous_inventory),
        "zero_perturbation_runtime_inventory_sha256": dict(zero_inventory),
        "snapshot_count": 20,
        "scene_contract": controlled.EXPECTED_TARGETS,
        "backend_contract_sha256": controlled.BACKEND_CONTRACT_SHA256,
        "backend_parameters": {
            backend: contract[backend]["parameters"] for backend in BACKENDS
        },
        "coarse_magnitudes_m": list(COARSE_MAGNITUDES_M),
        "auto_extension_magnitudes_m": list(AUTO_EXTENSION_MAGNITUDES_M),
        "MAX_TRANSLATION_PERTURBATION_M": MAX_TRANSLATION_PERTURBATION_M,
        "coarse_grid_total_rows": EXPECTED_BASE_COARSE_ROWS,
        "previous_50mm_rows_reused": EXPECTED_REUSED_50MM_ROWS,
        "new_base_coarse_backend_executions": EXPECTED_NEW_BASE_COARSE_EXECUTIONS,
        "fifty_mm_policy": "REUSED_BYTE_AUTHENTICATED_PREVIOUS_RESULTS",
        "extension_trigger": (
            "For one snapshot/direction/sign physical input, if either backend "
            "recovers at 1.6m, execute both backends at every frozen extension magnitude."
        ),
        "direction_contract": {
            "definition": "existing 3x3 translation H; lambda_min weak, lambda_max strong",
            "sign_canonicalization": "largest_absolute_component_nonnegative",
            "sign_canonicalization_scope": "audit/comparison/reporting only",
            "stability_metric": "within-scene pairwise absolute cosine similarity",
            "DIRECTION_STABILITY_THRESHOLD": DIRECTION_STABILITY_THRESHOLD,
            "audit_pass": direction_audit_pass,
            "stability": direction_stability,
            "SPATIAL_AXIS_INTERPRETATION": SPATIAL_AXIS_INTERPRETATION,
        },
        "overlap_contract": {
            "definition": "shared 0.50m one-nearest-target association fraction at T_initial",
            "LOW_INITIAL_OVERLAP_FRACTION_THRESHOLD": LOW_INITIAL_OVERLAP_FRACTION_THRESHOLD,
            "backend_independent": True,
            "used_to_change_icp": False,
        },
        "recovery_contract": {
            "translation_max_inclusive_m": controlled.RECOVERY_TRANSLATION_M,
            "rotation_max_inclusive_deg": controlled.RECOVERY_ROTATION_DEG,
            "unchanged_from_previous_pilot": True,
        },
        "boundary_contract": {
            "method": "bisection on monotonic success/failure bracket only",
            "resolution_max_m": BISECTION_RESOLUTION_M,
            "maximum_iterations": MAX_BISECTION_ITERATIONS,
            "non_monotonic_action": "AMBIGUOUS_NO_BISECTION",
            "right_censored_at_m": MAX_TRANSLATION_PERTURBATION_M,
        },
        "decision_gate_contract": {
            "condition_1": "Weak scene median weak-direction radius < Rich for both backends",
            "condition_2": "Weak scene weak-direction radius < strong-direction for both backends",
            "condition_3": (
                f"Weak weak-direction upper failures with overlap<=1% must be <"
                f"{OVERLAP_CONFOUNDED_FRACTION:.0%}"
            ),
            "condition_4": (
                f"overall right-censored fraction <= {RIGHT_CENSORED_FRACTION_MAX_FOR_EXPANSION:.0%}"
            ),
            "minimum_bounded_fraction_for_median": MIN_BOUNDED_FRACTION_FOR_MEDIAN_COMPARISON,
        },
        "rotation_perturbation_included": False,
        "parameter_modification_allowed": False,
        "statistics_scope": "pilot descriptive sensitivity analysis",
        "independence_warning": (
            "10 snapshots from one scene/station are repeated within-sequence observations, not 10 independent scenes."
        ),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        **FLAGS,
    }


def verify_experiment_state(
    repository: Path,
    previous_before: Mapping[str, str],
    previous_after: Mapping[str, str],
    coarse: Sequence[Mapping[str, Any]],
    boundary: Sequence[Mapping[str, Any]],
    captures: Sequence[Mapping[str, Any]],
    overlap_calculator: OverlapCalculator,
    sources: Mapping[str, np.ndarray],
) -> dict[str, bool]:
    base_rows = [
        row
        for row in coarse
        if float(row["perturbation_magnitude_m"]) in COARSE_MAGNITUDES_M
    ]
    reused = [row for row in coarse if row["new_icp_executed"] is False]
    checks = {
        "previous_pilot_unchanged": dict(previous_before) == dict(previous_after),
        "base_coarse_grid_exactly_1120": len(base_rows) == EXPECTED_BASE_COARSE_ROWS,
        "reused_50mm_exactly_160": len(reused) == EXPECTED_REUSED_50MM_ROWS,
        "new_base_coarse_exactly_960": sum(
            row["new_icp_executed"] is True
            and row["execution_origin"] == "NEW_BASE_COARSE_EXECUTION"
            for row in coarse
        )
        == EXPECTED_NEW_BASE_COARSE_EXECUTIONS,
        "all_run_ids_unique": len({row["run_id"] for row in [*coarse, *boundary]})
        == len(coarse) + len(boundary),
        "all_magnitudes_within_cap": all(
            float(row["perturbation_magnitude_m"])
            <= MAX_TRANSLATION_PERTURBATION_M
            for row in [*coarse, *boundary]
        ),
        "capture_record_count_160": len(captures) == 160,
        "all_bounded_resolutions_within_25mm": all(
            row["capture_radius_resolution_m"] is None
            or float(row["capture_radius_resolution_m"])
            <= BISECTION_RESOLUTION_M + 1e-12
            for row in captures
        ),
        "backend_contract_unchanged": sha256_file(
            repository / "frozen_assets/backend_parameter_contract.json"
        )
        == controlled.BACKEND_CONTRACT_SHA256,
        "formal_measurement_false": all(
            row["FORMAL_MEASUREMENT_RESULT"] is False
            for row in [*coarse, *boundary]
        ),
    }
    # Every base physical input has both backends and identical arrays/poses.
    paired: dict[tuple[str, str, int, float], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in base_rows:
        paired[
            (
                str(row["snapshot_id"]),
                str(row["direction_class"]),
                int(row["perturbation_sign"]),
                float(row["perturbation_magnitude_m"]),
            )
        ][str(row["backend"])] = row
    checks["base_backend_pairs_exactly_560"] = len(paired) == 560 and all(
        set(pair) == set(BACKENDS) for pair in paired.values()
    )
    checks["base_backend_inputs_identical"] = all(
        controlled.identical_backend_input_pair(pair["open3d"], pair["pcl"])
        and pair["open3d"]["initial_correspondence_count"]
        == pair["pcl"]["initial_correspondence_count"]
        and math.isclose(
            float(pair["open3d"]["initial_correspondence_fraction"]),
            float(pair["pcl"]["initial_correspondence_fraction"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for pair in paired.values()
    )
    math_pass = True
    recovery_pass = True
    overlap_pass = True
    overlap_cache: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in [*coarse, *boundary]:
        t_star = np.asarray(row["T_star"], dtype=np.float64)
        t_initial = np.asarray(row["T_initial"], dtype=np.float64)
        initial = controlled.pose_error(t_initial, t_star)
        math_pass &= math.isclose(
            float(initial["translation_error_m"]),
            float(row["perturbation_magnitude_m"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        if row["T_est"] is not None:
            final = controlled.pose_error(np.asarray(row["T_est"]), t_star)
            math_pass &= math.isclose(
                float(final["translation_error_m"]),
                float(row["final_translation_error_m"]),
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            recovery_pass &= row["pose_recovered"] is controlled.pose_recovered(
                float(final["translation_error_m"]),
                float(final["rotation_error_deg"]),
            )
        cache_key = (
            str(row["snapshot_id"]),
            str(row["direction_class"]),
            f"{int(row['perturbation_sign'])}:{float(row['perturbation_magnitude_m']):.15g}",
        )
        overlap = overlap_cache.get(cache_key)
        if overlap is None:
            overlap = overlap_calculator.calculate(
                str(row["scene_id"]),
                sources[str(row["snapshot_id"])],
                t_initial,
            )
            overlap_cache[cache_key] = overlap
        overlap_pass &= bool(
            int(overlap["initial_correspondence_count"])
            == int(row["initial_correspondence_count"])
            and math.isclose(
                float(overlap["initial_correspondence_fraction"]),
                float(row["initial_correspondence_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
    checks["all_pose_math_recomputed"] = math_pass
    checks["all_recovery_decisions_recomputed"] = recovery_pass
    checks["all_overlap_proxies_recomputed"] = overlap_pass
    checks["all_boundary_iterations_within_cap"] = all(
        int(row["boundary_iteration"]) <= MAX_BISECTION_ITERATIONS
        for row in boundary
    )
    checks["all_boundary_updates_valid"] = all(
        float(row["bracket_lower_before_m"])
        < float(row["bracket_upper_before_m"])
        and float(row["bracket_lower_after_m"])
        < float(row["bracket_upper_after_m"])
        for row in boundary
    )
    return checks


def _verification_report(
    checks: Mapping[str, bool],
    coarse: Sequence[Mapping[str, Any]],
    boundary: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> str:
    lines = [
        "MID-360 CAPTURE-BASIN PILOT VERIFICATION",
        f"schema={SCHEMA}",
        f"generated_at_utc={utc_now()}",
        f"coarse_rows={len(coarse)}",
        f"coarse_new_icp_executions={sum(row['new_icp_executed'] is True for row in coarse)}",
        f"boundary_new_icp_executions={len(boundary)}",
        f"solver_success={sum(row['solver_success'] is True for row in [*coarse, *boundary])}",
        f"finite_transform={sum(row['finite_transform'] is True for row in [*coarse, *boundary])}",
        "50mm_policy=reused authenticated previous results",
        "recovery_threshold=translation<=0.005m AND rotation<=0.2deg",
        "SPATIAL_AXIS_INTERPRETATION=UNRESOLVED",
        "FORMAL_MEASUREMENT_RESULT=false",
        "",
        "CHECKS",
        *(f"{'PASS' if value else 'FAIL'} {name}" for name, value in checks.items()),
        "",
        f"CAPTURE_BASIN_PILOT_READY={str(decision['CAPTURE_BASIN_PILOT_READY']).lower()}",
        f"CAPTURE_BASIN_PILOT_SUPPORTS_FORMAL_EXPANSION={str(decision['CAPTURE_BASIN_PILOT_SUPPORTS_FORMAL_EXPANSION']).lower()}",
        "FORMAL_MEASUREMENT_RESULT=false",
    ]
    return "\n".join(lines) + "\n"


def _readme(
    decision: Mapping[str, Any],
    coarse: Sequence[Mapping[str, Any]],
    boundary: Sequence[Mapping[str, Any]],
) -> str:
    return f"""# Mid-360 translation capture-basin Pilot

`CAPTURE_BASIN_PILOT_READY={str(decision['CAPTURE_BASIN_PILOT_READY']).lower()}`  
`CAPTURE_BASIN_PILOT_SUPPORTS_FORMAL_EXPANSION={str(decision['CAPTURE_BASIN_PILOT_SUPPORTS_FORMAL_EXPANSION']).lower()}`  
`FORMAL_MEASUREMENT_RESULT=false`

This is a Pilot descriptive sensitivity analysis. It does not provide a formal
paper measurement or population inference.

The 50 mm level was reused from the byte-authenticated previous controlled
perturbation Pilot; it was not rerun. The base coarse grid contains 1120 rows,
of which 960 are new ICP executions. Automatic extension uses only the frozen
2.4/3.2/4.8/6.4 m sequence and never exceeds 6.4 m. This run produced
{len(coarse)} coarse/extension rows and {len(boundary)} boundary executions.

Weak/strong directions use the unchanged existing 3x3 translation geometry
matrix. Eigenvector sign canonicalization is used only for audit/reporting; both
physical signs are executed. `SPATIAL_AXIS_INTERPRETATION=UNRESOLVED` because no
reliable mounting/scene-axis transform for these Mid-360 acquisitions was found.

The shared pre-ICP overlap proxy is the fraction of source points with one target
neighbor within the existing 0.50 m association distance at `T_initial`.
`LOW_INITIAL_OVERLAP=true` means fraction <= 0.01 and is explanatory only; it
does not change registration. Recovery remains translation <= 5 mm and rotation
<= 0.2 degree.

Non-monotonic profiles are marked ambiguous and are not forced into bisection.
Monotonic success/failure brackets are refined to <=25 mm or at most eight
iterations. Results without failure through 6.4 m are right-censored lower
bounds, not capture radii.

10 snapshots from one scene/station are repeated within-sequence observations,
not 10 independent scenes.
"""


def _write_sha256s(output: Path) -> None:
    lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_remaining_plots(
    output: Path,
    coarse: Sequence[Mapping[str, Any]],
    captures: Sequence[Mapping[str, Any]],
    asymmetry: Sequence[Mapping[str, Any]],
    agreement: Sequence[Mapping[str, Any]],
    stability: Mapping[str, Mapping[str, Any]],
) -> None:
    # 2 and 3: bounded descriptive distributions only.
    for filename, title, category, series in (
        (
            "02_capture_radius_rich_vs_weak.png",
            "Weak-direction capture radius: Rich vs Weak",
            ("Rich", "Weak"),
            [
                _capture_plot_values(captures, scene_class=scene, backend=backend, direction_class="weak")
                for backend in BACKENDS
                for scene in ("Rich", "Weak")
            ],
        ),
        (
            "03_capture_radius_weak_vs_strong.png",
            "Capture radius: weak vs strong direction",
            ("weak", "strong"),
            [
                _capture_plot_values(captures, scene_class=scene, backend=backend, direction_class=direction)
                for scene in ("Rich", "Weak")
                for backend in BACKENDS
                for direction in DIRECTION_CLASSES
            ],
        ),
    ):
        fig, axis = plt.subplots(figsize=(10, 5.5))
        nonempty = [values if values else [np.nan] for values in series]
        axis.boxplot(nonempty, showfliers=True)
        if len(series) == 4:
            labels = [f"{b}\n{s}" for b in BACKENDS for s in category]
        else:
            labels = [
                f"{scene}\n{backend}\n{direction}"
                for scene in ("Rich", "Weak")
                for backend in BACKENDS
                for direction in category
            ]
        axis.set_xticks(range(1, len(labels) + 1), labels)
        axis.set_ylim(bottom=0.0)
        axis.set_ylabel("Bounded capture-radius midpoint [m]")
        axis.set_title(f"{title} (Pilot, nonformal)")
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    pairs = [row for row in agreement if row["row_type"] == "PAIR" and row["both_bounded"]]
    fig, axis = plt.subplots(figsize=(6, 5.5))
    if pairs:
        x = [float(row["open3d_capture_radius_mid_m"]) for row in pairs]
        y = [float(row["pcl_capture_radius_mid_m"]) for row in pairs]
        maximum = max([*x, *y]) * 1.05
        axis.scatter(x, y, alpha=0.7)
        axis.plot([0, maximum], [0, maximum], "k--", linewidth=1)
        axis.set_xlim(0, maximum)
        axis.set_ylim(0, maximum)
    axis.set_xlabel("Open3D capture-radius midpoint [m]")
    axis.set_ylabel("PCL capture-radius midpoint [m]")
    axis.set_title("Backend capture-radius agreement (Pilot, nonformal)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "04_capture_radius_open3d_vs_pcl.png", dpi=180)
    plt.close(fig)

    comparable = [row for row in asymmetry if row["comparable_bounded_pair"]]
    fig, axis = plt.subplots(figsize=(6, 5.5))
    if comparable:
        x = [float(row["capture_radius_plus_m"]) for row in comparable]
        y = [float(row["capture_radius_minus_m"]) for row in comparable]
        maximum = max([*x, *y]) * 1.05
        axis.scatter(x, y, alpha=0.7)
        axis.plot([0, maximum], [0, maximum], "k--", linewidth=1)
        axis.set_xlim(0, maximum)
        axis.set_ylim(0, maximum)
    axis.set_xlabel("+ direction capture radius [m]")
    axis.set_ylabel("- direction capture radius [m]")
    axis.set_title("Sign asymmetry (Pilot, nonformal)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "05_sign_asymmetry.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, scene in zip(axes, ("Rich", "Weak")):
        for direction, color in (("weak", "#D55E00"), ("strong", "#0072B2")):
            xs, ys = [], []
            for magnitude in (*COARSE_MAGNITUDES_M, *AUTO_EXTENSION_MAGNITUDES_M):
                selected = [
                    float(row["initial_correspondence_fraction"])
                    for row in coarse
                    if row["scene_class"] == scene
                    and row["direction_class"] == direction
                    and row["backend"] == "open3d"
                    and math.isclose(float(row["perturbation_magnitude_m"]), magnitude)
                ]
                if selected:
                    xs.append(magnitude)
                    ys.append(float(np.median(selected)))
            axis.plot(xs, ys, marker="o", color=color, label=direction)
        axis.axhline(LOW_INITIAL_OVERLAP_FRACTION_THRESHOLD, color="black", linestyle="--")
        axis.set_title(scene)
        axis.set_xlabel("Initial translation perturbation [m]")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Median initial correspondence fraction")
    fig.suptitle("Shared pre-ICP overlap proxy (Pilot, nonformal)")
    fig.tight_layout()
    fig.savefig(output / "06_initial_overlap_vs_perturbation.png", dpi=180)
    plt.close(fig)

    labels = list(sorted(stability))
    medians = [float(stability[key]["absolute_cosine_median"]) for key in labels]
    minimums = [float(stability[key]["absolute_cosine_min"]) for key in labels]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(labels, medians, color="#009E73")
    axis.scatter(range(len(labels)), minimums, color="#D55E00", label="pairwise minimum")
    axis.axhline(DIRECTION_STABILITY_THRESHOLD, color="black", linestyle="--", label="audit threshold")
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Pairwise absolute cosine similarity")
    axis.set_title("Within-scene direction stability (Pilot audit)")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "07_direction_stability.png", dpi=180)
    plt.close(fig)


_RUN_BOOLEAN_FIELDS = {
    "solver_success",
    "finite_transform",
    "pose_recovered",
    "new_icp_executed",
    "LOW_INITIAL_OVERLAP",
    "converged_but_not_recovered",
    "PILOT_ONLY",
    "PILOT_NONFORMAL_DO_NOT_CITE",
    "FORMAL_MEASUREMENT_RESULT",
    "MEASUREMENT_EVIDENCE",
}
_CAPTURE_BOOLEAN_FIELDS = {
    "LEFT_CENSORED",
    "RIGHT_CENSORED",
    "BOUND_NOT_FOUND",
    "NON_MONOTONIC_RECOVERY_PATTERN",
    "CAPTURE_RADIUS_AMBIGUOUS",
    "upper_bound_low_initial_overlap",
    "PILOT_ONLY",
    "PILOT_NONFORMAL_DO_NOT_CITE",
    "FORMAL_MEASUREMENT_RESULT",
    "MEASUREMENT_EVIDENCE",
}


def _typed_csv_rows(
    path: Path,
    *,
    boolean_fields: set[str],
    json_fields: set[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in read_csv(path):
        row: dict[str, Any] = {}
        for field, value in raw.items():
            if value == "":
                row[field] = None
            elif field in boolean_fields:
                row[field] = value.lower() == "true"
            elif field in json_fields:
                row[field] = json.loads(value)
            else:
                row[field] = value
        output.append(row)
    return output


def run_capture_basin(
    repository: Path,
    output: Path,
    *,
    runtime: Path = controlled.ZERO_RUNTIME,
) -> dict[str, Any]:
    repo = repository.expanduser().resolve(strict=True)
    # Validate before creating even a pre-execution artifact.  The backend
    # adapter records runtime errors as failed trials by design, so without
    # this fail-fast guard a globally wrong Open3D build could masquerade as
    # a zero-width capture basin.
    validate_open3d_version()
    destination = ensure_fresh_output(output)
    previous_before = authenticate_previous_pilot(repo)
    zero_inventory = controlled.authenticate_zero_runtime(runtime)
    controlled.authenticate_bags(controlled.BAG_ROOT)
    _, contract = controlled.authenticate_contract(repo)
    pcl_executable = repo / "bin/pcl_point_to_plane_cli"
    controlled.verify_sha256_file(
        pcl_executable, controlled.PCL_EXECUTABLE_SHA256
    )
    manifest_rows, sources, targets = controlled.load_and_authenticate_inputs(runtime)
    directions_raw = controlled._geometry_directions(
        manifest_rows, sources, targets, runtime
    )
    direction_audit, stability, direction_audit_pass = build_direction_audit(
        directions_raw
    )
    directions = {
        (str(row["snapshot_id"]), str(row["direction_class"])): row
        for row in directions_raw
    }
    source_rows = {row["snapshot_id"]: row for row in manifest_rows}
    destination.mkdir(parents=True)
    write_csv(destination / "direction_audit.csv", direction_audit)
    (destination / "direction_audit.txt").write_text(
        direction_audit_text(stability, direction_audit_pass), encoding="utf-8"
    )
    manifest_payload = _manifest(
        repo,
        destination,
        previous_before,
        zero_inventory,
        stability,
        direction_audit_pass,
        contract,
    )
    write_json(destination / "manifest.json", manifest_payload)
    print(
        f"PREEXECUTION_FROZEN direction_audit={direction_audit_pass} "
        f"base_new={EXPECTED_NEW_BASE_COARSE_EXECUTIONS} reused50={EXPECTED_REUSED_50MM_ROWS}",
        flush=True,
    )

    overlap_calculator = OverlapCalculator(targets)
    previous_runs = _typed_previous_runs(repo / PREVIOUS_RESULT_RELATIVE / "runs.csv")
    reused = _reuse_previous_50mm(
        previous_runs, source_rows, directions, sources, overlap_calculator
    )
    coarse = _run_base_coarse(
        manifest_rows,
        source_rows,
        directions,
        sources,
        targets,
        contract,
        pcl_executable,
        overlap_calculator,
        reused,
        destination,
    )
    extension_count = _run_extensions(
        coarse,
        manifest_rows,
        source_rows,
        directions,
        sources,
        targets,
        contract,
        pcl_executable,
        overlap_calculator,
        destination,
    )
    write_csv(destination / "coarse_runs.csv", coarse)
    boundary = _run_boundaries(
        coarse,
        source_rows,
        directions,
        sources,
        targets,
        contract,
        pcl_executable,
        overlap_calculator,
        destination,
    )
    boundary_fields = [
        *list(coarse[0]),
        "bracket_lower_before_m",
        "bracket_upper_before_m",
        "bracket_lower_after_m",
        "bracket_upper_after_m",
    ]
    write_csv(destination / "boundary_runs.csv", boundary, boundary_fields)
    all_runs = [*coarse, *boundary]
    write_csv(destination / "all_runs.csv", all_runs, _union_fields(all_runs))

    captures = build_capture_radius_records(coarse, boundary)
    summary = build_capture_radius_summary(captures, coarse)
    rich_vs_weak = build_rich_vs_weak_capture(captures)
    weak_vs_strong = build_weak_vs_strong_capture(captures)
    asymmetry = build_sign_asymmetry(captures)
    agreement = build_backend_agreement_capture(captures)
    write_csv(destination / "capture_radius_per_snapshot.csv", captures)
    write_csv(destination / "capture_radius_summary.csv", summary)
    write_csv(destination / "rich_vs_weak_capture.csv", rich_vs_weak)
    write_csv(destination / "weak_vs_strong_capture.csv", weak_vs_strong)
    write_csv(destination / "sign_asymmetry.csv", asymmetry)
    write_csv(destination / "backend_agreement.csv", agreement)
    make_plots(destination, coarse, captures, asymmetry, agreement, stability)

    previous_after = authenticate_previous_pilot(repo)
    checks = verify_experiment_state(
        repo,
        previous_before,
        previous_after,
        coarse,
        boundary,
        captures,
        overlap_calculator,
        sources,
    )
    checks.update(
        {
            "direction_audit_has_20_rows": len(direction_audit) == 20,
            "direction_audit_pass": direction_audit_pass,
            "summary_has_16_rows": len(summary) == 16,
            "sign_asymmetry_has_80_rows": len(asymmetry) == 80,
            "backend_agreement_has_81_rows": len(agreement) == 81,
        }
    )
    decision = build_decision_gate(
        captures,
        rich_vs_weak,
        weak_vs_strong,
        direction_audit_pass=direction_audit_pass,
        previous_unchanged=dict(previous_before) == dict(previous_after),
        verification_pass=all(checks.values()),
    )
    write_json(destination / "decision_gate.json", decision)
    (destination / "verification_report.txt").write_text(
        _verification_report(checks, coarse, boundary, decision), encoding="utf-8"
    )
    (destination / "README.md").write_text(
        _readme(decision, coarse, boundary), encoding="utf-8"
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (destination / name).is_file()]
    if missing:
        raise PilotBagError(f"required outputs missing: {missing}")
    _write_sha256s(destination)
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "output": str(destination),
        "coarse_rows": len(coarse),
        "new_coarse_icp_executions": sum(
            row["new_icp_executed"] is True for row in coarse
        ),
        "extension_icp_executions": extension_count,
        "boundary_icp_executions": len(boundary),
        "solver_success_count": sum(
            row["solver_success"] is True for row in all_runs
        ),
        "finite_transform_count": sum(
            row["finite_transform"] is True for row in all_runs
        ),
        "checks": checks,
        "decision": decision,
    }


def verify_capture_output(
    repository: Path,
    output: Path,
    *,
    runtime: Path = controlled.ZERO_RUNTIME,
) -> dict[str, Any]:
    repo = repository.expanduser().resolve(strict=True)
    destination = output.expanduser().resolve(strict=True)
    manifest = controlled.read_json(destination / "manifest.json")
    previous_before = manifest["previous_pilot_inventory_sha256"]
    previous_after = authenticate_previous_pilot(repo)
    controlled.authenticate_zero_runtime(runtime)
    _, sources, targets = controlled.load_and_authenticate_inputs(runtime)
    coarse = _typed_csv_rows(
        destination / "coarse_runs.csv",
        boolean_fields=_RUN_BOOLEAN_FIELDS,
        json_fields={"T_star", "T_initial", "T_est", "Delta_T"},
    )
    boundary = _typed_csv_rows(
        destination / "boundary_runs.csv",
        boolean_fields=_RUN_BOOLEAN_FIELDS,
        json_fields={"T_star", "T_initial", "T_est", "Delta_T"},
    )
    captures = _typed_csv_rows(
        destination / "capture_radius_per_snapshot.csv",
        boolean_fields=_CAPTURE_BOOLEAN_FIELDS,
        json_fields={"coarse_profile_json"},
    )
    checks = verify_experiment_state(
        repo,
        previous_before,
        previous_after,
        coarse,
        boundary,
        captures,
        OverlapCalculator(targets),
        sources,
    )
    checksum_rows: dict[str, str] = {}
    for line in (destination / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        checksum_rows[relative.lstrip(" *")] = digest
    checks["output_sha256s_all_match"] = all(
        sha256_file(destination / relative) == digest
        for relative, digest in checksum_rows.items()
    )
    checks["required_outputs_present"] = all(
        (destination / name).is_file() for name in REQUIRED_OUTPUTS
    )
    decision = controlled.read_json(destination / "decision_gate.json")
    checks["decision_formal_false"] = decision["FORMAL_MEASUREMENT_RESULT"] is False
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "coarse_rows": len(coarse),
        "boundary_rows": len(boundary),
        "decision": decision,
    }


__all__ = [
    "AUTO_EXTENSION_MAGNITUDES_M",
    "BISECTION_RESOLUTION_M",
    "COARSE_MAGNITUDES_M",
    "DIRECTION_STABILITY_THRESHOLD",
    "LOW_INITIAL_OVERLAP_FRACTION_THRESHOLD",
    "MAX_BISECTION_ITERATIONS",
    "MAX_TRANSLATION_PERTURBATION_M",
    "OverlapCalculator",
    "authenticate_previous_pilot",
    "bisection_step",
    "build_capture_run_id",
    "build_direction_audit",
    "canonicalize_direction_sign",
    "classify_recovery_profile",
    "detect_non_monotonic_recovery",
    "directory_sha256_inventory",
    "ensure_fresh_output",
    "run_capture_basin",
    "verify_capture_output",
]
