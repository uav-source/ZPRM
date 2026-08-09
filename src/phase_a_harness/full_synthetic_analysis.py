"""Primary analysis for Full Synthetic Development.

The public calculation entry point accepts normalized trial and common-
association records.  The filesystem entry point separately validates and
normalizes the frozen Phase A IDEAL import and the 2,100 new raw JSON records.
Keeping those layers separate makes the statistical contract testable without
executing a registration backend and lets the independent verifier implement a
second raw reader.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata, spearmanr

from .contracts import (
    canonical_json_sha256,
    file_sha256,
    load_manifest,
    manifest_root,
    write_json,
)
from .local_metric_models import (
    compare_local_metric_models,
    find_automatic_nonequivalence_candidates,
    local_metric_incremental_value_gate,
)
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .rotation_metrics import project_to_so3, rotation_error_atan2


SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
CONDITIONS = (
    "IDEAL_MATCHED",
    "INDEPENDENT_NOISE_FREE",
    "SCAN_NOISE_ONLY",
    "MAP_NOISE_ONLY",
    "DROPOUT_ONLY",
    "FULL_NOISE",
)
NONIDEAL_CONDITIONS = CONDITIONS[1:]
MAIN_CONDITIONS = ("INDEPENDENT_NOISE_FREE", "FULL_NOISE")
GEOMETRY_SEEDS = (1850310744, 1957656152, 1334931069)
MEASUREMENT_SEEDS = (217775206, 1664898153)
REPEAT_INDICES = (0, 1, 2, 3, 4)
BACKENDS = (OPEN3D_BACKEND, PCL_BACKEND)
BACKEND_LABELS = {OPEN3D_BACKEND: "Open3D", PCL_BACKEND: "PCL"}
PLAN_BACKEND_TO_RESULT = {
    OPEN3D_BACKEND: OPEN3D_BACKEND,
    PCL_BACKEND: PCL_BACKEND,
    "pcl_iterative_closest_point_with_normals": PCL_BACKEND,
}
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 1191248828
COMMON_INVALID_REASONS = frozenset(
    {
        "NO_INITIAL_CORRESPONDENCE",
        "NO_FINAL_CORRESPONDENCE",
        "INSUFFICIENT_VALID_NORMALS",
        "NONFINITE_COMMON_METRICS",
        "OTHER",
    }
)

def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    raise KeyError(f"required field missing: {names}")


def _optional_field(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _project_so3(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("rotation must be finite 3x3")
    u, _, vh = np.linalg.svd(value)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = 1.0 if np.linalg.det(u @ vh) >= 0.0 else -1.0
    return u @ correction @ vh


def transform_error_vectors(
    reference: Sequence[Sequence[float]], estimated: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """Return frozen translation and reflection-safe SO(3) error vectors."""

    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimated, dtype=np.float64)
    if ref.shape != (4, 4) or est.shape != (4, 4) or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(est)):
        raise ValueError("reference and estimate must be finite 4x4")
    translation = est[:3, 3] - ref[:3, 3]
    projected_reference, _ = project_to_so3(ref[:3, :3])
    projected_estimate, _ = project_to_so3(est[:3, :3])
    formal_angle = rotation_error_atan2(
        projected_reference, projected_estimate
    )["rotation_error_rad"]
    relative = projected_reference.T @ projected_estimate
    relative = _project_so3(relative)
    rotation = Rotation.from_matrix(relative).as_rotvec()
    raw_norm = float(np.linalg.norm(rotation))
    if raw_norm > 1.0e-15:
        rotation = rotation * (formal_angle / raw_norm)
    else:
        rotation = np.zeros(3, dtype=np.float64)
    return {
        "rotation_error_rad": float(formal_angle),
        "rotation_vector": [float(value) for value in rotation],
        "translation_error_m": float(np.linalg.norm(translation)),
        "translation_vector": [float(value) for value in translation],
    }


def _describe(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "iqr": None,
            "maximum": None,
            "median": None,
            "minimum": None,
            "q25": None,
            "q75": None,
            "q95_linear": None,
        }
    q25, q75 = np.quantile(array, [0.25, 0.75], method="linear")
    return {
        "count": int(array.size),
        "iqr": float(q75 - q25),
        "maximum": float(np.max(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "q25": float(q25),
        "q75": float(q75),
        "q95_linear": float(np.quantile(array, 0.95, method="linear")),
    }


def _hierarchical_interval(
    observations: Sequence[Mapping[str, Any]],
    statistic,
    *,
    repetitions: int,
) -> tuple[float | None, float | None]:
    """Three-geometry outer bootstrap with an inner observation bootstrap."""

    by_geometry: dict[int, dict[tuple[int, int], list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in observations:
        by_geometry[int(row["geometry_seed"])][
            (int(row["measurement_seed"]), int(row["repeat_index"]))
        ].append(row)
    if set(by_geometry) != set(GEOMETRY_SEEDS) or repetitions <= 0:
        return None, None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: list[float] = []
    seeds = np.asarray(GEOMETRY_SEEDS, dtype=np.int64)
    for _ in range(repetitions):
        chosen_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sample: list[Mapping[str, Any]] = []
        for seed in chosen_seeds:
            block_map = by_geometry[int(seed)]
            block_keys = sorted(block_map)
            indices = rng.integers(0, len(block_keys), size=len(block_keys))
            for index in indices:
                sample.extend(block_map[block_keys[int(index)]])
        result = _finite(statistic(sample))
        if result is not None:
            values.append(result)
    if not values:
        return None, None
    low, high = np.quantile(np.asarray(values), [0.025, 0.975], method="linear")
    return float(low), float(high)


def systematic_offset_rows(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute 252 geometry-level systematic/repeatability summaries."""

    grouped: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[
            (
                str(row["scene_variant"]),
                int(row["geometry_seed"]),
                str(row["condition"]),
                str(row["backend_schema_name"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for scene in SCENES:
        for geometry_seed in GEOMETRY_SEEDS:
            for condition in CONDITIONS:
                for backend in BACKENDS:
                    selected = grouped.get((scene, geometry_seed, condition, backend), [])
                    successful = [
                        row
                        for row in selected
                        if not bool(row["solver_failure"])
                        and bool(row["finite_output"])
                        and row.get("translation_vector") is not None
                        and row.get("rotation_vector") is not None
                    ]
                    translations = np.asarray(
                        [row["translation_vector"] for row in successful], dtype=np.float64
                    )
                    rotations = np.asarray(
                        [row["rotation_vector"] for row in successful], dtype=np.float64
                    )
                    complete = len(translations) == 10 and len(rotations) == 10
                    mean_t = np.mean(translations, axis=0) if complete else None
                    mean_r = np.mean(rotations, axis=0) if complete else None
                    covariance_t = (
                        np.cov(translations, rowvar=False, ddof=1)
                        if complete
                        else None
                    )
                    covariance_r = (
                        np.cov(rotations, rowvar=False, ddof=1)
                        if complete
                        else None
                    )
                    offset_t = float(np.linalg.norm(mean_t)) if mean_t is not None else None
                    offset_r = float(np.linalg.norm(mean_r)) if mean_r is not None else None
                    repeat_t = (
                        float(np.sqrt(max(0.0, np.trace(covariance_t))))
                        if covariance_t is not None
                        else None
                    )
                    repeat_r = (
                        float(np.sqrt(max(0.0, np.trace(covariance_r))))
                        if covariance_r is not None
                        else None
                    )
                    norms = np.linalg.norm(translations, axis=1) if complete else np.asarray([])
                    denominator = float(np.mean(norms)) if norms.size else None
                    fraction = (
                        float(offset_t / denominator)
                        if offset_t is not None and denominator is not None and denominator > 1.0e-12
                        else None
                    )
                    nonzero = translations[norms > 1.0e-12] if norms.size else np.empty((0, 3))
                    concentration = (
                        float(np.linalg.norm(np.sum(nonzero / np.linalg.norm(nonzero, axis=1)[:, None], axis=0)) / len(nonzero))
                        if len(nonzero)
                        else None
                    )
                    output.append(
                        {
                            "backend": BACKEND_LABELS[backend],
                            "backend_schema_name": backend,
                            "condition": condition,
                            "direction_valid_nonzero_count": int(len(nonzero)),
                            "geometry_seed": geometry_seed,
                            "mean_rotation_vector": None if mean_r is None else [float(value) for value in mean_r],
                            "mean_translation_vector": None if mean_t is None else [float(value) for value in mean_t],
                            "planned_observation_count": 10,
                            "rotation_repeatability_covariance": None if covariance_r is None else covariance_r.tolist(),
                            "rotation_repeatability_rms_rad": repeat_r,
                            "scene_variant": scene,
                            "successful_observation_count": len(successful),
                            "systematic_fraction_translation": fraction,
                            "systematic_rotation_offset_rad": offset_r,
                            "systematic_translation_offset_m": offset_t,
                            "translation_direction_concentration": concentration,
                            "translation_repeatability_covariance": None if covariance_t is None else covariance_t.tolist(),
                            "translation_repeatability_rms_m": repeat_t,
                        }
                    )
    return output


def scene_condition_backend_rows(
    trials: Sequence[Mapping[str, Any]], *, bootstrap_repetitions: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[(str(row["scene_variant"]), str(row["condition"]), str(row["backend_schema_name"]))].append(row)
    output: list[dict[str, Any]] = []
    for scene in SCENES:
        for condition in CONDITIONS:
            for backend in BACKENDS:
                chosen = grouped.get((scene, condition, backend), [])
                successful = [
                    row for row in chosen
                    if not bool(row["solver_failure"])
                    and bool(row["finite_output"])
                    and _finite(row.get("translation_error_m")) is not None
                    and _finite(row.get("rotation_error_rad")) is not None
                ]
                translation = _describe(float(row["translation_error_m"]) for row in successful)
                rotation = _describe(float(row["rotation_error_rad"]) for row in successful)
                low, high = _hierarchical_interval(
                    successful,
                    lambda sample: np.median([float(row["translation_error_m"]) for row in sample]),
                    repetitions=bootstrap_repetitions,
                )
                output.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "exploratory_translation_median_ci95_high_m": high,
                        "exploratory_translation_median_ci95_low_m": low,
                        "planned_trial_count": 30,
                        "rotation_iqr_rad": rotation["iqr"],
                        "rotation_median_rad": rotation["median"],
                        "rotation_q25_rad": rotation["q25"],
                        "rotation_q75_rad": rotation["q75"],
                        "rotation_q95_rad": rotation["q95_linear"],
                        "scene_variant": scene,
                        "solver_failure_count": sum(bool(row["solver_failure"]) for row in chosen),
                        "success_rate": len(successful) / 30.0,
                        "successful_trial_count": len(successful),
                        "translation_iqr_m": translation["iqr"],
                        "translation_median_m": translation["median"],
                        "translation_q25_m": translation["q25"],
                        "translation_q75_m": translation["q75"],
                        "translation_q95_m": translation["q95_linear"],
                    }
                )
    return output


def _block_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["geometry_seed"]),
        int(row["measurement_seed"]),
        int(row["repeat_index"]),
    )


def primary_scene_effect_rows(
    trials: Sequence[Mapping[str, Any]], *, bootstrap_repetitions: int
) -> tuple[list[dict[str, Any]], bool]:
    lookup: dict[tuple[str, str, str], dict[tuple[int, int, int], Mapping[str, Any]]] = defaultdict(dict)
    for row in trials:
        if str(row["condition"]) not in MAIN_CONDITIONS:
            continue
        if str(row["scene_variant"]) not in {"LONG_CORRIDOR", "GEOMETRY_RICH_ROOM"}:
            continue
        if bool(row["solver_failure"]) or not bool(row["finite_output"]):
            continue
        lookup[(str(row["condition"]), str(row["backend_schema_name"]), str(row["scene_variant"]))][
            _block_key(row)
        ] = row
    output: list[dict[str, Any]] = []
    passes: list[bool] = []
    for condition in MAIN_CONDITIONS:
        for backend in BACKENDS:
            rich = lookup[(condition, backend, "GEOMETRY_RICH_ROOM")]
            corridor = lookup[(condition, backend, "LONG_CORRIDOR")]
            common = sorted(set(rich) & set(corridor))
            pairs = [
                {
                    "geometry_seed": key[0],
                    "measurement_seed": key[1],
                    "repeat_index": key[2],
                    "rich": float(rich[key]["translation_error_m"]),
                    "weak": float(corridor[key]["translation_error_m"]),
                }
                for key in common
            ]
            rich_values = np.asarray([row["rich"] for row in pairs], dtype=float)
            weak_values = np.asarray([row["weak"] for row in pairs], dtype=float)
            rich_median = float(np.median(rich_values)) if rich_values.size else None
            weak_median = float(np.median(weak_values)) if weak_values.size else None
            difference = None if rich_median is None or weak_median is None else weak_median - rich_median
            ratio = (
                weak_median / rich_median
                if rich_median is not None and weak_median is not None and rich_median > 1.0e-12
                else None
            )
            wins = sum(row["weak"] > row["rich"] for row in pairs)
            win_rate = wins / 30.0
            diff_low, diff_high = _hierarchical_interval(
                pairs,
                lambda sample: np.median([row["weak"] - row["rich"] for row in sample]),
                repetitions=bootstrap_repetitions,
            )
            ratio_low, ratio_high = _hierarchical_interval(
                pairs,
                lambda sample: (
                    np.median([row["weak"] for row in sample])
                    / np.median([row["rich"] for row in sample])
                    if np.median([row["rich"] for row in sample]) > 1.0e-12
                    else float("nan")
                ),
                repetitions=bootstrap_repetitions,
            )
            rich_success = len(rich) / 30.0
            corridor_success = len(corridor) / 30.0
            passed = bool(
                ratio is not None and ratio >= 5.0
                and difference is not None and difference >= 0.005
                and win_rate >= 0.80
                and rich_success >= 0.90
                and corridor_success >= 0.90
            )
            passes.append(passed)
            output.append(
                {
                    "absolute_median_difference_m": difference,
                    "backend": BACKEND_LABELS[backend],
                    "backend_schema_name": backend,
                    "condition": condition,
                    "corridor_median_m": weak_median,
                    "corridor_success_rate": corridor_success,
                    "exploratory_difference_ci95_high_m": diff_high,
                    "exploratory_difference_ci95_low_m": diff_low,
                    "exploratory_ratio_ci95_high": ratio_high,
                    "exploratory_ratio_ci95_low": ratio_low,
                    "gate_pass": passed,
                    "matched_block_count": len(pairs),
                    "paired_win_count": wins,
                    "paired_win_rate": win_rate,
                    "rich_median_m": rich_median,
                    "rich_success_rate": rich_success,
                    "weak_rich_median_ratio": ratio,
                }
            )
    return output, all(passes) and len(passes) == 4


def condition_contrast_rows(
    trials: Sequence[Mapping[str, Any]], *, bootstrap_repetitions: int
) -> list[dict[str, Any]]:
    successful: dict[tuple[str, str, str, tuple[int, int, int]], Mapping[str, Any]] = {}
    for row in trials:
        if str(row["condition"]) not in NONIDEAL_CONDITIONS:
            continue
        if bool(row["solver_failure"]) or not bool(row["finite_output"]):
            continue
        successful[(
            str(row["backend_schema_name"]), str(row["scene_variant"]),
            str(row["condition"]), _block_key(row),
        )] = row
    output: list[dict[str, Any]] = []
    baseline = "INDEPENDENT_NOISE_FREE"
    for backend in BACKENDS:
        for scene in SCENES:
            for condition in NONIDEAL_CONDITIONS[1:]:
                pairs: list[dict[str, Any]] = []
                for block in (
                    (geometry, measurement, repeat)
                    for geometry in GEOMETRY_SEEDS
                    for measurement in MEASUREMENT_SEEDS
                    for repeat in REPEAT_INDICES
                ):
                    base = successful.get((backend, scene, baseline, block))
                    contrast = successful.get((backend, scene, condition, block))
                    if base is not None and contrast is not None:
                        pairs.append(
                            {
                                "baseline": float(base["translation_error_m"]),
                                "contrast": float(contrast["translation_error_m"]),
                                "geometry_seed": block[0],
                                "measurement_seed": block[1],
                                "repeat_index": block[2],
                            }
                        )
                base_values = np.asarray([row["baseline"] for row in pairs])
                contrast_values = np.asarray([row["contrast"] for row in pairs])
                base_median = float(np.median(base_values)) if len(base_values) else None
                contrast_median = float(np.median(contrast_values)) if len(contrast_values) else None
                difference = (
                    float(
                        np.median(
                            [row["contrast"] - row["baseline"] for row in pairs]
                        )
                    )
                    if pairs
                    else None
                )
                ratio = (
                    contrast_median / base_median
                    if base_median is not None and contrast_median is not None and base_median > 1.0e-12
                    else None
                )
                low, high = _hierarchical_interval(
                    pairs,
                    lambda sample: np.median([row["contrast"] - row["baseline"] for row in sample]),
                    repetitions=bootstrap_repetitions,
                )
                output.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "baseline_condition": baseline,
                        "baseline_median_m": base_median,
                        "contrast_condition": condition,
                        "contrast_median_m": contrast_median,
                        "exploratory_paired_difference_ci95_high_m": high,
                        "exploratory_paired_difference_ci95_low_m": low,
                        "matched_block_count": len(pairs),
                        "median_ratio": ratio,
                        "paired_median_difference_m": difference,
                        "paired_win_rate": (
                            sum(row["contrast"] > row["baseline"] for row in pairs) / len(pairs)
                            if pairs else None
                        ),
                        "scene_variant": scene,
                    }
                )
    return output


def _spearman(left: Sequence[float], right: Sequence[float]) -> tuple[float | None, float | None]:
    if len(left) < 2 or len(left) != len(right):
        return None, None
    result = spearmanr(left, right)
    rho = float(result.statistic) if math.isfinite(float(result.statistic)) else None
    pvalue = float(result.pvalue) if math.isfinite(float(result.pvalue)) else None
    return rho, pvalue


def cross_backend_and_rank_rows(
    trials: Sequence[Mapping[str, Any]], *, bootstrap_repetitions: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, bool]:
    cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    paired: dict[tuple[str, str, tuple[int, int, int]], dict[str, float]] = defaultdict(dict)
    for row in trials:
        condition = str(row["condition"])
        if condition not in NONIDEAL_CONDITIONS:
            continue
        if bool(row["solver_failure"]) or not bool(row["finite_output"]):
            continue
        backend = str(row["backend_schema_name"])
        scene = str(row["scene_variant"])
        error = _finite(row.get("translation_error_m"))
        if error is None:
            continue
        cells[(condition, backend, scene)].append(error)
        paired[(condition, scene, _block_key(row))][backend] = error
    complete_blocks: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for condition in NONIDEAL_CONDITIONS:
        for geometry in GEOMETRY_SEEDS:
            complete_blocks[(condition, geometry)] = [
                (measurement, repeat)
                for measurement in MEASUREMENT_SEEDS
                for repeat in REPEAT_INDICES
                if all(
                    set(paired.get((condition, scene, (geometry, measurement, repeat)), {}))
                    == set(BACKENDS)
                    for scene in SCENES
                )
            ]

    scene_rank_rows: list[dict[str, Any]] = []
    rich_passes = 0
    weak_passes = 0
    for condition in NONIDEAL_CONDITIONS:
        for backend in BACKENDS:
            medians = [
                float(np.median(cells[(condition, backend, scene)]))
                if cells[(condition, backend, scene)] else None
                for scene in SCENES
            ]
            valid = all(value is not None for value in medians)
            ranks = rankdata(np.asarray(medians, dtype=float), method="average") if valid else np.full(7, np.nan)
            counts = Counter(medians) if valid else Counter()
            rank_lookup = dict(zip(SCENES, ranks))
            rich_pass = bool(valid and float(rank_lookup["GEOMETRY_RICH_ROOM"]) <= 2.0)
            weak_pass = bool(
                valid
                and max(
                    float(rank_lookup["LONG_CORRIDOR"]),
                    float(rank_lookup["END_FACE_TRANSITION_ABSENT"]),
                )
                >= 5.0
            )
            rich_passes += rich_pass
            weak_passes += weak_pass
            for scene, median, rank in zip(SCENES, medians, ranks):
                scene_rank_rows.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "is_tied": bool(valid and counts[median] > 1),
                        "rich_lowest_two_group_pass": rich_pass,
                        "scene_variant": scene,
                        "translation_median_m": median,
                        "translation_rank_ascending_average_ties": (
                            float(rank) if math.isfinite(float(rank)) else None
                        ),
                        "weak_highest_three_group_pass": weak_pass,
                    }
                )
    rank_stability_pass = rich_passes >= 8 and weak_passes >= 8

    cross_rows: list[dict[str, Any]] = []
    condition_rhos: dict[str, float | None] = {}
    for condition in NONIDEAL_CONDITIONS:
        open_values = [
            float(np.median(cells[(condition, OPEN3D_BACKEND, scene)]))
            if cells[(condition, OPEN3D_BACKEND, scene)] else float("nan")
            for scene in SCENES
        ]
        pcl_values = [
            float(np.median(cells[(condition, PCL_BACKEND, scene)]))
            if cells[(condition, PCL_BACKEND, scene)] else float("nan")
            for scene in SCENES
        ]
        rho, pvalue = (
            _spearman(open_values, pcl_values)
            if all(math.isfinite(value) for value in open_values + pcl_values)
            else (None, None)
        )
        condition_rhos[condition] = rho
        bootstrap_values: list[float] = []
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for _ in range(bootstrap_repetitions):
            selected_geometry = rng.choice(np.asarray(GEOMETRY_SEEDS), size=3, replace=True)
            left_by_scene: dict[str, list[float]] = defaultdict(list)
            right_by_scene: dict[str, list[float]] = defaultdict(list)
            for geometry in selected_geometry:
                blocks = complete_blocks[(condition, int(geometry))]
                if not blocks:
                    continue
                indices = rng.integers(0, len(blocks), size=len(blocks))
                for index in indices:
                    measurement, repeat = blocks[int(index)]
                    for scene in SCENES:
                        item = paired[
                            (
                                condition,
                                scene,
                                (int(geometry), measurement, repeat),
                            )
                        ]
                        left_by_scene[scene].append(item[OPEN3D_BACKEND])
                        right_by_scene[scene].append(item[PCL_BACKEND])
            if all(left_by_scene[scene] and right_by_scene[scene] for scene in SCENES):
                candidate, _ = _spearman(
                    [float(np.median(left_by_scene[scene])) for scene in SCENES],
                    [float(np.median(right_by_scene[scene])) for scene in SCENES],
                )
                if candidate is not None:
                    bootstrap_values.append(candidate)
        if bootstrap_values:
            low, high = np.quantile(bootstrap_values, [0.025, 0.975], method="linear")
            interval = (float(low), float(high))
        else:
            interval = (None, None)
        cross_rows.append(
            {
                "condition": condition,
                "exploratory_spearman_ci95_high": interval[1],
                "exploratory_spearman_ci95_low": interval[0],
                "pvalue_descriptive": pvalue,
                "scene_count": 7,
                "scope": "CONDITION",
                "spearman_rho": rho,
            }
        )

    pooled_left = []
    pooled_right = []
    for condition in NONIDEAL_CONDITIONS:
        for scene in SCENES:
            if cells[(condition, OPEN3D_BACKEND, scene)] and cells[(condition, PCL_BACKEND, scene)]:
                pooled_left.append(float(np.median(cells[(condition, OPEN3D_BACKEND, scene)])))
                pooled_right.append(float(np.median(cells[(condition, PCL_BACKEND, scene)])))
    pooled_rho, pooled_pvalue = _spearman(pooled_left, pooled_right)
    pooled_bootstrap_values: list[float] = []
    pooled_rng = np.random.default_rng(BOOTSTRAP_SEED)
    pooled_complete_blocks = {
        geometry: [
            (measurement, repeat)
            for measurement in MEASUREMENT_SEEDS
            for repeat in REPEAT_INDICES
            if all(
                set(
                    paired.get(
                        (
                            condition,
                            scene,
                            (geometry, measurement, repeat),
                        ),
                        {},
                    )
                )
                == set(BACKENDS)
                for condition in NONIDEAL_CONDITIONS
                for scene in SCENES
            )
        ]
        for geometry in GEOMETRY_SEEDS
    }
    for _ in range(bootstrap_repetitions):
        selected_geometry = pooled_rng.choice(
            np.asarray(GEOMETRY_SEEDS), size=3, replace=True
        )
        left_by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
        right_by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
        for geometry_value in selected_geometry:
            geometry = int(geometry_value)
            blocks = pooled_complete_blocks[geometry]
            if not blocks:
                continue
            selected_blocks = pooled_rng.integers(0, len(blocks), size=len(blocks))
            for selected_index in selected_blocks:
                measurement, repeat = blocks[int(selected_index)]
                for condition in NONIDEAL_CONDITIONS:
                    for scene in SCENES:
                        item = paired[
                            (condition, scene, (geometry, measurement, repeat))
                        ]
                        left_by_cell[(condition, scene)].append(item[OPEN3D_BACKEND])
                        right_by_cell[(condition, scene)].append(item[PCL_BACKEND])
        keys = [
            (condition, scene)
            for condition in NONIDEAL_CONDITIONS
            for scene in SCENES
        ]
        if all(left_by_cell[key] and right_by_cell[key] for key in keys):
            candidate, _ = _spearman(
                [float(np.median(left_by_cell[key])) for key in keys],
                [float(np.median(right_by_cell[key])) for key in keys],
            )
            if candidate is not None:
                pooled_bootstrap_values.append(candidate)
    if pooled_bootstrap_values:
        pooled_low, pooled_high = np.quantile(
            pooled_bootstrap_values, [0.025, 0.975], method="linear"
        )
        pooled_interval = (float(pooled_low), float(pooled_high))
    else:
        pooled_interval = (None, None)
    cross_rows.append(
        {
            "condition": "ALL_NONIDEAL",
            "exploratory_spearman_ci95_high": pooled_interval[1],
            "exploratory_spearman_ci95_low": pooled_interval[0],
            "pvalue_descriptive": pooled_pvalue,
            "scene_count": len(pooled_left),
            "scope": "POOLED_SCENE_CONDITION",
            "spearman_rho": pooled_rho,
        }
    )
    finite_rhos = [value for value in condition_rhos.values() if value is not None]
    cross_pass = bool(
        condition_rhos.get("INDEPENDENT_NOISE_FREE") is not None
        and condition_rhos["INDEPENDENT_NOISE_FREE"] >= 0.70
        and condition_rhos.get("FULL_NOISE") is not None
        and condition_rhos["FULL_NOISE"] >= 0.70
        and sum(value is not None and value >= 0.50 for value in condition_rhos.values()) >= 4
        and len(finite_rhos) == 5
        and float(np.median(finite_rhos)) >= 0.70
        and pooled_rho is not None
        and pooled_rho >= 0.75
    )
    return cross_rows, scene_rank_rows, cross_pass, rank_stability_pass


def systematic_claim_gate(
    systematic: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    output: list[dict[str, Any]] = []
    passes: list[bool] = []
    for condition in MAIN_CONDITIONS:
        for backend in BACKENDS:
            chosen = [
                row for row in systematic
                if row["scene_variant"] == "LONG_CORRIDOR"
                and row["condition"] == condition
                and row["backend_schema_name"] == backend
            ]
            qualified = sum(
                int(row["successful_observation_count"]) == 10
                and
                _finite(row["systematic_translation_offset_m"]) is not None
                and float(row["systematic_translation_offset_m"]) >= 0.005
                and _finite(row["systematic_fraction_translation"]) is not None
                and float(row["systematic_fraction_translation"]) >= 0.60
                for row in chosen
            )
            fractions = [
                float(row["systematic_fraction_translation"])
                for row in chosen
                if int(row["successful_observation_count"]) == 10
                and _finite(row["systematic_fraction_translation"]) is not None
            ]
            median_fraction = float(np.median(fractions)) if fractions else None
            passed = bool(
                len(chosen) == 3
                and qualified >= 2
                and len(fractions) == 3
                and median_fraction is not None
                and median_fraction >= 0.70
            )
            passes.append(passed)
            output.append(
                {
                    "backend": BACKEND_LABELS[backend],
                    "backend_schema_name": backend,
                    "condition": condition,
                    "gate_pass": passed,
                    "geometry_group_count": len(chosen),
                    "median_systematic_fraction_translation": median_fraction,
                    "qualified_geometry_group_count": qualified,
                    "scene_variant": "LONG_CORRIDOR",
                }
            )
    return output, len(passes) == 4 and all(passes)


def _common_valid(row: Mapping[str, Any]) -> bool:
    return bool(_optional_field(row, "common_association_valid", "valid", default=False))


def association_validity_and_turnover(
    trials: Sequence[Mapping[str, Any]],
    common_records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    eligible = {
        str(row["planned_trial_id"]): row
        for row in trials
        if str(row["condition"]) in NONIDEAL_CONDITIONS
        and not bool(row["solver_failure"])
        and bool(row["finite_output"])
    }
    records = {str(row["planned_trial_id"]): row for row in common_records}
    invalid_reasons = Counter()
    joined: list[dict[str, Any]] = []
    for trial_id, trial in eligible.items():
        record = records.get(trial_id)
        if record is None:
            invalid_reasons["OTHER"] += 1
            continue
        if not _common_valid(record):
            reason = str(
                _optional_field(
                    record,
                    "common_association_invalid_reason",
                    "invalid_reason",
                    "reason",
                    "failure_reason",
                    default="OTHER",
                )
            )
            invalid_reasons[reason if reason in COMMON_INVALID_REASONS else "OTHER"] += 1
            continue
        merged = dict(record)
        merged.update(
            {
                "backend": trial["backend"],
                "backend_schema_name": trial["backend_schema_name"],
                "condition": trial["condition"],
                "geometry_seed": trial["geometry_seed"],
                "measurement_seed": trial["measurement_seed"],
                "repeat_index": trial["repeat_index"],
                "scene_variant": trial["scene_variant"],
                "translation_error_m": trial["translation_error_m"],
            }
        )
        joined.append(merged)
    valid_fraction = len(joined) / len(eligible) if eligible else 0.0
    validity = {
        "COMMON_ASSOCIATION_ANALYSIS_PASS": bool(eligible and valid_fraction >= 0.95),
        "eligible_successful_nonideal_trial_count": len(eligible),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "missing_or_invalid_record_count": len(eligible) - len(joined),
        "valid_common_association_record_count": len(joined),
        "valid_fraction": valid_fraction,
    }

    correlation_rows: list[dict[str, Any]] = []
    pooled_rhos: dict[str, float | None] = {}
    centered_rhos: dict[str, float | None] = {}
    for backend in BACKENDS:
        chosen = [
            row for row in joined
            if row["backend_schema_name"] == backend
            and _finite(row.get("correspondence_turnover")) is not None
            and _finite(row.get("translation_error_m")) is not None
        ]
        turnover = [float(row["correspondence_turnover"]) for row in chosen]
        log_error = [math.log10(float(row["translation_error_m"]) + 1.0e-9) for row in chosen]
        pooled, pooled_pvalue = _spearman(turnover, log_error)
        pooled_rhos[backend] = pooled
        centered_turnover: list[float] = []
        centered_error: list[float] = []
        by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in chosen:
            by_cell[(str(row["scene_variant"]), str(row["condition"]))].append(row)
        centered_records: list[dict[str, Any]] = []
        for cell_rows in by_cell.values():
            cell_turnover = np.asarray([float(row["correspondence_turnover"]) for row in cell_rows])
            cell_error = np.asarray(
                [math.log10(float(row["translation_error_m"]) + 1.0e-9) for row in cell_rows]
            )
            turnover_median = float(np.median(cell_turnover))
            error_median = float(np.median(cell_error))
            for row, turnover_value, error_value in zip(cell_rows, cell_turnover, cell_error):
                centered_turnover.append(float(turnover_value - turnover_median))
                centered_error.append(float(error_value - error_median))
                centered_records.append(
                    {
                        "geometry_seed": int(row["geometry_seed"]),
                        "measurement_seed": int(row["measurement_seed"]),
                        "repeat_index": int(row["repeat_index"]),
                        "left": float(turnover_value - turnover_median),
                        "right": float(error_value - error_median),
                    }
                )
        centered, centered_pvalue = _spearman(centered_turnover, centered_error)
        centered_rhos[backend] = centered
        pooled_records = [
            {
                "geometry_seed": int(row["geometry_seed"]),
                "measurement_seed": int(row["measurement_seed"]),
                "repeat_index": int(row["repeat_index"]),
                "left": float(row["correspondence_turnover"]),
                "right": math.log10(float(row["translation_error_m"]) + 1.0e-9),
            }
            for row in chosen
        ]
        pooled_low, pooled_high = _hierarchical_interval(
            pooled_records,
            lambda sample: _spearman(
                [row["left"] for row in sample], [row["right"] for row in sample]
            )[0],
            repetitions=bootstrap_repetitions,
        )
        centered_low, centered_high = _hierarchical_interval(
            centered_records,
            lambda sample: _spearman(
                [row["left"] for row in sample], [row["right"] for row in sample]
            )[0],
            repetitions=bootstrap_repetitions,
        )
        correlation_rows.append(
            {
                "backend": BACKEND_LABELS[backend],
                "backend_schema_name": backend,
                "centered_exploratory_ci95_high": centered_high,
                "centered_exploratory_ci95_low": centered_low,
                "centered_pvalue_descriptive": centered_pvalue,
                "centered_spearman_rho": centered,
                "pooled_exploratory_ci95_high": pooled_high,
                "pooled_exploratory_ci95_low": pooled_low,
                "pooled_pvalue_descriptive": pooled_pvalue,
                "pooled_spearman_rho": pooled,
                "valid_trial_count": len(chosen),
            }
        )
    centered_values = [centered_rhos.get(backend) for backend in BACKENDS]
    mechanism = bool(
        all(pooled_rhos.get(backend) is not None and pooled_rhos[backend] >= 0.40 for backend in BACKENDS)
        and all(value is not None for value in centered_values)
        and max(float(value) for value in centered_values) >= 0.20
        and min(float(value) for value in centered_values) >= 0.00
    )
    return validity, correlation_rows, mechanism


def ridge_model_rows(common_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fixed Ridge(alpha=1) leave-one-geometry-seed-out comparison."""

    output: list[dict[str, Any]] = []
    for backend in BACKENDS:
        selected = [
            row for row in common_records
            if row.get("backend_schema_name") == backend and _common_valid(row)
        ]
        try:
            comparison = compare_local_metric_models(
                selected, backend=backend, backend_field="backend_schema_name"
            )
        except (KeyError, TypeError, ValueError):
            comparison = None
        output.append(
            {
                "alpha": 1.0,
                "backend": BACKEND_LABELS[backend],
                "backend_schema_name": backend,
                "fit_intercept": True,
                "fold_results": None if comparison is None else {
                    "model_a": comparison["model_a"]["folds"],
                    "model_b": comparison["model_b"]["folds"],
                },
                "model_a_cv_mae": None if comparison is None else comparison["model_a_cv_mae"],
                "model_a_feature_count": 6,
                "model_b_cv_mae": None if comparison is None else comparison["model_b_cv_mae"],
                "model_b_feature_count": 12,
                "relative_mae_improvement": None if comparison is None else comparison["relative_improvement"],
                "sample_count": len(selected) if comparison is not None else 0,
                "scaler_fit_scope": "TRAINING_FOLD_ONLY",
                "split": "LEAVE_ONE_GEOMETRY_SEED_OUT",
            }
        )
    return output


def automatic_nonequivalence_candidates(
    common_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for backend in BACKENDS:
        selected = [row for row in common_records if _common_valid(row)]
        found = find_automatic_nonequivalence_candidates(
            selected, backend=backend, backend_field="backend_schema_name"
        )
        output.extend(
            {
                **row,
                "backend": BACKEND_LABELS[backend],
                "backend_schema_name": backend,
            }
            for row in found
        )
    output.sort(key=lambda row: row["candidate_pair_id"])
    return output


def incremental_value_gate(
    models: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], bool]:
    canonical_models = [
        {
            "backend": row["backend_schema_name"],
            "relative_improvement": row["relative_mae_improvement"],
        }
        for row in models
    ]
    canonical_candidates = [
        {**row, "backend": row["backend_schema_name"]} for row in candidates
    ]
    canonical = local_metric_incremental_value_gate(
        canonical_models, canonical_candidates, backends=BACKENDS
    )
    counts = Counter(row["backend_schema_name"] for row in candidates)
    scene_pairs = {
        tuple(sorted((str(row["scene_a"]), str(row["scene_b"])))) for row in candidates
    }
    detail = {
        "automatic_candidate_count_by_backend": {
            BACKEND_LABELS[backend]: counts[backend] for backend in BACKENDS
        },
        "condition_a_model_improvement_pass": canonical["condition_a_model_improvement_pass"],
        "condition_b_candidate_pass": canonical["condition_b_candidate_coverage_pass"],
        "distinct_scene_pair_count": len(scene_pairs),
        "mean_relative_mae_improvement": (
            canonical["mean_relative_improvement"]
        ),
    }
    return detail, canonical["LOCAL_METRIC_INCREMENTAL_VALUE_PASS"]


def join_common_records(
    trials: Sequence[Mapping[str, Any]], common_records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["planned_trial_id"]): row for row in trials}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in common_records:
        trial_id = str(_field(record, "planned_trial_id"))
        if trial_id in seen or trial_id not in by_id:
            continue
        seen.add(trial_id)
        trial = by_id[trial_id]
        merged = dict(record)
        for name in (
            "backend", "backend_schema_name", "condition", "geometry_seed",
            "measurement_seed", "repeat_index", "scene_variant", "snapshot_id",
            "translation_error_m", "rotation_error_rad",
        ):
            merged[name] = trial[name]
        output.append(merged)
    output.sort(key=lambda row: str(row["planned_trial_id"]))
    return output


def _default_integrity(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    snapshot_ids = {str(row["snapshot_id"]) for row in trials}
    return {
        "PHASE_A_IDEAL_IMPORT_PASS": len([row for row in trials if row["condition"] == "IDEAL_MATCHED"]) == 420,
        "PHASE_B_SUBSET_REPRODUCTION_PASS": True,
        "backend_input_checksum_mismatch_count": 0,
        "combined_snapshot_count": len(snapshot_ids),
        "combined_trial_count": len(trials),
        "corrupt_trial_count": 0,
        "duplicate_trial_count": 0,
        "extra_trial_count": 0,
        "infrastructure_interruption_unresolved_count": 0,
        "missing_trial_count": 0,
        "new_snapshot_count": len({row["snapshot_id"] for row in trials if row["condition"] != "IDEAL_MATCHED"}),
        "new_trial_count": len([row for row in trials if row["condition"] != "IDEAL_MATCHED"]),
    }


def analyze_full_synthetic_record_collections(
    *,
    trials: Sequence[Mapping[str, Any]],
    common_records: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any] | None = None,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    """Evaluate all frozen Development statistics and gates from normalized rows."""

    rows = [dict(row) for row in trials]
    integrity_value = dict(_default_integrity(rows) if integrity is None else integrity)
    scene_summary = scene_condition_backend_rows(
        rows, bootstrap_repetitions=bootstrap_repetitions
    )
    systematic = systematic_offset_rows(rows)
    systematic_gate_detail, systematic_claim = systematic_claim_gate(systematic)
    scene_effect, primary_scene_pass = primary_scene_effect_rows(
        rows, bootstrap_repetitions=bootstrap_repetitions
    )
    condition_contrasts = condition_contrast_rows(
        rows, bootstrap_repetitions=bootstrap_repetitions
    )
    cross_ranking, scene_ranking, cross_pass, rank_pass = cross_backend_and_rank_rows(
        rows, bootstrap_repetitions=bootstrap_repetitions
    )
    joined_common = join_common_records(rows, common_records)
    common_validity, turnover_correlations, mechanism_pass = association_validity_and_turnover(
        rows, joined_common, bootstrap_repetitions=bootstrap_repetitions
    )
    model_rows = ridge_model_rows(joined_common)
    candidates = automatic_nonequivalence_candidates(joined_common)
    incremental_detail, incremental_pass = incremental_value_gate(model_rows, candidates)

    success_group_rows: list[dict[str, Any]] = []
    backend_condition_passes: list[bool] = []
    backend_condition_scene_passes: list[bool] = []
    for backend in BACKENDS:
        for condition in CONDITIONS:
            selected = [
                row for row in rows
                if row["backend_schema_name"] == backend and row["condition"] == condition
            ]
            success = sum(not bool(row["solver_failure"]) and bool(row["finite_output"]) for row in selected)
            rate = success / 210.0
            group_pass = len(selected) == 210 and rate >= 0.95
            backend_condition_passes.append(group_pass)
            success_group_rows.append(
                {
                    "backend": BACKEND_LABELS[backend],
                    "backend_schema_name": backend,
                    "condition": condition,
                    "gate_level": "BACKEND_CONDITION",
                    "gate_pass": group_pass,
                    "planned_count": 210,
                    "scene_variant": "ALL_SCENES",
                    "success_count": success,
                    "success_rate": rate,
                }
            )
            for scene in SCENES:
                cell = [row for row in selected if row["scene_variant"] == scene]
                cell_success = sum(
                    not bool(row["solver_failure"]) and bool(row["finite_output"])
                    for row in cell
                )
                cell_rate = cell_success / 30.0
                cell_pass = len(cell) == 30 and cell_rate >= 0.90
                backend_condition_scene_passes.append(cell_pass)
                success_group_rows.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "gate_level": "BACKEND_CONDITION_SCENE",
                        "gate_pass": cell_pass,
                        "planned_count": 30,
                        "scene_variant": scene,
                        "success_count": cell_success,
                        "success_rate": cell_rate,
                    }
                )
    robustness = bool(
        len(backend_condition_passes) == 12
        and all(backend_condition_passes)
        and len(backend_condition_scene_passes) == 84
        and all(backend_condition_scene_passes)
    )

    backend_counts = Counter(str(row["backend_schema_name"]) for row in rows)
    native_count = len(rows) - sum(backend_counts[backend] for backend in BACKENDS)
    backend_exceptions = sum(
        row.get("failure_classification") == "BACKEND_EXCEPTION" for row in rows
    )
    nonfinite = sum(not bool(row["finite_output"]) for row in rows)
    solver_failures = sum(bool(row["solver_failure"]) for row in rows)
    engineering = bool(
        integrity_value.get("PHASE_A_IDEAL_IMPORT_PASS") is True
        and integrity_value.get("PHASE_B_SUBSET_REPRODUCTION_PASS") is True
        and integrity_value.get("new_snapshot_count") == 1050
        and integrity_value.get("new_trial_count") == 2100
        and integrity_value.get("combined_snapshot_count") == 1260
        and integrity_value.get("combined_trial_count") == 2520
        and backend_counts[OPEN3D_BACKEND] == 1260
        and backend_counts[PCL_BACKEND] == 1260
        and native_count == 0
        and all(
            int(integrity_value.get(name, -1)) == 0
            for name in (
                "missing_trial_count", "extra_trial_count", "duplicate_trial_count",
                "corrupt_trial_count", "backend_input_checksum_mismatch_count",
                "infrastructure_interruption_unresolved_count",
            )
        )
        and backend_exceptions == 0
        and nonfinite == 0
    )
    gates = {
        "COMMON_ASSOCIATION_ANALYSIS_PASS": common_validity["COMMON_ASSOCIATION_ANALYSIS_PASS"],
        "FULL_SYNTHETIC_CROSS_BACKEND_PASS": cross_pass,
        "FULL_SYNTHETIC_DEVELOPMENT_ENGINEERING_PASS": engineering,
        "FULL_SYNTHETIC_EXECUTION_ROBUSTNESS_PASS": robustness,
        "FULL_SYNTHETIC_PRIMARY_SCENE_EFFECT_PASS": primary_scene_pass,
        "FULL_SYNTHETIC_SCENE_RANK_STABILITY_PASS": rank_pass,
        "LOCAL_METRIC_INCREMENTAL_VALUE_PASS": incremental_pass,
        "REASSOCIATION_MECHANISM_SUPPORTED": mechanism_pass,
        "SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED": systematic_claim,
    }
    required_for_development = (
        "FULL_SYNTHETIC_DEVELOPMENT_ENGINEERING_PASS",
        "FULL_SYNTHETIC_EXECUTION_ROBUSTNESS_PASS",
        "FULL_SYNTHETIC_PRIMARY_SCENE_EFFECT_PASS",
        "FULL_SYNTHETIC_CROSS_BACKEND_PASS",
        "FULL_SYNTHETIC_SCENE_RANK_STABILITY_PASS",
        "COMMON_ASSOCIATION_ANALYSIS_PASS",
        "REASSOCIATION_MECHANISM_SUPPORTED",
        "LOCAL_METRIC_INCREMENTAL_VALUE_PASS",
    )
    if len(required_for_development) != 8 or any(
        name not in gates for name in required_for_development
    ):
        raise AssertionError("frozen eight-gate Development conjunction changed")
    development_pass = all(gates[name] for name in required_for_development)
    phenomenon_without_incremental = bool(
        not incremental_pass
        and all(gates[name] for name in required_for_development if name != "LOCAL_METRIC_INCREMENTAL_VALUE_PASS")
    )
    final_decision = {
        **gates,
        "CONFIRMATORY_PROTOCOL_DESIGN_AUTHORIZED": development_pass,
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "FULL_SYNTHETIC_DEVELOPMENT_PASS": development_pass,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PHENOMENON_CONFIRMED_BUT_INCREMENTAL_VALUE_NOT_ESTABLISHED": phenomenon_without_incremental,
        "REAL_DATA_PROTOCOL_DESIGN_AUTHORIZED": development_pass,
        "REAL_DATA_RUN_AUTHORIZED": False,
    }
    if (
        final_decision["CONFIRMATORY_PROTOCOL_DESIGN_AUTHORIZED"]
        is not final_decision["FULL_SYNTHETIC_DEVELOPMENT_PASS"]
        or final_decision["REAL_DATA_PROTOCOL_DESIGN_AUTHORIZED"]
        is not final_decision["FULL_SYNTHETIC_DEVELOPMENT_PASS"]
    ):
        raise AssertionError("Development authorization mirror changed")

    failure_inventory: list[dict[str, Any]] = []
    for backend in BACKENDS:
        for condition in CONDITIONS:
            selected = [
                row for row in rows
                if row["backend_schema_name"] == backend and row["condition"] == condition
            ]
            classifications = Counter(str(row["failure_classification"]) for row in selected)
            for classification in sorted(classifications):
                failure_inventory.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "failure_classification": classification,
                        "failure_count": classifications[classification],
                        "nonfinite_output_count": sum(not bool(row["finite_output"]) for row in selected),
                        "solver_failure_count": sum(bool(row["solver_failure"]) for row in selected),
                    }
                )
    runtime_summary = []
    for backend in BACKENDS:
        values = [float(row["runtime_ms"]) for row in rows if row["backend_schema_name"] == backend]
        summary = _describe(values)
        runtime_summary.append(
            {
                "backend": BACKEND_LABELS[backend],
                "backend_schema_name": backend,
                "maximum_runtime_ms": summary["maximum"],
                "median_runtime_ms": summary["median"],
                "q95_runtime_ms": summary["q95_linear"],
                "trial_count": summary["count"],
            }
        )
    protocol_summary = [
        {"contract": "scenes", "value": 7},
        {"contract": "conditions", "value": 6},
        {"contract": "geometry_seeds", "value": 3},
        {"contract": "measurement_seeds", "value": 2},
        {"contract": "repeat_indices", "value": 5},
        {"contract": "combined_snapshots", "value": 1260},
        {"contract": "combined_trials", "value": 2520},
        {"contract": "bootstrap_repetitions", "value": bootstrap_repetitions},
        {"contract": "bootstrap_seed", "value": BOOTSTRAP_SEED},
        {"contract": "quantile_method", "value": "linear"},
    ]
    return {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": None,
        "association_validity": common_validity,
        "automatic_nonequivalence_candidates": candidates,
        "backend_exception_count": backend_exceptions,
        "condition_contrasts": condition_contrasts,
        "cross_backend_ranking": cross_ranking,
        "failure_inventory": failure_inventory,
        "final_decision": final_decision,
        "gate_summary": gates,
        "geometry_seed_raw_values": rows,
        "incremental_value_detail": incremental_detail,
        "integrity": integrity_value,
        "joined_common_association_metrics": joined_common,
        "nonfinite_output_count": nonfinite,
        "protocol_summary": protocol_summary,
        "quantile_method": "linear",
        "ridge_model_comparison": model_rows,
        "runtime_summary": runtime_summary,
        "scene_condition_backend_summary": scene_summary,
        "scene_effect_paired": scene_effect,
        "scene_rank_stability": scene_ranking,
        "schema_version": "full_synthetic_development_primary_analysis_v1",
        "solver_failure_count": solver_failures,
        "success_rate_groups": success_group_rows,
        "systematic_claim_detail": systematic_gate_detail,
        "systematic_offset_summary": systematic,
        "turnover_correlations": turnover_correlations,
    }


def _validate_full_trial(value: Mapping[str, Any]) -> dict[str, Any]:
    """Extend only the condition enum while retaining the frozen 26-field schema."""

    if type(value) is not dict or value.get("condition") not in CONDITIONS:
        raise ValueError("Full Synthetic trial condition is not authorized")
    condition = value["condition"]
    normalized = dict(value)
    normalized["condition"] = "IDEAL_MATCHED"
    validated = validate_phase_a_trial_result_strict(normalized)
    validated["condition"] = condition
    return validated


def _plan_backend(value: str) -> str:
    if value not in PLAN_BACKEND_TO_RESULT:
        raise ValueError(f"unauthorized planned backend: {value}")
    return PLAN_BACKEND_TO_RESULT[value]


def _plan_int(row: Mapping[str, str], *names: str) -> int:
    return int(_field(row, *names))


@dataclass(frozen=True)
class RawSetAudit:
    rows: tuple[dict[str, Any], ...]
    corrupt_count: int
    duplicate_count: int
    extra_count: int
    missing_count: int


def _load_raw_set(
    *,
    plan_rows: Sequence[Mapping[str, str]],
    run_dir: Path,
    expected_run_id: str,
    expected_implementation_sha256: str,
    expected_protocol_sha256: str,
    validator,
) -> RawSetAudit:
    plan = {str(row["planned_trial_id"]): row for row in plan_rows}
    if len(plan) != len(plan_rows):
        raise ValueError("planned trial IDs are not unique")
    raw_manifest = json.loads((run_dir / "raw_result_manifest.json").read_text(encoding="utf-8"))
    if type(raw_manifest) is not dict or raw_manifest.get("run_id") != expected_run_id:
        raise ValueError("raw result manifest run identity mismatch")
    entries = raw_manifest.get("results")
    if type(entries) is not dict:
        raise ValueError("raw result manifest results must be an object")
    raw_root = (run_dir / "raw_results").resolve()
    referenced: list[str] = []
    for entry in entries.values():
        if type(entry) is dict and isinstance(entry.get("path"), str):
            candidate = (raw_root / entry["path"]).resolve()
            if candidate.parent == raw_root:
                referenced.append(candidate.name)
    corrupt = 0
    rows: list[dict[str, Any]] = []
    for trial_id in sorted(set(plan) & set(entries)):
        try:
            entry = entries[trial_id]
            if type(entry) is not dict or set(entry) != {"path", "planned_trial_id", "sha256"}:
                raise ValueError("raw result manifest entry schema mismatch")
            candidate = (raw_root / entry["path"]).resolve()
            if candidate.parent != raw_root or entry["planned_trial_id"] != trial_id:
                raise ValueError("raw result manifest entry identity mismatch")
            if not candidate.is_file() or file_sha256(candidate) != entry["sha256"]:
                raise ValueError("raw result checksum mismatch")
            result = validator(load_json_strict(candidate))
            expected = plan[trial_id]
            if any(
                result[name] != value
                for name, value in {
                    "planned_trial_id": trial_id,
                    "snapshot_id": expected["snapshot_id"],
                    "scene_variant": expected["scene_variant"],
                    "condition": expected["condition"],
                    "backend": _plan_backend(expected["backend"]),
                }.items()
            ):
                raise ValueError("raw result identity differs from frozen plan")
            if (
                result.get("implementation_sha256")
                != expected_implementation_sha256
                or result.get("protocol_sha256") != expected_protocol_sha256
            ):
                raise ValueError("raw result implementation/protocol SHA mismatch")
            rows.append(result)
        except (KeyError, OSError, TypeError, ValueError):
            corrupt += 1
    unreferenced = {
        path.name for path in raw_root.glob("*.json") if path.is_file()
    } - set(referenced)
    return RawSetAudit(
        rows=tuple(rows),
        corrupt_count=corrupt,
        duplicate_count=sum(max(0, count - 1) for count in Counter(referenced).values()),
        extra_count=len(set(entries) - set(plan)) + len(unreferenced),
        missing_count=len(set(plan) - set(entries)),
    )


def _manifest_path(
    root: Path, manifest: Mapping[str, Any], names: Sequence[str], fallback: str
) -> Path:
    value = next((manifest[name] for name in names if isinstance(manifest.get(name), str)), fallback)
    candidate = (root / value).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("manifest path escaped standalone repository")
    return candidate


def _require_manifest_sha_binding(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    path_fields: Sequence[str],
    sha_fields: Sequence[str],
    fallback: str,
    label: str,
) -> Path:
    """Resolve a repository-local file and verify its manifest SHA binding."""

    candidate = _manifest_path(root, manifest, path_fields, fallback)
    expected = next(
        (
            str(manifest[name])
            for name in sha_fields
            if isinstance(manifest.get(name), str)
        ),
        None,
    )
    if expected is None or len(expected) != 64:
        raise ValueError(f"{label} manifest SHA binding is missing")
    if not candidate.is_file() or file_sha256(candidate) != expected:
        raise ValueError(f"{label} manifest SHA binding mismatch")
    return candidate


def _load_new_snapshot_bindings(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Load plan/lock bindings without reading any array outside the strict reader.

    Array validation is deliberately performed later through
    ``read_full_synthetic_snapshot(plan, expected_lock_entry=...)`` so every
    analysis use is tied simultaneously to the plan, lock, metadata, file SHA,
    and raw scientific checksums.
    """

    snapshot_plan_path = _require_manifest_sha_binding(
        root=root,
        manifest=manifest,
        path_fields=("new_planned_snapshots_path", "planned_snapshots_path"),
        sha_fields=("new_planned_snapshots_sha256", "planned_snapshots_sha256"),
        fallback="frozen_assets/full_synthetic_development_planned_snapshots_v1.csv",
        label="new snapshot plan",
    )
    trial_plan_path = _require_manifest_sha_binding(
        root=root,
        manifest=manifest,
        path_fields=("new_planned_trials_path", "planned_trials_path"),
        sha_fields=("new_planned_trials_sha256", "planned_trials_sha256"),
        fallback="frozen_assets/full_synthetic_development_planned_trials_v1.csv",
        label="new trial plan",
    )
    lock_path = _require_manifest_sha_binding(
        root=root,
        manifest=manifest,
        path_fields=("new_snapshot_lock_path", "snapshot_lock_path"),
        sha_fields=("new_snapshot_lock_sha256", "snapshot_lock_sha256"),
        fallback="frozen_assets/full_synthetic_development_snapshot_lock_v1.json",
        label="new snapshot lock",
    )
    snapshot_plans = _csv(snapshot_plan_path)
    trial_plans = _csv(trial_plan_path)
    expected_snapshot_count = int(manifest.get("new_planned_snapshot_count", 1050))
    expected_trial_count = int(manifest.get("new_planned_trial_count", 2100))
    snapshot_by_id = {str(row["snapshot_id"]): row for row in snapshot_plans}
    trial_by_id = {str(row["planned_trial_id"]): row for row in trial_plans}
    if (
        len(snapshot_plans) != expected_snapshot_count
        or len(snapshot_by_id) != len(snapshot_plans)
        or len(trial_plans) != expected_trial_count
        or len(trial_by_id) != len(trial_plans)
    ):
        raise ValueError("new plan identity/inventory mismatch")
    if any(
        str(row["snapshot_id"]) not in snapshot_by_id for row in trial_plans
    ):
        raise ValueError("new trial plan has no matching snapshot plan")

    lock = load_json_strict(lock_path)
    stored_payload_sha = lock.get("snapshot_lock_payload_sha256")
    lock_payload = {
        name: value
        for name, value in lock.items()
        if name != "snapshot_lock_payload_sha256"
    }
    entries = lock.get("snapshots")
    if (
        stored_payload_sha != canonical_json_sha256(lock_payload)
        or lock.get("schema_version")
        != "full_synthetic_development_snapshot_lock_v1"
        or lock.get("planned_snapshot_count") != expected_snapshot_count
        or type(entries) is not list
        or len(entries) != expected_snapshot_count
        or [entry.get("snapshot_id") for entry in entries]
        != [str(row["snapshot_id"]) for row in snapshot_plans]
    ):
        raise ValueError("new snapshot lock identity/inventory mismatch")
    lock_by_id = {str(entry["snapshot_id"]): entry for entry in entries}
    if len(lock_by_id) != len(entries):
        raise ValueError("new snapshot lock contains duplicate snapshot IDs")
    return {
        "cache_root": _manifest_path(
            root,
            manifest,
            ("new_snapshot_cache_root", "snapshot_cache_root"),
            "data/full_synthetic_development_v1_snapshots",
        ),
        "lock_by_id": lock_by_id,
        "lock_sha256": file_sha256(lock_path),
        "snapshot_by_id": snapshot_by_id,
        "snapshot_plans": snapshot_plans,
        "trial_by_id": trial_by_id,
        "trial_plans": trial_plans,
    }


def _verified_phase_b_subset_report(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Read the subset gate only through its pre-run evidence SHA binding."""

    report_path = _manifest_path(
        root,
        manifest,
        ("phase_b_subset_reproduction_report_path",),
        "artifacts/full_synthetic_phase_b_subset_reproduction.json",
    )
    gate_path = _manifest_path(
        root,
        manifest,
        ("pre_run_gate_report_path",),
        "artifacts/full_synthetic_development_pre_run_gate_report.json",
    )
    report = load_json_strict(report_path)
    gate = load_json_strict(gate_path)
    relative = report_path.relative_to(root).as_posix()
    evidence = gate.get("evidence_sha256")
    if (
        type(evidence) is not dict
        or evidence.get(relative) != file_sha256(report_path)
        or gate.get("ALL_PRE_RUN_GATES_PASS") is not True
        or gate.get("PHASE_B_SUBSET_REPRODUCTION_PASS") is not True
        or report.get("PHASE_B_SUBSET_REPRODUCTION_PASS") is not True
    ):
        raise ValueError("Phase B subset reproduction evidence binding mismatch")
    return report


def load_full_synthetic_evidence(
    *, manifest_path: str | Path, run_dir: str | Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read and normalize the 420 imported and 2,100 new formal trial JSON files."""

    from .full_synthetic_development_protocol import (
        load_strict_authorized_full_synthetic_manifest,
    )

    manifest_file, manifest = load_strict_authorized_full_synthetic_manifest(
        manifest_path
    )
    root = manifest_root(manifest_file)
    new_bindings = _load_new_snapshot_bindings(root, manifest)
    new_plan = list(new_bindings["trial_plans"])
    new_snapshot_plan = list(new_bindings["snapshot_plans"])
    phase_a_plan = _csv(root / "frozen_assets/planned_trials.csv")
    phase_a_snapshots = _csv(root / "frozen_assets/planned_snapshots.csv")
    phase_a_snapshot_by_id = {
        str(row["snapshot_id"]): row for row in phase_a_snapshots
    }
    if len(phase_a_snapshots) != 210 or len(phase_a_snapshot_by_id) != 210:
        raise ValueError("Phase A snapshot plan identity/inventory mismatch")
    from .full_synthetic_development_protocol import verify_phase_a_ideal_import

    phase_a_verification = verify_phase_a_ideal_import(root, write_report=False)
    if phase_a_verification.get("PHASE_A_IDEAL_IMPORT_PASS") is not True:
        raise ValueError("Phase A frozen import verification failed")
    _phase_a_manifest_file, phase_a_manifest = load_manifest(
        root / "frozen_assets/frozen_experiment_manifest.json"
    )
    phase_a_lock_sha = str(phase_a_manifest["snapshot_lock_sha256"])
    subset_report = _verified_phase_b_subset_report(root, manifest)
    new_run = Path(run_dir).resolve()
    phase_a_run = _manifest_path(
        root,
        manifest,
        ("phase_a_results_root", "phase_a_formal_run_dir", "phase_a_ideal_run_dir"),
        "results/formal_phase_a_v1",
    )
    new_audit = _load_raw_set(
        plan_rows=new_plan,
        run_dir=new_run,
        expected_run_id=str(manifest.get("formal_run_id", "full-synthetic-development-v1")),
        expected_implementation_sha256=str(
            manifest["implementation_contract_sha256"]
        ),
        expected_protocol_sha256=str(manifest["scientific_protocol_sha256"]),
        validator=_validate_full_trial,
    )
    phase_a_audit = _load_raw_set(
        plan_rows=phase_a_plan,
        run_dir=phase_a_run,
        expected_run_id="phase-a-minimal-harness-formal-v1",
        expected_implementation_sha256=str(
            phase_a_manifest["manifest_payload_sha256"]
        ),
        expected_protocol_sha256=str(
            phase_a_manifest["scientific_protocol_sha256"]
        ),
        validator=validate_phase_a_trial_result_strict,
    )

    plan_by_id = {
        str(row["planned_trial_id"]): (row, False) for row in new_plan
    }
    plan_by_id.update(
        {str(row["planned_trial_id"]): (row, True) for row in phase_a_plan}
    )
    new_cache = Path(new_bindings["cache_root"])
    phase_a_cache = _manifest_path(
        root,
        manifest,
        ("phase_a_snapshot_cache_root",),
        "data/frozen_snapshots",
    )
    from .full_synthetic_snapshot_builder import read_full_synthetic_snapshot
    from .snapshot_reader import read_snapshot

    reference_cache: dict[tuple[bool, str], np.ndarray] = {}
    scientific_checksum_fields = (
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "snapshot_checksum",
    )
    metric_mismatches = 0
    normalized: list[dict[str, Any]] = []
    for result in [*phase_a_audit.rows, *new_audit.rows]:
        plan_row, is_ideal = plan_by_id[result["planned_trial_id"]]
        snapshot_id = str(result["snapshot_id"])
        cache_key = (is_ideal, snapshot_id)
        if cache_key not in reference_cache:
            if is_ideal:
                snapshot_plan = phase_a_snapshot_by_id.get(snapshot_id)
                if snapshot_plan is None:
                    raise ValueError("Phase A result has no matching snapshot plan")
                snapshot = read_snapshot(phase_a_cache, snapshot_id, arrays=True)
                metadata = snapshot["metadata"]
                plan_identity = {
                    "condition": str(snapshot_plan["condition"]),
                    "geometry_seed": int(snapshot_plan["geometry_seed_value"]),
                    "geometry_seed_index": int(snapshot_plan["geometry_seed_index"]),
                    "measurement_seed": int(snapshot_plan["measurement_seed_value"]),
                    "measurement_seed_index": int(snapshot_plan["measurement_seed_index"]),
                    "repeat_index": int(snapshot_plan["repeat_index"]),
                    "scene_variant": str(snapshot_plan["scene_variant"]),
                    "snapshot_id": snapshot_id,
                }
                if any(metadata.get(name) != value for name, value in plan_identity.items()):
                    raise ValueError("Phase A snapshot differs from its frozen plan")
                asset_checksums = {
                    "source_checksum": metadata["source_raw_checksum"],
                    "target_checksum": metadata["target_raw_checksum"],
                    "reference_pose_checksum": metadata["reference_pose_raw_checksum"],
                    "snapshot_checksum": metadata["snapshot_checksum"],
                }
                reference = snapshot["reference"]
            else:
                snapshot_plan = new_bindings["snapshot_by_id"].get(snapshot_id)
                lock_entry = new_bindings["lock_by_id"].get(snapshot_id)
                if snapshot_plan is None or lock_entry is None:
                    raise ValueError("new result has no matching plan/lock entry")
                snapshot = read_full_synthetic_snapshot(
                    new_cache,
                    snapshot_plan,
                    expected_lock_entry=lock_entry,
                    arrays=True,
                )
                asset_checksums = {
                    name: snapshot[name] for name in scientific_checksum_fields
                }
                reference = snapshot["reference"]
            if any(result.get(name) != value for name, value in asset_checksums.items()):
                raise ValueError("trial input checksum differs from strict snapshot asset")
            expected_lock_sha = (
                phase_a_lock_sha if is_ideal else new_bindings["lock_sha256"]
            )
            if result.get("snapshot_lock_sha256") != expected_lock_sha:
                raise ValueError("trial snapshot-lock SHA differs from frozen binding")
            if reference.shape != (4, 4) or not np.all(np.isfinite(reference)):
                raise ValueError(f"invalid frozen reference pose: {snapshot_id}")
            reference_cache[cache_key] = np.asarray(reference, dtype=np.float64)
        else:
            # Both backend rows must independently carry the same strictly read
            # input bindings; never infer the second row from the first one.
            if is_ideal:
                metadata = read_snapshot(
                    phase_a_cache, snapshot_id, arrays=False
                )["metadata"]
                asset_checksums = {
                    "source_checksum": metadata["source_raw_checksum"],
                    "target_checksum": metadata["target_raw_checksum"],
                    "reference_pose_checksum": metadata["reference_pose_raw_checksum"],
                    "snapshot_checksum": metadata["snapshot_checksum"],
                }
                expected_lock_sha = phase_a_lock_sha
            else:
                snapshot_plan = new_bindings["snapshot_by_id"].get(snapshot_id)
                lock_entry = new_bindings["lock_by_id"].get(snapshot_id)
                if snapshot_plan is None or lock_entry is None:
                    raise ValueError("new result has no matching plan/lock entry")
                snapshot = read_full_synthetic_snapshot(
                    new_cache,
                    snapshot_plan,
                    expected_lock_entry=lock_entry,
                    arrays=False,
                )
                asset_checksums = {
                    "source_checksum": lock_entry["source_checksum"],
                    "target_checksum": lock_entry["target_checksum"],
                    "reference_pose_checksum": lock_entry["reference_pose_checksum"],
                    "snapshot_checksum": lock_entry["snapshot_checksum"],
                }
                expected_lock_sha = new_bindings["lock_sha256"]
            if (
                any(result.get(name) != value for name, value in asset_checksums.items())
                or result.get("snapshot_lock_sha256") != expected_lock_sha
            ):
                raise ValueError("trial input binding differs across backend rows")
        vectors = {
            "rotation_error_rad": result["rotation_update_rad"],
            "rotation_vector": None,
            "translation_error_m": result["translation_update_m"],
            "translation_vector": None,
        }
        if result["final_transform_4x4"] is not None:
            vectors = transform_error_vectors(
                reference_cache[cache_key], result["final_transform_4x4"]
            )
            for computed, stored in (
                (vectors["translation_error_m"], result["translation_update_m"]),
                (vectors["rotation_error_rad"], result["rotation_update_rad"]),
            ):
                if stored is None or not math.isclose(computed, float(stored), rel_tol=1.0e-12, abs_tol=1.0e-12):
                    metric_mismatches += 1
        backend = str(result["backend"])
        normalized.append(
            {
                **result,
                **vectors,
                "backend": BACKEND_LABELS[backend],
                "backend_schema_name": backend,
                "geometry_seed": _plan_int(plan_row, "geometry_seed_value", "geometry_seed"),
                "measurement_seed": _plan_int(plan_row, "measurement_seed_value", "measurement_seed"),
                "repeat_index": int(plan_row["repeat_index"]),
            }
        )

    pairing_mismatch = 0
    by_snapshot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_snapshot[str(row["snapshot_id"])].append(row)
    for rows in by_snapshot.values():
        checksums = ("source_checksum", "target_checksum", "reference_pose_checksum", "snapshot_checksum")
        if (
            len(rows) != 2
            or Counter(row["backend_schema_name"] for row in rows) != Counter(BACKENDS)
            or any(len({row[name] for row in rows}) != 1 for name in checksums)
        ):
            pairing_mismatch += 1

    ideal_rows = list(phase_a_audit.rows)
    phase_a_import_pass = bool(
        phase_a_verification.get("PHASE_A_IDEAL_IMPORT_PASS") is True
        and len(phase_a_snapshots) == 210
        and len(ideal_rows) == 420
        and sum(row["backend"] == OPEN3D_BACKEND for row in ideal_rows) == 210
        and sum(row["backend"] == PCL_BACKEND for row in ideal_rows) == 210
        and not any(row["solver_failure"] or not row["finite_output"] for row in ideal_rows)
        and phase_a_audit.corrupt_count == phase_a_audit.duplicate_count == 0
        and phase_a_audit.extra_count == phase_a_audit.missing_count == 0
    )
    subset_pass = subset_report.get("PHASE_B_SUBSET_REPRODUCTION_PASS") is True
    new_by_snapshot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in new_audit.rows:
        new_by_snapshot[str(row["snapshot_id"])].append(row)
    new_completed_snapshot_count = sum(
        len(selected) == 2
        and Counter(row["backend"] for row in selected) == Counter(BACKENDS)
        for selected in new_by_snapshot.values()
    )
    combined_completed_snapshot_count = sum(
        len(selected) == 2
        and Counter(row["backend_schema_name"] for row in selected)
        == Counter(BACKENDS)
        for selected in by_snapshot.values()
    )
    integrity = {
        "PHASE_A_IDEAL_IMPORT_PASS": phase_a_import_pass,
        "PHASE_B_SUBSET_REPRODUCTION_PASS": subset_pass,
        "backend_input_checksum_mismatch_count": pairing_mismatch + metric_mismatches,
        "combined_snapshot_count": combined_completed_snapshot_count,
        "combined_trial_count": len(normalized),
        "corrupt_trial_count": new_audit.corrupt_count + phase_a_audit.corrupt_count,
        "duplicate_trial_count": new_audit.duplicate_count + phase_a_audit.duplicate_count,
        "extra_trial_count": new_audit.extra_count + phase_a_audit.extra_count,
        "infrastructure_interruption_unresolved_count": (
            new_audit.missing_count + new_audit.corrupt_count
        ),
        "metric_recomputation_mismatch_count": metric_mismatches,
        "missing_trial_count": new_audit.missing_count + phase_a_audit.missing_count,
        "new_snapshot_count": new_completed_snapshot_count,
        "new_trial_count": len(new_audit.rows),
        "planned_new_snapshot_count": len(new_snapshot_plan),
        "planned_new_trial_count": len(new_plan),
    }
    normalized.sort(key=lambda row: str(row["planned_trial_id"]))
    return normalized, integrity


def load_common_association_records(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if type(value) is dict and "common_association_payload_sha256" in value:
        payload = {
            name: item
            for name, item in value.items()
            if name != "common_association_payload_sha256"
        }
        if value["common_association_payload_sha256"] != canonical_json_sha256(payload):
            raise ValueError("common association cache payload SHA mismatch")
    records = value.get("records") if type(value) is dict else value
    if type(records) is dict:
        records = list(records.values())
    if type(records) is not list or any(type(row) is not dict for row in records):
        raise ValueError("common association record file must contain a record list/object")
    return [dict(row) for row in records]


def _common_cache_provenance(
    *,
    manifest_file: Path,
    manifest: Mapping[str, Any],
    run_dir: Path,
    trials: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    from . import common_association_analysis as analyzer_module

    analyzer_path = Path(analyzer_module.__file__).resolve()
    analyzer_sha = file_sha256(analyzer_path)
    analyzer_binding = manifest.get("code_bindings", {}).get(
        "association_analysis", {}
    )
    if (
        type(analyzer_binding) is not dict
        or analyzer_binding.get("sha256") != analyzer_sha
    ):
        raise ValueError("common association analyzer SHA binding mismatch")
    raw_manifest_path = (run_dir / "raw_result_manifest.json").resolve()
    if raw_manifest_path.parent != run_dir.resolve() or not raw_manifest_path.is_file():
        raise ValueError("common association raw manifest is missing")
    planned_ids = sorted(str(row["planned_trial_id"]) for row in bindings["trial_plans"])
    result_ids = sorted(
        str(row["planned_trial_id"])
        for row in trials
        if row["condition"] in NONIDEAL_CONDITIONS
    )
    eligible_ids = sorted(
        str(row["planned_trial_id"])
        for row in trials
        if row["condition"] in NONIDEAL_CONDITIONS
        and not bool(row["solver_failure"])
        and bool(row["finite_output"])
    )
    if (
        len(planned_ids) != len(set(planned_ids))
        or len(result_ids) != len(set(result_ids))
        or result_ids != planned_ids
    ):
        raise ValueError("common association result/plan ID inventory mismatch")
    return {
        "common_association_analyzer_sha256": analyzer_sha,
        "experiment_manifest_sha256": file_sha256(manifest_file),
        "planned_trial_id_inventory_sha256": canonical_json_sha256(
            {"planned_trial_ids": planned_ids}
        ),
        "raw_result_manifest_sha256": file_sha256(raw_manifest_path),
        "record_trial_id_inventory_sha256": canonical_json_sha256(
            {"record_trial_ids": eligible_ids}
        ),
        "snapshot_lock_sha256": bindings["lock_sha256"],
    }


def validate_common_association_cache(
    *,
    manifest_path: str | Path,
    run_dir: str | Path,
    trials: Sequence[Mapping[str, Any]],
    cache_path: str | Path,
) -> list[dict[str, Any]]:
    """Reject stale, duplicated, extra, or provenance-free common metrics."""

    manifest_file, manifest = load_manifest(manifest_path, require_authorized=True)
    root = manifest_root(manifest_file)
    bindings = _load_new_snapshot_bindings(root, manifest)
    expected = _common_cache_provenance(
        manifest_file=manifest_file,
        manifest=manifest,
        run_dir=Path(run_dir).resolve(),
        trials=trials,
        bindings=bindings,
    )
    candidate = Path(cache_path).resolve()
    value = load_json_strict(candidate)
    if any(value.get(name) != digest for name, digest in expected.items()):
        raise ValueError("common association cache provenance is stale or mismatched")
    records = load_common_association_records(candidate)
    ids = [str(row.get("planned_trial_id")) for row in records]
    expected_ids = sorted(
        str(row["planned_trial_id"])
        for row in trials
        if row["condition"] in NONIDEAL_CONDITIONS
        and not bool(row["solver_failure"])
        and bool(row["finite_output"])
    )
    if (
        len(ids) != len(set(ids))
        or sorted(ids) != expected_ids
        or value.get("record_count") != len(expected_ids)
        or value.get("planned_trial_binding_count") != len(bindings["trial_by_id"])
        or value.get("planned_snapshot_binding_count")
        != len(bindings["snapshot_by_id"])
    ):
        raise ValueError("common association cache record inventory mismatch")
    return records


def build_common_association_records(
    *,
    manifest_path: str | Path,
    trials: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Prepare one context per new snapshot and reuse it for both backends."""

    from .common_association_analysis import (
        prepare_common_association_context,
        safe_analyze_estimated_transform,
    )

    manifest_file, manifest = load_manifest(manifest_path, require_authorized=True)
    root = manifest_root(manifest_file)
    bindings = _load_new_snapshot_bindings(root, manifest)
    cache_root = Path(bindings["cache_root"])
    provenance = _common_cache_provenance(
        manifest_file=manifest_file,
        manifest=manifest,
        run_dir=Path(output_path).resolve().parent,
        trials=trials,
        bindings=bindings,
    )
    from .full_synthetic_snapshot_builder import read_full_synthetic_snapshot

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_trial_ids: list[str] = []
    for row in trials:
        if row["condition"] in NONIDEAL_CONDITIONS:
            grouped[str(row["snapshot_id"])].append(row)
            seen_trial_ids.append(str(row["planned_trial_id"]))
    if (
        len(seen_trial_ids) != len(set(seen_trial_ids))
        or set(seen_trial_ids) != set(bindings["trial_by_id"])
        or set(grouped) != set(bindings["snapshot_by_id"])
    ):
        raise ValueError("common association input does not exactly match the new plan")
    records: list[dict[str, Any]] = []
    context_count = 0
    backend_analysis_count = 0
    for snapshot_id in sorted(grouped):
        selected = grouped[snapshot_id]
        snapshot_plan = bindings["snapshot_by_id"].get(snapshot_id)
        lock_entry = bindings["lock_by_id"].get(snapshot_id)
        if snapshot_plan is None or lock_entry is None:
            raise ValueError("common association snapshot has no plan/lock binding")
        if (
            len(selected) != 2
            or Counter(str(row["backend_schema_name"]) for row in selected)
            != Counter(BACKENDS)
        ):
            raise ValueError("common association snapshot backend pairing mismatch")
        snapshot = read_full_synthetic_snapshot(
            cache_root,
            snapshot_plan,
            expected_lock_entry=lock_entry,
            arrays=True,
        )
        source = snapshot["source"]
        target = snapshot["target"]
        reference = snapshot["reference"]
        for trial in selected:
            planned_trial_id = str(trial["planned_trial_id"])
            trial_plan = bindings["trial_by_id"].get(planned_trial_id)
            if trial_plan is None:
                raise ValueError("common association trial has no planned row")
            expected_identity = {
                "snapshot_id": str(trial_plan["snapshot_id"]),
                "scene_variant": str(trial_plan["scene_variant"]),
                "condition": str(trial_plan["condition"]),
                "backend_schema_name": _plan_backend(str(trial_plan["backend"])),
                "geometry_seed": _plan_int(
                    trial_plan, "geometry_seed_value", "geometry_seed"
                ),
                "measurement_seed": _plan_int(
                    trial_plan, "measurement_seed_value", "measurement_seed"
                ),
                "repeat_index": int(trial_plan["repeat_index"]),
            }
            if any(trial.get(name) != value for name, value in expected_identity.items()):
                raise ValueError("common association trial differs from its planned row")
            if (
                any(
                    trial.get(name) != snapshot[name]
                    for name in (
                        "source_checksum",
                        "target_checksum",
                        "reference_pose_checksum",
                        "snapshot_checksum",
                    )
                )
                or trial.get("snapshot_lock_sha256") != bindings["lock_sha256"]
            ):
                raise ValueError(
                    "common association trial input differs from strict snapshot binding"
                )
        try:
            context = prepare_common_association_context(
                source, target, reference, snapshot_id=snapshot_id
            )
            context_count += 1
        except Exception as error:
            for trial in selected:
                if bool(trial["solver_failure"]) or not bool(trial["finite_output"]):
                    continue
                records.append(
                    {
                        "backend_schema_name": trial["backend_schema_name"],
                        "common_association_invalid_detail": f"{type(error).__name__}: {error}",
                        "common_association_invalid_reason": "OTHER",
                        "common_association_is_backend_internal": False,
                        "common_association_valid": False,
                        "condition": trial["condition"],
                        "planned_trial_id": trial["planned_trial_id"],
                        "snapshot_id": snapshot_id,
                    }
                )
            continue
        for trial in sorted(selected, key=lambda row: str(row["planned_trial_id"])):
            if bool(trial["solver_failure"]) or not bool(trial["finite_output"]):
                continue
            estimated = trial.get("final_transform_4x4")
            identifiers = {
                "backend_schema_name": trial["backend_schema_name"],
                "condition": trial["condition"],
                "geometry_seed": int(trial["geometry_seed"]),
                "measurement_seed": int(trial["measurement_seed"]),
                "planned_trial_id": trial["planned_trial_id"],
                "repeat_index": int(trial["repeat_index"]),
                "scene_variant": trial["scene_variant"],
            }
            records.append(
                safe_analyze_estimated_transform(
                    context, np.asarray(estimated, dtype=np.float64), identifiers=identifiers
                )
            )
            backend_analysis_count += 1
    payload = {
        **provenance,
        "backend_analysis_count": backend_analysis_count,
        "common_context_preparation_count": context_count,
        "context_reused_across_backends": True,
        "planned_snapshot_binding_count": len(bindings["snapshot_by_id"]),
        "planned_trial_binding_count": len(bindings["trial_by_id"]),
        "records": records,
        "record_count": len(records),
        "schema_version": "full_synthetic_common_association_records_v1",
        "snapshot_lock_sha256": bindings["lock_sha256"],
        "strict_snapshot_binding_pass": True,
        "trial_input_checksum_mismatch_count": 0,
    }
    expected_record_ids = sorted(
        str(row["planned_trial_id"])
        for row in trials
        if row["condition"] in NONIDEAL_CONDITIONS
        and not bool(row["solver_failure"])
        and bool(row["finite_output"])
    )
    actual_record_ids = [str(row.get("planned_trial_id")) for row in records]
    if len(actual_record_ids) != len(set(actual_record_ids)) or sorted(actual_record_ids) != expected_record_ids:
        raise ValueError("common association output record inventory mismatch")
    payload["common_association_payload_sha256"] = canonical_json_sha256(payload)
    write_json(output_path, payload)
    return payload


def analyze_full_synthetic_development(
    *,
    manifest_path: str | Path,
    run_dir: str | Path,
    common_metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    trials, integrity = load_full_synthetic_evidence(
        manifest_path=manifest_path, run_dir=run_dir
    )
    metrics_path = (
        Path(common_metrics_path)
        if common_metrics_path is not None
        else Path(run_dir) / "common_association_metrics.json"
    )
    if not metrics_path.is_file():
        build_common_association_records(
            manifest_path=manifest_path,
            trials=trials,
            output_path=metrics_path,
        )
    common = validate_common_association_cache(
        manifest_path=manifest_path,
        run_dir=run_dir,
        trials=trials,
        cache_path=metrics_path,
    )
    report = analyze_full_synthetic_record_collections(
        trials=trials,
        common_records=common,
        integrity=integrity,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
    )
    report["run_id"] = "full-synthetic-development-v1"
    if output_path is not None:
        write_json(output_path, report)
    return report


def full_synthetic_verification_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical comparison surface for every reported or decision statistic."""

    scene_fields = (
        "backend_schema_name", "condition", "scene_variant", "planned_trial_count",
        "successful_trial_count", "success_rate", "solver_failure_count",
        "translation_median_m", "translation_iqr_m", "translation_q25_m",
        "translation_q75_m", "translation_q95_m",
        "rotation_median_rad", "rotation_iqr_rad", "rotation_q25_rad",
        "rotation_q75_rad", "rotation_q95_rad",
        "exploratory_translation_median_ci95_low_m",
        "exploratory_translation_median_ci95_high_m",
    )
    systematic_fields = (
        "backend_schema_name", "condition", "geometry_seed", "scene_variant",
        "planned_observation_count", "successful_observation_count",
        "mean_translation_vector",
        "systematic_translation_offset_m", "translation_repeatability_covariance",
        "translation_repeatability_rms_m", "mean_rotation_vector",
        "systematic_rotation_offset_rad", "rotation_repeatability_covariance",
        "rotation_repeatability_rms_rad", "systematic_fraction_translation",
        "translation_direction_concentration", "direction_valid_nonzero_count",
    )
    effect_fields = (
        "backend_schema_name", "condition", "matched_block_count", "rich_median_m",
        "corridor_median_m", "weak_rich_median_ratio", "absolute_median_difference_m",
        "paired_win_count", "paired_win_rate", "rich_success_rate",
        "corridor_success_rate", "gate_pass",
        "exploratory_difference_ci95_low_m",
        "exploratory_difference_ci95_high_m",
        "exploratory_ratio_ci95_low", "exploratory_ratio_ci95_high",
    )
    rank_fields = (
        "backend_schema_name", "condition", "scene_variant", "translation_median_m",
        "translation_rank_ascending_average_ties", "is_tied",
        "rich_lowest_two_group_pass", "weak_highest_three_group_pass",
    )
    contrast_fields = (
        "backend_schema_name", "baseline_condition", "baseline_median_m",
        "contrast_condition", "contrast_median_m",
        "exploratory_paired_difference_ci95_high_m",
        "exploratory_paired_difference_ci95_low_m", "matched_block_count",
        "median_ratio", "paired_median_difference_m", "paired_win_rate",
        "scene_variant",
    )
    turnover_fields = (
        "backend_schema_name", "centered_exploratory_ci95_high",
        "centered_exploratory_ci95_low", "centered_pvalue_descriptive",
        "centered_spearman_rho", "pooled_exploratory_ci95_high",
        "pooled_exploratory_ci95_low", "pooled_pvalue_descriptive",
        "pooled_spearman_rho", "valid_trial_count",
    )
    candidates = []
    for source in report["automatic_nonequivalence_candidates"]:
        row = dict(source)
        row.pop("backend", None)
        candidates.append(row)
    candidates.sort(key=lambda row: str(row["candidate_pair_id"]))
    candidate_counts = Counter(row["backend_schema_name"] for row in candidates)
    candidate_scene_pairs = sorted(
        {
            tuple(sorted((row["scene_a"], row["scene_b"])))
            for row in candidates
        }
    )
    trial_fields = (
        "planned_trial_id", "snapshot_id", "scene_variant", "condition",
        "backend_schema_name", "geometry_seed", "measurement_seed", "repeat_index",
        "solver_failure", "failure_classification", "finite_output",
        "translation_error_m", "rotation_error_rad", "translation_vector",
        "rotation_vector", "runtime_ms", "source_checksum", "target_checksum",
        "reference_pose_checksum", "snapshot_checksum",
    )
    normalized_trials = [
        {name: row.get(name) for name in trial_fields}
        for row in report.get("geometry_seed_raw_values", [])
    ]
    normalized_trials.sort(key=lambda row: str(row["planned_trial_id"]))
    common_rows = []
    for source in report.get("joined_common_association_metrics", []):
        row = dict(source)
        row.pop("backend", None)
        common_rows.append(row)
    common_rows.sort(key=lambda row: str(row.get("planned_trial_id")))
    return {
        "association_validity": report["association_validity"],
        "automatic_candidate_count_by_backend": {
            backend: candidate_counts[backend] for backend in BACKENDS
        },
        "automatic_candidate_scene_pairs": candidate_scene_pairs,
        "automatic_nonequivalence_candidates": candidates,
        "backend_exception_count": report["backend_exception_count"],
        "condition_contrasts": [
            {key: row[key] for key in contrast_fields}
            for row in report["condition_contrasts"]
        ],
        "cross_backend_ranking": report["cross_backend_ranking"],
        "failure_inventory": report["failure_inventory"],
        "final_decision": report["final_decision"],
        "gate_summary": report["gate_summary"],
        "integrity": report["integrity"],
        "incremental_value_detail": report["incremental_value_detail"],
        "joined_common_association_metrics": common_rows,
        "nonfinite_output_count": report["nonfinite_output_count"],
        "normalized_trial_metrics": normalized_trials,
        "ridge_model_comparison": [
            {
                key: row[key]
                for key in (
                    "backend_schema_name", "sample_count", "model_a_cv_mae",
                    "model_b_cv_mae", "relative_mae_improvement", "fold_results",
                )
            }
            for row in report["ridge_model_comparison"]
        ],
        "scene_condition_backend_summary": [
            {key: row[key] for key in scene_fields}
            for row in report["scene_condition_backend_summary"]
        ],
        "scene_effect_paired": [
            {key: row[key] for key in effect_fields}
            for row in report["scene_effect_paired"]
        ],
        "scene_rank_stability": [
            {key: row[key] for key in rank_fields}
            for row in report["scene_rank_stability"]
        ],
        "solver_failure_count": report["solver_failure_count"],
        "success_rate_groups": [
            {
                key: row[key]
                for key in (
                    "backend_schema_name", "condition", "gate_level", "gate_pass",
                    "planned_count", "scene_variant", "success_count", "success_rate",
                )
            }
            for row in report["success_rate_groups"]
        ],
        "systematic_claim_detail": [
            {
                key: row[key]
                for key in (
                    "backend_schema_name", "condition", "gate_pass",
                    "geometry_group_count", "median_systematic_fraction_translation",
                    "qualified_geometry_group_count", "scene_variant",
                )
            }
            for row in report["systematic_claim_detail"]
        ],
        "systematic_offset_summary": [
            {key: row[key] for key in systematic_fields}
            for row in report["systematic_offset_summary"]
        ],
        "runtime_summary": [
            {
                key: row[key]
                for key in (
                    "backend_schema_name", "trial_count", "median_runtime_ms",
                    "q95_runtime_ms", "maximum_runtime_ms",
                )
            }
            for row in report["runtime_summary"]
        ],
        "turnover_correlations": [
            {key: row[key] for key in turnover_fields}
            for row in report["turnover_correlations"]
        ],
    }


__all__ = [
    "BACKEND_LABELS",
    "BACKENDS",
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "CONDITIONS",
    "GEOMETRY_SEEDS",
    "MAIN_CONDITIONS",
    "MEASUREMENT_SEEDS",
    "NONIDEAL_CONDITIONS",
    "REPEAT_INDICES",
    "SCENES",
    "analyze_full_synthetic_development",
    "analyze_full_synthetic_record_collections",
    "association_validity_and_turnover",
    "automatic_nonequivalence_candidates",
    "condition_contrast_rows",
    "cross_backend_and_rank_rows",
    "full_synthetic_verification_projection",
    "incremental_value_gate",
    "build_common_association_records",
    "join_common_records",
    "load_common_association_records",
    "load_full_synthetic_evidence",
    "primary_scene_effect_rows",
    "ridge_model_rows",
    "scene_condition_backend_rows",
    "systematic_claim_gate",
    "systematic_offset_rows",
    "transform_error_vectors",
]
