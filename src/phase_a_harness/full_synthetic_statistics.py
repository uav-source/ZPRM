"""Exploratory Development summaries and frozen hierarchical bootstrap."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 1191248828
QUANTILE_METHOD = "linear"
EXPLORATORY_INTERVAL_QUANTILES = (0.025, 0.975)
GEOMETRY_SEED_COUNT = 3
INNER_BLOCK_DEFINITION = "measurement_seed_x_repeat_index"
DEVELOPMENT_GEOMETRY_SEEDS = (1850310744, 1957656152, 1334931069)
DEVELOPMENT_MEASUREMENT_SEEDS = (217775206, 1664898153)
DEVELOPMENT_REPEAT_INDICES = (0, 1, 2, 3, 4)


def _finite_vector(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a non-empty finite vector")
    return array


def descriptive_development_summary(values: Sequence[float]) -> dict[str, Any]:
    """Return median, linear-method IQR, and q95 without inferential claims."""

    array = _finite_vector(values, "values")
    q25, median, q75, q95 = np.quantile(
        array, [0.25, 0.50, 0.75, 0.95], method=QUANTILE_METHOD
    )
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "q95": float(q95),
        "max": float(np.max(array)),
        "quantile_method": QUANTILE_METHOD,
    }


def _bootstrap_groups(
    geometry_seed: Sequence[Any],
    measurement_seed: Sequence[Any],
    repeat_index: Sequence[Any],
) -> tuple[np.ndarray, dict[Any, dict[tuple[Any, Any], np.ndarray]]]:
    geometry = np.asarray(geometry_seed, dtype=object)
    measurement = np.asarray(measurement_seed, dtype=object)
    repeat = np.asarray(repeat_index, dtype=object)
    if geometry.ndim != 1 or measurement.shape != geometry.shape or repeat.shape != geometry.shape:
        raise ValueError("bootstrap key vectors must be aligned one-dimensional arrays")
    if geometry.size == 0:
        raise ValueError("bootstrap requires at least one row")
    observed_geometry = set(geometry.tolist())
    geometry_order: Sequence[Any]
    if observed_geometry == set(DEVELOPMENT_GEOMETRY_SEEDS):
        geometry_order = DEVELOPMENT_GEOMETRY_SEEDS
    else:
        geometry_order = sorted(observed_geometry, key=lambda item: str(item))
    unique_geometry = np.asarray(geometry_order, dtype=object)
    if unique_geometry.size != GEOMETRY_SEED_COUNT:
        raise ValueError("Development bootstrap requires exactly three geometry seeds")
    groups: dict[Any, dict[tuple[Any, Any], np.ndarray]] = {}
    for seed in unique_geometry:
        blocks: dict[tuple[Any, Any], list[int]] = defaultdict(list)
        for index in np.flatnonzero(geometry == seed):
            blocks[(measurement[index], repeat[index])].append(int(index))
        if not blocks:
            raise ValueError("geometry seed has no inner bootstrap blocks")
        formal_inner_order = [
            (measurement_value, repeat_value)
            for measurement_value in DEVELOPMENT_MEASUREMENT_SEEDS
            for repeat_value in DEVELOPMENT_REPEAT_INDICES
        ]
        if set(blocks) == set(formal_inner_order):
            block_order = formal_inner_order
        else:
            block_order = sorted(
                blocks, key=lambda key: tuple(str(value) for value in key)
            )
        groups[seed] = {
            key: np.asarray(blocks[key], dtype=np.int64) for key in block_order
        }
    return unique_geometry, groups


def hierarchical_bootstrap_indices(
    geometry_seed: Sequence[Any],
    measurement_seed: Sequence[Any],
    repeat_index: Sequence[Any],
) -> tuple[np.ndarray, ...]:
    """Generate exactly 2000 geometry-outer, measurement×repeat-inner samples."""

    geometry, groups = _bootstrap_groups(
        geometry_seed, measurement_seed, repeat_index
    )
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    samples: list[np.ndarray] = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        selected_geometry = generator.choice(
            geometry, size=geometry.size, replace=True
        )
        pieces: list[np.ndarray] = []
        for seed in selected_geometry:
            block_map = groups[seed]
            block_keys = list(block_map)
            selected_blocks = generator.integers(
                0, len(block_keys), size=len(block_keys)
            )
            pieces.extend(block_map[block_keys[int(index)]] for index in selected_blocks)
        samples.append(np.concatenate(pieces))
    return tuple(samples)


def _interval(replicates: Sequence[float]) -> tuple[float | None, float | None, int]:
    finite = np.asarray(
        [value for value in replicates if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return None, None, 0
    low, high = np.quantile(
        finite, EXPLORATORY_INTERVAL_QUANTILES, method=QUANTILE_METHOD
    )
    return float(low), float(high), int(finite.size)


def hierarchical_bootstrap_interval(
    values: Sequence[float],
    geometry_seed: Sequence[Any],
    measurement_seed: Sequence[Any],
    repeat_index: Sequence[Any],
    *,
    statistic: Callable[[np.ndarray], float] = np.median,
) -> dict[str, Any]:
    array = _finite_vector(values, "values")
    if not (
        len(geometry_seed)
        == len(measurement_seed)
        == len(repeat_index)
        == array.size
    ):
        raise ValueError("bootstrap keys and values are not aligned")
    samples = hierarchical_bootstrap_indices(
        geometry_seed, measurement_seed, repeat_index
    )
    if any(np.any(indices >= array.size) for indices in samples):
        raise ValueError("bootstrap keys and values are not aligned")
    replicates = [float(statistic(array[indices])) for indices in samples]
    low, high, valid_count = _interval(replicates)
    return {
        "exploratory_interval_low": low,
        "exploratory_interval_high": high,
        "valid_bootstrap_replicate_count": valid_count,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "outer_bootstrap_unit": "geometry_seed",
        "inner_bootstrap_unit": INNER_BLOCK_DEFINITION,
        "geometry_seed_count": GEOMETRY_SEED_COUNT,
        "development_only_not_confirmatory": True,
    }


def paired_development_summary(
    left: Sequence[float],
    right: Sequence[float],
    geometry_seed: Sequence[Any],
    measurement_seed: Sequence[Any],
    repeat_index: Sequence[Any],
) -> dict[str, Any]:
    """Summarize block-paired values and their hierarchical bootstrap intervals."""

    left_array = _finite_vector(left, "left")
    right_array = _finite_vector(right, "right")
    if left_array.shape != right_array.shape:
        raise ValueError("paired vectors must have equal length")
    if not (
        len(geometry_seed)
        == len(measurement_seed)
        == len(repeat_index)
        == left_array.size
    ):
        raise ValueError("bootstrap keys and paired values are not aligned")
    difference = left_array - right_array
    valid_ratio = right_array > 1.0e-12
    ratios = left_array[valid_ratio] / right_array[valid_ratio]
    samples = hierarchical_bootstrap_indices(
        geometry_seed, measurement_seed, repeat_index
    )
    if any(np.any(indices >= left_array.size) for indices in samples):
        raise ValueError("bootstrap keys and paired values are not aligned")
    difference_replicates = [float(np.median(difference[index])) for index in samples]
    ratio_of_medians_replicates: list[float] = []
    for index in samples:
        denominator = float(np.median(right_array[index]))
        if denominator > 1.0e-12:
            ratio_of_medians_replicates.append(
                float(np.median(left_array[index]) / denominator)
            )
    difference_low, difference_high, difference_valid = _interval(
        difference_replicates
    )
    ratio_low, ratio_high, ratio_valid = _interval(ratio_of_medians_replicates)
    right_median = float(np.median(right_array))
    return {
        "paired_count": int(left_array.size),
        "left_median": float(np.median(left_array)),
        "right_median": right_median,
        "paired_median_difference": float(np.median(difference)),
        "ratio_of_medians": (
            None
            if right_median <= 1.0e-12
            else float(np.median(left_array) / right_median)
        ),
        "paired_median_ratio": None if ratios.size == 0 else float(np.median(ratios)),
        "paired_ratio_valid_count": int(ratios.size),
        "paired_win_count": int(np.count_nonzero(left_array > right_array)),
        "paired_win_rate": float(np.mean(left_array > right_array)),
        "paired_tie_count": int(np.count_nonzero(left_array == right_array)),
        "paired_difference_exploratory_interval_low": difference_low,
        "paired_difference_exploratory_interval_high": difference_high,
        "paired_difference_valid_bootstrap_count": difference_valid,
        "ratio_of_medians_exploratory_interval_low": ratio_low,
        "ratio_of_medians_exploratory_interval_high": ratio_high,
        "ratio_of_medians_valid_bootstrap_count": ratio_valid,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "outer_bootstrap_unit": "geometry_seed",
        "inner_bootstrap_unit": INNER_BLOCK_DEFINITION,
        "development_only_not_confirmatory": True,
    }


def spearman_development_summary(
    left: Sequence[float],
    right: Sequence[float],
    geometry_seed: Sequence[Any],
    measurement_seed: Sequence[Any],
    repeat_index: Sequence[Any],
) -> dict[str, Any]:
    """Spearman rho with a descriptive hierarchical bootstrap interval."""

    left_array = _finite_vector(left, "left")
    right_array = _finite_vector(right, "right")
    if left_array.shape != right_array.shape:
        raise ValueError("Spearman vectors must have equal length")
    if not (
        len(geometry_seed)
        == len(measurement_seed)
        == len(repeat_index)
        == left_array.size
    ):
        raise ValueError("bootstrap keys and Spearman values are not aligned")
    observed = spearmanr(left_array, right_array)
    rho = float(observed.statistic)
    pvalue = float(observed.pvalue)
    samples = hierarchical_bootstrap_indices(
        geometry_seed, measurement_seed, repeat_index
    )
    if any(np.any(indices >= left_array.size) for indices in samples):
        raise ValueError("bootstrap keys and Spearman values are not aligned")
    replicates: list[float] = []
    for index in samples:
        result = spearmanr(left_array[index], right_array[index])
        value = float(result.statistic)
        if math.isfinite(value):
            replicates.append(value)
    low, high, valid_count = _interval(replicates)
    return {
        "spearman_rho": rho if math.isfinite(rho) else None,
        "spearman_pvalue_descriptive": pvalue if math.isfinite(pvalue) else None,
        "spearman_exploratory_interval_low": low,
        "spearman_exploratory_interval_high": high,
        "valid_bootstrap_replicate_count": valid_count,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "outer_bootstrap_unit": "geometry_seed",
        "inner_bootstrap_unit": INNER_BLOCK_DEFINITION,
        "geometry_seed_count": GEOMETRY_SEED_COUNT,
        "development_only_not_confirmatory": True,
    }


__all__ = [
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "DEVELOPMENT_GEOMETRY_SEEDS",
    "DEVELOPMENT_MEASUREMENT_SEEDS",
    "DEVELOPMENT_REPEAT_INDICES",
    "EXPLORATORY_INTERVAL_QUANTILES",
    "GEOMETRY_SEED_COUNT",
    "INNER_BLOCK_DEFINITION",
    "QUANTILE_METHOD",
    "descriptive_development_summary",
    "hierarchical_bootstrap_indices",
    "hierarchical_bootstrap_interval",
    "paired_development_summary",
    "spearman_development_summary",
]
