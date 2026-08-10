"""Deterministic GT-only overlap calculation in an already common world frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class OverlapContract:
    resample_rate_hz: float = 1.0
    radius_m: float = 5.0
    min_contiguous_duration_s: float = 5.0
    min_total_covered_duration_s: float = 150.0
    min_coverage_fraction: float = 0.60
    min_nonoverlapping_5s_intervals: int = 30


def _validate_trajectory(trajectory: np.ndarray) -> np.ndarray:
    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError("trajectory must have columns [timestamp,x,y,z]")
    if value.shape[0] < 2 or not np.isfinite(value).all():
        raise ValueError("trajectory must contain at least two finite rows")
    if not np.all(np.diff(value[:, 0]) > 0.0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    return value


def resample_positions(trajectory: np.ndarray, rate_hz: float) -> np.ndarray:
    value = _validate_trajectory(trajectory)
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    step = 1.0 / rate_hz
    timestamps = np.arange(value[0, 0], value[-1, 0] + step * 1e-9, step, dtype=np.float64)
    positions = np.column_stack(
        [np.interp(timestamps, value[:, 0], value[:, index]) for index in (1, 2, 3)]
    )
    return np.column_stack((timestamps, positions))


def _covered_intervals(timestamps: np.ndarray, covered: np.ndarray, step: float) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    start: int | None = None
    for index, flag in enumerate(np.append(covered, False)):
        if bool(flag) and start is None:
            start = index
        elif not bool(flag) and start is not None:
            stop = index - 1
            intervals.append(
                {
                    "start_time": float(timestamps[start]),
                    "end_time": float(timestamps[stop] + step),
                    "duration_s": float((stop - start + 1) * step),
                    "sample_count": int(stop - start + 1),
                }
            )
            start = None
    return intervals


def compute_gt_only_overlap(
    map_trajectory: np.ndarray,
    query_trajectory: np.ndarray,
    *,
    common_world_frame_proven: bool,
    contract: OverlapContract = OverlapContract(),
) -> dict[str, Any]:
    if common_world_frame_proven is not True:
        return {
            "eligibility_status": "FAIL",
            "failure_reason": "UNPROVEN_CROSS_SEQUENCE_WORLD_FRAME",
            "overlap_status": "NOT_COMPUTABLE",
        }
    map_resampled = resample_positions(map_trajectory, contract.resample_rate_hz)
    query_resampled = resample_positions(query_trajectory, contract.resample_rate_hz)
    distances, _ = cKDTree(map_resampled[:, 1:4]).query(query_resampled[:, 1:4], k=1)
    covered = distances <= contract.radius_m
    step = 1.0 / contract.resample_rate_hz
    intervals = _covered_intervals(query_resampled[:, 0], covered, step)
    eligible_intervals = [
        row for row in intervals if row["duration_s"] >= contract.min_contiguous_duration_s
    ]
    covered_count = int(covered.sum())
    query_count = int(covered.size)
    total_duration = float(covered_count * step)
    coverage_fraction = float(covered_count / query_count)
    nonoverlapping_count = int(
        sum(int(row["duration_s"] // 5.0) for row in eligible_intervals)
    )
    passed = (
        total_duration >= contract.min_total_covered_duration_s
        and coverage_fraction >= contract.min_coverage_fraction
        and nonoverlapping_count >= contract.min_nonoverlapping_5s_intervals
    )
    return {
        "contract": contract.__dict__,
        "covered_query_count": covered_count,
        "query_count": query_count,
        "coverage_fraction": coverage_fraction,
        "total_covered_duration_s": total_duration,
        "contiguous_covered_intervals": intervals,
        "eligible_nonoverlapping_5s_interval_count": nonoverlapping_count,
        "nearest_distance_median_m": float(np.median(distances)),
        "nearest_distance_q95_m": float(np.quantile(distances, 0.95)),
        "nearest_distance_max_m": float(np.max(distances)),
        "overlap_status": "PASS" if passed else "FAIL",
        "eligibility_status": "PASS" if passed else "FAIL",
    }
