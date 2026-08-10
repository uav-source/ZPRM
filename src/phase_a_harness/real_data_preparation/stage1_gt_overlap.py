"""Dataset-neutral, gap-aware GT-only Stage-1 overlap primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Stage1OverlapContract:
    resample_rate_hz: float = 1.0
    radius_m: float = 5.0
    min_contiguous_covered_duration_s: float = 5.0
    min_total_covered_duration_s: float = 150.0
    min_coverage_fraction: float = 0.60
    min_eligible_nonoverlapping_5s_intervals: int = 30
    maximum_native_gap_s: float = 0.2


FROZEN_STAGE1_OVERLAP_CONTRACT = Stage1OverlapContract()


def _validate_trajectory(trajectory: np.ndarray) -> np.ndarray:
    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4 or value.shape[0] < 2:
        raise ValueError("trajectory must contain at least two [timestamp,x,y,z] rows")
    if not np.isfinite(value).all():
        raise ValueError("trajectory contains non-finite values")
    if not np.all(np.diff(value[:, 0]) > 0.0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    return value


def resample_gap_aware(
    trajectory: np.ndarray,
    contract: Stage1OverlapContract = FROZEN_STAGE1_OVERLAP_CONTRACT,
) -> np.ndarray:
    value = _validate_trajectory(trajectory)
    if contract.resample_rate_hz <= 0.0 or contract.maximum_native_gap_s <= 0.0:
        raise ValueError("resampling contract must be positive")
    breaks = np.flatnonzero(np.diff(value[:, 0]) > contract.maximum_native_gap_s) + 1
    step = 1.0 / contract.resample_rate_hz
    output: list[np.ndarray] = []
    output_segment_id = 0
    for segment in np.split(value, breaks):
        if segment.shape[0] < 2 or segment[-1, 0] - segment[0, 0] + 1e-12 < step:
            continue
        timestamps = np.arange(segment[0, 0], segment[-1, 0] + step * 1e-9, step)
        positions = np.column_stack(
            [np.interp(timestamps, segment[:, 0], segment[:, column]) for column in (1, 2, 3)]
        )
        output.append(
            np.column_stack(
                (timestamps, positions, np.full(timestamps.size, output_segment_id, dtype=np.float64))
            )
        )
        output_segment_id += 1
    if not output:
        raise ValueError("no continuous segment is long enough to resample")
    return np.vstack(output)


def compute_stage1_gt_overlap(
    map_trajectory: np.ndarray,
    query_trajectory: np.ndarray,
    *,
    common_world_frame_proven: bool,
    contract: Stage1OverlapContract = FROZEN_STAGE1_OVERLAP_CONTRACT,
) -> dict[str, Any]:
    if common_world_frame_proven is not True:
        return {
            "failure_reason": "UNPROVEN_CROSS_SEQUENCE_FIXED_WORLD_FRAME",
            "overlap_status": "NOT_COMPUTABLE",
        }
    map_rows = resample_gap_aware(map_trajectory, contract)
    query_rows = resample_gap_aware(query_trajectory, contract)
    distances, _ = cKDTree(map_rows[:, 1:4]).query(query_rows[:, 1:4], k=1)
    covered = distances <= contract.radius_m
    step = 1.0 / contract.resample_rate_hz
    intervals: list[dict[str, Any]] = []
    eligible_nonoverlapping = 0
    for segment_id in np.unique(query_rows[:, 4]).astype(int):
        indices = np.flatnonzero(query_rows[:, 4] == segment_id)
        flags = covered[indices]
        start: int | None = None
        for local_index, flag in enumerate(np.append(flags, False)):
            if bool(flag) and start is None:
                start = local_index
            elif not bool(flag) and start is not None:
                stop = local_index - 1
                sample_count = stop - start + 1
                duration = sample_count * step
                row = {
                    "duration_s": float(duration),
                    "end_time": float(query_rows[indices[stop], 0] + step),
                    "sample_count": int(sample_count),
                    "segment_id": int(segment_id),
                    "start_time": float(query_rows[indices[start], 0]),
                }
                intervals.append(row)
                if duration >= contract.min_contiguous_covered_duration_s:
                    eligible_nonoverlapping += int(duration // 5.0)
                start = None
    covered_count = int(np.sum(covered))
    query_count = int(covered.size)
    total_duration = covered_count * step
    coverage_fraction = covered_count / query_count
    passed = (
        total_duration >= contract.min_total_covered_duration_s
        and coverage_fraction >= contract.min_coverage_fraction
        and eligible_nonoverlapping >= contract.min_eligible_nonoverlapping_5s_intervals
    )
    return {
        "contract": asdict(contract),
        "contiguous_covered_intervals": intervals,
        "coverage_fraction": float(coverage_fraction),
        "covered_query_count": covered_count,
        "eligible_nonoverlapping_5s_interval_count": int(eligible_nonoverlapping),
        "map_resampled_count": int(map_rows.shape[0]),
        "nearest_distance_max_m": float(np.max(distances)),
        "nearest_distance_median_m": float(np.median(distances)),
        "nearest_distance_q95_m": float(np.quantile(distances, 0.95)),
        "overlap_status": "PASS" if passed else "FAIL",
        "query_count": query_count,
        "query_resampled_segment_count": int(np.unique(query_rows[:, 4]).size),
        "total_covered_duration_s": float(total_duration),
    }


def rank_distinct_stage1_pairs(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if row.get("map_sequence_id") == row.get("query_sequence_id"):
            raise ValueError("same-sequence map/query pair is prohibited")
        if row.get("overlap_status") != "PASS":
            continue
        if not all(
            row.get(key) is True
            for key in (
                "map_independent_6dof",
                "query_independent_6dof",
                "common_world_frame_proven",
                "map_rig_lidar_transform_available",
                "query_rig_lidar_transform_available",
            )
        ):
            continue
        eligible.append(row)
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row["total_covered_duration_s"]),
            -float(row["coverage_fraction"]),
            -int(row["eligible_nonoverlapping_5s_interval_count"]),
            float(row["nearest_distance_q95_m"]),
            str(row["map_sequence_id"]),
            str(row["query_sequence_id"]),
        ),
    )
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    return ranked
