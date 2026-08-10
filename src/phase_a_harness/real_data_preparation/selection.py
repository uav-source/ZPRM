"""Registration-blind geometry-only interval ranking and snapshot selection."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


GEOMETRY_ONLY_FIELDS = (
    "initial_correspondence_count",
    "initial_valid_normal_correspondence_count",
    "lambda_min_trans",
    "lambda_mid_trans",
    "lambda_max_trans",
    "normalized_lambda_min_trans",
    "normalized_lambda_mid_trans",
    "normalized_lambda_max_trans",
    "condition_number_trans",
    "spectral_entropy_trans",
)

FORBIDDEN_SELECTOR_FRAGMENTS = (
    "residual",
    "error",
    "final",
    "turnover",
    "fitness",
    "rmse",
    "translation_displacement",
    "rotation_displacement",
    "solver",
    "convergence",
)


def geometry_only_view(record: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = sorted(
        key for key in record if any(fragment in key.lower() for fragment in FORBIDDEN_SELECTOR_FRAGMENTS)
    )
    if forbidden:
        raise ValueError(f"registration-derived fields are forbidden: {forbidden}")
    missing = [key for key in GEOMETRY_ONLY_FIELDS if key not in record]
    if missing:
        raise ValueError(f"missing geometry-only fields: {missing}")
    result = {key: record[key] for key in GEOMETRY_ONLY_FIELDS}
    if not all(np.isfinite(float(value)) for value in result.values()):
        raise ValueError("geometry-only metrics must be finite")
    return result


def candidate_scan_valid(row: Mapping[str, Any]) -> tuple[bool, str]:
    source_count = int(row["finite_source_point_count"])
    target_count = int(row["target_map_point_count"])
    valid_count = int(row["initial_valid_normal_correspondence_count"])
    if source_count < 1000:
        return False, "FINITE_SOURCE_POINT_COUNT_LT_1000"
    if target_count < 10000:
        return False, "TARGET_MAP_POINT_COUNT_LT_10000"
    if valid_count < max(100, math.ceil(0.05 * source_count)):
        return False, "INSUFFICIENT_VALID_NORMAL_CORRESPONDENCES"
    for field, reason in (
        ("reference_interpolation_valid", "REFERENCE_INTERPOLATION_INVALID"),
        ("gt_gap_within_limit", "GT_GAP_EXCEEDS_LIMIT"),
        ("target_map_coverage_valid", "TARGET_MAP_COVERAGE_INVALID"),
        ("deskew_uncertainty_within_limit", "DESKEW_UNCERTAINTY_EXCEEDS_LIMIT"),
    ):
        if row.get(field) is not True:
            return False, reason
    return True, ""


def score_interval(scans: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not scans:
        raise ValueError("interval has no scans")
    views = [geometry_only_view(row) for row in scans]
    return {
        "normalized_lambda_min_trans": float(
            np.median([row["normalized_lambda_min_trans"] for row in views])
        ),
        "condition_number_trans": float(
            np.median([row["condition_number_trans"] for row in views])
        ),
        "spectral_entropy_trans": float(
            np.median([row["spectral_entropy_trans"] for row in views])
        ),
    }


def _rank_key(row: Mapping[str, Any], label: str) -> tuple[Any, ...]:
    primary = float(row["normalized_lambda_min_trans"])
    condition = float(row["condition_number_trans"])
    entropy = float(row["spectral_entropy_trans"])
    start = float(row["interval_start_time"])
    if label == "CORRIDOR_OR_WEAK_GEOMETRY":
        return (primary, -condition, entropy, start, str(row["interval_id"]))
    if label == "GEOMETRY_RICH":
        return (-primary, condition, -entropy, start, str(row["interval_id"]))
    raise ValueError(f"unsupported label: {label}")


def _compatible(candidate: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> bool:
    start = float(candidate["interval_start_time"])
    end = float(candidate["interval_end_time"])
    center = (start + end) / 2.0
    position = np.asarray(candidate["center_world_position"], dtype=np.float64)
    for prior in selected:
        prior_start = float(prior["interval_start_time"])
        prior_end = float(prior["interval_end_time"])
        if max(start, prior_start) < min(end, prior_end):
            return False
        prior_center = (prior_start + prior_end) / 2.0
        prior_position = np.asarray(prior["center_world_position"], dtype=np.float64)
        if abs(center - prior_center) < 10.0 and np.linalg.norm(position - prior_position) < 1.0:
            return False
    return True


def select_scene_intervals(
    candidates: Iterable[Mapping[str, Any]], *, per_label: int = 10
) -> dict[str, list[dict[str, Any]]]:
    rows = [dict(row) for row in candidates]
    result: dict[str, list[dict[str, Any]]] = {}
    occupied: list[Mapping[str, Any]] = []
    for label in ("CORRIDOR_OR_WEAK_GEOMETRY", "GEOMETRY_RICH"):
        ranked = sorted(rows, key=lambda row: _rank_key(row, label))
        selected: list[dict[str, Any]] = []
        for row in ranked:
            if _compatible(row, occupied + selected):
                selected.append({**row, "scene_label": label})
            if len(selected) == per_label:
                break
        if len(selected) != per_label:
            raise ValueError(f"R06 insufficient {label} intervals: {len(selected)} != {per_label}")
        result[label] = selected
        occupied.extend(selected)
    return result


def select_five_scan_timestamps(timestamps: Sequence[float]) -> list[float]:
    values = sorted(set(float(value) for value in timestamps))
    if len(values) < 5:
        raise ValueError("at least five unique valid scan timestamps are required")
    chosen: list[float] = []
    array = np.asarray(values, dtype=np.float64)
    for quantile in (0.10, 0.30, 0.50, 0.70, 0.90):
        target = float(np.quantile(array, quantile, method="linear"))
        available = [value for value in values if value not in chosen]
        selected = min(available, key=lambda value: (abs(value - target), value))
        chosen.append(selected)
    return chosen


def assert_map_query_disjoint(map_source_ids: Iterable[str], query_source_ids: Iterable[str]) -> None:
    overlap = sorted(set(map_source_ids) & set(query_source_ids))
    if overlap:
        raise ValueError(f"query scans contaminate target map: {overlap[:10]}")
