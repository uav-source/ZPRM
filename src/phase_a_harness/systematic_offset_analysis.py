"""Systematic-offset and repeatability summaries for ten-observation groups."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


ZERO_VECTOR_EPSILON = 1.0e-12
EXPECTED_OBSERVATIONS_PER_GROUP = 10
DEFAULT_GROUP_FIELDS = (
    "scene_variant",
    "geometry_seed",
    "condition",
    "backend",
)


def _vectors(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 2:
        raise ValueError(f"{label} must contain at least two finite 3-vectors")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return np.ascontiguousarray(array)


def translation_direction_concentration(
    translation_vectors: np.ndarray,
) -> tuple[float | None, int]:
    """Mean-resultant length of nonzero translation directions."""

    vectors = np.asarray(translation_vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.all(np.isfinite(vectors)):
        raise ValueError("translation_vectors must be a finite Nx3 array")
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > ZERO_VECTOR_EPSILON
    count = int(np.count_nonzero(valid))
    if count == 0:
        return None, 0
    unit = vectors[valid] / norms[valid, None]
    concentration = float(np.linalg.norm(np.sum(unit, axis=0)) / count)
    return concentration, count


def summarize_systematic_offset(
    translation_vectors: np.ndarray,
    rotation_vectors: np.ndarray,
) -> dict[str, Any]:
    """Apply the frozen mean-vector and ddof=1 covariance definitions."""

    translation = _vectors(translation_vectors, "translation_vectors")
    rotation = _vectors(rotation_vectors, "rotation_vectors")
    if translation.shape[0] != rotation.shape[0]:
        raise ValueError("translation and rotation observation counts disagree")

    mean_translation = np.mean(translation, axis=0)
    mean_rotation = np.mean(rotation, axis=0)
    translation_covariance = np.cov(translation, rowvar=False, ddof=1)
    rotation_covariance = np.cov(rotation, rowvar=False, ddof=1)
    translation_offset = float(np.linalg.norm(mean_translation))
    rotation_offset = float(np.linalg.norm(mean_rotation))
    mean_translation_norm = float(np.mean(np.linalg.norm(translation, axis=1)))
    systematic_fraction = (
        None
        if mean_translation_norm <= ZERO_VECTOR_EPSILON
        else float(translation_offset / mean_translation_norm)
    )
    concentration, direction_count = translation_direction_concentration(translation)
    translation_rms = float(
        math.sqrt(max(float(np.trace(translation_covariance)), 0.0))
    )
    rotation_rms = float(math.sqrt(max(float(np.trace(rotation_covariance)), 0.0)))
    return {
        "systematic_offset_analysis_valid": True,
        "systematic_offset_invalid_reason": None,
        "observation_count": int(translation.shape[0]),
        "expected_observation_count": EXPECTED_OBSERVATIONS_PER_GROUP,
        "observation_count_contract_pass": bool(
            translation.shape[0] == EXPECTED_OBSERVATIONS_PER_GROUP
        ),
        "mean_translation_vector": mean_translation.tolist(),
        "systematic_translation_offset_m": translation_offset,
        "translation_repeatability_covariance": translation_covariance.tolist(),
        "translation_repeatability_covariance_ddof": 1,
        "translation_repeatability_rms_m": translation_rms,
        "mean_rotation_vector": mean_rotation.tolist(),
        "systematic_rotation_offset_rad": rotation_offset,
        "rotation_repeatability_covariance": rotation_covariance.tolist(),
        "rotation_repeatability_covariance_ddof": 1,
        "rotation_repeatability_rms_rad": rotation_rms,
        "mean_translation_norm_m": mean_translation_norm,
        "systematic_fraction_translation": systematic_fraction,
        "translation_direction_concentration": concentration,
        "translation_direction_valid_nonzero_count": direction_count,
        "single_trial_bias_claim_authorized": False,
    }


def _invalid_group_summary(observation_count: int, reason: str) -> dict[str, Any]:
    return {
        "systematic_offset_analysis_valid": False,
        "systematic_offset_invalid_reason": reason,
        "observation_count": int(observation_count),
        "expected_observation_count": EXPECTED_OBSERVATIONS_PER_GROUP,
        "observation_count_contract_pass": False,
        "mean_translation_vector": None,
        "systematic_translation_offset_m": None,
        "translation_repeatability_covariance": None,
        "translation_repeatability_covariance_ddof": 1,
        "translation_repeatability_rms_m": None,
        "mean_rotation_vector": None,
        "systematic_rotation_offset_rad": None,
        "rotation_repeatability_covariance": None,
        "rotation_repeatability_covariance_ddof": 1,
        "rotation_repeatability_rms_rad": None,
        "mean_translation_norm_m": None,
        "systematic_fraction_translation": None,
        "translation_direction_concentration": None,
        "translation_direction_valid_nonzero_count": 0,
        "single_trial_bias_claim_authorized": False,
    }


def summarize_systematic_offset_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = DEFAULT_GROUP_FIELDS,
    translation_field: str = "translation_vector",
    rotation_field: str = "rotation_vector",
) -> list[dict[str, Any]]:
    """Group ordinary records without silently dropping incomplete groups."""

    if not rows:
        return []
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        missing = [name for name in (*group_fields, translation_field, rotation_field) if name not in row]
        if missing:
            raise ValueError(f"systematic-offset row is missing fields: {missing}")
        groups[tuple(row[name] for name in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for identity in sorted(groups, key=lambda value: tuple(str(item) for item in value)):
        selected = groups[identity]
        if len(selected) < 2:
            summary = _invalid_group_summary(
                len(selected), "INSUFFICIENT_OBSERVATIONS_FOR_DDOF1"
            )
        else:
            try:
                summary = summarize_systematic_offset(
                    np.asarray(
                        [row[translation_field] for row in selected], dtype=np.float64
                    ),
                    np.asarray(
                        [row[rotation_field] for row in selected], dtype=np.float64
                    ),
                )
            except (TypeError, ValueError, FloatingPointError) as error:
                summary = _invalid_group_summary(
                    len(selected), f"INVALID_VECTOR_PAYLOAD: {type(error).__name__}"
                )
        output.append(
            {
                **dict(zip(group_fields, identity)),
                **summary,
            }
        )
    return output


def systematic_offset_claim_gate(
    group_summaries: Sequence[Mapping[str, Any]],
    *,
    scene_variant: str = "LONG_CORRIDOR",
    conditions: Sequence[str] = ("INDEPENDENT_NOISE_FREE", "FULL_NOISE"),
    backends: Sequence[str] = ("open3d_point_to_plane", "pcl_point_to_plane"),
    backend_field: str = "backend",
) -> dict[str, Any]:
    """Evaluate the frozen 2/3 geometry and median-fraction wording gate."""

    evaluations: list[dict[str, Any]] = []
    for backend in backends:
        for condition in conditions:
            selected = [
                row
                for row in group_summaries
                if row.get("scene_variant") == scene_variant
                and row.get(backend_field) == backend
                and row.get("condition") == condition
            ]
            fractions = [row.get("systematic_fraction_translation") for row in selected]
            qualifying = sum(
                row.get("systematic_offset_analysis_valid") is True
                and row.get("observation_count_contract_pass") is True
                and row.get("observation_count") == EXPECTED_OBSERVATIONS_PER_GROUP
                and row.get("systematic_translation_offset_m") is not None
                and float(row["systematic_translation_offset_m"]) >= 0.005
                and row.get("systematic_fraction_translation") is not None
                and float(row["systematic_fraction_translation"]) >= 0.60
                for row in selected
            )
            median_fraction = (
                None
                if len(selected) != 3 or any(value is None for value in fractions)
                else float(np.median(np.asarray(fractions, dtype=np.float64)))
            )
            passed = bool(
                len(selected) == 3
                and all(
                    row.get("systematic_offset_analysis_valid") is True
                    and row.get("observation_count_contract_pass") is True
                    and row.get("observation_count")
                    == EXPECTED_OBSERVATIONS_PER_GROUP
                    for row in selected
                )
                and qualifying >= 2
                and median_fraction is not None
                and median_fraction >= 0.70
            )
            evaluations.append(
                {
                    "backend": backend,
                    "condition": condition,
                    "geometry_group_count": len(selected),
                    "qualifying_geometry_group_count": qualifying,
                    "median_systematic_fraction_translation": median_fraction,
                    "pass": passed,
                }
            )
    return {
        "SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED": bool(
            len(evaluations) == len(backends) * len(conditions)
            and all(row["pass"] for row in evaluations)
        ),
        "evaluations": evaluations,
    }


__all__ = [
    "DEFAULT_GROUP_FIELDS",
    "EXPECTED_OBSERVATIONS_PER_GROUP",
    "ZERO_VECTOR_EPSILON",
    "summarize_systematic_offset",
    "summarize_systematic_offset_groups",
    "systematic_offset_claim_gate",
    "translation_direction_concentration",
]
