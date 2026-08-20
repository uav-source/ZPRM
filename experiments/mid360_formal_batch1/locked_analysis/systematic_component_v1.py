"""Frozen systematic-fraction summaries and descriptive Weak/Rich comparison."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contract_v1 import RICH_SCENES, SCENE_ORDER, WEAK_SCENES


class SystematicComponentError(ValueError):
    """Raised when systematic component input has the wrong grain."""


def station_systematic_fraction(
    vectors: Sequence[Sequence[float] | None],
) -> dict[str, Any]:
    if len(vectors) != 10:
        raise SystematicComponentError("station systematic fraction requires 10 planned vectors")
    finite_vectors: list[np.ndarray] = []
    for vector in vectors:
        if vector is None:
            continue
        array = np.asarray(vector, dtype=np.float64)
        if array.shape != (3,) or not np.isfinite(array).all():
            continue
        finite_vectors.append(array)
    if len(finite_vectors) != 10:
        return {
            "status": "SYSTEMATIC_FRACTION_UNDEFINED_INCOMPLETE_10_VECTORS",
            "planned_n": 10,
            "finite_vector_n": len(finite_vectors),
            "systematic_fraction": None,
            "mean_vector": None,
            "denominator": None,
        }
    array = np.stack(finite_vectors)
    mean_vector = np.mean(array, axis=0)
    denominator = float(np.mean(np.linalg.norm(array, axis=1)))
    if denominator == 0.0:
        return {
            "status": "ALL_ZERO_UPDATES_DEFINED_ZERO_SYSTEMATIC_FRACTION",
            "planned_n": 10,
            "finite_vector_n": 10,
            "systematic_fraction": 0.0,
            "mean_vector": [0.0, 0.0, 0.0],
            "denominator": 0.0,
        }
    return {
        "status": "SYSTEMATIC_FRACTION_DEFINED_COMPLETE_10_VECTORS",
        "planned_n": 10,
        "finite_vector_n": 10,
        "systematic_fraction": float(np.linalg.norm(mean_vector) / denominator),
        "mean_vector": [float(value) for value in mean_vector],
        "denominator": denominator,
    }


def scene_systematic_fraction(
    station_fractions: Mapping[str, float | None],
) -> dict[str, Any]:
    if set(station_fractions) != {"S01", "S02", "S03"}:
        raise SystematicComponentError("scene systematic fraction requires S01/S02/S03")
    values = [station_fractions[station] for station in ("S01", "S02", "S03")]
    if any(value is None or not np.isfinite(value) for value in values):
        return {
            "status": "SCENE_SYSTEMATIC_FRACTION_UNDEFINED_INCOMPLETE_3_STATIONS",
            "defined_station_n": sum(
                value is not None and np.isfinite(value) for value in values
            ),
            "scene_systematic_fraction": None,
        }
    return {
        "status": "SCENE_SYSTEMATIC_FRACTION_DEFINED_COMPLETE_3_STATIONS",
        "defined_station_n": 3,
        "scene_systematic_fraction": float(np.median(values)),
    }


def systematic_weak_rich_descriptive(
    scene_fractions: Mapping[str, float | None],
) -> dict[str, Any]:
    if set(scene_fractions) != set(SCENE_ORDER):
        raise SystematicComponentError("systematic comparison requires exactly six scenes")
    if any(value is None or not np.isfinite(value) for value in scene_fractions.values()):
        return {
            "status": "SYSTEMATIC_WEAK_RICH_UNDEFINED_INCOMPLETE_6_SCENES",
            "role": "SECONDARY_DESCRIPTIVE_MECHANISTIC_COMPARISON",
            "weak_minus_rich_median_difference": None,
            "formal_p_value_defined": False,
        }
    rich = [float(scene_fractions[scene]) for scene in RICH_SCENES]
    weak = [float(scene_fractions[scene]) for scene in WEAK_SCENES]
    rich_median = float(np.median(rich))
    weak_median = float(np.median(weak))
    return {
        "status": "SYSTEMATIC_WEAK_RICH_DESCRIPTIVE_DEFINED_6_SCENES",
        "role": "SECONDARY_DESCRIPTIVE_MECHANISTIC_COMPARISON",
        "three_rich_scene_values": rich,
        "three_weak_scene_values": weak,
        "rich_median": rich_median,
        "weak_median": weak_median,
        "weak_minus_rich_median_difference": weak_median - rich_median,
        "formal_hypothesis_test": False,
        "formal_p_value_defined": False,
        "directional_pass_fail": False,
    }
