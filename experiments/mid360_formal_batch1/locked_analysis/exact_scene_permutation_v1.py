"""Exact one-sided 20-allocation scene-level inference."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

import numpy as np

from .contract_v1 import RICH_SCENES, SCENE_ORDER, WEAK_SCENES


class ExactPermutationError(ValueError):
    """Raised when primary inference is called at the wrong analysis grain."""


def enumerate_allocations() -> list[tuple[str, str, str]]:
    allocations = [tuple(combo) for combo in combinations(SCENE_ORDER, 3)]
    if len(allocations) != 20 or len(set(allocations)) != 20:
        raise AssertionError("C(6,3) must produce exactly 20 unique allocations")
    if tuple(WEAK_SCENES) not in allocations:
        raise AssertionError("the observed Weak allocation must be enumerated")
    return allocations


def exact_weak_greater_than_rich(
    scene_summaries: Mapping[str, float | None],
) -> dict[str, Any]:
    if set(scene_summaries) != set(SCENE_ORDER) or len(scene_summaries) != 6:
        raise ExactPermutationError(
            "primary exact inference requires exactly six named scene summaries"
        )
    values: dict[str, float] = {}
    for scene in SCENE_ORDER:
        value = scene_summaries[scene]
        if value is None or isinstance(value, bool) or not np.isfinite(value):
            return {
                "status": "INFERENCE_UNDEFINED_INCOMPLETE_SIX_SCENE_COVERAGE",
                "estimand": None,
                "p_value": None,
                "allocation_count": 0,
                "p_value_denominator": None,
                "allocations": [],
            }
        values[scene] = float(value)
    observed = float(
        np.median([values[scene] for scene in WEAK_SCENES])
        - np.median([values[scene] for scene in RICH_SCENES])
    )
    rows: list[dict[str, Any]] = []
    extreme_count = 0
    all_scenes = set(SCENE_ORDER)
    for assigned_weak in enumerate_allocations():
        assigned_rich = tuple(scene for scene in SCENE_ORDER if scene not in assigned_weak)
        if len(set(assigned_weak)) != 3 or set(assigned_weak) | set(assigned_rich) != all_scenes:
            raise AssertionError("allocation must be a unique 3-versus-3 partition")
        statistic = float(
            np.median([values[scene] for scene in assigned_weak])
            - np.median([values[scene] for scene in assigned_rich])
        )
        extreme = statistic >= observed
        extreme_count += int(extreme)
        rows.append(
            {
                "assigned_weak_scene_ids": list(assigned_weak),
                "assigned_rich_scene_ids": list(assigned_rich),
                "statistic": statistic,
                "greater_than_or_equal_observed": extreme,
                "observed_allocation": tuple(assigned_weak) == tuple(WEAK_SCENES),
            }
        )
    p_value = extreme_count / 20.0
    if p_value <= 0.0 or p_value > 1.0 or not np.isclose(p_value * 20.0, round(p_value * 20.0)):
        raise AssertionError("exact p-value must be a positive 1/20 increment")
    return {
        "status": "EXACT_ONE_SIDED_20_ALLOCATION_SCENE_INFERENCE_DEFINED",
        "alternative": "WEAK_GREATER_THAN_RICH",
        "estimand": observed,
        "p_value": p_value,
        "allocation_count": 20,
        "p_value_denominator": 20,
        "plus_one_correction": False,
        "allocations": rows,
    }
