"""C1+C2 formal reassociation and centered sensitivity analysis."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .contract_v1 import (
    CENTERED_PERMUTATION_DRAW_COUNT,
    CENTERED_PERMUTATION_P_DENOMINATOR,
    CENTERED_PERMUTATION_SEED,
    SCENE_ORDER,
)
from .cross_backend_v1 import formal_spearman


FORMAL_TURNOVER_FIELD = "correspondence_turnover"
ACCEPTED_SOURCE_TURNOVER_ROLE = "SECONDARY_DESCRIPTIVE_COMPANION"


class ReassociationError(ValueError):
    """Raised when reassociation input violates C1+C2."""


def formal_scene_turnover(values: Sequence[float | None]) -> dict[str, Any]:
    if len(values) != 30:
        raise ReassociationError("formal scene turnover requires exactly 30 planned values")
    if any(
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        for value in values
    ):
        return {
            "status": "REASSOCIATION_SCENE_SUMMARY_UNDEFINED_INCOMPLETE_30_METRICS",
            "formal_turnover_field": FORMAL_TURNOVER_FIELD,
            "finite_metric_n": sum(
                value is not None
                and not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value)
                for value in values
            ),
            "median": None,
        }
    return {
        "status": "REASSOCIATION_SCENE_SUMMARY_DEFINED_COMPLETE_30_METRICS",
        "formal_turnover_field": FORMAL_TURNOVER_FIELD,
        "finite_metric_n": 30,
        "median": float(np.median(np.asarray(values, dtype=np.float64))),
    }


def scene_turnover_translation_spearman(
    scene_turnover_medians: Mapping[str, float | None],
    scene_translation_medians: Mapping[str, float | None],
) -> dict[str, Any]:
    result = formal_spearman(
        scene_turnover_medians,
        scene_translation_medians,
        ordered_ids=SCENE_ORDER,
        incomplete_status=(
            "REASSOCIATION_SCENE_ASSOCIATION_UNDEFINED_INCOMPLETE_6_PAIRS"
        ),
    )
    result["formal_turnover_field"] = FORMAL_TURNOVER_FIELD
    return result


def _complete_scene_arrays(
    values: Mapping[str, Sequence[float | None]], label: str
) -> dict[str, np.ndarray] | None:
    if set(values) != set(SCENE_ORDER):
        raise ReassociationError(f"{label} must contain exactly six frozen scenes")
    arrays: dict[str, np.ndarray] = {}
    for scene in SCENE_ORDER:
        row = values[scene]
        if len(row) != 30:
            return None
        if any(
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            for value in row
        ):
            return None
        arrays[scene] = np.asarray(row, dtype=np.float64)
    return arrays


def centered_association(
    turnover_by_scene: Mapping[str, Sequence[float | None]],
    translation_by_scene: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    turnover = _complete_scene_arrays(turnover_by_scene, "turnover")
    translation = _complete_scene_arrays(translation_by_scene, "translation")
    if turnover is None or translation is None:
        return {
            "status": "CENTERED_ASSOCIATION_UNDEFINED_INCOMPLETE_180_PAIRS",
            "rho": None,
            "formal_asymptotic_p_value": None,
            "complete_pair_n": 0,
            "centered_rows": [],
        }
    centered_rows: list[dict[str, Any]] = []
    x_parts = []
    y_parts = []
    x_by_scene: dict[str, np.ndarray] = {}
    y_by_scene: dict[str, np.ndarray] = {}
    for scene in SCENE_ORDER:
        x_centered = turnover[scene] - np.median(turnover[scene])
        y_centered = translation[scene] - np.median(translation[scene])
        x_by_scene[scene] = x_centered
        y_by_scene[scene] = y_centered
        x_parts.append(x_centered)
        y_parts.append(y_centered)
        centered_rows.extend(
            {
                "scene_id": scene,
                "scene_row_index": index,
                "centered_correspondence_turnover": float(x_value),
                "centered_translation_norm_m": float(y_value),
            }
            for index, (x_value, y_value) in enumerate(
                zip(x_centered, y_centered, strict=True), start=1
            )
        )
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return {
            "status": "CENTERED_SPEARMAN_UNDEFINED_CONSTANT_INPUT",
            "rho": None,
            "formal_asymptotic_p_value": None,
            "complete_pair_n": 180,
            "centered_rows": centered_rows,
            "centered_turnover_by_scene": x_by_scene,
            "centered_translation_by_scene": y_by_scene,
        }
    result = spearmanr(x, y)
    return {
        "status": "CENTERED_ASSOCIATION_DEFINED_COMPLETE_180_PAIRS",
        "rho": float(result.statistic),
        "formal_asymptotic_p_value": None,
        "complete_pair_n": 180,
        "both_variables_median_centered": True,
        "centered_rows": centered_rows,
        "centered_turnover_by_scene": x_by_scene,
        "centered_translation_by_scene": y_by_scene,
    }


def centered_permutation_sensitivity(
    centered: Mapping[str, Any],
    *,
    backend: str,
) -> dict[str, Any]:
    if centered.get("status") != "CENTERED_ASSOCIATION_DEFINED_COMPLETE_180_PAIRS":
        return {
            "status": "CENTERED_PERMUTATION_NOT_RUN_FORMAL_CENTERED_ASSOCIATION_UNDEFINED",
            "backend": backend,
            "draw_count": 0,
            "p_value": None,
            "draws": [],
        }
    rho_observed = float(centered["rho"])
    x_by_scene = centered["centered_turnover_by_scene"]
    y_by_scene = centered["centered_translation_by_scene"]
    x = np.concatenate([x_by_scene[scene] for scene in SCENE_ORDER])
    rng = np.random.Generator(np.random.PCG64(CENTERED_PERMUTATION_SEED))
    draws: list[dict[str, Any]] = []
    extreme_count = 0
    for draw_index in range(1, CENTERED_PERMUTATION_DRAW_COUNT + 1):
        y_permuted = np.concatenate(
            [rng.permutation(y_by_scene[scene]) for scene in SCENE_ORDER]
        )
        rho_permuted = float(spearmanr(x, y_permuted).statistic)
        extreme = abs(rho_permuted) >= abs(rho_observed)
        extreme_count += int(extreme)
        draws.append(
            {
                "backend": backend,
                "draw_index": draw_index,
                "rho_permuted": rho_permuted,
                "abs_rho_permuted": abs(rho_permuted),
                "greater_than_or_equal_abs_observed": extreme,
            }
        )
    return {
        "status": "SECONDARY_STRATIFIED_MONTE_CARLO_PERMUTATION_SENSITIVITY",
        "backend": backend,
        "observed_statistic": abs(rho_observed),
        "draw_count": CENTERED_PERMUTATION_DRAW_COUNT,
        "seed": CENTERED_PERMUTATION_SEED,
        "rng": "numpy.random.Generator(numpy.random.PCG64(20260820))",
        "p_value_denominator": CENTERED_PERMUTATION_P_DENOMINATOR,
        "extreme_count": extreme_count,
        "p_value": (1 + extreme_count) / CENTERED_PERMUTATION_P_DENOMINATOR,
        "scene_strata_preserved": True,
        "draws": draws,
    }


def accepted_source_turnover_descriptive(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {
        "role": ACCEPTED_SOURCE_TURNOVER_ROLE,
        "planned_n": len(values),
        "finite_n": len(finite),
        "median": float(np.median(finite)) if finite else None,
        "may_replace_formal_turnover": False,
    }
