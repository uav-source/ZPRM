"""Fixture-only semantic definitions frozen by the result-blind C2 clarification.

This module is deliberately not a formal-results loader or a complete locked-analysis
implementation.  It exposes small, pure functions so the five clarified definitions
can be qualified on hand-built synthetic fixtures before any scientific analysis.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


SCENE_ORDER = (
    "FMB1_R01",
    "FMB1_R02",
    "FMB1_R03",
    "FMB1_W01",
    "FMB1_W02",
    "FMB1_W03",
)
RICH_SCENES = SCENE_ORDER[:3]
WEAK_SCENES = SCENE_ORDER[3:]
FORMAL_TURNOVER_FIELD = "correspondence_turnover"
ACCEPTED_SOURCE_TURNOVER_ROLE = "SECONDARY_DESCRIPTIVE_COMPANION"
CENTERED_PERMUTATION_SEED = 20260820
CENTERED_PERMUTATION_COUNT = 10000
CENTERED_PERMUTATION_P_DENOMINATOR = 10001


class C2SemanticError(ValueError):
    """Raised when a fixture violates a frozen C2 semantic precondition."""


def _exact_scene_mapping(values: Mapping[str, float]) -> dict[str, float]:
    if set(values) != set(SCENE_ORDER):
        raise C2SemanticError("fixture must contain exactly the six frozen scene IDs")
    result = {scene: float(values[scene]) for scene in SCENE_ORDER}
    if not all(np.isfinite(value) for value in result.values()):
        raise C2SemanticError("scene values must be finite")
    return result


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def scene_ordering_agreement(
    open3d_scene_medians: Mapping[str, float],
    pcl_scene_medians: Mapping[str, float],
) -> dict[str, object]:
    """Return the frozen 15-pair descriptive scene-ordering agreement."""

    open3d = _exact_scene_mapping(open3d_scene_medians)
    pcl = _exact_scene_mapping(pcl_scene_medians)
    counts = {
        "CONCORDANT_NON_TIE": 0,
        "DISCORDANT": 0,
        "BOTH_TIED": 0,
        "ONE_BACKEND_TIED": 0,
    }
    pair_rows: list[dict[str, object]] = []
    for scene_i, scene_j in combinations(SCENE_ORDER, 2):
        sign_open3d = _sign(open3d[scene_i] - open3d[scene_j])
        sign_pcl = _sign(pcl[scene_i] - pcl[scene_j])
        if sign_open3d == 0 and sign_pcl == 0:
            classification = "BOTH_TIED"
        elif (sign_open3d == 0) != (sign_pcl == 0):
            classification = "ONE_BACKEND_TIED"
        elif sign_open3d == sign_pcl:
            classification = "CONCORDANT_NON_TIE"
        else:
            classification = "DISCORDANT"
        counts[classification] += 1
        pair_rows.append(
            {
                "scene_i": scene_i,
                "scene_j": scene_j,
                "open3d_sign": sign_open3d,
                "pcl_sign": sign_pcl,
                "classification": classification,
            }
        )
    pair_count = len(pair_rows)
    if pair_count != 15 or sum(counts.values()) != 15:
        raise AssertionError("the six-scene pair partition must contain exactly 15 pairs")
    return {
        "role": "DESCRIPTIVE_ONLY",
        "pair_count": pair_count,
        "concordant_non_tie_count": counts["CONCORDANT_NON_TIE"],
        "discordant_count": counts["DISCORDANT"],
        "both_tied_count": counts["BOTH_TIED"],
        "one_backend_tied_count": counts["ONE_BACKEND_TIED"],
        "pairwise_order_agreement_fraction": (
            counts["CONCORDANT_NON_TIE"] + counts["BOTH_TIED"]
        )
        / 15.0,
        "strict_pairwise_ordering_match": (
            counts["DISCORDANT"] == 0 and counts["ONE_BACKEND_TIED"] == 0
        ),
        "p_value": None,
        "pairs": pair_rows,
    }


def turnover_endpoint_mapping() -> dict[str, object]:
    """Return the fixed formal/companion turnover mapping."""

    return {
        "formal_primary_field": FORMAL_TURNOVER_FIELD,
        "unqualified_turnover_field": FORMAL_TURNOVER_FIELD,
        "accepted_source_turnover_retained": True,
        "accepted_source_turnover_role": ACCEPTED_SOURCE_TURNOVER_ROLE,
        "selection_by_larger_correlation_forbidden": True,
    }


def _scene_arrays(
    values: Mapping[str, Sequence[float]],
    *,
    required_per_scene: int,
    label: str,
) -> dict[str, np.ndarray]:
    if set(values) != set(SCENE_ORDER):
        raise C2SemanticError(f"{label} must contain exactly the six frozen scene IDs")
    arrays: dict[str, np.ndarray] = {}
    for scene in SCENE_ORDER:
        array = np.asarray(values[scene], dtype=np.float64)
        if array.ndim != 1 or array.size != required_per_scene:
            raise C2SemanticError(
                f"{label}[{scene}] must contain exactly {required_per_scene} values"
            )
        if not np.isfinite(array).all():
            raise C2SemanticError(f"{label}[{scene}] must be finite")
        arrays[scene] = array
    return arrays


def median_center_scene_pairs(
    turnover_by_scene: Mapping[str, Sequence[float]],
    displacement_by_scene: Mapping[str, Sequence[float]],
    *,
    required_per_scene: int = 30,
) -> dict[str, object]:
    """Median-center both variables within every frozen scene."""

    turnover = _scene_arrays(
        turnover_by_scene,
        required_per_scene=required_per_scene,
        label="turnover",
    )
    displacement = _scene_arrays(
        displacement_by_scene,
        required_per_scene=required_per_scene,
        label="displacement",
    )
    centered_turnover: dict[str, np.ndarray] = {}
    centered_displacement: dict[str, np.ndarray] = {}
    turnover_medians: dict[str, float] = {}
    displacement_medians: dict[str, float] = {}
    for scene in SCENE_ORDER:
        turnover_median = float(np.median(turnover[scene]))
        displacement_median = float(np.median(displacement[scene]))
        turnover_medians[scene] = turnover_median
        displacement_medians[scene] = displacement_median
        centered_turnover[scene] = turnover[scene] - turnover_median
        centered_displacement[scene] = displacement[scene] - displacement_median
    return {
        "center_statistic": "MEDIAN",
        "both_variables_centered": True,
        "required_per_scene": required_per_scene,
        "turnover_medians": turnover_medians,
        "displacement_medians": displacement_medians,
        "centered_turnover_by_scene": centered_turnover,
        "centered_displacement_by_scene": centered_displacement,
        "centered_turnover": np.concatenate(
            [centered_turnover[scene] for scene in SCENE_ORDER]
        ),
        "centered_displacement": np.concatenate(
            [centered_displacement[scene] for scene in SCENE_ORDER]
        ),
    }


def _constant(array: np.ndarray) -> bool:
    return bool(array.size == 0 or np.all(array == array[0]))


def centered_spearman(
    turnover_by_scene: Mapping[str, Sequence[float]],
    displacement_by_scene: Mapping[str, Sequence[float]],
    *,
    required_per_scene: int = 30,
) -> dict[str, object]:
    """Compute the frozen centered Spearman effect size on complete fixtures."""

    centered = median_center_scene_pairs(
        turnover_by_scene,
        displacement_by_scene,
        required_per_scene=required_per_scene,
    )
    x = centered["centered_turnover"]
    y = centered["centered_displacement"]
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    if _constant(x) or _constant(y):
        return {
            "status": "CENTERED_SPEARMAN_UNDEFINED_CONSTANT_INPUT",
            "rho_centered": None,
            "formal_asymptotic_p_value": None,
            "complete_pair_count": int(x.size),
            "both_variables_centered": True,
            "center_statistic": "MEDIAN",
        }
    statistic = spearmanr(x, y)
    rho = float(statistic.statistic)
    if not np.isfinite(rho):
        raise C2SemanticError("SciPy returned a nonfinite centered Spearman rho")
    return {
        "status": "CENTERED_ASSOCIATION_DEFINED_COMPLETE_180_PAIRS",
        "rho_centered": rho,
        "formal_asymptotic_p_value": None,
        "complete_pair_count": int(x.size),
        "both_variables_centered": True,
        "center_statistic": "MEDIAN",
    }


def _permuted_centered_displacement(
    centered_displacement_by_scene: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    return np.concatenate(
        [rng.permutation(centered_displacement_by_scene[scene]) for scene in SCENE_ORDER]
    )


def stratified_permutation_sensitivity(
    turnover_by_scene: Mapping[str, Sequence[float]],
    displacement_by_scene: Mapping[str, Sequence[float]],
    *,
    required_per_scene: int = 30,
) -> dict[str, object]:
    """Run the frozen 10,000-draw result-blind fixture permutation definition."""

    centered = median_center_scene_pairs(
        turnover_by_scene,
        displacement_by_scene,
        required_per_scene=required_per_scene,
    )
    x = centered["centered_turnover"]
    y = centered["centered_displacement"]
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    if _constant(x) or _constant(y):
        return {
            "status": "CENTERED_SPEARMAN_UNDEFINED_CONSTANT_INPUT",
            "draw_count": 0,
            "p_value": None,
        }
    rho_observed = float(spearmanr(x, y).statistic)
    if not np.isfinite(rho_observed):
        raise C2SemanticError("observed centered Spearman rho is nonfinite")
    rng = np.random.Generator(np.random.PCG64(CENTERED_PERMUTATION_SEED))
    centered_by_scene = centered["centered_displacement_by_scene"]
    assert isinstance(centered_by_scene, dict)
    exceedance_count = 0
    for _ in range(CENTERED_PERMUTATION_COUNT):
        permuted_y = _permuted_centered_displacement(centered_by_scene, rng)
        rho_permuted = float(spearmanr(x, permuted_y).statistic)
        if not np.isfinite(rho_permuted):
            raise C2SemanticError("permuted centered Spearman rho is nonfinite")
        if abs(rho_permuted) >= abs(rho_observed):
            exceedance_count += 1
    p_value = (1 + exceedance_count) / CENTERED_PERMUTATION_P_DENOMINATOR
    return {
        "status": "SECONDARY_STRATIFIED_MONTE_CARLO_PERMUTATION_SENSITIVITY",
        "observed_statistic_name": "ABS_CENTERED_SPEARMAN_RHO",
        "seed": CENTERED_PERMUTATION_SEED,
        "rng": "numpy.random.Generator(numpy.random.PCG64(20260820))",
        "draw_count": CENTERED_PERMUTATION_COUNT,
        "exceedance_count": exceedance_count,
        "p_value_denominator": CENTERED_PERMUTATION_P_DENOMINATOR,
        "p_value": p_value,
        "scene_strata_preserved": True,
        "duplicate_draws_retained": True,
        "backend_rng_reset_required": True,
    }


def systematic_descriptive_comparison(
    scene_systematic_fractions: Mapping[str, float],
) -> dict[str, object]:
    """Return the frozen secondary descriptive systematic comparison."""

    values = _exact_scene_mapping(scene_systematic_fractions)
    rich_values = [values[scene] for scene in RICH_SCENES]
    weak_values = [values[scene] for scene in WEAK_SCENES]
    rich_median = float(np.median(rich_values))
    weak_median = float(np.median(weak_values))
    return {
        "role": "SECONDARY_DESCRIPTIVE_MECHANISTIC_COMPARISON",
        "three_rich_scene_values": rich_values,
        "three_weak_scene_values": weak_values,
        "rich_median": rich_median,
        "weak_median": weak_median,
        "weak_minus_rich_median_difference": weak_median - rich_median,
        "formal_hypothesis_test": False,
        "directional_pass_fail": False,
    }
