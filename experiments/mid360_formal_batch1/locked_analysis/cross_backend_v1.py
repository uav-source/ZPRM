"""Frozen Open3D/PCL agreement summaries and descriptive direction cosine."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .contract_v1 import SCENE_ORDER


class CrossBackendError(ValueError):
    """Raised when backend agreement input violates the frozen grain."""


def formal_spearman(
    open3d_values: Mapping[str, float | None],
    pcl_values: Mapping[str, float | None],
    *,
    ordered_ids: Sequence[str],
    incomplete_status: str,
) -> dict[str, Any]:
    if set(open3d_values) != set(ordered_ids) or set(pcl_values) != set(ordered_ids):
        raise CrossBackendError("Spearman inputs must contain exactly the frozen IDs")
    pairs: list[dict[str, Any]] = []
    incomplete = False
    for identifier in ordered_ids:
        left = open3d_values[identifier]
        right = pcl_values[identifier]
        if (
            left is None
            or right is None
            or isinstance(left, bool)
            or isinstance(right, bool)
            or not np.isfinite(left)
            or not np.isfinite(right)
        ):
            incomplete = True
            pairs.append(
                {"identifier": identifier, "open3d_value": None,
                 "pcl_value": None, "complete": False}
            )
            continue
        pairs.append(
            {"identifier": identifier, "open3d_value": float(left),
             "pcl_value": float(right), "complete": True}
        )
    complete_pairs = [pair for pair in pairs if pair["complete"]]
    if incomplete:
        return {
            "status": incomplete_status,
            "required_complete_pair_n": len(ordered_ids),
            "complete_pair_n": len(complete_pairs),
            "rho": None,
            "p_value": None,
            "pairs": pairs,
        }
    left_array = np.asarray([pair["open3d_value"] for pair in complete_pairs])
    right_array = np.asarray([pair["pcl_value"] for pair in complete_pairs])
    if np.all(left_array == left_array[0]) or np.all(right_array == right_array[0]):
        return {
            "status": "SPEARMAN_UNDEFINED_CONSTANT_INPUT",
            "required_complete_pair_n": len(ordered_ids),
            "complete_pair_n": len(complete_pairs),
            "rho": None,
            "p_value": None,
            "pairs": pairs,
        }
    result = spearmanr(left_array, right_array)
    return {
        "status": "SPEARMAN_DEFINED_COMPLETE_PAIRS",
        "required_complete_pair_n": len(ordered_ids),
        "complete_pair_n": len(complete_pairs),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "pairs": pairs,
    }


def scene_ordering_agreement(
    open3d_scene_medians: Mapping[str, float | None],
    pcl_scene_medians: Mapping[str, float | None],
) -> dict[str, Any]:
    if set(open3d_scene_medians) != set(SCENE_ORDER) or set(pcl_scene_medians) != set(
        SCENE_ORDER
    ):
        raise CrossBackendError("scene ordering requires exactly six frozen scenes")
    if any(
        value is None or isinstance(value, bool) or not np.isfinite(value)
        for values in (open3d_scene_medians, pcl_scene_medians)
        for value in values.values()
    ):
        return {
            "status": "SCENE_ORDERING_UNDEFINED_INCOMPLETE_SIX_SCENE_COVERAGE",
            "role": "DESCRIPTIVE_ONLY", "pair_count": 15,
            "concordant_non_tie_count": None, "discordant_count": None,
            "both_tied_count": None, "one_backend_tied_count": None,
            "pairwise_order_agreement_fraction": None,
            "strict_pairwise_ordering_match": None, "p_value": None,
            "pairs": [],
        }
    counts = {
        "CONCORDANT_NON_TIE": 0,
        "DISCORDANT": 0,
        "BOTH_TIED": 0,
        "ONE_BACKEND_TIED": 0,
    }
    pair_rows = []
    for scene_i, scene_j in combinations(SCENE_ORDER, 2):
        left_delta = float(open3d_scene_medians[scene_i]) - float(
            open3d_scene_medians[scene_j]
        )
        right_delta = float(pcl_scene_medians[scene_i]) - float(
            pcl_scene_medians[scene_j]
        )
        left_sign = int(np.sign(left_delta))
        right_sign = int(np.sign(right_delta))
        if left_sign == 0 and right_sign == 0:
            classification = "BOTH_TIED"
        elif (left_sign == 0) != (right_sign == 0):
            classification = "ONE_BACKEND_TIED"
        elif left_sign == right_sign:
            classification = "CONCORDANT_NON_TIE"
        else:
            classification = "DISCORDANT"
        counts[classification] += 1
        pair_rows.append(
            {
                "scene_i": scene_i,
                "scene_j": scene_j,
                "open3d_sign": left_sign,
                "pcl_sign": right_sign,
                "classification": classification,
            }
        )
    return {
        "status": "SCENE_ORDERING_DEFINED_COMPLETE_15_PAIRS",
        "role": "DESCRIPTIVE_ONLY",
        "pair_count": 15,
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


def translation_direction_cosine(
    open3d_vector: Sequence[float] | None,
    pcl_vector: Sequence[float] | None,
) -> dict[str, Any]:
    if open3d_vector is None or pcl_vector is None:
        return {
            "status": "DIRECTION_COSINE_UNDEFINED_MISSING_OR_NONFINITE_VECTOR",
            "cosine": None,
        }
    left = np.asarray(open3d_vector, dtype=np.float64)
    right = np.asarray(pcl_vector, dtype=np.float64)
    if left.shape != (3,) or right.shape != (3,) or not (
        np.isfinite(left).all() and np.isfinite(right).all()
    ):
        return {
            "status": "DIRECTION_COSINE_UNDEFINED_MISSING_OR_NONFINITE_VECTOR",
            "cosine": None,
        }
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return {
            "status": "DIRECTION_COSINE_UNDEFINED_ZERO_VECTOR",
            "cosine": None,
        }
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return {
        "status": "DIRECTION_COSINE_DEFINED",
        "cosine": float(np.clip(cosine, -1.0, 1.0)),
    }
