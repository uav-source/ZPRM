"""Frozen local-geometry and reassociation Ridge model comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


RIDGE_ALPHA = 1.0
RIDGE_FIT_INTERCEPT = True
TARGET_EPSILON_M = 1.0e-9
LAMBDA_FLOOR = 1.0e-12
NONIDEAL_CONDITIONS = frozenset(
    {
        "INDEPENDENT_NOISE_FREE",
        "SCAN_NOISE_ONLY",
        "MAP_NOISE_ONLY",
        "DROPOUT_ONLY",
        "FULL_NOISE",
    }
)
MODEL_A_FEATURE_NAMES = (
    "log10_initial_residual_rmse_plus_1e-9",
    "log10_condition_number_trans_plus_1",
    "log10_inverse_lambda_min_trans",
    "spectral_entropy_trans",
    "log10_initial_correspondence_count_plus_1",
    "initial_translation_gradient_norm",
)
MODEL_B_ADDITIONAL_FEATURE_NAMES = (
    "correspondence_turnover",
    "accepted_source_turnover",
    "median_normal_angle_change_deg",
    "q95_normal_angle_change_deg",
    "residual_rmse_change",
    "correspondence_count_change_ratio",
)
MODEL_B_FEATURE_NAMES = MODEL_A_FEATURE_NAMES + MODEL_B_ADDITIONAL_FEATURE_NAMES
MODEL_NAMES = frozenset({"A", "B"})

_MODEL_INPUT_FIELDS = (
    "initial_residual_rmse",
    "condition_number_trans",
    "lambda_min_trans",
    "spectral_entropy_trans",
    "initial_correspondence_count",
    "initial_translation_gradient_norm",
)


def _number(row: Mapping[str, Any], name: str, *, nonnegative: bool = False) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"model field {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"model field {name} is invalid")
    return result


def _translation_error(row: Mapping[str, Any]) -> float:
    if "translation_error_m" in row:
        return _number(row, "translation_error_m", nonnegative=True)
    return _number(row, "translation_update_m", nonnegative=True)


def model_feature_vector(row: Mapping[str, Any], model: str) -> np.ndarray:
    """Apply the exact preregistered transforms for Model A or Model B."""

    if model not in MODEL_NAMES:
        raise ValueError("model must be 'A' or 'B'")
    if row.get("condition") not in NONIDEAL_CONDITIONS:
        raise ValueError("local metric models accept only five non-IDEAL conditions")
    if row.get("common_association_valid") is not True:
        raise ValueError("invalid common-association rows cannot enter a model")
    for field in _MODEL_INPUT_FIELDS:
        _number(row, field, nonnegative=True)
    initial_rmse = float(row["initial_residual_rmse"])
    condition = float(row["condition_number_trans"])
    lambda_min = float(row["lambda_min_trans"])
    correspondence_count = float(row["initial_correspondence_count"])
    features = [
        math.log10(initial_rmse + 1.0e-9),
        math.log10(condition + 1.0),
        math.log10(1.0 / max(lambda_min, LAMBDA_FLOOR)),
        float(row["spectral_entropy_trans"]),
        math.log10(correspondence_count + 1.0),
        float(row["initial_translation_gradient_norm"]),
    ]
    if model == "B":
        features.extend(
            _number(row, name)
            for name in MODEL_B_ADDITIONAL_FEATURE_NAMES
        )
    result = np.asarray(features, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("transformed model features are non-finite")
    return result


def model_matrix(
    rows: Sequence[Mapping[str, Any]], model: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("model rows cannot be empty")
    features = np.vstack([model_feature_vector(row, model) for row in rows])
    target = np.asarray(
        [math.log10(_translation_error(row) + TARGET_EPSILON_M) for row in rows],
        dtype=np.float64,
    )
    geometry = np.asarray([int(row["geometry_seed"]) for row in rows], dtype=np.int64)
    return features, target, geometry


def leave_one_geometry_seed_out_ridge(
    rows: Sequence[Mapping[str, Any]], model: str
) -> dict[str, Any]:
    """Three-fold geometry-seed extrapolation with train-only scaling."""

    features, target, geometry = model_matrix(rows, model)
    unique_geometry = np.asarray(sorted(set(geometry.tolist())), dtype=np.int64)
    if unique_geometry.size != 3:
        raise ValueError("leave-one-geometry-seed-out requires exactly three seeds")
    fold_rows: list[dict[str, Any]] = []
    all_absolute_errors: list[float] = []
    for held_out in unique_geometry:
        test = geometry == held_out
        train = ~test
        if not np.any(train) or not np.any(test):
            raise ValueError("geometry-seed fold is empty")
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_scaled = scaler.fit_transform(features[train])
        test_scaled = scaler.transform(features[test])
        estimator = Ridge(alpha=RIDGE_ALPHA, fit_intercept=RIDGE_FIT_INTERCEPT)
        estimator.fit(train_scaled, target[train])
        predicted = estimator.predict(test_scaled)
        absolute_error = np.abs(predicted - target[test])
        all_absolute_errors.extend(float(value) for value in absolute_error)
        fold_rows.append(
            {
                "held_out_geometry_seed": int(held_out),
                "training_geometry_seeds": [
                    int(value) for value in unique_geometry if value != held_out
                ],
                "training_row_count": int(np.count_nonzero(train)),
                "test_row_count": int(np.count_nonzero(test)),
                "fold_mae_log10_translation_error": float(np.mean(absolute_error)),
                "scaler_fit_scope": "training_fold_only",
                "scaler_mean": scaler.mean_.astype(float).tolist(),
                "scaler_scale": scaler.scale_.astype(float).tolist(),
                "ridge_alpha": float(estimator.alpha),
                "ridge_fit_intercept": bool(estimator.fit_intercept),
                "target_true": target[test].astype(float).tolist(),
                "target_predicted": predicted.astype(float).tolist(),
            }
        )
    names = MODEL_A_FEATURE_NAMES if model == "A" else MODEL_B_FEATURE_NAMES
    return {
        "model": model,
        "feature_names": list(names),
        "feature_count": len(names),
        "cv_scheme": "leave_one_geometry_seed_out",
        "fold_count": 3,
        "cv_mae_log10_translation_error": float(np.mean(all_absolute_errors)),
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_fit_intercept": RIDGE_FIT_INTERCEPT,
        "standard_scaler_fit_scope": "training_fold_only",
        "folds": fold_rows,
    }


def compare_local_metric_models(
    rows: Sequence[Mapping[str, Any]], *, backend: str, backend_field: str = "backend"
) -> dict[str, Any]:
    selected = [row for row in rows if row.get(backend_field) == backend]
    if not selected:
        raise ValueError(f"no model rows for backend: {backend}")
    model_a = leave_one_geometry_seed_out_ridge(selected, "A")
    model_b = leave_one_geometry_seed_out_ridge(selected, "B")
    mae_a = float(model_a["cv_mae_log10_translation_error"])
    mae_b = float(model_b["cv_mae_log10_translation_error"])
    improvement = None if mae_a <= 0.0 else float((mae_a - mae_b) / mae_a)
    return {
        "backend": backend,
        "model_a_cv_mae": mae_a,
        "model_b_cv_mae": mae_b,
        "relative_improvement": improvement,
        "model_a": model_a,
        "model_b": model_b,
    }


def _normalized_eigenvalues(row: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(
        [
            _number(row, "lambda_min_trans", nonnegative=True),
            _number(row, "lambda_mid_trans", nonnegative=True),
            _number(row, "lambda_max_trans", nonnegative=True),
        ],
        dtype=np.float64,
    )
    total = float(np.sum(values))
    if total <= 1.0e-12:
        raise ValueError("H_t eigenvalue sum is zero")
    return values / total


def _candidate_id(backend: str, condition: str, snapshot_a: str, snapshot_b: str) -> str:
    payload = json.dumps(
        {
            "backend": backend,
            "condition": condition,
            "snapshot_a": snapshot_a,
            "snapshot_b": snapshot_b,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"automatic::{hashlib.sha256(payload).hexdigest()}"


def find_automatic_nonequivalence_candidates(
    rows: Sequence[Mapping[str, Any]], *, backend: str, backend_field: str = "backend"
) -> list[dict[str, Any]]:
    """Apply every automatic pair criterion without promoting manual status."""

    selected = [
        row
        for row in rows
        if row.get(backend_field) == backend
        and row.get("condition") in NONIDEAL_CONDITIONS
        and row.get("common_association_valid") is True
    ]
    selected.sort(
        key=lambda row: (
            str(row.get("condition")),
            str(row.get("snapshot_id")),
        )
    )
    output: list[dict[str, Any]] = []
    for first_index, first in enumerate(selected):
        for second in selected[first_index + 1 :]:
            if first.get("condition") != second.get("condition"):
                continue
            if first.get("snapshot_id") == second.get("snapshot_id"):
                continue
            try:
                first_error = _translation_error(first)
                second_error = _translation_error(second)
                first_turnover = _number(first, "correspondence_turnover")
                second_turnover = _number(second, "correspondence_turnover")
                if min(first_error, second_error) <= 0.0:
                    continue
                high, low = (
                    (first, second)
                    if first_error >= second_error
                    else (second, first)
                )
                high_error = max(first_error, second_error)
                low_error = min(first_error, second_error)
                high_eigen = _normalized_eigenvalues(high)
                low_eigen = _normalized_eigenvalues(low)
                denominator = float(np.linalg.norm(high_eigen) * np.linalg.norm(low_eigen))
                if denominator <= 0.0:
                    continue
                similarity = float(np.dot(high_eigen, low_eigen) / denominator)
                high_condition = _number(high, "condition_number_trans", nonnegative=True)
                low_condition = _number(low, "condition_number_trans", nonnegative=True)
                high_rmse = _number(high, "initial_residual_rmse", nonnegative=True)
                low_rmse = _number(low, "initial_residual_rmse", nonnegative=True)
                if min(high_condition, low_condition, high_rmse, low_rmse) <= 0.0:
                    continue
                condition_log_difference = abs(math.log(high_condition / low_condition))
                rmse_log_difference = abs(math.log(high_rmse / low_rmse))
                high_count = _number(high, "initial_correspondence_count", nonnegative=True)
                low_count = _number(low, "initial_correspondence_count", nonnegative=True)
                if low_count <= 0.0:
                    continue
                count_ratio = high_count / low_count
                error_ratio = high_error / low_error
                turnover_difference = abs(
                    _number(high, "correspondence_turnover")
                    - _number(low, "correspondence_turnover")
                )
            except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
                continue
            if not (
                similarity >= 0.98
                and condition_log_difference <= 0.20
                and rmse_log_difference <= 0.20
                and 0.90 <= count_ratio <= 1.10
                and error_ratio >= 5.0
                and turnover_difference >= 0.15
                and math.isfinite(error_ratio)
            ):
                continue
            snapshot_a = str(high["snapshot_id"])
            snapshot_b = str(low["snapshot_id"])
            output.append(
                {
                    "candidate_pair_id": _candidate_id(
                        backend, str(high["condition"]), snapshot_a, snapshot_b
                    ),
                    "backend": backend,
                    "snapshot_a": snapshot_a,
                    "snapshot_b": snapshot_b,
                    "scene_a": str(high["scene_variant"]),
                    "scene_b": str(low["scene_variant"]),
                    "condition": str(high["condition"]),
                    "linear_metric_similarity": similarity,
                    "condition_log_ratio_abs": condition_log_difference,
                    "initial_rmse_log_ratio_abs": rmse_log_difference,
                    "correspondence_count_ratio": count_ratio,
                    "error_ratio": error_ratio,
                    "turnover_difference": turnover_difference,
                    "manual_review_status": "AUTOMATIC_CANDIDATE",
                }
            )
    return output


def local_metric_incremental_value_gate(
    model_comparisons: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    backends: Sequence[str] = ("open3d_point_to_plane", "pcl_point_to_plane"),
) -> dict[str, Any]:
    by_backend = {row.get("backend"): row for row in model_comparisons}
    improvements = [
        by_backend.get(backend, {}).get("relative_improvement") for backend in backends
    ]
    condition_a = bool(
        all(value is not None and math.isfinite(float(value)) for value in improvements)
        and float(np.mean(improvements)) >= 0.10
        and all(float(value) >= -0.02 for value in improvements)
    )
    candidate_counts = Counter(row.get("backend") for row in candidates)
    scene_pairs = {
        tuple(sorted((str(row.get("scene_a")), str(row.get("scene_b")))))
        for row in candidates
    }
    condition_b = bool(
        all(candidate_counts[backend] >= 10 for backend in backends)
        and len(scene_pairs) >= 3
    )
    return {
        "condition_a_model_improvement_pass": condition_a,
        "condition_b_candidate_coverage_pass": condition_b,
        "mean_relative_improvement": (
            None
            if any(value is None for value in improvements)
            else float(np.mean(improvements))
        ),
        "relative_improvement_by_backend": dict(zip(backends, improvements)),
        "automatic_candidate_count_by_backend": {
            backend: candidate_counts[backend] for backend in backends
        },
        "automatic_candidate_scene_pair_count": len(scene_pairs),
        "LOCAL_METRIC_INCREMENTAL_VALUE_PASS": bool(condition_a or condition_b),
    }


__all__ = [
    "LAMBDA_FLOOR",
    "MODEL_A_FEATURE_NAMES",
    "MODEL_B_ADDITIONAL_FEATURE_NAMES",
    "MODEL_B_FEATURE_NAMES",
    "NONIDEAL_CONDITIONS",
    "RIDGE_ALPHA",
    "RIDGE_FIT_INTERCEPT",
    "TARGET_EPSILON_M",
    "compare_local_metric_models",
    "find_automatic_nonequivalence_candidates",
    "leave_one_geometry_seed_out_ridge",
    "local_metric_incremental_value_gate",
    "model_feature_vector",
    "model_matrix",
]
