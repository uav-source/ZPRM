from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from phase_a_harness.common_association_analysis import (
    AssociationState,
    MAX_ASSOCIATION_DISTANCE_M,
    PCA_MIN_NEIGHBOR_COUNT,
    PCA_NEIGHBOR_COUNT,
    analyze_estimated_transform,
    associate_source_points,
    estimate_target_normals_pca,
    prepare_common_association_context,
    turnover_from_associations,
)
from phase_a_harness.full_synthetic_statistics import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    descriptive_development_summary,
    hierarchical_bootstrap_indices,
    paired_development_summary,
    spearman_development_summary,
)
from phase_a_harness.local_metric_models import (
    RIDGE_ALPHA,
    compare_local_metric_models,
    find_automatic_nonequivalence_candidates,
    leave_one_geometry_seed_out_ridge,
    local_metric_incremental_value_gate,
    model_feature_vector,
    model_matrix,
)
from phase_a_harness.systematic_offset_analysis import (
    summarize_systematic_offset,
    summarize_systematic_offset_groups,
    systematic_offset_claim_gate,
    translation_direction_concentration,
)


def _plane() -> np.ndarray:
    x, y = np.meshgrid(np.arange(10) * 0.1, np.arange(10) * 0.1, indexing="ij")
    return np.ascontiguousarray(
        np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))), dtype=np.float64
    )


def test_pca_normals_use_fixed_k50_and_are_unoriented_planar_normals():
    points = _plane()
    normals, valid = estimate_target_normals_pca(points, chunk_size=17)
    assert PCA_NEIGHBOR_COUNT == 50 and PCA_MIN_NEIGHBOR_COUNT == 10
    assert valid.all()
    assert np.allclose(np.abs(normals[:, 2]), 1.0, atol=1.0e-12)
    assert np.allclose(normals[:, :2], 0.0, atol=1.0e-12)


def test_common_nearest_neighbor_distance_limit_is_inclusive_050_m():
    target = np.asarray([[0.0, 0.0, 0.0]])
    source = np.asarray(
        [[MAX_ASSOCIATION_DISTANCE_M, 0.0, 0.0], [0.500001, 0.0, 0.0]]
    )
    state = associate_source_points(
        source,
        np.eye(4),
        target_tree=cKDTree(target),
        target_point_count=1,
    )
    assert state.count == 1
    assert state.source_indices.tolist() == [0]
    assert state.target_indices.tolist() == [0]


def _state(sources, targets) -> AssociationState:
    count = len(sources)
    return AssociationState(
        source_indices=np.asarray(sources, dtype=np.int64),
        target_indices=np.asarray(targets, dtype=np.int64),
        source_points_target=np.zeros((count, 3)),
        distances_m=np.zeros(count),
    )


def test_correspondence_and_accepted_source_turnover_are_jaccard():
    initial = _state([0, 1], [0, 1])
    final = _state([0, 2], [1, 2])
    pair_turnover, source_turnover = turnover_from_associations(initial, final)
    assert pair_turnover == 1.0
    assert np.isclose(source_turnover, 2.0 / 3.0)
    assert turnover_from_associations(_state([], []), _state([], [])) == (None, None)


def test_snapshot_context_caches_reference_metrics_for_both_backend_estimates():
    target = _plane()
    source = target[::3].copy()
    context = prepare_common_association_context(
        source, target, np.eye(4), snapshot_id="snapshot-1", pca_chunk_size=19
    )
    assert not np.shares_memory(context.source_points, source)
    assert not np.shares_memory(context.target_points, target)
    assert not np.shares_memory(context.target_tree.data, target)
    first = analyze_estimated_transform(
        context, np.eye(4), identifiers={"backend": "open3d_point_to_plane"}
    )
    second = analyze_estimated_transform(
        context, np.eye(4), identifiers={"backend": "pcl_point_to_plane"}
    )
    assert first["common_association_valid"] and second["common_association_valid"]
    assert first["snapshot_id"] == second["snapshot_id"] == "snapshot-1"
    assert first["correspondence_turnover"] == 0.0
    assert first["accepted_source_turnover"] == 0.0
    assert first["initial_residual_rmse"] == first["final_residual_rmse"] == 0.0
    assert first["median_normal_angle_change_deg"] == 0.0
    assert first["lambda_max_trans"] == 1.0
    assert first["spectral_entropy_trans"] == 0.0
    assert first["initial_translation_gradient_norm"] == 0.0
    assert context.target_tree is context.target_tree


def test_final_pose_is_rematched_and_no_final_correspondence_is_retained_as_invalid():
    target = _plane()
    source = target[::4].copy()
    context = prepare_common_association_context(source, target, np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 0.1
    rematched = analyze_estimated_transform(context, shifted)
    assert rematched["correspondence_turnover"] > 0.0
    assert rematched["accepted_source_turnover"] == 0.0
    far = np.eye(4)
    far[0, 3] = 10.0
    invalid = analyze_estimated_transform(context, far)
    assert not invalid["common_association_valid"]
    assert invalid["common_association_invalid_reason"] == "NO_FINAL_CORRESPONDENCE"
    assert invalid["final_correspondence_count"] == 0


def test_insufficient_pca_neighbors_produces_explicit_invalid_reason():
    target = np.column_stack((np.arange(9) * 0.01, np.zeros(9), np.zeros(9)))
    context = prepare_common_association_context(target.copy(), target, np.eye(4))
    result = analyze_estimated_transform(context, np.eye(4))
    assert not result["common_association_valid"]
    assert result["common_association_invalid_reason"] == "INSUFFICIENT_VALID_NORMALS"
    assert result["target_normal_valid_count"] == 0


def test_systematic_offset_covariance_uses_ddof1_and_direction_concentration():
    translation = np.tile([1.0, 0.0, 0.0], (10, 1))
    rotation = np.tile([0.0, 0.1, 0.0], (10, 1))
    summary = summarize_systematic_offset(translation, rotation)
    assert summary["observation_count_contract_pass"]
    assert summary["translation_repeatability_covariance_ddof"] == 1
    assert summary["rotation_repeatability_covariance_ddof"] == 1
    assert summary["translation_repeatability_rms_m"] == 0.0
    assert summary["systematic_translation_offset_m"] == 1.0
    assert summary["systematic_fraction_translation"] == 1.0
    assert summary["translation_direction_concentration"] == 1.0


def test_systematic_fraction_zero_denominator_and_zero_vectors_are_null():
    zero = np.zeros((10, 3))
    summary = summarize_systematic_offset(zero, zero)
    assert summary["systematic_fraction_translation"] is None
    assert summary["translation_direction_concentration"] is None
    assert summary["translation_direction_valid_nonzero_count"] == 0
    concentration, count = translation_direction_concentration(
        np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    assert concentration == 0.0 and count == 2
    incomplete = summarize_systematic_offset_groups(
        [
            {
                "scene_variant": "LONG_CORRIDOR",
                "geometry_seed": 1,
                "condition": "FULL_NOISE",
                "backend": "open3d_point_to_plane",
                "translation_vector": [0.01, 0.0, 0.0],
                "rotation_vector": [0.0, 0.0, 0.0],
            }
        ]
    )
    assert len(incomplete) == 1
    assert not incomplete[0]["systematic_offset_analysis_valid"]
    assert not incomplete[0]["observation_count_contract_pass"]
    gate = systematic_offset_claim_gate(incomplete)
    assert not gate["SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED"]


def _bootstrap_keys():
    geometry, measurement, repeat = [], [], []
    for g in (1, 2, 3):
        for m in (10, 20):
            for r in (0, 1):
                geometry.append(g)
                measurement.append(m)
                repeat.append(r)
    return geometry, measurement, repeat


def test_hierarchical_bootstrap_contract_is_exact_and_deterministic():
    keys = _bootstrap_keys()
    first = hierarchical_bootstrap_indices(*keys)
    second = hierarchical_bootstrap_indices(*keys)
    assert BOOTSTRAP_REPETITIONS == 2000 and BOOTSTRAP_SEED == 1191248828
    assert len(first) == 2000
    assert all(len(sample) == 12 for sample in first)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


def test_development_descriptive_paired_and_spearman_statistics():
    geometry, measurement, repeat = _bootstrap_keys()
    right = np.arange(1.0, 13.0)
    left = 2.0 * right
    descriptive = descriptive_development_summary(right)
    paired = paired_development_summary(
        left, right, geometry, measurement, repeat
    )
    correlation = spearman_development_summary(
        left, right, geometry, measurement, repeat
    )
    assert descriptive["median"] == 6.5
    assert descriptive["q95"] == np.quantile(right, 0.95, method="linear")
    assert paired["ratio_of_medians"] == paired["paired_median_ratio"] == 2.0
    assert paired["paired_win_rate"] == 1.0
    assert paired["paired_difference_valid_bootstrap_count"] == 2000
    assert correlation["spearman_rho"] == 1.0
    assert correlation["development_only_not_confirmatory"]


def _model_row(seed: int, index: int, backend: str = "open3d_point_to_plane"):
    scale = 1.0 + 0.1 * index + 0.5 * (seed - 1)
    turnover = 0.02 * index + 0.05 * seed
    error = 10 ** (-3.0 + 1.5 * turnover + 0.02 * index)
    return {
        "snapshot_id": f"g{seed}-{index}",
        "scene_variant": "LONG_CORRIDOR",
        "condition": "FULL_NOISE",
        "backend": backend,
        "geometry_seed": seed,
        "common_association_valid": True,
        "translation_error_m": error,
        "initial_residual_rmse": 0.001 * scale,
        "condition_number_trans": 10.0 * scale,
        "lambda_min_trans": 0.01 / scale,
        "lambda_mid_trans": 0.2,
        "lambda_max_trans": 0.79,
        "spectral_entropy_trans": 0.5 + 0.001 * index,
        "initial_correspondence_count": 100 + index,
        "initial_translation_gradient_norm": 0.01 * scale,
        "correspondence_turnover": turnover,
        "accepted_source_turnover": turnover / 2.0,
        "median_normal_angle_change_deg": turnover * 10.0,
        "q95_normal_angle_change_deg": turnover * 20.0,
        "residual_rmse_change": turnover * 0.001,
        "correspondence_count_change_ratio": turnover / 4.0,
    }


def test_geometry_seed_cv_uses_three_folds_train_only_scaler_and_fixed_ridge():
    rows = [_model_row(seed, index) for seed in (1, 2, 3) for index in range(6)]
    result = leave_one_geometry_seed_out_ridge(rows, "B")
    features, _, geometry = model_matrix(rows, "B")
    assert result["fold_count"] == 3 and result["ridge_alpha"] == RIDGE_ALPHA == 1.0
    for fold in result["folds"]:
        train = geometry != fold["held_out_geometry_seed"]
        assert np.allclose(fold["scaler_mean"], np.mean(features[train], axis=0))
        assert fold["scaler_fit_scope"] == "training_fold_only"
    comparison = compare_local_metric_models(rows, backend="open3d_point_to_plane")
    assert comparison["model_a_cv_mae"] >= 0.0
    assert comparison["model_b_cv_mae"] >= 0.0


def test_automatic_candidate_rules_and_status_cannot_self_validate():
    low = _model_row(1, 0)
    high = dict(low)
    low.update(
        snapshot_id="low",
        scene_variant="GEOMETRY_RICH_ROOM",
        translation_error_m=0.005,
        correspondence_turnover=0.05,
        initial_residual_rmse=0.010,
        condition_number_trans=100.0,
        initial_correspondence_count=100,
    )
    high.update(
        snapshot_id="high",
        scene_variant="LONG_CORRIDOR",
        translation_error_m=0.050,
        correspondence_turnover=0.25,
        initial_residual_rmse=0.011,
        condition_number_trans=105.0,
        initial_correspondence_count=100,
    )
    candidates = find_automatic_nonequivalence_candidates(
        [low, high], backend="open3d_point_to_plane"
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["snapshot_a"] == "high"
    assert candidate["error_ratio"] == 10.0
    assert candidate["turnover_difference"] == 0.20
    assert candidate["manual_review_status"] == "AUTOMATIC_CANDIDATE"
    assert "VALID_SCIENTIFIC_COUNTEREXAMPLE" not in candidate.values()
    gate = local_metric_incremental_value_gate(
        [
            {"backend": "open3d_point_to_plane", "relative_improvement": 0.12},
            {"backend": "pcl_point_to_plane", "relative_improvement": 0.10},
        ],
        [],
    )
    assert gate["condition_a_model_improvement_pass"]
    assert gate["LOCAL_METRIC_INCREMENTAL_VALUE_PASS"]
    vector = model_feature_vector(low, "A")
    assert vector.shape == (6,) and np.all(np.isfinite(vector))
