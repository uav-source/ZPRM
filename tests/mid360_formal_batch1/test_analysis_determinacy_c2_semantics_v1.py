from __future__ import annotations

import numpy as np
import pytest

from experiments.mid360_formal_batch1.analysis_determinacy_c2_semantics_v1 import (
    CENTERED_PERMUTATION_COUNT,
    CENTERED_PERMUTATION_P_DENOMINATOR,
    CENTERED_PERMUTATION_SEED,
    SCENE_ORDER,
    _permuted_centered_displacement,
    centered_spearman,
    median_center_scene_pairs,
    scene_ordering_agreement,
    stratified_permutation_sensitivity,
    systematic_descriptive_comparison,
    turnover_endpoint_mapping,
)


def scene_values(values):
    return dict(zip(SCENE_ORDER, values, strict=True))


def paired_fixture(*, reverse: bool = False):
    turnover = {}
    displacement = {}
    for index, scene in enumerate(SCENE_ORDER):
        x = np.arange(30, dtype=np.float64) + index * 100.0
        y = (x[::-1] if reverse else x * 2.0) + index * 7.0
        turnover[scene] = x
        displacement[scene] = y
    return turnover, displacement


def permutation_fixture():
    turnover = {}
    displacement = {}
    for index, scene in enumerate(SCENE_ORDER):
        base = np.arange(30, dtype=np.float64)
        turnover[scene] = base + index * 50.0
        displacement[scene] = np.roll(base, index + 1) + index * 3.0
    return turnover, displacement


def test_scene_ordering_perfect_same_order():
    values = scene_values([0, 1, 2, 3, 4, 5])
    report = scene_ordering_agreement(values, values)
    assert report["pair_count"] == 15
    assert report["concordant_non_tie_count"] == 15
    assert report["discordant_count"] == 0
    assert report["pairwise_order_agreement_fraction"] == 1.0
    assert report["strict_pairwise_ordering_match"] is True
    assert report["p_value"] is None


def test_scene_ordering_reversed_order():
    report = scene_ordering_agreement(
        scene_values([0, 1, 2, 3, 4, 5]),
        scene_values([5, 4, 3, 2, 1, 0]),
    )
    assert report["discordant_count"] == 15
    assert report["pairwise_order_agreement_fraction"] == 0.0
    assert report["strict_pairwise_ordering_match"] is False


def test_scene_ordering_double_tie():
    values = scene_values([0, 0, 2, 3, 4, 5])
    report = scene_ordering_agreement(values, values)
    assert report["both_tied_count"] == 1
    assert report["one_backend_tied_count"] == 0
    assert report["strict_pairwise_ordering_match"] is True


def test_scene_ordering_one_backend_tie():
    report = scene_ordering_agreement(
        scene_values([0, 0, 2, 3, 4, 5]),
        scene_values([0, 1, 2, 3, 4, 5]),
    )
    assert report["one_backend_tied_count"] == 1
    assert report["both_tied_count"] == 0
    assert report["strict_pairwise_ordering_match"] is False


def test_turnover_mapping_is_frozen_and_companion_retained():
    mapping = turnover_endpoint_mapping()
    assert mapping["formal_primary_field"] == "correspondence_turnover"
    assert mapping["unqualified_turnover_field"] == "correspondence_turnover"
    assert mapping["accepted_source_turnover_retained"] is True
    assert mapping["accepted_source_turnover_role"] == "SECONDARY_DESCRIPTIVE_COMPANION"
    assert mapping["selection_by_larger_correlation_forbidden"] is True


def test_centering_uses_scene_median_for_both_variables():
    turnover, displacement = paired_fixture()
    centered = median_center_scene_pairs(turnover, displacement)
    assert centered["center_statistic"] == "MEDIAN"
    assert centered["both_variables_centered"] is True
    for scene in SCENE_ORDER:
        assert np.median(centered["centered_turnover_by_scene"][scene]) == 0.0
        assert np.median(centered["centered_displacement_by_scene"][scene]) == 0.0


def test_centered_known_spearman_rho():
    turnover, displacement = paired_fixture()
    report = centered_spearman(turnover, displacement)
    assert report["status"] == "CENTERED_ASSOCIATION_DEFINED_COMPLETE_180_PAIRS"
    assert report["rho_centered"] == pytest.approx(1.0)
    assert report["formal_asymptotic_p_value"] is None
    assert report["complete_pair_count"] == 180


def test_centered_constant_input_is_null():
    turnover = {scene: np.ones(30) * index for index, scene in enumerate(SCENE_ORDER)}
    displacement = {scene: np.arange(30) for scene in SCENE_ORDER}
    report = centered_spearman(turnover, displacement)
    assert report == {
        "status": "CENTERED_SPEARMAN_UNDEFINED_CONSTANT_INPUT",
        "rho_centered": None,
        "formal_asymptotic_p_value": None,
        "complete_pair_count": 180,
        "both_variables_centered": True,
        "center_statistic": "MEDIAN",
    }


def test_single_permutation_preserves_every_scene_stratum():
    _, displacement = permutation_fixture()
    centered = median_center_scene_pairs(displacement, displacement)
    by_scene = centered["centered_displacement_by_scene"]
    rng = np.random.Generator(np.random.PCG64(CENTERED_PERMUTATION_SEED))
    permuted = _permuted_centered_displacement(by_scene, rng)
    for index, scene in enumerate(SCENE_ORDER):
        observed = permuted[index * 30 : (index + 1) * 30]
        assert sorted(observed.tolist()) == sorted(by_scene[scene].tolist())


def test_stratified_permutation_exact_draws_seed_reset_and_plus_one():
    turnover, displacement = permutation_fixture()
    first = stratified_permutation_sensitivity(turnover, displacement)
    second = stratified_permutation_sensitivity(turnover, displacement)
    assert first == second
    assert first["seed"] == CENTERED_PERMUTATION_SEED
    assert first["draw_count"] == CENTERED_PERMUTATION_COUNT == 10000
    assert first["p_value_denominator"] == CENTERED_PERMUTATION_P_DENOMINATOR == 10001
    assert first["p_value"] == (1 + first["exceedance_count"]) / 10001
    assert 0.0 < first["p_value"] <= 1.0
    assert first["scene_strata_preserved"] is True
    assert first["backend_rng_reset_required"] is True


def test_systematic_comparison_is_descriptive_weak_minus_rich_without_p_field():
    report = systematic_descriptive_comparison(
        scene_values([0.1, 0.2, 0.3, 0.5, 0.6, 0.7])
    )
    assert report["role"] == "SECONDARY_DESCRIPTIVE_MECHANISTIC_COMPARISON"
    assert report["rich_median"] == pytest.approx(0.2)
    assert report["weak_median"] == pytest.approx(0.6)
    assert report["weak_minus_rich_median_difference"] == pytest.approx(0.4)
    assert report["formal_hypothesis_test"] is False
    assert report["directional_pass_fail"] is False
    assert "p_value" not in report
    assert "permutation" not in report
