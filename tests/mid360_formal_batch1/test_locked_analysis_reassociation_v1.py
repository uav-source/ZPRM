import numpy as np

from experiments.mid360_formal_batch1.locked_analysis.contract_v1 import SCENE_ORDER
from experiments.mid360_formal_batch1.locked_analysis.reassociation_v1 import (
    ACCEPTED_SOURCE_TURNOVER_ROLE, FORMAL_TURNOVER_FIELD,
    accepted_source_turnover_descriptive, centered_association,
    centered_permutation_sensitivity, formal_scene_turnover,
)


def centered_fixture():
    x = {scene: list(np.linspace(i, i + 1, 30)) for i, scene in enumerate(SCENE_ORDER)}
    y = {scene: list(np.linspace(i * 2, i * 2 + 3, 30)) for i, scene in enumerate(SCENE_ORDER)}
    return centered_association(x, y)


def test_primary_turnover_and_secondary_companion_roles():
    summary = formal_scene_turnover([i / 30 for i in range(30)])
    assert summary["formal_turnover_field"] == FORMAL_TURNOVER_FIELD == "correspondence_turnover"
    companion = accepted_source_turnover_descriptive([0.1] * 30)
    assert companion["role"] == ACCEPTED_SOURCE_TURNOVER_ROLE
    assert companion["may_replace_formal_turnover"] is False


def test_both_variables_are_scene_median_centered():
    centered = centered_fixture()
    assert centered["status"] == "CENTERED_ASSOCIATION_DEFINED_COMPLETE_180_PAIRS"
    assert centered["both_variables_median_centered"] is True
    for scene in SCENE_ORDER:
        assert abs(float(np.median(centered["centered_turnover_by_scene"][scene]))) < 1e-12
        assert abs(float(np.median(centered["centered_translation_by_scene"][scene]))) < 1e-12


def test_registered_permutation_is_seeded_stratified_and_10000():
    result = centered_permutation_sensitivity(centered_fixture(), backend="OPEN3D_POINT_TO_PLANE")
    assert result["draw_count"] == 10000 and len(result["draws"]) == 10000
    assert result["seed"] == 20260820 and "PCG64" in result["rng"]
    assert result["p_value_denominator"] == 10001
    assert result["scene_strata_preserved"] is True
    assert result["p_value"] == (result["extreme_count"] + 1) / 10001
