from experiments.mid360_formal_batch1.locked_analysis.contract_v1 import SCENE_ORDER
from experiments.mid360_formal_batch1.locked_analysis.cross_backend_v1 import (
    formal_spearman, scene_ordering_agreement, translation_direction_cosine,
)


def test_scene_spearman_and_fifteen_pair_ordering():
    values = {scene: float(i) for i, scene in enumerate(SCENE_ORDER)}
    result = formal_spearman(values, values, ordered_ids=SCENE_ORDER,
                             incomplete_status="INCOMPLETE")
    assert result["rho"] == 1.0 and result["complete_pair_n"] == 6
    ordering = scene_ordering_agreement(values, values)
    assert ordering["pair_count"] == 15
    assert ordering["concordant_non_tie_count"] == 15
    assert ordering["strict_pairwise_ordering_match"] is True
    assert ordering["p_value"] is None


def test_ties_and_constant_spearman_are_explicit():
    left = {scene: 1.0 for scene in SCENE_ORDER}
    right = dict(left); right["FMB1_W03"] = 2.0
    ordering = scene_ordering_agreement(left, right)
    assert ordering["one_backend_tied_count"] == 5
    result = formal_spearman(left, left, ordered_ids=SCENE_ORDER,
                             incomplete_status="INCOMPLETE")
    assert result["status"] == "SPEARMAN_UNDEFINED_CONSTANT_INPUT"
    assert result["rho"] is result["p_value"] is None


def test_incomplete_ordering_and_spearman_are_null_not_partial_inference():
    left = {scene: float(i) for i, scene in enumerate(SCENE_ORDER)}
    right = dict(left); right["FMB1_W02"] = None
    spearman = formal_spearman(left, right, ordered_ids=SCENE_ORDER,
                               incomplete_status="INCOMPLETE")
    assert spearman["status"] == "INCOMPLETE" and spearman["complete_pair_n"] == 5
    ordering = scene_ordering_agreement(left, right)
    assert ordering["status"].startswith("SCENE_ORDERING_UNDEFINED")
    assert ordering["strict_pairwise_ordering_match"] is None


def test_direction_cosine_defined_zero_and_missing():
    assert translation_direction_cosine([1, 0, 0], [2, 0, 0])["cosine"] == 1.0
    assert translation_direction_cosine([0, 0, 0], [1, 0, 0])["status"].endswith("ZERO_VECTOR")
    assert translation_direction_cosine(None, [1, 0, 0])["status"].endswith("NONFINITE_VECTOR")
