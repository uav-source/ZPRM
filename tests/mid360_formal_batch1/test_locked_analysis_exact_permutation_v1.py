import pytest

from experiments.mid360_formal_batch1.locked_analysis.contract_v1 import SCENE_ORDER
from experiments.mid360_formal_batch1.locked_analysis.exact_scene_permutation_v1 import (
    ExactPermutationError, enumerate_allocations, exact_weak_greater_than_rich,
)


def test_exact_twenty_allocation_one_sided_rule():
    result = exact_weak_greater_than_rich({scene: float(i) for i, scene in enumerate(SCENE_ORDER)})
    assert len(enumerate_allocations()) == len(set(enumerate_allocations())) == 20
    assert result["allocation_count"] == result["p_value_denominator"] == 20
    assert result["plus_one_correction"] is False
    assert result["p_value"] * 20 == round(result["p_value"] * 20)
    assert result["p_value"] != 1 / 21


def test_exact_inference_rejects_snapshot_grain_and_undefined_scene():
    with pytest.raises(ExactPermutationError):
        exact_weak_greater_than_rich({str(i): float(i) for i in range(30)})
    values = {scene: float(i) for i, scene in enumerate(SCENE_ORDER)}; values["FMB1_W03"] = None
    result = exact_weak_greater_than_rich(values)
    assert result["p_value"] is None and result["allocation_count"] == 0
