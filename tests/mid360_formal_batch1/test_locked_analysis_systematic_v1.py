from experiments.mid360_formal_batch1.locked_analysis.contract_v1 import SCENE_ORDER
from experiments.mid360_formal_batch1.locked_analysis.systematic_component_v1 import (
    scene_systematic_fraction, station_systematic_fraction,
    systematic_weak_rich_descriptive,
)


def test_station_fraction_complete_and_all_zero_rule():
    result = station_systematic_fraction([[1.0, 0.0, 0.0]] * 10)
    assert result["systematic_fraction"] == 1.0
    zero = station_systematic_fraction([[0.0, 0.0, 0.0]] * 10)
    assert zero["systematic_fraction"] == 0.0
    assert zero["status"] == "ALL_ZERO_UPDATES_DEFINED_ZERO_SYSTEMATIC_FRACTION"


def test_station_and_scene_incomplete_are_null():
    result = station_systematic_fraction([[1, 0, 0]] * 9 + [None])
    assert result["systematic_fraction"] is None
    scene = scene_systematic_fraction({"S01": 0.1, "S02": 0.2, "S03": None})
    assert scene["scene_systematic_fraction"] is None


def test_weak_rich_is_descriptive_without_p_value():
    result = systematic_weak_rich_descriptive(
        {scene: float(i) for i, scene in enumerate(SCENE_ORDER)}
    )
    assert result["role"] == "SECONDARY_DESCRIPTIVE_MECHANISTIC_COMPARISON"
    assert result["formal_p_value_defined"] is False
    assert "p_value" not in result
