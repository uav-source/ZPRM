from __future__ import annotations

from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.protocol import (
    FormalBatchError,
    QUERY_QUANTILES,
    assert_no_backend_output_before_snapshot_freeze,
    geometry_admission,
    geometry_class,
    replacement_is_allowed,
)
from phase_a_harness.mid360_pilot.split import select_query_frames


def test_rich_gate_exact_thresholds_are_inclusive() -> None:
    result = geometry_admission(0.18, 3.0, 0.90)
    assert result["final_geometry_class"] == "RICH"
    assert result["admitted"] is True


def test_weak_gate_exact_thresholds_are_inclusive() -> None:
    result = geometry_admission(0.12, 6.0, 0.80)
    assert result["final_geometry_class"] == "WEAK"
    assert result["admitted"] is True


@pytest.mark.parametrize(
    "metrics",
    [
        (0.179999, 3.0, 0.90),
        (0.18, 3.000001, 0.90),
        (0.18, 3.0, 0.899999),
        (0.120001, 6.0, 0.80),
        (0.12, 5.999999, 0.80),
        (0.12, 6.0, 0.800001),
        (0.15, 4.0, 0.85),
    ],
)
def test_intermediate_is_not_force_classified(metrics: tuple[float, float, float]) -> None:
    result = geometry_admission(*metrics)
    assert result["final_geometry_class"] == "INTERMEDIATE"
    assert result["admitted"] is False
    assert result["exclusion_reason"] == "GEOMETRY_INTERMEDIATE"


def test_geometry_gate_uses_all_three_conditions_not_or_logic() -> None:
    assert geometry_class(0.25, 8.0, 0.97) == "INTERMEDIATE"
    assert geometry_class(0.08, 2.0, 0.70) == "INTERMEDIATE"


def test_replacement_is_blind_to_icp_and_limited_to_preregistered_reasons() -> None:
    assert replacement_is_allowed("BAG_INTEGRITY", icp_result_computed=False)
    assert replacement_is_allowed("GEOMETRY_INTERMEDIATE", icp_result_computed=False)
    assert not replacement_is_allowed("ICP_CAPTURE_RADIUS", icp_result_computed=False)
    assert not replacement_is_allowed("BAG_INTEGRITY", icp_result_computed=True)


def test_query_selection_directly_reuses_fixed_quantile_implementation() -> None:
    candidates = [
        {"frame_index": index, "timestamp": float(index), "point_count": 10, "finite_point_count": 10}
        for index in range(101)
    ]
    selected = select_query_frames(candidates, QUERY_QUANTILES)
    assert [row["quantile"] for row in selected] == list(QUERY_QUANTILES)
    assert [row["selected_sequence_rank"] for row in selected] == [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    assert len({row["frame_index"] for row in selected}) == 10


def test_snapshot_freeze_refuses_preexisting_backend_output(tmp_path: Path) -> None:
    assert_no_backend_output_before_snapshot_freeze(tmp_path)
    (tmp_path / "open3d_results.csv").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(FormalBatchError, match="backend outputs"):
        assert_no_backend_output_before_snapshot_freeze(tmp_path)


def test_geometry_gate_values_are_not_pilot_heuristic_values() -> None:
    assert geometry_class(0.15, 4.0, 0.86) == "INTERMEDIATE"
    assert geometry_class(0.254, 1.79, 0.970) == "RICH"
    assert geometry_class(0.086, 8.04, 0.727) == "WEAK"
