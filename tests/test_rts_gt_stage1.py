from __future__ import annotations

import numpy as np

from phase_a_harness.real_data_preparation.rts_gt import (
    OverlapContract,
    compute_overlap_pair,
    resample_gap_aware,
    select_best_pair,
)


def test_gap_aware_resampling_never_bridges_native_outage() -> None:
    trajectory = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0],
            [1.1, 10.0, 0.0, 0.0],
            [1.2, 10.1, 0.0, 0.0],
        ]
    )
    contract = OverlapContract(resample_rate_hz=10.0, maximum_native_gap_s=0.15)
    rows = resample_gap_aware(trajectory, contract)
    assert rows.shape == (4, 5)
    assert rows[:, 4].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert not np.any((rows[:, 0] > 0.1) & (rows[:, 0] < 1.1))


def test_overlap_thresholds_and_nonoverlapping_windows_are_frozen() -> None:
    timestamps = np.arange(0.0, 200.1, 0.1)
    trajectory = np.column_stack((timestamps, timestamps * 0.01, np.zeros_like(timestamps), np.zeros_like(timestamps)))
    result = compute_overlap_pair(trajectory, trajectory)
    assert result["overlap_status"] == "PASS"
    assert result["total_covered_duration_s"] >= 200.0
    assert result["coverage_fraction"] == 1.0
    assert result["eligible_nonoverlapping_5s_interval_count"] >= 40


def test_pair_sorting_uses_total_fraction_intervals_then_ids() -> None:
    rows = [
        {
            "map_experiment_id": "b",
            "query_experiment_id": "q",
            "overlap_status": "PASS",
            "total_covered_duration_s": 200.0,
            "coverage_fraction": 1.0,
            "eligible_nonoverlapping_5s_interval_count": 40,
        },
        {
            "map_experiment_id": "a",
            "query_experiment_id": "q",
            "overlap_status": "PASS",
            "total_covered_duration_s": 200.0,
            "coverage_fraction": 1.0,
            "eligible_nonoverlapping_5s_interval_count": 40,
        },
    ]
    assert select_best_pair(rows)["map_experiment_id"] == "a"
