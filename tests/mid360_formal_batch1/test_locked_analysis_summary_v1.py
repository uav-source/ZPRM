import numpy as np

from experiments.mid360_formal_batch1.locked_analysis.descriptive_summary_v1 import (
    summarize_scene, summarize_station,
)


def outcomes(values, endpoint="translation_norm_m"):
    return [
        {"trial_id": f"T{i}", "classification": "FINITE_SCIENTIFIC_VALUE",
         "resolved_infrastructure_attempt_n": 0,
         "authoritative_row": {endpoint: value, "solver_status": "CONVERGED"}}
        for i, value in enumerate(values)
    ]


def test_station_10_of_10_linear_quantiles():
    summary = summarize_station(outcomes(range(10)), "translation_norm_m")
    assert summary["formal_summary_status"] == "FORMAL_STATION_SUMMARY_DEFINED_COMPLETE_10_OF_10"
    assert summary["formal_statistics"] == {
        "median": 4.5, "q25": 2.25, "q75": 6.75,
        "q95": float(np.quantile(np.arange(10), .95, method="linear")),
    }


def test_scene_uses_direct_30_snapshots_and_three_station_medians():
    groups = {"S01": outcomes(range(10)), "S02": outcomes(range(100, 110)),
              "S03": outcomes(range(200, 210))}
    summary = summarize_scene(groups, "translation_norm_m")
    assert summary["planned_n"] == 30 and summary["defined_station_summary_n"] == 3
    assert summary["formal_statistics"]["median"] == 104.5
    assert summary["station_medians"] == {"S01": 4.5, "S02": 104.5, "S03": 204.5}
    assert summary["scene_statistics_are_direct_30_snapshot_statistics"] is True


def test_incomplete_available_case_is_never_formal():
    rows = outcomes(range(10)); rows[-1]["classification"] = "SCIENTIFIC_UNDEFINED_NONFINITE"
    summary = summarize_station(rows, "translation_norm_m")
    assert summary["formal_statistics"]["median"] is None
    assert summary["available_case"]["finite_n"] == 9
    assert summary["available_case"]["may_enter_inference"] is False
