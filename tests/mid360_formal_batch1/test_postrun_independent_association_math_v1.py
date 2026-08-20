import copy

import numpy as np
import pytest

from experiments.mid360_formal_batch1.postrun_verification.independent_association_math_v1 import (
    analyze_final,
    prepare_initial,
    prepare_target,
)
from experiments.mid360_formal_batch1.postrun_verification.independent_postrun_verifier_v1 import (
    compare_association_result,
    load_contract,
)


def plane() -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, 7)
    return np.asarray([[x, y, 0.0] for x in grid for y in grid], dtype=np.float64)


def fixture_metrics() -> dict:
    points = plane()
    target = prepare_target(points)
    initial = prepare_initial(points.copy(), target, np.eye(4))
    estimated = np.eye(4); estimated[2, 3] = 0.1
    return analyze_final(initial, estimated)


def stored_from(metrics: dict) -> dict:
    fields = (
        "initial_correspondence_count", "initial_valid_normal_correspondence_count",
        "final_correspondence_count", "final_valid_normal_correspondence_count",
        "correspondence_turnover", "accepted_source_turnover", "correspondence_count_change_ratio",
        "initial_residual_rmse", "final_residual_rmse", "residual_rmse_change",
        "median_normal_angle_change_deg", "q95_normal_angle_change_deg",
        "common_association_valid", "common_association_invalid_reason",
        "common_association_invalid_detail",
    )
    return {name: metrics[name] for name in fields}


def test_independent_point_to_plane_fixture() -> None:
    metrics = fixture_metrics()
    assert metrics["initial_correspondence_count"] == 49
    assert metrics["final_correspondence_count"] == 49
    assert metrics["initial_residual_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["final_residual_rmse"] == pytest.approx(0.1, abs=1e-12)
    assert metrics["correspondence_turnover"] == pytest.approx(0.0)
    assert metrics["accepted_source_turnover"] == pytest.approx(0.0)
    assert metrics["median_normal_angle_change_deg"] == pytest.approx(0.0)
    assert metrics["common_association_valid"] is True


@pytest.mark.parametrize(
    "field,delta,category",
    [
        ("initial_correspondence_count", 1, "association_pass"),
        ("correspondence_turnover", 1e-4, "association_pass"),
        ("accepted_source_turnover", 1e-4, "association_pass"),
        ("initial_residual_rmse", 1e-4, "association_pass"),
        ("final_residual_rmse", 1e-4, "association_pass"),
        ("median_normal_angle_change_deg", 1e-4, "association_pass"),
        ("q95_normal_angle_change_deg", 1e-4, "association_pass"),
    ],
)
def test_association_metric_tamper_fails_closed(field: str, delta: float, category: str) -> None:
    metrics = fixture_metrics(); stored = stored_from(metrics)
    stored[field] = stored[field] + delta
    result = compare_association_result(stored, metrics, load_contract()["tolerances"])
    assert not result[category]
    assert field in result["failure_field_names"]


def test_missingness_tamper_fails_closed() -> None:
    metrics = fixture_metrics(); stored = stored_from(metrics)
    stored["common_association_valid"] = False
    stored["common_association_invalid_reason"] = "OTHER"
    stored["common_association_invalid_detail"] = "fixture tamper"
    result = compare_association_result(stored, metrics, load_contract()["tolerances"])
    assert not result["missingness_pass"]


def test_no_correspondence_missingness_is_independent() -> None:
    target_points = plane()
    source = target_points + np.array([10.0, 0.0, 0.0])
    target = prepare_target(target_points)
    initial = prepare_initial(source, target, np.eye(4))
    metrics = analyze_final(initial, np.eye(4))
    assert metrics["common_association_valid"] is False
    assert metrics["common_association_invalid_reason"] == "NO_INITIAL_CORRESPONDENCE"
