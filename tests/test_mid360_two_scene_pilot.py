from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phase_a_harness.mid360_pilot.debug_registration import IDENTITY, transform_update
from phase_a_harness.mid360_two_scene_pilot import TWO_SCENE_FLAGS
from phase_a_harness.mid360_two_scene_pilot.pipeline import (
    REQUIRED_BAG_SHA256,
    SCENE_SPECS,
    build_preprocessing_contract,
    describe,
)
from phase_a_harness.mid360_two_scene_pilot.registration import _statistics
from phase_a_harness.mid360_two_scene_pilot.verifier import (
    EXPECTED_ROLE,
    TwoSceneVerificationError,
    _verify_csv_flags,
    _verify_flags,
    _verify_result_math,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_exact_scene_role_contract_is_frozen() -> None:
    assert EXPECTED_ROLE == {
        "mid360_20260818_203200_part1_20s.bag": ("R_TEST_01", "MAP"),
        "mid360_20260818_203200_part2_15s.bag": ("R_TEST_01", "QUERY"),
        "mid360_20260818_205021_part1_20s.bag": ("W_TEST_01", "MAP"),
        "mid360_20260818_205021_part2_15s.bag": ("W_TEST_01", "QUERY"),
    }
    assert len(REQUIRED_BAG_SHA256) == 4
    assert all(len(value) == 64 for value in REQUIRED_BAG_SHA256.values())


def test_two_scene_flags_never_authorize_formal_results() -> None:
    assert TWO_SCENE_FLAGS["MID360_TWO_SCENE_PILOT"] is True
    assert TWO_SCENE_FLAGS["INDEPENDENT_MAP_QUERY_ACQUISITION"] is True
    assert TWO_SCENE_FLAGS["SAME_BAG_MAP_QUERY"] is False
    assert TWO_SCENE_FLAGS["FORMAL_MEASUREMENT_RESULT"] is False
    assert TWO_SCENE_FLAGS["PILOT_NONFORMAL_DO_NOT_CITE"] is True


def test_prior_preprocessing_and_frozen_backends_are_bound_without_scene_tuning() -> None:
    contract = build_preprocessing_contract(
        REPOSITORY / "configs/mid360_pilot_config.json",
        REPOSITORY / "frozen_assets/backend_parameter_contract.json",
    )
    assert contract["target_voxel_size_m"] == 0.05
    assert contract["common_association_distance_m"] == 0.5
    assert contract["scene_specific_parameters"] is False
    assert contract["point_dtype"] == "little_endian_float64"
    assert contract["per_point_timestamp_order_assumed"] is False
    assert contract["backend_canonical_parameter_sha256"] == {
        "open3d": "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413",
        "pcl": "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd",
    }


def test_describe_has_requested_descriptive_quantiles() -> None:
    result = describe([1.0, 2.0, 3.0, 4.0])
    assert result == {
        "count": 4,
        "median": 2.5,
        "q25": 1.75,
        "q75": 3.25,
        "min": 1.0,
        "max": 4.0,
        "q95": pytest.approx(3.85),
    }


def _result(scene_id: str, snapshot_id: str, backend: str, value: float) -> dict:
    return {
        "scene_id": scene_id,
        "snapshot_id": snapshot_id,
        "backend": backend,
        "translation_norm_m": value,
        "translation_x_m": value,
        "translation_y_m": 0.0,
        "translation_z_m": 0.0,
        "rotation_angle_rad": value,
        "solver_success": True,
        "finite_result": True,
    }


def test_signal_direction_requires_weak_greater_for_both_backends() -> None:
    open_rows = []
    pcl_rows = []
    reassociation = []
    for scene_id, scale in (("R_TEST_01", 1.0), ("W_TEST_01", 2.0)):
        for index in range(10):
            snapshot_id = f"{scene_id}_Q{index + 1:02d}"
            for backend, destination in (
                ("open3d_point_to_plane", open_rows),
                ("pcl_point_to_plane", pcl_rows),
            ):
                destination.append(_result(scene_id, snapshot_id, backend, scale + index))
                reassociation.append(
                    {
                        "scene_id": scene_id,
                        "snapshot_id": snapshot_id,
                        "backend": backend,
                        "correspondence_turnover": 0.01 * scale + index * 0.001,
                    }
                )
    statistics = _statistics(
        open_rows,
        pcl_rows,
        reassociation,
        {"PILOT_GEOMETRY_DIRECTIONALLY_CONSISTENT": True},
    )
    assert statistics["PILOT_SIGNAL_DIRECTIONALLY_CONSISTENT"] is True
    assert statistics["PILOT_BACKEND_TREND_CONSISTENT"] is True
    assert statistics["cross_backend_translation_spearman_rho"]["pooled"] == 1.0


def _math_result() -> dict[str, str]:
    estimate = np.eye(4, dtype=np.float64)
    estimate[:3, 3] = [0.1, -0.2, 0.3]
    metrics = transform_update(IDENTITY, estimate)
    return {
        "snapshot_id": "R_TEST_01_Q01",
        "T0": json.dumps(IDENTITY.tolist()),
        "T_est": json.dumps(estimate.tolist()),
        "Delta_T": json.dumps(metrics["Delta_T"]),
        "translation_x_m": str(metrics["translation_x_m"]),
        "translation_y_m": str(metrics["translation_y_m"]),
        "translation_z_m": str(metrics["translation_z_m"]),
        "translation_norm_m": str(metrics["translation_norm_m"]),
        "rotation_angle_rad": str(metrics["rotation_angle_rad"]),
    }


def test_independent_verifier_recomputes_translation_result() -> None:
    row = _math_result()
    _verify_result_math([row], "fixture")
    row["translation_norm_m"] = "999"
    with pytest.raises(TwoSceneVerificationError, match="translation norm mismatch"):
        _verify_result_math([row], "fixture")


def test_independent_verifier_rejects_formal_flag_tampering() -> None:
    payload = dict(TWO_SCENE_FLAGS)
    _verify_flags(payload, "fixture")
    payload["FORMAL_MEASUREMENT_RESULT"] = True
    with pytest.raises(TwoSceneVerificationError, match="FORMAL_MEASUREMENT_RESULT"):
        _verify_flags(payload, "fixture")


def test_independent_verifier_rejects_csv_flag_tampering() -> None:
    row = {key: str(value).lower() for key, value in TWO_SCENE_FLAGS.items()}
    _verify_csv_flags([row], "fixture")
    row["INDEPENDENT_MAP_QUERY_ACQUISITION"] = "false"
    with pytest.raises(
        TwoSceneVerificationError, match="INDEPENDENT_MAP_QUERY_ACQUISITION"
    ):
        _verify_csv_flags([row], "fixture")


def test_transform_update_uses_generic_inverse_formula() -> None:
    t0 = np.eye(4)
    t0[0, 3] = 1.0
    estimate = np.eye(4)
    estimate[0, 3] = 1.25
    result = transform_update(t0, estimate)
    assert np.allclose(np.asarray(result["Delta_T"]), np.linalg.inv(t0) @ estimate)
    assert result["translation_norm_m"] == pytest.approx(0.25)
