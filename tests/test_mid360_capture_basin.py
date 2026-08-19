from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.mid360_capture_basin import capture_basin as capture
from experiments.mid360_controlled_perturbation import benchmark as controlled
from phase_a_harness.mid360_pilot.bag_reader import PilotBagError, sha256_file


REPOSITORY = Path(__file__).resolve().parents[1]


def test_previous_pilot_directory_is_authenticated_unchanged() -> None:
    inventory = capture.authenticate_previous_pilot(REPOSITORY)
    assert inventory["manifest.json"] == capture.PREVIOUS_MANIFEST_SHA256
    assert inventory["runs.csv"] == capture.PREVIOUS_RUNS_SHA256


def test_frozen_target_sha_verification() -> None:
    for scene_id, expected in controlled.EXPECTED_TARGETS.items():
        path = controlled.ZERO_RUNTIME / "target_maps" / scene_id / "target_points.npy"
        assert sha256_file(path) == expected["npy_sha256"]


def test_backend_contract_verification() -> None:
    path, contract = controlled.authenticate_contract(REPOSITORY)
    assert sha256_file(path) == controlled.BACKEND_CONTRACT_SHA256
    assert contract["backend_parameter_difference_count"] == 0


def test_frozen_open3d_version_verification() -> None:
    assert capture.validate_open3d_version() == "0.19.0+b012259"


def test_deterministic_direction_sign_canonicalization() -> None:
    positive = capture.canonicalize_direction_sign(np.asarray([-1.0, 0.2, 0.3]))
    negative = capture.canonicalize_direction_sign(np.asarray([1.0, -0.2, -0.3]))
    assert np.allclose(positive, negative, atol=1e-15, rtol=0.0)
    assert positive[np.argmax(np.abs(positive))] >= 0.0


def test_weak_strong_direction_normalization() -> None:
    for vector in (np.asarray([2.0, 0.0, 0.0]), np.asarray([0.0, -3.0, 4.0])):
        canonical = capture.canonicalize_direction_sign(vector)
        assert np.linalg.norm(canonical) == pytest.approx(1.0, abs=1e-12)


def test_coarse_magnitude_schedule_is_exact() -> None:
    assert capture.COARSE_MAGNITUDES_M == (
        0.05,
        0.10,
        0.20,
        0.40,
        0.80,
        1.20,
        1.60,
    )
    assert (
        20
        * len(capture.DIRECTION_CLASSES)
        * len(capture.COARSE_MAGNITUDES_M)
        * len(capture.SIGNS)
        * len(capture.BACKENDS)
        == capture.EXPECTED_BASE_COARSE_ROWS
    )


def test_auto_extension_schedule_is_exact() -> None:
    assert capture.AUTO_EXTENSION_MAGNITUDES_M == (2.40, 3.20, 4.80, 6.40)


def test_maximum_translation_perturbation_cap() -> None:
    assert capture.magnitude_token(6.4) == "M6400000UM"
    with pytest.raises(PilotBagError, match="exceeds frozen cap"):
        capture.magnitude_token(6.4000001)


def test_recovery_threshold_is_unchanged() -> None:
    assert controlled.RECOVERY_TRANSLATION_M == 0.005
    assert controlled.RECOVERY_ROTATION_DEG == 0.2
    assert controlled.pose_recovered(0.005, 0.2)
    assert not controlled.pose_recovered(0.0050000001, 0.2)
    assert not controlled.pose_recovered(0.005, 0.2000000001)


def test_monotonicity_violation_detection() -> None:
    assert capture.detect_non_monotonic_recovery([True, False, True])
    assert not capture.detect_non_monotonic_recovery([True, True, False, False])
    classified = capture.classify_recovery_profile(
        [(0.05, True), (0.10, False), (0.20, True)]
    )
    assert classified["CAPTURE_RADIUS_AMBIGUOUS"] is True
    assert classified["bracket"] is None


def test_boundary_bracket_correctness() -> None:
    classified = capture.classify_recovery_profile(
        [(0.05, True), (0.10, True), (0.20, False), (0.40, False)]
    )
    assert classified["bracket"] == (0.10, 0.20)
    assert classified["largest_recovered_coarse_magnitude"] == 0.10
    assert classified["smallest_failed_coarse_magnitude"] == 0.20


def test_bisection_stopping_rule() -> None:
    step = capture.bisection_step(0.10, 0.20)
    assert step["stop"] is False
    assert step["mid"] == pytest.approx(0.15)
    assert step["resolution_m"] == pytest.approx(0.10)
    assert capture.bisection_step(0.10, 0.125)["stop"] is True
    assert capture.BISECTION_RESOLUTION_M == 0.025
    assert capture.MAX_BISECTION_ITERATIONS == 8


def test_left_censoring() -> None:
    classified = capture.classify_recovery_profile(
        [(0.05, False), (0.10, False), (0.20, False)]
    )
    assert classified["LEFT_CENSORED"] is True
    assert classified["RIGHT_CENSORED"] is False


def test_right_censoring_only_at_frozen_cap() -> None:
    profile = [
        (magnitude, True)
        for magnitude in (
            *capture.COARSE_MAGNITUDES_M,
            *capture.AUTO_EXTENSION_MAGNITUDES_M,
        )
    ]
    classified = capture.classify_recovery_profile(profile)
    assert classified["RIGHT_CENSORED"] is True
    assert classified["largest_recovered_coarse_magnitude"] == 6.4


def test_open3d_pcl_inputs_are_identical() -> None:
    common = {
        "scene_id": "R_TEST_01",
        "snapshot_id": "R_TEST_01_Q01",
        "source_file_sha256": "1" * 64,
        "source_array_sha256": "2" * 64,
        "target_file_sha256": "3" * 64,
        "target_array_sha256": "4" * 64,
        "backend_contract_sha256": controlled.BACKEND_CONTRACT_SHA256,
        "direction_class": "weak",
        "direction_vector_x": 1.0,
        "direction_vector_y": 0.0,
        "direction_vector_z": 0.0,
        "perturbation_sign": 1,
        "perturbation_magnitude_m": 0.10,
        "perturbation_vector_x_m": 0.10,
        "perturbation_vector_y_m": 0.0,
        "perturbation_vector_z_m": 0.0,
        "T_star": np.eye(4).tolist(),
        "T_initial": np.eye(4).tolist(),
    }
    assert controlled.identical_backend_input_pair(
        {**common, "backend": "open3d"},
        {**common, "backend": "pcl"},
    )


def test_initial_overlap_calculation_is_backend_independent_and_repeatable() -> None:
    target = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    source = np.asarray([[0.1, 0.0, 0.0], [1.4, 0.0, 0.0], [2.0, 0.0, 0.0]])
    calculator = capture.OverlapCalculator({"scene": target})
    first = calculator.calculate("scene", source, np.eye(4))
    second = calculator.calculate("scene", source, np.eye(4))
    assert first == second
    assert first["initial_correspondence_count"] == 2
    assert first["initial_correspondence_fraction"] == pytest.approx(2.0 / 3.0)


def test_capture_run_ids_are_deterministic() -> None:
    first = capture.build_capture_run_id(
        "COARSE", "R_TEST_01_Q01", "weak", 0.10, -1, "open3d"
    )
    second = capture.build_capture_run_id(
        "COARSE", "R_TEST_01_Q01", "weak", 0.10, -1, "open3d"
    )
    assert first == second == "CB-COARSE-R_TEST_01_Q01-W-M0100000UM-NEG-O3D"


def test_boundary_run_id_includes_iteration() -> None:
    assert capture.build_capture_run_id(
        "BOUNDARY", "W_TEST_01_Q10", "strong", 0.175, 1, "pcl", 3
    ) == "CB-BOUNDARY-W_TEST_01_Q10-S-M0175000UM-POS-PCL-I03"


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "marker"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(PilotBagError, match="refusing overwrite"):
        capture.ensure_fresh_output(existing)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_previous_result_directory_is_rejected_as_output() -> None:
    with pytest.raises(PilotBagError, match="refusing overwrite"):
        capture.ensure_fresh_output(REPOSITORY / capture.PREVIOUS_RESULT_RELATIVE)


def test_reused_input_allows_only_float_roundtrip_noise() -> None:
    reference = np.eye(4)
    roundtrip = reference.copy()
    roundtrip[0, 3] = 1.0e-17
    changed = reference.copy()
    changed[0, 3] = 1.0e-10
    assert capture._reused_numeric_input_matches(reference, roundtrip)
    assert not capture._reused_numeric_input_matches(reference, changed)
