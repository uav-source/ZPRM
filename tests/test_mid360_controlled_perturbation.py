from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mid360_controlled_perturbation.benchmark import (
    BACKEND_CONTRACT_SHA256,
    BAG_ROOT,
    EXPECTED_BAGS,
    EXPECTED_TARGETS,
    MAGNITUDES_M,
    ZERO_RUNTIME,
    authenticate_bags,
    authenticate_contract,
    build_run_id,
    construct_translation_perturbation,
    deterministic_eigendirections,
    ensure_fresh_output,
    identical_backend_input_pair,
    pose_error,
    pose_recovered,
    verify_sha256_file,
)
from phase_a_harness.mid360_pilot.bag_reader import PilotBagError


REPOSITORY = Path(__file__).resolve().parents[1]


def test_frozen_bag_sha_verification() -> None:
    assert authenticate_bags(BAG_ROOT) == EXPECTED_BAGS


def test_frozen_target_sha_verification() -> None:
    for scene_id, expected in EXPECTED_TARGETS.items():
        path = ZERO_RUNTIME / "target_maps" / scene_id / "target_points.npy"
        assert verify_sha256_file(path, expected["npy_sha256"]) == expected["npy_sha256"]


def test_backend_contract_sha_verification() -> None:
    path, contract = authenticate_contract(REPOSITORY)
    assert verify_sha256_file(path, BACKEND_CONTRACT_SHA256) == BACKEND_CONTRACT_SHA256
    assert contract["backend_parameter_difference_count"] == 0


def test_weak_strong_eigenvector_ordering() -> None:
    matrix = np.diag([0.1, 0.4, 0.9])
    result = deterministic_eigendirections(matrix)
    assert np.allclose(result["eigenvalues"], [0.1, 0.4, 0.9])
    assert np.allclose(result["weak"], [1.0, 0.0, 0.0])
    assert np.allclose(result["strong"], [0.0, 0.0, 1.0])


def test_eigenvector_normalization_and_deterministic_sign() -> None:
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [-0.6, 0.0, -0.8], [0.8, 0.0, -0.6]],
        dtype=np.float64,
    )
    matrix = rotation @ np.diag([0.1, 0.4, 0.9]) @ rotation.T
    result = deterministic_eigendirections(matrix)
    for name in ("weak", "strong"):
        vector = result[name]
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-12)
        assert vector[np.argmax(np.abs(vector))] >= 0.0


@pytest.mark.parametrize("magnitude", MAGNITUDES_M)
def test_perturbation_magnitude_exactness(magnitude: float) -> None:
    t_star = np.eye(4)
    _, initial, _ = construct_translation_perturbation(
        t_star, np.asarray([2.0, -3.0, 6.0]), magnitude, 1
    )
    relative = initial @ np.linalg.inv(t_star)
    assert np.linalg.norm(relative[:3, 3]) == pytest.approx(magnitude, abs=1e-12)


def test_positive_negative_perturbation_symmetry() -> None:
    direction = np.asarray([0.2, -0.3, 0.4])
    _, positive, positive_vector = construct_translation_perturbation(
        np.eye(4), direction, 0.01, 1
    )
    _, negative, negative_vector = construct_translation_perturbation(
        np.eye(4), direction, 0.01, -1
    )
    assert np.allclose(positive_vector, -negative_vector, atol=1e-15, rtol=0.0)
    assert np.allclose(positive[:3, 3], -negative[:3, 3], atol=1e-15, rtol=0.0)


def test_source_to_target_left_perturbation_convention() -> None:
    # A nonidentity reference makes left-vs-right multiplication observable.
    t_star = np.eye(4)
    t_star[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    t_star[:3, 3] = [1.0, 2.0, 3.0]
    _, initial, vector = construct_translation_perturbation(
        t_star, np.asarray([1.0, 0.0, 0.0]), 0.01, 1
    )
    relative = initial @ np.linalg.inv(t_star)
    assert np.allclose(relative[:3, 3], [0.01, 0.0, 0.0], atol=1e-12, rtol=0.0)
    assert np.allclose(vector, [0.01, 0.0, 0.0], atol=1e-12, rtol=0.0)
    assert pose_error(initial, t_star)["translation_error_m"] == pytest.approx(
        0.01, abs=1e-12
    )


def test_open3d_pcl_identical_input_sha_contract() -> None:
    common = {
        "scene_id": "R_TEST_01",
        "snapshot_id": "R_TEST_01_Q01",
        "source_file_sha256": "1" * 64,
        "source_array_sha256": "2" * 64,
        "target_file_sha256": "3" * 64,
        "target_array_sha256": "4" * 64,
        "backend_contract_sha256": BACKEND_CONTRACT_SHA256,
        "direction_class": "weak",
        "direction_vector_x": 1.0,
        "direction_vector_y": 0.0,
        "direction_vector_z": 0.0,
        "perturbation_sign": 1,
        "perturbation_magnitude_m": 0.01,
        "perturbation_vector_x_m": 0.01,
        "perturbation_vector_y_m": 0.0,
        "perturbation_vector_z_m": 0.0,
        "T_star": np.eye(4).tolist(),
        "T_initial": np.eye(4).tolist(),
    }
    assert identical_backend_input_pair(
        {**common, "backend": "open3d"}, {**common, "backend": "pcl"}
    )
    assert not identical_backend_input_pair(
        {**common, "backend": "open3d"},
        {**common, "backend": "pcl", "source_array_sha256": "9" * 64},
    )


def test_deterministic_run_ids() -> None:
    first = build_run_id("R_TEST_01_Q01", "weak", 0.03, -1, "open3d")
    second = build_run_id("R_TEST_01_Q01", "weak", 0.03, -1, "open3d")
    assert first == second == "CP-R_TEST_01_Q01-W-30MM-NEG-O3D"


def test_recovery_threshold_correctness() -> None:
    assert pose_recovered(0.005, 0.2)
    assert not pose_recovered(0.0050000001, 0.2)
    assert not pose_recovered(0.005, 0.2000000001)


def test_no_overwrite_of_previous_or_existing_outputs(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "marker.json"
    marker.write_text(json.dumps({"preserve": True}), encoding="utf-8")
    with pytest.raises(PilotBagError, match="refusing overwrite"):
        ensure_fresh_output(existing)
    assert json.loads(marker.read_text(encoding="utf-8")) == {"preserve": True}

