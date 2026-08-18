from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import phase_a_harness.mid360_pilot.debug_registration as debug
from phase_a_harness.mid360_pilot import NONFORMAL_MARKER
from phase_a_harness.mid360_pilot.bag_reader import PilotBagError, sha256_file
from phase_a_harness.mid360_pilot.pilot_pipeline import (
    DEFAULT_BACKEND_PARAMETER_CONTRACT,
    DEFAULT_PCL_EXECUTABLE,
)


@pytest.fixture(scope="module")
def parameter_contract() -> dict[str, object]:
    return json.loads(DEFAULT_BACKEND_PARAMETER_CONTRACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def convention(parameter_contract: dict[str, object]) -> dict[str, object]:
    before_contract = sha256_file(DEFAULT_BACKEND_PARAMETER_CONTRACT)
    before_pcl = sha256_file(DEFAULT_PCL_EXECUTABLE)
    result = debug.verify_transform_convention(
        parameter_contract, DEFAULT_PCL_EXECUTABLE
    )
    assert sha256_file(DEFAULT_BACKEND_PARAMETER_CONTRACT) == before_contract
    assert sha256_file(DEFAULT_PCL_EXECUTABLE) == before_pcl
    return result


def test_pilot_target_sha_is_fixed() -> None:
    assert debug.EXPECTED_TARGET_NPY_SHA256 == (
        "7f01864633c56f175ece27f5996ab6df6f30457409ac5ebeb1a759311670f60f"
    )


def test_ten_query_selection_is_fixed() -> None:
    assert len(debug.EXPECTED_QUERY_TIMESTAMPS) == 10
    assert list(debug.EXPECTED_QUERY_TIMESTAMPS) == sorted(
        debug.EXPECTED_QUERY_TIMESTAMPS
    )
    assert debug.EXPECTED_QUERY_TIMESTAMPS[0] == 1786887982.8384867
    assert debug.EXPECTED_QUERY_TIMESTAMPS[-1] == 1786887988.1382859


def test_canonical_source_and_target_are_finite_float64_c_contiguous(
    tmp_path: Path,
) -> None:
    points = debug.canonical_points(np.arange(60, dtype=np.float32).reshape(20, 3))
    assert points.dtype == np.dtype("<f8")
    assert points.flags.c_contiguous
    assert np.isfinite(points).all()
    path = tmp_path / "points.npy"
    np.save(path, points, allow_pickle=False)
    np.testing.assert_array_equal(debug.load_canonical_npy(path), points)


def test_nonfinite_canonical_input_is_rejected() -> None:
    with pytest.raises(PilotBagError):
        debug.canonical_points(np.asarray([[np.nan, 0.0, 0.0]]))


def test_open3d_pcl_input_sha_is_identical() -> None:
    queries = [np.full((50, 3), index, dtype=np.float64) for index in range(10)]
    rows = debug.build_shared_input_trials(queries, np.ones((100, 3)))
    assert len(rows) == 20
    for query_index in range(10):
        pair = [row for row in rows if row["query_index"] == query_index]
        assert pair[0]["source_array_sha256"] == pair[1]["source_array_sha256"]
        assert pair[0]["target_array_sha256"] == pair[1]["target_array_sha256"]


def test_t0_is_exactly_identity() -> None:
    rows = debug.build_shared_input_trials(
        [np.ones((50, 3)) for _ in range(10)], np.ones((100, 3))
    )
    assert all(row["T0"] == np.eye(4).tolist() for row in rows)


def test_transform_update_uses_general_inverse_formula() -> None:
    t0 = np.eye(4)
    t0[:3, 3] = [1.0, 2.0, 3.0]
    delta = np.eye(4)
    delta[:3, 3] = [0.1, -0.2, 0.3]
    estimated = t0 @ delta
    result = debug.transform_update(t0, estimated)
    np.testing.assert_allclose(result["Delta_T"], delta, atol=1.0e-15)


def test_translation_extraction() -> None:
    estimated = np.eye(4)
    estimated[:3, 3] = [0.03, -0.04, 0.12]
    result = debug.transform_update(np.eye(4), estimated)
    assert result["translation_x_m"] == pytest.approx(0.03)
    assert result["translation_y_m"] == pytest.approx(-0.04)
    assert result["translation_z_m"] == pytest.approx(0.12)
    assert result["translation_norm_m"] == pytest.approx(0.13)


def test_reflection_safe_rotation_metric() -> None:
    estimated = np.eye(4)
    estimated[:3, :3] = Rotation.from_rotvec([0.0, 0.0, 0.2]).as_matrix()
    result = debug.transform_update(np.eye(4), estimated)
    assert result["rotation_angle_rad"] == pytest.approx(0.2)
    reflected = np.eye(4)
    reflected[0, 0] = -1.0
    with pytest.raises(PilotBagError):
        debug.transform_update(np.eye(4), reflected)


def test_transform_convention_is_source_to_target(convention: dict[str, object]) -> None:
    assert convention["status"] == "PASS"
    assert convention["transform_convention_verified"] is True
    assert convention["definition"] == "T_est maps source/query points into target/map frame"


def test_open3d_backend_runs_on_tiny_fixture(convention: dict[str, object]) -> None:
    row = convention["backends"]["open3d"]
    assert row["solver_success"] is True
    assert row["finite_result"] is True
    assert row["source_to_target_verified"] is True


def test_pcl_backend_runs_on_tiny_fixture(convention: dict[str, object]) -> None:
    row = convention["backends"]["pcl"]
    assert row["solver_success"] is True
    assert row["finite_result"] is True
    assert row["source_to_target_verified"] is True


def test_solver_failure_is_retained() -> None:
    base = {
        "query_id": "Q01",
        "backend": "open3d_point_to_plane",
        **debug.DEBUG_FLAGS,
    }
    row = debug._failure_result(base, RuntimeError("do not retry"))
    assert row["solver_success"] is False
    assert row["finite_result"] is False
    assert "do not retry" in row["failure_reason"]
    assert row["query_id"] == "Q01"


def test_no_query_reselection_vocabulary_exists() -> None:
    source = Path(debug.__file__).read_text(encoding="utf-8")
    assert "EXPECTED_QUERY_TIMESTAMPS" in source
    assert "frozen debug input manifest is incomplete or changed" in source
    assert "best query" not in source.lower()


def test_reassociation_calls_common_frozen_definition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "source.npy"
    np.save(source_path, np.ones((20, 3)), allow_pickle=False)
    calls = {"prepare": 0, "analyze": 0}

    def prepare(*args: object, **kwargs: object) -> object:
        calls["prepare"] += 1
        return object()

    def analyze(*args: object, identifiers: dict[str, object], **kwargs: object) -> dict[str, object]:
        calls["analyze"] += 1
        return {**identifiers, "correspondence_turnover": 0.1}

    monkeypatch.setattr(debug, "prepare_common_association_context", prepare)
    monkeypatch.setattr(debug, "safe_analyze_estimated_transform", analyze)
    manifest = [{"query_id": "Q01", "query_timestamp": "1.0", "source_path": str(source_path)}]
    results = [
        {"query_id": "Q01", "backend": backend, "T_est": np.eye(4).tolist()}
        for backend in ("open3d_point_to_plane", "pcl_point_to_plane")
    ]
    rows = debug._reassociation_rows(manifest, np.ones((50, 3)), results)
    assert calls == {"prepare": 1, "analyze": 2}
    assert len(rows) == 2


def test_nonformal_flags_and_formal_authorization_false() -> None:
    artifact = debug.mark_debug_artifact({"status": "PASS"})
    assert artifact[NONFORMAL_MARKER] is True
    assert artifact["FORMAL_MEASUREMENT_RESULT"] is False
    assert artifact["SAME_BAG_MAP_QUERY"] is True
    assert artifact["INDEPENDENT_ACQUISITION"] is False


def test_backend_parameter_contract_hashes_are_frozen(
    parameter_contract: dict[str, object]
) -> None:
    for backend in ("open3d", "pcl"):
        encoded = json.dumps(
            parameter_contract[backend]["parameters"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        assert hashlib.sha256(encoded).hexdigest() == parameter_contract[backend][
            "canonical_sha256"
        ]


def test_boreas_frozen_assets_are_not_debug_targets() -> None:
    source = Path(debug.__file__).read_text(encoding="utf-8")
    assert "real_data_boreas" not in source
    assert "PRIMARY_PAIR" not in source
    assert "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED" not in source
