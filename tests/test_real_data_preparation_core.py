from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phase_a_harness.real_data_preparation.guard import (
    NoRegistrationGuard,
    RegistrationForbiddenError,
    assert_preparation_sources_are_safe,
)
from phase_a_harness.real_data_preparation.eligibility import (
    assert_byte_identical_backend_bundle,
    mission_reference_eligible,
)
from phase_a_harness.real_data_preparation.geodesy import wgs84_geodetic_to_ecef
from phase_a_harness.real_data_preparation.io import atomic_write_json, canonical_json_bytes
from phase_a_harness.real_data_preparation.manifest import (
    BACKEND_FILE_SHA256,
    ManifestVerificationError,
    OPEN3D_PARAMETERS_SHA256,
    PCL_PARAMETERS_SHA256,
    build_frozen_manifest,
    verify_frozen_manifest,
)
from phase_a_harness.real_data_preparation.overlap import OverlapContract, compute_gt_only_overlap
from phase_a_harness.real_data_preparation.selection import (
    assert_map_query_disjoint,
    geometry_only_view,
    select_five_scan_timestamps,
    select_scene_intervals,
)
from phase_a_harness.real_data_preparation.transforms import (
    compose_world_sensor,
    deskew_with_independent_reference,
    interpolate_pose,
    transform_from_xyzw,
)
from phase_a_harness.real_data_preparation.uncertainty import conservative_uncertainty
from phase_a_harness.real_data_preparation.workflow import IILABS_ROSBAG_RANGE_AUDIT
from phase_a_harness.real_data_protocol import SNAPSHOT_SELECTION_FIELDS


REPOSITORY = Path(__file__).resolve().parents[1]


BACKEND_BINDINGS = {
    "backend_parameter_contract_file_sha256": BACKEND_FILE_SHA256,
    "open3d_parameter_canonical_sha256": OPEN3D_PARAMETERS_SHA256,
    "pcl_parameter_canonical_sha256": PCL_PARAMETERS_SHA256,
}


def _fake_open3d() -> SimpleNamespace:
    registration = SimpleNamespace(
        registration_icp=lambda *args: "forbidden",
        registration_generalized_icp=lambda *args: "forbidden",
        harmless_constant=1,
    )
    return SimpleNamespace(pipelines=SimpleNamespace(registration=registration))


def test_no_registration_guard_blocks_open3d_icp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    module = _fake_open3d()
    original = module.pipelines.registration.registration_icp
    with NoRegistrationGuard(open3d_module=module) as guard:
        with pytest.raises(RegistrationForbiddenError):
            module.pipelines.registration.registration_icp(None, None, 0.5, np.eye(4))
        attestation = guard.attestation(Path("/definitely/absent"))
        assert attestation["open3d_registration_call_count"] == 1
        assert attestation["pass"] is False
    assert module.pipelines.registration.registration_icp is original


def test_no_registration_guard_blocks_pcl_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        with pytest.raises(RegistrationForbiddenError):
            subprocess.run(["bin/pcl_point_to_plane_cli", "--config", "x"])
        attestation = guard.attestation(Path("/definitely/absent"))
        assert attestation["pcl_cli_invocation_count"] == 1
        assert attestation["other_registration_process_count"] == 1
        assert attestation["pass"] is False


def test_no_registration_guard_does_not_false_positive_grandtour(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard():
        completed = subprocess.run(["/bin/true", "grandtour_probe"], check=True)
    assert completed.returncode == 0


def test_no_registration_guard_requires_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", raising=False)
    with pytest.raises(RegistrationForbiddenError):
        NoRegistrationGuard().__enter__()


def test_static_preparation_source_audit_passes() -> None:
    report = assert_preparation_sources_are_safe(
        REPOSITORY / "src/phase_a_harness/real_data_preparation"
    )
    assert report["pass"] is True


def _trajectory(duration: int, offset: float = 0.0) -> np.ndarray:
    time = np.arange(duration + 1, dtype=np.float64)
    return np.column_stack((time, time * 0.1 + offset, np.zeros_like(time), np.zeros_like(time)))


def test_gt_only_overlap_is_deterministic() -> None:
    contract = OverlapContract(min_total_covered_duration_s=10, min_nonoverlapping_5s_intervals=2)
    first = compute_gt_only_overlap(_trajectory(20), _trajectory(20), common_world_frame_proven=True, contract=contract)
    second = compute_gt_only_overlap(_trajectory(20), _trajectory(20), common_world_frame_proven=True, contract=contract)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["overlap_status"] == "PASS"


def test_wgs84_ecef_is_deterministic_and_metric() -> None:
    first = wgs84_geodetic_to_ecef(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    second = wgs84_geodetic_to_ecef(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert np.array_equal(first, second)
    assert np.allclose(first[0], [6378137.0, 0.0, 0.0])


def test_iilabs_sequence_local_tum_is_not_treated_as_common_world() -> None:
    report = compute_gt_only_overlap(_trajectory(20), _trajectory(20), common_world_frame_proven=False)
    assert report == {
        "eligibility_status": "FAIL",
        "failure_reason": "UNPROVEN_CROSS_SEQUENCE_WORLD_FRAME",
        "overlap_status": "NOT_COMPUTABLE",
    }


def test_iilabs_raw_bag_index_audit_has_no_mocap_world_topic() -> None:
    assert IILABS_ROSBAG_RANGE_AUDIT["mocap_or_optitrack_topic_count"] == 0
    for sequence in ("nav_a_diff", "nav_a_omni"):
        row = IILABS_ROSBAG_RANGE_AUDIT[sequence]
        assert set(row["topic_message_counts"]) == {
            "/eve/motors_enc",
            "/tf_static",
            "/tf",
            "/eve/imu/data",
            "/eve/scan",
            "/eve/odom",
            "/eve/ouster/imu",
            "/eve/ouster/points",
        }
        assert row["first_odometry"]["translation_m"] == [0.0, 0.0, 0.0]


def test_transform_direction_is_world_base_then_base_sensor() -> None:
    world_base = np.eye(4)
    world_base[0, 3] = 3.0
    base_sensor = np.eye(4)
    base_sensor[1, 3] = 2.0
    result = compose_world_sensor(world_base, base_sensor)
    assert np.array_equal(result[:3, 3], [3.0, 2.0, 0.0])


def test_quaternion_is_xyzw() -> None:
    transform = transform_from_xyzw(np.zeros(3), np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]))
    assert np.allclose(transform[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15)


def test_timestamp_interpolation_translation_and_rotation() -> None:
    result = interpolate_pose(
        np.array([0.0, 2.0]),
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]),
        1.0,
        max_gap_s=2.0,
    )
    assert np.allclose(result[:3, 3], [1.0, 0.0, 0.0])
    assert np.allclose(result[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15)


def test_deskew_rejects_non_independent_reference() -> None:
    reference = {
        "lineage": "LIDAR_ODOMETRY",
        "timestamps": [0.0, 1.0],
        "translations": [[0, 0, 0], [0, 0, 0]],
        "quaternions_xyzw": [[0, 0, 0, 1], [0, 0, 0, 1]],
    }
    with pytest.raises(PermissionError):
        deskew_with_independent_reference(np.zeros((1, 3)), np.array([0.5]), reference, 0.5, max_gap_s=1.0)


def test_deskew_accepts_independent_reference_only() -> None:
    reference = {
        "lineage": "INDEPENDENT_NON_LIDAR_6DOF_REFERENCE",
        "timestamps": [0.0, 1.0],
        "translations": [[0, 0, 0], [1, 0, 0]],
        "quaternions_xyzw": [[0, 0, 0, 1], [0, 0, 0, 1]],
    }
    output = deskew_with_independent_reference(np.array([[1.0, 0.0, 0.0]]), np.array([0.25]), reference, 0.5, max_gap_s=1.0)
    assert output.dtype == np.dtype("<f8")
    assert output.flags.c_contiguous
    assert np.allclose(output[0], [0.75, 0.0, 0.0])


def test_query_scan_cannot_enter_map() -> None:
    with pytest.raises(ValueError, match="contaminate"):
        assert_map_query_disjoint(["map-1", "shared"], ["query-1", "shared"])


def test_disjoint_map_query_sources_pass() -> None:
    assert_map_query_disjoint(["map-1"], ["query-1"])


def _metric(**extra: object) -> dict[str, object]:
    return {
        "initial_correspondence_count": 1000,
        "initial_valid_normal_correspondence_count": 900,
        "lambda_min_trans": 1.0,
        "lambda_mid_trans": 2.0,
        "lambda_max_trans": 3.0,
        "normalized_lambda_min_trans": 0.1,
        "normalized_lambda_mid_trans": 0.2,
        "normalized_lambda_max_trans": 0.7,
        "condition_number_trans": 3.0,
        "spectral_entropy_trans": 0.8,
        **extra,
    }


def test_geometry_selector_rejects_residual_or_error_fields() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        geometry_only_view(_metric(initial_residual_rmse=0.2))
    with pytest.raises(ValueError, match="forbidden"):
        geometry_only_view(_metric(registration_error=0.2))


def _intervals() -> list[dict[str, object]]:
    return [
        {
            "interval_id": f"i{index:02d}",
            "interval_start_time": float(index * 20),
            "interval_end_time": float(index * 20 + 5),
            "center_world_position": [float(index * 2), 0.0, 0.0],
            "normalized_lambda_min_trans": float(index),
            "condition_number_trans": float(100 - index),
            "spectral_entropy_trans": float(index) / 100.0,
        }
        for index in range(20)
    ]


def test_weak_rich_selection_is_deterministic_and_disjoint() -> None:
    first = select_scene_intervals(_intervals())
    second = select_scene_intervals(reversed(_intervals()))
    assert first == second
    weak = {row["interval_id"] for row in first["CORRIDOR_OR_WEAK_GEOMETRY"]}
    rich = {row["interval_id"] for row in first["GEOMETRY_RICH"]}
    assert len(weak) == len(rich) == 10
    assert weak.isdisjoint(rich)


def test_five_scan_quantiles_are_unique_and_deterministic() -> None:
    assert select_five_scan_timestamps(range(10)) == select_five_scan_timestamps(reversed(range(10)))
    assert len(set(select_five_scan_timestamps(range(10)))) == 5


def test_unknown_uncertainty_is_not_zero_filled() -> None:
    report = conservative_uncertainty([{"name": "extrinsic", "value": "UNKNOWN"}], independence_and_one_sigma_proven=False)
    assert report["combined"] == "UNKNOWN"
    with pytest.raises(ValueError):
        conservative_uncertainty([{"name": "map", "value": 0, "evidence_type": "UNKNOWN"}], independence_and_one_sigma_proven=False)


def test_hidden_grandtour_reference_remains_ineligible() -> None:
    assert mission_reference_eligible(
        public_file_present=False,
        independent_non_lidar=True,
        complete_6dof=True,
        accuracy_evidence_present=True,
        reference_status="HIDDEN_OR_UNAVAILABLE",
    ) is False


def test_backend_bundle_sha_must_be_identical() -> None:
    assert_byte_identical_backend_bundle("a" * 64, "a" * 64)
    with pytest.raises(ValueError, match="differ"):
        assert_byte_identical_backend_bundle("a" * 64, "b" * 64)


def test_atomic_json_resume_is_byte_identical(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    before = path.read_bytes()
    atomic_write_json(path, {"a": 1, "b": 2})
    assert path.read_bytes() == before


def _empty_selection(path: Path) -> None:
    path.write_text(",".join(SNAPSHOT_SELECTION_FIELDS) + "\n", encoding="utf-8")


def test_failure_manifest_is_integrity_verifiable_but_not_success(tmp_path: Path) -> None:
    _empty_selection(tmp_path / "real_data_snapshot_selection_v1.csv")
    atomic_write_json(
        tmp_path / "NO_ICP_ATTESTATION.json",
        {
            "open3d_registration_call_count": 0,
            "pcl_cli_invocation_count": 0,
            "other_registration_process_count": 0,
            "real_trial_result_count": 0,
            "estimated_transform_file_count": 0,
            "registration_execution_count": 0,
            "pass": True,
        },
    )
    manifest = build_frozen_manifest(
        tmp_path,
        ["NO_ICP_ATTESTATION.json", "real_data_snapshot_selection_v1.csv"],
        source_commit="0" * 40,
        preregistration_ready=False,
        r14_freeze_pass=False,
        bindings=BACKEND_BINDINGS,
    )
    report = verify_frozen_manifest(tmp_path, manifest)
    assert report["artifact_integrity_pass"] is True
    assert report["preregistration_verification_pass"] is False


def test_failure_manifest_rebuild_and_resume_verification_are_byte_identical(tmp_path: Path) -> None:
    _empty_selection(tmp_path / "real_data_snapshot_selection_v1.csv")
    atomic_write_json(
        tmp_path / "NO_ICP_ATTESTATION.json",
        {
            "open3d_registration_call_count": 0,
            "pcl_cli_invocation_count": 0,
            "other_registration_process_count": 0,
            "real_trial_result_count": 0,
            "estimated_transform_file_count": 0,
            "registration_execution_count": 0,
            "pass": True,
        },
    )
    arguments = {
        "source_commit": "0" * 40,
        "preregistration_ready": False,
        "r14_freeze_pass": False,
        "bindings": BACKEND_BINDINGS,
    }
    first = build_frozen_manifest(
        tmp_path,
        ["NO_ICP_ATTESTATION.json", "real_data_snapshot_selection_v1.csv"],
        **arguments,
    )
    second = build_frozen_manifest(
        tmp_path,
        reversed(["NO_ICP_ATTESTATION.json", "real_data_snapshot_selection_v1.csv"]),
        **arguments,
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert verify_frozen_manifest(tmp_path, first) == verify_frozen_manifest(tmp_path, second)


def test_snapshot_selection_exact_50_50_counts(tmp_path: Path) -> None:
    path = tmp_path / "real_data_snapshot_selection_v1.csv"
    rows = []
    for dataset in ("IILABS_3D", "GRANDTOUR"):
        for label in ("CORRIDOR_OR_WEAK_GEOMETRY", "GEOMETRY_RICH"):
            for index in range(50):
                row = {field: "x" for field in SNAPSHOT_SELECTION_FIELDS}
                row.update(
                    {
                        "dataset_id": dataset,
                        "snapshot_id": f"{dataset}-{label}-{index:02d}",
                        "scene_label": label,
                        "labeler_blinded_to_registration_error": "true",
                        "open3d_pcl_shared_input": "true",
                        "eligibility_status": "ELIGIBLE",
                    }
                )
                rows.append(row)
    import csv
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SNAPSHOT_SELECTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    from phase_a_harness.real_data_preparation.manifest import verify_snapshot_selection
    report = verify_snapshot_selection(path)
    assert report == {
        "dataset_count": 2,
        "snapshot_count": 200,
        "weak_snapshot_count": 100,
        "rich_snapshot_count": 100,
        "complete_200_inventory": True,
    }


def test_manifest_tamper_is_detected(tmp_path: Path) -> None:
    _empty_selection(tmp_path / "real_data_snapshot_selection_v1.csv")
    atomic_write_json(tmp_path / "NO_ICP_ATTESTATION.json", {"pass": True})
    manifest = build_frozen_manifest(
        tmp_path,
        ["NO_ICP_ATTESTATION.json", "real_data_snapshot_selection_v1.csv"],
        source_commit="0" * 40,
        preregistration_ready=False,
        r14_freeze_pass=False,
        bindings=BACKEND_BINDINGS,
    )
    (tmp_path / "NO_ICP_ATTESTATION.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="checksum"):
        verify_frozen_manifest(tmp_path, manifest)


def test_registration_execution_count_is_structurally_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        report = guard.attestation(tmp_path)
    assert report["registration_execution_count"] == 0
    assert report["pass"] is True
