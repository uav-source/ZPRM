from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_a_harness import phase_b_backend_execution as bridge
from phase_a_harness.phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    TrialResultValidationError,
    canonical_json_bytes,
    validate_phase_a_trial_result_strict,
)
from phase_a_harness.phase_a_trial_resume import CorruptExistingResult
from phase_a_harness.phase_b_trial_result import (
    PHASE_B_CONDITIONS,
    validate_existing_phase_b_trial_result_for_resume,
    validate_phase_b_trial_result_strict,
    write_phase_b_trial_result,
)


def _diagnostics(backend: str) -> dict:
    if backend == OPEN3D_BACKEND:
        return {
            "fitness": 1.0,
            "inlier_rmse": 0.0,
            "correspondence_set_size": 100,
        }
    normals = {
        "finite_count": 100,
        "zero_count": 0,
        "nan_count": 0,
        "norm_min": 1.0,
        "norm_median": 1.0,
        "norm_max": 1.0,
    }
    return {
        "pcl_version": "1.15.1",
        "pcl_cli_sha256": "8" * 64,
        "exit_code": 0,
        "has_converged_raw": True,
        "fitness_score": 0.0,
        "iteration_count": 1,
        "correspondence_count": 100,
        "source_normal_statistics": dict(normals),
        "target_normal_statistics": dict(normals),
    }


def _payload(condition: str, backend: str = OPEN3D_BACKEND) -> dict:
    return {
        "schema_version": "phase_a_trial_result_v1",
        "planned_trial_id": f"snapshot-1/{backend}",
        "snapshot_id": "snapshot-1",
        "scene_variant": "GEOMETRY_RICH_ROOM",
        "condition": condition,
        "backend": backend,
        "protocol_sha256": "1" * 64,
        "snapshot_lock_sha256": "2" * 64,
        "snapshot_checksum": "3" * 64,
        "source_checksum": "4" * 64,
        "target_checksum": "5" * 64,
        "reference_pose_checksum": "6" * 64,
        "implementation_sha256": "7" * 64,
        "solver_failure": False,
        "failure_classification": "NONE",
        "failure_detail": None,
        "final_transform_4x4": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "translation_update_m": 0.0,
        "rotation_update_rad": 0.0,
        "raw_rotation_finite": True,
        "raw_rotation_determinant": 1.0,
        "orthogonality_defect_fro": 0.0,
        "projection_correction_fro": 0.0,
        "finite_output": True,
        "runtime_ms": 1.0,
        "backend_diagnostics": _diagnostics(backend),
    }


def _expected(value: dict) -> dict:
    names = (
        "planned_trial_id",
        "snapshot_id",
        "backend",
        "scene_variant",
        "condition",
        "protocol_sha256",
        "snapshot_lock_sha256",
        "snapshot_checksum",
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "implementation_sha256",
    )
    return {name: value[name] for name in names}


@pytest.mark.parametrize("condition", sorted(PHASE_B_CONDITIONS))
def test_phase_b_validator_preserves_frozen_26_field_contract(condition: str):
    value = _payload(condition)
    validated = validate_phase_b_trial_result_strict(value)
    assert validated == value
    assert validated["condition"] == condition
    assert validated["schema_version"] == "phase_a_trial_result_v1"
    assert len(validated) == 26


def test_phase_b_validator_rejects_non_phase_b_condition_without_mutation():
    value = _payload("IDEAL_MATCHED")
    before = dict(value)
    with pytest.raises(TrialResultValidationError, match="Phase B condition"):
        validate_phase_b_trial_result_strict(value)
    assert value == before


def test_phase_b_atomic_writer_uses_canonical_payload_and_refuses_overwrite(tmp_path: Path):
    value = _payload("FULL_NOISE")
    path, digest = write_phase_b_trial_result(tmp_path, value)
    assert path.read_bytes() == canonical_json_bytes(value)
    assert len(digest) == 64
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        write_phase_b_trial_result(tmp_path, value)


def test_phase_b_resume_validates_sha_schema_and_real_condition(tmp_path: Path):
    value = _payload("INDEPENDENT_NOISE_FREE")
    path, digest = write_phase_b_trial_result(tmp_path, value)
    entry = {
        "planned_trial_id": value["planned_trial_id"],
        "path": path.name,
        "sha256": digest,
    }
    assert validate_existing_phase_b_trial_result_for_resume(
        path, manifest_entry=entry, expected=_expected(value)
    ) == value
    wrong = _expected(value)
    wrong["source_checksum"] = "0" * 64
    with pytest.raises(CorruptExistingResult, match="source_checksum"):
        validate_existing_phase_b_trial_result_for_resume(
            path, manifest_entry=entry, expected=wrong
        )


def test_open3d_bridge_changes_only_temporary_condition(monkeypatch: pytest.MonkeyPatch):
    condition = "FULL_NOISE"
    common = {
        name: value
        for name, value in _payload(condition, OPEN3D_BACKEND).items()
        if name
        in {
            "backend",
            "condition",
            "implementation_sha256",
            "planned_trial_id",
            "protocol_sha256",
            "reference_pose_checksum",
            "scene_variant",
            "schema_version",
            "snapshot_checksum",
            "snapshot_id",
            "snapshot_lock_sha256",
            "source_checksum",
            "target_checksum",
        }
    }
    fixture = SimpleNamespace(condition=condition)
    parameters = {"frozen": object()}
    observed = {}

    def fake_executor(*, fixture, common, parameters):
        observed.update(fixture=fixture, common=common, parameters=parameters)
        value = _payload("IDEAL_MATCHED", OPEN3D_BACKEND)
        value.update(common)
        return validate_phase_a_trial_result_strict(value)

    monkeypatch.setattr(bridge, "execute_open3d_fixture", fake_executor)
    result = bridge.execute_phase_b_open3d_fixture(
        fixture=fixture, common=common, parameters=parameters
    )
    assert observed["fixture"] is fixture and observed["parameters"] is parameters
    assert observed["common"]["condition"] == "IDEAL_MATCHED"
    assert common["condition"] == condition and result["condition"] == condition
    normalized_result = dict(result)
    normalized_result["condition"] = "IDEAL_MATCHED"
    assert normalized_result == fake_executor(
        fixture=fixture,
        common=observed["common"],
        parameters=parameters,
    )


def test_pcl_bridge_preserves_parameters_cli_and_exposes_no_native(monkeypatch: pytest.MonkeyPatch):
    condition = "INDEPENDENT_NOISE_FREE"
    common = {
        name: value
        for name, value in _payload(condition, PCL_BACKEND).items()
        if name
        in {
            "backend",
            "condition",
            "implementation_sha256",
            "planned_trial_id",
            "protocol_sha256",
            "reference_pose_checksum",
            "scene_variant",
            "schema_version",
            "snapshot_checksum",
            "snapshot_id",
            "snapshot_lock_sha256",
            "source_checksum",
            "target_checksum",
        }
    }
    fixture = SimpleNamespace(condition=condition)
    parameters = {"frozen": object()}
    pcl_cli = Path("bin/pcl_point_to_plane_cli")
    observed = {}

    def fake_executor(*, fixture, common, parameters, pcl_cli):
        observed.update(
            fixture=fixture, common=common, parameters=parameters, pcl_cli=pcl_cli
        )
        value = _payload("IDEAL_MATCHED", PCL_BACKEND)
        value.update(common)
        return validate_phase_a_trial_result_strict(value)

    monkeypatch.setattr(bridge, "execute_pcl_fixture", fake_executor)
    result = bridge.execute_phase_b_pcl_fixture(
        fixture=fixture,
        common=common,
        parameters=parameters,
        pcl_cli=pcl_cli,
    )
    assert observed["fixture"] is fixture and observed["parameters"] is parameters
    assert observed["pcl_cli"] is pcl_cli
    assert observed["common"]["condition"] == "IDEAL_MATCHED"
    assert common["condition"] == condition and result["condition"] == condition
    assert not hasattr(bridge, "execute_phase_b_native_fixture")
