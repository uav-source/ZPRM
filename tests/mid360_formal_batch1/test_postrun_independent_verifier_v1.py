import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import experiments.mid360_formal_batch1.postrun_verification.independent_postrun_verifier_v1 as verifier


REPOSITORY = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (REPOSITORY / "experiments/mid360_formal_batch1/zero_perturbation_trial_result_schema_v1_1.json").read_text()
)
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def plan_row() -> dict:
    return {
        "trial_id": "FMB1-ZP11R1-FMB1-R01-S01-Q01-O3D", "batch_id": "FMB1",
        "amendment_id": "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1",
        "track_id": "ZERO_PERTURBATION_TRACK", "scene_id": "FMB1_R01",
        "final_geometry_class": "RICH", "station_id": "S01", "attempt": 1,
        "snapshot_id": "FMB1_R01_S01_Q01", "backend": "OPEN3D_POINT_TO_PLANE",
        "backend_version": "0.19.0+b012259", "source_reference": "fixture/source.npy",
        "source_sha256": "a" * 64, "source_array_sha256": "b" * 64,
        "source_point_count": 20, "target_reference": "fixture/target.npy",
        "target_sha256": "c" * 64, "target_array_sha256": "d" * 64,
        "target_point_count": 30, "T0": IDENTITY, "T_reference_nominal": IDENTITY,
        "backend_parameter_contract_sha256": "e" * 64,
        "backend_canonical_parameter_sha256": "f" * 64,
        "active_amendment_sha256": "1" * 64, "analysis_contract_sha256": "2" * 64,
        "translation_perturbation_m": 0.0, "rotation_perturbation_deg": 0.0,
    }


def result_row() -> dict:
    plan = plan_row()
    row = {
        "schema": "mid360_fmb1_zero_perturbation_trial_result_v1_1_r1",
        **{name: plan[name] for name in (
            "trial_id", "batch_id", "amendment_id", "track_id", "scene_id", "station_id",
            "attempt", "snapshot_id", "backend", "backend_version", "source_reference",
            "source_sha256", "target_reference", "target_sha256", "T0", "T_reference_nominal",
            "backend_parameter_contract_sha256", "backend_canonical_parameter_sha256",
            "active_amendment_sha256", "analysis_contract_sha256")},
        "geometry_class": plan["final_geometry_class"], "T_est": IDENTITY, "Delta_T": IDENTITY,
        "translation_x_m": 0.0, "translation_y_m": 0.0, "translation_z_m": 0.0,
        "translation_norm_m": 0.0, "rotation_angle_rad": 0.0, "rotation_angle_deg": 0.0,
        "solver_status": "FIXTURE_COMPLETED", "finite_result": True,
        "scientific_status": "COMPLETED", "infrastructure_status": "OK",
        "retry_eligible": False, "retry_reason": None,
        "common_association_valid": True, "common_association_invalid_reason": None,
        "common_association_invalid_detail": None, "initial_correspondence_count": 20,
        "initial_valid_normal_correspondence_count": 20, "final_correspondence_count": 20,
        "final_valid_normal_correspondence_count": 20, "correspondence_turnover": 0.0,
        "accepted_source_turnover": 0.0, "correspondence_count_change_ratio": 0.0,
        "initial_residual_rmse": 0.0, "final_residual_rmse": 0.0,
        "residual_rmse_change": 0.0, "median_normal_angle_change_deg": 0.0,
        "q95_normal_angle_change_deg": 0.0, "trial_plan_sha256": "3" * 64,
        "formal_lock_sha256": verifier.R3_LOCK_SHA256,
        "code_commit": verifier.R3_EXECUTION_CODE_COMMIT,
        "environment_identity": {
            "python": "3.11.15", "numpy": "1.26.4", "scipy": "1.11.4",
            "open3d": "0.19.0+b012259", "pcl": "1.15.1",
            "environment_manifest_sha256": "4" * 64,
        },
        "physical_reference_semantics": "NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT",
        "execution_kind": "FORMAL", "fixture_only": False,
        "created_at_utc": "2026-08-20T08:00:00+00:00",
    }
    return row


def test_frozen_contract_values_are_exact() -> None:
    contract = verifier.load_contract()
    assert contract["tolerances"]["matrix_atol"] == 1e-12
    assert contract["tolerances"]["angle_deg_atol"] == 1e-9
    assert contract["association"]["association_distance_limit_m"] == 0.5
    assert contract["real_result_values_read_before_contract_freeze"] is False


def test_independent_schema_accepts_fixture_result() -> None:
    verifier.validate_schema_value(result_row(), SCHEMA)
    verifier._validate_result_identity(
        result_row(), plan_row(), SCHEMA, trial_plan_sha="3" * 64,
        lock_sha=verifier.R3_LOCK_SHA256, environment_sha="4" * 64,
    )


def test_fixture_only_cli_reads_no_real_results() -> None:
    tool = REPOSITORY / "tools/mid360_formal_batch1/verify_zero_perturbation_exec_r3_postrun.py"
    run = subprocess.run([sys.executable, str(tool), "--fixture-only"], cwd=REPOSITORY, text=True, capture_output=True)
    assert run.returncode == 0
    report = json.loads(run.stdout)
    assert report["FIXTURE_ONLY"] is True and report["real_result_values_read"] is False


def test_real_confirmation_flag_absent_fails_before_file_read(tmp_path: Path) -> None:
    with pytest.raises(verifier.IndependentPostrunVerificationError, match="confirm-read"):
        verifier.verify_frozen_real_results(
            repository=tmp_path, execution_results_root=tmp_path / "exec",
            r3_lock_dir=tmp_path / "lock", raw_execution_commit=verifier.RAW_EXECUTION_COMMIT,
            raw_execution_tag="execution/fmb1-zero-perturbation-v1.1-exec-r3-formal-v1",
            verifier_code_commit="0" * 40, output_dir=tmp_path / "out",
            confirm_read_frozen_real_results=False,
        )


def test_canonical_array_file_and_raw_byte_hash(tmp_path: Path) -> None:
    points = np.ascontiguousarray(np.arange(60, dtype="<f8").reshape(20, 3))
    path = tmp_path / "source.npy"; np.save(path, points, allow_pickle=False)
    plan = plan_row(); plan["source_reference"] = "source.npy"
    plan["source_sha256"] = verifier.sha256_file(path)
    plan["source_array_sha256"] = hashlib.sha256(points.tobytes(order="C")).hexdigest()
    loaded, audit = verifier.load_canonical_array(tmp_path, plan, "source")
    assert np.array_equal(loaded, points)
    assert audit["array_sha256"] == plan["source_array_sha256"]


def test_backend_pair_input_identity() -> None:
    left = plan_row(); right = dict(left)
    right["trial_id"] = right["trial_id"].replace("O3D", "PCL")
    right["backend"] = "PCL_POINT_TO_PLANE"; right["backend_version"] = "1.15.1"
    verifier.validate_backend_pair_inputs([left, right], snapshot_count=1)


def test_inventory_identity_sets_fixture() -> None:
    report = verifier.validate_inventory_identity_sets(
        ["trial-a"], ["trial-a"], ["trial-a"], {"trial-a": [1]}, expected_count=1
    )
    assert report["missing_count"] == report["second_attempt_count"] == 0


def test_start_marker_authorization_and_utc_fixture() -> None:
    marker = {
        "schema": "mid360_fmb1_formal_trial_attempt_start_v1_1_r1",
        "trial_id": "trial-a", "attempt_number": 1,
        "authorization_sha256": "a" * 64,
        "started_at_utc": "2026-08-20T08:00:00+00:00",
    }
    parsed = verifier.validate_start_marker_payload(
        marker, trial_id="trial-a", authorization_sha256="a" * 64
    )
    assert parsed.utcoffset().total_seconds() == 0


def test_forbidden_import_and_subprocess_module_ast() -> None:
    forbidden = set(verifier.load_contract()["forbidden_imports"])
    paths = list((REPOSITORY / "experiments/mid360_formal_batch1/postrun_verification").glob("*.py"))
    paths.append(REPOSITORY / "tools/mid360_formal_batch1/verify_zero_perturbation_exec_r3_postrun.py")
    findings = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                findings.extend((str(path), alias.name) for alias in node.names if alias.name in forbidden)
            elif isinstance(node, ast.ImportFrom) and node.module in forbidden:
                findings.append((str(path), node.module))
            elif isinstance(node, ast.Call):
                for argument in (*node.args, *[item.value for item in node.keywords]):
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        findings.extend((str(path), name) for name in forbidden if name in argument.value)
    assert findings == []
