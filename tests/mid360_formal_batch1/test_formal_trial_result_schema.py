from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.formal_trial_result_schema import (
    BACKEND_PARAMETER_CONTRACT_SHA256,
    FormalTrialResultValidationError,
    IDENTITY_4X4,
    REQUIRED_FIELDS,
    SCHEMA_NAME,
    load_formal_trial_result_json,
    validate_formal_trial_result,
)


REPOSITORY = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPOSITORY
    / "experiments"
    / "mid360_formal_batch1"
    / "formal_trial_result_schema_v1.json"
)
TRIAL_ID = "FMB1-FIXTURE-ZERO-FMB1_R99_S01_Q01-open3d"


def _row(*, execution_kind: str = "FIXTURE") -> dict[str, object]:
    formal = execution_kind == "FORMAL"
    return {
        "schema": SCHEMA_NAME,
        "trial_id": TRIAL_ID,
        "batch_id": "FMB1",
        "track": "ZERO_PERTURBATION_MAINLINE",
        "track_classification": (
            "ZERO_PERTURBATION_TRACK" if formal else "FIXTURE_ONLY"
        ),
        "execution_kind": execution_kind,
        "fixture_only": not formal,
        "publish_eligible": formal,
        "formal_authority": formal,
        "formal_lock_sha256": "a" * 64,
        "scene_id": "FMB1_R99",
        "station_id": "S01",
        "snapshot_id": "FMB1_R99_S01_Q01",
        "backend": "open3d",
        "backend_version": "0.19.0+b012259",
        "backend_parameter_contract_sha256": BACKEND_PARAMETER_CONTRACT_SHA256,
        "source_sha256": "b" * 64,
        "target_sha256": "c" * 64,
        "source_point_count": 16000,
        "target_point_count": 150000,
        "T0": [list(row) for row in IDENTITY_4X4],
        "T_est": [list(row) for row in IDENTITY_4X4],
        "trial_status": "COMPLETED",
        "solver_status": "CONVERGED",
        "solver_success": True,
        "finite_transform": True,
        "finite_result": True,
        "pose_recovered": True,
        "translation_vector_m": [0.0, 0.0, 0.0],
        "translation_norm_m": 0.0,
        "rotation_vector_deg": [0.0, 0.0, 0.0],
        "rotation_angle_deg": 0.0,
        "initial_correspondence_count": 15500,
        "initial_correspondence_fraction": 0.96875,
        "LOW_INITIAL_OVERLAP": False,
        "final_correspondence_count": 15450,
        "correspondence_turnover": 0.02,
        "fitness": 0.96,
        "final_residual_rmse": 0.01,
        "final_translation_error_m": 0.001,
        "final_rotation_error_deg": 0.02,
        "iteration_count": 4,
        "runtime_ms": 12.5,
        "failure_reason": None,
        "created_at_utc": "2026-08-20T00:00:00+00:00",
        "MEASUREMENT_FINAL_RESULT": False,
    }


def _lock(*, execution_kind: str = "FIXTURE") -> dict[str, object]:
    formal = execution_kind == "FORMAL"
    row = _row(execution_kind=execution_kind)
    return {
        "FORMAL_AUTHORITY": formal,
        "FORMAL_REGISTRATION_AUTHORIZED": formal,
        "FIXTURE_ONLY": not formal,
        "NOT_REAL_FMB1": not formal,
        "formal_lock_sha256": row["formal_lock_sha256"],
        "backend_parameter_contract_sha256": BACKEND_PARAMETER_CONTRACT_SHA256,
        "expected_trials": {
            TRIAL_ID: {
                field: copy.deepcopy(row[field])
                for field in (
                    "scene_id",
                    "station_id",
                    "snapshot_id",
                    "track",
                    "execution_kind",
                    "backend",
                    "backend_version",
                    "source_sha256",
                    "target_sha256",
                    "T0",
                )
            }
        },
    }


def test_json_schema_is_strict_non_authorizing_and_matches_python_fields() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["x-formal-authority"] is False
    assert schema["x-formal-registration-authorized"] is False
    assert schema["x-fixture-publication-forbidden"] is True
    assert schema["x-backend-parameter-contract-sha256"] == BACKEND_PARAMETER_CONTRACT_SHA256
    assert set(schema["required"]) == set(REQUIRED_FIELDS)
    assert set(schema["properties"]) == set(REQUIRED_FIELDS)


def test_valid_locked_fixture_row_passes_validation_but_is_not_publishable() -> None:
    row = _row()
    validated = validate_formal_trial_result(row, _lock(), for_publication=False)
    assert validated == row
    assert validated is not row
    assert validated["fixture_only"] is True
    assert validated["publish_eligible"] is False
    assert validated["formal_authority"] is False


def test_fixture_displacement_components_are_bound_to_t0_and_estimate() -> None:
    row = _row()
    angle = math.radians(10.0)
    row["T_est"] = [
        [math.cos(angle), -math.sin(angle), 0.0, 0.01],
        [math.sin(angle), math.cos(angle), 0.0, -0.02],
        [0.0, 0.0, 1.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ]
    row["translation_vector_m"] = [0.01, -0.02, 0.03]
    row["translation_norm_m"] = math.sqrt(0.0014)
    row["rotation_vector_deg"] = [0.0, 0.0, 10.0]
    row["rotation_angle_deg"] = 10.0
    validated = validate_formal_trial_result(row, _lock())
    assert validated["track_classification"] == "FIXTURE_ONLY"


@pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
def test_every_required_field_is_enforced(missing_field: str) -> None:
    row = _row()
    del row[missing_field]
    with pytest.raises(FormalTrialResultValidationError, match="missing fields"):
        validate_formal_trial_result(row, _lock())


@pytest.mark.parametrize(
    ("field", "nonfinite"),
    [
        ("runtime_ms", float("nan")),
        ("runtime_ms", float("inf")),
        ("initial_correspondence_fraction", float("-inf")),
        ("fitness", float("nan")),
        ("final_residual_rmse", float("inf")),
        ("final_translation_error_m", float("nan")),
        ("final_rotation_error_deg", float("inf")),
        ("correspondence_turnover", float("nan")),
        ("translation_norm_m", float("inf")),
        ("rotation_angle_deg", float("nan")),
    ],
)
def test_nonfinite_scalar_is_rejected(field: str, nonfinite: float) -> None:
    row = _row()
    row[field] = nonfinite
    with pytest.raises(FormalTrialResultValidationError, match="finite"):
        validate_formal_trial_result(row, _lock())


@pytest.mark.parametrize("matrix_field", ["T0", "T_est"])
def test_nonfinite_matrix_is_rejected(matrix_field: str) -> None:
    row = _row()
    row[matrix_field][0][0] = float("nan")  # type: ignore[index]
    with pytest.raises(FormalTrialResultValidationError, match="finite"):
        validate_formal_trial_result(row, _lock())


def test_nonfinite_translation_vector_is_rejected() -> None:
    row = _row()
    row["translation_vector_m"][1] = float("nan")  # type: ignore[index]
    with pytest.raises(FormalTrialResultValidationError, match="finite"):
        validate_formal_trial_result(row, _lock())


def test_nonfinite_rotation_vector_is_rejected() -> None:
    row = _row()
    row["rotation_vector_deg"][2] = float("inf")  # type: ignore[index]
    with pytest.raises(FormalTrialResultValidationError, match="finite"):
        validate_formal_trial_result(row, _lock())


def test_backend_outside_frozen_pair_is_rejected() -> None:
    row = _row()
    row["backend"] = "ndt"
    with pytest.raises(FormalTrialResultValidationError, match="backend is not allowed"):
        validate_formal_trial_result(row, _lock())


def test_trial_id_outside_lock_is_rejected() -> None:
    row = _row()
    row["trial_id"] = "FMB1-ZERO-NOT-IN-LOCK"
    with pytest.raises(FormalTrialResultValidationError, match="outside the lock"):
        validate_formal_trial_result(row, _lock())


def test_track_must_match_exact_locked_trial() -> None:
    row = _row()
    row["track"] = "CAPTURE_BASIN"
    with pytest.raises(FormalTrialResultValidationError, match="track does not match"):
        validate_formal_trial_result(row, _lock())


def test_fixture_track_classification_cannot_claim_a_formal_track() -> None:
    row = _row()
    row["track_classification"] = "ZERO_PERTURBATION_TRACK"
    with pytest.raises(FormalTrialResultValidationError, match="FIXTURE_ONLY"):
        validate_formal_trial_result(row, _lock())


def test_zero_track_requires_identity_t0() -> None:
    row = _row()
    row["T0"][0][3] = 0.1  # type: ignore[index]
    with pytest.raises(FormalTrialResultValidationError, match="exact identity T0"):
        validate_formal_trial_result(row, _lock())


def test_capture_basin_t0_must_still_match_lock() -> None:
    row = _row()
    lock = _lock()
    row["track"] = "CAPTURE_BASIN"
    lock["expected_trials"][TRIAL_ID]["track"] = "CAPTURE_BASIN"  # type: ignore[index]
    lock["expected_trials"][TRIAL_ID]["T0"][0][3] = 0.10  # type: ignore[index]
    with pytest.raises(FormalTrialResultValidationError, match="T0 does not match"):
        validate_formal_trial_result(row, lock)


def test_formal_row_cannot_use_unauthorized_lock_context() -> None:
    row = _row(execution_kind="FORMAL")
    lock = _lock(execution_kind="FORMAL")
    lock["FORMAL_REGISTRATION_AUTHORIZED"] = False
    with pytest.raises(FormalTrialResultValidationError, match="already-authorized"):
        validate_formal_trial_result(row, lock)


def test_fixture_validates_only_with_non_authoritative_flags() -> None:
    row = _row(execution_kind="FIXTURE")
    validated = validate_formal_trial_result(
        row, _lock(execution_kind="FIXTURE"), for_publication=False
    )
    assert validated["fixture_only"] is True
    assert validated["publish_eligible"] is False
    assert validated["formal_authority"] is False


def test_fixture_cannot_be_published() -> None:
    with pytest.raises(FormalTrialResultValidationError, match="cannot be published"):
        validate_formal_trial_result(
            _row(execution_kind="FIXTURE"),
            _lock(execution_kind="FIXTURE"),
            for_publication=True,
        )


def test_fixture_cannot_claim_publish_eligibility() -> None:
    row = _row(execution_kind="FIXTURE")
    row["publish_eligible"] = True
    with pytest.raises(FormalTrialResultValidationError, match="FIXTURE requires"):
        validate_formal_trial_result(row, _lock(execution_kind="FIXTURE"))


def test_fixture_requires_explicit_not_real_lock_metadata() -> None:
    lock = _lock()
    lock["NOT_REAL_FMB1"] = False
    with pytest.raises(FormalTrialResultValidationError, match="NOT_REAL_FMB1"):
        validate_formal_trial_result(_row(), lock)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("translation_vector_m", [0.001, 0.0, 0.0]),
        ("translation_norm_m", 0.001),
        ("rotation_vector_deg", [0.0, 0.0, 0.1]),
        ("rotation_angle_deg", 0.1),
    ],
)
def test_reported_displacement_must_match_t0_and_estimate(
    field: str, value: object
) -> None:
    row = _row()
    row[field] = value
    with pytest.raises(FormalTrialResultValidationError, match="inconsistent"):
        validate_formal_trial_result(row, _lock())


def test_solver_status_and_finite_result_are_independent_explicit_gates() -> None:
    row = _row()
    row["solver_status"] = "NOT_CONVERGED"
    with pytest.raises(FormalTrialResultValidationError, match="solver_status"):
        validate_formal_trial_result(row, _lock())
    row = _row()
    row["finite_result"] = False
    with pytest.raises(FormalTrialResultValidationError, match="finite_result"):
        validate_formal_trial_result(row, _lock())


def test_backend_failure_is_retained_with_null_result_metrics() -> None:
    row = _row()
    row.update(
        {
            "T_est": None,
            "trial_status": "BACKEND_FAILURE",
            "solver_status": "BACKEND_ERROR",
            "solver_success": False,
            "finite_transform": False,
            "finite_result": False,
            "pose_recovered": None,
            "translation_vector_m": None,
            "translation_norm_m": None,
            "rotation_vector_deg": None,
            "rotation_angle_deg": None,
            "final_correspondence_count": None,
            "correspondence_turnover": None,
            "fitness": None,
            "final_residual_rmse": None,
            "final_translation_error_m": None,
            "final_rotation_error_deg": None,
            "iteration_count": None,
            "failure_reason": "backend returned no transform",
        }
    )
    assert validate_formal_trial_result(row, _lock())["trial_status"] == "BACKEND_FAILURE"


def test_json_loader_rejects_nonstandard_nan(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"runtime_ms": NaN}', encoding="utf-8")
    with pytest.raises(FormalTrialResultValidationError, match="non-finite"):
        load_formal_trial_result_json(path)


def test_validator_source_has_no_backend_import() -> None:
    source = (
        REPOSITORY
        / "experiments"
        / "mid360_formal_batch1"
        / "formal_trial_result_schema.py"
    ).read_text(encoding="utf-8")
    assert "import open3d" not in source
    assert "from open3d" not in source
    assert "import pcl" not in source
    assert "from pcl" not in source
    assert "debug_registration" not in source
