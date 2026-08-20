from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import ast
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.zero_perturbation_r1_trial_assets import (
    AMENDMENT_ID,
    BACKENDS,
    IDENTITY_4X4,
    PHYSICAL_REFERENCE_SEMANTICS,
    RESULT_SCHEMA_NAME,
    R1TrialAssetError,
    TRACK_ID,
    build_result_schema,
    validate_result_row,
)
from experiments.mid360_formal_batch1.zero_perturbation_r1_trial_verify import (
    R1TrialVerificationError,
    verify_assets,
    verify_plan_payload,
    verify_result_schema_payload,
)


REPOSITORY = Path(__file__).resolve().parents[2]
EXPERIMENT = REPOSITORY / "experiments/mid360_formal_batch1"
FINAL = REPOSITORY / "results/mid360_formal_batch1/final_dataset_v1"
PLAN_PATH = EXPERIMENT / "zero_perturbation_trial_plan_v1_1.json"
SCHEMA_PATH = EXPERIMENT / "zero_perturbation_trial_result_schema_v1_1.json"
SNAPSHOT_PATH = FINAL / "final_snapshot_manifest.csv"
TARGET_PATH = FINAL / "final_target_manifest.csv"
BACKEND_PATH = REPOSITORY / "frozen_assets/backend_parameter_contract.json"
AMENDMENT_PATH = EXPERIMENT / "amendments/zero_perturbation_mainline_v1_1_r1.json"
ANALYSIS_PATH = EXPERIMENT / "amendments/zero_perturbation_analysis_contract_v1_1_r1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def context() -> dict[str, object]:
    return {
        "plan": json.loads(PLAN_PATH.read_text(encoding="utf-8")),
        "snapshots": _csv(SNAPSHOT_PATH),
        "targets": _csv(TARGET_PATH),
        "snapshot_manifest_sha256": _sha(SNAPSHOT_PATH),
        "target_manifest_sha256": _sha(TARGET_PATH),
        "backend_contract": json.loads(BACKEND_PATH.read_text(encoding="utf-8")),
        "backend_contract_sha256": _sha(BACKEND_PATH),
        "amendment_sha256": _sha(AMENDMENT_PATH),
        "analysis_contract_sha256": _sha(ANALYSIS_PATH),
    }


def _verify(plan: dict[str, object], context: dict[str, object]) -> dict[str, object]:
    return verify_plan_payload(
        plan,
        context["snapshots"],  # type: ignore[arg-type]
        context["targets"],  # type: ignore[arg-type]
        snapshot_manifest_sha256=context["snapshot_manifest_sha256"],  # type: ignore[arg-type]
        target_manifest_sha256=context["target_manifest_sha256"],  # type: ignore[arg-type]
        backend_contract=context["backend_contract"],  # type: ignore[arg-type]
        backend_contract_sha256=context["backend_contract_sha256"],  # type: ignore[arg-type]
        amendment_sha256=context["amendment_sha256"],  # type: ignore[arg-type]
        analysis_contract_sha256=context["analysis_contract_sha256"],  # type: ignore[arg-type]
        verify_payload_files=False,
    )


def test_authoritative_assets_pass_independent_byte_verification() -> None:
    report = verify_assets(repository=REPOSITORY, verify_payload_files=True)
    assert report["status"] == "PASS"
    assert report["total_trial_count"] == 360
    assert report["source_payload_count"] == 180
    assert report["target_payload_count"] == 18
    assert report["registration_backend_call_count"] == 0


def test_plan_exact_inventory_and_w02_attempt2_only(context: dict[str, object]) -> None:
    plan = context["plan"]
    rows = plan["rows"]  # type: ignore[index]
    assert len(rows) == 360
    assert sum(row["backend"] == "OPEN3D_POINT_TO_PLANE" for row in rows) == 180
    assert sum(row["backend"] == "PCL_POINT_TO_PLANE" for row in rows) == 180
    assert {row["scene_id"] for row in rows} == {
        "FMB1_R01", "FMB1_R02", "FMB1_R03", "FMB1_W01", "FMB1_W02", "FMB1_W03"
    }
    assert {row["attempt"] for row in rows if row["scene_id"] == "FMB1_W02"} == {2}
    assert all("W04" not in json.dumps(row) for row in rows)
    assert all(row["T0"] == IDENTITY_4X4 for row in rows)
    assert all("T_est" not in row and "Delta_T" not in row for row in rows)
    assert _verify(plan, context)["status"] == "PASS"  # type: ignore[arg-type]


def test_plan_producer_and_independent_verifier_import_no_backend() -> None:
    for name in (
        "zero_perturbation_r1_trial_assets.py",
        "zero_perturbation_r1_trial_verify.py",
    ):
        tree = ast.parse((EXPERIMENT / name).read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert "open3d" not in imports
        assert "pcl" not in imports
        assert "pclpy" not in imports


def _mutate_w04(plan: dict[str, object]) -> None:
    plan["rows"][0]["scene_id"] = "FMB1_W04"  # type: ignore[index]


def _mutate_old_w02(plan: dict[str, object]) -> None:
    row = next(item for item in plan["rows"] if item["scene_id"] == "FMB1_W02")  # type: ignore[index]
    row["attempt"] = 1


def _mutate_drop_row(plan: dict[str, object]) -> None:
    plan["rows"].pop()  # type: ignore[index]


def _mutate_backend_balance(plan: dict[str, object]) -> None:
    row = next(item for item in plan["rows"] if item["backend"] == "PCL_POINT_TO_PLANE")  # type: ignore[index]
    row["backend"] = "OPEN3D_POINT_TO_PLANE"


def _mutate_t0(plan: dict[str, object]) -> None:
    plan["rows"][0]["T0"][0][3] = 0.001  # type: ignore[index]


def _mutate_capture_radius(plan: dict[str, object]) -> None:
    plan["rows"][0]["translation_perturbation_m"] = 0.001  # type: ignore[index]


def _mutate_source_sha(plan: dict[str, object]) -> None:
    plan["rows"][0]["source_sha256"] = "0" * 64  # type: ignore[index]


def _mutate_target_sha(plan: dict[str, object]) -> None:
    plan["rows"][0]["target_sha256"] = "0" * 64  # type: ignore[index]


def _mutate_geometry(plan: dict[str, object]) -> None:
    plan["rows"][0]["final_geometry_class"] = "WEAK"  # type: ignore[index]


def _mutate_backend_contract(plan: dict[str, object]) -> None:
    plan["rows"][0]["backend_parameter_contract_sha256"] = "0" * 64  # type: ignore[index]


def _mutate_snapshot_independence(plan: dict[str, object]) -> None:
    plan["snapshot_independence_claimed"] = True


def _mutate_authorization(plan: dict[str, object]) -> None:
    plan["FORMAL_REGISTRATION_AUTHORIZED"] = True


def _mutate_actual_trial_count(plan: dict[str, object]) -> None:
    plan["actual_formal_trials"] = 1


def _mutate_duplicate(plan: dict[str, object]) -> None:
    plan["rows"][1] = copy.deepcopy(plan["rows"][0])  # type: ignore[index]


@pytest.mark.parametrize(
    "mutator",
    [
        _mutate_w04,
        _mutate_old_w02,
        _mutate_drop_row,
        _mutate_backend_balance,
        _mutate_t0,
        _mutate_capture_radius,
        _mutate_source_sha,
        _mutate_target_sha,
        _mutate_geometry,
        _mutate_backend_contract,
        _mutate_snapshot_independence,
        _mutate_authorization,
        _mutate_actual_trial_count,
        _mutate_duplicate,
    ],
)
def test_plan_tamper_is_rejected(mutator, context: dict[str, object]) -> None:
    plan = copy.deepcopy(context["plan"])
    mutator(plan)
    with pytest.raises(R1TrialVerificationError):
        _verify(plan, context)


def test_result_schema_is_strict_r1_and_non_authorizing() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    verify_result_schema_payload(schema)
    assert schema == build_result_schema()
    assert schema["additionalProperties"] is False
    assert schema["x-registration-authority-granted"] is False
    assert schema["x-real-result-count-at-freeze"] == 0
    assert {
        "initial_correspondence_count",
        "final_correspondence_count",
        "initial_valid_normal_correspondence_count",
        "final_valid_normal_correspondence_count",
        "correspondence_turnover",
        "accepted_source_turnover",
        "correspondence_count_change_ratio",
        "initial_residual_rmse",
        "final_residual_rmse",
        "residual_rmse_change",
        "median_normal_angle_change_deg",
        "q95_normal_angle_change_deg",
        "common_association_valid",
        "common_association_invalid_reason",
        "common_association_invalid_detail",
    } <= set(schema["properties"])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("properties", "physical_reference_semantics", "const"), "SUBMILLIMETER_GT"),
        (("properties", "fixture_only", "const"), True),
        (("x-primary-experimental-unit",), "snapshot"),
        (("x-registration-authority-granted",), True),
    ],
)
def test_result_schema_tamper_is_rejected(path: tuple[str, ...], value: object) -> None:
    schema = build_result_schema()
    node = schema
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(R1TrialVerificationError):
        verify_result_schema_payload(schema)


def test_result_schema_additional_backend_is_rejected() -> None:
    schema = build_result_schema()
    schema["properties"]["backend"]["enum"].append("NATIVE")
    with pytest.raises(R1TrialVerificationError):
        verify_result_schema_payload(schema)


def _valid_result(plan: dict[str, object]) -> dict[str, object]:
    trial = plan["rows"][0]  # type: ignore[index]
    return {
        "schema": RESULT_SCHEMA_NAME,
        "trial_id": trial["trial_id"],
        "batch_id": "FMB1",
        "amendment_id": AMENDMENT_ID,
        "track_id": TRACK_ID,
        "scene_id": trial["scene_id"],
        "geometry_class": trial["final_geometry_class"],
        "station_id": trial["station_id"],
        "attempt": trial["attempt"],
        "snapshot_id": trial["snapshot_id"],
        "backend": trial["backend"],
        "backend_version": trial["backend_version"],
        "source_reference": trial["source_reference"],
        "source_sha256": trial["source_sha256"],
        "target_reference": trial["target_reference"],
        "target_sha256": trial["target_sha256"],
        "T0": copy.deepcopy(IDENTITY_4X4),
        "T_reference_nominal": copy.deepcopy(IDENTITY_4X4),
        "T_est": copy.deepcopy(IDENTITY_4X4),
        "Delta_T": copy.deepcopy(IDENTITY_4X4),
        "translation_x_m": 0.0,
        "translation_y_m": 0.0,
        "translation_z_m": 0.0,
        "translation_norm_m": 0.0,
        "rotation_angle_rad": 0.0,
        "rotation_angle_deg": 0.0,
        "solver_status": "CONVERGED",
        "finite_result": True,
        "scientific_status": "COMPLETED",
        "infrastructure_status": "OK",
        "retry_eligible": False,
        "retry_reason": None,
        "initial_correspondence_count": 100,
        "initial_valid_normal_correspondence_count": 95,
        "final_correspondence_count": 90,
        "final_valid_normal_correspondence_count": 85,
        "correspondence_turnover": 0.1,
        "accepted_source_turnover": 0.2,
        "correspondence_count_change_ratio": -0.1,
        "initial_residual_rmse": 0.01,
        "final_residual_rmse": 0.009,
        "residual_rmse_change": -0.001,
        "median_normal_angle_change_deg": 0.1,
        "q95_normal_angle_change_deg": 0.2,
        "common_association_valid": True,
        "common_association_invalid_reason": None,
        "common_association_invalid_detail": None,
        "backend_parameter_contract_sha256": trial["backend_parameter_contract_sha256"],
        "backend_canonical_parameter_sha256": trial["backend_canonical_parameter_sha256"],
        "active_amendment_sha256": trial["active_amendment_sha256"],
        "analysis_contract_sha256": trial["analysis_contract_sha256"],
        "trial_plan_sha256": "a" * 64,
        "formal_lock_sha256": "b" * 64,
        "code_commit": "c" * 40,
        "environment_identity": {
            "python": "3.11.15",
            "numpy": "1.26.4",
            "scipy": "1.11.4",
            "open3d": "0.19.0+b012259",
            "pcl": "1.15.1",
            "environment_manifest_sha256": "d" * 64,
        },
        "physical_reference_semantics": PHYSICAL_REFERENCE_SEMANTICS,
        "execution_kind": "FORMAL",
        "fixture_only": False,
        "created_at_utc": "2026-08-20T00:00:00+00:00",
    }


def test_future_result_row_is_bound_to_plan_without_running_backend(context: dict[str, object]) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    assert validate_result_row(
        row, context["plan"], trial_plan_sha256="a" * 64, formal_lock_sha256="b" * 64  # type: ignore[arg-type]
    ) == row


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_sha256", "0" * 64),
        ("target_sha256", "0" * 64),
        ("track_id", "CAPTURE_RADIUS_TRACK"),
        ("backend", "NATIVE"),
        ("fixture_only", True),
        ("physical_reference_semantics", "SUBMILLIMETER_GT"),
    ],
)
def test_future_result_binding_tamper_is_rejected(
    field: str, value: object, context: dict[str, object]
) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    row[field] = value
    with pytest.raises(R1TrialAssetError):
        validate_result_row(
            row, context["plan"], trial_plan_sha256="a" * 64, formal_lock_sha256="b" * 64  # type: ignore[arg-type]
        )


def test_nonfinite_transform_is_rejected(context: dict[str, object]) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    row["T_est"][0][0] = math.nan  # type: ignore[index]
    with pytest.raises(R1TrialAssetError, match="finite"):
        validate_result_row(
            row, context["plan"], trial_plan_sha256="a" * 64, formal_lock_sha256="b" * 64  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("angle_rad", [2.0e-9, 1.0e-8])
def test_tiny_finite_rotation_uses_frozen_atan2_metric_and_remains_terminal(
    context: dict[str, object], angle_rad: float,
) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    transform = copy.deepcopy(IDENTITY_4X4)
    transform[0][0], transform[0][1] = cosine, -sine
    transform[1][0], transform[1][1] = sine, cosine
    row["T_est"] = copy.deepcopy(transform)
    row["Delta_T"] = copy.deepcopy(transform)
    row["rotation_angle_rad"] = angle_rad
    row["rotation_angle_deg"] = math.degrees(angle_rad)
    validated = validate_result_row(
        row, context["plan"], trial_plan_sha256="a" * 64,
        formal_lock_sha256="b" * 64,  # type: ignore[arg-type]
    )
    assert validated["infrastructure_status"] == "OK"
    assert validated["finite_result"] is True
    assert validated["retry_eligible"] is False


def test_frozen_rotation_quality_threshold_is_the_only_so3_gate(
    context: dict[str, object],
) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    transform = copy.deepcopy(IDENTITY_4X4)
    transform[0][0] = 1.000001
    row["T_est"] = copy.deepcopy(transform)
    row["Delta_T"] = copy.deepcopy(transform)
    validated = validate_result_row(
        row, context["plan"], trial_plan_sha256="a" * 64,
        formal_lock_sha256="b" * 64,  # type: ignore[arg-type]
    )
    assert validated["infrastructure_status"] == "OK"
    assert validated["finite_result"] is True
    assert validated["retry_eligible"] is False


def test_frozen_rotation_quality_threshold_rejects_excessive_defect(
    context: dict[str, object],
) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    transform = copy.deepcopy(IDENTITY_4X4)
    transform[0][0] = 1.00002
    row["T_est"] = copy.deepcopy(transform)
    row["Delta_T"] = copy.deepcopy(transform)
    with pytest.raises(R1TrialAssetError, match="frozen quality audit"):
        validate_result_row(
            row, context["plan"], trial_plan_sha256="a" * 64,
            formal_lock_sha256="b" * 64,  # type: ignore[arg-type]
        )


def test_scientific_failure_cannot_be_marked_retryable(context: dict[str, object]) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    row["scientific_status"] = "SOLVER_NON_CONVERGENCE"
    row["retry_eligible"] = True
    row["retry_reason"] = "SOLVER_NON_CONVERGENCE"
    with pytest.raises(R1TrialAssetError, match="not retry eligible"):
        validate_result_row(
            row, context["plan"], trial_plan_sha256="a" * 64, formal_lock_sha256="b" * 64  # type: ignore[arg-type]
        )


def test_nonfinite_scientific_failure_is_retained_without_nan(context: dict[str, object]) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    row["finite_result"] = False
    row["scientific_status"] = "NONFINITE_RESULT_RECORDED_WITHOUT_NONFINITE_TRANSFORM"
    for field in (
        "T_est", "Delta_T", "translation_x_m", "translation_y_m", "translation_z_m",
        "translation_norm_m", "rotation_angle_rad", "rotation_angle_deg",
        "common_association_valid", "common_association_invalid_reason",
        "common_association_invalid_detail",
        "initial_correspondence_count", "initial_valid_normal_correspondence_count",
        "final_correspondence_count", "final_valid_normal_correspondence_count",
        "correspondence_turnover", "accepted_source_turnover",
        "correspondence_count_change_ratio", "initial_residual_rmse",
        "final_residual_rmse", "residual_rmse_change",
        "median_normal_angle_change_deg", "q95_normal_angle_change_deg",
    ):
        row[field] = None
    validated = validate_result_row(
        row, context["plan"], trial_plan_sha256="a" * 64, formal_lock_sha256="b" * 64  # type: ignore[arg-type]
    )
    assert validated["finite_result"] is False
    assert validated["retry_eligible"] is False


def test_nonfinite_pose_cannot_carry_reassociation_metrics(
    context: dict[str, object],
) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    row["finite_result"] = False
    row["scientific_status"] = "NONFINITE_RESULT_RECORDED_WITHOUT_NONFINITE_TRANSFORM"
    null_fields = (
        "T_est", "Delta_T", "translation_x_m", "translation_y_m", "translation_z_m",
        "translation_norm_m", "rotation_angle_rad", "rotation_angle_deg",
        "common_association_valid", "common_association_invalid_reason",
        "common_association_invalid_detail",
        "initial_valid_normal_correspondence_count", "final_correspondence_count",
        "final_valid_normal_correspondence_count", "correspondence_turnover",
        "accepted_source_turnover", "correspondence_count_change_ratio",
        "initial_residual_rmse", "final_residual_rmse", "residual_rmse_change",
        "median_normal_angle_change_deg", "q95_normal_angle_change_deg",
    )
    for field in null_fields:
        row[field] = None
    # Deliberately leave one exposed count to prove the cross-field gate.
    row["initial_correspondence_count"] = 100
    with pytest.raises(R1TrialAssetError, match="nonfinite pose"):
        validate_result_row(
            row, context["plan"], trial_plan_sha256="a" * 64,
            formal_lock_sha256="b" * 64,  # type: ignore[arg-type]
        )


def test_only_explicit_infrastructure_failure_is_retryable(context: dict[str, object]) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    row["infrastructure_status"] = "PROCESS_CRASH"
    row["scientific_status"] = "NOT_EVALUATED_INFRASTRUCTURE_FAILURE"
    row["retry_eligible"] = True
    row["retry_reason"] = "PROCESS_CRASH"
    row["finite_result"] = False
    for field in (
        "T_est", "Delta_T", "translation_x_m", "translation_y_m", "translation_z_m",
        "translation_norm_m", "rotation_angle_rad", "rotation_angle_deg",
        "initial_correspondence_count", "initial_valid_normal_correspondence_count",
        "final_correspondence_count", "final_valid_normal_correspondence_count",
        "correspondence_turnover", "accepted_source_turnover",
        "correspondence_count_change_ratio", "initial_residual_rmse",
        "final_residual_rmse", "residual_rmse_change",
        "median_normal_angle_change_deg", "q95_normal_angle_change_deg",
        "common_association_valid", "common_association_invalid_reason",
        "common_association_invalid_detail",
    ):
        row[field] = None
    assert validate_result_row(
        row, context["plan"], trial_plan_sha256="a" * 64, formal_lock_sha256="b" * 64  # type: ignore[arg-type]
    )["retry_eligible"] is True


def test_common_association_invalid_reason_is_retained_and_strict(
    context: dict[str, object],
) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    row["common_association_valid"] = False
    row["common_association_invalid_reason"] = "NO_FINAL_CORRESPONDENCE"
    row["common_association_invalid_detail"] = None
    assert validate_result_row(
        row, context["plan"], trial_plan_sha256="a" * 64,
        formal_lock_sha256="b" * 64,  # type: ignore[arg-type]
    )["common_association_invalid_reason"] == "NO_FINAL_CORRESPONDENCE"
    row["common_association_invalid_reason"] = "UNFROZEN_REASON"
    with pytest.raises(R1TrialAssetError, match="reason"):
        validate_result_row(
            row, context["plan"], trial_plan_sha256="a" * 64,
            formal_lock_sha256="b" * 64,  # type: ignore[arg-type]
        )
    row["common_association_invalid_reason"] = "OTHER"
    with pytest.raises(R1TrialAssetError, match="requires diagnostic"):
        validate_result_row(
            row, context["plan"], trial_plan_sha256="a" * 64,
            formal_lock_sha256="b" * 64,  # type: ignore[arg-type]
        )


def test_finite_result_cannot_drop_common_association_status(
    context: dict[str, object],
) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    row["common_association_valid"] = None
    with pytest.raises(R1TrialAssetError, match="retain"):
        validate_result_row(
            row, context["plan"], trial_plan_sha256="a" * 64,
            formal_lock_sha256="b" * 64,  # type: ignore[arg-type]
        )


def test_common_valid_cannot_hide_missing_exposed_metric(
    context: dict[str, object],
) -> None:
    row = _valid_result(context["plan"])  # type: ignore[arg-type]
    row["final_residual_rmse"] = None
    with pytest.raises(R1TrialAssetError, match="every exposed"):
        validate_result_row(
            row, context["plan"], trial_plan_sha256="a" * 64,
            formal_lock_sha256="b" * 64,  # type: ignore[arg-type]
        )
