from __future__ import annotations

import json
from pathlib import Path


from experiments.mid360_formal_batch1.reporting_correction.solver_status_reporting_correction_v1 import (
    ACTUAL_VALID_SOLVER_STATUS,
    ORIGINAL_ANALYSIS_SUMMARY_SHA256,
    ORIGINAL_SUMMARY_RELATIVE,
    build_reporting_correction,
    corrected_accounting,
    load_formal_rows,
    sha256_file,
)
from experiments.mid360_formal_batch1.reporting_correction.reporting_correction_verify_v1 import (
    verify_reporting_correction,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def test_authoritative_formal_status_accounting() -> None:
    rows = load_formal_rows(REPOSITORY)
    accounting = corrected_accounting(rows)
    assert accounting["formal_trial_count"] == 360
    assert accounting["formal_finite_result_n"] == 360
    assert accounting["formal_completed_n"] == 360
    assert accounting["formal_solver_nonconvergence_n"] == 0
    assert accounting["formal_infrastructure_ok_n"] == 360
    assert accounting["pcl_native_has_converged_true_n"] == 180
    assert accounting["pcl_native_has_converged_false_n"] == 0
    assert accounting["open3d_native_convergence_observable"] is False
    for backend in ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE"):
        assert accounting["by_backend"][backend]["solver_status_inventory"] == {
            ACTUAL_VALID_SOLVER_STATUS: 180
        }


def test_original_summary_identity_is_frozen() -> None:
    assert (
        sha256_file(REPOSITORY / ORIGINAL_SUMMARY_RELATIVE)
        == ORIGINAL_ANALYSIS_SUMMARY_SHA256
    )


def test_builder_changes_only_reporting_correction_path(tmp_path: Path) -> None:
    output = tmp_path / "correction"
    build_reporting_correction(REPOSITORY, output, generated_at_utc="2026-08-20T00:00:00Z")
    diff = json.loads((output / "json_path_correction_diff.json").read_text())
    assert diff["status"] == "PASS"
    assert diff["changed_path_count"] == 1
    assert [row["json_path"] for row in diff["changes"]] == [
        "/reporting_correction"
    ]
    assert diff["forbidden_scientific_path_change_count"] == 0


def test_scientific_values_are_semantically_identical(tmp_path: Path) -> None:
    output = tmp_path / "correction"
    build_reporting_correction(REPOSITORY, output, generated_at_utc="2026-08-20T00:00:00Z")
    check = json.loads(
        (output / "scientific_value_immutability_check.json").read_text()
    )
    assert check["status"] == "PASS"
    assert check[
        "ALL_ORIGINAL_SUMMARY_CONTENT_IDENTICAL_AFTER_CORRECTION_METADATA_REMOVAL"
    ] is True
    assert check["PRIMARY_SCIENTIFIC_NUMERICAL_RESULTS_CHANGED"] is False
    assert check["new_hypothesis_tests_added"] is False


def test_independent_verifier_passes_without_backend_calls(tmp_path: Path) -> None:
    output = tmp_path / "correction"
    build_reporting_correction(REPOSITORY, output, generated_at_utc="2026-08-20T00:00:00Z")
    report = verify_reporting_correction(REPOSITORY, output)
    assert report["INDEPENDENT_REPORTING_CORRECTION_VERIFICATION_PASS"] is True
    assert report["failure_count"] == 0
    assert report["registration_backend_call_count"] == 0
    assert report["formal_scientific_analysis_run_count"] == 0
    assert report["primary_estimand_recomputation_count"] == 0
    assert report["p_value_recomputation_count"] == 0
