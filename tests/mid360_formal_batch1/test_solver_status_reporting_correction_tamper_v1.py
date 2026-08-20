from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.mid360_formal_batch1.reporting_correction.solver_status_reporting_correction_v1 import (
    ORIGINAL_SUMMARY_RELATIVE,
    build_reporting_correction,
)
from experiments.mid360_formal_batch1.reporting_correction.reporting_correction_verify_v1 import (
    collect_observed,
    validate_payloads,
)


REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    output = tmp_path_factory.mktemp("reporting-correction") / "bundle"
    build_reporting_correction(
        REPOSITORY, output, generated_at_utc="2026-08-20T00:00:00Z"
    )
    names = {
        "notice": "reporting_correction_notice_v1.json",
        "accounting": "corrected_solver_status_accounting_v1.json",
        "corrected": "analysis_summary_reporting_corrected_v1.json",
        "diff_payload": "json_path_correction_diff.json",
        "immutability": "scientific_value_immutability_check.json",
        "manifest": "reporting_correction_manifest.json",
    }
    payload = {
        key: json.loads((output / name).read_text()) for key, name in names.items()
    }
    payload["original"] = json.loads(
        (REPOSITORY / ORIGINAL_SUMMARY_RELATIVE).read_text()
    )
    payload["observed"] = collect_observed(REPOSITORY, output)
    return payload


def _tamper(payload: dict[str, Any], case: str) -> None:
    if case == "completed_359":
        payload["accounting"]["formal_completed_n"] = 359
    elif case == "nonconvergence_1":
        payload["accounting"]["formal_solver_nonconvergence_n"] = 1
    elif case == "open3d_native_180":
        section = payload["accounting"]
        section["open3d_native_convergence_observable"] = True
        section["open3d_native_has_converged_true_n"] = 180
    elif case == "pcl_native_179":
        payload["accounting"]["pcl_native_has_converged_true_n"] = 179
    elif case == "primary_p_065":
        payload["corrected"]["primary_translation_inference"][0]["p_value"] = 0.65
    elif case == "translation_estimand_digit":
        payload["corrected"]["primary_translation_inference"][0]["estimand"] += 1e-12
    elif case == "centered_rho":
        payload["corrected"]["reassociation_centered_association"][0]["rho"] = 0.1
    elif case == "systematic_fraction":
        payload["corrected"]["systematic_scene_values"][0]["systematic_fraction"] = 0.1
    elif case == "wrong_original_sha":
        payload["manifest"]["original_scientific_analysis"]["analysis_summary_sha256"] = "0" * 64
    elif case == "wrong_diagnostic_commit":
        payload["manifest"]["solver_diagnostic"]["commit"] = "0" * 40
    elif case == "wrong_result_commit":
        payload["manifest"]["original_scientific_analysis"]["commit"] = "0" * 40
    elif case == "changed_raw_result":
        payload["observed"]["raw_execution_unchanged"] = False
    elif case == "old_tag_moved":
        payload["observed"]["result_tag_peel"] = "0" * 40
    elif case == "forbidden_json_path":
        payload["diff_payload"]["changes"].append(
            {
                "json_path": "/primary_translation_inference/0/p_value",
                "old_value": 0.7,
                "new_value": 0.65,
                "reason": "UNEXPECTED_VALUE_CHANGE",
            }
        )
    elif case == "registration_rerun":
        payload["notice"]["REGISTRATION_RERUN"] = True
    elif case == "scientific_analysis_rerun":
        payload["notice"]["SCIENTIFIC_ANALYSIS_RERUN"] = True
    elif case == "open3d_native_claim_true":
        payload["corrected"]["reporting_correction"][
            "open3d_native_convergence_observable"
        ] = True
    elif case == "malformed_manifest":
        del payload["manifest"]["status"]
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "completed_359",
        "nonconvergence_1",
        "open3d_native_180",
        "pcl_native_179",
        "primary_p_065",
        "translation_estimand_digit",
        "centered_rho",
        "systematic_fraction",
        "wrong_original_sha",
        "wrong_diagnostic_commit",
        "wrong_result_commit",
        "changed_raw_result",
        "old_tag_moved",
        "forbidden_json_path",
        "registration_rerun",
        "scientific_analysis_rerun",
        "open3d_native_claim_true",
        "malformed_manifest",
    ],
)
def test_tamper_fails_closed(baseline: dict[str, Any], case: str) -> None:
    payload = copy.deepcopy(baseline)
    _tamper(payload, case)
    failures = validate_payloads(**payload)
    assert failures, case
