from __future__ import annotations

import copy
from typing import Any, Callable

import pytest

from experiments.mid360_formal_batch1.prebackend_verify import (
    EXPECTED_ANALYSIS_PROTOCOL_SHA256,
    EXPECTED_BACKEND_SHA256,
    EXPECTED_PREREGISTRATION_SHA256,
    validate_evidence_payloads,
)


def _fixture_flags() -> dict[str, Any]:
    return {
        "classification": "FIXTURE_ONLY_DO_NOT_CITE",
        "FIXTURE_ONLY": True,
        "FIXTURE_ONLY_DO_NOT_CITE": True,
        "NOT_REAL_FMB1": True,
        "NOT_FORMAL_MEASUREMENT": True,
        "ACTUAL_REGISTRATION_EXECUTION": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "FORMAL_ICP_UNLOCKED": False,
    }


def _valid_payloads() -> dict[str, dict[str, Any]]:
    fixture_flags = _fixture_flags()
    fail_cases = {
        name: {"resume_rejected": True}
        for name in ("partial", "orphan", "checksum", "manifest", "duplicate", "missing")
    }
    required_schema_fields = (
        "batch_id",
        "scene_id",
        "station_id",
        "snapshot_id",
        "backend",
        "source_sha256",
        "target_sha256",
        "T0",
        "T_est",
        "track",
        "track_classification",
        "translation_vector_m",
        "translation_norm_m",
        "rotation_angle_deg",
        "solver_status",
        "finite_result",
    )
    schema_qualification = {
        "status": "PASS",
        "missing_field_rejected": True,
        "nonfinite_rejected": True,
        "invalid_backend_rejected": True,
        "unlocked_identity_rejected": True,
        "zero_track_nonidentity_t0_rejected": True,
        "capture_track_cannot_masquerade_as_zero": True,
        "fixture_cannot_publish": True,
    }
    return {
        "current_fmb1_state_reauthentication.json": {
            "FMB1_CURRENT_FAILURE_CLOSURE_REAUTHENTICATED": True,
            "W02_REJECTION_PRESERVED": True,
            "W02_REJECTION_REASON": "GEOMETRY_ONLY_INELIGIBLE",
            "W02_FINAL_GEOMETRY_CLASS": "RICH",
            "W02_GEOMETRY_ADMISSION_STATUS": "GEOMETRY_REJECTED",
            "rejected_scenes": ["FMB1_W02"],
            "W02_DATA_RETAINED": True,
            "W02_RAW_BAG_COUNT": 6,
            "W02_RAW_BAGS": [{"scene_id": "FMB1_W02"} for _ in range(6)],
            "W02_INCLUDED_IN_FUTURE_FORMAL_SET": False,
            "admitted_rich_scene_count": 3,
            "admitted_weak_scene_count": 2,
            "admitted_scene_count": 5,
            "authenticated_bag_count": 36,
            "acquisition_pass_station_count": 18,
            "provisional_snapshot_count": 180,
            "FMB1_CURRENT_REAL_BATCH_BLOCKED": True,
            "FMB1_CURRENT_BLOCK_REASON": "MISSING_ADMITTED_WEAK_REPLACEMENT_W04",
            "FMB1_PRE_REGISTRATION_DATA_READY": False,
            "W04_DATA_PRESENT": False,
            "W04_GEOMETRY_ADMITTED": False,
            "FORMAL_LOCK_ISSUED": False,
            "FORMAL_ICP_UNLOCKED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "actual_open3d_trials": 0,
            "actual_pcl_trials": 0,
            "actual_formal_trials": 0,
            "protected_file_sha256": {
                "backend_parameter_contract.json": EXPECTED_BACKEND_SHA256,
                "preregistration.yaml": EXPECTED_PREREGISTRATION_SHA256,
                "analysis_protocol.md": EXPECTED_ANALYSIS_PROTOCOL_SHA256,
            },
        },
        "w04_replacement_plan_verification.json": {
            "FMB1_W04_REPLACEMENT_PLAN_FROZEN": True,
            "rejected_candidate_scene_id": "FMB1_W02",
            "replacement_scene_id": "FMB1_W04",
            "rejection_reason": "GEOMETRY_ONLY_INELIGIBLE",
            "decision_before_any_icp": True,
            "formal_trial_count_at_decision": 0,
            "station_count": 3,
            "new_bag_count_required": 6,
            "station_ids": ["FMB1_W04_S01", "FMB1_W04_S02", "FMB1_W04_S03"],
        },
        "protocol_alignment_audit.json": {
            "registration_execution_count": 0,
            "conclusion": {
                "zero_perturbation_mainline_active": False,
                "capture_basin_preregistration_unchanged": True,
            },
        },
        "zero_perturbation_mainline_amendment_v1_1_PROPOSED.json": {
            "status": "PROPOSED_NOT_ACTIVE",
            "FORMAL_AUTHORITY": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "backend_execution_permitted": False,
        },
        "current_real_batch_preflight_report.json": {
            "status": "BLOCKED",
            "pass": False,
            "scientific_blocker_codes": ["MISSING_ADMITTED_WEAK_REPLACEMENT_W04"],
            "admitted_weak_scene_count": 2,
            "required_weak_scene_count": 3,
            "TRIAL_MATRIX_ISSUED": False,
            "FORMAL_BATCH_LOCK_ISSUED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "FORMAL_ICP_UNLOCKED": False,
            "open3d_registration_call_count": 0,
            "pcl_cli_invocation_count": 0,
            "other_registration_process_count": 0,
            "formal_trial_count": 0,
        },
        "formal_batch1_lock_schema.json": {},
        "formal_batch1_lock_template_UNISSUED.json": {
            "LOCK_STATUS": "UNISSUED",
            "REASON": "W04_NOT_YET_ACQUIRED_OR_ADMITTED",
            "FORMAL_LOCK_ISSUED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
        },
        "formal_trial_result_schema_v1.json": {
            "additionalProperties": False,
            "x-formal-authority": False,
            "x-formal-registration-authorized": False,
            "required": list(required_schema_fields),
            "properties": {
                **{name: {} for name in required_schema_fields},
                "track_classification": {
                    "enum": [
                        "ZERO_PERTURBATION_TRACK",
                        "CAPTURE_RADIUS_TRACK",
                        "FIXTURE_ONLY",
                    ]
                },
            },
        },
        "fixture_execution_path_qualification.json": {
            **fixture_flags,
            "status": "PASS_FIXTURE_ONLY",
            "planned_snapshot_count": 180,
            "planned_open3d_trials": 180,
            "planned_pcl_trials": 180,
            "planned_total_trials": 360,
            "actual_registration_execution_count": 0,
            "real_scene_access_count": 0,
            "real_backend_output_read_count": 0,
            "checks": {"lifecycle": True},
        },
        "fixture_resume_interruption_report.json": {
            **fixture_flags,
            "status": "PASS_FIXTURE_ONLY",
            "completed_before_interrupt": 73,
            "completed_after_resume": 360,
            "duplicate_count": 0,
            "missing_count": 0,
            "checksum_mismatch_count": 0,
            "orphan_count": 0,
            "unchanged_result_sha_count": 73,
            "checks": {"resume": True},
            "fail_closed_cases": {"cases": fail_cases},
        },
        "fixture_analysis_dry_run_report.json": {
            **fixture_flags,
            "scene_is_highest_independent_unit": True,
            "snapshots_are_independent_scenes": False,
        },
        "fixture_publication_dry_run_report.json": {
            **fixture_flags,
            "publication_performed": False,
            "citation_allowed": False,
        },
        "NO_ICP_ATTESTATION_TONIGHT.json": {
            "pass": True,
            "open3d_registration_call_count": 0,
            "pcl_cli_invocation_count": 0,
            "other_registration_process_count": 0,
            "real_formal_trial_count": 0,
            "real_T_est_file_count": 0,
        },
        "prebackend_qualification_summary.json": {
            "FMB1_PREBACKEND_EXECUTION_PATH_QUALIFIED": True,
            "FMB1_CURRENT_REAL_BATCH_BLOCKED": True,
            "FMB1_CURRENT_BLOCK_REASON": "MISSING_ADMITTED_WEAK_REPLACEMENT_W04",
            "FMB1_W04_REPLACEMENT_PLAN_FROZEN": True,
            "FORMAL_LOCK_ISSUED": False,
            "FORMAL_ICP_UNLOCKED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "PROPOSED_AMENDMENT_ACTIVE": False,
            "actual_open3d_trials": 0,
            "actual_pcl_trials": 0,
            "actual_formal_trials": 0,
            "result_schema_qualification": schema_qualification,
        },
    }


def test_logically_valid_qualification_payloads_pass() -> None:
    report = validate_evidence_payloads(_valid_payloads())
    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["failures"] == []


def _set(payloads: dict[str, dict[str, Any]], file_name: str, key: str, value: Any) -> None:
    payloads[file_name][key] = value


Tamper = Callable[[dict[str, dict[str, Any]]], None]


@pytest.mark.parametrize(
    ("case", "tamper"),
    [
        (
            "w02_relabelled_weak",
            lambda p: _set(p, "current_fmb1_state_reauthentication.json", "W02_FINAL_GEOMETRY_CLASS", "WEAK"),
        ),
        (
            "w02_rejection_deleted",
            lambda p: _set(p, "current_fmb1_state_reauthentication.json", "W02_REJECTION_PRESERVED", False),
        ),
        (
            "w02_data_renamed_w04",
            lambda p: p["current_fmb1_state_reauthentication.json"]["W02_RAW_BAGS"][0].update(scene_id="FMB1_W04"),
        ),
        (
            "weak_count_forged_three",
            lambda p: _set(p, "current_fmb1_state_reauthentication.json", "admitted_weak_scene_count", 3),
        ),
        (
            "lock_issued_without_w04",
            lambda p: _set(p, "formal_batch1_lock_template_UNISSUED.json", "FORMAL_LOCK_ISSUED", True),
        ),
        (
            "authorization_true",
            lambda p: _set(p, "prebackend_qualification_summary.json", "FORMAL_REGISTRATION_AUTHORIZED", True),
        ),
        (
            "real_backend_result_added",
            lambda p: _set(p, "NO_ICP_ATTESTATION_TONIGHT.json", "real_T_est_file_count", 1),
        ),
        (
            "open3d_called",
            lambda p: _set(p, "NO_ICP_ATTESTATION_TONIGHT.json", "open3d_registration_call_count", 1),
        ),
        (
            "pcl_cli_called",
            lambda p: _set(p, "NO_ICP_ATTESTATION_TONIGHT.json", "pcl_cli_invocation_count", 1),
        ),
        (
            "backend_contract_sha_changed",
            lambda p: p["current_fmb1_state_reauthentication.json"]["protected_file_sha256"].update({"backend_parameter_contract.json": "0" * 64}),
        ),
        (
            "admitted_snapshot_changed",
            lambda p: _set(p, "current_fmb1_state_reauthentication.json", "provisional_snapshot_count", 179),
        ),
        (
            "fixture_row_marked_formal",
            lambda p: _set(p, "fixture_execution_path_qualification.json", "FIXTURE_ONLY", False),
        ),
        (
            "capture_radius_masquerades_as_zero",
            lambda p: p["prebackend_qualification_summary.json"]["result_schema_qualification"].update(capture_track_cannot_masquerade_as_zero=False),
        ),
        (
            "zero_track_nonidentity_t0",
            lambda p: p["prebackend_qualification_summary.json"]["result_schema_qualification"].update(zero_track_nonidentity_t0_rejected=False),
        ),
        (
            "resume_duplicate_accepted",
            lambda p: _set(p, "fixture_resume_interruption_report.json", "duplicate_count", 1),
        ),
        (
            "orphan_autoaccepted",
            lambda p: p["fixture_resume_interruption_report.json"]["fail_closed_cases"]["cases"]["orphan"].update(resume_rejected=False),
        ),
        (
            "amendment_activated",
            lambda p: _set(p, "zero_perturbation_mainline_amendment_v1_1_PROPOSED.json", "status", "ACTIVE"),
        ),
        (
            "five_scenes_called_complete",
            lambda p: _set(p, "current_fmb1_state_reauthentication.json", "FMB1_CURRENT_REAL_BATCH_BLOCKED", False),
        ),
    ],
)
def test_each_required_prebackend_tamper_fails(
    case: str, tamper: Tamper
) -> None:
    payloads = copy.deepcopy(_valid_payloads())
    tamper(payloads)
    report = validate_evidence_payloads(payloads)
    assert report["status"] == "FAIL", case
    assert report["pass"] is False, case
    assert report["failures"], case
