"""Independent verifier for the FMB1 W04 pre-backend qualification bundle.

The verifier consumes only persisted evidence and immutable source contracts.
It does not import the evidence producer, fixture executor, registration guard,
or any backend module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, MutableSequence

import yaml


EXPECTED_BACKEND_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "76ae548874d8c1584fcc033685db4a1e7cf104fd881cbd1c334eba0bfe1a9beb"
)
EXPECTED_ANALYSIS_PROTOCOL_SHA256 = (
    "d453d12e713c546c5a054ceb1710b86eea12fa09e3f244705e46df9db255a879"
)
BLOCK_REASON = "MISSING_ADMITTED_WEAK_REPLACEMENT_W04"

JSON_EVIDENCE_NAMES = (
    "current_fmb1_state_reauthentication.json",
    "w04_replacement_plan_verification.json",
    "protocol_alignment_audit.json",
    "zero_perturbation_mainline_amendment_v1_1_PROPOSED.json",
    "current_real_batch_preflight_report.json",
    "formal_batch1_lock_schema.json",
    "formal_batch1_lock_template_UNISSUED.json",
    "formal_trial_result_schema_v1.json",
    "fixture_execution_path_qualification.json",
    "fixture_resume_interruption_report.json",
    "fixture_analysis_dry_run_report.json",
    "fixture_publication_dry_run_report.json",
    "NO_ICP_ATTESTATION_TONIGHT.json",
    "prebackend_qualification_summary.json",
)

REQUIRED_TOP_LEVEL_FILES = frozenset(
    {
        *JSON_EVIDENCE_NAMES,
        "current_fmb1_state_reauthentication.md",
        "W04_FIELD_CHECKLIST.md",
        "protocol_alignment_audit.md",
        "zero_perturbation_mainline_amendment_v1_1_PROPOSED.md",
        "current_real_batch_preflight_report.md",
        "fixture_execution_path_qualification.md",
        "prebackend_qualification_summary.md",
        "SHA256SUMS",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _check(
    checks: dict[str, bool],
    failures: MutableSequence[str],
    name: str,
    condition: Any,
) -> None:
    passed = condition is True
    checks[name] = passed
    if not passed:
        failures.append(name)


def _zero(payload: Mapping[str, Any], *names: str) -> bool:
    return all(payload.get(name, 0) == 0 for name in names)


def _false(payload: Mapping[str, Any], *names: str) -> bool:
    return all(payload.get(name, False) is False for name in names)


def _fixture_flags(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("FIXTURE_ONLY") is True
        and payload.get("FIXTURE_ONLY_DO_NOT_CITE") is True
        and payload.get("classification") == "FIXTURE_ONLY_DO_NOT_CITE"
        and payload.get("NOT_REAL_FMB1") is True
        and payload.get("NOT_FORMAL_MEASUREMENT") is True
        and payload.get("ACTUAL_REGISTRATION_EXECUTION") is False
        and payload.get("FORMAL_REGISTRATION_AUTHORIZED") is False
        and payload.get("FORMAL_ICP_UNLOCKED") is False
    )


def validate_evidence_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate cross-artifact logical invariants without reading files."""

    checks: dict[str, bool] = {}
    failures: list[str] = []
    missing = [name for name in JSON_EVIDENCE_NAMES if name not in payloads]
    _check(checks, failures, "all_required_json_payloads_present", not missing)
    if missing:
        return {"status": "FAIL", "pass": False, "checks": checks, "failures": failures}

    state = payloads["current_fmb1_state_reauthentication.json"]
    _check(
        checks,
        failures,
        "current_failure_closure_reauthenticated",
        state.get("FMB1_CURRENT_FAILURE_CLOSURE_REAUTHENTICATED") is True,
    )
    _check(
        checks,
        failures,
        "w02_rejection_preserved",
        state.get("W02_REJECTION_PRESERVED") is True
        and state.get("W02_REJECTION_REASON") == "GEOMETRY_ONLY_INELIGIBLE"
        and state.get("W02_FINAL_GEOMETRY_CLASS") == "RICH"
        and state.get("W02_GEOMETRY_ADMISSION_STATUS") == "GEOMETRY_REJECTED"
        and state.get("rejected_scenes") == ["FMB1_W02"],
    )
    _check(
        checks,
        failures,
        "w02_raw_data_retained",
        state.get("W02_DATA_RETAINED") is True
        and state.get("W02_RAW_BAG_COUNT") == 6
        and isinstance(state.get("W02_RAW_BAGS"), list)
        and len(state.get("W02_RAW_BAGS", [])) == 6
        and all(
            isinstance(row, Mapping) and row.get("scene_id") == "FMB1_W02"
            for row in state.get("W02_RAW_BAGS", [])
        ),
    )
    _check(
        checks,
        failures,
        "w02_excluded_from_future_formal_set",
        state.get("W02_INCLUDED_IN_FUTURE_FORMAL_SET") is False,
    )
    _check(
        checks,
        failures,
        "current_real_counts_are_3_rich_2_weak_5_total",
        state.get("admitted_rich_scene_count") == 3
        and state.get("admitted_weak_scene_count") == 2
        and state.get("admitted_scene_count") == 5
        and state.get("authenticated_bag_count") == 36
        and state.get("acquisition_pass_station_count") == 18
        and state.get("provisional_snapshot_count") == 180,
    )
    _check(
        checks,
        failures,
        "five_scenes_not_treated_as_complete",
        state.get("FMB1_CURRENT_REAL_BATCH_BLOCKED") is True
        and state.get("FMB1_CURRENT_BLOCK_REASON") == BLOCK_REASON
        and state.get("FMB1_PRE_REGISTRATION_DATA_READY") is False,
    )
    _check(
        checks,
        failures,
        "w04_currently_absent_and_unadmitted",
        state.get("W04_DATA_PRESENT") is False
        and state.get("W04_GEOMETRY_ADMITTED") is False,
    )
    _check(
        checks,
        failures,
        "current_real_authority_and_trials_zero",
        _false(
            state,
            "FORMAL_LOCK_ISSUED",
            "FORMAL_ICP_UNLOCKED",
            "FORMAL_REGISTRATION_AUTHORIZED",
        )
        and _zero(
            state,
            "actual_open3d_trials",
            "actual_pcl_trials",
            "actual_formal_trials",
        ),
    )
    protected_hashes = state.get("protected_file_sha256", {})
    _check(
        checks,
        failures,
        "state_binds_unchanged_protected_hashes",
        isinstance(protected_hashes, Mapping)
        and protected_hashes.get("backend_parameter_contract.json")
        == EXPECTED_BACKEND_SHA256
        and protected_hashes.get("preregistration.yaml")
        == EXPECTED_PREREGISTRATION_SHA256
        and protected_hashes.get("analysis_protocol.md")
        == EXPECTED_ANALYSIS_PROTOCOL_SHA256,
    )

    replacement = payloads["w04_replacement_plan_verification.json"]
    _check(
        checks,
        failures,
        "w04_plan_frozen_before_backend",
        replacement.get("FMB1_W04_REPLACEMENT_PLAN_FROZEN") is True
        and replacement.get("rejected_candidate_scene_id") == "FMB1_W02"
        and replacement.get("replacement_scene_id") == "FMB1_W04"
        and replacement.get("rejection_reason") == "GEOMETRY_ONLY_INELIGIBLE"
        and replacement.get("decision_before_any_icp") is True
        and replacement.get("formal_trial_count_at_decision") == 0,
    )
    _check(
        checks,
        failures,
        "w04_three_station_six_bag_plan",
        replacement.get("station_count") == 3
        and replacement.get("new_bag_count_required") == 6
        and replacement.get("station_ids")
        == ["FMB1_W04_S01", "FMB1_W04_S02", "FMB1_W04_S03"],
    )

    alignment = payloads["protocol_alignment_audit.json"]
    conclusion = alignment.get("conclusion", {})
    _check(
        checks,
        failures,
        "protocol_alignment_detects_track_offset",
        isinstance(conclusion, Mapping)
        and conclusion.get("zero_perturbation_mainline_active") is False
        and conclusion.get("capture_basin_preregistration_unchanged") is True
        and alignment.get("registration_execution_count") == 0,
    )

    amendment = payloads[
        "zero_perturbation_mainline_amendment_v1_1_PROPOSED.json"
    ]
    _check(
        checks,
        failures,
        "amendment_is_proposed_not_active",
        amendment.get("status") == "PROPOSED_NOT_ACTIVE"
        and amendment.get("FORMAL_AUTHORITY") is False
        and amendment.get("FORMAL_REGISTRATION_AUTHORIZED") is False
        and amendment.get("backend_execution_permitted") is False,
    )

    preflight = payloads["current_real_batch_preflight_report.json"]
    blocker_codes = preflight.get("scientific_blocker_codes")
    if blocker_codes is None:
        blocker_codes = [preflight.get("CURRENT_BLOCK_REASON")]
    _check(
        checks,
        failures,
        "real_preflight_fails_closed_for_w04_only",
        preflight.get("status") in {"BLOCKED", "FAIL_CLOSED"}
        and preflight.get("pass") is False
        and blocker_codes == [BLOCK_REASON]
        and preflight.get("admitted_weak_scene_count") == 2
        and preflight.get("required_weak_scene_count") == 3,
    )
    _check(
        checks,
        failures,
        "real_preflight_issues_no_matrix_lock_or_authority",
        preflight.get("TRIAL_MATRIX_ISSUED", preflight.get("FORMAL_RUN_MATRIX_ISSUED"))
        is False
        and preflight.get(
            "FORMAL_BATCH_LOCK_ISSUED", preflight.get("FORMAL_LOCK_ISSUED")
        )
        is False
        and preflight.get("FORMAL_REGISTRATION_AUTHORIZED") is False
        and preflight.get("FORMAL_ICP_UNLOCKED") is False
        and _zero(
            preflight,
            "open3d_registration_call_count",
            "pcl_cli_invocation_count",
            "other_registration_process_count",
            "formal_trial_count",
        ),
    )

    lock_template = payloads["formal_batch1_lock_template_UNISSUED.json"]
    _check(
        checks,
        failures,
        "formal_lock_template_is_unissued",
        lock_template.get("LOCK_STATUS") == "UNISSUED"
        and lock_template.get("REASON") == "W04_NOT_YET_ACQUIRED_OR_ADMITTED"
        and lock_template.get("FORMAL_LOCK_ISSUED") is False
        and lock_template.get("FORMAL_REGISTRATION_AUTHORIZED") is False,
    )

    schema = payloads["formal_trial_result_schema_v1.json"]
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    _check(
        checks,
        failures,
        "formal_trial_schema_is_strict_non_authorizing_and_track_aware",
        schema.get("additionalProperties") is False
        and schema.get("x-formal-authority") is False
        and schema.get("x-formal-registration-authorized") is False
        and all(
            field in required and field in properties
            for field in (
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
        )
        and properties.get("track_classification", {}).get("enum")
        == ["ZERO_PERTURBATION_TRACK", "CAPTURE_RADIUS_TRACK", "FIXTURE_ONLY"],
    )

    fixture = payloads["fixture_execution_path_qualification.json"]
    _check(
        checks,
        failures,
        "fixture_execution_path_isolated_and_qualified",
        _fixture_flags(fixture)
        and fixture.get("status") in {"PASS", "PASS_FIXTURE_ONLY"}
        and fixture.get("planned_snapshot_count", fixture.get("snapshot_count")) == 180
        and fixture.get("planned_open3d_trials") == 180
        and fixture.get("planned_pcl_trials") == 180
        and fixture.get("planned_total_trials", fixture.get("planned_trial_count")) == 360
        and fixture.get("actual_registration_execution_count", 0) == 0
        and fixture.get("real_scene_access_count", 0) == 0
        and fixture.get("real_backend_output_read_count", 0) == 0,
    )
    fixture_checks = fixture.get("checks", {})
    _check(
        checks,
        failures,
        "fixture_execution_all_internal_checks_pass",
        isinstance(fixture_checks, Mapping)
        and bool(fixture_checks)
        and all(value is True for value in fixture_checks.values()),
    )

    resume = payloads["fixture_resume_interruption_report.json"]
    resume_checks = resume.get("checks", {})
    fail_closed_cases = resume.get("fail_closed_cases", {})
    fail_closed_rows = (
        fail_closed_cases.get("cases", {})
        if isinstance(fail_closed_cases, Mapping)
        else {}
    )
    _check(
        checks,
        failures,
        "fixture_resume_interruption_pass",
        _fixture_flags(resume)
        and resume.get("status") in {"PASS", "PASS_FIXTURE_ONLY"}
        and resume.get("completed_before_interrupt", 0) > 0
        and resume.get("completed_after_resume") == 360
        and resume.get("duplicate_count") == 0
        and resume.get("missing_count") == 0
        and resume.get("checksum_mismatch_count") == 0
        and resume.get("orphan_count") == 0
        and resume.get("unchanged_result_sha_count", 0) > 0
        and isinstance(resume_checks, Mapping)
        and all(value is True for value in resume_checks.values())
        and isinstance(fail_closed_rows, Mapping)
        and {"partial", "orphan", "checksum", "manifest", "duplicate", "missing"}
        <= set(fail_closed_rows)
        and all(
            isinstance(row, Mapping) and row.get("resume_rejected") is True
            for row in fail_closed_rows.values()
        ),
    )

    analysis = payloads["fixture_analysis_dry_run_report.json"]
    publication = payloads["fixture_publication_dry_run_report.json"]
    _check(
        checks,
        failures,
        "fixture_analysis_preserves_scene_hierarchy",
        _fixture_flags(analysis)
        and analysis.get("scene_is_highest_independent_unit") is True
        and analysis.get("snapshots_are_independent_scenes") is False,
    )
    _check(
        checks,
        failures,
        "fixture_publication_is_non_citable",
        _fixture_flags(publication)
        and publication.get("publication_performed") is False
        and publication.get("citation_allowed") is False,
    )

    attestation = payloads["NO_ICP_ATTESTATION_TONIGHT.json"]
    _check(
        checks,
        failures,
        "tonight_no_icp_attestation_pass",
        attestation.get("pass") is True
        and _zero(
            attestation,
            "open3d_registration_call_count",
            "pcl_cli_invocation_count",
            "other_registration_process_count",
            "real_formal_trial_count",
            "real_T_est_file_count",
        ),
    )

    summary = payloads["prebackend_qualification_summary.json"]
    schema_qualification = summary.get("result_schema_qualification", {})
    _check(
        checks,
        failures,
        "summary_reports_qualified_path_and_blocked_real_batch",
        summary.get("FMB1_PREBACKEND_EXECUTION_PATH_QUALIFIED") is True
        and summary.get("FMB1_CURRENT_REAL_BATCH_BLOCKED") is True
        and summary.get("FMB1_CURRENT_BLOCK_REASON") == BLOCK_REASON
        and summary.get("FMB1_W04_REPLACEMENT_PLAN_FROZEN") is True
        and _false(
            summary,
            "FORMAL_LOCK_ISSUED",
            "FORMAL_ICP_UNLOCKED",
            "FORMAL_REGISTRATION_AUTHORIZED",
            "PROPOSED_AMENDMENT_ACTIVE",
        )
        and _zero(
            summary,
            "actual_open3d_trials",
            "actual_pcl_trials",
            "actual_formal_trials",
        )
        and isinstance(schema_qualification, Mapping)
        and schema_qualification.get("status") == "PASS"
        and all(
            schema_qualification.get(name) is True
            for name in (
                "missing_field_rejected",
                "nonfinite_rejected",
                "invalid_backend_rejected",
                "unlocked_identity_rejected",
                "zero_track_nonidentity_t0_rejected",
                "capture_track_cannot_masquerade_as_zero",
                "fixture_cannot_publish",
            )
        ),
    )

    return {
        "status": "PASS" if not failures else "FAIL",
        "pass": not failures,
        "checks": checks,
        "failures": failures,
    }


def _verify_checksum_file(output_dir: Path) -> tuple[bool, list[str]]:
    path = output_dir / "SHA256SUMS"
    if not path.is_file() or path.is_symlink():
        return False, ["SHA256SUMS missing or unsafe"]
    declared: dict[str, str] = {}
    failures: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if match is None or match.group(2) in declared:
            failures.append(f"malformed checksum row: {line}")
            continue
        digest, name = match.groups()
        candidate = output_dir / name
        if not candidate.is_file() or candidate.is_symlink():
            failures.append(f"checksum target missing or unsafe: {name}")
            continue
        if _sha256_file(candidate) != digest:
            failures.append(f"checksum mismatch: {name}")
        declared[name] = digest
    actual = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    # The independent report is created by this verifier.  Its absence from a
    # first-pass manifest is permitted only before the first report is written.
    unlisted = actual - set(declared)
    if unlisted:
        failures.append(f"unlisted qualification files: {sorted(unlisted)}")
    expected_listed = set(REQUIRED_TOP_LEVEL_FILES) - {"SHA256SUMS"}
    expected_listed.discard("prebackend_independent_verification.json")
    missing = expected_listed - set(declared)
    if missing:
        failures.append(f"required checksum entries missing: {sorted(missing)}")
    return not failures, failures


def verify_prebackend_qualification(
    repository: Path,
    output_dir: Path,
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    output_dir = output_dir.resolve(strict=True)
    payloads: dict[str, Mapping[str, Any]] = {}
    file_failures: list[str] = []
    for name in JSON_EVIDENCE_NAMES:
        path = output_dir / name
        try:
            payloads[name] = _load_json(path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            file_failures.append(f"{name}: {type(exc).__name__}: {exc}")
    logical = validate_evidence_payloads(payloads) if not file_failures else {
        "status": "FAIL",
        "pass": False,
        "checks": {},
        "failures": ["required_json_load"],
    }
    checks = dict(logical["checks"])
    failures = list(logical["failures"])
    failures.extend(file_failures)

    protected = {
        "backend_parameter_contract": repository / "frozen_assets/backend_parameter_contract.json",
        "preregistration": repository / "experiments/mid360_formal_batch1/preregistration.yaml",
        "analysis_protocol": repository / "experiments/mid360_formal_batch1/analysis_protocol.md",
    }
    expected = {
        "backend_parameter_contract": EXPECTED_BACKEND_SHA256,
        "preregistration": EXPECTED_PREREGISTRATION_SHA256,
        "analysis_protocol": EXPECTED_ANALYSIS_PROTOCOL_SHA256,
    }
    for name, path in protected.items():
        passed = path.is_file() and not path.is_symlink() and _sha256_file(path) == expected[name]
        _check(checks, failures, f"protected_{name}_unchanged", passed)

    lock_candidates = [
        repository / "results/mid360_formal_batch1/formal_batch1_lock.json",
        output_dir / "formal_batch1_lock.json",
    ]
    _check(
        checks,
        failures,
        "real_formal_lock_file_absent",
        not any(path.exists() for path in lock_candidates),
    )

    raw_inventory_path = repository / "results/mid360_formal_batch1/raw_bag_inventory.json"
    try:
        raw_inventory = _load_json(raw_inventory_path)
        raw_rows = raw_inventory.get("bags", [])
        w02_rows = [row for row in raw_rows if row.get("scene_id") == "FMB1_W02"]
        w04_rows = [row for row in raw_rows if row.get("scene_id") == "FMB1_W04"]
        raw_ok = len(raw_rows) == 36 and len(w02_rows) == 6 and not w04_rows
        for row in w02_rows:
            path = Path(row["raw_absolute_path"])
            raw_ok = bool(
                raw_ok
                and path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == row["bytes"]
                and _sha256_file(path) == row["sha256"]
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raw_ok = False
    _check(checks, failures, "w02_raw_files_rehashed_and_w04_data_absent", raw_ok)

    geometry_path = repository / "results/mid360_formal_batch1/geometry_scene_summary.csv"
    try:
        with geometry_path.open("r", encoding="utf-8", newline="") as stream:
            geometry_rows = list(csv.DictReader(stream))
        w02 = [row for row in geometry_rows if row.get("scene_id") == "FMB1_W02"]
        geometry_ok = bool(
            len(geometry_rows) == 6
            and len(w02) == 1
            and w02[0].get("final_geometry_class") == "RICH"
            and w02[0].get("geometry_admission_status") == "GEOMETRY_REJECTED"
            and not any(row.get("scene_id") == "FMB1_W04" for row in geometry_rows)
        )
    except (OSError, csv.Error):
        geometry_ok = False
    _check(checks, failures, "source_geometry_retains_w02_rejection", geometry_ok)

    plan_path = repository / "experiments/mid360_formal_batch1/replacement_plan_w04.yaml"
    try:
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        plan_ok = bool(
            isinstance(plan, Mapping)
            and plan.get("rejected_candidate_scene_id") == "FMB1_W02"
            and plan.get("rejection_reason") == "GEOMETRY_ONLY_INELIGIBLE"
            and plan.get("replacement_scene_id") == "FMB1_W04"
            and plan.get("decision_before_any_icp") is True
            and plan.get("formal_trial_count_at_decision") == 0
        )
    except (OSError, UnicodeError, yaml.YAMLError):
        plan_ok = False
    _check(checks, failures, "source_w04_replacement_plan_matches_evidence", plan_ok)

    checksum_pass, checksum_failures = _verify_checksum_file(output_dir)
    _check(checks, failures, "qualification_sha256_manifest_pass", checksum_pass)
    failures.extend(checksum_failures)

    status = "PASS" if not failures else "FAIL"
    return {
        "schema": "mid360_fmb1_prebackend_independent_verification_v1",
        "status": status,
        "pass": not failures,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "FMB1_PREBACKEND_EXECUTION_PATH_QUALIFIED": not failures,
        "FMB1_CURRENT_REAL_BATCH_BLOCKED": True,
        "FMB1_CURRENT_BLOCK_REASON": BLOCK_REASON,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    }


__all__ = [
    "JSON_EVIDENCE_NAMES",
    "REQUIRED_TOP_LEVEL_FILES",
    "validate_evidence_payloads",
    "verify_prebackend_qualification",
]
