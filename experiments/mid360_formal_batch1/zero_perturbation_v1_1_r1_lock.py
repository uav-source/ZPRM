"""Formal execution-lock builder for zero-perturbation v1.1-R1.

Building a payload is not authorization.  The issued lock deliberately keeps
the ICP unlock and registration authorization false and contains no result.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import subprocess
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .zero_perturbation_v1_1_r1_environment import (
    BACKEND_CONTRACT_SHA256,
    verify_environment_manifest,
)


LOCK_SCHEMA = "mid360_fmb1_zero_perturbation_formal_lock_v1_1_r1"
LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1.json"
AMENDMENT_ID = "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1"
PLAN_SCHEMA = "mid360_fmb1_zero_perturbation_trial_plan_v1_1_r1"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ORIGINAL_PROPOSAL_JSON_SHA256 = (
    "4837f6bd37e3a0f19c4b276964bbd066df09b86e58af288aa103aca704be936e"
)
ORIGINAL_PROPOSAL_MD_SHA256 = (
    "836822c553929a2e5b9e6cdf85b48d318fe71f91aa47447d908d030516e7f4ed"
)

DEFAULT_BINDING_PATHS = {
    "original_preregistration": "experiments/mid360_formal_batch1/preregistration.yaml",
    "original_capture_radius_analysis_protocol": "experiments/mid360_formal_batch1/analysis_protocol.md",
    "active_amendment": "experiments/mid360_formal_batch1/amendments/zero_perturbation_mainline_v1_1_r1.json",
    "active_amendment_md": "experiments/mid360_formal_batch1/amendments/zero_perturbation_mainline_v1_1_r1.md",
    "amendment_activation_record": "experiments/mid360_formal_batch1/amendments/amendment_activation_record_v1_1_r1.json",
    "active_protocol_pointer": "experiments/mid360_formal_batch1/ACTIVE_PROTOCOL.json",
    "final_dataset_pointer": "results/mid360_formal_batch1/CURRENT_FINAL_DATASET.json",
    "final_scene_registry": "results/mid360_formal_batch1/final_dataset_v1/final_scene_registry.yaml",
    "final_station_registry": "results/mid360_formal_batch1/final_dataset_v1/final_station_registry.yaml",
    "admitted_bag_manifest": "results/mid360_formal_batch1/final_dataset_v1/final_raw_bag_manifest.csv",
    "acquisition_attempt_lineage": "results/mid360_formal_batch1/final_dataset_v1/acquisition_attempt_lineage.json",
    "invalid_attempt_archive_manifest": "results/mid360_formal_batch1/final_dataset_v1/invalid_attempt_archive_manifest.json",
    "target_manifest": "results/mid360_formal_batch1/final_dataset_v1/final_target_manifest.csv",
    "snapshot_manifest": "results/mid360_formal_batch1/final_dataset_v1/final_snapshot_manifest.csv",
    "geometry_manifest": "results/mid360_formal_batch1/final_dataset_v1/final_geometry_manifest.csv",
    "trial_plan_json": "experiments/mid360_formal_batch1/zero_perturbation_trial_plan_v1_1.json",
    "trial_plan_csv": "experiments/mid360_formal_batch1/zero_perturbation_trial_plan_v1_1.csv",
    "result_schema": "experiments/mid360_formal_batch1/zero_perturbation_trial_result_schema_v1_1.json",
    "analysis_contract": "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_contract_v1_1_r1.json",
    "analysis_protocol": "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_protocol_v1_1_r1.md",
    "analysis_missingness_clarification": "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1.json",
    "analysis_missingness_clarification_md": "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1.md",
    "analysis_missingness_clarification_transition": "experiments/mid360_formal_batch1/amendments/analysis_missingness_clarification_transition_v1_1_r1_c1.json",
    "analysis_preclarification_history_inventory": "experiments/mid360_formal_batch1/amendments/history/zero_perturbation_v1_1_r1_active_pre_missingness_clarification/active_preclarification_inventory.json",
    "backend_parameter_contract": "frozen_assets/backend_parameter_contract.json",
    "pcl_executable": "bin/pcl_point_to_plane_cli",
    "environment_manifest": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/environment_manifest.json",
    "prelock_no_icp_attestation": "results/mid360_formal_batch1/final_dataset_v1/NO_ICP_ATTESTATION.json",
    "final_dataset_prelock_reauthentication": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/final_dataset_prelock_reauthentication.json",
    "protocol_transition_independent_verification": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/protocol_r1_active_transition_independent_verification.json",
    "protocol_c1_missingness_independent_verification": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/protocol_r1_c1_missingness_independent_verification.json",
    "activation_review": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/zero_perturbation_v1_1_activation_review.json",
    "trial_plan_independent_verification": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/trial_plan_independent_verification.json",
    "original_zero_perturbation_proposal_json": "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.json",
    "original_zero_perturbation_proposal_md": "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.md",
    "proposal_superseded_sidecar": "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.SUPERSEDED.json",
    "proposal_correction_record": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/proposal_correction_record_v1_1_r1.json",
    "proposal_difference_report": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/proposal_difference_report_v1_1_r1.json",
    "w04_superseded_history": "results/mid360_formal_batch1/history/final_dataset_w04_replacement_superseded_v1.SUPERSEDED.json",
    "execution_runner": "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_runner.py",
    "execution_runner_cli": "tools/mid360_formal_batch1/run_zero_perturbation_v1_1_r1.py",
    "execution_experiments_package_init": "experiments/__init__.py",
    "execution_mid360_formal_batch1_package_init": "experiments/mid360_formal_batch1/__init__.py",
    "execution_phase_a_harness_package_init": "src/phase_a_harness/__init__.py",
    "execution_environment": "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_environment.py",
    "execution_result_validator": "experiments/mid360_formal_batch1/zero_perturbation_r1_trial_assets.py",
    "execution_open3d_backend": "src/phase_a_harness/open3d_backend.py",
    "execution_pcl_backend": "src/phase_a_harness/pcl_backend.py",
    "execution_common_association": "src/phase_a_harness/common_association_analysis.py",
    "execution_rotation_metrics": "src/phase_a_harness/rotation_metrics.py",
    "execution_metrics": "src/phase_a_harness/metrics.py",
    "execution_types": "src/phase_a_harness/types.py",
}
EXECUTION_BINDING_NAMES = tuple(
    name for name in DEFAULT_BINDING_PATHS if name.startswith("execution_")
)
AUTHORITATIVE_RUNTIME_ROOT = (
    "zero_perturbation_runtime/mid360_formal_batch1_zero_perturbation_v1_1"
)
DEFAULT_AUTHORIZATION_PATH = (
    "results/mid360_formal_batch1/"
    "formal_registration_authorization_v1_1_r1.json"
)
LOCK_CORE_FILENAMES = (
    LOCK_FILENAME,
    "formal_batch1_zero_perturbation_lock_v1_1.sha256",
    "lock_inventory.csv",
    "lock_fingerprint.json",
    "NO_ICP_ATTESTATION.json",
    "environment_manifest.json",
    "final_dataset_prelock_reauthentication.json",
)


class R1LockError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise R1LockError(f"FMB1_R1_LOCK_FAIL: {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        _fail(f"expected JSON mapping: {path}")
    return value


def _resolve_binding(root: Path, value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail(f"binding {label} is lexically outside repository")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"binding {label} uses a symlink component: {cursor}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"binding {label} escapes repository")
    if not resolved.is_file():
        _fail(f"binding {label} is not a regular non-symlink file")
    return resolved


def _assert_zero_no_icp(payload: Mapping[str, Any]) -> None:
    for key in (
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "formal_trial_count",
        "actual_open3d_trials",
        "actual_pcl_trials",
        "actual_formal_trials",
        "actual_registration_trials",
        "registration_execution_count",
    ):
        if int(payload.get(key, 0)) != 0:
            _fail(f"prelock counter is nonzero: {key}")
    for key in ("FORMAL_ICP_UNLOCKED", "FORMAL_REGISTRATION_AUTHORIZED"):
        if payload.get(key) is not False:
            _fail(f"prelock authority flag must be false: {key}")


def _assert_no_execution_lifecycle(root: Path) -> None:
    """Require a pristine formal runtime and no separate authorization.

    A started/inflight marker is execution lifecycle evidence even if no
    terminal result row exists.  Every file in this dedicated runtime is
    therefore disqualifying before authorization and lock issuance.
    """

    authorization = root / DEFAULT_AUTHORIZATION_PATH
    if authorization.exists() or authorization.is_symlink():
        _fail("separate formal-registration authorization already exists")
    runtime = root / AUTHORITATIVE_RUNTIME_ROOT
    if runtime.is_symlink() or (runtime.exists() and not runtime.is_dir()):
        _fail("authoritative runtime is a symlink or non-directory")
    if runtime.exists():
        lifecycle_files = sorted(
            path.relative_to(runtime).as_posix()
            for path in runtime.rglob("*") if path.is_file() or path.is_symlink()
        )
        if lifecycle_files:
            _fail(
                "authoritative runtime already contains execution lifecycle "
                f"artifacts: {lifecycle_files[:3]}"
            )


def _validate_proposal_correction_history(paths: Mapping[str, Path]) -> None:
    proposal_json = paths["original_zero_perturbation_proposal_json"]
    proposal_md = paths["original_zero_perturbation_proposal_md"]
    if _sha256(proposal_json) != ORIGINAL_PROPOSAL_JSON_SHA256:
        _fail("historical proposal JSON is not byte-identical")
    if _sha256(proposal_md) != ORIGINAL_PROPOSAL_MD_SHA256:
        _fail("historical proposal Markdown is not byte-identical")

    superseded = _json(paths["proposal_superseded_sidecar"])
    historical_json = superseded.get("historical_proposal_json", {})
    historical_md = superseded.get("historical_proposal_markdown", {})
    if (
        superseded.get("status") != "SUPERSEDED_PROPOSAL"
        or superseded.get("historical_proposal_modified") is not False
        or superseded.get("correction_reason_code")
        != "OPERATOR_CONFIRMED_WRONG_SCENE_LOCATION"
        or superseded.get("correction_before_any_formal_icp") is not True
        or int(superseded.get("correction_at_formal_trial_count", -1)) != 0
        or superseded.get("registration_evidence_used") is not False
        or historical_json.get("path")
        != DEFAULT_BINDING_PATHS["original_zero_perturbation_proposal_json"]
        or historical_json.get("sha256") != ORIGINAL_PROPOSAL_JSON_SHA256
        or historical_md.get("path")
        != DEFAULT_BINDING_PATHS["original_zero_perturbation_proposal_md"]
        or historical_md.get("sha256") != ORIGINAL_PROPOSAL_MD_SHA256
    ):
        _fail("historical proposal supersession record differs")

    correction = _json(paths["proposal_correction_record"])
    old = correction.get("old_proposal", {})
    lineage = correction.get("lineage", {})
    if (
        correction.get("status") != "RECORDED_PRE_ACTIVATION"
        or correction.get("correction_reason")
        != "OPERATOR_CONFIRMED_WRONG_SCENE_LOCATION"
        or correction.get("correction_before_any_formal_icp") is not True
        or int(correction.get("correction_at_formal_trial_count", -1)) != 0
        or correction.get("registration_evidence_used") is not False
        or old.get("json_sha256") != ORIGINAL_PROPOSAL_JSON_SHA256
        or old.get("markdown_sha256") != ORIGINAL_PROPOSAL_MD_SHA256
        or old.get("supersession_record_sha256")
        != _sha256(paths["proposal_superseded_sidecar"])
        or lineage.get("scene_id") != "FMB1_W02"
        or lineage.get("attempt1_status") != "INVALID_ACQUISITION"
        or lineage.get("attempt2_status") != "GEOMETRY_ADMITTED"
        or lineage.get("attempt2_final_geometry_class") != "WEAK"
        or lineage.get("attempt2_in_final_dataset") is not True
        or lineage.get("w04_identifier_retired") is not True
        or lineage.get("w04_in_final_dataset") is not False
        or lineage.get("source_sha256")
        != _sha256(paths["acquisition_attempt_lineage"])
        or correction.get("FORMAL_LOCK_ISSUED") is not False
        or correction.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or int(correction.get("actual_formal_trials", -1)) != 0
    ):
        _fail("proposal correction record/lineage differs")

    difference = _json(paths["proposal_difference_report"])
    reviewed = {
        item.get("path"): item.get("sha256")
        for item in difference.get("reviewed_inputs", [])
        if isinstance(item, Mapping)
    }
    if (
        difference.get("status")
        != "DIFFERENCES_DOCUMENTED_VERSIONED_CORRECTION_REQUIRED"
        or difference.get("old_proposal_preserved_byte_for_byte") is not True
        or difference.get("old_proposal_json_sha256_before_r1")
        != ORIGINAL_PROPOSAL_JSON_SHA256
        or difference.get("old_proposal_md_sha256_before_r1")
        != ORIGINAL_PROPOSAL_MD_SHA256
        or reviewed.get(DEFAULT_BINDING_PATHS["original_zero_perturbation_proposal_json"])
        != ORIGINAL_PROPOSAL_JSON_SHA256
        or reviewed.get(DEFAULT_BINDING_PATHS["original_zero_perturbation_proposal_md"])
        != ORIGINAL_PROPOSAL_MD_SHA256
        or reviewed.get(DEFAULT_BINDING_PATHS["acquisition_attempt_lineage"])
        != _sha256(paths["acquisition_attempt_lineage"])
        or difference.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or int(difference.get("actual_formal_trials", -1)) != 0
    ):
        _fail("proposal difference report/history bindings differ")


def _validate_missingness_clarification(paths: Mapping[str, Path]) -> None:
    clarification_path = paths["analysis_missingness_clarification"]
    clarification = _json(clarification_path)
    rule = clarification.get("reassociation_missingness_rule", {})
    expected_reasons = [
        "NO_INITIAL_CORRESPONDENCE", "NO_FINAL_CORRESPONDENCE",
        "INSUFFICIENT_VALID_NORMALS", "NONFINITE_COMMON_METRICS", "OTHER",
    ]
    if (
        clarification.get("status") != "ACTIVE_PRELOCK_CLARIFICATION"
        or clarification.get("clarification_before_formal_lock") is not True
        or clarification.get("clarification_before_any_formal_icp") is not True
        or int(clarification.get("clarification_at_formal_trial_count", -1)) != 0
        or clarification.get("registration_result_used") is not False
        or rule.get("common_invalid_reason_retained") is not True
        or rule.get("required_result_status_fields") != [
            "common_association_valid", "common_association_invalid_reason",
            "common_association_invalid_detail",
        ]
        or rule.get("common_association_invalid_reason_enum") != expected_reasons
        or clarification.get("FORMAL_LOCK_ISSUED") is not False
        or clarification.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or int(clarification.get("actual_formal_trials", -1)) != 0
    ):
        _fail("R1-C1 missingness clarification semantics differ")
    contract = _json(paths["analysis_contract"])
    contract_binding = contract.get("prelock_missingness_clarification", {})
    if (
        contract_binding.get("path")
        != DEFAULT_BINDING_PATHS["analysis_missingness_clarification"]
        or contract_binding.get("sha256") != _sha256(clarification_path)
        or contract_binding.get("clarification_before_formal_lock") is not True
        or int(contract_binding.get("clarification_at_formal_trial_count", -1)) != 0
    ):
        _fail("analysis contract does not bind active R1-C1 clarification")
    inventory_path = paths["analysis_preclarification_history_inventory"]
    inventory = _json(inventory_path)
    if (
        inventory.get("status")
        != "ACTIVE_BYTES_PRESERVED_BEFORE_PRELOCK_CLARIFICATION"
        or int(inventory.get("formal_trial_count_at_archive", -1)) != 0
        or inventory.get("formal_lock_issued_at_archive") is not False
        or inventory.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or int(inventory.get("actual_formal_trials", -1)) != 0
    ):
        _fail("preclarification history inventory state differs")
    archive = inventory_path.parent
    rows = inventory.get("files")
    if not isinstance(rows, list) or len(rows) != 7:
        _fail("preclarification history inventory count differs")
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("preclarification history inventory row malformed")
        name, digest = row.get("name"), row.get("sha256")
        if not isinstance(name, str) or Path(name).name != name or not SHA_RE.fullmatch(str(digest)):
            _fail("preclarification history inventory path/SHA malformed")
        archived = archive / name
        if archived.is_symlink() or not archived.is_file() or _sha256(archived) != digest:
            _fail(f"preclarification archived byte/hash differs: {name}")
    transition = _json(paths["analysis_missingness_clarification_transition"])
    before, after = transition.get("before", {}), transition.get("after", {})
    expected_after = {
        "amendment_json_sha256": _sha256(paths["active_amendment"]),
        "amendment_md_sha256": _sha256(paths["active_amendment_md"]),
        "analysis_contract_sha256": _sha256(paths["analysis_contract"]),
        "analysis_protocol_sha256": _sha256(paths["analysis_protocol"]),
        "activation_record_sha256": _sha256(paths["amendment_activation_record"]),
        "active_protocol_pointer_sha256": _sha256(paths["active_protocol_pointer"]),
        "clarification_json_sha256": _sha256(clarification_path),
        "clarification_markdown_sha256": _sha256(
            paths["analysis_missingness_clarification_md"]
        ),
    }
    if (
        transition.get("status") != "COMPLETED_PRELOCK"
        or transition.get("clarification_before_formal_lock") is not True
        or transition.get("clarification_before_any_formal_icp") is not True
        or int(transition.get("clarification_at_formal_trial_count", -1)) != 0
        or transition.get("FORMAL_LOCK_ISSUED") is not False
        or transition.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or int(transition.get("actual_formal_trials", -1)) != 0
        or before.get("archive_inventory_path")
        != DEFAULT_BINDING_PATHS["analysis_preclarification_history_inventory"]
        or before.get("archive_inventory_sha256") != _sha256(inventory_path)
        or any(after.get(key) != value for key, value in expected_after.items())
    ):
        _fail("R1-C1 clarification transition/hash chain differs")


def _validate_activation(root: Path, paths: Mapping[str, Path]) -> None:
    amendment = _json(paths["active_amendment"])
    activation = _json(paths["amendment_activation_record"])
    pointer = _json(paths["active_protocol_pointer"])
    if amendment.get("amendment_id") != AMENDMENT_ID or amendment.get("status") != "ACTIVE":
        _fail("R1 amendment is not ACTIVE")
    if amendment.get("activation_effective") is not True:
        _fail("R1 amendment activation is not effective")
    if activation.get("amendment_id") != AMENDMENT_ID:
        _fail("activation record amendment ID differs")
    if activation.get("status") != "ACTIVE":
        _fail("activation record is not ACTIVE")
    if activation.get("ACTIVATED_BEFORE_ANY_FORMAL_ICP", activation.get("activated_before_any_formal_icp")) is not True:
        _fail("amendment was not activated before formal ICP")
    if int(activation.get("FORMAL_TRIAL_COUNT_AT_ACTIVATION", activation.get("formal_trial_count_at_activation", -1))) != 0:
        _fail("formal trials existed at amendment activation")
    if pointer.get("active_amendment_id", pointer.get("amendment_id")) != AMENDMENT_ID:
        _fail("ACTIVE_PROTOCOL does not point to R1")
    if pointer.get("status") != "ACTIVE":
        _fail("ACTIVE_PROTOCOL is not active")


def _validate_plan(path: Path) -> None:
    plan = _json(path)
    if plan.get("schema") != PLAN_SCHEMA:
        _fail("trial-plan schema differs")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 360:
        _fail("trial plan must contain 360 rows")
    counts = plan.get("counts")
    if not isinstance(counts, Mapping):
        _fail("trial-plan counts are missing")
    expected = {
        "scene_count": 6,
        "station_count": 18,
        "snapshot_count": 180,
        "open3d_trial_count": 180,
        "pcl_trial_count": 180,
        "total_trial_count": 360,
    }
    for key, value in expected.items():
        if int(counts.get(key, -1)) != value:
            _fail(f"trial-plan count differs: {key}")


def _validate_nested_reauthentication(root: Path, wrapper_path: Path) -> None:
    wrapper = _json(wrapper_path)
    if (wrapper.get("schema") != "mid360_fmb1_zero_perturbation_v1_1_r1_prelock_dataset_binding"
            or wrapper.get("status") != "PASS" or wrapper.get("pass") is not True):
        _fail("prelock dataset wrapper is not PASS")
    for prefix in ("source_report", "source_markdown"):
        source = _resolve_binding(root, str(wrapper.get(f"{prefix}_path")), prefix)
        if _sha256(source) != wrapper.get(f"{prefix}_sha256"):
            _fail(f"nested prelock {prefix} SHA differs")
    detail = _json(_resolve_binding(root, str(wrapper["source_report_path"]), "detailed reauthentication"))
    if (detail.get("status") != "PASS" or detail.get("pass") is not True
            or detail.get("final_scene_count") != 6
            or detail.get("final_station_count") != 18
            or detail.get("final_target_count") != 18
            or detail.get("final_snapshot_count") != 180):
        _fail("detailed final-dataset reauthentication differs")
    state = detail.get("formal_execution_state")
    if not isinstance(state, Mapping):
        _fail("detailed reauthentication lacks formal execution state")
    _assert_zero_no_icp(state)
    if detail.get("protected_assets", {}).get("unchanged") is not True:
        _fail("detailed reauthentication does not protect frozen assets")


def _validate_upstream_independent_reports(paths: Mapping[str, Path]) -> None:
    plan_report = _json(paths["trial_plan_independent_verification"])
    expected = {
        "status": "PASS", "payload_byte_hashes_verified": True,
        "single_authoritative_byte_source": True, "scene_count": 6,
        "station_count": 18, "snapshot_count": 180, "total_trial_count": 360,
        "open3d_trial_count": 180, "pcl_trial_count": 180,
        "identity_t0_count": 360, "w04_trial_count": 0,
        "old_w02_attempt1_trial_count": 0, "w02_attempt2_trial_count": 60,
        "registration_backend_call_count": 0,
        "FORMAL_REGISTRATION_AUTHORIZED": False, "actual_formal_trials": 0,
    }
    if any(plan_report.get(key) != value for key, value in expected.items()):
        _fail("trial-plan independent verification differs")
    if (plan_report.get("trial_plan_json_sha256") != _sha256(paths["trial_plan_json"])
            or plan_report.get("trial_plan_csv_sha256") != _sha256(paths["trial_plan_csv"])
            or plan_report.get("result_schema_sha256") != _sha256(paths["result_schema"])):
        _fail("trial-plan independent report hashes differ")
    old_report = _json(paths["protocol_transition_independent_verification"])
    inventory = _json(paths["analysis_preclarification_history_inventory"])
    archived = {row["name"]: row["sha256"] for row in inventory["files"]}
    old_hashes = {
        "amendment_json_sha256": archived["zero_perturbation_mainline_v1_1_r1.json"],
        "amendment_md_sha256": archived["zero_perturbation_mainline_v1_1_r1.md"],
        "analysis_contract_sha256": archived["zero_perturbation_analysis_contract_v1_1_r1.json"],
        "analysis_protocol_sha256": archived["zero_perturbation_analysis_protocol_v1_1_r1.md"],
    }
    if (old_report.get("pass") is not True
            or old_report.get("verification_status") != "PASS"
            or old_report.get("activation_effective") is not True
            or old_report.get("FORMAL_LOCK_ISSUED") is not False
            or old_report.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
            or old_report.get("actual_formal_trials") != 0
            or old_report.get("backend_calls") != 0
            or old_report.get("backend_modules_imported") != 0
            or old_report.get("active_hashes") != old_hashes
            or old_report.get("activation_record_sha256")
            != archived["amendment_activation_record_v1_1_r1.json"]
            or old_report.get("active_protocol_pointer_sha256")
            != archived["ACTIVE_PROTOCOL.json"]):
        _fail("historical initial-active transition report differs")
    c1 = _json(paths["protocol_c1_missingness_independent_verification"])
    current_hashes = {
        "amendment_json_sha256": _sha256(paths["active_amendment"]),
        "amendment_md_sha256": _sha256(paths["active_amendment_md"]),
        "analysis_contract_sha256": _sha256(paths["analysis_contract"]),
        "analysis_protocol_sha256": _sha256(paths["analysis_protocol"]),
    }
    expected_status_artifacts = {
        "execution_verifier_sha256": _sha256(paths["execution_environment"].parent / "zero_perturbation_v1_1_r1_verify.py"),
        "result_schema_sha256": _sha256(paths["result_schema"]),
        "runner_sha256": _sha256(paths["execution_runner"]),
        "result_validator_sha256": _sha256(paths["execution_result_validator"]),
        "experiments_package_init_sha256": _sha256(paths["execution_experiments_package_init"]),
        "mid360_formal_batch1_package_init_sha256": _sha256(paths["execution_mid360_formal_batch1_package_init"]),
        "phase_a_harness_package_init_sha256": _sha256(paths["execution_phase_a_harness_package_init"]),
    }
    if (c1.get("pass") is not True or c1.get("verification_status") != "PASS"
            or c1.get("phase") != "ACTIVE_R1_C1_PRE_LOCK"
            or c1.get("activation_effective") is not True
            or c1.get("common_association_status_fields_verified") is not True
            or c1.get("FORMAL_LOCK_ISSUED") is not False
            or c1.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
            or c1.get("actual_formal_trials") != 0
            or c1.get("backend_calls") != 0
            or c1.get("backend_modules_imported") != 0
            or c1.get("current_active_hashes") != current_hashes
            or c1.get("initial_active_hashes_preserved") != old_hashes
            or c1.get("activation_record_sha256") != _sha256(paths["amendment_activation_record"])
            or c1.get("active_protocol_pointer_sha256") != _sha256(paths["active_protocol_pointer"])
            or c1.get("clarification_json_sha256") != _sha256(paths["analysis_missingness_clarification"])
            or c1.get("clarification_markdown_sha256") != _sha256(paths["analysis_missingness_clarification_md"])
            or c1.get("clarification_transition_sha256") != _sha256(paths["analysis_missingness_clarification_transition"])
            or c1.get("prior_active_inventory_sha256") != _sha256(paths["analysis_preclarification_history_inventory"])
            or c1.get("common_association_status_artifacts") != expected_status_artifacts):
        _fail("R1-C1 independent transition/status report differs")

    review = _json(paths["activation_review"])
    review_verifier = review.get("missingness_clarification_independent_verification")
    if not isinstance(review_verifier, Mapping):
        _fail("activation review lacks C1 verifier binding")
    if (review.get("schema") != "mid360_fmb1_zero_perturbation_v1_1_r1_activation_review"
            or review.get("review_status") != "PASS_ACTIVE_R1_C1"
            or review.get("activation_effective") is not True
            or review.get("missingness_clarification_verifier_pass") is not True
            or review.get("FORMAL_LOCK_ISSUED") is not False
            or review.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
            or review.get("actual_formal_trials") != 0
            or review_verifier.get("status") != "PASS"
            or review_verifier.get("sha256")
            != _sha256(paths["protocol_c1_missingness_independent_verification"])):
        _fail("activation review does not bind the zero-trial R1-C1 PASS")
    try:
        review_time = datetime.fromisoformat(str(review.get("reviewed_at_utc")).replace("Z", "+00:00"))
        transition_time = datetime.fromisoformat(
            str(_json(paths["analysis_missingness_clarification_transition"])["transitioned_at_utc"])
            .replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail(f"activation review/transition timestamp is invalid: {error}")
    if (review_time.tzinfo is None or transition_time.tzinfo is None
            or review_time < transition_time):
        _fail("activation review predates the R1-C1 transition")


def _validate_execution_commit(root: Path, commit: str, paths: Mapping[str, Path]) -> None:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        if head != commit:
            _fail("execution-code commit must be current HEAD when lock is built")
        for name in EXECUTION_BINDING_NAMES:
            relative = paths[name].relative_to(root).as_posix()
            recorded = subprocess.run(
                ["git", "show", f"{commit}:{relative}"], cwd=root, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout
            if hashlib.sha256(recorded).hexdigest() != _sha256(paths[name]):
                _fail(f"execution asset differs from execution-code commit: {name}")
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", "replace")[-500:] if isinstance(error.stderr, bytes) else str(error.stderr)[-500:]
        _fail(f"execution-code commit is not authentic/complete: {detail}")


def build_lock_payload(
    repository: Path,
    *,
    execution_code_commit: str,
    binding_paths: Mapping[str, str] | None = None,
    issued_at_utc: str | None = None,
    remeasure_environment_versions: bool = True,
    verify_execution_commit: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build, but do not write, a closed-authorization formal lock."""

    root = Path(repository).resolve(strict=True)
    if COMMIT_RE.fullmatch(execution_code_commit) is None:
        _fail("execution-code commit must be a full lowercase Git SHA")
    _assert_no_execution_lifecycle(root)
    configured = dict(DEFAULT_BINDING_PATHS)
    if binding_paths:
        unknown = set(binding_paths) - set(configured)
        if unknown:
            _fail(f"unknown binding names: {sorted(unknown)}")
        if verify_execution_commit and any(
            str(value) != DEFAULT_BINDING_PATHS[key]
            for key, value in binding_paths.items()
        ):
            _fail("formal lock issuance cannot redirect canonical binding paths")
        configured.update({key: str(value) for key, value in binding_paths.items()})
    paths = {
        key: _resolve_binding(root, value, key) for key, value in configured.items()
    }
    if verify_execution_commit:
        _validate_execution_commit(root, execution_code_commit, paths)
    _validate_proposal_correction_history(paths)
    _validate_missingness_clarification(paths)
    _validate_activation(root, paths)
    _validate_plan(paths["trial_plan_json"])
    _validate_nested_reauthentication(root, paths["final_dataset_prelock_reauthentication"])
    _validate_upstream_independent_reports(paths)
    environment = _json(paths["environment_manifest"])
    verify_environment_manifest(
        environment, root, remeasure_versions=remeasure_environment_versions
    )
    prelock = _json(paths["prelock_no_icp_attestation"])
    _assert_zero_no_icp(prelock)
    if _sha256(paths["backend_parameter_contract"]) != BACKEND_CONTRACT_SHA256:
        _fail("backend parameter contract changed")
    if environment.get("pcl_cli", {}).get("sha256") != _sha256(paths["pcl_executable"]):
        _fail("environment/PCL executable binding differs")
    inventory: list[dict[str, Any]] = []
    for key in sorted(paths):
        path = paths[key]
        inventory.append(
            {
                "binding_name": key,
                "repository_relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    inventory_sha = hashlib.sha256(_canonical(inventory)).hexdigest()
    lock = {
        "schema": LOCK_SCHEMA,
        "lock_id": "FMB1_ZERO_PERTURBATION_V1_1_R1_FORMAL_EXECUTION_LOCK",
        "amendment_id": AMENDMENT_ID,
        "track_id": "ZERO_PERTURBATION_TRACK",
        "status": "ISSUED_AWAITING_SEPARATE_AUTHORIZATION",
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "execution_code_commit": execution_code_commit,
        "authoritative_runtime_root": AUTHORITATIVE_RUNTIME_ROOT,
        "binding_inventory_sha256": inventory_sha,
        "bindings": {row["binding_name"]: dict(row) for row in inventory},
        "counts": {
            "scene_count": 6,
            "station_count": 18,
            "snapshot_count": 180,
            "planned_open3d_trials": 180,
            "planned_pcl_trials": 180,
            "planned_total_trials": 360,
        },
        "FORMAL_LOCK_ISSUED": True,
        "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION": True,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "MEASUREMENT_FINAL_RESULT": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "registration_execution_count": 0,
    }
    return lock, inventory


def _inventory_csv(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=("binding_name", "repository_relative_path", "sha256", "bytes"),
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _build_r1_no_icp_attestation(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute an R1-scoped, registration-free pre-execution attestation."""

    _assert_no_execution_lifecycle(root)

    authority_files = (
        "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_runner.py",
        "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_lock.py",
        "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_verify.py",
        "experiments/__init__.py",
        "experiments/mid360_formal_batch1/__init__.py",
        "src/phase_a_harness/__init__.py",
    )
    package_init_files = {
        "experiments/__init__.py",
        "experiments/mid360_formal_batch1/__init__.py",
        "src/phase_a_harness/__init__.py",
    }
    hashes: dict[str, str] = {}
    runner_functions: set[str] = set()
    forbidden_imports = ("open3d", "open3d_backend", "pcl_backend", "debug_registration")
    for relative in authority_files:
        path = _resolve_binding(root, relative, f"R1 authority {relative}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if relative in package_init_files and any(
            isinstance(node, (ast.Import, ast.ImportFrom, ast.Call))
            for node in ast.walk(tree)
        ):
            _fail(f"execution package initializer has eager import/call: {relative}")
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            if any(token in name for name in names for token in forbidden_imports):
                _fail(f"R1 authority has eager backend import: {relative}")
            if relative.endswith("_runner.py") and isinstance(node, ast.FunctionDef):
                runner_functions.add(node.name)
        hashes[relative] = _sha256(path)
    if not {"_load_authorized_execution_adapter", "_execute_authorized_run",
            "preflight_or_dry_run"}.issubset(runner_functions):
        _fail("R1 runner future-execution boundary is incomplete")
    live_processes = 0
    proc = Path("/proc")
    if proc.is_dir():
        for directory in proc.iterdir():
            if not directory.name.isdigit():
                continue
            try:
                argv = [part for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
            except (OSError, PermissionError):
                continue
            if argv and Path(os.fsdecode(argv[0])).name == "pcl_point_to_plane_cli":
                live_processes += 1
    if live_processes:
        _fail("a PCL registration process is already running")
    prelock = lock["bindings"]["prelock_no_icp_attestation"]
    return {
        "schema": "mid360_fmb1_zero_perturbation_r1_no_icp_attestation_v1",
        "status": "PASS", "pass": True,
        "scope": "R1_LOCK_AND_FUTURE_EXECUTION_BOUNDARY_PRE_AUTHORIZATION",
        "authority_file_sha256": hashes,
        "future_execution_boundary_verified": True,
        "authoritative_runtime_root": AUTHORITATIVE_RUNTIME_ROOT,
        "execution_lifecycle_file_count": 0,
        "real_trial_result_file_count": 0,
        "live_registration_process_count": 0,
        "prelock_no_icp_attestation_path": prelock["repository_relative_path"],
        "prelock_no_icp_attestation_sha256": prelock["sha256"],
        "open3d_registration_call_count": 0,
        "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0,
        "formal_trial_count": 0,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "registration_execution_count": 0,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "MEASUREMENT_FINAL_RESULT": False,
    }


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            _fail(f"refusing to overwrite different lock artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        _fail(f"stale temporary lock artifact: {temporary}")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_lock_bundle(
    repository: Path,
    output_dir: Path,
    lock: Mapping[str, Any],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write an issued-but-unauthorized lock bundle atomically per file."""

    root = Path(repository).resolve(strict=True)
    output = Path(output_dir).resolve()
    lock_bytes = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode("utf-8")
    inventory_bytes = _inventory_csv(inventory)
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    inventory_file_sha = hashlib.sha256(inventory_bytes).hexdigest()
    fingerprint_material = {
        "lock_file_sha256": lock_sha,
        "lock_inventory_file_sha256": inventory_file_sha,
        "execution_code_commit": lock.get("execution_code_commit"),
    }
    fingerprint = hashlib.sha256(_canonical(fingerprint_material)).hexdigest()
    fingerprint_payload = {
        "schema": "mid360_fmb1_zero_perturbation_lock_fingerprint_v1_1_r1",
        **fingerprint_material,
        "lock_fingerprint": fingerprint,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    }
    no_icp_bytes = (
        json.dumps(_build_r1_no_icp_attestation(root, lock), indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    files = {
        LOCK_FILENAME: lock_bytes,
        "formal_batch1_zero_perturbation_lock_v1_1.sha256": (
            f"{lock_sha}  {LOCK_FILENAME}\n".encode("ascii")
        ),
        "lock_inventory.csv": inventory_bytes,
        "lock_fingerprint.json": (
            json.dumps(fingerprint_payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "NO_ICP_ATTESTATION.json": no_icp_bytes,
    }
    for name, content in files.items():
        _write_once(output / name, content)
    finalize_lock_core_checksums(output)
    return {
        "lock_file_sha256": lock_sha,
        "lock_fingerprint": fingerprint,
        "artifact_count": len(files) + 1,
        "SHA256SUMS_FINALIZED": False,
        "FORMAL_LOCK_ISSUED": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    }


def finalize_lock_core_checksums(output_dir: Path) -> Path:
    """Freeze only immutable lock inputs/core; exclude verifier outputs."""

    output = Path(output_dir).resolve(strict=True)
    missing = [name for name in LOCK_CORE_FILENAMES if not (output / name).is_file()]
    if missing:
        _fail(f"lock core is incomplete: {missing}")
    content = "".join(
        f"{_sha256(output / name)}  {name}\n" for name in LOCK_CORE_FILENAMES
    ).encode("ascii")
    _write_once(output / "LOCK_CORE_SHA256SUMS", content)
    return output / "LOCK_CORE_SHA256SUMS"


def finalize_lock_directory_checksums(output_dir: Path) -> Path:
    """Write the non-self-referential release SHA256SUMS after all reports."""

    output = Path(output_dir).resolve(strict=True)
    files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS" and not path.name.endswith(".tmp")
    )
    if not files:
        _fail("cannot finalize checksums for an empty lock directory")
    content = "".join(f"{_sha256(path)}  {path.name}\n" for path in files).encode("ascii")
    path = output / "SHA256SUMS"
    temporary = output / "SHA256SUMS.tmp"
    if temporary.exists():
        _fail("stale SHA256SUMS temporary exists")
    temporary.write_bytes(content)
    temporary.replace(path)
    return path
