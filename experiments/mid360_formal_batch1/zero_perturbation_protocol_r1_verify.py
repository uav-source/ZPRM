"""Independent, backend-free verifier for the FMB1 v1.1-R1 protocol candidate.

This verifier does not import a registration backend and does not reuse a
producer decision function.  It authenticates the retained proposal, derives
the active dataset structure from frozen manifests, and independently checks
the R1 amendment and analysis contract before activation can be considered.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


OLD_PROPOSAL_JSON_SHA256 = (
    "4837f6bd37e3a0f19c4b276964bbd066df09b86e58af288aa103aca704be936e"
)
OLD_PROPOSAL_MD_SHA256 = (
    "836822c553929a2e5b9e6cdf85b48d318fe71f91aa47447d908d030516e7f4ed"
)
ORIGINAL_PREREGISTRATION_SHA256 = (
    "76ae548874d8c1584fcc033685db4a1e7cf104fd881cbd1c334eba0bfe1a9beb"
)
ORIGINAL_ANALYSIS_PROTOCOL_SHA256 = (
    "d453d12e713c546c5a054ceb1710b86eea12fa09e3f244705e46df9db255a879"
)
BACKEND_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
COMMON_ASSOCIATION_SHA256 = (
    "458190c26d8640353004663474adb64305d1193d89ff0dc68f71f404de8f175d"
)
PHYSICAL_REFERENCE = "NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT"
AMENDMENT_ID = "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1"
CANDIDATE_STATUS = "CANDIDATE_PENDING_INDEPENDENT_ACTIVATION_VERIFIER"
ACTIVE_STATUS = "ACTIVE"
CLARIFICATION_ID = "FMB1_ZERO_PERTURBATION_MISSINGNESS_CLARIFICATION_V1_1_R1_C1"
EXPECTED_CANDIDATE_HASHES = {
    "amendment_json_sha256": "fc06bfcc44ded84df1e226d8203859201ef3b78e3e416c8e6eef2360e6e40eef",
    "amendment_md_sha256": "68b89240e4d94b88d1ee3f133c68b17623db9a6568118b8d174bc741d45b0f73",
    "analysis_contract_sha256": "e507ae54c5a9c997b40734006ac1643a337ffdfafd65c031ecb0251e3c7f0c3d",
    "analysis_protocol_sha256": "c1a4fbfcc93c64bd0d75d1f8bdf77bbfcbd086e5ed42d70b0b9a6f5506b47a59",
}
EXPECTED_INITIAL_ACTIVE_HASHES = {
    "amendment_json_sha256": "1ff006ce3226c849c6ad70a12ca538bddf0b69c932c3c84e59f540fe42e7c1ab",
    "amendment_md_sha256": "b506f2a37d4624132d5ff3c05b6611e5e4f199f9f22c902d6f38b496066c607e",
    "analysis_contract_sha256": "6f763eed64a438c58b2549c7d3741a67975af1560264b7c47158d92f8c33530a",
    "analysis_protocol_sha256": "b1f3b7ef54ae75e86e4131f171e8a2b74a3472c675e9c1c8210e918a2f056fd4",
}
EXPECTED_INITIAL_ACTIVATION_RECORD_SHA256 = (
    "cf6c7dde69e1c5ff0742c8f7dcb6189edc359686af366fcb1d96b3c904b5c097"
)
EXPECTED_INITIAL_ACTIVE_POINTER_SHA256 = (
    "f4933ca1048f873e2465b617184df8bbe5cfb3b11199fca97316aedab9bf12c2"
)
EXPECTED_INITIAL_ACTIVATION_TRANSITION_SHA256 = (
    "9beeb1d501369d25ca89f510641614c8e7641d27cbdaddb39ccb3af558a094d5"
)
EXPECTED_SCENES = {
    "FMB1_R01": "RICH",
    "FMB1_R02": "RICH",
    "FMB1_R03": "RICH",
    "FMB1_W01": "WEAK",
    "FMB1_W02": "WEAK",
    "FMB1_W03": "WEAK",
}
IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class ProtocolR1VerificationError(ValueError):
    """Raised when an R1 protocol candidate violates a frozen requirement."""


def _fail(message: str) -> None:
    raise ProtocolR1VerificationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be a plain JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProtocolR1VerificationError(f"non-finite JSON token {token} in {path}")
            ),
        )
    except ProtocolR1VerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolR1VerificationError(f"cannot load JSON {path}: {error}") from error
    return dict(_require_mapping(value, str(path)))


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ProtocolR1VerificationError(f"cannot load CSV {path}: {error}") from error


def validate_candidate_payloads(
    amendment_value: Mapping[str, Any],
    contract_value: Mapping[str, Any],
    supersession_value: Mapping[str, Any],
) -> None:
    """Validate detached payloads, enabling mutation-based negative tests."""

    amendment = _require_mapping(amendment_value, "amendment")
    contract = _require_mapping(contract_value, "analysis contract")
    supersession = _require_mapping(supersession_value, "supersession record")

    _require(
        amendment.get("schema")
        == "mid360_fmb1_zero_perturbation_mainline_amendment_v1_1_r1",
        "amendment schema mismatch",
    )
    _require(amendment.get("amendment_id") == AMENDMENT_ID, "amendment ID mismatch")
    _require(amendment.get("status") == CANDIDATE_STATUS, "candidate status mismatch")
    _require(amendment.get("activation_effective") is False, "candidate self-activates")
    for flag in ("FORMAL_AUTHORITY", "FORMAL_LOCK_ISSUED", "FORMAL_ICP_UNLOCKED", "FORMAL_REGISTRATION_AUTHORIZED", "backend_execution_permitted"):
        _require(amendment.get(flag) is False, f"candidate {flag} must be false")
    for count in ("actual_open3d_trials", "actual_pcl_trials", "actual_formal_trials"):
        _require(amendment.get(count) == 0, f"candidate {count} must be zero")

    correction = _require_mapping(amendment.get("correction"), "amendment.correction")
    _require(
        correction.get("correction_reason") == "OPERATOR_CONFIRMED_WRONG_SCENE_LOCATION",
        "correction reason mismatch",
    )
    _require(correction.get("correction_at_formal_trial_count") == 0, "correction is post-result")
    _require(correction.get("correction_before_any_formal_icp") is True, "correction timing mismatch")
    _require(correction.get("registration_evidence_used_for_correction") is False, "registration evidence used")
    _require(correction.get("original_proposal_modified") is False, "old proposal marked modified")
    _require(correction.get("original_proposal_json_sha256") == OLD_PROPOSAL_JSON_SHA256, "old proposal JSON binding mismatch")
    _require(correction.get("original_proposal_md_sha256") == OLD_PROPOSAL_MD_SHA256, "old proposal Markdown binding mismatch")
    attempt1 = _require_mapping(correction.get("w02_attempt1"), "w02 attempt 1")
    _require(attempt1.get("status") == "INVALID_ACQUISITION", "W02 attempt 1 status mismatch")
    _require("WRONG_SCENE_LOCATION" in str(attempt1.get("reason")), "W02 attempt 1 reason mismatch")
    _require(attempt1.get("retained_in_invalid_archive") is True, "W02 attempt 1 is not retained")
    _require(attempt1.get("included_in_final_dataset") is False, "W02 attempt 1 entered final data")
    _require(attempt1.get("included_in_trial_plan") is False, "W02 attempt 1 entered plan")
    attempt2 = _require_mapping(correction.get("w02_attempt2"), "w02 attempt 2")
    _require(attempt2.get("status") == "GEOMETRY_ADMITTED", "W02 attempt 2 status mismatch")
    _require(attempt2.get("final_geometry_class") == "WEAK", "W02 attempt 2 is not Weak")
    _require(attempt2.get("station_count") == 3, "W02 attempt 2 station count mismatch")
    _require(attempt2.get("snapshot_count") == 30, "W02 attempt 2 snapshot count mismatch")
    _require(correction.get("w04_identifier_retired") is True, "W04 is not retired")
    _require(correction.get("w04_included_in_final_dataset") is False, "W04 entered final data")

    preserved = _require_mapping(amendment.get("authority_preservation"), "authority preservation")
    _require(preserved.get("original_preregistration_sha256") == ORIGINAL_PREREGISTRATION_SHA256, "original preregistration binding mismatch")
    _require(preserved.get("original_capture_radius_analysis_protocol_sha256") == ORIGINAL_ANALYSIS_PROTOCOL_SHA256, "original analysis binding mismatch")
    _require(preserved.get("original_files_modified_by_r1") is False, "R1 marks original protocol modified")
    _require(preserved.get("capture_radius_track_status") == "PRESERVED_SUPPLEMENTARY_NOT_EXECUTED", "capture-radius status mismatch")
    _require(preserved.get("capture_radius_execution_authorized") is False, "capture-radius execution authorized")

    dataset = _require_mapping(amendment.get("final_dataset_binding"), "final dataset binding")
    _require(dataset.get("scene_ids") == list(EXPECTED_SCENES), "amendment scene order/set mismatch")
    _require(dataset.get("scene_count") == 6, "amendment scene count mismatch")
    _require(dataset.get("station_count") == 18, "amendment station count mismatch")
    _require(dataset.get("snapshot_count") == 180, "amendment snapshot count mismatch")
    _require(dataset.get("w04_forbidden") is True, "W04 not forbidden")
    _require(dataset.get("old_w02_attempt1_forbidden") is True, "old W02 attempt not forbidden")

    track = _require_mapping(amendment.get("zero_perturbation_track"), "zero track")
    _require(track.get("track_id") == "ZERO_PERTURBATION_TRACK", "track ID mismatch")
    _require(track.get("backends") == ["OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE"], "backend set mismatch")
    _require(track.get("planned_open3d_trials") == 180, "Open3D plan count mismatch")
    _require(track.get("planned_pcl_trials") == 180, "PCL plan count mismatch")
    _require(track.get("planned_total_trials") == 360, "total plan count mismatch")
    _require(track.get("native_trial_count") == 0, "Native plan count must be zero")
    _require(track.get("T0") == IDENTITY, "T0 is not Identity")
    _require(track.get("T_reference_nominal") == IDENTITY, "nominal reference is not Identity")
    _require(track.get("translation_perturbation_m") == 0.0, "translation perturbation is nonzero")
    _require(track.get("rotation_perturbation_deg") == 0.0, "rotation perturbation is nonzero")

    physical = _require_mapping(amendment.get("physical_reference"), "physical reference")
    _require(physical.get("physical_reference_semantics") == PHYSICAL_REFERENCE, "physical semantics mismatch")
    _require(physical.get("imu_evidence_scope") == "NO_OBVIOUS_MOTION_ONLY", "IMU scope overstated")
    _require(physical.get("independent_submillimeter_external_ground_truth_available") is False, "submillimeter ground truth falsely claimed")
    _require(physical.get("imu_proves_physical_displacement_below_1mm") is False, "IMU submillimeter claim present")

    hierarchy = _require_mapping(amendment.get("experimental_hierarchy"), "hierarchy")
    _require(hierarchy.get("highest_independent_unit") == "scene", "scene is not highest unit")
    _require(hierarchy.get("snapshots_are_independent_scenes") is False, "snapshot pseudoreplication enabled")
    backend = _require_mapping(amendment.get("backend_contract"), "backend contract")
    _require(backend.get("sha256") == BACKEND_CONTRACT_SHA256, "backend contract SHA mismatch")
    _require(backend.get("changed_by_r1") is False, "backend contract marked changed")

    _require(contract.get("schema") == "mid360_fmb1_zero_perturbation_analysis_contract_v1_1_r1", "analysis contract schema mismatch")
    _require(contract.get("amendment_id") == AMENDMENT_ID, "analysis contract amendment mismatch")
    _require(contract.get("status") == CANDIDATE_STATUS, "analysis contract status mismatch")
    _require(contract.get("activation_effective") is False, "analysis contract self-activates")
    _require(contract.get("FORMAL_REGISTRATION_AUTHORIZED") is False, "analysis contract authorizes execution")
    _require(contract.get("actual_formal_trials") == 0, "analysis contract trial count nonzero")
    _require(contract.get("track_id") == "ZERO_PERTURBATION_TRACK", "analysis track mismatch")
    contract_dataset = _require_mapping(contract.get("dataset"), "contract dataset")
    _require(contract_dataset.get("rich_scene_ids") == ["FMB1_R01", "FMB1_R02", "FMB1_R03"], "Rich scenes mismatch")
    _require(contract_dataset.get("weak_scene_ids") == ["FMB1_W01", "FMB1_W02", "FMB1_W03"], "Weak scenes mismatch")
    _require(contract_dataset.get("forbidden_scene_ids") == ["FMB1_W04"], "W04 missing from forbidden scenes")
    _require(contract_dataset.get("w02_active_attempt") == 2, "W02 active attempt mismatch")
    contract_physical = _require_mapping(contract.get("physical_reference"), "contract physical reference")
    _require(contract_physical.get("physical_reference_semantics") == PHYSICAL_REFERENCE, "contract physical semantics mismatch")
    _require(
        contract_physical.get("independent_submillimeter_external_ground_truth_available")
        is False,
        "contract falsely claims submillimeter external ground truth",
    )
    _require(
        contract_physical.get("imu_submillimeter_ground_truth_claim_forbidden") is True,
        "contract permits IMU submillimeter ground-truth claim",
    )
    _require(
        contract_physical.get("absolute_physical_displacement_claim_forbidden") is True,
        "contract permits absolute physical-displacement claim",
    )
    contract_units = _require_mapping(contract.get("experimental_units"), "contract units")
    _require(contract_units.get("highest_independent_unit") == "scene", "contract highest unit mismatch")
    _require(contract_units.get("primary_experimental_unit") == "scene", "contract primary unit mismatch")
    _require(contract_units.get("snapshots_are_independent_scenes") is False, "contract snapshot pseudoreplication enabled")
    primary = _require_mapping(contract.get("primary_endpoint"), "primary endpoint")
    inference = _require_mapping(primary.get("inference"), "primary inference")
    _require(inference.get("method") == "EXACT_ONE_SIDED_PERMUTATION_ALL_3_VS_3_SCENE_ALLOCATIONS", "primary inference mismatch")
    _require(inference.get("allocation_count") == 20, "permutation allocation count mismatch")
    _require(inference.get("forced_pass_conclusion") is False, "primary conclusion is forced")
    reassociation = _require_mapping(contract.get("reassociation_analysis"), "reassociation analysis")
    _require(reassociation.get("implementation_path") == "src/phase_a_harness/common_association_analysis.py", "common association path mismatch")
    _require(reassociation.get("implementation_sha256_at_candidate_freeze") == COMMON_ASSOCIATION_SHA256, "common association SHA mismatch")
    _require(reassociation.get("mid360_specific_redefinition_forbidden") is True, "Mid-360 reassociation redefinition allowed")
    permutation = _require_mapping(reassociation.get("optional_stratified_permutation"), "reassociation permutation")
    _require(permutation.get("preserve_scene_strata") is True, "reassociation permutation breaks scenes")
    _require(permutation.get("seed") == 20260820, "reassociation seed mismatch")
    _require(permutation.get("permutation_count") == 10000, "reassociation permutation count mismatch")
    systematic = _require_mapping(contract.get("systematic_component"), "systematic component")
    _require(systematic.get("formula") == "norm(mean(delta_translation_vectors)) / mean(norm(delta_translation_vectors))", "systematic formula mismatch")
    zero_rule = _require_mapping(systematic.get("zero_denominator_rule"), "systematic zero rule")
    _require(zero_rule.get("systematic_fraction") == 0.0, "systematic zero rule undefined")
    _require(zero_rule.get("station_retained") is True, "zero-denominator station excluded")
    transfer = _require_mapping(contract.get("synthetic_model_transfer"), "model transfer")
    _require(transfer.get("status") == "MODEL_TRANSFER_NOT_COMPATIBLE", "model transfer not fail-closed")
    _require(transfer.get("model_a_or_b_inference_permitted") is False, "model transfer inference permitted")
    _require(transfer.get("retraining_permitted") is False, "model retraining permitted")
    _require(transfer.get("required_for_formal_lock") is False, "model transfer incorrectly gates lock")
    capture = _require_mapping(contract.get("capture_radius_track"), "capture radius track")
    _require(capture.get("status") == "PRESERVED_SUPPLEMENTARY_NOT_EXECUTED", "contract capture-radius status mismatch")
    _require(capture.get("changed_by_r1") is False, "contract changes capture-radius track")

    _require(supersession.get("status") == "SUPERSEDED_PROPOSAL", "supersession status mismatch")
    _require(supersession.get("disposition") == "SUPERSEDED_AS_ACTIVATION_CANDIDATE_RETAINED_UNCHANGED", "supersession disposition mismatch")
    _require(supersession.get("historical_proposal_deleted") is False, "old proposal marked deleted")
    _require(supersession.get("historical_proposal_modified") is False, "old proposal marked modified")
    old_json = _require_mapping(supersession.get("historical_proposal_json"), "supersession old JSON")
    old_md = _require_mapping(supersession.get("historical_proposal_markdown"), "supersession old Markdown")
    _require(old_json.get("sha256") == OLD_PROPOSAL_JSON_SHA256, "supersession old JSON SHA mismatch")
    _require(old_md.get("sha256") == OLD_PROPOSAL_MD_SHA256, "supersession old Markdown SHA mismatch")
    _require(supersession.get("correction_at_formal_trial_count") == 0, "supersession correction is post-result")
    _require(supersession.get("registration_evidence_used") is False, "supersession used registration evidence")
    replacement = _require_mapping(supersession.get("replacement_candidate"), "replacement candidate")
    _require(replacement.get("amendment_id") == AMENDMENT_ID, "supersession candidate mismatch")
    _require(replacement.get("activation_effective") is False, "supersession candidate self-activates")


def validate_active_payloads(
    amendment_value: Mapping[str, Any],
    contract_value: Mapping[str, Any],
    supersession_value: Mapping[str, Any],
) -> None:
    """Validate active protocol semantics without weakening candidate checks."""

    amendment = copy.deepcopy(dict(_require_mapping(amendment_value, "active amendment")))
    contract = copy.deepcopy(dict(_require_mapping(contract_value, "active contract")))
    supersession = _require_mapping(supersession_value, "supersession record")
    _require(amendment.get("status") == ACTIVE_STATUS, "active amendment status mismatch")
    _require(amendment.get("activation_effective") is True, "active amendment is ineffective")
    _require(amendment.get("activated_before_any_formal_icp") is True, "active amendment timing mismatch")
    _require(amendment.get("formal_trial_count_at_activation") == 0, "amendment activated after results")
    _require(
        amendment.get("activation_effective_only_after")
        == [
            "R1_PROTOCOL_CANDIDATE_INDEPENDENT_VERIFIER_PASS",
            "FINAL_DATASET_PRELOCK_REAUTHENTICATION_PASS",
            "ACTIVATION_RECORD_CREATED_AT_FORMAL_TRIAL_COUNT_ZERO",
            "ACTIVE_PROTOCOL_POINTER_CREATED_AND_HASH_BOUND",
        ],
        "active amendment activation gates mismatch",
    )
    _require(
        amendment.get("formal_lock_prerequisites_not_activation_prerequisites")
        == [
            "FORMAL_ENVIRONMENT_QUALIFICATION_PASS",
            "TRIAL_PLAN_AND_RESULT_SCHEMA_INDEPENDENT_VERIFIER_PASS",
        ],
        "formal-lock gates are not separated from activation",
    )
    active_contract = _require_mapping(amendment.get("analysis_contract"), "active analysis contract reference")
    _require(active_contract.get("status") == ACTIVE_STATUS, "amendment contract reference is not active")

    _require(contract.get("status") == ACTIVE_STATUS, "active analysis contract status mismatch")
    _require(contract.get("activation_effective") is True, "active analysis contract is ineffective")
    _require(contract.get("activated_before_any_formal_icp") is True, "analysis contract activation timing mismatch")
    _require(contract.get("formal_trial_count_at_activation") == 0, "analysis contract activated after results")

    amendment_clarification = _require_mapping(
        amendment.get("prelock_scientific_clarification"),
        "amendment prelock scientific clarification",
    )
    _require(
        amendment_clarification.get("clarification_id") == CLARIFICATION_ID,
        "amendment clarification ID mismatch",
    )
    _require(
        amendment_clarification.get("status") == "ACTIVE_PRELOCK_CLARIFICATION",
        "amendment clarification status mismatch",
    )
    _require(
        amendment_clarification.get("clarification_before_formal_lock") is True,
        "amendment clarification is not pre-lock",
    )
    _require(
        amendment_clarification.get("clarification_before_any_formal_icp") is True,
        "amendment clarification is not pre-ICP",
    )
    _require(
        amendment_clarification.get("clarification_at_formal_trial_count") == 0,
        "amendment clarification is post-result",
    )

    _require(contract.get("document_version") == "1.1-R1-C1", "analysis contract C1 version mismatch")
    _require(contract.get("clarification_id") == CLARIFICATION_ID, "analysis contract clarification ID mismatch")
    contract_clarification = _require_mapping(
        contract.get("prelock_missingness_clarification"),
        "analysis contract prelock clarification",
    )
    _require(
        contract_clarification.get("status") == "ACTIVE_PRELOCK_CLARIFICATION",
        "analysis contract clarification status mismatch",
    )
    _require(
        contract_clarification.get("clarification_before_formal_lock") is True,
        "analysis clarification is not pre-lock",
    )
    _require(
        contract_clarification.get("clarification_before_any_formal_icp") is True,
        "analysis clarification is not pre-ICP",
    )
    _require(
        contract_clarification.get("clarification_at_formal_trial_count") == 0,
        "analysis clarification is post-result",
    )

    missingness = _require_mapping(
        contract.get("missingness_and_nonfinite_policy"),
        "analysis missingness policy",
    )
    _require(
        missingness.get("all_planned_trial_ids_retained_in_accounting") is True,
        "missingness policy drops planned trials",
    )
    _require(missingness.get("silent_exclusion_forbidden") is True, "silent exclusion allowed")
    _require(
        missingness.get("nonfinite_json_representation")
        == "NULL_WITH_EXPLICIT_STATUS_NEVER_NAN_OR_INFINITY",
        "nonfinite JSON representation mismatch",
    )
    _require(
        missingness.get("scientific_nonfinite_retry_forbidden") is True,
        "scientific nonfinite retry allowed",
    )
    _require(
        missingness.get("authoritative_trial_outcome")
        == "EARLIEST_SCHEMA_VALID_INFRASTRUCTURE_PASS_SCIENTIFIC_OUTCOME",
        "authoritative attempt selection mismatch",
    )
    station_missing = _require_mapping(
        missingness.get("station_formal_summary"), "station missingness rule"
    )
    scene_missing = _require_mapping(
        missingness.get("scene_formal_summary"), "scene missingness rule"
    )
    _require(
        station_missing.get("planned_n") == 10
        and station_missing.get("required_finite_n") == 10
        and station_missing.get("required_scientific_undefined_n") == 0
        and station_missing.get("required_unresolved_infrastructure_n") == 0,
        "station complete-coverage gate mismatch",
    )
    _require(
        scene_missing.get("planned_n") == 30
        and scene_missing.get("required_finite_n") == 30
        and scene_missing.get("required_defined_station_summaries") == 3
        and scene_missing.get("required_scientific_undefined_n") == 0
        and scene_missing.get("required_unresolved_infrastructure_n") == 0,
        "scene complete-coverage gate mismatch",
    )
    available = _require_mapping(
        missingness.get("available_case_descriptive"), "available-case rule"
    )
    _require(
        available.get("label")
        == "AVAILABLE_CASE_DESCRIPTIVE_NOT_PRIMARY_NOT_INFERENTIAL"
        and available.get("may_replace_formal_summary") is False,
        "available-case summary can replace formal inference",
    )
    _require(missingness.get("imputation_forbidden") is True, "imputation allowed")
    _require(missingness.get("winsorization_forbidden") is True, "winsorization allowed")
    active_primary = _require_mapping(contract.get("primary_endpoint"), "active primary endpoint")
    active_inference = _require_mapping(active_primary.get("inference"), "active inference")
    _require(
        active_inference.get("required_defined_scene_summaries") == 6
        and active_inference.get("required_defined_rich_scene_summaries") == 3
        and active_inference.get("required_defined_weak_scene_summaries") == 3,
        "exact permutation completeness gate mismatch",
    )
    _require(
        active_inference.get("incomplete_behavior")
        == "NULL_ESTIMAND_AND_P_VALUE_WITH_INFERENCE_UNDEFINED_INCOMPLETE_SIX_SCENE_COVERAGE",
        "incomplete permutation behavior mismatch",
    )
    _require(
        active_inference.get("partial_scene_permutation_forbidden") is True,
        "partial-scene permutation allowed",
    )
    agreement = _require_mapping(contract.get("cross_backend_analysis"), "cross-backend analysis")
    scene_spearman = _require_mapping(agreement.get("scene_median_spearman"), "scene Spearman")
    station_spearman = _require_mapping(agreement.get("station_median_spearman"), "station Spearman")
    _require(
        scene_spearman.get("required_complete_pairs") == 6
        and scene_spearman.get("constant_input_behavior")
        == "NULL_WITH_SPEARMAN_UNDEFINED_CONSTANT_INPUT",
        "scene Spearman undefined rule mismatch",
    )
    _require(
        station_spearman.get("required_complete_pairs") == 18
        and station_spearman.get("constant_input_behavior")
        == "NULL_WITH_SPEARMAN_UNDEFINED_CONSTANT_INPUT",
        "station Spearman undefined rule mismatch",
    )
    active_reassociation = _require_mapping(
        contract.get("reassociation_analysis"), "active reassociation analysis"
    )
    reassociation_missing = _require_mapping(
        active_reassociation.get("missingness"), "reassociation missingness"
    )
    expected_common_fields = [
        "common_association_valid",
        "common_association_invalid_reason",
        "common_association_invalid_detail",
    ]
    expected_common_reasons = [
        "NO_INITIAL_CORRESPONDENCE",
        "NO_FINAL_CORRESPONDENCE",
        "INSUFFICIENT_VALID_NORMALS",
        "NONFINITE_COMMON_METRICS",
        "OTHER",
    ]
    _require(
        reassociation_missing.get("required_result_status_fields")
        == expected_common_fields,
        "common-association status fields mismatch",
    )
    _require(
        reassociation_missing.get("common_association_invalid_reason_enum")
        == expected_common_reasons,
        "common-association invalid-reason enum mismatch",
    )
    _require(
        reassociation_missing.get("result_schema_runner_and_verifier_must_preserve_status_fields")
        is True,
        "common-association status preservation not required",
    )
    _require(
        reassociation_missing.get("formal_scene_metric_required_finite_n") == 30
        and reassociation_missing.get("scene_association_required_complete_pairs") == 6
        and reassociation_missing.get("centered_required_complete_pairs_per_scene") == 30
        and reassociation_missing.get("centered_required_complete_scene_strata") == 6
        and reassociation_missing.get("centered_required_total_complete_pairs") == 180,
        "reassociation completeness gate mismatch",
    )
    active_systematic = _require_mapping(
        contract.get("systematic_component"), "active systematic component"
    )
    _require(
        active_systematic.get("required_finite_translation_vector_count") == 10,
        "systematic fraction finite-vector gate mismatch",
    )
    incomplete_vectors = _require_mapping(
        active_systematic.get("incomplete_vector_rule"), "systematic incomplete-vector rule"
    )
    _require(
        incomplete_vectors.get("systematic_fraction") is None
        and incomplete_vectors.get("status")
        == "SYSTEMATIC_FRACTION_UNDEFINED_INCOMPLETE_10_VECTORS"
        and incomplete_vectors.get("station_retained") is True,
        "systematic incomplete-vector behavior mismatch",
    )

    # Reuse every substantive candidate rule after normalizing only lifecycle
    # fields. Candidate evidence remains independently hash-bound elsewhere.
    amendment["status"] = CANDIDATE_STATUS
    amendment["activation_effective"] = False
    amendment["analysis_contract"] = dict(active_contract)
    amendment["analysis_contract"]["status"] = CANDIDATE_STATUS
    contract["status"] = CANDIDATE_STATUS
    contract["activation_effective"] = False
    validate_candidate_payloads(amendment, contract, supersession)


def validate_active_transition_payloads(
    *,
    active_hashes: Mapping[str, str],
    activation_record: Mapping[str, Any],
    active_pointer: Mapping[str, Any],
    transition: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    candidate_report_sha256: str,
    candidate_inventory: Mapping[str, Any],
    candidate_inventory_sha256: str,
    prelock: Mapping[str, Any],
    prelock_sha256: str,
    activation_record_sha256: str,
    active_pointer_sha256: str,
) -> None:
    """Validate the candidate-to-active binding graph using supplied hashes."""

    candidate_report = _require_mapping(candidate_report, "candidate verifier report")
    _require(candidate_report.get("pass") is True, "candidate verifier did not pass")
    _require(candidate_report.get("verification_status") == "PASS", "candidate verifier status mismatch")
    _require(candidate_report.get("phase") == "CANDIDATE_PRE_ACTIVATION", "candidate verifier phase mismatch")
    _require(candidate_report.get("activation_effective") is False, "candidate verifier claims activation")
    _require(candidate_report.get("actual_formal_trials") == 0, "candidate verifier observed formal trials")
    for key, expected in EXPECTED_CANDIDATE_HASHES.items():
        _require(candidate_report.get(key) == expected, f"candidate verifier {key} mismatch")

    inventory = _require_mapping(candidate_inventory, "candidate archive inventory")
    _require(inventory.get("status") == "VERIFIED_CANDIDATE_BYTES_PRESERVED", "candidate inventory status mismatch")
    files = inventory.get("files")
    _require(type(files) is list and len(files) == 4, "candidate inventory file count mismatch")
    inventory_hashes = {Path(row["path"]).name: row.get("sha256") for row in files}
    _require(inventory_hashes.get("zero_perturbation_mainline_v1_1_r1.json") == EXPECTED_CANDIDATE_HASHES["amendment_json_sha256"], "archived candidate amendment JSON mismatch")
    _require(inventory_hashes.get("zero_perturbation_mainline_v1_1_r1.md") == EXPECTED_CANDIDATE_HASHES["amendment_md_sha256"], "archived candidate amendment Markdown mismatch")
    _require(inventory_hashes.get("zero_perturbation_analysis_contract_v1_1_r1.json") == EXPECTED_CANDIDATE_HASHES["analysis_contract_sha256"], "archived candidate contract mismatch")
    _require(inventory_hashes.get("zero_perturbation_analysis_protocol_v1_1_r1.md") == EXPECTED_CANDIDATE_HASHES["analysis_protocol_sha256"], "archived candidate protocol mismatch")
    _require(inventory.get("candidate_independent_verification_sha256") == candidate_report_sha256, "candidate inventory verifier SHA mismatch")

    prelock = _require_mapping(prelock, "prelock reauthentication")
    _require(prelock.get("pass") is True and prelock.get("status") == "PASS", "prelock reauthentication failed")
    _require(prelock.get("final_scene_count") == 6, "prelock scene count mismatch")
    _require(prelock.get("final_station_count") == 18, "prelock station count mismatch")
    _require(prelock.get("final_target_count") == 18, "prelock target count mismatch")
    _require(prelock.get("final_snapshot_count") == 180, "prelock snapshot count mismatch")
    _require(prelock.get("w02_attempt1_status") == "INVALID_ACQUISITION", "prelock W02 attempt 1 mismatch")
    _require(prelock.get("w02_attempt2_geometry_class") == "WEAK", "prelock W02 attempt 2 mismatch")
    _require(prelock.get("W04_INCLUDED_IN_FINAL_SET") is False, "prelock includes W04")
    _require(prelock.get("actual_formal_trials") == 0, "prelock has formal trials")
    _require(prelock.get("FORMAL_REGISTRATION_AUTHORIZED") is False, "prelock authorizes registration")

    record = _require_mapping(activation_record, "activation record")
    _require(record.get("status") == ACTIVE_STATUS, "activation record status mismatch")
    _require(record.get("amendment_id") == AMENDMENT_ID, "activation record amendment mismatch")
    _require(record.get("activated_before_any_formal_icp") is True, "activation record timing mismatch")
    _require(record.get("formal_trial_count_at_activation") == 0, "activation record trial count mismatch")
    record_candidate = _require_mapping(record.get("candidate_state"), "activation candidate state")
    for key, expected in EXPECTED_CANDIDATE_HASHES.items():
        _require(record_candidate.get(key) == expected, f"activation record candidate {key} mismatch")
    _require(record_candidate.get("independent_verification_sha256") == candidate_report_sha256, "activation record candidate verifier SHA mismatch")
    record_archive = _require_mapping(record.get("candidate_bytes_archive"), "activation candidate archive")
    _require(record_archive.get("inventory_sha256") == candidate_inventory_sha256, "activation record candidate inventory SHA mismatch")
    _require(record_archive.get("exact_candidate_file_count") == 4, "activation record candidate file count mismatch")
    record_active = _require_mapping(record.get("active_state"), "activation active state")
    for key, expected in active_hashes.items():
        _require(record_active.get(key) == expected, f"activation record active {key} mismatch")
    record_prelock = _require_mapping(record.get("final_dataset_prelock_reauthentication"), "activation prelock")
    _require(record_prelock.get("sha256") == prelock_sha256, "activation record prelock SHA mismatch")
    for flag in ("activation_grants_backend_execution", "FORMAL_AUTHORITY", "FORMAL_LOCK_ISSUED", "FORMAL_ICP_UNLOCKED", "FORMAL_REGISTRATION_AUTHORIZED"):
        _require(record.get(flag) is False, f"activation record {flag} must be false")
    _require(record.get("actual_formal_trials") == 0, "activation record has formal trials")

    pointer = _require_mapping(active_pointer, "ACTIVE_PROTOCOL")
    _require(pointer.get("status") == ACTIVE_STATUS, "ACTIVE_PROTOCOL status mismatch")
    _require(pointer.get("active_amendment_id") == AMENDMENT_ID, "ACTIVE_PROTOCOL amendment mismatch")
    _require(pointer.get("activation_effective") is True, "ACTIVE_PROTOCOL is ineffective")
    _require(pointer.get("formal_trial_count_at_activation") == 0, "ACTIVE_PROTOCOL trial count mismatch")
    _require(_require_mapping(pointer.get("active_amendment"), "pointer amendment").get("sha256") == active_hashes["amendment_json_sha256"], "pointer amendment SHA mismatch")
    _require(_require_mapping(pointer.get("active_analysis_contract"), "pointer contract").get("sha256") == active_hashes["analysis_contract_sha256"], "pointer contract SHA mismatch")
    _require(_require_mapping(pointer.get("active_analysis_protocol"), "pointer protocol").get("sha256") == active_hashes["analysis_protocol_sha256"], "pointer protocol SHA mismatch")
    _require(_require_mapping(pointer.get("activation_record"), "pointer activation record").get("sha256") == activation_record_sha256, "pointer activation record SHA mismatch")
    _require(_require_mapping(pointer.get("candidate_verification"), "pointer candidate verifier").get("sha256") == candidate_report_sha256, "pointer candidate verifier SHA mismatch")
    _require(_require_mapping(pointer.get("final_dataset_prelock_reauthentication"), "pointer prelock").get("sha256") == prelock_sha256, "pointer prelock SHA mismatch")
    _require(pointer.get("physical_reference_semantics") == PHYSICAL_REFERENCE, "pointer physical semantics mismatch")
    for flag in ("activation_grants_backend_execution", "FORMAL_AUTHORITY", "FORMAL_LOCK_ISSUED", "FORMAL_ICP_UNLOCKED", "FORMAL_REGISTRATION_AUTHORIZED"):
        _require(pointer.get(flag) is False, f"ACTIVE_PROTOCOL {flag} must be false")
    _require(pointer.get("actual_formal_trials") == 0, "ACTIVE_PROTOCOL has formal trials")

    transition = _require_mapping(transition, "activation transition")
    _require(transition.get("status") == "COMPLETED", "transition status mismatch")
    before = _require_mapping(transition.get("before"), "transition before")
    after = _require_mapping(transition.get("after"), "transition after")
    _require(before.get("status") == CANDIDATE_STATUS, "transition before status mismatch")
    _require(before.get("activation_effective") is False, "transition before is active")
    for key, expected in EXPECTED_CANDIDATE_HASHES.items():
        _require(before.get(key) == expected, f"transition before {key} mismatch")
    _require(before.get("candidate_verification_sha256") == candidate_report_sha256, "transition candidate verifier SHA mismatch")
    _require(before.get("candidate_archive_inventory_sha256") == candidate_inventory_sha256, "transition candidate inventory SHA mismatch")
    _require(after.get("status") == ACTIVE_STATUS, "transition after status mismatch")
    _require(after.get("activation_effective") is True, "transition after is inactive")
    for key, expected in active_hashes.items():
        _require(after.get(key) == expected, f"transition after {key} mismatch")
    _require(after.get("activation_record_sha256") == activation_record_sha256, "transition activation record SHA mismatch")
    _require(after.get("active_protocol_pointer_sha256") == active_pointer_sha256, "transition pointer SHA mismatch")
    for flag in ("activation_grants_backend_execution", "FORMAL_AUTHORITY", "FORMAL_ICP_UNLOCKED", "FORMAL_REGISTRATION_AUTHORIZED"):
        _require(transition.get(flag) is False, f"transition {flag} must be false")
    _require(transition.get("actual_formal_trials") == 0, "transition has formal trials")


def validate_missingness_clarification_payload(
    clarification_value: Mapping[str, Any],
) -> None:
    """Validate deterministic C1 missingness semantics without producer reuse."""

    clarification = _require_mapping(clarification_value, "C1 clarification")
    _require(
        clarification.get("schema")
        == "mid360_fmb1_zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1",
        "C1 schema mismatch",
    )
    _require(clarification.get("clarification_id") == CLARIFICATION_ID, "C1 ID mismatch")
    _require(
        clarification.get("status") == "ACTIVE_PRELOCK_CLARIFICATION",
        "C1 status mismatch",
    )
    _require(
        clarification.get("clarification_before_formal_lock") is True,
        "C1 is not pre-lock",
    )
    _require(
        clarification.get("clarification_before_any_formal_icp") is True,
        "C1 is not pre-ICP",
    )
    _require(
        clarification.get("clarification_at_formal_trial_count") == 0,
        "C1 was created after formal results",
    )
    _require(clarification.get("registration_result_used") is False, "C1 used registration results")
    principles = _require_mapping(clarification.get("principles"), "C1 principles")
    for field in (
        "all_360_planned_trial_ids_remain_in_accounting",
        "silent_row_deletion_forbidden",
        "scientific_failure_retry_forbidden",
        "infrastructure_retry_lineage_retained",
        "imputation_forbidden",
        "winsorization_forbidden",
        "undefined_formal_summary_never_replaced_by_available_case_summary",
        "result_dependent_threshold_changes_forbidden",
    ):
        _require(principles.get(field) is True, f"C1 principle {field} is not enforced")
    authoritative = _require_mapping(
        clarification.get("authoritative_trial_outcome"), "C1 authoritative outcome"
    )
    _require(
        authoritative.get("selection_order")
        == "LOWEST_ATTEMPT_INDEX_WITH_SCHEMA_VALID_INFRASTRUCTURE_PASS_SCIENTIFIC_OUTCOME",
        "C1 authoritative attempt selection mismatch",
    )
    _require(
        authoritative.get("duplicate_scientific_outcome_after_one_exists") == "FORBIDDEN",
        "C1 permits duplicate scientific outcomes",
    )
    _require(
        authoritative.get("scientific_nonfinite_outcome")
        == "AUTHORITATIVE_SCIENTIFIC_FAILURE_NOT_RETRYABLE",
        "C1 scientific nonfinite behavior mismatch",
    )
    row_classes = _require_mapping(
        clarification.get("row_endpoint_classification"), "C1 row classification"
    )
    _require(
        row_classes.get("FINITE_SCIENTIFIC_VALUE")
        == 'infrastructure_status="OK", scientific execution completed, finite_result true, and the endpoint value is finite',
        "C1 finite scientific row classification does not use infrastructure_status=\"OK\"",
    )
    _require(
        row_classes.get("SCIENTIFIC_UNDEFINED_NONFINITE")
        == 'infrastructure_status="OK" and scientific execution completed, but finite_result is false or the endpoint is nonfinite/undefined',
        "C1 nonfinite scientific row classification does not use infrastructure_status=\"OK\"",
    )
    mandatory = clarification.get("mandatory_counts_at_every_station_scene_backend_summary")
    _require(
        mandatory
        == [
            "planned_n",
            "authoritative_scientific_outcome_n",
            "finite_endpoint_n",
            "scientific_undefined_nonfinite_n",
            "unresolved_infrastructure_failure_n",
            "resolved_infrastructure_attempt_n",
            "solver_nonconverged_finite_n",
            "endpoint_specific_undefined_n",
            "formal_summary_status",
        ],
        "C1 mandatory count fields mismatch",
    )
    available = _require_mapping(
        clarification.get("available_case_descriptive_rule"),
        "C1 available-case descriptive rule",
    )
    _require(
        available.get("enabled") is True
        and available.get("minimum_finite_n") == 1
        and available.get("statistics") == ["median", "q25", "q75", "q95"]
        and available.get("quantile_method") == "linear"
        and available.get("label")
        == "AVAILABLE_CASE_DESCRIPTIVE_NOT_PRIMARY_NOT_INFERENTIAL"
        and available.get("may_replace_formal_summary") is False,
        "C1 available-case limits mismatch",
    )
    summary = _require_mapping(
        clarification.get("formal_translation_rotation_summary_rule"),
        "C1 formal summary rule",
    )
    station = _require_mapping(summary.get("station"), "C1 station summary")
    scene = _require_mapping(summary.get("scene"), "C1 scene summary")
    _require(
        station.get("planned_n") == 10
        and station.get("required_finite_endpoint_n") == 10
        and station.get("required_scientific_undefined_nonfinite_n") == 0
        and station.get("required_unresolved_infrastructure_failure_n") == 0,
        "C1 station completeness mismatch",
    )
    _require(
        scene.get("planned_n") == 30
        and scene.get("required_finite_endpoint_n") == 30
        and scene.get("required_defined_station_summaries") == 3
        and scene.get("required_scientific_undefined_nonfinite_n") == 0
        and scene.get("required_unresolved_infrastructure_failure_n") == 0,
        "C1 scene completeness mismatch",
    )
    _require(summary.get("nonfinite_or_missing_values_used_in_quantile") is False, "C1 quantiles use undefined values")
    _require(summary.get("available_case_values_used_for_formal_inference") is False, "C1 formal inference uses available cases")
    permutation = _require_mapping(
        clarification.get("weak_rich_exact_permutation_rule"), "C1 permutation rule"
    )
    _require(
        permutation.get("required_defined_scene_summaries") == 6
        and permutation.get("required_defined_rich_scene_summaries") == 3
        and permutation.get("required_defined_weak_scene_summaries") == 3
        and permutation.get("allocation_count_when_defined") == 20,
        "C1 permutation completeness mismatch",
    )
    undefined_permutation = _require_mapping(
        permutation.get("undefined_behavior"), "C1 undefined permutation"
    )
    _require(
        undefined_permutation.get("estimand") is None
        and undefined_permutation.get("p_value") is None
        and undefined_permutation.get("status")
        == "INFERENCE_UNDEFINED_INCOMPLETE_SIX_SCENE_COVERAGE",
        "C1 undefined permutation behavior mismatch",
    )
    spearman = _require_mapping(clarification.get("spearman_rules"), "C1 Spearman")
    _require(
        spearman.get("implementation") == "SCIPY_STATS_SPEARMANR_WITH_AVERAGE_RANKS",
        "C1 Spearman implementation mismatch",
    )
    _require(
        _require_mapping(spearman.get("scene_backend_agreement"), "C1 scene Spearman").get("required_complete_pairs") == 6,
        "C1 scene Spearman pair gate mismatch",
    )
    _require(
        _require_mapping(spearman.get("station_backend_agreement"), "C1 station Spearman").get("required_complete_pairs") == 18,
        "C1 station Spearman pair gate mismatch",
    )
    reassociation = _require_mapping(
        clarification.get("reassociation_missingness_rule"), "C1 reassociation"
    )
    _require(
        reassociation.get("required_result_status_fields")
        == [
            "common_association_valid",
            "common_association_invalid_reason",
            "common_association_invalid_detail",
        ],
        "C1 common status fields mismatch",
    )
    _require(
        reassociation.get("common_association_invalid_reason_enum")
        == [
            "NO_INITIAL_CORRESPONDENCE",
            "NO_FINAL_CORRESPONDENCE",
            "INSUFFICIENT_VALID_NORMALS",
            "NONFINITE_COMMON_METRICS",
            "OTHER",
        ],
        "C1 common invalid-reason enum mismatch",
    )
    centered = _require_mapping(
        reassociation.get("within_scene_centered_association"), "C1 centered association"
    )
    _require(
        centered.get("required_complete_snapshot_pairs_per_scene") == 30
        and centered.get("required_complete_scene_strata") == 6
        and centered.get("required_total_complete_pairs") == 180,
        "C1 centered association completeness mismatch",
    )
    systematic = _require_mapping(
        clarification.get("systematic_fraction_missingness_rule"), "C1 systematic rule"
    )
    _require(
        systematic.get("required_finite_translation_vectors_per_station") == 10,
        "C1 systematic vector gate mismatch",
    )
    incomplete = _require_mapping(
        systematic.get("incomplete_station_behavior"), "C1 incomplete systematic"
    )
    _require(
        incomplete.get("systematic_fraction") is None
        and incomplete.get("status")
        == "SYSTEMATIC_FRACTION_UNDEFINED_INCOMPLETE_10_VECTORS"
        and incomplete.get("station_retained") is True,
        "C1 incomplete systematic behavior mismatch",
    )
    infrastructure = _require_mapping(
        clarification.get("infrastructure_failure_rule"), "C1 infrastructure rule"
    )
    _require(infrastructure.get("all_attempts_retained") is True, "C1 drops infrastructure attempts")
    _require(infrastructure.get("unresolved_failure_trial_retained_in_planned_denominator") is True, "C1 drops unresolved infrastructure trial")
    _require(infrastructure.get("unresolved_failure_endpoint") is None, "C1 unresolved endpoint must be null")
    _require(
        infrastructure.get("unresolved_failure_affects_completeness_gate") is True,
        "C1 unresolved infrastructure does not affect completeness",
    )
    _require(
        infrastructure.get("infrastructure_failure_may_not_be_reclassified_from_scientific_failure")
        is True,
        "C1 permits scientific failure to be reclassified as infrastructure",
    )
    lock_precondition = _require_mapping(
        clarification.get("formal_lock_precondition"), "C1 lock precondition"
    )
    for field in (
        "result_schema_contains_common_association_status_fields",
        "runner_preserves_common_association_status_fields",
        "independent_verifier_rejects_missing_or_invalid_status_binding",
    ):
        _require(lock_precondition.get(field) is True, f"C1 lock precondition {field} missing")
    for flag in ("FORMAL_LOCK_ISSUED", "FORMAL_ICP_UNLOCKED", "FORMAL_REGISTRATION_AUTHORIZED"):
        _require(clarification.get(flag) is False, f"C1 {flag} must be false")
    _require(clarification.get("actual_formal_trials") == 0, "C1 has formal trials")


def validate_missingness_clarification_transition_payloads(
    *,
    initial_active_hashes: Mapping[str, str],
    current_active_hashes: Mapping[str, str],
    prior_inventory: Mapping[str, Any],
    prior_inventory_sha256: str,
    clarification: Mapping[str, Any],
    clarification_json_sha256: str,
    clarification_markdown_sha256: str,
    activation_record: Mapping[str, Any],
    activation_record_sha256: str,
    active_pointer: Mapping[str, Any],
    active_pointer_sha256: str,
    clarification_transition: Mapping[str, Any],
) -> None:
    """Validate the initial-ACTIVE to clarified-ACTIVE hash transition."""

    validate_missingness_clarification_payload(clarification)
    inventory = _require_mapping(prior_inventory, "pre-C1 active inventory")
    _require(
        inventory.get("status") == "ACTIVE_BYTES_PRESERVED_BEFORE_PRELOCK_CLARIFICATION",
        "pre-C1 inventory status mismatch",
    )
    _require(inventory.get("clarification_id") == CLARIFICATION_ID, "pre-C1 inventory ID mismatch")
    _require(inventory.get("formal_trial_count_at_archive") == 0, "pre-C1 archive is post-result")
    _require(inventory.get("formal_lock_issued_at_archive") is False, "pre-C1 archive is post-lock")
    files = inventory.get("files")
    _require(type(files) is list and len(files) == 7, "pre-C1 inventory file count mismatch")
    hashes = {row.get("name"): row.get("sha256") for row in files}
    _require(hashes.get("zero_perturbation_mainline_v1_1_r1.json") == initial_active_hashes["amendment_json_sha256"], "pre-C1 amendment JSON mismatch")
    _require(hashes.get("zero_perturbation_mainline_v1_1_r1.md") == initial_active_hashes["amendment_md_sha256"], "pre-C1 amendment Markdown mismatch")
    _require(hashes.get("zero_perturbation_analysis_contract_v1_1_r1.json") == initial_active_hashes["analysis_contract_sha256"], "pre-C1 analysis contract mismatch")
    _require(hashes.get("zero_perturbation_analysis_protocol_v1_1_r1.md") == initial_active_hashes["analysis_protocol_sha256"], "pre-C1 analysis protocol mismatch")
    _require(hashes.get("amendment_activation_record_v1_1_r1.json") == EXPECTED_INITIAL_ACTIVATION_RECORD_SHA256, "pre-C1 activation record mismatch")
    _require(hashes.get("amendment_activation_transition_v1_1_r1.json") == EXPECTED_INITIAL_ACTIVATION_TRANSITION_SHA256, "pre-C1 activation transition mismatch")
    _require(hashes.get("ACTIVE_PROTOCOL.json") == EXPECTED_INITIAL_ACTIVE_POINTER_SHA256, "pre-C1 active pointer mismatch")

    record = _require_mapping(activation_record, "clarified activation record")
    record_active = _require_mapping(record.get("active_state"), "clarified record active state")
    for key, expected in current_active_hashes.items():
        _require(record_active.get(key) == expected, f"clarified record {key} mismatch")
    record_c1 = _require_mapping(
        record.get("prelock_scientific_clarification"), "record C1 clarification"
    )
    _require(record_c1.get("clarification_id") == CLARIFICATION_ID, "record C1 ID mismatch")
    _require(record_c1.get("json_sha256") == clarification_json_sha256, "record C1 JSON SHA mismatch")
    _require(record_c1.get("markdown_sha256") == clarification_markdown_sha256, "record C1 Markdown SHA mismatch")
    _require(record_c1.get("prior_active_inventory_sha256") == prior_inventory_sha256, "record prior-active inventory SHA mismatch")
    _require(record_c1.get("clarification_before_formal_lock") is True, "record C1 is post-lock")
    _require(record_c1.get("clarification_at_formal_trial_count") == 0, "record C1 is post-result")

    pointer = _require_mapping(active_pointer, "clarified ACTIVE_PROTOCOL")
    _require(pointer.get("schema") == "mid360_fmb1_active_protocol_pointer_v1_1_r1_c1", "clarified pointer schema mismatch")
    _require(_require_mapping(pointer.get("active_amendment"), "clarified pointer amendment").get("sha256") == current_active_hashes["amendment_json_sha256"], "clarified pointer amendment mismatch")
    _require(_require_mapping(pointer.get("active_analysis_contract"), "clarified pointer contract").get("sha256") == current_active_hashes["analysis_contract_sha256"], "clarified pointer contract mismatch")
    _require(_require_mapping(pointer.get("active_analysis_protocol"), "clarified pointer protocol").get("sha256") == current_active_hashes["analysis_protocol_sha256"], "clarified pointer protocol mismatch")
    _require(_require_mapping(pointer.get("activation_record"), "clarified pointer record").get("sha256") == activation_record_sha256, "clarified pointer record mismatch")
    pointer_c1 = _require_mapping(
        pointer.get("active_prelock_scientific_clarification"), "clarified pointer C1"
    )
    _require(pointer_c1.get("clarification_id") == CLARIFICATION_ID, "pointer C1 ID mismatch")
    _require(pointer_c1.get("sha256") == clarification_json_sha256, "pointer C1 SHA mismatch")
    _require(pointer_c1.get("prior_active_inventory_sha256") == prior_inventory_sha256, "pointer prior-active inventory mismatch")

    transition = _require_mapping(
        clarification_transition, "C1 clarification transition"
    )
    _require(transition.get("status") == "COMPLETED_PRELOCK", "C1 transition status mismatch")
    _require(transition.get("clarification_id") == CLARIFICATION_ID, "C1 transition ID mismatch")
    _require(transition.get("clarification_before_formal_lock") is True, "C1 transition is post-lock")
    _require(transition.get("clarification_before_any_formal_icp") is True, "C1 transition is post-ICP")
    _require(transition.get("clarification_at_formal_trial_count") == 0, "C1 transition is post-result")
    before = _require_mapping(transition.get("before"), "C1 transition before")
    after = _require_mapping(transition.get("after"), "C1 transition after")
    for key, expected in initial_active_hashes.items():
        _require(before.get(key) == expected, f"C1 transition before {key} mismatch")
    _require(before.get("activation_record_sha256") == EXPECTED_INITIAL_ACTIVATION_RECORD_SHA256, "C1 transition before activation record mismatch")
    _require(before.get("active_protocol_pointer_sha256") == EXPECTED_INITIAL_ACTIVE_POINTER_SHA256, "C1 transition before pointer mismatch")
    _require(before.get("activation_transition_sha256") == EXPECTED_INITIAL_ACTIVATION_TRANSITION_SHA256, "C1 transition before activation transition mismatch")
    _require(before.get("archive_inventory_sha256") == prior_inventory_sha256, "C1 transition before inventory mismatch")
    for key, expected in current_active_hashes.items():
        _require(after.get(key) == expected, f"C1 transition after {key} mismatch")
    _require(after.get("activation_record_sha256") == activation_record_sha256, "C1 transition after activation record mismatch")
    _require(after.get("active_protocol_pointer_sha256") == active_pointer_sha256, "C1 transition after pointer mismatch")
    _require(after.get("clarification_json_sha256") == clarification_json_sha256, "C1 transition clarification JSON mismatch")
    _require(after.get("clarification_markdown_sha256") == clarification_markdown_sha256, "C1 transition clarification Markdown mismatch")
    for field in (
        "dataset_changed",
        "trial_plan_membership_changed",
        "identity_initialization_changed",
        "backend_parameter_contract_changed",
        "capture_radius_track_changed",
        "physical_reference_semantics_changed",
        "activation_grants_backend_execution",
        "FORMAL_LOCK_ISSUED",
        "FORMAL_ICP_UNLOCKED",
        "FORMAL_REGISTRATION_AUTHORIZED",
    ):
        _require(transition.get(field) is False, f"C1 transition {field} must be false")
    _require(transition.get("actual_formal_trials") == 0, "C1 transition has formal trials")


def validate_common_association_status_artifacts(repo_root: str | Path) -> dict[str, str]:
    """Fail closed unless schema, producer, and execution verifier retain C1 status."""

    root = Path(repo_root).resolve()
    schema_path = (
        root
        / "experiments/mid360_formal_batch1/zero_perturbation_trial_result_schema_v1_1.json"
    )
    runner_path = (
        root / "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_runner.py"
    )
    execution_verifier_path = (
        root / "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_verify.py"
    )
    result_validator_path = (
        root / "experiments/mid360_formal_batch1/zero_perturbation_r1_trial_assets.py"
    )
    experiments_init_path = root / "experiments/__init__.py"
    formal_package_init_path = root / "experiments/mid360_formal_batch1/__init__.py"
    harness_init_path = root / "src/phase_a_harness/__init__.py"
    schema = _load_json(schema_path)
    expected_fields = [
        "common_association_valid",
        "common_association_invalid_reason",
        "common_association_invalid_detail",
    ]
    expected_reasons = [
        "NO_INITIAL_CORRESPONDENCE",
        "NO_FINAL_CORRESPONDENCE",
        "INSUFFICIENT_VALID_NORMALS",
        "NONFINITE_COMMON_METRICS",
        "OTHER",
    ]
    required = schema.get("required")
    _require(type(required) is list, "result schema required list is absent")
    _require(
        all(field in required for field in expected_fields),
        "result schema does not require all common-association status fields",
    )
    properties = _require_mapping(schema.get("properties"), "result schema properties")
    valid_property = _require_mapping(
        properties.get("common_association_valid"), "common_association_valid schema"
    )
    valid_variants = valid_property.get("oneOf")
    _require(
        type(valid_variants) is list
        and {item.get("type") for item in valid_variants if type(item) is dict}
        == {"boolean", "null"},
        "common_association_valid must be boolean or null",
    )
    reason_property = _require_mapping(
        properties.get("common_association_invalid_reason"),
        "common_association_invalid_reason schema",
    )
    reason_variants = reason_property.get("oneOf")
    _require(type(reason_variants) is list, "common invalid reason variants absent")
    enum_variant = next(
        (item for item in reason_variants if type(item) is dict and "enum" in item),
        None,
    )
    _require(
        type(enum_variant) is dict and enum_variant.get("enum") == expected_reasons,
        "common invalid reason enum differs",
    )
    detail_property = _require_mapping(
        properties.get("common_association_invalid_detail"),
        "common_association_invalid_detail schema",
    )
    detail_variants = detail_property.get("oneOf")
    detail_string = next(
        (
            item
            for item in detail_variants
            if type(item) is dict and item.get("type") == "string"
        ),
        None,
    ) if type(detail_variants) is list else None
    _require(
        type(detail_string) is dict
        and detail_string.get("minLength") == 1
        and detail_string.get("maxLength") == 1000,
        "common invalid detail bounds differ",
    )

    conditional_rules = schema.get("allOf")
    _require(type(conditional_rules) is list, "result schema conditional rules absent")

    def consequence(field: str, value: Any) -> Mapping[str, Any]:
        for rule_value in conditional_rules:
            if type(rule_value) is not dict:
                continue
            condition = rule_value.get("if", {}).get("properties", {}).get(field, {})
            if condition.get("const") == value:
                return _require_mapping(
                    rule_value.get("then", {}).get("properties"),
                    f"result schema consequence for {field}={value!r}",
                )
        _fail(f"result schema lacks consequence for {field}={value!r}")

    valid_consequence = consequence("common_association_valid", True)
    _require(
        _require_mapping(
            valid_consequence.get("common_association_invalid_reason"),
            "valid common reason consequence",
        ).get("type")
        == "null"
        and _require_mapping(
            valid_consequence.get("common_association_invalid_detail"),
            "valid common detail consequence",
        ).get("type")
        == "null",
        "valid common association does not force null reason/detail",
    )
    invalid_consequence = consequence("common_association_valid", False)
    _require(
        _require_mapping(
            invalid_consequence.get("common_association_invalid_reason"),
            "invalid common reason consequence",
        ).get("enum")
        == expected_reasons,
        "invalid common association does not require frozen reason enum",
    )
    other_consequence = consequence("common_association_invalid_reason", "OTHER")
    other_detail = _require_mapping(
        other_consequence.get("common_association_invalid_detail"),
        "OTHER common detail consequence",
    )
    _require(
        other_detail.get("type") == "string"
        and other_detail.get("minLength") == 1
        and other_detail.get("maxLength") == 1000,
        "OTHER common invalid reason does not require bounded detail",
    )
    _require(
        schema.get("x-common-invalid-reason-enum") == expected_reasons
        and schema.get("x-common-invalid-reason-retained") is True
        and schema.get("x-common-status-null-only-without-finite-pose") is True
        and schema.get("x-common-invalid-detail-required-for-other") is True,
        "result schema common-status extension metadata differs",
    )

    runner_source = runner_path.read_text(encoding="utf-8")
    execution_verifier_source = execution_verifier_path.read_text(encoding="utf-8")
    _require(
        "common.safe_analyze_estimated_transform" in runner_source,
        "runner no longer calls the common reassociation implementation",
    )
    for field in expected_fields:
        _require(
            runner_source.count(field) >= 3,
            f"runner does not preserve {field} across producer/result paths",
        )
        _require(
            execution_verifier_source.count(field) >= 2,
            f"execution verifier does not independently check {field}",
        )
    for reason in expected_reasons:
        _require(
            reason in execution_verifier_source,
            f"execution verifier omits common invalid reason {reason}",
        )
    return {
        "result_schema_sha256": _sha256(schema_path),
        "runner_sha256": _sha256(runner_path),
        "execution_verifier_sha256": _sha256(execution_verifier_path),
        "result_validator_sha256": _sha256(result_validator_path),
        "experiments_package_init_sha256": _sha256(experiments_init_path),
        "mid360_formal_batch1_package_init_sha256": _sha256(formal_package_init_path),
        "phase_a_harness_package_init_sha256": _sha256(harness_init_path),
    }


def verify_candidate(repo_root: str | Path) -> dict[str, Any]:
    """Verify the candidate against independent on-disk evidence."""

    root = Path(repo_root).resolve()
    amendment_path = root / "experiments/mid360_formal_batch1/amendments/zero_perturbation_mainline_v1_1_r1.json"
    amendment_md_path = amendment_path.with_suffix(".md")
    contract_path = root / "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_contract_v1_1_r1.json"
    protocol_md_path = root / "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_protocol_v1_1_r1.md"
    supersession_path = root / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.SUPERSEDED.json"
    old_json_path = root / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.json"
    old_md_path = root / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.md"

    amendment = _load_json(amendment_path)
    contract = _load_json(contract_path)
    supersession = _load_json(supersession_path)
    validate_candidate_payloads(amendment, contract, supersession)

    amendment_sha = _sha256(amendment_path)
    amendment_md_sha = _sha256(amendment_md_path)
    contract_sha = _sha256(contract_path)
    protocol_md_sha = _sha256(protocol_md_path)
    supersession_sha = _sha256(supersession_path)
    replacement = _require_mapping(
        supersession.get("replacement_candidate"), "supersession replacement candidate"
    )
    _require(
        replacement.get("json_sha256_at_candidate_review") == amendment_sha,
        "supersession amendment JSON candidate SHA mismatch",
    )
    _require(
        replacement.get("markdown_sha256_at_candidate_review") == amendment_md_sha,
        "supersession amendment Markdown candidate SHA mismatch",
    )

    _require(_sha256(old_json_path) == OLD_PROPOSAL_JSON_SHA256, "old proposal JSON bytes changed")
    _require(_sha256(old_md_path) == OLD_PROPOSAL_MD_SHA256, "old proposal Markdown bytes changed")
    _require(_sha256(root / "experiments/mid360_formal_batch1/preregistration.yaml") == ORIGINAL_PREREGISTRATION_SHA256, "original preregistration changed")
    _require(_sha256(root / "experiments/mid360_formal_batch1/analysis_protocol.md") == ORIGINAL_ANALYSIS_PROTOCOL_SHA256, "original capture-radius analysis changed")
    _require(_sha256(root / "frozen_assets/backend_parameter_contract.json") == BACKEND_CONTRACT_SHA256, "backend parameter contract changed")
    _require(_sha256(root / "src/phase_a_harness/common_association_analysis.py") == COMMON_ASSOCIATION_SHA256, "common reassociation implementation changed")

    old_proposal = _load_json(old_json_path)
    _require(old_proposal.get("status") == "PROPOSED_NOT_ACTIVE", "old proposal status changed")
    _require(old_proposal.get("activation_forbidden_by_this_file") is True, "old proposal activation firewall changed")
    old_prerequisites = old_proposal.get("activation_prerequisites")
    _require(type(old_prerequisites) is list and any("W04" in item for item in old_prerequisites), "old W04 premise is not demonstrable")

    pointer = _load_json(root / "results/mid360_formal_batch1/CURRENT_FINAL_DATASET.json")
    _require(pointer.get("scene_ids") == list(EXPECTED_SCENES), "current pointer scene set mismatch")
    _require(pointer.get("W04_IDENTIFIER_RETIRED") is True, "current pointer does not retire W04")
    _require(pointer.get("W04_INCLUDED_IN_FINAL_SET") is False, "current pointer includes W04")
    _require(pointer.get("w02_active_attempt") == 2, "current pointer does not select W02 attempt 2")
    _require(pointer.get("actual_formal_trials") == 0, "current pointer has formal results")

    final_root = root / "results/mid360_formal_batch1/final_dataset_v1"
    readiness = _load_json(final_root / "final_dataset_readiness.json")
    _require(readiness.get("FMB1_FINAL_DATASET_READY") is True, "final dataset is not ready")
    _require(readiness.get("FINAL_SCENE_COUNT") == 6, "final readiness scene count mismatch")
    _require(readiness.get("FINAL_STATION_COUNT") == 18, "final readiness station count mismatch")
    _require(readiness.get("FINAL_SNAPSHOT_COUNT") == 180, "final readiness snapshot count mismatch")
    _require(readiness.get("FMB1_W02_ATTEMPT2_FINAL_GEOMETRY_CLASS") == "WEAK", "readiness W02 attempt 2 mismatch")
    _require(readiness.get("W04_INCLUDED_IN_FINAL_SET") is False, "readiness includes W04")
    _require(readiness.get("actual_formal_trials") == 0, "readiness has formal results")

    lineage = _load_json(final_root / "acquisition_attempt_lineage.json")
    _require(lineage.get("scene_id") == "FMB1_W02", "lineage scene mismatch")
    _require(lineage.get("invalid_attempt") == 1, "lineage invalid attempt mismatch")
    _require(lineage.get("invalid_attempt_status") == "INVALID_ACQUISITION", "lineage invalid status mismatch")
    _require("WRONG_SCENE_LOCATION" in str(lineage.get("invalid_attempt_reason")), "lineage wrong-location reason absent")
    _require(lineage.get("valid_attempt") == 2, "lineage valid attempt mismatch")
    _require(lineage.get("valid_attempt_final_geometry_class") == "WEAK", "lineage valid class mismatch")
    _require(lineage.get("W04_IDENTIFIER_RETIRED") is True, "lineage does not retire W04")
    _require(lineage.get("formal_trial_count_at_correction") == 0, "lineage correction is post-result")

    scene_registry = _load_json(final_root / "final_scene_registry.yaml")
    scenes = scene_registry.get("scenes")
    _require(type(scenes) is list and len(scenes) == 6, "scene registry count mismatch")
    derived_scenes = {row.get("scene_id"): row.get("final_geometry_class") for row in scenes}
    _require(derived_scenes == EXPECTED_SCENES, "scene registry classes mismatch")
    for row in scenes:
        _require(row.get("station_count") == 3, f"{row.get('scene_id')} station count mismatch")
        _require(row.get("snapshot_count") == 30, f"{row.get('scene_id')} snapshot count mismatch")
    w02_scene = next(row for row in scenes if row.get("scene_id") == "FMB1_W02")
    _require(w02_scene.get("attempt") == 2, "scene registry uses old W02 attempt")

    station_registry = _load_json(final_root / "final_station_registry.yaml")
    stations = station_registry.get("stations")
    _require(type(stations) is list and len(stations) == 18, "station registry count mismatch")
    station_pairs = {(row.get("scene_id"), row.get("station_id")) for row in stations}
    _require(len(station_pairs) == 18, "station registry duplicates exist")
    for scene_id in EXPECTED_SCENES:
        _require(sum(row.get("scene_id") == scene_id for row in stations) == 3, f"{scene_id} does not have three stations")
    _require(all(row.get("attempt") == 2 for row in stations if row.get("scene_id") == "FMB1_W02"), "station registry includes W02 attempt 1")

    snapshots = _read_csv(final_root / "final_snapshot_manifest.csv")
    _require(len(snapshots) == 180, "snapshot manifest count mismatch")
    _require(len({row.get("snapshot_id") for row in snapshots}) == 180, "snapshot IDs are duplicated")
    _require(all(row.get("scene_id") in EXPECTED_SCENES for row in snapshots), "snapshot has non-final scene")
    _require(not any("W04" in str(row) for row in snapshots), "W04 entered snapshot manifest")
    _require(all(row.get("attempt") == "2" for row in snapshots if row.get("scene_id") == "FMB1_W02"), "W02 attempt 1 entered snapshot manifest")

    no_icp = _load_json(final_root / "NO_ICP_ATTESTATION.json")
    _require(no_icp.get("NO_ICP_ATTESTATION_PASS") is True, "NO-ICP attestation failed")
    _require(no_icp.get("actual_formal_trials") == 0, "NO-ICP attestation has formal trials")
    _require(no_icp.get("open3d_registration_call_count") == 0, "Open3D call count is nonzero")
    _require(no_icp.get("pcl_cli_invocation_count") == 0, "PCL invocation count is nonzero")

    amendment_md = amendment_md_path.read_text(encoding="utf-8")
    protocol_md = protocol_md_path.read_text(encoding="utf-8")
    for text, label in ((amendment_md, "amendment Markdown"), (protocol_md, "analysis protocol Markdown")):
        _require(PHYSICAL_REFERENCE in text, f"{label} lacks physical limitation")
        _require("PRESERVED_SUPPLEMENTARY_NOT_EXECUTED" in text, f"{label} lacks capture-radius preservation")
        _require("W02 attempt 1" in text and "W02 attempt 2" in text, f"{label} lacks attempt lineage")
        _require("W04" in text, f"{label} lacks W04 exclusion")
    _require("20 allocations" in protocol_md or "20-allocation" in protocol_md, "analysis Markdown lacks exact permutation count")
    _require("MODEL_TRANSFER_NOT_COMPATIBLE" in protocol_md, "analysis Markdown overstates model transfer")

    correction_record = _load_json(root / "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/proposal_correction_record_v1_1_r1.json")
    _require(correction_record.get("correction_at_formal_trial_count") == 0, "correction record trial count mismatch")
    _require(correction_record.get("registration_evidence_used") is False, "correction record used registration evidence")
    correction_old = _require_mapping(correction_record.get("old_proposal"), "correction old proposal")
    _require(correction_old.get("registry_status") == "SUPERSEDED_PROPOSAL", "correction registry status mismatch")
    _require(correction_old.get("status") == "PROPOSED_NOT_ACTIVE", "correction inside-file status mismatch")
    _require(correction_old.get("json_sha256") == OLD_PROPOSAL_JSON_SHA256, "correction old JSON SHA mismatch")
    _require(correction_old.get("markdown_sha256") == OLD_PROPOSAL_MD_SHA256, "correction old Markdown SHA mismatch")
    _require(correction_old.get("supersession_record_sha256") == supersession_sha, "correction supersession SHA mismatch")
    correction_candidate = _require_mapping(correction_record.get("r1_candidate"), "correction R1 candidate")
    _require(correction_candidate.get("amendment_json_sha256") == amendment_sha, "correction amendment JSON SHA mismatch")
    _require(correction_candidate.get("amendment_md_sha256") == amendment_md_sha, "correction amendment Markdown SHA mismatch")
    _require(correction_candidate.get("analysis_contract_sha256") == contract_sha, "correction analysis contract SHA mismatch")
    _require(correction_candidate.get("analysis_protocol_sha256") == protocol_md_sha, "correction analysis protocol SHA mismatch")
    difference = _load_json(root / "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/proposal_difference_report_v1_1_r1.json")
    _require(difference.get("old_proposal_preserved_byte_for_byte") is True, "difference report does not preserve proposal")
    _require(difference.get("old_proposal_json_sha256_before_r1") == OLD_PROPOSAL_JSON_SHA256, "difference old JSON SHA mismatch")
    _require(difference.get("old_proposal_md_sha256_before_r1") == OLD_PROPOSAL_MD_SHA256, "difference old Markdown SHA mismatch")
    _require(difference.get("candidate_outcome") == "R1_CANDIDATE_CREATED_NOT_YET_ACTIVE", "difference report prematurely activates R1")
    activation = _load_json(root / "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/zero_perturbation_v1_1_activation_review.json")
    _require(activation.get("candidate_review_pass") is True, "candidate review did not pass")
    _require(activation.get("independent_activation_verifier_pass") is False, "pre-verifier review falsely claims independent pass")
    _require(activation.get("activation_effective") is False, "activation review self-activates")
    review_assets = _require_mapping(activation.get("candidate_assets"), "activation review candidate assets")
    _require(review_assets.get("amendment_json_sha256") == amendment_sha, "review amendment JSON SHA mismatch")
    _require(review_assets.get("amendment_md_sha256") == amendment_md_sha, "review amendment Markdown SHA mismatch")
    _require(review_assets.get("analysis_contract_sha256") == contract_sha, "review analysis contract SHA mismatch")
    _require(review_assets.get("analysis_protocol_sha256") == protocol_md_sha, "review analysis protocol SHA mismatch")

    active_pointer = root / "experiments/mid360_formal_batch1/ACTIVE_PROTOCOL.json"
    activation_record = root / "experiments/mid360_formal_batch1/amendments/amendment_activation_record_v1_1_r1.json"
    _require(not active_pointer.exists(), "ACTIVE_PROTOCOL exists during candidate verification")
    _require(not activation_record.exists(), "activation record exists during candidate verification")

    return {
        "schema": "mid360_fmb1_zero_perturbation_protocol_r1_candidate_independent_verification",
        "verification_status": "PASS",
        "pass": True,
        "phase": "CANDIDATE_PRE_ACTIVATION",
        "amendment_id": AMENDMENT_ID,
        "amendment_status": CANDIDATE_STATUS,
        "activation_effective": False,
        "verified_counts": {"scenes": 6, "stations": 18, "snapshots": 180},
        "verified_scene_classes": copy.deepcopy(EXPECTED_SCENES),
        "w02_attempt1_invalid_retained": True,
        "w02_attempt2_admitted_weak": True,
        "w04_retired_and_excluded": True,
        "old_proposal_json_sha256_before_and_after": OLD_PROPOSAL_JSON_SHA256,
        "old_proposal_md_sha256_before_and_after": OLD_PROPOSAL_MD_SHA256,
        "amendment_json_sha256": amendment_sha,
        "amendment_md_sha256": amendment_md_sha,
        "analysis_contract_sha256": contract_sha,
        "analysis_protocol_sha256": protocol_md_sha,
        "physical_reference_semantics": PHYSICAL_REFERENCE,
        "capture_radius_track_status": "PRESERVED_SUPPLEMENTARY_NOT_EXECUTED",
        "model_transfer_status": "MODEL_TRANSFER_NOT_COMPATIBLE",
        "backend_modules_imported": 0,
        "backend_calls": 0,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
        "next_gate": "CREATE_ACTIVATION_TRANSITION_ONLY_AFTER_PARENT_AUTHORIZATION",
    }


def verify_active_transition(repo_root: str | Path) -> dict[str, Any]:
    """Verify candidate, initial ACTIVE, and clarified ACTIVE C1 history."""

    root = Path(repo_root).resolve()
    amendment_root = root / "experiments/mid360_formal_batch1/amendments"
    amendment_path = amendment_root / "zero_perturbation_mainline_v1_1_r1.json"
    amendment_md_path = amendment_root / "zero_perturbation_mainline_v1_1_r1.md"
    contract_path = amendment_root / "zero_perturbation_analysis_contract_v1_1_r1.json"
    protocol_md_path = amendment_root / "zero_perturbation_analysis_protocol_v1_1_r1.md"
    clarification_path = amendment_root / "zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1.json"
    clarification_md_path = amendment_root / "zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1.md"
    activation_record_path = amendment_root / "amendment_activation_record_v1_1_r1.json"
    clarification_transition_path = amendment_root / "analysis_missingness_clarification_transition_v1_1_r1_c1.json"
    active_pointer_path = root / "experiments/mid360_formal_batch1/ACTIVE_PROTOCOL.json"
    supersession_path = root / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.SUPERSEDED.json"
    evidence_root = root / "results/mid360_formal_batch1/zero_perturbation_v1_1_lock"
    candidate_report_path = evidence_root / "protocol_r1_candidate_independent_verification.json"
    prelock_path = evidence_root / "final_dataset_prelock_reauthentication.json"
    candidate_history_root = amendment_root / "history/zero_perturbation_v1_1_r1_candidate_verified"
    candidate_inventory_path = candidate_history_root / "candidate_verified_inventory.json"
    initial_history_root = amendment_root / "history/zero_perturbation_v1_1_r1_active_pre_missingness_clarification"
    prior_inventory_path = initial_history_root / "active_preclarification_inventory.json"

    amendment = _load_json(amendment_path)
    contract = _load_json(contract_path)
    supersession = _load_json(supersession_path)
    clarification = _load_json(clarification_path)
    validate_active_payloads(amendment, contract, supersession)
    validate_missingness_clarification_payload(clarification)

    current_active_hashes = {
        "amendment_json_sha256": _sha256(amendment_path),
        "amendment_md_sha256": _sha256(amendment_md_path),
        "analysis_contract_sha256": _sha256(contract_path),
        "analysis_protocol_sha256": _sha256(protocol_md_path),
    }
    clarification_json_sha = _sha256(clarification_path)
    clarification_md_sha = _sha256(clarification_md_path)
    candidate_report_sha = _sha256(candidate_report_path)
    prelock_sha = _sha256(prelock_path)
    candidate_inventory_sha = _sha256(candidate_inventory_path)
    prior_inventory_sha = _sha256(prior_inventory_path)
    activation_record_sha = _sha256(activation_record_path)
    active_pointer_sha = _sha256(active_pointer_path)
    clarification_transition_sha = _sha256(clarification_transition_path)

    candidate_report = _load_json(candidate_report_path)
    prelock = _load_json(prelock_path)
    candidate_inventory = _load_json(candidate_inventory_path)
    prior_inventory = _load_json(prior_inventory_path)

    candidate_archived_paths = {
        "amendment_json_sha256": candidate_history_root / "zero_perturbation_mainline_v1_1_r1.json",
        "amendment_md_sha256": candidate_history_root / "zero_perturbation_mainline_v1_1_r1.md",
        "analysis_contract_sha256": candidate_history_root / "zero_perturbation_analysis_contract_v1_1_r1.json",
        "analysis_protocol_sha256": candidate_history_root / "zero_perturbation_analysis_protocol_v1_1_r1.md",
    }
    for key, path in candidate_archived_paths.items():
        _require(
            _sha256(path) == EXPECTED_CANDIDATE_HASHES[key],
            f"archived candidate {key} bytes mismatch",
        )

    initial_archived_paths = {
        "amendment_json_sha256": initial_history_root / "zero_perturbation_mainline_v1_1_r1.json",
        "amendment_md_sha256": initial_history_root / "zero_perturbation_mainline_v1_1_r1.md",
        "analysis_contract_sha256": initial_history_root / "zero_perturbation_analysis_contract_v1_1_r1.json",
        "analysis_protocol_sha256": initial_history_root / "zero_perturbation_analysis_protocol_v1_1_r1.md",
    }
    for key, path in initial_archived_paths.items():
        _require(
            _sha256(path) == EXPECTED_INITIAL_ACTIVE_HASHES[key],
            f"archived initial ACTIVE {key} bytes mismatch",
        )
    initial_record_path = initial_history_root / "amendment_activation_record_v1_1_r1.json"
    initial_pointer_path = initial_history_root / "ACTIVE_PROTOCOL.json"
    initial_transition_path = initial_history_root / "amendment_activation_transition_v1_1_r1.json"
    _require(
        _sha256(initial_record_path) == EXPECTED_INITIAL_ACTIVATION_RECORD_SHA256,
        "archived initial activation record bytes mismatch",
    )
    _require(
        _sha256(initial_pointer_path) == EXPECTED_INITIAL_ACTIVE_POINTER_SHA256,
        "archived initial ACTIVE_PROTOCOL bytes mismatch",
    )
    _require(
        _sha256(initial_transition_path) == EXPECTED_INITIAL_ACTIVATION_TRANSITION_SHA256,
        "archived initial activation transition bytes mismatch",
    )
    validate_active_transition_payloads(
        active_hashes=EXPECTED_INITIAL_ACTIVE_HASHES,
        activation_record=_load_json(initial_record_path),
        active_pointer=_load_json(initial_pointer_path),
        transition=_load_json(initial_transition_path),
        candidate_report=candidate_report,
        candidate_report_sha256=candidate_report_sha,
        candidate_inventory=candidate_inventory,
        candidate_inventory_sha256=candidate_inventory_sha,
        prelock=prelock,
        prelock_sha256=prelock_sha,
        activation_record_sha256=EXPECTED_INITIAL_ACTIVATION_RECORD_SHA256,
        active_pointer_sha256=EXPECTED_INITIAL_ACTIVE_POINTER_SHA256,
    )

    activation_record = _load_json(activation_record_path)
    active_pointer = _load_json(active_pointer_path)
    clarification_transition = _load_json(clarification_transition_path)
    validate_missingness_clarification_transition_payloads(
        initial_active_hashes=EXPECTED_INITIAL_ACTIVE_HASHES,
        current_active_hashes=current_active_hashes,
        prior_inventory=prior_inventory,
        prior_inventory_sha256=prior_inventory_sha,
        clarification=clarification,
        clarification_json_sha256=clarification_json_sha,
        clarification_markdown_sha256=clarification_md_sha,
        activation_record=activation_record,
        activation_record_sha256=activation_record_sha,
        active_pointer=active_pointer,
        active_pointer_sha256=active_pointer_sha,
        clarification_transition=clarification_transition,
    )
    common_status_artifacts = validate_common_association_status_artifacts(root)

    immutable_files = {
        root / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.json": OLD_PROPOSAL_JSON_SHA256,
        root / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.md": OLD_PROPOSAL_MD_SHA256,
        root / "experiments/mid360_formal_batch1/preregistration.yaml": ORIGINAL_PREREGISTRATION_SHA256,
        root / "experiments/mid360_formal_batch1/analysis_protocol.md": ORIGINAL_ANALYSIS_PROTOCOL_SHA256,
        root / "frozen_assets/backend_parameter_contract.json": BACKEND_CONTRACT_SHA256,
        root / "src/phase_a_harness/common_association_analysis.py": COMMON_ASSOCIATION_SHA256,
    }
    for path, expected in immutable_files.items():
        _require(_sha256(path) == expected, f"immutable protocol asset changed: {path}")

    no_icp = _load_json(
        root / "results/mid360_formal_batch1/final_dataset_v1/NO_ICP_ATTESTATION.json"
    )
    _require(no_icp.get("NO_ICP_ATTESTATION_PASS") is True, "NO-ICP attestation failed")
    _require(no_icp.get("actual_formal_trials") == 0, "formal trials exist")
    _require(no_icp.get("open3d_registration_call_count") == 0, "Open3D call exists")
    _require(no_icp.get("pcl_cli_invocation_count") == 0, "PCL call exists")

    review = _load_json(evidence_root / "zero_perturbation_v1_1_activation_review.json")
    review_status = review.get("review_status")
    _require(
        review_status
        in {
            "PASS_ACTIVE_R1_C1_PENDING_CLARIFICATION_VERIFIER",
            "PASS_ACTIVE_R1_C1",
        },
        "canonical activation review C1 status mismatch",
    )
    _require(review.get("activation_effective") is True, "activation review is ineffective")
    _require(review.get("independent_activation_verifier_pass") is True, "review lacks candidate verifier pass")
    _require(review.get("active_transition_verifier_pass") is True, "review lacks initial transition pass")
    review_active = _require_mapping(review.get("active_assets"), "review active assets")
    for key, expected in current_active_hashes.items():
        _require(review_active.get(key) == expected, f"review active {key} mismatch")
    _require(review_active.get("activation_record_sha256") == activation_record_sha, "review activation record mismatch")
    _require(review_active.get("active_protocol_pointer_sha256") == active_pointer_sha, "review pointer mismatch")
    review_c1 = _require_mapping(
        review.get("prelock_missingness_clarification"), "review C1 clarification"
    )
    _require(review_c1.get("json_sha256") == clarification_json_sha, "review C1 JSON mismatch")
    _require(review_c1.get("markdown_sha256") == clarification_md_sha, "review C1 Markdown mismatch")
    _require(review_c1.get("prior_active_inventory_sha256") == prior_inventory_sha, "review C1 history mismatch")
    _require(review_c1.get("clarification_transition_sha256") == clarification_transition_sha, "review C1 transition mismatch")
    if review_status == "PASS_ACTIVE_R1_C1":
        _require(
            review.get("missingness_clarification_verifier_pass") is True,
            "completed review lacks C1 verifier pass",
        )
        review_verifier = _require_mapping(
            review.get("missingness_clarification_independent_verification"),
            "review C1 verifier",
        )
        expected_report_relative = (
            "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/"
            "protocol_r1_c1_missingness_independent_verification.json"
        )
        _require(
            review_verifier.get("path") == expected_report_relative
            and review_verifier.get("status") == "PASS",
            "review C1 verifier path/status mismatch",
        )
        reviewed_report_path = root / expected_report_relative
        _require(
            review_verifier.get("sha256") == _sha256(reviewed_report_path),
            "review C1 verifier SHA mismatch",
        )
        reviewed_report = _load_json(reviewed_report_path)
        _require(
            reviewed_report.get("pass") is True
            and reviewed_report.get("verification_status") == "PASS"
            and reviewed_report.get("phase") == "ACTIVE_R1_C1_PRE_LOCK"
            and reviewed_report.get("actual_formal_trials") == 0,
            "review-bound C1 verifier report is not a zero-trial PASS",
        )
    else:
        _require(
            review.get("missingness_clarification_verifier_pass") is False,
            "pending review prematurely claims C1 verifier pass",
        )
    _require(review.get("FORMAL_LOCK_ISSUED") is False, "review prematurely issues lock")
    _require(review.get("FORMAL_ICP_UNLOCKED") is False, "review prematurely unlocks ICP")
    _require(review.get("FORMAL_REGISTRATION_AUTHORIZED") is False, "review authorizes registration")
    _require(review.get("actual_formal_trials") == 0, "review has formal trials")

    return {
        "schema": "mid360_fmb1_zero_perturbation_protocol_r1_c1_missingness_independent_verification",
        "verification_status": "PASS",
        "pass": True,
        "phase": "ACTIVE_R1_C1_PRE_LOCK",
        "amendment_id": AMENDMENT_ID,
        "clarification_id": CLARIFICATION_ID,
        "amendment_status": ACTIVE_STATUS,
        "activation_effective": True,
        "candidate_hashes_preserved": copy.deepcopy(EXPECTED_CANDIDATE_HASHES),
        "initial_active_hashes_preserved": copy.deepcopy(EXPECTED_INITIAL_ACTIVE_HASHES),
        "current_active_hashes": current_active_hashes,
        "clarification_json_sha256": clarification_json_sha,
        "clarification_markdown_sha256": clarification_md_sha,
        "candidate_inventory_sha256": candidate_inventory_sha,
        "candidate_verification_sha256": candidate_report_sha,
        "prelock_reauthentication_sha256": prelock_sha,
        "prior_active_inventory_sha256": prior_inventory_sha,
        "initial_activation_record_sha256": EXPECTED_INITIAL_ACTIVATION_RECORD_SHA256,
        "initial_active_protocol_pointer_sha256": EXPECTED_INITIAL_ACTIVE_POINTER_SHA256,
        "initial_activation_transition_sha256": EXPECTED_INITIAL_ACTIVATION_TRANSITION_SHA256,
        "activation_record_sha256": activation_record_sha,
        "active_protocol_pointer_sha256": active_pointer_sha,
        "clarification_transition_sha256": clarification_transition_sha,
        "common_association_status_artifacts": common_status_artifacts,
        "common_association_status_fields_verified": True,
        "old_proposal_json_sha256_before_and_after": OLD_PROPOSAL_JSON_SHA256,
        "old_proposal_md_sha256_before_and_after": OLD_PROPOSAL_MD_SHA256,
        "physical_reference_semantics": PHYSICAL_REFERENCE,
        "capture_radius_track_status": "PRESERVED_SUPPLEMENTARY_NOT_EXECUTED",
        "model_transfer_status": "MODEL_TRANSFER_NOT_COMPATIBLE",
        "backend_modules_imported": 0,
        "backend_calls": 0,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
        "next_gate": "FORMAL_LOCK_QUALIFICATION_WITHOUT_BACKEND_EXECUTION",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report")
    parser.add_argument("--phase", choices=("candidate", "active"), default="active")
    args = parser.parse_args(argv)
    try:
        report = (
            verify_candidate(args.repo_root)
            if args.phase == "candidate"
            else verify_active_transition(args.repo_root)
        )
    except ProtocolR1VerificationError as error:
        report = {
            "schema": "mid360_fmb1_zero_perturbation_protocol_r1_candidate_independent_verification",
            "verification_status": "FAIL",
            "pass": False,
            "error": str(error),
            "backend_modules_imported": 0,
            "backend_calls": 0,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "actual_formal_trials": 0,
        }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        Path(args.report).write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
