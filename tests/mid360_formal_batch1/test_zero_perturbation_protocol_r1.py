from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.zero_perturbation_protocol_r1_verify import (
    BACKEND_CONTRACT_SHA256,
    CLARIFICATION_ID,
    COMMON_ASSOCIATION_SHA256,
    EXPECTED_CANDIDATE_HASHES,
    EXPECTED_INITIAL_ACTIVATION_RECORD_SHA256,
    EXPECTED_INITIAL_ACTIVE_HASHES,
    EXPECTED_INITIAL_ACTIVE_POINTER_SHA256,
    OLD_PROPOSAL_JSON_SHA256,
    OLD_PROPOSAL_MD_SHA256,
    PHYSICAL_REFERENCE,
    ProtocolR1VerificationError,
    validate_active_payloads,
    validate_active_transition_payloads,
    validate_candidate_payloads,
    validate_common_association_status_artifacts,
    validate_missingness_clarification_payload,
    validate_missingness_clarification_transition_payloads,
    verify_active_transition,
)


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT_PATH = (
    ROOT
    / "experiments/mid360_formal_batch1/amendments/zero_perturbation_mainline_v1_1_r1.json"
)
CONTRACT_PATH = (
    ROOT
    / "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_contract_v1_1_r1.json"
)
SUPERSESSION_PATH = (
    ROOT
    / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.SUPERSEDED.json"
)
HISTORY_ROOT = (
    ROOT
    / "experiments/mid360_formal_batch1/amendments/history/zero_perturbation_v1_1_r1_candidate_verified"
)
INITIAL_ACTIVE_HISTORY_ROOT = (
    ROOT
    / "experiments/mid360_formal_batch1/amendments/history/zero_perturbation_v1_1_r1_active_pre_missingness_clarification"
)
CLARIFICATION_PATH = (
    ROOT
    / "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1.json"
)
EVIDENCE_ROOT = ROOT / "results/mid360_formal_batch1/zero_perturbation_v1_1_lock"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payloads():
    return _json(AMENDMENT_PATH), _json(CONTRACT_PATH), _json(SUPERSESSION_PATH)


def _reject(amendment, contract, supersession) -> None:
    with pytest.raises(ProtocolR1VerificationError):
        validate_active_payloads(amendment, contract, supersession)


def _transition_payloads():
    amendment_root = ROOT / "experiments/mid360_formal_batch1/amendments"
    active_pointer_path = INITIAL_ACTIVE_HISTORY_ROOT / "ACTIVE_PROTOCOL.json"
    activation_record_path = (
        INITIAL_ACTIVE_HISTORY_ROOT / "amendment_activation_record_v1_1_r1.json"
    )
    transition_path = (
        INITIAL_ACTIVE_HISTORY_ROOT / "amendment_activation_transition_v1_1_r1.json"
    )
    candidate_report_path = EVIDENCE_ROOT / "protocol_r1_candidate_independent_verification.json"
    inventory_path = HISTORY_ROOT / "candidate_verified_inventory.json"
    prelock_path = EVIDENCE_ROOT / "final_dataset_prelock_reauthentication.json"
    return {
        "active_hashes": copy.deepcopy(EXPECTED_INITIAL_ACTIVE_HASHES),
        "activation_record": _json(activation_record_path),
        "active_pointer": _json(active_pointer_path),
        "transition": _json(transition_path),
        "candidate_report": _json(candidate_report_path),
        "candidate_report_sha256": _sha(candidate_report_path),
        "candidate_inventory": _json(inventory_path),
        "candidate_inventory_sha256": _sha(inventory_path),
        "prelock": _json(prelock_path),
        "prelock_sha256": _sha(prelock_path),
        "activation_record_sha256": EXPECTED_INITIAL_ACTIVATION_RECORD_SHA256,
        "active_pointer_sha256": EXPECTED_INITIAL_ACTIVE_POINTER_SHA256,
    }


def _reject_transition(mutator) -> None:
    payloads = _transition_payloads()
    mutator(payloads)
    with pytest.raises(ProtocolR1VerificationError):
        validate_active_transition_payloads(**payloads)


def _clarification_payloads():
    amendment_root = ROOT / "experiments/mid360_formal_batch1/amendments"
    amendment_md_path = amendment_root / "zero_perturbation_mainline_v1_1_r1.md"
    contract_path = amendment_root / "zero_perturbation_analysis_contract_v1_1_r1.json"
    protocol_path = amendment_root / "zero_perturbation_analysis_protocol_v1_1_r1.md"
    clarification_md_path = amendment_root / "zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1.md"
    inventory_path = INITIAL_ACTIVE_HISTORY_ROOT / "active_preclarification_inventory.json"
    activation_record_path = amendment_root / "amendment_activation_record_v1_1_r1.json"
    pointer_path = ROOT / "experiments/mid360_formal_batch1/ACTIVE_PROTOCOL.json"
    transition_path = amendment_root / "analysis_missingness_clarification_transition_v1_1_r1_c1.json"
    return {
        "initial_active_hashes": copy.deepcopy(EXPECTED_INITIAL_ACTIVE_HASHES),
        "current_active_hashes": {
            "amendment_json_sha256": _sha(AMENDMENT_PATH),
            "amendment_md_sha256": _sha(amendment_md_path),
            "analysis_contract_sha256": _sha(contract_path),
            "analysis_protocol_sha256": _sha(protocol_path),
        },
        "prior_inventory": _json(inventory_path),
        "prior_inventory_sha256": _sha(inventory_path),
        "clarification": _json(CLARIFICATION_PATH),
        "clarification_json_sha256": _sha(CLARIFICATION_PATH),
        "clarification_markdown_sha256": _sha(clarification_md_path),
        "activation_record": _json(activation_record_path),
        "activation_record_sha256": _sha(activation_record_path),
        "active_pointer": _json(pointer_path),
        "active_pointer_sha256": _sha(pointer_path),
        "clarification_transition": _json(transition_path),
    }


def _reject_clarification_transition(mutator) -> None:
    payloads = _clarification_payloads()
    mutator(payloads)
    with pytest.raises(ProtocolR1VerificationError):
        validate_missingness_clarification_transition_payloads(**payloads)


def test_active_transition_independent_verifier_passes_without_backend_authority():
    report = verify_active_transition(ROOT)
    assert report["pass"] is True
    assert report["phase"] == "ACTIVE_R1_C1_PRE_LOCK"
    assert report["clarification_id"] == CLARIFICATION_ID
    assert report["amendment_status"] == "ACTIVE"
    assert report["activation_effective"] is True
    assert report["candidate_hashes_preserved"] == EXPECTED_CANDIDATE_HASHES
    assert report["initial_active_hashes_preserved"] == EXPECTED_INITIAL_ACTIVE_HASHES
    assert report["common_association_status_fields_verified"] is True
    assert report["backend_modules_imported"] == 0
    assert report["backend_calls"] == 0
    assert report["FORMAL_REGISTRATION_AUTHORIZED"] is False
    assert report["actual_formal_trials"] == 0


def test_historical_proposal_and_frozen_authorities_are_byte_preserved():
    assert _sha(
        ROOT
        / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.json"
    ) == OLD_PROPOSAL_JSON_SHA256
    assert _sha(
        ROOT
        / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.md"
    ) == OLD_PROPOSAL_MD_SHA256
    assert _sha(ROOT / "frozen_assets/backend_parameter_contract.json") == BACKEND_CONTRACT_SHA256
    assert _sha(ROOT / "src/phase_a_harness/common_association_analysis.py") == COMMON_ASSOCIATION_SHA256


def test_candidate_assets_have_no_backend_import():
    verifier_path = (
        ROOT
        / "experiments/mid360_formal_batch1/zero_perturbation_protocol_r1_verify.py"
    )
    tree = ast.parse(verifier_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert "open3d" not in imported
    assert "pcl" not in imported


def test_candidate_bytes_are_archived_exactly_and_still_validate_as_candidate():
    candidate_amendment = _json(HISTORY_ROOT / "zero_perturbation_mainline_v1_1_r1.json")
    candidate_contract = _json(
        HISTORY_ROOT / "zero_perturbation_analysis_contract_v1_1_r1.json"
    )
    supersession = _json(SUPERSESSION_PATH)
    validate_candidate_payloads(candidate_amendment, candidate_contract, supersession)
    assert _sha(HISTORY_ROOT / "zero_perturbation_mainline_v1_1_r1.json") == EXPECTED_CANDIDATE_HASHES["amendment_json_sha256"]
    assert _sha(HISTORY_ROOT / "zero_perturbation_mainline_v1_1_r1.md") == EXPECTED_CANDIDATE_HASHES["amendment_md_sha256"]
    assert _sha(HISTORY_ROOT / "zero_perturbation_analysis_contract_v1_1_r1.json") == EXPECTED_CANDIDATE_HASHES["analysis_contract_sha256"]
    assert _sha(HISTORY_ROOT / "zero_perturbation_analysis_protocol_v1_1_r1.md") == EXPECTED_CANDIDATE_HASHES["analysis_protocol_sha256"]


def test_canonical_protocol_is_active_but_execution_flags_remain_false():
    amendment, contract, _ = _payloads()
    assert amendment["status"] == "ACTIVE"
    assert amendment["activation_effective"] is True
    assert contract["status"] == "ACTIVE"
    assert contract["activation_effective"] is True
    pointer = _json(ROOT / "experiments/mid360_formal_batch1/ACTIVE_PROTOCOL.json")
    record = _json(
        ROOT
        / "experiments/mid360_formal_batch1/amendments/amendment_activation_record_v1_1_r1.json"
    )
    assert pointer["status"] == "ACTIVE"
    assert record["status"] == "ACTIVE"
    for payload in (amendment, contract, pointer, record):
        assert payload["FORMAL_REGISTRATION_AUTHORIZED"] is False
        assert payload["actual_formal_trials"] == 0


def test_physical_limitation_and_capture_radius_preservation_are_in_both_markdown_files():
    for name in (
        "zero_perturbation_mainline_v1_1_r1.md",
        "zero_perturbation_analysis_protocol_v1_1_r1.md",
    ):
        text = (ROOT / "experiments/mid360_formal_batch1/amendments" / name).read_text(
            encoding="utf-8"
        )
        assert PHYSICAL_REFERENCE in text
        assert "PRESERVED_SUPPLEMENTARY_NOT_EXECUTED" in text
        assert "submillimeter" in text
        assert "W02 attempt 1" in text
        assert "W02 attempt 2" in text
        assert "W04" in text


def test_tamper_candidate_status_or_authority_is_rejected():
    amendment, contract, supersession = _payloads()
    amendment["status"] = "CANDIDATE_PENDING_INDEPENDENT_ACTIVATION_VERIFIER"
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    amendment["FORMAL_REGISTRATION_AUTHORIZED"] = True
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    amendment["actual_formal_trials"] = 1
    _reject(amendment, contract, supersession)


def test_tamper_w02_lineage_or_w04_exclusion_is_rejected():
    amendment, contract, supersession = _payloads()
    amendment["correction"]["w02_attempt1"]["status"] = "GEOMETRY_REJECTED"
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    amendment["correction"]["w02_attempt2"]["final_geometry_class"] = "RICH"
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    amendment["correction"]["w04_included_in_final_dataset"] = True
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    contract["dataset"]["weak_scene_ids"][-1] = "FMB1_W04"
    _reject(amendment, contract, supersession)


def test_tamper_identity_plan_or_physical_reference_is_rejected():
    amendment, contract, supersession = _payloads()
    amendment["zero_perturbation_track"]["T0"][0][3] = 0.01
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    amendment["zero_perturbation_track"]["planned_total_trials"] = 359
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    amendment["physical_reference"]["physical_reference_semantics"] = (
        "SUBMILLIMETER_GROUND_TRUTH"
    )
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    contract["physical_reference"]["independent_submillimeter_external_ground_truth_available"] = True
    _reject(amendment, contract, supersession)


def test_tamper_scene_hierarchy_or_inference_is_rejected():
    amendment, contract, supersession = _payloads()
    amendment["experimental_hierarchy"]["snapshots_are_independent_scenes"] = True
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    contract["experimental_units"]["primary_experimental_unit"] = "snapshot"
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    contract["primary_endpoint"]["inference"]["allocation_count"] = 19
    _reject(amendment, contract, supersession)


def test_tamper_common_reassociation_or_systematic_rule_is_rejected():
    amendment, contract, supersession = _payloads()
    contract["reassociation_analysis"]["implementation_sha256_at_candidate_freeze"] = "0" * 64
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    contract["reassociation_analysis"]["optional_stratified_permutation"][
        "preserve_scene_strata"
    ] = False
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    contract["systematic_component"]["zero_denominator_rule"]["station_retained"] = False
    _reject(amendment, contract, supersession)


def test_tamper_model_transfer_or_capture_radius_preservation_is_rejected():
    amendment, contract, supersession = _payloads()
    contract["synthetic_model_transfer"]["status"] = "COMPATIBLE"
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    contract["synthetic_model_transfer"]["retraining_permitted"] = True
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    contract["capture_radius_track"]["changed_by_r1"] = True
    _reject(amendment, contract, supersession)


def test_tamper_supersession_record_is_rejected():
    amendment, contract, supersession = _payloads()
    supersession["status"] = "ACTIVE"
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    supersession["historical_proposal_modified"] = True
    _reject(amendment, contract, supersession)

    amendment, contract, supersession = _payloads()
    supersession["historical_proposal_json"]["sha256"] = "0" * 64
    _reject(amendment, contract, supersession)


def test_active_transition_payload_graph_passes_independently():
    validate_active_transition_payloads(**_transition_payloads())


def test_tamper_candidate_before_hash_or_archive_inventory_is_rejected():
    _reject_transition(
        lambda payloads: payloads["transition"]["before"].__setitem__(
            "amendment_json_sha256", "0" * 64
        )
    )
    _reject_transition(
        lambda payloads: payloads["candidate_inventory"]["files"][0].__setitem__(
            "sha256", "0" * 64
        )
    )


def test_tamper_activation_record_or_active_pointer_binding_is_rejected():
    _reject_transition(
        lambda payloads: payloads["activation_record"].__setitem__(
            "formal_trial_count_at_activation", 1
        )
    )
    _reject_transition(
        lambda payloads: payloads["active_pointer"]["active_amendment"].__setitem__(
            "sha256", "0" * 64
        )
    )
    _reject_transition(
        lambda payloads: payloads["transition"]["after"].__setitem__(
            "active_protocol_pointer_sha256", "0" * 64
        )
    )


def test_tamper_prelock_zero_trial_or_execution_authority_is_rejected():
    _reject_transition(
        lambda payloads: payloads["prelock"].__setitem__("actual_formal_trials", 1)
    )
    _reject_transition(
        lambda payloads: payloads["active_pointer"].__setitem__(
            "FORMAL_REGISTRATION_AUTHORIZED", True
        )
    )
    _reject_transition(
        lambda payloads: payloads["activation_record"].__setitem__(
            "activation_grants_backend_execution", True
        )
    )


def test_c1_missingness_payload_and_transition_graph_pass_independently():
    validate_missingness_clarification_payload(_json(CLARIFICATION_PATH))
    validate_missingness_clarification_transition_payloads(**_clarification_payloads())
    hashes = validate_common_association_status_artifacts(ROOT)
    assert set(hashes) == {
        "result_schema_sha256",
        "runner_sha256",
        "execution_verifier_sha256",
        "result_validator_sha256",
        "experiments_package_init_sha256",
        "mid360_formal_batch1_package_init_sha256",
        "phase_a_harness_package_init_sha256",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("formal_translation_rotation_summary_rule", "station", "required_finite_endpoint_n"), 9),
        (("formal_translation_rotation_summary_rule", "scene", "required_finite_endpoint_n"), 29),
        (("weak_rich_exact_permutation_rule", "required_defined_scene_summaries"), 5),
        (("spearman_rules", "scene_backend_agreement", "required_complete_pairs"), 5),
        (("spearman_rules", "station_backend_agreement", "required_complete_pairs"), 17),
        (("systematic_fraction_missingness_rule", "required_finite_translation_vectors_per_station"), 9),
    ],
)
def test_tamper_c1_complete_coverage_gates_is_rejected(path, value):
    clarification = _json(CLARIFICATION_PATH)
    target = clarification
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ProtocolR1VerificationError):
        validate_missingness_clarification_payload(clarification)


def test_tamper_c1_available_case_or_infrastructure_accounting_is_rejected():
    clarification = _json(CLARIFICATION_PATH)
    clarification["available_case_descriptive_rule"]["may_replace_formal_summary"] = True
    with pytest.raises(ProtocolR1VerificationError):
        validate_missingness_clarification_payload(clarification)

    clarification = _json(CLARIFICATION_PATH)
    clarification["infrastructure_failure_rule"][
        "unresolved_failure_trial_retained_in_planned_denominator"
    ] = False
    with pytest.raises(ProtocolR1VerificationError):
        validate_missingness_clarification_payload(clarification)

    clarification = _json(CLARIFICATION_PATH)
    clarification["principles"]["silent_row_deletion_forbidden"] = False
    with pytest.raises(ProtocolR1VerificationError):
        validate_missingness_clarification_payload(clarification)


def test_tamper_c1_exact_infrastructure_success_literal_is_rejected():
    clarification = _json(CLARIFICATION_PATH)
    clarification["row_endpoint_classification"]["FINITE_SCIENTIFIC_VALUE"] = (
        "infrastructure_status PASS"
    )
    with pytest.raises(ProtocolR1VerificationError):
        validate_missingness_clarification_payload(clarification)


def test_tamper_c1_common_status_fields_or_reason_enum_is_rejected():
    clarification = _json(CLARIFICATION_PATH)
    clarification["reassociation_missingness_rule"]["required_result_status_fields"].pop()
    with pytest.raises(ProtocolR1VerificationError):
        validate_missingness_clarification_payload(clarification)

    clarification = _json(CLARIFICATION_PATH)
    clarification["reassociation_missingness_rule"][
        "common_association_invalid_reason_enum"
    ][0] = "UNFROZEN_REASON"
    with pytest.raises(ProtocolR1VerificationError):
        validate_missingness_clarification_payload(clarification)


def test_tamper_c1_timing_or_authority_is_rejected():
    clarification = _json(CLARIFICATION_PATH)
    clarification["clarification_at_formal_trial_count"] = 1
    with pytest.raises(ProtocolR1VerificationError):
        validate_missingness_clarification_payload(clarification)

    clarification = _json(CLARIFICATION_PATH)
    clarification["FORMAL_LOCK_ISSUED"] = True
    with pytest.raises(ProtocolR1VerificationError):
        validate_missingness_clarification_payload(clarification)


def test_tamper_pre_c1_history_or_transition_before_hash_is_rejected():
    _reject_clarification_transition(
        lambda payloads: payloads["prior_inventory"]["files"][0].__setitem__(
            "sha256", "0" * 64
        )
    )
    _reject_clarification_transition(
        lambda payloads: payloads["clarification_transition"]["before"].__setitem__(
            "analysis_contract_sha256", "0" * 64
        )
    )


def test_tamper_c1_current_record_pointer_or_after_hash_is_rejected():
    _reject_clarification_transition(
        lambda payloads: payloads["activation_record"]["active_state"].__setitem__(
            "analysis_contract_sha256", "0" * 64
        )
    )
    _reject_clarification_transition(
        lambda payloads: payloads["active_pointer"]["active_prelock_scientific_clarification"].__setitem__(
            "sha256", "0" * 64
        )
    )
    _reject_clarification_transition(
        lambda payloads: payloads["clarification_transition"]["after"].__setitem__(
            "clarification_json_sha256", "0" * 64
        )
    )
