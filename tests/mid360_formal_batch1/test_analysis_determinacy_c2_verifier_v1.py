from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.analysis_determinacy_c2_verify_v1 import (
    BASIS_PATH,
    C1_CONTRACT_PATH,
    C2_PATH,
    C2VerificationError,
    EXPECTED_PROTECTED_SHA256,
    HISTORICAL_AUDIT_PATH,
    POSTRUN_REPORT_PATH,
    sha256_file,
    validate_c2_payloads,
    validate_reaudit_payload,
    verify_repository,
)


ROOT = Path(__file__).resolve().parents[2]


def load(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def payloads():
    return (
        load(C2_PATH),
        load(BASIS_PATH),
        load(C1_CONTRACT_PATH),
        load(HISTORICAL_AUDIT_PATH),
        load(POSTRUN_REPORT_PATH),
    )


def reject(c2, basis, c1, historical, postrun):
    with pytest.raises(C2VerificationError):
        validate_c2_payloads(c2, basis, c1, historical, postrun)


def test_protected_c1_and_historical_audit_bytes_are_unchanged():
    for relative_path, expected_sha in EXPECTED_PROTECTED_SHA256.items():
        assert sha256_file(ROOT / relative_path) == expected_sha


def test_canonical_c2_repository_independent_verifier_passes_23_checks():
    report = verify_repository(ROOT)
    assert report["status"] == "PASS"
    assert report["FMB1_ANALYSIS_DETERMINACY_C2_ACTIVE"] is True
    assert report["semantic_check_count"] == 23
    assert "POST_C2_IMPLEMENTATION_AND_LOCK_NOT_TRAVERSED" in report["semantic_checks"]
    assert report["protected_source_mismatch_count"] == 0
    assert report["formal_trial_result_files_read"] == 0
    assert report["scientific_aggregation_executed"] is False
    assert report["formal_p_value_computed"] is False


def test_c2_is_postrun_preanalysis_and_never_preregistration():
    c2, *_ = payloads()
    assert c2["clarification_phase"] == "BLINDED_POSTRUN_PRE_LOCKED_SCIENTIFIC_ANALYSIS"
    assert c2["formal_registration_complete"] is True
    assert c2["formal_trial_count"] == 360
    assert c2["clarification_trigger"] == "PREEXISTING_CONTRACT_DETERMINACY_AUDIT"


@pytest.mark.parametrize(
    ("mutator"),
    [
        lambda c2: c2["scope"].__setitem__("PRIMARY_TRANSLATION_ANALYSIS_CHANGED", True),
        lambda c2: c2["root_definitions"]["scene_ordering_agreement"].__setitem__(
            "role", "INFERENTIAL"
        ),
        lambda c2: c2["root_definitions"]["formal_reassociation_turnover"].__setitem__(
            "formal_primary_field", "accepted_source_turnover"
        ),
        lambda c2: c2["root_definitions"]["within_scene_centered_association"].__setitem__(
            "center_statistic", "MEAN"
        ),
        lambda c2: c2["root_definitions"][
            "registered_stratified_sensitivity_permutation"
        ].__setitem__("seed", 1),
        lambda c2: c2["root_definitions"][
            "registered_stratified_sensitivity_permutation"
        ].__setitem__("p_value_denominator", 10000),
        lambda c2: c2["root_definitions"]["systematic_weak_rich_comparison"].__setitem__(
            "formal_hypothesis_test", True
        ),
        lambda c2: c2["execution_authority"].__setitem__(
            "real_formal_result_files_read_by_c2_tooling", 1
        ),
    ],
)
def test_c2_semantic_tampering_fails_closed(mutator):
    c2, basis, c1, historical, postrun = copy.deepcopy(payloads())
    mutator(c2)
    reject(c2, basis, c1, historical, postrun)


def test_basis_tamper_fails_closed():
    c2, basis, c1, historical, postrun = copy.deepcopy(payloads())
    basis["definition_selected_from_result_values"] = True
    reject(c2, basis, c1, historical, postrun)


def test_source_audit_tamper_fails_closed():
    c2, basis, c1, historical, postrun = copy.deepcopy(payloads())
    historical["gate"]["unresolved_root_definition_count"] = 4
    reject(c2, basis, c1, historical, postrun)


def test_reaudit_validator_accepts_only_zero_unresolved_result_blind_payload():
    c2_sha = hashlib.sha256((ROOT / C2_PATH).read_bytes()).hexdigest()
    audit = {
        "status": "PASS_FULLY_DETERMINATE_C1_PLUS_C2",
        "source_bindings": {
            "c1_contract": {
                "sha256": EXPECTED_PROTECTED_SHA256[C1_CONTRACT_PATH]
            },
            "c2": {"sha256": c2_sha},
        },
        "determinacy_items": [
            {"id": index, "classification": "FULLY_DETERMINED"}
            for index in range(1, 19)
        ],
        "gate": {
            "required_under_specified_count": 0,
            "unresolved_root_definition_count": 0,
            "FMB1_LOCKED_ANALYSIS_CONTRACT_DETERMINATE": True,
            "READY_FOR_LOCKED_ANALYSIS_IMPLEMENTATION_AND_FREEZE": True,
            "READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS": False,
        },
        "audit_scope": {
            "formal_trial_result_files_opened": 0,
            "real_formal_scientific_values_read": 0,
            "real_scientific_aggregation_count": 0,
            "real_weak_rich_comparison_count": 0,
            "real_p_value_computation_count": 0,
        },
    }
    checks = validate_reaudit_payload(audit, c2_sha256=c2_sha)
    assert len(checks) == 5
    tampered = copy.deepcopy(audit)
    tampered["gate"]["required_under_specified_count"] = 1
    with pytest.raises(C2VerificationError):
        validate_reaudit_payload(tampered, c2_sha256=c2_sha)
