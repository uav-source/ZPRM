"""Independent, result-blind semantic verifier for analysis clarification C2."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


C2_ID = "FMB1_ZERO_PERTURBATION_ANALYSIS_DETERMINACY_CLARIFICATION_V1_1_R1_C2"
C2_STATUS = "ACTIVE_BLINDED_POSTRUN_PRE_ANALYSIS_CLARIFICATION"
C2_PHASE = "BLINDED_POSTRUN_PRE_LOCKED_SCIENTIFIC_ANALYSIS"
DETERMINACY_AUDIT_COMMIT = "21038ee1fd0b59b7edb5ab730d4683e4a9fe5170"
RAW_EXECUTION_COMMIT = "059e39533991d929a97ab208ad738643af82d09a"
POSTRUN_VERIFIER_CODE_COMMIT = "ec72d23f0f94cd91a84f2960b87672788266524f"
POSTRUN_VERIFICATION_COMMIT = "18bb94e62761f5193da8cdc5509c470cb4983244"
R3_FINGERPRINT = "fd601e8daf62c488a3a079beea05f65c9bd283399ae4d7f3acf16e63fc473a6d"

C1_CONTRACT_PATH = (
    "experiments/mid360_formal_batch1/amendments/"
    "zero_perturbation_analysis_contract_v1_1_r1.json"
)
C1_PROTOCOL_PATH = (
    "experiments/mid360_formal_batch1/amendments/"
    "zero_perturbation_analysis_protocol_v1_1_r1.md"
)
C1_MISSINGNESS_PATH = (
    "experiments/mid360_formal_batch1/amendments/"
    "zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1.json"
)
C2_PATH = (
    "experiments/mid360_formal_batch1/amendments/"
    "zero_perturbation_analysis_determinacy_clarification_v1_1_r1_c2.json"
)
C2_MD_PATH = (
    "experiments/mid360_formal_batch1/amendments/"
    "zero_perturbation_analysis_determinacy_clarification_v1_1_r1_c2.md"
)
HISTORICAL_AUDIT_PATH = (
    "results/mid360_formal_batch1/zero_perturbation_locked_analysis_preparation_v1/"
    "analysis_contract_determinacy_audit.json"
)
HISTORICAL_AUDIT_MD_PATH = (
    "results/mid360_formal_batch1/zero_perturbation_locked_analysis_preparation_v1/"
    "analysis_contract_determinacy_audit.md"
)
BASIS_PATH = (
    "results/mid360_formal_batch1/zero_perturbation_locked_analysis_preparation_v1/"
    "c2_clarification_basis.json"
)
POSTRUN_REPORT_PATH = (
    "results/mid360_formal_batch1/zero_perturbation_v1_1_postrun_verification_v1/"
    "postrun_independent_verification.json"
)

EXPECTED_PROTECTED_SHA256 = {
    C1_CONTRACT_PATH: "4120847bb471efbdbd32026ac23e959ab9ae4236709cf599b6e15701eb8b153d",
    C1_PROTOCOL_PATH: "9982193c3c4c2a6cda77c8db858fd73e73f61da7eb7ffe651610e5b1ed1c0b4c",
    C1_MISSINGNESS_PATH: "7b50dabef2352b666570657d519e98e9a5ae277da777be222c3483134fca69fb",
    HISTORICAL_AUDIT_PATH: "6153aadc46ef321770494b62321a19b75a896439407b7cc69daed7799cfe692d",
    HISTORICAL_AUDIT_MD_PATH: "7471580d1b68c486b4fa042dc3515fef6726181006061beaf111807c54818070",
    POSTRUN_REPORT_PATH: "8067cc8735fd5b2a849bce8dd3839a59b87cd7cac716009892be622903084f90",
    "frozen_assets/backend_parameter_contract.json": (
        "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
    ),
}

ROOT_KEYS = {
    "scene_ordering_agreement",
    "formal_reassociation_turnover",
    "within_scene_centered_association",
    "registered_stratified_sensitivity_permutation",
    "systematic_weak_rich_comparison",
}


class C2VerificationError(RuntimeError):
    """Raised when C2 or its result-blind evidence violates a frozen rule."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise C2VerificationError(f"expected JSON object: {path}")
    return payload


def _require(condition: bool, message: str, checks: list[str], check: str) -> None:
    if not condition:
        raise C2VerificationError(message)
    checks.append(check)


def validate_c2_payloads(
    c2: dict[str, Any],
    basis: dict[str, Any],
    c1_contract: dict[str, Any],
    historical_audit: dict[str, Any],
    postrun_report: dict[str, Any],
) -> list[str]:
    """Validate C2 semantics independently of any formal result rows."""

    checks: list[str] = []
    _require(
        c2.get("clarification_id") == C2_ID
        and c2.get("status") == C2_STATUS
        and c2.get("clarification_phase") == C2_PHASE,
        "C2 identity, status, or phase mismatch",
        checks,
        "C2_IDENTITY_AND_POSTRUN_PREANALYSIS_PHASE",
    )
    _require(
        c2.get("formal_registration_complete") is True
        and c2.get("formal_trial_count") == 360
        and c2.get("postrun_independent_verification_complete") is True,
        "C2 must truthfully follow all 360 formal registrations and Post-run verification",
        checks,
        "POSTRUN_TIMING_SEMANTICS",
    )
    _require(
        c2.get("scientific_aggregation_executed") is False
        and c2.get("weak_rich_comparison_executed") is False
        and c2.get("formal_p_value_computed") is False
        and c2.get("result_dependent_rule_selection") is False,
        "C2 result-blind counters or selection attestation mismatch",
        checks,
        "RESULT_BLIND_ZERO_SCIENTIFIC_COUNTERS",
    )
    access = c2.get("postrun_value_access_semantics", {})
    _require(
        access.get("postrun_integrity_verifier_machine_read_raw_trial_values") is True
        and access.get("c2_clarification_used_raw_trial_values_as_input") is False
        and access.get("c2_clarification_viewed_scene_aggregates") is False
        and access.get("c2_clarification_viewed_weak_rich_aggregates") is False
        and access.get("c2_definition_selected_from_effect_direction") is False
        and access.get("c2_definition_selected_from_actual_correlation") is False
        and access.get("c2_definition_selected_from_actual_p_value") is False,
        "C2 must distinguish prior machine integrity reading from result-blind rule selection",
        checks,
        "POSTRUN_MACHINE_READ_DISTINGUISHED_FROM_C2_INPUTS",
    )
    scope = c2.get("scope", {})
    _require(
        set(scope.get("allowed_changes", []))
        == {
            "SCENE_ORDERING_AGREEMENT_DEFINITION",
            "FORMAL_REASSOCIATION_TURNOVER_ENDPOINT",
            "WITHIN_SCENE_CENTERED_ASSOCIATION_STATISTIC",
            "REGISTERED_STRATIFIED_SENSITIVITY_PERMUTATION_STATISTIC",
            "SYSTEMATIC_WEAK_RICH_COMPARISON_ROLE_AND_ESTIMAND",
        }
        and scope.get("resolved_root_definition_count") == 5
        and all(
            scope.get(field) is False
            for field in (
                "PRIMARY_TRANSLATION_ANALYSIS_CHANGED",
                "PRIMARY_EXACT_PERMUTATION_CHANGED",
                "ROTATION_SECONDARY_ENDPOINT_CHANGED",
                "MISSINGNESS_C1_CHANGED",
                "DATASET_CHANGED",
                "BACKEND_PARAMETERS_CHANGED",
            )
        ),
        "C2 scope is not restricted to the five audited ambiguities",
        checks,
        "EXACT_FIVE_ROOT_SCOPE_AND_UNCHANGED_FLAGS",
    )
    roots = c2.get("root_definitions", {})
    _require(
        set(roots) == ROOT_KEYS,
        "C2 root-definition set must contain exactly five entries",
        checks,
        "EXACT_FIVE_ROOT_DEFINITIONS",
    )

    ordering = roots.get("scene_ordering_agreement", {})
    _require(
        ordering.get("definition")
        == "PAIRWISE_SIGN_ORDER_AGREEMENT_ACROSS_SIX_SCENES"
        and ordering.get("unordered_pair_count") == 15
        and ordering.get("role") == "DESCRIPTIVE_ONLY"
        and ordering.get("p_value") is None
        and ordering.get("replaces_scene_spearman") is False
        and ordering.get("outputs", {}).get("pair_count") == 15,
        "scene-ordering definition mismatch",
        checks,
        "SCENE_ORDERING_DESCRIPTIVE_PAIRWISE_SIGN_DEFINITION",
    )
    _require(
        set(ordering.get("classifications", {}))
        == {"CONCORDANT_NON_TIE", "DISCORDANT", "BOTH_TIED", "ONE_BACKEND_TIED"},
        "scene-ordering tie classes mismatch",
        checks,
        "SCENE_ORDERING_TIE_BEHAVIOR",
    )

    turnover = roots.get("formal_reassociation_turnover", {})
    companion = turnover.get("accepted_source_turnover", {})
    _require(
        turnover.get("formal_primary_field") == "correspondence_turnover"
        and companion.get("retained") is True
        and companion.get("role") == "SECONDARY_DESCRIPTIVE_COMPANION"
        and companion.get("may_replace_formal_primary") is False
        and companion.get("selection_by_larger_correlation_forbidden") is True,
        "formal turnover mapping mismatch",
        checks,
        "CORRESPONDENCE_TURNOVER_PRIMARY_ACCEPTED_SOURCE_RETAINED",
    )

    centered = roots.get("within_scene_centered_association", {})
    _require(
        centered.get("x_field") == "correspondence_turnover"
        and centered.get("y_field") == "translation_norm_m"
        and centered.get("both_variables_centered") is True
        and centered.get("center_statistic") == "MEDIAN"
        and centered.get("association_statistic") == "SPEARMAN_RHO"
        and centered.get("implementation") == "scipy.stats.spearmanr"
        and centered.get("rank_ties") == "AVERAGE_RANKS"
        and centered.get("scene_pair_count_required") == 30
        and centered.get("scene_strata_required") == 6
        and centered.get("total_pair_count_required") == 180,
        "centered association definition mismatch",
        checks,
        "BOTH_VARIABLES_MEDIAN_CENTERED_SPEARMAN_180",
    )
    _require(
        centered.get("scipy_asymptotic_p_value_is_formal_inference") is False
        and centered.get("snapshots_claimed_as_independent_scenes") is False
        and centered.get("registered_permutation_runs_when_undefined") is False,
        "centered association inferential boundary mismatch",
        checks,
        "CENTERED_EFFECT_SIZE_NO_ASYMPTOTIC_INFERENCE",
    )

    permutation = roots.get("registered_stratified_sensitivity_permutation", {})
    _require(
        permutation.get("seed") == 20260820
        and permutation.get("permutation_count") == 10000
        and permutation.get("bit_generator") == "PCG64"
        and permutation.get("p_value_denominator") == 10001
        and permutation.get("two_sided") is True
        and permutation.get("turnover_values_fixed") is True
        and permutation.get("cross_scene_permutation_forbidden") is True
        and permutation.get("duplicate_draws_retained") is True,
        "stratified permutation engine mismatch",
        checks,
        "STRATIFIED_PCG64_10000_DRAWS_SEED_20260820",
    )
    _require(
        permutation.get("observed_statistic")
        == "abs(Spearman(centered correspondence_turnover, centered translation_norm_m))"
        and permutation.get("exceedance_rule")
        == "abs(rho_perm) >= abs(rho_obs)"
        and permutation.get("p_value_formula")
        == "(1 + exceedance_count) / (10000 + 1)"
        and permutation.get("role")
        == "SECONDARY_STRATIFIED_MONTE_CARLO_PERMUTATION_SENSITIVITY"
        and permutation.get("is_primary_exact_20_allocation_test") is False,
        "stratified permutation statistic or Monte-Carlo correction mismatch",
        checks,
        "ABS_SPEARMAN_AND_PLUS_ONE_OVER_10001",
    )

    systematic = roots.get("systematic_weak_rich_comparison", {})
    _require(
        systematic.get("estimand")
        == (
            "median(W01,W02,W03 scene systematic fractions) - "
            "median(R01,R02,R03 scene systematic fractions)"
        )
        and systematic.get("role")
        == "SECONDARY_DESCRIPTIVE_MECHANISTIC_COMPARISON"
        and systematic.get("formal_hypothesis_test") is False
        and systematic.get("p_value") is None
        and systematic.get("directional_pass_fail") is False
        and systematic.get("permutation") is False
        and systematic.get("mann_whitney") is False
        and systematic.get("t_test") is False,
        "systematic Weak/Rich descriptive-only rule mismatch",
        checks,
        "SYSTEMATIC_DESCRIPTIVE_ESTIMAND_WITHOUT_INFERENCE",
    )

    primary = c2.get("unchanged_primary_analysis", {})
    c1_primary = c1_contract.get("primary_endpoint", {})
    _require(
        primary.get("endpoint") == c1_primary.get("row_value") == "translation_norm_m"
        and primary.get("scene_snapshot_count")
        == c1_primary.get("scene_summary", {}).get("input_count")
        == 30
        and primary.get("scene_statistics")
        == c1_primary.get("scene_summary", {}).get("statistics")
        == ["median", "q25", "q75", "q95"]
        and primary.get("quantile_method")
        == c1_primary.get("scene_summary", {}).get("quantile_method")
        == "linear"
        and primary.get("allocation_count")
        == c1_primary.get("inference", {}).get("allocation_count")
        == 20
        and primary.get("plus_one_correction") is False,
        "primary endpoint or exact 20-allocation inference changed",
        checks,
        "PRIMARY_ENDPOINT_AND_EXACT_20_PERMUTATION_UNCHANGED",
    )
    unchanged = c2.get("unchanged_rotation_and_cross_backend_rules", {})
    _require(
        unchanged.get("rotation_role") == "SECONDARY"
        and unchanged.get("rotation_uses_same_exact_20_allocation_one_sided_procedure")
        is True
        and unchanged.get("scene_spearman_required_pairs") == 6
        and unchanged.get("station_spearman_required_pairs") == 18,
        "rotation or existing Spearman rule changed",
        checks,
        "ROTATION_AND_CROSS_BACKEND_SPEARMAN_UNCHANGED",
    )
    missingness = c2.get("unchanged_missingness_c1", {})
    _require(
        missingness.get("station_required_complete") == "10_OF_10"
        and missingness.get("scene_required_complete") == "30_OF_30"
        and missingness.get("centered_required_complete")
        == "30_PAIRS_X_6_SCENES_EQUALS_180"
        and missingness.get("systematic_station_required_complete")
        == "10_OF_10_VECTORS"
        and missingness.get("imputation_forbidden") is True
        and missingness.get("winsorization_forbidden") is True
        and missingness.get("silent_exclusion_forbidden") is True
        and missingness.get("available_case_replaces_formal_inference") is False,
        "C1 missingness semantics changed",
        checks,
        "C1_MISSINGNESS_UNCHANGED",
    )
    authority = c2.get("execution_authority", {})
    _require(
        authority.get("implements_locked_analysis") is False
        and authority.get("authorizes_locked_scientific_analysis") is False
        and authority.get("authorizes_backend_execution") is False
        and authority.get("real_formal_result_files_read_by_c2_tooling") == 0
        and authority.get("real_scientific_aggregation_count") == 0
        and authority.get("real_weak_rich_comparison_count") == 0
        and authority.get("real_p_value_count") == 0,
        "C2 authority or real-analysis counters mismatch",
        checks,
        "NO_ANALYSIS_AUTHORITY_AND_ZERO_REAL_COUNTERS",
    )

    _require(
        historical_audit.get("starting_head") == POSTRUN_VERIFICATION_COMMIT
        and historical_audit.get("gate", {}).get("required_under_specified_count") == 6
        and historical_audit.get("gate", {}).get("unresolved_root_definition_count") == 5,
        "historical determinacy audit identity or blocker counts mismatch",
        checks,
        "HISTORICAL_DETERMINACY_AUDIT_PRESERVED",
    )
    _require(
        postrun_report.get("FMB1_POSTRUN_INDEPENDENT_VERIFICATION_PASS") is True
        and postrun_report.get("VERIFIED_TRIAL_COUNT") == 360
        and postrun_report.get("SCIENTIFIC_AGGREGATION_EXECUTED") is False
        and postrun_report.get("WEAK_RICH_COMPARISON_EXECUTED") is False
        and postrun_report.get("P_VALUE_COMPUTED") is False,
        "Post-run PASS report or pre-analysis state mismatch",
        checks,
        "POSTRUN_VERIFICATION_PASS_WITHOUT_SCIENTIFIC_AGGREGATION",
    )
    _require(
        basis.get("source_determinacy_audit_commit") == DETERMINACY_AUDIT_COMMIT
        and basis.get("unresolved_root_definition_count_before") == 5
        and basis.get("actual_scientific_aggregation_count") == 0
        and basis.get("weak_rich_comparison_count") == 0
        and basis.get("formal_p_value_count") == 0
        and basis.get("definition_selected_from_result_values") is False
        and basis.get("formal_trial_result_files_read_by_clarification_tooling") == 0
        and len(basis.get("roots", [])) == 5,
        "C2 clarification basis mismatch",
        checks,
        "C2_BASIS_FIVE_ROOTS_RESULT_BLIND",
    )
    return checks


def _verify_no_formal_result_read_capability(repository: Path) -> None:
    semantic_path = (
        repository
        / "experiments/mid360_formal_batch1/analysis_determinacy_c2_semantics_v1.py"
    )
    tool_path = (
        repository
        / "tools/mid360_formal_batch1/verify_analysis_determinacy_c2_v1.py"
    )
    semantic_tree = ast.parse(semantic_path.read_text(encoding="utf-8"))
    forbidden_semantic_calls = {"open", "read_text", "read_bytes", "glob", "rglob"}
    for node in ast.walk(semantic_tree):
        if isinstance(node, ast.Call):
            function = node.func
            name = function.id if isinstance(function, ast.Name) else (
                function.attr if isinstance(function, ast.Attribute) else ""
            )
            if name in forbidden_semantic_calls:
                raise C2VerificationError(
                    f"fixture semantic module has file-read capability: {name}"
                )
    tool_source = tool_path.read_text(encoding="utf-8")
    forbidden_cli_tokens = (
        "--formal-results-root",
        "--execution-results-root",
        "--confirm-read-frozen-formal-results",
        "raw_runtime_snapshot",
        "trial_results",
    )
    for token in forbidden_cli_tokens:
        if token in tool_source:
            raise C2VerificationError(f"C2 verifier CLI exposes forbidden input token: {token}")


def verify_repository(repository: Path) -> dict[str, Any]:
    """Verify canonical C2 bytes and semantics without traversing formal results."""

    repository = repository.resolve()
    for relative_path, expected_sha in EXPECTED_PROTECTED_SHA256.items():
        actual_sha = sha256_file(repository / relative_path)
        if actual_sha != expected_sha:
            raise C2VerificationError(
                f"protected source SHA mismatch: {relative_path}: {actual_sha}"
            )
    c2_path = repository / C2_PATH
    c2_md_path = repository / C2_MD_PATH
    basis_path = repository / BASIS_PATH
    c2 = _json(c2_path)
    basis = _json(basis_path)
    c1_contract = _json(repository / C1_CONTRACT_PATH)
    historical_audit = _json(repository / HISTORICAL_AUDIT_PATH)
    postrun_report = _json(repository / POSTRUN_REPORT_PATH)
    checks = validate_c2_payloads(
        c2,
        basis,
        c1_contract,
        historical_audit,
        postrun_report,
    )
    _verify_no_formal_result_read_capability(repository)
    checks.append("C2_TOOLING_HAS_NO_FORMAL_RESULT_READ_CAPABILITY")
    # C2 was originally issued before implementation.  A later implementation
    # or lock may exist, but this verifier must remain blind to both: it verifies
    # only the frozen C2 inputs above and never traverses post-C2 artifacts.
    checks.append("POST_C2_IMPLEMENTATION_AND_LOCK_NOT_TRAVERSED")
    if len(checks) != 23:
        raise C2VerificationError(f"expected exactly 23 semantic checks, got {len(checks)}")
    return {
        "schema": "mid360_fmb1_analysis_determinacy_c2_independent_verification_v1",
        "status": "PASS",
        "pass": True,
        "FMB1_ANALYSIS_DETERMINACY_C2_ACTIVE": True,
        "clarification_phase": C2_PHASE,
        "semantic_check_count": len(checks),
        "semantic_checks": checks,
        "c2_sha256": sha256_file(c2_path),
        "c2_md_sha256": sha256_file(c2_md_path),
        "basis_sha256": sha256_file(basis_path),
        "protected_source_count": len(EXPECTED_PROTECTED_SHA256),
        "protected_source_mismatch_count": 0,
        "primary_analysis_changed": False,
        "scientific_aggregation_executed": False,
        "weak_rich_comparison_executed": False,
        "formal_p_value_computed": False,
        "formal_trial_result_files_read": 0,
        "backend_execution_count": 0,
    }


def validate_reaudit_payload(
    audit: dict[str, Any],
    *,
    c2_sha256: str,
) -> list[str]:
    """Independently validate the later versioned C1+C2 determinacy re-audit."""

    checks: list[str] = []
    if audit.get("status") != "PASS_FULLY_DETERMINATE_C1_PLUS_C2":
        raise C2VerificationError("re-audit status is not PASS_FULLY_DETERMINATE_C1_PLUS_C2")
    checks.append("REAUDIT_STATUS_PASS")
    bindings = audit.get("source_bindings", {})
    if bindings.get("c2", {}).get("sha256") != c2_sha256:
        raise C2VerificationError("re-audit does not bind the canonical C2 SHA")
    if bindings.get("c1_contract", {}).get("sha256") != EXPECTED_PROTECTED_SHA256[
        C1_CONTRACT_PATH
    ]:
        raise C2VerificationError("re-audit C1 contract binding mismatch")
    checks.append("REAUDIT_BINDS_C1_AND_C2")
    items = audit.get("determinacy_items", [])
    if len(items) != 18 or [item.get("id") for item in items] != list(range(1, 19)):
        raise C2VerificationError("re-audit must contain the same ordered 18 required rows")
    allowed = {
        "FULLY_DETERMINED",
        "DETERMINED_BY_EXPLICIT_CROSS_REFERENCE",
        "DESCRIPTIVE_ONLY_NO_INFERENCE_REQUIRED",
    }
    if any(item.get("classification") not in allowed for item in items):
        raise C2VerificationError("re-audit retains an under-specified classification")
    checks.append("REAUDIT_ALL_18_ROWS_DETERMINATE")
    gate = audit.get("gate", {})
    if not (
        gate.get("required_under_specified_count") == 0
        and gate.get("unresolved_root_definition_count") == 0
        and gate.get("FMB1_LOCKED_ANALYSIS_CONTRACT_DETERMINATE") is True
        and gate.get("READY_FOR_LOCKED_ANALYSIS_IMPLEMENTATION_AND_FREEZE") is True
        and gate.get("READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS") is False
    ):
        raise C2VerificationError("re-audit gate values mismatch")
    checks.append("REAUDIT_ZERO_UNRESOLVED_AND_IMPLEMENTATION_READY")
    scope = audit.get("audit_scope", {})
    if not (
        scope.get("formal_trial_result_files_opened") == 0
        and scope.get("real_formal_scientific_values_read") == 0
        and scope.get("real_scientific_aggregation_count") == 0
        and scope.get("real_weak_rich_comparison_count") == 0
        and scope.get("real_p_value_computation_count") == 0
    ):
        raise C2VerificationError("re-audit is not result-blind")
    checks.append("REAUDIT_RESULT_BLIND_ZERO_COUNTERS")
    return checks
