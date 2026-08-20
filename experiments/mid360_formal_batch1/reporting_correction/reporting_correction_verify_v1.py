"""Independent verifier for the FMB1 solver-status reporting correction v1."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


ORIGINAL_RESULT_COMMIT = "2302f3fe6de804c934e2131b8fc77a49424bd83b"
ORIGINAL_RESULT_TAG = "results/fmb1-zero-perturbation-locked-analysis-v1"
DIAGNOSTIC_COMMIT = "a3ada7c345fefc728bdcf4b8a8b92465c974ef4e"
DIAGNOSTIC_TAG = "diagnostic/fmb1-solver-convergence-v1"
RAW_EXECUTION_COMMIT = "059e39533991d929a97ab208ad738643af82d09a"
LOCKED_ANALYSIS_CODE_COMMIT = "215a7961ed93dac9cf691e1e8ccd99d7ee868175"
ORIGINAL_SUMMARY_SHA256 = (
    "b23c303e013ef5194d109b997a35876153140f8ad911dfa5188b6b0d53c1f869"
)
ORIGINAL_SUMMARY_RELATIVE = (
    "results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/"
    "analysis_summary.json"
)
FORMAL_ROWS_RELATIVE = (
    "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1/"
    "raw_runtime_snapshot/trial_results"
)
DIAGNOSTIC_RELATIVE = "results/mid360_formal_batch1/solver_convergence_diagnostic_v1"
RAW_RESULTS_RELATIVE = (
    "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1"
)
POSTRUN_RELATIVE = (
    "results/mid360_formal_batch1/zero_perturbation_v1_1_postrun_verification_v1"
)
LOCKED_RESULTS_RELATIVE = (
    "results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1"
)
LOCKED_CODE_PATHS = (
    "experiments/mid360_formal_batch1/locked_analysis",
    "tools/mid360_formal_batch1/run_zero_perturbation_locked_analysis_v1.py",
)
PRODUCTION_PATHS = (
    "experiments/mid360_formal_batch1/reporting_correction/__init__.py",
    "experiments/mid360_formal_batch1/reporting_correction/solver_status_reporting_correction_v1.py",
    "experiments/mid360_formal_batch1/reporting_correction/reporting_correction_verify_v1.py",
    "tools/mid360_formal_batch1/build_solver_status_reporting_correction_v1.py",
    "tools/mid360_formal_batch1/verify_solver_status_reporting_correction_v1.py",
)
IMPLEMENTATION_PATHS = (
    "experiments/mid360_formal_batch1/reporting_correction/__init__.py",
    "experiments/mid360_formal_batch1/reporting_correction/solver_status_reporting_correction_v1.py",
    "experiments/mid360_formal_batch1/reporting_correction/reporting_correction_schema_v1.json",
    "experiments/mid360_formal_batch1/reporting_correction/reporting_correction_verify_v1.py",
    "tools/mid360_formal_batch1/build_solver_status_reporting_correction_v1.py",
    "tools/mid360_formal_batch1/verify_solver_status_reporting_correction_v1.py",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "open3d",
    "pcl",
    "phase_a_harness.open3d_backend",
    "phase_a_harness.pcl_backend",
)

EXPECTED_ACCOUNTING = {
    "formal_trial_count": 360,
    "formal_finite_result_n": 360,
    "formal_completed_n": 360,
    "formal_solver_nonconvergence_n": 0,
    "formal_infrastructure_ok_n": 360,
    "open3d_formal_completed_n": 180,
    "open3d_formal_solver_nonconvergence_n": 0,
    "pcl_formal_completed_n": 180,
    "pcl_formal_solver_nonconvergence_n": 0,
    "pcl_native_has_converged_true_n": 180,
    "pcl_native_has_converged_false_n": 0,
    "open3d_native_convergence_observable": False,
    "pcl_native_convergence_observable": True,
    "TRUE_NATIVE_NONCONVERGENCE_BOUNDS": "0..180",
}

REQUIRED_IMMUTABILITY_FLAGS = (
    "PRIMARY_TRANSLATION_VALUES_IDENTICAL",
    "PRIMARY_EXACT_P_VALUES_IDENTICAL",
    "ROTATION_VALUES_IDENTICAL",
    "CROSS_BACKEND_VALUES_IDENTICAL",
    "REASSOCIATION_VALUES_IDENTICAL",
    "CENTERED_PERMUTATION_VALUES_IDENTICAL",
    "SYSTEMATIC_VALUES_IDENTICAL",
    "PHYSICAL_REFERENCE_SEMANTICS_IDENTICAL",
    "CAPTURE_RADIUS_STATUS_IDENTICAL",
    "ALL_ORIGINAL_SUMMARY_CONTENT_IDENTICAL_AFTER_CORRECTION_METADATA_REMOVAL",
)


class ReportingCorrectionVerificationError(ValueError):
    """Raised when the independent reporting-correction verifier fails closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReportingCorrectionVerificationError(f"expected JSON object: {path}")
    return value


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git",) + args, cwd=repository, text=True
    ).strip()


def _git_diff_quiet(repository: Path, commit: str, *paths: str) -> bool:
    result = subprocess.run(
        ("git", "diff", "--quiet", commit, "--") + paths,
        cwd=repository,
        check=False,
    )
    return result.returncode == 0


def _verify_sums(directory: Path) -> dict[str, Any]:
    manifest = directory / "SHA256SUMS"
    failures: list[str] = []
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        target = directory / name.strip()
        count += 1
        if not target.is_file() or _sha256(target) != expected:
            failures.append(name.strip())
    return {
        "status": "PASS" if not failures else "FAIL",
        "entry_count": count,
        "failures": failures,
        "manifest_sha256": _sha256(manifest),
    }


def _row_accounting(repository: Path) -> dict[str, Any]:
    paths = sorted((repository / FORMAL_ROWS_RELATIVE).glob("*/attempt-0001.json"))
    rows = [_object(path) for path in paths]
    per_backend: dict[str, dict[str, Any]] = {}
    for backend in ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE"):
        selected = [row for row in rows if row.get("backend") == backend]
        statuses: dict[str, int] = {}
        for row in selected:
            key = str(row.get("solver_status"))
            statuses[key] = statuses.get(key, 0) + 1
        per_backend[backend] = {
            "planned_n": len(selected),
            "finite_result_n": sum(row.get("finite_result") is True for row in selected),
            "formal_completed_n": sum(
                row.get("scientific_status") == "COMPLETED" for row in selected
            ),
            "formal_solver_nonconvergence_n": sum(
                row.get("scientific_status") == "SOLVER_NON_CONVERGENCE"
                for row in selected
            ),
            "infrastructure_ok_n": sum(
                row.get("infrastructure_status") == "OK" for row in selected
            ),
            "solver_status_inventory": dict(sorted(statuses.items())),
        }
    return {
        "row_count": len(rows),
        "unique_trial_count": len({str(row.get("trial_id")) for row in rows}),
        "per_backend": per_backend,
    }


def _independent_json_diff(original: Any, corrected: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(original, dict) and isinstance(corrected, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(original) | set(corrected)):
            child = f"{path}/{key.replace('~', '~0').replace('/', '~1')}"
            if key not in original:
                rows.append(
                    {
                        "json_path": child,
                        "old_value": "MISSING_PATH",
                        "new_value": corrected[key],
                        "reason": "ADD_VERSIONED_SOLVER_ACCOUNTING_CORRECTION_PROVENANCE",
                    }
                )
            elif key not in corrected:
                rows.append(
                    {
                        "json_path": child,
                        "old_value": original[key],
                        "new_value": "MISSING_PATH",
                        "reason": "UNEXPECTED_REMOVAL",
                    }
                )
            else:
                rows.extend(_independent_json_diff(original[key], corrected[key], child))
        return rows
    if isinstance(original, list) and isinstance(corrected, list):
        rows = []
        for index in range(max(len(original), len(corrected))):
            child = f"{path}/{index}"
            if index >= len(original) or index >= len(corrected):
                rows.append(
                    {
                        "json_path": child,
                        "old_value": original[index] if index < len(original) else "MISSING_PATH",
                        "new_value": corrected[index] if index < len(corrected) else "MISSING_PATH",
                        "reason": "UNEXPECTED_LIST_SHAPE_CHANGE",
                    }
                )
            else:
                rows.extend(_independent_json_diff(original[index], corrected[index], child))
        return rows
    if original != corrected:
        return [
            {
                "json_path": path or "/",
                "old_value": original,
                "new_value": corrected,
                "reason": "UNEXPECTED_VALUE_CHANGE",
            }
        ]
    return []


def _static_source_safe(repository: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    for relative in PRODUCTION_PATHS:
        path = repository / relative
        if not path.is_file():
            findings.append({"path": relative, "reason": "MISSING_PRODUCTION_FILE"})
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES):
                    findings.append({"path": relative, "reason": f"FORBIDDEN_IMPORT:{name}"})
    return {"status": "PASS" if not findings else "FAIL", "findings": findings}


def _expected_scientific_references(original: Mapping[str, Any]) -> dict[str, Any]:
    primary = {
        str(row["backend"]): {
            "estimand_m": row["estimand"],
            "exact_p_value": row["p_value"],
        }
        for row in original["primary_translation_inference"]
    }
    centered = {
        str(row["backend"]): {"rho": row["rho"]}
        for row in original["reassociation_centered_association"]
    }
    centered_p = {
        str(row["backend"]): {
            "p_value": row["p_value"],
            "p_value_denominator": row["p_value_denominator"],
        }
        for row in original["reassociation_centered_permutation_sensitivity"]
    }
    return {
        "role": "ORIGINAL_FROZEN_SCIENTIFIC_RESULT_REFERENCE",
        "primary_translation": primary,
        "reassociation_centered": centered,
        "reassociation_centered_permutation": centered_p,
        "paper_rounded_references": {
            "open3d_weak_minus_rich_mm": "-0.026347782",
            "pcl_weak_minus_rich_mm": "-0.010915495",
            "open3d_centered_rho": "0.737691493",
            "pcl_centered_rho": "0.761171641",
            "centered_permutation_p": "1/10001",
        },
    }


def collect_observed(repository: str | Path, correction_dir: str | Path) -> dict[str, Any]:
    root = Path(repository).resolve(strict=True)
    correction = Path(correction_dir).resolve(strict=True)
    original_summary = root / ORIGINAL_SUMMARY_RELATIVE
    core_names = (
        "reporting_correction_notice_v1.json",
        "reporting_correction_notice_v1.md",
        "corrected_solver_status_accounting_v1.json",
        "corrected_solver_status_accounting_v1.csv",
        "analysis_summary_reporting_corrected_v1.json",
        "analysis_summary_reporting_corrected_v1.md",
        "scientific_value_immutability_check.json",
        "json_path_correction_diff.json",
    )
    return {
        "original_summary_sha256": _sha256(original_summary),
        "result_tag_type": _git(root, "cat-file", "-t", ORIGINAL_RESULT_TAG),
        "result_tag_peel": _git(root, "rev-parse", f"{ORIGINAL_RESULT_TAG}^{{}}"),
        "diagnostic_tag_type": _git(root, "cat-file", "-t", DIAGNOSTIC_TAG),
        "diagnostic_tag_peel": _git(root, "rev-parse", f"{DIAGNOSTIC_TAG}^{{}}"),
        "diagnostic_sha256sums": _verify_sums(root / DIAGNOSTIC_RELATIVE),
        "raw_sha256sums": _verify_sums(root / RAW_RESULTS_RELATIVE),
        "postrun_sha256sums": _verify_sums(root / POSTRUN_RELATIVE),
        "locked_analysis_sha256sums": _verify_sums(root / LOCKED_RESULTS_RELATIVE),
        "formal_rows": _row_accounting(root),
        "core_artifact_sha256": {
            name: _sha256(correction / name) for name in core_names
        },
        "implementation_artifact_sha256": {
            relative: _sha256(root / relative) for relative in IMPLEMENTATION_PATHS
        },
        "raw_execution_unchanged": _git_diff_quiet(
            root, RAW_EXECUTION_COMMIT, RAW_RESULTS_RELATIVE
        ),
        "locked_analysis_results_unchanged": _git_diff_quiet(
            root, ORIGINAL_RESULT_COMMIT, LOCKED_RESULTS_RELATIVE
        ),
        "solver_diagnostic_unchanged": _git_diff_quiet(
            root, DIAGNOSTIC_COMMIT, DIAGNOSTIC_RELATIVE
        ),
        "locked_analysis_code_unchanged": _git_diff_quiet(
            root, LOCKED_ANALYSIS_CODE_COMMIT, *LOCKED_CODE_PATHS
        ),
        "static_source_safety": _static_source_safe(root),
    }


def validate_payloads(
    *,
    original: Mapping[str, Any],
    notice: Mapping[str, Any],
    accounting: Mapping[str, Any],
    corrected: Mapping[str, Any],
    diff_payload: Mapping[str, Any],
    immutability: Mapping[str, Any],
    manifest: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []

    def check(condition: bool, code: str) -> None:
        if not condition:
            failures.append(code)

    check(notice.get("schema") == "fmb1_solver_status_reporting_correction_notice_v1", "NOTICE_SCHEMA_INVALID")
    check(notice.get("status") == "AUTHORITATIVE_VERSIONED_REPORTING_CORRECTION", "NOTICE_STATUS_INVALID")
    check(notice.get("CORRECTION_TYPE") == "VERSIONED_REPORTING_CORRECTION", "CORRECTION_TYPE_INVALID")
    check(notice.get("CORRECTION_SCOPE") == "SOLVER_STATUS_ACCOUNTING_ONLY", "CORRECTION_SCOPE_INVALID")
    for key in (
        "SCIENTIFIC_ANALYSIS_RERUN",
        "REGISTRATION_RERUN",
        "RAW_POSE_RECOMPUTATION",
        "PRIMARY_ESTIMAND_RECOMPUTATION",
        "P_VALUE_RECOMPUTATION",
        "REASSOCIATION_RECOMPUTATION",
        "SYSTEMATIC_COMPONENT_RECOMPUTATION",
    ):
        check(notice.get(key) is False, f"{key}_MUST_BE_FALSE")
    check(notice.get("POSTHOC_DIAGNOSTIC_TRIGGERED") is True, "POSTHOC_TRIGGER_MISSING")
    check(notice.get("ORIGINAL_REPORTED_SOLVER_NONCONVERGED_FINITE_N") == 360, "ORIGINAL_COUNT_INVALID")
    check(notice.get("CORRECTED_FORMAL_COMPLETED_N") == 360, "CORRECTED_COMPLETED_INVALID")
    check(notice.get("CORRECTED_FORMAL_SOLVER_NONCONVERGENCE_N") == 0, "CORRECTED_NONCONVERGENCE_INVALID")
    check(notice.get("ROOT_CAUSE") == "WRONG_SOLVER_STATUS_STRING_MAPPING", "ROOT_CAUSE_INVALID")

    for key, value in EXPECTED_ACCOUNTING.items():
        check(accounting.get(key) == value, f"ACCOUNTING_{key}_INVALID")
    by_backend = accounting.get("by_backend", {})
    expected_backend = {
        "planned_n": 180,
        "finite_result_n": 180,
        "formal_completed_n": 180,
        "formal_solver_nonconvergence_n": 0,
        "infrastructure_ok_n": 180,
        "solver_status_inventory": {"completed_finite_correspondences": 180},
    }
    check(by_backend.get("OPEN3D_POINT_TO_PLANE") == expected_backend, "OPEN3D_ACCOUNTING_INVALID")
    check(by_backend.get("PCL_POINT_TO_PLANE") == expected_backend, "PCL_ACCOUNTING_INVALID")
    check(notice.get("corrected_accounting") == accounting, "NOTICE_ACCOUNTING_BINDING_INVALID")

    stripped = copy.deepcopy(dict(corrected))
    correction_metadata = stripped.pop("reporting_correction", None)
    check(stripped == dict(original), "ORIGINAL_SUMMARY_CONTENT_CHANGED")
    check(isinstance(correction_metadata, dict), "CORRECTION_METADATA_MISSING")
    if isinstance(correction_metadata, dict):
        check(correction_metadata.get("corrected_accounting") == accounting, "CORRECTED_SUMMARY_ACCOUNTING_INVALID")
        check(correction_metadata.get("REGISTRATION_RERUN") is False, "CORRECTED_SUMMARY_REGISTRATION_RERUN")
        check(correction_metadata.get("SCIENTIFIC_ANALYSIS_RERUN") is False, "CORRECTED_SUMMARY_ANALYSIS_RERUN")
        check(correction_metadata.get("open3d_native_convergence_observable") is None, "OPEN3D_NATIVE_FIELD_MALFORMED")
        embedded = correction_metadata.get("corrected_accounting", {})
        check(embedded.get("open3d_native_convergence_observable") is False, "OPEN3D_NATIVE_CONVERGENCE_CLAIM_INVALID")
        deprecated = correction_metadata.get("deprecated_reporting_field", {})
        check(deprecated.get("deprecated") is True, "DEPRECATED_FIELD_MARKER_MISSING")
        check(deprecated.get("original_reported_aggregate_value") == 360, "DEPRECATED_ORIGINAL_VALUE_INVALID")

    expected_diff = _independent_json_diff(dict(original), dict(corrected))
    check(diff_payload.get("changes") == expected_diff, "JSON_PATH_DIFF_MISMATCH")
    changed_paths = [row.get("json_path") for row in diff_payload.get("changes", [])]
    check(changed_paths == ["/reporting_correction"], "FORBIDDEN_JSON_PATH_CHANGE")
    check(diff_payload.get("forbidden_scientific_path_change_count") == 0, "FORBIDDEN_PATH_COUNT_NONZERO")

    for key in REQUIRED_IMMUTABILITY_FLAGS:
        check(immutability.get(key) is True, f"IMMUTABILITY_{key}_FAIL")
    check(immutability.get("PRIMARY_SCIENTIFIC_NUMERICAL_RESULTS_CHANGED") is False, "SCIENTIFIC_VALUES_CHANGED")
    check(immutability.get("new_hypothesis_tests_added") is False, "NEW_HYPOTHESIS_TEST_ADDED")

    check(observed.get("original_summary_sha256") == ORIGINAL_SUMMARY_SHA256, "ORIGINAL_SUMMARY_SHA_MISMATCH")
    check(observed.get("result_tag_type") == "tag", "RESULT_TAG_NOT_ANNOTATED")
    check(observed.get("result_tag_peel") == ORIGINAL_RESULT_COMMIT, "RESULT_TAG_MOVED")
    check(observed.get("diagnostic_tag_type") == "tag", "DIAGNOSTIC_TAG_NOT_ANNOTATED")
    check(observed.get("diagnostic_tag_peel") == DIAGNOSTIC_COMMIT, "DIAGNOSTIC_TAG_MOVED")
    for key in (
        "diagnostic_sha256sums",
        "raw_sha256sums",
        "postrun_sha256sums",
        "locked_analysis_sha256sums",
    ):
        check(observed.get(key, {}).get("status") == "PASS", f"{key.upper()}_FAIL")
    row_status = observed.get("formal_rows", {})
    check(row_status.get("row_count") == 360, "FORMAL_ROW_COUNT_INVALID")
    check(row_status.get("unique_trial_count") == 360, "FORMAL_TRIAL_UNIQUENESS_INVALID")
    check(row_status.get("per_backend", {}).get("OPEN3D_POINT_TO_PLANE") == expected_backend, "OBSERVED_OPEN3D_INVALID")
    check(row_status.get("per_backend", {}).get("PCL_POINT_TO_PLANE") == expected_backend, "OBSERVED_PCL_INVALID")

    check(manifest.get("status") == "AUTHORITATIVE_VERSIONED_REPORTING_CORRECTION", "MANIFEST_STATUS_INVALID")
    check(manifest.get("CORRECTION_TYPE") == "VERSIONED_REPORTING_CORRECTION", "MANIFEST_TYPE_INVALID")
    check(manifest.get("CORRECTION_SCOPE") == "SOLVER_STATUS_ACCOUNTING_ONLY", "MANIFEST_SCOPE_INVALID")
    original_binding = manifest.get("original_scientific_analysis", {})
    check(original_binding.get("commit") == ORIGINAL_RESULT_COMMIT, "MANIFEST_RESULT_COMMIT_INVALID")
    check(original_binding.get("tag") == ORIGINAL_RESULT_TAG, "MANIFEST_RESULT_TAG_INVALID")
    check(original_binding.get("analysis_summary_sha256") == ORIGINAL_SUMMARY_SHA256, "MANIFEST_SUMMARY_SHA_INVALID")
    diagnostic_binding = manifest.get("solver_diagnostic", {})
    check(diagnostic_binding.get("commit") == DIAGNOSTIC_COMMIT, "MANIFEST_DIAGNOSTIC_COMMIT_INVALID")
    check(diagnostic_binding.get("tag") == DIAGNOSTIC_TAG, "MANIFEST_DIAGNOSTIC_TAG_INVALID")
    check(manifest.get("corrected_accounting") == accounting, "MANIFEST_ACCOUNTING_INVALID")
    check(
        manifest.get("scientific_value_references")
        == _expected_scientific_references(original),
        "SCIENTIFIC_REFERENCE_BINDING_INVALID",
    )
    check(manifest.get("core_artifact_sha256") == observed.get("core_artifact_sha256"), "CORE_ARTIFACT_SHA_MISMATCH")
    check(
        manifest.get("implementation_artifact_sha256")
        == observed.get("implementation_artifact_sha256"),
        "IMPLEMENTATION_ARTIFACT_SHA_MISMATCH",
    )
    check(manifest.get("forbidden_scientific_path_change_count") == 0, "MANIFEST_FORBIDDEN_PATH_CHANGE")
    for key in (
        "registration_backend_call_count",
        "formal_scientific_analysis_run_count",
        "primary_estimand_recomputation_count",
        "p_value_recomputation_count",
        "raw_result_modification_count",
        "frozen_analysis_result_modification_count",
    ):
        check(manifest.get(key) == 0, f"MANIFEST_{key.upper()}_NONZERO")
    check(manifest.get("not_a_new_scientific_experiment") is True, "NEW_EXPERIMENT_CLAIM_INVALID")

    for key in (
        "raw_execution_unchanged",
        "locked_analysis_results_unchanged",
        "solver_diagnostic_unchanged",
        "locked_analysis_code_unchanged",
    ):
        check(observed.get(key) is True, f"{key.upper()}_FAIL")
    check(observed.get("static_source_safety", {}).get("status") == "PASS", "STATIC_SOURCE_SAFETY_FAIL")
    return sorted(set(failures))


def verify_reporting_correction(
    repository: str | Path, correction_dir: str | Path
) -> dict[str, Any]:
    root = Path(repository).resolve(strict=True)
    correction = Path(correction_dir).resolve(strict=True)
    original = _object(root / ORIGINAL_SUMMARY_RELATIVE)
    notice = _object(correction / "reporting_correction_notice_v1.json")
    accounting = _object(correction / "corrected_solver_status_accounting_v1.json")
    corrected = _object(correction / "analysis_summary_reporting_corrected_v1.json")
    diff_payload = _object(correction / "json_path_correction_diff.json")
    immutability = _object(correction / "scientific_value_immutability_check.json")
    manifest = _object(correction / "reporting_correction_manifest.json")
    observed = collect_observed(root, correction)
    failures = validate_payloads(
        original=original,
        notice=notice,
        accounting=accounting,
        corrected=corrected,
        diff_payload=diff_payload,
        immutability=immutability,
        manifest=manifest,
        observed=observed,
    )
    return {
        "schema": "fmb1_independent_reporting_correction_verification_v1",
        "status": "PASS" if not failures else "FAIL",
        "pass": not failures,
        "INDEPENDENT_REPORTING_CORRECTION_VERIFICATION_PASS": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "original_analysis_summary_sha256": observed["original_summary_sha256"],
        "original_result_commit": ORIGINAL_RESULT_COMMIT,
        "original_result_tag": ORIGINAL_RESULT_TAG,
        "solver_diagnostic_commit": DIAGNOSTIC_COMMIT,
        "solver_diagnostic_tag": DIAGNOSTIC_TAG,
        "formal_trial_count": observed["formal_rows"]["row_count"],
        "formal_completed_n": 360 if not failures else None,
        "formal_solver_nonconvergence_n": 0 if not failures else None,
        "open3d_planned_n": observed["formal_rows"]["per_backend"].get("OPEN3D_POINT_TO_PLANE", {}).get("planned_n"),
        "pcl_planned_n": observed["formal_rows"]["per_backend"].get("PCL_POINT_TO_PLANE", {}).get("planned_n"),
        "json_path_diff_status": "PASS" if "JSON_PATH_DIFF_MISMATCH" not in failures else "FAIL",
        "scientific_value_immutability_status": "PASS" if not any(item.startswith("IMMUTABILITY_") or item == "ORIGINAL_SUMMARY_CONTENT_CHANGED" for item in failures) else "FAIL",
        "registration_backend_call_count": 0,
        "formal_scientific_analysis_run_count": 0,
        "primary_estimand_recomputation_count": 0,
        "p_value_recomputation_count": 0,
        "raw_result_modification_count": 0,
        "frozen_analysis_result_modification_count": 0,
    }


__all__ = [
    "ReportingCorrectionVerificationError",
    "collect_observed",
    "validate_payloads",
    "verify_reporting_correction",
]
