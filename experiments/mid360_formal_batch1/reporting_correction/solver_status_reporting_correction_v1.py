"""Build the FMB1 solver-status reporting correction without scientific recomputation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


CORRECTION_SCHEMA = "fmb1_solver_status_reporting_correction_notice_v1"
CORRECTION_STATUS = "AUTHORITATIVE_VERSIONED_REPORTING_CORRECTION"
CORRECTION_TYPE = "VERSIONED_REPORTING_CORRECTION"
CORRECTION_SCOPE = "SOLVER_STATUS_ACCOUNTING_ONLY"

RAW_EXECUTION_COMMIT = "059e39533991d929a97ab208ad738643af82d09a"
LOCKED_ANALYSIS_CODE_COMMIT = "215a7961ed93dac9cf691e1e8ccd99d7ee868175"
LOCKED_ANALYSIS_RESULT_COMMIT = "2302f3fe6de804c934e2131b8fc77a49424bd83b"
LOCKED_ANALYSIS_RESULT_TAG = "results/fmb1-zero-perturbation-locked-analysis-v1"
SOLVER_DIAGNOSTIC_COMMIT = "a3ada7c345fefc728bdcf4b8a8b92465c974ef4e"
SOLVER_DIAGNOSTIC_TAG = "diagnostic/fmb1-solver-convergence-v1"
ORIGINAL_ANALYSIS_SUMMARY_SHA256 = (
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
IMPLEMENTATION_PATHS = (
    "experiments/mid360_formal_batch1/reporting_correction/__init__.py",
    "experiments/mid360_formal_batch1/reporting_correction/solver_status_reporting_correction_v1.py",
    "experiments/mid360_formal_batch1/reporting_correction/reporting_correction_schema_v1.json",
    "experiments/mid360_formal_batch1/reporting_correction/reporting_correction_verify_v1.py",
    "tools/mid360_formal_batch1/build_solver_status_reporting_correction_v1.py",
    "tools/mid360_formal_batch1/verify_solver_status_reporting_correction_v1.py",
)

BACKENDS = ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE")
OLD_SUCCESS_ALLOWLIST = ("CONVERGED", "SUCCESS")
ACTUAL_VALID_SOLVER_STATUS = "completed_finite_correspondences"

PAPER_WORDING = {
    "allowed_formal_status_statement": (
        "All 360 formal registration trials produced finite scientific outcomes and "
        "were recorded as COMPLETED under the frozen formal result protocol."
    ),
    "allowed_pcl_native_statement": (
        "For PCL, native convergence was reported for all 180 trials."
    ),
    "required_open3d_limitation_statement": (
        "The frozen Open3D wrapper did not preserve the native iteration count or "
        "native stopping criterion; therefore native Open3D convergence cannot be "
        "retrospectively determined."
    ),
    "forbidden_statements": [
        "All 360 trials converged natively.",
        "Open3D converged 180/180.",
    ],
}

SCIENTIFIC_SECTIONS = {
    "PRIMARY_TRANSLATION_VALUES_IDENTICAL": (
        "translation_station_summaries",
        "translation_scene_summaries",
        "primary_translation_inference",
    ),
    "PRIMARY_EXACT_P_VALUES_IDENTICAL": ("primary_translation_inference",),
    "ROTATION_VALUES_IDENTICAL": (
        "rotation_station_summaries",
        "rotation_scene_summaries",
        "secondary_rotation_inference",
    ),
    "CROSS_BACKEND_VALUES_IDENTICAL": (
        "cross_backend_scene_spearman",
        "cross_backend_station_spearman",
        "scene_ordering_agreement",
        "snapshot_direction_cosines",
    ),
    "REASSOCIATION_VALUES_IDENTICAL": (
        "reassociation_scene_summaries",
        "reassociation_scene_association",
        "reassociation_centered_association",
        "accepted_source_turnover_descriptive",
    ),
    "CENTERED_PERMUTATION_VALUES_IDENTICAL": (
        "reassociation_centered_permutation_sensitivity",
    ),
    "SYSTEMATIC_VALUES_IDENTICAL": (
        "systematic_station_values",
        "systematic_scene_values",
        "systematic_weak_rich_descriptive",
    ),
    "PHYSICAL_REFERENCE_SEMANTICS_IDENTICAL": ("physical_reference_limitation",),
    "CAPTURE_RADIUS_STATUS_IDENTICAL": ("capture_radius_status",),
}


class CorrectionBuildError(ValueError):
    """Raised when a reporting correction cannot be built fail-closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CorrectionBuildError(f"expected JSON object: {path}")
    return payload


def _write_new(path: Path, data: bytes) -> None:
    if path.exists():
        raise CorrectionBuildError(f"correction output collision: {path}")
    path.write_bytes(data)


def verify_sha256sums(directory: Path) -> dict[str, Any]:
    manifest = directory / "SHA256SUMS"
    failures: list[str] = []
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        target = directory / name.strip()
        count += 1
        if not target.is_file() or sha256_file(target) != expected:
            failures.append(name.strip())
    return {
        "status": "PASS" if not failures else "FAIL",
        "entry_count": count,
        "failures": failures,
        "manifest_sha256": sha256_file(manifest),
    }


def load_formal_rows(repository: Path) -> list[dict[str, Any]]:
    root = repository / FORMAL_ROWS_RELATIVE
    paths = sorted(root.glob("*/attempt-0001.json"))
    if len(paths) != 360:
        raise CorrectionBuildError(f"expected 360 formal rows, got {len(paths)}")
    rows = [_read_object(path) for path in paths]
    trial_ids = [str(row.get("trial_id")) for row in rows]
    if len(set(trial_ids)) != 360:
        raise CorrectionBuildError("formal trial IDs are not unique")
    return rows


def corrected_accounting(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if len(materialized) != 360:
        raise CorrectionBuildError("formal accounting requires 360 rows")
    per_backend: dict[str, dict[str, Any]] = {}
    for backend in BACKENDS:
        selected = [row for row in materialized if row.get("backend") == backend]
        statuses: dict[str, int] = {}
        for row in selected:
            status = str(row.get("solver_status"))
            statuses[status] = statuses.get(status, 0) + 1
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
    expected = {
        backend: {
            "planned_n": 180,
            "finite_result_n": 180,
            "formal_completed_n": 180,
            "formal_solver_nonconvergence_n": 0,
            "infrastructure_ok_n": 180,
            "solver_status_inventory": {ACTUAL_VALID_SOLVER_STATUS: 180},
        }
        for backend in BACKENDS
    }
    if per_backend != expected:
        raise CorrectionBuildError("formal row status inventory differs from diagnosis")
    return {
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
        "by_backend": per_backend,
    }


def correction_metadata(accounting: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "fmb1_analysis_summary_reporting_correction_metadata_v1",
        "status": CORRECTION_STATUS,
        "CORRECTION_TYPE": CORRECTION_TYPE,
        "CORRECTION_SCOPE": CORRECTION_SCOPE,
        "POSTHOC_DIAGNOSTIC_TRIGGERED": True,
        "SCIENTIFIC_ANALYSIS_RERUN": False,
        "REGISTRATION_RERUN": False,
        "RAW_POSE_RECOMPUTATION": False,
        "PRIMARY_ESTIMAND_RECOMPUTATION": False,
        "P_VALUE_RECOMPUTATION": False,
        "REASSOCIATION_RECOMPUTATION": False,
        "SYSTEMATIC_COMPONENT_RECOMPUTATION": False,
        "original_analysis_summary_sha256": ORIGINAL_ANALYSIS_SUMMARY_SHA256,
        "original_scientific_analysis_commit": LOCKED_ANALYSIS_RESULT_COMMIT,
        "original_scientific_analysis_tag": LOCKED_ANALYSIS_RESULT_TAG,
        "solver_diagnostic_commit": SOLVER_DIAGNOSTIC_COMMIT,
        "solver_diagnostic_tag": SOLVER_DIAGNOSTIC_TAG,
        "deprecated_reporting_field": {
            "field_name": "solver_nonconverged_finite_n",
            "deprecated": True,
            "original_reported_aggregate_value": 360,
            "status": "SUPERSEDED_REPORTING_FIELD",
            "superseded_by": "reporting_correction.corrected_accounting.formal_solver_nonconvergence_n",
        },
        "corrected_accounting": copy.deepcopy(dict(accounting)),
        "root_cause": "WRONG_SOLVER_STATUS_STRING_MAPPING",
        "old_success_allowlist": list(OLD_SUCCESS_ALLOWLIST),
        "actual_valid_solver_status": ACTUAL_VALID_SOLVER_STATUS,
        "paper_wording": copy.deepcopy(PAPER_WORDING),
    }


def corrected_summary(
    original: Mapping[str, Any], accounting: Mapping[str, Any]
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(original))
    if "reporting_correction" in payload:
        raise CorrectionBuildError("original summary already contains reporting correction")
    payload["reporting_correction"] = correction_metadata(accounting)
    return payload


def json_path_diff(original: Any, corrected: Any, path: str = "") -> list[dict[str, Any]]:
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
                rows.extend(json_path_diff(original[key], corrected[key], child))
        return rows
    if isinstance(original, list) and isinstance(corrected, list):
        rows = []
        maximum = max(len(original), len(corrected))
        for index in range(maximum):
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
                rows.extend(json_path_diff(original[index], corrected[index], child))
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


def scientific_immutability(
    original: Mapping[str, Any], corrected: Mapping[str, Any]
) -> dict[str, Any]:
    stripped = copy.deepcopy(dict(corrected))
    stripped.pop("reporting_correction", None)
    checks = {
        name: all(original.get(key) == stripped.get(key) for key in keys)
        for name, keys in SCIENTIFIC_SECTIONS.items()
    }
    checks["ALL_ORIGINAL_SUMMARY_CONTENT_IDENTICAL_AFTER_CORRECTION_METADATA_REMOVAL"] = (
        dict(original) == stripped
    )
    if not all(checks.values()):
        raise CorrectionBuildError("scientific-value immutability check failed")
    original_hash = hashlib.sha256(_canonical_json(dict(original))).hexdigest()
    stripped_hash = hashlib.sha256(_canonical_json(stripped)).hexdigest()
    return {
        "schema": "fmb1_scientific_value_immutability_check_v1",
        "status": "PASS",
        **checks,
        "original_semantic_json_sha256": original_hash,
        "corrected_without_reporting_metadata_semantic_json_sha256": stripped_hash,
        "PRIMARY_SCIENTIFIC_NUMERICAL_RESULTS_CHANGED": False,
        "new_hypothesis_tests_added": False,
    }


def correction_notice(accounting: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CORRECTION_SCHEMA,
        "status": CORRECTION_STATUS,
        "CORRECTION_TYPE": CORRECTION_TYPE,
        "CORRECTION_SCOPE": CORRECTION_SCOPE,
        "POSTHOC_DIAGNOSTIC_TRIGGERED": True,
        "SCIENTIFIC_ANALYSIS_RERUN": False,
        "REGISTRATION_RERUN": False,
        "RAW_POSE_RECOMPUTATION": False,
        "PRIMARY_ESTIMAND_RECOMPUTATION": False,
        "P_VALUE_RECOMPUTATION": False,
        "REASSOCIATION_RECOMPUTATION": False,
        "SYSTEMATIC_COMPONENT_RECOMPUTATION": False,
        "ORIGINAL_REPORTED_SOLVER_NONCONVERGED_FINITE_N": 360,
        "CORRECTED_FORMAL_COMPLETED_N": 360,
        "CORRECTED_FORMAL_SOLVER_NONCONVERGENCE_N": 0,
        "OLD_SUCCESS_ALLOWLIST": list(OLD_SUCCESS_ALLOWLIST),
        "ACTUAL_VALID_SOLVER_STATUS": ACTUAL_VALID_SOLVER_STATUS,
        "ROOT_CAUSE": "WRONG_SOLVER_STATUS_STRING_MAPPING",
        "old_value_status": "SUPERSEDED_REPORTING_FIELD",
        "corrected_accounting": copy.deepcopy(dict(accounting)),
        "paper_wording": copy.deepcopy(PAPER_WORDING),
    }


def _scientific_references(original: Mapping[str, Any]) -> dict[str, Any]:
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


def _notice_markdown() -> str:
    return """# FMB1 solver-status reporting correction v1

This is an authoritative versioned reporting correction, not a second scientific analysis.

- Original reported aggregate `solver_nonconverged_finite_n`: **360** (superseded reporting field).
- Corrected formal `COMPLETED`: **360**.
- Corrected formal `SOLVER_NON_CONVERGENCE`: **0**.
- PCL native `has_converged=true`: **180/180**.
- Open3D native convergence: **not retrospectively observable**.

Root cause: the old reporting allowlist accepted only `CONVERGED` and `SUCCESS`, while every valid formal row used `completed_finite_correspondences`.

No registration, scientific analysis, estimand, p-value, reassociation, systematic-component, or raw-pose recomputation was performed.

## Permitted paper wording

All 360 formal registration trials produced finite scientific outcomes and were recorded as COMPLETED under the frozen formal result protocol.

For PCL, native convergence was reported for all 180 trials.

The frozen Open3D wrapper did not preserve the native iteration count or native stopping criterion; therefore native Open3D convergence cannot be retrospectively determined.
"""


def _corrected_summary_markdown(references: Mapping[str, Any]) -> str:
    primary = references["primary_translation"]
    centered = references["reassociation_centered"]
    centered_p = references["reassociation_centered_permutation"]
    return f"""# FMB1 locked analysis — reporting-corrected view v1

Scientific values remain sourced byte-for-value from the original frozen analysis summary. This file changes solver-status reporting only.

## Corrected solver accounting

- Formal finite outcomes: 360/360.
- Formal `COMPLETED`: 360/360.
- Formal `SOLVER_NON_CONVERGENCE`: 0/360.
- PCL native `has_converged=true`: 180/180.
- Open3D native stopping convergence: not retrospectively observable.

## Original frozen scientific-result references

- Open3D primary estimand: `{primary['OPEN3D_POINT_TO_PLANE']['estimand_m']}` m; exact p `{primary['OPEN3D_POINT_TO_PLANE']['exact_p_value']}`.
- PCL primary estimand: `{primary['PCL_POINT_TO_PLANE']['estimand_m']}` m; exact p `{primary['PCL_POINT_TO_PLANE']['exact_p_value']}`.
- Open3D centered reassociation rho: `{centered['OPEN3D_POINT_TO_PLANE']['rho']}`; p `{centered_p['OPEN3D_POINT_TO_PLANE']['p_value']}`.
- PCL centered reassociation rho: `{centered['PCL_POINT_TO_PLANE']['rho']}`; p `{centered_p['PCL_POINT_TO_PLANE']['p_value']}`.

These values were copied from the original frozen `analysis_summary.json`; no scientific endpoint was recalculated.
"""


def build_reporting_correction(
    repository: str | Path,
    output_dir: str | Path,
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    repository_path = Path(repository).resolve(strict=True)
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise CorrectionBuildError("reporting correction output must be absent or empty")
    destination.mkdir(parents=True, exist_ok=True)

    original_path = repository_path / ORIGINAL_SUMMARY_RELATIVE
    if sha256_file(original_path) != ORIGINAL_ANALYSIS_SUMMARY_SHA256:
        raise CorrectionBuildError("original analysis summary SHA mismatch")
    original = _read_object(original_path)
    rows = load_formal_rows(repository_path)
    accounting = corrected_accounting(rows)
    notice = correction_notice(accounting)
    corrected = corrected_summary(original, accounting)
    diff_rows = json_path_diff(original, corrected)
    if [row["json_path"] for row in diff_rows] != ["/reporting_correction"]:
        raise CorrectionBuildError("corrected summary changed outside reporting metadata")
    immutability = scientific_immutability(original, corrected)
    references = _scientific_references(original)

    diagnostic = verify_sha256sums(repository_path / DIAGNOSTIC_RELATIVE)
    if diagnostic["status"] != "PASS":
        raise CorrectionBuildError("solver diagnostic SHA256SUMS failed")

    generated = generated_at_utc or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    files: dict[str, bytes] = {
        "reporting_correction_notice_v1.json": _canonical_json(notice),
        "reporting_correction_notice_v1.md": _notice_markdown().encode("utf-8"),
        "corrected_solver_status_accounting_v1.json": _canonical_json(accounting),
        "analysis_summary_reporting_corrected_v1.json": _canonical_json(corrected),
        "analysis_summary_reporting_corrected_v1.md": _corrected_summary_markdown(
            references
        ).encode("utf-8"),
        "scientific_value_immutability_check.json": _canonical_json(immutability),
        "json_path_correction_diff.json": _canonical_json(
            {
                "schema": "fmb1_reporting_correction_json_path_diff_v1",
                "status": "PASS",
                "allowed_path_prefixes": ["/reporting_correction"],
                "changed_path_count": len(diff_rows),
                "forbidden_scientific_path_change_count": 0,
                "changes": diff_rows,
            }
        ),
    }
    csv_rows = []
    for backend in BACKENDS:
        item = accounting["by_backend"][backend]
        csv_rows.append(
            {
                "backend": backend,
                "planned_n": item["planned_n"],
                "finite_result_n": item["finite_result_n"],
                "formal_completed_n": item["formal_completed_n"],
                "formal_solver_nonconvergence_n": item[
                    "formal_solver_nonconvergence_n"
                ],
                "infrastructure_ok_n": item["infrastructure_ok_n"],
                "solver_status": ACTUAL_VALID_SOLVER_STATUS,
                "solver_status_count": 180,
                "native_convergence_observable": (
                    backend == "PCL_POINT_TO_PLANE"
                ),
                "native_has_converged_true_n": (
                    180 if backend == "PCL_POINT_TO_PLANE" else "UNDETERMINABLE"
                ),
                "native_has_converged_false_n": (
                    0 if backend == "PCL_POINT_TO_PLANE" else "UNDETERMINABLE"
                ),
            }
        )
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(csv_rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(csv_rows)
    files["corrected_solver_status_accounting_v1.csv"] = stream.getvalue().encode(
        "utf-8"
    )

    for name, data in files.items():
        _write_new(destination / name, data)

    artifact_hashes = {name: sha256_file(destination / name) for name in sorted(files)}
    manifest = {
        "schema": "fmb1_solver_status_reporting_correction_manifest_v1",
        "status": CORRECTION_STATUS,
        "generated_at_utc": generated,
        "CORRECTION_TYPE": CORRECTION_TYPE,
        "CORRECTION_SCOPE": CORRECTION_SCOPE,
        "original_scientific_analysis": {
            "commit": LOCKED_ANALYSIS_RESULT_COMMIT,
            "tag": LOCKED_ANALYSIS_RESULT_TAG,
            "analysis_summary_path": ORIGINAL_SUMMARY_RELATIVE,
            "analysis_summary_sha256": ORIGINAL_ANALYSIS_SUMMARY_SHA256,
            "role": "ORIGINAL_FROZEN_SCIENTIFIC_ANALYSIS",
        },
        "solver_diagnostic": {
            "commit": SOLVER_DIAGNOSTIC_COMMIT,
            "tag": SOLVER_DIAGNOSTIC_TAG,
            "sha256sums_status": diagnostic["status"],
            "sha256sums_sha256": diagnostic["manifest_sha256"],
        },
        "raw_execution_commit": RAW_EXECUTION_COMMIT,
        "locked_analysis_code_commit": LOCKED_ANALYSIS_CODE_COMMIT,
        "corrected_accounting": accounting,
        "scientific_value_references": references,
        "scientific_value_immutability_status": "PASS",
        "json_path_diff_status": "PASS",
        "forbidden_scientific_path_change_count": 0,
        "registration_backend_call_count": 0,
        "formal_scientific_analysis_run_count": 0,
        "primary_estimand_recomputation_count": 0,
        "p_value_recomputation_count": 0,
        "raw_result_modification_count": 0,
        "frozen_analysis_result_modification_count": 0,
        "corrected_reporting_artifact_count": len(files) + 1,
        "implementation_artifact_sha256": {
            relative: sha256_file(repository_path / relative)
            for relative in IMPLEMENTATION_PATHS
        },
        "core_artifact_sha256": artifact_hashes,
        "independent_verification_expected_path": (
            "independent_reporting_correction_verification.json"
        ),
        "role": "AUTHORITATIVE_VERSIONED_REPORTING_CORRECTION",
        "not_a_new_scientific_experiment": True,
    }
    _write_new(
        destination / "reporting_correction_manifest.json", _canonical_json(manifest)
    )
    return manifest


__all__ = [
    "CorrectionBuildError",
    "build_reporting_correction",
    "corrected_accounting",
    "corrected_summary",
    "json_path_diff",
    "scientific_immutability",
    "sha256_file",
]
