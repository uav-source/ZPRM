"""Frozen scientific-equivalence comparator for fresh and resumed fixture rounds."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .phase_a_trial_result_schema import OPEN3D_BACKEND, PCL_BACKEND


CONTRACT_RELATIVE_PATH = Path(
    "configs/zero_perturbation/resume_scientific_equivalence_v1.yaml"
)
CONTRACT_SCHEMA_VERSION = "phase_a_resume_scientific_equivalence_contract_v1"
REPORT_SCHEMA_VERSION = "phase_a_resume_scientific_equivalence_report_v1"
ABSOLUTE_TOLERANCE = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-12

COMPARISON_CLASSES = frozenset(
    {
        "EXACT",
        "NUMERIC_TOLERANCE",
        "IGNORED_RUNTIME_METADATA",
        "NULL_CONSISTENCY",
        "FAILURE_DETAIL_PRESENCE",
    }
)
COMMON_EXACT_FIELDS = (
    "schema_version",
    "planned_trial_id",
    "snapshot_id",
    "scene_variant",
    "condition",
    "backend",
    "protocol_sha256",
    "snapshot_lock_sha256",
    "snapshot_checksum",
    "source_checksum",
    "target_checksum",
    "reference_pose_checksum",
    "implementation_sha256",
    "solver_failure",
    "failure_classification",
    "raw_rotation_finite",
    "finite_output",
)
COMMON_NUMERIC_FIELDS = (
    "translation_update_m",
    "rotation_update_rad",
    "raw_rotation_determinant",
    "orthogonality_defect_fro",
    "projection_correction_fro",
)
OPEN3D_EXACT_DIAGNOSTICS = ("correspondence_set_size",)
OPEN3D_NUMERIC_DIAGNOSTICS = ("fitness", "inlier_rmse")
PCL_EXACT_DIAGNOSTICS = (
    "pcl_version",
    "pcl_cli_sha256",
    "exit_code",
    "has_converged_raw",
    "iteration_count",
    "correspondence_count",
)
PCL_NUMERIC_DIAGNOSTICS = ("fitness_score",)
NORMAL_EXACT_FIELDS = ("finite_count", "zero_count", "nan_count")
NORMAL_NUMERIC_FIELDS = ("norm_min", "norm_median", "norm_max")
IGNORED_RUNTIME_FIELDS = (
    "runtime_ms",
    "timestamp_utc",
    "attempt_event_timestamp",
    "temporary_directory_path",
    "process_id",
    "thread_id",
    "execution_order",
    "started_resumed_event_time_difference",
)


class ScientificEquivalenceContractError(ValueError):
    """Raised when the frozen comparator contract itself is invalid."""


def load_equivalence_contract(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    value = yaml.safe_load((repository / CONTRACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    if type(value) is not dict or value.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ScientificEquivalenceContractError("unknown scientific-equivalence contract")
    numeric = value.get("numeric_tolerance", {})
    if (
        numeric.get("absolute_tolerance") != ABSOLUTE_TOLERANCE
        or numeric.get("relative_tolerance") != RELATIVE_TOLERANCE
        or numeric.get("finite_values_only") is not True
        or numeric.get("rounding_before_comparison_forbidden") is not True
        or numeric.get("storage_precision_change_forbidden") is not True
    ):
        raise ScientificEquivalenceContractError("numeric tolerance contract changed")
    if value.get("exact_fields", {}).get("catalog_field_count") != 30:
        raise ScientificEquivalenceContractError("exact field catalog changed")
    if value.get("numeric_fields", {}).get("named_scalar_field_count") != 14:
        raise ScientificEquivalenceContractError("numeric field catalog changed")
    if value.get("numeric_fields", {}).get("matrix", {}).get("element_count") != 16:
        raise ScientificEquivalenceContractError("matrix element contract changed")
    if tuple(value.get("comparison_classes", ())) != tuple(
        [
            "EXACT",
            "NUMERIC_TOLERANCE",
            "IGNORED_RUNTIME_METADATA",
            "NULL_CONSISTENCY",
            "FAILURE_DETAIL_PRESENCE",
        ]
    ):
        raise ScientificEquivalenceContractError("comparison classes changed")
    ratios = value.get("tolerance_negligibility", {})
    if (
        ratios.get("translation_qualification_threshold_m") != 1.0e-3
        or ratios.get("rotation_qualification_threshold_rad")
        != 1.7453292519943296e-4
        or float(ratios.get("translation_threshold_to_atol_ratio", 0.0)) != 1.0e9
        or not math.isclose(
            float(ratios.get("rotation_threshold_to_atol_ratio", 0.0)),
            174532925.19943297,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ):
        raise ScientificEquivalenceContractError("scientific Gate ratio changed")
    return value


def _difference_row(
    *,
    planned_trial_id: str,
    path: str,
    fresh: Any,
    resumed: Any,
    comparison_class: str,
    passed: bool,
    absolute_difference: float | None = None,
    relative_difference: float | None = None,
    allowed_tolerance: float | None = None,
) -> dict[str, Any]:
    if comparison_class not in COMPARISON_CLASSES:
        raise ValueError(f"unknown comparison class: {comparison_class}")
    return {
        "absolute_difference": absolute_difference,
        "allowed_tolerance": allowed_tolerance,
        "comparison_class": comparison_class,
        "fresh_value": fresh,
        "json_field_path": path,
        "passed": bool(passed),
        "planned_trial_id": planned_trial_id,
        "relative_difference": relative_difference,
        "resumed_value": resumed,
    }


def compare_exact_field(
    fresh: Any,
    resumed: Any,
    *,
    planned_trial_id: str,
    path: str,
) -> list[dict[str, Any]]:
    if fresh == resumed and type(fresh) is type(resumed):
        return []
    return [
        _difference_row(
            planned_trial_id=planned_trial_id,
            path=path,
            fresh=fresh,
            resumed=resumed,
            comparison_class="EXACT",
            passed=False,
        )
    ]


def compare_optional_float(
    fresh: Any,
    resumed: Any,
    *,
    planned_trial_id: str,
    path: str,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
    relative_tolerance: float = RELATIVE_TOLERANCE,
) -> list[dict[str, Any]]:
    if fresh is None or resumed is None:
        if fresh is None and resumed is None:
            return []
        return [
            _difference_row(
                planned_trial_id=planned_trial_id,
                path=path,
                fresh=fresh,
                resumed=resumed,
                comparison_class="NULL_CONSISTENCY",
                passed=False,
            )
        ]
    if (
        isinstance(fresh, bool)
        or isinstance(resumed, bool)
        or not isinstance(fresh, (int, float))
        or not isinstance(resumed, (int, float))
    ):
        return [
            _difference_row(
                planned_trial_id=planned_trial_id,
                path=path,
                fresh=fresh,
                resumed=resumed,
                comparison_class="NUMERIC_TOLERANCE",
                passed=False,
            )
        ]
    a, b = float(fresh), float(resumed)
    if not math.isfinite(a) or not math.isfinite(b):
        return [
            _difference_row(
                planned_trial_id=planned_trial_id,
                path=path,
                fresh=fresh,
                resumed=resumed,
                comparison_class="NUMERIC_TOLERANCE",
                passed=False,
            )
        ]
    difference = abs(a - b)
    scale = max(abs(a), abs(b))
    allowed = absolute_tolerance + relative_tolerance * scale
    relative = difference / scale if scale > 0.0 else 0.0
    if difference == 0.0:
        return []
    return [
        _difference_row(
            planned_trial_id=planned_trial_id,
            path=path,
            fresh=fresh,
            resumed=resumed,
            absolute_difference=difference,
            relative_difference=relative,
            allowed_tolerance=allowed,
            comparison_class="NUMERIC_TOLERANCE",
            passed=difference <= allowed,
        )
    ]


def compare_float_matrix(
    fresh: Any,
    resumed: Any,
    *,
    planned_trial_id: str,
    path: str = "final_transform_4x4",
) -> list[dict[str, Any]]:
    if not (
        isinstance(fresh, list)
        and isinstance(resumed, list)
        and len(fresh) == len(resumed) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in fresh)
        and all(isinstance(row, list) and len(row) == 4 for row in resumed)
    ):
        return [
            _difference_row(
                planned_trial_id=planned_trial_id,
                path=path,
                fresh=fresh,
                resumed=resumed,
                comparison_class="NUMERIC_TOLERANCE",
                passed=False,
            )
        ]
    rows: list[dict[str, Any]] = []
    for row_index in range(4):
        for column_index in range(4):
            rows.extend(
                compare_optional_float(
                    fresh[row_index][column_index],
                    resumed[row_index][column_index],
                    planned_trial_id=planned_trial_id,
                    path=f"{path}[{row_index}][{column_index}]",
                )
            )
    return rows


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def compare_backend_diagnostics(
    fresh: Any,
    resumed: Any,
    *,
    backend: str,
    planned_trial_id: str,
) -> list[dict[str, Any]]:
    first, second = _mapping(fresh), _mapping(resumed)
    rows: list[dict[str, Any]] = []
    if backend == OPEN3D_BACKEND:
        for field in OPEN3D_EXACT_DIAGNOSTICS:
            rows.extend(
                compare_exact_field(
                    first.get(field),
                    second.get(field),
                    planned_trial_id=planned_trial_id,
                    path=f"backend_diagnostics.{field}",
                )
            )
        for field in OPEN3D_NUMERIC_DIAGNOSTICS:
            rows.extend(
                compare_optional_float(
                    first.get(field),
                    second.get(field),
                    planned_trial_id=planned_trial_id,
                    path=f"backend_diagnostics.{field}",
                )
            )
        return rows
    if backend != PCL_BACKEND:
        return compare_exact_field(
            fresh,
            resumed,
            planned_trial_id=planned_trial_id,
            path="backend_diagnostics",
        )
    for field in PCL_EXACT_DIAGNOSTICS:
        rows.extend(
            compare_exact_field(
                first.get(field),
                second.get(field),
                planned_trial_id=planned_trial_id,
                path=f"backend_diagnostics.{field}",
            )
        )
    for field in PCL_NUMERIC_DIAGNOSTICS:
        rows.extend(
            compare_optional_float(
                first.get(field),
                second.get(field),
                planned_trial_id=planned_trial_id,
                path=f"backend_diagnostics.{field}",
            )
        )
    for normal_name in ("source_normal_statistics", "target_normal_statistics"):
        first_normal = _mapping(first.get(normal_name))
        second_normal = _mapping(second.get(normal_name))
        for field in NORMAL_EXACT_FIELDS:
            rows.extend(
                compare_exact_field(
                    first_normal.get(field),
                    second_normal.get(field),
                    planned_trial_id=planned_trial_id,
                    path=f"backend_diagnostics.{normal_name}.{field}",
                )
            )
        for field in NORMAL_NUMERIC_FIELDS:
            rows.extend(
                compare_optional_float(
                    first_normal.get(field),
                    second_normal.get(field),
                    planned_trial_id=planned_trial_id,
                    path=f"backend_diagnostics.{normal_name}.{field}",
                )
            )
    return rows


def _failure_detail_rows(
    fresh: Mapping[str, Any], resumed: Mapping[str, Any], planned_trial_id: str
) -> list[dict[str, Any]]:
    first, second = fresh.get("failure_detail"), resumed.get("failure_detail")
    failed = fresh.get("solver_failure") is True and resumed.get("solver_failure") is True
    if failed:
        presence_pass = (
            isinstance(first, str)
            and bool(first.strip())
            and isinstance(second, str)
            and bool(second.strip())
        )
    else:
        presence_pass = first is None and second is None
    if presence_pass and first == second:
        return []
    return [
        _difference_row(
            planned_trial_id=planned_trial_id,
            path="failure_detail",
            fresh=first,
            resumed=second,
            comparison_class="FAILURE_DETAIL_PRESENCE",
            passed=presence_pass,
        )
    ]


def compare_trial_results(
    fresh: Mapping[str, Any], resumed: Mapping[str, Any]
) -> list[dict[str, Any]]:
    planned_trial_id = str(
        fresh.get("planned_trial_id") or resumed.get("planned_trial_id") or "<missing>"
    )
    rows: list[dict[str, Any]] = []
    for field in COMMON_EXACT_FIELDS:
        rows.extend(
            compare_exact_field(
                fresh.get(field),
                resumed.get(field),
                planned_trial_id=planned_trial_id,
                path=field,
            )
        )
    for field in COMMON_NUMERIC_FIELDS:
        rows.extend(
            compare_optional_float(
                fresh.get(field),
                resumed.get(field),
                planned_trial_id=planned_trial_id,
                path=field,
            )
        )
    rows.extend(
        compare_float_matrix(
            fresh.get("final_transform_4x4"),
            resumed.get("final_transform_4x4"),
            planned_trial_id=planned_trial_id,
        )
    )
    backend = str(fresh.get("backend") or resumed.get("backend") or "")
    rows.extend(
        compare_backend_diagnostics(
            fresh.get("backend_diagnostics"),
            resumed.get("backend_diagnostics"),
            backend=backend,
            planned_trial_id=planned_trial_id,
        )
    )
    rows.extend(_failure_detail_rows(fresh, resumed, planned_trial_id))
    for field in IGNORED_RUNTIME_FIELDS:
        if field in fresh or field in resumed:
            if fresh.get(field) != resumed.get(field):
                rows.append(
                    _difference_row(
                        planned_trial_id=planned_trial_id,
                        path=field,
                        fresh=fresh.get(field),
                        resumed=resumed.get(field),
                        comparison_class="IGNORED_RUNTIME_METADATA",
                        passed=True,
                    )
                )
    return rows


def _indexed_by_backend(rows: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, Sequence):
        return {}
    return {
        str(row.get("backend")): row
        for row in rows
        if isinstance(row, Mapping) and "backend" in row
    }


def _compare_summary_rows(
    fresh: Any,
    resumed: Any,
    *,
    path: str,
    ignored_runtime: bool = False,
) -> list[dict[str, Any]]:
    first, second = _indexed_by_backend(fresh), _indexed_by_backend(resumed)
    rows = compare_exact_field(
        sorted(first), sorted(second), planned_trial_id="<analysis>", path=f"{path}.backends"
    )
    for backend in sorted(set(first) | set(second)):
        a, b = first.get(backend, {}), second.get(backend, {})
        rows.extend(
            compare_exact_field(
                a.get("count"),
                b.get("count"),
                planned_trial_id="<analysis>",
                path=f"{path}.{backend}.count",
            )
        )
        for field in ("median", "q95", "q95_linear", "maximum", "mean"):
            if field not in a and field not in b:
                continue
            comparison = compare_optional_float(
                a.get(field),
                b.get(field),
                planned_trial_id="<analysis>",
                path=f"{path}.{backend}.{field}",
            )
            if ignored_runtime:
                for item in comparison:
                    item["comparison_class"] = "IGNORED_RUNTIME_METADATA"
                    item["passed"] = True
            rows.extend(comparison)
    return rows


def compare_analysis_outputs(
    fresh: Mapping[str, Any], resumed: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in (
        "trial_count",
        "trial_ids",
        "snapshot_ids",
        "backend_inventory",
        "failure_inventory",
        "input_pairing_audit",
        "decision",
    ):
        rows.extend(
            compare_exact_field(
                fresh.get(field),
                resumed.get(field),
                planned_trial_id="<analysis>",
                path=field,
            )
        )
    rows.extend(
        _compare_summary_rows(
            fresh.get("translation_summary"),
            resumed.get("translation_summary"),
            path="translation_summary",
        )
    )
    rows.extend(
        _compare_summary_rows(
            fresh.get("rotation_summary"),
            resumed.get("rotation_summary"),
            path="rotation_summary",
        )
    )
    rows.extend(
        _compare_summary_rows(
            fresh.get("runtime_summary"),
            resumed.get("runtime_summary"),
            path="runtime_summary",
            ignored_runtime=True,
        )
    )
    if fresh.get("attempt_event_count") != resumed.get("attempt_event_count"):
        rows.append(
            _difference_row(
                planned_trial_id="<analysis>",
                path="attempt_event_count",
                fresh=fresh.get("attempt_event_count"),
                resumed=resumed.get("attempt_event_count"),
                comparison_class="IGNORED_RUNTIME_METADATA",
                passed=True,
            )
        )
    return rows


def compare_final_decisions(
    fresh: Mapping[str, Any], resumed: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return compare_exact_field(
        dict(fresh),
        dict(resumed),
        planned_trial_id="<final-decision>",
        path="final_decision",
    )


def _maximum_numeric_difference(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    numeric = [
        row
        for row in rows
        if row.get("comparison_class") == "NUMERIC_TOLERANCE"
        and isinstance(row.get("absolute_difference"), (int, float))
    ]
    if not numeric:
        return {
            "absolute_difference": 0.0,
            "allowed_tolerance": ABSOLUTE_TOLERANCE,
            "json_field_path": None,
            "planned_trial_id": None,
            "relative_difference": 0.0,
        }
    maximum = max(numeric, key=lambda row: float(row["absolute_difference"]))
    return {
        name: maximum.get(name)
        for name in (
            "absolute_difference",
            "allowed_tolerance",
            "json_field_path",
            "planned_trial_id",
            "relative_difference",
        )
    }


def _maximum_relative_difference(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    numeric = [
        row
        for row in rows
        if row.get("comparison_class") == "NUMERIC_TOLERANCE"
        and isinstance(row.get("relative_difference"), (int, float))
    ]
    if not numeric:
        return {
            "absolute_difference": 0.0,
            "allowed_tolerance": ABSOLUTE_TOLERANCE,
            "json_field_path": None,
            "planned_trial_id": None,
            "relative_difference": 0.0,
        }
    maximum = max(numeric, key=lambda row: float(row["relative_difference"]))
    return {
        name: maximum.get(name)
        for name in (
            "absolute_difference",
            "allowed_tolerance",
            "json_field_path",
            "planned_trial_id",
            "relative_difference",
        )
    }


def build_equivalence_report(
    *,
    root: str | Path,
    fresh_results: Mapping[str, Mapping[str, Any]],
    resumed_results: Mapping[str, Mapping[str, Any]],
    fresh_analysis: Mapping[str, Any],
    resumed_analysis: Mapping[str, Any],
    fresh_final_decision: Mapping[str, Any],
    resumed_final_decision: Mapping[str, Any],
    resume_skipped_valid_result_count: int,
    resume_reexecuted_valid_result_count: int,
) -> dict[str, Any]:
    contract = load_equivalence_contract(root)
    rows: list[dict[str, Any]] = []
    trial_ids = sorted(set(fresh_results) | set(resumed_results))
    for trial_id in trial_ids:
        if trial_id not in fresh_results or trial_id not in resumed_results:
            rows.extend(
                compare_exact_field(
                    fresh_results.get(trial_id),
                    resumed_results.get(trial_id),
                    planned_trial_id=trial_id,
                    path="trial_presence",
                )
            )
            continue
        rows.extend(compare_trial_results(fresh_results[trial_id], resumed_results[trial_id]))
    analysis_rows = compare_analysis_outputs(fresh_analysis, resumed_analysis)
    decision_rows = compare_final_decisions(fresh_final_decision, resumed_final_decision)
    rows.extend(analysis_rows)
    rows.extend(decision_rows)

    exact_failures = [
        row for row in rows if row["comparison_class"] == "EXACT" and not row["passed"]
    ]
    numeric_rows = [row for row in rows if row["comparison_class"] == "NUMERIC_TOLERANCE"]
    numeric_outside = [row for row in numeric_rows if not row["passed"]]
    numeric_inside = [row for row in numeric_rows if row["passed"]]
    null_failures = [
        row
        for row in rows
        if row["comparison_class"] == "NULL_CONSISTENCY" and not row["passed"]
    ]
    failure_detail_failures = [
        row
        for row in rows
        if row["comparison_class"] == "FAILURE_DETAIL_PRESENCE" and not row["passed"]
    ]
    failure_detail_text_differences = [
        row
        for row in rows
        if row["comparison_class"] == "FAILURE_DETAIL_PRESENCE"
        and row["passed"]
        and row["fresh_value"] != row["resumed_value"]
    ]
    runtime_rows = [
        row for row in rows if row["comparison_class"] == "IGNORED_RUNTIME_METADATA"
    ]
    classification_mismatches = [
        row
        for row in exact_failures
        if row["json_field_path"] == "failure_classification"
    ]
    gate_mismatches = [
        row
        for row in analysis_rows
        if not row["passed"] and row["json_field_path"] == "decision"
    ]
    final_mismatches = [row for row in decision_rows if not row["passed"]]
    expected_counts = contract["resume_flow_counts"]
    count_pass = (
        len(fresh_results) == expected_counts["fresh_completed_trials"]
        and len(resumed_results) == expected_counts["resumed_completed_trials"]
        and resume_skipped_valid_result_count
        == expected_counts["resume_skipped_valid_results"]
        and resume_reexecuted_valid_result_count
        == expected_counts["resume_reexecuted_valid_results"]
    )
    gates = {
        "FRESH_AND_RESUMED_FINAL_DECISION_IDENTICAL": not final_mismatches,
        "FRESH_AND_RESUMED_GATE_DECISIONS_IDENTICAL": not gate_mismatches,
        "NUMERIC_EQUIVALENCE_TOLERANCE_NEGLIGIBLE_RELATIVE_TO_SCIENTIFIC_GATES": True,
        "RESUME_EQUIVALENCE_CONTRACT_PASS": True,
        "RESUME_EXACT_FIELDS_PASS": not exact_failures,
        "RESUME_FAILURE_CLASSIFICATION_PASS": not classification_mismatches,
        "RESUME_FINAL_DECISION_IDENTICAL": not final_mismatches,
        "RESUME_GATE_DECISIONS_IDENTICAL": not gate_mismatches,
        "RESUME_NULL_CONSISTENCY_PASS": not null_failures,
        "RESUME_NUMERIC_FIELDS_WITHIN_TOLERANCE": not numeric_outside,
        "RESUME_SCIENTIFIC_EQUIVALENCE_PASS": bool(
            count_pass
            and not exact_failures
            and not numeric_outside
            and not null_failures
            and not failure_detail_failures
            and not classification_mismatches
            and not gate_mismatches
            and not final_mismatches
        ),
        "RESUME_STRICT_FLOW_COUNTS_PASS": count_pass,
    }
    return {
        **gates,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "difference_inventory": rows,
        "exact_field_mismatch_count": len(exact_failures),
        "failure_classification_mismatch_count": len(classification_mismatches),
        "failure_detail_presence_violation_count": len(failure_detail_failures),
        "failure_detail_text_difference_count": len(failure_detail_text_differences),
        "final_decision_mismatch_count": len(final_mismatches),
        "fresh_trial_count": len(fresh_results),
        "gate_decision_mismatch_count": len(gate_mismatches),
        "maximum_absolute_numeric_difference": _maximum_numeric_difference(rows),
        "maximum_relative_numeric_difference": _maximum_relative_difference(rows),
        "null_consistency_violation_count": len(null_failures),
        "numeric_difference_inside_tolerance_count": len(numeric_inside),
        "numeric_difference_outside_tolerance_count": len(numeric_outside),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "resume_reexecuted_valid_result_count": resume_reexecuted_valid_result_count,
        "resume_skipped_valid_result_count": resume_skipped_valid_result_count,
        "resumed_trial_count": len(resumed_results),
        "runtime_metadata_difference_count": len(runtime_rows),
        "schema_version": REPORT_SCHEMA_VERSION,
    }


__all__ = [
    "ABSOLUTE_TOLERANCE",
    "RELATIVE_TOLERANCE",
    "build_equivalence_report",
    "compare_analysis_outputs",
    "compare_backend_diagnostics",
    "compare_exact_field",
    "compare_final_decisions",
    "compare_float_matrix",
    "compare_optional_float",
    "compare_trial_results",
    "load_equivalence_contract",
]
