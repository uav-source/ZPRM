"""Standalone verifier for the compact Full Synthetic Development artifact."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .contracts import file_sha256, write_json


TABLES = (
    "protocol_summary.csv",
    "snapshot_inventory.csv",
    "trial_results.csv",
    "failure_inventory.csv",
    "scene_condition_backend_summary.csv",
    "geometry_seed_raw_values.csv",
    "systematic_offset_summary.csv",
    "repeatability_summary.csv",
    "direction_concentration_summary.csv",
    "common_association_metrics.csv",
    "local_geometry_metrics.csv",
    "scene_effect_paired.csv",
    "cross_backend_ranking.csv",
    "scene_rank_stability.csv",
    "condition_contrasts.csv",
    "turnover_correlations.csv",
    "ridge_model_comparison.csv",
    "automatic_nonequivalence_candidates.csv",
    "runtime_summary.csv",
    "gate_summary.csv",
)
FIGURES = (
    "scene_condition_translation_error.png",
    "scene_condition_rotation_error.png",
    "corridor_rich_paired.png",
    "cross_backend_scene_ranking.png",
    "systematic_offset_vs_repeatability.png",
    "direction_concentration.png",
    "condition_effects.png",
    "turnover_vs_translation_error.png",
    "local_metric_vs_error.png",
    "ridge_model_comparison.png",
    "runtime_comparison.png",
)
ROOT_FILES = (
    "full_synthetic_development_report.md",
    "primary_analysis.json",
    "independent_verification.json",
    "final_decision.json",
    "run_manifest.json",
    "SHA256SUMS",
    "artifact_verification.json",
)
REQUIRED_FILES = tuple(f"tables/{name}" for name in TABLES) + tuple(
    f"figures/{name}" for name in FIGURES
) + ROOT_FILES
SHA256_EXCLUDED_FILES = frozenset({"SHA256SUMS", "artifact_verification.json"})
REQUIRED_DEVELOPMENT_GATES = (
    "FULL_SYNTHETIC_DEVELOPMENT_ENGINEERING_PASS",
    "FULL_SYNTHETIC_EXECUTION_ROBUSTNESS_PASS",
    "FULL_SYNTHETIC_PRIMARY_SCENE_EFFECT_PASS",
    "FULL_SYNTHETIC_CROSS_BACKEND_PASS",
    "FULL_SYNTHETIC_SCENE_RANK_STABILITY_PASS",
    "COMMON_ASSOCIATION_ANALYSIS_PASS",
    "REASSOCIATION_MECHANISM_SUPPORTED",
    "LOCAL_METRIC_INCREMENTAL_VALUE_PASS",
)

FIXED_ROW_COUNTS = {
    "tables/protocol_summary.csv": 10,
    "tables/snapshot_inventory.csv": 1260,
    "tables/trial_results.csv": 2520,
    "tables/scene_condition_backend_summary.csv": 84,
    "tables/geometry_seed_raw_values.csv": 2520,
    "tables/systematic_offset_summary.csv": 252,
    "tables/repeatability_summary.csv": 252,
    "tables/direction_concentration_summary.csv": 252,
    "tables/scene_effect_paired.csv": 4,
    "tables/cross_backend_ranking.csv": 6,
    "tables/scene_rank_stability.csv": 70,
    "tables/condition_contrasts.csv": 56,
    "tables/turnover_correlations.csv": 2,
    "tables/ridge_model_comparison.csv": 2,
    "tables/runtime_summary.csv": 2,
}


def _csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_full_synthetic_artifact(
    path: str | Path, *, write_report: bool = True
) -> dict[str, Any]:
    from .full_synthetic_independent_verifier import (
        full_synthetic_analysis_verifier_comparison,
    )

    root = Path(path).resolve()
    actual_files = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file()
    }
    virtual_files = set(actual_files)
    if write_report:
        virtual_files.add("artifact_verification.json")
    expected_files = set(REQUIRED_FILES)
    missing = sorted(expected_files - virtual_files)
    extra = sorted(actual_files - expected_files)
    row_mismatches = []
    for relative, expected in FIXED_ROW_COUNTS.items():
        candidate = root / relative
        if not candidate.is_file():
            continue
        try:
            actual = _csv_count(candidate)
        except (OSError, UnicodeError, csv.Error):
            actual = None
        if actual != expected:
            row_mismatches.append({"actual": actual, "expected": expected, "path": relative})
    headerless = []
    for name in TABLES:
        candidate = root / "tables" / name
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8", newline="") as stream:
                    if not csv.DictReader(stream).fieldnames:
                        headerless.append(f"tables/{name}")
            except (OSError, UnicodeError, csv.Error):
                headerless.append(f"tables/{name}")
    invalid_png = []
    png_dimensions: dict[str, list[int]] = {}
    for name in FIGURES:
        candidate = root / "figures" / name
        if candidate.is_file():
            data = candidate.read_bytes()
            width = int.from_bytes(data[16:20], "big") if len(data) >= 24 else 0
            height = int.from_bytes(data[20:24], "big") if len(data) >= 24 else 0
            png_dimensions[f"figures/{name}"] = [width, height]
            if (
                len(data) < 100
                or data[:8] != b"\x89PNG\r\n\x1a\n"
                or width <= 0
                or height <= 0
            ):
                invalid_png.append(f"figures/{name}")

    report_path = root / "full_synthetic_development_report.md"
    report_text = (
        report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    )
    report_missing_figures = [
        name for name in FIGURES if f"figures/{name}" not in report_text
    ]
    report_missing_tables = [
        name for name in TABLES if f"tables/{name}" not in report_text
    ]

    json_values = {}
    invalid_json = []
    for name in (
        "primary_analysis.json", "independent_verification.json",
        "final_decision.json", "run_manifest.json",
    ):
        candidate = root / name
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if type(value) is not dict:
                raise ValueError("root")
            json_values[name] = value
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            invalid_json.append(name)
    primary = json_values.get("primary_analysis.json", {})
    independent = json_values.get("independent_verification.json", {})
    decision = json_values.get("final_decision.json", {})
    run = json_values.get("run_manifest.json", {})
    try:
        comparison = full_synthetic_analysis_verifier_comparison(primary, independent)
        recomputed_difference = comparison["section_difference_count"]
    except (KeyError, TypeError, ValueError):
        comparison = {
            "absolute_tolerance": 1.0e-12,
            "leaf_difference_count": -1,
            "maximum_absolute_numeric_difference": None,
            "relative_tolerance": 1.0e-12,
            "section_difference_count": -1,
        }
        recomputed_difference = -1
    difference_pass = bool(
        recomputed_difference == 0
        and primary.get("ANALYSIS_VERIFIER_DIFFERENCE_COUNT") == 0
        and independent.get("ANALYSIS_VERIFIER_DIFFERENCE_COUNT") == 0
    )
    primary_decision = primary.get("final_decision")
    independent_decision = independent.get("final_decision")
    expected_publication_decision: dict[str, Any] = (
        dict(primary_decision) if type(primary_decision) is dict else {}
    )
    agreement = recomputed_difference == 0
    if expected_publication_decision:
        expected_publication_decision["ANALYSIS_VERIFIER_AGREEMENT_PASS"] = agreement
        expected_publication_decision["ANALYSIS_VERIFIER_DIFFERENCE_COUNT"] = (
            recomputed_difference
        )
        expected_publication_decision[
            "FULL_SYNTHETIC_DEVELOPMENT_ENGINEERING_PASS"
        ] = bool(
            expected_publication_decision.get(
                "FULL_SYNTHETIC_DEVELOPMENT_ENGINEERING_PASS"
            )
            is True
            and agreement
        )
        development = all(
            expected_publication_decision.get(name) is True
            for name in REQUIRED_DEVELOPMENT_GATES
        )
        expected_publication_decision["FULL_SYNTHETIC_DEVELOPMENT_PASS"] = development
        expected_publication_decision[
            "CONFIRMATORY_PROTOCOL_DESIGN_AUTHORIZED"
        ] = development
        expected_publication_decision["REAL_DATA_PROTOCOL_DESIGN_AUTHORIZED"] = development
        if not agreement:
            expected_publication_decision[
                "PHENOMENON_CONFIRMED_BUT_INCREMENTAL_VALUE_NOT_ESTABLISHED"
            ] = False
    decision_match = bool(
        decision
        and decision == expected_publication_decision
        and (
            primary_decision == independent_decision
            if agreement
            else True
        )
    )
    signal = decision.get("FULL_SYNTHETIC_DEVELOPMENT_PASS")
    gate_conjunction_pass = bool(
        len(REQUIRED_DEVELOPMENT_GATES) == 8
        and all(name in decision for name in REQUIRED_DEVELOPMENT_GATES)
        and signal
        == all(decision.get(name) is True for name in REQUIRED_DEVELOPMENT_GATES)
    )
    expected_phenomenon = bool(
        agreement
        and decision.get("LOCAL_METRIC_INCREMENTAL_VALUE_PASS") is False
        and all(
            decision.get(name) is True
            for name in REQUIRED_DEVELOPMENT_GATES
            if name != "LOCAL_METRIC_INCREMENTAL_VALUE_PASS"
        )
    )
    authorization_pass = bool(
        isinstance(signal, bool)
        and decision.get("CONFIRMATORY_PROTOCOL_DESIGN_AUTHORIZED") is signal
        and decision.get("REAL_DATA_PROTOCOL_DESIGN_AUTHORIZED") is signal
        and decision.get("CONFIRMATORY_RUN_AUTHORIZED") is False
        and decision.get("REAL_DATA_RUN_AUTHORIZED") is False
        and decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is False
        and decision.get("ANALYSIS_VERIFIER_AGREEMENT_PASS") is agreement
        and decision.get("ANALYSIS_VERIFIER_DIFFERENCE_COUNT")
        == recomputed_difference
        and decision.get(
            "PHENOMENON_CONFIRMED_BUT_INCREMENTAL_VALUE_NOT_ESTABLISHED"
        )
        is expected_phenomenon
    )
    run_count_pass = bool(
        run.get("run_id") == "full-synthetic-development-v1"
        and run.get("new_snapshot_count") == 1050
        and run.get("new_trial_count") == 2100
        and run.get("combined_snapshot_count") == 1260
        and run.get("combined_trial_count") == 2520
        and run.get("open3d_trial_count") == 1260
        and run.get("pcl_trial_count") == 1260
        and run.get("native_trial_count") == 0
        and _is_sha256(run.get("raw_result_manifest_sha256"))
        and _is_sha256(run.get("scientific_protocol_sha256"))
        and _is_sha256(run.get("experiment_manifest_sha256"))
        and _is_sha256(run.get("snapshot_lock_sha256"))
    )

    checksum_entries = 0
    checksum_missing = 0
    checksum_mismatch = 0
    unsafe = 0
    duplicate = 0
    listed = set()
    sums = root / "SHA256SUMS"
    expected_checksum_files = expected_files - SHA256_EXCLUDED_FILES
    if sums.is_file():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                digest, relative = line.split("  ", 1)
            except ValueError:
                checksum_mismatch += 1
                continue
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                checksum_mismatch += 1
            checksum_entries += 1
            duplicate += relative in listed
            listed.add(relative)
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                unsafe += 1
                continue
            if not candidate.is_file():
                checksum_missing += 1
            elif file_sha256(candidate) != digest:
                checksum_mismatch += 1
    unlisted_required = sorted(expected_checksum_files - listed)
    unexpected_listed = sorted(listed - expected_checksum_files)
    passed = bool(
        not missing and not row_mismatches and not headerless and not invalid_png
        and not report_missing_figures and not report_missing_tables
        and not extra and not invalid_json and difference_pass and decision_match
        and gate_conjunction_pass and authorization_pass and run_count_pass
        and checksum_entries == len(expected_checksum_files)
        and checksum_missing == checksum_mismatch == unsafe == duplicate == 0
        and not unlisted_required and not unexpected_listed
    )
    report = {
        "ARTIFACT_VERIFICATION_PASS": passed,
        "analysis_verifier_difference_pass": difference_pass,
        "analysis_verifier_difference_count_recomputed": recomputed_difference,
        "analysis_verifier_absolute_tolerance": comparison["absolute_tolerance"],
        "analysis_verifier_leaf_difference_count": comparison["leaf_difference_count"],
        "analysis_verifier_maximum_absolute_numeric_difference": comparison[
            "maximum_absolute_numeric_difference"
        ],
        "analysis_verifier_relative_tolerance": comparison["relative_tolerance"],
        "authorization_invariant_pass": authorization_pass,
        "duplicate_sha256_path_count": duplicate,
        "final_decision_match_pass": decision_match,
        "eight_required_gate_conjunction_pass": gate_conjunction_pass,
        "extra_file_count": len(extra),
        "extra_files": extra,
        "fixed_table_row_count_mismatch_count": len(row_mismatches),
        "fixed_table_row_count_mismatches": row_mismatches,
        "headerless_table_count": len(headerless),
        "headerless_tables": headerless,
        "invalid_json_count": len(invalid_json),
        "invalid_json_files": invalid_json,
        "invalid_png_count": len(invalid_png),
        "invalid_png_files": invalid_png,
        "png_dimensions": png_dimensions,
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "required_file_count": len(REQUIRED_FILES),
        "report_missing_figure_reference_count": len(report_missing_figures),
        "report_missing_figure_references": report_missing_figures,
        "report_missing_table_reference_count": len(report_missing_tables),
        "report_missing_table_references": report_missing_tables,
        "run_cardinality_pass": run_count_pass,
        "schema_version": "full_synthetic_development_artifact_verification_v1",
        "sha256_entry_count": checksum_entries,
        "sha256_mismatch_count": checksum_mismatch,
        "sha256_missing_count": checksum_missing,
        "sha256_unlisted_required_count": len(unlisted_required),
        "sha256_unlisted_required_files": unlisted_required,
        "sha256_unexpected_listed_count": len(unexpected_listed),
        "sha256_unexpected_listed_files": unexpected_listed,
        "unsafe_sha256_path_count": unsafe,
    }
    if write_report:
        write_json(root / "artifact_verification.json", report)
    return report


__all__ = [
    "FIGURES", "FIXED_ROW_COUNTS", "REQUIRED_FILES", "ROOT_FILES", "TABLES",
    "verify_full_synthetic_artifact",
]
