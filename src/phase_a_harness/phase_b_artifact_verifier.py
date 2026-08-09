"""Standalone structural and SHA verifier for the compact Phase B artifact."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .contracts import file_sha256, write_json


REQUIRED_PHASE_B_FILES = (
    "tables/snapshot_inventory.csv",
    "tables/trial_results.csv",
    "tables/open3d_trial_results.csv",
    "tables/pcl_trial_results.csv",
    "tables/backend_input_pairing.csv",
    "tables/scene_condition_backend_summary.csv",
    "tables/geometry_seed_raw_values.csv",
    "tables/cross_backend_ranking.csv",
    "tables/scene_ranking.csv",
    "tables/weak_rich_effect.csv",
    "tables/geometry_seed_consistency.csv",
    "tables/failure_inventory.csv",
    "tables/runtime_summary.csv",
    "tables/gate_summary.csv",
    "figures/scene_errors_independent_noise_free.png",
    "figures/scene_errors_full_noise.png",
    "figures/open3d_vs_pcl_scene_ranking.png",
    "figures/weak_rich_ratios.png",
    "figures/geometry_seed_consistency.png",
    "figures/backend_runtime.png",
    "phase_b_report.md",
    "primary_analysis.json",
    "independent_verification.json",
    "final_decision.json",
    "run_manifest.json",
    "SHA256SUMS",
)

EXPECTED_TABLE_ROWS = {
    "tables/snapshot_inventory.csv": 42,
    "tables/trial_results.csv": 84,
    "tables/open3d_trial_results.csv": 42,
    "tables/pcl_trial_results.csv": 42,
    "tables/backend_input_pairing.csv": 42,
    "tables/scene_condition_backend_summary.csv": 28,
    "tables/geometry_seed_raw_values.csv": 84,
    "tables/cross_backend_ranking.csv": 2,
    "tables/scene_ranking.csv": 28,
    "tables/weak_rich_effect.csv": 8,
    "tables/geometry_seed_consistency.csv": 12,
    "tables/runtime_summary.csv": 2,
}


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def verify_phase_b_artifact(
    path: str | Path, *, write_report: bool = True
) -> dict[str, Any]:
    """Verify required files, table cardinalities, decisions, PNGs, and SHA256SUMS."""

    root = Path(path).resolve()
    missing = [relative for relative in REQUIRED_PHASE_B_FILES if not (root / relative).is_file()]
    table_mismatches: list[dict[str, Any]] = []
    for relative, expected in EXPECTED_TABLE_ROWS.items():
        candidate = root / relative
        if not candidate.is_file():
            continue
        try:
            actual = _row_count(candidate)
        except (OSError, UnicodeError, csv.Error):
            actual = None
        if actual != expected:
            table_mismatches.append({"actual": actual, "expected": expected, "path": relative})

    png_invalid: list[str] = []
    for relative in REQUIRED_PHASE_B_FILES:
        if not relative.endswith(".png"):
            continue
        candidate = root / relative
        if candidate.is_file() and (
            candidate.stat().st_size < 100 or candidate.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
        ):
            png_invalid.append(relative)

    json_invalid: list[str] = []
    json_values: dict[str, Any] = {}
    for relative in (
        "primary_analysis.json", "independent_verification.json",
        "final_decision.json", "run_manifest.json",
    ):
        candidate = root / relative
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if type(value) is not dict:
                raise ValueError("JSON root is not an object")
            json_values[relative] = value
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            json_invalid.append(relative)

    decision_invariant_pass = False
    decision = json_values.get("final_decision.json")
    if type(decision) is dict:
        signal = decision.get("PHASE_B_SIGNAL_PASS")
        decision_invariant_pass = bool(
            decision.get("FULL_SYNTHETIC_DEVELOPMENT_RUN_AUTHORIZED") is False
            and decision.get("CONFIRMATORY_AUTHORIZED") is False
            and decision.get("REAL_DATA_AUTHORIZED") is False
            and decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is False
            and decision.get("FULL_SYNTHETIC_DEVELOPMENT_PROTOCOL_DESIGN_AUTHORIZED") is (signal is True)
        )
    analysis = json_values.get("primary_analysis.json", {})
    independent = json_values.get("independent_verification.json", {})
    difference_fields_pass = bool(
        analysis.get("ANALYSIS_VERIFIER_DIFFERENCE_COUNT") == 0
        and independent.get("ANALYSIS_VERIFIER_DIFFERENCE_COUNT") == 0
        and analysis.get("solver_failure_count_by_backend")
        == independent.get("solver_failure_count_by_backend")
        and analysis.get("nonfinite_output_count_by_backend")
        == independent.get("nonfinite_output_count_by_backend")
    )
    final_decision_match_pass = bool(
        type(decision) is dict
        and decision == analysis.get("final_decision")
        and decision == independent.get("final_decision")
    )
    formal_run = json_values.get("run_manifest.json", {})
    formal_run_cardinality_pass = bool(
        formal_run.get("run_id") == "phase-b-signal-v1"
        and
        formal_run.get("completed_snapshot_count") == 42
        and formal_run.get("completed_trial_count") == 84
        and formal_run.get("open3d_trial_count") == 42
        and formal_run.get("pcl_trial_count") == 42
        and formal_run.get("native_execution_count") == 0
        and formal_run.get("native_trial_count") == 0
    )

    sums = root / "SHA256SUMS"
    checksum_entries = 0
    checksum_missing = 0
    checksum_mismatch = 0
    duplicate_checksum_paths = 0
    unsafe_checksum_paths = 0
    listed: set[str] = set()
    if sums.is_file():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                digest, relative = line.split("  ", 1)
            except ValueError:
                checksum_mismatch += 1
                continue
            checksum_entries += 1
            if relative in listed:
                duplicate_checksum_paths += 1
            listed.add(relative)
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                unsafe_checksum_paths += 1
                continue
            if not candidate.is_file():
                checksum_missing += 1
            elif file_sha256(candidate) != digest:
                checksum_mismatch += 1
    required_checksum_paths = {
        relative for relative in REQUIRED_PHASE_B_FILES if relative != "SHA256SUMS"
    }
    checksum_unlisted_required = sorted(required_checksum_paths - listed)
    passed = bool(
        not missing
        and not table_mismatches
        and not png_invalid
        and not json_invalid
        and decision_invariant_pass
        and difference_fields_pass
        and final_decision_match_pass
        and formal_run_cardinality_pass
        and checksum_entries > 0
        and checksum_missing == 0
        and checksum_mismatch == 0
        and duplicate_checksum_paths == 0
        and unsafe_checksum_paths == 0
        and not checksum_unlisted_required
    )
    report = {
        "ARTIFACT_VERIFICATION_PASS": passed,
        "analysis_verifier_difference_fields_pass": difference_fields_pass,
        "decision_authorization_invariant_pass": decision_invariant_pass,
        "duplicate_sha256_path_count": duplicate_checksum_paths,
        "invalid_json_count": len(json_invalid),
        "invalid_json_files": json_invalid,
        "invalid_png_count": len(png_invalid),
        "invalid_png_files": png_invalid,
        "final_decision_match_pass": final_decision_match_pass,
        "formal_run_cardinality_pass": formal_run_cardinality_pass,
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "required_file_count": len(REQUIRED_PHASE_B_FILES),
        "schema_version": "phase_b_scene_signal_artifact_verification_v1",
        "sha256_entry_count": checksum_entries,
        "sha256_mismatch_count": checksum_mismatch,
        "sha256_missing_count": checksum_missing,
        "sha256_unlisted_required_count": len(checksum_unlisted_required),
        "sha256_unlisted_required_files": checksum_unlisted_required,
        "table_row_count_mismatch_count": len(table_mismatches),
        "table_row_count_mismatches": table_mismatches,
        "unsafe_sha256_path_count": unsafe_checksum_paths,
    }
    if write_report:
        write_json(root / "artifact_verification.json", report)
    return report


__all__ = ["EXPECTED_TABLE_ROWS", "REQUIRED_PHASE_B_FILES", "verify_phase_b_artifact"]
