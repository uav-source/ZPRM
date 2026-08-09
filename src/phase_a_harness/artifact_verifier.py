"""Standalone verifier for the compact formal Phase A artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import file_sha256, write_json


REQUIRED = (
    "tables/snapshot_inventory.csv",
    "tables/trial_results.csv",
    "tables/open3d_trial_results.csv",
    "tables/pcl_trial_results.csv",
    "tables/backend_input_pairing.csv",
    "tables/backend_summary.csv",
    "tables/scene_backend_summary.csv",
    "tables/failure_inventory.csv",
    "tables/runtime_summary.csv",
    "tables/gate_summary.csv",
    "figures/backend_translation_updates.png",
    "figures/backend_rotation_updates.png",
    "figures/scene_translation_updates.png",
    "figures/open3d_vs_pcl_updates.png",
    "figures/runtime_comparison.png",
    "phase_a_report.md",
    "final_decision.json",
    "run_manifest.json",
    "independent_verification.json",
    "source_export_manifest.csv",
    "environment_manifest.json",
    "SHA256SUMS",
)


def verify_formal_artifact(path: str | Path, *, write_report: bool = True) -> dict[str, Any]:
    root = Path(path).resolve()
    missing = [relative for relative in REQUIRED if not (root / relative).is_file()]
    checksum_missing = 0
    checksum_mismatch = 0
    entries = 0
    sums = root / "SHA256SUMS"
    if sums.is_file():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, relative = line.split("  ", 1)
            candidate = root / relative
            entries += 1
            if not candidate.is_file():
                checksum_missing += 1
            elif file_sha256(candidate) != digest:
                checksum_mismatch += 1
    passed = not missing and checksum_missing == 0 and checksum_mismatch == 0 and entries > 0
    report = {
        "ARTIFACT_VERIFICATION_PASS": passed,
        "required_file_count": len(REQUIRED),
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "sha256_entry_count": entries,
        "sha256_missing_count": checksum_missing,
        "sha256_mismatch_count": checksum_mismatch,
        "schema_version": "phase_a_minimal_harness_artifact_verification_v1",
    }
    if write_report:
        write_json(root / "artifact_verification.json", report)
    return report


__all__ = ["REQUIRED", "verify_formal_artifact"]

