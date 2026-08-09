"""Strict verifier for the compact Scientific Survival audit artifact."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


TABLES = (
    "replicate_uniqueness_by_cell.csv",
    "replicate_uniqueness_by_condition.csv",
    "measurement_seed_effectiveness.csv",
    "repeat_index_effectiveness.csv",
    "unique_unit_scene_effect.csv",
    "cross_backend_unique_unit.csv",
    "turnover_robustness.csv",
    "model_leakage_audit.csv",
    "model_weighting_sensitivity.csv",
    "model_fold_results.csv",
    "counterexample_review_shortlist.csv",
    "scientific_claim_authorization.csv",
    "gate_summary.csv",
)
FIGURES = (
    "effective_replicates_by_condition.png",
    "original_vs_unique_weighted_effect.png",
    "turnover_leave_one_scene_out.png",
    "model_weighting_sensitivity.png",
    "claim_authorization_matrix.png",
)
ROOT_FILES = (
    "pre_repair_result_inventory.csv",
    "publisher_only_change_scope.json",
    "publisher_repair_report.md",
    "replicate_uniqueness_audit.md",
    "counterexample_review_packet.md",
    "scientific_survival_audit_report.md",
    "final_decision.json",
    "run_manifest.json",
    "SHA256SUMS",
    "artifact_verification.json",
)
REQUIRED_FILES = tuple(f"tables/{name}" for name in TABLES) + tuple(
    f"figures/{name}" for name in FIGURES
) + ROOT_FILES
SHA_EXCLUDED = frozenset({"SHA256SUMS", "artifact_verification.json"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_sha256s(path: Path) -> tuple[dict[str, str], int, int]:
    entries: dict[str, str] = {}
    duplicate_count = 0
    unsafe_count = 0
    if not path.is_file():
        return entries, duplicate_count, unsafe_count
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line:
            continue
        parts = raw_line.split("  ", 1)
        if len(parts) != 2:
            unsafe_count += 1
            continue
        digest, relative = parts
        candidate = Path(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in {"", "."}
        ):
            unsafe_count += 1
            continue
        normalized = candidate.as_posix()
        if normalized in entries:
            duplicate_count += 1
        entries[normalized] = digest
    return entries, duplicate_count, unsafe_count


def verify_scientific_survival_artifact(
    root_path: str | Path, *, write_report: bool = True
) -> dict[str, Any]:
    root = Path(root_path).resolve()
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    } if root.is_dir() else set()
    virtual = set(actual)
    if write_report:
        virtual.add("artifact_verification.json")
    required = set(REQUIRED_FILES)
    missing = sorted(required - virtual)
    extra = sorted(actual - required)

    headerless: list[str] = []
    csv_row_counts: dict[str, int] = {}
    for name in TABLES:
        relative = f"tables/{name}"
        path = root / relative
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames:
                    headerless.append(relative)
                csv_row_counts[relative] = sum(1 for _ in reader)
        except (OSError, UnicodeError, csv.Error):
            headerless.append(relative)

    invalid_png: list[str] = []
    png_dimensions: dict[str, list[int]] = {}
    for name in FIGURES:
        relative = f"figures/{name}"
        path = root / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        width = int.from_bytes(data[16:20], "big") if len(data) >= 24 else 0
        height = int.from_bytes(data[20:24], "big") if len(data) >= 24 else 0
        png_dimensions[relative] = [width, height]
        if data[:8] != b"\x89PNG\r\n\x1a\n" or width <= 0 or height <= 0:
            invalid_png.append(relative)

    invalid_json: list[str] = []
    json_objects: dict[str, dict[str, Any]] = {}
    for name in ("publisher_only_change_scope.json", "final_decision.json", "run_manifest.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if type(value) is not dict:
                raise ValueError("JSON root is not an object")
            json_objects[name] = value
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            invalid_json.append(name)

    report = root / "scientific_survival_audit_report.md"
    report_text = report.read_text(encoding="utf-8") if report.is_file() else ""
    missing_table_references = [
        name for name in TABLES if f"tables/{name}" not in report_text
    ]
    missing_figure_references = [
        name for name in FIGURES if f"figures/{name}" not in report_text
    ]

    entries, duplicate_sha, unsafe_sha = _read_sha256s(root / "SHA256SUMS")
    checksum_expected = required - SHA_EXCLUDED
    listed = set(entries)
    sha_missing = sorted(checksum_expected - listed)
    sha_unexpected = sorted(listed - checksum_expected)
    sha_mismatch = sorted(
        relative
        for relative, digest in entries.items()
        if relative in checksum_expected
        and (not (root / relative).is_file() or _sha256(root / relative) != digest)
    )

    decision = json_objects.get("final_decision.json", {})
    required_gate_names = (
        "RAW_EVIDENCE_INTEGRITY_PASS",
        "ARTIFACT_PUBLICATION_PASS",
        "REPLICATE_UNIQUENESS_AUDIT_PASS",
        "PRIMARY_SCENE_EFFECT_UNIQUE_UNIT_PASS",
        "CROSS_BACKEND_UNIQUE_UNIT_PASS",
        "REASSOCIATION_ROBUSTNESS_PASS",
        "MODEL_DATA_LEAKAGE_AUDIT_PASS",
        "MODEL_INCREMENTAL_VALUE_ROBUST_PASS",
        "CLAIM_WORDING_BOUNDARY_PASS",
    )
    conjunction = bool(
        all(name in decision for name in required_gate_names)
        and decision.get("SCIENTIFIC_SURVIVAL_AUDIT_PASS")
        == all(decision.get(name) is True for name in required_gate_names)
    )
    fixed_authorization = bool(
        decision.get("CONFIRMATORY_RUN_AUTHORIZED") is False
        and decision.get("REAL_DATA_RUN_AUTHORIZED") is False
        and decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is False
        and decision.get("COUNTEREXAMPLE_CLAIM_AUTHORIZED") is False
    )
    verification = {
        "schema_version": "scientific_survival_artifact_verification_v1",
        "required_file_count": len(REQUIRED_FILES),
        "actual_file_count": len(actual) + (1 if write_report and "artifact_verification.json" not in actual else 0),
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "extra_file_count": len(extra),
        "extra_files": extra,
        "headerless_table_count": len(headerless),
        "headerless_tables": sorted(set(headerless)),
        "csv_row_counts": csv_row_counts,
        "invalid_png_count": len(invalid_png),
        "invalid_png_files": invalid_png,
        "png_dimensions": png_dimensions,
        "invalid_json_count": len(invalid_json),
        "invalid_json_files": invalid_json,
        "report_missing_table_reference_count": len(missing_table_references),
        "report_missing_table_references": missing_table_references,
        "report_missing_figure_reference_count": len(missing_figure_references),
        "report_missing_figure_references": missing_figure_references,
        "sha256_entry_count": len(entries),
        "duplicate_sha256_path_count": duplicate_sha,
        "unsafe_sha256_path_count": unsafe_sha,
        "sha256_missing_count": len(sha_missing),
        "sha256_missing_files": sha_missing,
        "sha256_mismatch_count": len(sha_mismatch),
        "sha256_mismatch_files": sha_mismatch,
        "sha256_unexpected_listed_count": len(sha_unexpected),
        "sha256_unexpected_listed_files": sha_unexpected,
        "survival_gate_conjunction_pass": conjunction,
        "fixed_authorization_boundary_pass": fixed_authorization,
    }
    verification["ARTIFACT_VERIFICATION_PASS"] = bool(
        not missing
        and not extra
        and not headerless
        and not invalid_png
        and not invalid_json
        and not missing_table_references
        and not missing_figure_references
        and duplicate_sha == 0
        and unsafe_sha == 0
        and not sha_missing
        and not sha_mismatch
        and not sha_unexpected
        and conjunction
        and fixed_authorization
    )
    if write_report:
        (root / "artifact_verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return verification


__all__ = [
    "FIGURES",
    "REQUIRED_FILES",
    "ROOT_FILES",
    "TABLES",
    "verify_scientific_survival_artifact",
]
