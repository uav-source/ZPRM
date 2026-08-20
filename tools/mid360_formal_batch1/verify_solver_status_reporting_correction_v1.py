#!/usr/bin/env python3
"""Independently verify and checksum the FMB1 reporting correction v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for candidate in (REPOSITORY, REPOSITORY / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.mid360_formal_batch1.reporting_correction.reporting_correction_verify_v1 import (  # noqa: E402
    verify_reporting_correction,
)


EXPECTED_CORE = {
    "reporting_correction_notice_v1.json",
    "reporting_correction_notice_v1.md",
    "corrected_solver_status_accounting_v1.json",
    "corrected_solver_status_accounting_v1.csv",
    "analysis_summary_reporting_corrected_v1.json",
    "analysis_summary_reporting_corrected_v1.md",
    "scientific_value_immutability_check.json",
    "json_path_correction_diff.json",
    "reporting_correction_manifest.json",
}
REPORT_NAME = "independent_reporting_correction_verification.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--correction-dir",
        type=Path,
        default=(
            REPOSITORY
            / "results/mid360_formal_batch1/"
            "zero_perturbation_locked_analysis_reporting_correction_v1"
        ),
    )
    args = parser.parse_args(argv)
    correction = args.correction_dir.resolve(strict=True)
    actual = {path.name for path in correction.iterdir() if path.is_file()}
    if actual != EXPECTED_CORE:
        raise SystemExit(f"correction core file set mismatch: {sorted(actual ^ EXPECTED_CORE)}")
    report = verify_reporting_correction(args.repository, correction)
    report_path = correction / REPORT_NAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        errors="strict",
    )
    if not report["pass"]:
        print("INDEPENDENT_REPORTING_CORRECTION_VERIFICATION_PASS=false")
        print(f"failure_count={report['failure_count']}")
        return 1
    names = sorted(EXPECTED_CORE | {REPORT_NAME})
    checksum_path = correction / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{_sha256(correction / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    print("INDEPENDENT_REPORTING_CORRECTION_VERIFICATION_PASS=true")
    print("failure_count=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
