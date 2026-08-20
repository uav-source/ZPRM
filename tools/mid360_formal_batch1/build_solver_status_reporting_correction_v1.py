#!/usr/bin/env python3
"""Build the versioned FMB1 solver-status reporting correction v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for candidate in (REPOSITORY, REPOSITORY / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.mid360_formal_batch1.reporting_correction import (  # noqa: E402
    build_reporting_correction,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPOSITORY
            / "results/mid360_formal_batch1/"
            "zero_perturbation_locked_analysis_reporting_correction_v1"
        ),
    )
    args = parser.parse_args(argv)
    manifest = build_reporting_correction(args.repository, args.output_dir)
    print("VERSIONED_REPORTING_CORRECTION_BUILT=true")
    print(f"CORRECTED_FORMAL_COMPLETED_N={manifest['corrected_accounting']['formal_completed_n']}")
    print(
        "CORRECTED_FORMAL_SOLVER_NONCONVERGENCE_N="
        f"{manifest['corrected_accounting']['formal_solver_nonconvergence_n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
