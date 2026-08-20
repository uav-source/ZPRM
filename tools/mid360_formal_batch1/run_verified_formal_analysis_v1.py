#!/usr/bin/env python3
"""Future canonical one-time execution wrapper; this task must not invoke it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for item in (REPOSITORY, REPOSITORY / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from experiments.mid360_formal_batch1.analysis_authorization.authorized_analysis_runner_v1 import run_authoritative_cli_once  # noqa: E402
from experiments.mid360_formal_batch1.locked_analysis.formal_firewall_v1 import FormalReadRequest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-results-root", type=Path, required=True)
    parser.add_argument("--postrun-verification-root", type=Path, required=True)
    parser.add_argument("--analysis-lock-dir", type=Path, required=True)
    parser.add_argument("--analysis-code-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-python", type=Path, required=True)
    parser.add_argument("--confirm-read-frozen-formal-results", action="store_true")
    args = parser.parse_args()
    if not args.confirm_read_frozen_formal_results:
        parser.error("--confirm-read-frozen-formal-results is required")
    request = FormalReadRequest(
        repository=REPOSITORY, formal_results_root=args.formal_results_root,
        postrun_verification_root=args.postrun_verification_root,
        analysis_lock_dir=args.analysis_lock_dir,
        analysis_code_commit=args.analysis_code_commit, output_dir=args.output_dir,
        confirm_read_frozen_formal_results=True,
    )
    run_authoritative_cli_once(
        request, frozen_python=args.frozen_python,
        frozen_cli=REPOSITORY / "tools/mid360_formal_batch1/run_zero_perturbation_locked_analysis_v1.py",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
