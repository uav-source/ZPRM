#!/usr/bin/env python3
"""Issue non-authorizing FMB1 Analysis Lock v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for item in (REPOSITORY, REPOSITORY / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from experiments.mid360_formal_batch1.analysis_authorization.analysis_lock_v2 import issue_analysis_lock_v2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--lock-dir", type=Path, required=True)
    parser.add_argument("--analysis-control-commit", required=True)
    parser.add_argument("--contract-audit", type=Path, required=True)
    parser.add_argument("--control-qualification", type=Path, required=True)
    parser.add_argument("--fixture-qualification", type=Path, required=True)
    parser.add_argument("--no-real-results-attestation", type=Path, required=True)
    parser.add_argument("--v1-supersession", type=Path, required=True)
    parser.add_argument("--blocked-attempt-audit", type=Path, required=True)
    args = parser.parse_args()
    lock = issue_analysis_lock_v2(
        args.repository, args.lock_dir,
        analysis_control_commit=args.analysis_control_commit,
        contract_audit_path=args.contract_audit,
        control_qualification_path=args.control_qualification,
        fixture_qualification_path=args.fixture_qualification,
        no_real_results_attestation_path=args.no_real_results_attestation,
        v1_supersession_path=args.v1_supersession,
        blocked_attempt_audit_path=args.blocked_attempt_audit,
    )
    print(json.dumps({
        "status": lock["status"],
        "analysis_lock_fingerprint": lock["analysis_lock_fingerprint"],
        "REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
