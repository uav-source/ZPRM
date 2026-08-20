#!/usr/bin/env python3
"""Issue the analysis lock after the code-freeze commit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for item in (REPOSITORY, REPOSITORY / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from experiments.mid360_formal_batch1.locked_analysis.lock_v1 import issue_analysis_lock  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--lock-dir", type=Path, required=True)
    parser.add_argument("--analysis-code-commit", required=True)
    parser.add_argument("--fixture-qualification", type=Path, required=True)
    parser.add_argument("--no-real-results-attestation", type=Path, required=True)
    args = parser.parse_args()
    payload = issue_analysis_lock(
        args.repository, args.lock_dir,
        analysis_code_commit=args.analysis_code_commit,
        qualification_path=args.fixture_qualification,
        attestation_path=args.no_real_results_attestation,
    )
    print(json.dumps({"status": payload["status"],
                      "analysis_lock_fingerprint": payload["analysis_lock_fingerprint"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
