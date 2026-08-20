#!/usr/bin/env python3
"""Independently verify an authorization candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for item in (REPOSITORY, REPOSITORY / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from experiments.mid360_formal_batch1.analysis_authorization.formal_analysis_authorization_verify_v1 import verify_authorization_candidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--analysis-lock-dir", type=Path, required=True)
    parser.add_argument("--analysis-control-commit", required=True)
    parser.add_argument("--analysis-lock-release-commit", required=True)
    parser.add_argument("--analysis-output-root", type=Path, required=True)
    args = parser.parse_args()
    report = verify_authorization_candidate(
        args.repository, args.analysis_lock_dir,
        expected_analysis_control_commit=args.analysis_control_commit,
        expected_lock_release_commit=args.analysis_lock_release_commit,
        expected_output_root=args.analysis_output_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
