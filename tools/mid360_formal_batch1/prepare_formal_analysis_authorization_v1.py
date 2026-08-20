#!/usr/bin/env python3
"""Prepare a non-authoritative formal-analysis authorization candidate."""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for item in (REPOSITORY, REPOSITORY / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from experiments.mid360_formal_batch1.analysis_authorization.formal_analysis_authorization_producer_v1 import prepare_authorization_candidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--analysis-lock-dir", type=Path, required=True)
    parser.add_argument("--analysis-control-commit", required=True)
    parser.add_argument("--analysis-lock-release-commit", required=True)
    parser.add_argument("--analysis-output-root", type=Path, required=True)
    parser.add_argument("--nonce", default=None)
    parser.add_argument("--confirm-explicit-user-analysis-authorization", action="store_true")
    args = parser.parse_args()
    payload = prepare_authorization_candidate(
        args.repository, args.analysis_lock_dir,
        analysis_control_commit=args.analysis_control_commit,
        analysis_lock_release_commit=args.analysis_lock_release_commit,
        nonce=args.nonce or secrets.token_hex(16),
        confirm_explicit_user_analysis_authorization=(
            args.confirm_explicit_user_analysis_authorization
        ),
        output_root=args.analysis_output_root,
    )
    print(f"CANDIDATE_PREPARED=true\nauthorization_id={payload['authorization_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
