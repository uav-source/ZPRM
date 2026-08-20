#!/usr/bin/env python3
"""Publish a candidate only after an exact independent PASS report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for item in (REPOSITORY, REPOSITORY / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from experiments.mid360_formal_batch1.analysis_authorization.formal_analysis_authorization_lifecycle_v1 import publish_verified_authorization  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-lock-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = publish_verified_authorization(args.analysis_lock_dir)
    print(f"VERIFIED_AUTHORIZATION_PUBLISHED=true\nauthorization_id={payload['authorization_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
