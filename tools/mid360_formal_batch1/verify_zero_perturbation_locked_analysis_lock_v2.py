#!/usr/bin/env python3
"""Independently verify issued FMB1 Analysis Lock v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for item in (REPOSITORY, REPOSITORY / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from experiments.mid360_formal_batch1.analysis_authorization.analysis_lock_v2_verify import verify_analysis_lock_v2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--lock-dir", type=Path, required=True)
    args = parser.parse_args()
    report = verify_analysis_lock_v2(args.repository, args.lock_dir, write_report=True)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
