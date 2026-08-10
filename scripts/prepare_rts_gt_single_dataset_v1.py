#!/usr/bin/env python3
"""Run the fail-closed RTS-GT metadata/GT-only Stage-1 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_a_harness.real_data_preparation.rts_gt import execute_stage1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("/home/lj/ZPRM"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--no-registration", action="store_true", required=True)
    arguments = parser.parse_args()
    summary = execute_stage1(
        repository=arguments.repository,
        data_root=arguments.data_root,
        runtime_root=arguments.runtime_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
