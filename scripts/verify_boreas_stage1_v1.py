#!/usr/bin/env python3
"""Independently verify Boreas Stage-1 evidence and its fail-closed result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_stage1_verifier import (  # noqa: E402
    BoreasStage1VerificationError,
    verify_boreas_stage1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", "--repository", dest="repository", type=Path, default=REPOSITORY)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        report = verify_boreas_stage1(
            repository=arguments.repository,
            data_root=arguments.data_root,
            runtime_root=arguments.runtime_root,
        )
    except (BoreasStage1VerificationError, OSError, ValueError) as error:
        print(json.dumps({
            "BOREAS_STAGE1_VERIFICATION_PASS": False,
            "error": str(error),
            "verification_pass": False,
        }, indent=2, sort_keys=True, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
