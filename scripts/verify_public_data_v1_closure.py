#!/usr/bin/env python3
"""Independently verify the frozen Public-data v1 screening closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.public_data_v1_closure_verifier import (  # noqa: E402
    PublicDataV1ClosureVerificationError,
    verify_public_data_v1_closure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        "--repository",
        dest="repository",
        type=Path,
        default=REPOSITORY,
    )
    parser.add_argument(
        "--closure-root",
        type=Path,
        default=REPOSITORY / "frozen_assets/public_data_validation_v1_closure",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        report = verify_public_data_v1_closure(
            repository=arguments.repository,
            closure_root=arguments.closure_root,
        )
    except (PublicDataV1ClosureVerificationError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "PUBLIC_DATA_V1_CLOSURE_VERIFICATION_PASS": False,
                    "error": str(error),
                    "verification_pass": False,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
