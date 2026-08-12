#!/usr/bin/env python3
"""Independently verify the Boreas external-validation v2 Stage-1 closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_external_v2_stage1_verifier import (  # noqa: E402
    BoreasExternalV2Stage1VerificationError,
    verify_boreas_external_v2_stage1,
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
        "--data-root",
        type=Path,
        default=Path.home() / "zero_perturbation_data/boreas_stage1_v1",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=REPOSITORY
        / "frozen_assets/public_data_external_validation_v2_boreas_stage1",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        report = verify_boreas_external_v2_stage1(
            repository=arguments.repository,
            data_root=arguments.data_root,
            runtime_root=arguments.runtime_root,
        )
    except (BoreasExternalV2Stage1VerificationError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "BOREAS_EXTERNAL_V2_STAGE1_VERIFICATION_PASS": False,
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
