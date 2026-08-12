#!/usr/bin/env python3
"""Independently verify the Boreas v2 Stage-2 storage-plan closure."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_stage2_storage_optimization_verifier import (  # noqa: E402,E501
    BoreasStage2StorageVerificationError,
    verify_boreas_v2_stage2_storage_plan,
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
        "--frozen-root",
        dest="runtime_root",
        type=Path,
        default=REPOSITORY
        / "frozen_assets/boreas_v2_stage2_storage_optimization",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        report = verify_boreas_v2_stage2_storage_plan(
            repository=arguments.repository,
            runtime_root=arguments.runtime_root,
            data_root=arguments.data_root,
        )
    except (
        BoreasStage2StorageVerificationError,
        IndexError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "BOREAS_V2_STAGE2_STORAGE_VERIFICATION_PASS": False,
                    "STAGE2_STORAGE_PLAN_VERIFICATION_PASS": False,
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
