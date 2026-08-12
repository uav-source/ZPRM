#!/usr/bin/env python3
"""Create the narrow Boreas v2 Stage-2 payload-preparation authorization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_v2_stage2_authorization import (  # noqa: E402
    BoreasStage2AuthorizationError,
    build_boreas_v2_stage2_download_authorization,
)
from phase_a_harness.real_data_preparation.guard import (  # noqa: E402
    NoRegistrationGuard,
    RegistrationForbiddenError,
)
from phase_a_harness.real_data_preparation.io import PreparationIOError  # noqa: E402

try:  # Bind installed Open3D registration entrypoints for the guard lifetime.
    import open3d as _open3d  # type: ignore[import-not-found]  # noqa: E402
except ImportError:  # pragma: no cover - source-only test environments
    _open3d = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "zero_perturbation_data/boreas_stage1_v1",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=(
            Path.home()
            / "zero_perturbation_runtime/real_data/boreas_v2_stage2_preparation"
        ),
    )
    parser.add_argument(
        "--temporary-root",
        type=Path,
        default=(
            Path.home()
            / "zero_perturbation_runtime/real_data/boreas_v2_stage2_preparation/tmp"
        ),
    )
    parser.add_argument(
        "--monitored-disk-path",
        type=Path,
        default=Path.home() / "zero_perturbation_runtime",
    )
    parser.add_argument(
        "--stage1-verification-report",
        type=Path,
        help=(
            "optional canonical comparison witness; the authoritative report is "
            "always recomputed and frozen under the runtime evidence directory"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    os.environ.setdefault("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    try:
        with NoRegistrationGuard(open3d_module=_open3d) as guard:
            value = build_boreas_v2_stage2_download_authorization(
                repository=arguments.repository,
                data_root=arguments.data_root,
                runtime_root=arguments.runtime_root,
                temporary_root=arguments.temporary_root,
                monitored_disk_path=arguments.monitored_disk_path,
                stage1_verification_report=arguments.stage1_verification_report,
                output_path=arguments.output,
                no_registration_guard=guard,
            )
    except (
        BoreasStage2AuthorizationError,
        PreparationIOError,
        RegistrationForbiddenError,
        OSError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "STAGE2_DOWNLOAD_AUTHORIZED": False,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
