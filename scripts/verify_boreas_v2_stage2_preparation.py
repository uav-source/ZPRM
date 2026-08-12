#!/usr/bin/env python3
"""Independently verify the Boreas External v2 Stage-2 preparation closure."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_v2_stage2_preparation_verifier import (  # noqa: E402
    BoreasStage2PreparationVerificationError,
    PreparationVerificationAuthority,
    verify_boreas_v2_stage2_preparation,
)
from phase_a_harness.real_data_preparation.guard import (  # noqa: E402
    NoRegistrationGuard,
    RegistrationForbiddenError,
)

try:  # Bind installed Open3D registration entrypoints for the verifier lifetime.
    import open3d as _open3d  # type: ignore[import-not-found]  # noqa: E402
except ImportError:  # pragma: no cover - source-only test environments
    _open3d = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        "--repository",
        dest="repository",
        type=Path,
        default=REPOSITORY,
        help="repository containing frozen Stage-1/storage authorities",
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=REPOSITORY / "frozen_assets/boreas_v2_stage2_preparation",
        help="small frozen Stage-2 closure",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path.home()
        / "zero_perturbation_runtime/real_data/boreas_v2_stage2_preparation",
        help="runtime root containing the target and 100 canonical bundles",
    )
    parser.add_argument(
        "--authority-profile",
        type=Path,
        help="canonical JSON authority profile (synthetic/offline testing only)",
    )
    parser.add_argument(
        "--reference-pose-path",
        type=Path,
        help="override retained PRIMARY query lidar_poses.csv for production verification",
    )
    parser.add_argument(
        "--map-reference-pose-path",
        type=Path,
        help="override retained PRIMARY map lidar_poses.csv for independent GT-overlap verification",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    os.environ.setdefault("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    try:
        with NoRegistrationGuard(open3d_module=_open3d):
            if arguments.authority_profile is not None:
                if (
                    arguments.reference_pose_path is not None
                    or arguments.map_reference_pose_path is not None
                ):
                    raise ValueError(
                        "reference-pose overrides cannot accompany --authority-profile"
                    )
                authority = PreparationVerificationAuthority.from_profile(
                    arguments.authority_profile
                )
            else:
                authority = PreparationVerificationAuthority.production(
                    arguments.repository
                )
                if arguments.reference_pose_path is not None:
                    authority = replace(
                        authority,
                        reference_pose_path=arguments.reference_pose_path.resolve(
                            strict=True
                        ),
                    )
                if arguments.map_reference_pose_path is not None:
                    authority = replace(
                        authority,
                        map_reference_pose_path=arguments.map_reference_pose_path.resolve(
                            strict=True
                        ),
                    )
            report = verify_boreas_v2_stage2_preparation(
                root=arguments.frozen_root,
                runtime_root=arguments.runtime_root,
                authority=authority,
            )
    except (
        BoreasStage2PreparationVerificationError,
        OSError,
        ValueError,
        KeyError,
        RegistrationForbiddenError,
    ) as error:
        print(
            json.dumps(
                {
                    "BOREAS_EXTERNAL_V2_STAGE2_VERIFICATION_PASS": False,
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
