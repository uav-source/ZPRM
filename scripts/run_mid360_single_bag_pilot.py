#!/usr/bin/env python3
"""Run the PILOT_ONLY Mid-360 single-bag audit or preparation phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_a_harness.mid360_pilot.pilot_pipeline import (
    DEFAULT_BACKEND_PARAMETER_CONTRACT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_PCL_EXECUTABLE,
    run_phase,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "zero_perturbation_data" / "mid360_pilot_v1",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--backend-parameter-contract",
        type=Path,
        default=DEFAULT_BACKEND_PARAMETER_CONTRACT,
    )
    parser.add_argument(
        "--pcl-executable", type=Path, default=DEFAULT_PCL_EXECUTABLE
    )
    parser.add_argument(
        "--phase",
        choices=(
            "audit",
            "prepare",
            "debug-inputs",
            "debug-registration",
            "debug-finalize",
            "debug-classify-limitations",
        ),
        default="audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_phase(
        args.phase,
        args.bag,
        args.runtime_root,
        config_path=args.config,
        data_root=args.data_root,
        backend_parameter_contract=args.backend_parameter_contract,
        pcl_executable=args.pcl_executable,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
