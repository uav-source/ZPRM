#!/usr/bin/env python3
"""Prepare or execute the nonformal independent two-scene Mid-360 Pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_a_harness.mid360_pilot.pilot_pipeline import (
    DEFAULT_BACKEND_PARAMETER_CONTRACT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_PCL_EXECUTABLE,
)
from phase_a_harness.mid360_two_scene_pilot.pipeline import (
    DEFAULT_RUNTIME_ROOT,
    prepare_two_scene_pilot,
)
from phase_a_harness.mid360_two_scene_pilot.registration import (
    execute_two_scene_registration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-root", type=Path, default=Path("/home/lj/ZPRM/bag"))
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--backend-parameter-contract",
        type=Path,
        default=DEFAULT_BACKEND_PARAMETER_CONTRACT,
    )
    parser.add_argument("--pcl-executable", type=Path, default=DEFAULT_PCL_EXECUTABLE)
    parser.add_argument("--phase", choices=("prepare", "register"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "prepare":
        result = prepare_two_scene_pilot(
            args.bag_root,
            args.runtime_root,
            config_path=args.config,
            parameter_contract_path=args.backend_parameter_contract,
        )
    else:
        result = execute_two_scene_registration(
            args.runtime_root,
            parameter_contract_path=args.backend_parameter_contract,
            pcl_executable=args.pcl_executable,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
