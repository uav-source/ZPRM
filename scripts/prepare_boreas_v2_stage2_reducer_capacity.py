#!/usr/bin/env python3
"""Compile/probe the Stage-2 reducer and freeze its production capacity plan."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_v2_stage2_runner import (  # noqa: E402
    MINIMUM_REDUCER_SAFETY_MARGIN_BYTES,
    generate_production_reducer_resource_plan,
)
from phase_a_harness.real_data_preparation.io import canonical_json_bytes  # noqa: E402
from phase_a_harness.real_data_preparation.guard import NoRegistrationGuard  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the pinned C++ voxel reducer, query its ABI/layout upper "
            "bound, independently recompute it, and freeze a replay-independent plan."
        )
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--safety-margin-bytes",
        type=int,
        default=MINIMUM_REDUCER_SAFETY_MARGIN_BYTES,
    )
    args = parser.parse_args()
    runtime = args.runtime_root.resolve(strict=True)
    output = runtime / "checkpoints/reducer_resource_plan.json"
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise RuntimeError(
            "runtime/checkpoints must be an existing canonical directory"
        )
    os.environ.setdefault("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard():
        result = generate_production_reducer_resource_plan(
            output_plan_path=output,
            reducer_binary_path=runtime / "map/boreas_stage2_voxel_reduce",
            safety_margin_bytes=args.safety_margin_bytes,
        )
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
