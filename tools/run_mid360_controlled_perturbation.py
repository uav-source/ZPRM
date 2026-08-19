#!/usr/bin/env python3
"""Run or independently verify the frozen Mid-360 perturbation Pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from experiments.mid360_controlled_perturbation.benchmark import (  # noqa: E402
    ZERO_RUNTIME,
    run_benchmark,
    verify_completed_output,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "results/mid360_controlled_perturbation_pilot",
    )
    parser.add_argument("--runtime", type=Path, default=ZERO_RUNTIME)
    args = parser.parse_args()
    if args.command == "run":
        result = run_benchmark(
            REPOSITORY, args.output, runtime=args.runtime
        )
    else:
        result = verify_completed_output(
            REPOSITORY, args.output, runtime=args.runtime
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(result.get("checks", {}).values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
