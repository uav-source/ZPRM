#!/usr/bin/env python3
"""Independently verify and freeze the nonformal two-scene Mid-360 Pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_a_harness.mid360_two_scene_pilot.pipeline import (
    DEFAULT_FROZEN_ROOT,
    DEFAULT_RUNTIME_ROOT,
)
from phase_a_harness.mid360_two_scene_pilot.verifier import (
    finalize_verification_and_freeze,
    verify_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument(
        "--finalize-and-freeze",
        action="store_true",
        help="write the verifier report and freeze small Git metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.finalize_and_freeze:
        result = finalize_verification_and_freeze(args.runtime_root, args.frozen_root)
    else:
        result = verify_runtime(args.runtime_root)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
