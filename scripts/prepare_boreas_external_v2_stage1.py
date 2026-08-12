#!/usr/bin/env python3
"""Build or publish the metadata/GT-only Boreas external v2 Stage-1 closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_external_v2_stage1 import (  # noqa: E402
    build_boreas_external_v2_stage1,
    freeze_runtime_assets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--source-only-collected", type=int, required=True)
    parser.add_argument("--source-only-passed", type=int, required=True)
    parser.add_argument("--source-only-skipped", type=int, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    summary = build_boreas_external_v2_stage1(
        repository=arguments.repository_root,
        data_root=arguments.data_root,
        runtime_root=arguments.runtime_root,
        source_only_collected=arguments.source_only_collected,
        source_only_passed=arguments.source_only_passed,
        source_only_skipped=arguments.source_only_skipped,
    )
    if arguments.frozen_root is not None:
        freeze_runtime_assets(
            runtime_root=arguments.runtime_root,
            frozen_root=arguments.frozen_root,
        )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
