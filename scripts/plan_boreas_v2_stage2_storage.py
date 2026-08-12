#!/usr/bin/env python3
"""Build and optionally freeze the metadata-only Boreas v2 Stage-2 storage audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_stage2_storage_optimization import (  # noqa: E402
    build_boreas_stage2_storage_optimization,
    freeze_boreas_stage2_storage_optimization,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "zero_perturbation_data/boreas_stage1_v1",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path.home()
        / "zero_perturbation_runtime/real_data/boreas_v2_stage2_storage_optimization",
    )
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--pytest-junit-xml", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    summary = build_boreas_stage2_storage_optimization(
        repository=arguments.repository_root,
        data_root=arguments.data_root,
        runtime_root=arguments.runtime_root,
        pytest_junit_xml=arguments.pytest_junit_xml,
        require_clean_worktree=True,
    )
    if arguments.frozen_root is not None:
        freeze_boreas_stage2_storage_optimization(
            repository=arguments.repository_root,
            data_root=arguments.data_root,
            runtime_root=arguments.runtime_root,
            frozen_root=arguments.frozen_root,
        )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
