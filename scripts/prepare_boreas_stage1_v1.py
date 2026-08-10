#!/usr/bin/env python3
"""Run or independently verify the Boreas metadata/GT-only Stage-1 audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_stage1 import (  # noqa: E402
    execute_boreas_stage1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-single-download-bytes", type=int, default=500_000_000)
    parser.add_argument("--max-total-download-bytes", type=int, default=5_000_000_000)
    parser.add_argument("--no-registration", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.verify_only:
        from phase_a_harness.real_data_preparation.boreas_stage1_verifier import (
            verify_boreas_stage1,
        )

        result = verify_boreas_stage1(
            repository=arguments.repository_root,
            data_root=arguments.data_root,
            runtime_root=arguments.runtime_root,
        )
    else:
        result = execute_boreas_stage1(
            repository=arguments.repository_root,
            data_root=arguments.data_root,
            runtime_root=arguments.runtime_root,
            mode=arguments.mode,
            metadata_only=arguments.metadata_only,
            maximum_single_download_bytes=arguments.max_single_download_bytes,
            maximum_total_download_bytes=arguments.max_total_download_bytes,
            no_registration=arguments.no_registration,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
