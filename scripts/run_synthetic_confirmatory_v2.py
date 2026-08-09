#!/usr/bin/env python3
"""Run or metadata-only dry-run the single frozen Confirmatory v2 manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")


def _assert_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("Confirmatory v2 requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError("Confirmatory v2 requires the frozen MAMBA_ROOT_PREFIX")
    source = SOURCE_REPOSITORY.resolve()
    entries = list(sys.path) + [
        value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("Python search path reaches the source repository")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Synthetic Confirmatory v2")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _assert_isolation()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from phase_a_harness.synthetic_confirmatory_v2_runner import (
        dry_run_synthetic_confirmatory,
        execute_synthetic_confirmatory,
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.resume:
            parser.error("--dry-run and --resume are mutually exclusive")
        result = dry_run_synthetic_confirmatory(
            manifest_path=args.manifest,
            run_id=args.run_id,
            output_dir=args.output_dir,
            workers=args.workers,
        )
    else:
        if not args.resume:
            parser.error("formal v2 execution requires --resume")
        result = execute_synthetic_confirmatory(
            manifest_path=args.manifest,
            run_id=args.run_id,
            output_dir=args.output_dir,
            workers=args.workers,
            resume=True,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
