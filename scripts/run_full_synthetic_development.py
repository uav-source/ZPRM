#!/usr/bin/env python3
"""Run the frozen Full Synthetic Development dry-run/subset/formal matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _assert_preimport_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("Full Synthetic scripts require PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != "/home/lj/.local/share/degen-lio-micromamba":
        raise PermissionError(
            "Full Synthetic scripts require the frozen MAMBA_ROOT_PREFIX"
        )
    source = Path("/home/lj/Degen-LIO").resolve()
    entries = list(sys.path) + [
        item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("Python search path resolves to the source repository")


_assert_preimport_isolation()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.contracts import write_json
from phase_a_harness.full_synthetic_development_protocol import DRY_RUN_REPORT_RELATIVE
from phase_a_harness.full_synthetic_development_runner import (
    dry_run_full_synthetic,
    execute_full_synthetic,
)


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--workers", type=int, required=True)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--phase-b-subset-only", action="store_true")
args = parser.parse_args()
if args.dry_run and args.phase_b_subset_only:
    parser.error("--dry-run and --phase-b-subset-only are mutually exclusive")

if args.dry_run:
    result = dry_run_full_synthetic(
        manifest_path=args.manifest,
        run_id=args.run_id,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    write_json(ROOT / DRY_RUN_REPORT_RELATIVE, result)
else:
    result = execute_full_synthetic(
        manifest_path=args.manifest,
        run_id=args.run_id,
        output_dir=args.output_dir,
        workers=args.workers,
        resume=args.resume,
        phase_b_subset_only=args.phase_b_subset_only,
    )
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
