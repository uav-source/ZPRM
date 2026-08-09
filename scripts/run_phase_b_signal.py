#!/usr/bin/env python3
"""Command-line entry point for the one frozen Phase B signal experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.contracts import write_json
from phase_a_harness.phase_b_runner import dry_run_phase_b, execute_phase_b


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--workers", type=int, required=True)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

if args.dry_run:
    result = dry_run_phase_b(
        manifest_path=args.manifest,
        run_id=args.run_id,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    write_json(ROOT / "artifacts/phase_b_dry_run_report.json", result)
else:
    result = execute_phase_b(
        manifest_path=args.manifest,
        run_id=args.run_id,
        output_dir=args.output_dir,
        workers=args.workers,
        resume=args.resume,
    )
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
