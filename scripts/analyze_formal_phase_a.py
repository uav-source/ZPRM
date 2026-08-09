#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.analysis import analyze_formal_phase_a


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, default=ROOT / "frozen_assets/frozen_experiment_manifest.json")
parser.add_argument("--run-dir", type=Path, default=ROOT / "results/formal_phase_a_v1")
args = parser.parse_args()
result = analyze_formal_phase_a(
    manifest_path=args.manifest, run_dir=args.run_dir, output_path=args.run_dir / "primary_analysis.json"
)
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))

