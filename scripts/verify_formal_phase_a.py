#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.contracts import write_json
from phase_a_harness.independent_verifier import comparison_difference_count, independently_verify_formal_phase_a


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, default=ROOT / "frozen_assets/frozen_experiment_manifest.json")
parser.add_argument("--run-dir", type=Path, default=ROOT / "results/formal_phase_a_v1")
args = parser.parse_args()
primary = json.loads((args.run_dir / "primary_analysis.json").read_text(encoding="utf-8"))
independent = independently_verify_formal_phase_a(manifest_path=args.manifest, run_dir=args.run_dir)
independent["ANALYSIS_VERIFIER_DIFFERENCE_COUNT"] = comparison_difference_count(primary, independent)
write_json(args.run_dir / "independent_verification.json", independent)
print(json.dumps(independent, indent=2, sort_keys=True, allow_nan=False))

