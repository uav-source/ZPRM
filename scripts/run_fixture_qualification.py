#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.fixture_qualification import qualify_fixtures


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, default=ROOT / "frozen_assets/frozen_experiment_manifest.json")
parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/fixture_qualification")
args = parser.parse_args()
print(json.dumps(qualify_fixtures(manifest_path=args.manifest, output_dir=args.output_dir), indent=2, sort_keys=True, allow_nan=False))

