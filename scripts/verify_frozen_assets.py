#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.asset_verifier import verify_frozen_assets


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, default=ROOT / "frozen_assets/frozen_experiment_manifest.json")
args = parser.parse_args()
print(json.dumps(verify_frozen_assets(args.manifest), indent=2, sort_keys=True, allow_nan=False))

