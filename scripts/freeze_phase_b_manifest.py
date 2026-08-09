#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.phase_b_manifest import (
    authorize_phase_b_manifest_once,
    create_phase_b_pre_run_gate_report,
    create_unauthorized_phase_b_manifest,
)


parser = argparse.ArgumentParser()
action = parser.add_mutually_exclusive_group(required=True)
action.add_argument("--create", action="store_true")
action.add_argument("--gate-report", action="store_true")
action.add_argument("--authorize", action="store_true")
args = parser.parse_args()

if args.create:
    result = create_unauthorized_phase_b_manifest(ROOT)
elif args.gate_report:
    result = create_phase_b_pre_run_gate_report(ROOT)
else:
    result = authorize_phase_b_manifest_once(ROOT)
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
