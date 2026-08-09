#!/usr/bin/env python3
"""Prepare and verify the frozen 42-snapshot Phase B signal cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.phase_b_snapshot_assets import prepare_phase_b_signal_assets


result = prepare_phase_b_signal_assets(ROOT)
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
