#!/usr/bin/env python3
"""Run primary Full Synthetic analysis and cached common association."""

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
        raise PermissionError("Full Synthetic scripts require the frozen MAMBA_ROOT_PREFIX")
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

from phase_a_harness.full_synthetic_analysis import analyze_full_synthetic_development


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--common-metrics", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

report = analyze_full_synthetic_development(
    manifest_path=args.manifest,
    run_dir=args.run_dir,
    common_metrics_path=args.common_metrics,
    output_path=args.output,
)
print(json.dumps({
    "final_decision": report["final_decision"],
    "output": str(args.output.resolve()),
    "schema_version": report["schema_version"],
}, indent=2, sort_keys=True, allow_nan=False))
