#!/usr/bin/env python3
"""Independently re-read and verify Synthetic Confirmatory evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _assert_preimport_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("Synthetic Confirmatory scripts require PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != "/home/lj/.local/share/degen-lio-micromamba":
        raise PermissionError("Synthetic Confirmatory scripts require the frozen MAMBA_ROOT_PREFIX")
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
from phase_a_harness.synthetic_confirmatory_independent_verifier import (
    compare_primary_and_independent,
    independently_verify_synthetic_confirmatory,
)


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--primary", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

# Raw evidence is independently authenticated before a primary-analysis file is used.
report = independently_verify_synthetic_confirmatory(
    manifest_path=args.manifest, run_dir=args.run_dir
)
primary = json.loads(args.primary.read_text(encoding="utf-8"))
comparison = compare_primary_and_independent(primary, report)
report["analysis_verifier_comparison"] = comparison
report["ANALYSIS_VERIFIER_DIFFERENCE_COUNT"] = comparison["leaf_difference_count"]
write_json(args.output, report)
print(json.dumps({
    "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": comparison["leaf_difference_count"],
    "output": str(args.output.resolve()),
}, indent=2, sort_keys=True, allow_nan=False))
