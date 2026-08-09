#!/usr/bin/env python3
"""Publish and verify the compact Full Synthetic artifact."""

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

from phase_a_harness.full_synthetic_publisher import publish_full_synthetic_development
from phase_a_harness.full_synthetic_development_protocol import (
    load_strict_authorized_full_synthetic_manifest,
)


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--primary", type=Path, required=True)
parser.add_argument("--independent", type=Path, required=True)
parser.add_argument("--artifact-dir", type=Path, required=True)
args = parser.parse_args()

# Authenticate the complete frozen closure before reading either analysis file.
load_strict_authorized_full_synthetic_manifest(args.manifest)

result = publish_full_synthetic_development(
    manifest_path=args.manifest,
    run_dir=args.run_dir,
    primary=json.loads(args.primary.read_text(encoding="utf-8")),
    independent=json.loads(args.independent.read_text(encoding="utf-8")),
    artifact_dir=args.artifact_dir,
)
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
