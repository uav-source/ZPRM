#!/usr/bin/env python3
"""Freeze protocol/plans, build snapshots, and manage the single manifest."""

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
        raise PermissionError(
            "Full Synthetic scripts require the frozen MAMBA_ROOT_PREFIX"
        )
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

from phase_a_harness.full_synthetic_development_protocol import (
    authorize_full_synthetic_manifest_once,
    create_unauthorized_full_synthetic_manifest,
    write_full_synthetic_plans,
    write_full_synthetic_protocol,
)
from phase_a_harness.full_synthetic_snapshot_builder import (
    prepare_full_synthetic_snapshots,
)


parser = argparse.ArgumentParser()
actions = parser.add_mutually_exclusive_group(required=True)
actions.add_argument("--protocol-and-plans", action="store_true")
actions.add_argument("--snapshots", action="store_true")
actions.add_argument("--create-manifest", action="store_true")
actions.add_argument("--authorize", action="store_true")
args = parser.parse_args()

if args.protocol_and_plans:
    result = {
        "plans": write_full_synthetic_plans(ROOT),
        "protocol": write_full_synthetic_protocol(ROOT),
    }
elif args.snapshots:
    result = prepare_full_synthetic_snapshots(ROOT)
elif args.create_manifest:
    result = create_unauthorized_full_synthetic_manifest(ROOT)
else:
    result = authorize_full_synthetic_manifest_once(ROOT)
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
