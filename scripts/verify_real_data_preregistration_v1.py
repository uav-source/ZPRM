#!/usr/bin/env python3
"""Independently verify a real-data preregistration manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.manifest import (
    ManifestVerificationError,
    verify_frozen_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    root = arguments.runtime_root.resolve(strict=True)
    try:
        manifest = json.loads((root / "frozen_manifest_v1.json").read_text(encoding="utf-8"))
        report = verify_frozen_manifest(root, manifest, repository_root=REPOSITORY)
    except (OSError, ValueError, ManifestVerificationError) as error:
        print(json.dumps({"artifact_integrity_pass": False, "error": str(error)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["preregistration_verification_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
