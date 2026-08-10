#!/usr/bin/env python3
"""Prepare real-data preregistration artifacts without executing registration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.manifest import verify_frozen_manifest
from phase_a_harness.real_data_preparation.workflow import execute_preparation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--synthetic-run-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-registration", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.metadata_only and arguments.verify_only:
        raise SystemExit("--metadata-only and --verify-only are mutually exclusive")
    if arguments.verify_only:
        root = arguments.runtime_root.resolve(strict=True)
        manifest = json.loads((root / "frozen_manifest_v1.json").read_text(encoding="utf-8"))
        report = verify_frozen_manifest(root, manifest, repository_root=REPOSITORY)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["preregistration_verification_pass"] else 2
    if arguments.metadata_only:
        print("metadata-only is represented by the ordered reference-first phase of the full preparation; use --mode fresh or resume without this flag for an immutable audit")
        return 2
    if arguments.no_registration:
        os.environ["ZPRM_REAL_DATA_PREP_NO_REGISTRATION"] = "1"
    report = execute_preparation(
        repository=arguments.repository_root,
        data_root=arguments.data_root,
        runtime_root=arguments.runtime_root,
        synthetic_run_root=arguments.synthetic_run_root,
        mode=arguments.mode,
        workers=arguments.workers,
        no_registration=arguments.no_registration,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verification"]["preregistration_verification_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
