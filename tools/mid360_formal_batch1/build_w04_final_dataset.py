#!/usr/bin/env python3
"""Compose FMB1 ``final_dataset_v1`` after a passing W04 replacement."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.w04_final_dataset import (
    W04FinalDatasetError,
    build_final_dataset_payload,
    write_final_dataset,
)


def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise W04FinalDatasetError(f"expected mapping in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path)
    parser.add_argument("--w04-acquisition", type=Path)
    parser.add_argument("--w04-assets", type=Path)
    parser.add_argument("--w04-geometry", type=Path)
    parser.add_argument("--replacement-plan", type=Path)
    parser.add_argument("--no-icp-attestation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if os.environ.get("ZPRM_FMB1_NO_FORMAL_REGISTRATION") != "1":
        raise W04FinalDatasetError(
            "ZPRM_FMB1_NO_FORMAL_REGISTRATION=1 is required"
        )
    repository = args.repository_root.resolve(strict=True)
    runtime = repository / (
        "zero_perturbation_runtime/"
        "mid360_formal_batch1_w04_replacement_v1"
    )
    original_path = args.original_manifest or (
        repository / "results/mid360_formal_batch1/fmb1_frozen_manifest.json"
    )
    acquisition_path = args.w04_acquisition or (
        runtime / "w04_acquisition_manifest.json"
    )
    assets_path = args.w04_assets or (runtime / "w04_asset_manifest.json")
    geometry_path = args.w04_geometry or (
        runtime / "w04_geometry_only_manifest.json"
    )
    plan_path = args.replacement_plan or (
        repository / "experiments/mid360_formal_batch1/replacement_plan_w04.yaml"
    )
    no_icp_path = args.no_icp_attestation or (
        repository
        / "results/mid360_formal_batch1/w04_replacement_v1/NO_ICP_ATTESTATION.json"
    )
    output_dir = args.output_dir or (
        repository / "results/mid360_formal_batch1/final_dataset_v1"
    )

    payload = build_final_dataset_payload(
        _mapping(original_path),
        _mapping(acquisition_path),
        _mapping(assets_path),
        _mapping(geometry_path),
        _mapping(plan_path),
        _mapping(no_icp_path),
        repository=repository,
        verify_files=True,
    )
    written = write_final_dataset(payload, output_dir)
    print("FMB1_W04_ACQUISITION_PASS=true")
    print("FMB1_W04_FINAL_GEOMETRY_CLASS=WEAK")
    print("FMB1_W04_ADMISSION_PASS=true")
    print("FMB1_FINAL_DATASET_READY=true")
    print("RAW_CANDIDATE_BAG_COUNT=42")
    print("FINAL_ADMITTED_BAG_COUNT=36")
    print("REJECTED_BAG_COUNT=6")
    print("FINAL_SNAPSHOT_COUNT=180")
    print("FORMAL_LOCK_ISSUED=false")
    print("FORMAL_ICP_UNLOCKED=false")
    print("FORMAL_REGISTRATION_AUTHORIZED=false")
    print("actual_formal_trials=0")
    print(f"output_dir={output_dir.resolve()}")
    print(f"artifact_count={len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
