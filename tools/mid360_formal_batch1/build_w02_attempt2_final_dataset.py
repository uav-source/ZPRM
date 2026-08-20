#!/usr/bin/env python3
"""Archive invalid W02 attempt 1 and build the corrected final dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for root in (REPOSITORY, REPOSITORY / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.mid360_formal_batch1.w02_attempt2_final_dataset import (  # noqa: E402
    archive_w02_attempt1,
    build_corrected_final_dataset,
    write_corrected_final_dataset,
)
from experiments.mid360_formal_batch1.w04_outputs import write_json_once  # noqa: E402


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=REPOSITORY / "zero_perturbation_runtime/mid360_formal_batch1_w02_attempt2_v1",
    )
    parser.add_argument(
        "--attempt2-results",
        type=Path,
        default=REPOSITORY / "results/mid360_formal_batch1/w02_attempt2_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY / "results/mid360_formal_batch1/final_dataset_w02_attempt2_v1",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=REPOSITORY / "archive/invalid_acquisition/FMB1_W02_attempt1_wrong_location",
    )
    args = parser.parse_args(argv)
    os.environ["NO_FORMAL_REGISTRATION"] = "true"
    os.environ["ZPRM_REAL_DATA_PREP_NO_REGISTRATION"] = "1"
    os.environ["ZPRM_FMB1_NO_FORMAL_REGISTRATION"] = "1"
    runtime = args.runtime_dir.expanduser().resolve(strict=True)
    results = args.attempt2_results.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    original = _load(REPOSITORY / "results/mid360_formal_batch1/fmb1_frozen_manifest.json")
    acquisition = _load(runtime / "w02_attempt2_acquisition_manifest.json")
    assets = _load(runtime / "w02_attempt2_asset_manifest.json")
    geometry = _load(runtime / "w02_attempt2_geometry_only_manifest.json")
    attestation = _load(results / "NO_ICP_ATTESTATION.json")
    archive = archive_w02_attempt1(
        original,
        repository=REPOSITORY,
        archive_dir=args.archive_dir,
    )
    write_json_once(
        args.archive_dir / "invalid_acquisition_archive_manifest.json", archive
    )
    payload = build_corrected_final_dataset(
        original,
        acquisition,
        assets,
        geometry,
        archive,
        attestation,
        repository=REPOSITORY,
    )
    write_corrected_final_dataset(payload, output, repository=REPOSITORY)
    print(
        json.dumps(
            {
                "status": "PASS",
                "final_scene_ids": list(
                    row["scene_id"]
                    for row in payload["final_scene_registry"]["scenes"]
                ),
                "w02_attempt1_status": "INVALID_ACQUISITION",
                "w02_attempt2_status": "GEOMETRY_ADMITTED",
                "W04_INCLUDED_IN_FINAL_SET": False,
                "FORMAL_REGISTRATION_AUTHORIZED": False,
                "actual_formal_trials": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
