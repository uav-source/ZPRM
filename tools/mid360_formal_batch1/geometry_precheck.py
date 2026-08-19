#!/usr/bin/env python3
"""Build registration-free FMB1 targets/snapshots and geometry admission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.protocol import build_geometry_assets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag-manifest",
        type=Path,
        default=REPOSITORY / "results/mid360_formal_batch1/bag_manifest.json",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPOSITORY / "results/mid360_formal_batch1",
    )
    args = parser.parse_args()
    config = json.loads(
        (REPOSITORY / "configs/mid360_pilot_config.json").read_text(encoding="utf-8")
    )
    targets, snapshots, geometry = build_geometry_assets(
        bag_manifest_path=args.bag_manifest,
        results_dir=args.results_dir,
        config=config,
    )
    print(
        json.dumps(
            {
                "target_count": targets["target_count"],
                "snapshot_count": snapshots["snapshot_count"],
                "scene_geometry": geometry["scenes"],
                "registration_executed": False,
                "FORMAL_ICP_UNLOCKED": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
