#!/usr/bin/env python3
"""Build the FMB1 bag manifest from retained pair metadata records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.protocol import build_bag_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "results/mid360_formal_batch1/bag_manifest.json",
    )
    args = parser.parse_args()
    metadata = sorted(args.bag_root.resolve(strict=True).rglob("*.pair.json"))
    if not metadata:
        parser.error("no *.pair.json metadata files found")
    manifest = build_bag_manifest(metadata, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
