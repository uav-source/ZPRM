#!/usr/bin/env python3
"""Create the FMB1 analysis lock only after every preregistered gate closes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.protocol import (  # noqa: E402
    FormalBatchError,
    freeze_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPOSITORY / "results/mid360_formal_batch1",
    )
    args = parser.parse_args()
    try:
        fingerprint = freeze_batch(REPOSITORY, args.results_dir)
    except FormalBatchError as exc:
        print(
            json.dumps(
                {
                    "FORMAL_BATCH1_FROZEN": False,
                    "FORMAL_ICP_UNLOCKED": False,
                    "FORMAL_MEASUREMENT_RESULT": False,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(fingerprint, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
