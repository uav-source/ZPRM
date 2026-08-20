#!/usr/bin/env python3
"""Finalize SHA256SUMS after every R1 lock report has been written."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for location in (REPOSITORY / "src", REPOSITORY):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_lock import (  # noqa: E402
    finalize_lock_directory_checksums,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=REPOSITORY / "results/mid360_formal_batch1/zero_perturbation_v1_1_lock",
    )
    args = parser.parse_args(argv)
    path = finalize_lock_directory_checksums(args.lock_dir.expanduser())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
