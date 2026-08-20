#!/usr/bin/env python3
"""Issue the closed, explicit-provenance FMB1 Exec-R3 lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for location in (REPOSITORY / "src", REPOSITORY):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_exec_r3_lock import (  # noqa: E402
    R3_RESULTS_DIR,
    build_exec_r3_lock,
    write_exec_r3_lock,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--execution-code-commit", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--issued-at-utc")
    args = parser.parse_args(argv)
    root = args.repository_root.expanduser().resolve(strict=True)
    output = args.output_dir or root / R3_RESULTS_DIR
    lock, inventory = build_exec_r3_lock(
        root,
        execution_code_commit=args.execution_code_commit,
        issued_at_utc=args.issued_at_utc,
    )
    report = write_exec_r3_lock(root, output, lock, inventory)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
