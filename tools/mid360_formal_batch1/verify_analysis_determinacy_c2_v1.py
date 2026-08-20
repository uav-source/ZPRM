#!/usr/bin/env python3
"""Verify result-blind FMB1 analysis determinacy clarification C2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.analysis_determinacy_c2_verify_v1 import (  # noqa: E402
    C2VerificationError,
    verify_repository,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_repository(args.repository_root)
    except (C2VerificationError, OSError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "reason": str(error)}, indent=2, sort_keys=True))
        return 1
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        if json.loads(output.read_text(encoding="utf-8")) != report:
            print(json.dumps({"status": "FAIL", "reason": "persisted report mismatch"}))
            return 1
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
