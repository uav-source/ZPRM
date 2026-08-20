#!/usr/bin/env python3
"""Independently verify the FMB1 zero-perturbation R1 plan/schema assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.zero_perturbation_r1_trial_verify import (  # noqa: E402
    R1TrialVerificationError,
    verify_assets,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--skip-payload-byte-hashes",
        action="store_true",
        help="For unit fixtures only; formal verification must not set this flag.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON evidence path; omitted for stdout-only verification.",
    )
    args = parser.parse_args(argv)
    os.environ["NO_FORMAL_REGISTRATION"] = "true"
    os.environ["ZPRM_FMB1_NO_FORMAL_REGISTRATION"] = "1"
    try:
        report = verify_assets(
            repository=args.repository.resolve(),
            verify_payload_files=not args.skip_payload_byte_hashes,
        )
    except (OSError, ValueError, R1TrialVerificationError) as error:
        print(json.dumps({"status": "FAIL", "reason": str(error)}, indent=2, sort_keys=True))
        return 1
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        persisted = json.loads(output.read_text(encoding="utf-8"))
        if persisted != report:
            print(
                json.dumps(
                    {"status": "FAIL", "reason": "persisted report self-check failed"},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
