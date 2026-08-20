#!/usr/bin/env python3
"""Generate the backend-free FMB1 zero-perturbation R1 plan/schema assets."""

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

from experiments.mid360_formal_batch1.zero_perturbation_r1_trial_assets import (  # noqa: E402
    FMB1_ROOT,
    generate_and_write,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=FMB1_ROOT)
    args = parser.parse_args(argv)
    os.environ["NO_FORMAL_REGISTRATION"] = "true"
    os.environ["ZPRM_FMB1_NO_FORMAL_REGISTRATION"] = "1"
    report = generate_and_write(output_root=args.output_root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
