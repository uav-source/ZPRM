#!/usr/bin/env python3
"""Run the independent corrected W02-attempt-2 final-dataset verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for root in (REPOSITORY, REPOSITORY / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.mid360_formal_batch1.w02_attempt2_final_dataset import (  # noqa: E402
    refresh_sha256sums,
)
from experiments.mid360_formal_batch1.w02_attempt2_final_verify import (  # noqa: E402
    verify_w02_attempt2_final_dataset,
)
from experiments.mid360_formal_batch1.w04_outputs import write_json_once  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY / "results/mid360_formal_batch1/final_dataset_w02_attempt2_v1",
    )
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve(strict=True)
    report = verify_w02_attempt2_final_dataset(REPOSITORY, output)
    write_json_once(output / "independent_verification.json", report)
    refresh_sha256sums(output)
    # Verify the just-written report and refreshed checksum inventory too.
    report = verify_w02_attempt2_final_dataset(REPOSITORY, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
