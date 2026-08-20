#!/usr/bin/env python3
"""Build immutable, registration-free core evidence for FMB1 pre-backend qualification."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.prebackend_qualification import (  # noqa: E402
    PrebackendQualificationError,
    write_core_qualification_evidence,
    write_text_once,
)


OUTPUT_DEFAULT = (
    REPOSITORY
    / "results/mid360_formal_batch1/prebackend_qualification_v1"
)

DESIGN_ARTIFACTS = (
    "protocol_alignment_audit.json",
    "protocol_alignment_audit.md",
    "zero_perturbation_mainline_amendment_v1_1_PROPOSED.json",
    "zero_perturbation_mainline_amendment_v1_1_PROPOSED.md",
    "formal_trial_result_schema_v1.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--skip-raw-bag-rehash",
        action="store_true",
        help="test-only shortcut; production evidence should re-hash all 36 bags",
    )
    args = parser.parse_args()
    if os.environ.get("ZPRM_FMB1_NO_FORMAL_REGISTRATION") != "1":
        print("ZPRM_FMB1_NO_FORMAL_REGISTRATION=1 is required", file=sys.stderr)
        return 2
    repository = args.repository_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        write_core_qualification_evidence(
            repository,
            output_dir,
            verify_raw_bag_bytes=not args.skip_raw_bag_rehash,
        )
        source_dir = repository / "experiments/mid360_formal_batch1"
        for name in DESIGN_ARTIFACTS:
            source = source_dir / name
            if not source.is_file():
                raise PrebackendQualificationError(
                    f"required design artifact is missing: {source}"
                )
            write_text_once(
                output_dir / name,
                source.read_text(encoding="utf-8"),
            )
    except PrebackendQualificationError as exc:
        print(f"FMB1_PREBACKEND_CORE_EVIDENCE_FAIL: {exc}", file=sys.stderr)
        return 2
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
