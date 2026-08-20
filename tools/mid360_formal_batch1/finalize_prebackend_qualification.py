#!/usr/bin/env python3
"""Finalize the non-authorizing FMB1 pre-backend summary and checksum manifest."""

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
    build_qualification_summary,
    render_qualification_summary_markdown,
    write_json_once,
    write_qualification_sha256sums,
    write_text_once,
)


DEFAULT_OUTPUT = (
    REPOSITORY / "results/mid360_formal_batch1/prebackend_qualification_v1"
)


def _test_counts(
    collected: int,
    passed: int,
    skipped: int,
    failed: int,
    errors: int,
) -> dict[str, int]:
    return {
        "collected": collected,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-collected", type=int, required=True)
    parser.add_argument("--baseline-passed", type=int, required=True)
    parser.add_argument("--baseline-skipped", type=int, required=True)
    parser.add_argument("--fmb1-collected", type=int, required=True)
    parser.add_argument("--fmb1-passed", type=int, required=True)
    parser.add_argument("--fmb1-skipped", type=int, required=True)
    parser.add_argument("--full-collected", type=int, required=True)
    parser.add_argument("--full-passed", type=int, required=True)
    parser.add_argument("--full-skipped", type=int, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("ZPRM_FMB1_NO_FORMAL_REGISTRATION") != "1":
        print("ZPRM_FMB1_NO_FORMAL_REGISTRATION=1 is required", file=sys.stderr)
        return 2
    output_dir = args.output_dir.expanduser().resolve(strict=True)
    try:
        summary = build_qualification_summary(
            output_dir,
            baseline_tests=_test_counts(
                args.baseline_collected,
                args.baseline_passed,
                args.baseline_skipped,
                0,
                0,
            ),
            fmb1_tests=_test_counts(
                args.fmb1_collected,
                args.fmb1_passed,
                args.fmb1_skipped,
                0,
                0,
            ),
            full_tests=_test_counts(
                args.full_collected,
                args.full_passed,
                args.full_skipped,
                0,
                0,
            ),
        )
        write_json_once(output_dir / "prebackend_qualification_summary.json", summary)
        write_text_once(
            output_dir / "prebackend_qualification_summary.md",
            render_qualification_summary_markdown(summary),
        )
        write_qualification_sha256sums(output_dir)
    except (PrebackendQualificationError, OSError, ValueError) as exc:
        print(f"FMB1_PREBACKEND_FINALIZE_FAIL: {exc}", file=sys.stderr)
        return 2
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
