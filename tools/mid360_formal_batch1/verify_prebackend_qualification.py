#!/usr/bin/env python3
"""Independently verify the FMB1 W04 pre-backend qualification evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.prebackend_verify import (  # noqa: E402
    verify_prebackend_qualification,
)


DEFAULT_EVIDENCE_DIR = (
    REPOSITORY / "results/mid360_formal_batch1/prebackend_qualification_v1"
)


def _write_report_once(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite a different verifier report: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence_dir = args.evidence_dir.expanduser().resolve()
    output = args.output or evidence_dir / "prebackend_independent_verification.json"
    try:
        report = verify_prebackend_qualification(
            args.repository_root.expanduser(), evidence_dir
        )
        _write_report_once(output.expanduser().resolve(), report)
    except Exception as exc:
        failure = {
            "schema": "mid360_fmb1_prebackend_independent_verification_v1",
            "status": "FAIL",
            "pass": False,
            "failure_count": 1,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "FMB1_PREBACKEND_EXECUTION_PATH_QUALIFIED": False,
            "FMB1_CURRENT_REAL_BATCH_BLOCKED": True,
            "FMB1_CURRENT_BLOCK_REASON": "MISSING_ADMITTED_WEAK_REPLACEMENT_W04",
            "FORMAL_LOCK_ISSUED": False,
            "FORMAL_ICP_UNLOCKED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "actual_formal_trials": 0,
        }
        try:
            _write_report_once(output.expanduser().resolve(), failure)
        except RuntimeError:
            pass
        print(json.dumps(failure, sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0 if report.get("pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
