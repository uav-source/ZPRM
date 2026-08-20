#!/usr/bin/env python3
"""Independently verify the W04 replacement and final FMB1 data freeze."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from experiments.mid360_formal_batch1.w04_final_verify import (  # noqa: E402
    verify_w04_final_dataset,
)


DEFAULT_FINAL_DIR = REPOSITORY / "results/mid360_formal_batch1/final_dataset_v1"


def _write_report_once(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite a different verifier report: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary verifier report exists: {temporary}")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-file-rehash",
        action="store_true",
        help="test-only structural mode; formal verification must omit this option",
    )
    parser.add_argument(
        "--no-checksum-verification",
        action="store_true",
        help="test-only structural mode; formal verification must omit this option",
    )
    args = parser.parse_args(argv)
    final_dir = args.final_dir.expanduser().resolve()
    output = args.output or final_dir / "independent_verification.json"
    try:
        report = verify_w04_final_dataset(
            args.repository_root.expanduser(),
            final_dir,
            verify_files=not args.no_file_rehash,
            verify_checksums=not args.no_checksum_verification,
        )
        report["verify_files"] = not args.no_file_rehash
        report["verify_checksums"] = not args.no_checksum_verification
        if args.no_file_rehash or args.no_checksum_verification:
            report["formal_qualification"] = False
            report["status"] = "STRUCTURAL_TEST_ONLY"
            report["pass"] = False
            exit_code = 2
        else:
            report["formal_qualification"] = True
            exit_code = 0
        _write_report_once(output.expanduser().resolve(), report)
    except Exception as exc:
        failure: dict[str, object] = {
            "schema": "mid360_fmb1_w04_final_independent_verification_v1",
            "status": "FAIL",
            "pass": False,
            "failure_count": 1,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "FMB1_FINAL_DATASET_READY": False,
            "READY_FOR_ZERO_PERTURBATION_AMENDMENT_ACTIVATION": False,
            "FORMAL_LOCK_ISSUED": False,
            "FORMAL_ICP_UNLOCKED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "actual_open3d_trials": 0,
            "actual_pcl_trials": 0,
            "actual_formal_trials": 0,
        }
        try:
            _write_report_once(output.expanduser().resolve(), failure)
        except RuntimeError:
            pass
        print(json.dumps(failure, sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

