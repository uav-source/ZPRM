#!/usr/bin/env python3
"""Independently verify the FMB1 zero-perturbation v1.1-R1 formal lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for location in (REPOSITORY / "src", REPOSITORY):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_verify import (  # noqa: E402
    verify_release_checksums,
    verify_r1_lock,
)


def _write_once(path: Path, payload: dict[str, object]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing to overwrite different independent report: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--lock-dir", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--expected-execution-code-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--release-checksums-only", action="store_true",
        help="verify finalized SHA256SUMS and write its report outside lock-dir",
    )
    args = parser.parse_args(argv)
    repository = args.repository_root.expanduser().resolve(strict=True)
    lock_dir = args.lock_dir or repository / (
        "results/mid360_formal_batch1/zero_perturbation_v1_1_lock"
    )
    commit = args.expected_execution_code_commit
    if args.release_checksums_only:
        output = args.output or lock_dir.parent / (
            "zero_perturbation_v1_1_release_checksum_verification.json"
        )
    else:
        output = args.output or lock_dir / "independent_verification.json"
    try:
        if args.release_checksums_only:
            try:
                output.resolve().relative_to(lock_dir.resolve())
            except ValueError:
                pass
            else:
                raise RuntimeError("release checksum report must be outside lock-dir")
            report = verify_release_checksums(lock_dir)
        else:
            report = verify_r1_lock(
                repository,
                lock_dir,
                expected_execution_code_commit=commit,
                runtime_root=args.runtime_root,
                remeasure_environment=True,
                remeasure_execution_commit=True,
            )
        _write_once(output, report)
    except Exception as exc:
        failure: dict[str, object] = {
            "schema": "mid360_fmb1_zero_perturbation_lock_independent_verification_v1_1_r1",
            "status": "FAIL",
            "pass": False,
            "failure_count": 1,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "FORMAL_LOCK_ISSUED": False,
            "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION": False,
            "FORMAL_ICP_UNLOCKED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "actual_formal_trials": 0,
        }
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
