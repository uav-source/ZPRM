#!/usr/bin/env python3
"""Independently verify an FMB1 Zero-Perturbation exec-r2 lock."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for location in (REPOSITORY / "src", REPOSITORY):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_exec_r2_lock import (  # noqa: E402
    R2_RESULTS_DIR,
    canonical_json_bytes,
)
from experiments.mid360_formal_batch1.zero_perturbation_v1_1_exec_r2_verify import (  # noqa: E402
    verify_exec_r2_lock,
    verify_release_checksums,
)


def _write_once(path: Path, payload: dict[str, object]) -> None:
    content = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite different verifier report: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--lock-dir", type=Path)
    parser.add_argument("--expected-execution-code-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-release-checksums", action="store_true")
    args = parser.parse_args(argv)
    root = args.repository_root.expanduser().resolve(strict=True)
    lock_dir = (args.lock_dir or root / R2_RESULTS_DIR).expanduser().resolve(strict=True)
    report = verify_exec_r2_lock(
        root, lock_dir,
        expected_execution_code_commit=args.expected_execution_code_commit,
    )
    if args.verify_release_checksums:
        report = {**report, "release_checksums": verify_release_checksums(lock_dir)}
    output = args.output or lock_dir / "independent_verification.json"
    _write_once(output.expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
