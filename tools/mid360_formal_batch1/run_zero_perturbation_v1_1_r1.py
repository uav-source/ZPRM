#!/usr/bin/env python3
"""Preflight/dry-run the locked FMB1 zero-perturbation v1.1-R1 plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for location in (REPOSITORY / "src", REPOSITORY):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_runner import (  # noqa: E402
    preflight_or_dry_run,
)


def _write_once(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"refusing to overwrite different dry-run report: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--lock-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--authorization-path", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    repository = args.repository_root.expanduser().resolve(strict=True)
    lock_dir = args.lock_dir or repository / (
        "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r2_lock"
    )
    action = "preflight" if args.preflight else "dry-run" if args.dry_run else "execute"
    report = preflight_or_dry_run(
        repository,
        lock_dir=lock_dir,
        plan_path=args.plan,
        authorization_path=args.authorization_path,
        runtime_root=args.runtime_root,
        workers=args.workers,
        action=action,
        mode=args.mode,
        remeasure_environment=True,
    )
    if args.report:
        _write_once(args.report.expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
