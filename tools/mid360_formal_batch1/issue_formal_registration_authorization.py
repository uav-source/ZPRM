#!/usr/bin/env python3
"""Issue one immutable, lock-matched FMB1 Exec-R3 authorization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for location in (REPOSITORY / "src", REPOSITORY):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from experiments.mid360_formal_batch1.authorization.formal_registration_authorization import (  # noqa: E402
    issue_formal_registration_authorization,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--lock", type=Path, required=True, help="Exec-R3 lock directory")
    parser.add_argument("--lock-fingerprint", required=True)
    parser.add_argument("--lock-release-commit", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--confirm-explicit-user-authorization", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_explicit_user_authorization:
        parser.error("--confirm-explicit-user-authorization is required")
    report = issue_formal_registration_authorization(
        args.repository_root,
        lock_dir=args.lock,
        expected_lock_fingerprint=args.lock_fingerprint,
        lock_release_commit=args.lock_release_commit,
        runtime_root=args.runtime_root,
        workers=args.workers,
        explicit_user_confirmation=True,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
