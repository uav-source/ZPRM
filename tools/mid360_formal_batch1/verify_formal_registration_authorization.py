#!/usr/bin/env python3
"""Independently verify one immutable FMB1 Exec-R3 authorization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for location in (REPOSITORY / "src", REPOSITORY):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from experiments.mid360_formal_batch1.authorization.formal_registration_authorization_verify import (  # noqa: E402
    verify_formal_registration_authorization,
)


def _write_once(path: Path, payload: dict[str, object]) -> None:
    content = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite different verifier report: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify_formal_registration_authorization(
        args.repository_root,
        lock_dir=args.lock,
        authorization_path=args.authorization,
        runtime_root=args.runtime_root,
        requested_mode=args.mode,
        workers=args.workers,
    )
    _write_once(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
