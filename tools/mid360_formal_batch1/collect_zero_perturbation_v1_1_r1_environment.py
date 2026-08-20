#!/usr/bin/env python3
"""Collect the exact FMB1 zero-perturbation v1.1-R1 environment manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for location in (REPOSITORY / "src", REPOSITORY):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_environment import (  # noqa: E402
    collect_environment_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--pcl-executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = collect_environment_manifest(
        args.repository_root.expanduser(), pcl_executable=args.pcl_executable
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != encoded:
        raise RuntimeError(f"refusing to overwrite different environment evidence: {output}")
    if not output.exists():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(output)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("qualification_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
