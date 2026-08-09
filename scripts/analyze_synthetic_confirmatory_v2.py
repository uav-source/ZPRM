#!/usr/bin/env python3
"""Analyze an already-complete Synthetic Confirmatory v2 raw run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")


def _assert_preimport_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("Confirmatory v2 analysis requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError(
            "Confirmatory v2 analysis requires the frozen MAMBA_ROOT_PREFIX"
        )
    source = SOURCE_REPOSITORY.resolve()
    entries = list(sys.path) + [
        item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("Python search path resolves to the source repository")


def main(argv: Sequence[str] | None = None) -> int:
    _assert_preimport_isolation()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from phase_a_harness.synthetic_confirmatory_v2_analysis import analyze_v2

    parser = argparse.ArgumentParser(description="Analyze Synthetic Confirmatory v2")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = analyze_v2(manifest_path=args.manifest, run_dir=args.run_dir)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        if args.output.exists():
            raise FileExistsError("refusing to replace v2 primary analysis")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
