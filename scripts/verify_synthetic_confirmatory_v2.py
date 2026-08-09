#!/usr/bin/env python3
"""Independently authenticate and verify Synthetic Confirmatory v2 evidence."""

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
        raise PermissionError("Confirmatory v2 verification requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError(
            "Confirmatory v2 verification requires the frozen MAMBA_ROOT_PREFIX"
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
    from phase_a_harness.contracts import write_json
    from phase_a_harness.synthetic_confirmatory_v2_independent_verifier import (
        compare_v2_primary_and_independent,
        independently_verify_v2,
    )

    parser = argparse.ArgumentParser(
        description="Independently verify Synthetic Confirmatory v2"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = independently_verify_v2(
        manifest_path=args.manifest, run_dir=args.run_dir
    )
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    comparison = compare_v2_primary_and_independent(primary, report)
    report["analysis_verifier_comparison"] = comparison
    report["ANALYSIS_VERIFIER_DIFFERENCE_COUNT"] = comparison[
        "leaf_difference_count"
    ]
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": comparison[
                    "leaf_difference_count"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
