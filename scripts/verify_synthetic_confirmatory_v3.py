#!/usr/bin/env python3
"""Thin independent-verification entry for frozen v3 formal evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


def _assert_preimport_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("Synthetic Confirmatory v3 requires PYTHONNOUSERSITE=1")
    source = Path("/home/lj/Degen-LIO").resolve()
    entries = list(sys.path) + [
        value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("Python search path reaches the source repository")


def main(argv: Sequence[str] | None = None) -> int:
    _assert_preimport_isolation()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))
    from phase_a_harness.synthetic_confirmatory_v3_prerun import (
        DEFAULT_MANIFEST_RELATIVE,
        FORMAL_RUNTIME_ROOT,
        load_v3_contract_compat,
        require_path_inside_formal_root,
        strict_json_object,
    )

    parser = argparse.ArgumentParser(
        description="Independently verify Synthetic Confirmatory v3 evidence"
    )
    parser.add_argument(
        "--manifest", type=Path, default=repository / DEFAULT_MANIFEST_RELATIVE
    )
    parser.add_argument("--runtime-root", type=Path, default=FORMAL_RUNTIME_ROOT)
    args = parser.parse_args(argv)

    _path, manifest, _loader = load_v3_contract_compat(
        args.manifest, repository=repository
    )
    primary_path = require_path_inside_formal_root(
        Path(manifest["analysis_path"]) / "primary_analysis.json",
        allow_existing=True,
    )
    if not primary_path.is_file():
        raise FileNotFoundError("v3 primary analysis is absent")
    output = require_path_inside_formal_root(
        Path(manifest["verification_path"]) / "independent_verification.json",
        allow_existing=False,
    )
    difference_output = require_path_inside_formal_root(
        Path(manifest["verification_path"]) / "primary_independent_difference.json",
        allow_existing=False,
    )

    from phase_a_harness.runtime_lifecycle_io import atomic_create_canonical_json
    from phase_a_harness.synthetic_confirmatory_independent_verifier import (
        compare_primary_and_independent,
    )
    from phase_a_harness.synthetic_confirmatory_v3_runner import (
        independently_verify_completed_v3,
    )

    independent = independently_verify_completed_v3(
        repository=repository,
        manifest_path=args.manifest,
        runtime_root=args.runtime_root,
    )
    comparison = compare_primary_and_independent(
        strict_json_object(primary_path), independent
    )
    independent["analysis_verifier_comparison"] = comparison
    independent["ANALYSIS_VERIFIER_DIFFERENCE_COUNT"] = comparison[
        "leaf_difference_count"
    ]
    atomic_create_canonical_json(output, independent)
    atomic_create_canonical_json(difference_output, comparison)
    print(
        json.dumps(
            {
                "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": comparison[
                    "leaf_difference_count"
                ],
                "difference_output": str(difference_output),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
