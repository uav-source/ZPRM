#!/usr/bin/env python3
"""Thin primary-analysis entry for a complete frozen v3 formal run."""

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
    )

    parser = argparse.ArgumentParser(
        description="Analyze a complete frozen Synthetic Confirmatory v3 run"
    )
    parser.add_argument(
        "--manifest", type=Path, default=repository / DEFAULT_MANIFEST_RELATIVE
    )
    parser.add_argument("--runtime-root", type=Path, default=FORMAL_RUNTIME_ROOT)
    args = parser.parse_args(argv)

    _path, manifest, _loader = load_v3_contract_compat(
        args.manifest, repository=repository
    )
    output = require_path_inside_formal_root(
        Path(manifest["analysis_path"]) / "primary_analysis.json",
        allow_existing=False,
    )

    # The v3 I/O adapter reads the external snapshot/result contract, then
    # delegates only the scientific record aggregation to the frozen core.
    from phase_a_harness.runtime_lifecycle_io import atomic_create_canonical_json
    from phase_a_harness.synthetic_confirmatory_v3_runner import (
        analyze_completed_v3,
    )

    report = analyze_completed_v3(
        repository=repository,
        manifest_path=args.manifest,
        runtime_root=args.runtime_root,
    )
    atomic_create_canonical_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "schema_version": report.get("schema_version"),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
