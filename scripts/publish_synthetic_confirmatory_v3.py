#!/usr/bin/env python3
"""Thin atomic-publisher entry for complete frozen v3 formal evidence."""

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
        description="Publish the complete frozen Synthetic Confirmatory v3 artifact"
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
    independent_path = require_path_inside_formal_root(
        Path(manifest["verification_path"]) / "independent_verification.json",
        allow_existing=True,
    )
    if not primary_path.is_file() or not independent_path.is_file():
        raise FileNotFoundError("v3 primary or independent analysis is absent")
    artifact_dir = require_path_inside_formal_root(
        Path(manifest["artifact_staging_path"]), allow_existing=False
    )

    from phase_a_harness.synthetic_confirmatory_v3_runner import (
        publish_completed_v3,
    )

    result = publish_completed_v3(
        repository=repository,
        manifest_path=args.manifest,
        runtime_root=args.runtime_root,
        primary=strict_json_object(primary_path),
        independent=strict_json_object(independent_path),
        artifact_dir=artifact_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
