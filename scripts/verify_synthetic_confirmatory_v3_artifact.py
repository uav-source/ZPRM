#!/usr/bin/env python3
"""Live verification entry for the formal v3 7/3/7 artifact."""

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
        validate_v3_postrun_entry,
    )

    parser = argparse.ArgumentParser(
        description="Verify the formal Synthetic Confirmatory v3 artifact"
    )
    parser.add_argument(
        "--manifest", type=Path, default=repository / DEFAULT_MANIFEST_RELATIVE
    )
    parser.add_argument("--runtime-root", type=Path, default=FORMAL_RUNTIME_ROOT)
    args = parser.parse_args(argv)

    _path, manifest, _loader = load_v3_contract_compat(
        args.manifest, repository=repository
    )
    validated = validate_v3_postrun_entry(
        repository=repository,
        manifest_path=args.manifest,
        runtime_root=args.runtime_root,
        checkpoint="V3_FORMAL_ARTIFACT_VERIFIER_ENTRY_GIT_GATE",
    )
    artifact = require_path_inside_formal_root(
        Path(manifest["artifact_staging_path"]), allow_existing=True
    )
    if artifact != Path(validated["manifest"]["artifact_staging_path"]):
        raise ValueError("v3 formal artifact path differs from the frozen manifest")
    if not artifact.is_dir():
        raise FileNotFoundError("v3 formal artifact is absent")

    from phase_a_harness.synthetic_confirmatory_v3_runner import (
        verify_v3_formal_artifact,
    )

    report = verify_v3_formal_artifact(artifact)
    if report.get("V3_FORMAL_ARTIFACT_VERIFICATION_PASS") is not True:
        raise RuntimeError("v3 formal artifact verification failed")
    output = require_path_inside_formal_root(
        Path(manifest["verification_path"]) / "artifact_verification.json",
        allow_existing=False,
    )
    from phase_a_harness.runtime_lifecycle_io import atomic_create_canonical_json

    atomic_create_canonical_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
