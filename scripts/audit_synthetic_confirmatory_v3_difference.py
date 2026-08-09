#!/usr/bin/env python3
"""Require the frozen primary/independent v3 difference to be exactly zero."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("Synthetic Confirmatory v3 requires PYTHONNOUSERSITE=1")
    repository = Path(__file__).resolve().parents[1]
    source = Path("/home/lj/Degen-LIO").resolve()
    for entry in [
        *sys.path,
        *(item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item),
    ]:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents or candidate in source.parents:
            raise PermissionError("Python search path reaches the source repository")
    sys.path.insert(0, str(repository / "src"))
    from phase_a_harness.synthetic_confirmatory_v3_prerun import (
        DEFAULT_MANIFEST_RELATIVE,
        FORMAL_RUNTIME_ROOT,
        load_v3_contract_compat,
        require_path_inside_formal_root,
        strict_json_object,
        validate_v3_postrun_entry,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=repository / DEFAULT_MANIFEST_RELATIVE
    )
    parser.add_argument("--runtime-root", type=Path, default=FORMAL_RUNTIME_ROOT)
    args = parser.parse_args(argv)
    _path, manifest, _loader = load_v3_contract_compat(
        args.manifest, repository=repository
    )
    validate_v3_postrun_entry(
        repository=repository,
        manifest_path=args.manifest,
        runtime_root=args.runtime_root,
        checkpoint="V3_FORMAL_DIFFERENCE_AUDIT_GIT_GATE",
    )
    path = require_path_inside_formal_root(
        Path(manifest["verification_path"]) / "primary_independent_difference.json",
        allow_existing=True,
    )
    report = strict_json_object(path)
    passed = bool(
        report.get("leaf_difference_count") == 0
        and report.get("section_difference_count") == 0
        and report.get("exact_match_pass") is True
    )
    output = {
        "V3_PRIMARY_INDEPENDENT_DIFFERENCE_AUDIT_PASS": passed,
        "difference_path": str(path),
        "leaf_difference_count": report.get("leaf_difference_count"),
        "maximum_absolute_numeric_difference": report.get(
            "maximum_absolute_numeric_difference"
        ),
        "schema_version": "synthetic_confirmatory_v3_difference_audit_v1",
        "section_difference_count": report.get("section_difference_count"),
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
