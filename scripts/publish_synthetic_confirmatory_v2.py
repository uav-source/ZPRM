#!/usr/bin/env python3
"""Atomically publish a complete Synthetic Confirmatory v2 artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _assert_preimport_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError(
            "Synthetic Confirmatory v2 scripts require PYTHONNOUSERSITE=1"
        )
    if (
        os.environ.get("MAMBA_ROOT_PREFIX")
        != "/home/lj/.local/share/degen-lio-micromamba"
    ):
        raise PermissionError(
            "Synthetic Confirmatory v2 scripts require the frozen "
            "MAMBA_ROOT_PREFIX"
        )
    source = Path("/home/lj/Degen-LIO").resolve()
    entries = list(sys.path) + [
        item
        for item in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if item
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("Python search path resolves to the source repository")


_assert_preimport_isolation()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.synthetic_confirmatory_v2_publisher import (
    publish_synthetic_confirmatory_v2,
)


def _strict_json(path: Path) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            value[key] = item
        return value

    result = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {path}: {token}")
        ),
    )
    if type(result) is not dict:
        raise ValueError(f"JSON root must be an object: {path}")
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--primary", type=Path, required=True)
parser.add_argument("--independent", type=Path, required=True)
parser.add_argument("--artifact-dir", type=Path, required=True)
args = parser.parse_args()

result = publish_synthetic_confirmatory_v2(
    manifest_path=args.manifest,
    run_dir=args.run_dir,
    primary=_strict_json(args.primary),
    independent=_strict_json(args.independent),
    artifact_dir=args.artifact_dir,
)
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
