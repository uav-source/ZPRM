#!/usr/bin/env python3
"""Read-only static and four-state preflight for a frozen v3 invocation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")


def _assert_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("Synthetic Confirmatory v3 requires PYTHONNOUSERSITE=1")
    source = SOURCE_REPOSITORY.resolve()
    entries = list(sys.path) + [
        value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents or candidate in source.parents:
            raise PermissionError("Python search path overlaps the source repository")


def main(argv: Sequence[str] | None = None) -> int:
    _assert_isolation()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))
    from phase_a_harness.synthetic_confirmatory_v3_prerun import (
        DEFAULT_MANIFEST_RELATIVE,
        FORMAL_RUNTIME_ROOT,
        validate_v3_frozen_contract,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=repository / DEFAULT_MANIFEST_RELATIVE
    )
    parser.add_argument("--runtime-root", type=Path, default=FORMAL_RUNTIME_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    args = parser.parse_args(argv)

    validated = validate_v3_frozen_contract(
        repository=repository,
        manifest_path=args.manifest,
        run_id=args.run_id,
        runtime_root=args.runtime_root,
        workers=args.workers,
        require_authorized=True,
        require_release_tag=True,
        require_runtime_absent=False,
        git_checkpoint="V3_FORMAL_PREFLIGHT_GIT_GATE",
    )
    from phase_a_harness.formal_runtime_state_machine import (
        FormalRuntimeState,
        build_formal_runner_command,
        inspect_formal_runtime,
    )

    frozen_fresh_command = build_formal_runner_command(
        repository=repository,
        manifest_path=args.manifest,
        run_id=args.run_id,
        runtime_root=args.runtime_root,
        workers=args.workers,
        mode="fresh",
        entry_script="scripts/run_synthetic_confirmatory_v3.py",
    )
    inspection = inspect_formal_runtime(
        args.runtime_root,
        expected_command=(
            None if not args.runtime_root.exists() else frozen_fresh_command
        ),
    )
    allowed = (
        {FormalRuntimeState.ABSENT, FormalRuntimeState.BOOTSTRAP_ONLY}
        if args.mode == "fresh"
        else {FormalRuntimeState.RESUMABLE}
    )
    passed = bool(
        validated.get("V3_FROZEN_CONTRACT_PASS") is True
        and validated.get("FORMAL_GIT_GATE_PASS") is True
        and inspection["state"] in allowed
    )
    if not passed:
        raise PermissionError(
            f"formal {args.mode} preflight rejected {inspection['state'].value}: "
            f"{inspection['reasons']}"
        )
    report = {
        "FORMAL_PREFLIGHT_PASS": True,
        "allowed_states": sorted(state.value for state in allowed),
        "formal_runtime_state": inspection["state"].value,
        "mode": args.mode,
        "run_id": args.run_id,
        "schema_version": "synthetic_confirmatory_v3_bootstrap_repair_preflight_v1",
        "workers": args.workers,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
