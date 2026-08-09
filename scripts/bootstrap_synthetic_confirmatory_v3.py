#!/usr/bin/env python3
"""Create or authenticate the seed-free Synthetic Confirmatory v3 bootstrap."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")


def _assert_preimport_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("Synthetic Confirmatory v3 requires PYTHONNOUSERSITE=1")
    source = SOURCE_REPOSITORY.resolve()
    entries = list(sys.path) + [
        value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents or candidate in source.parents:
            raise PermissionError("Python search path reaches the source repository")


def main(argv: Sequence[str] | None = None) -> int:
    _assert_preimport_isolation()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))
    from phase_a_harness.synthetic_confirmatory_v3_prerun import (
        DEFAULT_MANIFEST_RELATIVE,
        FORMAL_RUNTIME_ROOT,
        guard_synthetic_confirmatory_v3_execution,
    )

    parser = argparse.ArgumentParser(
        description="Bootstrap the external Synthetic Confirmatory v3 runtime"
    )
    parser.add_argument(
        "--manifest", type=Path, default=repository / DEFAULT_MANIFEST_RELATIVE
    )
    parser.add_argument("--runtime-root", type=Path, default=FORMAL_RUNTIME_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    args = parser.parse_args(argv)

    # Authenticate static science, final pre-run authority, release tag, Git,
    # and the exact external path before creating even the command-only root.
    guarded = guard_synthetic_confirmatory_v3_execution(
        manifest_path=args.manifest,
        run_id=args.run_id,
        runtime_root=args.runtime_root,
        workers=args.workers,
        repository=repository,
        resume=True,
    )
    from phase_a_harness.formal_runtime_state_machine import (
        bootstrap_formal_runtime,
        build_formal_runner_command,
    )

    # The immutable command log always records the first fresh runner command.
    # A resume invocation authenticates that original identity; it never
    # rewrites the command object to say "resume".
    command = build_formal_runner_command(
        repository=repository,
        manifest_path=args.manifest,
        run_id=args.run_id,
        runtime_root=args.runtime_root,
        workers=args.workers,
        mode="fresh",
        entry_script="scripts/run_synthetic_confirmatory_v3.py",
    )
    invocation_command = build_formal_runner_command(
        repository=repository,
        manifest_path=args.manifest,
        run_id=args.run_id,
        runtime_root=args.runtime_root,
        workers=args.workers,
        mode=args.mode,
        entry_script="scripts/run_synthetic_confirmatory_v3.py",
    )
    report = bootstrap_formal_runtime(
        args.runtime_root,
        command,
        mode=args.mode,
    )
    output = {
        **report,
        "FORMAL_BOOTSTRAP_PASS": True,
        "FORMAL_GIT_GATE_PASS": guarded["FORMAL_GIT_GATE_PASS"],
        "FORMAL_PRERUN_ARTIFACT_BINDING_PASS": guarded[
            "FORMAL_PRERUN_ARTIFACT_BINDING_PASS"
        ],
        "formal_command": command,
        "requested_runner_command": invocation_command,
        "formal_runtime_root": str(args.runtime_root),
        "run_id": args.run_id,
        "schema_version": "synthetic_confirmatory_v3_formal_bootstrap_report_v1",
        "workers": args.workers,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
