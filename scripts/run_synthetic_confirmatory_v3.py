#!/usr/bin/env python3
"""Zero-instantiation dry-run and fail-closed formal entry for v3."""

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
        value
        for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if value
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents or candidate in source.parents:
            raise PermissionError("Python search path reaches the source repository")


def build_parser(
    *, default_manifest: Path, default_runtime_root: Path
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or enter the frozen Synthetic Confirmatory v3 run"
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--run-id", required=True)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--runtime-root", dest="runtime_root", type=Path, default=default_runtime_root
    )
    destination.add_argument("--output-dir", dest="runtime_root", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--mode", choices=("fresh", "resume"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _assert_preimport_isolation()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))
    from phase_a_harness.synthetic_confirmatory_v3_prerun import (
        DEFAULT_MANIFEST_RELATIVE,
        FORMAL_RUNTIME_ROOT,
        dry_run_synthetic_confirmatory_v3,
    )

    parser = build_parser(
        default_manifest=repository / DEFAULT_MANIFEST_RELATIVE,
        default_runtime_root=FORMAL_RUNTIME_ROOT,
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.resume or args.mode is not None:
            parser.error("--dry-run cannot be combined with a formal mode")
        report = dry_run_synthetic_confirmatory_v3(
            manifest_path=args.manifest,
            run_id=args.run_id,
            runtime_root=args.runtime_root,
            workers=args.workers,
            repository=repository,
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0

    if args.mode is not None and args.resume:
        parser.error("--mode and legacy --resume are mutually exclusive")
    selected_mode = args.mode or ("resume" if args.resume else None)
    if selected_mode is None:
        parser.error("formal v3 execution requires --mode fresh|resume")
    from phase_a_harness.formal_runtime_state_machine import (
        build_formal_runner_command,
    )

    expected_command = build_formal_runner_command(
        repository=repository,
        manifest_path=args.manifest,
        run_id=args.run_id,
        runtime_root=args.runtime_root,
        workers=args.workers,
        mode=selected_mode,
        entry_script="scripts/run_synthetic_confirmatory_v3.py",
    )
    # The execution-capable runner is intentionally imported only on the
    # non-dry path.  Its own first action repeats the authorization artifact,
    # release-tag, clean-Git, plan, and exact external-path gates.
    from phase_a_harness.synthetic_confirmatory_v3_runner import (
        execute_synthetic_confirmatory_v3,
    )

    report = execute_synthetic_confirmatory_v3(
        repository=repository,
        manifest_path=args.manifest,
        run_id=args.run_id,
        runtime_root=args.runtime_root,
        workers=args.workers,
        resume=(selected_mode == "resume"),
        mode=selected_mode,
        expected_formal_command=expected_command,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
