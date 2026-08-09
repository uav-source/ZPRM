#!/usr/bin/env python3
"""Plan or execute the bounded Synthetic Confirmatory v2 qualification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")


def _assert_runtime_isolation(repository: Path) -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("qualification requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError("qualification requires the frozen MAMBA_ROOT_PREFIX")
    source = SOURCE_REPOSITORY.resolve()
    entries = list(sys.path) + [
        item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("qualification Python path reaches the source repository")
    if not (repository / ".git").is_dir():
        raise ValueError("qualification root is not the standalone harness repository")


def _assert_nonformal_output(repository: Path, candidate: Path) -> None:
    output = candidate.resolve()
    forbidden = (
        repository / "data/synthetic_confirmatory_v2_snapshots",
        repository / "results/synthetic_confirmatory_v2",
        repository / "artifacts/synthetic_confirmatory_v2",
        repository / "artifacts/synthetic_confirmatory_v2_prerun",
    )
    for path in forbidden:
        resolved = path.resolve()
        if output == resolved or resolved in output.parents:
            raise PermissionError("qualification output overlaps a formal v2 path")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or explicitly execute Development-only Synthetic Confirmatory v2 "
            "qualification; formal v2 execution is never authorized here"
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="standalone harness repository",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="metadata/file-binding plan only (default; no RNG or backend)",
    )
    mode.add_argument(
        "--execute-qualification",
        action="store_true",
        help="run 21+21 controls, 42 backends, and 1,050 Development regressions",
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--fixture-output-dir",
        type=Path,
        help="new absent directory; omit to emit the callable 3/6 fixture hook only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="new absent JSON report path; stdout is always emitted",
    )
    return parser


def _write_once(path: Path, value: dict[str, object]) -> None:
    from phase_a_harness.phase_a_trial_result_schema import canonical_json_bytes
    from phase_a_harness.phase_a_trial_result_writer import atomic_write_bytes

    destination = path.resolve()
    atomic_write_bytes(destination, canonical_json_bytes(value), replace=False)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.root.resolve()
    _assert_runtime_isolation(repository)
    if arguments.fixture_output_dir is not None and not arguments.execute_qualification:
        raise PermissionError("fixture execution requires --execute-qualification")
    if arguments.output is not None:
        _assert_nonformal_output(repository, arguments.output)
    sys.path.insert(0, str(repository / "src"))
    from phase_a_harness.synthetic_confirmatory_v2_qualification import (
        execute_bounded_qualification,
        qualification_execution_plan,
    )

    report = (
        execute_bounded_qualification(
            repository,
            workers=arguments.workers,
            fixture_output_dir=arguments.fixture_output_dir,
        )
        if arguments.execute_qualification
        else qualification_execution_plan(repository)
    )
    if arguments.output is not None:
        _write_once(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
