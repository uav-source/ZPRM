#!/usr/bin/env python3
"""Run one explicitly bound seed-free lifecycle qualification context."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )
    if type(value) is not dict:
        raise ValueError("lifecycle manifest must be a JSON object")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=("fresh", "resume"))
    parser.add_argument("--invocation-id")
    parser.add_argument("--skip-postrun", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest_path = (REPOSITORY / args.manifest).resolve()
    if REPOSITORY not in manifest_path.parents:
        raise PermissionError("manifest escaped the standalone repository")
    manifest = _strict_object(manifest_path)
    context_name = manifest.get("qualification_context_name")
    identity = manifest.get("git_identity_policy")
    if (
        context_name not in {"context_a", "context_b"}
        or type(identity) is not dict
    ):
        raise ValueError("qualification manifest identity is invalid")
    delays = manifest.get("durable_commit_delay_seconds")
    if type(delays) is not dict:
        raise ValueError("qualification delay binding is invalid")
    from phase_a_harness.formal_lifecycle import (
        execute_formal_lifecycle,
        execute_postrun_pipeline,
    )
    from phase_a_harness.runtime_lifecycle_fixture import (
        build_qualification_spec,
    )

    spec = build_qualification_spec(
        str(context_name),
        repository_root=REPOSITORY,
        expected_commit=_git("rev-parse", "HEAD"),
        expected_branch=str(identity["expected_branch"]),
        expected_tag=str(identity["expected_tag"]),
        run_id=args.run_id,
        workers=args.workers,
        runtime_root=Path(args.runtime_root),
        manifest_path=manifest_path,
        snapshot_commit_delay_seconds=float(delays["snapshot"]),
        trial_commit_delay_seconds=float(delays["trial"]),
    )
    if spec.run_id != args.run_id:
        raise ValueError("run ID differs from the immutable spec")
    completed = execute_formal_lifecycle(
        spec,
        requested_mode=args.mode,
        invocation_id=args.invocation_id or args.mode,
    )
    output: dict[str, Any] = {"lifecycle": completed.report()}
    if not args.skip_postrun:
        output["postrun"] = execute_postrun_pipeline(spec, completed).report()
    print(
        json.dumps(
            output,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
