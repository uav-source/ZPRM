#!/usr/bin/env python3
"""Issue the FMB1 zero-perturbation v1.1-R1 lock without authorization."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for location in (REPOSITORY / "src", REPOSITORY):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_lock import (  # noqa: E402
    build_lock_payload,
    write_lock_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--execution-code-commit")
    parser.add_argument("--binding-map", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-issue", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_issue:
        raise RuntimeError("--confirm-issue is required; no lock was written")
    repository = args.repository_root.expanduser().resolve(strict=True)
    commit = args.execution_code_commit
    if commit is None:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    bindings = None
    if args.binding_map:
        bindings = json.loads(args.binding_map.read_text(encoding="utf-8"))
        if not isinstance(bindings, dict):
            raise TypeError("binding map must be a JSON object")
    output = args.output_dir or repository / (
        "results/mid360_formal_batch1/zero_perturbation_v1_1_lock"
    )
    lock, inventory = build_lock_payload(
        repository,
        execution_code_commit=commit,
        binding_paths=bindings,
        remeasure_environment_versions=True,
    )
    report = write_lock_bundle(repository, output, lock, inventory)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
