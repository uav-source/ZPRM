#!/usr/bin/env python3
"""Qualify the v3 adapter with seed-free real writes before seed derivation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"


def _assert_environment() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("v3 adapter qualification requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError("v3 adapter qualification requires frozen MAMBA_ROOT_PREFIX")
    source = Path("/home/lj/Degen-LIO").resolve()
    for entry in [
        *sys.path,
        *(item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item),
    ]:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("source repository appears on the Python search path")


def main(argv: Sequence[str] | None = None) -> int:
    _assert_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-tag", required=True)
    args = parser.parse_args(argv)

    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))
    from phase_a_harness.asset_verifier import source_runtime_import_paths
    from phase_a_harness.runner import SourceAccessMonitor
    from phase_a_harness.runtime_git_gate import verify_runtime_git_gate
    from phase_a_harness.runtime_lifecycle_io import atomic_create_canonical_json
    from phase_a_harness.synthetic_confirmatory_v3_adapter import (
        qualification_layout,
        run_seed_free_adapter_fixture,
    )

    monitor = SourceAccessMonitor()
    monitor.install()
    gate_reports: list[dict[str, object]] = []

    def gate(checkpoint: str) -> dict[str, object]:
        value = verify_runtime_git_gate(
            repository,
            args.expected_commit,
            args.expected_branch,
            args.expected_tag,
            checkpoint=checkpoint,
        )
        gate_reports.append(value)
        return value

    report = run_seed_free_adapter_fixture(
        repository=repository,
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_tag=args.expected_tag,
        git_gate=gate,
    )
    gate("V3_FIXTURE_FINAL_GIT_GATE")
    imports = source_runtime_import_paths()
    report["git_gate_reports"] = gate_reports
    report["V3_GIT_GATE_FIXTURE_PASS"] = bool(
        gate_reports
        and all(row.get("RUNTIME_GIT_GATE_PASS") is True for row in gate_reports)
    )
    report["source_repository_runtime_file_read_count"] = monitor.count
    report["source_repository_runtime_import_count"] = len(imports)
    report["source_repository_runtime_import_paths"] = imports
    report["SOURCE_REPOSITORY_RUNTIME_ISOLATION_PASS"] = bool(
        monitor.count == 0 and not imports
    )
    report["V3_EXECUTION_ADAPTER_FIXTURE_PASS"] = bool(
        report["V3_EXECUTION_ADAPTER_FIXTURE_PASS"]
        and report["V3_GIT_GATE_FIXTURE_PASS"]
        and report["SOURCE_REPOSITORY_RUNTIME_ISOLATION_PASS"]
    )
    layout = qualification_layout(repository=repository, resume=True)
    output = layout.temporary_inventory / "v3_adapter_fixture_qualification.json"
    atomic_create_canonical_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["V3_EXECUTION_ADAPTER_FIXTURE_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
