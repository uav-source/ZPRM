#!/usr/bin/env python3
"""Fail closed unless the frozen v3 raw matrix is exactly complete."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


def _assert_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("Synthetic Confirmatory v3 requires PYTHONNOUSERSITE=1")
    source = Path("/home/lj/Degen-LIO").resolve()
    for entry in [
        *sys.path,
        *(item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item),
    ]:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents or candidate in source.parents:
            raise PermissionError("Python search path reaches the source repository")


def main(argv: Sequence[str] | None = None) -> int:
    _assert_isolation()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))
    from phase_a_harness.synthetic_confirmatory_v3_prerun import (
        DEFAULT_MANIFEST_RELATIVE,
        FORMAL_RUNTIME_ROOT,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=repository / DEFAULT_MANIFEST_RELATIVE
    )
    parser.add_argument("--runtime-root", type=Path, default=FORMAL_RUNTIME_ROOT)
    args = parser.parse_args(argv)

    from phase_a_harness.synthetic_confirmatory_v3_runner import (
        load_completed_v3_trials,
    )

    trials, raw, stack = load_completed_v3_trials(
        repository=repository,
        manifest_path=args.manifest,
        runtime_root=args.runtime_root,
        checkpoint="V3_FORMAL_COMPLETENESS_GIT_GATE",
    )
    backends = Counter(row["backend"] for row in trials)
    snapshots = {row["planned_snapshot_id"] for row in trials}
    report = {
        "V3_FORMAL_COMPLETENESS_PASS": bool(
            len(snapshots) == 595
            and len(trials) == 1190
            and backends
            == {"open3d_point_to_plane": 595, "pcl_point_to_plane": 595}
            and len(raw.get("results", {})) == 1190
        ),
        "backend_trial_counts": dict(sorted(backends.items())),
        "completed_snapshot_count": len(snapshots),
        "completed_trial_count": len(trials),
        "manifest_sha256": stack["manifest_sha256"],
        "raw_manifest_result_count": len(raw.get("results", {})),
        "schema_version": "synthetic_confirmatory_v3_completeness_report_v1",
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["V3_FORMAL_COMPLETENESS_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
