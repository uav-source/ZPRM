#!/usr/bin/env python3
"""Run the read-only Scientific Survival audit and gated protocol design."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


if os.environ.get("PYTHONNOUSERSITE") != "1":
    raise PermissionError("Scientific Survival audit requires PYTHONNOUSERSITE=1")
source = Path("/home/lj/Degen-LIO").resolve()
for entry in list(sys.path) + [
    item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item
]:
    candidate = (Path.cwd() if not entry else Path(entry)).resolve()
    if candidate == source or source in candidate.parents:
        raise PermissionError("Python search path resolves to the source repository")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.scientific_survival_audit import run_scientific_survival_audit


result = run_scientific_survival_audit(ROOT)
print(
    json.dumps(
        {
            **result["decision"],
            "artifact_file_count": result["artifact_verification"][
                "required_file_count"
            ],
            "artifact_sha256_mismatch_count": result["artifact_verification"][
                "sha256_mismatch_count"
            ],
            "confirmatory_planned_snapshot_count": result[
                "confirmatory_protocol"
            ]["planned_snapshot_count"],
            "confirmatory_planned_trial_count": result["confirmatory_protocol"][
                "planned_trial_count"
            ],
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
)
