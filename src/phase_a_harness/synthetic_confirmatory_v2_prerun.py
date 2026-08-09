"""Fail-closed decision and atomic publication for the v2 pre-run freeze.

The artifact published here contains qualification evidence only.  This
module has no snapshot builder, RNG, or registration-backend import.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .contracts import file_sha256, write_json


EVIDENCE_FILES = (
    "v1_failure_binding.json",
    "old_seed_retirement.json",
    "root_cause_binding.json",
    "v2_execution_chain_design.json",
    "v2_metadata_schema.json",
    "phase_a_closure_semantics_diff.json",
    "ideal_parent_lineage_qualification.json",
    "ideal_backend_control.json",
    "independent_negative_control.json",
    "nonideal_scientific_payload_regression.json",
    "v1_to_v2_scientific_diff.json",
    "frozen_model_binding.json",
    "v2_seed_schedule.json",
    "v2_seed_provenance_audit.json",
    "v2_plan_audit.json",
    "v2_dry_run_report.json",
    "fixture_regression.json",
    "primary_independent_difference.json",
    "test_report.json",
    "implementation_manifest.json",
)
GENERATED_FILES = (
    "artifact_verification.json",
    "final_decision.json",
    "run_manifest.json",
    "pre_run_report.md",
    "MANIFEST.csv",
    "SHA256SUMS",
)
PRERUN_ROOT_FILES = EVIDENCE_FILES + GENERATED_FILES
FIXTURE_PUBLICATION_DIRECTORY = "fixture_publication"
SELF_REFERENTIAL_FILES = frozenset(
    {"artifact_verification.json", "MANIFEST.csv", "SHA256SUMS"}
)

REQUIRED_TRUE_GATES = (
    "IMPLEMENTATION_ONLY_REPAIR",
    "V1_FAILURE_RECORD_PRESERVED",
    "IDEAL_PARENT_INDEX_LINEAGE_IMPLEMENTED",
    "INDEPENDENT_VERIFIER_LINEAGE_RECOMPUTATION_PASS",
    "IDEAL_OPEN3D_CONTROL_PASS",
    "IDEAL_PCL_CONTROL_PASS",
    "FIXTURE_EXECUTION_CHAIN_PASS",
    "V1_TO_V2_SCIENTIFIC_DIFF_PASS",
    "FROZEN_MODEL_SHA_MATCH",
    "V2_PLAN_PASS",
    "V2_DRY_RUN_PASS",
    "V2_ARTIFACT_PUBLICATION_PASS",
    "V2_ARTIFACT_VERIFICATION_PASS",
)
REQUIRED_ZERO_COUNTERS = (
    "PHASE_A_CLOSURE_SEMANTICS_DIFF_COUNT",
    "IDEAL_LINEAGE_VIOLATION_COUNT",
    "NONIDEAL_SCIENTIFIC_PAYLOAD_CHANGE_COUNT",
    "PRIMARY_VERIFIER_DIFFERENCE_COUNT",
    "NEW_V2_SEED_PROVENANCE_COLLISION_COUNT",
    "NEW_V2_RNG_INSTANTIATION_COUNT",
    "NEW_V2_SNAPSHOT_CONSTRUCTION_COUNT",
    "NEW_V2_BACKEND_EXECUTION_COUNT",
    "NEW_V2_TRIAL_RESULT_COUNT",
)


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("v2 pre-run evidence must be a JSON object")
    return json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )


def build_v2_prerun_decision(
    *,
    gates: Mapping[str, Any],
    counters: Mapping[str, Any],
    authorize: bool,
) -> dict[str, Any]:
    """Evaluate the exact v2 pre-run gate without optimistic defaults."""

    if type(authorize) is not bool:
        raise TypeError("authorize must be bool")
    normalized_gates = {name: gates.get(name) for name in REQUIRED_TRUE_GATES}
    normalized_counters = {
        name: counters.get(name) for name in REQUIRED_ZERO_COUNTERS
    }
    inventory_pass = bool(
        gates.get("ROOT_CAUSE") == "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON"
        and gates.get("OLD_V1_SEED_SET_REUSE_AUTHORIZED") is False
        and gates.get("NEW_V2_NAMESPACE_COLLISION") is False
        and gates.get("IDEAL_DEVELOPMENT_SNAPSHOT_COUNT") == 21
        and gates.get("INDEPENDENT_NEGATIVE_CONTROL_COUNT") == 21
        and gates.get("INDEPENDENT_FALSE_LINEAGE_COUNT") == 21
    )
    qualification = bool(
        inventory_pass
        and all(value is True for value in normalized_gates.values())
        and all(type(value) is int and value == 0 for value in normalized_counters.values())
    )
    if authorize and not qualification:
        raise PermissionError("cannot authorize an unqualified v2 formal run")
    return {
        "ROOT_CAUSE": gates.get("ROOT_CAUSE"),
        "OLD_V1_SEED_SET_REUSE_AUTHORIZED": False,
        "NEW_V2_NAMESPACE_COLLISION": gates.get("NEW_V2_NAMESPACE_COLLISION"),
        "IDEAL_DEVELOPMENT_SNAPSHOT_COUNT": gates.get(
            "IDEAL_DEVELOPMENT_SNAPSHOT_COUNT"
        ),
        "INDEPENDENT_NEGATIVE_CONTROL_COUNT": gates.get(
            "INDEPENDENT_NEGATIVE_CONTROL_COUNT"
        ),
        "INDEPENDENT_FALSE_LINEAGE_COUNT": gates.get(
            "INDEPENDENT_FALSE_LINEAGE_COUNT"
        ),
        **normalized_gates,
        **normalized_counters,
        "CONFIRMATORY_V2_RUN_AUTHORIZED": bool(authorize and qualification),
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_V2_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_V2_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V2_PASS": "NOT_EVALUATED",
        "SYNTHETIC_CONFIRMATORY_V2_PRE_RUN_QUALIFICATION_PASS": qualification,
        "required_true_gates": list(REQUIRED_TRUE_GATES),
        "required_zero_counters": list(REQUIRED_ZERO_COUNTERS),
        "schema_version": "synthetic_confirmatory_v2_prerun_decision_v1",
    }


def _report(decision: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]) -> str:
    plan = evidence["v2_plan_audit.json"]
    dry = evidence["v2_dry_run_report.json"]
    controls = evidence["ideal_backend_control.json"]
    backend_gates = controls.get("backend_gates", {})
    open3d_count = (
        backend_gates.get("open3d_point_to_plane", {}).get("trial_count")
        if type(backend_gates) is dict
        else None
    )
    pcl_count = (
        backend_gates.get("pcl_point_to_plane", {}).get("trial_count")
        if type(backend_gates) is dict
        else None
    )
    regression = evidence["nonideal_scientific_payload_regression.json"]
    return "\n".join(
        [
            "# Synthetic Confirmatory v2 — Formal Pre-Run Freeze",
            "",
            "This is a qualification-only artifact. No v2 formal snapshot, backend trial, or scientific result was produced.",
            "",
            f"- `SYNTHETIC_CONFIRMATORY_V2_PRE_RUN_QUALIFICATION_PASS = {str(decision['SYNTHETIC_CONFIRMATORY_V2_PRE_RUN_QUALIFICATION_PASS']).lower()}`",
            f"- `CONFIRMATORY_V2_RUN_AUTHORIZED = {str(decision['CONFIRMATORY_V2_RUN_AUTHORIZED']).lower()}`",
            "- `SYNTHETIC_CONFIRMATORY_V2_EXECUTED = false`",
            "- `SYNTHETIC_CONFIRMATORY_V2_PASS = NOT_EVALUATED`",
            "",
            "## Repair qualification",
            "",
            f"- Root cause: `{decision['ROOT_CAUSE']}`",
            f"- IDEAL Development snapshots: `{decision['IDEAL_DEVELOPMENT_SNAPSHOT_COUNT']}`",
            f"- IDEAL Open3D/PCL controls: `{open3d_count} / {pcl_count}`",
            f"- Non-IDEAL Development regression coverage: `{regression.get('snapshot_count')}`",
            "",
            "## Frozen formal design",
            "",
            f"- Planned snapshots/trials: `{plan.get('planned_snapshot_count')} / {plan.get('planned_trial_count')}`",
            f"- Dry-run snapshots/trials: `{dry.get('planned_snapshot_count')} / {dry.get('planned_trial_count')}`",
            "- New-v2 RNG/snapshot/backend/trial/STARTED counts: `0 / 0 / 0 / 0 / 0`",
            "",
            "The exact authorized manifest may be used only in a later turn after the clean tagged pre-run freeze.",
            "",
        ]
    )


def _checksum_inventory(root: Path) -> list[dict[str, Any]]:
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.relative_to(root).as_posix() not in SELF_REFERENTIAL_FILES
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]


def publish_v2_prerun_artifact_atomic(
    output_dir: str | Path,
    *,
    evidence: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    fixture_publication_dir: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Publish and independently verify the exact v2 pre-run inventory."""

    if set(evidence) != set(EVIDENCE_FILES):
        raise ValueError("v2 pre-run evidence inventory differs from the contract")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace v2 pre-run artifact: {destination}")
    fixture = Path(fixture_publication_dir).resolve()
    if not fixture.is_dir():
        raise FileNotFoundError("v2 seed-free fixture publication is missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    staging = temporary / "artifact"
    try:
        staging.mkdir()
        for name in EVIDENCE_FILES:
            write_json(staging / name, _json_copy(evidence[name]))
        write_json(staging / "final_decision.json", _json_copy(decision))
        write_json(staging / "run_manifest.json", _json_copy(run_manifest))
        (staging / "pre_run_report.md").write_text(
            _report(decision, evidence), encoding="utf-8"
        )
        shutil.copytree(
            fixture,
            staging / FIXTURE_PUBLICATION_DIRECTORY,
            copy_function=shutil.copy2,
        )
        inventory = _checksum_inventory(staging)
        with (staging / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "size_bytes"))
            writer.writeheader()
            writer.writerows(inventory)
        (staging / "SHA256SUMS").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in inventory),
            encoding="utf-8",
        )
        from .synthetic_confirmatory_v2_artifact_verifier import (
            verify_v2_prerun_artifact,
        )

        verification = verify_v2_prerun_artifact(
            staging, manifest_path=manifest_path, write_report=True
        )
        read_only = verify_v2_prerun_artifact(
            staging, manifest_path=manifest_path, write_report=False
        )
        if verification != read_only or verification.get(
            "V2_ARTIFACT_VERIFICATION_PASS"
        ) is not True:
            raise ValueError("v2 pre-run artifact did not independently verify")
        os.replace(staging, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verification


__all__ = [
    "EVIDENCE_FILES",
    "FIXTURE_PUBLICATION_DIRECTORY",
    "GENERATED_FILES",
    "PRERUN_ROOT_FILES",
    "REQUIRED_TRUE_GATES",
    "REQUIRED_ZERO_COUNTERS",
    "SELF_REFERENTIAL_FILES",
    "build_v2_prerun_decision",
    "publish_v2_prerun_artifact_atomic",
]
