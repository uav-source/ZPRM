"""Pre-run qualification decision and compact artifact publication."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import file_sha256, write_json


REQUIRED_GATE_NAMES = (
    "SCIENTIFIC_SURVIVAL_BINDING_PASS",
    "CONFIRMATORY_PROTOCOL_BINDING_PASS",
    "CONFIRMATORY_GATE_CONTRACT_PASS",
    "CONFIRMATORY_SEED_PROVENANCE_PASS",
    "CONFIRMATORY_PLAN_COUNT_PASS",
    "CONFIRMATORY_PLAN_UNIQUENESS_PASS",
    "CONFIRMATORY_PLAN_PAIRING_PASS",
    "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS",
    "FROZEN_MODEL_FILE_SHA_PASS",
    "FROZEN_MODEL_SCHEMA_PASS",
    "FROZEN_MODEL_FEATURE_ORDER_PASS",
    "FROZEN_MODEL_NO_REFIT_CONTRACT_PASS",
    "FROZEN_MODEL_PREDICTION_CROSSCHECK_PASS",
    "CONFIRMATORY_EXECUTION_CHAIN_FIXTURE_PASS",
    "CONFIRMATORY_DRY_RUN_PASS",
    "CONFIRMATORY_ARTIFACT_VERIFICATION_PASS",
)
ZERO_EXECUTION_COUNTER_NAMES = (
    "CONFIRMATORY_RNG_INSTANTIATION_COUNT",
    "CONFIRMATORY_SNAPSHOT_GENERATION_COUNT",
    "CONFIRMATORY_BACKEND_EXECUTION_COUNT",
    "CONFIRMATORY_TRIAL_RESULT_COUNT",
    "NATIVE_EXECUTION_COUNT",
)
ARTIFACT_FILES = (
    "protocol_binding.json",
    "gate_contract_audit.json",
    "seed_provenance_audit.json",
    "plan_audit.json",
    "frozen_model_audit.json",
    "frozen_model_prediction_crosscheck.json",
    "fixture_regression_report.json",
    "dry_run_report.json",
    "implementation_manifest.json",
    "test_report.json",
    "artifact_verification.json",
    "final_decision.json",
    "run_manifest.json",
    "pre_run_report.md",
    "SHA256SUMS",
)
SHA_EXCLUDED = frozenset({"artifact_verification.json", "SHA256SUMS"})
FIXTURE_PUBLICATION_DIRECTORY = "fixture_publication"


def _strict_json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("pre-run evidence must be a plain JSON object")
    return json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )


def build_prerun_qualification_decision(
    *,
    gates: Mapping[str, Any],
    counters: Mapping[str, Any],
    confirmatory_run_authorized: bool,
) -> dict[str, Any]:
    if type(confirmatory_run_authorized) is not bool:
        raise TypeError("confirmatory_run_authorized must be bool")
    normalized_gates = {
        name: gates.get(name) for name in REQUIRED_GATE_NAMES
    }
    normalized_counters = {
        name: counters.get(name) for name in ZERO_EXECUTION_COUNTER_NAMES
    }
    qualification = bool(
        all(value is True for value in normalized_gates.values())
        and all(value == 0 and type(value) is int for value in normalized_counters.values())
    )
    if confirmatory_run_authorized and not qualification:
        raise PermissionError("cannot authorize an unqualified Confirmatory run")
    return {
        **normalized_gates,
        **normalized_counters,
        "CONFIRMATORY_RUN_AUTHORIZED": bool(
            confirmatory_run_authorized and qualification
        ),
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_PASS": "NOT_EVALUATED",
        "SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS": qualification,
        "required_gate_names": list(REQUIRED_GATE_NAMES),
        "zero_execution_counter_names": list(ZERO_EXECUTION_COUNTER_NAMES),
    }


def _report(decision: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]) -> str:
    plan = evidence["plan_audit.json"]
    fixture = evidence["fixture_regression_report.json"]
    dry = evidence["dry_run_report.json"]
    lines = [
        "# Synthetic Confirmatory Pre-Run Qualification v1",
        "",
        "This artifact contains qualification evidence only. It contains no Confirmatory snapshot, trial result, or performance estimate.",
        "",
        f"- `SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS = {str(decision['SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS']).lower()}`",
        f"- `CONFIRMATORY_RUN_AUTHORIZED = {str(decision['CONFIRMATORY_RUN_AUTHORIZED']).lower()}`",
        "- `SYNTHETIC_CONFIRMATORY_EXECUTED = false`",
        "- `SYNTHETIC_CONFIRMATORY_COMPLETE = false`",
        "- `SYNTHETIC_CONFIRMATORY_PASS = NOT_EVALUATED`",
        "- `REAL_DATA_RUN_AUTHORIZED = false`",
        "- `MEASUREMENT_PAPER_MAINLINE_AUTHORIZED = false`",
        "",
        "## Zero formal-execution counters",
        "",
        *(
            f"- `{name} = {decision[name]}`"
            for name in ZERO_EXECUTION_COUNTER_NAMES
        ),
        "",
        "## Frozen plan",
        "",
        f"- Planned snapshots/trials: `{plan.get('planned_snapshot_count')} / {plan.get('planned_trial_count')}`",
        f"- Duplicate snapshot/trial IDs: `{plan.get('duplicate_snapshot_count')} / {plan.get('duplicate_trial_count')}`",
        f"- Pairing violations: `{plan.get('pairing_violation_count')}`",
        "",
        "## Seed-free execution-chain fixture",
        "",
        f"- Fixture snapshots/trials: `{fixture.get('fixture_snapshot_count')} / {fixture.get('fixture_trial_count')}`",
        f"- Fixture qualification: `{fixture.get('CONFIRMATORY_EXECUTION_CHAIN_FIXTURE_PASS')}`",
        "",
        "## Formal dry-run",
        "",
        f"- Enumerated snapshots/trials: `{dry.get('planned_snapshot_count')} / {dry.get('planned_trial_count')}`",
        "- Confirmatory RNG, snapshot generation, backend execution, trial results, and STARTED events: `0 / 0 / 0 / 0 / 0`",
        "",
        "## Interpretation boundary",
        "",
        "This qualification authorizes a later formal run using the exact frozen manifest. It does not execute or evaluate Synthetic Confirmatory science.",
        "",
        "## Evidence inventory",
        "",
        *(
            f"- `{name}`"
            for name in ARTIFACT_FILES
            if name not in {"artifact_verification.json", "pre_run_report.md"}
        ),
        f"- `{FIXTURE_PUBLICATION_DIRECTORY}/` (complete seed-free 3/6 fixture publication)",
        "",
    ]
    return "\n".join(lines)


def _write_artifact_into(
    destination: Path,
    *,
    evidence: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    fixture_publication_dir: str | Path,
) -> dict[str, Any]:
    from .synthetic_confirmatory_artifact_verifier import (
        verify_synthetic_confirmatory_prerun_artifact,
    )
    from .fixture_publication_artifact_verifier import (
        verify_fixture_publication_artifact,
    )

    destination.mkdir(parents=True, exist_ok=False)
    evidence_names = set(ARTIFACT_FILES) - {
        "artifact_verification.json",
        "final_decision.json",
        "run_manifest.json",
        "pre_run_report.md",
        "SHA256SUMS",
    }
    if set(evidence) != evidence_names:
        raise ValueError("pre-run evidence inventory differs from exact contract")
    for name, value in evidence.items():
        write_json(destination / name, _strict_json_copy(value))
    fixture_source = Path(fixture_publication_dir).resolve()
    fixture_verification = verify_fixture_publication_artifact(
        fixture_source, write_report=False
    )
    if fixture_verification.get("FIXTURE_ARTIFACT_VERIFICATION_PASS") is not True:
        raise ValueError("seed-free fixture publication did not verify before embedding")
    shutil.copytree(
        fixture_source,
        destination / FIXTURE_PUBLICATION_DIRECTORY,
        copy_function=shutil.copy2,
    )
    write_json(destination / "final_decision.json", _strict_json_copy(decision))
    write_json(destination / "run_manifest.json", _strict_json_copy(run_manifest))
    (destination / "pre_run_report.md").write_text(
        _report(decision, evidence), encoding="utf-8"
    )
    listed = sorted(
        candidate.relative_to(destination).as_posix()
        for candidate in destination.rglob("*")
        if candidate.is_file()
        and candidate.relative_to(destination).as_posix() not in SHA_EXCLUDED
    )
    (destination / "SHA256SUMS").write_text(
        "".join(
            f"{file_sha256(destination / name)}  {name}\n" for name in listed
        ),
        encoding="utf-8",
    )
    verification = verify_synthetic_confirmatory_prerun_artifact(
        destination, write_report=True
    )
    read_only = verify_synthetic_confirmatory_prerun_artifact(
        destination, write_report=False
    )
    if verification != read_only or verification.get(
        "CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"
    ) is not True:
        raise ValueError("Synthetic Confirmatory pre-run artifact verification failed")
    return verification


def publish_prerun_artifact_atomic(
    output_dir: str | Path,
    *,
    evidence: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    fixture_publication_dir: str | Path,
) -> dict[str, Any]:
    """Publish 15 root files plus complete fixture evidence without replacement."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace pre-run artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    # _write_artifact_into requires a not-yet-existing directory.
    staging = temporary / "artifact"
    try:
        verification = _write_artifact_into(
            staging,
            evidence=evidence,
            decision=decision,
            run_manifest=run_manifest,
            fixture_publication_dir=fixture_publication_dir,
        )
        os.replace(staging, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verification


__all__ = [
    "ARTIFACT_FILES",
    "FIXTURE_PUBLICATION_DIRECTORY",
    "REQUIRED_GATE_NAMES",
    "SHA_EXCLUDED",
    "ZERO_EXECUTION_COUNTER_NAMES",
    "build_prerun_qualification_decision",
    "publish_prerun_artifact_atomic",
]
