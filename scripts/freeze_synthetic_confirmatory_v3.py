#!/usr/bin/env python3
"""Freeze declarative Synthetic Confirmatory v3 assets without instantiation.

The script performs text/Git collision scans, SHA-256 derivation, CSV identity
enumeration, and strict manifest binding only.  It imports no random-number,
snapshot-builder, runner, or backend module and never creates the formal
runtime root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_EVIDENCE_DIR = Path("/tmp/synthetic_confirmatory_v3_prerun_evidence")
FIXTURE_REPORT = Path(
    "/home/lj/zero_perturbation_runtime/qualification/"
    "synthetic_confirmatory_v3_prerun_preseed_9bd944/working_inventory/"
    "v3_adapter_fixture_qualification.json"
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fields), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: "" if row[name] is None else row[name] for name in fields}
            )


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr}")
    return result.stdout.strip()


def _fixture_prerequisite() -> dict[str, Any]:
    if not FIXTURE_REPORT.is_file():
        raise FileNotFoundError("v3 seed-free adapter fixture evidence is missing")
    value = json.loads(FIXTURE_REPORT.read_text(encoding="utf-8"))
    required_true = (
        "V3_EXECUTION_ADAPTER_FIXTURE_PASS",
        "V3_RUNTIME_PATH_POLICY_PASS",
        "V3_GIT_GATE_FIXTURE_PASS",
        "V3_PRIMARY_VERIFIER_FIXTURE_PASS",
        "V3_PUBLISHER_FIXTURE_PASS",
        "V3_ARTIFACT_VERIFIER_FIXTURE_PASS",
    )
    if type(value) is not dict or any(value.get(name) is not True for name in required_true):
        raise PermissionError("v3 seed derivation prerequisite fixture did not pass")
    required_zero = (
        "formal_seed_reference_count",
        "formal_seed_rng_instantiation_count",
        "valid_snapshot_reexecution_count",
        "valid_trial_reexecution_count",
        "snapshot_checksum_change_after_resume",
        "trial_checksum_change_after_resume",
        "source_repository_runtime_file_read_count",
        "source_repository_runtime_import_count",
    )
    if any(value.get(name) != 0 for name in required_zero):
        raise PermissionError("v3 seed-free adapter fixture contains forbidden use")
    reports = value.get("git_gate_reports")
    if (
        type(reports) is not list
        or len(reports) != 22
        or any(row.get("RUNTIME_GIT_GATE_PASS") is not True for row in reports)
    ):
        raise PermissionError("v3 seed-free adapter Git Gate evidence is incomplete")
    return {
        "artifact_verifier_fixture_pass": True,
        "backend_trial_counts": value["fresh"]["backend_trial_counts"],
        "fixture_report_path": str(FIXTURE_REPORT),
        "fixture_report_sha256": hashlib.sha256(FIXTURE_REPORT.read_bytes()).hexdigest(),
        "fixture_snapshot_count": value["fresh"]["fixture_snapshot_count"],
        "fixture_trial_count": value["fresh"]["fixture_trial_count"],
        "fresh_resume_scientific_equivalence": True,
        "git_gate_pass_count": 22,
        "native_execution_count": value["fresh"]["native_execution_count"],
        "primary_independent_exact_match": True,
        "publisher_fixture_pass": True,
        "seed_free": True,
        "valid_snapshot_reexecution_count": 0,
        "valid_trial_reexecution_count": 0,
    }


def execution_profile_payload() -> dict[str, Any]:
    """Build the declarative formal profile without touching runtime state."""

    from phase_a_harness.synthetic_confirmatory_v3_contract import (
        EXECUTION_PROFILE_SCHEMA,
        FINAL_DECISION_SCHEMA,
        FORMAL_BRANCH,
        FORMAL_PRERUN_TAG,
        FORMAL_RUN_ID,
        FORMAL_RUNTIME_ROOT,
        FORMAL_WORKERS,
        MANIFEST_RELATIVE,
        PUBLISHER_FIGURES,
        PUBLISHER_ROOT_FILES,
        PUBLISHER_TABLES,
        RUNTIME_PATHS,
        canonical_identity_sha256,
    )

    core = {
        "configuration_authority": MANIFEST_RELATIVE.as_posix(),
        "expected_branch": FORMAL_BRANCH,
        "expected_release_tag": FORMAL_PRERUN_TAG,
        "final_decision_schema": FINAL_DECISION_SCHEMA,
        "entrypoints": {
            "analysis": "scripts/analyze_synthetic_confirmatory_v3.py",
            "artifact_verifier": (
                "scripts/verify_synthetic_confirmatory_v3_artifact.py"
            ),
            "independent_verifier": "scripts/verify_synthetic_confirmatory_v3.py",
            "publisher": "scripts/publish_synthetic_confirmatory_v3.py",
            "runner": "scripts/run_synthetic_confirmatory_v3.py",
        },
        "formal_cli_contract": {
            "boolean_switches": ["--resume", "--dry-run"],
            "required_options": [
                "--manifest",
                "--runtime-root",
                "--run-id",
                "--workers",
            ],
            "science_override_options": [],
        },
        "formal_execution_state": "NOT_EXECUTED",
        "formal_runtime_creation_count": 0,
        "fresh_resume_modes_frozen": True,
        "publisher_inventory": {
            "figure_count": len(PUBLISHER_FIGURES),
            "figures": list(PUBLISHER_FIGURES),
            "root_file_count": len(PUBLISHER_ROOT_FILES),
            "root_files": list(PUBLISHER_ROOT_FILES),
            "table_count": len(PUBLISHER_TABLES),
            "tables": list(PUBLISHER_TABLES),
        },
        "run_id": FORMAL_RUN_ID,
        "runtime_paths": {
            "runtime_root": str(FORMAL_RUNTIME_ROOT),
            **{name: str(path) for name, path in RUNTIME_PATHS.items()},
        },
        "schema_version": EXECUTION_PROFILE_SCHEMA,
        "scientific_evaluation_state": "NOT_EVALUATED",
        "this_file_executes_commands": False,
        "workers": FORMAL_WORKERS,
        "zero_instance_counters": {
            "backend_execution_count": 0,
            "rng_instantiation_count": 0,
            "scientific_result_count": 0,
            "snapshot_construction_count": 0,
            "started_event_count": 0,
            "trial_result_count": 0,
        },
    }
    return {
        **core,
        "execution_profile_payload_sha256": canonical_identity_sha256(core),
    }


def build_assets(root: Path, *, evidence_dir: Path) -> dict[str, Any]:
    repository = root.resolve()
    sys.path.insert(0, str(repository / "src"))
    from phase_a_harness.contracts import file_sha256, write_json
    from phase_a_harness.synthetic_confirmatory_v3_contract import (
        BACKENDS,
        BOOTSTRAP_SEED,
        CONDITION_SNAPSHOT_COUNTS,
        CONDITIONS,
        DERIVATION_PREREQUISITE_COMMIT,
        EXECUTION_PROFILE_RELATIVE,
        FORMAL_BRANCH,
        FORMAL_PRERUN_TAG,
        FORMAL_RUN_ID,
        FORMAL_RUNTIME_ROOT,
        FORMAL_WORKERS,
        FROZEN_MODEL_RELATIVE,
        FROZEN_MODEL_SHA256,
        GATE_RELATIVE,
        GATE_SCHEMA,
        GEOMETRY_SEEDS,
        LINEAGE_SCHEMA,
        MANIFEST_RELATIVE,
        MEASUREMENT_SEEDS,
        NAMESPACE,
        PRE_RUN_FINAL_DECISION_RELATIVE,
        PROTOCOL_DOCUMENT_RELATIVE,
        PROTOCOL_RELATIVE,
        PROTOCOL_SCHEMA,
        RUNTIME_LIFECYCLE_BASELINE_COMMIT,
        SCENES,
        SEED_SCHEDULE_RELATIVE,
        SEED_SCHEDULE_SCHEMA,
        SNAPSHOT_COUNT,
        SNAPSHOT_FIELDS,
        SNAPSHOT_PLAN_RELATIVE,
        TRIAL_COUNT,
        TRIAL_FIELDS,
        TRIAL_PLAN_RELATIVE,
        V2_GATE_RELATIVE,
        audit_v3_plan,
        canonical_identity_sha256,
        derive_seed,
        expected_snapshot_id,
        scientific_diff_v2_to_v3,
        write_manifest,
    )
    from phase_a_harness.synthetic_confirmatory_v3_seed_audit import (
        audit_v3_seed_provenance,
        scan_v3_prewrite_collisions,
        write_v3_seed_provenance_evidence,
    )

    if _git(repository, "branch", "--show-current") != FORMAL_BRANCH:
        raise PermissionError("v3 assets must be frozen on the exact pre-run branch")
    if _git(repository, "rev-parse", "HEAD") != DERIVATION_PREREQUISITE_COMMIT:
        raise PermissionError("v3 assets must derive from the adapter fixture commit")
    if (
        _git(
            repository,
            "merge-base",
            "--is-ancestor",
            RUNTIME_LIFECYCLE_BASELINE_COMMIT,
            "HEAD",
        )
        != ""
    ):
        raise RuntimeError("unexpected merge-base output")
    if FORMAL_RUNTIME_ROOT.exists():
        raise PermissionError("formal v3 runtime root exists before freeze")

    fixture = _fixture_prerequisite()
    collision_scan = scan_v3_prewrite_collisions(repository)
    if collision_scan.get("V3_PREWRITE_COLLISION_SCAN_PASS") is not True:
        raise PermissionError("v3 namespace or seed collision detected before declaration")

    records = []
    for domain, count in (("geometry", 5), ("measurement", 3), ("bootstrap", 1)):
        for index in range(count):
            payload = f"{NAMESPACE}|{domain}|{index}"
            records.append(
                {
                    "domain": domain,
                    "index": index,
                    "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    "seed": derive_seed(domain, index),
                }
            )
    schedule_core = {
        "allowed_use_classes_this_freeze": [
            "SEED_DECLARATION",
            "SEED_PLAN_REFERENCE",
        ],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "derivation": {
            "digest": "SHA256(UTF-8(namespace|domain|index))",
            "domains": {"bootstrap": [0], "geometry": [0, 1, 2, 3, 4], "measurement": [0, 1, 2]},
            "integer": (
                "int.from_bytes(digest[0:8], byteorder=big, signed=false) "
                "% 2147483647"
            ),
            "zero_remap": "0 -> 1",
        },
        "derivation_prerequisites": {
            "adapter_fixture": fixture,
            "derivation_prerequisite_commit": DERIVATION_PREREQUISITE_COMMIT,
            "runtime_lifecycle_baseline_commit": RUNTIME_LIFECYCLE_BASELINE_COMMIT,
        },
        "forbidden_use_counts_this_freeze": {
            "backend_execution_count": 0,
            "rng_instantiation_count": 0,
            "scientific_result_count": 0,
            "snapshot_construction_count": 0,
            "trial_result_count": 0,
        },
        "geometry_seeds": list(GEOMETRY_SEEDS),
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "namespace": NAMESPACE,
        "prewrite_collision_scan": collision_scan,
        "records": records,
        "schema_version": SEED_SCHEDULE_SCHEMA,
    }
    schedule = {
        **schedule_core,
        "seed_schedule_payload_sha256": canonical_identity_sha256(schedule_core),
    }
    _write_json(repository / SEED_SCHEDULE_RELATIVE, schedule)

    v2_gate = json.loads((repository / V2_GATE_RELATIVE).read_text(encoding="utf-8"))
    gate_core = {
        "all_hypotheses_required": v2_gate["all_hypotheses_required"],
        "hypotheses": v2_gate["hypotheses"],
        "hypothesis_count": v2_gate["hypothesis_count"],
        "schema_version": GATE_SCHEMA,
    }
    gate = {
        **gate_core,
        "gate_contract_payload_sha256": canonical_identity_sha256(gate_core),
    }
    _write_json(repository / GATE_RELATIVE, gate)

    snapshots: list[dict[str, Any]] = []
    for scene in SCENES:
        for geometry in GEOMETRY_SEEDS:
            for condition, semantics in (
                ("IDEAL_MATCHED", "ONE_CONTROL_INPUT"),
                (
                    "INDEPENDENT_NOISE_FREE",
                    "ONE_DETERMINISTIC_INDEPENDENT_INPUT",
                ),
            ):
                identity = {
                    "scene_variant": scene,
                    "condition": condition,
                    "geometry_seed": geometry,
                    "measurement_seed": None,
                    "repeat_index": 0,
                }
                snapshots.append(
                    {
                        "planned_snapshot_id": expected_snapshot_id(identity),
                        **identity,
                        "planned_backend_count": 2,
                        "replicate_semantics": semantics,
                    }
                )
            for measurement in MEASUREMENT_SEEDS:
                for repeat in range(5):
                    identity = {
                        "scene_variant": scene,
                        "condition": "FULL_NOISE",
                        "geometry_seed": geometry,
                        "measurement_seed": measurement,
                        "repeat_index": repeat,
                    }
                    snapshots.append(
                        {
                            "planned_snapshot_id": expected_snapshot_id(identity),
                            **identity,
                            "planned_backend_count": 2,
                            "replicate_semantics": (
                                "FIFTEEN_STOCHASTIC_INPUTS_PER_SCENE_GEOMETRY"
                            ),
                        }
                    )
    trials = [
        {
            "planned_trial_id": f"{row['planned_snapshot_id']}::{backend}",
            "planned_snapshot_id": row["planned_snapshot_id"],
            "scene_variant": row["scene_variant"],
            "condition": row["condition"],
            "geometry_seed": row["geometry_seed"],
            "measurement_seed": row["measurement_seed"],
            "repeat_index": row["repeat_index"],
            "backend": backend,
        }
        for row in snapshots
        for backend in BACKENDS
    ]
    _write_csv(repository / SNAPSHOT_PLAN_RELATIVE, SNAPSHOT_FIELDS, snapshots)
    _write_csv(repository / TRIAL_PLAN_RELATIVE, TRIAL_FIELDS, trials)
    plan = audit_v3_plan(
        repository / SNAPSHOT_PLAN_RELATIVE, repository / TRIAL_PLAN_RELATIVE
    )
    if plan.get("V3_PLAN_PASS") is not True:
        raise RuntimeError("generated v3 plan failed its independent static audit")

    v2_protocol = json.loads(
        (repository / "protocols/synthetic_confirmatory_protocol_v2.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_decision_path = (
        repository / "artifacts/runtime_lifecycle_qualification_v1/final_decision.json"
    )
    runtime_decision = json.loads(runtime_decision_path.read_text(encoding="utf-8"))
    if not (
        runtime_decision.get("RUNTIME_LIFECYCLE_QUALIFICATION_PASS") is True
        and runtime_decision.get("CONFIRMATORY_V3_PRE_RUN_DESIGN_AUTHORIZED") is True
        and runtime_decision.get("CONFIRMATORY_V3_SEED_DERIVATION_AUTHORIZED") is True
        and runtime_decision.get("CONFIRMATORY_V3_RUN_AUTHORIZED") is False
    ):
        raise PermissionError("runtime lifecycle authority does not authorize v3 derivation")
    protocol_core = {
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "CONFIRMATORY_V3_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_PROTOCOL_READY": True,
        "SYNTHETIC_CONFIRMATORY_V3_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_V3_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V3_PASS": "NOT_EVALUATED",
        "authorization_authority": (
            "VERIFIED_PRE_RUN_FINAL_DECISION_AT_"
            + PRE_RUN_FINAL_DECISION_RELATIVE.as_posix()
        ),
        "backend_count": 2,
        "backends": list(BACKENDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "condition_count": 3,
        "conditions": list(CONDITIONS),
        "confirmatory_seed_provenance_pass": True,
        "development_model_lock_sha256": FROZEN_MODEL_SHA256,
        "development_model_weighting": v2_protocol["development_model_weighting"],
        "formal_execution_state": "NOT_EXECUTED",
        "gate_contract_payload_sha256": gate["gate_contract_payload_sha256"],
        "geometry_seeds": list(GEOMETRY_SEEDS),
        "history": {
            "v1": {
                "failure_reason": "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON",
                "failure_tag": (
                    "archive/zero-perturbation-synthetic-confirmatory-v1-"
                    "ideal-lineage-fail"
                ),
                "scientific_state": "NOT_EVALUATED",
                "seed_reuse_authorized": False,
            },
            "v2": {
                "failure_reason": "RUNTIME_PATH_AND_GIT_GATE_LIFECYCLE_DEFECT",
                "failure_tag": (
                    "archive/zero-perturbation-synthetic-confirmatory-v2-"
                    "runtime-lifecycle-fail"
                ),
                "scientific_state": "NOT_EVALUATED",
                "seed_reuse_authorized": False,
            },
        },
        "lineage_schema_version": LINEAGE_SCHEMA,
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "native_trial_count": 0,
        "planned_snapshot_count": SNAPSHOT_COUNT,
        "planned_snapshot_identity_sha256": plan[
            "planned_snapshot_identity_sha256"
        ],
        "planned_trial_count": TRIAL_COUNT,
        "planned_trial_identity_sha256": plan["planned_trial_identity_sha256"],
        "registration_execution_count": 0,
        "runtime_lifecycle_qualification_binding": {
            "decision_path": (
                "artifacts/runtime_lifecycle_qualification_v1/final_decision.json"
            ),
            "decision_sha256": file_sha256(runtime_decision_path),
            "qualification_pass": True,
        },
        "scene_count": 7,
        "scenes": list(SCENES),
        "schema_version": PROTOCOL_SCHEMA,
        "scientific_evaluation_state": "NOT_EVALUATED",
        "scientific_survival_audit_pass": True,
        "seed_declaration_authority": SEED_SCHEDULE_RELATIVE.as_posix(),
        "seed_namespace": NAMESPACE,
        "seed_schedule_payload_sha256": schedule[
            "seed_schedule_payload_sha256"
        ],
        "snapshot_generation_count": 0,
    }
    protocol = {
        **protocol_core,
        "protocol_payload_sha256": canonical_identity_sha256(protocol_core),
    }
    _write_json(repository / PROTOCOL_RELATIVE, protocol)
    (repository / PROTOCOL_DOCUMENT_RELATIVE).write_text(
        "\n".join(
            [
                "# Synthetic Confirmatory Protocol v3",
                "",
                "Version 3 changes only seed identity and qualified external-runtime binding.",
                "The seven scenes, three conditions, 595 snapshots, 1,190 trials,",
                "Open3D/PCL algorithms and parameters, H1--H6, q95 method, common",
                "association, turnover, systematic offset, parent-index lineage, and",
                "frozen Development models are unchanged from the v2 scientific contract.",
                "",
                f"- Seed namespace: `{NAMESPACE}`",
                f"- Seed declaration authority: `{SEED_SCHEDULE_RELATIVE.as_posix()}`",
                f"- Parent lineage schema: `{LINEAGE_SCHEMA}`",
                f"- Formal run ID: `{FORMAL_RUN_ID}`",
                f"- Formal runtime root: `{FORMAL_RUNTIME_ROOT}`",
                "- INDEPENDENT_NOISE_FREE: one input per scene/geometry, no measurement seed, repeat 0",
                "- Native backend: prohibited; planned count 0",
                "- Formal state in this freeze: NOT_EXECUTED / NOT_EVALUATED",
                "- Run authorization: only a verified pre-run final_decision may authorize it",
                "",
            ]
        ),
        encoding="utf-8",
    )

    profile = execution_profile_payload()
    _write_json(repository / EXECUTION_PROFILE_RELATIVE, profile)

    manifest = write_manifest(repository, replace=True)
    if manifest.get("expected_branch") != FORMAL_BRANCH:
        raise RuntimeError("v3 manifest branch binding changed")
    if manifest.get("expected_release_tag") != FORMAL_PRERUN_TAG:
        raise RuntimeError("v3 manifest tag binding changed")
    if manifest.get("publisher_inventory", {}).get("table_count") != 7:
        raise RuntimeError("v3 publisher table inventory changed")
    if manifest.get("publisher_inventory", {}).get("figure_count") != 3:
        raise RuntimeError("v3 publisher figure inventory changed")
    if manifest.get("publisher_inventory", {}).get("root_file_count") != 7:
        raise RuntimeError("v3 publisher root-file inventory changed")

    provenance = audit_v3_seed_provenance(repository)
    if provenance.get("STATIC_V3_SEED_PROVENANCE_AUDIT_PASS") is not True:
        raise RuntimeError("generated v3 assets failed seed provenance audit")
    science_diff = scientific_diff_v2_to_v3(repository)
    if science_diff.get("V2_TO_V3_SCIENTIFIC_DIFF_PASS") is not True:
        raise RuntimeError("generated v3 assets changed frozen science")
    write_v3_seed_provenance_evidence(repository, evidence_dir)
    write_json(evidence_dir / "v2_to_v3_scientific_diff.json", science_diff)

    if FORMAL_RUNTIME_ROOT.exists():
        raise RuntimeError("v3 asset freeze created the formal runtime root")
    return {
        "formal_runtime_root_exists": False,
        "manifest": manifest,
        "plan_audit": plan,
        "protocol": protocol,
        "scientific_diff": science_diff,
        "seed_provenance_audit": provenance,
        "seed_schedule": schedule,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    args = parser.parse_args(argv)
    report = build_assets(args.root, evidence_dir=args.evidence_dir.resolve())
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
