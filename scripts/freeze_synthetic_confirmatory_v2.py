#!/usr/bin/env python3
"""Generate only declarative Synthetic Confirmatory v2 assets.

This script contains no NumPy import, RNG construction, snapshot builder, or
backend import.  Formal seeds are written as identities only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if row[name] is None else row[name] for name in fields})


def build_assets(root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root / "src"))
    from phase_a_harness.synthetic_confirmatory_v2_contract import (
        BACKENDS,
        BOOTSTRAP_SEED,
        GATE_RELATIVE,
        GATE_SCHEMA,
        GEOMETRY_SEEDS,
        LINEAGE_SCHEMA,
        MEASUREMENT_SEEDS,
        NAMESPACE,
        PROTOCOL_DOCUMENT_RELATIVE,
        PROTOCOL_RELATIVE,
        PROTOCOL_SCHEMA,
        SCENES,
        SEED_SCHEDULE_RELATIVE,
        SNAPSHOT_COUNT,
        SNAPSHOT_FIELDS,
        SNAPSHOT_PLAN_RELATIVE,
        TRIAL_COUNT,
        TRIAL_FIELDS,
        TRIAL_PLAN_RELATIVE,
        audit_v2_plan,
        canonical_identity_sha256,
        derive_seed,
        expected_snapshot_id,
    )

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
        "bootstrap_seed": BOOTSTRAP_SEED,
        "derivation": {
            "digest": "SHA256(UTF-8(namespace|domain|index))",
            "integer": "int.from_bytes(digest[0:8], byteorder=big, signed=false) % 2147483647",
            "zero_remap": "0 -> 1",
        },
        "geometry_seeds": list(GEOMETRY_SEEDS),
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "namespace": NAMESPACE,
        "records": records,
        "schema_version": "synthetic_confirmatory_v2_seed_schedule_v1",
    }
    schedule = {
        **schedule_core,
        "seed_schedule_payload_sha256": canonical_identity_sha256(schedule_core),
    }
    _write_json(root / SEED_SCHEDULE_RELATIVE, schedule)

    v1_gate = json.loads(
        (root / "protocols/synthetic_confirmatory_gate_contract.json").read_text(
            encoding="utf-8"
        )
    )
    gate_core = {
        "all_hypotheses_required": True,
        "hypotheses": v1_gate["hypotheses"],
        "hypothesis_count": 6,
        "schema_version": GATE_SCHEMA,
    }
    gate = {
        **gate_core,
        "gate_contract_payload_sha256": canonical_identity_sha256(gate_core),
    }
    _write_json(root / GATE_RELATIVE, gate)

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
    _write_csv(root / SNAPSHOT_PLAN_RELATIVE, SNAPSHOT_FIELDS, snapshots)
    _write_csv(root / TRIAL_PLAN_RELATIVE, TRIAL_FIELDS, trials)
    plan = audit_v2_plan(root / SNAPSHOT_PLAN_RELATIVE, root / TRIAL_PLAN_RELATIVE)
    if not all(
        plan[name] is True
        for name in (
            "CONFIRMATORY_PLAN_COUNT_PASS",
            "CONFIRMATORY_PLAN_UNIQUENESS_PASS",
            "CONFIRMATORY_PLAN_PAIRING_PASS",
            "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS",
        )
    ):
        raise RuntimeError("generated v2 plan failed its independent audit")

    v1_protocol = json.loads(
        (root / "protocols/synthetic_confirmatory_protocol_v1.json").read_text(
            encoding="utf-8"
        )
    )
    protocol_core = {
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_PROTOCOL_READY": True,
        "backend_count": 2,
        "backends": list(BACKENDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "condition_count": 3,
        "conditions": list(v1_protocol["conditions"]),
        "confirmatory_seed_provenance_pass": True,
        "development_model_lock_sha256": v1_protocol["development_model_lock_sha256"],
        "development_model_weighting": v1_protocol["development_model_weighting"],
        "gate_contract_payload_sha256": gate["gate_contract_payload_sha256"],
        "geometry_seeds": list(GEOMETRY_SEEDS),
        "history": {
            "root_cause": "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON",
            "v1_failure_tag": "archive/zero-perturbation-synthetic-confirmatory-v1-ideal-lineage-fail",
            "v1_status": "NOT_EVALUATED",
        },
        "lineage_schema_version": LINEAGE_SCHEMA,
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "native_trial_count": 0,
        "planned_snapshot_count": SNAPSHOT_COUNT,
        "planned_snapshot_identity_sha256": plan["planned_snapshot_identity_sha256"],
        "planned_trial_count": TRIAL_COUNT,
        "planned_trial_identity_sha256": plan["planned_trial_identity_sha256"],
        "registration_execution_count": 0,
        "scene_count": 7,
        "scenes": list(SCENES),
        "schema_version": PROTOCOL_SCHEMA,
        "scientific_survival_audit_pass": True,
        "seed_namespace": NAMESPACE,
        "seed_schedule_payload_sha256": schedule["seed_schedule_payload_sha256"],
        "snapshot_generation_count": 0,
    }
    protocol = {
        **protocol_core,
        "protocol_payload_sha256": canonical_identity_sha256(protocol_core),
    }
    _write_json(root / PROTOCOL_RELATIVE, protocol)
    (root / PROTOCOL_DOCUMENT_RELATIVE).write_text(
        "\n".join(
            [
                "# Synthetic Confirmatory Protocol v2",
                "",
                "Version 2 repairs only the IDEAL parent-lineage execution contract.",
                "The seven scenes, three conditions, 595 snapshots, 1,190 trials,",
                "Open3D/PCL parameters, H1--H6, q95 method, common association,",
                "turnover definition, and four frozen Development models are unchanged.",
                "",
                f"- Seed namespace: `{NAMESPACE}`",
                f"- Parent lineage schema: `{LINEAGE_SCHEMA}`",
                "- IDEAL parent selection: all Phase-A range-eligible canonical target rows",
                "- Source transform: `R_reference.T @ (parent_map_f64 - t_reference)`",
                "- Source quantization: exactly once to little-endian float32",
                "- Validation: parent-row correspondence plus the frozen Phase A closure contract",
                "- Formal execution in this pre-run freeze: not executed",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"plan_audit": plan, "protocol": protocol, "schedule": schedule}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.manifest:
        sys.path.insert(0, str(root / "src"))
        from phase_a_harness.synthetic_confirmatory_v2_contract import write_manifest

        value = write_manifest(root, authorized=False, replace=True)
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(json.dumps(build_assets(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
