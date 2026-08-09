"""Synthetic Confirmatory v2 I/O adapter over the frozen H1--H6 core."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common_association_analysis import (
    prepare_common_association_context,
    safe_analyze_estimated_transform,
)
from .rotation_metrics import rotation_metric_audit
from .synthetic_confirmatory_analysis import analyze_synthetic_confirmatory_records
from .synthetic_confirmatory_v2_contract import (
    FORMAL_ANALYSIS_SCHEMA,
    FORMAL_OUTPUT_DIR,
    FORMAL_RUN_ID,
    RAW_RESULT_MANIFEST_SCHEMA,
    SNAPSHOT_LOCK_RELATIVE,
    SNAPSHOT_PLAN_RELATIVE,
    TRIAL_PLAN_RELATIVE,
    typed_snapshot_rows,
    typed_trial_rows,
    verify_manifest,
)
from .synthetic_confirmatory_v2_snapshot_builder import (
    read_v2_snapshot,
    validate_v2_snapshot_lock,
)


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON root must be object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    value = (root / relative).resolve()
    if value != root and root not in value.parents:
        raise ValueError("v2 path escaped standalone harness")
    return value


def _validate_trial(value: Mapping[str, Any]) -> dict[str, Any]:
    from .phase_a_trial_result_schema import validate_phase_a_trial_result_strict
    from .full_synthetic_trial_result import validate_full_synthetic_trial_result_strict

    condition = value.get("condition")
    if condition == "IDEAL_MATCHED":
        return validate_phase_a_trial_result_strict(value)
    return validate_full_synthetic_trial_result_strict(value)


def load_v2_raw(
    *, manifest_path: str | Path, run_dir: str | Path
) -> tuple[list[dict[str, Any]], dict[str, Any], Path, dict[str, Any]]:
    repository, manifest = verify_manifest(manifest_path, require_authorized=True)
    plan_path = _inside(repository, manifest["planned_trials_path"])
    plans = typed_trial_rows(plan_path)
    if len(plans) != 1190 or len({row["planned_trial_id"] for row in plans}) != 1190:
        raise ValueError("v2 trial plan inventory mismatch")
    if _sha256(plan_path) != manifest["planned_trials_sha256"]:
        raise ValueError("v2 trial plan SHA mismatch")
    directory = Path(run_dir).resolve()
    if directory != _inside(repository, manifest["formal_output_dir"]):
        raise ValueError("v2 result directory differs from manifest")
    raw_path = directory / "raw_result_manifest.json"
    raw = _strict_object(raw_path)
    if (
        set(raw) != {"schema_version", "run_id", "results"}
        or raw.get("schema_version") != RAW_RESULT_MANIFEST_SCHEMA
        or raw.get("run_id") != FORMAL_RUN_ID
        or type(raw.get("results")) is not dict
    ):
        raise ValueError("v2 raw-result manifest schema mismatch")
    by_id = {row["planned_trial_id"]: row for row in plans}
    if set(raw["results"]) != set(by_id):
        raise ValueError("v2 raw-result inventory incomplete or has extras")
    raw_root = (directory / "raw_results").resolve()
    output = []
    for trial_id in sorted(by_id):
        entry = raw["results"][trial_id]
        if (
            type(entry) is not dict
            or set(entry) != {"planned_trial_id", "path", "sha256"}
            or entry.get("planned_trial_id") != trial_id
            or not isinstance(entry.get("path"), str)
            or Path(entry["path"]).name != entry["path"]
        ):
            raise ValueError("v2 raw-result entry mismatch")
        path = (raw_root / entry["path"]).resolve()
        if path.parent != raw_root or _sha256(path) != entry.get("sha256"):
            raise ValueError("v2 raw-result file SHA mismatch")
        result = _validate_trial(_strict_object(path))
        plan = by_id[trial_id]
        exact = {
            "planned_trial_id": trial_id,
            "snapshot_id": plan["planned_snapshot_id"],
            "scene_variant": plan["scene_variant"],
            "condition": plan["condition"],
            "backend": plan["backend"],
        }
        if any(result.get(name) != expected for name, expected in exact.items()):
            raise ValueError("v2 raw result differs from frozen plan")
        output.append(
            {
                **result,
                "backend_schema_name": result["backend"],
                "geometry_seed": plan["geometry_seed"],
                "measurement_seed": plan["measurement_seed"],
                "repeat_index": plan["repeat_index"],
                "planned_snapshot_id": plan["planned_snapshot_id"],
            }
        )
    return output, raw, repository, manifest


def validate_primary_lineage_inventory(
    *, repository: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    snapshots = typed_snapshot_rows(repository / SNAPSHOT_PLAN_RELATIVE)
    lock = validate_v2_snapshot_lock(
        repository / SNAPSHOT_LOCK_RELATIVE,
        repository / manifest["snapshot_cache_root"],
        snapshots,
    )
    lock_by_id = {row["snapshot_id"]: row for row in lock["snapshots"]}
    ideal = nonideal = lineage_violations = 0
    by_id: dict[str, dict[str, Any]] = {}
    for plan in snapshots:
        item = read_v2_snapshot(
            repository / manifest["snapshot_cache_root"], plan,
            expected_lock_entry=lock_by_id[plan["planned_snapshot_id"]],
        )
        metadata = item["metadata"]
        if plan["condition"] == "IDEAL_MATCHED":
            ideal += 1
            lineage_violations += int(
                not metadata["source_has_target_parent_lineage"]
                or not metadata["source_is_target_subset"]
                or metadata["lineage_closure_violation_count"] != 0
            )
        else:
            nonideal += 1
            lineage_violations += int(
                metadata["source_has_target_parent_lineage"]
                or metadata["source_is_target_subset"]
                or item["parent_indices"] is not None
            )
        by_id[plan["planned_snapshot_id"]] = item
    report = {
        "IDEAL_PARENT_LINEAGE_COUNT": ideal,
        "LINEAGE_INTEGRITY_PASS": ideal == 35 and nonideal == 560 and lineage_violations == 0,
        "LINEAGE_VIOLATION_COUNT": lineage_violations,
        "NONIDEAL_NO_LINEAGE_COUNT": nonideal,
        "schema_version": "synthetic_confirmatory_v2_primary_lineage_inventory_v1",
    }
    if report["LINEAGE_INTEGRITY_PASS"] is not True:
        raise ValueError("v2 primary lineage inventory failed")
    return report, by_id


def recompute_v2_common_records(
    *, trials: Sequence[Mapping[str, Any]], snapshots_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[str(row["planned_snapshot_id"])].append(row)
    normalized: list[dict[str, Any]] = []
    common: list[dict[str, Any]] = []
    for snapshot_id in sorted(grouped):
        snapshot = snapshots_by_id[snapshot_id]
        source, target, reference = (
            snapshot["source"], snapshot["target"], snapshot["reference"]
        )
        context = None
        if any(row["condition"] != "IDEAL_MATCHED" for row in grouped[snapshot_id]):
            context = prepare_common_association_context(
                source, target, reference, snapshot_id=snapshot_id
            )
        for row in grouped[snapshot_id]:
            if any(
                row.get(name) != snapshot[name]
                for name in (
                    "snapshot_checksum", "source_checksum", "target_checksum",
                    "reference_pose_checksum",
                )
            ):
                raise ValueError("v2 trial/snapshot checksum mismatch")
            updated = dict(row)
            transform = row.get("final_transform_4x4")
            if transform is not None and not row["solver_failure"] and row["finite_output"]:
                estimate = np.asarray(transform, dtype=np.float64)
                vector = estimate[:3, 3] - reference[:3, 3]
                rotation = rotation_metric_audit(
                    estimate[:3, :3], reference[:3, :3]
                )["rotation_error_rad"]
                if rotation is None:
                    raise ValueError("v2 rotation recomputation failed")
                if (
                    abs(float(row["translation_update_m"]) - float(np.linalg.norm(vector))) > 1e-12
                    or abs(float(row["rotation_update_rad"]) - float(rotation)) > 1e-12
                ):
                    raise ValueError("v2 stored metric recomputation mismatch")
                updated["translation_vector"] = vector.astype(float).tolist()
                updated["translation_error_m"] = float(np.linalg.norm(vector))
                updated["rotation_error_rad"] = float(rotation)
            normalized.append(updated)
            if row["condition"] != "IDEAL_MATCHED" and not row["solver_failure"] and row["finite_output"]:
                if context is None or transform is None:
                    raise ValueError("v2 successful nonideal trial lacks association input")
                identifiers = {
                    "planned_trial_id": row["planned_trial_id"],
                    "backend_schema_name": row["backend_schema_name"],
                    "scene_variant": row["scene_variant"],
                    "condition": row["condition"],
                    "geometry_seed": row["geometry_seed"],
                    "measurement_seed": row["measurement_seed"],
                    "repeat_index": row["repeat_index"],
                }
                common.append(
                    safe_analyze_estimated_transform(
                        context, np.asarray(transform, dtype=np.float64),
                        identifiers=identifiers,
                    )
                )
    return (
        sorted(normalized, key=lambda row: row["planned_trial_id"]),
        sorted(common, key=lambda row: row["planned_trial_id"]),
    )


def _v2_decision(report: dict[str, Any]) -> dict[str, Any]:
    decision = dict(report["final_decision"])
    decision.update(
        {
            "CONFIRMATORY_V2_RUN_AUTHORIZED": False,
            "SYNTHETIC_CONFIRMATORY_V2_COMPLETE": decision.pop(
                "SYNTHETIC_CONFIRMATORY_COMPLETE"
            ),
            "SYNTHETIC_CONFIRMATORY_V2_EXECUTED": decision.pop(
                "SYNTHETIC_CONFIRMATORY_EXECUTED"
            ),
            "SYNTHETIC_CONFIRMATORY_V2_PASS": decision.pop(
                "SYNTHETIC_CONFIRMATORY_PASS"
            ),
        }
    )
    decision.pop("CONFIRMATORY_RUN_AUTHORIZED", None)
    report["final_decision"] = decision
    return report


def analyze_v2(*, manifest_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    trials, _raw, repository, manifest = load_v2_raw(
        manifest_path=manifest_path, run_dir=run_dir
    )
    lineage, snapshots = validate_primary_lineage_inventory(
        repository=repository, manifest=manifest
    )
    trials, common = recompute_v2_common_records(
        trials=trials, snapshots_by_id=snapshots
    )
    report = analyze_synthetic_confirmatory_records(
        trials=trials,
        common_records=common,
        model_lock=_strict_object(repository / manifest["frozen_model_path"]),
        gate_contract=_strict_object(repository / manifest["gate_contract_path"]),
        expected_geometry_seeds=_strict_object(
            repository / manifest["scientific_protocol_path"]
        )["geometry_seeds"],
    )
    report = _v2_decision(report)
    report["schema_version"] = FORMAL_ANALYSIS_SCHEMA
    report["lineage_integrity"] = lineage
    report["run_id"] = FORMAL_RUN_ID
    report["raw_result_manifest_sha256"] = _sha256(
        Path(run_dir).resolve() / "raw_result_manifest.json"
    )
    return report


def analyze_v2_fixture_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in results]
    inventory = Counter(
        (row.get("backend"), row.get("condition"), row.get("failure_classification"))
        for row in rows
    )
    expected = {
        (backend, condition, classification): 1
        for backend in ("open3d_point_to_plane", "pcl_point_to_plane")
        for condition, classification in (
            ("FIXTURE_IDENTITY", "NONE"),
            ("FIXTURE_NONIDENTITY_REFERENCE", "NONE"),
            ("FIXTURE_NO_CORRESPONDENCE", "NO_CORRESPONDENCES"),
        )
    }
    passed = bool(
        len(rows) == 6
        and inventory == Counter(expected)
        and all(type(row.get("finite_output")) is bool for row in rows)
    )
    return {
        "decision": {
            "FIXTURE_EXECUTION_CHAIN_PASS": passed,
            "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED": False,
        },
        "failure_inventory": [
            {
                "backend": key[0], "condition": key[1],
                "failure_classification": key[2], "count": count,
            }
            for key, count in sorted(inventory.items())
        ],
        "fixture_snapshot_count": 3,
        "fixture_trial_count": len(rows),
        "results": sorted(rows, key=lambda row: str(row.get("planned_trial_id"))),
        "schema_version": "synthetic_confirmatory_v2_fixture_primary_analysis_v1",
    }


__all__ = [
    "analyze_v2", "analyze_v2_fixture_results", "load_v2_raw",
    "recompute_v2_common_records", "validate_primary_lineage_inventory",
]
