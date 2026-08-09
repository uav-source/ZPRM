"""Read-only evidence and publisher change-scope checks."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


FROZEN_SCIENTIFIC_FIELDS = (
    "FULL_SYNTHETIC_DEVELOPMENT_PASS",
    "FULL_SYNTHETIC_PRIMARY_SCENE_EFFECT_PASS",
    "FULL_SYNTHETIC_CROSS_BACKEND_PASS",
    "FULL_SYNTHETIC_SCENE_RANK_STABILITY_PASS",
    "COMMON_ASSOCIATION_ANALYSIS_PASS",
    "REASSOCIATION_MECHANISM_SUPPORTED",
    "LOCAL_METRIC_INCREMENTAL_VALUE_PASS",
    "SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pre_repair_inventory(root: Path, inventory_path: Path) -> dict[str, Any]:
    with inventory_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_fields = ["relative_path", "size_bytes", "sha256", "semantic_role"]
    if not rows or list(rows[0]) != expected_fields:
        raise ValueError("pre-repair inventory fields differ from contract")
    missing: list[str] = []
    size_mismatch: list[str] = []
    sha_mismatch: list[str] = []
    mismatched_roles: dict[str, int] = {}
    for row in rows:
        path = root / row["relative_path"]
        if not path.is_file():
            missing.append(row["relative_path"])
            mismatched_roles[row["semantic_role"]] = mismatched_roles.get(row["semantic_role"], 0) + 1
            continue
        size_failed = path.stat().st_size != int(row["size_bytes"])
        sha_failed = file_sha256(path) != row["sha256"]
        if size_failed:
            size_mismatch.append(row["relative_path"])
        if sha_failed:
            sha_mismatch.append(row["relative_path"])
        if size_failed or sha_failed:
            mismatched_roles[row["semantic_role"]] = mismatched_roles.get(row["semantic_role"], 0) + 1
    role_counts: dict[str, int] = {}
    for row in rows:
        role = row["semantic_role"]
        role_counts[role] = role_counts.get(role, 0) + 1
    result = {
        "inventory_row_count": len(rows),
        "semantic_role_counts": role_counts,
        "missing_count": len(missing),
        "missing_paths": missing,
        "size_mismatch_count": len(size_mismatch),
        "size_mismatch_paths": size_mismatch,
        "sha256_mismatch_count": len(sha_mismatch),
        "sha256_mismatch_paths": sha_mismatch,
        "mismatch_count_by_semantic_role": mismatched_roles,
    }
    result["RAW_EVIDENCE_INTEGRITY_PASS"] = bool(
        len(rows) == 2113
        and role_counts.get("raw_trial_result") == 2100
        and role_counts.get("snapshot_inventory") == 1
        and role_counts.get("trial_inventory") == 1
        and not missing
        and not size_mismatch
        and not sha_mismatch
    )
    return result


def publisher_only_change_scope(
    *,
    inventory_verification: Mapping[str, Any],
    source_decision: Mapping[str, Any],
    published_decision: Mapping[str, Any],
    changed_paths: Sequence[str],
    matplotlib_version: str,
) -> dict[str, Any]:
    frozen_fields = {
        name: {
            "before": source_decision.get(name),
            "after": published_decision.get(name),
            "identical": source_decision.get(name) == published_decision.get(name),
        }
        for name in FROZEN_SCIENTIFIC_FIELDS
    }
    changed = set(changed_paths)
    publisher_paths = {
        path for path in changed if path == "src/phase_a_harness/full_synthetic_publisher.py"
    }
    backend_files = {
        "src/phase_a_harness/open3d_backend.py",
        "src/phase_a_harness/pcl_backend.py",
        "src/phase_a_harness/full_synthetic_backend_execution.py",
        "bin/pcl_point_to_plane_cli",
    }
    snapshot_files = {
        "src/phase_a_harness/full_synthetic_snapshot_builder.py",
        "src/phase_a_harness/phase_b_snapshot_assets.py",
        "src/phase_a_harness/snapshot_reader.py",
    }
    metric_files = {
        "src/phase_a_harness/metrics.py",
        "src/phase_a_harness/rotation_metrics.py",
        "src/phase_a_harness/backend_phase_a_metrics.py",
        "src/phase_a_harness/common_association_analysis.py",
        "src/phase_a_harness/full_synthetic_statistics.py",
    }
    model_files = {"src/phase_a_harness/local_metric_models.py"}
    protocol_files = {
        "frozen_assets/full_synthetic_development_protocol_v1.json",
        "frozen_assets/full_synthetic_development_manifest_v1.json",
        "frozen_assets/full_synthetic_development_snapshot_lock_v1.json",
        "configs/zero_perturbation/development_v1.yaml",
    }
    raw_role_counts = inventory_verification.get("semantic_role_counts", {})
    inventory_mismatch = int(inventory_verification.get("missing_count", 0)) + int(
        inventory_verification.get("size_mismatch_count", 0)
    ) + int(inventory_verification.get("sha256_mismatch_count", 0))
    mismatched_roles = inventory_verification.get("mismatch_count_by_semantic_role", {})
    return {
        "schema_version": "publisher_only_change_scope_v1",
        "matplotlib_version": matplotlib_version,
        "trial_rerun_count": 0,
        "snapshot_regeneration_count": 0,
        "raw_result_change_count": int(mismatched_roles.get("raw_trial_result", 0)) + int(mismatched_roles.get("raw_result_inventory", 0)),
        "primary_analysis_change_count": int(mismatched_roles.get("frozen_primary_analysis", 0)),
        "independent_verification_change_count": int(mismatched_roles.get("frozen_independent_verification", 0)),
        "run_manifest_scientific_change_count": int(mismatched_roles.get("formal_run_manifest", 0)),
        "backend_code_change_count": len(changed & backend_files),
        "snapshot_code_change_count": len(changed & snapshot_files),
        "metric_code_change_count": len(changed & metric_files),
        "gate_change_count": len(changed & protocol_files),
        "threshold_change_count": len(changed & protocol_files),
        "model_change_count": len(changed & model_files),
        "publisher_code_change_count": len(publisher_paths),
        "publisher_changed_paths": sorted(publisher_paths),
        "frozen_scientific_fields": frozen_fields,
        "frozen_scientific_field_difference_count": sum(
            not row["identical"] for row in frozen_fields.values()
        ),
        "publisher_repair_scope_pass": bool(
            inventory_mismatch == 0
            and len(publisher_paths) > 0
            and not (changed & backend_files)
            and not (changed & snapshot_files)
            and not (changed & metric_files)
            and not (changed & model_files)
            and not (changed & protocol_files)
            and all(row["identical"] for row in frozen_fields.values())
        ),
    }


__all__ = [
    "FROZEN_SCIENTIFIC_FIELDS",
    "file_sha256",
    "publisher_only_change_scope",
    "verify_pre_repair_inventory",
]
