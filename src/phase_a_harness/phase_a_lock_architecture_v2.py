"""Layered Phase A lock architecture audit, snapshot binding, and verification."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .phase_a_formal_run_lock_v2 import SNAPSHOT_LOCK_RELATIVE
from .phase_a_scientific_lock_v2 import LEGACY_PROTOCOL_LOCK_RELATIVE
from .phase_a_trial_result_schema import canonical_json_sha256, file_sha256


ARTIFACT_RELATIVE = Path(
    "artifacts/current/zero_perturbation_phase_a_lock_architecture_v2"
)
SNAPSHOT_BINDING_NAME = "snapshot_lock_binding_v2.json"
STAGE0_ARTIFACT_RELATIVE = Path(
    "artifacts/current/zero_perturbation_backend_phase_a_v1_2_stage0"
)

CLASSIFICATIONS = (
    "SCIENTIFIC_PROTOCOL",
    "SNAPSHOT_PROVENANCE",
    "EXECUTION_IMPLEMENTATION",
    "FORMAL_RUN_AUTHORIZATION",
    "LEGACY_DUPLICATE",
    "DOCUMENT_METADATA",
)

REQUIRED_ARTIFACT_FILES = (
    "legacy_lock_field_classification.csv",
    "lock_dependency_graph.json",
    "scientific_protocol_projection_diff.json",
    "lock_architecture_v2_scientific_diff.json",
    "phase_a_scientific_protocol_lock_v2.json",
    "snapshot_lock_binding_v2.json",
    "phase_a_execution_implementation_lock_v2.json",
    "phase_a_formal_run_lock_v2.json",
    "execution_fixture_audit_report.md",
    "execution_fixture_final_decision.json",
    "resume_equivalence_report.json",
    "analysis_verifier_comparison.json",
    "tamper_rejection_report.json",
    "dry_run_report.json",
    "implementation_manifest.json",
    "lock_architecture_v2_report.md",
    "final_decision.json",
    "run_manifest.json",
    "artifact_verification.json",
    "SHA256SUMS",
)


class LockArchitectureError(RuntimeError):
    """The layered graph, snapshot binding, or artifact is invalid."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LockArchitectureError(f"cannot parse JSON: {path}") from error
    if type(value) is not dict:
        raise LockArchitectureError(f"JSON root is not an object: {path}")
    return value


def legacy_json_paths(value: Any) -> list[tuple[str, Any]]:
    """Return every non-root object, array, and leaf path exactly once."""

    result: list[tuple[str, Any]] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                child_path = f"{path}.{key}"
                result.append((child_path, child))
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                child_path = f"{path}[{index}]"
                result.append((child_path, child))
                walk(child, child_path)

    walk(value, "$")
    return result


def classify_legacy_json_path(path: str) -> tuple[str, str, str, bool, str]:
    """Uniquely classify a legacy lock path using the frozen manual audit rules."""

    if path == "$.implementation_sha256":
        return (
            "LEGACY_DUPLICATE",
            "duplicates $.implementation.implementation_sha256",
            "OMIT_DUPLICATE",
            False,
            "HIGH",
        )
    if path.startswith("$.implementation") or path.startswith("$.pcl_cli_binary"):
        return (
            "EXECUTION_IMPLEMENTATION",
            "binds executable source, binary, or implementation provenance",
            "Execution Implementation Lock v2",
            path not in {"$.implementation.files.stage1_engine.sha256", "$.implementation.files.stage1_runner.sha256"},
            "HIGH" if "stage1_" in path else "MEDIUM",
        )
    if path in {
        "$.planned_snapshot_count",
        "$.planned_snapshots_path",
        "$.planned_snapshots_sha256",
        "$.planned_trial_count",
        "$.planned_trials_path",
        "$.planned_trials_sha256",
        "$.protocol_path",
        "$.protocol_sha256",
        "$.scientific_base_protocol_sha256",
    }:
        return (
            "SCIENTIFIC_PROTOCOL",
            "identifies the frozen scientific protocol or enumerated scientific plan",
            "Scientific Protocol Lock v2",
            True,
            "LOW",
        )
    if path == "$.SEED_CONTINUATION_AFTER_PRE_BACKEND_ABORT_JUSTIFIED":
        return (
            "SNAPSHOT_PROVENANCE",
            "records the pre-backend Stage-0 seed-continuation provenance decision",
            "Snapshot Lock binding",
            True,
            "LOW",
        )
    if path in {
        "$.PHASE_A_V1_2_PROTOCOL_LOCK_PASS",
        "$.stage0_snapshot_build_authorized",
        "$.stage1_backend_run_authorized",
    }:
        return (
            "FORMAL_RUN_AUTHORIZATION",
            "is a historical pass or execution authorization flag",
            "Formal Run Lock v2",
            path != "$.stage1_backend_run_authorized",
            "HIGH",
        )
    if path in {
        "$.lock_payload_sha256",
        "$.protocol_document_path",
        "$.protocol_document_sha256",
        "$.protocol_lock_commit",
        "$.protocol_lock_tag",
        "$.schema_version",
    }:
        return (
            "DOCUMENT_METADATA",
            "describes the legacy document, schema, archive, or aggregate payload",
            "Historical source metadata only",
            True,
            "LOW",
        )
    raise LockArchitectureError(f"unclassified legacy JSON path: {path}")


def _value_summary(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return rendered if len(rendered) <= 180 else rendered[:177] + "..."


def legacy_field_classification(root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repository = Path(root).resolve()
    legacy = _json(repository / LEGACY_PROTOCOL_LOCK_RELATIVE)
    rows: list[dict[str, Any]] = []
    for path, value in legacy_json_paths(legacy):
        classification, reason, target, retained, risk = classify_legacy_json_path(path)
        rows.append(
            {
                "json_path": path,
                "value_summary": _value_summary(value),
                "classification": classification,
                "reason": reason,
                "target_v2_lock": target,
                "retained_unchanged": str(bool(retained)).lower(),
                "legacy_conflict_risk": risk,
            }
        )
    counts = Counter(row["classification"] for row in rows)
    inventory = {
        "LEGACY_LOCK_FIELD_CLASSIFICATION_PASS": (
            len(rows) == len({row["json_path"] for row in rows})
            and set(counts).issubset(CLASSIFICATIONS)
        ),
        "classification_counts": {name: counts.get(name, 0) for name in CLASSIFICATIONS},
        "legacy_field_total_count": len(rows),
        "machine_audit_field_count": len(legacy_json_paths(legacy)),
        "manual_audit_field_count": len(rows),
        "multiply_classified_field_count": 0,
        "schema_version": "phase_a_legacy_lock_field_classification_v2",
        "unclassified_field_count": 0,
    }
    return rows, inventory


def lock_dependency_graph() -> dict[str, Any]:
    nodes = [
        "Scientific Protocol Lock v2",
        "Snapshot Lock",
        "Execution Implementation Lock v2",
        "Formal Run Lock v2",
    ]
    edges = [
        {"from": nodes[0], "to": nodes[3]},
        {"from": nodes[1], "to": nodes[3]},
        {"from": nodes[2], "to": nodes[3]},
    ]
    pairs = [(edge["from"], edge["to"]) for edge in edges]
    duplicate_count = len(pairs) - len(set(pairs))
    adjacency = {node: [] for node in nodes}
    for source, target in pairs:
        adjacency[source].append(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def cycle(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(cycle(child) for child in adjacency[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    acyclic = not any(cycle(node) for node in nodes)
    return {
        "LOCK_GRAPH_ACYCLIC": acyclic,
        "LOCK_GRAPH_DUPLICATE_BINDING_COUNT": duplicate_count,
        "edges": edges,
        "nodes": nodes,
        "schema_version": "phase_a_lock_dependency_graph_v2",
    }


def build_snapshot_binding(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    stage0 = repository / STAGE0_ARTIFACT_RELATIVE
    snapshot_path = repository / SNAPSHOT_LOCK_RELATIVE
    snapshot = _json(snapshot_path)
    independent = _json(stage0 / "independent_verification.json")
    decision = _json(stage0 / "final_decision.json")
    manifest_path = stage0 / "run_manifest.json"
    manifest = _json(manifest_path)
    result: dict[str, Any] = {
        "schema_version": "phase_a_snapshot_lock_binding_v2",
        "snapshot_lock_path": SNAPSHOT_LOCK_RELATIVE.as_posix(),
        "snapshot_lock_file_sha256": file_sha256(snapshot_path),
        "snapshot_lock_payload_sha256": snapshot["lock_payload_sha256"],
        "stage0_cache_root": snapshot["snapshot_cache_root"],
        "stage0_manifest_path": (STAGE0_ARTIFACT_RELATIVE / "run_manifest.json").as_posix(),
        "stage0_manifest_sha256": file_sha256(manifest_path),
        "planned_snapshot_count": int(manifest["planned_snapshot_count"]),
        "complete_snapshot_count": int(manifest["complete_snapshot_count"]),
        "missing_snapshot_count": int(independent["missing_snapshot_count"]),
        "extra_snapshot_count": int(independent["extra_snapshot_count"]),
        "duplicate_snapshot_count": int(independent["duplicate_snapshot_count"]),
        "corrupt_snapshot_count": int(independent["corrupt_snapshot_count"]),
        "lineage_pass": bool(decision["SNAPSHOT_PROVENANCE_LINEAGE_PASS"]),
        "closure_pass": bool(decision["FLOAT32_RECONSTRUCTION_CLOSURE_PASS"]),
        "canonical_cache_pass": bool(decision["CANONICAL_INPUT_CACHE_PASS"]),
        "diversity_pass": bool(decision["IDEAL_MATCHED_SNAPSHOT_DIVERSITY_PASS"]),
    }
    required = (
        result["planned_snapshot_count"] == 210
        and result["complete_snapshot_count"] == 210
        and all(result[name] == 0 for name in (
            "missing_snapshot_count", "extra_snapshot_count",
            "duplicate_snapshot_count", "corrupt_snapshot_count"
        ))
        and all(result[name] is True for name in (
            "lineage_pass", "closure_pass", "canonical_cache_pass", "diversity_pass"
        ))
        and snapshot.get("PHASE_A_V1_2_STAGE0_PASS") is True
        and independent.get("verification_pass") is True
    )
    result["SNAPSHOT_LOCK_V2_BINDING_PASS"] = required
    result["binding_payload_sha256"] = canonical_json_sha256(result)
    return result


def validate_snapshot_binding(
    value: Mapping[str, Any], *, root: str | Path, expected_snapshot_path: str | None = None,
    expected_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    required_keys = {
        "schema_version", "snapshot_lock_path", "snapshot_lock_file_sha256",
        "snapshot_lock_payload_sha256", "stage0_cache_root", "stage0_manifest_path",
        "stage0_manifest_sha256", "planned_snapshot_count", "complete_snapshot_count",
        "missing_snapshot_count", "extra_snapshot_count", "duplicate_snapshot_count",
        "corrupt_snapshot_count", "lineage_pass", "closure_pass", "canonical_cache_pass",
        "diversity_pass", "SNAPSHOT_LOCK_V2_BINDING_PASS", "binding_payload_sha256",
    }
    if set(value) != required_keys:
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: binding fields")
    payload = dict(value)
    stored = payload.pop("binding_payload_sha256")
    if stored != canonical_json_sha256(payload):
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: binding payload SHA")
    if value["schema_version"] != "phase_a_snapshot_lock_binding_v2":
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: binding schema")
    if expected_snapshot_path is not None and value["snapshot_lock_path"] != expected_snapshot_path:
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: bound snapshot path")
    if expected_snapshot_sha256 is not None and value["snapshot_lock_file_sha256"] != expected_snapshot_sha256:
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: bound snapshot SHA")
    snapshot_path = repository / str(value["snapshot_lock_path"])
    manifest_path = repository / str(value["stage0_manifest_path"])
    if file_sha256(snapshot_path) != value["snapshot_lock_file_sha256"]:
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: snapshot file SHA")
    snapshot = _json(snapshot_path)
    snapshot_payload = dict(snapshot)
    snapshot_stored = snapshot_payload.pop("lock_payload_sha256")
    if snapshot_stored != canonical_json_sha256(snapshot_payload):
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: snapshot payload SHA")
    if snapshot_stored != value["snapshot_lock_payload_sha256"]:
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: snapshot payload binding")
    if snapshot["snapshot_cache_root"] != value["stage0_cache_root"]:
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: cache root")
    if file_sha256(manifest_path) != value["stage0_manifest_sha256"]:
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: manifest SHA")
    if value["planned_snapshot_count"] != 210 or value["complete_snapshot_count"] != 210:
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: snapshot count")
    zero_fields = ("missing_snapshot_count", "extra_snapshot_count", "duplicate_snapshot_count", "corrupt_snapshot_count")
    if any(value[name] != 0 for name in zero_fields):
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: snapshot inventory")
    pass_fields = ("lineage_pass", "closure_pass", "canonical_cache_pass", "diversity_pass", "SNAPSHOT_LOCK_V2_BINDING_PASS")
    if any(value[name] is not True for name in pass_fields):
        raise LockArchitectureError("SNAPSHOT_LOCK_INVALID: Stage-0 gate")
    return dict(value)


def verify_sha256sums(directory: str | Path) -> tuple[list[str], list[str]]:
    root = Path(directory).resolve()
    missing: list[str] = []
    mismatched: list[str] = []
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        return ["SHA256SUMS"], []
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        if not path.is_file():
            missing.append(relative.strip())
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatched.append(relative.strip())
    return missing, mismatched


def verify_lock_architecture_artifact(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).resolve()
    missing = sorted(name for name in REQUIRED_ARTIFACT_FILES if not (root / name).is_file())
    sha_missing, sha_mismatch = verify_sha256sums(root)
    semantic: list[str] = []
    if not missing:
        decision = _json(root / "final_decision.json")
        required_true = (
            "LEGACY_LOCK_FIELD_CLASSIFICATION_PASS",
            "LOCK_GRAPH_ACYCLIC",
            "SCIENTIFIC_PROTOCOL_PROJECTION_PASS",
            "SNAPSHOT_LOCK_V2_BINDING_PASS",
            "EXECUTION_FIXTURE_AUDIT_PASS",
            "EXECUTION_IMPLEMENTATION_LOCK_PASS",
            "FORMAL_RUN_LOCK_PASS",
            "LOCK_ARCHITECTURE_V2_TAMPER_REJECTION_PASS",
            "FORMAL_DRY_RUN_PASS",
            "PHASE_A_LOCK_ARCHITECTURE_V2_PASS",
        )
        semantic.extend(name for name in required_true if decision.get(name) is not True)
        for name in (
            "LOCK_GRAPH_DUPLICATE_BINDING_COUNT",
            "SCIENTIFIC_LOCK_IMPLEMENTATION_FIELD_COUNT",
            "IMPLEMENTATION_LOCK_SCIENTIFIC_FIELD_COUNT",
            "FORMAL_RUN_LOCK_DUPLICATED_CONTRACT_FIELD_COUNT",
            "FORMAL_SEED_ACCESS_COUNT",
            "FORMAL_BACKEND_EXECUTION_COUNT",
            "FORMAL_TRIAL_RESULT_COUNT",
            "NATIVE_EXECUTION_COUNT",
        ):
            if decision.get(name) != 0:
                semantic.append(name)
    return {
        "LOCK_ARCHITECTURE_V2_ARTIFACT_VERIFICATION_PASS": not (missing or sha_missing or sha_mismatch or semantic),
        "missing_files": missing,
        "schema_version": "phase_a_lock_architecture_v2_artifact_verification_v1",
        "semantic_failures": sorted(semantic),
        "sha256_mismatches": sorted(sha_mismatch),
        "sha256_missing_files": sorted(sha_missing),
    }


__all__ = [
    "ARTIFACT_RELATIVE",
    "CLASSIFICATIONS",
    "LockArchitectureError",
    "build_snapshot_binding",
    "classify_legacy_json_path",
    "legacy_field_classification",
    "legacy_json_paths",
    "lock_dependency_graph",
    "validate_snapshot_binding",
    "verify_lock_architecture_artifact",
    "verify_sha256sums",
]
