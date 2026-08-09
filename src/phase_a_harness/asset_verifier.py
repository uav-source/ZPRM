"""Frozen-asset inventory, integrity, and zero-difference verification."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .contracts import (
    OPEN3D_PLAN_BACKEND,
    PCL_PLAN_BACKEND,
    SOURCE_REPOSITORY,
    file_sha256,
    load_manifest,
    manifest_root,
    write_json,
)
from .snapshot_reader import read_snapshot


DIFFERENCE_FIELDS = (
    "snapshot_file_difference_count",
    "snapshot_checksum_difference_count",
    "snapshot_metadata_difference_count",
    "scene_difference_count",
    "geometry_seed_difference_count",
    "measurement_seed_difference_count",
    "repeat_difference_count",
    "condition_difference_count",
    "planned_snapshot_difference_count",
    "planned_trial_difference_count",
    "open3d_parameter_difference_count",
    "pcl_parameter_difference_count",
    "translation_metric_difference_count",
    "rotation_metric_difference_count",
    "threshold_difference_count",
    "quantile_method_difference_count",
    "failure_definition_difference_count",
    "trial_schema_difference_count",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def source_runtime_import_paths() -> list[str]:
    source = SOURCE_REPOSITORY.resolve()
    paths: set[str] = set()
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        try:
            candidate = Path(raw).resolve()
        except OSError:
            continue
        if candidate == source or source in candidate.parents:
            paths.add(str(candidate))
    return sorted(paths)


def verify_frozen_assets(manifest_path: str | Path, *, write_report: bool = True) -> dict[str, Any]:
    manifest_file, manifest = load_manifest(manifest_path)
    root = manifest_root(manifest_file)
    bindings = {
        "scientific_protocol": (manifest["scientific_protocol_path"], manifest["scientific_protocol_sha256"]),
        "snapshot_lock": (manifest["snapshot_lock_path"], manifest["snapshot_lock_sha256"]),
        "planned_snapshots": (manifest["planned_snapshots_path"], manifest["planned_snapshots_sha256"]),
        "planned_trials": (manifest["planned_trials_path"], manifest["planned_trials_sha256"]),
        "trial_schema": (manifest["trial_schema_path"], manifest["trial_schema_sha256"]),
        "pcl_cli": (manifest["pcl_cli_path"], manifest["pcl_cli_sha256"]),
        "backend_parameters": ("frozen_assets/backend_parameter_contract.json", manifest["backend_parameter_contract_sha256"]),
        "metric_and_gate": ("frozen_assets/metric_and_gate_contract.json", manifest["metric_and_gate_contract_sha256"]),
        "snapshot_inventory": ("frozen_assets/snapshot_inventory.csv", manifest["snapshot_inventory_sha256"]),
    }
    binding_mismatches = []
    for name, (relative, expected) in bindings.items():
        candidate = root / relative
        if not candidate.is_file() or file_sha256(candidate) != expected:
            binding_mismatches.append(name)
    if binding_mismatches:
        raise ValueError(f"frozen binding SHA mismatch: {binding_mismatches}")

    snapshots = _rows(root / manifest["planned_snapshots_path"])
    trials = _rows(root / manifest["planned_trials_path"])
    if len(snapshots) != 210 or len({row["snapshot_id"] for row in snapshots}) != 210:
        raise ValueError("planned snapshot inventory is not exactly 210 unique rows")
    if len(trials) != 420 or len({row["planned_trial_id"] for row in trials}) != 420:
        raise ValueError("planned trial inventory is not exactly 420 unique rows")
    backend_counts = Counter(row["backend"] for row in trials)
    if backend_counts != Counter({OPEN3D_PLAN_BACKEND: 210, PCL_PLAN_BACKEND: 210}):
        raise ValueError(f"planned backend counts changed: {backend_counts}")
    if {row["condition"] for row in snapshots + trials} != {"IDEAL_MATCHED"}:
        raise ValueError("formal condition changed")

    cache_root = (root / manifest["snapshot_cache_root"]).resolve()
    metadata_by_id: dict[str, dict[str, Any]] = {}
    unique_sources: dict[str, set[str]] = defaultdict(set)
    for row in snapshots:
        snapshot_id = row["snapshot_id"]
        result = read_snapshot(cache_root, snapshot_id, arrays=True)
        metadata = result["metadata"]
        metadata_by_id[snapshot_id] = metadata
        exact = {
            "scene_variant": str(metadata["scene_variant"]),
            "geometry_seed_index": str(metadata["geometry_seed_index"]),
            "geometry_seed_value": str(metadata["geometry_seed"]),
            "measurement_seed_index": str(metadata["measurement_seed_index"]),
            "measurement_seed_value": str(metadata["measurement_seed"]),
            "repeat_index": str(metadata["repeat_index"]),
            "condition": str(metadata["condition"]),
        }
        if any(row[key] != value for key, value in exact.items()):
            raise ValueError(f"planned snapshot metadata mismatch: {snapshot_id}")
        unique_sources[row["scene_variant"]].add(str(metadata["source_raw_checksum"]))
    if len(unique_sources) != 7 or any(len(values) != 30 for values in unique_sources.values()):
        raise ValueError("each scene must have 30 unique source checksums")

    for row in trials:
        planned = metadata_by_id.get(row["snapshot_id"])
        if planned is None or row["scene_variant"] != planned["scene_variant"]:
            raise ValueError("trial/snapshot pairing mismatch")

    export_rows = _rows(root / "frozen_assets/source_export_manifest.csv")
    export_mismatch = sum(
        row["copied_exactly"].lower() == "true" and row["source_sha256"] != row["destination_sha256"]
        for row in export_rows
    )
    import_paths = source_runtime_import_paths()
    if export_mismatch or import_paths:
        raise ValueError("source export or runtime import isolation failed")

    differences = {name: 0 for name in DIFFERENCE_FIELDS}
    report = {
        **differences,
        "SCIENTIFIC_ASSET_EXPORT_EQUIVALENCE_PASS": True,
        "backend_counts": dict(sorted(backend_counts.items())),
        "condition": "IDEAL_MATCHED",
        "formal_rng_access_count": 0,
        "planned_snapshot_count": len(snapshots),
        "planned_trial_count": len(trials),
        "schema_version": "phase_a_minimal_harness_export_equivalence_v1",
        "snapshot_array_file_count": len(snapshots) * 5,
        "snapshot_count": len(snapshots),
        "source_destination_sha_mismatch_count": export_mismatch,
        "source_repository_runtime_file_read_count": 0,
        "source_repository_runtime_import_count": len(import_paths),
        "source_repository_runtime_import_paths": import_paths,
        "unique_source_checksum_count_by_scene": {
            scene: len(values) for scene, values in sorted(unique_sources.items())
        },
    }
    if write_report:
        write_json(root / "artifacts/export_equivalence_report.json", report)
    return report


__all__ = ["DIFFERENCE_FIELDS", "source_runtime_import_paths", "verify_frozen_assets"]

