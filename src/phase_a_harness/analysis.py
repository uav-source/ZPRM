"""Primary raw-result analysis for the frozen 420-trial formal experiment."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import OPEN3D_PLAN_BACKEND, PCL_PLAN_BACKEND, load_manifest, manifest_root, write_json
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    file_sha256,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .snapshot_reader import read_snapshot


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(array.size),
        "maximum": float(np.max(array)) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "q95_linear": float(np.quantile(array, 0.95, method="linear")) if array.size else None,
    }


def load_and_validate_raw(
    *, manifest_path: str | Path, run_dir: str | Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    manifest_file, manifest = load_manifest(manifest_path, require_authorized=True)
    root = manifest_root(manifest_file)
    trials = _csv(root / manifest["planned_trials_path"])
    expected = {row["planned_trial_id"]: row for row in trials}
    directory = Path(run_dir).resolve()
    raw_manifest = json.loads((directory / "raw_result_manifest.json").read_text(encoding="utf-8"))
    entries = raw_manifest.get("results", {})
    if type(entries) is not dict:
        raise ValueError("raw result manifest results must be an object")
    rows: list[dict[str, Any]] = []
    for trial_id in sorted(set(entries) & set(expected)):
        entry = entries[trial_id]
        path = directory / "raw_results" / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"corrupt raw result SHA: {trial_id}")
        payload = validate_phase_a_trial_result_strict(load_json_strict(path))
        if payload["planned_trial_id"] != trial_id:
            raise ValueError(f"raw trial identity mismatch: {trial_id}")
        rows.append(payload)
    return manifest, rows, trials


def analyze_formal_phase_a(
    *, manifest_path: str | Path, run_dir: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    manifest_file, manifest = load_manifest(manifest_path, require_authorized=True)
    root = manifest_root(manifest_file)
    _, rows, trials = load_and_validate_raw(manifest_path=manifest_file, run_dir=run_dir)
    expected = {row["planned_trial_id"]: row for row in trials}
    actual = {row["planned_trial_id"]: row for row in rows}
    raw_files = list((Path(run_dir).resolve() / "raw_results").glob("*.json"))
    expected_result_names = {
        json.loads((Path(run_dir).resolve() / "raw_result_manifest.json").read_text(encoding="utf-8"))["results"][trial_id]["path"]
        for trial_id in actual
    }
    extra_files = [path.name for path in raw_files if path.name not in expected_result_names]
    missing_ids = sorted(set(expected) - set(actual))
    extra_ids = sorted(set(actual) - set(expected))
    pairing_mismatches = 0
    cache_root = root / manifest["snapshot_cache_root"]
    metadata_cache: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        metadata = metadata_cache.setdefault(
            row["snapshot_id"], read_snapshot(cache_root, row["snapshot_id"], arrays=False)["metadata"]
        )
        if any(
            row[name] != expected_value
            for name, expected_value in {
                "snapshot_checksum": metadata["snapshot_checksum"],
                "source_checksum": metadata["source_raw_checksum"],
                "target_checksum": metadata["target_raw_checksum"],
                "reference_pose_checksum": metadata["reference_pose_raw_checksum"],
            }.items()
        ):
            pairing_mismatches += 1

    backend_summaries: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    qualification: dict[str, bool] = {}
    for backend, label in ((OPEN3D_BACKEND, "Open3D"), (PCL_BACKEND, "PCL")):
        selected = [row for row in rows if row["backend"] == backend]
        translation = [float(row["translation_update_m"]) for row in selected if row["translation_update_m"] is not None]
        rotation = [float(row["rotation_update_rad"]) for row in selected if row["rotation_update_rad"] is not None]
        translation_stats = _stats(translation)
        rotation_stats = _stats(rotation)
        failures = sum(bool(row["solver_failure"]) for row in selected)
        nonfinite = sum(not bool(row["finite_output"]) for row in selected)
        within = float(np.mean(np.asarray(translation) <= 0.001)) if translation else 0.0
        by_scene: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            if row["translation_update_m"] is not None:
                by_scene[row["scene_variant"]].append(float(row["translation_update_m"]))
        scene_gate = len(by_scene) == 7
        for scene in sorted(by_scene):
            values = by_scene[scene]
            median = float(np.median(values))
            passed = len(values) == 30 and median <= 0.001
            scene_gate = scene_gate and passed
            scene_summaries.append(
                {
                    "backend": label,
                    "median_translation_update_m": median,
                    "scene_gate_pass": passed,
                    "scene_variant": scene,
                    "trial_count": len(values),
                }
            )
        qualified = bool(
            len(selected) == 210
            and failures == 0
            and nonfinite == 0
            and translation_stats["q95_linear"] is not None
            and translation_stats["q95_linear"] <= 0.001
            and rotation_stats["q95_linear"] is not None
            and rotation_stats["q95_linear"] <= 0.00017453292519943296
            and within >= 0.95
            and scene_gate
        )
        qualification[label] = qualified
        backend_summaries.append(
            {
                "backend": label,
                "backend_schema_name": backend,
                "finite_translation_count": len(translation),
                "fraction_translation_le_1mm": within,
                "nonfinite_output_count": nonfinite,
                "qualified": qualified,
                "rotation_max_rad": rotation_stats["maximum"],
                "rotation_median_rad": rotation_stats["median"],
                "rotation_q95_rad": rotation_stats["q95_linear"],
                "solver_failure_count": failures,
                "translation_max_m": translation_stats["maximum"],
                "translation_median_m": translation_stats["median"],
                "translation_q95_m": translation_stats["q95_linear"],
                "trial_count": len(selected),
            }
        )

    snapshot_ids = {row["snapshot_id"] for row in rows}
    open3d_count = sum(row["backend"] == OPEN3D_BACKEND for row in rows)
    pcl_count = sum(row["backend"] == PCL_BACKEND for row in rows)
    native_count = len(rows) - open3d_count - pcl_count
    complete = bool(
        len(rows) == 420
        and len(snapshot_ids) == 210
        and open3d_count == pcl_count == 210
        and native_count == 0
        and not missing_ids
        and not extra_ids
        and not extra_files
        and pairing_mismatches == 0
    )
    two_qualified = bool(complete and qualification == {"Open3D": True, "PCL": True})
    day1: bool | str = two_qualified if complete else "NOT_EVALUATED"
    decision = {
        "BACKEND_PHASE_A_COMPLETE": complete,
        "CONFIRMATORY_AUTHORIZED": False,
        "DAY1_SCIENTIFIC_VALIDATION_PASS": day1,
        "FULL_DEVELOPMENT_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PHASE_B_PROTOCOL_DESIGN_AUTHORIZED": two_qualified,
        "PHASE_B_RUN_AUTHORIZED": False,
        "REAL_DATA_AUTHORIZED": False,
        "TWO_INDEPENDENT_BACKENDS_QUALIFIED": two_qualified,
    }
    failure_inventory = [
        {"count": count, "failure_classification": name}
        for name, count in sorted(Counter(row["failure_classification"] for row in rows).items())
    ]
    report = {
        "backend_input_checksum_mismatch_count": pairing_mismatches,
        "backend_summary": backend_summaries,
        "completeness": {
            "completed_snapshot_count": len(snapshot_ids),
            "completed_trial_count": len(rows),
            "corrupt_trial_count": 0,
            "duplicate_trial_count": 0,
            "extra_snapshot_count": 0,
            "extra_trial_count": len(extra_ids) + len(extra_files),
            "missing_snapshot_count": 210 - len(snapshot_ids),
            "missing_trial_count": len(missing_ids),
            "native_trial_count": native_count,
            "open3d_trial_count": open3d_count,
            "pcl_trial_count": pcl_count,
        },
        "failure_inventory": failure_inventory,
        "final_decision": decision,
        "quantile_method": "linear",
        "run_id": "phase-a-minimal-harness-formal-v1",
        "scene_backend_summary": scene_summaries,
        "schema_version": "phase_a_minimal_harness_primary_analysis_v1",
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


__all__ = ["analyze_formal_phase_a", "load_and_validate_raw"]

