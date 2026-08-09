"""Independent raw-result recomputation for the formal Phase A decision."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import load_manifest, manifest_root, write_json
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    file_sha256,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .snapshot_reader import read_snapshot


def independently_verify_formal_phase_a(
    *, manifest_path: str | Path, run_dir: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    manifest_file, manifest = load_manifest(manifest_path, require_authorized=True)
    root = manifest_root(manifest_file)
    with (root / manifest["planned_trials_path"]).open("r", encoding="utf-8", newline="") as stream:
        plan_rows = list(csv.DictReader(stream))
    expected_ids = {row["planned_trial_id"] for row in plan_rows}
    directory = Path(run_dir).resolve()
    raw_manifest = json.loads((directory / "raw_result_manifest.json").read_text(encoding="utf-8"))
    entries = raw_manifest["results"]
    actual_ids = set(entries)
    corrupt = 0
    results: list[dict[str, Any]] = []
    for trial_id in sorted(actual_ids & expected_ids):
        entry = entries[trial_id]
        path = directory / "raw_results" / entry["path"]
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            corrupt += 1
            continue
        value = validate_phase_a_trial_result_strict(load_json_strict(path))
        if value["planned_trial_id"] != trial_id:
            corrupt += 1
            continue
        results.append(value)

    checksum_mismatch = 0
    metadata: dict[str, dict[str, Any]] = {}
    for row in results:
        if row["snapshot_id"] not in metadata:
            metadata[row["snapshot_id"]] = read_snapshot(
                root / manifest["snapshot_cache_root"], row["snapshot_id"], arrays=False
            )["metadata"]
        expected = metadata[row["snapshot_id"]]
        checksum_mismatch += any(
            row[key] != expected[value]
            for key, value in (
                ("snapshot_checksum", "snapshot_checksum"),
                ("source_checksum", "source_raw_checksum"),
                ("target_checksum", "target_raw_checksum"),
                ("reference_pose_checksum", "reference_pose_raw_checksum"),
            )
        )

    backend_summary: list[dict[str, Any]] = []
    scene_summary: list[dict[str, Any]] = []
    backend_passes: list[bool] = []
    for backend, label in ((OPEN3D_BACKEND, "Open3D"), (PCL_BACKEND, "PCL")):
        chosen = [row for row in results if row["backend"] == backend]
        translations = np.asarray(
            [float(row["translation_update_m"]) for row in chosen if row["translation_update_m"] is not None],
            dtype=np.float64,
        )
        rotations = np.asarray(
            [float(row["rotation_update_rad"]) for row in chosen if row["rotation_update_rad"] is not None],
            dtype=np.float64,
        )
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in chosen:
            if row["translation_update_m"] is not None:
                grouped[row["scene_variant"]].append(float(row["translation_update_m"]))
        scene_pass = len(grouped) == 7
        for scene in sorted(grouped):
            median = float(np.median(grouped[scene]))
            passed = len(grouped[scene]) == 30 and median <= 0.001
            scene_pass = scene_pass and passed
            scene_summary.append(
                {
                    "backend": label,
                    "median_translation_update_m": median,
                    "scene_gate_pass": passed,
                    "scene_variant": scene,
                    "trial_count": len(grouped[scene]),
                }
            )
        failures = sum(bool(row["solver_failure"]) for row in chosen)
        nonfinite = sum(not bool(row["finite_output"]) for row in chosen)
        fraction = float(np.count_nonzero(translations <= 0.001) / translations.size) if translations.size else 0.0
        translation_q95 = float(np.quantile(translations, 0.95, method="linear")) if translations.size else None
        rotation_q95 = float(np.quantile(rotations, 0.95, method="linear")) if rotations.size else None
        passed = bool(
            len(chosen) == 210
            and failures == 0
            and nonfinite == 0
            and translation_q95 is not None
            and translation_q95 <= 0.001
            and rotation_q95 is not None
            and rotation_q95 <= 0.00017453292519943296
            and fraction >= 0.95
            and scene_pass
        )
        backend_passes.append(passed)
        backend_summary.append(
            {
                "backend": label,
                "backend_schema_name": backend,
                "finite_translation_count": int(translations.size),
                "fraction_translation_le_1mm": fraction,
                "nonfinite_output_count": nonfinite,
                "qualified": passed,
                "rotation_max_rad": float(np.max(rotations)) if rotations.size else None,
                "rotation_median_rad": float(np.median(rotations)) if rotations.size else None,
                "rotation_q95_rad": rotation_q95,
                "solver_failure_count": failures,
                "translation_max_m": float(np.max(translations)) if translations.size else None,
                "translation_median_m": float(np.median(translations)) if translations.size else None,
                "translation_q95_m": translation_q95,
                "trial_count": len(chosen),
            }
        )

    snapshots = {row["snapshot_id"] for row in results}
    open3d_count = sum(row["backend"] == OPEN3D_BACKEND for row in results)
    pcl_count = sum(row["backend"] == PCL_BACKEND for row in results)
    native_count = len(results) - open3d_count - pcl_count
    extra_files = {
        path.name for path in (directory / "raw_results").glob("*.json")
    } - {entry["path"] for entry in entries.values()}
    completeness = {
        "completed_snapshot_count": len(snapshots),
        "completed_trial_count": len(results),
        "corrupt_trial_count": corrupt,
        "duplicate_trial_count": 0,
        "extra_snapshot_count": 0,
        "extra_trial_count": len(actual_ids - expected_ids) + len(extra_files),
        "missing_snapshot_count": 210 - len(snapshots),
        "missing_trial_count": len(expected_ids - actual_ids),
        "native_trial_count": native_count,
        "open3d_trial_count": open3d_count,
        "pcl_trial_count": pcl_count,
    }
    complete = bool(
        completeness["completed_snapshot_count"] == 210
        and completeness["completed_trial_count"] == 420
        and all(completeness[key] == 0 for key in (
            "corrupt_trial_count", "duplicate_trial_count", "extra_snapshot_count",
            "extra_trial_count", "missing_snapshot_count", "missing_trial_count", "native_trial_count"
        ))
        and checksum_mismatch == 0
    )
    two = bool(complete and all(backend_passes))
    report = {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": None,
        "backend_input_checksum_mismatch_count": checksum_mismatch,
        "backend_summary": backend_summary,
        "completeness": completeness,
        "final_decision": {
            "BACKEND_PHASE_A_COMPLETE": complete,
            "CONFIRMATORY_AUTHORIZED": False,
            "DAY1_SCIENTIFIC_VALIDATION_PASS": two if complete else "NOT_EVALUATED",
            "FULL_DEVELOPMENT_AUTHORIZED": False,
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
            "PHASE_B_PROTOCOL_DESIGN_AUTHORIZED": two,
            "PHASE_B_RUN_AUTHORIZED": False,
            "REAL_DATA_AUTHORIZED": False,
            "TWO_INDEPENDENT_BACKENDS_QUALIFIED": two,
        },
        "quantile_method": "linear",
        "scene_backend_summary": scene_summary,
        "schema_version": "phase_a_minimal_harness_independent_verification_v1",
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def comparison_difference_count(primary: dict[str, Any], independent: dict[str, Any]) -> int:
    keys = (
        "backend_input_checksum_mismatch_count",
        "backend_summary",
        "completeness",
        "final_decision",
        "quantile_method",
        "scene_backend_summary",
    )
    return sum(primary[key] != independent[key] for key in keys)


__all__ = ["comparison_difference_count", "independently_verify_formal_phase_a"]

