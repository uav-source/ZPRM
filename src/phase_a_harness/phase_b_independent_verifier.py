"""Independent raw-JSON recomputation of every Phase B decision gate.

This module intentionally does not call :func:`analyze_phase_b_signal` or its
calculation helpers.  It opens the formal manifest, plan, snapshot metadata and
raw trial JSON again, then performs a second implementation of the complete
calculation.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata, spearmanr

from .contracts import file_sha256, load_manifest, manifest_root, write_json
from .phase_a_trial_result_schema import OPEN3D_BACKEND, PCL_BACKEND, load_json_strict
from .phase_b_analysis import (
    BACKEND_LABELS,
    BACKENDS,
    CONDITIONS,
    GEOMETRY_SEEDS,
    RICH_SCENE,
    SCENES,
    WEAK_CANDIDATES,
)
from .phase_b_trial_result import validate_phase_b_trial_result_strict


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _seed(row: Mapping[str, str]) -> int:
    value = row.get("geometry_seed", row.get("geometry_seed_value"))
    if value is None or value == "":
        raise ValueError("geometry seed missing from Phase B plan")
    return int(value)


def _backend(value: str) -> str:
    aliases = {
        OPEN3D_BACKEND: OPEN3D_BACKEND,
        PCL_BACKEND: PCL_BACKEND,
        "pcl_iterative_closest_point_with_normals": PCL_BACKEND,
    }
    if value not in aliases:
        raise ValueError("invalid Phase B planned backend")
    return aliases[value]


def _meta(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    return metadata.get(key.replace("_checksum", "_raw_checksum"))


def _independent_load(
    manifest_path: str | Path, run_dir: str | Path
) -> tuple[Path, dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    manifest_file, manifest = load_manifest(manifest_path, require_authorized=True)
    root = manifest_root(manifest_file)
    trials = _csv(root / manifest["planned_trials_path"])
    snapshots = _csv(root / manifest["planned_snapshots_path"])
    plan = {row["planned_trial_id"]: row for row in trials}
    if len(plan) != len(trials):
        raise ValueError("duplicate frozen Phase B trial IDs")
    run = Path(run_dir).resolve()
    raw_manifest = json.loads((run / "raw_result_manifest.json").read_text(encoding="utf-8"))
    if (
        set(raw_manifest) != {"results", "run_id", "schema_version"}
        or raw_manifest.get("run_id") != manifest.get("formal_run_id", "phase-b-signal-v1")
        or raw_manifest.get("schema_version") != "phase_b_scene_signal_raw_result_manifest_v1"
    ):
        raise ValueError("raw result manifest identity mismatch")
    entries = raw_manifest.get("results")
    if type(entries) is not dict:
        raise ValueError("raw result manifest results must be an object")
    raw_root = (run / "raw_results").resolve()
    valid: list[dict[str, Any]] = []
    corrupt: list[str] = []
    names: list[str] = []
    for entry in entries.values():
        if type(entry) is not dict or not isinstance(entry.get("path"), str):
            continue
        candidate = (raw_root / entry["path"]).resolve()
        if candidate.parent == raw_root:
            names.append(candidate.name)
    for trial_id in sorted(set(plan) & set(entries)):
        try:
            entry = entries[trial_id]
            if type(entry) is not dict or set(entry) != {"planned_trial_id", "path", "sha256"}:
                raise ValueError("bad raw manifest entry")
            if entry["planned_trial_id"] != trial_id:
                raise ValueError("entry trial ID mismatch")
            candidate = (raw_root / entry["path"]).resolve()
            if candidate.parent != raw_root:
                raise ValueError("result path escape")
            if not candidate.is_file() or file_sha256(candidate) != entry["sha256"]:
                raise ValueError("result checksum mismatch")
            result = validate_phase_b_trial_result_strict(load_json_strict(candidate))
            expected = plan[trial_id]
            identities = (
                result["planned_trial_id"] == trial_id,
                result["snapshot_id"] == expected["snapshot_id"],
                result["scene_variant"] == expected["scene_variant"],
                result["condition"] == expected["condition"],
                result["backend"] == _backend(expected["backend"]),
            )
            if not all(identities):
                raise ValueError("trial identity mismatch")
            valid.append(result)
        except (KeyError, OSError, TypeError, ValueError):
            corrupt.append(trial_id)
    extra_snapshot_ids: set[str] = set()
    for trial_id in sorted(set(entries) - set(plan)):
        try:
            entry = entries[trial_id]
            candidate = (raw_root / entry["path"]).resolve()
            if candidate.parent != raw_root or not candidate.is_file() or file_sha256(candidate) != entry["sha256"]:
                continue
            payload = validate_phase_b_trial_result_strict(load_json_strict(candidate))
            if payload["planned_trial_id"] == trial_id:
                extra_snapshot_ids.add(payload["snapshot_id"])
        except (KeyError, OSError, TypeError, ValueError):
            continue
    raw_files = {path.name for path in raw_root.glob("*.json") if path.is_file()}
    counts = Counter(names)
    defects = {
        "corrupt": sorted(corrupt),
        "duplicate": sum(max(0, count - 1) for count in counts.values()),
        "extra": len(set(entries) - set(plan)) + len(raw_files - set(names)),
        "extra_snapshot_ids": sorted(extra_snapshot_ids),
        "missing": sorted(set(plan) - set(entries)),
    }
    return root, manifest, trials, snapshots, valid, defects


def independently_verify_phase_b_signal(
    *, manifest_path: str | Path, run_dir: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Re-read formal evidence and independently recompute the Phase B result."""

    root, manifest, trials, snapshots, results, defects = _independent_load(manifest_path, run_dir)
    plan = {row["planned_trial_id"]: row for row in trials}
    expected_snapshots = {row["snapshot_id"] for row in snapshots}
    result_snapshots = {row["snapshot_id"] for row in results}

    by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_snapshot[result["snapshot_id"]].append(result)
    pairing_rows: list[dict[str, Any]] = []
    mismatch = 0
    for snapshot_id in sorted(expected_snapshots):
        chosen = by_snapshot.get(snapshot_id, [])
        metadata: Mapping[str, Any] = {}
        readable = False
        try:
            candidate = json.loads(
                (root / manifest["snapshot_cache_root"] / snapshot_id / "metadata.json").read_text(encoding="utf-8")
            )
            if type(candidate) is dict:
                metadata = candidate
                readable = True
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        checksum_keys = ("snapshot_checksum", "source_checksum", "target_checksum", "reference_pose_checksum")
        two_backends = len(chosen) == 2 and Counter(row["backend"] for row in chosen) == Counter(BACKENDS)
        same_pair = bool(chosen and all(len({row[key] for row in chosen}) == 1 for key in checksum_keys))
        same_metadata = bool(
            readable and all(all(row[key] == _meta(metadata, key) for row in chosen) for key in checksum_keys)
        )
        passed = two_backends and same_pair and same_metadata
        mismatch += not passed
        pairing_rows.append(
            {
                "backend_count": len(chosen),
                "input_checksum_match": passed,
                "metadata_readable": readable,
                "open3d_count": sum(row["backend"] == OPEN3D_BACKEND for row in chosen),
                "pcl_count": sum(row["backend"] == PCL_BACKEND for row in chosen),
                "snapshot_id": snapshot_id,
            }
        )

    raw_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(result["condition"], result["backend"], result["scene_variant"])].append(result)
        raw_rows.append(
            {
                "backend": BACKEND_LABELS[result["backend"]],
                "backend_schema_name": result["backend"],
                "condition": result["condition"],
                "geometry_seed": _seed(plan[result["planned_trial_id"]]),
                "planned_trial_id": result["planned_trial_id"],
                "rotation_update_rad": result["rotation_update_rad"],
                "scene_variant": result["scene_variant"],
                "snapshot_id": result["snapshot_id"],
                "translation_update_m": result["translation_update_m"],
            }
        )
    raw_rows.sort(key=lambda row: (
        CONDITIONS.index(row["condition"]), BACKENDS.index(row["backend_schema_name"]),
        SCENES.index(row["scene_variant"]), GEOMETRY_SEEDS.index(row["geometry_seed"]),
    ))
    scene_rows: list[dict[str, Any]] = []
    medians: dict[tuple[str, str, str], float | None] = {}
    for condition in CONDITIONS:
        for backend in BACKENDS:
            for scene in SCENES:
                chosen = groups.get((condition, backend, scene), [])
                translations = np.asarray(
                    [float(row["translation_update_m"]) for row in chosen if row["translation_update_m"] is not None],
                    dtype=float,
                )
                rotations = np.asarray(
                    [float(row["rotation_update_rad"]) for row in chosen if row["rotation_update_rad"] is not None],
                    dtype=float,
                )
                median = float(np.median(translations)) if translations.size else None
                medians[(condition, backend, scene)] = median
                scene_rows.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "rotation_max_rad": float(np.max(rotations)) if rotations.size else None,
                        "rotation_median_rad": float(np.median(rotations)) if rotations.size else None,
                        "rotation_min_rad": float(np.min(rotations)) if rotations.size else None,
                        "scene_variant": scene,
                        "translation_max_m": float(np.max(translations)) if translations.size else None,
                        "translation_median_m": median,
                        "translation_min_m": float(np.min(translations)) if translations.size else None,
                        "trial_count": len(chosen),
                    }
                )

    scene_ranking: list[dict[str, Any]] = []
    scene_group_passes: list[bool] = []
    for condition in CONDITIONS:
        for backend in BACKENDS:
            values = [medians[(condition, backend, scene)] for scene in SCENES]
            complete = all(value is not None and len(groups[(condition, backend, scene)]) == 3 for scene, value in zip(SCENES, values))
            ranks = rankdata(np.asarray(values, dtype=float), method="average") if complete else np.full(7, np.nan)
            count = Counter(values) if complete else Counter()
            mapping = dict(zip(SCENES, ranks))
            rich_pass = complete and float(mapping[RICH_SCENE]) <= 2.0
            weak_rank_pass = complete and max(float(mapping[scene]) for scene in WEAK_CANDIDATES) >= 5.0
            scene_group_passes.append(bool(rich_pass and weak_rank_pass))
            for scene, value, rank in zip(SCENES, values, ranks):
                scene_ranking.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "is_tied": bool(complete and count[value] > 1),
                        "rich_lowest_two_pass": bool(rich_pass),
                        "scene_variant": scene,
                        "translation_median_m": value,
                        "translation_rank_ascending_average_ties": float(rank) if math.isfinite(float(rank)) else None,
                        "weak_highest_three_pass": bool(weak_rank_pass),
                    }
                )

    cross_rows: list[dict[str, Any]] = []
    rho_values: list[float | None] = []
    for condition in CONDITIONS:
        left = [medians[(condition, OPEN3D_BACKEND, scene)] for scene in SCENES]
        right = [medians[(condition, PCL_BACKEND, scene)] for scene in SCENES]
        rho = None
        pvalue = None
        if all(value is not None for value in left + right):
            computed = spearmanr(left, right)
            rho = float(computed.statistic) if math.isfinite(float(computed.statistic)) else None
            pvalue = float(computed.pvalue) if math.isfinite(float(computed.pvalue)) else None
        rho_values.append(rho)
        cross_rows.append(
            {
                "condition": condition,
                "rho": rho,
                "rho_ge_0_50": bool(rho is not None and rho >= 0.50),
                "rho_ge_0_70": bool(rho is not None and rho >= 0.70),
                "scene_count": 7,
                "spearman_pvalue_descriptive": pvalue,
            }
        )
    cross_pass = bool(
        all(value is not None and value >= 0.50 for value in rho_values)
        and any(value is not None and value >= 0.70 for value in rho_values)
    )

    weak_rows: list[dict[str, Any]] = []
    selected: dict[str, str | None] = {}
    condition_effect_passes: list[bool] = []
    for condition in CONDITIONS:
        candidate_passes: dict[str, bool] = {}
        condition_rows: list[dict[str, Any]] = []
        for candidate in WEAK_CANDIDATES:
            per_backend: list[bool] = []
            candidate_rows: list[dict[str, Any]] = []
            for backend in BACKENDS:
                rich = medians[(condition, backend, RICH_SCENE)]
                weak = medians[(condition, backend, candidate)]
                difference = float(weak - rich) if rich is not None and weak is not None else None
                ratio = float(weak / rich) if rich is not None and weak is not None and rich > 1.0e-12 else None
                passed = bool(ratio is not None and ratio >= 2.0 and difference is not None and difference >= 0.001)
                per_backend.append(passed)
                candidate_rows.append(
                    {
                        "absolute_difference_m": difference,
                        "backend": BACKEND_LABELS[backend],
                        "backend_pass": passed,
                        "backend_schema_name": backend,
                        "candidate_common_pass": False,
                        "condition": condition,
                        "rich_median_m": rich,
                        "rich_ratio_denominator_valid": bool(rich is not None and rich > 1.0e-12),
                        "selected_common_weak_scene": False,
                        "weak_median_m": weak,
                        "weak_rich_ratio": ratio,
                        "weak_scene": candidate,
                    }
                )
            candidate_passes[candidate] = all(per_backend)
            for row in candidate_rows:
                row["candidate_common_pass"] = candidate_passes[candidate]
            condition_rows.extend(candidate_rows)
        chosen = next((candidate for candidate in WEAK_CANDIDATES if candidate_passes[candidate]), None)
        selected[condition] = chosen
        condition_effect_passes.append(chosen is not None)
        for row in condition_rows:
            row["selected_common_weak_scene"] = row["weak_scene"] == chosen
        weak_rows.extend(condition_rows)

    raw_lookup = {
        (row["condition"], row["backend_schema_name"], row["scene_variant"], row["geometry_seed"]): row["translation_update_m"]
        for row in raw_rows
    }
    consistency_rows: list[dict[str, Any]] = []
    backend_consistency: list[bool] = []
    for condition in CONDITIONS:
        weak_scene = selected[condition]
        for backend in BACKENDS:
            subset: list[dict[str, Any]] = []
            for seed in GEOMETRY_SEEDS:
                rich = raw_lookup.get((condition, backend, RICH_SCENE, seed))
                weak = raw_lookup.get((condition, backend, weak_scene, seed)) if weak_scene else None
                subset.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "direction_pass": bool(rich is not None and weak is not None and weak > rich),
                        "geometry_seed": seed,
                        "rich_translation_update_m": rich,
                        "weak_scene": weak_scene,
                        "weak_translation_update_m": weak,
                    }
                )
            successes = sum(row["direction_pass"] for row in subset)
            passed = bool(weak_scene is not None and successes >= 2)
            backend_consistency.append(passed)
            for row in subset:
                row["backend_consistency_pass"] = passed
                row["direction_success_count"] = successes
            consistency_rows.extend(subset)

    open3d_count = sum(row["backend"] == OPEN3D_BACKEND for row in results)
    pcl_count = sum(row["backend"] == PCL_BACKEND for row in results)
    native_count = len(results) - open3d_count - pcl_count
    completeness = {
        "completed_snapshot_count": len(result_snapshots & expected_snapshots),
        "completed_trial_count": len(results),
        "corrupt_trial_count": len(defects["corrupt"]),
        "duplicate_trial_count": defects["duplicate"],
        "extra_snapshot_count": len(
            (result_snapshots - expected_snapshots) | set(defects["extra_snapshot_ids"])
        ),
        "extra_trial_count": defects["extra"],
        "missing_snapshot_count": len(expected_snapshots - result_snapshots),
        "missing_trial_count": len(defects["missing"]),
        "native_trial_count": native_count,
        "open3d_trial_count": open3d_count,
        "pcl_trial_count": pcl_count,
        "planned_snapshot_count": len(snapshots),
        "planned_trial_count": len(trials),
    }
    failures = {backend: sum(row["backend"] == backend and bool(row["solver_failure"]) for row in results) for backend in BACKENDS}
    nonfinite = {backend: sum(row["backend"] == backend and not bool(row["finite_output"]) for row in results) for backend in BACKENDS}
    engineering = bool(
        completeness["planned_snapshot_count"] == completeness["completed_snapshot_count"] == 42
        and completeness["planned_trial_count"] == completeness["completed_trial_count"] == 84
        and open3d_count == pcl_count == 42
        and all(completeness[key] == 0 for key in (
            "corrupt_trial_count", "duplicate_trial_count", "extra_snapshot_count", "extra_trial_count",
            "missing_snapshot_count", "missing_trial_count", "native_trial_count",
        ))
        and mismatch == 0
        and all(failures[backend] == nonfinite[backend] == 0 for backend in BACKENDS)
    )
    gates = {
        "PHASE_B_CROSS_BACKEND_RANKING_PASS": cross_pass,
        "PHASE_B_ENGINEERING_PASS": engineering,
        "PHASE_B_GEOMETRY_SEED_CONSISTENCY_PASS": all(backend_consistency),
        "PHASE_B_SCENE_RANKING_PASS": all(scene_group_passes),
        "PHASE_B_WEAK_RICH_EFFECT_PASS": all(condition_effect_passes),
    }
    incomplete = bool(
        (completeness["missing_trial_count"] or completeness["missing_snapshot_count"])
        and completeness["corrupt_trial_count"] == 0
        and completeness["duplicate_trial_count"] == 0
        and completeness["extra_trial_count"] == 0
        and completeness["extra_snapshot_count"] == 0
    )
    signal: bool | str = all(gates.values())
    if incomplete:
        signal = "NOT_EVALUATED"
    decision = {
        **gates,
        "CONFIRMATORY_AUTHORIZED": False,
        "FULL_SYNTHETIC_DEVELOPMENT_PROTOCOL_DESIGN_AUTHORIZED": signal is True,
        "FULL_SYNTHETIC_DEVELOPMENT_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PHASE_B_SIGNAL_PASS": signal,
        "REAL_DATA_AUTHORIZED": False,
    }

    failure_inventory: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for backend in BACKENDS:
        chosen = [row for row in results if row["backend"] == backend]
        counts = Counter(row["failure_classification"] for row in chosen)
        for classification in sorted(counts):
            failure_inventory.append(
                {
                    "backend": BACKEND_LABELS[backend],
                    "backend_schema_name": backend,
                    "failure_classification": classification,
                    "failure_count": counts[classification],
                    "backend_nonfinite_output_count": nonfinite[backend],
                    "backend_solver_failure_count": failures[backend],
                }
            )
        runtimes = np.asarray([float(row["runtime_ms"]) for row in chosen], dtype=float)
        runtime_rows.append(
            {
                "backend": BACKEND_LABELS[backend],
                "backend_schema_name": backend,
                "maximum_runtime_ms": float(np.max(runtimes)) if runtimes.size else None,
                "median_runtime_ms": float(np.median(runtimes)) if runtimes.size else None,
                "q95_runtime_ms_linear": float(np.quantile(runtimes, 0.95, method="linear")) if runtimes.size else None,
                "trial_count": int(runtimes.size),
            }
        )
    report = {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": None,
        "backend_input_checksum_mismatch_count": mismatch,
        "backend_input_pairing": pairing_rows,
        "completeness": completeness,
        "cross_backend_ranking": cross_rows,
        "failure_inventory": failure_inventory,
        "final_decision": decision,
        "gate_summary": gates,
        "geometry_seed_consistency": consistency_rows,
        "geometry_seed_raw_values": raw_rows,
        "nonfinite_output_count_by_backend": {
            BACKEND_LABELS[backend]: nonfinite[backend] for backend in BACKENDS
        },
        "quantile_method": "linear",
        "run_id": manifest.get("formal_run_id", "phase-b-signal-v1"),
        "runtime_summary": runtime_rows,
        "scene_condition_backend_summary": scene_rows,
        "scene_ranking": scene_ranking,
        "schema_version": "phase_b_scene_signal_independent_verification_v1",
        "selected_common_weak_scene": selected,
        "solver_failure_count_by_backend": {
            BACKEND_LABELS[backend]: failures[backend] for backend in BACKENDS
        },
        "weak_rich_effect": weak_rows,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


COMPARISON_KEYS = (
    "backend_input_checksum_mismatch_count",
    "backend_input_pairing",
    "completeness",
    "cross_backend_ranking",
    "failure_inventory",
    "final_decision",
    "gate_summary",
    "geometry_seed_consistency",
    "geometry_seed_raw_values",
    "nonfinite_output_count_by_backend",
    "quantile_method",
    "run_id",
    "runtime_summary",
    "scene_condition_backend_summary",
    "scene_ranking",
    "selected_common_weak_scene",
    "solver_failure_count_by_backend",
    "weak_rich_effect",
)


def phase_b_analysis_verifier_difference_count(
    primary: Mapping[str, Any], independent: Mapping[str, Any]
) -> int:
    """Count differing scientific/engineering report sections."""

    return sum(primary.get(key) != independent.get(key) for key in COMPARISON_KEYS)


__all__ = [
    "COMPARISON_KEYS",
    "independently_verify_phase_b_signal",
    "phase_b_analysis_verifier_difference_count",
]
