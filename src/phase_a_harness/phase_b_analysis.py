"""Primary analysis for the frozen Phase B scene-signal experiment.

The analysis is deliberately raw-result driven.  It validates every referenced
JSON result, joins it to the frozen trial plan, checks the shared snapshot
inputs, and then evaluates the five pre-declared Phase B gates without fitting
or tuning any parameter.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import rankdata, spearmanr

from .contracts import file_sha256, load_manifest, manifest_root, write_json
from .phase_a_trial_result_schema import OPEN3D_BACKEND, PCL_BACKEND, load_json_strict
from .phase_b_trial_result import validate_phase_b_trial_result_strict


CONDITIONS = ("INDEPENDENT_NOISE_FREE", "FULL_NOISE")
SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
GEOMETRY_SEEDS = (1850310744, 1957656152, 1334931069)
BACKENDS = (OPEN3D_BACKEND, PCL_BACKEND)
BACKEND_LABELS = {OPEN3D_BACKEND: "Open3D", PCL_BACKEND: "PCL"}
PLAN_BACKEND_TO_RESULT = {
    OPEN3D_BACKEND: OPEN3D_BACKEND,
    "pcl_iterative_closest_point_with_normals": PCL_BACKEND,
    PCL_BACKEND: PCL_BACKEND,
}
WEAK_CANDIDATES = ("LONG_CORRIDOR", "PARALLEL_WALLS")
RICH_SCENE = "GEOMETRY_RICH_ROOM"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _plan_seed(row: Mapping[str, str]) -> int:
    for key in ("geometry_seed", "geometry_seed_value"):
        if key in row and row[key] != "":
            return int(row[key])
    raise ValueError("planned trial omits geometry seed")


def _plan_trial_id(row: Mapping[str, str]) -> str:
    value = row.get("planned_trial_id")
    if not value:
        raise ValueError("planned trial omits planned_trial_id")
    return value


def _result_backend(plan_backend: str) -> str:
    try:
        return PLAN_BACKEND_TO_RESULT[plan_backend]
    except KeyError as error:
        raise ValueError(f"unauthorized planned backend: {plan_backend}") from error


@dataclass(frozen=True)
class PhaseBRawAudit:
    """Validated raw rows plus independently countable integrity defects."""

    manifest_file: Path
    manifest: dict[str, Any]
    root: Path
    planned_trials: list[dict[str, str]]
    planned_snapshots: list[dict[str, str]]
    rows: list[dict[str, Any]]
    corrupt_trial_ids: tuple[str, ...]
    duplicate_trial_count: int
    extra_snapshot_ids: tuple[str, ...]
    extra_trial_count: int
    missing_trial_ids: tuple[str, ...]


def load_and_validate_phase_b_raw(
    *, manifest_path: str | Path, run_dir: str | Path
) -> PhaseBRawAudit:
    """Strictly load each raw JSON listed by the formal result manifest."""

    manifest_file, manifest = load_manifest(manifest_path, require_authorized=True)
    root = manifest_root(manifest_file)
    planned_trials = _read_csv(root / manifest["planned_trials_path"])
    planned_snapshots = _read_csv(root / manifest["planned_snapshots_path"])
    expected = {_plan_trial_id(row): row for row in planned_trials}
    if len(expected) != len(planned_trials):
        raise ValueError("planned trial IDs are not unique")

    directory = Path(run_dir).resolve()
    raw_manifest = json.loads((directory / "raw_result_manifest.json").read_text(encoding="utf-8"))
    if (
        set(raw_manifest) != {"results", "run_id", "schema_version"}
        or raw_manifest.get("run_id") != manifest.get("formal_run_id", "phase-b-signal-v1")
        or raw_manifest.get("schema_version") != "phase_b_scene_signal_raw_result_manifest_v1"
    ):
        raise ValueError("raw result manifest identity mismatch")
    entries = raw_manifest.get("results")
    if type(entries) is not dict:
        raise ValueError("raw result manifest results must be an object")
    raw_root = (directory / "raw_results").resolve()
    referenced_names: list[str] = []
    for entry in entries.values():
        if type(entry) is not dict or not isinstance(entry.get("path"), str):
            continue
        candidate = (raw_root / entry["path"]).resolve()
        if candidate.parent == raw_root:
            referenced_names.append(candidate.name)
    corrupt: list[str] = []
    rows: list[dict[str, Any]] = []
    for trial_id in sorted(set(entries) & set(expected)):
        entry = entries[trial_id]
        try:
            if type(entry) is not dict or set(entry) != {"planned_trial_id", "path", "sha256"}:
                raise ValueError("raw manifest entry schema mismatch")
            if entry["planned_trial_id"] != trial_id or not isinstance(entry["path"], str):
                raise ValueError("raw manifest identity mismatch")
            path = (raw_root / entry["path"]).resolve()
            if path.parent != raw_root:
                raise ValueError("raw result path escapes raw_results")
            if not path.is_file() or file_sha256(path) != entry["sha256"]:
                raise ValueError("raw result file checksum mismatch")
            payload = validate_phase_b_trial_result_strict(load_json_strict(path))
            plan = expected[trial_id]
            expected_identity = {
                "planned_trial_id": trial_id,
                "snapshot_id": plan["snapshot_id"],
                "scene_variant": plan["scene_variant"],
                "condition": plan["condition"],
                "backend": _result_backend(plan["backend"]),
            }
            if any(payload[key] != value for key, value in expected_identity.items()):
                raise ValueError("raw result does not match frozen trial identity")
            rows.append(payload)
        except (KeyError, OSError, TypeError, ValueError):
            corrupt.append(trial_id)

    extra_snapshot_ids: set[str] = set()
    for trial_id in sorted(set(entries) - set(expected)):
        try:
            entry = entries[trial_id]
            path = (raw_root / entry["path"]).resolve()
            if path.parent != raw_root or not path.is_file() or file_sha256(path) != entry["sha256"]:
                continue
            payload = validate_phase_b_trial_result_strict(load_json_strict(path))
            if payload["planned_trial_id"] == trial_id:
                extra_snapshot_ids.add(payload["snapshot_id"])
        except (KeyError, OSError, TypeError, ValueError):
            continue
    referenced_counts = Counter(referenced_names)
    duplicate_count = sum(max(0, count - 1) for count in referenced_counts.values())
    raw_files = {path.name for path in raw_root.glob("*.json") if path.is_file()}
    unreferenced_files = raw_files - set(referenced_names)
    extra_entry_ids = set(entries) - set(expected)
    missing_ids = set(expected) - set(entries)
    return PhaseBRawAudit(
        manifest_file=manifest_file,
        manifest=manifest,
        root=root,
        planned_trials=planned_trials,
        planned_snapshots=planned_snapshots,
        rows=rows,
        corrupt_trial_ids=tuple(sorted(corrupt)),
        duplicate_trial_count=duplicate_count,
        extra_snapshot_ids=tuple(sorted(extra_snapshot_ids)),
        extra_trial_count=len(extra_entry_ids) + len(unreferenced_files),
        missing_trial_ids=tuple(sorted(missing_ids)),
    )


def _summary(values: Iterable[float]) -> tuple[float | None, float | None, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return None, None, None
    return float(np.min(array)), float(np.median(array)), float(np.max(array))


def _metadata_checksum(metadata: Mapping[str, Any], name: str) -> Any:
    for key in (name, name.replace("_checksum", "_raw_checksum")):
        if key in metadata:
            return metadata[key]
    return None


def _pairing_rows(audit: PhaseBRawAudit) -> tuple[list[dict[str, Any]], int]:
    by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit.rows:
        by_snapshot[row["snapshot_id"]].append(row)
    rows: list[dict[str, Any]] = []
    mismatch_count = 0
    expected_snapshot_ids = [row["snapshot_id"] for row in audit.planned_snapshots]
    for snapshot_id in sorted(set(expected_snapshot_ids)):
        selected = by_snapshot.get(snapshot_id, [])
        metadata_path = audit.root / audit.manifest["snapshot_cache_root"] / snapshot_id / "metadata.json"
        metadata: Mapping[str, Any] = {}
        metadata_readable = False
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            if type(value) is dict:
                metadata = value
                metadata_readable = True
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        exact_backends = Counter(row["backend"] for row in selected) == Counter(BACKENDS)
        checksums = ("snapshot_checksum", "source_checksum", "target_checksum", "reference_pose_checksum")
        pair_equal = bool(
            selected
            and all(len({row[key] for row in selected}) == 1 for key in checksums)
        )
        metadata_equal = bool(
            metadata_readable
            and all(
                all(row[key] == _metadata_checksum(metadata, key) for row in selected)
                for key in checksums
            )
        )
        passed = len(selected) == 2 and exact_backends and pair_equal and metadata_equal
        mismatch_count += not passed
        rows.append(
            {
                "backend_count": len(selected),
                "input_checksum_match": passed,
                "metadata_readable": metadata_readable,
                "open3d_count": sum(row["backend"] == OPEN3D_BACKEND for row in selected),
                "pcl_count": sum(row["backend"] == PCL_BACKEND for row in selected),
                "snapshot_id": snapshot_id,
            }
        )
    return rows, mismatch_count


def _raw_and_scene_summaries(
    rows: list[dict[str, Any]], plan_by_id: Mapping[str, Mapping[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["backend"], row["scene_variant"])
        grouped[key].append(row)
        raw.append(
            {
                "backend": BACKEND_LABELS[row["backend"]],
                "backend_schema_name": row["backend"],
                "condition": row["condition"],
                "geometry_seed": _plan_seed(plan_by_id[row["planned_trial_id"]]),
                "planned_trial_id": row["planned_trial_id"],
                "rotation_update_rad": row["rotation_update_rad"],
                "scene_variant": row["scene_variant"],
                "snapshot_id": row["snapshot_id"],
                "translation_update_m": row["translation_update_m"],
            }
        )
    raw.sort(key=lambda item: (CONDITIONS.index(item["condition"]), BACKENDS.index(item["backend_schema_name"]), SCENES.index(item["scene_variant"]), GEOMETRY_SEEDS.index(item["geometry_seed"])))

    summaries: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for backend in BACKENDS:
            for scene in SCENES:
                selected = grouped.get((condition, backend, scene), [])
                translation = [float(row["translation_update_m"]) for row in selected if row["translation_update_m"] is not None]
                rotation = [float(row["rotation_update_rad"]) for row in selected if row["rotation_update_rad"] is not None]
                t_min, t_median, t_max = _summary(translation)
                r_min, r_median, r_max = _summary(rotation)
                summaries.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "rotation_max_rad": r_max,
                        "rotation_median_rad": r_median,
                        "rotation_min_rad": r_min,
                        "scene_variant": scene,
                        "translation_max_m": t_max,
                        "translation_median_m": t_median,
                        "translation_min_m": t_min,
                        "trial_count": len(selected),
                    }
                )
    return raw, summaries


def _rankings(
    scene_summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, bool]:
    lookup = {
        (row["condition"], row["backend_schema_name"], row["scene_variant"]): row
        for row in scene_summaries
    }
    ranking_rows: list[dict[str, Any]] = []
    backend_scene_gate: list[bool] = []
    for condition in CONDITIONS:
        for backend in BACKENDS:
            selected = [lookup[(condition, backend, scene)] for scene in SCENES]
            medians = [row["translation_median_m"] for row in selected]
            valid = all(row["trial_count"] == 3 and value is not None for row, value in zip(selected, medians))
            ranks = rankdata(np.asarray(medians, dtype=float), method="average") if valid else np.full(7, np.nan)
            counts = Counter(medians) if valid else Counter()
            rank_by_scene = dict(zip(SCENES, ranks))
            rich_pass = valid and float(rank_by_scene[RICH_SCENE]) <= 2.0
            weak_rank_pass = valid and max(float(rank_by_scene[scene]) for scene in WEAK_CANDIDATES) >= 5.0
            backend_scene_gate.append(bool(rich_pass and weak_rank_pass))
            for scene, median, rank in zip(SCENES, medians, ranks):
                ranking_rows.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "is_tied": bool(valid and counts[median] > 1),
                        "rich_lowest_two_pass": bool(rich_pass),
                        "scene_variant": scene,
                        "translation_median_m": median,
                        "translation_rank_ascending_average_ties": float(rank) if math.isfinite(float(rank)) else None,
                        "weak_highest_three_pass": bool(weak_rank_pass),
                    }
                )

    cross_rows: list[dict[str, Any]] = []
    rhos: list[float | None] = []
    for condition in CONDITIONS:
        open_values = [lookup[(condition, OPEN3D_BACKEND, scene)]["translation_median_m"] for scene in SCENES]
        pcl_values = [lookup[(condition, PCL_BACKEND, scene)]["translation_median_m"] for scene in SCENES]
        valid = all(value is not None for value in open_values + pcl_values)
        rho: float | None = None
        pvalue: float | None = None
        if valid:
            statistic = spearmanr(open_values, pcl_values)
            if math.isfinite(float(statistic.statistic)):
                rho = float(statistic.statistic)
            if math.isfinite(float(statistic.pvalue)):
                pvalue = float(statistic.pvalue)
        rhos.append(rho)
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
    cross_pass = bool(all(rho is not None and rho >= 0.50 for rho in rhos) and any(rho is not None and rho >= 0.70 for rho in rhos))
    return ranking_rows, cross_rows, all(backend_scene_gate), cross_pass


def _weak_rich(
    scene_summaries: list[dict[str, Any]], raw_values: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str | None], list[dict[str, Any]], bool, bool]:
    medians = {
        (row["condition"], row["backend_schema_name"], row["scene_variant"]): row["translation_median_m"]
        for row in scene_summaries
    }
    rows: list[dict[str, Any]] = []
    selected: dict[str, str | None] = {}
    condition_passes: list[bool] = []
    for condition in CONDITIONS:
        common_pass: dict[str, bool] = {}
        temporary: list[dict[str, Any]] = []
        for candidate in WEAK_CANDIDATES:
            backend_passes: list[bool] = []
            candidate_rows: list[dict[str, Any]] = []
            for backend in BACKENDS:
                rich = medians.get((condition, backend, RICH_SCENE))
                weak = medians.get((condition, backend, candidate))
                difference = None if rich is None or weak is None else float(weak - rich)
                ratio = None if rich is None or weak is None or rich <= 1.0e-12 else float(weak / rich)
                passed = bool(ratio is not None and ratio >= 2.0 and difference is not None and difference >= 0.001)
                backend_passes.append(passed)
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
            common_pass[candidate] = all(backend_passes)
            for row in candidate_rows:
                row["candidate_common_pass"] = common_pass[candidate]
            temporary.extend(candidate_rows)
        chosen = next((candidate for candidate in WEAK_CANDIDATES if common_pass[candidate]), None)
        selected[condition] = chosen
        condition_passes.append(chosen is not None)
        for row in temporary:
            row["selected_common_weak_scene"] = row["weak_scene"] == chosen
        rows.extend(temporary)

    raw_lookup = {
        (row["condition"], row["backend_schema_name"], row["scene_variant"], row["geometry_seed"]): row["translation_update_m"]
        for row in raw_values
    }
    consistency_rows: list[dict[str, Any]] = []
    consistency_backend_passes: list[bool] = []
    for condition in CONDITIONS:
        weak_scene = selected[condition]
        for backend in BACKENDS:
            comparison_rows: list[dict[str, Any]] = []
            for seed in GEOMETRY_SEEDS:
                rich = raw_lookup.get((condition, backend, RICH_SCENE, seed))
                weak = raw_lookup.get((condition, backend, weak_scene, seed)) if weak_scene else None
                direction_pass = bool(rich is not None and weak is not None and weak > rich)
                comparison_rows.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "direction_pass": direction_pass,
                        "geometry_seed": seed,
                        "rich_translation_update_m": rich,
                        "weak_scene": weak_scene,
                        "weak_translation_update_m": weak,
                    }
                )
            successes = sum(row["direction_pass"] for row in comparison_rows)
            passed = bool(weak_scene is not None and len(comparison_rows) == 3 and successes >= 2)
            consistency_backend_passes.append(passed)
            for row in comparison_rows:
                row["backend_consistency_pass"] = passed
                row["direction_success_count"] = successes
            consistency_rows.extend(comparison_rows)
    return rows, selected, consistency_rows, all(condition_passes), all(consistency_backend_passes)


def analyze_phase_b_signal(
    *, manifest_path: str | Path, run_dir: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Evaluate the complete frozen Phase B contract from formal raw JSON."""

    audit = load_and_validate_phase_b_raw(manifest_path=manifest_path, run_dir=run_dir)
    plan_by_id = {_plan_trial_id(row): row for row in audit.planned_trials}
    pairing, checksum_mismatches = _pairing_rows(audit)
    raw_values, scene_summaries = _raw_and_scene_summaries(audit.rows, plan_by_id)
    scene_rankings, cross_rankings, scene_ranking_pass, cross_ranking_pass = _rankings(scene_summaries)
    weak_effect, selected_weak, seed_consistency, weak_pass, seed_pass = _weak_rich(scene_summaries, raw_values)

    expected_snapshot_ids = {row["snapshot_id"] for row in audit.planned_snapshots}
    actual_snapshot_ids = {row["snapshot_id"] for row in audit.rows}
    open3d_count = sum(row["backend"] == OPEN3D_BACKEND for row in audit.rows)
    pcl_count = sum(row["backend"] == PCL_BACKEND for row in audit.rows)
    native_count = len(audit.rows) - open3d_count - pcl_count
    failures = {backend: sum(row["backend"] == backend and bool(row["solver_failure"]) for row in audit.rows) for backend in BACKENDS}
    nonfinite = {backend: sum(row["backend"] == backend and not bool(row["finite_output"]) for row in audit.rows) for backend in BACKENDS}
    completeness = {
        "completed_snapshot_count": len(actual_snapshot_ids & expected_snapshot_ids),
        "completed_trial_count": len(audit.rows),
        "corrupt_trial_count": len(audit.corrupt_trial_ids),
        "duplicate_trial_count": audit.duplicate_trial_count,
        "extra_snapshot_count": len((actual_snapshot_ids - expected_snapshot_ids) | set(audit.extra_snapshot_ids)),
        "extra_trial_count": audit.extra_trial_count,
        "missing_snapshot_count": len(expected_snapshot_ids - actual_snapshot_ids),
        "missing_trial_count": len(audit.missing_trial_ids),
        "native_trial_count": native_count,
        "open3d_trial_count": open3d_count,
        "pcl_trial_count": pcl_count,
        "planned_snapshot_count": len(audit.planned_snapshots),
        "planned_trial_count": len(audit.planned_trials),
    }
    integrity_zero = all(
        completeness[name] == 0
        for name in (
            "corrupt_trial_count", "duplicate_trial_count", "extra_snapshot_count",
            "extra_trial_count", "missing_snapshot_count", "missing_trial_count", "native_trial_count",
        )
    )
    engineering_pass = bool(
        completeness["planned_snapshot_count"] == 42
        and completeness["planned_trial_count"] == 84
        and completeness["completed_snapshot_count"] == 42
        and completeness["completed_trial_count"] == 84
        and open3d_count == 42
        and pcl_count == 42
        and integrity_zero
        and checksum_mismatches == 0
        and all(failures[backend] == 0 and nonfinite[backend] == 0 for backend in BACKENDS)
    )
    # Missing entries without any contradictory/corrupt evidence are the only
    # state this analyzer can classify as an interrupted execution.  Corrupt,
    # duplicate, or extra evidence is an engineering failure, not an excuse to
    # emit NOT_EVALUATED.
    formal_incomplete = bool(
        (completeness["missing_trial_count"] > 0 or completeness["missing_snapshot_count"] > 0)
        and completeness["corrupt_trial_count"] == 0
        and completeness["duplicate_trial_count"] == 0
        and completeness["extra_trial_count"] == 0
        and completeness["extra_snapshot_count"] == 0
    )
    gates = {
        "PHASE_B_CROSS_BACKEND_RANKING_PASS": cross_ranking_pass,
        "PHASE_B_ENGINEERING_PASS": engineering_pass,
        "PHASE_B_GEOMETRY_SEED_CONSISTENCY_PASS": seed_pass,
        "PHASE_B_SCENE_RANKING_PASS": scene_ranking_pass,
        "PHASE_B_WEAK_RICH_EFFECT_PASS": weak_pass,
    }
    signal: bool | str = all(gates.values())
    if formal_incomplete:
        signal = "NOT_EVALUATED"
    final_decision = {
        **gates,
        "CONFIRMATORY_AUTHORIZED": False,
        "FULL_SYNTHETIC_DEVELOPMENT_PROTOCOL_DESIGN_AUTHORIZED": signal is True,
        "FULL_SYNTHETIC_DEVELOPMENT_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PHASE_B_SIGNAL_PASS": signal,
        "REAL_DATA_AUTHORIZED": False,
    }
    runtime_summary = []
    for backend in BACKENDS:
        values = np.asarray([float(row["runtime_ms"]) for row in audit.rows if row["backend"] == backend], dtype=float)
        runtime_summary.append(
            {
                "backend": BACKEND_LABELS[backend],
                "backend_schema_name": backend,
                "maximum_runtime_ms": float(np.max(values)) if values.size else None,
                "median_runtime_ms": float(np.median(values)) if values.size else None,
                "q95_runtime_ms_linear": float(np.quantile(values, 0.95, method="linear")) if values.size else None,
                "trial_count": int(values.size),
            }
        )
    failure_inventory = []
    for backend in BACKENDS:
        chosen = [row for row in audit.rows if row["backend"] == backend]
        classifications = Counter(row["failure_classification"] for row in chosen)
        for classification in sorted(classifications):
            failure_inventory.append(
                {
                    "backend": BACKEND_LABELS[backend],
                    "backend_schema_name": backend,
                    "failure_classification": classification,
                    "failure_count": classifications[classification],
                    "backend_nonfinite_output_count": nonfinite[backend],
                    "backend_solver_failure_count": failures[backend],
                }
            )
    report = {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": None,
        "backend_input_checksum_mismatch_count": checksum_mismatches,
        "backend_input_pairing": pairing,
        "completeness": completeness,
        "cross_backend_ranking": cross_rankings,
        "failure_inventory": failure_inventory,
        "final_decision": final_decision,
        "gate_summary": gates,
        "geometry_seed_consistency": seed_consistency,
        "geometry_seed_raw_values": raw_values,
        "nonfinite_output_count_by_backend": {
            BACKEND_LABELS[backend]: nonfinite[backend] for backend in BACKENDS
        },
        "quantile_method": "linear",
        "run_id": audit.manifest.get("formal_run_id", "phase-b-signal-v1"),
        "runtime_summary": runtime_summary,
        "scene_condition_backend_summary": scene_summaries,
        "scene_ranking": scene_rankings,
        "schema_version": "phase_b_scene_signal_primary_analysis_v1",
        "selected_common_weak_scene": selected_weak,
        "solver_failure_count_by_backend": {
            BACKEND_LABELS[backend]: failures[backend] for backend in BACKENDS
        },
        "weak_rich_effect": weak_effect,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


__all__ = [
    "BACKEND_LABELS",
    "BACKENDS",
    "CONDITIONS",
    "GEOMETRY_SEEDS",
    "PhaseBRawAudit",
    "RICH_SCENE",
    "SCENES",
    "WEAK_CANDIDATES",
    "analyze_phase_b_signal",
    "load_and_validate_phase_b_raw",
]
