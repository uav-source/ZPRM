"""Independent raw reread and recomputation for Full Synthetic Development.

The verifier has its own raw-manifest reader, transform-vector calculation,
grouping, ranking, gate evaluation, and model invocation.  It never calls the
primary analysis entry point or consumes ``primary_analysis.json``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .contracts import (
    canonical_json_sha256,
    file_sha256,
    load_manifest,
    manifest_root,
    write_json,
)
from .full_synthetic_analysis import (
    BACKEND_LABELS,
    BACKENDS,
    CONDITIONS,
    GEOMETRY_SEEDS,
    MAIN_CONDITIONS,
    MEASUREMENT_SEEDS,
    NONIDEAL_CONDITIONS,
    REPEAT_INDICES,
    SCENES,
    full_synthetic_verification_projection,
    validate_common_association_cache,
)
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from .rotation_metrics import project_to_so3, rotation_error_atan2


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _root_path(root: Path, manifest: Mapping[str, Any], fields: Sequence[str], fallback: str) -> Path:
    value = next((manifest[name] for name in fields if isinstance(manifest.get(name), str)), fallback)
    result = (root / value).resolve()
    if result != root and root not in result.parents:
        raise ValueError("independent verifier path escaped repository")
    return result


def _condition_validator(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or value.get("condition") not in CONDITIONS:
        raise ValueError("unauthorized Full Synthetic condition")
    condition = value["condition"]
    copy = dict(value)
    copy["condition"] = "IDEAL_MATCHED"
    checked = validate_phase_a_trial_result_strict(copy)
    checked["condition"] = condition
    return checked


def _backend(value: str) -> str:
    if value == "pcl_iterative_closest_point_with_normals":
        return PCL_BACKEND
    if value in BACKENDS:
        return value
    raise ValueError("planned backend is unauthorized")


def _independent_new_bindings(
    root: Path,
    manifest: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, str]],
    trials: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Independent plan/lock parser for post-run evidence authentication."""

    snapshot_plan_path = _root_path(
        root,
        manifest,
        ("new_planned_snapshots_path", "planned_snapshots_path"),
        "frozen_assets/full_synthetic_development_planned_snapshots_v1.csv",
    )
    trial_plan_path = _root_path(
        root,
        manifest,
        ("new_planned_trials_path", "planned_trials_path"),
        "frozen_assets/full_synthetic_development_planned_trials_v1.csv",
    )
    for path, field, label in (
        (snapshot_plan_path, "new_planned_snapshots_sha256", "snapshot plan"),
        (trial_plan_path, "new_planned_trials_sha256", "trial plan"),
    ):
        if not path.is_file() or file_sha256(path) != manifest.get(field):
            raise ValueError(f"independent {label} SHA binding mismatch")
    snapshot_by_id = {str(row["snapshot_id"]): row for row in snapshots}
    trial_by_id = {str(row["planned_trial_id"]): row for row in trials}
    if (
        len(snapshots) != int(manifest.get("new_planned_snapshot_count", -1))
        or len(snapshot_by_id) != len(snapshots)
        or len(trials) != int(manifest.get("new_planned_trial_count", -1))
        or len(trial_by_id) != len(trials)
        or any(str(row["snapshot_id"]) not in snapshot_by_id for row in trials)
    ):
        raise ValueError("independent new plan inventory mismatch")
    lock_path = _root_path(
        root,
        manifest,
        ("new_snapshot_lock_path", "snapshot_lock_path"),
        "frozen_assets/full_synthetic_development_snapshot_lock_v1.json",
    )
    lock_sha = file_sha256(lock_path)
    if lock_sha != manifest.get("new_snapshot_lock_sha256"):
        raise ValueError("independent snapshot-lock file SHA mismatch")
    lock = load_json_strict(lock_path)
    payload = {
        name: item
        for name, item in lock.items()
        if name != "snapshot_lock_payload_sha256"
    }
    entries = lock.get("snapshots")
    if (
        lock.get("snapshot_lock_payload_sha256") != canonical_json_sha256(payload)
        or lock.get("schema_version")
        != "full_synthetic_development_snapshot_lock_v1"
        or lock.get("planned_snapshot_count") != len(snapshots)
        or type(entries) is not list
        or [row.get("snapshot_id") for row in entries]
        != [str(row["snapshot_id"]) for row in snapshots]
    ):
        raise ValueError("independent snapshot-lock payload/inventory mismatch")
    lock_by_id = {str(row["snapshot_id"]): row for row in entries}
    if len(lock_by_id) != len(entries):
        raise ValueError("independent snapshot lock has duplicate IDs")
    return {
        "cache_root": _root_path(
            root,
            manifest,
            ("new_snapshot_cache_root", "snapshot_cache_root"),
            "data/full_synthetic_development_v1_snapshots",
        ),
        "lock_by_id": lock_by_id,
        "lock_sha256": lock_sha,
        "snapshot_by_id": snapshot_by_id,
        "trial_by_id": trial_by_id,
    }


def _independent_subset_gate(
    root: Path, manifest: Mapping[str, Any]
) -> bool:
    subset_path = _root_path(
        root,
        manifest,
        ("phase_b_subset_reproduction_report_path",),
        "artifacts/full_synthetic_phase_b_subset_reproduction.json",
    )
    gate_path = _root_path(
        root,
        manifest,
        ("pre_run_gate_report_path",),
        "artifacts/full_synthetic_development_pre_run_gate_report.json",
    )
    subset = load_json_strict(subset_path)
    gate = load_json_strict(gate_path)
    evidence = gate.get("evidence_sha256")
    relative = subset_path.relative_to(root).as_posix()
    if (
        type(evidence) is not dict
        or evidence.get(relative) != file_sha256(subset_path)
        or gate.get("ALL_PRE_RUN_GATES_PASS") is not True
        or gate.get("PHASE_B_SUBSET_REPRODUCTION_PASS") is not True
        or subset.get("PHASE_B_SUBSET_REPRODUCTION_PASS") is not True
    ):
        raise ValueError("independent Phase B subset evidence binding mismatch")
    return True


def _read_raw(
    plan_rows: Sequence[Mapping[str, str]],
    run_dir: Path,
    run_id: str,
    validator,
    *,
    expected_implementation_sha256: str,
    expected_protocol_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    plan = {row["planned_trial_id"]: row for row in plan_rows}
    if len(plan) != len(plan_rows):
        raise ValueError("independent verifier found duplicate plan IDs")
    manifest = json.loads((run_dir / "raw_result_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id or type(manifest.get("results")) is not dict:
        raise ValueError("independent raw manifest identity mismatch")
    entries = manifest["results"]
    raw_root = (run_dir / "raw_results").resolve()
    names = []
    for entry in entries.values():
        if type(entry) is dict and isinstance(entry.get("path"), str):
            candidate = (raw_root / entry["path"]).resolve()
            if candidate.parent == raw_root:
                names.append(candidate.name)
    rows = []
    corrupt = 0
    for trial_id in sorted(set(plan) & set(entries)):
        try:
            entry = entries[trial_id]
            if type(entry) is not dict or set(entry) != {"path", "planned_trial_id", "sha256"}:
                raise ValueError("entry schema")
            path = (raw_root / entry["path"]).resolve()
            if path.parent != raw_root or entry["planned_trial_id"] != trial_id:
                raise ValueError("entry identity")
            if not path.is_file() or file_sha256(path) != entry["sha256"]:
                raise ValueError("entry SHA")
            result = validator(load_json_strict(path))
            expected = plan[trial_id]
            if (
                result["planned_trial_id"] != trial_id
                or result["snapshot_id"] != expected["snapshot_id"]
                or result["scene_variant"] != expected["scene_variant"]
                or result["condition"] != expected["condition"]
                or result["backend"] != _backend(expected["backend"])
            ):
                raise ValueError("result-plan identity")
            if (
                result.get("implementation_sha256")
                != expected_implementation_sha256
                or result.get("protocol_sha256") != expected_protocol_sha256
            ):
                raise ValueError("result implementation/protocol SHA")
            rows.append(result)
        except (KeyError, OSError, TypeError, ValueError):
            corrupt += 1
    raw_files = {path.name for path in raw_root.glob("*.json") if path.is_file()}
    defects = {
        "corrupt": corrupt,
        "duplicate": sum(max(0, count - 1) for count in Counter(names).values()),
        "extra": len(set(entries) - set(plan)) + len(raw_files - set(names)),
        "missing": len(set(plan) - set(entries)),
    }
    return rows, defects


def _independent_vectors(reference: np.ndarray, estimate: Sequence[Sequence[float]]) -> dict[str, Any]:
    truth = np.asarray(reference, dtype=np.float64)
    result = np.asarray(estimate, dtype=np.float64)
    projected_truth, _ = project_to_so3(truth[:3, :3])
    projected_result, _ = project_to_so3(result[:3, :3])
    angle = rotation_error_atan2(projected_truth, projected_result)["rotation_error_rad"]
    relative = projected_truth.T @ projected_result
    u, _, vh = np.linalg.svd(relative)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = 1.0 if np.linalg.det(u @ vh) >= 0.0 else -1.0
    relative = u @ correction @ vh
    vector = Rotation.from_matrix(relative).as_rotvec()
    norm = float(np.linalg.norm(vector))
    vector = vector * (angle / norm) if norm > 1.0e-15 else np.zeros(3)
    translation = result[:3, 3] - truth[:3, 3]
    return {
        "rotation_error_rad": float(angle),
        "rotation_vector": vector.astype(float).tolist(),
        "translation_error_m": float(np.linalg.norm(translation)),
        "translation_vector": translation.astype(float).tolist(),
    }


def independently_load_full_synthetic_evidence(
    *, manifest_path: str | Path, run_dir: str | Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently authenticate and reread both frozen result populations."""

    from .full_synthetic_development_protocol import (
        load_strict_authorized_full_synthetic_manifest,
    )

    manifest_file, manifest = load_strict_authorized_full_synthetic_manifest(
        manifest_path
    )
    root = manifest_root(manifest_file)
    new_trial_path = _root_path(
        root,
        manifest,
        ("new_planned_trials_path", "planned_trials_path"),
        "frozen_assets/full_synthetic_development_planned_trials_v1.csv",
    )
    new_snapshot_path = _root_path(
        root,
        manifest,
        ("new_planned_snapshots_path", "planned_snapshots_path"),
        "frozen_assets/full_synthetic_development_planned_snapshots_v1.csv",
    )
    new_plan = _csv(new_trial_path)
    new_snapshots = _csv(new_snapshot_path)
    bindings = _independent_new_bindings(
        root, manifest, new_snapshots, new_plan
    )
    ideal_plan = _csv(root / "frozen_assets/planned_trials.csv")
    ideal_snapshots = _csv(root / "frozen_assets/planned_snapshots.csv")
    ideal_snapshot_by_id = {row["snapshot_id"]: row for row in ideal_snapshots}
    if len(ideal_snapshots) != 210 or len(ideal_snapshot_by_id) != 210:
        raise ValueError("independent Phase A plan inventory mismatch")

    from .full_synthetic_development_protocol import verify_phase_a_ideal_import
    from .full_synthetic_snapshot_builder import read_full_synthetic_snapshot
    from .snapshot_reader import read_snapshot

    phase_a_import = verify_phase_a_ideal_import(root, write_report=False)
    if phase_a_import.get("PHASE_A_IDEAL_IMPORT_PASS") is not True:
        raise ValueError("independent Phase A frozen import verification failed")
    _, ideal_manifest = load_manifest(
        root / "frozen_assets/frozen_experiment_manifest.json"
    )
    ideal_lock_sha = str(ideal_manifest["snapshot_lock_sha256"])
    subset_pass = _independent_subset_gate(root, manifest)
    new_run = Path(run_dir).resolve()
    ideal_run = _root_path(
        root, manifest, ("phase_a_results_root", "phase_a_formal_run_dir", "phase_a_ideal_run_dir"),
        "results/formal_phase_a_v1",
    )
    new_rows, new_defects = _read_raw(
        new_plan, new_run,
        str(manifest.get("formal_run_id", "full-synthetic-development-v1")),
        _condition_validator,
        expected_implementation_sha256=str(
            manifest["implementation_contract_sha256"]
        ),
        expected_protocol_sha256=str(manifest["scientific_protocol_sha256"]),
    )
    ideal_rows, ideal_defects = _read_raw(
        ideal_plan, ideal_run, "phase-a-minimal-harness-formal-v1",
        validate_phase_a_trial_result_strict,
        expected_implementation_sha256=str(
            ideal_manifest["manifest_payload_sha256"]
        ),
        expected_protocol_sha256=str(
            ideal_manifest["scientific_protocol_sha256"]
        ),
    )
    plan = {row["planned_trial_id"]: (row, False) for row in new_plan}
    plan.update({row["planned_trial_id"]: (row, True) for row in ideal_plan})
    new_cache = Path(bindings["cache_root"])
    ideal_cache = _root_path(
        root, manifest, ("phase_a_snapshot_cache_root",), "data/frozen_snapshots"
    )
    references: dict[tuple[bool, str], np.ndarray] = {}
    asset_checksums: dict[tuple[bool, str], dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    metric_mismatch = 0
    for raw in [*ideal_rows, *new_rows]:
        planned, ideal = plan[raw["planned_trial_id"]]
        key = (ideal, raw["snapshot_id"])
        if key not in references:
            snapshot_id = str(raw["snapshot_id"])
            if ideal:
                planned_snapshot = ideal_snapshot_by_id.get(snapshot_id)
                if planned_snapshot is None:
                    raise ValueError("independent Phase A snapshot plan join failed")
                item = read_snapshot(ideal_cache, snapshot_id, arrays=True)
                metadata = item["metadata"]
                expected_metadata = {
                    "condition": planned_snapshot["condition"],
                    "geometry_seed": int(planned_snapshot["geometry_seed_value"]),
                    "geometry_seed_index": int(planned_snapshot["geometry_seed_index"]),
                    "measurement_seed": int(planned_snapshot["measurement_seed_value"]),
                    "measurement_seed_index": int(planned_snapshot["measurement_seed_index"]),
                    "repeat_index": int(planned_snapshot["repeat_index"]),
                    "scene_variant": planned_snapshot["scene_variant"],
                    "snapshot_id": snapshot_id,
                }
                if any(metadata.get(name) != value for name, value in expected_metadata.items()):
                    raise ValueError("independent Phase A snapshot/plan mismatch")
                checksum_value = {
                    "source_checksum": metadata["source_raw_checksum"],
                    "target_checksum": metadata["target_raw_checksum"],
                    "reference_pose_checksum": metadata["reference_pose_raw_checksum"],
                    "snapshot_checksum": metadata["snapshot_checksum"],
                }
                reference = item["reference"]
                expected_lock_sha = ideal_lock_sha
            else:
                planned_snapshot = bindings["snapshot_by_id"].get(snapshot_id)
                lock_entry = bindings["lock_by_id"].get(snapshot_id)
                if planned_snapshot is None or lock_entry is None:
                    raise ValueError("independent new snapshot plan/lock join failed")
                item = read_full_synthetic_snapshot(
                    new_cache,
                    planned_snapshot,
                    expected_lock_entry=lock_entry,
                    arrays=True,
                )
                checksum_value = {
                    name: item[name]
                    for name in (
                        "source_checksum",
                        "target_checksum",
                        "reference_pose_checksum",
                        "snapshot_checksum",
                    )
                }
                reference = item["reference"]
                expected_lock_sha = bindings["lock_sha256"]
            references[key] = np.asarray(reference, dtype=np.float64)
            asset_checksums[key] = checksum_value
        expected_lock_sha = ideal_lock_sha if ideal else bindings["lock_sha256"]
        if (
            any(raw.get(name) != digest for name, digest in asset_checksums[key].items())
            or raw.get("snapshot_lock_sha256") != expected_lock_sha
        ):
            raise ValueError("independent trial/snapshot scientific binding mismatch")
        metrics = {
            "translation_error_m": raw["translation_update_m"],
            "rotation_error_rad": raw["rotation_update_rad"],
            "translation_vector": None,
            "rotation_vector": None,
        }
        if raw["final_transform_4x4"] is not None:
            metrics = _independent_vectors(references[key], raw["final_transform_4x4"])
            if (
                raw["translation_update_m"] is None
                or raw["rotation_update_rad"] is None
                or not math.isclose(metrics["translation_error_m"], float(raw["translation_update_m"]), abs_tol=1e-12, rel_tol=1e-12)
                or not math.isclose(metrics["rotation_error_rad"], float(raw["rotation_update_rad"]), abs_tol=1e-12, rel_tol=1e-12)
            ):
                metric_mismatch += 1
        backend = raw["backend"]
        rows.append(
            {
                **raw,
                **metrics,
                "backend": BACKEND_LABELS[backend],
                "backend_schema_name": backend,
                "geometry_seed": int(planned.get("geometry_seed_value", planned.get("geometry_seed"))),
                "measurement_seed": int(planned.get("measurement_seed_value", planned.get("measurement_seed"))),
                "repeat_index": int(planned["repeat_index"]),
            }
        )
    by_snapshot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_snapshot[row["snapshot_id"]].append(row)
    pairing = sum(
        len(selected) != 2
        or Counter(row["backend_schema_name"] for row in selected) != Counter(BACKENDS)
        or any(
            len({row[name] for row in selected}) != 1
            for name in ("source_checksum", "target_checksum", "reference_pose_checksum", "snapshot_checksum")
        )
        for selected in by_snapshot.values()
    )
    phase_a_pass = bool(
        phase_a_import.get("PHASE_A_IDEAL_IMPORT_PASS") is True
        and len(ideal_snapshots) == 210 and len(ideal_rows) == 420
        and not any(row["solver_failure"] or not row["finite_output"] for row in ideal_rows)
        and not any(ideal_defects.values())
    )
    new_by_snapshot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in new_rows:
        new_by_snapshot[str(row["snapshot_id"])].append(row)
    new_completed = sum(
        len(selected) == 2
        and Counter(row["backend"] for row in selected) == Counter(BACKENDS)
        for selected in new_by_snapshot.values()
    )
    combined_completed = sum(
        len(selected) == 2
        and Counter(row["backend_schema_name"] for row in selected)
        == Counter(BACKENDS)
        for selected in by_snapshot.values()
    )
    integrity = {
        "PHASE_A_IDEAL_IMPORT_PASS": phase_a_pass,
        "PHASE_B_SUBSET_REPRODUCTION_PASS": subset_pass,
        "backend_input_checksum_mismatch_count": pairing + metric_mismatch,
        "combined_snapshot_count": combined_completed,
        "combined_trial_count": len(rows),
        "corrupt_trial_count": new_defects["corrupt"] + ideal_defects["corrupt"],
        "duplicate_trial_count": new_defects["duplicate"] + ideal_defects["duplicate"],
        "extra_trial_count": new_defects["extra"] + ideal_defects["extra"],
        "infrastructure_interruption_unresolved_count": new_defects["missing"] + new_defects["corrupt"],
        "metric_recomputation_mismatch_count": metric_mismatch,
        "missing_trial_count": new_defects["missing"] + ideal_defects["missing"],
        "new_snapshot_count": new_completed,
        "new_trial_count": len(new_rows),
        "planned_new_snapshot_count": len(new_snapshots),
        "planned_new_trial_count": len(new_plan),
    }
    rows.sort(key=lambda row: row["planned_trial_id"])
    return rows, integrity


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


_INDEPENDENT_BOOTSTRAP_SEED = 1191248828


def _independent_interval(
    records: Sequence[Mapping[str, Any]],
    statistic,
    *,
    repetitions: int,
) -> tuple[float | None, float | None]:
    if not records or repetitions <= 0:
        return None, None
    groups: dict[int, dict[tuple[int, int], list[Mapping[str, Any]]]] = {
        seed: defaultdict(list) for seed in GEOMETRY_SEEDS
    }
    for row in records:
        geometry = int(row["geometry_seed"])
        groups[geometry][
            (int(row["measurement_seed"]), int(row["repeat_index"]))
        ].append(row)
    if any(not groups[seed] for seed in GEOMETRY_SEEDS):
        return None, None
    generator = np.random.default_rng(_INDEPENDENT_BOOTSTRAP_SEED)
    values: list[float] = []
    for _ in range(repetitions):
        sample: list[Mapping[str, Any]] = []
        for geometry_value in generator.choice(
            np.asarray(GEOMETRY_SEEDS), size=3, replace=True
        ):
            block_map = groups[int(geometry_value)]
            keys = sorted(block_map)
            for index in generator.integers(0, len(keys), size=len(keys)):
                sample.extend(block_map[keys[int(index)]])
        try:
            value = float(statistic(sample))
        except (FloatingPointError, TypeError, ValueError, ZeroDivisionError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None, None
    low, high = np.quantile(values, [0.025, 0.975], method="linear")
    return float(low), float(high)


def _independent_rho(left: Sequence[float], right: Sequence[float]) -> float:
    result = spearmanr(left, right)
    value = float(result.statistic)
    return value if math.isfinite(value) else float("nan")


def _description(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "median": None,
            "iqr": None,
            "q25": None,
            "q75": None,
            "q95": None,
        }
    q25, median, q75, q95 = np.quantile(
        array, [0.25, 0.5, 0.75, 0.95], method="linear"
    )
    return {
        "count": int(array.size),
        "median": float(median),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
    }


def _independent_systematic(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene_variant"], row["geometry_seed"], row["condition"], row["backend_schema_name"])].append(row)
    output = []
    for scene in SCENES:
        for geometry in GEOMETRY_SEEDS:
            for condition in CONDITIONS:
                for backend in BACKENDS:
                    chosen = grouped.get((scene, geometry, condition, backend), [])
                    successful = [
                        row for row in chosen
                        if not row["solver_failure"] and row["finite_output"]
                        and row.get("translation_vector") is not None
                        and row.get("rotation_vector") is not None
                    ]
                    translation = np.asarray(
                        [row["translation_vector"] for row in successful],
                        dtype=np.float64,
                    )
                    rotation = np.asarray(
                        [row["rotation_vector"] for row in successful],
                        dtype=np.float64,
                    )
                    complete = (
                        len(successful) == 10
                        and translation.shape == (10, 3)
                        and rotation.shape == (10, 3)
                        and np.all(np.isfinite(translation))
                        and np.all(np.isfinite(rotation))
                    )
                    mean_translation = np.mean(translation, axis=0) if complete else None
                    mean_rotation = np.mean(rotation, axis=0) if complete else None
                    translation_covariance = (
                        np.cov(translation, rowvar=False, ddof=1)
                        if complete
                        else None
                    )
                    rotation_covariance = (
                        np.cov(rotation, rowvar=False, ddof=1)
                        if complete
                        else None
                    )
                    translation_offset = (
                        float(np.linalg.norm(mean_translation))
                        if mean_translation is not None
                        else None
                    )
                    rotation_offset = (
                        float(np.linalg.norm(mean_rotation))
                        if mean_rotation is not None
                        else None
                    )
                    translation_norms = (
                        np.linalg.norm(translation, axis=1)
                        if complete
                        else np.asarray([], dtype=np.float64)
                    )
                    denominator = (
                        float(np.mean(translation_norms))
                        if translation_norms.size
                        else None
                    )
                    fraction = (
                        float(translation_offset / denominator)
                        if translation_offset is not None
                        and denominator is not None
                        and denominator > 1.0e-12
                        else None
                    )
                    nonzero = (
                        translation[translation_norms > 1.0e-12]
                        if translation_norms.size
                        else np.empty((0, 3), dtype=np.float64)
                    )
                    concentration = (
                        float(
                            np.linalg.norm(
                                np.sum(
                                    nonzero
                                    / np.linalg.norm(nonzero, axis=1)[:, None],
                                    axis=0,
                                )
                            )
                            / len(nonzero)
                        )
                        if len(nonzero)
                        else None
                    )
                    output.append(
                        {
                            "backend_schema_name": backend,
                            "condition": condition,
                            "direction_valid_nonzero_count": int(len(nonzero)),
                            "geometry_seed": geometry,
                            "mean_rotation_vector": None if mean_rotation is None else mean_rotation.tolist(),
                            "mean_translation_vector": None if mean_translation is None else mean_translation.tolist(),
                            "planned_observation_count": 10,
                            "rotation_repeatability_covariance": None if rotation_covariance is None else rotation_covariance.tolist(),
                            "rotation_repeatability_rms_rad": None if rotation_covariance is None else float(math.sqrt(max(float(np.trace(rotation_covariance)), 0.0))),
                            "scene_variant": scene,
                            "successful_observation_count": len(successful),
                            "systematic_fraction_translation": fraction,
                            "systematic_rotation_offset_rad": rotation_offset,
                            "systematic_translation_offset_m": translation_offset,
                            "translation_direction_concentration": concentration,
                            "translation_repeatability_covariance": None if translation_covariance is None else translation_covariance.tolist(),
                            "translation_repeatability_rms_m": None if translation_covariance is None else float(math.sqrt(max(float(np.trace(translation_covariance)), 0.0))),
                        }
                    )
    return output


_INDEPENDENT_MODEL_B_FIELDS = (
    "correspondence_turnover",
    "accepted_source_turnover",
    "median_normal_angle_change_deg",
    "q95_normal_angle_change_deg",
    "residual_rmse_change",
    "correspondence_count_change_ratio",
)


def _independent_model_vector(
    row: Mapping[str, Any], *, model_b: bool
) -> np.ndarray:
    required = (
        "initial_residual_rmse",
        "condition_number_trans",
        "lambda_min_trans",
        "spectral_entropy_trans",
        "initial_correspondence_count",
        "initial_translation_gradient_norm",
    )
    numbers = [_finite(row.get(name)) for name in required]
    if any(value is None or value < 0.0 for value in numbers):
        raise ValueError("independent model A feature is invalid")
    rmse, condition, lambda_min, entropy, count, gradient = (
        float(value) for value in numbers
    )
    transformed = [
        math.log10(rmse + 1.0e-9),
        math.log10(condition + 1.0),
        math.log10(1.0 / max(lambda_min, 1.0e-12)),
        entropy,
        math.log10(count + 1.0),
        gradient,
    ]
    if model_b:
        extra = [_finite(row.get(name)) for name in _INDEPENDENT_MODEL_B_FIELDS]
        if any(value is None for value in extra):
            raise ValueError("independent model B feature is invalid")
        transformed.extend(float(value) for value in extra)
    result = np.asarray(transformed, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("independent transformed model feature is nonfinite")
    return result


def _independent_model_cv(
    rows: Sequence[Mapping[str, Any]], *, backend: str, model_b: bool
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("backend_schema_name") == backend
        and row.get("condition") in NONIDEAL_CONDITIONS
        and row.get("common_association_valid") is True
    ]
    if not selected:
        raise ValueError("independent model has no rows")
    features = np.vstack(
        [_independent_model_vector(row, model_b=model_b) for row in selected]
    )
    target = np.asarray(
        [
            math.log10(float(row["translation_error_m"]) + 1.0e-9)
            for row in selected
        ],
        dtype=np.float64,
    )
    geometry = np.asarray(
        [int(row["geometry_seed"]) for row in selected], dtype=np.int64
    )
    seeds = np.asarray(sorted(set(geometry.tolist())), dtype=np.int64)
    if seeds.size != 3:
        raise ValueError("independent LOSO model requires three geometry seeds")
    absolute_errors: list[float] = []
    fold_rows = []
    for held_out in seeds:
        test = geometry == held_out
        train = ~test
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_x = scaler.fit_transform(features[train])
        test_x = scaler.transform(features[test])
        estimator = Ridge(alpha=1.0, fit_intercept=True)
        estimator.fit(train_x, target[train])
        prediction = estimator.predict(test_x)
        fold_error = np.abs(prediction - target[test])
        absolute_errors.extend(fold_error.astype(float).tolist())
        fold_rows.append(
            {
                "held_out_geometry_seed": int(held_out),
                "training_geometry_seeds": [
                    int(seed) for seed in seeds if seed != held_out
                ],
                "training_row_count": int(np.count_nonzero(train)),
                "test_row_count": int(np.count_nonzero(test)),
                "fold_mae_log10_translation_error": float(np.mean(fold_error)),
                "scaler_fit_scope": "training_fold_only",
                "scaler_mean": scaler.mean_.astype(float).tolist(),
                "scaler_scale": scaler.scale_.astype(float).tolist(),
                "ridge_alpha": float(estimator.alpha),
                "ridge_fit_intercept": bool(estimator.fit_intercept),
                "target_true": target[test].astype(float).tolist(),
                "target_predicted": prediction.astype(float).tolist(),
            }
        )
    return {
        "cv_mae": float(np.mean(np.asarray(absolute_errors, dtype=np.float64))),
        "folds": fold_rows,
    }


def _independent_model_comparisons(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for backend in BACKENDS:
        selected_count = sum(
            row.get("backend_schema_name") == backend
            and row.get("common_association_valid") is True
            for row in rows
        )
        try:
            model_a_result = _independent_model_cv(
                rows, backend=backend, model_b=False
            )
            model_b_result = _independent_model_cv(
                rows, backend=backend, model_b=True
            )
            model_a = model_a_result["cv_mae"]
            model_b = model_b_result["cv_mae"]
            improvement = None if model_a <= 0.0 else (model_a - model_b) / model_a
            folds = {
                "model_a": model_a_result["folds"],
                "model_b": model_b_result["folds"],
            }
        except (KeyError, TypeError, ValueError):
            model_a = model_b = improvement = None
            folds = None
            selected_count = 0
        output.append(
            {
                "backend_schema_name": backend,
                "fold_results": folds,
                "model_a_cv_mae": model_a,
                "model_b_cv_mae": model_b,
                "relative_mae_improvement": improvement,
                "sample_count": selected_count,
            }
        )
    return output


def _independent_candidate_id(
    backend: str, condition: str, snapshot_a: str, snapshot_b: str
) -> str:
    value = json.dumps(
        {
            "backend": backend,
            "condition": condition,
            "snapshot_a": snapshot_a,
            "snapshot_b": snapshot_b,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"automatic::{hashlib.sha256(value).hexdigest()}"


def _independent_candidates(
    rows: Sequence[Mapping[str, Any]], *, backend: str
) -> list[dict[str, Any]]:
    selected = sorted(
        (
            row
            for row in rows
            if row.get("backend_schema_name") == backend
            and row.get("condition") in NONIDEAL_CONDITIONS
            and row.get("common_association_valid") is True
        ),
        key=lambda row: (str(row.get("condition")), str(row.get("snapshot_id"))),
    )
    output = []
    for first_index, first in enumerate(selected):
        for second in selected[first_index + 1 :]:
            if (
                first.get("condition") != second.get("condition")
                or first.get("snapshot_id") == second.get("snapshot_id")
            ):
                continue
            try:
                first_error = float(first["translation_error_m"])
                second_error = float(second["translation_error_m"])
                if min(first_error, second_error) <= 0.0:
                    continue
                high, low = (
                    (first, second)
                    if first_error >= second_error
                    else (second, first)
                )
                high_error, low_error = max(first_error, second_error), min(
                    first_error, second_error
                )
                eigen_high = np.asarray(
                    [
                        float(high["lambda_min_trans"]),
                        float(high["lambda_mid_trans"]),
                        float(high["lambda_max_trans"]),
                    ],
                    dtype=np.float64,
                )
                eigen_low = np.asarray(
                    [
                        float(low["lambda_min_trans"]),
                        float(low["lambda_mid_trans"]),
                        float(low["lambda_max_trans"]),
                    ],
                    dtype=np.float64,
                )
                if np.any(eigen_high < 0.0) or np.any(eigen_low < 0.0):
                    continue
                high_sum = float(np.sum(eigen_high))
                low_sum = float(np.sum(eigen_low))
                if high_sum <= 1.0e-12 or low_sum <= 1.0e-12:
                    continue
                eigen_high /= high_sum
                eigen_low /= low_sum
                similarity = float(
                    np.dot(eigen_high, eigen_low)
                    / (np.linalg.norm(eigen_high) * np.linalg.norm(eigen_low))
                )
                condition_high = float(high["condition_number_trans"])
                condition_low = float(low["condition_number_trans"])
                rmse_high = float(high["initial_residual_rmse"])
                rmse_low = float(low["initial_residual_rmse"])
                count_high = float(high["initial_correspondence_count"])
                count_low = float(low["initial_correspondence_count"])
                if min(condition_high, condition_low, rmse_high, rmse_low, count_low) <= 0.0:
                    continue
                condition_difference = abs(math.log(condition_high / condition_low))
                rmse_difference = abs(math.log(rmse_high / rmse_low))
                count_ratio = count_high / count_low
                error_ratio = high_error / low_error
                turnover_difference = abs(
                    float(high["correspondence_turnover"])
                    - float(low["correspondence_turnover"])
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if not (
                similarity >= 0.98
                and condition_difference <= 0.20
                and rmse_difference <= 0.20
                and 0.90 <= count_ratio <= 1.10
                and error_ratio >= 5.0
                and turnover_difference >= 0.15
                and math.isfinite(error_ratio)
            ):
                continue
            snapshot_a = str(high["snapshot_id"])
            snapshot_b = str(low["snapshot_id"])
            output.append(
                {
                    "candidate_pair_id": _independent_candidate_id(
                        backend, str(high["condition"]), snapshot_a, snapshot_b
                    ),
                    "backend": backend,
                    "backend_schema_name": backend,
                    "snapshot_a": snapshot_a,
                    "snapshot_b": snapshot_b,
                    "scene_a": str(high["scene_variant"]),
                    "scene_b": str(low["scene_variant"]),
                    "condition": str(high["condition"]),
                    "linear_metric_similarity": similarity,
                    "condition_log_ratio_abs": condition_difference,
                    "initial_rmse_log_ratio_abs": rmse_difference,
                    "correspondence_count_ratio": count_ratio,
                    "error_ratio": error_ratio,
                    "turnover_difference": turnover_difference,
                    "manual_review_status": "AUTOMATIC_CANDIDATE",
                }
            )
    return output


def _independent_incremental_gate(
    comparisons: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    by_backend = {row["backend_schema_name"]: row for row in comparisons}
    improvements = [
        by_backend.get(backend, {}).get("relative_mae_improvement")
        for backend in BACKENDS
    ]
    model_pass = bool(
        all(value is not None and math.isfinite(float(value)) for value in improvements)
        and float(np.mean(np.asarray(improvements, dtype=np.float64))) >= 0.10
        and all(float(value) >= -0.02 for value in improvements)
    )
    candidate_counts = Counter(row.get("backend") for row in candidates)
    scene_pairs = {
        tuple(sorted((str(row.get("scene_a")), str(row.get("scene_b")))))
        for row in candidates
    }
    candidate_pass = bool(
        all(candidate_counts[backend] >= 10 for backend in BACKENDS)
        and len(scene_pairs) >= 3
    )
    detail = {
        "automatic_candidate_count_by_backend": {
            BACKEND_LABELS[backend]: candidate_counts[backend]
            for backend in BACKENDS
        },
        "condition_a_model_improvement_pass": model_pass,
        "condition_b_candidate_pass": candidate_pass,
        "distinct_scene_pair_count": len(scene_pairs),
        "mean_relative_mae_improvement": (
            None
            if any(value is None for value in improvements)
            else float(np.mean(np.asarray(improvements, dtype=np.float64)))
        ),
    }
    return bool(model_pass or candidate_pass), detail


def independently_recompute_full_synthetic(
    *,
    trials: Sequence[Mapping[str, Any]],
    common_records: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    bootstrap_repetitions: int = 2000,
) -> dict[str, Any]:
    """Second implementation of every decision-driving statistic and gate."""

    cells: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trials:
        cells[(row["scene_variant"], row["condition"], row["backend_schema_name"])].append(row)
    scene_rows = []
    for scene in SCENES:
        for condition in CONDITIONS:
            for backend in BACKENDS:
                chosen = cells[(scene, condition, backend)]
                successful = [
                    row for row in chosen
                    if not row["solver_failure"] and row["finite_output"]
                    and _finite(row.get("translation_error_m")) is not None
                    and _finite(row.get("rotation_error_rad")) is not None
                ]
                translation = _description([float(row["translation_error_m"]) for row in successful])
                rotation = _description([float(row["rotation_error_rad"]) for row in successful])
                ci_low, ci_high = _independent_interval(
                    successful,
                    lambda sample: np.median(
                        [float(row["translation_error_m"]) for row in sample]
                    ),
                    repetitions=bootstrap_repetitions,
                )
                scene_rows.append(
                    {
                        "backend_schema_name": backend,
                        "condition": condition,
                        "exploratory_translation_median_ci95_high_m": ci_high,
                        "exploratory_translation_median_ci95_low_m": ci_low,
                        "planned_trial_count": 30,
                        "rotation_iqr_rad": rotation["iqr"],
                        "rotation_median_rad": rotation["median"],
                        "rotation_q25_rad": rotation["q25"],
                        "rotation_q75_rad": rotation["q75"],
                        "rotation_q95_rad": rotation["q95"],
                        "scene_variant": scene,
                        "solver_failure_count": sum(row["solver_failure"] for row in chosen),
                        "success_rate": len(successful) / 30.0,
                        "successful_trial_count": len(successful),
                        "translation_iqr_m": translation["iqr"],
                        "translation_median_m": translation["median"],
                        "translation_q25_m": translation["q25"],
                        "translation_q75_m": translation["q75"],
                        "translation_q95_m": translation["q95"],
                    }
                )
    systematic = _independent_systematic(trials)

    successful_map: dict[tuple[str, str, str, tuple[int, int, int]], Mapping[str, Any]] = {}
    for row in trials:
        if not row["solver_failure"] and row["finite_output"]:
            successful_map[(
                row["backend_schema_name"], row["condition"], row["scene_variant"],
                (row["geometry_seed"], row["measurement_seed"], row["repeat_index"]),
            )] = row
    effect_rows = []
    effect_passes = []
    for condition in MAIN_CONDITIONS:
        for backend in BACKENDS:
            rich_all = [
                row for row in trials
                if row["backend_schema_name"] == backend and row["condition"] == condition
                and row["scene_variant"] == "GEOMETRY_RICH_ROOM"
                and not row["solver_failure"] and row["finite_output"]
            ]
            weak_all = [
                row for row in trials
                if row["backend_schema_name"] == backend and row["condition"] == condition
                and row["scene_variant"] == "LONG_CORRIDOR"
                and not row["solver_failure"] and row["finite_output"]
            ]
            rich_map = {
                (row["geometry_seed"], row["measurement_seed"], row["repeat_index"]): row
                for row in rich_all
            }
            weak_map = {
                (row["geometry_seed"], row["measurement_seed"], row["repeat_index"]): row
                for row in weak_all
            }
            blocks = set(rich_map) & set(weak_map)
            effect_records = [
                {
                    "geometry_seed": key[0],
                    "measurement_seed": key[1],
                    "repeat_index": key[2],
                    "rich": float(rich_map[key]["translation_error_m"]),
                    "weak": float(weak_map[key]["translation_error_m"]),
                }
                for key in sorted(blocks)
            ]
            rich_values = [row["rich"] for row in effect_records]
            weak_values = [row["weak"] for row in effect_records]
            rich_median = float(np.median(rich_values)) if rich_values else None
            weak_median = float(np.median(weak_values)) if weak_values else None
            ratio = weak_median / rich_median if rich_median is not None and rich_median > 1e-12 else None
            difference = weak_median - rich_median if weak_median is not None and rich_median is not None else None
            wins = sum(row["weak"] > row["rich"] for row in effect_records)
            win_rate = wins / 30.0
            rich_rate = len(rich_all) / 30.0
            weak_rate = len(weak_all) / 30.0
            difference_low, difference_high = _independent_interval(
                effect_records,
                lambda sample: np.median(
                    [row["weak"] - row["rich"] for row in sample]
                ),
                repetitions=bootstrap_repetitions,
            )
            ratio_low, ratio_high = _independent_interval(
                effect_records,
                lambda sample: (
                    np.median([row["weak"] for row in sample])
                    / np.median([row["rich"] for row in sample])
                    if np.median([row["rich"] for row in sample]) > 1.0e-12
                    else float("nan")
                ),
                repetitions=bootstrap_repetitions,
            )
            passed = bool(
                ratio is not None and ratio >= 5.0
                and difference is not None and difference >= 0.005 and win_rate >= 0.8
                and rich_rate >= 0.9 and weak_rate >= 0.9
            )
            effect_passes.append(passed)
            effect_rows.append(
                {
                    "absolute_median_difference_m": difference,
                    "backend_schema_name": backend,
                    "condition": condition,
                    "corridor_median_m": weak_median,
                    "corridor_success_rate": weak_rate,
                    "exploratory_difference_ci95_high_m": difference_high,
                    "exploratory_difference_ci95_low_m": difference_low,
                    "exploratory_ratio_ci95_high": ratio_high,
                    "exploratory_ratio_ci95_low": ratio_low,
                    "gate_pass": passed,
                    "matched_block_count": len(blocks),
                    "paired_win_count": wins,
                    "paired_win_rate": win_rate,
                    "rich_median_m": rich_median,
                    "rich_success_rate": rich_rate,
                    "weak_rich_median_ratio": ratio,
                }
            )

    condition_contrasts = []
    baseline_condition = "INDEPENDENT_NOISE_FREE"
    for backend in BACKENDS:
        for scene in SCENES:
            for condition in NONIDEAL_CONDITIONS[1:]:
                contrast_records = []
                for geometry in GEOMETRY_SEEDS:
                    for measurement in MEASUREMENT_SEEDS:
                        for repeat in REPEAT_INDICES:
                            block = (geometry, measurement, repeat)
                            baseline = successful_map.get(
                                (backend, baseline_condition, scene, block)
                            )
                            contrasted = successful_map.get(
                                (backend, condition, scene, block)
                            )
                            if baseline is not None and contrasted is not None:
                                contrast_records.append(
                                    {
                                        "baseline": float(
                                            baseline["translation_error_m"]
                                        ),
                                        "contrast": float(
                                            contrasted["translation_error_m"]
                                        ),
                                        "geometry_seed": geometry,
                                        "measurement_seed": measurement,
                                        "repeat_index": repeat,
                                    }
                                )
                baseline_values = [row["baseline"] for row in contrast_records]
                contrasted_values = [row["contrast"] for row in contrast_records]
                baseline_median = (
                    float(np.median(baseline_values)) if baseline_values else None
                )
                contrasted_median = (
                    float(np.median(contrasted_values)) if contrasted_values else None
                )
                paired_difference = (
                    float(
                        np.median(
                            [
                                row["contrast"] - row["baseline"]
                                for row in contrast_records
                            ]
                        )
                    )
                    if contrast_records
                    else None
                )
                ratio = (
                    contrasted_median / baseline_median
                    if baseline_median is not None
                    and contrasted_median is not None
                    and baseline_median > 1.0e-12
                    else None
                )
                ci_low, ci_high = _independent_interval(
                    contrast_records,
                    lambda sample: np.median(
                        [row["contrast"] - row["baseline"] for row in sample]
                    ),
                    repetitions=bootstrap_repetitions,
                )
                condition_contrasts.append(
                    {
                        "backend_schema_name": backend,
                        "baseline_condition": baseline_condition,
                        "baseline_median_m": baseline_median,
                        "contrast_condition": condition,
                        "contrast_median_m": contrasted_median,
                        "exploratory_paired_difference_ci95_high_m": ci_high,
                        "exploratory_paired_difference_ci95_low_m": ci_low,
                        "matched_block_count": len(contrast_records),
                        "median_ratio": ratio,
                        "paired_median_difference_m": paired_difference,
                        "paired_win_rate": (
                            sum(
                                row["contrast"] > row["baseline"]
                                for row in contrast_records
                            )
                            / len(contrast_records)
                            if contrast_records
                            else None
                        ),
                        "scene_variant": scene,
                    }
                )

    median_lookup = {
        (row["condition"], row["backend_schema_name"], row["scene_variant"]): row["translation_median_m"]
        for row in scene_rows
    }
    rank_rows = []
    rich_rank_pass_count = 0
    weak_rank_pass_count = 0
    for condition in NONIDEAL_CONDITIONS:
        for backend in BACKENDS:
            medians = [median_lookup[(condition, backend, scene)] for scene in SCENES]
            valid = all(value is not None for value in medians)
            ranks = rankdata(np.asarray(medians, dtype=float), method="average") if valid else np.full(7, np.nan)
            counts = Counter(medians) if valid else Counter()
            mapping = dict(zip(SCENES, ranks))
            rich_pass = bool(valid and mapping["GEOMETRY_RICH_ROOM"] <= 2)
            weak_pass = bool(valid and max(mapping["LONG_CORRIDOR"], mapping["END_FACE_TRANSITION_ABSENT"]) >= 5)
            rich_rank_pass_count += rich_pass
            weak_rank_pass_count += weak_pass
            for scene, median, rank in zip(SCENES, medians, ranks):
                rank_rows.append(
                    {
                        "backend_schema_name": backend,
                        "condition": condition,
                        "is_tied": bool(valid and counts[median] > 1),
                        "rich_lowest_two_group_pass": rich_pass,
                        "scene_variant": scene,
                        "translation_median_m": median,
                        "translation_rank_ascending_average_ties": float(rank) if np.isfinite(rank) else None,
                        "weak_highest_three_group_pass": weak_pass,
                    }
                )
    cross_rows = []
    rhos = {}
    complete_blocks_by_condition: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for condition in NONIDEAL_CONDITIONS:
        for geometry in GEOMETRY_SEEDS:
            complete_blocks_by_condition[(condition, geometry)] = [
                (measurement, repeat)
                for measurement in MEASUREMENT_SEEDS
                for repeat in REPEAT_INDICES
                if all(
                    (backend, condition, scene, (geometry, measurement, repeat))
                    in successful_map
                    for backend in BACKENDS
                    for scene in SCENES
                )
            ]
    for condition in NONIDEAL_CONDITIONS:
        left = [median_lookup[(condition, OPEN3D_BACKEND, scene)] for scene in SCENES]
        right = [median_lookup[(condition, PCL_BACKEND, scene)] for scene in SCENES]
        result = spearmanr(left, right) if all(value is not None for value in left + right) else None
        rho = float(result.statistic) if result is not None and np.isfinite(result.statistic) else None
        pvalue = float(result.pvalue) if result is not None and np.isfinite(result.pvalue) else None
        rhos[condition] = rho
        bootstrap_values = []
        rng = np.random.default_rng(_INDEPENDENT_BOOTSTRAP_SEED)
        for _ in range(bootstrap_repetitions):
            left_by_scene: dict[str, list[float]] = defaultdict(list)
            right_by_scene: dict[str, list[float]] = defaultdict(list)
            for geometry_value in rng.choice(
                np.asarray(GEOMETRY_SEEDS), size=3, replace=True
            ):
                geometry = int(geometry_value)
                blocks = complete_blocks_by_condition[(condition, geometry)]
                if not blocks:
                    continue
                for block_index in rng.integers(0, len(blocks), size=len(blocks)):
                    measurement, repeat = blocks[int(block_index)]
                    block = (geometry, measurement, repeat)
                    for scene in SCENES:
                        left_by_scene[scene].append(
                            float(
                                successful_map[
                                    (OPEN3D_BACKEND, condition, scene, block)
                                ]["translation_error_m"]
                            )
                        )
                        right_by_scene[scene].append(
                            float(
                                successful_map[
                                    (PCL_BACKEND, condition, scene, block)
                                ]["translation_error_m"]
                            )
                        )
            if all(left_by_scene[scene] and right_by_scene[scene] for scene in SCENES):
                boot = spearmanr(
                    [float(np.median(left_by_scene[scene])) for scene in SCENES],
                    [float(np.median(right_by_scene[scene])) for scene in SCENES],
                )
                if np.isfinite(boot.statistic):
                    bootstrap_values.append(float(boot.statistic))
        if bootstrap_values:
            low, high = np.quantile(
                bootstrap_values, [0.025, 0.975], method="linear"
            )
            interval = float(low), float(high)
        else:
            interval = None, None
        cross_rows.append(
            {
                "condition": condition,
                "exploratory_spearman_ci95_high": interval[1],
                "exploratory_spearman_ci95_low": interval[0],
                "pvalue_descriptive": pvalue,
                "scope": "CONDITION",
                "scene_count": 7,
                "spearman_rho": rho,
            }
        )
    pooled_left = [median_lookup[(condition, OPEN3D_BACKEND, scene)] for condition in NONIDEAL_CONDITIONS for scene in SCENES]
    pooled_right = [median_lookup[(condition, PCL_BACKEND, scene)] for condition in NONIDEAL_CONDITIONS for scene in SCENES]
    pooled_result = spearmanr(pooled_left, pooled_right)
    pooled_rho = float(pooled_result.statistic) if np.isfinite(pooled_result.statistic) else None
    pooled_pvalue = float(pooled_result.pvalue) if np.isfinite(pooled_result.pvalue) else None
    pooled_complete = {
        geometry: [
            (measurement, repeat)
            for measurement in MEASUREMENT_SEEDS
            for repeat in REPEAT_INDICES
            if all(
                (backend, condition, scene, (geometry, measurement, repeat))
                in successful_map
                for backend in BACKENDS
                for condition in NONIDEAL_CONDITIONS
                for scene in SCENES
            )
        ]
        for geometry in GEOMETRY_SEEDS
    }
    pooled_bootstrap = []
    pooled_rng = np.random.default_rng(_INDEPENDENT_BOOTSTRAP_SEED)
    all_cells = [
        (condition, scene)
        for condition in NONIDEAL_CONDITIONS
        for scene in SCENES
    ]
    for _ in range(bootstrap_repetitions):
        left_by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
        right_by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
        for geometry_value in pooled_rng.choice(
            np.asarray(GEOMETRY_SEEDS), size=3, replace=True
        ):
            geometry = int(geometry_value)
            blocks = pooled_complete[geometry]
            if not blocks:
                continue
            for block_index in pooled_rng.integers(0, len(blocks), size=len(blocks)):
                measurement, repeat = blocks[int(block_index)]
                block = (geometry, measurement, repeat)
                for condition, scene in all_cells:
                    left_by_cell[(condition, scene)].append(
                        float(
                            successful_map[
                                (OPEN3D_BACKEND, condition, scene, block)
                            ]["translation_error_m"]
                        )
                    )
                    right_by_cell[(condition, scene)].append(
                        float(
                            successful_map[
                                (PCL_BACKEND, condition, scene, block)
                            ]["translation_error_m"]
                        )
                    )
        if all(left_by_cell[cell] and right_by_cell[cell] for cell in all_cells):
            boot = spearmanr(
                [float(np.median(left_by_cell[cell])) for cell in all_cells],
                [float(np.median(right_by_cell[cell])) for cell in all_cells],
            )
            if np.isfinite(boot.statistic):
                pooled_bootstrap.append(float(boot.statistic))
    if pooled_bootstrap:
        pooled_low, pooled_high = np.quantile(
            pooled_bootstrap, [0.025, 0.975], method="linear"
        )
        pooled_interval = float(pooled_low), float(pooled_high)
    else:
        pooled_interval = None, None
    cross_rows.append(
        {
            "condition": "ALL_NONIDEAL",
            "exploratory_spearman_ci95_high": pooled_interval[1],
            "exploratory_spearman_ci95_low": pooled_interval[0],
            "pvalue_descriptive": pooled_pvalue,
            "scope": "POOLED_SCENE_CONDITION",
            "scene_count": 35,
            "spearman_rho": pooled_rho,
        }
    )
    cross_pass = bool(
        rhos["INDEPENDENT_NOISE_FREE"] is not None and rhos["INDEPENDENT_NOISE_FREE"] >= .7
        and rhos["FULL_NOISE"] is not None and rhos["FULL_NOISE"] >= .7
        and sum(value is not None and value >= .5 for value in rhos.values()) >= 4
        and all(value is not None for value in rhos.values()) and np.median(list(rhos.values())) >= .7
        and pooled_rho is not None and pooled_rho >= .75
    )

    trial_by_id = {row["planned_trial_id"]: row for row in trials}
    common_by_id = {row["planned_trial_id"]: row for row in common_records if row.get("planned_trial_id") in trial_by_id}
    joined_all = []
    joined_seen = set()
    for common in common_records:
        trial_id = str(common.get("planned_trial_id"))
        if trial_id in joined_seen or trial_id not in trial_by_id:
            continue
        joined_seen.add(trial_id)
        trial = trial_by_id[trial_id]
        merged_all = dict(common)
        for name in (
            "backend", "backend_schema_name", "condition", "geometry_seed",
            "measurement_seed", "repeat_index", "scene_variant", "snapshot_id",
            "translation_error_m", "rotation_error_rad",
        ):
            merged_all[name] = trial[name]
        joined_all.append(merged_all)
    joined_all.sort(key=lambda row: str(row["planned_trial_id"]))
    eligible = {
        trial_id: row for trial_id, row in trial_by_id.items()
        if row["condition"] in NONIDEAL_CONDITIONS
        and not row["solver_failure"] and row["finite_output"]
    }
    invalid_reasons = Counter()
    joined = []
    for trial_id, trial in eligible.items():
        common = common_by_id.get(trial_id)
        if common is None or common.get("common_association_valid") is not True:
            reason = "OTHER" if common is None else str(common.get("common_association_invalid_reason", "OTHER"))
            invalid_reasons[reason] += 1
            continue
        merged = dict(common)
        merged.update(
            {
                "backend": trial["backend_schema_name"],
                "backend_schema_name": trial["backend_schema_name"],
                "condition": trial["condition"],
                "geometry_seed": trial["geometry_seed"],
                "measurement_seed": trial["measurement_seed"],
                "repeat_index": trial["repeat_index"],
                "rotation_error_rad": trial["rotation_error_rad"],
                "scene_variant": trial["scene_variant"],
                "snapshot_id": trial["snapshot_id"],
                "translation_error_m": trial["translation_error_m"],
            }
        )
        joined.append(merged)
    joined.sort(key=lambda row: str(row["planned_trial_id"]))
    valid_fraction = len(joined) / len(eligible) if eligible else 0.0
    association_validity = {
        "COMMON_ASSOCIATION_ANALYSIS_PASS": bool(eligible and valid_fraction >= .95),
        "eligible_successful_nonideal_trial_count": len(eligible),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "missing_or_invalid_record_count": len(eligible) - len(joined),
        "valid_common_association_record_count": len(joined),
        "valid_fraction": valid_fraction,
    }
    turnover_rows = []
    pooled_turnover_rhos = {}
    centered_turnover_rhos = {}
    for backend in BACKENDS:
        chosen = [
            row for row in joined if row["backend_schema_name"] == backend
            and _finite(row.get("correspondence_turnover")) is not None
            and _finite(row.get("translation_error_m")) is not None
        ]
        turnover = np.asarray([float(row["correspondence_turnover"]) for row in chosen])
        error = np.asarray([math.log10(float(row["translation_error_m"]) + 1e-9) for row in chosen])
        result = spearmanr(turnover, error)
        pooled = float(result.statistic) if np.isfinite(result.statistic) else None
        pooled_pvalue = float(result.pvalue) if np.isfinite(result.pvalue) else None
        by_cell = defaultdict(list)
        for row in chosen:
            by_cell[(row["scene_variant"], row["condition"])].append(row)
        centered_turnover = []
        centered_error = []
        centered_records = []
        for cell in by_cell.values():
            t = np.asarray([float(row["correspondence_turnover"]) for row in cell])
            e = np.asarray([math.log10(float(row["translation_error_m"]) + 1e-9) for row in cell])
            centered_turnover.extend((t - np.median(t)).tolist())
            centered_error.extend((e - np.median(e)).tolist())
            for row, turnover_value, error_value in zip(cell, t, e):
                centered_records.append(
                    {
                        "geometry_seed": int(row["geometry_seed"]),
                        "measurement_seed": int(row["measurement_seed"]),
                        "repeat_index": int(row["repeat_index"]),
                        "left": float(turnover_value - np.median(t)),
                        "right": float(error_value - np.median(e)),
                    }
                )
        centered_result = spearmanr(centered_turnover, centered_error)
        centered = float(centered_result.statistic) if np.isfinite(centered_result.statistic) else None
        centered_pvalue = (
            float(centered_result.pvalue)
            if np.isfinite(centered_result.pvalue)
            else None
        )
        pooled_records = [
            {
                "geometry_seed": int(row["geometry_seed"]),
                "measurement_seed": int(row["measurement_seed"]),
                "repeat_index": int(row["repeat_index"]),
                "left": float(row["correspondence_turnover"]),
                "right": math.log10(float(row["translation_error_m"]) + 1.0e-9),
            }
            for row in chosen
        ]
        pooled_low, pooled_high = _independent_interval(
            pooled_records,
            lambda sample: _independent_rho(
                [row["left"] for row in sample],
                [row["right"] for row in sample],
            ),
            repetitions=bootstrap_repetitions,
        )
        centered_low, centered_high = _independent_interval(
            centered_records,
            lambda sample: _independent_rho(
                [row["left"] for row in sample],
                [row["right"] for row in sample],
            ),
            repetitions=bootstrap_repetitions,
        )
        pooled_turnover_rhos[backend] = pooled
        centered_turnover_rhos[backend] = centered
        turnover_rows.append(
            {
                "backend_schema_name": backend,
                "centered_exploratory_ci95_high": centered_high,
                "centered_exploratory_ci95_low": centered_low,
                "centered_pvalue_descriptive": centered_pvalue,
                "centered_spearman_rho": centered,
                "pooled_exploratory_ci95_high": pooled_high,
                "pooled_exploratory_ci95_low": pooled_low,
                "pooled_pvalue_descriptive": pooled_pvalue,
                "pooled_spearman_rho": pooled,
                "valid_trial_count": len(chosen),
            }
        )
    centered_values = [centered_turnover_rhos.get(backend) for backend in BACKENDS]
    mechanism_pass = bool(
        all(pooled_turnover_rhos.get(backend) is not None and pooled_turnover_rhos[backend] >= .4 for backend in BACKENDS)
        and all(value is not None for value in centered_values)
        and max(centered_values) >= .2 and min(centered_values) >= 0
    )

    model_comparisons = _independent_model_comparisons(joined)
    candidates = []
    for backend in BACKENDS:
        candidates.extend(_independent_candidates(joined, backend=backend))
    incremental_pass, incremental = _independent_incremental_gate(
        model_comparisons, candidates
    )

    systematic_combo_passes = []
    systematic_gate_detail = []
    for condition in MAIN_CONDITIONS:
        for backend in BACKENDS:
            chosen = [
                row for row in systematic
                if row["scene_variant"] == "LONG_CORRIDOR"
                and row["condition"] == condition and row["backend_schema_name"] == backend
            ]
            qualified = sum(
                row["successful_observation_count"] == 10
                and row["systematic_translation_offset_m"] is not None
                and row["systematic_translation_offset_m"] >= .005
                and row["systematic_fraction_translation"] is not None
                and row["systematic_fraction_translation"] >= .6
                for row in chosen
            )
            fractions = [
                row["systematic_fraction_translation"] for row in chosen
                if row["successful_observation_count"] == 10
                and row["systematic_fraction_translation"] is not None
            ]
            median_fraction = float(np.median(fractions)) if fractions else None
            combination_pass = bool(
                len(chosen) == 3
                and qualified >= 2
                and len(fractions) == 3
                and median_fraction is not None
                and median_fraction >= .7
            )
            systematic_combo_passes.append(combination_pass)
            systematic_gate_detail.append(
                {
                    "backend_schema_name": backend,
                    "condition": condition,
                    "gate_pass": combination_pass,
                    "geometry_group_count": len(chosen),
                    "median_systematic_fraction_translation": median_fraction,
                    "qualified_geometry_group_count": qualified,
                    "scene_variant": "LONG_CORRIDOR",
                }
            )
    systematic_claim = len(systematic_combo_passes) == 4 and all(systematic_combo_passes)

    robustness_groups = []
    success_group_rows = []
    for backend in BACKENDS:
        for condition in CONDITIONS:
            selected = [row for row in trials if row["backend_schema_name"] == backend and row["condition"] == condition]
            success = sum(not row["solver_failure"] and row["finite_output"] for row in selected)
            condition_pass = len(selected) == 210 and success / 210 >= .95
            robustness_groups.append(condition_pass)
            success_group_rows.append(
                {
                    "backend_schema_name": backend,
                    "condition": condition,
                    "gate_level": "BACKEND_CONDITION",
                    "gate_pass": condition_pass,
                    "planned_count": 210,
                    "scene_variant": "ALL_SCENES",
                    "success_count": success,
                    "success_rate": success / 210.0,
                }
            )
            for scene in SCENES:
                cell = [row for row in selected if row["scene_variant"] == scene]
                cell_success = sum(not row["solver_failure"] and row["finite_output"] for row in cell)
                cell_pass = len(cell) == 30 and cell_success / 30 >= .9
                robustness_groups.append(cell_pass)
                success_group_rows.append(
                    {
                        "backend_schema_name": backend,
                        "condition": condition,
                        "gate_level": "BACKEND_CONDITION_SCENE",
                        "gate_pass": cell_pass,
                        "planned_count": 30,
                        "scene_variant": scene,
                        "success_count": cell_success,
                        "success_rate": cell_success / 30.0,
                    }
                )
    robustness = all(robustness_groups) and len(robustness_groups) == 96

    backend_counts = Counter(row["backend_schema_name"] for row in trials)
    backend_exceptions = sum(row["failure_classification"] == "BACKEND_EXCEPTION" for row in trials)
    nonfinite = sum(not row["finite_output"] for row in trials)
    solver_failures = sum(row["solver_failure"] for row in trials)
    engineering = bool(
        integrity.get("PHASE_A_IDEAL_IMPORT_PASS") is True
        and integrity.get("PHASE_B_SUBSET_REPRODUCTION_PASS") is True
        and integrity.get("new_snapshot_count") == 1050
        and integrity.get("new_trial_count") == 2100
        and integrity.get("combined_snapshot_count") == 1260
        and integrity.get("combined_trial_count") == 2520
        and backend_counts[OPEN3D_BACKEND] == backend_counts[PCL_BACKEND] == 1260
        and all(int(integrity.get(name, -1)) == 0 for name in (
            "missing_trial_count", "extra_trial_count", "duplicate_trial_count", "corrupt_trial_count",
            "backend_input_checksum_mismatch_count", "infrastructure_interruption_unresolved_count",
        ))
        and backend_exceptions == 0 and nonfinite == 0
    )
    gates = {
        "COMMON_ASSOCIATION_ANALYSIS_PASS": association_validity["COMMON_ASSOCIATION_ANALYSIS_PASS"],
        "FULL_SYNTHETIC_CROSS_BACKEND_PASS": cross_pass,
        "FULL_SYNTHETIC_DEVELOPMENT_ENGINEERING_PASS": engineering,
        "FULL_SYNTHETIC_EXECUTION_ROBUSTNESS_PASS": robustness,
        "FULL_SYNTHETIC_PRIMARY_SCENE_EFFECT_PASS": len(effect_passes) == 4 and all(effect_passes),
        "FULL_SYNTHETIC_SCENE_RANK_STABILITY_PASS": rich_rank_pass_count >= 8 and weak_rank_pass_count >= 8,
        "LOCAL_METRIC_INCREMENTAL_VALUE_PASS": incremental_pass,
        "REASSOCIATION_MECHANISM_SUPPORTED": mechanism_pass,
        "SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED": systematic_claim,
    }
    required = (
        "FULL_SYNTHETIC_DEVELOPMENT_ENGINEERING_PASS",
        "FULL_SYNTHETIC_EXECUTION_ROBUSTNESS_PASS",
        "FULL_SYNTHETIC_PRIMARY_SCENE_EFFECT_PASS",
        "FULL_SYNTHETIC_CROSS_BACKEND_PASS",
        "FULL_SYNTHETIC_SCENE_RANK_STABILITY_PASS",
        "COMMON_ASSOCIATION_ANALYSIS_PASS",
        "REASSOCIATION_MECHANISM_SUPPORTED",
        "LOCAL_METRIC_INCREMENTAL_VALUE_PASS",
    )
    if len(required) != 8 or any(name not in gates for name in required):
        raise AssertionError("independent eight-gate conjunction changed")
    development_pass = all(gates[name] for name in required)
    final = {
        **gates,
        "CONFIRMATORY_PROTOCOL_DESIGN_AUTHORIZED": development_pass,
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "FULL_SYNTHETIC_DEVELOPMENT_PASS": development_pass,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PHENOMENON_CONFIRMED_BUT_INCREMENTAL_VALUE_NOT_ESTABLISHED": bool(
            not incremental_pass and all(gates[name] for name in required if name != "LOCAL_METRIC_INCREMENTAL_VALUE_PASS")
        ),
        "REAL_DATA_PROTOCOL_DESIGN_AUTHORIZED": development_pass,
        "REAL_DATA_RUN_AUTHORIZED": False,
    }
    if (
        final["CONFIRMATORY_PROTOCOL_DESIGN_AUTHORIZED"]
        is not final["FULL_SYNTHETIC_DEVELOPMENT_PASS"]
        or final["REAL_DATA_PROTOCOL_DESIGN_AUTHORIZED"]
        is not final["FULL_SYNTHETIC_DEVELOPMENT_PASS"]
        or final["CONFIRMATORY_RUN_AUTHORIZED"] is not False
        or final["REAL_DATA_RUN_AUTHORIZED"] is not False
        or final["MEASUREMENT_PAPER_MAINLINE_AUTHORIZED"] is not False
    ):
        raise AssertionError("independent authorization mirror changed")
    failure_inventory = []
    for backend in BACKENDS:
        for condition in CONDITIONS:
            selected = [row for row in trials if row["backend_schema_name"] == backend and row["condition"] == condition]
            counts = Counter(row["failure_classification"] for row in selected)
            for classification in sorted(counts):
                failure_inventory.append(
                    {
                        "backend": BACKEND_LABELS[backend],
                        "backend_schema_name": backend,
                        "condition": condition,
                        "failure_classification": classification,
                        "failure_count": counts[classification],
                        "nonfinite_output_count": sum(not row["finite_output"] for row in selected),
                        "solver_failure_count": sum(row["solver_failure"] for row in selected),
                    }
                )
    candidate_counts = Counter(row["backend"] for row in candidates)
    candidate_pairs = sorted({tuple(sorted((row["scene_a"], row["scene_b"]))) for row in candidates})
    canonical_candidates = []
    for source in candidates:
        row = dict(source)
        row.pop("backend", None)
        canonical_candidates.append(row)
    canonical_candidates.sort(key=lambda row: str(row["candidate_pair_id"]))
    common_for_hash = []
    for source in joined_all:
        row = dict(source)
        row.pop("backend", None)
        common_for_hash.append(row)
    common_for_hash.sort(key=lambda row: str(row["planned_trial_id"]))
    trial_fields = (
        "planned_trial_id", "snapshot_id", "scene_variant", "condition",
        "backend_schema_name", "geometry_seed", "measurement_seed", "repeat_index",
        "solver_failure", "failure_classification", "finite_output",
        "translation_error_m", "rotation_error_rad", "translation_vector",
        "rotation_vector", "runtime_ms", "source_checksum", "target_checksum",
        "reference_pose_checksum", "snapshot_checksum",
    )
    normalized_trials = [
        {name: row.get(name) for name in trial_fields} for row in trials
    ]
    normalized_trials.sort(key=lambda row: str(row["planned_trial_id"]))
    runtime_summary = []
    for backend in BACKENDS:
        values = np.asarray(
            [float(row["runtime_ms"]) for row in trials if row["backend_schema_name"] == backend],
            dtype=np.float64,
        )
        runtime_summary.append(
            {
                "backend_schema_name": backend,
                "maximum_runtime_ms": float(np.max(values)) if values.size else None,
                "median_runtime_ms": float(np.median(values)) if values.size else None,
                "q95_runtime_ms": float(np.quantile(values, .95, method="linear")) if values.size else None,
                "trial_count": int(values.size),
            }
        )
    return {
        "association_validity": association_validity,
        "automatic_candidate_count_by_backend": {backend: candidate_counts[backend] for backend in BACKENDS},
        "automatic_candidate_scene_pairs": candidate_pairs,
        "automatic_nonequivalence_candidates": canonical_candidates,
        "backend_exception_count": backend_exceptions,
        "cross_backend_ranking": cross_rows,
        "condition_contrasts": condition_contrasts,
        "failure_inventory": failure_inventory,
        "final_decision": final,
        "gate_summary": gates,
        "integrity": dict(integrity),
        "incremental_value_detail": incremental,
        "joined_common_association_metrics": common_for_hash,
        "nonfinite_output_count": nonfinite,
        "normalized_trial_metrics": normalized_trials,
        "ridge_model_comparison": model_comparisons,
        "runtime_summary": runtime_summary,
        "scene_condition_backend_summary": scene_rows,
        "scene_effect_paired": effect_rows,
        "scene_rank_stability": rank_rows,
        "solver_failure_count": solver_failures,
        "success_rate_groups": success_group_rows,
        "systematic_claim_detail": systematic_gate_detail,
        "systematic_offset_summary": systematic,
        "turnover_correlations": turnover_rows,
    }


VERIFICATION_SECTIONS = (
    "association_validity",
    "automatic_candidate_count_by_backend",
    "automatic_candidate_scene_pairs",
    "automatic_nonequivalence_candidates",
    "backend_exception_count",
    "condition_contrasts",
    "cross_backend_ranking",
    "failure_inventory",
    "final_decision",
    "gate_summary",
    "integrity",
    "incremental_value_detail",
    "joined_common_association_metrics",
    "nonfinite_output_count",
    "normalized_trial_metrics",
    "ridge_model_comparison",
    "runtime_summary",
    "scene_condition_backend_summary",
    "scene_effect_paired",
    "scene_rank_stability",
    "solver_failure_count",
    "success_rate_groups",
    "systematic_claim_detail",
    "systematic_offset_summary",
    "turnover_correlations",
)


ANALYSIS_ABSOLUTE_TOLERANCE = 1.0e-12
ANALYSIS_RELATIVE_TOLERANCE = 1.0e-12


def _deep_numeric_comparison(left: Any, right: Any) -> tuple[int, float]:
    """Compare IDs/discrete values exactly and continuous values at 1e-12."""

    if isinstance(left, bool) or isinstance(right, bool):
        return (0, 0.0) if type(left) is type(right) and left is right else (1, 0.0)
    if isinstance(left, float) or isinstance(right, float):
        if not isinstance(left, float) or not isinstance(right, float):
            return 1, 0.0
        left_value, right_value = left, right
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            return ((0, 0.0) if left_value == right_value else (1, float("inf")))
        difference = abs(left_value - right_value)
        tolerance = ANALYSIS_ABSOLUTE_TOLERANCE + ANALYSIS_RELATIVE_TOLERANCE * max(
            abs(left_value), abs(right_value)
        )
        return (0 if difference <= tolerance else 1), difference
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return 1, 0.0
        if set(left) != set(right):
            return 1, 0.0
        differences = 0
        maximum = 0.0
        for key in sorted(left, key=str):
            count, value = _deep_numeric_comparison(left[key], right[key])
            differences += count
            maximum = max(maximum, value)
        return differences, maximum
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if (
            not isinstance(left, (list, tuple))
            or not isinstance(right, (list, tuple))
            or len(left) != len(right)
        ):
            return 1, 0.0
        differences = 0
        maximum = 0.0
        for left_item, right_item in zip(left, right):
            count, value = _deep_numeric_comparison(left_item, right_item)
            differences += count
            maximum = max(maximum, value)
        return differences, maximum
    return (0, 0.0) if type(left) is type(right) and left == right else (1, 0.0)


def full_synthetic_analysis_verifier_comparison(
    primary: Mapping[str, Any], independent: Mapping[str, Any]
) -> dict[str, Any]:
    expected = full_synthetic_verification_projection(primary)
    actual = independent.get("verification_projection", independent)
    section_differences = 0
    leaf_differences = 0
    maximum = 0.0
    differing_sections = []
    for name in VERIFICATION_SECTIONS:
        count, value = _deep_numeric_comparison(expected.get(name), actual.get(name))
        if count:
            section_differences += 1
            differing_sections.append(name)
        leaf_differences += count
        maximum = max(maximum, value)
    return {
        "absolute_tolerance": ANALYSIS_ABSOLUTE_TOLERANCE,
        "differing_sections": differing_sections,
        "leaf_difference_count": leaf_differences,
        "maximum_absolute_numeric_difference": maximum,
        "relative_tolerance": ANALYSIS_RELATIVE_TOLERANCE,
        "section_difference_count": section_differences,
    }


def full_synthetic_analysis_verifier_difference_count(
    primary: Mapping[str, Any], independent: Mapping[str, Any]
) -> int:
    return int(
        full_synthetic_analysis_verifier_comparison(primary, independent)[
            "section_difference_count"
        ]
    )


def independently_recompute_common_association_records(
    *,
    manifest_path: str | Path,
    trials: Sequence[Mapping[str, Any]],
    cached_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild all common metrics from locked arrays and reject a re-signed tamper."""

    from .common_association_analysis import (
        prepare_common_association_context,
        safe_analyze_estimated_transform,
    )
    from .full_synthetic_snapshot_builder import read_full_synthetic_snapshot

    manifest_file, manifest = load_manifest(manifest_path, require_authorized=True)
    root = manifest_root(manifest_file)
    snapshot_path = _root_path(
        root,
        manifest,
        ("new_planned_snapshots_path", "planned_snapshots_path"),
        "frozen_assets/full_synthetic_development_planned_snapshots_v1.csv",
    )
    trial_path = _root_path(
        root,
        manifest,
        ("new_planned_trials_path", "planned_trials_path"),
        "frozen_assets/full_synthetic_development_planned_trials_v1.csv",
    )
    snapshot_plans = _csv(snapshot_path)
    trial_plans = _csv(trial_path)
    bindings = _independent_new_bindings(
        root, manifest, snapshot_plans, trial_plans
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    result_ids = []
    for row in trials:
        if row["condition"] in NONIDEAL_CONDITIONS:
            grouped[str(row["snapshot_id"])].append(row)
            result_ids.append(str(row["planned_trial_id"]))
    if (
        len(result_ids) != len(set(result_ids))
        or set(result_ids) != set(bindings["trial_by_id"])
        or set(grouped) != set(bindings["snapshot_by_id"])
    ):
        raise ValueError("independent common recomputation plan inventory mismatch")
    recomputed = []
    for snapshot_id in sorted(grouped):
        selected = grouped[snapshot_id]
        if (
            len(selected) != 2
            or Counter(row["backend_schema_name"] for row in selected)
            != Counter(BACKENDS)
        ):
            raise ValueError("independent common backend pairing mismatch")
        plan = bindings["snapshot_by_id"][snapshot_id]
        lock_entry = bindings["lock_by_id"][snapshot_id]
        snapshot = read_full_synthetic_snapshot(
            bindings["cache_root"],
            plan,
            expected_lock_entry=lock_entry,
            arrays=True,
        )
        for trial in selected:
            planned = bindings["trial_by_id"].get(str(trial["planned_trial_id"]))
            if planned is None:
                raise ValueError("independent common trial plan join failed")
            expected_identity = {
                "snapshot_id": planned["snapshot_id"],
                "scene_variant": planned["scene_variant"],
                "condition": planned["condition"],
                "backend_schema_name": _backend(planned["backend"]),
                "geometry_seed": int(planned["geometry_seed_value"]),
                "measurement_seed": int(planned["measurement_seed_value"]),
                "repeat_index": int(planned["repeat_index"]),
            }
            if any(trial.get(name) != value for name, value in expected_identity.items()):
                raise ValueError("independent common trial/plan identity mismatch")
            if (
                any(
                    trial.get(name) != snapshot[name]
                    for name in (
                        "source_checksum",
                        "target_checksum",
                        "reference_pose_checksum",
                        "snapshot_checksum",
                    )
                )
                or trial.get("snapshot_lock_sha256") != bindings["lock_sha256"]
            ):
                raise ValueError("independent common trial/snapshot binding mismatch")
        try:
            context = prepare_common_association_context(
                snapshot["source"],
                snapshot["target"],
                snapshot["reference"],
                snapshot_id=snapshot_id,
            )
        except Exception as error:
            for trial in selected:
                if trial["solver_failure"] or not trial["finite_output"]:
                    continue
                recomputed.append(
                    {
                        "backend_schema_name": trial["backend_schema_name"],
                        "common_association_invalid_detail": f"{type(error).__name__}: {error}",
                        "common_association_invalid_reason": "OTHER",
                        "common_association_is_backend_internal": False,
                        "common_association_valid": False,
                        "condition": trial["condition"],
                        "planned_trial_id": trial["planned_trial_id"],
                        "snapshot_id": snapshot_id,
                    }
                )
            continue
        for trial in sorted(selected, key=lambda row: str(row["planned_trial_id"])):
            if trial["solver_failure"] or not trial["finite_output"]:
                continue
            identifiers = {
                "backend_schema_name": trial["backend_schema_name"],
                "condition": trial["condition"],
                "geometry_seed": int(trial["geometry_seed"]),
                "measurement_seed": int(trial["measurement_seed"]),
                "planned_trial_id": trial["planned_trial_id"],
                "repeat_index": int(trial["repeat_index"]),
                "scene_variant": trial["scene_variant"],
            }
            recomputed.append(
                safe_analyze_estimated_transform(
                    context,
                    np.asarray(trial["final_transform_4x4"], dtype=np.float64),
                    identifiers=identifiers,
                )
            )
    recomputed.sort(key=lambda row: str(row["planned_trial_id"]))
    cached = [dict(row) for row in cached_records]
    cached.sort(key=lambda row: str(row.get("planned_trial_id")))
    cache_difference_count, _ = _deep_numeric_comparison(recomputed, cached)
    if cache_difference_count:
        raise ValueError(
            "common association cache differs from independent snapshot recomputation"
        )
    return recomputed


def independently_verify_full_synthetic_development(
    *,
    manifest_path: str | Path,
    run_dir: str | Path,
    common_metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    trials, integrity = independently_load_full_synthetic_evidence(
        manifest_path=manifest_path, run_dir=run_dir
    )
    common_path = (
        Path(common_metrics_path)
        if common_metrics_path is not None
        else Path(run_dir) / "common_association_metrics.json"
    )
    cached_common = validate_common_association_cache(
        manifest_path=manifest_path,
        run_dir=run_dir,
        trials=trials,
        cache_path=common_path,
    )
    common = independently_recompute_common_association_records(
        manifest_path=manifest_path,
        trials=trials,
        cached_records=cached_common,
    )
    projection = independently_recompute_full_synthetic(
        trials=trials, common_records=common, integrity=integrity
    )
    report = {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": None,
        "final_decision": projection["final_decision"],
        "raw_trial_count_recomputed": len(trials),
        "run_id": "full-synthetic-development-v1",
        "schema_version": "full_synthetic_development_independent_verification_v1",
        "verification_projection": projection,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


__all__ = [
    "ANALYSIS_ABSOLUTE_TOLERANCE",
    "ANALYSIS_RELATIVE_TOLERANCE",
    "VERIFICATION_SECTIONS",
    "full_synthetic_analysis_verifier_comparison",
    "full_synthetic_analysis_verifier_difference_count",
    "independently_load_full_synthetic_evidence",
    "independently_recompute_common_association_records",
    "independently_recompute_full_synthetic",
    "independently_verify_full_synthetic_development",
]
