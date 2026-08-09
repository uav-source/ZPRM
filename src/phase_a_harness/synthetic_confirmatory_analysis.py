"""Frozen H1--H6 analysis for Synthetic Confirmatory v1.

The record-level API is deliberately independent of filesystem execution so it
can be qualified with seed-free, hand-built fixtures.  The filesystem API is
strict and read-only: a missing or incomplete formal result set raises before
any report is written.
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
from scipy.stats import rankdata

from .common_association_analysis import (
    prepare_common_association_context,
    safe_analyze_estimated_transform,
)
from .phase_a_trial_result_schema import validate_phase_a_trial_result_strict
from .rotation_metrics import rotation_metric_audit
from .synthetic_confirmatory_snapshot_builder import (
    read_synthetic_confirmatory_snapshot,
)


SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
CONDITIONS = ("IDEAL_MATCHED", "INDEPENDENT_NOISE_FREE", "FULL_NOISE")
BACKENDS = ("open3d_point_to_plane", "pcl_point_to_plane")
FORMAL_RUN_ID = "synthetic-confirmatory-v1"
RAW_MANIFEST_SCHEMA = "synthetic_confirmatory_raw_result_manifest_v1"
TARGET_EPSILON_M = 1.0e-9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"headerless CSV: {path}")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"invalid CSV: {path}") from error


def _root_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes standalone repository")
    return candidate


def _trial_validator(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or value.get("condition") not in CONDITIONS:
        raise ValueError("Confirmatory trial condition is unauthorized")
    condition = value["condition"]
    normalized = dict(value)
    normalized["condition"] = "IDEAL_MATCHED"
    checked = validate_phase_a_trial_result_strict(normalized)
    checked["condition"] = condition
    return checked


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def _q95(values: Sequence[float]) -> float | None:
    return (
        float(np.quantile(np.asarray(values, dtype=np.float64), 0.95, method="linear"))
        if values else None
    )


def _spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return None
    x = np.asarray(rankdata(x_values, method="average"), dtype=np.float64)
    y = np.asarray(rankdata(y_values, method="average"), dtype=np.float64)
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def _backend(row: Mapping[str, Any]) -> str:
    value = str(row.get("backend_schema_name", row.get("backend", "")))
    if value not in BACKENDS:
        raise ValueError(f"unauthorized backend: {value}")
    return value


def _normalize_trials(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for source in trials:
        row = dict(source)
        planned_trial_id = str(row.get("planned_trial_id", ""))
        snapshot_id = str(row.get("planned_snapshot_id", row.get("snapshot_id", "")))
        if not planned_trial_id or not snapshot_id:
            raise ValueError("trial identity is missing")
        condition = str(row.get("condition", ""))
        scene = str(row.get("scene_variant", ""))
        if condition not in CONDITIONS or scene not in SCENES:
            raise ValueError("trial scene/condition is outside the frozen protocol")
        geometry_seed = int(row["geometry_seed"])
        solver_failure = bool(row.get("solver_failure", False))
        finite_output = bool(row.get("finite_output", True))
        translation = _float(row.get("translation_error_m", row.get("translation_update_m")))
        rotation = _float(row.get("rotation_error_rad", row.get("rotation_update_rad")))
        vector = row.get("translation_vector")
        if vector is not None:
            vector_array = np.asarray(vector, dtype=np.float64)
            if vector_array.shape != (3,) or not np.all(np.isfinite(vector_array)):
                raise ValueError("translation_vector must be finite length three")
            vector = vector_array.astype(float).tolist()
        rows.append(
            {
                **row,
                "backend_schema_name": _backend(row),
                "condition": condition,
                "finite_output": finite_output,
                "geometry_seed": geometry_seed,
                "measurement_seed": (
                    None if row.get("measurement_seed") in (None, "")
                    else int(row["measurement_seed"])
                ),
                "planned_snapshot_id": snapshot_id,
                "planned_trial_id": planned_trial_id,
                "repeat_index": int(row.get("repeat_index", 0)),
                "rotation_error_rad": rotation,
                "scene_variant": scene,
                "solver_failure": solver_failure,
                "translation_error_m": translation,
                "translation_vector": vector,
            }
        )
        ids.append(planned_trial_id)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate planned_trial_id")
    return sorted(rows, key=lambda row: row["planned_trial_id"])


def _normalize_common(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in records]
    ids = [str(row.get("planned_trial_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("common-association record IDs are missing or duplicated")
    for row in rows:
        row["backend_schema_name"] = _backend(row)
    return sorted(rows, key=lambda row: str(row["planned_trial_id"]))


def _feature_value(row: Mapping[str, Any], name: str) -> float:
    if name == "log10_initial_residual_rmse_plus_1e-9":
        value = _float(row.get("initial_residual_rmse"))
        return math.log10(float(value) + 1.0e-9) if value is not None and value >= 0 else math.nan
    if name == "log10_condition_number_trans_plus_1":
        value = _float(row.get("condition_number_trans"))
        return math.log10(float(value) + 1.0) if value is not None and value >= 0 else math.nan
    if name == "log10_inverse_lambda_min_trans":
        value = _float(row.get("lambda_min_trans"))
        return math.log10(1.0 / max(float(value), 1.0e-12)) if value is not None else math.nan
    if name == "log10_initial_correspondence_count_plus_1":
        value = _float(row.get("initial_correspondence_count"))
        return math.log10(float(value) + 1.0) if value is not None and value >= 0 else math.nan
    value = _float(row.get(name))
    return float(value) if value is not None else math.nan


def _predict(model: Mapping[str, Any], row: Mapping[str, Any]) -> float | None:
    names = model.get("feature_names")
    if type(names) is not list:
        return None
    vector = np.asarray([_feature_value(row, str(name)) for name in names], dtype=np.float64)
    mean = np.asarray(model.get("scaler_mean"), dtype=np.float64)
    scale = np.asarray(model.get("scaler_scale"), dtype=np.float64)
    coefficient = np.asarray(model.get("coefficient"), dtype=np.float64)
    if (
        vector.shape != mean.shape
        or vector.shape != scale.shape
        or vector.shape != coefficient.shape
        or not np.all(np.isfinite(vector))
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or not np.all(np.isfinite(coefficient))
        or np.any(scale <= 0.0)
        or _float(model.get("intercept")) is None
    ):
        return None
    return float(np.dot((vector - mean) / scale, coefficient) + float(model["intercept"]))


def analyze_synthetic_confirmatory_records(
    *,
    trials: Sequence[Mapping[str, Any]],
    common_records: Sequence[Mapping[str, Any]],
    model_lock: Mapping[str, Any],
    gate_contract: Mapping[str, Any],
    expected_geometry_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate the exact H1--H6 conjunction from normalized raw evidence."""

    rows = _normalize_trials(trials)
    common = _normalize_common(common_records)
    geometries = tuple(
        int(value) for value in (
            expected_geometry_seeds
            if expected_geometry_seeds is not None
            else sorted({row["geometry_seed"] for row in rows})
        )
    )
    if len(geometries) != 5 or len(set(geometries)) != 5:
        raise ValueError("Confirmatory analysis requires exactly five geometry seeds")
    hypotheses = gate_contract.get("hypotheses")
    if type(hypotheses) is not dict or set(hypotheses) != {
        "H1_IDEAL_CONTROL", "H2_LONG_CORRIDOR_SCENE_EFFECT",
        "H3_CROSS_BACKEND_SCENE_RANKING", "H4_REASSOCIATION_MECHANISM",
        "H5_FROZEN_MODEL_B_INCREMENTAL_VALUE",
        "H6_FULL_NOISE_SYSTEMATIC_OFFSET",
    }:
        raise ValueError("H1--H6 gate contract identity changed")

    expected_cell_counts = {
        "IDEAL_MATCHED": 1,
        "INDEPENDENT_NOISE_FREE": 1,
        "FULL_NOISE": 15,
    }
    cell_counts = Counter(
        (row["backend_schema_name"], row["scene_variant"], row["geometry_seed"], row["condition"])
        for row in rows
    )
    exact_plan = bool(
        len(rows) == 1190
        and set(cell_counts)
        == {
            (backend, scene, geometry, condition)
            for backend in BACKENDS for scene in SCENES for geometry in geometries
            for condition in CONDITIONS
        }
        and all(
            count == expected_cell_counts[key[3]] for key, count in cell_counts.items()
        )
    )
    success = lambda row: (
        not row["solver_failure"] and row["finite_output"]
        and row["translation_error_m"] is not None
        and row["rotation_error_rad"] is not None
    )

    h1_contract = hypotheses["H1_IDEAL_CONTROL"]
    h1_rows = []
    for backend in BACKENDS:
        selected = [row for row in rows if row["backend_schema_name"] == backend and row["condition"] == "IDEAL_MATCHED"]
        translations = [float(row["translation_error_m"]) for row in selected if success(row)]
        rotations = [float(row["rotation_error_rad"]) for row in selected if success(row)]
        solver_failures = sum(row["solver_failure"] for row in selected)
        nonfinite = sum(not row["finite_output"] for row in selected)
        translation_q95 = _q95(translations)
        rotation_q95 = _q95(rotations)
        passed = bool(
            len(selected) == 35 and len(translations) == 35
            and solver_failures <= h1_contract["solver_failure_count_max"]
            and nonfinite <= h1_contract["nonfinite_output_count_max"]
            and translation_q95 is not None
            and translation_q95 <= h1_contract["translation_q95_max_m"]
            and rotation_q95 is not None
            and rotation_q95 <= h1_contract["rotation_q95_max_rad"]
        )
        h1_rows.append({
            "backend_schema_name": backend, "planned_count": len(selected),
            "successful_count": len(translations), "solver_failure_count": solver_failures,
            "nonfinite_output_count": nonfinite, "translation_q95_m": translation_q95,
            "rotation_q95_rad": rotation_q95, "gate_pass": passed,
        })
    h1_pass = all(row["gate_pass"] for row in h1_rows)

    h2_contract = hypotheses["H2_LONG_CORRIDOR_SCENE_EFFECT"]
    h2_rows = []
    for backend in BACKENDS:
        for condition in h2_contract["conditions"]:
            geometry_rows = []
            for geometry in geometries:
                values: dict[str, list[float]] = {}
                for scene in ("GEOMETRY_RICH_ROOM", "LONG_CORRIDOR"):
                    values[scene] = [
                        float(row["translation_error_m"]) for row in rows
                        if row["backend_schema_name"] == backend
                        and row["condition"] == condition
                        and row["geometry_seed"] == geometry
                        and row["scene_variant"] == scene and success(row)
                    ]
                rich = _median(values["GEOMETRY_RICH_ROOM"])
                corridor = _median(values["LONG_CORRIDOR"])
                valid = rich is not None and corridor is not None
                ratio = (
                    (corridor / rich) if valid and rich > 0.0
                    else (float(np.finfo(np.float64).max) if valid and corridor > 0.0 else None)
                )
                difference = (corridor - rich) if valid else None
                geometry_rows.append({
                    "geometry_seed": geometry, "rich_median_m": rich,
                    "corridor_median_m": corridor, "ratio": ratio,
                    "difference_m": difference,
                    "corridor_greater": bool(valid and corridor > rich),
                })
            ratios = [float(row["ratio"]) for row in geometry_rows if row["ratio"] is not None]
            differences = [float(row["difference_m"]) for row in geometry_rows if row["difference_m"] is not None]
            wins = sum(row["corridor_greater"] for row in geometry_rows)
            median_ratio = _median(ratios)
            median_difference = _median(differences)
            passed = bool(
                len(ratios) == h2_contract["geometry_block_count"]
                and wins >= h2_contract["minimum_long_greater_than_rich_blocks"]
                and median_ratio is not None
                and median_ratio >= h2_contract["geometry_level_median_ratio_min"]
                and median_difference is not None
                and median_difference >= h2_contract["geometry_level_median_absolute_difference_min_m"]
            )
            h2_rows.append({
                "backend_schema_name": backend, "condition": condition,
                "geometry_rows": geometry_rows, "passing_geometry_count": wins,
                "median_geometry_ratio": median_ratio,
                "median_geometry_difference_m": median_difference, "gate_pass": passed,
            })
    h2_pass = len(h2_rows) == 4 and all(row["gate_pass"] for row in h2_rows)

    h3_contract = hypotheses["H3_CROSS_BACKEND_SCENE_RANKING"]
    h3_rows = []
    for condition in h3_contract["conditions"]:
        medians: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
        valid = True
        for backend in BACKENDS:
            for scene in SCENES:
                value = _median([
                    float(row["translation_error_m"]) for row in rows
                    if row["backend_schema_name"] == backend and row["condition"] == condition
                    and row["scene_variant"] == scene and success(row)
                ])
                if value is None:
                    valid = False
                else:
                    medians[backend].append(value)
        rho = _spearman(medians[BACKENDS[0]], medians[BACKENDS[1]]) if valid else None
        passed = bool(rho is not None and len(medians[BACKENDS[0]]) == h3_contract["scene_count"] and rho >= h3_contract["spearman_rho_min"])
        h3_rows.append({"condition": condition, "spearman_rho": rho, "scene_medians": medians, "gate_pass": passed})
    h3_pass = len(h3_rows) == 2 and all(row["gate_pass"] for row in h3_rows)

    common_by_id = {str(row["planned_trial_id"]): row for row in common}
    eligible_ids = {
        row["planned_trial_id"] for row in rows
        if row["condition"] != "IDEAL_MATCHED" and success(row)
    }
    common_inventory_pass = set(common_by_id) == eligible_ids
    joined = []
    for trial in rows:
        metric = common_by_id.get(trial["planned_trial_id"])
        if metric is not None:
            joined.append({**trial, **metric, "backend_schema_name": trial["backend_schema_name"]})

    h4_contract = hypotheses["H4_REASSOCIATION_MECHANISM"]
    h4_rows = []
    for backend in BACKENDS:
        selected = [
            row for row in joined if row["backend_schema_name"] == backend
            and row.get("common_association_valid") is True
            and _float(row.get("correspondence_turnover")) is not None
        ]
        x = [float(row["correspondence_turnover"]) for row in selected]
        y = [
            math.log10(float(row["translation_error_m"]) + TARGET_EPSILON_M)
            for row in selected
        ]
        pooled = _spearman(x, y)
        centered_x, centered_y = [], []
        by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_cell[(row["scene_variant"], row["condition"])].append(row)
        for cell_rows in by_cell.values():
            mx = float(np.median([
                float(row["correspondence_turnover"]) for row in cell_rows
            ]))
            my = float(np.median([
                math.log10(float(row["translation_error_m"]) + TARGET_EPSILON_M)
                for row in cell_rows
            ]))
            centered_x.extend(
                float(row["correspondence_turnover"]) - mx for row in cell_rows
            )
            centered_y.extend(
                math.log10(float(row["translation_error_m"]) + TARGET_EPSILON_M) - my
                for row in cell_rows
            )
        centered = _spearman(centered_x, centered_y)
        folds = []
        for held_out in geometries:
            training = [row for row in selected if row["geometry_seed"] != held_out]
            rho = _spearman(
                [float(row["correspondence_turnover"]) for row in training],
                [
                    math.log10(
                        float(row["translation_error_m"]) + TARGET_EPSILON_M
                    )
                    for row in training
                ],
            )
            folds.append({"held_out_geometry_seed": held_out, "spearman_rho": rho, "direction_positive": bool(rho is not None and rho > 0.0)})
        passed = bool(
            pooled is not None and pooled >= h4_contract["pooled_turnover_error_spearman_rho_min"]
            and centered is not None and centered >= h4_contract["scene_centered_spearman_rho_min"]
            and len(folds) == h4_contract["leave_one_geometry_seed_out_fold_count"]
            and all(row["direction_positive"] for row in folds)
        )
        h4_rows.append({
            "backend_schema_name": backend, "valid_trial_count": len(selected),
            "error_transform": "log10(translation_error_m+1e-9)",
            "centering_cell": ["scene_variant", "condition"],
            "pooled_spearman_rho": pooled, "scene_centered_spearman_rho": centered,
            "leave_one_geometry_out": folds, "gate_pass": passed,
        })
    h4_pass = common_inventory_pass and all(row["gate_pass"] for row in h4_rows)

    models = model_lock.get("models")
    if type(models) is not list:
        raise ValueError("frozen model lock has no model records")
    model_by_identity = {(str(row.get("backend_schema_name")), str(row.get("model"))): row for row in models}
    if set(model_by_identity) != {(backend, model) for backend in BACKENDS for model in ("A", "B")}:
        raise ValueError("frozen model identities changed")
    h5_contract = hypotheses["H5_FROZEN_MODEL_B_INCREMENTAL_VALUE"]
    h5_rows = []
    for backend in BACKENDS:
        selected = [
            row for row in rows
            if row["backend_schema_name"] == backend
            and row["condition"] != "IDEAL_MATCHED"
            and success(row)
        ]
        targets, prediction_a, prediction_b = [], [], []
        invalid_count = 0
        for trial in selected:
            metric = common_by_id.get(trial["planned_trial_id"])
            if metric is None:
                invalid_count += 1
                continue
            row = {
                **trial,
                **metric,
                "backend_schema_name": trial["backend_schema_name"],
            }
            if row.get("common_association_valid") is not True:
                invalid_count += 1
                continue
            error = _float(row.get("translation_error_m"))
            a = _predict(model_by_identity[(backend, "A")], row)
            b = _predict(model_by_identity[(backend, "B")], row)
            if error is not None and error >= 0.0 and a is not None and b is not None:
                targets.append(math.log10(error + TARGET_EPSILON_M))
                prediction_a.append(a)
                prediction_b.append(b)
            else:
                invalid_count += 1
        mae_a = float(np.mean(np.abs(np.asarray(prediction_a) - np.asarray(targets)))) if targets else None
        mae_b = float(np.mean(np.abs(np.asarray(prediction_b) - np.asarray(targets)))) if targets else None
        ratio = (mae_b / mae_a) if mae_a is not None and mae_a > 0.0 and mae_b is not None else None
        passed = bool(
            len(selected) > 0
            and invalid_count == 0
            and len(targets) == len(selected)
            and ratio is not None
            and ratio <= h5_contract["mae_b_to_mae_a_ratio_max"]
        )
        h5_rows.append({
            "backend_schema_name": backend,
            "candidate_count": len(selected),
            "evaluated_count": len(targets),
            "invalid_count": invalid_count,
            "model_a_mae": mae_a,
            "model_b_mae": mae_b,
            "mae_b_to_a_ratio": ratio,
            "gate_pass": passed,
        })
    h5_pass = common_inventory_pass and all(row["gate_pass"] for row in h5_rows)

    h6_contract = hypotheses["H6_FULL_NOISE_SYSTEMATIC_OFFSET"]
    h6_rows = []
    for backend in BACKENDS:
        for geometry in geometries:
            selected = [
                row for row in rows if row["backend_schema_name"] == backend
                and row["condition"] == h6_contract["condition"]
                and row["scene_variant"] == h6_contract["scene_variant"]
                and row["geometry_seed"] == geometry and success(row)
                and row.get("translation_vector") is not None
            ]
            unique_inputs = {
                (str(row.get("source_checksum", row["planned_snapshot_id"])), str(row.get("target_checksum", "")))
                for row in selected
            }
            vectors = np.asarray([row["translation_vector"] for row in selected], dtype=np.float64)
            offset = float(np.linalg.norm(np.mean(vectors, axis=0))) if len(vectors) else None
            mean_norm = float(np.mean(np.linalg.norm(vectors, axis=1))) if len(vectors) else None
            fraction = (offset / mean_norm) if offset is not None and mean_norm is not None and mean_norm > 0.0 else None
            passed = bool(
                len(unique_inputs) >= h6_contract["effective_replicate_count_min"]
                and offset is not None and offset >= h6_contract["systematic_translation_offset_min_m"]
                and fraction is not None and fraction >= h6_contract["systematic_fraction_min"]
            )
            h6_rows.append({"backend_schema_name": backend, "geometry_seed": geometry, "successful_count": len(selected), "effective_replicate_count": len(unique_inputs), "systematic_translation_offset_m": offset, "systematic_fraction": fraction, "group_gate_pass": passed})
    h6_backend = []
    for backend in BACKENDS:
        selected = [row for row in h6_rows if row["backend_schema_name"] == backend]
        fractions = [float(row["systematic_fraction"]) for row in selected if row["systematic_fraction"] is not None]
        passing = sum(row["group_gate_pass"] for row in selected)
        median_fraction = _median(fractions)
        passed = bool(len(selected) == h6_contract["geometry_group_count"] and passing >= h6_contract["minimum_passing_geometry_groups"] and median_fraction is not None and median_fraction >= h6_contract["median_systematic_fraction_min"])
        h6_backend.append({"backend_schema_name": backend, "passing_geometry_group_count": passing, "median_systematic_fraction": median_fraction, "gate_pass": passed})
    h6_pass = all(row["gate_pass"] for row in h6_backend)

    gate_summary = {
        "CONFIRMATORY_EVIDENCE_INTEGRITY_PASS": bool(exact_plan and common_inventory_pass),
        "H1_IDEAL_CONTROL_PASS": h1_pass,
        "H2_LONG_CORRIDOR_SCENE_EFFECT_PASS": h2_pass,
        "H3_CROSS_BACKEND_SCENE_RANKING_PASS": h3_pass,
        "H4_REASSOCIATION_MECHANISM_PASS": h4_pass,
        "H5_FROZEN_MODEL_B_INCREMENTAL_VALUE_PASS": h5_pass,
        "H6_FULL_NOISE_SYSTEMATIC_OFFSET_PASS": h6_pass,
    }
    validation_pass = all(gate_summary.values())
    final_decision = {
        **gate_summary,
        "SYNTHETIC_CONFIRMATORY_EXECUTED": True,
        "SYNTHETIC_CONFIRMATORY_COMPLETE": exact_plan,
        "SYNTHETIC_CONFIRMATORY_PASS": validation_pass,
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
    }
    return {
        "schema_version": "synthetic_confirmatory_primary_analysis_v1",
        "gate_summary": gate_summary,
        "final_decision": final_decision,
        "integrity": {
            "planned_trial_count": 1190, "observed_trial_count": len(rows),
            "exact_plan_cardinality_pass": exact_plan,
            "common_expected_count": len(eligible_ids),
            "common_observed_count": len(common),
            "common_record_inventory_pass": common_inventory_pass,
        },
        "h1_ideal_control": h1_rows,
        "h2_scene_effect": h2_rows,
        "h3_cross_backend_ranking": h3_rows,
        "h4_reassociation": h4_rows,
        "h5_frozen_models": h5_rows,
        "h6_systematic_groups": h6_rows,
        "h6_systematic_backend": h6_backend,
    }


def _typed_plan_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        **row,
        "geometry_seed": int(row["geometry_seed"]),
        "measurement_seed": None if row.get("measurement_seed") in (None, "") else int(row["measurement_seed"]),
        "repeat_index": int(row["repeat_index"]),
    }


def load_synthetic_confirmatory_raw(
    *, manifest_path: str | Path, run_dir: str | Path
) -> tuple[list[dict[str, Any]], dict[str, Any], Path, dict[str, Any]]:
    """Strictly read all 1,190 raw results; never accepts a partial run."""

    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError("Synthetic Confirmatory formal manifest is missing")
    manifest = _strict_object(manifest_file)
    root = manifest_file.parent.parent.resolve()
    from .synthetic_confirmatory_manifest import (
        MANIFEST_RELATIVE,
        verify_synthetic_confirmatory_manifest,
    )
    if manifest_file != (root / MANIFEST_RELATIVE).resolve():
        raise ValueError("Confirmatory analysis requires the single formal manifest")
    if verify_synthetic_confirmatory_manifest(
        root, require_authorized=True
    ) != manifest:
        raise ValueError("Confirmatory manifest authentication mismatch")
    plan_path = _root_path(root, str(manifest.get("planned_trials_path", "protocols/synthetic_confirmatory_planned_trials.csv")))
    plan_rows = [_typed_plan_row(row) for row in _csv(plan_path)]
    if len(plan_rows) != 1190 or len({row["planned_trial_id"] for row in plan_rows}) != 1190:
        raise ValueError("Confirmatory trial plan inventory changed")
    expected_plan_sha = manifest.get("planned_trials_sha256")
    if isinstance(expected_plan_sha, str) and _sha256(plan_path) != expected_plan_sha:
        raise ValueError("Confirmatory trial plan SHA mismatch")
    directory = Path(run_dir).resolve()
    expected_directory = _root_path(root, str(manifest.get("formal_output_dir", "results/synthetic_confirmatory_v1")))
    if directory != expected_directory:
        raise ValueError("Confirmatory result directory differs from manifest")
    raw_path = directory / "raw_result_manifest.json"
    if not raw_path.is_file():
        raise FileNotFoundError("Synthetic Confirmatory raw results do not exist")
    raw_manifest = _strict_object(raw_path)
    if set(raw_manifest) != {"schema_version", "run_id", "results"} or raw_manifest.get("schema_version") != RAW_MANIFEST_SCHEMA or raw_manifest.get("run_id") != manifest.get("formal_run_id", FORMAL_RUN_ID) or type(raw_manifest.get("results")) is not dict:
        raise ValueError("Confirmatory raw result manifest contract mismatch")
    entries = raw_manifest["results"]
    plan_by_id = {str(row["planned_trial_id"]): row for row in plan_rows}
    if set(entries) != set(plan_by_id):
        raise ValueError("Confirmatory raw results are incomplete or contain extras")
    raw_root = (directory / "raw_results").resolve()
    rows = []
    for trial_id in sorted(plan_by_id):
        entry = entries[trial_id]
        if type(entry) is not dict or set(entry) != {"planned_trial_id", "path", "sha256"} or entry.get("planned_trial_id") != trial_id or not isinstance(entry.get("path"), str) or Path(entry["path"]).name != entry["path"]:
            raise ValueError("Confirmatory raw manifest entry contract mismatch")
        result_path = (raw_root / entry["path"]).resolve()
        if result_path.parent != raw_root or not result_path.is_file() or _sha256(result_path) != entry.get("sha256"):
            raise ValueError("Confirmatory raw result file/SHA mismatch")
        result = _trial_validator(_strict_object(result_path))
        planned = plan_by_id[trial_id]
        if any(result[name] != expected for name, expected in {
            "planned_trial_id": trial_id,
            "snapshot_id": planned["planned_snapshot_id"],
            "scene_variant": planned["scene_variant"],
            "condition": planned["condition"],
            "backend": planned["backend"],
        }.items()):
            raise ValueError("Confirmatory raw result differs from frozen plan")
        rows.append({
            **result, "backend_schema_name": result["backend"],
            "geometry_seed": int(planned["geometry_seed"]),
            "measurement_seed": planned["measurement_seed"],
            "repeat_index": int(planned["repeat_index"]),
            "planned_snapshot_id": planned["planned_snapshot_id"],
        })
    return rows, raw_manifest, root, manifest


def _snapshot_directory(cache_root: Path, snapshot_id: str) -> Path:
    candidates = (cache_root / snapshot_id, cache_root / snapshot_id.split("::")[-1])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and (resolved == cache_root or cache_root in resolved.parents):
            return resolved
    raise FileNotFoundError(f"Confirmatory snapshot is missing: {snapshot_id}")


def recompute_confirmatory_common_records(
    *, trials: Sequence[Mapping[str, Any]], snapshot_cache_root: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute reference vectors and frozen offline common metrics in memory."""

    cache = Path(snapshot_cache_root).resolve()
    if not cache.is_dir():
        raise FileNotFoundError("Confirmatory snapshot cache does not exist")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[str(row["planned_snapshot_id"])].append(row)
    normalized = []
    common = []
    for snapshot_id in sorted(grouped):
        first = grouped[snapshot_id][0]
        snapshot = read_synthetic_confirmatory_snapshot(
            cache,
            {
                "planned_snapshot_id": snapshot_id,
                "scene_variant": first["scene_variant"],
                "condition": first["condition"],
                "geometry_seed": first["geometry_seed"],
                "measurement_seed": first["measurement_seed"],
                "repeat_index": first["repeat_index"],
            },
            arrays=True,
        )
        source = snapshot["source"]
        target = snapshot["target"]
        reference = snapshot["reference"]
        context = None
        if any(row["condition"] != "IDEAL_MATCHED" for row in grouped[snapshot_id]):
            context = prepare_common_association_context(source, target, reference, snapshot_id=snapshot_id)
        for row in grouped[snapshot_id]:
            if any(
                row.get(name) != snapshot[name]
                for name in (
                    "snapshot_checksum", "source_checksum", "target_checksum",
                    "reference_pose_checksum",
                )
            ):
                raise ValueError("Confirmatory trial/snapshot checksum mismatch")
            updated = dict(row)
            transform = row.get("final_transform_4x4")
            if (
                transform is not None
                and not row["solver_failure"]
                and row["finite_output"]
            ):
                estimate = np.asarray(transform, dtype=np.float64)
                vector = estimate[:3, 3] - reference[:3, 3]
                rotation = rotation_metric_audit(
                    estimate[:3, :3], reference[:3, :3]
                )["rotation_error_rad"]
                if rotation is None:
                    raise ValueError("Confirmatory rotation recomputation failed")
                if (
                    abs(float(row["translation_update_m"]) - float(np.linalg.norm(vector)))
                    > 1.0e-12
                    or abs(float(row["rotation_update_rad"]) - float(rotation))
                    > 1.0e-12
                ):
                    raise ValueError("Confirmatory stored metric recomputation mismatch")
                updated["translation_vector"] = vector.astype(float).tolist()
                updated["translation_error_m"] = float(np.linalg.norm(vector))
                updated["rotation_error_rad"] = float(rotation)
            normalized.append(updated)
            if row["condition"] != "IDEAL_MATCHED" and not row["solver_failure"] and row["finite_output"]:
                if context is None or transform is None:
                    raise ValueError("successful nonideal trial lacks common-analysis input")
                identifiers = {
                    "planned_trial_id": row["planned_trial_id"],
                    "backend_schema_name": row["backend_schema_name"],
                    "scene_variant": row["scene_variant"],
                    "condition": row["condition"],
                    "geometry_seed": row["geometry_seed"],
                    "measurement_seed": row["measurement_seed"],
                    "repeat_index": row["repeat_index"],
                }
                common.append(safe_analyze_estimated_transform(context, np.asarray(transform, dtype=np.float64), identifiers=identifiers))
    return sorted(normalized, key=lambda row: row["planned_trial_id"]), sorted(common, key=lambda row: row["planned_trial_id"])


def analyze_synthetic_confirmatory(
    *, manifest_path: str | Path, run_dir: str | Path
) -> dict[str, Any]:
    trials, _raw, root, manifest = load_synthetic_confirmatory_raw(
        manifest_path=manifest_path, run_dir=run_dir
    )
    cache = _root_path(root, str(manifest.get("snapshot_cache_root", "data/synthetic_confirmatory_v1_snapshots")))
    trials, common = recompute_confirmatory_common_records(trials=trials, snapshot_cache_root=cache)
    gate_path = _root_path(root, str(manifest.get("gate_contract_path", "protocols/synthetic_confirmatory_gate_contract.json")))
    model_path = _root_path(root, str(manifest.get("frozen_model_path", "frozen_assets/confirmatory_development_trained_models_v1.json")))
    protocol_path = _root_path(root, str(manifest.get("scientific_protocol_path", "protocols/synthetic_confirmatory_protocol_v1.json")))
    protocol = _strict_object(protocol_path)
    report = analyze_synthetic_confirmatory_records(
        trials=trials, common_records=common, model_lock=_strict_object(model_path),
        gate_contract=_strict_object(gate_path),
        expected_geometry_seeds=protocol["geometry_seeds"],
    )
    report["run_id"] = manifest.get("formal_run_id", FORMAL_RUN_ID)
    report["raw_result_manifest_sha256"] = _sha256(Path(run_dir) / "raw_result_manifest.json")
    return report


__all__ = [
    "BACKENDS", "CONDITIONS", "FORMAL_RUN_ID", "RAW_MANIFEST_SCHEMA", "SCENES",
    "analyze_synthetic_confirmatory", "analyze_synthetic_confirmatory_records",
    "load_synthetic_confirmatory_raw", "recompute_confirmatory_common_records",
]
