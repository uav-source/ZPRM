"""Read-only Model A/B survival checks and Development-trained model locks.

This module deliberately does not import a registration backend and never reads a
point cloud.  It consumes only the frozen primary-analysis rows.  The three
weighting schemes preserve the preregistered leave-one-geometry-seed-out split,
Ridge ``alpha=1.0``, and a scaler fitted on each training fold only.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .local_metric_models import (
    MODEL_A_FEATURE_NAMES,
    MODEL_B_ADDITIONAL_FEATURE_NAMES,
    MODEL_B_FEATURE_NAMES,
    NONIDEAL_CONDITIONS,
    RIDGE_ALPHA,
    RIDGE_FIT_INTERCEPT,
    TARGET_EPSILON_M,
    model_feature_vector,
)


BACKENDS = ("open3d_point_to_plane", "pcl_point_to_plane")
DEVELOPMENT_GEOMETRY_SEEDS = (1334931069, 1850310744, 1957656152)
WEIGHTING_SCHEMES = (
    "ORIGINAL_TRIAL_WEIGHTED",
    "UNIQUE_INPUT_WEIGHTED",
    "UNIQUE_INPUT_CONDITION_BALANCED",
)
FINAL_MODEL_WEIGHTING = "UNIQUE_INPUT_CONDITION_BALANCED"
MODEL_CLAIM_TYPE = "POST_REGISTRATION_EXPLANATORY"
FORBIDDEN_MODEL_CLAIM = "PRE_REGISTRATION_FAILURE_PREDICTION"

CONFIRMATORY_GEOMETRY_SEEDS = (
    248284635,
    376488233,
    198112089,
    229684695,
    226655024,
)
CONFIRMATORY_MEASUREMENT_SEEDS = (469989467, 1088311622, 916609326)
CONFIRMATORY_BOOTSTRAP_SEED = 1083684578
CONFIRMATORY_SEEDS = frozenset(
    (*CONFIRMATORY_GEOMETRY_SEEDS, *CONFIRMATORY_MEASUREMENT_SEEDS, CONFIRMATORY_BOOTSTRAP_SEED)
)

_SEED_USAGE_KEYS = frozenset(
    {
        "bootstrap_seed",
        "bootstrap_seed_value",
        "geometry_seed",
        "geometry_seed_value",
        "measurement_seed",
        "measurement_seed_value",
        "random_seed",
        "rng_seed",
        "seed_used",
    }
)
_SEED_USAGE_CONTAINER_KEYS = frozenset(
    {
        "geometry_seeds_used",
        "instantiated_seeds",
        "measurement_seeds_used",
        "rng_seeds_used",
        "seeds_instantiated",
        "seeds_used",
        "used_seeds",
    }
)
_STRUCTURED_SCAN_ROOTS = ("results", "data", "artifacts", "frozen_assets")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_primary_analysis(path: str | Path) -> dict[str, Any]:
    """Load one strict JSON object without accepting NaN or duplicate keys."""

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for name, value in pairs:
            if name in output:
                raise ValueError(f"duplicate JSON key: {name}")
            output[name] = value
        return output

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {token}")
        ),
    )
    if type(value) is not dict:
        raise ValueError("primary analysis root must be an object")
    return value


def join_development_model_rows(
    primary_analysis: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join frozen common metrics to trial input identity without recomputation."""

    trials = primary_analysis.get("geometry_seed_raw_values")
    common = primary_analysis.get("joined_common_association_metrics")
    if type(trials) is not list or type(common) is not list:
        raise ValueError("primary analysis lacks frozen model inputs")
    trial_by_id: dict[str, Mapping[str, Any]] = {}
    for row in trials:
        if type(row) is not dict:
            raise ValueError("trial row must be an object")
        trial_id = str(row.get("planned_trial_id"))
        if trial_id in trial_by_id:
            raise ValueError("duplicate frozen trial identity")
        trial_by_id[trial_id] = row
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for metric in common:
        if type(metric) is not dict:
            raise ValueError("common-association row must be an object")
        trial_id = str(metric.get("planned_trial_id"))
        if trial_id in seen or trial_id not in trial_by_id:
            raise ValueError("common-association trial identity is ambiguous")
        seen.add(trial_id)
        trial = trial_by_id[trial_id]
        identity_fields = (
            "backend_schema_name",
            "condition",
            "geometry_seed",
            "measurement_seed",
            "repeat_index",
            "scene_variant",
            "snapshot_id",
        )
        if any(metric.get(name) != trial.get(name) for name in identity_fields):
            raise ValueError("common/trial frozen identity mismatch")
        if (
            metric.get("condition") not in NONIDEAL_CONDITIONS
            or metric.get("backend_schema_name") not in BACKENDS
            or metric.get("common_association_valid") is not True
            or trial.get("solver_failure") is not False
            or trial.get("finite_output") is not True
        ):
            continue
        merged = dict(metric)
        for name in (
            "source_checksum",
            "target_checksum",
            "snapshot_checksum",
            "reference_pose_checksum",
        ):
            value = trial.get(name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"missing frozen trial checksum: {name}")
            merged[name] = value
        # The target is taken from the frozen primary trial surface; it is never
        # admitted to either feature vector.
        merged["translation_error_m"] = trial.get("translation_error_m")
        output.append(merged)
    output.sort(key=lambda row: str(row["planned_trial_id"]))
    if len(output) != len(common):
        raise ValueError("not every frozen common record was eligible for model survival")
    return output


def _unique_input_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["backend_schema_name"]),
        str(row["scene_variant"]),
        int(row["geometry_seed"]),
        str(row["condition"]),
        str(row["source_checksum"]),
        str(row["target_checksum"]),
    )


def model_sample_weights(
    rows: Sequence[Mapping[str, Any]], scheme: str
) -> np.ndarray:
    """Return frozen raw, unique-input, or unique+condition-balanced weights.

    Unique-input weights sum to the number of distinct source-target inputs,
    exactly matching a conceptual collapse while retaining every frozen row for
    audit.  Condition-balanced weights retain that total effective sample size.
    """

    if scheme not in WEIGHTING_SCHEMES:
        raise ValueError(f"unknown weighting scheme: {scheme}")
    if not rows:
        raise ValueError("model weighting rows cannot be empty")
    if scheme == "ORIGINAL_TRIAL_WEIGHTED":
        return np.ones(len(rows), dtype=np.float64)
    keys = [_unique_input_key(row) for row in rows]
    counts = Counter(keys)
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    if scheme == "UNIQUE_INPUT_WEIGHTED":
        return weights
    condition_totals: dict[str, float] = defaultdict(float)
    for row, weight in zip(rows, weights):
        condition_totals[str(row["condition"])] += float(weight)
    if set(condition_totals) != set(NONIDEAL_CONDITIONS):
        raise ValueError("condition-balanced model requires all five conditions")
    effective_total = float(np.sum(weights))
    target_per_condition = effective_total / len(condition_totals)
    return np.asarray(
        [
            weight
            * target_per_condition
            / condition_totals[str(row["condition"])]
            for row, weight in zip(rows, weights)
        ],
        dtype=np.float64,
    )


def _target(row: Mapping[str, Any]) -> float:
    value = row.get("translation_error_m")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("translation error target is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError("translation error target is invalid")
    return math.log10(number + TARGET_EPSILON_M)


def _weighted_mae(errors: np.ndarray, weights: np.ndarray) -> float:
    if errors.shape != weights.shape or not np.all(np.isfinite(errors)):
        raise ValueError("weighted MAE inputs are invalid")
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise ValueError("weighted MAE has zero weight")
    return float(np.dot(errors, weights) / denominator)


def _evaluate_one_model(
    rows: Sequence[Mapping[str, Any]], model: str, scheme: str
) -> dict[str, Any]:
    features = np.vstack([model_feature_vector(row, model) for row in rows])
    target = np.asarray([_target(row) for row in rows], dtype=np.float64)
    geometry = np.asarray([int(row["geometry_seed"]) for row in rows], dtype=np.int64)
    seeds = tuple(sorted(set(geometry.tolist())))
    if seeds != DEVELOPMENT_GEOMETRY_SEEDS:
        raise ValueError("survival LOGO requires the three frozen Development seeds")
    folds: list[dict[str, Any]] = []
    weighted_error_sum = 0.0
    test_weight_sum = 0.0
    for held_out in seeds:
        test_mask = geometry == held_out
        train_mask = ~test_mask
        train_rows = [row for row, keep in zip(rows, train_mask) if keep]
        test_rows = [row for row, keep in zip(rows, test_mask) if keep]
        train_weights = model_sample_weights(train_rows, scheme)
        test_weights = model_sample_weights(test_rows, scheme)
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(features[train_mask], sample_weight=train_weights)
        train_scaled = scaler.transform(features[train_mask])
        test_scaled = scaler.transform(features[test_mask])
        estimator = Ridge(alpha=RIDGE_ALPHA, fit_intercept=RIDGE_FIT_INTERCEPT)
        estimator.fit(train_scaled, target[train_mask], sample_weight=train_weights)
        absolute_error = np.abs(estimator.predict(test_scaled) - target[test_mask])
        fold_mae = _weighted_mae(absolute_error, test_weights)
        weighted_error_sum += float(np.dot(absolute_error, test_weights))
        test_weight_sum += float(np.sum(test_weights))
        folds.append(
            {
                "held_out_geometry_seed": int(held_out),
                "training_geometry_seeds": [int(seed) for seed in seeds if seed != held_out],
                "training_row_count": len(train_rows),
                "test_row_count": len(test_rows),
                "training_effective_unique_input_count": len(
                    {_unique_input_key(row) for row in train_rows}
                ),
                "test_effective_unique_input_count": len(
                    {_unique_input_key(row) for row in test_rows}
                ),
                "training_weight_sum": float(np.sum(train_weights)),
                "test_weight_sum": float(np.sum(test_weights)),
                "fold_mae_log10_translation_error": fold_mae,
                "ridge_alpha": float(estimator.alpha),
                "ridge_fit_intercept": bool(estimator.fit_intercept),
                "scaler_fit_scope": "TRAINING_FOLD_ONLY",
                "scaler_mean": scaler.mean_.astype(float).tolist(),
                "scaler_scale": scaler.scale_.astype(float).tolist(),
            }
        )
    names = MODEL_A_FEATURE_NAMES if model == "A" else MODEL_B_FEATURE_NAMES
    return {
        "model": model,
        "feature_names": list(names),
        "weighting_scheme": scheme,
        "cv_scheme": "LEAVE_ONE_GEOMETRY_SEED_OUT",
        "cv_mae_log10_translation_error": weighted_error_sum / test_weight_sum,
        "folds": folds,
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_fit_intercept": RIDGE_FIT_INTERCEPT,
        "scaler_fit_scope": "TRAINING_FOLD_ONLY",
    }


def evaluate_model_weighting_sensitivity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate Model A/B under all three frozen weighting schemes."""

    sensitivity_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    backend_passes: dict[str, bool] = {}
    for backend in BACKENDS:
        selected = [row for row in rows if row.get("backend_schema_name") == backend]
        if not selected:
            raise ValueError(f"missing frozen model rows for {backend}")
        scheme_passes: list[bool] = []
        for scheme in WEIGHTING_SCHEMES:
            model_a = _evaluate_one_model(selected, "A", scheme)
            model_b = _evaluate_one_model(selected, "B", scheme)
            mae_a = float(model_a["cv_mae_log10_translation_error"])
            mae_b = float(model_b["cv_mae_log10_translation_error"])
            relative = (mae_a - mae_b) / mae_a if mae_a > 0.0 else float("-inf")
            improvements: list[float] = []
            for fold_a, fold_b in zip(model_a["folds"], model_b["folds"]):
                if (
                    fold_a["held_out_geometry_seed"]
                    != fold_b["held_out_geometry_seed"]
                    or fold_a["training_geometry_seeds"]
                    != fold_b["training_geometry_seeds"]
                ):
                    raise AssertionError("Model A/B folds differ")
                fold_mae_a = float(fold_a["fold_mae_log10_translation_error"])
                fold_mae_b = float(fold_b["fold_mae_log10_translation_error"])
                improvement = (
                    (fold_mae_a - fold_mae_b) / fold_mae_a
                    if fold_mae_a > 0.0
                    else float("-inf")
                )
                improvements.append(improvement)
                fold_rows.append(
                    {
                        "backend_schema_name": backend,
                        "weighting_scheme": scheme,
                        "held_out_geometry_seed": fold_a["held_out_geometry_seed"],
                        "training_geometry_seeds": fold_a["training_geometry_seeds"],
                        "model_a_mae": fold_mae_a,
                        "model_b_mae": fold_mae_b,
                        "relative_improvement": improvement,
                        "model_b_better": fold_mae_b < fold_mae_a,
                        "degradation_within_10_percent": improvement >= -0.10,
                    }
                )
            positive_fold_count = sum(value > 0.0 for value in improvements)
            scheme_pass = bool(
                math.isfinite(relative)
                and relative >= 0.10
                and positive_fold_count >= 2
                and all(value >= -0.10 for value in improvements)
            )
            scheme_passes.append(scheme_pass)
            sensitivity_rows.append(
                {
                    "backend_schema_name": backend,
                    "weighting_scheme": scheme,
                    "row_count": len(selected),
                    "effective_unique_input_count": len(
                        {_unique_input_key(row) for row in selected}
                    ),
                    "model_a_cv_mae": mae_a,
                    "model_b_cv_mae": mae_b,
                    "relative_improvement": relative,
                    "model_b_better_fold_count": positive_fold_count,
                    "minimum_fold_relative_improvement": min(improvements),
                    "weighting_gate_pass": scheme_pass,
                }
            )
        backend_passes[backend] = len(scheme_passes) == 3 and all(scheme_passes)
    return {
        "MODEL_INCREMENTAL_VALUE_ROBUST_PASS": all(backend_passes.values()),
        "backend_gate_pass": backend_passes,
        "claim_type": MODEL_CLAIM_TYPE,
        "forbidden_claim_type": FORBIDDEN_MODEL_CLAIM,
        "model_fold_results": fold_rows,
        "model_weighting_sensitivity": sensitivity_rows,
        "schema_version": "scientific_survival_model_sensitivity_v1",
    }


def _fit_final_model(
    rows: Sequence[Mapping[str, Any]], model: str
) -> dict[str, Any]:
    features = np.vstack([model_feature_vector(row, model) for row in rows])
    target = np.asarray([_target(row) for row in rows], dtype=np.float64)
    weights = model_sample_weights(rows, FINAL_MODEL_WEIGHTING)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(features, sample_weight=weights)
    transformed = scaler.transform(features)
    estimator = Ridge(alpha=RIDGE_ALPHA, fit_intercept=RIDGE_FIT_INTERCEPT)
    estimator.fit(transformed, target, sample_weight=weights)
    names = MODEL_A_FEATURE_NAMES if model == "A" else MODEL_B_FEATURE_NAMES
    training_rows = [
        {
            "condition": str(row["condition"]),
            "feature_vector": model_feature_vector(row, model).astype(float).tolist(),
            "geometry_seed": int(row["geometry_seed"]),
            "planned_trial_id": str(row["planned_trial_id"]),
            "source_checksum": str(row["source_checksum"]),
            "target_checksum": str(row["target_checksum"]),
            "target_log10_translation_error": _target(row),
            "weight": float(weight),
        }
        for row, weight in zip(rows, weights)
    ]
    return {
        "alpha": float(estimator.alpha),
        "coefficient": estimator.coef_.astype(float).tolist(),
        "effective_unique_input_count": len({_unique_input_key(row) for row in rows}),
        "feature_names": list(names),
        "fit_intercept": bool(estimator.fit_intercept),
        "intercept": float(estimator.intercept_),
        "model": model,
        "row_count": len(rows),
        "scaler_fit_scope": "ALL_DEVELOPMENT_ROWS_ONLY",
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "training_data_sha256": canonical_json_sha256(training_rows),
        "weight_sum": float(np.sum(weights)),
        "weighting_scheme": FINAL_MODEL_WEIGHTING,
    }


def build_frozen_development_model_lock(
    rows: Sequence[Mapping[str, Any]], *, repository_root: str | Path
) -> dict[str, Any]:
    """Train exactly four Development-only models without touching Confirmatory seeds."""

    repository = Path(repository_root).resolve()
    source_path = repository / "src/phase_a_harness/local_metric_models.py"
    audit_path = repository / "src/phase_a_harness/scientific_survival_models.py"
    models: list[dict[str, Any]] = []
    for backend in BACKENDS:
        selected = [row for row in rows if row.get("backend_schema_name") == backend]
        for model in ("A", "B"):
            fitted = _fit_final_model(selected, model)
            fitted["backend_schema_name"] = backend
            fitted["model_id"] = f"{backend}::MODEL_{model}"
            models.append(fitted)
    population_rows = [
        {
            "backend_schema_name": str(row["backend_schema_name"]),
            "condition": str(row["condition"]),
            "geometry_seed": int(row["geometry_seed"]),
            "planned_trial_id": str(row["planned_trial_id"]),
            "source_checksum": str(row["source_checksum"]),
            "target_checksum": str(row["target_checksum"]),
            "translation_error_m": float(row["translation_error_m"]),
        }
        for row in sorted(rows, key=lambda item: str(item["planned_trial_id"]))
    ]
    core = {
        "alpha": RIDGE_ALPHA,
        "confirmatory_seed_access_count": 0,
        "feature_source_sha256": file_sha256(source_path),
        "fit_intercept": RIDGE_FIT_INTERCEPT,
        "model_claim_type": MODEL_CLAIM_TYPE,
        "model_count": 4,
        "models": models,
        "schema_version": "confirmatory_development_trained_models_v1",
        "survival_model_code_sha256": file_sha256(audit_path),
        "training_population_identity_sha256": canonical_json_sha256(population_rows),
        "training_row_count": len(population_rows),
        "training_population": "FULL_SYNTHETIC_DEVELOPMENT_NONIDEAL_ONLY",
        "weighting_scheme": FINAL_MODEL_WEIGHTING,
    }
    return {**core, "model_lock_payload_sha256": canonical_json_sha256(core)}


def _seed_hits_in_json(
    value: Any,
    *,
    relative_path: str,
    key_path: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if type(value) is dict:
        for name, item in value.items():
            path = (*key_path, str(name))
            normalized = str(name).lower()
            if (
                normalized in _SEED_USAGE_KEYS
                and not any("confirmatory" in part.lower() for part in path[:-1])
                and isinstance(item, int)
                and not isinstance(item, bool)
                and item in CONFIRMATORY_SEEDS
            ):
                hits.append(
                    {
                        "relative_path": relative_path,
                        "field_path": ".".join(path),
                        "seed": item,
                    }
                )
            hits.extend(
                _seed_hits_in_json(item, relative_path=relative_path, key_path=path)
            )
    elif type(value) is list:
        for index, item in enumerate(value):
            if (
                key_path
                and key_path[-1].lower() in _SEED_USAGE_CONTAINER_KEYS
                and isinstance(item, int)
                and not isinstance(item, bool)
                and item in CONFIRMATORY_SEEDS
            ):
                hits.append(
                    {
                        "relative_path": relative_path,
                        "field_path": ".".join((*key_path, str(index))),
                        "seed": item,
                    }
                )
            hits.extend(
                _seed_hits_in_json(
                    item,
                    relative_path=relative_path,
                    key_path=(*key_path, str(index)),
                )
            )
    return hits


def _structured_paths(repository: Path) -> Iterable[Path]:
    for name in _STRUCTURED_SCAN_ROOTS:
        root = repository / name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".ndjson"}:
                continue
            # Protocol declarations are not RNG instantiation.  Runtime manifests,
            # plans already executed, trial/snapshot records, and provenance remain
            # in scope.  The future protocols/ directory is intentionally not a
            # scan root for the same reason.
            if "protocol" in path.name.lower() and name == "frozen_assets":
                continue
            yield path


def scan_confirmatory_seed_provenance(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Scan structured experimental evidence, excluding prose/declarations."""

    repository = Path(repository_root).resolve()
    hits: list[dict[str, Any]] = []
    scanned = 0
    parse_failures: list[str] = []
    for path in sorted(_structured_paths(repository)):
        relative = path.relative_to(repository).as_posix()
        scanned += 1
        try:
            if path.suffix.lower() == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                hits.extend(_seed_hits_in_json(value, relative_path=relative))
            elif path.suffix.lower() == ".ndjson":
                for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                    if line.strip():
                        hits.extend(
                            _seed_hits_in_json(
                                json.loads(line),
                                relative_path=relative,
                                key_path=(str(line_index),),
                            )
                        )
            else:
                with path.open("r", encoding="utf-8", newline="") as stream:
                    for row_index, row in enumerate(csv.DictReader(stream)):
                        for name, item in row.items():
                            if name.lower() not in _SEED_USAGE_KEYS or item in (None, ""):
                                continue
                            try:
                                number = int(item)
                            except ValueError:
                                continue
                            if number in CONFIRMATORY_SEEDS:
                                hits.append(
                                    {
                                        "relative_path": relative,
                                        "field_path": f"row[{row_index}].{name}",
                                        "seed": number,
                                    }
                                )
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as error:
            parse_failures.append(f"{relative}: {type(error).__name__}")
    hits.sort(key=lambda row: (row["relative_path"], row["field_path"], row["seed"]))
    return {
        "CONFIRMATORY_SEED_PROVENANCE_PASS": not hits and not parse_failures,
        "confirmatory_seed_instantiation_count": len(hits),
        "ordinary_document_mentions_counted_as_use": 0,
        "parse_failure_count": len(parse_failures),
        "parse_failures": parse_failures,
        "schema_version": "confirmatory_seed_provenance_audit_v1",
        "structured_file_count": scanned,
        "structured_usage_hits": hits,
    }


def audit_model_data_leakage(
    primary_analysis: Mapping[str, Any],
    *,
    repository_root: str | Path,
    seed_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit the frozen implementation and its recorded folds without refitting."""

    repository = Path(repository_root).resolve()
    source_path = repository / "src/phase_a_harness/local_metric_models.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    forbidden_features = {
        "translation_error_m",
        "translation_update_m",
        "scene_variant",
        "scene_label",
        "backend",
        "backend_schema_name",
    }
    feature_names = set(MODEL_B_FEATURE_NAMES)
    recorded = primary_analysis.get("ridge_model_comparison")
    if type(recorded) is not list:
        recorded = []
    folds_valid = True
    same_folds = True
    recorded_backends: set[str] = set()
    for row in recorded:
        if type(row) is not dict:
            folds_valid = False
            continue
        backend = str(row.get("backend_schema_name"))
        recorded_backends.add(backend)
        folds = row.get("fold_results")
        model_a = folds.get("model_a") if type(folds) is dict else None
        model_b = folds.get("model_b") if type(folds) is dict else None
        if (
            type(model_a) is not list
            or type(model_b) is not list
            or len(model_a) != 3
            or len(model_b) != 3
        ):
            folds_valid = False
            same_folds = False
            continue
        a_identity = []
        b_identity = []
        for name, values, identities in (("A", model_a, a_identity), ("B", model_b, b_identity)):
            for fold in values:
                held = fold.get("held_out_geometry_seed")
                training = fold.get("training_geometry_seeds")
                identities.append((held, tuple(training) if isinstance(training, list) else ()))
                if (
                    held not in DEVELOPMENT_GEOMETRY_SEEDS
                    or type(training) is not list
                    or set(training) != set(DEVELOPMENT_GEOMETRY_SEEDS) - {held}
                    or held in training
                    or fold.get("scaler_fit_scope") != "training_fold_only"
                    or fold.get("ridge_alpha") != 1.0
                    or fold.get("ridge_fit_intercept") is not True
                ):
                    folds_valid = False
        same_folds = same_folds and a_identity == b_identity
    checks = [
        ("SPLIT_UNIT_IS_GEOMETRY_SEED", folds_valid),
        ("TRAIN_TEST_GEOMETRY_DISJOINT", folds_valid),
        ("STANDARD_SCALER_TRAIN_ONLY", folds_valid and "scaler.fit_transform(features[train])" in source),
        ("RIDGE_ALPHA_FIXED_1", RIDGE_ALPHA == 1.0 and "Ridge(alpha=RIDGE_ALPHA" in source),
        (
            "NO_TEST_FOLD_TUNING",
            not ({"GridSearchCV", "RandomizedSearchCV"} & call_names),
        ),
        ("TARGET_NOT_A_FEATURE", not (feature_names & forbidden_features)),
        ("SCENE_NOT_A_FEATURE", "scene_variant" not in feature_names and "scene_label" not in feature_names),
        ("BACKEND_LABEL_NOT_A_FEATURE", "backend" not in feature_names and "backend_schema_name" not in feature_names),
        (
            "CONFIRMATORY_SEED_NOT_ACCESSED",
            seed_provenance.get("CONFIRMATORY_SEED_PROVENANCE_PASS") is True,
        ),
        (
            "MODEL_A_B_USE_IDENTICAL_FOLDS",
            same_folds and recorded_backends == set(BACKENDS),
        ),
        (
            "MODEL_B_IS_POST_REGISTRATION_EXPLANATORY",
            set(MODEL_B_ADDITIONAL_FEATURE_NAMES)
            == {
                "correspondence_turnover",
                "accepted_source_turnover",
                "median_normal_angle_change_deg",
                "q95_normal_angle_change_deg",
                "residual_rmse_change",
                "correspondence_count_change_ratio",
            },
        ),
    ]
    rows = [
        {"audit_check": name, "pass": bool(passed)} for name, passed in checks
    ]
    return {
        "MODEL_CLAIM_TYPE": MODEL_CLAIM_TYPE,
        "MODEL_DATA_LEAKAGE_AUDIT_PASS": all(row["pass"] for row in rows),
        "forbidden_claim": FORBIDDEN_MODEL_CLAIM,
        "local_metric_models_sha256": file_sha256(source_path),
        "rows": rows,
        "schema_version": "scientific_survival_model_leakage_audit_v1",
    }


__all__ = [
    "BACKENDS",
    "CONFIRMATORY_BOOTSTRAP_SEED",
    "CONFIRMATORY_GEOMETRY_SEEDS",
    "CONFIRMATORY_MEASUREMENT_SEEDS",
    "CONFIRMATORY_SEEDS",
    "FINAL_MODEL_WEIGHTING",
    "FORBIDDEN_MODEL_CLAIM",
    "MODEL_CLAIM_TYPE",
    "WEIGHTING_SCHEMES",
    "audit_model_data_leakage",
    "build_frozen_development_model_lock",
    "canonical_json_sha256",
    "evaluate_model_weighting_sensitivity",
    "file_sha256",
    "join_development_model_rows",
    "load_primary_analysis",
    "model_sample_weights",
    "scan_confirmatory_seed_provenance",
]
