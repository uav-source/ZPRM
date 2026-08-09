"""Frozen, no-refit inference for the Synthetic Confirmatory experiment.

Only the serialized Development scaler and Ridge parameters are consumed.  No
estimator class is imported and no training API exists in this module.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from .local_metric_models import MODEL_A_FEATURE_NAMES, MODEL_B_FEATURE_NAMES


EXPECTED_MODEL_FILE_SHA256 = (
    "99806f83d3a0d2393ee7e36e8c06f14a1a8a44fb8d02b074ebc2d9fe750d0872"
)
BACKENDS = ("open3d_point_to_plane", "pcl_point_to_plane")
MODEL_IDENTITIES = tuple(
    (backend, model) for backend in BACKENDS for model in ("A", "B")
)
MODEL_CLAIM_TYPE = "POST_REGISTRATION_EXPLANATORY"
FORBIDDEN_MODEL_CLAIM_TYPE = "PRE_REGISTRATION_FAILURE_PREDICTION"
FINAL_MODEL_WEIGHTING = "UNIQUE_INPUT_CONDITION_BALANCED"
MODEL_FIT_CALL_COUNT = 0
SCALER_FIT_CALL_COUNT = 0

# Fixed by human-readable values, not by a seed or RNG.  These are already in
# the frozen transformed-feature coordinate system.
SEEDLESS_FEATURE_FIXTURE: dict[str, float] = {
    "log10_initial_residual_rmse_plus_1e-9": -2.10,
    "log10_condition_number_trans_plus_1": 2.50,
    "log10_inverse_lambda_min_trans": 3.00,
    "spectral_entropy_trans": 0.68,
    "log10_initial_correspondence_count_plus_1": 4.70,
    "initial_translation_gradient_norm": 0.00040,
    "correspondence_turnover": 0.31,
    "accepted_source_turnover": 0.00,
    "median_normal_angle_change_deg": 0.006,
    "q95_normal_angle_change_deg": 0.42,
    "residual_rmse_change": -0.00005,
    "correspondence_count_change_ratio": 0.00,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            output[key] = value
        return output

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=object_hook,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {path}: {token}")
        ),
    )
    if type(value) is not dict:
        raise ValueError("frozen model root must be an object")
    return value


def load_frozen_models(path: str | Path) -> dict[str, Any]:
    """Load the model lock as strict JSON; this function never fits a model."""

    return _strict_json(Path(path))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_vector(value: Any, length: int, *, positive: bool = False) -> bool:
    if type(value) is not list or len(value) != length:
        return False
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        number = float(item)
        if not math.isfinite(number) or (positive and number <= 0.0):
            return False
    return True


def _fit_calls_in_runtime_source(path: Path) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = {"fit", "fit_transform", "partial_fit"}
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden
        ):
            calls.append({"line": int(node.lineno), "method": node.func.attr})
    return calls


def audit_frozen_model_contract(
    repository_root: str | Path,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all four serialized models and the direct-inference contract."""

    repository = Path(repository_root).resolve()
    path = (
        Path(model_path)
        if model_path is not None
        else repository / "frozen_assets/confirmatory_development_trained_models_v1.json"
    )
    lock = load_frozen_models(path)
    actual_file_sha = _sha256(path)
    models = lock.get("models")
    model_rows = models if type(models) is list else []
    identities = [
        (row.get("backend_schema_name"), row.get("model"))
        for row in model_rows
        if type(row) is dict
    ]
    root_unsigned = {
        name: value
        for name, value in lock.items()
        if name != "model_lock_payload_sha256"
    }
    payload_sha_pass = lock.get("model_lock_payload_sha256") == _canonical_sha256(
        root_unsigned
    )
    feature_source_path = repository / "src/phase_a_harness/local_metric_models.py"
    survival_model_source_path = (
        repository / "src/phase_a_harness/scientific_survival_models.py"
    )
    feature_source_sha_pass = (
        lock.get("feature_source_sha256") == _sha256(feature_source_path)
    )
    survival_model_code_sha_pass = (
        lock.get("survival_model_code_sha256")
        == _sha256(survival_model_source_path)
    )
    root_schema_pass = (
        lock.get("schema_version") == "confirmatory_development_trained_models_v1"
        and lock.get("model_count") == 4
        and lock.get("alpha") == 1.0
        and lock.get("fit_intercept") is True
        and lock.get("model_claim_type") == MODEL_CLAIM_TYPE
        and lock.get("model_claim_type") != FORBIDDEN_MODEL_CLAIM_TYPE
        and lock.get("weighting_scheme") == FINAL_MODEL_WEIGHTING
        and lock.get("confirmatory_seed_access_count") == 0
        and lock.get("training_population")
        == "FULL_SYNTHETIC_DEVELOPMENT_NONIDEAL_ONLY"
        and lock.get("training_row_count") == 2100
        and _is_sha256(lock.get("training_population_identity_sha256"))
        and _is_sha256(lock.get("feature_source_sha256"))
        and _is_sha256(lock.get("survival_model_code_sha256"))
        and feature_source_sha_pass
        and survival_model_code_sha_pass
        and identities == list(MODEL_IDENTITIES)
        and payload_sha_pass
    )

    per_model: list[dict[str, Any]] = []
    feature_order_pass = True
    schema_pass = root_schema_pass
    for row in model_rows:
        if type(row) is not dict:
            schema_pass = False
            continue
        model = row.get("model")
        backend = row.get("backend_schema_name")
        expected_features: Sequence[str] = (
            MODEL_A_FEATURE_NAMES if model == "A" else MODEL_B_FEATURE_NAMES
        )
        feature_names = row.get("feature_names")
        order_pass = feature_names == list(expected_features)
        feature_order_pass = feature_order_pass and order_pass
        length = len(expected_features)
        parameters_pass = (
            _finite_vector(row.get("scaler_mean"), length)
            and _finite_vector(row.get("scaler_scale"), length, positive=True)
            and _finite_vector(row.get("coefficient"), length)
            and isinstance(row.get("intercept"), (int, float))
            and not isinstance(row.get("intercept"), bool)
            and math.isfinite(float(row.get("intercept", float("nan"))))
        )
        row_schema_pass = (
            backend in BACKENDS
            and model in ("A", "B")
            and row.get("model_id") == f"{backend}::MODEL_{model}"
            and row.get("alpha") == 1.0
            and row.get("fit_intercept") is True
            and row.get("scaler_fit_scope") == "ALL_DEVELOPMENT_ROWS_ONLY"
            and row.get("weighting_scheme") == FINAL_MODEL_WEIGHTING
            and row.get("row_count") == 1050
            and row.get("effective_unique_input_count") == 861
            and _is_sha256(row.get("training_data_sha256"))
            and order_pass
            and parameters_pass
        )
        schema_pass = schema_pass and row_schema_pass
        per_model.append(
            {
                "alpha": row.get("alpha"),
                "backend_schema_name": backend,
                "coefficient_count": (
                    len(row["coefficient"])
                    if type(row.get("coefficient")) is list
                    else 0
                ),
                "feature_count": len(feature_names) if type(feature_names) is list else 0,
                "feature_names": feature_names,
                "feature_order_pass": order_pass,
                "intercept_complete": isinstance(row.get("intercept"), (int, float))
                and not isinstance(row.get("intercept"), bool),
                "model": model,
                "model_id": row.get("model_id"),
                "parameters_complete": parameters_pass,
                "ridge_schema_pass": row_schema_pass,
                "scaler_mean_count": (
                    len(row["scaler_mean"])
                    if type(row.get("scaler_mean")) is list
                    else 0
                ),
                "scaler_scale_count": (
                    len(row["scaler_scale"])
                    if type(row.get("scaler_scale")) is list
                    else 0
                ),
                "training_data_sha256": row.get("training_data_sha256"),
            }
        )

    runtime_path = Path(__file__).resolve()
    forbidden_fit_calls = _fit_calls_in_runtime_source(runtime_path)
    no_refit_pass = (
        not forbidden_fit_calls
        and MODEL_FIT_CALL_COUNT == 0
        and SCALER_FIT_CALL_COUNT == 0
    )
    return {
        "FROZEN_MODEL_FEATURE_ORDER_PASS": feature_order_pass
        and len(per_model) == 4,
        "FROZEN_MODEL_FILE_SHA_PASS": actual_file_sha
        == EXPECTED_MODEL_FILE_SHA256,
        "FROZEN_MODEL_NO_REFIT_CONTRACT_PASS": no_refit_pass,
        "FROZEN_MODEL_SCHEMA_PASS": schema_pass and len(per_model) == 4,
        "MODEL_FIT_CALL_COUNT": MODEL_FIT_CALL_COUNT,
        "SCALER_FIT_CALL_COUNT": SCALER_FIT_CALL_COUNT,
        "actual_model_file_sha256": actual_file_sha,
        "expected_model_file_sha256": EXPECTED_MODEL_FILE_SHA256,
        "forbidden_fit_calls": forbidden_fit_calls,
        "model_claim_type": lock.get("model_claim_type"),
        "model_count": len(per_model),
        "models": per_model,
        "payload_sha256_pass": payload_sha_pass,
        "training_code_sha256": {
            "feature_source_sha256": lock.get("feature_source_sha256"),
            "feature_source_sha256_pass": feature_source_sha_pass,
            "survival_model_code_sha256": lock.get("survival_model_code_sha256"),
            "survival_model_code_sha256_pass": survival_model_code_sha_pass,
        },
        "weighting_scheme": lock.get("weighting_scheme"),
    }


def _validated_prediction_inputs(
    model: Mapping[str, Any], features: Mapping[str, Any]
) -> tuple[list[float], list[float], list[float], list[float], float]:
    feature_names = model.get("feature_names")
    if type(feature_names) is not list or not all(
        isinstance(name, str) for name in feature_names
    ):
        raise ValueError("frozen model feature order is invalid")
    if set(features) != set(feature_names):
        missing = sorted(set(feature_names) - set(features))
        extra = sorted(set(features) - set(feature_names))
        raise ValueError(f"feature mapping mismatch; missing={missing}, extra={extra}")
    values: list[float] = []
    for name in feature_names:
        value = features[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"feature {name} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"feature {name} is non-finite")
        values.append(number)
    length = len(values)
    mean = model.get("scaler_mean")
    scale = model.get("scaler_scale")
    coefficient = model.get("coefficient")
    if (
        not _finite_vector(mean, length)
        or not _finite_vector(scale, length, positive=True)
        or not _finite_vector(coefficient, length)
    ):
        raise ValueError("frozen scaler/Ridge parameters are invalid")
    intercept = model.get("intercept")
    if (
        isinstance(intercept, bool)
        or not isinstance(intercept, (int, float))
        or not math.isfinite(float(intercept))
    ):
        raise ValueError("frozen Ridge intercept is invalid")
    if model.get("alpha") != 1.0 or model.get("fit_intercept") is not True:
        raise ValueError("frozen Ridge contract changed")
    return (
        values,
        [float(item) for item in mean],
        [float(item) for item in scale],
        [float(item) for item in coefficient],
        float(intercept),
    )


def predict_frozen_model(
    model: Mapping[str, Any], features: Mapping[str, Any]
) -> float:
    """Compute Ridge prediction directly from mean/scale/coef/intercept."""

    values, mean, scale, coefficient, intercept = _validated_prediction_inputs(
        model, features
    )
    prediction = math.fsum(
        [intercept]
        + [
            weight * ((value - center) / spread)
            for value, center, spread, weight in zip(
                values, mean, scale, coefficient
            )
        ]
    )
    if not math.isfinite(prediction):
        raise ValueError("frozen model produced a non-finite prediction")
    return prediction


def predict_frozen_model_independent(
    model: Mapping[str, Any], features: Mapping[str, Any]
) -> float:
    """Independent high-precision scalar recomputation used only for audit."""

    values, mean, scale, coefficient, intercept = _validated_prediction_inputs(
        model, features
    )
    with localcontext() as context:
        context.prec = 50
        prediction = Decimal(str(intercept))
        for index in range(len(values)):
            transformed = (Decimal(str(values[index])) - Decimal(str(mean[index]))) / Decimal(
                str(scale[index])
            )
            prediction += Decimal(str(coefficient[index])) * transformed
    result = float(prediction)
    if not math.isfinite(result):
        raise ValueError("independent frozen model prediction is non-finite")
    return result


def frozen_model_lookup(model_lock: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    models = model_lock.get("models")
    if type(models) is not list:
        raise ValueError("frozen model list is missing")
    output: dict[str, Mapping[str, Any]] = {}
    for row in models:
        if type(row) is not dict or not isinstance(row.get("model_id"), str):
            raise ValueError("frozen model record is invalid")
        model_id = str(row["model_id"])
        if model_id in output:
            raise ValueError("duplicate frozen model identity")
        output[model_id] = row
    if set(output) != {
        f"{backend}::MODEL_{model}" for backend, model in MODEL_IDENTITIES
    }:
        raise ValueError("frozen model identity set changed")
    return output


def crosscheck_frozen_model_predictions(
    repository_root: str | Path,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Cross-check all four models on one fixed, seed-free feature fixture."""

    repository = Path(repository_root).resolve()
    path = (
        Path(model_path)
        if model_path is not None
        else repository / "frozen_assets/confirmatory_development_trained_models_v1.json"
    )
    lock = load_frozen_models(path)
    lookup = frozen_model_lookup(lock)
    rows: list[dict[str, Any]] = []
    maximum = 0.0
    for model_id in sorted(lookup):
        model = lookup[model_id]
        names = model["feature_names"]
        fixture = {name: SEEDLESS_FEATURE_FIXTURE[name] for name in names}
        main = predict_frozen_model(model, fixture)
        independent = predict_frozen_model_independent(model, fixture)
        difference = abs(main - independent)
        maximum = max(maximum, difference)
        rows.append(
            {
                "absolute_difference": difference,
                "feature_fixture": fixture,
                "model_id": model_id,
                "prediction_independent": independent,
                "prediction_main": main,
                "prediction_pass": difference <= 1.0e-12,
            }
        )
    passed = (
        len(rows) == 4
        and all(row["prediction_pass"] for row in rows)
        and MODEL_FIT_CALL_COUNT == 0
        and SCALER_FIT_CALL_COUNT == 0
    )
    return {
        "FROZEN_MODEL_PREDICTION_CROSSCHECK_PASS": passed,
        "MODEL_FIT_CALL_COUNT": MODEL_FIT_CALL_COUNT,
        "SCALER_FIT_CALL_COUNT": SCALER_FIT_CALL_COUNT,
        "absolute_tolerance": 1.0e-12,
        "fixture_rng_or_seed_use_count": 0,
        "maximum_absolute_prediction_difference": maximum,
        "models": rows,
    }


__all__ = [
    "BACKENDS",
    "EXPECTED_MODEL_FILE_SHA256",
    "FINAL_MODEL_WEIGHTING",
    "FORBIDDEN_MODEL_CLAIM_TYPE",
    "MODEL_CLAIM_TYPE",
    "MODEL_FIT_CALL_COUNT",
    "MODEL_IDENTITIES",
    "SCALER_FIT_CALL_COUNT",
    "SEEDLESS_FEATURE_FIXTURE",
    "audit_frozen_model_contract",
    "crosscheck_frozen_model_predictions",
    "frozen_model_lookup",
    "load_frozen_models",
    "predict_frozen_model",
    "predict_frozen_model_independent",
]
