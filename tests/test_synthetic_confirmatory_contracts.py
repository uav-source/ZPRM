from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from phase_a_harness.local_metric_models import (
    MODEL_A_FEATURE_NAMES,
    MODEL_B_FEATURE_NAMES,
)
from phase_a_harness.synthetic_confirmatory_models import (
    MODEL_FIT_CALL_COUNT,
    SCALER_FIT_CALL_COUNT,
    SEEDLESS_FEATURE_FIXTURE,
    audit_frozen_model_contract,
    crosscheck_frozen_model_predictions,
    frozen_model_lookup,
    load_frozen_models,
    predict_frozen_model,
    predict_frozen_model_independent,
)
from phase_a_harness.synthetic_confirmatory_protocol import (
    BACKENDS,
    EXPECTED_HYPOTHESES,
    audit_confirmatory_gate_contract,
    audit_confirmatory_plan,
    audit_confirmatory_protocol_contract,
    audit_confirmatory_seed_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PLAN = ROOT / "protocols/synthetic_confirmatory_planned_snapshots.csv"
TRIAL_PLAN = ROOT / "protocols/synthetic_confirmatory_planned_trials.csv"
MODEL_PATH = ROOT / "frozen_assets/confirmatory_development_trained_models_v1.json"


def _write_stored_seed_audit(root: Path) -> None:
    protocol_dir = root / "protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (protocol_dir / "confirmatory_seed_provenance_audit.json").write_text(
        json.dumps(
            {
                "CONFIRMATORY_SEED_PROVENANCE_PASS": True,
                "confirmatory_seed_instantiation_count": 0,
                "parse_failure_count": 0,
                "structured_usage_hits": [],
            }
        ),
        encoding="utf-8",
    )


def _canonical_sha(value: object) -> str:
    import hashlib

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


def test_plan_has_exact_counts_conditions_and_backends() -> None:
    report = audit_confirmatory_plan(SNAPSHOT_PLAN, TRIAL_PLAN)
    assert report["CONFIRMATORY_PLAN_COUNT_PASS"] is True
    assert report["planned_snapshot_count"] == 595
    assert report["condition_snapshot_counts"] == {
        "FULL_NOISE": 525,
        "IDEAL_MATCHED": 35,
        "INDEPENDENT_NOISE_FREE": 35,
    }
    assert report["planned_trial_count"] == 1190
    assert report["backend_trial_counts"] == {
        "open3d_point_to_plane": 595,
        "pcl_point_to_plane": 595,
    }
    assert report["native_trial_count"] == 0


def test_plan_is_unique_and_exactly_paired() -> None:
    report = audit_confirmatory_plan(SNAPSHOT_PLAN, TRIAL_PLAN)
    assert report["CONFIRMATORY_PLAN_UNIQUENESS_PASS"] is True
    assert report["CONFIRMATORY_PLAN_PAIRING_PASS"] is True
    assert report["planned_snapshot_unique_count"] == 595
    assert report["planned_trial_unique_count"] == 1190
    assert report["duplicate_snapshot_count"] == 0
    assert report["duplicate_trial_count"] == 0
    assert report["pairing_violation_count"] == 0
    assert report["snapshot_id_formula_mismatch_count"] == 0
    assert report["trial_id_formula_mismatch_count"] == 0


def test_independent_has_no_fake_repeats_and_full_has_fifteen() -> None:
    report = audit_confirmatory_plan(SNAPSHOT_PLAN, TRIAL_PLAN)
    assert report["CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS"] is True
    assert report["independent_pseudoreplication_plan_count"] == 0
    assert report["full_noise_replicates_per_scene_geometry"] == 15


def test_plan_audit_rejects_duplicate_and_pairing_loss(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    trials = tmp_path / "trials.csv"
    shutil.copyfile(SNAPSHOT_PLAN, snapshots)
    trial_lines = TRIAL_PLAN.read_text(encoding="utf-8").splitlines()
    snapshots.write_text(
        SNAPSHOT_PLAN.read_text(encoding="utf-8") + SNAPSHOT_PLAN.read_text(encoding="utf-8").splitlines()[1] + "\n",
        encoding="utf-8",
    )
    trials.write_text("\n".join(trial_lines[:-1]) + "\n", encoding="utf-8")
    report = audit_confirmatory_plan(snapshots, trials)
    assert report["CONFIRMATORY_PLAN_COUNT_PASS"] is False
    assert report["CONFIRMATORY_PLAN_UNIQUENESS_PASS"] is False
    assert report["CONFIRMATORY_PLAN_PAIRING_PASS"] is False
    assert report["duplicate_snapshot_count"] == 1
    assert report["pairing_violation_count"] > 0


def test_gate_contract_is_exact_h1_through_h6() -> None:
    report = audit_confirmatory_gate_contract(
        ROOT / "protocols/synthetic_confirmatory_gate_contract.json"
    )
    assert report["CONFIRMATORY_GATE_CONTRACT_PASS"] is True
    assert report["hypothesis_count"] == 6
    assert set(report["hypothesis_results"]) == set(EXPECTED_HYPOTHESES)
    assert all(report["hypothesis_results"].values())


def test_gate_contract_rejects_self_consistent_threshold_change(tmp_path: Path) -> None:
    source = json.loads(
        (ROOT / "protocols/synthetic_confirmatory_gate_contract.json").read_text(
            encoding="utf-8"
        )
    )
    source["hypotheses"]["H3_CROSS_BACKEND_SCENE_RANKING"][
        "spearman_rho_min"
    ] = 0.69
    unsigned = {
        key: value
        for key, value in source.items()
        if key != "gate_contract_payload_sha256"
    }
    source["gate_contract_payload_sha256"] = _canonical_sha(unsigned)
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    report = audit_confirmatory_gate_contract(path)
    assert report["CONFIRMATORY_GATE_CONTRACT_PASS"] is False
    assert report["hypothesis_results"]["H3_CROSS_BACKEND_SCENE_RANKING"] is False


def test_seed_declarations_do_not_count_as_rng_use(tmp_path: Path) -> None:
    _write_stored_seed_audit(tmp_path)
    frozen = tmp_path / "frozen_assets"
    frozen.mkdir()
    (frozen / "synthetic_confirmatory_formal_manifest_v1.json").write_text(
        json.dumps(
            {
                "bootstrap_seed": 1083684578,
                "geometry_seeds": [248284635, 376488233],
                "formal_execution_authorized": True,
            }
        ),
        encoding="utf-8",
    )
    report = audit_confirmatory_seed_provenance(tmp_path)
    assert report["CONFIRMATORY_SEED_PROVENANCE_PASS"] is True
    assert report["CONFIRMATORY_SEED_USAGE_HIT_COUNT"] == 0
    assert report["CONFIRMATORY_SEED_INSTANTIATION_COUNT"] == 0
    assert report["CONFIRMATORY_SEED_PARSE_ERROR_COUNT"] == 0
    assert report["structured_declaration_mention_count"] == 3


@pytest.mark.parametrize("runtime_kind", ["snapshot", "trial"])
def test_runtime_snapshot_or_trial_seed_is_a_usage_hit(
    tmp_path: Path, runtime_kind: str
) -> None:
    _write_stored_seed_audit(tmp_path)
    runtime = tmp_path / "results/synthetic_confirmatory_v1"
    runtime.mkdir(parents=True)
    (runtime / f"{runtime_kind}.json").write_text(
        json.dumps({"geometry_seed": 248284635, "kind": runtime_kind}),
        encoding="utf-8",
    )
    report = audit_confirmatory_seed_provenance(tmp_path)
    assert report["CONFIRMATORY_SEED_PROVENANCE_PASS"] is False
    assert report["CONFIRMATORY_SEED_USAGE_HIT_COUNT"] == 1
    assert report["CONFIRMATORY_SEED_INSTANTIATION_COUNT"] == 1
    assert report["structured_usage_hits"][0]["relative_path"].startswith("results/")


def test_scientific_survival_and_protocol_bindings_are_exact() -> None:
    report = audit_confirmatory_protocol_contract(ROOT)
    assert report["SCIENTIFIC_SURVIVAL_BINDING_PASS"] is True
    assert report["CONFIRMATORY_PROTOCOL_BINDING_PASS"] is True
    assert report["SYNTHETIC_CONFIRMATORY_PROTOCOL_CONTRACT_AUDIT_PASS"] is True
    assert report["file_sha256_pass"] is True
    assert report["scientific_survival_file_binding_pass"] is True
    assert report["seed_provenance_audit"]["CONFIRMATORY_RNG_INSTANTIATION_COUNT"] == 0


def test_frozen_model_file_schema_feature_order_and_no_refit() -> None:
    report = audit_frozen_model_contract(ROOT)
    assert report["FROZEN_MODEL_FILE_SHA_PASS"] is True
    assert report["FROZEN_MODEL_SCHEMA_PASS"] is True
    assert report["FROZEN_MODEL_FEATURE_ORDER_PASS"] is True
    assert report["FROZEN_MODEL_NO_REFIT_CONTRACT_PASS"] is True
    assert report["MODEL_FIT_CALL_COUNT"] == 0
    assert report["SCALER_FIT_CALL_COUNT"] == 0
    assert report["forbidden_fit_calls"] == []
    assert report["training_code_sha256"]["feature_source_sha256_pass"] is True
    assert report["training_code_sha256"]["survival_model_code_sha256_pass"] is True


def test_all_four_frozen_models_have_complete_parameters() -> None:
    report = audit_frozen_model_contract(ROOT)
    assert [row["model_id"] for row in report["models"]] == [
        "open3d_point_to_plane::MODEL_A",
        "open3d_point_to_plane::MODEL_B",
        "pcl_point_to_plane::MODEL_A",
        "pcl_point_to_plane::MODEL_B",
    ]
    assert [row["feature_count"] for row in report["models"]] == [6, 12, 6, 12]
    assert all(row["parameters_complete"] for row in report["models"])
    assert all(row["alpha"] == 1.0 for row in report["models"])
    assert report["models"][0]["feature_names"] == list(MODEL_A_FEATURE_NAMES)
    assert report["models"][1]["feature_names"] == list(MODEL_B_FEATURE_NAMES)


def test_model_audit_rejects_feature_reordering_without_refit(tmp_path: Path) -> None:
    lock = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    lock["models"][0]["feature_names"][0:2] = reversed(
        lock["models"][0]["feature_names"][0:2]
    )
    unsigned = {
        key: value for key, value in lock.items() if key != "model_lock_payload_sha256"
    }
    lock["model_lock_payload_sha256"] = _canonical_sha(unsigned)
    path = tmp_path / "models.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    report = audit_frozen_model_contract(ROOT, model_path=path)
    assert report["FROZEN_MODEL_FILE_SHA_PASS"] is False
    assert report["FROZEN_MODEL_SCHEMA_PASS"] is False
    assert report["FROZEN_MODEL_FEATURE_ORDER_PASS"] is False


def test_frozen_predictions_crosscheck_all_four_models() -> None:
    report = crosscheck_frozen_model_predictions(ROOT)
    assert report["FROZEN_MODEL_PREDICTION_CROSSCHECK_PASS"] is True
    assert len(report["models"]) == 4
    assert report["maximum_absolute_prediction_difference"] <= 1.0e-12
    assert report["MODEL_FIT_CALL_COUNT"] == MODEL_FIT_CALL_COUNT == 0
    assert report["SCALER_FIT_CALL_COUNT"] == SCALER_FIT_CALL_COUNT == 0
    assert report["fixture_rng_or_seed_use_count"] == 0


def test_direct_prediction_matches_independent_scalar_implementation() -> None:
    lookup = frozen_model_lookup(load_frozen_models(MODEL_PATH))
    for model in lookup.values():
        fixture = {
            name: SEEDLESS_FEATURE_FIXTURE[name] for name in model["feature_names"]
        }
        main = predict_frozen_model(model, fixture)
        independent = predict_frozen_model_independent(model, fixture)
        assert math.isfinite(main)
        assert abs(main - independent) <= 1.0e-12


def test_direct_prediction_rejects_feature_or_value_drift() -> None:
    model = next(iter(frozen_model_lookup(load_frozen_models(MODEL_PATH)).values()))
    fixture = {name: SEEDLESS_FEATURE_FIXTURE[name] for name in model["feature_names"]}
    missing = dict(fixture)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="feature mapping mismatch"):
        predict_frozen_model(model, missing)
    nonfinite = dict(fixture)
    nonfinite[next(iter(nonfinite))] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        predict_frozen_model(model, nonfinite)
