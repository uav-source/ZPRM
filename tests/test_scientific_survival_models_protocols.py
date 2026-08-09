from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from phase_a_harness.confirmatory_protocol import (
    BACKENDS,
    build_synthetic_confirmatory_gate_contract,
    build_synthetic_confirmatory_plan,
    build_synthetic_confirmatory_protocol,
)
from phase_a_harness.real_data_protocol import (
    ELIGIBILITY_REQUIREMENTS,
    SNAPSHOT_SELECTION_FIELDS,
    UNCERTAINTY_BUDGET_FIELDS,
    build_real_data_protocol_framework,
)
from phase_a_harness.scientific_survival_models import (
    DEVELOPMENT_GEOMETRY_SEEDS,
    MODEL_CLAIM_TYPE,
    WEIGHTING_SCHEMES,
    audit_model_data_leakage,
    build_frozen_development_model_lock,
    evaluate_model_weighting_sensitivity,
    model_sample_weights,
    scan_confirmatory_seed_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    "INDEPENDENT_NOISE_FREE",
    "SCAN_NOISE_ONLY",
    "MAP_NOISE_ONLY",
    "DROPOUT_ONLY",
    "FULL_NOISE",
)


def _checksum(value: int) -> str:
    return f"{value:064x}"


def _model_rows() -> list[dict[str, object]]:
    rows = []
    index = 1
    for backend_index, backend in enumerate(BACKENDS):
        for geometry in DEVELOPMENT_GEOMETRY_SEEDS:
            for condition_index, condition in enumerate(CONDITIONS):
                for unique_index in range(4):
                    turnover = 0.08 + 0.14 * unique_index + 0.015 * condition_index
                    target_log = -3.2 + 2.4 * turnover + 0.03 * backend_index
                    for duplicate_index in range(2):
                        rows.append(
                            {
                                "accepted_source_turnover": turnover * 0.9,
                                "backend_schema_name": backend,
                                "common_association_valid": True,
                                "condition": condition,
                                "condition_number_trans": 25.0,
                                "correspondence_count_change_ratio": turnover * 0.1,
                                "correspondence_turnover": turnover,
                                "geometry_seed": geometry,
                                "initial_correspondence_count": 800,
                                "initial_residual_rmse": 0.002,
                                "initial_translation_gradient_norm": 0.0002,
                                "lambda_min_trans": 0.01,
                                "median_normal_angle_change_deg": turnover * 5.0,
                                "planned_trial_id": f"trial-{index}",
                                "q95_normal_angle_change_deg": turnover * 8.0,
                                "residual_rmse_change": turnover * 0.002,
                                "scene_variant": "GEOMETRY_RICH_ROOM",
                                "source_checksum": _checksum(
                                    backend_index * 100000
                                    + geometry
                                    + condition_index * 100
                                    + unique_index
                                ),
                                "spectral_entropy_trans": 0.7,
                                "target_checksum": _checksum(
                                    backend_index * 200000
                                    + geometry
                                    + condition_index * 100
                                    + unique_index
                                    + 1
                                ),
                                "translation_error_m": 10.0**target_log - 1.0e-9,
                            }
                        )
                        index += 1
    return rows


def _recorded_model_analysis() -> dict[str, object]:
    comparisons = []
    for backend in BACKENDS:
        folds = []
        for held in DEVELOPMENT_GEOMETRY_SEEDS:
            folds.append(
                {
                    "held_out_geometry_seed": held,
                    "training_geometry_seeds": [
                        seed for seed in DEVELOPMENT_GEOMETRY_SEEDS if seed != held
                    ],
                    "scaler_fit_scope": "training_fold_only",
                    "ridge_alpha": 1.0,
                    "ridge_fit_intercept": True,
                }
            )
        comparisons.append(
            {
                "backend_schema_name": backend,
                "fold_results": {
                    "model_a": [dict(row) for row in folds],
                    "model_b": [dict(row) for row in folds],
                },
            }
        )
    return {"ridge_model_comparison": comparisons}


def test_unique_input_weights_collapse_backend_duplicates() -> None:
    rows = _model_rows()[:8]
    original = model_sample_weights(rows, "ORIGINAL_TRIAL_WEIGHTED")
    unique = model_sample_weights(rows, "UNIQUE_INPUT_WEIGHTED")
    assert float(np.sum(original)) == 8.0
    assert float(np.sum(unique)) == 4.0
    assert set(unique.tolist()) == {0.5}


def test_condition_balanced_weights_equalize_five_conditions() -> None:
    rows = _model_rows()
    selected = [row for row in rows if row["backend_schema_name"] == BACKENDS[0]]
    # Deliberately add copies from one condition; unique-input weights must
    # neutralize them before condition totals are balanced.
    selected.extend(dict(row, planned_trial_id=f"copy-{index}") for index, row in enumerate(selected[:8]))
    weights = model_sample_weights(selected, "UNIQUE_INPUT_CONDITION_BALANCED")
    totals: dict[str, float] = defaultdict(float)
    for row, weight in zip(selected, weights):
        totals[str(row["condition"])] += float(weight)
    assert set(totals) == set(CONDITIONS)
    assert max(totals.values()) - min(totals.values()) <= 1.0e-12


def test_model_survival_three_schemes_and_per_fold_gate() -> None:
    report = evaluate_model_weighting_sensitivity(_model_rows())
    assert report["MODEL_INCREMENTAL_VALUE_ROBUST_PASS"] is True
    assert len(report["model_weighting_sensitivity"]) == 6
    assert len(report["model_fold_results"]) == 18
    assert {
        row["weighting_scheme"] for row in report["model_weighting_sensitivity"]
    } == set(WEIGHTING_SCHEMES)
    assert all(row["relative_improvement"] >= 0.10 for row in report["model_weighting_sensitivity"])
    assert all(row["model_b_better_fold_count"] >= 2 for row in report["model_weighting_sensitivity"])
    assert all(row["minimum_fold_relative_improvement"] >= -0.10 for row in report["model_weighting_sensitivity"])


def test_frozen_development_model_serialization_has_four_models() -> None:
    lock = build_frozen_development_model_lock(_model_rows(), repository_root=ROOT)
    assert lock["model_count"] == 4
    assert lock["confirmatory_seed_access_count"] == 0
    assert len(lock["models"]) == 4
    assert {row["model"] for row in lock["models"]} == {"A", "B"}
    assert all(row["alpha"] == 1.0 for row in lock["models"])
    assert all(row["scaler_fit_scope"] == "ALL_DEVELOPMENT_ROWS_ONLY" for row in lock["models"])
    assert all(len(row["training_data_sha256"]) == 64 for row in lock["models"])


def test_model_code_and_fold_leakage_audit() -> None:
    report = audit_model_data_leakage(
        _recorded_model_analysis(),
        repository_root=ROOT,
        seed_provenance={"CONFIRMATORY_SEED_PROVENANCE_PASS": True},
    )
    assert report["MODEL_DATA_LEAKAGE_AUDIT_PASS"] is True
    assert report["MODEL_CLAIM_TYPE"] == MODEL_CLAIM_TYPE
    assert all(row["pass"] is True for row in report["rows"])


def test_confirmatory_seed_scan_ignores_prose_and_protocol_declaration(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/readme.md").write_text("248284635", encoding="utf-8")
    (tmp_path / "frozen_assets").mkdir()
    (tmp_path / "frozen_assets/example_protocol.json").write_text(
        json.dumps({"geometry_seed": 248284635}), encoding="utf-8"
    )
    report = scan_confirmatory_seed_provenance(tmp_path)
    assert report["CONFIRMATORY_SEED_PROVENANCE_PASS"] is True
    assert report["confirmatory_seed_instantiation_count"] == 0


def test_confirmatory_seed_scan_detects_structured_runtime_use(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results/run_manifest.json").write_text(
        json.dumps({"geometry_seed": 248284635}), encoding="utf-8"
    )
    report = scan_confirmatory_seed_provenance(tmp_path)
    assert report["CONFIRMATORY_SEED_PROVENANCE_PASS"] is False
    assert report["confirmatory_seed_instantiation_count"] == 1


def test_confirmatory_seed_scan_detects_rng_provenance_seed_list(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results/rng_provenance.json").write_text(
        json.dumps({"seeds_used": [248284635]}), encoding="utf-8"
    )
    report = scan_confirmatory_seed_provenance(tmp_path)
    assert report["CONFIRMATORY_SEED_PROVENANCE_PASS"] is False
    assert report["structured_usage_hits"][0]["field_path"] == "seeds_used.0"


def test_confirmatory_plan_is_exact_and_independent_has_no_fake_repeat() -> None:
    snapshots, trials = build_synthetic_confirmatory_plan()
    assert len(snapshots) == 595
    assert len(trials) == 1190
    assert sum(row["condition"] == "IDEAL_MATCHED" for row in snapshots) == 35
    independent = [row for row in snapshots if row["condition"] == "INDEPENDENT_NOISE_FREE"]
    assert len(independent) == 35
    assert all(row["measurement_seed"] is None and row["repeat_index"] == 0 for row in independent)
    assert sum(row["condition"] == "FULL_NOISE" for row in snapshots) == 525
    assert {row["backend"] for row in trials} == set(BACKENDS)


def test_confirmatory_h1_h6_contract_thresholds() -> None:
    contract = build_synthetic_confirmatory_gate_contract()
    hypotheses = contract["hypotheses"]
    assert len(hypotheses) == 6
    assert hypotheses["H1_IDEAL_CONTROL"]["translation_q95_max_m"] == 0.001
    assert hypotheses["H2_LONG_CORRIDOR_SCENE_EFFECT"]["minimum_long_greater_than_rich_blocks"] == 4
    assert hypotheses["H3_CROSS_BACKEND_SCENE_RANKING"]["spearman_rho_min"] == 0.70
    assert hypotheses["H4_REASSOCIATION_MECHANISM"]["pooled_turnover_error_spearman_rho_min"] == 0.40
    assert hypotheses["H5_FROZEN_MODEL_B_INCREMENTAL_VALUE"]["mae_b_to_mae_a_ratio_max"] == 0.90
    assert hypotheses["H6_FULL_NOISE_SYSTEMATIC_OFFSET"]["effective_replicate_count_min"] == 12


def test_confirmatory_protocol_fails_closed_then_builds_in_memory() -> None:
    lock = build_frozen_development_model_lock(_model_rows(), repository_root=ROOT)
    with pytest.raises(PermissionError):
        build_synthetic_confirmatory_protocol(
            scientific_survival_decision={"SCIENTIFIC_SURVIVAL_AUDIT_PASS": False},
            seed_provenance={"CONFIRMATORY_SEED_PROVENANCE_PASS": True},
            model_lock=lock,
        )
    bundle = build_synthetic_confirmatory_protocol(
        scientific_survival_decision={"SCIENTIFIC_SURVIVAL_AUDIT_PASS": True},
        seed_provenance={"CONFIRMATORY_SEED_PROVENANCE_PASS": True},
        model_lock=lock,
    )
    assert bundle["protocol"]["SYNTHETIC_CONFIRMATORY_PROTOCOL_READY"] is True
    assert bundle["protocol"]["CONFIRMATORY_RUN_AUTHORIZED"] is False
    assert bundle["protocol"]["REAL_DATA_RUN_AUTHORIZED"] is False
    assert bundle["protocol"]["MEASUREMENT_PAPER_MAINLINE_AUTHORIZED"] is False
    assert bundle["protocol"]["registration_execution_count"] == 0
    assert bundle["protocol"]["snapshot_generation_count"] == 0


def test_real_data_framework_has_exact_fourteen_requirements_and_templates() -> None:
    bundle = build_real_data_protocol_framework(scientific_survival_audit_pass=True)
    assert len(ELIGIBILITY_REQUIREMENTS) == 14
    assert len(bundle["checklist"]) == 14
    assert len({row["requirement_id"] for row in bundle["checklist"]}) == 14
    assert len(SNAPSHOT_SELECTION_FIELDS) >= 14
    assert len(UNCERTAINTY_BUDGET_FIELDS) >= 14
    assert bundle["protocol"]["REAL_DATA_PROTOCOL_FRAMEWORK_READY"] is True
    assert bundle["protocol"]["REAL_DATA_DATASET_ELIGIBILITY_COMPLETE"] is False
    assert bundle["protocol"]["REAL_DATA_RUN_AUTHORIZED"] is False
    assert bundle["protocol"]["CONFIRMATORY_RUN_AUTHORIZED"] is False
    assert bundle["protocol"]["MEASUREMENT_PAPER_MAINLINE_AUTHORIZED"] is False
    assert bundle["protocol"]["dataset_download_count"] == 0


def test_real_data_framework_stops_when_survival_fails() -> None:
    with pytest.raises(PermissionError):
        build_real_data_protocol_framework(scientific_survival_audit_pass=False)
