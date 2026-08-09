from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.scientific_survival_replicates import (
    BACKENDS,
    DETERMINISTIC_TERM,
    GEOMETRY_SEEDS,
    LIMITED_TERM,
    MEASUREMENT_SEEDS,
    NONIDEAL_CONDITIONS,
    PRIMARY_CONDITIONS,
    REPEATABLE_TERM,
    REPEAT_INDICES,
    SCENES,
    attach_observed_component_signatures,
    audit_replicate_uniqueness,
    classify_replicate_count,
    collapse_trials_to_unique_inputs,
    derive_observed_component_signatures,
    evaluate_cross_backend_unique_unit,
    evaluate_long_corridor_systematic_offset,
    evaluate_turnover_robustness,
    evaluate_unique_unit_primary_scene_effect,
)


def _cell_snapshots(
    *, scene: str, geometry_seed: int, condition: str, unique_count: int
) -> list[dict]:
    rows: list[dict] = []
    for measurement_index, measurement_seed in enumerate(MEASUREMENT_SEEDS):
        for repeat in REPEAT_INDICES:
            observation = measurement_index * 5 + repeat
            pair_index = observation % unique_count
            rows.append(
                {
                    "condition": condition,
                    "geometry_seed": geometry_seed,
                    "measurement_seed": measurement_seed,
                    "repeat_index": repeat,
                    "scene_variant": scene,
                    "snapshot_id": (
                        f"{scene}/{geometry_seed}/{condition}/"
                        f"{measurement_seed}/{repeat}"
                    ),
                    "source_checksum": f"source-{scene}-{geometry_seed}-{condition}-{pair_index}",
                    "source_point_count": 100 - pair_index,
                    "target_checksum": f"target-{scene}-{geometry_seed}-{condition}-{pair_index}",
                    "target_point_count": 100,
                }
            )
    return rows


def _single_trial(
    snapshot: dict,
    *,
    backend: str,
    error: float,
    turnover: float | None = None,
    suffix: str = "",
) -> dict:
    row = {
        "backend": backend,
        "condition": snapshot["condition"],
        "geometry_seed": snapshot["geometry_seed"],
        "planned_trial_id": f"{snapshot['snapshot_id']}/{backend}/{suffix}",
        "scene_variant": snapshot["scene_variant"],
        "snapshot_id": snapshot["snapshot_id"],
        "translation_error_m": error,
    }
    if turnover is not None:
        row.update(
            common_association_valid=True,
            correspondence_turnover=turnover,
        )
    return row


def test_frozen_replicate_class_boundaries_and_terms() -> None:
    assert classify_replicate_count(1) == "DETERMINISTIC_SINGLE_INPUT"
    assert classify_replicate_count(2) == "PARTIAL_REPLICATION"
    assert classify_replicate_count(7) == "PARTIAL_REPLICATION"
    assert classify_replicate_count(8) == "FULL_REPLICATION"
    assert classify_replicate_count(10) == "FULL_REPLICATION"
    with pytest.raises(ValueError, match=r"\[1, 10\]"):
        classify_replicate_count(0)


def test_unique_pair_audit_counts_inputs_not_backends_and_authorizes_balanced_full() -> None:
    deterministic = _cell_snapshots(
        scene="LONG_CORRIDOR",
        geometry_seed=GEOMETRY_SEEDS[0],
        condition="INDEPENDENT_NOISE_FREE",
        unique_count=1,
    )
    partial = _cell_snapshots(
        scene="LONG_CORRIDOR",
        geometry_seed=GEOMETRY_SEEDS[0],
        condition="DROPOUT_ONLY",
        unique_count=5,
    )
    full = _cell_snapshots(
        scene="LONG_CORRIDOR",
        geometry_seed=GEOMETRY_SEEDS[0],
        condition="FULL_NOISE",
        unique_count=10,
    )
    audit = audit_replicate_uniqueness(
        [*deterministic, *partial, *full], strict_contract=False
    )
    by_condition = {row["condition"]: row for row in audit["cell_rows"]}
    assert by_condition["INDEPENDENT_NOISE_FREE"]["effective_replicate_count"] == 1
    assert by_condition["INDEPENDENT_NOISE_FREE"]["duplicate_pair_count"] == 9
    assert by_condition["INDEPENDENT_NOISE_FREE"]["authorized_term"] == DETERMINISTIC_TERM
    assert by_condition["DROPOUT_ONLY"]["effective_replicate_count"] == 5
    assert by_condition["DROPOUT_ONLY"]["authorized_term"] == LIMITED_TERM
    assert by_condition["FULL_NOISE"]["effective_replicate_count"] == 10
    assert by_condition["FULL_NOISE"]["authorized_term"] == REPEATABLE_TERM
    assert by_condition["FULL_NOISE"]["systematic_offset_claim_authorized"] is True
    assert audit["PSEUDOREPLICATION_RISK_IDENTIFIED"] is True
    assert audit["REPLICATE_UNIQUENESS_AUDIT_PASS"] is True


def test_measurement_repeat_noise_and_dropout_effectiveness_are_audited() -> None:
    rows = _cell_snapshots(
        scene="GEOMETRY_RICH_ROOM",
        geometry_seed=GEOMETRY_SEEDS[1],
        condition="FULL_NOISE",
        unique_count=10,
    )
    result = audit_replicate_uniqueness(rows, strict_contract=False)
    cell = result["cell_rows"][0]
    assert cell["measurement_seed_changes_input"] is True
    assert cell["repeat_index_changes_input"] is True
    assert cell["unique_noise_checksum_count"] == 10
    assert cell["unique_dropout_checksum_count"] == 10
    assert {row["unique_pair_count"] for row in result["measurement_seed_rows"]} == {5}
    assert {row["unique_pair_count"] for row in result["repeat_index_rows"]} == {2}
    duplicate = dict(rows[0])
    with pytest.raises(ValueError, match="duplicate snapshot identity"):
        audit_replicate_uniqueness([*rows, duplicate], strict_contract=False)


def test_observed_component_signatures_reconstruct_exact_and_inferred_masks() -> None:
    base_source = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype="<f4",
    )
    base_target = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype="<f4",
    )
    exact = derive_observed_component_signatures(
        condition="DROPOUT_ONLY",
        source_points=base_source[[0, 2, 4]],
        target_points=base_target,
        independent_source_points=base_source,
        independent_target_points=base_target,
    )
    assert exact["dropout_kept_source_count"] == 3
    assert exact["dropout_mask_reconstruction_pass"] is True
    assert exact["dropout_signature_basis"].startswith("EXACT_ORDERED_SUBSEQUENCE")
    noisy_source = (base_source + np.float32(0.01))[[0, 1, 3, 4]]
    noisy_target = base_target + np.float32(0.001)
    inferred = derive_observed_component_signatures(
        condition="FULL_NOISE",
        source_points=noisy_source,
        target_points=noisy_target,
        independent_source_points=base_source,
        independent_target_points=base_target,
    )
    assert inferred["dropout_kept_source_count"] == 4
    assert inferred["dropout_mask_reconstruction_pass"] is True
    assert "NEAREST_BASELINE_INDEX" in inferred["dropout_signature_basis"]
    assert inferred["dropout_signature_limitation"] is not None
    assert inferred["full_noise_nearest_index_max_distance_m"] > 0.0

    arrays = {
        "INDEPENDENT_NOISE_FREE": {"source": base_source, "target": base_target},
        "SCAN_NOISE_ONLY": {
            "source": base_source + np.float32(0.01),
            "target": base_target,
        },
        "MAP_NOISE_ONLY": {
            "source": base_source,
            "target": noisy_target,
        },
        "DROPOUT_ONLY": {
            "source": base_source[[0, 2, 4]],
            "target": base_target,
        },
        "FULL_NOISE": {"source": noisy_source, "target": noisy_target},
    }
    rows = [
        {
            "condition": condition,
            "geometry_seed": GEOMETRY_SEEDS[0],
            "measurement_seed": MEASUREMENT_SEEDS[0],
            "repeat_index": 0,
            "scene_variant": "LONG_CORRIDOR",
            "snapshot_id": condition,
            "source_checksum": f"source-{condition}",
            "target_checksum": f"target-{condition}",
        }
        for condition in NONIDEAL_CONDITIONS
    ]
    enriched = attach_observed_component_signatures(
        rows, lambda row: arrays[row["condition"]]
    )
    assert len(enriched) == 5
    assert all(row["observed_component_signature_pass"] for row in enriched)
    assert all(len(row["noise_checksum"]) == 64 for row in enriched)


def test_long_corridor_full_noise_gate_and_independent_wording() -> None:
    snapshots: list[dict] = []
    for condition, count in (
        ("INDEPENDENT_NOISE_FREE", 1),
        ("FULL_NOISE", 10),
    ):
        for seed in GEOMETRY_SEEDS:
            snapshots.extend(
                _cell_snapshots(
                    scene="LONG_CORRIDOR",
                    geometry_seed=seed,
                    condition=condition,
                    unique_count=count,
                )
            )
    cell_rows = audit_replicate_uniqueness(
        snapshots, strict_contract=False
    )["cell_rows"]
    systematic: list[dict] = []
    for backend in BACKENDS:
        for condition in PRIMARY_CONDITIONS:
            for index, seed in enumerate(GEOMETRY_SEEDS):
                systematic.append(
                    {
                        "backend": backend,
                        "condition": condition,
                        "geometry_seed": seed,
                        "scene_variant": "LONG_CORRIDOR",
                        "systematic_translation_offset_m": 0.01
                        if index < 2
                        else 0.004,
                        "translation_repeatability_rms_m": 0.001,
                        "systematic_fraction_translation": 0.8,
                        "translation_direction_concentration": 0.9,
                    }
                )
    result = evaluate_long_corridor_systematic_offset(cell_rows, systematic)
    assert result["SYSTEMATIC_OFFSET_FULL_NOISE_PASS"] is True
    assert result["SYSTEMATIC_OFFSET_CLAIM_SCOPE"] == "LONG_CORRIDOR_FULL_NOISE"
    assert result["INDEPENDENT_NOISE_FREE_AUTHORIZED_TERM"] == DETERMINISTIC_TERM
    assert {row["qualifying_geometry_group_count"] for row in result["backend_rows"]} == {2}


def test_unique_input_collapse_medians_duplicate_executions_once() -> None:
    base = {
        "condition": "FULL_NOISE",
        "geometry_seed": GEOMETRY_SEEDS[0],
        "scene_variant": "LONG_CORRIDOR",
        "snapshot_id": "same-a",
        "source_checksum": "source-a",
        "target_checksum": "target-a",
    }
    duplicate_snapshot = {**base, "snapshot_id": "same-b"}
    trials = [
        _single_trial(base, backend="Open3D", error=0.1, suffix="a"),
        _single_trial(duplicate_snapshot, backend="Open3D", error=0.3, suffix="b"),
    ]
    collapsed = collapse_trials_to_unique_inputs(trials, [base, duplicate_snapshot])
    assert len(collapsed) == 1
    assert collapsed[0]["trial_count_collapsed"] == 2
    assert collapsed[0]["translation_error_m"] == pytest.approx(0.2)


def _primary_inputs() -> tuple[list[dict], list[dict]]:
    snapshots: list[dict] = []
    trials: list[dict] = []
    for condition in PRIMARY_CONDITIONS:
        for scene in ("LONG_CORRIDOR", "GEOMETRY_RICH_ROOM"):
            for seed_index, seed in enumerate(GEOMETRY_SEEDS):
                for repeat in range(2):
                    snapshot = {
                        "condition": condition,
                        "geometry_seed": seed,
                        "scene_variant": scene,
                        "snapshot_id": f"{condition}/{scene}/{seed}/{repeat}",
                        "source_checksum": f"source-{condition}-{scene}-{seed}",
                        "target_checksum": f"target-{condition}-{scene}-{seed}",
                    }
                    snapshots.append(snapshot)
                    for backend in BACKENDS:
                        rich = 0.001 + seed_index * 0.0001
                        error = rich if scene == "GEOMETRY_RICH_ROOM" else rich + 0.02
                        trials.append(
                            _single_trial(
                                snapshot,
                                backend=backend,
                                error=error,
                                suffix=str(repeat),
                            )
                        )
    return snapshots, trials


def test_unique_unit_primary_scene_effect_uses_three_geometry_blocks() -> None:
    snapshots, trials = _primary_inputs()
    result = evaluate_unique_unit_primary_scene_effect(trials, snapshots)
    assert result["PRIMARY_SCENE_EFFECT_UNIQUE_UNIT_PASS"] is True
    assert len(result["rows"]) == 4
    assert all(row["geometry_level_paired_win_count"] == 3 for row in result["rows"])
    assert all(row["unique_input_corridor_count"] == 3 for row in result["rows"])
    assert all(row["weak_rich_median_ratio"] >= 5.0 for row in result["rows"])


def _ranking_inputs() -> tuple[list[dict], list[dict]]:
    snapshots: list[dict] = []
    trials: list[dict] = []
    for condition_index, condition in enumerate(NONIDEAL_CONDITIONS):
        for scene_index, scene in enumerate(SCENES):
            snapshot = {
                "condition": condition,
                "geometry_seed": GEOMETRY_SEEDS[0],
                "scene_variant": scene,
                "snapshot_id": f"rank/{condition}/{scene}",
                "source_checksum": f"rank-source-{condition}-{scene}",
                "target_checksum": f"rank-target-{condition}-{scene}",
            }
            snapshots.append(snapshot)
            base = 0.001 + condition_index * 0.02 + scene_index * 0.001
            trials.extend(
                _single_trial(snapshot, backend=backend, error=base * scale)
                for backend, scale in (("Open3D", 1.0), ("PCL", 2.0))
            )
    return snapshots, trials


def test_cross_backend_unique_unit_applies_all_five_preregistered_thresholds() -> None:
    snapshots, trials = _ranking_inputs()
    result = evaluate_cross_backend_unique_unit(trials, snapshots)
    assert result["CROSS_BACKEND_UNIQUE_UNIT_PASS"] is True
    assert result["condition_rho_at_least_0_50_count"] == 5
    assert result["condition_rho_median"] == pytest.approx(1.0)
    assert result["pooled_scene_condition_spearman_rho"] == pytest.approx(1.0)


def _turnover_inputs(*, inverse: bool = False) -> tuple[list[dict], list[dict]]:
    snapshots: list[dict] = []
    trials: list[dict] = []
    for condition_index, condition in enumerate(NONIDEAL_CONDITIONS):
        for scene_index, scene in enumerate(SCENES):
            for geometry_index, seed in enumerate(GEOMETRY_SEEDS):
                snapshot = {
                    "condition": condition,
                    "geometry_seed": seed,
                    "scene_variant": scene,
                    "snapshot_id": f"turnover/{condition}/{scene}/{seed}",
                    "source_checksum": f"turn-source-{condition}-{scene}-{seed}",
                    "target_checksum": f"turn-target-{condition}-{scene}-{seed}",
                }
                snapshots.append(snapshot)
                turnover = 0.1 + geometry_index * 0.2
                direction = -turnover if inverse else turnover
                log_error = -4.0 + direction + condition_index * 0.001 + scene_index * 0.0001
                error = 10.0**log_error
                for backend in BACKENDS:
                    trials.append(
                        _single_trial(
                            snapshot,
                            backend=backend,
                            error=error,
                            turnover=turnover,
                        )
                    )
    return snapshots, trials


def test_turnover_robustness_reports_pooled_centered_loso_loco_and_unique() -> None:
    snapshots, trials = _turnover_inputs()
    result = evaluate_turnover_robustness(trials, snapshots)
    assert result["REASSOCIATION_ROBUSTNESS_PASS"] is True
    assert result["CAUSAL_REASSOCIATION_CLAIM_AUTHORIZED"] is False
    assert len(result["sensitivity_rows"]) == 2 * (7 + 5)
    for row in result["backend_rows"]:
        assert row["pooled_spearman_rho"] >= 0.40
        assert row["centered_spearman_rho"] >= 0.20
        assert row["leave_one_scene_out_positive_count"] == 7
        assert row["leave_one_condition_out_positive_count"] == 5
        assert row["unique_input_spearman_rho"] >= 0.20


def test_turnover_robustness_fails_closed_for_inverse_relationship() -> None:
    snapshots, trials = _turnover_inputs(inverse=True)
    result = evaluate_turnover_robustness(trials, snapshots)
    assert result["REASSOCIATION_ROBUSTNESS_PASS"] is False
    assert all(
        row["reassociation_robustness_backend_pass"] is False
        for row in result["backend_rows"]
    )
