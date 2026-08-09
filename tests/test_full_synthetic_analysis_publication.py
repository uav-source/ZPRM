from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from phase_a_harness.contracts import canonical_json_sha256, file_sha256
from phase_a_harness import full_synthetic_analysis as analysis_module
from phase_a_harness import full_synthetic_independent_verifier as independent_module
from phase_a_harness import full_synthetic_publisher as publisher_module
from phase_a_harness.full_synthetic_analysis import (
    BACKEND_LABELS,
    BACKENDS,
    CONDITIONS,
    GEOMETRY_SEEDS,
    MEASUREMENT_SEEDS,
    NONIDEAL_CONDITIONS,
    REPEAT_INDICES,
    SCENES,
    analyze_full_synthetic_record_collections,
    condition_contrast_rows,
    full_synthetic_verification_projection,
    primary_scene_effect_rows,
    systematic_offset_rows,
    transform_error_vectors,
)
from phase_a_harness.full_synthetic_artifact_verifier import (
    FIGURES,
    FIXED_ROW_COUNTS,
    REQUIRED_FILES,
    SHA256_EXCLUDED_FILES,
    TABLES,
    verify_full_synthetic_artifact,
)
from phase_a_harness.full_synthetic_independent_verifier import (
    _independent_vectors,
    full_synthetic_analysis_verifier_comparison,
    independently_recompute_common_association_records,
    independently_recompute_full_synthetic,
)
from phase_a_harness.phase_a_trial_result_schema import OPEN3D_BACKEND


def _integrity() -> dict[str, object]:
    return {
        "PHASE_A_IDEAL_IMPORT_PASS": True,
        "PHASE_B_SUBSET_REPRODUCTION_PASS": True,
        "backend_input_checksum_mismatch_count": 0,
        "combined_snapshot_count": 1260,
        "combined_trial_count": 2520,
        "corrupt_trial_count": 0,
        "duplicate_trial_count": 0,
        "extra_trial_count": 0,
        "infrastructure_interruption_unresolved_count": 0,
        "metric_recomputation_mismatch_count": 0,
        "missing_trial_count": 0,
        "new_snapshot_count": 1050,
        "new_trial_count": 2100,
        "planned_new_snapshot_count": 1050,
        "planned_new_trial_count": 2100,
    }


def _synthetic_matrix(*, valid_model_subset: bool = False):
    scene_scale = dict(zip(SCENES, (1.0, 12.0, 6.0, 2.0, 5.0, 10.0, 7.0)))
    trials = []
    common = []
    for scene in SCENES:
        for condition_index, condition in enumerate(CONDITIONS):
            for geometry_index, geometry in enumerate(GEOMETRY_SEEDS):
                for measurement_index, measurement in enumerate(MEASUREMENT_SEEDS):
                    for repeat in REPEAT_INDICES:
                        snapshot_id = (
                            f"{scene}/{condition}/{geometry}/{measurement}/{repeat}"
                        )
                        for backend in BACKENDS:
                            trial_id = f"{snapshot_id}/{backend}"
                            error = (
                                0.001
                                * scene_scale[scene]
                                * (1.0 + 0.2 * condition_index)
                                * (
                                    1.0
                                    + 0.11 * geometry_index
                                    + 0.013 * measurement_index
                                    + 0.007 * repeat
                                )
                                * (1.0 if backend == OPEN3D_BACKEND else 1.02)
                            )
                            trials.append(
                                {
                                    "backend": BACKEND_LABELS[backend],
                                    "backend_schema_name": backend,
                                    "condition": condition,
                                    "failure_classification": "NONE",
                                    "finite_output": True,
                                    "geometry_seed": geometry,
                                    "measurement_seed": measurement,
                                    "planned_trial_id": trial_id,
                                    "reference_pose_checksum": "c" * 64,
                                    "repeat_index": repeat,
                                    "rotation_error_rad": error / 10.0,
                                    "rotation_vector": [error / 10.0, 0.0, 0.0],
                                    "runtime_ms": 2.0 if backend == OPEN3D_BACKEND else 3.0,
                                    "scene_variant": scene,
                                    "snapshot_checksum": "d" * 64,
                                    "snapshot_id": snapshot_id,
                                    "solver_failure": False,
                                    "source_checksum": "a" * 64,
                                    "target_checksum": "b" * 64,
                                    "translation_error_m": error,
                                    "translation_vector": [error, 0.0, 0.0],
                                }
                            )
                            if condition == "IDEAL_MATCHED":
                                continue
                            valid = bool(
                                valid_model_subset
                                and condition == "FULL_NOISE"
                                and scene == "GEOMETRY_RICH_ROOM"
                            )
                            record = {
                                "common_association_invalid_reason": (
                                    None if valid else "OTHER"
                                ),
                                "common_association_valid": valid,
                                "planned_trial_id": trial_id,
                                "snapshot_id": snapshot_id,
                            }
                            if valid:
                                record.update(
                                    {
                                        "accepted_source_turnover": 0.05 + repeat * 0.005,
                                        "condition_number_trans": 10.0 + geometry_index,
                                        "correspondence_count_change_ratio": 0.01 * repeat,
                                        "correspondence_turnover": 0.1 + repeat * 0.01,
                                        "initial_correspondence_count": 100 + repeat,
                                        "initial_residual_rmse": 0.01 + geometry_index * 0.001,
                                        "initial_translation_gradient_norm": 0.01,
                                        "lambda_max_trans": 0.6,
                                        "lambda_mid_trans": 0.3,
                                        "lambda_min_trans": 0.1 + geometry_index * 0.01,
                                        "median_normal_angle_change_deg": 1.0 + repeat * 0.1,
                                        "q95_normal_angle_change_deg": 2.0 + repeat * 0.1,
                                        "residual_rmse_change": 0.001 * repeat,
                                        "spectral_entropy_trans": 0.7,
                                    }
                                )
                            common.append(record)
    return trials, common


def test_rotation_vector_contract_projects_relative_so3_and_matches_independent():
    reference = np.eye(4)
    reference[:3, :3] += np.asarray(
        [[0.0, 2e-13, 0.0], [-1e-13, 0.0, 1e-13], [0.0, 0.0, 0.0]]
    )
    estimate = np.eye(4)
    angle = np.deg2rad(0.5)
    estimate[:3, :3] = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    estimate[0, 3] = 0.01
    primary = transform_error_vectors(reference, estimate)
    independent = _independent_vectors(reference, estimate)
    comparison = independent_module._deep_numeric_comparison(primary, independent)
    assert comparison[0] == 0
    assert comparison[1] <= 1.0e-12
    assert np.isclose(np.linalg.norm(primary["rotation_vector"]), primary["rotation_error_rad"])


def test_systematic_summary_requires_exactly_ten_successes_and_ddof_one():
    base = {
        "backend_schema_name": BACKENDS[0],
        "condition": "FULL_NOISE",
        "finite_output": True,
        "geometry_seed": GEOMETRY_SEEDS[0],
        "rotation_vector": [0.0, 0.01, 0.0],
        "scene_variant": "LONG_CORRIDOR",
        "solver_failure": False,
        "translation_vector": [0.01, 0.0, 0.0],
    }
    complete = [
        {**base, "measurement_seed": MEASUREMENT_SEEDS[index // 5], "repeat_index": index % 5}
        for index in range(10)
    ]
    rows = systematic_offset_rows(complete)
    selected = next(
        row for row in rows
        if row["scene_variant"] == "LONG_CORRIDOR"
        and row["geometry_seed"] == GEOMETRY_SEEDS[0]
        and row["condition"] == "FULL_NOISE"
        and row["backend_schema_name"] == BACKENDS[0]
    )
    assert selected["successful_observation_count"] == 10
    assert selected["translation_repeatability_rms_m"] == pytest.approx(0.0, abs=1e-15)
    incomplete = systematic_offset_rows(complete[:-1])
    selected_incomplete = next(
        row for row in incomplete
        if row["scene_variant"] == "LONG_CORRIDOR"
        and row["geometry_seed"] == GEOMETRY_SEEDS[0]
        and row["condition"] == "FULL_NOISE"
        and row["backend_schema_name"] == BACKENDS[0]
    )
    assert selected_incomplete["successful_observation_count"] == 9
    assert selected_incomplete["systematic_translation_offset_m"] is None


def test_hierarchical_bootstrap_preserves_inner_blocks_and_paired_contrast_definition():
    records = []
    for geometry in GEOMETRY_SEEDS:
        for measurement in MEASUREMENT_SEEDS:
            for repeat in REPEAT_INDICES:
                for value in (-1.0, 1.0):
                    records.append(
                        {
                            "geometry_seed": geometry,
                            "measurement_seed": measurement,
                            "repeat_index": repeat,
                            "value": value,
                        }
                    )
    low, high = analysis_module._hierarchical_interval(
        records,
        lambda sample: np.mean([row["value"] for row in sample]),
        repetitions=30,
    )
    assert low == high == 0.0

    contrast_trials = []
    for geometry_index, geometry in enumerate(GEOMETRY_SEEDS):
        baseline, contrast = ((0.0, 1.0), (0.0, 101.0), (100.0, 101.0))[geometry_index]
        for measurement in MEASUREMENT_SEEDS:
            for repeat in REPEAT_INDICES:
                for condition, value in (
                    ("INDEPENDENT_NOISE_FREE", baseline),
                    ("FULL_NOISE", contrast),
                ):
                    contrast_trials.append(
                        {
                            "backend_schema_name": BACKENDS[0],
                            "condition": condition,
                            "finite_output": True,
                            "geometry_seed": geometry,
                            "measurement_seed": measurement,
                            "repeat_index": repeat,
                            "scene_variant": "GEOMETRY_RICH_ROOM",
                            "solver_failure": False,
                            "translation_error_m": value,
                        }
                    )
    row = next(
        item for item in condition_contrast_rows(
            contrast_trials, bootstrap_repetitions=5
        )
        if item["backend_schema_name"] == BACKENDS[0]
        and item["scene_variant"] == "GEOMETRY_RICH_ROOM"
        and item["contrast_condition"] == "FULL_NOISE"
    )
    assert row["paired_median_difference_m"] == 1.0
    assert row["contrast_median_m"] - row["baseline_median_m"] == 101.0


def test_primary_scene_gate_uses_registered_24_of_30_wins_without_private_intersection_gate():
    trials = []
    blocks = [
        (geometry, measurement, repeat)
        for geometry in GEOMETRY_SEEDS
        for measurement in MEASUREMENT_SEEDS
        for repeat in REPEAT_INDICES
    ]
    for scene, excluded, error in (
        ("GEOMETRY_RICH_ROOM", set(blocks[:3]), 0.001),
        ("LONG_CORRIDOR", set(blocks[-3:]), 0.010),
    ):
        for geometry, measurement, repeat in blocks:
            if (geometry, measurement, repeat) in excluded:
                continue
            trials.append(
                {
                    "backend_schema_name": BACKENDS[0],
                    "condition": "FULL_NOISE",
                    "finite_output": True,
                    "geometry_seed": geometry,
                    "measurement_seed": measurement,
                    "repeat_index": repeat,
                    "scene_variant": scene,
                    "solver_failure": False,
                    "translation_error_m": error,
                }
            )
    rows, _ = primary_scene_effect_rows(trials, bootstrap_repetitions=0)
    result = next(
        row for row in rows
        if row["condition"] == "FULL_NOISE"
        and row["backend_schema_name"] == BACKENDS[0]
    )
    assert result["matched_block_count"] == 24
    assert result["paired_win_count"] == 24
    assert result["rich_success_rate"] == result["corridor_success_rate"] == 0.9
    assert result["gate_pass"] is True


def test_full_projection_matches_independent_with_hierarchical_cross_backend_intervals():
    trials, common = _synthetic_matrix()
    primary = analyze_full_synthetic_record_collections(
        trials=trials,
        common_records=common,
        integrity=_integrity(),
        bootstrap_repetitions=5,
    )
    independent = independently_recompute_full_synthetic(
        trials=trials,
        common_records=common,
        integrity=_integrity(),
        bootstrap_repetitions=5,
    )
    comparison = full_synthetic_analysis_verifier_comparison(primary, independent)
    assert comparison["section_difference_count"] == 0
    pooled = next(
        row for row in primary["cross_backend_ranking"]
        if row["scope"] == "POOLED_SCENE_CONDITION"
    )
    assert pooled["exploratory_spearman_ci95_low"] is not None
    assert pooled["exploratory_spearman_ci95_high"] is not None
    discrete_tamper = json.loads(json.dumps(independent))
    discrete_tamper["integrity"]["combined_trial_count"] = 2520.0
    assert full_synthetic_analysis_verifier_comparison(
        primary, discrete_tamper
    )["section_difference_count"] == 1


def test_independent_systematic_ridge_and_candidate_formulas_match_primary_projection():
    trials, common = _synthetic_matrix(valid_model_subset=True)
    primary = analyze_full_synthetic_record_collections(
        trials=trials,
        common_records=common,
        integrity=_integrity(),
        bootstrap_repetitions=0,
    )
    independent = independently_recompute_full_synthetic(
        trials=trials,
        common_records=common,
        integrity=_integrity(),
        bootstrap_repetitions=0,
    )
    comparison = full_synthetic_analysis_verifier_comparison(primary, independent)
    assert comparison["section_difference_count"] == 0
    assert all(row["sample_count"] == 30 for row in primary["ridge_model_comparison"])
    assert len(primary["systematic_offset_summary"]) == 252


def test_common_builder_rejects_lock_sha_and_trial_input_checksum_mismatch(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen_assets"
    frozen.mkdir()
    snapshot_path = frozen / "snapshots.csv"
    trial_path = frozen / "trials.csv"
    snapshot = {
        "snapshot_id": "s",
        "scene_variant": SCENES[0],
        "geometry_seed_index": "0",
        "geometry_seed_value": str(GEOMETRY_SEEDS[0]),
        "measurement_seed_index": "0",
        "measurement_seed_value": str(MEASUREMENT_SEEDS[0]),
        "repeat_index": "0",
        "condition": "FULL_NOISE",
    }
    with snapshot_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(snapshot))
        writer.writeheader(); writer.writerow(snapshot)
    trial_rows = [
        {**snapshot, "backend": backend, "planned_trial_id": f"s/{backend}"}
        for backend in BACKENDS
    ]
    with trial_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0]))
        writer.writeheader(); writer.writerows(trial_rows)
    lock_entry = {"snapshot_id": "s"}
    lock = {
        "planned_snapshot_count": 1,
        "schema_version": "full_synthetic_development_snapshot_lock_v1",
        "snapshots": [lock_entry],
    }
    lock["snapshot_lock_payload_sha256"] = canonical_json_sha256(lock)
    lock_path = frozen / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    manifest = {
        "new_planned_snapshot_count": 1,
        "new_planned_snapshots_path": "frozen_assets/snapshots.csv",
        "new_planned_snapshots_sha256": file_sha256(snapshot_path),
        "new_planned_trial_count": 2,
        "new_planned_trials_path": "frozen_assets/trials.csv",
        "new_planned_trials_sha256": file_sha256(trial_path),
        "new_snapshot_cache_root": "cache",
        "new_snapshot_lock_path": "frozen_assets/lock.json",
        "new_snapshot_lock_sha256": file_sha256(lock_path),
    }
    bindings = analysis_module._load_new_snapshot_bindings(tmp_path, manifest)
    assert bindings["lock_sha256"] == file_sha256(lock_path)
    lock_path.write_text(lock_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="lock manifest SHA"):
        analysis_module._load_new_snapshot_bindings(tmp_path, manifest)

    checksums = {
        "source_checksum": "1" * 64,
        "target_checksum": "2" * 64,
        "reference_pose_checksum": "3" * 64,
        "snapshot_checksum": "4" * 64,
    }
    fake_bindings = {
        **bindings,
        "lock_sha256": "5" * 64,
        "lock_by_id": {"s": lock_entry},
        "snapshot_by_id": {"s": snapshot},
        "trial_by_id": {row["planned_trial_id"]: row for row in trial_rows},
        "trial_plans": trial_rows,
    }
    monkeypatch.setattr(analysis_module, "load_manifest", lambda *a, **k: (frozen / "manifest.json", {}))
    monkeypatch.setattr(analysis_module, "manifest_root", lambda _: tmp_path)
    monkeypatch.setattr(analysis_module, "_load_new_snapshot_bindings", lambda *a: fake_bindings)
    monkeypatch.setattr(analysis_module, "_common_cache_provenance", lambda **k: {})
    import phase_a_harness.full_synthetic_snapshot_builder as snapshot_module
    monkeypatch.setattr(snapshot_module, "read_full_synthetic_snapshot", lambda *a, **k: {**checksums, "source": np.zeros((12, 3)), "target": np.zeros((12, 3)), "reference": np.eye(4)})
    normalized = []
    for index, planned in enumerate(trial_rows):
        backend = independent_module._backend(planned["backend"])
        normalized.append(
            {
                **checksums,
                "source_checksum": "bad" if index == 1 else checksums["source_checksum"],
                "backend_schema_name": backend,
                "condition": "FULL_NOISE",
                "finite_output": True,
                "geometry_seed": GEOMETRY_SEEDS[0],
                "measurement_seed": MEASUREMENT_SEEDS[0],
                "planned_trial_id": planned["planned_trial_id"],
                "repeat_index": 0,
                "scene_variant": SCENES[0],
                "snapshot_id": "s",
                "snapshot_lock_sha256": "5" * 64,
                "solver_failure": False,
            }
        )
    with pytest.raises(ValueError, match="trial input differs"):
        analysis_module.build_common_association_records(
            manifest_path=frozen / "manifest.json",
            trials=normalized,
            output_path=tmp_path / "run" / "common.json",
        )

    raw_run = tmp_path / "sha_bound_run"
    (raw_run / "raw_results").mkdir(parents=True)
    plan = [{
        "planned_trial_id": "t", "snapshot_id": "s",
        "scene_variant": SCENES[0], "condition": "FULL_NOISE",
        "backend": BACKENDS[0],
    }]
    payload = {
        **plan[0], "implementation_sha256": "0" * 64,
        "protocol_sha256": "2" * 64,
    }
    result_path = raw_run / "raw_results/t.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    raw_manifest = {
        "run_id": "r",
        "results": {"t": {"path": "t.json", "planned_trial_id": "t", "sha256": file_sha256(result_path)}},
    }
    (raw_run / "raw_result_manifest.json").write_text(json.dumps(raw_manifest), encoding="utf-8")
    primary_audit = analysis_module._load_raw_set(
        plan_rows=plan, run_dir=raw_run, expected_run_id="r",
        expected_implementation_sha256="1" * 64,
        expected_protocol_sha256="2" * 64, validator=lambda value: value,
    )
    independent_rows, independent_defects = independent_module._read_raw(
        plan, raw_run, "r", lambda value: value,
        expected_implementation_sha256="1" * 64,
        expected_protocol_sha256="2" * 64,
    )
    assert not primary_audit.rows and primary_audit.corrupt_count == 1
    assert not independent_rows and independent_defects["corrupt"] == 1


def test_independent_common_recomputation_rejects_metric_tamper_even_when_cache_is_resigned(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen_assets"
    frozen.mkdir()
    manifest_path = frozen / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    snapshot_plan = {
        "snapshot_id": "s", "scene_variant": SCENES[0],
        "geometry_seed_index": "0", "geometry_seed_value": str(GEOMETRY_SEEDS[0]),
        "measurement_seed_index": "0", "measurement_seed_value": str(MEASUREMENT_SEEDS[0]),
        "repeat_index": "0", "condition": "FULL_NOISE",
    }
    trial_plans = [
        {**snapshot_plan, "backend": backend, "planned_trial_id": f"s/{backend}"}
        for backend in BACKENDS
    ]
    bindings = {
        "cache_root": tmp_path / "cache",
        "lock_by_id": {"s": {"snapshot_id": "s"}},
        "lock_sha256": "5" * 64,
        "snapshot_by_id": {"s": snapshot_plan},
        "trial_by_id": {row["planned_trial_id"]: row for row in trial_plans},
    }
    monkeypatch.setattr(independent_module, "load_manifest", lambda *a, **k: (manifest_path, {}))
    monkeypatch.setattr(independent_module, "manifest_root", lambda _: tmp_path)
    monkeypatch.setattr(independent_module, "_csv", lambda path: [snapshot_plan] if "snapshot" in path.name else trial_plans)
    monkeypatch.setattr(independent_module, "_independent_new_bindings", lambda *a: bindings)
    import phase_a_harness.full_synthetic_snapshot_builder as snapshot_module
    import phase_a_harness.common_association_analysis as common_module
    checksums = {"source_checksum": "1"*64, "target_checksum": "2"*64, "reference_pose_checksum": "3"*64, "snapshot_checksum": "4"*64}
    monkeypatch.setattr(snapshot_module, "read_full_synthetic_snapshot", lambda *a, **k: {**checksums, "source": np.zeros((12,3)), "target": np.zeros((12,3)), "reference": np.eye(4)})
    monkeypatch.setattr(common_module, "prepare_common_association_context", lambda *a, **k: object())
    monkeypatch.setattr(common_module, "safe_analyze_estimated_transform", lambda context, estimated, identifiers: {**identifiers, "snapshot_id": "s", "common_association_valid": True, "correspondence_turnover": 0.25})
    trials = []
    for planned in trial_plans:
        trials.append(
            {
                **checksums, "backend_schema_name": independent_module._backend(planned["backend"]),
                "condition": "FULL_NOISE", "final_transform_4x4": np.eye(4).tolist(),
                "finite_output": True, "geometry_seed": GEOMETRY_SEEDS[0],
                "measurement_seed": MEASUREMENT_SEEDS[0], "planned_trial_id": planned["planned_trial_id"],
                "repeat_index": 0, "scene_variant": SCENES[0], "snapshot_id": "s",
                "snapshot_lock_sha256": "5"*64, "solver_failure": False,
            }
        )
    cached = [
        {"backend_schema_name": row["backend_schema_name"], "condition": "FULL_NOISE", "geometry_seed": GEOMETRY_SEEDS[0], "measurement_seed": MEASUREMENT_SEEDS[0], "planned_trial_id": row["planned_trial_id"], "repeat_index": 0, "scene_variant": SCENES[0], "snapshot_id": "s", "common_association_valid": True, "correspondence_turnover": 0.50}
        for row in trials
    ]
    resigned_payload = {"records": cached}
    resigned_payload["common_association_payload_sha256"] = canonical_json_sha256(resigned_payload)
    assert resigned_payload["common_association_payload_sha256"]
    with pytest.raises(ValueError, match="differs from independent"):
        independently_recompute_common_association_records(
            manifest_path=manifest_path, trials=trials, cached_records=cached
        )


def test_publication_decision_preserves_source_decisions_and_fails_closed_on_difference(monkeypatch):
    base = {name: True for name in publisher_module.REQUIRED_DEVELOPMENT_GATES}
    base.update(
        {
            "CONFIRMATORY_PROTOCOL_DESIGN_AUTHORIZED": True,
            "CONFIRMATORY_RUN_AUTHORIZED": False,
            "FULL_SYNTHETIC_DEVELOPMENT_PASS": True,
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
            "PHENOMENON_CONFIRMED_BUT_INCREMENTAL_VALUE_NOT_ESTABLISHED": False,
            "REAL_DATA_PROTOCOL_DESIGN_AUTHORIZED": True,
            "REAL_DATA_RUN_AUTHORIZED": False,
            "SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED": True,
        }
    )
    primary = {"final_decision": dict(base)}
    independent = {"final_decision": dict(base)}
    monkeypatch.setattr(publisher_module, "full_synthetic_analysis_verifier_difference_count", lambda *a: 1)
    decision, difference = publisher_module.full_synthetic_publication_decision(primary, independent)
    assert difference == 1
    assert primary["final_decision"] == independent["final_decision"] == base
    assert decision["ANALYSIS_VERIFIER_AGREEMENT_PASS"] is False
    assert decision["FULL_SYNTHETIC_DEVELOPMENT_ENGINEERING_PASS"] is False
    assert decision["FULL_SYNTHETIC_DEVELOPMENT_PASS"] is False
    assert decision["PHENOMENON_CONFIRMED_BUT_INCREMENTAL_VALUE_NOT_ESTABLISHED"] is False


def test_artifact_verifier_recomputes_projection_rejects_extra_and_checksum_tamper(tmp_path):
    trials, common = _synthetic_matrix()
    primary = analyze_full_synthetic_record_collections(
        trials=trials, common_records=common, integrity=_integrity(), bootstrap_repetitions=0
    )
    projection = independently_recompute_full_synthetic(
        trials=trials, common_records=common, integrity=_integrity(), bootstrap_repetitions=0
    )
    independent = {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": 0,
        "final_decision": projection["final_decision"],
        "verification_projection": projection,
    }
    primary = {**primary, "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": 0}
    decision, difference = publisher_module.full_synthetic_publication_decision(primary, independent)
    assert difference == 0
    (tmp_path / "tables").mkdir(); (tmp_path / "figures").mkdir()
    for name in TABLES:
        relative = f"tables/{name}"
        count = FIXED_ROW_COUNTS.get(relative, 0)
        with (tmp_path / relative).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(["value"])
            writer.writerows([[index] for index in range(count)])
    for name in FIGURES:
        plt.imsave(tmp_path / "figures" / name, np.zeros((8, 9, 3)), vmin=0, vmax=1)
        assert plt.imread(tmp_path / "figures" / name).shape[:2] == (8, 9)
    (tmp_path / "full_synthetic_development_report.md").write_text(
        "# report\n"
        + "\n".join(f"figures/{name}" for name in FIGURES)
        + "\n"
        + "\n".join(f"tables/{name}" for name in TABLES)
        + "\n",
        encoding="utf-8",
    )
    for name, value in (
        ("primary_analysis.json", primary),
        ("independent_verification.json", independent),
        ("final_decision.json", decision),
        ("run_manifest.json", {"run_id": "full-synthetic-development-v1", "new_snapshot_count": 1050, "new_trial_count": 2100, "combined_snapshot_count": 1260, "combined_trial_count": 2520, "open3d_trial_count": 1260, "pcl_trial_count": 1260, "native_trial_count": 0, "raw_result_manifest_sha256": "a"*64, "scientific_protocol_sha256": "b"*64, "experiment_manifest_sha256": "c"*64, "snapshot_lock_sha256": "d"*64}),
    ):
        (tmp_path / name).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    checksum_targets = sorted(set(REQUIRED_FILES) - set(SHA256_EXCLUDED_FILES))
    (tmp_path / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(tmp_path / name)}  {name}\n" for name in checksum_targets),
        encoding="utf-8",
    )
    first = verify_full_synthetic_artifact(tmp_path, write_report=True)
    second = verify_full_synthetic_artifact(tmp_path, write_report=False)
    assert first["ARTIFACT_VERIFICATION_PASS"] is True
    assert second["ARTIFACT_VERIFICATION_PASS"] is True
    (tmp_path / "unexpected.txt").write_text("extra", encoding="utf-8")
    assert verify_full_synthetic_artifact(tmp_path, write_report=False)["extra_file_count"] == 1
    (tmp_path / "unexpected.txt").unlink()
    with (tmp_path / "primary_analysis.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    tampered = verify_full_synthetic_artifact(tmp_path, write_report=False)
    assert tampered["ARTIFACT_VERIFICATION_PASS"] is False
    assert tampered["sha256_mismatch_count"] == 1
