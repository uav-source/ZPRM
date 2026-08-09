from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.contracts import (
    OPEN3D_PLAN_BACKEND,
    PCL_PLAN_BACKEND,
    canonical_json_sha256,
    file_sha256,
)
from phase_a_harness import full_synthetic_development_protocol as protocol_module
from phase_a_harness import full_synthetic_development_runner as runner_module
from phase_a_harness import full_synthetic_snapshot_builder as snapshot_module
from phase_a_harness.full_synthetic_development_protocol import (
    ALL_CONDITIONS,
    CONDITION_PARAMETERS,
    NEW_CONDITIONS,
    phase_b_overlap_snapshots,
    planned_combined_snapshots,
    planned_new_snapshots,
    planned_trials,
    snapshot_id_for,
    validate_full_synthetic_plans,
    verify_phase_a_ideal_import,
    verify_full_synthetic_protocol_file,
    verify_junit_contract,
    verify_phase_b_pass_archive,
)
from phase_a_harness.full_synthetic_snapshot_builder import (
    build_full_synthetic_snapshot,
    load_full_synthetic_generator_protocol,
    read_full_synthetic_snapshot,
    verify_existing_snapshot_rebuild_equivalence,
    write_full_synthetic_snapshot_atomic,
)
from phase_a_harness.full_synthetic_trial_result import (
    validate_full_synthetic_trial_result_strict,
)
from phase_a_harness.phase_a_trial_result_schema import OPEN3D_BACKEND
from phase_a_harness.phase_a_trial_result_schema import canonical_json_bytes


def _source_only_package() -> bool:
    return "formal_runtime_artifacts_included=false" in (
        ROOT / "SOURCE_STATE.txt"
    ).read_text(encoding="utf-8")


def _skip_source_only_exclusion(reason: str) -> None:
    if _source_only_package():
        pytest.skip(reason)


def test_six_conditions_are_exact_and_only_five_are_new() -> None:
    assert ALL_CONDITIONS == (
        "IDEAL_MATCHED",
        "INDEPENDENT_NOISE_FREE",
        "SCAN_NOISE_ONLY",
        "MAP_NOISE_ONLY",
        "DROPOUT_ONLY",
        "FULL_NOISE",
    )
    assert NEW_CONDITIONS == ALL_CONDITIONS[1:]
    assert CONDITION_PARAMETERS["SCAN_NOISE_ONLY"]["scan_noise_sigma_m"] == 0.003
    assert CONDITION_PARAMETERS["MAP_NOISE_ONLY"]["map_noise_sigma_m"] == 0.001
    assert CONDITION_PARAMETERS["DROPOUT_ONLY"]["scan_dropout_fraction"] == 0.01
    assert CONDITION_PARAMETERS["FULL_NOISE"] == {
        "independent_sampling": True,
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.001,
        "scan_dropout_fraction": 0.01,
        "scan_noise_sigma_m": 0.003,
    }


def test_protocol_freezes_rank_systematic_candidate_and_statistical_contracts(
    tmp_path: Path,
) -> None:
    protocol = protocol_module.full_synthetic_protocol_payload()
    rank = protocol["gates"]["scene_rank_stability"]
    assert rank["condition_backend_combination_count"] == 10
    assert rank["geometry_rich_room_low_error_rank_max"] == 2
    assert rank["geometry_rich_room_required_combinations_min"] == 8
    assert rank["weak_scene_high_error_rank_max"] == 3
    assert rank["weak_scene_required_combinations_min"] == 8

    systematic = protocol["gates"]["systematic_offset_claim"]
    assert systematic["conditions"] == ["INDEPENDENT_NOISE_FREE", "FULL_NOISE"]
    assert systematic["group_observation_count"] == 10
    assert systematic["group_systematic_translation_offset_m_min"] == 0.005
    assert systematic["group_systematic_fraction_translation_min"] == 0.60
    assert systematic["qualified_geometry_groups_min"] == 2
    assert systematic["geometry_groups_per_backend_condition"] == 3
    assert systematic["median_systematic_fraction_translation_min"] == 0.70

    candidate = protocol["gates"]["automatic_nonequivalence_candidate"]
    assert candidate["normalized_hessian_eigenvalue_cosine_similarity_min"] == 0.98
    assert candidate["absolute_log_condition_ratio_max"] == 0.20
    assert candidate["absolute_log_initial_rmse_ratio_max"] == 0.20
    assert (
        candidate["correspondence_count_ratio_min"],
        candidate["correspondence_count_ratio_max"],
    ) == (0.90, 1.10)
    assert candidate["translation_error_ratio_min"] == 5.0
    assert candidate["turnover_absolute_difference_min"] == 0.15
    assert candidate["allowed_manual_review_status"] == "AUTOMATIC_CANDIDATE"
    assert protocol["gates"]["local_metric_incremental_value"][
        "alternative_scene_pair_coverage_min"
    ] == 3

    definitions = protocol["systematic_offset_and_repeatability"]
    assert definitions["translation_repeatability_covariance"] == "cov(rho_k, ddof=1)"
    assert definitions["systematic_fraction_zero_denominator_epsilon"] == 1e-12
    assert definitions["systematic_fraction_zero_denominator_value"] is None
    assert protocol["statistical_reporting"]["quantile_method"] == "linear"

    with pytest.raises(ValueError, match="missing or invalid"):
        verify_full_synthetic_protocol_file(tmp_path)
    assert verify_full_synthetic_protocol_file(ROOT)[
        "SCIENTIFIC_PROTOCOL_FROZEN_PASS"
    ] is True


def test_new_and_combined_plan_counts_are_exact() -> None:
    new = planned_new_snapshots()
    combined = planned_combined_snapshots()
    report = validate_full_synthetic_plans(
        new, planned_trials(new), combined, planned_trials(combined)
    )
    assert (len(new), len(planned_trials(new))) == (1050, 2100)
    assert (len(combined), len(planned_trials(combined))) == (1260, 2520)
    assert report["FULL_SYNTHETIC_PLAN_PASS"] is True
    assert report["native_trial_count"] == 0
    assert report["snapshot_backend_pairing_mismatch_count"] == 0


def test_each_condition_has_210_snapshots_and_each_snapshot_is_shared() -> None:
    combined = planned_combined_snapshots()
    trials = planned_trials(combined)
    assert Counter(row["condition"] for row in combined) == Counter(
        {condition: 210 for condition in ALL_CONDITIONS}
    )
    assert Counter(row["backend"] for row in trials) == Counter(
        {OPEN3D_PLAN_BACKEND: 1260, PCL_PLAN_BACKEND: 1260}
    )
    assert set(Counter(row["snapshot_id"] for row in trials).values()) == {2}
    assert not any("native" in row["backend"].lower() for row in trials)


def test_phase_b_overlap_retains_all_published_snapshot_and_trial_ids() -> None:
    overlap = phase_b_overlap_snapshots()
    published_snapshots = {
        row["snapshot_id"]
        for row in __import__("csv").DictReader(
            (ROOT / "frozen_assets/phase_b_planned_snapshots.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    }
    published_trials = {
        row["planned_trial_id"]
        for row in __import__("csv").DictReader(
            (ROOT / "frozen_assets/phase_b_planned_trials.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    }
    assert len(overlap) == 42
    assert {row["snapshot_id"] for row in overlap} == published_snapshots
    assert {row["planned_trial_id"] for row in planned_trials(overlap)} == published_trials
    assert snapshot_id_for(
        scene="LONG_CORRIDOR",
        geometry_seed_index=1,
        measurement_seed_index=0,
        repeat_index=0,
        condition="FULL_NOISE",
    ) == "phase-b-signal-v1/LONG_CORRIDOR/g1/FULL_NOISE"
    tag = subprocess.run(
        ["git", "rev-parse", f"{protocol_module.PHASE_B_PASS_TAG}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tag.returncode != 0:
        _skip_source_only_exclusion(
            "source-only ZIP excludes the historical Phase B tag object"
        )
    archive = verify_phase_b_pass_archive(ROOT)
    assert archive["PHASE_B_PASS_ARCHIVE_VALID"] is True
    assert archive["phase_b_pass_tag_commit"] == protocol_module.BASELINE_COMMIT
    environment = protocol_module.verify_frozen_runtime_environment(ROOT)
    assert environment["FROZEN_RUNTIME_ENVIRONMENT_PASS"] is True
    assert environment["open3d_version_hard_gate_pass"] is True
    assert environment["pcl_version_hard_gate_pass"] is True
    assert environment["observed_scikit_learn"] == "1.9.0"
    assert environment["scikit_learn_version_hard_gate_pass"] is True
    assert environment["matplotlib_version_hard_gate_pass"] is True
    assert environment["mamba_root_prefix_hard_gate_pass"] is True
    assert environment["development_runtime_versions"] == dict(
        protocol_module.DEVELOPMENT_RUNTIME_VERSIONS
    )
    assert environment["phase_a_environment_manifest_matplotlib_matches"] is False


def test_phase_a_ideal_import_is_read_only_and_complete() -> None:
    if not (ROOT / "results/formal_phase_a_v1/raw_result_manifest.json").is_file():
        _skip_source_only_exclusion(
            "source-only ZIP excludes the historical formal Phase A results"
        )
    report = verify_phase_a_ideal_import(ROOT, write_report=False)
    assert report["PHASE_A_IDEAL_IMPORT_PASS"] is True
    assert report["completed_snapshot_count"] == 210
    assert report["completed_trial_count"] == 420
    assert report["open3d_trial_count"] == report["pcl_trial_count"] == 210
    assert report["native_trial_count"] == 0
    assert report["artifact_sha256_mismatch_count"] == 0
    assert report["snapshot_plan_mismatch_count"] == 0
    assert report["trial_plan_mismatch_count"] == 0
    assert report["expected_trial_id_mismatch_count"] == 0
    assert report["snapshot_metadata_identity_mismatch_count"] == 0
    assert report["trial_identity_mismatch_count"] == 0
    assert report["reverse_raw_result_inventory_mismatch_count"] == 0


def test_generator_uses_both_development_measurement_seeds_without_confirmatory() -> None:
    development = load_full_synthetic_generator_protocol(ROOT)
    assert tuple(development.development_seeds["measurement"].values()) == (
        217775206,
        1664898153,
    )
    plan = next(
        row
        for row in planned_new_snapshots()
        if row["measurement_seed_index"] == "1"
        and row["condition"] == "SCAN_NOISE_ONLY"
    )
    value = build_full_synthetic_snapshot(ROOT, plan, protocol=development)
    audit = value["firewall_audit"]
    assert audit["CONFIRMATORY_SEED_INSTANTIATION_COUNT"] == 0
    assert audit["OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT"] == 0
    assert audit["GT_OPTIMIZATION_LEAKAGE_COUNT"] == 0


def test_snapshot_atomic_write_and_strict_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = next(
        row
        for row in planned_new_snapshots()
        if row["scene_variant"] == "GEOMETRY_RICH_ROOM"
        and row["condition"] == "INDEPENDENT_NOISE_FREE"
    )
    value = build_full_synthetic_snapshot(ROOT, plan)
    write_full_synthetic_snapshot_atomic(tmp_path, value)
    read = read_full_synthetic_snapshot(tmp_path, plan, arrays=True)
    assert read["source"].dtype == np.dtype("<f4")
    assert read["target"].flags.c_contiguous
    assert read["reference"].dtype == np.dtype("<f8")
    assert read["metadata"]["source_is_target_subset"] is False
    development = load_full_synthetic_generator_protocol(ROOT)
    resume = verify_existing_snapshot_rebuild_equivalence(
        ROOT, tmp_path, plan, protocol=development
    )
    assert resume["array_mismatch_count"] == 0
    original_build = snapshot_module.build_full_synthetic_snapshot

    def changed_rebuild(*args: object, **kwargs: object) -> dict[str, object]:
        rebuilt = original_build(*args, **kwargs)
        rebuilt["source"] = np.array(rebuilt["source"], copy=True)
        rebuilt["source"][0, 0] += np.float32(0.125)
        return rebuilt

    monkeypatch.setattr(
        snapshot_module, "build_full_synthetic_snapshot", changed_rebuild
    )
    with pytest.raises(ValueError, match="refusing to overwrite"):
        verify_existing_snapshot_rebuild_equivalence(
            ROOT, tmp_path, plan, protocol=development
        )
    source_path = read["directory"] / "source_points.npy"
    np.save(source_path, np.zeros((1, 3), dtype="<f4"), allow_pickle=False)
    with pytest.raises(ValueError, match="array file SHA"):
        read_full_synthetic_snapshot(tmp_path, plan, arrays=True)


def test_fsync_directory_opens_exactly_one_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = snapshot_module.os.open
    opened: list[Path] = []

    def recording_open(path: str | bytes | Path, flags: int) -> int:
        opened.append(Path(path))
        return real_open(path, flags)

    monkeypatch.setattr(snapshot_module.os, "open", recording_open)
    snapshot_module._fsync_directory(tmp_path)
    assert opened == [tmp_path]


def _open3d_payload(condition: str) -> dict[str, object]:
    sha = "0" * 64
    return {
        "schema_version": "phase_a_trial_result_v1",
        "planned_trial_id": f"trial/{condition}",
        "snapshot_id": f"snapshot/{condition}",
        "scene_variant": "GEOMETRY_RICH_ROOM",
        "condition": condition,
        "backend": OPEN3D_BACKEND,
        "protocol_sha256": sha,
        "snapshot_lock_sha256": sha,
        "snapshot_checksum": sha,
        "source_checksum": sha,
        "target_checksum": sha,
        "reference_pose_checksum": sha,
        "implementation_sha256": sha,
        "solver_failure": False,
        "failure_classification": "NONE",
        "failure_detail": None,
        "final_transform_4x4": np.eye(4).tolist(),
        "translation_update_m": 0.0,
        "rotation_update_rad": 0.0,
        "raw_rotation_finite": True,
        "raw_rotation_determinant": 1.0,
        "orthogonality_defect_fro": 0.0,
        "projection_correction_fro": 0.0,
        "finite_output": True,
        "runtime_ms": 0.0,
        "backend_diagnostics": {
            "correspondence_set_size": 1,
            "fitness": 1.0,
            "inlier_rmse": 0.0,
        },
    }


def test_trial_bridge_supports_exactly_five_new_conditions() -> None:
    for condition in NEW_CONDITIONS:
        assert validate_full_synthetic_trial_result_strict(_open3d_payload(condition))[
            "condition"
        ] == condition
    for forbidden in ("IDEAL_MATCHED", "CONFIRMATORY", "REAL_DATA"):
        with pytest.raises(ValueError, match="not authorized"):
            validate_full_synthetic_trial_result_strict(_open3d_payload(forbidden))


def test_dry_run_is_1050_2100_and_zero_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    new = planned_new_snapshots()
    new_trials = planned_trials(new)
    combined = planned_combined_snapshots()
    subset_ids = {
        row["planned_trial_id"] for row in planned_trials(phase_b_overlap_snapshots(new))
    }
    stack = {
        "cache_root": tmp_path / "cache",
        "combined_snapshots": combined,
        "combined_trials": planned_trials(combined),
        "lock_by_id": {row["snapshot_id"]: {} for row in new},
        "manifest": {
            "formal_execution_authorized": False,
            "formal_output_dir": "results/full_synthetic_development_v1",
            "formal_run_id": "full-synthetic-development-v1",
            "formal_workers": 2,
            "phase_b_reference_raw_manifest_path": "ignored.json",
        },
        "new_snapshots": new,
        "new_trials": new_trials,
        "root": tmp_path,
    }
    monkeypatch.setattr(runner_module, "load_full_synthetic_stack", lambda _: stack)
    monkeypatch.setattr(runner_module, "read_full_synthetic_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(runner_module, "source_runtime_import_paths", lambda: [])
    monkeypatch.setattr(
        runner_module,
        "_load_json",
        lambda *a, **k: {"results": {x: {} for x in subset_ids}},
    )
    report = runner_module.dry_run_full_synthetic(
        manifest_path=tmp_path / "frozen_assets/full_synthetic_development_manifest_v1.json",
        run_id="full-synthetic-development-v1",
        output_dir=tmp_path / "results/full_synthetic_development_v1",
        workers=2,
    )
    assert report["FULL_SYNTHETIC_DRY_RUN_PASS"] is True
    assert report["PHASE_B_SUBSET_REPRODUCTION_DESIGN_PASS"] is True
    assert (report["new_snapshot_count"], report["new_trial_count"]) == (1050, 2100)
    assert (report["combined_snapshot_count"], report["combined_trial_count"]) == (1260, 2520)
    assert report["backend_execution_count"] == report["formal_rng_access_count"] == 0
    assert report["trial_result_count"] == report["started_event_count"] == 0
    # The audit hook is process-global and not removable.  Its weak monitor
    # reference must become inert after dry_run_full_synthetic returns so a
    # shared pytest process can continue read-only source audits.
    # Exercise the process-global hook without issuing a real filesystem read
    # against the read-only source repository.  An inert weak monitor must let
    # the synthetic audit event return normally.
    sys.audit("open", "/home/lj/Degen-LIO/__inert_monitor_probe__", "r", 0)
    with pytest.raises(PermissionError, match="PYTHONNOUSERSITE"):
        protocol_module.assert_isolated_python_runtime(
            environ={}, search_paths=[str(ROOT / "src")]
        )
    with pytest.raises(PermissionError, match="MAMBA_ROOT_PREFIX"):
        protocol_module.assert_isolated_python_runtime(
            environ={"PYTHONNOUSERSITE": "1"}, search_paths=[str(ROOT / "src")]
        )
    with pytest.raises(PermissionError, match="source repository"):
        protocol_module.assert_isolated_python_runtime(
            environ={
                "MAMBA_ROOT_PREFIX": protocol_module.FROZEN_MAMBA_ROOT_PREFIX,
                "PYTHONNOUSERSITE": "1",
            },
            search_paths=["/home/lj/Degen-LIO"],
        )
    results_dir = tmp_path / "raw_results"
    results_dir.mkdir()
    (results_dir / "unlisted.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        runner_module._audit_raw_result_inventory(
            stack,
            results_dir,
            {"results": {}, "run_id": "x", "schema_version": "x"},
            {new_trials[0]["planned_trial_id"]: new_trials[0]},
        )
    monkeypatch.chdir("/home/lj/Degen-LIO")
    with pytest.raises(PermissionError, match="source repository"):
        protocol_module.assert_isolated_python_runtime(
            environ={
                "MAMBA_ROOT_PREFIX": protocol_module.FROZEN_MAMBA_ROOT_PREFIX,
                "PYTHONNOUSERSITE": "1",
            },
            search_paths=[""],
        )


def test_unauthorized_manifest_has_no_future_gate_report_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "frozen_assets").mkdir()
    (tmp_path / protocol_module.PROTOCOL_RELATIVE).write_text(
        json.dumps(
            protocol_module.full_synthetic_protocol_payload(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(protocol_module, "_sha", lambda *args: "0" * 64)
    payload = protocol_module.build_full_synthetic_manifest_payload(tmp_path)
    assert payload["formal_execution_authorized"] is False
    assert payload["phase_b_subset_execution_authorized"] is True
    assert "pre_run_gate_report_sha256" not in payload
    assert "package_init" in payload["code_bindings"]
    for dependency in (
        "execution_chain_audit",
        "open3d_backend",
        "pcl_backend",
        "rotation_metrics",
        "phase_a_trial_result_schema",
        "phase_a_trial_result_writer",
        "phase_a_trial_result_resume",
        "phase_a_attempt_events",
    ):
        assert dependency in payload["code_bindings"]
    assert payload["environment_manifest_path"] == "frozen_assets/environment_manifest.json"
    assert payload["formal_mamba_root_prefix"] == protocol_module.FROZEN_MAMBA_ROOT_PREFIX
    assert payload["development_runtime_versions"] == dict(
        protocol_module.DEVELOPMENT_RUNTIME_VERSIONS
    )
    assert payload["formal_execution_command"] == protocol_module.FORMAL_EXECUTION_COMMAND
    bound_generator_paths = {
        row["path"] for row in payload["frozen_generator_runtime_bindings"].values()
    }
    assert {
        "src/phase_a_harness/phase_b_generator_frozen/__init__.py",
        "src/phase_a_harness/phase_b_generator_frozen/capture_range/__init__.py",
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/__init__.py",
    } <= bound_generator_paths
    assert payload["test_execution_contract"]["existing"]["exact_test_count"] == 65
    assert payload["test_execution_contract"]["new"]["exact_test_count"] == 35
    assert payload["test_execution_contract"]["new"]["required_categories"] == list(
        protocol_module.REQUIRED_NEW_TEST_CATEGORIES
    )
    assert len(protocol_module.REQUIRED_NEW_TEST_CATEGORIES) == 30

    junit = tmp_path / "one_test.xml"
    junit.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.fake" name="test_only" />'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    one_sha = protocol_module._junit_collection_sha256(
        ["tests.fake::test_only"]
    )
    assert verify_junit_contract(
        junit, expected_count=65, expected_collection_sha256=one_sha
    )["pass"] is False


def test_authorization_rejects_forged_gate_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not (ROOT / "results/phase_b_signal_v1/raw_result_manifest.json").is_file():
        _skip_source_only_exclusion(
            "source-only ZIP excludes the historical Phase B raw results"
        )
    (tmp_path / "frozen_assets").mkdir()
    base = {
        "formal_execution_authorized": False,
        "manifest_version": "1",
        "phase_b_subset_execution_authorized": True,
    }
    monkeypatch.setattr(
        protocol_module, "build_full_synthetic_manifest_payload", lambda _: dict(base)
    )
    protocol_module.create_unauthorized_full_synthetic_manifest(tmp_path)
    gate = {name: True for name in protocol_module.REQUIRED_PRE_RUN_GATES}
    gate.update(
        {
            "ALL_PRE_RUN_GATES_PASS": True,
            "formal_execution_authorized_before_transition": False,
            "schema_version": "full_synthetic_development_pre_run_gate_v1",
        }
    )
    monkeypatch.setattr(
        protocol_module, "derive_full_synthetic_pre_run_gate_report", lambda _: gate
    )
    forged_path = tmp_path / protocol_module.PRE_RUN_GATE_REPORT_RELATIVE
    forged_path.parent.mkdir(parents=True)
    forged_path.write_text(json.dumps({"forged": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="not derived from raw evidence"):
        protocol_module.authorize_full_synthetic_manifest_once(tmp_path)

    evidence_root = tmp_path / "binding"
    evidence_root.mkdir()
    raw_evidence = evidence_root / "raw.json"
    raw_evidence.write_text('{"raw":true}\n', encoding="utf-8")
    gate_path = evidence_root / "gate.json"
    gates = {
        "evidence_sha256": {
            "raw.json": file_sha256(raw_evidence),
        }
    }
    gate_path.write_text(json.dumps(gates), encoding="utf-8")
    monkeypatch.setattr(
        protocol_module, "PRE_RUN_GATE_REPORT_RELATIVE", Path("gate.json")
    )
    bindings = protocol_module._authorized_evidence_bindings(evidence_root, gates)
    assert bindings["pre_run_evidence_sha256"] == gates["evidence_sha256"]
    assert bindings["pre_run_gate_report_sha256"] == file_sha256(gate_path)
    raw_evidence.write_text('{"raw":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="raw evidence SHA"):
        protocol_module._authorized_evidence_bindings(evidence_root, gates)

    strict_root = tmp_path / "strict"
    strict_assets = strict_root / "frozen_assets"
    strict_assets.mkdir(parents=True)
    authorized = {
        "formal_execution_authorized": True,
        "manifest_version": "1",
    }
    authorized["manifest_payload_sha256"] = canonical_json_sha256(authorized)
    strict_manifest = strict_assets / protocol_module.MANIFEST_RELATIVE.name
    strict_manifest.write_text(
        json.dumps(authorized, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        protocol_module,
        "expected_full_synthetic_manifest_payload",
        lambda *_args, **_kwargs: dict(authorized),
    )
    monkeypatch.setattr(
        protocol_module, "assert_isolated_python_runtime", lambda: {"pass": True}
    )
    monkeypatch.setattr(
        protocol_module,
        "verify_frozen_runtime_environment",
        lambda _root: {"pass": True},
    )
    operational_calls: list[Path] = []
    original_operational_git_gate = protocol_module.verify_formal_operational_git_gate
    monkeypatch.setattr(
        protocol_module,
        "verify_formal_operational_git_gate",
        lambda root: operational_calls.append(Path(root)) or {"pass": True},
    )
    loaded_path, loaded = protocol_module.load_strict_authorized_full_synthetic_manifest(
        strict_manifest
    )
    assert loaded_path == strict_manifest.resolve()
    assert loaded == authorized
    assert operational_calls == [strict_root]
    duplicate = strict_assets / "full_synthetic_development_manifest_copy.json"
    duplicate.write_text(strict_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple or ambiguous"):
        protocol_module.load_strict_authorized_full_synthetic_manifest(
            strict_manifest, require_operational_git_gate=False
        )
    duplicate.unlink()
    strict_manifest.write_text(
        '{"formal_execution_authorized":true,'
        '"formal_execution_authorized":true,"manifest_version":"1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        protocol_module.load_strict_authorized_full_synthetic_manifest(
            strict_manifest, require_operational_git_gate=False
        )
    monkeypatch.setattr(
        protocol_module,
        "verify_formal_operational_git_gate",
        original_operational_git_gate,
    )

    # Immutable 84-trial subset evidence must remain valid after the same raw
    # manifest grows with a legal Full Synthetic trial during formal resume.
    output = tmp_path / "subset_output"
    raw_results = output / "raw_results"
    raw_results.mkdir(parents=True)
    published = json.loads(
        (ROOT / "results/phase_b_signal_v1/raw_result_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    new_manifest = {
        "results": {},
        "run_id": "full-synthetic-development-v1",
        "schema_version": runner_module.RAW_MANIFEST_SCHEMA,
    }
    for trial_id, entry in published["results"].items():
        old = json.loads(
            (
                ROOT
                / "results/phase_b_signal_v1/raw_results"
                / entry["path"]
            ).read_text(encoding="utf-8")
        )
        old["implementation_sha256"] = "0" * 64
        old["protocol_sha256"] = "1" * 64
        old["snapshot_lock_sha256"] = "2" * 64
        payload = canonical_json_bytes(old)
        destination = raw_results / entry["path"]
        destination.write_bytes(payload)
        new_manifest["results"][trial_id] = {
            "path": entry["path"],
            "planned_trial_id": trial_id,
            "sha256": file_sha256(destination),
        }
    (output / "raw_result_manifest.json").write_bytes(
        canonical_json_bytes(new_manifest)
    )
    immutable = tmp_path / "immutable_subset.json"
    subset_report_path = tmp_path / "subset_report.json"
    monkeypatch.setattr(
        runner_module, "PHASE_B_SUBSET_EVIDENCE_RELATIVE", immutable
    )
    monkeypatch.setattr(
        runner_module, "PHASE_B_SUBSET_REPORT_RELATIVE", subset_report_path
    )
    initial = runner_module.verify_phase_b_trial_subset_reproduction(
        ROOT, output, create_immutable_evidence=True
    )
    assert initial["PHASE_B_SUBSET_REPRODUCTION_PASS"] is True
    initial_report_bytes = subset_report_path.read_bytes()
    legal_extra = next(
        row
        for row in planned_trials(planned_new_snapshots())
        if row["planned_trial_id"] not in new_manifest["results"]
    )
    new_manifest["results"][legal_extra["planned_trial_id"]] = {
        "path": "future-formal-result.json",
        "planned_trial_id": legal_extra["planned_trial_id"],
        "sha256": "f" * 64,
    }
    (output / "raw_result_manifest.json").write_bytes(
        canonical_json_bytes(new_manifest)
    )
    resumed = runner_module.verify_phase_b_trial_subset_reproduction(ROOT, output)
    assert resumed["PHASE_B_SUBSET_REPRODUCTION_PASS"] is True
    assert subset_report_path.read_bytes() == initial_report_bytes

    git_root = tmp_path / "operational_git"
    (git_root / "frozen_assets").mkdir(parents=True)
    (git_root / protocol_module.MANIFEST_RELATIVE).write_text(
        '{"formal_execution_authorized":true}\n', encoding="utf-8"
    )
    commands = (
        ("init", "-q"),
        ("checkout", "-q", "-b", protocol_module.DEVELOPMENT_BRANCH),
        ("config", "user.email", "audit@example.invalid"),
        ("config", "user.name", "Audit Test"),
        ("add", protocol_module.MANIFEST_RELATIVE.as_posix()),
        ("commit", "-q", "-m", "pre-run"),
        ("tag", protocol_module.PRE_RUN_TAG),
    )
    for command in commands:
        subprocess.run(["git", *command], cwd=git_root, check=True)
    operational = protocol_module.verify_formal_operational_git_gate(git_root)
    assert operational["FORMAL_OPERATIONAL_GIT_GATE_PASS"] is True
    (git_root / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(PermissionError, match="operational gate"):
        protocol_module.verify_formal_operational_git_gate(git_root)
