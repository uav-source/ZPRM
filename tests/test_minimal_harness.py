from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from phase_a_harness.asset_verifier import DIFFERENCE_FIELDS, source_runtime_import_paths, verify_frozen_assets
from phase_a_harness.backend_phase_a_metrics import transform_update
from phase_a_harness.contracts import file_sha256, load_manifest
from phase_a_harness.phase_a_trial_result_schema import (
    TrialResultValidationError,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)
from phase_a_harness.phase_a_trial_result_writer import write_phase_a_trial_result
from phase_a_harness.phase_a_trial_resume import CorruptExistingResult, validate_existing_trial_result_for_resume
from phase_a_harness.rotation_metrics import rotation_metric_audit
from phase_a_harness.snapshot_reader import read_snapshot


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "frozen_assets/frozen_experiment_manifest.json"


def _csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _fixture_rows():
    raw = json.loads((ROOT / "artifacts/fixture_qualification/raw_result_manifest.json").read_text())
    return [load_json_strict(ROOT / "artifacts/fixture_qualification/raw_results" / raw["results"][key]["path"]) for key in sorted(raw["results"])]


def _fixture(condition: str, backend: str):
    return next(row for row in _fixture_rows() if row["condition"] == condition and row["backend"] == backend)


def test_01_manifest_payload():
    load_manifest(MANIFEST_PATH)


def test_02_source_export_sha_identity():
    rows = _csv(ROOT / "frozen_assets/source_export_manifest.csv")
    assert all(row["source_sha256"] == row["destination_sha256"] for row in rows if row["copied_exactly"] == "true")


def test_03_asset_equivalence():
    assert verify_frozen_assets(MANIFEST_PATH, write_report=False)["SCIENTIFIC_ASSET_EXPORT_EQUIVALENCE_PASS"]


def test_04_snapshot_count():
    assert len(_csv(ROOT / "frozen_assets/planned_snapshots.csv")) == 210


def test_05_snapshot_lock_sha():
    assert file_sha256(ROOT / "frozen_assets/snapshot_lock.json") == "a63f8ea07e9420f72c08cf1c16812ee058f5e2bfa0575a5e3d39f6bda2f21f8a"


def test_06_planned_snapshot_ids_unique():
    rows = _csv(ROOT / "frozen_assets/planned_snapshots.csv")
    assert len({row["snapshot_id"] for row in rows}) == 210


def test_07_planned_trial_ids_unique():
    rows = _csv(ROOT / "frozen_assets/planned_trials.csv")
    assert len(rows) == len({row["planned_trial_id"] for row in rows}) == 420


def test_08_backend_counts():
    assert Counter(row["backend"] for row in _csv(ROOT / "frozen_assets/planned_trials.csv")) == Counter({"open3d_point_to_plane": 210, "pcl_iterative_closest_point_with_normals": 210})


def test_09_native_trial_count_zero():
    assert not any("native" in row["backend"] for row in _csv(ROOT / "frozen_assets/planned_trials.csv"))


def test_10_condition_ideal_matched_only():
    assert {row["condition"] for row in _csv(ROOT / "frozen_assets/planned_trials.csv")} == {"IDEAL_MATCHED"}


def _first_snapshot():
    identifier = _csv(ROOT / "frozen_assets/planned_snapshots.csv")[0]["snapshot_id"]
    return read_snapshot(ROOT / "data/frozen_snapshots", identifier, arrays=True)


def test_11_cache_dtype():
    item = _first_snapshot()
    assert item["source"].dtype == np.dtype("<f4") and item["reference"].dtype == np.dtype("<f8")


def test_12_cache_shape():
    item = _first_snapshot()
    assert item["source"].shape[1] == item["target"].shape[1] == 3 and item["reference"].shape == (4, 4)


def test_13_cache_c_order():
    item = _first_snapshot()
    assert all(item[name].flags.c_contiguous for name in ("source", "target", "reference", "parent_indices"))


def test_14_no_source_runtime_import():
    assert source_runtime_import_paths() == []


def test_15_no_scene_generator():
    # The frozen Phase A runner consumes exported snapshots and must never
    # regenerate scene geometry.  Later, explicitly versioned synthetic-study
    # modules live in the same package and legitimately own their builders, so
    # this repository-regression assertion is scoped to the Phase A execution
    # chain it was written to protect.
    phase_a_execution_modules = (
        "runner.py",
        "snapshot_reader.py",
        "analysis.py",
        "independent_verifier.py",
        "phase_a_stage1_analysis.py",
        "phase_a_stage1_independent_verifier.py",
    )
    assert not any(
        "build_scene_geometry" in (ROOT / "src/phase_a_harness" / name).read_text(
            encoding="utf-8"
        )
        for name in phase_a_execution_modules
    )


def test_16_no_formal_rng_access():
    assert json.loads((ROOT / "artifacts/dry_run_report.json").read_text())["formal_rng_access_count"] == 0


def test_17_identity_open3d():
    row = _fixture("FIXTURE_IDENTITY", "open3d_point_to_plane")
    assert not row["solver_failure"] and row["finite_output"]


def test_18_identity_pcl():
    row = _fixture("FIXTURE_IDENTITY", "pcl_point_to_plane")
    assert not row["solver_failure"] and row["finite_output"]


def test_19_nonidentity_open3d():
    assert _fixture("FIXTURE_NONIDENTITY_REFERENCE", "open3d_point_to_plane")["failure_classification"] == "NONE"


def test_20_nonidentity_pcl():
    assert _fixture("FIXTURE_NONIDENTITY_REFERENCE", "pcl_point_to_plane")["failure_classification"] == "NONE"


def test_21_no_correspondence_open3d():
    assert _fixture("FIXTURE_NO_CORRESPONDENCE", "open3d_point_to_plane")["solver_failure"]


def test_22_no_correspondence_pcl():
    assert _fixture("FIXTURE_NO_CORRESPONDENCE", "pcl_point_to_plane")["solver_failure"]


def test_23_fixture_pairing():
    assert json.loads((ROOT / "artifacts/fixture_qualification/fixture_qualification.json").read_text())["input_pairing_violation_count"] == 0


def test_24_strict_schema_valid():
    validate_phase_a_trial_result_strict(_fixture_rows()[0])


def test_25_strict_schema_missing_field():
    value = dict(_fixture_rows()[0]); value.pop("runtime_ms")
    with pytest.raises(TrialResultValidationError): validate_phase_a_trial_result_strict(value)


def test_26_strict_schema_unknown_field():
    value = dict(_fixture_rows()[0]); value["unknown"] = 1
    with pytest.raises(TrialResultValidationError): validate_phase_a_trial_result_strict(value)


def _resume_parts(tmp_path: Path):
    value = _fixture_rows()[0]
    path, digest = write_phase_a_trial_result(tmp_path, value)
    entry = {"path": path.name, "planned_trial_id": value["planned_trial_id"], "sha256": digest}
    expected = {key: value[key] for key in ("planned_trial_id", "snapshot_id", "backend", "scene_variant", "condition", "protocol_sha256", "snapshot_lock_sha256", "snapshot_checksum", "source_checksum", "target_checksum", "reference_pose_checksum", "implementation_sha256")}
    return value, path, entry, expected


def test_27_resume_wrong_checksum(tmp_path):
    _, path, entry, expected = _resume_parts(tmp_path); expected["source_checksum"] = "0" * 64
    with pytest.raises(CorruptExistingResult): validate_existing_trial_result_for_resume(path, manifest_entry=entry, expected=expected)


def test_28_resume_wrong_implementation(tmp_path):
    _, path, entry, expected = _resume_parts(tmp_path); expected["implementation_sha256"] = "0" * 64
    with pytest.raises(CorruptExistingResult): validate_existing_trial_result_for_resume(path, manifest_entry=entry, expected=expected)


def test_29_atomic_writer(tmp_path):
    _, path, _, _ = _resume_parts(tmp_path)
    assert path.is_file() and not list(tmp_path.glob("*.tmp"))


def test_30_resume_skips_valid_results():
    assert json.loads((ROOT / "artifacts/fixture_qualification/fixture_qualification.json").read_text())["resume_backend_execution_count"] == 0


def test_31_resume_rejects_corruption(tmp_path):
    _, path, entry, expected = _resume_parts(tmp_path)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CorruptExistingResult): validate_existing_trial_result_for_resume(path, manifest_entry=entry, expected=expected)


def test_32_fresh_resume_equivalence():
    assert json.loads((ROOT / "artifacts/fixture_qualification/fixture_qualification.json").read_text())["fresh_resume_scientific_equivalence"]


def test_33_translation_metric():
    estimate = np.eye(4); estimate[:3, 3] = [0.001, -0.002, 0.003]
    assert np.isclose(transform_update(np.eye(4), estimate)["translation_update_m"], np.linalg.norm([0.001, -0.002, 0.003]))


def test_34_rotation_metric():
    angle = 0.001; rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    assert np.isclose(rotation_metric_audit(rotation, np.eye(3))["rotation_error_rad"], angle)


def test_35_q95_linear():
    values = np.arange(20.0)
    assert np.quantile(values, 0.95, method="linear") == 18.05


def test_36_fixture_analysis_verifier():
    assert json.loads((ROOT / "artifacts/fixture_qualification/fixture_qualification.json").read_text())["analysis_verifier_difference_count"] == 0


def test_37_fixture_publisher():
    assert json.loads((ROOT / "artifacts/fixture_qualification/fixture_qualification.json").read_text())["publisher_pass"]


def test_38_fixture_artifact_verifier():
    assert json.loads((ROOT / "artifacts/fixture_qualification/fixture_qualification.json").read_text())["artifact_verifier_pass"]


def test_39_dry_run_inventory():
    value = json.loads((ROOT / "artifacts/dry_run_report.json").read_text())
    assert (value["planned_snapshot_count"], value["planned_trial_count"], value["open3d_trial_count"], value["pcl_trial_count"], value["native_trial_count"]) == (210, 420, 210, 210, 0)


def test_40_dry_run_zero_execution():
    value = json.loads((ROOT / "artifacts/dry_run_report.json").read_text())
    assert (value["backend_execution_count"], value["trial_result_count"], value["attempt_started_count"]) == (0, 0, 0)
