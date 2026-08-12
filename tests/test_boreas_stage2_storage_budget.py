from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from phase_a_harness.real_data_preparation import stage2_storage_planner as planner


REPOSITORY = Path(__file__).resolve().parents[1]
STAGE1_ROOT = (
    REPOSITORY / "frozen_assets/public_data_external_validation_v2_boreas_stage1"
)


@pytest.fixture(scope="module")
def inputs() -> planner.Stage1StorageInputs:
    return planner.load_frozen_stage1_inputs(STAGE1_ROOT)


def _mode(value: dict, name: str) -> dict:
    return next(row for row in value["modes"] if row["mode"] == name)


def test_frozen_stage1_pair_allowlist_and_bytes_are_authenticated(
    inputs: planner.Stage1StorageInputs,
) -> None:
    assert (inputs.map_sequence_id, inputs.query_sequence_id) == (
        "boreas-2021-11-14-09-47",
        "boreas-2021-01-26-11-22",
    )
    assert inputs.allowlist_object_count == planner.FROZEN_ALLOWLIST_OBJECT_COUNT == 20_061
    assert inputs.remote_payload_bytes == planner.FROZEN_REMOTE_PAYLOAD_BYTES == 104_158_637_472
    assert (inputs.map_object_count, inputs.map_remote_bytes) == (8_202, 41_998_817_280)
    assert (inputs.query_object_count, inputs.query_remote_bytes) == (
        11_859,
        62_159_820_192,
    )
    assert inputs.maximum_map_object_bytes == 5_573_256
    assert inputs.maximum_query_object_bytes == 5_591_160


def test_original_990_gib_budget_breakdown_closes_exactly(
    inputs: planner.Stage1StorageInputs,
) -> None:
    rows = planner.original_budget_breakdown(inputs)
    assert tuple(rows[0]) == planner.ORIGINAL_BREAKDOWN_FIELDS
    assert {row["component"] for row in rows} == {
        "raw lidar payload",
        "temporary download files",
        "decoded arrays",
        "map scan materialization",
        "accumulated target map",
        "voxel map copies",
        "query decoded points",
        "canonical source arrays",
        "canonical target arrays",
        "Open3D copies",
        "PCL copies",
        "per-snapshot target copies",
        "intermediate PCD/bin files",
        "geometry metrics",
        "publication artifacts",
        "resume/checkpoint files",
        "safety factor",
        "filesystem overhead",
    }
    safety = next(row for row in rows if row["component"] == "safety factor")
    assert safety["peak_contribution_bytes"] == 354_474_729_696
    assert sum(row["peak_contribution_bytes"] for row in rows) == 1_063_424_189_088
    assert sum(
        row["peak_contribution_bytes"]
        for row in rows
        if row["component"] != "safety factor"
    ) == 708_949_459_392
    unspecified = [
        row for row in rows if row["lifetime"] == "NOT_SEPARATELY_QUANTIFIED"
    ]
    assert unspecified
    assert all(row["raw_bytes"] == 0 for row in unspecified)


def test_object_lifecycle_has_all_required_classes_and_cleanup_rules() -> None:
    rows = planner.object_lifecycle_contract()
    assert all(tuple(row) == planner.LIFECYCLE_FIELDS for row in rows)
    assert {row["lifecycle_class"] for row in rows} == {
        "REMOTE_ONLY",
        "TEMPORARY_DOWNLOAD",
        "TEMPORARY_DECODED",
        "PERSISTENT_RAW",
        "PERSISTENT_CANONICAL",
        "PERSISTENT_MAP",
        "PERSISTENT_RESULT",
        "CHECKPOINT",
    }
    temporary = [row for row in rows if row["lifecycle_class"].startswith("TEMPORARY_")]
    assert temporary
    assert all(row["persistent"] is False for row in temporary)
    assert all("delete" in row["deletion_or_retention_rule"] for row in temporary)
    map_raw = next(row for row in rows if row["object_class"] == "map raw temp")
    assert "state transition" in map_raw["deletion_or_retention_rule"]
    pcl = next(row for row in rows if row["object_class"] == "PCL source/target PCD")
    assert "canonical NPY" in pcl["creation_or_source"]
    assert pcl["persistent"] is False


def test_storage_architecture_is_single_target_and_backend_shared(
    inputs: planner.Stage1StorageInputs,
) -> None:
    contract = planner.storage_architecture_contract(inputs)
    assert contract["unique_target_map_count"] == 1
    assert contract["snapshot_target_reference_count"] == 100
    assert contract["physical_target_map_copy_count"] == 1
    assert contract["per_snapshot_target_copy_allowed"] is False
    assert contract["snapshot_target_binding"] == "target_map_sha256"
    assert contract["canonical_source_physical_copy_count_per_snapshot"] == 1
    assert contract["canonical_target_physical_copy_count"] == 1
    assert contract[
        "future_open3d_source_sha256_equals_future_pcl_source_sha256"
    ] is True
    assert contract[
        "future_open3d_target_sha256_equals_future_pcl_target_sha256"
    ] is True
    assert contract["PCL_temporary_input_retained"] is False
    assert contract["execution_modes"]["STREAMING_LOW_DISK"][
        "two_pass_query_selection"
    ] is True
    assert set(contract["mode_scientific_equivalence_required"]) == {
        "target_map_sha256",
        "geometry_metrics_sha256",
        "selection_manifest_sha256",
        "selected_canonical_source_sha256",
        "backend_results",
    }


def test_theoretical_minimum_budget_is_exact_and_lifecycle_based(
    inputs: planner.Stage1StorageInputs,
) -> None:
    budget = planner.optimized_storage_budgets(inputs, current_free_bytes=118_785_650_688)
    row = _mode(budget, "THEORETICAL_MINIMUM")
    assert row["persistent_bytes"] == 42_637_784_151
    assert row["temporary_peak_bytes"] == 21_003_260_988
    assert row["simultaneously_live_peak_bytes"] == 63_641_045_139
    assert row["safety_margin_bytes"] == 6_364_104_514
    assert row["recommended_free_disk_bytes"] == 70_005_149_653
    assert row["recommended_free_disk_GiB"] == pytest.approx(65.19737621117383)
    assert row["CURRENT_DISK_SUFFICIENT"] is True


def test_recommended_operational_budget_is_exact_and_current_disk_is_sufficient(
    inputs: planner.Stage1StorageInputs,
) -> None:
    current = 118_785_650_688
    budget = planner.optimized_storage_budgets(inputs, current_free_bytes=current)
    row = _mode(budget, "RECOMMENDED_OPERATIONAL")
    assert row["persistent_bytes"] == 42_637_784_151
    assert row["temporary_peak_bytes"] == 41_998_817_408
    assert row["simultaneously_live_peak_bytes"] == 84_636_601_559
    assert row["safety_margin_bytes"] == 16_927_320_312
    assert row["recommended_free_disk_bytes"] == 101_563_921_871
    assert row["recommended_free_disk_GiB"] == pytest.approx(94.58877320494503)
    assert budget["CURRENT_DISK_SUFFICIENT"] is True
    assert budget["minimum_additional_GiB_required"] == 0
    assert budget["recommended_additional_GiB_required"] == 0
    assert budget["minimum_free_disk_required_before_start_bytes"] == 101_563_921_871
    assert budget["abort_if_free_disk_below_bytes"] == 16_927_320_312
    assert budget["runtime_low_disk_watermark_bytes"] == 16_927_320_312


def test_full_raw_cache_budget_adds_exactly_one_raw_payload_copy(
    inputs: planner.Stage1StorageInputs,
) -> None:
    budget = planner.optimized_storage_budgets(inputs, current_free_bytes=118_785_650_688)
    streaming = _mode(budget, "RECOMMENDED_OPERATIONAL")
    full = _mode(budget, "CONSERVATIVE_FULL_CACHE")
    assert full["persistent_bytes"] - streaming["persistent_bytes"] == 104_158_637_472
    assert full["persistent_bytes"] == 146_796_421_623
    assert full["temporary_peak_bytes"] == streaming["temporary_peak_bytes"]
    assert full["simultaneously_live_peak_bytes"] == 188_795_239_031
    assert full["safety_margin_bytes"] == 37_759_047_807
    assert full["recommended_free_disk_bytes"] == 226_554_286_838
    assert full["recommended_free_disk_GiB"] == pytest.approx(210.99512170813978)
    assert full["CURRENT_DISK_SUFFICIENT"] is False
    assert full["additional_GiB_required"] == pytest.approx(100.36736368201673)


def test_peak_accounting_uses_maximum_temporary_phase_not_naive_sum(
    inputs: planner.Stage1StorageInputs,
) -> None:
    budget = planner.optimized_storage_budgets(inputs, current_free_bytes=10**12)
    estimates = budget["size_estimates"]
    recommended = _mode(budget, "RECOMMENDED_OPERATIONAL")
    phases = (
        estimates["map_checkpoint_upper_bound_bytes"],
        estimates["pcl_runtime_conversion_upper_bound_bytes"],
        estimates["single_scan_streaming_upper_bound_bytes"],
    )
    assert recommended["temporary_peak_bytes"] == max(phases)
    assert recommended["temporary_peak_bytes"] < sum(phases)
    assert recommended["simultaneously_live_peak_bytes"] == (
        recommended["persistent_bytes"] + recommended["temporary_peak_bytes"]
    )


def test_low_disk_gate_fails_closed_and_reports_additional_space(
    inputs: planner.Stage1StorageInputs,
) -> None:
    current = 80 * planner.GIB
    budget = planner.optimized_storage_budgets(inputs, current_free_bytes=current)
    minimum = _mode(budget, "THEORETICAL_MINIMUM")
    recommended = _mode(budget, "RECOMMENDED_OPERATIONAL")
    assert minimum["CURRENT_DISK_SUFFICIENT"] is True
    assert recommended["CURRENT_DISK_SUFFICIENT"] is False
    assert budget["recommended_additional_GiB_required"] == pytest.approx(
        (101_563_921_871 - current) / planner.GIB
    )
    with pytest.raises(planner.InsufficientStage2DiskError, match="requires"):
        planner.assert_free_disk(current, budget)
    planner.assert_free_disk(101_563_921_871, budget)


def test_runtime_watermark_checks_projected_remaining_space(
    inputs: planner.Stage1StorageInputs,
) -> None:
    budget = planner.optimized_storage_budgets(inputs, current_free_bytes=10**12)
    watermark = 16_927_320_312
    planner.assert_runtime_watermark(watermark + 5_591_160, 5_591_160, budget)
    with pytest.raises(
        planner.InsufficientStage2DiskError,
        match="below runtime watermark",
    ):
        planner.assert_runtime_watermark(watermark + 5_591_159, 5_591_160, budget)
    with pytest.raises(ValueError, match="nonnegative"):
        planner.assert_runtime_watermark(watermark, -1, budget)


@pytest.mark.parametrize(
    "field",
    (
        "voxel_size",
        "range_limits",
        "normal_neighborhood",
        "association_distance",
        "geometry_metric",
        "weak_rich_thresholds",
        "snapshot_count",
        "primary_pair",
        "icp_parameters",
    ),
)
def test_storage_planner_rejects_scientific_parameter_configuration(field: str) -> None:
    with pytest.raises(
        planner.ScientificParameterModificationForbidden,
        match="cannot configure scientific fields",
    ):
        planner.assert_scientific_locks({field: 1})


def test_storage_planner_accepts_only_operational_configuration() -> None:
    planner.assert_scientific_locks(
        {
            "execution_mode": "STREAMING_LOW_DISK",
            "current_free_bytes": 123,
            "temporary_directory": "/stage2/tmp_download",
            "persistent_directory": "/stage2/target_map",
        }
    )
    with pytest.raises(planner.ScientificParameterModificationForbidden):
        planner.assert_scientific_locks({"unreviewed_option": True})
    with pytest.raises(planner.ScientificParameterModificationForbidden):
        planner.assert_scientific_locks(
            {"execution_mode": {"voxel_size": 9.0}}
        )
    with pytest.raises(planner.Stage2StoragePlanError, match="invalid execution_mode"):
        planner.assert_scientific_locks({"execution_mode": "FAST_BUT_DIFFERENT"})


def test_scientific_lock_contract_preserves_stage1_and_future_design(
    inputs: planner.Stage1StorageInputs,
) -> None:
    contract = planner.scientific_lock_contract(inputs)
    assert contract["scientific_fields_planner_configurable"] is False
    assert contract["voxel_size"] == planner.PREPROCESSING_PARAMETER_STATUS
    assert contract["future_success_design"] == {
        "weak_interval_count": 10,
        "rich_interval_count": 10,
        "snapshots_per_interval": 5,
        "weak_snapshot_count": 50,
        "rich_snapshot_count": 50,
        "snapshot_count": 100,
        "backend_trial_count": 200,
    }
    assert contract["allowlist_object_count"] == 20_061
    assert contract["remote_payload_bytes"] == 104_158_637_472


def test_budget_rejects_mutated_stage1_arithmetic(
    inputs: planner.Stage1StorageInputs,
) -> None:
    tampered = replace(inputs, old_subtotal_bytes=inputs.old_subtotal_bytes + 1)
    with pytest.raises(planner.Stage2StoragePlanError, match="does not close"):
        planner.original_budget_breakdown(tampered)


def test_planner_never_authorizes_download_or_registration(
    inputs: planner.Stage1StorageInputs,
) -> None:
    budget = planner.optimized_storage_budgets(inputs, current_free_bytes=10**12)
    assert budget["STAGE2_DOWNLOAD_AUTHORIZED"] is False
    assert budget["PUBLIC_DATA_V2_RUN_AUTHORIZED"] is False
    assert budget["REAL_REGISTRATION_AUTHORIZED"] is False
    assert budget["MEASUREMENT_PAPER_MAINLINE_AUTHORIZED"] is False
    assert budget["downloaded_lidar_payload_count"] == 0
    assert budget["downloaded_lidar_bytes"] == 0
    assert budget["registration_execution_count"] == 0
