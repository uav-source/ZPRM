from __future__ import annotations

from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.protocol import (
    INITIAL_SCENE_IDS,
    STATION_IDS,
    FormalBatchError,
    bag_filename,
    parse_bag_filename,
    read_json_yaml,
    sha256_file,
    validate_scene_id,
    validate_station_id,
)


REPOSITORY = Path(__file__).resolve().parents[2]
EXPERIMENT = REPOSITORY / "experiments/mid360_formal_batch1"


def test_scene_id_schema_and_initial_candidates() -> None:
    assert INITIAL_SCENE_IDS == (
        "FMB1_R01", "FMB1_R02", "FMB1_R03",
        "FMB1_W01", "FMB1_W02", "FMB1_W03",
    )
    assert [validate_scene_id(value) for value in INITIAL_SCENE_IDS] == list(INITIAL_SCENE_IDS)
    for invalid in ("R01", "FMB1_R1", "FMB1_X01", "FMB2_R01", "FMB1_R00"):
        with pytest.raises(FormalBatchError):
            validate_scene_id(invalid)


def test_station_id_schema_is_exactly_three() -> None:
    assert STATION_IDS == ("S01", "S02", "S03")
    for invalid in ("S00", "S1", "S04", "ST01"):
        with pytest.raises(FormalBatchError):
            validate_station_id(invalid)


def test_bag_naming_is_deterministic_and_parseable() -> None:
    name = bag_filename("FMB1_R01", "S01", "MAP", "20260820_101530")
    assert name == "FMB1_R01_S01_MAP_20260820_101530.bag"
    assert parse_bag_filename(name) == {
        "scene_id": "FMB1_R01",
        "station_id": "S01",
        "role": "MAP",
        "timestamp": "20260820_101530",
    }
    with pytest.raises(FormalBatchError):
        parse_bag_filename("FMB1_R01_S01_final.bag")


def test_preregistration_freezes_batch_size_and_independence() -> None:
    prereg = read_json_yaml(EXPERIMENT / "preregistration.yaml")
    units = prereg["experimental_units"]
    assert units["independent_scene_count"] == 6
    assert units["map_query_pair_count"] == 18
    assert units["bag_count"] == 36
    assert units["snapshot_count"] == 180
    assert units["highest_independent_unit"] == "scene"
    assert "NOT_INDEPENDENT_SCENES" in units["station_and_snapshot_semantics"]


def test_preregistration_freezes_acquisition_timing() -> None:
    acquisition = read_json_yaml(EXPERIMENT / "preregistration.yaml")["acquisition"]
    assert acquisition["map_duration_target_s"] == 20.0
    assert acquisition["query_duration_target_s"] == 15.0
    assert acquisition["map_duration_min_s"] == 19.0
    assert acquisition["query_duration_min_s"] == 14.0
    assert acquisition["TARGET_MAP_QUERY_GAP_S"] == 12.0
    assert acquisition["minimum_actual_gap_s"] == 10.0
    assert acquisition["map_query_overlap_allowed"] is False


def test_scene_and_station_registries_have_exact_preacquisition_rows() -> None:
    scenes = read_json_yaml(EXPERIMENT / "scene_registry.yaml")["scenes"]
    stations = read_json_yaml(EXPERIMENT / "station_registry.yaml")["stations"]
    assert len(scenes) == 6
    assert {row["scene_id"] for row in scenes} == set(INITIAL_SCENE_IDS)
    assert len(stations) == 18
    for scene_id in INITIAL_SCENE_IDS:
        rows = [row for row in stations if row["scene_id"] == scene_id]
        assert {row["station_id"] for row in rows} == set(STATION_IDS)
        assert {row["planned_order"] for row in rows} == {1, 2, 3}
        assert all(row["station_spacing_m"] is None for row in rows)


def test_future_schedule_recovery_and_sign_are_frozen() -> None:
    analysis = read_json_yaml(EXPERIMENT / "preregistration.yaml")[
        "future_capture_basin_analysis"
    ]
    assert analysis["coarse_translation_perturbations_m"] == [
        0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60
    ]
    assert analysis["extension_translation_perturbations_m"] == [2.40, 3.20, 4.80, 6.40]
    assert analysis["MAX_TRANSLATION_PERTURBATION_M"] == 6.4
    assert analysis["recovery"] == {
        "final_translation_error_m_max": 0.005,
        "final_rotation_error_deg_max": 0.2,
        "logic": "AND",
    }
    assert analysis["directions"] == [
        "weak_positive", "weak_negative", "strong_positive", "strong_negative"
    ]


def test_batch_is_variance_estimation_not_final_measurement() -> None:
    prereg = read_json_yaml(EXPERIMENT / "preregistration.yaml")
    assert prereg["purpose"] == "FORMAL_ACQUISITION_BLINDED_VARIANCE_ESTIMATION_FEASIBILITY_BATCH"
    assert prereg["not_final_sample_size"] is True
    assert prereg["FORMAL_MEASUREMENT_RESULT"] is False
    assert prereg["FORMAL_ICP_UNLOCKED"] is False


def test_preacquisition_manifest_hashes_every_protocol_and_tool() -> None:
    manifest = read_json_yaml(
        REPOSITORY / "results/mid360_formal_batch1/preacquisition_manifest.json"
    )
    assert manifest["initial_data_state"]["bag_count"] == 0
    assert manifest["initial_data_state"]["formal_icp_executions"] == 0
    for relative, expected in manifest["files"].items():
        assert sha256_file(REPOSITORY / relative) == expected
