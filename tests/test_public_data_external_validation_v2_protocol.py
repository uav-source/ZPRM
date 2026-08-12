from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "protocols"

PROTOCOL_PATH = PROTOCOLS / "public_data_external_validation_protocol_v2.json"
ANALYSIS_PATH = (
    PROTOCOLS / "public_data_external_validation_analysis_contract_v2.json"
)
PAIR_PATH = (
    PROTOCOLS / "public_data_external_validation_pair_selection_contract_v2.json"
)
ELIGIBILITY_PATH = (
    PROTOCOLS / "public_data_external_validation_eligibility_v2.csv"
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {token}")
        ),
    )
    assert type(value) is dict
    expected = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert path.read_bytes() == expected
    return value


def _payload_sha256(value: dict[str, Any], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    payload = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_protocol_role_numbering_authorizations_and_counts_are_frozen() -> None:
    protocol = _json(PROTOCOL_PATH)
    assert protocol["protocol_payload_sha256"] == _payload_sha256(
        protocol, "protocol_payload_sha256"
    )
    assert protocol["role"] == "SUPPLEMENTARY_EXTERNAL_GENERALIZATION"
    assert protocol["primary_metrological_evidence"] is False
    assert protocol["public_dataset_count_required"] == 1
    assert protocol["selected_dataset"] == "Boreas"
    assert set(protocol["eligibility_requirements"]) == {
        f"E{index:02d}" for index in range(1, 15)
    }
    assert not any(
        identifier.startswith("R")
        for identifier in protocol["eligibility_requirements"]
    )
    assert protocol["current_stage1_counts"] == {
        "actual_trials": 0,
        "planned_trials": 0,
        "rich_snapshot_count": 0,
        "snapshot_count": 0,
        "weak_snapshot_count": 0,
    }
    assert protocol["registration_execution_count"] == 0
    assert protocol["real_trial_result_count"] == 0
    assert protocol["downloaded_lidar_object_count"] == 0
    assert protocol["downloaded_lidar_bytes"] == 0
    assert protocol["downloaded_lidar_payload_count"] == 0
    assert protocol["PUBLIC_DATA_V2_RUN_AUTHORIZED"] is False
    assert protocol["REAL_REGISTRATION_AUTHORIZED"] is False
    assert protocol["REAL_DATA_MAIN_EXPERIMENT_AUTHORIZED"] is False
    assert protocol["MEASUREMENT_PAPER_MAINLINE_AUTHORIZED"] is False


def test_e02_keeps_trajectory_independence_and_all_extrinsic_limitations() -> None:
    e02 = _json(PROTOCOL_PATH)["eligibility_requirements"]["E02"]
    assert e02["trajectory_backbone"]["independent_of"] == (
        "evaluated_query_to_target_scan_to_map_registration"
    )
    assert e02["trajectory_backbone"]["forbidden_generation_methods"] == [
        "ICP",
        "scan_matching",
        "LiDAR_odometry",
        "SLAM",
    ]
    assert e02["allowed_adjudications"] == [
        "PASS",
        "PASS_WITH_DOCUMENTED_LIMITATION",
        "FAIL",
    ]
    assert e02["boreas_adjudication_when_all_conditions_evidenced"] == (
        "PASS_WITH_DOCUMENTED_LIMITATION"
    )
    extrinsic = e02["historical_lidar_assisted_static_extrinsic"]
    assert extrinsic["allowed"] is True
    assert extrinsic["all_conditions_required"] is True
    conditions = extrinsic["conditions"]
    assert [condition["condition_id"] for condition in conditions] == [
        f"E02-X{index:02d}" for index in range(1, 11)
    ]
    rendered = " ".join(condition["requirement"] for condition in conditions)
    assert "not re-estimated in this experiment" in rendered
    assert "primary or reserve pair" in rendered
    assert "weak/rich labels" in rendered
    assert "Open3D or PCL results" in rendered
    assert "query-to-target registration output" in rendered
    assert "same frozen byte-identical matrix" in rendered
    assert "LiDAR-assisted calibration provenance" in rendered
    assert "remains UNKNOWN" in rendered
    assert "no absolute metrological conclusion" in rendered
    assert "external trend/generalization validation" in rendered
    semantics = e02["documented_limitation_semantics"]
    assert "does not assert" in semantics
    assert "non-LiDAR" in semantics
    assert "uncertainty-free" in semantics


def test_e06_e09_e10_and_e11_preserve_scientific_boundaries() -> None:
    requirements = _json(PROTOCOL_PATH)["eligibility_requirements"]
    assert requirements["E06"]["future_stage2_success_design"] == {
        "backend_trial_count": 200,
        "rich_interval_count": 10,
        "rich_snapshot_count": 50,
        "snapshots_per_interval": 5,
        "total_interval_count": 20,
        "total_snapshot_count": 100,
        "weak_interval_count": 10,
        "weak_snapshot_count": 50,
    }
    assert requirements["E06"]["stage1_counts"] == {
        "actual_trials": 0,
        "planned_trials": 0,
        "rich": 0,
        "snapshots": 0,
        "weak": 0,
    }

    backend = requirements["E09"]["backend_contract"]
    backend_path = ROOT / backend["path"]
    assert backend["file_sha256"] == _file_sha256(backend_path)
    assert backend["boreas_specific_icp_parameter_override_allowed"] is False

    e10 = requirements["E10"]
    assert e10["unknown_value"] == "UNKNOWN"
    assert e10["unknown_must_not_be_zero"] is True
    assert e10["uncertainty_components"] == [
        "trajectory_position_RMSE",
        "orientation_uncertainty",
        "time_synchronization_uncertainty",
        "extrinsic_uncertainty",
        "interpolation_uncertainty",
        "deskew_uncertainty",
        "map_accumulation_uncertainty",
    ]

    e11 = requirements["E11"]
    assert e11["main_measurement_gate_applies"] is False
    assert e11["measurement_mainline_dependency"] == "Mid-360 static real experiment"
    assert e11["permitted_primary_external_conclusions"] == [
        "geometry_trend",
        "cross_backend_ranking",
        "reassociation_association",
    ]


def test_analysis_contract_freezes_v2_h1_through_h6() -> None:
    contract = _json(ANALYSIS_PATH)
    assert contract["contract_payload_sha256"] == _payload_sha256(
        contract, "contract_payload_sha256"
    )
    assert set(contract["hypotheses"]) == {
        "V2-H1",
        "V2-H2",
        "V2-H3",
        "V2-H4",
        "V2-H5",
        "V2-H6",
    }
    h1 = contract["hypotheses"]["V2-H1"]
    assert h1["initial_pose"] == "T0 = T_reference"
    assert h1["estimated_pose_relation"] == "T_est != T_reference"
    assert h1["quantile_probabilities"] == [0.5, 0.75, 0.95]
    assert h1["absolute_millimetre_truth_assertion"] is False

    h2 = contract["hypotheses"]["V2-H2"]
    assert h2["directional_criterion"] == "median_weak > median_rich"
    assert h2["bootstrap"]["cluster_block"] == "frozen_scene_interval"
    assert h2["bootstrap"]["confidence_interval_level"] == 0.95
    assert h2["bootstrap"]["snapshot_level_naive_bootstrap_allowed"] is False

    h3 = contract["hypotheses"]["V2-H3"]
    assert h3["ranking_unit"] == "frozen_scene_interval"
    assert h3["spearman_rho_reference_threshold"] == 0.70

    h4 = contract["hypotheses"]["V2-H4"]
    assert h4["pooled_spearman_rho_reference_threshold"] == 0.40
    assert h4["scene_centered_spearman_rho_reference_threshold"] == 0.20
    assert h4["causal_claim_authorized"] is False
    assert h4["association_definition"]["file_sha256"] == _file_sha256(
        ROOT / h4["association_definition"]["path"]
    )

    h5 = contract["hypotheses"]["V2-H5"]
    assert h5["normalized_result_when_not_computable"] == "NOT_COMPUTABLE"
    assert h5["unknown_imputed_as_zero"] is False
    assert contract["hypotheses"]["V2-H6"][
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED"
    ] is False


def test_pair_contract_freezes_gap_aware_thresholds_ranking_and_reserves() -> None:
    contract = _json(PAIR_PATH)
    assert contract["contract_payload_sha256"] == _payload_sha256(
        contract, "contract_payload_sha256"
    )
    implementation = contract["gap_aware_overlap_implementation"]
    assert implementation["file_sha256"] == _file_sha256(ROOT / implementation["path"])
    assert implementation["non_gap_aware_overlap_module_allowed"] is False
    assert contract["candidate_reference_sequences"] == {
        "eligible_reference_sequence_count": 29,
        "excluded_for_native_gap_above_0_20_s": 2,
        "public_gt_sequence_count": 31,
        "required_exclusions": [
            "test_ground_truth_hidden",
            "calibration_missing",
            "nonfinite_reference_value",
            "duplicate_timestamp",
            "nonmonotonic_timestamp",
            "native_reference_gap_above_0_20_s",
        ],
    }
    assert contract["directed_pair_contract"] == {
        "direction": "map_sequence_i -> query_sequence_j",
        "expected_pair_count_for_29_sequences": 812,
        "same_sequence_pair_allowed": False,
    }
    assert contract["eligibility_thresholds"] == {
        "GT_OVERLAP_RADIUS_M": 5.0,
        "GT_RESAMPLE_RATE_HZ": 1.0,
        "MAX_NATIVE_REFERENCE_GAP_S": 0.20,
        "MIN_CONTIGUOUS_COVERED_DURATION_S": 5.0,
        "MIN_COVERAGE_FRACTION": 0.60,
        "MIN_ELIGIBLE_NONOVERLAPPING_5S_INTERVALS": 30,
        "MIN_TOTAL_COVERED_DURATION_S": 150.0,
    }
    assert [
        (item["field"], item["direction"])
        for item in contract["pair_selection"]["ranking"]
    ] == [
        ("covered_duration_s", "descending"),
        ("coverage_fraction", "descending"),
        ("eligible_nonoverlapping_5s_intervals", "descending"),
        ("nearest_distance_q95_m", "ascending"),
        ("map_sequence_id", "lexicographic_ascending"),
        ("query_sequence_id", "lexicographic_ascending"),
    ]
    reserve = contract["reserve_pair_activation"]
    assert reserve["allowed_reasons"] == [
        "download_object_permanently_missing",
        "checksum_mismatch",
        "file_corruption",
        "GT_file_corruption",
        "official_object_withdrawn_for_primary_pair",
    ]
    assert reserve["disallowed_reasons"] == [
        "ICP_error_too_large",
        "weak_rich_result_unfavorable",
        "Open3D_PCL_disagreement",
        "correlation_not_significant",
        "publication_result_unfavorable",
    ]


def test_eligibility_csv_has_only_e01_through_e14_and_is_not_adjudicated() -> None:
    with ELIGIBILITY_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["requirement_id"] for row in rows] == [
        f"E{index:02d}" for index in range(1, 15)
    ]
    assert all(row["mandatory"] == "True" for row in rows)
    assert all(row["status"] == "NOT_EVALUATED" for row in rows)
    assert rows[1]["allowed_adjudications"] == (
        "PASS|PASS_WITH_DOCUMENTED_LIMITATION|FAIL"
    )
