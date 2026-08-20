from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

import pytest

import tools.mid360_formal_batch1.verify_pre_registration as verifier_cli

from experiments.mid360_formal_batch1.preregistration_verify import (
    EXPECTED_STATION_PREFIX,
    GEOMETRY_ONLY_FIELDS,
    MANIFEST_SCHEMA,
    validate_manifest_payload,
)
from experiments.mid360_formal_batch1.protocol import (
    BACKEND_CONTRACT_SHA256,
    INITIAL_SCENE_IDS,
    QUERY_QUANTILES,
    REQUIRED_POINT_FIELDS,
    REQUIRED_TOPICS,
    STATION_IDS,
    FormalBatchError,
)


REAL_REPOSITORY = Path(__file__).resolve().parents[2]


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _artifact(path: Path, content: bytes, materialize: bool) -> tuple[str, int]:
    if materialize:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return _sha(content), len(content)


def _repository(root: Path) -> Path:
    contract = root / "frozen_assets/backend_parameter_contract.json"
    contract.parent.mkdir(parents=True)
    shutil.copy2(
        REAL_REPOSITORY / "frozen_assets/backend_parameter_contract.json", contract
    )
    return root


def valid_payload(repository: Path, *, materialize: bool = False) -> dict:
    scenes = []
    stations = []
    bags = []
    audits = []
    targets = []
    snapshots = []
    geometry_metrics = []
    geometry_scenes = []
    canonical_inputs = []

    station_ordinal = 0
    for scene_id in INITIAL_SCENE_IDS:
        classification = "RICH" if scene_id.startswith("FMB1_R") else "WEAK"
        scenes.append(
            {
                "scene_id": scene_id,
                "semantic_candidate_label": f"{classification}_CANDIDATE",
                "final_geometry_class": classification,
            }
        )
        scene_metrics = (
            {
                "lambda_min_trans": 0.20,
                "lambda_mid_trans": 0.30,
                "lambda_max_trans": 0.50,
                "normalized_lambda_min_trans": 0.20,
                "normalized_lambda_mid_trans": 0.30,
                "normalized_lambda_max_trans": 0.50,
                "condition_number_trans": 2.5,
                "spectral_entropy_trans": 0.95,
            }
            if classification == "RICH"
            else {
                "lambda_min_trans": 0.10,
                "lambda_mid_trans": 0.10,
                "lambda_max_trans": 0.80,
                "normalized_lambda_min_trans": 0.10,
                "normalized_lambda_mid_trans": 0.10,
                "normalized_lambda_max_trans": 0.80,
                "condition_number_trans": 8.0,
                "spectral_entropy_trans": 0.70,
            }
        )
        geometry_scenes.append(
            {
                "scene_id": scene_id,
                "final_geometry_class": classification,
                "admitted": True,
                "geometry_lambda_min_median": scene_metrics[
                    "normalized_lambda_min_trans"
                ],
                "geometry_condition_median": scene_metrics["condition_number_trans"],
                "geometry_entropy_median": scene_metrics["spectral_entropy_trans"],
            }
        )

        for station_id in STATION_IDS:
            key = (scene_id, station_id)
            prefix = EXPECTED_STATION_PREFIX[key]
            station_ordinal += 1
            stations.append({"scene_id": scene_id, "station_id": station_id})
            role_shas = {}
            for role, part in (("MAP", "part1_20s"), ("QUERY", "part2_15s")):
                raw_name = f"mid360_{prefix}_{part}.bag"
                raw_path = repository / "bags" / raw_name
                raw_sha, raw_size = _artifact(
                    raw_path, f"raw:{raw_name}".encode("ascii"), materialize
                )
                role_shas[role] = raw_sha
                bags.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "role": role,
                        "raw_filename": raw_name,
                        "raw_absolute_path": str(raw_path),
                        "canonical_filename": (
                            f"{scene_id}_{station_id}_{role}_{prefix}.bag"
                        ),
                        "capture_prefix": prefix,
                        "sha256": raw_sha,
                        "bytes": raw_size,
                        "mtime": 0,
                    }
                )

            role_audit = {
                "topic_types": dict(REQUIRED_TOPICS),
                "lidar_rate_hz": 10.0,
                "imu_rate_hz": 200.0,
                "lidar_count": 200,
                "imu_count": 4000,
                "frame_ids": ["livox_frame"],
                "pointcloud2_fields": sorted(REQUIRED_POINT_FIELDS),
                "motion_status": "NO_OBVIOUS_MOTION",
            }
            audits.append(
                {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "acquisition_status": "ACQUISITION_PASS",
                    "map": {**role_audit, "duration_s": 20.0},
                    "query": {**role_audit, "duration_s": 15.0},
                    "pair": {"gap_s": 12.0, "no_overlap": True},
                }
            )

            target_path = (
                repository
                / "zero_perturbation_runtime/mid360_formal_batch1_ingest_v1/targets"
                / scene_id
                / station_id
                / "target_points.npy"
            )
            target_sha, target_size = _artifact(
                target_path,
                f"target:{scene_id}:{station_id}".encode("ascii"),
                materialize,
            )
            targets.append(
                {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "map_bag_sha256": role_shas["MAP"],
                    "target_path": str(target_path),
                    "target_npy_sha256": target_sha,
                    "target_size_bytes": target_size,
                    "target_point_count": 1000,
                    "map_frame_count": 200,
                    "raw_point_count": 20000,
                    "filtered_point_count": 15000,
                    "input_roles": ["MAP"],
                    "construction": "DIRECT_SAME_SENSOR_FRAME_MERGE_NO_REGISTRATION",
                    "registration_called": False,
                    "odometry_called": False,
                    "scan_matching_called": False,
                    "query_contribution_to_target": 0,
                }
            )

            for selection_index, quantile in enumerate(QUERY_QUANTILES):
                snapshot_id = f"{scene_id}_{station_id}_Q{selection_index + 1:02d}"
                timestamp = float(station_ordinal * 1000 + selection_index)
                source_path = (
                    repository
                    / "zero_perturbation_runtime/mid360_formal_batch1_ingest_v1/snapshots"
                    / scene_id
                    / station_id
                    / f"{snapshot_id}.npy"
                )
                source_sha, source_size = _artifact(
                    source_path, f"source:{snapshot_id}".encode("ascii"), materialize
                )
                snapshots.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "snapshot_id": snapshot_id,
                        "selection_index": selection_index,
                        "quantile": quantile,
                        "query_timestamp": timestamp,
                        "query_frame_index": selection_index,
                        "query_bag_sha256": role_shas["QUERY"],
                        "source_path": str(source_path),
                        "source_npy_sha256": source_sha,
                        "source_size_bytes": source_size,
                        "source_point_count": 100,
                        "target_npy_sha256": target_sha,
                        "selection_method": (
                            "FIXED_QUANTILE_SEQUENCE_RANK_NEAREST_UNUSED_EARLIER_TIE"
                        ),
                        "selection_frozen_before_registration": True,
                    }
                )
                metric = {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "snapshot_id": snapshot_id,
                    "selection_index": selection_index,
                    "query_timestamp": timestamp,
                    "initial_correspondence_count": 900,
                    "initial_valid_normal_correspondence_count": 850,
                    **scene_metrics,
                }
                assert set(GEOMETRY_ONLY_FIELDS).issubset(metric)
                geometry_metrics.append(metric)
                canonical_inputs.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "snapshot_id": snapshot_id,
                        "query_timestamp": timestamp,
                        "source_npy_sha256": source_sha,
                        "target_npy_sha256": target_sha,
                        "T0": "IDENTITY_4X4",
                    }
                )

    return {
        "schema": MANIFEST_SCHEMA,
        "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
        "scenes": scenes,
        "stations": stations,
        "bags": bags,
        "station_acquisition_audits": audits,
        "targets": targets,
        "snapshots": snapshots,
        "geometry_metrics": geometry_metrics,
        "geometry_scenes": geometry_scenes,
        "canonical_inputs": canonical_inputs,
        "no_icp_attestation": {
            "status": "PASS",
            "open3d_registration_call_count": 0,
            "pcl_cli_invocation_count": 0,
            "other_registration_process_count": 0,
            "formal_trial_count": 0,
        },
        "readiness": {
            "FMB1_PRE_REGISTRATION_DATA_READY": True,
            "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION": True,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "FORMAL_MEASUREMENT_RESULT": False,
            "scene_count": 6,
            "rich_scene_count": 3,
            "weak_scene_count": 3,
            "station_count": 18,
            "snapshot_count": 180,
            "rich_snapshot_count": 90,
            "weak_snapshot_count": 90,
            "planned_open3d_trials": 180,
            "planned_pcl_trials": 180,
            "planned_total_trials": 360,
            "actual_trials": 0,
            "actual_registration_trials": 0,
            "registration_execution_count": 0,
        },
    }


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    return _repository(tmp_path)


def test_valid_complete_manifest_passes_without_opening_data_files(repository: Path) -> None:
    result = validate_manifest_payload(valid_payload(repository), repository)
    assert result["status"] == "PASS"
    assert result["counts"] == {
        "scenes": 6,
        "stations": 18,
        "bags": 36,
        "targets": 18,
        "snapshots": 180,
        "geometry_rows": 180,
        "canonical_inputs": 180,
    }
    assert result["checks"]["file_authentication"] is False


def test_complete_frozen_manifest_aliases_and_flat_station_rows_are_supported(
    repository: Path,
) -> None:
    canonical = valid_payload(repository)
    details = []
    for bag in canonical["bags"]:
        role = bag["role"].lower()
        station = next(
            row
            for row in canonical["station_acquisition_audits"]
            if row["scene_id"] == bag["scene_id"]
            and row["station_id"] == bag["station_id"]
        )
        audit = station[role]
        details.append(
            {
                "scene_id": bag["scene_id"],
                "station_id": bag["station_id"],
                "role": bag["role"],
                "sha256": bag["sha256"],
                "inventory": {
                    "bag_sha256": bag["sha256"],
                    "topics": [
                        {
                            "topic": topic,
                            "message_type": message_type,
                        }
                        for topic, message_type in REQUIRED_TOPICS.items()
                    ],
                },
                "bag_audit": {
                    "duration_s": audit["duration_s"],
                    "frequencies_hz": {
                        "/livox/lidar": audit["lidar_rate_hz"],
                        "/livox/imu": audit["imu_rate_hz"],
                    },
                    "message_counts": {
                        "/livox/lidar": audit["lidar_count"],
                        "/livox/imu": audit["imu_count"],
                    },
                    "frame_ids": audit["frame_ids"],
                    "pointcloud2_fields": audit["pointcloud2_fields"],
                    "motion_audit_status": audit["motion_status"],
                    "ACQUISITION_AUDIT_PASS": True,
                },
            }
        )
    flat_stations = []
    for station in canonical["station_acquisition_audits"]:
        flat_stations.append(
            {
                "scene_id": station["scene_id"],
                "station_id": station["station_id"],
                "acquisition_status": station["acquisition_status"],
                "map_duration_s": station["map"]["duration_s"],
                "query_duration_s": station["query"]["duration_s"],
                "map_lidar_rate_hz": station["map"]["lidar_rate_hz"],
                "query_lidar_rate_hz": station["query"]["lidar_rate_hz"],
                "map_imu_rate_hz": station["map"]["imu_rate_hz"],
                "query_imu_rate_hz": station["query"]["imu_rate_hz"],
                "map_lidar_count": station["map"]["lidar_count"],
                "query_lidar_count": station["query"]["lidar_count"],
                "map_imu_count": station["map"]["imu_count"],
                "query_imu_count": station["query"]["imu_count"],
                "map_motion_status": station["map"]["motion_status"],
                "query_motion_status": station["query"]["motion_status"],
                "gap_s": station["pair"]["gap_s"],
                "map_query_no_overlap": station["pair"]["no_overlap"],
            }
        )
    readiness = dict(canonical["readiness"])
    readiness["MEASUREMENT_FINAL_RESULT"] = readiness.pop(
        "FORMAL_MEASUREMENT_RESULT"
    )
    complete = {
        **canonical,
        "schema": "mid360_fmb1_complete_frozen_manifest_v1",
        "mapping": canonical["bags"],
        "raw_bags": details,
        "stations": flat_stations,
        "readiness": readiness,
    }
    complete.pop("bags")
    complete.pop("scenes")
    complete.pop("station_acquisition_audits")
    for row in complete["geometry_scenes"]:
        row["semantic_candidate_label"] = (
            "RICH_CANDIDATE"
            if row["scene_id"].startswith("FMB1_R")
            else "WEAK_CANDIDATE"
        )
    for row in complete["geometry_metrics"]:
        row.pop("query_timestamp")
    for row in complete["canonical_inputs"]:
        row["T0"] = json.dumps(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            separators=(",", ":"),
        )
    assert validate_manifest_payload(complete, repository)["status"] == "PASS"


def _mapping(payload: dict) -> None:
    payload["bags"][0]["capture_prefix"] = "20260819_999999"


def _swap_roles(payload: dict) -> None:
    payload["bags"][0]["role"], payload["bags"][1]["role"] = (
        payload["bags"][1]["role"],
        payload["bags"][0]["role"],
    )


def _bag_sha(payload: dict) -> None:
    payload["bags"][0]["sha256"] = "f" * 64


def _gap(payload: dict) -> None:
    payload["station_acquisition_audits"][0]["pair"]["gap_s"] = 9.999


def _invalid_station_claimed_pass(payload: dict) -> None:
    payload["station_acquisition_audits"][0]["map"]["duration_s"] = 18.999


def _query_timestamp(payload: dict) -> None:
    payload["snapshots"][0]["query_timestamp"] += 0.25


def _quantile(payload: dict) -> None:
    payload["snapshots"][0]["quantile"] = 0.06


def _geometry_label(payload: dict) -> None:
    payload["geometry_scenes"][0]["final_geometry_class"] = "WEAK"


def _delete_snapshot(payload: dict) -> None:
    payload["snapshots"].pop()


def _delete_station(payload: dict) -> None:
    payload["stations"].pop()


def _target_sha(payload: dict) -> None:
    payload["targets"][0]["target_npy_sha256"] = "e" * 64


def _query_enters_target(payload: dict) -> None:
    payload["targets"][0]["query_contribution_to_target"] = 1


def _registration_result(payload: dict) -> None:
    payload["geometry_metrics"][0]["fitness"] = 0.99


def _backend_contract(payload: dict) -> None:
    payload["backend_parameter_contract_sha256"] = "d" * 64


@pytest.mark.parametrize(
    "mutator,error",
    [
        (_mapping, "capture prefix changed"),
        (_swap_roles, "raw filename/capture mapping changed"),
        (_bag_sha, "not bound to its MAP bag SHA"),
        (_gap, "gap is below 10 seconds"),
        (_invalid_station_claimed_pass, "duration is below hard minimum"),
        (_query_timestamp, "query timestamp binding changed"),
        (_quantile, "snapshot quantiles changed"),
        (_geometry_label, "geometry label changed"),
        (_delete_snapshot, "snapshot count must be 180"),
        (_delete_station, "station count must be 18"),
        (_target_sha, "target SHA binding changed"),
        (_query_enters_target, "contains QUERY contribution"),
        (_registration_result, "registration-derived/result field is forbidden"),
        (_backend_contract, "backend parameter contract SHA differs"),
    ],
    ids=[
        "scene-mapping",
        "map-query-role-swap",
        "bag-sha",
        "gap",
        "invalid-station-pass",
        "query-timestamp",
        "snapshot-quantile",
        "geometry-label",
        "snapshot-deletion",
        "three-stations-to-two",
        "target-sha",
        "query-in-target",
        "registration-result",
        "backend-parameter-contract",
    ],
)
def test_structural_tampering_fails_closed(
    repository: Path, mutator: Callable[[dict], None], error: str
) -> None:
    payload = valid_payload(repository)
    mutator(payload)
    with pytest.raises(FormalBatchError, match=error):
        validate_manifest_payload(payload, repository)


@pytest.mark.parametrize(
    "kind,error",
    [
        ("raw", "raw bag.*SHA256 mismatch"),
        ("target", "target.*SHA256 mismatch"),
        ("source", "snapshot source.*SHA256 mismatch"),
    ],
)
def test_verify_files_rehashes_raw_target_and_source(
    repository: Path, kind: str, error: str
) -> None:
    payload = valid_payload(repository, materialize=True)
    result = validate_manifest_payload(payload, repository, verify_files=True)
    assert result["checks"]["file_authentication"] is True

    if kind == "raw":
        path = Path(payload["bags"][0]["raw_absolute_path"])
    elif kind == "target":
        path = Path(payload["targets"][0]["target_path"])
    else:
        path = Path(payload["snapshots"][0]["source_path"])
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 0x01]) + original[1:])
    with pytest.raises(FormalBatchError, match=error):
        validate_manifest_payload(payload, repository, verify_files=True)


def test_manifest_sha_is_checked_when_declared(repository: Path) -> None:
    payload = valid_payload(repository)
    first = validate_manifest_payload(payload, repository)
    payload["manifest_sha256"] = first["manifest_sha256"]
    assert validate_manifest_payload(payload, repository)["status"] == "PASS"
    payload["readiness"]["planned_total_trials"] = 359
    with pytest.raises(FormalBatchError, match="manifest_sha256 mismatch"):
        validate_manifest_payload(payload, repository)


def test_independent_cli_report_records_geometry_failure_instead_of_crashing(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = valid_payload(repository, materialize=True)
    for row in payload["geometry_metrics"]:
        if row["scene_id"] == "FMB1_W02":
            row.update(
                normalized_lambda_min_trans=0.15,
                condition_number_trans=4.0,
                spectral_entropy_trans=0.85,
            )
    scene = next(
        row for row in payload["geometry_scenes"] if row["scene_id"] == "FMB1_W02"
    )
    scene.update(
        final_geometry_class="INTERMEDIATE",
        admitted=False,
        geometry_lambda_min_median=0.15,
        geometry_condition_median=4.0,
        geometry_entropy_median=0.85,
    )
    manifest = repository / "results/fmb1_frozen_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(verifier_cli, "REPOSITORY", repository)
    report, exit_code = verifier_cli.build_report(manifest)
    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["FORMAL_REGISTRATION_AUTHORIZED"] is False
    assert report["actual_registration_trials"] == 0
    assert "FMB1_W02" in report["error"]["message"]
