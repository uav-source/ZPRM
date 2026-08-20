from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.protocol import BACKEND_CONTRACT_SHA256, QUERY_QUANTILES
from experiments.mid360_formal_batch1.w04_final_dataset import (
    FINAL_SCENES,
    ORIGINAL_SCENES,
    W04FinalDatasetError,
    build_final_dataset_payload,
    verify_final_dataset_payload,
    write_final_dataset,
    write_sha256sums,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metric(scene: str, station: str, selection: int, classification: str) -> dict:
    if classification == "RICH":
        nlambda, condition, entropy = 0.20, 2.0, 0.95
    else:
        nlambda, condition, entropy = 0.10, 7.0, 0.70
    return {
        "scene_id": scene,
        "station_id": station,
        "snapshot_id": f"{scene}_{station}_Q{selection:02d}",
        "selection_index": selection,
        "initial_correspondence_count": 100,
        "initial_valid_normal_correspondence_count": 90,
        "lambda_min_trans": nlambda,
        "lambda_mid_trans": 0.3,
        "lambda_max_trans": 0.6,
        "normalized_lambda_min_trans": nlambda,
        "normalized_lambda_mid_trans": 0.3,
        "normalized_lambda_max_trans": 0.6,
        "condition_number_trans": condition,
        "spectral_entropy_trans": entropy,
    }


def _raw(scene: str, station: str, role: str, prefix: str, path: str) -> dict:
    part = "part1_20s" if role == "MAP" else "part2_15s"
    return {
        "scene_id": scene,
        "station_id": station,
        "role": role,
        "raw_filename": f"mid360_{prefix}_{part}.bag",
        "raw_absolute_path": path,
        "capture_prefix": prefix,
        "sha256": _digest(f"{scene}/{station}/{role}"),
        "bytes": 3,
    }


def _inputs(paths: dict[str, str] | None = None) -> tuple[dict, ...]:
    raw_path = paths["raw"] if paths else "/fixture/raw.bag"
    target_path = paths["target"] if paths else "/fixture/target.npy"
    source_path = paths["source"] if paths else "/fixture/source.npy"
    original_raw = []
    stations = []
    targets = []
    snapshots = []
    metrics = []
    geometry_scenes = []
    for scene_index, scene in enumerate(ORIGINAL_SCENES):
        classification = "RICH" if scene.startswith("FMB1_R") or scene == "FMB1_W02" else "WEAK"
        prefix = f"20260101_{scene_index:02d}0000"
        for station in ("S01", "S02", "S03"):
            for role in ("MAP", "QUERY"):
                original_raw.append(_raw(scene, station, role, prefix, raw_path))
            stations.append(
                {
                    "scene_id": scene,
                    "station_id": station,
                    "acquisition_status": "ACQUISITION_PASS",
                }
            )
            target_sha = _digest(f"target/{scene}/{station}")
            map_sha = _digest(f"{scene}/{station}/MAP")
            targets.append(
                {
                    "scene_id": scene,
                    "station_id": station,
                    "map_bag_sha256": map_sha,
                    "target_path": target_path,
                    "target_npy_sha256": target_sha,
                    "target_size_bytes": 6,
                    "input_roles": "MAP",
                    "query_contribution_to_target": 0,
                    "query_frame_count": 0,
                    "construction": "DIRECT_MERGE_NO_REGISTRATION",
                    "registration_called": False,
                    "odometry_called": False,
                    "scan_matching_called": False,
                }
            )
            for selection, quantile in enumerate(QUERY_QUANTILES):
                snapshot_id = f"{scene}_{station}_Q{selection:02d}"
                snapshots.append(
                    {
                        "scene_id": scene,
                        "station_id": station,
                        "snapshot_id": snapshot_id,
                        "selection_index": selection,
                        "quantile": quantile,
                        "query_bag_sha256": _digest(f"{scene}/{station}/QUERY"),
                        "source_path": source_path,
                        "source_npy_sha256": _digest(f"source/{scene}/{station}/{selection}"),
                        "source_size_bytes": 6,
                        "target_npy_sha256": target_sha,
                    }
                )
                metrics.append(_metric(scene, station, selection, classification))
        geometry_scenes.append(
            {
                "scene_id": scene,
                "final_geometry_class": classification,
                "geometry_admission_status": (
                    "GEOMETRY_REJECTED" if scene == "FMB1_W02" else "GEOMETRY_ADMITTED"
                ),
                "failure_reason": (
                    "SEMANTIC_CANDIDATE_GEOMETRY_CLASS_MISMATCH"
                    if scene == "FMB1_W02"
                    else None
                ),
            }
        )
    original = {
        "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
        "raw_bags": original_raw,
        "stations": stations,
        "targets": targets,
        "snapshots": snapshots,
        "geometry_metrics": metrics,
        "geometry_scenes": geometry_scenes,
        "formal_trial_count": 0,
    }

    prefixes = {"S01": "20260820_081749", "S02": "20260820_081954", "S03": "20260820_082207"}
    w04_raw = []
    w04_stations = []
    w04_targets = []
    w04_snapshots = []
    w04_metrics = []
    for station, prefix in prefixes.items():
        for role in ("MAP", "QUERY"):
            w04_raw.append(_raw("FMB1_W04", station, role, prefix, raw_path))
        w04_stations.append(
            {
                "scene_id": "FMB1_W04",
                "station_id": station,
                "acquisition_status": "ACQUISITION_PASS",
            }
        )
        target_sha = _digest(f"target/FMB1_W04/{station}")
        w04_targets.append(
            {
                "scene_id": "FMB1_W04",
                "station_id": station,
                "map_bag_sha256": _digest(f"FMB1_W04/{station}/MAP"),
                "target_path": target_path,
                "target_npy_sha256": target_sha,
                "target_size_bytes": 6,
                "input_roles": ["MAP"],
                "query_contribution_to_target": 0,
                "query_frame_count": 0,
                "construction": "DIRECT_MERGE_NO_REGISTRATION",
                "registration_called": False,
                "odometry_called": False,
                "scan_matching_called": False,
            }
        )
        for selection, quantile in enumerate(QUERY_QUANTILES):
            w04_snapshots.append(
                {
                    "scene_id": "FMB1_W04",
                    "station_id": station,
                    "snapshot_id": f"FMB1_W04_{station}_Q{selection:02d}",
                    "selection_index": selection,
                    "quantile": quantile,
                    "query_bag_sha256": _digest(f"FMB1_W04/{station}/QUERY"),
                    "source_path": source_path,
                    "source_npy_sha256": _digest(f"source/FMB1_W04/{station}/{selection}"),
                    "source_size_bytes": 6,
                    "target_npy_sha256": target_sha,
                }
            )
            w04_metrics.append(_metric("FMB1_W04", station, selection, "WEAK"))
    acquisition = {
        "raw_bags": w04_raw,
        "stations": w04_stations,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
    }
    assets = {"targets": w04_targets, "snapshots": w04_snapshots, "failures": [], "formal_trial_count": 0}
    geometry = {
        "snapshot_metrics": w04_metrics,
        "scene_summaries": [
            {
                "scene_id": "FMB1_W04",
                "final_geometry_class": "WEAK",
                "geometry_admission_status": "GEOMETRY_ADMITTED",
                "median_normalized_lambda_min_trans": 0.10,
                "median_condition_number_trans": 7.0,
                "median_spectral_entropy_trans": 0.70,
            }
        ],
        "failures": [],
        "formal_trial_count": 0,
        "registration_executed": False,
    }
    plan = {
        "rejected_candidate_scene_id": "FMB1_W02",
        "rejection_reason": "GEOMETRY_ONLY_INELIGIBLE",
        "replacement_scene_id": "FMB1_W04",
        "semantic_candidate_label": "WEAK_CANDIDATE",
        "decision_before_any_icp": True,
        "formal_trial_count_at_decision": 0,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "decision_timestamp": "2026-08-19T15:33:07+00:00",
    }
    no_icp = {
        "status": "PASS",
        "pass": True,
        "NO_ICP_ATTESTATION_PASS": True,
        "NO_FORMAL_REGISTRATION": True,
        "open3d_registration_call_count": 0,
        "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0,
        "formal_trial_count": 0,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
    }
    return original, acquisition, assets, geometry, plan, no_icp


def _build() -> dict:
    return build_final_dataset_payload(*_inputs(), verify_files=False)


def test_builds_exact_final_and_historical_counts() -> None:
    payload = _build()
    assert len(payload["raw_candidate_bags"]) == 42
    assert len(payload["final_raw_bags"]) == 36
    assert len(payload["rejected_candidates"]) == 6
    assert len(payload["final_snapshots"]) == 180
    assert {row["scene_id"] for row in payload["final_snapshots"]} == set(FINAL_SCENES)
    assert all(row["scene_id"] != "FMB1_W02" for row in payload["final_snapshots"])
    assert verify_final_dataset_payload(payload, verify_files=False)["PASS"] is True


def test_w02_lineage_is_retained_and_closed() -> None:
    payload = _build()
    lineage = payload["replacement_lineage"]
    assert lineage["rejected_candidate_scene_id"] == "FMB1_W02"
    assert lineage["replacement_scene_id"] == "FMB1_W04"
    assert lineage["rejection_reason"] == "GEOMETRY_ONLY_INELIGIBLE"
    assert lineage["formal_trial_count_at_decision"] == 0
    assert lineage["formal_trial_count_at_admission"] == 0
    assert all(row["retained"] and not row["included_in_final_set"] for row in payload["rejected_candidates"])


def test_w04_nonweak_geometry_fails_closed() -> None:
    inputs = list(_inputs())
    for row in inputs[3]["snapshot_metrics"]:
        row["normalized_lambda_min_trans"] = 0.20
        row["condition_number_trans"] = 2.0
        row["spectral_entropy_trans"] = 0.95
    with pytest.raises(W04FinalDatasetError, match="geometry class is RICH"):
        build_final_dataset_payload(*inputs, verify_files=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda inputs: inputs[2]["snapshots"].pop(), "W04 snapshot keys differ"),
        (lambda inputs: inputs[2]["targets"][0].update(query_contribution_to_target=1), "QUERY contribution"),
        (lambda inputs: inputs[5].update(formal_trial_count=1), "formal_trial_count must be 0"),
        (lambda inputs: inputs[4].update(rejection_reason="CHANGED"), "replacement plan rejection_reason"),
    ],
)
def test_builder_tamper_cases_fail_closed(mutation, message: str) -> None:
    inputs = list(_inputs())
    mutation(inputs)
    with pytest.raises(W04FinalDatasetError, match=message):
        build_final_dataset_payload(*inputs, verify_files=False)


def test_payload_verifier_rejects_lock_or_w02_snapshot() -> None:
    payload = _build()
    locked = copy.deepcopy(payload)
    locked["readiness"]["FORMAL_LOCK_ISSUED"] = True
    with pytest.raises(W04FinalDatasetError, match="FORMAL_LOCK_ISSUED"):
        verify_final_dataset_payload(locked, verify_files=False)

    included = copy.deepcopy(payload)
    included["final_snapshots"][0]["scene_id"] = "FMB1_W02"
    with pytest.raises(W04FinalDatasetError, match="final snapshot keys differ"):
        verify_final_dataset_payload(included, verify_files=False)


def test_writer_emits_required_names_and_refreshes_independent_sha(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.bag"
    target = tmp_path / "target.npy"
    source = tmp_path / "source.npy"
    raw.write_bytes(b"raw")
    target.write_bytes(b"target")
    source.write_bytes(b"source")
    inputs = list(_inputs({"raw": str(raw), "target": str(target), "source": str(source)}))
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    target_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    for row in inputs[0]["raw_bags"] + inputs[1]["raw_bags"]:
        row["sha256"] = raw_sha
        row["bytes"] = raw.stat().st_size
    for row in inputs[0]["targets"] + inputs[2]["targets"]:
        row["map_bag_sha256"] = raw_sha
        row["target_npy_sha256"] = target_sha
        row["target_size_bytes"] = target.stat().st_size
    for row in inputs[0]["snapshots"] + inputs[2]["snapshots"]:
        row["query_bag_sha256"] = raw_sha
        row["source_npy_sha256"] = source_sha
        row["source_size_bytes"] = source.stat().st_size
        row["target_npy_sha256"] = target_sha
    payload = build_final_dataset_payload(*inputs, verify_files=True)
    output = tmp_path / "final_dataset_v1"
    write_final_dataset(payload, output)
    required = {
        "final_dataset_manifest.json",
        "final_scene_registry.yaml",
        "final_station_registry.yaml",
        "final_raw_bag_manifest.csv",
        "final_target_manifest.csv",
        "final_snapshot_manifest.csv",
        "final_geometry_manifest.csv",
        "rejected_candidate_manifest.csv",
        "replacement_lineage.json",
        "final_dataset_readiness.json",
        "NO_ICP_ATTESTATION.json",
        "SHA256SUMS",
    }
    assert required <= {path.name for path in output.iterdir()}
    (output / "independent_verification.json").write_text(
        json.dumps({"pass": True}) + "\n", encoding="utf-8"
    )
    write_sha256sums(output)
    assert "independent_verification.json" in (output / "SHA256SUMS").read_text()
