from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_environment import (
    EXPECTED_VERSIONS,
    collect_environment_manifest,
)
from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_lock import (
    DEFAULT_BINDING_PATHS,
    build_lock_payload,
    finalize_lock_directory_checksums,
    write_lock_bundle,
)


COMMIT = "a" * 40
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
SCENE_CLASS = {
    "FMB1_R01": "RICH", "FMB1_R02": "RICH", "FMB1_R03": "RICH",
    "FMB1_W01": "WEAK", "FMB1_W02": "WEAK", "FMB1_W03": "WEAK",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def build_valid_r1_lock(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    source_root = Path(__file__).resolve().parents[2]
    contract = root / "frozen_assets/backend_parameter_contract.json"
    contract.parent.mkdir(parents=True)
    shutil.copyfile(source_root / "frozen_assets/backend_parameter_contract.json", contract)
    pcl = root / "bin/pcl_point_to_plane_cli"
    pcl.parent.mkdir(parents=True)
    pcl.write_bytes(b"test PCL 1.15.1 executable")
    pcl.chmod(0o755)

    prereg = root / DEFAULT_BINDING_PATHS["original_preregistration"]
    protocol = root / DEFAULT_BINDING_PATHS["original_capture_radius_analysis_protocol"]
    prereg.parent.mkdir(parents=True, exist_ok=True)
    prereg.write_text("frozen preregistration\n")
    protocol.write_text("CAPTURE_RADIUS_FEASIBILITY_PROTOCOL preserved\n")
    amendment = root / DEFAULT_BINDING_PATHS["active_amendment"]
    write_json(amendment, {
        "schema": "mid360_fmb1_zero_perturbation_mainline_amendment_v1_1_r1",
        "amendment_id": "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1",
        "status": "ACTIVE", "activation_effective": True,
        "FORMAL_TRIAL_COUNT_AT_ACTIVATION": 0,
    })
    activation = root / DEFAULT_BINDING_PATHS["amendment_activation_record"]
    write_json(activation, {
        "amendment_id": "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1",
        "status": "ACTIVE", "ACTIVATED_BEFORE_ANY_FORMAL_ICP": True,
        "FORMAL_TRIAL_COUNT_AT_ACTIVATION": 0,
    })
    write_json(root / DEFAULT_BINDING_PATHS["active_protocol_pointer"], {
        "status": "ACTIVE",
        "active_amendment_id": "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1",
    })
    analysis = root / DEFAULT_BINDING_PATHS["analysis_contract"]
    write_json(analysis, {
        "schema": "mid360_fmb1_zero_perturbation_analysis_contract_v1_1_r1",
        "status": "ACTIVE", "activation_effective": True,
        "prelock_missingness_clarification": {
            "status": "ACTIVE_PRELOCK_CLARIFICATION",
            "path": DEFAULT_BINDING_PATHS["analysis_missingness_clarification"],
            "sha256": sha(source_root / DEFAULT_BINDING_PATHS["analysis_missingness_clarification"]),
            "clarification_before_formal_lock": True,
            "clarification_before_any_formal_icp": True,
            "clarification_at_formal_trial_count": 0,
        },
        "experimental_units": {
            "hierarchy": ["scene", "station", "snapshot", "backend"],
            "highest_independent_unit": "scene", "primary_experimental_unit": "scene",
            "snapshots_are_independent_scenes": False,
            "snapshot_level_naive_p_value_forbidden": True,
            "pooled_180_row_inference_as_independent_samples_forbidden": True,
        },
        "physical_reference": {"physical_reference_semantics": "NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT"},
        "reassociation_analysis": {
            "implementation_path": "src/phase_a_harness/common_association_analysis.py",
            "mid360_specific_redefinition_forbidden": True,
            "frozen_fields": [
                "initial_correspondence_count", "initial_valid_normal_correspondence_count",
                "final_correspondence_count", "final_valid_normal_correspondence_count",
                "correspondence_turnover", "accepted_source_turnover",
                "correspondence_count_change_ratio", "initial_residual_rmse",
                "final_residual_rmse", "residual_rmse_change",
                "median_normal_angle_change_deg", "q95_normal_angle_change_deg",
            ],
            "missingness": {
                "required_result_status_fields": [
                    "common_association_valid", "common_association_invalid_reason",
                    "common_association_invalid_detail",
                ],
                "common_association_invalid_reason_enum": [
                    "NO_INITIAL_CORRESPONDENCE", "NO_FINAL_CORRESPONDENCE",
                    "INSUFFICIENT_VALID_NORMALS", "NONFINITE_COMMON_METRICS", "OTHER",
                ],
                "result_schema_runner_and_verifier_must_preserve_status_fields": True,
            },
        },
    })
    for name in (
        "active_amendment_md", "analysis_protocol",
        "analysis_missingness_clarification",
        "analysis_missingness_clarification_md",
    ):
        destination = root / DEFAULT_BINDING_PATHS[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / DEFAULT_BINDING_PATHS[name], destination)
    history_inventory = root / DEFAULT_BINDING_PATHS["analysis_preclarification_history_inventory"]
    history_inventory.parent.mkdir(parents=True, exist_ok=True)
    source_history = source_root / DEFAULT_BINDING_PATHS["analysis_preclarification_history_inventory"]
    shutil.copyfile(source_history, history_inventory)
    for row in json.loads(source_history.read_text())["files"]:
        shutil.copyfile(source_history.parent / row["name"], history_inventory.parent / row["name"])
    clarification_path = root / DEFAULT_BINDING_PATHS["analysis_missingness_clarification"]
    clarification_md = root / DEFAULT_BINDING_PATHS["analysis_missingness_clarification_md"]
    write_json(root / DEFAULT_BINDING_PATHS["analysis_missingness_clarification_transition"], {
        "schema": "mid360_fmb1_zero_perturbation_analysis_missingness_clarification_transition_v1_1_r1_c1",
        "status": "COMPLETED_PRELOCK",
        "transitioned_at_utc": "2026-08-20T02:00:00+00:00",
        "clarification_before_formal_lock": True,
        "clarification_before_any_formal_icp": True,
        "clarification_at_formal_trial_count": 0,
        "before": {
            "archive_inventory_path": DEFAULT_BINDING_PATHS["analysis_preclarification_history_inventory"],
            "archive_inventory_sha256": sha(history_inventory),
        },
        "after": {
            "amendment_json_sha256": sha(amendment),
            "amendment_md_sha256": sha(root / DEFAULT_BINDING_PATHS["active_amendment_md"]),
            "analysis_contract_sha256": sha(analysis),
            "analysis_protocol_sha256": sha(root / DEFAULT_BINDING_PATHS["analysis_protocol"]),
            "activation_record_sha256": sha(activation),
            "active_protocol_pointer_sha256": sha(root / DEFAULT_BINDING_PATHS["active_protocol_pointer"]),
            "clarification_json_sha256": sha(clarification_path),
            "clarification_markdown_sha256": sha(clarification_md),
        },
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    })
    result_schema = root / DEFAULT_BINDING_PATHS["result_schema"]
    shutil.copyfile(
        source_root / "experiments/mid360_formal_batch1/zero_perturbation_trial_result_schema_v1_1.json",
        result_schema,
    )

    # The lock binds every future execution asset even though tests never load
    # a real backend.  Commit authentication is disabled only inside this
    # synthetic, non-Git fixture.
    for name in (
        "execution_runner", "execution_runner_cli",
        "execution_experiments_package_init",
        "execution_mid360_formal_batch1_package_init",
        "execution_phase_a_harness_package_init", "execution_environment",
        "execution_result_validator",
        "execution_open3d_backend", "execution_pcl_backend",
        "execution_common_association", "execution_rotation_metrics",
        "execution_metrics", "execution_types",
    ):
        destination = root / DEFAULT_BINDING_PATHS[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / DEFAULT_BINDING_PATHS[name], destination)
    for relative in (
        "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_lock.py",
        "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_verify.py",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, destination)
    for name in (
        "original_zero_perturbation_proposal_json",
        "original_zero_perturbation_proposal_md",
        "proposal_superseded_sidecar",
        "w04_superseded_history",
    ):
        destination = root / DEFAULT_BINDING_PATHS[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / DEFAULT_BINDING_PATHS[name], destination)

    final_dir = root / "results/mid360_formal_batch1/final_dataset_v1"
    final_dir.mkdir(parents=True)
    scene_rows = [
        {"scene_id": scene, "attempt": 2 if scene == "FMB1_W02" else 1,
         "final_geometry_class": klass, "admitted": True}
        for scene, klass in SCENE_CLASS.items()
    ]
    station_rows: list[dict[str, object]] = []
    for scene in SCENE_CLASS:
        for station in range(1, 4):
            row: dict[str, object] = {
                "scene_id": scene, "station_id": f"S0{station}",
                "attempt": 2 if scene == "FMB1_W02" else 1,
            }
            if scene == "FMB1_W02":
                row.update({
                    "station_acquisition_status": "ACQUISITION_PASS",
                    "attempt_status": "VALID_ACQUISITION",
                    "map_bag_status": "PASS", "query_bag_status": "PASS",
                    "pair_audit": {"FORMAL_PAIR_VALID": True},
                })
            else:
                row["acquisition_status"] = "ACQUISITION_PASS"
            station_rows.append(row)
    write_json(final_dir / "final_scene_registry.yaml", {"scenes": scene_rows})
    write_json(final_dir / "final_station_registry.yaml", {"stations": station_rows})
    write_json(final_dir / "acquisition_attempt_lineage.json", {
        "scene_id": "FMB1_W02", "invalid_attempt": 1, "valid_attempt": 2,
        "invalid_attempt_status": "INVALID_ACQUISITION",
        "valid_attempt_status": "GEOMETRY_ADMITTED",
        "valid_attempt_final_geometry_class": "WEAK",
        "invalid_attempt_snapshot_count_in_final": 0,
        "valid_attempt_snapshot_count_in_final": 30,
        "W04_IDENTIFIER_RETIRED": True, "W04_INCLUDED_IN_FINAL_SET": False,
        "formal_trial_count_at_correction": 0,
    })
    write_json(final_dir / "invalid_attempt_archive_manifest.json", {
        "attempt": 1, "archived_bag_count": 6,
        "bags": [{"included_in_active_dataset": False, "sha256": str(i) * 64} for i in range(1, 7)],
    })
    write_json(final_dir / "NO_ICP_ATTESTATION.json", {
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "open3d_registration_call_count": 0, "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0, "formal_trial_count": 0,
        "actual_formal_trials": 0,
    })
    raw_rows = [
        {"scene_id": scene, "station_id": f"S0{station}", "attempt": 2 if scene == "FMB1_W02" else 1,
         "role": role, "sha256": hashlib.sha256(f"{scene}/{station}/{role}".encode()).hexdigest()}
        for scene in SCENE_CLASS for station in range(1, 4) for role in ("MAP", "QUERY")
    ]
    write_csv(final_dir / "final_raw_bag_manifest.csv", raw_rows)
    target_rows: list[dict[str, object]] = []
    snapshot_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    for scene in SCENE_CLASS:
        attempt = 2 if scene == "FMB1_W02" else 1
        for station_index in range(1, 4):
            station = f"S0{station_index}"
            target = root / "runtime/targets" / scene / station / "target.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"target/{scene}/{station}".encode())
            target_rows.append({
                "scene_id": scene, "station_id": station, "attempt": attempt,
                "target_path": str(target), "target_npy_sha256": sha(target),
                "query_contribution_to_target": 0,
            })
            for index in range(10):
                sid = f"{scene}_{station}_Q{index+1:02d}"
                source = root / "runtime/sources" / scene / station / f"{sid}.npy"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(f"source/{sid}".encode())
                snapshot_rows.append({
                    "scene_id": scene, "station_id": station, "attempt": attempt,
                    "snapshot_id": sid, "selection_index": index,
                    "source_path": str(source), "source_npy_sha256": sha(source),
                    "target_npy_sha256": sha(target), "query_timestamp": float(index),
                })
                geometry_rows.append({"scene_id": scene, "station_id": station, "attempt": attempt, "snapshot_id": sid})
    write_csv(final_dir / "final_target_manifest.csv", target_rows)
    write_csv(final_dir / "final_snapshot_manifest.csv", snapshot_rows)
    write_csv(final_dir / "final_geometry_manifest.csv", geometry_rows)
    final_manifest = final_dir / "final_dataset_manifest.json"
    write_json(final_manifest, {"schema": "test_final", "scene_count": 6, "snapshot_count": 180})
    pointer = root / DEFAULT_BINDING_PATHS["final_dataset_pointer"]
    write_json(pointer, {
        "active_dataset_path": "results/mid360_formal_batch1/final_dataset_v1",
        "active_manifest_sha256": sha(final_manifest),
        "scene_ids": list(SCENE_CLASS), "w02_active_attempt": 2,
        "W04_IDENTIFIER_RETIRED": True, "W04_INCLUDED_IN_FINAL_SET": False,
        "actual_formal_trials": 0,
    })

    amendment_sha, analysis_sha = sha(amendment), sha(analysis)
    plan_rows: list[dict[str, object]] = []
    target_by_key = {(str(row["scene_id"]), str(row["station_id"])): row for row in target_rows}
    for snapshot in snapshot_rows:
        scene, station, sid = str(snapshot["scene_id"]), str(snapshot["station_id"]), str(snapshot["snapshot_id"])
        target = target_by_key[(scene, station)]
        for backend in ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE"):
            code = "O3D" if backend.startswith("OPEN3D") else "PCL"
            plan_rows.append({
                "trial_id": f"FMB1-ZP11R1-{sid.replace('_', '-')}-{code}",
                "batch_id": "FMB1",
                "amendment_id": "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1",
                "track_id": "ZERO_PERTURBATION_TRACK", "scene_id": scene,
                "final_geometry_class": SCENE_CLASS[scene], "station_id": station,
                "attempt": int(snapshot["attempt"]), "snapshot_id": sid,
                "query_timestamp": snapshot["query_timestamp"], "backend": backend,
                "backend_version": "0.19.0+b012259" if backend.startswith("OPEN3D") else "1.15.1",
                "source_reference": snapshot["source_path"], "source_sha256": snapshot["source_npy_sha256"],
                "source_array_sha256": snapshot["source_npy_sha256"], "source_point_count": 1,
                "target_reference": target["target_path"], "target_sha256": target["target_npy_sha256"],
                "target_array_sha256": target["target_npy_sha256"], "target_point_count": 1,
                "T0": IDENTITY, "T_reference_nominal": IDENTITY,
                "translation_perturbation_m": 0, "rotation_perturbation_deg": 0,
                "backend_parameter_contract_sha256": sha(contract),
                "active_amendment_sha256": amendment_sha,
                "analysis_contract_sha256": analysis_sha,
                "backend_canonical_parameter_sha256": json.loads(contract.read_text())["open3d" if backend.startswith("OPEN3D") else "pcl"]["canonical_sha256"],
                "planned_status": "PLANNED_NOT_AUTHORIZED_NOT_EXECUTED",
            })
    plan_json = root / DEFAULT_BINDING_PATHS["trial_plan_json"]
    write_json(plan_json, {
        "schema": "mid360_fmb1_zero_perturbation_trial_plan_v1_1_r1",
        "plan_id": "FMB1_ZERO_PERTURBATION_TRIAL_PLAN_V1_1_R1",
        "amendment_id": "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1",
        "track_id": "ZERO_PERTURBATION_TRACK",
        "counts": {"scene_count": 6, "station_count": 18, "snapshot_count": 180,
                   "open3d_trial_count": 180, "pcl_trial_count": 180, "total_trial_count": 360},
        "rows": plan_rows,
    })
    write_csv(root / DEFAULT_BINDING_PATHS["trial_plan_csv"], plan_rows)

    lock_dir = root / "results/mid360_formal_batch1/zero_perturbation_v1_1_lock"
    lock_dir.mkdir(parents=True)
    historical_json = root / DEFAULT_BINDING_PATHS["original_zero_perturbation_proposal_json"]
    historical_md = root / DEFAULT_BINDING_PATHS["original_zero_perturbation_proposal_md"]
    superseded = root / DEFAULT_BINDING_PATHS["proposal_superseded_sidecar"]
    lineage_path = root / DEFAULT_BINDING_PATHS["acquisition_attempt_lineage"]
    write_json(root / DEFAULT_BINDING_PATHS["proposal_correction_record"], {
        "schema": "mid360_fmb1_zero_perturbation_proposal_correction_record_v1_1_r1",
        "status": "RECORDED_PRE_ACTIVATION",
        "correction_reason": "OPERATOR_CONFIRMED_WRONG_SCENE_LOCATION",
        "correction_before_any_formal_icp": True,
        "correction_at_formal_trial_count": 0,
        "registration_evidence_used": False,
        "old_proposal": {
            "json_sha256": sha(historical_json),
            "markdown_sha256": sha(historical_md),
            "supersession_record_sha256": sha(superseded),
        },
        "lineage": {
            "scene_id": "FMB1_W02", "attempt1_status": "INVALID_ACQUISITION",
            "attempt2_status": "GEOMETRY_ADMITTED",
            "attempt2_final_geometry_class": "WEAK",
            "attempt2_in_final_dataset": True,
            "w04_identifier_retired": True, "w04_in_final_dataset": False,
            "source_sha256": sha(lineage_path),
        },
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    })
    write_json(root / DEFAULT_BINDING_PATHS["proposal_difference_report"], {
        "schema": "mid360_fmb1_zero_perturbation_proposal_difference_report_v1_1_r1",
        "status": "DIFFERENCES_DOCUMENTED_VERSIONED_CORRECTION_REQUIRED",
        "old_proposal_preserved_byte_for_byte": True,
        "old_proposal_json_sha256_before_r1": sha(historical_json),
        "old_proposal_md_sha256_before_r1": sha(historical_md),
        "reviewed_inputs": [
            {"path": DEFAULT_BINDING_PATHS["original_zero_perturbation_proposal_json"],
             "sha256": sha(historical_json)},
            {"path": DEFAULT_BINDING_PATHS["original_zero_perturbation_proposal_md"],
             "sha256": sha(historical_md)},
            {"path": DEFAULT_BINDING_PATHS["acquisition_attempt_lineage"],
             "sha256": sha(lineage_path)},
        ],
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    })
    env = collect_environment_manifest(
        root, pcl_executable=pcl,
        observed_versions={key: value for key, value in EXPECTED_VERSIONS.items() if key != "pcl"},
        pcl_version_text="1.15.1", ldd_text="libpcl_common.so => /test/libpcl_common.so",
    )
    write_json(lock_dir / "environment_manifest.json", env)
    detailed = root / "results/mid360_formal_batch1/final_dataset_prelock_reauthentication.json"
    detailed_md = detailed.with_suffix(".md")
    write_json(detailed, {
        "status": "PASS", "pass": True, "final_scene_count": 6,
        "final_station_count": 18, "final_target_count": 18,
        "final_snapshot_count": 180, "protected_assets": {"unchanged": True},
        "formal_execution_state": {
            "actual_open3d_trials": 0, "actual_pcl_trials": 0,
            "actual_formal_trials": 0, "registration_execution_count": 0,
            "FORMAL_ICP_UNLOCKED": False, "FORMAL_REGISTRATION_AUTHORIZED": False,
        },
    })
    detailed_md.write_text("PASS\n")
    write_json(lock_dir / "final_dataset_prelock_reauthentication.json", {
        "schema": "mid360_fmb1_zero_perturbation_v1_1_r1_prelock_dataset_binding",
        "status": "PASS", "pass": True,
        "source_report_path": detailed.relative_to(root).as_posix(),
        "source_report_sha256": sha(detailed),
        "source_markdown_path": detailed_md.relative_to(root).as_posix(),
        "source_markdown_sha256": sha(detailed_md),
    })
    write_json(root / DEFAULT_BINDING_PATHS["trial_plan_independent_verification"], {
        "status": "PASS", "payload_byte_hashes_verified": True,
        "single_authoritative_byte_source": True, "scene_count": 6,
        "station_count": 18, "snapshot_count": 180, "total_trial_count": 360,
        "open3d_trial_count": 180, "pcl_trial_count": 180,
        "identity_t0_count": 360, "w04_trial_count": 0,
        "old_w02_attempt1_trial_count": 0, "w02_attempt2_trial_count": 60,
        "registration_backend_call_count": 0,
        "FORMAL_REGISTRATION_AUTHORIZED": False, "actual_formal_trials": 0,
        "trial_plan_json_sha256": sha(plan_json),
        "trial_plan_csv_sha256": sha(root / DEFAULT_BINDING_PATHS["trial_plan_csv"]),
        "result_schema_sha256": sha(result_schema),
    })
    history_rows = {
        row["name"]: row["sha256"]
        for row in json.loads(history_inventory.read_text())["files"]
    }
    old_hashes = {
        "amendment_json_sha256": history_rows["zero_perturbation_mainline_v1_1_r1.json"],
        "amendment_md_sha256": history_rows["zero_perturbation_mainline_v1_1_r1.md"],
        "analysis_contract_sha256": history_rows["zero_perturbation_analysis_contract_v1_1_r1.json"],
        "analysis_protocol_sha256": history_rows["zero_perturbation_analysis_protocol_v1_1_r1.md"],
    }
    write_json(root / DEFAULT_BINDING_PATHS["protocol_transition_independent_verification"], {
        "pass": True, "verification_status": "PASS", "activation_effective": True,
        "FORMAL_LOCK_ISSUED": False, "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0, "backend_calls": 0, "backend_modules_imported": 0,
        "activation_record_sha256": history_rows["amendment_activation_record_v1_1_r1.json"],
        "active_protocol_pointer_sha256": history_rows["ACTIVE_PROTOCOL.json"],
        "active_hashes": old_hashes,
    })
    write_json(root / DEFAULT_BINDING_PATHS["protocol_c1_missingness_independent_verification"], {
        "pass": True, "verification_status": "PASS", "phase": "ACTIVE_R1_C1_PRE_LOCK",
        "activation_effective": True, "common_association_status_fields_verified": True,
        "FORMAL_LOCK_ISSUED": False, "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0, "backend_calls": 0, "backend_modules_imported": 0,
        "current_active_hashes": {
            "amendment_json_sha256": sha(root / DEFAULT_BINDING_PATHS["active_amendment"]),
            "amendment_md_sha256": sha(root / DEFAULT_BINDING_PATHS["active_amendment_md"]),
            "analysis_contract_sha256": sha(root / DEFAULT_BINDING_PATHS["analysis_contract"]),
            "analysis_protocol_sha256": sha(root / DEFAULT_BINDING_PATHS["analysis_protocol"]),
        },
        "initial_active_hashes_preserved": old_hashes,
        "activation_record_sha256": sha(root / DEFAULT_BINDING_PATHS["amendment_activation_record"]),
        "active_protocol_pointer_sha256": sha(root / DEFAULT_BINDING_PATHS["active_protocol_pointer"]),
        "clarification_json_sha256": sha(root / DEFAULT_BINDING_PATHS["analysis_missingness_clarification"]),
        "clarification_markdown_sha256": sha(root / DEFAULT_BINDING_PATHS["analysis_missingness_clarification_md"]),
        "clarification_transition_sha256": sha(root / DEFAULT_BINDING_PATHS["analysis_missingness_clarification_transition"]),
        "prior_active_inventory_sha256": sha(history_inventory),
        "common_association_status_artifacts": {
            "execution_verifier_sha256": sha(root / "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_verify.py"),
            "result_schema_sha256": sha(result_schema),
            "runner_sha256": sha(root / DEFAULT_BINDING_PATHS["execution_runner"]),
            "result_validator_sha256": sha(root / DEFAULT_BINDING_PATHS["execution_result_validator"]),
            "experiments_package_init_sha256": sha(root / DEFAULT_BINDING_PATHS["execution_experiments_package_init"]),
            "mid360_formal_batch1_package_init_sha256": sha(root / DEFAULT_BINDING_PATHS["execution_mid360_formal_batch1_package_init"]),
            "phase_a_harness_package_init_sha256": sha(root / DEFAULT_BINDING_PATHS["execution_phase_a_harness_package_init"]),
        },
    })
    c1_report = root / DEFAULT_BINDING_PATHS["protocol_c1_missingness_independent_verification"]
    write_json(root / DEFAULT_BINDING_PATHS["activation_review"], {
        "schema": "mid360_fmb1_zero_perturbation_v1_1_r1_activation_review",
        "review_status": "PASS_ACTIVE_R1_C1",
        "activation_effective": True,
        "reviewed_at_utc": "2026-08-20T03:00:00+00:00",
        "missingness_clarification_verifier_pass": True,
        "missingness_clarification_independent_verification": {
            "path": DEFAULT_BINDING_PATHS["protocol_c1_missingness_independent_verification"],
            "sha256": sha(c1_report),
            "status": "PASS",
        },
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    })
    lock, inventory = build_lock_payload(
        root, execution_code_commit=COMMIT,
        issued_at_utc="2026-08-20T00:00:00+00:00",
        remeasure_environment_versions=False,
        verify_execution_commit=False,
    )
    write_lock_bundle(root, lock_dir, lock, inventory)
    finalize_lock_directory_checksums(lock_dir)
    return root, lock_dir, root / (
        "zero_perturbation_runtime/mid360_formal_batch1_zero_perturbation_v1_1"
    )


def build_valid_exec_r2_runner_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Adapt the frozen R1 unit fixture to the exec-r2 runner contract.

    This is intentionally only a synthetic unit-test bundle.  The dedicated
    exec-r2 integration tests exercise the real producer and independent
    verifier against a two-commit temporary Git repository.
    """

    root, old_lock_dir, _old_runtime = build_valid_r1_lock(tmp_path)
    old_lock_path = old_lock_dir / "formal_batch1_zero_perturbation_lock_v1_1.json"
    lock = json.loads(old_lock_path.read_text(encoding="utf-8"))
    lock.update({
        "schema": "mid360_fmb1_zero_perturbation_formal_lock_v1_1_exec_r2",
        "lock_id": "FIXTURE_ONLY_EXEC_R2_LOCK",
        "execution_lock_revision": 2,
        "status": "ISSUED_AWAITING_SEPARATE_AUTHORIZATION",
        "authoritative_runtime_root": (
            "zero_perturbation_runtime/"
            "mid360_zero_perturbation_v1_1_formal_execution_v1"
        ),
        "AUTHORIZATION_PRODUCER_READY": True,
        "INDEPENDENT_AUTHORIZATION_VERIFIER_READY": True,
        "AUTHORIZATION_LIFECYCLE_QUALIFIED": True,
        "FORMAL_LOCK_ISSUED": True,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "registration_execution_count": 0,
    })
    lock_dir = root / "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r2_lock"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.json"
    write_json(lock_dir / lock_name, lock)
    (lock_dir / "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.sha256").write_text(
        f"{sha(lock_dir / lock_name)}  {lock_name}\n", encoding="ascii"
    )
    shutil.copyfile(old_lock_dir / "lock_inventory.csv", lock_dir / "lock_inventory.csv")
    material = {
        "lock_file_sha256": sha(lock_dir / lock_name),
        "lock_inventory_file_sha256": sha(lock_dir / "lock_inventory.csv"),
        "execution_code_commit": COMMIT,
    }
    fingerprint = hashlib.sha256(
        (json.dumps(material, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    write_json(lock_dir / "lock_fingerprint.json", {
        **material, "lock_fingerprint": fingerprint, "execution_lock_revision": 2,
    })
    shutil.copyfile(old_lock_dir / "environment_manifest.json", lock_dir / "environment_manifest.json")
    write_json(lock_dir / "NO_REGISTRATION_ATTESTATION.json", {
        "status": "PASS", "pass": True,
        "open3d_registration_call_count": 0, "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0, "formal_trial_count": 0,
        "actual_formal_trials": 0,
    })
    write_json(lock_dir / "execution_control_patch_report.json", {
        "status": "PASS", "SCIENTIFIC_PROTOCOL_CHANGED": False,
        "FINAL_DATASET_CHANGED": False,
        "TRIAL_PLAN_SCIENTIFIC_CONTENT_CHANGED": False,
        "BACKEND_PARAMETERS_CHANGED": False, "ACTUAL_FORMAL_TRIALS": 0,
    })
    write_json(lock_dir / "authorization_lifecycle_test_report.json", {
        "status": "PASS", "AUTHORIZATION_LIFECYCLE_QUALIFIED": True,
        "tamper_case_count": 25, "REAL_FORMAL_TRIALS": 0,
    })
    core_names = (
        lock_name, "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.sha256",
        "lock_inventory.csv", "lock_fingerprint.json",
        "NO_REGISTRATION_ATTESTATION.json", "environment_manifest.json",
        "execution_control_patch_report.json",
        "authorization_lifecycle_test_report.json",
    )
    (lock_dir / "LOCK_CORE_SHA256SUMS").write_text(
        "".join(f"{sha(lock_dir / name)}  {name}\n" for name in core_names),
        encoding="ascii",
    )
    runtime = root / lock["authoritative_runtime_root"]
    return root, lock_dir, runtime
