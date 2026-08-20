from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1 import w04_final_verify as verify
from tools.mid360_formal_batch1 import verify_w04_final_dataset as verify_cli


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _refresh_checksums(final: Path) -> None:
    lines = [f"{_sha(final / name)}  {name}" for name in verify.REQUIRED_FILES]
    (final / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def valid_bundle(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repo"
    final = repository / "results/mid360_formal_batch1/final_dataset_v1"
    final.mkdir(parents=True)
    backend = repository / "frozen_assets/backend_parameter_contract.json"
    backend.parent.mkdir(parents=True)
    source_backend = Path(__file__).resolve().parents[2] / "frozen_assets/backend_parameter_contract.json"
    shutil.copyfile(source_backend, backend)
    proposal = repository / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.json"
    _write_json(proposal, {"status": "PROPOSED_NOT_ACTIVE", "FORMAL_AUTHORITY": False})

    scenes: list[dict[str, object]] = []
    stations: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []
    geometry: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    raw_sha: dict[tuple[str, str, str], str] = {}
    target_sha: dict[tuple[str, str], str] = {}

    for scene in verify.FINAL_SCENES:
        geometry_class = "RICH" if scene in verify.RICH_SCENES else "WEAK"
        scenes.append(
            {
                "scene_id": scene,
                "geometry": {
                    "final_geometry_class": geometry_class,
                    "admitted": True,
                },
            }
        )
        for station in verify.STATIONS:
            stations.append(
                {
                    "scene_id": scene,
                    "station_id": station,
                    "acquisition_status": "ACQUISITION_PASS",
                }
            )
            for role in ("MAP", "QUERY"):
                if scene == "FMB1_W04":
                    prefix = verify.W04_PREFIXES[station]
                else:
                    prefix = f"20260819_{len(raw):06d}"
                suffix = "part1_20s.bag" if role == "MAP" else "part2_15s.bag"
                path = repository / "bags" / f"mid360_{prefix}_{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{scene}/{station}/{role}".encode())
                digest = _sha(path)
                raw_sha[(scene, station, role)] = digest
                raw.append(
                    {
                        "scene_id": scene,
                        "station_id": station,
                        "role": role,
                        "raw_filename": path.name,
                        "raw_absolute_path": str(path),
                        "sha256": digest,
                        "bytes": path.stat().st_size,
                    }
                )
            target = repository / "runtime/targets" / scene / station / "target.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"target/{scene}/{station}".encode())
            digest = _sha(target)
            target_sha[(scene, station)] = digest
            targets.append(
                {
                    "scene_id": scene,
                    "station_id": station,
                    "map_bag_sha256": raw_sha[(scene, station, "MAP")],
                    "target_path": str(target),
                    "target_npy_sha256": digest,
                    "target_size_bytes": target.stat().st_size,
                    "input_roles": "MAP",
                    "query_contribution_to_target": 0,
                    "construction": "DIRECT_SAME_SENSOR_FRAME_MERGE_NO_REGISTRATION",
                    "registration_called": False,
                    "odometry_called": False,
                    "scan_matching_called": False,
                }
            )
            for index, quantile in enumerate(verify.QUANTILES):
                snapshot_id = f"{scene}_{station}_Q{index + 1:02d}"
                source = repository / "runtime/snapshots" / scene / station / f"{snapshot_id}.npy"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(f"source/{snapshot_id}".encode())
                snapshots.append(
                    {
                        "scene_id": scene,
                        "station_id": station,
                        "snapshot_id": snapshot_id,
                        "selection_index": index,
                        "quantile": quantile,
                        "query_bag_sha256": raw_sha[(scene, station, "QUERY")],
                        "source_path": str(source),
                        "source_npy_sha256": _sha(source),
                        "source_size_bytes": source.stat().st_size,
                        "target_npy_sha256": digest,
                    }
                )
                weak = scene in verify.WEAK_SCENES
                geometry.append(
                    {
                        "scene_id": scene,
                        "station_id": station,
                        "snapshot_id": snapshot_id,
                        "selection_index": index,
                        "initial_correspondence_count": 100,
                        "initial_valid_normal_correspondence_count": 90,
                        "lambda_min_trans": 0.05 if weak else 0.25,
                        "lambda_mid_trans": 0.15 if weak else 0.30,
                        "lambda_max_trans": 0.80 if weak else 0.45,
                        "normalized_lambda_min_trans": 0.05 if weak else 0.25,
                        "normalized_lambda_mid_trans": 0.15 if weak else 0.30,
                        "normalized_lambda_max_trans": 0.80 if weak else 0.45,
                        "condition_number_trans": 16.0 if weak else 1.8,
                        "spectral_entropy_trans": 0.60 if weak else 0.96,
                    }
                )

    for station in verify.STATIONS:
        for role in ("MAP", "QUERY"):
            path = repository / "bags" / f"w02_{station}_{role}.bag"
            path.write_bytes(f"W02/{station}/{role}".encode())
            rejected.append(
                {
                    "scene_id": "FMB1_W02",
                    "station_id": station,
                    "role": role,
                    "candidate_status": "GEOMETRY_REJECTED",
                    "exclusion_reason": "GEOMETRY_ONLY_INELIGIBLE",
                    "raw_absolute_path": str(path),
                    "sha256": _sha(path),
                    "bytes": path.stat().st_size,
                }
            )

    _write_json(final / "final_scene_registry.yaml", {"scenes": scenes})
    _write_json(final / "final_station_registry.yaml", {"stations": stations})
    _write_csv(final / "final_raw_bag_manifest.csv", raw)
    _write_csv(final / "final_target_manifest.csv", targets)
    _write_csv(final / "final_snapshot_manifest.csv", snapshots)
    _write_csv(
        final / "final_geometry_manifest.csv",
        geometry,
        list(verify.GEOMETRY_METADATA_FIELDS) + list(verify.GEOMETRY_FIELDS),
    )
    _write_csv(final / "rejected_candidate_manifest.csv", rejected)
    lineage = {
        "rejected_scene_id": "FMB1_W02",
        "replacement_scene_id": "FMB1_W04",
        "rejection_reason": "GEOMETRY_ONLY_INELIGIBLE",
        "decision_before_any_icp": True,
        "w04_admission_before_any_icp": True,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    }
    _write_json(final / "replacement_lineage.json", lineage)
    readiness = {
        "FMB1_FINAL_DATASET_READY": True,
        "READY_FOR_ZERO_PERTURBATION_AMENDMENT_ACTIVATION": True,
        "FINAL_RICH_SCENE_COUNT": 3,
        "FINAL_WEAK_SCENE_COUNT": 3,
        "FINAL_SCENE_COUNT": 6,
        "FINAL_STATION_COUNT": 18,
        "FINAL_SNAPSHOT_COUNT": 180,
        "RAW_CANDIDATE_BAG_COUNT": 42,
        "FINAL_ADMITTED_BAG_COUNT": 36,
        "REJECTED_BAG_COUNT": 6,
        "W02_RETAINED": True,
        "W02_INCLUDED_IN_FINAL_SET": False,
        "FMB1_W04_ACQUISITION_PASS": True,
        "FMB1_W04_FINAL_GEOMETRY_CLASS": "WEAK",
        "FMB1_W04_ADMISSION_PASS": True,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "backend_parameter_contract_sha256": verify.BACKEND_CONTRACT_SHA256,
    }
    _write_json(final / "final_dataset_readiness.json", readiness)
    no_icp = {
        "NO_ICP_ATTESTATION_PASS": True,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "open3d_registration_call_count": 0,
        "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0,
        "formal_trial_count": 0,
        "backend_parameter_contract_sha256": verify.BACKEND_CONTRACT_SHA256,
    }
    _write_json(final / "NO_ICP_ATTESTATION.json", no_icp)
    geometry_scenes = [
        {
            "scene_id": scene,
            "final_geometry_class": "RICH" if scene in verify.RICH_SCENES else "WEAK",
            "median_normalized_lambda_min_trans": 0.25 if scene in verify.RICH_SCENES else 0.05,
            "median_condition_number_trans": 1.8 if scene in verify.RICH_SCENES else 16.0,
            "median_spectral_entropy_trans": 0.96 if scene in verify.RICH_SCENES else 0.60,
        }
        for scene in verify.FINAL_SCENES
    ]
    _write_json(
        final / "final_dataset_manifest.json",
        {
            "schema": "test_mid360_fmb1_final_dataset_v1",
            "backend_parameter_contract_sha256": verify.BACKEND_CONTRACT_SHA256,
            "raw_candidate_bags": raw + rejected,
            "final_scene_registry": {"scenes": scenes},
            "final_station_registry": {"stations": stations},
            "final_raw_bags": raw,
            "final_targets": targets,
            "final_snapshots": snapshots,
            "final_geometry_metrics": geometry,
            "final_geometry_scenes": geometry_scenes,
            "rejected_candidates": rejected,
            "replacement_lineage": lineage,
            "no_icp_attestation": no_icp,
            "readiness": readiness,
        },
    )
    _refresh_checksums(final)
    return repository, final


def test_valid_bundle_passes_with_actual_file_rehash(valid_bundle: tuple[Path, Path]) -> None:
    report = verify.verify_w04_final_dataset(*valid_bundle)
    assert report["pass"] is True
    assert report["checks"]["w04_snapshot_count"] == 30
    assert report["checks"]["w02_snapshot_count"] == 0
    assert report["registration_backend_imports_or_calls"] == 0


def test_cli_writes_formal_independent_report(valid_bundle: tuple[Path, Path]) -> None:
    repository, final = valid_bundle
    output = final / "independent_verification.json"
    exit_code = verify_cli.main(
        [
            "--repository-root",
            str(repository),
            "--final-dir",
            str(final),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["pass"] is True
    assert report["formal_qualification"] is True
    assert report["verify_files"] is True
    assert report["verify_checksums"] is True


def _mutate_csv(final: Path, name: str, mutate: object) -> None:
    path = final / name
    rows = _read_csv(path)
    fields = list(rows[0])
    mutate(rows, fields)  # type: ignore[operator]
    _write_csv(path, rows, fields)


@pytest.mark.parametrize(
    "case",
    [
        "query_contribution",
        "query_input_role",
        "map_sha_binding",
        "missing_w04_snapshot",
        "duplicate_snapshot",
        "w02_snapshot",
        "geometry_t_est",
        "missing_geometry",
        "w04_rich_geometry",
        "w02_reason",
        "missing_w04_raw",
        "w04_mapping",
        "formal_lock",
        "formal_authority",
        "formal_trial",
        "lineage_after_icp",
        "readiness_count",
        "proposal_active",
        "source_bytes",
        "target_bytes",
        "raw_bytes",
    ],
)
def test_tampering_fails_closed(valid_bundle: tuple[Path, Path], case: str) -> None:
    repository, final = valid_bundle
    refresh = True
    if case == "query_contribution":
        _mutate_csv(final, "final_target_manifest.csv", lambda rows, _: rows[0].update(query_contribution_to_target="1"))
    elif case == "query_input_role":
        _mutate_csv(final, "final_target_manifest.csv", lambda rows, _: rows[0].update(input_roles="MAP;QUERY"))
    elif case == "map_sha_binding":
        _mutate_csv(final, "final_target_manifest.csv", lambda rows, _: rows[0].update(map_bag_sha256="0" * 64))
    elif case == "missing_w04_snapshot":
        _mutate_csv(final, "final_snapshot_manifest.csv", lambda rows, _: rows.pop())
    elif case == "duplicate_snapshot":
        def duplicate(rows: list[dict[str, str]], _: list[str]) -> None:
            rows[-1] = dict(rows[-2])
        _mutate_csv(final, "final_snapshot_manifest.csv", duplicate)
    elif case == "w02_snapshot":
        _mutate_csv(final, "final_snapshot_manifest.csv", lambda rows, _: rows[-1].update(scene_id="FMB1_W02"))
    elif case == "geometry_t_est":
        def add_t_est(rows: list[dict[str, str]], fields: list[str]) -> None:
            fields.append("T_est")
            for row in rows:
                row["T_est"] = "IDENTITY"
        _mutate_csv(final, "final_geometry_manifest.csv", add_t_est)
    elif case == "missing_geometry":
        _mutate_csv(final, "final_geometry_manifest.csv", lambda rows, _: rows.pop())
    elif case == "w04_rich_geometry":
        def rich(rows: list[dict[str, str]], _: list[str]) -> None:
            for row in rows:
                if row["scene_id"] == "FMB1_W04":
                    row.update(normalized_lambda_min_trans="0.25", condition_number_trans="1.8", spectral_entropy_trans="0.96")
        _mutate_csv(final, "final_geometry_manifest.csv", rich)
    elif case == "w02_reason":
        _mutate_csv(final, "rejected_candidate_manifest.csv", lambda rows, _: rows[0].update(exclusion_reason="MANUAL_REJECTION"))
    elif case == "missing_w04_raw":
        _mutate_csv(final, "final_raw_bag_manifest.csv", lambda rows, _: rows.pop())
    elif case == "w04_mapping":
        _mutate_csv(final, "final_raw_bag_manifest.csv", lambda rows, _: rows[-1].update(raw_filename="mid360_20260820_000000_part2_15s.bag"))
    elif case == "formal_lock":
        _write_json(final / "formal_batch1_lock.json", {"FORMAL_LOCK_ISSUED": True})
    elif case == "formal_authority":
        payload = json.loads((final / "final_dataset_readiness.json").read_text())
        payload["FORMAL_REGISTRATION_AUTHORIZED"] = True
        _write_json(final / "final_dataset_readiness.json", payload)
    elif case == "formal_trial":
        payload = json.loads((final / "NO_ICP_ATTESTATION.json").read_text())
        payload["formal_trial_count"] = 1
        _write_json(final / "NO_ICP_ATTESTATION.json", payload)
    elif case == "lineage_after_icp":
        payload = json.loads((final / "replacement_lineage.json").read_text())
        payload["w04_admission_before_any_icp"] = False
        _write_json(final / "replacement_lineage.json", payload)
    elif case == "readiness_count":
        payload = json.loads((final / "final_dataset_readiness.json").read_text())
        payload["FINAL_SNAPSHOT_COUNT"] = 179
        _write_json(final / "final_dataset_readiness.json", payload)
    elif case == "proposal_active":
        proposal = repository / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.json"
        _write_json(proposal, {"status": "ACTIVE", "FORMAL_AUTHORITY": True})
    elif case == "source_bytes":
        row = _read_csv(final / "final_snapshot_manifest.csv")[0]
        Path(row["source_path"]).write_bytes(b"tampered source")
    elif case == "target_bytes":
        row = _read_csv(final / "final_target_manifest.csv")[0]
        Path(row["target_path"]).write_bytes(b"tampered target")
    elif case == "raw_bytes":
        row = _read_csv(final / "final_raw_bag_manifest.csv")[0]
        Path(row["raw_absolute_path"]).write_bytes(b"tampered raw")
    else:  # pragma: no cover
        raise AssertionError(case)
    if refresh:
        _refresh_checksums(final)
    with pytest.raises(verify.W04FinalVerificationError):
        verify.verify_w04_final_dataset(repository, final)


def test_bad_sha256sums_fails_closed(valid_bundle: tuple[Path, Path]) -> None:
    repository, final = valid_bundle
    lines = (final / "SHA256SUMS").read_text().splitlines()
    lines[0] = "0" * 64 + lines[0][64:]
    (final / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    with pytest.raises(verify.W04FinalVerificationError):
        verify.verify_w04_final_dataset(repository, final)


def test_integral_float_csv_text_is_accepted_but_fraction_is_rejected() -> None:
    assert verify._int("16007.0", "correspondence", 1) == 16007
    assert verify._int("16007.000", "correspondence", 1) == 16007
    with pytest.raises(verify.W04FinalVerificationError):
        verify._int("16007.5", "correspondence", 1)
