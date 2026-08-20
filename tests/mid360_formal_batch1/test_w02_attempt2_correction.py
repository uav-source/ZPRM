from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.w02_attempt2 import (
    as_w02_attempt2,
    as_w04_compatibility,
)
from experiments.mid360_formal_batch1.w02_attempt2_final_dataset import (
    ATTEMPT1_PREFIXES,
    W02CorrectionError,
    archive_w02_attempt1,
    verify_corrected_final_dataset,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_attempt_relabel_preserves_paths_and_changes_exact_identifiers() -> None:
    payload = {
        "scene_id": "FMB1_W04",
        "snapshot_id": "FMB1_W04_S01_Q01",
        "target_path": "/runtime/FMB1_W04/S01/target.npy",
        "rows": [{"scene_id": "FMB1_W04"}],
    }
    corrected = as_w02_attempt2(payload)
    assert corrected["scene_id"] == "FMB1_W02"
    assert corrected["snapshot_id"] == "FMB1_W02_S01_Q01"
    assert corrected["target_path"] == payload["target_path"]
    assert as_w04_compatibility(corrected)["scene_id"] == "FMB1_W04"


def _archive_fixture(root: Path) -> tuple[dict[str, object], Path]:
    bags = root / "bags"
    bags.mkdir(parents=True)
    rows = []
    for station, prefix in ATTEMPT1_PREFIXES.items():
        for role, part in (("MAP", "part1_20s"), ("QUERY", "part2_15s")):
            name = f"mid360_{prefix}_{part}.bag"
            path = bags / name
            path.write_bytes(f"{station}-{role}".encode())
            rows.append(
                {
                    "scene_id": "FMB1_W02",
                    "station_id": station,
                    "role": role,
                    "raw_filename": name,
                    "raw_absolute_path": str(path),
                    "sha256": _sha(path),
                    "bytes": path.stat().st_size,
                }
            )
    archive = root / "archive/invalid_acquisition/FMB1_W02_attempt1_wrong_location"
    return {"raw_bags": rows}, archive


def test_attempt1_archive_moves_but_retains_all_six(tmp_path: Path) -> None:
    original, archive = _archive_fixture(tmp_path)
    manifest = archive_w02_attempt1(
        original, repository=tmp_path, archive_dir=archive
    )
    assert manifest["archived_bag_count"] == 6
    assert manifest["failed_raw_data_retained"] is True
    assert not any((tmp_path / "bags" / row["raw_filename"]).exists() for row in manifest["bags"])
    assert all(Path(row["archived_absolute_path"]).is_file() for row in manifest["bags"])
    # The operation is intentionally restart-safe after the move.
    repeated = archive_w02_attempt1(
        original, repository=tmp_path, archive_dir=archive
    )
    assert repeated["bags"] == manifest["bags"]


def test_attempt1_archive_fails_on_hash_change(tmp_path: Path) -> None:
    original, archive = _archive_fixture(tmp_path)
    original["raw_bags"][0]["sha256"] = "0" * 64
    with pytest.raises(W02CorrectionError, match="SHA changed"):
        archive_w02_attempt1(original, repository=tmp_path, archive_dir=archive)


def _corrected_fixture(repository: Path, data_file: Path) -> dict[str, object]:
    sha = _sha(data_file)
    scenes = (
        "FMB1_R01", "FMB1_R02", "FMB1_R03", "FMB1_W01", "FMB1_W02", "FMB1_W03"
    )
    stations = ("S01", "S02", "S03")
    candidate = []
    final_raw = []
    targets = []
    snapshots = []
    metrics = []
    scene_rows = []
    station_rows = []
    for scene in scenes:
        attempt = 2 if scene == "FMB1_W02" else 1
        expected = "RICH" if scene.startswith("FMB1_R") else "WEAK"
        scene_rows.append(
            {
                "scene_id": scene,
                "attempt": attempt,
                "final_geometry_class": expected,
                "geometry_admission_status": "GEOMETRY_ADMITTED",
            }
        )
        for station in stations:
            station_rows.append(
                {"scene_id": scene, "station_id": station, "attempt": attempt}
            )
            for role in ("MAP", "QUERY"):
                row = {
                    "scene_id": scene,
                    "station_id": station,
                    "role": role,
                    "attempt": attempt,
                    "raw_filename": f"{scene}_{station}_{role}.bag",
                    "raw_absolute_path": str(data_file),
                    "sha256": sha,
                }
                final_raw.append(row)
                candidate.append(copy.deepcopy(row))
            targets.append(
                {
                    "scene_id": scene,
                    "station_id": station,
                    "attempt": attempt,
                    "map_bag_sha256": sha,
                    "input_roles": ["MAP"],
                    "query_contribution_to_target": 0,
                    "target_path": str(data_file),
                    "target_npy_sha256": sha,
                }
            )
            for index, quantile in enumerate(
                (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
            ):
                snapshots.append(
                    {
                        "scene_id": scene,
                        "station_id": station,
                        "attempt": attempt,
                        "selection_index": index,
                        "quantile": quantile,
                        "target_npy_sha256": sha,
                        "source_path": str(data_file),
                        "source_npy_sha256": sha,
                    }
                )
                metrics.append(
                    {
                        "scene_id": scene,
                        "station_id": station,
                        "attempt": attempt,
                        "selection_index": index,
                        "normalized_lambda_min_trans": 0.20 if expected == "RICH" else 0.10,
                        "condition_number_trans": 2.0 if expected == "RICH" else 7.0,
                        "spectral_entropy_trans": 0.95 if expected == "RICH" else 0.70,
                    }
                )
    for station in stations:
        for role in ("MAP", "QUERY"):
            candidate.append(
                {
                    "scene_id": "FMB1_W02",
                    "station_id": station,
                    "role": role,
                    "attempt": 1,
                    "raw_filename": f"old_{station}_{role}.bag",
                    "raw_absolute_path": str(data_file),
                    "sha256": sha,
                }
            )
    rejected = [
        {
            "scene_id": "FMB1_W02",
            "station_id": station,
            "role": role,
            "attempt": 1,
            "candidate_status": "INVALID_ACQUISITION",
            "retained": True,
            "included_in_formal_trials": False,
        }
        for station in stations
        for role in ("MAP", "QUERY")
    ]
    return {
        "schema": "mid360_fmb1_w02_attempt2_corrected_final_dataset_v1",
        "backend_parameter_contract_sha256": "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9",
        "raw_candidate_bags": candidate,
        "final_raw_bags": final_raw,
        "final_targets": targets,
        "final_snapshots": snapshots,
        "final_geometry_metrics": metrics,
        "final_geometry_scenes": scene_rows,
        "final_scene_registry": {"scenes": scene_rows},
        "final_station_registry": {"stations": station_rows},
        "rejected_candidates": rejected,
        "acquisition_attempt_lineage": {
            "correction_reason": "OPERATOR_CONFIRMED_WRONG_SCENE_LOCATION",
            "correction_before_any_icp": True,
            "formal_trial_count_at_correction": 0,
            "invalid_attempt": 1,
            "valid_attempt": 2,
            "failed_attempts_retained": True,
            "failed_raw_data_retained": True,
            "W04_IDENTIFIER_RETIRED": True,
            "W04_INCLUDED_IN_FINAL_SET": False,
        },
        "readiness": {
            "FMB1_FINAL_DATASET_READY": True,
            "W04_INCLUDED_IN_FINAL_SET": False,
            "FMB1_W02_ATTEMPT2_INCLUDED_IN_FINAL_SET": True,
            "FMB1_W02_ATTEMPT1_INCLUDED_IN_FINAL_SET": False,
            "FORMAL_LOCK_ISSUED": False,
            "FORMAL_ICP_UNLOCKED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
        },
        "no_icp_attestation": {
            "FORMAL_ICP_UNLOCKED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "formal_trial_count": 0,
        },
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("readiness", "FORMAL_REGISTRATION_AUTHORIZED"), True),
        (("readiness", "W04_INCLUDED_IN_FINAL_SET"), True),
        (("acquisition_attempt_lineage", "correction_before_any_icp"), False),
        (("final_targets", 0, "query_contribution_to_target"), 1),
        (("final_snapshots", 0, "quantile"), 0.07),
        (("final_geometry_metrics", 0, "selection_index"), 1),
        (("final_raw_bags", 0, "attempt"), 2),
    ],
)
def test_corrected_manifest_tampering_fails(
    tmp_path: Path, path: tuple[object, ...], value: object
) -> None:
    data_file = tmp_path / "evidence.bin"
    data_file.write_bytes(b"evidence")
    payload = _corrected_fixture(Path.cwd(), data_file)
    cursor: object = payload
    for component in path[:-1]:
        cursor = cursor[component]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    with pytest.raises(W02CorrectionError):
        verify_corrected_final_dataset(
            payload, repository=Path.cwd(), verify_files=True
        )


def test_corrected_fixture_passes(tmp_path: Path) -> None:
    data_file = tmp_path / "evidence.bin"
    data_file.write_bytes(b"evidence")
    result = verify_corrected_final_dataset(
        _corrected_fixture(Path.cwd(), data_file),
        repository=Path.cwd(),
        verify_files=True,
    )
    assert result["status"] == "PASS"
    assert result["W04_INCLUDED_IN_FINAL_SET"] is False
