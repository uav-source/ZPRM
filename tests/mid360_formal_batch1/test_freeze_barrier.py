from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.protocol import (
    BACKEND_CONTRACT_SHA256,
    FormalBatchError,
    directory_manifest,
    formal_icp_authorized,
    freeze_batch,
    require_formal_icp_authorization,
    sha256_file,
    verify_pilot_anchors,
)


REAL_REPOSITORY = Path(__file__).resolve().parents[2]
SHA = "a" * 64


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def create_valid_fixture(root: Path) -> tuple[Path, Path]:
    experiment = root / "experiments/mid360_formal_batch1"
    results = root / "results/mid360_formal_batch1"
    frozen = root / "frozen_assets"
    experiment.mkdir(parents=True)
    results.mkdir(parents=True)
    frozen.mkdir(parents=True)
    shutil.copy2(
        REAL_REPOSITORY / "experiments/mid360_formal_batch1/preregistration.yaml",
        experiment / "preregistration.yaml",
    )
    shutil.copy2(
        REAL_REPOSITORY / "frozen_assets/backend_parameter_contract.json",
        frozen / "backend_parameter_contract.json",
    )
    assert sha256_file(frozen / "backend_parameter_contract.json") == BACKEND_CONTRACT_SHA256

    scenes = []
    geometry_scenes = []
    stations = []
    selected_pairs = []
    targets = []
    snapshots = []
    station_geometry = []
    snapshot_geometry = []
    for prefix, classification in (("R", "RICH"), ("W", "WEAK")):
        metric_values = (0.20, 2.0, 0.95) if classification == "RICH" else (0.10, 7.0, 0.70)
        for scene_index in range(1, 4):
            scene_id = f"FMB1_{prefix}{scene_index:02d}"
            scenes.append(
                {
                    "scene_id": scene_id,
                    "semantic_candidate_label": f"{classification}_CANDIDATE",
                    "final_geometry_class": classification,
                    "candidate_status": "ADMITTED",
                    "exclusion_reason": None,
                    "replacement_scene_id": None,
                }
            )
            geometry_scenes.append(
                {
                    "scene_id": scene_id,
                    "final_geometry_class": classification,
                    "admitted": True,
                    "geometry_lambda_min_median": metric_values[0],
                    "geometry_condition_median": metric_values[1],
                    "geometry_entropy_median": metric_values[2],
                }
            )
            for station_index in range(1, 4):
                station_id = f"S{station_index:02d}"
                stations.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "planned_order": station_index,
                    }
                )
                selected_pairs.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "map_audit": {"SHA256": SHA},
                        "query_audit": {"SHA256": SHA},
                        "pair_audit": {"FORMAL_PAIR_VALID": True},
                    }
                )
                targets.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "target_npy_sha256": SHA,
                        "target_point_count": 100,
                        "registration_called": False,
                    }
                )
                station_geometry.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "snapshot_count": 10,
                        "geometry_only": True,
                        "registration_executed": False,
                    }
                )
                for snapshot_index in range(1, 11):
                    snapshot_id = f"{scene_id}_{station_id}_Q{snapshot_index:02d}"
                    snapshots.append(
                        {
                            "scene_id": scene_id,
                            "station_id": station_id,
                            "snapshot_id": snapshot_id,
                            "source_npy_sha256": SHA,
                        }
                    )
                    snapshot_geometry.append(
                        {
                            "scene_id": scene_id,
                            "station_id": station_id,
                            "snapshot_id": snapshot_id,
                            "normalized_lambda_min_trans": metric_values[0],
                            "condition_number_trans": metric_values[1],
                            "spectral_entropy_trans": metric_values[2],
                        }
                    )
    dump(experiment / "scene_registry.yaml", {"scenes": scenes})
    dump(experiment / "station_registry.yaml", {"stations": stations})
    dump(results / "bag_manifest.json", {"selected_pairs": selected_pairs})
    dump(results / "target_manifest.json", {"targets": targets})
    dump(
        results / "snapshot_manifest.json",
        {
            "snapshots": snapshots,
            "backend_invocation_count_at_freeze": 0,
            "selection_frozen_before_icp": True,
        },
    )
    dump(
        results / "geometry_manifest.json",
        {
            "labels_frozen": True,
            "registration_executed": False,
            "scenes": geometry_scenes,
            "station_summaries": station_geometry,
            "snapshot_metrics": snapshot_geometry,
        },
    )
    dump(
        results / "preregistration_anchor.json",
        {"PREREGISTRATION_SHA256": sha256_file(experiment / "preregistration.yaml")},
    )

    pilot_anchors = []
    for name in ("mid360_controlled_perturbation_pilot", "mid360_capture_basin_pilot"):
        relative = f"results/{name}"
        pilot = root / relative
        pilot.mkdir(parents=True)
        (pilot / "retained.txt").write_text(f"{name}\n", encoding="utf-8")
        anchor = directory_manifest(pilot)
        anchor["relative_path"] = relative
        pilot_anchors.append(anchor)
    dump(results / "pilot_freeze_anchors.json", {"pilot_directories": pilot_anchors})
    return root, results


@pytest.fixture
def valid_fixture(tmp_path: Path) -> tuple[Path, Path]:
    return create_valid_fixture(tmp_path)


def test_formal_icp_is_blocked_before_freeze(valid_fixture: tuple[Path, Path]) -> None:
    repository, results = valid_fixture
    assert formal_icp_authorized(repository, results) is False
    with pytest.raises(FormalBatchError, match="BLOCKED"):
        require_formal_icp_authorization(repository, results)


def test_valid_lock_requires_three_rich_three_weak_and_allows_future_runner(
    valid_fixture: tuple[Path, Path],
) -> None:
    repository, results = valid_fixture
    fingerprint = freeze_batch(repository, results)
    assert len(fingerprint["PREREGISTRATION_SHA256"]) == 64
    assert len(fingerprint["FORMAL_BATCH1_LOCK_SHA256"]) == 64
    assert formal_icp_authorized(repository, results) is True
    lock = load(results / "formal_batch1_lock.json")
    assert lock["admitted_class_counts"] == {"RICH": 3, "WEAK": 3}
    assert lock["valid_station_count"] == 18
    assert lock["snapshot_count"] == 180


def test_freeze_rejects_not_exactly_three_rich_and_three_weak(
    valid_fixture: tuple[Path, Path],
) -> None:
    repository, results = valid_fixture
    manifest = load(results / "geometry_manifest.json")
    manifest["scenes"][0].update(final_geometry_class="INTERMEDIATE", admitted=False)
    dump(results / "geometry_manifest.json", manifest)
    with pytest.raises(FormalBatchError, match="exactly 3"):
        freeze_batch(repository, results)


def test_freeze_rejects_missing_third_station(valid_fixture: tuple[Path, Path]) -> None:
    repository, results = valid_fixture
    manifest = load(results / "bag_manifest.json")
    manifest["selected_pairs"].pop()
    dump(results / "bag_manifest.json", manifest)
    with pytest.raises(FormalBatchError, match="three valid selected pairs"):
        freeze_batch(repository, results)


def test_freeze_rejects_missing_pair_sha(valid_fixture: tuple[Path, Path]) -> None:
    repository, results = valid_fixture
    manifest = load(results / "bag_manifest.json")
    manifest["selected_pairs"][0]["map_audit"]["SHA256"] = None
    dump(results / "bag_manifest.json", manifest)
    with pytest.raises(FormalBatchError, match="pair SHA"):
        freeze_batch(repository, results)


def test_freeze_rejects_snapshot_after_backend_invocation(
    valid_fixture: tuple[Path, Path],
) -> None:
    repository, results = valid_fixture
    manifest = load(results / "snapshot_manifest.json")
    manifest["backend_invocation_count_at_freeze"] = 1
    dump(results / "snapshot_manifest.json", manifest)
    with pytest.raises(FormalBatchError, match="before ICP"):
        freeze_batch(repository, results)


def test_preregistration_mutation_breaks_anchor_and_existing_lock(
    valid_fixture: tuple[Path, Path],
) -> None:
    repository, results = valid_fixture
    freeze_batch(repository, results)
    prereg = repository / "experiments/mid360_formal_batch1/preregistration.yaml"
    prereg.write_text(prereg.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert formal_icp_authorized(repository, results) is False


def test_backend_contract_mismatch_fails_fast(valid_fixture: tuple[Path, Path]) -> None:
    repository, results = valid_fixture
    contract = repository / "frozen_assets/backend_parameter_contract.json"
    contract.write_text(contract.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FormalBatchError, match="backend contract mismatch"):
        freeze_batch(repository, results)


def test_real_pilot_directories_match_full_byte_anchors() -> None:
    verify_pilot_anchors(
        REAL_REPOSITORY,
        REAL_REPOSITORY / "results/mid360_formal_batch1/pilot_freeze_anchors.json",
    )


def test_pilot_directory_mutation_is_rejected(valid_fixture: tuple[Path, Path]) -> None:
    repository, results = valid_fixture
    pilot = repository / "results/mid360_capture_basin_pilot/retained.txt"
    pilot.write_text("changed\n", encoding="utf-8")
    with pytest.raises(FormalBatchError, match="Pilot directory changed"):
        freeze_batch(repository, results)


def test_lock_and_fingerprint_are_never_overwritten(
    valid_fixture: tuple[Path, Path],
) -> None:
    repository, results = valid_fixture
    freeze_batch(repository, results)
    original = (results / "formal_batch1_lock.json").read_bytes()
    with pytest.raises(FormalBatchError, match="overwrite"):
        freeze_batch(repository, results)
    assert (results / "formal_batch1_lock.json").read_bytes() == original
