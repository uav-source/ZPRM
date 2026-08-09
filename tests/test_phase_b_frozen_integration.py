from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from phase_a_harness.asset_verifier import source_runtime_import_paths
from phase_a_harness.contracts import file_sha256, load_manifest
from phase_a_harness.phase_b_snapshot_assets import (
    read_phase_b_plans,
    read_phase_b_snapshot,
    validate_phase_b_snapshot_lock,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frozen_assets/phase_b_signal_manifest.json"


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_phase_b_generator_export_and_seven_anchor_reproduction() -> None:
    with (ROOT / "frozen_assets/phase_b_generator_export_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        exports = list(csv.DictReader(stream))
    assert len(exports) == 18
    assert all(row["copied_exactly"] == "true" for row in exports)
    assert all(row["source_sha256"] == row["destination_sha256"] for row in exports)
    assert all(
        file_sha256(ROOT / row["destination_path"]) == row["destination_sha256"]
        for row in exports
    )
    report = _json("artifacts/phase_b_generator_reproduction.json")
    assert report["GENERATOR_REPRODUCTION_PASS"] is True
    assert report["reproduction_snapshot_count"] == 7
    assert report["GENERATOR_REPRODUCTION_CHECKSUM_MISMATCH_COUNT"] == 0
    assert all(row["mismatch_count"] == 0 for row in report["rows"])
    assert report["firewall_totals"]["CONFIRMATORY_SEED_INSTANTIATION_COUNT"] == 0
    assert report["firewall_totals"]["OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT"] == 0


def test_phase_b_protocol_and_single_manifest_are_exact() -> None:
    protocol = _json("frozen_assets/phase_b_signal_protocol.json")
    _, manifest = load_manifest(MANIFEST, require_authorized=False)
    assert list((ROOT / "frozen_assets").glob("phase_b_signal_manifest*.json")) == [MANIFEST]
    assert set(protocol["conditions"]) == {"INDEPENDENT_NOISE_FREE", "FULL_NOISE"}
    assert len(protocol["scenes"]) == 7
    assert protocol["geometry_seeds"] == [1850310744, 1957656152, 1334931069]
    assert protocol["measurement_seed"] == 217775206 and protocol["repeat_index"] == 0
    assert protocol["planned_snapshot_count"] == 42 and protocol["planned_trial_count"] == 84
    assert protocol["quantile_method"] == "linear"
    assert np.quantile(np.arange(20.0), 0.95, method="linear") == 18.05
    assert protocol["conditions"]["INDEPENDENT_NOISE_FREE"] == {
        "independent_sampling": True,
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.0,
        "scan_dropout_fraction": 0.0,
        "scan_noise_sigma_m": 0.0,
    }
    assert protocol["conditions"]["FULL_NOISE"] == {
        "independent_sampling": True,
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.001,
        "scan_dropout_fraction": 0.01,
        "scan_noise_sigma_m": 0.003,
    }
    assert manifest["phase_a_formal_pass_commit"] == "50fd4415deb6af3f3e07d88db5a2f21227d05f69"
    assert manifest["trial_schema_sha256"] == "4d1b9da79d11b36b19350bec1c0b9eb84fc64a3699be682a4eecca684b451773"
    assert manifest["native_trial_count"] == 0
    assert type(manifest["formal_execution_authorized"]) is bool
    assert manifest["confirmatory_authorized"] is False
    assert manifest["real_data_authorized"] is False


def test_phase_b_actual_snapshot_lock_inventory_and_pairing() -> None:
    snapshots, trials = read_phase_b_plans(
        ROOT / "frozen_assets/phase_b_planned_snapshots.csv",
        ROOT / "frozen_assets/phase_b_planned_trials.csv",
    )
    lock = validate_phase_b_snapshot_lock(
        ROOT / "frozen_assets/phase_b_snapshot_lock.json",
        ROOT / "data/phase_b_signal_snapshots",
        snapshots,
    )
    assert len(lock["snapshots"]) == len(snapshots) == 42
    assert len(trials) == 84
    assert Counter(row["condition"] for row in snapshots) == Counter(
        {"INDEPENDENT_NOISE_FREE": 21, "FULL_NOISE": 21}
    )
    assert set(Counter(row["scene_variant"] for row in snapshots).values()) == {6}
    assert set(Counter(row["snapshot_id"] for row in trials).values()) == {2}
    assert not any("native" in row["backend"].lower() for row in trials)
    for row in snapshots:
        item = read_phase_b_snapshot(
            ROOT / "data/phase_b_signal_snapshots", row["snapshot_id"], arrays=True
        )
        assert item["metadata"]["independent_sampling"] is True
        assert item["metadata"]["source_is_target_subset"] is False
        assert item["metadata"]["initial_pose"] == "reference_pose_exact"
        assert item["reference"].shape == (4, 4)


def test_phase_b_dry_run_and_runtime_isolation_are_zero_execution() -> None:
    dry = _json("artifacts/phase_b_dry_run_report.json")
    preparation = _json("artifacts/phase_b_snapshot_preparation_report.json")
    assert dry["FORMAL_DRY_RUN_PASS"] is True
    assert (
        dry["planned_snapshot_count"],
        dry["planned_trial_count"],
        dry["open3d_trial_count"],
        dry["pcl_trial_count"],
        dry["native_trial_count"],
    ) == (42, 84, 42, 42, 0)
    assert dry["condition_trial_counts"] == {
        "FULL_NOISE": 42,
        "INDEPENDENT_NOISE_FREE": 42,
    }
    assert (
        dry["formal_rng_access_count"],
        dry["backend_execution_count"],
        dry["trial_result_count"],
        dry["started_event_count"],
    ) == (0, 0, 0, 0)
    assert dry["source_repository_runtime_file_read_count"] == 0
    assert dry["source_repository_runtime_import_count"] == 0
    assert preparation["source_repository_runtime_file_read_count"] == 0
    assert preparation["source_repository_runtime_import_count"] == 0
    assert preparation["confirmatory_seed_instantiation_count"] == 0
    assert source_runtime_import_paths() == []
