from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_a_harness.contracts import canonical_json_sha256, file_sha256
from phase_a_harness.phase_b_generator import GENERATOR_SHA256, SNAPSHOT_BUILDER_SHA256
from phase_a_harness.phase_b_snapshot_assets import (
    materialize_or_verify_phase_b_snapshots,
    planned_phase_b_snapshots,
    planned_phase_b_trials,
    read_phase_b_snapshot,
    validate_phase_b_plans,
)
from phase_a_harness import phase_b_runner


def _raw_sha(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _write_valid_snapshot(cache: Path) -> tuple[str, Path]:
    snapshot_id = "phase-b-signal-v1/GEOMETRY_RICH_ROOM/g0/INDEPENDENT_NOISE_FREE"
    directory = cache / snapshot_id
    directory.mkdir(parents=True)
    source = np.ascontiguousarray([[0.0, 0.0, 0.0], [0.2, 0.1, -0.1]], dtype="<f4")
    target = np.ascontiguousarray([[5.0, 0.0, 0.0], [5.2, 0.1, -0.1]], dtype="<f4")
    reference = np.eye(4, dtype="<f8", order="C")
    checksums = {
        "source_checksum": _raw_sha(source),
        "target_checksum": _raw_sha(target),
        "reference_pose_checksum": _raw_sha(reference),
    }
    np.save(directory / "source_points.npy", source, allow_pickle=False)
    np.save(directory / "target_points.npy", target, allow_pickle=False)
    np.save(directory / "reference_pose.npy", reference, allow_pickle=False)
    metadata = {
        "array_file_sha256": {
            name: file_sha256(directory / name)
            for name in ("reference_pose.npy", "source_points.npy", "target_points.npy")
        },
        "condition": "INDEPENDENT_NOISE_FREE",
        "dropout_parameters": {
            "map_dropout_fraction": 0.0,
            "scan_dropout_fraction": 0.0,
        },
        "generator_sha256": GENERATOR_SHA256,
        "generator_firewall_audit": {
            "CONFIRMATORY_SEED_INSTANTIATION_COUNT": 0,
            "GT_OPTIMIZATION_LEAKAGE_COUNT": 0,
            "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT": 0,
            "RNG_CONSTRUCTION_COUNT": 8,
            "SNAPSHOT_ACCESS_COUNT": 8,
        },
        "geometry_seed": 1850310744,
        "geometry_seed_index": 0,
        "independent_sampling": True,
        "initial_pose": "reference_pose_exact",
        "measurement_seed": 217775206,
        "noise_parameters": {
            "map_noise_sigma_m": 0.0,
            "scan_noise_sigma_m": 0.0,
        },
        "reference_pose_checksum": checksums["reference_pose_checksum"],
        "repeat_index": 0,
        "scene_variant": "GEOMETRY_RICH_ROOM",
        "snapshot_checksum": canonical_json_sha256(
            {"snapshot_id": snapshot_id, **checksums}
        ),
        "snapshot_builder_sha256": SNAPSHOT_BUILDER_SHA256,
        "snapshot_id": snapshot_id,
        "source_checksum": checksums["source_checksum"],
        "source_is_target_subset": False,
        "source_point_count": len(source),
        "target_checksum": checksums["target_checksum"],
        "target_point_count": len(target),
    }
    metadata["metadata_payload_sha256"] = canonical_json_sha256(metadata)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return snapshot_id, directory


def test_phase_b_plan_is_exactly_42_snapshots_and_84_paired_trials() -> None:
    snapshots = planned_phase_b_snapshots()
    trials = planned_phase_b_trials(snapshots)
    report = validate_phase_b_plans(snapshots, trials)
    assert report["planned_snapshot_count"] == 42
    assert report["planned_trial_count"] == 84
    assert report["condition_trial_counts"] == {
        "FULL_NOISE": 42,
        "INDEPENDENT_NOISE_FREE": 42,
    }
    assert report["native_trial_count"] == 0
    assert report["snapshot_backend_pairing_mismatch_count"] == 0


def test_phase_b_plan_rejects_backend_or_order_change() -> None:
    snapshots = planned_phase_b_snapshots()
    trials = planned_phase_b_trials(snapshots)
    altered = [dict(row) for row in trials]
    altered[0]["backend"] = "native_full"
    with pytest.raises(ValueError, match="planned trial"):
        validate_phase_b_plans(snapshots, altered)
    with pytest.raises(ValueError, match="snapshot rows/order"):
        validate_phase_b_plans(list(reversed(snapshots)), trials)


def test_phase_b_snapshot_reader_checks_dtype_raw_and_metadata(tmp_path: Path) -> None:
    snapshot_id, _ = _write_valid_snapshot(tmp_path)
    result = read_phase_b_snapshot(tmp_path, snapshot_id, arrays=True)
    assert result["source"].dtype == np.dtype("<f4")
    assert result["target"].flags.c_contiguous
    assert result["reference"].dtype == np.dtype("<f8")
    assert result["metadata"]["metadata_payload_sha256"]
    assert result["snapshot_checksum"] == result["metadata"]["snapshot_checksum"]


def test_phase_b_snapshot_reader_rejects_changed_array_bytes(tmp_path: Path) -> None:
    snapshot_id, directory = _write_valid_snapshot(tmp_path)
    changed = np.ascontiguousarray([[9.0, 9.0, 9.0]], dtype="<f4")
    np.save(directory / "source_points.npy", changed, allow_pickle=False)
    with pytest.raises(ValueError, match="array-file SHA|point count|raw array checksum"):
        read_phase_b_snapshot(tmp_path, snapshot_id, arrays=True)


def test_nonempty_partial_cache_is_never_extended_or_overwritten(tmp_path: Path) -> None:
    cache = tmp_path / "data/phase_b_signal_snapshots/rogue"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty Phase B cache"):
        materialize_or_verify_phase_b_snapshots(
            tmp_path, planned_phase_b_snapshots()
        )
    assert (cache / "metadata.json").read_text(encoding="utf-8") == "{}\n"


def test_phase_b_dry_run_reports_zero_execution_and_exact_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = planned_phase_b_snapshots()
    trials = planned_phase_b_trials(snapshots)
    stack = {
        "cache_root": tmp_path / "cache",
        "lock_by_id": {row["snapshot_id"]: {} for row in snapshots},
        "manifest": {
            "formal_execution_authorized": False,
            "formal_output_dir": "results/phase_b_signal_v1",
            "formal_workers": 2,
        },
        "root": tmp_path,
        "snapshots": snapshots,
        "trials": trials,
    }
    monkeypatch.setattr(phase_b_runner, "load_phase_b_stack", lambda *args, **kwargs: stack)
    monkeypatch.setattr(phase_b_runner, "read_phase_b_snapshot", lambda *args, **kwargs: {})
    monkeypatch.setattr(phase_b_runner, "source_runtime_import_paths", lambda: [])
    report = phase_b_runner.dry_run_phase_b(
        manifest_path=tmp_path / "frozen_assets/phase_b_signal_manifest.json",
        run_id="phase-b-signal-v1",
        output_dir=tmp_path / "results/phase_b_signal_v1",
        workers=2,
    )
    assert report["FORMAL_DRY_RUN_PASS"] is True
    assert report["planned_snapshot_count"] == 42
    assert report["planned_trial_count"] == 84
    assert report["condition_trial_counts"] == {
        "FULL_NOISE": 42,
        "INDEPENDENT_NOISE_FREE": 42,
    }
    assert report["backend_execution_count"] == 0
    assert report["formal_rng_access_count"] == 0
    assert report["started_event_count"] == 0
    assert report["trial_result_count"] == 0
