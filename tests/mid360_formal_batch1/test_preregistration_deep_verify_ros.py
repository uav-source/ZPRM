from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from experiments.mid360_formal_batch1 import preregistration_deep_verify_ros as deep


def _minimal_acquisition(repository: Path) -> dict:
    return {
        "schema": "mid360_formal_batch1_acquisition_ingest_v1",
        "repository": str(repository),
        "bags_dir": str(repository),
        "NO_FORMAL_REGISTRATION": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "raw_bags": [],
        "mapping": [],
        "stations": [],
        "inventory_gate": {"status": "PASS"},
        "bag_audit_pass_count": 0,
        "station_acquisition_pass_count": 0,
        "station_count": 0,
        "status": "PASS",
    }


def test_independent_acquisition_rerun_tamper_fails_before_ros_assets(
    monkeypatch, tmp_path: Path
) -> None:
    frozen = _minimal_acquisition(tmp_path)
    rerun = copy.deepcopy(frozen)
    rerun["inventory_gate"]["status"] = "FAIL"
    calls = []

    def fake_run(repository, bags_dir, config):
        calls.append((repository, bags_dir, config))
        return rerun

    monkeypatch.setattr(deep, "run_acquisition", fake_run)
    result = deep.verify_ros_evidence(frozen, {"targets": [], "snapshots": []}, {})
    assert len(calls) == 1
    assert result["status"] == "FAIL"
    assert result["checks"]["acquisition_rerun_exact"] is False
    assert any(
        failure.get("path") == "inventory_gate.status"
        for failure in result["failures"]
    )


def test_array_sha_is_canonical_and_array_equality_is_exact(tmp_path: Path) -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]], dtype=">f8")
    canonical = np.ascontiguousarray(points, dtype="<f8")
    expected_sha = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    assert deep._array_sha256(points) == expected_sha

    path = tmp_path / "source.npy"
    np.save(path, canonical, allow_pickle=False)
    row = {
        "source_path": str(path),
        "source_npy_sha256": deep.sha256_file(path),
        "source_array_sha256": expected_sha,
        "source_point_count": 2,
        "source_shape": [2, 3],
        "source_dtype": "float64",
    }
    checks, failures = deep._verify_point_array(canonical.copy(), row, "source", tmp_path)
    assert not failures
    assert all(checks.values())

    changed = canonical.copy()
    changed[1, 2] += 1.0
    changed_checks, changed_failures = deep._verify_point_array(
        changed, row, "source", tmp_path
    )
    assert changed_checks["source_np_array_equal"] is False
    assert changed_failures
