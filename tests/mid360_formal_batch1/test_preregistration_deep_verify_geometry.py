from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from experiments.mid360_formal_batch1 import preregistration_deep_verify_geometry as deep
from experiments.mid360_formal_batch1.preregistration_deep_verify_geometry import (
    DeepGeometryVerificationError,
    verify_geometry_evidence,
)
from experiments.mid360_formal_batch1.protocol import (
    INITIAL_SCENE_IDS,
    STATION_IDS,
    geometry_admission,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    GEOMETRY_ONLY_FIELDS,
)


RICH = {
    "initial_correspondence_count": 1,
    "initial_valid_normal_correspondence_count": 1,
    "lambda_min_trans": 0.20,
    "lambda_mid_trans": 0.30,
    "lambda_max_trans": 0.50,
    "normalized_lambda_min_trans": 0.20,
    "normalized_lambda_mid_trans": 0.30,
    "normalized_lambda_max_trans": 0.50,
    "condition_number_trans": 2.5,
    "spectral_entropy_trans": 0.95,
}
WEAK = {
    "initial_correspondence_count": 1,
    "initial_valid_normal_correspondence_count": 1,
    "lambda_min_trans": 0.10,
    "lambda_mid_trans": 0.10,
    "lambda_max_trans": 0.80,
    "normalized_lambda_min_trans": 0.10,
    "normalized_lambda_mid_trans": 0.10,
    "normalized_lambda_max_trans": 0.80,
    "condition_number_trans": 8.0,
    "spectral_entropy_trans": 0.70,
}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def _metrics_for_scene(scene_id: str) -> dict[str, float | int]:
    # W02 intentionally computes RICH. It is a valid, consistently frozen
    # rejection, not a reason for this evidence verifier to force readiness.
    if scene_id.startswith("FMB1_R") or scene_id == "FMB1_W02":
        return dict(RICH)
    return dict(WEAK)


def _payloads(tmp_path: Path) -> tuple[dict, dict]:
    targets = []
    snapshots = []
    metric_rows = []
    station_summaries = []
    by_scene = defaultdict(list)
    for scene_index, scene_id in enumerate(INITIAL_SCENE_IDS):
        scene_metrics = _metrics_for_scene(scene_id)
        marker = 1.0 if scene_metrics == RICH else -1.0
        for station_index, station_id in enumerate(STATION_IDS):
            target = np.asarray([[marker, scene_index, station_index]], dtype="<f8")
            target_path = tmp_path / "targets" / scene_id / station_id / "target.npy"
            target_path.parent.mkdir(parents=True)
            np.save(target_path, target, allow_pickle=False)
            target_sha = _file_sha(target_path)
            targets.append(
                {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "target_path": str(target_path),
                    "target_npy_sha256": target_sha,
                    "target_array_sha256": _array_sha(target),
                    "target_point_count": 1,
                    "input_roles": ["MAP"],
                    "query_contribution_to_target": 0,
                    "registration_called": False,
                    "odometry_called": False,
                    "scan_matching_called": False,
                }
            )
            station_rows = []
            for selection_index in range(10):
                snapshot_id = f"{scene_id}_{station_id}_Q{selection_index + 1:02d}"
                source = np.asarray(
                    [[marker, selection_index, station_index]], dtype="<f8"
                )
                source_path = (
                    tmp_path / "sources" / scene_id / station_id / f"{snapshot_id}.npy"
                )
                source_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(source_path, source, allow_pickle=False)
                snapshots.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "snapshot_id": snapshot_id,
                        "selection_index": selection_index,
                        "source_path": str(source_path),
                        "source_npy_sha256": _file_sha(source_path),
                        "source_array_sha256": _array_sha(source),
                        "source_point_count": 1,
                        "target_npy_sha256": target_sha,
                    }
                )
                row = {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "snapshot_id": snapshot_id,
                    "selection_index": selection_index,
                    **scene_metrics,
                }
                metric_rows.append(row)
                station_rows.append(row)
                by_scene[scene_id].append(row)
            station_summaries.append(
                {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "snapshot_count": 10,
                    "aggregation": "MEDIAN_OVER_10_FROZEN_QUERY_SNAPSHOTS",
                    "median": {
                        field: float(np.median([float(row[field]) for row in station_rows]))
                        for field in GEOMETRY_ONLY_FIELDS
                    },
                    "geometry_only": True,
                    "registration_executed": False,
                }
            )

    scene_summaries = []
    for scene_id in INITIAL_SCENE_IDS:
        rows = by_scene[scene_id]
        medians = {
            field: float(np.median([float(row[field]) for row in rows]))
            for field in GEOMETRY_ONLY_FIELDS
        }
        gate = geometry_admission(
            medians["normalized_lambda_min_trans"],
            medians["condition_number_trans"],
            medians["spectral_entropy_trans"],
        )
        candidate = "RICH" if scene_id.startswith("FMB1_R") else "WEAK"
        aligned = gate["final_geometry_class"] == candidate
        if aligned:
            status, reason = "GEOMETRY_ADMITTED", None
        elif gate["final_geometry_class"] == "INTERMEDIATE":
            status, reason = "GEOMETRY_REVIEW", "GEOMETRY_INTERMEDIATE"
        else:
            status = "GEOMETRY_REJECTED"
            reason = "SEMANTIC_CANDIDATE_GEOMETRY_CLASS_MISMATCH"
        scene_summaries.append(
            {
                "scene_id": scene_id,
                "semantic_candidate_label": f"{candidate}_CANDIDATE",
                "aggregation": "MEDIAN_OVER_ALL_30_NESTED_STATION_SNAPSHOTS",
                "snapshot_count": 30,
                "station_count": 3,
                **{f"median_{field}": value for field, value in medians.items()},
                **gate,
                "candidate_class_alignment": aligned,
                "geometry_admission_status": status,
                "failure_reason": reason,
                "replacement_allowed_under_preregistration": status
                != "GEOMETRY_ADMITTED",
            }
        )
    assets = {"targets": targets, "snapshots": snapshots}
    geometry = {
        "metric_fields": list(GEOMETRY_ONLY_FIELDS),
        "T0": "IDENTITY_4X4",
        "geometry_only": True,
        "registration_executed": False,
        "snapshot_metrics": metric_rows,
        "station_summaries": station_summaries,
        "scene_summaries": scene_summaries,
        "open3d_registration_call_count": 0,
        "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0,
        "formal_trial_count": 0,
    }
    return assets, geometry


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict]:
    assets, geometry = _payloads(tmp_path)
    calls = {"prepare": 0, "compute": 0}

    class FakeContext:
        @classmethod
        def prepare(cls, target):
            calls["prepare"] += 1
            return cls()

    def fake_compute(source, transform, *, context):
        del transform, context
        calls["compute"] += 1
        return _metrics_for_scene(
            "FMB1_R01" if float(source[0, 0]) > 0.0 else "FMB1_W01"
        )

    monkeypatch.setattr(deep, "TargetGeometryContext", FakeContext)
    monkeypatch.setattr(deep, "compute_geometry_only_initial_metrics", fake_compute)
    return assets, geometry, calls


def test_recomputes_18_targets_and_180_rows_and_accepts_consistent_w02_reject(
    evidence: tuple[dict, dict, dict],
) -> None:
    assets, geometry, calls = evidence
    report = verify_geometry_evidence(assets, geometry, {})
    assert report["status"] == "PASS"
    assert report["geometry_row_count"] == 180
    assert calls == {"prepare": 18, "compute": 180}
    w02 = next(row for row in report["verified_scenes"] if row["scene_id"] == "FMB1_W02")
    assert w02 == {
        "scene_id": "FMB1_W02",
        "final_geometry_class": "RICH",
        "geometry_admission_status": "GEOMETRY_REJECTED",
        "candidate_class_alignment": False,
    }


def test_metric_tamper_beyond_one_e_minus_twelve_fails(
    evidence: tuple[dict, dict, dict],
) -> None:
    assets, geometry, _ = evidence
    geometry["snapshot_metrics"][0]["spectral_entropy_trans"] += 2.0e-12
    with pytest.raises(DeepGeometryVerificationError, match="spectral_entropy_trans changed"):
        verify_geometry_evidence(assets, geometry, {})


def test_sub_tolerance_rounding_is_accepted(evidence: tuple[dict, dict, dict]) -> None:
    assets, geometry, _ = evidence
    geometry["snapshot_metrics"][0]["spectral_entropy_trans"] += 0.5e-12
    assert verify_geometry_evidence(assets, geometry, {})["pass"] is True


def test_forbidden_result_field_fails_before_geometry_kernel(
    evidence: tuple[dict, dict, dict],
) -> None:
    assets, geometry, calls = evidence
    geometry["snapshot_metrics"][0]["fitness"] = 1.0
    with pytest.raises(DeepGeometryVerificationError, match="forbidden result field"):
        verify_geometry_evidence(assets, geometry, {})
    assert calls["compute"] == 0


def test_scene_admission_tamper_fails_but_consistent_rejection_does_not(
    evidence: tuple[dict, dict, dict],
) -> None:
    assets, geometry, _ = evidence
    w02 = next(row for row in geometry["scene_summaries"] if row["scene_id"] == "FMB1_W02")
    assert w02["geometry_admission_status"] == "GEOMETRY_REJECTED"
    w02["geometry_admission_status"] = "GEOMETRY_ADMITTED"
    with pytest.raises(DeepGeometryVerificationError, match="geometry_admission_status changed"):
        verify_geometry_evidence(assets, geometry, {})


def test_source_file_sha_tamper_fails(evidence: tuple[dict, dict, dict]) -> None:
    assets, geometry, _ = evidence
    path = Path(assets["snapshots"][0]["source_path"])
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(DeepGeometryVerificationError, match="file SHA256 changed"):
        verify_geometry_evidence(assets, geometry, {})


def test_geometry_kernel_extra_result_field_is_rejected(
    evidence: tuple[dict, dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    assets, geometry, _ = evidence

    def contaminated(source, transform, *, context):
        del source, transform, context
        return {**RICH, "final_residual": 0.0}

    monkeypatch.setattr(deep, "compute_geometry_only_initial_metrics", contaminated)
    with pytest.raises(DeepGeometryVerificationError, match="kernel schema changed"):
        verify_geometry_evidence(assets, geometry, {})
