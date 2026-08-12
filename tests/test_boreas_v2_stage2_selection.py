from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import phase_a_harness.common_association_analysis as common
import phase_a_harness.real_data_preparation.boreas_v2_stage2_selection as selection_module
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    FIRST_PASS_SCAN_FIELDS,
    GEOMETRY_ONLY_FIELDS,
    RICH_LABEL,
    ReferencePoseSeries,
    SYNTHETIC_AUTHORITY,
    SelectionBindings,
    Stage2SelectionContract,
    TargetGeometryContext,
    WEAK_LABEL,
    build_blind_selection_manifest,
    build_candidate_intervals,
    build_candidate_scan_inventory,
    compute_geometry_only_initial_metrics,
    select_frozen_intervals,
    select_interval_quantile_snapshots,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _contract(*, scans_per_window: int = 5) -> Stage2SelectionContract:
    return Stage2SelectionContract(
        parameter_authority=SYNTHETIC_AUTHORITY,
        expected_candidate_scan_count=20 * scans_per_window,
        expected_candidate_interval_count=20,
    )


def _bindings() -> SelectionBindings:
    return SelectionBindings(
        primary_query_sequence_id="boreas-query",
        gt_sha256=SHA_A,
        calibration_sha256=SHA_B,
        preprocessing_contract_sha256=SHA_C,
        target_map_sha256=SHA_D,
    )


def _windows() -> list[dict[str, int]]:
    return [
        {
            "window_index": index,
            "interval_index": index,
            "start_time_us": index * 20_000_000,
            "end_time_us": index * 20_000_000 + 5_000_000,
            "duration_us": 5_000_000,
        }
        for index in range(20)
    ]


def _first_pass_rows(
    *, scans_per_window: int = 5, invalid: set[tuple[int, int]] | None = None
) -> list[dict[str, object]]:
    invalid = invalid or set()
    rows: list[dict[str, object]] = []
    offsets = (
        np.linspace(500_000, 4_500_000, scans_per_window)
        .round()
        .astype(np.int64)
        .tolist()
    )
    for interval_index, window in enumerate(_windows()):
        for local_index, offset in enumerate(offsets):
            timestamp = window["start_time_us"] + int(offset)
            normalized = (interval_index + 1) / 100.0
            row: dict[str, object] = {
                "query_ordinal": len(rows),
                "sequence_id": "boreas-query",
                "object_key": f"boreas-query/lidar/{timestamp}.bin",
                "timestamp_us": timestamp,
                "remote_size_bytes": 2048,
                "last_modified": "2026-01-01T00:00:00Z",
                "etag": f"etag-{len(rows)}",
                "payload_sha256": f"{len(rows):064x}",
                "finite_source_point_count": 2000,
                "target_map_point_count": 20000,
                "reference_interpolation_valid": True,
                "reference_gap_s": 0.1,
                "gt_overlap_within_5m": True,
                "target_map_frozen_complete": True,
                "deskew_processing_contract_valid": (
                    interval_index,
                    local_index,
                )
                not in invalid,
                "gt_sha256": SHA_A,
                "calibration_sha256": SHA_B,
                "preprocessing_contract_sha256": SHA_C,
                "target_map_sha256": SHA_D,
                "initial_correspondence_count": 1000,
                "initial_valid_normal_correspondence_count": 900,
                "lambda_min_trans": normalized,
                "lambda_mid_trans": 0.3,
                "lambda_max_trans": 0.6,
                "normalized_lambda_min_trans": normalized,
                "normalized_lambda_mid_trans": 0.3,
                "normalized_lambda_max_trans": 0.7 - normalized,
                "condition_number_trans": 100.0 - interval_index,
                "spectral_entropy_trans": interval_index / 100.0,
            }
            assert set(row) == set(FIRST_PASS_SCAN_FIELDS)
            rows.append(row)
    return rows


def _poses() -> ReferencePoseSeries:
    centers = np.asarray(
        [window["start_time_us"] + 2_500_000 for window in _windows()],
        dtype=np.int64,
    )
    translations = np.column_stack(
        (np.arange(20, dtype=np.float64) * 2.0, np.zeros(20), np.zeros(20))
    )
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (20, 1))
    return ReferencePoseSeries(centers, translations, quaternions)


def _pipeline(
    *, scans_per_window: int = 5, invalid: set[tuple[int, int]] | None = None
) -> tuple[
    Stage2SelectionContract,
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    contract = _contract(scans_per_window=scans_per_window)
    scans, _ = build_candidate_scan_inventory(
        _first_pass_rows(scans_per_window=scans_per_window, invalid=invalid),
        _windows(),
        bindings=_bindings(),
        contract=contract,
    )
    intervals, _ = build_candidate_intervals(
        scans, _windows(), _poses(), contract=contract
    )
    selected_intervals = select_frozen_intervals(intervals, contract=contract)
    snapshots = select_interval_quantile_snapshots(
        scans, selected_intervals, contract=contract
    )
    return contract, scans, intervals, selected_intervals, snapshots


def test_geometry_only_entry_never_calls_residual_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "_residuals",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("forbidden")),
    )
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, 5), np.linspace(-0.2, 0.2, 5))
    target = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    source = target[:12].copy()
    result = compute_geometry_only_initial_metrics(source, target, np.eye(4))
    assert tuple(result) == GEOMETRY_ONLY_FIELDS
    assert result["initial_correspondence_count"] == source.shape[0]
    assert result["initial_valid_normal_correspondence_count"] == source.shape[0]
    assert result["lambda_min_trans"] == pytest.approx(0.0)
    assert not any(
        fragment in key
        for key in result
        for fragment in ("residual", "gradient", "final", "turnover", "displacement")
    )


def test_geometry_only_entry_represents_noncomputable_spectrum_as_null() -> None:
    points = np.zeros((5, 3), dtype=np.float64)
    result = compute_geometry_only_initial_metrics(points, points, np.eye(4))
    assert result["initial_correspondence_count"] == 5
    assert result["initial_valid_normal_correspondence_count"] == 0
    assert all(result[field] is None for field in GEOMETRY_ONLY_FIELDS[2:])


def test_geometry_target_context_is_prepared_once_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = selection_module._estimate_target_normals_with_tree

    def counted(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        selection_module, "_estimate_target_normals_with_tree", counted
    )
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, 5), np.linspace(-0.2, 0.2, 5))
    target = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype("<f8")
    context = TargetGeometryContext.prepare(target)
    assert calls == 1
    first = compute_geometry_only_initial_metrics(
        target[:12], np.eye(4), context=context
    )
    second = context.compute(target[1:13], np.eye(4))
    assert calls == 1
    assert first == compute_geometry_only_initial_metrics(
        target[:12], target, np.eye(4)
    )
    assert calls == 2
    assert second["initial_correspondence_count"] == 12


def test_geometry_target_context_accepts_readonly_memmap_without_copy(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "target.f8"
    target = np.arange(90, dtype="<f8").reshape(30, 3) / 100.0
    target.tofile(target_path)
    mapped = np.memmap(target_path, dtype="<f8", mode="r", shape=(30, 3))
    context = TargetGeometryContext.prepare(mapped)
    assert np.shares_memory(context.target_points, mapped)


def test_full_blind_selection_is_10x5_weak_and_rich_and_deterministic() -> None:
    first = _pipeline()
    second = _pipeline()
    assert first == second
    contract, scans, intervals, selected_intervals, snapshots = first
    assert len(scans) == 100
    assert len(intervals) == 20
    assert len(selected_intervals) == 20
    assert [row["window_index"] for row in selected_intervals[:10]] == list(range(10))
    assert [row["window_index"] for row in selected_intervals[10:]] == list(
        range(19, 9, -1)
    )
    assert sum(row["scene_label"] == WEAK_LABEL for row in snapshots) == 50
    assert sum(row["scene_label"] == RICH_LABEL for row in snapshots) == 50
    assert len({row["object_key"] for row in snapshots}) == 100
    manifest = build_blind_selection_manifest(
        candidate_scans=scans,
        candidate_intervals=intervals,
        selected_intervals=selected_intervals,
        selected_snapshots=snapshots,
        contract=contract,
        bindings=_bindings(),
    )
    assert manifest["r14_selection_frozen"] is True
    assert manifest["registration_execution_count"] == 0
    assert manifest["weak_snapshot_count"] == manifest["rich_snapshot_count"] == 50


def test_interval_eighty_percent_gate_is_integer_and_valid_only() -> None:
    invalid = {(0, 0), (1, 0), (1, 1)}
    contract = _contract()
    scans, excluded_scans = build_candidate_scan_inventory(
        _first_pass_rows(invalid=invalid),
        _windows(),
        bindings=_bindings(),
        contract=contract,
    )
    intervals, excluded_intervals = build_candidate_intervals(
        scans, _windows(), _poses(), contract=contract
    )
    assert len(excluded_scans) == 3
    assert intervals[0]["geometry_valid_scan_count"] == 4
    assert intervals[0]["interval_valid"] is True
    assert intervals[1]["geometry_valid_scan_count"] == 3
    assert intervals[1]["interval_valid"] is False
    assert [row["interval_id"] for row in excluded_intervals] == [
        "boreas-v2-window-001"
    ]
    expected = np.median(
        [
            float(row["normalized_lambda_min_trans"])
            for row in scans
            if row["interval_id"] == "boreas-v2-window-000"
            and row["geometry_valid"] is True
        ]
    )
    assert intervals[0]["normalized_lambda_min_trans"] == expected
    assert intervals[1]["normalized_lambda_min_trans"] is None


def test_deskew_processing_validity_is_not_numeric_uncertainty() -> None:
    rows = _first_pass_rows(invalid={(0, 0)})
    assert "deskew_uncertainty" not in rows[0]
    scans, excluded = build_candidate_scan_inventory(
        rows,
        _windows(),
        bindings=_bindings(),
        contract=_contract(),
    )
    assert scans[0]["geometry_valid"] is False
    assert scans[0]["exclusion_reason"] == "DESKEW_PROCESSING_CONTRACT_INVALID"
    assert len(excluded) == 1


def test_midpoint_pose_interpolation_uses_linear_translation_and_slerp() -> None:
    series = ReferencePoseSeries(
        np.asarray([0, 200_000], dtype=np.int64),
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
            ]
        ),
    )
    translation, quaternion, lower, upper = series.interpolate_midpoint(
        100_000, maximum_gap_s=0.2
    )
    assert np.allclose(translation, [1.0, 0.0, 0.0])
    vector = Rotation_from_xyzw(quaternion) @ np.asarray([1.0, 0.0, 0.0])
    assert np.allclose(vector, [np.sqrt(0.5), np.sqrt(0.5), 0.0])
    assert (lower, upper) == (0, 200_000)


def Rotation_from_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    # Keep scipy out of the production assertion: this explicit formula is an
    # independent check of the interpolated 45-degree Z rotation.
    x, y, z, w = (float(value) for value in quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_independence_rule_rejects_only_when_time_and_space_are_both_close() -> None:
    contract = _contract()
    base = {
        "start_time_us": 0,
        "end_time_us": 5_000_000,
        "center_world_position": [0.0, 0.0, 0.0],
    }
    close_time_far_space = {
        "start_time_us": 6_000_000,
        "end_time_us": 11_000_000,
        "center_world_position": [2.0, 0.0, 0.0],
    }
    boundary_time_close_space = {
        "start_time_us": 10_000_000,
        "end_time_us": 15_000_000,
        "center_world_position": [0.0, 0.0, 0.0],
    }
    close_both = {
        "start_time_us": 6_000_000,
        "end_time_us": 11_000_000,
        "center_world_position": [0.5, 0.0, 0.0],
    }
    assert selection_module._compatible(close_time_far_space, [base], contract) is True
    assert selection_module._compatible(boundary_time_close_space, [base], contract) is True
    assert selection_module._compatible(close_both, [base], contract) is False


def test_quantile_rule_uses_nearest_unused_and_earlier_tie() -> None:
    contract, scans, intervals, selected_intervals, _ = _pipeline(scans_per_window=6)
    snapshots = select_interval_quantile_snapshots(
        scans, selected_intervals, contract=contract
    )
    first = [
        row["selected_timestamp_us"]
        for row in snapshots
        if row["interval_id"] == "boreas-v2-window-000"
    ]
    # linspace offsets are 0.5, 1.3, 2.1, 2.9, 3.7, 4.5 seconds.
    assert first == [500_000, 1_300_000, 2_100_000, 2_900_000, 3_700_000]


def test_first_pass_schema_rejects_registration_or_unknown_fields() -> None:
    row = _first_pass_rows()[0]
    row["initial_residual_rmse"] = 0.0
    rows = _first_pass_rows()
    rows[0] = row
    with pytest.raises(selection_module.BoreasStage2SelectionError, match="field set"):
        build_candidate_scan_inventory(
            rows,
            _windows(),
            bindings=_bindings(),
            contract=_contract(),
        )
