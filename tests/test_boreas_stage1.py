from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from phase_a_harness.real_data_preparation import boreas_stage1 as producer
from phase_a_harness.real_data_preparation import boreas_stage1_verifier as verifier
from phase_a_harness.real_data_preparation.guard import (
    NoRegistrationGuard,
    RegistrationForbiddenError,
    assert_preparation_sources_are_safe,
)
from phase_a_harness.real_data_preparation.io import canonical_json_bytes, compact_sha256, sha256_file
from phase_a_harness.real_data_preparation.stage1_gt_overlap import (
    FROZEN_STAGE1_OVERLAP_CONTRACT,
    compute_stage1_gt_overlap,
    rank_distinct_stage1_pairs,
)


def _pose_row(timestamp: float, xyz=(0.0, 0.0, 0.0), rph=(0.0, 0.0, 0.0)) -> list[float]:
    roll, pitch, heading = rph
    return [timestamp, *xyz, 0.0, 0.0, 0.0, roll, pitch, heading, 0.0, 0.0, 0.0]


def _write_pose(path: Path, rows: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(producer.POSE_HEADER)
        writer.writerows(rows)


def _make_closure(root: Path) -> None:
    for name in verifier.REQUIRED_FILES - {"SHA256SUMS", "boreas_stage1_manifest.json"}:
        (root / name).write_bytes(b"{}\n")
    unsigned = {
        "data_evidence": [],
        "eligibility_sha256": sha256_file(root / "boreas_stage1_eligibility.json"),
        "official_source_identity": {
            "aws_bucket": verifier.AWS_URI,
            "paper_doi": verifier.PAPER_DOI,
            "pyboreas_commit": verifier.PYBOREAS_COMMIT,
        },
        "payload": [
            {"path": name, "sha256": sha256_file(root / name), "size_bytes": (root / name).stat().st_size}
            for name in sorted(verifier.REQUIRED_FILES - {"SHA256SUMS", "boreas_stage1_manifest.json"})
        ],
        "producer_commit": "1" * 40,
        "schema_version": "boreas_stage1_manifest_v1",
    }
    manifest = {**unsigned, "manifest_payload_sha256": compact_sha256(unsigned)}
    (root / "boreas_stage1_manifest.json").write_bytes(canonical_json_bytes(manifest))
    names = sorted(verifier.REQUIRED_FILES - {"SHA256SUMS"})
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in names), encoding="utf-8"
    )


def test_01_s3_sequence_listing_is_exact_and_sorted() -> None:
    text = "                           PRE boreas-2020-11-26-13-58/\n                           PRE boreas-2020-12-01-13-26/\n"
    assert producer.parse_top_level_s3_listing(text) == [
        "boreas-2020-11-26-13-58", "boreas-2020-12-01-13-26"
    ]


def test_02_s3_sequence_listing_rejects_unsorted_or_duplicate() -> None:
    with pytest.raises(producer.BoreasStage1Error):
        producer.parse_top_level_s3_listing("PRE boreas-2021-01-01-00-01/\nPRE boreas-2021-01-01-00-00/\n")


def test_03_s3_object_listing_row_parser() -> None:
    row = producer.parse_s3_ls_line("2023-03-09 14:02:00 499 boreas-2020-11-26-13-58/lidar/160.bin")
    assert row == {
        "key": "boreas-2020-11-26-13-58/lidar/160.bin",
        "last_modified": "2023-03-09T14:02:00Z",
        "size_bytes": 499,
    }


def test_04_test_sequences_are_excluded_from_public_gt() -> None:
    assert len(producer.TRAIN_SEQUENCES) == 31
    assert len(producer.TEST_SEQUENCES) == 13
    assert set(producer.TRAIN_SEQUENCES).isdisjoint(producer.TEST_SEQUENCES)
    assert producer.REFERENCE_SEQUENCE in producer.TRAIN_SEQUENCES


def test_05_single_object_over_500mb_is_forbidden() -> None:
    with pytest.raises(producer.Stage1LargeFileDownloadForbidden, match="STAGE1_LARGE_FILE"):
        producer.assert_download_budget(500_000_001, 0, maximum_single=500_000_000, maximum_total=5_000_000_000)


def test_06_single_object_exactly_500mb_is_allowed() -> None:
    producer.assert_download_budget(500_000_000, 0, maximum_single=500_000_000, maximum_total=5_000_000_000)


def test_07_stage1_total_over_5gb_is_forbidden() -> None:
    with pytest.raises(producer.Stage1DownloadBudgetExceeded, match="BLOCKED_STAGE1_DOWNLOAD_BUDGET"):
        producer.assert_download_budget(2, 4_999_999_999, maximum_single=500_000_000, maximum_total=5_000_000_000)


def test_08_caller_cannot_raise_frozen_download_ceilings() -> None:
    with pytest.raises(producer.Stage1LargeFileDownloadForbidden):
        producer.assert_download_budget(1, 0, maximum_single=500_000_001, maximum_total=5_000_000_000)
    with pytest.raises(producer.Stage1DownloadBudgetExceeded):
        producer.assert_download_budget(1, 0, maximum_single=500_000_000, maximum_total=5_000_000_001)


def test_09_fixed_enu_reference_identity_is_frozen() -> None:
    assert producer.REFERENCE_SEQUENCE == "boreas-2020-11-26-13-58"
    assert producer.PYBOREAS_COMMIT == verifier.PYBOREAS_COMMIT


def test_10_roll_pitch_yaw_convention_matches_devkit_order() -> None:
    heading, pitch, roll = 0.3, -0.2, 0.1
    expected = producer._roll(roll) @ producer._pitch(pitch) @ producer._yaw(heading)
    assert np.array_equal(producer.yaw_pitch_roll_to_rotation(heading, pitch, roll), expected)


def test_11_pose_row_uses_heading_pitch_roll_columns() -> None:
    row = _pose_row(0.0, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3))
    transform = producer.pose_row_to_transform(row)
    assert np.array_equal(transform[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(transform[:3, :3], producer.yaw_pitch_roll_to_rotation(0.3, 0.2, 0.1))


def test_12_transform_direction_is_right_multiplication() -> None:
    enu_applanix = np.eye(4)
    enu_applanix[:3, 3] = [1, 2, 3]
    applanix_lidar = np.eye(4)
    applanix_lidar[:3, 3] = [4, 5, 6]
    assert np.array_equal(producer.compose_enu_lidar(enu_applanix, applanix_lidar)[:3, 3], [5, 7, 9])


def test_13_calibration_parser_rejects_nonhomogeneous_matrix(tmp_path: Path) -> None:
    path = tmp_path / "T.txt"
    matrix = np.eye(4)
    matrix[3, 0] = 1
    np.savetxt(path, matrix)
    with pytest.raises(producer.BoreasStage1Error):
        producer.parse_calibration(path)


def test_14_pose_csv_accepts_utc_microseconds(tmp_path: Path) -> None:
    path = tmp_path / "lidar_poses.csv"
    _write_pose(path, [_pose_row(1_600_000_000_000_000 + index * 100_000) for index in range(4)])
    report, trajectory, _ = producer.parse_pose_csv(path, "sequence")
    assert report["timestamp_scale_to_seconds"] == 1e-6
    assert report["median_rate_hz"] == pytest.approx(10.0)
    assert trajectory[-1, 0] - trajectory[0, 0] == pytest.approx(0.3)


def test_15_gps_post_process_200hz_rate_estimate(tmp_path: Path) -> None:
    path = tmp_path / "gps_post_process.csv"
    _write_pose(path, [_pose_row(1_600_000_000.0 + index * 0.005) for index in range(100)])
    report, _, _ = producer.parse_pose_csv(path, "reference")
    assert report["median_rate_hz"] == pytest.approx(200.0, abs=0.01)


def test_16_pose_schema_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("time,x\n1,2\n2,3\n", encoding="utf-8")
    with pytest.raises(producer.BoreasStage1Error, match="header"):
        producer.parse_pose_csv(path, "bad")


def test_17_duplicate_gt_timestamps_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    _write_pose(path, [_pose_row(1.0), _pose_row(1.0)])
    with pytest.raises(producer.BoreasStage1Error, match="nonmonotonic"):
        producer.parse_pose_csv(path, "bad")


def test_18_pose_chain_consistency_recomputes_identity() -> None:
    gps = np.asarray([_pose_row(1_600_000_000 + index * 0.005, (index * 0.01, 0, 0)) for index in range(401)])
    lidar = np.asarray([_pose_row(1_600_000_000_000_000 + index * 100_000, (index * 0.2, 0, 0)) for index in range(21)])
    result = producer.transform_chain_consistency(gps, lidar, np.eye(4))
    assert result["status"] == "PASS"
    assert result["maximum_translation_error_m"] < 1e-6
    assert result["maximum_rotation_error_rad"] < 1e-12


def test_19_gt_only_overlap_is_deterministic() -> None:
    times = np.arange(0.0, 201.0, 0.1)
    trajectory = np.column_stack((times, times, np.zeros_like(times), np.zeros_like(times)))
    first = compute_stage1_gt_overlap(trajectory, trajectory.copy(), common_world_frame_proven=True)
    second = compute_stage1_gt_overlap(trajectory, trajectory.copy(), common_world_frame_proven=True)
    assert first == second
    assert first["overlap_status"] == "PASS"


def test_20_unproven_common_world_blocks_overlap() -> None:
    trajectory = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
    assert compute_stage1_gt_overlap(trajectory, trajectory, common_world_frame_proven=False)["overlap_status"] == "NOT_COMPUTABLE"


def test_21_same_sequence_pair_is_prohibited() -> None:
    with pytest.raises(ValueError, match="same-sequence"):
        rank_distinct_stage1_pairs([{"map_sequence_id": "x", "query_sequence_id": "x"}])


def test_22_overlap_thresholds_are_immutable() -> None:
    contract = FROZEN_STAGE1_OVERLAP_CONTRACT
    assert contract.min_total_covered_duration_s == 150.0
    assert contract.min_coverage_fraction == 0.60
    assert contract.min_eligible_nonoverlapping_5s_intervals == 30
    assert contract.radius_m == 5.0 and contract.resample_rate_hz == 1.0


def test_23_pair_ranking_is_deterministic() -> None:
    base = {
        "overlap_status": "PASS", "map_independent_6dof": True, "query_independent_6dof": True,
        "common_world_frame_proven": True, "map_rig_lidar_transform_available": True,
        "query_rig_lidar_transform_available": True, "total_covered_duration_s": 200,
        "coverage_fraction": 0.8, "eligible_nonoverlapping_5s_interval_count": 40,
        "nearest_distance_q95_m": 2.0,
    }
    rows = [{**base, "map_sequence_id": "b", "query_sequence_id": "c"}, {**base, "map_sequence_id": "a", "query_sequence_id": "c"}]
    ranked = rank_distinct_stage1_pairs(reversed(rows))
    assert [row["map_sequence_id"] for row in ranked] == ["a", "b"]


def test_24_rmse_cannot_be_relabelled_as_1sigma() -> None:
    rows = producer._uncertainty_rows("a" * 64, "b" * 64, "c" * 64)
    position = next(row for row in rows if row["component"] == "GNSS/RTX position uncertainty")
    assert position["uncertainty_type"] == "RMSE"
    assert "0.02-0.04" in position["value"]
    assert all(row["uncertainty_type"] != "1SIGMA" for row in rows)


def test_25_unknown_uncertainty_cannot_become_zero() -> None:
    rows = producer._uncertainty_rows("a" * 64, "b" * 64, "c" * 64)
    unknown = [row for row in rows if row["uncertainty_type"] == "UNKNOWN"]
    assert len(unknown) == 8
    assert all(row["value"] == "UNKNOWN" and row["status"] == "UNKNOWN" for row in unknown)


def test_26_lidar_assisted_extrinsic_provenance_is_explicit() -> None:
    source = Path(producer.__file__).read_text(encoding="utf-8")
    assert '"uses_lidar_pointclouds": True' in source
    assert '"frozen_R02_independence_violation": True' in source
    assert 'r02 = "FAIL"' in source


def test_27_steam_icp_is_blocked_by_runtime_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard(open3d_module=None):
        with pytest.raises(RegistrationForbiddenError):
            subprocess.run(["steam_icp", "--help"])


def test_28_boreas_sources_pass_static_no_registration_audit() -> None:
    source_root = Path(producer.__file__).parent
    assert assert_preparation_sources_are_safe(source_root)["pass"] is True


def test_29_manifest_payload_tamper_fails(tmp_path: Path) -> None:
    _make_closure(tmp_path)
    verifier._verify_closure(tmp_path)
    (tmp_path / "boreas_stage1_summary.json").write_bytes(b'{"tampered":true}\n')
    with pytest.raises(verifier.BoreasStage1VerificationError, match="SHA256 mismatch"):
        verifier._verify_closure(tmp_path)


def test_30_manifest_self_hash_tamper_fails(tmp_path: Path) -> None:
    _make_closure(tmp_path)
    manifest = json.loads((tmp_path / "boreas_stage1_manifest.json").read_text(encoding="utf-8"))
    manifest["manifest_payload_sha256"] = "0" * 64
    (tmp_path / "boreas_stage1_manifest.json").write_bytes(canonical_json_bytes(manifest))
    (tmp_path / "SHA256SUMS").write_text(
        (tmp_path / "SHA256SUMS").read_text().replace(
            next(line.split()[0] for line in (tmp_path / "SHA256SUMS").read_text().splitlines() if line.endswith("boreas_stage1_manifest.json")),
            sha256_file(tmp_path / "boreas_stage1_manifest.json"),
        ), encoding="utf-8"
    )
    with pytest.raises(verifier.BoreasStage1VerificationError, match="manifest payload SHA"):
        verifier._verify_closure(tmp_path)


def test_31_summary_has_exactly_29_answers() -> None:
    summary = {"answers": [str(index) for index in range(1, 30)], "boreas_stage1_ready": False, "final_conclusion": "FAIL"}
    markdown = producer._summary_markdown(summary)
    assert markdown.count("\n") >= 31
    assert "29. 29" in markdown


def test_32_expected_devkit_hashes_match_between_implementations() -> None:
    assert verifier.EXPECTED_DEVKIT_HASHES == {
        "README.md": producer.README_SHA256,
        "DATA_REFERENCE.md": producer.DATA_REFERENCE_SHA256,
        "DATA_LICENSE.md": producer.DATA_LICENSE_SHA256,
        "LICENSE": producer.CODE_LICENSE_SHA256,
        "pyboreas/data/splits.py": producer.SPLITS_SHA256,
        "pyboreas/data/calib.py": producer.CALIB_PARSER_SHA256,
        "pyboreas/utils/utils.py": producer.UTILS_SHA256,
    }


def test_33_r09_backend_outer_and_inner_hashes_are_frozen() -> None:
    path = Path("frozen_assets/backend_parameter_contract.json")
    backend = json.loads(path.read_text(encoding="utf-8"))
    assert sha256_file(path) == producer.BACKEND_PARAMETER_SHA256 == verifier.BACKEND_PARAMETER_SHA256
    for name, expected in (
        ("open3d", producer.OPEN3D_PARAMETER_CANONICAL_SHA256),
        ("pcl", producer.PCL_PARAMETER_CANONICAL_SHA256),
    ):
        assert compact_sha256(backend[name]["parameters"]) == expected
        assert backend[name]["canonical_sha256"] == expected


def test_34_sequence_local_reset_detection_uses_published_first_pose() -> None:
    reports = [
        {"sequence_id": "reference", "first_position_m": [0.0, 0.0, 0.0]},
        {"sequence_id": "global", "first_position_m": [321.0, -42.0, 1.0]},
        {"sequence_id": "reset", "first_position_m": [1e-8, -1e-8, 0.0]},
    ]
    assert producer.detect_sequence_local_resets(reports) == ["reset"]


def test_35_gps_seconds_and_lidar_microseconds_are_not_conflated(tmp_path: Path) -> None:
    gps = tmp_path / "gps.csv"
    lidar = tmp_path / "lidar.csv"
    _write_pose(gps, [_pose_row(1_606_417_077.0 + index * 0.005) for index in range(3)])
    _write_pose(lidar, [_pose_row(1_606_417_077_000_000 + index * 100_000) for index in range(3)])
    gps_report, _, _ = producer.parse_pose_csv(gps, "gps")
    lidar_report, _, _ = producer.parse_pose_csv(lidar, "lidar")
    assert gps_report["timestamp_scale_to_seconds"] == 1.0
    assert lidar_report["timestamp_scale_to_seconds"] == 1e-6


def test_36_official_paper_pdf_is_pinned() -> None:
    assert producer.PAPER_PDF_SHA256 == verifier.PAPER_PDF_SHA256
    assert len(producer.PAPER_PDF_SHA256) == 64
    assert set(producer.PAPER_PDF_SHA256) <= set("0123456789abcdef")
