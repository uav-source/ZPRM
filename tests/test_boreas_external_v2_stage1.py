from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phase_a_harness.real_data_preparation import boreas_external_v2_stage1 as producer
from phase_a_harness.real_data_preparation.guard import (
    NoRegistrationGuard,
    RegistrationForbiddenError,
    assert_preparation_sources_are_safe,
)
from phase_a_harness.real_data_preparation.io import canonical_json_bytes, sha256_file
from phase_a_harness.real_data_preparation.stage1_gt_overlap import (
    rank_distinct_stage1_pairs,
)


REPOSITORY = Path(__file__).resolve().parents[1]
BOREAS_V1_ROOT = REPOSITORY / "frozen_assets/real_data_boreas_stage1_v1"
PROTOCOL_ROOT = REPOSITORY / "protocols"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _make_synthetic_v1_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, list[Path]]:
    """Make an 84-file/76-receipt closure without relying on machine data."""

    data_root = tmp_path / "data"
    frozen_root = tmp_path / "boreas-v1"
    data_root.mkdir()
    frozen_root.mkdir()
    evidence: list[dict[str, Any]] = []
    evidence_paths: list[Path] = []
    for index in range(84):
        relative = f"stage1_payload/sequence-{index:03d}/small-{index:03d}.csv"
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"authenticated-small-file-{index}\n".encode("ascii"))
        downloaded = index < 76
        row = {
            "etag": f'"etag-{index:03d}"' if downloaded else None,
            "last_modified": "2025-01-01T00:00:00Z" if downloaded else None,
            "local_path": str(path),
            "relative_path": relative,
            "s3_key": f"sequence-{index:03d}/small-{index:03d}.csv" if downloaded else None,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "status": (
                "DOWNLOADED_ALLOWLISTED_STAGE1_OBJECT"
                if downloaded
                else "GENERATED_FROM_AUTHENTICATED_STAGE1_OBJECT"
            ),
            "version_id": None,
        }
        if downloaded:
            _write_json(
                path.with_name(path.name + ".receipt.json"),
                {
                    "etag": row["etag"],
                    "key": row["s3_key"],
                    "last_modified": row["last_modified"],
                    "local_path": str(path),
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                    "version_id": None,
                },
            )
        evidence.append(row)
        evidence_paths.append(path)

    eligibility = frozen_root / "boreas_stage1_eligibility.json"
    extrinsic = frozen_root / "boreas_lidar_extrinsic_provenance.json"
    manifest = frozen_root / "boreas_stage1_manifest.json"
    download = frozen_root / "download_manifest.json"
    _write_json(eligibility, {"BOREAS_STAGE1_READY": False, "R02": "FAIL"})
    _write_json(extrinsic, {"extrinsic_uncertainty": "UNKNOWN"})
    _write_json(manifest, {"data_evidence": evidence})
    _write_json(download, {"materialized_files": evidence})
    signed_names = (
        eligibility.name,
        extrinsic.name,
        manifest.name,
        download.name,
    )
    sums = frozen_root / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256_file(frozen_root / name)}  {name}\n" for name in signed_names),
        encoding="utf-8",
    )
    monkeypatch.setattr(producer, "BOREAS_V1_ELIGIBILITY_SHA256", sha256_file(eligibility))
    monkeypatch.setattr(
        producer, "BOREAS_V1_EXTRINSIC_PROVENANCE_SHA256", sha256_file(extrinsic)
    )
    monkeypatch.setattr(producer, "BOREAS_V1_MANIFEST_SHA256", sha256_file(manifest))
    monkeypatch.setattr(producer, "BOREAS_V1_SHA256SUMS_SHA256", sha256_file(sums))
    return data_root, frozen_root, evidence_paths


def _all_pair_inputs() -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, str]]:
    # Two continuous 80 s islands separated by a deliberate 20 s native gap.
    # The gap-aware implementation must never interpolate across that hole.
    timestamps = np.concatenate(
        (
            np.arange(0, 801, dtype=np.float64) / 10.0,
            np.arange(1000, 1801, dtype=np.float64) / 10.0,
        )
    )
    base = np.column_stack(
        (
            timestamps,
            np.sin(timestamps / 20.0),
            np.cos(timestamps / 20.0),
            np.zeros_like(timestamps),
        )
    )
    sequence_ids = [f"sequence-{index:02d}" for index in range(29)]
    reports = [{"duration_s": 160.0, "sequence_id": sequence} for sequence in sequence_ids]
    trajectories = {sequence: base.copy() for sequence in sequence_ids}
    calibration_sha = {sequence: "a" * 64 for sequence in sequence_ids}
    return reports, trajectories, calibration_sha


def test_source_only_and_external_baseline_semantics_are_distinct() -> None:
    value = producer.make_test_baseline_report(
        source_only_collected=800,
        source_only_passed=790,
        source_only_skipped=10,
    )
    assert value["SOURCE_ONLY_TEST_BASELINE"] == "PASS"
    source = value["source_only_baseline"]
    fields = ("collected", "errors", "failed", "passed", "skipped", "status")
    assert {key: source[key] for key in fields} == {
        "collected": 800,
        "errors": 0,
        "failed": 0,
        "passed": 790,
        "skipped": 10,
        "status": "PASS",
    }
    external = value["external_qualification_baseline"]
    assert value["EXTERNAL_QUALIFICATION_BASELINE"] == "UNAVAILABLE"
    assert external["status"] == "UNAVAILABLE"
    assert external["failed"] == 1 and external["passed"] == 0
    assert external["missing_path"] == str(producer.PCL_EXTERNAL_BUNDLE)
    assert "source-only package" in external["reason"]


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("source_only_baseline", "failed"), 1, "source-only"),
        (("external_qualification_baseline", "status"), "PASS", "external"),
        (("external_qualification_baseline", "passed"), 1, "external"),
    ),
)
def test_baseline_semantic_tampering_is_rejected(
    path: tuple[str, str], replacement: object, message: str
) -> None:
    value = producer.make_test_baseline_report(
        source_only_collected=800,
        source_only_passed=790,
        source_only_skipped=10,
    )
    tampered = copy.deepcopy(value)
    tampered[path[0]][path[1]] = replacement
    with pytest.raises(producer.BoreasExternalV2Stage1Error, match=message):
        producer._validate_test_baseline(tampered)


def test_historical_boreas_v1_hash_bindings_are_immutable() -> None:
    expected = {
        "boreas_stage1_eligibility.json": producer.BOREAS_V1_ELIGIBILITY_SHA256,
        "boreas_lidar_extrinsic_provenance.json": (
            producer.BOREAS_V1_EXTRINSIC_PROVENANCE_SHA256
        ),
        "boreas_stage1_manifest.json": producer.BOREAS_V1_MANIFEST_SHA256,
        "SHA256SUMS": producer.BOREAS_V1_SHA256SUMS_SHA256,
    }
    assert expected == {
        "boreas_stage1_eligibility.json": (
            "b76102144ad63e9383ed3027b574ce33c7023eda4b328b6b99621bb8d577f8c7"
        ),
        "boreas_lidar_extrinsic_provenance.json": (
            "fda2be032b760caefdc8db89619a9d47c7deec1a513b72f7714c01f5791a9cf0"
        ),
        "boreas_stage1_manifest.json": (
            "29c582361eb83392e54d34ea4465ae4fb6b1813c89a33a5fa1eb85e1abbab112"
        ),
        "SHA256SUMS": (
            "69a3a138617a315a3853dd33a519f9f142bd5bd9d08a3ace65b1dca9ba8d1975"
        ),
    }
    assert {name: sha256_file(BOREAS_V1_ROOT / name) for name in expected} == expected


def test_e02_keeps_v1_failure_and_discloses_v2_lidar_assisted_limitation() -> None:
    v1 = _read_json(BOREAS_V1_ROOT / "boreas_stage1_eligibility.json")
    reference = _read_json(
        BOREAS_V1_ROOT / "boreas_reference_trajectory_provenance.json"
    )
    extrinsic = _read_json(
        BOREAS_V1_ROOT / "boreas_lidar_extrinsic_provenance.json"
    )
    protocol = _read_json(PROTOCOL_ROOT / "public_data_external_validation_protocol_v2.json")
    e02 = protocol["eligibility_requirements"]["E02"]
    assert (v1["R02"], v1["R10"], v1["BOREAS_STAGE1_READY"]) == (
        "FAIL",
        "PARTIAL",
        False,
    )
    assert {
        "uses_lidar": reference["uses_lidar"],
        "uses_icp": reference["uses_icp"],
        "uses_scan_matching": reference["uses_scan_matching"],
    } == {"uses_lidar": False, "uses_icp": False, "uses_scan_matching": False}
    assert e02["boreas_adjudication_when_all_conditions_evidenced"] == (
        "PASS_WITH_DOCUMENTED_LIMITATION"
    )
    assisted = e02["historical_lidar_assisted_static_extrinsic"]
    assert assisted["allowed"] is True and assisted["all_conditions_required"] is True
    assert [row["condition_id"] for row in assisted["conditions"]] == [
        f"E02-X{index:02d}" for index in range(1, 11)
    ]
    assert extrinsic["uses_lidar_pointclouds"] is True
    assert extrinsic["common_static_calibration"] is True
    assert extrinsic["all_sequences_byte_identical"] is True
    assert {
        extrinsic["calibration_date_or_session"],
        extrinsic["query_sequence_used_for_calibration"],
        extrinsic["extrinsic_uncertainty"],
    } == {"UNKNOWN"}
    assert "does not assert" in e02["documented_limitation_semantics"]


def test_e10_unknown_components_cannot_be_relabelled_zero() -> None:
    protocol = _read_json(PROTOCOL_ROOT / "public_data_external_validation_protocol_v2.json")
    e10 = protocol["eligibility_requirements"]["E10"]
    assert e10["unknown_must_not_be_zero"] is True
    assert e10["unknown_value"] == "UNKNOWN"
    assert {
        "orientation_uncertainty",
        "time_synchronization_uncertainty",
        "extrinsic_uncertainty",
        "interpolation_uncertainty",
        "deskew_uncertainty",
        "map_accumulation_uncertainty",
    }.issubset(e10["uncertainty_components"])


def test_authenticated_v1_evidence_is_reused_without_redownload_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, frozen_root, paths = _make_synthetic_v1_evidence(tmp_path, monkeypatch)
    before = [(path.stat().st_mtime_ns, sha256_file(path)) for path in paths]
    report, bound_paths = producer.authenticate_boreas_v1_evidence(
        data_root=data_root, boreas_v1_root=frozen_root
    )
    after = [(path.stat().st_mtime_ns, sha256_file(path)) for path in paths]
    assert report["data_evidence_file_count"] == 84
    assert report["downloaded_stage1_small_object_count"] == 76
    assert report["files_redownloaded"] == 0
    assert report["registration_execution_count"] == 0
    assert len(bound_paths) == 84
    assert before == after


def test_authenticated_v1_evidence_content_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, frozen_root, paths = _make_synthetic_v1_evidence(tmp_path, monkeypatch)
    paths[-1].write_bytes(b"tampered\n")
    with pytest.raises(producer.BoreasExternalV2Stage1Error, match="content changed"):
        producer.authenticate_boreas_v1_evidence(
            data_root=data_root, boreas_v1_root=frozen_root
        )


def test_gap_aware_all_directed_pair_closure_is_complete_and_deterministic() -> None:
    reports, trajectories, calibration_sha = _all_pair_inputs()
    first_rows, first_ranked, first_raw = producer.compute_all_directed_pairs(
        reports=reports,
        trajectories=trajectories,
        calibration_sha=calibration_sha,
    )
    second_rows, second_ranked, second_raw = producer.compute_all_directed_pairs(
        reports=list(reversed(reports)),
        trajectories=dict(reversed(list(trajectories.items()))),
        calibration_sha=calibration_sha,
    )
    assert first_rows == second_rows
    assert first_ranked == second_ranked
    assert first_raw == second_raw
    assert len(first_rows) == 29 * 28 == 812
    assert len(first_ranked) == 812
    assert all(row["map_sequence_id"] != row["query_sequence_id"] for row in first_rows)
    assert [
        first_ranked[0]["map_sequence_id"], first_ranked[0]["query_sequence_id"]
    ] == ["sequence-00", "sequence-01"]
    sample = first_raw[("sequence-00", "sequence-01")]
    assert sample["overlap_status"] == "PASS"
    assert sample["total_covered_duration_s"] == 162.0
    assert sample["eligible_nonoverlapping_5s_interval_count"] == 32
    assert len(sample["contiguous_covered_intervals"]) == 2
    assert sample["contiguous_covered_intervals"][0]["end_time"] == 81.0
    assert sample["contiguous_covered_intervals"][1]["start_time"] == 100.0


def test_pair_ranking_is_deterministic_and_same_sequence_is_prohibited() -> None:
    base = {
        "common_world_frame_proven": True,
        "coverage_fraction": 0.8,
        "eligible_nonoverlapping_5s_interval_count": 40,
        "map_independent_6dof": True,
        "map_rig_lidar_transform_available": True,
        "nearest_distance_q95_m": 2.0,
        "overlap_status": "PASS",
        "query_independent_6dof": True,
        "query_rig_lidar_transform_available": True,
        "total_covered_duration_s": 200.0,
    }
    rows = [
        {**base, "map_sequence_id": "map-b", "query_sequence_id": "query"},
        {**base, "map_sequence_id": "map-a", "query_sequence_id": "query"},
    ]
    assert [row["map_sequence_id"] for row in rank_distinct_stage1_pairs(reversed(rows))] == [
        "map-a",
        "map-b",
    ]
    with pytest.raises(ValueError, match="same-sequence"):
        rank_distinct_stage1_pairs(
            [{**base, "map_sequence_id": "same", "query_sequence_id": "same"}]
        )


def test_primary_lidar_key_selection_is_gt_only_and_deterministic() -> None:
    timestamps = np.arange(0, 101, dtype=np.float64) / 10.0
    query = np.column_stack(
        (timestamps, np.zeros((timestamps.size, 3), dtype=np.float64))
    )
    map_positions = np.zeros((timestamps.size, 3), dtype=np.float64)
    map_positions[51:, 0] = 100.0
    map_trajectory = np.column_stack((timestamps, map_positions))
    map_values = np.zeros((timestamps.size, 13), dtype=np.float64)
    query_values = np.zeros((timestamps.size, 13), dtype=np.float64)
    map_values[:, 0] = 1_600_000_000_000_000 + np.arange(timestamps.size) * 100_000
    query_values[:, 0] = 1_700_000_000_000_000 + np.arange(timestamps.size) * 100_000
    overlap = {
        "contiguous_covered_intervals": [
            {
                "duration_s": 10.0,
                "end_time": 10.0,
                "sample_count": 10,
                "segment_id": 0,
                "start_time": 0.0,
            }
        ],
        "eligible_nonoverlapping_5s_interval_count": 2,
    }
    first = producer.select_primary_lidar_keys(
        map_trajectory=map_trajectory,
        query_trajectory=query,
        map_pose_values=map_values,
        query_pose_values=query_values,
        primary_overlap=overlap,
    )
    second = producer.select_primary_lidar_keys(
        map_trajectory=map_trajectory.copy(),
        query_trajectory=query.copy(),
        map_pose_values=map_values.copy(),
        query_pose_values=query_values.copy(),
        primary_overlap=copy.deepcopy(overlap),
    )
    assert first == second
    assert len(first["complete_five_second_windows"]) == 2
    assert first["selected_query_1hz_sample_count"] == 10
    assert first["map_selected_timestamp_us"] == [int(value) for value in map_values[:51, 0]]
    assert first["query_selected_timestamp_us"] == [
        int(value) for value in query_values[:100, 0]
    ]
    assert first["selection_definition"]["point_cloud_payload_consulted"] is False
    assert first["selection_definition"]["registration_result_consulted"] is False


def test_reserve_pairs_can_only_activate_for_frozen_infrastructure_failures() -> None:
    contract = _read_json(
        PROTOCOL_ROOT / "public_data_external_validation_pair_selection_contract_v2.json"
    )
    reserve = contract["reserve_pair_activation"]
    assert reserve["allowed_only_for_infrastructure_failure"] is True
    assert set(reserve["allowed_reasons"]) == {
        "download_object_permanently_missing",
        "checksum_mismatch",
        "file_corruption",
        "GT_file_corruption",
        "official_object_withdrawn_for_primary_pair",
    }
    assert {
        "ICP_error_too_large",
        "weak_rich_result_unfavorable",
        "Open3D_PCL_disagreement",
        "correlation_not_significant",
        "publication_result_unfavorable",
    } == set(reserve["disallowed_reasons"])
    reproducibility = contract["selection_reproducibility"]
    assert reproducibility["registration_result_access_allowed"] is False
    assert reproducibility["manual_pair_selection_allowed"] is False
    assert reproducibility["threshold_lowering_allowed"] is False


def test_v2_preparation_source_and_runtime_no_registration_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = Path(producer.__file__).resolve().parent
    assert assert_preparation_sources_are_safe(source_root)["pass"] is True
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard(open3d_module=None) as guard:
        with pytest.raises(RegistrationForbiddenError, match="forbidden registration process"):
            subprocess.run(["pcl_point_to_plane_cli", "--help"], check=True)
        report = guard.attestation(Path("/definitely/absent"))
    assert report["registration_execution_count"] == 0
    assert report["pcl_cli_invocation_count"] == 1
    assert report["pass"] is False


class _MetadataOnlyPopen:
    command: list[str] | None = None

    def __init__(self, command: list[str], **_: Any) -> None:
        type(self).command = command
        self.stdout = io.StringIO(
            "2023-03-09 14:02:00 499 sequence/lidar/1600000000000000.bin\n"
        )
        self.returncode = 0

    def communicate(self) -> tuple[str, str]:
        return "", ""


def test_lidar_inventory_command_is_metadata_only_and_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = {
        "key": "sequence/lidar/1600000000000000.bin",
        "last_modified": "2023-03-09T14:02:00Z",
        "size_bytes": 499,
    }
    digest = hashlib.sha256(
        json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
        + b"\n"
    ).hexdigest()
    expected = {
        "first_lidar_timestamp": 1_600_000_000_000_000,
        "last_lidar_timestamp": 1_600_000_000_000_000,
        "lidar_listing_rows_sha256": digest,
        "lidar_object_count": 1,
        "lidar_remote_bytes": 499,
    }
    monkeypatch.setattr(producer.subprocess, "Popen", _MetadataOnlyPopen)
    rows, listing_sha = producer._list_lidar_metadata(
        aws=Path("/isolated/aws"), sequence_id="sequence", expected_inventory=expected
    )
    assert listing_sha == digest
    assert len(rows) == 1 and rows[0]["size_bytes"] == 499
    command = _MetadataOnlyPopen.command
    assert command == [
        "/isolated/aws",
        "s3",
        "ls",
        "s3://boreas/sequence/lidar/",
        "--recursive",
        "--no-sign-request",
    ]
    assert not {"cp", "sync"}.intersection(command)


def test_lidar_inventory_hash_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(producer.subprocess, "Popen", _MetadataOnlyPopen)
    with pytest.raises(
        producer.BoreasExternalV2Stage1Error,
        match="no longer matches authenticated v1 inventory",
    ):
        producer._list_lidar_metadata(
            aws=Path("/isolated/aws"),
            sequence_id="sequence",
            expected_inventory={
                "first_lidar_timestamp": 1_600_000_000_000_000,
                "last_lidar_timestamp": 1_600_000_000_000_000,
                "lidar_listing_rows_sha256": "0" * 64,
                "lidar_object_count": 1,
                "lidar_remote_bytes": 499,
            },
        )


def test_stage1_protocol_freezes_zero_payload_and_all_authorization_boundaries() -> None:
    protocol = _read_json(PROTOCOL_ROOT / "public_data_external_validation_protocol_v2.json")
    assert {
        "downloaded_lidar_bytes": protocol["downloaded_lidar_bytes"],
        "downloaded_lidar_object_count": protocol["downloaded_lidar_object_count"],
        "downloaded_lidar_payload_count": protocol["downloaded_lidar_payload_count"],
        "registration_execution_count": protocol["registration_execution_count"],
        "real_trial_result_count": protocol["real_trial_result_count"],
        "actual_trials": protocol["actual_trials"],
        "planned_trials": protocol["planned_trials"],
        "snapshot_count": protocol["snapshot_count"],
        "weak_snapshot_count": protocol["weak_snapshot_count"],
        "rich_snapshot_count": protocol["rich_snapshot_count"],
    } == {
        "downloaded_lidar_bytes": 0,
        "downloaded_lidar_object_count": 0,
        "downloaded_lidar_payload_count": 0,
        "registration_execution_count": 0,
        "real_trial_result_count": 0,
        "actual_trials": 0,
        "planned_trials": 0,
        "snapshot_count": 0,
        "weak_snapshot_count": 0,
        "rich_snapshot_count": 0,
    }
    assert protocol["PUBLIC_DATA_V2_RUN_AUTHORIZED"] is False
    assert protocol["REAL_REGISTRATION_AUTHORIZED"] is False
    assert protocol["REAL_DATA_MAIN_EXPERIMENT_AUTHORIZED"] is False
    assert protocol["MEASUREMENT_PAPER_MAINLINE_AUTHORIZED"] is False
