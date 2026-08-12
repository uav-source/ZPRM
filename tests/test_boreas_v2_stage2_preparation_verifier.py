from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import phase_a_harness.real_data_preparation.boreas_v2_stage2_preparation_verifier as verifier_module

from phase_a_harness.real_data_preparation.boreas_v2_stage2_preparation_verifier import (
    CANONICAL_INPUT_FIELDS,
    CANONICAL_SOURCE_VERIFICATION_FIELDS,
    DOWNLOAD_RECEIPT_SCHEMA,
    FIRST_PASS_SELECTION_PROJECTION_FIELDS,
    GEOMETRY_METRIC_VERIFICATION_FIELDS,
    PAYLOAD_FILES,
    RECEIPT_FIELDS,
    REQUIRED_UNKNOWN_COMPONENTS,
    UNCERTAINTY_FIELDS,
    BoreasStage2PreparationVerificationError,
    PreparationVerificationAuthority,
    verify_boreas_v2_stage2_preparation,
)
from phase_a_harness.real_data_preparation.boreas_stage2_external_voxel import (
    compile_external_voxel_reducer,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    FIRST_PASS_SCAN_FIELDS,
    RICH_LABEL,
    SYNTHETIC_AUTHORITY,
    WEAK_LABEL,
    ReferencePoseSeries,
    SelectionBindings,
    Stage2SelectionContract,
    build_blind_selection_manifest,
    build_candidate_intervals,
    build_candidate_scan_inventory,
    candidate_interval_csv_rows,
    candidate_scan_csv_rows,
    geometry_metric_csv_rows,
    select_frozen_intervals,
    select_interval_quantile_snapshots,
    selected_interval_csv_rows,
    selected_snapshot_csv_rows,
)
from phase_a_harness.real_data_preparation.io import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_bytes,
    compact_sha256,
    sha256_file,
)


MAP_SEQUENCE = "map-synthetic"
QUERY_SEQUENCE = "query-synthetic"
BACKEND_SHA = "b" * 64
MAP_COUNT = 2
MAP_REPLAY_POINTS_PER_SCAN = 5_000
WINDOW_COUNT = 20
SCANS_PER_WINDOW = 5
QUERY_COUNT = WINDOW_COUNT * SCANS_PER_WINDOW
TARGET_POINT_COUNT = 10_000


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def test_summary_nested_resource_schema_rejects_resigned_extra_field() -> None:
    resources = {
        "target_context": {
            "capacity_measurement": {},
            "capacity_measurement_file_sha256": "a" * 64,
            "capacity_measurement_provenance": {},
            "capacity_measurement_provenance_file_sha256": "b" * 64,
            "resource_plan": {},
            "resource_plan_file_sha256": "c" * 64,
        },
        "reducer": {
            "capacity_layout_verification": {},
            "capacity_layout_verification_file_sha256": "d" * 64,
            "capacity_replay_binding": {},
            "capacity_replay_binding_file_sha256": "e" * 64,
            "resource_plan": {},
            "resource_plan_file_sha256": "f" * 64,
        },
        "target_map_content_address": {
            "metadata": {},
            "metadata_file_sha256": "1" * 64,
            "target_map_path": "target_maps/a/target_points.npy",
            "target_map_sha256": "2" * 64,
        },
        "runtime_reverification_dependencies": {
            "authenticated_replay_ledger_path": "map/replay_ledger.jsonl",
            "authenticated_replay_ledger_sha256": "3" * 64,
            "authenticated_replay_path": "map/transformed_xyz.f64le",
            "authenticated_replay_size_bytes": 24,
            "retention_policy": "fixture",
        },
    }
    verifier_module._verify_summary_resource_exact_schema(resources)
    resources["target_context"]["resigned_extra"] = True
    with pytest.raises(
        BoreasStage2PreparationVerificationError, match="exact schema"
    ):
        verifier_module._verify_summary_resource_exact_schema(resources)


def _canonical_npy(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(value, dtype="<f8"),
        version=(1, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _synthetic_map_active_points(ordinal: int) -> np.ndarray:
    x = np.arange(MAP_REPLAY_POINTS_PER_SCAN, dtype=np.float64) + ordinal * 0.5
    return np.ascontiguousarray(
        np.column_stack((x, np.zeros_like(x), np.zeros_like(x))), dtype="<f8"
    )


def _synthetic_target_points() -> np.ndarray:
    points = np.concatenate(
        [_synthetic_map_active_points(ordinal) for ordinal in range(MAP_COUNT)]
    )
    # Every point occupies a distinct 0.25 m voxel.  This is the same frozen
    # lexicographic voxel-key order used by the production reducer.
    keys = np.floor(points / 0.25).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return np.ascontiguousarray(points[order], dtype="<f8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value, overwrite=True)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), [dict(row) for row in reader]


def _write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    atomic_write_csv(path, rows, fields, overwrite=True)


def _windows() -> list[dict[str, int]]:
    return [
        {
            "window_index": index,
            "interval_index": index,
            "start_time_us": index * 20_000_000,
            "end_time_us": index * 20_000_000 + 5_000_000,
            "duration_us": 5_000_000,
        }
        for index in range(WINDOW_COUNT)
    ]


def _write_reference_poses(path: Path) -> tuple[str, ReferencePoseSeries]:
    fields = [
        "GPSTime",
        "easting",
        "northing",
        "altitude",
        "vel_east",
        "vel_north",
        "vel_up",
        "roll",
        "pitch",
        "heading",
    ]
    rows: list[dict[str, Any]] = []
    timestamps: list[int] = []
    translations: list[list[float]] = []
    offsets = (500_000, 1_500_000, 2_500_000, 3_500_000, 4_500_000)
    for index, window in enumerate(_windows()):
        translation = [index * 2.0, 0.0, 0.0]
        for offset in offsets:
            timestamp = window["start_time_us"] + offset
            timestamps.append(timestamp)
            translations.append(translation)
            rows.append(
                {
                    "GPSTime": timestamp,
                    "easting": translation[0],
                    "northing": translation[1],
                    "altitude": translation[2],
                    "vel_east": 0.0,
                    "vel_north": 0.0,
                    "vel_up": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "heading": 0.0,
                }
            )
    _write_csv(path, fields, rows)
    return sha256_file(path), ReferencePoseSeries(
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(translations, dtype=np.float64),
        np.tile([0.0, 0.0, 0.0, 1.0], (QUERY_COUNT, 1)),
    )


def _spectral_fields(index: int) -> dict[str, float]:
    unnormalized = np.asarray([0.01 + index / 1000.0, 0.30, 0.60])
    eigenvalues = unnormalized / np.sum(unnormalized)
    normalized = eigenvalues.copy()
    entropy = -float(np.sum(normalized * np.log(normalized))) / np.log(3.0)
    return {
        "lambda_min_trans": float(eigenvalues[0]),
        "lambda_mid_trans": float(eigenvalues[1]),
        "lambda_max_trans": float(eigenvalues[2]),
        "normalized_lambda_min_trans": float(normalized[0]),
        "normalized_lambda_mid_trans": float(normalized[1]),
        "normalized_lambda_max_trans": float(normalized[2]),
        "condition_number_trans": float(eigenvalues[2] / eigenvalues[0]),
        "spectral_entropy_trans": entropy,
    }


def _resign(frozen: Path, *, include_extra_payloads: bool = False) -> None:
    selection_path = frozen / "boreas_v2_stage2_selection_manifest.json"
    selection = _read_json(selection_path)
    for name in list(selection["artifact_sha256"]):
        selection["artifact_sha256"][name] = sha256_file(frozen / name)
    unsigned_selection = dict(selection)
    unsigned_selection.pop("selection_manifest_sha256", None)
    selection["selection_manifest_sha256"] = _hash_json(unsigned_selection)
    _write_json(selection_path, selection)

    manifest_path = frozen / "boreas_v2_stage2_frozen_manifest.json"
    manifest = _read_json(manifest_path)
    payload_names = sorted(PAYLOAD_FILES)
    if include_extra_payloads:
        payload_names = sorted(
            path.name
            for path in frozen.iterdir()
            if path.name
            not in {"SHA256SUMS", "boreas_v2_stage2_frozen_manifest.json"}
        )
    manifest["payload"] = [
        {
            "path": name,
            "sha256": sha256_file(frozen / name),
            "size_bytes": (frozen / name).stat().st_size,
        }
        for name in payload_names
    ]
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_root_sha256", None)
    manifest["manifest_root_sha256"] = compact_sha256(unsigned_manifest)
    _write_json(manifest_path, manifest)

    names = sorted(path.name for path in frozen.iterdir() if path.name != "SHA256SUMS")
    lines = "".join(f"{sha256_file(frozen / name)}  {name}\n" for name in names)
    atomic_write_bytes(frozen / "SHA256SUMS", lines.encode("utf-8"), overwrite=True)


def _build_synthetic_closure(
    base: Path,
    *,
    coherent_target_delta_m: float = 0.0,
    coherent_resource_tamper: str | None = None,
) -> tuple[Path, Path, Path]:
    frozen = base / "frozen"
    runtime = base / "runtime"
    authority_root = base / "authority"
    frozen.mkdir(parents=True)
    runtime.mkdir(parents=True)
    authority_root.mkdir(parents=True)
    (runtime / "temporary").mkdir()

    stage1_manifest = authority_root / "stage1_manifest.json"
    storage_manifest = authority_root / "storage_manifest.json"
    preprocessing = authority_root / "preprocessing.json"
    reference_pose = authority_root / "lidar_poses.csv"
    map_reference_pose = authority_root / "map_lidar_poses.csv"
    _write_json(stage1_manifest, {"schema": "synthetic-stage1"})
    _write_json(storage_manifest, {"schema": "synthetic-storage"})
    preprocessing_value = {
        "schema": "boreas_v2_stage2_preprocessing_contract_v1",
        "target_map_voxel_size_m": 0.25,
        "source_voxel_size_m": 0.10,
        "deskew": {"enabled": True, "algorithm": "FROZEN_SE3_INTERPOLATION"},
    }
    _write_json(preprocessing, preprocessing_value)
    preprocessing_markdown = preprocessing.with_suffix(".md")
    atomic_write_bytes(
        preprocessing_markdown,
        b"# Synthetic frozen preprocessing contract\n",
    )
    reference_pose_sha, pose_series = _write_reference_poses(reference_pose)
    atomic_write_bytes(map_reference_pose, reference_pose.read_bytes())
    map_reference_pose_sha = sha256_file(map_reference_pose)
    extrinsic_path = authority_root / "T_applanix_lidar.txt"
    atomic_write_bytes(extrinsic_path, b"synthetic frozen extrinsic\n")
    extrinsic_sha = sha256_file(extrinsic_path)
    chain_consistency_path = authority_root / "transform_chain_consistency.json"
    _write_json(chain_consistency_path, {"status": "PASS"})
    chain_manifest_path = authority_root / "transform_chain_manifest.json"
    _write_json(
        chain_manifest_path,
        {
            "composition": "T_ENU_lidar(t)=T_ENU_applanix(t)@T_applanix_lidar",
            "consistency_sha256": sha256_file(chain_consistency_path),
            "direction_verified_against_official_lidar_poses": True,
            "inverse_used": False,
            "source_sha256": extrinsic_sha,
            "status": "PASS",
        },
    )
    witness_path = (
        Path(__file__).resolve().parents[1]
        / "src/phase_a_harness/real_data_preparation/boreas_v2_stage2_canonical_witness.py"
    )
    witness_sha = sha256_file(witness_path)
    preprocessing_value["primary_pair"] = {
        "map_lidar_pose_sha256": map_reference_pose_sha,
        "map_sequence_id": MAP_SEQUENCE,
        "query_lidar_pose_sha256": reference_pose_sha,
        "query_sequence_id": QUERY_SEQUENCE,
        "static_t_applanix_lidar_sha256": extrinsic_sha,
    }
    preprocessing_value["implementation_bindings"] = {
        "independent_canonical_source_witness": {
            "file_sha256": witness_sha,
            "path": str(witness_path),
        }
    }
    _write_json(preprocessing, preprocessing_value)

    pair_path = authority_root / "pair.json"
    pair = {
        "PRIMARY_PAIR": {
            "map_sequence_id": MAP_SEQUENCE,
            "query_sequence_id": QUERY_SEQUENCE,
            "query_calibration_sha256": extrinsic_sha,
        },
        "primary_complete_five_second_windows": _windows(),
    }
    _write_json(pair_path, pair)

    map_allowlist: list[dict[str, Any]] = []
    for index in range(MAP_COUNT):
        timestamp = 1_000_000 + index * 100_000
        map_allowlist.append(
            {
                "selection_role": "TARGET_MAP",
                "sequence_id": MAP_SEQUENCE,
                "key": f"{MAP_SEQUENCE}/lidar/{timestamp}.bin",
                "timestamp_us": timestamp,
                "last_modified": "2026-01-01T00:00:00Z",
                "size_bytes": MAP_REPLAY_POINTS_PER_SCAN * 24,
                "selection_reason": "synthetic frozen GT overlap",
            }
        )
    query_allowlist: list[dict[str, Any]] = []
    offsets = (500_000, 1_500_000, 2_500_000, 3_500_000, 4_500_000)
    for window in _windows():
        for offset in offsets:
            timestamp = window["start_time_us"] + offset
            query_allowlist.append(
                {
                    "selection_role": "QUERY",
                    "sequence_id": QUERY_SEQUENCE,
                    "key": f"{QUERY_SEQUENCE}/lidar/{timestamp}.bin",
                    "timestamp_us": timestamp,
                    "last_modified": "2026-01-01T00:00:00Z",
                    "size_bytes": 24_000,
                    "selection_reason": "synthetic frozen complete five-second window",
                }
            )
    allowlist_path = authority_root / "allowlist.csv"
    allowlist_fields = (
        "selection_role",
        "sequence_id",
        "key",
        "timestamp_us",
        "last_modified",
        "size_bytes",
        "selection_reason",
    )
    _write_csv(allowlist_path, allowlist_fields, map_allowlist + query_allowlist)

    target_points = _synthetic_target_points()
    if coherent_target_delta_m:
        target_points = target_points.copy()
        target_points[0, 0] += coherent_target_delta_m
    target_payload = _canonical_npy(target_points)
    target_sha = hashlib.sha256(target_payload).hexdigest()
    target_relative = f"target_maps/{target_sha}/target_points.npy"
    target_path = runtime / target_relative
    target_path.parent.mkdir(parents=True)
    atomic_write_bytes(target_path, target_payload)
    target_path.chmod(0o444)

    contract = Stage2SelectionContract(
        parameter_authority=SYNTHETIC_AUTHORITY,
        expected_candidate_scan_count=QUERY_COUNT,
        expected_candidate_interval_count=WINDOW_COUNT,
    )
    preprocessing_sha = sha256_file(preprocessing)
    bindings = SelectionBindings(
        primary_query_sequence_id=QUERY_SEQUENCE,
        gt_sha256=reference_pose_sha,
        calibration_sha256=extrinsic_sha,
        preprocessing_contract_sha256=preprocessing_sha,
        target_map_sha256=target_sha,
    )
    first_pass: list[dict[str, Any]] = []
    query_payload_sha: dict[str, str] = {}
    for ordinal, allow in enumerate(query_allowlist):
        interval_index = ordinal // SCANS_PER_WINDOW
        payload_sha = hashlib.sha256(allow["key"].encode("utf-8")).hexdigest()
        query_payload_sha[allow["key"]] = payload_sha
        row = {
            "query_ordinal": ordinal,
            "sequence_id": QUERY_SEQUENCE,
            "object_key": allow["key"],
            "timestamp_us": allow["timestamp_us"],
            "remote_size_bytes": allow["size_bytes"],
            "last_modified": allow["last_modified"],
            "etag": f"synthetic-etag-{ordinal:03d}",
            "payload_sha256": payload_sha,
            "finite_source_point_count": 1000,
            "target_map_point_count": TARGET_POINT_COUNT,
            "reference_interpolation_valid": True,
            "reference_gap_s": 0.0,
            "gt_overlap_within_5m": True,
            "target_map_frozen_complete": True,
            "deskew_processing_contract_valid": True,
            "gt_sha256": reference_pose_sha,
            "calibration_sha256": extrinsic_sha,
            "preprocessing_contract_sha256": preprocessing_sha,
            "target_map_sha256": target_sha,
            "initial_correspondence_count": 1000,
            "initial_valid_normal_correspondence_count": 900,
            **_spectral_fields(interval_index),
        }
        assert set(row) == set(FIRST_PASS_SCAN_FIELDS)
        first_pass.append(row)

    scans, excluded_scans = build_candidate_scan_inventory(
        first_pass, _windows(), bindings=bindings, contract=contract
    )
    intervals, excluded_intervals = build_candidate_intervals(
        scans, _windows(), pose_series, contract=contract
    )
    selected_intervals = select_frozen_intervals(intervals, contract=contract)
    selected = select_interval_quantile_snapshots(
        scans, selected_intervals, contract=contract
    )
    blind = build_blind_selection_manifest(
        candidate_scans=scans,
        candidate_intervals=intervals,
        selected_intervals=selected_intervals,
        selected_snapshots=selected,
        contract=contract,
        bindings=bindings,
    )

    from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
        CANDIDATE_INTERVAL_CSV_FIELDS,
        CANDIDATE_SCAN_CSV_FIELDS,
        GEOMETRY_METRIC_CSV_FIELDS,
        SELECTED_INTERVAL_CSV_FIELDS,
        SELECTED_SNAPSHOT_CSV_FIELDS,
    )

    _write_csv(
        frozen / "all_candidate_scans.csv",
        CANDIDATE_SCAN_CSV_FIELDS,
        candidate_scan_csv_rows(scans),
    )
    _write_csv(
        frozen / "excluded_candidate_scans.csv",
        CANDIDATE_SCAN_CSV_FIELDS,
        candidate_scan_csv_rows(excluded_scans),
    )
    _write_csv(
        frozen / "geometry_only_metrics.csv",
        GEOMETRY_METRIC_CSV_FIELDS,
        geometry_metric_csv_rows(scans),
    )
    _write_csv(
        frozen / "all_candidate_intervals.csv",
        CANDIDATE_INTERVAL_CSV_FIELDS,
        candidate_interval_csv_rows(intervals),
    )
    _write_csv(
        frozen / "excluded_intervals.csv",
        CANDIDATE_INTERVAL_CSV_FIELDS,
        candidate_interval_csv_rows(excluded_intervals),
    )
    _write_csv(
        frozen / "selected_scene_intervals.csv",
        SELECTED_INTERVAL_CSV_FIELDS,
        selected_interval_csv_rows(selected_intervals),
    )
    _write_csv(
        frozen / "selected_snapshots.csv",
        SELECTED_SNAPSHOT_CSV_FIELDS,
        selected_snapshot_csv_rows(selected),
    )

    source_template = np.zeros((1000, 3), dtype=np.float64)
    source_template[:, 0] = np.arange(1000) / 1000.0
    source_payload = _canonical_npy(source_template)
    source_sha = hashlib.sha256(source_payload).hexdigest()

    candidate_by_key = {row["object_key"]: row for row in scans}

    def receipt_row(
        *,
        stage: str,
        role: str,
        allow: dict[str, Any],
        etag: str,
        payload_sha: str,
        processing_result_sha: str,
        event_index: int,
    ) -> dict[str, Any]:
        native = {
            "downloaded_at_utc": f"2026-01-01T00:00:00.{event_index:06d}Z",
            "etag": etag,
            "key": allow["key"],
            "last_modified": allow["last_modified"],
            "local_temporary_sha256": payload_sha,
            "remote_size_bytes": allow["size_bytes"],
            "schema": DOWNLOAD_RECEIPT_SCHEMA,
            "selection_role": role,
            "sequence_id": allow["sequence_id"],
            "timestamp_us": allow["timestamp_us"],
        }
        return {
            "execution_stage": stage,
            **native,
            "receipt_sha256": _hash_json(native),
            "processing_result_sha256": processing_result_sha,
            "preprocessing_contract_sha256": preprocessing_sha,
            "gt_sha256": (
                map_reference_pose_sha if role == "TARGET_MAP" else reference_pose_sha
            ),
            "extrinsic_sha256": extrinsic_sha,
            "checkpoint_status": "COMMITTED",
        }

    receipt_rows: list[dict[str, Any]] = []
    for map_ordinal, allow in enumerate(map_allowlist):
        payload_sha = hashlib.sha256(allow["key"].encode()).hexdigest()
        receipt_rows.append(
            receipt_row(
                stage="MAP_INGEST",
                role="TARGET_MAP",
                allow=allow,
                etag=f"map-etag-{allow['timestamp_us']}",
                payload_sha=payload_sha,
                processing_result_sha=hashlib.sha256(
                    f"replay:{allow['key']}".encode()
                ).hexdigest(),
                event_index=map_ordinal,
            )
        )
    query_by_key = {row["key"]: row for row in query_allowlist}
    for stage, rows in (
        ("QUERY_GEOMETRY_FIRST_PASS", query_allowlist),
        (
            "QUERY_CANONICAL_SECOND_PASS",
            [query_by_key[row["object_key"]] for row in selected],
        ),
    ):
        for allow in rows:
            ordinal = next(
                index for index, child in enumerate(query_allowlist) if child["key"] == allow["key"]
            )
            receipt_rows.append(
                receipt_row(
                    stage=stage,
                    role="QUERY",
                    allow=allow,
                    etag=f"synthetic-etag-{ordinal:03d}",
                    payload_sha=query_payload_sha[allow["key"]],
                    processing_result_sha=(
                        candidate_by_key[allow["key"]]["candidate_scan_row_sha256"]
                        if stage == "QUERY_GEOMETRY_FIRST_PASS"
                        else source_sha
                    ),
                    event_index=len(receipt_rows),
                )
            )
    map_receipts = {
        row["key"]: row
        for row in receipt_rows
        if row["execution_stage"] == "MAP_INGEST"
    }
    replay_relative = "map_replay/transformed_xyz.f8"
    ledger_relative = "map_replay/ledger.jsonl"
    replay_path = runtime / replay_relative
    ledger_path = runtime / ledger_relative
    replay_path.parent.mkdir(parents=True)
    voxel_rule_sha = "2" * 64
    descriptors: list[dict[str, Any]] = []
    byte_start = 0
    for ordinal, allow in enumerate(map_allowlist):
        size = int(allow["size_bytes"])
        descriptors.append(
            {
                "etag": map_receipts[allow["key"]]["etag"],
                "last_modified": allow["last_modified"],
                "object_key": allow["key"],
                "ordinal": ordinal,
                "remote_size_bytes": size,
                "byte_end_exclusive": byte_start + size,
                "byte_start": byte_start,
                "point_capacity": size // 24,
            }
        )
        byte_start += size
    plan = {
        "allowlist": descriptors,
        "allowlist_sha256": _hash_json(descriptors),
        "boreas_raw_point_stride_bytes": 24,
        "calibration_sha256": extrinsic_sha,
        "gt_sha256": map_reference_pose_sha,
        "numeric_format": "LITTLE_ENDIAN_FLOAT64_XYZ_C_ORDER_NO_HEADER",
        "processing_contract_sha256": preprocessing_sha,
        "padding_rule": "ZERO_FLOAT64_XYZ_UNUSED_SUFFIX_V1",
        "replay_path": str(replay_path),
        "replay_point_stride_bytes": 24,
        "track_python_voxel_state": False,
        "total_point_capacity": sum(row["point_capacity"] for row in descriptors),
        "total_replay_bytes": byte_start,
        "voxel_rule_sha256": voxel_rule_sha,
    }

    def compact_line(value: dict[str, Any]) -> bytes:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode()

    def ledger_envelope(
        sequence: int, previous: str, record_type: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        core = {
            "payload": payload,
            "previous_record_sha256": previous,
            "record_type": record_type,
            "schema": "zprm.stage2.production_map_replay_ledger.v2",
            "sequence_number": sequence,
        }
        return {**core, "record_sha256": hashlib.sha256(compact_line(core)).hexdigest()}

    plan_envelope = ledger_envelope(1, "0" * 64, "PLAN", plan)
    envelopes = [plan_envelope]
    replay_chunks: list[bytes] = []
    lineage_rows: list[dict[str, Any]] = []
    previous_map_transition = "0" * 64
    previous_replay_transition = "0" * 64
    for ordinal, (descriptor, allow) in enumerate(zip(descriptors, map_allowlist)):
        active = _synthetic_map_active_points(ordinal).tobytes(order="C")
        replay_range = active + bytes(int(allow["size_bytes"]) - len(active))
        replay_chunks.append(replay_range)
        range_sha = hashlib.sha256(replay_range).hexdigest()
        active_sha = hashlib.sha256(active).hexdigest()
        receipt = map_receipts[allow["key"]]
        receipt["processing_result_sha256"] = range_sha
        payload: dict[str, Any] = {
            "byte_end_exclusive": descriptor["byte_end_exclusive"],
            "byte_start": descriptor["byte_start"],
            "calibration_sha256": extrinsic_sha,
            "completed_at_utc": f"2026-01-01T01:00:00.{ordinal:06d}Z",
            "etag": receipt["etag"],
            "gt_sha256": map_reference_pose_sha,
            "last_modified": allow["last_modified"],
            "local_temporary_sha256": receipt["local_temporary_sha256"],
            "map_state_transition_kind": "AUTHENTICATED_REPLAY_RANGE_CHAIN_V1",
            "object_key": allow["key"],
            "ordinal": ordinal,
            "padding_point_count": descriptor["point_capacity"]
            - MAP_REPLAY_POINTS_PER_SCAN,
            "padding_rule": "ZERO_FLOAT64_XYZ_UNUSED_SUFFIX_V1",
            "point_count": MAP_REPLAY_POINTS_PER_SCAN,
            "previous_map_state_transition_sha256": previous_map_transition,
            "previous_replay_state_transition_sha256": previous_replay_transition,
            "processing_contract_sha256": preprocessing_sha,
            "remote_size_bytes": int(allow["size_bytes"]),
            "replay_range_sha256": range_sha,
            "transformed_xyz_sha256": active_sha,
        }
        map_core = {
            key: payload[key]
            for key in (
                "byte_end_exclusive",
                "byte_start",
                "calibration_sha256",
                "etag",
                "gt_sha256",
                "last_modified",
                "local_temporary_sha256",
                "map_state_transition_kind",
                "object_key",
                "ordinal",
                "padding_point_count",
                "padding_rule",
                "point_count",
                "previous_map_state_transition_sha256",
                "processing_contract_sha256",
                "remote_size_bytes",
                "replay_range_sha256",
                "transformed_xyz_sha256",
            )
        }
        payload["map_state_transition_sha256"] = _hash_json(map_core)
        payload["replay_state_transition_sha256"] = _hash_json(payload)
        envelope = ledger_envelope(
            ordinal + 2,
            envelopes[-1]["record_sha256"],
            "SCAN",
            payload,
        )
        envelopes.append(envelope)
        lineage_rows.append(
            {
                "map_ordinal": ordinal,
                "object_key": allow["key"],
                "receipt_sha256": receipt["receipt_sha256"],
                "raw_payload_sha256": receipt["local_temporary_sha256"],
                "processing_result_sha256": range_sha,
                "replay_range_sha256": range_sha,
                "replay_active_point_count": MAP_REPLAY_POINTS_PER_SCAN,
                "transformed_xyz_sha256": active_sha,
                "previous_map_state_transition_sha256": previous_map_transition,
                "map_state_transition_sha256": payload["map_state_transition_sha256"],
                "replay_state_transition_sha256": payload[
                    "replay_state_transition_sha256"
                ],
                "replay_ledger_record_sha256": envelope["record_sha256"],
            }
        )
        previous_map_transition = payload["map_state_transition_sha256"]
        previous_replay_transition = payload["replay_state_transition_sha256"]
    atomic_write_bytes(replay_path, b"".join(replay_chunks))
    atomic_write_bytes(ledger_path, b"".join(compact_line(row) for row in envelopes))

    def native_receipt(row: dict[str, Any]) -> dict[str, Any]:
        return {
            field: (int(row[field]) if field in {"remote_size_bytes", "timestamp_us"} else row[field])
            for field in (
                "downloaded_at_utc",
                "etag",
                "key",
                "last_modified",
                "local_temporary_sha256",
                "remote_size_bytes",
                "receipt_sha256",
                "schema",
                "selection_role",
                "sequence_id",
                "timestamp_us",
            )
        }

    def chained_event(
        rows: list[dict[str, Any]],
        schema: str,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        unsigned = {
            **event,
            "previous_event_sha256": rows[-1]["event_sha256"] if rows else "0" * 64,
            "schema": schema,
            "sequence_number": len(rows) + 1,
        }
        result = {
            **unsigned,
            "event_sha256": hashlib.sha256(compact_line(unsigned)).hexdigest(),
        }
        rows.append(result)
        return result

    map_journal_rows: list[dict[str, Any]] = []
    for ordinal, receipt in enumerate(receipt_rows[:MAP_COUNT]):
        download = chained_event(
            map_journal_rows,
            "zprm.boreas.v2.stage2.receipt_checkpoint.v1",
            {
                "event_kind": "DOWNLOADED",
                "execution_stage": "MAP_INGEST",
                "extrinsic_sha256": extrinsic_sha,
                "gt_sha256": map_reference_pose_sha,
                "preprocessing_contract_sha256": preprocessing_sha,
                "receipt": native_receipt(receipt),
            },
        )
        preprocessed = chained_event(
            map_journal_rows,
            "zprm.boreas.v2.stage2.receipt_checkpoint.v1",
            {
                "canonical_source_witness_sha256": hashlib.sha256(
                    f"map-source:{receipt['key']}".encode()
                ).hexdigest(),
                "download_event_sha256": download["event_sha256"],
                "event_kind": "PREPROCESSED",
                "execution_stage": "MAP_INGEST",
                "key": receipt["key"],
                "receipt_sha256": receipt["receipt_sha256"],
                "source_point_count": MAP_REPLAY_POINTS_PER_SCAN,
                "source_witness_metadata": {"synthetic": True},
            },
        )
        chained_event(
            map_journal_rows,
            "zprm.boreas.v2.stage2.receipt_checkpoint.v1",
            {
                "checkpoint_status": "COMMITTED",
                "download_event_sha256": download["event_sha256"],
                "event_kind": "REPLAY_COMMITTED",
                "execution_stage": "MAP_INGEST",
                "key": receipt["key"],
                "preprocessed_event_sha256": preprocessed["event_sha256"],
                "processing_result_sha256": receipt["processing_result_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
                "replay_ordinal": ordinal,
            },
        )
    checkpoint_root = runtime / "checkpoints"
    checkpoint_root.mkdir()
    map_journal_relative = "checkpoints/receipt_checkpoints.jsonl"
    map_journal_path = runtime / map_journal_relative
    atomic_write_bytes(
        map_journal_path,
        b"".join(compact_line(row) for row in map_journal_rows),
    )
    map_receipt_rows = receipt_rows[:MAP_COUNT]
    _write_csv(frozen / "map_download_receipts.csv", RECEIPT_FIELDS, map_receipt_rows)
    map_download_bytes = sum(int(row["remote_size_bytes"]) for row in map_receipt_rows)
    _write_json(
        frozen / "MAP_LIDAR_DOWNLOAD_AUDIT.json",
        {
            "aborted_download_event_count": 0,
            "canonical_source_witness_count": MAP_COUNT,
            "committed_payload_bytes": map_download_bytes,
            "committed_payload_event_count": MAP_COUNT,
            "raw_payload_persistent_bytes": 0,
            "receipt_checkpoint_final_chain_sha256": map_journal_rows[-1]["event_sha256"],
            "receipt_checkpoint_journal_path": map_journal_relative,
            "receipt_checkpoint_journal_sha256": sha256_file(map_journal_path),
            "replay_committed_event_count": MAP_COUNT,
            "retry_download_event_count": 0,
            "retry_download_payload_bytes": 0,
            "schema": "zprm.boreas.v2.stage2.lidar_download_audit.v2",
            "stream_deleted_raw_bytes": map_download_bytes,
            "successful_download_event_count": MAP_COUNT,
            "successful_payload_bytes": map_download_bytes,
            "unique_allowlist_object_count": MAP_COUNT,
        },
    )
    _write_csv(frozen / "download_receipts.csv", RECEIPT_FIELDS, receipt_rows)

    lineage_core = {
        "schema_version": "boreas_v2_stage2_map_lineage_v1",
        "map_sequence_id": MAP_SEQUENCE,
        "source_object_count": MAP_COUNT,
        "query_contribution_count": 0,
        "preprocessing_contract_sha256": preprocessing_sha,
        "gt_sha256": map_reference_pose_sha,
        "extrinsic_sha256": extrinsic_sha,
        "replay_array_path": replay_relative,
        "replay_ledger_path": ledger_relative,
        "source_objects": lineage_rows,
        "final_map_state_transition_sha256": previous_map_transition,
    }
    _write_json(
        frozen / "map_lineage_manifest.json",
        {**lineage_core, "map_lineage_manifest_sha256": _hash_json(lineage_core)},
    )
    reducer_source_path = (
        Path(__file__).resolve().parents[1]
        / "src/phase_a_harness/real_data_preparation/boreas_stage2_voxel_reduce.cpp"
    )
    reducer_binary_relative = "tools/synthetic-external-voxel"
    reducer_binary_path = runtime / reducer_binary_relative
    reducer_binary_path.parent.mkdir(parents=True)
    compile_external_voxel_reducer(
        source=reducer_source_path,
        output=reducer_binary_path,
    )
    capacity_max_voxels = TARGET_POINT_COUNT
    constructor_capacity = 1024
    while (
        constructor_capacity * 7 // 10 < capacity_max_voxels
        and constructor_capacity < 2**20
    ):
        constructor_capacity *= 2
    table_capacity = 1024
    while table_capacity * 7 // 10 < capacity_max_voxels:
        table_capacity *= 2
    final_table_bytes = table_capacity * 56
    largest_old = table_capacity // 2 if table_capacity > constructor_capacity else 0
    peak_growth_table_bytes = (table_capacity + largest_old) * 56
    input_buffer_bytes = (1 << 18) * 24
    output_buffer_bytes = 4096 * 24
    fixed_overhead_bytes = 64 * 1024 * 1024
    sorted_index_upper_bytes = capacity_max_voxels * 8
    target_npy_upper_bytes = capacity_max_voxels * 24 + 4096
    capacity_layout = {
        "constructor_capacity": constructor_capacity,
        "final_table_bytes": final_table_bytes,
        "fixed_overhead_bytes": fixed_overhead_bytes,
        "input_buffer_bytes": input_buffer_bytes,
        "key_bytes": 12,
        "max_voxels": capacity_max_voxels,
        "maximum_load_denominator": 10,
        "maximum_load_numerator": 7,
        "output_buffer_bytes": output_buffer_bytes,
        "peak_growth_table_bytes": peak_growth_table_bytes,
        "schema": "zprm-boreas-stage2-reducer-capacity-layout-v1",
        "size_t_bytes": 8,
        "slot_bytes": 56,
        "sorted_index_upper_bytes": sorted_index_upper_bytes,
        "table_capacity": table_capacity,
        "target_npy_upper_bytes": target_npy_upper_bytes,
        "total_peak_upper_bound_bytes": max(
            peak_growth_table_bytes + input_buffer_bytes + fixed_overhead_bytes,
            final_table_bytes
            + sorted_index_upper_bytes
            + input_buffer_bytes
            + output_buffer_bytes
            + target_npy_upper_bytes
            + fixed_overhead_bytes,
        ),
    }
    if coherent_resource_tamper == "layout":
        capacity_layout["total_peak_upper_bound_bytes"] += 1
    capacity_layout_sha = _hash_json(capacity_layout)
    capacity_margin = 64 * 1024**2
    plan_core = {
        "capacity_probe_evidence_sha256": capacity_layout_sha,
        "capacity_probe_kind": "COMPILED_LAYOUT_EXACT_UPPER_BOUND_V1",
        "estimated_peak_memory_bytes": capacity_layout[
            "total_peak_upper_bound_bytes"
        ],
        "max_voxels": capacity_max_voxels,
        "minimum_live_available_memory_bytes": capacity_layout[
            "total_peak_upper_bound_bytes"
        ]
        + capacity_margin,
        "production_approved": True,
        "reducer_binary_projected_bytes": reducer_binary_path.stat().st_size,
        "reducer_binary_sha256": sha256_file(reducer_binary_path),
        "reducer_source_sha256": sha256_file(reducer_source_path),
        "safety_margin_bytes": capacity_margin,
        "schema": "zprm.boreas.v2.stage2.reducer_resource_plan.v1",
    }
    if coherent_resource_tamper == "plan":
        plan_core["reducer_binary_projected_bytes"] += 1
    resource_plan = {**plan_core, "plan_payload_sha256": _hash_json(plan_core)}
    resource_plan_path = runtime / "checkpoints/reducer_resource_plan.json"
    _write_json(resource_plan_path, resource_plan)
    layout_core = {
        "binary_path": reducer_binary_relative,
        "binary_sha256": sha256_file(reducer_binary_path),
        "capacity_layout": capacity_layout,
        "capacity_layout_sha256": capacity_layout_sha,
        "minimum_live_available_memory_bytes": resource_plan[
            "minimum_live_available_memory_bytes"
        ],
        "resource_plan_sha256": resource_plan["plan_payload_sha256"],
        "schema": "zprm.boreas.v2.stage2.reducer_capacity_layout_verification.v1",
        "source_path": "src/phase_a_harness/real_data_preparation/boreas_stage2_voxel_reduce.cpp",
        "source_sha256": sha256_file(reducer_source_path),
        "verification_status": "PASS_COMPILED_LAYOUT_AND_LIVE_MEMORY_GATE",
    }
    layout_evidence = {
        **layout_core,
        "verification_payload_sha256": _hash_json(layout_core),
    }
    layout_path = runtime / "checkpoints/reducer_capacity_layout_verification.json"
    _write_json(layout_path, layout_evidence)
    binding_core = {
        "capacity_layout_verification_file_sha256": sha256_file(layout_path),
        "replay_plan_sha256": _hash_json(plan),
        "resource_plan_sha256": resource_plan["plan_payload_sha256"],
        "schema": "zprm.boreas.v2.stage2.reducer_capacity_replay_binding.v1",
        "verification_status": "PASS_CAPACITY_LAYOUT_BOUND_TO_RECONCILED_REPLAY",
    }
    if coherent_resource_tamper == "binding":
        binding_core["replay_plan_sha256"] = "f" * 64
    if coherent_resource_tamper not in {None, "plan", "layout", "binding"}:
        raise AssertionError(coherent_resource_tamper)
    replay_binding = {
        **binding_core,
        "binding_payload_sha256": _hash_json(binding_core),
    }
    replay_binding_path = runtime / "checkpoints/reducer_capacity_replay_binding.json"
    _write_json(replay_binding_path, replay_binding)
    reducer_core = {
        "schema_version": "boreas_v2_stage2_target_reducer_verification_v1",
        "map_lineage_manifest_sha256": sha256_file(
            frozen / "map_lineage_manifest.json"
        ),
        "replay_ledger_path": ledger_relative,
        "replay_ledger_sha256": sha256_file(ledger_path),
        "replay_plan_sha256": _hash_json(plan),
        "replay_descriptor_count": MAP_COUNT,
        "replay_authenticated_ranges_sha256": _hash_json(
            [row["replay_range_sha256"] for row in lineage_rows]
        ),
        "replay_total_bytes": byte_start,
        "reducer_implementation_path": "src/phase_a_harness/real_data_preparation/boreas_stage2_voxel_reduce.cpp",
        "reducer_implementation_sha256": sha256_file(reducer_source_path),
        "reducer_binary_path": reducer_binary_relative,
        "reducer_binary_sha256": sha256_file(reducer_binary_path),
        "reducer_capacity_layout_verification_file_sha256": sha256_file(
            layout_path
        ),
        "reducer_capacity_replay_binding_file_sha256": sha256_file(
            replay_binding_path
        ),
        "reducer_resource_plan_file_sha256": sha256_file(resource_plan_path),
        "producer_target_map_sha256": target_sha,
        "reducer_output_target_map_sha256": target_sha,
        "target_map_size_bytes": len(target_payload),
        "target_point_count": TARGET_POINT_COUNT,
        "final_map_state_transition_sha256": previous_map_transition,
        "verification_scope": (
            "PRODUCER_REDUCER_LINEAGE_AND_OUTPUT_BINDINGS_"
            "FINAL_VERIFIER_REPLAYS_EXACT_TARGET_BYTES"
        ),
        "verification_status": "PASS_PRODUCER_BINDINGS_AWAITING_FINAL_EXACT_REPLAY",
    }
    _write_json(
        frozen / "target_map_reducer_verification.json",
        {
            **reducer_core,
            "target_map_reducer_verification_sha256": _hash_json(reducer_core),
        },
    )
    _write_json(
        frozen / "target_map_freeze_manifest.json",
        {
            "schema_version": "boreas_v2_stage2_target_map_freeze_v1",
            "map_lineage_manifest_sha256": sha256_file(
                frozen / "map_lineage_manifest.json"
            ),
            "preprocessing_contract_sha256": preprocessing_sha,
            "target_map_reducer_verification_sha256": sha256_file(
                frozen / "target_map_reducer_verification.json"
            ),
            "unique_target_map_count": 1,
            "physical_target_map_copy_count": 1,
            "query_contribution_count": 0,
            "target_map_immutable": True,
            "target_map_path": target_relative,
            "target_map_sha256": target_sha,
            "target_map_size_bytes": len(target_payload),
            "target_point_count": TARGET_POINT_COUNT,
            "voxel_rule_sha256": voxel_rule_sha,
            "final_map_state_transition_sha256": previous_map_transition,
        },
    )

    canonical_rows: list[dict[str, Any]] = []
    witness_rows: list[dict[str, Any]] = []
    expected_t: dict[str, str] = {}
    receipts_by_stage_key = {
        (row["execution_stage"], row["key"]): row for row in receipt_rows
    }
    query_journal_rows: list[dict[str, Any]] = []
    target_context_measurement_path = (
        runtime / "evidence/target_context_capacity_measurement.json"
    )
    target_context_plan_path = runtime / "evidence/target_context_resource_plan.json"
    target_context_provenance_path = (
        runtime / "evidence/target_context_capacity_measurement_provenance.json"
    )
    _write_json(
        target_context_measurement_path,
        {
            "measurement_method": "SYNTHETIC_FIXTURE_FIXED",
            "schema": "zprm.boreas.v2.stage2.target_context_capacity_measurement.v1",
            "target_map_sha256": target_sha,
            "target_point_count": TARGET_POINT_COUNT,
        },
    )
    _write_json(
        target_context_plan_path,
        {
            "measurement_evidence_sha256": sha256_file(
                target_context_measurement_path
            ),
            "production_approved": False,
            "schema": "zprm.boreas.v2.stage2.target_context_resource_plan.v1",
        },
    )
    _write_json(
        target_context_provenance_path,
        {
            "measurement_evidence_sha256": sha256_file(
                target_context_measurement_path
            ),
            "schema": (
                "zprm.boreas.v2.stage2."
                "target_context_measurement_provenance.v1"
            ),
            "target_map_sha256": target_sha,
        },
    )
    target_barrier = chained_event(
        query_journal_rows,
        "zprm.boreas.v2.stage2.query_journal.v1",
        {
            "event_kind": "TARGET_FROZEN",
            "execution_stage": "QUERY_BARRIER",
            "object_key": "",
            "payload": {
                "target_freeze_file_sha256": sha256_file(
                    frozen / "target_map_freeze_manifest.json"
                ),
                "target_map_sha256": target_sha,
                "target_context_resource_plan_sha256": sha256_file(
                    target_context_plan_path
                ),
                "target_context_measurement_evidence_sha256": sha256_file(
                    target_context_measurement_path
                ),
                "target_context_measurement_provenance_sha256": sha256_file(
                    target_context_provenance_path
                ),
            },
        },
    )
    geometry_witness_rows: list[dict[str, Any]] = []
    first_pass_commits: list[dict[str, Any]] = []
    for ordinal, (allow, first_row) in enumerate(zip(query_allowlist, first_pass)):
        receipt = receipts_by_stage_key[("QUERY_GEOMETRY_FIRST_PASS", allow["key"])]
        relative = f"temporary/query-first-{ordinal:05d}.bin.partial"
        intent = chained_event(
            query_journal_rows,
            "zprm.boreas.v2.stage2.query_journal.v1",
            {
                "event_kind": "TRANSFER_INTENT",
                "execution_stage": "QUERY_GEOMETRY_FIRST_PASS",
                "object_key": allow["key"],
                "payload": {
                    "etag": receipt["etag"],
                    "last_modified": allow["last_modified"],
                    "remote_size_bytes": int(allow["size_bytes"]),
                    "sequence_id": allow["sequence_id"],
                    "temporary_relative_path": relative,
                    "timestamp_us": int(allow["timestamp_us"]),
                },
            },
        )
        download = chained_event(
            query_journal_rows,
            "zprm.boreas.v2.stage2.query_journal.v1",
            {
                "event_kind": "DOWNLOADED",
                "execution_stage": "QUERY_GEOMETRY_FIRST_PASS",
                "object_key": allow["key"],
                "payload": {
                    "receipt": native_receipt(receipt),
                    "temporary_relative_path": relative,
                    "transfer_intent_event_sha256": intent["event_sha256"],
                },
            },
        )
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = (ordinal // SCANS_PER_WINDOW) * 2.0
        transform_sha = hashlib.sha256(_canonical_npy(transform)).hexdigest()
        geometry = {field: first_row[field] for field in (
            "initial_correspondence_count",
            "initial_valid_normal_correspondence_count",
            "lambda_min_trans",
            "lambda_mid_trans",
            "lambda_max_trans",
            "normalized_lambda_min_trans",
            "normalized_lambda_mid_trans",
            "normalized_lambda_max_trans",
            "condition_number_trans",
            "spectral_entropy_trans",
        )}
        geometry_sha = _hash_json(geometry)
        geometry_core = {
            "query_ordinal": ordinal,
            "object_key": allow["key"],
            "raw_payload_sha256": receipt["local_temporary_sha256"],
            "receipt_sha256": receipt["receipt_sha256"],
            "preprocessing_contract_sha256": preprocessing_sha,
            "gt_sha256": reference_pose_sha,
            "extrinsic_sha256": extrinsic_sha,
            "target_map_sha256": target_sha,
            "producer_source_sha256": source_sha,
            "independent_source_sha256": source_sha,
            "producer_T_reference_sha256": transform_sha,
            "independent_T_reference_sha256": transform_sha,
            "producer_geometry_sha256": geometry_sha,
            "independent_geometry_sha256": geometry_sha,
            "raw_point_count": 1000,
            "nonfinite_excluded_count": 0,
            "range_excluded_count": 0,
            "post_filter_point_count": 1000,
            "source_voxel_reduced_count": 0,
            "canonical_source_point_count": 1000,
            "verification_status": "PASS_DUAL_PATH_GEOMETRY_EXACT_SHARED_FROZEN_PRIMITIVES",
        }
        geometry_witness = {
            **geometry_core,
            "geometry_metric_verification_row_sha256": _hash_json(geometry_core),
        }
        geometry_witness_rows.append(geometry_witness)
        first_pass_commits.append(
            chained_event(
                query_journal_rows,
                "zprm.boreas.v2.stage2.query_journal.v1",
                {
                    "event_kind": "FIRST_PASS_COMMITTED",
                    "execution_stage": "QUERY_GEOMETRY_FIRST_PASS",
                    "object_key": allow["key"],
                    "payload": {
                        "download_event_sha256": download["event_sha256"],
                        "first_pass_row": first_row,
                        "geometry_metric_verification_row": geometry_witness,
                        "processing_result_sha256": _hash_json(first_row),
                        "receipt_sha256": receipt["receipt_sha256"],
                    },
                },
            )
        )
    _write_csv(
        frozen / "geometry_metric_verification.csv",
        GEOMETRY_METRIC_VERIFICATION_FIELDS,
        geometry_witness_rows,
    )
    first_complete = chained_event(
        query_journal_rows,
        "zprm.boreas.v2.stage2.query_journal.v1",
        {
            "event_kind": "QUERY_FIRST_PASS_COMPLETE",
            "execution_stage": "QUERY_BARRIER",
            "object_key": "",
            "payload": {
                "first_pass_count": QUERY_COUNT,
                "first_pass_rows_sha256": _hash_json(first_pass),
                "target_freeze_event_sha256": target_barrier["event_sha256"],
            },
        },
    )
    selection_artifact_names = {
        "all_candidate_scans.csv",
        "all_candidate_intervals.csv",
        "excluded_candidate_scans.csv",
        "excluded_intervals.csv",
        "geometry_only_metrics.csv",
        "geometry_metric_verification.csv",
        "selected_scene_intervals.csv",
        "selected_snapshots.csv",
    }
    selection_artifact_sha = {
        name: sha256_file(frozen / name) for name in sorted(selection_artifact_names)
    }
    selection_freeze_core = {
        "schema": "zprm.boreas.v2.stage2.selection_freeze.v1",
        "target_freeze_file_sha256": sha256_file(
            frozen / "target_map_freeze_manifest.json"
        ),
        "first_pass_complete_event_sha256": first_complete["event_sha256"],
        "selection_contract_sha256": contract.sha256,
        "candidate_scan_count": len(scans),
        "candidate_scan_rows_sha256": _hash_json(
            [row["candidate_scan_row_sha256"] for row in scans]
        ),
        "candidate_interval_count": len(intervals),
        "candidate_interval_rows_sha256": _hash_json(
            [row["candidate_interval_row_sha256"] for row in intervals]
        ),
        "selected_interval_count": len(selected_intervals),
        "selected_interval_rows_sha256": _hash_json(
            [row["selected_interval_row_sha256"] for row in selected_intervals]
        ),
        "selected_snapshot_count": len(selected),
        "selected_snapshot_rows_sha256": _hash_json(
            [row["selected_snapshot_row_sha256"] for row in selected]
        ),
        "artifact_sha256": selection_artifact_sha,
        "blind_selection": blind,
        "R14_frozen": True,
        "registration_execution_count": 0,
    }
    _write_json(
        frozen / "boreas_v2_stage2_selection_freeze.json",
        {
            **selection_freeze_core,
            "selection_freeze_payload_sha256": _hash_json(selection_freeze_core),
        },
    )
    selection_barrier = chained_event(
        query_journal_rows,
        "zprm.boreas.v2.stage2.query_journal.v1",
        {
            "event_kind": "R14_SELECTION_FROZEN",
            "execution_stage": "QUERY_BARRIER",
            "object_key": "",
            "payload": {
                "artifact_sha256": selection_artifact_sha,
                "first_pass_complete_event_sha256": first_complete["event_sha256"],
                "selected_snapshot_count": len(selected),
                "selection_contract_sha256": contract.sha256,
                "selection_freeze_file_sha256": sha256_file(
                    frozen / "boreas_v2_stage2_selection_freeze.json"
                ),
            },
        },
    )
    projection_rows: list[dict[str, Any]] = []
    for ordinal, (commit, scan) in enumerate(zip(first_pass_commits, scans)):
        projection_core = {
            "execution_stage": "QUERY_GEOMETRY_FIRST_PASS",
            "query_ordinal": ordinal,
            "object_key": scan["object_key"],
            "first_pass_event_sha256": commit["event_sha256"],
            "first_pass_result_sha256": commit["payload"]["processing_result_sha256"],
            "candidate_scan_row_sha256": scan["candidate_scan_row_sha256"],
            "selection_freeze_event_sha256": selection_barrier["event_sha256"],
        }
        projection_rows.append(
            {**projection_core, "projection_row_sha256": _hash_json(projection_core)}
        )
    _write_csv(
        frozen / "first_pass_selection_projection.csv",
        FIRST_PASS_SELECTION_PROJECTION_FIELDS,
        projection_rows,
    )
    for snapshot in selected:
        snapshot_root = runtime / "snapshots" / snapshot["snapshot_id"]
        snapshot_root.mkdir(parents=True)
        source_relative = f"snapshots/{snapshot['snapshot_id']}/source_points.npy"
        transform_relative = f"snapshots/{snapshot['snapshot_id']}/T_reference.npy"
        metadata_relative = f"snapshots/{snapshot['snapshot_id']}/metadata.json"
        atomic_write_bytes(runtime / source_relative, source_payload)
        transform = np.eye(4, dtype=np.float64)
        window_index = int(str(snapshot["interval_id"]).rsplit("-", 1)[1])
        transform[0, 3] = window_index * 2.0
        transform_payload = _canonical_npy(transform)
        transform_sha = hashlib.sha256(transform_payload).hexdigest()
        atomic_write_bytes(runtime / transform_relative, transform_payload)
        expected_t[str(snapshot["object_key"])] = transform_sha
        metadata = {
            "snapshot_id": snapshot["snapshot_id"],
            "object_key": snapshot["object_key"],
            "source_points_sha256": source_sha,
            "T_reference_sha256": transform_sha,
            "target_map_sha256": target_sha,
        }
        _write_json(runtime / metadata_relative, metadata)
        row: dict[str, Any] = {
            "selection_index": snapshot["selection_index"],
            "snapshot_id": snapshot["snapshot_id"],
            "scene_label": snapshot["scene_label"],
            "interval_id": snapshot["interval_id"],
            "object_key": snapshot["object_key"],
            "source_points_path": source_relative,
            "source_points_sha256": source_sha,
            "source_points_size_bytes": len(source_payload),
            "source_point_count": 1000,
            "T_reference_path": transform_relative,
            "T_reference_sha256": transform_sha,
            "T_reference_size_bytes": len(transform_payload),
            "snapshot_metadata_path": metadata_relative,
            "snapshot_metadata_sha256": sha256_file(runtime / metadata_relative),
            "target_map_path": target_relative,
            "target_map_sha256": target_sha,
            "target_map_size_bytes": len(target_payload),
            "target_point_count": TARGET_POINT_COUNT,
            "preprocessing_contract_sha256": preprocessing_sha,
            "selection_contract_sha256": contract.sha256,
            "backend_parameter_contract_sha256": BACKEND_SHA,
            "future_open3d_source_sha256": source_sha,
            "future_pcl_source_sha256": source_sha,
            "future_open3d_target_sha256": target_sha,
            "future_pcl_target_sha256": target_sha,
            "byte_identical_for_both_backends": True,
        }
        # Canonical bundle hashes are defined over the exact scalar CSV values.
        scalar_core = {key: str(value) for key, value in row.items()}
        row["bundle_sha256"] = _hash_json(scalar_core)
        canonical_rows.append(row)
        first_receipt = receipts_by_stage_key[
            ("QUERY_GEOMETRY_FIRST_PASS", snapshot["object_key"])
        ]
        second_receipt = receipts_by_stage_key[
            ("QUERY_CANONICAL_SECOND_PASS", snapshot["object_key"])
        ]
        witness_core = {
            "selection_index": snapshot["selection_index"],
            "snapshot_id": snapshot["snapshot_id"],
            "object_key": snapshot["object_key"],
            "raw_payload_sha256": second_receipt["local_temporary_sha256"],
            "raw_size_bytes": second_receipt["remote_size_bytes"],
            "first_pass_receipt_sha256": first_receipt["receipt_sha256"],
            "second_pass_receipt_sha256": second_receipt["receipt_sha256"],
            "preprocessing_contract_sha256": preprocessing_sha,
            "gt_sha256": reference_pose_sha,
            "extrinsic_sha256": extrinsic_sha,
            "witness_implementation_sha256": witness_sha,
            "producer_source_sha256": source_sha,
            "independent_source_sha256": source_sha,
            "producer_T_reference_sha256": transform_sha,
            "independent_T_reference_sha256": transform_sha,
            "raw_point_count": 1000,
            "nonfinite_excluded_count": 0,
            "range_excluded_count": 0,
            "post_filter_point_count": 1000,
            "source_voxel_reduced_count": 0,
            "canonical_source_point_count": 1000,
            "verification_status": (
                "PASS_DUAL_PATH_BYTE_IDENTITY_SHARED_FROZEN_PRIMITIVES"
            ),
        }
        witness_rows.append(
            {
                **witness_core,
                "canonical_source_verification_row_sha256": _hash_json(
                    witness_core
                ),
            }
        )
    _write_csv(
        frozen / "canonical_input_manifest.csv",
        CANONICAL_INPUT_FIELDS,
        canonical_rows,
    )
    _write_csv(
        frozen / "canonical_source_verification.csv",
        CANONICAL_SOURCE_VERIFICATION_FIELDS,
        witness_rows,
    )

    for canonical_row, witness_row, snapshot in zip(
        canonical_rows, witness_rows, selected
    ):
        receipt = receipts_by_stage_key[
            ("QUERY_CANONICAL_SECOND_PASS", snapshot["object_key"])
        ]
        relative = (
            f"temporary/query-second-{int(snapshot['selection_index']):05d}.bin.partial"
        )
        intent = chained_event(
            query_journal_rows,
            "zprm.boreas.v2.stage2.query_journal.v1",
            {
                "event_kind": "TRANSFER_INTENT",
                "execution_stage": "QUERY_CANONICAL_SECOND_PASS",
                "object_key": snapshot["object_key"],
                "payload": {
                    "etag": receipt["etag"],
                    "last_modified": receipt["last_modified"],
                    "remote_size_bytes": int(receipt["remote_size_bytes"]),
                    "sequence_id": receipt["sequence_id"],
                    "temporary_relative_path": relative,
                    "timestamp_us": int(receipt["timestamp_us"]),
                },
            },
        )
        download = chained_event(
            query_journal_rows,
            "zprm.boreas.v2.stage2.query_journal.v1",
            {
                "event_kind": "DOWNLOADED",
                "execution_stage": "QUERY_CANONICAL_SECOND_PASS",
                "object_key": snapshot["object_key"],
                "payload": {
                    "receipt": native_receipt(receipt),
                    "temporary_relative_path": relative,
                    "transfer_intent_event_sha256": intent["event_sha256"],
                },
            },
        )
        chained_event(
            query_journal_rows,
            "zprm.boreas.v2.stage2.query_journal.v1",
            {
                "event_kind": "SELECTED_SOURCE_COMMITTED",
                "execution_stage": "QUERY_CANONICAL_SECOND_PASS",
                "object_key": snapshot["object_key"],
                "payload": {
                    "canonical_input_row": canonical_row,
                    "canonical_source_verification_row": witness_row,
                    "download_event_sha256": download["event_sha256"],
                    "processing_result_sha256": canonical_row["source_points_sha256"],
                    "receipt_sha256": receipt["receipt_sha256"],
                    "selected_snapshot_row_sha256": snapshot[
                        "selected_snapshot_row_sha256"
                    ],
                },
            },
        )
    query_journal_relative = "checkpoints/query_journal.jsonl"
    query_journal_path = runtime / query_journal_relative
    atomic_write_bytes(
        query_journal_path,
        b"".join(compact_line(row) for row in query_journal_rows),
    )
    query_receipt_rows = receipt_rows[MAP_COUNT:]
    query_bytes = sum(int(row["remote_size_bytes"]) for row in query_receipt_rows)
    query_audit = {
        "schema": "zprm.boreas.v2.stage2.query_download_audit.v1",
        "successful_download_event_count": len(query_receipt_rows),
        "successful_payload_bytes": query_bytes,
        "committed_payload_event_count": len(query_receipt_rows),
        "committed_payload_bytes": query_bytes,
        "retry_download_event_count": 0,
        "retry_download_payload_bytes": 0,
        "raw_payload_persistent_bytes": 0,
        "first_pass_committed_count": QUERY_COUNT,
        "selected_source_committed_count": len(selected),
        "query_journal_path": query_journal_relative,
        "query_journal_sha256": sha256_file(query_journal_path),
        "query_journal_final_event_sha256": query_journal_rows[-1]["event_sha256"],
        "target_frozen_event_sha256": target_barrier["event_sha256"],
        "first_pass_complete_event_sha256": first_complete["event_sha256"],
        "R14_selection_frozen_event_sha256": selection_barrier["event_sha256"],
        "registration_execution_count": 0,
    }
    _write_json(frozen / "QUERY_LIDAR_DOWNLOAD_AUDIT.json", query_audit)
    map_audit = _read_json(frozen / "MAP_LIDAR_DOWNLOAD_AUDIT.json")
    _write_json(
        frozen / "LIDAR_DOWNLOAD_AUDIT.json",
        {
            "schema": "zprm.boreas.v2.stage2.lidar_download_audit.v3",
            "successful_download_event_count": MAP_COUNT + len(query_receipt_rows),
            "successful_payload_bytes": map_download_bytes + query_bytes,
            "committed_payload_event_count": len(receipt_rows),
            "committed_payload_bytes": sum(
                int(row["remote_size_bytes"]) for row in receipt_rows
            ),
            "unique_allowlist_object_count": MAP_COUNT + QUERY_COUNT,
            "raw_payload_persistent_bytes": 0,
            "map_receipt_journal_path": map_journal_relative,
            "map_receipt_journal_sha256": sha256_file(map_journal_path),
            "map_receipt_final_chain_sha256": map_journal_rows[-1]["event_sha256"],
            "map_retry_download_event_count": 0,
            "map_retry_download_payload_bytes": 0,
            "query_journal_path": query_journal_relative,
            "query_journal_sha256": sha256_file(query_journal_path),
            "query_journal_final_event_sha256": query_journal_rows[-1]["event_sha256"],
            "query_retry_download_event_count": 0,
            "query_retry_download_payload_bytes": 0,
            "target_frozen_event_sha256": target_barrier["event_sha256"],
            "query_first_pass_complete_event_sha256": first_complete["event_sha256"],
            "R14_selection_frozen_event_sha256": selection_barrier["event_sha256"],
            "map_receipt_count": MAP_COUNT,
            "query_first_pass_receipt_count": QUERY_COUNT,
            "query_second_pass_receipt_count": len(selected),
            "registration_execution_count": 0,
        },
    )

    uncertainty_rows = [
        {
            "component": component,
            "value": "UNKNOWN",
            "unit": "UNKNOWN",
            "uncertainty_type": "UNKNOWN",
            "status": "UNKNOWN",
            "evidence": "not numerically supported by frozen public evidence",
        }
        for component in sorted(REQUIRED_UNKNOWN_COMPONENTS)
    ]
    _write_csv(
        frozen / "boreas_v2_stage2_uncertainty_budget.csv",
        UNCERTAINTY_FIELDS,
        uncertainty_rows,
    )
    # Match the strings read back from the authoritative CSV projection.
    _, uncertainty_strings = _read_csv(
        frozen / "boreas_v2_stage2_uncertainty_budget.csv"
    )
    _write_json(
        frozen / "boreas_v2_stage2_uncertainty_budget.json",
        {"rows": uncertainty_strings},
    )

    _write_json(
        frozen / "NO_ICP_ATTESTATION.json",
        {
            "estimated_transform_count": 0,
            "estimated_transform_evidence": [],
            "estimated_transform_file_count": 0,
            "estimated_transform_files": [],
            "open3d_registration_call_count": 0,
            "pcl_cli_invocation_count": 0,
            "other_registration_process_count": 0,
            "registration_execution_count": 0,
            "real_trial_result_count": 0,
            "actual_open3d_trials": 0,
            "actual_pcl_trials": 0,
            "actual_trials": 0,
            "pass": True,
            "status": "PASS",
            "structured_result_scan_error_count": 0,
            "structured_result_scan_error_files": [],
        },
    )
    authorization_path = frozen / "boreas_v2_stage2_download_authorization.json"
    synthetic_device = runtime.stat().st_dev
    authorization_core = {
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
        "REAL_REGISTRATION_AUTHORIZED": False,
        "STAGE2_DOWNLOAD_AUTHORIZED": True,
        "actual_trials": 0,
        "allowlist_object_count": MAP_COUNT + QUERY_COUNT,
        "allowlist_remote_bytes": sum(
            int(row["size_bytes"]) for row in map_allowlist + query_allowlist
        ),
        "allowlist_sha256": sha256_file(allowlist_path),
        "authorization_path": str(authorization_path),
        "authorization_scope": "BOREAS_V2_STAGE2_LIDAR_DATA_PREPARATION_ONLY",
        "branch": "run/boreas-v2-stage2-data-preparation",
        "bucket": "boreas",
        "commit": "a" * 40,
        "disk_free_bytes_at_authorization": 2_000_000_000,
        "execution_mode": "STREAMING_LOW_DISK",
        "extrinsic_limitation": "PASS_WITH_DOCUMENTED_LIMITATION",
        "minimum_start_free_disk_bytes": 1_000_000_000,
        "no_icp_guard_active": True,
        "no_registration_environment_active": True,
        "preprocessing_contract_payload_sha256": "1" * 64,
        "preprocessing_contract_sha256": preprocessing_sha,
        "primary_pair": {
            "map_sequence_id": MAP_SEQUENCE,
            "query_sequence_id": QUERY_SEQUENCE,
        },
        "primary_pair_sha256": sha256_file(pair_path),
        "registration_execution_count": 0,
        "root_bindings": {
            "repository": {
                "access": "READ_EXECUTE",
                "path": str(authority_root),
                "st_dev": synthetic_device,
            },
            "stage1_data": {
                "access": "READ_EXECUTE",
                "path": str(authority_root),
                "st_dev": synthetic_device,
            },
            "stage2_runtime": {
                "access": "READ_WRITE_EXECUTE",
                "path": str(runtime),
                "st_dev": synthetic_device,
            },
            "stage2_temporary": {
                "access": "READ_WRITE_EXECUTE",
                "path": str(runtime / "temporary"),
                "st_dev": synthetic_device,
            },
            "monitored_disk": {
                "access": "READ_WRITE_EXECUTE",
                "path": str(runtime),
                "st_dev": synthetic_device,
            },
        },
        "runtime_low_disk_watermark_bytes": 100_000_000,
        "schema_version": "boreas_v2_stage2_download_authorization_v2",
        "self_hash_semantics": "UNKEYED_SHA256_INTEGRITY_ONLY_LIVE_REAUTHENTICATION_REQUIRED",
        "stage1_manifest_file_sha256": sha256_file(stage1_manifest),
        "stage1_verification_report_path": str(runtime / "stage1_report.json"),
        "stage1_verification_report_sha256": "2" * 64,
        "static_source_audit": {"pass": True},
        "storage_budget_sha256": "3" * 64,
        "storage_contract_sha256": "4" * 64,
        "storage_manifest_file_sha256": sha256_file(storage_manifest),
        "storage_verification": {"verification_pass": True},
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "worktree_clean_before_authorization": True,
    }
    _write_json(
        authorization_path,
        {
            **authorization_core,
            "authorization_payload_sha256": _hash_json(authorization_core),
        },
    )
    _write_json(
        frozen / "boreas_v2_stage2_readiness.json",
        {
            "BOREAS_EXTERNAL_V2_STAGE2_READY": True,
            "READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION": True,
            "weak_snapshot_count": 50,
            "rich_snapshot_count": 50,
            "snapshot_count": 100,
            "planned_future_open3d_trials": 100,
            "planned_future_pcl_trials": 100,
            "planned_future_trials": 200,
            "actual_open3d_trials": 0,
            "actual_pcl_trials": 0,
            "actual_trials": 0,
            "registration_execution_count": 0,
            "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
            "REAL_REGISTRATION_AUTHORIZED": False,
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
            "schema": "zprm.boreas.v2.stage2.readiness.v1",
        },
    )
    _write_json(
        frozen / "boreas_v2_stage2_summary.json",
        {"snapshot_count": 100, "weak_snapshot_count": 50, "rich_snapshot_count": 50},
    )
    atomic_write_bytes(
        frozen / "boreas_v2_stage2_summary.md",
        b"# Synthetic Boreas v2 Stage-2 preparation\n",
    )
    atomic_write_bytes(
        frozen / "boreas_v2_stage2_preprocessing_contract.md",
        preprocessing_markdown.read_bytes(),
    )
    atomic_write_bytes(
        frozen / "boreas_v2_stage2_preprocessing_contract.json",
        preprocessing.read_bytes(),
    )

    artifact_names = {
        "all_candidate_scans.csv",
        "all_candidate_intervals.csv",
        "excluded_candidate_scans.csv",
        "excluded_intervals.csv",
        "geometry_only_metrics.csv",
        "geometry_metric_verification.csv",
        "first_pass_selection_projection.csv",
        "selected_scene_intervals.csv",
        "selected_snapshots.csv",
        "canonical_input_manifest.csv",
        "canonical_source_verification.csv",
        "map_lineage_manifest.json",
        "target_map_reducer_verification.json",
        "target_map_freeze_manifest.json",
        "boreas_v2_stage2_preprocessing_contract.json",
        "boreas_v2_stage2_uncertainty_budget.csv",
        "boreas_v2_stage2_uncertainty_budget.json",
        "NO_ICP_ATTESTATION.json",
    }
    selection_core = {
        "schema_version": "boreas_v2_stage2_selection_manifest_v1",
        "authority_bindings": {
            "stage1_manifest_sha256": sha256_file(stage1_manifest),
            "storage_manifest_sha256": sha256_file(storage_manifest),
            "stage1_allowlist_sha256": sha256_file(allowlist_path),
            "pair_selection_sha256": sha256_file(pair_path),
            "preprocessing_contract_sha256": preprocessing_sha,
            "primary_pair": {
                "map_sequence_id": MAP_SEQUENCE,
                "query_sequence_id": QUERY_SEQUENCE,
            },
        },
        "artifact_sha256": {
            name: sha256_file(frozen / name) for name in sorted(artifact_names)
        },
        "blind_selection": blind,
        "R14_frozen": True,
        "registration_execution_count": 0,
    }
    _write_json(
        frozen / "boreas_v2_stage2_selection_manifest.json",
        {
            **selection_core,
            "selection_manifest_sha256": _hash_json(selection_core),
        },
    )

    manifest_core = {
        "schema_version": "boreas_v2_stage2_preparation_frozen_manifest_v1",
        "stage1_manifest_sha256": sha256_file(stage1_manifest),
        "storage_manifest_sha256": sha256_file(storage_manifest),
        "stage1_allowlist_sha256": sha256_file(allowlist_path),
        "pair_selection_sha256": sha256_file(pair_path),
        "preprocessing_contract_sha256": preprocessing_sha,
        "payload": [
            {
                "path": name,
                "sha256": sha256_file(frozen / name),
                "size_bytes": (frozen / name).stat().st_size,
            }
            for name in sorted(PAYLOAD_FILES)
        ],
    }
    _write_json(
        frozen / "boreas_v2_stage2_frozen_manifest.json",
        {**manifest_core, "manifest_root_sha256": compact_sha256(manifest_core)},
    )
    names = sorted(path.name for path in frozen.iterdir())
    atomic_write_bytes(
        frozen / "SHA256SUMS",
        "".join(f"{sha256_file(frozen / name)}  {name}\n" for name in names).encode(),
    )

    profile = authority_root / "profile.json"
    _write_json(
        profile,
        {
            "stage1_manifest_path": str(stage1_manifest),
            "storage_manifest_path": str(storage_manifest),
            "stage1_allowlist_path": str(allowlist_path),
            "pair_selection_path": str(pair_path),
            "preprocessing_contract_path": str(preprocessing),
            "reference_pose_path": str(reference_pose),
            "expected_reference_pose_sha256": reference_pose_sha,
            "map_reference_pose_path": str(map_reference_pose),
            "expected_map_reference_pose_sha256": map_reference_pose_sha,
            "extrinsic_path": str(extrinsic_path),
            "expected_extrinsic_sha256": extrinsic_sha,
            "transform_chain_consistency_path": str(chain_consistency_path),
            "transform_chain_manifest_path": str(chain_manifest_path),
            "canonical_witness_implementation_path": str(witness_path),
            "target_reducer_implementation_path": str(reducer_source_path),
            "expected_map_sequence_id": MAP_SEQUENCE,
            "expected_query_sequence_id": QUERY_SEQUENCE,
            "expected_candidate_scan_count": QUERY_COUNT,
            "expected_candidate_interval_count": WINDOW_COUNT,
            "expected_map_object_count": MAP_COUNT,
            "expected_backend_parameter_sha256": BACKEND_SHA,
            "selection_parameter_authority": SYNTHETIC_AUTHORITY,
            "expected_t_reference_sha256_by_object_key": expected_t,
        },
    )
    return frozen, runtime, profile


@pytest.fixture(scope="session")
def synthetic_closure(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    return _build_synthetic_closure(tmp_path_factory.mktemp("boreas-stage2-verifier"))


def _verify(paths: tuple[Path, Path, Path]) -> dict[str, Any]:
    frozen, runtime, profile = paths
    return verify_boreas_v2_stage2_preparation(
        root=frozen,
        runtime_root=runtime,
        authority=PreparationVerificationAuthority.from_profile(profile),
    )


def test_independent_verifier_accepts_complete_synthetic_closure(
    synthetic_closure: tuple[Path, Path, Path],
) -> None:
    result = _verify(synthetic_closure)
    assert result["BOREAS_EXTERNAL_V2_STAGE2_VERIFICATION_PASS"] is True
    assert result["snapshot_count"] == 100
    assert result["registration_execution_count"] == 0


def test_independent_verifier_cli_accepts_complete_synthetic_closure(
    synthetic_closure: tuple[Path, Path, Path],
) -> None:
    frozen, runtime, profile = synthetic_closure
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/verify_boreas_v2_stage2_preparation.py"),
            "--frozen-root",
            str(frozen),
            "--runtime-root",
            str(runtime),
            "--authority-profile",
            str(profile),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert json.loads(completed.stdout)[
        "BOREAS_EXTERNAL_V2_STAGE2_VERIFICATION_PASS"
    ] is True


def test_independent_verifier_rejects_coherently_resigned_arbitrary_target(
    tmp_path: Path,
) -> None:
    # This constructs every downstream artifact from the altered target SHA:
    # freeze/reducer evidence, candidates, selection, journals, canonical
    # metadata, summaries, frozen manifest, and SHA256SUMS are all mutually
    # coherent.  Only an independent replay of the authenticated map ranges can
    # distinguish it from the genuine deterministic target.
    paths = _build_synthetic_closure(
        tmp_path / "coherent-target-tamper", coherent_target_delta_m=0.125
    )
    with pytest.raises(
        BoreasStage2PreparationVerificationError,
        match="independent target replay byte comparison failed",
    ):
        _verify(paths)


@pytest.mark.parametrize(
    ("artifact", "message"),
    (
        ("plan", "reducer resource plan reducer_binary_projected_bytes mismatch"),
        ("layout", "compiled reducer capacity independent formula mismatch"),
        ("binding", "reducer replay binding replay_plan_sha256 mismatch"),
    ),
)
def test_independent_verifier_rejects_coherently_resigned_resource_evidence(
    tmp_path: Path, artifact: str, message: str
) -> None:
    # Each invalid semantic is introduced before the producer-side evidence,
    # target freeze, query journal, selection, summaries, outer manifest, and
    # SHA256SUMS are constructed.  Thus every downstream hash is coherent and
    # only the independent resource semantics distinguish the tamper.
    paths = _build_synthetic_closure(
        tmp_path / f"coherent-resource-{artifact}",
        coherent_resource_tamper=artifact,
    )
    with pytest.raises(BoreasStage2PreparationVerificationError, match=message):
        _verify(paths)


Tamper = Callable[[Path, Path], None]


def _json_tamper(name: str, mutate: Callable[[dict[str, Any]], None]) -> Tamper:
    def apply(frozen: Path, runtime: Path) -> None:
        path = frozen / name
        value = _read_json(path)
        mutate(value)
        _write_json(path, value)

    return apply


def _csv_tamper(
    name: str, mutate: Callable[[list[dict[str, str]]], None]
) -> Tamper:
    def apply(frozen: Path, runtime: Path) -> None:
        path = frozen / name
        fields, rows = _read_csv(path)
        mutate(rows)
        _write_csv(path, fields, rows)

    return apply


def _change_primary_pair(value: dict[str, Any]) -> None:
    value["authority_bindings"]["primary_pair"]["map_sequence_id"] = "reserve-map"


def _change_voxel(value: dict[str, Any]) -> None:
    value["target_map_voxel_size_m"] = 0.30


def _change_deskew(value: dict[str, Any]) -> None:
    value["deskew"]["enabled"] = False


def _change_target_sha(value: dict[str, Any]) -> None:
    value["target_map_sha256"] = "f" * 64


def _contaminate_map(value: dict[str, Any]) -> None:
    value["source_objects"].append(
        {
            "object_key": f"{QUERY_SEQUENCE}/lidar/500000.bin",
            "payload_sha256": "a" * 64,
        }
    )
    value["source_object_count"] += 1


def _weak_label(rows: list[dict[str, str]]) -> None:
    rows[0]["scene_label"] = RICH_LABEL


def _rich_label(rows: list[dict[str, str]]) -> None:
    rows[10]["scene_label"] = WEAK_LABEL


def _interval_start(rows: list[dict[str, str]]) -> None:
    rows[0]["start_time_us"] = str(int(rows[0]["start_time_us"]) + 1)


def _snapshot_object(rows: list[dict[str, str]]) -> None:
    rows[0]["object_key"] = rows[1]["object_key"]


def _canonical_source_sha(rows: list[dict[str, str]]) -> None:
    rows[0]["source_points_sha256"] = "f" * 64


def _canonical_transform_sha(rows: list[dict[str, str]]) -> None:
    rows[0]["T_reference_sha256"] = "f" * 64


def _backend_target_sha(rows: list[dict[str, str]]) -> None:
    rows[0]["future_pcl_target_sha256"] = "f" * 64


def _unknown_to_zero_csv(rows: list[dict[str, str]]) -> None:
    row = next(child for child in rows if child["component"] == "deskew_uncertainty")
    row["value"] = "0"
    row["unit"] = "m"
    row["uncertainty_type"] = "BOUND"
    row["status"] = "KNOWN"


def _uncertainty_both(frozen: Path, runtime: Path) -> None:
    csv_path = frozen / "boreas_v2_stage2_uncertainty_budget.csv"
    fields, rows = _read_csv(csv_path)
    _unknown_to_zero_csv(rows)
    _write_csv(csv_path, fields, rows)
    value = _read_json(frozen / "boreas_v2_stage2_uncertainty_budget.json")
    _unknown_to_zero_csv(value["rows"])
    _write_json(frozen / "boreas_v2_stage2_uncertainty_budget.json", value)


def _drop_snapshot(rows: list[dict[str, str]]) -> None:
    rows.pop()


def _weak_49(rows: list[dict[str, str]]) -> None:
    row = next(child for child in rows if child["scene_label"] == WEAK_LABEL)
    row["scene_label"] = RICH_LABEL


def _add_registration_result(frozen: Path, runtime: Path) -> None:
    _write_json(frozen / "registration_result.json", {"fitness": 1.0})


def _change_allowlist_binding(value: dict[str, Any]) -> None:
    value["authority_bindings"]["stage1_allowlist_sha256"] = "f" * 64


def _midpoint_position(rows: list[dict[str, str]]) -> None:
    rows[0]["center_world_x_m"] = str(float(rows[0]["center_world_x_m"]) + 0.25)


def _first_pass_etag(rows: list[dict[str, str]]) -> None:
    row = next(child for child in rows if child["execution_stage"] == "QUERY_GEOMETRY_FIRST_PASS")
    row["etag"] = "tampered-etag"


def _geometry_witness_count(rows: list[dict[str, str]]) -> None:
    rows[0]["raw_point_count"] = str(int(rows[0]["raw_point_count"]) + 1)


def _projection_event(rows: list[dict[str, str]]) -> None:
    rows[0]["first_pass_event_sha256"] = "f" * 64


def _selection_freeze_count(value: dict[str, Any]) -> None:
    value["selected_snapshot_count"] = 99


def _query_audit_retry(value: dict[str, Any]) -> None:
    value["retry_download_event_count"] = 1


def _map_audit_retry(value: dict[str, Any]) -> None:
    value["retry_download_event_count"] = 1


def _aggregate_success(value: dict[str, Any]) -> None:
    value["successful_download_event_count"] += 1


def _map_receipt_result(rows: list[dict[str, str]]) -> None:
    rows[0]["processing_result_sha256"] = "f" * 64


def _authorization_scope(value: dict[str, Any]) -> None:
    value["authorization_scope"] = "EXPANDED_UNAUTHORIZED_SCOPE"


def _runtime_jsonl_tamper(relative: str) -> Tamper:
    def apply(frozen: Path, runtime: Path) -> None:
        path = runtime / relative
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["event_sha256"] = "f" * 64
        atomic_write_bytes(
            path,
            b"".join(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for row in rows
            ),
            overwrite=True,
        )

    return apply


TAMPERS: tuple[tuple[str, Tamper, bool], ...] = (
    (
        "primary_pair",
        _json_tamper("boreas_v2_stage2_selection_manifest.json", _change_primary_pair),
        False,
    ),
    (
        "target_voxel",
        _json_tamper("boreas_v2_stage2_preprocessing_contract.json", _change_voxel),
        False,
    ),
    (
        "deskew_contract",
        _json_tamper("boreas_v2_stage2_preprocessing_contract.json", _change_deskew),
        False,
    ),
    (
        "target_map_sha",
        _json_tamper("target_map_freeze_manifest.json", _change_target_sha),
        False,
    ),
    (
        "map_query_contamination",
        _json_tamper("map_lineage_manifest.json", _contaminate_map),
        False,
    ),
    ("weak_interval_label", _csv_tamper("selected_scene_intervals.csv", _weak_label), False),
    ("rich_interval_label", _csv_tamper("selected_scene_intervals.csv", _rich_label), False),
    ("interval_start", _csv_tamper("all_candidate_intervals.csv", _interval_start), False),
    ("selected_snapshot", _csv_tamper("selected_snapshots.csv", _snapshot_object), False),
    ("canonical_source_sha", _csv_tamper("canonical_input_manifest.csv", _canonical_source_sha), False),
    ("T_reference", _csv_tamper("canonical_input_manifest.csv", _canonical_transform_sha), False),
    ("backend_target_sha", _csv_tamper("canonical_input_manifest.csv", _backend_target_sha), False),
    ("unknown_to_zero", _uncertainty_both, False),
    ("snapshot_count_99", _csv_tamper("selected_snapshots.csv", _drop_snapshot), False),
    ("weak_count_49", _csv_tamper("selected_snapshots.csv", _weak_49), False),
    ("registration_result", _add_registration_result, True),
    (
        "stage1_allowlist_binding",
        _json_tamper("boreas_v2_stage2_selection_manifest.json", _change_allowlist_binding),
        False,
    ),
    (
        "geometry_witness_counter",
        _csv_tamper("geometry_metric_verification.csv", _geometry_witness_count),
        False,
    ),
    (
        "first_pass_projection_event",
        _csv_tamper("first_pass_selection_projection.csv", _projection_event),
        False,
    ),
    (
        "selection_freeze_count",
        _json_tamper("boreas_v2_stage2_selection_freeze.json", _selection_freeze_count),
        False,
    ),
    (
        "query_audit_retry",
        _json_tamper("QUERY_LIDAR_DOWNLOAD_AUDIT.json", _query_audit_retry),
        False,
    ),
    (
        "map_audit_retry",
        _json_tamper("MAP_LIDAR_DOWNLOAD_AUDIT.json", _map_audit_retry),
        False,
    ),
    (
        "aggregate_success",
        _json_tamper("LIDAR_DOWNLOAD_AUDIT.json", _aggregate_success),
        False,
    ),
    (
        "map_receipt_result",
        _csv_tamper("map_download_receipts.csv", _map_receipt_result),
        False,
    ),
    (
        "authorization_scope",
        _json_tamper("boreas_v2_stage2_download_authorization.json", _authorization_scope),
        False,
    ),
    (
        "query_journal_chain",
        _runtime_jsonl_tamper("checkpoints/query_journal.jsonl"),
        False,
    ),
    (
        "map_journal_chain",
        _runtime_jsonl_tamper("checkpoints/receipt_checkpoints.jsonl"),
        False,
    ),
)


@pytest.mark.parametrize(
    ("case_name", "tamper", "include_extra"),
    TAMPERS,
    ids=[row[0] for row in TAMPERS],
)
def test_independent_verifier_rejects_all_resigned_tampers(
    synthetic_closure: tuple[Path, Path, Path],
    tmp_path: Path,
    case_name: str,
    tamper: Tamper,
    include_extra: bool,
) -> None:
    source_frozen, source_runtime, source_profile = synthetic_closure
    frozen = tmp_path / "frozen"
    runtime = tmp_path / "runtime"
    authority_root = tmp_path / "authority"
    shutil.copytree(source_frozen, frozen)
    shutil.copytree(source_runtime, runtime)
    shutil.copytree(source_profile.parent, authority_root)
    tamper(frozen, runtime)
    _resign(frozen, include_extra_payloads=include_extra)
    authority = PreparationVerificationAuthority.from_profile(
        authority_root / source_profile.name
    )
    with pytest.raises(BoreasStage2PreparationVerificationError):
        verify_boreas_v2_stage2_preparation(
            root=frozen,
            runtime_root=runtime,
            authority=authority,
        )
