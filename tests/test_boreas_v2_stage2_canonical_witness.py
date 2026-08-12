from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from phase_a_harness.real_data_preparation.boreas_v2_stage2_canonical_witness import (
    BoreasStage2CanonicalWitnessError,
    CanonicalSourceWitnessBindings,
    independently_verify_canonical_source,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_preprocessing import (
    POSE_HEADER,
    PRIMARY_QUERY_SEQUENCE_ID,
    BoreasLidarPoseIndex,
    canonical_source_npy_bytes,
    canonical_t_reference_npy_bytes,
    decode_boreas_velodyne_payload,
    preprocess_primary_boreas_scan,
)


TIMESTAMP_US = 1_600_000_000_000_000
SHA_A = "a" * 64


def _fixture(tmp_path: Path) -> tuple[bytes, str, BoreasLidarPoseIndex, bytes, bytes]:
    object_key = f"{PRIMARY_QUERY_SEQUENCE_ID}/lidar/{TIMESTAMP_US}.bin"
    raw = np.asarray(
        [
            [2.0, 0.0, 0.0, 1.0, 1.0, -0.05],
            [2.2, 0.0, 0.0, 1.0, 1.0, 0.05],
        ],
        dtype="<f4",
    ).tobytes()
    pose_path = tmp_path / "lidar_poses.csv"
    with pose_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(POSE_HEADER)
        writer.writerow(
            [TIMESTAMP_US, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        writer.writerow(
            [TIMESTAMP_US + 100_000, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
    pose_sha = hashlib.sha256(pose_path.read_bytes()).hexdigest()
    index = BoreasLidarPoseIndex.from_csv(
        pose_path,
        sequence_id=PRIMARY_QUERY_SEQUENCE_ID,
        expected_sha256=pose_sha,
    )
    scan = preprocess_primary_boreas_scan(
        decode_boreas_velodyne_payload(raw, object_key=object_key),
        pose_index=index,
        role="QUERY",
    )
    return (
        raw,
        object_key,
        index,
        canonical_source_npy_bytes(scan),
        canonical_t_reference_npy_bytes(scan),
    )


def _bindings() -> CanonicalSourceWitnessBindings:
    return CanonicalSourceWitnessBindings(
        selection_index=0,
        snapshot_id="boreas-v2-weak-01-q10",
        first_pass_receipt_sha256=SHA_A,
        second_pass_receipt_sha256="b" * 64,
        preprocessing_contract_sha256="c" * 64,
        gt_sha256="d" * 64,
        extrinsic_sha256="e" * 64,
        witness_implementation_sha256="f" * 64,
    )


def test_independent_canonical_witness_proves_exact_bytes(tmp_path: Path) -> None:
    raw, key, index, source, transform = _fixture(tmp_path)
    row = independently_verify_canonical_source(
        raw,
        object_key=key,
        pose_index=index,
        producer_source_npy=source,
        producer_t_reference_npy=transform,
        bindings=_bindings(),
    )
    assert row["verification_status"] == (
        "PASS_DUAL_PATH_BYTE_IDENTITY_SHARED_FROZEN_PRIMITIVES"
    )
    assert row["raw_payload_sha256"] == hashlib.sha256(raw).hexdigest()
    assert row["producer_source_sha256"] == row["independent_source_sha256"]
    assert row["producer_T_reference_sha256"] == row[
        "independent_T_reference_sha256"
    ]


def test_independent_canonical_witness_rejects_producer_byte_change(
    tmp_path: Path,
) -> None:
    raw, key, index, source, transform = _fixture(tmp_path)
    with pytest.raises(BoreasStage2CanonicalWitnessError, match="bytes differ"):
        independently_verify_canonical_source(
            raw,
            object_key=key,
            pose_index=index,
            producer_source_npy=source[:-1] + bytes([source[-1] ^ 1]),
            producer_t_reference_npy=transform,
            bindings=_bindings(),
        )
