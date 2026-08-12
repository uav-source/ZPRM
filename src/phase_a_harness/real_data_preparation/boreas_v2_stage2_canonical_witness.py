"""Pre-deletion dual-path witness for Stage-2 canonical query bundles.

This module is deliberately a separate orchestration entry point.  It starts
from the authenticated raw bytes, invokes the same frozen public preprocessing
primitives through a separate orchestration path, serializes source/T bytes
itself, and compares them with producer bytes before the temporary raw object
may be deleted.  This proves byte identity across two orchestration paths; it
does not claim an algorithmically independent preprocessing implementation.  It never
performs registration, selection, network access, or file deletion.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .boreas_v2_stage2_preprocessing import (
    BoreasLidarPoseIndex,
    BoreasStage2PreprocessingError,
    decode_boreas_velodyne_payload,
    preprocess_primary_boreas_scan,
)
from .io import canonical_json_bytes


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_WITNESS_SCHEMA = "zprm.boreas.v2.stage2.canonical_source_witness.v1"


class BoreasStage2CanonicalWitnessError(RuntimeError):
    """The independent canonical witness could not prove byte identity."""


def _sha(value: Any, field: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise BoreasStage2CanonicalWitnessError(
            f"{field} must be a lowercase SHA-256"
        )
    return text


def _npy_v1(value: Any, *, exact_shape: tuple[int, ...] | None = None) -> bytes:
    array = np.asarray(value, dtype="<f8")
    if exact_shape is not None:
        if array.shape != exact_shape:
            raise BoreasStage2CanonicalWitnessError("canonical witness shape differs")
    elif array.ndim != 2 or array.shape[1:] != (3,) or array.shape[0] == 0:
        raise BoreasStage2CanonicalWitnessError("canonical witness points must be Nx3")
    if not np.all(np.isfinite(array)):
        raise BoreasStage2CanonicalWitnessError("canonical witness array is nonfinite")
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(array, dtype="<f8"),
        version=(1, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


@dataclass(frozen=True)
class CanonicalSourceWitnessBindings:
    selection_index: int
    snapshot_id: str
    first_pass_receipt_sha256: str
    second_pass_receipt_sha256: str
    preprocessing_contract_sha256: str
    gt_sha256: str
    extrinsic_sha256: str
    witness_implementation_sha256: str

    def validated(self) -> "CanonicalSourceWitnessBindings":
        if (
            isinstance(self.selection_index, bool)
            or not isinstance(self.selection_index, int)
            or self.selection_index < 0
            or not self.snapshot_id
        ):
            raise BoreasStage2CanonicalWitnessError("invalid witness snapshot identity")
        return CanonicalSourceWitnessBindings(
            selection_index=self.selection_index,
            snapshot_id=str(self.snapshot_id),
            first_pass_receipt_sha256=_sha(
                self.first_pass_receipt_sha256, "first-pass receipt SHA"
            ),
            second_pass_receipt_sha256=_sha(
                self.second_pass_receipt_sha256, "second-pass receipt SHA"
            ),
            preprocessing_contract_sha256=_sha(
                self.preprocessing_contract_sha256, "preprocessing contract SHA"
            ),
            gt_sha256=_sha(self.gt_sha256, "GT SHA"),
            extrinsic_sha256=_sha(self.extrinsic_sha256, "extrinsic SHA"),
            witness_implementation_sha256=_sha(
                self.witness_implementation_sha256, "witness implementation SHA"
            ),
        )


def independently_verify_canonical_source(
    raw_payload: bytes | bytearray | memoryview,
    *,
    object_key: str,
    pose_index: BoreasLidarPoseIndex,
    producer_source_npy: bytes | bytearray | memoryview,
    producer_t_reference_npy: bytes | bytearray | memoryview,
    bindings: CanonicalSourceWitnessBindings,
) -> dict[str, Any]:
    """Return a hash-bound PASS row only when both paths' bytes match exactly."""

    bound = bindings.validated()
    payload = bytes(raw_payload)
    source_bytes = bytes(producer_source_npy)
    transform_bytes = bytes(producer_t_reference_npy)
    if not payload:
        raise BoreasStage2CanonicalWitnessError("raw witness payload is empty")
    try:
        decoded = decode_boreas_velodyne_payload(payload, object_key=object_key)
        independently_preprocessed = preprocess_primary_boreas_scan(
            decoded,
            pose_index=pose_index,
            role="QUERY",
        )
    except BoreasStage2PreprocessingError as error:
        raise BoreasStage2CanonicalWitnessError(
            "independent preprocessing witness failed"
        ) from error
    independent_source = _npy_v1(independently_preprocessed.points_xyz)
    independent_transform = _npy_v1(
        independently_preprocessed.t_reference, exact_shape=(4, 4)
    )
    producer_source_sha = hashlib.sha256(source_bytes).hexdigest()
    independent_source_sha = hashlib.sha256(independent_source).hexdigest()
    producer_transform_sha = hashlib.sha256(transform_bytes).hexdigest()
    independent_transform_sha = hashlib.sha256(independent_transform).hexdigest()
    if source_bytes != independent_source or transform_bytes != independent_transform:
        raise BoreasStage2CanonicalWitnessError(
            "producer and independent canonical bundle bytes differ"
        )
    core = {
        "selection_index": bound.selection_index,
        "snapshot_id": bound.snapshot_id,
        "object_key": str(object_key),
        "raw_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_size_bytes": len(payload),
        "first_pass_receipt_sha256": bound.first_pass_receipt_sha256,
        "second_pass_receipt_sha256": bound.second_pass_receipt_sha256,
        "preprocessing_contract_sha256": bound.preprocessing_contract_sha256,
        "gt_sha256": bound.gt_sha256,
        "extrinsic_sha256": bound.extrinsic_sha256,
        "witness_implementation_sha256": bound.witness_implementation_sha256,
        "producer_source_sha256": producer_source_sha,
        "independent_source_sha256": independent_source_sha,
        "producer_T_reference_sha256": producer_transform_sha,
        "independent_T_reference_sha256": independent_transform_sha,
        "raw_point_count": independently_preprocessed.raw_point_count,
        "nonfinite_excluded_count": independently_preprocessed.nonfinite_excluded_count,
        "range_excluded_count": independently_preprocessed.range_excluded_count,
        "post_filter_point_count": independently_preprocessed.post_filter_point_count,
        "source_voxel_reduced_count": independently_preprocessed.source_voxel_reduced_count,
        "canonical_source_point_count": int(
            independently_preprocessed.points_xyz.shape[0]
        ),
        "verification_status": (
            "PASS_DUAL_PATH_BYTE_IDENTITY_SHARED_FROZEN_PRIMITIVES"
        ),
    }
    return {
        **core,
        "canonical_source_verification_row_sha256": hashlib.sha256(
            canonical_json_bytes(core)
        ).hexdigest(),
    }


__all__ = [
    "BOREAS_STAGE2_CANONICAL_WITNESS_SCHEMA",
    "BoreasStage2CanonicalWitnessError",
    "CANONICAL_WITNESS_SCHEMA",
    "CanonicalSourceWitnessBindings",
    "independently_verify_canonical_source",
]

# Compatibility-free explicit export name used in implementation manifests.
BOREAS_STAGE2_CANONICAL_WITNESS_SCHEMA = CANONICAL_WITNESS_SCHEMA
