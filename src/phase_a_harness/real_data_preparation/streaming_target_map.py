"""Deterministic, resumable streaming construction of a voxel target map.

This module is deliberately data-set agnostic.  In particular, it contains no
Boreas download code and it does not choose a voxel size.  A scientific
preprocessing rule must be supplied by the caller; storage planning is not an
authority for changing that rule.

The accumulator uses one fixed reduction order (scan ordinal, then point
ordinal), float64 sums, and a lexicographic final voxel order.  ``worker_count``
is accepted as an operational setting but never changes that reduction order.
This makes batch, incremental, resumed, and different-worker-count builds
byte-identical for the same inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import fcntl
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .io import atomic_write_json, canonical_json_bytes


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CENTROID_RULE = "CENTROID_FLOAT64_SCAN_THEN_POINT_ORDER_V1"
SUPPORTED_RULE_AUTHORITIES = frozenset(
    {"STAGE2_PREREGISTRATION", "SYNTHETIC_FIXTURE_ONLY"}
)
STORAGE_PLANNER_AUTHORITY = "STORAGE_PLANNER"
MAP_STATE_SCHEMA = "zprm.stage2.streaming_target_map_state.v1"
MAP_RESULT_SCHEMA = "zprm.stage2.streaming_target_map_result.v1"
PRODUCTION_REPLAY_LEDGER_SCHEMA = "zprm.stage2.production_map_replay_ledger.v2"
PRODUCTION_REPLAY_ALLOCATION_INTENT_SCHEMA = (
    "zprm.stage2.production_map_replay_allocation_intent.v1"
)
PRODUCTION_REPLAY_FORMAT = "LITTLE_ENDIAN_FLOAT64_XYZ_C_ORDER_NO_HEADER"
PRODUCTION_REPLAY_PADDING_RULE = "ZERO_FLOAT64_XYZ_UNUSED_SUFFIX_V1"
PYTHON_VOXEL_TRANSITION_KIND = "PYTHON_VOXEL_ACCUMULATOR_V1"
AUTHENTICATED_RANGE_TRANSITION_KIND = "AUTHENTICATED_REPLAY_RANGE_CHAIN_V1"
BOREAS_RAW_POINT_STRIDE_BYTES = 24
REPLAY_POINT_STRIDE_BYTES = 24
ZERO_SHA256 = "0" * 64


class StreamingTargetMapError(RuntimeError):
    """Raised when the deterministic map-building contract is violated."""


def _require_sha256(value: str, *, field: str) -> str:
    text = str(value).lower()
    if SHA256_RE.fullmatch(text) is None:
        raise StreamingTargetMapError(f"{field} must be a lowercase SHA-256")
    return text


def _canonical_float64(array: Any, *, shape_tail: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim != len(shape_tail) + 1 or value.shape[1:] != shape_tail:
        raise StreamingTargetMapError(
            f"array must have shape (N, {', '.join(map(str, shape_tail))})"
        )
    if not np.all(np.isfinite(value)):
        raise StreamingTargetMapError("point/transform arrays must be finite")
    return np.ascontiguousarray(value, dtype="<f8")


def canonical_array_sha256(array: Any) -> str:
    """Hash an array with an explicit dtype/shape header and C-order bytes."""

    value = np.asarray(array)
    if value.dtype.kind == "f":
        value = np.asarray(value, dtype="<f8")
    elif value.dtype.kind in "iu":
        value = np.asarray(value, dtype="<i8")
    else:
        raise StreamingTargetMapError(f"unsupported canonical array dtype: {value.dtype}")
    value = np.ascontiguousarray(value)
    header = canonical_json_bytes(
        {
            "dtype": value.dtype.str,
            "shape": [int(item) for item in value.shape],
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    # Hash through a byte view so multi-gigabyte read-only memmaps are not
    # copied into one equally large Python ``bytes`` object.
    byte_view = memoryview(value).cast("B")
    chunk_size = 8 * 1024 * 1024
    for start in range(0, len(byte_view), chunk_size):
        digest.update(byte_view[start : start + chunk_size])
    return digest.hexdigest()


def canonical_float64_npy_bytes(array: Any) -> bytes:
    """Canonical NPY v1.0 bytes compatible with ``ContentAddressedStore``."""

    value = np.asarray(array, dtype="<f8")
    if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
        raise StreamingTargetMapError("target points must be a finite float64 (N, 3) array")
    value = np.ascontiguousarray(value)
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


@dataclass(frozen=True)
class VoxelRule:
    """A caller-supplied scientific voxel rule.

    ``voxel_size_m`` intentionally has no default.  Real execution must use an
    independently frozen ``STAGE2_PREREGISTRATION`` contract.  The second
    authority exists solely for the artificial fixtures required by the
    storage-architecture audit.
    """

    voxel_size_m: float
    representative_rule: str
    parameter_authority: str
    scientific_contract_sha256: str
    origin_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.parameter_authority == STORAGE_PLANNER_AUTHORITY:
            raise StreamingTargetMapError(
                "storage planner is prohibited from choosing or changing voxel size"
            )
        if self.parameter_authority not in SUPPORTED_RULE_AUTHORITIES:
            raise StreamingTargetMapError(
                "voxel rule authority must be STAGE2_PREREGISTRATION or "
                "SYNTHETIC_FIXTURE_ONLY"
            )
        size = float(self.voxel_size_m)
        if not math.isfinite(size) or size <= 0.0:
            raise StreamingTargetMapError("voxel_size_m must be finite and positive")
        if self.representative_rule != CENTROID_RULE:
            raise StreamingTargetMapError(
                f"unsupported representative rule: {self.representative_rule!r}"
            )
        origin = tuple(float(value) for value in self.origin_xyz_m)
        if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
            raise StreamingTargetMapError("origin_xyz_m must contain three finite values")
        contract_sha256 = _require_sha256(
            self.scientific_contract_sha256, field="scientific_contract_sha256"
        )
        object.__setattr__(self, "voxel_size_m", size)
        object.__setattr__(self, "origin_xyz_m", origin)
        object.__setattr__(self, "scientific_contract_sha256", contract_sha256)

    def as_dict(self) -> dict[str, Any]:
        return {
            "duplicate_handling": "KEEP_ALL_FINITE_POINTS_IN_SUFFICIENT_STATISTICS",
            "numeric_dtype": "float64-little-endian",
            "origin_xyz_m": list(self.origin_xyz_m),
            "parameter_authority": self.parameter_authority,
            "representative_rule": self.representative_rule,
            "scientific_contract_sha256": self.scientific_contract_sha256,
            "voxel_size_m": self.voxel_size_m,
        }

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


@dataclass(frozen=True)
class MapScan:
    """One already-decoded scan and its frozen reference transform."""

    ordinal: int
    object_key: str
    points_xyz: np.ndarray
    reference_from_sensor: np.ndarray
    remote_size_bytes: int
    etag: str
    last_modified: str
    gt_sha256: str
    calibration_sha256: str
    object_sha256: str | None = None

    def validated(self) -> "MapScan":
        ordinal = int(self.ordinal)
        if ordinal < 0:
            raise StreamingTargetMapError("scan ordinal must be nonnegative")
        if not self.object_key or self.object_key.startswith("/"):
            raise StreamingTargetMapError("object_key must be a nonempty relative key")
        points = _canonical_float64(self.points_xyz, shape_tail=(3,))
        transform = np.asarray(self.reference_from_sensor, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise StreamingTargetMapError("reference_from_sensor must be finite 4x4")
        transform = np.ascontiguousarray(transform, dtype="<f8")
        if int(self.remote_size_bytes) < 0:
            raise StreamingTargetMapError("remote_size_bytes must be nonnegative")
        gt_sha256 = _require_sha256(self.gt_sha256, field="gt_sha256")
        calibration_sha256 = _require_sha256(
            self.calibration_sha256, field="calibration_sha256"
        )
        object_sha256 = self.object_sha256
        if self.object_sha256 is not None:
            object_sha256 = _require_sha256(self.object_sha256, field="object_sha256")
        return MapScan(
            ordinal=ordinal,
            object_key=str(self.object_key),
            points_xyz=points,
            reference_from_sensor=transform,
            remote_size_bytes=int(self.remote_size_bytes),
            etag=str(self.etag),
            last_modified=str(self.last_modified),
            gt_sha256=gt_sha256,
            calibration_sha256=calibration_sha256,
            object_sha256=object_sha256,
        )


@dataclass(frozen=True)
class TargetMapResult:
    points_xyz: np.ndarray
    voxel_keys: np.ndarray
    target_map_sha256: str
    final_state_sha256: str
    voxel_rule_sha256: str
    processed_object_count: int

    def as_manifest(self) -> dict[str, Any]:
        return {
            "final_order": "LEXICOGRAPHIC_VOXEL_KEY_X_Y_Z",
            "final_state_sha256": self.final_state_sha256,
            "numeric_dtype": "float64-little-endian",
            "processed_object_count": self.processed_object_count,
            "schema": MAP_RESULT_SCHEMA,
            "target_map_sha256": self.target_map_sha256,
            "target_map_sha_semantics": "CANONICAL_FLOAT64_NPY_PAYLOAD_SHA256",
            "voxel_count": int(self.points_xyz.shape[0]),
            "voxel_rule_sha256": self.voxel_rule_sha256,
        }


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    # Express the operation explicitly to avoid a BLAS-dependent matrix product.
    result = np.empty_like(points, dtype="<f8")
    for row_index in range(points.shape[0]):
        x, y, z = (float(value) for value in points[row_index])
        for dimension in range(3):
            result[row_index, dimension] = (
                transform[dimension, 0] * x
                + transform[dimension, 1] * y
                + transform[dimension, 2] * z
                + transform[dimension, 3]
            )
    if not np.all(np.isfinite(result)):
        raise StreamingTargetMapError("reference transform produced non-finite points")
    return result


class StreamingTargetMapBuilder:
    """Fixed-order float64 voxel accumulator with authenticated resume state."""

    def __init__(self, voxel_rule: VoxelRule, *, worker_count: int = 1) -> None:
        if not isinstance(voxel_rule, VoxelRule):
            raise StreamingTargetMapError("an explicit VoxelRule is required")
        if int(worker_count) <= 0:
            raise StreamingTargetMapError("worker_count must be positive")
        self.voxel_rule = voxel_rule
        self.worker_count = int(worker_count)  # Operational only; excluded from state/hash.
        self._voxels: dict[tuple[int, int, int], list[Any]] = {}
        self._processed: list[dict[str, Any]] = []
        self._next_ordinal = 0

    @property
    def next_ordinal(self) -> int:
        return self._next_ordinal

    @property
    def processed_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(json.dumps(row)) for row in self._processed)

    def _core_payload(self) -> dict[str, Any]:
        voxels = []
        for key in sorted(self._voxels):
            count, sums = self._voxels[key]
            voxels.append(
                {
                    "count": int(count),
                    "key": list(key),
                    "sum_xyz_float_hex": [float(value).hex() for value in sums],
                }
            )
        return {
            "next_input_ordinal": self._next_ordinal,
            "voxel_rule": self.voxel_rule.as_dict(),
            "voxel_rule_sha256": self.voxel_rule.contract_sha256,
            "voxels": voxels,
        }

    def _state_payload(self) -> dict[str, Any]:
        return {
            "core": self._core_payload(),
            "processed_objects": self._processed,
            "schema": MAP_STATE_SCHEMA,
        }

    @property
    def state_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self._state_payload())).hexdigest()

    @property
    def _core_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self._core_payload())).hexdigest()

    def _identity(self, scan: MapScan, scan_sha256: str) -> dict[str, Any]:
        return {
            "calibration_sha256": scan.calibration_sha256,
            "etag": scan.etag,
            "gt_sha256": scan.gt_sha256,
            "last_modified": scan.last_modified,
            "local_temporary_sha256": scan.object_sha256,
            "object_key": scan.object_key,
            "ordinal": scan.ordinal,
            "processing_contract_sha256": self.voxel_rule.contract_sha256,
            "remote_size_bytes": scan.remote_size_bytes,
            "scan_and_transform_sha256": scan_sha256,
        }

    def _scan_sha256(self, scan: MapScan) -> str:
        digest = hashlib.sha256()
        digest.update(canonical_array_sha256(scan.points_xyz).encode("ascii"))
        digest.update(canonical_array_sha256(scan.reference_from_sensor).encode("ascii"))
        return digest.hexdigest()

    def _check_resumed_scan(self, scan: MapScan, identity: Mapping[str, Any]) -> dict[str, Any]:
        if scan.ordinal >= len(self._processed):
            raise StreamingTargetMapError("checkpoint is missing an earlier processed object")
        prior = self._processed[scan.ordinal]
        prior_identity = prior["object_identity"]
        if dict(identity) != prior_identity:
            raise StreamingTargetMapError(
                f"authenticated object changed on resume at ordinal {scan.ordinal}"
            )
        return json.loads(json.dumps(prior))

    def process_scan(self, scan: MapScan) -> dict[str, Any]:
        """Process or authenticate-skip exactly one ordinal.

        Earlier ordinals are never accumulated twice: replaying them after a
        resume only verifies their complete identity.  A changed ETag, payload,
        transform, GT, calibration, or processing contract fails closed.
        """

        scan = scan.validated()
        scan_sha256 = self._scan_sha256(scan)
        identity = self._identity(scan, scan_sha256)
        if scan.ordinal < self._next_ordinal:
            return self._check_resumed_scan(scan, identity)
        if scan.ordinal != self._next_ordinal:
            raise StreamingTargetMapError(
                f"fixed input order violated: expected {self._next_ordinal}, got {scan.ordinal}"
            )
        if any(row["object_identity"]["object_key"] == scan.object_key for row in self._processed):
            raise StreamingTargetMapError(f"duplicate object key: {scan.object_key}")

        before_state_sha256 = self.state_sha256
        before_core_sha256 = self._core_sha256
        transformed = _transform_points(scan.points_xyz, scan.reference_from_sensor)
        origin = np.asarray(self.voxel_rule.origin_xyz_m, dtype=np.float64)
        scaled = np.floor((transformed - origin) / self.voxel_rule.voxel_size_m)
        limit = float(np.iinfo(np.int64).max)
        if np.any(scaled < -limit) or np.any(scaled > limit):
            raise StreamingTargetMapError("voxel key exceeds int64 range")
        voxel_keys = scaled.astype(np.int64)

        # This loop is the single scientific reduction order.  It is not
        # parallelized even when transforms/decoding are performed by workers.
        for point_index in range(transformed.shape[0]):
            key = tuple(int(value) for value in voxel_keys[point_index])
            point = transformed[point_index]
            if key not in self._voxels:
                self._voxels[key] = [0, np.zeros(3, dtype="<f8")]
            cell = self._voxels[key]
            cell[0] += 1
            for dimension in range(3):
                cell[1][dimension] = cell[1][dimension] + point[dimension]

        self._next_ordinal += 1
        after_core_sha256 = self._core_sha256
        transition_payload = {
            "after_core_sha256": after_core_sha256,
            "before_core_sha256": before_core_sha256,
            "before_state_sha256": before_state_sha256,
            "object_identity": identity,
            "previous_transition_sha256": (
                None if not self._processed else self._processed[-1]["map_state_transition_sha256"]
            ),
        }
        transition_sha256 = hashlib.sha256(
            canonical_json_bytes(transition_payload)
        ).hexdigest()
        record = {
            **transition_payload,
            "map_state_transition_sha256": transition_sha256,
        }
        self._processed.append(record)
        return json.loads(json.dumps(record))

    def process_scans(self, scans: Iterable[MapScan]) -> None:
        for scan in scans:
            self.process_scan(scan)

    def checkpoint(self) -> dict[str, Any]:
        payload = self._state_payload()
        return {
            "checkpoint_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
            "state": payload,
        }

    def write_checkpoint(self, path: str | Path) -> None:
        atomic_write_json(path, self.checkpoint())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        expected_voxel_rule: VoxelRule,
        worker_count: int = 1,
    ) -> "StreamingTargetMapBuilder":
        try:
            state = dict(checkpoint["state"])
            claimed_sha = str(checkpoint["checkpoint_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StreamingTargetMapError("invalid target-map checkpoint envelope") from exc
        actual_sha = hashlib.sha256(canonical_json_bytes(state)).hexdigest()
        if claimed_sha != actual_sha:
            raise StreamingTargetMapError("target-map checkpoint SHA mismatch")
        if state.get("schema") != MAP_STATE_SCHEMA:
            raise StreamingTargetMapError("unsupported target-map checkpoint schema")
        core = state.get("core")
        processed = state.get("processed_objects")
        if not isinstance(core, Mapping) or not isinstance(processed, list):
            raise StreamingTargetMapError("invalid target-map checkpoint structure")
        if core.get("voxel_rule") != expected_voxel_rule.as_dict():
            raise StreamingTargetMapError("checkpoint voxel rule differs from frozen rule")
        if core.get("voxel_rule_sha256") != expected_voxel_rule.contract_sha256:
            raise StreamingTargetMapError("checkpoint voxel-rule SHA mismatch")

        result = cls(expected_voxel_rule, worker_count=worker_count)
        initial_core_sha256 = result._core_sha256
        initial_state_sha256 = result.state_sha256
        try:
            result._next_ordinal = int(core["next_input_ordinal"])
            result._voxels = {}
            for row in core["voxels"]:
                key_values = row["key"]
                if len(key_values) != 3:
                    raise ValueError("voxel key dimension")
                key = tuple(int(value) for value in key_values)
                if key in result._voxels:
                    raise ValueError("duplicate voxel key")
                count = int(row["count"])
                if count <= 0:
                    raise ValueError("invalid voxel count")
                sums = np.asarray(
                    [float.fromhex(str(value)) for value in row["sum_xyz_float_hex"]],
                    dtype="<f8",
                )
                if sums.shape != (3,) or not np.all(np.isfinite(sums)):
                    raise ValueError("invalid voxel sums")
                result._voxels[key] = [count, sums]
            result._processed = json.loads(json.dumps(processed))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise StreamingTargetMapError("invalid target-map checkpoint values") from exc

        if result._next_ordinal != len(result._processed):
            raise StreamingTargetMapError("orphan or non-contiguous target-map checkpoint")
        if not result._processed and (result._next_ordinal != 0 or result._voxels):
            raise StreamingTargetMapError("orphan voxel state without processed objects")
        previous_transition: str | None = None
        previous_after_core_sha256 = initial_core_sha256
        for expected_ordinal, row in enumerate(result._processed):
            try:
                identity = row["object_identity"]
                if int(identity["ordinal"]) != expected_ordinal:
                    raise ValueError("ordinal")
                payload = {
                    key: row[key]
                    for key in (
                        "after_core_sha256",
                        "before_core_sha256",
                        "before_state_sha256",
                        "object_identity",
                        "previous_transition_sha256",
                    )
                }
                transition_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
                if transition_sha != row["map_state_transition_sha256"]:
                    raise ValueError("transition SHA")
                if row["previous_transition_sha256"] != previous_transition:
                    raise ValueError("transition chain")
                if row["before_core_sha256"] != previous_after_core_sha256:
                    raise ValueError("core transition chain")
                if expected_ordinal == 0 and row["before_state_sha256"] != initial_state_sha256:
                    raise ValueError("initial state anchor")
                previous_transition = transition_sha
                previous_after_core_sha256 = row["after_core_sha256"]
            except (KeyError, TypeError, ValueError) as exc:
                raise StreamingTargetMapError(
                    "orphan or corrupted target-map transition record"
                ) from exc
        if result._processed and result._processed[-1]["after_core_sha256"] != result._core_sha256:
            raise StreamingTargetMapError("checkpoint core is orphaned from transition chain")
        if result.state_sha256 != claimed_sha:
            raise StreamingTargetMapError("reconstructed target-map state SHA mismatch")
        return result

    @classmethod
    def read_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_voxel_rule: VoxelRule,
        worker_count: int = 1,
    ) -> "StreamingTargetMapBuilder":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StreamingTargetMapError("unable to read target-map checkpoint") from exc
        return cls.from_checkpoint(
            value, expected_voxel_rule=expected_voxel_rule, worker_count=worker_count
        )

    def finalize(self) -> TargetMapResult:
        keys = sorted(self._voxels)
        voxel_keys = np.asarray(keys, dtype="<i8").reshape((-1, 3))
        points = np.empty((len(keys), 3), dtype="<f8")
        for index, key in enumerate(keys):
            count, sums = self._voxels[key]
            points[index] = sums / np.float64(count)
        # Canonicalize signed zero without affecting any nonzero value.
        points[points == 0.0] = 0.0
        return TargetMapResult(
            points_xyz=points,
            voxel_keys=voxel_keys,
            # This is the physical content-address used by ContentAddressedStore.
            # Voxel keys, rule, transition chain, and input identities remain
            # separately authenticated in the final state/checkpoint hashes.
            target_map_sha256=hashlib.sha256(
                canonical_float64_npy_bytes(points)
            ).hexdigest(),
            final_state_sha256=self.state_sha256,
            voxel_rule_sha256=self.voxel_rule.contract_sha256,
            processed_object_count=self._next_ordinal,
        )


@dataclass(frozen=True)
class ReplayMapObject:
    """Frozen remote identity used to lay out the production replay array.

    Boreas lidar objects contain six little-endian float32 values per point,
    so the frozen remote byte size divided by 24 is the only permitted source
    of the replay capacity.  The replay retains XYZ as three little-endian
    float64 values: also 24 bytes per point.
    """

    ordinal: int
    object_key: str
    remote_size_bytes: int
    etag: str
    last_modified: str

    @classmethod
    def from_value(cls, value: "ReplayMapObject | Mapping[str, Any]") -> "ReplayMapObject":
        if isinstance(value, cls):
            candidate = value
        elif isinstance(value, Mapping):
            candidate = cls(
                ordinal=int(value.get("ordinal", -1)),
                object_key=str(value.get("object_key", value.get("s3_key", value.get("key", "")))),
                remote_size_bytes=int(
                    value.get("remote_size_bytes", value.get("size_bytes", -1))
                ),
                etag=str(value.get("etag", "")),
                last_modified=str(value.get("last_modified", "")),
            )
        else:
            raise StreamingTargetMapError("replay allowlist row must be an object")
        if candidate.ordinal < 0:
            raise StreamingTargetMapError("replay ordinal must be nonnegative")
        if not candidate.object_key or candidate.object_key.startswith("/"):
            raise StreamingTargetMapError("replay object_key must be a relative key")
        if not candidate.etag or any(character.isspace() for character in candidate.etag):
            raise StreamingTargetMapError("replay ETag must be nonempty without whitespace")
        if not candidate.last_modified:
            raise StreamingTargetMapError("replay LastModified must be nonempty")
        if candidate.remote_size_bytes <= 0:
            raise StreamingTargetMapError("replay remote size must be positive")
        if candidate.remote_size_bytes % BOREAS_RAW_POINT_STRIDE_BYTES:
            raise StreamingTargetMapError(
                "Boreas raw object size is not divisible by the frozen 24-byte point stride"
            )
        return candidate

    @property
    def point_capacity(self) -> int:
        return self.remote_size_bytes // BOREAS_RAW_POINT_STRIDE_BYTES

    def identity(self) -> dict[str, Any]:
        return {
            "etag": self.etag,
            "last_modified": self.last_modified,
            "object_key": self.object_key,
            "ordinal": self.ordinal,
            "remote_size_bytes": self.remote_size_bytes,
        }


def _compact_json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _ledger_record_sha256(envelope_without_digest: Mapping[str, Any]) -> str:
    return hashlib.sha256(_compact_json_line(envelope_without_digest)).hexdigest()


def _ledger_envelope(
    *, sequence_number: int, previous_record_sha256: str, record_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    envelope = {
        "payload": dict(payload),
        "previous_record_sha256": previous_record_sha256,
        "record_type": record_type,
        "schema": PRODUCTION_REPLAY_LEDGER_SCHEMA,
        "sequence_number": sequence_number,
    }
    envelope["record_sha256"] = _ledger_record_sha256(envelope)
    return envelope


def _parse_replay_ledger(payload: bytes) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise StreamingTargetMapError("production replay ledger is empty or truncated")
    result: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    for sequence_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            envelope = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StreamingTargetMapError("production replay ledger is invalid JSONL") from exc
        if not isinstance(envelope, dict) or _compact_json_line(envelope) != line:
            raise StreamingTargetMapError("production replay ledger is not canonical JSONL")
        if set(envelope) != {
            "payload",
            "previous_record_sha256",
            "record_sha256",
            "record_type",
            "schema",
            "sequence_number",
        }:
            raise StreamingTargetMapError("production replay ledger field set differs")
        if envelope["schema"] != PRODUCTION_REPLAY_LEDGER_SCHEMA:
            raise StreamingTargetMapError("production replay ledger schema differs")
        if envelope["sequence_number"] != sequence_number:
            raise StreamingTargetMapError("production replay ledger sequence is non-contiguous")
        if envelope["previous_record_sha256"] != previous:
            raise StreamingTargetMapError("production replay ledger hash chain breaks")
        digest_payload = dict(envelope)
        supplied = digest_payload.pop("record_sha256")
        actual = _ledger_record_sha256(digest_payload)
        if supplied != actual:
            raise StreamingTargetMapError("production replay ledger record SHA differs")
        if not isinstance(envelope["payload"], dict):
            raise StreamingTargetMapError("production replay ledger payload is not an object")
        result.append(envelope)
        previous = actual
    if result[0]["record_type"] != "PLAN" or any(
        row["record_type"] != "SCAN" for row in result[1:]
    ):
        raise StreamingTargetMapError("production replay ledger record ordering differs")
    return result


def production_replay_plan_payload(
    allowlist: Sequence[ReplayMapObject | Mapping[str, Any]],
    *,
    replay_path: str | Path,
    processing_contract_sha256: str,
    gt_sha256: str,
    calibration_sha256: str,
    voxel_rule_sha256: str,
    track_python_voxel_state: bool,
) -> dict[str, Any]:
    """Return the canonical fixed replay plan without materializing storage."""

    rows = tuple(ReplayMapObject.from_value(row) for row in allowlist)
    if [row.ordinal for row in rows] != list(range(len(rows))):
        raise StreamingTargetMapError(
            "replay allowlist ordinals must be exactly 0..N-1"
        )
    if len({row.object_key for row in rows}) != len(rows):
        raise StreamingTargetMapError("replay allowlist object keys must be unique")
    replay = Path(replay_path)
    if not replay.is_absolute():
        raise StreamingTargetMapError("replay array path must be absolute")
    if not isinstance(track_python_voxel_state, bool):
        raise StreamingTargetMapError("track_python_voxel_state must be boolean")
    plan_rows: list[dict[str, Any]] = []
    byte_start = 0
    for item in rows:
        byte_count = item.point_capacity * REPLAY_POINT_STRIDE_BYTES
        plan_rows.append(
            {
                **item.identity(),
                "byte_end_exclusive": byte_start + byte_count,
                "byte_start": byte_start,
                "point_capacity": item.point_capacity,
            }
        )
        byte_start += byte_count
    return {
        "allowlist": plan_rows,
        "allowlist_sha256": hashlib.sha256(
            canonical_json_bytes(plan_rows)
        ).hexdigest(),
        "boreas_raw_point_stride_bytes": BOREAS_RAW_POINT_STRIDE_BYTES,
        "calibration_sha256": _require_sha256(
            calibration_sha256, field="calibration_sha256"
        ),
        "gt_sha256": _require_sha256(gt_sha256, field="gt_sha256"),
        "numeric_format": PRODUCTION_REPLAY_FORMAT,
        "processing_contract_sha256": _require_sha256(
            processing_contract_sha256, field="processing_contract_sha256"
        ),
        "padding_rule": PRODUCTION_REPLAY_PADDING_RULE,
        "replay_path": str(replay),
        "replay_point_stride_bytes": REPLAY_POINT_STRIDE_BYTES,
        "track_python_voxel_state": track_python_voxel_state,
        "total_point_capacity": sum(row.point_capacity for row in rows),
        "total_replay_bytes": byte_start,
        "voxel_rule_sha256": _require_sha256(
            voxel_rule_sha256, field="voxel_rule_sha256"
        ),
    }


def production_replay_allocation_evidence(
    plan_payload: Mapping[str, Any], *, ledger_path: str | Path
) -> tuple[dict[str, Any], bytes]:
    """Return exact allocation intent and PLAN bytes for pre-gate recovery."""

    replay_path = Path(str(plan_payload.get("replay_path", "")))
    ledger = Path(ledger_path)
    if not replay_path.is_absolute() or not ledger.is_absolute():
        raise StreamingTargetMapError("replay allocation paths must be absolute")
    total = plan_payload.get("total_replay_bytes")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise StreamingTargetMapError("replay allocation byte count is invalid")
    unsigned = {
        "ledger_path": str(ledger),
        "plan_payload_sha256": hashlib.sha256(
            canonical_json_bytes(dict(plan_payload))
        ).hexdigest(),
        "replay_path": str(replay_path),
        "schema": PRODUCTION_REPLAY_ALLOCATION_INTENT_SCHEMA,
        "total_replay_bytes": total,
    }
    intent = {
        **unsigned,
        "intent_sha256": hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    }
    plan = _ledger_envelope(
        sequence_number=1,
        previous_record_sha256=ZERO_SHA256,
        record_type="PLAN",
        payload=plan_payload,
    )
    return intent, _compact_json_line(plan)


def _safe_zero_managed_file(path: Path, *, maximum_size: int, label: str) -> os.stat_result:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or os.lstat(path).st_nlink != 1
    ):
        raise StreamingTargetMapError(f"{label} is unsafe")
    descriptor = _open_replay_file(path, os.O_RDONLY)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_size > maximum_size:
            raise StreamingTargetMapError(f"{label} exceeds the planned allocation")
        offset = 0
        while offset < metadata.st_size:
            block = os.pread(
                descriptor,
                min(8 * 1024 * 1024, metadata.st_size - offset),
                offset,
            )
            if not block or any(block):
                raise StreamingTargetMapError(f"{label} contains nonzero bytes")
            offset += len(block)
        return metadata
    finally:
        os.close(descriptor)


def recover_production_replay_allocation_before_gate(
    *,
    plan_payload: Mapping[str, Any],
    replay_path: str | Path,
    ledger_path: str | Path,
) -> bool:
    """Authenticate/repair allocation-only crash state before disk RESUME.

    Returns ``True`` only when the complete nonsparse replay allocation remains.
    A zero partial/sparse allocation and a truncated exact PLAN prefix are
    safely removed under the exact durable intent, allowing a fresh start gate.
    """

    replay = Path(replay_path)
    ledger = Path(ledger_path)
    intent_path = ledger.parent / "replay_allocation_intent.json"
    replay_exists = replay.exists() or replay.is_symlink()
    ledger_exists = ledger.exists() or ledger.is_symlink()
    intent_exists = intent_path.exists() or intent_path.is_symlink()
    if not intent_exists:
        if replay_exists != ledger_exists:
            raise StreamingTargetMapError("orphan replay array or ledger")
        if not replay_exists:
            return False
        total = plan_payload.get("total_replay_bytes")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise StreamingTargetMapError("replay allocation byte count is invalid")
        for path, label in ((replay, "replay array"), (ledger, "replay ledger")):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.resolve(strict=True) != path
                or os.lstat(path).st_nlink != 1
            ):
                raise StreamingTargetMapError(f"{label} is unsafe")
        replay_metadata = os.lstat(replay)
        if replay_metadata.st_size != total:
            raise StreamingTargetMapError("replay array byte size differs")
        if replay_metadata.st_blocks * 512 < total:
            raise StreamingTargetMapError(
                "replay array is sparse or incompletely allocated"
            )
        if os.lstat(ledger).st_size <= 0:
            raise StreamingTargetMapError("replay ledger is empty")
        return True
    expected_intent, expected_plan_line = production_replay_allocation_evidence(
        plan_payload, ledger_path=ledger
    )
    if (
        intent_path.is_symlink()
        or not intent_path.is_file()
        or intent_path.resolve(strict=True) != intent_path
        or os.lstat(intent_path).st_nlink != 1
    ):
        raise StreamingTargetMapError("replay allocation intent is unsafe")
    try:
        intent_raw = intent_path.read_bytes()
        intent = json.loads(intent_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StreamingTargetMapError("replay allocation intent is invalid") from exc
    if intent_raw != canonical_json_bytes(intent) or intent != expected_intent:
        raise StreamingTargetMapError("replay allocation intent binding differs")

    if ledger_exists:
        if (
            ledger.is_symlink()
            or not ledger.is_file()
            or ledger.resolve(strict=True) != ledger
            or os.lstat(ledger).st_nlink != 1
        ):
            raise StreamingTargetMapError("replay allocation ledger is unsafe")
        ledger_raw = ledger.read_bytes()
        if not expected_plan_line.startswith(ledger_raw):
            raise StreamingTargetMapError(
                "replay allocation ledger is not an exact PLAN prefix"
            )
    total = int(plan_payload["total_replay_bytes"])
    complete = False
    if replay_exists:
        metadata = _safe_zero_managed_file(
            replay, maximum_size=total, label="replay allocation"
        )
        complete = metadata.st_size == total and metadata.st_blocks * 512 >= total
    if complete:
        return True

    for path in (ledger, replay):
        if path.exists():
            path.unlink()
            _fsync_directory(path.parent)
    intent_path.unlink()
    _fsync_directory(intent_path.parent)
    return False


def _safe_absolute_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise StreamingTargetMapError(f"{label} path must be absolute")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate.parent.is_symlink() or candidate.parent.resolve(strict=True) != candidate.parent:
        raise StreamingTargetMapError(f"{label} parent is not canonical")
    if candidate.exists() and (candidate.is_symlink() or not candidate.is_file()):
        raise StreamingTargetMapError(f"{label} path is unsafe")
    return candidate


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _ReplayFileIdentity:
    device: int
    inode: int
    size_bytes: int
    modified_time_ns: int
    changed_time_ns: int


def _replay_file_identity(descriptor: int) -> _ReplayFileIdentity:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise StreamingTargetMapError("production replay path is not a regular file")
    return _ReplayFileIdentity(
        device=int(metadata.st_dev),
        inode=int(metadata.st_ino),
        size_bytes=int(metadata.st_size),
        modified_time_ns=int(metadata.st_mtime_ns),
        changed_time_ns=int(metadata.st_ctime_ns),
    )


def _same_fixed_replay_file(
    left: _ReplayFileIdentity, right: _ReplayFileIdentity
) -> bool:
    return (
        left.device,
        left.inode,
        left.size_bytes,
    ) == (
        right.device,
        right.inode,
        right.size_bytes,
    )


def _open_replay_file(path: Path, access_flags: int) -> int:
    flags = access_flags | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise StreamingTargetMapError(
            f"cannot safely open production replay array: {path}"
        ) from exc


def _read_exact_range_from_descriptor(
    descriptor: int,
    start: int,
    length: int,
    *,
    expected_identity: _ReplayFileIdentity,
) -> bytes:
    before = _replay_file_identity(descriptor)
    if before != expected_identity:
        raise StreamingTargetMapError("production replay file identity changed")
    if start < 0 or length < 0 or start + length > before.size_bytes:
        raise StreamingTargetMapError("production replay byte range exceeds the fixed array")
    chunks: list[bytes] = []
    offset = start
    remaining = length
    while remaining:
        block = os.pread(descriptor, min(8 * 1024 * 1024, remaining), offset)
        if not block:
            raise StreamingTargetMapError("production replay array has a short byte range")
        chunks.append(block)
        offset += len(block)
        remaining -= len(block)
    if _replay_file_identity(descriptor) != before:
        raise StreamingTargetMapError("production replay file identity changed during read")
    return b"".join(chunks)


def _read_exact_range(
    path: Path,
    start: int,
    length: int,
    *,
    expected_identity: _ReplayFileIdentity,
) -> bytes:
    descriptor = _open_replay_file(path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        return _read_exact_range_from_descriptor(
            descriptor,
            start,
            length,
            expected_identity=expected_identity,
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _capture_replay_file_identity(path: Path) -> _ReplayFileIdentity:
    descriptor = _open_replay_file(path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        return _replay_file_identity(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class ProductionMapReplayArray:
    """Disk-bounded, authenticated map ingestion for the real Stage-2 runner.

    Creation physically preallocates one fixed-size XYZ array.  A scan range is
    written and fsynced before its canonical ledger record is appended and
    fsynced.  Therefore bytes without a ledger record are never trusted, while
    every completed prefix can be verified and replayed without reacquiring its
    remote object.
    """

    def __init__(
        self,
        allowlist: Sequence["ReplayMapObject | Mapping[str, Any]"],
        *,
        replay_path: str | Path,
        ledger_path: str | Path,
        processing_contract_sha256: str,
        gt_sha256: str,
        calibration_sha256: str,
        voxel_rule: VoxelRule,
        voxel_rule_sha256: str | None = None,
        track_python_voxel_state: bool = True,
        allocation_fault_hook: Any | None = None,
    ) -> None:
        self.allowlist = tuple(ReplayMapObject.from_value(row) for row in allowlist)
        if [row.ordinal for row in self.allowlist] != list(range(len(self.allowlist))):
            raise StreamingTargetMapError("replay allowlist ordinals must be exactly 0..N-1")
        keys = [row.object_key for row in self.allowlist]
        if len(keys) != len(set(keys)):
            raise StreamingTargetMapError("replay allowlist object keys must be unique")
        self.processing_contract_sha256 = _require_sha256(
            processing_contract_sha256, field="processing_contract_sha256"
        )
        if not isinstance(voxel_rule, VoxelRule):
            raise StreamingTargetMapError(
                "production replay requires the actual authenticated VoxelRule"
            )
        self.voxel_rule_sha256 = _require_sha256(
            voxel_rule_sha256 or voxel_rule.contract_sha256,
            field="voxel_rule_sha256",
        )
        if voxel_rule.contract_sha256 != self.voxel_rule_sha256:
            raise StreamingTargetMapError("voxel rule differs from supplied replay binding")
        self.voxel_rule = voxel_rule
        if not isinstance(track_python_voxel_state, bool):
            raise StreamingTargetMapError("track_python_voxel_state must be boolean")
        self.track_python_voxel_state = track_python_voxel_state
        if allocation_fault_hook is not None and not callable(allocation_fault_hook):
            raise StreamingTargetMapError("allocation_fault_hook must be callable")
        self._allocation_fault_hook = allocation_fault_hook
        self.gt_sha256 = _require_sha256(gt_sha256, field="gt_sha256")
        self.calibration_sha256 = _require_sha256(
            calibration_sha256, field="calibration_sha256"
        )
        self.replay_path = _safe_absolute_file(replay_path, label="replay array")
        self.ledger_path = _safe_absolute_file(ledger_path, label="replay ledger")
        if self.replay_path == self.ledger_path:
            raise StreamingTargetMapError("replay array and ledger paths must differ")
        self._plan_payload = self._make_plan_payload()
        self.allocation_intent_path = (
            self.ledger_path.parent / "replay_allocation_intent.json"
        )
        if self.allocation_intent_path in {self.replay_path, self.ledger_path}:
            raise StreamingTargetMapError("replay allocation intent path collides")
        replay_exists = self.replay_path.exists() or self.replay_path.is_symlink()
        ledger_exists = self.ledger_path.exists() or self.ledger_path.is_symlink()
        intent_exists = (
            self.allocation_intent_path.exists()
            or self.allocation_intent_path.is_symlink()
        )
        if replay_exists != ledger_exists and not intent_exists:
            raise StreamingTargetMapError("orphan replay array or ledger")
        if intent_exists:
            self._recover_allocation_intent()
            replay_exists = self.replay_path.exists() or self.replay_path.is_symlink()
            ledger_exists = self.ledger_path.exists() or self.ledger_path.is_symlink()
        if replay_exists != ledger_exists:
            raise StreamingTargetMapError("orphan replay array or ledger")
        if not replay_exists:
            self._initialize_files()
        self._replay_file_identity = _capture_replay_file_identity(self.replay_path)
        # The Python dictionary accumulator is an exact synthetic/reference
        # primitive, but is not bounded for the real 8202-scan map.  Production
        # ingestion may authenticate replay ranges without constructing it; an
        # independent exact-order external reducer then consumes those ranges.
        self._map_builder = (
            StreamingTargetMapBuilder(voxel_rule)
            if self.track_python_voxel_state
            else None
        )
        self._records = self._load_and_verify()

    def _make_plan_payload(self) -> dict[str, Any]:
        return production_replay_plan_payload(
            self.allowlist,
            replay_path=self.replay_path,
            processing_contract_sha256=self.processing_contract_sha256,
            gt_sha256=self.gt_sha256,
            calibration_sha256=self.calibration_sha256,
            voxel_rule_sha256=self.voxel_rule_sha256,
            track_python_voxel_state=self.track_python_voxel_state,
        )

    def _initialize_files(self) -> None:
        intent = self._allocation_intent()
        if self.allocation_intent_path.exists() or self.allocation_intent_path.is_symlink():
            raise StreamingTargetMapError("replay allocation intent already exists")
        atomic_write_json(self.allocation_intent_path, intent, overwrite=False)
        _fsync_directory(self.allocation_intent_path.parent)
        self._allocation_fault("AFTER_ALLOCATION_INTENT")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        replay_descriptor = os.open(self.replay_path, flags, 0o600)
        try:
            size = int(self._plan_payload["total_replay_bytes"])
            if size and hasattr(os, "posix_fallocate"):
                os.posix_fallocate(replay_descriptor, 0, size)
            else:
                os.ftruncate(replay_descriptor, size)
            os.fsync(replay_descriptor)
        finally:
            os.close(replay_descriptor)
        _fsync_directory(self.replay_path.parent)
        self._allocation_fault("AFTER_REPLAY_ALLOCATION")
        plan = _ledger_envelope(
            sequence_number=1,
            previous_record_sha256=ZERO_SHA256,
            record_type="PLAN",
            payload=self._plan_payload,
        )
        ledger_flags = flags | os.O_APPEND
        try:
            ledger_descriptor = os.open(self.ledger_path, ledger_flags, 0o600)
        except Exception:
            # The orphan is intentionally left visible and will fail closed on resume.
            raise
        try:
            view = memoryview(_compact_json_line(plan))
            while view:
                written = os.write(ledger_descriptor, view)
                if written <= 0:
                    raise OSError("zero-byte production replay plan write")
                view = view[written:]
            os.fsync(ledger_descriptor)
        finally:
            os.close(ledger_descriptor)
        _fsync_directory(self.replay_path.parent)
        if self.ledger_path.parent != self.replay_path.parent:
            _fsync_directory(self.ledger_path.parent)
        self._allocation_fault("AFTER_PLAN_LEDGER")
        self._clear_allocation_intent()

    def _allocation_fault(self, label: str) -> None:
        if self._allocation_fault_hook is not None:
            self._allocation_fault_hook(label)

    def _allocation_intent(self) -> dict[str, Any]:
        return production_replay_allocation_evidence(
            self._plan_payload, ledger_path=self.ledger_path
        )[0]

    def _load_allocation_intent(self) -> dict[str, Any]:
        path = self.allocation_intent_path
        if path.is_symlink() or not path.is_file() or os.lstat(path).st_nlink != 1:
            raise StreamingTargetMapError("replay allocation intent is unsafe")
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StreamingTargetMapError("replay allocation intent is invalid") from exc
        if (
            not isinstance(value, dict)
            or raw != canonical_json_bytes(value)
            or value != self._allocation_intent()
        ):
            raise StreamingTargetMapError("replay allocation intent binding differs")
        return value

    def _clear_allocation_intent(self) -> None:
        path = self.allocation_intent_path
        if not path.exists():
            return
        self._load_allocation_intent()
        path.unlink()
        _fsync_directory(path.parent)

    @staticmethod
    def _is_all_zero_file(path: Path, size: int) -> bool:
        descriptor = _open_replay_file(path, os.O_RDONLY)
        try:
            metadata = os.fstat(descriptor)
            if (
                metadata.st_size != size
                or metadata.st_nlink != 1
                or metadata.st_blocks * 512 < size
            ):
                return False
            offset = 0
            while offset < size:
                block = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
                if not block or any(block):
                    return False
                offset += len(block)
            return True
        finally:
            os.close(descriptor)

    def _recover_allocation_intent(self) -> None:
        """Close the only legal replay-without-PLAN crash windows."""

        self._load_allocation_intent()
        replay_exists = self.replay_path.exists()
        ledger_exists = self.ledger_path.exists()
        expected_plan = _compact_json_line(
            _ledger_envelope(
                sequence_number=1,
                previous_record_sha256=ZERO_SHA256,
                record_type="PLAN",
                payload=self._plan_payload,
            )
        )
        if ledger_exists:
            if (
                self.ledger_path.is_symlink()
                or not self.ledger_path.is_file()
                or self.ledger_path.resolve(strict=True) != self.ledger_path
                or os.lstat(self.ledger_path).st_nlink != 1
            ):
                raise StreamingTargetMapError("replay allocation ledger is unsafe")
            ledger_raw = self.ledger_path.read_bytes()
            if not expected_plan.startswith(ledger_raw):
                raise StreamingTargetMapError(
                    "allocation intent accompanies a non-PLAN replay ledger prefix"
                )
            if ledger_raw != expected_plan:
                self.ledger_path.unlink()
                _fsync_directory(self.ledger_path.parent)
                ledger_exists = False
        if not replay_exists:
            if ledger_exists:
                self.ledger_path.unlink()
                _fsync_directory(self.ledger_path.parent)
            self._clear_allocation_intent()
            return
        total = int(self._plan_payload["total_replay_bytes"])
        if replay_exists:
            metadata = _safe_zero_managed_file(
                self.replay_path,
                maximum_size=total,
                label="allocation intent replay",
            )
            complete = metadata.st_size == total and metadata.st_blocks * 512 >= total
            if not complete:
                if ledger_exists:
                    self.ledger_path.unlink()
                    _fsync_directory(self.ledger_path.parent)
                self.replay_path.unlink()
                _fsync_directory(self.replay_path.parent)
                self._clear_allocation_intent()
                return
        if ledger_exists:
            if not replay_exists or not self._is_all_zero_file(
                self.replay_path, total
            ):
                raise StreamingTargetMapError(
                    "allocation intent replay is not exact nonsparse zero-filled storage"
                )
            self._clear_allocation_intent()
            return
        if replay_exists:
            if not self._is_all_zero_file(
                self.replay_path, int(self._plan_payload["total_replay_bytes"])
            ):
                raise StreamingTargetMapError(
                    "allocation intent replay is not exact nonsparse zero-filled storage"
                )
            plan = _ledger_envelope(
                sequence_number=1,
                previous_record_sha256=ZERO_SHA256,
                record_type="PLAN",
                payload=self._plan_payload,
            )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.ledger_path, flags, 0o600)
            try:
                view = memoryview(_compact_json_line(plan))
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("zero-byte recovered replay plan write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(self.ledger_path.parent)
            self._clear_allocation_intent()
            return
        # Crash after intent but before O_EXCL allocation: restart allocation.
        self._clear_allocation_intent()

    @staticmethod
    def _transition_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key != "replay_state_transition_sha256"
        }

    @staticmethod
    def _authenticated_range_transition_sha256(payload: Mapping[str, Any]) -> str:
        """Bind one active prefix inside an authenticated fixed-capacity range."""

        core = {
            "byte_end_exclusive": payload["byte_end_exclusive"],
            "byte_start": payload["byte_start"],
            "calibration_sha256": payload["calibration_sha256"],
            "etag": payload["etag"],
            "gt_sha256": payload["gt_sha256"],
            "last_modified": payload["last_modified"],
            "local_temporary_sha256": payload["local_temporary_sha256"],
            "map_state_transition_kind": AUTHENTICATED_RANGE_TRANSITION_KIND,
            "object_key": payload["object_key"],
            "ordinal": payload["ordinal"],
            "padding_point_count": payload["padding_point_count"],
            "padding_rule": payload["padding_rule"],
            "point_count": payload["point_count"],
            "previous_map_state_transition_sha256": payload[
                "previous_map_state_transition_sha256"
            ],
            "processing_contract_sha256": payload["processing_contract_sha256"],
            "remote_size_bytes": payload["remote_size_bytes"],
            "replay_range_sha256": payload["replay_range_sha256"],
            "transformed_xyz_sha256": payload["transformed_xyz_sha256"],
        }
        return hashlib.sha256(canonical_json_bytes(core)).hexdigest()

    @staticmethod
    def _active_range_bytes(
        range_bytes: bytes, payload: Mapping[str, Any], plan: Mapping[str, Any]
    ) -> bytes:
        point_count = payload["point_count"]
        capacity = plan["point_capacity"]
        if (
            isinstance(point_count, bool)
            or not isinstance(point_count, int)
            or point_count < 0
            or point_count > capacity
        ):
            raise StreamingTargetMapError(
                "production replay active point count exceeds fixed capacity"
            )
        if payload["padding_point_count"] != capacity - point_count:
            raise StreamingTargetMapError("production replay padding point count differs")
        if payload["padding_rule"] != PRODUCTION_REPLAY_PADDING_RULE:
            raise StreamingTargetMapError("production replay padding rule differs")
        active_end = point_count * REPLAY_POINT_STRIDE_BYTES
        if any(range_bytes[active_end:]):
            raise StreamingTargetMapError("production replay inactive padding is nonzero")
        return range_bytes[:active_end]

    def _load_and_verify(self) -> list[dict[str, Any]]:
        if self.replay_path.is_symlink() or self.ledger_path.is_symlink():
            raise StreamingTargetMapError("production replay files cannot be symlinks")
        expected_size = int(self._plan_payload["total_replay_bytes"])
        if self.replay_path.stat().st_size != expected_size:
            raise StreamingTargetMapError("production replay array size differs from plan")
        envelopes = _parse_replay_ledger(self.ledger_path.read_bytes())
        if envelopes[0]["payload"] != self._plan_payload:
            raise StreamingTargetMapError("production replay plan differs from frozen allowlist")
        completed: list[dict[str, Any]] = []
        previous_transition = ZERO_SHA256
        previous_map_transition = ZERO_SHA256
        previous_end = 0
        if len(envelopes) - 1 > len(self.allowlist):
            raise StreamingTargetMapError("production replay ledger exceeds the allowlist")
        for expected_ordinal, envelope in enumerate(envelopes[1:]):
            payload = envelope["payload"]
            plan = self._plan_payload["allowlist"][expected_ordinal]
            required = {
                "byte_end_exclusive",
                "byte_start",
                "calibration_sha256",
                "completed_at_utc",
                "etag",
                "gt_sha256",
                "last_modified",
                "local_temporary_sha256",
                "map_state_transition_sha256",
                "map_state_transition_kind",
                "object_key",
                "ordinal",
                "padding_point_count",
                "padding_rule",
                "point_count",
                "previous_map_state_transition_sha256",
                "previous_replay_state_transition_sha256",
                "processing_contract_sha256",
                "remote_size_bytes",
                "replay_range_sha256",
                "replay_state_transition_sha256",
                "transformed_xyz_sha256",
            }
            if set(payload) != required:
                raise StreamingTargetMapError("production replay scan ledger fields differ")
            for field in (
                "ordinal",
                "object_key",
                "remote_size_bytes",
                "etag",
                "last_modified",
                "byte_start",
                "byte_end_exclusive",
            ):
                if payload[field] != plan[field]:
                    raise StreamingTargetMapError(
                        f"production replay identity/range differs at ordinal {expected_ordinal}"
                    )
            if payload["ordinal"] != expected_ordinal or payload["byte_start"] != previous_end:
                raise StreamingTargetMapError("production replay completed ranges are not a prefix")
            if payload["previous_map_state_transition_sha256"] != previous_map_transition:
                raise StreamingTargetMapError("production replay map-state transition chain breaks")
            expected_transition_kind = (
                PYTHON_VOXEL_TRANSITION_KIND
                if self.track_python_voxel_state
                else AUTHENTICATED_RANGE_TRANSITION_KIND
            )
            if payload["map_state_transition_kind"] != expected_transition_kind:
                raise StreamingTargetMapError(
                    "production replay map-state transition kind differs"
                )
            for field, expected in (
                ("processing_contract_sha256", self.processing_contract_sha256),
                ("gt_sha256", self.gt_sha256),
                ("calibration_sha256", self.calibration_sha256),
            ):
                if payload[field] != expected:
                    raise StreamingTargetMapError(f"production replay {field} differs")
            for field in (
                "local_temporary_sha256",
                "replay_range_sha256",
                "transformed_xyz_sha256",
            ):
                _require_sha256(str(payload[field]), field=field)
            try:
                completed_at = datetime.fromisoformat(
                    str(payload["completed_at_utc"]).replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise StreamingTargetMapError(
                    "production replay completion time is invalid"
                ) from exc
            if (
                completed_at.tzinfo is None
                or completed_at.utcoffset() is None
                or completed_at.utcoffset().total_seconds() != 0
            ):
                raise StreamingTargetMapError(
                    "production replay completion time must be UTC"
                )
            if payload["previous_replay_state_transition_sha256"] != previous_transition:
                raise StreamingTargetMapError("production replay state transition chain breaks")
            actual_transition = hashlib.sha256(
                canonical_json_bytes(self._transition_payload(payload))
            ).hexdigest()
            if payload["replay_state_transition_sha256"] != actual_transition:
                raise StreamingTargetMapError("production replay state transition SHA differs")
            byte_count = payload["byte_end_exclusive"] - payload["byte_start"]
            range_bytes = _read_exact_range(
                self.replay_path,
                payload["byte_start"],
                byte_count,
                expected_identity=self._replay_file_identity,
            )
            if hashlib.sha256(range_bytes).hexdigest() != payload["replay_range_sha256"]:
                raise StreamingTargetMapError("authenticated production replay byte range differs")
            active_bytes = self._active_range_bytes(range_bytes, payload, plan)
            if hashlib.sha256(active_bytes).hexdigest() != payload["transformed_xyz_sha256"]:
                raise StreamingTargetMapError(
                    "authenticated production replay active prefix differs"
                )
            if self.track_python_voxel_state:
                assert self._map_builder is not None
                transition = self._map_builder.process_scan(
                    MapScan(
                        ordinal=plan["ordinal"],
                        object_key=plan["object_key"],
                        points_xyz=np.frombuffer(active_bytes, dtype="<f8").reshape((-1, 3)),
                        reference_from_sensor=np.eye(4, dtype="<f8"),
                        remote_size_bytes=plan["remote_size_bytes"],
                        etag=plan["etag"],
                        last_modified=plan["last_modified"],
                        gt_sha256=self.gt_sha256,
                        calibration_sha256=self.calibration_sha256,
                        object_sha256=payload["local_temporary_sha256"],
                    )
                )
                expected_map_transition = transition[
                    "map_state_transition_sha256"
                ]
            else:
                expected_map_transition = self._authenticated_range_transition_sha256(
                    payload
                )
            if payload["map_state_transition_sha256"] != expected_map_transition:
                raise StreamingTargetMapError(
                    "production replay map-state transition differs"
                )
            completed.append(payload)
            previous_transition = actual_transition
            previous_map_transition = payload["map_state_transition_sha256"]
            previous_end = payload["byte_end_exclusive"]
        return completed

    @property
    def completed_object_count(self) -> int:
        return len(self._records)

    @property
    def completed_keys(self) -> tuple[str, ...]:
        return tuple(row["object_key"] for row in self._records)

    @property
    def pending_objects(self) -> tuple[ReplayMapObject, ...]:
        """Only these objects may be downloaded by a production caller."""

        return self.allowlist[len(self._records) :]

    @property
    def tracks_python_voxel_state(self) -> bool:
        """Whether ingestion also maintains the unbounded reference accumulator."""

        return self.track_python_voxel_state

    @property
    def plan_identity(self) -> dict[str, Any]:
        """Public immutable identity consumed by an external exact-order reducer."""

        return {
            "allowlist_sha256": self._plan_payload["allowlist_sha256"],
            "ledger_path": str(self.ledger_path),
            "numeric_format": self._plan_payload["numeric_format"],
            "padding_rule": self._plan_payload["padding_rule"],
            "plan_sha256": hashlib.sha256(
                canonical_json_bytes(self._plan_payload)
            ).hexdigest(),
            "processing_contract_sha256": self.processing_contract_sha256,
            "replay_path": str(self.replay_path),
            "total_point_capacity": self._plan_payload["total_point_capacity"],
            "total_replay_bytes": self._plan_payload["total_replay_bytes"],
            "track_python_voxel_state": self.track_python_voxel_state,
            "voxel_rule_sha256": self.voxel_rule_sha256,
        }

    @property
    def authenticated_range_records(self) -> tuple[dict[str, Any], ...]:
        """Return the completed-prefix range ledger after path-identity recheck."""

        if _capture_replay_file_identity(self.replay_path) != self._replay_file_identity:
            raise StreamingTargetMapError("production replay file identity changed")
        result: list[dict[str, Any]] = []
        for row in self._records:
            exported = json.loads(json.dumps(row))
            exported["byte_offset"] = exported["byte_start"]
            result.append(exported)
        return tuple(result)

    def append_transformed_scan(
        self,
        replay_object: ReplayMapObject | Mapping[str, Any],
        transformed_xyz: Any,
        *,
        local_temporary_sha256: str,
    ) -> dict[str, Any]:
        """Commit one fixed-order scan range before authenticating it in the ledger."""

        item = ReplayMapObject.from_value(replay_object)
        local_sha = _require_sha256(
            local_temporary_sha256, field="local_temporary_sha256"
        )
        if item.ordinal < len(self._records):
            prior = self._records[item.ordinal]
            if item.identity() != {
                key: prior[key] for key in item.identity()
            }:
                raise StreamingTargetMapError("completed replay object identity changed")
            if prior["local_temporary_sha256"] != local_sha:
                raise StreamingTargetMapError("completed replay payload SHA changed")
            return json.loads(json.dumps(prior))
        if item.ordinal != len(self._records) or item != self.allowlist[item.ordinal]:
            raise StreamingTargetMapError("production replay input is not the next frozen object")
        points = _canonical_float64(transformed_xyz, shape_tail=(3,))
        if points.shape[0] > item.point_capacity:
            raise StreamingTargetMapError(
                "transformed XYZ count exceeds allowlist size/24 capacity"
            )
        active_bytes = points.tobytes(order="C")
        padding_point_count = item.point_capacity - int(points.shape[0])
        replay_bytes = active_bytes + (
            b"\x00" * (padding_point_count * REPLAY_POINT_STRIDE_BYTES)
        )
        plan = self._plan_payload["allowlist"][item.ordinal]
        if len(replay_bytes) != plan["byte_end_exclusive"] - plan["byte_start"]:
            raise StreamingTargetMapError("transformed XYZ byte range length differs")

        ledger_flags = os.O_RDWR | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            ledger_flags |= os.O_NOFOLLOW
        ledger_descriptor = os.open(self.ledger_path, ledger_flags)
        try:
            fcntl.flock(ledger_descriptor, fcntl.LOCK_EX)
            os.lseek(ledger_descriptor, 0, os.SEEK_SET)
            ledger_bytes = b""
            while True:
                block = os.read(ledger_descriptor, 1024 * 1024)
                if not block:
                    break
                ledger_bytes += block
            current = _parse_replay_ledger(ledger_bytes)
            if len(current) - 1 != len(self._records):
                raise StreamingTargetMapError("production replay ledger changed concurrently")

            replay_descriptor = _open_replay_file(self.replay_path, os.O_WRONLY)
            try:
                fcntl.flock(replay_descriptor, fcntl.LOCK_EX)
                if _replay_file_identity(replay_descriptor) != self._replay_file_identity:
                    raise StreamingTargetMapError("production replay file identity changed")
                view = memoryview(replay_bytes)
                offset = plan["byte_start"]
                while view:
                    written = os.pwrite(replay_descriptor, view, offset)
                    if written <= 0:
                        raise OSError("zero-byte production replay write")
                    offset += written
                    view = view[written:]
                os.fsync(replay_descriptor)
                updated_replay_identity = _replay_file_identity(replay_descriptor)
                if not _same_fixed_replay_file(
                    updated_replay_identity, self._replay_file_identity
                ):
                    raise StreamingTargetMapError(
                        "production replay file identity changed during write"
                    )
                # The authenticated write legitimately advances mtime/ctime.
                # Publish the new identity while still holding the exclusive
                # advisory lock so subsequent readers expect exactly it.
                self._replay_file_identity = updated_replay_identity
            finally:
                try:
                    fcntl.flock(replay_descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(replay_descriptor)

            prior_transition = (
                ZERO_SHA256
                if not self._records
                else self._records[-1]["replay_state_transition_sha256"]
            )
            prior_map_transition = (
                ZERO_SHA256
                if not self._records
                else self._records[-1]["map_state_transition_sha256"]
            )
            payload = {
                "byte_end_exclusive": plan["byte_end_exclusive"],
                "byte_start": plan["byte_start"],
                "calibration_sha256": self.calibration_sha256,
                "completed_at_utc": datetime.now(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z"),
                "etag": item.etag,
                "gt_sha256": self.gt_sha256,
                "last_modified": item.last_modified,
                "local_temporary_sha256": local_sha,
                "map_state_transition_kind": (
                    PYTHON_VOXEL_TRANSITION_KIND
                    if self.track_python_voxel_state
                    else AUTHENTICATED_RANGE_TRANSITION_KIND
                ),
                "object_key": item.object_key,
                "ordinal": item.ordinal,
                "padding_point_count": padding_point_count,
                "padding_rule": PRODUCTION_REPLAY_PADDING_RULE,
                "point_count": int(points.shape[0]),
                "previous_map_state_transition_sha256": prior_map_transition,
                "previous_replay_state_transition_sha256": prior_transition,
                "processing_contract_sha256": self.processing_contract_sha256,
                "remote_size_bytes": item.remote_size_bytes,
                "replay_range_sha256": hashlib.sha256(replay_bytes).hexdigest(),
                "transformed_xyz_sha256": hashlib.sha256(active_bytes).hexdigest(),
            }
            if self.track_python_voxel_state:
                assert self._map_builder is not None
                map_transition = self._map_builder.process_scan(
                    MapScan(
                        ordinal=item.ordinal,
                        object_key=item.object_key,
                        points_xyz=points,
                        reference_from_sensor=np.eye(4, dtype="<f8"),
                        remote_size_bytes=item.remote_size_bytes,
                        etag=item.etag,
                        last_modified=item.last_modified,
                        gt_sha256=self.gt_sha256,
                        calibration_sha256=self.calibration_sha256,
                        object_sha256=local_sha,
                    )
                )
                payload["map_state_transition_sha256"] = map_transition[
                    "map_state_transition_sha256"
                ]
            else:
                payload["map_state_transition_sha256"] = (
                    self._authenticated_range_transition_sha256(payload)
                )
            payload["replay_state_transition_sha256"] = hashlib.sha256(
                canonical_json_bytes(payload)
            ).hexdigest()
            envelope = _ledger_envelope(
                sequence_number=len(current) + 1,
                previous_record_sha256=current[-1]["record_sha256"],
                record_type="SCAN",
                payload=payload,
            )
            view = memoryview(_compact_json_line(envelope))
            while view:
                written = os.write(ledger_descriptor, view)
                if written <= 0:
                    raise OSError("zero-byte production replay ledger append")
                view = view[written:]
            os.fsync(ledger_descriptor)
        finally:
            try:
                fcntl.flock(ledger_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(ledger_descriptor)
        _fsync_directory(self.ledger_path.parent)
        self._records.append(payload)
        return json.loads(json.dumps(payload))

    def append_scan(self, scan: MapScan) -> dict[str, Any]:
        """Transform and commit a newly materialized scan; completed scans need not call this."""

        validated = scan.validated()
        if validated.object_sha256 is None:
            raise StreamingTargetMapError("production replay requires the downloaded payload SHA")
        item = ReplayMapObject(
            ordinal=validated.ordinal,
            object_key=validated.object_key,
            remote_size_bytes=validated.remote_size_bytes,
            etag=validated.etag,
            last_modified=validated.last_modified,
        )
        transformed = _transform_points(
            validated.points_xyz, validated.reference_from_sensor
        )
        return self.append_transformed_scan(
            item,
            transformed,
            local_temporary_sha256=validated.object_sha256,
        )

    def _read_authenticated_payload(
        self, row: Mapping[str, Any], *, descriptor: int | None = None
    ) -> bytes:
        start = int(row["byte_start"])
        length = int(row["byte_end_exclusive"]) - start
        if descriptor is None:
            payload = _read_exact_range(
                self.replay_path,
                start,
                length,
                expected_identity=self._replay_file_identity,
            )
        else:
            payload = _read_exact_range_from_descriptor(
                descriptor,
                start,
                length,
                expected_identity=self._replay_file_identity,
            )
        if hashlib.sha256(payload).hexdigest() != row["replay_range_sha256"]:
            raise StreamingTargetMapError(
                "authenticated production replay byte range differs at use time"
            )
        return payload

    def read_transformed_scan(self, ordinal: int) -> np.ndarray:
        if ordinal < 0 or ordinal >= len(self._records):
            raise StreamingTargetMapError("requested replay scan is not authenticated complete")
        row = self._records[ordinal]
        payload = self._read_authenticated_payload(row)
        plan = self._plan_payload["allowlist"][ordinal]
        active = self._active_range_bytes(payload, row, plan)
        return np.frombuffer(active, dtype="<f8").reshape((-1, 3)).copy()

    def build_target_map(
        self, voxel_rule: VoxelRule, *, worker_count: int = 1
    ) -> TargetMapResult:
        """Replay one scan at a time into the deterministic accumulator.

        The content-addressed target NPY is byte-exact with direct processing.
        The accumulator's internal ``final_state_sha256`` intentionally binds
        the replay representation (pretransformed XYZ plus identity transform)
        and can therefore differ from the direct-download state hash.
        """

        if len(self._records) != len(self.allowlist):
            raise StreamingTargetMapError("cannot finalize an incomplete production replay prefix")
        if not self.track_python_voxel_state:
            raise StreamingTargetMapError(
                "production replay disabled Python voxel state; use the authenticated "
                "range plan with an independently qualified external reducer"
            )
        if voxel_rule.contract_sha256 != self.voxel_rule_sha256:
            raise StreamingTargetMapError("voxel rule differs from authenticated replay plan")
        builder = StreamingTargetMapBuilder(voxel_rule, worker_count=worker_count)
        replay_descriptor = _open_replay_file(self.replay_path, os.O_RDONLY)
        try:
            # Hold a shared lock for the whole replay.  Cooperative writers use
            # LOCK_EX, so a build cannot observe a half-written scan prefix.
            fcntl.flock(replay_descriptor, fcntl.LOCK_SH)
            if _replay_file_identity(replay_descriptor) != self._replay_file_identity:
                raise StreamingTargetMapError("production replay file identity changed")
            for item, record in zip(self.allowlist, self._records):
                # Only one authenticated range is resident at once; builder
                # retains sufficient statistics, never the full point set.
                payload = self._read_authenticated_payload(
                    record, descriptor=replay_descriptor
                )
                active = self._active_range_bytes(
                    payload,
                    record,
                    self._plan_payload["allowlist"][item.ordinal],
                )
                points = np.frombuffer(active, dtype="<f8").reshape((-1, 3)).copy()
                scan = MapScan(
                    ordinal=item.ordinal,
                    object_key=item.object_key,
                    points_xyz=points,
                    reference_from_sensor=np.eye(4, dtype="<f8"),
                    remote_size_bytes=item.remote_size_bytes,
                    etag=item.etag,
                    last_modified=item.last_modified,
                    gt_sha256=self.gt_sha256,
                    calibration_sha256=self.calibration_sha256,
                    object_sha256=record["local_temporary_sha256"],
                )
                builder.process_scan(scan)
                del scan, points, active, payload
            if _capture_replay_file_identity(self.replay_path) != self._replay_file_identity:
                raise StreamingTargetMapError(
                    "production replay path identity changed during build"
                )
        finally:
            try:
                fcntl.flock(replay_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(replay_descriptor)
        return builder.finalize()


def build_target_map_incremental(
    scans: Sequence[MapScan], voxel_rule: VoxelRule, *, worker_count: int = 1
) -> TargetMapResult:
    builder = StreamingTargetMapBuilder(voxel_rule, worker_count=worker_count)
    builder.process_scans(scans)
    return builder.finalize()


def build_target_map_batch(
    scans: Sequence[MapScan], voxel_rule: VoxelRule, *, worker_count: int = 1
) -> TargetMapResult:
    """Reference batch facade using the exact same mandated reduction order."""

    validated = [scan.validated() for scan in scans]
    if [scan.ordinal for scan in validated] != list(range(len(validated))):
        raise StreamingTargetMapError("batch input order must be explicit ordinals 0..N-1")
    return build_target_map_incremental(validated, voxel_rule, worker_count=worker_count)


def assert_target_maps_exact(left: TargetMapResult, right: TargetMapResult) -> None:
    """Fail unless two builds are byte-exact, not merely numerically close."""

    if left.target_map_sha256 != right.target_map_sha256:
        raise StreamingTargetMapError("target-map SHA mismatch")
    if left.voxel_keys.dtype != right.voxel_keys.dtype or not np.array_equal(
        left.voxel_keys, right.voxel_keys
    ):
        raise StreamingTargetMapError("target-map voxel ordering differs")
    if left.points_xyz.dtype != right.points_xyz.dtype or not np.array_equal(
        left.points_xyz, right.points_xyz
    ):
        raise StreamingTargetMapError("target-map points are not byte-exact")
