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
import re
from dataclasses import dataclass
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
    digest.update(value.tobytes(order="C"))
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
    origin_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

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
        previous_transition: str | None = None
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
                previous_transition = transition_sha
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
