"""Frozen Boreas v2 Stage-2 LiDAR preprocessing primitives.

This module performs decoding and reference-based preprocessing only.  It has
no network client, selector, geometry-ranking implementation, or registration
backend.  The scientific constants mirror the prospectively frozen contract at
``protocols/boreas_v2_stage2_preprocessing_contract.json``.

The motion-correction rule is a deterministic, serial reproduction of the
official pyboreas ``PointCloud.remove_motion`` sorted-timestamp path.  In
particular, it uses 21 uniformly spaced time bins and the left endpoint of each
of the 20 intervals.  A scan whose per-point timestamps are not nondecreasing
is rejected instead of entering pyboreas' multiprocessing fallback.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .io import sha256_file
from .streaming_target_map import CENTROID_RULE, VoxelRule


PRIMARY_MAP_SEQUENCE_ID = "boreas-2021-11-14-09-47"
PRIMARY_QUERY_SEQUENCE_ID = "boreas-2021-01-26-11-22"
ALLOWED_PRIMARY_SEQUENCES = frozenset(
    {PRIMARY_MAP_SEQUENCE_ID, PRIMARY_QUERY_SEQUENCE_ID}
)

RAW_POINT_DTYPE = np.dtype("<f4")
RAW_POINT_FIELD_COUNT = 6
RAW_POINT_STRIDE_BYTES = RAW_POINT_DTYPE.itemsize * RAW_POINT_FIELD_COUNT
RAW_POINT_FIELDS = (
    "x_m",
    "y_m",
    "z_m",
    "intensity",
    "laser_id",
    "time_offset_from_scan_middle_s",
)
CANONICAL_POINT_DTYPE = np.dtype("<f8")

MINIMUM_RANGE_M = 1.0
MAXIMUM_RANGE_M = 80.0
SOURCE_VOXEL_SIZE_M = 0.10
TARGET_VOXEL_SIZE_M = 0.10
SOURCE_VOXEL_ORIGIN_M = (0.0, 0.0, 0.0)
TARGET_VOXEL_ORIGIN_ENU_REF_M = (0.0, 0.0, 0.0)
MOTION_COMPENSATION_BIN_COUNT = 21
MAXIMUM_NATIVE_REFERENCE_GAP_S = 0.20

POSE_HEADER = (
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
    "angvel_z",
    "angvel_y",
    "angvel_x",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OBJECT_KEY_RE = re.compile(
    r"^(boreas-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2})/"
    r"lidar/([0-9]{16})\.bin$"
)
TIMESTAMP_RE = re.compile(r"^[0-9]{16}$")


class BoreasStage2PreprocessingError(RuntimeError):
    """Raised when frozen Boreas preprocessing cannot be followed exactly."""


def _require_sha256(value: str, *, field: str) -> str:
    text = str(value).lower()
    if SHA256_RE.fullmatch(text) is None:
        raise BoreasStage2PreprocessingError(f"{field} must be a lowercase SHA-256")
    return text


def boreas_microseconds_to_float_seconds(timestamp_us: int) -> float:
    """Reproduce pinned pyboreas ``micro_to_sec`` exactly.

    The explicit six-decimal rounding is scientifically relevant at Unix-epoch
    magnitudes: ``float(timestamp_us) * 1e-6`` can differ by one binary64 ULP,
    which can move a point across a left-endpoint motion-correction bin.
    """

    return float(np.round(int(timestamp_us) / 1.0e6, 6))


def _canonical_points(value: Any, *, allow_empty: bool = False) -> np.ndarray:
    points = np.asarray(value, dtype=CANONICAL_POINT_DTYPE)
    if points.ndim != 2 or points.shape[1] != 3:
        raise BoreasStage2PreprocessingError("points must have shape (N, 3)")
    if not allow_empty and points.shape[0] == 0:
        raise BoreasStage2PreprocessingError("preprocessing produced no points")
    if not np.all(np.isfinite(points)):
        raise BoreasStage2PreprocessingError("canonical points must be finite")
    result = np.array(points, dtype=CANONICAL_POINT_DTYPE, order="C", copy=True)
    result.setflags(write=False)
    return result


def _canonical_transform(value: Any) -> np.ndarray:
    transform = np.asarray(value, dtype=CANONICAL_POINT_DTYPE)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise BoreasStage2PreprocessingError("T_reference must be finite 4x4")
    if not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12, rtol=0.0
    ):
        raise BoreasStage2PreprocessingError("T_reference homogeneous row is invalid")
    rotation = transform[:3, :3]
    if not np.allclose(
        rotation @ rotation.T, np.eye(3), atol=1.0e-10, rtol=0.0
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10, rtol=0.0):
        raise BoreasStage2PreprocessingError("T_reference rotation is not proper SO(3)")
    result = np.array(transform, dtype=CANONICAL_POINT_DTYPE, order="C", copy=True)
    result.setflags(write=False)
    return result


def parse_primary_lidar_object_key(object_key: str) -> tuple[str, int]:
    """Return ``(sequence_id, timestamp_us)`` for one frozen-primary key."""

    match = OBJECT_KEY_RE.fullmatch(str(object_key))
    if match is None:
        raise BoreasStage2PreprocessingError("object key is not a Boreas lidar/*.bin key")
    sequence_id, timestamp_text = match.groups()
    if sequence_id not in ALLOWED_PRIMARY_SEQUENCES:
        raise BoreasStage2PreprocessingError("object key is outside the frozen PRIMARY_PAIR")
    return sequence_id, int(timestamp_text)


@dataclass(frozen=True)
class DecodedBoreasScan:
    """One authenticated raw scan decoded without changing raw point order."""

    object_key: str
    sequence_id: str
    timestamp_us: int
    fields: np.ndarray

    def __post_init__(self) -> None:
        sequence_id, timestamp_us = parse_primary_lidar_object_key(self.object_key)
        if self.sequence_id != sequence_id or int(self.timestamp_us) != timestamp_us:
            raise BoreasStage2PreprocessingError("decoded scan identity disagrees with object key")
        fields = np.asarray(self.fields, dtype=CANONICAL_POINT_DTYPE)
        if fields.ndim != 2 or fields.shape[1] != RAW_POINT_FIELD_COUNT:
            raise BoreasStage2PreprocessingError("decoded Boreas scan must have shape (N, 6)")
        if fields.shape[0] == 0:
            raise BoreasStage2PreprocessingError("decoded Boreas scan is empty")
        canonical = np.array(fields, dtype=CANONICAL_POINT_DTYPE, order="C", copy=True)
        canonical.setflags(write=False)
        object.__setattr__(self, "timestamp_us", timestamp_us)
        object.__setattr__(self, "fields", canonical)

    @property
    def point_count(self) -> int:
        return int(self.fields.shape[0])

    @property
    def point_timestamps_s(self) -> np.ndarray:
        scan_middle_s = boreas_microseconds_to_float_seconds(self.timestamp_us)
        result = np.asarray(
            scan_middle_s + self.fields[:, 5], dtype=CANONICAL_POINT_DTYPE
        )
        result.setflags(write=False)
        return result


def decode_boreas_velodyne_payload(
    payload: bytes | bytearray | memoryview,
    *,
    object_key: str,
) -> DecodedBoreasScan:
    """Decode an in-memory Boreas Velodyne object using its official wire format."""

    sequence_id, timestamp_us = parse_primary_lidar_object_key(object_key)
    view = memoryview(payload)
    if len(view) == 0 or len(view) % RAW_POINT_STRIDE_BYTES:
        raise BoreasStage2PreprocessingError(
            "Boreas lidar payload is empty or not divisible by the 24-byte point stride"
        )
    raw = np.frombuffer(view, dtype=RAW_POINT_DTYPE)
    fields = raw.reshape((-1, RAW_POINT_FIELD_COUNT)).astype(
        CANONICAL_POINT_DTYPE, copy=True
    )
    return DecodedBoreasScan(
        object_key=str(object_key),
        sequence_id=sequence_id,
        timestamp_us=timestamp_us,
        fields=fields,
    )


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BoreasStage2PreprocessingError(f"cannot safely open lidar payload: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BoreasStage2PreprocessingError("lidar payload is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        if len(payload) != int(metadata.st_size):
            raise BoreasStage2PreprocessingError("lidar payload changed while it was read")
        return payload
    finally:
        os.close(descriptor)


def decode_authenticated_boreas_velodyne_file(
    path: str | Path,
    *,
    object_key: str,
    expected_size_bytes: int,
    expected_sha256: str,
) -> DecodedBoreasScan:
    """Decode one already-downloaded object only after size and SHA authentication."""

    expected_digest = _require_sha256(expected_sha256, field="expected_sha256")
    payload = _read_regular_file(Path(path))
    if len(payload) != int(expected_size_bytes):
        raise BoreasStage2PreprocessingError("downloaded lidar size differs from receipt")
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise BoreasStage2PreprocessingError("downloaded lidar SHA-256 differs from receipt")
    return decode_boreas_velodyne_payload(payload, object_key=object_key)


def _roll(value: float) -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(value), math.sin(value)],
            [0.0, -math.sin(value), math.cos(value)],
        ],
        dtype=np.float64,
    )


def _pitch(value: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(value), 0.0, -math.sin(value)],
            [0.0, 1.0, 0.0],
            [math.sin(value), 0.0, math.cos(value)],
        ],
        dtype=np.float64,
    )


def _yaw(value: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(value), math.sin(value), 0.0],
            [-math.sin(value), math.cos(value), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _pose_transform(row: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _roll(float(row[7])) @ _pitch(float(row[8])) @ _yaw(
        float(row[9])
    )
    transform[:3, 3] = np.asarray(row[1:4], dtype=np.float64)
    return _canonical_transform(transform)


@dataclass(frozen=True)
class BoreasLidarPose:
    """Official exact-timestamp lidar pose and Applanix-derived body twist."""

    timestamp_us: int
    t_enu_lidar: np.ndarray
    body_rate_lidar: np.ndarray

    def __post_init__(self) -> None:
        if int(self.timestamp_us) < 0:
            raise BoreasStage2PreprocessingError("pose timestamp must be nonnegative")
        transform = _canonical_transform(self.t_enu_lidar)
        body_rate = np.asarray(self.body_rate_lidar, dtype=CANONICAL_POINT_DTYPE)
        if body_rate.shape not in {(6,), (6, 1)} or not np.all(np.isfinite(body_rate)):
            raise BoreasStage2PreprocessingError("body_rate_lidar must be finite length 6")
        canonical_rate = np.array(
            body_rate.reshape(6), dtype=CANONICAL_POINT_DTYPE, order="C", copy=True
        )
        canonical_rate.setflags(write=False)
        object.__setattr__(self, "timestamp_us", int(self.timestamp_us))
        object.__setattr__(self, "t_enu_lidar", transform)
        object.__setattr__(self, "body_rate_lidar", canonical_rate)


@dataclass(frozen=True)
class BoreasLidarPoseIndex:
    """Immutable, SHA-bound exact-timestamp index of official lidar poses."""

    sequence_id: str
    source_sha256: str
    poses: Mapping[int, BoreasLidarPose]
    maximum_native_gap_s: float

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        sequence_id: str,
        expected_sha256: str,
    ) -> "BoreasLidarPoseIndex":
        if sequence_id not in ALLOWED_PRIMARY_SEQUENCES:
            raise BoreasStage2PreprocessingError("pose sequence is outside PRIMARY_PAIR")
        expected_digest = _require_sha256(expected_sha256, field="pose expected_sha256")
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise BoreasStage2PreprocessingError("pose CSV must be a regular non-symlink file")
        if sha256_file(source) != expected_digest:
            raise BoreasStage2PreprocessingError("pose CSV SHA-256 differs from frozen evidence")

        poses: dict[int, BoreasLidarPose] = {}
        timestamps: list[int] = []
        with source.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = tuple(next(reader, ()))
            if header != POSE_HEADER:
                raise BoreasStage2PreprocessingError("unexpected Boreas lidar_poses.csv header")
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(POSE_HEADER) or TIMESTAMP_RE.fullmatch(row[0]) is None:
                    raise BoreasStage2PreprocessingError(
                        f"invalid Boreas pose row at line {line_number}"
                    )
                try:
                    timestamp_us = int(row[0])
                    values = np.asarray([float(value) for value in row], dtype=np.float64)
                except ValueError as exc:
                    raise BoreasStage2PreprocessingError(
                        f"nonnumeric Boreas pose row at line {line_number}"
                    ) from exc
                if not np.all(np.isfinite(values)):
                    raise BoreasStage2PreprocessingError(
                        f"nonfinite Boreas pose row at line {line_number}"
                    )
                if timestamps and timestamp_us <= timestamps[-1]:
                    raise BoreasStage2PreprocessingError(
                        "pose timestamps must be unique and strictly increasing"
                    )
                transform = _pose_transform(values)
                linear_velocity_enu = values[4:7]
                linear_velocity_lidar = transform[:3, :3].T @ linear_velocity_enu
                body_rate = np.concatenate(
                    (linear_velocity_lidar, values[[12, 11, 10]])
                )
                poses[timestamp_us] = BoreasLidarPose(
                    timestamp_us=timestamp_us,
                    t_enu_lidar=transform,
                    body_rate_lidar=body_rate,
                )
                timestamps.append(timestamp_us)
        if len(timestamps) < 2:
            raise BoreasStage2PreprocessingError("pose CSV has fewer than two rows")
        maximum_gap_s = max(
            (right - left) * 1.0e-6
            for left, right in zip(timestamps, timestamps[1:])
        )
        if maximum_gap_s > MAXIMUM_NATIVE_REFERENCE_GAP_S:
            raise BoreasStage2PreprocessingError(
                "pose CSV exceeds frozen 0.20 s native-reference gap"
            )
        return cls(
            sequence_id=sequence_id,
            source_sha256=expected_digest,
            poses=MappingProxyType(poses),
            maximum_native_gap_s=float(maximum_gap_s),
        )

    def exact_pose(self, timestamp_us: int) -> BoreasLidarPose:
        try:
            return self.poses[int(timestamp_us)]
        except KeyError as exc:
            raise BoreasStage2PreprocessingError(
                "no exact lidar pose row exists; interpolation and nearest-pose fallback are forbidden"
            ) from exc


def load_t_applanix_lidar(
    path: str | Path, *, expected_sha256: str
) -> np.ndarray:
    """Load and authenticate the frozen static lidar-to-Applanix transform."""

    expected_digest = _require_sha256(expected_sha256, field="extrinsic expected_sha256")
    source = Path(path)
    if source.is_symlink() or not source.is_file() or sha256_file(source) != expected_digest:
        raise BoreasStage2PreprocessingError("extrinsic file is unsafe or differs from frozen SHA")
    try:
        value = np.loadtxt(source, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise BoreasStage2PreprocessingError("cannot parse T_applanix_lidar") from exc
    return _canonical_transform(value)


def compose_t_enu_lidar(
    t_enu_applanix: np.ndarray, t_applanix_lidar: np.ndarray
) -> np.ndarray:
    """Compose the frozen direction ``T_ENU_applanix @ T_applanix_lidar``."""

    left = _canonical_transform(t_enu_applanix)
    right = _canonical_transform(t_applanix_lidar)
    return _canonical_transform(left @ right)


def _carrot(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )


def _official_se3_exp(twist: np.ndarray) -> np.ndarray:
    """Reproduce pyboreas ``lgmath._vec2tran`` for one length-6 twist."""

    value = np.asarray(twist, dtype=np.float64).reshape(6)
    translation = value[:3].reshape(3, 1)
    axis_angle = value[3:].reshape(3, 1)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1.0e-12:
        rotation = np.eye(3, dtype=np.float64)
        jacobian = np.eye(3, dtype=np.float64)
    else:
        axis = axis_angle / angle
        skew = _carrot(axis)
        sine_over_angle = math.sin(angle) / angle
        one_minus_cosine_over_angle = (1.0 - math.cos(angle)) / angle
        rotation = (
            math.cos(angle) * np.eye(3)
            + (1.0 - math.cos(angle)) * (axis @ axis.T)
            + math.sin(angle) * skew
        )
        jacobian = (
            sine_over_angle * np.eye(3)
            + (1.0 - sine_over_angle) * (axis @ axis.T)
            + one_minus_cosine_over_angle * skew
        )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3:] = jacobian @ translation
    return result


def deskew_official_21_bin_constant_body_twist(
    points_xyz: np.ndarray,
    point_timestamps_s: np.ndarray,
    *,
    scan_reference_timestamp_s: float,
    body_rate_lidar: np.ndarray,
) -> np.ndarray:
    """Deskew to the scan midpoint using the frozen official 21-bin rule."""

    points = _canonical_points(points_xyz)
    timestamps = np.asarray(point_timestamps_s, dtype=np.float64)
    body_rate = np.asarray(body_rate_lidar, dtype=np.float64).reshape(-1)
    reference_time = float(scan_reference_timestamp_s)
    if timestamps.shape != (points.shape[0],) or not np.all(np.isfinite(timestamps)):
        raise BoreasStage2PreprocessingError("point timestamps must be finite length N")
    if body_rate.shape != (6,) or not np.all(np.isfinite(body_rate)):
        raise BoreasStage2PreprocessingError("body_rate_lidar must be finite length 6")
    if not math.isfinite(reference_time):
        raise BoreasStage2PreprocessingError("scan reference timestamp must be finite")
    if timestamps.size > 1 and np.any(np.diff(timestamps) < 0.0):
        raise BoreasStage2PreprocessingError(
            "point timestamps are not nondecreasing; official multiprocessing fallback is forbidden"
        )
    minimum_time = float(timestamps[0])
    maximum_time = float(timestamps[-1])
    span = maximum_time - minimum_time
    if not math.isfinite(span) or span <= 0.0:
        raise BoreasStage2PreprocessingError("point timestamp span is degenerate")
    delta = span / float(MOTION_COMPENSATION_BIN_COUNT - 1)
    bin_times = np.asarray(
        [minimum_time + index * delta for index in range(MOTION_COMPENSATION_BIN_COUNT)],
        dtype=np.float64,
    )
    boundaries = bin_times.copy()
    boundaries[-1] += 1.0e-6
    transforms = [
        _official_se3_exp((time_value - reference_time) * body_rate)
        for time_value in bin_times
    ]
    output = np.array(points, dtype=np.float64, order="C", copy=True)
    assigned = np.zeros(points.shape[0], dtype=bool)
    for index in range(MOTION_COMPENSATION_BIN_COUNT - 1):
        selected = (timestamps >= boundaries[index]) & (
            timestamps < boundaries[index + 1]
        )
        if not np.any(selected):
            continue
        transform = transforms[index]
        output[selected] = (
            points[selected] @ transform[:3, :3].T + transform[:3, 3]
        )
        assigned[selected] = True
    if not np.all(assigned):
        raise BoreasStage2PreprocessingError(
            "official 21-bin rule did not assign every point timestamp"
        )
    return _canonical_points(output)


def deterministic_voxel_centroids(
    points_xyz: np.ndarray,
    *,
    voxel_size_m: float,
    origin_xyz_m: Sequence[float],
) -> np.ndarray:
    """Fixed-order float64 centroid voxelization with lexicographic output."""

    points = _canonical_points(points_xyz)
    size = float(voxel_size_m)
    origin = np.asarray(origin_xyz_m, dtype=np.float64)
    if not math.isfinite(size) or size <= 0.0 or origin.shape != (3,) or not np.all(
        np.isfinite(origin)
    ):
        raise BoreasStage2PreprocessingError("voxel rule is invalid")
    scaled = np.floor((points - origin) / size)
    int64 = np.iinfo(np.int64)
    if np.any(scaled < int64.min) or np.any(scaled > int64.max):
        raise BoreasStage2PreprocessingError("voxel key exceeds int64 range")
    keys = scaled.astype(np.int64)
    # Include the raw ordinal as the least-significant sorting key.  This makes
    # equal-voxel runs retain encounter order without relying on a sort's
    # stability contract.  Most 0.10 m lidar voxels are singletons, so doing
    # this in arrays also avoids one Python dict/list/ndarray allocation per
    # point.  Collision groups still use explicit serial additions below:
    # np.add.reduce(at) is deliberately not used because it may reassociate
    # non-associative floating-point sums.
    ordinals = np.arange(points.shape[0], dtype=np.int64)
    order = np.lexsort((ordinals, keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered_keys = keys[order]
    ordered_points = points[order]
    is_first = np.empty(points.shape[0], dtype=bool)
    is_first[0] = True
    is_first[1:] = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
    starts = np.flatnonzero(is_first)
    ends = np.concatenate((starts[1:], np.asarray([points.shape[0]], dtype=np.int64)))
    counts = ends - starts

    # For singleton groups this vectorized addition exactly implements the
    # frozen accumulator initialization ``sum=[0,0,0]; sum += point`` (including
    # signed-zero behavior).  Only true collision groups need a Python loop.
    output = np.zeros((starts.shape[0], 3), dtype=CANONICAL_POINT_DTYPE)
    output += ordered_points[starts]
    for group_index in np.flatnonzero(counts > 1):
        sums = np.zeros(3, dtype=CANONICAL_POINT_DTYPE)
        for point in ordered_points[starts[group_index] : ends[group_index]]:
            sums += point
        output[group_index] = sums / np.float64(counts[group_index])
    return _canonical_points(output)


@dataclass(frozen=True)
class PreprocessedBoreasScan:
    """Canonical Stage-2 scan before target-map accumulation or screening."""

    role: str
    object_key: str
    sequence_id: str
    timestamp_us: int
    points_xyz: np.ndarray
    t_reference: np.ndarray
    raw_point_count: int
    nonfinite_excluded_count: int
    range_excluded_count: int
    post_filter_point_count: int
    source_voxel_reduced_count: int

    def __post_init__(self) -> None:
        if self.role not in {"TARGET_MAP", "QUERY"}:
            raise BoreasStage2PreprocessingError("scan role must be TARGET_MAP or QUERY")
        expected_sequence = (
            PRIMARY_MAP_SEQUENCE_ID if self.role == "TARGET_MAP" else PRIMARY_QUERY_SEQUENCE_ID
        )
        if self.sequence_id != expected_sequence:
            raise BoreasStage2PreprocessingError("scan sequence disagrees with frozen role")
        points = _canonical_points(self.points_xyz)
        transform = _canonical_transform(self.t_reference)
        counts = [
            int(self.raw_point_count),
            int(self.nonfinite_excluded_count),
            int(self.range_excluded_count),
            int(self.post_filter_point_count),
            int(self.source_voxel_reduced_count),
        ]
        if any(value < 0 for value in counts):
            raise BoreasStage2PreprocessingError("preprocessing counts cannot be negative")
        if counts[0] != counts[1] + counts[2] + counts[3]:
            raise BoreasStage2PreprocessingError("filter counts do not close")
        expected_reduced = counts[3] - points.shape[0] if self.role == "QUERY" else 0
        if counts[4] != expected_reduced:
            raise BoreasStage2PreprocessingError("source voxel reduction count does not close")
        object.__setattr__(self, "points_xyz", points)
        object.__setattr__(self, "t_reference", transform)


def preprocess_primary_boreas_scan(
    decoded: DecodedBoreasScan,
    *,
    pose_index: BoreasLidarPoseIndex,
    role: str,
) -> PreprocessedBoreasScan:
    """Apply the single frozen map/query preprocessing rule to one scan."""

    expected_sequence = (
        PRIMARY_MAP_SEQUENCE_ID if role == "TARGET_MAP" else PRIMARY_QUERY_SEQUENCE_ID
        if role == "QUERY"
        else None
    )
    if expected_sequence is None:
        raise BoreasStage2PreprocessingError("scan role must be TARGET_MAP or QUERY")
    if decoded.sequence_id != expected_sequence or pose_index.sequence_id != expected_sequence:
        raise BoreasStage2PreprocessingError("scan/pose sequence disagrees with frozen role")
    pose = pose_index.exact_pose(decoded.timestamp_us)
    fields = decoded.fields
    finite = np.all(np.isfinite(fields[:, :3]), axis=1) & np.isfinite(fields[:, 5])
    finite_points = fields[finite, :3]
    finite_offsets = fields[finite, 5]
    ranges = np.linalg.norm(finite_points, axis=1)
    within_range = (ranges >= MINIMUM_RANGE_M) & (ranges <= MAXIMUM_RANGE_M)
    filtered_points = finite_points[within_range]
    filtered_offsets = finite_offsets[within_range]
    if filtered_points.shape[0] == 0:
        raise BoreasStage2PreprocessingError("finite inclusive range filtering removed all points")
    scan_reference_timestamp_s = boreas_microseconds_to_float_seconds(
        decoded.timestamp_us
    )
    point_times = scan_reference_timestamp_s + filtered_offsets.astype(np.float64)
    deskewed = deskew_official_21_bin_constant_body_twist(
        filtered_points,
        point_times,
        scan_reference_timestamp_s=scan_reference_timestamp_s,
        body_rate_lidar=pose.body_rate_lidar,
    )
    if role == "QUERY":
        canonical_points = deterministic_voxel_centroids(
            deskewed,
            voxel_size_m=SOURCE_VOXEL_SIZE_M,
            origin_xyz_m=SOURCE_VOXEL_ORIGIN_M,
        )
    else:
        canonical_points = deskewed
    raw_count = decoded.point_count
    nonfinite_count = int(raw_count - np.count_nonzero(finite))
    range_count = int(finite_points.shape[0] - np.count_nonzero(within_range))
    post_filter_count = int(filtered_points.shape[0])
    return PreprocessedBoreasScan(
        role=role,
        object_key=decoded.object_key,
        sequence_id=decoded.sequence_id,
        timestamp_us=decoded.timestamp_us,
        points_xyz=canonical_points,
        t_reference=pose.t_enu_lidar,
        raw_point_count=raw_count,
        nonfinite_excluded_count=nonfinite_count,
        range_excluded_count=range_count,
        post_filter_point_count=post_filter_count,
        source_voxel_reduced_count=(
            post_filter_count - canonical_points.shape[0] if role == "QUERY" else 0
        ),
    )


def transform_preprocessed_map_scan_to_enu_ref(
    scan: PreprocessedBoreasScan,
) -> np.ndarray:
    """Transform a map-role scan to fixed ENU_ref without voxelizing it twice."""

    if scan.role != "TARGET_MAP" or scan.sequence_id != PRIMARY_MAP_SEQUENCE_ID:
        raise BoreasStage2PreprocessingError("only PRIMARY map scans may enter target map")
    transform = scan.t_reference
    result = scan.points_xyz @ transform[:3, :3].T + transform[:3, 3]
    return _canonical_points(result)


def target_map_voxel_rule(*, preprocessing_contract_sha256: str) -> VoxelRule:
    """Construct the global ENU_ref target rule bound to the frozen contract."""

    return VoxelRule(
        voxel_size_m=TARGET_VOXEL_SIZE_M,
        representative_rule=CENTROID_RULE,
        parameter_authority="STAGE2_PREREGISTRATION",
        scientific_contract_sha256=_require_sha256(
            preprocessing_contract_sha256, field="preprocessing_contract_sha256"
        ),
        origin_xyz_m=TARGET_VOXEL_ORIGIN_ENU_REF_M,
    )


def canonical_npy_bytes(value: np.ndarray, *, expected_shape_tail: tuple[int, ...]) -> bytes:
    """Return deterministic little-endian float64 NPY v1.0 bytes."""

    array = np.asarray(value, dtype=CANONICAL_POINT_DTYPE)
    if array.ndim != len(expected_shape_tail) + 1 or array.shape[1:] != expected_shape_tail:
        raise BoreasStage2PreprocessingError("canonical array shape differs from contract")
    if not np.all(np.isfinite(array)):
        raise BoreasStage2PreprocessingError("canonical array is nonfinite")
    canonical = np.ascontiguousarray(array, dtype=CANONICAL_POINT_DTYPE)
    stream = io.BytesIO()
    np.lib.format.write_array(stream, canonical, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def canonical_source_npy_bytes(scan: PreprocessedBoreasScan) -> bytes:
    if scan.role != "QUERY":
        raise BoreasStage2PreprocessingError("canonical source must be a QUERY scan")
    return canonical_npy_bytes(scan.points_xyz, expected_shape_tail=(3,))


def canonical_t_reference_npy_bytes(scan: PreprocessedBoreasScan) -> bytes:
    array = np.asarray(scan.t_reference, dtype=CANONICAL_POINT_DTYPE)
    if array.shape != (4, 4):
        raise BoreasStage2PreprocessingError("T_reference shape differs from contract")
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(array, dtype=CANONICAL_POINT_DTYPE),
        version=(1, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


__all__ = [
    "ALLOWED_PRIMARY_SEQUENCES",
    "BoreasLidarPose",
    "BoreasLidarPoseIndex",
    "BoreasStage2PreprocessingError",
    "CANONICAL_POINT_DTYPE",
    "DecodedBoreasScan",
    "MAXIMUM_NATIVE_REFERENCE_GAP_S",
    "MAXIMUM_RANGE_M",
    "MINIMUM_RANGE_M",
    "MOTION_COMPENSATION_BIN_COUNT",
    "PRIMARY_MAP_SEQUENCE_ID",
    "PRIMARY_QUERY_SEQUENCE_ID",
    "PreprocessedBoreasScan",
    "RAW_POINT_FIELDS",
    "RAW_POINT_STRIDE_BYTES",
    "SOURCE_VOXEL_ORIGIN_M",
    "SOURCE_VOXEL_SIZE_M",
    "TARGET_VOXEL_ORIGIN_ENU_REF_M",
    "TARGET_VOXEL_SIZE_M",
    "boreas_microseconds_to_float_seconds",
    "canonical_npy_bytes",
    "canonical_source_npy_bytes",
    "canonical_t_reference_npy_bytes",
    "compose_t_enu_lidar",
    "decode_authenticated_boreas_velodyne_file",
    "decode_boreas_velodyne_payload",
    "deskew_official_21_bin_constant_body_twist",
    "deterministic_voxel_centroids",
    "load_t_applanix_lidar",
    "parse_primary_lidar_object_key",
    "preprocess_primary_boreas_scan",
    "target_map_voxel_rule",
    "transform_preprocessed_map_scan_to_enu_ref",
]
