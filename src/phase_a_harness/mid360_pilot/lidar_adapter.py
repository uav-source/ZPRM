"""PointCloud2 and Livox CustomMsg adapters used only by the pilot."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .bag_reader import PilotBagError, detect_lidar_message_type
from . import PILOT_FLAGS


POINT_FIELD_FORMATS: dict[int, str] = {
    1: "i1",  # INT8
    2: "u1",  # UINT8
    3: "i2",  # INT16
    4: "u2",  # UINT16
    5: "i4",  # INT32
    6: "u4",  # UINT32
    7: "f4",  # FLOAT32
    8: "f8",  # FLOAT64
}


def pointcloud2_dtype(message: Any) -> np.dtype[Any]:
    endian = ">" if bool(message.is_bigendian) else "<"
    names: list[str] = []
    formats: list[Any] = []
    offsets: list[int] = []
    seen: set[str] = set()
    for field in message.fields:
        name = str(field.name)
        if not name or name in seen:
            raise PilotBagError(f"invalid or duplicate PointCloud2 field: {name!r}")
        seen.add(name)
        base = POINT_FIELD_FORMATS.get(int(field.datatype))
        if base is None:
            raise PilotBagError(
                f"unsupported PointCloud2 datatype {field.datatype} for field {name}"
            )
        count = int(field.count)
        if count <= 0:
            raise PilotBagError(f"invalid PointCloud2 field count for {name}: {count}")
        scalar = np.dtype(base if base in {"i1", "u1"} else endian + base)
        names.append(name)
        formats.append(scalar if count == 1 else (scalar, (count,)))
        offsets.append(int(field.offset))
    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(message.point_step),
        }
    )
    return dtype


def pointcloud2_to_structured(message: Any) -> np.ndarray:
    if str(getattr(message, "_type", "")) != "sensor_msgs/PointCloud2":
        raise PilotBagError("message is not sensor_msgs/PointCloud2")
    height = int(message.height)
    width = int(message.width)
    point_step = int(message.point_step)
    row_step = int(message.row_step)
    if height <= 0 or width < 0 or point_step <= 0 or row_step < width * point_step:
        raise PilotBagError("invalid PointCloud2 dimensions or strides")
    required = row_step * height
    if len(message.data) < required:
        raise PilotBagError("PointCloud2 data buffer is shorter than row_step * height")
    dtype = pointcloud2_dtype(message)
    organized = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=message.data,
        strides=(row_step, point_step),
    )
    return np.array(organized.reshape(-1), copy=True)


def livox_custom_to_structured(message: Any) -> np.ndarray:
    points = getattr(message, "points", None)
    if points is None:
        raise PilotBagError("Livox CustomMsg has no points sequence")
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
            ("tag", "u1"),
            ("line", "u1"),
            ("offset_time", "<u4"),
        ]
    )
    output = np.empty(len(points), dtype=dtype)
    for index, point in enumerate(points):
        output[index] = (
            float(point.x),
            float(point.y),
            float(point.z),
            float(getattr(point, "reflectivity", getattr(point, "intensity", 0.0))),
            int(point.tag),
            int(point.line),
            int(point.offset_time),
        )
    return output


def lidar_message_to_structured(message: Any) -> np.ndarray:
    message_type = detect_lidar_message_type(message)
    if message_type == "sensor_msgs/PointCloud2":
        output = pointcloud2_to_structured(message)
    else:
        output = livox_custom_to_structured(message)
    missing = {"x", "y", "z"} - set(output.dtype.names or ())
    if missing:
        raise PilotBagError(f"LiDAR payload missing coordinate fields: {sorted(missing)}")
    return output


def xyz_array(points: np.ndarray) -> np.ndarray:
    return np.column_stack((points["x"], points["y"], points["z"])).astype(
        np.float64, copy=False
    )


def infer_point_timestamp(points: np.ndarray, header_timestamp: float) -> dict[str, Any]:
    names = set(points.dtype.names or ())
    field = "timestamp" if "timestamp" in names else "offset_time" if "offset_time" in names else None
    if field is None or points.size == 0:
        return {
            "field": field,
            "available": False,
            "unit_inference": None,
            "usable": False,
            "raw_min": None,
            "raw_max": None,
            "span_seconds": None,
            "nondecreasing": None,
        }
    raw = np.asarray(points[field], dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return {
            "field": field,
            "available": True,
            "unit_inference": None,
            "usable": False,
            "raw_min": None,
            "raw_max": None,
            "span_seconds": None,
            "nondecreasing": None,
        }
    raw_min = float(np.min(finite))
    raw_max = float(np.max(finite))
    median = float(np.median(finite))
    if field == "offset_time":
        scale = 1.0e-9
        unit = "nanoseconds_relative_to_header"
    else:
        candidates = (
            (1.0, "seconds_absolute"),
            (1.0e-3, "milliseconds_absolute"),
            (1.0e-6, "microseconds_absolute"),
            (1.0e-9, "nanoseconds_absolute"),
        )
        scale, unit = min(
            candidates,
            key=lambda candidate: abs(median * candidate[0] - header_timestamp),
        )
        if abs(median * scale - header_timestamp) > 10.0:
            unit = "unknown"
            scale = float("nan")
    span = (raw_max - raw_min) * scale if np.isfinite(scale) else None
    diffs = np.diff(raw)
    nondecreasing = bool(np.all(diffs >= 0.0)) if raw.size > 1 else True
    usable = bool(
        unit != "unknown"
        and finite.size == raw.size
        and span is not None
        and 0.0 <= span <= 1.0
    )
    return {
        "field": field,
        "available": True,
        "unit_inference": unit,
        "usable": usable,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "span_seconds": span,
        "nondecreasing": nondecreasing,
    }


def audit_lidar_messages(
    records: Iterable[tuple[int, Any, float, float]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    field_signatures: set[tuple[str, ...]] = set()
    timestamp_units: set[str] = set()
    message_types: set[str] = set()
    previous_timestamp: float | None = None
    for frame_index, message, bag_timestamp, timestamp in records:
        message_type = detect_lidar_message_type(message)
        structured = lidar_message_to_structured(message)
        xyz = xyz_array(structured)
        finite_mask = np.all(np.isfinite(xyz), axis=1)
        finite_xyz = xyz[finite_mask]
        ranges = np.linalg.norm(finite_xyz, axis=1) if finite_xyz.size else np.empty(0)
        time_info = infer_point_timestamp(structured, timestamp)
        fields = tuple(structured.dtype.names or ())
        field_signatures.add(fields)
        message_types.add(message_type)
        if time_info["unit_inference"] is not None:
            timestamp_units.add(str(time_info["unit_inference"]))
        interval = None if previous_timestamp is None else timestamp - previous_timestamp
        previous_timestamp = timestamp
        coordinate_stats: dict[str, float | None] = {}
        for axis_index, axis in enumerate(("x", "y", "z")):
            values = finite_xyz[:, axis_index] if finite_xyz.size else np.empty(0)
            coordinate_stats[f"{axis}_min"] = float(np.min(values)) if values.size else None
            coordinate_stats[f"{axis}_max"] = float(np.max(values)) if values.size else None
        range_stats = {
            "range_min": float(np.min(ranges)) if ranges.size else None,
            "range_median": float(np.median(ranges)) if ranges.size else None,
            "range_q95": float(np.quantile(ranges, 0.95)) if ranges.size else None,
            "range_max": float(np.max(ranges)) if ranges.size else None,
        }
        rows.append(
            {
                "frame_index": int(frame_index),
                "timestamp": float(timestamp),
                "bag_timestamp": float(bag_timestamp),
                "inter_frame_interval_seconds": interval,
                "point_count": int(xyz.shape[0]),
                "finite_point_count": int(np.count_nonzero(finite_mask)),
                "nonfinite_point_count": int(xyz.shape[0] - np.count_nonzero(finite_mask)),
                **coordinate_stats,
                **range_stats,
                "point_timestamp_field": time_info["field"],
                "point_timestamp_unit_inference": time_info["unit_inference"],
                "point_timestamp_raw_min": time_info["raw_min"],
                "point_timestamp_raw_max": time_info["raw_max"],
                "point_timestamp_span_seconds": time_info["span_seconds"],
                "point_timestamp_nondecreasing": time_info["nondecreasing"],
                "point_timestamp_usable": time_info["usable"],
            }
        )
    if not rows:
        raise PilotBagError("LiDAR topic is empty")
    timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
    intervals = np.diff(timestamps)
    counts = np.asarray([row["point_count"] for row in rows], dtype=np.int64)
    median_interval = float(np.median(intervals)) if intervals.size else None
    median_count = float(np.median(counts))
    for row in rows:
        interval = row["inter_frame_interval_seconds"]
        row["frame_rate_anomaly"] = bool(
            interval is not None
            and median_interval is not None
            and (interval <= 0.0 or interval > 3.0 * median_interval)
        )
        row["point_count_drop"] = bool(row["point_count"] < 0.5 * median_count)
    duration = float(timestamps[-1] - timestamps[0])
    monotonic = bool(np.all(intervals >= 0.0))
    strictly_monotonic = bool(np.all(intervals > 0.0))
    empty_frames = sum(row["point_count"] == 0 for row in rows)
    nonfinite_total = sum(int(row["nonfinite_point_count"]) for row in rows)
    usable_point_times = sum(bool(row["point_timestamp_usable"]) for row in rows)
    nondecreasing_point_times = sum(
        row["point_timestamp_nondecreasing"] is True for row in rows
    )
    status = (
        "PASS"
        if empty_frames == 0 and strictly_monotonic and usable_point_times == len(rows)
        else "FAIL"
    )
    summary = {
        "schema": "mid360_pilot_lidar_quality_summary_v1",
        **PILOT_FLAGS,
        "status": status,
        "message_types": sorted(message_types),
        "frame_count": len(rows),
        "first_header_timestamp": float(timestamps[0]),
        "last_header_timestamp": float(timestamps[-1]),
        "lidar_duration_seconds": duration,
        "effective_frame_rate_hz": (len(rows) - 1) / duration if duration > 0 else None,
        "timestamp_monotonic": monotonic,
        "timestamp_strictly_monotonic": strictly_monotonic,
        "median_inter_frame_interval_seconds": median_interval,
        "minimum_inter_frame_interval_seconds": float(np.min(intervals)) if intervals.size else None,
        "maximum_inter_frame_interval_seconds": float(np.max(intervals)) if intervals.size else None,
        "frame_rate_anomaly_count": sum(bool(row["frame_rate_anomaly"]) for row in rows),
        "empty_frame_count": empty_frames,
        "nonfinite_point_count": nonfinite_total,
        "point_count_min": int(np.min(counts)),
        "point_count_median": median_count,
        "point_count_mean": float(np.mean(counts)),
        "point_count_max": int(np.max(counts)),
        "point_count_drop_frame_count": sum(bool(row["point_count_drop"]) for row in rows),
        "field_signatures": [list(fields) for fields in sorted(field_signatures)],
        "point_coordinate_unit": "UNCONFIRMED_MESSAGE_HAS_NO_UNIT_METADATA_INTERPRETED_AS_METERS",
        "point_timestamp_frame_count": sum(row["point_timestamp_field"] is not None for row in rows),
        "point_timestamp_usable_frame_count": usable_point_times,
        "point_timestamp_all_frames_usable": usable_point_times == len(rows),
        "point_timestamp_nondecreasing_frame_count": nondecreasing_point_times,
        "point_timestamp_all_frames_nondecreasing": nondecreasing_point_times == len(rows),
        "point_timestamp_unit_inferences": sorted(timestamp_units),
        "point_timestamp_semantics": "PER_POINT_ABSOLUTE_OR_OFFSET_TIME_INFERRED_BY_HEADER_MAGNITUDE",
        "point_timestamp_order_note": (
            "SERIALIZED_POINT_ORDER_IS_NOT_TIME_MONOTONIC_USE_PER_POINT_VALUES_OR_EXPLICIT_SORT"
            if nondecreasing_point_times != len(rows)
            else "SERIALIZED_POINT_ORDER_IS_TIME_NONDECREASING"
        ),
    }
    return rows, summary


__all__ = [
    "infer_point_timestamp",
    "audit_lidar_messages",
    "lidar_message_to_structured",
    "livox_custom_to_structured",
    "pointcloud2_dtype",
    "pointcloud2_to_structured",
    "xyz_array",
]
