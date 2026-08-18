"""Non-integrating IMU staticity screen for the single-bag pilot."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from . import PILOT_FLAGS
from .bag_reader import PilotBagError


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "median": median,
        "mad": float(np.median(np.abs(array - median))),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
        "absolute_max": float(np.max(np.abs(array))),
    }


def audit_imu_messages(
    records: Iterable[tuple[int, Any, float, float]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for message_index, message, bag_timestamp, timestamp in records:
        if str(getattr(message, "_type", "")) != "sensor_msgs/Imu":
            raise PilotBagError(f"unsupported IMU message type: {getattr(message, '_type', '')}")
        gyro = np.asarray(
            [message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z],
            dtype=np.float64,
        )
        accel = np.asarray(
            [message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z],
            dtype=np.float64,
        )
        raw.append(
            {
                "message_index": int(message_index),
                "timestamp": float(timestamp),
                "bag_timestamp": float(bag_timestamp),
                "gyro_x": float(gyro[0]),
                "gyro_y": float(gyro[1]),
                "gyro_z": float(gyro[2]),
                "gyro_norm": float(np.linalg.norm(gyro)),
                "accel_x": float(accel[0]),
                "accel_y": float(accel[1]),
                "accel_z": float(accel[2]),
                "accel_norm": float(np.linalg.norm(accel)),
            }
        )
    if not raw:
        raise PilotBagError("IMU topic is empty")
    timestamps = np.asarray([row["timestamp"] for row in raw], dtype=np.float64)
    intervals = np.diff(timestamps)
    values = {
        name: np.asarray([row[name] for row in raw], dtype=np.float64)
        for name in (
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "gyro_norm",
            "accel_x",
            "accel_y",
            "accel_z",
            "accel_norm",
        )
    }
    if not all(np.all(np.isfinite(array)) for array in values.values()):
        raise PilotBagError("IMU contains nonfinite angular velocity or acceleration")
    gyro_vectors = np.column_stack((values["gyro_x"], values["gyro_y"], values["gyro_z"]))
    accel_vectors = np.column_stack((values["accel_x"], values["accel_y"], values["accel_z"]))
    gyro_steps = np.linalg.norm(np.diff(gyro_vectors, axis=0), axis=1)
    accel_steps = np.linalg.norm(np.diff(accel_vectors, axis=0), axis=1)
    median_interval = float(np.median(intervals)) if intervals.size else None
    thresholds = config["imu_staticity_screen"]
    missing_threshold = 3.0 * median_interval if median_interval is not None else float("inf")
    gyro_step_limit = float(thresholds["gyro_step_max"])
    accel_step_limit = float(thresholds["accel_step_max_raw"])
    for index, row in enumerate(raw):
        interval = None if index == 0 else float(intervals[index - 1])
        gyro_step = None if index == 0 else float(gyro_steps[index - 1])
        accel_step = None if index == 0 else float(accel_steps[index - 1])
        row["inter_message_interval_seconds"] = interval
        row["missing_interval"] = bool(interval is not None and interval > missing_threshold)
        row["gyro_step"] = gyro_step
        row["accel_step_raw"] = accel_step
        row["large_spike"] = bool(
            (gyro_step is not None and gyro_step > gyro_step_limit)
            or (accel_step is not None and accel_step > accel_step_limit)
        )
    stats = {name: _summary(array) for name, array in values.items()}
    accel_median = stats["accel_norm"]["median"]
    accel_abs_deviation = float(np.max(np.abs(values["accel_norm"] - accel_median)))
    criteria = {
        "gyro_norm_q99_exceeded": stats["gyro_norm"]["q99"]
        > float(thresholds["gyro_norm_q99_max"]),
        "gyro_norm_max_exceeded": stats["gyro_norm"]["max"]
        > float(thresholds["gyro_norm_max"]),
        "gyro_step_exceeded": bool(gyro_steps.size and np.max(gyro_steps) > gyro_step_limit),
        "accel_norm_q99_spread_exceeded": stats["accel_norm"]["q99"] - accel_median
        > float(thresholds["accel_norm_q99_minus_median_max_raw"]),
        "accel_norm_absolute_deviation_exceeded": accel_abs_deviation
        > float(thresholds["accel_norm_absolute_deviation_max_raw"]),
        "accel_step_exceeded": bool(accel_steps.size and np.max(accel_steps) > accel_step_limit),
    }
    monotonic = bool(np.all(intervals >= 0.0))
    strictly_monotonic = bool(np.all(intervals > 0.0))
    if not strictly_monotonic:
        screen = "INCONCLUSIVE"
    elif any(criteria.values()):
        screen = "MOTION_SUSPECTED"
    else:
        screen = "NO_OBVIOUS_MOTION"
    duration = float(timestamps[-1] - timestamps[0])
    acceleration_scale_note = (
        "UNCONFIRMED_RAW_APPROXIMATELY_1G_SCALE_NO_CONVERSION_APPLIED"
        if 0.5 <= accel_median <= 1.5
        else "UNCONFIRMED_RAW_UNIT_NO_CONVERSION_APPLIED"
    )
    summary = {
        "schema": "mid360_pilot_imu_staticity_summary_v1",
        **PILOT_FLAGS,
        "status": "PASS" if strictly_monotonic else "FAIL",
        "STATICITY_SCREEN": screen,
        "interpretation": "PILOT_QA_ONLY_NOT_GROUND_TRUTH_NO_IMU_INTEGRATION",
        "message_count": len(raw),
        "first_header_timestamp": float(timestamps[0]),
        "last_header_timestamp": float(timestamps[-1]),
        "duration_seconds": duration,
        "effective_rate_hz": (len(raw) - 1) / duration if duration > 0.0 else None,
        "timestamp_monotonic": monotonic,
        "timestamp_strictly_monotonic": strictly_monotonic,
        "median_inter_message_interval_seconds": median_interval,
        "maximum_inter_message_interval_seconds": float(np.max(intervals)) if intervals.size else None,
        "missing_interval_count": sum(bool(row["missing_interval"]) for row in raw),
        "large_spike_count": sum(bool(row["large_spike"]) for row in raw),
        "gyro_semantic_unit": "sensor_msgs/Imu_declares_rad_per_second_not_independently_calibrated",
        "acceleration_unit_confirmed": False,
        "acceleration_unit_note": acceleration_scale_note,
        "acceleration_conversion_applied": False,
        "statistics": stats,
        "maximum_gyro_step": float(np.max(gyro_steps)) if gyro_steps.size else 0.0,
        "maximum_accel_step_raw": float(np.max(accel_steps)) if accel_steps.size else 0.0,
        "maximum_accel_norm_absolute_deviation_raw": accel_abs_deviation,
        "motion_screen_thresholds": dict(thresholds),
        "motion_screen_criteria": criteria,
    }
    return raw, summary


__all__ = ["audit_imu_messages"]
