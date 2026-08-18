"""Read-only ROS1 bag inventory helpers for the Mid-360 pilot."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import PILOT_FLAGS


LIDAR_TOPIC = "/livox/lidar"
IMU_TOPIC = "/livox/imu"
SUPPORTED_LIDAR_TYPES = frozenset(
    {"sensor_msgs/PointCloud2", "livox_ros_driver/CustomMsg", "livox_ros_driver2/CustomMsg"}
)


class PilotBagError(RuntimeError):
    """Raised when a bag cannot satisfy the pilot parsing contract."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rosbag_module() -> Any:
    try:
        import rosbag  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on host ROS install
        raise PilotBagError(
            "ROS1 Python rosbag is unavailable; use the existing ROS1 environment"
        ) from exc
    return rosbag


def detect_bag_format(path: Path) -> str:
    with path.open("rb") as stream:
        magic = stream.readline(64).decode("ascii", errors="replace").strip()
    if magic != "#ROSBAG V2.0":
        raise PilotBagError(f"unsupported bag header: {magic!r}")
    return "ROS1_BAG_V2.0"


def detect_lidar_message_type(message: Any) -> str:
    message_type = str(getattr(message, "_type", ""))
    if message_type not in SUPPORTED_LIDAR_TYPES:
        raise PilotBagError(f"unsupported LiDAR message type: {message_type or '<missing>'}")
    return message_type


def message_header_timestamp(message: Any, bag_timestamp: Any) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and hasattr(stamp, "to_sec"):
        value = float(stamp.to_sec())
        if value > 0.0:
            return value
    return float(bag_timestamp.to_sec())


def iter_topic_messages(path: Path, topic: str) -> Iterator[tuple[int, Any, float, float]]:
    """Yield topic-local index, message, bag time, and effective header time."""

    rosbag = _rosbag_module()
    with rosbag.Bag(str(path), "r") as bag:
        for index, (_, message, bag_time) in enumerate(bag.read_messages(topics=[topic])):
            yield (
                index,
                message,
                float(bag_time.to_sec()),
                message_header_timestamp(message, bag_time),
            )


def _iso_utc(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def build_bag_inventory(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise PilotBagError(f"bag is not a regular file: {path}")
    bag_format = detect_bag_format(path)
    rosbag = _rosbag_module()
    with rosbag.Bag(str(path), "r") as bag:
        start = float(bag.get_start_time())
        end = float(bag.get_end_time())
        type_info = bag.get_type_and_topic_info()
        topic_rows: list[dict[str, Any]] = []
        for topic in sorted(type_info.topics):
            info = type_info.topics[topic]
            bag_times: list[float] = []
            header_times: list[float] = []
            frame_ids: set[str] = set()
            message_md5: set[str] = set()
            fields: list[dict[str, Any]] | None = None
            for _, message, bag_time in bag.read_messages(topics=[topic]):
                bag_seconds = float(bag_time.to_sec())
                bag_times.append(bag_seconds)
                header_times.append(message_header_timestamp(message, bag_time))
                header = getattr(message, "header", None)
                if header is not None:
                    frame_ids.add(str(getattr(header, "frame_id", "")))
                message_md5.add(str(getattr(message, "_md5sum", "")))
                if fields is None and hasattr(message, "fields"):
                    fields = [
                        {
                            "name": str(field.name),
                            "offset": int(field.offset),
                            "datatype": int(field.datatype),
                            "count": int(field.count),
                        }
                        for field in message.fields
                    ]
            count = int(info.message_count)
            duration = bag_times[-1] - bag_times[0] if len(bag_times) > 1 else 0.0
            average_rate = (len(bag_times) - 1) / duration if duration > 0.0 else None
            topic_rows.append(
                {
                    "topic": topic,
                    "message_type": str(info.msg_type),
                    "message_md5": sorted(message_md5),
                    "message_count": count,
                    "connections": int(info.connections),
                    "rosbag_reported_frequency_hz": (
                        float(info.frequency) if info.frequency is not None else None
                    ),
                    "average_bag_record_rate_hz": average_rate,
                    "first_bag_timestamp": bag_times[0] if bag_times else None,
                    "last_bag_timestamp": bag_times[-1] if bag_times else None,
                    "first_header_timestamp": header_times[0] if header_times else None,
                    "last_header_timestamp": header_times[-1] if header_times else None,
                    "frame_ids": sorted(frame_ids),
                    "fields": fields or [],
                }
            )

    by_topic = {row["topic"]: row for row in topic_rows}
    missing = [topic for topic in (LIDAR_TOPIC, IMU_TOPIC) if topic not in by_topic]
    if missing:
        raise PilotBagError(f"required topics missing: {', '.join(missing)}")
    lidar_type = str(by_topic[LIDAR_TOPIC]["message_type"])
    if lidar_type not in SUPPORTED_LIDAR_TYPES:
        raise PilotBagError(f"unsupported LiDAR topic type: {lidar_type}")
    if by_topic[IMU_TOPIC]["message_type"] != "sensor_msgs/Imu":
        raise PilotBagError(
            f"unsupported IMU topic type: {by_topic[IMU_TOPIC]['message_type']}"
        )
    return {
        "schema": "mid360_pilot_bag_inventory_v1",
        **PILOT_FLAGS,
        "bag_path": str(path),
        "bag_sha256": sha256_file(path),
        "bag_format": bag_format,
        "bag_size_bytes": path.stat().st_size,
        "start_timestamp": start,
        "start_time_utc": _iso_utc(start),
        "end_timestamp": end,
        "end_time_utc": _iso_utc(end),
        "duration_seconds": end - start,
        "message_count": sum(int(row["message_count"]) for row in topic_rows),
        "topics": topic_rows,
        "status": "PASS",
    }


def inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# Mid-360 single-bag inventory (PILOT_ONLY)",
        "",
        "This inventory is not formal real-data or measurement evidence.",
        "",
        f"- Bag: `{inventory['bag_path']}`",
        f"- SHA256: `{inventory['bag_sha256']}`",
        f"- Format: `{inventory['bag_format']}`",
        f"- Duration: `{inventory['duration_seconds']:.9f} s`",
        f"- Start: `{inventory['start_timestamp']:.9f}`",
        f"- End: `{inventory['end_timestamp']:.9f}`",
        "",
        "| Topic | Type | Messages | Average rate (Hz) | frame_id |",
        "|---|---|---:|---:|---|",
    ]
    for row in inventory["topics"]:
        rate = row["average_bag_record_rate_hz"]
        rate_text = "n/a" if rate is None else f"{rate:.6f}"
        frames = ", ".join(row["frame_ids"]) or "(none)"
        lines.append(
            f"| `{row['topic']}` | `{row['message_type']}` | "
            f"{row['message_count']} | {rate_text} | `{frames}` |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "IMU_TOPIC",
    "LIDAR_TOPIC",
    "PilotBagError",
    "build_bag_inventory",
    "detect_bag_format",
    "detect_lidar_message_type",
    "inventory_markdown",
    "iter_topic_messages",
    "message_header_timestamp",
    "sha256_file",
]
