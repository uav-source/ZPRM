"""Registration-free target map construction for a stationary Mid-360 pilot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import PILOT_FLAGS
from .bag_reader import PilotBagError


def finite_range_filter(
    points: np.ndarray, *, minimum_range_m: float, maximum_range_m: float
) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise PilotBagError("points must be an Nx3 array")
    finite = np.all(np.isfinite(xyz), axis=1)
    ranges = np.linalg.norm(xyz, axis=1)
    accepted = finite & (ranges >= minimum_range_m) & (ranges <= maximum_range_m)
    return np.ascontiguousarray(xyz[accepted], dtype=np.float64)


def deterministic_voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise PilotBagError("voxel input must be a finite Nx3 array")
    if voxel_size_m <= 0.0:
        raise PilotBagError("voxel_size_m must be positive")
    if xyz.size == 0:
        raise PilotBagError("cannot voxelize an empty point set")
    keys = np.floor(xyz / float(voxel_size_m)).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    starts = np.r_[True, np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)]
    group_ids = np.cumsum(starts) - 1
    count = int(group_ids[-1]) + 1
    sums = np.zeros((count, 3), dtype=np.float64)
    np.add.at(sums, group_ids, xyz[order])
    counts = np.bincount(group_ids, minlength=count).astype(np.float64)
    return np.ascontiguousarray(sums / counts[:, None], dtype=np.float64)


def build_static_target_map(
    scan_points: Iterable[np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    map_config = config["map"]
    minimum = float(map_config["minimum_range_m"])
    maximum = float(map_config["maximum_range_m"])
    voxel = float(map_config["voxel_size_m"])
    filtered: list[np.ndarray] = []
    raw_count = 0
    accepted_count = 0
    scan_count = 0
    for scan in scan_points:
        scan_count += 1
        raw_count += int(np.asarray(scan).shape[0])
        accepted = finite_range_filter(
            scan, minimum_range_m=minimum, maximum_range_m=maximum
        )
        accepted_count += int(accepted.shape[0])
        if accepted.size:
            filtered.append(accepted)
    if scan_count == 0 or not filtered:
        raise PilotBagError("map interval contains no usable LiDAR points")
    merged = np.concatenate(filtered, axis=0)
    target = deterministic_voxel_downsample(merged, voxel)
    metadata = {
        "schema": "mid360_pilot_target_map_metadata_v1",
        **PILOT_FLAGS,
        "construction": "DIRECT_SAME_SENSOR_FRAME_MERGE_NO_REGISTRATION",
        "registration_called": False,
        "scan_matching_called": False,
        "T_map_from_lidar": "IDENTITY_BY_STATIC_PILOT_ASSUMPTION",
        "map_scan_count": scan_count,
        "raw_point_count": raw_count,
        "finite_range_filtered_point_count": accepted_count,
        "target_map_point_count": int(target.shape[0]),
        "minimum_range_m": minimum,
        "maximum_range_m": maximum,
        "voxel_size_m": voxel,
        "canonical_dtype": "float64",
        "canonical_shape": list(target.shape),
        "status": "PASS",
    }
    return target, metadata


def npy_array_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_ascii_pcd(path: Path, points: np.ndarray) -> None:
    xyz = np.asarray(points, dtype=np.float64)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z\nSIZE 8 8 8\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {xyz.shape[0]}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {xyz.shape[0]}\nDATA ascii\n"
    )
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(header)
        np.savetxt(stream, xyz, fmt="%.10g %.10g %.10g")


__all__ = [
    "build_static_target_map",
    "deterministic_voxel_downsample",
    "finite_range_filter",
    "npy_array_sha256",
    "write_ascii_pcd",
]
