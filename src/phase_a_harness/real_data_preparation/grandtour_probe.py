"""Read downloaded GrandTour reference Zarrs in the isolated data-tool venv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .geodesy import wgs84_geodetic_to_ecef
from .overlap import compute_gt_only_overlap


MISSIONS = {
    "SPX-1": "2024-11-02-17-10-25",
    "SPX-3": "2024-11-02-17-43-10",
}


def _trajectory(data_root: Path, mission: str) -> tuple[np.ndarray, dict[str, Any]]:
    base = data_root / "grandtour" / mission / "reference_extracted"
    navsat = zarr.open_group(str(base / "navsatfix_cpt7_ie_tc"), mode="r")
    odometry = zarr.open_group(str(base / "cpt7_ie_tc_odometry"), mode="r")
    timestamps = np.asarray(navsat["timestamp"][:], dtype=np.float64)
    odometry_timestamps = np.asarray(odometry["timestamp"][:], dtype=np.float64)
    if not np.array_equal(timestamps, odometry_timestamps):
        raise ValueError(f"NavSatFix/odometry timestamps differ for {mission}")
    positions = wgs84_geodetic_to_ecef(
        navsat["lat"][:], navsat["long"][:], navsat["alt"][:]
    )
    orientations = np.asarray(odometry["pose_orien"][:], dtype=np.float64)
    if orientations.shape != (timestamps.size, 4):
        raise ValueError(f"invalid 6DoF orientation array for {mission}")
    norms = np.linalg.norm(orientations, axis=1)
    navsat_covariance = np.asarray(navsat["cov"][:], dtype=np.float64)
    pose_covariance = np.asarray(odometry["pose_cov"][:], dtype=np.float64)
    stats = {
        "frame_id": navsat.attrs.get("frame_id"),
        "odometry_frame_id": odometry.attrs.get("frame_id"),
        "orientation_present": True,
        "orientation_quaternion_norm_max_error": float(np.max(np.abs(norms - 1.0))),
        "pose_count": int(timestamps.size),
        "pose_covariance_all_zero": bool(np.count_nonzero(pose_covariance) == 0),
        "position_covariance_all_zero": bool(np.count_nonzero(navsat_covariance) == 0),
        "position_covariance_diagonal_median": np.median(
            np.diagonal(navsat_covariance, axis1=1, axis2=2), axis=0
        ).tolist(),
        "start_timestamp": float(timestamps[0]),
        "end_timestamp": float(timestamps[-1]),
        "max_timestamp_gap_s": float(np.max(np.diff(timestamps))),
        "median_timestamp_gap_s": float(np.median(np.diff(timestamps))),
        "strictly_monotonic": bool(np.all(np.diff(timestamps) > 0.0)),
    }
    return np.column_stack((timestamps, positions)), stats


def build_report(data_root: Path) -> dict[str, Any]:
    trajectories: dict[str, np.ndarray] = {}
    stats: dict[str, Any] = {}
    for short_name, mission in MISSIONS.items():
        trajectories[short_name], stats[short_name] = _trajectory(data_root, mission)
    overlaps = {}
    for map_id, query_id in (("SPX-1", "SPX-3"), ("SPX-3", "SPX-1")):
        overlaps[f"{map_id}->{query_id}"] = compute_gt_only_overlap(
            trajectories[map_id],
            trajectories[query_id],
            common_world_frame_proven=True,
        )
    return {
        "common_position_frame": "WGS84_ECEF_FROM_PUBLISHED_IE_TC_NAVSATFIX",
        "gt_only_overlap": overlaps,
        "missions": stats,
        "no_point_cloud_or_registration_input_read": True,
        "probe_pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build_report(arguments.data_root.resolve(strict=True)), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
