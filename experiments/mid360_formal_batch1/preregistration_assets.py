"""Registration-free FMB1 target, snapshot, and geometry-only builders.

This module is deliberately split into two executable phases.  ``materialize``
must run in the ROS Noetic Python environment so it can read the source bags.
``analyze_geometry`` consumes only frozen ``.npy`` files and may run in the
project's Python 3.11 environment.  Neither phase imports or calls a
registration backend.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from phase_a_harness.mid360_pilot.bag_reader import (
    LIDAR_TOPIC,
    PilotBagError,
    iter_topic_messages,
    sha256_file,
)
from phase_a_harness.mid360_pilot.lidar_adapter import (
    lidar_message_to_structured,
    xyz_array,
)
from phase_a_harness.mid360_pilot.pilot_geometry import compute_pilot_geometry
from phase_a_harness.mid360_pilot.split import select_query_frames
from phase_a_harness.mid360_pilot.static_map import (
    build_static_target_map,
    finite_range_filter,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    GEOMETRY_ONLY_FIELDS,
)

from .protocol import QUERY_QUANTILES, STATION_IDS, geometry_admission


class PreRegistrationAssetError(RuntimeError):
    """Raised when a frozen pre-registration asset contract is violated."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_save(path: Path, points: np.ndarray) -> dict[str, Any]:
    canonical = np.ascontiguousarray(points, dtype="<f8")
    if canonical.ndim != 2 or canonical.shape[1] != 3:
        raise PreRegistrationAssetError("canonical point array must be Nx3")
    if canonical.shape[0] == 0 or not np.all(np.isfinite(canonical)):
        raise PreRegistrationAssetError("canonical point array must be nonempty and finite")
    if path.exists():
        raise PreRegistrationAssetError(f"refusing to overwrite frozen asset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, canonical, allow_pickle=False)
    temporary.replace(path)
    return {
        "path": str(path.resolve(strict=True)),
        "npy_sha256": sha256_file(path),
        "array_sha256": _array_sha256(canonical),
        "point_count": int(canonical.shape[0]),
        "dtype": str(canonical.dtype),
        "shape": list(canonical.shape),
        "c_contiguous": bool(canonical.flags.c_contiguous),
    }


def _raw_bag_rows(acquisition: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = acquisition.get("raw_bags")
    if not isinstance(rows, list):
        rows = acquisition.get("bags")
    if not isinstance(rows, list):
        raise PreRegistrationAssetError("acquisition payload has no raw_bags list")
    return rows


def _bag_row(
    acquisition: Mapping[str, Any], scene_id: str, station_id: str, role: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in _raw_bag_rows(acquisition)
        if str(row.get("scene_id")) == scene_id
        and str(row.get("station_id")) == station_id
        and str(row.get("role")) == role
    ]
    if len(matches) != 1:
        raise PreRegistrationAssetError(
            f"expected one {role} bag for {scene_id}/{station_id}, got {len(matches)}"
        )
    return matches[0]


def _raw_path(row: Mapping[str, Any]) -> Path:
    value = row.get("raw_absolute_path", row.get("raw_path"))
    if value is None:
        inventory = row.get("inventory", {})
        value = inventory.get("bag_path") if isinstance(inventory, Mapping) else None
    if value is None:
        raise PreRegistrationAssetError("raw bag row has no path")
    return Path(str(value)).resolve(strict=True)


def _raw_sha(row: Mapping[str, Any]) -> str:
    value = row.get("sha256", row.get("SHA256"))
    if value is None:
        inventory = row.get("inventory", {})
        value = inventory.get("bag_sha256") if isinstance(inventory, Mapping) else None
    if not isinstance(value, str) or len(value) != 64:
        raise PreRegistrationAssetError("raw bag row has no valid SHA256")
    return value


def _passing_station_keys(acquisition: Mapping[str, Any]) -> list[tuple[str, str]]:
    stations = acquisition.get("stations")
    if not isinstance(stations, list):
        raise PreRegistrationAssetError("acquisition payload has no stations list")
    keys: list[tuple[str, str]] = []
    for row in stations:
        status = row.get(
            "acquisition_status",
            row.get("station_status", row.get("station_acquisition_status")),
        )
        if status in {"ACQUISITION_PASS", "PASS"}:
            keys.append((str(row["scene_id"]), str(row["station_id"])))
    if len(keys) != len(set(keys)):
        raise PreRegistrationAssetError("duplicate station acquisition records")
    return sorted(keys)


def _iter_xyz_scans(path: Path) -> Iterator[np.ndarray]:
    found = False
    for _, message, _, _ in iter_topic_messages(path, LIDAR_TOPIC):
        found = True
        yield xyz_array(lidar_message_to_structured(message))
    if not found:
        raise PilotBagError(f"LiDAR topic is empty: {path}")


def _query_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_index, message, bag_timestamp, timestamp in iter_topic_messages(
        path, LIDAR_TOPIC
    ):
        points = xyz_array(lidar_message_to_structured(message))
        rows.append(
            {
                "frame_index": int(frame_index),
                "timestamp": float(timestamp),
                "bag_timestamp": float(bag_timestamp),
                "point_count": int(points.shape[0]),
                "finite_point_count": int(
                    np.count_nonzero(np.all(np.isfinite(points), axis=1))
                ),
            }
        )
    if not rows:
        raise PreRegistrationAssetError(f"QUERY has no LiDAR frames: {path}")
    return rows


def _load_selected_query_points(
    path: Path, selected: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[int, np.ndarray]:
    selected_indexes = {int(row["frame_index"]) for row in selected}
    output: dict[int, np.ndarray] = {}
    minimum = float(config["map"]["minimum_range_m"])
    maximum = float(config["map"]["maximum_range_m"])
    for frame_index, message, _, _ in iter_topic_messages(path, LIDAR_TOPIC):
        if int(frame_index) not in selected_indexes:
            continue
        points = xyz_array(lidar_message_to_structured(message))
        accepted = finite_range_filter(
            points, minimum_range_m=minimum, maximum_range_m=maximum
        )
        if accepted.shape[0] == 0:
            raise PreRegistrationAssetError(
                f"selected QUERY frame {frame_index} is empty after frozen filtering"
            )
        output[int(frame_index)] = accepted
    if set(output) != selected_indexes:
        raise PreRegistrationAssetError("not all frozen QUERY selections were materialized")
    return output


def materialize_registration_free_assets(
    acquisition: Mapping[str, Any],
    *,
    runtime_dir: Path,
    config: Mapping[str, Any],
    source_bindings: Mapping[str, str],
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Build MAP-only targets and exact-quantile QUERY snapshots.

    This function intentionally has no backend option and no transform output.
    It processes stations sequentially to keep the direct-merge memory peak
    bounded.
    """

    targets: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for scene_id, station_id in _passing_station_keys(acquisition):
        map_row = _bag_row(acquisition, scene_id, station_id, "MAP")
        query_row = _bag_row(acquisition, scene_id, station_id, "QUERY")
        map_path = _raw_path(map_row)
        query_path = _raw_path(query_row)
        try:
            target, metadata = build_static_target_map(
                _iter_xyz_scans(map_path), config
            )
            target_asset = _canonical_save(
                runtime_dir / "targets" / scene_id / station_id / "target_points.npy",
                target,
            )
            target_row = {
                "scene_id": scene_id,
                "station_id": station_id,
                "map_bag_path": str(map_path),
                "map_bag_sha256": _raw_sha(map_row),
                "input_roles": ["MAP"],
                "query_frame_count": 0,
                "query_contribution_to_target": 0,
                "construction": "DIRECT_SAME_SENSOR_FRAME_MERGE_NO_REGISTRATION",
                "registration_called": False,
                "odometry_called": False,
                "scan_matching_called": False,
                "map_frame_count": int(metadata["map_scan_count"]),
                "raw_point_count": int(metadata["raw_point_count"]),
                "filtered_point_count": int(
                    metadata["finite_range_filtered_point_count"]
                ),
                "target_point_count": int(metadata["target_map_point_count"]),
                "minimum_range_m": float(metadata["minimum_range_m"]),
                "maximum_range_m": float(metadata["maximum_range_m"]),
                "voxel_size_m": float(metadata["voxel_size_m"]),
                "preprocessing_source_sha256": dict(source_bindings),
                **{f"target_{key}": value for key, value in target_asset.items()},
            }
            targets.append(target_row)
            del target

            candidates = _query_candidates(query_path)
            # Selection is over every audited QUERY LiDAR frame.  Empty or bad
            # selected frames fail the station; they are never silently removed
            # before re-ranking.
            selected = select_query_frames(candidates, QUERY_QUANTILES)
            selected_points = _load_selected_query_points(query_path, selected, config)
            for item in selected:
                selection_index = int(item["selection_index"])
                snapshot_id = f"{scene_id}_{station_id}_Q{selection_index + 1:02d}"
                frame_index = int(item["frame_index"])
                source_asset = _canonical_save(
                    runtime_dir
                    / "snapshots"
                    / scene_id
                    / station_id
                    / f"{snapshot_id}.npy",
                    selected_points[frame_index],
                )
                snapshots.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "snapshot_id": snapshot_id,
                        "selection_index": selection_index,
                        "quantile": float(item["quantile"]),
                        "candidate_count": len(candidates),
                        "target_sequence_rank": float(item["target_sequence_rank"]),
                        "selected_sequence_rank": int(item["selected_sequence_rank"]),
                        "query_frame_index": frame_index,
                        "query_timestamp": float(item["timestamp"]),
                        "query_bag_path": str(query_path),
                        "query_bag_sha256": _raw_sha(query_row),
                        "selection_method": (
                            "FIXED_QUANTILE_SEQUENCE_RANK_NEAREST_UNUSED_EARLIER_TIE"
                        ),
                        "selection_frozen_before_registration": True,
                        "target_npy_sha256": target_row["target_npy_sha256"],
                        "target_array_sha256": target_row["target_array_sha256"],
                        **{f"source_{key}": value for key, value in source_asset.items()},
                    }
                )
            del selected_points
        except Exception as exc:
            failures.append(
                {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "stage": "REGISTRATION_FREE_ASSET_BUILD",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            gc.collect()

    by_scene: dict[str, list[str]] = {}
    for row in targets:
        by_scene.setdefault(str(row["scene_id"]), []).append(str(row["station_id"]))
    complete_scenes = sorted(
        scene
        for scene, station_ids in by_scene.items()
        if sorted(station_ids) == sorted(STATION_IDS)
    )
    payload = {
        "schema": "mid360_fmb1_preregistration_assets_v1",
        "target_construction": "MAP_ONLY_DIRECT_MERGE_NO_REGISTRATION",
        "query_contribution_to_every_target": 0,
        "quantiles": list(QUERY_QUANTILES),
        "targets": targets,
        "snapshots": snapshots,
        "target_count": len(targets),
        "snapshot_count": len(snapshots),
        "complete_scene_ids": complete_scenes,
        "failures": failures,
        "open3d_registration_call_count": 0,
        "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0,
        "formal_trial_count": 0,
    }
    if write_manifest:
        _write_json(runtime_dir / "asset_manifest.json", payload)
    return payload


def analyze_geometry_only(
    assets: Mapping[str, Any],
    *,
    runtime_dir: Path,
    config: Mapping[str, Any],
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Calculate only the ten initial geometry metrics at identity."""

    target_rows = assets.get("targets", [])
    snapshot_rows = assets.get("snapshots", [])
    metrics: list[dict[str, Any]] = []
    station_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for target_row in sorted(
        target_rows, key=lambda row: (str(row["scene_id"]), str(row["station_id"]))
    ):
        scene_id = str(target_row["scene_id"])
        station_id = str(target_row["station_id"])
        station_snapshots = sorted(
            (
                row
                for row in snapshot_rows
                if str(row["scene_id"]) == scene_id
                and str(row["station_id"]) == station_id
            ),
            key=lambda row: int(row["selection_index"]),
        )
        try:
            if len(station_snapshots) != 10:
                raise PreRegistrationAssetError(
                    f"{scene_id}/{station_id} does not have exactly 10 snapshots"
                )
            target = np.load(
                Path(str(target_row["target_path"])), mmap_mode="r", allow_pickle=False
            )
            sources = [
                np.load(Path(str(row["source_path"])), allow_pickle=False)
                for row in station_snapshots
            ]
            metadata = [
                {
                    "frame_index": int(row["query_frame_index"]),
                    "timestamp": float(row["query_timestamp"]),
                }
                for row in station_snapshots
            ]
            station_metrics, station_summary = compute_pilot_geometry(
                sources, metadata, target, config
            )
            if len(station_metrics) != 10:
                raise PreRegistrationAssetError("geometry core did not return 10 rows")
            for snapshot, raw_metrics in zip(station_snapshots, station_metrics):
                observed = {key: raw_metrics[key] for key in GEOMETRY_ONLY_FIELDS}
                if set(observed) != set(GEOMETRY_ONLY_FIELDS):
                    raise PreRegistrationAssetError("geometry-only schema drift")
                metrics.append(
                    {
                        "scene_id": scene_id,
                        "station_id": station_id,
                        "snapshot_id": snapshot["snapshot_id"],
                        "selection_index": int(snapshot["selection_index"]),
                        **observed,
                    }
                )
            station_summaries.append(
                {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "snapshot_count": 10,
                    "aggregation": "MEDIAN_OVER_10_FROZEN_QUERY_SNAPSHOTS",
                    "median": {
                        key: float(station_summary["median"][key])
                        for key in GEOMETRY_ONLY_FIELDS
                    },
                    "geometry_only": True,
                    "registration_executed": False,
                }
            )
            del target, sources
        except Exception as exc:
            failures.append(
                {
                    "scene_id": scene_id,
                    "station_id": station_id,
                    "stage": "GEOMETRY_ONLY",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            gc.collect()

    scene_summaries: list[dict[str, Any]] = []
    for scene_id in sorted({str(row["scene_id"]) for row in target_rows}):
        rows = [row for row in metrics if str(row["scene_id"]) == scene_id]
        stations = {str(row["station_id"]) for row in rows}
        if len(rows) != 30 or stations != set(STATION_IDS):
            continue
        medians = {
            key: float(np.median([float(row[key]) for row in rows]))
            for key in GEOMETRY_ONLY_FIELDS
        }
        gate = geometry_admission(
            medians["normalized_lambda_min_trans"],
            medians["condition_number_trans"],
            medians["spectral_entropy_trans"],
        )
        expected = "RICH" if scene_id.startswith("FMB1_R") else "WEAK"
        aligned = gate["final_geometry_class"] == expected
        if aligned and gate["admitted"]:
            admission_status = "GEOMETRY_ADMITTED"
            reason = None
        elif gate["final_geometry_class"] == "INTERMEDIATE":
            admission_status = "GEOMETRY_REVIEW"
            reason = "GEOMETRY_INTERMEDIATE"
        else:
            admission_status = "GEOMETRY_REJECTED"
            reason = "SEMANTIC_CANDIDATE_GEOMETRY_CLASS_MISMATCH"
        scene_summaries.append(
            {
                "scene_id": scene_id,
                "semantic_candidate_label": f"{expected}_CANDIDATE",
                "aggregation": "MEDIAN_OVER_ALL_30_NESTED_STATION_SNAPSHOTS",
                "snapshot_count": 30,
                "station_count": 3,
                **{
                    f"median_{key}": value for key, value in medians.items()
                },
                **gate,
                "candidate_class_alignment": aligned,
                "geometry_admission_status": admission_status,
                "failure_reason": reason,
                "replacement_allowed_under_preregistration": bool(
                    admission_status != "GEOMETRY_ADMITTED"
                ),
            }
        )
    payload = {
        "schema": "mid360_fmb1_geometry_only_v1",
        "metric_fields": list(GEOMETRY_ONLY_FIELDS),
        "T0": "IDENTITY_4X4",
        "geometry_only": True,
        "registration_executed": False,
        "snapshot_metrics": metrics,
        "station_summaries": station_summaries,
        "scene_summaries": scene_summaries,
        "failures": failures,
        "open3d_registration_call_count": 0,
        "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0,
        "formal_trial_count": 0,
    }
    if write_manifest:
        _write_json(runtime_dir / "geometry_only_manifest.json", payload)
    return payload


__all__ = [
    "PreRegistrationAssetError",
    "analyze_geometry_only",
    "materialize_registration_free_assets",
]
