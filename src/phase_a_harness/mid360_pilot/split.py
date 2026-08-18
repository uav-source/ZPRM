"""Frozen relative-time split and deterministic query selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import PILOT_FLAGS
from .bag_reader import PilotBagError


def build_split_contract(
    lidar_timestamps: Sequence[float], config: Mapping[str, Any]
) -> dict[str, Any]:
    if not lidar_timestamps:
        raise PilotBagError("no LiDAR timestamps are available")
    timestamps = [float(value) for value in lidar_timestamps]
    if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise PilotBagError("LiDAR header timestamps are not monotonic")
    section = config["split_seconds_relative_to_first_lidar_header_stamp"]
    required_duration = float(section["minimum_lidar_duration"])
    duration = timestamps[-1] - timestamps[0]
    if duration < required_duration:
        raise PilotBagError(
            f"PILOT_SPLIT_NOT_APPLICABLE: LiDAR duration {duration:.9f}s < {required_duration:.9f}s"
        )
    map_start, map_end = map(float, section["map"])
    guard_start, guard_end = map(float, section["guard"])
    query_start, query_end = map(float, section["query"])
    if not (0.0 <= map_start < map_end == guard_start < guard_end == query_start < query_end):
        raise PilotBagError("pilot split configuration is invalid")
    return {
        "schema": "mid360_single_bag_pilot_split_contract_v1",
        **PILOT_FLAGS,
        "same_bag": True,
        "independent_time_segments": True,
        "independent_acquisition": False,
        "formal_use_forbidden": True,
        "status": "PASS",
        "time_origin": "FIRST_LIDAR_HEADER_TIMESTAMP",
        "time_origin_timestamp": timestamps[0],
        "actual_lidar_duration_seconds": duration,
        "minimum_required_lidar_duration_seconds": required_duration,
        "interval_semantics": "LEFT_CLOSED_RIGHT_OPEN",
        "map_interval_relative_seconds": [map_start, map_end],
        "guard_gap_relative_seconds": [guard_start, guard_end],
        "query_interval_relative_seconds": [query_start, query_end],
    }


def partition_lidar_frames(
    frames: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    origin = float(contract["time_origin_timestamp"])
    map_start, map_end = map(float, contract["map_interval_relative_seconds"])
    guard_start, guard_end = map(float, contract["guard_gap_relative_seconds"])
    query_start, query_end = map(float, contract["query_interval_relative_seconds"])
    output = {"map": [], "guard": [], "query": [], "discarded": []}
    for source in frames:
        row = dict(source)
        relative = float(row["timestamp"]) - origin
        row["relative_time_seconds"] = relative
        if map_start <= relative < map_end:
            output["map"].append(row)
        elif guard_start <= relative < guard_end:
            output["guard"].append(row)
        elif query_start <= relative < query_end:
            output["query"].append(row)
        else:
            output["discarded"].append(row)
    return output


def select_query_frames(
    candidates: Sequence[Mapping[str, Any]], quantiles: Sequence[float]
) -> list[dict[str, Any]]:
    """Select nearest sequence ranks, resolving reuse by distance then earlier time."""

    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (float(row["timestamp"]), int(row["frame_index"])),
    )
    if len(ordered) < len(quantiles):
        raise PilotBagError(
            f"query candidate count {len(ordered)} is smaller than {len(quantiles)}"
        )
    used: set[int] = set()
    selected: list[dict[str, Any]] = []
    for selection_index, quantile in enumerate(quantiles):
        q = float(quantile)
        if not 0.0 <= q <= 1.0:
            raise PilotBagError(f"invalid query quantile: {q}")
        target_rank = q * (len(ordered) - 1)
        rank_candidates = sorted(
            range(len(ordered)), key=lambda rank: (abs(rank - target_rank), rank)
        )
        chosen_rank = next(rank for rank in rank_candidates if rank not in used)
        used.add(chosen_rank)
        row = dict(ordered[chosen_rank])
        row.update(
            {
                "selection_index": selection_index,
                "quantile": q,
                "target_sequence_rank": target_rank,
                "selected_sequence_rank": chosen_rank,
            }
        )
        selected.append(row)
    if len(selected) != 10 or len({row["frame_index"] for row in selected}) != 10:
        raise PilotBagError("deterministic pilot query selection did not yield 10 unique frames")
    return selected


def build_lineage(
    partition: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_queries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def indexes(rows: Sequence[Mapping[str, Any]]) -> list[int]:
        return [int(row["frame_index"]) for row in rows]

    def timestamps(rows: Sequence[Mapping[str, Any]]) -> list[float]:
        return [float(row["timestamp"]) for row in rows]

    map_indexes = indexes(partition["map"])
    candidate_indexes = indexes(partition["query"])
    selected_indexes = indexes(selected_queries)
    guard_indexes = indexes(partition["guard"])
    map_timestamps = timestamps(partition["map"])
    candidate_timestamps = timestamps(partition["query"])
    selected_timestamps = timestamps(selected_queries)
    index_overlap = sorted(set(map_indexes) & set(candidate_indexes))
    timestamp_overlap = sorted(set(map_timestamps) & set(candidate_timestamps))
    guard_used = bool(
        set(guard_indexes) & (set(map_indexes) | set(candidate_indexes) | set(selected_indexes))
    )
    status = "PASS" if not index_overlap and not timestamp_overlap and not guard_used else "FAIL"
    return {
        "schema": "mid360_pilot_map_query_lineage_v1",
        **PILOT_FLAGS,
        "same_bag": True,
        "map_frame_indexes": map_indexes,
        "query_candidate_frame_indexes": candidate_indexes,
        "selected_query_frame_indexes": selected_indexes,
        "guard_frame_indexes": guard_indexes,
        "map_timestamps": map_timestamps,
        "query_candidate_timestamps": candidate_timestamps,
        "selected_query_timestamps": selected_timestamps,
        "map_query_message_index_intersection": index_overlap,
        "map_query_timestamp_intersection": timestamp_overlap,
        "guard_gap_used": guard_used,
        "map_query_strictly_disjoint": status == "PASS",
        "status": status,
    }


__all__ = [
    "build_lineage",
    "build_split_contract",
    "partition_lidar_frames",
    "select_query_frames",
]
