"""Read-only acquisition ingest for the 2026-08-19 Mid-360 FMB1 bags.

This module deliberately stops at the station acquisition gate.  It contains
no target-map, geometry, backend, or registration import.  Raw bags are never
renamed, linked, copied, or written.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

from phase_a_harness.mid360_pilot.bag_reader import (
    IMU_TOPIC,
    LIDAR_TOPIC,
    PilotBagError,
    build_bag_inventory,
    iter_topic_messages,
    sha256_file,
)
from phase_a_harness.mid360_pilot.imu_audit import audit_imu_messages
from phase_a_harness.mid360_pilot.lidar_adapter import audit_lidar_messages

from .protocol import REQUIRED_POINT_FIELDS, audit_pair, evaluate_bag_audit


RAW_BAG_GLOB = "mid360_20260819_*_part*.bag"
RAW_BAG_RE = re.compile(
    r"^mid360_(20260819_\d{6})_(part1_20s|part2_15s)\.bag$"
)
ROLE_BY_PART = {"part1_20s": "MAP", "part2_15s": "QUERY"}
PART_BY_ROLE = {role: part for part, role in ROLE_BY_PART.items()}

# The order is scientific state: acquisition pair 01 through pair 18.
EXPECTED_MAPPING: tuple[dict[str, Any], ...] = (
    {"pair_index": 1, "scene_id": "FMB1_R01", "station_id": "S01", "capture_prefix": "20260819_205431"},
    {"pair_index": 2, "scene_id": "FMB1_R01", "station_id": "S02", "capture_prefix": "20260819_205707"},
    {"pair_index": 3, "scene_id": "FMB1_R01", "station_id": "S03", "capture_prefix": "20260819_205939"},
    {"pair_index": 4, "scene_id": "FMB1_R02", "station_id": "S01", "capture_prefix": "20260819_210312"},
    {"pair_index": 5, "scene_id": "FMB1_R02", "station_id": "S02", "capture_prefix": "20260819_210526"},
    {"pair_index": 6, "scene_id": "FMB1_R02", "station_id": "S03", "capture_prefix": "20260819_210826"},
    {"pair_index": 7, "scene_id": "FMB1_R03", "station_id": "S01", "capture_prefix": "20260819_211442"},
    {"pair_index": 8, "scene_id": "FMB1_R03", "station_id": "S02", "capture_prefix": "20260819_211806"},
    {"pair_index": 9, "scene_id": "FMB1_R03", "station_id": "S03", "capture_prefix": "20260819_212018"},
    {"pair_index": 10, "scene_id": "FMB1_W01", "station_id": "S01", "capture_prefix": "20260819_212448"},
    {"pair_index": 11, "scene_id": "FMB1_W01", "station_id": "S02", "capture_prefix": "20260819_212824"},
    {"pair_index": 12, "scene_id": "FMB1_W01", "station_id": "S03", "capture_prefix": "20260819_213035"},
    {"pair_index": 13, "scene_id": "FMB1_W02", "station_id": "S01", "capture_prefix": "20260819_213446"},
    {"pair_index": 14, "scene_id": "FMB1_W02", "station_id": "S02", "capture_prefix": "20260819_213744"},
    {"pair_index": 15, "scene_id": "FMB1_W02", "station_id": "S03", "capture_prefix": "20260819_214044"},
    {"pair_index": 16, "scene_id": "FMB1_W03", "station_id": "S01", "capture_prefix": "20260819_214807"},
    {"pair_index": 17, "scene_id": "FMB1_W03", "station_id": "S02", "capture_prefix": "20260819_214952"},
    {"pair_index": 18, "scene_id": "FMB1_W03", "station_id": "S03", "capture_prefix": "20260819_215146"},
)


class AcquisitionInventoryError(RuntimeError):
    """A fail-closed raw-name/count/pair/mapping inventory error."""

    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def expected_raw_filenames() -> tuple[str, ...]:
    """Return all 36 expected raw basenames in pair then MAP/QUERY order."""

    output: list[str] = []
    for row in EXPECTED_MAPPING:
        prefix = str(row["capture_prefix"])
        for role in ("MAP", "QUERY"):
            output.append(f"mid360_{prefix}_{PART_BY_ROLE[role]}.bag")
    return tuple(output)


def _canonical_filename(row: Mapping[str, Any], role: str) -> str:
    return (
        f"{row['scene_id']}_{row['station_id']}_{role}_"
        f"{row['capture_prefix']}.bag"
    )


def validate_raw_bag_names(names: Sequence[str]) -> list[dict[str, Any]]:
    """Validate a basename-only inventory and bind it to ``EXPECTED_MAPPING``.

    This helper performs no filesystem or ROS access so the inventory contract
    can be tamper-tested independently.
    """

    supplied = [str(name) for name in names]
    non_basenames = sorted(name for name in supplied if Path(name).name != name)
    duplicate_names = sorted(
        name for name, count in Counter(supplied).items() if count != 1
    )
    parsed: list[tuple[str, str, str]] = []
    malformed: list[str] = []
    for name in supplied:
        match = RAW_BAG_RE.fullmatch(name)
        if match is None:
            malformed.append(name)
            continue
        capture_prefix, part = match.groups()
        parsed.append((capture_prefix, part, name))

    grouped: dict[str, list[str]] = defaultdict(list)
    for capture_prefix, part, _ in parsed:
        grouped[capture_prefix].append(part)
    invalid_pairs = {
        prefix: sorted(parts)
        for prefix, parts in sorted(grouped.items())
        if Counter(parts) != Counter(ROLE_BY_PART.keys())
    }
    inventory_details = {
        "supplied_names": sorted(supplied),
        "supplied_file_count": len(supplied),
        "unique_file_count": len(set(supplied)),
        "parsed_file_count": len(parsed),
        "unique_capture_prefix_count": len(grouped),
        "non_basename_entries": non_basenames,
        "duplicate_names": duplicate_names,
        "malformed_names": sorted(malformed),
        "invalid_pairs": invalid_pairs,
    }
    if (
        len(supplied) != 36
        or len(set(supplied)) != 36
        or len(parsed) != 36
        or len(grouped) != 18
        or non_basenames
        or duplicate_names
        or malformed
        or invalid_pairs
    ):
        raise AcquisitionInventoryError(
            "FMB1_INPUT_INVENTORY_FAIL",
            "raw bag inventory must contain exactly 18 complete MAP/QUERY pairs",
            details=inventory_details,
        )

    expected_prefixes = [str(row["capture_prefix"]) for row in EXPECTED_MAPPING]
    actual_prefixes = sorted(grouped)
    if actual_prefixes != expected_prefixes:
        missing = sorted(set(expected_prefixes) - set(actual_prefixes))
        unexpected = sorted(set(actual_prefixes) - set(expected_prefixes))
        raise AcquisitionInventoryError(
            "USER_MAPPING_CONFLICT",
            "actual acquisition prefixes disagree with the frozen user mapping",
            details={
                **inventory_details,
                "expected_capture_prefixes": expected_prefixes,
                "actual_capture_prefixes": actual_prefixes,
                "missing_capture_prefixes": missing,
                "unexpected_capture_prefixes": unexpected,
            },
        )

    supplied_set = set(supplied)
    output: list[dict[str, Any]] = []
    for expected in EXPECTED_MAPPING:
        for role in ("MAP", "QUERY"):
            part = PART_BY_ROLE[role]
            raw_name = f"mid360_{expected['capture_prefix']}_{part}.bag"
            if raw_name not in supplied_set:  # defensive: pair checks above should catch it
                raise AcquisitionInventoryError(
                    "FMB1_INPUT_INVENTORY_FAIL",
                    f"expected raw bag is absent: {raw_name}",
                )
            output.append(
                {
                    **expected,
                    "semantic_candidate_label": (
                        "RICH_CANDIDATE"
                        if str(expected["scene_id"]).startswith("FMB1_R")
                        else "WEAK_CANDIDATE"
                    ),
                    "role": role,
                    "raw_filename": raw_name,
                    "raw_part": part,
                    "canonical_filename": _canonical_filename(expected, role),
                }
            )
    return output


def discover_raw_bags(bags_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Read the actual bag directory and return its validated 36-file binding."""

    try:
        root = bags_dir.expanduser().resolve(strict=True)
    except OSError as exc:
        raise AcquisitionInventoryError(
            "FMB1_INPUT_INVENTORY_FAIL",
            f"bags_dir cannot be resolved: {bags_dir}",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        ) from exc
    if not root.is_dir():
        raise AcquisitionInventoryError(
            "FMB1_INPUT_INVENTORY_FAIL", f"bags_dir is not a directory: {root}"
        )
    paths = sorted(root.glob(RAW_BAG_GLOB), key=lambda path: path.name)
    non_files = sorted(str(path) for path in paths if not path.is_file())
    if non_files:
        raise AcquisitionInventoryError(
            "FMB1_INPUT_INVENTORY_FAIL",
            "matching bag entries must be regular files",
            details={"non_file_entries": non_files},
        )
    binding = validate_raw_bag_names([path.name for path in paths])
    path_by_name = {path.name: path.resolve(strict=True) for path in paths}
    for row in binding:
        row["raw_absolute_path"] = str(path_by_name[str(row["raw_filename"])])
    return root, binding


def _config_mapping(config: Union[Mapping[str, Any], Path, str]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config JSON must contain an object")
    return payload


def _audit_one_bag(spec: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(spec["raw_absolute_path"]))
    inventory: dict[str, Any] | None = None
    lidar_summary: dict[str, Any] | None = None
    imu_summary: dict[str, Any] | None = None
    fallback_sha256: str | None = None
    try:
        # Each frozen reader/auditor is invoked exactly once for this bag.
        inventory = build_bag_inventory(path)
        lidar_rows, lidar_summary = audit_lidar_messages(
            iter_topic_messages(path, LIDAR_TOPIC)
        )
        _, imu_summary = audit_imu_messages(
            iter_topic_messages(path, IMU_TOPIC), config
        )
        point_count = sum(int(row["point_count"]) for row in lidar_rows)
        finite_count = sum(int(row["finite_point_count"]) for row in lidar_rows)
        finite_fraction = finite_count / point_count if point_count else 0.0
        bag_audit = evaluate_bag_audit(
            inventory,
            role=str(spec["role"]),
            lidar_summary=lidar_summary,
            imu_summary=imu_summary,
            finite_point_fraction=finite_fraction,
        )
        all_field_signatures_pass = bool(lidar_summary.get("field_signatures")) and all(
            set(str(field) for field in signature) >= set(REQUIRED_POINT_FIELDS)
            for signature in lidar_summary.get("field_signatures", [])
        )
        bag_audit["checks"]["all_pointcloud2_field_signatures"] = (
            all_field_signatures_pass
        )
        bag_audit["all_pointcloud2_field_signatures_pass"] = (
            all_field_signatures_pass
        )
        if not all_field_signatures_pass:
            bag_audit["failure_reasons"] = sorted(
                set(bag_audit.get("failure_reasons", []))
                | {"all_pointcloud2_field_signatures"}
            )
            bag_audit["REVIEW_REQUIRED"] = True
            bag_audit["ACQUISITION_AUDIT_PASS"] = False
        status = "PASS" if bag_audit["ACQUISITION_AUDIT_PASS"] else "FAIL"
        error = None
    except PilotBagError as exc:
        # build_bag_inventory authenticates successful bags.  If it fails
        # before returning, authenticate the retained raw file separately so
        # content failure never erases raw-data identity.
        if inventory is None:
            fallback_sha256 = sha256_file(path)
        bag_audit = None
        status = "FAIL"
        error = {
            "error_type": type(exc).__name__,
            "failure_reason": str(exc),
            "failure_code": "PILOT_BAG_ERROR",
        }

    stat = path.stat()
    return {
        **dict(spec),
        "bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": inventory.get("bag_sha256") if inventory else fallback_sha256,
        "inventory": inventory,
        "lidar_summary": lidar_summary,
        "imu_summary": imu_summary,
        "bag_audit": bag_audit,
        "status": status,
        "error": error,
    }


def _failed_pair_audit(map_row: Mapping[str, Any], query_row: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if map_row.get("bag_audit") is None:
        reasons.append("MAP_PILOT_BAG_ERROR")
    if query_row.get("bag_audit") is None:
        reasons.append("QUERY_PILOT_BAG_ERROR")
    return {
        "schema": "mid360_formal_batch1_pair_audit_v1",
        "actual_gap_s": None,
        "map_query_no_overlap": False,
        "actual_gap_minimum_pass": False,
        "timing_pass": False,
        "FORMAL_PAIR_VALID": False,
        "exclusion_reasons": reasons or ["PAIR_AUDIT_UNAVAILABLE"],
        "raw_data_retained": True,
    }


def _station_rows(raw_bags: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(row["scene_id"]), str(row["station_id"]), str(row["role"])): row
        for row in raw_bags
    }
    output: list[dict[str, Any]] = []
    for expected in EXPECTED_MAPPING:
        key = (str(expected["scene_id"]), str(expected["station_id"]))
        map_row = by_key[(key[0], key[1], "MAP")]
        query_row = by_key[(key[0], key[1], "QUERY")]
        if map_row.get("bag_audit") is not None and query_row.get("bag_audit") is not None:
            pair = audit_pair(map_row["bag_audit"], query_row["bag_audit"])
        else:
            pair = _failed_pair_audit(map_row, query_row)
        review = any(
            bool(row.get("bag_audit", {}).get("REVIEW_REQUIRED"))
            for row in (map_row, query_row)
            if row.get("bag_audit") is not None
        )
        station_status = (
            "ACQUISITION_PASS"
            if pair["FORMAL_PAIR_VALID"]
            else "REVIEW"
            if review
            else "ACQUISITION_FAIL"
        )
        output.append(
            {
                **expected,
                "semantic_candidate_label": (
                    "RICH_CANDIDATE"
                    if str(expected["scene_id"]).startswith("FMB1_R")
                    else "WEAK_CANDIDATE"
                ),
                "map_raw_filename": map_row["raw_filename"],
                "query_raw_filename": query_row["raw_filename"],
                "map_bag_status": map_row["status"],
                "query_bag_status": query_row["status"],
                "map_audit": map_row.get("bag_audit"),
                "query_audit": query_row.get("bag_audit"),
                "pair_audit": pair,
                "station_acquisition_status": station_status,
            }
        )
    return output


def _authenticated_mapping_rows(
    raw_bags: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "pair_index",
        "scene_id",
        "station_id",
        "semantic_candidate_label",
        "role",
        "capture_prefix",
        "raw_part",
        "raw_filename",
        "raw_absolute_path",
        "canonical_filename",
        "sha256",
        "bytes",
        "mtime",
        "mtime_utc",
        "status",
    )
    return [{field: row.get(field) for field in fields} for row in raw_bags]


def to_json_serializable(value: Any) -> Any:
    """Recursively normalize acquisition payloads for strict JSON encoding."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): to_json_serializable(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_json_serializable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [to_json_serializable(item) for item in sorted(value, key=repr)]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return to_json_serializable(item_method())
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _inventory_failure_payload(
    repository: Path,
    bags_dir: Path,
    error: AcquisitionInventoryError,
) -> dict[str, Any]:
    return {
        "schema": "mid360_formal_batch1_acquisition_ingest_v1",
        "repository": str(repository),
        "bags_dir": str(bags_dir),
        "NO_FORMAL_REGISTRATION": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "raw_bags": [],
        "mapping": [],
        "stations": [],
        "inventory_gate": {
            "status": "FAIL",
            "failure_code": error.code,
            "failure_reason": str(error),
            "details": error.details,
            "expected_file_count": 36,
            "expected_pair_count": 18,
        },
        "status": "FAIL",
    }


def run_acquisition(
    repository: Union[Path, str],
    bags_dir: Union[Path, str],
    config: Union[Mapping[str, Any], Path, str],
) -> dict[str, Any]:
    """Authenticate and audit all 36 FMB1 bags without writing artifacts."""

    repo = Path(repository).expanduser().resolve(strict=True)
    requested_bags = Path(bags_dir).expanduser()
    if not requested_bags.is_absolute():
        requested_bags = repo / requested_bags
    try:
        resolved_bags, mapping = discover_raw_bags(requested_bags)
    except AcquisitionInventoryError as exc:
        payload = _inventory_failure_payload(repo, requested_bags, exc)
        return to_json_serializable(payload)

    config_payload = _config_mapping(config)
    raw_bags = [_audit_one_bag(spec, config_payload) for spec in mapping]
    authenticated_mapping = _authenticated_mapping_rows(raw_bags)
    stations = _station_rows(raw_bags)
    passed_station_count = sum(
        row["station_acquisition_status"] == "ACQUISITION_PASS"
        for row in stations
    )
    payload = {
        "schema": "mid360_formal_batch1_acquisition_ingest_v1",
        "repository": str(repo),
        "bags_dir": str(resolved_bags),
        "NO_FORMAL_REGISTRATION": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "raw_bags": raw_bags,
        "mapping": authenticated_mapping,
        "stations": stations,
        "inventory_gate": {
            "status": "PASS",
            "failure_code": None,
            "expected_file_count": 36,
            "actual_file_count": len(raw_bags),
            "expected_pair_count": 18,
            "actual_pair_count": len(stations),
            "expected_mapping_match": True,
            "part1_role": "MAP",
            "part2_role": "QUERY",
        },
        "bag_audit_pass_count": sum(row["status"] == "PASS" for row in raw_bags),
        "station_acquisition_pass_count": passed_station_count,
        "station_count": len(stations),
        "status": "PASS" if passed_station_count == 18 else "FAIL",
    }
    normalized = to_json_serializable(payload)
    # Assert the public return value is strict-JSON compatible without writing it.
    json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    return normalized


__all__ = [
    "AcquisitionInventoryError",
    "EXPECTED_MAPPING",
    "PART_BY_ROLE",
    "RAW_BAG_GLOB",
    "RAW_BAG_RE",
    "ROLE_BY_PART",
    "discover_raw_bags",
    "expected_raw_filenames",
    "run_acquisition",
    "to_json_serializable",
    "validate_raw_bag_names",
]
