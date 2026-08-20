"""Strict, registration-free ingest contract for the FMB1 W04 replacement.

The six raw bags are discovered from the real repository tree and are never
renamed, copied, linked, or modified.  Bag, LiDAR, IMU, and pair decisions are
delegated to the already-frozen Formal Batch-1 acquisition implementation.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

from .preregistration_acquisition import _audit_one_bag, to_json_serializable
from .protocol import audit_pair


W04_RAW_GLOB = "mid360_20260820_08*_part*.bag"
W04_RAW_RE = re.compile(
    r"^mid360_(20260820_\d{6})_(part1_20s|part2_15s)\.bag$"
)
ROLE_BY_PART = {"part1_20s": "MAP", "part2_15s": "QUERY"}
PART_BY_ROLE = {role: part for part, role in ROLE_BY_PART.items()}
W04_EXPECTED_MAPPING: tuple[dict[str, Any], ...] = (
    {
        "pair_index": 1,
        "scene_id": "FMB1_W04",
        "station_id": "S01",
        "capture_prefix": "20260820_081749",
    },
    {
        "pair_index": 2,
        "scene_id": "FMB1_W04",
        "station_id": "S02",
        "capture_prefix": "20260820_081954",
    },
    {
        "pair_index": 3,
        "scene_id": "FMB1_W04",
        "station_id": "S03",
        "capture_prefix": "20260820_082207",
    },
)


class W04InventoryError(RuntimeError):
    """A strict W04 raw inventory or mapping gate failed."""

    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def expected_w04_raw_filenames() -> tuple[str, ...]:
    output: list[str] = []
    for row in W04_EXPECTED_MAPPING:
        for role in ("MAP", "QUERY"):
            output.append(
                f"mid360_{row['capture_prefix']}_{PART_BY_ROLE[role]}.bag"
            )
    return tuple(output)


def validate_w04_raw_names(names: Sequence[str]) -> list[dict[str, Any]]:
    """Validate the exact six basenames and freeze their scientific mapping."""

    supplied = [str(name) for name in names]
    parsed: list[tuple[str, str, str]] = []
    malformed: list[str] = []
    for name in supplied:
        match = W04_RAW_RE.fullmatch(name)
        if match is None or Path(name).name != name:
            malformed.append(name)
        else:
            parsed.append((match.group(1), match.group(2), name))
    grouped: dict[str, list[str]] = defaultdict(list)
    for prefix, part, _ in parsed:
        grouped[prefix].append(part)
    duplicate_names = sorted(
        name for name, count in Counter(supplied).items() if count != 1
    )
    invalid_pairs = {
        prefix: sorted(parts)
        for prefix, parts in sorted(grouped.items())
        if Counter(parts) != Counter(ROLE_BY_PART.keys())
    }
    details = {
        "supplied_names": sorted(supplied),
        "supplied_file_count": len(supplied),
        "unique_file_count": len(set(supplied)),
        "parsed_file_count": len(parsed),
        "unique_capture_prefix_count": len(grouped),
        "duplicate_names": duplicate_names,
        "malformed_names": sorted(malformed),
        "invalid_pairs": invalid_pairs,
    }
    if (
        len(supplied) != 6
        or len(set(supplied)) != 6
        or len(parsed) != 6
        or len(grouped) != 3
        or duplicate_names
        or malformed
        or invalid_pairs
    ):
        raise W04InventoryError(
            "W04_INPUT_INVENTORY_FAIL",
            "W04 inventory must contain exactly three complete MAP/QUERY pairs",
            details=details,
        )
    expected_prefixes = [
        str(row["capture_prefix"]) for row in W04_EXPECTED_MAPPING
    ]
    actual_prefixes = sorted(grouped)
    if actual_prefixes != expected_prefixes:
        raise W04InventoryError(
            "W04_USER_MAPPING_CONFLICT",
            "actual W04 prefixes disagree with the frozen mapping",
            details={
                **details,
                "expected_capture_prefixes": expected_prefixes,
                "actual_capture_prefixes": actual_prefixes,
            },
        )
    supplied_set = set(supplied)
    output: list[dict[str, Any]] = []
    for expected in W04_EXPECTED_MAPPING:
        for role in ("MAP", "QUERY"):
            part = PART_BY_ROLE[role]
            raw_name = f"mid360_{expected['capture_prefix']}_{part}.bag"
            if raw_name not in supplied_set:
                raise W04InventoryError(
                    "W04_INPUT_INVENTORY_FAIL", f"missing expected bag: {raw_name}"
                )
            output.append(
                {
                    **expected,
                    "semantic_candidate_label": "WEAK_CANDIDATE",
                    "role": role,
                    "raw_part": part,
                    "raw_filename": raw_name,
                    "canonical_filename": (
                        f"FMB1_W04_{expected['station_id']}_{role}_"
                        f"{expected['capture_prefix']}.bag"
                    ),
                }
            )
    return output


def discover_w04_raw_bags(repository: Union[Path, str]) -> list[dict[str, Any]]:
    """Search the actual repository tree and bind the exact six W04 files."""

    root = Path(repository).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise W04InventoryError(
            "W04_INPUT_INVENTORY_FAIL", f"repository is not a directory: {root}"
        )
    paths = sorted(root.rglob(W04_RAW_GLOB), key=lambda value: str(value))
    unsafe = [str(path) for path in paths if not path.is_file() or path.is_symlink()]
    if unsafe:
        raise W04InventoryError(
            "W04_INPUT_INVENTORY_FAIL",
            "matching W04 entries must be regular non-symlink files",
            details={"unsafe_entries": unsafe},
        )
    binding = validate_w04_raw_names([path.name for path in paths])
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_name[path.name].append(path.resolve(strict=True))
    duplicate_locations = {
        name: [str(path) for path in locations]
        for name, locations in by_name.items()
        if len(locations) != 1
    }
    if duplicate_locations:
        raise W04InventoryError(
            "W04_INPUT_INVENTORY_FAIL",
            "a W04 raw basename exists at multiple locations",
            details={"duplicate_locations": duplicate_locations},
        )
    for row in binding:
        row["raw_absolute_path"] = str(by_name[str(row["raw_filename"])][0])
    return binding


def _failed_pair(map_row: Mapping[str, Any], query_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "mid360_formal_batch1_pair_audit_v1",
        "actual_gap_s": None,
        "map_query_no_overlap": False,
        "actual_gap_minimum_pass": False,
        "timing_pass": False,
        "FORMAL_PAIR_VALID": False,
        "exclusion_reasons": [
            label
            for label, row in (("MAP", map_row), ("QUERY", query_row))
            if row.get("bag_audit") is None
        ]
        or ["PAIR_AUDIT_UNAVAILABLE"],
        "raw_data_retained": True,
    }


def _station_rows(raw_bags: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(row["station_id"]), str(row["role"])): row for row in raw_bags
    }
    output: list[dict[str, Any]] = []
    for expected in W04_EXPECTED_MAPPING:
        station_id = str(expected["station_id"])
        map_row = by_key[(station_id, "MAP")]
        query_row = by_key[(station_id, "QUERY")]
        if map_row.get("bag_audit") is not None and query_row.get("bag_audit") is not None:
            pair = audit_pair(map_row["bag_audit"], query_row["bag_audit"])
        else:
            pair = _failed_pair(map_row, query_row)
        review = any(
            bool(row.get("bag_audit", {}).get("REVIEW_REQUIRED"))
            for row in (map_row, query_row)
            if row.get("bag_audit") is not None
        )
        status = (
            "ACQUISITION_PASS"
            if pair["FORMAL_PAIR_VALID"]
            else "REVIEW"
            if review
            else "ACQUISITION_FAIL"
        )
        output.append(
            {
                **expected,
                "semantic_candidate_label": "WEAK_CANDIDATE",
                "map_raw_filename": map_row["raw_filename"],
                "query_raw_filename": query_row["raw_filename"],
                "map_bag_status": map_row["status"],
                "query_bag_status": query_row["status"],
                "map_audit": map_row.get("bag_audit"),
                "query_audit": query_row.get("bag_audit"),
                "pair_audit": pair,
                "station_acquisition_status": status,
            }
        )
    return output


def run_w04_acquisition(
    repository: Union[Path, str], config: Union[Mapping[str, Any], Path, str]
) -> dict[str, Any]:
    """Authenticate and audit the exact six W04 bags without writing files."""

    root = Path(repository).expanduser().resolve(strict=True)
    try:
        mapping = discover_w04_raw_bags(root)
    except W04InventoryError as exc:
        return {
            "schema": "mid360_fmb1_w04_acquisition_v1",
            "repository": str(root),
            "W04_INPUT_INVENTORY_FAIL": True,
            "W04_ACQUISITION_PASS": False,
            "W04_GEOMETRY_ADMISSION_NOT_RUN": True,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "raw_bags": [],
            "mapping": [],
            "stations": [],
            "inventory_gate": {
                "status": "FAIL",
                "failure_code": exc.code,
                "failure_reason": str(exc),
                "details": exc.details,
            },
            "status": "FAIL",
        }
    if isinstance(config, Mapping):
        config_payload = dict(config)
    else:
        config_path = Path(config).expanduser().resolve(strict=True)
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise TypeError("config JSON must contain an object")
    raw_bags = [_audit_one_bag(row, config_payload) for row in mapping]
    stations = _station_rows(raw_bags)
    passed = all(
        row["station_acquisition_status"] == "ACQUISITION_PASS"
        for row in stations
    ) and len(stations) == 3
    mapping_fields = (
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
    payload = {
        "schema": "mid360_fmb1_w04_acquisition_v1",
        "repository": str(root),
        "bags_dir": str((root / "bags").resolve(strict=True)),
        "scene_id": "FMB1_W04",
        "semantic_candidate_label": "WEAK_CANDIDATE",
        "W04_INPUT_INVENTORY_FAIL": False,
        "W04_ACQUISITION_PASS": passed,
        "W04_GEOMETRY_ADMISSION_NOT_RUN": not passed,
        "NO_FORMAL_REGISTRATION": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "raw_bags": raw_bags,
        "mapping": [
            {field: row.get(field) for field in mapping_fields} for row in raw_bags
        ],
        "stations": stations,
        "inventory_gate": {
            "status": "PASS",
            "actual_file_count": len(raw_bags),
            "actual_pair_count": len(stations),
            "expected_file_count": 6,
            "expected_pair_count": 3,
            "expected_mapping_match": True,
            "part1_role": "MAP",
            "part2_role": "QUERY",
        },
        "bag_audit_pass_count": sum(row["status"] == "PASS" for row in raw_bags),
        "station_acquisition_pass_count": sum(
            row["station_acquisition_status"] == "ACQUISITION_PASS"
            for row in stations
        ),
        "raw_bag_count": len(raw_bags),
        "station_count": len(stations),
        "status": "PASS" if passed else "FAIL",
    }
    normalized = to_json_serializable(payload)
    json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    return normalized


__all__ = [
    "W04_EXPECTED_MAPPING",
    "W04InventoryError",
    "discover_w04_raw_bags",
    "expected_w04_raw_filenames",
    "run_w04_acquisition",
    "validate_w04_raw_names",
]
