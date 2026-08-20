"""Independent, registration-free verifier for the FMB1 ingest manifest.

The verifier deliberately does not import an orchestration/producer module.  It
accepts one JSON-native bundle with the following canonical top-level lists::

    scenes, stations, bags, station_acquisition_audits, targets,
    snapshots, geometry_metrics, geometry_scenes, canonical_inputs

and the mappings ``no_icp_attestation`` and ``readiness``.  For convenience a
list may also be wrapped in a mapping under its own plural noun (for example
``{"scene_registry": {"scenes": [...]}}``); the aliases accepted by
``_rows`` below are intentionally narrow and deterministic.

This module validates structure and redundant manifest bindings.  It never
opens a ROS bag through ROS, constructs a point cloud, imports Open3D/PCL, or
runs registration.  With ``verify_files=True`` it additionally re-hashes the
declared raw bag, target ``.npy``, and source ``.npy`` files and verifies their
declared byte sizes.  It does not rebuild point clouds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .protocol import (
    BACKEND_CONTRACT_SHA256,
    IMU_RATE_RANGE_HZ,
    INITIAL_SCENE_IDS,
    LIDAR_RATE_RANGE_HZ,
    MAP_MIN_DURATION_S,
    MINIMUM_MAP_QUERY_GAP_S,
    QUERY_MIN_DURATION_S,
    QUERY_QUANTILES,
    REQUIRED_FRAME_ID,
    REQUIRED_POINT_FIELDS,
    REQUIRED_TOPICS,
    STATION_IDS,
    FormalBatchError,
    geometry_class,
    sha256_file,
)


MANIFEST_SCHEMA = "mid360_fmb1_preregistration_manifest_v1"
COMPLETE_FROZEN_MANIFEST_SCHEMA = "mid360_fmb1_complete_frozen_manifest_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_BAG_RE = re.compile(
    r"^mid360_(\d{8}_\d{6})_(part1_20s|part2_15s)\.bag$"
)

CAPTURE_PREFIXES: tuple[str, ...] = (
    "20260819_205431",
    "20260819_205707",
    "20260819_205939",
    "20260819_210312",
    "20260819_210526",
    "20260819_210826",
    "20260819_211442",
    "20260819_211806",
    "20260819_212018",
    "20260819_212448",
    "20260819_212824",
    "20260819_213035",
    "20260819_213446",
    "20260819_213744",
    "20260819_214044",
    "20260819_214807",
    "20260819_214952",
    "20260819_215146",
)

EXPECTED_STATION_PREFIX: dict[tuple[str, str], str] = {
    (scene_id, station_id): CAPTURE_PREFIXES[index]
    for index, (scene_id, station_id) in enumerate(
        (scene_id, station_id)
        for scene_id in INITIAL_SCENE_IDS
        for station_id in STATION_IDS
    )
}

GEOMETRY_ONLY_FIELDS: tuple[str, ...] = (
    "initial_correspondence_count",
    "initial_valid_normal_correspondence_count",
    "lambda_min_trans",
    "lambda_mid_trans",
    "lambda_max_trans",
    "normalized_lambda_min_trans",
    "normalized_lambda_mid_trans",
    "normalized_lambda_max_trans",
    "condition_number_trans",
    "spectral_entropy_trans",
)

GEOMETRY_METADATA_FIELDS = frozenset(
    {
        "scene_id",
        "station_id",
        "snapshot_id",
        "selection_index",
        "frame_index",
        "query_frame_index",
        "timestamp",
        "query_timestamp",
        "query_finite_point_count",
        "source_point_count",
        "geometry_only",
    }
)

ALLOWED_REGISTRATION_ATTESTATION_FIELDS = frozenset(
    {
        "FORMAL_REGISTRATION_AUTHORIZED",
        "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION",
        "NO_FORMAL_REGISTRATION",
        "actual_registration_trials",
        "actual_trials",
        "registration_execution_count",
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "formal_trial_count",
        "registration_called",
        "registration_executed",
        "registration_derived_fields_forbidden",
    }
)

FORBIDDEN_RESULT_KEY_FRAGMENTS = (
    "t_est",
    "estimated_transform",
    "translation_error",
    "rotation_error",
    "final_residual",
    "turnover",
    "fitness",
    "solver",
    "backend_result",
    "registration_result",
    "icp_result",
)


def _fail(message: str) -> None:
    raise FormalBatchError(f"FMB1_MANIFEST_INVALID: {message}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    return value


def _rows(
    payload: Mapping[str, Any],
    canonical: str,
    *aliases: str,
) -> list[Mapping[str, Any]]:
    """Return a required list, accepting a narrow wrapper/alias vocabulary."""

    for key in (canonical, *aliases):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, Mapping):
            candidate_keys = (canonical, canonical.rstrip("s"), "rows")
            found = [candidate for candidate in candidate_keys if candidate in value]
            if len(found) != 1:
                _fail(f"{key} wrapper must contain exactly one {canonical} list")
            value = value[found[0]]
        if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
            _fail(f"{key} must be a list of mappings")
        return list(value)
    _fail(f"required list is missing: {canonical}")
    raise AssertionError("unreachable")


def _one_of(row: Mapping[str, Any], names: Sequence[str], label: str) -> Any:
    present = [name for name in names if name in row]
    if not present:
        _fail(f"{label} is missing (accepted keys: {', '.join(names)})")
    values = [row[name] for name in present]
    if any(value != values[0] for value in values[1:]):
        _fail(f"conflicting aliases for {label}: {present}")
    return values[0]


def _text(row: Mapping[str, Any], names: Sequence[str], label: str) -> str:
    value = _one_of(row, names, label)
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _key(row: Mapping[str, Any], label: str) -> tuple[str, str]:
    scene_id = _text(row, ("scene_id",), f"{label}.scene_id")
    station_id = _text(row, ("station_id",), f"{label}.station_id")
    key = (scene_id, station_id)
    if key not in EXPECTED_STATION_PREFIX:
        _fail(f"unexpected {label} scene/station: {key}")
    return key


def _exact_unique_keys(
    rows: Sequence[Mapping[str, Any]],
    expected: set[Any],
    key_function: Any,
    label: str,
) -> dict[Any, Mapping[str, Any]]:
    keyed: dict[Any, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        key = key_function(row, f"{label}[{index}]")
        if key in keyed:
            _fail(f"duplicate {label} key: {key}")
        keyed[key] = row
    actual = set(keyed)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(f"{label} keys differ; missing={missing}, extra={extra}")
    return keyed


def _recursive_result_firewall(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if key not in ALLOWED_REGISTRATION_ATTESTATION_FIELDS and any(
                fragment in lowered for fragment in FORBIDDEN_RESULT_KEY_FRAGMENTS
            ):
                _fail(f"registration-derived/result field is forbidden: {path}.{key}")
            _recursive_result_firewall(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _recursive_result_firewall(child, f"{path}[{index}]")


def _role_mapping(row: Mapping[str, Any], role: str, label: str) -> Mapping[str, Any]:
    lowered = role.lower()
    for key in (lowered, f"{lowered}_audit"):
        if key in row:
            return _mapping(row[key], f"{label}.{key}")
    # Flat station-audit CSV/JSON rows are also accepted.
    return row


def _role_value(
    station_row: Mapping[str, Any],
    role: str,
    names: Sequence[str],
    label: str,
) -> Any:
    nested = _role_mapping(station_row, role, label)
    lowered = role.lower()
    candidates: list[tuple[Mapping[str, Any], str]] = []
    for name in names:
        candidates.append((nested, name))
        candidates.append((station_row, f"{lowered}_{name}"))
    present = [(source, name) for source, name in candidates if name in source]
    if not present:
        _fail(f"{label}.{lowered}.{names[0]} is missing")
    values = [source[name] for source, name in present]
    if any(value != values[0] for value in values[1:]):
        _fail(f"conflicting values for {label}.{lowered}.{names[0]}")
    return values[0]


def _role_rate(station_row: Mapping[str, Any], role: str, topic: str, label: str) -> float:
    nested = _role_mapping(station_row, role, label)
    frequencies = nested.get("frequencies_hz")
    if isinstance(frequencies, Mapping) and topic in frequencies:
        return _finite(frequencies[topic], f"{label}.{role.lower()}.{topic}.rate")
    name = "lidar_rate_hz" if topic == "/livox/lidar" else "imu_rate_hz"
    return _finite(
        _role_value(station_row, role, (name, name.replace("_rate", "")), label),
        f"{label}.{role.lower()}.{name}",
    )


def _role_count(station_row: Mapping[str, Any], role: str, topic: str, label: str) -> int:
    nested = _role_mapping(station_row, role, label)
    counts = nested.get("message_counts")
    if isinstance(counts, Mapping) and topic in counts:
        return _integer(counts[topic], f"{label}.{role.lower()}.{topic}.count", minimum=1)
    name = "lidar_count" if topic == "/livox/lidar" else "imu_count"
    return _integer(
        _role_value(station_row, role, (name, name.replace("_count", "_message_count")), label),
        f"{label}.{role.lower()}.{name}",
        minimum=1,
    )


def _role_topics(station_row: Mapping[str, Any], role: str, label: str) -> Mapping[str, Any]:
    nested = _role_mapping(station_row, role, label)
    topic_types = nested.get("topic_types")
    if not isinstance(topic_types, Mapping):
        topic_types = station_row.get(f"{role.lower()}_topic_types")
    if not isinstance(topic_types, Mapping):
        _fail(f"{label}.{role.lower()}.topic_types is missing")
    return topic_types


def _role_frames(station_row: Mapping[str, Any], role: str, label: str) -> set[str]:
    value = _role_value(station_row, role, ("frame_ids", "frame_id"), label)
    if isinstance(value, str):
        return {part for part in value.split(";") if part}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(part) for part in value}
    _fail(f"{label}.{role.lower()}.frame_ids must be a string or list")
    raise AssertionError("unreachable")


def _role_fields(station_row: Mapping[str, Any], role: str, label: str) -> set[str]:
    value = _role_value(
        station_row, role, ("pointcloud2_fields", "point_fields"), label
    )
    if isinstance(value, str):
        return {part for part in re.split(r"[;,]", value) if part}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(part) for part in value}
    _fail(f"{label}.{role.lower()}.pointcloud2_fields must be a string or list")
    raise AssertionError("unreachable")


def _pair_value(row: Mapping[str, Any], names: Sequence[str], label: str) -> Any:
    pair = row.get("pair") or row.get("pair_audit")
    mappings = [_mapping(pair, f"{label}.pair")] if pair is not None else []
    mappings.append(row)
    present: list[Any] = []
    for source in mappings:
        for name in names:
            if name in source:
                present.append(source[name])
    if not present:
        _fail(f"{label}.{names[0]} is missing")
    if any(value != present[0] for value in present[1:]):
        _fail(f"conflicting aliases for {label}.{names[0]}")
    return present[0]


def _verify_declared_file(
    repository: Path,
    row: Mapping[str, Any],
    *,
    path_names: Sequence[str],
    sha_names: Sequence[str],
    size_names: Sequence[str],
    label: str,
) -> Path:
    declared_path = Path(_text(row, path_names, f"{label}.path"))
    if not declared_path.is_absolute():
        declared_path = repository / declared_path
    try:
        resolved = declared_path.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label} file is missing/unreadable: {declared_path}: {exc}")
    try:
        resolved.relative_to(repository)
    except ValueError:
        _fail(f"{label} file escapes repository: {resolved}")
    if not resolved.is_file():
        _fail(f"{label} is not a regular file: {resolved}")
    expected_size = _integer(_one_of(row, size_names, f"{label}.size"), f"{label}.size", minimum=1)
    if resolved.stat().st_size != expected_size:
        _fail(f"{label} size mismatch")
    expected_sha = _sha(_one_of(row, sha_names, f"{label}.sha256"), f"{label}.sha256")
    if sha256_file(resolved) != expected_sha:
        _fail(f"{label} SHA256 mismatch")
    return resolved


def _identity(value: Any) -> bool:
    if isinstance(value, str):
        if value in ("IDENTITY", "Identity", "IDENTITY_4X4"):
            return True
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return False
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    rows = list(value)
    if len(rows) == 16:
        rows = [rows[index : index + 4] for index in range(0, 16, 4)]
    expected = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    try:
        return all(float(rows[i][j]) == expected[i][j] for i in range(4) for j in range(4))
    except (IndexError, TypeError, ValueError):
        return False


def _canonical_payload_sha(payload: Mapping[str, Any]) -> str:
    material = copy.deepcopy(dict(payload))
    material.pop("manifest_sha256", None)
    encoded = json.dumps(
        material, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest_payload(
    payload: Mapping[str, Any],
    repository: Path,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Validate a complete FMB1 pre-registration manifest bundle.

    Returns a JSON-native checklist on success.  Any missing, inconsistent, or
    tampered value raises :class:`FormalBatchError`.  ``verify_files`` only adds
    byte-level authentication; all structural and redundant-binding checks run
    in both modes.
    """

    payload = _mapping(payload, "payload")
    try:
        repository = Path(repository).resolve(strict=True)
    except OSError as exc:
        _fail(f"repository is missing: {exc}")
    if not repository.is_dir():
        _fail("repository must be a directory")

    schema = payload.get("schema", MANIFEST_SCHEMA)
    if schema not in {MANIFEST_SCHEMA, COMPLETE_FROZEN_MANIFEST_SCHEMA}:
        _fail(f"unsupported schema: {schema}")
    _recursive_result_firewall(payload)

    if "manifest_sha256" in payload:
        declared_manifest_sha = _sha(payload["manifest_sha256"], "manifest_sha256")
        if declared_manifest_sha != _canonical_payload_sha(payload):
            _fail("manifest_sha256 mismatch")

    declared_backend_sha = _sha(
        payload.get("backend_parameter_contract_sha256"),
        "backend_parameter_contract_sha256",
    )
    if declared_backend_sha != BACKEND_CONTRACT_SHA256:
        _fail("backend parameter contract SHA differs from frozen constant")
    backend_path = repository / "frozen_assets/backend_parameter_contract.json"
    if not backend_path.is_file() or sha256_file(backend_path) != BACKEND_CONTRACT_SHA256:
        _fail("repository backend parameter contract is absent or changed")

    expected_scene_ids = set(INITIAL_SCENE_IDS)
    expected_station_keys = set(EXPECTED_STATION_PREFIX)

    scene_rows = _rows(
        payload,
        "scenes",
        "scene_registry",
        "scene_registry_frozen",
        "geometry_scenes",
    )
    if len(scene_rows) != 6:
        _fail(f"scene count must be 6, got {len(scene_rows)}")
    scenes = _exact_unique_keys(
        scene_rows,
        expected_scene_ids,
        lambda row, label: _text(row, ("scene_id",), f"{label}.scene_id"),
        "scenes",
    )
    for scene_id, row in scenes.items():
        expected_candidate = "RICH_CANDIDATE" if scene_id.startswith("FMB1_R") else "WEAK_CANDIDATE"
        candidate = _text(
            row,
            ("semantic_candidate_label", "candidate_label"),
            f"scene {scene_id} candidate label",
        )
        if candidate != expected_candidate:
            _fail(f"scene {scene_id} candidate label changed")

    station_rows = _rows(payload, "stations", "station_registry", "station_registry_frozen")
    if len(station_rows) != 18:
        _fail(f"station count must be 18, got {len(station_rows)}")
    _exact_unique_keys(station_rows, expected_station_keys, _key, "stations")
    if any(
        {station for scene, station in expected_station_keys if scene == scene_id}
        != set(STATION_IDS)
        for scene_id in INITIAL_SCENE_IDS
    ):
        _fail("every scene must contain exactly S01/S02/S03")

    bag_rows = _rows(payload, "bags", "raw_bag_inventory", "mapping", "raw_bags")
    if len(bag_rows) != 36:
        _fail(f"bag count must be 36, got {len(bag_rows)}")
    expected_bag_keys = {(scene, station, role) for scene, station in expected_station_keys for role in ("MAP", "QUERY")}

    def bag_key(row: Mapping[str, Any], label: str) -> tuple[str, str, str]:
        scene_id, station_id = _key(row, label)
        role = _text(row, ("role",), f"{label}.role")
        if role not in {"MAP", "QUERY"}:
            _fail(f"{label}.role must be MAP or QUERY")
        return scene_id, station_id, role

    bags = _exact_unique_keys(bag_rows, expected_bag_keys, bag_key, "bags")
    raw_names: set[str] = set()
    raw_paths: set[str] = set()
    canonical_names: set[str] = set()
    for key, row in bags.items():
        scene_id, station_id, role = key
        prefix = EXPECTED_STATION_PREFIX[(scene_id, station_id)]
        expected_part = "part1_20s" if role == "MAP" else "part2_15s"
        raw_name = _text(row, ("raw_filename", "raw_name"), f"bag {key} raw filename")
        match = RAW_BAG_RE.fullmatch(raw_name)
        if match is None or match.groups() != (prefix, expected_part):
            _fail(f"bag {key} raw filename/capture mapping changed: {raw_name}")
        capture_prefix = _text(
            row,
            ("capture_prefix", "capture_timestamp"),
            f"bag {key} capture prefix",
        )
        if capture_prefix != prefix:
            _fail(f"bag {key} capture prefix changed")
        canonical_name = _text(
            row, ("canonical_filename", "canonical_name"), f"bag {key} canonical filename"
        )
        if canonical_name != f"{scene_id}_{station_id}_{role}_{prefix}.bag":
            _fail(f"bag {key} canonical filename changed")
        if raw_name in raw_names or canonical_name in canonical_names:
            _fail("raw or canonical bag filename is reused")
        raw_path = _text(
            row,
            ("raw_absolute_path", "raw_path", "bag_path"),
            f"bag {key} raw path",
        )
        if Path(raw_path).name != raw_name:
            _fail(f"bag {key} raw path basename differs from raw filename")
        if raw_path in raw_paths:
            _fail(f"raw bag path is reused: {raw_path}")
        raw_names.add(raw_name)
        raw_paths.add(raw_path)
        canonical_names.add(canonical_name)
        _sha(_one_of(row, ("sha256", "SHA256"), f"bag {key} sha256"), f"bag {key} sha256")
        _integer(
            _one_of(row, ("bytes", "file_size", "file_size_bytes"), f"bag {key} size"),
            f"bag {key} size",
            minimum=1,
        )
        if verify_files:
            resolved = _verify_declared_file(
                repository,
                row,
                path_names=("raw_absolute_path", "raw_path", "bag_path"),
                sha_names=("sha256", "SHA256"),
                size_names=("bytes", "file_size", "file_size_bytes"),
                label=f"raw bag {key}",
            )
            if resolved.name != raw_name:
                _fail(f"raw bag {key} path basename differs from manifest")

    # The canonical bundle keeps compact mapping rows and may additionally
    # retain the producer's detailed, read-only bag audits.  Use those details
    # only to authenticate the flat station rows; never trust them as a second
    # mapping authority.
    raw_detail_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    raw_detail_value = payload.get("raw_bags")
    if isinstance(raw_detail_value, list):
        if len(raw_detail_value) != 36 or any(
            not isinstance(row, Mapping) for row in raw_detail_value
        ):
            _fail("raw_bags details must contain exactly 36 mappings")
        for index, row in enumerate(raw_detail_value):
            detail_key = bag_key(row, f"raw_bags[{index}]")
            if detail_key in raw_detail_by_key:
                _fail(f"duplicate raw_bags detail key: {detail_key}")
            raw_detail_by_key[detail_key] = row
            detail_sha = row.get("sha256", row.get("SHA256"))
            if detail_sha is None and isinstance(row.get("inventory"), Mapping):
                detail_sha = row["inventory"].get("bag_sha256")
            if _sha(detail_sha, f"raw_bags detail {detail_key} SHA") != _sha(
                _one_of(
                    bags[detail_key],
                    ("sha256", "SHA256"),
                    f"mapping bag {detail_key} SHA",
                ),
                f"mapping bag {detail_key} SHA",
            ):
                _fail(f"raw_bags detail SHA disagrees with mapping for {detail_key}")
        if set(raw_detail_by_key) != expected_bag_keys:
            _fail("raw_bags detail keys differ from the frozen 36-bag mapping")

    def enriched_acquisition_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
        key = _key(row, "station acquisition")
        if "map" in row or "map_audit" in row:
            return row
        if not raw_detail_by_key:
            return row
        enriched = dict(row)
        for role in ("MAP", "QUERY"):
            detail = raw_detail_by_key[(*key, role)]
            inventory = detail.get("inventory")
            audit = detail.get("bag_audit", detail.get("audit"))
            if not isinstance(inventory, Mapping) or not isinstance(audit, Mapping):
                _fail(f"raw_bags detail for {(*key, role)} lacks inventory/audit")
            if audit.get("ACQUISITION_AUDIT_PASS") is not True:
                _fail(f"raw_bags detail for {(*key, role)} is not an audited PASS")
            topic_rows = inventory.get("topics")
            if not isinstance(topic_rows, list):
                _fail(f"raw_bags detail for {(*key, role)} lacks topic rows")
            topics = {
                str(topic["topic"]): topic
                for topic in topic_rows
                if isinstance(topic, Mapping) and "topic" in topic
            }
            if set(topics) != set(REQUIRED_TOPICS):
                _fail(f"raw_bags detail topics changed for {(*key, role)}")
            enriched[role.lower()] = {
                "duration_s": audit.get("duration_s"),
                "frequencies_hz": audit.get("frequencies_hz"),
                "message_counts": audit.get("message_counts"),
                "topic_types": {
                    topic: topics[topic].get("message_type") for topic in REQUIRED_TOPICS
                },
                "frame_ids": audit.get("frame_ids"),
                "pointcloud2_fields": audit.get("pointcloud2_fields"),
                "motion_audit_status": audit.get("motion_audit_status"),
            }
        return enriched

    acquisition_rows = _rows(
        payload,
        "station_acquisition_audits",
        "station_acquisition_audit",
        "acquisition_audits",
        "stations",
    )
    if len(acquisition_rows) != 18:
        _fail(f"station acquisition audit count must be 18, got {len(acquisition_rows)}")
    acquisitions = _exact_unique_keys(
        acquisition_rows, expected_station_keys, _key, "station_acquisition_audits"
    )
    for key, compact_row in acquisitions.items():
        row = enriched_acquisition_row(compact_row)
        status = _text(
            row,
            ("acquisition_status", "station_acquisition_status", "status"),
            f"acquisition {key} status",
        )
        if status != "ACQUISITION_PASS":
            _fail(f"station {key} is not ACQUISITION_PASS")
        for role in ("MAP", "QUERY"):
            duration = _finite(
                _role_value(row, role, ("duration_s", "actual_duration_s"), f"acquisition {key}"),
                f"acquisition {key} {role} duration",
            )
            minimum = MAP_MIN_DURATION_S if role == "MAP" else QUERY_MIN_DURATION_S
            if duration < minimum:
                _fail(f"station {key} {role} duration is below hard minimum")
            lidar_rate = _role_rate(row, role, "/livox/lidar", f"acquisition {key}")
            imu_rate = _role_rate(row, role, "/livox/imu", f"acquisition {key}")
            if not LIDAR_RATE_RANGE_HZ[0] <= lidar_rate <= LIDAR_RATE_RANGE_HZ[1]:
                _fail(f"station {key} {role} LiDAR rate is outside frozen range")
            if not IMU_RATE_RANGE_HZ[0] <= imu_rate <= IMU_RATE_RANGE_HZ[1]:
                _fail(f"station {key} {role} IMU rate is outside frozen range")
            _role_count(row, role, "/livox/lidar", f"acquisition {key}")
            _role_count(row, role, "/livox/imu", f"acquisition {key}")
            if dict(_role_topics(row, role, f"acquisition {key}")) != REQUIRED_TOPICS:
                _fail(f"station {key} {role} topic/type mapping changed")
            if _role_frames(row, role, f"acquisition {key}") != {REQUIRED_FRAME_ID}:
                _fail(f"station {key} {role} frame_id is not exactly {REQUIRED_FRAME_ID}")
            if not REQUIRED_POINT_FIELDS.issubset(
                _role_fields(row, role, f"acquisition {key}")
            ):
                _fail(f"station {key} {role} PointCloud2 fields are incomplete")
            motion = _text(
                _role_mapping(row, role, f"acquisition {key}"),
                ("motion_status", "motion_audit_status", "STATICITY_SCREEN"),
                f"acquisition {key} {role} motion status",
            )
            if motion != "NO_OBVIOUS_MOTION":
                _fail(f"station {key} {role} motion screen did not pass")
        gap = _finite(
            _pair_value(row, ("gap_s", "actual_gap_s"), f"acquisition {key}"),
            f"acquisition {key} gap",
        )
        if gap < MINIMUM_MAP_QUERY_GAP_S:
            _fail(f"station {key} gap is below 10 seconds")
        no_overlap = _pair_value(
            row, ("no_overlap", "map_query_no_overlap"), f"acquisition {key}"
        )
        if no_overlap is not True:
            _fail(f"station {key} MAP/QUERY overlap is present or ambiguous")

    target_rows = _rows(payload, "targets", "target_map_manifest")
    if len(target_rows) != 18:
        _fail(f"target count must be 18, got {len(target_rows)}")
    targets = _exact_unique_keys(target_rows, expected_station_keys, _key, "targets")
    target_shas: dict[tuple[str, str], str] = {}
    for key, row in targets.items():
        target_sha = _sha(
            _one_of(row, ("target_sha256", "target_npy_sha256"), f"target {key} SHA"),
            f"target {key} SHA",
        )
        map_sha = _sha(
            _one_of(row, ("map_bag_sha256", "map_sha256"), f"target {key} map SHA"),
            f"target {key} map SHA",
        )
        if map_sha != _sha(
            _one_of(bags[(*key, "MAP")], ("sha256", "SHA256"), f"MAP bag {key} SHA"),
            f"MAP bag {key} SHA",
        ):
            _fail(f"target {key} is not bound to its MAP bag SHA")
        contribution = _one_of(
            row,
            ("query_contribution_to_target", "query_contribution", "query_frame_count"),
            f"target {key} query contribution",
        )
        if contribution != 0:
            _fail(f"target {key} contains QUERY contribution")
        input_roles = row.get("input_roles")
        if isinstance(input_roles, str):
            input_roles = [input_roles]
        if input_roles != ["MAP"]:
            _fail(f"target {key} input_roles must be MAP-only")
        construction = _text(row, ("construction",), f"target {key} construction")
        if "NO_REGISTRATION" not in construction:
            _fail(f"target {key} construction is not registration-free")
        for flag in ("registration_called", "odometry_called", "scan_matching_called"):
            if row.get(flag) is not False:
                _fail(f"target {key} {flag} must be false")
        target_point_count = _integer(
            _one_of(row, ("target_point_count",), f"target {key} point count"),
            f"target {key} point count",
            minimum=1,
        )
        map_frame_count = _integer(
            _one_of(row, ("map_frame_count",), f"target {key} MAP frame count"),
            f"target {key} MAP frame count",
            minimum=1,
        )
        raw_point_count = _integer(
            _one_of(row, ("raw_point_count",), f"target {key} raw point count"),
            f"target {key} raw point count",
            minimum=1,
        )
        filtered_point_count = _integer(
            _one_of(
                row,
                ("filtered_point_count", "finite_range_filtered_point_count"),
                f"target {key} filtered point count",
            ),
            f"target {key} filtered point count",
            minimum=1,
        )
        if not target_point_count <= filtered_point_count <= raw_point_count:
            _fail(f"target {key} point-count lineage is inconsistent")
        if map_frame_count <= 0:  # explicit for readability in verifier output
            _fail(f"target {key} contains no MAP frames")
        target_shas[key] = target_sha
        if verify_files:
            _verify_declared_file(
                repository,
                row,
                path_names=("target_path", "target_npy_path"),
                sha_names=("target_sha256", "target_npy_sha256"),
                size_names=("target_bytes", "target_size_bytes", "bytes"),
                label=f"target {key}",
            )

    snapshot_rows = _rows(
        payload, "snapshots", "snapshot_inventory_frozen", "snapshot_inventory"
    )
    if len(snapshot_rows) != 180:
        _fail(f"snapshot count must be 180, got {len(snapshot_rows)}")
    snapshots: dict[str, Mapping[str, Any]] = {}
    snapshots_by_station: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(snapshot_rows):
        key = _key(row, f"snapshots[{index}]")
        snapshot_id = _text(row, ("snapshot_id",), f"snapshot[{index}].snapshot_id")
        if snapshot_id in snapshots:
            _fail(f"duplicate snapshot_id: {snapshot_id}")
        if not snapshot_id.startswith(f"{key[0]}_{key[1]}_Q"):
            _fail(f"snapshot_id is not bound to scene/station: {snapshot_id}")
        snapshots[snapshot_id] = row
        snapshots_by_station[key].append(row)
        _sha(
            _one_of(row, ("source_sha256", "source_npy_sha256"), f"snapshot {snapshot_id} source SHA"),
            f"snapshot {snapshot_id} source SHA",
        )
        query_bag_sha = _sha(
            _one_of(row, ("query_bag_sha256",), f"snapshot {snapshot_id} QUERY bag SHA"),
            f"snapshot {snapshot_id} QUERY bag SHA",
        )
        expected_query_sha = _sha(
            _one_of(
                bags[(*key, "QUERY")],
                ("sha256", "SHA256"),
                f"QUERY bag {key} SHA",
            ),
            f"QUERY bag {key} SHA",
        )
        if query_bag_sha != expected_query_sha:
            _fail(f"snapshot {snapshot_id} is not bound to its QUERY bag SHA")
        row_target_sha = _sha(
            _one_of(row, ("target_sha256", "target_npy_sha256"), f"snapshot {snapshot_id} target SHA"),
            f"snapshot {snapshot_id} target SHA",
        )
        if row_target_sha != target_shas[key]:
            _fail(f"snapshot {snapshot_id} target SHA binding changed")
        _finite(
            _one_of(row, ("query_timestamp", "timestamp"), f"snapshot {snapshot_id} timestamp"),
            f"snapshot {snapshot_id} timestamp",
        )
        _integer(
            _one_of(
                row,
                ("query_frame_index", "frame_index"),
                f"snapshot {snapshot_id} query frame index",
            ),
            f"snapshot {snapshot_id} query frame index",
        )
        _integer(
            _one_of(row, ("source_point_count",), f"snapshot {snapshot_id} point count"),
            f"snapshot {snapshot_id} point count",
            minimum=1,
        )
        if row.get("selection_method") != (
            "FIXED_QUANTILE_SEQUENCE_RANK_NEAREST_UNUSED_EARLIER_TIE"
        ):
            _fail(f"snapshot {snapshot_id} selection method changed")
        if row.get("selection_frozen_before_registration") is not True:
            _fail(f"snapshot {snapshot_id} was not frozen before registration")
        if verify_files:
            _verify_declared_file(
                repository,
                row,
                path_names=("source_path", "source_npy_path"),
                sha_names=("source_sha256", "source_npy_sha256"),
                size_names=("source_bytes", "source_size_bytes", "bytes"),
                label=f"snapshot source {snapshot_id}",
            )

    if set(snapshots_by_station) != expected_station_keys:
        _fail("snapshot station keys differ from frozen 18 stations")
    for key, rows in snapshots_by_station.items():
        if len(rows) != 10:
            _fail(f"station {key} must have exactly 10 snapshots")
        ordered = sorted(
            rows,
            key=lambda row: _integer(
                _one_of(row, ("selection_index",), f"snapshot {key} selection index"),
                f"snapshot {key} selection index",
            ),
        )
        indexes = [int(row["selection_index"]) for row in ordered]
        if indexes != list(range(10)):
            _fail(f"station {key} snapshot selection indexes must be 0..9")
        quantiles = [
            _finite(_one_of(row, ("quantile",), f"snapshot {key} quantile"), f"snapshot {key} quantile")
            for row in ordered
        ]
        if tuple(quantiles) != QUERY_QUANTILES:
            _fail(f"station {key} snapshot quantiles changed")
        timestamps = [
            _finite(
                _one_of(row, ("query_timestamp", "timestamp"), f"snapshot {key} timestamp"),
                f"snapshot {key} timestamp",
            )
            for row in ordered
        ]
        if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
            _fail(f"station {key} snapshot timestamps must be strictly increasing")

    geometry_rows = _rows(payload, "geometry_metrics", "geometry_snapshot_metrics")
    if len(geometry_rows) != 180:
        _fail(f"geometry metric row count must be 180, got {len(geometry_rows)}")
    geometry_by_snapshot: dict[str, Mapping[str, Any]] = {}
    geometry_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    geometry_counts: Counter[tuple[str, str]] = Counter()
    metric_field_set = set(GEOMETRY_ONLY_FIELDS)
    for index, row in enumerate(geometry_rows):
        unknown = set(row) - GEOMETRY_METADATA_FIELDS - metric_field_set
        missing = metric_field_set - set(row)
        if unknown or missing:
            _fail(
                f"geometry_metrics[{index}] must contain exactly ten scientific fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        key = _key(row, f"geometry_metrics[{index}]")
        snapshot_id = _text(row, ("snapshot_id",), f"geometry_metrics[{index}].snapshot_id")
        if snapshot_id not in snapshots or snapshot_id in geometry_by_snapshot:
            _fail(f"geometry snapshot binding is missing/duplicated: {snapshot_id}")
        geometry_by_snapshot[snapshot_id] = row
        geometry_by_scene[key[0]].append(row)
        geometry_counts[key] += 1
        initial_count = _integer(row["initial_correspondence_count"], f"geometry {snapshot_id} correspondence count")
        valid_count = _integer(
            row["initial_valid_normal_correspondence_count"],
            f"geometry {snapshot_id} valid-normal count",
        )
        if valid_count > initial_count:
            _fail(f"geometry {snapshot_id} valid-normal count exceeds correspondence count")
        for field in GEOMETRY_ONLY_FIELDS[2:]:
            _finite(row[field], f"geometry {snapshot_id}.{field}")
        snapshot = snapshots[snapshot_id]
        geometry_selection = _integer(
            _one_of(row, ("selection_index",), f"geometry {snapshot_id} selection index"),
            f"geometry {snapshot_id} selection index",
        )
        snapshot_selection = _integer(
            _one_of(snapshot, ("selection_index",), f"snapshot {snapshot_id} selection index"),
            f"snapshot {snapshot_id} selection index",
        )
        if geometry_selection != snapshot_selection:
            _fail(f"selection index binding changed for {snapshot_id}")
        if "query_timestamp" in row or "timestamp" in row:
            geometry_timestamp = _finite(
                _one_of(
                    row,
                    ("query_timestamp", "timestamp"),
                    f"geometry {snapshot_id} timestamp",
                ),
                f"geometry {snapshot_id} timestamp",
            )
            snapshot_timestamp = _finite(
                _one_of(
                    snapshot,
                    ("query_timestamp", "timestamp"),
                    f"snapshot {snapshot_id} timestamp",
                ),
                f"snapshot {snapshot_id} timestamp",
            )
            if geometry_timestamp != snapshot_timestamp:
                _fail(f"query timestamp binding changed for {snapshot_id}")
    if set(geometry_by_snapshot) != set(snapshots):
        _fail("geometry metrics do not bind exactly all 180 snapshots")
    if set(geometry_counts) != expected_station_keys or set(geometry_counts.values()) != {10}:
        _fail("geometry metrics must contain exactly 10 rows per station")
    if set(geometry_by_scene) != expected_scene_ids or any(
        len(rows) != 30 for rows in geometry_by_scene.values()
    ):
        _fail("geometry metrics must contain exactly 30 rows per scene")

    geometry_scene_rows = _rows(
        payload, "geometry_scenes", "geometry_scene_summary", "geometry_admission"
    )
    if len(geometry_scene_rows) != 6:
        _fail(f"geometry scene count must be 6, got {len(geometry_scene_rows)}")
    geometry_scenes = _exact_unique_keys(
        geometry_scene_rows,
        expected_scene_ids,
        lambda row, label: _text(row, ("scene_id",), f"{label}.scene_id"),
        "geometry_scenes",
    )
    for scene_id, row in geometry_scenes.items():
        rows = geometry_by_scene[scene_id]
        medians = (
            float(median(float(metric["normalized_lambda_min_trans"]) for metric in rows)),
            float(median(float(metric["condition_number_trans"]) for metric in rows)),
            float(median(float(metric["spectral_entropy_trans"]) for metric in rows)),
        )
        expected_class = "RICH" if scene_id.startswith("FMB1_R") else "WEAK"
        if geometry_class(*medians) != expected_class:
            _fail(f"scene {scene_id} geometry does not satisfy expected {expected_class} gate")
        declared_class = _text(
            row, ("final_geometry_class", "geometry_class"), f"scene {scene_id} geometry class"
        )
        if declared_class != expected_class:
            _fail(f"scene {scene_id} geometry label changed or mismatches candidate class")
        admitted = row.get("admitted")
        status = row.get("geometry_admission_status", row.get("status"))
        if admitted is not True and status != "GEOMETRY_ADMITTED":
            _fail(f"scene {scene_id} is not geometry-admitted")
        stored = (
            _finite(
                _one_of(
                    row,
                    (
                        "geometry_lambda_min_median",
                        "normalized_lambda_min_trans_median",
                        "median_normalized_lambda_min_trans",
                    ),
                    f"scene {scene_id} lambda median",
                ),
                f"scene {scene_id} lambda median",
            ),
            _finite(
                _one_of(
                    row,
                    (
                        "geometry_condition_median",
                        "condition_number_trans_median",
                        "median_condition_number_trans",
                    ),
                    f"scene {scene_id} condition median",
                ),
                f"scene {scene_id} condition median",
            ),
            _finite(
                _one_of(
                    row,
                    (
                        "geometry_entropy_median",
                        "spectral_entropy_trans_median",
                        "median_spectral_entropy_trans",
                    ),
                    f"scene {scene_id} entropy median",
                ),
                f"scene {scene_id} entropy median",
            ),
        )
        if any(not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12) for left, right in zip(stored, medians)):
            _fail(f"scene {scene_id} geometry medians do not recompute")
        scene_declared_class = _text(
            scenes[scene_id],
            ("final_geometry_class", "geometry_class"),
            f"scene registry {scene_id} final class",
        )
        if scene_declared_class != expected_class:
            _fail(f"scene registry {scene_id} final geometry class is not frozen")

    canonical_rows = _rows(payload, "canonical_inputs", "canonical_input_manifest")
    if len(canonical_rows) != 180:
        _fail(f"canonical input count must be 180, got {len(canonical_rows)}")
    canonical_by_snapshot: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(canonical_rows):
        key = _key(row, f"canonical_inputs[{index}]")
        snapshot_id = _text(row, ("snapshot_id",), f"canonical_inputs[{index}].snapshot_id")
        if snapshot_id not in snapshots or snapshot_id in canonical_by_snapshot:
            _fail(f"canonical input snapshot is missing/duplicated: {snapshot_id}")
        canonical_by_snapshot[snapshot_id] = row
        snapshot = snapshots[snapshot_id]
        source_sha = _sha(
            _one_of(row, ("source_sha256", "source_npy_sha256"), f"canonical {snapshot_id} source SHA"),
            f"canonical {snapshot_id} source SHA",
        )
        target_sha = _sha(
            _one_of(row, ("target_sha256", "target_npy_sha256"), f"canonical {snapshot_id} target SHA"),
            f"canonical {snapshot_id} target SHA",
        )
        if source_sha != _sha(
            _one_of(snapshot, ("source_sha256", "source_npy_sha256"), f"snapshot {snapshot_id} source SHA"),
            f"snapshot {snapshot_id} source SHA",
        ) or target_sha != target_shas[key]:
            _fail(f"canonical input SHA binding changed for {snapshot_id}")
        canonical_timestamp = _finite(
            _one_of(row, ("query_timestamp", "timestamp"), f"canonical {snapshot_id} timestamp"),
            f"canonical {snapshot_id} timestamp",
        )
        snapshot_timestamp = _finite(
            _one_of(snapshot, ("query_timestamp", "timestamp"), f"snapshot {snapshot_id} timestamp"),
            f"snapshot {snapshot_id} timestamp",
        )
        if canonical_timestamp != snapshot_timestamp:
            _fail(f"canonical input query timestamp changed for {snapshot_id}")
        if not _identity(_one_of(row, ("T0", "t0"), f"canonical {snapshot_id} T0")):
            _fail(f"canonical input {snapshot_id} T0 is not identity")
    if set(canonical_by_snapshot) != set(snapshots):
        _fail("canonical inputs do not bind exactly all 180 snapshots")

    attestation = _mapping(payload.get("no_icp_attestation"), "no_icp_attestation")
    for field in (
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "formal_trial_count",
    ):
        if _integer(attestation.get(field), f"no_icp_attestation.{field}") != 0:
            _fail(f"no_icp_attestation.{field} must be zero")
    if attestation.get("status", "PASS") != "PASS":
        _fail("NO_ICP_ATTESTATION is not PASS")
    if attestation.get("FORMAL_REGISTRATION_AUTHORIZED", False) is not False:
        _fail("NO_ICP_ATTESTATION authorizes formal registration")
    if attestation.get("FORMAL_ICP_UNLOCKED", False) is not False:
        _fail("NO_ICP_ATTESTATION reports FORMAL_ICP_UNLOCKED")

    readiness = _mapping(payload.get("readiness"), "readiness")
    required_true = (
        "FMB1_PRE_REGISTRATION_DATA_READY",
        "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION",
    )
    for field in required_true:
        if readiness.get(field) is not True:
            _fail(f"readiness.{field} must be true")
    if readiness.get("FORMAL_REGISTRATION_AUTHORIZED") is not False:
        _fail("readiness.FORMAL_REGISTRATION_AUTHORIZED must be false")
    if readiness.get("FORMAL_ICP_UNLOCKED", False) is not False:
        _fail("readiness.FORMAL_ICP_UNLOCKED must be false")
    if readiness.get("NO_FORMAL_REGISTRATION", True) is not True:
        _fail("readiness.NO_FORMAL_REGISTRATION must be true")
    final_result_values = [
        readiness[field]
        for field in ("FORMAL_MEASUREMENT_RESULT", "MEASUREMENT_FINAL_RESULT")
        if field in readiness
    ]
    if not final_result_values or any(value is not False for value in final_result_values):
        _fail("readiness measurement-final-result flag must be false")
    expected_counts = {
        "scene_count": 6,
        "rich_scene_count": 3,
        "weak_scene_count": 3,
        "station_count": 18,
        "snapshot_count": 180,
        "rich_snapshot_count": 90,
        "weak_snapshot_count": 90,
        "planned_open3d_trials": 180,
        "planned_pcl_trials": 180,
        "planned_total_trials": 360,
        "actual_trials": 0,
        "actual_registration_trials": 0,
        "registration_execution_count": 0,
    }
    for field, expected in expected_counts.items():
        if _integer(readiness.get(field), f"readiness.{field}") != expected:
            _fail(f"readiness.{field} must be {expected}")

    return {
        "schema": "mid360_fmb1_preregistration_verification_v1",
        "status": "PASS",
        "verify_files": bool(verify_files),
        "checks": {
            "backend_contract": True,
            "scene_registry": True,
            "station_registry": True,
            "raw_bag_mapping": True,
            "acquisition_audit": True,
            "target_map_lineage": True,
            "snapshot_freeze": True,
            "geometry_only_firewall": True,
            "geometry_admission": True,
            "canonical_inputs": True,
            "no_icp_attestation": True,
            "readiness": True,
            "file_authentication": bool(verify_files),
        },
        "counts": {
            "scenes": len(scene_rows),
            "stations": len(station_rows),
            "bags": len(bag_rows),
            "targets": len(target_rows),
            "snapshots": len(snapshot_rows),
            "geometry_rows": len(geometry_rows),
            "canonical_inputs": len(canonical_rows),
        },
        "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
        "manifest_sha256": _canonical_payload_sha(payload),
    }


__all__ = [
    "CAPTURE_PREFIXES",
    "EXPECTED_STATION_PREFIX",
    "GEOMETRY_ONLY_FIELDS",
    "MANIFEST_SCHEMA",
    "validate_manifest_payload",
]
