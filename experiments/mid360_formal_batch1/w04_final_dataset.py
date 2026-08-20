"""Fail-closed FMB1 final-dataset composer for the W04 replacement.

This module does not read ROS messages, construct point clouds, or import a
registration backend.  It combines already-frozen pre-registration evidence
only after independently checking the replacement boundary:

* W02 remains a retained, geometry-only rejected candidate;
* W04 has three passing stations, six authenticated bags, three MAP-only
  targets, thirty frozen snapshots, and a scene-level WEAK classification;
* the final set is exactly R01/R02/R03/W01/W03/W04; and
* every registration/authorization/lock counter remains closed.

The output deliberately distinguishes the 42 historical candidate bags from
the 36 admitted bags.  It never edits the original FMB1 frozen manifest.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    GEOMETRY_ONLY_FIELDS as _PROTOCOL_GEOMETRY_ONLY_FIELDS,
)

from .protocol import (
    BACKEND_CONTRACT_SHA256,
    QUERY_QUANTILES,
    STATION_IDS,
    geometry_class,
)


class W04FinalDatasetError(RuntimeError):
    """Raised when final-dataset evidence is incomplete or inconsistent."""


ORIGINAL_SCENES = (
    "FMB1_R01",
    "FMB1_R02",
    "FMB1_R03",
    "FMB1_W01",
    "FMB1_W02",
    "FMB1_W03",
)
FINAL_SCENES = (
    "FMB1_R01",
    "FMB1_R02",
    "FMB1_R03",
    "FMB1_W01",
    "FMB1_W03",
    "FMB1_W04",
)
RICH_SCENES = frozenset({"FMB1_R01", "FMB1_R02", "FMB1_R03"})
WEAK_SCENES = frozenset({"FMB1_W01", "FMB1_W03", "FMB1_W04"})
REJECTED_SCENE = "FMB1_W02"
REPLACEMENT_SCENE = "FMB1_W04"
EXPECTED_W04_PREFIX = {
    "S01": "20260820_081749",
    "S02": "20260820_081954",
    "S03": "20260820_082207",
}
GEOMETRY_ONLY_FIELDS = tuple(_PROTOCOL_GEOMETRY_ONLY_FIELDS)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_FORBIDDEN_RESULT_FRAGMENTS = (
    "t_est",
    "estimated_transform",
    "final_transform",
    "final_residual",
    "fitness",
    "turnover",
    "solver_result",
    "backend_result",
    "registration_result",
    "translation_error",
    "rotation_error",
    "capture_radius",
)
_ZERO_COUNTERS = (
    "open3d_registration_call_count",
    "pcl_cli_invocation_count",
    "other_registration_process_count",
    "formal_trial_count",
    "actual_open3d_trials",
    "actual_pcl_trials",
    "actual_formal_trials",
    "actual_registration_trials",
    "actual_trials",
    "registration_execution_count",
)
_FALSE_FLAGS = (
    "FORMAL_AUTHORITY",
    "FORMAL_LOCK_ISSUED",
    "FORMAL_ICP_UNLOCKED",
    "FORMAL_REGISTRATION_AUTHORIZED",
    "MEASUREMENT_FINAL_RESULT",
    "PROPOSED_AMENDMENT_ACTIVE",
)


def _fail(message: str) -> None:
    raise W04FinalDatasetError(f"FMB1_FINAL_DATASET_INVALID: {message}")


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
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


def _rows(payload: Mapping[str, Any], *names: str) -> list[Mapping[str, Any]]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
            return list(value)
    _fail(f"required row list missing: {'/'.join(names)}")
    raise AssertionError("unreachable")


def _station_id(value: Any, scene_id: str, label: str) -> str:
    station = str(value)
    prefix = f"{scene_id}_"
    if station.startswith(prefix):
        station = station[len(prefix) :]
    if station not in STATION_IDS:
        _fail(f"{label} has invalid station_id {value!r}")
    return station


def _status(row: Mapping[str, Any]) -> str:
    return str(
        row.get(
            "acquisition_status",
            row.get("station_status", row.get("station_acquisition_status", "")),
        )
    )


def _row_sha(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        if name in row:
            return _sha(row[name], name)
    inventory = row.get("inventory")
    if isinstance(inventory, Mapping) and "bag_sha256" in inventory:
        return _sha(inventory["bag_sha256"], "inventory.bag_sha256")
    _fail(f"row is missing SHA field: {'/'.join(names)}")
    raise AssertionError("unreachable")


def _row_path(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value:
            return value
    inventory = row.get("inventory")
    if isinstance(inventory, Mapping):
        value = inventory.get("bag_path")
        if isinstance(value, str) and value:
            return value
    _fail(f"row is missing path field: {'/'.join(names)}")
    raise AssertionError("unreachable")


def _assert_no_result_fields(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if any(fragment in key for fragment in _FORBIDDEN_RESULT_FRAGMENTS):
                _fail(f"registration-derived field is forbidden: {path}.{raw_key}")
            if key in {
                "registration_called",
                "registration_executed",
                "odometry_called",
                "scan_matching_called",
            } and child is not False:
                _fail(f"{path}.{raw_key} must be false")
            _assert_no_result_fields(child, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_result_fields(child, f"{path}[{index}]")


def _assert_closed_control(payload: Mapping[str, Any], label: str) -> None:
    for key in _ZERO_COUNTERS:
        if key in payload and payload[key] != 0:
            _fail(f"{label}.{key} must be 0")
    for key in _FALSE_FLAGS:
        if key in payload and payload[key] is not False:
            _fail(f"{label}.{key} must be false")


def _unique(
    rows: Sequence[Mapping[str, Any]], key_fn: Any, expected: set[Any], label: str
) -> dict[Any, Mapping[str, Any]]:
    output: dict[Any, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        key = key_fn(row)
        if key in output:
            _fail(f"duplicate {label} key at row {index}: {key}")
        output[key] = row
    if set(output) != expected:
        _fail(
            f"{label} keys differ; missing={sorted(expected - set(output))}, "
            f"extra={sorted(set(output) - expected)}"
        )
    return output


def _bag_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    scene = str(row.get("scene_id"))
    return (scene, _station_id(row.get("station_id"), scene, "bag"), str(row.get("role")))


def _station_key(row: Mapping[str, Any]) -> tuple[str, str]:
    scene = str(row.get("scene_id"))
    return (scene, _station_id(row.get("station_id"), scene, "station"))


def _snapshot_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    scene = str(row.get("scene_id"))
    selection = row.get("selection_index", row.get("quantile_index"))
    return (
        scene,
        _station_id(row.get("station_id"), scene, "snapshot"),
        _integer(selection, "snapshot.selection_index"),
    )


def _metric_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return _snapshot_key(row)


def _normalize_raw(row: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(row))
    scene = str(output.get("scene_id"))
    station = _station_id(output.get("station_id"), scene, "raw bag")
    role = str(output.get("role"))
    if role not in {"MAP", "QUERY"}:
        _fail(f"raw bag role must be MAP or QUERY: {role}")
    path = _row_path(output, "raw_absolute_path", "absolute_path", "raw_path")
    name = str(output.get("raw_filename") or Path(path).name)
    sha = _row_sha(output, "sha256", "raw_sha256")
    output.update(
        {
            "scene_id": scene,
            "station_id": station,
            "role": role,
            "raw_filename": name,
            "raw_absolute_path": path,
            "sha256": sha,
        }
    )
    inventory = output.get("inventory")
    inventory = inventory if isinstance(inventory, Mapping) else {}
    for target, aliases in {
        "start_timestamp": ("start_timestamp",),
        "end_timestamp": ("end_timestamp",),
        "bytes": ("bytes", "file_size"),
    }.items():
        if output.get(target) is None:
            for alias in aliases:
                if inventory.get(alias) is not None:
                    output[target] = inventory[alias]
                    break
    return output


def _normalize_target(row: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(row))
    scene = str(output.get("scene_id"))
    station = _station_id(output.get("station_id"), scene, "target")
    output["scene_id"] = scene
    output["station_id"] = station
    output["target_path"] = _row_path(output, "target_path", "path")
    output["target_npy_sha256"] = _row_sha(
        output, "target_npy_sha256", "target_sha256", "npy_sha256"
    )
    target_file = Path(output["target_path"])
    if output.get("target_size_bytes") is None and target_file.is_file():
        output["target_size_bytes"] = target_file.stat().st_size
    output["map_bag_sha256"] = _row_sha(output, "map_bag_sha256")
    query_count = output.get(
        "query_contribution_to_target", output.get("query_contribution_count")
    )
    if query_count != 0:
        _fail(f"{scene}/{station} QUERY contribution to target must be 0")
    if output.get("query_frame_count", 0) != 0:
        _fail(f"{scene}/{station} target query_frame_count must be 0")
    roles = output.get("input_roles", "MAP")
    if roles not in ("MAP", ["MAP"], ("MAP",)):
        _fail(f"{scene}/{station} target input_roles must be MAP-only")
    for key in ("registration_called", "odometry_called", "scan_matching_called"):
        if output.get(key, False) is not False:
            _fail(f"{scene}/{station} target {key} must be false")
    output["input_roles"] = "MAP"
    output["query_contribution_to_target"] = 0
    output["query_frame_count"] = 0
    output["registration_called"] = False
    output["odometry_called"] = False
    output["scan_matching_called"] = False
    return output


def _normalize_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(row))
    scene = str(output.get("scene_id"))
    station = _station_id(output.get("station_id"), scene, "snapshot")
    selection = _integer(
        output.get("selection_index", output.get("quantile_index")),
        "snapshot.selection_index",
    )
    output.update(
        {
            "scene_id": scene,
            "station_id": station,
            "selection_index": selection,
            "source_path": _row_path(output, "source_path", "path"),
            "source_npy_sha256": _row_sha(
                output, "source_npy_sha256", "source_sha256", "npy_sha256"
            ),
            "target_npy_sha256": _row_sha(
                output, "target_npy_sha256", "target_sha256"
            ),
        }
    )
    if not isinstance(output.get("snapshot_id"), str) or not output["snapshot_id"]:
        output["snapshot_id"] = f"{scene}_{station}_Q{selection:02d}"
    source_file = Path(output["source_path"])
    if output.get("source_size_bytes") is None and source_file.is_file():
        output["source_size_bytes"] = source_file.stat().st_size
    return output


def _normalize_metric(row: Mapping[str, Any]) -> dict[str, Any]:
    scene, station, selection = _metric_key(row)
    output: dict[str, Any] = {
        "scene_id": scene,
        "station_id": station,
        "selection_index": selection,
        "snapshot_id": str(row.get("snapshot_id") or f"{scene}_{station}_Q{selection:02d}"),
    }
    for field in GEOMETRY_ONLY_FIELDS:
        if field not in row:
            _fail(f"geometry metric is missing {field}: {scene}/{station}/{selection}")
        output[field] = _finite(row[field], f"geometry metric {field}")
    extra = set(row) - set(output) - {"geometry_only", "registration_executed"}
    forbidden = [
        key
        for key in extra
        if any(fragment in str(key).lower() for fragment in _FORBIDDEN_RESULT_FRAGMENTS)
    ]
    if forbidden:
        _fail(f"geometry metric contains forbidden fields: {sorted(forbidden)}")
    if row.get("registration_executed", False) is not False:
        _fail("geometry metric registration_executed must be false")
    return output


def _scene_geometry(
    scene_id: str, rows: Sequence[Mapping[str, Any]], expected_class: str
) -> dict[str, Any]:
    if len(rows) != 30:
        _fail(f"{scene_id} must have exactly 30 geometry rows")
    stations = Counter(str(row["station_id"]) for row in rows)
    if stations != Counter({station: 10 for station in STATION_IDS}):
        _fail(f"{scene_id} geometry station distribution must be 10/10/10")
    medians = {
        field: float(median(float(row[field]) for row in rows))
        for field in GEOMETRY_ONLY_FIELDS
    }
    classification = geometry_class(
        medians["normalized_lambda_min_trans"],
        medians["condition_number_trans"],
        medians["spectral_entropy_trans"],
    )
    if classification != expected_class:
        _fail(
            f"{scene_id} geometry class is {classification}, expected {expected_class}"
        )
    return {
        "scene_id": scene_id,
        "semantic_candidate_label": f"{expected_class}_CANDIDATE",
        "aggregation": "MEDIAN_OVER_ALL_30_NESTED_STATION_SNAPSHOTS",
        "station_count": 3,
        "snapshot_count": 30,
        **{f"median_{field}": value for field, value in medians.items()},
        "final_geometry_class": classification,
        "geometry_admission_status": "GEOMETRY_ADMITTED",
        "candidate_class_alignment": True,
        "admitted": True,
        "failure_reason": None,
        "replacement_allowed_under_preregistration": False,
    }


def _verify_declared_file(path_value: Any, sha_value: Any, label: str) -> None:
    path = Path(str(path_value)).resolve(strict=True)
    expected = _sha(sha_value, f"{label}.sha256")
    if _sha256_file(path) != expected:
        _fail(f"{label} file SHA mismatch: {path}")


def _validate_w04_filenames(raw: Sequence[Mapping[str, Any]]) -> None:
    expected: dict[tuple[str, str], str] = {}
    for station, prefix in EXPECTED_W04_PREFIX.items():
        expected[(station, "MAP")] = f"mid360_{prefix}_part1_20s.bag"
        expected[(station, "QUERY")] = f"mid360_{prefix}_part2_15s.bag"
    actual = {(str(row["station_id"]), str(row["role"])): str(row["raw_filename"]) for row in raw}
    if actual != expected:
        _fail(f"W04 raw filename/station mapping differs: {actual}")


def _validate_replacement_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "rejected_candidate_scene_id": REJECTED_SCENE,
        "rejection_reason": "GEOMETRY_ONLY_INELIGIBLE",
        "replacement_scene_id": REPLACEMENT_SCENE,
        "semantic_candidate_label": "WEAK_CANDIDATE",
        "decision_before_any_icp": True,
        "formal_trial_count_at_decision": 0,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            _fail(f"replacement plan {key} must be {value!r}")
    _assert_closed_control(plan, "replacement_plan")


def build_final_dataset_payload(
    original_manifest: Mapping[str, Any],
    w04_acquisition: Mapping[str, Any],
    w04_assets: Mapping[str, Any],
    w04_geometry: Mapping[str, Any],
    replacement_plan: Mapping[str, Any],
    no_icp_attestation: Mapping[str, Any],
    *,
    repository: Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Compose and validate the final replacement dataset in memory.

    ``verify_files`` defaults to true for production use.  Unit tests may set
    it false when using manifest-only synthetic fixtures.
    """

    for label, payload in (
        ("original_manifest", original_manifest),
        ("w04_acquisition", w04_acquisition),
        ("w04_assets", w04_assets),
        ("w04_geometry", w04_geometry),
        ("replacement_plan", replacement_plan),
        ("no_icp_attestation", no_icp_attestation),
    ):
        if not isinstance(payload, Mapping):
            _fail(f"{label} must be a mapping")
        _assert_no_result_fields(payload, label)
        _assert_closed_control(payload, label)

    if original_manifest.get("backend_parameter_contract_sha256") != BACKEND_CONTRACT_SHA256:
        _fail("original backend parameter contract binding changed")
    if repository is not None:
        backend = repository / "frozen_assets/backend_parameter_contract.json"
        if _sha256_file(backend.resolve(strict=True)) != BACKEND_CONTRACT_SHA256:
            _fail("live backend parameter contract changed")
    _validate_replacement_plan(replacement_plan)
    attestation_pass = no_icp_attestation.get(
        "NO_ICP_ATTESTATION_PASS",
        no_icp_attestation.get("pass", no_icp_attestation.get("status") == "PASS"),
    )
    if attestation_pass is not True:
        _fail("source W04 NO-ICP attestation must PASS")
    if no_icp_attestation.get("NO_FORMAL_REGISTRATION") is not True:
        _fail("source W04 NO-ICP attestation must keep NO_FORMAL_REGISTRATION=true")
    for key in (
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "formal_trial_count",
    ):
        if no_icp_attestation.get(key) != 0:
            _fail(f"source W04 NO-ICP attestation requires {key}=0")
    for key in ("FORMAL_ICP_UNLOCKED", "FORMAL_REGISTRATION_AUTHORIZED"):
        if no_icp_attestation.get(key) is not False:
            _fail(f"source W04 NO-ICP attestation requires {key}=false")
    source_backend_sha = no_icp_attestation.get(
        "backend_parameter_contract_sha256",
        no_icp_attestation.get("backend_contract_sha256"),
    )
    if source_backend_sha != BACKEND_CONTRACT_SHA256:
        _fail("source W04 NO-ICP backend contract binding changed")

    original_raw = [_normalize_raw(row) for row in _rows(original_manifest, "raw_bags")]
    original_bag_keys = {
        (scene, station, role)
        for scene in ORIGINAL_SCENES
        for station in STATION_IDS
        for role in ("MAP", "QUERY")
    }
    _unique(original_raw, _bag_key, original_bag_keys, "original raw bag")

    original_stations = [copy.deepcopy(dict(row)) for row in _rows(original_manifest, "stations")]
    original_station_keys = {
        (scene, station) for scene in ORIGINAL_SCENES for station in STATION_IDS
    }
    _unique(original_stations, _station_key, original_station_keys, "original station")
    if any(_status(row) not in {"PASS", "ACQUISITION_PASS"} for row in original_stations):
        _fail("all original stations must retain acquisition PASS")

    original_targets = [_normalize_target(row) for row in _rows(original_manifest, "targets")]
    _unique(original_targets, _station_key, original_station_keys, "original target")
    original_snapshots = [
        _normalize_snapshot(row) for row in _rows(original_manifest, "snapshots")
    ]
    original_snapshot_keys = {
        (scene, station, selection)
        for scene in ORIGINAL_SCENES
        for station in STATION_IDS
        for selection in range(10)
    }
    _unique(
        original_snapshots,
        _snapshot_key,
        original_snapshot_keys,
        "original snapshot",
    )
    original_metrics = [
        _normalize_metric(row) for row in _rows(original_manifest, "geometry_metrics")
    ]
    _unique(original_metrics, _metric_key, original_snapshot_keys, "original geometry")

    original_scene_rows = _rows(original_manifest, "geometry_scenes")
    original_scene_map = _unique(
        original_scene_rows,
        lambda row: str(row.get("scene_id")),
        set(ORIGINAL_SCENES),
        "original geometry scene",
    )
    w02_scene = original_scene_map[REJECTED_SCENE]
    if w02_scene.get("geometry_admission_status") != "GEOMETRY_REJECTED":
        _fail("W02 must remain GEOMETRY_REJECTED")
    if w02_scene.get("final_geometry_class") != "RICH":
        _fail("W02 retained geometry evidence must remain RICH")
    if w02_scene.get("failure_reason") != "SEMANTIC_CANDIDATE_GEOMETRY_CLASS_MISMATCH":
        _fail("W02 original geometry failure reason changed")

    w04_raw = [_normalize_raw(row) for row in _rows(w04_acquisition, "raw_bags", "bags")]
    w04_bag_keys = {
        (REPLACEMENT_SCENE, station, role)
        for station in STATION_IDS
        for role in ("MAP", "QUERY")
    }
    _unique(w04_raw, _bag_key, w04_bag_keys, "W04 raw bag")
    _validate_w04_filenames(w04_raw)

    w04_stations = [
        copy.deepcopy(dict(row))
        for row in _rows(w04_acquisition, "stations", "station_audits")
    ]
    w04_station_keys = {(REPLACEMENT_SCENE, station) for station in STATION_IDS}
    _unique(w04_stations, _station_key, w04_station_keys, "W04 station")
    if any(_status(row) not in {"PASS", "ACQUISITION_PASS"} for row in w04_stations):
        _fail("all three W04 stations must pass acquisition")

    if w04_assets.get("failures") not in (None, []):
        _fail("W04 asset construction contains failures")
    w04_targets = [_normalize_target(row) for row in _rows(w04_assets, "targets")]
    _unique(w04_targets, _station_key, w04_station_keys, "W04 target")
    w04_snapshots = [
        _normalize_snapshot(row) for row in _rows(w04_assets, "snapshots")
    ]
    w04_snapshot_keys = {
        (REPLACEMENT_SCENE, station, selection)
        for station in STATION_IDS
        for selection in range(10)
    }
    _unique(w04_snapshots, _snapshot_key, w04_snapshot_keys, "W04 snapshot")

    if w04_geometry.get("failures") not in (None, []):
        _fail("W04 geometry contains failures")
    w04_metrics = [
        _normalize_metric(row)
        for row in _rows(w04_geometry, "snapshot_metrics", "geometry_metrics")
    ]
    _unique(w04_metrics, _metric_key, w04_snapshot_keys, "W04 geometry")
    w04_scene = _scene_geometry(REPLACEMENT_SCENE, w04_metrics, "WEAK")
    declared_w04_scenes = _rows(w04_geometry, "scene_summaries", "geometry_scenes")
    if len(declared_w04_scenes) != 1 or declared_w04_scenes[0].get("scene_id") != REPLACEMENT_SCENE:
        _fail("W04 geometry must contain exactly one W04 scene summary")
    declared = declared_w04_scenes[0]
    if declared.get("final_geometry_class") != "WEAK":
        _fail("declared W04 final geometry class must be WEAK")
    if declared.get("geometry_admission_status") != "GEOMETRY_ADMITTED":
        _fail("declared W04 geometry admission must be GEOMETRY_ADMITTED")
    for field in (
        "median_normalized_lambda_min_trans",
        "median_condition_number_trans",
        "median_spectral_entropy_trans",
    ):
        if field in declared and not math.isclose(
            _finite(declared[field], f"W04.{field}"),
            float(w04_scene[field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail(f"declared W04 {field} does not match the 30-snapshot median")

    final_raw = [row for row in original_raw if row["scene_id"] != REJECTED_SCENE] + w04_raw
    final_stations = [
        row for row in original_stations if str(row.get("scene_id")) != REJECTED_SCENE
    ] + w04_stations
    final_targets = [
        row for row in original_targets if row["scene_id"] != REJECTED_SCENE
    ] + w04_targets
    final_snapshots = [
        row for row in original_snapshots if row["scene_id"] != REJECTED_SCENE
    ] + w04_snapshots
    final_metrics = [
        row for row in original_metrics if row["scene_id"] != REJECTED_SCENE
    ] + w04_metrics

    metrics_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in final_metrics:
        metrics_by_scene[str(row["scene_id"])].append(row)
    final_geometry_scenes = [
        _scene_geometry(
            scene,
            metrics_by_scene[scene],
            "RICH" if scene in RICH_SCENES else "WEAK",
        )
        for scene in FINAL_SCENES
    ]

    rejected_raw = [row for row in original_raw if row["scene_id"] == REJECTED_SCENE]
    rejected_candidates = [
        {
            "scene_id": REJECTED_SCENE,
            "station_id": row["station_id"],
            "role": row["role"],
            "raw_filename": row["raw_filename"],
            "raw_absolute_path": row["raw_absolute_path"],
            "sha256": row["sha256"],
            "bytes": row.get("bytes"),
            "candidate_status": "GEOMETRY_REJECTED",
            "final_geometry_class": "RICH",
            "original_geometry_failure_reason": "SEMANTIC_CANDIDATE_GEOMETRY_CLASS_MISMATCH",
            "exclusion_reason": "GEOMETRY_ONLY_INELIGIBLE",
            "replacement_scene_id": REPLACEMENT_SCENE,
            "rejected_before_icp": True,
            "retained": True,
            "included_in_final_set": False,
        }
        for row in rejected_raw
    ]

    lineage = {
        "schema": "mid360_fmb1_replacement_lineage_v1",
        "rejected_candidate_scene_id": REJECTED_SCENE,
        "candidate_status": "GEOMETRY_REJECTED",
        "rejection_reason": "GEOMETRY_ONLY_INELIGIBLE",
        "original_geometry_failure_reason": "SEMANTIC_CANDIDATE_GEOMETRY_CLASS_MISMATCH",
        "replacement_scene_id": REPLACEMENT_SCENE,
        "W02_RETAINED": True,
        "W02_INCLUDED_IN_FINAL_SET": False,
        "W02_REJECTED_BEFORE_ANY_ICP": True,
        "W04_ADMISSION_OCCURRED_BEFORE_ANY_ICP": True,
        "w04_admission_before_any_icp": True,
        "decision_timestamp": replacement_plan.get("decision_timestamp"),
        "decision_before_any_icp": True,
        "formal_trial_count_at_decision": 0,
        "formal_trial_count_at_w04_admission": 0,
        "formal_trial_count_at_admission": 0,
        "replacement_decision_used_registration_evidence": False,
        "w02_raw_bag_count": 6,
        "w02_snapshot_count_excluded": 30,
        "w04_raw_bag_count": 6,
        "w04_snapshot_count_admitted": 30,
    }

    safe_attestation = {
        "schema": "mid360_fmb1_final_dataset_no_icp_attestation_v1",
        "NO_FORMAL_REGISTRATION": True,
        "NO_ICP_ATTESTATION_PASS": True,
        "pass": True,
        "open3d_registration_call_count": 0,
        "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0,
        "formal_trial_count": 0,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "actual_registration_trials": 0,
        "registration_execution_count": 0,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "MEASUREMENT_FINAL_RESULT": False,
        "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
        "source_attestation_sha256": _sha256_bytes(_canonical_json(no_icp_attestation)),
    }

    final_scene_registry = {
        "schema": "mid360_fmb1_final_scene_registry_v1",
        "frozen_before_registration": True,
        "scenes": final_geometry_scenes,
    }
    final_station_registry = {
        "schema": "mid360_fmb1_final_station_registry_v1",
        "frozen_before_registration": True,
        "stations": sorted(final_stations, key=_station_key),
    }
    readiness = {
        "schema": "mid360_fmb1_final_dataset_readiness_v1",
        "FMB1_W04_ACQUISITION_PASS": True,
        "FMB1_W04_FINAL_GEOMETRY_CLASS": "WEAK",
        "FMB1_W04_ADMISSION_PASS": True,
        "FMB1_FINAL_DATASET_READY": True,
        "FINAL_RICH_SCENE_COUNT": 3,
        "FINAL_WEAK_SCENE_COUNT": 3,
        "FINAL_SCENE_COUNT": 6,
        "FINAL_STATION_COUNT": 18,
        "FINAL_SNAPSHOT_COUNT": 180,
        "RICH_SNAPSHOT_COUNT": 90,
        "WEAK_SNAPSHOT_COUNT": 90,
        "RAW_CANDIDATE_BAG_COUNT": 42,
        "FINAL_ADMITTED_BAG_COUNT": 36,
        "REJECTED_BAG_COUNT": 6,
        "W02_RETAINED": True,
        "W02_INCLUDED_IN_FINAL_SET": False,
        "W04_INCLUDED_IN_FINAL_SET": True,
        "W02_SNAPSHOT_COUNT_IN_FINAL_SET": 0,
        "FINAL_SNAPSHOT_DUPLICATE_COUNT": 0,
        "FINAL_SNAPSHOT_MISSING_COUNT": 0,
        "FINAL_SNAPSHOT_ORPHAN_COUNT": 0,
        "FINAL_SNAPSHOT_WRONG_SCENE_COUNT": 0,
        "FINAL_SNAPSHOT_WRONG_STATION_COUNT": 0,
        "QUERY_CONTRIBUTION_TO_TARGET_TOTAL": 0,
        "READY_FOR_ZERO_PERTURBATION_AMENDMENT_ACTIVATION": True,
        "ZERO_PERTURBATION_MAINLINE_AMENDMENT_STATUS": "PROPOSED_NOT_ACTIVE",
        "PROPOSED_AMENDMENT_ACTIVE": False,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "MEASUREMENT_FINAL_RESULT": False,
        "NO_FORMAL_REGISTRATION": True,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "actual_registration_trials": 0,
        "registration_execution_count": 0,
        "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
    }

    payload = {
        "schema": "mid360_fmb1_final_dataset_v1",
        "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
        "raw_candidate_bags": sorted(original_raw + w04_raw, key=_bag_key),
        "final_scene_registry": final_scene_registry,
        "final_station_registry": final_station_registry,
        "final_raw_bags": sorted(final_raw, key=_bag_key),
        "final_targets": sorted(final_targets, key=_station_key),
        "final_snapshots": sorted(final_snapshots, key=_snapshot_key),
        "final_geometry_metrics": sorted(final_metrics, key=_metric_key),
        "final_geometry_scenes": final_geometry_scenes,
        "rejected_candidates": sorted(
            rejected_candidates,
            key=lambda row: (row["scene_id"], row["station_id"], row["role"]),
        ),
        "replacement_lineage": lineage,
        "no_icp_attestation": safe_attestation,
        "readiness": readiness,
    }
    verify_final_dataset_payload(payload, verify_files=verify_files)
    return payload


def verify_final_dataset_payload(
    payload: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Verify the composed final dataset without importing producer/backends."""

    if payload.get("schema") != "mid360_fmb1_final_dataset_v1":
        _fail("unexpected final dataset schema")
    if payload.get("backend_parameter_contract_sha256") != BACKEND_CONTRACT_SHA256:
        _fail("final backend parameter contract binding changed")
    _assert_no_result_fields(payload)
    _assert_closed_control(payload.get("readiness", {}), "readiness")
    _assert_closed_control(payload.get("no_icp_attestation", {}), "no_icp_attestation")

    candidate_raw = [_normalize_raw(row) for row in _rows(payload, "raw_candidate_bags")]
    candidate_keys = {
        (scene, station, role)
        for scene in (*ORIGINAL_SCENES, REPLACEMENT_SCENE)
        for station in STATION_IDS
        for role in ("MAP", "QUERY")
    }
    _unique(candidate_raw, _bag_key, candidate_keys, "candidate raw bag")

    final_raw = [_normalize_raw(row) for row in _rows(payload, "final_raw_bags")]
    final_bag_keys = {
        (scene, station, role)
        for scene in FINAL_SCENES
        for station in STATION_IDS
        for role in ("MAP", "QUERY")
    }
    final_raw_map = _unique(final_raw, _bag_key, final_bag_keys, "final raw bag")

    scene_registry = payload.get("final_scene_registry")
    if not isinstance(scene_registry, Mapping):
        _fail("final_scene_registry must be a mapping")
    scenes = _rows(scene_registry, "scenes")
    scene_map = _unique(
        scenes, lambda row: str(row.get("scene_id")), set(FINAL_SCENES), "final scene"
    )
    for scene in FINAL_SCENES:
        expected_class = "RICH" if scene in RICH_SCENES else "WEAK"
        row = scene_map[scene]
        if row.get("final_geometry_class") != expected_class:
            _fail(f"{scene} final geometry class must be {expected_class}")
        if row.get("geometry_admission_status") != "GEOMETRY_ADMITTED":
            _fail(f"{scene} must be GEOMETRY_ADMITTED")

    station_registry = payload.get("final_station_registry")
    if not isinstance(station_registry, Mapping):
        _fail("final_station_registry must be a mapping")
    stations = _rows(station_registry, "stations")
    final_station_keys = {(scene, station) for scene in FINAL_SCENES for station in STATION_IDS}
    _unique(stations, _station_key, final_station_keys, "final station")
    if any(_status(row) not in {"PASS", "ACQUISITION_PASS"} for row in stations):
        _fail("all final stations must pass acquisition")

    targets = [_normalize_target(row) for row in _rows(payload, "final_targets")]
    target_map = _unique(targets, _station_key, final_station_keys, "final target")
    for key, target in target_map.items():
        map_bag = final_raw_map[(key[0], key[1], "MAP")]
        if target["map_bag_sha256"] != map_bag["sha256"]:
            _fail(f"{key} target MAP bag SHA binding differs")

    snapshots = [_normalize_snapshot(row) for row in _rows(payload, "final_snapshots")]
    snapshot_keys = {
        (scene, station, selection)
        for scene in FINAL_SCENES
        for station in STATION_IDS
        for selection in range(10)
    }
    snapshot_map = _unique(snapshots, _snapshot_key, snapshot_keys, "final snapshot")
    snapshot_ids = [row["snapshot_id"] for row in snapshots]
    if len(snapshot_ids) != len(set(snapshot_ids)):
        _fail("final snapshot_id values must be unique")
    for key, snapshot in snapshot_map.items():
        expected_quantile = QUERY_QUANTILES[key[2]]
        if "quantile" in snapshot and not math.isclose(
            _finite(snapshot["quantile"], "snapshot.quantile"),
            expected_quantile,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail(f"{key} snapshot quantile changed")
        if snapshot["target_npy_sha256"] != target_map[key[:2]]["target_npy_sha256"]:
            _fail(f"{key} snapshot target SHA binding differs")

    metrics = [_normalize_metric(row) for row in _rows(payload, "final_geometry_metrics")]
    metric_map = _unique(metrics, _metric_key, snapshot_keys, "final geometry metric")
    for key in snapshot_keys:
        if metric_map[key]["snapshot_id"] != snapshot_map[key]["snapshot_id"]:
            _fail(f"{key} geometry/snapshot ID binding differs")

    declared_geometry = _rows(payload, "final_geometry_scenes")
    declared_geometry_map = _unique(
        declared_geometry,
        lambda row: str(row.get("scene_id")),
        set(FINAL_SCENES),
        "final geometry scene",
    )
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_scene[str(row["scene_id"])].append(row)
    for scene in FINAL_SCENES:
        expected_class = "RICH" if scene in RICH_SCENES else "WEAK"
        recomputed = _scene_geometry(scene, by_scene[scene], expected_class)
        declared = declared_geometry_map[scene]
        for field in (
            "median_normalized_lambda_min_trans",
            "median_condition_number_trans",
            "median_spectral_entropy_trans",
        ):
            if not math.isclose(
                _finite(declared.get(field), f"{scene}.{field}"),
                float(recomputed[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                _fail(f"{scene} declared geometry median changed: {field}")

    rejected = _rows(payload, "rejected_candidates")
    rejected_keys = {(REJECTED_SCENE, station, role) for station in STATION_IDS for role in ("MAP", "QUERY")}
    rejected_map = _unique(rejected, _bag_key, rejected_keys, "rejected candidate")
    candidate_map = {_bag_key(row): row for row in candidate_raw}
    for key, row in rejected_map.items():
        if row.get("candidate_status") != "GEOMETRY_REJECTED":
            _fail("W02 rejected candidate status changed")
        if row.get("exclusion_reason") != "GEOMETRY_ONLY_INELIGIBLE":
            _fail("W02 exclusion reason changed")
        if row.get("replacement_scene_id") != REPLACEMENT_SCENE:
            _fail("W02 replacement scene changed")
        if row.get("rejected_before_icp") is not True or row.get("retained") is not True:
            _fail("W02 must be retained and rejected before ICP")
        if row.get("included_in_final_set") is not False:
            _fail("W02 must not enter the final set")
        if row["sha256"] != candidate_map[key]["sha256"]:
            _fail(f"rejected candidate SHA binding changed: {key}")

    lineage = payload.get("replacement_lineage")
    if not isinstance(lineage, Mapping):
        _fail("replacement_lineage must be a mapping")
    lineage_expected = {
        "rejected_candidate_scene_id": REJECTED_SCENE,
        "candidate_status": "GEOMETRY_REJECTED",
        "rejection_reason": "GEOMETRY_ONLY_INELIGIBLE",
        "replacement_scene_id": REPLACEMENT_SCENE,
        "W02_RETAINED": True,
        "W02_INCLUDED_IN_FINAL_SET": False,
        "W02_REJECTED_BEFORE_ANY_ICP": True,
        "W04_ADMISSION_OCCURRED_BEFORE_ANY_ICP": True,
        "formal_trial_count_at_decision": 0,
        "formal_trial_count_at_w04_admission": 0,
        "replacement_decision_used_registration_evidence": False,
        "w02_raw_bag_count": 6,
        "w02_snapshot_count_excluded": 30,
        "w04_raw_bag_count": 6,
        "w04_snapshot_count_admitted": 30,
    }
    for key, expected in lineage_expected.items():
        if lineage.get(key) != expected:
            _fail(f"replacement_lineage.{key} must be {expected!r}")

    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        _fail("readiness must be a mapping")
    readiness_expected = {
        "FMB1_W04_ACQUISITION_PASS": True,
        "FMB1_W04_FINAL_GEOMETRY_CLASS": "WEAK",
        "FMB1_W04_ADMISSION_PASS": True,
        "FMB1_FINAL_DATASET_READY": True,
        "FINAL_RICH_SCENE_COUNT": 3,
        "FINAL_WEAK_SCENE_COUNT": 3,
        "FINAL_SCENE_COUNT": 6,
        "FINAL_STATION_COUNT": 18,
        "FINAL_SNAPSHOT_COUNT": 180,
        "RICH_SNAPSHOT_COUNT": 90,
        "WEAK_SNAPSHOT_COUNT": 90,
        "RAW_CANDIDATE_BAG_COUNT": 42,
        "FINAL_ADMITTED_BAG_COUNT": 36,
        "REJECTED_BAG_COUNT": 6,
        "W02_RETAINED": True,
        "W02_INCLUDED_IN_FINAL_SET": False,
        "W04_INCLUDED_IN_FINAL_SET": True,
        "W02_SNAPSHOT_COUNT_IN_FINAL_SET": 0,
        "FINAL_SNAPSHOT_DUPLICATE_COUNT": 0,
        "FINAL_SNAPSHOT_MISSING_COUNT": 0,
        "FINAL_SNAPSHOT_ORPHAN_COUNT": 0,
        "FINAL_SNAPSHOT_WRONG_SCENE_COUNT": 0,
        "FINAL_SNAPSHOT_WRONG_STATION_COUNT": 0,
        "QUERY_CONTRIBUTION_TO_TARGET_TOTAL": 0,
        "READY_FOR_ZERO_PERTURBATION_AMENDMENT_ACTIVATION": True,
        "ZERO_PERTURBATION_MAINLINE_AMENDMENT_STATUS": "PROPOSED_NOT_ACTIVE",
        "PROPOSED_AMENDMENT_ACTIVE": False,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "MEASUREMENT_FINAL_RESULT": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "actual_registration_trials": 0,
        "registration_execution_count": 0,
    }
    for key, expected in readiness_expected.items():
        if readiness.get(key) != expected:
            _fail(f"readiness.{key} must be {expected!r}")

    attestation = payload.get("no_icp_attestation")
    if not isinstance(attestation, Mapping):
        _fail("no_icp_attestation must be a mapping")
    if attestation.get("NO_ICP_ATTESTATION_PASS") is not True or attestation.get("pass") is not True:
        _fail("NO_ICP attestation must pass")
    for key in _ZERO_COUNTERS:
        if attestation.get(key, 0) != 0:
            _fail(f"no_icp_attestation.{key} must be 0")
    for key in ("FORMAL_LOCK_ISSUED", "FORMAL_ICP_UNLOCKED", "FORMAL_REGISTRATION_AUTHORIZED"):
        if attestation.get(key) is not False:
            _fail(f"no_icp_attestation.{key} must be false")

    if verify_files:
        for index, row in enumerate(candidate_raw):
            _verify_declared_file(
                row["raw_absolute_path"], row["sha256"], f"candidate_raw[{index}]"
            )
        for index, row in enumerate(targets):
            _verify_declared_file(
                row["target_path"], row["target_npy_sha256"], f"target[{index}]"
            )
        for index, row in enumerate(snapshots):
            _verify_declared_file(
                row["source_path"], row["source_npy_sha256"], f"snapshot[{index}]"
            )

    return {
        "schema": "mid360_fmb1_final_dataset_payload_verification_v1",
        "PASS": True,
        "scene_count": 6,
        "station_count": 18,
        "snapshot_count": 180,
        "raw_candidate_bag_count": 42,
        "final_admitted_bag_count": 36,
        "rejected_bag_count": 6,
        "w02_in_final_snapshot_count": 0,
        "query_contribution_to_target_total": 0,
        "formal_trial_count": 0,
        "files_rehashed": bool(verify_files),
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for raw_row in rows:
        row = dict(raw_row)
        for key, value in list(row.items()):
            if isinstance(value, (Mapping, list, tuple)):
                row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            _fail(f"refusing to overwrite different final dataset artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        _fail(f"stale temporary output exists: {temporary}")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_sha256sums(output_dir: Path) -> Path:
    """Atomically refresh hashes after an independent verifier writes its report."""

    output_dir = output_dir.resolve(strict=True)
    manifest = output_dir / "SHA256SUMS"
    if manifest.is_file():
        for line in manifest.read_text(encoding="ascii").splitlines():
            if not line:
                continue
            digest, name = line.split("  ", 1)
            path = output_dir / name
            if path.is_file() and _sha256_file(path) != digest:
                _fail(f"existing final artifact changed before SHA refresh: {name}")
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS"
        and not path.name.endswith(".tmp")
    )
    content = "".join(f"{_sha256_file(path)}  {path.name}\n" for path in files).encode("ascii")
    temporary = output_dir / "SHA256SUMS.tmp"
    if temporary.exists():
        _fail(f"stale checksum temporary exists: {temporary}")
    temporary.write_bytes(content)
    temporary.replace(manifest)
    return manifest


def write_final_dataset(payload: Mapping[str, Any], output_dir: Path) -> list[Path]:
    """Write the required ``final_dataset_v1`` artifacts idempotently."""

    verify_final_dataset_payload(payload, verify_files=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, bytes] = {
        "final_dataset_manifest.json": _canonical_json(payload),
        "final_scene_registry.yaml": _canonical_json(payload["final_scene_registry"]),
        "final_station_registry.yaml": _canonical_json(payload["final_station_registry"]),
        "replacement_lineage.json": _canonical_json(payload["replacement_lineage"]),
        "final_dataset_readiness.json": _canonical_json(payload["readiness"]),
        "NO_ICP_ATTESTATION.json": _canonical_json(payload["no_icp_attestation"]),
    }
    csv_specs = {
        "final_raw_bag_manifest.csv": (
            payload["final_raw_bags"],
            (
                "scene_id", "station_id", "role", "raw_filename",
                "raw_absolute_path", "canonical_filename", "capture_prefix",
                "sha256", "bytes", "start_timestamp", "end_timestamp",
            ),
        ),
        "final_target_manifest.csv": (
            payload["final_targets"],
            (
                "scene_id", "station_id", "map_bag_sha256", "map_frame_count",
                "raw_point_count", "filtered_point_count", "target_point_count",
                "target_path", "target_npy_sha256", "target_array_sha256",
                "target_size_bytes",
                "input_roles", "query_frame_count", "query_contribution_to_target",
                "construction", "registration_called", "odometry_called",
                "scan_matching_called",
            ),
        ),
        "final_snapshot_manifest.csv": (
            payload["final_snapshots"],
            (
                "scene_id", "station_id", "snapshot_id", "selection_index",
                "quantile", "query_frame_index", "query_timestamp",
                "query_bag_sha256", "source_path", "source_npy_sha256",
                "source_array_sha256", "source_point_count",
                "source_size_bytes",
                "target_npy_sha256", "target_array_sha256",
                "selection_method", "selection_frozen_before_registration",
            ),
        ),
        "final_geometry_manifest.csv": (
            payload["final_geometry_metrics"],
            ("scene_id", "station_id", "snapshot_id", "selection_index", *GEOMETRY_ONLY_FIELDS),
        ),
        "rejected_candidate_manifest.csv": (
            payload["rejected_candidates"],
            (
                "scene_id", "station_id", "role", "raw_filename",
                "raw_absolute_path", "sha256", "candidate_status",
                "bytes",
                "final_geometry_class", "original_geometry_failure_reason",
                "exclusion_reason", "replacement_scene_id", "rejected_before_icp",
                "retained", "included_in_final_set",
            ),
        ),
    }
    for name, (rows, fields) in csv_specs.items():
        artifacts[name] = _csv_bytes(rows, fields)
    written: list[Path] = []
    for name, content in artifacts.items():
        path = output_dir / name
        _write_once(path, content)
        written.append(path)
    written.append(write_sha256sums(output_dir))
    return sorted(written)


__all__ = [
    "FINAL_SCENES",
    "ORIGINAL_SCENES",
    "W04FinalDatasetError",
    "build_final_dataset_payload",
    "verify_final_dataset_payload",
    "write_final_dataset",
    "write_sha256sums",
]
