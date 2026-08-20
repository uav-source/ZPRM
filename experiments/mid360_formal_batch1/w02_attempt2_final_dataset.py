"""Versioned administrative correction for the FMB1 W02 reacquisition.

This producer preserves the earlier W04-labelled evidence as historical input,
but the active dataset is rebuilt with the clean R01/R02/R03/W01/W02/W03
identity.  The old W02 capture remains byte-authenticated as invalid attempt 1;
the 2026-08-20 capture is admitted as attempt 2.  No backend is imported.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .protocol import BACKEND_CONTRACT_SHA256, QUERY_QUANTILES, geometry_class
from .w02_attempt2 import (
    CORRECTION_REASON,
    INVALID_ATTEMPT_REASON,
    as_w02_attempt2,
    as_w04_compatibility,
)
from .w04_final_dataset import build_final_dataset_payload


FINAL_SCENES = (
    "FMB1_R01",
    "FMB1_R02",
    "FMB1_R03",
    "FMB1_W01",
    "FMB1_W02",
    "FMB1_W03",
)
RICH_SCENES = frozenset(FINAL_SCENES[:3])
WEAK_SCENES = frozenset(FINAL_SCENES[3:])
STATIONS = ("S01", "S02", "S03")
ROLES = ("MAP", "QUERY")
ATTEMPT1_PREFIXES = {
    "S01": "20260819_213446",
    "S02": "20260819_213744",
    "S03": "20260819_214044",
}
ATTEMPT2_PREFIXES = {
    "S01": "20260820_081749",
    "S02": "20260820_081954",
    "S03": "20260820_082207",
}


class W02CorrectionError(RuntimeError):
    """A correction, archive, or final-dataset invariant failed."""


def _fail(message: str) -> None:
    raise W02CorrectionError(f"FMB1_W02_CORRECTION_INVALID: {message}")


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(value: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    rows = value.get(key)
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        _fail(f"{key} must be a list of objects")
    return [copy.deepcopy(dict(row)) for row in rows]


def _attempt1_name(station: str, role: str) -> str:
    part = "part1_20s" if role == "MAP" else "part2_15s"
    return f"mid360_{ATTEMPT1_PREFIXES[station]}_{part}.bag"


def archive_w02_attempt1(
    original_manifest: Mapping[str, Any],
    *,
    repository: Path,
    archive_dir: Path,
) -> dict[str, Any]:
    """Move the six invalid raw bags out of the active directory, preserving bytes."""

    root = repository.resolve(strict=True)
    active_dir = (root / "bags").resolve(strict=True)
    archive_dir = archive_dir.expanduser().resolve()
    expected_archive_parent = (
        root / "archive/invalid_acquisition/FMB1_W02_attempt1_wrong_location"
    ).resolve()
    if archive_dir != expected_archive_parent:
        _fail(f"unexpected archive target: {archive_dir}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    if archive_dir.is_symlink():
        _fail("archive directory must not be a symlink")

    candidates = [
        row
        for row in _rows(original_manifest, "raw_bags")
        if row.get("scene_id") == "FMB1_W02"
    ]
    if len(candidates) != 6:
        _fail("original W02 attempt 1 must contain exactly six bags")
    expected = {
        (station, role): _attempt1_name(station, role)
        for station in STATIONS
        for role in ROLES
    }
    actual = {
        (str(row.get("station_id")), str(row.get("role"))): str(
            row.get("raw_filename")
        )
        for row in candidates
    }
    if actual != expected:
        _fail(f"attempt-1 filename mapping changed: {actual}")

    prepared: list[tuple[Path, Path, dict[str, Any]]] = []
    for row in sorted(candidates, key=lambda item: (item["station_id"], item["role"])):
        name = str(row["raw_filename"])
        source = active_dir / name
        destination = archive_dir / name
        if source.is_symlink() or destination.is_symlink():
            _fail(f"raw archive endpoints must not be symlinks: {name}")
        source_exists = source.is_file()
        destination_exists = destination.is_file()
        if source_exists == destination_exists:
            _fail(
                f"exactly one active/archive copy must exist before move: {name}"
            )
        current = source if source_exists else destination
        declared_sha = str(row.get("sha256"))
        actual_sha = _sha256(current)
        if actual_sha != declared_sha:
            _fail(f"attempt-1 SHA changed: {name}")
        declared_bytes = int(row.get("bytes", current.stat().st_size))
        if current.stat().st_size != declared_bytes:
            _fail(f"attempt-1 byte count changed: {name}")
        prepared.append((source, destination, row))

    # Every endpoint and byte binding has been validated before the first move.
    for source, destination, _ in prepared:
        if source.is_file():
            os.replace(source, destination)

    archived: list[dict[str, Any]] = []
    for source, destination, row in prepared:
        if source.exists() or not destination.is_file() or destination.is_symlink():
            _fail(f"attempt-1 move did not finish cleanly: {destination.name}")
        if _sha256(destination) != row["sha256"]:
            _fail(f"attempt-1 SHA changed during archive move: {destination.name}")
        archived.append(
            {
                "scene_id": "FMB1_W02",
                "station_id": row["station_id"],
                "role": row["role"],
                "attempt": 1,
                "attempt_status": "INVALID_ACQUISITION",
                "invalidation_reason": INVALID_ATTEMPT_REASON,
                "raw_filename": row["raw_filename"],
                "original_active_path": str(source),
                "archived_absolute_path": str(destination),
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "retained": True,
                "included_in_active_dataset": False,
                "included_in_geometry_summary": False,
                "included_in_formal_trials": False,
            }
        )
    return {
        "schema": "mid360_fmb1_invalid_acquisition_archive_v1",
        "scene_id": "FMB1_W02",
        "attempt": 1,
        "status": "INVALID_ACQUISITION",
        "reason": INVALID_ATTEMPT_REASON,
        "operator_confirmation": CORRECTION_REASON,
        "correction_before_any_icp": True,
        "formal_trial_count_at_correction": 0,
        "failed_attempts_retained": True,
        "failed_raw_data_retained": True,
        "archive_dir": str(archive_dir),
        "archived_bag_count": 6,
        "bags": archived,
    }


def _update_original_paths(
    original: Mapping[str, Any], archive: Mapping[str, Any]
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(original))
    paths = {
        str(row["raw_filename"]): str(row["archived_absolute_path"])
        for row in archive["bags"]
    }
    for row in payload.get("raw_bags", []):
        if row.get("scene_id") == "FMB1_W02":
            new_path = paths[str(row["raw_filename"])]
            row["raw_absolute_path"] = new_path
            inventory = row.get("inventory")
            if isinstance(inventory, dict):
                inventory["bag_path"] = new_path
    for row in payload.get("mapping", []):
        if row.get("scene_id") == "FMB1_W02":
            row["raw_absolute_path"] = paths[str(row["raw_filename"])]
    return payload


def _assign_attempt(row: dict[str, Any]) -> dict[str, Any]:
    scene = str(row.get("scene_id"))
    name = str(row.get("raw_filename", ""))
    if scene == "FMB1_W02" and "20260819_" in name:
        row["attempt"] = 1
    elif scene == "FMB1_W02":
        row["attempt"] = 2
    else:
        row["attempt"] = 1
    return row


def _relabel_base_payload(
    base: Mapping[str, Any], archive: Mapping[str, Any]
) -> dict[str, Any]:
    payload = as_w02_attempt2(base)
    payload["schema"] = "mid360_fmb1_w02_attempt2_corrected_final_dataset_v1"
    archived_by_name = {
        str(row["raw_filename"]): row for row in archive["bags"]
    }

    candidate_rows = []
    for raw in payload["raw_candidate_bags"]:
        row = _assign_attempt(copy.deepcopy(dict(raw)))
        if row["scene_id"] == "FMB1_W02" and row["attempt"] == 1:
            archived = archived_by_name[str(row["raw_filename"])]
            row["raw_absolute_path"] = archived["archived_absolute_path"]
            row["attempt_status"] = "INVALID_ACQUISITION"
            row["invalidation_reason"] = INVALID_ATTEMPT_REASON
            row["retained"] = True
            row["included_in_active_dataset"] = False
        else:
            row["attempt_status"] = "VALID_ACQUISITION"
            row["included_in_active_dataset"] = True
        candidate_rows.append(row)
    payload["raw_candidate_bags"] = sorted(
        candidate_rows,
        key=lambda row: (
            row["scene_id"],
            int(row["attempt"]),
            row["station_id"],
            row["role"],
        ),
    )

    for key in (
        "final_raw_bags",
        "final_targets",
        "final_snapshots",
        "final_geometry_metrics",
        "final_geometry_scenes",
    ):
        for row in payload[key]:
            row["attempt"] = 2 if row["scene_id"] == "FMB1_W02" else 1
            if key == "final_targets" and row.get("input_roles") == "MAP":
                row["input_roles"] = ["MAP"]
    for row in payload["final_scene_registry"]["scenes"]:
        row["attempt"] = 2 if row["scene_id"] == "FMB1_W02" else 1
    for row in payload["final_station_registry"]["stations"]:
        row["attempt"] = 2 if row["scene_id"] == "FMB1_W02" else 1
    scene_order = {scene: index for index, scene in enumerate(FINAL_SCENES)}
    payload["final_scene_registry"]["scenes"].sort(
        key=lambda row: scene_order[row["scene_id"]]
    )
    payload["final_station_registry"]["stations"].sort(
        key=lambda row: (scene_order[row["scene_id"]], row["station_id"])
    )
    payload["final_raw_bags"].sort(
        key=lambda row: (
            scene_order[row["scene_id"]], row["station_id"], row["role"]
        )
    )
    payload["final_targets"].sort(
        key=lambda row: (scene_order[row["scene_id"]], row["station_id"])
    )
    payload["final_snapshots"].sort(
        key=lambda row: (
            scene_order[row["scene_id"]],
            row["station_id"],
            int(row["selection_index"]),
        )
    )
    payload["final_geometry_metrics"].sort(
        key=lambda row: (
            scene_order[row["scene_id"]],
            row["station_id"],
            int(row["selection_index"]),
        )
    )
    payload["final_geometry_scenes"].sort(
        key=lambda row: scene_order[row["scene_id"]]
    )

    rejected = []
    for archived in archive["bags"]:
        rejected.append(
            {
                **copy.deepcopy(archived),
                "candidate_status": "INVALID_ACQUISITION",
                "acquisition_status": "INVALID_ACQUISITION",
                "exclusion_reason": "WRONG_SCENE_LOCATION",
                "operator_error": "OPERATOR_SCENE_SELECTION_ERROR",
                "reacquired_scene_id": "FMB1_W02",
                "reacquired_attempt": 2,
                "historical_geometry_evidence_retained_but_superseded": True,
                "correction_before_any_icp": True,
            }
        )
    payload["rejected_candidates"] = sorted(
        rejected, key=lambda row: (row["station_id"], row["role"])
    )
    payload["acquisition_attempt_lineage"] = {
        "schema": "mid360_fmb1_w02_acquisition_attempt_lineage_v1",
        "correction_type": "PRE_BACKEND_ADMINISTRATIVE_ACQUISITION_CORRECTION",
        "correction_reason": CORRECTION_REASON,
        "correction_before_any_icp": True,
        "formal_trial_count_at_correction": 0,
        "scene_id": "FMB1_W02",
        "invalid_attempt": 1,
        "invalid_attempt_status": "INVALID_ACQUISITION",
        "invalid_attempt_reason": INVALID_ATTEMPT_REASON,
        "valid_attempt": 2,
        "valid_attempt_status": "GEOMETRY_ADMITTED",
        "valid_attempt_final_geometry_class": "WEAK",
        "failed_attempts_retained": True,
        "failed_raw_data_retained": True,
        "invalid_attempt_raw_bag_count": 6,
        "valid_attempt_raw_bag_count": 6,
        "invalid_attempt_snapshot_count_in_final": 0,
        "valid_attempt_snapshot_count_in_final": 30,
        "W04_IDENTIFIER_RETIRED": True,
        "W04_INCLUDED_IN_FINAL_SET": False,
        "registration_evidence_used_for_correction": False,
    }
    payload.pop("replacement_lineage", None)

    readiness = payload["readiness"]
    for key in list(readiness):
        if "W04" in key or key in {
            "W02_INCLUDED_IN_FINAL_SET",
            "W02_SNAPSHOT_COUNT_IN_FINAL_SET",
        }:
            readiness.pop(key)
    readiness.update(
        {
            "schema": "mid360_fmb1_w02_attempt2_final_readiness_v1",
            "FMB1_W02_ATTEMPT2_ACQUISITION_PASS": True,
            "FMB1_W02_ATTEMPT2_FINAL_GEOMETRY_CLASS": "WEAK",
            "FMB1_W02_ATTEMPT2_ADMISSION_PASS": True,
            "FMB1_W02_ATTEMPT1_STATUS": "INVALID_ACQUISITION",
            "FMB1_W02_ATTEMPT1_RETAINED": True,
            "FMB1_W02_ATTEMPT1_INCLUDED_IN_FINAL_SET": False,
            "FMB1_W02_ATTEMPT2_INCLUDED_IN_FINAL_SET": True,
            "FMB1_W02_ATTEMPT1_SNAPSHOT_COUNT_IN_FINAL_SET": 0,
            "FMB1_W02_ATTEMPT2_SNAPSHOT_COUNT_IN_FINAL_SET": 30,
            "W04_IDENTIFIER_RETIRED": True,
            "W04_INCLUDED_IN_FINAL_SET": False,
            "CORRECTION_REASON": CORRECTION_REASON,
            "CORRECTION_BEFORE_ANY_ICP": True,
            "FORMAL_TRIAL_COUNT_AT_CORRECTION": 0,
            "failed_attempts_retained": True,
            "failed_raw_data_retained": True,
            "READY_FOR_ZERO_PERTURBATION_AMENDMENT_ACTIVATION": False,
            "ZERO_PERTURBATION_MAINLINE_AMENDMENT_STATUS": "NOT_REQUIRED_NUMBERING_CORRECTION",
        }
    )
    payload["invalid_acquisition_archive"] = copy.deepcopy(dict(archive))
    return payload


def build_corrected_final_dataset(
    original_manifest: Mapping[str, Any],
    attempt2_acquisition: Mapping[str, Any],
    attempt2_assets: Mapping[str, Any],
    attempt2_geometry: Mapping[str, Any],
    archive_manifest: Mapping[str, Any],
    no_icp_attestation: Mapping[str, Any],
    *,
    repository: Path,
) -> dict[str, Any]:
    """Compose the corrected dataset after strict compatibility validation."""

    original_compat = _update_original_paths(original_manifest, archive_manifest)
    source_attestation = copy.deepcopy(dict(no_icp_attestation))
    source_attestation.setdefault("FORMAL_LOCK_ISSUED", False)
    source_attestation.setdefault("FORMAL_ICP_UNLOCKED", False)
    plan = {
        "rejected_candidate_scene_id": "FMB1_W02",
        "rejection_reason": "GEOMETRY_ONLY_INELIGIBLE",
        "replacement_scene_id": "FMB1_W04",
        "semantic_candidate_label": "WEAK_CANDIDATE",
        "decision_before_any_icp": True,
        "formal_trial_count_at_decision": 0,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
    }
    base = build_final_dataset_payload(
        original_compat,
        as_w04_compatibility(attempt2_acquisition),
        as_w04_compatibility(attempt2_assets),
        as_w04_compatibility(attempt2_geometry),
        plan,
        source_attestation,
        repository=repository,
        verify_files=True,
    )
    payload = _relabel_base_payload(base, archive_manifest)
    verify_corrected_final_dataset(payload, repository=repository, verify_files=True)
    return payload


def _unique(rows: Iterable[Mapping[str, Any]], key_fn: Any, label: str) -> dict[Any, Mapping[str, Any]]:
    output: dict[Any, Mapping[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        if key in output:
            _fail(f"duplicate {label}: {key}")
        output[key] = row
    return output


def _assert_zero_control(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in {
                "open3d_registration_call_count",
                "pcl_cli_invocation_count",
                "other_registration_process_count",
                "formal_trial_count",
                "actual_open3d_trials",
                "actual_pcl_trials",
                "actual_formal_trials",
                "actual_registration_trials",
                "registration_execution_count",
            } and child != 0:
                _fail(f"nonzero pre-backend counter: {path}.{key}")
            if normalized in {
                "formal_lock_issued",
                "formal_icp_unlocked",
                "formal_registration_authorized",
                "measurement_final_result",
            } and child is not False:
                _fail(f"pre-backend control must be false: {path}.{key}")
            _assert_zero_control(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_zero_control(child, f"{path}[{index}]")


def verify_corrected_final_dataset(
    payload: Mapping[str, Any], *, repository: Path, verify_files: bool = True
) -> dict[str, Any]:
    if payload.get("schema") != "mid360_fmb1_w02_attempt2_corrected_final_dataset_v1":
        _fail("unexpected corrected dataset schema")
    if payload.get("backend_parameter_contract_sha256") != BACKEND_CONTRACT_SHA256:
        _fail("backend parameter contract binding changed")
    if _sha256(repository / "frozen_assets/backend_parameter_contract.json") != BACKEND_CONTRACT_SHA256:
        _fail("live backend parameter contract changed")
    _assert_zero_control(payload)

    candidates = _rows(payload, "raw_candidate_bags")
    if len(candidates) != 42:
        _fail("raw candidate bag count must be 42")
    candidate_map = _unique(
        candidates,
        lambda row: (
            row["scene_id"],
            int(row["attempt"]),
            row["station_id"],
            row["role"],
        ),
        "candidate bag",
    )
    expected_candidate_keys = {
        (scene, 1, station, role)
        for scene in FINAL_SCENES
        for station in STATIONS
        for role in ROLES
    } | {
        ("FMB1_W02", 2, station, role)
        for station in STATIONS
        for role in ROLES
    }
    if set(candidate_map) != expected_candidate_keys:
        _fail("candidate attempt inventory differs")

    final_raw = _rows(payload, "final_raw_bags")
    final_raw_map = _unique(
        final_raw,
        lambda row: (row["scene_id"], row["station_id"], row["role"]),
        "final raw bag",
    )
    expected_final_bags = {
        (scene, station, role)
        for scene in FINAL_SCENES
        for station in STATIONS
        for role in ROLES
    }
    if set(final_raw_map) != expected_final_bags or len(final_raw) != 36:
        _fail("final raw inventory must be clean 6x3x2")
    for key, row in final_raw_map.items():
        expected_attempt = 2 if key[0] == "FMB1_W02" else 1
        if row.get("attempt") != expected_attempt:
            _fail(f"wrong final attempt for {key}")

    scenes = _rows(payload["final_scene_registry"], "scenes")
    scene_map = _unique(scenes, lambda row: row["scene_id"], "scene")
    if set(scene_map) != set(FINAL_SCENES):
        _fail("final scene registry must contain R01/R02/R03/W01/W02/W03")
    if any("W04" in str(value) for value in scene_map):
        _fail("W04 must not remain in the scene registry")
    for scene, row in scene_map.items():
        expected_class = "RICH" if scene in RICH_SCENES else "WEAK"
        if row.get("final_geometry_class") != expected_class:
            _fail(f"{scene} must be {expected_class}")
        if row.get("geometry_admission_status") != "GEOMETRY_ADMITTED":
            _fail(f"{scene} must be admitted")

    stations = _rows(payload["final_station_registry"], "stations")
    if len(stations) != 18:
        _fail("final station count must be 18")
    station_keys = _unique(
        stations, lambda row: (row["scene_id"], row["station_id"]), "station"
    )
    expected_station_keys = {
        (scene, station) for scene in FINAL_SCENES for station in STATIONS
    }
    if set(station_keys) != expected_station_keys:
        _fail("final station identity differs")

    targets = _rows(payload, "final_targets")
    target_map = _unique(
        targets, lambda row: (row["scene_id"], row["station_id"]), "target"
    )
    if set(target_map) != expected_station_keys or len(targets) != 18:
        _fail("final target inventory differs")
    for key, row in target_map.items():
        if row.get("input_roles") != ["MAP"]:
            _fail(f"target is not MAP-only: {key}")
        if row.get("query_contribution_to_target") != 0:
            _fail(f"QUERY entered target: {key}")
        if row.get("map_bag_sha256") != final_raw_map[(key[0], key[1], "MAP")]["sha256"]:
            _fail(f"target MAP lineage changed: {key}")

    snapshots = _rows(payload, "final_snapshots")
    if len(snapshots) != 180:
        _fail("final snapshot count must be 180")
    snapshot_map = _unique(
        snapshots,
        lambda row: (row["scene_id"], row["station_id"], int(row["selection_index"])),
        "snapshot",
    )
    expected_snapshot_keys = {
        (scene, station, index)
        for scene in FINAL_SCENES
        for station in STATIONS
        for index in range(10)
    }
    if set(snapshot_map) != expected_snapshot_keys:
        _fail("final snapshot identity differs")
    for key, row in snapshot_map.items():
        if not math.isclose(float(row["quantile"]), QUERY_QUANTILES[key[2]], abs_tol=1e-12):
            _fail(f"snapshot quantile changed: {key}")
        if row["target_npy_sha256"] != target_map[key[:2]]["target_npy_sha256"]:
            _fail(f"snapshot target lineage changed: {key}")

    metrics = _rows(payload, "final_geometry_metrics")
    metric_map = _unique(
        metrics,
        lambda row: (row["scene_id"], row["station_id"], int(row["selection_index"])),
        "geometry metric",
    )
    if set(metric_map) != expected_snapshot_keys:
        _fail("final geometry row identity differs")
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_scene[str(row["scene_id"])].append(row)
    for scene in FINAL_SCENES:
        values = by_scene[scene]
        classification = geometry_class(
            median(float(row["normalized_lambda_min_trans"]) for row in values),
            median(float(row["condition_number_trans"]) for row in values),
            median(float(row["spectral_entropy_trans"]) for row in values),
        )
        expected_class = "RICH" if scene in RICH_SCENES else "WEAK"
        if classification != expected_class:
            _fail(f"recomputed geometry class differs for {scene}")

    rejected = _rows(payload, "rejected_candidates")
    if len(rejected) != 6:
        _fail("invalid attempt must retain six rejected raw bags")
    for row in rejected:
        if row.get("attempt") != 1 or row.get("candidate_status") != "INVALID_ACQUISITION":
            _fail("rejected attempt identity/status differs")
        if row.get("retained") is not True or row.get("included_in_formal_trials") is not False:
            _fail("invalid raw evidence retention differs")

    lineage = payload.get("acquisition_attempt_lineage", {})
    expected_lineage = {
        "correction_reason": CORRECTION_REASON,
        "correction_before_any_icp": True,
        "formal_trial_count_at_correction": 0,
        "invalid_attempt": 1,
        "valid_attempt": 2,
        "failed_attempts_retained": True,
        "failed_raw_data_retained": True,
        "W04_IDENTIFIER_RETIRED": True,
        "W04_INCLUDED_IN_FINAL_SET": False,
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            _fail(f"lineage.{key} changed")
    readiness = payload.get("readiness", {})
    if readiness.get("FMB1_FINAL_DATASET_READY") is not True:
        _fail("corrected final dataset is not ready")
    if readiness.get("W04_INCLUDED_IN_FINAL_SET") is not False:
        _fail("W04 must be retired")
    if readiness.get("FMB1_W02_ATTEMPT2_INCLUDED_IN_FINAL_SET") is not True:
        _fail("W02 attempt 2 must be active")
    if readiness.get("FMB1_W02_ATTEMPT1_INCLUDED_IN_FINAL_SET") is not False:
        _fail("W02 attempt 1 must be inactive")

    if verify_files:
        for row in candidates:
            path = Path(str(row["raw_absolute_path"])).resolve(strict=True)
            if path.is_symlink() or _sha256(path) != row["sha256"]:
                _fail(f"candidate raw file changed: {path}")
        for row in targets:
            path = Path(str(row["target_path"])).resolve(strict=True)
            if path.is_symlink() or _sha256(path) != row["target_npy_sha256"]:
                _fail(f"target file changed: {path}")
        for row in snapshots:
            path = Path(str(row["source_path"])).resolve(strict=True)
            if path.is_symlink() or _sha256(path) != row["source_npy_sha256"]:
                _fail(f"source file changed: {path}")
    return {
        "schema": "mid360_fmb1_w02_attempt2_payload_verification_v1",
        "status": "PASS",
        "pass": True,
        "raw_candidate_bag_count": 42,
        "final_admitted_bag_count": 36,
        "invalid_attempt_bag_count": 6,
        "scene_count": 6,
        "station_count": 18,
        "target_count": 18,
        "snapshot_count": 180,
        "w02_attempt1_final_snapshot_count": 0,
        "w02_attempt2_final_snapshot_count": 30,
        "W04_INCLUDED_IN_FINAL_SET": False,
        "files_rehashed": verify_files,
        "actual_formal_trials": 0,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for raw in rows:
        row = dict(raw)
        for key, value in list(row.items()):
            if isinstance(value, (Mapping, list, tuple)):
                row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            _fail(f"refusing to overwrite different corrected artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        _fail(f"stale temporary artifact exists: {temporary}")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_corrected_final_dataset(
    payload: Mapping[str, Any], output_dir: Path, *, repository: Path
) -> list[Path]:
    verify_corrected_final_dataset(payload, repository=repository, verify_files=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "final_dataset_manifest.json": _canonical_json(payload),
        "final_scene_registry.yaml": _canonical_json(payload["final_scene_registry"]),
        "final_station_registry.yaml": _canonical_json(payload["final_station_registry"]),
        "acquisition_attempt_lineage.json": _canonical_json(
            payload["acquisition_attempt_lineage"]
        ),
        "invalid_attempt_archive_manifest.json": _canonical_json(
            payload["invalid_acquisition_archive"]
        ),
        "final_dataset_readiness.json": _canonical_json(payload["readiness"]),
        "NO_ICP_ATTESTATION.json": _canonical_json(payload["no_icp_attestation"]),
    }
    csv_specs = {
        "final_raw_bag_manifest.csv": (
            payload["final_raw_bags"],
            (
                "scene_id", "station_id", "attempt", "role", "raw_filename",
                "raw_absolute_path", "canonical_filename", "capture_prefix",
                "sha256", "bytes", "start_timestamp", "end_timestamp",
            ),
        ),
        "final_target_manifest.csv": (
            payload["final_targets"],
            (
                "scene_id", "station_id", "attempt", "map_bag_sha256",
                "map_frame_count", "raw_point_count", "filtered_point_count",
                "target_point_count", "target_path", "target_npy_sha256",
                "target_array_sha256", "target_size_bytes", "input_roles",
                "query_frame_count", "query_contribution_to_target", "construction",
            ),
        ),
        "final_snapshot_manifest.csv": (
            payload["final_snapshots"],
            (
                "scene_id", "station_id", "attempt", "snapshot_id",
                "selection_index", "quantile", "query_frame_index",
                "query_timestamp", "query_bag_sha256", "source_path",
                "source_npy_sha256", "source_array_sha256", "source_point_count",
                "source_size_bytes", "target_npy_sha256", "target_array_sha256",
            ),
        ),
        "final_geometry_manifest.csv": (
            payload["final_geometry_metrics"],
            tuple(payload["final_geometry_metrics"][0].keys()),
        ),
        "invalid_attempt_manifest.csv": (
            payload["rejected_candidates"],
            tuple(payload["rejected_candidates"][0].keys()),
        ),
    }
    for name, content in artifacts.items():
        _write_once(output_dir / name, content)
    for name, (rows, fields) in csv_specs.items():
        _write_once(output_dir / name, _csv_bytes(rows, fields))
    refresh_sha256sums(output_dir)
    return sorted(path for path in output_dir.iterdir() if path.is_file())


def refresh_sha256sums(output_dir: Path) -> Path:
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS" and not path.name.endswith(".tmp")
    )
    content = "".join(f"{_sha256(path)}  {path.name}\n" for path in files).encode("ascii")
    temporary = output_dir / "SHA256SUMS.tmp"
    temporary.write_bytes(content)
    temporary.replace(output_dir / "SHA256SUMS")
    return output_dir / "SHA256SUMS"


__all__ = [
    "ATTEMPT1_PREFIXES",
    "ATTEMPT2_PREFIXES",
    "FINAL_SCENES",
    "W02CorrectionError",
    "archive_w02_attempt1",
    "build_corrected_final_dataset",
    "refresh_sha256sums",
    "verify_corrected_final_dataset",
    "write_corrected_final_dataset",
]
