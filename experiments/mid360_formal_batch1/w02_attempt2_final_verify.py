"""Independent verifier for the corrected W02-attempt-2 final dataset.

This verifier intentionally does not import the producer or any registration
backend.  It rehashes every raw/target/source file, recomputes scene medians,
and binds the JSON/CSV/registry/archive/checksum evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping


BACKEND_SHA = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
SCENES = (
    "FMB1_R01",
    "FMB1_R02",
    "FMB1_R03",
    "FMB1_W01",
    "FMB1_W02",
    "FMB1_W03",
)
STATIONS = ("S01", "S02", "S03")
QUANTILES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


class W02IndependentVerificationError(RuntimeError):
    """Independent verification failed closed."""


def _fail(message: str) -> None:
    raise W02IndependentVerificationError(
        f"FMB1_W02_ATTEMPT2_FINAL_VERIFY_FAIL: {message}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        _fail(f"expected object: {path}")
    return value


def _rows(value: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    rows = value.get(key)
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        _fail(f"missing row list: {key}")
    return rows


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _classify(lambda_min: float, condition: float, entropy: float) -> str:
    if lambda_min >= 0.18 and condition <= 3.0 and entropy >= 0.90:
        return "RICH"
    if lambda_min <= 0.12 and condition >= 6.0 and entropy <= 0.80:
        return "WEAK"
    return "INTERMEDIATE"


def _verify_sha_file(output_dir: Path) -> int:
    path = output_dir / "SHA256SUMS"
    lines = [line for line in path.read_text(encoding="ascii").splitlines() if line]
    names = set()
    for line in lines:
        digest, name = line.split("  ", 1)
        if name in names:
            _fail(f"duplicate checksum entry: {name}")
        names.add(name)
        target = output_dir / name
        if not target.is_file() or _sha256(target) != digest:
            _fail(f"checksum mismatch: {name}")
    expected = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS" and not path.name.endswith(".tmp")
    }
    if names != expected:
        _fail("SHA256SUMS file inventory differs")
    return len(lines)


def _check_zero_controls(value: Any, path: str = "payload") -> None:
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
                _fail(f"nonzero counter: {path}.{key}")
            if normalized in {
                "formal_lock_issued",
                "formal_icp_unlocked",
                "formal_registration_authorized",
                "measurement_final_result",
            } and child is not False:
                _fail(f"control flag is not false: {path}.{key}")
            _check_zero_controls(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_zero_controls(child, f"{path}[{index}]")


def verify_w02_attempt2_final_dataset(
    repository: Path, output_dir: Path
) -> dict[str, Any]:
    root = repository.resolve(strict=True)
    output = output_dir.resolve(strict=True)
    manifest = _mapping(output / "final_dataset_manifest.json")
    if manifest.get("schema") != "mid360_fmb1_w02_attempt2_corrected_final_dataset_v1":
        _fail("unexpected manifest schema")
    if manifest.get("backend_parameter_contract_sha256") != BACKEND_SHA:
        _fail("manifest backend binding changed")
    if _sha256(root / "frozen_assets/backend_parameter_contract.json") != BACKEND_SHA:
        _fail("live backend contract changed")
    _check_zero_controls(manifest)

    scene_registry = _mapping(output / "final_scene_registry.yaml")
    station_registry = _mapping(output / "final_station_registry.yaml")
    archive = _mapping(output / "invalid_attempt_archive_manifest.json")
    lineage = _mapping(output / "acquisition_attempt_lineage.json")
    readiness = _mapping(output / "final_dataset_readiness.json")
    attestation = _mapping(output / "NO_ICP_ATTESTATION.json")
    if scene_registry != manifest.get("final_scene_registry"):
        _fail("scene registry is not bound to manifest")
    if station_registry != manifest.get("final_station_registry"):
        _fail("station registry is not bound to manifest")
    if archive != manifest.get("invalid_acquisition_archive"):
        _fail("archive registry is not bound to manifest")
    if lineage != manifest.get("acquisition_attempt_lineage"):
        _fail("attempt lineage is not bound to manifest")
    if readiness != manifest.get("readiness"):
        _fail("readiness is not bound to manifest")
    if attestation != manifest.get("no_icp_attestation"):
        _fail("NO-ICP attestation is not bound to manifest")

    scenes = _rows(scene_registry, "scenes")
    scene_ids = [str(row.get("scene_id")) for row in scenes]
    if set(scene_ids) != set(SCENES) or len(scene_ids) != 6:
        _fail("final scenes are not clean R01/R02/R03/W01/W02/W03")
    if any("W04" in scene for scene in scene_ids):
        _fail("W04 remains in the final scene registry")
    expected_classes = {
        scene: "RICH" if scene.startswith("FMB1_R") else "WEAK" for scene in SCENES
    }
    for row in scenes:
        scene = str(row["scene_id"])
        if row.get("final_geometry_class") != expected_classes[scene]:
            _fail(f"scene class changed: {scene}")
        if row.get("geometry_admission_status") != "GEOMETRY_ADMITTED":
            _fail(f"scene is not admitted: {scene}")

    candidates = _rows(manifest, "raw_candidate_bags")
    final_raw = _rows(manifest, "final_raw_bags")
    targets = _rows(manifest, "final_targets")
    snapshots = _rows(manifest, "final_snapshots")
    metrics = _rows(manifest, "final_geometry_metrics")
    rejected = _rows(manifest, "rejected_candidates")
    if (len(candidates), len(final_raw), len(targets), len(snapshots), len(metrics), len(rejected)) != (42, 36, 18, 180, 180, 6):
        _fail("final cardinalities changed")

    candidate_keys = {
        (row["scene_id"], int(row["attempt"]), row["station_id"], row["role"])
        for row in candidates
    }
    expected_candidate_keys = {
        (scene, 1, station, role)
        for scene in SCENES
        for station in STATIONS
        for role in ("MAP", "QUERY")
    } | {
        ("FMB1_W02", 2, station, role)
        for station in STATIONS
        for role in ("MAP", "QUERY")
    }
    if candidate_keys != expected_candidate_keys or len(candidate_keys) != 42:
        _fail("attempt-aware candidate inventory changed")
    final_raw_keys = {
        (row["scene_id"], row["station_id"], row["role"]): row for row in final_raw
    }
    expected_final_raw = {
        (scene, station, role)
        for scene in SCENES
        for station in STATIONS
        for role in ("MAP", "QUERY")
    }
    if set(final_raw_keys) != expected_final_raw or len(final_raw_keys) != 36:
        _fail("final raw inventory changed")
    for key, row in final_raw_keys.items():
        if row.get("attempt") != (2 if key[0] == "FMB1_W02" else 1):
            _fail(f"final attempt changed: {key}")

    target_map = {
        (row["scene_id"], row["station_id"]): row for row in targets
    }
    if len(target_map) != 18:
        _fail("target keys changed")
    for key, row in target_map.items():
        if row.get("input_roles") != ["MAP"] or row.get("query_contribution_to_target") != 0:
            _fail(f"target is not MAP-only: {key}")
        if row.get("map_bag_sha256") != final_raw_keys[(key[0], key[1], "MAP")]["sha256"]:
            _fail(f"target MAP lineage changed: {key}")

    snapshot_keys = set()
    per_scene = Counter()
    for row in snapshots:
        key = (row["scene_id"], row["station_id"], int(row["selection_index"]))
        if key in snapshot_keys:
            _fail(f"duplicate snapshot: {key}")
        snapshot_keys.add(key)
        per_scene[row["scene_id"]] += 1
        if not math.isclose(float(row["quantile"]), QUANTILES[key[2]], abs_tol=1e-12):
            _fail(f"snapshot quantile changed: {key}")
        if row["target_npy_sha256"] != target_map[key[:2]]["target_npy_sha256"]:
            _fail(f"snapshot target binding changed: {key}")
    if per_scene != Counter({scene: 30 for scene in SCENES}):
        _fail("snapshot scene balance changed")

    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    metric_keys = set()
    for row in metrics:
        key = (row["scene_id"], row["station_id"], int(row["selection_index"]))
        if key in metric_keys:
            _fail(f"duplicate geometry row: {key}")
        metric_keys.add(key)
        by_scene[str(row["scene_id"])].append(row)
    if metric_keys != snapshot_keys:
        _fail("geometry/snapshot identity differs")
    recomputed = {}
    for scene in SCENES:
        rows = by_scene[scene]
        values = {
            "normalized_lambda_min_trans": median(
                float(row["normalized_lambda_min_trans"]) for row in rows
            ),
            "condition_number_trans": median(
                float(row["condition_number_trans"]) for row in rows
            ),
            "spectral_entropy_trans": median(
                float(row["spectral_entropy_trans"]) for row in rows
            ),
        }
        classification = _classify(
            values["normalized_lambda_min_trans"],
            values["condition_number_trans"],
            values["spectral_entropy_trans"],
        )
        if classification != expected_classes[scene]:
            _fail(f"recomputed geometry class changed: {scene}")
        recomputed[scene] = {**values, "final_geometry_class": classification}

    if archive.get("archived_bag_count") != 6 or archive.get("failed_raw_data_retained") is not True:
        _fail("invalid acquisition archive contract changed")
    archive_names = set()
    for row in archive.get("bags", []):
        name = str(row["raw_filename"])
        archive_names.add(name)
        archived_path = Path(str(row["archived_absolute_path"])).resolve(strict=True)
        if archived_path.is_symlink() or _sha256(archived_path) != row["sha256"]:
            _fail(f"archived attempt-1 raw changed: {name}")
        if (root / "bags" / name).exists():
            _fail(f"invalid attempt remains in active bags directory: {name}")
    if len(archive_names) != 6:
        _fail("archive filename inventory changed")

    if lineage.get("W04_IDENTIFIER_RETIRED") is not True or lineage.get("W04_INCLUDED_IN_FINAL_SET") is not False:
        _fail("W04 retirement lineage changed")
    if lineage.get("correction_reason") != "OPERATOR_CONFIRMED_WRONG_SCENE_LOCATION":
        _fail("correction reason changed")
    if lineage.get("correction_before_any_icp") is not True or lineage.get("formal_trial_count_at_correction") != 0:
        _fail("correction timing changed")
    if readiness.get("FMB1_FINAL_DATASET_READY") is not True:
        _fail("readiness is false")
    if readiness.get("FMB1_W02_ATTEMPT2_INCLUDED_IN_FINAL_SET") is not True:
        _fail("W02 attempt 2 is not active")
    if readiness.get("FMB1_W02_ATTEMPT1_INCLUDED_IN_FINAL_SET") is not False:
        _fail("W02 attempt 1 entered final set")
    if attestation.get("pass") is not True or attestation.get("FORMAL_REGISTRATION_AUTHORIZED") is not False:
        _fail("NO-ICP attestation changed")

    rehashed_raw = rehashed_targets = rehashed_sources = 0
    for row in candidates:
        path = Path(str(row["raw_absolute_path"])).resolve(strict=True)
        if path.is_symlink() or _sha256(path) != row["sha256"]:
            _fail(f"candidate raw SHA changed: {path}")
        rehashed_raw += 1
    for row in targets:
        path = Path(str(row["target_path"])).resolve(strict=True)
        if path.is_symlink() or _sha256(path) != row["target_npy_sha256"]:
            _fail(f"target SHA changed: {path}")
        rehashed_targets += 1
    for row in snapshots:
        path = Path(str(row["source_path"])).resolve(strict=True)
        if path.is_symlink() or _sha256(path) != row["source_npy_sha256"]:
            _fail(f"source SHA changed: {path}")
        rehashed_sources += 1

    # CSV row counts and primary identities bind the exported tables to JSON.
    csv_expectations = {
        "final_raw_bag_manifest.csv": (36, {(r["scene_id"], r["station_id"], r["role"]) for r in final_raw}),
        "final_target_manifest.csv": (18, {(r["scene_id"], r["station_id"]) for r in targets}),
        "final_snapshot_manifest.csv": (180, {(r["scene_id"], r["station_id"], str(r["selection_index"])) for r in snapshots}),
        "final_geometry_manifest.csv": (180, {(r["scene_id"], r["station_id"], str(r["selection_index"])) for r in metrics}),
        "invalid_attempt_manifest.csv": (6, {(r["scene_id"], r["station_id"], r["role"]) for r in rejected}),
    }
    for name, (count, expected_keys) in csv_expectations.items():
        rows = _csv(output / name)
        if len(rows) != count:
            _fail(f"CSV row count changed: {name}")
        if "selection_index" in rows[0]:
            actual_keys = {(r["scene_id"], r["station_id"], r["selection_index"]) for r in rows}
        elif "role" in rows[0]:
            actual_keys = {(r["scene_id"], r["station_id"], r["role"]) for r in rows}
        else:
            actual_keys = {(r["scene_id"], r["station_id"]) for r in rows}
        if actual_keys != expected_keys:
            _fail(f"CSV identity binding changed: {name}")

    checksum_count = _verify_sha_file(output)
    if (output / "independent_verification.json").is_file():
        checksum_count -= 1
    return {
        "schema": "mid360_fmb1_w02_attempt2_independent_verification_v1",
        "status": "PASS",
        "pass": True,
        "scene_ids": list(SCENES),
        "rich_scene_count": 3,
        "weak_scene_count": 3,
        "station_count": 18,
        "raw_candidate_bag_count": 42,
        "final_admitted_bag_count": 36,
        "invalid_attempt_bag_count": 6,
        "target_count": 18,
        "snapshot_count": 180,
        "W04_INCLUDED_IN_FINAL_SET": False,
        "w02_attempt1_snapshot_count_in_final": 0,
        "w02_attempt2_snapshot_count_in_final": 30,
        "rehashed_raw_count": rehashed_raw,
        "rehashed_target_count": rehashed_targets,
        "rehashed_source_count": rehashed_sources,
        "checksum_entry_count": checksum_count,
        "recomputed_scene_geometry": recomputed,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
        "failure_count": 0,
        "failures": [],
    }


__all__ = [
    "W02IndependentVerificationError",
    "verify_w02_attempt2_final_dataset",
]
