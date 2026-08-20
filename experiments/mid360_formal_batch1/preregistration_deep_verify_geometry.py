"""Independent deep verification of frozen FMB1 geometry-only evidence.

The verifier consumes only frozen point arrays and preregistration manifests.
It recomputes the ten allowed initial-association geometry fields at identity,
then recomputes station and scene medians.  It neither imports nor invokes a
registration backend.  A caller is expected to wrap execution in the FMB1
``NoRegistrationGuard``.
"""

from __future__ import annotations

import gc
import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    GEOMETRY_ONLY_FIELDS,
    TargetGeometryContext,
    compute_geometry_only_initial_metrics,
)

from .protocol import INITIAL_SCENE_IDS, STATION_IDS, geometry_admission, geometry_class


ABSOLUTE_TOLERANCE = 1.0e-12
_METADATA_FIELDS = frozenset(
    {
        "scene_id",
        "station_id",
        "snapshot_id",
        "selection_index",
        "query_timestamp",
        "timestamp",
        "query_frame_index",
        "frame_index",
    }
)
_REQUIRED_METADATA_FIELDS = frozenset(
    {"scene_id", "station_id", "snapshot_id", "selection_index"}
)
_FORBIDDEN_RESULT_NAMES = frozenset(
    {
        "t_est",
        "t_estimated",
        "estimated_transform",
        "estimated_pose",
        "final_pose",
        "translation_error",
        "translation_error_m",
        "rotation_error",
        "rotation_error_deg",
        "final_residual",
        "final_residual_rmse",
        "correspondence_turnover",
        "accepted_source_turnover",
        "fitness",
        "fitness_score",
        "solver_result",
        "solver_success",
        "registration_result",
        "registration_output",
        "inlier_rmse",
    }
)
_ZERO_ONLY_FIELDS = frozenset(
    {
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "formal_trial_count",
        "registration_execution_count",
        "actual_registration_trials",
    }
)
_FALSE_ONLY_FIELDS = frozenset(
    {
        "registration_executed",
        "registration_called",
        "odometry_called",
        "scan_matching_called",
        "formal_registration_authorized",
    }
)


class DeepGeometryVerificationError(RuntimeError):
    """Frozen geometry evidence failed independent recomputation."""


def _fail(message: str) -> None:
    raise DeepGeometryVerificationError(f"FMB1_DEEP_GEOMETRY_INVALID: {message}")


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_forbidden_result_name(name: str) -> bool:
    if name in _FORBIDDEN_RESULT_NAMES:
        return True
    return (
        name.startswith("t_est_")
        or "translation_error" in name
        or "rotation_error" in name
        or "final_residual" in name
        or "turnover" in name
        or name.startswith("fitness_")
        or name.endswith("_fitness")
        or "solver_result" in name
        or "registration_result" in name
        or "registration_output" in name
    )


def _scan_forbidden_fields(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = _normalized_name(str(key))
            child_path = f"{path}.{key}"
            if _is_forbidden_result_name(name):
                _fail(f"forbidden result field: {child_path}")
            if name in _ZERO_ONLY_FIELDS and (
                isinstance(child, bool) or not isinstance(child, int) or child != 0
            ):
                _fail(f"registration/trial count is nonzero: {child_path}")
            if name in _FALSE_ONLY_FIELDS and child is not False:
                _fail(f"registration/odometry flag is not false: {child_path}")
            _scan_forbidden_fields(child, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_forbidden_fields(child, path=f"{path}[{index}]")


def _rows(payload: Mapping[str, Any], *names: str) -> list[Mapping[str, Any]]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list):
            if not all(isinstance(row, Mapping) for row in value):
                _fail(f"{name} must contain objects")
            return list(value)
    _fail(f"missing row list; accepted names={names}")


def _key(row: Mapping[str, Any], label: str) -> tuple[str, str]:
    try:
        scene_id = str(row["scene_id"])
        station_id = str(row["station_id"])
    except KeyError as exc:
        _fail(f"{label} is missing scene/station identity")
    if scene_id not in INITIAL_SCENE_IDS or station_id not in STATION_IDS:
        _fail(f"{label} has invalid scene/station identity: {scene_id}/{station_id}")
    return scene_id, station_id


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        _fail(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        _fail(f"{label} must be >= {minimum}")
    return result


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        _fail(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DeepGeometryVerificationError(
            f"FMB1_DEEP_GEOMETRY_INVALID: {label} must be numeric"
        ) from exc
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _same_number(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        if _strict_int(actual, label) != int(expected):
            _fail(f"{label} changed: {actual!r} != {expected!r}")
        return
    left = _finite_float(actual, label)
    right = _finite_float(expected, f"recomputed {label}")
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE):
        _fail(f"{label} changed: {left!r} != {right!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    # Bound the transient allocation: hashing one production target must not
    # create a second target-sized byte string.
    rows_per_chunk = max(1, (1024 * 1024) // max(1, canonical.shape[1] * 8))
    for start in range(0, canonical.shape[0], rows_per_chunk):
        digest.update(canonical[start : start + rows_per_chunk].tobytes(order="C"))
    return digest.hexdigest()


def _path(row: Mapping[str, Any], names: Sequence[str], label: str) -> Path:
    for name in names:
        value = row.get(name)
        if value is not None:
            supplied = Path(str(value))
            if supplied.is_symlink():
                _fail(f"{label} must not be a symlink: {supplied}")
            path = supplied.resolve(strict=True)
            if not path.is_file():
                _fail(f"{label} is not a regular non-symlink file: {path}")
            return path
    _fail(f"{label} path is missing")


def _declared_sha(row: Mapping[str, Any], names: Sequence[str], label: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            return value
    _fail(f"{label} SHA256 is missing or invalid")


def _load_frozen_array(
    row: Mapping[str, Any],
    *,
    path_names: Sequence[str],
    file_sha_names: Sequence[str],
    array_sha_names: Sequence[str],
    point_count_names: Sequence[str],
    mmap_mode: str | None,
    label: str,
) -> np.ndarray:
    path = _path(row, path_names, label)
    expected_file_sha = _declared_sha(row, file_sha_names, label)
    actual_file_sha = _sha256_file(path)
    if actual_file_sha != expected_file_sha:
        _fail(f"{label} file SHA256 changed")
    array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or array.shape[0] == 0
        or not np.all(np.isfinite(array))
    ):
        _fail(f"{label} must be a nonempty finite Nx3 array")
    expected_count = None
    for name in point_count_names:
        if name in row:
            expected_count = _strict_int(row[name], f"{label}.{name}", minimum=1)
            break
    if expected_count is None or expected_count != int(array.shape[0]):
        _fail(f"{label} point count changed")
    declared_array_sha = next(
        (
            str(row[name])
            for name in array_sha_names
            if isinstance(row.get(name), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(row[name]))
        ),
        None,
    )
    if declared_array_sha is not None and _array_sha256(array) != declared_array_sha:
        _fail(f"{label} array SHA256 changed")
    size_value = next(
        (
            row[name]
            for name in (
                "target_size_bytes",
                "source_size_bytes",
                "target_bytes",
                "source_bytes",
            )
            if name in row
        ),
        None,
    )
    if size_value is not None and _strict_int(size_value, f"{label} size") != path.stat().st_size:
        _fail(f"{label} file size changed")
    return array


def _metric_schema(row: Mapping[str, Any], label: str) -> None:
    fields = set(row)
    missing_metadata = _REQUIRED_METADATA_FIELDS - fields
    missing_metrics = set(GEOMETRY_ONLY_FIELDS) - fields
    unknown = fields - _METADATA_FIELDS - set(GEOMETRY_ONLY_FIELDS)
    if missing_metadata or missing_metrics or unknown:
        _fail(
            f"{label} schema changed: missing_metadata={sorted(missing_metadata)}, "
            f"missing_metrics={sorted(missing_metrics)}, unknown={sorted(unknown)}"
        )


def _expected_scene_summary(
    scene_id: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    medians = {
        field: float(np.median([float(row[field]) for row in rows]))
        for field in GEOMETRY_ONLY_FIELDS
    }
    final_class = geometry_class(
        medians["normalized_lambda_min_trans"],
        medians["condition_number_trans"],
        medians["spectral_entropy_trans"],
    )
    gate = geometry_admission(
        medians["normalized_lambda_min_trans"],
        medians["condition_number_trans"],
        medians["spectral_entropy_trans"],
    )
    if gate["final_geometry_class"] != final_class:
        _fail(f"formal geometry gate implementations disagree for {scene_id}")
    candidate_class = "RICH" if scene_id.startswith("FMB1_R") else "WEAK"
    aligned = final_class == candidate_class
    if aligned and gate["admitted"]:
        status = "GEOMETRY_ADMITTED"
        reason = None
    elif final_class == "INTERMEDIATE":
        status = "GEOMETRY_REVIEW"
        reason = "GEOMETRY_INTERMEDIATE"
    else:
        status = "GEOMETRY_REJECTED"
        reason = "SEMANTIC_CANDIDATE_GEOMETRY_CLASS_MISMATCH"
    return {
        "medians": medians,
        "gate": gate,
        "candidate_class": candidate_class,
        "candidate_class_alignment": aligned,
        "geometry_admission_status": status,
        "failure_reason": reason,
        "replacement_allowed_under_preregistration": status != "GEOMETRY_ADMITTED",
    }


def _verify_station_summary(
    row: Mapping[str, Any],
    expected_medians: Mapping[str, float],
    key: tuple[str, str],
) -> None:
    if _strict_int(row.get("snapshot_count"), f"station {key} snapshot_count") != 10:
        _fail(f"station {key} summary snapshot count changed")
    if row.get("geometry_only") is not True or row.get("registration_executed") is not False:
        _fail(f"station {key} summary is not geometry-only")
    if row.get("aggregation") != "MEDIAN_OVER_10_FROZEN_QUERY_SNAPSHOTS":
        _fail(f"station {key} summary aggregation changed")
    medians = row.get("median")
    if not isinstance(medians, Mapping) or set(medians) != set(GEOMETRY_ONLY_FIELDS):
        _fail(f"station {key} median schema changed")
    for field in GEOMETRY_ONLY_FIELDS:
        _same_number(medians[field], expected_medians[field], f"station {key}.{field}")


def _verify_scene_summary(
    row: Mapping[str, Any], expected: Mapping[str, Any], scene_id: str
) -> None:
    if str(row.get("scene_id")) != scene_id:
        _fail(f"scene summary identity changed for {scene_id}")
    if row.get("aggregation") != "MEDIAN_OVER_ALL_30_NESTED_STATION_SNAPSHOTS":
        _fail(f"scene {scene_id} aggregation changed")
    if _strict_int(row.get("snapshot_count"), f"scene {scene_id} snapshot_count") != 30:
        _fail(f"scene {scene_id} snapshot count changed")
    if _strict_int(row.get("station_count"), f"scene {scene_id} station_count") != 3:
        _fail(f"scene {scene_id} station count changed")
    medians = expected["medians"]
    for field in GEOMETRY_ONLY_FIELDS:
        _same_number(
            row.get(f"median_{field}"),
            medians[field],
            f"scene {scene_id}.median_{field}",
        )
    gate = expected["gate"]
    for field, value in (
        ("geometry_lambda_min_median", gate["geometry_lambda_min_median"]),
        ("geometry_condition_median", gate["geometry_condition_median"]),
        ("geometry_entropy_median", gate["geometry_entropy_median"]),
    ):
        _same_number(row.get(field), value, f"scene {scene_id}.{field}")
    exact = {
        "semantic_candidate_label": f"{expected['candidate_class']}_CANDIDATE",
        "final_geometry_class": gate["final_geometry_class"],
        "admitted": gate["admitted"],
        "exclusion_reason": gate["exclusion_reason"],
        "candidate_class_alignment": expected["candidate_class_alignment"],
        "geometry_admission_status": expected["geometry_admission_status"],
        "failure_reason": expected["failure_reason"],
        "replacement_allowed_under_preregistration": expected[
            "replacement_allowed_under_preregistration"
        ],
    }
    for field, value in exact.items():
        if row.get(field) != value:
            _fail(
                f"scene {scene_id}.{field} changed: {row.get(field)!r} != {value!r}"
            )


def verify_geometry_evidence(
    assets: Mapping[str, Any],
    geometry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and verify all 180 frozen initial-only geometry rows.

    Geometry class/admission consistency is verified, not forced to readiness.
    A candidate/class mismatch such as W02 computing as RICH and therefore
    being marked ``GEOMETRY_REJECTED`` is valid evidence when every stored
    field agrees with recomputation.
    """

    if not isinstance(assets, Mapping) or not isinstance(geometry, Mapping):
        _fail("assets and geometry must be mappings")
    if not isinstance(config, Mapping):
        _fail("config must be a mapping")
    _scan_forbidden_fields(assets, path="$.assets")
    _scan_forbidden_fields(geometry, path="$.geometry")
    if geometry.get("geometry_only") is not True:
        _fail("geometry_only flag is not true")
    if geometry.get("registration_executed") is not False:
        _fail("registration_executed flag is not false")
    if tuple(geometry.get("metric_fields", ())) != tuple(GEOMETRY_ONLY_FIELDS):
        _fail("geometry metric field allowlist changed")
    if geometry.get("T0") not in {"IDENTITY_4X4", None}:
        _fail("geometry T0 is not identity")

    expected_station_keys = {
        (scene_id, station_id)
        for scene_id in INITIAL_SCENE_IDS
        for station_id in STATION_IDS
    }
    target_rows = _rows(assets, "targets", "target_maps")
    if len(target_rows) != 18:
        _fail(f"target count must be 18, got {len(target_rows)}")
    targets: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(target_rows):
        key = _key(row, f"targets[{index}]")
        if key in targets:
            _fail(f"duplicate target station: {key}")
        targets[key] = row
    if set(targets) != expected_station_keys:
        _fail("target station set is not frozen 6x3 inventory")

    snapshot_rows = _rows(assets, "snapshots", "snapshot_inventory_frozen")
    if len(snapshot_rows) != 180:
        _fail(f"snapshot count must be 180, got {len(snapshot_rows)}")
    snapshots: dict[str, Mapping[str, Any]] = {}
    snapshots_by_station: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(snapshot_rows):
        key = _key(row, f"snapshots[{index}]")
        snapshot_id = str(row.get("snapshot_id", ""))
        if not snapshot_id or snapshot_id in snapshots:
            _fail(f"snapshot identity is missing/duplicated: {snapshot_id!r}")
        selection_index = _strict_int(
            row.get("selection_index"), f"snapshot {snapshot_id}.selection_index"
        )
        expected_id = f"{key[0]}_{key[1]}_Q{selection_index + 1:02d}"
        if snapshot_id != expected_id:
            _fail(f"snapshot identity changed: {snapshot_id} != {expected_id}")
        snapshots[snapshot_id] = row
        snapshots_by_station[key].append(row)
    if set(snapshots_by_station) != expected_station_keys or any(
        len(rows) != 10 for rows in snapshots_by_station.values()
    ):
        _fail("snapshot inventory must contain exactly 10 rows per frozen station")

    metric_rows = _rows(geometry, "snapshot_metrics", "geometry_metrics")
    if len(metric_rows) != 180:
        _fail(f"geometry metric row count must be 180, got {len(metric_rows)}")
    stored_metrics: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(metric_rows):
        _metric_schema(row, f"snapshot_metrics[{index}]")
        snapshot_id = str(row["snapshot_id"])
        if snapshot_id not in snapshots or snapshot_id in stored_metrics:
            _fail(f"geometry snapshot identity is missing/duplicated: {snapshot_id}")
        if _key(row, f"snapshot_metrics[{index}]") != _key(
            snapshots[snapshot_id], f"snapshot {snapshot_id}"
        ):
            _fail(f"geometry station binding changed for {snapshot_id}")
        if _strict_int(row["selection_index"], f"geometry {snapshot_id}.selection_index") != _strict_int(
            snapshots[snapshot_id]["selection_index"],
            f"snapshot {snapshot_id}.selection_index",
        ):
            _fail(f"geometry selection binding changed for {snapshot_id}")
        stored_metrics[snapshot_id] = row
    if set(stored_metrics) != set(snapshots):
        _fail("geometry rows do not bind exactly all frozen snapshots")

    station_summary_rows = _rows(geometry, "station_summaries")
    station_summaries: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(station_summary_rows):
        key = _key(row, f"station_summaries[{index}]")
        if key in station_summaries:
            _fail(f"duplicate station geometry summary: {key}")
        station_summaries[key] = row
    if set(station_summaries) != expected_station_keys:
        _fail("station geometry summaries do not cover frozen 18 stations")

    recomputed_rows: list[dict[str, Any]] = []
    station_medians: dict[tuple[str, str], dict[str, float]] = {}
    for key in sorted(expected_station_keys):
        target_row = targets[key]
        target = None
        context = None
        try:
            target = _load_frozen_array(
                target_row,
                path_names=("target_path", "target_npy_path"),
                file_sha_names=("target_npy_sha256", "target_sha256"),
                array_sha_names=("target_array_sha256",),
                point_count_names=("target_point_count",),
                mmap_mode="r",
                label=f"target {key}",
            )
            context = TargetGeometryContext.prepare(target)
            station_rows: list[dict[str, Any]] = []
            ordered_snapshots = sorted(
                snapshots_by_station[key], key=lambda row: int(row["selection_index"])
            )
            if [int(row["selection_index"]) for row in ordered_snapshots] != list(range(10)):
                _fail(f"station {key} selection indexes are not 0..9")
            for snapshot in ordered_snapshots:
                snapshot_id = str(snapshot["snapshot_id"])
                source = None
                try:
                    declared_target_sha = _declared_sha(
                        snapshot,
                        ("target_npy_sha256", "target_sha256"),
                        f"snapshot {snapshot_id} target",
                    )
                    target_sha = _declared_sha(
                        target_row,
                        ("target_npy_sha256", "target_sha256"),
                        f"target {key}",
                    )
                    if declared_target_sha != target_sha:
                        _fail(f"snapshot {snapshot_id} target SHA binding changed")
                    source = _load_frozen_array(
                        snapshot,
                        path_names=("source_path", "source_npy_path"),
                        file_sha_names=("source_npy_sha256", "source_sha256"),
                        array_sha_names=("source_array_sha256",),
                        point_count_names=("source_point_count",),
                        mmap_mode=None,
                        label=f"source {snapshot_id}",
                    )
                    recomputed = compute_geometry_only_initial_metrics(
                        source,
                        np.eye(4, dtype=np.float64),
                        context=context,
                    )
                    if tuple(recomputed) != tuple(GEOMETRY_ONLY_FIELDS):
                        _fail(f"geometry kernel schema changed for {snapshot_id}")
                    stored = stored_metrics[snapshot_id]
                    row = {
                        "scene_id": key[0],
                        "station_id": key[1],
                        "snapshot_id": snapshot_id,
                        "selection_index": int(snapshot["selection_index"]),
                    }
                    for field in GEOMETRY_ONLY_FIELDS:
                        expected_value = recomputed[field]
                        if expected_value is None:
                            _fail(f"geometry kernel returned noncomputable {field} for {snapshot_id}")
                        _same_number(
                            stored[field], expected_value, f"geometry {snapshot_id}.{field}"
                        )
                        row[field] = expected_value
                    station_rows.append(row)
                    recomputed_rows.append(row)
                finally:
                    if source is not None:
                        del source
                    gc.collect()
            medians = {
                field: float(np.median([float(row[field]) for row in station_rows]))
                for field in GEOMETRY_ONLY_FIELDS
            }
            station_medians[key] = medians
            _verify_station_summary(station_summaries[key], medians, key)
        finally:
            if context is not None:
                del context
            if target is not None:
                del target
            gc.collect()

    counts = Counter((row["scene_id"], row["station_id"]) for row in recomputed_rows)
    if len(recomputed_rows) != 180 or set(counts.values()) != {10}:
        _fail("recomputed geometry row inventory is not 18x10")
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in recomputed_rows:
        by_scene[str(row["scene_id"])].append(row)
    if set(by_scene) != set(INITIAL_SCENE_IDS) or any(len(rows) != 30 for rows in by_scene.values()):
        _fail("recomputed scene geometry inventory is not 6x30")

    scene_rows = _rows(geometry, "scene_summaries", "geometry_scenes")
    scene_summaries: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(scene_rows):
        scene_id = str(row.get("scene_id", ""))
        if scene_id not in INITIAL_SCENE_IDS or scene_id in scene_summaries:
            _fail(f"scene geometry summary identity is invalid/duplicated: {scene_id}")
        scene_summaries[scene_id] = row
    if set(scene_summaries) != set(INITIAL_SCENE_IDS):
        _fail("scene geometry summaries do not cover frozen six scenes")

    verified_scenes: list[dict[str, Any]] = []
    for scene_id in INITIAL_SCENE_IDS:
        expected = _expected_scene_summary(scene_id, by_scene[scene_id])
        _verify_scene_summary(scene_summaries[scene_id], expected, scene_id)
        verified_scenes.append(
            {
                "scene_id": scene_id,
                "final_geometry_class": expected["gate"]["final_geometry_class"],
                "geometry_admission_status": expected["geometry_admission_status"],
                "candidate_class_alignment": expected["candidate_class_alignment"],
            }
        )

    return {
        "schema": "mid360_fmb1_deep_geometry_verification_v1",
        "status": "PASS",
        "pass": True,
        "target_count": 18,
        "station_count": 18,
        "snapshot_count": 180,
        "geometry_row_count": 180,
        "scene_count": 6,
        "metric_fields": list(GEOMETRY_ONLY_FIELDS),
        "absolute_comparison_tolerance": ABSOLUTE_TOLERANCE,
        "identity_initial_transform": True,
        "geometry_only": True,
        "registration_executed": False,
        "verified_scenes": verified_scenes,
    }


__all__ = [
    "ABSOLUTE_TOLERANCE",
    "DeepGeometryVerificationError",
    "verify_geometry_evidence",
]
