"""Independent ROS-level verifier for frozen FMB1 acquisition/assets.

The verifier rebuilds evidence in memory from the original bags.  It never
writes a target or snapshot and has no registration/backend import or call.
Geometry-only verification intentionally lives in a separate module.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Union

import numpy as np

from phase_a_harness.mid360_pilot.bag_reader import (
    LIDAR_TOPIC,
    iter_topic_messages,
    sha256_file,
)
from phase_a_harness.mid360_pilot.lidar_adapter import (
    lidar_message_to_structured,
    xyz_array,
)
from phase_a_harness.mid360_pilot.split import select_query_frames
from phase_a_harness.mid360_pilot.static_map import (
    build_static_target_map,
    finite_range_filter,
)

from .preregistration_acquisition import run_acquisition, to_json_serializable
from .protocol import INITIAL_SCENE_IDS, QUERY_QUANTILES, STATION_IDS


ACQUISITION_COMPARE_KEYS = (
    "schema",
    "repository",
    "bags_dir",
    "NO_FORMAL_REGISTRATION",
    "FORMAL_REGISTRATION_AUTHORIZED",
    "raw_bags",
    "mapping",
    "stations",
    "inventory_gate",
    "bag_audit_pass_count",
    "station_acquisition_pass_count",
    "station_count",
    "status",
)
EXPECTED_STATION_KEYS = tuple(
    (scene_id, station_id)
    for scene_id in INITIAL_SCENE_IDS
    for station_id in STATION_IDS
)


class DeepRosVerificationError(RuntimeError):
    """Raised by strict helper contracts before a PASS can be emitted."""


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _without_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


def _canonical_points(array: np.ndarray, label: str) -> np.ndarray:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    if canonical.ndim != 2 or canonical.shape[1] != 3:
        raise DeepRosVerificationError(f"{label} must be an Nx3 point array")
    if canonical.shape[0] == 0 or not np.all(np.isfinite(canonical)):
        raise DeepRosVerificationError(f"{label} must be nonempty and finite")
    return canonical


def _load_config(config: Union[Mapping[str, Any], Path, str]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DeepRosVerificationError("config JSON must contain an object")
    return payload


def _diff_values(
    frozen: Any,
    observed: Any,
    path: str,
    failures: list[dict[str, Any]],
) -> int:
    """Compare every JSON leaf, preserving mapping keys and list order."""

    leaf_count = 0
    if isinstance(frozen, Mapping) and isinstance(observed, Mapping):
        frozen_keys = {str(key) for key in frozen}
        observed_keys = {str(key) for key in observed}
        if frozen_keys != observed_keys:
            failures.append(
                {
                    "check": "ACQUISITION_FIELD_KEYS",
                    "path": path,
                    "missing": sorted(frozen_keys - observed_keys),
                    "unexpected": sorted(observed_keys - frozen_keys),
                }
            )
        for key in sorted(frozen_keys & observed_keys):
            leaf_count += _diff_values(
                frozen[key], observed[key], f"{path}.{key}", failures
            )
        return leaf_count
    if isinstance(frozen, list) and isinstance(observed, list):
        if len(frozen) != len(observed):
            failures.append(
                {
                    "check": "ACQUISITION_LIST_LENGTH",
                    "path": path,
                    "expected": len(frozen),
                    "actual": len(observed),
                }
            )
        for index, (left, right) in enumerate(zip(frozen, observed)):
            leaf_count += _diff_values(
                left, right, f"{path}[{index}]", failures
            )
        return leaf_count
    leaf_count += 1
    if type(frozen) is not type(observed) or frozen != observed:
        failures.append(
            {
                "check": "ACQUISITION_FIELD_VALUE",
                "path": path,
                "expected": frozen,
                "actual": observed,
            }
        )
    return leaf_count


def _compare_acquisition_evidence(
    frozen: Mapping[str, Any], observed: Mapping[str, Any]
) -> tuple[int, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    leaf_count = 0
    for key in ACQUISITION_COMPARE_KEYS:
        if key not in frozen or key not in observed:
            failures.append(
                {
                    "check": "ACQUISITION_TOP_LEVEL_FIELD",
                    "path": key,
                    "expected_present": key in frozen,
                    "actual_present": key in observed,
                }
            )
            continue
        leaf_count += _diff_values(
            frozen[key], observed[key], key, failures
        )
    return leaf_count, failures


def _rows(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise DeepRosVerificationError(f"assets.{key} must be a list of mappings")
    return list(value)


def _keyed_rows(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    label: str,
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    output: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        try:
            key = tuple(str(row[field]) for field in fields)
        except KeyError as exc:
            raise DeepRosVerificationError(f"{label} row lacks {exc.args[0]}") from exc
        if key in output:
            raise DeepRosVerificationError(f"duplicate {label} key: {key}")
        output[key] = row
    return output


def _raw_bag(
    acquisition: Mapping[str, Any], scene_id: str, station_id: str, role: str
) -> Mapping[str, Any]:
    rows = acquisition.get("raw_bags")
    if not isinstance(rows, list):
        raise DeepRosVerificationError("observed acquisition has no raw_bags list")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("scene_id")) == scene_id
        and str(row.get("station_id")) == station_id
        and str(row.get("role")) == role
    ]
    if len(matches) != 1:
        raise DeepRosVerificationError(
            f"expected one {role} bag for {scene_id}/{station_id}, got {len(matches)}"
        )
    return matches[0]


def _raw_path(row: Mapping[str, Any]) -> Path:
    value = row.get("raw_absolute_path")
    if value is None:
        raise DeepRosVerificationError("raw bag row has no raw_absolute_path")
    path = Path(str(value)).resolve(strict=True)
    if not path.is_file():
        raise DeepRosVerificationError(f"raw bag is not a regular file: {path}")
    return path


def _iter_xyz(path: Path) -> Iterator[np.ndarray]:
    found = False
    for _, message, _, _ in iter_topic_messages(path, LIDAR_TOPIC):
        found = True
        yield xyz_array(lidar_message_to_structured(message))
    if not found:
        raise DeepRosVerificationError(f"MAP LiDAR topic is empty: {path}")


def _query_candidates(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for frame_index, message, bag_timestamp, timestamp in iter_topic_messages(
        path, LIDAR_TOPIC
    ):
        points = xyz_array(lidar_message_to_structured(message))
        output.append(
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
        del points
    if not output:
        raise DeepRosVerificationError(f"QUERY LiDAR topic is empty: {path}")
    return output


def _resolve_asset_path(
    value: Any, repository: Path, label: str
) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = repository / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DeepRosVerificationError(f"{label} is missing: {path}: {exc}") from exc
    if not resolved.is_file():
        raise DeepRosVerificationError(f"{label} is not a regular file: {resolved}")
    return resolved


def _check(
    checks: dict[str, bool],
    failures: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    expected: Any = None,
    actual: Any = None,
) -> None:
    checks[name] = bool(passed)
    if not passed:
        failures.append(
            {
                "check": name,
                "expected": to_json_serializable(expected),
                "actual": to_json_serializable(actual),
            }
        )


def _verify_point_array(
    rebuilt: np.ndarray,
    row: Mapping[str, Any],
    prefix: str,
    repository: Path,
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    """Compare an in-memory rebuild with one declared frozen ``.npy``."""

    checks: dict[str, bool] = {}
    failures: list[dict[str, Any]] = []
    canonical = _canonical_points(rebuilt, f"rebuilt {prefix}")
    path = _resolve_asset_path(row.get(f"{prefix}_path"), repository, f"{prefix}_path")
    frozen = np.load(path, allow_pickle=False)
    try:
        actual_file_sha = sha256_file(path)
        _check(
            checks,
            failures,
            f"{prefix}_file_sha256",
            actual_file_sha == row.get(f"{prefix}_npy_sha256"),
            expected=row.get(f"{prefix}_npy_sha256"),
            actual=actual_file_sha,
        )
        frozen_sha = _array_sha256(frozen)
        rebuilt_sha = _array_sha256(canonical)
        _check(
            checks,
            failures,
            f"{prefix}_frozen_array_sha256",
            frozen_sha == row.get(f"{prefix}_array_sha256"),
            expected=row.get(f"{prefix}_array_sha256"),
            actual=frozen_sha,
        )
        _check(
            checks,
            failures,
            f"{prefix}_rebuilt_array_sha256",
            rebuilt_sha == row.get(f"{prefix}_array_sha256"),
            expected=row.get(f"{prefix}_array_sha256"),
            actual=rebuilt_sha,
        )
        _check(
            checks,
            failures,
            f"{prefix}_np_array_equal",
            bool(np.array_equal(canonical, frozen)),
            expected=True,
            actual=False,
        )
        _check(
            checks,
            failures,
            f"{prefix}_point_count",
            int(canonical.shape[0]) == int(row.get(f"{prefix}_point_count", -1))
            == int(frozen.shape[0]),
            expected=row.get(f"{prefix}_point_count"),
            actual={"rebuilt": int(canonical.shape[0]), "frozen": int(frozen.shape[0])},
        )
        _check(
            checks,
            failures,
            f"{prefix}_shape",
            list(canonical.shape) == list(row.get(f"{prefix}_shape", []))
            == list(frozen.shape),
            expected=row.get(f"{prefix}_shape"),
            actual={"rebuilt": list(canonical.shape), "frozen": list(frozen.shape)},
        )
        _check(
            checks,
            failures,
            f"{prefix}_dtype",
            str(canonical.dtype) == str(row.get(f"{prefix}_dtype"))
            == str(frozen.dtype),
            expected=row.get(f"{prefix}_dtype"),
            actual={"rebuilt": str(canonical.dtype), "frozen": str(frozen.dtype)},
        )
    finally:
        del frozen
    return checks, failures


def _selection_check(
    selected: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    candidate_count: int,
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    checks: dict[str, bool] = {}
    failures: list[dict[str, Any]] = []
    by_index = {int(row.get("selection_index", -1)): row for row in snapshots}
    _check(
        checks,
        failures,
        "ten_unique_frozen_snapshots",
        len(snapshots) == 10 and set(by_index) == set(range(10)),
        expected=list(range(10)),
        actual=sorted(by_index),
    )
    fields = (
        ("quantile", float),
        ("target_sequence_rank", float),
        ("selected_sequence_rank", int),
        ("query_frame_index", int),
        ("query_timestamp", float),
    )
    for item in selected:
        selection_index = int(item["selection_index"])
        row = by_index.get(selection_index)
        if row is None:
            continue
        _check(
            checks,
            failures,
            f"Q{selection_index + 1:02d}_candidate_count",
            int(row.get("candidate_count", -1)) == candidate_count,
            expected=candidate_count,
            actual=row.get("candidate_count"),
        )
        expected_values = {
            "quantile": item["quantile"],
            "target_sequence_rank": item["target_sequence_rank"],
            "selected_sequence_rank": item["selected_sequence_rank"],
            "query_frame_index": item["frame_index"],
            "query_timestamp": item["timestamp"],
        }
        for field, converter in fields:
            try:
                actual = converter(row.get(field))
                expected = converter(expected_values[field])
                passed = actual == expected
            except (TypeError, ValueError):
                actual = row.get(field)
                expected = expected_values[field]
                passed = False
            _check(
                checks,
                failures,
                f"Q{selection_index + 1:02d}_{field}",
                passed,
                expected=expected,
                actual=actual,
            )
    return checks, failures


def _verify_selected_sources(
    query_path: Path,
    selected: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    repository: Path,
) -> tuple[dict[str, bool], list[dict[str, Any]], int]:
    checks: dict[str, bool] = {}
    failures: list[dict[str, Any]] = []
    by_index = {int(row["selection_index"]): row for row in snapshots}
    selected_by_frame = {
        int(item["frame_index"]): (item, by_index.get(int(item["selection_index"])))
        for item in selected
    }
    seen: set[int] = set()
    minimum = float(config["map"]["minimum_range_m"])
    maximum = float(config["map"]["maximum_range_m"])
    for frame_index, message, _, timestamp in iter_topic_messages(
        query_path, LIDAR_TOPIC
    ):
        if int(frame_index) not in selected_by_frame:
            continue
        item, snapshot = selected_by_frame[int(frame_index)]
        if snapshot is None:
            continue
        raw = xyz_array(lidar_message_to_structured(message))
        accepted = finite_range_filter(
            raw, minimum_range_m=minimum, maximum_range_m=maximum
        )
        selection_index = int(item["selection_index"])
        prefix = f"source_Q{selection_index + 1:02d}"
        # Generic array binding uses the producer's ``source_*`` columns.
        array_checks, array_failures = _verify_point_array(
            accepted, snapshot, "source", repository
        )
        for name, passed in array_checks.items():
            checks[f"{prefix}_{_without_prefix(name, 'source_')}"] = passed
        for failure in array_failures:
            failures.append(
                {
                    **failure,
                    "check": f"{prefix}_{_without_prefix(str(failure['check']), 'source_')}",
                }
            )
        _check(
            checks,
            failures,
            f"{prefix}_timestamp_on_reload",
            float(timestamp) == float(item["timestamp"]),
            expected=item["timestamp"],
            actual=timestamp,
        )
        seen.add(int(frame_index))
        del raw, accepted
    _check(
        checks,
        failures,
        "all_selected_query_frames_reloaded",
        seen == set(selected_by_frame),
        expected=sorted(selected_by_frame),
        actual=sorted(seen),
    )
    return checks, failures, len(seen)


def _verify_station_ros(
    scene_id: str,
    station_id: str,
    acquisition: Mapping[str, Any],
    target_row: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    repository: Path,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    failures: list[dict[str, Any]] = []
    map_row = _raw_bag(acquisition, scene_id, station_id, "MAP")
    query_row = _raw_bag(acquisition, scene_id, station_id, "QUERY")
    map_path = _raw_path(map_row)
    query_path = _raw_path(query_row)

    target, metadata = build_static_target_map(_iter_xyz(map_path), config)
    try:
        target_checks, target_failures = _verify_point_array(
            target, target_row, "target", repository
        )
        checks.update(target_checks)
        failures.extend(target_failures)
        metadata_bindings = {
            "map_frame_count": "map_scan_count",
            "raw_point_count": "raw_point_count",
            "filtered_point_count": "finite_range_filtered_point_count",
            "target_point_count": "target_map_point_count",
        }
        for declared, rebuilt in metadata_bindings.items():
            _check(
                checks,
                failures,
                f"target_{declared}",
                int(target_row.get(declared, -1)) == int(metadata[rebuilt]),
                expected=metadata[rebuilt],
                actual=target_row.get(declared),
            )
        _check(
            checks,
            failures,
            "target_map_bag_sha256",
            target_row.get("map_bag_sha256") == map_row.get("sha256"),
            expected=map_row.get("sha256"),
            actual=target_row.get("map_bag_sha256"),
        )
        _check(
            checks,
            failures,
            "target_map_bag_path",
            target_row.get("map_bag_path") == str(map_path),
            expected=str(map_path),
            actual=target_row.get("map_bag_path"),
        )
        for field in ("minimum_range_m", "maximum_range_m", "voxel_size_m"):
            _check(
                checks,
                failures,
                f"target_{field}",
                float(target_row.get(field, float("nan"))) == float(metadata[field]),
                expected=metadata[field],
                actual=target_row.get(field),
            )
        _check(
            checks,
            failures,
            "query_contribution_to_target_zero",
            int(target_row.get("query_contribution_to_target", -1)) == 0,
            expected=0,
            actual=target_row.get("query_contribution_to_target"),
        )
    finally:
        del target
        gc.collect()

    candidates = _query_candidates(query_path)
    selected = select_query_frames(candidates, QUERY_QUANTILES)
    selection_checks, selection_failures = _selection_check(
        selected, snapshots, len(candidates)
    )
    candidate_count = len(candidates)
    checks.update(selection_checks)
    failures.extend(selection_failures)
    source_checks, source_failures, source_count = _verify_selected_sources(
        query_path, selected, snapshots, config, repository
    )
    checks.update(source_checks)
    failures.extend(source_failures)
    expected_query_sha = query_row.get("sha256")
    for row in snapshots:
        selection_index = int(row.get("selection_index", -1))
        expected_snapshot_id = f"{scene_id}_{station_id}_Q{selection_index + 1:02d}"
        _check(
            checks,
            failures,
            f"Q{selection_index + 1:02d}_snapshot_identity",
            row.get("scene_id") == scene_id
            and row.get("station_id") == station_id
            and row.get("snapshot_id") == expected_snapshot_id,
            expected={
                "scene_id": scene_id,
                "station_id": station_id,
                "snapshot_id": expected_snapshot_id,
            },
            actual={
                "scene_id": row.get("scene_id"),
                "station_id": row.get("station_id"),
                "snapshot_id": row.get("snapshot_id"),
            },
        )
        _check(
            checks,
            failures,
            f"Q{selection_index + 1:02d}_query_bag_path",
            row.get("query_bag_path") == str(query_path),
            expected=str(query_path),
            actual=row.get("query_bag_path"),
        )
        _check(
            checks,
            failures,
            f"Q{selection_index + 1:02d}_query_bag_sha256",
            row.get("query_bag_sha256") == expected_query_sha,
            expected=expected_query_sha,
            actual=row.get("query_bag_sha256"),
        )
        _check(
            checks,
            failures,
            f"Q{selection_index + 1:02d}_target_array_sha256_binding",
            row.get("target_array_sha256") == target_row.get("target_array_sha256"),
            expected=target_row.get("target_array_sha256"),
            actual=row.get("target_array_sha256"),
        )
        _check(
            checks,
            failures,
            f"Q{selection_index + 1:02d}_target_npy_sha256_binding",
            row.get("target_npy_sha256") == target_row.get("target_npy_sha256"),
            expected=target_row.get("target_npy_sha256"),
            actual=row.get("target_npy_sha256"),
        )
    del candidates, selected
    gc.collect()
    return {
        "scene_id": scene_id,
        "station_id": station_id,
        "status": "PASS" if not failures and all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "target_rebuilt": 1,
            "query_candidates": candidate_count,
            "snapshots_compared": source_count,
        },
        "failures": failures,
    }


def _failure_payload(message: str, *, failures: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    return to_json_serializable(
        {
            "schema": "mid360_fmb1_preregistration_deep_ros_verifier_v1",
            "status": "FAIL",
            "NO_FORMAL_REGISTRATION": True,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "checks": {"fail_closed": True},
            "counts": {
                "acquisition_fields_compared": 0,
                "target_rebuild_count": 0,
                "source_array_compare_count": 0,
            },
            "station_results": [],
            "failures": [{"check": "DEEP_ROS_VERIFIER", "reason": message}, *failures],
        }
    )


def verify_ros_evidence(
    acquisition: Mapping[str, Any],
    assets: Mapping[str, Any],
    config: Union[Mapping[str, Any], Path, str],
) -> dict[str, Any]:
    """Re-read all raw ROS evidence and compare it with frozen artifacts."""

    try:
        repository = Path(str(acquisition["repository"])).resolve(strict=True)
        bags_dir = Path(str(acquisition["bags_dir"]))
        observed = run_acquisition(repository, bags_dir, config)
        compared_fields, acquisition_failures = _compare_acquisition_evidence(
            acquisition, observed
        )
    except Exception as exc:
        return _failure_payload(f"{type(exc).__name__}: {exc}")

    if acquisition_failures:
        payload = _failure_payload(
            "independent acquisition rerun differs from frozen acquisition",
            failures=acquisition_failures,
        )
        payload["counts"]["acquisition_fields_compared"] = compared_fields
        payload["checks"]["acquisition_rerun_exact"] = False
        return payload
    if observed.get("status") != "PASS":
        payload = _failure_payload("independent acquisition rerun did not PASS")
        payload["counts"]["acquisition_fields_compared"] = compared_fields
        payload["checks"]["acquisition_rerun_exact"] = True
        return payload

    try:
        config_payload = _load_config(config)
        targets = _rows(assets, "targets")
        snapshots = _rows(assets, "snapshots")
        if assets.get("failures") not in (None, []):
            raise DeepRosVerificationError("asset producer recorded failures")
        if [float(value) for value in assets.get("quantiles", [])] != [
            float(value) for value in QUERY_QUANTILES
        ]:
            raise DeepRosVerificationError("asset quantile contract changed")
        if int(assets.get("target_count", -1)) != len(targets):
            raise DeepRosVerificationError("declared target_count differs from targets")
        if int(assets.get("snapshot_count", -1)) != len(snapshots):
            raise DeepRosVerificationError("declared snapshot_count differs from snapshots")
        if int(assets.get("query_contribution_to_every_target", -1)) != 0:
            raise DeepRosVerificationError("QUERY contribution to target is not zero")
        for counter in (
            "open3d_registration_call_count",
            "pcl_cli_invocation_count",
            "other_registration_process_count",
            "formal_trial_count",
        ):
            if int(assets.get(counter, -1)) != 0:
                raise DeepRosVerificationError(f"asset counter is nonzero: {counter}")
        target_by_key = _keyed_rows(targets, ("scene_id", "station_id"), "target")
        snapshot_by_key = _keyed_rows(
            snapshots, ("scene_id", "station_id", "snapshot_id"), "snapshot"
        )
        if set(target_by_key) != set(EXPECTED_STATION_KEYS):
            raise DeepRosVerificationError("target keys are not the exact 18 stations")
        expected_snapshot_count = len(EXPECTED_STATION_KEYS) * 10
        if len(snapshot_by_key) != expected_snapshot_count:
            raise DeepRosVerificationError(
                f"snapshot count must be {expected_snapshot_count}, got {len(snapshot_by_key)}"
            )
    except Exception as exc:
        payload = _failure_payload(f"{type(exc).__name__}: {exc}")
        payload["counts"]["acquisition_fields_compared"] = compared_fields
        payload["checks"]["acquisition_rerun_exact"] = True
        return payload

    station_results: list[dict[str, Any]] = []
    for scene_id, station_id in EXPECTED_STATION_KEYS:
        station_snapshots = sorted(
            (
                row
                for row in snapshots
                if str(row.get("scene_id")) == scene_id
                and str(row.get("station_id")) == station_id
            ),
            key=lambda row: int(row.get("selection_index", -1)),
        )
        try:
            result = _verify_station_ros(
                scene_id,
                station_id,
                observed,
                target_by_key[(scene_id, station_id)],
                station_snapshots,
                config_payload,
                repository,
            )
        except Exception as exc:
            result = {
                "scene_id": scene_id,
                "station_id": station_id,
                "status": "FAIL",
                "checks": {},
                "counts": {"target_rebuilt": 0, "snapshots_compared": 0},
                "failures": [
                    {
                        "check": "STATION_DEEP_ROS_VERIFY",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                ],
            }
        station_results.append(result)
        gc.collect()

    station_failures = [
        {
            "scene_id": row["scene_id"],
            "station_id": row["station_id"],
            **failure,
        }
        for row in station_results
        for failure in row["failures"]
    ]
    target_count = sum(
        int(row["counts"].get("target_rebuilt", 0)) for row in station_results
    )
    source_count = sum(
        int(row["counts"].get("snapshots_compared", 0)) for row in station_results
    )
    all_stations_pass = all(row["status"] == "PASS" for row in station_results)
    payload = {
        "schema": "mid360_fmb1_preregistration_deep_ros_verifier_v1",
        "status": "PASS" if all_stations_pass and not station_failures else "FAIL",
        "NO_FORMAL_REGISTRATION": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "independent_acquisition_rerun": True,
        "checks": {
            "acquisition_rerun_exact": True,
            "raw_bag_count_36": len(observed.get("raw_bags", [])) == 36,
            "station_count_18": len(observed.get("stations", [])) == 18,
            "target_count_18": len(targets) == 18,
            "snapshot_count_180": len(snapshots) == 180,
            "all_target_rebuilds_array_equal": target_count == 18
            and all_stations_pass,
            "all_source_rebuilds_array_equal": source_count == 180
            and all_stations_pass,
            "all_fixed_quantile_selections_match": all_stations_pass,
        },
        "counts": {
            "acquisition_fields_compared": compared_fields,
            "raw_bag_count": len(observed.get("raw_bags", [])),
            "station_count": len(observed.get("stations", [])),
            "target_rebuild_count": target_count,
            "source_array_compare_count": source_count,
            "station_failure_count": sum(
                row["status"] != "PASS" for row in station_results
            ),
            "mismatch_count": len(station_failures),
        },
        "station_results": station_results,
        "failures": station_failures,
    }
    normalized = to_json_serializable(payload)
    json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    return normalized


__all__ = [
    "DeepRosVerificationError",
    "verify_ros_evidence",
]
