"""Independent no-backend verification for the W02 attempt-2 evidence."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Mapping, Union

from .preregistration_acquisition import to_json_serializable
from .preregistration_assets import analyze_geometry_only
from .preregistration_deep_verify_geometry import _scan_forbidden_fields
from .preregistration_deep_verify_ros import (
    _compare_acquisition_evidence,
    _keyed_rows,
    _load_config,
    _rows,
    _verify_station_ros,
)
from .w02_attempt2 import run_w02_attempt2_acquisition


STATION_KEYS = tuple(("FMB1_W02", station) for station in ("S01", "S02", "S03"))


def _failure(schema: str, reason: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "status": "FAIL",
        "pass": False,
        "NO_FORMAL_REGISTRATION": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
        "failure_count": 1,
        "failures": [reason],
    }


def verify_w02_attempt2_ros_evidence(
    acquisition: Mapping[str, Any],
    assets: Mapping[str, Any],
    config: Union[Mapping[str, Any], Path, str],
) -> dict[str, Any]:
    schema = "mid360_fmb1_w02_attempt2_deep_ros_verification_v1"
    try:
        repository = Path(str(acquisition["repository"])).resolve(strict=True)
        observed = run_w02_attempt2_acquisition(repository, config)
        compared_fields, acquisition_failures = _compare_acquisition_evidence(
            acquisition, observed
        )
        if acquisition_failures:
            raise ValueError(
                "independent acquisition differs: "
                + json.dumps(acquisition_failures, sort_keys=True)
            )
        if observed.get("W02_ATTEMPT2_ACQUISITION_PASS") is not True:
            raise ValueError("independent W02 attempt-2 acquisition did not PASS")
        targets = _rows(assets, "targets")
        snapshots = _rows(assets, "snapshots")
        if assets.get("failures") not in (None, []):
            raise ValueError("asset producer recorded failures")
        for counter in (
            "open3d_registration_call_count",
            "pcl_cli_invocation_count",
            "other_registration_process_count",
            "formal_trial_count",
        ):
            if int(assets.get(counter, -1)) != 0:
                raise ValueError(f"nonzero asset counter: {counter}")
        if int(assets.get("query_contribution_to_every_target", -1)) != 0:
            raise ValueError("QUERY contribution is not zero")
        target_by_key = _keyed_rows(
            targets, ("scene_id", "station_id"), "W02 attempt-2 target"
        )
        snapshot_by_key = _keyed_rows(
            snapshots,
            ("scene_id", "station_id", "snapshot_id"),
            "W02 attempt-2 snapshot",
        )
        if set(target_by_key) != set(STATION_KEYS):
            raise ValueError("target keys are not exact W02 S01-S03")
        if len(snapshot_by_key) != 30:
            raise ValueError("W02 attempt 2 snapshot count is not 30")
        config_payload = _load_config(config)
    except Exception as exc:
        return _failure(schema, f"{type(exc).__name__}: {exc}")

    station_results: list[dict[str, Any]] = []
    for scene_id, station_id in STATION_KEYS:
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
                "failures": [f"{type(exc).__name__}: {exc}"],
            }
        station_results.append(result)
        gc.collect()
    failures = [
        {
            "scene_id": row["scene_id"],
            "station_id": row["station_id"],
            "detail": failure,
        }
        for row in station_results
        for failure in row.get("failures", [])
    ]
    target_count = sum(
        int(row.get("counts", {}).get("target_rebuilt", 0))
        for row in station_results
    )
    source_count = sum(
        int(row.get("counts", {}).get("snapshots_compared", 0))
        for row in station_results
    )
    passed = not failures and target_count == 3 and source_count == 30
    return to_json_serializable(
        {
            "schema": schema,
            "status": "PASS" if passed else "FAIL",
            "pass": passed,
            "NO_FORMAL_REGISTRATION": True,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "actual_formal_trials": 0,
            "independent_acquisition_rerun": True,
            "acquisition_fields_compared": compared_fields,
            "raw_bag_count": len(observed.get("raw_bags", [])),
            "station_count": len(observed.get("stations", [])),
            "target_rebuild_count": target_count,
            "source_array_compare_count": source_count,
            "station_results": station_results,
            "failure_count": len(failures),
            "failures": failures,
        }
    )


def _geometry_compare_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema",
        "metric_fields",
        "T0",
        "geometry_only",
        "registration_executed",
        "snapshot_metrics",
        "station_summaries",
        "scene_summaries",
        "failures",
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "formal_trial_count",
    )
    return {field: payload.get(field) for field in fields}


def verify_w02_attempt2_geometry_evidence(
    assets: Mapping[str, Any],
    geometry: Mapping[str, Any],
    config: Union[Mapping[str, Any], Path, str],
) -> dict[str, Any]:
    schema = "mid360_fmb1_w02_attempt2_deep_geometry_verification_v1"
    try:
        _scan_forbidden_fields(geometry)
        observed = analyze_geometry_only(
            assets,
            runtime_dir=Path("."),
            config=_load_config(config),
            write_manifest=False,
        )
        if _geometry_compare_payload(geometry) != _geometry_compare_payload(observed):
            raise ValueError("independent geometry recomputation differs")
        scenes = geometry.get("scene_summaries")
        metrics = geometry.get("snapshot_metrics")
        stations = geometry.get("station_summaries")
        if not isinstance(metrics, list) or len(metrics) != 30:
            raise ValueError("geometry metric count is not 30")
        if not isinstance(stations, list) or len(stations) != 3:
            raise ValueError("geometry station summary count is not 3")
        if not isinstance(scenes, list) or len(scenes) != 1:
            raise ValueError("geometry scene summary count is not 1")
        scene = scenes[0]
        if scene.get("scene_id") != "FMB1_W02":
            raise ValueError("geometry scene is not FMB1_W02")
        for counter in (
            "open3d_registration_call_count",
            "pcl_cli_invocation_count",
            "other_registration_process_count",
            "formal_trial_count",
        ):
            if int(geometry.get(counter, -1)) != 0:
                raise ValueError(f"nonzero geometry counter: {counter}")
    except Exception as exc:
        return _failure(schema, f"{type(exc).__name__}: {exc}")
    admitted = bool(
        scene.get("final_geometry_class") == "WEAK"
        and scene.get("geometry_admission_status") == "GEOMETRY_ADMITTED"
    )
    return {
        "schema": schema,
        "status": "PASS" if admitted else "FAIL",
        "pass": admitted,
        "NO_FORMAL_REGISTRATION": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
        "geometry_row_count": 30,
        "station_count": 3,
        "scene_count": 1,
        "FMB1_W02_ATTEMPT2_FINAL_GEOMETRY_CLASS": scene.get(
            "final_geometry_class"
        ),
        "FMB1_W02_ATTEMPT2_ADMISSION_PASS": admitted,
        "failure_count": 0 if admitted else 1,
        "failures": [] if admitted else ["W02 attempt 2 is not admitted as WEAK"],
    }


__all__ = [
    "STATION_KEYS",
    "verify_w02_attempt2_geometry_evidence",
    "verify_w02_attempt2_ros_evidence",
]
