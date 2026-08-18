"""Execute exactly 40 authorized nonformal Open3D/PCL trials and summarize them."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from phase_a_harness.common_association_analysis import (
    prepare_common_association_context,
    safe_analyze_estimated_transform,
)
from phase_a_harness.mid360_pilot.bag_reader import PilotBagError, sha256_file
from phase_a_harness.mid360_pilot.debug_registration import (
    IDENTITY,
    _parameter_sha,
    _run_open3d_trial,
    _run_pcl_trial,
    array_sha256,
    load_canonical_npy,
    verify_transform_convention,
)
from phase_a_harness.mid360_pilot.pilot_pipeline import (
    DEFAULT_BACKEND_PARAMETER_CONTRACT,
    DEFAULT_PCL_EXECUTABLE,
    validate_runtime_root,
)
from phase_a_harness.open3d_backend import validate_open3d_version

from . import TWO_SCENE_FLAGS
from .pipeline import (
    DEFAULT_RUNTIME_ROOT,
    describe,
    read_csv,
    read_json,
    utc_now,
    write_csv,
    write_json,
)


RESULT_OUTPUTS = (
    "backend_execution_started.json",
    "transform_convention.json",
    "open3d_results.csv",
    "pcl_results.csv",
    "backend_execution_ledger.csv",
    "backend_input_identity_audit.csv",
    "mid360_two_scene_reassociation.csv",
    "mid360_two_scene_statistics.json",
    "mid360_two_scene_readiness.json",
    "mid360_two_scene_summary.json",
    "SHA256SUMS",
)


def _rho(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def _canonical_parameter_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    for backend in ("open3d", "pcl"):
        actual = _parameter_sha(contract[backend]["parameters"])
        if actual != contract[backend]["canonical_sha256"]:
            raise PilotBagError(f"frozen {backend} parameter SHA mismatch")
    return contract


def _trial_manifest(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "query_id": row["snapshot_id"],
        "query_timestamp": row["query_timestamp"],
        "source_sha256": row["source_npy_sha256"],
        "source_array_sha256": row["source_array_sha256"],
        "target_sha256": row["target_npy_sha256"],
        "target_array_sha256": row["target_array_sha256"],
        "source_point_count": row["source_point_count"],
        "target_point_count": row["target_point_count"],
    }


def _decorate_result(
    result: Mapping[str, Any], source: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "scene_id": source["scene_id"],
        "semantic_scene": source["semantic_scene"],
        "scene_type": source["scene_type"],
        "snapshot_id": source["snapshot_id"],
        "trial_id": f"{source['snapshot_id']}::{result['backend']}",
        **result,
        **TWO_SCENE_FLAGS,
    }


def _authenticate_authorization(
    runtime: Path,
    manifest: Sequence[Mapping[str, str]],
    parameter_contract_path: Path,
) -> dict[str, Any]:
    authorization = read_json(
        runtime / "mid360_two_scene_debug_registration_authorization.json"
    )
    checks = {
        "pilot_only": authorization.get("PILOT_ONLY") is True,
        "formal_false": authorization.get("FORMAL_MEASUREMENT_RESULT") is False,
        "max_trials_40": authorization.get("MAX_ALLOWED_TRIALS") == 40,
        "authorized_trials_40": authorization.get("authorized_trial_count") == 40,
        "authorized_snapshots_20": authorization.get("authorized_snapshot_count") == 20,
        "manifest_rows_20": len(manifest) == 20,
        "selection_csv_fixed": authorization.get("query_selection_csv_sha256")
        == sha256_file(runtime / "query_selection_frozen.csv"),
        "selection_json_fixed": authorization.get("query_selection_json_sha256")
        == sha256_file(runtime / "query_selection_frozen.json"),
        "geometry_metrics_fixed": authorization.get("geometry_only_metrics_sha256")
        == sha256_file(runtime / "geometry_only_metrics.csv"),
        "geometry_summary_fixed": authorization.get("geometry_scene_summary_sha256")
        == sha256_file(runtime / "geometry_scene_summary.json"),
        "canonical_manifest_fixed": authorization.get(
            "canonical_input_manifest_sha256"
        )
        == sha256_file(runtime / "canonical_input_manifest.csv"),
        "target_manifest_fixed": authorization.get("target_map_manifest_sha256")
        == sha256_file(runtime / "target_map_manifest.csv"),
        "preprocessing_fixed": authorization.get("preprocessing_contract_sha256")
        == sha256_file(runtime / "mid360_two_scene_preprocessing_contract.json"),
        "parameter_contract_fixed": authorization.get(
            "backend_parameter_contract_sha256"
        )
        == sha256_file(parameter_contract_path),
        "identity_t0": authorization.get("T0") == IDENTITY.tolist(),
        "selection_precedes_backend": authorization.get(
            "query_selection_frozen_before_backend"
        )
        is True,
        "formal_multisite_not_authorized": authorization.get(
            "formal_multisite_experiment_authorized"
        )
        is False,
    }
    expected_trials = {
        (row["snapshot_id"], backend)
        for row in manifest
        for backend in ("open3d_point_to_plane", "pcl_point_to_plane")
    }
    actual_trials = {
        (str(row["snapshot_id"]), str(row["backend"]))
        for row in authorization.get("authorized_trials", [])
    }
    checks["exact_trial_set"] = actual_trials == expected_trials
    snapshots = {str(row["snapshot_id"]): row for row in authorization.get("snapshots", [])}
    checks["exact_snapshot_set"] = set(snapshots) == {
        row["snapshot_id"] for row in manifest
    }
    for row in manifest:
        snapshot = snapshots.get(row["snapshot_id"], {})
        if not (
            snapshot.get("source_npy_sha256") == row["source_npy_sha256"]
            and snapshot.get("source_array_sha256") == row["source_array_sha256"]
            and snapshot.get("target_npy_sha256") == row["target_npy_sha256"]
            and snapshot.get("target_array_sha256") == row["target_array_sha256"]
            and float(snapshot.get("query_timestamp", float("nan")))
            == float(row["query_timestamp"])
        ):
            checks[f"snapshot_binding_{row['snapshot_id']}"] = False
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise PilotBagError(f"40-trial authorization authentication failed: {failed}")
    return {"authorization": authorization, "checks": checks}


def _authenticate_inputs(
    manifest: Sequence[Mapping[str, str]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sources: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for row in manifest:
        snapshot_id = row["snapshot_id"]
        source_path = Path(row["source_path"]).resolve(strict=True)
        target_path = Path(row["target_path"]).resolve(strict=True)
        source = load_canonical_npy(source_path)
        target = targets.get(row["scene_id"])
        if target is None:
            target = load_canonical_npy(target_path)
            targets[row["scene_id"]] = target
        if not (
            sha256_file(source_path) == row["source_npy_sha256"]
            and array_sha256(source) == row["source_array_sha256"]
            and sha256_file(target_path) == row["target_npy_sha256"]
            and array_sha256(target) == row["target_array_sha256"]
            and json.loads(row["T0"]) == IDENTITY.tolist()
            and row["little_endian_float64"] == "true"
            and row["c_contiguous"] == "true"
            and row["finite"] == "true"
        ):
            raise PilotBagError(f"canonical input authentication failed: {snapshot_id}")
        sources[snapshot_id] = source
    if len(sources) != 20 or set(targets) != {"R_TEST_01", "W_TEST_01"}:
        raise PilotBagError("canonical inputs do not close to 20 sources and 2 targets")
    return sources, targets


def _input_identity_audit(
    manifest: Sequence[Mapping[str, str]],
    open3d_rows: Sequence[Mapping[str, Any]],
    pcl_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    open_by_id = {str(row["snapshot_id"]): row for row in open3d_rows}
    pcl_by_id = {str(row["snapshot_id"]): row for row in pcl_rows}
    output: list[dict[str, Any]] = []
    for source in manifest:
        snapshot_id = source["snapshot_id"]
        open_row = open_by_id[snapshot_id]
        pcl_row = pcl_by_id[snapshot_id]
        source_file_match = (
            open_row["source_sha256"]
            == pcl_row["source_sha256"]
            == source["source_npy_sha256"]
        )
        target_file_match = (
            open_row["target_sha256"]
            == pcl_row["target_sha256"]
            == source["target_npy_sha256"]
        )
        source_array_match = (
            open_row["source_array_sha256"]
            == pcl_row["source_array_sha256"]
            == source["source_array_sha256"]
        )
        target_array_match = (
            open_row["target_array_sha256"]
            == pcl_row["target_array_sha256"]
            == source["target_array_sha256"]
        )
        output.append(
            {
                "scene_id": source["scene_id"],
                "scene_type": source["scene_type"],
                "snapshot_id": snapshot_id,
                "query_timestamp": float(source["query_timestamp"]),
                "open3d_source_sha256": open_row["source_sha256"],
                "pcl_source_sha256": pcl_row["source_sha256"],
                "open3d_target_sha256": open_row["target_sha256"],
                "pcl_target_sha256": pcl_row["target_sha256"],
                "open3d_source_array_sha256": open_row["source_array_sha256"],
                "pcl_source_array_sha256": pcl_row["source_array_sha256"],
                "open3d_target_array_sha256": open_row["target_array_sha256"],
                "pcl_target_array_sha256": pcl_row["target_array_sha256"],
                "source_npy_sha_identical": source_file_match,
                "target_npy_sha_identical": target_file_match,
                "source_array_sha_identical": source_array_match,
                "target_array_sha_identical": target_array_match,
                "input_identity_pass": source_file_match
                and target_file_match
                and source_array_match
                and target_array_match,
                **TWO_SCENE_FLAGS,
            }
        )
    return output


def _reassociation(
    manifest: Sequence[Mapping[str, str]],
    sources: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results_by_snapshot: dict[str, list[Mapping[str, Any]]] = {}
    for row in results:
        results_by_snapshot.setdefault(str(row["snapshot_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for source_row in manifest:
        snapshot_id = source_row["snapshot_id"]
        context = prepare_common_association_context(
            sources[snapshot_id],
            targets[source_row["scene_id"]],
            IDENTITY,
            snapshot_id=snapshot_id,
        )
        for result in results_by_snapshot[snapshot_id]:
            if result["T_est"] is None:
                diagnostics: dict[str, Any] = {
                    "snapshot_id": snapshot_id,
                    "common_association_valid": False,
                    "common_association_invalid_reason": "OTHER",
                }
            else:
                diagnostics = dict(
                    safe_analyze_estimated_transform(
                        context,
                        np.asarray(result["T_est"], dtype=np.float64),
                        identifiers={
                            "scene_id": source_row["scene_id"],
                            "scene_type": source_row["scene_type"],
                            "snapshot_id": snapshot_id,
                            "query_timestamp": float(source_row["query_timestamp"]),
                            "backend": result["backend"],
                        },
                    )
                )
            output.append(
                {
                    "scene_id": source_row["scene_id"],
                    "semantic_scene": source_row["semantic_scene"],
                    "scene_type": source_row["scene_type"],
                    "snapshot_id": snapshot_id,
                    "query_timestamp": float(source_row["query_timestamp"]),
                    "backend": result["backend"],
                    **diagnostics,
                    **TWO_SCENE_FLAGS,
                }
            )
    return output


def _translation_direction(
    open_row: Mapping[str, Any], pcl_row: Mapping[str, Any]
) -> float | None:
    first = np.asarray(
        [open_row["translation_x_m"], open_row["translation_y_m"], open_row["translation_z_m"]],
        dtype=np.float64,
    )
    second = np.asarray(
        [pcl_row["translation_x_m"], pcl_row["translation_y_m"], pcl_row["translation_z_m"]],
        dtype=np.float64,
    )
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return None if denominator <= 1.0e-15 else float(np.dot(first, second) / denominator)


def _statistics(
    open3d_rows: Sequence[Mapping[str, Any]],
    pcl_rows: Sequence[Mapping[str, Any]],
    reassociation: Sequence[Mapping[str, Any]],
    geometry_summary: Mapping[str, Any],
) -> dict[str, Any]:
    by_backend = {"open3d": list(open3d_rows), "pcl": list(pcl_rows)}
    backend_scene: dict[str, Any] = {}
    turnover_lookup = {
        (str(row["backend"]), str(row["snapshot_id"])): row for row in reassociation
    }
    turnover_correlations: dict[str, Any] = {}
    for backend, rows in by_backend.items():
        backend_scene[backend] = {}
        turnover_correlations[backend] = {}
        for scene_id in ("R_TEST_01", "W_TEST_01"):
            selected = [
                row for row in rows if row["scene_id"] == scene_id and row["finite_result"]
            ]
            translations = [float(row["translation_norm_m"]) for row in selected]
            rotations = [float(row["rotation_angle_rad"]) for row in selected]
            turnovers = [
                float(turnover_lookup[(str(row["backend"]), str(row["snapshot_id"]))]["correspondence_turnover"])
                for row in selected
                if turnover_lookup[(str(row["backend"]), str(row["snapshot_id"]))].get(
                    "correspondence_turnover"
                )
                is not None
            ]
            paired_translations = [
                float(row["translation_norm_m"])
                for row in selected
                if turnover_lookup[(str(row["backend"]), str(row["snapshot_id"]))].get(
                    "correspondence_turnover"
                )
                is not None
            ]
            backend_scene[backend][scene_id] = {
                "translation_norm_m": describe(translations),
                "rotation_angle_rad": describe(rotations),
                "rotation_angle_deg": describe([math.degrees(value) for value in rotations]),
                "correspondence_turnover": describe(turnovers),
                "solver_success_count": sum(row["solver_success"] is True for row in selected),
                "finite_result_count": len(selected),
            }
            turnover_correlations[backend][scene_id] = _rho(
                turnovers, paired_translations
            )
        finite = [row for row in rows if row["finite_result"]]
        pooled_turnover = [
            float(turnover_lookup[(str(row["backend"]), str(row["snapshot_id"]))]["correspondence_turnover"])
            for row in finite
            if turnover_lookup[(str(row["backend"]), str(row["snapshot_id"]))].get(
                "correspondence_turnover"
            )
            is not None
        ]
        pooled_translation = [
            float(row["translation_norm_m"])
            for row in finite
            if turnover_lookup[(str(row["backend"]), str(row["snapshot_id"]))].get(
                "correspondence_turnover"
            )
            is not None
        ]
        turnover_correlations[backend]["pooled"] = _rho(
            pooled_turnover, pooled_translation
        )
    comparisons: dict[str, Any] = {}
    for backend in ("open3d", "pcl"):
        rich = backend_scene[backend]["R_TEST_01"]
        weak = backend_scene[backend]["W_TEST_01"]
        rich_t = float(rich["translation_norm_m"]["median"])
        weak_t = float(weak["translation_norm_m"]["median"])
        rich_r = float(rich["rotation_angle_rad"]["median"])
        weak_r = float(weak["rotation_angle_rad"]["median"])
        rich_turn = float(rich["correspondence_turnover"]["median"])
        weak_turn = float(weak["correspondence_turnover"]["median"])
        comparisons[backend] = {
            "weak_minus_rich_median_translation_m": weak_t - rich_t,
            "weak_div_rich_median_translation": weak_t / rich_t,
            "weak_minus_rich_median_rotation_rad": weak_r - rich_r,
            "weak_div_rich_median_rotation": weak_r / rich_r,
            "weak_minus_rich_median_turnover": weak_turn - rich_turn,
            "weak_translation_greater_than_rich": weak_t > rich_t,
            "weak_turnover_greater_than_rich": weak_turn > rich_turn,
        }
    open_by_id = {str(row["snapshot_id"]): row for row in open3d_rows}
    pcl_by_id = {str(row["snapshot_id"]): row for row in pcl_rows}
    paired_ids = [
        snapshot_id
        for snapshot_id in open_by_id
        if open_by_id[snapshot_id]["finite_result"] and pcl_by_id[snapshot_id]["finite_result"]
    ]
    direction_rows = [
        {
            "scene_id": open_by_id[snapshot_id]["scene_id"],
            "snapshot_id": snapshot_id,
            "translation_direction_cosine": _translation_direction(
                open_by_id[snapshot_id], pcl_by_id[snapshot_id]
            ),
        }
        for snapshot_id in paired_ids
    ]
    cross_backend_rho: dict[str, float | None] = {}
    for label, scene_id in (
        ("pooled", None),
        ("rich", "R_TEST_01"),
        ("weak", "W_TEST_01"),
    ):
        selected_ids = [
            snapshot_id
            for snapshot_id in paired_ids
            if scene_id is None or open_by_id[snapshot_id]["scene_id"] == scene_id
        ]
        cross_backend_rho[label] = _rho(
            [float(open_by_id[value]["translation_norm_m"]) for value in selected_ids],
            [float(pcl_by_id[value]["translation_norm_m"]) for value in selected_ids],
        )
    direction_summary: dict[str, Any] = {}
    for label, scene_id in (
        ("pooled", None),
        ("rich", "R_TEST_01"),
        ("weak", "W_TEST_01"),
    ):
        values = [
            float(row["translation_direction_cosine"])
            for row in direction_rows
            if row["translation_direction_cosine"] is not None
            and (scene_id is None or row["scene_id"] == scene_id)
        ]
        direction_summary[label] = describe(values)
    signal_consistent = bool(
        comparisons["open3d"]["weak_translation_greater_than_rich"]
        and comparisons["pcl"]["weak_translation_greater_than_rich"]
    )
    open_difference = float(
        comparisons["open3d"]["weak_minus_rich_median_translation_m"]
    )
    pcl_difference = float(comparisons["pcl"]["weak_minus_rich_median_translation_m"])
    backend_trend = bool(
        open_difference != 0.0
        and pcl_difference != 0.0
        and (open_difference > 0.0) == (pcl_difference > 0.0)
    )
    reassociation_consistent = bool(
        comparisons["open3d"]["weak_turnover_greater_than_rich"]
        and comparisons["pcl"]["weak_turnover_greater_than_rich"]
    )
    return {
        "schema": "mid360_two_scene_statistics_v1",
        **TWO_SCENE_FLAGS,
        "backend_scene_statistics": backend_scene,
        "weak_rich_comparisons": comparisons,
        "cross_backend_translation_spearman_rho": cross_backend_rho,
        "translation_direction_cosine_by_snapshot": direction_rows,
        "translation_direction_cosine_summary": direction_summary,
        "turnover_translation_spearman_rho": turnover_correlations,
        "PILOT_SIGNAL_DIRECTIONALLY_CONSISTENT": signal_consistent,
        "PILOT_GEOMETRY_DIRECTIONALLY_CONSISTENT": geometry_summary[
            "PILOT_GEOMETRY_DIRECTIONALLY_CONSISTENT"
        ],
        "PILOT_REASSOCIATION_DIRECTIONALLY_CONSISTENT": reassociation_consistent,
        "PILOT_BACKEND_TREND_CONSISTENT": backend_trend,
        "descriptive_only_no_independent_scene_inference": True,
        "formal_p_values_reported": False,
    }


def _statistics_markdown(statistics: Mapping[str, Any]) -> str:
    lines = [
        "# Mid-360 two-scene Pilot descriptive statistics",
        "",
        "`PILOT_NONFORMAL_DO_NOT_CITE=true`; one station per semantic class; no population inference.",
        "",
    ]
    for backend in ("open3d", "pcl"):
        lines.extend([f"## {backend}", ""])
        for scene_id, label in (("R_TEST_01", "Rich"), ("W_TEST_01", "Weak")):
            stats = statistics["backend_scene_statistics"][backend][scene_id]
            translation = stats["translation_norm_m"]
            rotation = stats["rotation_angle_deg"]
            turnover = stats["correspondence_turnover"]
            lines.extend(
                [
                    f"- {label} translation median/q95 (m): {translation['median']} / {translation['q95']}",
                    f"- {label} rotation median/max (deg): {rotation['median']} / {rotation['max']}",
                    f"- {label} turnover median/range: {turnover['median']} / {turnover['min']}–{turnover['max']}",
                ]
            )
        comparison = statistics["weak_rich_comparisons"][backend]
        lines.extend(
            [
                f"- Weak/Rich median translation ratio: {comparison['weak_div_rich_median_translation']}",
                "",
            ]
        )
    lines.extend(
        [
            f"PILOT_SIGNAL_DIRECTIONALLY_CONSISTENT={str(statistics['PILOT_SIGNAL_DIRECTIONALLY_CONSISTENT']).lower()}",
            "",
            "These snapshots are repeated observations within two Pilot stations, not independent scenes.",
        ]
    )
    return "\n".join(lines) + "\n"


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    return f"""# Mid-360 independent two-scene Pilot summary

`MID360_TWO_SCENE_PILOT=true`, `INDEPENDENT_MAP_QUERY_ACQUISITION=true`, `FORMAL_MEASUREMENT_RESULT=false`, `PILOT_NONFORMAL_DO_NOT_CITE=true`.

- Open3D finite: {summary['open3d_finite_count']}/20
- PCL finite: {summary['pcl_finite_count']}/20
- Exact authorized Pilot trials: {summary['executed_trial_count']}/40
- Backend input identity: {summary['backend_input_identity_pass']}
- Transform convention verified: {summary['transform_convention_verified']}
- Geometry weaker direction: {summary['PILOT_GEOMETRY_DIRECTIONALLY_CONSISTENT']}
- Signal directionally consistent: {summary['PILOT_SIGNAL_DIRECTIONALLY_CONSISTENT']}
- Readiness class: {summary['readiness_class']}
- MID360_TWO_SCENE_PILOT_READY: {summary['MID360_TWO_SCENE_PILOT_READY']}

These 40 trials are nonformal two-station software/debug evidence and are not paper Measurement Results.
"""


def write_runtime_sha256s(runtime: Path) -> None:
    paths = sorted(
        path
        for path in runtime.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(runtime).as_posix()}\n" for path in paths
    ]
    (runtime / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def execute_two_scene_registration(
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    *,
    parameter_contract_path: Path = DEFAULT_BACKEND_PARAMETER_CONTRACT,
    pcl_executable: Path = DEFAULT_PCL_EXECUTABLE,
) -> dict[str, Any]:
    runtime = validate_runtime_root(runtime_root).resolve(strict=True)
    existing = [name for name in RESULT_OUTPUTS if (runtime / name).exists()]
    if existing:
        raise PilotBagError(
            "backend execution artifacts already exist; refusing retry/overwrite: "
            + ", ".join(existing)
        )
    manifest = read_csv(runtime / "canonical_input_manifest.csv")
    if len(manifest) != 20:
        raise PilotBagError("canonical manifest must contain exactly 20 snapshots")
    if sum(row["scene_id"] == "R_TEST_01" for row in manifest) != 10 or sum(
        row["scene_id"] == "W_TEST_01" for row in manifest
    ) != 10:
        raise PilotBagError("canonical manifest scene balance changed")
    parameter_contract_path = parameter_contract_path.resolve(strict=True)
    parameter_contract = _canonical_parameter_contract(parameter_contract_path)
    if validate_open3d_version() != parameter_contract["open3d"]["parameters"]["version"]:
        raise PilotBagError("Open3D version does not match frozen contract")
    pcl_cli = pcl_executable.resolve(strict=True)
    authorization = _authenticate_authorization(
        runtime, manifest, parameter_contract_path
    )
    sources, targets = _authenticate_inputs(manifest)
    convention = verify_transform_convention(parameter_contract, pcl_cli)
    convention.update(TWO_SCENE_FLAGS)
    write_json(runtime / "transform_convention.json", convention)
    if convention["status"] != "PASS":
        raise PilotBagError("transform convention fixture failed")
    started = {
        "schema": "mid360_two_scene_backend_execution_started_v1",
        **TWO_SCENE_FLAGS,
        "started_at_utc": utc_now(),
        "query_selection_csv_sha256": sha256_file(
            runtime / "query_selection_frozen.csv"
        ),
        "geometry_only_metrics_sha256": sha256_file(
            runtime / "geometry_only_metrics.csv"
        ),
        "authorization_sha256": sha256_file(
            runtime / "mid360_two_scene_debug_registration_authorization.json"
        ),
        "authorized_pilot_trial_count": 40,
        "nontrial_transform_fixture_backend_invocation_count": 2,
        "pilot_backend_trial_count_before_start": 0,
    }
    write_json(runtime / "backend_execution_started.json", started)
    open3d_rows: list[dict[str, Any]] = []
    pcl_rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for row in manifest:
        trial_started = utc_now()
        result = _run_open3d_trial(
            sources[row["snapshot_id"]],
            targets[row["scene_id"]],
            _trial_manifest(row),
            parameter_contract["open3d"]["parameters"],
        )
        decorated = _decorate_result(result, row)
        open3d_rows.append(decorated)
        ledger.append(
            {
                "execution_index": len(ledger) + 1,
                "trial_id": decorated["trial_id"],
                "scene_id": row["scene_id"],
                "snapshot_id": row["snapshot_id"],
                "backend": decorated["backend"],
                "started_at_utc": trial_started,
                "completed_at_utc": utc_now(),
                "solver_success": decorated["solver_success"],
                "finite_result": decorated["finite_result"],
                **TWO_SCENE_FLAGS,
            }
        )
    for row in manifest:
        trial_started = utc_now()
        result = _run_pcl_trial(
            sources[row["snapshot_id"]],
            targets[row["scene_id"]],
            _trial_manifest(row),
            parameter_contract["pcl"]["parameters"],
            pcl_cli,
        )
        decorated = _decorate_result(result, row)
        pcl_rows.append(decorated)
        ledger.append(
            {
                "execution_index": len(ledger) + 1,
                "trial_id": decorated["trial_id"],
                "scene_id": row["scene_id"],
                "snapshot_id": row["snapshot_id"],
                "backend": decorated["backend"],
                "started_at_utc": trial_started,
                "completed_at_utc": utc_now(),
                "solver_success": decorated["solver_success"],
                "finite_result": decorated["finite_result"],
                **TWO_SCENE_FLAGS,
            }
        )
    write_csv(runtime / "open3d_results.csv", open3d_rows)
    write_csv(runtime / "pcl_results.csv", pcl_rows)
    write_csv(runtime / "backend_execution_ledger.csv", ledger)
    identity = _input_identity_audit(manifest, open3d_rows, pcl_rows)
    write_csv(runtime / "backend_input_identity_audit.csv", identity)
    reassociation = _reassociation(
        manifest, sources, targets, [*open3d_rows, *pcl_rows]
    )
    write_csv(runtime / "mid360_two_scene_reassociation.csv", reassociation)
    geometry_summary = read_json(runtime / "geometry_scene_summary.json")
    statistics = _statistics(
        open3d_rows, pcl_rows, reassociation, geometry_summary
    )
    write_json(runtime / "mid360_two_scene_statistics.json", statistics)
    (runtime / "mid360_two_scene_statistics.md").write_text(
        _statistics_markdown(statistics), encoding="utf-8"
    )
    all_results = [*open3d_rows, *pcl_rows]
    finite_count = sum(row["finite_result"] is True for row in all_results)
    failures = [
        row["trial_id"] for row in all_results if row["solver_success"] is not True
    ]
    translation_suspects = [
        row["trial_id"]
        for row in all_results
        if row["translation_norm_m"] is not None
        and float(row["translation_norm_m"]) > 1.0
    ]
    rotation_suspects = [
        row["trial_id"]
        for row in all_results
        if row["rotation_angle_deg"] is not None
        and float(row["rotation_angle_deg"]) > 10.0
    ]
    open_by_id = {str(row["snapshot_id"]): row for row in open3d_rows}
    pcl_by_id = {str(row["snapshot_id"]): row for row in pcl_rows}
    opposite_direction = []
    for row in statistics["translation_direction_cosine_by_snapshot"]:
        cosine = row["translation_direction_cosine"]
        snapshot_id = row["snapshot_id"]
        if (
            cosine is not None
            and float(cosine) < -0.5
            and float(open_by_id[snapshot_id]["translation_norm_m"]) > 0.01
            and float(pcl_by_id[snapshot_id]["translation_norm_m"]) > 0.01
        ):
            opposite_direction.append(snapshot_id)
    input_identity = all(row["input_identity_pass"] is True for row in identity)
    exact_trials = (
        len(open3d_rows) == 20
        and len(pcl_rows) == 20
        and len(ledger) == 40
        and len({row["trial_id"] for row in ledger}) == 40
    )
    obvious_bug = bool(translation_suspects or rotation_suspects or opposite_direction)
    hard_ready = bool(
        exact_trials
        and finite_count == 40
        and input_identity
        and convention.get("transform_convention_verified") is True
        and authorization["authorization"]["MAX_ALLOWED_TRIALS"] == 40
        and not obvious_bug
    )
    limitations = ["ACCELERATION_UNIT_UNCONFIRMED"]
    if failures:
        limitations.append("SOLVER_NONCONVERGENCE_RECORDED")
    readiness_class = "READY_WITH_LIMITATION" if hard_ready else "FAIL"
    readiness = {
        "schema": "mid360_two_scene_readiness_v1",
        **TWO_SCENE_FLAGS,
        "MID360_TWO_SCENE_PILOT_READY": hard_ready,
        "readiness_class": readiness_class,
        "bag_authentication_count": 4,
        "target_map_count": 2,
        "snapshot_count": 20,
        "open3d_result_count": len(open3d_rows),
        "pcl_result_count": len(pcl_rows),
        "finite_result_count": finite_count,
        "authorized_trial_count": 40,
        "executed_trial_count": len(ledger),
        "backend_input_identity_pass": input_identity,
        "transform_convention_verified": convention.get(
            "transform_convention_verified"
        ),
        "solver_failures": failures,
        "translation_over_1m_suspects": translation_suspects,
        "rotation_over_10deg_suspects": rotation_suspects,
        "opposite_large_direction_suspects": opposite_direction,
        "obvious_unit_or_frame_bug_detected": obvious_bug,
        "documented_limitations": limitations,
        "independent_verifier_pass": None,
        "status": "PASS" if hard_ready else "FAIL",
    }
    write_json(runtime / "mid360_two_scene_readiness.json", readiness)
    summary = {
        "schema": "mid360_two_scene_summary_v1",
        **TWO_SCENE_FLAGS,
        "MID360_TWO_SCENE_PILOT_READY": hard_ready,
        "readiness_class": readiness_class,
        "documented_limitations": limitations,
        "open3d_solver_success_count": sum(
            row["solver_success"] is True for row in open3d_rows
        ),
        "pcl_solver_success_count": sum(
            row["solver_success"] is True for row in pcl_rows
        ),
        "open3d_finite_count": sum(row["finite_result"] is True for row in open3d_rows),
        "pcl_finite_count": sum(row["finite_result"] is True for row in pcl_rows),
        "solver_failures": failures,
        "executed_trial_count": len(ledger),
        "backend_input_identity_pass": input_identity,
        "transform_convention_verified": convention.get(
            "transform_convention_verified"
        ),
        "obvious_unit_error_detected": bool(translation_suspects),
        "frame_or_transform_direction_error_detected": bool(
            rotation_suspects or opposite_direction
        ),
        "PILOT_SIGNAL_DIRECTIONALLY_CONSISTENT": statistics[
            "PILOT_SIGNAL_DIRECTIONALLY_CONSISTENT"
        ],
        "PILOT_GEOMETRY_DIRECTIONALLY_CONSISTENT": statistics[
            "PILOT_GEOMETRY_DIRECTIONALLY_CONSISTENT"
        ],
        "PILOT_REASSOCIATION_DIRECTIONALLY_CONSISTENT": statistics[
            "PILOT_REASSOCIATION_DIRECTIONALLY_CONSISTENT"
        ],
        "PILOT_BACKEND_TREND_CONSISTENT": statistics[
            "PILOT_BACKEND_TREND_CONSISTENT"
        ],
        "SEMANTIC_LABEL_GEOMETRY_MISMATCH": geometry_summary[
            "SEMANTIC_LABEL_GEOMETRY_MISMATCH"
        ],
        "statistics": statistics,
        "formal_result_eligibility": "NO",
        "formal_multisite_experiment_started": False,
        "independent_verifier_pass": None,
        "backend_parameter_contract_path": str(parameter_contract_path),
        "backend_parameter_contract_sha256": sha256_file(parameter_contract_path),
        "open3d_version": validate_open3d_version(),
        "pcl_executable_path": str(pcl_cli),
        "pcl_executable_sha256": sha256_file(pcl_cli),
        "pcl_version": next(
            (row.get("pcl_version") for row in pcl_rows if row.get("pcl_version")),
            None,
        ),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    write_json(runtime / "mid360_two_scene_summary.json", summary)
    (runtime / "mid360_two_scene_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    write_runtime_sha256s(runtime)
    return summary


__all__ = [
    "execute_two_scene_registration",
    "write_runtime_sha256s",
]
