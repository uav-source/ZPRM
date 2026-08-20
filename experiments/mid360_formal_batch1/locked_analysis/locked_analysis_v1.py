"""Complete C1+C2 locked analysis with fixture-only qualification entry points."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from jsonschema import Draft202012Validator

from .authoritative_outcomes_v1 import select_authoritative_outcomes
from .contract_v1 import (
    BACKENDS,
    FUTURE_OUTPUT_FILES,
    PHYSICAL_REFERENCE_SEMANTICS,
    SCENE_ORDER,
    STATION_ORDER,
)
from .cross_backend_v1 import (
    formal_spearman,
    scene_ordering_agreement,
    translation_direction_cosine,
)
from .descriptive_summary_v1 import summarize_scene, summarize_station
from .exact_scene_permutation_v1 import exact_weak_greater_than_rich
from .reassociation_v1 import (
    accepted_source_turnover_descriptive,
    centered_association,
    centered_permutation_sensitivity,
    formal_scene_turnover,
    scene_turnover_translation_spearman,
)
from .systematic_component_v1 import (
    scene_systematic_fraction,
    station_systematic_fraction,
    systematic_weak_rich_descriptive,
)


class LockedAnalysisError(RuntimeError):
    """Raised when analysis inputs or output persistence violate the lock contract."""


def _station_short(station_id: str) -> str:
    station = str(station_id).split("_")[-1]
    if station not in STATION_ORDER:
        raise LockedAnalysisError(f"unexpected station ID: {station_id}")
    return station


def _identity(outcome: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, str]:
    row = outcome.get("authoritative_row") or plan
    return {
        "scene_id": str(row["scene_id"]),
        "station_id": _station_short(str(row["station_id"])),
        "snapshot_id": str(row["snapshot_id"]),
        "backend": str(row["backend"]),
    }


def build_fixture_plan_and_attempts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct an entirely artificial 6x3x10x2 fixture."""

    plan_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(SCENE_ORDER, start=1):
        for station_index, station in enumerate(STATION_ORDER, start=1):
            for snapshot_index in range(1, 11):
                snapshot_id = f"{scene}_{station}_Q{snapshot_index:02d}"
                for backend_index, backend in enumerate(BACKENDS):
                    trial_id = f"FIXTURE-{snapshot_id}-{backend_index}"
                    plan_rows.append(
                        {
                            "trial_id": trial_id,
                            "scene_id": scene,
                            "station_id": station,
                            "snapshot_id": snapshot_id,
                            "backend": backend,
                        }
                    )
                    base = (
                        scene_index * 0.01
                        + station_index * 0.001
                        + snapshot_index * 0.0001
                        + backend_index * 0.00003
                    )
                    vector = np.array(
                        [
                            base,
                            ((-1) ** snapshot_index) * base * 0.2,
                            (station_index - 2) * base * 0.1,
                        ],
                        dtype=np.float64,
                    )
                    rotation_rad = base * 0.05
                    turnover = (
                        (snapshot_index * 7 + station_index * 3 + scene_index) % 29
                    ) / 29.0
                    attempts.append(
                        {
                            **plan_rows[-1],
                            "attempt": 1,
                            "schema_valid": True,
                            "infrastructure_status": "OK",
                            "finite_result": True,
                            "scientific_status": "COMPLETED",
                            "solver_status": (
                                "MAX_ITERATIONS"
                                if scene_index == 1
                                and station_index == 1
                                and snapshot_index == 1
                                and backend_index == 0
                                else "CONVERGED"
                            ),
                            "translation_x_m": float(vector[0]),
                            "translation_y_m": float(vector[1]),
                            "translation_z_m": float(vector[2]),
                            "translation_norm_m": float(np.linalg.norm(vector)),
                            "rotation_angle_rad": rotation_rad,
                            "rotation_angle_deg": float(np.degrees(rotation_rad)),
                            "correspondence_turnover": turnover,
                            "accepted_source_turnover": min(1.0, turnover * 0.8 + 0.05),
                        }
                    )
    if len(plan_rows) != 360 or len(attempts) != 360:
        raise AssertionError("fixture must contain exactly 360 trial identities")
    return plan_rows, attempts


def _group_outcomes(
    plan_rows: list[dict[str, Any]], outcomes: list[dict[str, Any]]
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    plan_by_id = {row["trial_id"]: row for row in plan_rows}
    if len(plan_by_id) != len(plan_rows):
        raise LockedAnalysisError("trial plan IDs must be unique")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    identity_by_trial: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        trial_id = outcome["trial_id"]
        identity = _identity(outcome, plan_by_id[trial_id])
        identity_by_trial[trial_id] = identity
        grouped[
            (identity["backend"], identity["scene_id"], identity["station_id"])
        ].append(outcome)
    for key, values in grouped.items():
        values.sort(key=lambda outcome: identity_by_trial[outcome["trial_id"]]["snapshot_id"])
        if len(values) != 10:
            raise LockedAnalysisError(f"station group {key} must contain 10 planned trials")
    return grouped, identity_by_trial


def analyze_attempts(
    plan_rows: list[dict[str, Any]],
    attempts: Iterable[Mapping[str, Any]],
    *,
    fixture_only: bool,
    analysis_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Run every frozen C1+C2 endpoint from already-loaded validated rows."""

    if len(plan_rows) != 360:
        raise LockedAnalysisError("analysis requires exactly 360 planned trial identities")
    planned_ids = [str(row["trial_id"]) for row in plan_rows]
    outcomes = select_authoritative_outcomes(planned_ids, attempts)
    grouped, identity_by_trial = _group_outcomes(plan_rows, outcomes)

    translation_station: list[dict[str, Any]] = []
    translation_scene: list[dict[str, Any]] = []
    rotation_station: list[dict[str, Any]] = []
    rotation_scene: list[dict[str, Any]] = []
    missingness: list[dict[str, Any]] = []
    scene_translation_medians: dict[str, dict[str, float | None]] = {
        backend: {} for backend in BACKENDS
    }
    station_translation_medians: dict[str, dict[str, float | None]] = {
        backend: {} for backend in BACKENDS
    }
    rotation_scene_medians: dict[str, dict[str, dict[str, float | None]]] = {
        endpoint: {backend: {} for backend in BACKENDS}
        for endpoint in ("rotation_angle_rad", "rotation_angle_deg")
    }

    for backend in BACKENDS:
        for scene in SCENE_ORDER:
            by_station = {
                station: grouped[(backend, scene, station)] for station in STATION_ORDER
            }
            for station in STATION_ORDER:
                station_id = f"{scene}_{station}"
                translation = summarize_station(
                    by_station[station], "translation_norm_m"
                )
                translation.update(
                    {"backend": backend, "scene_id": scene, "station_id": station}
                )
                translation_station.append(translation)
                station_translation_medians[backend][station_id] = translation[
                    "formal_statistics"
                ]["median"]
                missingness.append(
                    {
                        "level": "station",
                        "backend": backend,
                        "scene_id": scene,
                        "station_id": station,
                        **{
                            key: translation[key]
                            for key in (
                                "endpoint",
                                "planned_n",
                                "authoritative_scientific_outcome_n",
                                "finite_endpoint_n",
                                "scientific_undefined_nonfinite_n",
                                "unresolved_infrastructure_failure_n",
                                "resolved_infrastructure_attempt_n",
                                "solver_nonconverged_finite_n",
                                "endpoint_specific_undefined_n",
                                "formal_summary_status",
                            )
                        },
                    }
                )
                for endpoint in ("rotation_angle_rad", "rotation_angle_deg"):
                    summary = summarize_station(by_station[station], endpoint)
                    summary.update(
                        {"backend": backend, "scene_id": scene, "station_id": station}
                    )
                    rotation_station.append(summary)
            scene_summary = summarize_scene(by_station, "translation_norm_m")
            scene_summary.update({"backend": backend, "scene_id": scene})
            translation_scene.append(scene_summary)
            scene_translation_medians[backend][scene] = scene_summary["formal_statistics"][
                "median"
            ]
            missingness.append(
                {
                    "level": "scene",
                    "backend": backend,
                    "scene_id": scene,
                    "station_id": None,
                    **{
                        key: scene_summary[key]
                        for key in (
                            "endpoint",
                            "planned_n",
                            "authoritative_scientific_outcome_n",
                            "finite_endpoint_n",
                            "scientific_undefined_nonfinite_n",
                            "unresolved_infrastructure_failure_n",
                            "resolved_infrastructure_attempt_n",
                            "solver_nonconverged_finite_n",
                            "endpoint_specific_undefined_n",
                            "formal_summary_status",
                        )
                    },
                }
            )
            for endpoint in ("rotation_angle_rad", "rotation_angle_deg"):
                summary = summarize_scene(by_station, endpoint)
                summary.update({"backend": backend, "scene_id": scene})
                rotation_scene.append(summary)
                rotation_scene_medians[endpoint][backend][scene] = summary[
                    "formal_statistics"
                ]["median"]

    primary_inference = [
        {"backend": backend, **exact_weak_greater_than_rich(scene_translation_medians[backend])}
        for backend in BACKENDS
    ]
    rotation_inference = [
        {
            "backend": backend,
            "endpoint": endpoint,
            "role": "SECONDARY",
            **exact_weak_greater_than_rich(rotation_scene_medians[endpoint][backend]),
        }
        for endpoint in ("rotation_angle_rad", "rotation_angle_deg")
        for backend in BACKENDS
    ]
    scene_spearman = formal_spearman(
        scene_translation_medians[BACKENDS[0]],
        scene_translation_medians[BACKENDS[1]],
        ordered_ids=SCENE_ORDER,
        incomplete_status="SCENE_SPEARMAN_UNDEFINED_INCOMPLETE_6_PAIRS",
    )
    station_ids = tuple(
        f"{scene}_{station}" for scene in SCENE_ORDER for station in STATION_ORDER
    )
    station_spearman = formal_spearman(
        station_translation_medians[BACKENDS[0]],
        station_translation_medians[BACKENDS[1]],
        ordered_ids=station_ids,
        incomplete_status="STATION_SPEARMAN_UNDEFINED_INCOMPLETE_18_PAIRS",
    )
    ordering = scene_ordering_agreement(
        scene_translation_medians[BACKENDS[0]],
        scene_translation_medians[BACKENDS[1]],
    )

    outcome_by_snapshot_backend: dict[tuple[str, str], dict[str, Any]] = {}
    for outcome in outcomes:
        identity = identity_by_trial[outcome["trial_id"]]
        outcome_by_snapshot_backend[(identity["snapshot_id"], identity["backend"])] = outcome
    direction_rows = []
    for snapshot_id in sorted({row["snapshot_id"] for row in plan_rows}):
        vectors = {}
        for backend in BACKENDS:
            outcome = outcome_by_snapshot_backend[(snapshot_id, backend)]
            row = outcome.get("authoritative_row")
            if not isinstance(row, Mapping) or outcome["classification"] != "FINITE_SCIENTIFIC_VALUE":
                vectors[backend] = None
            else:
                vector = [
                    row.get("translation_x_m"),
                    row.get("translation_y_m"),
                    row.get("translation_z_m"),
                ]
                vectors[backend] = vector
        direction_rows.append(
            {
                "snapshot_id": snapshot_id,
                **translation_direction_cosine(vectors[BACKENDS[0]], vectors[BACKENDS[1]]),
            }
        )

    reassociation_scene_rows = []
    reassociation_scene_association_rows = []
    reassociation_centered_rows = []
    reassociation_permutation_rows = []
    accepted_source_rows = []
    systematic_station_rows = []
    systematic_scene_rows = []
    systematic_comparison_rows = []
    for backend in BACKENDS:
        turnover_by_scene: dict[str, list[float | None]] = {}
        translation_by_scene: dict[str, list[float | None]] = {}
        turnover_scene_medians: dict[str, float | None] = {}
        accepted_by_scene: dict[str, list[float | None]] = {}
        station_fraction_map: dict[str, dict[str, float | None]] = {
            scene: {} for scene in SCENE_ORDER
        }
        for scene in SCENE_ORDER:
            scene_outcomes = [
                outcome
                for station in STATION_ORDER
                for outcome in grouped[(backend, scene, station)]
            ]
            turnover_values = []
            translation_values = []
            accepted_values = []
            for outcome in scene_outcomes:
                row = outcome.get("authoritative_row")
                finite = outcome["classification"] == "FINITE_SCIENTIFIC_VALUE"
                turnover_values.append(row.get("correspondence_turnover") if finite else None)
                translation_values.append(row.get("translation_norm_m") if finite else None)
                accepted_values.append(row.get("accepted_source_turnover") if finite else None)
            turnover_by_scene[scene] = turnover_values
            translation_by_scene[scene] = translation_values
            accepted_by_scene[scene] = accepted_values
            scene_turnover = formal_scene_turnover(turnover_values)
            scene_turnover.update({"backend": backend, "scene_id": scene})
            reassociation_scene_rows.append(scene_turnover)
            turnover_scene_medians[scene] = scene_turnover["median"]
            accepted_summary = accepted_source_turnover_descriptive(accepted_values)
            accepted_summary.update({"backend": backend, "scene_id": scene})
            accepted_source_rows.append(accepted_summary)
            for station in STATION_ORDER:
                vectors = []
                for outcome in grouped[(backend, scene, station)]:
                    row = outcome.get("authoritative_row")
                    if outcome["classification"] != "FINITE_SCIENTIFIC_VALUE":
                        vectors.append(None)
                    else:
                        vectors.append(
                            [
                                row.get("translation_x_m"),
                                row.get("translation_y_m"),
                                row.get("translation_z_m"),
                            ]
                        )
                station_fraction = station_systematic_fraction(vectors)
                station_fraction.update(
                    {"backend": backend, "scene_id": scene, "station_id": station}
                )
                systematic_station_rows.append(station_fraction)
                station_fraction_map[scene][station] = station_fraction[
                    "systematic_fraction"
                ]
        association = scene_turnover_translation_spearman(
            turnover_scene_medians, scene_translation_medians[backend]
        )
        association["backend"] = backend
        reassociation_scene_association_rows.append(association)
        centered = centered_association(turnover_by_scene, translation_by_scene)
        centered_output = {
            key: value
            for key, value in centered.items()
            if key not in (
                "centered_rows",
                "centered_turnover_by_scene",
                "centered_translation_by_scene",
            )
        }
        centered_output["backend"] = backend
        reassociation_centered_rows.append(centered_output)
        permutation = centered_permutation_sensitivity(centered, backend=backend)
        permutation_output = {key: value for key, value in permutation.items() if key != "draws"}
        reassociation_permutation_rows.append(permutation_output)
        for scene in SCENE_ORDER:
            scene_fraction = scene_systematic_fraction(station_fraction_map[scene])
            scene_fraction.update({"backend": backend, "scene_id": scene})
            systematic_scene_rows.append(scene_fraction)
        systematic_comparison_rows.append(
            {
                "backend": backend,
                **systematic_weak_rich_descriptive(
                    {
                        row["scene_id"]: row["scene_systematic_fraction"]
                        for row in systematic_scene_rows
                        if row["backend"] == backend
                    }
                ),
            }
        )
        # Retain the rows/draws for the frozen future CSV outputs without treating
        # snapshots as independent scenes.
        centered_output["centered_rows"] = centered.get("centered_rows", [])
        permutation_output["draws"] = permutation.get("draws", [])

    backend_counts = []
    for backend in BACKENDS:
        backend_outcomes = [
            outcome
            for outcome in outcomes
            if identity_by_trial[outcome["trial_id"]]["backend"] == backend
        ]
        backend_counts.append(
            {
                "backend": backend,
                "planned_n": len(backend_outcomes),
                "finite_scientific_value_n": sum(
                    outcome["classification"] == "FINITE_SCIENTIFIC_VALUE"
                    for outcome in backend_outcomes
                ),
                "scientific_undefined_nonfinite_n": sum(
                    outcome["classification"] == "SCIENTIFIC_UNDEFINED_NONFINITE"
                    for outcome in backend_outcomes
                ),
                "unresolved_infrastructure_failure_n": sum(
                    outcome["classification"] == "UNRESOLVED_INFRASTRUCTURE_FAILURE"
                    for outcome in backend_outcomes
                ),
            }
        )

    has_undefined = any(
        row["scientific_undefined_nonfinite_n"] > 0
        or row["unresolved_infrastructure_failure_n"] > 0
        or row["endpoint_specific_undefined_n"] > 0
        for row in missingness
    )
    output = {
        "schema": "mid360_fmb1_zero_perturbation_locked_analysis_output_v1",
        "status": (
            "FIXTURE_ANALYSIS_COMPLETE"
            if fixture_only
            else (
                "FORMAL_ANALYSIS_COMPLETE_WITH_EXPLICIT_UNDEFINED_ENDPOINTS"
                if has_undefined
                else "FORMAL_ANALYSIS_COMPLETE"
            )
        ),
        "analysis_provenance": dict(analysis_provenance),
        "dataset_accounting": {
            "planned_trial_n": len(plan_rows),
            "authoritative_outcome_n": len(outcomes),
            "scene_n": 6,
            "station_n": 18,
            "snapshot_n": 180,
            "backend_n": 2,
            "scene_is_highest_independent_unit": True,
            "snapshots_are_independent_scenes": False,
        },
        "backend_status_counts": backend_counts,
        "translation_station_summaries": translation_station,
        "translation_scene_summaries": translation_scene,
        "primary_translation_inference": primary_inference,
        "rotation_station_summaries": rotation_station,
        "rotation_scene_summaries": rotation_scene,
        "secondary_rotation_inference": rotation_inference,
        "cross_backend_scene_spearman": scene_spearman,
        "cross_backend_station_spearman": station_spearman,
        "scene_ordering_agreement": ordering,
        "snapshot_direction_cosines": direction_rows,
        "reassociation_scene_summaries": reassociation_scene_rows,
        "reassociation_scene_association": reassociation_scene_association_rows,
        "reassociation_centered_association": reassociation_centered_rows,
        "reassociation_centered_permutation_sensitivity": reassociation_permutation_rows,
        "accepted_source_turnover_descriptive": accepted_source_rows,
        "systematic_station_values": systematic_station_rows,
        "systematic_scene_values": systematic_scene_rows,
        "systematic_weak_rich_descriptive": systematic_comparison_rows,
        "missingness_accounting": missingness,
        "physical_reference_limitation": {
            "physical_reference_semantics": PHYSICAL_REFERENCE_SEMANTICS,
            "independent_submillimeter_external_ground_truth_available": False,
            "controlled_static_nominal_identity_update_only": True,
            "absolute_physical_displacement_claim_forbidden": True,
        },
        "capture_radius_status": "PRESERVED_SUPPLEMENTARY_NOT_EXECUTED",
        "synthetic_transfer_status": "MODEL_TRANSFER_NOT_COMPATIBLE",
        "output_contract": {
            "save_reassociation_centered_permutation_draws": True,
            "reassociation_centered_permutation_draw_count_per_backend": 10000,
            "json_nan_infinity_forbidden": True,
        },
    }
    validate_output_schema(output)
    json.dumps(output, allow_nan=False)
    return output


def output_schema_path() -> Path:
    return Path(__file__).with_name("output_schema_v1.json")


def validate_output_schema(output: Mapping[str, Any]) -> None:
    schema = json.loads(output_schema_path().read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(output), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        raise LockedAnalysisError(
            f"analysis output schema failure at {list(first.path)}: {first.message}"
        )


def run_fixture_analysis() -> tuple[dict[str, Any], dict[str, Any]]:
    plan_rows, attempts = build_fixture_plan_and_attempts()
    output = analyze_attempts(
        plan_rows,
        attempts,
        fixture_only=True,
        analysis_provenance={
            "mode": "FIXTURE_ONLY",
            "artificial_fixture": True,
            "formal_result_files_read": 0,
            "real_scientific_aggregation_count": 0,
        },
    )
    qualification = {
        "schema": "mid360_fmb1_locked_analysis_fixture_qualification_v1",
        "status": "PASS",
        "pass": True,
        "FIXTURE_ANALYSIS_PASS": True,
        "fixture_trial_count": 360,
        "fixture_scene_count": 6,
        "fixture_station_count": 18,
        "fixture_snapshot_count": 180,
        "fixture_backend_count": 2,
        "primary_exact_allocation_count": 20,
        "primary_exact_p_denominator": 20,
        "centered_association_statistic": "SPEARMAN_RHO",
        "centered_permutation_draw_count": 10000,
        "centered_permutation_seed": 20260820,
        "centered_permutation_p_denominator": 10001,
        "systematic_formal_p_value_defined": False,
        "REAL_FORMAL_RESULT_FILES_READ": 0,
        "REAL_SCIENTIFIC_AGGREGATION_COUNT": 0,
        "REAL_WEAK_RICH_COMPARISON_COUNT": 0,
        "REAL_P_VALUE_COUNT": 0,
        "registration_backend_calls": 0,
    }
    return output, qualification


def _flatten(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                item[key] = json.dumps(value, sort_keys=True, allow_nan=False)
            else:
                item[key] = value
        flattened.append(item)
    return flattened


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    flattened = _flatten(rows)
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def write_analysis_outputs(output: Mapping[str, Any], output_dir: Path) -> None:
    """Persist the predeclared future output set; never called by this task."""

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LockedAnalysisError("formal analysis output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis_summary.md").write_text(
        "# FMB1 Zero-Perturbation locked analysis\n\n"
        f"Status: `{output['status']}`\n\n"
        "Physical semantics: `NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT`.\n",
        encoding="utf-8",
    )
    csv_bindings = {
        "translation_station_summaries.csv": output["translation_station_summaries"],
        "translation_scene_summaries.csv": output["translation_scene_summaries"],
        "rotation_station_summaries.csv": output["rotation_station_summaries"],
        "rotation_scene_summaries.csv": output["rotation_scene_summaries"],
        "snapshot_direction_cosines.csv": output["snapshot_direction_cosines"],
        "reassociation_scene_summaries.csv": output["reassociation_scene_summaries"],
        "systematic_station_values.csv": output["systematic_station_values"],
        "systematic_scene_values.csv": output["systematic_scene_values"],
        "analysis_missingness_inventory.csv": output["missingness_accounting"],
    }
    for filename, rows in csv_bindings.items():
        _write_csv(output_dir / filename, rows)
    _write_csv(
        output_dir / "translation_exact_permutations.csv",
        [
            {"backend": inference["backend"], **row}
            for inference in output["primary_translation_inference"]
            for row in inference["allocations"]
        ],
    )
    _write_csv(
        output_dir / "rotation_exact_permutations.csv",
        [
            {
                "backend": inference["backend"],
                "endpoint": inference["endpoint"],
                **row,
            }
            for inference in output["secondary_rotation_inference"]
            for row in inference["allocations"]
        ],
    )
    _write_csv(output_dir / "cross_backend_scene_pairs.csv", output["scene_ordering_agreement"]["pairs"])
    _write_csv(
        output_dir / "cross_backend_station_pairs.csv",
        output["cross_backend_station_spearman"]["pairs"],
    )
    _write_csv(
        output_dir / "reassociation_centered_rows.csv",
        [
            {"backend": item["backend"], **row}
            for item in output["reassociation_centered_association"]
            for row in item.get("centered_rows", [])
        ],
    )
    permutation_summaries = [
        {key: value for key, value in item.items() if key != "draws"}
        for item in output["reassociation_centered_permutation_sensitivity"]
    ]
    (output_dir / "reassociation_centered_permutation_summary.json").write_text(
        json.dumps(permutation_summaries, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "reassociation_centered_permutation_draws.csv",
        [
            row
            for item in output["reassociation_centered_permutation_sensitivity"]
            for row in item.get("draws", [])
        ],
    )
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    expected_without_sums = set(FUTURE_OUTPUT_FILES) - {"SHA256SUMS"}
    if actual != expected_without_sums:
        raise LockedAnalysisError(
            f"future output file set mismatch: missing={expected_without_sums-actual}, extra={actual-expected_without_sums}"
        )
    checksum_lines = []
    for name in sorted(actual):
        digest = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {name}\n")
    (output_dir / "SHA256SUMS").write_text("".join(checksum_lines), encoding="utf-8")
