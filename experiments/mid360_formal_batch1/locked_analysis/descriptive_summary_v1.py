"""Complete-coverage station and scene summaries with explicit missingness."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .authoritative_outcomes_v1 import (
    FINITE_SCIENTIFIC_VALUE,
    SCIENTIFIC_UNDEFINED_NONFINITE,
    UNRESOLVED_INFRASTRUCTURE_FAILURE,
)


class SummaryError(ValueError):
    """Raised when fixture summary grain or identifiers are invalid."""


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and bool(np.isfinite(value))
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25, method="linear")),
        "q75": float(np.quantile(array, 0.75, method="linear")),
        "q95": float(np.quantile(array, 0.95, method="linear")),
    }


def summarize_outcomes(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    endpoint: str,
    planned_n: int,
    defined_status: str,
    undefined_status: str,
) -> dict[str, Any]:
    rows = list(outcomes)
    if len(rows) != planned_n:
        raise SummaryError(f"expected {planned_n} planned outcomes, got {len(rows)}")
    finite_values: list[float] = []
    authoritative_n = 0
    scientific_undefined_n = 0
    unresolved_n = 0
    resolved_infrastructure_n = 0
    solver_nonconverged_finite_n = 0
    endpoint_specific_undefined_n = 0
    undefined_trial_ids: list[str] = []
    for outcome in rows:
        classification = outcome.get("classification")
        resolved_infrastructure_n += int(
            outcome.get("resolved_infrastructure_attempt_n", 0)
        )
        if classification == UNRESOLVED_INFRASTRUCTURE_FAILURE:
            unresolved_n += 1
            undefined_trial_ids.append(str(outcome.get("trial_id")))
            continue
        authoritative_n += 1
        row = outcome.get("authoritative_row")
        if not isinstance(row, Mapping):
            raise SummaryError("authoritative scientific outcome must bind a row")
        if classification == SCIENTIFIC_UNDEFINED_NONFINITE:
            scientific_undefined_n += 1
            undefined_trial_ids.append(str(outcome.get("trial_id")))
            continue
        if classification != FINITE_SCIENTIFIC_VALUE:
            raise SummaryError(f"unknown classification: {classification}")
        value = row.get(endpoint)
        if not _finite_number(value):
            endpoint_specific_undefined_n += 1
            undefined_trial_ids.append(str(outcome.get("trial_id")))
            continue
        finite_values.append(float(value))
        if row.get("solver_status") not in ("CONVERGED", "SUCCESS"):
            solver_nonconverged_finite_n += 1
    complete = (
        len(finite_values) == planned_n
        and scientific_undefined_n == 0
        and unresolved_n == 0
        and endpoint_specific_undefined_n == 0
    )
    available = _quantiles(finite_values) if finite_values else {
        "median": None,
        "q25": None,
        "q75": None,
        "q95": None,
    }
    formal = _quantiles(finite_values) if complete else {
        "median": None,
        "q25": None,
        "q75": None,
        "q95": None,
    }
    return {
        "endpoint": endpoint,
        "planned_n": planned_n,
        "authoritative_scientific_outcome_n": authoritative_n,
        "finite_endpoint_n": len(finite_values),
        "scientific_undefined_nonfinite_n": scientific_undefined_n,
        "unresolved_infrastructure_failure_n": unresolved_n,
        "resolved_infrastructure_attempt_n": resolved_infrastructure_n,
        "solver_nonconverged_finite_n": solver_nonconverged_finite_n,
        "endpoint_specific_undefined_n": endpoint_specific_undefined_n,
        "formal_summary_status": defined_status if complete else undefined_status,
        "formal_statistics": formal,
        "available_case": {
            "label": "AVAILABLE_CASE_DESCRIPTIVE_NOT_PRIMARY_NOT_INFERENTIAL",
            "finite_n": len(finite_values),
            "statistics": available,
            "may_enter_inference": False,
        },
        "undefined_trial_ids": undefined_trial_ids,
    }


def summarize_station(
    outcomes: Iterable[Mapping[str, Any]], endpoint: str
) -> dict[str, Any]:
    return summarize_outcomes(
        outcomes,
        endpoint=endpoint,
        planned_n=10,
        defined_status="FORMAL_STATION_SUMMARY_DEFINED_COMPLETE_10_OF_10",
        undefined_status=(
            "FORMAL_STATION_SUMMARY_UNDEFINED_INCOMPLETE_FINITE_COVERAGE"
        ),
    )


def summarize_scene(
    outcomes_by_station: Mapping[str, Iterable[Mapping[str, Any]]],
    endpoint: str,
) -> dict[str, Any]:
    if set(outcomes_by_station) != {"S01", "S02", "S03"}:
        raise SummaryError("scene summary requires exactly S01, S02, and S03")
    station_summaries = {
        station: summarize_station(outcomes_by_station[station], endpoint)
        for station in ("S01", "S02", "S03")
    }
    combined = [
        outcome
        for station in ("S01", "S02", "S03")
        for outcome in outcomes_by_station[station]
    ]
    summary = summarize_outcomes(
        combined,
        endpoint=endpoint,
        planned_n=30,
        defined_status="FORMAL_SCENE_SUMMARY_DEFINED_COMPLETE_30_OF_30",
        undefined_status="FORMAL_SCENE_SUMMARY_UNDEFINED_INCOMPLETE_FINITE_COVERAGE",
    )
    defined_stations = sum(
        value["formal_summary_status"]
        == "FORMAL_STATION_SUMMARY_DEFINED_COMPLETE_10_OF_10"
        for value in station_summaries.values()
    )
    if defined_stations != 3:
        summary["formal_summary_status"] = (
            "FORMAL_SCENE_SUMMARY_UNDEFINED_INCOMPLETE_FINITE_COVERAGE"
        )
        summary["formal_statistics"] = {
            "median": None,
            "q25": None,
            "q75": None,
            "q95": None,
        }
    summary["defined_station_summary_n"] = defined_stations
    summary["station_medians"] = {
        station: station_summary["formal_statistics"]["median"]
        for station, station_summary in station_summaries.items()
    }
    summary["station_summaries"] = station_summaries
    summary["scene_statistics_are_direct_30_snapshot_statistics"] = True
    return summary
