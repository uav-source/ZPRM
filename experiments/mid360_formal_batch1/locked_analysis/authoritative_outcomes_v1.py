"""C1 authoritative scientific outcome selection and retained accounting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


FINITE_SCIENTIFIC_VALUE = "FINITE_SCIENTIFIC_VALUE"
SCIENTIFIC_UNDEFINED_NONFINITE = "SCIENTIFIC_UNDEFINED_NONFINITE"
UNRESOLVED_INFRASTRUCTURE_FAILURE = "UNRESOLVED_INFRASTRUCTURE_FAILURE"


class OutcomeSelectionError(ValueError):
    """Raised when trial attempt lineage violates C1."""


def _attempt_index(row: Mapping[str, Any]) -> int:
    value = row.get("attempt", row.get("attempt_index"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OutcomeSelectionError("attempt index must be a positive integer")
    return value


def _scientific_outcome(row: Mapping[str, Any]) -> bool:
    return row.get("schema_valid") is True and row.get("infrastructure_status") == "OK"


def select_authoritative_outcomes(
    planned_trial_ids: Iterable[str],
    attempts: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select the lowest-attempt schema-valid infrastructure-PASS outcome."""

    planned = list(planned_trial_ids)
    if len(planned) != len(set(planned)):
        raise OutcomeSelectionError("planned trial IDs must be unique")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in attempts:
        trial_id = row.get("trial_id")
        if trial_id not in set(planned):
            raise OutcomeSelectionError(f"unexpected trial ID: {trial_id}")
        grouped[str(trial_id)].append(row)
    outcomes: list[dict[str, Any]] = []
    for trial_id in planned:
        rows = sorted(grouped.get(trial_id, []), key=_attempt_index)
        indices = [_attempt_index(row) for row in rows]
        if len(indices) != len(set(indices)):
            raise OutcomeSelectionError(f"duplicate attempt index for {trial_id}")
        candidates = [row for row in rows if _scientific_outcome(row)]
        if len(candidates) > 1:
            raise OutcomeSelectionError(f"scientific result retry forbidden: {trial_id}")
        if not candidates:
            outcomes.append(
                {
                    "trial_id": trial_id,
                    "classification": UNRESOLVED_INFRASTRUCTURE_FAILURE,
                    "authoritative_row": None,
                    "attempt_count": len(rows),
                    "resolved_infrastructure_attempt_n": len(rows),
                    "attempt_lineage": [dict(row) for row in rows],
                }
            )
            continue
        selected = candidates[0]
        selected_index = _attempt_index(selected)
        preceding_infrastructure = sum(
            _attempt_index(row) < selected_index and not _scientific_outcome(row)
            for row in rows
        )
        finite = selected.get("finite_result") is True
        classification = (
            FINITE_SCIENTIFIC_VALUE if finite else SCIENTIFIC_UNDEFINED_NONFINITE
        )
        outcomes.append(
            {
                "trial_id": trial_id,
                "classification": classification,
                "authoritative_row": dict(selected),
                "attempt_count": len(rows),
                "resolved_infrastructure_attempt_n": preceding_infrastructure,
                "attempt_lineage": [dict(row) for row in rows],
            }
        )
    return outcomes
