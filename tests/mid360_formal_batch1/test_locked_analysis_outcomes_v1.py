import pytest

from experiments.mid360_formal_batch1.locked_analysis.authoritative_outcomes_v1 import (
    FINITE_SCIENTIFIC_VALUE, SCIENTIFIC_UNDEFINED_NONFINITE,
    UNRESOLVED_INFRASTRUCTURE_FAILURE, OutcomeSelectionError,
    select_authoritative_outcomes,
)


def row(trial, attempt, infra, finite=True, schema=True):
    return {"trial_id": trial, "attempt": attempt, "schema_valid": schema,
            "infrastructure_status": infra, "finite_result": finite}


def test_lowest_scientific_outcome_after_infrastructure_attempt():
    outcome = select_authoritative_outcomes(["T"], [row("T", 1, "FAIL"), row("T", 2, "OK")])[0]
    assert outcome["classification"] == FINITE_SCIENTIFIC_VALUE
    assert outcome["authoritative_row"]["attempt"] == 2
    assert outcome["resolved_infrastructure_attempt_n"] == 1


def test_nonfinite_is_retained_and_unresolved_is_retained():
    outcomes = select_authoritative_outcomes(
        ["A", "B"], [row("A", 1, "OK", finite=False), row("B", 1, "CRASH")]
    )
    assert outcomes[0]["classification"] == SCIENTIFIC_UNDEFINED_NONFINITE
    assert outcomes[1]["classification"] == UNRESOLVED_INFRASTRUCTURE_FAILURE


def test_scientific_retry_and_duplicate_attempt_fail_closed():
    with pytest.raises(OutcomeSelectionError, match="scientific result retry"):
        select_authoritative_outcomes(["T"], [row("T", 1, "OK"), row("T", 2, "OK")])
    with pytest.raises(OutcomeSelectionError, match="duplicate attempt"):
        select_authoritative_outcomes(["T"], [row("T", 1, "FAIL"), row("T", 1, "CRASH")])
