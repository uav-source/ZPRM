from experiments.mid360_formal_batch1.locked_analysis.descriptive_summary_v1 import summarize_station


def test_nonfinite_unresolved_and_endpoint_missing_are_separately_counted():
    rows = []
    for index in range(10):
        rows.append({"trial_id": str(index), "classification": "FINITE_SCIENTIFIC_VALUE",
                     "resolved_infrastructure_attempt_n": 0,
                     "authoritative_row": {"translation_norm_m": float(index),
                                             "solver_status": "CONVERGED"}})
    rows[0]["classification"] = "SCIENTIFIC_UNDEFINED_NONFINITE"
    rows[1] = {"trial_id": "1", "classification": "UNRESOLVED_INFRASTRUCTURE_FAILURE",
               "resolved_infrastructure_attempt_n": 2, "authoritative_row": None}
    rows[2]["authoritative_row"]["translation_norm_m"] = None
    summary = summarize_station(rows, "translation_norm_m")
    assert summary["scientific_undefined_nonfinite_n"] == 1
    assert summary["unresolved_infrastructure_failure_n"] == 1
    assert summary["endpoint_specific_undefined_n"] == 1
    assert summary["formal_statistics"] == {"median": None, "q25": None, "q75": None, "q95": None}
