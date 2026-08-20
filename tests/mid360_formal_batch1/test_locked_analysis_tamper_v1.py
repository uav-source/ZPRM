import copy

import pytest

from experiments.mid360_formal_batch1.locked_analysis.exact_scene_permutation_v1 import (
    ExactPermutationError, exact_weak_greater_than_rich,
)
from experiments.mid360_formal_batch1.locked_analysis.locked_analysis_v1 import (
    LockedAnalysisError, run_fixture_analysis, validate_output_schema,
)


@pytest.mark.parametrize("field,value", [
    ("capture_radius_status", "EXECUTED"),
    ("synthetic_transfer_status", "COMPATIBLE"),
])
def test_top_level_contract_tamper_fails(field, value):
    output, _ = run_fixture_analysis(); output[field] = value
    with pytest.raises(LockedAnalysisError): validate_output_schema(output)


def test_additional_output_field_and_nan_fail():
    output, _ = run_fixture_analysis(); extra = copy.deepcopy(output); extra["T_est"] = []
    with pytest.raises(LockedAnalysisError): validate_output_schema(extra)
    output["primary_translation_inference"][0]["p_value"] = float("nan")
    with pytest.raises(ValueError):
        import json; json.dumps(output, allow_nan=False)


def test_seven_scene_inference_tamper_fails():
    with pytest.raises(ExactPermutationError):
        exact_weak_greater_than_rich({f"scene{i}": float(i) for i in range(7)})
