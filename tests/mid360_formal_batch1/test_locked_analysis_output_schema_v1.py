import hashlib

import pytest

from experiments.mid360_formal_batch1.locked_analysis.locked_analysis_v1 import (
    LockedAnalysisError, run_fixture_analysis, validate_output_schema,
    write_analysis_outputs,
)


def test_fixture_output_schema_and_physical_limitations():
    output, qualification = run_fixture_analysis()
    validate_output_schema(output)
    assert qualification["FIXTURE_ANALYSIS_PASS"] is True
    assert output["physical_reference_limitation"]["independent_submillimeter_external_ground_truth_available"] is False
    assert output["synthetic_transfer_status"] == "MODEL_TRANSFER_NOT_COMPATIBLE"


def test_schema_rejects_physical_claim_tamper():
    output, _ = run_fixture_analysis()
    output["physical_reference_limitation"]["absolute_physical_displacement_claim_forbidden"] = False
    with pytest.raises(LockedAnalysisError):
        validate_output_schema(output)


def test_fixture_output_files_are_byte_deterministic(tmp_path):
    output, _ = run_fixture_analysis()
    left, right = tmp_path / "left", tmp_path / "right"
    write_analysis_outputs(output, left); write_analysis_outputs(output, right)
    left_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in left.iterdir()}
    right_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in right.iterdir()}
    assert left_hashes == right_hashes and len(left_hashes) == 19
