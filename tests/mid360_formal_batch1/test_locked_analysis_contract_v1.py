from pathlib import Path

from experiments.mid360_formal_batch1.locked_analysis.contract_v1 import (
    BACKENDS, C2_COMMIT, C2_TAG_OBJECT, PROTECTED_BINDINGS, SCENE_ORDER,
    verify_contract_bindings,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def test_contract_bindings_are_exact_and_determinate():
    report = verify_contract_bindings(REPOSITORY)
    assert report["status"] == "PASS"
    assert report["binding_count"] == len(PROTECTED_BINDINGS)
    assert report["required_under_specified_count"] == 0
    assert report["unresolved_root_definition_count"] == 0


def test_frozen_identity_cardinality():
    assert len(SCENE_ORDER) == 6
    assert len(BACKENDS) == 2
    assert C2_COMMIT == "364c0ae76801c373d9ea6a4d9dd36e2fdf211e09"
    assert C2_TAG_OBJECT == "b0491f91723f726f22cb5137e0d044ef9f03d120"
