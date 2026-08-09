from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from phase_a_harness import (
    synthetic_confirmatory_v3_fixture_publisher_envelope_adapter as envelope_adapter,
)
from phase_a_harness.synthetic_confirmatory_v2_artifact_verifier import (
    FIXTURE_RUN_SCHEMA,
    verify_synthetic_confirmatory_v2_fixture_artifact,
)
from phase_a_harness.synthetic_confirmatory_v2_publisher import (
    publish_synthetic_confirmatory_v2_fixture,
)


FROZEN_PUBLISHER_ENVELOPE_FIELDS = (
    envelope_adapter.FROZEN_PUBLISHER_ENVELOPE_FIELDS
)
SEED_AUDIT_FIELDS = envelope_adapter.SEED_AUDIT_FIELDS
V3_FIXTURE_RUN_SCHEMA = envelope_adapter.V3_FIXTURE_RUN_SCHEMA
FixturePublisherEnvelopeCompatibilityError = (
    envelope_adapter.FixturePublisherEnvelopeCompatibilityError
)
audit_fixture_envelope_scientific_equivalence = (
    envelope_adapter.audit_fixture_envelope_scientific_equivalence
)
build_frozen_fixture_publisher_envelope = (
    envelope_adapter.build_frozen_fixture_publisher_envelope
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_FIXTURE = (
    ROOT / "artifacts/synthetic_confirmatory_v2_prerun/fixture_publication"
)


def _read_fixture_json(name: str) -> dict[str, Any]:
    value = json.loads((FROZEN_FIXTURE / name).read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


@pytest.fixture
def primary() -> dict[str, Any]:
    return _read_fixture_json("primary_analysis.json")


@pytest.fixture
def independent() -> dict[str, Any]:
    return _read_fixture_json("independent_verification.json")


@pytest.fixture
def raw_v3_envelope() -> dict[str, Any]:
    return {
        "schema_version": V3_FIXTURE_RUN_SCHEMA,
        "backend_execution_count": 6,
        "fixture_snapshot_count": 3,
        "fixture_trial_count": 6,
        "formal_confirmatory_science_evaluated": False,
        "formal_v3_seed_reference_count": 0,
        "fresh_resume_scientific_equivalence": True,
        "resume_backend_execution_count": 0,
    }


@pytest.fixture
def seed_audit() -> dict[str, int]:
    return {name: 0 for name in SEED_AUDIT_FIELDS}


def _adapt(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> dict[str, Any]:
    return build_frozen_fixture_publisher_envelope(
        v3_fixture_run_envelope=raw_v3_envelope,
        primary_result=primary,
        independent_result=independent,
        seed_audit=seed_audit,
    )


def test_01_valid_v3_fixture_envelope_is_converted(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    converted = _adapt(raw_v3_envelope, primary, independent, seed_audit)
    assert converted["fixture_snapshot_count"] == 3
    assert converted["fixture_trial_count"] == 6
    assert converted["backend_execution_count"] == 6


def test_02_output_schema_reads_the_frozen_fixture_run_schema(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    converted = _adapt(raw_v3_envelope, primary, independent, seed_audit)
    assert converted["schema_version"] == FIXTURE_RUN_SCHEMA
    assert converted["schema_version"] == "synthetic_confirmatory_v2_fixture_run_v1"


def test_03_output_field_set_is_exactly_the_frozen_publisher_contract(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    converted = _adapt(raw_v3_envelope, primary, independent, seed_audit)
    assert frozenset(converted) == FROZEN_PUBLISHER_ENVELOPE_FIELDS
    assert len(converted) == 8


def test_04_output_extra_field_count_is_zero(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    converted = _adapt(raw_v3_envelope, primary, independent, seed_audit)
    assert set(converted) - set(FROZEN_PUBLISHER_ENVELOPE_FIELDS) == set()
    assert "formal_v3_seed_reference_count" not in converted


def test_05_legacy_v2_seed_reference_is_the_exact_integer_zero(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    converted = _adapt(raw_v3_envelope, primary, independent, seed_audit)
    assert type(converted["formal_v2_seed_reference_count"]) is int
    assert converted["formal_v2_seed_reference_count"] == 0


def test_06_adapter_does_not_modify_any_input(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    inputs = (raw_v3_envelope, primary, independent, seed_audit)
    before = copy.deepcopy(inputs)
    _adapt(raw_v3_envelope, primary, independent, seed_audit)
    assert inputs == before


def test_07_scientific_projection_has_zero_leaf_difference(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    converted = _adapt(raw_v3_envelope, primary, independent, seed_audit)
    audit = audit_fixture_envelope_scientific_equivalence(
        v3_fixture_run_envelope=raw_v3_envelope,
        frozen_publisher_envelope=converted,
        primary_result=primary,
        independent_result=independent,
    )
    assert audit["FIXTURE_ENVELOPE_SCIENTIFIC_LEAF_DIFFERENCE_COUNT"] == 0
    assert audit["FIXTURE_ENVELOPE_SCIENTIFIC_EQUIVALENCE_PASS"] is True


def test_08_scientific_projection_has_zero_maximum_numerical_difference(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    converted = _adapt(raw_v3_envelope, primary, independent, seed_audit)
    audit = audit_fixture_envelope_scientific_equivalence(
        v3_fixture_run_envelope=raw_v3_envelope,
        frozen_publisher_envelope=converted,
        primary_result=primary,
        independent_result=independent,
    )
    assert audit["FIXTURE_ENVELOPE_SCIENTIFIC_MAX_NUMERICAL_DIFFERENCE"] == 0.0


def test_09_nonzero_v3_seed_reference_is_rejected(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    raw_v3_envelope["formal_v3_seed_reference_count"] = 1
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        _adapt(raw_v3_envelope, primary, independent, seed_audit)


def test_10_nonzero_v2_seed_reference_is_rejected(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    seed_audit["formal_v2_seed_reference_count"] = 1
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        _adapt(raw_v3_envelope, primary, independent, seed_audit)


def test_11_nonzero_v1_seed_reference_is_rejected(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    seed_audit["formal_v1_seed_reference_count"] = 1
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        _adapt(raw_v3_envelope, primary, independent, seed_audit)


def test_12_nonzero_confirmatory_seed_access_is_rejected(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    seed_audit["confirmatory_seed_access_count"] = 1
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        _adapt(raw_v3_envelope, primary, independent, seed_audit)


def test_13_nonzero_confirmatory_rng_instantiation_is_rejected(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    seed_audit["confirmatory_rng_instantiation_count"] = 1
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        _adapt(raw_v3_envelope, primary, independent, seed_audit)


def test_14_missing_seed_provenance_audit_is_rejected(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    del seed_audit["formal_v3_seed_reference_count"]
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        _adapt(raw_v3_envelope, primary, independent, seed_audit)


def test_15_missing_primary_result_is_rejected(
    raw_v3_envelope: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        build_frozen_fixture_publisher_envelope(
            v3_fixture_run_envelope=raw_v3_envelope,
            primary_result={},
            independent_result=independent,
            seed_audit=seed_audit,
        )


def test_16_missing_independent_result_is_rejected(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        build_frozen_fixture_publisher_envelope(
            v3_fixture_run_envelope=raw_v3_envelope,
            primary_result=primary,
            independent_result={},
            seed_audit=seed_audit,
        )


def test_17_primary_independent_difference_is_rejected(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    independent["fixture_trial_count"] = 5
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        _adapt(raw_v3_envelope, primary, independent, seed_audit)


def test_18_frozen_publisher_accepts_the_adapted_envelope(
    tmp_path: Path,
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    converted = _adapt(raw_v3_envelope, primary, independent, seed_audit)
    publication = publish_synthetic_confirmatory_v2_fixture(
        primary=primary,
        independent=independent,
        run_manifest=converted,
        artifact_dir=tmp_path / "adapted-fixture",
    )
    assert publication["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is True
    assert publication["published_file_count"] == 17


def test_19_frozen_publisher_rejects_the_raw_v3_envelope(
    tmp_path: Path,
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="publication input contract failed"):
        publish_synthetic_confirmatory_v2_fixture(
            primary=primary,
            independent=independent,
            run_manifest=raw_v3_envelope,
            artifact_dir=tmp_path / "raw-v3-fixture",
        )


def test_20_frozen_artifact_verifier_accepts_the_seventeen_file_output(
    tmp_path: Path,
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    converted = _adapt(raw_v3_envelope, primary, independent, seed_audit)
    artifact = tmp_path / "adapted-fixture"
    publish_synthetic_confirmatory_v2_fixture(
        primary=primary,
        independent=independent,
        run_manifest=converted,
        artifact_dir=artifact,
    )
    verification = verify_synthetic_confirmatory_v2_fixture_artifact(
        artifact, write_report=False
    )
    assert verification["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is True
    assert verification["required_file_count"] == 17
    assert verification["actual_file_count"] == 17
    assert verification["missing_required_files"] == []
    assert verification["extra_files"] == []


def test_21_nonzero_confirmatory_snapshot_construction_is_rejected(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    seed_audit["confirmatory_snapshot_construction_count"] = 1
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        _adapt(raw_v3_envelope, primary, independent, seed_audit)


def test_22_nonzero_confirmatory_backend_execution_is_rejected(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    seed_audit["confirmatory_backend_execution_count"] = 1
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        _adapt(raw_v3_envelope, primary, independent, seed_audit)


def test_23_boolean_cannot_masquerade_as_an_exact_zero_seed_counter(
    raw_v3_envelope: dict[str, Any],
    primary: dict[str, Any],
    independent: dict[str, Any],
    seed_audit: dict[str, int],
) -> None:
    seed_audit["formal_v3_seed_reference_count"] = False
    with pytest.raises(FixturePublisherEnvelopeCompatibilityError):
        _adapt(raw_v3_envelope, primary, independent, seed_audit)
