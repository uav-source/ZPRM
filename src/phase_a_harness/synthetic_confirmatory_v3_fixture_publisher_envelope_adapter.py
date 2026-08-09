"""Fail-closed v3 fixture-envelope adapter for the frozen v2 publisher.

This module serves only the deterministic three-snapshot/six-trial
qualification fixture.  It does not adapt formal Synthetic Confirmatory v3
results, derive or access a Confirmatory seed, construct a snapshot, or invoke
a registration backend.

The frozen fixture publisher predates v3 and therefore requires its historical
``FIXTURE_RUN_SCHEMA`` and ``formal_v2_seed_reference_count`` field.  The
adapter proves that every seed-reference and formal Confirmatory execution
counter is the exact integer zero before constructing that historical eight-
field envelope.  Adapter provenance and scientific-equivalence evidence are
returned separately and must never be added to the publisher input.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from .synthetic_confirmatory_v2_artifact_verifier import (
    FIXTURE_INDEPENDENT_SCHEMA,
    FIXTURE_PRIMARY_SCHEMA,
    FIXTURE_RUN_SCHEMA,
)
from .synthetic_confirmatory_v2_independent_verifier import (
    compare_v2_fixture_primary_and_independent,
)


V3_FIXTURE_RUN_SCHEMA = "synthetic_confirmatory_v3_bootstrap_fixture_run_v1"

RAW_V3_FIXTURE_RUN_FIELDS = frozenset(
    {
        "backend_execution_count",
        "fixture_snapshot_count",
        "fixture_trial_count",
        "formal_confirmatory_science_evaluated",
        "formal_v3_seed_reference_count",
        "fresh_resume_scientific_equivalence",
        "resume_backend_execution_count",
        "schema_version",
    }
)

FROZEN_PUBLISHER_ENVELOPE_FIELDS = frozenset(
    {
        "backend_execution_count",
        "fixture_snapshot_count",
        "fixture_trial_count",
        "formal_confirmatory_science_evaluated",
        "formal_v2_seed_reference_count",
        "fresh_resume_scientific_equivalence",
        "resume_backend_execution_count",
        "schema_version",
    }
)

SEED_AUDIT_FIELDS = frozenset(
    {
        "formal_v1_seed_reference_count",
        "formal_v2_seed_reference_count",
        "formal_v3_seed_reference_count",
        "confirmatory_seed_access_count",
        "confirmatory_rng_instantiation_count",
        "confirmatory_snapshot_construction_count",
        "confirmatory_backend_execution_count",
    }
)

_SCIENTIFIC_RUN_FIELDS = (
    "backend_execution_count",
    "fixture_snapshot_count",
    "fixture_trial_count",
    "formal_confirmatory_science_evaluated",
    "fresh_resume_scientific_equivalence",
    "resume_backend_execution_count",
)


class FixturePublisherEnvelopeCompatibilityError(ValueError):
    """The qualification evidence cannot enter the frozen fixture publisher."""


def _require_plain_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixturePublisherEnvelopeCompatibilityError(
            f"{label} must be a mapping"
        )
    return value


def _require_exact_integer(
    value: Mapping[str, Any], name: str, expected: int, label: str
) -> None:
    if name not in value or type(value[name]) is not int or value[name] != expected:
        raise FixturePublisherEnvelopeCompatibilityError(
            f"{label}.{name} must be the exact integer {expected}"
        )


def _require_exact_boolean(
    value: Mapping[str, Any], name: str, expected: bool, label: str
) -> None:
    if (
        name not in value
        or type(value[name]) is not bool
        or value[name] is not expected
    ):
        raise FixturePublisherEnvelopeCompatibilityError(
            f"{label}.{name} must be the exact boolean {expected}"
        )


def _validate_seed_audit(seed_audit: Any) -> Mapping[str, Any]:
    audit = _require_plain_mapping(seed_audit, "seed_audit")
    actual = frozenset(audit)
    if actual != SEED_AUDIT_FIELDS:
        missing = sorted(SEED_AUDIT_FIELDS - actual)
        extra = sorted(actual - SEED_AUDIT_FIELDS)
        raise FixturePublisherEnvelopeCompatibilityError(
            f"seed_audit field set mismatch; missing={missing}, extra={extra}"
        )
    for name in sorted(SEED_AUDIT_FIELDS):
        _require_exact_integer(audit, name, 0, "seed_audit")
    return audit


def _validate_raw_v3_envelope(value: Any) -> Mapping[str, Any]:
    envelope = _require_plain_mapping(value, "v3_fixture_run_envelope")

    # Detect a non-zero legacy reference explicitly before the exact-field gate.
    # A zero-valued legacy field is still rejected as an extra raw-v3 field.
    for name in ("formal_v1_seed_reference_count", "formal_v2_seed_reference_count"):
        if name in envelope and (
            type(envelope[name]) is not int or envelope[name] != 0
        ):
            raise FixturePublisherEnvelopeCompatibilityError(
                f"v3_fixture_run_envelope.{name} must not be non-zero"
            )

    actual = frozenset(envelope)
    if actual != RAW_V3_FIXTURE_RUN_FIELDS:
        missing = sorted(RAW_V3_FIXTURE_RUN_FIELDS - actual)
        extra = sorted(actual - RAW_V3_FIXTURE_RUN_FIELDS)
        raise FixturePublisherEnvelopeCompatibilityError(
            "v3 fixture envelope field set mismatch; "
            f"missing={missing}, extra={extra}"
        )
    if envelope.get("schema_version") != V3_FIXTURE_RUN_SCHEMA:
        raise FixturePublisherEnvelopeCompatibilityError(
            "v3 fixture envelope schema is not the qualification schema"
        )
    _require_exact_integer(
        envelope, "backend_execution_count", 6, "v3_fixture_run_envelope"
    )
    _require_exact_integer(
        envelope, "fixture_snapshot_count", 3, "v3_fixture_run_envelope"
    )
    _require_exact_integer(
        envelope, "fixture_trial_count", 6, "v3_fixture_run_envelope"
    )
    _require_exact_integer(
        envelope, "formal_v3_seed_reference_count", 0, "v3_fixture_run_envelope"
    )
    _require_exact_integer(
        envelope, "resume_backend_execution_count", 0, "v3_fixture_run_envelope"
    )
    _require_exact_boolean(
        envelope,
        "formal_confirmatory_science_evaluated",
        False,
        "v3_fixture_run_envelope",
    )
    _require_exact_boolean(
        envelope,
        "fresh_resume_scientific_equivalence",
        True,
        "v3_fixture_run_envelope",
    )
    return envelope


def _validate_primary_and_independent(
    primary_result: Any, independent_result: Any
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    primary = _require_plain_mapping(primary_result, "primary_result")
    independent = _require_plain_mapping(independent_result, "independent_result")
    if primary.get("schema_version") != FIXTURE_PRIMARY_SCHEMA:
        raise FixturePublisherEnvelopeCompatibilityError(
            "primary_result is missing the frozen fixture schema"
        )
    if independent.get("schema_version") != FIXTURE_INDEPENDENT_SCHEMA:
        raise FixturePublisherEnvelopeCompatibilityError(
            "independent_result is missing the frozen fixture schema"
        )
    try:
        comparison = compare_v2_fixture_primary_and_independent(primary, independent)
    except (KeyError, TypeError, ValueError) as error:
        raise FixturePublisherEnvelopeCompatibilityError(
            "primary and independent fixture evidence is malformed"
        ) from error
    if comparison.get("exact_match_pass") is not True:
        raise FixturePublisherEnvelopeCompatibilityError(
            "primary and independent fixture evidence differs"
        )
    return primary, independent


def build_frozen_fixture_publisher_envelope(
    *,
    v3_fixture_run_envelope: Mapping[str, Any],
    primary_result: Mapping[str, Any],
    independent_result: Mapping[str, Any],
    seed_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact eight-field envelope accepted by the frozen publisher.

    All inputs are read-only.  The returned ``formal_v2_seed_reference_count``
    is historical fixture-publisher compatibility metadata; it neither declares
    nor accesses a v2 Confirmatory seed.
    """

    raw = _validate_raw_v3_envelope(v3_fixture_run_envelope)
    _validate_seed_audit(seed_audit)
    _validate_primary_and_independent(primary_result, independent_result)

    result: dict[str, Any] = {
        "schema_version": FIXTURE_RUN_SCHEMA,
        "backend_execution_count": raw["backend_execution_count"],
        "fixture_snapshot_count": raw["fixture_snapshot_count"],
        "fixture_trial_count": raw["fixture_trial_count"],
        "formal_confirmatory_science_evaluated": raw[
            "formal_confirmatory_science_evaluated"
        ],
        "formal_v2_seed_reference_count": 0,
        "fresh_resume_scientific_equivalence": raw[
            "fresh_resume_scientific_equivalence"
        ],
        "resume_backend_execution_count": raw["resume_backend_execution_count"],
    }
    if frozenset(result) != FROZEN_PUBLISHER_ENVELOPE_FIELDS:
        raise AssertionError("internal frozen publisher envelope field drift")
    return result


def _scientific_projection(
    run_envelope: Mapping[str, Any],
    primary_result: Mapping[str, Any],
    independent_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "fixture_run": {
            name: copy.deepcopy(run_envelope[name]) for name in _SCIENTIFIC_RUN_FIELDS
        },
        "primary_result": copy.deepcopy(dict(primary_result)),
        "independent_result": copy.deepcopy(dict(independent_result)),
    }


def _compare_leaves(left: Any, right: Any) -> tuple[int, float]:
    if type(left) is not type(right):
        return 1, 0.0
    if isinstance(left, dict):
        keys = set(left) | set(right)
        differences = 0
        maximum = 0.0
        for key in keys:
            if key not in left or key not in right:
                differences += 1
                continue
            child_differences, child_maximum = _compare_leaves(left[key], right[key])
            differences += child_differences
            maximum = max(maximum, child_maximum)
        return differences, maximum
    if isinstance(left, list):
        differences = abs(len(left) - len(right))
        maximum = 0.0
        for left_item, right_item in zip(left, right):
            child_differences, child_maximum = _compare_leaves(left_item, right_item)
            differences += child_differences
            maximum = max(maximum, child_maximum)
        return differences, maximum
    if isinstance(left, tuple):
        return _compare_leaves(list(left), list(right))
    if isinstance(left, bool) or left is None or isinstance(left, str):
        return (0, 0.0) if left == right else (1, 0.0)
    if isinstance(left, (int, float)):
        if not (math.isfinite(float(left)) and math.isfinite(float(right))):
            return (0, 0.0) if left == right else (1, 0.0)
        difference = abs(float(left) - float(right))
        return (int(left != right), difference)
    return (0, 0.0) if left == right else (1, 0.0)


def audit_fixture_envelope_scientific_equivalence(
    *,
    v3_fixture_run_envelope: Mapping[str, Any],
    frozen_publisher_envelope: Mapping[str, Any],
    primary_result: Mapping[str, Any],
    independent_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the complete fixture scientific projection before and after mapping."""

    raw = _validate_raw_v3_envelope(v3_fixture_run_envelope)
    primary, independent = _validate_primary_and_independent(
        primary_result, independent_result
    )
    published = _require_plain_mapping(
        frozen_publisher_envelope, "frozen_publisher_envelope"
    )
    if frozenset(published) != FROZEN_PUBLISHER_ENVELOPE_FIELDS:
        raise FixturePublisherEnvelopeCompatibilityError(
            "frozen publisher envelope field set is not exact"
        )
    if published.get("schema_version") != FIXTURE_RUN_SCHEMA:
        raise FixturePublisherEnvelopeCompatibilityError(
            "frozen publisher envelope schema is not exact"
        )
    _require_exact_integer(
        published, "formal_v2_seed_reference_count", 0, "frozen_publisher_envelope"
    )

    before = _scientific_projection(raw, primary, independent)
    after = _scientific_projection(published, primary, independent)
    leaf_difference_count, maximum_numerical_difference = _compare_leaves(before, after)
    return {
        "schema_version": (
            "synthetic_confirmatory_v3_fixture_envelope_scientific_equivalence_v1"
        ),
        "FIXTURE_ENVELOPE_SCIENTIFIC_LEAF_DIFFERENCE_COUNT": (
            leaf_difference_count
        ),
        "FIXTURE_ENVELOPE_SCIENTIFIC_MAX_NUMERICAL_DIFFERENCE": (
            maximum_numerical_difference
        ),
        "FIXTURE_ENVELOPE_SCIENTIFIC_EQUIVALENCE_PASS": bool(
            leaf_difference_count == 0 and maximum_numerical_difference == 0.0
        ),
        "allowed_envelope_differences": {
            "schema_version": {
                "before": V3_FIXTURE_RUN_SCHEMA,
                "after": FIXTURE_RUN_SCHEMA,
            },
            "seed_reference_field": {
                "before": "formal_v3_seed_reference_count",
                "after": "formal_v2_seed_reference_count",
                "value": 0,
            },
        },
    }


__all__ = [
    "FROZEN_PUBLISHER_ENVELOPE_FIELDS",
    "FixturePublisherEnvelopeCompatibilityError",
    "RAW_V3_FIXTURE_RUN_FIELDS",
    "SEED_AUDIT_FIELDS",
    "V3_FIXTURE_RUN_SCHEMA",
    "audit_fixture_envelope_scientific_equivalence",
    "build_frozen_fixture_publisher_envelope",
]
