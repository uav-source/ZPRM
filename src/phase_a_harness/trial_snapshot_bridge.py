"""Immutable, fail-closed trial-to-snapshot plan binding.

This module never reads snapshot payloads and never dispatches a backend.  It
only binds exact typed trial identities to full snapshot rows authenticated by
an explicitly supplied :class:`ExecutionContext`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .execution_context import ExecutionContext, ExecutionContextError


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(child) for child in value]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    return type(value).__name__


class TrialSnapshotBridgeError(ValueError):
    def __init__(
        self,
        classification: str,
        *,
        field: str,
        actual: Any,
        expected: Any,
        detail: str,
    ) -> None:
        super().__init__(f"{classification}: {detail}")
        self.classification = classification
        self.category = classification
        self.field = field
        self.actual = actual
        self.expected = expected
        self.detail = detail

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "trial_snapshot_bridge_error_v1",
            "failure_classification": self.classification,
            "field": self.field,
            "actual_present": self.actual is not _MISSING,
            "expected_present": self.expected is not _MISSING,
            "actual": None if self.actual is _MISSING else _json_native(self.actual),
            "expected": (
                None if self.expected is _MISSING else _json_native(self.expected)
            ),
            "detail": self.detail,
        }


class _Missing:
    pass


_MISSING = _Missing()


@dataclass(frozen=True)
class SharedIdentityFieldContract:
    snapshot_fields: tuple[str, ...]
    trial_fields: tuple[str, ...]
    shared_identity_fields: tuple[str, ...]
    snapshot_only_fields: tuple[str, ...]
    trial_only_fields: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "shared_identity_field_contract_v1",
            "snapshot_fields": list(self.snapshot_fields),
            "trial_fields": list(self.trial_fields),
            "shared_identity_fields": list(self.shared_identity_fields),
            "snapshot_only_fields": list(self.snapshot_only_fields),
            "trial_only_fields": list(self.trial_only_fields),
        }


@dataclass(frozen=True)
class CanonicalSnapshotIndex:
    rows: Mapping[str, Mapping[str, Any]]
    field_contract: SharedIdentityFieldContract
    execution_context: ExecutionContext
    source_row_count: int
    index_key: str = "planned_snapshot_id"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, snapshot_id: str) -> Mapping[str, Any]:
        return self.rows[snapshot_id]


@dataclass(frozen=True)
class TrialSnapshotBindings:
    rows: Mapping[str, Mapping[str, Any]]
    audit: Mapping[str, Any]

    def __getitem__(self, trial_id: str) -> Mapping[str, Any]:
        return self.rows[trial_id]


def derive_shared_identity_contract(
    execution_context: ExecutionContext,
) -> SharedIdentityFieldContract:
    snapshot = tuple(execution_context.snapshot_schema.fields)
    trial = tuple(execution_context.trial_schema.fields)
    trial_set = set(trial)
    snapshot_set = set(snapshot)
    shared = tuple(name for name in snapshot if name in trial_set)
    if "planned_snapshot_id" not in shared:
        raise TrialSnapshotBridgeError(
            "EXECUTION_CONTEXT_SCHEMA_MISMATCH",
            field="shared_identity_fields",
            actual=list(shared),
            expected="planned_snapshot_id in schema intersection",
            detail="plan schemas cannot form a snapshot lookup key",
        )
    return SharedIdentityFieldContract(
        snapshot_fields=snapshot,
        trial_fields=trial,
        shared_identity_fields=shared,
        snapshot_only_fields=tuple(name for name in snapshot if name not in trial_set),
        trial_only_fields=tuple(name for name in trial if name not in snapshot_set),
    )


def _require_exact_fields(
    row: Mapping[str, Any], fields: Sequence[str], *, label: str, classification: str
) -> None:
    if not isinstance(row, Mapping):
        raise TrialSnapshotBridgeError(
            classification,
            field=f"{label}.fields",
            actual=type(row).__name__,
            expected=sorted(fields),
            detail=f"{label} is not a mapping",
        )
    actual = set(row)
    expected = set(fields)
    if actual != expected:
        raise TrialSnapshotBridgeError(
            classification,
            field=f"{label}.fields",
            actual=sorted(actual),
            expected=sorted(expected),
            detail=f"{label} differs from the exact bridge schema",
        )


def build_canonical_snapshot_index(
    snapshot_rows: Sequence[Mapping[str, Any]],
    *,
    execution_context: ExecutionContext,
) -> CanonicalSnapshotIndex:
    fields = derive_shared_identity_contract(execution_context)
    indexed: dict[str, Mapping[str, Any]] = {}
    validated_rows: list[Mapping[str, Any]] = []
    for position, row in enumerate(snapshot_rows):
        _require_exact_fields(
            row,
            fields.snapshot_fields,
            label=f"snapshot_row[{position}]",
            classification="SNAPSHOT_ROW_VALIDATION_FAILURE",
        )
        try:
            validated = execution_context.validate_snapshot_row(row)
        except ExecutionContextError:
            raise
        snapshot_id = validated["planned_snapshot_id"]
        if snapshot_id in indexed:
            raise TrialSnapshotBridgeError(
                "DUPLICATE_SNAPSHOT_PLAN_ID",
                field="planned_snapshot_id",
                actual=snapshot_id,
                expected="unique planned_snapshot_id",
                detail="canonical snapshot index contains a duplicate key",
            )
        indexed[snapshot_id] = MappingProxyType(dict(validated))
        validated_rows.append(validated)
    execution_context.validate_snapshot_plan_binding(validated_rows)
    return CanonicalSnapshotIndex(
        rows=MappingProxyType(dict(indexed)),
        field_contract=fields,
        execution_context=execution_context,
        source_row_count=len(snapshot_rows),
    )


_FIELD_FAILURES = {
    "scene_variant": "TRIAL_SNAPSHOT_SCENE_MISMATCH",
    "condition": "TRIAL_SNAPSHOT_CONDITION_MISMATCH",
    "geometry_seed": "TRIAL_SNAPSHOT_GEOMETRY_SEED_MISMATCH",
    "measurement_seed": "TRIAL_SNAPSHOT_MEASUREMENT_SEED_MISMATCH",
    "repeat_index": "TRIAL_SNAPSHOT_REPEAT_MISMATCH",
}


def bind_trial_to_snapshot(
    index: CanonicalSnapshotIndex,
    trial_row: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    fields = index.field_contract
    if isinstance(trial_row, Mapping):
        override = sorted(set(trial_row) & set(fields.snapshot_only_fields))
        if override:
            name = override[0]
            raise TrialSnapshotBridgeError(
                "TRIAL_ROW_VALIDATION_FAILURE",
                field=name,
                actual=trial_row[name],
                expected=_MISSING,
                detail="trial row attempted to provide a snapshot-only field",
            )
    _require_exact_fields(
        trial_row,
        fields.trial_fields,
        label="trial_row",
        classification="TRIAL_ROW_VALIDATION_FAILURE",
    )
    validated = index.execution_context.validate_trial_row(trial_row)
    snapshot_id = validated["planned_snapshot_id"]
    snapshot = index.rows.get(snapshot_id)
    if snapshot is None:
        raise TrialSnapshotBridgeError(
            "SNAPSHOT_PLAN_ID_NOT_FOUND",
            field="planned_snapshot_id",
            actual=snapshot_id,
            expected="one canonical snapshot row",
            detail="trial references no canonical snapshot row",
        )
    for name in fields.shared_identity_fields:
        actual = validated[name]
        expected = snapshot[name]
        if type(actual) is not type(expected) or actual != expected:
            raise TrialSnapshotBridgeError(
                _FIELD_FAILURES.get(
                    name, "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH"
                ),
                field=name,
                actual={"type": type(actual).__name__, "value": actual},
                expected={"type": type(expected).__name__, "value": expected},
                detail="trial and snapshot shared identity differ",
            )
    return validated, snapshot


def build_trial_snapshot_bindings(
    index: CanonicalSnapshotIndex,
    trial_rows: Sequence[Mapping[str, Any]],
    *,
    require_complete_pairing: bool = True,
) -> TrialSnapshotBindings:
    bindings: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    validated_trials: list[Mapping[str, Any]] = []
    for row in trial_rows:
        trial, snapshot = bind_trial_to_snapshot(index, row)
        validated_trials.append(trial)
        trial_id = trial["planned_trial_id"]
        if trial_id in bindings:
            raise TrialSnapshotBridgeError(
                "TRIAL_ROW_VALIDATION_FAILURE",
                field="planned_trial_id",
                actual=trial_id,
                expected="unique planned_trial_id",
                detail="trial plan contains a duplicate ID",
            )
        bindings[trial_id] = snapshot
        groups[snapshot["planned_snapshot_id"]].append(trial["backend"])
    expected_backends = set(index.execution_context.backend_policy.allowed_backends)
    pairing = sum(
        len(groups.get(snapshot_id, ())) != len(expected_backends)
        or set(groups.get(snapshot_id, ())) != expected_backends
        for snapshot_id in index.rows
    )
    extra_groups = len(set(groups) - set(index.rows))
    pairing += extra_groups
    if require_complete_pairing and pairing:
        raise TrialSnapshotBridgeError(
            "TRIAL_SNAPSHOT_SHARED_IDENTITY_MISMATCH",
            field="backend_pairing",
            actual={name: list(values) for name, values in sorted(groups.items())},
            expected=sorted(expected_backends),
            detail="snapshot backend pairing is incomplete or inconsistent",
        )
    index.execution_context.validate_trial_plan_binding(validated_trials)
    backend_counts = Counter(row["backend"] for row in trial_rows)
    audit = {
        "schema_version": "trial_snapshot_binding_audit_v1",
        "canonical_snapshot_count": len(index),
        "canonical_snapshot_source_row_count": index.source_row_count,
        "planned_trial_count": len(trial_rows),
        "unique_trial_count": len(bindings),
        "backend_trial_counts": dict(sorted(backend_counts.items())),
        "native_trial_count": sum(
            count
            for backend, count in backend_counts.items()
            if backend not in expected_backends
        ),
        "pairing_violation_count": pairing,
        "CANONICAL_SNAPSHOT_INDEX_PASS": bool(
            len(index) == index.source_row_count
        ),
        "TRIAL_TO_SNAPSHOT_BINDING_PASS": bool(
            len(bindings) == len(trial_rows) and pairing == 0
        ),
        "SHARED_IDENTITY_VALIDATION_PASS": True,
    }
    return TrialSnapshotBindings(
        rows=MappingProxyType(dict(bindings)),
        audit=MappingProxyType(audit),
    )


__all__ = [
    "CanonicalSnapshotIndex",
    "SharedIdentityFieldContract",
    "TrialSnapshotBindings",
    "TrialSnapshotBridgeError",
    "bind_trial_to_snapshot",
    "build_canonical_snapshot_index",
    "build_trial_snapshot_bindings",
    "derive_shared_identity_contract",
]
