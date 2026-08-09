"""Explicit, immutable execution dependencies for formal and qualification runs.

Nothing in this module inspects environment variables, filesystem contents,
scene names, or paths to infer an execution mode.  A context is accepted only
when every supplied dependency exactly matches its declared contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


class ExecutionMode(str, Enum):
    FORMAL = "FORMAL"
    QUALIFICATION = "QUALIFICATION"


def _json_native(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_native(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(child) for child in value]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    return type(value).__name__


class ExecutionContextError(ValueError):
    """Structured fail-closed context or injected-validator rejection."""

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
        self.actual = _json_native(actual)
        self.expected = _json_native(expected)
        self.detail = detail

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "execution_context_error_v1",
            "failure_classification": self.classification,
            "field": self.field,
            "actual": self.actual,
            "expected": self.expected,
            "detail": self.detail,
        }


def _fail(
    classification: str,
    field: str,
    actual: Any,
    expected: Any,
    detail: str,
) -> None:
    raise ExecutionContextError(
        classification,
        field=field,
        actual=actual,
        expected=expected,
        detail=detail,
    )


def _require_mode(value: Any, field: str) -> ExecutionMode:
    if type(value) is not ExecutionMode:
        _fail(
            "EXECUTION_CONTEXT_MODE_MISMATCH",
            field,
            value,
            tuple(mode.value for mode in ExecutionMode),
            "execution mode must be supplied explicitly as ExecutionMode",
        )
    return value


def _require_name(value: Any, field: str, classification: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(classification, field, value, "non-empty canonical string", f"{field} is invalid")
    return value


def _require_names(value: Any, field: str, classification: str) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(type(item) is not str or not item or item.strip() != item for item in value)
        or len(value) != len(set(value))
    ):
        _fail(
            classification,
            field,
            value,
            "non-empty tuple of unique canonical strings",
            f"{field} is invalid",
        )
    return value


def _require_path(value: Any, field: str, classification: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute() or value == Path(value.anchor):
        _fail(
            classification,
            field,
            value,
            "non-root absolute pathlib.Path",
            f"{field} is not an explicit absolute path",
        )
    if any(part in {".", ".."} for part in value.parts):
        _fail(
            classification,
            field,
            value,
            "lexically canonical absolute path",
            f"{field} contains a relative component",
        )
    if value.resolve(strict=False) != value:
        _fail(
            classification,
            field,
            value,
            "canonical absolute path",
            f"{field} is not canonical",
        )
    current = Path(value.anchor)
    for component in value.parts[1:]:
        current /= component
        if current.is_symlink():
            _fail(
                classification,
                field,
                value,
                "path without symbolic-link components",
                f"{field} contains a symbolic-link component",
            )
    return value


@dataclass(frozen=True)
class SchemaBinding:
    mode: ExecutionMode
    schema_id: str
    fields: tuple[str, ...]
    allow_extra_fields: bool = False

    def __post_init__(self) -> None:
        _require_mode(self.mode, "schema.mode")
        _require_name(self.schema_id, "schema.schema_id", "EXECUTION_CONTEXT_SCHEMA_MISMATCH")
        _require_names(self.fields, "schema.fields", "EXECUTION_CONTEXT_SCHEMA_MISMATCH")
        if type(self.allow_extra_fields) is not bool:
            _fail(
                "EXECUTION_CONTEXT_SCHEMA_MISMATCH",
                "schema.allow_extra_fields",
                self.allow_extra_fields,
                "bool",
                "schema extra-field policy is not explicit",
            )

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "schema_id": self.schema_id,
            "fields": list(self.fields),
            "allow_extra_fields": self.allow_extra_fields,
        }


@dataclass(frozen=True)
class SeedPolicy:
    mode: ExecutionMode
    policy_id: str
    confirmatory_seed_allowed: bool

    def __post_init__(self) -> None:
        _require_mode(self.mode, "seed_policy.mode")
        _require_name(
            self.policy_id,
            "seed_policy.policy_id",
            "EXECUTION_CONTEXT_SEED_POLICY_MISMATCH",
        )
        if type(self.confirmatory_seed_allowed) is not bool:
            _fail(
                "EXECUTION_CONTEXT_SEED_POLICY_MISMATCH",
                "seed_policy.confirmatory_seed_allowed",
                self.confirmatory_seed_allowed,
                "bool",
                "confirmatory seed permission is not boolean",
            )
        if self.mode is ExecutionMode.QUALIFICATION and self.confirmatory_seed_allowed:
            _fail(
                "EXECUTION_CONTEXT_SEED_POLICY_MISMATCH",
                "seed_policy.confirmatory_seed_allowed",
                True,
                False,
                "qualification mode cannot authorize Confirmatory seeds",
            )

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "policy_id": self.policy_id,
            "confirmatory_seed_allowed": self.confirmatory_seed_allowed,
        }


@dataclass(frozen=True)
class IdPolicy:
    mode: ExecutionMode
    policy_id: str

    def __post_init__(self) -> None:
        _require_mode(self.mode, "id_policy.mode")
        _require_name(self.policy_id, "id_policy.policy_id", "EXECUTION_CONTEXT_ID_POLICY_MISMATCH")

    def report(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "policy_id": self.policy_id}


@dataclass(frozen=True)
class BackendPolicy:
    mode: ExecutionMode
    policy_id: str
    allowed_backends: tuple[str, ...]
    planned_backend_count: int

    def __post_init__(self) -> None:
        _require_mode(self.mode, "backend_policy.mode")
        _require_name(
            self.policy_id,
            "backend_policy.policy_id",
            "EXECUTION_CONTEXT_BACKEND_POLICY_MISMATCH",
        )
        _require_names(
            self.allowed_backends,
            "backend_policy.allowed_backends",
            "EXECUTION_CONTEXT_BACKEND_POLICY_MISMATCH",
        )
        if (
            type(self.planned_backend_count) is not int
            or self.planned_backend_count <= 0
            or self.planned_backend_count != len(self.allowed_backends)
        ):
            _fail(
                "EXECUTION_CONTEXT_BACKEND_POLICY_MISMATCH",
                "backend_policy.planned_backend_count",
                self.planned_backend_count,
                len(self.allowed_backends),
                "planned backend count differs from the explicit backend set",
            )

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "policy_id": self.policy_id,
            "allowed_backends": list(self.allowed_backends),
            "planned_backend_count": self.planned_backend_count,
        }


@dataclass(frozen=True)
class SnapshotReaderPolicy:
    """Complete non-payload policy consumed by the parameterized reader."""

    mode: ExecutionMode
    policy_id: str
    metadata_fields: tuple[str, ...]
    lock_entry_fields: tuple[str, ...]
    metadata_schema: str
    snapshot_schema_version: str
    lineage_schema_version: str
    seed_namespace: str | None
    snapshot_builder_contract_version: str
    lineage_required_conditions: tuple[str, ...]
    expected_rng_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        classification = "EXECUTION_CONTEXT_READER_POLICY_MISMATCH"
        _require_mode(self.mode, "snapshot_reader_policy.mode")
        for field in (
            "policy_id",
            "metadata_schema",
            "snapshot_schema_version",
            "lineage_schema_version",
            "snapshot_builder_contract_version",
        ):
            _require_name(getattr(self, field), f"snapshot_reader_policy.{field}", classification)
        if self.seed_namespace is None:
            if self.mode is not ExecutionMode.QUALIFICATION:
                _fail(
                    classification,
                    "snapshot_reader_policy.seed_namespace",
                    None,
                    "formal non-empty namespace",
                    "only qualification mode may explicitly have no seed namespace",
                )
        else:
            _require_name(
                self.seed_namespace,
                "snapshot_reader_policy.seed_namespace",
                classification,
            )
        _require_names(
            self.metadata_fields,
            "snapshot_reader_policy.metadata_fields",
            classification,
        )
        _require_names(
            self.lock_entry_fields,
            "snapshot_reader_policy.lock_entry_fields",
            classification,
        )
        _require_names(
            self.lineage_required_conditions,
            "snapshot_reader_policy.lineage_required_conditions",
            classification,
        )
        if type(self.expected_rng_counts) not in {dict, MappingProxyType}:
            _fail(
                classification,
                "snapshot_reader_policy.expected_rng_counts",
                self.expected_rng_counts,
                "mapping[str, nonnegative int]",
                "expected RNG counts are not an explicit mapping",
            )
        counts = dict(self.expected_rng_counts)
        if (
            not counts
            or any(type(key) is not str or not key for key in counts)
            or any(type(count) is not int or count < 0 for count in counts.values())
        ):
            _fail(
                classification,
                "snapshot_reader_policy.expected_rng_counts",
                counts,
                "non-empty mapping[str, nonnegative int]",
                "expected RNG counts are invalid",
            )
        object.__setattr__(self, "expected_rng_counts", MappingProxyType(counts))

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "policy_id": self.policy_id,
            "metadata_fields": list(self.metadata_fields),
            "lock_entry_fields": list(self.lock_entry_fields),
            "metadata_schema": self.metadata_schema,
            "snapshot_schema_version": self.snapshot_schema_version,
            "lineage_schema_version": self.lineage_schema_version,
            "seed_namespace": self.seed_namespace,
            "snapshot_builder_contract_version": self.snapshot_builder_contract_version,
            "lineage_required_conditions": list(self.lineage_required_conditions),
            "expected_rng_counts": dict(sorted(self.expected_rng_counts.items())),
        }


RowValidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def canonical_plan_rows_sha256(rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]) -> str:
    """Return the deterministic identity of an ordered, JSON-native plan."""

    payload = json.dumps(
        [dict(row) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_plan_sha256(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(
            "EXECUTION_CONTEXT_PLAN_MISMATCH",
            field,
            value,
            "lowercase SHA-256",
            "plan binding is not a canonical SHA-256",
        )
    return value


@dataclass(frozen=True)
class ExecutionContract:
    mode: ExecutionMode
    contract_id: str
    plan_id: str
    expected_snapshot_plan_sha256: str
    expected_trial_plan_sha256: str
    expected_snapshot_count: int
    expected_trial_count: int
    expected_cache_root: Path
    expected_runtime_root: Path
    snapshot_schema: SchemaBinding
    trial_schema: SchemaBinding
    allowed_scenes: tuple[str, ...]
    allowed_conditions: tuple[str, ...]
    allowed_seed_policy: SeedPolicy
    id_policy: IdPolicy
    backend_policy: BackendPolicy
    snapshot_reader_policy: SnapshotReaderPolicy
    snapshot_validator: RowValidator
    trial_validator: RowValidator
    result_route: str = "FORMAL_CONDITION_ROUTING_V1"

    def __post_init__(self) -> None:
        mode = _require_mode(self.mode, "contract.mode")
        _require_name(self.contract_id, "contract.contract_id", "EXECUTION_CONTEXT_CONTRACT_MISMATCH")
        _require_name(self.plan_id, "contract.plan_id", "EXECUTION_CONTEXT_PLAN_MISMATCH")
        _require_plan_sha256(
            self.expected_snapshot_plan_sha256,
            "contract.expected_snapshot_plan_sha256",
        )
        _require_plan_sha256(
            self.expected_trial_plan_sha256,
            "contract.expected_trial_plan_sha256",
        )
        for field in ("expected_snapshot_count", "expected_trial_count"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                _fail(
                    "EXECUTION_CONTEXT_PLAN_MISMATCH",
                    f"contract.{field}",
                    value,
                    "positive int",
                    "plan cardinality is not explicitly bound",
                )
        _require_name(
            self.result_route,
            "contract.result_route",
            "EXECUTION_CONTEXT_CONTRACT_MISMATCH",
        )
        runtime = _require_path(
            self.expected_runtime_root,
            "contract.expected_runtime_root",
            "EXECUTION_CONTEXT_RUNTIME_MISMATCH",
        )
        cache = _require_path(
            self.expected_cache_root,
            "contract.expected_cache_root",
            "EXECUTION_CONTEXT_CACHE_MISMATCH",
        )
        if runtime not in cache.parents:
            _fail(
                "EXECUTION_CONTEXT_CACHE_MISMATCH",
                "contract.expected_cache_root",
                cache,
                f"descendant of {runtime}",
                "snapshot cache is outside the declared runtime root",
            )
        if type(self.snapshot_schema) is not SchemaBinding or type(self.trial_schema) is not SchemaBinding:
            _fail(
                "EXECUTION_CONTEXT_SCHEMA_MISMATCH",
                "contract.schemas",
                type(self.snapshot_schema).__name__,
                "SchemaBinding pair",
                "contract schemas are not explicit bindings",
            )
        for label, dependency, classification in (
            ("snapshot_schema", self.snapshot_schema, "EXECUTION_CONTEXT_SCHEMA_MISMATCH"),
            ("trial_schema", self.trial_schema, "EXECUTION_CONTEXT_SCHEMA_MISMATCH"),
            ("allowed_seed_policy", self.allowed_seed_policy, "EXECUTION_CONTEXT_SEED_POLICY_MISMATCH"),
            ("id_policy", self.id_policy, "EXECUTION_CONTEXT_ID_POLICY_MISMATCH"),
            ("backend_policy", self.backend_policy, "EXECUTION_CONTEXT_BACKEND_POLICY_MISMATCH"),
            ("snapshot_reader_policy", self.snapshot_reader_policy, "EXECUTION_CONTEXT_READER_POLICY_MISMATCH"),
        ):
            if getattr(dependency, "mode", None) is not mode:
                _fail(
                    classification,
                    f"contract.{label}.mode",
                    getattr(dependency, "mode", None),
                    mode,
                    f"{label} belongs to another execution mode",
                )
        scenes = _require_names(
            self.allowed_scenes,
            "contract.allowed_scenes",
            "EXECUTION_CONTEXT_SCENE_POLICY_MISMATCH",
        )
        conditions = _require_names(
            self.allowed_conditions,
            "contract.allowed_conditions",
            "EXECUTION_CONTEXT_CONDITION_POLICY_MISMATCH",
        )
        if set(self.snapshot_reader_policy.lineage_required_conditions) - set(conditions):
            _fail(
                "EXECUTION_CONTEXT_READER_POLICY_MISMATCH",
                "contract.snapshot_reader_policy.lineage_required_conditions",
                self.snapshot_reader_policy.lineage_required_conditions,
                conditions,
                "reader lineage conditions escape the contract vocabulary",
            )
        if set(self.snapshot_reader_policy.expected_rng_counts) != set(conditions):
            _fail(
                "EXECUTION_CONTEXT_READER_POLICY_MISMATCH",
                "contract.snapshot_reader_policy.expected_rng_counts",
                tuple(self.snapshot_reader_policy.expected_rng_counts),
                conditions,
                "reader RNG policy does not cover exactly the allowed conditions",
            )
        if "planned_snapshot_id" not in self.snapshot_schema.fields or not {
            "planned_trial_id",
            "planned_snapshot_id",
            "backend",
        }.issubset(self.trial_schema.fields):
            _fail(
                "EXECUTION_CONTEXT_SCHEMA_MISMATCH",
                "contract.plan_schema_fields",
                {
                    "snapshot": self.snapshot_schema.fields,
                    "trial": self.trial_schema.fields,
                },
                "bridge identity fields",
                "plan schemas cannot support trial-to-snapshot binding",
            )
        if not callable(self.snapshot_validator) or not callable(self.trial_validator):
            _fail(
                "EXECUTION_CONTEXT_CONTRACT_MISMATCH",
                "contract.validators",
                {
                    "snapshot": callable(self.snapshot_validator),
                    "trial": callable(self.trial_validator),
                },
                {"snapshot": True, "trial": True},
                "contract validators must be explicit callables",
            )
        del scenes

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "contract_id": self.contract_id,
            "plan_id": self.plan_id,
            "expected_snapshot_plan_sha256": self.expected_snapshot_plan_sha256,
            "expected_trial_plan_sha256": self.expected_trial_plan_sha256,
            "expected_snapshot_count": self.expected_snapshot_count,
            "expected_trial_count": self.expected_trial_count,
            "expected_cache_root": str(self.expected_cache_root),
            "expected_runtime_root": str(self.expected_runtime_root),
            "snapshot_schema": self.snapshot_schema.report(),
            "trial_schema": self.trial_schema.report(),
            "allowed_scenes": list(self.allowed_scenes),
            "allowed_conditions": list(self.allowed_conditions),
            "allowed_seed_policy": self.allowed_seed_policy.report(),
            "id_policy": self.id_policy.report(),
            "backend_policy": self.backend_policy.report(),
            "snapshot_reader_policy": self.snapshot_reader_policy.report(),
            "snapshot_validator": _callable_name(self.snapshot_validator),
            "trial_validator": _callable_name(self.trial_validator),
            "result_route": self.result_route,
        }


def _callable_name(value: Callable[..., Any]) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


@dataclass(frozen=True)
class ExecutionContext:
    mode: ExecutionMode
    contract: ExecutionContract
    plan_id: str
    cache_root: Path
    runtime_root: Path
    snapshot_schema: SchemaBinding
    trial_schema: SchemaBinding
    allowed_scenes: tuple[str, ...]
    allowed_conditions: tuple[str, ...]
    allowed_seed_policy: SeedPolicy
    id_policy: IdPolicy
    backend_policy: BackendPolicy
    snapshot_reader_policy: SnapshotReaderPolicy

    def __post_init__(self) -> None:
        mode = _require_mode(self.mode, "mode")
        if type(self.contract) is not ExecutionContract:
            _fail(
                "EXECUTION_CONTEXT_CONTRACT_MISMATCH",
                "contract",
                type(self.contract).__name__,
                "ExecutionContract",
                "execution contract is not an explicit binding",
            )
        self._match("mode", mode, self.contract.mode, "EXECUTION_CONTEXT_MODE_MISMATCH")
        runtime = _require_path(self.runtime_root, "runtime_root", "EXECUTION_CONTEXT_RUNTIME_MISMATCH")
        cache = _require_path(self.cache_root, "cache_root", "EXECUTION_CONTEXT_CACHE_MISMATCH")
        self._match("runtime_root", runtime, self.contract.expected_runtime_root, "EXECUTION_CONTEXT_RUNTIME_MISMATCH")
        self._match("cache_root", cache, self.contract.expected_cache_root, "EXECUTION_CONTEXT_CACHE_MISMATCH")
        self._match("plan_id", self.plan_id, self.contract.plan_id, "EXECUTION_CONTEXT_PLAN_MISMATCH")
        self._match("snapshot_schema", self.snapshot_schema, self.contract.snapshot_schema, "EXECUTION_CONTEXT_SCHEMA_MISMATCH")
        self._match("trial_schema", self.trial_schema, self.contract.trial_schema, "EXECUTION_CONTEXT_SCHEMA_MISMATCH")
        self._match("allowed_scenes", self.allowed_scenes, self.contract.allowed_scenes, "EXECUTION_CONTEXT_SCENE_POLICY_MISMATCH")
        self._match("allowed_conditions", self.allowed_conditions, self.contract.allowed_conditions, "EXECUTION_CONTEXT_CONDITION_POLICY_MISMATCH")
        self._match("allowed_seed_policy", self.allowed_seed_policy, self.contract.allowed_seed_policy, "EXECUTION_CONTEXT_SEED_POLICY_MISMATCH")
        self._match("id_policy", self.id_policy, self.contract.id_policy, "EXECUTION_CONTEXT_ID_POLICY_MISMATCH")
        self._match("backend_policy", self.backend_policy, self.contract.backend_policy, "EXECUTION_CONTEXT_BACKEND_POLICY_MISMATCH")
        self._match("snapshot_reader_policy", self.snapshot_reader_policy, self.contract.snapshot_reader_policy, "EXECUTION_CONTEXT_READER_POLICY_MISMATCH")

    @staticmethod
    def _match(field: str, actual: Any, expected: Any, classification: str) -> None:
        if type(actual) is not type(expected) or actual != expected:
            _fail(classification, field, actual, expected, f"{field} differs from the injected contract")

    def _validate_row(
        self,
        row: Mapping[str, Any],
        *,
        schema: SchemaBinding,
        validator: RowValidator,
        classification: str,
        label: str,
    ) -> dict[str, Any]:
        row_fields = set(row) if isinstance(row, Mapping) else set()
        expected_fields = set(schema.fields)
        schema_match = (
            expected_fields.issubset(row_fields)
            if schema.allow_extra_fields
            else row_fields == expected_fields
        )
        if not isinstance(row, Mapping) or not schema_match:
            _fail(
                classification,
                f"{label}.fields",
                sorted(row) if isinstance(row, Mapping) else type(row).__name__,
                sorted(schema.fields),
                f"{label} does not match its injected schema",
            )
        try:
            validated = validator(row)
        except ExecutionContextError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ExecutionContextError(
                classification,
                field=label,
                actual=type(error).__name__,
                expected="injected validator acceptance",
                detail=f"{label} failed the injected validator",
            ) from error
        if not isinstance(validated, Mapping) or set(validated) != set(schema.fields):
            _fail(
                classification,
                f"{label}.validated_fields",
                sorted(validated) if isinstance(validated, Mapping) else type(validated).__name__,
                sorted(schema.fields),
                "injected validator returned an incompatible row",
            )
        return {field: validated[field] for field in schema.fields}

    def validate_snapshot_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return self._validate_row(
            row,
            schema=self.snapshot_schema,
            validator=self.contract.snapshot_validator,
            classification="SNAPSHOT_ROW_VALIDATION_FAILURE",
            label="snapshot_row",
        )

    def validate_trial_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return self._validate_row(
            row,
            schema=self.trial_schema,
            validator=self.contract.trial_validator,
            classification="TRIAL_ROW_VALIDATION_FAILURE",
            label="trial_row",
        )

    def validate_snapshot_plan_binding(
        self, rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]
    ) -> None:
        self._validate_plan_binding(
            rows,
            label="snapshot_plan",
            expected_count=self.contract.expected_snapshot_count,
            expected_sha256=self.contract.expected_snapshot_plan_sha256,
        )

    def validate_trial_plan_binding(
        self, rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]
    ) -> None:
        self._validate_plan_binding(
            rows,
            label="trial_plan",
            expected_count=self.contract.expected_trial_count,
            expected_sha256=self.contract.expected_trial_plan_sha256,
        )

    @staticmethod
    def _validate_plan_binding(
        rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
        *,
        label: str,
        expected_count: int,
        expected_sha256: str,
    ) -> None:
        actual = {
            "count": len(rows),
            "sha256": canonical_plan_rows_sha256(rows),
        }
        expected = {"count": expected_count, "sha256": expected_sha256}
        if actual != expected:
            _fail(
                "EXECUTION_CONTEXT_PLAN_MISMATCH",
                label,
                actual,
                expected,
                f"{label} differs from the injected contract",
            )

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "execution_context_v1",
            "EXECUTION_CONTEXT_PARAMETERIZATION_PASS": True,
            "mode": self.mode.value,
            "contract": self.contract.report(),
            "plan_id": self.plan_id,
            "cache_root": str(self.cache_root),
            "runtime_root": str(self.runtime_root),
            "snapshot_schema": self.snapshot_schema.report(),
            "trial_schema": self.trial_schema.report(),
            "allowed_scenes": list(self.allowed_scenes),
            "allowed_conditions": list(self.allowed_conditions),
            "allowed_seed_policy": self.allowed_seed_policy.report(),
            "id_policy": self.id_policy.report(),
            "backend_policy": self.backend_policy.report(),
            "snapshot_reader_policy": self.snapshot_reader_policy.report(),
        }


ExecutionContractBinding = ExecutionContract

__all__ = [
    "BackendPolicy",
    "canonical_plan_rows_sha256",
    "ExecutionContext",
    "ExecutionContextError",
    "ExecutionContract",
    "ExecutionContractBinding",
    "ExecutionMode",
    "IdPolicy",
    "SchemaBinding",
    "SeedPolicy",
    "SnapshotReaderPolicy",
]
