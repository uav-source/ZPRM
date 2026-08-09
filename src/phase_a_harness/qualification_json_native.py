"""Fail-closed JSON-native normalization for qualification evidence only.

The frozen runtime writer intentionally accepts only exact Python JSON types.
Qualification reports may contain :class:`FormalRuntimeState` leaves returned
by the seed-agnostic lifecycle classifier, so this module provides the one
explicit conversion justified by the r2 final-aggregate type inventory.

No conversion is enabled for tuples, paths, NumPy values, arbitrary enums, or
custom objects.  Adding another conversion requires a separate inventory and
contract review; it must not be inferred from an object's convenience APIs.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from pathlib import PurePath
from typing import Any

from .formal_runtime_state_machine import FormalRuntimeState


FORMAL_RUNTIME_STATE_CONVERSION_RULE = (
    "EXACT_FORMAL_RUNTIME_STATE_TO_FROZEN_VALUE_V1"
)
FORMAL_RUNTIME_STATE_VALUES = frozenset(
    {
        "FORMAL_RUNTIME_ABSENT",
        "FORMAL_RUNTIME_BOOTSTRAP_ONLY",
        "FORMAL_RUNTIME_RESUMABLE",
        "FORMAL_RUNTIME_INVALID",
    }
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class StrictJsonNativeTreeError(ValueError):
    """A value cannot be represented under the qualification whitelist."""

    def __init__(self, violations: list[dict[str, Any]]) -> None:
        if not violations:
            raise ValueError("StrictJsonNativeTreeError requires violations")
        self.violations = tuple(dict(row) for row in violations)
        details = "; ".join(
            (
                f"{row['json_path']}: {row['python_type']} "
                f"({row['representative_value']}): {row['notes']}"
            )
            for row in violations
        )
        super().__init__(f"value is not a strict JSON-native tree: {details}")


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _child_path(path: str, key: str) -> str:
    if _IDENTIFIER.fullmatch(key):
        return f"{path}.{key}"
    escaped = (
        key.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'{path}["{escaped}"]'


def _safe_summary(value: Any) -> str:
    """Summarize without invoking an unknown object's ``str`` or ``repr``."""

    value_type = type(value)
    if value_type is FormalRuntimeState:
        return f"FormalRuntimeState.{value.name}"
    if value is None:
        return "null"
    if value_type is bool:
        return "true" if value else "false"
    if value_type is int:
        return format(value, "d")
    if value_type is float:
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
        return value.hex()
    if value_type is str:
        return f"<string length={len(value)}>"
    if value_type is list:
        return f"<list length={len(value)}>"
    if value_type is dict:
        return f"<dict length={len(value)}>"
    if value_type is tuple:
        return f"<tuple length={len(value)}>"
    if value_type is set:
        return f"<set length={len(value)}>"
    if value_type in {bytes, bytearray}:
        return f"<{value_type.__name__} length={len(value)}>"
    if isinstance(value, Enum):
        return f"<{_qualified_type(value)} enum member>"
    if isinstance(value, PurePath):
        return f"<{_qualified_type(value)} path value redacted>"
    return f"<{_qualified_type(value)} value redacted>"


def _violation(
    value: Any,
    *,
    path: str,
    conversion_required: bool,
    proposed_conversion: str,
    notes: str,
) -> dict[str, Any]:
    cls = type(value)
    return {
        "json_path": path,
        "python_type": _qualified_type(value),
        "module": cls.__module__,
        "class_name": cls.__qualname__,
        "representative_value": _safe_summary(value),
        "conversion_required": conversion_required,
        "proposed_conversion": proposed_conversion,
        "scientific_field": False,
        "notes": notes,
    }


def _unsupported_notes(value: Any) -> str:
    value_type = type(value)
    if isinstance(value, Enum):
        return "only the exact FormalRuntimeState class is convertible"
    if value_type is tuple:
        return "tuple conversion disabled: r2 aggregate inventory count was zero"
    if isinstance(value, PurePath):
        return "Path conversion disabled: r2 aggregate inventory count was zero"
    if value_type is set:
        return "set has no exact JSON identity"
    if value_type in {bytes, bytearray}:
        return "binary values have no exact JSON identity"
    if value_type.__module__ == "numpy" or value_type.__module__.startswith(
        "numpy."
    ):
        return "NumPy conversion disabled: r2 aggregate inventory count was zero"
    return "type is absent from the qualification conversion whitelist"


def scan_non_json_native_leaves(
    value: Any, path: str = "$"
) -> list[dict[str, Any]]:
    """Return a JSON-native inventory of every forbidden node.

    Container cycles are reported rather than followed.  Shared, acyclic
    containers are valid because their JSON representation is a value tree.
    """

    if type(path) is not str or not path:
        raise ValueError("path must be a non-empty exact string")
    violations: list[dict[str, Any]] = []
    active: dict[int, str] = {}

    def visit(item: Any, item_path: str) -> None:
        item_type = type(item)
        if item is None or item_type in {str, bool, int}:
            return
        if item_type is float:
            if not math.isfinite(item):
                violations.append(
                    _violation(
                        item,
                        path=item_path,
                        conversion_required=False,
                        proposed_conversion="REJECT_NONFINITE_NUMBER",
                        notes="non-finite JSON numbers are forbidden",
                    )
                )
            return
        if item_type is FormalRuntimeState:
            violations.append(
                _violation(
                    item,
                    path=item_path,
                    conversion_required=True,
                    proposed_conversion="value.value",
                    notes=FORMAL_RUNTIME_STATE_CONVERSION_RULE,
                )
            )
            return
        if item_type not in {list, dict}:
            violations.append(
                _violation(
                    item,
                    path=item_path,
                    conversion_required=False,
                    proposed_conversion="REJECT",
                    notes=_unsupported_notes(item),
                )
            )
            return

        identity = id(item)
        if identity in active:
            violations.append(
                _violation(
                    item,
                    path=item_path,
                    conversion_required=False,
                    proposed_conversion="REJECT_CYCLE",
                    notes=f"cyclic container reference to {active[identity]}",
                )
            )
            return
        active[identity] = item_path
        try:
            if item_type is list:
                for index, child in enumerate(item):
                    visit(child, f"{item_path}[{index}]")
                return
            for index, (key, child) in enumerate(item.items()):
                if type(key) is not str:
                    violations.append(
                        _violation(
                            key,
                            path=f"{item_path}.<non_string_key[{index}]>",
                            conversion_required=False,
                            proposed_conversion="REJECT_NON_STRING_KEY",
                            notes="JSON object keys must be exact strings",
                        )
                    )
                    visit(child, f"{item_path}.<value_for_non_string_key[{index}]>")
                else:
                    visit(child, _child_path(item_path, key))
        finally:
            del active[identity]

    visit(value, path)
    return violations


def assert_strict_json_native_tree(value: Any, path: str = "$") -> None:
    """Raise with a complete path/type inventory unless *value* is strict."""

    violations = scan_non_json_native_leaves(value, path)
    if violations:
        raise StrictJsonNativeTreeError(violations)


def to_strict_json_native(
    value: Any,
    path: str = "$",
    *,
    conversion_audit: list[dict[str, Any]] | None = None,
) -> Any:
    """Return a new strict JSON-native value under the frozen whitelist.

    The caller-supplied audit list is extended only after the complete
    normalization and native-tree assertion succeed.  Failed conversions
    therefore cannot leave partial audit evidence behind.
    """

    if type(path) is not str or not path:
        raise ValueError("path must be a non-empty exact string")
    if conversion_audit is not None and type(conversion_audit) is not list:
        raise TypeError("conversion_audit must be an exact list or None")

    events: list[dict[str, Any]] = []
    active: dict[int, str] = {}

    def normalize(item: Any, item_path: str) -> Any:
        item_type = type(item)
        if item is None or item_type is str:
            return item
        if item_type is bool:
            return item
        if item_type is int:
            return item
        if item_type is float:
            if not math.isfinite(item):
                raise StrictJsonNativeTreeError(
                    scan_non_json_native_leaves(item, item_path)
                )
            return item
        if item_type is FormalRuntimeState:
            normalized = item.value
            if type(normalized) is not str or normalized not in FORMAL_RUNTIME_STATE_VALUES:
                raise StrictJsonNativeTreeError(
                    [
                        _violation(
                            item,
                            path=item_path,
                            conversion_required=False,
                            proposed_conversion="REJECT_INVALID_STATE_VALUE",
                            notes="FormalRuntimeState value is outside the frozen state set",
                        )
                    ]
                )
            events.append(
                {
                    "json_path": item_path,
                    "source_type": _qualified_type(item),
                    "target_type": "builtins.str",
                    "source_summary": _safe_summary(item),
                    "target_value": normalized,
                    "conversion_rule": FORMAL_RUNTIME_STATE_CONVERSION_RULE,
                }
            )
            return normalized
        if item_type not in {list, dict}:
            raise StrictJsonNativeTreeError(
                scan_non_json_native_leaves(item, item_path)
            )

        identity = id(item)
        if identity in active:
            raise StrictJsonNativeTreeError(
                [
                    _violation(
                        item,
                        path=item_path,
                        conversion_required=False,
                        proposed_conversion="REJECT_CYCLE",
                        notes=f"cyclic container reference to {active[identity]}",
                    )
                ]
            )
        active[identity] = item_path
        try:
            if item_type is list:
                return [
                    normalize(child, f"{item_path}[{index}]")
                    for index, child in enumerate(item)
                ]
            normalized_dict: dict[str, Any] = {}
            for index, (key, child) in enumerate(item.items()):
                if type(key) is not str:
                    raise StrictJsonNativeTreeError(
                        [
                            _violation(
                                key,
                                path=f"{item_path}.<non_string_key[{index}]>",
                                conversion_required=False,
                                proposed_conversion="REJECT_NON_STRING_KEY",
                                notes="JSON object keys must be exact strings",
                            )
                        ]
                    )
                normalized_dict[key] = normalize(child, _child_path(item_path, key))
            return normalized_dict
        finally:
            del active[identity]

    normalized_value = normalize(value, path)
    assert_strict_json_native_tree(normalized_value, path)
    if conversion_audit is not None:
        conversion_audit.extend(events)
    return normalized_value


__all__ = [
    "FORMAL_RUNTIME_STATE_CONVERSION_RULE",
    "FORMAL_RUNTIME_STATE_VALUES",
    "StrictJsonNativeTreeError",
    "assert_strict_json_native_tree",
    "scan_non_json_native_leaves",
    "to_strict_json_native",
]
