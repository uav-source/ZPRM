from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from phase_a_harness.formal_runtime_state_machine import FormalRuntimeState
from phase_a_harness.qualification_json_native import (
    FORMAL_RUNTIME_STATE_CONVERSION_RULE,
    StrictJsonNativeTreeError,
    assert_strict_json_native_tree,
    scan_non_json_native_leaves,
    to_strict_json_native,
)
from phase_a_harness.runtime_lifecycle_io import (
    atomic_create_canonical_json,
    read_canonical_json,
)


def _r2_aggregate_state_inventory() -> dict[str, object]:
    """The eleven enum leaves identified in the complete r2 aggregate."""

    return {
        "states": {
            "absent": {"state": FormalRuntimeState.ABSENT},
            "bootstrap_only": {"state": FormalRuntimeState.BOOTSTRAP_ONLY},
            "resumable": {"state": FormalRuntimeState.RESUMABLE},
        },
        "bootstrap": {
            "state_before": FormalRuntimeState.ABSENT,
            "state_after": FormalRuntimeState.BOOTSTRAP_ONLY,
        },
        "bootstrap_resume": {
            "state_before": FormalRuntimeState.RESUMABLE,
            "state_after": FormalRuntimeState.RESUMABLE,
        },
        "lock_transition": {
            "state_before": FormalRuntimeState.BOOTSTRAP_ONLY,
            "state_after": FormalRuntimeState.RESUMABLE,
        },
        "lock_resume": {
            "state_before": FormalRuntimeState.RESUMABLE,
            "state_after": FormalRuntimeState.RESUMABLE,
        },
    }


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (FormalRuntimeState.ABSENT, "FORMAL_RUNTIME_ABSENT"),
        (FormalRuntimeState.BOOTSTRAP_ONLY, "FORMAL_RUNTIME_BOOTSTRAP_ONLY"),
        (FormalRuntimeState.RESUMABLE, "FORMAL_RUNTIME_RESUMABLE"),
        (FormalRuntimeState.INVALID, "FORMAL_RUNTIME_INVALID"),
    ],
)
def test_each_exact_formal_runtime_state_uses_frozen_value(
    state: FormalRuntimeState, expected: str
) -> None:
    audit: list[dict[str, object]] = []

    assert to_strict_json_native(state, conversion_audit=audit) == expected
    assert audit == [
        {
            "json_path": "$",
            "source_type": (
                "phase_a_harness.formal_runtime_state_machine.FormalRuntimeState"
            ),
            "target_type": "builtins.str",
            "source_summary": f"FormalRuntimeState.{state.name}",
            "target_value": expected,
            "conversion_rule": FORMAL_RUNTIME_STATE_CONVERSION_RULE,
        }
    ]


def test_nested_dict_and_list_states_are_converted_with_exact_paths() -> None:
    source = {
        "nested": {"state": FormalRuntimeState.BOOTSTRAP_ONLY},
        "items": [FormalRuntimeState.INVALID],
    }
    audit: list[dict[str, object]] = []

    normalized = to_strict_json_native(source, conversion_audit=audit)

    assert normalized == {
        "nested": {"state": "FORMAL_RUNTIME_BOOTSTRAP_ONLY"},
        "items": ["FORMAL_RUNTIME_INVALID"],
    }
    assert [row["json_path"] for row in audit] == [
        "$.nested.state",
        "$.items[0]",
    ]


def test_complete_r2_inventory_has_exactly_eleven_convertible_leaves() -> None:
    inventory = scan_non_json_native_leaves(_r2_aggregate_state_inventory())

    assert len(inventory) == 11
    assert {row["python_type"] for row in inventory} == {
        "phase_a_harness.formal_runtime_state_machine.FormalRuntimeState"
    }
    assert all(row["conversion_required"] is True for row in inventory)
    assert all(row["proposed_conversion"] == "value.value" for row in inventory)
    assert {row["json_path"] for row in inventory} == {
        "$.states.absent.state",
        "$.states.bootstrap_only.state",
        "$.states.resumable.state",
        "$.bootstrap.state_before",
        "$.bootstrap.state_after",
        "$.bootstrap_resume.state_before",
        "$.bootstrap_resume.state_after",
        "$.lock_transition.state_before",
        "$.lock_transition.state_after",
        "$.lock_resume.state_before",
        "$.lock_resume.state_after",
    }


def test_complete_r2_inventory_normalizes_to_strict_tree() -> None:
    source = _r2_aggregate_state_inventory()
    audit: list[dict[str, object]] = []

    normalized = to_strict_json_native(source, conversion_audit=audit)

    assert len(audit) == 11
    assert scan_non_json_native_leaves(normalized) == []
    assert_strict_json_native_tree(normalized)


def test_complete_r2_inventory_atomic_write_and_reload_are_exact(tmp_path: Path) -> None:
    source = _r2_aggregate_state_inventory()
    normalized = to_strict_json_native(source)
    destination = tmp_path / "qualification_final_aggregate.json"

    atomic_create_canonical_json(destination, normalized)

    assert read_canonical_json(destination) == normalized


def test_frozen_writer_still_rejects_raw_enum(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="not an exact JSON type"):
        atomic_create_canonical_json(
            tmp_path / "raw_enum.json", {"state": FormalRuntimeState.ABSENT}
        )


def test_input_object_is_not_mutated_and_containers_are_new() -> None:
    source = _r2_aggregate_state_inventory()
    before = copy.deepcopy(source)

    normalized = to_strict_json_native(source)

    assert source == before
    assert normalized is not source
    assert normalized["states"] is not source["states"]
    assert source["states"]["absent"]["state"] is FormalRuntimeState.ABSENT


def test_shared_acyclic_container_is_copied_without_aliasing() -> None:
    shared = [FormalRuntimeState.RESUMABLE]
    source = {"left": shared, "right": shared}

    normalized = to_strict_json_native(source)

    assert normalized == {
        "left": ["FORMAL_RUNTIME_RESUMABLE"],
        "right": ["FORMAL_RUNTIME_RESUMABLE"],
    }
    assert normalized["left"] is not normalized["right"]


@pytest.mark.parametrize("value", [None, "text", True, False, 0, -8, 2.5])
def test_exact_native_scalars_are_accepted_without_conversion(value: object) -> None:
    audit: list[dict[str, object]] = []

    assert to_strict_json_native(value, conversion_audit=audit) == value
    assert audit == []


def test_native_dict_and_list_are_recursively_copied() -> None:
    source = {"a": [1, {"b": None}], "c": 1.5}

    normalized = to_strict_json_native(source)

    assert normalized == source
    assert normalized is not source
    assert normalized["a"] is not source["a"]
    assert normalized["a"][1] is not source["a"][1]


class OtherState(str, Enum):
    ABSENT = "FORMAL_RUNTIME_ABSENT"


def test_other_enum_is_rejected_even_with_same_string_value() -> None:
    with pytest.raises(StrictJsonNativeTreeError, match="only the exact") as caught:
        to_strict_json_native(OtherState.ABSENT)

    assert caught.value.violations[0]["proposed_conversion"] == "REJECT"


class UnknownObject:
    def __str__(self) -> str:
        raise AssertionError("unknown __str__ must not be called")

    def __repr__(self) -> str:
        raise AssertionError("unknown __repr__ must not be called")


def test_unknown_object_is_rejected_without_stringification() -> None:
    with pytest.raises(StrictJsonNativeTreeError, match="UnknownObject"):
        to_strict_json_native(UnknownObject())


class ObjectWithDictionary:
    def __init__(self) -> None:
        self.apparently_native = {"value": 1}


def test_unknown_object_dictionary_is_not_expanded() -> None:
    inventory = scan_non_json_native_leaves(ObjectWithDictionary())

    assert len(inventory) == 1
    assert inventory[0]["class_name"] == "ObjectWithDictionary"
    assert "apparently_native" not in inventory[0]["representative_value"]


@dataclass
class ExampleDataclass:
    value: int


def test_dataclass_instance_is_rejected() -> None:
    with pytest.raises(StrictJsonNativeTreeError, match="ExampleDataclass"):
        to_strict_json_native(ExampleDataclass(1))


@pytest.mark.parametrize("value", [{1, 2}, b"abc", bytearray(b"abc")])
def test_set_and_binary_values_are_rejected(value: object) -> None:
    with pytest.raises(StrictJsonNativeTreeError):
        to_strict_json_native(value)


def test_tuple_conversion_is_disabled_by_zero_inventory() -> None:
    with pytest.raises(StrictJsonNativeTreeError, match="tuple conversion disabled"):
        to_strict_json_native((FormalRuntimeState.ABSENT,))


def test_path_conversion_is_disabled_and_value_is_redacted() -> None:
    sensitive = Path("/not/a/real/path/private-value")

    with pytest.raises(StrictJsonNativeTreeError) as caught:
        to_strict_json_native(sensitive)

    violation = caught.value.violations[0]
    assert "Path conversion disabled" in violation["notes"]
    assert "private-value" not in violation["representative_value"]


@pytest.mark.parametrize(
    "value",
    [np.int64(3), np.float64(0.25), np.bool_(True), np.array([1, 2])],
)
def test_numpy_values_are_rejected_by_zero_inventory(value: object) -> None:
    with pytest.raises(StrictJsonNativeTreeError, match="NumPy conversion disabled"):
        to_strict_json_native(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numbers_are_rejected(value: float) -> None:
    with pytest.raises(StrictJsonNativeTreeError) as caught:
        to_strict_json_native({"metric": value})

    violation = caught.value.violations[0]
    assert violation["json_path"] == "$.metric"
    assert violation["proposed_conversion"] == "REJECT_NONFINITE_NUMBER"


@pytest.mark.parametrize("key", [1, False, ("tuple",)])
def test_non_string_dictionary_keys_are_rejected(key: object) -> None:
    with pytest.raises(StrictJsonNativeTreeError) as caught:
        to_strict_json_native({key: "value"})

    violation = caught.value.violations[0]
    assert violation["json_path"] == "$.<non_string_key[0]>"
    assert violation["proposed_conversion"] == "REJECT_NON_STRING_KEY"


def test_scan_reports_all_independent_violations() -> None:
    value = {
        "enum": FormalRuntimeState.INVALID,
        "tuple": (1, 2),
        "number": float("inf"),
        "unknown": UnknownObject(),
    }

    inventory = scan_non_json_native_leaves(value)

    assert [row["json_path"] for row in inventory] == [
        "$.enum",
        "$.tuple",
        "$.number",
        "$.unknown",
    ]
    assert_strict_json_native_tree(inventory)


def test_assertion_error_lists_every_invalid_path() -> None:
    with pytest.raises(StrictJsonNativeTreeError) as caught:
        assert_strict_json_native_tree(
            {"first": FormalRuntimeState.ABSENT, "second": b"x"}
        )

    assert [row["json_path"] for row in caught.value.violations] == [
        "$.first",
        "$.second",
    ]
    assert "$.first" in str(caught.value)
    assert "$.second" in str(caught.value)


def test_native_tree_assertion_returns_none() -> None:
    assert assert_strict_json_native_tree({"finite": [1.0, True, None]}) is None


def test_conversion_audit_is_not_partially_appended_after_failure() -> None:
    audit: list[dict[str, object]] = [{"preexisting": True}]
    source = [FormalRuntimeState.ABSENT, UnknownObject()]

    with pytest.raises(StrictJsonNativeTreeError):
        to_strict_json_native(source, conversion_audit=audit)

    assert audit == [{"preexisting": True}]


def test_conversion_audit_appends_after_success() -> None:
    audit: list[dict[str, object]] = [{"preexisting": True}]

    to_strict_json_native(FormalRuntimeState.INVALID, conversion_audit=audit)

    assert audit[0] == {"preexisting": True}
    assert audit[1]["conversion_rule"] == FORMAL_RUNTIME_STATE_CONVERSION_RULE


def test_conversion_audit_requires_exact_list() -> None:
    with pytest.raises(TypeError, match="exact list"):
        to_strict_json_native(1, conversion_audit=())  # type: ignore[arg-type]


def test_invalid_custom_path_argument_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty exact string"):
        to_strict_json_native(1, path="")
    with pytest.raises(ValueError, match="non-empty exact string"):
        scan_non_json_native_leaves(1, path="")


def test_cyclic_list_is_rejected_with_origin_path() -> None:
    value: list[object] = []
    value.append(value)

    with pytest.raises(StrictJsonNativeTreeError) as caught:
        to_strict_json_native(value)

    assert caught.value.violations[0]["json_path"] == "$[0]"
    assert caught.value.violations[0]["notes"] == "cyclic container reference to $"


class StringSubclass(str):
    pass


class ListSubclass(list[object]):
    pass


@pytest.mark.parametrize("value", [StringSubclass("x"), ListSubclass([1])])
def test_native_type_subclasses_are_not_silently_accepted(value: object) -> None:
    with pytest.raises(StrictJsonNativeTreeError):
        to_strict_json_native(value)


def test_module_contains_no_json_default_coercion_path() -> None:
    import phase_a_harness.qualification_json_native as module

    source = inspect.getsource(module)
    assert "default=str" not in source
    assert "default = str" not in source
    assert ".__dict__" not in source

