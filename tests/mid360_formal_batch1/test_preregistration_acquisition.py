from __future__ import annotations

import json

import pytest

from experiments.mid360_formal_batch1.preregistration_acquisition import (
    AcquisitionInventoryError,
    EXPECTED_MAPPING,
    expected_raw_filenames,
    validate_raw_bag_names,
)


def test_expected_names_bind_strict_mapping_and_roles() -> None:
    rows = validate_raw_bag_names(expected_raw_filenames())
    assert len(rows) == 36
    assert len(EXPECTED_MAPPING) == 18
    assert [row["pair_index"] for row in rows[::2]] == list(range(1, 19))
    assert [row["role"] for row in rows[:4]] == ["MAP", "QUERY", "MAP", "QUERY"]
    assert rows[0]["raw_filename"] == "mid360_20260819_205431_part1_20s.bag"
    assert rows[0]["canonical_filename"] == "FMB1_R01_S01_MAP_20260819_205431.bag"
    assert rows[1]["raw_filename"] == "mid360_20260819_205431_part2_15s.bag"
    assert rows[1]["canonical_filename"] == "FMB1_R01_S01_QUERY_20260819_205431.bag"
    json.dumps(rows, allow_nan=False)


def test_part_role_tamper_fails_closed() -> None:
    names = list(expected_raw_filenames())
    names[0] = names[0].replace("part1_20s", "part1_15s")
    with pytest.raises(AcquisitionInventoryError) as captured:
        validate_raw_bag_names(names)
    assert captured.value.code == "FMB1_INPUT_INVENTORY_FAIL"


def test_file_count_tamper_fails_closed() -> None:
    with pytest.raises(AcquisitionInventoryError) as captured:
        validate_raw_bag_names(expected_raw_filenames()[:-1])
    assert captured.value.code == "FMB1_INPUT_INVENTORY_FAIL"
    assert captured.value.details["supplied_file_count"] == 35


def test_capture_prefix_tamper_is_user_mapping_conflict() -> None:
    names = list(expected_raw_filenames())
    for index, name in enumerate(names):
        if "20260819_215146" in name:
            names[index] = name.replace("20260819_215146", "20260819_215147")
    with pytest.raises(AcquisitionInventoryError) as captured:
        validate_raw_bag_names(names)
    assert captured.value.code == "USER_MAPPING_CONFLICT"
    assert captured.value.details["missing_capture_prefixes"] == ["20260819_215146"]
    assert captured.value.details["unexpected_capture_prefixes"] == ["20260819_215147"]
