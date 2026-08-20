from __future__ import annotations

from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.w04_replacement import (
    W04InventoryError,
    expected_w04_raw_filenames,
    validate_w04_raw_names,
)


def test_exact_w04_mapping_is_frozen() -> None:
    rows = validate_w04_raw_names(expected_w04_raw_filenames())
    assert len(rows) == 6
    assert [row["station_id"] for row in rows] == [
        "S01",
        "S01",
        "S02",
        "S02",
        "S03",
        "S03",
    ]
    assert [row["role"] for row in rows] == [
        "MAP",
        "QUERY",
        "MAP",
        "QUERY",
        "MAP",
        "QUERY",
    ]
    assert all(row["scene_id"] == "FMB1_W04" for row in rows)
    assert all(row["semantic_candidate_label"] == "WEAK_CANDIDATE" for row in rows)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_w04_inventory_count_and_pair_tampering_fails(mutation: str) -> None:
    names = list(expected_w04_raw_filenames())
    if mutation == "missing":
        names.pop()
    elif mutation == "duplicate":
        names[-1] = names[0]
    else:
        names.append("mid360_20260820_082999_part1_20s.bag")
    with pytest.raises(W04InventoryError) as error:
        validate_w04_raw_names(names)
    assert error.value.code == "W04_INPUT_INVENTORY_FAIL"


def test_w04_mapping_prefix_tampering_fails_closed() -> None:
    names = list(expected_w04_raw_filenames())
    names[-2:] = [
        "mid360_20260820_082208_part1_20s.bag",
        "mid360_20260820_082208_part2_15s.bag",
    ]
    with pytest.raises(W04InventoryError) as error:
        validate_w04_raw_names(names)
    assert error.value.code == "W04_USER_MAPPING_CONFLICT"


def test_w04_canonical_names_are_manifest_only_identifiers() -> None:
    rows = validate_w04_raw_names(expected_w04_raw_filenames())
    assert rows[0]["canonical_filename"] == (
        "FMB1_W04_S01_MAP_20260820_081749.bag"
    )
    assert all(Path(row["raw_filename"]).name == row["raw_filename"] for row in rows)
