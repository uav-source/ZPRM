from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from phase_a_harness.scientific_survival_integrity import publisher_only_change_scope


FIELDS = (
    "FULL_SYNTHETIC_DEVELOPMENT_PASS",
    "FULL_SYNTHETIC_PRIMARY_SCENE_EFFECT_PASS",
    "FULL_SYNTHETIC_CROSS_BACKEND_PASS",
    "FULL_SYNTHETIC_SCENE_RANK_STABILITY_PASS",
    "COMMON_ASSOCIATION_ANALYSIS_PASS",
    "REASSOCIATION_MECHANISM_SUPPORTED",
    "LOCAL_METRIC_INCREMENTAL_VALUE_PASS",
    "SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED",
)


def _inventory() -> dict[str, object]:
    return {
        "missing_count": 0,
        "size_mismatch_count": 0,
        "sha256_mismatch_count": 0,
        "mismatch_count_by_semantic_role": {},
        "semantic_role_counts": {"raw_trial_result": 2100},
    }


def test_publisher_scope_accepts_only_publisher_scientific_code_change() -> None:
    decision = {name: True for name in FIELDS}
    report = publisher_only_change_scope(
        inventory_verification=_inventory(),
        source_decision=decision,
        published_decision=decision,
        changed_paths=[
            "src/phase_a_harness/full_synthetic_publisher.py",
            "src/phase_a_harness/scientific_survival_integrity.py",
        ],
        matplotlib_version="3.8.2",
    )
    assert report["publisher_repair_scope_pass"] is True
    assert report["publisher_code_change_count"] == 1
    assert report["trial_rerun_count"] == 0


def test_publisher_scope_rejects_backend_or_frozen_field_change() -> None:
    before = {name: True for name in FIELDS}
    after = dict(before)
    after["FULL_SYNTHETIC_CROSS_BACKEND_PASS"] = False
    report = publisher_only_change_scope(
        inventory_verification=_inventory(),
        source_decision=before,
        published_decision=after,
        changed_paths=[
            "src/phase_a_harness/full_synthetic_publisher.py",
            "src/phase_a_harness/pcl_backend.py",
        ],
        matplotlib_version="3.8.2",
    )
    assert report["publisher_repair_scope_pass"] is False
    assert report["backend_code_change_count"] == 1
    assert report["frozen_scientific_field_difference_count"] == 1
