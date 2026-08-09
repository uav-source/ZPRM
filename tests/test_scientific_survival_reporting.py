from __future__ import annotations

from collections import Counter

from phase_a_harness.scientific_survival_reporting import (
    claim_wording_boundary_pass,
    deterministic_counterexample_shortlist,
    scientific_claim_matrix,
)


def _candidates() -> list[dict[str, object]]:
    rows = []
    scenes = (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"))
    conditions = ("C1", "C2", "C3", "C4")
    for backend in ("Open3D", "PCL"):
        for index in range(36):
            scene_a, scene_b = scenes[index % len(scenes)]
            rows.append(
                {
                    "backend": backend,
                    "candidate_pair_id": f"{backend}-{index:02d}",
                    "scene_a": scene_a,
                    "scene_b": scene_b,
                    "condition": conditions[index % len(conditions)],
                    "snapshot_a": f"{backend}-a-{index}",
                    "snapshot_b": f"{backend}-b-{index}",
                    "error_ratio": 5.0 + index,
                    "turnover_difference": 0.15 + index / 100.0,
                    "manual_review_status": "AUTOMATIC_CANDIDATE",
                }
            )
    return rows


def test_shortlist_is_deterministic_and_bounded() -> None:
    rows = _candidates()
    assert deterministic_counterexample_shortlist(rows) == deterministic_counterexample_shortlist(list(reversed(rows)))
    result = deterministic_counterexample_shortlist(rows)
    assert len(result) <= 24
    assert Counter(row["backend"] for row in result).most_common(1)[0][1] <= 12
    for backend in ("Open3D", "PCL"):
        rows = [row for row in result if row["backend"] == backend]
        assert len({row["scene_pair"] for row in rows}) >= 3
        assert len({row["condition"] for row in rows}) >= 3
        assert {row["error_ratio_stratum"] for row in rows} == {"LOW", "MID", "HIGH"}
        assert {row["turnover_difference_stratum"] for row in rows} == {"LOW", "MID", "HIGH"}
        use = Counter(
            snapshot
            for row in rows
            for snapshot in (row["snapshot_a"], row["snapshot_b"])
        )
        assert max(use.values()) <= 2
        assert all(row["manual_review_status"] == "PENDING_HUMAN_REVIEW" for row in rows)
        assert all(row["counterexample_claim_authorized"] is False for row in rows)


def test_claim_matrix_enforces_causal_and_universal_prohibitions() -> None:
    rows = scientific_claim_matrix(
        primary_scene_pass=True,
        cross_backend_pass=True,
        reassociation_pass=True,
        model_leakage_pass=True,
        model_robust_pass=True,
        systematic_full_noise_pass=True,
        independent_authorized_term="DETERMINISTIC_ZERO_INITIALIZATION_DISPLACEMENT",
    )
    assert len(rows) == 8
    assert rows[5]["authorized"] is False
    assert rows[6]["authorized"] is False
    assert rows[7]["authorized"] is False
    assert claim_wording_boundary_pass(rows) is True
    claim = rows[3]
    assert claim["authorized"] is True
    assert claim["required_wording"] == "incremental explanatory value"
    assert claim["forbidden_wording"] == "pre-registration failure prediction"
