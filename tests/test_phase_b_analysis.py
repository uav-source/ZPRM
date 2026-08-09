from __future__ import annotations

import csv
import json
from pathlib import Path

from phase_a_harness.contracts import file_sha256
from phase_a_harness.phase_a_trial_result_schema import OPEN3D_BACKEND, PCL_BACKEND
from phase_a_harness.phase_b_analysis import (
    BACKEND_LABELS,
    BACKENDS,
    CONDITIONS,
    GEOMETRY_SEEDS,
    RICH_SCENE,
    SCENES,
    _rankings,
    _weak_rich,
)
from phase_a_harness.phase_b_artifact_verifier import (
    EXPECTED_TABLE_ROWS,
    REQUIRED_PHASE_B_FILES,
    verify_phase_b_artifact,
)
from phase_a_harness.phase_b_independent_verifier import (
    COMPARISON_KEYS,
    phase_b_analysis_verifier_difference_count,
)


BASE = {
    "GEOMETRY_RICH_ROOM": 0.0005,
    "END_FACE_TRANSITION_PRESENT": 0.0010,
    "END_FACE_TRANSITION_WEAK": 0.0020,
    "END_FACE_TRANSITION_ABSENT": 0.0030,
    "REPEATED_STRUCTURE": 0.0040,
    "PARALLEL_WALLS": 0.0050,
    "LONG_CORRIDOR": 0.0060,
}


def _summaries(values: dict[str, float] | None = None):
    selected = BASE if values is None else values
    return [
        {
            "backend": BACKEND_LABELS[backend],
            "backend_schema_name": backend,
            "condition": condition,
            "rotation_max_rad": 3e-5,
            "rotation_median_rad": 2e-5,
            "rotation_min_rad": 1e-5,
            "scene_variant": scene,
            "translation_max_m": selected[scene] + 1e-5,
            "translation_median_m": selected[scene],
            "translation_min_m": max(0.0, selected[scene] - 1e-5),
            "trial_count": 3,
        }
        for condition in CONDITIONS
        for backend in BACKENDS
        for scene in SCENES
    ]


def _raw(*, pcl_last_seed_fails: bool = False):
    rows = []
    for condition in CONDITIONS:
        for backend in BACKENDS:
            for scene in SCENES:
                for index, seed in enumerate(GEOMETRY_SEEDS):
                    value = BASE[scene] + (index - 1) * 1e-5
                    if scene == "LONG_CORRIDOR" and index == 2:
                        value = 0.0004
                    if pcl_last_seed_fails and backend == PCL_BACKEND and scene == "LONG_CORRIDOR" and index == 1:
                        value = 0.0004
                    rows.append(
                        {
                            "backend": BACKEND_LABELS[backend],
                            "backend_schema_name": backend,
                            "condition": condition,
                            "geometry_seed": seed,
                            "planned_trial_id": f"{condition}/{backend}/{scene}/{seed}",
                            "rotation_update_rad": 1e-5,
                            "scene_variant": scene,
                            "snapshot_id": f"{condition}/{scene}/{seed}",
                            "translation_update_m": value,
                        }
                    )
    return rows


def test_phase_b_spearman_and_scene_rank_gates_pass():
    ranking, cross, scene_pass, cross_pass = _rankings(_summaries())
    assert scene_pass and cross_pass
    assert [row["rho"] for row in cross] == [1.0, 1.0]
    assert len(ranking) == 28
    rich = [row for row in ranking if row["scene_variant"] == RICH_SCENE]
    assert all(row["translation_rank_ascending_average_ties"] == 1.0 for row in rich)


def test_phase_b_average_rank_ties_are_reported():
    values = dict(BASE)
    values["END_FACE_TRANSITION_PRESENT"] = values[RICH_SCENE]
    ranking, _, scene_pass, _ = _rankings(_summaries(values))
    rich = [row for row in ranking if row["scene_variant"] == RICH_SCENE]
    assert scene_pass
    assert all(row["is_tied"] for row in rich)
    assert all(row["translation_rank_ascending_average_ties"] == 1.5 for row in rich)


def test_phase_b_common_weak_scene_uses_frozen_candidate_order():
    effects, selected, consistency, effect_pass, seed_pass = _weak_rich(_summaries(), _raw())
    assert effect_pass and seed_pass
    assert selected == {condition: "LONG_CORRIDOR" for condition in CONDITIONS}
    chosen = [row for row in effects if row["selected_common_weak_scene"]]
    assert len(chosen) == 4 and all(row["backend_pass"] for row in chosen)
    assert all(row["absolute_difference_m"] >= 0.001 for row in chosen)
    assert all(row["weak_rich_ratio"] >= 2.0 for row in chosen)
    assert len(consistency) == 12


def test_phase_b_near_zero_rich_denominator_is_null_and_fails():
    values = dict(BASE)
    values[RICH_SCENE] = 1e-13
    effects, selected, _, effect_pass, _ = _weak_rich(_summaries(values), _raw())
    assert not effect_pass
    assert selected == {condition: None for condition in CONDITIONS}
    assert all(row["weak_rich_ratio"] is None for row in effects)
    assert not any(row["backend_pass"] for row in effects)


def test_phase_b_seed_consistency_requires_two_of_three_per_backend():
    _, _, rows, _, seed_pass = _weak_rich(_summaries(), _raw(pcl_last_seed_fails=True))
    assert not seed_pass
    pcl = [row for row in rows if row["backend_schema_name"] == PCL_BACKEND]
    assert all(row["direction_success_count"] == 1 for row in pcl)
    assert not any(row["backend_consistency_pass"] for row in pcl)


def test_phase_b_analysis_verifier_difference_is_section_count():
    primary = {key: {"same": True} for key in COMPARISON_KEYS}
    independent = {key: {"same": True} for key in COMPARISON_KEYS}
    assert phase_b_analysis_verifier_difference_count(primary, independent) == 0
    independent["weak_rich_effect"] = {"same": False}
    independent["final_decision"] = {"same": False}
    assert phase_b_analysis_verifier_difference_count(primary, independent) == 2


def _write_csv(path: Path, count: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row"])
        writer.writeheader()
        writer.writerows({"row": index} for index in range(count))


def _synthetic_artifact(root: Path):
    for relative, count in EXPECTED_TABLE_ROWS.items():
        _write_csv(root / relative, count)
    _write_csv(root / "tables/failure_inventory.csv", 2)
    _write_csv(root / "tables/gate_summary.csv", 6)
    for relative in REQUIRED_PHASE_B_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".png"):
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"synthetic" * 20)
    (root / "phase_b_report.md").write_text("# synthetic\n", encoding="utf-8")
    decision = {
        "PHASE_B_SIGNAL_PASS": True,
        "FULL_SYNTHETIC_DEVELOPMENT_PROTOCOL_DESIGN_AUTHORIZED": True,
        "FULL_SYNTHETIC_DEVELOPMENT_RUN_AUTHORIZED": False,
        "CONFIRMATORY_AUTHORIZED": False,
        "REAL_DATA_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
    }
    analysis = {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": 0,
        "solver_failure_count_by_backend": {"Open3D": 0, "PCL": 0},
        "nonfinite_output_count_by_backend": {"Open3D": 0, "PCL": 0},
        "final_decision": decision,
    }
    (root / "primary_analysis.json").write_text(json.dumps(analysis) + "\n", encoding="utf-8")
    (root / "independent_verification.json").write_text(json.dumps(analysis) + "\n", encoding="utf-8")
    (root / "final_decision.json").write_text(json.dumps(decision) + "\n", encoding="utf-8")
    (root / "run_manifest.json").write_text(json.dumps({
        "run_id": "phase-b-signal-v1",
        "completed_snapshot_count": 42,
        "completed_trial_count": 84,
        "open3d_trial_count": 42,
        "pcl_trial_count": 42,
        "native_execution_count": 0,
        "native_trial_count": 0,
    }) + "\n", encoding="utf-8")
    checksum_paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "artifact_verification.json"}
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in checksum_paths),
        encoding="utf-8",
    )


def test_phase_b_artifact_verifier_accepts_complete_synthetic_artifact(tmp_path):
    _synthetic_artifact(tmp_path)
    assert verify_phase_b_artifact(tmp_path, write_report=False)["ARTIFACT_VERIFICATION_PASS"]


def test_phase_b_artifact_verifier_detects_sha_tampering(tmp_path):
    _synthetic_artifact(tmp_path)
    (tmp_path / "tables/trial_results.csv").write_text("row\ntampered\n", encoding="utf-8")
    report = verify_phase_b_artifact(tmp_path, write_report=False)
    assert not report["ARTIFACT_VERIFICATION_PASS"]
    assert report["sha256_mismatch_count"] == 1
    assert report["table_row_count_mismatch_count"] == 1
