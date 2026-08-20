from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.prebackend_fixture_execution import (
    FIXTURE_CLASSIFICATION,
    PLANNED_TRIAL_COUNT,
    FixtureExecutionError,
    analyze_fixture_run,
    build_fixture_plan,
    build_fixture_publication_report,
    expected_fixture_state_sha256,
    fresh_fixture_run,
    resume_fixture_run,
)
from tools.mid360_formal_batch1.qualify_formal_execution_path import (
    qualify_fixture_execution_path,
)


def test_plan_is_exact_360_fixture_only_and_never_unlocks() -> None:
    plan = build_fixture_plan()
    assert plan["scene_count"] == 6
    assert plan["station_count"] == 18
    assert plan["snapshot_count"] == 180
    assert plan["backend_schema_path_count"] == 2
    assert plan["planned_snapshot_count"] == 180
    assert plan["planned_open3d_trials"] == 180
    assert plan["planned_pcl_trials"] == 180
    assert plan["planned_total_trials"] == 360
    assert plan["planned_trial_count"] == PLANNED_TRIAL_COUNT == 360
    assert len({row["trial_id"] for row in plan["trials"]}) == 360
    assert plan["classification"] == FIXTURE_CLASSIFICATION
    assert plan["FIXTURE_ONLY"] is True
    assert plan["FIXTURE_ONLY_DO_NOT_CITE"] is True
    assert plan["NOT_REAL_FMB1"] is True
    assert plan["NOT_FORMAL_MEASUREMENT"] is True
    assert plan["ACTUAL_REGISTRATION_EXECUTION"] is False
    assert plan["FORMAL_REGISTRATION_AUTHORIZED"] is False
    assert plan["FORMAL_ICP_UNLOCKED"] is False
    assert plan["FORMAL_EXECUTION_UNLOCKED"] is False
    assert plan["real_scene_access_count"] == 0
    assert plan["real_backend_output_read_count"] == 0
    assert all(row["scene_id"].startswith("FIXTURE_SCENE_") for row in plan["trials"])


def test_fresh_completed_resume_and_workers_are_content_invariant(tmp_path: Path) -> None:
    expected = expected_fixture_state_sha256()
    first_root = tmp_path / "workers1"
    fresh = fresh_fixture_run(first_root, workers=1)
    assert fresh["status"] == "COMPLETE_FIXTURE_ONLY"
    assert fresh["fixture_trial_materialization_count_this_invocation"] == 360
    assert fresh["fixture_state_sha256"] == expected

    resumed = resume_fixture_run(first_root, workers=7)
    assert resumed["fixture_trial_materialization_count_this_invocation"] == 0
    assert resumed["resume_skipped_authenticated_trial_count"] == 360
    assert resumed["fixture_state_sha256"] == fresh["fixture_state_sha256"]

    second = fresh_fixture_run(tmp_path / "workers7", workers=7)
    assert second["fixture_state_sha256"] == expected

    analysis = analyze_fixture_run(first_root)
    publication = build_fixture_publication_report(analysis, resumed)
    assert analysis["authenticated_fixture_result_count"] == 360
    assert analysis["backend_invocation_count"] == 0
    assert analysis["scientific_result_field_count"] == 0
    assert analysis["scene_is_highest_independent_unit"] is True
    assert analysis["stations_are_nested_repeats"] is True
    assert analysis["snapshots_are_nested_repeats"] is True
    assert analysis["snapshots_are_independent_scenes"] is False
    assert analysis["scientific_values_produced"] is False
    assert analysis["dry_run_structure_endpoint_count"] == 6
    assert analysis["dry_run_structure_endpoint_names"] == [
        "scene_summaries",
        "station_variance",
        "snapshot_repeated_observations",
        "cross_backend",
        "weak_vs_rich",
        "reassociation",
    ]
    assert all(analysis["dry_run_structure_endpoints"].values())
    assert publication["publication_performed"] is False
    assert publication["citation_allowed"] is False
    assert publication["FORMAL_ICP_UNLOCKED"] is False


def test_interrupted_after_index_resumes_without_repeating_results(tmp_path: Path) -> None:
    root = tmp_path / "indexed_interruption"
    interrupted = fresh_fixture_run(
        root,
        workers=3,
        interrupt_after=5,
        interrupt_window="after_index_commit",
    )
    assert interrupted["status"] == "INTERRUPTED_FIXTURE_ONLY"
    assert interrupted["final_completed_trial_count"] == 5
    before = {
        path.name: path.read_bytes()
        for path in sorted((root / "fixture_results").glob("*.json"))
    }

    resumed = resume_fixture_run(root, workers=2)
    assert resumed["recovered_orphan_result_count"] == 0
    assert resumed["fixture_trial_materialization_count_this_invocation"] == 355
    assert resumed["final_completed_trial_count"] == 360
    assert resumed["fixture_state_sha256"] == expected_fixture_state_sha256()
    assert resumed["ACTUAL_REGISTRATION_EXECUTION"] is False
    assert all(
        (root / "fixture_results" / name).read_bytes() == content
        for name, content in before.items()
    )


def test_resume_rejects_canonical_but_unindexed_orphan(tmp_path: Path) -> None:
    root = tmp_path / "orphan"
    fresh_fixture_run(
        root,
        interrupt_after=5,
        interrupt_window="after_result_before_index",
    )
    with pytest.raises(FixtureExecutionError, match="orphan"):
        resume_fixture_run(root)


@pytest.mark.parametrize(
    "tamper",
    ["partial", "checksum", "manifest", "duplicate", "missing"],
)
def test_resume_rejects_partial_or_tampered_fixture_state(
    tmp_path: Path, tamper: str
) -> None:
    root = tmp_path / tamper
    fresh_fixture_run(root, interrupt_after=3)
    first = root / "fixture_results" / "T0001.json"
    if tamper == "partial":
        (root / "fixture_results" / "crash.json.partial").write_text(
            "partial", encoding="utf-8"
        )
    elif tamper == "checksum":
        payload = json.loads(first.read_text(encoding="utf-8"))
        payload["lifecycle_status"] = "TAMPERED"
        first.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "manifest":
        manifest = root / "fixture_trial_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["planned_trial_count"] = 359
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "duplicate":
        (root / "fixture_results" / "duplicate.json").write_bytes(first.read_bytes())
    elif tamper == "missing":
        first.unlink()
    with pytest.raises(FixtureExecutionError):
        resume_fixture_run(root)


def test_qualification_tool_emits_all_fixture_only_report_shapes(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    bundle = qualify_fixture_execution_path(
        tmp_path / "qualification",
        comparison_workers=3,
        interruption_after=7,
        evidence_dir=evidence_dir,
    )
    assert bundle["status"] == "PASS_FIXTURE_ONLY"
    assert bundle["FIXTURE_ONLY_DO_NOT_CITE"] is True
    assert bundle["ACTUAL_REGISTRATION_EXECUTION"] is False
    assert set(bundle["report_paths"]) == {
        "qualification",
        "qualification_markdown",
        "resume",
        "analysis",
        "publication",
    }
    assert Path(bundle["report_paths"]["qualification"]).name == (
        "fixture_execution_path_qualification.json"
    )
    assert Path(bundle["report_paths"]["qualification_markdown"]).name == (
        "fixture_execution_path_qualification.md"
    )
    assert Path(bundle["report_paths"]["resume"]).name == (
        "fixture_resume_interruption_report.json"
    )
    assert Path(bundle["report_paths"]["analysis"]).name == (
        "fixture_analysis_dry_run_report.json"
    )
    assert Path(bundle["report_paths"]["publication"]).name == (
        "fixture_publication_dry_run_report.json"
    )
    assert all(bundle["qualification"]["checks"].values())
    assert bundle["qualification"]["planned_snapshot_count"] == 180
    assert bundle["qualification"]["planned_open3d_trials"] == 180
    assert bundle["qualification"]["planned_pcl_trials"] == 180
    assert bundle["qualification"]["planned_total_trials"] == 360
    resume = bundle["resume"]
    assert resume["completed_before_interrupt"] == 7
    assert resume["completed_after_resume"] == 360
    assert resume["duplicate_count"] == 0
    assert resume["missing_count"] == 0
    assert resume["checksum_mismatch_count"] == 0
    assert resume["orphan_count"] == 0
    assert resume["detected_and_rejected_case_count"] == 6
    assert resume["unchanged_result_sha_count"] == 7
    fail_closed = bundle["qualification"]["fail_closed_cases"]
    assert fail_closed["all_cases_rejected"] is True
    assert set(fail_closed["cases"]) == {
        "partial",
        "orphan",
        "checksum",
        "manifest",
        "duplicate",
        "missing",
    }
    for name, path in bundle["report_paths"].items():
        assert Path(path).parent == evidence_dir.resolve()
        if name == "qualification_markdown":
            assert "FIXTURE_ONLY_DO_NOT_CITE" in Path(path).read_text(
                encoding="utf-8"
            )
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["FIXTURE_ONLY"] is True
        assert payload["FIXTURE_ONLY_DO_NOT_CITE"] is True
        assert payload["NOT_REAL_FMB1"] is True
        assert payload["NOT_FORMAL_MEASUREMENT"] is True
        assert payload["ACTUAL_REGISTRATION_EXECUTION"] is False
