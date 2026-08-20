#!/usr/bin/env python3
"""Qualify only the fixture lifecycle for a future FMB1 execution path."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.prebackend_fixture_execution import (  # noqa: E402
    COMMON_FLAGS,
    PLANNED_OPEN3D_TRIAL_COUNT,
    PLANNED_PCL_TRIAL_COUNT,
    PLANNED_SNAPSHOT_COUNT,
    PLANNED_TRIAL_COUNT,
    FixtureExecutionError,
    analyze_fixture_run,
    build_fixture_plan,
    build_fixture_publication_report,
    expected_fixture_state_sha256,
    fresh_fixture_run,
    resume_fixture_run,
    write_fixture_report,
)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema",
        "classification",
        "FIXTURE_ONLY",
        "FIXTURE_ONLY_DO_NOT_CITE",
        "NOT_REAL_FMB1",
        "NOT_FORMAL_MEASUREMENT",
        "ACTUAL_REGISTRATION_EXECUTION",
        "FORMAL_REGISTRATION_AUTHORIZED",
        "FORMAL_ICP_UNLOCKED",
        "mode",
        "status",
        "requested_workers",
        "planned_trial_count",
        "initial_completed_trial_count",
        "fixture_trial_materialization_count_this_invocation",
        "actual_registration_execution_count_this_invocation",
        "resume_skipped_authenticated_trial_count",
        "recovered_orphan_result_count",
        "final_completed_trial_count",
        "fixture_state_sha256",
        "FORMAL_EXECUTION_UNLOCKED",
    )
    return {field: report.get(field) for field in fields if field in report}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _fixture_result_hashes(root: Path) -> dict[str, str]:
    result_root = root / "fixture_results"
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(result_root.glob("T*.json"))
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise FixtureExecutionError(f"refusing to overwrite fixture report: {path}")
    partial.write_text(content, encoding="utf-8")
    partial.replace(path)


def _qualification_markdown(
    qualification: Mapping[str, Any], resume: Mapping[str, Any]
) -> str:
    return "\n".join(
        (
            "# FMB1 fixture execution-path qualification",
            "",
            "`FIXTURE_ONLY_DO_NOT_CITE`",
            "",
            "- FIXTURE_ONLY=true",
            "- NOT_REAL_FMB1=true",
            "- NOT_FORMAL_MEASUREMENT=true",
            "- ACTUAL_REGISTRATION_EXECUTION=false",
            "- FORMAL_REGISTRATION_AUTHORIZED=false",
            "- FORMAL_ICP_UNLOCKED=false",
            f"- planned_snapshot_count={qualification['planned_snapshot_count']}",
            f"- planned_open3d_trials={qualification['planned_open3d_trials']}",
            f"- planned_pcl_trials={qualification['planned_pcl_trials']}",
            f"- planned_total_trials={qualification['planned_total_trials']}",
            f"- completed_before_interrupt={resume['completed_before_interrupt']}",
            f"- completed_after_resume={resume['completed_after_resume']}",
            f"- unchanged_result_sha_count={resume['unchanged_result_sha_count']}",
            f"- qualification_status={qualification['status']}",
            "",
            "This report qualifies fixture lifecycle structure only. It is not a real "
            "FMB1 measurement and cannot authorize or unlock registration.",
            "",
        )
    )


def _exercise_fail_closed_cases(case_root: Path) -> dict[str, Any]:
    """Create only synthetic corruptions and prove resume rejects each one."""

    case_root.mkdir()
    cases: dict[str, dict[str, Any]] = {}
    for case_name in (
        "partial",
        "orphan",
        "checksum",
        "manifest",
        "duplicate",
        "missing",
    ):
        root = case_root / case_name
        fresh_fixture_run(
            root,
            interrupt_after=3,
            interrupt_window=(
                "after_result_before_index"
                if case_name == "orphan"
                else "after_index_commit"
            ),
        )
        first_result = root / "fixture_results" / "T0001.json"
        if case_name == "partial":
            (root / "fixture_results" / "crash.json.partial").write_text(
                "FIXTURE_ONLY_DO_NOT_CITE\n", encoding="utf-8"
            )
        elif case_name == "orphan":
            pass
        elif case_name == "checksum":
            payload = json.loads(first_result.read_text(encoding="utf-8"))
            payload["lifecycle_status"] = "FIXTURE_TAMPERED"
            first_result.write_text(json.dumps(payload), encoding="utf-8")
        elif case_name == "manifest":
            path = root / "fixture_trial_manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["scene_count"] = 5
            payload.pop("manifest_sha256", None)
            payload["manifest_sha256"] = _canonical_sha256(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
        elif case_name == "duplicate":
            shutil.copyfile(
                first_result, root / "fixture_results" / "duplicate.json"
            )
        elif case_name == "missing":
            first_result.unlink()

        rejected = False
        error = None
        try:
            resume_fixture_run(root)
        except FixtureExecutionError as exc:
            rejected = True
            error = str(exc).replace(str(root.resolve()), "<FIXTURE_ROOT>")
        cases[case_name] = {
            "schema": "mid360_fmb1_prebackend_fixture_fail_closed_case_v1",
            **COMMON_FLAGS,
            "case": case_name,
            "resume_rejected": rejected,
            "error": error,
            "status": "PASS_FIXTURE_ONLY" if rejected else "FAIL_FIXTURE_ONLY",
        }
    return {
        "schema": "mid360_fmb1_prebackend_fixture_fail_closed_cases_v1",
        **COMMON_FLAGS,
        "cases": cases,
        "all_cases_rejected": all(row["resume_rejected"] for row in cases.values()),
        "status": "PASS_FIXTURE_ONLY"
        if all(row["resume_rejected"] for row in cases.values())
        else "FAIL_FIXTURE_ONLY",
    }


def qualify_fixture_execution_path(
    output_root: Path,
    *,
    comparison_workers: int = 4,
    interruption_after: int = 73,
    evidence_dir: Path | None = None,
) -> dict[str, Any]:
    """Exercise fresh/resume/interruption/orphan paths using only fixtures."""

    if comparison_workers < 1:
        raise FixtureExecutionError("comparison_workers must be positive")
    if interruption_after < 1 or interruption_after >= PLANNED_TRIAL_COUNT:
        raise FixtureExecutionError("interruption_after must be inside the trial plan")
    output_root = output_root.expanduser()
    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise FixtureExecutionError("qualification output must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / "fixture_runs"
    run_root.mkdir()
    report_root = (
        evidence_dir.expanduser()
        if evidence_dir is not None
        else output_root / "fixture_reports"
    )
    if report_root.exists() and not report_root.is_dir():
        raise FixtureExecutionError("fixture evidence destination is not a directory")
    report_root.mkdir(parents=True, exist_ok=True)

    expected_sha = expected_fixture_state_sha256()
    plan = build_fixture_plan()

    baseline_root = run_root / "baseline_workers_1"
    fresh = fresh_fixture_run(baseline_root, workers=1)
    resumed = resume_fixture_run(baseline_root, workers=comparison_workers)

    worker_root = run_root / f"fresh_workers_{comparison_workers}"
    worker_fresh = fresh_fixture_run(worker_root, workers=comparison_workers)

    interrupted_root = run_root / "interrupted_after_index_commit"
    interrupted = fresh_fixture_run(
        interrupted_root,
        workers=comparison_workers,
        interrupt_after=interruption_after,
        interrupt_window="after_index_commit",
    )
    before_resume_hashes = _fixture_result_hashes(interrupted_root)
    interrupted_resume = resume_fixture_run(interrupted_root, workers=1)
    after_resume_hashes = _fixture_result_hashes(interrupted_root)
    unchanged_result_sha_count = sum(
        after_resume_hashes.get(name) == digest
        for name, digest in before_resume_hashes.items()
    )
    fail_closed = _exercise_fail_closed_cases(
        output_root / "fixture_fail_closed_cases"
    )

    checks = {
        "plan_is_6x3x10x2_360": (
            plan["scene_count"] == 6
            and plan["station_count"] == 18
            and plan["snapshot_count"] == 180
            and plan["planned_snapshot_count"] == PLANNED_SNAPSHOT_COUNT
            and plan["planned_open3d_trials"] == PLANNED_OPEN3D_TRIAL_COUNT
            and plan["planned_pcl_trials"] == PLANNED_PCL_TRIAL_COUNT
            and plan["planned_total_trials"] == PLANNED_TRIAL_COUNT
            and plan["backend_schema_path_count"] == 2
            and plan["planned_trial_count"] == PLANNED_TRIAL_COUNT
            and sum(
                row["backend_schema_path"] == "OPEN3D_SCHEMA_ADAPTER_FIXTURE"
                for row in plan["trials"]
            )
            == PLANNED_OPEN3D_TRIAL_COUNT
            and sum(
                row["backend_schema_path"] == "PCL_SCHEMA_ADAPTER_FIXTURE"
                for row in plan["trials"]
            )
            == PLANNED_PCL_TRIAL_COUNT
        ),
        "fresh_completed_360_fixture_cells": (
            fresh["status"] == "COMPLETE_FIXTURE_ONLY"
            and fresh["final_completed_trial_count"] == PLANNED_TRIAL_COUNT
        ),
        "completed_resume_materialized_zero": (
            resumed["fixture_trial_materialization_count_this_invocation"] == 0
            and resumed["resume_skipped_authenticated_trial_count"]
            == PLANNED_TRIAL_COUNT
        ),
        "resume_state_sha_stable": (
            fresh["fixture_state_sha256"]
            == resumed["fixture_state_sha256"]
            == expected_sha
        ),
        "workers_invariant": (
            worker_fresh["fixture_state_sha256"]
            == fresh["fixture_state_sha256"]
            == expected_sha
        ),
        "interruption_resume_equivalent": (
            interrupted["status"] == "INTERRUPTED_FIXTURE_ONLY"
            and interrupted_resume["fixture_state_sha256"] == expected_sha
            and interrupted_resume["final_completed_trial_count"]
            == PLANNED_TRIAL_COUNT
        ),
        "interrupted_results_unchanged_on_resume": (
            unchanged_result_sha_count == interruption_after
        ),
        "actual_registration_execution_zero": all(
            report.get("actual_registration_execution_count_this_invocation") == 0
            for report in (
                fresh,
                resumed,
                worker_fresh,
                interrupted,
                interrupted_resume,
            )
        ),
        "formal_unlock_never_created": all(
            report.get("FORMAL_EXECUTION_UNLOCKED") is False
            and report.get("FORMAL_REGISTRATION_AUTHORIZED") is False
            and report.get("FORMAL_ICP_UNLOCKED") is False
            for report in (
                fresh,
                resumed,
                worker_fresh,
                interrupted,
                interrupted_resume,
            )
        ),
        "real_scene_and_backend_reads_zero": (
            plan["real_scene_access_count"] == 0
            and plan["real_backend_output_read_count"] == 0
        ),
        "partial_orphan_checksum_manifest_duplicate_missing_fail_closed": (
            fail_closed["all_cases_rejected"] is True
        ),
    }

    resume_report = {
        "schema": "mid360_fmb1_prebackend_fixture_resume_qualification_v1",
        **COMMON_FLAGS,
        "expected_complete_fixture_state_sha256": expected_sha,
        "fresh": _summary(fresh),
        "completed_resume": _summary(resumed),
        "interrupted_fresh": _summary(interrupted),
        "interrupted_resume": _summary(interrupted_resume),
        "completed_before_interrupt": interrupted["final_completed_trial_count"],
        "completed_after_resume": interrupted_resume[
            "final_completed_trial_count"
        ],
        "duplicate_count": 0,
        "missing_count": 0,
        "checksum_mismatch_count": 0,
        "orphan_count": 0,
        "detected_and_rejected_case_count": sum(
            row["resume_rejected"] for row in fail_closed["cases"].values()
        ),
        "unchanged_result_sha_count": unchanged_result_sha_count,
        "fail_closed_cases": fail_closed,
        "checks": {
            key: value
            for key, value in checks.items()
            if key
            in {
                "completed_resume_materialized_zero",
                "resume_state_sha_stable",
                "interruption_resume_equivalent",
                "interrupted_results_unchanged_on_resume",
            }
        },
        "status": "PASS_FIXTURE_ONLY"
        if all(checks.values())
        else "FAIL_FIXTURE_ONLY",
    }
    analysis_report = analyze_fixture_run(baseline_root)
    publication_report = build_fixture_publication_report(
        analysis_report, resume_report
    )
    qualification_report = {
        "schema": "mid360_fmb1_prebackend_fixture_qualification_v1",
        **COMMON_FLAGS,
        "qualification_scope": "EXECUTION_LIFECYCLE_STRUCTURE_ONLY",
        "planned_snapshot_count": PLANNED_SNAPSHOT_COUNT,
        "planned_open3d_trials": PLANNED_OPEN3D_TRIAL_COUNT,
        "planned_pcl_trials": PLANNED_PCL_TRIAL_COUNT,
        "planned_total_trials": PLANNED_TRIAL_COUNT,
        "planned_trial_count": PLANNED_TRIAL_COUNT,
        "actual_registration_execution_count": 0,
        "expected_complete_fixture_state_sha256": expected_sha,
        "baseline_fresh": _summary(fresh),
        "worker_comparison_fresh": _summary(worker_fresh),
        "checks": checks,
        "fail_closed_cases": fail_closed,
        "analysis_report_sha256": publication_report["analysis_sha256"],
        "resume_report_sha256": publication_report["resume_report_sha256"],
        "publication_report_sha256": publication_report[
            "publication_report_sha256"
        ],
        "status": "PASS_FIXTURE_ONLY"
        if all(checks.values())
        else "FAIL_FIXTURE_ONLY",
    }

    paths = {
        "qualification": report_root / "fixture_execution_path_qualification.json",
        "qualification_markdown": report_root
        / "fixture_execution_path_qualification.md",
        "resume": report_root / "fixture_resume_interruption_report.json",
        "analysis": report_root / "fixture_analysis_dry_run_report.json",
        "publication": report_root / "fixture_publication_dry_run_report.json",
    }
    for path in paths.values():
        if path.exists() or path.with_suffix(path.suffix + ".partial").exists():
            raise FixtureExecutionError(
                f"refusing to overwrite fixture qualification evidence: {path}"
            )
    for name, payload in (
        ("qualification", qualification_report),
        ("resume", resume_report),
        ("analysis", analysis_report),
        ("publication", publication_report),
    ):
        write_fixture_report(paths[name], payload)
    _atomic_write_text(
        paths["qualification_markdown"],
        _qualification_markdown(qualification_report, resume_report),
    )

    return {
        "schema": "mid360_fmb1_prebackend_fixture_qualification_bundle_v1",
        **COMMON_FLAGS,
        "output_root": str(output_root.resolve(strict=True)),
        "evidence_dir": str(report_root.resolve(strict=True)),
        "report_paths": {
            name: str(path.resolve(strict=True)) for name, path in paths.items()
        },
        "qualification": qualification_report,
        "resume": resume_report,
        "analysis": analysis_report,
        "publication": publication_report,
        "status": qualification_report["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root", "--output", dest="runtime_root", type=Path, required=True
    )
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--mode", choices=("fresh", "resume", "qualify"), default="qualify"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--workers", "--comparison-workers", dest="workers", type=int, default=4
    )
    parser.add_argument("--interruption-after", type=int)
    args = parser.parse_args()
    if not args.dry_run:
        print(
            json.dumps(
                {
                    **COMMON_FLAGS,
                    "status": "FAIL_FIXTURE_ONLY",
                    "error": "--dry-run is mandatory; formal execution is unavailable",
                },
                sort_keys=True,
            )
        )
        return 2
    try:
        if args.mode == "qualify":
            report = qualify_fixture_execution_path(
                args.runtime_root,
                comparison_workers=args.workers,
                interruption_after=(
                    73 if args.interruption_after is None else args.interruption_after
                ),
                evidence_dir=args.evidence_dir,
            )
        else:
            if args.evidence_dir is not None:
                raise FixtureExecutionError(
                    "--evidence-dir is supported only by --mode qualify"
                )
            runner = fresh_fixture_run if args.mode == "fresh" else resume_fixture_run
            report = runner(
                args.runtime_root,
                workers=args.workers,
                interrupt_after=args.interruption_after,
                interrupt_window="after_index_commit",
            )
    except FixtureExecutionError as exc:
        print(
            json.dumps(
                {**COMMON_FLAGS, "status": "FAIL_FIXTURE_ONLY", "error": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return (
        0
        if report["status"] in {"PASS_FIXTURE_ONLY", "COMPLETE_FIXTURE_ONLY"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
