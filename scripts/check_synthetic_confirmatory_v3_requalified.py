#!/usr/bin/env python3
"""Pre-run qualification for the local Synthetic Confirmatory v3 route."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

DEFAULT_QUALIFICATION_ROOT = Path(
    "/home/lj/zero_perturbation_runtime/qualification/"
    "synthetic_confirmatory_v3_requalified"
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _run(arguments: Sequence[str]) -> dict[str, Any]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        list(arguments),
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return {
        "argv": list(arguments),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    from phase_a_harness.runtime_lifecycle_io import atomic_create_canonical_json

    atomic_create_canonical_json(path, dict(value))


def _git(*arguments: str) -> str:
    result = _run(("git", *arguments))
    if result["exit_code"] != 0:
        raise RuntimeError(
            f"git command failed: {' '.join(arguments)}\n{result['stderr']}"
        )
    return str(result["stdout"]).strip()


def _qualification_identity() -> dict[str, Any]:
    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        DEFAULT_MANIFEST_RELATIVE,
        EXECUTION_CLASSIFICATION,
        EXPECTED_BRANCH,
        EXPECTED_TAG,
        build_requalified_manifest,
        canonical_json_sha256,
        file_sha256,
        scientific_asset_hashes,
        strict_json_object,
    )

    manifest_path = REPOSITORY / DEFAULT_MANIFEST_RELATIVE
    manifest = strict_json_object(manifest_path)
    generated = build_requalified_manifest(repository_root=REPOSITORY)
    tag_type = _git("cat-file", "-t", EXPECTED_TAG)
    commit = _git("rev-parse", "HEAD")
    tag_commit = _git("rev-parse", f"{EXPECTED_TAG}^{{}}")
    status = _git("status", "--porcelain=v1")
    assets = scientific_asset_hashes(REPOSITORY)
    result = {
        "schema_version": (
            "synthetic_confirmatory_v3_requalified_qualification_identity_v1"
        ),
        "execution_classification": EXECUTION_CLASSIFICATION,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "expected_branch": EXPECTED_BRANCH,
        "tag": EXPECTED_TAG,
        "tag_object_type": tag_type,
        "tag_peeled_commit": tag_commit,
        "worktree_clean": status == "",
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_payload_sha256": manifest.get("manifest_payload_sha256"),
        "component_binding_sha256": canonical_json_sha256(
            manifest.get("formal_lifecycle_components")
        ),
        "scientific_assets": assets,
        "generated_manifest_exact_match": manifest == generated,
    }
    result["qualification_identity_pass"] = bool(
        result["branch"] == EXPECTED_BRANCH
        and tag_type == "tag"
        and tag_commit == commit
        and result["worktree_clean"]
        and result["generated_manifest_exact_match"]
        and assets["all_assets_match"] is True
    )
    return result


def _select_smoke_rows() -> list[dict[str, Any]]:
    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        load_requalified_plans,
    )

    snapshots, _trials = load_requalified_plans(REPOSITORY)
    selectors = (
        ("GEOMETRY_RICH_ROOM", "IDEAL_MATCHED"),
        ("LONG_CORRIDOR", "INDEPENDENT_NOISE_FREE"),
        ("REPEATED_STRUCTURE", "FULL_NOISE"),
    )
    selected: list[dict[str, Any]] = []
    for scene, condition in selectors:
        row = next(
            dict(item)
            for item in snapshots
            if item["scene_variant"] == scene and item["condition"] == condition
        )
        selected.append(row)
    return selected


def run_smoke_qualification(root: Path) -> dict[str, Any]:
    """Run three real v3 snapshots and both real frozen backends."""

    from phase_a_harness import synthetic_confirmatory_v3_contract as contract
    from phase_a_harness.formal_lifecycle_contract import deep_thaw
    from phase_a_harness.runtime_lifecycle_io import atomic_create_bytes
    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        EXECUTION_CLASSIFICATION,
        build_requalified_execution_context,
        build_requalified_paths,
        canonical_json_sha256,
        requalified_backend_input_builder,
        requalified_common_record_builder,
        requalified_execute_open3d,
        requalified_execute_pcl,
        requalified_result_validator,
        requalified_snapshot_materializer,
        requalified_snapshot_reader,
        requalified_snapshot_validator,
    )

    smoke_runtime = root / "smoke_runtime"
    paths = build_requalified_paths(smoke_runtime)
    context = build_requalified_execution_context(runtime_paths=paths)
    qualification_identity = {
        "schema_version": "synthetic_confirmatory_v3_requalified_smoke_contract_v1",
        "execution_classification": EXECUTION_CLASSIFICATION,
        "formal_confirmatory_result": False,
        "snapshot_count": 3,
        "trial_count": 6,
    }
    manifest = {
        **qualification_identity,
        "manifest_payload_sha256": canonical_json_sha256(qualification_identity),
    }
    spec = SimpleNamespace(
        repository_root=REPOSITORY,
        paths=paths,
        execution_context=context,
        manifest=manifest,
    )
    paths.snapshot_cache.mkdir(parents=True, exist_ok=False)
    paths.raw_results.mkdir(parents=True, exist_ok=False)
    selected = _select_smoke_rows()
    plan_trials = contract.typed_trial_rows(
        REPOSITORY / "protocols/synthetic_confirmatory_planned_trials_v3.csv"
    )
    results: list[dict[str, Any]] = []
    snapshot_entries: list[dict[str, Any]] = []
    rng_count = 0
    seed_access_count = 0
    for snapshot_row in selected:
        destination = paths.snapshot_cache / snapshot_row["planned_snapshot_id"]
        receipt = requalified_snapshot_materializer(
            spec=spec,
            snapshot_row=snapshot_row,
            destination=destination,
        )
        rng_count += int(receipt["confirmatory_rng_instantiation_count"])
        seed_access_count += int(receipt["confirmatory_seed_access_count"])
        authenticated = requalified_snapshot_reader(
            spec=spec,
            snapshot_row=snapshot_row,
            expected_lock_entry=None,
            arrays=True,
        )
        snapshot_entries.append(
            requalified_snapshot_validator(
                spec=spec,
                snapshot_row=snapshot_row,
                authenticated_snapshot=authenticated,
            )
        )
        snapshot_lock_sha = canonical_json_sha256(snapshot_entries)
        rows = [
            row
            for row in plan_trials
            if row["planned_snapshot_id"] == snapshot_row["planned_snapshot_id"]
        ]
        if len(rows) != 2:
            raise RuntimeError("smoke snapshot does not have exactly two trials")
        for trial_row in rows:
            backend_input = requalified_backend_input_builder(
                spec=spec,
                trial_row=trial_row,
                snapshot_row=snapshot_row,
                authenticated_snapshot=authenticated,
            )
            common = requalified_common_record_builder(
                spec=spec,
                trial_row=trial_row,
                snapshot_row=snapshot_row,
                backend_input=backend_input,
                snapshot_lock_sha256=snapshot_lock_sha,
            )
            raw = (
                requalified_execute_open3d(
                    backend_input=backend_input, common=common
                )
                if trial_row["backend"] == "open3d_point_to_plane"
                else requalified_execute_pcl(
                    backend_input=backend_input, common=common
                )
            )
            validated = requalified_result_validator(
                spec=spec,
                trial_row=trial_row,
                common=common,
                value=raw,
            )
            path = paths.raw_results / (
                trial_row["planned_trial_id"].replace("/", "_").replace(":", "_")
                + ".json"
            )
            atomic_create_bytes(path, _canonical_bytes(validated))
            results.append(
                {
                    "planned_trial_id": trial_row["planned_trial_id"],
                    "planned_snapshot_id": trial_row["planned_snapshot_id"],
                    "scene_variant": trial_row["scene_variant"],
                    "condition": trial_row["condition"],
                    "backend": trial_row["backend"],
                    "schema_version": validated["schema_version"],
                    "solver_failure": validated["solver_failure"],
                    "finite_output": validated["finite_output"],
                    "failure_classification": validated[
                        "failure_classification"
                    ],
                    "result_path": str(path),
                }
            )
    backend_counts = {
        backend: sum(row["backend"] == backend for row in results)
        for backend in ("open3d_point_to_plane", "pcl_point_to_plane")
    }
    report = {
        "schema_version": "synthetic_confirmatory_v3_requalified_smoke_report_v1",
        "execution_classification": EXECUTION_CLASSIFICATION,
        "formal_confirmatory_result": False,
        "mock_backend_used": False,
        "selected_snapshots": selected,
        "snapshot_count": len(selected),
        "trial_count": len(results),
        "backend_trial_counts": backend_counts,
        "native_trial_count": sum(
            row["backend"] not in backend_counts for row in results
        ),
        "confirmatory_seed_access_count": seed_access_count,
        "confirmatory_rng_instantiation_count": rng_count,
        "strict_schema_validation_count": len(results),
        "results": results,
        "runtime_root": str(smoke_runtime),
    }
    report["smoke_qualification_pass"] = bool(
        report["snapshot_count"] == 3
        and report["trial_count"] == 6
        and backend_counts
        == {"open3d_point_to_plane": 3, "pcl_point_to_plane": 3}
        and report["native_trial_count"] == 0
        and report["strict_schema_validation_count"] == 6
        and report["mock_backend_used"] is False
    )
    return report


def run_resume_qualification() -> dict[str, Any]:
    tests = (
        "tests/test_formal_lifecycle_fresh_resume.py",
        "tests/test_formal_lifecycle_snapshot_interrupt.py",
        "tests/test_formal_lifecycle_trial_interrupt.py",
        (
            "tests/test_synthetic_confirmatory_v3_requalified.py::"
            "test_requalified_resume_rejects_uncommitted_orphan"
        ),
    )
    result = _run((sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests))
    report = {
        "schema_version": "synthetic_confirmatory_v3_requalified_resume_qualification_v1",
        "execution_classification": "REQUALIFIED_LOCAL_EXECUTION",
        "tests": list(tests),
        "pytest": result,
        "completed_result_sha_unchanged_asserted": True,
        "valid_trial_reexecution_zero_asserted": True,
        "valid_snapshot_reexecution_zero_asserted": True,
        "unauthenticated_orphan_rejected_asserted": True,
    }
    report["resume_qualification_pass"] = result["exit_code"] == 0
    return report


def run_full_pytest() -> dict[str, Any]:
    result = _run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
        )
    )
    return {
        "schema_version": "synthetic_confirmatory_v3_requalified_test_report_v1",
        "execution_classification": "REQUALIFIED_LOCAL_EXECUTION",
        "command": result,
        "pytest_pass": result["exit_code"] == 0,
    }


def run_static_dry_run() -> dict[str, Any]:
    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        DEFAULT_RUNTIME_ROOT,
    )

    command = _run(
        (
            sys.executable,
            "scripts/run_synthetic_confirmatory_v3_requalified.py",
            "--runtime-root",
            str(DEFAULT_RUNTIME_ROOT),
            "--workers",
            "2",
            "--dry-run",
        )
    )
    value = json.loads(command["stdout"]) if command["stdout"] else {}
    if type(value) is not dict:
        value = {}
    return {
        "schema_version": "synthetic_confirmatory_v3_requalified_static_check_v1",
        "command": command,
        "dry_run": value,
        "static_plan_pass": bool(
            command["exit_code"] == 0
            and value.get("zero_instantiation_pass") is True
            and value.get("planned_snapshot_count") == 595
            and value.get("planned_trial_count") == 1190
            and value.get("planned_snapshot_unique_count") == 595
            and value.get("planned_trial_unique_count") == 1190
            and value.get("backend_trial_counts")
            == {"open3d_point_to_plane": 595, "pcl_point_to_plane": 595}
            and value.get("native_trial_count") == 0
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qualification-root", default=str(DEFAULT_QUALIFICATION_ROOT)
    )
    parser.add_argument("--skip-full-pytest", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(os.path.abspath(args.qualification_root))
    if os.path.lexists(root):
        raise FileExistsError("qualification root must be fresh")
    test_report = (
        {
            "schema_version": "synthetic_confirmatory_v3_requalified_test_report_v1",
            "execution_classification": "REQUALIFIED_LOCAL_EXECUTION",
            "pytest_pass": False,
            "skipped": True,
        }
        if args.skip_full_pytest
        else run_full_pytest()
    )
    if not args.skip_full_pytest and test_report["pytest_pass"] is not True:
        print(json.dumps(test_report, sort_keys=True, separators=(",", ":")))
        return 1
    static = run_static_dry_run()
    if static["static_plan_pass"] is not True:
        print(json.dumps(static, sort_keys=True, separators=(",", ":")))
        return 1
    recovery = run_resume_qualification()
    if recovery["resume_qualification_pass"] is not True:
        print(json.dumps(recovery, sort_keys=True, separators=(",", ":")))
        return 1
    identity = _qualification_identity()
    if identity["qualification_identity_pass"] is not True:
        print(json.dumps(identity, sort_keys=True, separators=(",", ":")))
        return 1
    root.mkdir(parents=True, exist_ok=False)
    smoke = run_smoke_qualification(root)
    if smoke["smoke_qualification_pass"] is not True:
        raise RuntimeError("real-backend smoke qualification failed")
    if _qualification_identity() != identity:
        raise RuntimeError("repository identity changed during qualification")
    for report in (test_report, static, recovery, smoke):
        report["qualification_identity"] = identity
    _write_json(root / "test_report.json", test_report)
    _write_json(root / "static_plan_report.json", static)
    _write_json(root / "resume_qualification_report.json", recovery)
    _write_json(root / "smoke_qualification_report.json", smoke)
    result = {
        "schema_version": "synthetic_confirmatory_v3_requalified_qualification_v1",
        "qualification_root": str(root),
        "pytest_pass": test_report["pytest_pass"],
        "static_plan_pass": static["static_plan_pass"],
        "resume_qualification_pass": recovery["resume_qualification_pass"],
        "smoke_qualification_pass": smoke["smoke_qualification_pass"],
        "qualification_identity": identity,
    }
    result["qualification_pass"] = all(
        result[name] is True
        for name in (
            "pytest_pass",
            "static_plan_pass",
            "resume_qualification_pass",
            "smoke_qualification_pass",
        )
    ) and identity["qualification_identity_pass"] is True
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["qualification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
