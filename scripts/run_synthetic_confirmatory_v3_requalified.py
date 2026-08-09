#!/usr/bin/env python3
"""Execute the independently requalified Synthetic Confirmatory v3 route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

QUALIFICATION_ROOT = Path(
    "/home/lj/zero_perturbation_runtime/qualification/"
    "synthetic_confirmatory_v3_requalified"
)


def _command(arguments: Sequence[str], *, check: bool = False) -> dict[str, Any]:
    completed = subprocess.run(
        list(arguments),
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"{completed.stderr}"
        )
    return {
        "argv": list(arguments),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _git(*arguments: str) -> str:
    result = _command(("git", *arguments), check=True)
    return str(result["stdout"]).strip()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _verify_initial_source_archive() -> dict[str, Any]:
    """Re-authenticate the immutable initial ZIP commit against SHA256SUMS."""

    initial = _git("rev-list", "--max-parents=0", "HEAD").splitlines()
    if len(initial) != 1:
        raise RuntimeError("repository must have exactly one initial archive commit")
    commit = initial[0]
    sums = _command(("git", "show", f"{commit}:SHA256SUMS"), check=True)[
        "stdout"
    ]
    checked = mismatch = 0
    mismatches: list[str] = []
    for line in str(sums).splitlines():
        digest, relative = line.split("  ", 1)
        payload = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
        )
        actual = hashlib.sha256(payload.stdout).hexdigest()
        passed = payload.returncode == 0 and actual == digest
        checked += 1
        mismatch += int(not passed)
        if not passed:
            mismatches.append(relative)
    return {
        "initial_commit": commit,
        "checked_file_count": checked,
        "mismatch_count": mismatch,
        "mismatch_files": mismatches,
        "pass": checked == 267 and mismatch == 0,
    }


def collect_environment_report() -> dict[str, Any]:
    """Collect the complete executable/environment evidence without mutation."""

    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        EXECUTION_CLASSIFICATION,
        FROZEN_PYTHON,
        PCL_CLI_RELATIVE,
        PCL_CLI_SHA256,
        file_sha256,
    )

    if Path(sys.executable).resolve() != FROZEN_PYTHON.resolve():
        raise RuntimeError(
            f"frozen Python required: {FROZEN_PYTHON}; actual: {sys.executable}"
        )
    import numpy
    import open3d
    import scipy

    expected_packages = {
        "Python": "3.11.15",
        "numpy": "1.26.4",
        "scipy": "1.11.4",
        "Open3D": "0.19.0+b012259",
        "jsonschema": "4.25.1",
        "matplotlib": "3.10.3",
        "PyYAML": "6.0.2",
    }
    actual_packages = {
        "Python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "Open3D": open3d.__version__,
        "jsonschema": _package_version("jsonschema"),
        "matplotlib": _package_version("matplotlib"),
        "PyYAML": _package_version("PyYAML"),
    }
    package_match = {
        name: actual_packages[name] == expected
        for name, expected in expected_packages.items()
    }
    os_release: dict[str, str] = {}
    release_path = Path("/etc/os-release")
    if release_path.is_file():
        for line in release_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip().strip('"')
    cpu = _command(("lscpu",), check=True)
    memory = _command(("free", "-h"), check=True)
    pip_freeze = _command((sys.executable, "-m", "pip", "freeze"), check=True)
    source_archive_check = _command(("sha256sum", "-c", "SHA256SUMS"))
    initial_source_archive_check = _verify_initial_source_archive()
    pcl_path = REPOSITORY / PCL_CLI_RELATIVE
    file_report = _command(("file", str(pcl_path)), check=True)
    ldd_report = _command(("ldd", str(pcl_path)), check=True)
    pcl_probe = _command((str(pcl_path),))
    try:
        pcl_self_report = json.loads(str(pcl_probe["stdout"]))
    except (json.JSONDecodeError, TypeError):
        pcl_self_report = None
    ldd_text = str(ldd_report["stdout"])
    missing = sorted(
        line.strip() for line in ldd_text.splitlines() if "not found" in line
    )
    expected_pcl_libraries = {
        "libpcl_common.so.1.15",
        "libpcl_features.so.1.15",
        "libpcl_filters.so.1.15",
        "libpcl_io.so.1.15",
        "libpcl_io_ply.so.1.15",
        "libpcl_kdtree.so.1.15",
        "libpcl_octree.so.1.15",
        "libpcl_sample_consensus.so.1.15",
        "libpcl_search.so.1.15",
    }
    pcl_libraries = sorted(
        {
            match.group(1)
            for match in re.finditer(r"(libpcl_[^\s]+\.so\.1\.15)", ldd_text)
        }
    )
    pcl_pass = bool(
        file_sha256(pcl_path) == PCL_CLI_SHA256
        and os.access(pcl_path, os.X_OK)
        and ldd_report["exit_code"] == 0
        and not missing
        and set(pcl_libraries) == expected_pcl_libraries
        and type(pcl_self_report) is dict
        and pcl_self_report.get("pcl_version") == "1.15.1"
    )
    git_status = _git("status", "--porcelain")
    report = {
        "schema_version": "synthetic_confirmatory_v3_environment_report_v1",
        "execution_classification": EXECUTION_CLASSIFICATION,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "os_pretty_name": os_release.get("PRETTY_NAME"),
            "os_id": os_release.get("ID"),
            "os_version_id": os_release.get("VERSION_ID"),
            "kernel": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "lscpu": cpu["stdout"],
            "memory": memory["stdout"],
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
            "pip_version": _package_version("pip"),
        },
        "packages": {
            "expected": expected_packages,
            "actual": actual_packages,
            "exact_match": package_match,
            "pip_freeze": str(pip_freeze["stdout"]).splitlines(),
        },
        "open3d": {
            "version": open3d.__version__,
            "module_path": str(Path(open3d.__file__).resolve()),
            "build_config": dict(getattr(open3d, "_build_config", {})),
        },
        "pcl": {
            "expected_version": "1.15.1",
            "cli_path": str(pcl_path),
            "cli_sha256": file_sha256(pcl_path),
            "expected_cli_sha256": PCL_CLI_SHA256,
            "executable": os.access(pcl_path, os.X_OK),
            "file": file_report,
            "ldd": ldd_report,
            "missing_dynamic_libraries": missing,
            "expected_libpcl_1_15": sorted(expected_pcl_libraries),
            "resolved_libpcl_1_15": pcl_libraries,
            "self_report": pcl_self_report,
            "self_report_exit_code": pcl_probe["exit_code"],
            "pcl_qualification_pass": pcl_pass,
        },
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "tags_at_head": _git("tag", "--points-at", "HEAD").splitlines(),
            "worktree_clean": git_status == "",
            "porcelain": git_status.splitlines(),
        },
        "current_worktree_sha256sums": {
            **source_archive_check,
            "scope": (
                "post-implementation worktree; authorized source changes "
                "are expected"
            ),
        },
        "initial_zip_source_sha256sums": {
            **initial_source_archive_check,
            "scope": (
                "immutable initial Git commit imported after "
                "sha256sum -c SHA256SUMS passed"
            ),
        },
    }
    report["environment_exact_match"] = bool(
        all(package_match.values())
        and pcl_pass
        and initial_source_archive_check["pass"] is True
    )
    return report


def build_dry_run_report(runtime_root: Path, workers: int) -> dict[str, Any]:
    """Audit the frozen plan without creating RNGs, snapshots, or backends."""

    from phase_a_harness import synthetic_confirmatory_v3_contract as contract
    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        EXECUTION_CLASSIFICATION,
        load_requalified_plans,
    )

    snapshots, trials = load_requalified_plans(REPOSITORY)
    audit = contract.audit_v3_plan(
        REPOSITORY / "protocols/synthetic_confirmatory_planned_snapshots_v3.csv",
        REPOSITORY / "protocols/synthetic_confirmatory_planned_trials_v3.csv",
    )
    condition_counts = Counter(str(row["condition"]) for row in snapshots)
    backend_counts = Counter(str(row["backend"]) for row in trials)
    report = {
        "schema_version": "synthetic_confirmatory_v3_requalified_dry_run_v1",
        "execution_classification": EXECUTION_CLASSIFICATION,
        "runtime_root": str(runtime_root),
        "workers": workers,
        **audit,
        "condition_snapshot_counts": dict(sorted(condition_counts.items())),
        "backend_trial_counts": dict(sorted(backend_counts.items())),
        "snapshot_construction_count": 0,
        "confirmatory_seed_access_count": 0,
        "confirmatory_rng_instantiation_count": 0,
        "backend_execution_count": 0,
        "trial_result_count": 0,
        "started_event_count": 0,
        "output_dir_created": runtime_root.exists(),
    }
    report["zero_instantiation_pass"] = bool(
        report.get("V3_PLAN_PASS") is True
        and len(snapshots) == 595
        and len(trials) == 1190
        and backend_counts
        == Counter({"open3d_point_to_plane": 595, "pcl_point_to_plane": 595})
        and all(
            report[name] == 0
            for name in (
                "snapshot_construction_count",
                "confirmatory_seed_access_count",
                "confirmatory_rng_instantiation_count",
                "backend_execution_count",
                "trial_result_count",
                "started_event_count",
            )
        )
    )
    return report


def build_preflight_report(runtime_root: Path, workers: int) -> dict[str, Any]:
    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        EXECUTION_CLASSIFICATION,
        EXPECTED_BRANCH,
        EXPECTED_TAG,
        scientific_asset_hashes,
    )

    environment = collect_environment_report()
    assets = scientific_asset_hashes(REPOSITORY)
    dry_run = build_dry_run_report(runtime_root, workers)
    report = {
        "schema_version": "synthetic_confirmatory_v3_requalified_preflight_v1",
        "execution_classification": EXECUTION_CLASSIFICATION,
        "environment_exact_match": environment["environment_exact_match"],
        "scientific_assets_match": assets["all_assets_match"],
        "dry_run_pass": dry_run["zero_instantiation_pass"],
        "pcl_qualification_pass": environment["pcl"]["pcl_qualification_pass"],
        "git_commit": environment["git"]["commit"],
        "git_branch": environment["git"]["branch"],
        "git_tags_at_head": environment["git"]["tags_at_head"],
        "git_worktree_clean": environment["git"]["worktree_clean"],
        "expected_branch": EXPECTED_BRANCH,
        "expected_tag": EXPECTED_TAG,
        "runtime_root": str(runtime_root),
        "runtime_root_exists": runtime_root.exists(),
        "environment": environment,
        "source_assets": assets,
        "dry_run": dry_run,
    }
    report["preflight_pass"] = bool(
        report["environment_exact_match"]
        and report["scientific_assets_match"]
        and report["dry_run_pass"]
        and report["pcl_qualification_pass"]
        and report["git_worktree_clean"]
        and report["git_branch"] == EXPECTED_BRANCH
        and EXPECTED_TAG in report["git_tags_at_head"]
    )
    return report


def _qualification_reports() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        DEFAULT_MANIFEST_RELATIVE,
        EXPECTED_BRANCH,
        EXPECTED_TAG,
        canonical_json_sha256,
        file_sha256,
        scientific_asset_hashes,
        strict_json_object,
    )

    smoke = strict_json_object(QUALIFICATION_ROOT / "smoke_qualification_report.json")
    recovery = strict_json_object(QUALIFICATION_ROOT / "resume_qualification_report.json")
    tests = strict_json_object(QUALIFICATION_ROOT / "test_report.json")
    static = strict_json_object(QUALIFICATION_ROOT / "static_plan_report.json")
    identities = [
        report.get("qualification_identity")
        for report in (smoke, recovery, tests, static)
    ]
    identity = identities[0]
    manifest_path = REPOSITORY / DEFAULT_MANIFEST_RELATIVE
    manifest = strict_json_object(manifest_path)
    current_assets = scientific_asset_hashes(REPOSITORY)
    if (
        smoke.get("smoke_qualification_pass") is not True
        or recovery.get("resume_qualification_pass") is not True
        or tests.get("pytest_pass") is not True
        or static.get("static_plan_pass") is not True
        or type(identity) is not dict
        or any(value != identity for value in identities[1:])
        or identity.get("qualification_identity_pass") is not True
        or identity.get("commit") != _git("rev-parse", "HEAD")
        or identity.get("branch") != EXPECTED_BRANCH
        or identity.get("tag") != EXPECTED_TAG
        or identity.get("tag_object_type") != "tag"
        or identity.get("tag_peeled_commit") != _git("rev-parse", "HEAD")
        or identity.get("manifest_sha256") != file_sha256(manifest_path)
        or identity.get("manifest_payload_sha256")
        != manifest.get("manifest_payload_sha256")
        or identity.get("component_binding_sha256")
        != canonical_json_sha256(manifest.get("formal_lifecycle_components"))
        or identity.get("scientific_assets") != current_assets
        or _git("status", "--porcelain=v1") != ""
    ):
        raise RuntimeError(
            "pre-run qualification reports are stale, unbound, or failed"
        )
    return smoke, recovery, tests


def _pre_execution_evidence(
    *,
    runtime_root: Path,
    environment: Mapping[str, Any],
    source_assets_before: Mapping[str, Any],
    preflight: Mapping[str, Any],
    smoke: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Reuse already committed pre-run evidence across strict resume."""

    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        strict_json_object,
    )

    current = {
        "environment_report.json": dict(environment),
        "source_asset_hashes_before.json": dict(source_assets_before),
        "preflight_report.json": dict(preflight),
        "smoke_qualification_report.json": dict(smoke),
    }
    selected: dict[str, dict[str, Any]] = {}
    for name, value in current.items():
        path = runtime_root / name
        selected[name] = strict_json_object(path) if path.is_file() else value
    if (
        selected["environment_report.json"].get("environment_exact_match")
        is not True
        or selected["preflight_report.json"].get("preflight_pass") is not True
        or selected["smoke_qualification_report.json"].get(
            "smoke_qualification_pass"
        )
        is not True
        or selected["source_asset_hashes_before.json"]
        != dict(source_assets_before)
    ):
        raise RuntimeError("persisted pre-execution evidence failed authentication")
    return selected


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    from phase_a_harness.runtime_lifecycle_io import (
        atomic_create_canonical_json,
        read_canonical_json,
    )

    if path.exists():
        if read_canonical_json(path) != dict(value):
            raise RuntimeError(f"existing evidence differs on resume: {path}")
        return
    atomic_create_canonical_json(path, dict(value))


def _alias_postrun_files(runtime_root: Path) -> None:
    from phase_a_harness.runtime_lifecycle_io import read_canonical_json

    aliases = (
        (runtime_root / "analysis/primary.json", runtime_root / "analysis/primary_analysis.json"),
        (
            runtime_root / "verification/independent.json",
            runtime_root / "verification/independent_verification.json",
        ),
        (
            runtime_root / "verification/difference.json",
            runtime_root / "verification/primary_independent_difference.json",
        ),
    )
    for source, destination in aliases:
        _write_once_json(destination, read_canonical_json(source))


def _completeness_report(completed: Any) -> dict[str, Any]:
    from phase_a_harness.formal_lifecycle_contract import deep_thaw
    from phase_a_harness.runtime_lifecycle_io import read_canonical_json

    spec = completed.spec
    rows = [deep_thaw(row) for row in completed.rows]
    manifest = read_canonical_json(spec.paths.raw_manifest)
    results = manifest.get("results", {})
    trial_ids = [str(row["planned_trial_id"]) for row in rows]
    backend_counts = Counter(str(row["backend"]) for row in rows)
    expected_files = {
        str(entry["path"])
        for entry in results.values()
        if type(entry) is dict and type(entry.get("path")) is str
    }
    actual_files = (
        {path.name for path in spec.paths.raw_results.iterdir() if path.is_file()}
        if spec.paths.raw_results.is_dir()
        else set()
    )
    checksum_mismatch = 0
    from phase_a_harness.synthetic_confirmatory_v3_requalified import file_sha256

    for entry in results.values():
        if type(entry) is not dict or type(entry.get("path")) is not str:
            checksum_mismatch += 1
            continue
        path = spec.paths.raw_results / entry["path"]
        checksum_mismatch += int(
            not path.is_file() or file_sha256(path) != entry.get("sha256")
        )
    report = {
        "schema_version": "synthetic_confirmatory_v3_requalified_completeness_v1",
        "execution_classification": "REQUALIFIED_LOCAL_EXECUTION",
        "completed_snapshot_count": len({str(row["snapshot_id"]) for row in rows}),
        "completed_trial_count": len(rows),
        "raw_manifest_result_count": len(results),
        "open3d_point_to_plane": backend_counts.get("open3d_point_to_plane", 0),
        "pcl_point_to_plane": backend_counts.get("pcl_point_to_plane", 0),
        "native": sum(count for name, count in backend_counts.items() if name not in {"open3d_point_to_plane", "pcl_point_to_plane"}),
        "missing": 1190 - len(set(trial_ids)),
        "duplicate": len(trial_ids) - len(set(trial_ids)),
        "checksum_mismatch": checksum_mismatch,
        "orphan": len(actual_files - expected_files),
    }
    report["completeness_pass"] = bool(
        report["completed_snapshot_count"] == 595
        and report["completed_trial_count"] == 1190
        and report["raw_manifest_result_count"] == 1190
        and report["open3d_point_to_plane"] == 595
        and report["pcl_point_to_plane"] == 595
        and all(report[name] == 0 for name in ("native", "missing", "duplicate", "checksum_mismatch", "orphan"))
    )
    return report


def _backend_outcomes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for backend in ("open3d_point_to_plane", "pcl_point_to_plane"):
        selected = [row for row in rows if row.get("backend") == backend]
        success = sum(
            row.get("solver_failure") is False
            and row.get("finite_output") is True
            for row in selected
        )
        result[backend] = {
            "total": len(selected),
            "success": success,
            "failure": len(selected) - success,
            "solver_failure": sum(row.get("solver_failure") is True for row in selected),
            "non_finite": sum(row.get("finite_output") is False for row in selected),
        }
    return result


def _selected_fields(
    rows: Any, fields: Sequence[str]
) -> list[dict[str, Any]]:
    if type(rows) is not list:
        return []
    return [
        {name: row.get(name) for name in fields}
        for row in rows
        if type(row) is dict
    ]


def _hypothesis_evidence(
    primary: Mapping[str, Any], gate_contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    gates = primary.get("gate_summary", {})
    hypotheses = gate_contract.get("hypotheses", {})
    identities = {
        "H1": ("H1_IDEAL_CONTROL_PASS", "H1_IDEAL_CONTROL"),
        "H2": (
            "H2_LONG_CORRIDOR_SCENE_EFFECT_PASS",
            "H2_LONG_CORRIDOR_SCENE_EFFECT",
        ),
        "H3": (
            "H3_CROSS_BACKEND_SCENE_RANKING_PASS",
            "H3_CROSS_BACKEND_SCENE_RANKING",
        ),
        "H4": (
            "H4_REASSOCIATION_MECHANISM_PASS",
            "H4_REASSOCIATION_MECHANISM",
        ),
        "H5": (
            "H5_FROZEN_MODEL_B_INCREMENTAL_VALUE_PASS",
            "H5_FROZEN_MODEL_B_INCREMENTAL_VALUE",
        ),
        "H6": (
            "H6_FULL_NOISE_SYSTEMATIC_OFFSET_PASS",
            "H6_FULL_NOISE_SYSTEMATIC_OFFSET",
        ),
    }
    statistics: dict[str, Any] = {
        "H1": _selected_fields(
            primary.get("h1_ideal_control"),
            (
                "backend_schema_name",
                "successful_count",
                "solver_failure_count",
                "nonfinite_output_count",
                "translation_q95_m",
                "rotation_q95_rad",
                "gate_pass",
            ),
        ),
        "H2": _selected_fields(
            primary.get("h2_scene_effect"),
            (
                "backend_schema_name",
                "condition",
                "passing_geometry_count",
                "median_geometry_ratio",
                "median_geometry_difference_m",
                "gate_pass",
            ),
        ),
        "H3": _selected_fields(
            primary.get("h3_cross_backend_ranking"),
            ("condition", "spearman_rho", "gate_pass"),
        ),
        "H4": [
            {
                **row,
                "positive_leave_one_geometry_out_count": sum(
                    item.get("direction_positive") is True
                    for item in source.get("leave_one_geometry_out", [])
                    if type(item) is dict
                ),
            }
            for source, row in zip(
                (
                    item
                    for item in primary.get("h4_reassociation", [])
                    if type(item) is dict
                ),
                _selected_fields(
                    primary.get("h4_reassociation"),
                    (
                        "backend_schema_name",
                        "valid_trial_count",
                        "pooled_spearman_rho",
                        "scene_centered_spearman_rho",
                        "gate_pass",
                    ),
                ),
            )
        ],
        "H5": _selected_fields(
            primary.get("h5_frozen_models"),
            (
                "backend_schema_name",
                "evaluated_count",
                "invalid_count",
                "model_a_mae",
                "model_b_mae",
                "mae_b_to_a_ratio",
                "gate_pass",
            ),
        ),
        "H6": {
            "backend_summary": _selected_fields(
                primary.get("h6_systematic_backend"),
                (
                    "backend_schema_name",
                    "passing_geometry_group_count",
                    "median_systematic_fraction",
                    "gate_pass",
                ),
            ),
            "geometry_groups": _selected_fields(
                primary.get("h6_systematic_groups"),
                (
                    "backend_schema_name",
                    "geometry_seed",
                    "effective_replicate_count",
                    "systematic_translation_offset_m",
                    "systematic_fraction",
                    "group_gate_pass",
                ),
            ),
        },
    }
    return {
        label: {
            "pass": gates.get(gate_name) is True,
            "statistics": statistics[label],
            "frozen_thresholds": hypotheses.get(contract_name),
        }
        for label, (gate_name, contract_name) in identities.items()
    }


def _hypothesis_lines(
    evidence: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    lines: list[str] = []
    for label in ("H1", "H2", "H3", "H4", "H5", "H6"):
        item = evidence[label]
        statistics = json.dumps(
            item["statistics"], sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(",", ":"),
        )
        thresholds = json.dumps(
            item["frozen_thresholds"], sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(",", ":"),
        )
        lines.extend(
            (
                f"- {label}: **{'PASS' if item['pass'] is True else 'FAIL'}**",
                f"  - 关键统计量：`{statistics}`",
                f"  - 冻结阈值：`{thresholds}`",
            )
        )
    return lines


def _write_run_summary(
    *, completed: Any, postrun: Any, environment: Mapping[str, Any], completeness: Mapping[str, Any]
) -> None:
    from phase_a_harness.formal_lifecycle_contract import deep_thaw
    from phase_a_harness.runtime_lifecycle_io import atomic_create_bytes

    spec = completed.spec
    rows = [deep_thaw(row) for row in completed.rows]
    primary = deep_thaw(postrun.primary)
    difference = deep_thaw(postrun.difference)
    verification = deep_thaw(postrun.artifact_verification)
    outcomes = _backend_outcomes(rows)
    gates = primary.get("gate_summary", {})
    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        strict_json_object,
    )

    gate_contract = strict_json_object(
        REPOSITORY / "protocols/synthetic_confirmatory_gate_contract_v3.json"
    )
    hypothesis_evidence = _hypothesis_evidence(primary, gate_contract)
    all_hypotheses = all(
        value["pass"] is True for value in hypothesis_evidence.values()
    )
    final_decision = primary.get("final_decision", {})
    paper_authorized = bool(
        type(final_decision) is dict
        and final_decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is True
    )
    measurement_ready = bool(
        completeness.get("completeness_pass") is True
        and difference.get("exact_match_pass") is True
        and verification.get("REQUALIFIED_V3_ARTIFACT_VERIFICATION_PASS") is True
        and all_hypotheses
        and paper_authorized
    )
    artifact_path = Path(deep_thaw(postrun.publication)["publication_path"])
    paths = {
        "runtime_root": str(spec.paths.runtime_root),
        "environment_report": str(spec.paths.runtime_root / "environment_report.json"),
        "source_asset_hashes_before": str(
            spec.paths.runtime_root / "source_asset_hashes_before.json"
        ),
        "source_asset_hashes_after": str(
            spec.paths.runtime_root / "source_asset_hashes_after.json"
        ),
        "preflight_report": str(spec.paths.runtime_root / "preflight_report.json"),
        "smoke_qualification_report": str(
            spec.paths.runtime_root / "smoke_qualification_report.json"
        ),
        "completeness_report": str(
            spec.paths.runtime_root / "completeness_report.json"
        ),
        "raw_results_directory": str(spec.paths.raw_results),
        "snapshot_cache_directory": str(spec.paths.snapshot_cache),
        "primary_analysis": str(spec.paths.analysis / "primary_analysis.json"),
        "independent_verification": str(
            spec.paths.verification / "independent_verification.json"
        ),
        "primary_independent_difference": str(
            spec.paths.verification / "primary_independent_difference.json"
        ),
        "artifact_root": str(artifact_path),
        "artifact_root_files": [
            str(artifact_path / name)
            for name in spec.publisher_inventory.root_files
        ],
        "artifact_tables": [
            str(artifact_path / "tables" / name)
            for name in spec.publisher_inventory.tables
        ],
        "artifact_figures": [
            str(artifact_path / "figures" / name)
            for name in spec.publisher_inventory.figures
        ],
    }
    summary = {
        "schema_version": "synthetic_confirmatory_v3_requalified_run_summary_v1",
        "execution_classification": "REQUALIFIED_LOCAL_EXECUTION",
        "commit": spec.identity_policy.expected_commit,
        "branch": spec.identity_policy.expected_branch,
        "tag": spec.identity_policy.expected_tag,
        "environment_exact_match": environment["environment_exact_match"],
        "completeness": dict(completeness),
        "backend_outcomes": outcomes,
        "gate_summary": gates,
        "hypothesis_results": hypothesis_evidence,
        "primary_independent_difference": difference,
        "artifact_verification": verification,
        "artifact_path": str(artifact_path),
        "analysis_path": str(spec.paths.analysis / "primary_analysis.json"),
        "independent_verification_path": str(
            spec.paths.verification / "independent_verification.json"
        ),
        "difference_path": str(
            spec.paths.verification / "primary_independent_difference.json"
        ),
        "paths": paths,
        "measurement_paper_mainline_authorized": paper_authorized,
        "measurement_paper_writing_ready": measurement_ready,
    }
    _write_once_json(spec.paths.runtime_root / "run_summary.json", summary)
    lines = [
        "# Synthetic Confirmatory v3 本机重资格执行总结",
        "",
        "- 执行分类：`REQUALIFIED_LOCAL_EXECUTION`（非历史归档正式运行）",
        f"- 代码 commit：`{summary['commit']}`",
        f"- 分支：`{summary['branch']}`",
        f"- annotated tag：`{summary['tag']}`",
        f"- 环境与冻结记录完全一致：`{str(summary['environment_exact_match']).lower()}`",
        f"- 完整矩阵：`{completeness['completed_snapshot_count']}/595 snapshots`，`{completeness['completed_trial_count']}/1190 trials`",
        f"- Open3D：成功 {outcomes['open3d_point_to_plane']['success']}，失败 {outcomes['open3d_point_to_plane']['failure']}（其中 solver 失败 {outcomes['open3d_point_to_plane']['solver_failure']}），非有限 {outcomes['open3d_point_to_plane']['non_finite']}",
        f"- PCL：成功 {outcomes['pcl_point_to_plane']['success']}，失败 {outcomes['pcl_point_to_plane']['failure']}（其中 solver 失败 {outcomes['pcl_point_to_plane']['solver_failure']}），非有限 {outcomes['pcl_point_to_plane']['non_finite']}",
        "",
        "## H1–H6",
        "",
        *_hypothesis_lines(hypothesis_evidence),
        "",
        "## 审计与路径",
        "",
        f"- primary/independent leaf differences：`{difference.get('leaf_difference_count')}`",
        f"- primary/independent section differences：`{difference.get('section_difference_count')}`",
        f"- exact match：`{str(difference.get('exact_match_pass')).lower()}`",
        f"- artifact verification：`{str(verification.get('REQUALIFIED_V3_ARTIFACT_VERIFICATION_PASS')).lower()}`",
        f"- 主分析：`{summary['analysis_path']}`",
        f"- 独立复核：`{summary['independent_verification_path']}`",
        f"- 差异审计：`{summary['difference_path']}`",
        f"- 发布物和图表：`{summary['artifact_path']}`",
        "",
        "## 全部结果与图表绝对路径",
        "",
        *(
            f"- `{path}`"
            for key, value in paths.items()
            for path in (value if type(value) is list else [value])
        ),
        "",
        (
            "是否可以进入 Measurement 论文写作阶段："
            f"**{'是' if measurement_ready else '否'}**。"
            "冻结协议保持 `MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false`；"
            "本次仅完成 Synthetic Confirmatory v3，不越权授权论文主线。"
        ),
        "",
    ]
    path = spec.paths.runtime_root / "run_summary.md"
    payload = "\n".join(lines).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("existing run_summary.md differs on resume")
    else:
        atomic_create_bytes(path, payload)


def build_parser() -> argparse.ArgumentParser:
    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        DEFAULT_MANIFEST_RELATIVE,
        DEFAULT_RUNTIME_ROOT,
        RUN_ID,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_RELATIVE))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--invocation-id")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-postrun", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runtime_root = Path(os.path.abspath(args.runtime_root))
    if args.workers != 2:
        raise ValueError("Synthetic Confirmatory v3 requires exactly 2 workers")
    if args.preflight or args.dry_run:
        value = (
            build_preflight_report(runtime_root, args.workers)
            if args.preflight
            else build_dry_run_report(runtime_root, args.workers)
        )
        print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0 if value.get("preflight_pass", value.get("zero_instantiation_pass")) else 1

    from phase_a_harness.formal_lifecycle import (
        execute_formal_lifecycle,
        execute_postrun_pipeline,
    )
    from phase_a_harness.formal_lifecycle_contract import deep_thaw
    from phase_a_harness.synthetic_confirmatory_v3_requalified import (
        EXPECTED_BRANCH,
        EXPECTED_TAG,
        build_requalified_formal_lifecycle_spec,
        scientific_asset_hashes,
    )

    if args.mode == "fresh" and os.path.lexists(runtime_root):
        raise FileExistsError("fresh runtime root must not exist")
    if args.mode == "resume" and not runtime_root.is_dir():
        raise FileNotFoundError("resume runtime root is absent")
    smoke, recovery, test_report = _qualification_reports()
    before = scientific_asset_hashes(REPOSITORY)
    environment = collect_environment_report()
    preflight = build_preflight_report(runtime_root, args.workers)
    if preflight["preflight_pass"] is not True:
        raise RuntimeError("formal preflight failed")
    spec = build_requalified_formal_lifecycle_spec(
        repository=REPOSITORY,
        manifest_path=args.manifest,
        run_id=args.run_id,
        runtime_root=runtime_root,
        workers=args.workers,
        expected_commit=_git("rev-parse", "HEAD"),
        expected_branch=EXPECTED_BRANCH,
        expected_tag=EXPECTED_TAG,
    )
    pre_execution_evidence = _pre_execution_evidence(
        runtime_root=runtime_root,
        environment=environment,
        source_assets_before=before,
        preflight=preflight,
        smoke=smoke,
    )
    completed = execute_formal_lifecycle(
        spec,
        requested_mode=args.mode,
        invocation_id=args.invocation_id or f"{args.mode}-{uuid.uuid4().hex}",
        pre_execution_evidence=pre_execution_evidence,
    )
    after = scientific_asset_hashes(REPOSITORY)
    if before != after or after["all_assets_match"] is not True:
        raise RuntimeError("scientific source assets changed during execution")
    completeness = _completeness_report(completed)
    if completeness["completeness_pass"] is not True:
        raise RuntimeError("completed formal matrix failed completeness audit")
    _write_once_json(runtime_root / "source_asset_hashes_after.json", after)
    _write_once_json(runtime_root / "completeness_report.json", completeness)
    _write_once_json(
        runtime_root / "working_inventory/resume_qualification_report.json", recovery
    )
    _write_once_json(runtime_root / "working_inventory/test_report.json", test_report)
    postrun = execute_postrun_pipeline(spec, completed) if args.run_postrun else None
    if postrun is not None:
        _alias_postrun_files(runtime_root)
        _write_run_summary(
            completed=completed,
            postrun=postrun,
            environment=pre_execution_evidence["environment_report.json"],
            completeness=completeness,
        )
    output = {
        "lifecycle": completed.report(),
        "postrun": None if postrun is None else postrun.report(),
        "completeness": completeness,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
