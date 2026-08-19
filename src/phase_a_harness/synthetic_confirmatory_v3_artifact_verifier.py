"""Strict verifier for the Synthetic Confirmatory v3 pre-run package.

The verifier is deliberately independent of the v3 runner and scientific
implementation.  It accepts exactly the frozen flat 30-file inventory, checks
both inventories byte-for-byte, rejects links and special files, and then
recomputes the pre-run authorization gate from the packaged evidence.

Nothing in this module constructs a random generator, snapshot, or backend
invocation.  Verification is read-only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


V3_NAMESPACE = (
    "zero_perturbation_synthetic_confirmatory_v3_20260730_"
    "runtime_lifecycle_qualified"
)
V3_RUN_ID = "synthetic-confirmatory-v3"
V3_EXPECTED_BRANCH = "feature/zero-perturbation-synthetic-confirmatory-v3-prerun"
V3_EXPECTED_RELEASE_TAG = (
    "archive/zero-perturbation-synthetic-confirmatory-v3-pre-run-pass"
)
V3_FORMAL_RUNTIME_ROOT = Path(
    "/home/lj/ZPRM/zero_perturbation_runtime/confirmatory/synthetic_confirmatory_v3"
)
FROZEN_MODEL_SHA256 = (
    "99806f83d3a0d2393ee7e36e8c06f14a1a8a44fb8d02b074ebc2d9fe750d0872"
)

PAYLOAD_FILES = (
    "runtime_lifecycle_qualification_binding.json",
    "v1_failure_binding.json",
    "v2_failure_binding.json",
    "retired_seed_sets.json",
    "v3_design_audit.json",
    "scientific_core_binding.json",
    "runtime_lifecycle_core_binding.json",
    "runtime_path_contract.json",
    "v3_seed_schedule.json",
    "v3_seed_provenance_audit.json",
    "v3_plan_audit.json",
    "v2_to_v3_scientific_diff.json",
    "frozen_model_binding.json",
    "backend_binding.json",
    "fixture_execution_report.json",
    "fixture_resume_report.json",
    "primary_independent_difference.json",
    "publisher_inventory.json",
    "fixture_artifact_verification.json",
    "v3_dry_run_report.json",
    "test_report.json",
    "implementation_manifest.json",
    "formal_execution_profile.json",
    "formal_run_commands.sh",
    "final_binding_audit.json",
    "final_decision.json",
    "run_manifest.json",
    "pre_run_report.md",
)
ROOT_FILES = PAYLOAD_FILES + ("MANIFEST.csv", "SHA256SUMS")
JSON_FILES = tuple(name for name in PAYLOAD_FILES if name.endswith(".json"))
MANIFEST_HEADER = ("relative_path", "size_bytes", "sha256")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SCIENTIFIC_ZERO_FIELDS = (
    "scene_difference_count",
    "condition_difference_count",
    "planned_snapshot_count_difference",
    "planned_trial_count_difference",
    "backend_algorithm_difference_count",
    "backend_parameter_difference_count",
    "trial_schema_scientific_field_difference_count",
    "translation_metric_difference_count",
    "rotation_metric_difference_count",
    "quantile_method_difference_count",
    "H1_definition_difference_count",
    "H1_threshold_difference_count",
    "H2_definition_difference_count",
    "H2_threshold_difference_count",
    "H3_definition_difference_count",
    "H3_threshold_difference_count",
    "H4_definition_difference_count",
    "H4_threshold_difference_count",
    "H5_definition_difference_count",
    "H5_threshold_difference_count",
    "H6_definition_difference_count",
    "H6_threshold_difference_count",
    "common_association_difference_count",
    "turnover_definition_difference_count",
    "systematic_offset_definition_difference_count",
    "frozen_model_file_difference_count",
    "frozen_model_feature_difference_count",
    "frozen_model_parameter_difference_count",
)
PROVENANCE_ZERO_FIELDS = (
    "V3_SEED_PROVENANCE_COLLISION_COUNT",
    "V3_SEED_COLLISION_WITH_DEVELOPMENT_COUNT",
    "V3_SEED_COLLISION_WITH_V1_COUNT",
    "V3_SEED_COLLISION_WITH_V2_COUNT",
    "V3_SEED_COLLISION_WITH_OLD_TEST_COUNT",
    "V3_UNEXPECTED_SEED_PATH_COUNT",
    "V3_RNG_INSTANTIATION_COUNT",
    "V3_SNAPSHOT_CONSTRUCTION_COUNT",
    "V3_BACKEND_EXECUTION_COUNT",
    "V3_TRIAL_RESULT_COUNT",
    "V3_SCIENTIFIC_RESULT_COUNT",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {token}")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_float=_finite_float,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {token}")
        ),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON root is not an object: {path.name}")
    return value


def _safe_relative_name(value: str) -> bool:
    candidate = Path(value)
    return bool(
        value
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and candidate.as_posix() == value
        and len(candidate.parts) == 1
    )


def _inventory(root: Path) -> tuple[set[str], list[str], list[str]]:
    files: set[str] = set()
    links: list[str] = []
    invalid_types: list[str] = []
    try:
        root_mode = os.lstat(root).st_mode
    except FileNotFoundError:
        return files, links, ["."]
    if stat.S_ISLNK(root_mode):
        links.append(".")
        return files, links, invalid_types
    if not stat.S_ISDIR(root_mode):
        invalid_types.append(".")
        return files, links, invalid_types
    for entry in os.scandir(root):
        name = entry.name
        mode = entry.stat(follow_symlinks=False).st_mode
        if stat.S_ISLNK(mode):
            links.append(name)
        elif stat.S_ISREG(mode):
            files.add(name)
        else:
            invalid_types.append(name)
    return files, sorted(links), sorted(invalid_types)


def _manifest_audit(root: Path) -> dict[str, Any]:
    expected = set(PAYLOAD_FILES)
    path = root / "MANIFEST.csv"
    rows: dict[str, tuple[int, str]] = {}
    duplicate = malformed = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != MANIFEST_HEADER:
                malformed += 1
            for row in reader:
                relative = row.get("relative_path", "")
                size_text = row.get("size_bytes", "")
                digest = row.get("sha256", "")
                if relative in rows:
                    duplicate += 1
                    continue
                try:
                    size = int(size_text)
                except (TypeError, ValueError):
                    malformed += 1
                    continue
                if (
                    not _safe_relative_name(relative)
                    or size < 0
                    or str(size) != size_text
                    or SHA256_RE.fullmatch(digest) is None
                ):
                    malformed += 1
                    continue
                rows[relative] = (size, digest)
    except (OSError, UnicodeError, csv.Error):
        malformed += 1
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    mismatch = []
    for name in sorted(expected & set(rows)):
        candidate = root / name
        if not candidate.is_file():
            mismatch.append(name)
            continue
        size, digest = rows[name]
        if candidate.stat().st_size != size or _file_sha256(candidate) != digest:
            mismatch.append(name)
    canonical_order = list(rows) == sorted(rows)
    return {
        "manifest_entry_count": len(rows),
        "manifest_duplicate_path_count": duplicate,
        "manifest_malformed_row_count": malformed,
        "manifest_missing_files": missing,
        "manifest_unexpected_files": extra,
        "manifest_mismatch_files": mismatch,
        "manifest_canonical_order_pass": canonical_order,
        "manifest_verification_pass": bool(
            len(rows) == 28
            and not duplicate
            and not malformed
            and not missing
            and not extra
            and not mismatch
            and canonical_order
        ),
    }


def _sha256sums_audit(root: Path) -> dict[str, Any]:
    expected = set(PAYLOAD_FILES) | {"MANIFEST.csv"}
    entries: dict[str, str] = {}
    duplicate = malformed = 0
    try:
        lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        lines = []
        malformed += 1
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            malformed += 1
            continue
        digest, relative = line[:64], line[66:]
        if relative in entries:
            duplicate += 1
            continue
        if SHA256_RE.fullmatch(digest) is None or not _safe_relative_name(relative):
            malformed += 1
            continue
        entries[relative] = digest
    missing = sorted(expected - set(entries))
    extra = sorted(set(entries) - expected)
    mismatch = sorted(
        name
        for name in expected & set(entries)
        if not (root / name).is_file()
        or _file_sha256(root / name) != entries[name]
    )
    canonical_order = list(entries) == sorted(entries)
    return {
        "sha256_entry_count": len(entries),
        "duplicate_sha256_path_count": duplicate,
        "malformed_or_unsafe_sha256_count": malformed,
        "sha256_missing_files": missing,
        "sha256_unexpected_files": extra,
        "sha256_mismatch_files": mismatch,
        "sha256_canonical_order_pass": canonical_order,
        "sha256_verification_pass": bool(
            len(entries) == 29
            and not duplicate
            and not malformed
            and not missing
            and not extra
            and not mismatch
            and canonical_order
        ),
    }


def derive_v3_seed(namespace: str, domain: str, index: int) -> int:
    """Recompute a declared v3 seed without constructing an RNG."""

    if domain not in {"geometry", "measurement", "bootstrap"}:
        raise ValueError("unsupported v3 seed domain")
    if type(index) is not int or index < 0:
        raise ValueError("seed index must be a nonnegative integer")
    payload = f"{namespace}|{domain}|{index}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2147483647
    return 1 if value == 0 else value


def _is_zero(record: Mapping[str, Any], name: str) -> bool:
    return type(record.get(name)) is int and record[name] == 0


def _path_contract_pass(record: Mapping[str, Any]) -> bool:
    root = str(V3_FORMAL_RUNTIME_ROOT)
    expected = {
        "runtime_root": root,
        "snapshot_cache_path": f"{root}/snapshot_cache",
        "snapshot_lock_path": f"{root}/snapshot_lock.json",
        "raw_results_path": f"{root}/raw_results",
        "event_log_path": f"{root}/event_logs",
        "analysis_path": f"{root}/analysis",
        "verification_path": f"{root}/verification",
        "artifact_staging_path": f"{root}/artifact_staging",
    }
    return bool(
        record.get("V3_RUNTIME_PATH_POLICY_PASS") is True
        and record.get("V3_FORMAL_RUNTIME_ROOT_NOT_CREATED") is True
        and record.get("all_paths_absolute") is True
        and record.get("all_paths_canonical") is True
        and record.get("all_paths_outside_repository") is True
        and record.get("symlink_component_count") == 0
        and record.get("overlap_count") == 0
        and all(record.get(name) == value for name, value in expected.items())
    )


def _seed_schedule_pass(record: Mapping[str, Any]) -> bool:
    geometry = record.get("geometry_seeds")
    measurement = record.get("measurement_seeds")
    bootstrap = record.get("bootstrap_seed")
    return bool(
        record.get("namespace") == V3_NAMESPACE
        and geometry == [derive_v3_seed(V3_NAMESPACE, "geometry", i) for i in range(5)]
        and measurement
        == [derive_v3_seed(V3_NAMESPACE, "measurement", i) for i in range(3)]
        and bootstrap == derive_v3_seed(V3_NAMESPACE, "bootstrap", 0)
        and len(set([*geometry, *measurement, bootstrap])) == 9
    )


def _suite_pass(record: Mapping[str, Any]) -> bool:
    suites = record.get("suites")
    if type(suites) is not dict:
        return False
    required = {
        "v3_specialized",
        "runtime_lifecycle_specialized",
        "v2_scientific_chain_regression",
        "full_harness",
        "pcl_backend_v3",
    }
    if set(suites) != required:
        return False
    return all(
        type(value) is dict
        and value.get("pass") is True
        and value.get("failure_count") == 0
        and value.get("error_count") == 0
        and value.get("unexpected_skip_count") == 0
        for value in suites.values()
    )


def _semantic_checks(objects: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    get = objects.__getitem__
    runtime = get("runtime_lifecycle_qualification_binding.json")
    v1 = get("v1_failure_binding.json")
    v2 = get("v2_failure_binding.json")
    retired = get("retired_seed_sets.json")
    design = get("v3_design_audit.json")
    science = get("scientific_core_binding.json")
    lifecycle = get("runtime_lifecycle_core_binding.json")
    paths = get("runtime_path_contract.json")
    schedule = get("v3_seed_schedule.json")
    provenance = get("v3_seed_provenance_audit.json")
    plan = get("v3_plan_audit.json")
    difference = get("v2_to_v3_scientific_diff.json")
    model = get("frozen_model_binding.json")
    backend = get("backend_binding.json")
    fixture = get("fixture_execution_report.json")
    resume = get("fixture_resume_report.json")
    primary = get("primary_independent_difference.json")
    publisher = get("publisher_inventory.json")
    fixture_artifact = get("fixture_artifact_verification.json")
    dry = get("v3_dry_run_report.json")
    tests = get("test_report.json")
    implementation = get("implementation_manifest.json")
    profile = get("formal_execution_profile.json")
    final_binding = get("final_binding_audit.json")
    decision = get("final_decision.json")
    run = get("run_manifest.json")
    return {
        "runtime_lifecycle_binding_pass": bool(
            runtime.get("RUNTIME_LIFECYCLE_QUALIFICATION_BINDING_PASS") is True
            and runtime.get("RUNTIME_LIFECYCLE_QUALIFICATION_PASS") is True
        ),
        "failure_history_pass": bool(
            v1.get("V1_FAILURE_HISTORY_PRESERVED") is True
            and v1.get("SYNTHETIC_CONFIRMATORY_V1_PASS") == "NOT_EVALUATED"
            and v2.get("V2_FAILURE_HISTORY_PRESERVED") is True
            and v2.get("SYNTHETIC_CONFIRMATORY_V2_PASS") == "NOT_EVALUATED"
        ),
        "retired_seed_sets_pass": bool(
            retired.get("V1_SEED_SET_REUSE_AUTHORIZED") is False
            and retired.get("V2_SEED_SET_REUSE_AUTHORIZED") is False
            and retired.get("RETIRED_SEED_SETS_PASS") is True
        ),
        "design_pass": design.get("V3_DESIGN_AUDIT_PASS") is True,
        "scientific_core_pass": bool(
            science.get("SCIENTIFIC_CORE_BINDING_PASS") is True
            and _is_zero(science, "SCIENTIFIC_CORE_FILE_CHANGE_COUNT")
        ),
        "runtime_lifecycle_core_pass": bool(
            lifecycle.get("RUNTIME_LIFECYCLE_CORE_BINDING_PASS") is True
            and _is_zero(lifecycle, "RUNTIME_LIFECYCLE_CORE_FILE_CHANGE_COUNT")
        ),
        "runtime_path_contract_pass": _path_contract_pass(paths),
        "seed_schedule_pass": _seed_schedule_pass(schedule),
        "seed_provenance_pass": bool(
            provenance.get("NEW_V3_NAMESPACE_COLLISION") is False
            and all(_is_zero(provenance, name) for name in PROVENANCE_ZERO_FIELDS)
        ),
        "plan_pass": bool(
            plan.get("V3_PLAN_PASS") is True
            and plan.get("planned_snapshot_count") == 595
            and plan.get("unique_snapshot_count") == 595
            and plan.get("planned_trial_count") == 1190
            and plan.get("unique_trial_count") == 1190
            and plan.get("condition_counts")
            == {
                "FULL_NOISE": 525,
                "IDEAL_MATCHED": 35,
                "INDEPENDENT_NOISE_FREE": 35,
            }
            and plan.get("backend_trial_counts")
            == {"Native": 0, "Open3D": 595, "PCL": 595}
            and plan.get("independent_pseudoreplication_count") == 0
        ),
        "scientific_diff_pass": bool(
            difference.get("V2_TO_V3_SCIENTIFIC_DIFF_PASS") is True
            and all(_is_zero(difference, name) for name in SCIENTIFIC_ZERO_FIELDS)
        ),
        "frozen_model_pass": bool(
            model.get("FROZEN_MODEL_SHA_MATCH") is True
            and model.get("sha256") == FROZEN_MODEL_SHA256
        ),
        "backend_binding_pass": backend.get("BACKEND_BINDING_MATCH") is True,
        "fixture_execution_pass": bool(
            fixture.get("V3_EXECUTION_ADAPTER_FIXTURE_PASS") is True
            and fixture.get("V3_RUNTIME_PATH_POLICY_PASS") is True
            and fixture.get("V3_GIT_GATE_FIXTURE_PASS") is True
            and fixture.get("V3_PRIMARY_VERIFIER_FIXTURE_PASS") is True
            and fixture.get("V3_PUBLISHER_FIXTURE_PASS") is True
            and fixture.get("V3_ARTIFACT_VERIFIER_FIXTURE_PASS") is True
            and fixture.get("fixture_snapshot_count") == 3
            and fixture.get("fixture_trial_count") == 6
            and fixture.get("backend_trial_counts")
            == {"open3d_point_to_plane": 3, "pcl_point_to_plane": 3}
            and fixture.get("native_execution_count") == 0
        ),
        "fixture_resume_pass": bool(
            resume.get("FIXTURE_FRESH_RESUME_PASS") is True
            and resume.get("valid_snapshot_reexecution_count") == 0
            and resume.get("valid_trial_reexecution_count") == 0
            and resume.get("snapshot_checksum_change_after_resume") == 0
            and resume.get("trial_checksum_change_after_resume") == 0
            and resume.get("resume_backend_execution_count") == 0
        ),
        "primary_independent_pass": bool(
            primary.get("exact_match_pass") is True
            and primary.get("leaf_difference_count") == 0
            and primary.get("section_difference_count") == 0
        ),
        "publisher_pass": bool(
            publisher.get("V3_PUBLISHER_FIXTURE_PASS") is True
            and publisher.get("publisher_table_count") == 7
            and publisher.get("publisher_figure_count") == 3
            and publisher.get("publisher_root_file_count") == 7
            and publisher.get("missing_count") == 0
            and publisher.get("extra_count") == 0
            and publisher.get("sha256_mismatch_count") == 0
        ),
        "fixture_artifact_pass": bool(
            fixture_artifact.get("V3_ARTIFACT_VERIFIER_FIXTURE_PASS") is True
            and fixture_artifact.get("FIXTURE_ARTIFACT_VERIFICATION_PASS") is True
        ),
        "dry_run_pass": bool(
            dry.get("V3_DRY_RUN_PASS") is True
            and dry.get("V3_FORMAL_RUNTIME_ROOT_NOT_CREATED") is True
            and dry.get("planned_snapshot_count") == 595
            and dry.get("unique_snapshot_count") == 595
            and dry.get("planned_trial_count") == 1190
            and dry.get("unique_trial_count") == 1190
            and dry.get("condition_counts")
            == {
                "FULL_NOISE": 525,
                "IDEAL_MATCHED": 35,
                "INDEPENDENT_NOISE_FREE": 35,
            }
            and dry.get("backend_trial_counts")
            == {"Native": 0, "Open3D": 595, "PCL": 595}
            and all(
                _is_zero(dry, name)
                for name in (
                    "duplicate_snapshot_count",
                    "duplicate_trial_count",
                    "pairing_violation_count",
                    "independent_pseudoreplication_count",
                    "V3_RNG_INSTANTIATION_COUNT",
                    "V3_SNAPSHOT_CONSTRUCTION_COUNT",
                    "V3_BACKEND_EXECUTION_COUNT",
                    "V3_TRIAL_RESULT_COUNT",
                    "V3_STARTED_EVENT_COUNT",
                )
            )
        ),
        "test_suite_pass": bool(
            tests.get("V3_TEST_SUITE_PASS") is True
            and tests.get("source_degen_lio_pytest_executed") is False
            and _suite_pass(tests)
        ),
        "implementation_binding_pass": bool(
            implementation.get("V3_IMPLEMENTATION_BINDING_PASS") is True
            and implementation.get("candidate_commit")
            and implementation.get("candidate_tag")
            and type(implementation.get("files")) is dict
            and bool(implementation["files"])
        ),
        "formal_profile_pass": bool(
            profile.get("FORMAL_EXECUTION_PROFILE_PASS") is True
            and profile.get("run_id") == V3_RUN_ID
            and profile.get("workers") == 2
            and profile.get("runtime_root") == str(V3_FORMAL_RUNTIME_ROOT)
            and profile.get("expected_branch") == V3_EXPECTED_BRANCH
            and profile.get("expected_release_tag") == V3_EXPECTED_RELEASE_TAG
            and profile.get("formal_execution_count") == 0
        ),
        "final_binding_pass": final_binding.get("V3_FINAL_GIT_BINDING_PASS") is True,
        "decision_pass": bool(
            decision.get("SYNTHETIC_CONFIRMATORY_V3_PRE_RUN_QUALIFICATION_PASS")
            is True
            and decision.get("CONFIRMATORY_V3_RUN_AUTHORIZED") is True
            and decision.get("SYNTHETIC_CONFIRMATORY_V3_EXECUTED") is False
            and decision.get("SYNTHETIC_CONFIRMATORY_V3_COMPLETE") is False
            and decision.get("SYNTHETIC_CONFIRMATORY_V3_PASS") == "NOT_EVALUATED"
            and decision.get("REAL_DATA_RUN_AUTHORIZED") is False
            and decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is False
        ),
        "run_manifest_pass": bool(
            run.get("run_id") == V3_RUN_ID
            and run.get("planned_snapshot_count") == 595
            and run.get("planned_trial_count") == 1190
            and run.get("formal_snapshot_construction_count") == 0
            and run.get("formal_backend_execution_count") == 0
            and run.get("formal_trial_result_count") == 0
            and run.get("native_execution_count") == 0
        ),
    }


def verify_synthetic_confirmatory_v3_prerun_artifact(
    path: str | Path,
    *,
    require_formal_runtime_absent: bool = True,
) -> dict[str, Any]:
    """Verify the exact frozen v3 pre-run package without modifying it."""

    lexical_root = Path(os.path.abspath(os.fspath(path)))
    actual, symlinks, invalid_types = _inventory(lexical_root)
    expected = set(ROOT_FILES)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    objects: dict[str, dict[str, Any]] = {}
    invalid_json: list[str] = []
    for name in JSON_FILES:
        if name not in actual:
            continue
        try:
            objects[name] = _strict_json(lexical_root / name)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            invalid_json.append(name)
    semantic_checks: dict[str, bool] = {}
    if not invalid_json and set(objects) == set(JSON_FILES):
        try:
            semantic_checks = _semantic_checks(objects)
        except (KeyError, TypeError, ValueError):
            semantic_checks = {"semantic_evaluation_completed": False}
    semantic_failures = sorted(
        name for name, passed in semantic_checks.items() if passed is not True
    )
    manifest = _manifest_audit(lexical_root)
    checksums = _sha256sums_audit(lexical_root)
    report_text = ""
    commands_text = ""
    text_errors: list[str] = []
    for name in ("pre_run_report.md", "formal_run_commands.sh"):
        if name not in actual:
            continue
        try:
            value = (lexical_root / name).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            text_errors.append(name)
            continue
        if name.endswith(".md"):
            report_text = value
        else:
            commands_text = value
    report_pass = all(
        token in report_text
        for token in (
            "SYNTHETIC_CONFIRMATORY_V3_PRE_RUN_QUALIFICATION_PASS = true",
            "CONFIRMATORY_V3_RUN_AUTHORIZED = true",
            "SYNTHETIC_CONFIRMATORY_V3_EXECUTED = false",
            "SYNTHETIC_CONFIRMATORY_V3_PASS = NOT_EVALUATED",
        )
    )
    commands_pass = all(
        token in commands_text
        for token in (
            V3_RUN_ID,
            V3_EXPECTED_BRANCH,
            V3_EXPECTED_RELEASE_TAG,
            str(V3_FORMAL_RUNTIME_ROOT),
            "--resume",
        )
    )
    live_runtime_absent = not V3_FORMAL_RUNTIME_ROOT.exists()
    result = {
        "schema_version": "synthetic_confirmatory_v3_prerun_artifact_verification_v1",
        "actual_file_count": len(actual),
        "required_file_count": len(ROOT_FILES),
        "missing_required_files": missing,
        "extra_files": extra,
        "symlink_paths": symlinks,
        "invalid_file_type_paths": invalid_types,
        "invalid_json_files": sorted(invalid_json),
        "invalid_text_files": sorted(text_errors),
        "evidence_semantic_checks": semantic_checks,
        "evidence_semantic_failures": semantic_failures,
        "pre_run_report_semantics_pass": report_pass,
        "formal_run_commands_semantics_pass": commands_pass,
        "formal_runtime_root_absent": live_runtime_absent,
        **manifest,
        **checksums,
    }
    result["V3_PRE_RUN_ARTIFACT_VERIFICATION_PASS"] = bool(
        len(actual) == 30
        and not missing
        and not extra
        and not symlinks
        and not invalid_types
        and not invalid_json
        and not text_errors
        and semantic_checks
        and not semantic_failures
        and manifest["manifest_verification_pass"]
        and checksums["sha256_verification_pass"]
        and report_pass
        and commands_pass
        and (live_runtime_absent or not require_formal_runtime_absent)
    )
    return result


verify_v3_prerun_artifact = verify_synthetic_confirmatory_v3_prerun_artifact


__all__ = [
    "FROZEN_MODEL_SHA256",
    "JSON_FILES",
    "MANIFEST_HEADER",
    "PAYLOAD_FILES",
    "ROOT_FILES",
    "SCIENTIFIC_ZERO_FIELDS",
    "V3_EXPECTED_BRANCH",
    "V3_EXPECTED_RELEASE_TAG",
    "V3_FORMAL_RUNTIME_ROOT",
    "V3_NAMESPACE",
    "V3_RUN_ID",
    "derive_v3_seed",
    "verify_synthetic_confirmatory_v3_prerun_artifact",
    "verify_v3_prerun_artifact",
]
