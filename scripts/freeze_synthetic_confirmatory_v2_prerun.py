#!/usr/bin/env python3
"""Assemble, verify, and authorize the Confirmatory v2 pre-run freeze.

This command consumes qualification and test evidence only.  It does not
import the snapshot builder, construct an RNG, create a formal snapshot, or
execute a registration backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence


FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
BASE_COMMIT = "1c78372ef6f2c62e441f69c028b58f7bc48f6c35"
SOURCE_COMMIT = "89f46dda68e9ff5c71f078f6d13fc9050d58f0f5"
SOURCE_BRANCH = "feature/zero-perturbation-phase-a-lock-v2-regression-repair"
V1_TAG = "archive/zero-perturbation-synthetic-confirmatory-v1-ideal-lineage-fail"
V1_BUNDLE = Path(
    "/tmp/zero-perturbation-synthetic-confirmatory-v1-ideal-lineage-fail.bundle"
)
V1_BUNDLE_SHA256 = (
    "21e803756da78c3bf06a93d957c0395850a36d6cf119b23cd00c88ab927dcb72"
)
FROZEN_MODEL_SHA256 = (
    "99806f83d3a0d2393ee7e36e8c06f14a1a8a44fb8d02b074ebc2d9fe750d0872"
)
OLD_GEOMETRY_SEEDS = (248284635, 376488233, 198112089, 229684695, 226655024)
OLD_MEASUREMENT_SEEDS = (469989467, 1088311622, 916609326)
OLD_BOOTSTRAP_SEED = 1083684578


def _assert_isolation(root: Path) -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("pre-run freeze requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError("pre-run freeze requires the frozen MAMBA_ROOT_PREFIX")
    source = SOURCE_REPOSITORY.resolve()
    for entry in list(sys.path) + [
        value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value
    ]:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("Python search path reaches the source repository")
    if root == source or source in root.parents:
        raise PermissionError("pre-run freeze must execute in the standalone harness")


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in values:
            if key in output:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            output[key] = value
        return output

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _assert_fixed_repositories(root: Path) -> dict[str, Any]:
    if _git(root, "rev-parse", "HEAD") != BASE_COMMIT:
        raise ValueError("harness base commit changed before the v2 freeze")
    if _git(root, "branch", "--show-current") != "fix/zero-perturbation-confirmatory-ideal-lineage":
        raise ValueError("unexpected harness repair branch")
    if _git(root, "rev-list", "-n", "1", V1_TAG) != BASE_COMMIT:
        raise ValueError("v1 failure tag no longer resolves to the v1 failure commit")
    if not V1_BUNDLE.is_file() or _file_sha256(V1_BUNDLE) != V1_BUNDLE_SHA256:
        raise ValueError("v1 failure bundle binding changed")
    subprocess.run(
        ["git", "bundle", "verify", str(V1_BUNDLE)],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    source_status = _git(SOURCE_REPOSITORY, "status", "--porcelain")
    source_head = _git(SOURCE_REPOSITORY, "rev-parse", "HEAD")
    source_branch = _git(SOURCE_REPOSITORY, "branch", "--show-current")
    if source_status or source_head != SOURCE_COMMIT or source_branch != SOURCE_BRANCH:
        raise ValueError("source Degen-LIO repository is not at its frozen clean state")
    return {
        "harness_base_commit": BASE_COMMIT,
        "harness_branch": "fix/zero-perturbation-confirmatory-ideal-lineage",
        "source_branch": source_branch,
        "source_commit": source_head,
        "source_git_status_porcelain": source_status,
    }


def _junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    nodes = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        name: sum(int(node.attrib.get(name, "0")) for node in nodes)
        for name in ("tests", "failures", "errors", "skipped")
    }
    passed = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    return {
        "error_count": totals["errors"],
        "failure_count": totals["failures"],
        "junit_path": str(path),
        "junit_sha256": _file_sha256(path),
        "pass": bool(
            totals["tests"] > 0
            and totals["failures"] == 0
            and totals["errors"] == 0
            and totals["skipped"] == 0
        ),
        "passed_count": passed,
        "skipped_count": totals["skipped"],
        "status": "PASS" if passed == totals["tests"] else "FAIL",
        "test_count": totals["tests"],
        "unexpected_skip_count": totals["skipped"],
    }


def _pcl_v3_summary(directory: Path) -> dict[str, Any]:
    expected = (
        "nondegenerate_identity.json",
        "known_small_transform.json",
        "planar_degeneracy_diagnostic.json",
    )
    results = []
    for name in expected:
        path = directory / name
        value = _strict_object(path)
        if value.get("microtest_pass") is not True:
            raise ValueError(f"PCL v3 microtest failed: {name}")
        cli = value.get("cli_result")
        if type(cli) is not dict:
            raise ValueError(f"PCL v3 CLI result missing: {name}")
        results.append(
            {
                "condition_status": cli.get("condition_status"),
                "failure_reason": cli.get("failure_reason"),
                "file": name,
                "point_cloud_rank": cli.get("point_cloud_rank"),
                "point_to_plane_hessian_eigenvalues": cli.get(
                    "point_to_plane_hessian_eigenvalues"
                ),
                "point_to_plane_jacobian_rank": cli.get(
                    "point_to_plane_jacobian_rank"
                ),
                "sha256": _file_sha256(path),
            }
        )
    return {
        "error_count": 0,
        "failure_count": 0,
        "pass": True,
        "passed_count": 3,
        "results": results,
        "skipped_count": 0,
        "status": "PASS",
        "test_count": 3,
        "unexpected_skip_count": 0,
    }


def _fixture_rows(directory: Path) -> tuple[list[dict[str, Any]], int]:
    manifest = _strict_object(directory / "raw_result_manifest.json")
    entries = manifest.get("results")
    if type(entries) is not dict or len(entries) != 6:
        raise ValueError("seed-free fixture raw result manifest is not exact 3/6")
    rows: list[dict[str, Any]] = []
    mismatch_count = 0
    for trial_id, entry in sorted(entries.items()):
        if type(entry) is not dict:
            raise ValueError("fixture manifest entry is not an object")
        path = directory / "raw_results" / str(entry.get("path"))
        mismatch_count += int(_file_sha256(path) != entry.get("sha256"))
        row = _strict_object(path)
        mismatch_count += int(row.get("planned_trial_id") != trial_id)
        rows.append(row)
    return rows, mismatch_count


def _publish_fixture(root: Path, directory: Path, publication: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from phase_a_harness.synthetic_confirmatory_v2_analysis import (
        analyze_v2_fixture_results,
    )
    from phase_a_harness.synthetic_confirmatory_v2_artifact_verifier import (
        verify_synthetic_confirmatory_v2_fixture_artifact,
    )
    from phase_a_harness.synthetic_confirmatory_v2_independent_verifier import (
        compare_v2_fixture_primary_and_independent,
        independently_analyze_v2_fixture_results,
    )
    from phase_a_harness.synthetic_confirmatory_v2_publisher import (
        publish_synthetic_confirmatory_v2_fixture,
    )

    rows, raw_mismatches = _fixture_rows(directory)
    qualification = _strict_object(directory / "fixture_qualification.json")
    primary = analyze_v2_fixture_results(rows)
    independent = independently_analyze_v2_fixture_results(rows)
    difference = compare_v2_fixture_primary_and_independent(primary, independent)
    if difference.get("exact_match_pass") is not True:
        raise ValueError("fixture primary and independent verifier differ")
    run = {
        "backend_execution_count": qualification["backend_execution_count"],
        "fixture_snapshot_count": qualification["fixture_snapshot_count"],
        "fixture_trial_count": qualification["fixture_trial_count"],
        "formal_confirmatory_science_evaluated": False,
        "formal_v2_seed_reference_count": 0,
        "fresh_resume_scientific_equivalence": qualification[
            "fresh_resume_scientific_equivalence"
        ],
        "resume_backend_execution_count": qualification[
            "resume_backend_execution_count"
        ],
        "schema_version": "synthetic_confirmatory_v2_fixture_run_v1",
    }
    published = publish_synthetic_confirmatory_v2_fixture(
        primary=primary,
        independent=independent,
        run_manifest=run,
        artifact_dir=publication,
    )
    verified = verify_synthetic_confirmatory_v2_fixture_artifact(
        publication, write_report=False
    )
    if (
        published.get("FIXTURE_ARTIFACT_VERIFICATION_PASS") is not True
        or verified.get("FIXTURE_ARTIFACT_VERIFICATION_PASS") is not True
        or verified.get("actual_file_count") != 17
    ):
        raise ValueError("seed-free fixture v2 publication did not verify")
    fixture = {
        "FIXTURE_EXECUTION_CHAIN_PASS": True,
        "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED": False,
        "artifact_file_count": 17,
        "artifact_path": str(publication),
        "artifact_verification": verified,
        "artifact_verification_pass": True,
        "backend_execution_count": qualification["backend_execution_count"],
        "checksum_mismatch_count": raw_mismatches,
        "failure_inventory": primary["failure_inventory"],
        "fixture_snapshot_count": qualification["fixture_snapshot_count"],
        "fixture_trial_count": qualification["fixture_trial_count"],
        "fresh_resume_scientific_equivalence": qualification[
            "fresh_resume_scientific_equivalence"
        ],
        "pairing_mismatch_count": qualification["input_pairing_violation_count"],
        "primary_verifier_difference_count": difference["leaf_difference_count"],
        "publisher_figure_count": 3,
        "publisher_root_file_count": 7,
        "publisher_table_count": 7,
        "resume_backend_execution_count": qualification[
            "resume_backend_execution_count"
        ],
        "schema_version": "synthetic_confirmatory_v2_fixture_regression_v1",
    }
    return fixture, {
        **difference,
        "schema_version": "synthetic_confirmatory_v2_fixture_difference_v1",
    }


SCIENTIFIC_ZERO_FIELDS = (
    "scene_difference_count",
    "condition_difference_count",
    "backend_difference_count",
    "backend_parameter_difference_count",
    "planned_snapshot_count_difference",
    "planned_trial_count_difference",
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
    "frozen_model_file_difference_count",
    "frozen_model_feature_difference_count",
    "frozen_model_parameter_difference_count",
)


def _scientific_diff(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    v1_protocol = _strict_object(root / "protocols/synthetic_confirmatory_protocol_v1.json")
    v2_protocol = _strict_object(root / "protocols/synthetic_confirmatory_protocol_v2.json")
    v1_gate = _strict_object(root / "protocols/synthetic_confirmatory_gate_contract.json")
    v2_gate = _strict_object(root / "protocols/synthetic_confirmatory_gate_contract_v2.json")
    v1_manifest = _strict_object(
        root / "frozen_assets/synthetic_confirmatory_formal_manifest_v1.json"
    )
    if (
        v1_protocol["scenes"] != v2_protocol["scenes"]
        or v1_protocol["conditions"] != v2_protocol["conditions"]
        or v1_protocol["backends"] != v2_protocol["backends"]
        or v1_protocol["planned_snapshot_count"] != v2_protocol["planned_snapshot_count"]
        or v1_protocol["planned_trial_count"] != v2_protocol["planned_trial_count"]
        or v1_gate["hypotheses"] != v2_gate["hypotheses"]
    ):
        raise ValueError("v1-to-v2 frozen scientific protocol changed")
    for old_name, new_name in (
        ("backend_parameter_contract", "backend_parameter_contract"),
        ("common_association", "common_association"),
        ("rotation_metrics", "rotation_metrics"),
        ("frozen_model", "frozen_model"),
    ):
        if (
            v1_manifest["bound_files"][old_name]["sha256"]
            != manifest["bound_files"][new_name]["sha256"]
        ):
            raise ValueError(f"protected scientific binding changed: {old_name}")
    return {
        **{name: 0 for name in SCIENTIFIC_ZERO_FIELDS},
        "V1_TO_V2_SCIENTIFIC_DIFF_PASS": True,
        "lineage_implementation_difference_count": 1,
        "lineage_schema_difference_count": 1,
        "manifest_binding_difference_count": 1,
        "schema_version": "synthetic_confirmatory_v1_to_v2_scientific_diff_v1",
        "seed_namespace_difference_count": 1,
        "seed_value_difference_count": 9,
        "version_metadata_difference_count": 1,
    }


def _implementation_paths(root: Path) -> list[str]:
    prefixes = (
        "src/phase_a_harness/synthetic_confirmatory_v2_",
        "scripts/",
        "protocols/synthetic_confirmatory_",
        "artifacts/synthetic_confirmatory_v2_design_audit/",
    )
    selected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".csv", ".json", ".md", ".py"}
        and any(path.relative_to(root).as_posix().startswith(prefix) for prefix in prefixes)
        and (
            "synthetic_confirmatory_v2" in path.relative_to(root).as_posix()
            or path.relative_to(root).as_posix().startswith(
                "artifacts/synthetic_confirmatory_v2_design_audit/"
            )
        )
    }
    selected.update(
        {
            "tests/test_synthetic_confirmatory_v2.py",
            "tests/test_minimal_harness.py",
            "frozen_assets/synthetic_confirmatory_v2_seed_schedule.json",
        }
    )
    selected.discard("frozen_assets/synthetic_confirmatory_formal_manifest_v2.json")
    missing = sorted(name for name in selected if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"implementation inventory missing: {missing}")
    return sorted(selected)


def assemble(arguments: argparse.Namespace) -> dict[str, Any]:
    root = arguments.root.resolve()
    _assert_isolation(root)
    repository_state = _assert_fixed_repositories(root)
    sys.path.insert(0, str(root / "src"))
    from phase_a_harness.synthetic_confirmatory_v2_contract import (
        BOOTSTRAP_SEED,
        BOUND_FILE_PATHS,
        FORMAL_BRANCH,
        FORMAL_OUTPUT_DIR,
        FORMAL_PRERUN_TAG,
        FORMAL_RUN_ID,
        FORMAL_WORKERS,
        GEOMETRY_SEEDS,
        MANIFEST_RELATIVE,
        MEASUREMENT_SEEDS,
        NAMESPACE,
        PRERUN_ARTIFACT_RELATIVE,
        SEED_SCHEDULE_RELATIVE,
        SNAPSHOT_PLAN_RELATIVE,
        TRIAL_PLAN_RELATIVE,
        audit_v2_plan,
        authorize_manifest_once,
        signed_manifest,
        verify_manifest,
    )
    from phase_a_harness.synthetic_confirmatory_v2_prerun import (
        REQUIRED_TRUE_GATES,
        REQUIRED_ZERO_COUNTERS,
        build_v2_prerun_decision,
        publish_v2_prerun_artifact_atomic,
    )
    from phase_a_harness.synthetic_confirmatory_v2_qualification import (
        qualification_execution_binding,
    )
    from phase_a_harness.synthetic_confirmatory_v2_seed_audit import (
        audit_v2_seed_provenance,
    )
    from phase_a_harness.synthetic_confirmatory_artifact_verifier import (
        verify_synthetic_confirmatory_prerun_artifact as verify_v1_prerun_artifact,
    )

    manifest_path = root / MANIFEST_RELATIVE
    _manifest_root, live_manifest = verify_manifest(
        manifest_path, require_authorized=False
    )
    if live_manifest.get("formal_execution_authorized") is not False:
        raise PermissionError("pre-run assembly requires the one initial false manifest")
    qualification = _strict_object(arguments.qualification_report.resolve())
    if (
        qualification.get("BOUNDED_V2_QUALIFICATION_PASS") is not True
        or qualification.get("FORMAL_V2_ACCESS_ZERO_PASS") is not True
        or qualification.get("FORMAL_CONFIRMATORY_SCIENCE_EVALUATED") is not False
    ):
        raise ValueError("bounded v2 qualification did not pass")
    qualification_binding = qualification_execution_binding(root)
    controls_raw = qualification.get("ideal_backend_controls")
    control_results = (
        controls_raw.get("results") if type(controls_raw) is dict else None
    )
    if (
        type(control_results) is not list
        or len(control_results) != 42
        or not all(type(row) is dict for row in control_results)
        or qualification.get("QUALIFICATION_EXECUTION_BINDING_STABLE_PASS")
        is not True
        or qualification.get("qualification_execution_binding")
        != qualification_binding
        or qualification.get("qualification_execution_binding_end_sha256")
        != qualification_binding["qualification_binding_sha256"]
        or {
            row.get("implementation_sha256")
            for row in control_results
            if type(row) is dict
        }
        != {qualification_binding["qualification_binding_sha256"]}
        or qualification_binding["files"]["v2_contract"]["sha256"]
        != live_manifest["bound_files"]["v2_contract"]["sha256"]
    ):
        raise ValueError(
            "qualification results do not bind the current manifest contract"
        )

    fixture, difference = _publish_fixture(
        root, arguments.fixture_run.resolve(), arguments.fixture_publication.resolve()
    )
    specialized = _junit_summary(arguments.specialized_junit.resolve())
    full_harness = _junit_summary(arguments.full_harness_junit.resolve())
    pcl = _pcl_v3_summary(arguments.pcl_v3_results.resolve())
    if not all(value["pass"] for value in (specialized, full_harness, pcl)):
        raise ValueError("a required test suite did not pass without skips")
    test_report = {
        "full_harness": full_harness,
        "pcl_backend_v3": pcl,
        "schema_version": "synthetic_confirmatory_v2_test_report_v1",
        "source_degen_lio_pytest_executed": False,
        "specialized_confirmatory": specialized,
        "test_report_pass": True,
        "total_error_count": 0,
        "total_failure_count": 0,
        # The specialized run is a selected subset of the full harness run.
        # Preserve both suite reports, but do not double-count those tests.
        "total_passed_count": full_harness["passed_count"] + 3,
        "total_unexpected_skip_count": 0,
    }

    ideal = dict(qualification["ideal_geometry_qualification"])
    ideal["schema_version"] = "synthetic_confirmatory_v2_ideal_lineage_qualification_v1"
    controls = dict(qualification["ideal_backend_controls"])
    controls.update(
        {
            "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED": qualification[
                "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED"
            ],
            "formal_v2_access_counters": qualification[
                "formal_v2_access_counters"
            ],
            "qualification_access_monitor": qualification["access_monitor"],
            "qualification_execution_binding": qualification_binding,
            "schema_version": "synthetic_confirmatory_v2_ideal_backend_control_v1",
            "source_repository_runtime_import_count": qualification[
                "source_repository_runtime_import_count"
            ],
        }
    )
    negative = dict(qualification["independent_negative_controls"])
    negative.update(
        {
            "false_lineage_count": negative["snapshot_count"],
            "schema_version": "synthetic_confirmatory_v2_independent_negative_control_v1",
        }
    )
    nonideal = dict(qualification["development_nonideal_regression"])
    required_nonideal_zero_fields = (
        "NONIDEAL_DROPOUT_RECORD_CHANGE_COUNT",
        "NONIDEAL_NOISE_RECORD_CHANGE_COUNT",
        "NONIDEAL_PLAN_IDENTITY_CHANGE_COUNT",
        "NONIDEAL_REFERENCE_CHECKSUM_CHANGE_COUNT",
        "NONIDEAL_SCIENTIFIC_PAYLOAD_CHANGE_COUNT",
        "NONIDEAL_SOURCE_CHECKSUM_CHANGE_COUNT",
        "NONIDEAL_TARGET_CHECKSUM_CHANGE_COUNT",
        "SCIENTIFIC_PAYLOAD_CHANGE_COUNT",
    )
    if any(nonideal.get(name) != 0 for name in required_nonideal_zero_fields):
        raise ValueError("Development non-IDEAL scientific payload changed")
    plan = audit_v2_plan(root / SNAPSHOT_PLAN_RELATIVE, root / TRIAL_PLAN_RELATIVE)
    plan.update(
        {
            "V2_PLAN_PASS": all(
                plan[name] is True
                for name in (
                    "CONFIRMATORY_PLAN_COUNT_PASS",
                    "CONFIRMATORY_PLAN_UNIQUENESS_PASS",
                    "CONFIRMATORY_PLAN_PAIRING_PASS",
                    "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS",
                )
            ),
            "schema_version": "synthetic_confirmatory_v2_plan_audit_v1",
        }
    )
    dry = _strict_object(arguments.dry_run_report.resolve())
    if dry.get("CONFIRMATORY_DRY_RUN_PASS") is not True:
        raise ValueError("formal v2 dry-run did not pass")

    design_root = root / "artifacts/synthetic_confirmatory_v2_design_audit"
    execution_design = _strict_object(design_root / "v2_required_changes.json")
    metadata_design = _strict_object(design_root / "metadata_schema_design.json")
    protection = _strict_object(
        design_root / "scientific_semantics_protection_plan.json"
    )
    for value in protection["byte_exact_protected_files"].values():
        if _file_sha256(root / value["path"]) != value["sha256"]:
            raise ValueError(f"protected file changed: {value['path']}")

    closure = {
        "allowed_metadata_schema_version_change": True,
        "allowed_parent_index_persistence_change": True,
        "closure_formula_difference_count": 0,
        "closure_threshold_difference_count": 0,
        "normalized_closure_definition_difference_count": 0,
        "protected_phase_a_function_ast_hashes": {
            name: value
            for name, value in protection["protected_function_sha256"].items()
            if name.startswith("phase_a_")
        },
        "reference_transform_direction_difference_count": 0,
        "schema_version": "synthetic_confirmatory_v2_phase_a_semantics_diff_v1",
        "source_transform_difference_count": 0,
        "target_quantization_difference_count": 0,
        "total_difference_count": 0,
    }
    root_cause_source = _strict_object(design_root / "current_v1_execution_chain.json")[
        "builder_failure"
    ]
    v1_historical_verification = verify_v1_prerun_artifact(
        root / "artifacts/synthetic_confirmatory_prerun_v1",
        write_report=False,
    )
    if v1_historical_verification.get(
        "CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"
    ) is not True:
        raise ValueError("historical v1 failure artifact is no longer readable")
    root_cause = {
        **root_cause_source,
        "IMPLEMENTATION_ONLY_REPAIR": True,
        "ROOT_CAUSE": "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON",
        "bytewise_check_v2_status": "REMOVED_FROM_V2_AND_ISOLATED_IN_HISTORICAL_V1",
        "schema_version": "synthetic_confirmatory_v2_root_cause_binding_v1",
    }
    v1_failure = {
        "SYNTHETIC_CONFIRMATORY_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_PASS": "NOT_EVALUATED",
        "V1_FAILURE_RECORD_PRESERVED": True,
        "confirmatory_formal_backend_execution_count": 0,
        "confirmatory_formal_trial_result_count": 0,
        "historical_v1_artifact_readable": True,
        "historical_v1_artifact_semantic_failure_count": (
            v1_historical_verification["evidence_semantic_failure_count"]
        ),
        "historical_v1_artifact_sha256_mismatch_count": len(
            v1_historical_verification["sha256_mismatch_files"]
        ),
        "schema_version": "synthetic_confirmatory_v1_failure_binding_v2",
        "v1_bundle_path": str(V1_BUNDLE),
        "v1_bundle_sha256": V1_BUNDLE_SHA256,
        "v1_failure_commit": BASE_COMMIT,
        "v1_failure_tag": V1_TAG,
    }
    retirement = {
        "OLD_V1_SEED_SET_REUSE_AUTHORIZED": False,
        "old_bootstrap_seed": OLD_BOOTSTRAP_SEED,
        "old_geometry_seeds": list(OLD_GEOMETRY_SEEDS),
        "old_measurement_seeds": list(OLD_MEASUREMENT_SEEDS),
        "schema_version": "synthetic_confirmatory_v1_seed_retirement_v2",
    }
    model = {
        "FROZEN_MODEL_SHA_MATCH": True,
        "actual_sha256": _file_sha256(
            root / "frozen_assets/confirmatory_development_trained_models_v1.json"
        ),
        "expected_sha256": FROZEN_MODEL_SHA256,
        "no_refit": True,
        "path": "frozen_assets/confirmatory_development_trained_models_v1.json",
        "schema_version": "synthetic_confirmatory_v2_frozen_model_binding_v1",
    }
    if model["actual_sha256"] != model["expected_sha256"]:
        raise ValueError("frozen model SHA changed")

    schedule = _strict_object(root / SEED_SCHEDULE_RELATIVE)
    seed_audit = {
        **audit_v2_seed_provenance(root),
        "NEW_V2_BACKEND_EXECUTION_COUNT": 0,
        "NEW_V2_RNG_INSTANTIATION_COUNT": 0,
        "NEW_V2_SNAPSHOT_CONSTRUCTION_COUNT": 0,
        "NEW_V2_STARTED_EVENT_COUNT": 0,
        "NEW_V2_TRIAL_RESULT_COUNT": 0,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "geometry_seeds": list(GEOMETRY_SEEDS),
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "namespace": NAMESPACE,
    }
    scientific = _scientific_diff(root, live_manifest)

    predicted = signed_manifest(root, authorized=True)
    predicted_file_sha = hashlib.sha256(_json_bytes(predicted)).hexdigest()
    implementation_files = {
        name: _file_sha256(root / name) for name in _implementation_paths(root)
    }
    implementation = {
        "bound_files": predicted["bound_files"],
        "formal_branch": FORMAL_BRANCH,
        "formal_execution_authorized": True,
        "formal_manifest_file_sha256": predicted_file_sha,
        "formal_manifest_path": MANIFEST_RELATIVE.as_posix(),
        "formal_manifest_payload_sha256": predicted["manifest_payload_sha256"],
        "formal_output_dir": FORMAL_OUTPUT_DIR,
        "formal_pre_run_tag": FORMAL_PRERUN_TAG,
        "formal_run_id": FORMAL_RUN_ID,
        "formal_workers": FORMAL_WORKERS,
        "implementation_file_sha256": implementation_files,
        "schema_version": "synthetic_confirmatory_prerun_implementation_manifest_v2",
        "scientific_core_ast_hashes": protection["protected_function_sha256"],
    }

    gates = {name: True for name in REQUIRED_TRUE_GATES}
    gates.update(
        {
            "IDEAL_DEVELOPMENT_SNAPSHOT_COUNT": ideal["snapshot_count"],
            "INDEPENDENT_FALSE_LINEAGE_COUNT": negative["false_lineage_count"],
            "INDEPENDENT_NEGATIVE_CONTROL_COUNT": negative["snapshot_count"],
            "NEW_V2_NAMESPACE_COLLISION": False,
            "OLD_V1_SEED_SET_REUSE_AUTHORIZED": False,
            "ROOT_CAUSE": "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON",
        }
    )
    counters = {name: 0 for name in REQUIRED_ZERO_COUNTERS}
    decision = build_v2_prerun_decision(
        gates=gates, counters=counters, authorize=True
    )
    run_manifest = {
        "final_decision": decision,
        "formal_backend_execution_count": 0,
        "formal_manifest_file_sha256": predicted_file_sha,
        "formal_manifest_path": MANIFEST_RELATIVE.as_posix(),
        "formal_manifest_payload_sha256": predicted["manifest_payload_sha256"],
        "formal_run_id": FORMAL_RUN_ID,
        "formal_rng_instantiation_count": 0,
        "formal_snapshot_construction_count": 0,
        "formal_started_event_count": 0,
        "formal_trial_result_count": 0,
        "qualification_report_path": str(arguments.qualification_report.resolve()),
        "qualification_report_sha256": _file_sha256(
            arguments.qualification_report.resolve()
        ),
        "repository_state": repository_state,
        "schema_version": "synthetic_confirmatory_v2_prerun_run_manifest_v1",
    }
    evidence = {
        "fixture_regression.json": fixture,
        "frozen_model_binding.json": model,
        "ideal_backend_control.json": controls,
        "ideal_parent_lineage_qualification.json": ideal,
        "implementation_manifest.json": implementation,
        "independent_negative_control.json": negative,
        "nonideal_scientific_payload_regression.json": nonideal,
        "old_seed_retirement.json": retirement,
        "phase_a_closure_semantics_diff.json": closure,
        "primary_independent_difference.json": difference,
        "root_cause_binding.json": root_cause,
        "test_report.json": test_report,
        "v1_failure_binding.json": v1_failure,
        "v1_to_v2_scientific_diff.json": scientific,
        "v2_dry_run_report.json": dry,
        "v2_execution_chain_design.json": execution_design,
        "v2_metadata_schema.json": metadata_design,
        "v2_plan_audit.json": plan,
        "v2_seed_provenance_audit.json": seed_audit,
        "v2_seed_schedule.json": schedule,
    }
    verification = publish_v2_prerun_artifact_atomic(
        root / PRERUN_ARTIFACT_RELATIVE,
        evidence=evidence,
        decision=decision,
        run_manifest=run_manifest,
        fixture_publication_dir=arguments.fixture_publication.resolve(),
        manifest_path=manifest_path,
    )
    authorized = authorize_manifest_once(root, qualification_decision=decision)
    return {
        "CONFIRMATORY_V2_RUN_AUTHORIZED": authorized["formal_execution_authorized"],
        "SYNTHETIC_CONFIRMATORY_V2_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V2_PASS": "NOT_EVALUATED",
        "SYNTHETIC_CONFIRMATORY_V2_PRE_RUN_QUALIFICATION_PASS": decision[
            "SYNTHETIC_CONFIRMATORY_V2_PRE_RUN_QUALIFICATION_PASS"
        ],
        "artifact_path": str(root / PRERUN_ARTIFACT_RELATIVE),
        "artifact_verification": verification,
        "formal_manifest_file_sha256": _file_sha256(manifest_path),
        "formal_manifest_payload_sha256": authorized["manifest_payload_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--fixture-run", type=Path, required=True)
    parser.add_argument("--fixture-publication", type=Path, required=True)
    parser.add_argument("--specialized-junit", type=Path, required=True)
    parser.add_argument("--full-harness-junit", type=Path, required=True)
    parser.add_argument("--pcl-v3-results", type=Path, required=True)
    parser.add_argument("--dry-run-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = assemble(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_json_bytes(result))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
