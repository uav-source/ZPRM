from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from phase_a_harness.synthetic_confirmatory_v3_artifact_verifier import (
    FROZEN_MODEL_SHA256,
    PAYLOAD_FILES,
    PROVENANCE_ZERO_FIELDS,
    ROOT_FILES,
    SCIENTIFIC_ZERO_FIELDS,
    V3_EXPECTED_BRANCH,
    V3_EXPECTED_RELEASE_TAG,
    V3_FORMAL_RUNTIME_ROOT,
    V3_NAMESPACE,
    V3_RUN_ID,
    derive_v3_seed,
    verify_v3_prerun_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan_counts() -> dict[str, Any]:
    return {
        "planned_snapshot_count": 595,
        "unique_snapshot_count": 595,
        "planned_trial_count": 1190,
        "unique_trial_count": 1190,
        "condition_counts": {
            "FULL_NOISE": 525,
            "IDEAL_MATCHED": 35,
            "INDEPENDENT_NOISE_FREE": 35,
        },
        "backend_trial_counts": {"Native": 0, "Open3D": 595, "PCL": 595},
        "duplicate_snapshot_count": 0,
        "duplicate_trial_count": 0,
        "pairing_violation_count": 0,
        "independent_pseudoreplication_count": 0,
    }


def _valid_json_payload() -> dict[str, dict[str, Any]]:
    runtime_root = str(V3_FORMAL_RUNTIME_ROOT)
    plan = _plan_counts()
    suites = {
        name: {
            "pass": True,
            "test_count": 1,
            "passed_count": 1,
            "failure_count": 0,
            "error_count": 0,
            "unexpected_skip_count": 0,
        }
        for name in (
            "v3_specialized",
            "runtime_lifecycle_specialized",
            "v2_scientific_chain_regression",
            "full_harness",
            "pcl_backend_v3",
        )
    }
    return {
        "runtime_lifecycle_qualification_binding.json": {
            "RUNTIME_LIFECYCLE_QUALIFICATION_BINDING_PASS": True,
            "RUNTIME_LIFECYCLE_QUALIFICATION_PASS": True,
        },
        "v1_failure_binding.json": {
            "V1_FAILURE_HISTORY_PRESERVED": True,
            "SYNTHETIC_CONFIRMATORY_V1_PASS": "NOT_EVALUATED",
        },
        "v2_failure_binding.json": {
            "V2_FAILURE_HISTORY_PRESERVED": True,
            "SYNTHETIC_CONFIRMATORY_V2_PASS": "NOT_EVALUATED",
        },
        "retired_seed_sets.json": {
            "V1_SEED_SET_REUSE_AUTHORIZED": False,
            "V2_SEED_SET_REUSE_AUTHORIZED": False,
            "RETIRED_SEED_SETS_PASS": True,
        },
        "v3_design_audit.json": {"V3_DESIGN_AUDIT_PASS": True},
        "scientific_core_binding.json": {
            "SCIENTIFIC_CORE_BINDING_PASS": True,
            "SCIENTIFIC_CORE_FILE_CHANGE_COUNT": 0,
        },
        "runtime_lifecycle_core_binding.json": {
            "RUNTIME_LIFECYCLE_CORE_BINDING_PASS": True,
            "RUNTIME_LIFECYCLE_CORE_FILE_CHANGE_COUNT": 0,
        },
        "runtime_path_contract.json": {
            "V3_RUNTIME_PATH_POLICY_PASS": True,
            "V3_FORMAL_RUNTIME_ROOT_NOT_CREATED": True,
            "all_paths_absolute": True,
            "all_paths_canonical": True,
            "all_paths_outside_repository": True,
            "symlink_component_count": 0,
            "overlap_count": 0,
            "runtime_root": runtime_root,
            "snapshot_cache_path": f"{runtime_root}/snapshot_cache",
            "snapshot_lock_path": f"{runtime_root}/snapshot_lock.json",
            "raw_results_path": f"{runtime_root}/raw_results",
            "event_log_path": f"{runtime_root}/event_logs",
            "analysis_path": f"{runtime_root}/analysis",
            "verification_path": f"{runtime_root}/verification",
            "artifact_staging_path": f"{runtime_root}/artifact_staging",
        },
        "v3_seed_schedule.json": {
            "namespace": V3_NAMESPACE,
            "geometry_seeds": [
                derive_v3_seed(V3_NAMESPACE, "geometry", index)
                for index in range(5)
            ],
            "measurement_seeds": [
                derive_v3_seed(V3_NAMESPACE, "measurement", index)
                for index in range(3)
            ],
            "bootstrap_seed": derive_v3_seed(V3_NAMESPACE, "bootstrap", 0),
        },
        "v3_seed_provenance_audit.json": {
            "NEW_V3_NAMESPACE_COLLISION": False,
            **{name: 0 for name in PROVENANCE_ZERO_FIELDS},
        },
        "v3_plan_audit.json": {"V3_PLAN_PASS": True, **plan},
        "v2_to_v3_scientific_diff.json": {
            "V2_TO_V3_SCIENTIFIC_DIFF_PASS": True,
            **{name: 0 for name in SCIENTIFIC_ZERO_FIELDS},
        },
        "frozen_model_binding.json": {
            "FROZEN_MODEL_SHA_MATCH": True,
            "sha256": FROZEN_MODEL_SHA256,
        },
        "backend_binding.json": {"BACKEND_BINDING_MATCH": True},
        "fixture_execution_report.json": {
            "V3_EXECUTION_ADAPTER_FIXTURE_PASS": True,
            "V3_RUNTIME_PATH_POLICY_PASS": True,
            "V3_GIT_GATE_FIXTURE_PASS": True,
            "V3_PRIMARY_VERIFIER_FIXTURE_PASS": True,
            "V3_PUBLISHER_FIXTURE_PASS": True,
            "V3_ARTIFACT_VERIFIER_FIXTURE_PASS": True,
            "fixture_snapshot_count": 3,
            "fixture_trial_count": 6,
            "backend_trial_counts": {
                "open3d_point_to_plane": 3,
                "pcl_point_to_plane": 3,
            },
            "native_execution_count": 0,
        },
        "fixture_resume_report.json": {
            "FIXTURE_FRESH_RESUME_PASS": True,
            "valid_snapshot_reexecution_count": 0,
            "valid_trial_reexecution_count": 0,
            "snapshot_checksum_change_after_resume": 0,
            "trial_checksum_change_after_resume": 0,
            "resume_backend_execution_count": 0,
        },
        "primary_independent_difference.json": {
            "exact_match_pass": True,
            "leaf_difference_count": 0,
            "section_difference_count": 0,
        },
        "publisher_inventory.json": {
            "V3_PUBLISHER_FIXTURE_PASS": True,
            "publisher_table_count": 7,
            "publisher_figure_count": 3,
            "publisher_root_file_count": 7,
            "missing_count": 0,
            "extra_count": 0,
            "sha256_mismatch_count": 0,
        },
        "fixture_artifact_verification.json": {
            "V3_ARTIFACT_VERIFIER_FIXTURE_PASS": True,
            "FIXTURE_ARTIFACT_VERIFICATION_PASS": True,
        },
        "v3_dry_run_report.json": {
            "V3_DRY_RUN_PASS": True,
            "V3_FORMAL_RUNTIME_ROOT_NOT_CREATED": True,
            **plan,
            "V3_RNG_INSTANTIATION_COUNT": 0,
            "V3_SNAPSHOT_CONSTRUCTION_COUNT": 0,
            "V3_BACKEND_EXECUTION_COUNT": 0,
            "V3_TRIAL_RESULT_COUNT": 0,
            "V3_STARTED_EVENT_COUNT": 0,
        },
        "test_report.json": {
            "V3_TEST_SUITE_PASS": True,
            "source_degen_lio_pytest_executed": False,
            "suites": suites,
        },
        "implementation_manifest.json": {
            "V3_IMPLEMENTATION_BINDING_PASS": True,
            "candidate_commit": "a" * 40,
            "candidate_tag": "archive/candidate",
            "files": {"example.py": "b" * 64},
        },
        "formal_execution_profile.json": {
            "FORMAL_EXECUTION_PROFILE_PASS": True,
            "run_id": V3_RUN_ID,
            "workers": 2,
            "runtime_root": runtime_root,
            "expected_branch": V3_EXPECTED_BRANCH,
            "expected_release_tag": V3_EXPECTED_RELEASE_TAG,
            "formal_execution_count": 0,
        },
        "final_binding_audit.json": {"V3_FINAL_GIT_BINDING_PASS": True},
        "final_decision.json": {
            "SYNTHETIC_CONFIRMATORY_V3_PRE_RUN_QUALIFICATION_PASS": True,
            "CONFIRMATORY_V3_RUN_AUTHORIZED": True,
            "SYNTHETIC_CONFIRMATORY_V3_EXECUTED": False,
            "SYNTHETIC_CONFIRMATORY_V3_COMPLETE": False,
            "SYNTHETIC_CONFIRMATORY_V3_PASS": "NOT_EVALUATED",
            "REAL_DATA_RUN_AUTHORIZED": False,
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        },
        "run_manifest.json": {
            "run_id": V3_RUN_ID,
            "planned_snapshot_count": 595,
            "planned_trial_count": 1190,
            "formal_snapshot_construction_count": 0,
            "formal_backend_execution_count": 0,
            "formal_trial_result_count": 0,
            "native_execution_count": 0,
        },
    }


def _refresh_inventories(root: Path) -> None:
    manifest_rows = []
    for name in sorted(PAYLOAD_FILES):
        path = root / name
        manifest_rows.append((name, path.stat().st_size, _sha256(path)))
    with (root / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("relative_path", "size_bytes", "sha256"))
        writer.writerows(manifest_rows)
    checksum_names = sorted({*PAYLOAD_FILES, "MANIFEST.csv"})
    (root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in checksum_names),
        encoding="utf-8",
    )


def _valid_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "artifact"
    root.mkdir()
    objects = _valid_json_payload()
    for name, value in objects.items():
        (root / name).write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    (root / "formal_run_commands.sh").write_text(
        "\n".join(
            (
                "#!/usr/bin/env bash",
                V3_RUN_ID,
                V3_EXPECTED_BRANCH,
                V3_EXPECTED_RELEASE_TAG,
                str(V3_FORMAL_RUNTIME_ROOT),
                "--resume",
                "",
            )
        ),
        encoding="utf-8",
    )
    (root / "pre_run_report.md").write_text(
        "\n".join(
            (
                "SYNTHETIC_CONFIRMATORY_V3_PRE_RUN_QUALIFICATION_PASS = true",
                "CONFIRMATORY_V3_RUN_AUTHORIZED = true",
                "SYNTHETIC_CONFIRMATORY_V3_EXECUTED = false",
                "SYNTHETIC_CONFIRMATORY_V3_PASS = NOT_EVALUATED",
                "",
            )
        ),
        encoding="utf-8",
    )
    _refresh_inventories(root)
    return root


def test_exact_30_file_v3_prerun_artifact_verifies(tmp_path: Path) -> None:
    root = _valid_artifact(tmp_path)
    report = verify_v3_prerun_artifact(
        root, require_formal_runtime_absent=False
    )
    assert set(path.name for path in root.iterdir()) == set(ROOT_FILES)
    assert report["actual_file_count"] == 30
    assert report["required_file_count"] == 30
    assert report["manifest_entry_count"] == 28
    assert report["sha256_entry_count"] == 29
    assert report["evidence_semantic_failures"] == []
    assert report["V3_PRE_RUN_ARTIFACT_VERIFICATION_PASS"] is True


@pytest.mark.parametrize("mutation", ("missing", "extra", "tamper", "symlink"))
def test_verifier_rejects_inventory_and_byte_tampering(
    tmp_path: Path, mutation: str
) -> None:
    root = _valid_artifact(tmp_path)
    if mutation == "missing":
        (root / "backend_binding.json").unlink()
    elif mutation == "extra":
        (root / "extra.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "tamper":
        (root / "backend_binding.json").write_text(
            '{"BACKEND_BINDING_MATCH":false}\n', encoding="utf-8"
        )
    else:
        commands = root / "formal_run_commands.sh"
        target = tmp_path / "commands-target"
        target.write_text(commands.read_text(encoding="utf-8"), encoding="utf-8")
        commands.unlink()
        commands.symlink_to(target)
    report = verify_v3_prerun_artifact(
        root, require_formal_runtime_absent=False
    )
    assert report["V3_PRE_RUN_ARTIFACT_VERIFICATION_PASS"] is False


def test_semantic_gate_tamper_is_rejected_even_after_rehash(tmp_path: Path) -> None:
    root = _valid_artifact(tmp_path)
    decision_path = root / "final_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["CONFIRMATORY_V3_RUN_AUTHORIZED"] = False
    decision_path.write_text(
        json.dumps(decision, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_inventories(root)
    report = verify_v3_prerun_artifact(
        root, require_formal_runtime_absent=False
    )
    assert report["manifest_verification_pass"] is True
    assert report["sha256_verification_pass"] is True
    assert "decision_pass" in report["evidence_semantic_failures"]
    assert report["V3_PRE_RUN_ARTIFACT_VERIFICATION_PASS"] is False


def test_v3_seed_derivation_is_declaration_only_and_exact() -> None:
    schedule = json.loads(
        (
            ROOT / "frozen_assets/synthetic_confirmatory_v3_seed_schedule.json"
        ).read_text(encoding="utf-8")
    )
    assert schedule["namespace"] == V3_NAMESPACE
    assert [
        derive_v3_seed(V3_NAMESPACE, "geometry", index) for index in range(5)
    ] == schedule["geometry_seeds"]
    assert [
        derive_v3_seed(V3_NAMESPACE, "measurement", index) for index in range(3)
    ] == schedule["measurement_seeds"]
    assert (
        derive_v3_seed(V3_NAMESPACE, "bootstrap", 0)
        == schedule["bootstrap_seed"]
    )


def test_prerun_verifier_and_qualifier_do_not_import_rng_or_snapshot_builders() -> None:
    paths = (
        ROOT / "src/phase_a_harness/synthetic_confirmatory_v3_artifact_verifier.py",
        ROOT / "scripts/qualify_synthetic_confirmatory_v3_prerun.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        assert not any(name in {"random", "numpy"} for name in imported)
        assert not any("snapshot_builder" in name for name in imported)


def test_manifest_and_checksum_contract_has_no_self_reference(tmp_path: Path) -> None:
    root = _valid_artifact(tmp_path)
    manifest_names = {
        row["relative_path"]
        for row in csv.DictReader(
            (root / "MANIFEST.csv").open("r", encoding="utf-8", newline="")
        )
    }
    checksum_names = {
        line[66:]
        for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert manifest_names == set(PAYLOAD_FILES)
    assert checksum_names == set(PAYLOAD_FILES) | {"MANIFEST.csv"}
    assert "SHA256SUMS" not in checksum_names


def test_root_symlink_is_rejected(tmp_path: Path) -> None:
    root = _valid_artifact(tmp_path)
    link = tmp_path / "artifact-link"
    link.symlink_to(root, target_is_directory=True)
    report = verify_v3_prerun_artifact(
        link, require_formal_runtime_absent=False
    )
    assert report["symlink_paths"] == ["."]
    assert report["V3_PRE_RUN_ARTIFACT_VERIFICATION_PASS"] is False


def test_execution_profile_binds_the_single_formal_runtime_contract() -> None:
    profile = json.loads(
        (
            ROOT
            / "frozen_assets/synthetic_confirmatory_v3_execution_profile.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            ROOT
            / "frozen_assets/synthetic_confirmatory_formal_manifest_v3.json"
        ).read_text(encoding="utf-8")
    )
    expected_paths = {
        name: manifest[name]
        for name in (
            "runtime_root",
            "snapshot_cache_path",
            "snapshot_lock_path",
            "raw_results_path",
            "event_log_path",
            "backend_temporary_path",
            "analysis_path",
            "verification_path",
            "publisher_staging_path",
            "artifact_staging_path",
            "temporary_inventory_path",
        )
    }
    assert profile["runtime_paths"] == expected_paths
    assert profile["expected_branch"] == manifest["expected_branch"]
    assert profile["expected_release_tag"] == manifest["expected_release_tag"]
    assert profile["run_id"] == manifest["run_id"]
    assert profile["workers"] == manifest["workers"]
    assert profile["publisher_inventory"] == manifest["publisher_inventory"]
    assert profile["final_decision_schema"] == manifest["final_decision_schema"]
    unsigned = {
        key: value
        for key, value in profile.items()
        if key != "execution_profile_payload_sha256"
    }
    encoded = (
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert profile["execution_profile_payload_sha256"] == hashlib.sha256(
        encoded
    ).hexdigest()
