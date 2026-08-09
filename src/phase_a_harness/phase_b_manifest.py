"""Build and authorize the single frozen Phase B experiment manifest."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .contracts import canonical_json_sha256, file_sha256, load_manifest, write_json


PHASE_A_FORMAL_PASS_COMMIT = "50fd4415deb6af3f3e07d88db5a2f21227d05f69"
PHASE_A_FORMAL_PASS_BUNDLE_SHA256 = (
    "c10c48a4a0bff4d893f95279e8f377b74bc803333e18f86fa424542e166c1f78"
)
SOURCE_COMMIT = "89f46dda68e9ff5c71f078f6d13fc9050d58f0f5"
REQUIRED_PRE_RUN_GATES = (
    "PHASE_A_MINIMAL_TEST_PASS",
    "PHASE_B_MINIMAL_TEST_PASS",
    "GENERATOR_EXPORT_EQUIVALENCE_PASS",
    "GENERATOR_REPRODUCTION_PASS",
    "PHASE_B_SNAPSHOT_VERIFICATION_PASS",
    "PHASE_B_TRIAL_PLAN_PASS",
    "PHASE_B_BACKEND_PAIRING_PASS",
    "PHASE_B_DRY_RUN_PASS",
    "SOURCE_RUNTIME_ISOLATION_PASS",
    "DRY_RUN_ZERO_EXECUTION_PASS",
)

PHASE_A_TEST_REPORT = "artifacts/phase_a_regression_test_results.xml"
PHASE_B_TEST_REPORT = "artifacts/phase_b_test_results.xml"
PRE_RUN_GATE_REPORT = "artifacts/phase_b_pre_run_gate_report.json"


def _sha(root: Path, relative: str) -> str:
    candidate = root / relative
    if not candidate.is_file():
        raise FileNotFoundError(f"Phase B manifest binding is missing: {relative}")
    return file_sha256(candidate)


def build_phase_b_manifest_payload(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    protocol_path = "frozen_assets/phase_b_signal_protocol.json"
    protocol = json.loads((repository / protocol_path).read_text(encoding="utf-8"))
    gates = protocol["gates"]
    payload: dict[str, Any] = {
        "analysis_path": "src/phase_a_harness/phase_b_analysis.py",
        "analysis_sha256": _sha(repository, "src/phase_a_harness/phase_b_analysis.py"),
        "artifact_verifier_path": "src/phase_a_harness/phase_b_artifact_verifier.py",
        "artifact_verifier_sha256": _sha(repository, "src/phase_a_harness/phase_b_artifact_verifier.py"),
        "backend_parameter_contract_path": "frozen_assets/backend_parameter_contract.json",
        "backend_parameter_contract_sha256": _sha(repository, "frozen_assets/backend_parameter_contract.json"),
        "conditions": list(protocol["conditions"]),
        "confirmatory_authorized": False,
        "formal_execution_authorized": False,
        "formal_output_dir": str(protocol["formal_execution"]["output_directory"]),
        "formal_run_id": str(protocol["formal_execution"]["run_id"]),
        "formal_workers": int(protocol["formal_execution"]["workers"]),
        "full_synthetic_development_run_authorized": False,
        "generator_export_manifest_path": "frozen_assets/phase_b_generator_export_manifest.csv",
        "generator_export_manifest_sha256": _sha(repository, "frozen_assets/phase_b_generator_export_manifest.csv"),
        "generator_export_verification_path": "artifacts/phase_b_generator_export_verification.json",
        "generator_export_verification_sha256": _sha(repository, "artifacts/phase_b_generator_export_verification.json"),
        "generator_exported_path": "src/phase_a_harness/phase_b_generator_frozen/capture_range/day2_development_scene.py",
        "generator_reproduction_path": "artifacts/phase_b_generator_reproduction.json",
        "generator_reproduction_sha256": _sha(repository, "artifacts/phase_b_generator_reproduction.json"),
        "generator_source_sha256": "f1632095ab6c433e1e917cf0c9a49551683fce2f106761d41c3c3a5e6517b922",
        "generator_wrapper_path": "src/phase_a_harness/phase_b_generator.py",
        "generator_wrapper_sha256": _sha(repository, "src/phase_a_harness/phase_b_generator.py"),
        "geometry_seeds": list(protocol["geometry_seeds"]),
        "independent_verifier_path": "src/phase_a_harness/phase_b_independent_verifier.py",
        "independent_verifier_sha256": _sha(repository, "src/phase_a_harness/phase_b_independent_verifier.py"),
        "manifest_version": "1",
        "measurement_paper_mainline_authorized": False,
        "measurement_seed": int(protocol["measurement_seed"]),
        "native_trial_count": 0,
        "open3d_adapter_sha256": _sha(repository, "src/phase_a_harness/open3d_backend.py"),
        "open3d_parameter_sha256": "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413",
        "open3d_version": "0.19.0+b012259",
        "pcl_adapter_sha256": _sha(repository, "src/phase_a_harness/pcl_backend.py"),
        "pcl_cli_path": "bin/pcl_point_to_plane_cli",
        "pcl_cli_sha256": _sha(repository, "bin/pcl_point_to_plane_cli"),
        "pcl_parameter_sha256": "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd",
        "pcl_version": "1.15.1",
        "phase_a_formal_pass_bundle_sha256": PHASE_A_FORMAL_PASS_BUNDLE_SHA256,
        "phase_a_formal_pass_commit": PHASE_A_FORMAL_PASS_COMMIT,
        "phase_a_formal_pass_tag": "archive/zero-perturbation-phase-a-formal-pass",
        "phase_a_trial_validator_sha256": _sha(repository, "src/phase_a_harness/phase_a_trial_result_schema.py"),
        "phase_b_backend_bridge_path": "src/phase_a_harness/phase_b_backend_execution.py",
        "phase_b_backend_bridge_sha256": _sha(repository, "src/phase_a_harness/phase_b_backend_execution.py"),
        "phase_b_gate_sha256": canonical_json_sha256(gates),
        "phase_b_trial_bridge_path": "src/phase_a_harness/phase_b_trial_result.py",
        "phase_b_trial_bridge_sha256": _sha(repository, "src/phase_a_harness/phase_b_trial_result.py"),
        "planned_snapshot_count": int(protocol["planned_snapshot_count"]),
        "planned_snapshots_path": "frozen_assets/phase_b_planned_snapshots.csv",
        "planned_snapshots_sha256": _sha(repository, "frozen_assets/phase_b_planned_snapshots.csv"),
        "planned_trial_count": int(protocol["planned_trial_count"]),
        "planned_trials_path": "frozen_assets/phase_b_planned_trials.csv",
        "planned_trials_sha256": _sha(repository, "frozen_assets/phase_b_planned_trials.csv"),
        "publisher_path": "src/phase_a_harness/phase_b_publisher.py",
        "publisher_sha256": _sha(repository, "src/phase_a_harness/phase_b_publisher.py"),
        "quantile_method": "linear",
        "real_data_authorized": False,
        "repeat_index": int(protocol["repeat_index"]),
        "rotation_metric_sha256": _sha(repository, "src/phase_a_harness/rotation_metrics.py"),
        "runner_path": "src/phase_a_harness/phase_b_runner.py",
        "runner_sha256": _sha(repository, "src/phase_a_harness/phase_b_runner.py"),
        "runner_script_path": "scripts/run_phase_b_signal.py",
        "runner_script_sha256": _sha(repository, "scripts/run_phase_b_signal.py"),
        "scenes": list(protocol["scenes"]),
        "scientific_protocol_path": protocol_path,
        "scientific_protocol_sha256": _sha(repository, protocol_path),
        "scientific_protocol_document_path": "docs/phase_b_signal_protocol.md",
        "scientific_protocol_document_sha256": _sha(repository, "docs/phase_b_signal_protocol.md"),
        "snapshot_builder_sha256": "6718fc442439e52327e01622df6356a0167456a6d7d8070f25d6e684ef69b72e",
        "phase_b_snapshot_assets_path": "src/phase_a_harness/phase_b_snapshot_assets.py",
        "phase_b_snapshot_assets_sha256": _sha(repository, "src/phase_a_harness/phase_b_snapshot_assets.py"),
        "snapshot_cache_root": "data/phase_b_signal_snapshots",
        "snapshot_lock_path": "frozen_assets/phase_b_snapshot_lock.json",
        "snapshot_lock_sha256": _sha(repository, "frozen_assets/phase_b_snapshot_lock.json"),
        "source_commit": SOURCE_COMMIT,
        "source_repository": "/home/lj/Degen-LIO",
        "preparation_script_path": "scripts/prepare_phase_b_signal.py",
        "preparation_script_sha256": _sha(repository, "scripts/prepare_phase_b_signal.py"),
        "translation_metric_sha256": _sha(repository, "src/phase_a_harness/backend_phase_a_metrics.py"),
        "trial_schema_path": "frozen_assets/trial_result_schema.json",
        "trial_schema_sha256": _sha(repository, "frozen_assets/trial_result_schema.json"),
    }
    return payload


def create_unauthorized_phase_b_manifest(
    root: str | Path,
    destination: str | Path = "frozen_assets/phase_b_signal_manifest.json",
) -> dict[str, Any]:
    repository = Path(root).resolve()
    target = repository / destination
    if target.exists():
        raise FileExistsError(f"refusing to replace Phase B manifest: {target}")
    value = build_phase_b_manifest_payload(repository)
    value["manifest_payload_sha256"] = canonical_json_sha256(value)
    write_json(target, value)
    return value


def _pytest_counts(path: Path) -> dict[str, int]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"invalid pytest JUnit evidence: {path}") from error
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError(f"pytest JUnit evidence has no testsuite: {path}")
    return {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def derive_phase_b_pre_run_gate_report(root: str | Path) -> dict[str, Any]:
    """Derive every authorization gate from the immutable pre-run evidence."""

    repository = Path(root).resolve()
    manifest_path = repository / "frozen_assets/phase_b_signal_manifest.json"
    _, manifest = load_manifest(manifest_path, require_authorized=False)
    if manifest.get("formal_execution_authorized") is not False:
        raise PermissionError("pre-run gate derivation requires an unauthorized manifest")
    evidence_paths = {
        "phase_a_pytest": PHASE_A_TEST_REPORT,
        "phase_b_pytest": PHASE_B_TEST_REPORT,
        "generator_export": "artifacts/phase_b_generator_export_verification.json",
        "generator_reproduction": "artifacts/phase_b_generator_reproduction.json",
        "snapshot_preparation": "artifacts/phase_b_snapshot_preparation_report.json",
        "dry_run": "artifacts/phase_b_dry_run_report.json",
        "unauthorized_manifest": "frozen_assets/phase_b_signal_manifest.json",
    }
    evidence_sha256 = {
        name: _sha(repository, relative) for name, relative in evidence_paths.items()
    }
    phase_a_tests = _pytest_counts(repository / PHASE_A_TEST_REPORT)
    phase_b_tests = _pytest_counts(repository / PHASE_B_TEST_REPORT)
    exported = json.loads(
        (repository / evidence_paths["generator_export"]).read_text(encoding="utf-8")
    )
    reproduced = json.loads(
        (repository / evidence_paths["generator_reproduction"]).read_text(encoding="utf-8")
    )
    prepared = json.loads(
        (repository / evidence_paths["snapshot_preparation"]).read_text(encoding="utf-8")
    )
    dry_run = json.loads(
        (repository / evidence_paths["dry_run"]).read_text(encoding="utf-8")
    )
    phase_a_test_pass = phase_a_tests == {
        "tests": 40,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    phase_b_test_pass = phase_b_tests == {
        "tests": 25,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    export_pass = bool(
        exported.get("GENERATOR_EXPORT_EQUIVALENCE_PASS") is True
        and exported.get("exported_file_count") == 18
        and exported.get("source_destination_sha_mismatch_count") == 0
        and exported.get("destination_file_sha_mismatch_count") == 0
    )
    reproduction_pass = bool(
        reproduced.get("GENERATOR_REPRODUCTION_PASS") is True
        and reproduced.get("reproduction_snapshot_count") == 7
        and reproduced.get("GENERATOR_REPRODUCTION_CHECKSUM_MISMATCH_COUNT") == 0
        and reproduced.get("firewall_totals", {}).get(
            "CONFIRMATORY_SEED_INSTANTIATION_COUNT"
        )
        == 0
        and reproduced.get("firewall_totals", {}).get(
            "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT"
        )
        == 0
    )
    zero_snapshot_defects = all(
        prepared.get(name) == 0
        for name in (
            "snapshot_missing_count",
            "snapshot_extra_count",
            "snapshot_duplicate_count",
            "snapshot_corrupt_count",
            "snapshot_file_sha_mismatch_count",
            "snapshot_checksum_mismatch_count",
            "metadata_checksum_mismatch_count",
        )
    )
    snapshot_pass = bool(
        prepared.get("PHASE_B_SNAPSHOT_ASSET_PASS") is True
        and prepared.get("actual_snapshot_count") == 42
        and prepared.get("verified_snapshot_count") == 42
        and prepared.get("planned_snapshot_count") == 42
        and zero_snapshot_defects
    )
    trial_plan_pass = bool(
        prepared.get("planned_trial_count") == 84
        and prepared.get("native_trial_count") == 0
        and prepared.get("backend_counts")
        == {
            "open3d_point_to_plane": 42,
            "pcl_iterative_closest_point_with_normals": 42,
        }
    )
    pairing_pass = prepared.get("snapshot_backend_pairing_mismatch_count") == 0
    dry_run_pass = bool(
        dry_run.get("FORMAL_DRY_RUN_PASS") is True
        and dry_run.get("planned_snapshot_count") == 42
        and dry_run.get("planned_trial_count") == 84
        and dry_run.get("open3d_trial_count") == 42
        and dry_run.get("pcl_trial_count") == 42
        and dry_run.get("native_trial_count") == 0
        and dry_run.get("condition_trial_counts")
        == {"FULL_NOISE": 42, "INDEPENDENT_NOISE_FREE": 42}
    )
    source_isolation_pass = bool(
        prepared.get("source_repository_runtime_file_read_count") == 0
        and prepared.get("source_repository_runtime_import_count") == 0
        and dry_run.get("source_repository_runtime_file_read_count") == 0
        and dry_run.get("source_repository_runtime_import_count") == 0
        and prepared.get("confirmatory_seed_instantiation_count") == 0
    )
    dry_zero_pass = all(
        dry_run.get(name) == 0
        for name in (
            "formal_rng_access_count",
            "backend_execution_count",
            "trial_result_count",
            "started_event_count",
            "attempt_started_count",
        )
    )
    report: dict[str, Any] = {
        "DRY_RUN_ZERO_EXECUTION_PASS": dry_zero_pass,
        "GENERATOR_EXPORT_EQUIVALENCE_PASS": export_pass,
        "GENERATOR_REPRODUCTION_PASS": reproduction_pass,
        "PHASE_A_MINIMAL_TEST_PASS": phase_a_test_pass,
        "PHASE_B_BACKEND_PAIRING_PASS": pairing_pass,
        "PHASE_B_DRY_RUN_PASS": dry_run_pass,
        "PHASE_B_MINIMAL_TEST_PASS": phase_b_test_pass,
        "PHASE_B_SNAPSHOT_VERIFICATION_PASS": snapshot_pass,
        "PHASE_B_TRIAL_PLAN_PASS": trial_plan_pass,
        "SOURCE_RUNTIME_ISOLATION_PASS": source_isolation_pass,
        "evidence_paths": evidence_paths,
        "evidence_sha256": evidence_sha256,
        "formal_execution_authorized_before_transition": False,
        "phase_a_pytest_counts": phase_a_tests,
        "phase_b_pytest_counts": phase_b_tests,
        "schema_version": "phase_b_scene_signal_pre_run_gate_v1",
    }
    report["report_payload_sha256"] = canonical_json_sha256(report)
    return report


def create_phase_b_pre_run_gate_report(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    destination = repository / PRE_RUN_GATE_REPORT
    if destination.exists():
        raise FileExistsError(f"refusing to replace Phase B pre-run gate report: {destination}")
    report = derive_phase_b_pre_run_gate_report(repository)
    if any(report.get(name) is not True for name in REQUIRED_PRE_RUN_GATES):
        raise PermissionError("derived Phase B pre-run gates did not all pass")
    write_json(destination, report)
    return report


def authorize_phase_b_manifest_once(
    root: str | Path,
    *,
    manifest_path: str | Path = "frozen_assets/phase_b_signal_manifest.json",
    gate_report_path: str | Path = PRE_RUN_GATE_REPORT,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    target = repository / manifest_path
    _, current = load_manifest(target, require_authorized=False)
    if current.get("formal_execution_authorized") is not False:
        raise PermissionError("Phase B manifest authorization is not a one-way false-to-true transition")
    expected = build_phase_b_manifest_payload(repository)
    if current != {**expected, "manifest_payload_sha256": canonical_json_sha256(expected)}:
        raise ValueError("Phase B manifest changed before authorization")
    gate_path = repository / gate_report_path
    gates = json.loads(gate_path.read_text(encoding="utf-8"))
    derived = derive_phase_b_pre_run_gate_report(repository)
    if gates != derived:
        raise ValueError("stored Phase B pre-run gate report differs from raw evidence")
    stored_payload = gates.get("report_payload_sha256")
    gate_payload = {key: value for key, value in gates.items() if key != "report_payload_sha256"}
    if stored_payload != canonical_json_sha256(gate_payload):
        raise ValueError("Phase B pre-run gate report payload SHA mismatch")
    if any(gates.get(name) is not True for name in REQUIRED_PRE_RUN_GATES):
        raise PermissionError("Phase B pre-run gates are incomplete")
    if gates.get("formal_execution_authorized_before_transition") is not False:
        raise PermissionError("Phase B authorization audit boundary changed")
    authorized = dict(current)
    authorized["formal_execution_authorized"] = True
    authorized.pop("manifest_payload_sha256")
    authorized["manifest_payload_sha256"] = canonical_json_sha256(authorized)
    write_json(target, authorized)
    return authorized


__all__ = [
    "PHASE_A_FORMAL_PASS_BUNDLE_SHA256",
    "PHASE_A_FORMAL_PASS_COMMIT",
    "REQUIRED_PRE_RUN_GATES",
    "authorize_phase_b_manifest_once",
    "build_phase_b_manifest_payload",
    "create_phase_b_pre_run_gate_report",
    "create_unauthorized_phase_b_manifest",
    "derive_phase_b_pre_run_gate_report",
]
