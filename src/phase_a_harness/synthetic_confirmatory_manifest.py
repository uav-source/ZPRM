"""Single formal-manifest contract for Synthetic Confirmatory v1.

The builders in this module only bind already-existing files.  They never
instantiate a random-number generator, create a snapshot, or execute a
registration backend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import canonical_json_sha256, file_sha256, write_json


SCIENTIFIC_SURVIVAL_COMMIT = "ffc15334f4ded25fdba5e709b45657dbad481dfc"
SCIENTIFIC_SURVIVAL_TAG = "archive/zero-perturbation-scientific-survival-audit-v1"
MANIFEST_RELATIVE = Path("frozen_assets/synthetic_confirmatory_formal_manifest_v1.json")
FORMAL_RUN_ID = "synthetic-confirmatory-v1"
FORMAL_OUTPUT_DIR = "results/synthetic_confirmatory_v1"
FORMAL_WORKERS = 2
SNAPSHOT_CACHE_ROOT = "data/synthetic_confirmatory_v1_snapshots"
RAW_RESULT_MANIFEST_SCHEMA = "synthetic_confirmatory_raw_result_manifest_v1"


FILE_BINDINGS: Mapping[str, str] = {
    "gitignore_runtime_outputs": ".gitignore",
    "core_contracts": "src/phase_a_harness/contracts.py",
    "scientific_protocol": "protocols/synthetic_confirmatory_protocol_v1.json",
    "scientific_protocol_document": "protocols/synthetic_confirmatory_protocol_v1.md",
    "gate_contract": "protocols/synthetic_confirmatory_gate_contract.json",
    "planned_snapshots": "protocols/synthetic_confirmatory_planned_snapshots.csv",
    "planned_trials": "protocols/synthetic_confirmatory_planned_trials.csv",
    "seed_provenance_audit": "protocols/confirmatory_seed_provenance_audit.json",
    "frozen_model": "frozen_assets/confirmatory_development_trained_models_v1.json",
    "backend_parameter_contract": "frozen_assets/backend_parameter_contract.json",
    "pcl_cli": "bin/pcl_point_to_plane_cli",
    "trial_schema": "frozen_assets/trial_result_schema.json",
    "generator": (
        "src/phase_a_harness/phase_b_generator_frozen/"
        "capture_range/day2_development_scene.py"
    ),
    "generator_wrapper": "src/phase_a_harness/phase_b_generator.py",
    "generator_development_protocol": "configs/zero_perturbation/development_v1.yaml",
    "generator_frozen_capture_init": (
        "src/phase_a_harness/phase_b_generator_frozen/capture_range/__init__.py"
    ),
    "generator_frozen_capture_protocol": (
        "src/phase_a_harness/phase_b_generator_frozen/capture_range/"
        "day2_development_protocol.py"
    ),
    "generator_frozen_capture_types": (
        "src/phase_a_harness/phase_b_generator_frozen/capture_range/types.py"
    ),
    "generator_frozen_package_init": (
        "src/phase_a_harness/phase_b_generator_frozen/__init__.py"
    ),
    "generator_frozen_zero_init": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/__init__.py"
    ),
    "generator_frozen_zero_protocol": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/protocol.py"
    ),
    "generator_frozen_zero_snapshot_builder": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/"
        "snapshot_builder.py"
    ),
    "generator_frozen_zero_types": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/types.py"
    ),
    "generator_frozen_phase_a_v1_2": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/"
        "backend_phase_a_v1_2.py"
    ),
    "generator_frozen_phase_a_protocol": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/"
        "backend_phase_a_protocol.py"
    ),
    "snapshot_builder": "src/phase_a_harness/synthetic_confirmatory_snapshot_builder.py",
    "full_synthetic_snapshot_builder": (
        "src/phase_a_harness/full_synthetic_snapshot_builder.py"
    ),
    "full_synthetic_development_protocol": (
        "src/phase_a_harness/full_synthetic_development_protocol.py"
    ),
    "full_synthetic_development_runner": (
        "src/phase_a_harness/full_synthetic_development_runner.py"
    ),
    "phase_b_snapshot_assets": "src/phase_a_harness/phase_b_snapshot_assets.py",
    "asset_verifier": "src/phase_a_harness/asset_verifier.py",
    "snapshot_reader": "src/phase_a_harness/snapshot_reader.py",
    "open3d_adapter": "src/phase_a_harness/open3d_backend.py",
    "pcl_adapter": "src/phase_a_harness/pcl_backend.py",
    "backend_metrics": "src/phase_a_harness/metrics.py",
    "backend_types": "src/phase_a_harness/types.py",
    "backend_phase_a_metrics": "src/phase_a_harness/backend_phase_a_metrics.py",
    "rotation_metrics": "src/phase_a_harness/rotation_metrics.py",
    "phase_a_execution_chain": (
        "src/phase_a_harness/phase_a_execution_chain_audit.py"
    ),
    "phase_a_execution_fixture": (
        "src/phase_a_harness/phase_a_execution_chain_fixture.py"
    ),
    "fixture_source_access_monitor": "src/phase_a_harness/runner.py",
    "fixture_qualification_script": "scripts/run_fixture_qualification.py",
    "fixture_qualification": "src/phase_a_harness/fixture_qualification.py",
    "fixture_publication": "src/phase_a_harness/fixture_publication.py",
    "fixture_publication_artifact_verifier": (
        "src/phase_a_harness/fixture_publication_artifact_verifier.py"
    ),
    "fixture_primary_analysis": (
        "src/phase_a_harness/phase_a_stage1_analysis.py"
    ),
    "fixture_independent_verifier": (
        "src/phase_a_harness/phase_a_stage1_independent_verifier.py"
    ),
    "fixture_plan": "frozen_assets/fixtures/fixture_plan.json",
    "fixture_snapshot_lock": (
        "frozen_assets/fixtures/fixture_snapshot_lock.json"
    ),
    "fixture_backend_parameter_lock": (
        "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
    ),
    "full_synthetic_backend_execution": (
        "src/phase_a_harness/full_synthetic_backend_execution.py"
    ),
    "full_synthetic_trial_result": (
        "src/phase_a_harness/full_synthetic_trial_result.py"
    ),
    "phase_b_trial_result": "src/phase_a_harness/phase_b_trial_result.py",
    "phase_a_trial_result_schema": (
        "src/phase_a_harness/phase_a_trial_result_schema.py"
    ),
    "trial_result_writer": "src/phase_a_harness/phase_a_trial_result_writer.py",
    "trial_resume": "src/phase_a_harness/phase_a_trial_resume.py",
    "phase_a_attempt_events": "src/phase_a_harness/phase_a_attempt_events.py",
    "common_association": "src/phase_a_harness/common_association_analysis.py",
    "confirmatory_protocol_builder": (
        "src/phase_a_harness/confirmatory_protocol.py"
    ),
    "scientific_survival_models": (
        "src/phase_a_harness/scientific_survival_models.py"
    ),
    "local_metric_models": "src/phase_a_harness/local_metric_models.py",
    "frozen_model_inference": "src/phase_a_harness/synthetic_confirmatory_models.py",
    "protocol_auditor": "src/phase_a_harness/synthetic_confirmatory_protocol.py",
    "analysis": "src/phase_a_harness/synthetic_confirmatory_analysis.py",
    "independent_verifier": (
        "src/phase_a_harness/synthetic_confirmatory_independent_verifier.py"
    ),
    "publisher": "src/phase_a_harness/synthetic_confirmatory_publisher.py",
    "artifact_verifier": (
        "src/phase_a_harness/synthetic_confirmatory_artifact_verifier.py"
    ),
    "runner": "src/phase_a_harness/synthetic_confirmatory_runner.py",
    "runner_script": "scripts/run_synthetic_confirmatory.py",
    "analysis_script": "scripts/analyze_synthetic_confirmatory.py",
    "independent_verifier_script": "scripts/verify_synthetic_confirmatory.py",
    "publisher_script": "scripts/publish_synthetic_confirmatory.py",
    "manifest_builder": "src/phase_a_harness/synthetic_confirmatory_manifest.py",
    "prerun_qualification": (
        "src/phase_a_harness/synthetic_confirmatory_prerun.py"
    ),
    "prerun_qualification_script": (
        "scripts/qualify_synthetic_confirmatory_prerun.py"
    ),
}


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _serialized_json_sha256(value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bound_files(repository: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, relative in FILE_BINDINGS.items():
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"Confirmatory manifest binding is missing: {relative}")
        result[name] = {"path": relative, "sha256": file_sha256(path)}
    return result


def build_synthetic_confirmatory_manifest_payload(
    root: str | Path, *, authorized: bool
) -> dict[str, Any]:
    """Derive the one exact v1 manifest payload from repository files."""

    if type(authorized) is not bool:
        raise TypeError("authorized must be bool")
    repository = Path(root).resolve()
    protocol = _load_json_object(
        repository / FILE_BINDINGS["scientific_protocol"],
        "Synthetic Confirmatory protocol",
    )
    parameters = _load_json_object(
        repository / FILE_BINDINGS["backend_parameter_contract"],
        "backend parameter contract",
    )
    model = _load_json_object(
        repository / FILE_BINDINGS["frozen_model"],
        "frozen Development models",
    )
    if (
        protocol.get("planned_snapshot_count") != 595
        or protocol.get("planned_trial_count") != 1190
        or protocol.get("bootstrap_seed") != 1083684578
        or protocol.get("native_trial_count") != 0
    ):
        raise ValueError("Synthetic Confirmatory protocol cardinality changed")
    if (
        parameters.get("open3d", {}).get("canonical_sha256")
        != "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413"
        or parameters.get("pcl", {}).get("canonical_sha256")
        != "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd"
        or model.get("model_claim_type") != "POST_REGISTRATION_EXPLANATORY"
        or model.get("alpha") != 1.0
    ):
        raise ValueError("frozen backend/model contract changed")
    payload: dict[str, Any] = {
        "backend_count": 2,
        "bootstrap_seed": 1083684578,
        "bound_files": _bound_files(repository),
        "formal_execution_authorized": authorized,
        "formal_output_dir": FORMAL_OUTPUT_DIR,
        "formal_run_id": FORMAL_RUN_ID,
        "formal_workers": FORMAL_WORKERS,
        "manifest_version": "1",
        "native_trial_count": 0,
        "open3d_parameter_sha256": parameters["open3d"]["canonical_sha256"],
        "open3d_version": parameters["open3d"]["parameters"]["version"],
        "pcl_parameter_sha256": parameters["pcl"]["canonical_sha256"],
        "pcl_version": parameters["pcl"]["parameters"]["version"],
        "planned_snapshot_count": 595,
        "planned_trial_count": 1190,
        "raw_result_manifest_schema": RAW_RESULT_MANIFEST_SCHEMA,
        "scientific_survival_commit": SCIENTIFIC_SURVIVAL_COMMIT,
        "scientific_survival_tag": SCIENTIFIC_SURVIVAL_TAG,
        "snapshot_cache_root": SNAPSHOT_CACHE_ROOT,
    }
    # Flat path/SHA aliases keep the strict runner interface compact while the
    # bound_files object remains the single inventory.
    for name in (
        "scientific_protocol",
        "scientific_protocol_document",
        "gate_contract",
        "planned_snapshots",
        "planned_trials",
        "seed_provenance_audit",
        "frozen_model",
    ):
        payload[f"{name}_path"] = payload["bound_files"][name]["path"]
        payload[f"{name}_sha256"] = payload["bound_files"][name]["sha256"]
    return payload


def signed_synthetic_confirmatory_manifest(
    root: str | Path, *, authorized: bool
) -> dict[str, Any]:
    payload = build_synthetic_confirmatory_manifest_payload(
        root, authorized=authorized
    )
    return {**payload, "manifest_payload_sha256": canonical_json_sha256(payload)}


def create_unauthorized_synthetic_confirmatory_manifest(
    root: str | Path,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    destination = repository / MANIFEST_RELATIVE
    candidates = sorted(
        path.resolve()
        for path in destination.parent.glob("synthetic_confirmatory_formal_manifest*.json")
        if path.is_file()
    )
    if candidates:
        raise FileExistsError(
            f"refusing multiple or replacement Confirmatory manifests: {candidates}"
        )
    value = signed_synthetic_confirmatory_manifest(repository, authorized=False)
    write_json(destination, value)
    return value


def authorize_synthetic_confirmatory_manifest_once(
    root: str | Path, *, qualification_decision: Mapping[str, Any]
) -> dict[str, Any]:
    """Make the sole false→true authorization transition after all gates pass."""

    from .synthetic_confirmatory_prerun import (
        REQUIRED_GATE_NAMES,
        ZERO_EXECUTION_COUNTER_NAMES,
    )

    repository = Path(root).resolve()
    destination = repository / MANIFEST_RELATIVE
    actual = _load_json_object(destination, "Synthetic Confirmatory manifest")
    expected = signed_synthetic_confirmatory_manifest(repository, authorized=False)
    if actual != expected:
        raise ValueError("unauthorized Confirmatory manifest is not the exact payload")
    required_gates = tuple(qualification_decision.get("required_gate_names", ()))
    counters = tuple(qualification_decision.get("zero_execution_counter_names", ()))
    if (
        required_gates != REQUIRED_GATE_NAMES
        or counters != ZERO_EXECUTION_COUNTER_NAMES
        or any(qualification_decision.get(name) is not True for name in required_gates)
        or any(
            type(qualification_decision.get(name)) is not int
            or qualification_decision.get(name) != 0
            for name in counters
        )
        or qualification_decision.get(
            "SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS"
        )
        is not True
        or qualification_decision.get("CONFIRMATORY_RUN_AUTHORIZED") is not True
        or qualification_decision.get("SYNTHETIC_CONFIRMATORY_EXECUTED") is not False
        or qualification_decision.get("SYNTHETIC_CONFIRMATORY_COMPLETE") is not False
        or qualification_decision.get("SYNTHETIC_CONFIRMATORY_PASS")
        != "NOT_EVALUATED"
        or qualification_decision.get("REAL_DATA_RUN_AUTHORIZED") is not False
        or qualification_decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED")
        is not False
    ):
        raise PermissionError("Confirmatory pre-run qualification is incomplete")

    from .synthetic_confirmatory_artifact_verifier import (
        verify_synthetic_confirmatory_prerun_artifact,
    )

    artifact = repository / "artifacts/synthetic_confirmatory_prerun_v1"
    verification = verify_synthetic_confirmatory_prerun_artifact(
        artifact, write_report=False
    )
    if verification.get("CONFIRMATORY_ARTIFACT_VERIFICATION_PASS") is not True:
        raise PermissionError("verified pre-run artifact is required for authorization")
    recorded_decision = _load_json_object(
        artifact / "final_decision.json", "pre-run final decision"
    )
    recorded_verification = _load_json_object(
        artifact / "artifact_verification.json",
        "pre-run artifact verification",
    )
    implementation = _load_json_object(
        artifact / "implementation_manifest.json", "pre-run implementation manifest"
    )
    run_manifest = _load_json_object(
        artifact / "run_manifest.json", "pre-run run manifest"
    )
    expected_authorized = signed_synthetic_confirmatory_manifest(
        repository, authorized=True
    )
    if (
        recorded_decision != dict(qualification_decision)
        or recorded_verification != verification
        or run_manifest.get("final_decision") != recorded_decision
        or implementation.get("formal_execution_authorized") is not True
        or implementation.get("formal_branch")
        != "feature/zero-perturbation-synthetic-confirmatory-prerun"
        or implementation.get("formal_pre_run_tag")
        != "archive/zero-perturbation-synthetic-confirmatory-pre-run-pass"
        or implementation.get("formal_manifest_payload_sha256")
        != expected_authorized["manifest_payload_sha256"]
        or implementation.get("bound_files") != expected_authorized["bound_files"]
        or implementation.get("formal_manifest_file_sha256")
        != _serialized_json_sha256(expected_authorized)
    ):
        raise PermissionError("pre-run artifact is not bound to the authorized manifest")
    value = expected_authorized
    write_json(destination, value)
    return value


def verify_synthetic_confirmatory_manifest(
    root: str | Path, *, require_authorized: bool
) -> dict[str, Any]:
    repository = Path(root).resolve()
    destination = repository / MANIFEST_RELATIVE
    candidates = sorted(
        path.resolve()
        for path in destination.parent.glob("synthetic_confirmatory_formal_manifest*.json")
        if path.is_file()
    )
    if candidates != [destination.resolve()]:
        raise ValueError("exactly one Synthetic Confirmatory manifest is required")
    actual = _load_json_object(destination, "Synthetic Confirmatory manifest")
    authorization = actual.get("formal_execution_authorized")
    if type(authorization) is not bool:
        raise ValueError("formal_execution_authorized must be bool")
    expected = signed_synthetic_confirmatory_manifest(
        repository, authorized=authorization
    )
    if actual != expected:
        raise ValueError("Synthetic Confirmatory manifest differs from exact bindings")
    if require_authorized and authorization is not True:
        raise PermissionError("Synthetic Confirmatory formal run is not authorized")
    return actual


__all__ = [
    "FILE_BINDINGS",
    "FORMAL_OUTPUT_DIR",
    "FORMAL_RUN_ID",
    "FORMAL_WORKERS",
    "MANIFEST_RELATIVE",
    "RAW_RESULT_MANIFEST_SCHEMA",
    "SCIENTIFIC_SURVIVAL_COMMIT",
    "SCIENTIFIC_SURVIVAL_TAG",
    "SNAPSHOT_CACHE_ROOT",
    "authorize_synthetic_confirmatory_manifest_once",
    "build_synthetic_confirmatory_manifest_payload",
    "create_unauthorized_synthetic_confirmatory_manifest",
    "signed_synthetic_confirmatory_manifest",
    "verify_synthetic_confirmatory_manifest",
]
