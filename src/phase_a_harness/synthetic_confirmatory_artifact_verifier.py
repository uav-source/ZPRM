"""Strict pre-run and formal artifact verification for Synthetic Confirmatory."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PRERUN_ROOT_FILES = (
    "protocol_binding.json",
    "gate_contract_audit.json",
    "seed_provenance_audit.json",
    "plan_audit.json",
    "frozen_model_audit.json",
    "frozen_model_prediction_crosscheck.json",
    "fixture_regression_report.json",
    "dry_run_report.json",
    "implementation_manifest.json",
    "test_report.json",
    "artifact_verification.json",
    "final_decision.json",
    "run_manifest.json",
    "pre_run_report.md",
    "SHA256SUMS",
)
PRERUN_SHA_EXCLUDED = frozenset({"SHA256SUMS", "artifact_verification.json"})
PRERUN_FIXTURE_PUBLICATION_DIRECTORY = "fixture_publication"
PRERUN_EVIDENCE_FILES = (
    "protocol_binding.json",
    "gate_contract_audit.json",
    "seed_provenance_audit.json",
    "plan_audit.json",
    "frozen_model_audit.json",
    "frozen_model_prediction_crosscheck.json",
    "fixture_regression_report.json",
    "dry_run_report.json",
    "implementation_manifest.json",
    "test_report.json",
)
PRERUN_REQUIRED_GATE_NAMES = (
    "SCIENTIFIC_SURVIVAL_BINDING_PASS",
    "CONFIRMATORY_PROTOCOL_BINDING_PASS",
    "CONFIRMATORY_GATE_CONTRACT_PASS",
    "CONFIRMATORY_SEED_PROVENANCE_PASS",
    "CONFIRMATORY_PLAN_COUNT_PASS",
    "CONFIRMATORY_PLAN_UNIQUENESS_PASS",
    "CONFIRMATORY_PLAN_PAIRING_PASS",
    "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS",
    "FROZEN_MODEL_FILE_SHA_PASS",
    "FROZEN_MODEL_SCHEMA_PASS",
    "FROZEN_MODEL_FEATURE_ORDER_PASS",
    "FROZEN_MODEL_NO_REFIT_CONTRACT_PASS",
    "FROZEN_MODEL_PREDICTION_CROSSCHECK_PASS",
    "CONFIRMATORY_EXECUTION_CHAIN_FIXTURE_PASS",
    "CONFIRMATORY_DRY_RUN_PASS",
    "CONFIRMATORY_ARTIFACT_VERIFICATION_PASS",
)
PRERUN_ZERO_EXECUTION_COUNTER_NAMES = (
    "CONFIRMATORY_RNG_INSTANTIATION_COUNT",
    "CONFIRMATORY_SNAPSHOT_GENERATION_COUNT",
    "CONFIRMATORY_BACKEND_EXECUTION_COUNT",
    "CONFIRMATORY_TRIAL_RESULT_COUNT",
    "NATIVE_EXECUTION_COUNT",
)

FORMAL_TABLES = (
    "h1_ideal_control.csv", "h2_scene_effect.csv",
    "h3_cross_backend_ranking.csv", "h4_reassociation.csv",
    "h5_frozen_models.csv", "h6_systematic_groups.csv", "gate_summary.csv",
)
FORMAL_FIGURES = ("gate_matrix.png", "scene_effect.png", "model_comparison.png")
FORMAL_ROOT_FILES = (
    "synthetic_confirmatory_report.md", "primary_analysis.json",
    "independent_verification.json", "final_decision.json", "run_manifest.json",
    "SHA256SUMS", "artifact_verification.json",
)
FORMAL_REQUIRED_FILES = tuple(f"tables/{name}" for name in FORMAL_TABLES) + tuple(
    f"figures/{name}" for name in FORMAL_FIGURES
) + FORMAL_ROOT_FILES


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError("JSON root is not an object")
    return value


def _checksums(root: Path, expected: set[str]) -> dict[str, Any]:
    path = root / "SHA256SUMS"
    entries: dict[str, str] = {}
    malformed = duplicate = unsafe = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("  ", 1)
            if len(parts) != 2:
                malformed += 1
                continue
            digest, relative = parts
            candidate = Path(relative)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest) or candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
                malformed += 1
                continue
            duplicate += relative in entries
            entries[relative] = digest
    listed = set(entries)
    missing = sorted(expected - listed)
    unexpected = sorted(listed - expected)
    mismatch = sorted(
        name for name, digest in entries.items()
        if name in expected and (not (root / name).is_file() or _sha(root / name) != digest)
    )
    return {
        "sha256_entry_count": len(entries), "sha256_missing_files": missing,
        "sha256_unexpected_files": unexpected, "sha256_mismatch_files": mismatch,
        "duplicate_sha256_path_count": duplicate, "malformed_or_unsafe_sha256_count": malformed,
        "pass": not missing and not unexpected and not mismatch and not duplicate and not malformed,
    }


def _is_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _all_true_mapping(value: Any, *, exact_count: int | None = None) -> bool:
    return bool(
        type(value) is dict
        and value
        and (exact_count is None or len(value) == exact_count)
        and all(item is True for item in value.values())
    )


def _fixture_publication_crosscheck(
    root: Path, fixture_report: dict[str, Any]
) -> dict[str, Any]:
    """Independently verify and bind the embedded seed-free fixture evidence."""

    from .fixture_publication_artifact_verifier import (
        verify_fixture_publication_artifact,
    )

    publication = root / PRERUN_FIXTURE_PUBLICATION_DIRECTORY
    try:
        live = verify_fixture_publication_artifact(publication, write_report=False)
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError):
        live = {"FIXTURE_ARTIFACT_VERIFICATION_PASS": False}
    inventory = [
        {
            "path": candidate.relative_to(publication).as_posix(),
            "sha256": _sha(candidate),
        }
        for candidate in sorted(
            (path for path in publication.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(publication).as_posix(),
        )
    ] if publication.is_dir() else []
    canonical_inventory = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    binding = fixture_report.get("fixture_publication_binding")
    expected = {
        "artifact_relative_path": PRERUN_FIXTURE_PUBLICATION_DIRECTORY,
        "artifact_verification_file_sha256": (
            _sha(publication / "artifact_verification.json")
            if (publication / "artifact_verification.json").is_file()
            else None
        ),
        "file_count": len(inventory),
        "file_inventory": inventory,
        "file_inventory_sha256": hashlib.sha256(canonical_inventory).hexdigest(),
        "live_verification": live,
        "schema_version": "synthetic_confirmatory_fixture_publication_binding_v1",
        "sha256sums_file_sha256": (
            _sha(publication / "SHA256SUMS")
            if (publication / "SHA256SUMS").is_file()
            else None
        ),
    }
    live_pass = bool(
        live.get("FIXTURE_ARTIFACT_VERIFICATION_PASS") is True
        and live.get("recorded_verification_match_pass") is True
        and live.get("artifact_inventory_pass") is True
        and live.get("sha256_validation_pass") is True
        and _is_int(live.get("sha256_mismatch_count"), 0)
        and _is_int(live.get("semantic_error_count"), 0)
        and _is_int(live.get("analysis_verifier_difference_count"), 0)
    )
    binding_pass = type(binding) is dict and binding == expected
    return {
        "fixture_publication_binding_pass": binding_pass,
        "fixture_publication_crosscheck_pass": bool(binding_pass and live_pass),
        "fixture_publication_file_count": len(inventory),
        "fixture_publication_live_verification": live,
        "fixture_publication_live_verification_pass": live_pass,
    }


def _prerun_semantic_checks(
    objects: dict[str, dict[str, Any]],
) -> tuple[dict[str, bool], list[str]]:
    """Cross-check every pre-run evidence object against the frozen decision.

    SHA verification proves that files did not change after publication.  These
    checks separately prove that the ten evidence files are non-empty and that
    their gates, cardinalities, and zero-execution statements agree with each
    other, ``final_decision.json``, and ``run_manifest.json``.
    """

    decision = objects.get("final_decision.json", {})
    run = objects.get("run_manifest.json", {})
    protocol = objects.get("protocol_binding.json", {})
    gate = objects.get("gate_contract_audit.json", {})
    seed = objects.get("seed_provenance_audit.json", {})
    plan = objects.get("plan_audit.json", {})
    model = objects.get("frozen_model_audit.json", {})
    prediction = objects.get("frozen_model_prediction_crosscheck.json", {})
    fixture = objects.get("fixture_regression_report.json", {})
    dry = objects.get("dry_run_report.json", {})
    implementation = objects.get("implementation_manifest.json", {})
    test = objects.get("test_report.json", {})

    checks: dict[str, bool] = {}

    def record(name: str, value: Any) -> None:
        checks[name] = value is True

    for name in PRERUN_EVIDENCE_FILES:
        record(f"{name}:nonempty_object", bool(objects.get(name)))

    # Protocol/survival binding: the top-level gates are copied into the final
    # decision, while the subordinate hashes/checks demonstrate why they pass.
    for name in (
        "SCIENTIFIC_SURVIVAL_BINDING_PASS",
        "CONFIRMATORY_PROTOCOL_BINDING_PASS",
    ):
        record(
            f"protocol_binding.json:{name}",
            protocol.get(name) is True and decision.get(name) is protocol.get(name),
        )
    record(
        "protocol_binding.json:overall_contract",
        protocol.get("SYNTHETIC_CONFIRMATORY_PROTOCOL_CONTRACT_AUDIT_PASS") is True,
    )
    record("protocol_binding.json:file_sha", protocol.get("file_sha256_pass") is True)
    record(
        "protocol_binding.json:file_sha_inventory",
        type(protocol.get("expected_file_sha256")) is dict
        and bool(protocol["expected_file_sha256"])
        and protocol.get("file_sha256") == protocol["expected_file_sha256"]
        and all(_is_sha256(value) for value in protocol["expected_file_sha256"].values()),
    )
    record(
        "protocol_binding.json:binding_checks",
        _all_true_mapping(protocol.get("protocol_binding_checks")),
    )
    record(
        "protocol_binding.json:protocol_payload",
        _is_sha256(protocol.get("protocol_payload_sha256_computed"))
        and protocol.get("protocol_payload_sha256_computed")
        == protocol.get("protocol_payload_sha256_recorded"),
    )
    record(
        "protocol_binding.json:survival_file_binding",
        protocol.get("scientific_survival_file_binding_pass") is True,
    )
    record(
        "protocol_binding.json:survival_identity",
        _is_git_commit(protocol.get("scientific_survival_commit"))
        and isinstance(protocol.get("scientific_survival_tag"), str)
        and bool(protocol.get("scientific_survival_tag")),
    )

    record(
        "gate_contract_audit.json:gate",
        gate.get("CONFIRMATORY_GATE_CONTRACT_PASS") is True
        and decision.get("CONFIRMATORY_GATE_CONTRACT_PASS")
        is gate.get("CONFIRMATORY_GATE_CONTRACT_PASS"),
    )
    record(
        "gate_contract_audit.json:hypotheses",
        gate.get("exact_hypothesis_set") is True
        and _is_int(gate.get("hypothesis_count"), 6)
        and _all_true_mapping(gate.get("hypothesis_results"), exact_count=6),
    )
    record(
        "gate_contract_audit.json:payload",
        _is_sha256(gate.get("computed_gate_contract_payload_sha256"))
        and gate.get("computed_gate_contract_payload_sha256")
        == gate.get("recorded_gate_contract_payload_sha256"),
    )

    record(
        "seed_provenance_audit.json:gate",
        seed.get("CONFIRMATORY_SEED_PROVENANCE_PASS") is True
        and decision.get("CONFIRMATORY_SEED_PROVENANCE_PASS")
        is seed.get("CONFIRMATORY_SEED_PROVENANCE_PASS"),
    )
    record(
        "seed_provenance_audit.json:zero_seed_use",
        all(
            _is_int(seed.get(name), 0)
            for name in (
                "CONFIRMATORY_RNG_INSTANTIATION_COUNT",
                "CONFIRMATORY_SEED_INSTANTIATION_COUNT",
                "CONFIRMATORY_SEED_PARSE_ERROR_COUNT",
                "CONFIRMATORY_SEED_USAGE_HIT_COUNT",
            )
        )
        and seed.get("structured_usage_hits") == []
        and seed.get("stored_provenance_pass") is True
        and seed.get("CONFIRMATORY_RNG_INSTANTIATION_COUNT")
        == decision.get("CONFIRMATORY_RNG_INSTANTIATION_COUNT"),
    )
    record(
        "seed_provenance_audit.json:scan_inventory",
        type(seed.get("structured_declaration_mention_count")) is int
        and seed["structured_declaration_mention_count"] >= 0
        and type(seed.get("structured_file_count")) is int
        and seed["structured_file_count"] > 0,
    )

    for name in (
        "CONFIRMATORY_PLAN_COUNT_PASS",
        "CONFIRMATORY_PLAN_UNIQUENESS_PASS",
        "CONFIRMATORY_PLAN_PAIRING_PASS",
        "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS",
    ):
        record(
            f"plan_audit.json:{name}",
            plan.get(name) is True and decision.get(name) is plan.get(name),
        )
    record(
        "plan_audit.json:cardinality",
        _is_int(plan.get("planned_snapshot_count"), 595)
        and _is_int(plan.get("planned_snapshot_unique_count"), 595)
        and _is_int(plan.get("planned_trial_count"), 1190)
        and _is_int(plan.get("planned_trial_unique_count"), 1190)
        and plan.get("condition_snapshot_counts")
        == {
            "FULL_NOISE": 525,
            "IDEAL_MATCHED": 35,
            "INDEPENDENT_NOISE_FREE": 35,
        }
        and plan.get("backend_trial_counts")
        == {"open3d_point_to_plane": 595, "pcl_point_to_plane": 595}
        and _is_int(plan.get("full_noise_replicates_per_scene_geometry"), 15),
    )
    record(
        "plan_audit.json:zero_violations",
        all(
            _is_int(plan.get(name), 0)
            for name in (
                "duplicate_snapshot_count",
                "duplicate_trial_count",
                "independent_pseudoreplication_plan_count",
                "native_trial_count",
                "pairing_violation_count",
                "semantic_violation_count",
                "snapshot_id_formula_mismatch_count",
                "trial_id_formula_mismatch_count",
                "trial_metadata_mismatch_count",
            )
        ),
    )
    record(
        "plan_audit.json:identity_sha",
        _is_sha256(plan.get("planned_snapshot_identity_sha256"))
        and _is_sha256(plan.get("planned_trial_identity_sha256")),
    )

    for name in (
        "FROZEN_MODEL_FILE_SHA_PASS",
        "FROZEN_MODEL_SCHEMA_PASS",
        "FROZEN_MODEL_FEATURE_ORDER_PASS",
        "FROZEN_MODEL_NO_REFIT_CONTRACT_PASS",
    ):
        record(
            f"frozen_model_audit.json:{name}",
            model.get(name) is True and decision.get(name) is model.get(name),
        )
    model_rows = model.get("models") if type(model.get("models")) is list else []
    record(
        "frozen_model_audit.json:file_and_payload",
        _is_sha256(model.get("actual_model_file_sha256"))
        and model.get("actual_model_file_sha256")
        == model.get("expected_model_file_sha256")
        and model.get("payload_sha256_pass") is True,
    )
    record(
        "frozen_model_audit.json:no_fit",
        _is_int(model.get("MODEL_FIT_CALL_COUNT"), 0)
        and _is_int(model.get("SCALER_FIT_CALL_COUNT"), 0)
        and model.get("forbidden_fit_calls") == [],
    )
    record(
        "frozen_model_audit.json:model_inventory",
        _is_int(model.get("model_count"), 4)
        and len(model_rows) == 4
        and len({row.get("model_id") for row in model_rows if type(row) is dict}) == 4
        and all(
            type(row) is dict
            and row.get("feature_order_pass") is True
            and row.get("parameters_complete") is True
            and row.get("ridge_schema_pass") is True
            and type(row.get("feature_names")) is list
            and _is_sha256(row.get("training_data_sha256"))
            for row in model_rows
        ),
    )
    training_code = model.get("training_code_sha256")
    record(
        "frozen_model_audit.json:training_code",
        type(training_code) is dict
        and training_code.get("feature_source_sha256_pass") is True
        and training_code.get("survival_model_code_sha256_pass") is True
        and _is_sha256(training_code.get("feature_source_sha256"))
        and _is_sha256(training_code.get("survival_model_code_sha256")),
    )

    prediction_rows = (
        prediction.get("models") if type(prediction.get("models")) is list else []
    )
    tolerance = prediction.get("absolute_tolerance")
    record(
        "frozen_model_prediction_crosscheck.json:gate",
        prediction.get("FROZEN_MODEL_PREDICTION_CROSSCHECK_PASS") is True
        and decision.get("FROZEN_MODEL_PREDICTION_CROSSCHECK_PASS")
        is prediction.get("FROZEN_MODEL_PREDICTION_CROSSCHECK_PASS"),
    )
    record(
        "frozen_model_prediction_crosscheck.json:no_fit_or_rng",
        _is_int(prediction.get("MODEL_FIT_CALL_COUNT"), 0)
        and _is_int(prediction.get("SCALER_FIT_CALL_COUNT"), 0)
        and _is_int(prediction.get("fixture_rng_or_seed_use_count"), 0),
    )
    record(
        "frozen_model_prediction_crosscheck.json:predictions",
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and float(tolerance) == 1.0e-12
        and len(prediction_rows) == 4
        and len(
            {
                row.get("model_id")
                for row in prediction_rows
                if type(row) is dict
            }
        )
        == 4
        and all(
            type(row) is dict
            and row.get("prediction_pass") is True
            and isinstance(row.get("absolute_difference"), (int, float))
            and not isinstance(row.get("absolute_difference"), bool)
            and 0.0 <= float(row["absolute_difference"]) <= float(tolerance)
            and type(row.get("feature_fixture")) is dict
            and bool(row["feature_fixture"])
            for row in prediction_rows
        )
        and isinstance(
            prediction.get("maximum_absolute_prediction_difference"), (int, float)
        )
        and not isinstance(
            prediction.get("maximum_absolute_prediction_difference"), bool
        )
        and 0.0
        <= float(prediction["maximum_absolute_prediction_difference"])
        <= float(tolerance),
    )

    record(
        "fixture_regression_report.json:gate",
        fixture.get("CONFIRMATORY_EXECUTION_CHAIN_FIXTURE_PASS") is True
        and decision.get("CONFIRMATORY_EXECUTION_CHAIN_FIXTURE_PASS")
        is fixture.get("CONFIRMATORY_EXECUTION_CHAIN_FIXTURE_PASS"),
    )
    record(
        "fixture_regression_report.json:qualification",
        fixture.get("FIXTURE_QUALIFICATION_PASS") is True
        and _is_int(fixture.get("fixture_snapshot_count"), 3)
        and _is_int(fixture.get("fixture_trial_count"), 6)
        and _is_int(fixture.get("backend_execution_count"), 6)
        and _is_int(fixture.get("resume_backend_execution_count"), 0)
        and _is_int(fixture.get("input_pairing_violation_count"), 0)
        and fixture.get("fresh_resume_scientific_equivalence") is True
        and _is_int(fixture.get("analysis_verifier_difference_count"), 0)
        and fixture.get("publisher_pass") is True
        and fixture.get("artifact_verifier_pass") is True,
    )
    record(
        "fixture_regression_report.json:confirmatory_absence",
        _is_int(fixture.get("confirmatory_seed_or_rng_use_count"), 0)
        and _is_int(
            fixture.get("fixture_backend_execution_is_confirmatory_count"), 0
        ),
    )

    record(
        "dry_run_report.json:gate",
        dry.get("CONFIRMATORY_DRY_RUN_PASS") is True
        and decision.get("CONFIRMATORY_DRY_RUN_PASS")
        is dry.get("CONFIRMATORY_DRY_RUN_PASS"),
    )
    record(
        "dry_run_report.json:identity",
        dry.get("schema_version") == "synthetic_confirmatory_dry_run_v1"
        and dry.get("run_id") == implementation.get("formal_run_id")
        and isinstance(dry.get("output_dir"), str)
        and isinstance(implementation.get("formal_output_dir"), str)
        and (
            dry.get("output_dir") == implementation.get("formal_output_dir")
            or dry["output_dir"].endswith(
                f"/{implementation['formal_output_dir']}"
            )
        )
        and dry.get("workers") == implementation.get("formal_workers"),
    )
    record(
        "dry_run_report.json:cardinality",
        _is_int(dry.get("planned_snapshot_count"), 595)
        and _is_int(dry.get("planned_trial_count"), 1190)
        and _is_int(dry.get("open3d_trial_count"), 595)
        and _is_int(dry.get("pcl_trial_count"), 595)
        and _is_int(dry.get("native_trial_count"), 0)
        and dry.get("condition_snapshot_counts")
        == {
            "IDEAL_MATCHED": 35,
            "INDEPENDENT_NOISE_FREE": 35,
            "FULL_NOISE": 525,
        },
    )
    record(
        "dry_run_report.json:zero_violations",
        all(
            _is_int(dry.get(name), 0)
            for name in (
                "attempt_started_event_count",
                "duplicate_snapshot_count",
                "duplicate_trial_count",
                "independent_pseudoreplication_plan_count",
                "pairing_violation_count",
            )
        ),
    )
    record(
        "dry_run_report.json:no_execution",
        all(
            _is_int(dry.get(name), 0)
            for name in PRERUN_ZERO_EXECUTION_COUNTER_NAMES
        )
        and all(
            dry.get(name) == decision.get(name)
            for name in PRERUN_ZERO_EXECUTION_COUNTER_NAMES
        )
        and all(
            _is_int(dry.get(name), 0)
            for name in (
                "confirmatory_backend_execution_count",
                "confirmatory_rng_instantiation_count",
                "confirmatory_snapshot_generation_count",
                "confirmatory_trial_result_count",
            )
        )
        and dry.get("SYNTHETIC_CONFIRMATORY_EXECUTED") is False
        and dry.get("formal_execution_authorized_before_freeze") is False
        and dry.get("output_dir_created") is False,
    )

    bound_files = implementation.get("bound_files")
    record(
        "implementation_manifest.json:identity",
        implementation.get("schema_version")
        == "synthetic_confirmatory_prerun_implementation_manifest_v1"
        and implementation.get("formal_execution_authorized") is True
        and implementation.get("formal_manifest_path")
        == "frozen_assets/synthetic_confirmatory_formal_manifest_v1.json"
        and implementation.get("formal_run_id") == "synthetic-confirmatory-v1"
        and implementation.get("formal_output_dir")
        == "results/synthetic_confirmatory_v1"
        and _is_int(implementation.get("formal_workers"), 2)
        and _is_sha256(implementation.get("formal_manifest_file_sha256"))
        and _is_sha256(implementation.get("formal_manifest_payload_sha256")),
    )
    record(
        "implementation_manifest.json:bound_files",
        type(bound_files) is dict
        and bool(bound_files)
        and all(
            type(row) is dict
            and isinstance(row.get("path"), str)
            and bool(row.get("path"))
            and _is_sha256(row.get("sha256"))
            for row in bound_files.values()
        ),
    )
    record(
        "implementation_manifest.json:runner_interface",
        implementation.get("runner_arguments")
        == [
            "--manifest",
            "--run-id",
            "--output-dir",
            "--workers",
            "--resume",
            "--dry-run",
        ]
        and implementation.get("forbidden_runner_arguments")
        == [
            "--override",
            "--replace-seed",
            "--change-gate",
            "--change-model",
            "--fit-model",
            "--backend-subset",
            "--exclude-scene",
            "--native",
            "--ignore-manifest",
        ],
    )
    record(
        "implementation_manifest.json:survival_binding",
        implementation.get("scientific_survival_commit")
        == protocol.get("scientific_survival_commit")
        and implementation.get("scientific_survival_tag")
        == protocol.get("scientific_survival_tag"),
    )

    specialized = test.get("specialized_confirmatory")
    full = test.get("full_harness")
    pcl = test.get("pcl_backend_v3")
    record(
        "test_report.json:overall",
        test.get("schema_version")
        == "synthetic_confirmatory_prerun_test_report_v1"
        and test.get("test_report_pass") is True
        and test.get("PCL_V3_FIXTURE_QUALIFICATION_PASS") is True
        and test.get("SOURCE_DEGEN_LIO_PYTEST_EXECUTED") is False,
    )
    for label, suite in (("specialized", specialized), ("full", full)):
        record(
            f"test_report.json:{label}",
            type(suite) is dict
            and suite.get("test_pass") is True
            and type(suite.get("tests")) is int
            and suite["tests"] > 0
            and suite.get("expected_tests") == suite["tests"]
            and suite.get("passed") == suite["tests"]
            and _is_int(suite.get("failures"), 0)
            and _is_int(suite.get("errors"), 0)
            and _is_int(suite.get("skipped"), 0)
            and _is_sha256(suite.get("sha256")),
        )
    record(
        "test_report.json:pcl_v3",
        type(pcl) is dict
        and pcl.get("test_pass") is True
        and _is_int(pcl.get("tests"), 3)
        and _is_int(pcl.get("expected_tests"), 3)
        and _is_int(pcl.get("passed"), 3)
        and _is_int(pcl.get("failed"), 0)
        and _is_int(pcl.get("skipped"), 0)
        and type(pcl.get("test_names")) is list
        and len(pcl["test_names"]) == 3
        and len(set(pcl["test_names"])) == 3
        and _is_sha256(pcl.get("sha256")),
    )

    record(
        "run_manifest.json:identity",
        run.get("schema_version") == "synthetic_confirmatory_prerun_run_manifest_v1"
        and run.get("final_decision") == decision
        and run.get("formal_run_id") == implementation.get("formal_run_id")
        and run.get("formal_manifest_payload_sha256")
        == implementation.get("formal_manifest_payload_sha256"),
    )
    record(
        "run_manifest.json:fixture_counts",
        run.get("fixture_snapshot_count") == fixture.get("fixture_snapshot_count")
        and run.get("fixture_trial_count") == fixture.get("fixture_trial_count")
        and run.get("fixture_backend_execution_count")
        == fixture.get("backend_execution_count")
        and _is_int(run.get("pcl_v3_fixture_test_count"), 3),
    )
    record(
        "run_manifest.json:no_formal_execution",
        _is_int(run.get("confirmatory_formal_snapshot_count"), 0)
        and _is_int(run.get("confirmatory_formal_backend_execution_count"), 0)
        and _is_int(run.get("confirmatory_formal_trial_result_count"), 0)
        and run.get("confirmatory_formal_snapshot_count")
        == decision.get("CONFIRMATORY_SNAPSHOT_GENERATION_COUNT")
        and run.get("confirmatory_formal_backend_execution_count")
        == decision.get("CONFIRMATORY_BACKEND_EXECUTION_COUNT")
        and run.get("confirmatory_formal_trial_result_count")
        == decision.get("CONFIRMATORY_TRIAL_RESULT_COUNT")
        and run.get("source_degen_lio_pytest_executed") is False,
    )

    return checks, sorted(name for name, passed in checks.items() if not passed)


def verify_synthetic_confirmatory_prerun_artifact(
    path: str | Path, *, write_report: bool = False
) -> dict[str, Any]:
    """Verify 15 root files and the complete seed-free fixture publication."""

    root = Path(path).resolve()
    actual = {
        candidate.relative_to(root).as_posix() for candidate in root.rglob("*")
        if candidate.is_file()
    } if root.is_dir() else set()
    virtual = set(actual)
    if write_report:
        virtual.add("artifact_verification.json")
    expected_root = set(PRERUN_ROOT_FILES)
    embedded = {
        name
        for name in actual
        if name.startswith(f"{PRERUN_FIXTURE_PUBLICATION_DIRECTORY}/")
    }
    expected = expected_root | embedded
    missing = sorted(expected_root - virtual)
    extra = sorted(
        name
        for name in actual - expected_root
        if not name.startswith(f"{PRERUN_FIXTURE_PUBLICATION_DIRECTORY}/")
    )
    subdirectory_file_count = sum("/" in name for name in actual)
    json_errors = []
    objects: dict[str, dict[str, Any]] = {}
    for name in PRERUN_ROOT_FILES:
        if not name.endswith(".json"):
            continue
        candidate = root / name
        if not candidate.is_file():
            continue
        try:
            objects[name] = _json(candidate)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            json_errors.append(name)
    semantic_checks, semantic_failures = _prerun_semantic_checks(objects)
    fixture_crosscheck = _fixture_publication_crosscheck(
        root, objects.get("fixture_regression_report.json", {})
    )
    empty_evidence_files = sorted(
        name for name in PRERUN_EVIDENCE_FILES if not objects.get(name)
    )
    decision = objects.get("final_decision.json", {})
    required_gate_order_pass = (
        decision.get("required_gate_names") == list(PRERUN_REQUIRED_GATE_NAMES)
    )
    zero_counter_order_pass = (
        decision.get("zero_execution_counter_names")
        == list(PRERUN_ZERO_EXECUTION_COUNTER_NAMES)
    )
    required_gates_pass = all(
        decision.get(name) is True for name in PRERUN_REQUIRED_GATE_NAMES
    )
    zero_execution_counters_pass = all(
        type(decision.get(name)) is int and decision[name] == 0
        for name in PRERUN_ZERO_EXECUTION_COUNTER_NAMES
    )
    fixed_decision = bool(
        decision.get("SYNTHETIC_CONFIRMATORY_EXECUTED") is False
        and decision.get("SYNTHETIC_CONFIRMATORY_COMPLETE") is False
        and decision.get("SYNTHETIC_CONFIRMATORY_PASS") == "NOT_EVALUATED"
        and decision.get("REAL_DATA_RUN_AUTHORIZED") is False
        and decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is False
        and decision.get("SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS") is True
        and decision.get("CONFIRMATORY_RUN_AUTHORIZED") is True
        and required_gate_order_pass
        and zero_counter_order_pass
        and required_gates_pass
        and zero_execution_counters_pass
    )
    run = objects.get("run_manifest.json", {})
    run_decision = run.get("final_decision")
    run_decision_pass = run_decision is None or run_decision == decision
    execution_absence_pass = zero_execution_counters_pass
    report_path = root / "pre_run_report.md"
    markdown = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    required_report_statements = (
        "SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS",
        "CONFIRMATORY_RUN_AUTHORIZED",
        "SYNTHETIC_CONFIRMATORY_EXECUTED",
        "SYNTHETIC_CONFIRMATORY_PASS",
        "no Confirmatory snapshot",
    )
    missing_references = [
        statement for statement in required_report_statements
        if statement not in markdown
    ]
    checksums = _checksums(root, expected - PRERUN_SHA_EXCLUDED)
    result = {
        "schema_version": "synthetic_confirmatory_prerun_artifact_verification_v1",
        "required_file_count": len(expected), "actual_file_count": len(virtual),
        "missing_required_files": missing, "extra_files": extra,
        "subdirectory_file_count": subdirectory_file_count,
        "invalid_json_files": json_errors,
        "empty_evidence_files": empty_evidence_files,
        "evidence_semantic_checks": semantic_checks,
        "evidence_semantic_check_count": len(semantic_checks),
        "evidence_semantic_failure_count": len(semantic_failures),
        "evidence_semantic_failures": semantic_failures,
        "evidence_semantic_crosscheck_pass": not semantic_failures,
        "pre_run_report_missing_references": missing_references,
        "fixed_pre_run_decision_pass": fixed_decision,
        "required_gate_name_order_pass": required_gate_order_pass,
        "required_gate_value_pass": required_gates_pass,
        "zero_execution_counter_name_order_pass": zero_counter_order_pass,
        "zero_execution_counter_value_pass": zero_execution_counters_pass,
        "run_manifest_decision_match_pass": run_decision_pass,
        "formal_execution_absence_pass": execution_absence_pass,
        **fixture_crosscheck,
        **{name: value for name, value in checksums.items() if name != "pass"},
    }
    result["CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"] = bool(
        not missing and not extra and subdirectory_file_count > 0
        and not json_errors and not empty_evidence_files
        and not semantic_failures and not missing_references and fixed_decision
        and run_decision_pass and execution_absence_pass and checksums["pass"]
        and fixture_crosscheck["fixture_publication_crosscheck_pass"]
    )
    if write_report:
        (root / "artifact_verification.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def verify_synthetic_confirmatory_artifact(
    path: str | Path, *, write_report: bool = False
) -> dict[str, Any]:
    """Verify the compact future formal-result publication."""

    from .synthetic_confirmatory_independent_verifier import compare_primary_and_independent

    root = Path(path).resolve()
    actual = {candidate.relative_to(root).as_posix() for candidate in root.rglob("*") if candidate.is_file()} if root.is_dir() else set()
    virtual = set(actual)
    if write_report:
        virtual.add("artifact_verification.json")
    expected = set(FORMAL_REQUIRED_FILES)
    missing, extra = sorted(expected - virtual), sorted(actual - expected)
    csv_errors = []
    for name in FORMAL_TABLES:
        candidate = root / "tables" / name
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8", newline="") as stream:
                    if not csv.DictReader(stream).fieldnames: csv_errors.append(name)
            except (OSError, UnicodeError, csv.Error): csv_errors.append(name)
    png_errors = []
    for name in FORMAL_FIGURES:
        candidate = root / "figures" / name
        if candidate.is_file():
            data = candidate.read_bytes()
            if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 100: png_errors.append(name)
    objects = {}
    json_errors = []
    for name in ("primary_analysis.json", "independent_verification.json", "final_decision.json", "run_manifest.json"):
        candidate = root / name
        if candidate.is_file():
            try: objects[name] = _json(candidate)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError): json_errors.append(name)
    try:
        comparison = compare_primary_and_independent(objects["primary_analysis.json"], objects["independent_verification.json"])
    except (KeyError, TypeError, ValueError):
        comparison = {"section_difference_count": -1, "leaf_difference_count": -1, "maximum_absolute_numeric_difference": None}
    primary_object = objects.get("primary_analysis.json", {})
    independent_object = objects.get("independent_verification.json", {})
    run_object = objects.get("run_manifest.json", {})
    source_decision = primary_object.get("final_decision")
    published_decision = objects.get("final_decision.json")
    exact_analysis_pass = bool(
        comparison["leaf_difference_count"] == 0
        and comparison["maximum_absolute_numeric_difference"] == 0.0
    )
    analysis_input_binding_pass = bool(
        isinstance(run_object.get("run_id"), str)
        and bool(run_object["run_id"])
        and _is_sha256(run_object.get("raw_result_manifest_sha256"))
        and all(
            report.get("run_id") == run_object["run_id"]
            and report.get("raw_result_manifest_sha256")
            == run_object["raw_result_manifest_sha256"]
            for report in (primary_object, independent_object)
        )
    )
    decision_pass = bool(
        type(source_decision) is dict
        and published_decision == source_decision
        and exact_analysis_pass
        and analysis_input_binding_pass
    )
    checksums = _checksums(root, expected - {"SHA256SUMS", "artifact_verification.json"})
    report = root / "synthetic_confirmatory_report.md"
    markdown = report.read_text(encoding="utf-8") if report.is_file() else ""
    missing_refs = [name for name in (*FORMAL_TABLES, *FORMAL_FIGURES) if name not in markdown]
    result = {
        "schema_version": "synthetic_confirmatory_artifact_verification_v1",
        "required_file_count": len(expected), "actual_file_count": len(virtual),
        "missing_required_files": missing, "extra_files": extra,
        "invalid_csv_files": csv_errors, "invalid_png_files": png_errors,
        "invalid_json_files": json_errors, "report_missing_references": missing_refs,
        "analysis_verifier_comparison": comparison,
        "analysis_verifier_exact_match_pass": exact_analysis_pass,
        "analysis_input_binding_pass": analysis_input_binding_pass,
        "final_decision_match_pass": decision_pass,
        **{name: value for name, value in checksums.items() if name != "pass"},
    }
    result["ARTIFACT_VERIFICATION_PASS"] = bool(
        not missing and not extra and not csv_errors and not png_errors
        and not json_errors and not missing_refs and decision_pass and checksums["pass"]
    )
    if write_report:
        (root / "artifact_verification.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


__all__ = [
    "FORMAL_FIGURES", "FORMAL_REQUIRED_FILES", "FORMAL_ROOT_FILES", "FORMAL_TABLES",
    "PRERUN_EVIDENCE_FILES", "PRERUN_REQUIRED_GATE_NAMES", "PRERUN_ROOT_FILES",
    "PRERUN_ZERO_EXECUTION_COUNTER_NAMES", "verify_synthetic_confirmatory_artifact",
    "verify_synthetic_confirmatory_prerun_artifact",
]
