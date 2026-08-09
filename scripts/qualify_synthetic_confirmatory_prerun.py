#!/usr/bin/env python3
"""Qualify and atomically freeze the Synthetic Confirmatory v1 pre-run.

This entry point is deliberately incapable of building a Confirmatory snapshot
or executing a Confirmatory backend.  It audits frozen assets, performs the
metadata-only dry-run, verifies already-produced seed-free fixture/test
evidence, makes the single false-to-true manifest authorization transition,
and publishes the exact pre-run artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence


FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
FIXTURE_REPORT_PATH = Path(
    "/tmp/synthetic_confirmatory_seed_free_fixture_v1/fixture_qualification.json"
)
FIXTURE_PUBLICATION_PATH = Path(
    "/tmp/synthetic_confirmatory_seed_free_fixture_publication_v1"
)
SPECIALIZED_JUNIT_PATH = Path("/tmp/synthetic_confirmatory_specialized_junit.xml")
FULL_JUNIT_PATH = Path("/tmp/synthetic_confirmatory_full_harness_junit.xml")
PCL_V3_LOG_PATH = Path("/tmp/synthetic_confirmatory_pcl_v3_ctest.log")
SPECIALIZED_EXPECTED_TEST_COUNT = 104
FULL_HARNESS_EXPECTED_TEST_COUNT = 234


def _assert_preimport_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("pre-run qualification requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError(
            f"pre-run qualification requires MAMBA_ROOT_PREFIX={FROZEN_MAMBA_ROOT_PREFIX}"
        )
    source = SOURCE_REPOSITORY.resolve()
    entries = list(sys.path) + [
        item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("Python search path resolves to the source repository")


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _junit_summary(
    path: Path,
    label: str,
    *,
    expected_tests: int,
    exact_classnames: set[str] | None = None,
) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"invalid {label} JUnit XML") from error
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError(f"{label} JUnit XML has no testsuite")
    counts = {
        name: sum(int(float(suite.attrib.get(name, "0"))) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    testcases = list(root.findall(".//testcase"))
    classnames = {item.attrib.get("classname", "") for item in testcases}
    classnames_pass = exact_classnames is None or classnames == exact_classnames
    return {
        **counts,
        "expected_tests": expected_tests,
        "label": label,
        "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "test_classnames": sorted(classnames),
        "test_pass": bool(
            expected_tests > 0
            and counts["tests"] == expected_tests
            and len(testcases) == expected_tests
            and counts["failures"] == 0
            and counts["errors"] == 0
            and counts["skipped"] == 0
            and classnames_pass
        ),
    }


def _pcl_ctest_summary(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError("invalid PCL v3 CTest log") from error
    names = (
        "pcl_v3_nondegenerate_identity",
        "pcl_v3_known_small_transform",
        "pcl_v3_planar_degeneracy_diagnostic",
    )
    result_names = re.findall(r"Test #\d+:\s+([a-zA-Z0-9_]+).*?\bPassed\b", text)
    start_names = re.findall(r"Start \d+:\s+([a-zA-Z0-9_]+)", text)
    passed = bool(
        tuple(start_names) == names
        and tuple(result_names) == names
        and "100% tests passed, 0 tests failed out of 3" in text
        and "Failed" not in text
    )
    return {
        "errors": 0 if passed else None,
        "expected_tests": 3,
        "failed": 0 if passed else None,
        "label": "PCL backend qualification v3 fixture",
        "passed": 3 if passed else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "skipped": 0 if passed else None,
        "test_names": list(names),
        "test_pass": passed,
        "tests": 3,
    }


def _verify_fixture_artifact(report_path: Path) -> dict[str, Any]:
    if report_path.resolve() != FIXTURE_REPORT_PATH:
        raise ValueError("fixture report path differs from the qualification contract")
    root = report_path.parent
    checksum_path = root / "SHA256SUMS"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("fixture artifact checksum inventory is missing") from error
    entries: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError("fixture artifact has a malformed checksum entry")
        digest, relative = parts
        candidate = Path(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != relative
            or relative in entries
        ):
            raise ValueError("fixture artifact has an unsafe checksum entry")
        entries[relative] = digest
    actual = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.name != "SHA256SUMS"
    }
    if set(entries) != actual:
        raise ValueError("fixture artifact checksum inventory is incomplete")
    for relative, digest in entries.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("fixture artifact checksum mismatch")
    seed_strings = {
        "248284635",
        "376488233",
        "198112089",
        "229684695",
        "226655024",
        "469989467",
        "1088311622",
        "916609326",
        "1083684578",
    }
    structured = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".md", ".txt", ".ndjson"}
    ]
    if any(
        seed in path.read_text(encoding="utf-8")
        for path in structured
        for seed in seed_strings
    ):
        raise PermissionError("fixture artifact contains a Confirmatory seed")
    report = _strict_object(report_path, "fixture report")
    expected_inventory = {
        (backend, condition, classification, 1)
        for backend in ("open3d_point_to_plane", "pcl_point_to_plane")
        for condition, classification in (
            ("FIXTURE_IDENTITY", "NONE"),
            ("FIXTURE_NONIDENTITY_REFERENCE", "NONE"),
            ("FIXTURE_NO_CORRESPONDENCE", "NO_CORRESPONDENCES"),
        )
    }
    observed_inventory = {
        (
            row.get("backend"),
            row.get("condition"),
            row.get("failure_classification"),
            row.get("count"),
        )
        for row in report.get("failure_inventory", [])
        if type(row) is dict
    }
    if observed_inventory != expected_inventory:
        raise ValueError("fixture backend outcome inventory changed")
    if report.get("source_repository_runtime_file_read_count") != 0:
        raise PermissionError("fixture read the source repository at runtime")
    return report


def _predicted_json_file_sha(value: Mapping[str, Any]) -> str:
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


def _fixture_publication_binding(root: Path, verification: Mapping[str, Any]) -> dict[str, Any]:
    inventory = [
        {
            "path": candidate.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        }
        for candidate in sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    ]
    canonical_inventory = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "artifact_relative_path": "fixture_publication",
        "artifact_verification_file_sha256": hashlib.sha256(
            (root / "artifact_verification.json").read_bytes()
        ).hexdigest(),
        "file_count": len(inventory),
        "file_inventory": inventory,
        "file_inventory_sha256": hashlib.sha256(canonical_inventory).hexdigest(),
        "live_verification": dict(verification),
        "schema_version": "synthetic_confirmatory_fixture_publication_binding_v1",
        "sha256sums_file_sha256": hashlib.sha256(
            (root / "SHA256SUMS").read_bytes()
        ).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify and freeze Synthetic Confirmatory v1 without executing it"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-report", type=Path, required=True)
    parser.add_argument("--specialized-junit", type=Path, required=True)
    parser.add_argument("--full-junit", type=Path, required=True)
    parser.add_argument("--pcl-v3-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _assert_preimport_isolation()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from phase_a_harness.contracts import file_sha256
    from phase_a_harness.synthetic_confirmatory_manifest import (
        FORMAL_OUTPUT_DIR,
        FORMAL_RUN_ID,
        FORMAL_WORKERS,
        SCIENTIFIC_SURVIVAL_COMMIT,
        SCIENTIFIC_SURVIVAL_TAG,
        authorize_synthetic_confirmatory_manifest_once,
        signed_synthetic_confirmatory_manifest,
        verify_synthetic_confirmatory_manifest,
    )
    from phase_a_harness.synthetic_confirmatory_models import (
        audit_frozen_model_contract,
        crosscheck_frozen_model_predictions,
    )
    from phase_a_harness.synthetic_confirmatory_prerun import (
        ARTIFACT_FILES,
        REQUIRED_GATE_NAMES,
        ZERO_EXECUTION_COUNTER_NAMES,
        build_prerun_qualification_decision,
        publish_prerun_artifact_atomic,
    )
    from phase_a_harness.fixture_publication import (
        audit_and_publish_existing_fixture_results,
    )
    from phase_a_harness.fixture_publication_artifact_verifier import (
        verify_fixture_publication_artifact,
    )
    from phase_a_harness.synthetic_confirmatory_protocol import (
        audit_confirmatory_protocol_contract,
    )
    from phase_a_harness.synthetic_confirmatory_runner import (
        dry_run_synthetic_confirmatory,
    )
    from phase_a_harness.synthetic_confirmatory_artifact_verifier import (
        verify_synthetic_confirmatory_prerun_artifact,
    )

    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if root != manifest_path.parent.parent.resolve():
        raise ValueError("manifest must belong to this standalone harness")
    if output_dir != (root / "artifacts/synthetic_confirmatory_prerun_v1").resolve():
        raise ValueError("pre-run artifact output path differs from the frozen contract")
    survival_bundle = Path(
        "/tmp/zero-perturbation-scientific-survival-audit-v1.bundle"
    )
    if (
        not survival_bundle.is_file()
        or file_sha256(survival_bundle)
        != "06ca0c34b22e220d3c7cd905307715351c7bc73b8f7229066b72eaa15f0f41c7"
    ):
        raise PermissionError("Scientific Survival archive bundle changed")
    unauthorized = verify_synthetic_confirmatory_manifest(
        root, require_authorized=False
    )
    if unauthorized.get("formal_execution_authorized") is not False:
        raise PermissionError("pre-run qualification requires the initial unauthorized manifest")

    protocol = audit_confirmatory_protocol_contract(root)
    model = audit_frozen_model_contract(root)
    prediction = crosscheck_frozen_model_predictions(root)
    fixture_source = _verify_fixture_artifact(args.fixture_report.resolve())
    fixture_run_dir = args.fixture_report.resolve().parent
    fixture_manifest = root / "frozen_assets/frozen_experiment_manifest.json"
    if FIXTURE_PUBLICATION_PATH.exists():
        fixture_publication_verification = verify_fixture_publication_artifact(
            FIXTURE_PUBLICATION_PATH, write_report=False
        )
    else:
        fixture_publication_result = audit_and_publish_existing_fixture_results(
            manifest_path=fixture_manifest,
            fixture_run_dir=fixture_run_dir,
            artifact_dir=FIXTURE_PUBLICATION_PATH,
        )
        if (
            fixture_publication_result.get("FIXTURE_PUBLICATION_PASS") is not True
            or fixture_publication_result.get(
                "FIXTURE_ARTIFACT_VERIFICATION_PASS"
            )
            is not True
        ):
            raise PermissionError("seed-free fixture publication failed")
        fixture_publication_verification = verify_fixture_publication_artifact(
            FIXTURE_PUBLICATION_PATH, write_report=False
        )
    if (
        fixture_publication_verification.get(
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        )
        is not True
        or fixture_publication_verification.get(
            "recorded_verification_match_pass"
        )
        is not True
        or fixture_publication_verification.get("sha256_validation_pass") is not True
        or fixture_publication_verification.get("sha256_mismatch_count") != 0
        or fixture_publication_verification.get("semantic_error_count") != 0
        or fixture_publication_verification.get(
            "analysis_verifier_difference_count"
        )
        != 0
    ):
        raise PermissionError("live seed-free fixture publication verification failed")
    fixture_publication_decision = _strict_object(
        FIXTURE_PUBLICATION_PATH / "final_decision.json",
        "fixture publication decision",
    )
    fixture_publication_binding = _fixture_publication_binding(
        FIXTURE_PUBLICATION_PATH, fixture_publication_verification
    )
    fixture_pass = bool(
        fixture_source.get("FIXTURE_QUALIFICATION_PASS") is True
        and fixture_source.get("fixture_snapshot_count") == 3
        and fixture_source.get("fixture_trial_count") == 6
        and fixture_source.get("backend_execution_count") == 6
        and fixture_source.get("resume_backend_execution_count") == 0
        and fixture_source.get("input_pairing_violation_count") == 0
        and fixture_source.get("fresh_resume_scientific_equivalence") is True
        and fixture_source.get("analysis_verifier_difference_count") == 0
        and fixture_publication_decision.get("fixture_snapshot_count") == 3
        and fixture_publication_decision.get("fixture_trial_count") == 6
        and fixture_publication_decision.get("open3d_trial_count") == 3
        and fixture_publication_decision.get("pcl_trial_count") == 3
        and fixture_publication_decision.get("backend_execution_count") == 0
        and fixture_publication_decision.get("resume_backend_execution_count") == 0
        and fixture_publication_decision.get("rng_instantiation_count") == 0
        and fixture_publication_verification.get(
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        )
        is True
    )
    fixture = {
        **fixture_source,
        "CONFIRMATORY_EXECUTION_CHAIN_FIXTURE_PASS": fixture_pass,
        "artifact_verifier_pass": fixture_publication_verification.get(
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        ),
        "confirmatory_seed_or_rng_use_count": 0,
        "fixture_backend_execution_is_confirmatory_count": 0,
        "fixture_publication_binding": fixture_publication_binding,
        "publisher_pass": fixture_publication_verification.get(
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        ),
    }
    dry = dry_run_synthetic_confirmatory(
        manifest_path=manifest_path,
        run_id=FORMAL_RUN_ID,
        output_dir=root / FORMAL_OUTPUT_DIR,
        workers=FORMAL_WORKERS,
    )

    if args.specialized_junit.resolve() != SPECIALIZED_JUNIT_PATH:
        raise ValueError("specialized JUnit path differs from the qualification contract")
    if args.full_junit.resolve() != FULL_JUNIT_PATH:
        raise ValueError("full-harness JUnit path differs from the qualification contract")
    if args.pcl_v3_log.resolve() != PCL_V3_LOG_PATH:
        raise ValueError("PCL v3 log path differs from the qualification contract")
    specialized = _junit_summary(
        args.specialized_junit.resolve(),
        "Confirmatory specialized",
        expected_tests=SPECIALIZED_EXPECTED_TEST_COUNT,
        exact_classnames={
            "tests.test_synthetic_confirmatory_analysis_publication",
            "tests.test_synthetic_confirmatory_contracts",
            "tests.test_synthetic_confirmatory_runner",
        },
    )
    full = _junit_summary(
        args.full_junit.resolve(),
        "standalone harness full suite",
        expected_tests=FULL_HARNESS_EXPECTED_TEST_COUNT,
    )
    pcl = _pcl_ctest_summary(args.pcl_v3_log.resolve())
    test_report = {
        "PCL_V3_FIXTURE_QUALIFICATION_PASS": pcl["test_pass"],
        "SOURCE_DEGEN_LIO_PYTEST_EXECUTED": False,
        "full_harness": full,
        "pcl_backend_v3": pcl,
        "schema_version": "synthetic_confirmatory_prerun_test_report_v1",
        "specialized_confirmatory": specialized,
        "test_report_pass": bool(
            specialized["test_pass"] and full["test_pass"] and pcl["test_pass"]
        ),
    }
    if not test_report["test_report_pass"]:
        raise PermissionError("pre-run test evidence did not pass")

    gate = protocol["gate_contract_audit"]
    seed = protocol["seed_provenance_audit"]
    plan = protocol["plan_audit"]
    gates = {
        "SCIENTIFIC_SURVIVAL_BINDING_PASS": protocol.get(
            "SCIENTIFIC_SURVIVAL_BINDING_PASS"
        ),
        "CONFIRMATORY_PROTOCOL_BINDING_PASS": protocol.get(
            "CONFIRMATORY_PROTOCOL_BINDING_PASS"
        ),
        "CONFIRMATORY_GATE_CONTRACT_PASS": gate.get(
            "CONFIRMATORY_GATE_CONTRACT_PASS"
        ),
        "CONFIRMATORY_SEED_PROVENANCE_PASS": seed.get(
            "CONFIRMATORY_SEED_PROVENANCE_PASS"
        ),
        "CONFIRMATORY_PLAN_COUNT_PASS": plan.get("CONFIRMATORY_PLAN_COUNT_PASS"),
        "CONFIRMATORY_PLAN_UNIQUENESS_PASS": plan.get(
            "CONFIRMATORY_PLAN_UNIQUENESS_PASS"
        ),
        "CONFIRMATORY_PLAN_PAIRING_PASS": plan.get(
            "CONFIRMATORY_PLAN_PAIRING_PASS"
        ),
        "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS": plan.get(
            "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS"
        ),
        "FROZEN_MODEL_FILE_SHA_PASS": model.get("FROZEN_MODEL_FILE_SHA_PASS"),
        "FROZEN_MODEL_SCHEMA_PASS": model.get("FROZEN_MODEL_SCHEMA_PASS"),
        "FROZEN_MODEL_FEATURE_ORDER_PASS": model.get(
            "FROZEN_MODEL_FEATURE_ORDER_PASS"
        ),
        "FROZEN_MODEL_NO_REFIT_CONTRACT_PASS": model.get(
            "FROZEN_MODEL_NO_REFIT_CONTRACT_PASS"
        ),
        "FROZEN_MODEL_PREDICTION_CROSSCHECK_PASS": prediction.get(
            "FROZEN_MODEL_PREDICTION_CROSSCHECK_PASS"
        ),
        "CONFIRMATORY_EXECUTION_CHAIN_FIXTURE_PASS": bool(
            fixture_pass and pcl["test_pass"]
        ),
        "CONFIRMATORY_DRY_RUN_PASS": dry.get("CONFIRMATORY_DRY_RUN_PASS"),
        # The exact live artifact is verified below before authorization.
        "CONFIRMATORY_ARTIFACT_VERIFICATION_PASS": True,
    }
    counters = {name: dry.get(name) for name in ZERO_EXECUTION_COUNTER_NAMES}
    if any(gates.get(name) is not True for name in REQUIRED_GATE_NAMES[:-1]):
        raise PermissionError("one or more pre-run qualification gates failed")
    if any(type(counters.get(name)) is not int or counters[name] != 0 for name in ZERO_EXECUTION_COUNTER_NAMES):
        raise PermissionError("Confirmatory execution counters are not all zero")
    if (
        seed.get("CONFIRMATORY_SEED_USAGE_HIT_COUNT") != 0
        or seed.get("CONFIRMATORY_SEED_INSTANTIATION_COUNT") != 0
        or seed.get("CONFIRMATORY_SEED_PARSE_ERROR_COUNT") != 0
    ):
        raise PermissionError("Confirmatory seed provenance is not clean")

    expected_authorized = signed_synthetic_confirmatory_manifest(
        root, authorized=True
    )
    expected_manifest_file_sha = _predicted_json_file_sha(expected_authorized)
    decision = build_prerun_qualification_decision(
        gates=gates,
        counters=counters,
        confirmatory_run_authorized=True,
    )
    protocol_binding = {
        key: value
        for key, value in protocol.items()
        if key not in {"gate_contract_audit", "plan_audit", "seed_provenance_audit"}
    }
    implementation = {
        "bound_files": expected_authorized["bound_files"],
        "formal_execution_authorized": True,
        "formal_branch": "feature/zero-perturbation-synthetic-confirmatory-prerun",
        "formal_manifest_file_sha256": expected_manifest_file_sha,
        "formal_manifest_path": str(manifest_path.relative_to(root)),
        "formal_manifest_payload_sha256": expected_authorized[
            "manifest_payload_sha256"
        ],
        "formal_output_dir": FORMAL_OUTPUT_DIR,
        "formal_pre_run_tag": "archive/zero-perturbation-synthetic-confirmatory-pre-run-pass",
        "formal_run_id": FORMAL_RUN_ID,
        "formal_workers": FORMAL_WORKERS,
        "forbidden_runner_arguments": [
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
        "runner_arguments": [
            "--manifest",
            "--run-id",
            "--output-dir",
            "--workers",
            "--resume",
            "--dry-run",
        ],
        "schema_version": "synthetic_confirmatory_prerun_implementation_manifest_v1",
        "scientific_survival_commit": SCIENTIFIC_SURVIVAL_COMMIT,
        "scientific_survival_bundle_path": (
            "/tmp/zero-perturbation-scientific-survival-audit-v1.bundle"
        ),
        "scientific_survival_bundle_sha256": (
            "06ca0c34b22e220d3c7cd905307715351c7bc73b8f7229066b72eaa15f0f41c7"
        ),
        "scientific_survival_tag": SCIENTIFIC_SURVIVAL_TAG,
    }
    evidence = {
        "dry_run_report.json": dry,
        "fixture_regression_report.json": fixture,
        "frozen_model_audit.json": model,
        "frozen_model_prediction_crosscheck.json": prediction,
        "gate_contract_audit.json": gate,
        "implementation_manifest.json": implementation,
        "plan_audit.json": plan,
        "protocol_binding.json": protocol_binding,
        "seed_provenance_audit.json": seed,
        "test_report.json": test_report,
    }
    run_manifest = {
        "confirmatory_formal_backend_execution_count": 0,
        "confirmatory_formal_snapshot_count": 0,
        "confirmatory_formal_trial_result_count": 0,
        "final_decision": decision,
        "fixture_backend_execution_count": 6,
        "fixture_snapshot_count": 3,
        "fixture_trial_count": 6,
        "formal_manifest_payload_sha256": expected_authorized[
            "manifest_payload_sha256"
        ],
        "formal_run_id": FORMAL_RUN_ID,
        "pcl_v3_fixture_test_count": 3,
        "schema_version": "synthetic_confirmatory_prerun_run_manifest_v1",
        "source_degen_lio_pytest_executed": False,
    }

    # Atomically place the exact evidence while execution is still forbidden.
    # Authorization then verifies this live artifact before the sole false-to-
    # true transition.  A crash between the two steps is safely resumable.
    if output_dir.exists():
        verification = verify_synthetic_confirmatory_prerun_artifact(
            output_dir, write_report=False
        )
        if verification.get("CONFIRMATORY_ARTIFACT_VERIFICATION_PASS") is not True:
            raise FileExistsError(
                "existing pre-run artifact is not an exact resumable artifact"
            )
    else:
        verification = publish_prerun_artifact_atomic(
            output_dir,
            evidence=evidence,
            decision=decision,
            run_manifest=run_manifest,
            fixture_publication_dir=FIXTURE_PUBLICATION_PATH,
        )
    if verification.get("CONFIRMATORY_ARTIFACT_VERIFICATION_PASS") is not True:
        raise PermissionError("live pre-run artifact verification failed")

    authorized = authorize_synthetic_confirmatory_manifest_once(
        root, qualification_decision=decision
    )
    if (
        authorized != expected_authorized
        or file_sha256(manifest_path) != expected_manifest_file_sha
        or verify_synthetic_confirmatory_manifest(root, require_authorized=True)
        != expected_authorized
    ):
        raise ValueError("final manifest authorization differs from the qualified payload")

    result = {
        "artifact_output_dir": str(output_dir),
        "artifact_verification": verification,
        "formal_manifest_file_sha256": file_sha256(manifest_path),
        "formal_manifest_payload_sha256": authorized["manifest_payload_sha256"],
        "pre_run_artifact_file_count": sum(
            1 for path in output_dir.rglob("*") if path.is_file()
        ),
        "pre_run_artifact_root_file_count": len(ARTIFACT_FILES),
        **decision,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
