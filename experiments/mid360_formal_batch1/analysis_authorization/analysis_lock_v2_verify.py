"""Independent verifier and fixture qualification for Analysis Lock v2."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from experiments.mid360_formal_batch1.locked_analysis.contract_v1 import PROTECTED_BINDINGS
from experiments.mid360_formal_batch1.locked_analysis.formal_firewall_v1 import (
    FormalReadRequest,
    FormalResultFirewallError,
    validate_formal_read_request,
)
from experiments.mid360_formal_batch1.locked_analysis.lock_v1 import ANALYSIS_CODE_PATHS

from .analysis_lock_v2 import LOCK_V2_CORE_FILES
from .authorized_analysis_runner_v1 import run_fixture_lifecycle
from .contract_v1 import (
    ANALYSIS_OUTPUT_RELATIVE,
    AUTHORIZATION_SCHEMA_RELATIVE,
    CANDIDATE_NAME,
    CONTROL_CODE_PATHS,
    FINAL_NAME,
    IN_USE_NAME,
    LOCKED_ANALYSIS_CODE_COMMIT,
    LOCK_V1_FINGERPRINT,
    LOCK_V1_RELEASE_COMMIT,
    LOCK_V1_RELATIVE,
    LOCK_V1_SHA256,
    POSTRUN_RELATIVE,
    RAW_EXECUTION_COMMIT,
    RECEIPT_NAME,
    R3_LOCK_FINGERPRINT,
    AuthorizationControlError,
    canonical_fingerprint,
    load_object,
    sha256_file,
)
from .formal_analysis_authorization_lifecycle_v1 import publish_verified_authorization
from .formal_analysis_authorization_producer_v1 import prepare_authorization_candidate
from .formal_analysis_authorization_verify_v1 import verify_authorization_candidate


class IndependentAnalysisLockV2VerificationError(AuthorizationControlError):
    """Raised on an independently observed v2 lock mismatch."""


def _git(repository: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["git", *args], cwd=repository, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if process.returncode:
        raise IndependentAnalysisLockV2VerificationError(
            process.stderr.decode("utf-8", "replace").strip()
        )
    return process.stdout


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_sums(lock_dir: Path, filename: str, expected: set[str]) -> None:
    rows: list[tuple[str, str]] = []
    for line in (lock_dir / filename).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        rows.append((digest, name))
    if len(rows) != len(expected) or {name for _, name in rows} != expected:
        raise IndependentAnalysisLockV2VerificationError(
            f"{filename} exact-set mismatch"
        )
    for digest, name in rows:
        path = lock_dir / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise IndependentAnalysisLockV2VerificationError(
                f"{filename} file mismatch: {name}"
            )


def _verify_inventory(
    repository: Path, inventory_path: Path, commit: str, expected_paths: tuple[str, ...]
) -> None:
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row.get("path") for row in rows] != list(expected_paths):
        raise IndependentAnalysisLockV2VerificationError("inventory path/order mismatch")
    if _git(repository, "cat-file", "-t", commit).strip() != b"commit":
        raise IndependentAnalysisLockV2VerificationError("inventory commit missing")
    for row in rows:
        relative = str(row["path"])
        data = (repository / relative).read_bytes()
        if not (
            hashlib.sha256(data).hexdigest() == row.get("sha256")
            and len(data) == int(str(row.get("bytes")))
            and row.get("git_blob_exact") in ("True", "true")
            and _git(repository, "show", f"{commit}:{relative}") == data
        ):
            raise IndependentAnalysisLockV2VerificationError(
                f"inventory byte mismatch: {relative}"
            )


def _static_control_audit(repository: Path) -> dict[str, Any]:
    findings: list[str] = []
    verifier_relative = (
        "experiments/mid360_formal_batch1/analysis_authorization/"
        "formal_analysis_authorization_verify_v1.py"
    )
    for relative in CONTROL_CODE_PATHS:
        if not relative.endswith(".py"):
            continue
        tree = ast.parse((repository / relative).read_text(encoding="utf-8"), relative)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if name == "open3d" or name.startswith(
                    ("phase_a_harness.open3d_backend", "phase_a_harness.pcl_backend")
                ):
                    findings.append(f"{relative}:{node.lineno}:denied-import:{name}")
                if relative == verifier_relative and name.endswith(
                    "formal_analysis_authorization_producer_v1"
                ):
                    findings.append(f"{relative}:{node.lineno}:producer-import:{name}")
    if findings:
        raise IndependentAnalysisLockV2VerificationError(
            f"authorization control static audit failed: {findings}"
        )
    return {
        "status": "PASS", "finding_count": 0,
        "scanned_path_count": len(CONTROL_CODE_PATHS),
        "candidate_verifier_imports_producer": False,
    }


def _fixture_repository(source_repository: Path, root: Path) -> tuple[Path, Path, Path]:
    repository = root / "repository"
    schema_source = source_repository / AUTHORIZATION_SCHEMA_RELATIVE
    schema_target = repository / AUTHORIZATION_SCHEMA_RELATIVE
    schema_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(schema_source, schema_target)
    formal_root = repository / "fixture_formal_results"
    formal_root.mkdir(parents=True)
    postrun_root = repository / POSTRUN_RELATIVE
    postrun_root.mkdir(parents=True)
    _write_json(postrun_root / "postrun_independent_verification.json", {
        "pass": True,
        "FMB1_POSTRUN_INDEPENDENT_VERIFICATION_PASS": True,
        "RAW_EXECUTION_BYTES_UNCHANGED": True,
        "VERIFIED_TRIAL_COUNT": 360,
        "raw_execution_commit": RAW_EXECUTION_COMMIT,
        "r3_lock_fingerprint": R3_LOCK_FINGERPRINT,
    })
    lock_dir = repository / "fixture_lock"
    lock_dir.mkdir(parents=True)
    output_root = repository / "fixture_analysis_output"
    return repository, lock_dir, output_root


def _fixture_lock(
    source_repository: Path, repository: Path, lock_dir: Path,
    output_root: Path, control_commit: str,
) -> None:
    lock: dict[str, Any] = {
        "schema": "fmb1_zero_perturbation_locked_analysis_lock_v2",
        "lock_id": "FMB1_FIXTURE_LOCK_V2",
        "analysis_lock_revision": 2,
        "status": "ISSUED_AWAITING_SEPARATE_REAL_ANALYSIS_AUTHORIZATION",
        "LOCKED_ANALYSIS_CODE_COMMIT": LOCKED_ANALYSIS_CODE_COMMIT,
        "analysis_control_commit": control_commit,
        "analysis_authorization_schema_sha256": sha256_file(
            repository / AUTHORIZATION_SCHEMA_RELATIVE
        ),
        "formal_results_root": str((repository / "fixture_formal_results").resolve()),
        "postrun_verification_root": str((repository / POSTRUN_RELATIVE).resolve()),
        "analysis_output_root": str(output_root.resolve()),
        "READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS": True,
        "AUTHORIZATION_INFRASTRUCTURE_READY": True,
        "FORMAL_ANALYSIS_AUTHORIZATION_PRESENT": False,
        "POSTRUN_VERIFICATION_COMMIT": "18bb94e62761f5193da8cdc5509c470cb4983244",
        "RAW_EXECUTION_COMMIT": RAW_EXECUTION_COMMIT,
        "R3_LOCK_FINGERPRINT": R3_LOCK_FINGERPRINT,
    }
    lock["analysis_lock_fingerprint"] = canonical_fingerprint(lock)
    _write_json(lock_dir / "locked_analysis_lock_v2.json", lock)
    (lock_dir / "locked_analysis_lock_v1.json").write_bytes(
        (lock_dir / "locked_analysis_lock_v2.json").read_bytes()
    )


def _fixture_request(repository: Path, lock_dir: Path, output_root: Path) -> FormalReadRequest:
    return FormalReadRequest(
        repository=repository,
        formal_results_root=repository / "fixture_formal_results",
        postrun_verification_root=repository / POSTRUN_RELATIVE,
        analysis_lock_dir=lock_dir,
        analysis_code_commit=LOCKED_ANALYSIS_CODE_COMMIT,
        output_dir=output_root,
        confirm_read_frozen_formal_results=True,
    )


def _run_one_fixture(source_repository: Path, *, interrupted: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fmb1-auth-fixture-") as temporary:
        repository, lock_dir, output_root = _fixture_repository(
            source_repository, Path(temporary)
        )
        control_commit = "1" * 40
        release_commit = "2" * 40
        _fixture_lock(source_repository, repository, lock_dir, output_root, control_commit)
        prepare_authorization_candidate(
            repository, lock_dir, analysis_control_commit=control_commit,
            analysis_lock_release_commit=release_commit, nonce="a" * 32,
            confirm_explicit_user_analysis_authorization=True,
            output_root=output_root, issued_at_utc="2026-08-20T00:00:00Z",
            fixture_only=True,
        )
        request = _fixture_request(repository, lock_dir, output_root)
        candidate_rejected = False
        try:
            validate_formal_read_request(request)
        except FormalResultFirewallError:
            candidate_rejected = True
        report = verify_authorization_candidate(
            repository, lock_dir,
            expected_analysis_control_commit=control_commit,
            expected_lock_release_commit=release_commit,
            expected_output_root=output_root, fixture_only=True,
        )
        candidate_plus_report_rejected = False
        try:
            validate_formal_read_request(request)
        except FormalResultFirewallError:
            candidate_plus_report_rejected = True
        publish_verified_authorization(
            lock_dir, published_at_utc="2026-08-20T00:01:00Z"
        )
        boundary = validate_formal_read_request(request)
        if interrupted:
            def callback() -> str:
                raise RuntimeError("intentional fixture interruption")
            try:
                run_fixture_lifecycle(
                    request, fixture_callback=callback,
                    first_read_utc="2026-08-20T00:02:00Z",
                    completed_utc="2026-08-20T00:03:00Z",
                )
            except RuntimeError:
                pass
        else:
            receipt = run_fixture_lifecycle(
                request, fixture_callback=lambda: "b" * 64,
                first_read_utc="2026-08-20T00:02:00Z",
                completed_utc="2026-08-20T00:03:00Z",
            )
            if receipt.get("lifecycle_state") != "CONSUMED":
                raise IndependentAnalysisLockV2VerificationError(
                    "fixture success was not consumed"
                )
        receipt = load_object(lock_dir / RECEIPT_NAME)
        return {
            "candidate_rejected": candidate_rejected,
            "candidate_plus_report_rejected": candidate_plus_report_rejected,
            "verifier_pass": report.get("pass") is True,
            "published_firewall_pass": boundary.get("status")
            == "PASS_AUTHORIZED_FORMAL_READ_BOUNDARY",
            "receipt_state": receipt.get("lifecycle_state"),
            "receipt_consumed": receipt.get("consumed") is True,
            "in_use_present": (lock_dir / IN_USE_NAME).is_file(),
            "final_present": (lock_dir / FINAL_NAME).is_file(),
        }


def run_fixture_qualification(repository: Path) -> dict[str, Any]:
    """Exercise only artificial metadata and callbacks in temporary roots."""

    repository = repository.resolve(strict=True)
    success = _run_one_fixture(repository, interrupted=False)
    interrupted = _run_one_fixture(repository, interrupted=True)
    passed = (
        success == {
            "candidate_rejected": True,
            "candidate_plus_report_rejected": True,
            "verifier_pass": True,
            "published_firewall_pass": True,
            "receipt_state": "CONSUMED",
            "receipt_consumed": True,
            "in_use_present": True,
            "final_present": True,
        }
        and interrupted.get("receipt_state")
        == "CONSUMED_BY_FAILED_ANALYSIS_ATTEMPT"
        and interrupted.get("receipt_consumed") is True
    )
    return {
        "schema": "fmb1_analysis_authorization_fixture_qualification_v1",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "ANALYSIS_AUTHORIZATION_FIXTURE_QUALIFICATION_PASS": passed,
        "candidate_cannot_directly_authorize": success["candidate_rejected"],
        "candidate_plus_verifier_cannot_authorize_before_publish": success[
            "candidate_plus_report_rejected"
        ],
        "published_authorization_firewall_gate_pass": success[
            "published_firewall_pass"
        ],
        "success_lifecycle_pass": success.get("receipt_state") == "CONSUMED",
        "interruption_lifecycle_pass": interrupted.get("receipt_state")
        == "CONSUMED_BY_FAILED_ANALYSIS_ATTEMPT",
        "fixture_only": True,
        "REAL_FORMAL_RESULT_FILES_READ": 0,
        "REAL_SCIENTIFIC_VALUES_READ": 0,
        "REAL_WEAK_RICH_COMPARISON_COUNT": 0,
        "REAL_P_VALUE_COUNT": 0,
        "REGISTRATION_BACKEND_CALLS": 0,
    }


def verify_analysis_lock_v2(
    repository: Path, lock_dir: Path, *, write_report: bool = True
) -> dict[str, Any]:
    """Verify v2 issuance while requiring that real authorization is absent."""

    repository = repository.resolve(strict=True)
    lock_dir = lock_dir.resolve(strict=True)
    lock_path = lock_dir / "locked_analysis_lock_v2.json"
    lock = load_object(lock_path)
    failures: list[str] = []
    static: dict[str, Any] = {"status": "FAIL", "finding_count": None}
    try:
        compatibility = lock_dir / "locked_analysis_lock_v1.json"
        if lock_path.is_symlink() or compatibility.is_symlink():
            raise IndependentAnalysisLockV2VerificationError("lock symlink forbidden")
        if compatibility.read_bytes() != lock_path.read_bytes():
            raise IndependentAnalysisLockV2VerificationError(
                "frozen-firewall compatibility view mismatch"
            )
        exact = {
            "schema": "fmb1_zero_perturbation_locked_analysis_lock_v2",
            "lock_id": "FMB1_ZERO_PERTURBATION_LOCKED_ANALYSIS_V2",
            "analysis_lock_revision": 2,
            "status": "ISSUED_AWAITING_SEPARATE_REAL_ANALYSIS_AUTHORIZATION",
            "LOCKED_ANALYSIS_CODE_COMMIT": LOCKED_ANALYSIS_CODE_COMMIT,
            "supersedes_lock_v1_release_commit": LOCK_V1_RELEASE_COMMIT,
            "supersedes_lock_v1_sha256": LOCK_V1_SHA256,
            "supersedes_lock_v1_fingerprint": LOCK_V1_FINGERPRINT,
            "READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS": True,
            "AUTHORIZATION_INFRASTRUCTURE_READY": True,
            "REAL_ANALYSIS_AUTHORIZATION_PRODUCER_READY": True,
            "INDEPENDENT_REAL_ANALYSIS_AUTHORIZATION_VERIFIER_READY": True,
            "VERIFIED_AUTHORIZATION_PUBLISHER_READY": True,
            "REAL_ANALYSIS_AUTHORIZATION_LIFECYCLE_QUALIFIED": True,
            "FROZEN_FIREWALL_COMPATIBILITY_PASS": True,
            "FORMAL_ANALYSIS_AUTHORIZATION_PRESENT": False,
            "REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED": False,
            "REAL_SCIENTIFIC_ANALYSIS_EXECUTED": False,
            "REAL_VALUES_ALREADY_UNBLINDED": False,
            "REAL_FORMAL_RESULT_FILES_READ": 0,
            "REAL_P_VALUE_COUNT": 0,
            "SCIENTIFIC_ANALYSIS_IMPLEMENTATION_CHANGED": False,
            "READY_FOR_SEPARATE_REAL_ANALYSIS_AUTHORIZATION": True,
        }
        for key, expected in exact.items():
            if lock.get(key) != expected:
                raise IndependentAnalysisLockV2VerificationError(
                    f"v2 lock field mismatch: {key}"
                )
        if lock.get("analysis_lock_fingerprint") != canonical_fingerprint(lock):
            raise IndependentAnalysisLockV2VerificationError("v2 fingerprint mismatch")
        lock_sha = sha256_file(lock_path)
        for stem in ("locked_analysis_lock_v2", "locked_analysis_lock_v1"):
            if (lock_dir / f"{stem}.sha256").read_text(encoding="utf-8").split()[0] != lock_sha:
                raise IndependentAnalysisLockV2VerificationError(
                    f"v2 lock sidecar mismatch: {stem}"
                )
        v1_path = repository / LOCK_V1_RELATIVE / "locked_analysis_lock_v1.json"
        if sha256_file(v1_path) != LOCK_V1_SHA256:
            raise IndependentAnalysisLockV2VerificationError("historical v1 changed")
        if _git(repository, "show", f"{LOCK_V1_RELEASE_COMMIT}:{LOCK_V1_RELATIVE}/locked_analysis_lock_v1.json") != v1_path.read_bytes():
            raise IndependentAnalysisLockV2VerificationError("historical v1 Git binding failed")
        control_commit = str(lock.get("analysis_control_commit"))
        _verify_inventory(
            repository, lock_dir / "analysis_code_inventory.csv",
            LOCKED_ANALYSIS_CODE_COMMIT, tuple(ANALYSIS_CODE_PATHS),
        )
        _verify_inventory(
            repository, lock_dir / "analysis_control_inventory.csv",
            control_commit, tuple(CONTROL_CODE_PATHS),
        )
        for role, (relative, digest) in PROTECTED_BINDINGS.items():
            if sha256_file(repository / relative) != digest:
                raise IndependentAnalysisLockV2VerificationError(
                    f"protected scientific binding mismatch: {role}"
                )
            if lock.get("protected_bindings", {}).get(role) != {
                "path": relative, "sha256": digest,
            }:
                raise IndependentAnalysisLockV2VerificationError(
                    f"v2 protected binding mismatch: {role}"
                )
        audit = load_object(lock_dir / "analysis_authorization_contract_audit.json")
        control_qualification = load_object(
            lock_dir / "analysis_authorization_control_qualification.json"
        )
        qualification = load_object(
            lock_dir / "analysis_authorization_fixture_qualification.json"
        )
        attestation = load_object(lock_dir / "analysis_no_real_results_attestation.json")
        supersession = load_object(
            lock_dir / "analysis_lock_v1_supersession_binding.json"
        )
        if not (
            audit.get("EXTERNAL_AUTH_INFRA_CAN_SATISFY_FROZEN_FIREWALL") is True
            and control_qualification.get("ANALYSIS_AUTH_CONTROL_COMMIT")
            == control_commit
            and control_qualification.get("LOCKED_ANALYSIS_CODE_COMMIT_UNCHANGED") is True
            and control_qualification.get("SCIENTIFIC_ANALYSIS_IMPLEMENTATION_CHANGED") is False
            and control_qualification.get("REAL_SCIENTIFIC_VALUES_READ") == 0
            and qualification.get("ANALYSIS_AUTHORIZATION_FIXTURE_QUALIFICATION_PASS") is True
            and attestation.get("pass") is True
            and supersession.get("status")
            == "SUPERSEDED_BEFORE_REAL_SCIENTIFIC_ANALYSIS"
        ):
            raise IndependentAnalysisLockV2VerificationError(
                "v2 authorization evidence mismatch"
            )
        live_qualification = run_fixture_qualification(repository)
        for key, value in live_qualification.items():
            if qualification.get(key) != value:
                raise IndependentAnalysisLockV2VerificationError(
                    f"fixture qualification mismatch: {key}"
                )
        static = _static_control_audit(repository)
        forbidden = (
            CANDIDATE_NAME, "formal_analysis_authorization.candidate.sha256",
            "formal_analysis_authorization_verification.json", FINAL_NAME,
            "formal_analysis_authorization.sha256", IN_USE_NAME, RECEIPT_NAME,
            "formal_analysis_authorization_invalidation.json",
        )
        if any((lock_dir / name).exists() for name in forbidden):
            raise IndependentAnalysisLockV2VerificationError(
                "real authorization/lifecycle artifact exists at issuance"
            )
        output_root = Path(str(lock.get("analysis_output_root", ""))).resolve(strict=False)
        if output_root != (repository / ANALYSIS_OUTPUT_RELATIVE).resolve(strict=False):
            raise IndependentAnalysisLockV2VerificationError("analysis output root mismatch")
        if output_root.exists() and any(
            output_root.iterdir() if output_root.is_dir() else (output_root,)
        ):
            raise IndependentAnalysisLockV2VerificationError(
                "real analysis output exists"
            )
        _verify_sums(lock_dir, "LOCK_CORE_SHA256SUMS", set(LOCK_V2_CORE_FILES))
    except (OSError, ValueError, KeyError, AuthorizationControlError) as exc:
        failures.append(str(exc))

    report = {
        "schema": "fmb1_locked_analysis_v2_independent_verification_v1",
        "status": "PASS" if not failures else "FAIL",
        "pass": not failures,
        "LOCKED_ANALYSIS_V2_LOCK_VERIFIER_PASS": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "analysis_lock_sha256": sha256_file(lock_path),
        "analysis_lock_fingerprint": lock.get("analysis_lock_fingerprint"),
        "LOCKED_ANALYSIS_CODE_COMMIT": lock.get("LOCKED_ANALYSIS_CODE_COMMIT"),
        "analysis_control_commit": lock.get("analysis_control_commit"),
        "statistical_implementation_bytes_unchanged": not failures,
        "c1_unchanged": not failures,
        "c2_unchanged": not failures,
        "raw_execution_unchanged": not failures,
        "postrun_verification_pass": not failures,
        "determinacy_pass": not failures,
        "authorization_infrastructure_ready": not failures,
        "candidate_cannot_directly_authorize": not failures,
        "fixture_lifecycle_pass": not failures,
        "final_real_authorization_absent": not failures,
        "real_analysis_output_absent": not failures,
        "REAL_FORMAL_RESULT_FILES_READ": 0,
        "REAL_SCIENTIFIC_VALUES_READ": 0,
        "REAL_P_VALUE_COUNT": 0,
        "REGISTRATION_BACKEND_CALLS": 0,
        "static_control_audit": static,
    }
    if write_report:
        report_path = lock_dir / "independent_analysis_lock_verification.json"
        _write_json(report_path, report)
        actual = {
            path.name for path in lock_dir.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        }
        lines = [
            f"{sha256_file(lock_dir / name)}  {name}\n" for name in sorted(actual)
        ]
        (lock_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
        _verify_sums(lock_dir, "SHA256SUMS", actual)
    return report
