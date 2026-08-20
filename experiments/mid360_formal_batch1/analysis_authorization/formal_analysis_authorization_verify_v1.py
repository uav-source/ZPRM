"""Independent candidate verifier; deliberately does not import the producer."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from experiments.mid360_formal_batch1.locked_analysis.contract_v1 import PROTECTED_BINDINGS

from .contract_v1 import (
    ANALYSIS_OUTPUT_RELATIVE,
    AUTHORIZATION_SCHEMA_RELATIVE,
    C2_CLARIFICATION_COMMIT,
    CANDIDATE_NAME,
    CANDIDATE_SHA_NAME,
    CONTROL_CODE_PATHS,
    FINAL_NAME,
    IN_USE_NAME,
    LOCKED_ANALYSIS_CODE_COMMIT,
    LOCK_V2_RELATIVE,
    OUTPUT_SCHEMA_SHA256,
    POSTRUN_RELATIVE,
    POSTRUN_VERIFICATION_COMMIT,
    RAW_EXECUTION_COMMIT,
    RECEIPT_NAME,
    R3_LOCK_FINGERPRINT,
    VERIFICATION_NAME,
    AuthorizationControlError,
    atomic_write_json,
    canonical_fingerprint,
    ensure_no_symlink_components,
    load_object,
    sha256_file,
)


class IndependentAuthorizationVerificationError(AuthorizationControlError):
    """Raised when independent candidate verification fails."""


def _git(repository: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["git", *args], cwd=repository, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if process.returncode:
        raise IndependentAuthorizationVerificationError(
            process.stderr.decode("utf-8", "replace").strip()
        )
    return process.stdout


def _expected_permissions(candidate: dict[str, Any]) -> None:
    exact_true = (
        "allow_read_frozen_formal_results", "allow_scientific_aggregation",
        "allow_primary_exact_permutation", "allow_secondary_rotation",
        "allow_cross_backend_analysis", "allow_reassociation_analysis",
        "allow_systematic_component_analysis", "immutable",
        "REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED",
    )
    exact_false = (
        "allow_registration", "allow_capture_radius", "allow_modify_formal_results",
        "allow_modify_analysis_code", "allow_change_statistics", "reusable", "consumed",
    )
    if any(candidate.get(key) is not True for key in exact_true):
        raise IndependentAuthorizationVerificationError("required permission is not exact true")
    if any(candidate.get(key) is not False for key in exact_false):
        raise IndependentAuthorizationVerificationError("forbidden permission is not exact false")


def verify_authorization_candidate(
    repository: Path,
    lock_dir: Path,
    *,
    expected_analysis_control_commit: str,
    expected_lock_release_commit: str,
    expected_output_root: Path,
    fixture_only: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    """Independently verify candidate bytes, frozen identities, scope, and paths."""

    repository = repository.resolve(strict=True)
    ensure_no_symlink_components(lock_dir)
    lock_dir = lock_dir.resolve(strict=True)
    ensure_no_symlink_components(expected_output_root)
    expected_output_root = expected_output_root.resolve(strict=False)
    candidate_path = lock_dir / CANDIDATE_NAME
    sidecar_path = lock_dir / CANDIDATE_SHA_NAME
    if candidate_path.is_symlink() or sidecar_path.is_symlink():
        raise IndependentAuthorizationVerificationError("candidate/sidecar symlink forbidden")
    candidate = load_object(candidate_path)
    declared_sha = sidecar_path.read_text(encoding="utf-8").split()[0]
    candidate_sha = sha256_file(candidate_path)
    if declared_sha != candidate_sha:
        raise IndependentAuthorizationVerificationError("candidate SHA mismatch")
    schema_path = repository / AUTHORIZATION_SCHEMA_RELATIVE
    schema = load_object(schema_path)
    errors = sorted(Draft202012Validator(schema).iter_errors(candidate),
                    key=lambda error: list(error.path))
    if errors:
        raise IndependentAuthorizationVerificationError(
            f"candidate schema failure at {list(errors[0].path)}: {errors[0].message}"
        )
    _expected_permissions(candidate)
    if candidate.get("lifecycle_state") != "ISSUED" or candidate.get("published_at_utc") is not None:
        raise IndependentAuthorizationVerificationError("candidate lifecycle must be ISSUED/unpublished")
    if candidate.get("verification_report_sha256") is not None:
        raise IndependentAuthorizationVerificationError("candidate cannot pre-bind a verifier report")
    try:
        datetime.fromisoformat(str(candidate["issued_at_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise IndependentAuthorizationVerificationError("candidate timestamp invalid") from exc
    if candidate.get("authorization_id") != (
        f"FMB1_ZP_LOCKED_ANALYSIS_AUTH_{str(candidate['nonce']).upper()}"
    ):
        raise IndependentAuthorizationVerificationError("authorization ID/nonce mismatch")

    lock_v2_path = lock_dir / "locked_analysis_lock_v2.json"
    compatibility_path = lock_dir / "locked_analysis_lock_v1.json"
    if lock_v2_path.is_symlink() or compatibility_path.is_symlink():
        raise IndependentAuthorizationVerificationError("lock symlink forbidden")
    if lock_v2_path.read_bytes() != compatibility_path.read_bytes():
        raise IndependentAuthorizationVerificationError("compatibility lock byte mismatch")
    lock = load_object(lock_v2_path)
    lock_sha = sha256_file(lock_v2_path)
    if not (
        lock.get("analysis_lock_revision") == 2
        and lock.get("analysis_lock_fingerprint") == canonical_fingerprint(lock)
        and lock.get("LOCKED_ANALYSIS_CODE_COMMIT") == LOCKED_ANALYSIS_CODE_COMMIT
        and lock.get("analysis_control_commit") == expected_analysis_control_commit
        and lock.get("AUTHORIZATION_INFRASTRUCTURE_READY") is True
        and lock.get("FORMAL_ANALYSIS_AUTHORIZATION_PRESENT") is False
    ):
        raise IndependentAuthorizationVerificationError("analysis lock v2 mismatch")
    exact_candidate = {
        "analysis_lock_revision": 2,
        "analysis_lock_fingerprint": lock["analysis_lock_fingerprint"],
        "analysis_lock_file_sha256": lock_sha,
        "analysis_lock_sha256": lock_sha,
        "analysis_lock_release_commit": expected_lock_release_commit,
        "analysis_control_commit": expected_analysis_control_commit,
        "locked_analysis_code_commit": LOCKED_ANALYSIS_CODE_COMMIT,
        "analysis_code_commit": LOCKED_ANALYSIS_CODE_COMMIT,
        "raw_execution_commit": RAW_EXECUTION_COMMIT,
        "postrun_verification_commit": POSTRUN_VERIFICATION_COMMIT,
        "c2_clarification_commit": C2_CLARIFICATION_COMMIT,
        "r3_lock_fingerprint": R3_LOCK_FINGERPRINT,
        "analysis_output_schema_sha256": OUTPUT_SCHEMA_SHA256,
        "track_id": "ZERO_PERTURBATION_TRACK",
        "authorization_scope": "ONE_AUTHORITATIVE_LOCKED_SCIENTIFIC_ANALYSIS_RUN",
        "authorization_type": "ONE_TIME_FMB1_LOCKED_SCIENTIFIC_ANALYSIS",
        "authorization_basis": "EXPLICIT_USER_INSTRUCTION_RECORDED_BY_OPERATOR",
        "fixture_only": fixture_only,
    }
    for key, expected in exact_candidate.items():
        if candidate.get(key) != expected:
            raise IndependentAuthorizationVerificationError(f"candidate binding mismatch: {key}")
    try:
        formal_root = Path(str(candidate["formal_results_root"])).resolve(strict=True)
        postrun_root = Path(str(candidate["postrun_verification_root"])).resolve(strict=True)
        output_root = Path(str(candidate["analysis_output_root"])).resolve(strict=False)
    except OSError as exc:
        raise IndependentAuthorizationVerificationError(
            "candidate root binding is missing or invalid"
        ) from exc
    if output_root != expected_output_root:
        raise IndependentAuthorizationVerificationError("candidate output root mismatch")
    if output_root.exists() and any(output_root.iterdir() if output_root.is_dir() else (output_root,)):
        raise IndependentAuthorizationVerificationError("analysis output root is not empty")
    if not fixture_only:
        if lock_dir != (repository / LOCK_V2_RELATIVE).resolve(strict=True):
            raise IndependentAuthorizationVerificationError("noncanonical lock directory")
        if formal_root != (repository / "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1").resolve(strict=True):
            raise IndependentAuthorizationVerificationError("formal results root mismatch")
        if postrun_root != (repository / POSTRUN_RELATIVE).resolve(strict=True):
            raise IndependentAuthorizationVerificationError("postrun root mismatch")
        if output_root != (repository / ANALYSIS_OUTPUT_RELATIVE).resolve(strict=False):
            raise IndependentAuthorizationVerificationError("formal output root mismatch")

    postrun = load_object(postrun_root / "postrun_independent_verification.json")
    if not (
        postrun.get("pass") is True
        and postrun.get("FMB1_POSTRUN_INDEPENDENT_VERIFICATION_PASS") is True
        and postrun.get("RAW_EXECUTION_BYTES_UNCHANGED") is True
        and postrun.get("VERIFIED_TRIAL_COUNT") == 360
        and postrun.get("raw_execution_commit") == RAW_EXECUTION_COMMIT
        and postrun.get("r3_lock_fingerprint") == R3_LOCK_FINGERPRINT
    ):
        raise IndependentAuthorizationVerificationError("Post-run verification mismatch")
    if not fixture_only:
        for role, (relative, digest) in PROTECTED_BINDINGS.items():
            if sha256_file(repository / relative) != digest:
                raise IndependentAuthorizationVerificationError(
                    f"protected scientific binding mismatch: {role}"
                )
        if _git(repository, "cat-file", "-t", expected_analysis_control_commit).strip() != b"commit":
            raise IndependentAuthorizationVerificationError("analysis control commit absent")
        if _git(repository, "cat-file", "-t", expected_lock_release_commit).strip() != b"commit":
            raise IndependentAuthorizationVerificationError("lock release commit absent")
        relative_lock = f"{LOCK_V2_RELATIVE}/locked_analysis_lock_v2.json"
        if _git(repository, "show", f"{expected_lock_release_commit}:{relative_lock}") != lock_v2_path.read_bytes():
            raise IndependentAuthorizationVerificationError("lock bytes differ from release commit")
        for relative in CONTROL_CODE_PATHS:
            if _git(repository, "show", f"{expected_analysis_control_commit}:{relative}") != (repository / relative).read_bytes():
                raise IndependentAuthorizationVerificationError(f"control code byte mismatch: {relative}")
    for name in (FINAL_NAME, IN_USE_NAME, RECEIPT_NAME):
        if (lock_dir / name).exists():
            raise IndependentAuthorizationVerificationError(f"prior live lifecycle artifact: {name}")

    report = {
        "schema": "fmb1_formal_analysis_authorization_independent_verification_v1",
        "status": "PASS", "pass": True,
        "candidate_sha256": candidate_sha,
        "analysis_lock_sha256": lock_sha,
        "analysis_lock_fingerprint": lock["analysis_lock_fingerprint"],
        "analysis_control_commit": expected_analysis_control_commit,
        "analysis_lock_release_commit": expected_lock_release_commit,
        "candidate_schema_pass": True,
        "candidate_permissions_pass": True,
        "candidate_cannot_directly_authorize": True,
        "output_root_empty": True,
        "no_prior_live_authorization": True,
        "independent_verifier_imports_producer": False,
        "REAL_FORMAL_RESULT_FILES_READ": 0,
        "REAL_SCIENTIFIC_VALUES_READ": 0,
        "REAL_WEAK_RICH_COMPARISON_COUNT": 0,
        "REAL_P_VALUE_COUNT": 0,
        "REGISTRATION_BACKEND_CALLS": 0,
        "fixture_only": fixture_only,
    }
    if write_report:
        atomic_write_json(lock_dir / VERIFICATION_NAME, report)
    return report
