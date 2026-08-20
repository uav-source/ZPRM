"""Prepare a non-authoritative authorization candidate; never publish it."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contract_v1 import (
    ANALYSIS_OUTPUT_RELATIVE,
    AUTHORIZATION_SCHEMA_RELATIVE,
    C2_CLARIFICATION_COMMIT,
    CANDIDATE_NAME,
    CANDIDATE_SHA_NAME,
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
    AuthorizationControlError,
    atomic_write_json,
    canonical_fingerprint,
    ensure_no_symlink_components,
    load_object,
    sha256_file,
)


def _utc(value: str | None) -> str:
    if value is not None:
        if not value.endswith("Z"):
            raise AuthorizationControlError("issued_at_utc must use UTC Z suffix")
        return value
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def prepare_authorization_candidate(
    repository: Path,
    lock_dir: Path,
    *,
    analysis_control_commit: str,
    analysis_lock_release_commit: str,
    nonce: str,
    confirm_explicit_user_analysis_authorization: bool,
    output_root: Path,
    issued_at_utc: str | None = None,
    fixture_only: bool = False,
) -> dict[str, Any]:
    """Write a write-once candidate whose filename cannot satisfy the firewall."""

    if not confirm_explicit_user_analysis_authorization:
        raise AuthorizationControlError(
            "--confirm-explicit-user-analysis-authorization is required"
        )
    if len(analysis_control_commit) != 40 or len(analysis_lock_release_commit) != 40:
        raise AuthorizationControlError("control and lock release commits must be 40 hex")
    if len(nonce) < 32 or len(nonce) > 64 or any(c not in "0123456789abcdef" for c in nonce):
        raise AuthorizationControlError("nonce must be 32-64 lowercase hex characters")
    repository = repository.resolve(strict=True)
    ensure_no_symlink_components(lock_dir)
    lock_dir = lock_dir.resolve(strict=True)
    ensure_no_symlink_components(output_root)
    output_root = output_root.resolve(strict=False)
    if not fixture_only:
        expected_lock = (repository / LOCK_V2_RELATIVE).resolve(strict=True)
        expected_output = (repository / ANALYSIS_OUTPUT_RELATIVE).resolve(strict=False)
        if lock_dir != expected_lock or output_root != expected_output:
            raise AuthorizationControlError("formal lock/output root is not canonical")
    if output_root.exists() and any(output_root.iterdir() if output_root.is_dir() else (output_root,)):
        raise AuthorizationControlError("analysis output root already contains an artifact")
    for name in (CANDIDATE_NAME, FINAL_NAME, IN_USE_NAME, RECEIPT_NAME):
        if (lock_dir / name).exists():
            raise AuthorizationControlError(f"authorization lifecycle collision: {name}")

    lock_path = lock_dir / "locked_analysis_lock_v2.json"
    compatibility_path = lock_dir / "locked_analysis_lock_v1.json"
    if lock_path.is_symlink() or compatibility_path.is_symlink():
        raise AuthorizationControlError("lock files must not be symlinks")
    if lock_path.read_bytes() != compatibility_path.read_bytes():
        raise AuthorizationControlError("frozen-firewall compatibility lock is not byte-identical")
    lock = load_object(lock_path)
    lock_sha = sha256_file(lock_path)
    if not (
        lock.get("analysis_lock_revision") == 2
        and lock.get("LOCKED_ANALYSIS_CODE_COMMIT") == LOCKED_ANALYSIS_CODE_COMMIT
        and lock.get("analysis_control_commit") == analysis_control_commit
        and lock.get("analysis_lock_fingerprint") == canonical_fingerprint(lock)
        and lock.get("status") == "ISSUED_AWAITING_SEPARATE_REAL_ANALYSIS_AUTHORIZATION"
        and lock.get("AUTHORIZATION_INFRASTRUCTURE_READY") is True
        and lock.get("FORMAL_ANALYSIS_AUTHORIZATION_PRESENT") is False
    ):
        raise AuthorizationControlError("analysis lock v2 identity/lifecycle mismatch")
    formal_root = Path(str(lock["formal_results_root"])).resolve(strict=True)
    postrun_root = (repository / POSTRUN_RELATIVE).resolve(strict=True)
    schema_path = (repository / AUTHORIZATION_SCHEMA_RELATIVE).resolve(strict=True)
    if sha256_file(schema_path) != lock.get("analysis_authorization_schema_sha256"):
        raise AuthorizationControlError("authorization schema is not lock-bound")
    payload = {
        "schema": "fmb1_formal_analysis_authorization_v1",
        "authorization_id": f"FMB1_ZP_LOCKED_ANALYSIS_AUTH_{nonce.upper()}",
        "authorization_type": "ONE_TIME_FMB1_LOCKED_SCIENTIFIC_ANALYSIS",
        "authorization_basis": "EXPLICIT_USER_INSTRUCTION_RECORDED_BY_OPERATOR",
        "issued_at_utc": _utc(issued_at_utc),
        "published_at_utc": None,
        "nonce": nonce,
        "lifecycle_state": "ISSUED",
        "analysis_lock_revision": 2,
        "analysis_lock_fingerprint": lock["analysis_lock_fingerprint"],
        "analysis_lock_file_sha256": lock_sha,
        # Alias required by the already-frozen formal firewall.
        "analysis_lock_sha256": lock_sha,
        "analysis_lock_release_commit": analysis_lock_release_commit,
        "analysis_control_commit": analysis_control_commit,
        "locked_analysis_code_commit": LOCKED_ANALYSIS_CODE_COMMIT,
        # Alias required by the already-frozen formal firewall.
        "analysis_code_commit": LOCKED_ANALYSIS_CODE_COMMIT,
        "raw_execution_commit": RAW_EXECUTION_COMMIT,
        "postrun_verification_commit": POSTRUN_VERIFICATION_COMMIT,
        "c2_clarification_commit": C2_CLARIFICATION_COMMIT,
        "r3_lock_fingerprint": R3_LOCK_FINGERPRINT,
        "formal_results_root": str(formal_root),
        "postrun_verification_root": str(postrun_root),
        "analysis_output_root": str(output_root),
        "analysis_output_schema_sha256": OUTPUT_SCHEMA_SHA256,
        "track_id": "ZERO_PERTURBATION_TRACK",
        "authorization_scope": "ONE_AUTHORITATIVE_LOCKED_SCIENTIFIC_ANALYSIS_RUN",
        "allow_read_frozen_formal_results": True,
        "allow_scientific_aggregation": True,
        "allow_primary_exact_permutation": True,
        "allow_secondary_rotation": True,
        "allow_cross_backend_analysis": True,
        "allow_reassociation_analysis": True,
        "allow_systematic_component_analysis": True,
        "allow_registration": False,
        "allow_capture_radius": False,
        "allow_modify_formal_results": False,
        "allow_modify_analysis_code": False,
        "allow_change_statistics": False,
        "reusable": False,
        "immutable": True,
        "REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED": True,
        "consumed": False,
        "verification_report_sha256": None,
        "fixture_only": fixture_only,
    }
    candidate_path = lock_dir / CANDIDATE_NAME
    atomic_write_json(candidate_path, payload)
    digest = sha256_file(candidate_path)
    sidecar = lock_dir / CANDIDATE_SHA_NAME
    if sidecar.exists():
        raise AuthorizationControlError("candidate SHA sidecar already exists")
    sidecar.write_text(f"{digest}  {CANDIDATE_NAME}\n", encoding="utf-8")
    return payload
