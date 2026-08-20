"""Write-once publisher and one-time authorization lifecycle artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contract_v1 import (
    CANDIDATE_NAME,
    FINAL_NAME,
    FINAL_SHA_NAME,
    IN_USE_NAME,
    INVALIDATION_NAME,
    RECEIPT_NAME,
    VERIFICATION_NAME,
    AuthorizationControlError,
    atomic_write_json,
    load_object,
    sha256_file,
)


def _utc(value: str | None = None) -> str:
    return value or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def publish_verified_authorization(
    lock_dir: Path, *, published_at_utc: str | None = None
) -> dict[str, Any]:
    """Publish only a candidate whose independent report is exact PASS."""

    lock_dir = lock_dir.resolve(strict=True)
    candidate_path = lock_dir / CANDIDATE_NAME
    report_path = lock_dir / VERIFICATION_NAME
    if candidate_path.is_symlink() or report_path.is_symlink():
        raise AuthorizationControlError("candidate/verifier symlink forbidden")
    candidate = load_object(candidate_path)
    report = load_object(report_path)
    candidate_sha = sha256_file(candidate_path)
    if not (
        report.get("pass") is True
        and report.get("status") == "PASS"
        and report.get("candidate_sha256") == candidate_sha
        and report.get("candidate_schema_pass") is True
        and report.get("candidate_permissions_pass") is True
        and report.get("no_prior_live_authorization") is True
    ):
        raise AuthorizationControlError("independent verifier report is not exact PASS")
    lock_path = lock_dir / "locked_analysis_lock_v1.json"
    lock_v2_path = lock_dir / "locked_analysis_lock_v2.json"
    if lock_path.read_bytes() != lock_v2_path.read_bytes():
        raise AuthorizationControlError("frozen-firewall compatibility lock mismatch")
    if candidate.get("analysis_lock_sha256") != sha256_file(lock_path):
        raise AuthorizationControlError("candidate lock SHA changed after verification")
    output_root = Path(str(candidate["analysis_output_root"])).resolve(strict=False)
    if output_root.exists() and any(output_root.iterdir() if output_root.is_dir() else (output_root,)):
        raise AuthorizationControlError("analysis output root contains an artifact")
    for name in (FINAL_NAME, FINAL_SHA_NAME, IN_USE_NAME, RECEIPT_NAME, INVALIDATION_NAME):
        if (lock_dir / name).exists():
            raise AuthorizationControlError(f"live authorization artifact already exists: {name}")
    final = dict(candidate)
    final["lifecycle_state"] = "VERIFIED_PUBLISHED"
    final["published_at_utc"] = _utc(published_at_utc)
    final["verification_report_sha256"] = sha256_file(report_path)
    final_path = lock_dir / FINAL_NAME
    atomic_write_json(final_path, final)
    final_sha = sha256_file(final_path)
    sidecar = lock_dir / FINAL_SHA_NAME
    sidecar.write_text(f"{final_sha}  {FINAL_NAME}\n", encoding="utf-8")
    return final


def mark_authorization_in_use(
    lock_dir: Path, *, first_real_value_read_utc: str | None = None
) -> dict[str, Any]:
    """Mark unblinding before the canonical runner opens its first value row."""

    lock_dir = lock_dir.resolve(strict=True)
    final_path = lock_dir / FINAL_NAME
    final = load_object(final_path)
    declared = (lock_dir / FINAL_SHA_NAME).read_text(encoding="utf-8").split()[0]
    if declared != sha256_file(final_path):
        raise AuthorizationControlError("published authorization SHA mismatch")
    if not (
        final.get("lifecycle_state") == "VERIFIED_PUBLISHED"
        and final.get("REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED") is True
        and final.get("consumed") is False
        and final.get("reusable") is False
    ):
        raise AuthorizationControlError("published authorization is not live one-time authority")
    for name in (IN_USE_NAME, RECEIPT_NAME, INVALIDATION_NAME):
        if (lock_dir / name).exists():
            raise AuthorizationControlError(f"authorization is not unused: {name}")
    marker = {
        "schema": "fmb1_formal_analysis_authorization_in_use_v1",
        "lifecycle_state": "IN_USE",
        "authorization_id": final["authorization_id"],
        "authorization_sha256": sha256_file(final_path),
        "analysis_lock_fingerprint": final["analysis_lock_fingerprint"],
        "analysis_code_commit": final["analysis_code_commit"],
        "first_real_value_read_utc": _utc(first_real_value_read_utc),
        "REAL_VALUES_ALREADY_UNBLINDED": True,
        "reusable": False,
    }
    atomic_write_json(lock_dir / IN_USE_NAME, marker)
    return marker


def consume_authorization(
    lock_dir: Path,
    *,
    analysis_completed_utc: str | None = None,
    analysis_output_manifest_sha256: str | None,
    success: bool,
) -> dict[str, Any]:
    """Consume permanently after success or any post-unblinding failure."""

    lock_dir = lock_dir.resolve(strict=True)
    marker = load_object(lock_dir / IN_USE_NAME)
    final_path = lock_dir / FINAL_NAME
    final = load_object(final_path)
    if (lock_dir / RECEIPT_NAME).exists():
        raise AuthorizationControlError("authorization already consumed")
    if marker.get("authorization_id") != final.get("authorization_id"):
        raise AuthorizationControlError("in-use marker authorization mismatch")
    if success and (
        not isinstance(analysis_output_manifest_sha256, str)
        or len(analysis_output_manifest_sha256) != 64
    ):
        raise AuthorizationControlError("successful consumption requires manifest SHA")
    receipt = {
        "schema": "fmb1_formal_analysis_authorization_consumption_receipt_v1",
        "lifecycle_state": (
            "CONSUMED" if success else "CONSUMED_BY_FAILED_ANALYSIS_ATTEMPT"
        ),
        "authorization_id": final["authorization_id"],
        "authorization_sha256": sha256_file(final_path),
        "analysis_lock_fingerprint": final["analysis_lock_fingerprint"],
        "analysis_code_commit": final["analysis_code_commit"],
        "first_real_value_read_utc": marker["first_real_value_read_utc"],
        "analysis_completed_utc": _utc(analysis_completed_utc),
        "analysis_output_manifest_sha256": analysis_output_manifest_sha256,
        "consumed": True,
        "reusable": False,
        "REAL_VALUES_ALREADY_UNBLINDED": True,
        "analysis_success": success,
    }
    atomic_write_json(lock_dir / RECEIPT_NAME, receipt)
    return receipt


def invalidate_published_before_unblinding(
    lock_dir: Path, *, reason: str, invalidated_at_utc: str | None = None
) -> dict[str, Any]:
    """Void a published authorization only while no in-use marker exists."""

    lock_dir = lock_dir.resolve(strict=True)
    final_path = lock_dir / FINAL_NAME
    final = load_object(final_path)
    if (lock_dir / IN_USE_NAME).exists() or (lock_dir / RECEIPT_NAME).exists():
        raise AuthorizationControlError("cannot invalidate after unblinding")
    if (lock_dir / INVALIDATION_NAME).exists():
        raise AuthorizationControlError("authorization already invalidated")
    record = {
        "schema": "fmb1_formal_analysis_authorization_invalidation_v1",
        "status": "VOID_BEFORE_REAL_VALUE_READ",
        "authorization_id": final["authorization_id"],
        "authorization_sha256": sha256_file(final_path),
        "reason": reason,
        "invalidated_at_utc": _utc(invalidated_at_utc),
        "REAL_VALUES_ALREADY_UNBLINDED": False,
        "reusable": False,
    }
    atomic_write_json(lock_dir / INVALIDATION_NAME, record)
    return record
