"""Fail-closed boundary for any future read of frozen formal result rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .contract_v1 import (
    POSTRUN_VERIFICATION_COMMIT,
    R3_LOCK_FINGERPRINT,
    RAW_EXECUTION_COMMIT,
    sha256_file,
)


class FormalResultFirewallError(RuntimeError):
    """Raised before formal values are opened when authorization is incomplete."""


@dataclass(frozen=True)
class FormalReadRequest:
    repository: Path
    formal_results_root: Path
    postrun_verification_root: Path
    analysis_lock_dir: Path
    analysis_code_commit: str
    output_dir: Path
    confirm_read_frozen_formal_results: bool


def reject_ambiguous_mode(*, fixture_only: bool, formal_arguments_present: bool,
                          confirmed: bool) -> str:
    """Perform flag-only rejection without touching any supplied path."""

    if fixture_only and formal_arguments_present:
        raise FormalResultFirewallError(
            "--fixture-only is mutually exclusive with every formal-result argument"
        )
    if fixture_only:
        if confirmed:
            raise FormalResultFirewallError("fixture mode must not use the formal-read confirmation")
        return "FIXTURE_ONLY"
    if not formal_arguments_present:
        raise FormalResultFirewallError("choose --fixture-only or provide the complete formal mode")
    if not confirmed:
        raise FormalResultFirewallError(
            "formal paths are rejected without --confirm-read-frozen-formal-results"
        )
    return "FORMAL"


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FormalResultFirewallError(f"expected JSON object: {path}")
    return payload


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_formal_read_request(request: FormalReadRequest) -> dict[str, Any]:
    """Validate all metadata gates before returning a value-bearing result path."""

    if not request.confirm_read_frozen_formal_results:
        raise FormalResultFirewallError("formal-read confirmation is absent")
    repository = request.repository.resolve(strict=True)
    result_root = request.formal_results_root.resolve(strict=True)
    postrun_root = request.postrun_verification_root.resolve(strict=True)
    lock_dir = request.analysis_lock_dir.resolve(strict=True)
    output = request.output_dir.resolve(strict=False)
    if _inside(output, result_root) or _inside(output, postrun_root):
        raise FormalResultFirewallError("analysis output must be outside frozen result roots")
    if output.exists() and any(output.iterdir() if output.is_dir() else (output,)):
        raise FormalResultFirewallError("analysis output path already contains an artifact")

    lock_path = lock_dir / "locked_analysis_lock_v1.json"
    if not lock_path.is_file():
        raise FormalResultFirewallError("canonical analysis lock is absent")
    lock = _object(lock_path)
    if lock.get("status") != "ISSUED_AWAITING_SEPARATE_REAL_ANALYSIS_AUTHORIZATION":
        raise FormalResultFirewallError("analysis lock lifecycle is not the frozen issued state")
    if lock.get("READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS") is not True:
        raise FormalResultFirewallError("analysis lock is not ready")
    if lock.get("LOCKED_ANALYSIS_CODE_COMMIT") != request.analysis_code_commit:
        raise FormalResultFirewallError("analysis code commit does not match the lock")

    # The lock deliberately grants no authority.  A later task must place a
    # separately verified, one-use authorization beside it.  Merely supplying
    # the CLI confirmation flag cannot manufacture that authority.
    authorization_path = lock_dir / "formal_analysis_authorization.json"
    if not authorization_path.is_file():
        raise FormalResultFirewallError("separate formal-analysis authorization is absent")
    authorization = _object(authorization_path)
    if not (
        authorization.get("REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED") is True
        and authorization.get("analysis_lock_sha256") == sha256_file(lock_path)
        and authorization.get("analysis_code_commit") == request.analysis_code_commit
        and authorization.get("consumed") is False
    ):
        raise FormalResultFirewallError("separate formal-analysis authorization is invalid")

    postrun_path = postrun_root / "postrun_independent_verification.json"
    postrun = _object(postrun_path)
    if not (
        postrun.get("FMB1_POSTRUN_INDEPENDENT_VERIFICATION_PASS") is True
        and postrun.get("pass") is True
        and postrun.get("RAW_EXECUTION_BYTES_UNCHANGED") is True
        and postrun.get("raw_execution_commit") == RAW_EXECUTION_COMMIT
        and postrun.get("r3_lock_fingerprint") == R3_LOCK_FINGERPRINT
        and postrun.get("VERIFIED_TRIAL_COUNT") == 360
    ):
        raise FormalResultFirewallError("Post-run independent verification is not exact PASS")
    if lock.get("POSTRUN_VERIFICATION_COMMIT") != POSTRUN_VERIFICATION_COMMIT:
        raise FormalResultFirewallError("Post-run verification commit mismatch")
    if lock.get("RAW_EXECUTION_COMMIT") != RAW_EXECUTION_COMMIT:
        raise FormalResultFirewallError("raw execution commit mismatch")
    if lock.get("R3_LOCK_FINGERPRINT") != R3_LOCK_FINGERPRINT:
        raise FormalResultFirewallError("R3 lock fingerprint mismatch")
    if Path(lock.get("formal_results_root", "")).resolve() != result_root:
        raise FormalResultFirewallError("formal result root is not the lock-bound root")
    return {
        "status": "PASS_AUTHORIZED_FORMAL_READ_BOUNDARY",
        "repository": str(repository),
        "formal_results_root": str(result_root),
        "postrun_verification_report_sha256": sha256_file(postrun_path),
        "analysis_lock_sha256": sha256_file(lock_path),
        "analysis_code_commit": request.analysis_code_commit,
        "authorization_sha256": sha256_file(authorization_path),
    }


def load_frozen_formal_attempts(
    request: FormalReadRequest, result_schema: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open value-bearing rows only after the complete boundary passes."""

    boundary = validate_formal_read_request(request)
    root = request.formal_results_root.resolve(strict=True)
    result_paths = sorted((root / "raw_runtime_snapshot" / "trial_results").glob(
        "*/attempt-[0-9][0-9][0-9][0-9].json"
    ))
    if len(result_paths) != 360:
        raise FormalResultFirewallError("frozen formal result inventory must contain 360 rows")
    validator = Draft202012Validator(result_schema)
    rows: list[dict[str, Any]] = []
    for path in result_paths:
        payload = _object(path)
        errors = list(validator.iter_errors(payload))
        if errors:
            raise FormalResultFirewallError(f"formal result schema failure: {path.name}")
        payload["schema_valid"] = True
        rows.append(payload)
    boundary["formal_result_file_count"] = len(rows)
    return rows, boundary
