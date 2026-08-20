"""Canonical one-time wrapper around the already-frozen analysis CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from experiments.mid360_formal_batch1.locked_analysis.contract_v1 import sha256_file
from experiments.mid360_formal_batch1.locked_analysis.formal_firewall_v1 import (
    FormalReadRequest,
    validate_formal_read_request,
)

from .contract_v1 import AuthorizationControlError, RECEIPT_NAME
from .formal_analysis_authorization_lifecycle_v1 import (
    consume_authorization,
    mark_authorization_in_use,
)


def run_fixture_lifecycle(
    request: FormalReadRequest,
    *,
    fixture_callback: Callable[[], str],
    first_read_utc: str,
    completed_utc: str,
) -> dict:
    """Qualification-only wrapper; callback must not read real FMB1 values."""

    validate_formal_read_request(request)
    if (request.analysis_lock_dir / RECEIPT_NAME).exists():
        raise AuthorizationControlError("authorization already consumed")
    mark_authorization_in_use(request.analysis_lock_dir,
                              first_real_value_read_utc=first_read_utc)
    try:
        manifest_sha = fixture_callback()
    except Exception:
        consume_authorization(
            request.analysis_lock_dir, analysis_completed_utc=completed_utc,
            analysis_output_manifest_sha256=None, success=False,
        )
        raise
    return consume_authorization(
        request.analysis_lock_dir, analysis_completed_utc=completed_utc,
        analysis_output_manifest_sha256=manifest_sha, success=True,
    )


def run_authoritative_cli_once(
    request: FormalReadRequest,
    *,
    frozen_python: Path,
    frozen_cli: Path,
) -> dict:
    """Future production entrypoint; not called by the infrastructure task."""

    validate_formal_read_request(request)
    lock_dir = request.analysis_lock_dir.resolve(strict=True)
    if (lock_dir / RECEIPT_NAME).exists():
        raise AuthorizationControlError("authorization already consumed")
    mark_authorization_in_use(lock_dir)
    command = [
        str(frozen_python), str(frozen_cli),
        "--formal-results-root", str(request.formal_results_root),
        "--postrun-verification-root", str(request.postrun_verification_root),
        "--analysis-lock-dir", str(request.analysis_lock_dir),
        "--analysis-code-commit", request.analysis_code_commit,
        "--output-dir", str(request.output_dir),
        "--confirm-read-frozen-formal-results",
    ]
    process = subprocess.run(command, check=False)
    if process.returncode:
        consume_authorization(
            lock_dir, analysis_output_manifest_sha256=None, success=False,
        )
        raise AuthorizationControlError(
            f"frozen analysis CLI failed after unblinding: {process.returncode}"
        )
    summary = request.output_dir / "analysis_summary.json"
    if not summary.is_file():
        consume_authorization(
            lock_dir, analysis_output_manifest_sha256=None, success=False,
        )
        raise AuthorizationControlError("analysis summary missing after frozen CLI")
    return consume_authorization(
        lock_dir, analysis_output_manifest_sha256=sha256_file(summary), success=True,
    )
