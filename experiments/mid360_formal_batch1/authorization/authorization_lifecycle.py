"""Write-once lifecycle markers for one FMB1 formal authorization."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .formal_registration_authorization_verify import (
    AUTHORIZATION_FILENAME,
    IN_USE_FILENAME,
    RECEIPT_FILENAME,
    canonical_json_bytes,
    sha256_file,
)


class AuthorizationLifecycleError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise AuthorizationLifecycleError(f"FMB1_AUTHORIZATION_LIFECYCLE_FAIL: {message}")


def _atomic_create_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _fail(f"write-once lifecycle artifact already exists: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    if temporary.exists():
        _fail(f"stale lifecycle temporary exists: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_create_bytes(path, canonical_json_bytes(payload))


def mark_authorization_in_use(
    authorization_path: Path,
    *,
    lock_fingerprint: str,
    runtime_relative: str,
    started_at_utc: str | None = None,
) -> dict[str, Any]:
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    payload = {
        "schema": "mid360_fmb1_authorization_in_use_exec_r2_v1",
        "state": "IN_USE",
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": sha256_file(authorization_path),
        "lock_fingerprint": lock_fingerprint,
        "runtime_root": runtime_relative,
        "started_at_utc": started_at_utc or datetime.now(timezone.utc).isoformat(),
        "reusable_for_new_fresh": False,
    }
    _atomic_create(authorization_path.parent / IN_USE_FILENAME, payload)
    return payload


def consume_authorization(
    authorization_path: Path,
    *,
    lock_fingerprint: str,
    first_backend_invocation_utc: str,
    last_backend_invocation_utc: str,
    actual_open3d_trials: int,
    actual_pcl_trials: int,
    execution_result_manifest_sha256: str,
    failed_attempt: bool = False,
) -> dict[str, Any]:
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    marker_path = authorization_path.parent / IN_USE_FILENAME
    if marker_path.is_symlink() or not marker_path.is_file():
        _fail("authorization cannot be consumed without an IN_USE marker")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("schema") != "mid360_fmb1_authorization_in_use_exec_r2_v1"
        or marker.get("state") != "IN_USE"
        or marker.get("authorization_id") != authorization.get("authorization_id")
        or marker.get("authorization_sha256") != sha256_file(authorization_path)
        or marker.get("lock_fingerprint") != lock_fingerprint
        or marker.get("reusable_for_new_fresh") is not False
    ):
        _fail("authorization IN_USE marker is not bound to this consumption")
    if (
        isinstance(actual_open3d_trials, bool)
        or isinstance(actual_pcl_trials, bool)
        or actual_open3d_trials < 0 or actual_pcl_trials < 0
        or (not failed_attempt and (actual_open3d_trials, actual_pcl_trials) != (180, 180))
    ):
        _fail("authorization consumption trial counts differ")
    if re.fullmatch(r"[0-9a-f]{64}", execution_result_manifest_sha256) is None:
        _fail("authorization consumption manifest SHA is invalid")
    for label, value in (
        ("first_backend_invocation_utc", first_backend_invocation_utc),
        ("last_backend_invocation_utc", last_backend_invocation_utc),
    ):
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            _fail(f"{label} is invalid")
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            _fail(f"{label} lacks timezone")
    total = actual_open3d_trials + actual_pcl_trials
    payload = {
        "schema": "mid360_fmb1_authorization_consumption_receipt_exec_r2_v1",
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": sha256_file(authorization_path),
        "lock_fingerprint": lock_fingerprint,
        "first_backend_invocation_utc": first_backend_invocation_utc,
        "last_backend_invocation_utc": last_backend_invocation_utc,
        "actual_open3d_trials": actual_open3d_trials,
        "actual_pcl_trials": actual_pcl_trials,
        "actual_total_trials": total,
        "execution_result_manifest_sha256": execution_result_manifest_sha256,
        "consumed": True,
        "state": "CONSUMED_BY_FAILED_ATTEMPT" if failed_attempt else "CONSUMED",
        "reusable": False,
        "FURTHER_REGISTRATION_AUTHORIZED": False,
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_create(authorization_path.parent / RECEIPT_FILENAME, payload)
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    sha_path = authorization_path.parent / "authorization_consumption_receipt.sha256"
    _atomic_create_bytes(
        sha_path, f"{digest}  {RECEIPT_FILENAME}\n".encode("ascii")
    )
    return payload
