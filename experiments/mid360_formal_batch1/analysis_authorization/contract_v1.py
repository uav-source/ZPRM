"""Frozen identities and path rules for analysis-authorization control v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LOCKED_ANALYSIS_CODE_COMMIT = "215a7961ed93dac9cf691e1e8ccd99d7ee868175"
LOCK_V1_RELEASE_COMMIT = "74b7b4f19f0d756158dedfafcb396309988361be"
LOCK_V1_SHA256 = "ab3acf55db181772a55a8af59b592902c2da9d0b5a1b50e3511f79c41438e21a"
LOCK_V1_FINGERPRINT = "24fa8bf5d6713b744a1ccbac8fa46ec4bd6f8af7227b248e7db371f2a0c7e300"
RAW_EXECUTION_COMMIT = "059e39533991d929a97ab208ad738643af82d09a"
POSTRUN_VERIFICATION_COMMIT = "18bb94e62761f5193da8cdc5509c470cb4983244"
C2_CLARIFICATION_COMMIT = "364c0ae76801c373d9ea6a4d9dd36e2fdf211e09"
R3_LOCK_FINGERPRINT = "fd601e8daf62c488a3a079beea05f65c9bd283399ae4d7f3acf16e63fc473a6d"
OUTPUT_SCHEMA_SHA256 = "707e2d5f7030c090a30ebb5cfb74f07373e04a223ac1beae94e53bc502eec23e"

FORMAL_RESULTS_RELATIVE = (
    "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1"
)
POSTRUN_RELATIVE = (
    "results/mid360_formal_batch1/zero_perturbation_v1_1_postrun_verification_v1"
)
ANALYSIS_OUTPUT_RELATIVE = (
    "results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1"
)
LOCK_V1_RELATIVE = (
    "results/mid360_formal_batch1/zero_perturbation_locked_analysis_lock_v1"
)
LOCK_V2_RELATIVE = (
    "results/mid360_formal_batch1/zero_perturbation_locked_analysis_lock_v2"
)
AUTHORIZATION_SCHEMA_RELATIVE = (
    "experiments/mid360_formal_batch1/analysis_authorization/"
    "formal_analysis_authorization_schema_v1.json"
)

CANDIDATE_NAME = "formal_analysis_authorization.candidate.json"
CANDIDATE_SHA_NAME = "formal_analysis_authorization.candidate.sha256"
VERIFICATION_NAME = "formal_analysis_authorization_verification.json"
FINAL_NAME = "formal_analysis_authorization.json"
FINAL_SHA_NAME = "formal_analysis_authorization.sha256"
IN_USE_NAME = "formal_analysis_authorization_in_use.json"
RECEIPT_NAME = "formal_analysis_authorization_consumption_receipt.json"
INVALIDATION_NAME = "formal_analysis_authorization_invalidation.json"

CONTROL_CODE_PATHS = (
    "experiments/mid360_formal_batch1/analysis_authorization/__init__.py",
    "experiments/mid360_formal_batch1/analysis_authorization/contract_v1.py",
    "experiments/mid360_formal_batch1/analysis_authorization/formal_analysis_authorization_schema_v1.json",
    "experiments/mid360_formal_batch1/analysis_authorization/formal_analysis_authorization_contract_v1.md",
    "experiments/mid360_formal_batch1/analysis_authorization/formal_analysis_authorization_producer_v1.py",
    "experiments/mid360_formal_batch1/analysis_authorization/formal_analysis_authorization_verify_v1.py",
    "experiments/mid360_formal_batch1/analysis_authorization/formal_analysis_authorization_lifecycle_v1.py",
    "experiments/mid360_formal_batch1/analysis_authorization/authorized_analysis_runner_v1.py",
    "experiments/mid360_formal_batch1/analysis_authorization/analysis_lock_v2.py",
    "experiments/mid360_formal_batch1/analysis_authorization/analysis_lock_v2_verify.py",
    "tools/mid360_formal_batch1/prepare_formal_analysis_authorization_v1.py",
    "tools/mid360_formal_batch1/verify_formal_analysis_authorization_v1.py",
    "tools/mid360_formal_batch1/issue_verified_formal_analysis_authorization_v1.py",
    "tools/mid360_formal_batch1/run_verified_formal_analysis_v1.py",
    "tools/mid360_formal_batch1/issue_zero_perturbation_locked_analysis_lock_v2.py",
    "tools/mid360_formal_batch1/verify_zero_perturbation_locked_analysis_lock_v2.py",
)


class AuthorizationControlError(RuntimeError):
    """Raised when an authorization-control boundary does not match exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizationControlError(f"invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise AuthorizationControlError(f"expected JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any], *, write_once: bool = True) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if write_once or path.read_text(encoding="utf-8") != encoded:
            raise AuthorizationControlError(f"write-once artifact already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise AuthorizationControlError(f"temporary collision: {temporary}")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def canonical_fingerprint(payload: dict[str, Any]) -> str:
    item = dict(payload)
    item.pop("analysis_lock_fingerprint", None)
    encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def ensure_regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise AuthorizationControlError(f"regular file required: {path}")
    return path.resolve(strict=True)


def ensure_no_symlink_components(path: Path) -> None:
    current = path
    while True:
        if current.exists() and current.is_symlink():
            raise AuthorizationControlError(f"symlink path component forbidden: {current}")
        if current == current.parent:
            return
        current = current.parent


def repository_paths(repository: Path) -> dict[str, Path]:
    root = repository.resolve(strict=True)
    return {
        "repository": root,
        "formal_results_root": (root / FORMAL_RESULTS_RELATIVE).resolve(strict=True),
        "postrun_verification_root": (root / POSTRUN_RELATIVE).resolve(strict=True),
        "analysis_output_root": (root / ANALYSIS_OUTPUT_RELATIVE).resolve(strict=False),
        "lock_v1_dir": (root / LOCK_V1_RELATIVE).resolve(strict=True),
        "lock_v2_dir": (root / LOCK_V2_RELATIVE).resolve(strict=False),
    }
