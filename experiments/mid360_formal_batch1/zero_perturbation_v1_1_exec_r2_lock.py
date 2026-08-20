"""Execution-lock revision 2 builder for frozen FMB1 R1 science.

Revision 2 changes only execution-control bindings.  It authenticates the
byte-identical revision-1 lock and scientific bindings, adds the qualified
one-time authorization infrastructure, and remains closed/zero-trial.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .zero_perturbation_v1_1_r1_environment import verify_environment_manifest


LOCK_SCHEMA = "mid360_fmb1_zero_perturbation_formal_lock_v1_1_exec_r2"
LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.json"
LOCK_SHA_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.sha256"
OLD_LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1.json"
OLD_LOCK_SHA256 = "a191894d76f5c3bd3be2aa53f4a2b8bb6b641b77d9603a6f300993b49bbf9c0c"
OLD_LOCK_FINGERPRINT = "9fcd01bd439e1c4e8c2bd0a220a3af1d16c394177bb17197850c4d769109cbba"
AMENDMENT_ID = "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1"
AUTHORITATIVE_RUNTIME_ROOT = (
    "zero_perturbation_runtime/"
    "mid360_zero_perturbation_v1_1_formal_execution_v1"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

OLD_LOCK_PATH = (
    "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/"
    "formal_batch1_zero_perturbation_lock_v1_1.json"
)
OLD_FINGERPRINT_PATH = (
    "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/lock_fingerprint.json"
)
R2_RESULTS_DIR = "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r2_lock"

EXECUTION_PATHS = {
    "execution_runner": "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_runner.py",
    "execution_runner_cli": "tools/mid360_formal_batch1/run_zero_perturbation_v1_1_r1.py",
    "execution_experiments_package_init": "experiments/__init__.py",
    "execution_mid360_formal_batch1_package_init": "experiments/mid360_formal_batch1/__init__.py",
    "execution_phase_a_harness_package_init": "src/phase_a_harness/__init__.py",
    "execution_environment": "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_environment.py",
    "execution_result_validator": "experiments/mid360_formal_batch1/zero_perturbation_r1_trial_assets.py",
    "execution_open3d_backend": "src/phase_a_harness/open3d_backend.py",
    "execution_pcl_backend": "src/phase_a_harness/pcl_backend.py",
    "execution_common_association": "src/phase_a_harness/common_association_analysis.py",
    "execution_rotation_metrics": "src/phase_a_harness/rotation_metrics.py",
    "execution_metrics": "src/phase_a_harness/metrics.py",
    "execution_types": "src/phase_a_harness/types.py",
    "execution_authorization_producer": "experiments/mid360_formal_batch1/authorization/formal_registration_authorization.py",
    "execution_authorization_verifier": "experiments/mid360_formal_batch1/authorization/formal_registration_authorization_verify.py",
    "execution_authorization_lifecycle": "experiments/mid360_formal_batch1/authorization/authorization_lifecycle.py",
    "execution_authorization_producer_cli": "tools/mid360_formal_batch1/issue_formal_registration_authorization.py",
    "execution_authorization_verifier_cli": "tools/mid360_formal_batch1/verify_formal_registration_authorization.py",
    "execution_exec_r2_lock_builder": "experiments/mid360_formal_batch1/zero_perturbation_v1_1_exec_r2_lock.py",
    "execution_exec_r2_lock_verifier": "experiments/mid360_formal_batch1/zero_perturbation_v1_1_exec_r2_verify.py",
    "execution_exec_r2_lock_issuer_cli": "tools/mid360_formal_batch1/issue_zero_perturbation_v1_1_exec_r2_lock.py",
    "execution_exec_r2_lock_verifier_cli": "tools/mid360_formal_batch1/verify_zero_perturbation_v1_1_exec_r2_lock.py",
}

CONTROL_BINDING_PATHS = {
    "authorization_schema": "experiments/mid360_formal_batch1/authorization/formal_registration_authorization_schema_v1.json",
    "authorization_contract": "experiments/mid360_formal_batch1/authorization/authorization_contract_v1.md",
    "old_lock_file": OLD_LOCK_PATH,
    "old_lock_fingerprint": OLD_FINGERPRINT_PATH,
    "old_lock_supersession": "results/mid360_formal_batch1/history/zero_perturbation_v1_1_exec_r1_lock.SUPERSEDED.json",
    "blocked_attempt_record": "archive/formal_execution_attempts/blocked_attempt_001_missing_authorization_infrastructure/BLOCKED_EXECUTION_ATTEMPT.json",
    "blocked_attempt_no_registration": "archive/formal_execution_attempts/blocked_attempt_001_missing_authorization_infrastructure/NO_REGISTRATION_ATTESTATION.json",
    "blocked_attempt_sha256sums": "archive/formal_execution_attempts/blocked_attempt_001_missing_authorization_infrastructure/SHA256SUMS",
    "final_dataset_independent_verification": "results/mid360_formal_batch1/final_dataset_v1/independent_verification.json",
    "execution_control_patch_report": "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r2_lock/execution_control_patch_report.json",
    "authorization_lifecycle_test_report": "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r2_lock/authorization_lifecycle_test_report.json",
    "environment_manifest": "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r2_lock/environment_manifest.json",
}

LOCK_CORE_FILENAMES = (
    LOCK_FILENAME, LOCK_SHA_FILENAME, "lock_inventory.csv", "lock_fingerprint.json",
    "NO_REGISTRATION_ATTESTATION.json", "environment_manifest.json",
    "execution_control_patch_report.json", "authorization_lifecycle_test_report.json",
)


class ExecR2LockError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise ExecR2LockError(f"FMB1_EXEC_R2_LOCK_FAIL: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        _fail(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, relative: str, label: str) -> Path:
    path = root / relative
    cursor = root
    for part in Path(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"{label} uses symlink component")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes repository")
    if not resolved.is_file():
        _fail(f"{label} is not a regular file")
    return resolved


def _assert_no_execution(root: Path) -> None:
    runtime = root / AUTHORITATIVE_RUNTIME_ROOT
    if runtime.exists() or runtime.is_symlink():
        _fail("authoritative exec-r2 runtime already exists")
    execution_results = root / "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1"
    if execution_results.exists() or execution_results.is_symlink():
        _fail("formal execution results already exist")
    for key in ("FORMAL_REGISTRATION_AUTHORIZED", "FORMAL_ICP_UNLOCKED"):
        if os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}:
            _fail(f"authorization environment is active: {key}")


def _verify_old_lock(root: Path) -> Mapping[str, Any]:
    path = _resolve(root, OLD_LOCK_PATH, "old lock")
    if sha256_file(path) != OLD_LOCK_SHA256:
        _fail("old lock bytes changed")
    fingerprint = _json(_resolve(root, OLD_FINGERPRINT_PATH, "old fingerprint"))
    if fingerprint.get("lock_fingerprint") != OLD_LOCK_FINGERPRINT:
        _fail("old lock fingerprint changed")
    supersession = _json(_resolve(root, CONTROL_BINDING_PATHS["old_lock_supersession"], "old lock supersession"))
    if (
        supersession.get("OLD_LOCK_STATUS")
        != "SUPERSEDED_BEFORE_EXECUTION_BY_EXECUTION_CONTROL_PATCH"
        or supersession.get("OLD_LOCK_ACTUAL_FORMAL_TRIALS") != 0
        or supersession.get("old_lock_file_modified") is not False
    ):
        _fail("old lock supersession semantics differ")
    return _json(path)


def _verify_execution_commit(root: Path, commit: str, paths: Mapping[str, Path]) -> None:
    if COMMIT_RE.fullmatch(commit) is None:
        _fail("execution code commit is invalid")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if head != commit:
        _fail("execution code commit must be current HEAD at R2 lock build")
    for name in EXECUTION_PATHS:
        relative = paths[name].relative_to(root).as_posix()
        content = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
        if hashlib.sha256(content).hexdigest() != sha256_file(paths[name]):
            _fail(f"execution control differs from commit: {name}")


def build_exec_r2_lock(
    repository: Path,
    *,
    execution_code_commit: str,
    issued_at_utc: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(repository).resolve(strict=True)
    _assert_no_execution(root)
    old = _verify_old_lock(root)
    configured: dict[str, str] = {}
    for name, binding in old["bindings"].items():
        if name.startswith("execution_") or name == "environment_manifest":
            continue
        configured[name] = str(binding["repository_relative_path"])
    configured.update(EXECUTION_PATHS)
    configured.update(CONTROL_BINDING_PATHS)
    paths = {name: _resolve(root, relative, name) for name, relative in configured.items()}
    _verify_execution_commit(root, execution_code_commit, paths)
    for name, old_binding in old["bindings"].items():
        if name.startswith("execution_") or name == "environment_manifest":
            continue
        if sha256_file(paths[name]) != old_binding["sha256"]:
            _fail(f"scientific/historical binding changed: {name}")
    patch = _json(paths["execution_control_patch_report"])
    required_false = (
        "SCIENTIFIC_PROTOCOL_CHANGED", "FINAL_DATASET_CHANGED",
        "TRIAL_PLAN_SCIENTIFIC_CONTENT_CHANGED", "BACKEND_PARAMETERS_CHANGED",
    )
    if any(patch.get(key) is not False for key in required_false) or patch.get("ACTUAL_FORMAL_TRIALS") != 0:
        _fail("execution-control patch report does not preserve science")
    lifecycle = _json(paths["authorization_lifecycle_test_report"])
    if (
        lifecycle.get("status") != "PASS"
        or lifecycle.get("AUTHORIZATION_LIFECYCLE_QUALIFIED") is not True
        or lifecycle.get("REAL_FORMAL_TRIALS") != 0
    ):
        _fail("authorization lifecycle fixture report is not PASS/zero-real")
    environment = _json(paths["environment_manifest"])
    verify_environment_manifest(environment, root, remeasure_versions=True)
    final_verify = _json(paths["final_dataset_independent_verification"])
    if final_verify.get("status") != "PASS" or final_verify.get("pass") is not True:
        _fail("final dataset independent verification is not PASS")
    inventory = []
    for name in sorted(paths):
        path = paths[name]
        inventory.append({
            "binding_name": name,
            "repository_relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    lock = {
        "schema": LOCK_SCHEMA,
        "lock_id": "FMB1_ZERO_PERTURBATION_V1_1_R1_FORMAL_EXECUTION_LOCK_EXEC_R2",
        "execution_lock_revision": 2,
        "supersedes_lock_fingerprint": OLD_LOCK_FINGERPRINT,
        "amendment_id": AMENDMENT_ID,
        "track_id": "ZERO_PERTURBATION_TRACK",
        "status": "ISSUED_AWAITING_SEPARATE_AUTHORIZATION",
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "execution_code_commit": execution_code_commit,
        "authoritative_runtime_root": AUTHORITATIVE_RUNTIME_ROOT,
        "binding_inventory_sha256": hashlib.sha256(canonical_json_bytes(inventory)).hexdigest(),
        "bindings": {row["binding_name"]: row for row in inventory},
        "counts": {
            "scene_count": 6, "station_count": 18, "snapshot_count": 180,
            "planned_open3d_trials": 180, "planned_pcl_trials": 180,
            "planned_total_trials": 360,
        },
        "FORMAL_LOCK_ISSUED": True,
        "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION": True,
        "AUTHORIZATION_PRODUCER_READY": True,
        "INDEPENDENT_AUTHORIZATION_VERIFIER_READY": True,
        "AUTHORIZATION_LIFECYCLE_QUALIFIED": True,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "MEASUREMENT_FINAL_RESULT": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "registration_execution_count": 0,
    }
    return lock, inventory


def _inventory_csv(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=(
        "binding_name", "repository_relative_path", "sha256", "bytes"
    ))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            _fail(f"refusing to overwrite different R2 lock artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content); temporary.replace(path)


def write_exec_r2_lock(
    repository: Path, output_dir: Path, lock: Mapping[str, Any],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    root = Path(repository).resolve(strict=True)
    output = Path(output_dir).resolve()
    lock_bytes = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode("utf-8")
    inventory_bytes = _inventory_csv(inventory)
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    material = {
        "lock_file_sha256": lock_sha,
        "lock_inventory_file_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "execution_code_commit": lock["execution_code_commit"],
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    fingerprint_payload = {
        "schema": "mid360_fmb1_zero_perturbation_lock_fingerprint_v1_1_exec_r2",
        **material,
        "lock_fingerprint": fingerprint,
        "execution_lock_revision": 2,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    }
    no_registration = {
        "schema": "mid360_fmb1_exec_r2_no_registration_attestation_v1",
        "status": "PASS", "pass": True,
        "old_lock_sha256": OLD_LOCK_SHA256,
        "old_lock_fingerprint": OLD_LOCK_FINGERPRINT,
        "new_lock_file_sha256": lock_sha,
        "new_lock_fingerprint": fingerprint,
        "open3d_registration_call_count": 0,
        "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0,
        "formal_trial_count": 0,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
    }
    files = {
        LOCK_FILENAME: lock_bytes,
        LOCK_SHA_FILENAME: f"{lock_sha}  {LOCK_FILENAME}\n".encode("ascii"),
        "lock_inventory.csv": inventory_bytes,
        "lock_fingerprint.json": (json.dumps(fingerprint_payload, indent=2, sort_keys=True) + "\n").encode(),
        "NO_REGISTRATION_ATTESTATION.json": (json.dumps(no_registration, indent=2, sort_keys=True) + "\n").encode(),
    }
    for name, content in files.items():
        _write_once(output / name, content)
    finalize_core_checksums(output)
    return {
        "FORMAL_LOCK_ISSUED": True,
        "execution_lock_revision": 2,
        "lock_file_sha256": lock_sha,
        "lock_fingerprint": fingerprint,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    }


def finalize_core_checksums(output_dir: Path) -> Path:
    output = Path(output_dir).resolve(strict=True)
    missing = [name for name in LOCK_CORE_FILENAMES if not (output / name).is_file()]
    if missing:
        _fail(f"R2 core files missing: {missing}")
    content = "".join(
        f"{sha256_file(output / name)}  {name}\n" for name in LOCK_CORE_FILENAMES
    ).encode("ascii")
    path = output / "LOCK_CORE_SHA256SUMS"
    _write_once(path, content)
    return path


def finalize_release_checksums(output_dir: Path) -> Path:
    output = Path(output_dir).resolve(strict=True)
    files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS" and not path.name.endswith(".tmp")
    )
    content = "".join(f"{sha256_file(path)}  {path.name}\n" for path in files).encode("ascii")
    target = output / "SHA256SUMS"
    temporary = output / "SHA256SUMS.tmp"
    temporary.write_bytes(content); temporary.replace(target)
    return target
