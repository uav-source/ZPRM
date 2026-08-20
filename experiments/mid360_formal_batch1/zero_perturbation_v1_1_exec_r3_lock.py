"""Build the closed FMB1 Zero-Perturbation Exec-R3 lock.

Exec-R3 changes execution-control provenance only.  Every binding carries an
explicit class, verification source, and commit role.  No identifier or path
text is used to infer provenance.
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

from .authorization.binding_provenance import (
    EXECUTION_CODE,
    EXECUTION_CODE_COMMIT,
    FROZEN_ENVIRONMENT_CONTRACT,
    FROZEN_SCIENCE_OR_DATA,
    GIT_BLOB_AT_COMMIT,
    INHERITED_LOCK_SHA256,
    LOCK_RELEASE_COMMIT,
    LOCK_RELEASE_EVIDENCE,
    NO_COMMIT,
    ENVIRONMENT_OR_BINARY,
    validate_binding_metadata,
)
from .zero_perturbation_v1_1_r1_environment import verify_environment_manifest


LOCK_SCHEMA = "mid360_fmb1_zero_perturbation_formal_lock_v1_1_exec_r3"
LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r3.json"
LOCK_SHA_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r3.sha256"
LOCK_ID = "FMB1_ZERO_PERTURBATION_V1_1_R1_FORMAL_EXECUTION_LOCK_EXEC_R3"
R3_RESULTS_DIR = "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r3_lock"
R2_RESULTS_DIR = "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r2_lock"
R2_LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.json"
R2_LOCK_SHA256 = "fe8846fd80e879b497c8ef68568aa9f83a5ad339e0a29e8113afc36577eb3e4c"
R2_LOCK_FINGERPRINT = "5db11e845c4b1307d40ea9ba3598ead6ba2778a0370b0ce853c8408e8d21e8c5"
AMENDMENT_ID = "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1"
AUTHORITATIVE_RUNTIME_ROOT = (
    "zero_perturbation_runtime/"
    "mid360_zero_perturbation_v1_1_formal_execution_v1"
)
FORMAL_RESULTS_ROOT = (
    "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


EXECUTION_CODE_PATHS = {
    "runner": "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_runner.py",
    "runner_cli": "tools/mid360_formal_batch1/run_zero_perturbation_v1_1_r1.py",
    "experiments_package_init": "experiments/__init__.py",
    "mid360_formal_batch1_package_init": "experiments/mid360_formal_batch1/__init__.py",
    "phase_a_harness_package_init": "src/phase_a_harness/__init__.py",
    "execution_environment": "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_environment.py",
    "result_validator": "experiments/mid360_formal_batch1/zero_perturbation_r1_trial_assets.py",
    "open3d_backend": "src/phase_a_harness/open3d_backend.py",
    "pcl_backend": "src/phase_a_harness/pcl_backend.py",
    "common_association": "src/phase_a_harness/common_association_analysis.py",
    "rotation_metrics": "src/phase_a_harness/rotation_metrics.py",
    "metrics": "src/phase_a_harness/metrics.py",
    "execution_types": "src/phase_a_harness/types.py",
    "binding_provenance_contract": "experiments/mid360_formal_batch1/authorization/binding_provenance.py",
    "authorization_schema": "experiments/mid360_formal_batch1/authorization/formal_registration_authorization_schema_v1.json",
    "authorization_contract": "experiments/mid360_formal_batch1/authorization/authorization_contract_v1.md",
    "authorization_producer": "experiments/mid360_formal_batch1/authorization/formal_registration_authorization.py",
    "authorization_verifier": "experiments/mid360_formal_batch1/authorization/formal_registration_authorization_verify.py",
    "authorization_lifecycle": "experiments/mid360_formal_batch1/authorization/authorization_lifecycle.py",
    "authorization_producer_cli": "tools/mid360_formal_batch1/issue_formal_registration_authorization.py",
    "authorization_verifier_cli": "tools/mid360_formal_batch1/verify_formal_registration_authorization.py",
    "exec_r3_lock_builder": "experiments/mid360_formal_batch1/zero_perturbation_v1_1_exec_r3_lock.py",
    "exec_r3_lock_verifier": "experiments/mid360_formal_batch1/zero_perturbation_v1_1_exec_r3_verify.py",
    "exec_r3_lock_issuer_cli": "tools/mid360_formal_batch1/issue_zero_perturbation_v1_1_exec_r3_lock.py",
    "exec_r3_lock_verifier_cli": "tools/mid360_formal_batch1/verify_zero_perturbation_v1_1_exec_r3_lock.py",
}

# These exact R2 bindings are replaced by R3 execution or release bindings.
# This is an explicit historical migration list, never a naming heuristic.
REPLACED_R2_BINDING_IDS = {
    "authorization_contract",
    "authorization_lifecycle_test_report",
    "authorization_schema",
    "environment_manifest",
    "execution_authorization_lifecycle",
    "execution_authorization_producer",
    "execution_authorization_producer_cli",
    "execution_authorization_verifier",
    "execution_authorization_verifier_cli",
    "execution_common_association",
    "execution_control_patch_report",
    "execution_environment",
    "execution_exec_r2_lock_builder",
    "execution_exec_r2_lock_issuer_cli",
    "execution_exec_r2_lock_verifier",
    "execution_exec_r2_lock_verifier_cli",
    "execution_experiments_package_init",
    "execution_metrics",
    "execution_mid360_formal_batch1_package_init",
    "execution_open3d_backend",
    "execution_pcl_backend",
    "execution_phase_a_harness_package_init",
    "execution_result_validator",
    "execution_rotation_metrics",
    "execution_runner",
    "execution_runner_cli",
    "execution_types",
    "backend_parameter_contract",
    "pcl_executable",
}

RELEASE_EVIDENCE_PATHS = {
    "authorization_binding_provenance_defect_audit_json":
        f"{R3_RESULTS_DIR}/authorization_binding_provenance_defect_audit.json",
    "authorization_binding_provenance_defect_audit_md":
        f"{R3_RESULTS_DIR}/authorization_binding_provenance_defect_audit.md",
    # Deliberately begins with execution_: its explicit class controls routing.
    "execution_control_patch_report":
        f"{R3_RESULTS_DIR}/r3_execution_control_fix_report.json",
    "authorization_lifecycle_test_report":
        f"{R3_RESULTS_DIR}/r3_authorization_fixture_qualification.json",
    "r2_lock_supersession":
        "results/mid360_formal_batch1/history/zero_perturbation_v1_1_exec_r2_lock.SUPERSEDED.json",
    "attempt_003_authorization_invalidation":
        f"{R3_RESULTS_DIR}/AUTHORIZATION_INVALIDATION_RECORD.json",
    "superseded_r2_lock": f"{R2_RESULTS_DIR}/{R2_LOCK_FILENAME}",
    "superseded_r2_fingerprint": f"{R2_RESULTS_DIR}/lock_fingerprint.json",
}

ENVIRONMENT_PATHS = {
    "environment_manifest": f"{R2_RESULTS_DIR}/environment_manifest.json",
    "pcl_executable": "bin/pcl_point_to_plane_cli",
    "backend_parameter_contract": "frozen_assets/backend_parameter_contract.json",
}

LOCK_CORE_FILENAMES = (
    LOCK_FILENAME,
    LOCK_SHA_FILENAME,
    "lock_inventory.csv",
    "lock_fingerprint.json",
    "NO_REGISTRATION_ATTESTATION.json",
    "authorization_binding_provenance_defect_audit.json",
    "r3_execution_control_fix_report.json",
    "r3_authorization_fixture_qualification.json",
    "AUTHORIZATION_INVALIDATION_RECORD.json",
)


class ExecR3LockError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise ExecR3LockError(f"FMB1_EXEC_R3_LOCK_FAIL: {message}")


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
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        _fail(f"{label} missing/escapes repository: {error}")
    if not resolved.is_file():
        _fail(f"{label} is not a regular file")
    return resolved


def _assert_closed(root: Path) -> None:
    for relative in (AUTHORITATIVE_RUNTIME_ROOT, FORMAL_RESULTS_ROOT):
        path = root / relative
        if path.exists() or path.is_symlink():
            _fail(f"formal runtime/result path already exists: {relative}")
    for key in ("FORMAL_REGISTRATION_AUTHORIZED", "FORMAL_ICP_UNLOCKED"):
        if os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}:
            _fail(f"authorization environment is active: {key}")


def _binding_row(
    root: Path, binding_id: str, path: Path, binding_class: str,
) -> dict[str, Any]:
    if binding_class == EXECUTION_CODE:
        source, role = GIT_BLOB_AT_COMMIT, EXECUTION_CODE_COMMIT
    elif binding_class == LOCK_RELEASE_EVIDENCE:
        source, role = GIT_BLOB_AT_COMMIT, LOCK_RELEASE_COMMIT
    elif binding_class == FROZEN_SCIENCE_OR_DATA:
        source, role = INHERITED_LOCK_SHA256, NO_COMMIT
    elif binding_class == ENVIRONMENT_OR_BINARY:
        source, role = FROZEN_ENVIRONMENT_CONTRACT, NO_COMMIT
    else:
        _fail(f"unknown binding class: {binding_class}")
    row = {
        "binding_id": binding_id,
        "repository_relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "binding_class": binding_class,
        "verification_source": source,
        "commit_role": role,
    }
    try:
        validate_binding_metadata(binding_id, row)
    except ValueError as error:
        _fail(str(error))
    return row


def _verify_execution_commit(
    root: Path, execution_code_commit: str, paths: Mapping[str, Path],
) -> None:
    if COMMIT_RE.fullmatch(execution_code_commit) is None:
        _fail("execution code commit is invalid")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if head != execution_code_commit:
        _fail("execution-code commit must be current HEAD at R3 build")
    for binding_id, path in paths.items():
        relative = path.relative_to(root).as_posix()
        try:
            content = subprocess.run(
                ["git", "show", f"{execution_code_commit}:{relative}"],
                cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout
        except subprocess.CalledProcessError:
            _fail(f"execution binding absent from execution commit: {binding_id}")
        if hashlib.sha256(content).hexdigest() != sha256_file(path):
            _fail(f"execution binding differs from commit: {binding_id}")


def build_exec_r3_lock(
    repository: Path,
    *,
    execution_code_commit: str,
    issued_at_utc: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(repository).resolve(strict=True)
    _assert_closed(root)
    r2_path = _resolve(root, f"{R2_RESULTS_DIR}/{R2_LOCK_FILENAME}", "R2 lock")
    if sha256_file(r2_path) != R2_LOCK_SHA256:
        _fail("R2 lock bytes changed")
    r2 = _json(r2_path)
    r2_fingerprint = _json(_resolve(root, f"{R2_RESULTS_DIR}/lock_fingerprint.json", "R2 fingerprint"))
    if r2_fingerprint.get("lock_fingerprint") != R2_LOCK_FINGERPRINT:
        _fail("R2 fingerprint changed")

    execution_paths = {
        name: _resolve(root, relative, name)
        for name, relative in EXECUTION_CODE_PATHS.items()
    }
    _verify_execution_commit(root, execution_code_commit, execution_paths)

    inventory: list[dict[str, Any]] = []
    for binding_id, binding in sorted(r2["bindings"].items()):
        if binding_id in REPLACED_R2_BINDING_IDS:
            continue
        path = _resolve(root, str(binding["repository_relative_path"]), binding_id)
        if sha256_file(path) != binding["sha256"] or path.stat().st_size != int(binding["bytes"]):
            _fail(f"inherited R2 science/data binding changed: {binding_id}")
        inventory.append(_binding_row(root, binding_id, path, FROZEN_SCIENCE_OR_DATA))
    for binding_id, path in sorted(execution_paths.items()):
        inventory.append(_binding_row(root, binding_id, path, EXECUTION_CODE))
    for binding_id, relative in sorted(RELEASE_EVIDENCE_PATHS.items()):
        inventory.append(_binding_row(
            root, binding_id, _resolve(root, relative, binding_id),
            LOCK_RELEASE_EVIDENCE,
        ))
    for binding_id, relative in sorted(ENVIRONMENT_PATHS.items()):
        inventory.append(_binding_row(
            root, binding_id, _resolve(root, relative, binding_id),
            ENVIRONMENT_OR_BINARY,
        ))
    if len({row["binding_id"] for row in inventory}) != len(inventory):
        _fail("R3 binding IDs are duplicated")

    env = next(row for row in inventory if row["binding_id"] == "environment_manifest")
    verify_environment_manifest(_json(root / env["repository_relative_path"]), root, remeasure_versions=True)
    final = next(row for row in inventory if row["binding_id"] == "final_dataset_independent_verification")
    final_report = _json(root / final["repository_relative_path"])
    if final_report.get("status") != "PASS" or final_report.get("pass") is not True:
        _fail("final dataset independent verification is not PASS")

    inventory.sort(key=lambda row: str(row["binding_id"]))
    lock = {
        "schema": LOCK_SCHEMA,
        "lock_id": LOCK_ID,
        "execution_lock_revision": 3,
        "supersedes_lock_revision": 2,
        "supersedes_lock_sha256": R2_LOCK_SHA256,
        "supersedes_lock_fingerprint": R2_LOCK_FINGERPRINT,
        "supersession_reason": "AUTHORIZATION_VERIFIER_BINDING_PROVENANCE_DEFECT",
        "amendment_id": AMENDMENT_ID,
        "track_id": "ZERO_PERTURBATION_TRACK",
        "status": "ISSUED_AWAITING_SEPARATE_AUTHORIZATION",
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "execution_code_commit": execution_code_commit,
        "authoritative_runtime_root": AUTHORITATIVE_RUNTIME_ROOT,
        "formal_results_root": FORMAL_RESULTS_ROOT,
        "binding_provenance_contract": "EXPLICIT_PER_BINDING_V1",
        "prefix_based_provenance_inference": False,
        "binding_inventory_sha256": hashlib.sha256(canonical_json_bytes(inventory)).hexdigest(),
        "bindings": {row["binding_id"]: row for row in inventory},
        "counts": {
            "scene_count": 6,
            "station_count": 18,
            "snapshot_count": 180,
            "planned_open3d_trials": 180,
            "planned_pcl_trials": 180,
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
        "binding_id", "repository_relative_path", "sha256", "bytes",
        "binding_class", "verification_source", "commit_role",
    ))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != content:
            _fail(f"refusing to overwrite different R3 artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_exec_r3_lock(
    repository: Path,
    output_dir: Path,
    lock: Mapping[str, Any],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    Path(repository).resolve(strict=True)
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
        "schema": "mid360_fmb1_zero_perturbation_lock_fingerprint_v1_1_exec_r3",
        **material,
        "lock_fingerprint": fingerprint,
        "execution_lock_revision": 3,
        "binding_provenance_contract": "EXPLICIT_PER_BINDING_V1",
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    }
    no_registration = {
        "schema": "mid360_fmb1_exec_r3_no_registration_attestation_v1",
        "status": "PASS",
        "pass": True,
        "superseded_r2_lock_sha256": R2_LOCK_SHA256,
        "superseded_r2_lock_fingerprint": R2_LOCK_FINGERPRINT,
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
        "lock_fingerprint.json": (
            json.dumps(fingerprint_payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "NO_REGISTRATION_ATTESTATION.json": (
            json.dumps(no_registration, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    for name, content in files.items():
        _write_once(output / name, content)
    finalize_core_checksums(output)
    return {
        "FORMAL_LOCK_ISSUED": True,
        "execution_lock_revision": 3,
        "lock_file_sha256": lock_sha,
        "lock_fingerprint": fingerprint,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    }


def finalize_core_checksums(output_dir: Path) -> Path:
    output = Path(output_dir).resolve(strict=True)
    missing = [name for name in LOCK_CORE_FILENAMES if not (output / name).is_file()]
    if missing:
        _fail(f"R3 core files missing: {missing}")
    content = "".join(
        f"{sha256_file(output / name)}  {name}\n" for name in LOCK_CORE_FILENAMES
    ).encode("ascii")
    target = output / "LOCK_CORE_SHA256SUMS"
    _write_once(target, content)
    return target


def finalize_release_checksums(output_dir: Path) -> Path:
    output = Path(output_dir).resolve(strict=True)
    files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS" and not path.name.endswith(".tmp")
    )
    content = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in files
    ).encode("ascii")
    target = output / "SHA256SUMS"
    temporary = output / "SHA256SUMS.tmp"
    temporary.write_bytes(content)
    temporary.replace(target)
    return target
