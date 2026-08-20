"""Independent verifier for the issued FMB1 locked-analysis lock."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .contract_v1 import (
    C2_COMMIT,
    C2_TAG,
    C2_TAG_OBJECT,
    POSTRUN_VERIFICATION_COMMIT,
    POSTRUN_VERIFICATION_TAG,
    POSTRUN_VERIFIER_CODE_COMMIT,
    PROTECTED_BINDINGS,
    R3_LOCK_FINGERPRINT,
    RAW_EXECUTION_COMMIT,
    RAW_EXECUTION_TAG,
    sha256_file,
)
from .locked_analysis_v1 import run_fixture_analysis
from .lock_v1 import ANALYSIS_CODE_PATHS, LOCK_CORE_FILES


class IndependentLockVerificationError(RuntimeError):
    """Raised on the first independently observed lock mismatch."""


DENIED_IMPORT_ROOTS = {"open3d"}
DENIED_IMPORT_PREFIXES = (
    "phase_a_harness.open3d_backend",
    "phase_a_harness.pcl_backend",
)
DENIED_CALL_NAMES = {
    "registration_icp", "run_open3d_point_to_plane", "run_pcl_point_to_plane",
    "subprocess.run", "subprocess.Popen", "os.system",
}


def _git(repository: Path, *args: str) -> bytes:
    process = subprocess.run(["git", *args], cwd=repository, check=False,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise IndependentLockVerificationError(
            process.stderr.decode("utf-8", "replace").strip()
        )
    return process.stdout


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IndependentLockVerificationError(f"expected object: {path}")
    return payload


def _fingerprint(lock: dict[str, Any]) -> str:
    payload = dict(lock); payload.pop("analysis_lock_fingerprint", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _verify_tag(repository: Path, tag: str, commit: str, *, object_sha: str | None = None) -> None:
    peeled = _git(repository, "rev-parse", f"refs/tags/{tag}^{{}}").decode().strip()
    actual_object = _git(repository, "rev-parse", f"refs/tags/{tag}").decode().strip()
    kind = _git(repository, "cat-file", "-t", actual_object).decode().strip()
    if peeled != commit or kind != "tag" or (object_sha and actual_object != object_sha):
        raise IndependentLockVerificationError(f"annotated tag mismatch: {tag}")


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    item: ast.AST = node.func
    while isinstance(item, ast.Attribute):
        parts.append(item.attr); item = item.value
    if isinstance(item, ast.Name):
        parts.append(item.id)
    return ".".join(reversed(parts))


def _verify_ast(repository: Path) -> dict[str, Any]:
    findings: list[str] = []
    scanned = 0
    for relative in ANALYSIS_CODE_PATHS:
        if not relative.endswith(".py"):
            continue
        scanned += 1
        tree = ast.parse((repository / relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            for name in names:
                if name.split(".")[0] in DENIED_IMPORT_ROOTS or name.startswith(DENIED_IMPORT_PREFIXES):
                    findings.append(f"{relative}:{getattr(node, 'lineno', 0)}:import:{name}")
            if isinstance(node, ast.Call) and _call_name(node) in DENIED_CALL_NAMES:
                findings.append(f"{relative}:{node.lineno}:call:{_call_name(node)}")
    # subprocess is permitted only in the lock builder/verifier for read-only Git
    # inspection.  Scientific modules and the formal read CLI remain backend-free.
    findings = [item for item in findings if not (
        item.startswith("experiments/mid360_formal_batch1/locked_analysis/lock_v1.py")
        or item.startswith("experiments/mid360_formal_batch1/locked_analysis/lock_verify_v1.py")
    )]
    if findings:
        raise IndependentLockVerificationError(f"backend/static firewall findings: {findings}")
    return {"status": "PASS", "scanned_python_file_count": scanned, "finding_count": 0}


def _verify_sums(lock_dir: Path, filename: str, expected_names: set[str]) -> None:
    path = lock_dir / filename
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1); rows.append((digest, name))
    if {name for _, name in rows} != expected_names or len(rows) != len(expected_names):
        raise IndependentLockVerificationError(f"{filename} exact-set mismatch")
    for digest, name in rows:
        if sha256_file(lock_dir / name) != digest:
            raise IndependentLockVerificationError(f"{filename} digest mismatch: {name}")


def verify_analysis_lock(repository: Path, lock_dir: Path, *, write_report: bool = True) -> dict[str, Any]:
    repository = repository.resolve(strict=True); lock_dir = lock_dir.resolve(strict=True)
    lock_path = lock_dir / "locked_analysis_lock_v1.json"
    lock = _load(lock_path)
    failures: list[str] = []
    try:
        if lock.get("status") != "ISSUED_AWAITING_SEPARATE_REAL_ANALYSIS_AUTHORIZATION":
            raise IndependentLockVerificationError("lock status mismatch")
        exact = {
            "RAW_EXECUTION_COMMIT": RAW_EXECUTION_COMMIT,
            "RAW_EXECUTION_TAG": RAW_EXECUTION_TAG,
            "POSTRUN_VERIFIER_CODE_COMMIT": POSTRUN_VERIFIER_CODE_COMMIT,
            "POSTRUN_VERIFICATION_COMMIT": POSTRUN_VERIFICATION_COMMIT,
            "POSTRUN_VERIFICATION_TAG": POSTRUN_VERIFICATION_TAG,
            "R3_LOCK_FINGERPRINT": R3_LOCK_FINGERPRINT,
            "C2_COMMIT": C2_COMMIT, "C2_TAG": C2_TAG, "C2_TAG_OBJECT": C2_TAG_OBJECT,
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "primary_exact_allocation_count": 20, "primary_p_value_denominator": 20,
            "centered_permutation_draw_count": 10000,
            "centered_permutation_p_value_denominator": 10001,
            "scene_strata_preserved": True, "systematic_formal_p_value_defined": False,
            "capture_radius_status": "PRESERVED_SUPPLEMENTARY_NOT_EXECUTED",
            "synthetic_transfer_status": "MODEL_TRANSFER_NOT_COMPATIBLE",
            "REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED": False,
            "REAL_SCIENTIFIC_ANALYSIS_EXECUTED": False,
            "REAL_FORMAL_RESULT_FILES_READ_BEFORE_FREEZE": 0,
            "READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS": True,
        }
        for key, expected in exact.items():
            if lock.get(key) != expected:
                raise IndependentLockVerificationError(f"lock field mismatch: {key}")
        if lock.get("analysis_lock_fingerprint") != _fingerprint(lock):
            raise IndependentLockVerificationError("analysis lock fingerprint mismatch")
        if sha256_file(lock_path) != (lock_dir / "locked_analysis_lock_v1.sha256").read_text().split()[0]:
            raise IndependentLockVerificationError("analysis lock SHA sidecar mismatch")

        _verify_tag(repository, RAW_EXECUTION_TAG, RAW_EXECUTION_COMMIT)
        _verify_tag(repository, POSTRUN_VERIFICATION_TAG, POSTRUN_VERIFICATION_COMMIT)
        _verify_tag(repository, C2_TAG, C2_COMMIT, object_sha=C2_TAG_OBJECT)
        r3_lock = repository / "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r3_lock/formal_batch1_zero_perturbation_lock_v1_1_exec_r3.json"
        if sha256_file(r3_lock) != lock.get("R3_LOCK_SHA"):
            raise IndependentLockVerificationError("R3 lock SHA mismatch")
        postrun_path = repository / PROTECTED_BINDINGS["postrun_verification_report"][0]
        postrun = _load(postrun_path)
        if sha256_file(postrun_path) != PROTECTED_BINDINGS["postrun_verification_report"][1] or not (
            postrun.get("pass") is True
            and postrun.get("FMB1_POSTRUN_INDEPENDENT_VERIFICATION_PASS") is True
            and postrun.get("RAW_EXECUTION_BYTES_UNCHANGED") is True
            and postrun.get("VERIFIED_TRIAL_COUNT") == 360
            and postrun.get("raw_execution_commit") == RAW_EXECUTION_COMMIT
        ):
            raise IndependentLockVerificationError("Post-run verification binding mismatch")

        for role, (relative, digest) in PROTECTED_BINDINGS.items():
            if sha256_file(repository / relative) != digest:
                raise IndependentLockVerificationError(f"protected binding mismatch: {role}")
            bound = lock.get("protected_bindings", {}).get(role, {})
            if bound != {"path": relative, "sha256": digest}:
                raise IndependentLockVerificationError(f"lock protected binding mismatch: {role}")
        determinacy = _load(lock_dir / "analysis_determinacy_binding.json")
        if not (determinacy.get("FMB1_LOCKED_ANALYSIS_CONTRACT_DETERMINATE") is True
                and determinacy.get("required_under_specified_count") == 0
                and determinacy.get("unresolved_root_definition_count") == 0):
            raise IndependentLockVerificationError("determinacy binding mismatch")

        commit = str(lock.get("LOCKED_ANALYSIS_CODE_COMMIT"))
        if _git(repository, "cat-file", "-t", commit).strip() != b"commit":
            raise IndependentLockVerificationError("analysis code commit absent")
        with (lock_dir / "analysis_code_inventory.csv").open(encoding="utf-8", newline="") as handle:
            inventory = list(csv.DictReader(handle))
        if [row["path"] for row in inventory] != list(ANALYSIS_CODE_PATHS):
            raise IndependentLockVerificationError("analysis code inventory path/order mismatch")
        for row in inventory:
            path = repository / row["path"]; data = path.read_bytes()
            if (hashlib.sha256(data).hexdigest() != row["sha256"]
                    or len(data) != int(row["bytes"])
                    or _git(repository, "show", f"{commit}:{row['path']}") != data):
                raise IndependentLockVerificationError(f"analysis code byte mismatch: {row['path']}")
        static = _verify_ast(repository)

        qualification = _load(lock_dir / "analysis_fixture_qualification.json")
        _, live_qualification = run_fixture_analysis()
        for key, value in live_qualification.items():
            if qualification.get(key) != value:
                raise IndependentLockVerificationError(f"fixture qualification mismatch: {key}")
        attestation = _load(lock_dir / "analysis_no_real_results_attestation.json")
        if not (attestation.get("pass") is True and all(
            value == 0 for key, value in attestation.items()
            if key.startswith("real_") or key == "registration_backend_calls"
        )):
            raise IndependentLockVerificationError("no-real-result attestation mismatch")
        future_root = Path(str(lock["future_analysis_output_root"]))
        if future_root.exists() and any(future_root.iterdir()):
            raise IndependentLockVerificationError("real analysis output already exists")
        if (lock_dir / "formal_analysis_authorization.json").exists():
            raise IndependentLockVerificationError("formal analysis authorization unexpectedly exists")
        _verify_sums(lock_dir, "LOCK_CORE_SHA256SUMS", set(LOCK_CORE_FILES))
    except (IndependentLockVerificationError, OSError, ValueError, KeyError) as exc:
        failures.append(str(exc))
        static = {"status": "FAIL", "finding_count": None}

    report = {
        "schema": "fmb1_locked_analysis_independent_lock_verification_v1",
        "status": "PASS" if not failures else "FAIL",
        "pass": not failures,
        "LOCKED_ANALYSIS_LOCK_VERIFIER_PASS": not failures,
        "failure_count": len(failures), "failures": failures,
        "analysis_lock_sha256": sha256_file(lock_path),
        "analysis_lock_fingerprint": lock.get("analysis_lock_fingerprint"),
        "LOCKED_ANALYSIS_CODE_COMMIT": lock.get("LOCKED_ANALYSIS_CODE_COMMIT"),
        "raw_execution_commit_verified": not failures,
        "raw_execution_bytes_unchanged": not failures,
        "postrun_verification_pass": not failures,
        "c1_unchanged": not failures, "c2_valid": not failures,
        "determinacy_under_specified_count": 0 if not failures else None,
        "determinacy_unresolved_root_count": 0 if not failures else None,
        "primary_exact_allocation_count": 20,
        "primary_p_value_denominator": 20,
        "centered_permutation_draw_count": 10000,
        "centered_permutation_p_value_denominator": 10001,
        "scene_strata_preserved": True,
        "systematic_formal_p_value_defined": False,
        "capture_radius_executed": False,
        "synthetic_transfer_status": "MODEL_TRANSFER_NOT_COMPATIBLE",
        "REAL_FORMAL_RESULT_FILES_READ": 0,
        "REAL_SCIENTIFIC_ANALYSIS_EXECUTED": False,
        "REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED": False,
        "registration_backend_calls": 0,
        "static_firewall": static,
    }
    if write_report:
        report_path = lock_dir / "independent_analysis_lock_verification.json"
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if report_path.exists() and report_path.read_text(encoding="utf-8") != encoded:
            raise IndependentLockVerificationError("refusing to overwrite differing verifier report")
        report_path.write_text(encoded, encoding="utf-8")
        files = {path.name for path in lock_dir.iterdir() if path.is_file()} - {"SHA256SUMS"}
        lines = [f"{sha256_file(lock_dir / name)}  {name}\n" for name in sorted(files)]
        (lock_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
        _verify_sums(lock_dir, "SHA256SUMS", files)
    return report
