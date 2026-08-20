"""Independent verifier for the FMB1 Zero-Perturbation exec-r2 lock."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .zero_perturbation_v1_1_r1_environment import verify_environment_manifest


LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.json"
LOCK_SCHEMA = "mid360_fmb1_zero_perturbation_formal_lock_v1_1_exec_r2"
OLD_LOCK_PATH = Path(
    "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/"
    "formal_batch1_zero_perturbation_lock_v1_1.json"
)
OLD_LOCK_SHA256 = "a191894d76f5c3bd3be2aa53f4a2b8bb6b641b77d9603a6f300993b49bbf9c0c"
OLD_LOCK_FINGERPRINT = "9fcd01bd439e1c4e8c2bd0a220a3af1d16c394177bb17197850c4d769109cbba"
EXPECTED_RUNTIME = Path(
    "zero_perturbation_runtime/mid360_zero_perturbation_v1_1_formal_execution_v1"
)
BACKENDS = ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE")
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
CORE_REQUIRED = {
    LOCK_FILENAME, "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.sha256",
    "lock_inventory.csv", "lock_fingerprint.json", "NO_REGISTRATION_ATTESTATION.json",
    "environment_manifest.json", "execution_control_patch_report.json",
    "authorization_lifecycle_test_report.json",
}
EXECUTION_BINDING_NAMES = {
    "execution_runner", "execution_runner_cli",
    "execution_experiments_package_init",
    "execution_mid360_formal_batch1_package_init",
    "execution_phase_a_harness_package_init", "execution_environment",
    "execution_result_validator", "execution_open3d_backend",
    "execution_pcl_backend", "execution_common_association",
    "execution_rotation_metrics", "execution_metrics", "execution_types",
    "execution_authorization_producer", "execution_authorization_verifier",
    "execution_authorization_lifecycle",
    "execution_authorization_producer_cli",
    "execution_authorization_verifier_cli",
    "execution_exec_r2_lock_builder", "execution_exec_r2_lock_verifier",
    "execution_exec_r2_lock_issuer_cli",
    "execution_exec_r2_lock_verifier_cli",
}


class ExecR2IndependentVerificationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise ExecR2IndependentVerificationError(f"FMB1_EXEC_R2_VERIFY_FAIL: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"cannot parse JSON {path}: {error}")
    if not isinstance(value, Mapping):
        _fail(f"JSON is not an object: {path}")
    return value


def _resolve(root: Path, relative: str, label: str) -> Path:
    lexical = root / relative
    cursor = root
    for part in Path(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"{label} uses symlink component")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        _fail(f"{label} missing/escapes repository: {error}")
    if not resolved.is_file():
        _fail(f"{label} is not a regular file")
    return resolved


def _verify_core(lock_dir: Path) -> int:
    sums = lock_dir / "LOCK_CORE_SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        _fail("LOCK_CORE_SHA256SUMS missing")
    declared: dict[str, str] = {}
    for line in sums.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            _fail("malformed core checksum line")
        digest, name = match.groups()
        path = lock_dir / name
        if name in declared or path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            _fail(f"core checksum differs: {name}")
        declared[name] = digest
    if set(declared) != CORE_REQUIRED:
        _fail(f"core checksum set differs: {sorted(set(declared)^CORE_REQUIRED)}")
    return len(declared)


def _identity(value: Any) -> bool:
    try:
        return len(value) == 4 and all(len(row) == 4 for row in value) and all(
            float(value[i][j]) == IDENTITY[i][j]
            for i in range(4) for j in range(4)
        )
    except (TypeError, ValueError, IndexError):
        return False


def _verify_plan(path: Path) -> None:
    plan = _json(path); rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 360:
        _fail("plan row count differs")
    ids: set[str] = set(); counts: Counter[str] = Counter()
    snapshots: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("trial_id"), str) or row["trial_id"] in ids:
            _fail("plan trial ID differs")
        ids.add(row["trial_id"]); counts[str(row.get("backend"))] += 1
        snapshots[str(row.get("snapshot_id"))].append(row)
        if row.get("backend") not in BACKENDS or row.get("track_id") != "ZERO_PERTURBATION_TRACK" or not _identity(row.get("T0")):
            _fail("plan backend/track/T0 differs")
        if row.get("scene_id") == "FMB1_W04" or (row.get("scene_id") == "FMB1_W02" and row.get("attempt") != 2):
            _fail("retired acquisition entered plan")
    if counts != Counter({BACKENDS[0]: 180, BACKENDS[1]: 180}) or len(snapshots) != 180:
        _fail("plan counts differ")
    if any(len(rows) != 2 or {row["backend"] for row in rows} != set(BACKENDS) for rows in snapshots.values()):
        _fail("plan backend pairs differ")


def _static_control_audit(paths: Mapping[str, Path]) -> None:
    denied = {"open3d", "phase_a_harness.open3d_backend", "phase_a_harness.pcl_backend"}
    for name, path in paths.items():
        if not name.startswith("execution_authorization") and "exec_r2_lock" not in name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imports = {node.module or ""}
            else:
                imports = set()
            if imports & denied:
                _fail(f"execution-control file eagerly imports backend: {name}")


def verify_exec_r2_lock(
    repository: Path, lock_dir: Path, *, expected_execution_code_commit: str,
) -> dict[str, Any]:
    root = Path(repository).resolve(strict=True); locked = Path(lock_dir).resolve(strict=True)
    core_count = _verify_core(locked)
    lock_path = locked / LOCK_FILENAME; lock = _json(lock_path)
    if (
        lock.get("schema") != LOCK_SCHEMA
        or lock.get("execution_lock_revision") != 2
        or lock.get("execution_code_commit") != expected_execution_code_commit
        or lock.get("amendment_id") != "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1"
        or lock.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or lock.get("FORMAL_ICP_UNLOCKED") is not False
        or lock.get("actual_formal_trials") != 0
    ):
        _fail("exec-r2 lock state differs")
    inventory_path = locked / "lock_inventory.csv"
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["binding_name"] for row in rows}) != len(rows):
        _fail("lock inventory is empty/duplicated")
    material = {
        "lock_file_sha256": sha256_file(lock_path),
        "lock_inventory_file_sha256": sha256_file(inventory_path),
        "execution_code_commit": expected_execution_code_commit,
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    fingerprint_payload = _json(locked / "lock_fingerprint.json")
    if fingerprint_payload.get("lock_fingerprint") != fingerprint or fingerprint_payload.get("lock_file_sha256") != material["lock_file_sha256"]:
        _fail("lock fingerprint differs")
    bindings = lock.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {row["binding_name"] for row in rows}:
        _fail("lock/inventory binding set differs")
    paths: dict[str, Path] = {}
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            _fail(f"binding malformed: {name}")
        path = _resolve(root, str(binding.get("repository_relative_path")), f"binding {name}")
        if sha256_file(path) != binding.get("sha256") or path.stat().st_size != int(binding.get("bytes", -1)):
            _fail(f"binding bytes differ: {name}")
        paths[name] = path
    old_path = _resolve(root, OLD_LOCK_PATH.as_posix(), "old lock")
    if sha256_file(old_path) != OLD_LOCK_SHA256 or lock.get("supersedes_lock_fingerprint") != OLD_LOCK_FINGERPRINT:
        _fail("old lock history differs")
    old = _json(old_path)
    for name, old_binding in old["bindings"].items():
        if name.startswith("execution_") or name == "environment_manifest":
            continue
        if name not in bindings or bindings[name]["sha256"] != old_binding["sha256"]:
            _fail(f"scientific binding changed from old lock: {name}")
    commit = expected_execution_code_commit
    subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root, check=True)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    for name, path in paths.items():
        if name not in EXECUTION_BINDING_NAMES:
            continue
        relative = path.relative_to(root).as_posix()
        recorded = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
        if hashlib.sha256(recorded).hexdigest() != bindings[name]["sha256"]:
            _fail(f"execution binding differs from commit: {name}")
    if set(name for name in bindings if name.startswith("execution_") and name != "execution_control_patch_report") != EXECUTION_BINDING_NAMES:
        _fail("execution binding name set differs")
    _static_control_audit(paths)
    schema = _json(paths["authorization_schema"])
    if schema.get("additionalProperties") is not False or len(schema.get("required", [])) < 30:
        _fail("authorization schema is not strict")
    lifecycle = _json(paths["authorization_lifecycle_test_report"])
    if (
        lifecycle.get("status") != "PASS"
        or lifecycle.get("tamper_case_count", 0) < 25
        or lifecycle.get("REAL_FORMAL_TRIALS") != 0
        or lifecycle.get("execution_code_commit") != commit
    ):
        _fail("authorization lifecycle/tamper qualification differs")
    patch = _json(paths["execution_control_patch_report"])
    for key in ("SCIENTIFIC_PROTOCOL_CHANGED", "FINAL_DATASET_CHANGED", "TRIAL_PLAN_SCIENTIFIC_CONTENT_CHANGED", "BACKEND_PARAMETERS_CHANGED"):
        if patch.get(key) is not False:
            _fail(f"execution patch changed science: {key}")
    if patch.get("NEW_EXECUTION_CODE_COMMIT") != commit or patch.get("ACTUAL_FORMAL_TRIALS") != 0:
        _fail("execution patch commit/trial binding differs")
    _verify_plan(paths["trial_plan_json"])
    verify_environment_manifest(_json(paths["environment_manifest"]), root, remeasure_versions=True)
    final = _json(paths["final_dataset_independent_verification"])
    if final.get("status") != "PASS" or final.get("pass") is not True:
        _fail("final dataset verifier differs")
    supersession = _json(paths["old_lock_supersession"])
    if (
        supersession.get("OLD_LOCK_STATUS")
        != "SUPERSEDED_BEFORE_EXECUTION_BY_EXECUTION_CONTROL_PATCH"
        or supersession.get("OLD_LOCK_ACTUAL_FORMAL_TRIALS") != 0
        or supersession.get("old_lock_file_modified") is not False
    ):
        _fail("old lock supersession trial count differs")
    no_registration = _json(locked / "NO_REGISTRATION_ATTESTATION.json")
    if no_registration.get("pass") is not True or any(int(no_registration.get(key, -1)) != 0 for key in (
        "open3d_registration_call_count", "pcl_cli_invocation_count", "formal_trial_count"
    )):
        _fail("R2 no-registration attestation differs")
    if (root / EXPECTED_RUNTIME).exists() or (root / "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1").exists():
        _fail("runtime or real execution results exist")
    return {
        "schema": "mid360_fmb1_zero_perturbation_exec_r2_lock_independent_verification_v1",
        "status": "PASS", "pass": True,
        "NEW_LOCK_VERIFIER_PASS": True,
        "failure_count": 0,
        "execution_lock_revision": 2,
        "lock_file_sha256": sha256_file(lock_path),
        "lock_fingerprint": fingerprint,
        "execution_code_commit": commit,
        "binding_count": len(bindings),
        "lock_core_checksum_count": core_count,
        "scene_count": 6, "station_count": 18, "snapshot_count": 180,
        "planned_open3d_trials": 180, "planned_pcl_trials": 180,
        "planned_total_trials": 360,
        "AUTHORIZATION_PRODUCER_READY": True,
        "INDEPENDENT_AUTHORIZATION_VERIFIER_READY": True,
        "AUTHORIZATION_LIFECYCLE_QUALIFIED": True,
        "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
        "registration_backend_calls": 0,
    }


def verify_release_checksums(lock_dir: Path) -> dict[str, Any]:
    locked = Path(lock_dir).resolve(strict=True); sums = locked / "SHA256SUMS"
    declared: dict[str, str] = {}
    for line in sums.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            _fail("release checksum line malformed")
        digest, name = match.groups(); declared[name] = digest
    actual = {path.name: path for path in locked.iterdir() if path.is_file() and path.name != "SHA256SUMS"}
    if set(declared) != set(actual):
        _fail("release checksum set differs")
    for name, path in actual.items():
        if path.is_symlink() or sha256_file(path) != declared[name]:
            _fail(f"release checksum differs: {name}")
    return {
        "schema": "mid360_fmb1_exec_r2_release_checksum_verification_v1",
        "status": "PASS", "pass": True,
        "covered_regular_file_count": len(actual),
        "sha256sums_sha256": sha256_file(sums),
    }
