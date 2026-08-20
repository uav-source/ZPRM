"""Fail-closed producer for one-time FMB1 exec-r2 authorization.

This producer is deliberately separate from the independent verifier.  It
never imports a registration backend and cannot issue without an explicit CLI
confirmation propagated as ``explicit_user_confirmation=True``.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..zero_perturbation_v1_1_r1_environment import verify_environment_manifest


AUTHORIZATION_SCHEMA = "mid360_fmb1_formal_registration_authorization_exec_r2_v1"
AUTHORIZATION_FILENAME = "formal_registration_authorization.json"
AUTHORIZATION_SHA_FILENAME = "formal_registration_authorization.sha256"
LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.json"
LOCK_SCHEMA = "mid360_fmb1_zero_perturbation_formal_lock_v1_1_exec_r2"
EXPECTED_RUNTIME = (
    "zero_perturbation_runtime/"
    "mid360_zero_perturbation_v1_1_formal_execution_v1"
)
BACKENDS = ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE")
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


class FormalAuthorizationIssuanceError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise FormalAuthorizationIssuanceError(
        f"FMB1_FORMAL_AUTHORIZATION_ISSUANCE_FAIL: {message}"
    )


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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"cannot parse JSON {path}: {error}")
    if not isinstance(value, Mapping):
        _fail(f"JSON is not an object: {path}")
    return value


def _inside(root: Path, path: Path, label: str, *, must_exist: bool = True) -> Path:
    lexical = path if path.is_absolute() else root / path
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        _fail(f"{label} is outside repository")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"{label} uses symlink component: {cursor}")
    if must_exist:
        resolved = lexical.resolve(strict=True)
        if not resolved.is_file():
            _fail(f"{label} is not a regular file")
        return resolved
    return lexical


def _identity(value: Any) -> bool:
    try:
        return len(value) == 4 and all(len(row) == 4 for row in value) and all(
            float(value[i][j]) == IDENTITY[i][j]
            for i in range(4) for j in range(4)
        )
    except (TypeError, ValueError, IndexError):
        return False


def _verify_git(root: Path, lock_release_commit: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if head != lock_release_commit:
        _fail("HEAD is not the explicitly supplied lock-release commit")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout
    if dirty:
        _fail("worktree is not clean")


def _verify_fingerprint(lock_dir: Path, lock: Mapping[str, Any]) -> tuple[str, str]:
    lock_path = lock_dir / LOCK_FILENAME
    fingerprint_path = lock_dir / "lock_fingerprint.json"
    inventory_path = lock_dir / "lock_inventory.csv"
    fingerprint = _json(fingerprint_path)
    lock_sha = sha256_file(lock_path)
    material = {
        "lock_file_sha256": lock_sha,
        "lock_inventory_file_sha256": sha256_file(inventory_path),
        "execution_code_commit": lock.get("execution_code_commit"),
    }
    reproduced = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    if (
        fingerprint.get("lock_file_sha256") != lock_sha
        or fingerprint.get("lock_fingerprint") != reproduced
    ):
        _fail("lock fingerprint does not reproduce")
    return lock_sha, reproduced


def _verify_plan(root: Path, lock: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    binding = lock.get("bindings", {}).get("trial_plan_json", {})
    path = _inside(root, Path(str(binding.get("repository_relative_path"))), "trial plan")
    if sha256_file(path) != binding.get("sha256"):
        _fail("trial plan SHA differs from lock")
    plan = _json(path)
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 360:
        _fail("trial plan is not exactly 360 rows")
    ids: set[str] = set()
    pairs: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("trial plan row is not an object")
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in ids:
            _fail("trial plan has invalid/duplicate trial ID")
        ids.add(trial_id)
        backend = str(row.get("backend"))
        if backend not in BACKENDS or row.get("track_id") != "ZERO_PERTURBATION_TRACK":
            _fail("trial plan backend/track differs")
        if not _identity(row.get("T0")):
            _fail("trial plan contains non-Identity T0")
        if float(row.get("translation_perturbation_m", -1)) != 0.0 or float(
            row.get("rotation_perturbation_deg", -1)
        ) != 0.0:
            _fail("capture-radius row entered zero-perturbation plan")
        if row.get("scene_id") == "FMB1_W04" or (
            row.get("scene_id") == "FMB1_W02" and row.get("attempt") != 2
        ):
            _fail("retired W04 or W02 attempt1 entered trial plan")
        counts[backend] += 1
        pairs[str(row.get("snapshot_id"))].append(row)
    if counts != Counter({BACKENDS[0]: 180, BACKENDS[1]: 180}) or len(pairs) != 180:
        _fail("trial-plan backend/snapshot counts differ")
    if any(len(pair) != 2 or {row["backend"] for row in pair} != set(BACKENDS)
           for pair in pairs.values()):
        _fail("snapshot backend pairing differs")
    return path, plan


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _fail(f"refusing duplicate/overwrite authorization artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        _fail(f"stale authorization temporary exists: {temporary}")
    temporary.write_bytes(content)
    temporary.replace(path)


def issue_formal_registration_authorization(
    repository: Path,
    *,
    lock_dir: Path,
    expected_lock_fingerprint: str,
    lock_release_commit: str,
    runtime_root: Path,
    workers: int,
    explicit_user_confirmation: bool,
    issued_at_utc: str | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Issue one immutable authorization after all producer-side gates pass."""

    if explicit_user_confirmation is not True:
        _fail("explicit user authorization confirmation flag is required")
    if workers != 2:
        _fail("formal authorization requires exactly two workers")
    root = Path(repository).resolve(strict=True)
    locked = Path(lock_dir).resolve(strict=True)
    try:
        locked.relative_to(root)
    except ValueError:
        _fail("lock directory escapes repository")
    _verify_git(root, lock_release_commit)
    lock_path = _inside(root, locked / LOCK_FILENAME, "exec-r2 lock")
    lock = _json(lock_path)
    if (
        lock.get("schema") != LOCK_SCHEMA
        or lock.get("execution_lock_revision") != 2
        or lock.get("status") != "ISSUED_AWAITING_SEPARATE_AUTHORIZATION"
        or lock.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or lock.get("actual_formal_trials") != 0
    ):
        _fail("exec-r2 lock is not a closed zero-trial lock")
    lock_sha, fingerprint = _verify_fingerprint(locked, lock)
    if fingerprint != expected_lock_fingerprint:
        _fail("explicit lock fingerprint differs")
    independent = _json(locked / "independent_verification.json")
    if (
        independent.get("pass") is not True
        or independent.get("NEW_LOCK_VERIFIER_PASS") is not True
        or independent.get("lock_fingerprint") != fingerprint
    ):
        _fail("independent exec-r2 lock verifier is not PASS")
    try:
        from ..zero_perturbation_v1_1_exec_r2_verify import verify_exec_r2_lock
        live_lock_verification = verify_exec_r2_lock(
            root, locked,
            expected_execution_code_commit=str(lock["execution_code_commit"]),
        )
    except Exception as error:
        _fail(f"live independent exec-r2 lock verification failed: {error}")
    if (
        live_lock_verification.get("NEW_LOCK_VERIFIER_PASS") is not True
        or live_lock_verification.get("lock_fingerprint") != fingerprint
    ):
        _fail("live independent exec-r2 lock verification is not PASS/bound")
    plan_path, _ = _verify_plan(root, lock)
    bindings = lock.get("bindings")
    if not isinstance(bindings, Mapping):
        _fail("lock bindings are missing")
    final_verify_path = _inside(
        root,
        Path(str(bindings["final_dataset_independent_verification"]["repository_relative_path"])),
        "final dataset verifier",
    )
    final_verify = _json(final_verify_path)
    if final_verify.get("status") != "PASS" or final_verify.get("pass") is not True:
        _fail("final dataset independent verifier is not PASS")
    environment_path = _inside(
        root, Path(str(bindings["environment_manifest"]["repository_relative_path"])),
        "environment manifest",
    )
    environment = _json(environment_path)
    verify_environment_manifest(environment, root, remeasure_versions=True)
    pcl_path = _inside(
        root, Path(str(bindings["pcl_executable"]["repository_relative_path"])),
        "PCL executable",
    )
    if sha256_file(pcl_path) != bindings["pcl_executable"]["sha256"]:
        _fail("PCL executable SHA differs")
    selected_runtime = _inside(root, runtime_root, "runtime root", must_exist=False)
    expected_runtime = root / EXPECTED_RUNTIME
    if selected_runtime != expected_runtime:
        _fail("runtime root is not the canonical exec-r2 runtime")
    if selected_runtime.exists() and any(selected_runtime.iterdir()):
        _fail("formal runtime is not empty before authorization issuance")
    execution_results = root / "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1"
    if execution_results.exists():
        _fail("formal execution results already exist")
    runtime_parent = root / "zero_perturbation_runtime"
    if runtime_parent.exists():
        existing = list(runtime_parent.rglob(AUTHORIZATION_FILENAME))
        if existing:
            _fail("another formal authorization already exists")
    timestamp = issued_at_utc or datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        _fail("issued_at_utc is invalid")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail("issued_at_utc must be timezone-aware")
    selected_nonce = nonce or secrets.token_hex(32)
    if len(selected_nonce) != 64 or any(ch not in "0123456789abcdef" for ch in selected_nonce):
        _fail("nonce must be 64 lowercase hexadecimal characters")
    authorization_id = f"FMB1-AUTH-{uuid.uuid4().hex}"
    payload = {
        "schema": AUTHORIZATION_SCHEMA,
        "authorization_id": authorization_id,
        "authorization_type": "ONE_TIME_FMB1_ZERO_PERTURBATION_FORMAL_EXECUTION",
        "authorization_basis": "EXPLICIT_USER_INSTRUCTION_RECORDED_BY_OPERATOR",
        "issued_at_utc": timestamp,
        "nonce": selected_nonce,
        "status": "ISSUED",
        "track_id": "ZERO_PERTURBATION_TRACK",
        "active_amendment_id": "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1",
        "lock_revision": 2,
        "lock_fingerprint": fingerprint,
        "lock_file_sha256": lock_sha,
        "lock_release_commit": lock_release_commit,
        "trial_plan_sha256": sha256_file(plan_path),
        "analysis_contract_sha256": bindings["analysis_contract"]["sha256"],
        "backend_contract_sha256": bindings["backend_parameter_contract"]["sha256"],
        "execution_code_commit": lock["execution_code_commit"],
        "environment_manifest_sha256": sha256_file(environment_path),
        "pcl_binary_sha256": sha256_file(pcl_path),
        "planned_trial_count": 360,
        "planned_open3d_count": 180,
        "planned_pcl_count": 180,
        "allowed_backends": list(BACKENDS),
        "T0_policy": "IDENTITY_ONLY",
        "execution_mode_initial": "fresh",
        "resume_allowed": True,
        "workers": 2,
        "capture_radius_authorized": False,
        "other_backend_authorized": False,
        "parameter_change_authorized": False,
        "trial_reselection_authorized": False,
        "authorization_scope": "ONE_FORMAL_EXECUTION_ATTEMPT_WITH_RESUME_ONLY",
        "authoritative_runtime_root": EXPECTED_RUNTIME,
        "immutable": True,
        "FORMAL_ICP_UNLOCKED": True,
        "FORMAL_REGISTRATION_AUTHORIZED": True,
    }
    content = canonical_json_bytes(payload)
    auth_dir = selected_runtime / "authorization"
    authorization_path = auth_dir / AUTHORIZATION_FILENAME
    sha_path = auth_dir / AUTHORIZATION_SHA_FILENAME
    digest = hashlib.sha256(content).hexdigest()
    _write_once(authorization_path, content)
    _write_once(sha_path, f"{digest}  {AUTHORIZATION_FILENAME}\n".encode("ascii"))
    return {
        "schema": "mid360_fmb1_formal_authorization_issuance_report_exec_r2_v1",
        "status": "ISSUED_AWAITING_INDEPENDENT_VERIFICATION",
        "authorization_path": str(authorization_path),
        "authorization_sha256": digest,
        "authorization_id": authorization_id,
        "lock_fingerprint": fingerprint,
        "FORMAL_REGISTRATION_AUTHORIZED": True,
        "actual_formal_trials": 0,
    }
