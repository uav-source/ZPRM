"""Independent verifier for immutable one-time FMB1 exec-r2 authorization.

This module does not import or call the producer and independently recomputes
every authority binding needed before the runner may load a backend.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ..zero_perturbation_v1_1_r1_environment import verify_environment_manifest


AUTHORIZATION_SCHEMA = "mid360_fmb1_formal_registration_authorization_exec_r2_v1"
AUTHORIZATION_FILENAME = "formal_registration_authorization.json"
AUTHORIZATION_SHA_FILENAME = "formal_registration_authorization.sha256"
VERIFICATION_REPORT_FILENAME = "authorization_verification_report.json"
IN_USE_FILENAME = "authorization_in_use.json"
RECEIPT_FILENAME = "authorization_consumption_receipt.json"
LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.json"
LOCK_SCHEMA = "mid360_fmb1_zero_perturbation_formal_lock_v1_1_exec_r2"
EXPECTED_RUNTIME = (
    "zero_perturbation_runtime/"
    "mid360_zero_perturbation_v1_1_formal_execution_v1"
)
BACKENDS = ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE")
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
AUTH_ID_RE = re.compile(r"^FMB1-AUTH-[0-9a-f]{32}$")

EXPECTED_KEYS = {
    "schema", "authorization_id", "authorization_type", "authorization_basis",
    "issued_at_utc", "nonce", "status", "track_id", "active_amendment_id",
    "lock_revision", "lock_fingerprint", "lock_file_sha256",
    "lock_release_commit", "trial_plan_sha256", "analysis_contract_sha256",
    "backend_contract_sha256", "execution_code_commit",
    "environment_manifest_sha256", "pcl_binary_sha256", "planned_trial_count",
    "planned_open3d_count", "planned_pcl_count", "allowed_backends",
    "T0_policy", "execution_mode_initial", "resume_allowed", "workers",
    "capture_radius_authorized", "other_backend_authorized",
    "parameter_change_authorized", "trial_reselection_authorized",
    "authorization_scope", "authoritative_runtime_root", "immutable",
    "FORMAL_ICP_UNLOCKED", "FORMAL_REGISTRATION_AUTHORIZED",
}


class FormalAuthorizationVerificationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise FormalAuthorizationVerificationError(
        f"FMB1_FORMAL_AUTHORIZATION_VERIFY_FAIL: {message}"
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


def _resolve(root: Path, path: Path, label: str) -> Path:
    lexical = path if path.is_absolute() else root / path
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        _fail(f"{label} is outside repository")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"{label} uses a symlink component")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        _fail(f"{label} is missing: {error}")
    if not resolved.is_file():
        _fail(f"{label} is not a regular file")
    return resolved


def _identity(value: Any) -> bool:
    try:
        return len(value) == 4 and all(len(row) == 4 for row in value) and all(
            float(value[i][j]) == IDENTITY[i][j]
            for i in range(4) for j in range(4)
        )
    except (TypeError, ValueError, IndexError):
        return False


def _validate_payload_shape(payload: Mapping[str, Any]) -> None:
    if set(payload) != EXPECTED_KEYS:
        _fail(f"authorization keys differ: missing={sorted(EXPECTED_KEYS-set(payload))}, extra={sorted(set(payload)-EXPECTED_KEYS)}")
    exact = {
        "schema": AUTHORIZATION_SCHEMA,
        "authorization_type": "ONE_TIME_FMB1_ZERO_PERTURBATION_FORMAL_EXECUTION",
        "authorization_basis": "EXPLICIT_USER_INSTRUCTION_RECORDED_BY_OPERATOR",
        "status": "ISSUED",
        "track_id": "ZERO_PERTURBATION_TRACK",
        "active_amendment_id": "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1",
        "lock_revision": 2,
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
    for key, value in exact.items():
        if payload.get(key) != value:
            _fail(f"authorization contract differs: {key}")
    if AUTH_ID_RE.fullmatch(str(payload.get("authorization_id"))) is None:
        _fail("authorization ID is invalid")
    for key in (
        "nonce", "lock_fingerprint", "lock_file_sha256", "trial_plan_sha256",
        "analysis_contract_sha256", "backend_contract_sha256",
        "environment_manifest_sha256", "pcl_binary_sha256",
    ):
        if SHA_RE.fullmatch(str(payload.get(key))) is None:
            _fail(f"authorization SHA/nonce field is invalid: {key}")
    for key in ("lock_release_commit", "execution_code_commit"):
        if COMMIT_RE.fullmatch(str(payload.get(key))) is None:
            _fail(f"authorization commit field is invalid: {key}")
    try:
        issued = datetime.fromisoformat(str(payload["issued_at_utc"]).replace("Z", "+00:00"))
    except ValueError:
        _fail("issued_at_utc is invalid")
    if issued.tzinfo is None or issued.utcoffset() is None:
        _fail("issued_at_utc lacks timezone")


def _verify_schema_file(root: Path, lock: Mapping[str, Any]) -> None:
    binding = lock["bindings"]["authorization_schema"]
    path = _resolve(root, Path(binding["repository_relative_path"]), "authorization schema")
    if sha256_file(path) != binding["sha256"]:
        _fail("authorization schema SHA differs")
    schema = _json(path)
    if set(schema.get("required", [])) != EXPECTED_KEYS:
        _fail("authorization JSON Schema required fields differ")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != EXPECTED_KEYS:
        _fail("authorization JSON Schema properties differ")
    if schema.get("additionalProperties") is not False:
        _fail("authorization JSON Schema is not closed")


def _verify_git_and_lock(
    root: Path, lock_dir: Path, payload: Mapping[str, Any]
) -> tuple[Mapping[str, Any], str]:
    lock_path = _resolve(root, lock_dir / LOCK_FILENAME, "exec-r2 lock")
    lock = _json(lock_path)
    if lock.get("schema") != LOCK_SCHEMA or lock.get("execution_lock_revision") != 2:
        _fail("exec-r2 lock schema/revision differs")
    if (
        lock.get("status") != "ISSUED_AWAITING_SEPARATE_AUTHORIZATION"
        or lock.get("FORMAL_LOCK_ISSUED") is not True
        or lock.get("FORMAL_ICP_UNLOCKED") is not False
        or lock.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or lock.get("actual_formal_trials") != 0
        or lock.get("registration_execution_count") != 0
        or lock.get("AUTHORIZATION_PRODUCER_READY") is not True
        or lock.get("INDEPENDENT_AUTHORIZATION_VERIFIER_READY") is not True
        or lock.get("AUTHORIZATION_LIFECYCLE_QUALIFIED") is not True
    ):
        _fail("exec-r2 lock is not a closed qualified zero-trial lock")
    sums = _resolve(root, lock_dir / "LOCK_CORE_SHA256SUMS", "lock core checksums")
    declared: dict[str, str] = {}
    for line in sums.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            _fail("lock core checksum line is malformed")
        digest, name = match.groups()
        artifact = _resolve(root, lock_dir / name, f"lock core {name}")
        if name in declared or sha256_file(artifact) != digest:
            _fail(f"lock core checksum differs: {name}")
        declared[name] = digest
    required_core = {
        LOCK_FILENAME,
        "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.sha256",
        "lock_inventory.csv", "lock_fingerprint.json",
        "NO_REGISTRATION_ATTESTATION.json", "environment_manifest.json",
        "execution_control_patch_report.json",
        "authorization_lifecycle_test_report.json",
    }
    if set(declared) != required_core:
        _fail("lock core checksum coverage differs")
    if sha256_file(lock_path) != payload["lock_file_sha256"]:
        _fail("authorization lock SHA differs")
    inventory = _resolve(root, lock_dir / "lock_inventory.csv", "lock inventory")
    material = {
        "lock_file_sha256": sha256_file(lock_path),
        "lock_inventory_file_sha256": sha256_file(inventory),
        "execution_code_commit": lock.get("execution_code_commit"),
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    fingerprint_file = _json(_resolve(root, lock_dir / "lock_fingerprint.json", "lock fingerprint"))
    if fingerprint != payload["lock_fingerprint"] or fingerprint_file.get("lock_fingerprint") != fingerprint:
        _fail("authorization lock fingerprint differs")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if head != payload["lock_release_commit"]:
        _fail("current HEAD differs from authorization lock-release commit")
    relative = lock_path.relative_to(root).as_posix()
    recorded = subprocess.run(
        ["git", "show", f"{head}:{relative}"], cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if hashlib.sha256(recorded).hexdigest() != payload["lock_file_sha256"]:
        _fail("lock-release commit does not contain the authorized lock bytes")
    if lock.get("execution_code_commit") != payload["execution_code_commit"]:
        _fail("authorization execution commit differs from lock")
    for name, binding in lock.get("bindings", {}).items():
        if not isinstance(binding, Mapping):
            _fail(f"lock binding malformed: {name}")
        path = _resolve(root, Path(str(binding.get("repository_relative_path"))), f"binding {name}")
        if sha256_file(path) != binding.get("sha256") or path.stat().st_size != binding.get("bytes"):
            _fail(f"lock binding changed: {name}")
        if name.startswith("execution_"):
            path_relative = path.relative_to(root).as_posix()
            code_bytes = subprocess.run(
                ["git", "show", f"{payload['execution_code_commit']}:{path_relative}"],
                cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout
            if hashlib.sha256(code_bytes).hexdigest() != binding.get("sha256"):
                _fail(f"execution binding differs from execution commit: {name}")
    _verify_schema_file(root, lock)
    return lock, fingerprint


def _verify_plan_and_environment(
    root: Path, lock: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    bindings = lock["bindings"]
    plan_path = _resolve(root, Path(bindings["trial_plan_json"]["repository_relative_path"]), "trial plan")
    if sha256_file(plan_path) != payload["trial_plan_sha256"]:
        _fail("authorization trial-plan SHA differs")
    plan = _json(plan_path)
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 360:
        _fail("trial plan does not contain 360 rows")
    counts: Counter[str] = Counter()
    pairs: defaultdict[str, int] = defaultdict(int)
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("trial_id") in ids:
            _fail("trial plan has malformed/duplicate row")
        ids.add(str(row["trial_id"]))
        backend = str(row.get("backend")); counts[backend] += 1
        pairs[str(row.get("snapshot_id"))] += 1
        if backend not in BACKENDS or row.get("track_id") != "ZERO_PERTURBATION_TRACK" or not _identity(row.get("T0")):
            _fail("trial plan backend/track/T0 differs")
        if (
            float(row.get("translation_perturbation_m", -1.0)) != 0.0
            or float(row.get("rotation_perturbation_deg", -1.0)) != 0.0
        ):
            _fail("capture-radius perturbation entered authorization plan")
        if row.get("scene_id") == "FMB1_W04" or (
            row.get("scene_id") == "FMB1_W02" and row.get("attempt") != 2
        ):
            _fail("retired acquisition entered plan")
    if counts != Counter({BACKENDS[0]: 180, BACKENDS[1]: 180}) or len(pairs) != 180 or set(pairs.values()) != {2}:
        _fail("trial plan counts/pairing differ")
    if payload["analysis_contract_sha256"] != bindings["analysis_contract"]["sha256"]:
        _fail("authorization analysis contract differs")
    if payload["backend_contract_sha256"] != bindings["backend_parameter_contract"]["sha256"]:
        _fail("authorization backend contract differs")
    env_path = _resolve(root, Path(bindings["environment_manifest"]["repository_relative_path"]), "environment")
    if sha256_file(env_path) != payload["environment_manifest_sha256"]:
        _fail("authorization environment SHA differs")
    verify_environment_manifest(_json(env_path), root, remeasure_versions=True)
    pcl_path = _resolve(root, Path(bindings["pcl_executable"]["repository_relative_path"]), "PCL binary")
    if sha256_file(pcl_path) != payload["pcl_binary_sha256"]:
        _fail("authorization PCL SHA differs")


def _verify_lifecycle(
    root: Path, authorization_path: Path, payload: Mapping[str, Any],
    runtime_root: Path, requested_mode: str,
) -> str:
    if requested_mode not in {"fresh", "resume"}:
        _fail("requested mode is invalid")
    expected_runtime = root / EXPECTED_RUNTIME
    selected_runtime = runtime_root if runtime_root.is_absolute() else root / runtime_root
    if selected_runtime != expected_runtime or payload["authoritative_runtime_root"] != EXPECTED_RUNTIME:
        _fail("authorization runtime differs")
    auth_dir = authorization_path.parent
    if auth_dir != selected_runtime / "authorization":
        _fail("authorization is outside canonical runtime authorization directory")
    duplicates = []
    runtime_parent = root / "zero_perturbation_runtime"
    if runtime_parent.exists():
        duplicates = [path.resolve() for path in runtime_parent.rglob(AUTHORIZATION_FILENAME)]
    if {path for path in duplicates} != {authorization_path.resolve()} or len(duplicates) != 1:
        _fail("duplicate/second live authorization exists")
    receipt = auth_dir / RECEIPT_FILENAME
    marker = auth_dir / IN_USE_FILENAME
    if receipt.exists() or receipt.is_symlink():
        _fail("authorization is already consumed")
    if marker.exists() or marker.is_symlink():
        marker_payload = _json(_resolve(root, marker, "authorization in-use marker"))
        expected = {
            "schema": "mid360_fmb1_authorization_in_use_exec_r2_v1",
            "state": "IN_USE",
            "authorization_id": payload["authorization_id"],
            "authorization_sha256": sha256_file(authorization_path),
            "lock_fingerprint": payload["lock_fingerprint"],
            "runtime_root": EXPECTED_RUNTIME,
            "reusable_for_new_fresh": False,
        }
        if set(marker_payload) != set(expected) | {"started_at_utc"}:
            _fail("authorization in-use marker keys differ")
        for key, value in expected.items():
            if marker_payload.get(key) != value:
                _fail(f"authorization in-use marker differs: {key}")
        try:
            started = datetime.fromisoformat(
                str(marker_payload["started_at_utc"]).replace("Z", "+00:00")
            )
        except ValueError:
            _fail("authorization in-use timestamp is invalid")
        if started.tzinfo is None or started.utcoffset() is None:
            _fail("authorization in-use timestamp lacks timezone")
        if requested_mode != "resume":
            _fail("IN_USE authorization permits resume only")
        return "IN_USE_RESUME_SAME_ATTEMPT"
    if requested_mode != "fresh":
        _fail("resume requires an IN_USE authorization marker")
    return "VERIFIED_READY_FOR_ONE_FRESH"


def verify_formal_registration_authorization(
    repository: Path,
    *,
    lock_dir: Path,
    authorization_path: Path,
    runtime_root: Path,
    requested_mode: str,
    workers: int,
) -> dict[str, Any]:
    """Independently verify one authorization without producer code reuse."""

    if workers != 2:
        _fail("authorization requires exactly two workers")
    root = Path(repository).resolve(strict=True)
    auth = _resolve(root, authorization_path, "formal authorization")
    payload = _json(auth)
    _validate_payload_shape(payload)
    sidecar = _resolve(root, auth.parent / AUTHORIZATION_SHA_FILENAME, "authorization SHA sidecar")
    expected_sidecar = f"{sha256_file(auth)}  {AUTHORIZATION_FILENAME}\n"
    if sidecar.read_text(encoding="ascii") != expected_sidecar:
        _fail("authorization SHA sidecar differs")
    if auth.read_bytes() != canonical_json_bytes(payload):
        _fail("authorization JSON is not canonical/immutable bytes")
    lock, fingerprint = _verify_git_and_lock(root, Path(lock_dir).resolve(strict=True), payload)
    _verify_plan_and_environment(root, lock, payload)
    lifecycle = _verify_lifecycle(root, auth, payload, Path(runtime_root), requested_mode)
    return {
        "schema": "mid360_fmb1_formal_authorization_independent_verification_exec_r2_v1",
        "status": "PASS",
        "pass": True,
        "AUTHORIZATION_VERIFICATION_PASS": True,
        "authorization_id": payload["authorization_id"],
        "authorization_sha256": sha256_file(auth),
        "lock_fingerprint": fingerprint,
        "lifecycle_state": lifecycle,
        "requested_mode": requested_mode,
        "workers": 2,
        "planned_trial_count": 360,
        "planned_open3d_count": 180,
        "planned_pcl_count": 180,
        "FORMAL_REGISTRATION_AUTHORIZED": True,
        "actual_formal_trials": 0,
        "failure_count": 0,
    }
