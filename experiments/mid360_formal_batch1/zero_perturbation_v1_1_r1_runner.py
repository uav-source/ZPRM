"""Locked FMB1 zero-perturbation v1.1-R1 runner.

Importing this module and running preflight/dry-run never imports a backend.
The real adapter is dynamically loaded only after the immutable lock, inputs,
execution commit, and a separate authorization have all passed.  Scientific
outcomes are write-once terminal records; only infrastructure failures may be
retried by ``resume``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .zero_perturbation_v1_1_r1_environment import (
    BACKEND_CONTRACT_SHA256,
    EXPECTED_VERSIONS,
    verify_environment_manifest,
)


LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r3.json"
LOCK_SCHEMA = "mid360_fmb1_zero_perturbation_formal_lock_v1_1_exec_r3"
PLAN_SCHEMA = "mid360_fmb1_zero_perturbation_trial_plan_v1_1_r1"
PLAN_ID = "FMB1_ZERO_PERTURBATION_TRIAL_PLAN_V1_1_R1"
RESULT_SCHEMA = "mid360_fmb1_zero_perturbation_trial_result_v1_1_r1"
AUTHORIZATION_SCHEMA = "mid360_fmb1_formal_registration_authorization_exec_r3_v1"
AUTHORIZATION_FILENAME = "formal_registration_authorization.json"
AUTHORIZATION_SCOPE = "ONE_FORMAL_EXECUTION_ATTEMPT_WITH_RESUME_ONLY"
PHYSICAL_REFERENCE_SEMANTICS = "NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT"
SCENE_CLASS = {
    "FMB1_R01": "RICH", "FMB1_R02": "RICH", "FMB1_R03": "RICH",
    "FMB1_W01": "WEAK", "FMB1_W02": "WEAK", "FMB1_W03": "WEAK",
}
BACKENDS = ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE")
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ATTEMPT_RE = re.compile(r"^attempt-(\d{4})\.json$")
START_RE = re.compile(r"^attempt-(\d{4})\.started\.json$")

# Independent verification AST-audits this exact future-execution boundary.
AUTHORIZED_EXECUTION_MODULES = (
    "numpy",
    "phase_a_harness.open3d_backend",
    "phase_a_harness.pcl_backend",
    "phase_a_harness.common_association_analysis",
    "phase_a_harness.rotation_metrics",
)
RESULT_VALIDATOR_MODULE = "experiments.mid360_formal_batch1.zero_perturbation_r1_trial_assets"
EXECUTION_CODE_PATHS = (
    "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_runner.py",
    "tools/mid360_formal_batch1/run_zero_perturbation_v1_1_r1.py",
    "experiments/__init__.py",
    "experiments/mid360_formal_batch1/__init__.py",
    "src/phase_a_harness/__init__.py",
    "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_environment.py",
    "experiments/mid360_formal_batch1/zero_perturbation_r1_trial_assets.py",
    "src/phase_a_harness/open3d_backend.py",
    "src/phase_a_harness/pcl_backend.py",
    "src/phase_a_harness/common_association_analysis.py",
    "src/phase_a_harness/rotation_metrics.py",
    "src/phase_a_harness/metrics.py",
    "src/phase_a_harness/types.py",
    "experiments/mid360_formal_batch1/authorization/formal_registration_authorization.py",
    "experiments/mid360_formal_batch1/authorization/formal_registration_authorization_verify.py",
    "experiments/mid360_formal_batch1/authorization/authorization_lifecycle.py",
    "experiments/mid360_formal_batch1/authorization/binding_provenance.py",
    "tools/mid360_formal_batch1/issue_formal_registration_authorization.py",
    "tools/mid360_formal_batch1/verify_formal_registration_authorization.py",
    "experiments/mid360_formal_batch1/zero_perturbation_v1_1_exec_r3_lock.py",
    "experiments/mid360_formal_batch1/zero_perturbation_v1_1_exec_r3_verify.py",
    "tools/mid360_formal_batch1/issue_zero_perturbation_v1_1_exec_r3_lock.py",
    "tools/mid360_formal_batch1/verify_zero_perturbation_v1_1_exec_r3_lock.py",
)


class R1RunnerError(RuntimeError):
    pass


class AuthorizedBackendInfrastructureError(RuntimeError):
    def __init__(self, status: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


ExecutionAdapter = Callable[
    [Mapping[str, Any], Path, Path, Mapping[str, Any], Path], Mapping[str, Any]
]
ResultValidator = Callable[..., Mapping[str, Any]]


def _fail(message: str) -> None:
    raise R1RunnerError(f"FMB1_R1_RUNNER_BLOCKED: {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"cannot read JSON {path}: {error}")
    if not isinstance(value, Mapping):
        _fail(f"expected JSON mapping: {path}")
    return value


def _identity(value: Any) -> bool:
    try:
        return len(value) == 4 and all(len(row) == 4 for row in value) and all(
            float(value[i][j]) == IDENTITY[i][j] for i in range(4) for j in range(4)
        )
    except (TypeError, ValueError, IndexError):
        return False


def _lexical(root: Path, value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail(f"{label} is lexically outside repository")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"{label} uses a symlink component: {cursor}")
    return path


def _resolve(root: Path, value: Any, label: str, *, directory: bool = False) -> Path:
    path = _lexical(root, value, label)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        _fail(f"{label} missing/escapes repository: {error}")
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        _fail(f"{label} has the wrong file type")
    return resolved


def _runtime_path(root: Path, value: Path) -> Path:
    path = _lexical(root, value, "runtime root")
    if path == root:
        _fail("runtime root is dangerously broad")
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        _fail("runtime root is a symlink or non-directory")
    return path


def _validate_resume_runtime_layout(runtime: Path) -> None:
    """Reject every non-canonical lifecycle object before a backend loader."""

    allowed = {
        "authorization", "run_contract.json", "trial_results", "inflight",
        "run_manifest.json",
    }
    entries = {entry.name: entry for entry in runtime.iterdir()}
    unexpected = set(entries) - allowed
    if unexpected:
        _fail(f"unexpected runtime-root artifacts: {sorted(unexpected)}")
    contract = entries.get("run_contract.json")
    if contract is None or contract.is_symlink() or not contract.is_file():
        _fail("resume run_contract.json is not a regular non-symlink file")
    for name in ("trial_results", "inflight"):
        entry = entries.get(name)
        if entry is not None and (entry.is_symlink() or not entry.is_dir()):
            _fail(f"resume {name} is not a regular non-symlink directory")
    manifest = entries.get("run_manifest.json")
    if manifest is not None and (manifest.is_symlink() or not manifest.is_file()):
        _fail("resume run_manifest.json is not a regular non-symlink file")
    authorization = entries.get("authorization")
    if authorization is None or authorization.is_symlink() or not authorization.is_dir():
        _fail("resume authorization directory is missing or invalid")


def _verify_core_checksums(lock_dir: Path) -> None:
    """Verify immutable lock-core coverage; never claim report self-coverage."""

    sums = lock_dir / "LOCK_CORE_SHA256SUMS"
    if not sums.is_file() or sums.is_symlink():
        _fail("LOCK_CORE_SHA256SUMS is missing or symlinked")
    declared: dict[str, str] = {}
    for number, line in enumerate(sums.read_text(encoding="ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/].*)", line)
        if match is None:
            _fail(f"malformed core checksum line {number}")
        digest, name = match.groups()
        if name in {".", ".."} or "/" in name or "\\" in name:
            _fail(f"unsafe core checksum path: {name}")
        if name in declared or name in {"LOCK_CORE_SHA256SUMS", "SHA256SUMS"}:
            _fail(f"duplicate/self core checksum entry: {name}")
        path = lock_dir / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
            _fail(f"lock core checksum differs: {name}")
        declared[name] = digest
    required = {
        LOCK_FILENAME, "formal_batch1_zero_perturbation_lock_v1_1_exec_r3.sha256",
        "lock_inventory.csv", "lock_fingerprint.json",
        "NO_REGISTRATION_ATTESTATION.json",
        "authorization_binding_provenance_defect_audit.json",
        "r3_execution_control_fix_report.json",
        "r3_authorization_fixture_qualification.json",
        "AUTHORIZATION_INVALIDATION_RECORD.json",
    }
    if not required.issubset(declared):
        _fail(f"core checksum coverage lacks: {sorted(required - set(declared))}")


def _validate_lock(root: Path, lock_dir: Path, *, remeasure_environment: bool) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _verify_core_checksums(lock_dir)
    lock_path = _resolve(root, lock_dir / LOCK_FILENAME, "formal lock")
    lock = _json(lock_path)
    if lock.get("schema") != LOCK_SCHEMA or lock.get("FORMAL_LOCK_ISSUED") is not True:
        _fail("formal R1 lock is absent or not issued")
    if lock.get("execution_lock_revision") != 3:
        _fail("formal execution lock revision is not Exec-R3")
    if (
        lock.get("binding_provenance_contract") != "EXPLICIT_PER_BINDING_V1"
        or lock.get("prefix_based_provenance_inference") is not False
    ):
        _fail("formal execution lock lacks explicit binding provenance")
    if lock.get("status") != "ISSUED_AWAITING_SEPARATE_AUTHORIZATION":
        _fail("formal lock lifecycle status differs")
    for key in (
        "AUTHORIZATION_PRODUCER_READY",
        "INDEPENDENT_AUTHORIZATION_VERIFIER_READY",
        "AUTHORIZATION_LIFECYCLE_QUALIFIED",
    ):
        if lock.get(key) is not True:
            _fail(f"Exec-R3 authorization infrastructure is not qualified: {key}")
    if lock.get("FORMAL_ICP_UNLOCKED") is not False or lock.get("FORMAL_REGISTRATION_AUTHORIZED") is not False:
        _fail("lock itself contains authorization/unlock")
    for key in ("actual_open3d_trials", "actual_pcl_trials", "actual_formal_trials", "registration_execution_count"):
        if int(lock.get(key, -1)) != 0:
            _fail(f"lock counter is nonzero: {key}")
    commit = lock.get("execution_code_commit")
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        _fail("lock execution-code commit is invalid")
    bindings = lock.get("bindings")
    if not isinstance(bindings, Mapping):
        _fail("lock bindings are missing")
    for name, row in bindings.items():
        if not isinstance(row, Mapping):
            _fail(f"lock binding is malformed: {name}")
        path = _resolve(root, row.get("repository_relative_path"), f"binding {name}")
        if _sha256(path) != row.get("sha256") or path.stat().st_size != int(row.get("bytes", -1)):
            _fail(f"lock binding changed: {name}")
    fingerprint = _json(_resolve(root, lock_dir / "lock_fingerprint.json", "lock fingerprint"))
    lock_sha = _sha256(lock_path)
    inventory = _resolve(root, lock_dir / "lock_inventory.csv", "lock inventory")
    material = {"lock_file_sha256": lock_sha,
                "lock_inventory_file_sha256": _sha256(inventory),
                "execution_code_commit": commit}
    reproduced = hashlib.sha256(_canonical(material)).hexdigest()
    if fingerprint.get("lock_file_sha256") != lock_sha or fingerprint.get("lock_fingerprint") != reproduced:
        _fail("lock fingerprint is not independently reproducible")
    environment_path = _resolve(root, bindings["environment_manifest"]["repository_relative_path"], "environment manifest")
    verify_environment_manifest(_json(environment_path), root, remeasure_versions=remeasure_environment)
    return lock, fingerprint


def _validate_plan(root: Path, path: Path, lock: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, int]]:
    plan = _json(path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("plan_id") != PLAN_ID or plan.get("track_id") != "ZERO_PERTURBATION_TRACK":
        _fail("trial plan schema/id/track differs")
    if _sha256(path) != lock["bindings"]["trial_plan_json"]["sha256"]:
        _fail("lock does not bind this trial plan")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 360 or any(not isinstance(row, Mapping) for row in rows):
        _fail("trial plan must contain 360 mapping rows")
    ids: set[str] = set()
    pairs: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    backend_counts: Counter[str] = Counter()
    scenes, stations = set(), set()
    for index, row in enumerate(rows):
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id or trial_id in ids:
            _fail(f"duplicate/invalid trial ID at row {index}")
        ids.add(trial_id)
        scene, station, snapshot = str(row.get("scene_id")), str(row.get("station_id")), str(row.get("snapshot_id"))
        backend = str(row.get("backend"))
        if scene not in SCENE_CLASS or row.get("final_geometry_class") != SCENE_CLASS[scene]:
            _fail(f"non-final scene/class at row {index}")
        if int(row.get("attempt", -1)) != (2 if scene == "FMB1_W02" else 1):
            _fail(f"wrong acquisition attempt at row {index}")
        if backend not in BACKENDS or row.get("track_id") != "ZERO_PERTURBATION_TRACK":
            _fail(f"backend/track differs at row {index}")
        version_key = "open3d" if backend == BACKENDS[0] else "pcl"
        if row.get("backend_version") != EXPECTED_VERSIONS[version_key]:
            _fail(f"backend version differs at row {index}")
        if not _identity(row.get("T0")) or not _identity(row.get("T_reference_nominal")):
            _fail(f"non-identity transform at row {index}")
        if float(row.get("translation_perturbation_m", math.nan)) != 0.0 or float(row.get("rotation_perturbation_deg", math.nan)) != 0.0:
            _fail(f"capture-radius perturbation entered row {index}")
        if row.get("backend_parameter_contract_sha256") != BACKEND_CONTRACT_SHA256:
            _fail(f"backend contract differs at row {index}")
        if not isinstance(row.get("backend_canonical_parameter_sha256"), str) or SHA_RE.fullmatch(str(row.get("backend_canonical_parameter_sha256"))) is None:
            _fail(f"canonical parameter SHA missing at row {index}")
        if row.get("active_amendment_sha256") != lock["bindings"]["active_amendment"]["sha256"] or row.get("analysis_contract_sha256") != lock["bindings"]["analysis_contract"]["sha256"]:
            _fail(f"protocol hash differs at row {index}")
        for prefix in ("source", "target"):
            payload = _resolve(root, row.get(f"{prefix}_reference"), f"row {index} {prefix}")
            if _sha256(payload) != row.get(f"{prefix}_sha256"):
                _fail(f"row {index} {prefix} file SHA differs")
            if not isinstance(row.get(f"{prefix}_array_sha256"), str) or SHA_RE.fullmatch(str(row.get(f"{prefix}_array_sha256"))) is None:
                _fail(f"row {index} {prefix} array SHA differs")
            count = row.get(f"{prefix}_point_count")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                _fail(f"row {index} {prefix} point count differs")
        if row.get("planned_status") != "PLANNED_NOT_AUTHORIZED_NOT_EXECUTED":
            _fail(f"planned status differs at row {index}")
        pairs[snapshot].append(row); backend_counts[backend] += 1
        scenes.add(scene); stations.add((scene, station))
    if scenes != set(SCENE_CLASS) or len(stations) != 18 or len(pairs) != 180 or backend_counts != Counter({BACKENDS[0]: 180, BACKENDS[1]: 180}):
        _fail("plan hierarchy/counts differ")
    shared = ("scene_id", "station_id", "attempt", "snapshot_id", "query_timestamp",
              "source_reference", "source_sha256", "source_array_sha256", "source_point_count",
              "target_reference", "target_sha256", "target_array_sha256", "target_point_count")
    for snapshot, pair in pairs.items():
        if len(pair) != 2 or {row["backend"] for row in pair} != set(BACKENDS):
            _fail(f"backend pair differs for {snapshot}")
        if any(pair[0].get(field) != pair[1].get(field) for field in shared):
            _fail(f"byte-identical backend input binding differs for {snapshot}")
    return plan, {"trial_count": 360, "open3d_trial_count": 180,
                  "pcl_trial_count": 180, "snapshot_count": 180,
                  "scene_count": 6, "station_count": 18}


def _validate_authorization(
    root: Path, path: Path, *, lock_dir: Path, lock: Mapping[str, Any],
    fingerprint: Mapping[str, Any], plan_path: Path, runtime_root: Path,
    mode: str, workers: int,
) -> tuple[bool, bool, Mapping[str, Any] | None, Path | None]:
    lexical = _lexical(root, path, "formal authorization")
    if not lexical.exists():
        return False, False, None, lexical
    resolved = _resolve(root, lexical, "formal authorization")
    saved_report_path = resolved.parent / "authorization_verification_report.json"
    saved_report = _json(_resolve(root, saved_report_path, "authorization verifier report"))
    if (
        saved_report.get("AUTHORIZATION_VERIFICATION_PASS") is not True
        or saved_report.get("authorization_sha256") != _sha256(resolved)
        or saved_report.get("lock_fingerprint") != fingerprint.get("lock_fingerprint")
    ):
        _fail("saved independent authorization verifier report is not PASS/bound")
    try:
        from .authorization.formal_registration_authorization_verify import (
            verify_formal_registration_authorization,
        )
    except Exception as error:
        _fail(f"independent authorization verifier is unavailable: {error}")
    try:
        report = verify_formal_registration_authorization(
            root,
            lock_dir=lock_dir,
            authorization_path=resolved,
            runtime_root=runtime_root,
            requested_mode=mode,
            workers=workers,
        )
    except Exception as error:
        _fail(f"independent authorization verification failed: {error}")
    payload = _json(resolved)
    valid = bool(
        report.get("AUTHORIZATION_VERIFICATION_PASS") is True
        and report.get("lock_fingerprint") == fingerprint.get("lock_fingerprint")
        and payload.get("schema") == AUTHORIZATION_SCHEMA
        and payload.get("authorization_scope") == AUTHORIZATION_SCOPE
        and payload.get("trial_plan_sha256") == _sha256(plan_path)
        and payload.get("execution_code_commit") == lock.get("execution_code_commit")
    )
    return True, valid, payload, resolved


def _verify_execution_code_commit(root: Path, commit: str) -> None:
    try:
        subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for relative in EXECUTION_CODE_PATHS:
            current = _resolve(root, relative, f"execution code {relative}")
            recorded = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root,
                                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            if hashlib.sha256(recorded).hexdigest() != _sha256(current):
                _fail(f"execution code differs from locked commit: {relative}")
    except subprocess.CalledProcessError as error:
        _fail(f"execution-code commit cannot be authenticated: {error.stderr.decode('utf-8', 'replace')[-500:]}")


def _load_result_validator() -> ResultValidator:
    import importlib
    module = importlib.import_module(RESULT_VALIDATOR_MODULE)
    validator = getattr(module, "validate_result_row", None)
    if not callable(validator):
        _fail("formal result validator is unavailable")
    return validator


def _load_authorized_execution_adapter() -> ExecutionAdapter:
    """Load the real adapter only from the post-authorization call path."""

    import importlib
    modules = {name: importlib.import_module(name) for name in AUTHORIZED_EXECUTION_MODULES}
    np = modules["numpy"]
    o3d = modules["phase_a_harness.open3d_backend"]
    pcl = modules["phase_a_harness.pcl_backend"]
    common = modules["phase_a_harness.common_association_analysis"]
    rotation = modules["phase_a_harness.rotation_metrics"]

    def load_array(path: Path, array_sha: str, count: int) -> Any:
        try:
            value = np.load(path, allow_pickle=False)
        except Exception as error:
            raise AuthorizedBackendInfrastructureError(
                "FILE_READ_CORRUPTION", f"{type(error).__name__}: {error}"
            ) from error
        if (value.dtype != np.dtype("<f8") or not value.flags.c_contiguous
                or value.ndim != 2 or value.shape != (count, 3)
                or not np.all(np.isfinite(value))):
            raise AuthorizedBackendInfrastructureError(
                "FILE_READ_CORRUPTION", "canonical array contract differs"
            )
        observed = hashlib.sha256(
            np.ascontiguousarray(value, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        if observed != array_sha:
            raise AuthorizedBackendInfrastructureError(
                "FILE_READ_CORRUPTION", "canonical array SHA differs"
            )
        return value

    def pose_metrics(initial: Any, estimated: Any) -> dict[str, Any]:
        # Keep the general formula even though the frozen R1 plan requires I.
        delta = np.linalg.inv(initial) @ estimated
        audit = rotation.rotation_metric_audit(delta[:3, :3], np.eye(3))
        if audit.get("rotation_matrix_quality_pass") is not True:
            raise ValueError("estimated rotation failed quality audit")
        translation = delta[:3, 3]
        angle = float(audit["rotation_error_rad"])
        return {
            "T_est": estimated.tolist(), "Delta_T": delta.tolist(),
            "translation_x_m": float(translation[0]),
            "translation_y_m": float(translation[1]),
            "translation_z_m": float(translation[2]),
            "translation_norm_m": float(np.linalg.norm(translation)),
            "rotation_angle_rad": angle,
            "rotation_angle_deg": float(math.degrees(angle)),
        }

    association_fields = (
        "initial_correspondence_count", "initial_valid_normal_correspondence_count",
        "final_correspondence_count", "final_valid_normal_correspondence_count",
        "correspondence_turnover", "accepted_source_turnover",
        "correspondence_count_change_ratio", "initial_residual_rmse",
        "final_residual_rmse", "residual_rmse_change",
        "median_normal_angle_change_deg", "q95_normal_angle_change_deg",
    )

    def adapter(trial: Mapping[str, Any], source_path: Path, target_path: Path,
                contract: Mapping[str, Any], pcl_executable: Path) -> Mapping[str, Any]:
        source = load_array(source_path, str(trial["source_array_sha256"]),
                            int(trial["source_point_count"]))
        target = load_array(target_path, str(trial["target_array_sha256"]),
                            int(trial["target_point_count"]))
        initial = np.asarray(trial["T0"], dtype=np.float64)
        backend = str(trial["backend"])
        try:
            if backend == BACKENDS[0]:
                parameters = contract["open3d"]["parameters"]
                result = o3d.run_open3d_full(
                    source, target, initial,
                    {"registration_method": "point_to_plane",
                     "maximum_correspondence_distance_m": float(parameters["maximum_correspondence_distance_m"]),
                     "target_normal_estimation": dict(parameters["target_normal_estimation"]),
                     "icp_convergence": dict(parameters["convergence"])},
                    backend_seed=0, input_checksum=str(trial["source_array_sha256"]),
                )
                estimated = np.asarray(result.final_pose, dtype=np.float64)
                solver_success = bool(result.solver_converged)
                finite = bool(result.finite_result and np.all(np.isfinite(estimated)))
                solver_status = str(result.termination_reason)
            elif backend == BACKENDS[1]:
                reference_sha = hashlib.sha256(
                    np.ascontiguousarray(initial, dtype="<f8").tobytes(order="C")
                ).hexdigest()
                checksums = {
                    "source_checksum": str(trial["source_array_sha256"]),
                    "target_checksum": str(trial["target_array_sha256"]),
                    "reference_pose_checksum": reference_sha,
                    "snapshot_checksum": hashlib.sha256(
                        (f"fmb1-r1|{trial['trial_id']}|{trial['source_array_sha256']}|"
                         f"{trial['target_array_sha256']}|{reference_sha}").encode()
                    ).hexdigest(),
                }
                result = pcl.run_pcl_point_to_plane(
                    source, target, initial, trial_id=str(trial["trial_id"]),
                    checksums=checksums, executable=pcl_executable,
                    parameters=pcl.frozen_parameters(contract["pcl"]["parameters"]),
                )
                estimated = (None if result.final_transformation is None
                             else np.asarray(result.final_transformation, dtype=np.float64))
                solver_success = bool(result.has_converged)
                finite = bool(result.finite_output and estimated is not None
                              and np.all(np.isfinite(estimated)))
                solver_status = str(result.failure_reason or
                                    ("completed_finite_correspondences" if solver_success
                                     else "solver_non_convergence"))
            else:
                raise ValueError(f"unsupported backend {backend}")
        except AuthorizedBackendInfrastructureError:
            raise
        except Exception as error:
            status = ("PCL_EXECUTABLE_INFRASTRUCTURE_FAILURE"
                      if backend == BACKENDS[1] else "PROCESS_CRASH")
            raise AuthorizedBackendInfrastructureError(
                status, f"{type(error).__name__}: {error}"
            ) from error
        null_pose = {name: None for name in (
            "T_est", "Delta_T", "translation_x_m", "translation_y_m",
            "translation_z_m", "translation_norm_m", "rotation_angle_rad",
            "rotation_angle_deg",
        )}
        if not finite or estimated is None:
            return {**null_pose, **{name: None for name in association_fields},
                    "common_association_valid": None,
                    "common_association_invalid_reason": None,
                    "common_association_invalid_detail": None,
                    "solver_success": solver_success, "finite_result": False,
                    "solver_status": solver_status or "NONFINITE_RESULT"}
        try:
            pose = pose_metrics(initial, estimated)
        except Exception:
            return {**null_pose, **{name: None for name in association_fields},
                    "common_association_valid": None,
                    "common_association_invalid_reason": None,
                    "common_association_invalid_detail": None,
                    "solver_success": solver_success, "finite_result": False,
                    "solver_status": "NONFINITE_OR_INVALID_TRANSFORM"}
        context = common.prepare_common_association_context(
            source, target, initial, snapshot_id=str(trial["snapshot_id"])
        )
        metrics = common.safe_analyze_estimated_transform(
            context, estimated,
            identifiers={"trial_id": str(trial["trial_id"]), "backend": backend},
        )
        return {**pose, **{name: metrics.get(name) for name in association_fields},
                "common_association_valid": metrics.get("common_association_valid"),
                "common_association_invalid_reason": metrics.get(
                    "common_association_invalid_reason"
                ),
                "common_association_invalid_detail": (
                    str(metrics["common_association_invalid_detail"])[:1000]
                    if metrics.get("common_association_invalid_detail") else None
                ),
                "solver_success": solver_success, "finite_result": True,
                "solver_status": solver_status or
                                 ("CONVERGED" if solver_success else "SOLVER_NON_CONVERGENCE")}

    return adapter


def _base_result(trial: Mapping[str, Any], *, lock: Mapping[str, Any], plan_sha: str,
                 lock_sha: str, environment: Mapping[str, Any], environment_sha: str,
                 created_at_utc: str) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA, "trial_id": trial["trial_id"],
        "batch_id": trial["batch_id"], "amendment_id": trial["amendment_id"],
        "track_id": trial["track_id"], "scene_id": trial["scene_id"],
        "geometry_class": trial["final_geometry_class"], "station_id": trial["station_id"],
        "attempt": trial["attempt"], "snapshot_id": trial["snapshot_id"],
        "backend": trial["backend"], "backend_version": trial["backend_version"],
        "source_reference": trial["source_reference"], "source_sha256": trial["source_sha256"],
        "target_reference": trial["target_reference"], "target_sha256": trial["target_sha256"],
        "T0": trial["T0"], "T_reference_nominal": trial["T_reference_nominal"],
        "backend_parameter_contract_sha256": trial["backend_parameter_contract_sha256"],
        "backend_canonical_parameter_sha256": trial["backend_canonical_parameter_sha256"],
        "active_amendment_sha256": trial["active_amendment_sha256"],
        "analysis_contract_sha256": trial["analysis_contract_sha256"],
        "trial_plan_sha256": plan_sha, "formal_lock_sha256": lock_sha,
        "code_commit": lock["execution_code_commit"],
        "environment_identity": {**dict(environment["versions"]),
                                 "environment_manifest_sha256": environment_sha},
        "physical_reference_semantics": PHYSICAL_REFERENCE_SEMANTICS,
        "execution_kind": "FORMAL", "fixture_only": False,
        "created_at_utc": created_at_utc,
    }


RESULT_METRICS = (
    "T_est", "Delta_T", "translation_x_m", "translation_y_m", "translation_z_m",
    "translation_norm_m", "rotation_angle_rad", "rotation_angle_deg",
    "initial_correspondence_count", "initial_valid_normal_correspondence_count",
    "final_correspondence_count", "final_valid_normal_correspondence_count",
    "correspondence_turnover", "accepted_source_turnover",
    "correspondence_count_change_ratio", "initial_residual_rmse",
    "final_residual_rmse", "residual_rmse_change",
    "median_normal_angle_change_deg", "q95_normal_angle_change_deg",
    "common_association_valid", "common_association_invalid_reason",
    "common_association_invalid_detail",
)


def _scientific_result(trial: Mapping[str, Any], core: Mapping[str, Any], **base: Any) -> dict[str, Any]:
    finite, converged = core.get("finite_result") is True, core.get("solver_success") is True
    status = ("COMPLETED" if finite and converged else
              "SOLVER_NON_CONVERGENCE" if finite else
              "NONFINITE_RESULT_RECORDED_WITHOUT_NONFINITE_TRANSFORM")
    return {**_base_result(trial, **base), **{name: core.get(name) for name in RESULT_METRICS},
            "solver_status": str(core.get("solver_status") or status),
            "finite_result": finite, "scientific_status": status,
            "infrastructure_status": "OK", "retry_eligible": False, "retry_reason": None}


def _infrastructure_result(trial: Mapping[str, Any], *, status: str, detail: str,
                           **base: Any) -> dict[str, Any]:
    allowed = {"PROCESS_CRASH", "FILE_READ_CORRUPTION", "RESULT_SCHEMA_WRITE_FAILURE",
               "PCL_EXECUTABLE_INFRASTRUCTURE_FAILURE"}
    if status not in allowed:
        _fail(f"unsupported infrastructure status {status}")
    return {**_base_result(trial, **base), **{name: None for name in RESULT_METRICS},
            "solver_status": f"{status}:{detail}"[:1000], "finite_result": False,
            "scientific_status": "NOT_EVALUATED_INFRASTRUCTURE_FAILURE",
            "infrastructure_status": status, "retry_eligible": True, "retry_reason": status}


def _atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    content = _canonical(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            _fail(f"write-once artifact differs: {path}")
        return
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    if temporary.exists():
        _fail(f"stale runtime temporary exists: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != content:
            _fail(f"concurrent write-once artifact differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _attempt_paths(runtime: Path, trial_id: str, attempt: int) -> tuple[Path, Path]:
    return (runtime / "inflight" / trial_id / f"attempt-{attempt:04d}.started.json",
            runtime / "trial_results" / trial_id / f"attempt-{attempt:04d}.json")


def _validate_start_marker(
    marker: Path, *, trial_id: str, attempt: int, authorization_sha: str,
) -> Mapping[str, Any]:
    if marker.is_symlink() or not marker.is_file():
        _fail(f"result attempt lacks regular start marker: {marker}")
    payload = _json(marker)
    required = {
        "schema", "trial_id", "attempt_number", "authorization_sha256",
        "started_at_utc",
    }
    if (
        set(payload) != required
        or payload.get("schema")
        != "mid360_fmb1_formal_trial_attempt_start_v1_1_r1"
        or payload.get("trial_id") != trial_id
        or payload.get("attempt_number") != attempt
        or payload.get("authorization_sha256") != authorization_sha
    ):
        _fail(f"start marker identity/authorization differs: {marker}")
    try:
        started = datetime.fromisoformat(
            str(payload.get("started_at_utc")).replace("Z", "+00:00")
        )
    except ValueError:
        _fail(f"start marker timestamp is invalid: {marker}")
    if started.tzinfo is None or started.utcoffset() is None:
        _fail(f"start marker timestamp lacks timezone: {marker}")
    return payload


def _scan_attempts(runtime: Path, plan: Mapping[str, Any], validator: ResultValidator,
                   *, plan_sha: str, lock_sha: str, code_commit: str,
                   environment_sha: str,
                   authorization_sha: str) -> dict[str, list[Mapping[str, Any]]]:
    expected = {str(row["trial_id"]): row for row in plan["rows"]}
    output: dict[str, list[Mapping[str, Any]]] = {trial_id: [] for trial_id in expected}
    root = runtime / "trial_results"
    if not root.exists():
        return output
    if root.is_symlink() or not root.is_dir():
        _fail("trial_results is not a regular directory")
    for directory in root.iterdir():
        if directory.is_symlink() or not directory.is_dir() or directory.name not in expected:
            _fail(f"orphan/symlink result directory: {directory}")
        indexed: list[tuple[int, Path]] = []
        for path in directory.iterdir():
            match = ATTEMPT_RE.fullmatch(path.name)
            if match is None or path.is_symlink() or not path.is_file():
                _fail(f"unexpected trial-result artifact: {path}")
            indexed.append((int(match.group(1)), path))
        indexed.sort()
        if [number for number, _ in indexed] != list(range(1, len(indexed) + 1)):
            _fail(f"non-contiguous result attempts: {directory.name}")
        terminal = False
        for number, path in indexed:
            marker, _ = _attempt_paths(runtime, directory.name, number)
            _validate_start_marker(
                marker, trial_id=directory.name, attempt=number,
                authorization_sha=authorization_sha,
            )
            row = _json(path)
            if row.get("trial_id") != directory.name:
                _fail(f"trial-result identity mismatch: {path}")
            try:
                validated = validator(row, plan, trial_plan_sha256=plan_sha,
                                      formal_lock_sha256=lock_sha)
            except Exception as error:
                _fail(f"trial-result schema/binding failure {path}: {error}")
            if (validated.get("code_commit") != code_commit
                    or validated.get("environment_identity", {}).get(
                        "environment_manifest_sha256"
                    ) != environment_sha):
                _fail(f"trial-result execution/environment binding differs: {path}")
            if terminal:
                _fail(f"scientific result was retried: {directory.name}")
            terminal = validated.get("infrastructure_status") == "OK"
            output[directory.name].append(validated)
    return output


def _reconcile_interrupted(runtime: Path, plan: Mapping[str, Any],
                           attempts: dict[str, list[Mapping[str, Any]]],
                           validator: ResultValidator, base: Mapping[str, Any],
                           *, plan_sha: str, lock_sha: str,
                           authorization_sha: str) -> None:
    expected = {str(row["trial_id"]): row for row in plan["rows"]}
    root = runtime / "inflight"
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        _fail("inflight is not a regular directory")
    pending: list[tuple[str, Path, Mapping[str, Any]]] = []
    for directory in root.iterdir():
        if directory.is_symlink() or not directory.is_dir() or directory.name not in expected:
            _fail(f"orphan/symlink inflight directory: {directory}")
        indexed_markers: list[tuple[int, Path]] = []
        for marker in directory.iterdir():
            match = START_RE.fullmatch(marker.name)
            if match is None or marker.is_symlink() or not marker.is_file():
                _fail(f"unexpected inflight artifact: {marker}")
            number = int(match.group(1))
            indexed_markers.append((number, marker))
        indexed_markers.sort()
        if [number for number, _ in indexed_markers] != list(
            range(1, len(indexed_markers) + 1)
        ):
            _fail(f"non-contiguous start-marker attempts: {directory.name}")
        unmatched = 0
        for number, marker in indexed_markers:
            _validate_start_marker(
                marker, trial_id=directory.name, attempt=number,
                authorization_sha=authorization_sha,
            )
            _, result_path = _attempt_paths(runtime, directory.name, number)
            if result_path.exists():
                continue
            unmatched += 1
            if unmatched > 1:
                _fail(f"multiple orphan start markers: {directory.name}")
            if any(
                row.get("infrastructure_status") == "OK"
                for row in attempts[directory.name]
            ):
                _fail(
                    "terminal scientific outcome has an extra orphan start marker: "
                    f"{directory.name}"
                )
            if number != len(attempts[directory.name]) + 1:
                _fail(f"interrupted attempt sequence differs: {directory.name}")
            pending.append((directory.name, result_path, expected[directory.name]))

    # Validation is deliberately two-phase: no reconciliation result is
    # written until every marker in the authoritative runtime has passed.
    for trial_id, result_path, trial in pending:
        row = _infrastructure_result(
            trial, status="PROCESS_CRASH",
            detail="previous authorized process ended without an atomic result",
            created_at_utc=datetime.now(timezone.utc).isoformat(), **base,
        )
        try:
            validated = validator(row, plan, trial_plan_sha256=plan_sha,
                                  formal_lock_sha256=lock_sha)
        except Exception as error:
            _fail(f"cannot retain interrupted attempt: {error}")
        _atomic_create(result_path, validated)
        attempts[trial_id].append(validated)


def _terminal_trial_ids(
    attempts: Mapping[str, list[Mapping[str, Any]]],
) -> set[str]:
    terminal: set[str] = set()
    for trial_id, rows in attempts.items():
        ok_count = sum(row.get("infrastructure_status") == "OK" for row in rows)
        if ok_count > 1:
            _fail(f"multiple terminal scientific outcomes exist: {trial_id}")
        if ok_count == 1:
            terminal.add(trial_id)
    return terminal


def _complete_run_manifest(
    plan: Mapping[str, Any], attempts: Mapping[str, list[Mapping[str, Any]]],
    *, lock_sha: str, lock_fingerprint: str, plan_sha: str, code_commit: str,
    authorization_sha: str, environment_sha: str,
) -> dict[str, Any]:
    by_id = {str(row["trial_id"]): row for row in plan["rows"]}
    terminal = _terminal_trial_ids(attempts)
    if terminal != set(by_id):
        _fail("complete manifest requested without 360 terminal scientific outcomes")
    counts = Counter(by_id[trial_id]["backend"] for trial_id in terminal)
    attempt_count = sum(len(rows) for rows in attempts.values())
    infrastructure_attempt_count = sum(
        row.get("infrastructure_status") != "OK"
        for rows in attempts.values() for row in rows
    )
    return {
        "schema": "mid360_fmb1_zero_perturbation_formal_run_v1_1_r1",
        "status": "COMPLETE",
        "mode": "resume" if attempt_count > 360 else "fresh",
        "formal_lock_sha256": lock_sha,
        "lock_fingerprint": lock_fingerprint,
        "trial_plan_sha256": plan_sha,
        "execution_code_commit": code_commit,
        "authorization_sha256": authorization_sha,
        "environment_manifest_sha256": environment_sha,
        "completed_trial_count": 360,
        "completed_open3d_trial_count": counts[BACKENDS[0]],
        "completed_pcl_trial_count": counts[BACKENDS[1]],
        "retryable_trial_count": 0,
        "retryable_trial_ids": [],
        "formal_attempt_record_count": attempt_count,
        "infrastructure_attempt_count": infrastructure_attempt_count,
        "scientific_result_retry_count": 0,
        "FORMAL_REGISTRATION_AUTHORIZED": True,
        "physical_reference_semantics": PHYSICAL_REFERENCE_SEMANTICS,
    }


def _execute_authorized_run(
    root: Path, *, lock_dir: Path, lock: Mapping[str, Any],
    fingerprint: Mapping[str, Any], plan_path: Path, plan: Mapping[str, Any],
    authorization_path: Path, authorization: Mapping[str, Any], runtime_root: Path,
    workers: int, mode: str, execution_adapter: ExecutionAdapter | None,
    result_validator: ResultValidator | None,
) -> dict[str, Any]:
    """Future real execution entry; caller has already authenticated authority."""

    runtime = _runtime_path(root, runtime_root)
    authorization_dir = runtime / "authorization"
    try:
        authorization_path.relative_to(authorization_dir)
    except ValueError:
        _fail("formal authorization is not inside the canonical runtime")
    if mode == "fresh":
        if not runtime.is_dir() or runtime.is_symlink():
            _fail("fresh requires the producer-created canonical runtime")
        entries = {entry.name: entry for entry in runtime.iterdir()}
        if set(entries) != {"authorization"}:
            _fail("fresh runtime contains non-authorization artifacts")
        if authorization_dir.is_symlink() or not authorization_dir.is_dir():
            _fail("fresh authorization directory is invalid")
        allowed_authorization = {
            "formal_registration_authorization.json",
            "formal_registration_authorization.sha256",
            "authorization_verification_report.json",
        }
        if {entry.name for entry in authorization_dir.iterdir()} != allowed_authorization:
            _fail("fresh authorization directory contains unexpected lifecycle artifacts")
    elif not runtime.is_dir():
        _fail("resume requires an existing runtime directory")
    else:
        _validate_resume_runtime_layout(runtime)
    validator = result_validator or _load_result_validator()
    lock_sha, plan_sha = _sha256(lock_dir / LOCK_FILENAME), _sha256(plan_path)
    environment_path = _resolve(
        root, lock["bindings"]["environment_manifest"]["repository_relative_path"],
        "environment manifest",
    )
    environment = _json(environment_path)
    environment_sha = _sha256(environment_path)
    authorization_sha = _sha256(authorization_path)
    from .authorization.authorization_lifecycle import (
        consume_authorization,
        mark_authorization_in_use,
    )
    if mode == "fresh":
        in_use = mark_authorization_in_use(
            authorization_path,
            lock_fingerprint=str(fingerprint["lock_fingerprint"]),
            runtime_relative=str(lock["authoritative_runtime_root"]),
        )
    else:
        in_use = _json(authorization_dir / "authorization_in_use.json")
    base = {"lock": lock, "plan_sha": plan_sha, "lock_sha": lock_sha,
            "environment": environment, "environment_sha": environment_sha}
    run_contract = {
        "schema": "mid360_fmb1_zero_perturbation_run_contract_v1_1_r1",
        "lock_file_sha256": lock_sha, "lock_fingerprint": fingerprint["lock_fingerprint"],
        "trial_plan_sha256": plan_sha, "authorization_sha256": authorization_sha,
        "authorization_id": authorization["authorization_id"],
        "execution_code_commit": lock["execution_code_commit"],
        "planned_trial_count": 360, "workers_at_fresh_start": workers,
        "scientific_results_are_terminal": True,
        "retry_policy": "INFRASTRUCTURE_FAILURE_ONLY",
    }
    if mode == "fresh":
        runtime.mkdir(parents=True, exist_ok=True)
        _atomic_create(runtime / "run_contract.json", run_contract)
    else:
        if _json(runtime / "run_contract.json") != run_contract:
            # Workers are an operational choice and may differ on resume.
            recorded = dict(_json(runtime / "run_contract.json"))
            candidate = dict(run_contract)
            candidate["workers_at_fresh_start"] = recorded.get("workers_at_fresh_start")
            if recorded != candidate:
                _fail("resume run contract differs from locked/authorized inputs")
    attempts = _scan_attempts(
        runtime, plan, validator, plan_sha=plan_sha, lock_sha=lock_sha,
        code_commit=str(lock["execution_code_commit"]),
        environment_sha=environment_sha,
        authorization_sha=authorization_sha,
    )
    manifest_path = runtime / "run_manifest.json"
    terminal = _terminal_trial_ids(attempts)
    if manifest_path.exists() and len(terminal) != 360:
        _fail("run manifest exists before all 360 terminal scientific outcomes")
    _reconcile_interrupted(runtime, plan, attempts, validator, base,
                           plan_sha=plan_sha, lock_sha=lock_sha,
                           authorization_sha=authorization_sha)
    terminal = _terminal_trial_ids(attempts)
    if len(terminal) == 360:
        expected_manifest = _complete_run_manifest(
            plan, attempts, lock_sha=lock_sha,
            lock_fingerprint=str(fingerprint["lock_fingerprint"]),
            plan_sha=plan_sha, code_commit=str(lock["execution_code_commit"]),
            authorization_sha=authorization_sha,
            environment_sha=environment_sha,
        )
        if manifest_path.exists():
            if _json(manifest_path) != expected_manifest:
                _fail("completed resume manifest differs from verified attempts/lock")
        else:
            _atomic_create(manifest_path, expected_manifest)
        consume_authorization(
            authorization_path,
            lock_fingerprint=str(fingerprint["lock_fingerprint"]),
            first_backend_invocation_utc=str(in_use["started_at_utc"]),
            last_backend_invocation_utc=datetime.now(timezone.utc).isoformat(),
            actual_open3d_trials=180,
            actual_pcl_trials=180,
            execution_result_manifest_sha256=_sha256(manifest_path),
        )
        return expected_manifest

    # First backend import/instantiation point.  No preflight/dry-run path calls it.
    adapter = execution_adapter or _load_authorized_execution_adapter()
    parameter_contract = _json(_resolve(
        root, lock["bindings"]["backend_parameter_contract"]["repository_relative_path"],
        "backend parameter contract",
    ))
    pcl_executable = _resolve(
        root, lock["bindings"]["pcl_executable"]["repository_relative_path"],
        "PCL executable",
    )
    by_id = {str(row["trial_id"]): row for row in plan["rows"]}
    pending = [row for row in plan["rows"] if str(row["trial_id"]) not in terminal]

    def run_one(trial: Mapping[str, Any]) -> tuple[str, bool, bool]:
        trial_id = str(trial["trial_id"])
        number = len(attempts[trial_id]) + 1
        marker, result_path = _attempt_paths(runtime, trial_id, number)
        _atomic_create(marker, {
            "schema": "mid360_fmb1_formal_trial_attempt_start_v1_1_r1",
            "trial_id": trial_id, "attempt_number": number,
            "authorization_sha256": authorization_sha,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        created, invoked = datetime.now(timezone.utc).isoformat(), False
        try:
            source = _resolve(root, trial["source_reference"], f"{trial_id} source")
            target = _resolve(root, trial["target_reference"], f"{trial_id} target")
            invoked = True
            core = adapter(trial, source, target, parameter_contract, pcl_executable)
            result = _scientific_result(trial, core, created_at_utc=created, **base)
        except AuthorizedBackendInfrastructureError as error:
            result = _infrastructure_result(trial, status=error.status, detail=error.detail,
                                            created_at_utc=created, **base)
        except Exception as error:
            result = _infrastructure_result(
                trial, status="PROCESS_CRASH", detail=f"{type(error).__name__}: {error}",
                created_at_utc=created, **base,
            )
        try:
            validated = validator(result, plan, trial_plan_sha256=plan_sha,
                                  formal_lock_sha256=lock_sha)
        except Exception as error:
            fallback = _infrastructure_result(
                trial, status="RESULT_SCHEMA_WRITE_FAILURE",
                detail=f"{type(error).__name__}: {error}", created_at_utc=created, **base,
            )
            try:
                validated = validator(fallback, plan, trial_plan_sha256=plan_sha,
                                      formal_lock_sha256=lock_sha)
            except Exception as second:
                _fail(f"result validator cannot retain infrastructure failure: {second}")
        _atomic_create(result_path, validated)
        return trial_id, invoked, validated.get("infrastructure_status") == "OK"

    from concurrent.futures import ThreadPoolExecutor
    outcomes: list[tuple[str, bool, bool]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fmb1-r1") as pool:
        outcomes.extend(pool.map(run_one, pending))
    attempts = _scan_attempts(
        runtime, plan, validator, plan_sha=plan_sha, lock_sha=lock_sha,
        code_commit=str(lock["execution_code_commit"]),
        environment_sha=environment_sha,
        authorization_sha=authorization_sha,
    )
    terminal = _terminal_trial_ids(attempts)
    retryable = sorted(set(by_id) - terminal)
    counts = Counter(by_id[trial_id]["backend"] for trial_id in terminal)
    report = {
        "schema": "mid360_fmb1_zero_perturbation_formal_run_v1_1_r1",
        "status": ("COMPLETE" if len(terminal) == 360
                   else "INCOMPLETE_RETRYABLE_INFRASTRUCTURE_FAILURES"),
        "mode": mode, "lock_fingerprint": fingerprint["lock_fingerprint"],
        "trial_plan_sha256": plan_sha, "execution_code_commit": lock["execution_code_commit"],
        "completed_trial_count": len(terminal),
        "completed_open3d_trial_count": counts[BACKENDS[0]],
        "completed_pcl_trial_count": counts[BACKENDS[1]],
        "retryable_trial_count": len(retryable), "retryable_trial_ids": retryable,
        "backend_invocation_attempt_count_this_process": sum(item[1] for item in outcomes),
        "scientific_result_retry_count": 0,
        "FORMAL_REGISTRATION_AUTHORIZED": True,
        "physical_reference_semantics": PHYSICAL_REFERENCE_SEMANTICS,
    }
    if len(terminal) == 360:
        report = _complete_run_manifest(
            plan, attempts, lock_sha=lock_sha,
            lock_fingerprint=str(fingerprint["lock_fingerprint"]),
            plan_sha=plan_sha, code_commit=str(lock["execution_code_commit"]),
            authorization_sha=authorization_sha,
            environment_sha=environment_sha,
        )
        _atomic_create(runtime / "run_manifest.json", report)
        consume_authorization(
            authorization_path,
            lock_fingerprint=str(fingerprint["lock_fingerprint"]),
            first_backend_invocation_utc=str(in_use["started_at_utc"]),
            last_backend_invocation_utc=datetime.now(timezone.utc).isoformat(),
            actual_open3d_trials=180,
            actual_pcl_trials=180,
            execution_result_manifest_sha256=_sha256(runtime / "run_manifest.json"),
        )
    return report


def preflight_or_dry_run(
    repository: Path, *, lock_dir: Path, plan_path: Path | None = None,
    authorization_path: Path | None = None, runtime_root: Path, workers: int = 2,
    action: str = "preflight", mode: str = "fresh",
    remeasure_environment: bool = True, remeasure_execution_commit: bool = True,
    execution_adapter: ExecutionAdapter | None = None,
    result_validator: ResultValidator | None = None,
) -> dict[str, Any]:
    """Validate only, or enter the authorized future execution lifecycle."""

    if action not in {"preflight", "dry-run", "execute"}:
        _fail(f"unsupported action: {action}")
    if mode not in {"fresh", "resume"} or isinstance(workers, bool) or workers < 1:
        _fail("mode/workers are invalid")
    if action == "execute":
        if execution_adapter is not None or result_validator is not None:
            _fail("public formal execution forbids injected adapter/result validator")
        if remeasure_environment is not True:
            _fail("formal execution requires live environment remeasurement")
    root = Path(repository).resolve(strict=True)
    locked_dir = _resolve(root, lock_dir, "lock directory", directory=True)
    lock, fingerprint = _validate_lock(root, locked_dir,
                                       remeasure_environment=remeasure_environment)
    selected_plan = (root / lock["bindings"]["trial_plan_json"]["repository_relative_path"]
                     if plan_path is None else plan_path)
    resolved_plan = _resolve(root, selected_plan, "trial plan")
    plan, summary = _validate_plan(root, resolved_plan, lock)
    if action == "execute" or remeasure_execution_commit:
        _verify_execution_code_commit(root, str(lock["execution_code_commit"]))
    selected_runtime = _runtime_path(root, runtime_root)
    auth_path = authorization_path or (
        selected_runtime / "authorization" / AUTHORIZATION_FILENAME
    )
    present, valid, authorization, resolved_auth = _validate_authorization(
        root, auth_path, lock_dir=locked_dir, lock=lock,
        fingerprint=fingerprint, plan_path=resolved_plan,
        runtime_root=selected_runtime, mode=mode, workers=workers,
    )
    expected_runtime = root / str(lock.get("authoritative_runtime_root", ""))
    if selected_runtime != expected_runtime:
        _fail("runtime root differs from the authoritative locked location")
    if action == "execute":
        if not valid or authorization is None or resolved_auth is None:
            _fail("fresh/resume denied: separate formal authorization is absent or invalid")
        return _execute_authorized_run(
            root, lock_dir=locked_dir, lock=lock, fingerprint=fingerprint,
            plan_path=resolved_plan, plan=plan, authorization_path=resolved_auth,
            authorization=authorization, runtime_root=runtime_root, workers=workers,
            mode=mode, execution_adapter=execution_adapter,
            result_validator=result_validator,
        )
    # No backend loader or runtime writer is reachable for these two actions.
    return {
        "schema": "mid360_fmb1_zero_perturbation_runner_dry_run_v1_1_r1",
        "status": (
            "PASS_AUTHORIZED_NO_BACKEND_DRY_RUN"
            if valid else "PASS_LOCKED_AWAITING_SEPARATE_AUTHORIZATION"
        ),
        "action": action, "mode": mode, "runtime_root": str(runtime_root),
        "workers": workers, "LOCK_VALID": True, "FORMAL_PLAN_VALID": True,
        "FORMAL_RUN_AUTHORIZATION_PRESENT": present,
        "FORMAL_RUN_AUTHORIZATION_VALID": valid,
        "EXECUTION_BLOCKED": not valid,
        "EXECUTION_CODE_COMMIT_VERIFIED": remeasure_execution_commit,
        **summary, "FORMAL_ICP_UNLOCKED": valid,
        "FORMAL_REGISTRATION_AUTHORIZED": valid,
        "REAL_BACKEND_IMPORT_COUNT": 0, "REAL_BACKEND_CALL_COUNT": 0,
        "open3d_registration_call_count": 0, "pcl_cli_invocation_count": 0,
        "actual_formal_trials": 0, "real_trial_result_files_written": 0,
    }
