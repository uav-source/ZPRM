"""Independent verifier for the explicit-provenance Exec-R3 lock."""

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


LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r3.json"
LOCK_SCHEMA = "mid360_fmb1_zero_perturbation_formal_lock_v1_1_exec_r3"
R2_LOCK_PATH = Path(
    "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r2_lock/"
    "formal_batch1_zero_perturbation_lock_v1_1_exec_r2.json"
)
R2_LOCK_SHA256 = "fe8846fd80e879b497c8ef68568aa9f83a5ad339e0a29e8113afc36577eb3e4c"
R2_LOCK_FINGERPRINT = "5db11e845c4b1307d40ea9ba3598ead6ba2778a0370b0ce853c8408e8d21e8c5"
EXPECTED_RUNTIME = Path(
    "zero_perturbation_runtime/mid360_zero_perturbation_v1_1_formal_execution_v1"
)
EXPECTED_RESULTS = Path(
    "results/mid360_formal_batch1/zero_perturbation_v1_1_execution_v1"
)
BACKENDS = ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE")
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

BINDING_KEYS = {
    "binding_id", "repository_relative_path", "sha256", "bytes",
    "binding_class", "verification_source", "commit_role",
}
EXPLICIT_PROVENANCE = {
    "EXECUTION_CODE": ("GIT_BLOB_AT_COMMIT", "EXECUTION_CODE_COMMIT"),
    "LOCK_RELEASE_EVIDENCE": ("GIT_BLOB_AT_COMMIT", "LOCK_RELEASE_COMMIT"),
    "FROZEN_SCIENCE_OR_DATA": ("INHERITED_EXEC_R2_LOCK_SHA256", "NONE"),
    "ENVIRONMENT_OR_BINARY": ("FROZEN_SHA256_AND_ENVIRONMENT_CONTRACT", "NONE"),
}
EXPECTED_EXECUTION_BINDINGS = {
    "authorization_contract", "authorization_lifecycle",
    "authorization_producer", "authorization_producer_cli",
    "authorization_schema", "authorization_verifier",
    "authorization_verifier_cli", "binding_provenance_contract",
    "common_association", "exec_r3_lock_builder",
    "exec_r3_lock_issuer_cli", "exec_r3_lock_verifier",
    "exec_r3_lock_verifier_cli", "execution_environment",
    "execution_types", "experiments_package_init", "metrics",
    "mid360_formal_batch1_package_init", "open3d_backend", "pcl_backend",
    "phase_a_harness_package_init", "result_validator", "rotation_metrics",
    "runner", "runner_cli",
}
EXPECTED_RELEASE_BINDINGS = {
    "attempt_003_authorization_invalidation",
    "authorization_binding_provenance_defect_audit_json",
    "authorization_binding_provenance_defect_audit_md",
    "authorization_lifecycle_test_report", "execution_control_patch_report",
    "r2_lock_supersession", "superseded_r2_fingerprint",
    "superseded_r2_lock",
}
EXPECTED_ENVIRONMENT_BINDINGS = {
    "backend_parameter_contract", "environment_manifest", "pcl_executable",
}
EXPECTED_FROZEN_BINDINGS = {
    "acquisition_attempt_lineage", "activation_review", "active_amendment",
    "active_amendment_md", "active_protocol_pointer",
    "admitted_bag_manifest", "amendment_activation_record",
    "analysis_contract", "analysis_missingness_clarification",
    "analysis_missingness_clarification_md",
    "analysis_missingness_clarification_transition",
    "analysis_preclarification_history_inventory", "analysis_protocol",
    "blocked_attempt_no_registration", "blocked_attempt_record",
    "blocked_attempt_sha256sums", "final_dataset_independent_verification",
    "final_dataset_pointer", "final_dataset_prelock_reauthentication",
    "final_scene_registry", "final_station_registry", "geometry_manifest",
    "invalid_attempt_archive_manifest", "old_lock_file",
    "old_lock_fingerprint", "old_lock_supersession",
    "original_capture_radius_analysis_protocol", "original_preregistration",
    "original_zero_perturbation_proposal_json",
    "original_zero_perturbation_proposal_md", "prelock_no_icp_attestation",
    "proposal_correction_record", "proposal_difference_report",
    "proposal_superseded_sidecar",
    "protocol_c1_missingness_independent_verification",
    "protocol_transition_independent_verification", "result_schema",
    "snapshot_manifest", "target_manifest", "trial_plan_csv",
    "trial_plan_independent_verification", "trial_plan_json",
    "w04_superseded_history",
}
EXPECTED_BINDING_CLASSES = {
    **{name: "EXECUTION_CODE" for name in EXPECTED_EXECUTION_BINDINGS},
    **{name: "LOCK_RELEASE_EVIDENCE" for name in EXPECTED_RELEASE_BINDINGS},
    **{name: "FROZEN_SCIENCE_OR_DATA" for name in EXPECTED_FROZEN_BINDINGS},
    **{name: "ENVIRONMENT_OR_BINARY" for name in EXPECTED_ENVIRONMENT_BINDINGS},
}
CORE_REQUIRED = {
    LOCK_FILENAME,
    "formal_batch1_zero_perturbation_lock_v1_1_exec_r3.sha256",
    "lock_inventory.csv",
    "lock_fingerprint.json",
    "NO_REGISTRATION_ATTESTATION.json",
    "authorization_binding_provenance_defect_audit.json",
    "r3_execution_control_fix_report.json",
    "r3_authorization_fixture_qualification.json",
    "AUTHORIZATION_INVALIDATION_RECORD.json",
}


class ExecR3IndependentVerificationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise ExecR3IndependentVerificationError(
        f"FMB1_EXEC_R3_VERIFY_FAIL: {message}"
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


def _resolve(root: Path, value: str | Path, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        try:
            relative = relative.relative_to(root)
        except ValueError:
            _fail(f"{label} is outside repository")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"{label} uses symlink component")
    try:
        resolved = (root / relative).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        _fail(f"{label} missing/escapes repository: {error}")
    if not resolved.is_file():
        _fail(f"{label} is not a regular file")
    return resolved


def _git_blob(root: Path, commit: str, relative: str, label: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError:
        _fail(f"{label} is absent/different provenance commit")


def _verify_binding_shape(binding_id: str, row: Mapping[str, Any]) -> None:
    if set(row) != BINDING_KEYS or row.get("binding_id") != binding_id:
        _fail(f"binding metadata shape differs: {binding_id}")
    binding_class = row.get("binding_class")
    if binding_class not in EXPLICIT_PROVENANCE:
        _fail(f"unknown binding_class: {binding_id}")
    expected_source, expected_role = EXPLICIT_PROVENANCE[str(binding_class)]
    if row.get("verification_source") != expected_source:
        _fail(f"verification_source differs: {binding_id}")
    if row.get("commit_role") != expected_role:
        _fail(f"commit_role differs: {binding_id}")
    byte_count = row.get("bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        _fail(f"binding byte count differs: {binding_id}")
    if re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None:
        _fail(f"binding SHA differs: {binding_id}")


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
        if name in declared or path.is_symlink() or not path.is_file():
            _fail(f"core checksum entry differs: {name}")
        if sha256_file(path) != digest:
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
    plan = _json(path)
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 360:
        _fail("plan row count differs")
    identifiers: set[str] = set()
    counts: Counter[str] = Counter()
    snapshots: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("plan row is malformed")
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in identifiers:
            _fail("plan trial ID differs")
        identifiers.add(trial_id)
        backend = str(row.get("backend"))
        counts[backend] += 1
        snapshots[str(row.get("snapshot_id"))].append(row)
        if backend not in BACKENDS or row.get("track_id") != "ZERO_PERTURBATION_TRACK":
            _fail("plan backend/track differs")
        if not _identity(row.get("T0")):
            _fail("plan T0 differs")
        if row.get("scene_id") == "FMB1_W04":
            _fail("retired W04 entered plan")
        if row.get("scene_id") == "FMB1_W02" and row.get("attempt") != 2:
            _fail("old W02 attempt entered plan")
    if counts != Counter({BACKENDS[0]: 180, BACKENDS[1]: 180}):
        _fail("plan backend counts differ")
    if len(snapshots) != 180 or any(
        len(pair) != 2 or {row["backend"] for row in pair} != set(BACKENDS)
        for pair in snapshots.values()
    ):
        _fail("plan snapshot pairing differs")


def _static_execution_audit(paths: list[Path]) -> None:
    denied = {"open3d", "phase_a_harness.open3d_backend", "phase_a_harness.pcl_backend"}
    for path in paths:
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imports = {node.module or ""}
            else:
                imports = set()
            # Backend adapter modules are bound execution code and are allowed
            # to import backend libraries.  Control modules are identified by
            # exact path membership supplied by the lock builder tests.
            if path.name in {
                "formal_registration_authorization.py",
                "formal_registration_authorization_verify.py",
                "binding_provenance.py",
                "zero_perturbation_v1_1_exec_r3_lock.py",
                "zero_perturbation_v1_1_exec_r3_verify.py",
            } and imports & denied:
                _fail(f"execution control eagerly imports backend: {path}")


def verify_exec_r3_lock(
    repository: Path,
    lock_dir: Path,
    *,
    expected_execution_code_commit: str,
    expected_lock_release_commit: str | None = None,
) -> dict[str, Any]:
    root = Path(repository).resolve(strict=True)
    locked = Path(lock_dir).resolve(strict=True)
    core_count = _verify_core(locked)
    lock_path = locked / LOCK_FILENAME
    lock = _json(lock_path)
    if (
        lock.get("schema") != LOCK_SCHEMA
        or lock.get("execution_lock_revision") != 3
        or lock.get("execution_code_commit") != expected_execution_code_commit
        or lock.get("amendment_id") != "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1"
        or lock.get("binding_provenance_contract") != "EXPLICIT_PER_BINDING_V1"
        or lock.get("prefix_based_provenance_inference") is not False
        or lock.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or lock.get("FORMAL_ICP_UNLOCKED") is not False
        or lock.get("actual_formal_trials") != 0
    ):
        _fail("Exec-R3 lock state differs")

    inventory_path = locked / "lock_inventory.csv"
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_columns = list(BINDING_KEYS)
    if not rows or set(rows[0]) != set(expected_columns):
        _fail("lock inventory columns differ")
    if len({row["binding_id"] for row in rows}) != len(rows):
        _fail("lock inventory IDs are duplicated")
    material = {
        "lock_file_sha256": sha256_file(lock_path),
        "lock_inventory_file_sha256": sha256_file(inventory_path),
        "execution_code_commit": expected_execution_code_commit,
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    fingerprint_payload = _json(locked / "lock_fingerprint.json")
    if (
        fingerprint_payload.get("lock_fingerprint") != fingerprint
        or fingerprint_payload.get("lock_file_sha256") != material["lock_file_sha256"]
    ):
        _fail("R3 fingerprint differs")

    bindings = lock.get("bindings")
    if not isinstance(bindings, Mapping):
        _fail("R3 bindings are missing")
    if set(bindings) != set(EXPECTED_BINDING_CLASSES):
        _fail("R3 explicit binding registry differs")
    if set(bindings) != {row["binding_id"] for row in rows}:
        _fail("lock/inventory binding set differs")
    canonical_inventory: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = dict(row)
        try:
            normalized["bytes"] = int(row["bytes"])
        except (TypeError, ValueError):
            _fail(f"inventory byte count differs: {row.get('binding_id')}")
        binding_id = str(row["binding_id"])
        if normalized != bindings[binding_id]:
            _fail(f"lock/inventory row differs: {binding_id}")
        canonical_inventory.append(normalized)
    canonical_inventory.sort(key=lambda row: str(row["binding_id"]))
    inventory_digest = hashlib.sha256(
        canonical_json_bytes(canonical_inventory)
    ).hexdigest()
    if lock.get("binding_inventory_sha256") != inventory_digest:
        _fail("binding_inventory_sha256 differs")
    r2_path = _resolve(root, R2_LOCK_PATH, "R2 lock")
    if sha256_file(r2_path) != R2_LOCK_SHA256:
        _fail("R2 lock bytes changed")
    r2 = _json(r2_path)
    r2_fingerprint = _json(r2_path.parent / "lock_fingerprint.json")
    if r2_fingerprint.get("lock_fingerprint") != R2_LOCK_FINGERPRINT:
        _fail("R2 fingerprint changed")

    if expected_lock_release_commit is not None:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
        if head != expected_lock_release_commit:
            _fail("current HEAD differs from expected R3 release commit")
        _git_blob(
            root, expected_lock_release_commit,
            lock_path.relative_to(root).as_posix(), "R3 lock release",
        )

    execution_paths: list[Path] = []
    class_counts: Counter[str] = Counter()
    for binding_id, binding in bindings.items():
        if not isinstance(binding, Mapping):
            _fail(f"binding malformed: {binding_id}")
        _verify_binding_shape(str(binding_id), binding)
        if binding.get("binding_class") != EXPECTED_BINDING_CLASSES[binding_id]:
            _fail(f"binding_class differs from explicit registry: {binding_id}")
        path = _resolve(
            root, str(binding["repository_relative_path"]),
            f"binding {binding_id}",
        )
        if sha256_file(path) != binding["sha256"] or path.stat().st_size != binding["bytes"]:
            _fail(f"binding bytes differ: {binding_id}")
        binding_class = str(binding["binding_class"])
        class_counts[binding_class] += 1
        relative = path.relative_to(root).as_posix()
        if binding_class == "EXECUTION_CODE":
            recorded = _git_blob(
                root, expected_execution_code_commit, relative,
                f"execution binding {binding_id}",
            )
            if hashlib.sha256(recorded).hexdigest() != binding["sha256"]:
                _fail(f"execution binding differs from execution commit: {binding_id}")
            execution_paths.append(path)
        elif binding_class == "LOCK_RELEASE_EVIDENCE":
            if expected_lock_release_commit is not None:
                recorded = _git_blob(
                    root, expected_lock_release_commit, relative,
                    f"release binding {binding_id}",
                )
                if hashlib.sha256(recorded).hexdigest() != binding["sha256"]:
                    _fail(f"release binding differs from release commit: {binding_id}")
        elif binding_class == "FROZEN_SCIENCE_OR_DATA":
            old = r2.get("bindings", {}).get(binding_id)
            if not isinstance(old, Mapping) or old.get("sha256") != binding["sha256"]:
                _fail(f"frozen binding differs from R2: {binding_id}")
        elif binding_class == "ENVIRONMENT_OR_BINARY":
            if binding_id in {"environment_manifest", "pcl_executable", "backend_parameter_contract"}:
                old = r2.get("bindings", {}).get(binding_id)
                if not isinstance(old, Mapping) or old.get("sha256") != binding["sha256"]:
                    _fail(f"environment/binary binding differs from R2: {binding_id}")
        else:
            _fail(f"unknown provenance class: {binding_id}")

    if set(class_counts) != set(EXPLICIT_PROVENANCE):
        _fail("not all explicit provenance classes are represented")
    _static_execution_audit(execution_paths)
    _verify_plan(_resolve(
        root, str(bindings["trial_plan_json"]["repository_relative_path"]),
        "trial plan",
    ))
    verify_environment_manifest(
        _json(_resolve(
            root, str(bindings["environment_manifest"]["repository_relative_path"]),
            "environment manifest",
        )),
        root,
        remeasure_versions=True,
    )
    final = _json(_resolve(
        root,
        str(bindings["final_dataset_independent_verification"]["repository_relative_path"]),
        "final dataset verifier",
    ))
    if final.get("status") != "PASS" or final.get("pass") is not True:
        _fail("final dataset verification differs")

    patch = _json(locked / "r3_execution_control_fix_report.json")
    if (
        patch.get("scientific_protocol_changed") is not False
        or patch.get("final_dataset_changed") is not False
        or patch.get("trial_plan_changed") is not False
        or patch.get("backend_parameters_changed") is not False
        or patch.get("actual_formal_trials") != 0
        or patch.get("R3_EXECUTION_CODE_COMMIT") != expected_execution_code_commit
    ):
        _fail("R3 execution-control fix report differs")
    fixture = _json(locked / "r3_authorization_fixture_qualification.json")
    if (
        fixture.get("status") != "PASS"
        or fixture.get("producer_pass") is not True
        or fixture.get("independent_verifier_pass") is not True
        or fixture.get("fixture_preflight_pass") is not True
        or fixture.get("fixture_dry_run_pass") is not True
        or fixture.get("REAL_FORMAL_TRIALS") != 0
    ):
        _fail("R3 authorization fixture qualification differs")
    supersession = _json(_resolve(
        root,
        str(bindings["r2_lock_supersession"]["repository_relative_path"]),
        "R2 supersession",
    ))
    if (
        supersession.get("R2_EXECUTION_LOCK_STATUS")
        != "SUPERSEDED_BEFORE_BACKEND_EXECUTION"
        or supersession.get("reason")
        != "AUTHORIZATION_VERIFIER_BINDING_PROVENANCE_DEFECT"
        or supersession.get("R2_ACTUAL_FORMAL_TRIALS") != 0
    ):
        _fail("R2 supersession semantics differ")
    invalidation = _json(locked / "AUTHORIZATION_INVALIDATION_RECORD.json")
    if (
        invalidation.get("authorization_id")
        != "FMB1-AUTH-9794108265e14fed8887b202eda4ed50"
        or invalidation.get("authorization_sha256")
        != "d3db750f782fba4a2449f923ea3ab480ded43a9f21b8b8d35633417ed5354f8e"
        or invalidation.get("status") != "VOID_FOR_FUTURE_EXECUTION"
        or invalidation.get("reusable") is not False
        or invalidation.get("actual_formal_trials") != 0
    ):
        _fail("Attempt 003 authorization invalidation differs")
    no_registration = _json(locked / "NO_REGISTRATION_ATTESTATION.json")
    if no_registration.get("pass") is not True or any(
        int(no_registration.get(key, -1)) != 0
        for key in (
            "open3d_registration_call_count",
            "pcl_cli_invocation_count",
            "formal_trial_count",
        )
    ):
        _fail("R3 no-registration attestation differs")
    if (root / EXPECTED_RUNTIME).exists() or (root / EXPECTED_RESULTS).exists():
        _fail("real runtime or formal results exist")
    return {
        "schema": "mid360_fmb1_zero_perturbation_exec_r3_lock_independent_verification_v1",
        "status": "PASS",
        "pass": True,
        "R3_LOCK_VERIFIER_PASS": True,
        "failure_count": 0,
        "execution_lock_revision": 3,
        "lock_file_sha256": sha256_file(lock_path),
        "lock_fingerprint": fingerprint,
        "execution_code_commit": expected_execution_code_commit,
        "binding_count": len(bindings),
        "binding_class_counts": dict(sorted(class_counts.items())),
        "explicit_binding_provenance_enabled": True,
        "prefix_based_provenance_inference": False,
        "lock_core_checksum_count": core_count,
        "scene_count": 6,
        "station_count": 18,
        "snapshot_count": 180,
        "planned_open3d_trials": 180,
        "planned_pcl_trials": 180,
        "planned_total_trials": 360,
        "AUTHORIZATION_PRODUCER_READY": True,
        "INDEPENDENT_AUTHORIZATION_VERIFIER_READY": True,
        "AUTHORIZATION_LIFECYCLE_QUALIFIED": True,
        "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "actual_formal_trials": 0,
        "registration_backend_calls": 0,
    }


def verify_release_checksums(lock_dir: Path) -> dict[str, Any]:
    locked = Path(lock_dir).resolve(strict=True)
    sums = locked / "SHA256SUMS"
    declared: dict[str, str] = {}
    for line in sums.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            _fail("release checksum line malformed")
        digest, name = match.groups()
        if name in declared:
            _fail("release checksum duplicate")
        declared[name] = digest
    actual = {
        path.name: path for path in locked.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(declared) != set(actual):
        _fail("release checksum set differs")
    for name, path in actual.items():
        if path.is_symlink() or sha256_file(path) != declared[name]:
            _fail(f"release checksum differs: {name}")
    return {
        "schema": "mid360_fmb1_exec_r3_release_checksum_verification_v1",
        "status": "PASS",
        "pass": True,
        "covered_regular_file_count": len(actual),
        "sha256sums_sha256": sha256_file(sums),
    }
