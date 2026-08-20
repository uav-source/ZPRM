"""Independent post-run verifier for frozen FMB1 Exec-R3 result bytes.

The verifier reimplements schema, transform, rotation and association checks.
It neither imports nor invokes the formal runner, result validator, backend,
or the production association/rotation helpers.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .independent_association_math_v1 import analyze_final, prepare_initial, prepare_target
from .independent_transform_math_v1 import (
    absolute_relative_error,
    close,
    independent_transform_audit,
    require_identity,
)


RAW_EXECUTION_COMMIT = "059e39533991d929a97ab208ad738643af82d09a"
RAW_EXECUTION_TAG_OBJECT = "26e245fa3a0e3f378d3ccd074042d18f1011efe7"
R3_LOCK_SHA256 = "f7349142d8a5b3dd0277f64672297017366725c1d47de870580bf5bec4c4dd30"
R3_FINGERPRINT = "fd601e8daf62c488a3a079beea05f65c9bd283399ae4d7f3acf16e63fc473a6d"
R3_EXECUTION_CODE_COMMIT = "c2702b266c427e108209d476bb5e07c3292735c5"
LOCK_FILENAME = "formal_batch1_zero_perturbation_lock_v1_1_exec_r3.json"
CONTRACT_FILENAME = "postrun_verifier_contract_v1.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ATTEMPT_RE = re.compile(r"^attempt-(\d{4})\.json$")
START_RE = re.compile(r"^attempt-(\d{4})\.started\.json$")
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
PAIR_BINDINGS = (
    "snapshot_id", "scene_id", "station_id", "attempt",
    "source_reference", "source_sha256", "source_array_sha256", "source_point_count",
    "target_reference", "target_sha256", "target_array_sha256", "target_point_count",
    "T0", "T_reference_nominal",
)
RESULT_PLAN_BINDINGS = {
    "trial_id": "trial_id", "batch_id": "batch_id", "amendment_id": "amendment_id",
    "track_id": "track_id", "scene_id": "scene_id", "geometry_class": "final_geometry_class",
    "station_id": "station_id", "attempt": "attempt", "snapshot_id": "snapshot_id",
    "backend": "backend", "backend_version": "backend_version",
    "source_reference": "source_reference", "source_sha256": "source_sha256",
    "target_reference": "target_reference", "target_sha256": "target_sha256",
    "T0": "T0", "T_reference_nominal": "T_reference_nominal",
    "backend_parameter_contract_sha256": "backend_parameter_contract_sha256",
    "backend_canonical_parameter_sha256": "backend_canonical_parameter_sha256",
    "active_amendment_sha256": "active_amendment_sha256",
    "analysis_contract_sha256": "analysis_contract_sha256",
}
COUNT_FIELDS = (
    "initial_correspondence_count", "initial_valid_normal_correspondence_count",
    "final_correspondence_count", "final_valid_normal_correspondence_count",
)
TURNOVER_FIELDS = (
    "correspondence_turnover", "accepted_source_turnover", "correspondence_count_change_ratio",
)
RESIDUAL_FIELDS = ("initial_residual_rmse", "final_residual_rmse", "residual_rmse_change")
NORMAL_ANGLE_FIELDS = ("median_normal_angle_change_deg", "q95_normal_angle_change_deg")
ASSOCIATION_FIELDS = COUNT_FIELDS + TURNOVER_FIELDS + RESIDUAL_FIELDS + NORMAL_ANGLE_FIELDS
COMMON_STATUS_FIELDS = (
    "common_association_valid", "common_association_invalid_reason",
    "common_association_invalid_detail",
)
TRIAL_MANIFEST_FIELDS = (
    "trial_id", "backend", "result_file_sha256", "attempt_count", "schema_pass",
    "identity_binding_pass", "transform_arithmetic_pass", "rotation_recomputation_pass",
    "association_recomputation_pass", "missingness_pass", "overall_pass",
    "failure_field_names",
)


class IndependentPostrunVerificationError(RuntimeError):
    """Fail-closed error from the independent verifier."""


def _fail(message: str) -> None:
    raise IndependentPostrunVerificationError(f"FMB1_POSTRUN_VERIFIER_BLOCKED: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"cannot parse JSON {path}: {error}")
    if type(value) is not dict:
        _fail(f"JSON root is not an object: {path}")
    return value


def _within(root: Path, candidate: Path, *, must_exist: bool = True) -> Path:
    lexical = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        _fail(f"path escapes root: {candidate}")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"symlink path component is forbidden: {cursor}")
    if must_exist:
        try:
            resolved = lexical.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            _fail(f"path missing or escapes root: {candidate}: {error}")
        return resolved
    return lexical


def load_contract() -> dict[str, Any]:
    path = Path(__file__).with_name(CONTRACT_FILENAME)
    contract = _load_json(path)
    if (
        contract.get("schema") != "mid360_fmb1_exec_r3_postrun_verifier_contract_v1"
        or contract.get("status") != "FROZEN_BEFORE_REAL_RESULT_VALUE_READ"
        or contract.get("real_result_values_read_before_contract_freeze") is not False
        or contract.get("scientific_aggregation_permitted") is not False
    ):
        _fail("verifier contract is not frozen before real-result read")
    expected_tolerances = {
        "matrix_atol": 1.0e-12, "matrix_rtol": 1.0e-10,
        "scalar_atol": 1.0e-12, "scalar_rtol": 1.0e-10,
        "angle_rad_atol": 1.0e-12, "angle_rad_rtol": 1.0e-10,
        "angle_deg_atol": 1.0e-9, "angle_deg_rtol": 1.0e-10,
    }
    expected_association = {
        "association_distance_limit_m": 0.5, "target_normal_pca_k": 50,
        "target_normal_pca_min_neighbors": 10, "target_normal_pca_chunk_size": 2048,
        "normal_norm_epsilon": 1.0e-12, "quantile_method": "linear",
        "nearest_neighbor_k": 1, "ckdtree_workers": 1,
    }
    if contract.get("tolerances") != expected_tolerances or contract.get("association") != expected_association:
        _fail("frozen tolerance/association contract differs")
    return contract


def parse_checksum_manifest(execution_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    root = execution_root.resolve(strict=True)
    sums = root / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        _fail("SHA256SUMS is missing or symlinked")
    declared: dict[str, str] = {}
    failures: list[str] = []
    unsafe: list[str] = []
    duplicates: list[str] = []
    for number, line in enumerate(sums.read_text(encoding="ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            _fail(f"malformed SHA256SUMS line {number}")
        digest, name = match.groups()
        path = Path(name)
        if name in declared:
            duplicates.append(name); continue
        if path.is_absolute() or ".." in path.parts or "\\" in name or name == "SHA256SUMS":
            unsafe.append(name); continue
        target = root / path
        try:
            target.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            unsafe.append(name); continue
        if target.is_symlink() or not target.is_file():
            unsafe.append(name); continue
        observed = sha256_file(target)
        if observed != digest:
            failures.append(name)
        declared[name] = digest
    if len(declared) != 743 or duplicates or unsafe or failures:
        _fail(
            f"checksum audit failed entries={len(declared)} duplicates={duplicates} "
            f"unsafe={unsafe} failures={failures}"
        )
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    unknown = actual - set(declared) - {"SHA256SUMS"}
    missing = set(declared) - actual
    symlinks = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()]
    if unknown or missing or symlinks:
        _fail(f"execution-root coverage differs unknown={unknown} missing={missing} symlinks={symlinks}")
    freeze = _load_json(root / "formal_execution_raw_freeze_manifest.json")
    inventory = freeze.get("frozen_primary_artifacts")
    if not isinstance(inventory, list) or len(inventory) != 740:
        _fail("raw-freeze primary inventory count differs")
    primary: set[str] = set()
    for row in inventory:
        if type(row) is not dict:
            _fail("raw-freeze inventory row is malformed")
        name = row.get("results_relative_path")
        if not isinstance(name, str) or name in primary:
            _fail("raw-freeze inventory path is invalid/duplicate")
        path = root / name
        if (
            name not in declared or path.is_symlink() or not path.is_file()
            or sha256_file(path) != row.get("sha256")
            or path.stat().st_size != row.get("bytes")
        ):
            _fail(f"raw-freeze inventory binding differs: {name}")
        primary.add(name)
    non_primary = set(declared) - primary
    expected_non_primary = {
        "formal_execution_raw_freeze_manifest.json",
        "formal_execution_summary.json",
        "formal_execution_summary.md",
    }
    if non_primary != expected_non_primary:
        _fail(f"raw-freeze non-primary coverage differs: {non_primary}")
    return declared, {
        "schema": "mid360_fmb1_postrun_checksum_audit_v1", "status": "PASS", "pass": True,
        "checksum_entry_count": 743, "checksum_failure_count": 0,
        "duplicate_path_count": 0, "unsafe_path_count": 0, "symlink_count": 0,
        "unknown_file_count": 0, "missing_declared_file_count": 0,
        "raw_freeze_primary_inventory_count": 740, "raw_freeze_inventory_match": True,
    }


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if check and result.returncode != 0:
        _fail(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_provenance_audit(
    repository: Path, execution_root: Path, output_dir: Path, *,
    raw_execution_commit: str, raw_execution_tag: str, verifier_code_commit: str,
) -> dict[str, Any]:
    root = repository.resolve(strict=True)
    if _git(root, "status", "--short"):
        _fail("worktree is not clean before real verification")
    head = _git(root, "rev-parse", "HEAD")
    if head != verifier_code_commit or COMMIT_RE.fullmatch(verifier_code_commit) is None:
        _fail("HEAD does not equal verifier code commit")
    if raw_execution_commit != RAW_EXECUTION_COMMIT:
        _fail("raw execution commit argument differs")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", raw_execution_commit, head], cwd=root
    ).returncode != 0:
        _fail("raw execution commit is not an ancestor of verifier commit")
    tag_ref = f"refs/tags/{raw_execution_tag}"
    tag_object = _git(root, "rev-parse", tag_ref)
    peeled = _git(root, "rev-parse", f"{tag_ref}^{{}}")
    if tag_object != RAW_EXECUTION_TAG_OBJECT or peeled != raw_execution_commit:
        _fail("raw execution annotated tag identity differs")
    if _git(root, "cat-file", "-t", tag_ref) != "tag":
        _fail("raw execution tag is not annotated")
    relative_execution = str(execution_root.resolve(strict=True).relative_to(root))
    diff = subprocess.run(
        ["git", "diff", "--exit-code", raw_execution_commit, "--", relative_execution],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if diff.returncode != 0:
        _fail("formal execution root differs from raw execution commit")
    output = _within(root, output_dir, must_exist=False)
    try:
        output.relative_to(execution_root.resolve(strict=True))
    except ValueError:
        pass
    else:
        _fail("output directory is inside formal execution root")
    if output.exists():
        _fail("real verification output directory already exists")
    return {
        "schema": "mid360_fmb1_postrun_git_provenance_audit_v1", "status": "PASS", "pass": True,
        "raw_execution_commit": raw_execution_commit, "raw_execution_tag": raw_execution_tag,
        "raw_execution_tag_object": tag_object, "raw_execution_tag_peeled_commit": peeled,
        "verifier_code_commit": verifier_code_commit, "head_equals_verifier_code_commit": True,
        "raw_execution_is_ancestor": True, "raw_execution_root_diff_count": 0,
        "worktree_clean_before_real_read": True,
    }


def _schema_error(path: str, message: str) -> None:
    raise IndependentPostrunVerificationError(f"SCHEMA_ERROR {path}: {message}")


def _json_type(value: Any, expected: str) -> bool:
    if expected == "null": return value is None
    if expected == "boolean": return type(value) is bool
    if expected == "integer": return type(value) is int
    if expected == "number": return type(value) in {int, float} and math.isfinite(float(value))
    if expected == "string": return isinstance(value, str)
    if expected == "array": return isinstance(value, list)
    if expected == "object": return type(value) is dict
    return False


def _schema_matches(value: Any, schema: Mapping[str, Any], path: str) -> bool:
    try:
        validate_schema_value(value, schema, path)
    except IndependentPostrunVerificationError:
        return False
    return True


def validate_schema_value(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    if "oneOf" in schema:
        matches = sum(_schema_matches(value, option, path) for option in schema["oneOf"])
        if matches != 1: _schema_error(path, f"oneOf matched {matches} alternatives")
    if "type" in schema and not _json_type(value, str(schema["type"])):
        _schema_error(path, f"expected type {schema['type']}")
    if "const" in schema and value != schema["const"]: _schema_error(path, "const differs")
    if "enum" in schema and value not in schema["enum"]: _schema_error(path, "enum differs")
    if isinstance(value, str):
        if "pattern" in schema and re.fullmatch(str(schema["pattern"]), value) is None:
            _schema_error(path, "pattern differs")
        if len(value) < int(schema.get("minLength", 0)): _schema_error(path, "string too short")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]): _schema_error(path, "string too long")
    if type(value) in {int, float} and not isinstance(value, bool):
        if "minimum" in schema and float(value) < float(schema["minimum"]): _schema_error(path, "below minimum")
        if "maximum" in schema and float(value) > float(schema["maximum"]): _schema_error(path, "above maximum")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)): _schema_error(path, "array too short")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]): _schema_error(path, "array too long")
        if "items" in schema:
            for index, child in enumerate(value): validate_schema_value(child, schema["items"], f"{path}[{index}]")
    if type(value) is dict:
        required = set(schema.get("required", []))
        if not required.issubset(value): _schema_error(path, f"missing keys {sorted(required-set(value))}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            _schema_error(path, f"unknown keys {sorted(set(value)-set(properties))}")
        for key, child_schema in properties.items():
            if key in value: validate_schema_value(value[key], child_schema, f"{path}.{key}")
    for clause in schema.get("allOf", []):
        condition = clause.get("if")
        selected = clause.get("then") if condition is not None and _schema_matches(value, condition, path) else clause.get("else")
        if selected is not None: validate_schema_value(value, selected, path)


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str): _fail(f"{label} is not a timestamp string")
    try: parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError: _fail(f"{label} timestamp is malformed")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail(f"{label} is not timezone-aware UTC")
    return parsed


def _companion(path: Path, expected_name: str) -> str:
    fields = path.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != expected_name or SHA256_RE.fullmatch(fields[0]) is None:
        _fail(f"malformed SHA companion: {path}")
    return fields[0]


def validate_start_marker_payload(
    marker: Mapping[str, Any], *, trial_id: str, authorization_sha256: str
) -> datetime:
    if set(marker) != {"schema", "trial_id", "attempt_number", "authorization_sha256", "started_at_utc"}:
        _fail(f"start marker schema differs: {trial_id}")
    if (
        marker.get("schema") != "mid360_fmb1_formal_trial_attempt_start_v1_1_r1"
        or marker.get("trial_id") != trial_id or marker.get("attempt_number") != 1
        or marker.get("authorization_sha256") != authorization_sha256
    ): _fail(f"start marker identity/authorization differs: {trial_id}")
    return _parse_utc(marker.get("started_at_utc"), f"{trial_id} start")


def authorization_lifecycle_audit(repository: Path, execution_root: Path, run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    auth_dir = execution_root / "authorization"
    expected_names = {
        "formal_registration_authorization.json", "formal_registration_authorization.sha256",
        "authorization_verification_report.json", "authorization_in_use.json",
        "authorization_consumption_receipt.json", "authorization_consumption_receipt.sha256",
    }
    if {p.name for p in auth_dir.iterdir()} != expected_names or any(p.is_symlink() for p in auth_dir.iterdir()):
        _fail("authorization frozen inventory differs")
    auth_path = auth_dir / "formal_registration_authorization.json"
    auth = _load_json(auth_path); auth_sha = sha256_file(auth_path)
    if _companion(auth_dir / "formal_registration_authorization.sha256", auth_path.name) != auth_sha:
        _fail("authorization SHA companion differs")
    if auth.get("immutable") is not True or auth.get("status") != "ISSUED":
        _fail("authorization issue state/immutability differs")
    if (
        auth.get("lock_fingerprint") != R3_FINGERPRINT
        or auth.get("lock_file_sha256") != R3_LOCK_SHA256
        or auth.get("execution_code_commit") != R3_EXECUTION_CODE_COMMIT
        or auth.get("planned_open3d_count") != 180
        or auth.get("planned_pcl_count") != 180
        or auth.get("planned_trial_count") != 360
    ): _fail("authorization lock/count bindings differ")
    verification = _load_json(auth_dir / "authorization_verification_report.json")
    if verification.get("AUTHORIZATION_VERIFICATION_PASS") is not True or verification.get("authorization_sha256") != auth_sha:
        _fail("frozen authorization verification report differs")
    in_use = _load_json(auth_dir / "authorization_in_use.json")
    if (
        in_use.get("authorization_id") != auth.get("authorization_id")
        or in_use.get("authorization_sha256") != auth_sha
        or in_use.get("lock_fingerprint") != R3_FINGERPRINT
        or in_use.get("state") != "IN_USE"
        or in_use.get("reusable_for_new_fresh") is not False
    ): _fail("IN_USE lifecycle marker differs")
    receipt_path = auth_dir / "authorization_consumption_receipt.json"
    receipt = _load_json(receipt_path); receipt_sha = sha256_file(receipt_path)
    if _companion(auth_dir / "authorization_consumption_receipt.sha256", receipt_path.name) != receipt_sha:
        _fail("consumption receipt SHA companion differs")
    manifest_sha = sha256_file(execution_root / "execution/run_manifest.json")
    if (
        receipt.get("authorization_id") != auth.get("authorization_id")
        or receipt.get("authorization_sha256") != auth_sha
        or receipt.get("lock_fingerprint") != R3_FINGERPRINT
        or receipt.get("state") != "CONSUMED" or receipt.get("consumed") is not True
        or receipt.get("reusable") is not False
        or receipt.get("actual_open3d_trials") != 180
        or receipt.get("actual_pcl_trials") != 180
        or receipt.get("actual_total_trials") != 360
        or receipt.get("FURTHER_REGISTRATION_AUTHORIZED") is not False
        or receipt.get("execution_result_manifest_sha256") != manifest_sha
    ): _fail("authorization consumption receipt differs")
    runtime = repository / str(auth.get("authoritative_runtime_root"))
    live = runtime / "authorization/formal_registration_authorization.json"
    if not live.is_file() or live.is_symlink() or sha256_file(live) != auth_sha:
        _fail("authoritative runtime authorization identity differs")
    live_candidates = list((repository / "zero_perturbation_runtime").rglob("formal_registration_authorization.json"))
    if len(live_candidates) != 1 or live_candidates[0].resolve() != live.resolve():
        _fail("a second live authorization exists")
    return {
        "schema": "mid360_fmb1_postrun_authorization_lifecycle_audit_v1", "status": "PASS", "pass": True,
        "authorization_id": auth["authorization_id"], "authorization_sha256": auth_sha,
        "authorization_immutable": True, "authorization_issue_status": "ISSUED",
        "in_use_marker_verified": True, "consumption_receipt_verified": True,
        "consumption_receipt_sha256": receipt_sha, "run_manifest_sha256": manifest_sha,
        "receipt_binds_run_manifest": True, "actual_open3d_trials": 180,
        "actual_pcl_trials": 180, "actual_total_trials": 360,
        "authorization_consumed": True, "authorization_reusable": False,
        "further_registration_authorized": False, "second_live_authorization_count": 0,
    }


def load_canonical_array(repository: Path, row: Mapping[str, Any], prefix: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = _within(repository, Path(str(row[f"{prefix}_reference"])))
    if path.suffix != ".npy" or path.is_symlink() or not path.is_file():
        _fail(f"{prefix} canonical array path differs")
    file_sha = sha256_file(path)
    if file_sha != row[f"{prefix}_sha256"]: _fail(f"{prefix} file SHA differs: {path}")
    try: value = np.load(path, allow_pickle=False)
    except Exception as error: _fail(f"cannot load {prefix} array {path}: {error}")
    count = row[f"{prefix}_point_count"]
    if (
        value.dtype.str != "<f8" or value.ndim != 2 or value.shape != (count, 3)
        or not value.flags.c_contiguous or not np.all(np.isfinite(value))
    ): _fail(f"{prefix} canonical array contract differs: {path}")
    array_sha = hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes(order="C")).hexdigest()
    if array_sha != row[f"{prefix}_array_sha256"]: _fail(f"{prefix} array byte SHA differs: {path}")
    return value, {
        "reference": str(row[f"{prefix}_reference"]), "file_sha256": file_sha,
        "array_sha256": array_sha, "point_count": int(count),
        "dtype": value.dtype.str, "shape": list(value.shape), "c_contiguous": True, "finite": True,
    }


def _load_lock_plan_schema(repository: Path, lock_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    lock_path = lock_dir / LOCK_FILENAME
    if sha256_file(lock_path) != R3_LOCK_SHA256: _fail("R3 lock SHA differs")
    lock = _load_json(lock_path)
    fingerprint = _load_json(lock_dir / "lock_fingerprint.json")
    if fingerprint.get("lock_fingerprint") != R3_FINGERPRINT: _fail("R3 fingerprint differs")
    bindings = lock.get("bindings")
    if type(bindings) is not dict: _fail("R3 lock bindings missing")
    plan_path = _within(repository, Path(bindings["trial_plan_json"]["repository_relative_path"]))
    schema_path = _within(repository, Path(bindings["result_schema"]["repository_relative_path"]))
    for binding, path in ((bindings["trial_plan_json"], plan_path), (bindings["result_schema"], schema_path)):
        if sha256_file(path) != binding["sha256"] or path.stat().st_size != binding["bytes"]:
            _fail("R3 lock-bound plan/schema bytes differ")
    plan = _load_json(plan_path); schema = _load_json(schema_path)
    if len(plan.get("rows", [])) != 360: _fail("trial plan does not contain 360 rows")
    return lock, plan, schema, plan_path, schema_path


def validate_inventory_identity_sets(
    expected_ids: Iterable[str], result_ids: Iterable[str], marker_ids: Iterable[str],
    attempts_by_trial: Mapping[str, list[int]], *, expected_count: int,
) -> dict[str, int]:
    expected_list = list(expected_ids)
    result_list = list(result_ids)
    marker_list = list(marker_ids)
    expected = set(expected_list); results = set(result_list); markers = set(marker_list)
    duplicate_plan = len(expected_list) - len(expected)
    duplicate_result = len(result_list) - len(results)
    missing = expected - results
    unauthorized = results - expected
    orphan_markers = markers - expected
    missing_markers = expected - markers
    second_attempt = sum(
        number >= 2 for numbers in attempts_by_trial.values() for number in numbers
    )
    bad_attempt_sequence = {
        trial_id: numbers for trial_id, numbers in attempts_by_trial.items()
        if numbers != [1]
    }
    if (
        len(expected) != expected_count or duplicate_plan or duplicate_result or missing
        or unauthorized or orphan_markers or missing_markers or second_attempt
        or bad_attempt_sequence or set(attempts_by_trial) != results
    ):
        _fail(
            "inventory identity sets differ: "
            f"expected={len(expected)} duplicate_plan={duplicate_plan} "
            f"duplicate_result={duplicate_result} missing={sorted(missing)} "
            f"unauthorized={sorted(unauthorized)} orphan_markers={sorted(orphan_markers)} "
            f"missing_markers={sorted(missing_markers)} second_attempt={second_attempt} "
            f"bad_attempt_sequence={bad_attempt_sequence}"
        )
    return {
        "expected_count": len(expected), "result_count": len(results),
        "marker_count": len(markers), "missing_count": 0, "duplicate_count": 0,
        "unauthorized_count": 0, "orphan_count": 0, "second_attempt_count": 0,
    }


def validate_backend_pair_inputs(rows: Iterable[Mapping[str, Any]], *, snapshot_count: int) -> None:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: grouped[str(row.get("snapshot_id"))].append(row)
    if len(grouped) != snapshot_count: _fail("snapshot pair inventory count differs")
    for snapshot, pair in grouped.items():
        if len(pair) != 2 or {row.get("backend") for row in pair} != {
            "OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE"
        }:
            _fail(f"snapshot backend pair differs: {snapshot}")
        if any(pair[0].get(field) != pair[1].get(field) for field in PAIR_BINDINGS):
            _fail(f"snapshot backend inputs differ: {snapshot}")


def validate_plan_contract(plan: Mapping[str, Any], *, expected_count: int = 360) -> dict[str, int]:
    rows = plan.get("rows")
    if (
        plan.get("schema") != "mid360_fmb1_zero_perturbation_trial_plan_v1_1_r1"
        or plan.get("track_id") != "ZERO_PERTURBATION_TRACK"
        or not isinstance(rows, list) or len(rows) != expected_count
        or any(type(row) is not dict for row in rows)
    ): _fail("trial plan top-level contract differs")
    expected_scenes = {"FMB1_R01", "FMB1_R02", "FMB1_R03", "FMB1_W01", "FMB1_W02", "FMB1_W03"}
    seen_ids: set[str] = set(); snapshots: set[str] = set(); stations: set[tuple[str, str]] = set()
    backends: Counter[str] = Counter()
    for row in rows:
        trial_id = row.get("trial_id"); scene = row.get("scene_id"); station = row.get("station_id")
        if not isinstance(trial_id, str) or trial_id in seen_ids: _fail("trial plan ID is missing/duplicate")
        if scene not in expected_scenes or station not in {"S01", "S02", "S03"}: _fail("trial plan scene/station differs")
        if row.get("attempt") != (2 if scene == "FMB1_W02" else 1): _fail("trial plan acquisition attempt differs")
        if row.get("track_id") != "ZERO_PERTURBATION_TRACK": _fail("capture-radius/nonformal track entered plan")
        backend = row.get("backend")
        expected_version = {"OPEN3D_POINT_TO_PLANE": "0.19.0+b012259", "PCL_POINT_TO_PLANE": "1.15.1"}.get(backend)
        if expected_version is None or row.get("backend_version") != expected_version: _fail("plan backend/version differs")
        require_identity(row.get("T0")); require_identity(row.get("T_reference_nominal"), "T_reference_nominal")
        if row.get("translation_perturbation_m") != 0.0 or row.get("rotation_perturbation_deg") != 0.0:
            _fail("non-zero/capture-radius perturbation entered plan")
        for name in (
            "source_sha256", "source_array_sha256", "target_sha256", "target_array_sha256",
            "backend_parameter_contract_sha256", "backend_canonical_parameter_sha256",
            "active_amendment_sha256", "analysis_contract_sha256",
        ):
            if not isinstance(row.get(name), str) or SHA256_RE.fullmatch(row[name]) is None:
                _fail(f"plan SHA field differs: {name}")
        for name in ("source_point_count", "target_point_count"):
            if type(row.get(name)) is not int or row[name] <= 0: _fail(f"plan point count differs: {name}")
        seen_ids.add(trial_id); snapshots.add(str(row.get("snapshot_id")))
        stations.add((str(scene), str(station))); backends[str(backend)] += 1
    if expected_count == 360:
        if (
            {row["scene_id"] for row in rows} != expected_scenes or len(stations) != 18
            or len(snapshots) != 180
            or backends != Counter({"OPEN3D_POINT_TO_PLANE": 180, "PCL_POINT_TO_PLANE": 180})
        ): _fail("trial plan hierarchy/backend counts differ")
        validate_backend_pair_inputs(rows, snapshot_count=180)
    return {
        "trial_count": len(rows), "snapshot_count": len(snapshots),
        "station_count": len(stations), "open3d_count": backends["OPEN3D_POINT_TO_PLANE"],
        "pcl_count": backends["PCL_POINT_TO_PLANE"],
    }


def _validate_result_identity(
    result: Mapping[str, Any], plan_row: Mapping[str, Any], schema: Mapping[str, Any], *,
    trial_plan_sha: str, lock_sha: str, environment_sha: str,
) -> None:
    validate_schema_value(result, schema)
    if set(result) != set(schema["required"]): _fail("result keys differ from frozen schema required set")
    for result_name, plan_name in RESULT_PLAN_BINDINGS.items():
        if result.get(result_name) != plan_row.get(plan_name):
            _fail(f"result {result_name} differs from frozen trial plan")
    if result.get("trial_plan_sha256") != trial_plan_sha: _fail("result trial-plan SHA differs")
    if result.get("formal_lock_sha256") != lock_sha: _fail("result formal-lock SHA differs")
    if result.get("code_commit") != R3_EXECUTION_CODE_COMMIT: _fail("result execution commit differs")
    environment = result.get("environment_identity")
    if type(environment) is not dict or environment.get("environment_manifest_sha256") != environment_sha:
        _fail("result environment identity differs")
    if result.get("track_id") != "ZERO_PERTURBATION_TRACK" or result.get("execution_kind") != "FORMAL":
        _fail("result track/execution kind differs")
    if result.get("fixture_only") is not False: _fail("fixture result entered formal result set")
    if result.get("physical_reference_semantics") != "NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT":
        _fail("physical reference semantics differ")
    require_identity(result.get("T0")); require_identity(result.get("T_reference_nominal"), "T_reference_nominal")


def discover_and_validate_inventory(
    repository: Path, execution_root: Path, plan: Mapping[str, Any], schema: Mapping[str, Any],
    checksums: Mapping[str, str], authorization_sha: str, *,
    trial_plan_sha: str, lock_sha: str, environment_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = plan.get("rows")
    if not isinstance(rows, list) or any(type(row) is not dict for row in rows): _fail("plan rows malformed")
    by_id = {str(row.get("trial_id")): row for row in rows}
    if len(by_id) != 360 or "None" in by_id: _fail("plan trial IDs are not 360 unique IDs")
    result_root = execution_root / "raw_runtime_snapshot/trial_results"
    marker_root = execution_root / "raw_runtime_snapshot/inflight"
    result_ids = {path.name for path in result_root.iterdir() if path.is_dir() and not path.is_symlink()}
    marker_ids = {path.name for path in marker_root.iterdir() if path.is_dir() and not path.is_symlink()}
    attempts_by_trial: dict[str, list[int]] = {}
    for trial_id in result_ids:
        attempts_by_trial[trial_id] = sorted(
            int(match.group(1)) for path in (result_root / trial_id).iterdir()
            if (match := ATTEMPT_RE.fullmatch(path.name)) is not None
        )
    validate_inventory_identity_sets(
        [str(row.get("trial_id")) for row in rows], result_ids, marker_ids,
        attempts_by_trial, expected_count=360,
    )
    records: list[dict[str, Any]] = []
    backend_counts: Counter[str] = Counter(); snapshot_pairs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial_id in sorted(by_id):
        result_files = list((result_root / trial_id).iterdir())
        marker_files = list((marker_root / trial_id).iterdir())
        if len(result_files) != 1 or ATTEMPT_RE.fullmatch(result_files[0].name) is None or result_files[0].name != "attempt-0001.json":
            _fail(f"trial does not have exactly attempt-0001 result: {trial_id}")
        if len(marker_files) != 1 or START_RE.fullmatch(marker_files[0].name) is None or marker_files[0].name != "attempt-0001.started.json":
            _fail(f"trial does not have exactly attempt-0001 marker: {trial_id}")
        result_path, marker_path = result_files[0], marker_files[0]
        if result_path.is_symlink() or marker_path.is_symlink(): _fail("result/marker symlink forbidden")
        result_rel = str(result_path.relative_to(execution_root))
        marker_rel = str(marker_path.relative_to(execution_root))
        if checksums.get(result_rel) != sha256_file(result_path) or checksums.get(marker_rel) != sha256_file(marker_path):
            _fail(f"frozen result/marker SHA differs: {trial_id}")
        marker = _load_json(marker_path)
        started = validate_start_marker_payload(
            marker, trial_id=trial_id, authorization_sha256=authorization_sha
        )
        result = _load_json(result_path)
        _validate_result_identity(
            result, by_id[trial_id], schema, trial_plan_sha=trial_plan_sha,
            lock_sha=lock_sha, environment_sha=environment_sha,
        )
        created = _parse_utc(result.get("created_at_utc"), f"{trial_id} result")
        if created < started: _fail(f"result timestamp precedes start marker: {trial_id}")
        record = {
            "trial": by_id[trial_id], "result": result, "result_path": result_path,
            "result_file_sha256": checksums[result_rel], "attempt_count": 1,
            "schema_pass": True, "identity_binding_pass": True,
        }
        records.append(record); backend_counts[str(result["backend"])] += 1
        snapshot_pairs[str(result["snapshot_id"])].append(record)
    if backend_counts != Counter({"OPEN3D_POINT_TO_PLANE": 180, "PCL_POINT_TO_PLANE": 180}):
        _fail("Open3D/PCL result inventory differs")
    validate_backend_pair_inputs((record["trial"] for record in records), snapshot_count=180)
    w04 = sum(record["result"]["scene_id"] == "FMB1_W04" for record in records)
    old_w02 = sum(record["result"]["scene_id"] == "FMB1_W02" and record["result"]["attempt"] != 2 for record in records)
    capture = sum(record["result"]["track_id"] != "ZERO_PERTURBATION_TRACK" for record in records)
    infrastructure = sum(record["result"]["infrastructure_status"] != "OK" for record in records)
    if w04 or old_w02 or capture or infrastructure: _fail("excluded track/attempt or infrastructure row entered results")
    inventory = {
        "schema": "mid360_fmb1_postrun_plan_result_inventory_v1", "status": "PASS", "pass": True,
        "planned_trial_id_count": 360, "verified_terminal_trial_id_count": 360,
        "open3d_trial_count": 180, "pcl_trial_count": 180, "snapshot_count": 180,
        "missing_trial_count": 0, "duplicate_trial_count": 0, "orphan_trial_count": 0,
        "unauthorized_trial_count": 0, "second_attempt_row_count": 0,
        "start_marker_count": 360, "result_timestamp_before_start_count": 0,
        "result_authorization_binding_via_start_marker_count": 360,
        "infrastructure_attempt_count": 0, "scientific_result_retry_count": 0,
        "W04_row_count": 0, "old_W02_attempt1_row_count": 0, "capture_radius_row_count": 0,
        "byte_identical_backend_input_pair_count": 180,
    }
    return records, inventory


def audit_input_arrays(repository: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    rows = plan["rows"]
    unique_source: dict[str, Mapping[str, Any]] = {}
    unique_target: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique_source.setdefault(str(row["source_reference"]), row)
        unique_target.setdefault(str(row["target_reference"]), row)
    if len(unique_source) != 180 or len(unique_target) != 18: _fail("unique source/target count differs")
    for row in unique_source.values(): load_canonical_array(repository, row, "source")
    for row in unique_target.values(): load_canonical_array(repository, row, "target")
    return {
        "schema": "mid360_fmb1_postrun_input_freeze_audit_v1", "status": "PASS", "pass": True,
        "source_file_count": 180, "target_file_count": 18,
        "source_file_sha_mismatch_count": 0, "target_file_sha_mismatch_count": 0,
        "source_array_sha_mismatch_count": 0, "target_array_sha_mismatch_count": 0,
        "dtype_mismatch_count": 0, "shape_mismatch_count": 0,
        "contiguity_mismatch_count": 0, "nonfinite_array_count": 0,
        "allow_pickle": False, "array_dtype": "<f8", "array_ndim": 2,
    }


def _update_error(maxima: dict[str, dict[str, float]], category: str, observed: Any, expected: Any) -> None:
    absolute, relative = absolute_relative_error(observed, expected)
    slot = maxima.setdefault(category, {"max_absolute_error": 0.0, "max_relative_error": 0.0})
    slot["max_absolute_error"] = max(slot["max_absolute_error"], absolute)
    slot["max_relative_error"] = max(slot["max_relative_error"], relative)


def _compare_nullable(observed: Any, expected: Any, *, atol: float, rtol: float) -> bool:
    if observed is None or expected is None: return observed is expected
    return close(observed, expected, atol=atol, rtol=rtol)


def compare_association_result(
    result: Mapping[str, Any], metrics: Mapping[str, Any], tolerances: Mapping[str, float]
) -> dict[str, Any]:
    field_pass: dict[str, bool] = {}
    for name in COUNT_FIELDS:
        field_pass[name] = type(result.get(name)) is int and result.get(name) == metrics.get(name)
    for name in TURNOVER_FIELDS + RESIDUAL_FIELDS:
        field_pass[name] = _compare_nullable(
            result.get(name), metrics.get(name),
            atol=tolerances["scalar_atol"], rtol=tolerances["scalar_rtol"],
        )
    for name in NORMAL_ANGLE_FIELDS:
        field_pass[name] = _compare_nullable(
            result.get(name), metrics.get(name),
            atol=tolerances["angle_deg_atol"], rtol=tolerances["angle_deg_rtol"],
        )
    missingness_pass = all(result.get(name) == metrics.get(name) for name in COMMON_STATUS_FIELDS)
    return {
        "field_pass": field_pass,
        "association_pass": all(field_pass.values()),
        "missingness_pass": missingness_pass,
        "failure_field_names": sorted(
            [name for name, passed in field_pass.items() if not passed]
            + [name for name in COMMON_STATUS_FIELDS if result.get(name) != metrics.get(name)]
        ),
    }


def recompute_transforms(records: list[dict[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    tol = contract["tolerances"]; maxima: dict[str, dict[str, float]] = {}
    finite_count = 0; nonfinite_count = 0
    delta_mismatch = translation_mismatch = rotation_mismatch = 0
    for record in records:
        result = record["result"]; failures: list[str] = []
        if result["finite_result"] is False:
            nonfinite_count += 1
            null_fields = ("T_est", "Delta_T", "translation_x_m", "translation_y_m", "translation_z_m", "translation_norm_m", "rotation_angle_rad", "rotation_angle_deg")
            if any(result[name] is not None for name in null_fields): failures.extend(null_fields)
            record["transform_arithmetic_pass"] = not failures
            record["rotation_recomputation_pass"] = not failures
        else:
            finite_count += 1
            try: audit = independent_transform_audit(result["T0"], result["T_est"])
            except Exception:
                failures.append("T_est_rotation_quality")
                record["transform_arithmetic_pass"] = False
                record["rotation_recomputation_pass"] = False
            else:
                transform_ok = close(result["Delta_T"], audit.delta, atol=tol["matrix_atol"], rtol=tol["matrix_rtol"])
                _update_error(maxima, "Delta_T", result["Delta_T"], audit.delta)
                if not transform_ok: failures.append("Delta_T"); delta_mismatch += not transform_ok
                expected_translation = [*audit.translation, audit.translation_norm_m]
                stored_translation = [result["translation_x_m"], result["translation_y_m"], result["translation_z_m"], result["translation_norm_m"]]
                trans_ok = close(stored_translation, expected_translation, atol=tol["scalar_atol"], rtol=tol["scalar_rtol"])
                _update_error(maxima, "translation", stored_translation, expected_translation)
                if not trans_ok: failures.append("translation"); translation_mismatch += not trans_ok
                rad_ok = close(result["rotation_angle_rad"], audit.rotation.angle_rad, atol=tol["angle_rad_atol"], rtol=tol["angle_rad_rtol"])
                deg_ok = close(result["rotation_angle_deg"], audit.rotation.angle_deg, atol=tol["angle_deg_atol"], rtol=tol["angle_deg_rtol"])
                _update_error(maxima, "rotation_rad", result["rotation_angle_rad"], audit.rotation.angle_rad)
                _update_error(maxima, "rotation_deg", result["rotation_angle_deg"], audit.rotation.angle_deg)
                if not rad_ok: failures.append("rotation_angle_rad")
                if not deg_ok: failures.append("rotation_angle_deg")
                rotation_mismatch += not (rad_ok and deg_ok)
                record["transform_arithmetic_pass"] = transform_ok and trans_ok
                record["rotation_recomputation_pass"] = rad_ok and deg_ok
        record.setdefault("failure_field_names", []).extend(failures)
    return {
        "schema": "mid360_fmb1_postrun_transform_recomputation_audit_v1", "status": "PASS" if not (delta_mismatch or translation_mismatch or rotation_mismatch) else "FAIL",
        "pass": not (delta_mismatch or translation_mismatch or rotation_mismatch),
        "finite_transform_count": finite_count, "valid_nonfinite_terminal_count": nonfinite_count,
        "Delta_T_mismatch_count": delta_mismatch, "translation_mismatch_count": translation_mismatch,
        "rotation_recomputation_mismatch_count": rotation_mismatch,
        "numeric_error_maxima": maxima,
    }


def recompute_associations(repository: Path, records: list[dict[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    tol = contract["tolerances"]; maxima: dict[str, dict[str, float]] = {}
    by_target: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records: by_target[str(record["trial"]["target_reference"])].append(record)
    if len(by_target) != 18: _fail("association target grouping differs")
    count_mismatch = turnover_mismatch = residual_mismatch = angle_mismatch = missingness_mismatch = 0
    recomputed = 0
    for target_reference in sorted(by_target):
        group = by_target[target_reference]
        target_array, _ = load_canonical_array(repository, group[0]["trial"], "target")
        target = prepare_target(target_array)
        by_snapshot: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in group: by_snapshot[str(record["trial"]["snapshot_id"])].append(record)
        if len(by_snapshot) != 10: _fail(f"target does not have ten snapshots: {target_reference}")
        for snapshot in sorted(by_snapshot):
            pair = by_snapshot[snapshot]
            source_array, _ = load_canonical_array(repository, pair[0]["trial"], "source")
            initial = prepare_initial(source_array, target, pair[0]["trial"]["T0"])
            for record in pair:
                result = record["result"]; failures: list[str] = []
                if result["finite_result"] is False:
                    association_ok = all(result[name] is None for name in ASSOCIATION_FIELDS)
                    missing_ok = all(result[name] is None for name in COMMON_STATUS_FIELDS)
                else:
                    metrics = analyze_final(initial, result["T_est"]); recomputed += 1
                    comparison = compare_association_result(result, metrics, tol)
                    field_ok = comparison["field_pass"]
                    for name in COUNT_FIELDS:
                        if not field_ok[name]: failures.append(name); count_mismatch += 1
                    for name in TURNOVER_FIELDS:
                        if result[name] is not None and metrics[name] is not None: _update_error(maxima, name, result[name], metrics[name])
                        if not field_ok[name]: failures.append(name); turnover_mismatch += 1
                    for name in RESIDUAL_FIELDS:
                        if result[name] is not None and metrics[name] is not None: _update_error(maxima, name, result[name], metrics[name])
                        if not field_ok[name]: failures.append(name); residual_mismatch += 1
                    for name in NORMAL_ANGLE_FIELDS:
                        if result[name] is not None and metrics[name] is not None: _update_error(maxima, name, result[name], metrics[name])
                        if not field_ok[name]: failures.append(name); angle_mismatch += 1
                    association_ok = comparison["association_pass"]
                    missing_ok = comparison["missingness_pass"]
                    if not missing_ok:
                        failures.extend(name for name in COMMON_STATUS_FIELDS if result[name] != metrics[name])
                        missingness_mismatch += 1
                record["association_recomputation_pass"] = association_ok
                record["missingness_pass"] = missing_ok
                record.setdefault("failure_field_names", []).extend(failures)
            del source_array, initial
        del target_array, target
        gc.collect()
    passed = not any((count_mismatch, turnover_mismatch, residual_mismatch, angle_mismatch, missingness_mismatch))
    return {
        "schema": "mid360_fmb1_postrun_association_recomputation_audit_v1",
        "status": "PASS" if passed else "FAIL", "pass": passed,
        "finite_association_recomputation_count": recomputed,
        "correspondence_count_mismatch_count": count_mismatch,
        "turnover_mismatch_count": turnover_mismatch,
        "residual_rmse_mismatch_count": residual_mismatch,
        "normal_angle_mismatch_count": angle_mismatch,
        "missingness_mismatch_count": missingness_mismatch,
        "numeric_error_maxima": maxima,
        "target_processed_count": 18, "snapshot_processed_count": 180,
        "scientific_aggregation_executed": False,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def _write_outputs(
    output: Path, *, contract: Mapping[str, Any], input_audit: Mapping[str, Any],
    checksum_audit: Mapping[str, Any], git_audit: Mapping[str, Any], authorization_audit: Mapping[str, Any],
    inventory: Mapping[str, Any], transform_audit: Mapping[str, Any], association_audit: Mapping[str, Any],
    records: list[dict[str, Any]], execution_root: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / CONTRACT_FILENAME, contract)
    _write_json(output / "postrun_input_freeze_audit.json", input_audit)
    _write_json(output / "postrun_checksum_audit.json", checksum_audit)
    _write_json(output / "postrun_git_provenance_audit.json", git_audit)
    _write_json(output / "postrun_authorization_lifecycle_audit.json", authorization_audit)
    _write_json(output / "postrun_plan_result_inventory.json", inventory)
    _write_json(output / "postrun_transform_recomputation_audit.json", transform_audit)
    _write_json(output / "postrun_association_recomputation_audit.json", association_audit)
    status_counts = Counter(str(record["result"]["scientific_status"]) for record in records)
    finite_count = sum(record["result"]["finite_result"] is True for record in records)
    nonfinite_count = len(records) - finite_count
    per_trial=[]
    for record in records:
        overall = all(record.get(name) is True for name in (
            "schema_pass", "identity_binding_pass", "transform_arithmetic_pass",
            "rotation_recomputation_pass", "association_recomputation_pass", "missingness_pass",
        ))
        record["overall_pass"] = overall
        per_trial.append({
            "trial_id": record["result"]["trial_id"], "backend": record["result"]["backend"],
            "result_file_sha256": record["result_file_sha256"], "attempt_count": 1,
            "schema_pass": record["schema_pass"], "identity_binding_pass": record["identity_binding_pass"],
            "transform_arithmetic_pass": record["transform_arithmetic_pass"],
            "rotation_recomputation_pass": record["rotation_recomputation_pass"],
            "association_recomputation_pass": record["association_recomputation_pass"],
            "missingness_pass": record["missingness_pass"], "overall_pass": overall,
            "failure_field_names": sorted(set(record.get("failure_field_names", []))),
        })
    with (output / "postrun_trial_verification_manifest.csv").open("x", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=TRIAL_MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in per_trial:
            writer.writerow({**row, "failure_field_names": json.dumps(row["failure_field_names"], separators=(",", ":"))})
    status_inventory={
        "schema":"mid360_fmb1_postrun_status_inventory_v1", "status":"PASS", "pass":True,
        "scientific_status_counts":dict(sorted(status_counts.items())),
        "finite_result_count":finite_count,"valid_nonfinite_terminal_count":nonfinite_count,
        "infrastructure_status_OK_count":360,"infrastructure_failure_count":0,
        "scientific_aggregation_executed":False,
    }
    _write_json(output/"postrun_status_inventory.json",status_inventory)
    overall_pass=(
        input_audit["pass"] and checksum_audit["pass"] and git_audit["pass"]
        and authorization_audit["pass"] and inventory["pass"] and transform_audit["pass"]
        and association_audit["pass"] and all(row["overall_pass"] for row in per_trial)
    )
    final={
        "schema":"mid360_fmb1_exec_r3_postrun_independent_verification_v1",
        "status":"PASS" if overall_pass else "FAIL", "pass":overall_pass,
        "FMB1_POSTRUN_INDEPENDENT_VERIFICATION_PASS":overall_pass,
        "RAW_EXECUTION_COMMIT_VERIFIED":True,"RAW_EXECUTION_TAG_VERIFIED":True,
        "RAW_EXECUTION_BYTES_UNCHANGED":True,
        "VERIFIED_TRIAL_COUNT":360,"VERIFIED_OPEN3D_TRIAL_COUNT":180,"VERIFIED_PCL_TRIAL_COUNT":180,
        "MISSING_TRIALS":0,"DUPLICATE_TRIALS":0,"ORPHAN_TRIALS":0,"UNAUTHORIZED_TRIALS":0,
        "TRANSFORM_ARITHMETIC_MISMATCH_COUNT":transform_audit["Delta_T_mismatch_count"]+transform_audit["translation_mismatch_count"],
        "ROTATION_RECOMPUTATION_MISMATCH_COUNT":transform_audit["rotation_recomputation_mismatch_count"],
        "ASSOCIATION_RECOMPUTATION_MISMATCH_COUNT":association_audit["correspondence_count_mismatch_count"]+association_audit["turnover_mismatch_count"]+association_audit["residual_rmse_mismatch_count"]+association_audit["normal_angle_mismatch_count"],
        "MISSINGNESS_MISMATCH_COUNT":association_audit["missingness_mismatch_count"],
        "AUTHORIZATION_LIFECYCLE_VERIFIED":True,"AUTHORIZATION_REUSABLE":False,
        "SCIENTIFIC_AGGREGATION_EXECUTED":False,"WEAK_RICH_COMPARISON_EXECUTED":False,
        "P_VALUE_COMPUTED":False,"READY_FOR_LOCKED_ANALYSIS_IMPLEMENTATION_AND_FREEZE":overall_pass,
        "READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS":False,
        "raw_execution_commit":RAW_EXECUTION_COMMIT,"r3_lock_fingerprint":R3_FINGERPRINT,
        "verifier_code_commit":git_audit["verifier_code_commit"],
        "raw_checksum_entry_count":743,"raw_checksum_failure_count":0,
        "finite_transform_count":transform_audit["finite_transform_count"],
        "valid_nonfinite_terminal_count":transform_audit["valid_nonfinite_terminal_count"],
        "trial_manifest_row_count":len(per_trial),"failed_trial_count":sum(not row["overall_pass"] for row in per_trial),
        "failure_count":0 if overall_pass else 1,
    }
    _write_json(output/"postrun_independent_verification.json",final)
    md=f"""# FMB1 Exec-R3 Independent Post-run Verification v1\n\nStatus: `{final['status']}`\n\n- Independently verified terminal trials: 360\n- Open3D / PCL: 180 / 180\n- Missing / duplicate / orphan / unauthorized: 0 / 0 / 0 / 0\n- Transform / rotation / association / missingness mismatches: {final['TRANSFORM_ARITHMETIC_MISMATCH_COUNT']} / {final['ROTATION_RECOMPUTATION_MISMATCH_COUNT']} / {final['ASSOCIATION_RECOMPUTATION_MISMATCH_COUNT']} / {final['MISSINGNESS_MISMATCH_COUNT']}\n- Raw execution bytes unchanged: true\n- Authorization lifecycle consumed and non-reusable: true\n- Scientific aggregation executed: false\n- Ready for locked scientific analysis: false\n\nNo scene-class aggregation, hypothesis test, correlation, systematic fraction, capture-radius analysis, or scientific conclusion was computed.\n"""
    with (output/"postrun_independent_verification.md").open("x",encoding="utf-8") as stream: stream.write(md)
    checksum_lines=[]
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n")
    with (output/"SHA256SUMS").open("x",encoding="ascii",newline="") as stream: stream.writelines(checksum_lines)
    return final


def verify_frozen_real_results(
    *, repository: Path, execution_results_root: Path, r3_lock_dir: Path,
    raw_execution_commit: str, raw_execution_tag: str, verifier_code_commit: str,
    output_dir: Path, confirm_read_frozen_real_results: bool,
) -> dict[str, Any]:
    if confirm_read_frozen_real_results is not True:
        _fail("--confirm-read-frozen-real-results is required")
    contract=load_contract(); root=repository.resolve(strict=True)
    execution=_within(root,execution_results_root); lock_dir=_within(root,r3_lock_dir)
    if not execution.is_dir() or not lock_dir.is_dir(): _fail("execution/lock root is not a directory")
    git_audit=git_provenance_audit(
        root,execution,output_dir,raw_execution_commit=raw_execution_commit,
        raw_execution_tag=raw_execution_tag,verifier_code_commit=verifier_code_commit,
    )
    checksums,checksum_audit=parse_checksum_manifest(execution)
    lock,plan,schema,plan_path,_=_load_lock_plan_schema(root,lock_dir)
    validate_plan_contract(plan)
    plan_sha=sha256_file(plan_path); lock_sha=sha256_file(lock_dir/LOCK_FILENAME)
    env_path=_within(root,Path(lock["bindings"]["environment_manifest"]["repository_relative_path"]))
    env_sha=sha256_file(env_path)
    run_manifest=_load_json(execution/"execution/run_manifest.json")
    if (
        run_manifest.get("status")!="COMPLETE" or run_manifest.get("completed_trial_count")!=360
        or run_manifest.get("lock_fingerprint")!=R3_FINGERPRINT
        or run_manifest.get("execution_code_commit")!=R3_EXECUTION_CODE_COMMIT
    ): _fail("frozen run manifest bindings differ")
    authorization_audit=authorization_lifecycle_audit(root,execution,run_manifest)
    records,inventory=discover_and_validate_inventory(
        root,execution,plan,schema,checksums,authorization_audit["authorization_sha256"],
        trial_plan_sha=plan_sha,lock_sha=lock_sha,environment_sha=env_sha,
    )
    input_audit=audit_input_arrays(root,plan)
    transform_audit=recompute_transforms(records,contract)
    association_audit=recompute_associations(root,records,contract)
    return _write_outputs(
        _within(root,output_dir,must_exist=False),contract=contract,input_audit=input_audit,
        checksum_audit=checksum_audit,git_audit=git_audit,authorization_audit=authorization_audit,
        inventory=inventory,transform_audit=transform_audit,association_audit=association_audit,
        records=records,execution_root=execution,
    )


def fixture_only_self_test() -> dict[str, Any]:
    translation=np.eye(4); translation[:3,3]=[0.1,-0.2,0.3]
    audit=independent_transform_audit(IDENTITY,translation)
    if not close(audit.translation,[0.1,-0.2,0.3],atol=1e-12,rtol=1e-10): _fail("fixture translation failed")
    angle=0.2; rotation=np.eye(4); rotation[:2,:2]=[[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]]
    rotated=independent_transform_audit(IDENTITY,rotation)
    if not close(rotated.rotation.angle_rad,angle,atol=1e-12,rtol=1e-10): _fail("fixture rotation failed")
    return {
        "schema":"mid360_fmb1_postrun_verifier_fixture_only_v1","status":"PASS","pass":True,
        "FIXTURE_ONLY":True,"NOT_REAL_FMB1":True,"real_result_values_read":False,
        "backend_import_count":0,"backend_call_count":0,"scientific_aggregation_executed":False,
    }


__all__=[
    "IndependentPostrunVerificationError","audit_input_arrays","authorization_lifecycle_audit",
    "discover_and_validate_inventory","fixture_only_self_test","git_provenance_audit",
    "compare_association_result","load_canonical_array","load_contract","parse_checksum_manifest","recompute_associations",
    "recompute_transforms","sha256_file","validate_backend_pair_inputs",
    "validate_inventory_identity_sets","validate_plan_contract","validate_schema_value",
    "validate_start_marker_payload","verify_frozen_real_results",
]
