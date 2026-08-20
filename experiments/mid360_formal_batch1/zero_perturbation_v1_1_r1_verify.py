"""Independent verifier for the FMB1 zero-perturbation v1.1-R1 lock.

This module intentionally imports neither the lock producer nor the runner.
All counts, identities, hashes, hierarchy, and authority gates are recomputed.
"""

from __future__ import annotations

import csv
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


LOCK_FILE = "formal_batch1_zero_perturbation_lock_v1_1.json"
LOCK_SCHEMA = "mid360_fmb1_zero_perturbation_formal_lock_v1_1_r1"
PLAN_SCHEMA = "mid360_fmb1_zero_perturbation_trial_plan_v1_1_r1"
AMENDMENT_ID = "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1"
PHYSICAL_SEMANTICS = "NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT"
BACKEND_SHA = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
VERSIONS = {
    "python": "3.11.15", "numpy": "1.26.4", "scipy": "1.11.4",
    "open3d": "0.19.0+b012259", "pcl": "1.15.1",
}
SCENE_CLASS = {
    "FMB1_R01": "RICH", "FMB1_R02": "RICH", "FMB1_R03": "RICH",
    "FMB1_W01": "WEAK", "FMB1_W02": "WEAK", "FMB1_W03": "WEAK",
}
BACKENDS = {"OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE"}
IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ORIGINAL_PROPOSAL_JSON_SHA256 = (
    "4837f6bd37e3a0f19c4b276964bbd066df09b86e58af288aa103aca704be936e"
)
ORIGINAL_PROPOSAL_MD_SHA256 = (
    "836822c553929a2e5b9e6cdf85b48d318fe71f91aa47447d908d030516e7f4ed"
)
PCL_PKG_CONFIG_SHA = "7590aa30f53362ced6ba582bd8b34cfe3fca67e0f84fb6b01a0542d3c1bc4e03"
PCL_COMMON_PC_SHA = "7af03303621644e94bcb6c1ffdd54e01ee94e203ce54cf5e54ff5d1eb4bef73b"
EXECUTION_BINDINGS = {
    "execution_runner", "execution_runner_cli",
    "execution_experiments_package_init",
    "execution_mid360_formal_batch1_package_init",
    "execution_phase_a_harness_package_init", "execution_environment",
    "execution_result_validator",
    "execution_open3d_backend", "execution_pcl_backend",
    "execution_common_association", "execution_rotation_metrics",
    "execution_metrics", "execution_types",
}
AUTHORITATIVE_RUNTIME_ROOT = (
    "zero_perturbation_runtime/mid360_formal_batch1_zero_perturbation_v1_1"
)
DEFAULT_AUTHORIZATION_PATH = (
    "results/mid360_formal_batch1/"
    "formal_registration_authorization_v1_1_r1.json"
)
# Kept independently from the producer on purpose: a lock may not relabel a
# copied file as one of the protocol's canonical authorities.
EXPECTED_BINDING_PATHS = {
    "original_preregistration": "experiments/mid360_formal_batch1/preregistration.yaml",
    "original_capture_radius_analysis_protocol": "experiments/mid360_formal_batch1/analysis_protocol.md",
    "active_amendment": "experiments/mid360_formal_batch1/amendments/zero_perturbation_mainline_v1_1_r1.json",
    "active_amendment_md": "experiments/mid360_formal_batch1/amendments/zero_perturbation_mainline_v1_1_r1.md",
    "amendment_activation_record": "experiments/mid360_formal_batch1/amendments/amendment_activation_record_v1_1_r1.json",
    "active_protocol_pointer": "experiments/mid360_formal_batch1/ACTIVE_PROTOCOL.json",
    "final_dataset_pointer": "results/mid360_formal_batch1/CURRENT_FINAL_DATASET.json",
    "final_scene_registry": "results/mid360_formal_batch1/final_dataset_v1/final_scene_registry.yaml",
    "final_station_registry": "results/mid360_formal_batch1/final_dataset_v1/final_station_registry.yaml",
    "admitted_bag_manifest": "results/mid360_formal_batch1/final_dataset_v1/final_raw_bag_manifest.csv",
    "acquisition_attempt_lineage": "results/mid360_formal_batch1/final_dataset_v1/acquisition_attempt_lineage.json",
    "invalid_attempt_archive_manifest": "results/mid360_formal_batch1/final_dataset_v1/invalid_attempt_archive_manifest.json",
    "target_manifest": "results/mid360_formal_batch1/final_dataset_v1/final_target_manifest.csv",
    "snapshot_manifest": "results/mid360_formal_batch1/final_dataset_v1/final_snapshot_manifest.csv",
    "geometry_manifest": "results/mid360_formal_batch1/final_dataset_v1/final_geometry_manifest.csv",
    "trial_plan_json": "experiments/mid360_formal_batch1/zero_perturbation_trial_plan_v1_1.json",
    "trial_plan_csv": "experiments/mid360_formal_batch1/zero_perturbation_trial_plan_v1_1.csv",
    "result_schema": "experiments/mid360_formal_batch1/zero_perturbation_trial_result_schema_v1_1.json",
    "analysis_contract": "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_contract_v1_1_r1.json",
    "analysis_protocol": "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_protocol_v1_1_r1.md",
    "analysis_missingness_clarification": "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1.json",
    "analysis_missingness_clarification_md": "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1.md",
    "analysis_missingness_clarification_transition": "experiments/mid360_formal_batch1/amendments/analysis_missingness_clarification_transition_v1_1_r1_c1.json",
    "analysis_preclarification_history_inventory": "experiments/mid360_formal_batch1/amendments/history/zero_perturbation_v1_1_r1_active_pre_missingness_clarification/active_preclarification_inventory.json",
    "backend_parameter_contract": "frozen_assets/backend_parameter_contract.json",
    "pcl_executable": "bin/pcl_point_to_plane_cli",
    "environment_manifest": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/environment_manifest.json",
    "prelock_no_icp_attestation": "results/mid360_formal_batch1/final_dataset_v1/NO_ICP_ATTESTATION.json",
    "final_dataset_prelock_reauthentication": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/final_dataset_prelock_reauthentication.json",
    "protocol_transition_independent_verification": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/protocol_r1_active_transition_independent_verification.json",
    "protocol_c1_missingness_independent_verification": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/protocol_r1_c1_missingness_independent_verification.json",
    "activation_review": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/zero_perturbation_v1_1_activation_review.json",
    "trial_plan_independent_verification": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/trial_plan_independent_verification.json",
    "original_zero_perturbation_proposal_json": "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.json",
    "original_zero_perturbation_proposal_md": "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.md",
    "proposal_superseded_sidecar": "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.SUPERSEDED.json",
    "proposal_correction_record": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/proposal_correction_record_v1_1_r1.json",
    "proposal_difference_report": "results/mid360_formal_batch1/zero_perturbation_v1_1_lock/proposal_difference_report_v1_1_r1.json",
    "w04_superseded_history": "results/mid360_formal_batch1/history/final_dataset_w04_replacement_superseded_v1.SUPERSEDED.json",
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
}


class R1IndependentVerificationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise R1IndependentVerificationError(f"FMB1_R1_INDEPENDENT_VERIFY_FAIL: {message}")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"JSON root must be mapping: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            _fail(f"CSV header missing: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        _fail(f"CSV row exceeds header: {path}")
    return rows


def _resolve(root: Path, value: Any, label: str) -> Path:
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
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        _fail(f"{label} missing/escapes repository: {exc}")
    if not resolved.is_file() or resolved.is_symlink():
        _fail(f"{label} must be regular non-symlink file")
    return resolved


def _identity(value: Any) -> bool:
    try:
        return len(value) == 4 and all(len(row) == 4 for row in value) and all(
            float(value[i][j]) == IDENTITY[i][j] for i in range(4) for j in range(4)
        )
    except (TypeError, ValueError, IndexError):
        return False


def _recursive_values(value: Any) -> list[Any]:
    output = [value]
    if isinstance(value, Mapping):
        for child in value.values():
            output.extend(_recursive_values(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_recursive_values(child))
    return output


def _verify_sums(lock_dir: Path) -> int:
    """Verify immutable lock core without a verifier-report self-reference."""

    sums = lock_dir / "LOCK_CORE_SHA256SUMS"
    if not sums.is_file():
        _fail("LOCK_CORE_SHA256SUMS missing")
    declared: dict[str, str] = {}
    for number, line in enumerate(sums.read_text(encoding="ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/].*)", line)
        if match is None:
            _fail(f"malformed SHA256SUMS line {number}")
        digest, name = match.groups()
        if name in {".", ".."} or "/" in name or "\\" in name:
            _fail(f"unsafe core checksum path: {name}")
        if name in declared or name in {"SHA256SUMS", "LOCK_CORE_SHA256SUMS"}:
            _fail(f"duplicate/self checksum entry: {name}")
        path = lock_dir / name
        if path.is_symlink() or not path.is_file() or _sha(path) != digest:
            _fail(f"checksum mismatch: {name}")
        declared[name] = digest
    required = {
        LOCK_FILE, "formal_batch1_zero_perturbation_lock_v1_1.sha256",
        "lock_inventory.csv", "lock_fingerprint.json", "NO_ICP_ATTESTATION.json",
        "environment_manifest.json", "final_dataset_prelock_reauthentication.json",
    }
    if not required.issubset(declared):
        _fail(f"core sums lack required files: {sorted(required-set(declared))}")
    return len(declared)


def verify_release_checksums(lock_dir: Path) -> dict[str, Any]:
    """Independently verify the finalized, non-self-referential release list.

    The resulting report must be stored outside ``lock_dir`` so the directory's
    SHA256SUMS can cover every regular release artifact without self-reference.
    """

    lexical = Path(lock_dir)
    if lexical.is_symlink() or not lexical.is_dir():
        _fail("release lock directory is missing, non-directory, or symlinked")
    checksum = lexical / "SHA256SUMS"
    if checksum.is_symlink() or not checksum.is_file():
        _fail("final release SHA256SUMS is missing or symlinked")
    declared: dict[str, str] = {}
    for number, line in enumerate(checksum.read_text(encoding="ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/].*)", line)
        if match is None:
            _fail(f"malformed release SHA256SUMS line {number}")
        digest, name = match.groups()
        if name in {".", ".."} or "/" in name or "\\" in name:
            _fail(f"unsafe release checksum path: {name}")
        if name in declared or name == "SHA256SUMS":
            _fail(f"duplicate/self release checksum entry: {name}")
        declared[name] = digest
    actual: dict[str, Path] = {}
    for path in lexical.iterdir():
        if path.name == "SHA256SUMS":
            continue
        if path.is_symlink():
            _fail(f"release directory contains symlink: {path.name}")
        if path.is_dir():
            _fail(f"release directory contains unexpected subdirectory: {path.name}")
        if not path.is_file():
            _fail(f"release directory contains non-regular entry: {path.name}")
        actual[path.name] = path
    if set(declared) != set(actual):
        _fail(
            "release checksum set differs from regular files: "
            f"missing={sorted(set(actual)-set(declared))}, "
            f"extra={sorted(set(declared)-set(actual))}"
        )
    for name, path in actual.items():
        if _sha(path) != declared[name]:
            _fail(f"release checksum mismatch: {name}")
    return {
        "schema": "mid360_fmb1_zero_perturbation_release_checksum_verification_v1_1_r1",
        "status": "PASS", "pass": True, "failure_count": 0,
        "sha256sums_sha256": _sha(checksum),
        "covered_regular_file_count": len(actual),
        "set_equality_verified": True,
        "path_safety_verified": True,
        "symlink_count": 0,
    }


def _verify_environment(payload: Mapping[str, Any], root: Path, *, remeasure: bool) -> None:
    if payload.get("schema") != "mid360_fmb1_zero_perturbation_environment_v1_1_r1" or payload.get("qualification_pass") is not True:
        _fail("environment is not R1-qualified")
    if payload.get("versions") != VERSIONS or payload.get("expected_versions") != VERSIONS:
        _fail("environment versions differ")
    if remeasure:
        observed = {"python": platform.python_version()}
        for name in ("numpy", "scipy"):
            try:
                observed[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                observed[name] = "NOT_INSTALLED"
        evidence = payload.get("open3d_version_evidence")
        if not isinstance(evidence, Mapping):
            _fail("Open3D static version evidence missing")
        source = Path(str(evidence.get("path")))
        if source.is_symlink() or not source.is_file() or _sha(source) != evidence.get("sha256"):
            _fail("Open3D static version source changed")
        spec = importlib.util.find_spec("open3d")
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != source.resolve():
            _fail("current interpreter resolves a different Open3D package")
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        literals = [statement.value.value for statement in tree.body
                    if isinstance(statement, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "__version__"
                            for target in statement.targets)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)]
        observed["open3d"] = literals[0] if len(literals) == 1 else "STATIC_PROBE_INVALID"
        if any(observed[key] != VERSIONS[key] for key in observed):
            _fail(f"live Python environment differs: {observed}")
    pcl = payload.get("pcl_cli")
    contract = payload.get("backend_contract")
    if not isinstance(pcl, Mapping) or not isinstance(contract, Mapping):
        _fail("environment file evidence missing")
    pcl_path = _resolve(root, pcl.get("path"), "PCL executable")
    contract_path = _resolve(root, contract.get("path"), "backend contract")
    if _sha(pcl_path) != pcl.get("sha256") or pcl.get("ldd_audit_pass") is not True or pcl.get("ldd_missing_dependencies") != []:
        _fail("PCL SHA/ldd evidence differs")
    if not bool(pcl_path.stat().st_mode & 0o111):
        _fail("PCL executable bit differs")
    if pcl.get("registration_executable_invocation_count") != 0:
        _fail("environment audit invoked PCL registration CLI")
    pkg_config = Path(str(pcl.get("pkg_config_executable")))
    common_pc = Path(str(pcl.get("pcl_common_pc")))
    if (pkg_config.is_symlink() or common_pc.is_symlink() or not pkg_config.is_file()
            or not common_pc.is_file() or _sha(pkg_config) != PCL_PKG_CONFIG_SHA
            or _sha(common_pc) != PCL_COMMON_PC_SHA
            or pcl.get("pkg_config_executable_sha256") != PCL_PKG_CONFIG_SHA
            or pcl.get("pcl_common_pc_sha256") != PCL_COMMON_PC_SHA):
        _fail("frozen PCL pkg-config evidence differs")
    prefix = Path(str(pcl.get("frozen_environment_prefix"))).resolve(strict=True)
    if (pkg_config.resolve() != prefix / "bin/pkg-config"
            or common_pc.resolve() != prefix / "lib/pkgconfig/pcl_common.pc"
            or pcl.get("probe_path") != f"{prefix / 'bin'}:/usr/bin:/bin"
            or pcl.get("probe_pkg_config_path") != str(prefix / "lib/pkgconfig")
            or pcl.get("probe_ld_library_path") != str(prefix / "lib")):
        _fail("frozen PCL probe PATH/PKG_CONFIG_PATH differs")
    if remeasure:
        live = subprocess.run(
            ["/usr/bin/ldd", str(pcl_path)], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={"PATH": str(pcl.get("probe_path")),
                 "PKG_CONFIG_PATH": str(pcl.get("probe_pkg_config_path")),
                 "LD_LIBRARY_PATH": str(pcl.get("probe_ld_library_path")),
                 "LANG": "C", "LC_ALL": "C"},
        ).stdout.splitlines()
        if not live or any("not found" in line.lower() for line in live):
            _fail("live PCL ldd dependency audit failed")
    if _sha(contract_path) != BACKEND_SHA or contract.get("sha256") != BACKEND_SHA:
        _fail("backend contract SHA differs")
    source = _json(contract_path)
    for backend in ("open3d", "pcl"):
        row = source.get(backend)
        if not isinstance(row, Mapping):
            _fail(f"backend canonical section missing: {backend}")
        computed = hashlib.sha256(json.dumps(row.get("parameters"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if computed != row.get("canonical_sha256"):
            _fail(f"backend canonical SHA differs: {backend}")


def _verify_final_dataset(root: Path, bindings: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    pointer_path = _resolve(root, bindings["final_dataset_pointer"]["repository_relative_path"], "final pointer")
    pointer = _json(pointer_path)
    if pointer.get("W04_INCLUDED_IN_FINAL_SET") is not False or pointer.get("W04_IDENTIFIER_RETIRED") is not True:
        _fail("W04 is not retired/excluded")
    if pointer.get("w02_active_attempt") != 2 or pointer.get("actual_formal_trials") != 0:
        _fail("final pointer does not select W02 attempt2 pre-ICP")
    active_dir = (root / str(pointer.get("active_dataset_path"))).resolve(strict=True)
    manifest_path = active_dir / "final_dataset_manifest.json"
    if _sha(manifest_path) != pointer.get("active_manifest_sha256"):
        _fail("final pointer manifest SHA differs")

    scene_doc = _json(_resolve(root, bindings["final_scene_registry"]["repository_relative_path"], "scene registry"))
    scenes = scene_doc.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 6:
        _fail("scene registry count differs")
    scene_map = {str(row.get("scene_id")): row for row in scenes if isinstance(row, Mapping)}
    if set(scene_map) != set(SCENE_CLASS):
        _fail("final scenes differ")
    for scene, expected in SCENE_CLASS.items():
        row = scene_map[scene]
        if row.get("final_geometry_class") != expected or row.get("admitted") is not True:
            _fail(f"geometry class/admission differs: {scene}")
        if int(row.get("attempt", -1)) != (2 if scene == "FMB1_W02" else 1):
            _fail(f"scene attempt differs: {scene}")

    station_doc = _json(_resolve(root, bindings["final_station_registry"]["repository_relative_path"], "station registry"))
    stations = station_doc.get("stations")
    if not isinstance(stations, list) or len(stations) != 18:
        _fail("station registry count differs")
    station_keys = {(str(row.get("scene_id")), str(row.get("station_id"))) for row in stations if isinstance(row, Mapping)}
    if station_keys != {(scene, f"S0{i}") for scene in SCENE_CLASS for i in range(1, 4)}:
        _fail("station registry keys differ")
    for row in stations:
        scene = str(row.get("scene_id"))
        if (int(row.get("attempt", -1)) != (2 if scene == "FMB1_W02" else 1)
                or row.get("acquisition_status") != "ACQUISITION_PASS"):
            _fail(f"station attempt/status differs: {scene}/{row.get('station_id')}")

    raw_rows = _csv(_resolve(root, bindings["admitted_bag_manifest"]["repository_relative_path"], "admitted bags"))
    if len(raw_rows) != 36:
        _fail("admitted bag count differs")
    raw_roles: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in raw_rows:
        scene, station = row.get("scene_id", ""), row.get("station_id", "")
        if scene not in SCENE_CLASS or (scene, station) not in station_keys:
            _fail("non-final admitted bag row")
        if row.get("attempt") != ("2" if scene == "FMB1_W02" else "1"):
            _fail(f"admitted bag attempt differs: {scene}/{station}")
        if SHA_RE.fullmatch(row.get("sha256", "")) is None:
            _fail("admitted bag SHA differs")
        raw_roles[(scene, station)].add(row.get("role", ""))
    if set(raw_roles) != station_keys or any(roles != {"MAP", "QUERY"} for roles in raw_roles.values()):
        _fail("admitted bags are not exactly MAP/QUERY per station")

    lineage = _json(_resolve(root, bindings["acquisition_attempt_lineage"]["repository_relative_path"], "attempt lineage"))
    expected_lineage = {
        "scene_id": "FMB1_W02", "invalid_attempt": 1, "valid_attempt": 2,
        "invalid_attempt_status": "INVALID_ACQUISITION",
        "valid_attempt_status": "GEOMETRY_ADMITTED",
        "valid_attempt_final_geometry_class": "WEAK",
        "invalid_attempt_snapshot_count_in_final": 0,
        "valid_attempt_snapshot_count_in_final": 30,
        "W04_IDENTIFIER_RETIRED": True, "W04_INCLUDED_IN_FINAL_SET": False,
        "formal_trial_count_at_correction": 0,
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            _fail(f"attempt lineage differs: {key}")
    archive = _json(_resolve(root, bindings["invalid_attempt_archive_manifest"]["repository_relative_path"], "invalid archive"))
    bags = archive.get("bags")
    if archive.get("attempt") != 1 or archive.get("archived_bag_count") != 6 or not isinstance(bags, list) or len(bags) != 6:
        _fail("old W02 attempt1 archive lineage differs")
    if any(row.get("included_in_active_dataset") is not False for row in bags if isinstance(row, Mapping)):
        _fail("old W02 attempt1 entered active dataset")

    snapshot_rows = _csv(_resolve(root, bindings["snapshot_manifest"]["repository_relative_path"], "snapshot manifest"))
    if len(snapshot_rows) != 180:
        _fail("final snapshot count differs")
    snapshots: dict[str, dict[str, str]] = {}
    for row in snapshot_rows:
        sid = row.get("snapshot_id", "")
        if not sid or sid in snapshots or row.get("scene_id") not in SCENE_CLASS:
            _fail("duplicate/orphan/non-final snapshot")
        expected_attempt = "2" if row["scene_id"] == "FMB1_W02" else "1"
        if row.get("attempt") != expected_attempt:
            _fail(f"snapshot attempt differs: {sid}")
        snapshots[sid] = row
    if any("W04" in sid for sid in snapshots) or sum(row["scene_id"] == "FMB1_W02" for row in snapshots.values()) != 30:
        _fail("W04/old W02 attempt snapshot boundary differs")
    snapshot_counts = Counter((row["scene_id"], row["station_id"]) for row in snapshots.values())
    if set(snapshot_counts) != station_keys or any(value != 10 for value in snapshot_counts.values()):
        _fail("snapshot count is not exactly 10 per station")

    target_rows = _csv(_resolve(root, bindings["target_manifest"]["repository_relative_path"], "target manifest"))
    if len(target_rows) != 18:
        _fail("target count differs")
    targets: dict[tuple[str, str], dict[str, str]] = {}
    for row in target_rows:
        key = (row.get("scene_id", ""), row.get("station_id", ""))
        if (key in targets or key not in station_keys
                or int(row.get("query_contribution_to_target", "-1")) != 0
                or row.get("attempt") != ("2" if key[0] == "FMB1_W02" else "1")):
            _fail("duplicate target or QUERY contribution")
        targets[key] = row
    if set(targets) != station_keys:
        _fail("target keys differ from final stations")
    geometry = _csv(_resolve(root, bindings["geometry_manifest"]["repository_relative_path"], "geometry manifest"))
    if len(geometry) != 180 or {row.get("snapshot_id") for row in geometry} != set(snapshots):
        _fail("geometry manifest snapshot inventory differs")
    for row in geometry:
        scene = row.get("scene_id", "")
        if row.get("attempt") != ("2" if scene == "FMB1_W02" else "1"):
            _fail("geometry attempt differs")
    return snapshots, targets


def _verify_activation(root: Path, bindings: Mapping[str, Mapping[str, Any]]) -> None:
    amendment = _json(_resolve(root, bindings["active_amendment"]["repository_relative_path"], "amendment"))
    activation = _json(_resolve(root, bindings["amendment_activation_record"]["repository_relative_path"], "activation"))
    pointer = _json(_resolve(root, bindings["active_protocol_pointer"]["repository_relative_path"], "ACTIVE_PROTOCOL"))
    if amendment.get("amendment_id") != AMENDMENT_ID or amendment.get("status") != "ACTIVE" or amendment.get("activation_effective") is not True:
        _fail("R1 amendment is not active")
    if activation.get("amendment_id") != AMENDMENT_ID or activation.get("status") != "ACTIVE":
        _fail("R1 activation record differs")
    if activation.get("ACTIVATED_BEFORE_ANY_FORMAL_ICP", activation.get("activated_before_any_formal_icp")) is not True:
        _fail("activation occurred after ICP")
    if int(activation.get("FORMAL_TRIAL_COUNT_AT_ACTIVATION", activation.get("formal_trial_count_at_activation", -1))) != 0:
        _fail("activation trial count is nonzero")
    if pointer.get("active_amendment_id", pointer.get("amendment_id")) != AMENDMENT_ID or pointer.get("status") != "ACTIVE":
        _fail("ACTIVE_PROTOCOL pointer differs")


def _verify_proposal_correction_history(
    root: Path, bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    def bound(name: str) -> Path:
        return _resolve(root, bindings[name]["repository_relative_path"], name)

    proposal_json = bound("original_zero_perturbation_proposal_json")
    proposal_md = bound("original_zero_perturbation_proposal_md")
    if _sha(proposal_json) != ORIGINAL_PROPOSAL_JSON_SHA256:
        _fail("historical proposal JSON is not byte-identical")
    if _sha(proposal_md) != ORIGINAL_PROPOSAL_MD_SHA256:
        _fail("historical proposal Markdown is not byte-identical")
    sidecar_path = bound("proposal_superseded_sidecar")
    sidecar = _json(sidecar_path)
    old_json = sidecar.get("historical_proposal_json", {})
    old_md = sidecar.get("historical_proposal_markdown", {})
    if (
        sidecar.get("status") != "SUPERSEDED_PROPOSAL"
        or sidecar.get("historical_proposal_modified") is not False
        or sidecar.get("correction_reason_code")
        != "OPERATOR_CONFIRMED_WRONG_SCENE_LOCATION"
        or sidecar.get("correction_before_any_formal_icp") is not True
        or int(sidecar.get("correction_at_formal_trial_count", -1)) != 0
        or sidecar.get("registration_evidence_used") is not False
        or old_json.get("path")
        != EXPECTED_BINDING_PATHS["original_zero_perturbation_proposal_json"]
        or old_json.get("sha256") != ORIGINAL_PROPOSAL_JSON_SHA256
        or old_md.get("path")
        != EXPECTED_BINDING_PATHS["original_zero_perturbation_proposal_md"]
        or old_md.get("sha256") != ORIGINAL_PROPOSAL_MD_SHA256
    ):
        _fail("historical proposal supersession semantics differ")
    correction = _json(bound("proposal_correction_record"))
    old = correction.get("old_proposal", {})
    lineage = correction.get("lineage", {})
    lineage_path = bound("acquisition_attempt_lineage")
    if (
        correction.get("status") != "RECORDED_PRE_ACTIVATION"
        or correction.get("correction_reason")
        != "OPERATOR_CONFIRMED_WRONG_SCENE_LOCATION"
        or correction.get("correction_before_any_formal_icp") is not True
        or int(correction.get("correction_at_formal_trial_count", -1)) != 0
        or correction.get("registration_evidence_used") is not False
        or old.get("json_sha256") != ORIGINAL_PROPOSAL_JSON_SHA256
        or old.get("markdown_sha256") != ORIGINAL_PROPOSAL_MD_SHA256
        or old.get("supersession_record_sha256") != _sha(sidecar_path)
        or lineage.get("scene_id") != "FMB1_W02"
        or lineage.get("attempt1_status") != "INVALID_ACQUISITION"
        or lineage.get("attempt2_status") != "GEOMETRY_ADMITTED"
        or lineage.get("attempt2_final_geometry_class") != "WEAK"
        or lineage.get("attempt2_in_final_dataset") is not True
        or lineage.get("w04_identifier_retired") is not True
        or lineage.get("w04_in_final_dataset") is not False
        or lineage.get("source_sha256") != _sha(lineage_path)
        or correction.get("FORMAL_LOCK_ISSUED") is not False
        or correction.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or int(correction.get("actual_formal_trials", -1)) != 0
    ):
        _fail("proposal correction/lineage semantics differ")
    difference = _json(bound("proposal_difference_report"))
    reviewed = {
        item.get("path"): item.get("sha256")
        for item in difference.get("reviewed_inputs", [])
        if isinstance(item, Mapping)
    }
    if (
        difference.get("status")
        != "DIFFERENCES_DOCUMENTED_VERSIONED_CORRECTION_REQUIRED"
        or difference.get("old_proposal_preserved_byte_for_byte") is not True
        or difference.get("old_proposal_json_sha256_before_r1")
        != ORIGINAL_PROPOSAL_JSON_SHA256
        or difference.get("old_proposal_md_sha256_before_r1")
        != ORIGINAL_PROPOSAL_MD_SHA256
        or reviewed.get(EXPECTED_BINDING_PATHS["original_zero_perturbation_proposal_json"])
        != ORIGINAL_PROPOSAL_JSON_SHA256
        or reviewed.get(EXPECTED_BINDING_PATHS["original_zero_perturbation_proposal_md"])
        != ORIGINAL_PROPOSAL_MD_SHA256
        or reviewed.get(EXPECTED_BINDING_PATHS["acquisition_attempt_lineage"])
        != _sha(lineage_path)
        or difference.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or int(difference.get("actual_formal_trials", -1)) != 0
    ):
        _fail("proposal difference/history bindings differ")


def _verify_missingness_clarification(
    root: Path, bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    def bound(name: str) -> Path:
        return _resolve(root, bindings[name]["repository_relative_path"], name)

    clarification_path = bound("analysis_missingness_clarification")
    clarification = _json(clarification_path)
    rule = clarification.get("reassociation_missingness_rule", {})
    reasons = [
        "NO_INITIAL_CORRESPONDENCE", "NO_FINAL_CORRESPONDENCE",
        "INSUFFICIENT_VALID_NORMALS", "NONFINITE_COMMON_METRICS", "OTHER",
    ]
    if (
        clarification.get("status") != "ACTIVE_PRELOCK_CLARIFICATION"
        or clarification.get("clarification_before_formal_lock") is not True
        or clarification.get("clarification_before_any_formal_icp") is not True
        or int(clarification.get("clarification_at_formal_trial_count", -1)) != 0
        or clarification.get("registration_result_used") is not False
        or rule.get("common_invalid_reason_retained") is not True
        or rule.get("required_result_status_fields") != [
            "common_association_valid", "common_association_invalid_reason",
            "common_association_invalid_detail",
        ]
        or rule.get("common_association_invalid_reason_enum") != reasons
        or clarification.get("FORMAL_LOCK_ISSUED") is not False
        or clarification.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or int(clarification.get("actual_formal_trials", -1)) != 0
    ):
        _fail("R1-C1 missingness clarification semantics differ")
    contract = _json(bound("analysis_contract"))
    contract_link = contract.get("prelock_missingness_clarification", {})
    if (
        contract_link.get("path")
        != EXPECTED_BINDING_PATHS["analysis_missingness_clarification"]
        or contract_link.get("sha256") != _sha(clarification_path)
        or contract_link.get("clarification_before_formal_lock") is not True
        or int(contract_link.get("clarification_at_formal_trial_count", -1)) != 0
    ):
        _fail("analysis contract/R1-C1 clarification hash differs")
    inventory_path = bound("analysis_preclarification_history_inventory")
    inventory = _json(inventory_path)
    if (
        inventory.get("status")
        != "ACTIVE_BYTES_PRESERVED_BEFORE_PRELOCK_CLARIFICATION"
        or int(inventory.get("formal_trial_count_at_archive", -1)) != 0
        or inventory.get("formal_lock_issued_at_archive") is not False
        or inventory.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or int(inventory.get("actual_formal_trials", -1)) != 0
    ):
        _fail("preclarification history inventory state differs")
    rows = inventory.get("files")
    if not isinstance(rows, list) or len(rows) != 7:
        _fail("preclarification history inventory count differs")
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("preclarification history row malformed")
        name, digest = row.get("name"), row.get("sha256")
        if not isinstance(name, str) or Path(name).name != name or SHA_RE.fullmatch(str(digest)) is None:
            _fail("preclarification history path/SHA malformed")
        archived = inventory_path.parent / name
        if archived.is_symlink() or not archived.is_file() or _sha(archived) != digest:
            _fail(f"preclarification archived byte/hash differs: {name}")
    transition = _json(bound("analysis_missingness_clarification_transition"))
    before, after = transition.get("before", {}), transition.get("after", {})
    expected_after = {
        "amendment_json_sha256": _sha(bound("active_amendment")),
        "amendment_md_sha256": _sha(bound("active_amendment_md")),
        "analysis_contract_sha256": _sha(bound("analysis_contract")),
        "analysis_protocol_sha256": _sha(bound("analysis_protocol")),
        "activation_record_sha256": _sha(bound("amendment_activation_record")),
        "active_protocol_pointer_sha256": _sha(bound("active_protocol_pointer")),
        "clarification_json_sha256": _sha(clarification_path),
        "clarification_markdown_sha256": _sha(
            bound("analysis_missingness_clarification_md")
        ),
    }
    if (
        transition.get("status") != "COMPLETED_PRELOCK"
        or transition.get("clarification_before_formal_lock") is not True
        or transition.get("clarification_before_any_formal_icp") is not True
        or int(transition.get("clarification_at_formal_trial_count", -1)) != 0
        or transition.get("FORMAL_LOCK_ISSUED") is not False
        or transition.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
        or int(transition.get("actual_formal_trials", -1)) != 0
        or before.get("archive_inventory_path")
        != EXPECTED_BINDING_PATHS["analysis_preclarification_history_inventory"]
        or before.get("archive_inventory_sha256") != _sha(inventory_path)
        or any(after.get(key) != value for key, value in expected_after.items())
    ):
        _fail("R1-C1 clarification transition/hash chain differs")


def _verify_plan(root: Path, bindings: Mapping[str, Mapping[str, Any]], snapshots: Mapping[str, Mapping[str, str]], targets: Mapping[tuple[str, str], Mapping[str, str]]) -> None:
    json_path = _resolve(root, bindings["trial_plan_json"]["repository_relative_path"], "plan JSON")
    csv_path = _resolve(root, bindings["trial_plan_csv"]["repository_relative_path"], "plan CSV")
    plan = _json(json_path)
    rows = plan.get("rows")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("track_id") != "ZERO_PERTURBATION_TRACK" or not isinstance(rows, list) or len(rows) != 360:
        _fail("plan schema/track/count differs")
    csv_rows = _csv(csv_path)
    if len(csv_rows) != 360:
        _fail("plan CSV count differs")
    csv_ids = {row.get("trial_id") for row in csv_rows}
    ids: set[str] = set()
    backend_counts: Counter[str] = Counter()
    snapshot_backends: defaultdict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"plan row is not mapping: {index}")
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in ids:
            _fail(f"duplicate trial ID: {trial_id}")
        ids.add(trial_id)
        scene, station, snapshot = str(row.get("scene_id")), str(row.get("station_id")), str(row.get("snapshot_id"))
        if scene not in SCENE_CLASS or scene == "FMB1_W04" or snapshot not in snapshots:
            _fail(f"non-final plan row: {trial_id}")
        if int(row.get("attempt", -1)) != (2 if scene == "FMB1_W02" else 1):
            _fail(f"wrong attempt in plan: {trial_id}")
        backend = str(row.get("backend"))
        if backend not in BACKENDS or row.get("track_id") != "ZERO_PERTURBATION_TRACK":
            _fail(f"backend/track differs: {trial_id}")
        if not _identity(row.get("T0")) or not _identity(row.get("T_reference_nominal")):
            _fail(f"non-identity transform: {trial_id}")
        if float(row.get("translation_perturbation_m", math.nan)) != 0 or float(row.get("rotation_perturbation_deg", math.nan)) != 0:
            _fail(f"capture-radius perturbation entered plan: {trial_id}")
        snap = snapshots[snapshot]
        target = targets[(scene, station)]
        for prefix, manifest_row, sha_name in (
            ("source", snap, "source_npy_sha256"), ("target", target, "target_npy_sha256")
        ):
            path = _resolve(root, row.get(f"{prefix}_reference"), f"{trial_id}/{prefix}")
            if _sha(path) != row.get(f"{prefix}_sha256") or row.get(f"{prefix}_sha256") != manifest_row.get(sha_name):
                _fail(f"{prefix} SHA/manifest binding differs: {trial_id}")
        if row.get("backend_parameter_contract_sha256") != BACKEND_SHA:
            _fail(f"backend contract differs: {trial_id}")
        if row.get("planned_status") != "PLANNED_NOT_AUTHORIZED_NOT_EXECUTED":
            _fail(f"planned status differs: {trial_id}")
        if row.get("active_amendment_sha256") != bindings["active_amendment"]["sha256"] or row.get("analysis_contract_sha256") != bindings["analysis_contract"]["sha256"]:
            _fail(f"protocol hash binding differs: {trial_id}")
        backend_counts[backend] += 1
        snapshot_backends[snapshot].add(backend)
    if ids != csv_ids or backend_counts != Counter({backend: 180 for backend in BACKENDS}):
        _fail("plan JSON/CSV IDs or backend counts differ")
    if len(snapshot_backends) != 180 or any(value != BACKENDS for value in snapshot_backends.values()):
        _fail("each snapshot lacks exactly two backend rows")


def _verify_contracts(root: Path, bindings: Mapping[str, Mapping[str, Any]]) -> None:
    result_schema = _json(_resolve(root, bindings["result_schema"]["repository_relative_path"], "result schema"))
    values = _recursive_values(result_schema)
    required = result_schema.get("required")
    result_fields = {
        "trial_id", "track_id", "scene_id", "geometry_class", "station_id",
        "snapshot_id", "backend", "source_sha256", "target_sha256", "T0",
        "T_est", "Delta_T", "translation_norm_m", "rotation_angle_rad",
        "solver_status", "finite_result", "infrastructure_status",
        "physical_reference_semantics",
        "initial_correspondence_count", "initial_valid_normal_correspondence_count",
        "final_correspondence_count", "final_valid_normal_correspondence_count",
        "correspondence_turnover", "accepted_source_turnover",
        "correspondence_count_change_ratio", "initial_residual_rmse",
        "final_residual_rmse", "residual_rmse_change",
        "median_normal_angle_change_deg", "q95_normal_angle_change_deg",
        "common_association_valid", "common_association_invalid_reason",
        "common_association_invalid_detail",
    }
    if not isinstance(required, list) or not result_fields.issubset(set(required)):
        _fail("result schema required fields are incomplete")
    if PHYSICAL_SEMANTICS not in values or "ZERO_PERTURBATION_TRACK" not in values:
        _fail("result schema physical/track semantics differ")
    if (result_schema.get("x-registration-authority-granted") is not False
            or result_schema.get("x-real-result-count-at-freeze") != 0
            or result_schema.get("x-snapshot-independent-scene-claim-forbidden") is not True
            or result_schema.get("x-common-invalid-reason-retained") is not True
            or set(result_schema.get("x-common-invalid-reason-enum", [])) != {
                "NO_INITIAL_CORRESPONDENCE", "NO_FINAL_CORRESPONDENCE",
                "INSUFFICIENT_VALID_NORMALS", "NONFINITE_COMMON_METRICS", "OTHER",
            }
            or result_schema.get("x-common-status-null-only-without-finite-pose") is not True
            or result_schema.get("x-common-invalid-detail-required-for-other") is not True):
        _fail("future result schema authority/freeze boundary differs")

    analysis = _json(_resolve(root, bindings["analysis_contract"]["repository_relative_path"], "analysis contract"))
    units = analysis.get("experimental_units")
    if not isinstance(units, Mapping):
        _fail("analysis experimental_units object is missing")
    if units.get("highest_independent_unit") != "scene" or units.get("primary_experimental_unit") != "scene":
        _fail("analysis highest unit is not scene")
    if (units.get("hierarchy") != ["scene", "station", "snapshot", "backend"]
            or units.get("snapshots_are_independent_scenes") is not False
            or units.get("snapshot_level_naive_p_value_forbidden") is not True
            or units.get("pooled_180_row_inference_as_independent_samples_forbidden") is not True):
        _fail("analysis permits snapshot pseudo-replication")
    values = _recursive_values(analysis)
    if PHYSICAL_SEMANTICS not in values:
        _fail("analysis physical-reference limitation differs")
    reassociation = analysis.get("reassociation_analysis")
    frozen = {
        "initial_correspondence_count", "initial_valid_normal_correspondence_count",
        "final_correspondence_count", "final_valid_normal_correspondence_count",
        "correspondence_turnover", "accepted_source_turnover",
        "correspondence_count_change_ratio", "initial_residual_rmse",
        "final_residual_rmse", "residual_rmse_change",
        "median_normal_angle_change_deg", "q95_normal_angle_change_deg",
    }
    if (not isinstance(reassociation, Mapping)
            or reassociation.get("implementation_path") != "src/phase_a_harness/common_association_analysis.py"
            or set(reassociation.get("frozen_fields", [])) != frozen
            or reassociation.get("mid360_specific_redefinition_forbidden") is not True):
        _fail("analysis does not bind frozen reassociation implementation")
    missingness = reassociation.get("missingness")
    if (not isinstance(missingness, Mapping)
            or missingness.get("required_result_status_fields") != [
                "common_association_valid", "common_association_invalid_reason",
                "common_association_invalid_detail",
            ]
            or set(missingness.get("common_association_invalid_reason_enum", [])) != {
                "NO_INITIAL_CORRESPONDENCE", "NO_FINAL_CORRESPONDENCE",
                "INSUFFICIENT_VALID_NORMALS", "NONFINITE_COMMON_METRICS", "OTHER",
            }
            or missingness.get("result_schema_runner_and_verifier_must_preserve_status_fields") is not True):
        _fail("analysis common-association missingness status is not frozen")


def _verify_nested_reauthentication(root: Path, bindings: Mapping[str, Mapping[str, Any]]) -> None:
    wrapper = _json(_resolve(
        root, bindings["final_dataset_prelock_reauthentication"]["repository_relative_path"],
        "prelock dataset wrapper",
    ))
    if (wrapper.get("schema") != "mid360_fmb1_zero_perturbation_v1_1_r1_prelock_dataset_binding"
            or wrapper.get("status") != "PASS" or wrapper.get("pass") is not True):
        _fail("prelock dataset wrapper differs")
    for prefix in ("source_report", "source_markdown"):
        source = _resolve(root, wrapper.get(f"{prefix}_path"), f"nested {prefix}")
        if _sha(source) != wrapper.get(f"{prefix}_sha256"):
            _fail(f"nested {prefix} SHA differs")
    detail = _json(_resolve(root, wrapper["source_report_path"], "detailed prelock reauthentication"))
    if (detail.get("status") != "PASS" or detail.get("pass") is not True
            or [detail.get(name) for name in (
                "final_scene_count", "final_station_count", "final_target_count",
                "final_snapshot_count",
            )] != [6, 18, 18, 180]
            or detail.get("protected_assets", {}).get("unchanged") is not True):
        _fail("detailed prelock reauthentication differs")
    state = detail.get("formal_execution_state")
    if not isinstance(state, Mapping):
        _fail("detailed prelock execution state is missing")
    for key in ("actual_open3d_trials", "actual_pcl_trials", "actual_formal_trials",
                "registration_execution_count"):
        if int(state.get(key, -1)) != 0:
            _fail(f"detailed prelock counter is nonzero: {key}")
    if state.get("FORMAL_ICP_UNLOCKED") is not False or state.get("FORMAL_REGISTRATION_AUTHORIZED") is not False:
        _fail("detailed prelock report contains authority")


def _verify_upstream_reports(root: Path, bindings: Mapping[str, Mapping[str, Any]]) -> None:
    plan_report = _json(_resolve(
        root, bindings["trial_plan_independent_verification"]["repository_relative_path"],
        "trial-plan independent report",
    ))
    expected = {"status": "PASS", "payload_byte_hashes_verified": True,
                "single_authoritative_byte_source": True, "scene_count": 6,
                "station_count": 18, "snapshot_count": 180, "total_trial_count": 360,
                "open3d_trial_count": 180, "pcl_trial_count": 180,
                "identity_t0_count": 360, "w04_trial_count": 0,
                "old_w02_attempt1_trial_count": 0, "w02_attempt2_trial_count": 60,
                "registration_backend_call_count": 0,
                "FORMAL_REGISTRATION_AUTHORIZED": False, "actual_formal_trials": 0}
    if any(plan_report.get(key) != value for key, value in expected.items()):
        _fail("trial-plan independent report differs")
    for report_key, binding_name in (
        ("trial_plan_json_sha256", "trial_plan_json"),
        ("trial_plan_csv_sha256", "trial_plan_csv"),
        ("result_schema_sha256", "result_schema"),
    ):
        path = _resolve(root, bindings[binding_name]["repository_relative_path"], binding_name)
        if plan_report.get(report_key) != _sha(path):
            _fail(f"trial-plan report hash differs: {report_key}")
    old_report = _json(_resolve(
        root, bindings["protocol_transition_independent_verification"]["repository_relative_path"],
        "protocol transition report",
    ))
    history = _json(_resolve(
        root, bindings["analysis_preclarification_history_inventory"]["repository_relative_path"],
        "preclarification history inventory",
    ))
    archived = {row["name"]: row["sha256"] for row in history["files"]}
    old_hashes = {
        "amendment_json_sha256": archived["zero_perturbation_mainline_v1_1_r1.json"],
        "amendment_md_sha256": archived["zero_perturbation_mainline_v1_1_r1.md"],
        "analysis_contract_sha256": archived["zero_perturbation_analysis_contract_v1_1_r1.json"],
        "analysis_protocol_sha256": archived["zero_perturbation_analysis_protocol_v1_1_r1.md"],
    }
    if (old_report.get("pass") is not True
            or old_report.get("verification_status") != "PASS"
            or old_report.get("activation_effective") is not True
            or old_report.get("FORMAL_LOCK_ISSUED") is not False
            or old_report.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
            or old_report.get("actual_formal_trials") != 0
            or old_report.get("backend_calls") != 0
            or old_report.get("backend_modules_imported") != 0
            or old_report.get("active_hashes") != old_hashes
            or old_report.get("activation_record_sha256")
            != archived["amendment_activation_record_v1_1_r1.json"]
            or old_report.get("active_protocol_pointer_sha256")
            != archived["ACTIVE_PROTOCOL.json"]):
        _fail("historical initial-active transition report differs")
    c1 = _json(_resolve(
        root, bindings["protocol_c1_missingness_independent_verification"]["repository_relative_path"],
        "R1-C1 independent report",
    ))
    def bound_sha(name: str) -> str:
        return _sha(_resolve(root, bindings[name]["repository_relative_path"], name))
    current_hashes = {
        "amendment_json_sha256": bound_sha("active_amendment"),
        "amendment_md_sha256": bound_sha("active_amendment_md"),
        "analysis_contract_sha256": bound_sha("analysis_contract"),
        "analysis_protocol_sha256": bound_sha("analysis_protocol"),
    }
    expected_artifacts = {
        "execution_verifier_sha256": _sha(_resolve(
            root, "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_verify.py",
            "R1 execution verifier",
        )),
        "result_schema_sha256": bound_sha("result_schema"),
        "runner_sha256": bound_sha("execution_runner"),
        "result_validator_sha256": bound_sha("execution_result_validator"),
        "experiments_package_init_sha256": bound_sha("execution_experiments_package_init"),
        "mid360_formal_batch1_package_init_sha256": bound_sha("execution_mid360_formal_batch1_package_init"),
        "phase_a_harness_package_init_sha256": bound_sha("execution_phase_a_harness_package_init"),
    }
    if (c1.get("pass") is not True or c1.get("verification_status") != "PASS"
            or c1.get("phase") != "ACTIVE_R1_C1_PRE_LOCK"
            or c1.get("activation_effective") is not True
            or c1.get("common_association_status_fields_verified") is not True
            or c1.get("FORMAL_LOCK_ISSUED") is not False
            or c1.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
            or c1.get("actual_formal_trials") != 0
            or c1.get("backend_calls") != 0
            or c1.get("backend_modules_imported") != 0
            or c1.get("current_active_hashes") != current_hashes
            or c1.get("initial_active_hashes_preserved") != old_hashes
            or c1.get("activation_record_sha256") != bound_sha("amendment_activation_record")
            or c1.get("active_protocol_pointer_sha256") != bound_sha("active_protocol_pointer")
            or c1.get("clarification_json_sha256") != bound_sha("analysis_missingness_clarification")
            or c1.get("clarification_markdown_sha256") != bound_sha("analysis_missingness_clarification_md")
            or c1.get("clarification_transition_sha256") != bound_sha("analysis_missingness_clarification_transition")
            or c1.get("prior_active_inventory_sha256") != bound_sha("analysis_preclarification_history_inventory")
            or c1.get("common_association_status_artifacts") != expected_artifacts):
        _fail("R1-C1 independent transition/status report differs")

    review = _json(_resolve(
        root, bindings["activation_review"]["repository_relative_path"],
        "R1-C1 activation review",
    ))
    review_verifier = review.get("missingness_clarification_independent_verification")
    if not isinstance(review_verifier, Mapping):
        _fail("activation review lacks C1 verifier binding")
    if (review.get("schema") != "mid360_fmb1_zero_perturbation_v1_1_r1_activation_review"
            or review.get("review_status") != "PASS_ACTIVE_R1_C1"
            or review.get("activation_effective") is not True
            or review.get("missingness_clarification_verifier_pass") is not True
            or review.get("FORMAL_LOCK_ISSUED") is not False
            or review.get("FORMAL_REGISTRATION_AUTHORIZED") is not False
            or review.get("actual_formal_trials") != 0
            or review_verifier.get("status") != "PASS"
            or review_verifier.get("sha256")
            != bound_sha("protocol_c1_missingness_independent_verification")):
        _fail("activation review does not bind the zero-trial R1-C1 PASS")
    transition = _json(_resolve(
        root,
        bindings["analysis_missingness_clarification_transition"]["repository_relative_path"],
        "R1-C1 transition",
    ))
    try:
        review_time = datetime.fromisoformat(str(review.get("reviewed_at_utc")).replace("Z", "+00:00"))
        transition_time = datetime.fromisoformat(
            str(transition["transitioned_at_utc"]).replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail(f"activation review/transition timestamp is invalid: {error}")
    if (review_time.tzinfo is None or transition_time.tzinfo is None
            or review_time < transition_time):
        _fail("activation review predates the R1-C1 transition")


def _verify_execution_boundary(root: Path) -> None:
    """Independently audit files excluded from the preregistration firewall."""

    for relative in (
        "experiments/__init__.py",
        "experiments/mid360_formal_batch1/__init__.py",
        "src/phase_a_harness/__init__.py",
    ):
        initializer = _resolve(root, relative, f"execution package initializer {relative}")
        init_tree = ast.parse(initializer.read_text(encoding="utf-8"), filename=str(initializer))
        if any(
            isinstance(node, (ast.Import, ast.ImportFrom, ast.Call))
            for node in ast.walk(init_tree)
        ):
            _fail(f"execution package initializer has eager import/call: {relative}")

    runner = _resolve(
        root, "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_runner.py",
        "future formal runner",
    )
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(runner))
    forbidden = ("open3d", "pcl_backend", "open3d_backend", "debug_registration",
                 "common_association_analysis")
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(token in name for name in names for token in forbidden):
            _fail(f"runner has eager backend import: {names}")
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = {"_load_authorized_execution_adapter", "_execute_authorized_run",
                "preflight_or_dry_run"}
    if not required.issubset(functions):
        _fail("runner authorization/load boundary functions differ")
    dynamic_calls: list[tuple[str, ast.Call]] = []
    loader_calls: list[str] = []
    for function_name, function in functions.items():
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                dynamic_calls.append((function_name, node))
            if isinstance(node.func, ast.Name) and node.func.id == "_load_authorized_execution_adapter":
                loader_calls.append(function_name)
    if not dynamic_calls or any(name not in {"_load_authorized_execution_adapter", "_load_result_validator"}
                                for name, _ in dynamic_calls):
        _fail("dynamic import exists outside the authorized loader boundary")
    if loader_calls != ["_execute_authorized_run"]:
        _fail("backend loader call graph differs")
    preflight_text = ast.get_source_segment(source, functions["preflight_or_dry_run"]) or ""
    if ("if action == \"execute\"" not in preflight_text
            or "if not valid" not in preflight_text
            or "_execute_authorized_run(" not in preflight_text):
        _fail("execute call is not structurally guarded by valid authorization")
    expected_modules = {
        "numpy", "phase_a_harness.open3d_backend", "phase_a_harness.pcl_backend",
        "phase_a_harness.common_association_analysis", "phase_a_harness.rotation_metrics",
    }
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name)
                           and target.id == "AUTHORIZED_EXECUTION_MODULES"
                           for target in node.targets)]
    if len(assignments) != 1:
        _fail("authorized execution module allowlist differs")
    try:
        modules = set(ast.literal_eval(assignments[0].value))
    except (ValueError, TypeError):
        _fail("authorized execution module allowlist is not literal")
    if modules != expected_modules:
        _fail("authorized execution module allowlist changed")
    assets = _resolve(root, "experiments/mid360_formal_batch1/zero_perturbation_r1_trial_assets.py",
                      "future result-schema producer")
    asset_tree = ast.parse(assets.read_text(encoding="utf-8"), filename=str(assets))
    for node in asset_tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(token in name for name in names for token in forbidden):
            _fail("result-schema producer imports a backend")


def _verify_execution_commit(root: Path, commit: str,
                             bindings: Mapping[str, Mapping[str, Any]]) -> None:
    if not EXECUTION_BINDINGS.issubset(bindings):
        _fail(f"lock lacks execution-code bindings: {sorted(EXECUTION_BINDINGS-set(bindings))}")
    try:
        subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for name in EXECUTION_BINDINGS:
            relative = str(bindings[name]["repository_relative_path"])
            recorded = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root,
                                      check=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE).stdout
            path = _resolve(root, relative, f"execution binding {name}")
            if hashlib.sha256(recorded).hexdigest() != _sha(path):
                _fail(f"execution binding differs from locked commit: {name}")
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", "replace")[-500:]
        _fail(f"execution-code commit cannot be authenticated: {detail}")


def _verify_no_results(runtime_root: Path) -> None:
    if not runtime_root.exists():
        return
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        _fail("authoritative runtime is a symlink or non-directory")
    lifecycle = sorted(
        path.relative_to(runtime_root).as_posix()
        for path in runtime_root.rglob("*") if path.is_file() or path.is_symlink()
    )
    if lifecycle:
        _fail(f"formal execution lifecycle files already exist: {lifecycle[:3]}")


def verify_r1_lock(
    repository: Path,
    lock_dir: Path,
    *,
    expected_execution_code_commit: str | None = None,
    runtime_root: Path | None = None,
    remeasure_environment: bool = True,
    remeasure_execution_commit: bool = True,
) -> dict[str, Any]:
    root = Path(repository).resolve(strict=True)
    lexical_lock = Path(lock_dir)
    if not lexical_lock.is_absolute():
        lexical_lock = root / lexical_lock
    try:
        relative_lock = lexical_lock.relative_to(root)
    except ValueError:
        _fail("lock directory is lexically outside repository")
    cursor = root
    for part in relative_lock.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"lock directory uses symlink component: {cursor}")
    locked = lexical_lock.resolve(strict=True)
    if not locked.is_dir():
        _fail("lock directory is not a directory")
    checksum_count = _verify_sums(locked)
    lock_path = locked / LOCK_FILE
    lock = _json(lock_path)
    if lock.get("schema") != LOCK_SCHEMA or lock.get("FORMAL_LOCK_ISSUED") is not True:
        _fail("lock schema/status differs")
    locked_commit = str(lock.get("execution_code_commit"))
    expected_commit = expected_execution_code_commit or locked_commit
    if COMMIT_RE.fullmatch(expected_commit) is None or locked_commit != expected_commit:
        _fail("execution-code commit binding differs")
    if lock.get("FORMAL_ICP_UNLOCKED") is not False or lock.get("FORMAL_REGISTRATION_AUTHORIZED") is not False:
        _fail("lock contains registration authority")
    for key in ("actual_open3d_trials", "actual_pcl_trials", "actual_formal_trials", "registration_execution_count"):
        if int(lock.get(key, -1)) != 0:
            _fail(f"lock counter nonzero: {key}")
    bindings = lock.get("bindings")
    if not isinstance(bindings, Mapping):
        _fail("lock bindings missing")
    if set(bindings) != set(EXPECTED_BINDING_PATHS):
        _fail("lock binding names differ from canonical R1 inventory")
    inventory = _csv(locked / "lock_inventory.csv")
    inventory_map = {row["binding_name"]: row for row in inventory}
    if set(inventory_map) != set(bindings):
        _fail("lock inventory/binding names differ")
    canonical_inventory = [
        {
            "binding_name": row["binding_name"],
            "repository_relative_path": row["repository_relative_path"],
            "sha256": row["sha256"],
            "bytes": int(row["bytes"]),
        }
        for row in sorted(inventory, key=lambda value: value["binding_name"])
    ]
    if hashlib.sha256(_canonical(canonical_inventory)).hexdigest() != lock.get("binding_inventory_sha256"):
        _fail("canonical binding inventory fingerprint differs")
    for name, row in bindings.items():
        if not isinstance(row, Mapping):
            _fail(f"binding malformed: {name}")
        if row.get("repository_relative_path") != EXPECTED_BINDING_PATHS[name]:
            _fail(f"binding path is not canonical: {name}")
        inv = inventory_map[name]
        if row.get("repository_relative_path") != inv["repository_relative_path"] or row.get("sha256") != inv["sha256"] or int(row.get("bytes", -1)) != int(inv["bytes"]):
            _fail(f"binding/inventory differs: {name}")
        path = _resolve(root, row.get("repository_relative_path"), f"binding {name}")
        if _sha(path) != row.get("sha256") or path.stat().st_size != int(row.get("bytes", -1)):
            _fail(f"bound file changed: {name}")
    lock_sha_line = (locked / "formal_batch1_zero_perturbation_lock_v1_1.sha256").read_text().strip()
    if lock_sha_line != f"{_sha(lock_path)}  {LOCK_FILE}":
        _fail("lock .sha256 sidecar differs")
    fingerprint = _json(locked / "lock_fingerprint.json")
    material = {
        "lock_file_sha256": _sha(lock_path),
        "lock_inventory_file_sha256": _sha(locked / "lock_inventory.csv"),
        "execution_code_commit": expected_commit,
    }
    expected_fingerprint = hashlib.sha256(_canonical(material)).hexdigest()
    if fingerprint.get("lock_fingerprint") != expected_fingerprint or any(fingerprint.get(key) != value for key, value in material.items()):
        _fail("lock fingerprint differs")

    environment = _json(_resolve(root, bindings["environment_manifest"]["repository_relative_path"], "environment"))
    _verify_environment(environment, root, remeasure=remeasure_environment)
    snapshots, targets = _verify_final_dataset(root, bindings)
    _verify_proposal_correction_history(root, bindings)
    _verify_missingness_clarification(root, bindings)
    _verify_activation(root, bindings)
    _verify_plan(root, bindings, snapshots, targets)
    _verify_contracts(root, bindings)
    _verify_nested_reauthentication(root, bindings)
    _verify_upstream_reports(root, bindings)
    _verify_execution_boundary(root)
    if remeasure_execution_commit:
        _verify_execution_commit(root, expected_commit, bindings)
    no_icp = _json(locked / "NO_ICP_ATTESTATION.json")
    if (no_icp.get("schema") != "mid360_fmb1_zero_perturbation_r1_no_icp_attestation_v1"
            or no_icp.get("status") != "PASS" or no_icp.get("pass") is not True
            or no_icp.get("future_execution_boundary_verified") is not True
            or int(no_icp.get("execution_lifecycle_file_count", -1)) != 0
            or int(no_icp.get("real_trial_result_file_count", -1)) != 0):
        _fail("R1-scoped NO_ICP attestation differs")
    for key in ("open3d_registration_call_count", "pcl_cli_invocation_count", "other_registration_process_count", "formal_trial_count", "actual_formal_trials"):
        if int(no_icp.get(key, 0)) != 0:
            _fail(f"NO_ICP counter nonzero: {key}")
    if no_icp.get("FORMAL_REGISTRATION_AUTHORIZED") is not False:
        _fail("NO_ICP authorization true")
    authority_hashes = no_icp.get("authority_file_sha256")
    if not isinstance(authority_hashes, Mapping):
        _fail("R1 NO_ICP authority hashes missing")
    for relative, digest in authority_hashes.items():
        if _sha(_resolve(root, relative, f"NO_ICP authority {relative}")) != digest:
            _fail(f"R1 NO_ICP authority hash differs: {relative}")
    if lock.get("authoritative_runtime_root") != AUTHORITATIVE_RUNTIME_ROOT:
        _fail("lock authoritative runtime is not canonical")
    authoritative_runtime = root / AUTHORITATIVE_RUNTIME_ROOT
    if no_icp.get("authoritative_runtime_root") != lock.get("authoritative_runtime_root"):
        _fail("NO_ICP/lock authoritative runtime differs")
    if runtime_root is not None and Path(runtime_root) != authoritative_runtime:
        _fail("requested runtime root differs from lock")
    authorization = root / DEFAULT_AUTHORIZATION_PATH
    if authorization.exists() or authorization.is_symlink():
        _fail("separate formal-registration authorization already exists")
    _verify_no_results(authoritative_runtime)
    return {
        "schema": "mid360_fmb1_zero_perturbation_lock_independent_verification_v1_1_r1",
        "status": "PASS",
        "pass": True,
        "failure_count": 0,
        "lock_fingerprint": expected_fingerprint,
        "scene_count": 6,
        "station_count": 18,
        "snapshot_count": 180,
        "planned_open3d_trials": 180,
        "planned_pcl_trials": 180,
        "planned_total_trials": 360,
        "checksum_entry_count": checksum_count,
        "lock_core_checksum_entry_count": checksum_count,
        "release_checksum_status": "DERIVED_AFTER_INDEPENDENT_REPORT_NOT_SELF_VERIFIED",
        "FORMAL_LOCK_ISSUED": True,
        "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION": True,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
        "registration_backend_imports_or_calls": 0,
    }
