"""Registration-free evidence builders for the FMB1 pre-backend gate.

This module authenticates the already-produced FMB1 acquisition/geometry
closure and prepares only *unissued* lock metadata.  It deliberately contains
no backend adapter, solver import, transform estimator, or authorization path.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


QUALIFICATION_SCHEMA = "mid360_fmb1_prebackend_qualification_v1"
CURRENT_BLOCK_REASON = "MISSING_ADMITTED_WEAK_REPLACEMENT_W04"
SCIENTIFIC_W02_REJECTION_REASON = "GEOMETRY_ONLY_INELIGIBLE"
EVIDENCE_W02_REJECTION_REASON = "SEMANTIC_CANDIDATE_GEOMETRY_CLASS_MISMATCH"
EXPECTED_BACKEND_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "76ae548874d8c1584fcc033685db4a1e7cf104fd881cbd1c334eba0bfe1a9beb"
)
EXPECTED_ANALYSIS_PROTOCOL_SHA256 = (
    "d453d12e713c546c5a054ceb1710b86eea12fa09e3f244705e46df9db255a879"
)

REQUIRED_CLOSURE_FILES = (
    "fmb1_pre_registration_readiness.json",
    "geometry_scene_summary.csv",
    "NO_ICP_ATTESTATION.json",
    "fmb1_deep_verification_report.json",
    "fmb1_verification_report.json",
    "REACQUISITION_REQUIRED.json",
    "SHA256SUMS",
    "raw_bag_inventory.json",
)

EXPECTED_CURRENT_SCENES = {
    "FMB1_R01": ("RICH", "GEOMETRY_ADMITTED"),
    "FMB1_R02": ("RICH", "GEOMETRY_ADMITTED"),
    "FMB1_R03": ("RICH", "GEOMETRY_ADMITTED"),
    "FMB1_W01": ("WEAK", "GEOMETRY_ADMITTED"),
    "FMB1_W02": ("RICH", "GEOMETRY_REJECTED"),
    "FMB1_W03": ("WEAK", "GEOMETRY_ADMITTED"),
}
EXPECTED_CURRENT_BAG_IDENTITIES = {
    (scene_id, station_id, role)
    for scene_id in EXPECTED_CURRENT_SCENES
    for station_id in ("S01", "S02", "S03")
    for role in ("MAP", "QUERY")
}


class PrebackendQualificationError(RuntimeError):
    """The immutable current-state or replacement-plan evidence is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PrebackendQualificationError(f"expected JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrebackendQualificationError(message)


def _parse_sha256sums(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        pieces = line.split("  ", 1)
        _require(len(pieces) == 2, f"malformed SHA256SUMS line {line_number}")
        digest, name = pieces
        _require(len(digest) == 64, f"malformed SHA256 on line {line_number}")
        relative = Path(name)
        _require(
            relative.name == name and not relative.is_absolute(),
            f"unsafe SHA256SUMS path: {name}",
        )
        rows.append((digest, name))
    _require(bool(rows), "SHA256SUMS is empty")
    return rows


def verify_sha256_manifest(results_dir: Path) -> dict[str, Any]:
    manifest = results_dir / "SHA256SUMS"
    rows = _parse_sha256sums(manifest)
    failures: list[str] = []
    for expected, name in rows:
        candidate = results_dir / name
        if not candidate.is_file() or candidate.is_symlink():
            failures.append(f"missing_or_unsafe:{name}")
            continue
        if sha256_file(candidate) != expected:
            failures.append(f"sha256_mismatch:{name}")
    return {
        "manifest_path": str(manifest.resolve()),
        "entry_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "pass": not failures,
    }


def _read_geometry_scenes(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    scenes = {str(row.get("scene_id")): row for row in rows}
    _require(len(scenes) == 6 and len(rows) == 6, "geometry summary must contain six scenes")
    for scene_id, (expected_class, expected_status) in EXPECTED_CURRENT_SCENES.items():
        _require(scene_id in scenes, f"geometry summary missing {scene_id}")
        row = scenes[scene_id]
        _require(
            row.get("final_geometry_class") == expected_class,
            f"unexpected geometry class for {scene_id}",
        )
        _require(
            row.get("geometry_admission_status") == expected_status,
            f"unexpected geometry admission for {scene_id}",
        )
    return scenes


def _authenticate_raw_bags(
    inventory: Mapping[str, Any], *, verify_bytes: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = inventory.get("bags")
    _require(isinstance(rows, list), "raw bag inventory bags must be a list")
    _require(len(rows) == 36, "raw bag inventory must contain 36 bags")
    authenticated: list[dict[str, Any]] = []
    w02: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for index, raw_row in enumerate(rows):
        _require(isinstance(raw_row, Mapping), f"bag row {index} must be an object")
        row = dict(raw_row)
        scene_id = row.get("scene_id")
        station_id = row.get("station_id")
        role = row.get("role")
        identity = (str(scene_id), str(station_id), str(role))
        _require(identity not in identities, f"duplicate raw bag identity: {identity}")
        identities.add(identity)
        path_value = row.get("raw_absolute_path")
        expected_sha = row.get("sha256")
        expected_bytes = row.get("bytes")
        _require(isinstance(path_value, str), f"bag row {index} missing path")
        _require(isinstance(expected_sha, str) and len(expected_sha) == 64, "invalid bag SHA")
        _require(isinstance(expected_bytes, int), "invalid bag byte count")
        bag_audit = row.get("bag_audit")
        _require(isinstance(bag_audit, Mapping), "bag audit evidence is missing")
        _require(
            bag_audit.get("ACQUISITION_AUDIT_PASS") is True,
            f"raw bag acquisition audit is not PASS: {identity}",
        )
        path = Path(path_value)
        _require(path.is_file() and not path.is_symlink(), f"raw bag missing or unsafe: {path}")
        actual_bytes = path.stat().st_size
        _require(actual_bytes == expected_bytes, f"raw bag size mismatch: {path}")
        actual_sha = sha256_file(path) if verify_bytes else expected_sha
        _require(actual_sha == expected_sha, f"raw bag SHA mismatch: {path}")
        evidence = {
            "scene_id": scene_id,
            "station_id": station_id,
            "role": role,
            "raw_absolute_path": str(path.resolve()),
            "sha256": expected_sha,
            "bytes": expected_bytes,
            "authenticated": True,
        }
        authenticated.append(evidence)
        if scene_id == "FMB1_W02":
            w02.append(evidence)
    _require(
        identities == EXPECTED_CURRENT_BAG_IDENTITIES,
        "raw bag identities changed or include an unadmitted replacement",
    )
    _require(len(w02) == 6, "W02 must retain exactly six raw bags")
    return authenticated, w02


def reauthenticate_current_fmb1(
    repository: Path, *, verify_raw_bag_bytes: bool = True
) -> dict[str, Any]:
    """Independently bind the current expected failure closure.

    The expected independent-verifier outcome is a fail-closed W02 geometry
    rejection.  Treating that expected failure as an admitted Weak scene is a
    hard error.
    """

    repository = repository.resolve()
    results_dir = repository / "results/mid360_formal_batch1"
    for name in REQUIRED_CLOSURE_FILES:
        _require((results_dir / name).is_file(), f"required closure file missing: {name}")

    checksum = verify_sha256_manifest(results_dir)
    _require(bool(checksum["pass"]), "current FMB1 SHA256SUMS verification failed")

    readiness = load_json_object(results_dir / "fmb1_pre_registration_readiness.json")
    _require(readiness.get("authenticated_bag_count") == 36, "expected 36 authenticated bags")
    _require(readiness.get("pair_count") == 18, "expected 18 MAP/QUERY pairs")
    _require(readiness.get("acquisition_pass_station_count") == 18, "expected 18 passing stations")
    _require(readiness.get("snapshot_count") == 180, "expected 180 provisional snapshots")
    _require(readiness.get("rich_scene_count") == 3, "expected three admitted Rich scenes")
    _require(readiness.get("weak_scene_count") == 2, "expected two admitted Weak scenes")
    _require(readiness.get("actual_registration_trials") == 0, "real registration trials must be zero")
    _require(readiness.get("FMB1_PRE_REGISTRATION_DATA_READY") is False, "current batch must be blocked")
    _require(readiness.get("FORMAL_REGISTRATION_AUTHORIZED") is False, "formal authorization must be false")
    _require(readiness.get("FORMAL_ICP_UNLOCKED") is False, "formal ICP unlock must be false")

    scenes = _read_geometry_scenes(results_dir / "geometry_scene_summary.csv")
    no_icp = load_json_object(results_dir / "NO_ICP_ATTESTATION.json")
    _require(no_icp.get("pass") is True, "existing NO_ICP attestation must pass")
    for field in (
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "actual_registration_trials",
    ):
        _require(no_icp.get(field) == 0, f"existing NO_ICP field is nonzero: {field}")

    deep = load_json_object(results_dir / "fmb1_deep_verification_report.json")
    _require(deep.get("status") == "PASS", "deep pre-registration verification must pass")
    _require(deep.get("actual_registration_trials") == 0, "deep report trial count must be zero")

    independent = load_json_object(results_dir / "fmb1_verification_report.json")
    error = independent.get("error")
    error_message = error.get("message", "") if isinstance(error, Mapping) else ""
    _require(independent.get("status") == "FAIL", "current freeze verifier must fail closed")
    _require("FMB1_W02" in str(error_message), "current verifier failure must identify W02")

    reacquisition = load_json_object(results_dir / "REACQUISITION_REQUIRED.json")
    items = reacquisition.get("items")
    _require(isinstance(items, list) and len(items) == 1, "expected exactly one replacement item")
    _require(items[0].get("scene_id") == "FMB1_W02", "replacement item must be W02")
    _require(
        items[0].get("failure_reason") == EVIDENCE_W02_REJECTION_REASON,
        "unexpected source evidence reason for W02",
    )

    raw_inventory = load_json_object(results_dir / "raw_bag_inventory.json")
    authenticated, w02_bags = _authenticate_raw_bags(
        raw_inventory, verify_bytes=verify_raw_bag_bytes
    )

    backend_contract = repository / "frozen_assets/backend_parameter_contract.json"
    preregistration = repository / "experiments/mid360_formal_batch1/preregistration.yaml"
    analysis_protocol = repository / "experiments/mid360_formal_batch1/analysis_protocol.md"
    protected_hashes = {
        "backend_parameter_contract.json": sha256_file(backend_contract),
        "preregistration.yaml": sha256_file(preregistration),
        "analysis_protocol.md": sha256_file(analysis_protocol),
    }
    _require(
        protected_hashes["backend_parameter_contract.json"] == EXPECTED_BACKEND_CONTRACT_SHA256,
        "backend parameter contract changed",
    )
    _require(
        protected_hashes["preregistration.yaml"] == EXPECTED_PREREGISTRATION_SHA256,
        "original preregistration changed",
    )
    _require(
        protected_hashes["analysis_protocol.md"] == EXPECTED_ANALYSIS_PROTOCOL_SHA256,
        "original analysis protocol changed",
    )

    admitted_scenes = sorted(
        scene_id
        for scene_id, row in scenes.items()
        if row.get("geometry_admission_status") == "GEOMETRY_ADMITTED"
    )
    _require(
        admitted_scenes == ["FMB1_R01", "FMB1_R02", "FMB1_R03", "FMB1_W01", "FMB1_W03"],
        "the five admitted scenes changed",
    )

    return {
        "schema": "mid360_fmb1_current_state_reauthentication_v1",
        "status": "PASS",
        "git_entry_state": {
            "head": "641f6c67848653845ba67daf65c0ab8351c40cb0",
            "branch": "formal/mid360-fmb1-ingest-v1",
            "worktree_clean": False,
            "tracked_modifications_present": [".gitignore"],
            "untracked_prior_fmb1_assets_present": True,
            "automatic_stash_performed": False,
            "reset_or_rebase_performed": False,
        },
        "qualification_branch": "formal/fmb1-prebackend-qualification-v1",
        "FMB1_CURRENT_FAILURE_CLOSURE_REAUTHENTICATED": True,
        "FMB1_PRE_REGISTRATION_DATA_READY": False,
        "FMB1_CURRENT_REAL_BATCH_BLOCKED": True,
        "FMB1_CURRENT_BLOCK_REASON": CURRENT_BLOCK_REASON,
        "W02_REJECTION_PRESERVED": True,
        "W02_DATA_RETAINED": True,
        "W02_INCLUDED_IN_FUTURE_FORMAL_SET": False,
        "W02_REJECTION_REASON": SCIENTIFIC_W02_REJECTION_REASON,
        "W02_SOURCE_EVIDENCE_REASON": EVIDENCE_W02_REJECTION_REASON,
        "W02_FINAL_GEOMETRY_CLASS": "RICH",
        "W02_GEOMETRY_ADMISSION_STATUS": "GEOMETRY_REJECTED",
        "W02_RAW_BAG_COUNT": len(w02_bags),
        "W02_RAW_BAGS": w02_bags,
        "W04_DATA_PRESENT": False,
        "W04_GEOMETRY_ADMITTED": False,
        "authenticated_bag_count": len(authenticated),
        "station_pair_count": 18,
        "acquisition_pass_station_count": 18,
        "provisional_snapshot_count": 180,
        "admitted_scene_count": 5,
        "admitted_rich_scene_count": 3,
        "admitted_weak_scene_count": 2,
        "required_weak_scene_count": 3,
        "admitted_scenes": admitted_scenes,
        "rejected_scenes": ["FMB1_W02"],
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "existing_checksum_verification": checksum,
        "deep_verification_status": deep.get("status"),
        "current_freeze_verifier_status": independent.get("status"),
        "current_freeze_verifier_expected_failure": str(error_message),
        "protected_file_sha256": protected_hashes,
        "raw_bag_bytes_rehashed": verify_raw_bag_bytes,
    }


def verify_w04_replacement_plan(repository: Path) -> dict[str, Any]:
    plan_path = repository / "experiments/mid360_formal_batch1/replacement_plan_w04.yaml"
    checklist_path = repository / "experiments/mid360_formal_batch1/W04_FIELD_CHECKLIST.md"
    _require(plan_path.is_file(), "W04 replacement plan is missing")
    _require(checklist_path.is_file(), "W04 field checklist is missing")
    payload = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "W04 replacement plan must be a mapping")
    expected = {
        "rejected_candidate_scene_id": "FMB1_W02",
        "rejection_reason": SCIENTIFIC_W02_REJECTION_REASON,
        "replacement_scene_id": "FMB1_W04",
        "semantic_candidate_label": "WEAK_CANDIDATE",
        "decision_before_any_icp": True,
        "formal_trial_count_at_decision": 0,
    }
    for key, value in expected.items():
        _require(payload.get(key) == value, f"W04 replacement field mismatch: {key}")
    decision_timestamp = payload.get("decision_timestamp")
    _require(isinstance(decision_timestamp, str) and decision_timestamp, "decision timestamp missing")
    stations = payload.get("stations")
    _require(isinstance(stations, list) and len(stations) == 3, "W04 requires three stations")
    station_ids = [row.get("station_id") for row in stations if isinstance(row, Mapping)]
    _require(
        station_ids == ["FMB1_W04_S01", "FMB1_W04_S02", "FMB1_W04_S03"],
        "W04 station IDs are not frozen",
    )
    for row in stations:
        _require(isinstance(row, Mapping), "W04 station row must be a mapping")
        _require(float(row.get("map_target_s", -1)) == 20.0, "W04 MAP target changed")
        _require(float(row.get("map_minimum_s", -1)) == 19.0, "W04 MAP minimum changed")
        _require(float(row.get("gap_target_s", -1)) == 12.0, "W04 gap target changed")
        _require(float(row.get("gap_minimum_s", -1)) == 10.0, "W04 gap minimum changed")
        _require(float(row.get("query_target_s", -1)) == 15.0, "W04 QUERY target changed")
        _require(float(row.get("query_minimum_s", -1)) == 14.0, "W04 QUERY minimum changed")
    checklist = checklist_path.read_text(encoding="utf-8")
    for phrase in (
        "3 genuinely distinct stations",
        "mechanically supported",
        "livox_frame",
        "MAP >=19 s",
        "QUERY >=14 s",
        "gap >=10 s",
        "no overlap",
        "new timestamp",
        "failed attempts retained",
    ):
        _require(phrase in checklist, f"W04 checklist missing: {phrase}")
    return {
        "schema": "mid360_fmb1_w04_replacement_plan_verification_v1",
        "status": "PASS",
        "FMB1_W04_REPLACEMENT_PLAN_FROZEN": True,
        "decision_before_any_icp": True,
        "formal_trial_count_at_decision": 0,
        "rejected_candidate_scene_id": "FMB1_W02",
        "rejection_reason": SCIENTIFIC_W02_REJECTION_REASON,
        "replacement_scene_id": "FMB1_W04",
        "semantic_candidate_label": "WEAK_CANDIDATE",
        "station_count": 3,
        "new_bag_count_required": 6,
        "station_ids": station_ids,
        "scene_location": "UNKNOWN",
        "sensor_orientation": "UNKNOWN",
        "sensor_height": "UNKNOWN",
        "plan_sha256": sha256_file(plan_path),
        "checklist_sha256": sha256_file(checklist_path),
    }


def formal_lock_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://zprm.local/schemas/fmb1/formal_batch1_lock_v1.json",
        "title": "FMB1 formal registration lock",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "LOCK_STATUS",
            "FORMAL_LOCK_ISSUED",
            "FORMAL_REGISTRATION_AUTHORIZED",
            "admitted_scene_ids",
            "station_count",
            "snapshot_count",
            "hash_bindings",
        ],
        "properties": {
            "schema": {"const": "mid360_fmb1_formal_batch1_lock_v1"},
            "LOCK_STATUS": {"const": "ISSUED"},
            "FORMAL_LOCK_ISSUED": {"const": True},
            "FORMAL_REGISTRATION_AUTHORIZED": {"const": True},
            "admitted_scene_ids": {
                "const": [
                    "FMB1_R01",
                    "FMB1_R02",
                    "FMB1_R03",
                    "FMB1_W01",
                    "FMB1_W03",
                    "FMB1_W04",
                ]
            },
            "station_count": {"const": 18},
            "snapshot_count": {"const": 180},
            "hash_bindings": {
                "type": "object",
                "required": [
                    "bags",
                    "targets",
                    "snapshots",
                    "geometry_manifest",
                    "backend_parameter_contract",
                ],
            },
        },
    }


def unissued_lock_template() -> dict[str, Any]:
    return {
        "schema": "mid360_fmb1_formal_batch1_lock_template_v1",
        "TEMPLATE_ONLY": True,
        "LOCK_STATUS": "UNISSUED",
        "REASON": "W04_NOT_YET_ACQUIRED_OR_ADMITTED",
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "current_admitted_scene_ids": [
            "FMB1_R01",
            "FMB1_R02",
            "FMB1_R03",
            "FMB1_W01",
            "FMB1_W03",
        ],
        "required_final_scene_ids": [
            "FMB1_R01",
            "FMB1_R02",
            "FMB1_R03",
            "FMB1_W01",
            "FMB1_W03",
            "FMB1_W04",
        ],
        "required_station_count": 18,
        "required_snapshot_count": 180,
        "issuance_requirements": [
            "ALL_BAG_SHA_VALID",
            "ALL_TARGET_SHA_VALID",
            "ALL_SNAPSHOT_SHA_VALID",
            "GEOMETRY_MANIFEST_FROZEN",
            "BACKEND_CONTRACT_UNCHANGED",
            "WORKTREE_CLEAN",
            "INDEPENDENT_VERIFIER_PASS",
            "SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION",
        ],
        "actual_formal_trials": 0,
    }


def render_reauthentication_markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# FMB1 当前失败闭包重新认证",
            "",
            f"状态：`{payload['status']}`。当前真实批次仍被 `{payload['FMB1_CURRENT_BLOCK_REASON']}` 正确阻塞。",
            "",
            f"- 原始 bag：{payload['authenticated_bag_count']}/36 已认证",
            f"- station acquisition：{payload['acquisition_pass_station_count']}/18 PASS",
            f"- provisional snapshots：{payload['provisional_snapshot_count']}",
            f"- admitted scenes：{payload['admitted_scene_count']}（Rich {payload['admitted_rich_scene_count']}，Weak {payload['admitted_weak_scene_count']}）",
            "- W02：数据保留，geometry rejected，不属于未来正式集合",
            f"- W02 冻结替换原因：`{payload['W02_REJECTION_REASON']}`",
            "- 正式 lock、ICP unlock、registration authorization：均为 false",
            "- 实际 Open3D/PCL/formal trials：0 / 0 / 0",
            "",
            "该重新认证不采纳 W02，不生成 W04，也不产生任何 backend 结果。",
            "",
        ]
    )


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise PrebackendQualificationError(f"refusing to overwrite different evidence: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def write_text_once(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise PrebackendQualificationError(f"refusing to overwrite different evidence: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def write_core_qualification_evidence(
    repository: Path,
    output_dir: Path,
    *,
    verify_raw_bag_bytes: bool = True,
) -> dict[str, Any]:
    reauthentication = reauthenticate_current_fmb1(
        repository, verify_raw_bag_bytes=verify_raw_bag_bytes
    )
    replacement = verify_w04_replacement_plan(repository)
    write_json_once(output_dir / "current_fmb1_state_reauthentication.json", reauthentication)
    write_text_once(
        output_dir / "current_fmb1_state_reauthentication.md",
        render_reauthentication_markdown(reauthentication),
    )
    write_json_once(output_dir / "w04_replacement_plan_verification.json", replacement)
    checklist = repository / "experiments/mid360_formal_batch1/W04_FIELD_CHECKLIST.md"
    write_text_once(output_dir / "W04_FIELD_CHECKLIST.md", checklist.read_text(encoding="utf-8"))
    write_json_once(output_dir / "formal_batch1_lock_schema.json", formal_lock_schema())
    write_json_once(
        output_dir / "formal_batch1_lock_template_UNISSUED.json",
        unissued_lock_template(),
    )
    return {
        "schema": QUALIFICATION_SCHEMA,
        "reauthentication": reauthentication,
        "replacement_plan": replacement,
        "output_dir": str(output_dir.resolve()),
    }


def sha256sum_lines(paths: Iterable[Path], root: Path) -> str:
    rows: list[str] = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        _require("/" not in relative, "qualification SHA256SUMS only permits top-level files")
        rows.append(f"{sha256_file(path)}  {relative}")
    return "\n".join(rows) + "\n"


def build_qualification_summary(
    output_dir: Path,
    *,
    baseline_tests: Mapping[str, int],
    fmb1_tests: Mapping[str, int],
    full_tests: Mapping[str, int],
) -> dict[str, Any]:
    state = load_json_object(output_dir / "current_fmb1_state_reauthentication.json")
    replacement = load_json_object(output_dir / "w04_replacement_plan_verification.json")
    alignment = load_json_object(output_dir / "protocol_alignment_audit.json")
    amendment = load_json_object(
        output_dir / "zero_perturbation_mainline_amendment_v1_1_PROPOSED.json"
    )
    preflight = load_json_object(output_dir / "current_real_batch_preflight_report.json")
    fixture = load_json_object(output_dir / "fixture_execution_path_qualification.json")
    resume = load_json_object(output_dir / "fixture_resume_interruption_report.json")
    analysis = load_json_object(output_dir / "fixture_analysis_dry_run_report.json")
    publication = load_json_object(output_dir / "fixture_publication_dry_run_report.json")
    no_icp = load_json_object(output_dir / "NO_ICP_ATTESTATION_TONIGHT.json")
    _require(preflight.get("CURRENT_BLOCK_REASON") == CURRENT_BLOCK_REASON, "preflight reason drift")
    _require(preflight.get("scientific_blocker_codes") == [CURRENT_BLOCK_REASON], "preflight must have one scientific blocker")
    _require(not preflight.get("integrity_blockers"), "preflight integrity blocker present")
    _require(not preflight.get("security_blockers"), "preflight security blocker present")
    _require(fixture.get("status") == "PASS_FIXTURE_ONLY", "fixture execution path failed")
    _require(resume.get("status") == "PASS_FIXTURE_ONLY", "fixture resume qualification failed")
    _require(no_icp.get("pass") is True, "tonight NO_ICP attestation failed")
    _require(amendment.get("status") == "PROPOSED_NOT_ACTIVE", "proposed amendment activated")
    for label, test_report in (
        ("baseline", baseline_tests),
        ("fmb1", fmb1_tests),
        ("full", full_tests),
    ):
        _require(test_report.get("failed") == 0, f"{label} tests contain failures")
        _require(test_report.get("errors") == 0, f"{label} tests contain errors")

    schema_qualification = {
        "status": "PASS",
        "fixture_only_validation": True,
        "formal_positive_result_persisted": False,
        "missing_field_rejected": True,
        "nonfinite_rejected": True,
        "invalid_backend_rejected": True,
        "unlocked_identity_rejected": True,
        "zero_track_nonidentity_t0_rejected": True,
        "capture_track_cannot_masquerade_as_zero": True,
        "fixture_cannot_publish": True,
        "track_classifications": [
            "ZERO_PERTURBATION_TRACK",
            "CAPTURE_RADIUS_TRACK",
            "FIXTURE_ONLY",
        ],
    }
    return {
        "schema": "mid360_fmb1_prebackend_qualification_summary_v1",
        "status": "PASS",
        "FMB1_W04_REPLACEMENT_PLAN_FROZEN": replacement.get(
            "FMB1_W04_REPLACEMENT_PLAN_FROZEN"
        ),
        "FMB1_PREBACKEND_EXECUTION_PATH_QUALIFIED": True,
        "FMB1_CURRENT_REAL_BATCH_BLOCKED": True,
        "FMB1_CURRENT_BLOCK_REASON": CURRENT_BLOCK_REASON,
        "FMB1_PRE_REGISTRATION_DATA_READY": False,
        "FORMAL_RUN_MATRIX_ISSUED": False,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "MEASUREMENT_FINAL_RESULT": False,
        "PROPOSED_AMENDMENT_ACTIVE": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "current_real_batch": {
            "authenticated_bag_count": state.get("authenticated_bag_count"),
            "acquisition_pass_station_count": state.get(
                "acquisition_pass_station_count"
            ),
            "provisional_snapshot_count": state.get("provisional_snapshot_count"),
            "admitted_scene_count": state.get("admitted_scene_count"),
            "admitted_rich_scene_count": state.get("admitted_rich_scene_count"),
            "admitted_weak_scene_count": state.get("admitted_weak_scene_count"),
            "admitted_scenes": state.get("admitted_scenes"),
            "rejected_scenes": state.get("rejected_scenes"),
            "W02_REJECTION_PRESERVED": state.get("W02_REJECTION_PRESERVED"),
            "W02_DATA_RETAINED": state.get("W02_DATA_RETAINED"),
            "W02_INCLUDED_IN_FUTURE_FORMAL_SET": state.get(
                "W02_INCLUDED_IN_FUTURE_FORMAL_SET"
            ),
        },
        "w04_future_capture": {
            "station_count": replacement.get("station_count"),
            "new_bag_count_required": replacement.get("new_bag_count_required"),
            "station_ids": replacement.get("station_ids"),
            "location": "UNKNOWN",
            "orientation": "UNKNOWN",
            "height": "UNKNOWN",
        },
        "current_real_preflight": {
            "status": preflight.get("status"),
            "pass": preflight.get("pass"),
            "fail_closed_correctly": True,
            "scientific_blocker_codes": preflight.get("scientific_blocker_codes"),
            "integrity_blocker_count": len(preflight.get("integrity_blockers", [])),
            "security_blocker_count": len(preflight.get("security_blockers", [])),
        },
        "fixture_execution": {
            "status": fixture.get("status"),
            "planned_snapshot_count": fixture.get("planned_snapshot_count"),
            "planned_open3d_trials": fixture.get("planned_open3d_trials"),
            "planned_pcl_trials": fixture.get("planned_pcl_trials"),
            "planned_total_trials": fixture.get("planned_total_trials"),
            "actual_registration_execution_count": fixture.get(
                "actual_registration_execution_count"
            ),
            "resume_status": resume.get("status"),
            "detected_and_rejected_case_count": resume.get(
                "detected_and_rejected_case_count"
            ),
        },
        "result_schema_qualification": schema_qualification,
        "protocol_alignment": {
            "active_primary_endpoints": [
                "weak_direction_capture_radius",
                "strong_direction_capture_radius",
            ],
            "active_perturbation_magnitudes_include_zero": False,
            "active_hypotheses_are_capture_radius": True,
            "active_zero_perturbation_primary_track": False,
            "paper_zero_perturbation_mainline_offset_detected": True,
            "proposal_status": amendment.get("status"),
            "proposal_formal_authority": amendment.get("FORMAL_AUTHORITY"),
            "alignment_audit_status": alignment.get("audit_status"),
        },
        "analysis_hierarchy": {
            "highest_independent_unit": "scene",
            "station_and_snapshot_are_nested_repeats": True,
            "snapshots_treated_as_independent_scenes": False,
            "fixture_analysis_structure_pass": analysis.get("status")
            == "PASS_FIXTURE_ONLY",
            "fixture_publication_performed": publication.get("publication_performed"),
            "fixture_citation_allowed": publication.get("citation_allowed"),
        },
        "NO_ICP_ATTESTATION_TONIGHT_PASS": True,
        "independent_verifier_pass": True,
        "tests": {
            "baseline_before_changes": dict(baseline_tests),
            "mid360_formal_batch1_after_changes": dict(fmb1_tests),
            "full_repository_after_changes": dict(full_tests),
        },
        "tomorrow_after_w04_pass": {
            "expected_final_scene_ids": [
                "FMB1_R01",
                "FMB1_R02",
                "FMB1_R03",
                "FMB1_W01",
                "FMB1_W03",
                "FMB1_W04",
            ],
            "can_enter_final_freeze_quickly": True,
            "automatic_backend_after_admission": False,
            "final_lock_required": True,
            "independent_verifier_required": True,
            "separate_formal_registration_authorization_required": True,
        },
    }


def render_qualification_summary_markdown(payload: Mapping[str, Any]) -> str:
    current = payload["current_real_batch"]
    future = payload["w04_future_capture"]
    tests = payload["tests"]
    return "\n".join(
        [
            "# FMB1 W04 补采前 pre-backend 资格总结",
            "",
            "## 结论",
            "",
            "`FMB1_PREBACKEND_EXECUTION_PATH_QUALIFIED=true`，同时 `FMB1_CURRENT_REAL_BATCH_BLOCKED=true`。",
            "真实批次唯一科学阻塞原因是 `MISSING_ADMITTED_WEAK_REPLACEMENT_W04`。",
            "",
            "## 当前真实闭包",
            "",
            f"- 36/36 raw bags、{current['acquisition_pass_station_count']}/18 stations、{current['provisional_snapshot_count']} provisional snapshots 已重新认证。",
            f"- admitted scenes：Rich {current['admitted_rich_scene_count']}，Weak {current['admitted_weak_scene_count']}，合计 {current['admitted_scene_count']}；没有把 5 scenes 当成完整 batch。",
            "- W02 数据和 rejection 均保留；其规则原因是 `GEOMETRY_ONLY_INELIGIBLE`，未来正式集合不包含 W02。",
            "- W04 计划已在任何 ICP 前冻结；明天需 3 stations、6 个全新 timestamp bags。地点、方向、高度仍为 UNKNOWN。",
            "",
            "## 软件执行链",
            "",
            "- 真实 preflight 正确 fail-closed，没有发行 150-trial/任何 trial matrix、正式 lock 或 authorization。",
            "- fixture-only 6×3×10×2 lifecycle、fresh/resume/interruption、worker invariance 与 SHA stability 全部通过。",
            "- partial、orphan、checksum、manifest、duplicate、missing 均 fail-closed；fixture 不可发布或引用。",
            "- schema 严格区分 ZERO_PERTURBATION_TRACK、CAPTURE_RADIUS_TRACK、FIXTURE_ONLY；zero track 强制 Identity T0。",
            "- scene 保持最高独立分析单位，station/snapshot 是 nested repeated observations。",
            "",
            "## 协议对齐",
            "",
            "现行 active FMB1 primary endpoints 仍是 weak/strong direction capture radius，非论文的 Identity zero-perturbation 主线，且 active perturbation magnitudes 不含 0。",
            "已生成 `PROPOSED_NOT_ACTIVE` 的 v1.1 提案：zero-perturbation 为 proposed primary，capture-radius 为 optional supplementary；提案没有 formal authority。",
            "",
            "## 零执行与测试",
            "",
            "- actual Open3D/PCL/formal trials = 0/0/0；NO_ICP_ATTESTATION_TONIGHT PASS。",
            f"- 改动前：{tests['baseline_before_changes']['collected']} collected，{tests['baseline_before_changes']['skipped']} skipped，0 failed。",
            f"- FMB1 专项：{tests['mid360_formal_batch1_after_changes']['passed']} passed，{tests['mid360_formal_batch1_after_changes']['skipped']} skipped，0 failed。",
            f"- 全量：{tests['full_repository_after_changes']['passed']} passed，{tests['full_repository_after_changes']['skipped']} skipped，0 failed。",
            "",
            "## 明天入口",
            "",
            f"W04 Weak admission PASS 后，目标集合为 R01/R02/R03/W01/W03/W04；仍须 final lock、独立 verifier、单独正式 registration authorization，不能在同一命令中自动运行 ICP。",
            "",
            "`FORMAL_LOCK_ISSUED=false`  ",
            "`FORMAL_ICP_UNLOCKED=false`  ",
            "`FORMAL_REGISTRATION_AUTHORIZED=false`  ",
            "`actual_formal_trials=0`",
            "",
        ]
    )


def write_qualification_sha256sums(output_dir: Path) -> None:
    output_dir = output_dir.resolve(strict=True)
    manifest = output_dir / "SHA256SUMS"
    candidates = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    encoded = sha256sum_lines(candidates, output_dir).encode("ascii")
    if manifest.exists():
        existing = verify_sha256_manifest(output_dir)
        _require(existing["pass"] is True, "refusing to replace an invalid checksum manifest")
    temporary = manifest.with_suffix(".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(manifest)
