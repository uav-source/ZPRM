import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import experiments.mid360_formal_batch1.postrun_verification.independent_postrun_verifier_v1 as verifier
from test_postrun_independent_verifier_v1 import SCHEMA, plan_row, result_row


def sha(path: Path) -> str:
    return verifier.sha256_file(path)


@pytest.mark.parametrize(
    "kind,expected,results,markers,attempts",
    [
        ("missing", ["a"], [], ["a"], {}),
        ("duplicate", ["a"], ["a", "a"], ["a"], {"a": [1]}),
        ("orphan", ["a"], ["a"], ["a", "orphan"], {"a": [1]}),
        ("unauthorized", ["a"], ["a", "bad"], ["a", "bad"], {"a": [1], "bad": [1]}),
        ("second_attempt", ["a"], ["a"], ["a"], {"a": [1, 2]}),
        ("missing_marker", ["a"], ["a"], [], {"a": [1]}),
    ],
)
def test_inventory_tamper_fails_closed(kind, expected, results, markers, attempts) -> None:
    with pytest.raises(verifier.IndependentPostrunVerificationError, match="inventory"):
        verifier.validate_inventory_identity_sets(expected, results, markers, attempts, expected_count=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scene_id", "FMB1_W04"),
        ("attempt", 2),
        ("track_id", "CAPTURE_BASIN"),
        ("source_sha256", "9" * 64),
        ("target_sha256", "8" * 64),
        ("code_commit", "7" * 40),
        ("formal_lock_sha256", "6" * 64),
    ],
)
def test_result_identity_tamper_fails_closed(field: str, value) -> None:
    row = result_row(); row[field] = value
    with pytest.raises((verifier.IndependentPostrunVerificationError, ValueError)):
        verifier._validate_result_identity(
            row, plan_row(), SCHEMA, trial_plan_sha="3" * 64,
            lock_sha=verifier.R3_LOCK_SHA256, environment_sha="4" * 64,
        )


@pytest.mark.parametrize("prefix", ["source", "target"])
def test_array_byte_sha_tamper_fails_closed(tmp_path: Path, prefix: str) -> None:
    points = np.ascontiguousarray(np.arange(60, dtype="<f8").reshape(20, 3))
    path = tmp_path / f"{prefix}.npy"; np.save(path, points, allow_pickle=False)
    plan = plan_row(); plan[f"{prefix}_reference"] = path.name
    plan[f"{prefix}_point_count"] = 20
    plan[f"{prefix}_sha256"] = sha(path)
    plan[f"{prefix}_array_sha256"] = "0" * 64
    with pytest.raises(verifier.IndependentPostrunVerificationError, match="array byte SHA"):
        verifier.load_canonical_array(tmp_path, plan, prefix)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def build_authorization_fixture(root: Path) -> tuple[Path, dict]:
    execution = root / "execution-results"; auth_dir = execution / "authorization"
    runtime_rel = "zero_perturbation_runtime/canonical"
    auth = {
        "authorization_id": "fixture-auth", "immutable": True, "status": "ISSUED",
        "lock_fingerprint": verifier.R3_FINGERPRINT, "lock_file_sha256": verifier.R3_LOCK_SHA256,
        "execution_code_commit": verifier.R3_EXECUTION_CODE_COMMIT,
        "planned_open3d_count": 180, "planned_pcl_count": 180, "planned_trial_count": 360,
        "authoritative_runtime_root": runtime_rel,
    }
    auth_path = auth_dir / "formal_registration_authorization.json"; write_json(auth_path, auth)
    auth_sha = sha(auth_path)
    (auth_dir / "formal_registration_authorization.sha256").write_text(f"{auth_sha}  {auth_path.name}\n")
    write_json(auth_dir / "authorization_verification_report.json", {
        "AUTHORIZATION_VERIFICATION_PASS": True, "authorization_sha256": auth_sha})
    write_json(auth_dir / "authorization_in_use.json", {
        "authorization_id": "fixture-auth", "authorization_sha256": auth_sha,
        "lock_fingerprint": verifier.R3_FINGERPRINT, "state": "IN_USE",
        "reusable_for_new_fresh": False})
    run_path = execution / "execution/run_manifest.json"; write_json(run_path, {"status": "COMPLETE"})
    receipt = {
        "authorization_id": "fixture-auth", "authorization_sha256": auth_sha,
        "lock_fingerprint": verifier.R3_FINGERPRINT, "state": "CONSUMED", "consumed": True,
        "reusable": False, "actual_open3d_trials": 180, "actual_pcl_trials": 180,
        "actual_total_trials": 360, "FURTHER_REGISTRATION_AUTHORIZED": False,
        "execution_result_manifest_sha256": sha(run_path),
    }
    receipt_path=auth_dir / "authorization_consumption_receipt.json"; write_json(receipt_path, receipt)
    (auth_dir / "authorization_consumption_receipt.sha256").write_text(f"{sha(receipt_path)}  {receipt_path.name}\n")
    live=root/runtime_rel/"authorization/formal_registration_authorization.json"
    write_json(live, auth)
    return execution, receipt


@pytest.mark.parametrize("tamper", ["receipt", "reusable", "manifest_sha", "fingerprint"])
def test_authorization_lifecycle_tamper_fails_closed(tmp_path: Path, tamper: str) -> None:
    execution, receipt = build_authorization_fixture(tmp_path)
    auth_dir=execution/"authorization"; receipt_path=auth_dir/"authorization_consumption_receipt.json"
    if tamper == "receipt": receipt["actual_total_trials"] = 359
    elif tamper == "reusable": receipt["reusable"] = True
    elif tamper == "manifest_sha": receipt["execution_result_manifest_sha256"] = "0" * 64
    else:
        in_use=json.loads((auth_dir/"authorization_in_use.json").read_text())
        in_use["lock_fingerprint"]="0"*64; write_json(auth_dir/"authorization_in_use.json",in_use)
    if tamper != "fingerprint":
        write_json(receipt_path, receipt)
        (auth_dir/"authorization_consumption_receipt.sha256").write_text(f"{sha(receipt_path)}  {receipt_path.name}\n")
    with pytest.raises(verifier.IndependentPostrunVerificationError):
        verifier.authorization_lifecycle_audit(tmp_path, execution, {"status": "COMPLETE"})


def build_checksum_fixture(root: Path) -> Path:
    root.mkdir()
    primary=[]
    for index in range(740):
        path=root/"primary"/f"{index:04d}.bin"; path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"fixture-{index}".encode())
        primary.append({"results_relative_path":str(path.relative_to(root)),"sha256":sha(path),"bytes":path.stat().st_size})
    freeze=root/"formal_execution_raw_freeze_manifest.json"
    write_json(freeze,{"frozen_primary_artifacts":primary})
    write_json(root/"formal_execution_summary.json",{"fixture":True})
    (root/"formal_execution_summary.md").write_text("fixture\n")
    files=sorted(p for p in root.rglob("*") if p.is_file())
    (root/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(root)}\n" for p in files))
    return root


@pytest.mark.parametrize("tamper", ["malformed", "unsafe"])
def test_checksum_manifest_tamper_fails_closed(tmp_path: Path, tamper: str) -> None:
    root=build_checksum_fixture(tmp_path/"exec")
    lines=(root/"SHA256SUMS").read_text().splitlines()
    lines[0] = "malformed" if tamper == "malformed" else f"{'0'*64}  ../escape"
    (root/"SHA256SUMS").write_text("\n".join(lines)+"\n")
    with pytest.raises(verifier.IndependentPostrunVerificationError):
        verifier.parse_checksum_manifest(root)


def test_backend_pair_source_tamper_fails_closed() -> None:
    left=plan_row(); right=copy.deepcopy(left)
    right["trial_id"]=right["trial_id"].replace("O3D","PCL")
    right["backend"]="PCL_POINT_TO_PLANE"; right["backend_version"]="1.15.1"
    right["source_array_sha256"]="0"*64
    with pytest.raises(verifier.IndependentPostrunVerificationError, match="inputs differ"):
        verifier.validate_backend_pair_inputs([left,right],snapshot_count=1)


@pytest.mark.parametrize("tamper", ["authorization", "timezone"])
def test_start_marker_tamper_fails_closed(tamper: str) -> None:
    marker = {
        "schema": "mid360_fmb1_formal_trial_attempt_start_v1_1_r1",
        "trial_id": "trial-a", "attempt_number": 1,
        "authorization_sha256": "a" * 64,
        "started_at_utc": "2026-08-20T08:00:00+00:00",
    }
    if tamper == "authorization": marker["authorization_sha256"] = "b" * 64
    else: marker["started_at_utc"] = "2026-08-20T08:00:00+08:00"
    with pytest.raises(verifier.IndependentPostrunVerificationError):
        verifier.validate_start_marker_payload(
            marker, trial_id="trial-a", authorization_sha256="a" * 64
        )
