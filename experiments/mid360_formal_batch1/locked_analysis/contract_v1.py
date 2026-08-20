"""C1+C2 bindings and constants for the locked-analysis implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RAW_EXECUTION_COMMIT = "059e39533991d929a97ab208ad738643af82d09a"
RAW_EXECUTION_TAG = "execution/fmb1-zero-perturbation-v1.1-exec-r3-formal-v1"
POSTRUN_VERIFIER_CODE_COMMIT = "ec72d23f0f94cd91a84f2960b87672788266524f"
POSTRUN_VERIFICATION_COMMIT = "18bb94e62761f5193da8cdc5509c470cb4983244"
POSTRUN_VERIFICATION_TAG = "verification/fmb1-zero-perturbation-postrun-v1"
R3_LOCK_FINGERPRINT = "fd601e8daf62c488a3a079beea05f65c9bd283399ae4d7f3acf16e63fc473a6d"
C2_COMMIT = "364c0ae76801c373d9ea6a4d9dd36e2fdf211e09"
C2_TAG = "freeze/fmb1-zero-perturbation-analysis-determinacy-c2"
C2_TAG_OBJECT = "b0491f91723f726f22cb5137e0d044ef9f03d120"

SCENE_ORDER = (
    "FMB1_R01",
    "FMB1_R02",
    "FMB1_R03",
    "FMB1_W01",
    "FMB1_W02",
    "FMB1_W03",
)
RICH_SCENES = SCENE_ORDER[:3]
WEAK_SCENES = SCENE_ORDER[3:]
STATION_ORDER = ("S01", "S02", "S03")
BACKENDS = ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE")
SNAPSHOTS_PER_STATION = 10
SNAPSHOTS_PER_SCENE = 30
PRIMARY_EXACT_ALLOCATION_COUNT = 20
PRIMARY_EXACT_P_DENOMINATOR = 20
CENTERED_PERMUTATION_DRAW_COUNT = 10000
CENTERED_PERMUTATION_SEED = 20260820
CENTERED_PERMUTATION_P_DENOMINATOR = 10001
PHYSICAL_REFERENCE_SEMANTICS = (
    "NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT"
)

PROTECTED_BINDINGS = {
    "c1_contract": (
        "experiments/mid360_formal_batch1/amendments/"
        "zero_perturbation_analysis_contract_v1_1_r1.json",
        "4120847bb471efbdbd32026ac23e959ab9ae4236709cf599b6e15701eb8b153d",
    ),
    "c1_protocol": (
        "experiments/mid360_formal_batch1/amendments/"
        "zero_perturbation_analysis_protocol_v1_1_r1.md",
        "9982193c3c4c2a6cda77c8db858fd73e73f61da7eb7ffe651610e5b1ed1c0b4c",
    ),
    "c1_missingness": (
        "experiments/mid360_formal_batch1/amendments/"
        "zero_perturbation_analysis_missingness_clarification_v1_1_r1_c1.json",
        "7b50dabef2352b666570657d519e98e9a5ae277da777be222c3483134fca69fb",
    ),
    "c2_clarification": (
        "experiments/mid360_formal_batch1/amendments/"
        "zero_perturbation_analysis_determinacy_clarification_v1_1_r1_c2.json",
        "a446d069436c2972a11316e8a709aeb3c4265817b5e9889fa9583275ffc7be6b",
    ),
    "c1_c2_determinacy_audit": (
        "results/mid360_formal_batch1/zero_perturbation_locked_analysis_preparation_v1/"
        "analysis_contract_determinacy_audit_c1_c2.json",
        "bb9e821efac1db457facdfaa8857f10ccb8edc4a6e7cde8eb1b201449b8675b6",
    ),
    "c1_c2_determinacy_verifier": (
        "results/mid360_formal_batch1/zero_perturbation_locked_analysis_preparation_v1/"
        "analysis_contract_determinacy_audit_c1_c2_independent_verification.json",
        "0007648c65634932b9ed78aed7ebb77b280fdf4d902493da05413bc765d14df9",
    ),
    "postrun_verification_report": (
        "results/mid360_formal_batch1/zero_perturbation_v1_1_postrun_verification_v1/"
        "postrun_independent_verification.json",
        "8067cc8735fd5b2a849bce8dd3839a59b87cd7cac716009892be622903084f90",
    ),
    "trial_plan": (
        "experiments/mid360_formal_batch1/zero_perturbation_trial_plan_v1_1.json",
        "e96917f93ffbcccf0e2c1d75ee75ba86e342f08a9e4651732e2457b7ad867427",
    ),
    "formal_result_schema": (
        "experiments/mid360_formal_batch1/zero_perturbation_trial_result_schema_v1_1.json",
        "cf226717911eef9107a45523cff39519b6c70c8d5e53f48b2e3accb18964fab2",
    ),
    "backend_contract": (
        "frozen_assets/backend_parameter_contract.json",
        "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9",
    ),
}

FUTURE_OUTPUT_FILES = (
    "analysis_summary.json",
    "analysis_summary.md",
    "translation_station_summaries.csv",
    "translation_scene_summaries.csv",
    "translation_exact_permutations.csv",
    "rotation_station_summaries.csv",
    "rotation_scene_summaries.csv",
    "rotation_exact_permutations.csv",
    "cross_backend_scene_pairs.csv",
    "cross_backend_station_pairs.csv",
    "snapshot_direction_cosines.csv",
    "reassociation_scene_summaries.csv",
    "reassociation_centered_rows.csv",
    "reassociation_centered_permutation_summary.json",
    "reassociation_centered_permutation_draws.csv",
    "systematic_station_values.csv",
    "systematic_scene_values.csv",
    "analysis_missingness_inventory.csv",
    "SHA256SUMS",
)


class ContractBindingError(RuntimeError):
    """Raised when a frozen analysis dependency no longer matches."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractBindingError(f"expected object: {path}")
    return payload


def verify_contract_bindings(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    verified: dict[str, dict[str, Any]] = {}
    for role, (relative_path, expected_sha) in PROTECTED_BINDINGS.items():
        path = repository / relative_path
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ContractBindingError(
                f"{role} SHA mismatch: expected {expected_sha}, got {actual_sha}"
            )
        verified[role] = {
            "path": relative_path,
            "sha256": actual_sha,
            "bytes": path.stat().st_size,
        }
    audit = read_json(repository / PROTECTED_BINDINGS["c1_c2_determinacy_audit"][0])
    audit_verifier = read_json(
        repository / PROTECTED_BINDINGS["c1_c2_determinacy_verifier"][0]
    )
    postrun = read_json(repository / PROTECTED_BINDINGS["postrun_verification_report"][0])
    if not (
        audit.get("gate", {}).get("required_under_specified_count") == 0
        and audit.get("gate", {}).get("unresolved_root_definition_count") == 0
        and audit.get("gate", {}).get("FMB1_LOCKED_ANALYSIS_CONTRACT_DETERMINATE")
        is True
        and audit_verifier.get("pass") is True
        and postrun.get("FMB1_POSTRUN_INDEPENDENT_VERIFICATION_PASS") is True
        and postrun.get("VERIFIED_TRIAL_COUNT") == 360
    ):
        raise ContractBindingError("determinacy or Post-run PASS gate mismatch")
    return {
        "status": "PASS",
        "binding_count": len(verified),
        "bindings": verified,
        "required_under_specified_count": 0,
        "unresolved_root_definition_count": 0,
        "postrun_verified_trial_count": 360,
    }
