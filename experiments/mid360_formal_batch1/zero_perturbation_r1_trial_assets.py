"""Build and validate the frozen FMB1 zero-perturbation R1 trial assets.

This module is deliberately registration-free.  It reads authenticated
manifests, hashes byte payloads, and emits a deterministic execution plan and
future result schema.  It never imports Open3D/PCL and never estimates a pose.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from phase_a_harness.rotation_metrics import rotation_metric_audit


REPOSITORY = Path(__file__).resolve().parents[2]
FMB1_ROOT = REPOSITORY / "experiments/mid360_formal_batch1"
FINAL_DATASET_ROOT = REPOSITORY / "results/mid360_formal_batch1/final_dataset_v1"

AMENDMENT_ID = "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1"
TRACK_ID = "ZERO_PERTURBATION_TRACK"
PLAN_SCHEMA = "mid360_fmb1_zero_perturbation_trial_plan_v1_1_r1"
PLAN_ID = "FMB1_ZERO_PERTURBATION_TRIAL_PLAN_V1_1_R1"
RESULT_SCHEMA_NAME = "mid360_fmb1_zero_perturbation_trial_result_v1_1_r1"
PHYSICAL_REFERENCE_SEMANTICS = "NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT"
PLANNED_STATUS = "PLANNED_NOT_AUTHORIZED_NOT_EXECUTED"

RICH_SCENES = ("FMB1_R01", "FMB1_R02", "FMB1_R03")
WEAK_SCENES = ("FMB1_W01", "FMB1_W02", "FMB1_W03")
SCENE_CLASSES = {
    **{scene: "RICH" for scene in RICH_SCENES},
    **{scene: "WEAK" for scene in WEAK_SCENES},
}
STATIONS = ("S01", "S02", "S03")
BACKENDS = ("OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE")
BACKEND_VERSIONS = {
    "OPEN3D_POINT_TO_PLANE": "0.19.0+b012259",
    "PCL_POINT_TO_PLANE": "1.15.1",
}
BACKEND_CODES = {
    "OPEN3D_POINT_TO_PLANE": "O3D",
    "PCL_POINT_TO_PLANE": "PCL",
}
BACKEND_CONTRACT_KEYS = {
    "OPEN3D_POINT_TO_PLANE": "open3d",
    "PCL_POINT_TO_PLANE": "pcl",
}
IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

SNAPSHOT_MANIFEST_PATH = FINAL_DATASET_ROOT / "final_snapshot_manifest.csv"
TARGET_MANIFEST_PATH = FINAL_DATASET_ROOT / "final_target_manifest.csv"
BACKEND_PARAMETER_CONTRACT_PATH = REPOSITORY / "frozen_assets/backend_parameter_contract.json"
AMENDMENT_PATH = FMB1_ROOT / "amendments/zero_perturbation_mainline_v1_1_r1.json"
ANALYSIS_CONTRACT_PATH = (
    FMB1_ROOT / "amendments/zero_perturbation_analysis_contract_v1_1_r1.json"
)

PLAN_CSV_PATH = FMB1_ROOT / "zero_perturbation_trial_plan_v1_1.csv"
PLAN_JSON_PATH = FMB1_ROOT / "zero_perturbation_trial_plan_v1_1.json"
RESULT_SCHEMA_PATH = FMB1_ROOT / "zero_perturbation_trial_result_schema_v1_1.json"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRIAL_ID_RE = re.compile(
    r"^FMB1-ZP11R1-FMB1-[RW]0[1-3]-S0[1-3]-Q(?:0[1-9]|10)-(?:O3D|PCL)$"
)

CSV_FIELDS = (
    "trial_id",
    "batch_id",
    "amendment_id",
    "track_id",
    "scene_id",
    "final_geometry_class",
    "station_id",
    "attempt",
    "snapshot_id",
    "snapshot_quantile",
    "query_timestamp",
    "backend",
    "backend_version",
    "backend_canonical_parameter_sha256",
    "source_reference",
    "source_sha256",
    "source_array_sha256",
    "source_point_count",
    "target_reference",
    "target_sha256",
    "target_array_sha256",
    "target_point_count",
    "T0",
    "T_reference_nominal",
    "translation_perturbation_m",
    "rotation_perturbation_deg",
    "backend_parameter_contract_sha256",
    "active_amendment_sha256",
    "analysis_contract_sha256",
    "planned_status",
)


class R1TrialAssetError(ValueError):
    """Raised when an R1 plan, schema, or future result fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise R1TrialAssetError(f"expected JSON object: {path}")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise R1TrialAssetError(f"{label} is not a lowercase SHA256")
    return value


def _require_identity(value: Any, label: str) -> None:
    if value != IDENTITY_4X4:
        raise R1TrialAssetError(f"{label} must be exact Identity")


def _scene_attempt(scene_id: str) -> int:
    return 2 if scene_id == "FMB1_W02" else 1


def deterministic_trial_id(snapshot_id: str, backend: str) -> str:
    if backend not in BACKEND_CODES:
        raise R1TrialAssetError(f"unsupported backend: {backend}")
    return f"FMB1-ZP11R1-{snapshot_id.replace('_', '-')}-{BACKEND_CODES[backend]}"


def _validate_dependency_identity(amendment: Mapping[str, Any], analysis: Mapping[str, Any]) -> None:
    amendment_id = amendment.get("amendment_id", amendment.get("AMENDMENT_ID"))
    if amendment_id != AMENDMENT_ID:
        raise R1TrialAssetError("R1 amendment identity mismatch")
    analysis_amendment = analysis.get("amendment_id", analysis.get("AMENDMENT_ID"))
    if analysis_amendment != AMENDMENT_ID:
        raise R1TrialAssetError("analysis contract is not bound to R1 amendment")
    experimental_units = analysis.get("experimental_units")
    if type(experimental_units) is not dict or experimental_units.get("primary_experimental_unit") != "scene":
        raise R1TrialAssetError("analysis contract must make scene the primary unit")
    amendment_phase = (amendment.get("status"), amendment.get("activation_effective"))
    analysis_phase = (analysis.get("status"), analysis.get("activation_effective"))
    allowed_phases = {
        ("CANDIDATE_PENDING_INDEPENDENT_ACTIVATION_VERIFIER", False),
        ("ACTIVE", True),
    }
    if amendment_phase not in allowed_phases or analysis_phase not in allowed_phases:
        raise R1TrialAssetError("R1 dependency status/activation phase is inconsistent")
    if amendment_phase != analysis_phase:
        raise R1TrialAssetError("amendment and analysis contract phases disagree")
    for label, document in (("amendment", amendment), ("analysis contract", analysis)):
        if document.get("FORMAL_REGISTRATION_AUTHORIZED") is not False:
            raise R1TrialAssetError(f"{label} unexpectedly authorizes registration")
        if document.get("actual_formal_trials") != 0:
            raise R1TrialAssetError(f"{label} was not frozen at trial count zero")


def build_trial_plan(
    *,
    snapshot_manifest_path: Path = SNAPSHOT_MANIFEST_PATH,
    target_manifest_path: Path = TARGET_MANIFEST_PATH,
    backend_contract_path: Path = BACKEND_PARAMETER_CONTRACT_PATH,
    amendment_path: Path = AMENDMENT_PATH,
    analysis_contract_path: Path = ANALYSIS_CONTRACT_PATH,
    verify_payload_bytes: bool = True,
) -> dict[str, Any]:
    """Return the deterministic 360-row R1 plan without loading a backend."""

    snapshots = _load_csv(snapshot_manifest_path)
    targets = _load_csv(target_manifest_path)
    backend_contract = _load_json_object(backend_contract_path)
    amendment = _load_json_object(amendment_path)
    analysis = _load_json_object(analysis_contract_path)
    _validate_dependency_identity(amendment, analysis)

    if len(snapshots) != 180:
        raise R1TrialAssetError(f"expected 180 snapshots, found {len(snapshots)}")
    if len(targets) != 18:
        raise R1TrialAssetError(f"expected 18 targets, found {len(targets)}")

    target_by_station: dict[tuple[str, str], dict[str, str]] = {}
    for target in targets:
        key = (target["scene_id"], target["station_id"])
        if key in target_by_station:
            raise R1TrialAssetError(f"duplicate target: {key}")
        if key[0] not in SCENE_CLASSES or key[1] not in STATIONS:
            raise R1TrialAssetError(f"target outside final R1 dataset: {key}")
        if int(target["attempt"]) != _scene_attempt(key[0]):
            raise R1TrialAssetError(f"wrong acquisition attempt for target: {key}")
        if int(target["query_contribution_to_target"]) != 0:
            raise R1TrialAssetError(f"query contribution detected for target: {key}")
        target_by_station[key] = target

    expected_station_keys = {
        (scene, station) for scene in SCENE_CLASSES for station in STATIONS
    }
    if set(target_by_station) != expected_station_keys:
        raise R1TrialAssetError("target scene/station inventory is incomplete")

    contract_sha = sha256_file(backend_contract_path)
    amendment_sha = sha256_file(amendment_path)
    analysis_sha = sha256_file(analysis_contract_path)
    backend_canonical: dict[str, str] = {}
    for backend, key in BACKEND_CONTRACT_KEYS.items():
        entry = backend_contract.get(key)
        if type(entry) is not dict:
            raise R1TrialAssetError(f"missing backend contract entry: {key}")
        canonical = _require_sha(entry.get("canonical_sha256"), f"{key}.canonical_sha256")
        version = entry.get("parameters", {}).get("version")
        if version != BACKEND_VERSIONS[backend]:
            raise R1TrialAssetError(f"{key} version mismatch")
        backend_canonical[backend] = canonical

    seen_snapshots: set[str] = set()
    station_counts: dict[tuple[str, str], int] = {}
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        scene = snapshot["scene_id"]
        station = snapshot["station_id"]
        snapshot_id = snapshot["snapshot_id"]
        key = (scene, station)
        if scene not in SCENE_CLASSES or station not in STATIONS:
            raise R1TrialAssetError(f"snapshot outside final R1 dataset: {snapshot_id}")
        if "W04" in snapshot_id or scene == "FMB1_W04":
            raise R1TrialAssetError("retired W04 identity entered the plan source")
        attempt = int(snapshot["attempt"])
        if attempt != _scene_attempt(scene):
            raise R1TrialAssetError(f"wrong acquisition attempt for {snapshot_id}")
        if snapshot_id in seen_snapshots:
            raise R1TrialAssetError(f"duplicate snapshot: {snapshot_id}")
        seen_snapshots.add(snapshot_id)
        station_counts[key] = station_counts.get(key, 0) + 1

        target = target_by_station[key]
        if snapshot["target_npy_sha256"] != target["target_npy_sha256"]:
            raise R1TrialAssetError(f"snapshot/target SHA mismatch: {snapshot_id}")
        if snapshot["target_array_sha256"] != target["target_array_sha256"]:
            raise R1TrialAssetError(f"snapshot/target array SHA mismatch: {snapshot_id}")

        source_path = Path(snapshot["source_path"])
        target_path = Path(target["target_path"])
        source_sha = _require_sha(snapshot["source_npy_sha256"], "source_npy_sha256")
        target_sha = _require_sha(target["target_npy_sha256"], "target_npy_sha256")
        if verify_payload_bytes:
            if sha256_file(source_path) != source_sha:
                raise R1TrialAssetError(f"source payload SHA mismatch: {snapshot_id}")
            if sha256_file(target_path) != target_sha:
                raise R1TrialAssetError(f"target payload SHA mismatch: {key}")

        for backend in BACKENDS:
            rows.append(
                {
                    "trial_id": deterministic_trial_id(snapshot_id, backend),
                    "batch_id": "FMB1",
                    "amendment_id": AMENDMENT_ID,
                    "track_id": TRACK_ID,
                    "scene_id": scene,
                    "final_geometry_class": SCENE_CLASSES[scene],
                    "station_id": station,
                    "attempt": attempt,
                    "snapshot_id": snapshot_id,
                    "snapshot_quantile": float(snapshot["quantile"]),
                    "query_timestamp": float(snapshot["query_timestamp"]),
                    "backend": backend,
                    "backend_version": BACKEND_VERSIONS[backend],
                    "backend_canonical_parameter_sha256": backend_canonical[backend],
                    "source_reference": str(source_path),
                    "source_sha256": source_sha,
                    "source_array_sha256": _require_sha(
                        snapshot["source_array_sha256"], "source_array_sha256"
                    ),
                    "source_point_count": int(snapshot["source_point_count"]),
                    "target_reference": str(target_path),
                    "target_sha256": target_sha,
                    "target_array_sha256": _require_sha(
                        target["target_array_sha256"], "target_array_sha256"
                    ),
                    "target_point_count": int(target["target_point_count"]),
                    "T0": [list(row) for row in IDENTITY_4X4],
                    "T_reference_nominal": [list(row) for row in IDENTITY_4X4],
                    "translation_perturbation_m": 0.0,
                    "rotation_perturbation_deg": 0.0,
                    "backend_parameter_contract_sha256": contract_sha,
                    "active_amendment_sha256": amendment_sha,
                    "analysis_contract_sha256": analysis_sha,
                    "planned_status": PLANNED_STATUS,
                }
            )

    if set(station_counts) != expected_station_keys or set(station_counts.values()) != {10}:
        raise R1TrialAssetError("every final station must contribute exactly 10 snapshots")
    if len(rows) != 360 or len({row["trial_id"] for row in rows}) != 360:
        raise R1TrialAssetError("trial plan must contain 360 unique deterministic rows")

    counts = {
        "scene_count": len({row["scene_id"] for row in rows}),
        "station_count": len({(row["scene_id"], row["station_id"]) for row in rows}),
        "snapshot_count": len({row["snapshot_id"] for row in rows}),
        "rich_snapshot_count": len(
            {row["snapshot_id"] for row in rows if row["final_geometry_class"] == "RICH"}
        ),
        "weak_snapshot_count": len(
            {row["snapshot_id"] for row in rows if row["final_geometry_class"] == "WEAK"}
        ),
        "open3d_trial_count": sum(
            row["backend"] == "OPEN3D_POINT_TO_PLANE" for row in rows
        ),
        "pcl_trial_count": sum(row["backend"] == "PCL_POINT_TO_PLANE" for row in rows),
        "native_trial_count": 0,
        "total_trial_count": len(rows),
        "duplicate_trial_count": len(rows) - len({row["trial_id"] for row in rows}),
        "missing_trial_count": 0,
    }
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": PLAN_ID,
        "amendment_id": AMENDMENT_ID,
        "track_id": TRACK_ID,
        "physical_reference_semantics": PHYSICAL_REFERENCE_SEMANTICS,
        "primary_experimental_unit": "scene",
        "snapshot_independence_claimed": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "NO_REGISTRATION_BACKEND_IMPORTED_OR_CALLED": True,
        "generated_from": {
            "snapshot_manifest": str(snapshot_manifest_path.relative_to(REPOSITORY)),
            "snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
            "target_manifest": str(target_manifest_path.relative_to(REPOSITORY)),
            "target_manifest_sha256": sha256_file(target_manifest_path),
        },
        "bindings": {
            "backend_parameter_contract": str(backend_contract_path.relative_to(REPOSITORY)),
            "backend_parameter_contract_sha256": contract_sha,
            "active_amendment": str(amendment_path.relative_to(REPOSITORY)),
            "active_amendment_sha256": amendment_sha,
            "analysis_contract": str(analysis_contract_path.relative_to(REPOSITORY)),
            "analysis_contract_sha256": analysis_sha,
        },
        "counts": counts,
        "rows": rows,
    }


def _matrix_schema(*, nullable: bool = False) -> dict[str, Any]:
    matrix = {
        "type": "array",
        "minItems": 4,
        "maxItems": 4,
        "items": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "number"},
        },
    }
    return {"oneOf": [matrix, {"type": "null"}]} if nullable else matrix


def _nullable_number(*, minimum: float | None = None, maximum: float | None = None) -> dict[str, Any]:
    number: dict[str, Any] = {"type": "number"}
    if minimum is not None:
        number["minimum"] = minimum
    if maximum is not None:
        number["maximum"] = maximum
    return {"oneOf": [number, {"type": "null"}]}


def build_result_schema() -> dict[str, Any]:
    """Return the strict schema for future, plan-bound R1 result rows."""

    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    nullable_count = {
        "oneOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]
    }
    invalid_reasons = [
        "NO_INITIAL_CORRESPONDENCE", "NO_FINAL_CORRESPONDENCE",
        "INSUFFICIENT_VALID_NORMALS", "NONFINITE_COMMON_METRICS", "OTHER",
    ]
    properties: dict[str, Any] = {
        "schema": {"const": RESULT_SCHEMA_NAME},
        "trial_id": {"type": "string", "pattern": TRIAL_ID_RE.pattern},
        "batch_id": {"const": "FMB1"},
        "amendment_id": {"const": AMENDMENT_ID},
        "track_id": {"const": TRACK_ID},
        "scene_id": {"enum": list(SCENE_CLASSES)},
        "geometry_class": {"enum": ["RICH", "WEAK"]},
        "station_id": {"enum": list(STATIONS)},
        "attempt": {"type": "integer", "enum": [1, 2]},
        "snapshot_id": {
            "type": "string",
            "pattern": r"^FMB1_[RW]0[1-3]_S0[1-3]_Q(?:0[1-9]|10)$",
        },
        "backend": {"enum": list(BACKENDS)},
        "backend_version": {"enum": list(BACKEND_VERSIONS.values())},
        "source_reference": {"type": "string", "minLength": 1},
        "source_sha256": sha,
        "target_reference": {"type": "string", "minLength": 1},
        "target_sha256": sha,
        "T0": {"const": IDENTITY_4X4},
        "T_reference_nominal": {"const": IDENTITY_4X4},
        "T_est": _matrix_schema(nullable=True),
        "Delta_T": _matrix_schema(nullable=True),
        "translation_x_m": _nullable_number(),
        "translation_y_m": _nullable_number(),
        "translation_z_m": _nullable_number(),
        "translation_norm_m": _nullable_number(minimum=0.0),
        "rotation_angle_rad": _nullable_number(minimum=0.0),
        "rotation_angle_deg": _nullable_number(minimum=0.0),
        "solver_status": {"type": "string", "minLength": 1},
        "finite_result": {"type": "boolean"},
        "scientific_status": {
            "enum": [
                "COMPLETED",
                "SOLVER_NON_CONVERGENCE",
                "FINITE_SCIENTIFIC_FAILURE",
                "NONFINITE_RESULT_RECORDED_WITHOUT_NONFINITE_TRANSFORM",
                "NOT_EVALUATED_INFRASTRUCTURE_FAILURE",
            ]
        },
        "infrastructure_status": {
            "enum": [
                "OK",
                "PROCESS_CRASH",
                "FILE_READ_CORRUPTION",
                "RESULT_SCHEMA_WRITE_FAILURE",
                "PCL_EXECUTABLE_INFRASTRUCTURE_FAILURE",
            ]
        },
        "retry_eligible": {"type": "boolean"},
        "retry_reason": {
            "oneOf": [{"type": "string", "minLength": 1}, {"type": "null"}]
        },
        "common_association_valid": {
            "oneOf": [{"type": "boolean"}, {"type": "null"}]
        },
        "common_association_invalid_reason": {
            "oneOf": [{"enum": invalid_reasons}, {"type": "null"}]
        },
        "common_association_invalid_detail": {
            "oneOf": [
                {"type": "string", "minLength": 1, "maxLength": 1000},
                {"type": "null"},
            ]
        },
        "initial_correspondence_count": nullable_count,
        "initial_valid_normal_correspondence_count": nullable_count,
        "final_correspondence_count": nullable_count,
        "final_valid_normal_correspondence_count": nullable_count,
        "correspondence_turnover": _nullable_number(minimum=0.0, maximum=1.0),
        "accepted_source_turnover": _nullable_number(minimum=0.0, maximum=1.0),
        "correspondence_count_change_ratio": _nullable_number(minimum=-1.0),
        "initial_residual_rmse": _nullable_number(minimum=0.0),
        "final_residual_rmse": _nullable_number(minimum=0.0),
        "residual_rmse_change": _nullable_number(),
        "median_normal_angle_change_deg": _nullable_number(minimum=0.0, maximum=180.0),
        "q95_normal_angle_change_deg": _nullable_number(minimum=0.0, maximum=180.0),
        "backend_parameter_contract_sha256": sha,
        "backend_canonical_parameter_sha256": sha,
        "active_amendment_sha256": sha,
        "analysis_contract_sha256": sha,
        "trial_plan_sha256": sha,
        "formal_lock_sha256": sha,
        "code_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "environment_identity": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "python",
                "numpy",
                "scipy",
                "open3d",
                "pcl",
                "environment_manifest_sha256",
            ],
            "properties": {
                "python": {"const": "3.11.15"},
                "numpy": {"const": "1.26.4"},
                "scipy": {"const": "1.11.4"},
                "open3d": {"const": "0.19.0+b012259"},
                "pcl": {"const": "1.15.1"},
                "environment_manifest_sha256": sha,
            },
        },
        "physical_reference_semantics": {"const": PHYSICAL_REFERENCE_SEMANTICS},
        "execution_kind": {"const": "FORMAL"},
        "fixture_only": {"const": False},
        "created_at_utc": {"type": "string", "format": "date-time"},
    }
    finite_matrix = _matrix_schema(nullable=False)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:zprm:fmb1:zero-perturbation:trial-result:v1.1-r1",
        "title": "FMB1 Zero-Perturbation Mainline v1.1 R1 Trial Result",
        "description": (
            "Future formal result row. T_est/Delta_T are null only when a nonfinite or "
            "infrastructure failure is retained without serializing NaN/Infinity."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
        "x-amendment-id": AMENDMENT_ID,
        "x-track-id": TRACK_ID,
        "x-primary-experimental-unit": "scene",
        "x-snapshot-independent-scene-claim-forbidden": True,
        "x-registration-authority-granted": False,
        "x-real-result-count-at-freeze": 0,
        "x-scientific-failures-retained": True,
        "x-retry-policy": "INFRASTRUCTURE_FAILURE_ONLY",
        "x-reassociation-definition": "src/phase_a_harness/common_association_analysis.py",
        "x-common-invalid-reason-retained": True,
        "x-common-invalid-reason-enum": invalid_reasons,
        "x-common-status-null-only-without-finite-pose": True,
        "x-common-invalid-detail-required-for-other": True,
        "x-physical-reference-semantics": PHYSICAL_REFERENCE_SEMANTICS,
        "allOf": [
            {
                "if": {"properties": {"finite_result": {"const": True}}},
                "then": {
                    "properties": {
                        "T_est": finite_matrix,
                        "Delta_T": finite_matrix,
                        "translation_x_m": {"type": "number"},
                        "translation_y_m": {"type": "number"},
                        "translation_z_m": {"type": "number"},
                        "translation_norm_m": {"type": "number", "minimum": 0.0},
                        "rotation_angle_rad": {"type": "number", "minimum": 0.0},
                        "rotation_angle_deg": {"type": "number", "minimum": 0.0},
                        "common_association_valid": {"type": "boolean"},
                    }
                },
                "else": {
                    "properties": {
                        "T_est": {"type": "null"},
                        "Delta_T": {"type": "null"},
                        "translation_x_m": {"type": "null"},
                        "translation_y_m": {"type": "null"},
                        "translation_z_m": {"type": "null"},
                        "translation_norm_m": {"type": "null"},
                        "rotation_angle_rad": {"type": "null"},
                        "rotation_angle_deg": {"type": "null"},
                        "common_association_valid": {"type": "null"},
                        "common_association_invalid_reason": {"type": "null"},
                        "common_association_invalid_detail": {"type": "null"},
                        "initial_correspondence_count": {"type": "null"},
                        "initial_valid_normal_correspondence_count": {"type": "null"},
                        "final_correspondence_count": {"type": "null"},
                        "final_valid_normal_correspondence_count": {"type": "null"},
                        "correspondence_turnover": {"type": "null"},
                        "accepted_source_turnover": {"type": "null"},
                        "correspondence_count_change_ratio": {"type": "null"},
                        "initial_residual_rmse": {"type": "null"},
                        "final_residual_rmse": {"type": "null"},
                        "residual_rmse_change": {"type": "null"},
                        "median_normal_angle_change_deg": {"type": "null"},
                        "q95_normal_angle_change_deg": {"type": "null"},
                    }
                },
            },
            {
                "if": {
                    "properties": {"common_association_valid": {"const": True}}
                },
                "then": {
                    "properties": {
                        "common_association_invalid_reason": {"type": "null"},
                        "common_association_invalid_detail": {"type": "null"},
                        "initial_correspondence_count": {
                            "type": "integer", "minimum": 0,
                        },
                        "initial_valid_normal_correspondence_count": {
                            "type": "integer", "minimum": 0,
                        },
                        "final_correspondence_count": {
                            "type": "integer", "minimum": 0,
                        },
                        "final_valid_normal_correspondence_count": {
                            "type": "integer", "minimum": 0,
                        },
                        "correspondence_turnover": {
                            "type": "number", "minimum": 0.0, "maximum": 1.0,
                        },
                        "accepted_source_turnover": {
                            "type": "number", "minimum": 0.0, "maximum": 1.0,
                        },
                        "correspondence_count_change_ratio": {
                            "type": "number", "minimum": -1.0,
                        },
                        "initial_residual_rmse": {
                            "type": "number", "minimum": 0.0,
                        },
                        "final_residual_rmse": {
                            "type": "number", "minimum": 0.0,
                        },
                        "residual_rmse_change": {"type": "number"},
                        "median_normal_angle_change_deg": {
                            "type": "number", "minimum": 0.0, "maximum": 180.0,
                        },
                        "q95_normal_angle_change_deg": {
                            "type": "number", "minimum": 0.0, "maximum": 180.0,
                        },
                    }
                },
            },
            {
                "if": {
                    "properties": {"common_association_valid": {"const": False}}
                },
                "then": {
                    "properties": {
                        "common_association_invalid_reason": {"enum": invalid_reasons},
                    }
                },
            },
            {
                "if": {
                    "properties": {
                        "common_association_invalid_reason": {"const": "OTHER"}
                    }
                },
                "then": {
                    "properties": {
                        "common_association_invalid_detail": {
                            "type": "string", "minLength": 1, "maxLength": 1000,
                        }
                    }
                },
            },
        ],
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R1TrialAssetError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise R1TrialAssetError(f"{label} must be finite")
    return number


def _finite_matrix(value: Any, label: str) -> list[list[float]]:
    if type(value) is not list or len(value) != 4:
        raise R1TrialAssetError(f"{label} must be 4x4")
    matrix: list[list[float]] = []
    for row_index, row in enumerate(value):
        if type(row) is not list or len(row) != 4:
            raise R1TrialAssetError(f"{label}[{row_index}] must have four values")
        matrix.append(
            [_finite_number(item, f"{label}[{row_index}]") for item in row]
        )
    if any(abs(matrix[3][index] - expected) > 1e-12 for index, expected in enumerate((0, 0, 0, 1))):
        raise R1TrialAssetError(f"{label} has invalid homogeneous row")
    return matrix


def _nullable_nonnegative_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise R1TrialAssetError(f"{label} must be a nonnegative integer or null")
    return value


def validate_result_row(
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    trial_plan_sha256: str,
    formal_lock_sha256: str,
) -> dict[str, Any]:
    """Fail-closed validation for a future result; it executes no backend."""

    schema = build_result_schema()
    required = set(schema["required"])
    if set(row) != required:
        missing = sorted(required - set(row))
        unknown = sorted(set(row) - required)
        raise R1TrialAssetError(f"result fields mismatch; missing={missing}, unknown={unknown}")
    planned = {item["trial_id"]: item for item in plan.get("rows", [])}
    trial = planned.get(row.get("trial_id"))
    if trial is None:
        raise R1TrialAssetError("trial is not in the frozen R1 plan")
    binding = {
        "batch_id": "batch_id",
        "amendment_id": "amendment_id",
        "track_id": "track_id",
        "scene_id": "scene_id",
        "geometry_class": "final_geometry_class",
        "station_id": "station_id",
        "attempt": "attempt",
        "snapshot_id": "snapshot_id",
        "backend": "backend",
        "backend_version": "backend_version",
        "source_reference": "source_reference",
        "source_sha256": "source_sha256",
        "target_reference": "target_reference",
        "target_sha256": "target_sha256",
        "T0": "T0",
        "T_reference_nominal": "T_reference_nominal",
        "backend_parameter_contract_sha256": "backend_parameter_contract_sha256",
        "backend_canonical_parameter_sha256": "backend_canonical_parameter_sha256",
        "active_amendment_sha256": "active_amendment_sha256",
        "analysis_contract_sha256": "analysis_contract_sha256",
    }
    for result_field, plan_field in binding.items():
        if row[result_field] != trial[plan_field]:
            raise R1TrialAssetError(f"{result_field} does not match frozen plan")
    if row["trial_plan_sha256"] != _require_sha(trial_plan_sha256, "trial plan SHA"):
        raise R1TrialAssetError("trial_plan_sha256 mismatch")
    if row["formal_lock_sha256"] != _require_sha(formal_lock_sha256, "lock SHA"):
        raise R1TrialAssetError("formal_lock_sha256 mismatch")
    if row["schema"] != RESULT_SCHEMA_NAME or row["track_id"] != TRACK_ID:
        raise R1TrialAssetError("result schema/track mismatch")
    if row["execution_kind"] != "FORMAL" or row["fixture_only"] is not False:
        raise R1TrialAssetError("fixture/non-formal row cannot enter R1 formal results")
    if row["physical_reference_semantics"] != PHYSICAL_REFERENCE_SEMANTICS:
        raise R1TrialAssetError("physical reference semantics were altered")
    if not isinstance(row["solver_status"], str) or not row["solver_status"].strip():
        raise R1TrialAssetError("solver_status must be nonempty")
    allowed_scientific = set(
        schema["properties"]["scientific_status"]["enum"]
    )
    if row["scientific_status"] not in allowed_scientific:
        raise R1TrialAssetError("unknown scientific status")
    if not isinstance(row["code_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", row["code_commit"]) is None:
        raise R1TrialAssetError("code_commit must be a full lowercase Git SHA")
    environment = row["environment_identity"]
    if type(environment) is not dict or set(environment) != {
        "python", "numpy", "scipy", "open3d", "pcl", "environment_manifest_sha256"
    }:
        raise R1TrialAssetError("environment identity fields mismatch")
    expected_environment = {
        "python": "3.11.15",
        "numpy": "1.26.4",
        "scipy": "1.11.4",
        "open3d": "0.19.0+b012259",
        "pcl": "1.15.1",
    }
    for name, version in expected_environment.items():
        if environment[name] != version:
            raise R1TrialAssetError(f"environment version mismatch: {name}")
    _require_sha(environment["environment_manifest_sha256"], "environment manifest SHA")
    if not isinstance(row["created_at_utc"], str):
        raise R1TrialAssetError("created_at_utc must be a timestamp")
    try:
        created = datetime.fromisoformat(row["created_at_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise R1TrialAssetError("created_at_utc is invalid") from error
    if created.tzinfo is None or created.utcoffset() is None:
        raise R1TrialAssetError("created_at_utc must carry timezone information")
    _require_identity(row["T0"], "result T0")

    infrastructure = row["infrastructure_status"]
    allowed_infrastructure = set(
        schema["properties"]["infrastructure_status"]["enum"]
    )
    if infrastructure not in allowed_infrastructure:
        raise R1TrialAssetError("unknown infrastructure status")
    if infrastructure == "OK":
        if row["retry_eligible"] is not False or row["retry_reason"] is not None:
            raise R1TrialAssetError("scientific outcomes are not retry eligible")
        if row["scientific_status"] == "NOT_EVALUATED_INFRASTRUCTURE_FAILURE":
            raise R1TrialAssetError("OK infrastructure cannot claim not-evaluated failure")
    else:
        if row["retry_eligible"] is not True or row["retry_reason"] != infrastructure:
            raise R1TrialAssetError("only explicit infrastructure failure permits retry")
        if infrastructure == "PCL_EXECUTABLE_INFRASTRUCTURE_FAILURE" and row["backend"] != "PCL_POINT_TO_PLANE":
            raise R1TrialAssetError("PCL executable failure cannot be assigned to Open3D")
        if row["scientific_status"] != "NOT_EVALUATED_INFRASTRUCTURE_FAILURE":
            raise R1TrialAssetError("infrastructure failure must not invent a scientific outcome")

    finite = row["finite_result"]
    pose_fields = (
        "T_est",
        "Delta_T",
        "translation_x_m",
        "translation_y_m",
        "translation_z_m",
        "translation_norm_m",
        "rotation_angle_rad",
        "rotation_angle_deg",
    )
    if finite is True:
        if infrastructure != "OK":
            raise R1TrialAssetError("infrastructure failure cannot contain an estimated transform")
        if row["scientific_status"] in {
            "NONFINITE_RESULT_RECORDED_WITHOUT_NONFINITE_TRANSFORM",
            "NOT_EVALUATED_INFRASTRUCTURE_FAILURE",
        }:
            raise R1TrialAssetError("finite result has incompatible scientific status")
        estimate = _finite_matrix(row["T_est"], "T_est")
        delta = _finite_matrix(row["Delta_T"], "Delta_T")
        if any(abs(estimate[i][j] - delta[i][j]) > 1e-12 for i in range(4) for j in range(4)):
            raise R1TrialAssetError("Delta_T must equal inverse(T0) @ T_est")
        xyz = tuple(_finite_number(row[field], field) for field in (
            "translation_x_m", "translation_y_m", "translation_z_m"
        ))
        if any(abs(xyz[i] - delta[i][3]) > 1e-12 for i in range(3)):
            raise R1TrialAssetError("translation components do not match Delta_T")
        norm = math.sqrt(sum(value * value for value in xyz))
        if abs(_finite_number(row["translation_norm_m"], "translation_norm_m") - norm) > 1e-12:
            raise R1TrialAssetError("translation norm does not match Delta_T")
        rotation_audit = rotation_metric_audit(
            [row[:3] for row in delta[:3]],
            [row[:3] for row in IDENTITY_4X4[:3]],
        )
        if rotation_audit.get("rotation_matrix_quality_pass") is not True:
            raise R1TrialAssetError("Delta_T rotation failed frozen quality audit")
        angle_rad = _finite_number(
            rotation_audit.get("rotation_error_rad"), "frozen rotation angle"
        )
        if abs(_finite_number(row["rotation_angle_rad"], "rotation_angle_rad") - angle_rad) > 1e-9:
            raise R1TrialAssetError("rotation angle radians do not match Delta_T")
        if abs(_finite_number(row["rotation_angle_deg"], "rotation_angle_deg") - math.degrees(angle_rad)) > 1e-7:
            raise R1TrialAssetError("rotation angle degrees do not match Delta_T")
    elif finite is False:
        if any(row[field] is not None for field in pose_fields):
            raise R1TrialAssetError("nonfinite result must retain no NaN/Inf transform payload")
        expected_status = (
            "NONFINITE_RESULT_RECORDED_WITHOUT_NONFINITE_TRANSFORM"
            if infrastructure == "OK"
            else "NOT_EVALUATED_INFRASTRUCTURE_FAILURE"
        )
        if row["scientific_status"] != expected_status:
            raise R1TrialAssetError("null transform status does not preserve failure semantics")
    else:
        raise R1TrialAssetError("finite_result must be bool")

    common_valid = row["common_association_valid"]
    common_reason = row["common_association_invalid_reason"]
    common_detail = row["common_association_invalid_detail"]
    allowed_common_reasons = {
        "NO_INITIAL_CORRESPONDENCE", "NO_FINAL_CORRESPONDENCE",
        "INSUFFICIENT_VALID_NORMALS", "NONFINITE_COMMON_METRICS", "OTHER",
    }
    if finite is True:
        if type(common_valid) is not bool:
            raise R1TrialAssetError(
                "finite scientific result must retain common-association validity"
            )
        if common_valid:
            if common_reason is not None or common_detail is not None:
                raise R1TrialAssetError(
                    "valid common association cannot carry invalid reason/detail"
                )
        else:
            if common_reason not in allowed_common_reasons:
                raise R1TrialAssetError("invalid common-association reason is not frozen")
            if common_reason == "OTHER" and (
                not isinstance(common_detail, str) or not common_detail.strip()
            ):
                raise R1TrialAssetError(
                    "common-association OTHER reason requires diagnostic detail"
                )
            if common_detail is not None and (
                not isinstance(common_detail, str)
                or not common_detail.strip() or len(common_detail) > 1000
            ):
                raise R1TrialAssetError(
                    "common-association invalid detail must be null or 1..1000 characters"
                )
    elif any(value is not None for value in (common_valid, common_reason, common_detail)):
        raise R1TrialAssetError(
            "common-association status must be null when no finite pose was analyzed"
        )

    count_fields = (
        "initial_correspondence_count",
        "initial_valid_normal_correspondence_count",
        "final_correspondence_count",
        "final_valid_normal_correspondence_count",
    )
    counts = {field: _nullable_nonnegative_integer(row[field], field) for field in count_fields}
    for valid_field, total_field in (
        ("initial_valid_normal_correspondence_count", "initial_correspondence_count"),
        ("final_valid_normal_correspondence_count", "final_correspondence_count"),
    ):
        valid_count = counts[valid_field]
        total_count = counts[total_field]
        if valid_count is not None and total_count is not None and valid_count > total_count:
            raise R1TrialAssetError(f"{valid_field} cannot exceed {total_field}")

    metric_fields = (
        "correspondence_turnover",
        "accepted_source_turnover",
        "correspondence_count_change_ratio",
        "initial_residual_rmse",
        "final_residual_rmse",
        "residual_rmse_change",
        "median_normal_angle_change_deg",
        "q95_normal_angle_change_deg",
    )
    metrics: dict[str, float | None] = {}
    for field in metric_fields:
        if row[field] is not None:
            value = _finite_number(row[field], field)
            metrics[field] = value
            if field in {"correspondence_turnover", "accepted_source_turnover"} and not 0.0 <= value <= 1.0:
                raise R1TrialAssetError(f"{field} must be within [0,1]")
            if field == "correspondence_count_change_ratio" and value < -1.0:
                raise R1TrialAssetError(f"{field} must be >= -1")
            if field in {"initial_residual_rmse", "final_residual_rmse"} and value < 0.0:
                raise R1TrialAssetError(f"{field} must be nonnegative")
            if field in {"median_normal_angle_change_deg", "q95_normal_angle_change_deg"} and not 0.0 <= value <= 180.0:
                raise R1TrialAssetError(f"{field} must be within [0,180]")
        else:
            metrics[field] = None
    initial_count = counts["initial_correspondence_count"]
    final_count = counts["final_correspondence_count"]
    change_ratio = metrics["correspondence_count_change_ratio"]
    if initial_count == 0 and change_ratio is not None:
        raise R1TrialAssetError("count-change ratio must be null for zero initial count")
    if initial_count not in (None, 0) and final_count is not None and change_ratio is not None:
        expected_ratio = (final_count - initial_count) / initial_count
        if abs(change_ratio - expected_ratio) > 1e-12:
            raise R1TrialAssetError("correspondence count-change ratio is inconsistent")
    initial_rmse = metrics["initial_residual_rmse"]
    final_rmse = metrics["final_residual_rmse"]
    residual_change = metrics["residual_rmse_change"]
    if initial_rmse is not None and final_rmse is not None and residual_change is not None:
        if abs(residual_change - (final_rmse - initial_rmse)) > 1e-12:
            raise R1TrialAssetError("residual RMSE change is inconsistent")
    median_angle = metrics["median_normal_angle_change_deg"]
    q95_angle = metrics["q95_normal_angle_change_deg"]
    if median_angle is not None and q95_angle is not None and q95_angle < median_angle:
        raise R1TrialAssetError("q95 normal-angle change cannot be below median")
    if common_valid is True and any(
        value is None for value in (*counts.values(), *metrics.values())
    ):
        raise R1TrialAssetError(
            "valid common association must retain every exposed count/metric"
        )
    if finite is False and any(
        value is not None for value in (*counts.values(), *metrics.values())
    ):
        raise R1TrialAssetError(
            "nonfinite pose cannot carry common-association counts/metrics"
        )
    if infrastructure != "OK":
        association_fields = count_fields + metric_fields + (
            "common_association_valid", "common_association_invalid_reason",
            "common_association_invalid_detail",
        )
        if any(row[field] is not None for field in association_fields):
            raise R1TrialAssetError("infrastructure failure cannot contain invented association metrics")
    return json.loads(json.dumps(dict(row), allow_nan=False))


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return value


def write_trial_assets(
    plan: Mapping[str, Any],
    result_schema: Mapping[str, Any],
    *,
    output_root: Path = FMB1_ROOT,
) -> dict[str, Any]:
    """Write the single authoritative R1 plan and result-schema byte source."""

    output_root.mkdir(parents=True, exist_ok=True)
    canonical_json = output_root / PLAN_JSON_PATH.name
    canonical_csv = output_root / PLAN_CSV_PATH.name
    canonical_schema = output_root / RESULT_SCHEMA_PATH.name

    json_bytes = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    schema_bytes = (json.dumps(result_schema, indent=2, sort_keys=True) + "\n").encode("utf-8")
    canonical_json.write_bytes(json_bytes)
    canonical_schema.write_bytes(schema_bytes)
    with canonical_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in plan["rows"]:
            writer.writerow({field: _csv_value(row[field]) for field in CSV_FIELDS})
    return {
        "schema": "mid360_fmb1_zero_perturbation_trial_assets_v1_1_r1",
        "amendment_id": AMENDMENT_ID,
        "single_authoritative_byte_source": True,
        "trial_plan_json_sha256": sha256_file(canonical_json),
        "trial_plan_csv_sha256": sha256_file(canonical_csv),
        "result_schema_sha256": sha256_file(canonical_schema),
    }


def generate_and_write(*, output_root: Path = FMB1_ROOT) -> dict[str, Any]:
    plan = build_trial_plan()
    schema = build_result_schema()
    assets = write_trial_assets(plan, schema, output_root=output_root)
    return {
        "status": "PASS",
        "amendment_id": AMENDMENT_ID,
        "counts": plan["counts"],
        "trial_plan_sha256": assets["trial_plan_json_sha256"],
        "trial_plan_csv_sha256": assets["trial_plan_csv_sha256"],
        "result_schema_sha256": assets["result_schema_sha256"],
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
        "registration_backend_call_count": 0,
    }
