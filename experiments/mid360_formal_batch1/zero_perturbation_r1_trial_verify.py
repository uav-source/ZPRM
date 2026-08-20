"""Independent fail-closed verifier for FMB1 zero-perturbation R1 trial assets.

The verifier intentionally does not import the producer module or any
registration backend.  Its expectations are reconstructed directly from the
frozen final-dataset manifests and immutable dependency bytes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[2]
EXPERIMENT = REPOSITORY / "experiments/mid360_formal_batch1"
FINAL = REPOSITORY / "results/mid360_formal_batch1/final_dataset_v1"

AMENDMENT_ID = "FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1"
TRACK_ID = "ZERO_PERTURBATION_TRACK"
PLAN_SCHEMA = "mid360_fmb1_zero_perturbation_trial_plan_v1_1_r1"
PLAN_ID = "FMB1_ZERO_PERTURBATION_TRIAL_PLAN_V1_1_R1"
RESULT_SCHEMA = "mid360_fmb1_zero_perturbation_trial_result_v1_1_r1"
PHYSICAL_REFERENCE = "NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT"
IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
SCENES = {
    "FMB1_R01": "RICH",
    "FMB1_R02": "RICH",
    "FMB1_R03": "RICH",
    "FMB1_W01": "WEAK",
    "FMB1_W02": "WEAK",
    "FMB1_W03": "WEAK",
}
STATIONS = {"S01", "S02", "S03"}
BACKENDS = {
    "OPEN3D_POINT_TO_PLANE": ("O3D", "0.19.0+b012259", "open3d"),
    "PCL_POINT_TO_PLANE": ("PCL", "1.15.1", "pcl"),
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

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


class R1TrialVerificationError(ValueError):
    """Raised on the first independently observed integrity violation."""


def _fail(message: str) -> None:
    raise R1TrialVerificationError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        _fail(f"expected object: {path}")
    return value


def _csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def _csv_objects(path: Path) -> list[dict[str, str]]:
    return _csv(path)[1]


def _expect_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        _fail(f"{label} is not SHA256")
    return value


def _attempt(scene: str) -> int:
    return 2 if scene == "FMB1_W02" else 1


def _trial_id(snapshot: str, backend: str) -> str:
    return f"FMB1-ZP11R1-{snapshot.replace('_', '-')}-{BACKENDS[backend][0]}"


def _compact(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _dependency_paths(repository: Path) -> dict[str, Path]:
    experiment = repository / "experiments/mid360_formal_batch1"
    final = repository / "results/mid360_formal_batch1/final_dataset_v1"
    return {
        "snapshot": final / "final_snapshot_manifest.csv",
        "target": final / "final_target_manifest.csv",
        "backend": repository / "frozen_assets/backend_parameter_contract.json",
        "amendment": experiment / "amendments/zero_perturbation_mainline_v1_1_r1.json",
        "analysis": experiment / "amendments/zero_perturbation_analysis_contract_v1_1_r1.json",
        "plan_json": experiment / "zero_perturbation_trial_plan_v1_1.json",
        "plan_csv": experiment / "zero_perturbation_trial_plan_v1_1.csv",
        "schema": experiment / "zero_perturbation_trial_result_schema_v1_1.json",
    }


def verify_plan_payload(
    plan: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, str]],
    targets: Sequence[Mapping[str, str]],
    *,
    snapshot_manifest_sha256: str,
    target_manifest_sha256: str,
    backend_contract: Mapping[str, Any],
    backend_contract_sha256: str,
    amendment_sha256: str,
    analysis_contract_sha256: str,
    verify_payload_files: bool = False,
) -> dict[str, Any]:
    """Independently compare a plan object with frozen manifest truth."""

    if plan.get("schema") != PLAN_SCHEMA:
        _fail("plan schema mismatch")
    if plan.get("plan_id") != PLAN_ID:
        _fail("plan ID mismatch")
    if plan.get("amendment_id") != AMENDMENT_ID or plan.get("track_id") != TRACK_ID:
        _fail("plan amendment/track mismatch")
    if plan.get("physical_reference_semantics") != PHYSICAL_REFERENCE:
        _fail("physical reference semantics mismatch")
    if plan.get("primary_experimental_unit") != "scene":
        _fail("snapshot pseudo-replication detected")
    if plan.get("snapshot_independence_claimed") is not False:
        _fail("snapshots cannot be declared independent scenes")
    if plan.get("FORMAL_REGISTRATION_AUTHORIZED") is not False:
        _fail("plan unexpectedly authorizes registration")
    if any(plan.get(field) != 0 for field in (
        "actual_open3d_trials", "actual_pcl_trials", "actual_formal_trials"
    )):
        _fail("plan contains executed formal trials")
    if plan.get("NO_REGISTRATION_BACKEND_IMPORTED_OR_CALLED") is not True:
        _fail("no-backend attestation missing")

    generated = plan.get("generated_from")
    bindings = plan.get("bindings")
    if type(generated) is not dict or type(bindings) is not dict:
        _fail("plan lineage objects missing")
    expected_generated = {
        "snapshot_manifest": "results/mid360_formal_batch1/final_dataset_v1/final_snapshot_manifest.csv",
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "target_manifest": "results/mid360_formal_batch1/final_dataset_v1/final_target_manifest.csv",
        "target_manifest_sha256": target_manifest_sha256,
    }
    if generated != expected_generated:
        _fail("frozen manifest lineage mismatch")
    expected_bindings = {
        "backend_parameter_contract": "frozen_assets/backend_parameter_contract.json",
        "backend_parameter_contract_sha256": backend_contract_sha256,
        "active_amendment": "experiments/mid360_formal_batch1/amendments/zero_perturbation_mainline_v1_1_r1.json",
        "active_amendment_sha256": amendment_sha256,
        "analysis_contract": "experiments/mid360_formal_batch1/amendments/zero_perturbation_analysis_contract_v1_1_r1.json",
        "analysis_contract_sha256": analysis_contract_sha256,
    }
    if bindings != expected_bindings:
        _fail("R1 dependency binding mismatch")

    if len(snapshots) != 180 or len(targets) != 18:
        _fail("frozen manifest inventory is not 180 snapshots/18 targets")
    target_by_key: dict[tuple[str, str], Mapping[str, str]] = {}
    for target in targets:
        key = (target["scene_id"], target["station_id"])
        if key in target_by_key:
            _fail("duplicate frozen target")
        if key[0] not in SCENES or key[1] not in STATIONS:
            _fail("target outside final six-scene dataset")
        if int(target["attempt"]) != _attempt(key[0]):
            _fail("target attempt lineage mismatch")
        if int(target["query_contribution_to_target"]) != 0:
            _fail("QUERY entered a target")
        target_by_key[key] = target
    if len(target_by_key) != 18:
        _fail("target scene/station inventory incomplete")

    expected_by_trial: dict[str, dict[str, Any]] = {}
    seen_snapshots: set[str] = set()
    station_snapshot_count: dict[tuple[str, str], int] = {}
    file_hash_cache: dict[str, str] = {}
    backend_canonical: dict[str, str] = {}
    for backend, (_, version, contract_key) in BACKENDS.items():
        entry = backend_contract.get(contract_key)
        if type(entry) is not dict:
            _fail(f"missing {contract_key} contract")
        if entry.get("parameters", {}).get("version") != version:
            _fail(f"{contract_key} version mismatch")
        backend_canonical[backend] = _expect_sha(
            entry.get("canonical_sha256"), f"{contract_key} canonical SHA"
        )

    for snapshot in snapshots:
        scene = snapshot["scene_id"]
        station = snapshot["station_id"]
        snapshot_id = snapshot["snapshot_id"]
        if scene not in SCENES or station not in STATIONS or "W04" in snapshot_id:
            _fail("retired/unknown scene entered snapshot source")
        if int(snapshot["attempt"]) != _attempt(scene):
            _fail("old W02 attempt or wrong attempt entered plan source")
        if snapshot_id in seen_snapshots:
            _fail("duplicate frozen snapshot")
        seen_snapshots.add(snapshot_id)
        key = (scene, station)
        station_snapshot_count[key] = station_snapshot_count.get(key, 0) + 1
        target = target_by_key.get(key)
        if target is None:
            _fail("snapshot has no station target")
        if snapshot["target_npy_sha256"] != target["target_npy_sha256"]:
            _fail("snapshot target SHA disagrees with target manifest")
        if snapshot["target_array_sha256"] != target["target_array_sha256"]:
            _fail("snapshot target array SHA disagrees with target manifest")

        for backend, (_, version, _) in BACKENDS.items():
            trial_id = _trial_id(snapshot_id, backend)
            expected_by_trial[trial_id] = {
                "trial_id": trial_id,
                "batch_id": "FMB1",
                "amendment_id": AMENDMENT_ID,
                "track_id": TRACK_ID,
                "scene_id": scene,
                "final_geometry_class": SCENES[scene],
                "station_id": station,
                "attempt": _attempt(scene),
                "snapshot_id": snapshot_id,
                "snapshot_quantile": float(snapshot["quantile"]),
                "query_timestamp": float(snapshot["query_timestamp"]),
                "backend": backend,
                "backend_version": version,
                "backend_canonical_parameter_sha256": backend_canonical[backend],
                "source_reference": snapshot["source_path"],
                "source_sha256": _expect_sha(snapshot["source_npy_sha256"], "source SHA"),
                "source_array_sha256": _expect_sha(
                    snapshot["source_array_sha256"], "source array SHA"
                ),
                "source_point_count": int(snapshot["source_point_count"]),
                "target_reference": target["target_path"],
                "target_sha256": _expect_sha(target["target_npy_sha256"], "target SHA"),
                "target_array_sha256": _expect_sha(
                    target["target_array_sha256"], "target array SHA"
                ),
                "target_point_count": int(target["target_point_count"]),
                "T0": IDENTITY,
                "T_reference_nominal": IDENTITY,
                "translation_perturbation_m": 0.0,
                "rotation_perturbation_deg": 0.0,
                "backend_parameter_contract_sha256": backend_contract_sha256,
                "active_amendment_sha256": amendment_sha256,
                "analysis_contract_sha256": analysis_contract_sha256,
                "planned_status": "PLANNED_NOT_AUTHORIZED_NOT_EXECUTED",
            }
        if verify_payload_files:
            for path_text, expected_sha in (
                (snapshot["source_path"], snapshot["source_npy_sha256"]),
                (target["target_path"], target["target_npy_sha256"]),
            ):
                if path_text not in file_hash_cache:
                    file_hash_cache[path_text] = _sha(Path(path_text))
                if file_hash_cache[path_text] != expected_sha:
                    _fail(f"payload hash mismatch: {path_text}")

    if set(station_snapshot_count.values()) != {10} or len(station_snapshot_count) != 18:
        _fail("each final station must contain ten snapshots")
    rows = plan.get("rows")
    if type(rows) is not list or len(rows) != 360:
        _fail("plan must contain exactly 360 rows")
    observed_ids: set[str] = set()
    for row in rows:
        if type(row) is not dict:
            _fail("plan row is not an object")
        if set(row) != set(CSV_FIELDS):
            _fail("plan row fields changed")
        trial_id = row.get("trial_id")
        if trial_id in observed_ids:
            _fail("duplicate trial id")
        observed_ids.add(trial_id)
        expected = expected_by_trial.get(trial_id)
        if expected is None or row != expected:
            _fail(f"trial differs from frozen expectation: {trial_id}")
        if row["T0"] != IDENTITY or row["T_reference_nominal"] != IDENTITY:
            _fail("non-Identity initialization/reference")
        if row["translation_perturbation_m"] != 0.0 or row["rotation_perturbation_deg"] != 0.0:
            _fail("capture-radius perturbation entered zero track")
    if observed_ids != set(expected_by_trial):
        _fail("trial plan has missing or extra rows")

    counts = plan.get("counts")
    expected_counts = {
        "scene_count": 6,
        "station_count": 18,
        "snapshot_count": 180,
        "rich_snapshot_count": 90,
        "weak_snapshot_count": 90,
        "open3d_trial_count": 180,
        "pcl_trial_count": 180,
        "native_trial_count": 0,
        "total_trial_count": 360,
        "duplicate_trial_count": 0,
        "missing_trial_count": 0,
    }
    if counts != expected_counts:
        _fail("declared plan counts differ from independently observed inventory")
    serialized = json.dumps(plan, sort_keys=True)
    if "W04" in serialized or "CAPTURE_RADIUS" in serialized:
        _fail("retired W04/capture-radius content entered zero plan")
    return {
        "status": "PASS",
        "scene_count": 6,
        "station_count": 18,
        "snapshot_count": 180,
        "open3d_trial_count": 180,
        "pcl_trial_count": 180,
        "total_trial_count": 360,
        "identity_t0_count": 360,
        "w02_attempt2_trial_count": 60,
        "w04_trial_count": 0,
        "old_w02_attempt1_trial_count": 0,
        "actual_formal_trials": 0,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "registration_backend_call_count": 0,
    }


def verify_result_schema_payload(schema: Mapping[str, Any]) -> None:
    if schema.get("additionalProperties") is not False:
        _fail("result schema is not fail-closed")
    properties = schema.get("properties")
    required = schema.get("required")
    if type(properties) is not dict or type(required) is not list:
        _fail("result schema fields missing")
    if set(properties) != set(required):
        _fail("every result property must be required")
    constants = {
        "schema": RESULT_SCHEMA,
        "batch_id": "FMB1",
        "amendment_id": AMENDMENT_ID,
        "track_id": TRACK_ID,
        "T0": IDENTITY,
        "T_reference_nominal": IDENTITY,
        "physical_reference_semantics": PHYSICAL_REFERENCE,
        "execution_kind": "FORMAL",
        "fixture_only": False,
    }
    for field, value in constants.items():
        if properties.get(field, {}).get("const") != value:
            _fail(f"result schema constant changed: {field}")
    if set(properties.get("scene_id", {}).get("enum", [])) != set(SCENES):
        _fail("result schema scenes changed")
    if set(properties.get("backend", {}).get("enum", [])) != set(BACKENDS):
        _fail("result schema backends changed")
    common_fields = {
        "initial_correspondence_count",
        "final_correspondence_count",
        "initial_valid_normal_correspondence_count",
        "final_valid_normal_correspondence_count",
        "correspondence_turnover",
        "accepted_source_turnover",
        "correspondence_count_change_ratio",
        "initial_residual_rmse",
        "final_residual_rmse",
        "residual_rmse_change",
        "median_normal_angle_change_deg",
        "q95_normal_angle_change_deg",
        "common_association_valid",
        "common_association_invalid_reason",
        "common_association_invalid_detail",
    }
    if not common_fields <= set(properties):
        _fail("result schema omits frozen common reassociation fields")
    if schema.get("x-primary-experimental-unit") != "scene":
        _fail("result schema hierarchy changed")
    if schema.get("x-registration-authority-granted") is not False:
        _fail("result schema grants registration authority")
    if schema.get("x-real-result-count-at-freeze") != 0:
        _fail("result schema freeze contains real results")
    if schema.get("x-reassociation-definition") != "src/phase_a_harness/common_association_analysis.py":
        _fail("result schema changed common reassociation definition")
    expected_reasons = {
        "NO_INITIAL_CORRESPONDENCE", "NO_FINAL_CORRESPONDENCE",
        "INSUFFICIENT_VALID_NORMALS", "NONFINITE_COMMON_METRICS", "OTHER",
    }
    if (schema.get("x-common-invalid-reason-retained") is not True
            or set(schema.get("x-common-invalid-reason-enum", [])) != expected_reasons
            or schema.get("x-common-status-null-only-without-finite-pose") is not True
            or schema.get("x-common-invalid-detail-required-for-other") is not True):
        _fail("result schema does not preserve frozen common invalid status")
    if "W04" in json.dumps(schema, sort_keys=True):
        _fail("result schema references retired W04")


def verify_assets(*, repository: Path = REPOSITORY, verify_payload_files: bool = True) -> dict[str, Any]:
    paths = _dependency_paths(repository)
    for path in paths.values():
        if not path.is_file():
            _fail(f"required R1 trial asset missing: {path}")
    snapshots = _csv_objects(paths["snapshot"])
    targets = _csv_objects(paths["target"])
    backend = _json(paths["backend"])
    amendment = _json(paths["amendment"])
    analysis = _json(paths["analysis"])
    amendment_identity = amendment.get("amendment_id", amendment.get("AMENDMENT_ID"))
    analysis_identity = analysis.get("amendment_id", analysis.get("AMENDMENT_ID"))
    if amendment_identity != AMENDMENT_ID or analysis_identity != AMENDMENT_ID:
        _fail("R1 dependency identity mismatch")
    experimental_units = analysis.get("experimental_units")
    if type(experimental_units) is not dict or experimental_units.get("primary_experimental_unit") != "scene":
        _fail("analysis contract does not use scene as primary unit")
    amendment_phase = (amendment.get("status"), amendment.get("activation_effective"))
    analysis_phase = (analysis.get("status"), analysis.get("activation_effective"))
    allowed_phases = {
        ("CANDIDATE_PENDING_INDEPENDENT_ACTIVATION_VERIFIER", False),
        ("ACTIVE", True),
    }
    if amendment_phase not in allowed_phases or analysis_phase not in allowed_phases:
        _fail("R1 status/activation phase is invalid")
    if amendment_phase != analysis_phase:
        _fail("R1 amendment/analysis activation phases differ")
    for document in (amendment, analysis):
        if document.get("FORMAL_REGISTRATION_AUTHORIZED") is not False:
            _fail("R1 dependency authorizes registration")
        if document.get("actual_formal_trials") != 0:
            _fail("R1 dependency was not frozen at trial count zero")

    plan = _json(paths["plan_json"])
    report = verify_plan_payload(
        plan,
        snapshots,
        targets,
        snapshot_manifest_sha256=_sha(paths["snapshot"]),
        target_manifest_sha256=_sha(paths["target"]),
        backend_contract=backend,
        backend_contract_sha256=_sha(paths["backend"]),
        amendment_sha256=_sha(paths["amendment"]),
        analysis_contract_sha256=_sha(paths["analysis"]),
        verify_payload_files=verify_payload_files,
    )

    fieldnames, csv_rows = _csv(paths["plan_csv"])
    if tuple(fieldnames) != CSV_FIELDS or len(csv_rows) != 360:
        _fail("canonical CSV header/count mismatch")
    for json_row, csv_row in zip(plan["rows"], csv_rows):
        expected_csv = {field: _compact(json_row[field]) for field in CSV_FIELDS}
        if csv_row != expected_csv:
            _fail(f"CSV/JSON row mismatch: {json_row['trial_id']}")

    schema = _json(paths["schema"])
    verify_result_schema_payload(schema)
    report.update(
        {
            "plan_id": PLAN_ID,
            "trial_plan_json_sha256": _sha(paths["plan_json"]),
            "trial_plan_csv_sha256": _sha(paths["plan_csv"]),
            "result_schema_sha256": _sha(paths["schema"]),
            "active_amendment_sha256": _sha(paths["amendment"]),
            "analysis_contract_sha256": _sha(paths["analysis"]),
            "backend_parameter_contract_sha256": _sha(paths["backend"]),
            "verifier_implementation": "experiments/mid360_formal_batch1/zero_perturbation_r1_trial_verify.py",
            "verifier_implementation_sha256": _sha(
                repository
                / "experiments/mid360_formal_batch1/zero_perturbation_r1_trial_verify.py"
            ),
            "single_authoritative_byte_source": True,
            "payload_byte_hashes_verified": verify_payload_files,
            "source_payload_count": 180,
            "target_payload_count": 18,
        }
    )
    return report
