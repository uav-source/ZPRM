"""Single source of truth for Synthetic Confirmatory v2 execution identity.

This module is metadata-only.  Importing it cannot construct an RNG, generate a
snapshot, execute a backend, or inspect a formal result directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import canonical_json_sha256, file_sha256, write_json


NAMESPACE = "zero_perturbation_synthetic_confirmatory_v2_20260729_ideal_lineage_fix"
GEOMETRY_SEEDS = (1632408808, 2098780325, 2113923543, 143826534, 1099133161)
MEASUREMENT_SEEDS = (740417430, 23963997, 925756274)
BOOTSTRAP_SEED = 146517424
OLD_V1_GEOMETRY_SEEDS = (248284635, 376488233, 198112089, 229684695, 226655024)
OLD_V1_MEASUREMENT_SEEDS = (469989467, 1088311622, 916609326)
OLD_V1_BOOTSTRAP_SEED = 1083684578

SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
CONDITIONS = ("IDEAL_MATCHED", "INDEPENDENT_NOISE_FREE", "FULL_NOISE")
BACKENDS = ("open3d_point_to_plane", "pcl_point_to_plane")

SNAPSHOT_COUNT = 595
TRIAL_COUNT = 1190
FORMAL_RUN_ID = "synthetic-confirmatory-v2"
FORMAL_OUTPUT_DIR = "results/synthetic_confirmatory_v2"
FORMAL_WORKERS = 2
FORMAL_BRANCH = "fix/zero-perturbation-confirmatory-ideal-lineage"
FORMAL_PRERUN_TAG = "archive/zero-perturbation-synthetic-confirmatory-v2-pre-run-pass"
MANIFEST_RELATIVE = Path("frozen_assets/synthetic_confirmatory_formal_manifest_v2.json")
SNAPSHOT_CACHE_ROOT = "data/synthetic_confirmatory_v2_snapshots"
SNAPSHOT_LOCK_RELATIVE = Path("data/synthetic_confirmatory_v2_snapshot_lock.json")
PRERUN_ARTIFACT_RELATIVE = Path("artifacts/synthetic_confirmatory_v2_prerun")
FORMAL_ARTIFACT_RELATIVE = Path("artifacts/synthetic_confirmatory_v2")

PROTOCOL_SCHEMA = "synthetic_confirmatory_protocol_v2"
GATE_SCHEMA = "synthetic_confirmatory_gate_contract_v2"
MANIFEST_SCHEMA = "synthetic_confirmatory_formal_manifest_v2"
SNAPSHOT_SCHEMA = "synthetic_confirmatory_snapshot_v2"
METADATA_SCHEMA = "synthetic_confirmatory_metadata_v2"
LINEAGE_SCHEMA = "synthetic_confirmatory_parent_lineage_v2"
SNAPSHOT_LOCK_SCHEMA = "synthetic_confirmatory_snapshot_lock_v2"
RAW_RESULT_MANIFEST_SCHEMA = "synthetic_confirmatory_raw_result_manifest_v2"
FORMAL_RUN_SCHEMA = "synthetic_confirmatory_formal_run_v2"
DRY_RUN_SCHEMA = "synthetic_confirmatory_dry_run_v2"
PRERUN_SCHEMA = "synthetic_confirmatory_prerun_v2"
FORMAL_ANALYSIS_SCHEMA = "synthetic_confirmatory_primary_analysis_v2"
INDEPENDENT_SCHEMA = "synthetic_confirmatory_independent_verification_v2"
ARTIFACT_SCHEMA = "synthetic_confirmatory_formal_artifact_v2"
ARTIFACT_INVENTORY_VERSION = "7_tables_3_figures_7_root_files_v1"

PROTOCOL_RELATIVE = Path("protocols/synthetic_confirmatory_protocol_v2.json")
PROTOCOL_DOCUMENT_RELATIVE = Path("protocols/synthetic_confirmatory_protocol_v2.md")
GATE_RELATIVE = Path("protocols/synthetic_confirmatory_gate_contract_v2.json")
SNAPSHOT_PLAN_RELATIVE = Path("protocols/synthetic_confirmatory_planned_snapshots_v2.csv")
TRIAL_PLAN_RELATIVE = Path("protocols/synthetic_confirmatory_planned_trials_v2.csv")
SEED_SCHEDULE_RELATIVE = Path("frozen_assets/synthetic_confirmatory_v2_seed_schedule.json")
FROZEN_MODEL_RELATIVE = Path("frozen_assets/confirmatory_development_trained_models_v1.json")
BACKEND_PARAMETER_RELATIVE = Path("frozen_assets/backend_parameter_contract.json")
PCL_CLI_RELATIVE = Path("bin/pcl_point_to_plane_cli")

SNAPSHOT_FIELDS = (
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
    "planned_backend_count",
    "replicate_semantics",
)
TRIAL_FIELDS = (
    "planned_trial_id",
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
    "backend",
)

PUBLISHER_TABLES = (
    "h1_ideal_control.csv",
    "h2_scene_effect.csv",
    "h3_cross_backend_ranking.csv",
    "h4_reassociation.csv",
    "h5_frozen_models.csv",
    "h6_systematic_groups.csv",
    "gate_summary.csv",
)
PUBLISHER_FIGURES = ("gate_matrix.png", "scene_effect.png", "model_comparison.png")
PUBLISHER_ROOT_FILES = (
    "synthetic_confirmatory_report.md",
    "primary_analysis.json",
    "independent_verification.json",
    "final_decision.json",
    "run_manifest.json",
    "SHA256SUMS",
    "artifact_verification.json",
)

BOUND_FILE_PATHS: Mapping[str, str] = {
    "scientific_protocol": PROTOCOL_RELATIVE.as_posix(),
    "scientific_protocol_document": PROTOCOL_DOCUMENT_RELATIVE.as_posix(),
    "gate_contract": GATE_RELATIVE.as_posix(),
    "planned_snapshots": SNAPSHOT_PLAN_RELATIVE.as_posix(),
    "planned_trials": TRIAL_PLAN_RELATIVE.as_posix(),
    "seed_schedule": SEED_SCHEDULE_RELATIVE.as_posix(),
    "frozen_model": FROZEN_MODEL_RELATIVE.as_posix(),
    "backend_parameter_contract": BACKEND_PARAMETER_RELATIVE.as_posix(),
    "pcl_cli": PCL_CLI_RELATIVE.as_posix(),
    "trial_schema": "frozen_assets/trial_result_schema.json",
    "trial_schema_validator": "src/phase_a_harness/phase_a_trial_result_schema.py",
    "trial_result_writer": "src/phase_a_harness/phase_a_trial_result_writer.py",
    "trial_resume": "src/phase_a_harness/phase_a_trial_resume.py",
    "attempt_events": "src/phase_a_harness/phase_a_attempt_events.py",
    "backend_metrics": "src/phase_a_harness/backend_phase_a_metrics.py",
    "backend_types": "src/phase_a_harness/types.py",
    "v2_contract": "src/phase_a_harness/synthetic_confirmatory_v2_contract.py",
    "v2_snapshot_builder": "src/phase_a_harness/synthetic_confirmatory_v2_snapshot_builder.py",
    "v2_runner": "src/phase_a_harness/synthetic_confirmatory_v2_runner.py",
    "v2_analysis": "src/phase_a_harness/synthetic_confirmatory_v2_analysis.py",
    "v2_independent_verifier": "src/phase_a_harness/synthetic_confirmatory_v2_independent_verifier.py",
    "v2_publisher": "src/phase_a_harness/synthetic_confirmatory_v2_publisher.py",
    "v2_artifact_verifier": "src/phase_a_harness/synthetic_confirmatory_v2_artifact_verifier.py",
    "v2_prerun": "src/phase_a_harness/synthetic_confirmatory_v2_prerun.py",
    "v2_qualification": "src/phase_a_harness/synthetic_confirmatory_v2_qualification.py",
    "v2_seed_provenance_audit": "src/phase_a_harness/synthetic_confirmatory_v2_seed_audit.py",
    "v2_runner_script": "scripts/run_synthetic_confirmatory_v2.py",
    "v2_analysis_script": "scripts/analyze_synthetic_confirmatory_v2.py",
    "v2_verifier_script": "scripts/verify_synthetic_confirmatory_v2.py",
    "v2_publisher_script": "scripts/publish_synthetic_confirmatory_v2.py",
    "v2_qualification_script": "scripts/qualify_synthetic_confirmatory_v2.py",
    "v2_asset_freezer_script": "scripts/freeze_synthetic_confirmatory_v2.py",
    "primary_scientific_core": "src/phase_a_harness/synthetic_confirmatory_analysis.py",
    "independent_scientific_core": "src/phase_a_harness/synthetic_confirmatory_independent_verifier.py",
    "frozen_model_inference": "src/phase_a_harness/synthetic_confirmatory_models.py",
    "common_association": "src/phase_a_harness/common_association_analysis.py",
    "rotation_metrics": "src/phase_a_harness/rotation_metrics.py",
    "open3d_adapter": "src/phase_a_harness/open3d_backend.py",
    "pcl_adapter": "src/phase_a_harness/pcl_backend.py",
    "backend_execution": "src/phase_a_harness/phase_a_execution_chain_audit.py",
    "full_synthetic_backend_execution": "src/phase_a_harness/full_synthetic_backend_execution.py",
    "full_synthetic_snapshot_builder": "src/phase_a_harness/full_synthetic_snapshot_builder.py",
    "generator_wrapper": "src/phase_a_harness/phase_b_generator.py",
    "generator": "src/phase_a_harness/phase_b_generator_frozen/capture_range/day2_development_scene.py",
    "phase_a_lineage": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_v1_2.py",
    "generator_capture_init": "src/phase_a_harness/phase_b_generator_frozen/capture_range/__init__.py",
    "generator_capture_protocol": "src/phase_a_harness/phase_b_generator_frozen/capture_range/day2_development_protocol.py",
    "generator_capture_types": "src/phase_a_harness/phase_b_generator_frozen/capture_range/types.py",
    "generator_package_init": "src/phase_a_harness/phase_b_generator_frozen/__init__.py",
    "generator_zero_init": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/__init__.py",
    "generator_zero_protocol": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/protocol.py",
    "generator_zero_snapshot_builder": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/snapshot_builder.py",
    "generator_zero_types": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/types.py",
    "fixture_plan": "frozen_assets/fixtures/fixture_plan.json",
    "fixture_snapshot_lock": "frozen_assets/fixtures/fixture_snapshot_lock.json",
    "fixture_backend_parameter_lock": "frozen_assets/fixtures/fixture_backend_parameter_lock.json",
    "fixture_qualification": "src/phase_a_harness/fixture_qualification.py",
    "fixture_publication": "src/phase_a_harness/fixture_publication.py",
    "fixture_artifact_verifier": "src/phase_a_harness/fixture_publication_artifact_verifier.py",
}


def canonical_identity_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition": str(row["condition"]),
        "geometry_seed": int(row["geometry_seed"]),
        "measurement_seed": (
            None if row.get("measurement_seed") in (None, "")
            else int(row["measurement_seed"])
        ),
        "repeat_index": int(row["repeat_index"]),
        "scene_variant": str(row["scene_variant"]),
    }


def expected_snapshot_id(row: Mapping[str, Any]) -> str:
    return f"synthetic-confirmatory-v2::{canonical_identity_sha256(snapshot_identity(row))}"


def derive_seed(domain: str, index: int) -> int:
    if domain not in {"geometry", "measurement", "bootstrap"}:
        raise ValueError("unauthorized v2 seed domain")
    allowed = {"geometry": range(5), "measurement": range(3), "bootstrap": range(1)}
    if isinstance(index, bool) or not isinstance(index, int) or index not in allowed[domain]:
        raise ValueError("seed index is outside the frozen v2 schedule")
    payload = f"{NAMESPACE}|{domain}|{index}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2147483647
    result = 1 if value == 0 else value
    expected = {
        "geometry": GEOMETRY_SEEDS,
        "measurement": MEASUREMENT_SEEDS,
        "bootstrap": (BOOTSTRAP_SEED,),
    }[domain][index]
    if result != expected:
        raise RuntimeError("v2 seed derivation differs from the frozen schedule")
    return result


def _integer(value: str, label: str) -> int:
    if not value or value.strip() != value:
        raise ValueError(f"{label} is not canonical")
    result = int(value)
    if str(result) != value:
        raise ValueError(f"{label} is not canonical")
    return result


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"v2 CSV schema mismatch: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"v2 CSV row has excess fields: {path}")
    return rows


def typed_snapshot_rows(path: str | Path) -> list[dict[str, Any]]:
    return [
        {
            "planned_snapshot_id": row["planned_snapshot_id"],
            "scene_variant": row["scene_variant"],
            "condition": row["condition"],
            "geometry_seed": _integer(row["geometry_seed"], "geometry_seed"),
            "measurement_seed": (
                None if row["measurement_seed"] == ""
                else _integer(row["measurement_seed"], "measurement_seed")
            ),
            "repeat_index": _integer(row["repeat_index"], "repeat_index"),
            "planned_backend_count": _integer(
                row["planned_backend_count"], "planned_backend_count"
            ),
            "replicate_semantics": row["replicate_semantics"],
        }
        for row in _read_csv(Path(path), SNAPSHOT_FIELDS)
    ]


def typed_trial_rows(path: str | Path) -> list[dict[str, Any]]:
    return [
        {
            "planned_trial_id": row["planned_trial_id"],
            "planned_snapshot_id": row["planned_snapshot_id"],
            "scene_variant": row["scene_variant"],
            "condition": row["condition"],
            "geometry_seed": _integer(row["geometry_seed"], "geometry_seed"),
            "measurement_seed": (
                None if row["measurement_seed"] == ""
                else _integer(row["measurement_seed"], "measurement_seed")
            ),
            "repeat_index": _integer(row["repeat_index"], "repeat_index"),
            "backend": row["backend"],
        }
        for row in _read_csv(Path(path), TRIAL_FIELDS)
    ]


def validate_snapshot_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    required = set(SNAPSHOT_FIELDS)
    if not required.issubset(row):
        raise ValueError("v2 snapshot plan row is incomplete")
    value = {
        "planned_snapshot_id": str(row["planned_snapshot_id"]),
        **snapshot_identity(row),
        "planned_backend_count": int(row["planned_backend_count"]),
        "replicate_semantics": str(row["replicate_semantics"]),
    }
    condition = value["condition"]
    measurement = value["measurement_seed"]
    repeat = value["repeat_index"]
    if (
        value["scene_variant"] not in SCENES
        or condition not in CONDITIONS
        or value["geometry_seed"] not in GEOMETRY_SEEDS
        or value["planned_backend_count"] != 2
        or value["planned_snapshot_id"] != expected_snapshot_id(value)
    ):
        raise ValueError("v2 snapshot identity is outside the frozen design")
    if condition == "IDEAL_MATCHED":
        expected_semantics = "ONE_CONTROL_INPUT"
        valid = measurement is None and repeat == 0
    elif condition == "INDEPENDENT_NOISE_FREE":
        expected_semantics = "ONE_DETERMINISTIC_INDEPENDENT_INPUT"
        valid = measurement is None and repeat == 0
    else:
        expected_semantics = "FIFTEEN_STOCHASTIC_INPUTS_PER_SCENE_GEOMETRY"
        valid = measurement in MEASUREMENT_SEEDS and repeat in range(5)
    if not valid or value["replicate_semantics"] != expected_semantics:
        raise ValueError("v2 snapshot replicate semantics changed")
    return value


def audit_v2_plan(snapshot_path: str | Path, trial_path: str | Path) -> dict[str, Any]:
    snapshots = typed_snapshot_rows(snapshot_path)
    trials = typed_trial_rows(trial_path)
    snapshot_errors = 0
    for row in snapshots:
        try:
            validate_snapshot_plan_row(row)
        except (TypeError, ValueError):
            snapshot_errors += 1
    snapshot_ids = [row["planned_snapshot_id"] for row in snapshots]
    trial_ids = [row["planned_trial_id"] for row in trials]
    by_snapshot = {row["planned_snapshot_id"]: row for row in snapshots}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trial_errors = 0
    for trial in trials:
        groups[trial["planned_snapshot_id"]].append(trial)
        parent = by_snapshot.get(trial["planned_snapshot_id"])
        if (
            parent is None
            or trial["backend"] not in BACKENDS
            or trial["planned_trial_id"]
            != f"{trial['planned_snapshot_id']}::{trial['backend']}"
            or any(
                trial[name] != parent[name]
                for name in (
                    "scene_variant", "condition", "geometry_seed",
                    "measurement_seed", "repeat_index",
                )
            )
        ):
            trial_errors += 1
    pairing = sum(
        len(groups.get(snapshot_id, ())) != 2
        or {row["backend"] for row in groups.get(snapshot_id, ())} != set(BACKENDS)
        for snapshot_id in snapshot_ids
    ) + sum(snapshot_id not in by_snapshot for snapshot_id in groups)
    condition_counts = Counter(row["condition"] for row in snapshots)
    backend_counts = Counter(row["backend"] for row in trials)
    duplicate_snapshots = len(snapshot_ids) - len(set(snapshot_ids))
    duplicate_trials = len(trial_ids) - len(set(trial_ids))
    independent_pseudoreplication = sum(
        row["measurement_seed"] is not None or row["repeat_index"] != 0
        for row in snapshots if row["condition"] == "INDEPENDENT_NOISE_FREE"
    )
    count_pass = bool(
        len(snapshots) == SNAPSHOT_COUNT
        and len(trials) == TRIAL_COUNT
        and condition_counts == Counter(
            {"IDEAL_MATCHED": 35, "INDEPENDENT_NOISE_FREE": 35, "FULL_NOISE": 525}
        )
        and backend_counts == Counter({BACKENDS[0]: 595, BACKENDS[1]: 595})
        and snapshot_errors == 0
        and trial_errors == 0
    )
    return {
        "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS": independent_pseudoreplication == 0,
        "CONFIRMATORY_PLAN_COUNT_PASS": count_pass,
        "CONFIRMATORY_PLAN_PAIRING_PASS": pairing == 0,
        "CONFIRMATORY_PLAN_UNIQUENESS_PASS": duplicate_snapshots == duplicate_trials == 0,
        "backend_trial_counts": dict(sorted(backend_counts.items())),
        "condition_snapshot_counts": dict(sorted(condition_counts.items())),
        "duplicate_snapshot_count": duplicate_snapshots,
        "duplicate_trial_count": duplicate_trials,
        "independent_pseudoreplication_plan_count": independent_pseudoreplication,
        "native_trial_count": sum(name not in BACKENDS for name in backend_counts.elements()),
        "pairing_violation_count": pairing,
        "planned_snapshot_count": len(snapshots),
        "planned_snapshot_identity_sha256": canonical_identity_sha256(snapshots),
        "planned_snapshot_unique_count": len(set(snapshot_ids)),
        "planned_trial_count": len(trials),
        "planned_trial_identity_sha256": canonical_identity_sha256(trials),
        "planned_trial_unique_count": len(set(trial_ids)),
        "semantic_violation_count": snapshot_errors + trial_errors,
    }


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON root must be object: {path}")
    return value


def manifest_payload(root: str | Path, *, authorized: bool) -> dict[str, Any]:
    if type(authorized) is not bool:
        raise TypeError("authorized must be bool")
    repository = Path(root).resolve()
    missing = [relative for relative in BOUND_FILE_PATHS.values() if not (repository / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"v2 manifest binding missing: {missing}")
    protocol = _strict_object(repository / PROTOCOL_RELATIVE)
    parameter_contract = _strict_object(repository / BACKEND_PARAMETER_RELATIVE)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("planned_snapshot_count") != SNAPSHOT_COUNT
        or protocol.get("planned_trial_count") != TRIAL_COUNT
        or protocol.get("geometry_seeds") != list(GEOMETRY_SEEDS)
        or protocol.get("measurement_seeds") != list(MEASUREMENT_SEEDS)
        or protocol.get("bootstrap_seed") != BOOTSTRAP_SEED
    ):
        raise ValueError("v2 protocol identity changed")
    bound = {
        name: {"path": relative, "sha256": file_sha256(repository / relative)}
        for name, relative in BOUND_FILE_PATHS.items()
    }
    payload: dict[str, Any] = {
        "backend_count": 2,
        "artifact_inventory_version": ARTIFACT_INVENTORY_VERSION,
        "artifact_schema": ARTIFACT_SCHEMA,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bound_files": bound,
        "formal_branch": FORMAL_BRANCH,
        "formal_execution_authorized": authorized,
        "formal_output_dir": FORMAL_OUTPUT_DIR,
        "formal_pre_run_tag": FORMAL_PRERUN_TAG,
        "formal_run_id": FORMAL_RUN_ID,
        "formal_workers": FORMAL_WORKERS,
        "manifest_schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA,
        "manifest_version": "2",
        "geometry_seeds": list(GEOMETRY_SEEDS),
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "native_trial_count": 0,
        "open3d_parameter_sha256": parameter_contract["open3d"]["canonical_sha256"],
        "open3d_version": parameter_contract["open3d"]["parameters"]["version"],
        "pcl_parameter_sha256": parameter_contract["pcl"]["canonical_sha256"],
        "pcl_version": parameter_contract["pcl"]["parameters"]["version"],
        "planned_snapshot_count": SNAPSHOT_COUNT,
        "planned_trial_count": TRIAL_COUNT,
        "raw_result_manifest_schema": RAW_RESULT_MANIFEST_SCHEMA,
        "seed_namespace": NAMESPACE,
        "seed_schedule_payload_sha256": protocol["seed_schedule_payload_sha256"],
        "snapshot_cache_root": SNAPSHOT_CACHE_ROOT,
        "snapshot_lock_path": SNAPSHOT_LOCK_RELATIVE.as_posix(),
    }
    for name in (
        "scientific_protocol", "scientific_protocol_document", "gate_contract",
        "planned_snapshots", "planned_trials", "seed_schedule", "frozen_model",
    ):
        payload[f"{name}_path"] = bound[name]["path"]
        payload[f"{name}_sha256"] = bound[name]["sha256"]
    return payload


def signed_manifest(root: str | Path, *, authorized: bool) -> dict[str, Any]:
    payload = manifest_payload(root, authorized=authorized)
    return {**payload, "manifest_payload_sha256": canonical_json_sha256(payload)}


def verify_manifest(path: str | Path, *, require_authorized: bool) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).resolve()
    repository = manifest_path.parent.parent.resolve()
    if manifest_path != (repository / MANIFEST_RELATIVE).resolve():
        raise ValueError("v2 runner requires the exact v2 manifest path")
    value = _strict_object(manifest_path)
    authorization = value.get("formal_execution_authorized")
    if type(authorization) is not bool:
        raise ValueError("v2 manifest authorization must be bool")
    if value != signed_manifest(repository, authorized=authorization):
        raise ValueError("v2 manifest differs from exact live bindings")
    if require_authorized and authorization is not True:
        raise PermissionError("v2 formal execution is not authorized")
    return repository, value


def write_manifest(root: str | Path, *, authorized: bool, replace: bool = False) -> dict[str, Any]:
    repository = Path(root).resolve()
    path = repository / MANIFEST_RELATIVE
    if path.exists() and not replace:
        raise FileExistsError("refusing to replace v2 manifest")
    if path.exists() and replace:
        current = _strict_object(path)
        if current.get("formal_execution_authorized") is True:
            raise PermissionError("refusing to rewrite an authorized v2 manifest")
    if authorized:
        raise PermissionError("use authorize_manifest_once for the true transition")
    value = signed_manifest(repository, authorized=authorized)
    write_json(path, value)
    return value


def authorize_manifest_once(
    root: str | Path, *, qualification_decision: Mapping[str, Any]
) -> dict[str, Any]:
    """Perform the sole false-to-true v2 authorization transition.

    The pre-run artifact is verified while the manifest is still false.  No
    formal cache/result may exist at the transition boundary.
    """

    repository = Path(root).resolve()
    path = repository / MANIFEST_RELATIVE
    _verified_root, current = verify_manifest(path, require_authorized=False)
    if current.get("formal_execution_authorized") is not False:
        raise PermissionError("v2 manifest is not in its one initial false state")
    required = {
        "SYNTHETIC_CONFIRMATORY_V2_PRE_RUN_QUALIFICATION_PASS": True,
        "CONFIRMATORY_V2_RUN_AUTHORIZED": True,
        "SYNTHETIC_CONFIRMATORY_V2_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V2_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_V2_PASS": "NOT_EVALUATED",
        "REAL_DATA_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
    }
    if type(qualification_decision) is not dict or any(
        qualification_decision.get(name) != expected
        for name, expected in required.items()
    ):
        raise PermissionError("v2 pre-run decision does not authorize the transition")
    if (
        (repository / SNAPSHOT_CACHE_ROOT).exists()
        or (repository / SNAPSHOT_LOCK_RELATIVE).exists()
        or (repository / FORMAL_OUTPUT_DIR).exists()
    ):
        raise PermissionError("formal v2 runtime state exists before authorization")
    from .synthetic_confirmatory_v2_artifact_verifier import (
        verify_v2_prerun_artifact,
    )

    artifact = repository / PRERUN_ARTIFACT_RELATIVE
    verification = verify_v2_prerun_artifact(
        artifact, manifest_path=path, write_report=False
    )
    if verification.get("V2_ARTIFACT_VERIFICATION_PASS") is not True:
        raise PermissionError("v2 pre-run artifact did not verify")
    authorized = signed_manifest(repository, authorized=True)
    write_json(path, authorized)
    _final_root, final = verify_manifest(path, require_authorized=True)
    if _final_root != repository or final != authorized:
        raise ValueError("v2 manifest authorization write did not round-trip")
    final_verification = verify_v2_prerun_artifact(
        artifact, manifest_path=path, write_report=False
    )
    if final_verification.get("V2_ARTIFACT_VERIFICATION_PASS") is not True:
        raise PermissionError("authorized v2 manifest no longer matches pre-run artifact")
    return final


__all__ = [name for name in globals() if name.isupper()] + [
    "audit_v2_plan", "canonical_identity_sha256", "derive_seed",
    "expected_snapshot_id", "manifest_payload", "signed_manifest",
    "snapshot_identity", "typed_snapshot_rows", "typed_trial_rows",
    "validate_snapshot_plan_row", "verify_manifest", "write_manifest",
    "authorize_manifest_once",
]
