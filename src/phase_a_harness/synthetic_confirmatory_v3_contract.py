"""Declarative contract for Synthetic Confirmatory v3.

This module is the single source of execution identity for v3.  It is
deliberately metadata-only: importing it cannot initialize an RNG, construct a
snapshot, create a runtime directory, execute a backend, or read a formal
result.  The only permitted seed operations are deterministic declaration and
plan-identity validation with :mod:`hashlib`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import canonical_json_sha256, file_sha256, write_json


RUNTIME_LIFECYCLE_BASELINE_COMMIT = "b35acb88bf8de9df6486f91a4ab59069ec5ef945"
DERIVATION_PREREQUISITE_COMMIT = "9bd944002cff7b7f6eb22b7f636d081f6a7211eb"
NAMESPACE = (
    "zero_perturbation_synthetic_confirmatory_v3_"
    "20260730_runtime_lifecycle_qualified"
)


def _declared_seed_value(domain: str, index: int) -> int:
    payload = f"{NAMESPACE}|{domain}|{index}".encode("utf-8")
    value = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False
    ) % 2147483647
    return 1 if value == 0 else value


GEOMETRY_SEEDS = tuple(_declared_seed_value("geometry", index) for index in range(5))
MEASUREMENT_SEEDS = tuple(
    _declared_seed_value("measurement", index) for index in range(3)
)
BOOTSTRAP_SEED = _declared_seed_value("bootstrap", 0)

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
CONDITION_SNAPSHOT_COUNTS = {
    "IDEAL_MATCHED": 35,
    "INDEPENDENT_NOISE_FREE": 35,
    "FULL_NOISE": 525,
}
BACKEND_TRIAL_COUNTS = {
    "open3d_point_to_plane": 595,
    "pcl_point_to_plane": 595,
}

FORMAL_RUN_ID = "synthetic-confirmatory-v3"
FORMAL_WORKERS = 2
FORMAL_BRANCH = "fix/zero-perturbation-v3-qualification-json-native-r3"
FORMAL_PRERUN_TAG = (
    "archive/zero-perturbation-synthetic-confirmatory-v3-"
    "bootstrap-repair-r3-pre-run-pass"
)
FORMAL_RUNTIME_ROOT = Path(
    "/home/lj/ZPRM/zero_perturbation_runtime/confirmatory/synthetic_confirmatory_v3"
)
QUALIFICATION_RUNTIME_ROOT = Path(
    "/home/lj/ZPRM/zero_perturbation_runtime/qualification/"
    "v3_bootstrap_state_machine_requalification_v3"
)
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
RUNTIME_ARCHIVE_ROOT = Path("/home/lj/ZPRM/zero_perturbation_runtime_archive")
V2_FAILURE_ARCHIVE_ROOT = (
    RUNTIME_ARCHIVE_ROOT
    / "synthetic_confirmatory_v2_runtime_lifecycle_failure_20260730"
)

RUNTIME_PATHS: Mapping[str, Path] = {
    "snapshot_cache_path": FORMAL_RUNTIME_ROOT / "snapshot_cache",
    "snapshot_lock_path": FORMAL_RUNTIME_ROOT / "snapshot_lock.json",
    "raw_results_path": FORMAL_RUNTIME_ROOT / "raw_results",
    "event_log_path": FORMAL_RUNTIME_ROOT / "event_logs",
    "backend_temporary_path": FORMAL_RUNTIME_ROOT / "backend_tmp",
    "analysis_path": FORMAL_RUNTIME_ROOT / "analysis",
    "verification_path": FORMAL_RUNTIME_ROOT / "verification",
    "publisher_staging_path": FORMAL_RUNTIME_ROOT / "publisher_staging",
    "artifact_staging_path": FORMAL_RUNTIME_ROOT / "artifact_staging",
    "temporary_inventory_path": FORMAL_RUNTIME_ROOT / "working_inventory",
}

PROTOCOL_SCHEMA = "synthetic_confirmatory_protocol_v3"
GATE_SCHEMA = "synthetic_confirmatory_gate_contract_v3"
MANIFEST_SCHEMA = "synthetic_confirmatory_formal_manifest_v3_bootstrap_repair_r3"
SEED_SCHEDULE_SCHEMA = "synthetic_confirmatory_v3_seed_schedule_v1"
EXECUTION_PROFILE_SCHEMA = (
    "synthetic_confirmatory_v3_execution_profile_bootstrap_repair_r3"
)
SNAPSHOT_SCHEMA = "synthetic_confirmatory_snapshot_v3"
METADATA_SCHEMA = "synthetic_confirmatory_metadata_v3"
LINEAGE_SCHEMA = "synthetic_confirmatory_parent_lineage_v2"
SNAPSHOT_LOCK_SCHEMA = "synthetic_confirmatory_snapshot_lock_v3"
RAW_RESULT_MANIFEST_SCHEMA = "synthetic_confirmatory_raw_result_manifest_v3"
FORMAL_RUN_SCHEMA = "synthetic_confirmatory_formal_run_v3"
DRY_RUN_SCHEMA = "synthetic_confirmatory_dry_run_v3"
FORMAL_ANALYSIS_SCHEMA = "synthetic_confirmatory_primary_analysis_v3"
INDEPENDENT_SCHEMA = "synthetic_confirmatory_independent_verification_v3"
ARTIFACT_SCHEMA = "synthetic_confirmatory_formal_artifact_v3"
FINAL_DECISION_SCHEMA = "synthetic_confirmatory_v3_final_decision_v1"
ARTIFACT_INVENTORY_VERSION = "7_tables_3_figures_7_root_files_v1"

MANIFEST_RELATIVE = Path(
    "frozen_assets/synthetic_confirmatory_formal_manifest_v3_bootstrap_repair_r3.json"
)
PROTOCOL_RELATIVE = Path("protocols/synthetic_confirmatory_protocol_v3.json")
PROTOCOL_DOCUMENT_RELATIVE = Path("protocols/synthetic_confirmatory_protocol_v3.md")
GATE_RELATIVE = Path("protocols/synthetic_confirmatory_gate_contract_v3.json")
V2_GATE_RELATIVE = Path("protocols/synthetic_confirmatory_gate_contract_v2.json")
SNAPSHOT_PLAN_RELATIVE = Path(
    "protocols/synthetic_confirmatory_planned_snapshots_v3.csv"
)
TRIAL_PLAN_RELATIVE = Path("protocols/synthetic_confirmatory_planned_trials_v3.csv")
SEED_SCHEDULE_RELATIVE = Path(
    "frozen_assets/synthetic_confirmatory_v3_seed_schedule.json"
)
EXECUTION_PROFILE_RELATIVE = Path(
    "frozen_assets/synthetic_confirmatory_v3_execution_profile_bootstrap_repair_r3.json"
)
FROZEN_MODEL_RELATIVE = Path(
    "frozen_assets/confirmatory_development_trained_models_v1.json"
)
BACKEND_PARAMETER_RELATIVE = Path("frozen_assets/backend_parameter_contract.json")
PCL_CLI_RELATIVE = Path("bin/pcl_point_to_plane_cli")
SCIENTIFIC_CORE_AUTHORITY_RELATIVE = Path(
    "artifacts/runtime_lifecycle_qualification_v1/scientific_core_binding.json"
)
RUNTIME_CORE_AUTHORITY_RELATIVE = Path(
    "artifacts/runtime_lifecycle_qualification_v1/implementation_manifest.json"
)
RUNTIME_QUALIFICATION_DECISION_RELATIVE = Path(
    "artifacts/runtime_lifecycle_qualification_v1/final_decision.json"
)
V1_FAILURE_BINDING_RELATIVE = Path(
    "artifacts/synthetic_confirmatory_v2_prerun/v1_failure_binding.json"
)
V1_SEED_RETIREMENT_RELATIVE = Path(
    "artifacts/synthetic_confirmatory_v2_prerun/old_seed_retirement.json"
)
V2_FAILURE_BINDING_RELATIVE = Path(
    "artifacts/runtime_lifecycle_qualification_v1/v2_failure_binding.json"
)
V2_FAILURE_ARCHIVE_VERIFICATION_RELATIVE = Path(
    "artifacts/runtime_lifecycle_qualification_v1/v2_failure_archive_verification.json"
)
V2_SEED_RETIREMENT_RELATIVE = Path(
    "artifacts/runtime_lifecycle_qualification_v1/v2_seed_retirement.json"
)
PRE_RUN_FINAL_DECISION_RELATIVE = Path(
    "artifacts/synthetic_confirmatory_v3_bootstrap_repair_r3_prerun/"
    "final_decision.json"
)

FROZEN_MODEL_SHA256 = (
    "99806f83d3a0d2393ee7e36e8c06f14a1a8a44fb8d02b074ebc2d9fe750d0872"
)
V2_FAILURE_ARCHIVE_TAR_SHA256 = (
    "5979ed924d21afc43bbbcb12267fda471c67ed642ebafd71167883ed0ee34176"
)

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
PUBLISHER_FIGURES = (
    "gate_matrix.png",
    "scene_effect.png",
    "model_comparison.png",
)
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
    "v2_scientific_gate_source": V2_GATE_RELATIVE.as_posix(),
    "planned_snapshots": SNAPSHOT_PLAN_RELATIVE.as_posix(),
    "planned_trials": TRIAL_PLAN_RELATIVE.as_posix(),
    "seed_schedule": SEED_SCHEDULE_RELATIVE.as_posix(),
    "execution_profile": EXECUTION_PROFILE_RELATIVE.as_posix(),
    "frozen_model": FROZEN_MODEL_RELATIVE.as_posix(),
    "backend_parameter_contract": BACKEND_PARAMETER_RELATIVE.as_posix(),
    "pcl_cli": PCL_CLI_RELATIVE.as_posix(),
    "v3_contract": "src/phase_a_harness/synthetic_confirmatory_v3_contract.py",
    "v3_seed_audit": "src/phase_a_harness/synthetic_confirmatory_v3_seed_audit.py",
    "v3_adapter": "src/phase_a_harness/synthetic_confirmatory_v3_adapter.py",
    "v3_prerun": "src/phase_a_harness/synthetic_confirmatory_v3_prerun.py",
    "v3_runner": "src/phase_a_harness/synthetic_confirmatory_v3_runner.py",
    "formal_runtime_state_machine": (
        "src/phase_a_harness/formal_runtime_state_machine.py"
    ),
    "v3_snapshot_builder": (
        "src/phase_a_harness/synthetic_confirmatory_v3_snapshot_builder.py"
    ),
    "v3_prerun_artifact_verifier": (
        "src/phase_a_harness/synthetic_confirmatory_v3_artifact_verifier.py"
    ),
    "v3_run_cli": "scripts/run_synthetic_confirmatory_v3.py",
    "v3_preflight_cli": "scripts/preflight_synthetic_confirmatory_v3.py",
    "v3_bootstrap_cli": "scripts/bootstrap_synthetic_confirmatory_v3.py",
    "v3_analysis_cli": "scripts/analyze_synthetic_confirmatory_v3.py",
    "v3_completeness_cli": (
        "scripts/check_synthetic_confirmatory_v3_completeness.py"
    ),
    "v3_independent_verifier_cli": "scripts/verify_synthetic_confirmatory_v3.py",
    "v3_difference_audit_cli": (
        "scripts/audit_synthetic_confirmatory_v3_difference.py"
    ),
    "v3_publisher_cli": "scripts/publish_synthetic_confirmatory_v3.py",
    "v3_formal_artifact_verifier_cli": (
        "scripts/verify_synthetic_confirmatory_v3_artifact.py"
    ),
    "v3_adapter_fixture_qualifier": (
        "scripts/qualify_synthetic_confirmatory_v3_adapter_fixture.py"
    ),
    "v3_prerun_qualifier": "scripts/qualify_synthetic_confirmatory_v3_prerun.py",
    "v3_asset_freezer": "scripts/freeze_synthetic_confirmatory_v3.py",
    "v3_bootstrap_repair_freezer": (
        "scripts/freeze_synthetic_confirmatory_v3_bootstrap_repair.py"
    ),
    "v3_bootstrap_repair_qualifier": (
        "scripts/qualify_synthetic_confirmatory_v3_bootstrap_repair.py"
    ),
    "v3_bootstrap_repair_artifact_verifier": (
        "scripts/verify_synthetic_confirmatory_v3_bootstrap_repair_artifact.py"
    ),
    "v3_bootstrap_repair_artifact_verifier_core": (
        "src/phase_a_harness/"
        "synthetic_confirmatory_v3_bootstrap_repair_artifact_verifier.py"
    ),
    "v3_bootstrap_repair_artifact_builder": (
        "scripts/build_synthetic_confirmatory_v3_bootstrap_repair_prerun.py"
    ),
    "v3_fixture_publisher_envelope_adapter": (
        "src/phase_a_harness/"
        "synthetic_confirmatory_v3_fixture_publisher_envelope_adapter.py"
    ),
    "v3_fixture_publisher_envelope_adapter_tests": (
        "tests/test_synthetic_confirmatory_v3_fixture_publisher_envelope_adapter.py"
    ),
    "v3_bootstrap_repair_r2_qualifier": (
        "scripts/qualify_synthetic_confirmatory_v3_bootstrap_repair_r2.py"
    ),
    "v3_bootstrap_repair_r2_freezer": (
        "scripts/freeze_synthetic_confirmatory_v3_bootstrap_repair_r2.py"
    ),
    "v3_bootstrap_repair_r2_artifact_builder": (
        "scripts/build_synthetic_confirmatory_v3_bootstrap_repair_r2_prerun.py"
    ),
    "v3_bootstrap_repair_r2_artifact_verifier": (
        "scripts/verify_synthetic_confirmatory_v3_bootstrap_repair_r2_artifact.py"
    ),
    "v3_bootstrap_repair_r2_artifact_verifier_core": (
        "src/phase_a_harness/"
        "synthetic_confirmatory_v3_bootstrap_repair_r2_artifact_verifier.py"
    ),
    "qualification_json_native": (
        "src/phase_a_harness/qualification_json_native.py"
    ),
    "qualification_json_native_tests": (
        "tests/test_qualification_json_native.py"
    ),
    "qualification_final_aggregate_tests": (
        "tests/test_qualification_final_aggregate.py"
    ),
    "v3_bootstrap_repair_r3_qualifier": (
        "scripts/qualify_synthetic_confirmatory_v3_bootstrap_repair_r3.py"
    ),
    "v3_bootstrap_repair_r3_freezer": (
        "scripts/freeze_synthetic_confirmatory_v3_bootstrap_repair_r3.py"
    ),
    "v3_bootstrap_repair_r3_artifact_builder": (
        "scripts/build_synthetic_confirmatory_v3_bootstrap_repair_r3_prerun.py"
    ),
    "v3_bootstrap_repair_r3_artifact_verifier": (
        "scripts/verify_synthetic_confirmatory_v3_bootstrap_repair_r3_artifact.py"
    ),
    "v3_bootstrap_repair_r3_artifact_verifier_core": (
        "src/phase_a_harness/"
        "synthetic_confirmatory_v3_bootstrap_repair_r3_artifact_verifier.py"
    ),
    "r2_json_failure_manifest": (
        "artifacts/synthetic_confirmatory_v3_bootstrap_repair_r2_json_failure/"
        "MANIFEST.csv"
    ),
    "r2_json_failure_sha256sums": (
        "artifacts/synthetic_confirmatory_v3_bootstrap_repair_r2_json_failure/"
        "SHA256SUMS"
    ),
    "v3_adapter_tests": "tests/test_synthetic_confirmatory_v3_adapter.py",
    "v3_prerun_tests": "tests/test_synthetic_confirmatory_v3_prerun.py",
    "v3_execution_tests": "tests/test_synthetic_confirmatory_v3_execution.py",
    "v3_bootstrap_state_machine_tests": (
        "tests/test_synthetic_confirmatory_v3_bootstrap_state_machine.py"
    ),
    "old_v3_bootstrap_contract_invalidation": (
        "artifacts/synthetic_confirmatory_v3_bootstrap_contract_invalidation/"
        "final_decision.json"
    ),
    "runtime_path_policy": "src/phase_a_harness/runtime_path_policy.py",
    "runtime_git_gate": "src/phase_a_harness/runtime_git_gate.py",
    "runtime_lifecycle_io": "src/phase_a_harness/runtime_lifecycle_io.py",
    "runtime_lifecycle_fixture": "src/phase_a_harness/runtime_lifecycle_fixture.py",
    "primary_scientific_core": (
        "src/phase_a_harness/synthetic_confirmatory_analysis.py"
    ),
    "independent_scientific_core": (
        "src/phase_a_harness/synthetic_confirmatory_independent_verifier.py"
    ),
    "frozen_model_inference": (
        "src/phase_a_harness/synthetic_confirmatory_models.py"
    ),
    "formal_publisher_core": (
        "src/phase_a_harness/synthetic_confirmatory_v2_publisher.py"
    ),
    "formal_artifact_verifier_core": (
        "src/phase_a_harness/synthetic_confirmatory_v2_artifact_verifier.py"
    ),
    "scientific_core_authority": SCIENTIFIC_CORE_AUTHORITY_RELATIVE.as_posix(),
    "runtime_core_authority": RUNTIME_CORE_AUTHORITY_RELATIVE.as_posix(),
    "runtime_qualification_decision": (
        RUNTIME_QUALIFICATION_DECISION_RELATIVE.as_posix()
    ),
    "v1_failure_binding": V1_FAILURE_BINDING_RELATIVE.as_posix(),
    "v1_seed_retirement": V1_SEED_RETIREMENT_RELATIVE.as_posix(),
    "v2_failure_binding": V2_FAILURE_BINDING_RELATIVE.as_posix(),
    "v2_failure_archive_verification": (
        V2_FAILURE_ARCHIVE_VERIFICATION_RELATIVE.as_posix()
    ),
    "v2_seed_retirement": V2_SEED_RETIREMENT_RELATIVE.as_posix(),
}


def canonical_identity_sha256(value: Any) -> str:
    """Hash strict canonical JSON with the repository's terminating newline."""

    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_seed(domain: str, index: int) -> int:
    """Recompute one frozen declaration without creating a random generator."""

    allowed = {"geometry": range(5), "measurement": range(3), "bootstrap": range(1)}
    if domain not in allowed:
        raise ValueError("unauthorized v3 seed domain")
    if isinstance(index, bool) or not isinstance(index, int) or index not in allowed[domain]:
        raise ValueError("seed index is outside the frozen v3 schedule")
    payload = f"{NAMESPACE}|{domain}|{index}".encode("utf-8")
    value = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False
    ) % 2147483647
    return 1 if value == 0 else value


def snapshot_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition": str(row["condition"]),
        "geometry_seed": int(row["geometry_seed"]),
        "measurement_seed": (
            None
            if row.get("measurement_seed") in (None, "")
            else int(row["measurement_seed"])
        ),
        "repeat_index": int(row["repeat_index"]),
        "scene_variant": str(row["scene_variant"]),
    }


def expected_snapshot_id(row: Mapping[str, Any]) -> str:
    return f"synthetic-confirmatory-v3::{canonical_identity_sha256(snapshot_identity(row))}"


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
            raise ValueError(f"v3 CSV schema mismatch: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"v3 CSV row has excess fields: {path}")
    return rows


def typed_snapshot_rows(path: str | Path) -> list[dict[str, Any]]:
    return [
        {
            "planned_snapshot_id": row["planned_snapshot_id"],
            "scene_variant": row["scene_variant"],
            "condition": row["condition"],
            "geometry_seed": _integer(row["geometry_seed"], "geometry_seed"),
            "measurement_seed": (
                None
                if row["measurement_seed"] == ""
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
                None
                if row["measurement_seed"] == ""
                else _integer(row["measurement_seed"], "measurement_seed")
            ),
            "repeat_index": _integer(row["repeat_index"], "repeat_index"),
            "backend": row["backend"],
        }
        for row in _read_csv(Path(path), TRIAL_FIELDS)
    ]


def _validate_formal_snapshot_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The byte-for-byte scientific rules of the original formal validator."""

    if not set(SNAPSHOT_FIELDS).issubset(row):
        raise ValueError("v3 snapshot plan row is incomplete")
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
        raise ValueError("v3 snapshot identity is outside the frozen design")
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
        raise ValueError("v3 snapshot replicate semantics changed")
    return value


def _validate_formal_trial_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one typed formal trial row at the execution boundary."""

    if set(row) != set(TRIAL_FIELDS):
        raise ValueError("v3 trial plan row schema changed")
    value = {name: row[name] for name in TRIAL_FIELDS}
    if (
        type(value["planned_trial_id"]) is not str
        or not value["planned_trial_id"]
        or type(value["planned_snapshot_id"]) is not str
        or not value["planned_snapshot_id"]
        or type(value["scene_variant"]) is not str
        or type(value["condition"]) is not str
        or type(value["geometry_seed"]) is not int
        or (
            value["measurement_seed"] is not None
            and type(value["measurement_seed"]) is not int
        )
        or type(value["repeat_index"]) is not int
        or type(value["backend"]) is not str
    ):
        raise ValueError("v3 trial plan row types changed")
    if (
        value["scene_variant"] not in SCENES
        or value["condition"] not in CONDITIONS
        or value["geometry_seed"] not in GEOMETRY_SEEDS
        or value["backend"] not in BACKENDS
        or value["planned_trial_id"]
        != f"{value['planned_snapshot_id']}::{value['backend']}"
    ):
        raise ValueError("v3 trial identity is outside the frozen design")
    condition = value["condition"]
    measurement = value["measurement_seed"]
    repeat = value["repeat_index"]
    if condition in {"IDEAL_MATCHED", "INDEPENDENT_NOISE_FREE"}:
        valid = measurement is None and repeat == 0
    else:
        valid = measurement in MEASUREMENT_SEEDS and repeat in range(5)
    if not valid:
        raise ValueError("v3 trial realization semantics changed")
    return value


def formal_execution_context() -> Any:
    """Create the one explicit context authorized for the frozen v3 route."""

    from .execution_context import (
        BackendPolicy,
        ExecutionContext,
        ExecutionContract,
        ExecutionMode,
        IdPolicy,
        SchemaBinding,
        SeedPolicy,
        SnapshotReaderPolicy,
    )
    from .synthetic_confirmatory_v3_snapshot_builder import (
        METADATA_FIELDS,
        SNAPSHOT_BUILDER_CONTRACT_VERSION,
        _LOCK_ENTRY_FIELDS,
    )

    mode = ExecutionMode.FORMAL
    snapshot_schema = SchemaBinding(
        mode,
        "synthetic_confirmatory_v3_formal_snapshot_plan_row_v1",
        SNAPSHOT_FIELDS,
        allow_extra_fields=True,
    )
    trial_schema = SchemaBinding(
        mode,
        "synthetic_confirmatory_v3_formal_trial_plan_row_v1",
        TRIAL_FIELDS,
    )
    seed_policy = SeedPolicy(
        mode,
        "synthetic_confirmatory_v3_declared_seed_policy_v1",
        True,
    )
    id_policy = IdPolicy(
        mode,
        "synthetic_confirmatory_v3_canonical_identity_sha256_v1",
    )
    backend_policy = BackendPolicy(
        mode,
        "synthetic_confirmatory_v3_dual_backend_policy_v1",
        BACKENDS,
        2,
    )
    reader_policy = SnapshotReaderPolicy(
        mode,
        "synthetic_confirmatory_v3_formal_reader_policy_v1",
        tuple(sorted(METADATA_FIELDS)),
        tuple(sorted(_LOCK_ENTRY_FIELDS)),
        METADATA_SCHEMA,
        SNAPSHOT_SCHEMA,
        LINEAGE_SCHEMA,
        NAMESPACE,
        SNAPSHOT_BUILDER_CONTRACT_VERSION,
        ("IDEAL_MATCHED",),
        {
            "IDEAL_MATCHED": 0,
            "INDEPENDENT_NOISE_FREE": 0,
            "FULL_NOISE": 3,
        },
    )
    plan_id = (
        "synthetic-confirmatory-v3-plan:"
        "dbe75e9df8545b1c61c61bac2baa3ee2c525aeda65b9db10215c61863b3b29ef:"
        "f5b2f6fb84e3a64e686eacfc22c8ec90d6bc9c14860116e8b51f4d628a77b8d4"
    )
    contract = ExecutionContract(
        mode=mode,
        contract_id="synthetic_confirmatory_v3_formal_execution_contract_v1",
        plan_id=plan_id,
        expected_snapshot_plan_sha256=(
            "138df4d7530f51b1a289d9a370c383153733d8d03c00d2340046e86575038919"
        ),
        expected_trial_plan_sha256=(
            "a4e110b2f7d2801643fa3b726ba6c708a8ef0d24e3d4bdf8577323c4f6c14dfa"
        ),
        expected_snapshot_count=595,
        expected_trial_count=1190,
        expected_cache_root=RUNTIME_PATHS["snapshot_cache_path"],
        expected_runtime_root=FORMAL_RUNTIME_ROOT,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=SCENES,
        allowed_conditions=CONDITIONS,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
        snapshot_validator=_validate_formal_snapshot_plan_row,
        trial_validator=_validate_formal_trial_plan_row,
        result_route="FORMAL_CONDITION_ROUTING_V1",
    )
    return ExecutionContext(
        mode=mode,
        contract=contract,
        plan_id=plan_id,
        cache_root=RUNTIME_PATHS["snapshot_cache_path"],
        runtime_root=FORMAL_RUNTIME_ROOT,
        snapshot_schema=snapshot_schema,
        trial_schema=trial_schema,
        allowed_scenes=SCENES,
        allowed_conditions=CONDITIONS,
        allowed_seed_policy=seed_policy,
        id_policy=id_policy,
        backend_policy=backend_policy,
        snapshot_reader_policy=reader_policy,
    )


def validate_snapshot_plan_row(
    row: Mapping[str, Any], *, execution_context: Any
) -> dict[str, Any]:
    """Validate through an explicit formal or qualification context."""

    return execution_context.validate_snapshot_row(row)


def validate_trial_plan_row(
    row: Mapping[str, Any], *, execution_context: Any
) -> dict[str, Any]:
    return execution_context.validate_trial_row(row)


def validate_formal_snapshot_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit formal compatibility wrapper used by declarative plan audits."""

    return validate_snapshot_plan_row(row, execution_context=formal_execution_context())


def audit_v3_plan(snapshot_path: str | Path, trial_path: str | Path) -> dict[str, Any]:
    """Validate identity, count, pairing, and anti-pseudoreplication invariants."""

    snapshots = typed_snapshot_rows(snapshot_path)
    trials = typed_trial_rows(trial_path)
    snapshot_errors = 0
    for row in snapshots:
        try:
            validate_formal_snapshot_plan_row(row)
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
                    "scene_variant",
                    "condition",
                    "geometry_seed",
                    "measurement_seed",
                    "repeat_index",
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
    independent = [
        row for row in snapshots if row["condition"] == "INDEPENDENT_NOISE_FREE"
    ]
    independent_pseudoreplication = sum(
        row["measurement_seed"] is not None or row["repeat_index"] != 0
        for row in independent
    )
    independent_duplicate_science_identity = len(independent) - len(
        {
            (row["scene_variant"], row["geometry_seed"])
            for row in independent
        }
    )
    count_pass = bool(
        len(snapshots) == SNAPSHOT_COUNT
        and len(trials) == TRIAL_COUNT
        and condition_counts == Counter(CONDITION_SNAPSHOT_COUNTS)
        and backend_counts == Counter(BACKEND_TRIAL_COUNTS)
        and snapshot_errors == 0
        and trial_errors == 0
    )
    return {
        "V3_PLAN_PASS": bool(
            count_pass
            and pairing == 0
            and duplicate_snapshots == 0
            and duplicate_trials == 0
            and independent_pseudoreplication == 0
            and independent_duplicate_science_identity == 0
        ),
        "CONFIRMATORY_PLAN_COUNT_PASS": count_pass,
        "CONFIRMATORY_PLAN_PAIRING_PASS": pairing == 0,
        "CONFIRMATORY_PLAN_UNIQUENESS_PASS": (
            duplicate_snapshots == duplicate_trials == 0
        ),
        "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS": (
            independent_pseudoreplication == 0
            and independent_duplicate_science_identity == 0
        ),
        "backend_trial_counts": dict(sorted(backend_counts.items())),
        "condition_snapshot_counts": dict(sorted(condition_counts.items())),
        "duplicate_snapshot_count": duplicate_snapshots,
        "duplicate_trial_count": duplicate_trials,
        "independent_duplicate_science_identity_count": (
            independent_duplicate_science_identity
        ),
        "independent_pseudoreplication_plan_count": independent_pseudoreplication,
        "native_trial_count": sum(
            count for name, count in backend_counts.items() if name not in BACKENDS
        ),
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
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _verify_payload_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    if type(value) is not dict or type(value.get(field)) is not str:
        raise ValueError(f"{label} payload digest is missing")
    payload = {name: item for name, item in value.items() if name != field}
    if value[field] != canonical_identity_sha256(payload):
        raise ValueError(f"{label} payload digest mismatch")


def _scientific_gate_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: item
        for name, item in value.items()
        if name not in {"schema_version", "gate_contract_payload_sha256"}
    }


def _symlink_components(path: Path) -> list[str]:
    current = Path(path.anchor)
    result: list[str] = []
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            result.append(str(current))
    return result


def audit_formal_runtime_paths(repository: str | Path) -> dict[str, Any]:
    """Audit the exact formal layout without creating or opening it."""

    root = Path(repository).resolve()
    protected = (
        root,
        SOURCE_REPOSITORY.resolve(strict=False),
        RUNTIME_ARCHIVE_ROOT.resolve(strict=False),
        QUALIFICATION_RUNTIME_ROOT.resolve(strict=False),
    )
    paths = {"runtime_root": FORMAL_RUNTIME_ROOT, **RUNTIME_PATHS}
    rows: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        canonical = path.resolve(strict=False)
        inside_run = path == FORMAL_RUNTIME_ROOT or FORMAL_RUNTIME_ROOT in path.parents
        overlap = [
            str(item)
            for item in protected
            if path == item or item in path.parents or path in item.parents
        ]
        rows[name] = {
            "absolute": path.is_absolute(),
            "canonical": path == canonical,
            "inside_formal_runtime_root": inside_run,
            "path": str(path),
            "protected_overlap_paths": overlap,
            "symlink_components": _symlink_components(path),
        }
    children = list(RUNTIME_PATHS.values())
    child_overlap_count = sum(
        first == second or first in second.parents or second in first.parents
        for index, first in enumerate(children)
        for second in children[index + 1 :]
    )
    passed = bool(
        all(row["absolute"] for row in rows.values())
        and all(row["canonical"] for row in rows.values())
        and all(row["inside_formal_runtime_root"] for row in rows.values())
        and all(not row["protected_overlap_paths"] for row in rows.values())
        and all(not row["symlink_components"] for row in rows.values())
        and child_overlap_count == 0
        and FORMAL_RUNTIME_ROOT
        == Path(
            "/home/lj/ZPRM/zero_perturbation_runtime/confirmatory/"
            "synthetic_confirmatory_v3"
        )
    )
    return {
        "V3_RUNTIME_PATH_POLICY_PASS": passed,
        "all_paths_absolute": all(row["absolute"] for row in rows.values()),
        "all_paths_canonical": all(row["canonical"] for row in rows.values()),
        "all_paths_without_symlink_components": all(
            not row["symlink_components"] for row in rows.values()
        ),
        "child_path_overlap_count": child_overlap_count,
        "formal_runtime_root_exists": FORMAL_RUNTIME_ROOT.exists(),
        "formal_runtime_root_outside_repository": not (
            root == FORMAL_RUNTIME_ROOT
            or root in FORMAL_RUNTIME_ROOT.parents
            or FORMAL_RUNTIME_ROOT in root.parents
        ),
        "path_checks": rows,
        "protected_overlap_count": sum(
            len(row["protected_overlap_paths"]) for row in rows.values()
        ),
        "schema_version": "synthetic_confirmatory_v3_runtime_path_audit_v1",
        "symlink_component_count": sum(
            len(row["symlink_components"]) for row in rows.values()
        ),
    }


def _scientific_core_binding(repository: Path) -> dict[str, Any]:
    path = repository / SCIENTIFIC_CORE_AUTHORITY_RELATIVE
    authority = _strict_object(path)
    if (
        authority.get("SCIENTIFIC_CORE_FILE_CHANGE_COUNT") != 0
        or authority.get("H1_H6_SEMANTICS_CHANGE_COUNT") != 0
        or authority.get("FROZEN_MODEL_CHANGE_COUNT") != 0
        or authority.get("BACKEND_BINDING_CHANGE_COUNT") != 0
    ):
        raise ValueError("scientific core authority is not a zero-change binding")
    file_rows = authority.get("file_bindings")
    ast_rows = authority.get("ast_bindings")
    if type(file_rows) is not list or type(ast_rows) is not list:
        raise ValueError("scientific core authority inventory is malformed")
    normalized_files = []
    for row in file_rows:
        if type(row) is not dict:
            raise ValueError("scientific core file binding is malformed")
        relative = str(row["path"])
        expected = str(row["expected_sha256"])
        if row.get("match") is not True or file_sha256(repository / relative) != expected:
            raise ValueError(f"scientific core binding changed: {relative}")
        normalized_files.append(
            {"binding": str(row["binding"]), "path": relative, "sha256": expected}
        )
    normalized_ast = []
    for row in ast_rows:
        if type(row) is not dict or row.get("match") is not True:
            raise ValueError("scientific core AST authority is malformed")
        normalized_ast.append(
            {
                "binding": str(row["binding"]),
                "function": str(row["function"]),
                "path": str(row["path"]),
                "sha256": str(row["expected_ast_sha256"]),
            }
        )
    return {
        "SCIENTIFIC_CORE_FILE_CHANGE_COUNT": 0,
        "authority_path": SCIENTIFIC_CORE_AUTHORITY_RELATIVE.as_posix(),
        "authority_sha256": file_sha256(path),
        "ast_binding_count": len(normalized_ast),
        "ast_binding_set_sha256": canonical_identity_sha256(normalized_ast),
        "file_binding_count": len(normalized_files),
        "file_binding_set_sha256": canonical_identity_sha256(normalized_files),
    }


def _runtime_core_binding(repository: Path) -> dict[str, Any]:
    path = repository / RUNTIME_CORE_AUTHORITY_RELATIVE
    authority = _strict_object(path)
    files = authority.get("files")
    if type(files) is not dict or not files:
        raise ValueError("runtime lifecycle core authority inventory is malformed")
    authorized_repair_paths = {
        "src/phase_a_harness/runtime_lifecycle_fixture.py",
        "src/phase_a_harness/runtime_lifecycle_io.py",
    }
    normalized: dict[str, str] = {}
    authorized_changes: list[dict[str, str]] = []
    for relative, expected in sorted(files.items()):
        if type(relative) is not str or type(expected) is not str:
            raise ValueError("runtime lifecycle core binding is malformed")
        actual = file_sha256(repository / relative)
        if actual != expected and relative not in authorized_repair_paths:
            raise ValueError(f"runtime lifecycle core binding changed: {relative}")
        if actual != expected:
            authorized_changes.append(
                {
                    "path": relative,
                    "baseline_sha256": expected,
                    "repair_sha256": actual,
                }
            )
        normalized[relative] = actual
    execution_files = {
        relative: file_sha256(repository / relative)
        for relative in (
            "src/phase_a_harness/runtime_path_policy.py",
            "src/phase_a_harness/runtime_git_gate.py",
            "src/phase_a_harness/runtime_lifecycle_io.py",
        )
    }
    decision = _strict_object(repository / RUNTIME_QUALIFICATION_DECISION_RELATIVE)
    required = {
        "RUNTIME_LIFECYCLE_QUALIFICATION_PASS": True,
        "CONFIRMATORY_V3_PRE_RUN_DESIGN_AUTHORIZED": True,
        "CONFIRMATORY_V3_SEED_DERIVATION_AUTHORIZED": True,
        "CONFIRMATORY_V3_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_V3_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V3_PASS": "NOT_EVALUATED",
    }
    if any(decision.get(name) != expected for name, expected in required.items()):
        raise ValueError("runtime lifecycle qualification decision changed")
    return {
        "RUNTIME_LIFECYCLE_CORE_FILE_CHANGE_COUNT": len(authorized_changes),
        "RUNTIME_LIFECYCLE_UNAUTHORIZED_FILE_CHANGE_COUNT": 0,
        "authorized_bootstrap_repair_changes": authorized_changes,
        "authority_path": RUNTIME_CORE_AUTHORITY_RELATIVE.as_posix(),
        "authority_sha256": file_sha256(path),
        "file_binding_count": len(normalized),
        "file_binding_set_sha256": canonical_identity_sha256(normalized),
        "formal_execution_core_files": execution_files,
        "formal_execution_core_sha256": canonical_json_sha256(execution_files),
        "qualification_decision_path": (
            RUNTIME_QUALIFICATION_DECISION_RELATIVE.as_posix()
        ),
        "qualification_decision_sha256": file_sha256(
            repository / RUNTIME_QUALIFICATION_DECISION_RELATIVE
        ),
        "qualification_pass": True,
    }


def _history_binding(repository: Path) -> dict[str, Any]:
    v1 = _strict_object(repository / V1_FAILURE_BINDING_RELATIVE)
    v1_seeds = _strict_object(repository / V1_SEED_RETIREMENT_RELATIVE)
    v2 = _strict_object(repository / V2_FAILURE_BINDING_RELATIVE)
    v2_archive = _strict_object(repository / V2_FAILURE_ARCHIVE_VERIFICATION_RELATIVE)
    v2_seeds = _strict_object(repository / V2_SEED_RETIREMENT_RELATIVE)
    if not (
        v1.get("V1_FAILURE_RECORD_PRESERVED") is True
        and v1.get("SYNTHETIC_CONFIRMATORY_PASS") == "NOT_EVALUATED"
        and v1_seeds.get("OLD_V1_SEED_SET_REUSE_AUTHORIZED") is False
        and v2.get("SYNTHETIC_CONFIRMATORY_V2_PASS") == "NOT_EVALUATED"
        and v2_archive.get("V2_FAILURE_ARCHIVE_PASS") is True
        and v2_archive.get("archive_tar_sha256") == V2_FAILURE_ARCHIVE_TAR_SHA256
        and v2_seeds.get("V2_SEED_RETIREMENT_PASS") is True
        and v2_seeds.get("V2_CONFIRMATORY_SEED_SET_REUSE_AUTHORIZED") is False
    ):
        raise ValueError("v1/v2 failure or seed-retirement history changed")
    return {
        "v1": {
            "failure_binding_path": V1_FAILURE_BINDING_RELATIVE.as_posix(),
            "failure_binding_sha256": file_sha256(
                repository / V1_FAILURE_BINDING_RELATIVE
            ),
            "failure_reason": "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON",
            "failure_tag": v1["v1_failure_tag"],
            "scientific_state": "NOT_EVALUATED",
            "seed_reuse_authorized": False,
            "seed_retirement_path": V1_SEED_RETIREMENT_RELATIVE.as_posix(),
            "seed_retirement_sha256": file_sha256(
                repository / V1_SEED_RETIREMENT_RELATIVE
            ),
        },
        "v2": {
            "archive_tar_path": str(v2_archive["archive_tar_path"]),
            "archive_tar_sha256": v2_archive["archive_tar_sha256"],
            "failure_binding_path": V2_FAILURE_BINDING_RELATIVE.as_posix(),
            "failure_binding_sha256": file_sha256(
                repository / V2_FAILURE_BINDING_RELATIVE
            ),
            "failure_reason": "RUNTIME_PATH_AND_GIT_GATE_LIFECYCLE_DEFECT",
            "failure_tag": v2["failure_tag"],
            "scientific_state": "NOT_EVALUATED",
            "seed_reuse_authorized": False,
            "seed_retirement_path": V2_SEED_RETIREMENT_RELATIVE.as_posix(),
            "seed_retirement_sha256": file_sha256(
                repository / V2_SEED_RETIREMENT_RELATIVE
            ),
        },
    }


def _backend_bindings(repository: Path) -> dict[str, Any]:
    contract = _strict_object(repository / BACKEND_PARAMETER_RELATIVE)
    return {
        "native": {"authorized": False, "planned_trial_count": 0},
        "open3d": {
            "adapter_path": "src/phase_a_harness/open3d_backend.py",
            "adapter_sha256": file_sha256(
                repository / "src/phase_a_harness/open3d_backend.py"
            ),
            "algorithm": contract["open3d"]["parameters"]["algorithm"],
            "parameter_sha256": contract["open3d"]["canonical_sha256"],
            "planned_trial_count": 595,
            "version": contract["open3d"]["parameters"]["version"],
        },
        "pcl": {
            "adapter_path": "src/phase_a_harness/pcl_backend.py",
            "adapter_sha256": file_sha256(
                repository / "src/phase_a_harness/pcl_backend.py"
            ),
            "algorithm": contract["pcl"]["parameters"]["algorithm"],
            "cli_path": PCL_CLI_RELATIVE.as_posix(),
            "cli_sha256": file_sha256(repository / PCL_CLI_RELATIVE),
            "parameter_sha256": contract["pcl"]["canonical_sha256"],
            "planned_trial_count": 595,
            "version": contract["pcl"]["parameters"]["version"],
        },
    }


SCIENTIFIC_ZERO_DIFFERENCE_FIELDS = (
    "scene_difference_count",
    "condition_difference_count",
    "planned_snapshot_count_difference",
    "planned_trial_count_difference",
    "backend_algorithm_difference_count",
    "backend_parameter_difference_count",
    "trial_schema_scientific_field_difference_count",
    "translation_metric_difference_count",
    "rotation_metric_difference_count",
    "quantile_method_difference_count",
    "H1_definition_difference_count",
    "H1_threshold_difference_count",
    "H2_definition_difference_count",
    "H2_threshold_difference_count",
    "H3_definition_difference_count",
    "H3_threshold_difference_count",
    "H4_definition_difference_count",
    "H4_threshold_difference_count",
    "H5_definition_difference_count",
    "H5_threshold_difference_count",
    "H6_definition_difference_count",
    "H6_threshold_difference_count",
    "common_association_difference_count",
    "turnover_definition_difference_count",
    "systematic_offset_definition_difference_count",
    "frozen_model_file_difference_count",
    "frozen_model_feature_difference_count",
    "frozen_model_parameter_difference_count",
)


def scientific_diff_v2_to_v3(root: str | Path) -> dict[str, Any]:
    """Prove that v3 changes identity/lifecycle metadata, not frozen science."""

    repository = Path(root).resolve()
    v2_protocol = _strict_object(
        repository / "protocols/synthetic_confirmatory_protocol_v2.json"
    )
    v3_protocol = _strict_object(repository / PROTOCOL_RELATIVE)
    v2_gate = _strict_object(repository / V2_GATE_RELATIVE)
    v3_gate = _strict_object(repository / GATE_RELATIVE)
    scientific = _scientific_core_binding(repository)
    runtime = _runtime_core_binding(repository)
    v2_manifest = _strict_object(
        repository / "frozen_assets/synthetic_confirmatory_formal_manifest_v2.json"
    )
    checks = {
        "scenes": v3_protocol.get("scenes") == v2_protocol.get("scenes") == list(SCENES),
        "conditions": (
            v3_protocol.get("conditions")
            == v2_protocol.get("conditions")
            == list(CONDITIONS)
        ),
        "counts": bool(
            v3_protocol.get("planned_snapshot_count")
            == v2_protocol.get("planned_snapshot_count")
            == SNAPSHOT_COUNT
            and v3_protocol.get("planned_trial_count")
            == v2_protocol.get("planned_trial_count")
            == TRIAL_COUNT
        ),
        "gate_science": _scientific_gate_core(v3_gate) == _scientific_gate_core(v2_gate),
        "frozen_model": bool(
            file_sha256(repository / FROZEN_MODEL_RELATIVE) == FROZEN_MODEL_SHA256
            and v2_manifest.get("frozen_model_sha256") == FROZEN_MODEL_SHA256
        ),
        "backend_parameters": bool(
            v2_manifest.get("open3d_parameter_sha256")
            == _backend_bindings(repository)["open3d"]["parameter_sha256"]
            and v2_manifest.get("pcl_parameter_sha256")
            == _backend_bindings(repository)["pcl"]["parameter_sha256"]
        ),
        "lineage": bool(
            v3_protocol.get("lineage_schema_version")
            == v2_protocol.get("lineage_schema_version")
            == LINEAGE_SCHEMA
        ),
        "scientific_core": scientific["SCIENTIFIC_CORE_FILE_CHANGE_COUNT"] == 0,
        "runtime_core": (
            runtime["RUNTIME_LIFECYCLE_UNAUTHORIZED_FILE_CHANGE_COUNT"] == 0
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"v2-to-v3 scientific binding changed: {failed}")
    zero = {name: 0 for name in SCIENTIFIC_ZERO_DIFFERENCE_FIELDS}
    return {
        "V2_TO_V3_SCIENTIFIC_DIFF_PASS": True,
        **zero,
        "allowed_differences": {
            "expected_tag_difference_count": 1,
            "manifest_binding_difference_count": 1,
            "runtime_path_binding_difference_count": 1,
            "runtime_bootstrap_implementation_difference_count": runtime[
                "RUNTIME_LIFECYCLE_CORE_FILE_CHANGE_COUNT"
            ],
            "seed_namespace_difference_count": 1,
            "seed_value_difference_count": 9,
            "version_metadata_difference_count": 1,
        },
        "binding_checks": checks,
        "schema_version": "synthetic_confirmatory_v2_to_v3_scientific_diff_v1",
        "scientific_core_file_set_sha256": scientific["file_binding_set_sha256"],
        "runtime_lifecycle_core_file_set_sha256": runtime[
            "file_binding_set_sha256"
        ],
    }


def manifest_payload(root: str | Path) -> dict[str, Any]:
    """Build the exact immutable v3 configuration payload from live bindings."""

    repository = Path(root).resolve()
    missing = [
        relative
        for relative in BOUND_FILE_PATHS.values()
        if not (repository / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"v3 manifest binding missing: {missing}")

    schedule = _strict_object(repository / SEED_SCHEDULE_RELATIVE)
    protocol = _strict_object(repository / PROTOCOL_RELATIVE)
    gate = _strict_object(repository / GATE_RELATIVE)
    v2_gate = _strict_object(repository / V2_GATE_RELATIVE)
    profile = _strict_object(repository / EXECUTION_PROFILE_RELATIVE)
    _verify_payload_digest(schedule, "seed_schedule_payload_sha256", "v3 schedule")
    _verify_payload_digest(protocol, "protocol_payload_sha256", "v3 protocol")
    _verify_payload_digest(gate, "gate_contract_payload_sha256", "v3 gate")
    _verify_payload_digest(profile, "execution_profile_payload_sha256", "v3 profile")
    if (
        schedule.get("schema_version") != SEED_SCHEDULE_SCHEMA
        or schedule.get("namespace") != NAMESPACE
        or schedule.get("geometry_seeds") != list(GEOMETRY_SEEDS)
        or schedule.get("measurement_seeds") != list(MEASUREMENT_SEEDS)
        or schedule.get("bootstrap_seed") != BOOTSTRAP_SEED
    ):
        raise ValueError("v3 seed schedule identity changed")
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("planned_snapshot_count") != SNAPSHOT_COUNT
        or protocol.get("planned_trial_count") != TRIAL_COUNT
        or protocol.get("seed_namespace") != NAMESPACE
        or protocol.get("formal_execution_state") != "NOT_EXECUTED"
        or protocol.get("scientific_evaluation_state") != "NOT_EVALUATED"
    ):
        raise ValueError("v3 protocol identity changed")
    if (
        gate.get("schema_version") != GATE_SCHEMA
        or _scientific_gate_core(gate) != _scientific_gate_core(v2_gate)
    ):
        raise ValueError("v3 gate differs from the frozen v2 scientific gate")
    if profile.get("schema_version") != EXECUTION_PROFILE_SCHEMA:
        raise ValueError("v3 execution profile identity changed")
    from .formal_runtime_state_machine import (
        build_formal_runner_command,
        formal_command_sha256,
    )

    fresh_command = build_formal_runner_command(
        repository=repository,
        manifest_path=MANIFEST_RELATIVE,
        run_id=FORMAL_RUN_ID,
        runtime_root=FORMAL_RUNTIME_ROOT,
        workers=FORMAL_WORKERS,
        mode="fresh",
        entry_script="scripts/run_synthetic_confirmatory_v3.py",
    )
    resume_command = build_formal_runner_command(
        repository=repository,
        manifest_path=MANIFEST_RELATIVE,
        run_id=FORMAL_RUN_ID,
        runtime_root=FORMAL_RUNTIME_ROOT,
        workers=FORMAL_WORKERS,
        mode="resume",
        entry_script="scripts/run_synthetic_confirmatory_v3.py",
    )
    profile_commands = profile.get("commands")
    profile_bootstrap = profile.get("bootstrap_contract")
    if (
        profile.get("configuration_authority") != MANIFEST_RELATIVE.as_posix()
        or profile.get("expected_branch") != FORMAL_BRANCH
        or profile.get("expected_release_tag") != FORMAL_PRERUN_TAG
        or profile.get("run_id") != FORMAL_RUN_ID
        or profile.get("workers") != FORMAL_WORKERS
        or type(profile_commands) is not dict
        or profile_commands.get("step_03_runner_fresh") != fresh_command
        or profile_commands.get("step_03_runner_resume") != resume_command
        or type(profile_bootstrap) is not dict
        or profile_bootstrap.get("formal_command_log_text") != fresh_command
        or profile_bootstrap.get("formal_command_log_sha256")
        != formal_command_sha256(fresh_command)
    ):
        raise ValueError("v3 bootstrap execution profile binding changed")

    plan = audit_v3_plan(
        repository / SNAPSHOT_PLAN_RELATIVE, repository / TRIAL_PLAN_RELATIVE
    )
    if plan.get("V3_PLAN_PASS") is not True:
        raise ValueError("v3 formal plan did not pass exact validation")
    runtime_audit = audit_formal_runtime_paths(repository)
    if runtime_audit.get("V3_RUNTIME_PATH_POLICY_PASS") is not True:
        raise ValueError("v3 formal runtime path contract failed")
    if file_sha256(repository / FROZEN_MODEL_RELATIVE) != FROZEN_MODEL_SHA256:
        raise ValueError("frozen Development model binding changed")

    bound = {
        name: {"path": relative, "sha256": file_sha256(repository / relative)}
        for name, relative in BOUND_FILE_PATHS.items()
    }
    scientific_core = _scientific_core_binding(repository)
    runtime_core = _runtime_core_binding(repository)
    payload: dict[str, Any] = {
        "artifact_inventory_version": ARTIFACT_INVENTORY_VERSION,
        "artifact_schema": ARTIFACT_SCHEMA,
        "artifact_staging_path": str(RUNTIME_PATHS["artifact_staging_path"]),
        "artifact_verifier": {
            "delegation_core_path": bound["formal_artifact_verifier_core"]["path"],
            "delegation_core_sha256": bound["formal_artifact_verifier_core"]["sha256"],
            "verification_mode": "EXTERNAL_RUNTIME_READ_THEN_COMPACT_IMPORT",
        },
        "authorization_binding": {
            "authorization_authority": "VERIFIED_PRE_RUN_FINAL_DECISION",
            "expected_release_tag": FORMAL_PRERUN_TAG,
            "pre_run_final_decision_path": PRE_RUN_FINAL_DECISION_RELATIVE.as_posix(),
            "required_authorization_field": "CONFIRMATORY_V3_RUN_AUTHORIZED",
            "required_authorization_value": True,
        },
        "backend_bindings": _backend_bindings(repository),
        "backend_temporary_path": str(RUNTIME_PATHS["backend_temporary_path"]),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bound_files": bound,
        "configuration_authority": "THIS_MANIFEST",
        "core_bindings": {
            "runtime_lifecycle": runtime_core,
            "scientific": scientific_core,
        },
        "event_log_path": str(RUNTIME_PATHS["event_log_path"]),
        "execution_profile_path": EXECUTION_PROFILE_RELATIVE.as_posix(),
        "expected_branch": FORMAL_BRANCH,
        "expected_release_tag": FORMAL_PRERUN_TAG,
        "final_decision_schema": FINAL_DECISION_SCHEMA,
        "formal_execution_state": "NOT_EXECUTED",
        "formal_run_authorization_state": "PENDING_VERIFIED_PRE_RUN_FINAL_DECISION",
        "frozen_model_path": FROZEN_MODEL_RELATIVE.as_posix(),
        "frozen_model_sha256": FROZEN_MODEL_SHA256,
        "gate_contract_path": GATE_RELATIVE.as_posix(),
        "gate_contract_sha256": bound["gate_contract"]["sha256"],
        "geometry_seeds": list(GEOMETRY_SEEDS),
        "history": _history_binding(repository),
        "manifest_version": "3-bootstrap-repair-r3",
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "native_trial_count": 0,
        "planned_snapshot_count": SNAPSHOT_COUNT,
        "planned_snapshots_path": SNAPSHOT_PLAN_RELATIVE.as_posix(),
        "planned_snapshots_sha256": bound["planned_snapshots"]["sha256"],
        "planned_trial_count": TRIAL_COUNT,
        "planned_trials_path": TRIAL_PLAN_RELATIVE.as_posix(),
        "planned_trials_sha256": bound["planned_trials"]["sha256"],
        "protocol_path": PROTOCOL_RELATIVE.as_posix(),
        "protocol_sha256": bound["scientific_protocol"]["sha256"],
        "protocol_version": "3",
        "formal_bootstrap_contract": {
            "allowed_bootstrap_files": [
                "formal_command.log",
                "formal_command.log.sha256",
            ],
            "command_log_path": str(FORMAL_RUNTIME_ROOT / "formal_command.log"),
            "command_log_sha256_path": str(
                FORMAL_RUNTIME_ROOT / "formal_command.log.sha256"
            ),
            "implementation_revision": (
                "synthetic_confirmatory_v3_bootstrap_repair_r1"
            ),
            "immutable_run_lock_path": str(
                FORMAL_RUNTIME_ROOT / "immutable_run_lock.json"
            ),
            "state_machine_schema": (
                "synthetic_confirmatory_v3_formal_runtime_state_machine_v1"
            ),
            "states": [
                "FORMAL_RUNTIME_ABSENT",
                "FORMAL_RUNTIME_BOOTSTRAP_ONLY",
                "FORMAL_RUNTIME_RESUMABLE",
                "FORMAL_RUNTIME_INVALID",
            ],
        },
        "publisher_inventory": {
            "figure_count": len(PUBLISHER_FIGURES),
            "figures": list(PUBLISHER_FIGURES),
            "root_file_count": len(PUBLISHER_ROOT_FILES),
            "root_files": list(PUBLISHER_ROOT_FILES),
            "table_count": len(PUBLISHER_TABLES),
            "tables": list(PUBLISHER_TABLES),
        },
        "publisher_staging_path": str(RUNTIME_PATHS["publisher_staging_path"]),
        "raw_result_manifest_schema": RAW_RESULT_MANIFEST_SCHEMA,
        "raw_results_path": str(RUNTIME_PATHS["raw_results_path"]),
        "run_id": FORMAL_RUN_ID,
        "runtime_path_policy_sha256": bound["runtime_path_policy"]["sha256"],
        "runtime_lifecycle_core_sha256": runtime_core[
            "formal_execution_core_sha256"
        ],
        "runtime_lifecycle_qualification_file_set_sha256": runtime_core[
            "file_binding_set_sha256"
        ],
        "runtime_root": str(FORMAL_RUNTIME_ROOT),
        "schema_version": MANIFEST_SCHEMA,
        "scientific_core_sha256": scientific_core["file_binding_set_sha256"],
        "scientific_evaluation_state": "NOT_EVALUATED",
        "seed_namespace": NAMESPACE,
        "seed_declaration_authority": SEED_SCHEDULE_RELATIVE.as_posix(),
        "seed_schedule_path": SEED_SCHEDULE_RELATIVE.as_posix(),
        "seed_schedule_payload_sha256": schedule["seed_schedule_payload_sha256"],
        "seed_schedule_sha256": bound["seed_schedule"]["sha256"],
        "snapshot_cache_path": str(RUNTIME_PATHS["snapshot_cache_path"]),
        "snapshot_lock_path": str(RUNTIME_PATHS["snapshot_lock_path"]),
        "temporary_inventory_path": str(RUNTIME_PATHS["temporary_inventory_path"]),
        "verification_path": str(RUNTIME_PATHS["verification_path"]),
        "analysis_path": str(RUNTIME_PATHS["analysis_path"]),
        "workers": FORMAL_WORKERS,
        "zero_instance_pre_run": {
            "backend_execution_count": 0,
            "rng_instantiation_count": 0,
            "scientific_result_count": 0,
            "snapshot_construction_count": 0,
            "started_event_count": 0,
            "trial_result_count": 0,
        },
    }
    return payload


def signed_manifest(root: str | Path) -> dict[str, Any]:
    payload = manifest_payload(root)
    return {**payload, "manifest_payload_sha256": canonical_json_sha256(payload)}


def load_v3_contract(path: str | Path) -> dict[str, Any]:
    """Load and verify the exact v3 manifest, returning a plain dictionary."""

    manifest_path = Path(path).resolve()
    repository = manifest_path.parent.parent.resolve()
    if manifest_path != (repository / MANIFEST_RELATIVE).resolve():
        raise ValueError("v3 requires the exact formal manifest path")
    value = _strict_object(manifest_path)
    if value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("not a Synthetic Confirmatory v3 manifest")
    expected = signed_manifest(repository)
    if value != expected:
        raise ValueError("v3 manifest differs from exact live bindings")
    return value


load_contract = load_v3_contract
verify_manifest = load_v3_contract


def write_manifest(root: str | Path, *, replace: bool = False) -> dict[str, Any]:
    repository = Path(root).resolve()
    path = repository / MANIFEST_RELATIVE
    if path.exists() and not replace:
        raise FileExistsError("refusing to replace v3 manifest")
    value = signed_manifest(repository)
    write_json(path, value)
    return value


def dry_run_v3_manifest(path: str | Path) -> dict[str, Any]:
    """Enumerate the plan without creating runtime state or consuming a seed."""

    manifest = load_v3_contract(path)
    repository = Path(path).resolve().parent.parent
    plan = audit_v3_plan(
        repository / manifest["planned_snapshots_path"],
        repository / manifest["planned_trials_path"],
    )
    runtime_exists = Path(manifest["runtime_root"]).exists()
    report = {
        "V3_BACKEND_EXECUTION_COUNT": 0,
        "V3_DRY_RUN_PASS": bool(plan["V3_PLAN_PASS"] and not runtime_exists),
        "V3_FORMAL_RUNTIME_ROOT_NOT_CREATED": not runtime_exists,
        "V3_RNG_INSTANTIATION_COUNT": 0,
        "V3_SCIENTIFIC_RESULT_COUNT": 0,
        "V3_SNAPSHOT_CONSTRUCTION_COUNT": 0,
        "V3_STARTED_EVENT_COUNT": 0,
        "V3_TRIAL_RESULT_COUNT": 0,
        "backend_trial_counts": plan["backend_trial_counts"],
        "condition_snapshot_counts": plan["condition_snapshot_counts"],
        "duplicate_snapshot_count": plan["duplicate_snapshot_count"],
        "duplicate_trial_count": plan["duplicate_trial_count"],
        "formal_execution_state": "NOT_EXECUTED",
        "formal_runtime_root": manifest["runtime_root"],
        "formal_runtime_root_exists": runtime_exists,
        "independent_pseudoreplication_count": plan[
            "independent_pseudoreplication_plan_count"
        ],
        "native_trial_count": plan["native_trial_count"],
        "pairing_violation_count": plan["pairing_violation_count"],
        "planned_snapshot_count": plan["planned_snapshot_count"],
        "planned_snapshot_unique_count": plan["planned_snapshot_unique_count"],
        "planned_trial_count": plan["planned_trial_count"],
        "planned_trial_unique_count": plan["planned_trial_unique_count"],
        "schema_version": DRY_RUN_SCHEMA,
        "scientific_evaluation_state": "NOT_EVALUATED",
    }
    if report["V3_DRY_RUN_PASS"] is not True:
        raise RuntimeError("v3 zero-instance dry-run failed")
    return report


__all__ = [name for name in globals() if name.isupper()] + [
    "audit_formal_runtime_paths",
    "audit_v3_plan",
    "canonical_identity_sha256",
    "derive_seed",
    "dry_run_v3_manifest",
    "expected_snapshot_id",
    "formal_execution_context",
    "load_contract",
    "load_v3_contract",
    "manifest_payload",
    "scientific_diff_v2_to_v3",
    "signed_manifest",
    "snapshot_identity",
    "typed_snapshot_rows",
    "typed_trial_rows",
    "validate_snapshot_plan_row",
    "validate_trial_plan_row",
    "validate_formal_snapshot_plan_row",
    "verify_manifest",
    "write_manifest",
]
