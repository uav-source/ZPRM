"""Strict publication verification for Synthetic Confirmatory v2.

The formal verifier accepts only the v2 manifest and the frozen 7-table /
3-figure / 7-root-file inventory.  The seed-free fixture publication has a
separate, equally compact inventory and explicitly cannot carry H1--H6 claims.
The pre-run verifier binds that fixture publication into the exact 26-file
pre-run evidence package.

All operations in this module are read-only except the explicit
``write_report=True`` transition, which writes only ``artifact_verification``.
No function constructs an RNG, a snapshot, or a backend invocation.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import file_sha256
from .synthetic_confirmatory_v2_contract import (
    ARTIFACT_INVENTORY_VERSION,
    ARTIFACT_SCHEMA,
    BACKEND_PARAMETER_RELATIVE,
    BACKENDS,
    BOOTSTRAP_SEED,
    FORMAL_BRANCH,
    FORMAL_ANALYSIS_SCHEMA,
    FORMAL_OUTPUT_DIR,
    FORMAL_PRERUN_TAG,
    FORMAL_RUN_ID,
    FORMAL_RUN_SCHEMA,
    FORMAL_WORKERS,
    FROZEN_MODEL_RELATIVE,
    GEOMETRY_SEEDS,
    INDEPENDENT_SCHEMA,
    LINEAGE_SCHEMA,
    MANIFEST_RELATIVE,
    MEASUREMENT_SEEDS,
    NAMESPACE,
    OLD_V1_BOOTSTRAP_SEED,
    OLD_V1_GEOMETRY_SEEDS,
    OLD_V1_MEASUREMENT_SEEDS,
    PCL_CLI_RELATIVE,
    PUBLISHER_FIGURES,
    PUBLISHER_ROOT_FILES,
    PUBLISHER_TABLES,
    SNAPSHOT_PLAN_RELATIVE,
    SEED_SCHEDULE_RELATIVE,
    signed_manifest,
    typed_snapshot_rows,
    verify_manifest,
)
from .synthetic_confirmatory_v2_independent_verifier import (
    compare_v2_fixture_primary_and_independent,
    compare_v2_primary_and_independent,
)


FROZEN_MODEL_SHA256 = (
    "99806f83d3a0d2393ee7e36e8c06f14a1a8a44fb8d02b074ebc2d9fe750d0872"
)
BACKEND_PARAMETER_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
OPEN3D_PARAMETER_SHA256 = (
    "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413"
)
PCL_PARAMETER_SHA256 = (
    "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd"
)
PCL_CLI_SHA256 = (
    "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff"
)

# These are verifier-owned copies of the qualification identity and protected
# scientific core.  They are intentionally not imported from the
# qualification implementation or from its design-audit JSON: the pre-run
# verifier must be able to detect coordinated tampering of either source.
QUALIFICATION_BINDING_FILES: Mapping[str, str] = {
    "development_protocol": "configs/zero_perturbation/development_v1.yaml",
    "frozen_scene_generator": (
        "src/phase_a_harness/phase_b_generator_frozen/capture_range/"
        "day2_development_scene.py"
    ),
    "frozen_snapshot_builder": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/"
        "snapshot_builder.py"
    ),
    "v2_snapshot_builder": (
        "src/phase_a_harness/synthetic_confirmatory_v2_snapshot_builder.py"
    ),
    "v2_contract": "src/phase_a_harness/synthetic_confirmatory_v2_contract.py",
    "qualification_orchestrator": (
        "src/phase_a_harness/synthetic_confirmatory_v2_qualification.py"
    ),
    "qualification_script": "scripts/qualify_synthetic_confirmatory_v2.py",
    "backend_parameter_contract": "frozen_assets/backend_parameter_contract.json",
    "open3d_adapter": "src/phase_a_harness/open3d_backend.py",
    "pcl_adapter": "src/phase_a_harness/pcl_backend.py",
    "backend_execution": "src/phase_a_harness/phase_a_execution_chain_audit.py",
    "trial_schema_validator": "src/phase_a_harness/phase_a_trial_result_schema.py",
    "pcl_cli": "bin/pcl_point_to_plane_cli",
    "fixture_plan": "frozen_assets/fixtures/fixture_plan.json",
    "fixture_snapshot_lock": "frozen_assets/fixtures/fixture_snapshot_lock.json",
    "fixture_parameter_lock": (
        "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
    ),
    "fixture_qualification": "src/phase_a_harness/fixture_qualification.py",
}
QUALIFICATION_MANIFEST_BINDING_NAMES: Mapping[str, str] = {
    "frozen_scene_generator": "generator",
    "frozen_snapshot_builder": "generator_zero_snapshot_builder",
    "v2_snapshot_builder": "v2_snapshot_builder",
    "v2_contract": "v2_contract",
    "qualification_orchestrator": "v2_qualification",
    "qualification_script": "v2_qualification_script",
    "backend_parameter_contract": "backend_parameter_contract",
    "open3d_adapter": "open3d_adapter",
    "pcl_adapter": "pcl_adapter",
    "backend_execution": "backend_execution",
    "trial_schema_validator": "trial_schema_validator",
    "pcl_cli": "pcl_cli",
    "fixture_plan": "fixture_plan",
    "fixture_snapshot_lock": "fixture_snapshot_lock",
    "fixture_parameter_lock": "fixture_backend_parameter_lock",
    "fixture_qualification": "fixture_qualification",
}
QUALIFICATION_FORMAL_ZERO_COUNTERS = (
    "FORMAL_V2_SEED_RNG_ACCESS_COUNT",
    "FORMAL_V2_SNAPSHOT_ACCESS_COUNT",
    "FORMAL_V2_SNAPSHOT_GENERATION_COUNT",
    "FORMAL_V2_BACKEND_EXECUTION_COUNT",
    "FORMAL_V2_TRIAL_RESULT_COUNT",
    "NATIVE_EXECUTION_COUNT",
)
QUALIFICATION_ACCESS_MONITOR_COUNTERS = (
    "formal_v2_result_file_read_count",
    "formal_v2_snapshot_file_read_count",
    "source_repository_runtime_file_read_count",
)
DEVELOPMENT_GEOMETRY_SEEDS = (1850310744, 1957656152, 1334931069)
QUALIFICATION_SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
DEVELOPMENT_NONIDEAL_CONDITIONS = (
    "INDEPENDENT_NOISE_FREE",
    "SCAN_NOISE_ONLY",
    "MAP_NOISE_ONLY",
    "DROPOUT_ONLY",
    "FULL_NOISE",
)
SCIENTIFIC_CORE_AST_SPECS: Mapping[str, tuple[str, str, str]] = {
    "phase_a_canonical_target": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/"
        "backend_phase_a_v1_2.py",
        "canonical_target",
        "5684f5bbcbd9001be0b29fcdd670470d0e19d7758d698fc39aaba4b73be3a5c8",
    ),
    "phase_a_eligible_parent_indices": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/"
        "backend_phase_a_v1_2.py",
        "eligible_parent_indices",
        "f8fd5aed7876e30cacfac41c1b699518d04e7c134f23d189c8f217f9af5e275a",
    ),
    "phase_a_quantization_closure": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/"
        "backend_phase_a_v1_2.py",
        "quantization_closure",
        "2ec95cab5329881772ccd56518b73705efded616254710d013db95324594ebac",
    ),
    "phase_a_reference_pose": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/"
        "backend_phase_a_v1_2.py",
        "reference_pose_from_development",
        "c9f8faa358bc09ee6e226547ec936f9bb2e95e7f98cac662b3a10da8b608caa1",
    ),
    "phase_a_source_from_parent_indices": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/"
        "backend_phase_a_v1_2.py",
        "source_from_parent_indices",
        "787fc81f8a0a00dab2e0fa9fe761e7a7b61d65f7b14305cc32627bb84265fa94",
    ),
    "primary_h1_h6_core": (
        "src/phase_a_harness/synthetic_confirmatory_analysis.py",
        "analyze_synthetic_confirmatory_records",
        "cff961f392b35694cb7d29c7310960d89ed0fed8636d0cc52cea8ad753167485",
    ),
    "independent_h1_h6_core": (
        "src/phase_a_harness/synthetic_confirmatory_independent_verifier.py",
        "independently_recompute_synthetic_confirmatory",
        "2c91d28d22ea812e544bcf82cb9421a62e0b758a1d25ea8f0e805800ab657d54",
    ),
}

FORMAL_TABLES = tuple(PUBLISHER_TABLES)
FORMAL_FIGURES = tuple(PUBLISHER_FIGURES)
FORMAL_ROOT_FILES = tuple(PUBLISHER_ROOT_FILES)
FORMAL_REQUIRED_FILES = (
    tuple(f"tables/{name}" for name in FORMAL_TABLES)
    + tuple(f"figures/{name}" for name in FORMAL_FIGURES)
    + FORMAL_ROOT_FILES
)

FIXTURE_TABLES = (
    "fixture_trial_inventory.csv",
    "fixture_snapshot_inventory.csv",
    "fixture_backend_summary.csv",
    "fixture_failure_inventory.csv",
    "fixture_pairing_audit.csv",
    "fixture_resume_equivalence.csv",
    "fixture_gate_summary.csv",
)
FIXTURE_FIGURES = (
    "fixture_execution_matrix.png",
    "fixture_failure_matrix.png",
    "fixture_update_metrics.png",
)
FIXTURE_ROOT_FILES = FORMAL_ROOT_FILES
FIXTURE_REQUIRED_FILES = (
    tuple(f"tables/{name}" for name in FIXTURE_TABLES)
    + tuple(f"figures/{name}" for name in FIXTURE_FIGURES)
    + FIXTURE_ROOT_FILES
)
FIXTURE_PRIMARY_SCHEMA = "synthetic_confirmatory_v2_fixture_primary_analysis_v1"
FIXTURE_INDEPENDENT_SCHEMA = (
    "synthetic_confirmatory_v2_fixture_independent_verification_v1"
)
FIXTURE_RUN_SCHEMA = "synthetic_confirmatory_v2_fixture_run_v1"
FIXTURE_ARTIFACT_SCHEMA = (
    "synthetic_confirmatory_v2_fixture_artifact_verification_v1"
)

PRERUN_ROOT_FILES = (
    "v1_failure_binding.json",
    "old_seed_retirement.json",
    "root_cause_binding.json",
    "v2_execution_chain_design.json",
    "v2_metadata_schema.json",
    "phase_a_closure_semantics_diff.json",
    "ideal_parent_lineage_qualification.json",
    "ideal_backend_control.json",
    "independent_negative_control.json",
    "nonideal_scientific_payload_regression.json",
    "v1_to_v2_scientific_diff.json",
    "frozen_model_binding.json",
    "v2_seed_schedule.json",
    "v2_seed_provenance_audit.json",
    "v2_plan_audit.json",
    "v2_dry_run_report.json",
    "fixture_regression.json",
    "primary_independent_difference.json",
    "test_report.json",
    "implementation_manifest.json",
    "artifact_verification.json",
    "final_decision.json",
    "run_manifest.json",
    "pre_run_report.md",
    "MANIFEST.csv",
    "SHA256SUMS",
)
PRERUN_SPECIAL_INVENTORY_FILES = frozenset(
    {"artifact_verification.json", "MANIFEST.csv", "SHA256SUMS"}
)
PRERUN_FIXTURE_DIRECTORY = "fixture_publication"
PRERUN_ARTIFACT_SCHEMA = (
    "synthetic_confirmatory_v2_prerun_artifact_verification_v1"
)


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {path}: {token}")
        ),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_verification(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(value))


def _actual_files(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _directory_errors(root: Path, allowed: set[str]) -> list[str]:
    if not root.is_dir():
        return ["."]
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    return sorted(actual - allowed)


def _symlink_errors(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    )


def _parse_sha256sums(path: Path) -> tuple[dict[str, str], int, int]:
    entries: dict[str, str] = {}
    malformed = 0
    duplicate = 0
    if not path.is_file():
        return entries, malformed, duplicate
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2:
            malformed += 1
            continue
        digest, relative = fields
        candidate = Path(relative)
        if (
            not _is_sha256(digest)
            or not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != relative
        ):
            malformed += 1
            continue
        duplicate += int(relative in entries)
        entries[relative] = digest
    return entries, malformed, duplicate


def _checksum_audit(root: Path, expected: set[str]) -> dict[str, Any]:
    entries, malformed, duplicate = _parse_sha256sums(root / "SHA256SUMS")
    listed = set(entries)
    missing = sorted(expected - listed)
    extra = sorted(listed - expected)
    mismatch = sorted(
        relative
        for relative in expected & listed
        if not (root / relative).is_file()
        or file_sha256(root / relative) != entries[relative]
    )
    return {
        "duplicate_sha256_path_count": duplicate,
        "malformed_or_unsafe_sha256_count": malformed,
        "sha256_entry_count": len(entries),
        "sha256_mismatch_files": mismatch,
        "sha256_missing_files": missing,
        "sha256_unexpected_files": extra,
        "sha256_verification_pass": bool(
            not malformed and not duplicate and not missing and not extra and not mismatch
        ),
    }


def _read_csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not fields or any(None in row for row in rows):
        raise ValueError(f"invalid CSV structure: {path}")
    return fields, rows


def _png_errors(root: Path, names: Sequence[str]) -> list[str]:
    errors = []
    for name in names:
        path = root / "figures" / name
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if payload[:8] != b"\x89PNG\r\n\x1a\n" or len(payload) < 100:
            errors.append(name)
    return errors


def _comparison(
    primary: Mapping[str, Any], independent: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return compare_v2_primary_and_independent(primary, independent)
    except (KeyError, TypeError, ValueError):
        return {
            "absolute_tolerance": 0.0,
            "relative_tolerance": 0.0,
            "section_difference_count": -1,
            "leaf_difference_count": -1,
            "maximum_absolute_numeric_difference": None,
            "exact_match_pass": False,
            "differing_sections": ["INVALID_OR_MISSING_VERIFICATION_PROJECTION"],
        }


def _formal_table_rows(primary: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        FORMAL_TABLES[0]: [dict(row) for row in primary["h1_ideal_control"]],
        FORMAL_TABLES[1]: [dict(row) for row in primary["h2_scene_effect"]],
        FORMAL_TABLES[2]: [dict(row) for row in primary["h3_cross_backend_ranking"]],
        FORMAL_TABLES[3]: [dict(row) for row in primary["h4_reassociation"]],
        FORMAL_TABLES[4]: [dict(row) for row in primary["h5_frozen_models"]],
        FORMAL_TABLES[5]: [
            *(
                {**dict(row), "row_type": "geometry"}
                for row in primary["h6_systematic_groups"]
            ),
            *(
                {**dict(row), "row_type": "backend"}
                for row in primary["h6_systematic_backend"]
            ),
        ],
        FORMAL_TABLES[6]: [
            {"gate": name, "pass": value}
            for name, value in primary["gate_summary"].items()
        ],
    }


def _csv_serialized_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
    return value


def _expected_csv_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    import io

    values = [dict(row) for row in rows]
    fields = sorted({key for row in values for key in row})
    if not fields:
        raise ValueError("publication table cannot be empty")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in values:
        writer.writerow(
            {key: _csv_serialized_value(value) for key, value in row.items()}
        )
    return stream.getvalue().encode("utf-8")


def _table_projection_errors(
    root: Path, expected_rows: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[str]:
    errors = []
    for name, rows in expected_rows.items():
        path = root / "tables" / name
        if not path.is_file():
            continue
        try:
            expected = _expected_csv_bytes(rows)
        except (KeyError, TypeError, ValueError):
            errors.append(name)
            continue
        if path.read_bytes() != expected:
            errors.append(name)
    return errors


def _formal_lineage_audit(
    repository: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Strictly re-read every v2 snapshot and its parent sidecar.

    ``read_v2_snapshot`` recomputes row correspondence and Phase-A closure.  The
    artifact verifier additionally checks the condition-specific file presence
    and raw parent SHA so the final report exposes these exact counters.
    """

    from .synthetic_confirmatory_v2_snapshot_builder import (
        PARENT_INDEX_FILENAME,
        read_v2_snapshot,
        validate_v2_snapshot_lock,
    )

    plans = typed_snapshot_rows(repository / SNAPSHOT_PLAN_RELATIVE)
    cache = repository / str(manifest["snapshot_cache_root"])
    lock_path = repository / str(manifest["snapshot_lock_path"])
    lock = validate_v2_snapshot_lock(lock_path, cache, plans)
    entries = {row["snapshot_id"]: row for row in lock["snapshots"]}
    ideal_present = nonideal_absent = 0
    raw_sha_mismatch = lineage_violation = 0
    for plan in plans:
        snapshot_id = plan["planned_snapshot_id"]
        directory = cache / snapshot_id
        parent_path = directory / PARENT_INDEX_FILENAME
        item = read_v2_snapshot(
            cache, plan, expected_lock_entry=entries[snapshot_id], arrays=True
        )
        metadata = item["metadata"]
        parent = item["parent_indices"]
        if plan["condition"] == "IDEAL_MATCHED":
            ideal_present += int(parent_path.is_file() and parent is not None)
            if parent is None:
                lineage_violation += 1
                continue
            raw = hashlib.sha256(parent.tobytes(order="C")).hexdigest()
            raw_sha_mismatch += int(
                raw != metadata.get("source_parent_target_indices_sha256")
            )
            lineage_violation += int(
                metadata.get("lineage_schema_version") != LINEAGE_SCHEMA
                or metadata.get("source_has_target_parent_lineage") is not True
                or metadata.get("source_is_target_subset") is not True
                or metadata.get("lineage_closure_violation_count") != 0
                or metadata.get("parent_index_count") != len(parent)
                or metadata.get("parent_index_unique_count")
                != len(np.unique(parent))
                or metadata.get("parent_index_out_of_range_count") != 0
            )
        else:
            nonideal_absent += int(not parent_path.exists() and parent is None)
            lineage_violation += int(
                parent_path.exists()
                or parent is not None
                or metadata.get("source_has_target_parent_lineage") is not False
                or metadata.get("source_is_target_subset") is not False
                or metadata.get("source_parent_target_indices_sha256") is not None
            )
    report = {
        "ideal_parent_sidecar_present_count": ideal_present,
        "nonideal_parent_sidecar_absent_count": nonideal_absent,
        "parent_raw_sha_mismatch_count": raw_sha_mismatch,
        "lineage_violation_count": lineage_violation,
        "snapshot_count": len(plans),
        "snapshot_lock_sha256": file_sha256(lock_path),
    }
    report["lineage_artifact_binding_pass"] = bool(
        len(plans) == 595
        and ideal_present == 35
        and nonideal_absent == 560
        and raw_sha_mismatch == 0
        and lineage_violation == 0
    )
    return report


def _binding_audit(repository: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _strict_json(repository / BACKEND_PARAMETER_RELATIVE)
    bound = manifest.get("bound_files")

    def bound_file_pass(name: str) -> bool:
        if type(bound) is not dict or type(bound.get(name)) is not dict:
            return False
        row = bound[name]
        relative = row.get("path")
        if not isinstance(relative, str):
            return False
        path = (repository / relative).resolve()
        return bool(
            repository in path.parents
            and path.is_file()
            and row.get("sha256") == file_sha256(path)
        )

    checks = {
        "artifact_inventory_version_pass": (
            manifest.get("artifact_inventory_version") == ARTIFACT_INVENTORY_VERSION
        ),
        "artifact_schema_pass": manifest.get("artifact_schema") == ARTIFACT_SCHEMA,
        "backend_parameter_contract_file_sha_pass": (
            file_sha256(repository / BACKEND_PARAMETER_RELATIVE)
            == BACKEND_PARAMETER_CONTRACT_SHA256
        ),
        "frozen_model_file_sha_pass": (
            file_sha256(repository / FROZEN_MODEL_RELATIVE) == FROZEN_MODEL_SHA256
        ),
        "open3d_parameter_sha_pass": (
            parameters.get("open3d", {}).get("canonical_sha256")
            == OPEN3D_PARAMETER_SHA256
            and manifest.get("open3d_parameter_sha256") == OPEN3D_PARAMETER_SHA256
        ),
        "open3d_implementation_binding_pass": bound_file_pass(
            "open3d_adapter"
        ),
        "pcl_cli_file_sha_pass": (
            file_sha256(repository / PCL_CLI_RELATIVE) == PCL_CLI_SHA256
        ),
        "pcl_parameter_sha_pass": (
            parameters.get("pcl", {}).get("canonical_sha256")
            == PCL_PARAMETER_SHA256
            and manifest.get("pcl_parameter_sha256") == PCL_PARAMETER_SHA256
        ),
        "pcl_implementation_binding_pass": (
            bound_file_pass("pcl_adapter")
            and bound_file_pass("full_synthetic_backend_execution")
        ),
    }
    return {
        **checks,
        "model_backend_binding_pass": all(checks.values()),
    }


def _lineage_summary_pass(
    primary: Mapping[str, Any], independent: Mapping[str, Any]
) -> tuple[bool, dict[str, Any], Any]:
    primary_lineage = primary.get("lineage_integrity")
    independent_lineage = independent.get("lineage_integrity")
    if independent_lineage is None and type(independent.get("verification_projection")) is dict:
        independent_lineage = independent["verification_projection"].get(
            "lineage_integrity"
        )
    expected = {
        "IDEAL_PARENT_LINEAGE_COUNT": 35,
        "LINEAGE_INTEGRITY_PASS": True,
        "LINEAGE_VIOLATION_COUNT": 0,
        "NONIDEAL_NO_LINEAGE_COUNT": 560,
    }
    primary_schema_pass = bool(
        type(primary_lineage) is dict
        and primary_lineage.get("schema_version")
        == "synthetic_confirmatory_v2_primary_lineage_inventory_v1"
    )
    independent_schema_pass = bool(
        type(independent_lineage) is dict
        and independent_lineage.get("schema_version")
        in (None, "synthetic_confirmatory_v2_independent_lineage_inventory_v1")
    )
    passed = bool(
        type(primary_lineage) is dict
        and type(independent_lineage) is dict
        and primary_schema_pass
        and independent_schema_pass
        and all(
            primary_lineage.get(key) == independent_lineage.get(key)
            for key in expected
        )
        and all(primary_lineage.get(key) == value for key, value in expected.items())
    )
    return passed, dict(primary_lineage or {}), independent_lineage


def verify_synthetic_confirmatory_v2_artifact(
    path: str | Path,
    *,
    manifest_path: str | Path,
    write_report: bool = False,
) -> dict[str, Any]:
    """Verify one complete formal v2 artifact against live frozen evidence."""

    root = Path(path).resolve()
    actual = _actual_files(root)
    virtual = set(actual)
    if write_report:
        virtual.add("artifact_verification.json")
    expected = set(FORMAL_REQUIRED_FILES)
    missing = sorted(expected - virtual)
    extra = sorted(actual - expected)
    directory_errors = _directory_errors(root, {"tables", "figures"})
    symlink_errors = _symlink_errors(root)

    objects: dict[str, dict[str, Any]] = {}
    json_errors = []
    for name in (
        "primary_analysis.json",
        "independent_verification.json",
        "final_decision.json",
        "run_manifest.json",
    ):
        candidate = root / name
        if not candidate.is_file():
            continue
        try:
            objects[name] = _strict_json(candidate)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            json_errors.append(name)

    primary = objects.get("primary_analysis.json", {})
    independent = objects.get("independent_verification.json", {})
    decision = objects.get("final_decision.json", {})
    run = objects.get("run_manifest.json", {})
    comparison = _comparison(primary, independent)
    lineage_summary_pass, primary_lineage, independent_lineage = (
        _lineage_summary_pass(primary, independent)
    )
    schema_pass = bool(
        primary.get("schema_version") == FORMAL_ANALYSIS_SCHEMA
        and independent.get("schema_version") == INDEPENDENT_SCHEMA
        and run.get("schema_version") == FORMAL_RUN_SCHEMA
    )
    decision_pass = bool(
        primary.get("final_decision") == decision
        and independent.get("final_decision") == decision
        and decision.get("SYNTHETIC_CONFIRMATORY_V2_EXECUTED") is True
        and decision.get("SYNTHETIC_CONFIRMATORY_V2_COMPLETE") is True
        and type(decision.get("SYNTHETIC_CONFIRMATORY_V2_PASS")) is bool
        and decision.get("CONFIRMATORY_V2_RUN_AUTHORIZED") is False
        and decision.get("REAL_DATA_RUN_AUTHORIZED") is False
        and decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is False
        and "SYNTHETIC_CONFIRMATORY_EXECUTED" not in decision
    )
    raw_binding_pass = bool(
        run.get("run_id") == FORMAL_RUN_ID
        and _is_sha256(run.get("raw_result_manifest_sha256"))
        and all(
            report.get("run_id") == FORMAL_RUN_ID
            and report.get("raw_result_manifest_sha256")
            == run.get("raw_result_manifest_sha256")
            for report in (primary, independent)
        )
    )
    run_integrity_pass = bool(
        run.get("completed_snapshot_count") == 595
        and run.get("completed_trial_count") == 1190
        and run.get("open3d_trial_count") == 595
        and run.get("pcl_trial_count") == 595
        and run.get("native_trial_count") == 0
        and run.get("native_execution_count") == 0
        and run.get("condition_trial_counts")
        == {
            "FULL_NOISE": 1050,
            "IDEAL_MATCHED": 70,
            "INDEPENDENT_NOISE_FREE": 70,
        }
        and all(
            run.get(name) == 0
            for name in (
                "backend_input_checksum_mismatch_count",
                "corrupt_trial_count",
                "duplicate_trial_count",
                "event_identity_mismatch_count",
                "extra_trial_count",
                "missing_trial_count",
                "reverse_raw_result_inventory_mismatch_count",
                "source_repository_runtime_file_read_count",
                "source_repository_runtime_import_count",
                "trial_result_checksum_mismatch_count",
            )
        )
        and run.get("source_repository_runtime_import_paths") == []
    )

    csv_errors = []
    for name in FORMAL_TABLES:
        candidate = root / "tables" / name
        if not candidate.is_file():
            continue
        try:
            _read_csv_rows(candidate)
        except (OSError, UnicodeError, csv.Error, ValueError):
            csv_errors.append(name)
    try:
        projection_errors = _table_projection_errors(
            root, _formal_table_rows(primary)
        )
    except (KeyError, TypeError, ValueError):
        projection_errors = list(FORMAL_TABLES)
    png_errors = _png_errors(root, FORMAL_FIGURES)
    checksums = _checksum_audit(
        root, expected - {"SHA256SUMS", "artifact_verification.json"}
    )

    manifest_error = None
    lineage_binding: dict[str, Any] = {
        "lineage_artifact_binding_pass": False
    }
    bindings: dict[str, Any] = {"model_backend_binding_pass": False}
    try:
        repository, manifest = verify_manifest(
            manifest_path, require_authorized=True
        )
        bindings = _binding_audit(repository, manifest)
        bindings["run_workers_match_pass"] = (
            run.get("workers") == manifest.get("formal_workers")
        )
        lineage_binding = _formal_lineage_audit(repository, manifest)
        if run.get("snapshot_lock_sha256") != lineage_binding.get(
            "snapshot_lock_sha256"
        ):
            lineage_binding["lineage_artifact_binding_pass"] = False
            lineage_binding["run_snapshot_lock_sha_match_pass"] = False
        else:
            lineage_binding["run_snapshot_lock_sha_match_pass"] = True
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, PermissionError) as error:
        manifest_error = f"{type(error).__name__}: {error}"

    report_path = root / "synthetic_confirmatory_report.md"
    markdown = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    missing_references = [
        name for name in (*FORMAL_TABLES, *FORMAL_FIGURES) if name not in markdown
    ]
    v1_masquerade_rejected = bool(
        schema_pass
        and run.get("run_id") == FORMAL_RUN_ID
        and "Synthetic Confirmatory v2" in markdown
        and "Synthetic Confirmatory v1" not in markdown
    )
    result = {
        "schema_version": "synthetic_confirmatory_v2_artifact_verification_v1",
        "actual_file_count": len(virtual),
        "analysis_input_binding_pass": raw_binding_pass,
        "analysis_verifier_comparison": comparison,
        "analysis_verifier_exact_match_pass": comparison.get("exact_match_pass") is True,
        "formal_run_integrity_pass": run_integrity_pass,
        "directory_inventory_errors": directory_errors,
        "extra_files": extra,
        "final_decision_match_pass": decision_pass,
        "invalid_csv_files": csv_errors,
        "invalid_json_files": json_errors,
        "invalid_png_files": png_errors,
        "lineage_summary_match_pass": lineage_summary_pass,
        "manifest_verification_error": manifest_error,
        "missing_required_files": missing,
        "primary_lineage_integrity": primary_lineage,
        "independent_lineage_integrity": independent_lineage,
        "report_missing_references": missing_references,
        "required_file_count": len(expected),
        "schema_identity_pass": schema_pass,
        "symlink_paths": symlink_errors,
        "table_projection_mismatch_files": projection_errors,
        "v1_artifact_masquerade_rejected": v1_masquerade_rejected,
        **bindings,
        **lineage_binding,
        **checksums,
    }
    result["ARTIFACT_VERIFICATION_PASS"] = bool(
        not missing
        and not extra
        and not directory_errors
        and not symlink_errors
        and not json_errors
        and not csv_errors
        and not projection_errors
        and not png_errors
        and not missing_references
        and schema_pass
        and decision_pass
        and raw_binding_pass
        and run_integrity_pass
        and comparison.get("exact_match_pass") is True
        and lineage_summary_pass
        and manifest_error is None
        and bindings.get("model_backend_binding_pass") is True
        and bindings.get("run_workers_match_pass") is True
        and lineage_binding.get("lineage_artifact_binding_pass") is True
        and v1_masquerade_rejected
        and checksums["sha256_verification_pass"]
    )
    if write_report:
        _write_verification(root / "artifact_verification.json", result)
    return result


def audit_synthetic_confirmatory_v2_fixture_rows(
    rows: Any,
) -> dict[str, Any]:
    """Audit the fixed, seed-free fixture row inventory and backend pairing."""

    conditions = (
        "FIXTURE_IDENTITY",
        "FIXTURE_NONIDENTITY_REFERENCE",
        "FIXTURE_NO_CORRESPONDENCE",
    )
    expected_identities = {
        (condition, backend)
        for condition in conditions
        for backend in BACKENDS
    }
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        return {
            "fixture_row_contract_pass": False,
            "input_checksum_violation_count": 0,
            "pairing_violation_count": 0,
            "snapshot_count": 0,
            "trial_count": 0,
        }
    identities = {
        (row.get("condition"), row.get("backend"))
        for row in rows
        if isinstance(row.get("condition"), str)
        and isinstance(row.get("backend"), str)
    }
    raw_snapshot_ids = [row.get("snapshot_id") for row in rows]
    snapshot_ids = {
        value for value in raw_snapshot_ids if isinstance(value, str)
    }
    trial_ids = [row.get("planned_trial_id") for row in rows]
    by_snapshot: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        snapshot_id = row.get("snapshot_id")
        if isinstance(snapshot_id, str):
            by_snapshot.setdefault(snapshot_id, []).append(row)
    checksum_names = (
        "snapshot_checksum",
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
    )
    pairing_violations = 0
    checksum_violations = 0
    condition_snapshot_ids: dict[str, set[str]] = {
        condition: set() for condition in conditions
    }
    for snapshot_id, group in by_snapshot.items():
        del snapshot_id
        group_conditions = {row.get("condition") for row in group}
        if len(group_conditions) == 1:
            condition = next(iter(group_conditions))
            if condition in condition_snapshot_ids:
                condition_snapshot_ids[str(condition)].add(
                    str(group[0].get("snapshot_id"))
                )
        pairing_violations += int(
            len(group) != 2
            or {row.get("backend") for row in group} != set(BACKENDS)
            or len(group_conditions) != 1
        )
        for name in checksum_names:
            values = [row.get(name) for row in group]
            checksum_violations += int(
                not all(_is_sha256(value) for value in values)
                or len(set(values)) != 1
            )
    outcome_violations = sum(
        int(
            type(row.get("finite_output")) is not bool
            or type(row.get("solver_failure")) is not bool
            or (
                row.get("condition") == "FIXTURE_NO_CORRESPONDENCE"
                and (
                    row.get("failure_classification") != "NO_CORRESPONDENCES"
                    or row.get("solver_failure") is not True
                )
            )
            or (
                row.get("condition") != "FIXTURE_NO_CORRESPONDENCE"
                and (
                    row.get("failure_classification") != "NONE"
                    or row.get("solver_failure") is not False
                    or row.get("finite_output") is not True
                )
            )
        )
        for row in rows
    )
    pass_value = bool(
        len(rows) == 6
        and identities == expected_identities
        and len(snapshot_ids) == 3
        and all(isinstance(value, str) and value for value in raw_snapshot_ids)
        and all(isinstance(value, str) and value for value in trial_ids)
        and len(trial_ids) == len(set(trial_ids)) == 6
        and all(len(values) == 1 for values in condition_snapshot_ids.values())
        and pairing_violations == 0
        and checksum_violations == 0
        and outcome_violations == 0
    )
    return {
        "fixture_row_contract_pass": pass_value,
        "input_checksum_violation_count": checksum_violations,
        "outcome_semantics_violation_count": outcome_violations,
        "pairing_violation_count": pairing_violations,
        "snapshot_count": len(snapshot_ids),
        "trial_count": len(rows),
    }


def _fixture_groups(primary: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = [dict(row) for row in primary.get("results", [])]
    by_snapshot: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_snapshot.setdefault(str(row.get("snapshot_id")), []).append(row)
    snapshot_rows = []
    pairing_rows = []
    for snapshot_id, group in sorted(by_snapshot.items()):
        checksums = {
            name: sorted({str(row.get(name)) for row in group})
            for name in (
                "snapshot_checksum",
                "source_checksum",
                "target_checksum",
                "reference_pose_checksum",
            )
        }
        checksum_match = all(
            len(values) == 1 and all(_is_sha256(value) for value in values)
            for values in checksums.values()
        )
        backends = sorted({str(row.get("backend")) for row in group})
        snapshot_rows.append(
            {
                "condition": group[0].get("condition") if group else None,
                "snapshot_id": snapshot_id,
                "trial_count": len(group),
                "backend_count": len(backends),
                "input_checksum_match": checksum_match,
            }
        )
        pairing_rows.append(
            {
                "snapshot_id": snapshot_id,
                "trial_count": len(group),
                "backends": backends,
                "pairing_pass": len(group) == 2
                and set(backends) == set(BACKENDS)
                and checksum_match,
            }
        )
    backend_rows = []
    for backend in BACKENDS:
        selected = [row for row in rows if row.get("backend") == backend]
        backend_rows.append(
            {
                "backend": backend,
                "trial_count": len(selected),
                "solver_failure_count": sum(
                    bool(row.get("solver_failure")) for row in selected
                ),
                "nonfinite_output_count": sum(
                    row.get("finite_output") is not True for row in selected
                ),
            }
        )
    return {
        FIXTURE_TABLES[0]: rows,
        FIXTURE_TABLES[1]: snapshot_rows,
        FIXTURE_TABLES[2]: backend_rows,
        FIXTURE_TABLES[3]: [
            dict(row) for row in primary.get("failure_inventory", [])
        ],
        FIXTURE_TABLES[4]: pairing_rows,
    }


def _fixture_table_rows(
    primary: Mapping[str, Any], run: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    result = _fixture_groups(primary)
    result[FIXTURE_TABLES[5]] = [
        {
            "fresh_resume_scientific_equivalence": run.get(
                "fresh_resume_scientific_equivalence"
            ),
            "resume_backend_execution_count": run.get(
                "resume_backend_execution_count"
            ),
        }
    ]
    result[FIXTURE_TABLES[6]] = [
        {"gate": key, "value": value}
        for key, value in primary.get("decision", {}).items()
    ]
    return result


def verify_synthetic_confirmatory_v2_fixture_artifact(
    path: str | Path, *, write_report: bool = False
) -> dict[str, Any]:
    """Verify the seed-free 3-snapshot/6-trial publication without H1--H6."""

    root = Path(path).resolve()
    actual = _actual_files(root)
    virtual = set(actual)
    if write_report:
        virtual.add("artifact_verification.json")
    expected = set(FIXTURE_REQUIRED_FILES)
    missing = sorted(expected - virtual)
    extra = sorted(actual - expected)
    directory_errors = _directory_errors(root, {"tables", "figures"})
    symlink_errors = _symlink_errors(root)
    objects: dict[str, dict[str, Any]] = {}
    json_errors = []
    for name in (
        "primary_analysis.json",
        "independent_verification.json",
        "final_decision.json",
        "run_manifest.json",
    ):
        candidate = root / name
        if not candidate.is_file():
            continue
        try:
            objects[name] = _strict_json(candidate)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            json_errors.append(name)
    primary = objects.get("primary_analysis.json", {})
    independent = objects.get("independent_verification.json", {})
    decision = objects.get("final_decision.json", {})
    run = objects.get("run_manifest.json", {})
    try:
        fixture_comparison = compare_v2_fixture_primary_and_independent(
            primary, independent
        )
    except (KeyError, TypeError, ValueError):
        fixture_comparison = {"exact_match_pass": False}
    comparison_pass = fixture_comparison.get("exact_match_pass") is True
    schema_pass = bool(
        primary.get("schema_version") == FIXTURE_PRIMARY_SCHEMA
        and independent.get("schema_version") == FIXTURE_INDEPENDENT_SCHEMA
        and run.get("schema_version") == FIXTURE_RUN_SCHEMA
    )
    science_absence_pass = bool(
        decision.get("FIXTURE_EXECUTION_CHAIN_PASS") is True
        and decision.get("FORMAL_CONFIRMATORY_SCIENCE_EVALUATED") is False
        and primary.get("decision") == decision
        and independent.get("final_decision", independent.get("decision")) == decision
        and not any(
            key.startswith(("h1_", "h2_", "h3_", "h4_", "h5_", "h6_"))
            for report in (primary, independent)
            for key in report
        )
    )
    rows = primary.get("results")
    fixture_row_audit = audit_synthetic_confirmatory_v2_fixture_rows(rows)
    fixture_counts_pass = bool(
        type(rows) is list
        and len(rows) == 6
        and primary.get("fixture_snapshot_count") == 3
        and primary.get("fixture_trial_count") == 6
        and run.get("fixture_snapshot_count") == 3
        and run.get("fixture_trial_count") == 6
        and run.get("backend_execution_count") == 6
        and run.get("resume_backend_execution_count") == 0
        and run.get("fresh_resume_scientific_equivalence") is True
        and run.get("formal_v2_seed_reference_count") == 0
        and run.get("formal_confirmatory_science_evaluated") is False
        and fixture_row_audit["fixture_row_contract_pass"] is True
    )
    table_errors = []
    try:
        expected_tables = _fixture_table_rows(primary, run)
    except (KeyError, TypeError, ValueError):
        expected_tables = {}
    for name in FIXTURE_TABLES:
        candidate = root / "tables" / name
        if not candidate.is_file():
            continue
        try:
            _read_csv_rows(candidate)
            if (
                name not in expected_tables
                or candidate.read_bytes()
                != _expected_csv_bytes(expected_tables[name])
            ):
                table_errors.append(name)
        except (OSError, UnicodeError, csv.Error, ValueError):
            table_errors.append(name)
    png_errors = _png_errors(root, FIXTURE_FIGURES)
    checksums = _checksum_audit(
        root, expected - {"SHA256SUMS", "artifact_verification.json"}
    )
    report_path = root / "synthetic_confirmatory_report.md"
    markdown = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    report_pass = bool(
        "Seed-Free Fixture" in markdown
        and "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED = false" in markdown
        and "H1--H6" not in markdown
        and all(name in markdown for name in (*FIXTURE_TABLES, *FIXTURE_FIGURES))
    )
    result = {
        "schema_version": FIXTURE_ARTIFACT_SCHEMA,
        "actual_file_count": len(virtual),
        "directory_inventory_errors": directory_errors,
        "extra_files": extra,
        "fixture_cardinality_and_pairing_pass": fixture_counts_pass,
        "fixture_row_audit": fixture_row_audit,
        "fixture_primary_independent_match_pass": comparison_pass,
        "formal_science_absence_pass": science_absence_pass,
        "invalid_json_files": json_errors,
        "invalid_png_files": png_errors,
        "invalid_or_mismatched_csv_files": sorted(set(table_errors)),
        "missing_required_files": missing,
        "report_semantics_pass": report_pass,
        "required_file_count": len(expected),
        "schema_identity_pass": schema_pass,
        "symlink_paths": symlink_errors,
        **checksums,
    }
    result["FIXTURE_ARTIFACT_VERIFICATION_PASS"] = bool(
        not missing
        and not extra
        and not directory_errors
        and not symlink_errors
        and not json_errors
        and not table_errors
        and not png_errors
        and schema_pass
        and comparison_pass
        and science_absence_pass
        and fixture_counts_pass
        and report_pass
        and checksums["sha256_verification_pass"]
    )
    if write_report:
        _write_verification(root / "artifact_verification.json", result)
    return result


def _manifest_csv_audit(
    root: Path, expected: set[str]
) -> dict[str, Any]:
    path = root / "MANIFEST.csv"
    rows: list[dict[str, str]] = []
    malformed = duplicate = 0
    if path.is_file():
        try:
            fields, rows = _read_csv_rows(path)
            if fields != ("path", "sha256", "size_bytes"):
                malformed += 1
        except (OSError, UnicodeError, csv.Error, ValueError):
            malformed += 1
            rows = []
    seen: set[str] = set()
    valid_rows: dict[str, tuple[str, int]] = {}
    for row in rows:
        relative = row.get("path", "")
        digest = row.get("sha256")
        size = row.get("size_bytes")
        candidate = Path(relative)
        if (
            relative in seen
            or not _is_sha256(digest)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != relative
        ):
            duplicate += int(relative in seen)
            malformed += 1
            continue
        seen.add(relative)
        try:
            parsed_size = int(size)
        except (TypeError, ValueError):
            malformed += 1
            continue
        if str(parsed_size) != str(size) or parsed_size < 0:
            malformed += 1
            continue
        valid_rows[relative] = (str(digest), parsed_size)
    missing = sorted(expected - set(valid_rows))
    extra = sorted(set(valid_rows) - expected)
    mismatch = sorted(
        relative
        for relative in expected & set(valid_rows)
        if not (root / relative).is_file()
        or file_sha256(root / relative) != valid_rows[relative][0]
        or (root / relative).stat().st_size != valid_rows[relative][1]
    )
    return {
        "manifest_duplicate_path_count": duplicate,
        "manifest_entry_count": len(valid_rows),
        "manifest_malformed_row_count": malformed,
        "manifest_mismatch_files": mismatch,
        "manifest_missing_files": missing,
        "manifest_unexpected_files": extra,
        "manifest_verification_pass": bool(
            not duplicate and not malformed and not missing and not extra and not mismatch
        ),
    }


def _all_zero(record: Mapping[str, Any], names: Sequence[str]) -> bool:
    return all(type(record.get(name)) is int and record[name] == 0 for name in names)


def _has_fields(record: Mapping[str, Any], names: Sequence[str]) -> bool:
    return all(name in record for name in names)


def _schema_present(record: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(record.get("schema_version"), str)
        and record["schema_version"]
    )


def _test_suite_pass(value: Any) -> bool:
    if value is True:
        return True
    if type(value) is not dict:
        return False
    explicit = value.get(
        "pass", value.get("PASS", value.get("passed", value.get("test_pass")))
    )
    status_pass = value.get("status") in (None, "PASS", "passed")
    return bool(
        (explicit is True or value.get("returncode", value.get("exit_code")) == 0)
        and status_pass
        and all(
            value.get(name, 0) == 0
            for name in (
                "error_count",
                "errors",
                "failed",
                "failure_count",
                "unexpected_skips",
                "unexpected_skip_count",
            )
        )
    )


def _finite_leq(value: Any, upper: float) -> bool:
    return bool(
        type(value) in (int, float)
        and math.isfinite(float(value))
        and float(value) <= upper
    )


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _function_ast_sha256(path: Path, function_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(nodes) != 1:
        raise ValueError(
            f"scientific function inventory is not unique: {path}:{function_name}"
        )
    return hashlib.sha256(
        ast.dump(nodes[0], include_attributes=False).encode("utf-8")
    ).hexdigest()


def _independent_scientific_core_ast_audit(
    repository: Path, implementation: Mapping[str, Any]
) -> bool:
    expected = {
        name: expected_sha
        for name, (_relative, _function, expected_sha) in SCIENTIFIC_CORE_AST_SPECS.items()
    }
    recorded = implementation.get("scientific_core_ast_hashes")
    if recorded != expected:
        return False
    live = {
        name: _function_ast_sha256(repository / relative, function)
        for name, (relative, function, _expected_sha) in SCIENTIFIC_CORE_AST_SPECS.items()
    }
    return live == expected


def _independent_qualification_execution_binding(
    repository: Path,
) -> dict[str, Any]:
    """Recompute the qualification binding without importing its implementation."""

    files: dict[str, dict[str, str]] = {}
    for name, relative in QUALIFICATION_BINDING_FILES.items():
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"qualification binding input missing: {relative}")
        files[name] = {"path": relative, "sha256": file_sha256(path)}
    if (
        files["backend_parameter_contract"]["sha256"]
        != BACKEND_PARAMETER_CONTRACT_SHA256
        or files["pcl_cli"]["sha256"] != PCL_CLI_SHA256
    ):
        raise ValueError("frozen backend qualification input SHA changed")
    core = {
        "files": files,
        "formal_v2_execution_authorized": False,
        "qualification_only": True,
        "schema_version": "synthetic_confirmatory_v2_qualification_binding_v1",
    }
    return {**core, "qualification_binding_sha256": _canonical_mapping_sha256(core)}


def _expected_implementation_paths(repository: Path) -> set[str]:
    """Independently reconstruct the implementation inventory frozen by the publisher."""

    prefixes = (
        "src/phase_a_harness/synthetic_confirmatory_v2_",
        "scripts/",
        "protocols/synthetic_confirmatory_",
        "artifacts/synthetic_confirmatory_v2_design_audit/",
    )
    selected = {
        path.relative_to(repository).as_posix()
        for path in repository.rglob("*")
        if path.is_file()
        and path.suffix in {".csv", ".json", ".md", ".py"}
        and any(
            path.relative_to(repository).as_posix().startswith(prefix)
            for prefix in prefixes
        )
        and (
            "synthetic_confirmatory_v2"
            in path.relative_to(repository).as_posix()
            or path.relative_to(repository).as_posix().startswith(
                "artifacts/synthetic_confirmatory_v2_design_audit/"
            )
        )
    }
    selected.update(
        {
            "tests/test_synthetic_confirmatory_v2.py",
            "tests/test_minimal_harness.py",
            "frozen_assets/synthetic_confirmatory_v2_seed_schedule.json",
        }
    )
    selected.discard("frozen_assets/synthetic_confirmatory_formal_manifest_v2.json")
    return selected


def _implementation_file_sha256_audit(
    repository: Path, implementation: Mapping[str, Any]
) -> bool:
    recorded = implementation.get("implementation_file_sha256")
    if type(recorded) is not dict:
        return False
    expected_paths = _expected_implementation_paths(repository)
    if set(recorded) != expected_paths:
        return False
    for relative in sorted(expected_paths):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            return False
        path = (repository / candidate).resolve()
        if repository not in path.parents or not path.is_file():
            return False
        if recorded.get(relative) != file_sha256(path):
            return False
    return True


def _qualification_lock_audit(
    value: Any, *, condition: str, lineage_expected: bool
) -> tuple[bool, list[dict[str, Any]]]:
    if type(value) is not dict:
        return False, []
    entries = value.get("entries")
    if type(entries) is not list or any(type(row) is not dict for row in entries):
        return False, []
    core = {key: item for key, item in value.items() if key != "qualification_lock_sha256"}
    integrity = bool(
        value.get("qualification_lock_sha256") == _canonical_mapping_sha256(core)
        and value.get("schema_version")
        == "synthetic_confirmatory_v2_qualification_lock_v1"
        and value.get("condition") == condition
        and value.get("lineage_expected") is lineage_expected
        and value.get("formal_v2_snapshot_lock") is False
        and value.get("qualification_only") is True
        and value.get("planned_snapshot_count") == len(entries)
    )
    return integrity, [dict(row) for row in entries]


def _qualification_firewall_zero(
    value: Any, *, expected_geometry_access_count: int
) -> bool:
    return bool(
        type(value) is dict
        and set(value)
        == {
            "geometry_access_count",
            "measurement_seed_access_count",
            "repeat_randomness_count",
            "rng_instantiation_count",
        }
        and value.get("geometry_access_count") == expected_geometry_access_count
        and value.get("measurement_seed_access_count") == 0
        and value.get("repeat_randomness_count") == 0
        and value.get("rng_instantiation_count") == 0
    )


def _ideal_qualification_reaggregation(ideal: Mapping[str, Any]) -> bool:
    lock_pass, entries = _qualification_lock_audit(
        ideal.get("qualification_lock"),
        condition="IDEAL_MATCHED",
        lineage_expected=True,
    )
    identities = {
        (row.get("scene_variant"), row.get("geometry_seed")) for row in entries
    }
    expected_identities = {
        (scene, seed)
        for scene in QUALIFICATION_SCENES
        for seed in DEVELOPMENT_GEOMETRY_SEEDS
    }
    violations = [
        str(row.get("snapshot_id"))
        for row in entries
        if row.get("condition") != "IDEAL_MATCHED"
        or row.get("measurement_seed") is not None
        or row.get("repeat_index") != 0
        or row.get("source_has_target_parent_lineage") is not True
        or type(row.get("parent_index_count")) is not int
        or row.get("parent_index_count", 0) <= 0
        or not _is_sha256(row.get("parent_index_sha256"))
        or not _qualification_firewall_zero(
            row.get("firewall_audit"), expected_geometry_access_count=1
        )
        or not all(
            _is_sha256(row.get(name))
            for name in (
                "reference_pose_checksum",
                "snapshot_checksum",
                "source_checksum",
                "target_checksum",
            )
        )
    ]
    snapshot_ids = [row.get("snapshot_id") for row in entries]
    geometry_access = sum(
        int(row["firewall_audit"]["geometry_access_count"])
        for row in entries
        if type(row.get("firewall_audit")) is dict
        and type(row["firewall_audit"].get("geometry_access_count")) is int
    )
    passed = bool(
        lock_pass
        and len(entries) == 21
        and identities == expected_identities
        and len(snapshot_ids) == len(set(snapshot_ids)) == 21
        and not violations
    )
    return bool(
        passed
        and ideal.get("IDEAL_GEOMETRY_QUALIFICATION_PASS") is passed
        and ideal.get("snapshot_count") == len(entries)
        and ideal.get("lineage_violation_count") == len(violations)
        and ideal.get("violation_snapshot_ids") == violations
        and ideal.get("geometry_access_count") == geometry_access
        and ideal.get("measurement_seed_access_count") == 0
        and ideal.get("repeat_randomness_count") == 0
        and ideal.get("rng_instantiation_count") == 0
        and ideal.get("confirmatory_seed_access_count") == 0
    )


def _negative_qualification_reaggregation(
    negative: Mapping[str, Any], repository: Path
) -> bool:
    lock_pass, entries = _qualification_lock_audit(
        negative.get("qualification_lock"),
        condition="INDEPENDENT_NOISE_FREE",
        lineage_expected=False,
    )
    rows = negative.get("rows")
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        return False
    expected_identities = {
        (scene, seed)
        for scene in QUALIFICATION_SCENES
        for seed in DEVELOPMENT_GEOMETRY_SEEDS
    }
    entry_identities = {
        (row.get("scene_variant"), row.get("geometry_seed")) for row in entries
    }
    row_identities = {
        (row.get("scene_variant"), row.get("geometry_seed")) for row in rows
    }
    entry_by_id = {row.get("snapshot_id"): row for row in entries}
    array_mismatches = sum(
        len(row.get("array_mismatch_fields", []))
        for row in rows
        if type(row.get("array_mismatch_fields")) is list
    )
    checksum_mismatches = sum(
        len(row.get("checksum_mismatch_fields", []))
        for row in rows
        if type(row.get("checksum_mismatch_fields")) is list
    )
    lineage_violations = sum(row.get("lineage_violation") is not False for row in rows)
    firewall_violations = sum(
        not _qualification_firewall_zero(
            entry.get("firewall_audit"), expected_geometry_access_count=3
        )
        for entry in entries
    )
    row_contract_pass = bool(
        len(rows) == 21
        and len(entries) == 21
        and row_identities == entry_identities == expected_identities
        and len({row.get("qualification_snapshot_id") for row in rows}) == 21
        and all(
            type(row.get("array_mismatch_fields")) is list
            and type(row.get("checksum_mismatch_fields")) is list
            and row.get("qualification_snapshot_id") in entry_by_id
            and entry_by_id[row.get("qualification_snapshot_id")].get("scene_variant")
            == row.get("scene_variant")
            and entry_by_id[row.get("qualification_snapshot_id")].get("geometry_seed")
            == row.get("geometry_seed")
            for row in rows
        )
        and all(
            entry.get("condition") == "INDEPENDENT_NOISE_FREE"
            and entry.get("measurement_seed") is None
            and entry.get("repeat_index") == 0
            and entry.get("source_has_target_parent_lineage") is False
            and entry.get("parent_index_count") == 0
            and entry.get("parent_index_sha256") is None
            for entry in entries
        )
    )
    baseline_lock = repository / "frozen_assets/phase_b_snapshot_lock.json"
    baseline_binding_pass = bool(
        baseline_lock.is_file()
        and negative.get("baseline_phase_b_snapshot_lock_sha256")
        == file_sha256(baseline_lock)
    )
    passed = bool(
        lock_pass
        and row_contract_pass
        and array_mismatches == 0
        and checksum_mismatches == 0
        and lineage_violations == 0
        and firewall_violations == 0
        and baseline_binding_pass
    )
    return bool(
        passed
        and negative.get("INDEPENDENT_NEGATIVE_CONTROL_PASS") is passed
        and negative.get("snapshot_count") == len(rows)
        and negative.get("false_lineage_count") == len(entries)
        and negative.get("array_mismatch_count") == array_mismatches
        and negative.get("checksum_mismatch_count") == checksum_mismatches
        and negative.get("lineage_violation_count") == lineage_violations
        and negative.get("firewall_violation_count") == firewall_violations
        and negative.get("rng_instantiation_count") == 0
        and negative.get("confirmatory_seed_access_count") == 0
    )


def _independent_backend_gate(
    results: Sequence[Mapping[str, Any]], backend: str
) -> dict[str, Any]:
    rows = [row for row in results if row.get("backend") == backend]
    failures = sum(row.get("solver_failure") is not False for row in rows)
    nonfinite = sum(row.get("finite_output") is not True for row in rows)
    translations = [
        float(row["translation_update_m"])
        for row in rows
        if type(row.get("translation_update_m")) in (int, float)
        and not isinstance(row.get("translation_update_m"), bool)
        and math.isfinite(float(row["translation_update_m"]))
    ]
    rotations = [
        float(row["rotation_update_rad"])
        for row in rows
        if type(row.get("rotation_update_rad")) in (int, float)
        and not isinstance(row.get("rotation_update_rad"), bool)
        and math.isfinite(float(row["rotation_update_rad"]))
    ]
    translation_q95 = (
        None
        if len(translations) != 21
        else float(np.quantile(translations, q=0.95, method="linear"))
    )
    rotation_q95 = (
        None
        if len(rotations) != 21
        else float(np.quantile(rotations, q=0.95, method="linear"))
    )
    passed = bool(
        len(rows) == 21
        and failures == 0
        and nonfinite == 0
        and translation_q95 is not None
        and translation_q95 <= 0.001
        and rotation_q95 is not None
        and rotation_q95 <= 0.00017453292519943296
    )
    return {
        "QUALIFICATION_BACKEND_GATE_PASS": passed,
        "backend": backend,
        "finite_metric_count": len(translations),
        "nonfinite_output_count": nonfinite,
        "quantile_method": "linear",
        "rotation_q95_rad": rotation_q95,
        "rotation_threshold_rad": 0.00017453292519943296,
        "solver_failure_count": failures,
        "translation_q95_m": translation_q95,
        "translation_threshold_m": 0.001,
        "trial_count": len(rows),
    }


def _backend_control_reaggregation(
    controls: Mapping[str, Any], ideal: Mapping[str, Any], binding: Mapping[str, Any]
) -> bool:
    results = controls.get("results")
    if type(results) is not list or any(type(row) is not dict for row in results):
        return False
    lock = ideal.get("qualification_lock")
    lock_entries = lock.get("entries") if type(lock) is dict else None
    if type(lock_entries) is not list:
        return False
    snapshot_ids = {row.get("snapshot_id") for row in lock_entries}
    identities = [(row.get("snapshot_id"), row.get("backend")) for row in results]
    expected_identities = {
        (snapshot_id, backend) for snapshot_id in snapshot_ids for backend in BACKENDS
    }
    gates = {
        backend: _independent_backend_gate(results, backend) for backend in BACKENDS
    }
    scientific_projection = [
        {key: value for key, value in row.items() if key != "runtime_ms"}
        for row in results
    ]
    result_lock_sha = _canonical_mapping_sha256({"results": scientific_projection})
    result_contract_pass = bool(
        len(results) == 42
        and len(set(identities)) == 42
        and set(identities) == expected_identities
        and len({row.get("planned_trial_id") for row in results}) == 42
        and all(
            row.get("condition") == "IDEAL_MATCHED"
            and row.get("planned_trial_id")
            == f"{row.get('snapshot_id')}/{row.get('backend')}"
            and row.get("implementation_sha256")
            == binding.get("qualification_binding_sha256")
            and row.get("snapshot_lock_sha256")
            == lock.get("qualification_lock_sha256")
            and _is_sha256(row.get("protocol_sha256"))
            and all(
                _is_sha256(row.get(name))
                for name in (
                    "reference_pose_checksum",
                    "snapshot_checksum",
                    "source_checksum",
                    "target_checksum",
                )
            )
            for row in results
        )
    )
    passed = bool(
        result_contract_pass
        and all(gate["QUALIFICATION_BACKEND_GATE_PASS"] for gate in gates.values())
    )
    return bool(
        passed
        and controls.get("IDEAL_DUAL_BACKEND_CONTROL_PASS") is passed
        and controls.get("backend_execution_count") == len(results)
        and controls.get("trial_count") == len(results)
        and controls.get("backend_gates") == gates
        and controls.get("qualification_result_lock_sha256") == result_lock_sha
        and controls.get("formal_v2_backend_execution_count") == 0
        and controls.get("native_execution_count") == 0
    )


def _read_development_plan(repository: Path) -> dict[str, dict[str, Any]]:
    path = repository / "frozen_assets/full_synthetic_development_new_snapshots_v1.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) != 1050:
        raise ValueError("Development non-IDEAL plan is not 1,050 rows")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        snapshot_id = row["snapshot_id"]
        if snapshot_id in result:
            raise ValueError("duplicate Development snapshot identity")
        result[snapshot_id] = {
            "condition": row["condition"],
            "geometry_seed": int(row["geometry_seed_value"]),
            "measurement_seed": int(row["measurement_seed_value"]),
            "repeat_index": int(row["repeat_index"]),
            "scene_variant": row["scene_variant"],
        }
    return result


def _nonideal_reaggregation(
    nonideal: Mapping[str, Any], repository: Path
) -> bool:
    rows = nonideal.get("rows")
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        return False
    plan = _read_development_plan(repository)
    condition_counts = Counter(row.get("condition") for row in rows)

    def list_length(row: Mapping[str, Any], name: str) -> int:
        value = row.get(name)
        return len(value) if type(value) is list else -1

    array_mismatches = sum(list_length(row, "array_mismatch_fields") for row in rows)
    metadata_mismatches = sum(
        list_length(row, "metadata_mismatch_fields") for row in rows
    )
    checksum_mismatches = sum(
        list_length(row, "checksum_mismatch_fields") for row in rows
    )
    persisted_record_mismatches = sum(
        list_length(row, "persisted_record_mismatch_fields") for row in rows
    )
    oracle_record_mismatches = sum(
        list_length(row, "record_oracle_mismatch_fields") for row in rows
    )
    candidate_record_missing = sum(
        list_length(row, "candidate_record_missing_fields") for row in rows
    )
    baseline_record_available = sum(
        list_length(row, "record_baseline_available_fields") for row in rows
    )
    before_dropout_mismatches = sum(
        row.get("before_dropout_count_mismatch") is not False for row in rows
    )
    firewall_violations = sum(row.get("firewall_violation") is not False for row in rows)
    aggregate_checksum_differences = sum(
        row.get("aggregate_snapshot_checksum_equal") is not True for row in rows
    )
    candidate_rng_count = sum(
        int(row.get("v2_candidate_rng_construction_count", -1)) for row in rows
    )
    reference_rng_count = sum(
        int(row.get("reference_rng_construction_count", -1)) for row in rows
    )
    source_changes = sum(
        "source_checksum" in row.get("checksum_mismatch_fields", [])
        or "source" in row.get("array_mismatch_fields", [])
        for row in rows
    )
    target_changes = sum(
        "target_checksum" in row.get("checksum_mismatch_fields", [])
        or "target" in row.get("array_mismatch_fields", [])
        for row in rows
    )
    reference_changes = sum(
        "reference_pose_checksum" in row.get("checksum_mismatch_fields", [])
        or "reference" in row.get("array_mismatch_fields", [])
        for row in rows
    )

    def record_changes(row: Mapping[str, Any]) -> set[str]:
        return (
            set(row.get("persisted_record_mismatch_fields", []))
            | set(row.get("record_oracle_mismatch_fields", []))
            | set(row.get("candidate_record_missing_fields", []))
        )

    noise_changes = sum("noise_checksum" in record_changes(row) for row in rows)
    dropout_changes = sum(
        "dropout_checksum" in record_changes(row) for row in rows
    )
    plan_identity_fields = {
        "condition",
        "geometry_seed",
        "geometry_seed_index",
        "measurement_seed",
        "measurement_seed_index",
        "repeat_index",
        "scene_variant",
        "snapshot_id",
    }
    plan_identity_changes = 0
    scientific_payload_changes = 0
    metadata_only_changes = 0
    frozen_plan_identity_mismatches = 0
    for row in rows:
        expected = plan.get(str(row.get("snapshot_id")))
        identity = {
            "condition": row.get("condition"),
            "geometry_seed": row.get("geometry_seed"),
            "measurement_seed": row.get("measurement_seed"),
            "repeat_index": row.get("repeat_index"),
            "scene_variant": row.get("scene_variant"),
        }
        identity_changed = expected != identity
        frozen_plan_identity_mismatches += int(identity_changed)
        recorded_plan_identity_change = bool(
            plan_identity_fields & set(row.get("metadata_mismatch_fields", []))
        )
        plan_identity_changes += int(recorded_plan_identity_change)
        payload_changed = bool(
            row.get("array_mismatch_fields")
            or row.get("metadata_mismatch_fields")
            or row.get("checksum_mismatch_fields")
            or record_changes(row)
            or row.get("before_dropout_count_mismatch") is not False
        )
        scientific_payload_changes += int(payload_changed)
        metadata_only_changes += int(
            row.get("aggregate_snapshot_checksum_equal") is not True
            and not payload_changed
        )

    snapshot_ids = [row.get("snapshot_id") for row in rows]
    row_contract_pass = bool(
        len(rows) == 1050
        and len(snapshot_ids) == len(set(snapshot_ids)) == len(plan)
        and set(snapshot_ids) == set(plan)
        and frozen_plan_identity_mismatches == 0
        and condition_counts
        == Counter({condition: 210 for condition in DEVELOPMENT_NONIDEAL_CONDITIONS})
        and all(
            type(row.get(name)) is list
            for row in rows
            for name in (
                "array_mismatch_fields",
                "metadata_mismatch_fields",
                "checksum_mismatch_fields",
                "persisted_record_mismatch_fields",
                "record_oracle_mismatch_fields",
                "candidate_record_missing_fields",
                "record_baseline_available_fields",
            )
        )
        and all(
            _is_sha256(row.get("candidate_noise_checksum"))
            and _is_sha256(row.get("candidate_dropout_checksum"))
            and _is_sha256(row.get("scientific_payload_sha256"))
            and type(row.get("v2_candidate_rng_construction_count")) is int
            and type(row.get("reference_rng_construction_count")) is int
            for row in rows
        )
    )
    lock_core = {
        "entries": [
            {
                "dropout_checksum": row.get("candidate_dropout_checksum"),
                "noise_checksum": row.get("candidate_noise_checksum"),
                "scientific_payload_sha256": row.get("scientific_payload_sha256"),
                "snapshot_id": row.get("snapshot_id"),
            }
            for row in rows
        ],
        "schema_version": "development_nonideal_scientific_payload_lock_v1",
    }
    expected_lock = {
        **lock_core,
        "scientific_payload_lock_sha256": _canonical_mapping_sha256(lock_core),
    }
    aggregate_fields = {
        "array_mismatch_count": array_mismatches,
        "baseline_noise_dropout_record_available_field_count": baseline_record_available,
        "before_dropout_count_mismatch_count": before_dropout_mismatches,
        "candidate_noise_dropout_record_count": 2 * len(rows),
        "candidate_noise_dropout_record_missing_count": candidate_record_missing,
        "candidate_v2_helper_call_count": len(rows),
        "checksum_mismatch_count": checksum_mismatches,
        "development_record_oracle_rng_construction_count": reference_rng_count,
        "metadata_mismatch_count": metadata_mismatches,
        "metadata_only_v1_v2_snapshot_checksum_difference_count": (
            aggregate_checksum_differences
        ),
        "noise_dropout_record_mismatch_count_where_persisted_baseline_available": (
            persisted_record_mismatches
        ),
        "record_oracle_mismatch_count": oracle_record_mismatches,
        "security_firewall_violation_count": firewall_violations,
        "snapshot_count": len(rows),
        "v2_candidate_rng_construction_count": candidate_rng_count,
        "NONIDEAL_SOURCE_CHECKSUM_CHANGE_COUNT": source_changes,
        "NONIDEAL_TARGET_CHECKSUM_CHANGE_COUNT": target_changes,
        "NONIDEAL_REFERENCE_CHECKSUM_CHANGE_COUNT": reference_changes,
        "NONIDEAL_NOISE_RECORD_CHANGE_COUNT": noise_changes,
        "NONIDEAL_DROPOUT_RECORD_CHANGE_COUNT": dropout_changes,
        "NONIDEAL_PLAN_IDENTITY_CHANGE_COUNT": plan_identity_changes,
        "NONIDEAL_SCIENTIFIC_PAYLOAD_CHANGE_COUNT": scientific_payload_changes,
        "METADATA_ONLY_CHANGE_COUNT": metadata_only_changes,
    }
    metadata_only_pass = bool(
        nonideal.get("METADATA_ONLY_CHANGE_COUNT") == metadata_only_changes
        and nonideal.get("metadata_only_v1_v2_snapshot_checksum_difference_count")
        == metadata_only_changes
    )
    frozen_lock = (
        repository / "frozen_assets/full_synthetic_development_snapshot_lock_v1.json"
    )
    aggregate_pass = all(nonideal.get(name) == value for name, value in aggregate_fields.items())
    passed = bool(
        row_contract_pass
        and aggregate_pass
        and metadata_only_pass
        and nonideal.get("condition_snapshot_counts")
        == dict(sorted(condition_counts.items()))
        and nonideal.get("scientific_payload_lock") == expected_lock
        and frozen_lock.is_file()
        and nonideal.get("frozen_snapshot_lock_file_sha256") == file_sha256(frozen_lock)
        and nonideal.get("expected_development_rng_construction_count") == 1260
        and candidate_rng_count == reference_rng_count == 1260
        and nonideal.get("confirmatory_seed_access_count") == 0
        and nonideal.get("formal_v2_rng_instantiation_count") == 0
        and nonideal.get("SCIENTIFIC_PAYLOAD_CHANGE_COUNT")
        == scientific_payload_changes
        and nonideal.get("shared_numerical_helper") == "_nonideal_arrays"
        and nonideal.get("candidate_v2_helper_callable")
        == (
            "phase_a_harness.synthetic_confirmatory_v2_snapshot_builder."
            "build_development_nonideal_regression_snapshot"
        )
    )
    expected_gate = bool(
        passed
        and array_mismatches == 0
        and metadata_mismatches == 0
        and checksum_mismatches == 0
        and persisted_record_mismatches == 0
        and oracle_record_mismatches == 0
        and candidate_record_missing == 0
        and before_dropout_mismatches == 0
        and firewall_violations == 0
        and scientific_payload_changes == 0
    )
    return bool(
        expected_gate
        and nonideal.get("DEVELOPMENT_NONIDEAL_SCIENTIFIC_PAYLOAD_REGRESSION_PASS")
        is expected_gate
        and nonideal.get("V2_NONIDEAL_SHARED_PATH_REGRESSION_PASS") is expected_gate
    )


def _qualification_zero_access_audit(controls: Mapping[str, Any]) -> bool:
    formal = controls.get("formal_v2_access_counters")
    access = controls.get("qualification_access_monitor")
    return bool(
        controls.get("FORMAL_CONFIRMATORY_SCIENCE_EVALUATED") is False
        and controls.get("source_repository_runtime_import_count") == 0
        and controls.get("formal_v2_backend_execution_count") == 0
        and controls.get("native_execution_count") == 0
        and type(formal) is dict
        and set(formal) == set(QUALIFICATION_FORMAL_ZERO_COUNTERS)
        and _all_zero(formal, QUALIFICATION_FORMAL_ZERO_COUNTERS)
        and type(access) is dict
        and set(access) == set(QUALIFICATION_ACCESS_MONITOR_COUNTERS)
        and _all_zero(access, QUALIFICATION_ACCESS_MONITOR_COUNTERS)
    )


def _live_prerun_evidence_audit(
    *, manifest_path: str | Path, objects: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    """Recompute evidence from live, read-only inputs without backend or RNG use."""

    checks = {
        "qualification_execution_binding_recomputed_pass": False,
        "qualification_result_implementation_binding_pass": False,
        "qualification_binding_live_manifest_contract_pass": False,
        "implementation_file_sha256_live_recomputation_pass": False,
        "seed_provenance_static_live_recomputation_pass": False,
        "historical_v1_artifact_live_reverification_pass": False,
        "historical_v1_recorded_fields_match_pass": False,
        "scientific_core_seven_ast_live_recomputation_pass": False,
        "ideal_lock_live_reaggregation_pass": False,
        "independent_negative_live_reaggregation_pass": False,
        "ideal_backend_controls_live_reaggregation_pass": False,
        "nonideal_1050_live_reaggregation_pass": False,
        "qualification_access_and_formal_counters_zero_pass": False,
    }
    try:
        candidate = Path(manifest_path).resolve()
        repository = candidate.parent.parent.resolve()
        verified_repository, live_manifest = verify_manifest(
            candidate, require_authorized=False
        )
        if verified_repository != repository:
            return checks
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, PermissionError):
        return checks

    controls = objects.get("ideal_backend_control.json", {})
    ideal = objects.get("ideal_parent_lineage_qualification.json", {})
    negative = objects.get("independent_negative_control.json", {})
    nonideal = objects.get("nonideal_scientific_payload_regression.json", {})
    implementation = objects.get("implementation_manifest.json", {})
    seed = objects.get("v2_seed_provenance_audit.json", {})
    v1 = objects.get("v1_failure_binding.json", {})

    try:
        binding = _independent_qualification_execution_binding(repository)
        checks["qualification_execution_binding_recomputed_pass"] = bool(
            controls.get("qualification_execution_binding") == binding
        )
        results = controls.get("results")
        checks["qualification_result_implementation_binding_pass"] = bool(
            type(results) is list
            and len(results) == 42
            and all(
                type(row) is dict
                and row.get("implementation_sha256")
                == binding["qualification_binding_sha256"]
                for row in results
            )
        )
        bound = live_manifest.get("bound_files")
        checks["qualification_binding_live_manifest_contract_pass"] = bool(
            type(bound) is dict
            and all(
                type(bound.get(manifest_name)) is dict
                and bound[manifest_name].get("path")
                == binding["files"][qualification_name]["path"]
                and bound[manifest_name].get("sha256")
                == binding["files"][qualification_name]["sha256"]
                for qualification_name, manifest_name
                in QUALIFICATION_MANIFEST_BINDING_NAMES.items()
            )
        )
        checks["ideal_backend_controls_live_reaggregation_pass"] = (
            _backend_control_reaggregation(controls, ideal, binding)
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError):
        pass

    try:
        checks["implementation_file_sha256_live_recomputation_pass"] = (
            _implementation_file_sha256_audit(repository, implementation)
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError):
        pass

    try:
        from .synthetic_confirmatory_v2_seed_audit import (
            audit_v2_seed_provenance,
        )

        live_seed = audit_v2_seed_provenance(repository)
        checks["seed_provenance_static_live_recomputation_pass"] = bool(
            live_seed.get("STATIC_V2_SEED_PROVENANCE_AUDIT_PASS") is True
            and live_seed.get("NEW_V2_SEED_PROVENANCE_COLLISION_COUNT") == 0
            and live_seed.get("current_seed_literal_unexpected_path_count") == 0
            and live_seed.get("namespace_unexpected_path_count") == 0
            and live_seed.get("test_formal_seed_constructor_hit_count") == 0
            and live_seed.get("base_commit_seed_literal_hits") == []
            and live_seed.get("namespace_base_commit_hits") == []
            and live_seed.get("reference_count") == 3360
            and all(seed.get(key) == value for key, value in live_seed.items())
        )
    except (KeyError, OSError, RuntimeError, TypeError, UnicodeError, ValueError):
        pass

    try:
        from .synthetic_confirmatory_artifact_verifier import (
            verify_synthetic_confirmatory_prerun_artifact as verify_v1_prerun,
        )

        v1_live = verify_v1_prerun(
            repository / "artifacts/synthetic_confirmatory_prerun_v1",
            write_report=False,
        )
        checks["historical_v1_artifact_live_reverification_pass"] = bool(
            v1_live.get("CONFIRMATORY_ARTIFACT_VERIFICATION_PASS") is True
            and v1_live.get("evidence_semantic_failure_count") == 0
            and v1_live.get("sha256_mismatch_files") == []
            and v1_live.get("fixed_pre_run_decision_pass") is True
            and v1_live.get("formal_execution_absence_pass") is True
        )
        checks["historical_v1_recorded_fields_match_pass"] = bool(
            v1.get("historical_v1_artifact_readable")
            is checks["historical_v1_artifact_live_reverification_pass"]
            and v1.get("historical_v1_artifact_semantic_failure_count")
            == v1_live.get("evidence_semantic_failure_count")
            and v1.get("historical_v1_artifact_sha256_mismatch_count")
            == len(v1_live.get("sha256_mismatch_files", []))
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError):
        pass

    try:
        checks["scientific_core_seven_ast_live_recomputation_pass"] = (
            _independent_scientific_core_ast_audit(repository, implementation)
        )
    except (OSError, SyntaxError, TypeError, UnicodeError, ValueError):
        pass
    try:
        checks["ideal_lock_live_reaggregation_pass"] = (
            _ideal_qualification_reaggregation(ideal)
        )
    except (KeyError, TypeError, ValueError):
        pass
    try:
        checks["independent_negative_live_reaggregation_pass"] = (
            _negative_qualification_reaggregation(negative, repository)
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError):
        pass
    try:
        checks["nonideal_1050_live_reaggregation_pass"] = (
            _nonideal_reaggregation(nonideal, repository)
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError):
        pass
    checks["qualification_access_and_formal_counters_zero_pass"] = (
        _qualification_zero_access_audit(controls)
    )
    return checks


def _manifest_transition_audit(
    *,
    manifest_path: str | Path,
    implementation: Mapping[str, Any],
    run: Mapping[str, Any],
    seed_schedule: Mapping[str, Any],
) -> dict[str, bool]:
    """Bind pre-run evidence to the sole predicted false-to-true transition.

    Staging necessarily happens while the live manifest is still the exact
    ``authorized=false`` object.  The identical artifact must continue to
    verify after the one-way transition, so both embedded manifests bind the
    payload and pretty-printed file SHA of ``signed_manifest(..., True)``.
    """

    live_binding_pass = False
    transition_state_pass = False
    implementation_binding_pass = False
    run_binding_pass = False
    seed_schedule_binding_pass = False
    try:
        candidate = Path(manifest_path).resolve()
        repository = candidate.parent.parent.resolve()
        path_pass = candidate == (repository / MANIFEST_RELATIVE).resolve()
        verified_repository, live = verify_manifest(
            candidate, require_authorized=False
        )
        predicted = signed_manifest(repository, authorized=True)
        predicted_payload_sha = predicted["manifest_payload_sha256"]
        predicted_file_sha = hashlib.sha256(_json_bytes(predicted)).hexdigest()
        live_authorization = live.get("formal_execution_authorized")
        live_binding_pass = bool(
            path_pass
            and verified_repository == repository
            and live_authorization in (False, True)
        )
        transition_state_pass = bool(
            live_binding_pass
            and (
                live == predicted
                if live_authorization is True
                else live == signed_manifest(repository, authorized=False)
            )
        )

        def binding_pass(value: Mapping[str, Any]) -> bool:
            return bool(
                value.get("formal_manifest_path") == MANIFEST_RELATIVE.as_posix()
                and value.get("formal_manifest_payload_sha256")
                == predicted_payload_sha
                and value.get("formal_manifest_file_sha256")
                == predicted_file_sha
            )

        implementation_binding_pass = bool(
            binding_pass(implementation)
            and implementation.get("bound_files") == predicted.get("bound_files")
            and implementation.get("formal_branch") == FORMAL_BRANCH
            and implementation.get("formal_output_dir") == FORMAL_OUTPUT_DIR
            and implementation.get("formal_pre_run_tag") == FORMAL_PRERUN_TAG
            and implementation.get("formal_run_id") == FORMAL_RUN_ID
            and implementation.get("formal_workers") == FORMAL_WORKERS
        )
        run_binding_pass = bool(
            binding_pass(run) and run.get("formal_run_id") == FORMAL_RUN_ID
        )
        seed_schedule_binding_pass = bool(
            seed_schedule
            == _strict_json(repository / SEED_SCHEDULE_RELATIVE)
        )
    except (KeyError, OSError, TypeError, ValueError):
        pass
    return {
        "live_manifest_exact_binding_pass": live_binding_pass,
        "live_manifest_transition_state_pass": transition_state_pass,
        "implementation_predicted_authorized_manifest_binding_pass": (
            implementation_binding_pass
        ),
        "run_predicted_authorized_manifest_binding_pass": run_binding_pass,
        "seed_schedule_live_file_binding_pass": seed_schedule_binding_pass,
    }


def _prerun_semantics(
    objects: Mapping[str, Mapping[str, Any]],
    *,
    manifest_checks: Mapping[str, bool],
) -> dict[str, bool]:
    decision = objects.get("final_decision.json", {})
    root_cause = objects.get("root_cause_binding.json", {})
    v1 = objects.get("v1_failure_binding.json", {})
    retirement = objects.get("old_seed_retirement.json", {})
    closure = objects.get("phase_a_closure_semantics_diff.json", {})
    ideal = objects.get("ideal_parent_lineage_qualification.json", {})
    controls = objects.get("ideal_backend_control.json", {})
    negative = objects.get("independent_negative_control.json", {})
    nonideal = objects.get("nonideal_scientific_payload_regression.json", {})
    scientific = objects.get("v1_to_v2_scientific_diff.json", {})
    model = objects.get("frozen_model_binding.json", {})
    seed = objects.get("v2_seed_provenance_audit.json", {})
    plan = objects.get("v2_plan_audit.json", {})
    dry = objects.get("v2_dry_run_report.json", {})
    difference = objects.get("primary_independent_difference.json", {})
    tests = objects.get("test_report.json", {})
    implementation = objects.get("implementation_manifest.json", {})
    run = objects.get("run_manifest.json", {})
    execution_design = objects.get("v2_execution_chain_design.json", {})
    metadata_design = objects.get("v2_metadata_schema.json", {})
    schedule = objects.get("v2_seed_schedule.json", {})
    fixture = objects.get("fixture_regression.json", {})

    required_scientific_zero_names = (
        "scene_difference_count",
        "condition_difference_count",
        "backend_difference_count",
        "backend_parameter_difference_count",
        "planned_snapshot_count_difference",
        "planned_trial_count_difference",
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
        "frozen_model_file_difference_count",
        "frozen_model_feature_difference_count",
        "frozen_model_parameter_difference_count",
    )
    closure_zero_names = tuple(
        name for name in closure if name.endswith("difference_count")
    )
    backend_gates = controls.get("backend_gates")
    backend_gate_pass = bool(
        type(backend_gates) is dict
        and set(backend_gates) == set(BACKENDS)
        and all(
            type(gate) is dict
            and gate.get("QUALIFICATION_BACKEND_GATE_PASS") is True
            and gate.get("trial_count") == 21
            and gate.get("solver_failure_count") == 0
            and gate.get("nonfinite_output_count") == 0
            and _finite_leq(gate.get("translation_q95_m"), 0.001)
            and _finite_leq(
                gate.get("rotation_q95_rad"), 0.00017453292519943296
            )
            and gate.get("quantile_method") == "linear"
            for gate in backend_gates.values()
        )
    )
    nonideal_zero_names = (
        "array_mismatch_count",
        "before_dropout_count_mismatch_count",
        "candidate_noise_dropout_record_missing_count",
        "checksum_mismatch_count",
        "confirmatory_seed_access_count",
        "formal_v2_rng_instantiation_count",
        "metadata_mismatch_count",
        "noise_dropout_record_mismatch_count_where_persisted_baseline_available",
        "record_oracle_mismatch_count",
        "security_firewall_violation_count",
    )
    fixture_required = (
        "schema_version",
        "FIXTURE_EXECUTION_CHAIN_PASS",
        "fixture_snapshot_count",
        "fixture_trial_count",
        "backend_execution_count",
        "resume_backend_execution_count",
        "pairing_mismatch_count",
        "checksum_mismatch_count",
        "fresh_resume_scientific_equivalence",
        "primary_verifier_difference_count",
        "publisher_table_count",
        "publisher_figure_count",
        "publisher_root_file_count",
        "artifact_verification_pass",
        "artifact_path",
        "artifact_file_count",
        "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED",
    )
    formal_count_names = tuple(
        name
        for name in run
        if "formal" in name.lower() and name.lower().endswith("count")
    )
    negative_rng_count_names = tuple(
        name
        for name in negative
        if "rng" in name.lower() and name.lower().endswith("count")
    )
    scientific_core_hashes = implementation.get("scientific_core_ast_hashes")
    test_required = (
        "schema_version",
        "test_report_pass",
        "specialized_confirmatory",
        "full_harness",
        "pcl_backend_v3",
        "source_degen_lio_pytest_executed",
    )
    checks = {
        "required_evidence_objects_pass": all(
            type(objects.get(name)) is dict and bool(objects[name])
            for name in PRERUN_ROOT_FILES
            if name.endswith(".json")
            and name not in {
                "artifact_verification.json",
                "final_decision.json",
                "run_manifest.json",
            }
        ),
        "root_cause_pass": (
            _schema_present(root_cause)
            and root_cause.get("ROOT_CAUSE")
            == "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON"
            and root_cause.get("IMPLEMENTATION_ONLY_REPAIR") is True
            and root_cause.get("legal_parent_row_count") == 44214
            and root_cause.get("bytewise_mismatch_count") == 3439
            and root_cause.get("first_difference_m")
            == 2.9802322387695312e-08
            and root_cause.get("phase_a_closure_residual_max_m") == 0.0
            and root_cause.get("phase_a_closure_violation_count") == 0
        ),
        "v1_failure_preserved_pass": (
            _schema_present(v1)
            and v1.get("V1_FAILURE_RECORD_PRESERVED") is True
            and v1.get("v1_failure_tag")
            == "archive/zero-perturbation-synthetic-confirmatory-v1-ideal-lineage-fail"
            and v1.get("v1_failure_commit")
            == "1c78372ef6f2c62e441f69c028b58f7bc48f6c35"
            and v1.get("v1_bundle_path")
            == "/tmp/zero-perturbation-synthetic-confirmatory-v1-ideal-lineage-fail.bundle"
            and v1.get("v1_bundle_sha256")
            == "21e803756da78c3bf06a93d957c0395850a36d6cf119b23cd00c88ab927dcb72"
            and Path(str(v1.get("v1_bundle_path"))).is_file()
            and file_sha256(Path(str(v1["v1_bundle_path"])))
            == v1["v1_bundle_sha256"]
            and v1.get("SYNTHETIC_CONFIRMATORY_EXECUTED") is False
            and v1.get("SYNTHETIC_CONFIRMATORY_COMPLETE") is False
            and v1.get("SYNTHETIC_CONFIRMATORY_PASS") == "NOT_EVALUATED"
            and v1.get("confirmatory_formal_backend_execution_count") == 0
            and v1.get("confirmatory_formal_trial_result_count") == 0
        ),
        "old_seed_retired_pass": (
            _schema_present(retirement)
            and retirement.get("OLD_V1_SEED_SET_REUSE_AUTHORIZED") is False
            and retirement.get("old_geometry_seeds")
            == list(OLD_V1_GEOMETRY_SEEDS)
            and retirement.get("old_measurement_seeds")
            == list(OLD_V1_MEASUREMENT_SEEDS)
            and retirement.get("old_bootstrap_seed") == OLD_V1_BOOTSTRAP_SEED
        ),
        "phase_a_closure_semantics_pass": bool(
            _schema_present(closure)
            and len(closure_zero_names) >= 7
            and "total_difference_count" in closure_zero_names
            and _all_zero(closure, closure_zero_names)
        ),
        "ideal_lineage_qualification_pass": (
            _schema_present(ideal)
            and ideal.get("IDEAL_GEOMETRY_QUALIFICATION_PASS") is True
            and ideal.get("snapshot_count") == 21
            and ideal.get("lineage_violation_count") == 0
            and ideal.get("measurement_seed_access_count") == 0
            and ideal.get("repeat_randomness_count") == 0
            and ideal.get("rng_instantiation_count") == 0
            and type(ideal.get("qualification_lock")) is dict
        ),
        "ideal_backend_control_pass": (
            controls.get("IDEAL_DUAL_BACKEND_CONTROL_PASS") is True
            and controls.get("backend_execution_count") == 42
            and controls.get("trial_count") == 42
            and type(controls.get("results")) is list
            and len(controls["results"]) == 42
            and backend_gate_pass
        ),
        "independent_negative_control_pass": (
            negative.get("INDEPENDENT_NEGATIVE_CONTROL_PASS") is True
            and negative.get("snapshot_count") == 21
            and negative.get("false_lineage_count") == 21
            and negative.get("array_mismatch_count") == 0
            and negative.get("checksum_mismatch_count") == 0
            and bool(negative_rng_count_names)
            and _all_zero(negative, negative_rng_count_names)
        ),
        "nonideal_regression_pass": (
            nonideal.get(
                "DEVELOPMENT_NONIDEAL_SCIENTIFIC_PAYLOAD_REGRESSION_PASS"
            )
            is True
            and nonideal.get("V2_NONIDEAL_SHARED_PATH_REGRESSION_PASS") is True
            and nonideal.get("snapshot_count") == 1050
            and _all_zero(nonideal, nonideal_zero_names)
            and nonideal.get(
                "NONIDEAL_SCIENTIFIC_PAYLOAD_CHANGE_COUNT",
                nonideal.get("SCIENTIFIC_PAYLOAD_CHANGE_COUNT", 0),
            )
            == 0
        ),
        "scientific_diff_pass": bool(
            _schema_present(scientific)
            and _has_fields(scientific, required_scientific_zero_names)
            and _all_zero(scientific, required_scientific_zero_names)
            and scientific.get("V1_TO_V2_SCIENTIFIC_DIFF_PASS") is True
        ),
        "frozen_model_binding_pass": (
            _schema_present(model)
            and model.get("FROZEN_MODEL_SHA_MATCH") is True
            and model.get("expected_sha256") == FROZEN_MODEL_SHA256
            and model.get("actual_sha256") == FROZEN_MODEL_SHA256
            and model.get("path") == FROZEN_MODEL_RELATIVE.as_posix()
            and model.get("no_refit") is True
        ),
        "seed_schedule_pass": (
            _schema_present(schedule)
            and schedule.get("namespace") == NAMESPACE
            and schedule.get("geometry_seeds") == list(GEOMETRY_SEEDS)
            and schedule.get("measurement_seeds") == list(MEASUREMENT_SEEDS)
            and schedule.get("bootstrap_seed") == BOOTSTRAP_SEED
            and manifest_checks.get("seed_schedule_live_file_binding_pass") is True
        ),
        "seed_provenance_pass": (
            _schema_present(seed)
            and seed.get("NEW_V2_NAMESPACE_COLLISION") is False
            and _all_zero(
                seed,
                (
                    "NEW_V2_SEED_PROVENANCE_COLLISION_COUNT",
                    "NEW_V2_RNG_INSTANTIATION_COUNT",
                    "NEW_V2_SNAPSHOT_CONSTRUCTION_COUNT",
                    "NEW_V2_BACKEND_EXECUTION_COUNT",
                    "NEW_V2_TRIAL_RESULT_COUNT",
                ),
            )
            and type(seed.get("declaration_count")) is int
            and seed["declaration_count"] > 0
            and type(seed.get("reference_count")) is int
            and seed["reference_count"] >= seed["declaration_count"]
        ),
        "execution_chain_design_pass": (
            _schema_present(execution_design)
            and bool(
                execution_design.get("execution_only_changes")
                or execution_design.get("execution_chain")
            )
        ),
        "metadata_design_pass": (
            _schema_present(metadata_design)
            and metadata_design.get("metadata_schema_version")
            == "synthetic_confirmatory_metadata_v2"
            and metadata_design.get("snapshot_schema_version")
            == "synthetic_confirmatory_snapshot_v2"
            and type(metadata_design.get("lineage_fields")) is list
            and all(
                isinstance(value, str)
                for value in metadata_design["lineage_fields"]
            )
            and {
                "lineage_schema_version",
                "source_has_target_parent_lineage",
                "source_parent_target_indices_sha256",
                "parent_index_count",
                "parent_index_unique_count",
                "parent_index_out_of_range_count",
                "lineage_closure_violation_count",
                "lineage_validation_method",
            }.issubset(set(metadata_design["lineage_fields"]))
        ),
        "fixture_regression_pass": (
            _has_fields(fixture, fixture_required)
            and fixture.get("FIXTURE_EXECUTION_CHAIN_PASS") is True
            and fixture.get("fixture_snapshot_count") == 3
            and fixture.get("fixture_trial_count") == 6
            and fixture.get("backend_execution_count") == 6
            and fixture.get("resume_backend_execution_count") == 0
            and fixture.get("pairing_mismatch_count") == 0
            and fixture.get("checksum_mismatch_count") == 0
            and fixture.get("fresh_resume_scientific_equivalence") is True
            and fixture.get("primary_verifier_difference_count") == 0
            and fixture.get("publisher_table_count") == 7
            and fixture.get("publisher_figure_count") == 3
            and fixture.get("publisher_root_file_count") == 7
            and fixture.get("artifact_verification_pass") is True
            and isinstance(fixture.get("artifact_path"), str)
            and fixture.get("artifact_file_count") == 17
            and fixture.get("FORMAL_CONFIRMATORY_SCIENCE_EVALUATED") is False
        ),
        "frozen_model_sha_constant_pass": (
            model.get("actual_sha256")
            == FROZEN_MODEL_SHA256
        ),
        "plan_pass": (
            _schema_present(plan)
            and plan.get("V2_PLAN_PASS") is True
            and plan.get("CONFIRMATORY_PLAN_COUNT_PASS") is True
            and plan.get("CONFIRMATORY_PLAN_UNIQUENESS_PASS") is True
            and plan.get("CONFIRMATORY_PLAN_PAIRING_PASS") is True
            and plan.get("CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS")
            is True
            and plan.get("planned_snapshot_count") == 595
            and plan.get("planned_snapshot_unique_count") == 595
            and plan.get("planned_trial_count") == 1190
            and plan.get("planned_trial_unique_count") == 1190
            and plan.get("native_trial_count") == 0
            and plan.get("duplicate_snapshot_count") == 0
            and plan.get("duplicate_trial_count") == 0
            and plan.get("pairing_violation_count") == 0
            and plan.get("independent_pseudoreplication_plan_count") == 0
        ),
        "dry_run_pass": (
            _schema_present(dry)
            and dry.get("CONFIRMATORY_DRY_RUN_PASS", dry.get("V2_DRY_RUN_PASS")) is True
            and dry.get("planned_snapshot_count") == 595
            and dry.get("planned_snapshot_unique_count") == 595
            and dry.get("planned_trial_count") == 1190
            and dry.get("planned_trial_unique_count") == 1190
            and dry.get("open3d_trial_count") == 595
            and dry.get("pcl_trial_count") == 595
            and dry.get("native_trial_count") == 0
            and dry.get("output_dir_created") is False
            and dry.get("snapshot_cache_dir_created") is False
            and dry.get("snapshot_lock_created") is False
            and _all_zero(
                dry,
                (
                    "NEW_V2_RNG_INSTANTIATION_COUNT",
                    "NEW_V2_SNAPSHOT_CONSTRUCTION_COUNT",
                    "NEW_V2_BACKEND_EXECUTION_COUNT",
                    "NEW_V2_TRIAL_RESULT_COUNT",
                    "NEW_V2_STARTED_EVENT_COUNT",
                ),
            )
        ),
        "primary_independent_match_pass": (
            _schema_present(difference)
            and difference.get("leaf_difference_count") == 0
            and difference.get("section_difference_count") == 0
            and difference.get("exact_match_pass") is True
        ),
        "tests_pass": (
            _has_fields(tests, test_required)
            and tests.get("test_report_pass") is True
            and _test_suite_pass(tests.get("specialized_confirmatory"))
            and _test_suite_pass(tests.get("full_harness"))
            and _test_suite_pass(tests.get("pcl_backend_v3"))
            and tests.get("source_degen_lio_pytest_executed") is False
        ),
        "implementation_manifest_pass": (
            implementation.get("schema_version")
            == "synthetic_confirmatory_prerun_implementation_manifest_v2"
            and implementation.get("formal_manifest_path")
            == "frozen_assets/synthetic_confirmatory_formal_manifest_v2.json"
            and implementation.get("formal_execution_authorized") is True
            and type(scientific_core_hashes) is dict
            and bool(scientific_core_hashes)
            and all(_is_sha256(value) for value in scientific_core_hashes.values())
            and manifest_checks.get(
                "implementation_predicted_authorized_manifest_binding_pass"
            )
            is True
        ),
        "run_manifest_binding_pass": (
            manifest_checks.get(
                "run_predicted_authorized_manifest_binding_pass"
            )
            is True
        ),
        "run_manifest_absence_pass": (
            _schema_present(run)
            and len(formal_count_names) >= 3
            and _all_zero(run, formal_count_names)
            and run.get("final_decision") == decision
            and run.get("formal_run_id") == FORMAL_RUN_ID
        ),
        "fixed_decision_pass": (
            decision.get("SYNTHETIC_CONFIRMATORY_V2_PRE_RUN_QUALIFICATION_PASS") is True
            and decision.get("CONFIRMATORY_V2_RUN_AUTHORIZED") is True
            and decision.get("SYNTHETIC_CONFIRMATORY_V2_EXECUTED") is False
            and decision.get("SYNTHETIC_CONFIRMATORY_V2_COMPLETE") is False
            and decision.get("SYNTHETIC_CONFIRMATORY_V2_PASS") == "NOT_EVALUATED"
            and decision.get("REAL_DATA_RUN_AUTHORIZED") is False
            and decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is False
            and decision.get("ROOT_CAUSE")
            == "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON"
            and decision.get("IMPLEMENTATION_ONLY_REPAIR") is True
            and decision.get("V1_FAILURE_RECORD_PRESERVED") is True
            and decision.get("OLD_V1_SEED_SET_REUSE_AUTHORIZED") is False
            and decision.get("IDEAL_PARENT_INDEX_LINEAGE_IMPLEMENTED") is True
            and decision.get("INDEPENDENT_VERIFIER_LINEAGE_RECOMPUTATION_PASS")
            is True
            and decision.get("FIXTURE_EXECUTION_CHAIN_PASS") is True
            and decision.get("V1_TO_V2_SCIENTIFIC_DIFF_PASS") is True
            and decision.get("FROZEN_MODEL_SHA_MATCH") is True
            and decision.get("V2_PLAN_PASS") is True
            and decision.get("V2_DRY_RUN_PASS") is True
            and decision.get("V2_ARTIFACT_PUBLICATION_PASS") is True
            and decision.get("V2_ARTIFACT_VERIFICATION_PASS") is True
            and _all_zero(
                decision,
                (
                    "PHASE_A_CLOSURE_SEMANTICS_DIFF_COUNT",
                    "IDEAL_LINEAGE_VIOLATION_COUNT",
                    "NONIDEAL_SCIENTIFIC_PAYLOAD_CHANGE_COUNT",
                    "PRIMARY_VERIFIER_DIFFERENCE_COUNT",
                    "NEW_V2_SEED_PROVENANCE_COLLISION_COUNT",
                    "NEW_V2_RNG_INSTANTIATION_COUNT",
                    "NEW_V2_SNAPSHOT_CONSTRUCTION_COUNT",
                    "NEW_V2_BACKEND_EXECUTION_COUNT",
                    "NEW_V2_TRIAL_RESULT_COUNT",
                ),
            )
        ),
        **dict(manifest_checks),
    }
    return checks


def verify_synthetic_confirmatory_v2_prerun_artifact(
    path: str | Path,
    *,
    manifest_path: str | Path,
    write_report: bool = False,
) -> dict[str, Any]:
    """Verify the exact 26-root-file pre-run artifact and fixture subtree."""

    root = Path(path).resolve()
    actual = _actual_files(root)
    fixture_prefix = f"{PRERUN_FIXTURE_DIRECTORY}/"
    fixture_actual = {name for name in actual if name.startswith(fixture_prefix)}
    expected_fixture = {
        f"{fixture_prefix}{name}" for name in FIXTURE_REQUIRED_FILES
    }
    root_actual = {name for name in actual if "/" not in name}
    virtual_root = set(root_actual)
    if write_report:
        virtual_root.add("artifact_verification.json")
    expected_root = set(PRERUN_ROOT_FILES)
    missing = sorted((expected_root - virtual_root) | (expected_fixture - fixture_actual))
    extra = sorted(
        (root_actual - expected_root)
        | (fixture_actual - expected_fixture)
        | {
            name
            for name in actual
            if "/" in name and not name.startswith(fixture_prefix)
        }
    )
    directory_errors = _directory_errors(
        root,
        {
            PRERUN_FIXTURE_DIRECTORY,
            f"{PRERUN_FIXTURE_DIRECTORY}/tables",
            f"{PRERUN_FIXTURE_DIRECTORY}/figures",
        },
    )
    symlink_errors = _symlink_errors(root)

    objects: dict[str, dict[str, Any]] = {}
    invalid_json = []
    for name in PRERUN_ROOT_FILES:
        if not name.endswith(".json") or name == "artifact_verification.json":
            continue
        candidate = root / name
        if not candidate.is_file():
            continue
        try:
            objects[name] = _strict_json(candidate)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            invalid_json.append(name)
    manifest_checks = _manifest_transition_audit(
        manifest_path=manifest_path,
        implementation=objects.get("implementation_manifest.json", {}),
        run=objects.get("run_manifest.json", {}),
        seed_schedule=objects.get("v2_seed_schedule.json", {}),
    )
    manifest_checks.update(
        _live_prerun_evidence_audit(
            manifest_path=manifest_path,
            objects=objects,
        )
    )
    semantic_checks = _prerun_semantics(
        objects, manifest_checks=manifest_checks
    )
    semantic_failures = sorted(
        name for name, passed in semantic_checks.items() if not passed
    )
    fixture = verify_synthetic_confirmatory_v2_fixture_artifact(
        root / PRERUN_FIXTURE_DIRECTORY, write_report=False
    )
    inventory_expected = (
        (expected_root - PRERUN_SPECIAL_INVENTORY_FILES) | expected_fixture
    )
    checksums = _checksum_audit(root, inventory_expected)
    manifest = _manifest_csv_audit(root, inventory_expected)
    report_path = root / "pre_run_report.md"
    markdown = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    report_pass = all(
        token in markdown
        for token in (
            "SYNTHETIC_CONFIRMATORY_V2_PRE_RUN_QUALIFICATION_PASS",
            "CONFIRMATORY_V2_RUN_AUTHORIZED",
            "SYNTHETIC_CONFIRMATORY_V2_EXECUTED",
            "SYNTHETIC_CONFIRMATORY_V2_PASS",
            "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON",
        )
    )
    result = {
        "schema_version": PRERUN_ARTIFACT_SCHEMA,
        "actual_file_count": len(actual) + int(write_report and "artifact_verification.json" not in actual),
        "directory_inventory_errors": directory_errors,
        "evidence_semantic_checks": semantic_checks,
        "evidence_semantic_failures": semantic_failures,
        "extra_files": extra,
        "fixture_artifact_verification": fixture,
        "fixture_artifact_verification_pass": fixture.get(
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        )
        is True,
        "invalid_json_files": invalid_json,
        "missing_required_files": missing,
        "pre_run_report_semantics_pass": report_pass,
        "required_file_count": len(expected_root) + len(expected_fixture),
        "symlink_paths": symlink_errors,
        **checksums,
        **manifest,
    }
    result["V2_PRERUN_ARTIFACT_VERIFICATION_PASS"] = bool(
        not missing
        and not extra
        and not directory_errors
        and not symlink_errors
        and not invalid_json
        and not semantic_failures
        and fixture.get("FIXTURE_ARTIFACT_VERIFICATION_PASS") is True
        and checksums["sha256_verification_pass"]
        and manifest["manifest_verification_pass"]
        and report_pass
    )
    result["V2_ARTIFACT_VERIFICATION_PASS"] = result[
        "V2_PRERUN_ARTIFACT_VERIFICATION_PASS"
    ]
    if write_report:
        _write_verification(root / "artifact_verification.json", result)
    return result


def verify_v2_prerun_artifact(
    path: str | Path,
    *,
    manifest_path: str | Path,
    write_report: bool = False,
) -> dict[str, Any]:
    """Stable pre-run verifier entry point used by freeze/authorization code."""

    return verify_synthetic_confirmatory_v2_prerun_artifact(
        path,
        manifest_path=manifest_path,
        write_report=write_report,
    )


__all__ = [
    "FIXTURE_FIGURES",
    "FIXTURE_REQUIRED_FILES",
    "FIXTURE_ROOT_FILES",
    "FIXTURE_TABLES",
    "FORMAL_FIGURES",
    "FORMAL_REQUIRED_FILES",
    "FORMAL_ROOT_FILES",
    "FORMAL_TABLES",
    "PRERUN_ROOT_FILES",
    "audit_synthetic_confirmatory_v2_fixture_rows",
    "verify_synthetic_confirmatory_v2_artifact",
    "verify_synthetic_confirmatory_v2_fixture_artifact",
    "verify_synthetic_confirmatory_v2_prerun_artifact",
    "verify_v2_prerun_artifact",
]
