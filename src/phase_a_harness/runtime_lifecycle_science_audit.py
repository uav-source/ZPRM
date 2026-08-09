"""Read-only scientific invariance checks for runtime lifecycle qualification.

The lifecycle repair is intentionally outside the scientific analysis path.
This module binds the existing fixture payload, H1--H6 functions, frozen model,
and backend implementations to the already-published Confirmatory-v2 pre-run
artifact.  It never imports or reads a Confirmatory seed schedule.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import file_sha256


BASELINE_IMPLEMENTATION = Path(
    "artifacts/synthetic_confirmatory_v2_prerun/implementation_manifest.json"
)
BASELINE_FIXTURE_PRIMARY = Path(
    "artifacts/synthetic_confirmatory_v2_prerun/fixture_publication/primary_analysis.json"
)

SCIENTIFIC_FILE_BINDINGS = (
    "backend_metrics",
    "backend_parameter_contract",
    "backend_types",
    "common_association",
    "fixture_backend_parameter_lock",
    "fixture_plan",
    "fixture_snapshot_lock",
    "frozen_model",
    "frozen_model_inference",
    "full_synthetic_backend_execution",
    "full_synthetic_snapshot_builder",
    "generator",
    "generator_capture_init",
    "generator_capture_protocol",
    "generator_capture_types",
    "generator_package_init",
    "generator_wrapper",
    "generator_zero_init",
    "generator_zero_protocol",
    "generator_zero_snapshot_builder",
    "generator_zero_types",
    "independent_scientific_core",
    "open3d_adapter",
    "pcl_adapter",
    "pcl_cli",
    "phase_a_lineage",
    "primary_scientific_core",
    "rotation_metrics",
    "trial_schema",
    "trial_schema_validator",
    "v2_analysis",
    "v2_independent_verifier",
    "v2_snapshot_builder",
)

AST_BINDINGS: Mapping[str, tuple[str, str, str]] = {
    "phase_a_canonical_target": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_v1_2.py",
        "canonical_target",
        "5684f5bbcbd9001be0b29fcdd670470d0e19d7758d698fc39aaba4b73be3a5c8",
    ),
    "phase_a_eligible_parent_indices": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_v1_2.py",
        "eligible_parent_indices",
        "f8fd5aed7876e30cacfac41c1b699518d04e7c134f23d189c8f217f9af5e275a",
    ),
    "phase_a_quantization_closure": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_v1_2.py",
        "quantization_closure",
        "2ec95cab5329881772ccd56518b73705efded616254710d013db95324594ebac",
    ),
    "phase_a_reference_pose": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_v1_2.py",
        "reference_pose_from_development",
        "c9f8faa358bc09ee6e226547ec936f9bb2e95e7f98cac662b3a10da8b608caa1",
    ),
    "phase_a_source_from_parent_indices": (
        "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_v1_2.py",
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


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON object required: {path}")
    return value


def _function_ast_sha256(path: Path, function_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(nodes) != 1:
        raise ValueError(f"expected one function {function_name!r} in {path}")
    return hashlib.sha256(
        ast.dump(nodes[0], include_attributes=False).encode("utf-8")
    ).hexdigest()


def scientific_core_binding(repository: str | Path) -> dict[str, Any]:
    root = Path(repository).resolve()
    baseline = _strict_object(root / BASELINE_IMPLEMENTATION)
    bound = baseline.get("bound_files")
    if type(bound) is not dict:
        raise ValueError("baseline implementation bound_files is missing")
    files: list[dict[str, Any]] = []
    for name in SCIENTIFIC_FILE_BINDINGS:
        entry = bound.get(name)
        if type(entry) is not dict:
            raise ValueError(f"missing baseline scientific binding: {name}")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"invalid baseline scientific binding: {name}")
        candidate = (root / relative).resolve()
        if root not in candidate.parents or candidate.is_symlink():
            raise ValueError(f"unsafe scientific binding path: {relative}")
        actual = file_sha256(candidate)
        files.append(
            {
                "binding": name,
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )
    ast_records = []
    for name, (relative, function, expected) in AST_BINDINGS.items():
        actual = _function_ast_sha256(root / relative, function)
        ast_records.append(
            {
                "binding": name,
                "path": relative,
                "function": function,
                "expected_ast_sha256": expected,
                "actual_ast_sha256": actual,
                "match": actual == expected,
            }
        )
    file_changes = sum(not row["match"] for row in files)
    ast_changes = sum(not row["match"] for row in ast_records)
    names = {row["binding"]: row for row in files}
    return {
        "schema_version": "runtime_lifecycle_scientific_core_binding_v1",
        "SCIENTIFIC_CORE_FILE_CHANGE_COUNT": file_changes,
        "H1_H6_SEMANTICS_CHANGE_COUNT": ast_changes,
        "FROZEN_MODEL_CHANGE_COUNT": int(not names["frozen_model"]["match"]),
        "BACKEND_BINDING_CHANGE_COUNT": sum(
            not names[name]["match"]
            for name in (
                "backend_parameter_contract",
                "open3d_adapter",
                "pcl_adapter",
                "pcl_cli",
            )
        ),
        "ast_bindings": ast_records,
        "file_bindings": files,
    }


def _scientific_trial(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "runtime_ms"}


def fixture_scientific_regression(
    repository: str | Path, current_results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    root = Path(repository).resolve()
    historical = _strict_object(root / BASELINE_FIXTURE_PRIMARY)
    historical_rows = historical.get("results")
    if type(historical_rows) is not list:
        raise ValueError("historical fixture results are missing")
    baseline = {
        str(row["planned_trial_id"]): _scientific_trial(row)
        for row in historical_rows
    }
    current = {
        str(row["planned_trial_id"]): _scientific_trial(row)
        for row in current_results
    }
    ids = sorted(set(baseline) | set(current))
    field_differences: list[dict[str, Any]] = []
    for trial_id in ids:
        left = baseline.get(trial_id)
        right = current.get(trial_id)
        keys = sorted(set(left or {}) | set(right or {}))
        for key in keys:
            if (left or {}).get(key) != (right or {}).get(key):
                field_differences.append(
                    {
                        "planned_trial_id": trial_id,
                        "field": key,
                        "baseline": (left or {}).get(key),
                        "current": (right or {}).get(key),
                    }
                )
    checksum_fields = (
        "snapshot_checksum",
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
    )
    payload_change_count = sum(
        1
        for trial_id in ids
        for field in checksum_fields
        if (baseline.get(trial_id) or {}).get(field)
        != (current.get(trial_id) or {}).get(field)
    )
    return {
        "schema_version": "runtime_lifecycle_fixture_scientific_regression_v1",
        "FIXTURE_SCIENTIFIC_PAYLOAD_CHANGE_COUNT": payload_change_count,
        "FIXTURE_TRIAL_SCIENTIFIC_FIELD_CHANGE_COUNT": len(field_differences),
        "baseline_trial_count": len(baseline),
        "current_trial_count": len(current),
        "excluded_fields": ["runtime_ms"],
        "field_differences": field_differences,
    }


__all__ = [
    "AST_BINDINGS",
    "SCIENTIFIC_FILE_BINDINGS",
    "fixture_scientific_regression",
    "scientific_core_binding",
]
