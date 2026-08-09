"""Post-run adapters for the requalified local Synthetic Confirmatory v3 run.

This module contains only lifecycle component adapters.  It does not alter the
frozen H1--H6 implementation, gate contract, model lock, snapshot builder, or
backend code.  The public callables deliberately match the signatures consumed
by :func:`phase_a_harness.formal_lifecycle.execute_postrun_pipeline`.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


EXECUTION_CLASS = "REQUALIFIED_LOCAL_EXECUTION"
REQUALIFIED_RUN_SCHEMA = "synthetic_confirmatory_v3_requalified_formal_run_v1"
REQUALIFIED_DIFFERENCE_SCHEMA = (
    "synthetic_confirmatory_v3_requalified_primary_independent_difference_v1"
)
REQUALIFIED_ARTIFACT_VERIFICATION_SCHEMA = (
    "synthetic_confirmatory_v3_requalified_artifact_verification_v1"
)

_PROTOCOL_RELATIVE = Path("protocols/synthetic_confirmatory_protocol_v3.json")
_GATE_RELATIVE = Path("protocols/synthetic_confirmatory_gate_contract_v3.json")
_MODEL_RELATIVE = Path(
    "frozen_assets/confirmatory_development_trained_models_v1.json"
)
_EXPECTED_SNAPSHOT_COUNT = 595
_EXPECTED_TRIAL_COUNT = 1190
_EXPECTED_BACKEND_COUNTS = {
    "open3d_point_to_plane": 595,
    "pcl_point_to_plane": 595,
}


class RequalifiedPostrunError(RuntimeError):
    """Authenticated requalified post-run evidence could not be processed."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(path: Path) -> dict[str, Any]:
    from .runtime_lifecycle_io import read_regular_bytes, strict_json_loads

    value = strict_json_loads(read_regular_bytes(path))
    if type(value) is not dict:
        raise RequalifiedPostrunError(f"JSON root must be an object: {path}")
    return value


def _thaw(value: Any) -> Any:
    from .formal_lifecycle_contract import deep_thaw

    return deep_thaw(value)


def _assert_completed_run(spec: Any, completed_run: Any) -> None:
    if getattr(completed_run, "spec", None) is not spec:
        raise RequalifiedPostrunError("completed run belongs to another spec")
    if len(getattr(spec, "snapshot_plan", ())) != _EXPECTED_SNAPSHOT_COUNT:
        raise RequalifiedPostrunError("requalified snapshot plan is not 595 rows")
    if len(getattr(spec, "trial_plan", ())) != _EXPECTED_TRIAL_COUNT:
        raise RequalifiedPostrunError("requalified trial plan is not 1,190 rows")
    if len(getattr(completed_run, "rows", ())) != _EXPECTED_TRIAL_COUNT:
        raise RequalifiedPostrunError("completed result inventory is not 1,190 rows")
    manifest = getattr(spec, "manifest", {})
    declared = (
        manifest.get("execution_class"),
        manifest.get("execution_classification"),
    )
    if any(value not in (None, EXECUTION_CLASS) for value in declared):
        raise RequalifiedPostrunError("manifest execution class is not requalified")


def _plan_rows(spec: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snapshots = [_thaw(row) for row in spec.snapshot_plan]
    trials = [_thaw(row) for row in spec.trial_plan]
    snapshot_ids = [str(row.get("planned_snapshot_id", "")) for row in snapshots]
    trial_ids = [str(row.get("planned_trial_id", "")) for row in trials]
    if (
        any(not value for value in (*snapshot_ids, *trial_ids))
        or len(set(snapshot_ids)) != _EXPECTED_SNAPSHOT_COUNT
        or len(set(trial_ids)) != _EXPECTED_TRIAL_COUNT
    ):
        raise RequalifiedPostrunError("frozen plan identity inventory is invalid")
    return snapshots, trials


def _enriched_completed_rows(
    spec: Any, completed_run: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind every strict 26-field result back to its frozen v3 trial row."""

    _assert_completed_run(spec, completed_run)
    snapshots, trials = _plan_rows(spec)
    plan_by_id = {str(row["planned_trial_id"]): row for row in trials}
    raw_rows = [_thaw(row) for row in completed_run.rows]
    raw_by_id: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        if type(row) is not dict:
            raise RequalifiedPostrunError("completed trial result is not an object")
        trial_id = str(row.get("planned_trial_id", ""))
        if not trial_id or trial_id in raw_by_id:
            raise RequalifiedPostrunError(
                "completed trial identity is missing or duplicated"
            )
        raw_by_id[trial_id] = row
    if set(raw_by_id) != set(plan_by_id):
        raise RequalifiedPostrunError("completed trials differ from the frozen plan")

    enriched: list[dict[str, Any]] = []
    for trial_id in sorted(plan_by_id):
        plan = plan_by_id[trial_id]
        row = raw_by_id[trial_id]
        expected = {
            "planned_trial_id": trial_id,
            "snapshot_id": plan["planned_snapshot_id"],
            "scene_variant": plan["scene_variant"],
            "condition": plan["condition"],
            "backend": plan["backend"],
        }
        if any(row.get(name) != value for name, value in expected.items()):
            raise RequalifiedPostrunError(
                f"completed trial differs from its frozen row: {trial_id}"
            )
        enriched.append(
            {
                **row,
                "backend_schema_name": plan["backend"],
                "geometry_seed": plan["geometry_seed"],
                "measurement_seed": plan["measurement_seed"],
                "planned_snapshot_id": plan["planned_snapshot_id"],
                "repeat_index": plan["repeat_index"],
            }
        )
    return enriched, snapshots, trials


def _minimal_v3_stack(
    spec: Any,
    completed_run: Any,
    snapshots: Sequence[Mapping[str, Any]],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build only the authenticated fields consumed by v3 ``_recompute_common``."""

    from .runtime_lifecycle_io import read_canonical_json
    from .trial_snapshot_bridge import (
        build_canonical_snapshot_index,
        build_trial_snapshot_bindings,
    )

    lock_value = read_canonical_json(spec.paths.snapshot_lock)
    entries = lock_value.get("snapshots") if type(lock_value) is dict else None
    if type(entries) is not list:
        raise RequalifiedPostrunError("requalified snapshot lock is malformed")
    lock_by_id: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RequalifiedPostrunError("snapshot lock entry is not an object")
        snapshot_id = str(entry.get("snapshot_id", ""))
        if not snapshot_id or snapshot_id in lock_by_id:
            raise RequalifiedPostrunError(
                "snapshot lock identity is missing or duplicated"
            )
        lock_by_id[snapshot_id] = entry
    expected_ids = {str(row["planned_snapshot_id"]) for row in snapshots}
    if set(lock_by_id) != expected_ids:
        raise RequalifiedPostrunError(
            "snapshot lock differs from the frozen snapshot plan"
        )
    snapshot_lock_sha256 = _file_sha256(spec.paths.snapshot_lock)
    if snapshot_lock_sha256 != completed_run.snapshot_lock_sha256:
        raise RequalifiedPostrunError("snapshot lock changed before analysis")

    index = build_canonical_snapshot_index(
        snapshots,
        execution_context=spec.execution_context,
    )
    bindings = build_trial_snapshot_bindings(index, trials)
    return {
        "canonical_snapshot_index": index,
        "execution_context": spec.execution_context,
        "lock_by_id": lock_by_id,
        "runtime_paths": {"snapshot_cache": spec.paths.snapshot_cache},
        "snapshot_lock_sha256": snapshot_lock_sha256,
        "snapshots": list(snapshots),
        "trials": list(trials),
        "trial_snapshot_bindings": bindings,
    }


def _normalized_scientific_records(
    spec: Any,
    completed_run: Any,
    *,
    independent: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from .synthetic_confirmatory_v3_runner import _recompute_common

    enriched, snapshots, trials = _enriched_completed_rows(spec, completed_run)
    stack = _minimal_v3_stack(spec, completed_run, snapshots, trials)
    return _recompute_common(enriched, stack, independent=independent)


def _scientific_assets(
    spec: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(spec.repository_root)
    protocol_path = root / _PROTOCOL_RELATIVE
    gate_path = root / _GATE_RELATIVE
    model_path = root / _MODEL_RELATIVE
    frozen_hashes = getattr(spec, "manifest", {}).get("scientific_asset_sha256")
    if not isinstance(frozen_hashes, Mapping) or any(
        frozen_hashes.get(relative.as_posix()) != _file_sha256(path)
        for relative, path in (
            (_PROTOCOL_RELATIVE, protocol_path),
            (_GATE_RELATIVE, gate_path),
            (_MODEL_RELATIVE, model_path),
        )
    ):
        raise RequalifiedPostrunError("frozen v3 scientific asset hash changed")
    protocol = _strict_object(protocol_path)
    gate = _strict_object(gate_path)
    model = _strict_object(model_path)
    if (
        protocol.get("schema_version") != "synthetic_confirmatory_protocol_v3"
        or gate.get("schema_version") != "synthetic_confirmatory_gate_contract_v3"
        or protocol.get("planned_snapshot_count") != _EXPECTED_SNAPSHOT_COUNT
        or protocol.get("planned_trial_count") != _EXPECTED_TRIAL_COUNT
        or protocol.get("development_model_lock_sha256") != _file_sha256(model_path)
        or protocol.get("gate_contract_payload_sha256")
        != gate.get("gate_contract_payload_sha256")
    ):
        raise RequalifiedPostrunError("frozen v3 scientific asset binding failed")
    geometries = protocol.get("geometry_seeds")
    if (
        type(geometries) is not list
        or len(geometries) != 5
        or len(set(geometries)) != 5
    ):
        raise RequalifiedPostrunError("frozen v3 geometry-seed inventory changed")
    return protocol, gate, model


def _bind_analysis_identity(
    result: Mapping[str, Any],
    *,
    spec: Any,
    completed_run: Any,
    independent: bool,
) -> dict[str, Any]:
    from .synthetic_confirmatory_v3_runner import (
        INDEPENDENT_SCHEMA,
        PRIMARY_ANALYSIS_SCHEMA,
        _adapt_v3_analysis,
    )

    output = _adapt_v3_analysis(result, independent=independent)
    output["schema_version"] = (
        INDEPENDENT_SCHEMA if independent else PRIMARY_ANALYSIS_SCHEMA
    )
    output["run_id"] = spec.run_id
    output["raw_result_manifest_sha256"] = completed_run.raw_manifest_sha256
    output["execution_class"] = EXECUTION_CLASS
    output["execution_classification"] = EXECUTION_CLASS
    return output


def primary_analyzer(*, spec: Any, completed_run: Any) -> dict[str, Any]:
    """Recompute the frozen primary H1--H6 analysis from authenticated results."""

    from .synthetic_confirmatory_analysis import analyze_synthetic_confirmatory_records

    normalized, common = _normalized_scientific_records(
        spec,
        completed_run,
        independent=False,
    )
    protocol, gate, model = _scientific_assets(spec)
    result = analyze_synthetic_confirmatory_records(
        trials=normalized,
        common_records=common,
        model_lock=model,
        gate_contract=gate,
        expected_geometry_seeds=protocol["geometry_seeds"],
    )
    return _bind_analysis_identity(
        result,
        spec=spec,
        completed_run=completed_run,
        independent=False,
    )


def independent_verifier(*, spec: Any, completed_run: Any) -> dict[str, Any]:
    """Independently recompute common metrics, H1--H6, and the v3 decision."""

    from .synthetic_confirmatory_independent_verifier import (
        independently_recompute_synthetic_confirmatory,
    )

    normalized, common = _normalized_scientific_records(
        spec,
        completed_run,
        independent=True,
    )
    protocol, gate, model = _scientific_assets(spec)
    result = independently_recompute_synthetic_confirmatory(
        trials=normalized,
        common_records=common,
        model_lock=model,
        gate_contract=gate,
        expected_geometry_seeds=protocol["geometry_seeds"],
    )
    return _bind_analysis_identity(
        result,
        spec=spec,
        completed_run=completed_run,
        independent=True,
    )


def difference_auditor(
    *,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the frozen primary and independent projections to match exactly."""

    from .synthetic_confirmatory_independent_verifier import (
        compare_primary_and_independent,
    )

    comparison = compare_primary_and_independent(primary, independent)
    return {
        **comparison,
        "schema_version": REQUALIFIED_DIFFERENCE_SCHEMA,
        "execution_class": EXECUTION_CLASS,
        "execution_classification": EXECUTION_CLASS,
    }


def _artifact_run_manifest(spec: Any, completed_run: Any) -> dict[str, Any]:
    value = _thaw(completed_run.run_manifest)
    if type(value) is not dict:
        raise RequalifiedPostrunError("completed run manifest is not an object")
    source_schema = value.get("schema_version")
    value.update(
        {
            "schema_version": REQUALIFIED_RUN_SCHEMA,
            "source_lifecycle_run_manifest_schema": source_schema,
            "execution_class": EXECUTION_CLASS,
            "execution_classification": EXECUTION_CLASS,
            "run_id": spec.run_id,
            "raw_result_manifest_sha256": completed_run.raw_manifest_sha256,
        }
    )
    return value


def _replace_publication_envelope(
    root: Path,
    *,
    spec: Any,
    completed_run: Any,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
) -> None:
    from .runtime_lifecycle_io import (
        atomic_replace_bytes,
        atomic_replace_canonical_json,
    )

    decision = primary.get("final_decision")
    if type(decision) is not dict or independent.get("final_decision") != decision:
        raise RequalifiedPostrunError("v3 final decisions are absent or inconsistent")
    for name, value in (
        ("primary_analysis.json", dict(primary)),
        ("independent_verification.json", dict(independent)),
        ("final_decision.json", dict(decision)),
        ("run_manifest.json", _artifact_run_manifest(spec, completed_run)),
    ):
        atomic_replace_canonical_json(root / name, value)

    report_path = root / "synthetic_confirmatory_report.md"
    report = report_path.read_text(encoding="utf-8")
    if not report.startswith("# Synthetic Confirmatory v1\n"):
        raise RequalifiedPostrunError("delegated publisher report identity changed")
    report = report.replace(
        "# Synthetic Confirmatory v1\n",
        (
            "# Synthetic Confirmatory v3 — Requalified Local Execution\n\n"
            f"- Execution class: `{EXECUTION_CLASS}`\n"
            f"- Requalified run ID: `{spec.run_id}`\n"
        ),
        1,
    )
    for generic, v3 in (
        ("SYNTHETIC_CONFIRMATORY_EXECUTED", "SYNTHETIC_CONFIRMATORY_V3_EXECUTED"),
        ("SYNTHETIC_CONFIRMATORY_COMPLETE", "SYNTHETIC_CONFIRMATORY_V3_COMPLETE"),
        ("SYNTHETIC_CONFIRMATORY_PASS", "SYNTHETIC_CONFIRMATORY_V3_PASS"),
        ("CONFIRMATORY_RUN_AUTHORIZED", "CONFIRMATORY_V3_RUN_AUTHORIZED"),
    ):
        report = report.replace(generic, v3)
    atomic_replace_bytes(report_path, (report.rstrip("\n") + "\n").encode("utf-8"))
    _rebuild_sha256sums(root)


def _rebuild_sha256sums(root: Path) -> None:
    from .runtime_lifecycle_io import atomic_replace_bytes

    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.name not in {"SHA256SUMS", "artifact_verification.json"}
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    payload = "".join(
        f"{_file_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in paths
    ).encode("utf-8")
    atomic_replace_bytes(root / "SHA256SUMS", payload)


def publisher(
    *,
    spec: Any,
    completed_run: Any,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Atomically publish the frozen 7/3/7 artifact with requalified identity."""

    from .runtime_lifecycle_io import atomic_publish_directory
    from .synthetic_confirmatory_independent_verifier import (
        compare_primary_and_independent,
    )
    from .synthetic_confirmatory_publisher import _publish_into
    from .synthetic_confirmatory_v3_runner import _genericize_v3_analysis

    _assert_completed_run(spec, completed_run)
    if primary.get("execution_class") != EXECUTION_CLASS or independent.get(
        "execution_class"
    ) != EXECUTION_CLASS:
        raise RequalifiedPostrunError("publisher inputs lack requalified identity")
    comparison = compare_primary_and_independent(primary, independent)
    if comparison.get("exact_match_pass") is not True:
        raise RequalifiedPostrunError("primary and independent v3 analyses differ")
    expected_destination = spec.paths.publisher_staging / "formal_publication"
    if Path(destination) != expected_destination:
        raise RequalifiedPostrunError(
            "publisher destination differs from lifecycle path"
        )

    generic_primary = _genericize_v3_analysis(primary, independent=False)
    generic_independent = _genericize_v3_analysis(independent, independent=True)
    run_manifest = _artifact_run_manifest(spec, completed_run)
    written: dict[str, Any] = {}

    def populate(staging_root: Path) -> None:
        inner = staging_root / "artifact"
        _publish_into(
            inner,
            primary=generic_primary,
            independent=generic_independent,
            run_manifest=run_manifest,
        )
        _replace_publication_envelope(
            inner,
            spec=spec,
            completed_run=completed_run,
            primary=primary,
            independent=independent,
        )
        recorded = artifact_verifier(
            spec=spec,
            artifact_path=inner,
            write_report=True,
        )
        live = artifact_verifier(
            spec=spec,
            artifact_path=inner,
            write_report=False,
        )
        if recorded != live or live.get(
            "REQUALIFIED_V3_ARTIFACT_VERIFICATION_PASS"
        ) is not True:
            raise RequalifiedPostrunError(
                "requalified staged artifact verification failed"
            )
        written.update(live)
        for child in tuple(inner.iterdir()):
            os.replace(child, staging_root / child.name)
        inner.rmdir()

    atomic_publish_directory(destination, populate)
    final = artifact_verifier(
        spec=spec,
        artifact_path=destination,
        write_report=False,
    )
    if final != written or final.get(
        "REQUALIFIED_V3_ARTIFACT_VERIFICATION_PASS"
    ) is not True:
        raise RequalifiedPostrunError(
            "published requalified artifact failed verification"
        )
    return {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": comparison["leaf_difference_count"],
        "REQUALIFIED_V3_ARTIFACT_VERIFICATION_PASS": True,
        "execution_class": EXECUTION_CLASS,
        "execution_classification": EXECUTION_CLASS,
        "artifact_path": str(destination),
        "publication_path": str(destination),
        "published_file_count": final["actual_file_count"],
        "publisher_execution_count_this_invocation": 1,
        "sha256_mismatch_count": len(final["sha256_mismatch_files"]),
    }


def _strict_737_inventory(root: Path, spec: Any) -> dict[str, Any]:
    from .synthetic_confirmatory_v3_contract import (
        PUBLISHER_FIGURES,
        PUBLISHER_ROOT_FILES,
        PUBLISHER_TABLES,
    )

    if not root.is_dir() or root.is_symlink():
        raise RequalifiedPostrunError("artifact root is not a real directory")

    def entries(directory: Path) -> tuple[list[str], list[str]]:
        files: list[str] = []
        directories: list[str] = []
        with os.scandir(directory) as iterator:
            for entry in iterator:
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise RequalifiedPostrunError(
                        f"artifact contains a symbolic link: {entry.path}"
                    )
                if stat.S_ISREG(metadata.st_mode):
                    files.append(entry.name)
                elif stat.S_ISDIR(metadata.st_mode):
                    directories.append(entry.name)
                else:
                    raise RequalifiedPostrunError(
                        f"artifact contains a special file: {entry.path}"
                    )
        return sorted(files), sorted(directories)

    root_files, root_directories = entries(root)
    if root_directories != ["figures", "tables"]:
        raise RequalifiedPostrunError("artifact root directory inventory changed")
    tables, table_directories = entries(root / "tables")
    figures, figure_directories = entries(root / "figures")
    expected_tables = tuple(PUBLISHER_TABLES)
    expected_figures = tuple(PUBLISHER_FIGURES)
    expected_roots = tuple(PUBLISHER_ROOT_FILES)
    declared = spec.publisher_inventory
    declared_pass = bool(
        tuple(declared.tables) == expected_tables
        and tuple(declared.figures) == expected_figures
        and tuple(declared.root_files) == expected_roots
    )
    inventory_pass = bool(
        declared_pass
        and not table_directories
        and not figure_directories
        and set(tables) == set(expected_tables)
        and set(figures) == set(expected_figures)
        and set(root_files) == set(expected_roots)
    )
    return {
        "tables": tables,
        "figures": figures,
        "root_files": root_files,
        "table_count": len(tables),
        "figure_count": len(figures),
        "root_file_count": len(root_files),
        "declared_frozen_inventory_pass": declared_pass,
        "publisher_7_table_3_figure_7_root_pass": inventory_pass,
    }


def artifact_verifier(
    *,
    spec: Any,
    artifact_path: Path,
    write_report: bool,
) -> dict[str, Any]:
    """Verify exact 7/3/7 contents and the requalified v3 evidence identity."""

    from .runtime_lifecycle_io import (
        assert_no_symlink_path,
        atomic_create_canonical_json,
        atomic_replace_canonical_json,
    )
    from .synthetic_confirmatory_artifact_verifier import (
        verify_synthetic_confirmatory_artifact,
    )
    from .synthetic_confirmatory_independent_verifier import (
        compare_primary_and_independent,
    )
    from .synthetic_confirmatory_v3_runner import (
        INDEPENDENT_SCHEMA,
        PRIMARY_ANALYSIS_SCHEMA,
        _validate_v3_final_decision,
    )

    if type(write_report) is not bool:
        raise TypeError("write_report must be bool")
    root = assert_no_symlink_path(Path(artifact_path))
    inventory = _strict_737_inventory(root, spec)
    generic = verify_synthetic_confirmatory_artifact(root, write_report=False)
    primary = _strict_object(root / "primary_analysis.json")
    independent = _strict_object(root / "independent_verification.json")
    decision = _strict_object(root / "final_decision.json")
    run = _strict_object(root / "run_manifest.json")
    comparison = compare_primary_and_independent(primary, independent)
    try:
        authenticated_decision = _validate_v3_final_decision(decision)
    except (TypeError, ValueError):
        authenticated_decision = None
    live_raw_sha256 = _file_sha256(spec.paths.raw_manifest)
    live_snapshot_lock_sha256 = _file_sha256(spec.paths.snapshot_lock)
    report_path = root / "synthetic_confirmatory_report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    marker_pass = bool(
        primary.get("execution_class") == EXECUTION_CLASS
        and primary.get("execution_classification") == EXECUTION_CLASS
        and independent.get("execution_class") == EXECUTION_CLASS
        and independent.get("execution_classification") == EXECUTION_CLASS
        and run.get("execution_class") == EXECUTION_CLASS
        and run.get("execution_classification") == EXECUTION_CLASS
        and EXECUTION_CLASS in report
        and report.startswith(
            "# Synthetic Confirmatory v3 — Requalified Local Execution\n"
        )
    )
    schema_pass = bool(
        primary.get("schema_version") == PRIMARY_ANALYSIS_SCHEMA
        and independent.get("schema_version") == INDEPENDENT_SCHEMA
        and run.get("schema_version") == REQUALIFIED_RUN_SCHEMA
    )
    decision_pass = bool(
        authenticated_decision is not None
        and primary.get("final_decision") == authenticated_decision
        and independent.get("final_decision") == authenticated_decision
        and independent.get("verification_projection", {}).get("final_decision")
        == authenticated_decision
    )
    binding_pass = bool(
        run.get("run_id") == spec.run_id
        and run.get("raw_result_manifest_sha256") == live_raw_sha256
        and run.get("snapshot_lock_sha256") == live_snapshot_lock_sha256
        and all(
            value.get("run_id") == spec.run_id
            and value.get("raw_result_manifest_sha256") == live_raw_sha256
            for value in (primary, independent)
        )
    )
    completeness_pass = bool(
        run.get("planned_snapshot_count") == _EXPECTED_SNAPSHOT_COUNT
        and run.get("planned_trial_count") == _EXPECTED_TRIAL_COUNT
        and run.get("completed_snapshot_count") == _EXPECTED_SNAPSHOT_COUNT
        and run.get("completed_trial_count") == _EXPECTED_TRIAL_COUNT
        and run.get("backend_trial_counts") == _EXPECTED_BACKEND_COUNTS
        and run.get("native_trial_count") == 0
        and run.get("native_execution_count") == 0
        and run.get("complete") is True
    )
    exact_match_pass = bool(
        comparison.get("exact_match_pass") is True
        and comparison.get("section_difference_count") == 0
        and comparison.get("leaf_difference_count") == 0
        and comparison.get("maximum_absolute_numeric_difference") == 0.0
    )
    pre_persist_pass = bool(
        generic.get("ARTIFACT_VERIFICATION_PASS") is True
        and inventory["publisher_7_table_3_figure_7_root_pass"] is True
        and marker_pass
        and schema_pass
        and decision_pass
        and binding_pass
        and completeness_pass
        and exact_match_pass
    )
    base = {
        **generic,
        **inventory,
        "schema_version": REQUALIFIED_ARTIFACT_VERIFICATION_SCHEMA,
        "execution_class": EXECUTION_CLASS,
        "execution_classification": EXECUTION_CLASS,
        "analysis_verifier_comparison": comparison,
        "analysis_verifier_exact_match_pass": exact_match_pass,
        "requalified_execution_marker_pass": marker_pass,
        "requalified_schema_pass": schema_pass,
        "v3_final_decision_identity_pass": decision_pass,
        "live_raw_and_snapshot_binding_pass": binding_pass,
        "formal_matrix_completeness_pass": completeness_pass,
    }
    expected_record = {
        **base,
        "persisted_artifact_verification_match_pass": True,
        "REQUALIFIED_V3_ARTIFACT_VERIFICATION_PASS": pre_persist_pass,
    }
    verification_path = root / "artifact_verification.json"
    if write_report:
        if verification_path.exists():
            atomic_replace_canonical_json(verification_path, expected_record)
        else:
            atomic_create_canonical_json(verification_path, expected_record)
        return expected_record
    try:
        recorded = _strict_object(verification_path)
    except (OSError, ValueError, RequalifiedPostrunError):
        recorded = None
    persisted_match = recorded == expected_record
    return {
        **base,
        "persisted_artifact_verification_match_pass": persisted_match,
        "REQUALIFIED_V3_ARTIFACT_VERIFICATION_PASS": bool(
            pre_persist_pass and persisted_match
        ),
    }


# Explicit aliases used by the requalified component-bundle builder.  The
# underlying signatures remain the generic lifecycle signatures above.
requalified_primary_analyzer = primary_analyzer
requalified_independent_verifier = independent_verifier
requalified_difference_auditor = difference_auditor
requalified_publisher = publisher
requalified_artifact_verifier = artifact_verifier


__all__ = [
    "EXECUTION_CLASS",
    "REQUALIFIED_ARTIFACT_VERIFICATION_SCHEMA",
    "REQUALIFIED_DIFFERENCE_SCHEMA",
    "REQUALIFIED_RUN_SCHEMA",
    "RequalifiedPostrunError",
    "artifact_verifier",
    "difference_auditor",
    "independent_verifier",
    "primary_analyzer",
    "publisher",
    "requalified_artifact_verifier",
    "requalified_difference_auditor",
    "requalified_independent_verifier",
    "requalified_primary_analyzer",
    "requalified_publisher",
]
