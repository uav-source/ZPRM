"""Atomic publishers for formal and seed-free-fixture Confirmatory v2 evidence."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .contracts import file_sha256, write_json
from .synthetic_confirmatory_v2_analysis import load_v2_raw
from .synthetic_confirmatory_v2_artifact_verifier import (
    FIXTURE_FIGURES,
    FIXTURE_INDEPENDENT_SCHEMA,
    FIXTURE_RUN_SCHEMA,
    FIXTURE_TABLES,
    FORMAL_FIGURES,
    FORMAL_TABLES,
    audit_synthetic_confirmatory_v2_fixture_rows,
    verify_synthetic_confirmatory_v2_artifact,
    verify_synthetic_confirmatory_v2_fixture_artifact,
)
from .synthetic_confirmatory_v2_independent_verifier import (
    compare_v2_fixture_primary_and_independent,
    compare_v2_primary_and_independent,
)
from .synthetic_confirmatory_v2_contract import (
    BACKENDS,
    FORMAL_ANALYSIS_SCHEMA,
    FORMAL_RUN_ID,
    FORMAL_RUN_SCHEMA,
    INDEPENDENT_SCHEMA,
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


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    fields = sorted({key for row in values for key in row})
    if not fields:
        raise ValueError(f"refusing empty publication table: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    key: json.dumps(
                        value,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _write_sha256sums(destination: Path) -> None:
    paths = sorted(
        (
            path
            for path in destination.rglob("*")
            if path.is_file()
            and path.name not in {"SHA256SUMS", "artifact_verification.json"}
        ),
        key=lambda path: path.relative_to(destination).as_posix(),
    )
    (destination / "SHA256SUMS").write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(destination).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )


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


def _formal_figures(destination: Path, primary: Mapping[str, Any]) -> None:
    figures = destination / "figures"
    gates = list(primary["gate_summary"].items())
    figure, axis = plt.subplots(figsize=(9.0, 4.5))
    positions = np.arange(len(gates))
    axis.bar(
        positions,
        [1.0 if value is True else 0.0 for _, value in gates],
        color=["#2d7f5e" if value is True else "#a33a3a" for _, value in gates],
    )
    axis.set_xticks(positions, labels=[name for name, _ in gates])
    axis.tick_params(axis="x", labelrotation=35)
    axis.set_ylim(0.0, 1.1)
    axis.set_ylabel("pass (1) / fail (0)")
    axis.set_title("Frozen v2 gate matrix")
    figure.tight_layout()
    figure.savefig(figures / FORMAL_FIGURES[0], dpi=160)
    plt.close(figure)

    rows = list(primary["h3_cross_backend_ranking"])
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for axis, row in zip(axes, rows):
        for backend, values in row["scene_medians"].items():
            axis.plot(np.arange(len(values)), values, marker="o", label=backend)
        axis.set_title(str(row["condition"]))
        axis.set_xlabel("frozen scene order")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("translation median (m)")
    axes[-1].legend(fontsize=8)
    figure.suptitle("Cross-backend scene effects")
    figure.tight_layout()
    figure.savefig(figures / FORMAL_FIGURES[1], dpi=160)
    plt.close(figure)

    models = list(primary["h5_frozen_models"])
    locations = np.arange(len(models))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.bar(
        locations - width / 2,
        [float(row["model_a_mae"]) for row in models],
        width,
        label="Frozen Model A",
    )
    axis.bar(
        locations + width / 2,
        [float(row["model_b_mae"]) for row in models],
        width,
        label="Frozen Model B",
    )
    axis.set_xticks(
        locations, labels=[row["backend_schema_name"] for row in models]
    )
    axis.set_ylabel("MAE on log10(error + 1e-9)")
    axis.set_title("Frozen-model confirmatory comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / FORMAL_FIGURES[2], dpi=160)
    plt.close(figure)


def _formal_report(
    primary: Mapping[str, Any], comparison: Mapping[str, Any]
) -> str:
    decision = primary["final_decision"]
    lines = [
        "# Synthetic Confirmatory v2",
        "",
        "This artifact reports the preregistered frozen H1--H6 analysis over the complete v2 evidence.",
        "",
        "## Decision",
        "",
        f"- `SYNTHETIC_CONFIRMATORY_V2_EXECUTED = {str(decision['SYNTHETIC_CONFIRMATORY_V2_EXECUTED']).lower()}`",
        f"- `SYNTHETIC_CONFIRMATORY_V2_COMPLETE = {str(decision['SYNTHETIC_CONFIRMATORY_V2_COMPLETE']).lower()}`",
        f"- `SYNTHETIC_CONFIRMATORY_V2_PASS = {str(decision['SYNTHETIC_CONFIRMATORY_V2_PASS']).lower()}`",
        f"- Analysis/verifier differing leaves: `{comparison['leaf_difference_count']}`",
        "- `REAL_DATA_RUN_AUTHORIZED = false`",
        "- `MEASUREMENT_PAPER_MAINLINE_AUTHORIZED = false`",
        "",
        "## Evidence inventory",
        "",
    ]
    lines.extend(f"- `tables/{name}`" for name in FORMAL_TABLES)
    lines.extend(f"- `figures/{name}`" for name in FORMAL_FIGURES)
    lines.extend(
        [
            "",
            "IDEAL lineage is backed by persisted parent-index arrays and Phase A quantization closure.",
            "Association results do not authorize causal claims.",
            "Frozen model parameters were applied without refitting or restandardization.",
            "",
        ]
    )
    return "\n".join(lines)


def _formal_preflight(
    *,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    run: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = compare_v2_primary_and_independent(primary, independent)
    if comparison.get("exact_match_pass") is not True:
        raise ValueError("v2 primary and independent formal analysis differ")
    decision = primary.get("final_decision")
    independent_lineage = independent.get("lineage_integrity")
    if independent_lineage is None and type(independent.get("verification_projection")) is dict:
        independent_lineage = independent["verification_projection"].get(
            "lineage_integrity"
        )
    lineage_fields = (
        "IDEAL_PARENT_LINEAGE_COUNT",
        "LINEAGE_INTEGRITY_PASS",
        "LINEAGE_VIOLATION_COUNT",
        "NONIDEAL_NO_LINEAGE_COUNT",
    )
    primary_lineage = primary.get("lineage_integrity")
    lineage_semantics_match = bool(
        type(primary_lineage) is dict
        and type(independent_lineage) is dict
        and primary_lineage.get("schema_version")
        == "synthetic_confirmatory_v2_primary_lineage_inventory_v1"
        and independent_lineage.get("schema_version")
        in (None, "synthetic_confirmatory_v2_independent_lineage_inventory_v1")
        and all(
            primary_lineage.get(name) == independent_lineage.get(name)
            for name in lineage_fields
        )
    )
    if (
        primary.get("schema_version") != FORMAL_ANALYSIS_SCHEMA
        or independent.get("schema_version") != INDEPENDENT_SCHEMA
        or run.get("schema_version") != FORMAL_RUN_SCHEMA
        or type(decision) is not dict
        or independent.get("final_decision") != decision
        or decision.get("SYNTHETIC_CONFIRMATORY_V2_EXECUTED") is not True
        or decision.get("SYNTHETIC_CONFIRMATORY_V2_COMPLETE") is not True
        or type(decision.get("SYNTHETIC_CONFIRMATORY_V2_PASS")) is not bool
        or not lineage_semantics_match
        or primary.get("lineage_integrity", {}).get("LINEAGE_INTEGRITY_PASS")
        is not True
        or run.get("run_id") != FORMAL_RUN_ID
        or not isinstance(run.get("raw_result_manifest_sha256"), str)
        or any(
            report.get("run_id") != FORMAL_RUN_ID
            or report.get("raw_result_manifest_sha256")
            != run.get("raw_result_manifest_sha256")
            for report in (primary, independent)
        )
    ):
        raise ValueError("v2 formal publication input contract failed")
    return comparison


def _publish_formal_into(
    destination: Path,
    *,
    manifest_path: str | Path,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    run: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = _formal_preflight(
        primary=primary, independent=independent, run=run
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "tables").mkdir()
    (destination / "figures").mkdir()
    for name, rows in _formal_table_rows(primary).items():
        _write_csv(destination / "tables" / name, rows)
    _formal_figures(destination, primary)
    write_json(destination / "primary_analysis.json", dict(primary))
    write_json(destination / "independent_verification.json", dict(independent))
    write_json(destination / "final_decision.json", dict(primary["final_decision"]))
    write_json(destination / "run_manifest.json", dict(run))
    (destination / "synthetic_confirmatory_report.md").write_text(
        _formal_report(primary, comparison), encoding="utf-8"
    )
    _write_sha256sums(destination)
    written = verify_synthetic_confirmatory_v2_artifact(
        destination, manifest_path=manifest_path, write_report=True
    )
    read_only = verify_synthetic_confirmatory_v2_artifact(
        destination, manifest_path=manifest_path, write_report=False
    )
    if written != read_only or written.get("ARTIFACT_VERIFICATION_PASS") is not True:
        raise ValueError("v2 formal artifact verification failed")
    return written


def publish_synthetic_confirmatory_v2(
    *,
    manifest_path: str | Path,
    run_dir: str | Path,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Authenticate, stage, verify twice, and atomically publish formal v2."""

    _trials, raw, _repository, manifest = load_v2_raw(
        manifest_path=manifest_path, run_dir=run_dir
    )
    run_path = Path(run_dir).resolve() / "run_manifest.json"
    run = _strict_json(run_path)
    raw_sha = file_sha256(Path(run_dir).resolve() / "raw_result_manifest.json")
    if (
        raw.get("run_id") != manifest.get("formal_run_id")
        or run.get("run_id") != manifest.get("formal_run_id")
        or run.get("raw_result_manifest_sha256") != raw_sha
    ):
        raise ValueError("v2 run/raw manifest binding mismatch")
    _formal_preflight(primary=primary, independent=independent, run=run)

    destination = Path(artifact_dir).resolve()
    if destination.exists():
        raise FileExistsError("refusing to replace v2 formal artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    ).resolve()
    staging = temporary / "artifact"
    try:
        verification = _publish_formal_into(
            staging,
            manifest_path=manifest_path,
            primary=primary,
            independent=independent,
            run=run,
        )
        os.replace(staging, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": verification[
            "analysis_verifier_comparison"
        ]["leaf_difference_count"],
        "ARTIFACT_VERIFICATION_PASS": verification["ARTIFACT_VERIFICATION_PASS"],
        "artifact_path": str(destination),
        "published_file_count": sum(path.is_file() for path in destination.rglob("*")),
        "sha256_mismatch_count": len(verification["sha256_mismatch_files"]),
    }


def _fixture_table_rows(
    primary: Mapping[str, Any], run: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    trials = [dict(row) for row in primary["results"]]
    by_snapshot: dict[str, list[dict[str, Any]]] = {}
    for row in trials:
        by_snapshot.setdefault(str(row["snapshot_id"]), []).append(row)
    snapshots = []
    pairing = []
    for snapshot_id, rows in sorted(by_snapshot.items()):
        checksum_names = (
            "snapshot_checksum",
            "source_checksum",
            "target_checksum",
            "reference_pose_checksum",
        )
        checksum_match = all(
            len({str(row.get(name)) for row in rows}) == 1
            for name in checksum_names
        )
        backends = sorted({str(row["backend"]) for row in rows})
        snapshots.append(
            {
                "backend_count": len(backends),
                "condition": rows[0]["condition"],
                "input_checksum_match": checksum_match,
                "snapshot_id": snapshot_id,
                "trial_count": len(rows),
            }
        )
        pairing.append(
            {
                "backends": backends,
                "pairing_pass": len(rows) == 2
                and set(backends) == set(BACKENDS)
                and checksum_match,
                "snapshot_id": snapshot_id,
                "trial_count": len(rows),
            }
        )
    backend_summary = []
    for backend in BACKENDS:
        rows = [row for row in trials if row["backend"] == backend]
        backend_summary.append(
            {
                "backend": backend,
                "nonfinite_output_count": sum(
                    row.get("finite_output") is not True for row in rows
                ),
                "solver_failure_count": sum(
                    bool(row.get("solver_failure")) for row in rows
                ),
                "trial_count": len(rows),
            }
        )
    return {
        FIXTURE_TABLES[0]: trials,
        FIXTURE_TABLES[1]: snapshots,
        FIXTURE_TABLES[2]: backend_summary,
        FIXTURE_TABLES[3]: [dict(row) for row in primary["failure_inventory"]],
        FIXTURE_TABLES[4]: pairing,
        FIXTURE_TABLES[5]: [
            {
                "fresh_resume_scientific_equivalence": run[
                    "fresh_resume_scientific_equivalence"
                ],
                "resume_backend_execution_count": run[
                    "resume_backend_execution_count"
                ],
            }
        ],
        FIXTURE_TABLES[6]: [
            {"gate": key, "value": value}
            for key, value in primary["decision"].items()
        ],
    }


def _fixture_figures(destination: Path, primary: Mapping[str, Any]) -> None:
    rows = [dict(row) for row in primary["results"]]
    conditions = (
        "FIXTURE_IDENTITY",
        "FIXTURE_NONIDENTITY_REFERENCE",
        "FIXTURE_NO_CORRESPONDENCE",
    )
    figures = destination / "figures"
    matrix = np.zeros((len(BACKENDS), len(conditions)), dtype=float)
    failure_matrix = np.zeros_like(matrix)
    update_matrix = np.zeros_like(matrix)
    for backend_index, backend in enumerate(BACKENDS):
        for condition_index, condition in enumerate(conditions):
            selected = [
                row
                for row in rows
                if row.get("backend") == backend and row.get("condition") == condition
            ]
            if selected:
                row = selected[0]
                matrix[backend_index, condition_index] = 1.0
                failure_matrix[backend_index, condition_index] = float(
                    bool(row.get("solver_failure"))
                )
                metric = row.get("translation_update_m")
                update_matrix[backend_index, condition_index] = (
                    float(metric) if isinstance(metric, (int, float)) else 0.0
                )
    for filename, data, title, label in (
        (FIXTURE_FIGURES[0], matrix, "Seed-free fixture execution", "present"),
        (FIXTURE_FIGURES[1], failure_matrix, "Frozen failure classification", "solver failure"),
        (FIXTURE_FIGURES[2], update_matrix, "Fixture translation updates", "metres"),
    ):
        figure, axis = plt.subplots(figsize=(8.8, 3.8))
        image = axis.imshow(data, aspect="auto", cmap="viridis")
        axis.set_xticks(np.arange(len(conditions)), labels=conditions, rotation=25)
        axis.set_yticks(np.arange(len(BACKENDS)), labels=BACKENDS)
        axis.set_title(title)
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label(label)
        figure.tight_layout()
        figure.savefig(figures / filename, dpi=160)
        plt.close(figure)


def _fixture_report() -> str:
    lines = [
        "# Synthetic Confirmatory v2 Seed-Free Fixture",
        "",
        "This artifact is an execution-chain qualification over three deterministic fixtures and six backend trials.",
        "",
        "- `FORMAL_CONFIRMATORY_SCIENCE_EVALUATED = false`",
        "- No formal v2 seed, snapshot, trial, or scientific gate was evaluated.",
        "",
        "## Evidence inventory",
        "",
    ]
    lines.extend(f"- `tables/{name}`" for name in FIXTURE_TABLES)
    lines.extend(f"- `figures/{name}`" for name in FIXTURE_FIGURES)
    lines.append("")
    return "\n".join(lines)


def _fixture_preflight(
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    run: Mapping[str, Any],
) -> None:
    comparison = compare_v2_fixture_primary_and_independent(
        primary, independent
    )
    rows = primary.get("results")
    row_audit = audit_synthetic_confirmatory_v2_fixture_rows(rows)
    if (
        primary.get("schema_version")
        != "synthetic_confirmatory_v2_fixture_primary_analysis_v1"
        or independent.get("schema_version") != FIXTURE_INDEPENDENT_SCHEMA
        or comparison.get("exact_match_pass") is not True
        or independent.get("decision") != primary.get("decision")
        or primary.get("decision", {}).get("FIXTURE_EXECUTION_CHAIN_PASS") is not True
        or primary.get("decision", {}).get("FORMAL_CONFIRMATORY_SCIENCE_EVALUATED")
        is not False
        or type(rows) is not list
        or row_audit.get("fixture_row_contract_pass") is not True
        or len(rows) != 6
        or primary.get("fixture_snapshot_count") != 3
        or primary.get("fixture_trial_count") != 6
        or run.get("schema_version") != FIXTURE_RUN_SCHEMA
        or run.get("fixture_snapshot_count") != 3
        or run.get("fixture_trial_count") != 6
        or run.get("backend_execution_count") != 6
        or run.get("resume_backend_execution_count") != 0
        or run.get("fresh_resume_scientific_equivalence") is not True
        or run.get("formal_v2_seed_reference_count") != 0
        or run.get("formal_confirmatory_science_evaluated") is not False
    ):
        raise ValueError("seed-free v2 fixture publication input contract failed")


def _publish_fixture_into(
    destination: Path,
    *,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    run: Mapping[str, Any],
) -> dict[str, Any]:
    _fixture_preflight(primary, independent, run)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "tables").mkdir()
    (destination / "figures").mkdir()
    for name, rows in _fixture_table_rows(primary, run).items():
        _write_csv(destination / "tables" / name, rows)
    _fixture_figures(destination, primary)
    write_json(destination / "primary_analysis.json", dict(primary))
    write_json(destination / "independent_verification.json", dict(independent))
    write_json(destination / "final_decision.json", dict(primary["decision"]))
    write_json(destination / "run_manifest.json", dict(run))
    (destination / "synthetic_confirmatory_report.md").write_text(
        _fixture_report(), encoding="utf-8"
    )
    _write_sha256sums(destination)
    written = verify_synthetic_confirmatory_v2_fixture_artifact(
        destination, write_report=True
    )
    read_only = verify_synthetic_confirmatory_v2_fixture_artifact(
        destination, write_report=False
    )
    if (
        written != read_only
        or written.get("FIXTURE_ARTIFACT_VERIFICATION_PASS") is not True
    ):
        raise ValueError("v2 fixture artifact verification failed")
    return written


def publish_synthetic_confirmatory_v2_fixture(
    *,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Publish only the seed-free 3/6 execution-chain fixture evidence."""

    _fixture_preflight(primary, independent, run_manifest)
    destination = Path(artifact_dir).resolve()
    if destination.exists():
        raise FileExistsError("refusing to replace v2 fixture artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    ).resolve()
    staging = temporary / "artifact"
    try:
        verification = _publish_fixture_into(
            staging,
            primary=primary,
            independent=independent,
            run=run_manifest,
        )
        os.replace(staging, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "FIXTURE_ARTIFACT_VERIFICATION_PASS": verification[
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        ],
        "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED": False,
        "artifact_path": str(destination),
        "published_file_count": sum(path.is_file() for path in destination.rglob("*")),
        "sha256_mismatch_count": len(verification["sha256_mismatch_files"]),
    }


__all__ = [
    "publish_synthetic_confirmatory_v2",
    "publish_synthetic_confirmatory_v2_fixture",
]
