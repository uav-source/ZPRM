"""Publish the compact 20-table, 11-figure Full Synthetic artifact."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .contracts import file_sha256, manifest_root, write_json
from .full_synthetic_analysis import (
    BACKENDS,
    CONDITIONS,
    NONIDEAL_CONDITIONS,
    SCENES,
)
from .full_synthetic_artifact_verifier import (
    REQUIRED_DEVELOPMENT_GATES,
    TABLES,
    verify_full_synthetic_artifact,
)
from .full_synthetic_independent_verifier import (
    full_synthetic_analysis_verifier_comparison,
    full_synthetic_analysis_verifier_difference_count,
)


MUTED_BLUE = "#4C78A8"
MUTED_AMBER = "#D29B3D"
MUTED_GREEN = "#6E9F78"
MUTED_RED = "#B85C5C"
CONDITION_COLORS = (
    "#4C78A8", "#D29B3D", "#6E9F78", "#B85C5C", "#8A74A6", "#6F8F9D"
)


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    empty_fields: Sequence[str] = (),
) -> None:
    values = list(rows)
    fields = sorted({key for row in values for key in row}) or list(empty_fields)
    if not fields:
        raise ValueError(f"CSV has no columns: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _scene_condition_figure(
    path: Path, rows: Sequence[Mapping[str, Any]], metric: str, ylabel: str, title: str
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, backend in zip(axes, ("Open3D", "PCL")):
        for condition_index, condition in enumerate(CONDITIONS):
            lookup = {
                row["scene_variant"]: row[metric]
                for row in rows
                if row["backend"] == backend and row["condition"] == condition
            }
            axis.plot(
                np.arange(len(SCENES)),
                [
                    np.nan if lookup.get(scene) is None else lookup[scene]
                    for scene in SCENES
                ],
                color=CONDITION_COLORS[condition_index], marker="o",
                linewidth=1.2, label=condition,
            )
        axis.set_ylabel(ylabel)
        axis.set_title(backend)
        axis.grid(alpha=.25)
    axes[0].legend(ncol=3, fontsize=7)
    axes[-1].set_xticks(
        np.arange(len(SCENES)),
        [scene.replace("END_FACE_TRANSITION_", "EFT_") for scene in SCENES],
        rotation=25, ha="right",
    )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _paired_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = [f"{row['condition'][:4]}\n{row['backend']}" for row in rows]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - .18, [np.nan if row["rich_median_m"] is None else row["rich_median_m"] for row in rows], .36, color=MUTED_BLUE, label="Rich room")
    axis.bar(x + .18, [np.nan if row["corridor_median_m"] is None else row["corridor_median_m"] for row in rows], .36, color=MUTED_AMBER, label="Long corridor")
    axis.set_xticks(x, labels)
    axis.set_ylabel("translation median (m)")
    axis.set_title("Frozen corridor/rich paired comparison")
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _cross_backend_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    selected = [row for row in rows if row["scope"] == "CONDITION"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(np.arange(len(selected)), [np.nan if row["spearman_rho"] is None else row["spearman_rho"] for row in selected], color="#457b9d")
    axis.axhline(.7, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(selected)), [row["condition"].replace("_NOISE", "") for row in selected], rotation=25, ha="right")
    axis.set_ylim(-1, 1)
    axis.set_ylabel("Spearman rho")
    axis.set_title("Cross-backend scene ranking")
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _systematic_scatter(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7, 5.5))
    for backend, marker, color in (("Open3D", "o", MUTED_BLUE), ("PCL", "s", MUTED_AMBER)):
        chosen = [
            row for row in rows
            if row["backend"] == backend
            and row["systematic_translation_offset_m"] is not None
            and row["translation_repeatability_rms_m"] is not None
        ]
        axis.scatter(
            [row["translation_repeatability_rms_m"] for row in chosen],
            [row["systematic_translation_offset_m"] for row in chosen],
            color=color, s=18, alpha=.55, marker=marker, label=backend,
        )
    axis.set_xlabel("translation repeatability RMS (m)")
    axis.set_ylabel("systematic translation offset (m)")
    axis.set_title("Systematic offset vs repeatability")
    axis.grid(alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _direction_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    groups = [
        [
            row["translation_direction_concentration"]
            for row in rows
            if row["backend"] == backend
            and row["translation_direction_concentration"] is not None
        ]
        for backend in ("Open3D", "PCL")
    ]
    figure, axis = plt.subplots(figsize=(6, 5))
    boxes = axis.boxplot(
        groups, labels=["Open3D", "PCL"], showfliers=True, patch_artist=True
    )
    for box, color in zip(boxes["boxes"], (MUTED_BLUE, MUTED_AMBER)):
        box.set_facecolor(color)
        box.set_alpha(.65)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("direction concentration")
    axis.set_title("Translation direction concentration")
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _condition_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITIONS))
    width = .36
    for offset, backend, color in ((-.18, "Open3D", MUTED_BLUE), (.18, "PCL", MUTED_AMBER)):
        values = [
            (np.median([
                row["translation_median_m"] for row in rows
                if row["backend"] == backend and row["condition"] == condition
                and row["translation_median_m"] is not None
            ]) if any(
                row["translation_median_m"] is not None
                for row in rows if row["backend"] == backend and row["condition"] == condition
            ) else np.nan)
            for condition in CONDITIONS
        ]
        axis.bar(x + offset, values, width, color=color, label=backend)
    axis.set_xticks(x, [condition.replace("_NOISE", "") for condition in CONDITIONS], rotation=25, ha="right")
    axis.set_ylabel("median of scene medians (m)")
    axis.set_title("Condition effects (Development descriptive)")
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _common_scatter(
    path: Path, rows: Sequence[Mapping[str, Any]], x_field: str, title: str, xlabel: str
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5.5))
    for backend, marker, color in (("Open3D", "o", MUTED_BLUE), ("PCL", "s", MUTED_AMBER)):
        chosen = [
            row for row in rows
            if row.get("backend") == backend
            and row.get(x_field) is not None
            and row.get("translation_error_m") is not None
        ]
        axis.scatter(
            [row[x_field] for row in chosen],
            [row["translation_error_m"] for row in chosen],
            color=color, s=12, alpha=.4, marker=marker, label=backend,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("translation error (m)")
    axis.set_title(title)
    axis.grid(alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _ridge_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(6.5, 5))
    axis.bar(x - .18, [np.nan if row["model_a_cv_mae"] is None else row["model_a_cv_mae"] for row in rows], .36, color=MUTED_BLUE, label="Model A")
    axis.bar(x + .18, [np.nan if row["model_b_cv_mae"] is None else row["model_b_cv_mae"] for row in rows], .36, color=MUTED_AMBER, label="Model B")
    axis.set_xticks(x, [row["backend"] for row in rows])
    axis.set_ylabel("CV MAE, log10 translation error")
    axis.set_title("Leave-one-geometry-seed-out Ridge")
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _runtime_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    x = np.arange(len(rows))
    axis.bar(x - .18, [row["median_runtime_ms"] for row in rows], .36, color=MUTED_BLUE, label="median")
    axis.bar(x + .18, [row["q95_runtime_ms"] for row in rows], .36, color=MUTED_AMBER, label="q95")
    axis.set_xticks(x, [row["backend"] for row in rows])
    axis.set_ylabel("runtime (ms)")
    axis.set_title("Backend runtime")
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def full_synthetic_publication_decision(
    primary: Mapping[str, Any], independent: Mapping[str, Any]
) -> tuple[dict[str, Any], int]:
    """Preserve both source decisions and derive one fail-closed publication decision."""

    difference = full_synthetic_analysis_verifier_difference_count(
        primary, independent
    )
    source = primary.get("final_decision")
    if type(source) is not dict:
        raise ValueError("primary analysis final decision is missing")
    decision = dict(source)
    agreement = difference == 0
    decision["ANALYSIS_VERIFIER_AGREEMENT_PASS"] = agreement
    decision["ANALYSIS_VERIFIER_DIFFERENCE_COUNT"] = difference
    decision["FULL_SYNTHETIC_DEVELOPMENT_ENGINEERING_PASS"] = bool(
        decision.get("FULL_SYNTHETIC_DEVELOPMENT_ENGINEERING_PASS") is True
        and agreement
    )
    development_pass = all(
        decision.get(name) is True for name in REQUIRED_DEVELOPMENT_GATES
    )
    decision["FULL_SYNTHETIC_DEVELOPMENT_PASS"] = development_pass
    decision["CONFIRMATORY_PROTOCOL_DESIGN_AUTHORIZED"] = development_pass
    decision["REAL_DATA_PROTOCOL_DESIGN_AUTHORIZED"] = development_pass
    if not agreement:
        decision[
            "PHENOMENON_CONFIRMED_BUT_INCREMENTAL_VALUE_NOT_ESTABLISHED"
        ] = False
    return decision, difference


def _chart_map() -> list[dict[str, str]]:
    return [
        {
            "figure": "scene_condition_translation_error.png",
            "question": "How does translation error vary by scene, condition, and backend?",
            "fields": "scene_variant, condition, backend, translation_median_m",
            "supported_claim": "descriptive scene-condition translation separation",
            "palette": "fixed muted categorical",
            "output_path": "figures/scene_condition_translation_error.png",
        },
        {
            "figure": "scene_condition_rotation_error.png",
            "question": "How does rotation error vary by scene, condition, and backend?",
            "fields": "scene_variant, condition, backend, rotation_median_rad",
            "supported_claim": "descriptive scene-condition rotation separation",
            "palette": "fixed muted categorical",
            "output_path": "figures/scene_condition_rotation_error.png",
        },
        {
            "figure": "corridor_rich_paired.png",
            "question": "Does the corridor exceed the rich room under the four primary combinations?",
            "fields": "condition, backend, rich_median_m, corridor_median_m",
            "supported_claim": "primary paired scene-effect gate",
            "palette": "blue and amber",
            "output_path": "figures/corridor_rich_paired.png",
        },
        {
            "figure": "cross_backend_scene_ranking.png",
            "question": "Do the two backends rank scene difficulty similarly?",
            "fields": "condition, spearman_rho",
            "supported_claim": "cross-backend rank agreement",
            "palette": "single blue",
            "output_path": "figures/cross_backend_scene_ranking.png",
        },
        {
            "figure": "systematic_offset_vs_repeatability.png",
            "question": "How large are systematic offsets relative to repeatability?",
            "fields": "translation_repeatability_rms_m, systematic_translation_offset_m",
            "supported_claim": "systematic-offset wording gate context",
            "palette": "blue and amber",
            "output_path": "figures/systematic_offset_vs_repeatability.png",
        },
        {
            "figure": "direction_concentration.png",
            "question": "Are translation-error directions concentrated within groups?",
            "fields": "backend, translation_direction_concentration",
            "supported_claim": "directional concentration description",
            "palette": "blue and amber",
            "output_path": "figures/direction_concentration.png",
        },
        {
            "figure": "condition_effects.png",
            "question": "How do the six synthetic conditions differ descriptively?",
            "fields": "condition, backend, translation_median_m",
            "supported_claim": "condition contrast context",
            "palette": "blue and amber",
            "output_path": "figures/condition_effects.png",
        },
        {
            "figure": "turnover_vs_translation_error.png",
            "question": "Is common correspondence turnover associated with error?",
            "fields": "correspondence_turnover, translation_error_m",
            "supported_claim": "reassociation mechanism correlation",
            "palette": "blue and amber",
            "output_path": "figures/turnover_vs_translation_error.png",
        },
        {
            "figure": "local_metric_vs_error.png",
            "question": "How does the local Hessian metric relate to error?",
            "fields": "lambda_min_trans, translation_error_m",
            "supported_claim": "local geometry baseline context",
            "palette": "blue and amber",
            "output_path": "figures/local_metric_vs_error.png",
        },
        {
            "figure": "ridge_model_comparison.png",
            "question": "Does Model B reduce LOSO prediction error relative to Model A?",
            "fields": "model_a_cv_mae, model_b_cv_mae",
            "supported_claim": "incremental-value model condition",
            "palette": "blue and amber",
            "output_path": "figures/ridge_model_comparison.png",
        },
        {
            "figure": "runtime_comparison.png",
            "question": "What are the backend runtime distributions?",
            "fields": "backend, median_runtime_ms, q95_runtime_ms",
            "supported_claim": "engineering runtime description",
            "palette": "blue and amber",
            "output_path": "figures/runtime_comparison.png",
        },
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _technical_report(
    *,
    primary: Mapping[str, Any],
    decision: Mapping[str, Any],
    comparison: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> str:
    """Build an answer-first report that points every claim to its audit surface."""

    lines = [
        "# Zero-Perturbation Full Synthetic Development",
        "",
        "## Answer-first decision",
        "",
        (
            "This is exploratory Development evidence over **1,260 snapshots and "
            "2,520 paired backend trials**. The frozen eight-gate conjunction "
            f"evaluated to **{_fmt(decision['FULL_SYNTHETIC_DEVELOPMENT_PASS'])}**."
        ),
        "",
        f"- `FULL_SYNTHETIC_DEVELOPMENT_PASS = {_fmt(decision['FULL_SYNTHETIC_DEVELOPMENT_PASS'])}`",
        f"- `SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED = {_fmt(decision['SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED'])}`",
        f"- `CONFIRMATORY_PROTOCOL_DESIGN_AUTHORIZED = {_fmt(decision['CONFIRMATORY_PROTOCOL_DESIGN_AUTHORIZED'])}`",
        f"- `REAL_DATA_PROTOCOL_DESIGN_AUTHORIZED = {_fmt(decision['REAL_DATA_PROTOCOL_DESIGN_AUTHORIZED'])}`",
        "- `CONFIRMATORY_RUN_AUTHORIZED = false`",
        "- `REAL_DATA_RUN_AUTHORIZED = false`",
        "- `MEASUREMENT_PAPER_MAINLINE_AUTHORIZED = false`",
        f"- `ANALYSIS_VERIFIER_DIFFERENCE_COUNT = {comparison['section_difference_count']}`",
        "",
        "## Frozen Development gates",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for name in REQUIRED_DEVELOPMENT_GATES:
        lines.append(f"| `{name}` | `{_fmt(decision[name])}` |")
    lines.extend(
        [
            "",
            "The Development decision is exactly the conjunction of the eight rows above; "
            "the systematic-offset wording gate is reported separately and is not a hidden ninth gate.",
            "",
            "## Four primary corridor-versus-rich combinations",
            "",
            "| Condition | Backend | Rich median (m) | Corridor median (m) | Ratio | Difference (m) | Wins / 30 | Gate |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary["scene_effect_paired"]:
        lines.append(
            "| {condition} | {backend} | {rich} | {corridor} | {ratio} | {difference} | {wins} / 30 | {gate} |".format(
                condition=row["condition"], backend=row["backend"],
                rich=_fmt(row["rich_median_m"]), corridor=_fmt(row["corridor_median_m"]),
                ratio=_fmt(row["weak_rich_median_ratio"]),
                difference=_fmt(row["absolute_median_difference_m"]),
                wins=row["paired_win_count"], gate=_fmt(row["gate_pass"]),
            )
        )
    lines.extend(
        [
            "",
            "All 84 scene × condition × backend summaries, including success, median, IQR, "
            "q95 and three-seed hierarchical intervals, are in "
            "[`scene_condition_backend_summary.csv`](tables/scene_condition_backend_summary.csv).",
            "",
            "## Cross-backend and scene-rank evidence",
            "",
            "[`cross_backend_ranking.csv`](tables/cross_backend_ranking.csv) records the five "
            "condition-level correlations plus the pooled correlation and its hierarchical "
            "interval. [`scene_rank_stability.csv`](tables/scene_rank_stability.csv) retains "
            "all 70 scene ranks, average-tie handling, and the frozen rich/weak group checks.",
            "",
            "## Systematic offset, repeatability, and direction",
            "",
            f"The systematic wording decision is `{_fmt(decision['SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED'])}`. "
            "Each geometry-level summary requires exactly ten successful observations and uses "
            "sample covariance (`ddof=1`). The 252 complete rows are retained in "
            "[`systematic_offset_summary.csv`](tables/systematic_offset_summary.csv), with "
            "focused repeatability and direction audit views in their adjacent tables.",
            "",
            "## Common association and reassociation",
            "",
            f"`COMMON_ASSOCIATION_ANALYSIS_PASS = {_fmt(decision['COMMON_ASSOCIATION_ANALYSIS_PASS'])}` and "
            f"`REASSOCIATION_MECHANISM_SUPPORTED = {_fmt(decision['REASSOCIATION_MECHANISM_SUPPORTED'])}`. "
            "The common analyzer is offline and deliberately is not either backend's internal "
            "correspondence set. Validity, turnover, centered and pooled Spearman statistics and "
            "hierarchical intervals are auditable in `common_association_metrics.csv` and "
            "`turnover_correlations.csv`.",
            "",
            "## Local models and automatic candidates",
            "",
            f"`LOCAL_METRIC_INCREMENTAL_VALUE_PASS = {_fmt(decision['LOCAL_METRIC_INCREMENTAL_VALUE_PASS'])}`. "
            "Ridge features are standardized within each training fold and evaluated by "
            "leave-one-geometry-seed-out validation. Every fold result is embedded in "
            "`ridge_model_comparison.csv`. Automatic pairs remain labelled "
            "`AUTOMATIC_CANDIDATE`; they are not promoted to scientific findings.",
            "",
            "## Engineering and independent verification",
            "",
            f"The run manifest binds `{run_manifest['new_trial_count']}` new and "
            f"`{run_manifest['combined_trial_count']}` combined trials. Primary versus independent "
            f"comparison used `atol={comparison['absolute_tolerance']}` and "
            f"`rtol={comparison['relative_tolerance']}`; the maximum numeric difference was "
            f"`{_fmt(comparison['maximum_absolute_numeric_difference'])}` and the section "
            f"difference count was `{comparison['section_difference_count']}`. Artifact SHA and "
            "cardinality checks are recorded in `artifact_verification.json`.",
            "",
            "## Methods and limitation",
            "",
            "Translation error is the Euclidean norm of the estimated-minus-reference translation. "
            "Rotation error is reflection-safe SO(3) geodesic angle, with the relative rotation "
            "projected to SO(3) before its rotation vector is formed. Quantiles use NumPy's linear "
            "method. Exploratory intervals use a three-geometry outer bootstrap and a paired "
            "measurement-seed × repeat inner block bootstrap.",
            "",
            "Only **three geometry seeds** are available. Therefore intervals and p-values here are "
            "descriptive Development diagnostics, not Confirmatory population inference.",
            "",
            "## Authorization boundary",
            "",
            "A true protocol-design authorization permits only design work. It does not authorize a "
            "Confirmatory run, a real-data run, or a measurement-paper mainline claim; those three "
            "authorizations remain false in this artifact.",
            "",
            "## Figure registry",
            "",
        ]
    )
    for item in _chart_map():
        lines.extend(
            [
                f"### {item['figure']}",
                "",
                f"Question: {item['question']} Fields: `{item['fields']}`. "
                f"Supported claim: {item['supported_claim']}. Palette: {item['palette']}.",
                "",
                f"![{item['question']}]({item['output_path']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Table audit index",
            "",
            "| Table | Audit role |",
            "|---|---|",
        ]
    )
    for name in TABLES:
        lines.append(f"| [`{name}`](tables/{name}) | Frozen publication surface |")
    lines.append("")
    return "\n".join(lines)


def _publish_full_synthetic_into(
    *,
    run_dir: str | Path,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    artifact_dir: str | Path,
    manifest_file: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(primary["geometry_seed_raw_values"])
    if len(rows) != 2520:
        raise ValueError("Full Synthetic publication requires 2,520 normalized trials")
    destination = Path(artifact_dir).resolve()
    if not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("publication staging directory must exist and be empty")
    tables = destination / "tables"
    figures = destination / "figures"
    tables.mkdir()
    figures.mkdir()
    comparison = full_synthetic_analysis_verifier_comparison(primary, independent)
    decision, difference = full_synthetic_publication_decision(primary, independent)
    primary_output = dict(primary)
    independent_output = dict(independent)
    primary_output["ANALYSIS_VERIFIER_DIFFERENCE_COUNT"] = difference
    primary_output["ANALYSIS_VERIFIER_COMPARISON"] = comparison
    primary_output["chart_map"] = _chart_map()
    independent_output["ANALYSIS_VERIFIER_DIFFERENCE_COUNT"] = difference
    independent_output["ANALYSIS_VERIFIER_COMPARISON"] = comparison

    snapshot_by_id = {}
    for row in rows:
        snapshot_by_id.setdefault(
            row["snapshot_id"],
            {
                "condition": row["condition"],
                "geometry_seed": row["geometry_seed"],
                "measurement_seed": row["measurement_seed"],
                "reference_pose_checksum": row["reference_pose_checksum"],
                "repeat_index": row["repeat_index"],
                "scene_variant": row["scene_variant"],
                "snapshot_checksum": row["snapshot_checksum"],
                "snapshot_id": row["snapshot_id"],
                "source_checksum": row["source_checksum"],
                "target_checksum": row["target_checksum"],
            },
        )
    trial_fields = (
        "planned_trial_id", "snapshot_id", "scene_variant", "condition", "backend",
        "backend_schema_name", "geometry_seed", "measurement_seed", "repeat_index",
        "solver_failure", "failure_classification", "finite_output", "translation_error_m",
        "rotation_error_rad", "runtime_ms", "source_checksum", "target_checksum",
        "reference_pose_checksum", "snapshot_checksum",
    )
    trial_table = [{name: row.get(name) for name in trial_fields} for row in rows]
    raw_fields = trial_fields + ("translation_vector", "rotation_vector")
    raw_table = [{name: row.get(name) for name in raw_fields} for row in rows]
    systematic = list(primary["systematic_offset_summary"])
    repeatability = [
        {
            key: row[key]
            for key in (
                "backend", "backend_schema_name", "condition", "geometry_seed", "scene_variant",
                "successful_observation_count", "translation_repeatability_covariance",
                "translation_repeatability_rms_m", "rotation_repeatability_covariance",
                "rotation_repeatability_rms_rad",
            )
        }
        for row in systematic
    ]
    direction = [
        {
            key: row[key]
            for key in (
                "backend", "backend_schema_name", "condition", "geometry_seed", "scene_variant",
                "direction_valid_nonzero_count", "translation_direction_concentration",
            )
        }
        for row in systematic
    ]
    common = list(primary["joined_common_association_metrics"])
    local_fields = (
        "planned_trial_id", "snapshot_id", "scene_variant", "condition", "backend",
        "backend_schema_name", "geometry_seed", "translation_error_m",
        "initial_residual_rmse", "initial_correspondence_count", "lambda_min_trans",
        "lambda_mid_trans", "lambda_max_trans", "condition_number_trans",
        "spectral_entropy_trans", "initial_translation_gradient_norm",
    )
    local_geometry = [{name: row.get(name) for name in local_fields} for row in common]
    gates = [
        {"gate": name, "pass": value, "contract": "frozen Development v1"}
        for name, value in sorted(primary["gate_summary"].items())
    ]
    gates.append(
        {
            "gate": "ANALYSIS_VERIFIER_DIFFERENCE_COUNT_EQ_0",
            "pass": difference == 0,
            "contract": "0",
            "value": difference,
        }
    )

    _write_csv(tables / "protocol_summary.csv", primary["protocol_summary"])
    _write_csv(tables / "snapshot_inventory.csv", [snapshot_by_id[key] for key in sorted(snapshot_by_id)])
    _write_csv(tables / "trial_results.csv", trial_table)
    _write_csv(tables / "failure_inventory.csv", primary["failure_inventory"])
    _write_csv(tables / "scene_condition_backend_summary.csv", primary["scene_condition_backend_summary"])
    _write_csv(tables / "geometry_seed_raw_values.csv", raw_table)
    _write_csv(tables / "systematic_offset_summary.csv", systematic)
    _write_csv(tables / "repeatability_summary.csv", repeatability)
    _write_csv(tables / "direction_concentration_summary.csv", direction)
    _write_csv(
        tables / "common_association_metrics.csv",
        common,
        empty_fields=("planned_trial_id", "common_association_valid"),
    )
    _write_csv(
        tables / "local_geometry_metrics.csv",
        local_geometry,
        empty_fields=local_fields,
    )
    _write_csv(tables / "scene_effect_paired.csv", primary["scene_effect_paired"])
    _write_csv(tables / "cross_backend_ranking.csv", primary["cross_backend_ranking"])
    _write_csv(tables / "scene_rank_stability.csv", primary["scene_rank_stability"])
    _write_csv(tables / "condition_contrasts.csv", primary["condition_contrasts"])
    _write_csv(tables / "turnover_correlations.csv", primary["turnover_correlations"])
    _write_csv(tables / "ridge_model_comparison.csv", primary["ridge_model_comparison"])
    _write_csv(
        tables / "automatic_nonequivalence_candidates.csv",
        primary["automatic_nonequivalence_candidates"],
        empty_fields=(
            "candidate_pair_id", "backend", "snapshot_a", "snapshot_b", "scene_a",
            "scene_b", "condition", "linear_metric_similarity", "error_ratio",
            "turnover_difference", "manual_review_status",
        ),
    )
    _write_csv(tables / "runtime_summary.csv", primary["runtime_summary"])
    _write_csv(tables / "gate_summary.csv", gates)

    _scene_condition_figure(
        figures / "scene_condition_translation_error.png",
        primary["scene_condition_backend_summary"],
        "translation_median_m", "translation median (m)", "Scene × condition translation error",
    )
    _scene_condition_figure(
        figures / "scene_condition_rotation_error.png",
        primary["scene_condition_backend_summary"],
        "rotation_median_rad", "rotation median (rad)", "Scene × condition rotation error",
    )
    _paired_figure(figures / "corridor_rich_paired.png", primary["scene_effect_paired"])
    _cross_backend_figure(figures / "cross_backend_scene_ranking.png", primary["cross_backend_ranking"])
    _systematic_scatter(figures / "systematic_offset_vs_repeatability.png", systematic)
    _direction_figure(figures / "direction_concentration.png", systematic)
    _condition_figure(figures / "condition_effects.png", primary["scene_condition_backend_summary"])
    _common_scatter(
        figures / "turnover_vs_translation_error.png", common,
        "correspondence_turnover", "Turnover vs translation error", "common correspondence turnover",
    )
    _common_scatter(
        figures / "local_metric_vs_error.png", common,
        "lambda_min_trans", "Local geometry vs translation error", "lambda_min(H_t)",
    )
    _ridge_figure(figures / "ridge_model_comparison.png", primary["ridge_model_comparison"])
    _runtime_figure(figures / "runtime_comparison.png", primary["runtime_summary"])

    write_json(destination / "primary_analysis.json", primary_output)
    write_json(destination / "independent_verification.json", independent_output)
    write_json(destination / "final_decision.json", decision)
    root = manifest_root(manifest_file)
    from .full_synthetic_development_protocol import read_full_synthetic_plans

    new_snapshots, new_trials, combined_snapshots, combined_trials = (
        read_full_synthetic_plans(root)
    )
    row_ids = [str(row["planned_trial_id"]) for row in rows]
    snapshot_ids = [str(row["snapshot_id"]) for row in rows]
    planned_ids = [str(row["planned_trial_id"]) for row in combined_trials]
    planned_snapshot_ids = [str(row["snapshot_id"]) for row in combined_snapshots]
    backend_counts = Counter(str(row["backend_schema_name"]) for row in rows)
    nonideal = [row for row in rows if row["condition"] != "IDEAL_MATCHED"]
    if (
        len(row_ids) != len(set(row_ids))
        or sorted(row_ids) != sorted(planned_ids)
        or set(snapshot_ids) != set(planned_snapshot_ids)
        or len(set(snapshot_ids)) != 1260
        or len(new_snapshots) != 1050
        or len(new_trials) != 2100
        or len(nonideal) != 2100
        or backend_counts != Counter({BACKENDS[0]: 1260, BACKENDS[1]: 1260})
    ):
        raise ValueError("publication raw rows differ from the frozen combined plan")
    formal_dir = Path(run_dir).resolve()
    if formal_dir != (root / str(manifest["formal_output_dir"])).resolve():
        raise ValueError("publication run directory differs from frozen manifest")
    raw_result_manifest = formal_dir / "raw_result_manifest.json"
    if not raw_result_manifest.is_file():
        raise FileNotFoundError("formal raw result manifest is missing")
    combined_run = {
        "combined_snapshot_count": len(set(snapshot_ids)),
        "combined_trial_count": len(row_ids),
        "experiment_manifest_sha256": file_sha256(manifest_file),
        "implementation_contract_sha256": manifest[
            "implementation_contract_sha256"
        ],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "native_trial_count": 0,
        "new_snapshot_count": len({str(row["snapshot_id"]) for row in nonideal}),
        "new_trial_count": len(nonideal),
        "open3d_trial_count": backend_counts[BACKENDS[0]],
        "pcl_trial_count": backend_counts[BACKENDS[1]],
        "raw_result_manifest_sha256": file_sha256(raw_result_manifest),
        "run_id": str(manifest["formal_run_id"]),
        "scientific_protocol_sha256": manifest["scientific_protocol_sha256"],
        "snapshot_lock_sha256": manifest["new_snapshot_lock_sha256"],
    }
    write_json(destination / "run_manifest.json", combined_run)
    (destination / "full_synthetic_development_report.md").write_text(
        _technical_report(
            primary=primary,
            decision=decision,
            comparison=comparison,
            run_manifest=combined_run,
        ),
        encoding="utf-8",
    )
    checksum_paths = sorted(
        path for path in destination.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "artifact_verification.json"}
    )
    (destination / "SHA256SUMS").write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(destination).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )
    verification = verify_full_synthetic_artifact(destination, write_report=True)
    read_only_verification = verify_full_synthetic_artifact(
        destination, write_report=False
    )
    if (
        verification.get("ARTIFACT_VERIFICATION_PASS") is not True
        or read_only_verification.get("ARTIFACT_VERIFICATION_PASS") is not True
        or verification != read_only_verification
    ):
        raise ValueError("Full Synthetic artifact failed write/read-only verification")
    return {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": difference,
        "ARTIFACT_VERIFICATION_PASS": verification["ARTIFACT_VERIFICATION_PASS"],
        "artifact_path": str(destination),
        "published_file_count": sum(path.is_file() for path in destination.rglob("*")),
        "sha256_mismatch_count": verification["sha256_mismatch_count"],
    }


def publish_full_synthetic_development(
    *,
    manifest_path: str | Path,
    run_dir: str | Path,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Strictly authorize, stage, double-verify, then atomically publish."""

    from .full_synthetic_development_protocol import (
        load_strict_authorized_full_synthetic_manifest,
    )

    manifest_file, manifest = load_strict_authorized_full_synthetic_manifest(
        manifest_path
    )
    destination = Path(artifact_dir).resolve()
    if destination.exists():
        raise FileExistsError("final Full Synthetic artifact directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    ).resolve()
    try:
        result = _publish_full_synthetic_into(
            run_dir=run_dir,
            primary=primary,
            independent=independent,
            artifact_dir=staging,
            manifest_file=manifest_file,
            manifest=manifest,
        )
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    result["artifact_path"] = str(destination)
    return result


__all__ = ["publish_full_synthetic_development"]
