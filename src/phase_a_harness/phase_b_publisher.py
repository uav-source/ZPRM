"""Publish the compact, reviewable Phase B scene-signal artifact."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .contracts import file_sha256, load_manifest, manifest_root, write_json
from .phase_b_analysis import (
    BACKEND_LABELS,
    BACKENDS,
    CONDITIONS,
    SCENES,
    load_and_validate_phase_b_raw,
)
from .phase_b_artifact_verifier import verify_phase_b_artifact
from .phase_b_independent_verifier import phase_b_analysis_verifier_difference_count


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = sorted({key for row in values for key in row})
    if not fields:
        raise ValueError(f"cannot publish empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _condition_scene_bars(path: Path, summaries: list[Mapping[str, Any]], condition: str) -> None:
    figure, axis = plt.subplots(figsize=(11.0, 5.2))
    x = np.arange(len(SCENES), dtype=float)
    width = 0.36
    for offset, backend in ((-width / 2, "Open3D"), (width / 2, "PCL")):
        lookup = {
            row["scene_variant"]: row["translation_median_m"]
            for row in summaries
            if row["condition"] == condition and row["backend"] == backend
        }
        values = [float(lookup[scene]) for scene in SCENES]
        axis.bar(x + offset, values, width=width, label=backend)
    axis.set_xticks(x, [scene.replace("END_FACE_TRANSITION_", "EFT_") for scene in SCENES], rotation=25, ha="right")
    axis.set_ylabel("median translation update (m)")
    axis.set_title(f"Scene signal — {condition}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _ranking_scatter(path: Path, rows: list[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.7))
    markers = {"INDEPENDENT_NOISE_FREE": "o", "FULL_NOISE": "s"}
    for condition in CONDITIONS:
        by_backend = {
            backend: {
                row["scene_variant"]: row["translation_rank_ascending_average_ties"]
                for row in rows
                if row["condition"] == condition and row["backend"] == backend
            }
            for backend in ("Open3D", "PCL")
        }
        axis.scatter(
            [by_backend["Open3D"][scene] for scene in SCENES],
            [by_backend["PCL"][scene] for scene in SCENES],
            marker=markers[condition], s=48, label=condition,
        )
    axis.plot([1, 7], [1, 7], "k--", linewidth=1)
    axis.set(xlim=(0.7, 7.3), ylim=(0.7, 7.3), xlabel="Open3D average rank", ylabel="PCL average rank")
    axis.set_title("Cross-backend scene ranks")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _weak_ratio_bars(path: Path, rows: list[Mapping[str, Any]]) -> None:
    labels = [f"{row['condition'][:4]}\n{row['backend']}\n{row['weak_scene'].replace('PARALLEL_', 'P_').replace('LONG_', 'L_')}" for row in rows]
    values = [0.0 if row["weak_rich_ratio"] is None else float(row["weak_rich_ratio"]) for row in rows]
    colors = ["#2a9d8f" if row["selected_common_weak_scene"] else "#9ca3af" for row in rows]
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    axis.bar(np.arange(len(rows)), values, color=colors)
    axis.axhline(2.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(rows)), labels, fontsize=7)
    axis.set_ylabel("weak / rich translation median")
    axis.set_title("Frozen weak/rich effect candidates")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _consistency_bars(path: Path, rows: list[Mapping[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["backend"])].append(row)
    keys = [(condition, backend) for condition in CONDITIONS for backend in ("Open3D", "PCL")]
    values = [sum(bool(row["direction_pass"]) for row in grouped[key]) for key in keys]
    labels = [f"{condition[:4]}\n{backend}" for condition, backend in keys]
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.bar(np.arange(len(keys)), values, color="#457b9d")
    axis.axhline(2.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(keys)), labels)
    axis.set_ylim(0, 3.3)
    axis.set_ylabel("seeds with weak error > rich error (of 3)")
    axis.set_title("Geometry-seed direction consistency")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _runtime_bars(path: Path, rows: list[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    labels = [row["backend"] for row in rows]
    medians = [float(row["median_runtime_ms"]) for row in rows]
    maxima = [float(row["maximum_runtime_ms"]) for row in rows]
    x = np.arange(len(rows))
    axis.bar(x - 0.18, medians, width=0.36, label="median")
    axis.bar(x + 0.18, maxima, width=0.36, label="maximum")
    axis.set_xticks(x, labels)
    axis.set_ylabel("runtime (ms)")
    axis.set_title("Backend runtime (descriptive)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _gate_rows(primary: Mapping[str, Any], difference_count: int) -> list[dict[str, Any]]:
    rows = [
        {"gate": name, "pass": passed, "threshold_or_contract": "frozen protocol v1"}
        for name, passed in sorted(primary["gate_summary"].items())
    ]
    rows.append(
        {
            "gate": "ANALYSIS_VERIFIER_DIFFERENCE_COUNT_EQ_0",
            "pass": difference_count == 0,
            "threshold_or_contract": "0",
            "value": difference_count,
        }
    )
    for metric in ("solver_failure_count_by_backend", "nonfinite_output_count_by_backend"):
        for backend, value in sorted(primary[metric].items()):
            rows.append(
                {
                    "backend": backend,
                    "gate": f"{metric}_eq_0",
                    "pass": value == 0,
                    "threshold_or_contract": "0",
                    "value": value,
                }
            )
    return rows


def publish_phase_b_signal(
    *,
    manifest_path: str | Path,
    run_dir: str | Path,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Write all required compact evidence and run the standalone verifier."""

    manifest_file, _ = load_manifest(manifest_path, require_authorized=True)
    root = manifest_root(manifest_file)
    audit = load_and_validate_phase_b_raw(manifest_path=manifest_file, run_dir=run_dir)
    if len(audit.rows) != 84:
        raise ValueError("compact Phase B publication requires all 84 valid raw trials")
    destination = Path(artifact_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    tables = destination / "tables"
    figures = destination / "figures"
    tables.mkdir()
    figures.mkdir()

    difference_count = phase_b_analysis_verifier_difference_count(primary, independent)
    primary_output = dict(primary)
    independent_output = dict(independent)
    primary_output["ANALYSIS_VERIFIER_DIFFERENCE_COUNT"] = difference_count
    independent_output["ANALYSIS_VERIFIER_DIFFERENCE_COUNT"] = difference_count
    decision = dict(primary["final_decision"])
    if difference_count:
        decision["PHASE_B_SIGNAL_PASS"] = "NOT_EVALUATED"
        decision["FULL_SYNTHETIC_DEVELOPMENT_PROTOCOL_DESIGN_AUTHORIZED"] = False
        decision["PHASE_B_ENGINEERING_PASS"] = False

    _write_csv(tables / "snapshot_inventory.csv", audit.planned_snapshots)
    plan = {row["planned_trial_id"]: row for row in audit.planned_trials}
    flattened: list[dict[str, Any]] = []
    for row in audit.rows:
        plan_row = plan[row["planned_trial_id"]]
        flattened.append(
            {
                "backend": BACKEND_LABELS[row["backend"]],
                "backend_schema_name": row["backend"],
                "condition": row["condition"],
                "failure_classification": row["failure_classification"],
                "finite_output": row["finite_output"],
                "geometry_seed": int(plan_row.get("geometry_seed", plan_row.get("geometry_seed_value"))),
                "planned_trial_id": row["planned_trial_id"],
                "reference_pose_checksum": row["reference_pose_checksum"],
                "rotation_update_rad": row["rotation_update_rad"],
                "runtime_ms": row["runtime_ms"],
                "scene_variant": row["scene_variant"],
                "snapshot_checksum": row["snapshot_checksum"],
                "snapshot_id": row["snapshot_id"],
                "solver_failure": row["solver_failure"],
                "source_checksum": row["source_checksum"],
                "target_checksum": row["target_checksum"],
                "translation_update_m": row["translation_update_m"],
            }
        )
    flattened.sort(key=lambda row: row["planned_trial_id"])
    _write_csv(tables / "trial_results.csv", flattened)
    _write_csv(tables / "open3d_trial_results.csv", [row for row in flattened if row["backend"] == "Open3D"])
    _write_csv(tables / "pcl_trial_results.csv", [row for row in flattened if row["backend"] == "PCL"])
    _write_csv(tables / "backend_input_pairing.csv", primary["backend_input_pairing"])
    _write_csv(tables / "scene_condition_backend_summary.csv", primary["scene_condition_backend_summary"])
    _write_csv(tables / "geometry_seed_raw_values.csv", primary["geometry_seed_raw_values"])
    _write_csv(tables / "cross_backend_ranking.csv", primary["cross_backend_ranking"])
    _write_csv(tables / "scene_ranking.csv", primary["scene_ranking"])
    _write_csv(tables / "weak_rich_effect.csv", primary["weak_rich_effect"])
    _write_csv(tables / "geometry_seed_consistency.csv", primary["geometry_seed_consistency"])
    _write_csv(tables / "failure_inventory.csv", primary["failure_inventory"])
    _write_csv(tables / "runtime_summary.csv", primary["runtime_summary"])
    _write_csv(tables / "gate_summary.csv", _gate_rows(primary, difference_count))

    summaries = list(primary["scene_condition_backend_summary"])
    _condition_scene_bars(figures / "scene_errors_independent_noise_free.png", summaries, "INDEPENDENT_NOISE_FREE")
    _condition_scene_bars(figures / "scene_errors_full_noise.png", summaries, "FULL_NOISE")
    _ranking_scatter(figures / "open3d_vs_pcl_scene_ranking.png", list(primary["scene_ranking"]))
    _weak_ratio_bars(figures / "weak_rich_ratios.png", list(primary["weak_rich_effect"]))
    _consistency_bars(figures / "geometry_seed_consistency.png", list(primary["geometry_seed_consistency"]))
    _runtime_bars(figures / "backend_runtime.png", list(primary["runtime_summary"]))

    write_json(destination / "primary_analysis.json", primary_output)
    write_json(destination / "independent_verification.json", independent_output)
    write_json(destination / "final_decision.json", decision)
    run_manifest = Path(run_dir).resolve() / "run_manifest.json"
    shutil.copyfile(run_manifest, destination / "run_manifest.json")

    cross = {row["condition"]: row["rho"] for row in primary["cross_backend_ranking"]}
    selected = primary["selected_common_weak_scene"]
    report = [
        "# Zero-Perturbation Phase B — Cross-Backend Scene-Effect Survival Test",
        "",
        "This compact artifact reports the frozen 42-snapshot, 84-trial Phase B run.",
        "No confidence interval is reported for the three-seed scene cells.",
        "",
        "## Decision",
        "",
        f"- `PHASE_B_SIGNAL_PASS = {str(decision['PHASE_B_SIGNAL_PASS']).lower()}`",
        f"- `FULL_SYNTHETIC_DEVELOPMENT_PROTOCOL_DESIGN_AUTHORIZED = {str(decision['FULL_SYNTHETIC_DEVELOPMENT_PROTOCOL_DESIGN_AUTHORIZED']).lower()}`",
        "- `FULL_SYNTHETIC_DEVELOPMENT_RUN_AUTHORIZED = false`",
        "- `CONFIRMATORY_AUTHORIZED = false`",
        "- `REAL_DATA_AUTHORIZED = false`",
        "- `MEASUREMENT_PAPER_MAINLINE_AUTHORIZED = false`",
        "",
        "## Frozen signal checks",
        "",
        f"- Independent-noise-free Spearman rho: `{cross['INDEPENDENT_NOISE_FREE']}`",
        f"- Full-noise Spearman rho: `{cross['FULL_NOISE']}`",
        f"- Independent-noise-free common weak scene: `{selected['INDEPENDENT_NOISE_FREE']}`",
        f"- Full-noise common weak scene: `{selected['FULL_NOISE']}`",
        f"- Main/independent difference count: `{difference_count}`",
        "",
        "All scene medians, three geometry-seed values, average-tie ranks, effect ratios,",
        "absolute differences, failure records, input pairing, and runtimes are in `tables/`.",
    ]
    (destination / "phase_b_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    checksum_paths = sorted(
        path for path in destination.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "artifact_verification.json"}
    )
    (destination / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(path)}  {path.relative_to(destination).as_posix()}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    verification = verify_phase_b_artifact(destination, write_report=True)
    return {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": difference_count,
        "ARTIFACT_VERIFICATION_PASS": verification["ARTIFACT_VERIFICATION_PASS"],
        "artifact_path": str(destination),
        "published_file_count": sum(path.is_file() for path in destination.rglob("*")),
        "sha256_mismatch_count": verification["sha256_mismatch_count"],
    }


__all__ = ["publish_phase_b_signal"]
