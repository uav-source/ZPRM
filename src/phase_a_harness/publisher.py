"""Publish compact tables, figures, decision, and integrity inventory."""

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

from .analysis import load_and_validate_raw
from .artifact_verifier import verify_formal_artifact
from .contracts import file_sha256, load_manifest, manifest_root, write_json


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = sorted({key for row in values for key in row})
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


def _boxplot(path: Path, labels: list[str], groups: list[list[float]], title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.boxplot(groups, labels=labels, showfliers=True)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def publish_formal_phase_a(
    *, manifest_path: str | Path, run_dir: str | Path, primary: Mapping[str, Any],
    independent: Mapping[str, Any], artifact_dir: str | Path,
) -> dict[str, Any]:
    manifest_file, manifest = load_manifest(manifest_path, require_authorized=True)
    root = manifest_root(manifest_file)
    destination = Path(artifact_dir).resolve()
    tables = destination / "tables"
    figures = destination / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    _, rows, _ = load_and_validate_raw(manifest_path=manifest_file, run_dir=run_dir)
    shutil.copyfile(root / "frozen_assets/snapshot_inventory.csv", tables / "snapshot_inventory.csv")

    flattened = [
        {
            "backend": "Open3D" if row["backend"] == "open3d_point_to_plane" else "PCL",
            "condition": row["condition"],
            "failure_classification": row["failure_classification"],
            "finite_output": row["finite_output"],
            "planned_trial_id": row["planned_trial_id"],
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
        for row in rows
    ]
    _csv(tables / "trial_results.csv", flattened)
    _csv(tables / "open3d_trial_results.csv", [row for row in flattened if row["backend"] == "Open3D"])
    _csv(tables / "pcl_trial_results.csv", [row for row in flattened if row["backend"] == "PCL"])
    paired: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flattened:
        paired[row["snapshot_id"]].append(row)
    pairing_rows = []
    for snapshot_id, selected in sorted(paired.items()):
        pairing_rows.append(
            {
                "backend_count": len(selected),
                "input_checksum_match": len(selected) == 2
                and len({row["source_checksum"] for row in selected}) == 1
                and len({row["target_checksum"] for row in selected}) == 1
                and len({row["snapshot_checksum"] for row in selected}) == 1,
                "snapshot_id": snapshot_id,
            }
        )
    _csv(tables / "backend_input_pairing.csv", pairing_rows)
    _csv(tables / "backend_summary.csv", primary["backend_summary"])
    _csv(tables / "scene_backend_summary.csv", primary["scene_backend_summary"])
    _csv(tables / "failure_inventory.csv", primary["failure_inventory"])
    runtime_summary = []
    for backend in ("Open3D", "PCL"):
        values = np.asarray([row["runtime_ms"] for row in flattened if row["backend"] == backend], dtype=float)
        runtime_summary.append(
            {
                "backend": backend,
                "maximum_runtime_ms": float(np.max(values)),
                "median_runtime_ms": float(np.median(values)),
                "q95_runtime_ms": float(np.quantile(values, 0.95, method="linear")),
                "trial_count": int(values.size),
            }
        )
    _csv(tables / "runtime_summary.csv", runtime_summary)
    gate_rows = []
    for row in primary["backend_summary"]:
        gate_rows.extend(
            [
                {"backend": row["backend"], "gate": "trial_count_210", "pass": row["trial_count"] == 210, "value": row["trial_count"]},
                {"backend": row["backend"], "gate": "solver_failure_count_0", "pass": row["solver_failure_count"] == 0, "value": row["solver_failure_count"]},
                {"backend": row["backend"], "gate": "nonfinite_output_count_0", "pass": row["nonfinite_output_count"] == 0, "value": row["nonfinite_output_count"]},
                {"backend": row["backend"], "gate": "translation_q95_le_0.001", "pass": row["translation_q95_m"] <= 0.001, "value": row["translation_q95_m"]},
                {"backend": row["backend"], "gate": "rotation_q95_le_threshold", "pass": row["rotation_q95_rad"] <= 0.00017453292519943296, "value": row["rotation_q95_rad"]},
                {"backend": row["backend"], "gate": "fraction_translation_le_1mm_ge_0.95", "pass": row["fraction_translation_le_1mm"] >= 0.95, "value": row["fraction_translation_le_1mm"]},
            ]
        )
    _csv(tables / "gate_summary.csv", gate_rows)

    translation_groups = [[row["translation_update_m"] for row in flattened if row["backend"] == backend] for backend in ("Open3D", "PCL")]
    rotation_groups = [[row["rotation_update_rad"] for row in flattened if row["backend"] == backend] for backend in ("Open3D", "PCL")]
    runtime_groups = [[row["runtime_ms"] for row in flattened if row["backend"] == backend] for backend in ("Open3D", "PCL")]
    _boxplot(figures / "backend_translation_updates.png", ["Open3D", "PCL"], translation_groups, "Zero-Perturbation Translation Updates", "translation update (m)")
    _boxplot(figures / "backend_rotation_updates.png", ["Open3D", "PCL"], rotation_groups, "Zero-Perturbation Rotation Updates", "rotation update (rad)")
    _boxplot(figures / "runtime_comparison.png", ["Open3D", "PCL"], runtime_groups, "Backend Runtime Comparison", "runtime (ms)")
    scenes = sorted({row["scene_variant"] for row in flattened})
    scene_groups = [[row["translation_update_m"] for row in flattened if row["scene_variant"] == scene] for scene in scenes]
    _boxplot(figures / "scene_translation_updates.png", scenes, scene_groups, "Translation Updates by Scene", "translation update (m)")
    open_by_id = {row["snapshot_id"]: row["translation_update_m"] for row in flattened if row["backend"] == "Open3D"}
    pcl_by_id = {row["snapshot_id"]: row["translation_update_m"] for row in flattened if row["backend"] == "PCL"}
    ids = sorted(set(open_by_id) & set(pcl_by_id))
    figure, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.scatter([open_by_id[item] for item in ids], [pcl_by_id[item] for item in ids], s=13, alpha=0.65)
    maximum = max([*open_by_id.values(), *pcl_by_id.values(), 1e-12])
    axis.plot([0.0, maximum], [0.0, maximum], "k--", linewidth=1)
    axis.set_xlabel("Open3D translation update (m)")
    axis.set_ylabel("PCL translation update (m)")
    axis.set_title("Paired Backend Translation Updates")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "open3d_vs_pcl_updates.png", dpi=160)
    plt.close(figure)

    write_json(destination / "final_decision.json", primary["final_decision"])
    write_json(destination / "run_manifest.json", json.loads((Path(run_dir) / "run_manifest.json").read_text(encoding="utf-8")))
    write_json(destination / "independent_verification.json", independent)
    write_json(destination / "primary_analysis.json", primary)
    shutil.copyfile(root / "frozen_assets/source_export_manifest.csv", destination / "source_export_manifest.csv")
    shutil.copyfile(root / "frozen_assets/environment_manifest.json", destination / "environment_manifest.json")
    decision = primary["final_decision"]
    report_lines = [
        "# Zero-Perturbation Phase A — Minimal Standalone Harness",
        "",
        "Formal scientific run over 210 frozen snapshots and 420 paired trials.",
        "",
        "## Decision",
        "",
        f"- `TWO_INDEPENDENT_BACKENDS_QUALIFIED = {str(decision['TWO_INDEPENDENT_BACKENDS_QUALIFIED']).lower()}`",
        f"- `BACKEND_PHASE_A_COMPLETE = {str(decision['BACKEND_PHASE_A_COMPLETE']).lower()}`",
        f"- `DAY1_SCIENTIFIC_VALIDATION_PASS = {str(decision['DAY1_SCIENTIFIC_VALIDATION_PASS']).lower()}`",
        f"- `PHASE_B_PROTOCOL_DESIGN_AUTHORIZED = {str(decision['PHASE_B_PROTOCOL_DESIGN_AUTHORIZED']).lower()}`",
        "- `PHASE_B_RUN_AUTHORIZED = false`",
        "",
        "The main analysis and independent verifier both re-read the raw trial files.",
    ]
    (destination / "phase_a_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    checksum_paths = sorted(
        path for path in destination.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "artifact_verification.json"}
    )
    (destination / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(path)}  {path.relative_to(destination).as_posix()}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    verification = verify_formal_artifact(destination, write_report=True)
    return {
        "ARTIFACT_VERIFICATION_PASS": verification["ARTIFACT_VERIFICATION_PASS"],
        "artifact_path": str(destination),
        "published_file_count": sum(path.is_file() for path in destination.rglob("*")),
        "sha256_mismatch_count": verification["sha256_mismatch_count"],
    }


__all__ = ["publish_formal_phase_a"]

