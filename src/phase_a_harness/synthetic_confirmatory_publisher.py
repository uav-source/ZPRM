"""Atomic publication of future Synthetic Confirmatory formal evidence."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .contracts import file_sha256, write_json
from .synthetic_confirmatory_analysis import load_synthetic_confirmatory_raw
from .synthetic_confirmatory_artifact_verifier import (
    FORMAL_FIGURES,
    FORMAL_TABLES,
    verify_synthetic_confirmatory_artifact,
)
from .synthetic_confirmatory_independent_verifier import (
    compare_primary_and_independent,
)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    fields = sorted({key for row in values for key in row})
    if not fields:
        raise ValueError(f"refusing headerless Confirmatory table: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in values:
            writer.writerow({
                key: json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def _strict_json(path: Path) -> dict[str, Any]:
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
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _figures(destination: Path, primary: Mapping[str, Any]) -> None:
    figures = destination / "figures"
    gate_items = list(primary["gate_summary"].items())
    figure, axis = plt.subplots(figsize=(9.0, 4.5))
    axis.bar(
        np.arange(len(gate_items)),
        [1.0 if value is True else 0.0 for _, value in gate_items],
        color=["#2d7f5e" if value is True else "#a33a3a" for _, value in gate_items],
    )
    axis.set_xticks(np.arange(len(gate_items)), labels=[name for name, _ in gate_items])
    axis.tick_params(axis="x", labelrotation=35)
    axis.set_ylim(0.0, 1.1)
    axis.set_ylabel("pass (1) / fail (0)")
    axis.set_title("Frozen H1--H6 gate matrix")
    figure.tight_layout()
    figure.savefig(figures / FORMAL_FIGURES[0], dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    h3 = primary["h3_cross_backend_ranking"]
    for axis, row in zip(axes, h3):
        medians = row["scene_medians"]
        for backend, values in medians.items():
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

    h5 = primary["h5_frozen_models"]
    locations = np.arange(len(h5))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.bar(
        locations - width / 2,
        [row["model_a_mae"] for row in h5],
        width,
        label="Frozen Model A",
    )
    axis.bar(
        locations + width / 2,
        [row["model_b_mae"] for row in h5],
        width,
        label="Frozen Model B",
    )
    axis.set_xticks(
        locations, labels=[row["backend_schema_name"] for row in h5]
    )
    axis.set_ylabel("MAE on log10(error + 1e-9)")
    axis.set_title("Frozen-model confirmatory comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / FORMAL_FIGURES[2], dpi=160)
    plt.close(figure)


def _report(primary: Mapping[str, Any], comparison: Mapping[str, Any]) -> str:
    decision = primary["final_decision"]
    lines = [
        "# Synthetic Confirmatory v1",
        "",
        "This compact artifact reports the preregistered H1--H6 analysis over the complete frozen formal evidence.",
        "",
        "## Decision",
        "",
        f"- `SYNTHETIC_CONFIRMATORY_EXECUTED = {str(decision['SYNTHETIC_CONFIRMATORY_EXECUTED']).lower()}`",
        f"- `SYNTHETIC_CONFIRMATORY_COMPLETE = {str(decision['SYNTHETIC_CONFIRMATORY_COMPLETE']).lower()}`",
        f"- `SYNTHETIC_CONFIRMATORY_PASS = {str(decision['SYNTHETIC_CONFIRMATORY_PASS']).lower()}`",
        f"- Analysis/verifier differing leaves: `{comparison['leaf_difference_count']}`",
        "- `REAL_DATA_RUN_AUTHORIZED = false`",
        "- `MEASUREMENT_PAPER_MAINLINE_AUTHORIZED = false`",
        "",
        "## Evidence inventory",
        "",
    ]
    lines.extend(f"- `tables/{name}`" for name in FORMAL_TABLES)
    lines.extend(f"- `figures/{name}`" for name in FORMAL_FIGURES)
    lines.extend([
        "",
        "H4 is an association result only; no causal claim is authorized.",
        "Frozen Model A/B parameters were applied without fitting, restandardization, feature changes, or alpha changes.",
        "",
    ])
    return "\n".join(lines)


def _publish_into(
    destination: Path,
    *,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "tables").mkdir()
    (destination / "figures").mkdir()
    comparison = compare_primary_and_independent(primary, independent)
    if (
        comparison["leaf_difference_count"] != 0
        or comparison["maximum_absolute_numeric_difference"] != 0.0
    ):
        raise ValueError("primary/independent Confirmatory analysis differs")
    decision = primary.get("final_decision")
    if (
        type(decision) is not dict
        or decision.get("SYNTHETIC_CONFIRMATORY_EXECUTED") is not True
        or decision.get("SYNTHETIC_CONFIRMATORY_COMPLETE") is not True
        or independent.get("final_decision") != decision
    ):
        raise ValueError("Confirmatory formal decision is incomplete or inconsistent")
    live_run_id = run_manifest.get("run_id")
    live_raw_sha = run_manifest.get("raw_result_manifest_sha256")
    if (
        not isinstance(live_run_id, str)
        or not isinstance(live_raw_sha, str)
        or any(
            report.get("run_id") != live_run_id
            or report.get("raw_result_manifest_sha256") != live_raw_sha
            for report in (primary, independent)
        )
    ):
        raise ValueError("Confirmatory analyses are not bound to live raw evidence")

    table_rows = {
        FORMAL_TABLES[0]: primary["h1_ideal_control"],
        FORMAL_TABLES[1]: primary["h2_scene_effect"],
        FORMAL_TABLES[2]: primary["h3_cross_backend_ranking"],
        FORMAL_TABLES[3]: primary["h4_reassociation"],
        FORMAL_TABLES[4]: primary["h5_frozen_models"],
        FORMAL_TABLES[5]: [
            *({**row, "row_type": "geometry"} for row in primary["h6_systematic_groups"]),
            *({**row, "row_type": "backend"} for row in primary["h6_systematic_backend"]),
        ],
        FORMAL_TABLES[6]: [
            {"gate": name, "pass": value}
            for name, value in primary["gate_summary"].items()
        ],
    }
    for name, rows in table_rows.items():
        _write_csv(destination / "tables" / name, rows)
    _figures(destination, primary)
    write_json(destination / "primary_analysis.json", dict(primary))
    write_json(destination / "independent_verification.json", dict(independent))
    write_json(destination / "final_decision.json", dict(decision))
    write_json(destination / "run_manifest.json", dict(run_manifest))
    (destination / "synthetic_confirmatory_report.md").write_text(
        _report(primary, comparison), encoding="utf-8"
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
    verification = verify_synthetic_confirmatory_artifact(
        destination, write_report=True
    )
    read_only = verify_synthetic_confirmatory_artifact(
        destination, write_report=False
    )
    if (
        verification != read_only
        or verification.get("ARTIFACT_VERIFICATION_PASS") is not True
    ):
        raise ValueError("Synthetic Confirmatory artifact verification failed")
    return verification


def publish_synthetic_confirmatory(
    *,
    manifest_path: str | Path,
    run_dir: str | Path,
    primary: Mapping[str, Any],
    independent: Mapping[str, Any],
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Validate complete raw evidence, stage, verify twice, then publish once."""

    # This preflight is intentionally before any output-directory mutation.
    _trials, raw_manifest, _root, manifest = load_synthetic_confirmatory_raw(
        manifest_path=manifest_path, run_dir=run_dir
    )
    run_manifest_path = Path(run_dir).resolve() / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError("Synthetic Confirmatory run manifest is missing")
    run_manifest = _strict_json(run_manifest_path)
    live_raw_sha = file_sha256(Path(run_dir).resolve() / "raw_result_manifest.json")
    if (
        run_manifest.get("run_id") != manifest["formal_run_id"]
        or raw_manifest.get("run_id") != manifest["formal_run_id"]
    ):
        raise ValueError("Synthetic Confirmatory run identity mismatch")
    comparison = compare_primary_and_independent(primary, independent)
    if (
        comparison["leaf_difference_count"] != 0
        or comparison["maximum_absolute_numeric_difference"] != 0.0
    ):
        raise ValueError("primary/independent Confirmatory analysis differs")
    if any(
        report.get("run_id") != manifest["formal_run_id"]
        or report.get("raw_result_manifest_sha256") != live_raw_sha
        for report in (primary, independent)
    ):
        raise ValueError("supplied analysis is not bound to the live raw manifest")
    run_manifest = {
        **run_manifest,
        "run_id": manifest["formal_run_id"],
        "raw_result_manifest_sha256": live_raw_sha,
    }

    destination = Path(artifact_dir).resolve()
    if destination.exists():
        raise FileExistsError("refusing to replace Synthetic Confirmatory artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.staging-", dir=destination.parent
    )).resolve()
    staging = temporary / "artifact"
    try:
        verification = _publish_into(
            staging,
            primary=primary,
            independent=independent,
            run_manifest=run_manifest,
        )
        os.replace(staging, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "ANALYSIS_VERIFIER_DIFFERENCE_COUNT": comparison["leaf_difference_count"],
        "ARTIFACT_VERIFICATION_PASS": verification["ARTIFACT_VERIFICATION_PASS"],
        "artifact_path": str(destination),
        "published_file_count": sum(
            path.is_file() for path in destination.rglob("*")
        ),
        "sha256_mismatch_count": len(verification["sha256_mismatch_files"]),
    }


__all__ = ["publish_synthetic_confirmatory"]
