#!/usr/bin/env python3
"""Shared, read-only source and rendering helpers for frozen FMB1 figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
FIG_ROOT = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[4]
RESULT_ROOT = (
    REPO_ROOT
    / "results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1"
)
CORRECTION_ROOT = (
    REPO_ROOT
    / "results/mid360_formal_batch1/zero_perturbation_locked_analysis_reporting_correction_v1"
)
FINAL_DATASET_ROOT = REPO_ROOT / "results/mid360_formal_batch1/final_dataset_v1"

RESULT_TAG = "results/fmb1-zero-perturbation-locked-analysis-v1"
CORRECTION_TAG = "correction/fmb1-solver-convergence-reporting-v1"
DIAGNOSTIC_TAG = "diagnostic/fmb1-solver-convergence-v1"
FIGURE_TAG = "figures/fmb1-paper-data-figures-v1"
STARTING_HEAD = "81f4b566252d09ec5d055bc97c7daa964983ddc8"

SCENE_ORDER = [
    "FMB1_R01",
    "FMB1_R02",
    "FMB1_R03",
    "FMB1_W01",
    "FMB1_W02",
    "FMB1_W03",
]
SCENE_LABEL = {scene: scene.replace("FMB1_", "") for scene in SCENE_ORDER}
SCENE_GROUP = {
    scene: ("Rich" if scene.startswith("FMB1_R") else "Weak")
    for scene in SCENE_ORDER
}
STATION_ORDER = ["S01", "S02", "S03"]
BACKEND_ORDER = ["OPEN3D_POINT_TO_PLANE", "PCL_POINT_TO_PLANE"]
BACKEND_LABEL = {
    "OPEN3D_POINT_TO_PLANE": "Open3D",
    "PCL_POINT_TO_PLANE": "PCL",
}

FONT_FAMILY = "Times New Roman"
BACKEND_STYLE = {
    "OPEN3D_POINT_TO_PLANE": {"color": "#0072B2", "marker": "o"},
    "PCL_POINT_TO_PLANE": {"color": "#D55E00", "marker": "s"},
}
INK = "#252525"
MID_GREY = "#777777"
LIGHT_GREY = "#D7D7D7"
GRID_GREY = "#E5E5E5"
RICH_BG = "#EEF4FA"
WEAK_BG = "#FAF3E8"


SOURCE_SPECS = [
    (FINAL_DATASET_ROOT / "final_geometry_manifest.csv", "Fig05 snapshot geometry", RESULT_TAG),
    (FINAL_DATASET_ROOT / "final_scene_registry.yaml", "Fig05 frozen scene geometry medians and W02 attempt", RESULT_TAG),
    (FINAL_DATASET_ROOT / "final_dataset_readiness.json", "final scene/W02/W04 admission audit", RESULT_TAG),
    (FINAL_DATASET_ROOT / "SHA256SUMS", "final dataset checksum manifest", RESULT_TAG),
    (RESULT_ROOT / "translation_scene_summaries.csv", "Fig11/Fig12 translation scene summaries", RESULT_TAG),
    (RESULT_ROOT / "translation_station_summaries.csv", "Fig11 translation station summaries", RESULT_TAG),
    (RESULT_ROOT / "translation_exact_permutations.csv", "Fig12 frozen exact allocations", RESULT_TAG),
    (RESULT_ROOT / "cross_backend_scene_pairs.csv", "Fig13 frozen scene ordering pairs", RESULT_TAG),
    (RESULT_ROOT / "cross_backend_station_pairs.csv", "Fig13 station value pairs", RESULT_TAG),
    (RESULT_ROOT / "snapshot_direction_cosines.csv", "Fig13 snapshot direction cosines", RESULT_TAG),
    (RESULT_ROOT / "reassociation_scene_summaries.csv", "Fig14 reassociation scene summaries", RESULT_TAG),
    (RESULT_ROOT / "reassociation_centered_rows.csv", "Fig14 frozen centered rows", RESULT_TAG),
    (RESULT_ROOT / "reassociation_centered_permutation_summary.json", "Fig14 frozen permutation summary", RESULT_TAG),
    (RESULT_ROOT / "reassociation_centered_permutation_draws.csv", "Fig14 frozen permutation draws", RESULT_TAG),
    (RESULT_ROOT / "systematic_scene_values.csv", "Fig15 systematic scene values", RESULT_TAG),
    (RESULT_ROOT / "systematic_station_values.csv", "Fig15 systematic station values", RESULT_TAG),
    (RESULT_ROOT / "rotation_scene_summaries.csv", "FigS01 rotation scene summaries", RESULT_TAG),
    (RESULT_ROOT / "rotation_station_summaries.csv", "FigS01 rotation station summaries", RESULT_TAG),
    (RESULT_ROOT / "rotation_exact_permutations.csv", "FigS01 frozen exact allocations", RESULT_TAG),
    (RESULT_ROOT / "analysis_summary.json", "frozen scientific statistics and pair values", RESULT_TAG),
    (RESULT_ROOT / "SHA256SUMS", "scientific result checksum manifest", RESULT_TAG),
    (
        CORRECTION_ROOT / "analysis_summary_reporting_corrected_v1.json",
        "solver/accounting reporting correction v1",
        CORRECTION_TAG,
    ),
    (CORRECTION_ROOT / "SHA256SUMS", "reporting correction checksum manifest", CORRECTION_TAG),
]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY, "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.axisbelow": True,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def verify_checksum_manifest(directory: Path) -> list[dict[str, object]]:
    manifest = directory / "SHA256SUMS"
    if not manifest.is_file():
        raise RuntimeError(f"Missing checksum manifest: {manifest}")
    records: list[dict[str, object]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, filename = line.split(maxsplit=1)
        filename = filename.lstrip("* ")
        target = directory / filename
        actual = sha256_file(target)
        passed = actual == expected
        records.append(
            {
                "relative_path": relative(target),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "pass": passed,
            }
        )
        if not passed:
            raise RuntimeError(f"Frozen checksum mismatch: {target}")
    return records


def source_hash_map() -> dict[str, str]:
    return {relative(path): sha256_file(path) for path, _, _ in SOURCE_SPECS}


def require_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"FAIL CLOSED: {relative(source)} missing columns {missing}")


def require_exact_order(values: Sequence[str], expected: Sequence[str], label: str) -> None:
    if list(values) != list(expected):
        raise RuntimeError(f"FAIL CLOSED: {label} order {list(values)} != {list(expected)}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def source_annotation(paths: Sequence[Path], hashes: Mapping[str, str]) -> tuple[str, str]:
    rels = [relative(path) for path in paths]
    return ";".join(rels), ";".join(hashes[rel] for rel in rels)


def add_source_columns(
    frame: pd.DataFrame, paths: Sequence[Path], hashes: Mapping[str, str]
) -> pd.DataFrame:
    result = frame.copy()
    source_path, source_sha256 = source_annotation(paths, hashes)
    result["source_path"] = source_path
    result["source_sha256"] = source_sha256
    return result


def write_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = FIG_ROOT / "figure_data" / filename
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")
    return path


def scene_sort(frame: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    result = frame.copy()
    result["_scene_order"] = pd.Categorical(
        result["scene_id"], categories=SCENE_ORDER, ordered=True
    )
    columns = ["_scene_order", *extra]
    result = result.sort_values(columns, kind="stable").drop(columns="_scene_order")
    return result.reset_index(drop=True)


def backend_scene_sort(frame: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    result = frame.copy()
    result["_backend_order"] = pd.Categorical(
        result["backend"], categories=BACKEND_ORDER, ordered=True
    )
    result["_scene_order"] = pd.Categorical(
        result["scene_id"], categories=SCENE_ORDER, ordered=True
    )
    result = result.sort_values(
        ["_backend_order", "_scene_order", *extra], kind="stable"
    ).drop(columns=["_backend_order", "_scene_order"])
    return result.reset_index(drop=True)


def formal_stats(cell: str) -> dict[str, float]:
    parsed = json.loads(cell)
    expected = {"median", "q25", "q75", "q95"}
    if set(parsed) != expected:
        raise RuntimeError(f"FAIL CLOSED: unexpected formal_statistics keys {sorted(parsed)}")
    return {key: float(parsed[key]) for key in sorted(expected)}


def add_scene_background(ax: plt.Axes) -> None:
    ax.axvspan(-0.5, 2.5, color=RICH_BG, zorder=-20)
    ax.axvspan(2.5, 5.5, color=WEAK_BG, zorder=-20)
    ax.axvline(2.5, color=MID_GREY, linewidth=0.7, zorder=-10)
    ax.text(1.0, 1.015, "Rich scenes", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.3)
    ax.text(4.0, 1.015, "Weak scenes", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.3)


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, which="major", color=GRID_GREY, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.65, length=3)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.14, 1.075, text, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold")


def set_scene_ticks(ax: plt.Axes) -> None:
    ax.set_xticks(np.arange(6), [SCENE_LABEL[scene] for scene in SCENE_ORDER])
    ax.set_xlim(-0.5, 5.5)


def summary_legend_handles(backend: str) -> list[Line2D]:
    style = BACKEND_STYLE[backend]
    return [
        Line2D([0], [0], color=style["color"], linewidth=0.8, label="q25–q95 summary interval"),
        Line2D([0], [0], color=style["color"], linewidth=3.0, label="q25–q75 summary interval"),
        Line2D([0], [0], marker=style["marker"], linestyle="none", color=style["color"], markerfacecolor=style["color"], markersize=5.5, label="Scene median"),
        Line2D([0], [0], marker=style["marker"], linestyle="none", color=style["color"], markerfacecolor="white", markersize=4, label="Station median"),
    ]


def save_figure(fig: plt.Figure, basename: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for extension in ["pdf", "svg", "png", "tiff"]:
        target = FIG_ROOT / "outputs" / extension / f"{basename}.{extension}"
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.035}
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(target, **kwargs)
        outputs[extension] = relative(target)
    plt.close(fig)
    return outputs


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_directory_checksums() -> None:
    target = FIG_ROOT / "SHA256SUMS"
    files = [
        path
        for path in FIG_ROOT.rglob("*")
        if path.is_file()
        and path != target
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]
    lines = [f"{sha256_file(path)}  {path.relative_to(FIG_ROOT).as_posix()}" for path in sorted(files)]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

