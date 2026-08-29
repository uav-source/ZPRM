#!/usr/bin/env python3
"""Fail-closed validation for frozen FMB1 paper figures and figure data."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image

from paper_figure_common import (
    BACKEND_ORDER,
    CORRECTION_ROOT,
    FIG_ROOT,
    FINAL_DATASET_ROOT,
    RESULT_ROOT,
    SCENE_ORDER,
    SOURCE_SPECS,
    relative,
    sha256_file,
    verify_checksum_manifest,
)


FIGURE_BASENAMES = [
    "Fig05_geometry_characterization",
    "Fig11_six_scene_translation_zpru",
    "Fig12_rich_weak_exact_permutation",
    "Fig13_cross_backend_agreement",
    "Fig14_reassociation_association",
    "Fig15_systematic_component",
    "FigS01_rotation_secondary",
]

FIGURE_DATA_FILES = [
    "Fig05_data.csv",
    "Fig11_data.csv",
    "Fig12_scene_data.csv",
    "Fig12_permutation_data.csv",
    "Fig13_scene_pairs.csv",
    "Fig13_station_pairs.csv",
    "Fig13_direction_cosines.csv",
    "Fig14_scene_data.csv",
    "Fig14_centered_data.csv",
    "Fig14_permutation_draws.csv",
    "Fig15_scene_data.csv",
    "Fig15_station_data.csv",
    "FigS01_scene_data.csv",
    "FigS01_station_data.csv",
    "FigS01_permutation_data.csv",
]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"VALIDATION FAILED: {message}")


def ordered_unique(series: pd.Series) -> list:
    return series.drop_duplicates().tolist()


def validate_source_hashes() -> None:
    manifest_path = FIG_ROOT / "audit/source_file_manifest.json"
    check(manifest_path.is_file(), "source_file_manifest.json missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["files"]
    check(len(records) == len(SOURCE_SPECS), "source manifest record count mismatch")
    for record in records:
        path = FIG_ROOT.parents[2] / record["relative_path"]
        check(path.is_file(), f"source missing: {path}")
        check(sha256_file(path) == record["sha256"], f"source SHA mismatch: {path}")
        check(path.stat().st_size == record["bytes"], f"source byte count mismatch: {path}")
    verify_checksum_manifest(RESULT_ROOT)
    verify_checksum_manifest(CORRECTION_ROOT)
    verify_checksum_manifest(FINAL_DATASET_ROOT)


def validate_scene_and_backend_contracts() -> None:
    for filename in FIGURE_DATA_FILES:
        path = FIG_ROOT / "figure_data" / filename
        check(path.is_file(), f"missing figure data: {filename}")
        frame = pd.read_csv(path)
        if "scene_id" in frame:
            scenes = set(frame.scene_id.dropna())
            check(scenes <= set(SCENE_ORDER), f"unexpected scene ID in {filename}: {scenes}")
            text = "\n".join(frame.astype(str).to_numpy().ravel())
            check("FMB1_W04" not in text and "W04" not in text, f"W04 found in {filename}")
        if "backend" in frame:
            backends = ordered_unique(frame.backend)
            if set(backends) == set(BACKEND_ORDER):
                check(backends == BACKEND_ORDER, f"backend order mismatch in {filename}")

    fig05 = pd.read_csv(FIG_ROOT / "figure_data/Fig05_data.csv")
    check(len(fig05) == 180, "Fig05 data must have 180 rows")
    check(ordered_unique(fig05.scene_id) == SCENE_ORDER, "Fig05 scene order mismatch")
    check(set(fig05.loc[fig05.scene_id == "FMB1_W02", "attempt"]) == {2}, "W02 is not attempt 2")
    check(not ((fig05.scene_id == "FMB1_W02") & (fig05.attempt == 1)).any(), "old W02 attempt 1 found")

    fig11 = pd.read_csv(FIG_ROOT / "figure_data/Fig11_data.csv")
    check(len(fig11) == 12, "Fig11 scene summary must have 12 rows")
    for backend in BACKEND_ORDER:
        check(ordered_unique(fig11.loc[fig11.backend == backend, "scene_id"]) == SCENE_ORDER, f"Fig11 scene order mismatch for {backend}")


def validate_frozen_source_counts() -> None:
    translation_scene = pd.read_csv(RESULT_ROOT / "translation_scene_summaries.csv")
    translation_station = pd.read_csv(RESULT_ROOT / "translation_station_summaries.csv")
    translation_perm = pd.read_csv(RESULT_ROOT / "translation_exact_permutations.csv")
    check(len(translation_scene.loc[translation_scene.endpoint == "translation_norm_m"]) == 12, "translation scene rows != 12")
    check(len(translation_station.loc[translation_station.endpoint == "translation_norm_m"]) == 36, "translation station rows != 36")
    check(len(translation_perm) == 40, "translation exact permutation rows != 40")

    analysis = json.loads((RESULT_ROOT / "analysis_summary.json").read_text(encoding="utf-8"))
    check(len(analysis["cross_backend_scene_spearman"]["pairs"]) == 6, "cross-backend scene pairs != 6")
    check(len(pd.read_csv(RESULT_ROOT / "cross_backend_station_pairs.csv")) == 18, "cross-backend station pairs != 18")
    check(len(pd.read_csv(RESULT_ROOT / "snapshot_direction_cosines.csv")) == 180, "direction cosines != 180")

    centered = pd.read_csv(RESULT_ROOT / "reassociation_centered_rows.csv")
    draws = pd.read_csv(RESULT_ROOT / "reassociation_centered_permutation_draws.csv")
    for backend in BACKEND_ORDER:
        check(len(centered.loc[centered.backend == backend]) == 180, f"centered rows != 180 for {backend}")
        check(len(draws.loc[draws.backend == backend]) == 10000, f"permutation draws != 10000 for {backend}")

    check(len(pd.read_csv(RESULT_ROOT / "systematic_scene_values.csv")) == 12, "systematic scene rows != 12")
    check(len(pd.read_csv(RESULT_ROOT / "systematic_station_values.csv")) == 36, "systematic station rows != 36")
    rotation_scene = pd.read_csv(RESULT_ROOT / "rotation_scene_summaries.csv")
    check(len(rotation_scene.loc[rotation_scene.endpoint == "rotation_angle_deg"]) == 12, "rotation degree scene rows != 12")


def validate_frozen_statistics() -> None:
    fig12 = pd.read_csv(FIG_ROOT / "figure_data/Fig12_scene_data.csv")
    check(set(fig12.exact_one_sided_p) == {0.7}, "Fig12 does not report true p=0.7")
    expected_translation = {
        BACKEND_ORDER[0]: -0.026347781758088518,
        BACKEND_ORDER[1]: -0.010915494546904852,
    }
    for backend, expected in expected_translation.items():
        actual = float(fig12.loc[fig12.backend == backend, "observed_weak_minus_rich_mm"].iloc[0])
        check(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), f"Fig12 observed statistic mismatch for {backend}")

    fig13_scene = pd.read_csv(FIG_ROOT / "figure_data/Fig13_scene_pairs.csv")
    fig13_station = pd.read_csv(FIG_ROOT / "figure_data/Fig13_station_pairs.csv")
    check(set(fig13_scene.frozen_scene_spearman_rho) == {1.0}, "Fig13 scene rho mismatch")
    check(math.isclose(float(fig13_station.frozen_station_spearman_rho.iloc[0]), 0.9917440660474718, rel_tol=0.0, abs_tol=1e-15), "Fig13 station rho mismatch")

    fig14_centered = pd.read_csv(FIG_ROOT / "figure_data/Fig14_centered_data.csv")
    expected_centered = {
        BACKEND_ORDER[0]: 0.7376914925357778,
        BACKEND_ORDER[1]: 0.7611716411000339,
    }
    for backend, expected in expected_centered.items():
        actual = float(fig14_centered.loc[fig14_centered.backend == backend, "frozen_centered_rho"].iloc[0])
        check(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15), f"Fig14 centered rho mismatch for {backend}")
        p = float(fig14_centered.loc[fig14_centered.backend == backend, "frozen_permutation_sensitivity_p"].iloc[0])
        check(math.isclose(p, 1.0 / 10001.0, rel_tol=0.0, abs_tol=1e-15), f"Fig14 p_sensitivity mismatch for {backend}")

    fig15 = pd.read_csv(FIG_ROOT / "figure_data/Fig15_scene_data.csv")
    check("formal_p_value" not in fig15.columns and "p_value" not in fig15.columns, "Fig15 must not add a p-value")
    figs01 = pd.read_csv(FIG_ROOT / "figure_data/FigS01_permutation_data.csv")
    check(set(figs01.endpoint) == {"rotation_angle_deg"}, "FigS01 endpoint is not degree rotation")
    p_map = {backend: float(figs01.loc[figs01.backend == backend, "exact_one_sided_p"].iloc[0]) for backend in BACKEND_ORDER}
    check(p_map == {BACKEND_ORDER[0]: 0.9, BACKEND_ORDER[1]: 0.8}, "FigS01 exact p-values mismatch")


def validate_no_nan_infinity() -> None:
    for filename in FIGURE_DATA_FILES:
        frame = pd.read_csv(FIG_ROOT / "figure_data" / filename)
        check(not frame.isna().any().any(), f"NaN found in {filename}")
        numeric = frame.select_dtypes(include="number")
        for column in numeric:
            check(numeric[column].map(math.isfinite).all(), f"nonfinite value in {filename}:{column}")


def validate_outputs() -> None:
    for basename in FIGURE_BASENAMES:
        for extension in ["pdf", "svg", "png", "tiff"]:
            path = FIG_ROOT / "outputs" / extension / f"{basename}.{extension}"
            check(path.is_file() and path.stat().st_size > 0, f"missing/empty figure output: {path}")
        pdf = (FIG_ROOT / "outputs/pdf" / f"{basename}.pdf").read_bytes()
        check(b"/FontFile2" in pdf or b"/FontFile3" in pdf, f"PDF font is not embedded: {basename}")
        check(b"/Type3" not in pdf, f"Type 3 font found in PDF: {basename}")
        for extension in ["png", "tiff"]:
            path = FIG_ROOT / "outputs" / extension / f"{basename}.{extension}"
            with Image.open(path) as image:
                dpi = image.info.get("dpi")
                check(dpi is not None, f"missing DPI metadata: {path}")
                check(abs(float(dpi[0]) - 600.0) < 0.1 and abs(float(dpi[1]) - 600.0) < 0.1, f"raster is not 600 dpi: {path} {dpi}")
                if extension == "tiff":
                    check(image.info.get("compression") == "tiff_lzw", f"TIFF is not LZW: {path}")


def dotted_call_name(node: ast.AST) -> str:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def validate_script_ast() -> None:
    forbidden_modules = {
        "open3d",
        "seaborn",
        "scipy",
        "statsmodels",
        "sklearn",
        "subprocess",
        "phase_a_harness.open3d_backend",
        "phase_a_harness.pcl_backend",
    }
    forbidden_call_tokens = {
        "median",
        "quantile",
        "percentile",
        "spearmanr",
        "permutation_test",
        "bootstrap",
        "ttest_ind",
        "mannwhitneyu",
        "linregress",
        "polyfit",
        "lstsq",
        "lowess",
        "run_icp",
        "registration_icp",
        "default_rng",
    }
    violations = []
    for path in sorted((FIG_ROOT / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules or alias.name.split(".")[0] in forbidden_modules:
                        violations.append((path.name, node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_modules or module.split(".")[0] in forbidden_modules:
                    violations.append((path.name, node.lineno, f"from {module}"))
            elif isinstance(node, ast.Call):
                name = dotted_call_name(node.func)
                if name.split(".")[-1] in forbidden_call_tokens:
                    violations.append((path.name, node.lineno, f"call {name}"))
    check(not violations, f"forbidden scientific/backend calls: {violations}")
    audit = json.loads((FIG_ROOT / "audit/no_registration_import_audit.json").read_text(encoding="utf-8"))
    check(audit["REGISTRATION_BACKEND_CALL_COUNT"] == 0, "registration audit count is not zero")


def validate_caption_language() -> None:
    forbidden = [
        "intrinsic error",
        "positioning error",
        "reproducibility",
        "caused by reassociation",
        "360 independent samples",
        "significance stars",
    ]
    for filename in ["figure_captions_en.md", "figure_captions_zh.md"]:
        text = (FIG_ROOT / "captions" / filename).read_text(encoding="utf-8").lower()
        for term in forbidden:
            check(term not in text, f"forbidden caption term {term!r} in {filename}")
    english = (FIG_ROOT / "captions/figure_captions_en.md").read_text(encoding="utf-8")
    check("association and does not establish" in english, "Fig14 association limitation missing")
    check("no formal p-value was specified" in english, "Fig15 no-formal-p statement missing")
    check("secondary endpoint" in english.lower(), "rotation secondary role missing")


def validate_directory_checksums() -> None:
    manifest = FIG_ROOT / "SHA256SUMS"
    check(manifest.is_file(), "directory SHA256SUMS missing")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relpath = line.split(maxsplit=1)
        relpath = relpath.lstrip("* ")
        listed.add(relpath)
        path = FIG_ROOT / relpath
        check(path.is_file(), f"SHA256SUMS target missing: {relpath}")
        check(sha256_file(path) == expected, f"directory checksum mismatch: {relpath}")
    actual = {
        path.relative_to(FIG_ROOT).as_posix()
        for path in FIG_ROOT.rglob("*")
        if path.is_file()
        and path != manifest
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    check(listed == actual, "directory SHA256SUMS coverage mismatch")


def main() -> None:
    validate_source_hashes()
    validate_scene_and_backend_contracts()
    validate_frozen_source_counts()
    validate_frozen_statistics()
    validate_no_nan_infinity()
    validate_outputs()
    validate_script_ast()
    validate_caption_language()
    validate_directory_checksums()
    print("PAPER_FIGURE_VALIDATION=PASS")
    print("SOURCE_SHA_VALIDATION=PASS")
    print("SCENE_ORDER_VALIDATION=PASS")
    print("W04_COUNT=0")
    print("OLD_W02_ATTEMPT1_COUNT=0")
    print("REGISTRATION_BACKEND_CALL_COUNT=0")
    print("NEW_HYPOTHESIS_TEST_COUNT=0")
    print("FORMAL_STATISTIC_RECOMPUTATION_COUNT=0")
    print("FROZEN_RESULT_MODIFICATION_COUNT=0")


if __name__ == "__main__":
    main()

