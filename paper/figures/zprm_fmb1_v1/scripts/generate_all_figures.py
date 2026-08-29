#!/usr/bin/env python3
"""Generate seven paper figures from frozen FMB1 CSV/JSON sources only."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from paper_figure_common import (
    BACKEND_LABEL,
    BACKEND_ORDER,
    BACKEND_STYLE,
    CORRECTION_ROOT,
    CORRECTION_TAG,
    DIAGNOSTIC_TAG,
    FIGURE_TAG,
    FIG_ROOT,
    FINAL_DATASET_ROOT,
    FONT_FAMILY,
    GRID_GREY,
    INK,
    LIGHT_GREY,
    MID_GREY,
    REPO_ROOT,
    RESULT_ROOT,
    RESULT_TAG,
    RICH_BG,
    SCENE_GROUP,
    SCENE_LABEL,
    SCENE_ORDER,
    SOURCE_SPECS,
    STARTING_HEAD,
    STATION_ORDER,
    WEAK_BG,
    add_scene_background,
    add_source_columns,
    backend_scene_sort,
    configure_matplotlib,
    formal_stats,
    panel_label,
    read_csv,
    read_json,
    relative,
    require_columns,
    save_figure,
    scene_sort,
    set_scene_ticks,
    sha256_file,
    source_annotation,
    source_hash_map,
    style_axis,
    summary_legend_handles,
    verify_checksum_manifest,
    write_csv,
    write_directory_checksums,
    write_json,
)


WIDTH_IN = 178.0 / 25.4
GROUP_MEDIAN_SCENE = {"Rich": "FMB1_R01", "Weak": "FMB1_W02"}
TASK_SPECIFIED_DIRECTION_COSINE_MEDIAN = 0.997739005
POINT_LABEL_OFFSETS = {
    "FMB1_R01": (4, 6),
    "FMB1_R02": (4, 4),
    "FMB1_R03": (4, 5),
    "FMB1_W01": (4, -13),
    "FMB1_W02": (-27, 3),
    "FMB1_W03": (4, 5),
}


def ensure_directories() -> None:
    for relative_dir in [
        "scripts",
        "figure_data",
        "outputs/pdf",
        "outputs/svg",
        "outputs/png",
        "outputs/tiff",
        "captions",
        "audit",
    ]:
        (FIG_ROOT / relative_dir).mkdir(parents=True, exist_ok=True)


def verify_frozen_roots() -> dict[str, object]:
    checks = {
        relative(RESULT_ROOT): verify_checksum_manifest(RESULT_ROOT),
        relative(CORRECTION_ROOT): verify_checksum_manifest(CORRECTION_ROOT),
        relative(FINAL_DATASET_ROOT): verify_checksum_manifest(FINAL_DATASET_ROOT),
    }
    return {"all_pass": True, "checksum_roots": checks}


def build_source_audits(hashes: dict[str, str], checksum_audit: dict[str, object]) -> None:
    records = []
    for path, role, tag in SOURCE_SPECS:
        if not path.is_file():
            raise RuntimeError(f"FAIL CLOSED: missing source {path}")
        rel = relative(path)
        records.append(
            {
                "relative_path": rel,
                "sha256": hashes[rel],
                "bytes": path.stat().st_size,
                "role": role,
                "scientific_source_commit_or_tag": tag,
            }
        )
    manifest = {
        "schema": "zprm_fmb1_paper_figure_source_manifest_v1",
        "generation_base_commit": STARTING_HEAD,
        "source_tags": [RESULT_TAG, CORRECTION_TAG, DIAGNOSTIC_TAG],
        "checksum_verification": checksum_audit,
        "files": records,
    }
    write_json(FIG_ROOT / "audit/source_file_manifest.json", manifest)

    csv_records = []
    for path, role, tag in SOURCE_SPECS:
        if path.suffix.lower() != ".csv":
            continue
        frame = read_csv(path)

        def unique(column: str):
            if column not in frame.columns:
                return []
            values = []
            for value in frame[column].drop_duplicates().tolist():
                if pd.isna(value):
                    values.append(None)
                elif isinstance(value, np.generic):
                    values.append(value.item())
                else:
                    values.append(value)
            return values

        csv_records.append(
            {
                "relative_path": relative(path),
                "role": role,
                "scientific_source_commit_or_tag": tag,
                "columns": frame.columns.tolist(),
                "row_count": int(len(frame)),
                "unique_scene_ids": unique("scene_id"),
                "unique_station_ids": unique("station_id"),
                "unique_backends": unique("backend"),
                "endpoint_names": unique("endpoint"),
                "formal_summary_status": unique("formal_summary_status"),
                "status_values": unique("status"),
                "attempt_values": unique("attempt"),
                "field_presence": {
                    key: key in frame.columns
                    for key in [
                        "scene_id",
                        "station_id",
                        "backend",
                        "endpoint",
                        "formal_summary_status",
                        "status",
                        "attempt",
                    ]
                },
            }
        )
    schema_audit = {
        "schema": "zprm_fmb1_source_schema_audit_v1",
        "fail_closed_on_ambiguous_schema": True,
        "csv_files": csv_records,
    }
    write_json(FIG_ROOT / "audit/source_schema_audit.json", schema_audit)
    lines = [
        "# Frozen source schema audit",
        "",
        "All columns below were read directly from the frozen CSV headers. An empty unique-value list means that the named field is absent, not inferred.",
        "",
    ]
    for record in csv_records:
        lines.extend(
            [
                f"## `{record['relative_path']}`",
                "",
                f"- Role: {record['role']}",
                f"- Rows: {record['row_count']}",
                f"- Columns: `{', '.join(record['columns'])}`",
                f"- Scene IDs: `{record['unique_scene_ids']}`",
                f"- Station IDs: `{record['unique_station_ids']}`",
                f"- Backends: `{record['unique_backends']}`",
                f"- Endpoints: `{record['endpoint_names']}`",
                f"- formal_summary_status: `{record['formal_summary_status']}`",
                f"- status: `{record['status_values']}`",
                f"- attempt: `{record['attempt_values']}`",
                "",
            ]
        )
    (FIG_ROOT / "audit/source_schema_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def assert_scene_contract(frame: pd.DataFrame, source: str) -> None:
    if "scene_id" not in frame.columns:
        raise RuntimeError(f"FAIL CLOSED: {source} has no scene_id")
    scenes = frame["scene_id"].drop_duplicates().tolist()
    if scenes != SCENE_ORDER:
        raise RuntimeError(f"FAIL CLOSED: {source} scene order {scenes}")


def build_fig05_data(hashes: dict[str, str]) -> pd.DataFrame:
    geometry_path = FINAL_DATASET_ROOT / "final_geometry_manifest.csv"
    registry_path = FINAL_DATASET_ROOT / "final_scene_registry.yaml"
    readiness_path = FINAL_DATASET_ROOT / "final_dataset_readiness.json"
    geometry = read_csv(geometry_path)
    require_columns(
        geometry,
        [
            "scene_id",
            "station_id",
            "selection_index",
            "snapshot_id",
            "normalized_lambda_min_trans",
            "condition_number_trans",
            "spectral_entropy_trans",
            "attempt",
        ],
        geometry_path,
    )
    if len(geometry) != 180:
        raise RuntimeError("FAIL CLOSED: geometry manifest must have 180 snapshot rows")
    assert_scene_contract(geometry, relative(geometry_path))
    if set(geometry.loc[geometry.scene_id == "FMB1_W02", "attempt"]) != {2}:
        raise RuntimeError("FAIL CLOSED: W02 geometry is not exclusively attempt 2")
    registry = read_json(registry_path)
    if registry.get("schema") != "mid360_fmb1_final_scene_registry_v1":
        raise RuntimeError("FAIL CLOSED: unexpected final scene registry schema")
    registry_rows = {row["scene_id"]: row for row in registry["scenes"]}
    if list(registry_rows) != SCENE_ORDER:
        raise RuntimeError("FAIL CLOSED: final scene registry order mismatch")
    readiness = read_json(readiness_path)
    if not readiness["FMB1_W02_ATTEMPT2_INCLUDED_IN_FINAL_SET"]:
        raise RuntimeError("FAIL CLOSED: readiness does not admit W02 attempt 2")
    if readiness["FMB1_W02_ATTEMPT1_SNAPSHOT_COUNT_IN_FINAL_SET"] != 0:
        raise RuntimeError("FAIL CLOSED: W02 attempt 1 appears in final set")
    if readiness["W04_INCLUDED_IN_FINAL_SET"]:
        raise RuntimeError("FAIL CLOSED: W04 appears in final set")

    output = geometry[
        [
            "scene_id",
            "station_id",
            "selection_index",
            "snapshot_id",
            "attempt",
            "normalized_lambda_min_trans",
            "condition_number_trans",
            "spectral_entropy_trans",
        ]
    ].copy()
    output["scene_label"] = output.scene_id.map(SCENE_LABEL)
    output["geometry_group"] = output.scene_id.map(SCENE_GROUP)
    output["scene_median_normalized_lambda_min_trans"] = output.scene_id.map(
        {scene: registry_rows[scene]["median_normalized_lambda_min_trans"] for scene in SCENE_ORDER}
    )
    output["scene_median_condition_number_trans"] = output.scene_id.map(
        {scene: registry_rows[scene]["median_condition_number_trans"] for scene in SCENE_ORDER}
    )
    output["scene_median_spectral_entropy_trans"] = output.scene_id.map(
        {scene: registry_rows[scene]["median_spectral_entropy_trans"] for scene in SCENE_ORDER}
    )
    output = scene_sort(output, ["station_id", "selection_index"])
    output = add_source_columns(
        output, [geometry_path, registry_path, readiness_path], hashes
    )
    write_csv(output, "Fig05_data.csv")
    return output


def extract_interval_data(
    scene_path: Path,
    station_path: Path,
    endpoint: str,
    value_scale: float,
    hashes: dict[str, str],
    output_filename: str,
) -> pd.DataFrame:
    scene = read_csv(scene_path)
    station = read_csv(station_path)
    require_columns(
        scene,
        ["backend", "scene_id", "endpoint", "formal_statistics", "formal_summary_status"],
        scene_path,
    )
    require_columns(
        station,
        [
            "backend",
            "scene_id",
            "station_id",
            "endpoint",
            "formal_statistics",
            "formal_summary_status",
        ],
        station_path,
    )
    scene = scene.loc[scene.endpoint == endpoint].copy()
    station = station.loc[station.endpoint == endpoint].copy()
    if len(scene) != 12 or len(station) != 36:
        raise RuntimeError(
            f"FAIL CLOSED: {endpoint} must have 12 scene rows and 36 station rows"
        )
    records = []
    for backend in BACKEND_ORDER:
        for scene_id in SCENE_ORDER:
            scene_match = scene.loc[
                (scene.backend == backend) & (scene.scene_id == scene_id)
            ]
            if len(scene_match) != 1:
                raise RuntimeError("FAIL CLOSED: ambiguous scene summary row")
            row = scene_match.iloc[0]
            if row.formal_summary_status != "FORMAL_SCENE_SUMMARY_DEFINED_COMPLETE_30_OF_30":
                raise RuntimeError("FAIL CLOSED: incomplete scene summary")
            stats = formal_stats(row.formal_statistics)
            record = {
                "backend": backend,
                "backend_label": BACKEND_LABEL[backend],
                "scene_id": scene_id,
                "scene_label": SCENE_LABEL[scene_id],
                "geometry_group": SCENE_GROUP[scene_id],
                "endpoint": endpoint,
                "scene_median": stats["median"] * value_scale,
                "scene_q25": stats["q25"] * value_scale,
                "scene_q75": stats["q75"] * value_scale,
                "scene_q95": stats["q95"] * value_scale,
            }
            for station_id in STATION_ORDER:
                station_match = station.loc[
                    (station.backend == backend)
                    & (station.scene_id == scene_id)
                    & (station.station_id == station_id)
                ]
                if len(station_match) != 1:
                    raise RuntimeError("FAIL CLOSED: ambiguous station summary row")
                station_row = station_match.iloc[0]
                if station_row.formal_summary_status != "FORMAL_STATION_SUMMARY_DEFINED_COMPLETE_10_OF_10":
                    raise RuntimeError("FAIL CLOSED: incomplete station summary")
                record[f"{station_id}_median"] = (
                    formal_stats(station_row.formal_statistics)["median"] * value_scale
                )
            records.append(record)
    output = backend_scene_sort(pd.DataFrame(records))
    output = add_source_columns(output, [scene_path, station_path], hashes)
    write_csv(output, output_filename)
    return output


def build_fig12_data(
    translation: pd.DataFrame, hashes: dict[str, str], analysis: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    permutation_path = RESULT_ROOT / "translation_exact_permutations.csv"
    summary_path = RESULT_ROOT / "analysis_summary.json"
    permutations = read_csv(permutation_path)
    require_columns(
        permutations,
        [
            "assigned_rich_scene_ids",
            "assigned_weak_scene_ids",
            "backend",
            "greater_than_or_equal_observed",
            "observed_allocation",
            "statistic",
        ],
        permutation_path,
    )
    if len(permutations) != 40:
        raise RuntimeError("FAIL CLOSED: translation exact permutations must have 40 rows")
    inference = {row["backend"]: row for row in analysis["primary_translation_inference"]}
    scene_data = translation[
        [
            "backend",
            "backend_label",
            "scene_id",
            "scene_label",
            "geometry_group",
            "scene_median",
        ]
    ].rename(columns={"scene_median": "translation_scene_median_mm"})
    scene_data["group_median_marker"] = [
        scene_id == GROUP_MEDIAN_SCENE[group]
        for scene_id, group in zip(scene_data.scene_id, scene_data.geometry_group)
    ]
    scene_data["observed_weak_minus_rich_mm"] = scene_data.backend.map(
        {backend: inference[backend]["estimand"] * 1000.0 for backend in BACKEND_ORDER}
    )
    scene_data["exact_one_sided_p"] = scene_data.backend.map(
        {backend: inference[backend]["p_value"] for backend in BACKEND_ORDER}
    )
    for backend in BACKEND_ORDER:
        subset = scene_data.loc[scene_data.backend == backend]
        rich_value = float(
            subset.loc[subset.scene_id == GROUP_MEDIAN_SCENE["Rich"], "translation_scene_median_mm"].iloc[0]
        )
        weak_value = float(
            subset.loc[subset.scene_id == GROUP_MEDIAN_SCENE["Weak"], "translation_scene_median_mm"].iloc[0]
        )
        frozen_observed = inference[backend]["estimand"] * 1000.0
        if not math.isclose(weak_value - rich_value, frozen_observed, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("FAIL CLOSED: displayed group median markers do not reproduce frozen observed statistic")
        if inference[backend]["p_value"] != 0.7:
            raise RuntimeError("FAIL CLOSED: frozen translation exact p-value is not 0.7")
    source_path, source_sha = source_annotation(
        [RESULT_ROOT / "translation_scene_summaries.csv", summary_path], hashes
    )
    scene_data["source_path"] = source_path
    scene_data["source_sha256"] = source_sha
    write_csv(scene_data, "Fig12_scene_data.csv")

    permutation_data = permutations.copy()
    permutation_data["allocation_index"] = permutation_data.groupby(
        "backend", sort=False
    ).cumcount() + 1
    permutation_data["statistic_mm"] = permutation_data.statistic * 1000.0
    permutation_data["observed_statistic_mm"] = permutation_data.backend.map(
        {backend: inference[backend]["estimand"] * 1000.0 for backend in BACKEND_ORDER}
    )
    permutation_data["exact_one_sided_p"] = permutation_data.backend.map(
        {backend: inference[backend]["p_value"] for backend in BACKEND_ORDER}
    )
    permutation_data = permutation_data[
        [
            "backend",
            "allocation_index",
            "assigned_rich_scene_ids",
            "assigned_weak_scene_ids",
            "observed_allocation",
            "greater_than_or_equal_observed",
            "statistic_mm",
            "observed_statistic_mm",
            "exact_one_sided_p",
        ]
    ]
    permutation_data = add_source_columns(
        permutation_data, [permutation_path, summary_path], hashes
    )
    write_csv(permutation_data, "Fig12_permutation_data.csv")
    return scene_data, permutation_data


def build_fig13_data(hashes: dict[str, str], analysis: dict):
    scene_order_path = RESULT_ROOT / "cross_backend_scene_pairs.csv"
    station_path = RESULT_ROOT / "cross_backend_station_pairs.csv"
    cosine_path = RESULT_ROOT / "snapshot_direction_cosines.csv"
    summary_path = RESULT_ROOT / "analysis_summary.json"

    ordering = read_csv(scene_order_path)
    require_columns(
        ordering, ["classification", "open3d_sign", "pcl_sign", "scene_i", "scene_j"], scene_order_path
    )
    if len(ordering) != 15 or set(ordering.classification) != {"CONCORDANT_NON_TIE"}:
        raise RuntimeError("FAIL CLOSED: scene ordering pairs do not match frozen complete agreement")

    pairs = analysis["cross_backend_scene_spearman"]["pairs"]
    if len(pairs) != 6:
        raise RuntimeError("FAIL CLOSED: frozen scene value pairs must have 6 rows")
    scene_rows = []
    for pair in pairs:
        scene_id = pair["identifier"]
        scene_rows.append(
            {
                "scene_id": scene_id,
                "scene_label": SCENE_LABEL[scene_id],
                "geometry_group": SCENE_GROUP[scene_id],
                "open3d_translation_zpru_mm": pair["open3d_value"] * 1000.0,
                "pcl_translation_zpru_mm": pair["pcl_value"] * 1000.0,
                "frozen_scene_spearman_rho": analysis["cross_backend_scene_spearman"]["rho"],
            }
        )
    scene_data = scene_sort(pd.DataFrame(scene_rows))
    scene_data = add_source_columns(scene_data, [scene_order_path, summary_path], hashes)
    write_csv(scene_data, "Fig13_scene_pairs.csv")

    station = read_csv(station_path)
    require_columns(station, ["complete", "identifier", "open3d_value", "pcl_value"], station_path)
    if len(station) != 18 or not station.complete.all():
        raise RuntimeError("FAIL CLOSED: station cross-backend pairs incomplete")
    station_rows = []
    for row in station.itertuples(index=False):
        scene_id, station_id = row.identifier.rsplit("_", 1)
        if scene_id not in SCENE_ORDER or station_id not in STATION_ORDER:
            raise RuntimeError("FAIL CLOSED: station pair identifier is ambiguous")
        station_rows.append(
            {
                "identifier": row.identifier,
                "scene_id": scene_id,
                "scene_label": SCENE_LABEL[scene_id],
                "station_id": station_id,
                "geometry_group": SCENE_GROUP[scene_id],
                "open3d_translation_zpru_mm": row.open3d_value * 1000.0,
                "pcl_translation_zpru_mm": row.pcl_value * 1000.0,
                "frozen_station_spearman_rho": analysis["cross_backend_station_spearman"]["rho"],
            }
        )
    station_data = backend_scene_sort(
        pd.DataFrame(station_rows).assign(backend=BACKEND_ORDER[0]), ["station_id"]
    ).drop(columns="backend")
    station_data = add_source_columns(station_data, [station_path, summary_path], hashes)
    write_csv(station_data, "Fig13_station_pairs.csv")

    cosine = read_csv(cosine_path)
    require_columns(cosine, ["cosine", "snapshot_id", "status"], cosine_path)
    if len(cosine) != 180 or set(cosine.status) != {"DIRECTION_COSINE_DEFINED"}:
        raise RuntimeError("FAIL CLOSED: direction cosine rows/status mismatch")
    cosine_data = add_source_columns(cosine, [cosine_path, summary_path], hashes)
    write_csv(cosine_data, "Fig13_direction_cosines.csv")
    return scene_data, station_data, cosine_data


def build_fig14_data(hashes: dict[str, str], analysis: dict):
    scene_path = RESULT_ROOT / "reassociation_scene_summaries.csv"
    centered_path = RESULT_ROOT / "reassociation_centered_rows.csv"
    permutation_summary_path = RESULT_ROOT / "reassociation_centered_permutation_summary.json"
    draws_path = RESULT_ROOT / "reassociation_centered_permutation_draws.csv"
    summary_path = RESULT_ROOT / "analysis_summary.json"

    scene_source = read_csv(scene_path)
    require_columns(
        scene_source,
        ["backend", "finite_metric_n", "formal_turnover_field", "median", "scene_id", "status"],
        scene_path,
    )
    if len(scene_source) != 12:
        raise RuntimeError("FAIL CLOSED: reassociation scene summaries must have 12 rows")
    association = {row["backend"]: row for row in analysis["reassociation_scene_association"]}
    scene_rows = []
    for backend in BACKEND_ORDER:
        pairs = association[backend]["pairs"]
        if len(pairs) != 6:
            raise RuntimeError("FAIL CLOSED: reassociation scene association must have 6 pairs")
        for pair in pairs:
            scene_id = pair["identifier"]
            source_match = scene_source.loc[
                (scene_source.backend == backend) & (scene_source.scene_id == scene_id)
            ]
            if len(source_match) != 1:
                raise RuntimeError("FAIL CLOSED: reassociation scene source mismatch")
            turnover = float(source_match.iloc[0]["median"])
            if not math.isclose(turnover, pair["open3d_value"], rel_tol=0.0, abs_tol=1e-15):
                raise RuntimeError("FAIL CLOSED: reassociation turnover pair mismatch")
            scene_rows.append(
                {
                    "backend": backend,
                    "backend_label": BACKEND_LABEL[backend],
                    "scene_id": scene_id,
                    "scene_label": SCENE_LABEL[scene_id],
                    "geometry_group": SCENE_GROUP[scene_id],
                    "scene_median_correspondence_turnover": turnover,
                    "scene_median_translation_zpru_mm": pair["pcl_value"] * 1000.0,
                    "frozen_spearman_rho": association[backend]["rho"],
                    "descriptive_spearman_p": association[backend]["p_value"],
                }
            )
    scene_data = backend_scene_sort(pd.DataFrame(scene_rows))
    scene_data = add_source_columns(scene_data, [scene_path, summary_path], hashes)
    write_csv(scene_data, "Fig14_scene_data.csv")

    centered = read_csv(centered_path)
    require_columns(
        centered,
        [
            "backend",
            "centered_correspondence_turnover",
            "centered_translation_norm_m",
            "scene_id",
            "scene_row_index",
        ],
        centered_path,
    )
    counts = centered.backend.value_counts().to_dict()
    if counts != {BACKEND_ORDER[0]: 180, BACKEND_ORDER[1]: 180}:
        raise RuntimeError("FAIL CLOSED: centered reassociation must have 180 rows/backend")
    centered_summary = {row["backend"]: row for row in analysis["reassociation_centered_association"]}
    permutation_summary = {
        row["backend"]: row for row in read_json(permutation_summary_path)
    }
    centered_data = centered.copy()
    centered_data["scene_label"] = centered_data.scene_id.map(SCENE_LABEL)
    centered_data["geometry_group"] = centered_data.scene_id.map(SCENE_GROUP)
    centered_data["centered_translation_zpru_mm"] = centered_data.centered_translation_norm_m * 1000.0
    centered_data["frozen_centered_rho"] = centered_data.backend.map(
        {backend: centered_summary[backend]["rho"] for backend in BACKEND_ORDER}
    )
    centered_data["frozen_permutation_sensitivity_p"] = centered_data.backend.map(
        {backend: permutation_summary[backend]["p_value"] for backend in BACKEND_ORDER}
    )
    centered_data = centered_data.drop(columns="centered_translation_norm_m")
    centered_data = backend_scene_sort(centered_data, ["scene_row_index"])
    centered_data = add_source_columns(
        centered_data, [centered_path, permutation_summary_path, summary_path], hashes
    )
    write_csv(centered_data, "Fig14_centered_data.csv")

    draws = read_csv(draws_path)
    require_columns(
        draws,
        ["abs_rho_permuted", "backend", "draw_index", "greater_than_or_equal_abs_observed", "rho_permuted"],
        draws_path,
    )
    draw_counts = draws.backend.value_counts().to_dict()
    if draw_counts != {BACKEND_ORDER[0]: 10000, BACKEND_ORDER[1]: 10000}:
        raise RuntimeError("FAIL CLOSED: frozen reassociation draws must have 10,000/backend")
    draws_data = add_source_columns(draws, [draws_path, permutation_summary_path], hashes)
    write_csv(draws_data, "Fig14_permutation_draws.csv")
    return scene_data, centered_data, draws_data, permutation_summary


def build_fig15_data(hashes: dict[str, str], analysis: dict):
    scene_path = RESULT_ROOT / "systematic_scene_values.csv"
    station_path = RESULT_ROOT / "systematic_station_values.csv"
    summary_path = RESULT_ROOT / "analysis_summary.json"
    scene = read_csv(scene_path)
    station = read_csv(station_path)
    require_columns(
        scene, ["backend", "defined_station_n", "scene_id", "scene_systematic_fraction", "status"], scene_path
    )
    require_columns(
        station,
        ["backend", "scene_id", "station_id", "status", "systematic_fraction"],
        station_path,
    )
    if len(scene) != 12 or len(station) != 36:
        raise RuntimeError("FAIL CLOSED: systematic scene/station row counts mismatch")
    descriptive = {row["backend"]: row for row in analysis["systematic_weak_rich_descriptive"]}
    scene_data = scene.copy()
    scene_data["scene_label"] = scene_data.scene_id.map(SCENE_LABEL)
    scene_data["geometry_group"] = scene_data.scene_id.map(SCENE_GROUP)
    scene_data["group_median"] = [
        descriptive[backend]["rich_median" if group == "Rich" else "weak_median"]
        for backend, group in zip(scene_data.backend, scene_data.geometry_group)
    ]
    scene_data["weak_minus_rich_difference"] = scene_data.backend.map(
        {
            backend: descriptive[backend]["weak_minus_rich_median_difference"]
            for backend in BACKEND_ORDER
        }
    )
    scene_data = backend_scene_sort(scene_data)
    scene_data = add_source_columns(scene_data, [scene_path, summary_path], hashes)
    write_csv(scene_data, "Fig15_scene_data.csv")

    station_data = station.copy()
    station_data["scene_label"] = station_data.scene_id.map(SCENE_LABEL)
    station_data["geometry_group"] = station_data.scene_id.map(SCENE_GROUP)
    station_data = backend_scene_sort(station_data, ["station_id"])
    station_data = add_source_columns(station_data, [station_path], hashes)
    write_csv(station_data, "Fig15_station_data.csv")
    return scene_data, station_data


def build_figs01_permutation_data(hashes: dict[str, str], analysis: dict) -> pd.DataFrame:
    permutation_path = RESULT_ROOT / "rotation_exact_permutations.csv"
    summary_path = RESULT_ROOT / "analysis_summary.json"
    frame = read_csv(permutation_path)
    require_columns(
        frame,
        [
            "assigned_rich_scene_ids",
            "assigned_weak_scene_ids",
            "backend",
            "endpoint",
            "greater_than_or_equal_observed",
            "observed_allocation",
            "statistic",
        ],
        permutation_path,
    )
    frame = frame.loc[frame.endpoint == "rotation_angle_deg"].copy()
    if len(frame) != 40:
        raise RuntimeError("FAIL CLOSED: degree rotation exact allocations must have 40 rows")
    inference = {
        row["backend"]: row
        for row in analysis["secondary_rotation_inference"]
        if row["endpoint"] == "rotation_angle_deg"
    }
    frame["allocation_index"] = frame.groupby("backend", sort=False).cumcount() + 1
    frame["observed_statistic_deg"] = frame.backend.map(
        {backend: inference[backend]["estimand"] for backend in BACKEND_ORDER}
    )
    frame["exact_one_sided_p"] = frame.backend.map(
        {backend: inference[backend]["p_value"] for backend in BACKEND_ORDER}
    )
    frame = add_source_columns(frame, [permutation_path, summary_path], hashes)
    write_csv(frame, "FigS01_permutation_data.csv")
    return frame


def plot_fig05(data: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.55))
    specs = [
        ("normalized_lambda_min_trans", "scene_median_normalized_lambda_min_trans", "Normalized minimum\ntranslational eigenvalue", (0.0, 0.32)),
        ("condition_number_trans", "scene_median_condition_number_trans", "Translational condition number", (0.0, 22.0)),
        ("spectral_entropy_trans", "scene_median_spectral_entropy_trans", "Spectral entropy", (0.0, 1.01)),
    ]
    for index, (ax, spec) in enumerate(zip(axes, specs)):
        value_col, scene_col, ylabel, ylim = spec
        add_scene_background(ax)
        for scene_index, scene_id in enumerate(SCENE_ORDER):
            subset = data.loc[data.scene_id == scene_id].sort_values(
                ["station_id", "selection_index"], kind="stable"
            )
            offsets = np.linspace(-0.18, 0.18, len(subset))
            ax.scatter(
                scene_index + offsets,
                subset[value_col],
                s=8,
                facecolor="white",
                edgecolor=MID_GREY,
                linewidth=0.35,
                zorder=2,
            )
            ax.scatter(
                scene_index,
                subset[scene_col].iloc[0],
                s=34,
                marker="D",
                facecolor=INK,
                edgecolor="white",
                linewidth=0.55,
                zorder=4,
            )
        set_scene_ticks(ax)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        style_axis(ax)
        panel_label(ax, f"({chr(97 + index)})")
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.18, top=0.82, wspace=0.34)
    return save_figure(fig, "Fig05_geometry_characterization")


def plot_interval_figure(
    data: pd.DataFrame,
    basename: str,
    ylabel: str,
    ylim: tuple[float, float],
    secondary_annotations: dict[str, str] | None = None,
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.85), sharey=True)
    for index, (ax, backend) in enumerate(zip(axes, BACKEND_ORDER)):
        style = BACKEND_STYLE[backend]
        subset = data.loc[data.backend == backend]
        add_scene_background(ax)
        for scene_index, scene_id in enumerate(SCENE_ORDER):
            row = subset.loc[subset.scene_id == scene_id].iloc[0]
            ax.vlines(scene_index, row.scene_q25, row.scene_q95, color=style["color"], linewidth=0.8, zorder=2)
            ax.vlines(scene_index, row.scene_q25, row.scene_q75, color=style["color"], linewidth=3.0, zorder=3)
            ax.scatter(scene_index, row.scene_median, s=34, marker=style["marker"], color=style["color"], edgecolor="white", linewidth=0.55, zorder=5)
            for station_offset, station_id in zip([-0.13, 0.0, 0.13], STATION_ORDER):
                ax.scatter(
                    scene_index + station_offset,
                    row[f"{station_id}_median"],
                    s=15,
                    marker=style["marker"],
                    facecolor="white",
                    edgecolor=style["color"],
                    linewidth=0.7,
                    zorder=4,
                )
        set_scene_ticks(ax)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel if index == 0 else "")
        ax.text(0.98, 0.94, BACKEND_LABEL[backend], transform=ax.transAxes, ha="right", va="top", fontweight="bold", color=style["color"])
        if secondary_annotations:
            ax.text(
                0.03,
                0.94,
                secondary_annotations[backend],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
            )
        style_axis(ax)
        panel_label(ax, f"({chr(97 + index)})")
    fig.legend(
        handles=summary_legend_handles(BACKEND_ORDER[0]),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.16, top=0.75, wspace=0.16)
    return save_figure(fig, basename)


def plot_fig12(scene_data: pd.DataFrame, permutation_data: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.45))
    panel_map = [
        (axes[0, 0], BACKEND_ORDER[0], "scene", "(a)"),
        (axes[0, 1], BACKEND_ORDER[0], "permutation", "(b)"),
        (axes[1, 0], BACKEND_ORDER[1], "scene", "(c)"),
        (axes[1, 1], BACKEND_ORDER[1], "permutation", "(d)"),
    ]
    for ax, backend, kind, label in panel_map:
        style = BACKEND_STYLE[backend]
        if kind == "scene":
            subset = scene_data.loc[scene_data.backend == backend]
            ax.axvspan(-0.5, 0.5, color=RICH_BG, zorder=-20)
            ax.axvspan(0.5, 1.5, color=WEAK_BG, zorder=-20)
            for group_index, group in enumerate(["Rich", "Weak"]):
                group_rows = subset.loc[subset.geometry_group == group]
                for offset, row in zip([-0.18, 0.0, 0.18], group_rows.itertuples(index=False)):
                    face = style["color"] if group == "Rich" else "white"
                    ax.scatter(group_index + offset, row.translation_scene_median_mm, s=34, marker=style["marker"], facecolor=face, edgecolor=style["color"], linewidth=0.8, zorder=3)
                    text_offset = (0, -13) if row.scene_id == "FMB1_W01" else (0, 7)
                    ax.annotate(
                        row.scene_label,
                        (group_index + offset, row.translation_scene_median_mm),
                        xytext=text_offset,
                        textcoords="offset points",
                        ha="center",
                        va="top" if text_offset[1] < 0 else "bottom",
                        fontsize=6.7,
                    )
                median_row = group_rows.loc[group_rows.group_median_marker].iloc[0]
                ax.hlines(median_row.translation_scene_median_mm, group_index - 0.28, group_index + 0.28, color=INK, linewidth=1.6, zorder=2)
            ax.set_xticks([0, 1], ["Rich", "Weak"])
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(0.0, 0.82)
            ax.set_ylabel("Translation ZPRU (mm)")
            ax.text(0.98, 0.94, BACKEND_LABEL[backend], transform=ax.transAxes, ha="right", va="top", fontweight="bold", color=style["color"])
            ax.text(0.03, 0.05, "Horizontal bar: median of\nthree scene medians", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color=INK)
            style_axis(ax)
        else:
            subset = permutation_data.loc[permutation_data.backend == backend]
            ax.scatter(subset.statistic_mm, subset.allocation_index, s=19, marker=style["marker"], facecolor=style["color"], edgecolor="white", linewidth=0.4, zorder=3)
            observed = float(subset.observed_statistic_mm.iloc[0])
            p_value = float(subset.exact_one_sided_p.iloc[0])
            ax.axvline(observed, color=INK, linewidth=1.15, zorder=2)
            ax.text(observed, 20.7, "Observed statistic", ha="center", va="bottom", fontsize=6.8)
            ax.text(0.97, 0.08, f"Exact one-sided p = {p_value:.1f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.3)
            ax.set_xlim(-0.22, 0.22)
            ax.set_ylim(0, 22)
            ax.set_yticks([1, 5, 10, 15, 20])
            ax.set_ylabel("Exact allocation index")
            ax.set_xlabel("Weak − Rich translation difference (mm)")
            style_axis(ax, grid_axis="both")
        panel_label(ax, label)
    fig.subplots_adjust(left=0.09, right=0.992, bottom=0.105, top=0.96, wspace=0.31, hspace=0.34)
    return save_figure(fig, "Fig12_rich_weak_exact_permutation")


def plot_agreement_panel(ax, data: pd.DataFrame, limit: float, rho: float, label: str, show_labels: bool) -> None:
    for group in ["Rich", "Weak"]:
        subset = data.loc[data.geometry_group == group]
        ax.scatter(
            subset.open3d_translation_zpru_mm,
            subset.pcl_translation_zpru_mm,
            s=32 if show_labels else 22,
            marker="o",
            facecolor=INK if group == "Rich" else "white",
            edgecolor=INK,
            linewidth=0.75,
            label=f"{group} scenes",
            zorder=3,
        )
        if show_labels:
            for row in subset.itertuples(index=False):
                ax.annotate(
                    row.scene_label,
                    (row.open3d_translation_zpru_mm, row.pcl_translation_zpru_mm),
                    xytext=POINT_LABEL_OFFSETS[row.scene_id],
                    textcoords="offset points",
                    fontsize=6.6,
                )
    ax.plot([0, limit], [0, limit], color=MID_GREY, linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Open3D translation ZPRU (mm)")
    ax.set_ylabel("PCL translation ZPRU (mm)")
    ax.text(0.04, 0.94, f"Spearman ρ = {rho:.4f}" if rho != 1.0 else "Spearman ρ = 1.000", transform=ax.transAxes, ha="left", va="top")
    style_axis(ax, grid_axis="both")
    panel_label(ax, label)


def plot_fig13(scene_data: pd.DataFrame, station_data: pd.DataFrame, cosine_data: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.75))
    plot_agreement_panel(axes[0], scene_data, 0.8, float(scene_data.frozen_scene_spearman_rho.iloc[0]), "(a)", True)
    plot_agreement_panel(axes[1], station_data, 1.05, float(station_data.frozen_station_spearman_rho.iloc[0]), "(b)", False)
    cosine_sorted = np.sort(cosine_data.cosine.to_numpy())
    ecdf = np.arange(1, len(cosine_sorted) + 1) / len(cosine_sorted)
    axes[2].step(cosine_sorted, ecdf, where="post", color=INK, linewidth=1.2)
    axes[2].axvline(TASK_SPECIFIED_DIRECTION_COSINE_MEDIAN, color="#0072B2", linewidth=1.0, linestyle="--")
    axes[2].text(0.04, 0.94, "Frozen median = 0.997739005\ndefined = 180\nzero-vector = 0\nmissing/nonfinite = 0", transform=axes[2].transAxes, ha="left", va="top", fontsize=7.0)
    axes[2].set_xlim(-1.0, 1.01)
    axes[2].set_ylim(0.0, 1.01)
    axes[2].set_xlabel("Snapshot direction cosine")
    axes[2].set_ylabel("ECDF")
    style_axis(axes[2], grid_axis="both")
    panel_label(axes[2], "(c)")
    axes[0].legend(loc="lower right", frameon=False, handletextpad=0.4, borderaxespad=0.2)
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.18, top=0.91, wspace=0.43)
    return save_figure(fig, "Fig13_cross_backend_agreement")


def plot_fig14(
    scene_data: pd.DataFrame,
    centered_data: pd.DataFrame,
    draws_data: pd.DataFrame,
    permutation_summary: dict,
) -> dict[str, str]:
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.45))
    for index, backend in enumerate(BACKEND_ORDER):
        style = BACKEND_STYLE[backend]
        scene_ax = axes[0, index]
        scene_subset = scene_data.loc[scene_data.backend == backend]
        for group in ["Rich", "Weak"]:
            subset = scene_subset.loc[scene_subset.geometry_group == group]
            scene_ax.scatter(
                subset.scene_median_correspondence_turnover,
                subset.scene_median_translation_zpru_mm,
                s=36,
                marker=style["marker"],
                facecolor=style["color"] if group == "Rich" else "white",
                edgecolor=style["color"],
                linewidth=0.8,
                label=group,
                zorder=3,
            )
            for row in subset.itertuples(index=False):
                scene_ax.annotate(
                    row.scene_label,
                    (row.scene_median_correspondence_turnover, row.scene_median_translation_zpru_mm),
                    xytext=POINT_LABEL_OFFSETS[row.scene_id],
                    textcoords="offset points",
                    fontsize=6.6,
                )
        rho = float(scene_subset.frozen_spearman_rho.iloc[0])
        p_value = float(scene_subset.descriptive_spearman_p.iloc[0])
        scene_ax.text(0.04, 0.94, f"ρ = {rho:.9f}\ndescriptive Spearman p = {p_value:.4g}", transform=scene_ax.transAxes, ha="left", va="top", fontsize=7.0)
        scene_ax.text(0.98, 0.94, BACKEND_LABEL[backend], transform=scene_ax.transAxes, ha="right", va="top", color=style["color"], fontweight="bold")
        scene_ax.set_xlim(0.0, 0.065)
        scene_ax.set_ylim(0.0, 0.8)
        scene_ax.set_xlabel("Scene-median correspondence turnover")
        scene_ax.set_ylabel("Scene-median translation ZPRU (mm)" if index == 0 else "")
        style_axis(scene_ax, grid_axis="both")
        panel_label(scene_ax, "(a)" if index == 0 else "(b)")
        if index == 0:
            scene_ax.legend(loc="lower right", frameon=False)

        centered_ax = axes[1, index]
        centered_subset = centered_data.loc[centered_data.backend == backend]
        for group in ["Rich", "Weak"]:
            subset = centered_subset.loc[centered_subset.geometry_group == group]
            centered_ax.scatter(
                subset.centered_correspondence_turnover,
                subset.centered_translation_zpru_mm,
                s=12,
                marker=style["marker"],
                facecolor=style["color"] if group == "Rich" else "white",
                edgecolor=style["color"],
                linewidth=0.45,
                alpha=0.55,
                zorder=2,
            )
        centered_ax.axhline(0, color=LIGHT_GREY, linewidth=0.7)
        centered_ax.axvline(0, color=LIGHT_GREY, linewidth=0.7)
        rho = float(centered_subset.frozen_centered_rho.iloc[0])
        centered_ax.text(0.04, 0.94, f"ρ = {rho:.9f}\np_sensitivity = 1/10001", transform=centered_ax.transAxes, ha="left", va="top", fontsize=7.0)
        centered_ax.set_xlim(-0.05, 0.23)
        centered_ax.set_ylim(-0.6, 1.8)
        centered_ax.set_xlabel("Scene-median-centered correspondence turnover")
        centered_ax.set_ylabel("Scene-median-centered translation ZPRU (mm)" if index == 0 else "")
        style_axis(centered_ax, grid_axis="both")
        panel_label(centered_ax, "(c)" if index == 0 else "(d)")

        inset = centered_ax.inset_axes([0.60, 0.12, 0.36, 0.29])
        draws = np.sort(draws_data.loc[draws_data.backend == backend, "abs_rho_permuted"].to_numpy())
        inset_ecdf = np.arange(1, len(draws) + 1) / len(draws)
        inset.step(draws, inset_ecdf, where="post", color=MID_GREY, linewidth=0.8)
        inset.axvline(abs(permutation_summary[backend]["observed_statistic"]), color=style["color"], linewidth=1.0)
        inset.set_xlim(0, 0.8)
        inset.set_ylim(0, 1.01)
        inset.set_xlabel("|ρ|", fontsize=6)
        inset.set_ylabel("ECDF", fontsize=6)
        inset.tick_params(labelsize=5.5, width=0.5, length=2)
        inset.grid(True, color=GRID_GREY, linewidth=0.4)
    fig.subplots_adjust(left=0.095, right=0.992, bottom=0.105, top=0.96, wspace=0.28, hspace=0.34)
    return save_figure(fig, "Fig14_reassociation_association")


def plot_fig15(scene_data: pd.DataFrame, station_data: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.85), sharey=True)
    for index, (ax, backend) in enumerate(zip(axes, BACKEND_ORDER)):
        style = BACKEND_STYLE[backend]
        scene_subset = scene_data.loc[scene_data.backend == backend]
        station_subset = station_data.loc[station_data.backend == backend]
        add_scene_background(ax)
        for scene_index, scene_id in enumerate(SCENE_ORDER):
            stations = station_subset.loc[station_subset.scene_id == scene_id]
            for offset, row in zip([-0.13, 0.0, 0.13], stations.itertuples(index=False)):
                ax.scatter(scene_index + offset, row.systematic_fraction, s=16, marker=style["marker"], facecolor="white", edgecolor=style["color"], linewidth=0.7, zorder=3)
            scene_row = scene_subset.loc[scene_subset.scene_id == scene_id].iloc[0]
            ax.scatter(scene_index, scene_row.scene_systematic_fraction, s=35, marker=style["marker"], facecolor=style["color"], edgecolor="white", linewidth=0.55, zorder=4)
        rich_median = float(scene_subset.loc[scene_subset.geometry_group == "Rich", "group_median"].iloc[0])
        weak_median = float(scene_subset.loc[scene_subset.geometry_group == "Weak", "group_median"].iloc[0])
        ax.hlines(rich_median, -0.35, 2.35, color=INK, linewidth=1.3)
        ax.hlines(weak_median, 2.65, 5.35, color=INK, linewidth=1.3)
        difference = float(scene_subset.weak_minus_rich_difference.iloc[0])
        ax.text(0.03, 0.06, f"Weak − Rich = {difference:.9f}", transform=ax.transAxes, ha="left", va="bottom")
        ax.text(0.98, 0.94, BACKEND_LABEL[backend], transform=ax.transAxes, ha="right", va="top", fontweight="bold", color=style["color"])
        set_scene_ticks(ax)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Systematic fraction" if index == 0 else "")
        style_axis(ax)
        panel_label(ax, "(a)" if index == 0 else "(b)")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BACKEND_STYLE[BACKEND_ORDER[0]]["color"], markeredgecolor="white", markersize=5.5, label="Scene fraction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=BACKEND_STYLE[BACKEND_ORDER[0]]["color"], markersize=4.3, label="Station fraction"),
        Line2D([0], [0], color=INK, linewidth=1.3, label="Rich/Weak group median"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.16, top=0.77, wspace=0.16)
    return save_figure(fig, "Fig15_systematic_component")


def write_captions() -> None:
    en = """# Figure captions (English)

## Fig. 5. Rich/Weak geometry characterization

Panels show (a) normalized minimum translational eigenvalue, (b) translational condition number, and (c) spectral entropy for the six prespecified scenes. Small open points are the 30 frozen snapshot values per scene; diamonds are the frozen scene medians from the final scene registry. The panels document the prespecified coarse geometry classification used in the formal dataset. W02 is acquisition attempt 2. These descriptive geometry panels do not imply that Weak geometry necessarily produces a larger zero-perturbation registration update (ZPRU).

## Fig. 11. Translation ZPRU across six scenes

(a) Open3D and (b) PCL translation ZPRU. Thin intervals show frozen q25–q95 summaries, thick intervals show q25–q75 summaries, large filled markers show scene medians, and small open markers show the three nested station medians. Values are converted from metres to millimetres for display. Stations and snapshots are nested repeated observations; scene is the highest-level independent unit. The intervals are summary intervals, not Tukey boxplots, and the figure does not assert an ordering between Rich and Weak scenes.

## Fig. 12. Rich/Weak scene comparison and exact permutation distribution

(a, c) The six frozen scene medians are shown separately for Open3D and PCL; horizontal bars identify the median of the three scene medians in each prespecified group. (b, d) All 20 frozen exact allocation statistics are shown with the frozen observed statistic. Open3D: Weak − Rich = −0.026347782 mm, exact one-sided p = 0.7. PCL: Weak − Rich = −0.010915495 mm, exact one-sided p = 0.7. Scene is the highest-level independent unit. No significance threshold was prespecified; the exact one-sided p-value is reported descriptively.

## Fig. 13. Cross-backend agreement

(a) Six scene-level translation ZPRU pairs, (b) 18 station-level pairs, and (c) the ECDF of 180 snapshot-level update-direction cosines. Identity lines in (a, b) are absolute-agreement references; Spearman rho describes rank agreement and does not establish equality of magnitudes. The frozen scene and station rho values are 1.000 and 0.9917, respectively. In (c), the frozen median cosine is 0.997739005; all 180 values are defined, with zero zero-vector and zero missing/nonfinite cases. The two separately implemented backends used identical input point clouds and initialization conditions; this evaluates implementation-level agreement, not repeatability across independent measurement systems. Scene remains the highest-level independent unit.

## Fig. 14. Association with correspondence reassociation

(a, b) Scene-median correspondence turnover versus scene-median translation ZPRU for six scenes per backend, with frozen descriptive Spearman rho and p. (c, d) The 180 frozen within-scene median-centered rows per backend; insets show the ECDFs of the existing 10,000 stratified permutation |rho| draws and the frozen observed |rho|. No regression line or causal model is fitted. Both turnover and update magnitude depend on the estimated terminal pose; the observed relationship is an association and does not establish correspondence reassociation as an independent causal determinant. Scene is the highest-level independent unit, and the 180 centered rows are nested observations rather than independent scenes.

## Fig. 15. Systematic component

(a) Open3D and (b) PCL systematic fractions. Small open markers show three station values per scene, large filled markers show frozen scene values, and horizontal bars show frozen Rich/Weak group medians. Frozen Weak − Rich descriptive differences are −0.154058605 and −0.301378567, respectively. Secondary descriptive mechanistic comparison; no formal p-value was specified. Scene is the highest-level independent unit.

## Fig. S1. Secondary rotational ZPRU

(a) Open3D and (b) PCL rotational ZPRU in degrees. Glyphs match Fig. 11: frozen q25–q95 and q25–q75 summary intervals, scene medians, and three nested station medians. Open3D: Weak − Rich = −0.003238022 deg, exact p = 0.9. PCL: Weak − Rich = −0.001570443 deg, exact p = 0.8. This is a secondary endpoint and is not co-primary. Scene is the highest-level independent unit.
"""
    zh = """# 图注（中文）

## Fig. 5. Rich/Weak 几何特征

各 panel 分别展示六个预设场景的 (a) 归一化最小平移特征值、(b) 平移条件数和 (c) 谱熵。小空心点为每个场景 30 个冻结 snapshot 值，菱形为 final scene registry 中冻结的场景中位数。本图记录正式数据集中预设的粗粒度几何分类；W02 为 acquisition attempt 2。本图是描述性几何展示，不表示 Weak geometry 必然对应更大的 zero-perturbation registration update（ZPRU）。

## Fig. 11. 六场景 Translation ZPRU

(a) Open3D，(b) PCL。细区间为冻结 q25–q95，粗区间为 q25–q75，大实心点为场景中位数，小空心点为三个嵌套 station 的中位数；数值仅为显示而由 m 换算为 mm。station 和 snapshot 是嵌套重复观测，scene 始终是最高层级独立单位。这些是 summary intervals，不是 Tukey boxplot；本图不宣称 Rich/Weak 的方向性排序。

## Fig. 12. Rich/Weak 场景比较与 exact permutation

(a, c) 分别显示 Open3D 和 PCL 的六个冻结场景中位数，横线标出各预设组中三个场景中位数的中位数。(b, d) 显示全部 20 个冻结 exact allocation statistics 与冻结 observed statistic。Open3D：Weak − Rich = −0.026347782 mm，exact one-sided p = 0.7；PCL：Weak − Rich = −0.010915495 mm，exact one-sided p = 0.7。scene 是最高层级独立单位。未预设显著性阈值，exact one-sided p-value 仅作描述性报告。

## Fig. 13. Cross-backend agreement

(a) 6 个 scene-level translation ZPRU 配对，(b) 18 个 station-level 配对，(c) 180 个 snapshot-level update-direction cosine 的 ECDF。(a, b) 的 identity line 仅是绝对一致性参考；Spearman rho 描述秩一致性，不能证明数值幅度相等。冻结的 scene/station rho 分别为 1.000 和 0.9917。(c) 的冻结中位数为 0.997739005；180 个值均已定义，zero-vector 与 missing/nonfinite 均为 0。两个独立实现的 backend 使用完全相同的输入点云和初始化条件，因此本图评估 implementation-level agreement，不评估独立测量系统之间的重复性。scene 仍是最高层级独立单位。

## Fig. 14. 与 correspondence reassociation 的关联

(a, b) 每个 backend 的 6 个 scene-median correspondence turnover 与 scene-median translation ZPRU，并标注冻结的描述性 Spearman rho 和 p。(c, d) 每个 backend 的 180 个冻结 within-scene median-centered rows；inset 展示既有 10,000 个分层 permutation |rho| draws 的 ECDF 和冻结 observed |rho|。图中不拟合回归线或因果模型。turnover 与 update magnitude 都依赖估计的 terminal pose；观察到的关系是 association，不能把 correspondence reassociation 确立为独立因果决定因素。scene 是最高层级独立单位，180 行是嵌套观测而不是独立场景。

## Fig. 15. Systematic component

(a) Open3D，(b) PCL。小空心点为每个 scene 的 3 个 station fraction，大实心点为冻结 scene fraction，横线为冻结 Rich/Weak 组中位数。冻结的 Weak − Rich 描述性差值分别为 −0.154058605 和 −0.301378567。该图为 secondary descriptive mechanistic comparison；未指定 formal p-value。scene 是最高层级独立单位。

## Fig. S1. Secondary rotational ZPRU

(a) Open3D，(b) PCL，单位为 deg。glyph 与 Fig. 11 一致：冻结 q25–q95、q25–q75 summary intervals、scene median 和 3 个嵌套 station medians。Open3D：Weak − Rich = −0.003238022 deg，exact p = 0.9；PCL：Weak − Rich = −0.001570443 deg，exact p = 0.8。这是 secondary endpoint，不提升为 co-primary。scene 是最高层级独立单位。
"""
    (FIG_ROOT / "captions/figure_captions_en.md").write_text(en, encoding="utf-8")
    (FIG_ROOT / "captions/figure_captions_zh.md").write_text(zh, encoding="utf-8")


def write_readme(checksum_audit: dict[str, object]) -> None:
    correction = read_json(CORRECTION_ROOT / "analysis_summary_reporting_corrected_v1.json")
    corrected = correction["reporting_correction"]["corrected_accounting"]
    if corrected["formal_completed_n"] != 360:
        raise RuntimeError("FAIL CLOSED: corrected formal completed count is not 360")
    if corrected["pcl_native_has_converged_true_n"] != 180:
        raise RuntimeError("FAIL CLOSED: corrected PCL native convergence count is not 180")
    if corrected["open3d_native_convergence_observable"]:
        raise RuntimeError("FAIL CLOSED: Open3D native convergence is unexpectedly observable")
    readme = f"""# ZPRM FMB1 frozen paper data figures v1

This directory contains seven paper-candidate data figures generated only from frozen scientific analysis and final geometry sources. The figure-generation base commit is `{STARTING_HEAD}`; the self commit is resolved by annotated tag `{FIGURE_TAG}` after commit creation. Scientific source tags are `{RESULT_TAG}`, `{CORRECTION_TAG}`, and `{DIAGNOSTIC_TAG}`.

No registration or ICP backend was run. No scientific analysis, p-value, Spearman rho, exact allocation, or 10,000-draw permutation was rerun. No new hypothesis test was created, no outlier or counter-directional observation was removed, and no frozen result was modified. Scene is the highest-level independent unit; stations and snapshots are nested repeated observations.

ZPRU means zero-perturbation registration update, where zero perturbation means zero intentionally imposed initialization perturbation. It must not be interpreted as a physical displacement measure. Reassociation is reported only as an association. Cross-backend panels report implementation-level agreement under identical inputs and initialization conditions.

Solver/accounting wording comes only from reporting correction v1:

- 360/360 formal COMPLETED.
- PCL native `has_converged = 180/180`.
- Open3D native stopping criterion is not retrospectively observable.

The frozen checksum roots passed before plotting: `{checksum_audit['all_pass']}`. Run:

```bash
python3 paper/figures/zprm_fmb1_v1/scripts/generate_all_figures.py
python3 paper/figures/zprm_fmb1_v1/scripts/validate_paper_figures.py
```

Outputs are written as vector PDF/SVG and 600 dpi PNG/TIFF (LZW). The actual font is `{FONT_FAMILY}` with PDF Type 42 settings. `audit/source_file_manifest.json`, `audit/source_schema_audit.json`, `audit/figure_provenance_manifest.json`, and the directory-level `SHA256SUMS` provide the audit trail.
"""
    (FIG_ROOT / "README.md").write_text(readme, encoding="utf-8")


def write_no_registration_audit() -> None:
    forbidden_imports = {
        "open3d",
        "phase_a_harness.open3d_backend",
        "phase_a_harness.pcl_backend",
    }
    imported = []
    forbidden_calls = []
    for path in sorted((FIG_ROOT / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append({"file": path.name, "module": alias.name})
            elif isinstance(node, ast.ImportFrom):
                imported.append({"file": path.name, "module": node.module or ""})
            elif isinstance(node, ast.Call):
                name = ""
                current = node.func
                while isinstance(current, ast.Attribute):
                    name = f".{current.attr}{name}"
                    current = current.value
                if isinstance(current, ast.Name):
                    name = f"{current.id}{name}"
                lowered = name.lower()
                if any(token in lowered for token in ["run_icp", "registration_icp", "formal_runner", "pcl_backend", "open3d_backend"]):
                    forbidden_calls.append({"file": path.name, "call": name, "line": node.lineno})
    forbidden_import_hits = [entry for entry in imported if entry["module"] in forbidden_imports]
    result = {
        "schema": "zprm_fmb1_no_registration_import_audit_v1",
        "files_scanned": sorted(path.name for path in (FIG_ROOT / "scripts").glob("*.py")),
        "imports": imported,
        "forbidden_import_hits": forbidden_import_hits,
        "forbidden_backend_calls": forbidden_calls,
        "REGISTRATION_BACKEND_CALL_COUNT": len(forbidden_import_hits) + len(forbidden_calls),
    }
    if result["REGISTRATION_BACKEND_CALL_COUNT"] != 0:
        raise RuntimeError("Registration/backend import or call found in figure scripts")
    write_json(FIG_ROOT / "audit/no_registration_import_audit.json", result)


def write_provenance(outputs: dict[str, dict[str, str]], hashes: dict[str, str]) -> None:
    figure_sources = {
        "Fig05_geometry_characterization": ["final_geometry_manifest.csv", "final_scene_registry.yaml", "final_dataset_readiness.json"],
        "Fig11_six_scene_translation_zpru": ["translation_scene_summaries.csv", "translation_station_summaries.csv"],
        "Fig12_rich_weak_exact_permutation": ["translation_scene_summaries.csv", "translation_exact_permutations.csv", "analysis_summary.json"],
        "Fig13_cross_backend_agreement": ["cross_backend_scene_pairs.csv", "cross_backend_station_pairs.csv", "snapshot_direction_cosines.csv", "analysis_summary.json"],
        "Fig14_reassociation_association": ["reassociation_scene_summaries.csv", "reassociation_centered_rows.csv", "reassociation_centered_permutation_summary.json", "reassociation_centered_permutation_draws.csv", "analysis_summary.json"],
        "Fig15_systematic_component": ["systematic_scene_values.csv", "systematic_station_values.csv", "analysis_summary.json"],
        "FigS01_rotation_secondary": ["rotation_scene_summaries.csv", "rotation_station_summaries.csv", "rotation_exact_permutations.csv", "analysis_summary.json"],
    }
    panels = {
        "Fig05_geometry_characterization": ["(a) normalized minimum translational eigenvalue", "(b) translational condition number", "(c) spectral entropy"],
        "Fig11_six_scene_translation_zpru": ["(a) Open3D", "(b) PCL"],
        "Fig12_rich_weak_exact_permutation": ["(a) Open3D scene medians", "(b) Open3D exact allocations", "(c) PCL scene medians", "(d) PCL exact allocations"],
        "Fig13_cross_backend_agreement": ["(a) scene pairs", "(b) station pairs", "(c) direction-cosine ECDF"],
        "Fig14_reassociation_association": ["(a) Open3D scene association", "(b) PCL scene association", "(c) Open3D centered rows and frozen draws", "(d) PCL centered rows and frozen draws"],
        "Fig15_systematic_component": ["(a) Open3D", "(b) PCL"],
        "FigS01_rotation_secondary": ["(a) Open3D secondary endpoint", "(b) PCL secondary endpoint"],
    }
    source_records = {
        relative(path): {"sha256": hashes[relative(path)], "tag": tag, "role": role}
        for path, role, tag in SOURCE_SPECS
    }
    provenance = {
        "schema": "zprm_fmb1_figure_provenance_manifest_v1",
        "generation_base_commit": STARTING_HEAD,
        "figure_commit": f"SELF_RESOLVED_BY_ANNOTATED_TAG:{FIGURE_TAG}",
        "source_tags": [RESULT_TAG, CORRECTION_TAG, DIAGNOSTIC_TAG],
        "scene_order": SCENE_ORDER,
        "backend_order": BACKEND_ORDER,
        "w02_attempt": 2,
        "scene_is_highest_independent_unit": True,
        "font_family": FONT_FAMILY,
        "pdf_fonttype": 42,
        "raster_dpi": 600,
        "tiff_compression": "LZW",
        "counts": {
            "REGISTRATION_BACKEND_CALL_COUNT": 0,
            "NEW_HYPOTHESIS_TEST_COUNT": 0,
            "FORMAL_STATISTIC_RECOMPUTATION_COUNT": 0,
            "FROZEN_RESULT_MODIFICATION_COUNT": 0,
        },
        "figures": {
            name: {
                "panels": panels[name],
                "source_basenames": figure_sources[name],
                "outputs": outputs[name],
            }
            for name in outputs
        },
        "sources": source_records,
    }
    write_json(FIG_ROOT / "audit/figure_provenance_manifest.json", provenance)


def main() -> None:
    ensure_directories()
    configure_matplotlib()
    checksum_audit = verify_frozen_roots()
    hashes = source_hash_map()
    build_source_audits(hashes, checksum_audit)
    analysis = read_json(RESULT_ROOT / "analysis_summary.json")
    if analysis["dataset_accounting"]["scene_is_highest_independent_unit"] is not True:
        raise RuntimeError("FAIL CLOSED: scene is not the highest independent unit")

    fig05_data = build_fig05_data(hashes)
    fig11_data = extract_interval_data(
        RESULT_ROOT / "translation_scene_summaries.csv",
        RESULT_ROOT / "translation_station_summaries.csv",
        "translation_norm_m",
        1000.0,
        hashes,
        "Fig11_data.csv",
    )
    fig12_scene, fig12_permutation = build_fig12_data(fig11_data, hashes, analysis)
    fig13_scene, fig13_station, fig13_cosine = build_fig13_data(hashes, analysis)
    fig14_scene, fig14_centered, fig14_draws, permutation_summary = build_fig14_data(hashes, analysis)
    fig15_scene, fig15_station = build_fig15_data(hashes, analysis)
    figs01_scene = extract_interval_data(
        RESULT_ROOT / "rotation_scene_summaries.csv",
        RESULT_ROOT / "rotation_station_summaries.csv",
        "rotation_angle_deg",
        1.0,
        hashes,
        "FigS01_scene_data.csv",
    )
    figs01_station = read_csv(RESULT_ROOT / "rotation_station_summaries.csv")
    figs01_station = figs01_station.loc[figs01_station.endpoint == "rotation_angle_deg"].copy()
    rows = []
    for row in figs01_station.itertuples(index=False):
        rows.append(
            {
                "backend": row.backend,
                "scene_id": row.scene_id,
                "scene_label": SCENE_LABEL[row.scene_id],
                "geometry_group": SCENE_GROUP[row.scene_id],
                "station_id": row.station_id,
                "endpoint": row.endpoint,
                "station_median_deg": formal_stats(row.formal_statistics)["median"],
                "formal_summary_status": row.formal_summary_status,
            }
        )
    figs01_station_data = backend_scene_sort(pd.DataFrame(rows), ["station_id"])
    figs01_station_data = add_source_columns(
        figs01_station_data, [RESULT_ROOT / "rotation_station_summaries.csv"], hashes
    )
    write_csv(figs01_station_data, "FigS01_station_data.csv")
    build_figs01_permutation_data(hashes, analysis)

    rotation_inference = {
        row["backend"]: row
        for row in analysis["secondary_rotation_inference"]
        if row["endpoint"] == "rotation_angle_deg"
    }
    rotation_annotations = {
        backend: f"Secondary endpoint\nWeak − Rich = {rotation_inference[backend]['estimand']:.9f} deg\nexact p = {rotation_inference[backend]['p_value']:.1f}"
        for backend in BACKEND_ORDER
    }

    outputs = {
        "Fig05_geometry_characterization": plot_fig05(fig05_data),
        "Fig11_six_scene_translation_zpru": plot_interval_figure(
            fig11_data,
            "Fig11_six_scene_translation_zpru",
            "Translation ZPRU (mm)",
            (0.0, 1.6),
        ),
        "Fig12_rich_weak_exact_permutation": plot_fig12(fig12_scene, fig12_permutation),
        "Fig13_cross_backend_agreement": plot_fig13(fig13_scene, fig13_station, fig13_cosine),
        "Fig14_reassociation_association": plot_fig14(
            fig14_scene, fig14_centered, fig14_draws, permutation_summary
        ),
        "Fig15_systematic_component": plot_fig15(fig15_scene, fig15_station),
        "FigS01_rotation_secondary": plot_interval_figure(
            figs01_scene,
            "FigS01_rotation_secondary",
            "Rotational ZPRU (deg)",
            (0.0, 0.07),
            rotation_annotations,
        ),
    }

    write_captions()
    write_readme(checksum_audit)
    write_no_registration_audit()
    write_provenance(outputs, hashes)
    write_directory_checksums()

    for figure in ["FIG05", "FIG11", "FIG12", "FIG13", "FIG14", "FIG15", "FIGS01"]:
        print(f"{figure}_GENERATED=true")
    print("REGISTRATION_BACKEND_CALL_COUNT=0")
    print("NEW_HYPOTHESIS_TEST_COUNT=0")
    print("FORMAL_STATISTIC_RECOMPUTATION_COUNT=0")
    print("FROZEN_RESULT_MODIFICATION_COUNT=0")


if __name__ == "__main__":
    main()
