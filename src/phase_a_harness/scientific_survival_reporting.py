"""Deterministic shortlist, claim wording, plots, and reports for survival audit."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#4C78A8"
AMBER = "#D29B3D"
INK = "#263238"
GREY = "#A7ADB4"


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] = ()) -> None:
    values = list(rows)
    fieldnames = list(fields) or sorted({key for row in values for key in row})
    if not fieldnames:
        raise ValueError(f"CSV requires columns: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
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


def _tercile(value: float, boundaries: tuple[float, float]) -> str:
    if value <= boundaries[0]:
        return "LOW"
    if value <= boundaries[1]:
        return "MID"
    return "HIGH"


def deterministic_counterexample_shortlist(
    candidates: Sequence[Mapping[str, Any]], *, per_backend_limit: int = 12
) -> list[dict[str, Any]]:
    """Greedily cover frozen audit strata while capping snapshot reuse at two."""

    output: list[dict[str, Any]] = []
    backends = sorted({str(row["backend"]) for row in candidates})
    for backend in backends:
        rows = [dict(row) for row in candidates if str(row["backend"]) == backend]
        if not rows:
            continue
        error_bounds = tuple(
            float(value)
            for value in np.quantile(
                [float(row["error_ratio"]) for row in rows],
                [1.0 / 3.0, 2.0 / 3.0],
                method="linear",
            )
        )
        turnover_bounds = tuple(
            float(value)
            for value in np.quantile(
                [float(row["turnover_difference"]) for row in rows],
                [1.0 / 3.0, 2.0 / 3.0],
                method="linear",
            )
        )
        prepared: list[dict[str, Any]] = []
        for row in rows:
            row["scene_pair"] = " :: ".join(sorted((str(row["scene_a"]), str(row["scene_b"]))))
            row["error_ratio_stratum"] = _tercile(float(row["error_ratio"]), error_bounds)
            row["turnover_difference_stratum"] = _tercile(
                float(row["turnover_difference"]), turnover_bounds
            )
            row["original_manual_review_status"] = row.get("manual_review_status")
            row["manual_review_status"] = "PENDING_HUMAN_REVIEW"
            prepared.append(row)
        prepared.sort(key=lambda row: str(row["candidate_pair_id"]))
        selected: list[dict[str, Any]] = []
        snapshot_use: Counter[str] = Counter()
        covered_scene_pairs: set[str] = set()
        covered_conditions: set[str] = set()
        covered_error: set[str] = set()
        covered_turnover: set[str] = set()
        remaining = list(prepared)
        while remaining and len(selected) < per_backend_limit:
            feasible = [
                row
                for row in remaining
                if snapshot_use[str(row["snapshot_a"])] < 2
                and snapshot_use[str(row["snapshot_b"])] < 2
            ]
            if not feasible:
                break

            def score(row: Mapping[str, Any]) -> tuple[int, int, str]:
                coverage = 0
                if len(covered_scene_pairs) < 3 and row["scene_pair"] not in covered_scene_pairs:
                    coverage += 8
                if len(covered_conditions) < 3 and row["condition"] not in covered_conditions:
                    coverage += 8
                if row["error_ratio_stratum"] not in covered_error:
                    coverage += 6
                if row["turnover_difference_stratum"] not in covered_turnover:
                    coverage += 6
                novelty = sum(
                    (
                        row["scene_pair"] not in covered_scene_pairs,
                        row["condition"] not in covered_conditions,
                        row["error_ratio_stratum"] not in covered_error,
                        row["turnover_difference_stratum"] not in covered_turnover,
                    )
                )
                return coverage, novelty, str(row["candidate_pair_id"])

            chosen = max(feasible, key=score)
            remaining.remove(chosen)
            selected.append(chosen)
            snapshot_use[str(chosen["snapshot_a"])] += 1
            snapshot_use[str(chosen["snapshot_b"])] += 1
            covered_scene_pairs.add(str(chosen["scene_pair"]))
            covered_conditions.add(str(chosen["condition"]))
            covered_error.add(str(chosen["error_ratio_stratum"]))
            covered_turnover.add(str(chosen["turnover_difference_stratum"]))
            if (
                len(selected) >= 6
                and len(covered_scene_pairs) >= 3
                and len(covered_conditions) >= 3
                and covered_error == {"LOW", "MID", "HIGH"}
                and covered_turnover == {"LOW", "MID", "HIGH"}
            ):
                break
        if not (
            len(covered_scene_pairs) >= 3
            and len(covered_conditions) >= 3
            and covered_error == {"LOW", "MID", "HIGH"}
            and covered_turnover == {"LOW", "MID", "HIGH"}
            and max(snapshot_use.values(), default=0) <= 2
        ):
            raise ValueError(f"candidate shortlist coverage failed for {backend}")
        for rank, row in enumerate(selected, 1):
            row["review_priority_rank_within_backend"] = rank
            row["error_tercile_lower_boundary"] = error_bounds[0]
            row["error_tercile_upper_boundary"] = error_bounds[1]
            row["turnover_tercile_lower_boundary"] = turnover_bounds[0]
            row["turnover_tercile_upper_boundary"] = turnover_bounds[1]
            row["counterexample_claim_authorized"] = False
        output.extend(selected)
    if len(output) > 24:
        raise ValueError("counterexample shortlist exceeds 24 rows")
    return output


def scientific_claim_matrix(
    *,
    primary_scene_pass: bool,
    cross_backend_pass: bool,
    reassociation_pass: bool,
    model_leakage_pass: bool,
    model_robust_pass: bool,
    systematic_full_noise_pass: bool,
    independent_authorized_term: str,
) -> list[dict[str, Any]]:
    explanatory = bool(model_leakage_pass and model_robust_pass)
    independent_systematic = independent_authorized_term == "REPEATABLE_SYSTEMATIC_OFFSET"
    return [
        {
            "claim": "Scene-dependent zero-initialization registration displacement",
            "authorized": bool(primary_scene_pass),
            "authorized_scope": "Development synthetic Long Corridor versus Geometry Rich Room under INDEPENDENT_NOISE_FREE and FULL_NOISE",
            "required_wording": "scene-dependent zero-initialization registration displacement",
            "forbidden_wording": "universal environment-intrinsic failure",
            "supporting_analysis": "unique-unit scene effect",
            "remaining_limitation": "synthetic Development evidence with three geometry seeds",
        },
        {
            "claim": "Cross-backend reproducibility of scene ranking",
            "authorized": bool(cross_backend_pass),
            "authorized_scope": "two frozen point-to-plane backends and five non-IDEAL Development conditions",
            "required_wording": "cross-backend reproducibility of scene ranking",
            "forbidden_wording": "backend-independent universal ranking",
            "supporting_analysis": "unique-input-weighted Spearman",
            "remaining_limitation": "Open3D and PCL only",
        },
        {
            "claim": "Association turnover is statistically associated with displacement",
            "authorized": bool(reassociation_pass),
            "authorized_scope": "existing synthetic Development trials under leave-one-scene/condition stress tests",
            "required_wording": "association turnover is statistically associated with displacement",
            "forbidden_wording": "association turnover causes displacement",
            "supporting_analysis": "pooled, centered, LOSO, LOCO, and unique-input Spearman",
            "remaining_limitation": "observational post-registration association",
        },
        {
            "claim": "Reassociation diagnostics provide incremental explanatory value",
            "authorized": explanatory,
            "authorized_scope": "post-registration Development diagnostics under three frozen weighting schemes",
            "required_wording": "incremental explanatory value",
            "forbidden_wording": "pre-registration failure prediction",
            "supporting_analysis": "leakage audit and weighted geometry-fold Ridge sensitivity",
            "remaining_limitation": "Model B uses post-registration diagnostics",
        },
        {
            "claim": "Repeatable systematic offset under FULL_NOISE",
            "authorized": bool(systematic_full_noise_pass),
            "authorized_scope": "Long Corridor FULL_NOISE cells meeting effective-replicate and systematic-fraction gates",
            "required_wording": "repeatable systematic offset under FULL_NOISE",
            "forbidden_wording": "universal systematic bias",
            "supporting_analysis": "replicate-uniqueness-qualified systematic offset",
            "remaining_limitation": "synthetic Long Corridor and two backends",
        },
        {
            "claim": "Systematic bias under INDEPENDENT_NOISE_FREE",
            "authorized": independent_systematic,
            "authorized_scope": "none when each cell contains one unique source-target pair",
            "required_wording": independent_authorized_term,
            "forbidden_wording": "repeatable systematic bias from duplicated deterministic inputs",
            "supporting_analysis": "replicate uniqueness audit",
            "remaining_limitation": "repeated executions do not create independent inputs",
        },
        {
            "claim": "Causal effect of reassociation",
            "authorized": False,
            "authorized_scope": "not authorized",
            "required_wording": "association only; no causal claim",
            "forbidden_wording": "reassociation causes registration error",
            "supporting_analysis": "none capable of causal identification",
            "remaining_limitation": "no randomized intervention on reassociation",
        },
        {
            "claim": "Universal environment-intrinsic localizability",
            "authorized": False,
            "authorized_scope": "not authorized",
            "required_wording": "scene-dependent synthetic Development result",
            "forbidden_wording": "universal environment-intrinsic localizability",
            "supporting_analysis": "none beyond seven synthetic scenes",
            "remaining_limitation": "no real-data validation and limited synthetic scene family",
        },
    ]


def claim_wording_boundary_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    by_claim = {str(row["claim"]): row for row in rows}
    return bool(
        len(rows) == 8
        and len(by_claim) == 8
        and by_claim["Causal effect of reassociation"]["authorized"] is False
        and by_claim["Universal environment-intrinsic localizability"]["authorized"] is False
        and all(str(row.get("required_wording", "")).strip() for row in rows)
        and all(str(row.get("forbidden_wording", "")).strip() for row in rows)
        and all(str(row.get("remaining_limitation", "")).strip() for row in rows)
    )


def build_survival_figures(
    output_dir: Path,
    *,
    replicate_condition_rows: Sequence[Mapping[str, Any]],
    scene_effect_rows: Sequence[Mapping[str, Any]],
    turnover_rows: Sequence[Mapping[str, Any]],
    model_sensitivity_rows: Sequence[Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [str(row["condition"]) for row in replicate_condition_rows]
    medians = [
        float(
            row.get(
                "effective_replicate_count_median",
                row.get("effective_replicate_median"),
            )
        )
        for row in replicate_condition_rows
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(np.arange(len(conditions)), medians, color=BLUE)
    ax.set_xticks(np.arange(len(conditions)), [value.replace("_NOISE", "") for value in conditions], rotation=25, ha="right")
    ax.set_ylabel("median unique source-target pairs per cell")
    ax.set_title("Effective replicates by non-IDEAL condition")
    ax.grid(axis="y", color=GREY, alpha=.3)
    fig.tight_layout(); fig.savefig(output_dir / "effective_replicates_by_condition.png", dpi=160); plt.close(fig)

    labels = [f"{row['condition'][:4]}\n{row['backend']}" for row in scene_effect_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - .19, [float(row.get("original_weak_rich_median_ratio", row.get("original_trial_weighted_ratio"))) for row in scene_effect_rows], .38, color=GREY, label="Original trial-weighted")
    ax.bar(x + .19, [float(row.get("weak_rich_median_ratio", row.get("unique_unit_ratio"))) for row in scene_effect_rows], .38, color=BLUE, label="Unique input")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Long / rich median ratio")
    ax.set_title("Original versus unique-input scene effect")
    ax.grid(axis="y", color=GREY, alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "original_vs_unique_weighted_effect.png", dpi=160); plt.close(fig)

    loso = [
        row
        for row in turnover_rows
        if row.get("robustness_type") == "LEAVE_ONE_SCENE_OUT_CENTERED"
        or row.get("scope") == "LEAVE_ONE_SCENE_OUT"
    ]
    scenes = sorted(
        {str(row.get("omitted_level", row.get("omitted_group"))) for row in loso}
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for backend, marker, color in (("Open3D", "o", BLUE), ("PCL", "s", AMBER)):
        lookup = {
            str(row.get("omitted_level", row.get("omitted_group"))): float(row["spearman_rho"])
            for row in loso
            if row["backend"] == backend
        }
        ax.plot(np.arange(len(scenes)), [lookup[scene] for scene in scenes], marker=marker, color=color, label=backend)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(np.arange(len(scenes)), [scene.replace("END_FACE_TRANSITION_", "EFT_") for scene in scenes], rotation=25, ha="right")
    ax.set_ylabel("centered Spearman rho")
    ax.set_title("Turnover robustness: leave one scene out")
    ax.grid(color=GREY, alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "turnover_leave_one_scene_out.png", dpi=160); plt.close(fig)

    schemes = [
        "ORIGINAL_TRIAL_WEIGHTED",
        "UNIQUE_INPUT_WEIGHTED",
        "UNIQUE_INPUT_CONDITION_BALANCED",
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(schemes))
    for offset, backend, color in ((-.19, "Open3D", BLUE), (.19, "PCL", AMBER)):
        backend_schema = (
            "open3d_point_to_plane" if backend == "Open3D" else "pcl_point_to_plane"
        )
        lookup = {
            str(row["weighting_scheme"]): float(row["relative_improvement"])
            for row in model_sensitivity_rows
            if row.get("backend", row.get("backend_schema_name"))
            in {backend, backend_schema}
        }
        ax.bar(x + offset, [lookup[scheme] for scheme in schemes], .38, color=color, label=backend)
    ax.axhline(.10, color=INK, linestyle="--", linewidth=1)
    ax.set_xticks(x, ["Trial", "Unique input", "Condition balanced"])
    ax.set_ylabel("relative MAE improvement")
    ax.set_title("Model B sensitivity across frozen weighting schemes")
    ax.grid(axis="y", color=GREY, alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "model_weighting_sensitivity.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"Claim {index}" for index in range(1, len(claim_rows) + 1)]
    values = [1 if row["authorized"] else 0 for row in claim_rows]
    ax.barh(np.arange(len(labels)), values, color=[BLUE if value else GREY for value in values])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlim(0, 1.05); ax.set_xticks((0, 1), ("Not authorized", "Authorized"))
    ax.set_title("Scientific claim authorization matrix")
    ax.grid(axis="x", color=GREY, alpha=.3)
    fig.tight_layout(); fig.savefig(output_dir / "claim_authorization_matrix.png", dpi=160); plt.close(fig)


__all__ = [
    "build_survival_figures",
    "claim_wording_boundary_pass",
    "deterministic_counterexample_shortlist",
    "scientific_claim_matrix",
    "write_csv",
]
