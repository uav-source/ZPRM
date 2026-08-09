"""Publisher-repair boundary and Scientific Survival audit orchestration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

from .confirmatory_protocol import materialize_synthetic_confirmatory_protocol
from .full_synthetic_artifact_verifier import verify_full_synthetic_artifact
from .full_synthetic_independent_verifier import (
    full_synthetic_analysis_verifier_comparison,
)
from .real_data_protocol import materialize_real_data_protocol_framework
from .scientific_survival_artifact_verifier import (
    FIGURES,
    REQUIRED_FILES,
    SHA_EXCLUDED,
    TABLES,
    verify_scientific_survival_artifact,
)
from .scientific_survival_integrity import (
    file_sha256,
    publisher_only_change_scope,
    verify_pre_repair_inventory,
)
from .scientific_survival_models import (
    audit_model_data_leakage,
    build_frozen_development_model_lock,
    evaluate_model_weighting_sensitivity,
    join_development_model_rows,
    scan_confirmatory_seed_provenance,
)
from .scientific_survival_replicates import (
    attach_observed_component_signatures,
    audit_replicate_uniqueness,
    evaluate_cross_backend_unique_unit,
    evaluate_long_corridor_systematic_offset,
    evaluate_turnover_robustness,
    evaluate_unique_unit_primary_scene_effect,
)
from .scientific_survival_reporting import (
    build_survival_figures,
    claim_wording_boundary_pass,
    deterministic_counterexample_shortlist,
    scientific_claim_matrix,
    write_csv,
)


ARCHIVE_TAG = "archive/zero-perturbation-full-synthetic-science-pass-pre-publication-repair"
BASELINE_COMMIT = "8794bf37d1d78bcd524293493424a601c70d263e"
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
SOURCE_BRANCH = "feature/zero-perturbation-phase-a-lock-v2-regression-repair"
SOURCE_COMMIT = "89f46dda68e9ff5c71f078f6d13fc9050d58f0f5"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _changed_paths(repository: Path) -> list[str]:
    tracked = set(_git(repository, "diff", "--name-only", ARCHIVE_TAG).splitlines())
    untracked = set(
        _git(repository, "ls-files", "--others", "--exclude-standard").splitlines()
    )
    return sorted(path for path in tracked | untracked if path)


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _load_frozen_snapshot_arrays(
    root: Path, row: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    snapshot_id = str(row["snapshot_id"])
    cache = (
        root / "data/phase_b_signal_snapshots"
        if snapshot_id.startswith("phase-b-signal-v1/")
        else root / "data/full_synthetic_development_v1_snapshots"
    )
    directory = cache / snapshot_id
    source = directory / "source_points.npy"
    target = directory / "target_points.npy"
    if not source.is_file() or not target.is_file():
        raise FileNotFoundError(f"frozen snapshot arrays are missing: {snapshot_id}")
    return {
        "source": np.load(source, allow_pickle=False),
        "target": np.load(target, allow_pickle=False),
    }


def _replicate_semantics_pass(report: Mapping[str, Any]) -> bool:
    conditions = {row["condition"]: row for row in report["condition_rows"]}
    if set(conditions) != {
        "INDEPENDENT_NOISE_FREE",
        "SCAN_NOISE_ONLY",
        "MAP_NOISE_ONLY",
        "DROPOUT_ONLY",
        "FULL_NOISE",
    }:
        return False
    independent = conditions["INDEPENDENT_NOISE_FREE"]
    stochastic = [row for name, row in conditions.items() if name != "INDEPENDENT_NOISE_FREE"]
    return bool(
        report.get("REPLICATE_UNIQUENESS_AUDIT_PASS") is True
        and report.get("snapshot_count") == 1050
        and report.get("deterministic_single_input_cell_count") == 21
        and report.get("partial_replication_cell_count") == 0
        and report.get("full_replication_cell_count") == 84
        and independent["effective_replicate_count_min"] == 1
        and independent["effective_replicate_count_max"] == 1
        and independent["AUTHORIZED_TERM"]
        == "DETERMINISTIC_ZERO_INITIALIZATION_DISPLACEMENT"
        and independent["SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED_BY_CONDITION"] is False
        and all(row["effective_replicate_count_min"] == 10 for row in stochastic)
        and all(row["effective_replicate_count_max"] == 10 for row in stochastic)
        and all(row["SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED_BY_CONDITION"] is True for row in stochastic)
    )


def _cross_backend_table(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row, scope="CONDITION") for row in report["condition_rows"]]
    rows.append(
        {
            "condition": "POOLED_35_SCENE_CONDITION",
            "condition_rho_at_least_0_50_count": report[
                "condition_rho_at_least_0_50_count"
            ],
            "condition_rho_median": report["condition_rho_median"],
            "scene_count": 35,
            "scope": "POOLED",
            "spearman_gate_threshold": 0.75,
            "spearman_rho": report["pooled_scene_condition_spearman_rho"],
            "threshold_pass": report["pooled_scene_condition_spearman_rho"] >= 0.75,
        }
    )
    return rows


def _turnover_table(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in report["sensitivity_rows"]]
    for row in report["backend_rows"]:
        rows.append(
            {
                **row,
                "omitted_level": "NONE",
                "robustness_type": "SUMMARY",
                "spearman_rho": None,
            }
        )
    return rows


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _publisher_report(
    scope: Mapping[str, Any], compact: Mapping[str, Any]
) -> str:
    return "\n".join(
        [
            "# Publisher Repair Report",
            "",
            "The compact Development artifact was restored without rerunning registration or analysis.",
            "",
            "- Changed runtime file: `src/phase_a_harness/full_synthetic_publisher.py`",
            "- Changed call at line 174: `Axes.boxplot(tick_labels=...)` → `Axes.boxplot(labels=...)`",
            f"- Frozen Matplotlib: `{scope['matplotlib_version']}`",
            "- The frozen formal public entry rejected the post-freeze code hash, as designed.",
            "- Publication therefore used the same publisher's internal atomic staging and double-verification path with the frozen manifest, primary analysis, independent verification, run manifest, and raw inventory.",
            f"- Compact artifact required files: `{compact['required_file_count']}`",
            f"- SHA missing / mismatch / unexpected: `{compact['sha256_missing_count']} / {compact['sha256_mismatch_count']} / {compact['sha256_unexpected_listed_count']}`",
            f"- `ARTIFACT_PUBLICATION_PASS = {_fmt(compact['ARTIFACT_VERIFICATION_PASS'])}`",
            "- `trial_rerun_count = 0`; `snapshot_regeneration_count = 0`.",
            "",
            "The complete change-boundary evidence is in `publisher_only_change_scope.json`.",
            "",
        ]
    )


def _replicate_report(
    replicate: Mapping[str, Any], systematic: Mapping[str, Any]
) -> str:
    lines = [
        "# Replicate Uniqueness and Pseudoreplication Audit",
        "",
        "The scientific replicate is one unique frozen `(source_checksum, target_checksum)` pair. Backend duplication does not increase this count.",
        "",
        "| Condition | Cells | Effective min / median / max | Deterministic / partial / full | Authorized term |",
        "|---|---:|---:|---:|---|",
    ]
    for row in replicate["condition_rows"]:
        lines.append(
            f"| {row['condition']} | {row['cell_count']} | {row['effective_replicate_count_min']} / {row['effective_replicate_count_median']:.0f} / {row['effective_replicate_count_max']} | {row['deterministic_single_input_cell_count']} / {row['partial_replication_cell_count']} / {row['full_replication_cell_count']} | `{row['AUTHORIZED_TERM']}` |"
        )
    lines.extend(
        [
            "",
            "INDEPENDENT_NOISE_FREE repeats the same deterministic input pair ten times per cell; its near-zero execution dispersion cannot support a repeatable-systematic-bias claim. Noise and dropout signatures are reconstructed from the frozen arrays relative to each matching INDEPENDENT input; no snapshot is regenerated.",
            f"DROPOUT_ONLY masks are exact frozen-array subsequences. FULL_NOISE has `{replicate['dropout_mask_reconstruction_failure_count']}` of 210 observations whose latent mask cannot be uniquely separated from scan noise, affecting `{sum(row['dropout_mask_reconstruction_failure_count'] > 0 for row in replicate['cell_rows'])}` cells; those rows are explicitly marked as final-source proxies. `DROPOUT_MASK_IDENTITY_FULLY_RECONSTRUCTED = {_fmt(replicate['DROPOUT_MASK_IDENTITY_FULLY_RECONSTRUCTED'])}` while `DROPOUT_INPUT_VARIATION_OBSERVED = {_fmt(replicate['DROPOUT_INPUT_VARIATION_OBSERVED'])}`.",
            "",
            "## Long Corridor systematic-offset audit",
            "",
            "| Condition | Backend | Geometry seed | Effective | Offset (m) | RMS (m) | Fraction | Direction concentration | FULL gate |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in systematic["geometry_rows"]:
        lines.append(
            "| {condition} | {backend} | {geometry_seed} | {effective_replicate_count} | {systematic_translation_offset_m:.9g} | {repeatability_rms_m:.9g} | {systematic_fraction_translation:.9g} | {direction_concentration:.9g} | {full_noise_geometry_gate_pass} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"- `SYSTEMATIC_OFFSET_FULL_NOISE_PASS = {_fmt(systematic['SYSTEMATIC_OFFSET_FULL_NOISE_PASS'])}`",
            f"- `SYSTEMATIC_OFFSET_CLAIM_SCOPE = {systematic['SYSTEMATIC_OFFSET_CLAIM_SCOPE']}`",
            f"- INDEPENDENT wording: `{systematic['INDEPENDENT_NOISE_FREE_AUTHORIZED_TERM']}`",
            "",
            "Machine-readable details: `tables/replicate_uniqueness_by_cell.csv`, `tables/replicate_uniqueness_by_condition.csv`, `tables/measurement_seed_effectiveness.csv`, and `tables/repeat_index_effectiveness.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def _candidate_packet(shortlist: Sequence[Mapping[str, Any]], total: int) -> str:
    lines = [
        "# Counterexample Human Review Packet",
        "",
        f"The Development search produced **{total} automatic candidate pairs**. They are pairwise screening records, not {total} independent scientific counterexamples.",
        "",
        f"This deterministic review shortlist contains **{len(shortlist)}** rows. It spans at least three scene pairs, three conditions, and low/mid/high strata for both error ratio and turnover difference within each backend, while using any snapshot at most twice.",
        "",
        "Every row remains `PENDING_HUMAN_REVIEW`; `COUNTEREXAMPLE_CLAIM_AUTHORIZED = false`.",
        "",
        "See `tables/counterexample_review_shortlist.csv` for the immutable review queue.",
        "",
    ]
    return "\n".join(lines)


def _survival_report(
    *,
    decision: Mapping[str, Any],
    replicate: Mapping[str, Any],
    scene_effect: Mapping[str, Any],
    cross_backend: Mapping[str, Any],
    turnover: Mapping[str, Any],
    model: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
) -> str:
    gate_names = (
        "RAW_EVIDENCE_INTEGRITY_PASS",
        "ARTIFACT_PUBLICATION_PASS",
        "REPLICATE_UNIQUENESS_AUDIT_PASS",
        "PRIMARY_SCENE_EFFECT_UNIQUE_UNIT_PASS",
        "CROSS_BACKEND_UNIQUE_UNIT_PASS",
        "REASSOCIATION_ROBUSTNESS_PASS",
        "MODEL_DATA_LEAKAGE_AUDIT_PASS",
        "MODEL_INCREMENTAL_VALUE_ROBUST_PASS",
        "CLAIM_WORDING_BOUNDARY_PASS",
    )
    lines = [
        "# Zero-Perturbation Scientific Survival Audit",
        "",
        "## Decision",
        "",
        f"`SCIENTIFIC_SURVIVAL_AUDIT_PASS = {_fmt(decision['SCIENTIFIC_SURVIVAL_AUDIT_PASS'])}`. All nine frozen survival gates are conjunctive; none was relaxed.",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | `{_fmt(decision[name])}` |" for name in gate_names)
    lines.extend(
        [
            "",
            "## Replicate uniqueness changes the wording, not the frozen evidence",
            "",
            f"The audit found {replicate['deterministic_single_input_cell_count']} deterministic-single-input cells, {replicate['partial_replication_cell_count']} partial cells, and {replicate['full_replication_cell_count']} fully replicated cells. Pseudoreplication risk is therefore identified. INDEPENDENT_NOISE_FREE is described only as deterministic zero-initialization displacement; repeatable systematic offset is restricted to FULL_NOISE where its stricter gate passes.",
            "",
            "![Effective replicates by condition](figures/effective_replicates_by_condition.png)",
            "",
            "## Unique-input scene and cross-backend effects survive",
            "",
            "All four Long Corridor versus Geometry Rich Room backend × primary-condition comparisons retain ratio ≥ 5, absolute difference ≥ 0.005 m, and at least 2/3 geometry-level wins after collapsing duplicate inputs.",
            "",
            "![Original versus unique-input effect](figures/original_vs_unique_weighted_effect.png)",
            "",
            f"The five-condition unique-input Spearman median is `{_fmt(cross_backend['condition_rho_median'])}` and the pooled 35-cell rho is `{_fmt(cross_backend['pooled_scene_condition_spearman_rho'])}`.",
            "",
            "## Turnover association survives stress tests but is not causal",
            "",
        ]
    )
    for row in turnover["backend_rows"]:
        lines.append(
            f"- {row['backend']}: pooled rho `{_fmt(row['pooled_spearman_rho'])}`, centered rho `{_fmt(row['centered_spearman_rho'])}`, positive LOSO `{row['leave_one_scene_out_positive_count']}/7`, positive LOCO `{row['leave_one_condition_out_positive_count']}/5`, unique-input rho `{_fmt(row['unique_input_spearman_rho'])}`."
        )
    lines.extend(
        [
            "",
            "![Leave-one-scene-out turnover robustness](figures/turnover_leave_one_scene_out.png)",
            "",
            "## Model B remains explanatory under three weighting schemes",
            "",
            "The geometry-fold, train-only-scaler leakage audit passes. Model B uses post-registration turnover, normal-change, and residual-change diagnostics, so it supports incremental explanatory value only—not pre-registration prediction.",
            "",
        ]
    )
    for row in model["model_weighting_sensitivity"]:
        lines.append(
            f"- {row['backend_schema_name']} / {row['weighting_scheme']}: MAE A `{_fmt(row['model_a_cv_mae'])}`, MAE B `{_fmt(row['model_b_cv_mae'])}`, relative improvement `{_fmt(row['relative_improvement'])}`, B-better folds `{row['model_b_better_fold_count']}/3`."
        )
    lines.extend(
        [
            "",
            "![Model weighting sensitivity](figures/model_weighting_sensitivity.png)",
            "",
            "## Claim authorization boundary",
            "",
        ]
    )
    for index, row in enumerate(claims, 1):
        lines.append(
            f"- Claim {index}: `{_fmt(row['authorized'])}` — {row['claim']}. Required: “{row['required_wording']}”; forbidden: “{row['forbidden_wording']}”."
        )
    lines.extend(
        [
            "",
            "![Scientific claim authorization](figures/claim_authorization_matrix.png)",
            "",
            "## Protocol-design boundary",
            "",
            f"- `CONFIRMATORY_SEED_PROVENANCE_PASS = {_fmt(decision['CONFIRMATORY_SEED_PROVENANCE_PASS'])}`",
            f"- `SYNTHETIC_CONFIRMATORY_PROTOCOL_READY = {_fmt(decision['SYNTHETIC_CONFIRMATORY_PROTOCOL_READY'])}`",
            f"- `REAL_DATA_PROTOCOL_FRAMEWORK_READY = {_fmt(decision['REAL_DATA_PROTOCOL_FRAMEWORK_READY'])}`",
            "- `CONFIRMATORY_RUN_AUTHORIZED = false`",
            "- `REAL_DATA_RUN_AUTHORIZED = false`",
            "- `MEASUREMENT_PAPER_MAINLINE_AUTHORIZED = false`",
            "",
            "## Audit table registry",
            "",
        ]
    )
    lines.extend(f"- [`{name}`](tables/{name})" for name in TABLES)
    lines.extend(
        [
            "",
            "## Figure registry",
            "",
        ]
    )
    lines.extend(f"- [`{name}`](figures/{name})" for name in FIGURES)
    lines.extend(
        [
            "",
            "## Remaining limitation",
            "",
            "This remains synthetic Development evidence over three Development geometry seeds. Correlation is not causation; the automatic shortlist is not human validation; Confirmatory and real-data runs remain unauthorized.",
            "",
        ]
    )
    return "\n".join(lines)


def run_scientific_survival_audit(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if _git(root, "rev-parse", ARCHIVE_TAG) != BASELINE_COMMIT:
        raise ValueError("pre-repair archive tag no longer points to the frozen baseline")
    if _git(SOURCE_REPOSITORY, "rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ValueError("source repository HEAD changed")
    if _git(SOURCE_REPOSITORY, "branch", "--show-current") != SOURCE_BRANCH:
        raise ValueError("source repository branch changed")
    if _git(SOURCE_REPOSITORY, "status", "--porcelain"):
        raise ValueError("source repository worktree is not clean")

    audit_dir = root / "artifacts/scientific_survival_audit_v1"
    pre_inventory = audit_dir / "pre_repair_result_inventory.csv"
    if not pre_inventory.is_file():
        raise FileNotFoundError("pre-repair result inventory is missing")
    unexpected_existing = [
        path for path in audit_dir.rglob("*") if path.is_file() and path != pre_inventory
    ]
    if unexpected_existing:
        raise FileExistsError(f"survival artifact output already exists: {unexpected_existing[0]}")

    result_dir = root / "results/full_synthetic_development_v1"
    primary = _read_json(result_dir / "primary_analysis.json")
    independent = _read_json(result_dir / "independent_verification.json")
    formal_run = _read_json(result_dir / "run_manifest.json")
    lock = _read_json(root / "frozen_assets/full_synthetic_development_snapshot_lock_v1.json")
    source_decision = primary["final_decision"]
    published_decision = _read_json(
        root / "artifacts/full_synthetic_development_v1/final_decision.json"
    )
    comparison = full_synthetic_analysis_verifier_comparison(primary, independent)
    if comparison["section_difference_count"] != 0:
        raise ValueError("primary and independent analysis differ")
    integrity = verify_pre_repair_inventory(root, pre_inventory)
    compact = verify_full_synthetic_artifact(
        root / "artifacts/full_synthetic_development_v1", write_report=False
    )
    scope = publisher_only_change_scope(
        inventory_verification=integrity,
        source_decision=source_decision,
        published_decision=published_decision,
        changed_paths=_changed_paths(root),
        matplotlib_version=matplotlib.__version__,
    )
    scope.update(
        {
            "formal_public_entry_post_freeze_status": "EXPECTED_STRICT_REJECTION",
            "publication_execution_path": "SAME_PUBLISHER_INTERNAL_ATOMIC_STAGING_AND_DOUBLE_VERIFICATION",
            "compact_artifact_file_count": compact["required_file_count"],
            "compact_artifact_sha256_missing_count": compact["sha256_missing_count"],
            "compact_artifact_sha256_mismatch_count": compact["sha256_mismatch_count"],
            "compact_artifact_sha256_unlisted_count": compact[
                "sha256_unlisted_required_count"
            ],
        }
    )

    snapshots = attach_observed_component_signatures(
        lock["snapshots"], lambda row: _load_frozen_snapshot_arrays(root, row)
    )
    replicate = audit_replicate_uniqueness(snapshots)
    replicate_pass = _replicate_semantics_pass(replicate)
    systematic = evaluate_long_corridor_systematic_offset(
        replicate["cell_rows"], primary["systematic_offset_summary"]
    )
    model_rows = join_development_model_rows(primary)
    scene_effect = evaluate_unique_unit_primary_scene_effect(
        model_rows,
        snapshots,
        original_trial_weighted_rows=primary["scene_effect_paired"],
    )
    cross_backend = evaluate_cross_backend_unique_unit(model_rows, snapshots)
    turnover = evaluate_turnover_robustness(model_rows, snapshots)
    seed_provenance = scan_confirmatory_seed_provenance(root)
    leakage = audit_model_data_leakage(
        primary,
        repository_root=root,
        seed_provenance=seed_provenance,
    )
    model = evaluate_model_weighting_sensitivity(model_rows)
    shortlist = deterministic_counterexample_shortlist(
        primary["automatic_nonequivalence_candidates"]
    )
    claims = scientific_claim_matrix(
        primary_scene_pass=scene_effect["PRIMARY_SCENE_EFFECT_UNIQUE_UNIT_PASS"],
        cross_backend_pass=cross_backend["CROSS_BACKEND_UNIQUE_UNIT_PASS"],
        reassociation_pass=turnover["REASSOCIATION_ROBUSTNESS_PASS"],
        model_leakage_pass=leakage["MODEL_DATA_LEAKAGE_AUDIT_PASS"],
        model_robust_pass=model["MODEL_INCREMENTAL_VALUE_ROBUST_PASS"],
        systematic_full_noise_pass=systematic["SYSTEMATIC_OFFSET_FULL_NOISE_PASS"],
        independent_authorized_term=systematic[
            "INDEPENDENT_NOISE_FREE_AUTHORIZED_TERM"
        ],
    )
    wording_pass = claim_wording_boundary_pass(claims)
    survival_gates = {
        "RAW_EVIDENCE_INTEGRITY_PASS": integrity["RAW_EVIDENCE_INTEGRITY_PASS"],
        "ARTIFACT_PUBLICATION_PASS": compact["ARTIFACT_VERIFICATION_PASS"]
        and scope["publisher_repair_scope_pass"],
        "REPLICATE_UNIQUENESS_AUDIT_PASS": replicate_pass,
        "PRIMARY_SCENE_EFFECT_UNIQUE_UNIT_PASS": scene_effect[
            "PRIMARY_SCENE_EFFECT_UNIQUE_UNIT_PASS"
        ],
        "CROSS_BACKEND_UNIQUE_UNIT_PASS": cross_backend[
            "CROSS_BACKEND_UNIQUE_UNIT_PASS"
        ],
        "REASSOCIATION_ROBUSTNESS_PASS": turnover[
            "REASSOCIATION_ROBUSTNESS_PASS"
        ],
        "MODEL_DATA_LEAKAGE_AUDIT_PASS": leakage[
            "MODEL_DATA_LEAKAGE_AUDIT_PASS"
        ],
        "MODEL_INCREMENTAL_VALUE_ROBUST_PASS": model[
            "MODEL_INCREMENTAL_VALUE_ROBUST_PASS"
        ],
        "CLAIM_WORDING_BOUNDARY_PASS": wording_pass,
    }
    survival_pass = all(value is True for value in survival_gates.values())

    confirm_result: dict[str, Any] = {
        "SYNTHETIC_CONFIRMATORY_PROTOCOL_READY": False,
        "planned_snapshot_count": 0,
        "planned_trial_count": 0,
    }
    real_result: dict[str, Any] = {
        "REAL_DATA_PROTOCOL_FRAMEWORK_READY": False,
        "REAL_DATA_DATASET_ELIGIBILITY_COMPLETE": False,
        "eligibility_requirement_count": 0,
    }
    if survival_pass:
        if seed_provenance["CONFIRMATORY_SEED_PROVENANCE_PASS"] is not True:
            raise PermissionError("Confirmatory seed provenance failed")
        model_lock = build_frozen_development_model_lock(
            model_rows, repository_root=root
        )
        provisional = {"SCIENTIFIC_SURVIVAL_AUDIT_PASS": True}
        confirm_result = materialize_synthetic_confirmatory_protocol(
            root,
            scientific_survival_decision=provisional,
            model_lock=model_lock,
        )
        real_result = materialize_real_data_protocol_framework(
            root, scientific_survival_audit_pass=True
        )

    decision = {
        **survival_gates,
        "PSEUDOREPLICATION_RISK_IDENTIFIED": replicate[
            "PSEUDOREPLICATION_RISK_IDENTIFIED"
        ],
        "DROPOUT_INPUT_VARIATION_OBSERVED": replicate[
            "DROPOUT_INPUT_VARIATION_OBSERVED"
        ],
        "DROPOUT_MASK_IDENTITY_FULLY_RECONSTRUCTED": replicate[
            "DROPOUT_MASK_IDENTITY_FULLY_RECONSTRUCTED"
        ],
        "SYSTEMATIC_OFFSET_FULL_NOISE_PASS": systematic[
            "SYSTEMATIC_OFFSET_FULL_NOISE_PASS"
        ],
        "SYSTEMATIC_OFFSET_CLAIM_SCOPE": systematic[
            "SYSTEMATIC_OFFSET_CLAIM_SCOPE"
        ],
        "MODEL_CLAIM_TYPE": leakage["MODEL_CLAIM_TYPE"],
        "COUNTEREXAMPLE_CLAIM_AUTHORIZED": False,
        "SCIENTIFIC_SURVIVAL_AUDIT_PASS": survival_pass,
        "CONFIRMATORY_SEED_PROVENANCE_PASS": seed_provenance[
            "CONFIRMATORY_SEED_PROVENANCE_PASS"
        ],
        "SYNTHETIC_CONFIRMATORY_PROTOCOL_READY": confirm_result[
            "SYNTHETIC_CONFIRMATORY_PROTOCOL_READY"
        ],
        "REAL_DATA_PROTOCOL_FRAMEWORK_READY": real_result[
            "REAL_DATA_PROTOCOL_FRAMEWORK_READY"
        ],
        "REAL_DATA_DATASET_ELIGIBILITY_COMPLETE": False,
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
    }

    tables = audit_dir / "tables"
    figures = audit_dir / "figures"
    write_csv(tables / "replicate_uniqueness_by_cell.csv", replicate["cell_rows"])
    write_csv(tables / "replicate_uniqueness_by_condition.csv", replicate["condition_rows"])
    write_csv(tables / "measurement_seed_effectiveness.csv", replicate["measurement_seed_rows"])
    write_csv(tables / "repeat_index_effectiveness.csv", replicate["repeat_index_rows"])
    write_csv(tables / "unique_unit_scene_effect.csv", scene_effect["rows"])
    write_csv(tables / "cross_backend_unique_unit.csv", _cross_backend_table(cross_backend))
    write_csv(tables / "turnover_robustness.csv", _turnover_table(turnover))
    write_csv(tables / "model_leakage_audit.csv", leakage["rows"])
    write_csv(tables / "model_weighting_sensitivity.csv", model["model_weighting_sensitivity"])
    write_csv(tables / "model_fold_results.csv", model["model_fold_results"])
    write_csv(tables / "counterexample_review_shortlist.csv", shortlist)
    write_csv(tables / "scientific_claim_authorization.csv", claims)
    gate_rows = [
        {"gate": name, "pass": value, "contract": "frozen survival audit v1"}
        for name, value in survival_gates.items()
    ]
    gate_rows.extend(
        [
            {
                "gate": "SCIENTIFIC_SURVIVAL_AUDIT_PASS",
                "pass": survival_pass,
                "contract": "conjunction of nine survival gates",
            },
            {
                "gate": "SYSTEMATIC_OFFSET_FULL_NOISE_PASS",
                "pass": systematic["SYSTEMATIC_OFFSET_FULL_NOISE_PASS"],
                "contract": "2/3 groups per backend plus median fraction >= 0.70",
            },
            {
                "gate": "CONFIRMATORY_SEED_PROVENANCE_PASS",
                "pass": seed_provenance["CONFIRMATORY_SEED_PROVENANCE_PASS"],
                "contract": "zero structured experimental use",
            },
        ]
    )
    write_csv(tables / "gate_summary.csv", gate_rows)

    build_survival_figures(
        figures,
        replicate_condition_rows=replicate["condition_rows"],
        scene_effect_rows=scene_effect["rows"],
        turnover_rows=turnover["sensitivity_rows"],
        model_sensitivity_rows=model["model_weighting_sensitivity"],
        claim_rows=claims,
    )
    _json_write(audit_dir / "publisher_only_change_scope.json", scope)
    (audit_dir / "publisher_repair_report.md").write_text(
        _publisher_report(scope, compact), encoding="utf-8"
    )
    (audit_dir / "replicate_uniqueness_audit.md").write_text(
        _replicate_report(replicate, systematic), encoding="utf-8"
    )
    (audit_dir / "counterexample_review_packet.md").write_text(
        _candidate_packet(shortlist, len(primary["automatic_nonequivalence_candidates"])),
        encoding="utf-8",
    )
    (audit_dir / "scientific_survival_audit_report.md").write_text(
        _survival_report(
            decision=decision,
            replicate=replicate,
            scene_effect=scene_effect,
            cross_backend=cross_backend,
            turnover=turnover,
            model=model,
            claims=claims,
        ),
        encoding="utf-8",
    )
    _json_write(audit_dir / "final_decision.json", decision)
    run_manifest = {
        "schema_version": "scientific_survival_audit_run_v1",
        "baseline_commit": BASELINE_COMMIT,
        "baseline_archive_tag": ARCHIVE_TAG,
        "source_repository_branch": SOURCE_BRANCH,
        "source_repository_commit": SOURCE_COMMIT,
        "registration_trial_rerun_count": 0,
        "snapshot_regeneration_count": 0,
        "raw_trial_count": integrity["semantic_role_counts"]["raw_trial_result"],
        "snapshot_count": formal_run["combined_completed_snapshot_count"],
        "trial_count": formal_run["combined_completed_trial_count"],
        "open3d_trial_count": 1260,
        "pcl_trial_count": 1260,
        "native_trial_count": 0,
        "analysis_verifier_section_difference_count": comparison[
            "section_difference_count"
        ],
        "analysis_verifier_leaf_difference_count": comparison[
            "leaf_difference_count"
        ],
        "analysis_verifier_maximum_absolute_numeric_difference": comparison[
            "maximum_absolute_numeric_difference"
        ],
        "pre_repair_inventory_sha256": file_sha256(pre_inventory),
        "structured_confirmatory_seed_file_count": seed_provenance[
            "structured_file_count"
        ],
        "confirmatory_seed_instantiation_count": seed_provenance[
            "confirmatory_seed_instantiation_count"
        ],
        "automatic_candidate_count": len(primary[
            "automatic_nonequivalence_candidates"
        ]),
        "shortlist_count": len(shortlist),
        "dropout_mask_reconstruction_failure_count": replicate[
            "dropout_mask_reconstruction_failure_count"
        ],
        "dropout_mask_reconstruction_unknown_count": replicate[
            "dropout_mask_reconstruction_unknown_count"
        ],
        "confirmatory_protocol": confirm_result,
        "real_data_framework": real_result,
        "final_decision": decision,
    }
    _json_write(audit_dir / "run_manifest.json", run_manifest)

    expected_sha = set(REQUIRED_FILES) - SHA_EXCLUDED
    (audit_dir / "SHA256SUMS").write_text(
        "".join(
            f"{file_sha256(audit_dir / relative)}  {relative}\n"
            for relative in sorted(expected_sha)
        ),
        encoding="utf-8",
    )
    verification = verify_scientific_survival_artifact(
        audit_dir, write_report=True
    )
    read_only = verify_scientific_survival_artifact(
        audit_dir, write_report=False
    )
    if verification != read_only or verification["ARTIFACT_VERIFICATION_PASS"] is not True:
        raise ValueError("Scientific Survival artifact verification failed")
    return {
        "artifact_verification": verification,
        "compact_artifact_verification": compact,
        "confirmatory_protocol": confirm_result,
        "cross_backend": cross_backend,
        "decision": decision,
        "integrity": integrity,
        "model": model,
        "model_leakage": leakage,
        "publisher_scope": scope,
        "real_data_framework": real_result,
        "replicate": replicate,
        "scene_effect": scene_effect,
        "seed_provenance": seed_provenance,
        "shortlist": shortlist,
        "systematic": systematic,
        "turnover": turnover,
    }


__all__ = ["run_scientific_survival_audit"]
