"""Pure builders for the gated Synthetic Confirmatory protocol.

Importing or calling a ``build_*`` function never creates a snapshot and never
initializes an RNG.  Repository artifacts are written only by the explicitly
guarded ``materialize_synthetic_confirmatory_protocol`` entry point after the
scientific-survival and seed-provenance gates have both passed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scientific_survival_models import (
    BACKENDS,
    CONFIRMATORY_BOOTSTRAP_SEED,
    CONFIRMATORY_GEOMETRY_SEEDS,
    CONFIRMATORY_MEASUREMENT_SEEDS,
    FINAL_MODEL_WEIGHTING,
    canonical_json_sha256,
    scan_confirmatory_seed_provenance,
)


SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
CONDITIONS = ("IDEAL_MATCHED", "INDEPENDENT_NOISE_FREE", "FULL_NOISE")
SNAPSHOT_COUNT = 595
TRIAL_COUNT = 1190


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _snapshot_id(
    *,
    scene: str,
    condition: str,
    geometry_seed: int,
    measurement_seed: int | None,
    repeat_index: int,
) -> str:
    identity = {
        "condition": condition,
        "geometry_seed": geometry_seed,
        "measurement_seed": measurement_seed,
        "repeat_index": repeat_index,
        "scene_variant": scene,
    }
    return f"synthetic-confirmatory-v1::{canonical_json_sha256(identity)}"


def build_synthetic_confirmatory_plan() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the frozen 595-snapshot/1,190-trial design without RNG use."""

    snapshots: list[dict[str, Any]] = []
    for scene in SCENES:
        for geometry_seed in CONFIRMATORY_GEOMETRY_SEEDS:
            for condition in ("IDEAL_MATCHED", "INDEPENDENT_NOISE_FREE"):
                snapshots.append(
                    {
                        "planned_snapshot_id": _snapshot_id(
                            scene=scene,
                            condition=condition,
                            geometry_seed=geometry_seed,
                            measurement_seed=None,
                            repeat_index=0,
                        ),
                        "scene_variant": scene,
                        "condition": condition,
                        "geometry_seed": geometry_seed,
                        "measurement_seed": None,
                        "repeat_index": 0,
                        "planned_backend_count": 2,
                        "replicate_semantics": (
                            "ONE_CONTROL_INPUT"
                            if condition == "IDEAL_MATCHED"
                            else "ONE_DETERMINISTIC_INDEPENDENT_INPUT"
                        ),
                    }
                )
            for measurement_seed in CONFIRMATORY_MEASUREMENT_SEEDS:
                for repeat_index in range(5):
                    snapshots.append(
                        {
                            "planned_snapshot_id": _snapshot_id(
                                scene=scene,
                                condition="FULL_NOISE",
                                geometry_seed=geometry_seed,
                                measurement_seed=measurement_seed,
                                repeat_index=repeat_index,
                            ),
                            "scene_variant": scene,
                            "condition": "FULL_NOISE",
                            "geometry_seed": geometry_seed,
                            "measurement_seed": measurement_seed,
                            "repeat_index": repeat_index,
                            "planned_backend_count": 2,
                            "replicate_semantics": "FIFTEEN_STOCHASTIC_INPUTS_PER_SCENE_GEOMETRY",
                        }
                    )
    snapshots.sort(
        key=lambda row: (
            SCENES.index(str(row["scene_variant"])),
            CONFIRMATORY_GEOMETRY_SEEDS.index(int(row["geometry_seed"])),
            CONDITIONS.index(str(row["condition"])),
            -1 if row["measurement_seed"] is None else CONFIRMATORY_MEASUREMENT_SEEDS.index(int(row["measurement_seed"])),
            int(row["repeat_index"]),
        )
    )
    trials = [
        {
            "planned_trial_id": f"{snapshot['planned_snapshot_id']}::{backend}",
            "planned_snapshot_id": snapshot["planned_snapshot_id"],
            "scene_variant": snapshot["scene_variant"],
            "condition": snapshot["condition"],
            "geometry_seed": snapshot["geometry_seed"],
            "measurement_seed": snapshot["measurement_seed"],
            "repeat_index": snapshot["repeat_index"],
            "backend": backend,
        }
        for snapshot in snapshots
        for backend in BACKENDS
    ]
    if (
        len(snapshots) != SNAPSHOT_COUNT
        or len({row["planned_snapshot_id"] for row in snapshots}) != SNAPSHOT_COUNT
        or len(trials) != TRIAL_COUNT
        or len({row["planned_trial_id"] for row in trials}) != TRIAL_COUNT
        or sum(row["condition"] == "IDEAL_MATCHED" for row in snapshots) != 35
        or sum(row["condition"] == "INDEPENDENT_NOISE_FREE" for row in snapshots) != 35
        or sum(row["condition"] == "FULL_NOISE" for row in snapshots) != 525
        or any(row["backend"] not in BACKENDS for row in trials)
    ):
        raise AssertionError("Synthetic Confirmatory plan cardinality changed")
    return snapshots, trials


def build_synthetic_confirmatory_gate_contract() -> dict[str, Any]:
    """Return the exact H1-H6 preregistration contract."""

    hypotheses = {
        "H1_IDEAL_CONTROL": {
            "backends": list(BACKENDS),
            "solver_failure_count_max": 0,
            "nonfinite_output_count_max": 0,
            "translation_q95_max_m": 0.001,
            "rotation_q95_max_deg": 0.01,
            "rotation_q95_max_rad": math.radians(0.01),
            "quantile_method": "linear",
        },
        "H2_LONG_CORRIDOR_SCENE_EFFECT": {
            "conditions": ["INDEPENDENT_NOISE_FREE", "FULL_NOISE"],
            "backends": list(BACKENDS),
            "comparison": "LONG_CORRIDOR_vs_GEOMETRY_RICH_ROOM",
            "geometry_block_count": 5,
            "minimum_long_greater_than_rich_blocks": 4,
            "geometry_level_median_ratio_min": 5.0,
            "geometry_level_median_absolute_difference_min_m": 0.005,
        },
        "H3_CROSS_BACKEND_SCENE_RANKING": {
            "conditions": ["INDEPENDENT_NOISE_FREE", "FULL_NOISE"],
            "scene_count": 7,
            "spearman_rho_min": 0.70,
        },
        "H4_REASSOCIATION_MECHANISM": {
            "backends": list(BACKENDS),
            "pooled_turnover_error_spearman_rho_min": 0.40,
            "scene_centered_spearman_rho_min": 0.20,
            "leave_one_geometry_seed_out_fold_count": 5,
            "direction_stability_rule": "ALL_FIVE_HELD_OUT_RHO_STRICTLY_POSITIVE",
            "causal_claim_authorized": False,
        },
        "H5_FROZEN_MODEL_B_INCREMENTAL_VALUE": {
            "backends": list(BACKENDS),
            "refit_authorized": False,
            "restandardization_authorized": False,
            "feature_change_authorized": False,
            "alpha_change_authorized": False,
            "mae_b_to_mae_a_ratio_max": 0.90,
        },
        "H6_FULL_NOISE_SYSTEMATIC_OFFSET": {
            "scene_variant": "LONG_CORRIDOR",
            "condition": "FULL_NOISE",
            "backends": list(BACKENDS),
            "geometry_group_count": 5,
            "minimum_passing_geometry_groups": 4,
            "planned_replicates_per_geometry": 15,
            "effective_replicate_count_min": 12,
            "systematic_translation_offset_min_m": 0.005,
            "systematic_fraction_min": 0.60,
            "median_systematic_fraction_min": 0.70,
        },
    }
    core = {
        "all_hypotheses_required": True,
        "hypotheses": hypotheses,
        "hypothesis_count": 6,
        "schema_version": "synthetic_confirmatory_gate_contract_v1",
    }
    return {**core, "gate_contract_payload_sha256": canonical_json_sha256(core)}


def _validate_model_lock(model_lock: Mapping[str, Any]) -> None:
    unsigned = {
        name: value
        for name, value in model_lock.items()
        if name != "model_lock_payload_sha256"
    }
    models = model_lock.get("models")
    identities = {
        (row.get("backend_schema_name"), row.get("model"))
        for row in models
    } if type(models) is list else set()
    if (
        model_lock.get("schema_version") != "confirmatory_development_trained_models_v1"
        or model_lock.get("model_count") != 4
        or model_lock.get("weighting_scheme") != FINAL_MODEL_WEIGHTING
        or model_lock.get("confirmatory_seed_access_count") != 0
        or identities != {(backend, model) for backend in BACKENDS for model in ("A", "B")}
        or model_lock.get("model_lock_payload_sha256") != canonical_json_sha256(unsigned)
    ):
        raise ValueError("frozen Development model lock is invalid")


def build_synthetic_confirmatory_protocol(
    *,
    scientific_survival_decision: Mapping[str, Any],
    seed_provenance: Mapping[str, Any],
    model_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the protocol in memory, failing closed before any output write."""

    if scientific_survival_decision.get("SCIENTIFIC_SURVIVAL_AUDIT_PASS") is not True:
        raise PermissionError("scientific survival did not authorize protocol design")
    if seed_provenance.get("CONFIRMATORY_SEED_PROVENANCE_PASS") is not True:
        raise PermissionError("Confirmatory seed provenance failed")
    _validate_model_lock(model_lock)
    snapshots, trials = build_synthetic_confirmatory_plan()
    gate_contract = build_synthetic_confirmatory_gate_contract()
    model_sha = hashlib.sha256(_canonical_json_bytes(dict(model_lock))).hexdigest()
    core = {
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_PROTOCOL_READY": True,
        "backend_count": 2,
        "backends": list(BACKENDS),
        "bootstrap_seed": CONFIRMATORY_BOOTSTRAP_SEED,
        "condition_count": 3,
        "conditions": list(CONDITIONS),
        "confirmatory_seed_provenance_pass": True,
        "development_model_lock_sha256": model_sha,
        "development_model_weighting": FINAL_MODEL_WEIGHTING,
        "gate_contract_payload_sha256": gate_contract["gate_contract_payload_sha256"],
        "geometry_seeds": list(CONFIRMATORY_GEOMETRY_SEEDS),
        "measurement_seeds": list(CONFIRMATORY_MEASUREMENT_SEEDS),
        "native_trial_count": 0,
        "planned_snapshot_count": len(snapshots),
        "planned_snapshot_identity_sha256": canonical_json_sha256(snapshots),
        "planned_trial_count": len(trials),
        "planned_trial_identity_sha256": canonical_json_sha256(trials),
        "registration_execution_count": 0,
        "scene_count": 7,
        "scenes": list(SCENES),
        "schema_version": "synthetic_confirmatory_protocol_v1",
        "scientific_survival_audit_pass": True,
        "snapshot_generation_count": 0,
    }
    protocol = {**core, "protocol_payload_sha256": canonical_json_sha256(core)}
    return {
        "gate_contract": gate_contract,
        "model_lock": dict(model_lock),
        "planned_snapshots": snapshots,
        "planned_trials": trials,
        "protocol": protocol,
        "seed_provenance": dict(seed_provenance),
    }


def synthetic_confirmatory_protocol_markdown(bundle: Mapping[str, Any]) -> str:
    protocol = bundle["protocol"]
    gates = bundle["gate_contract"]["hypotheses"]
    lines = [
        "# Synthetic Confirmatory Protocol v1",
        "",
        "This preregistration is design-only. It creates no point clouds and authorizes no run.",
        "",
        f"- Planned snapshots: **{protocol['planned_snapshot_count']}**",
        f"- Planned trials: **{protocol['planned_trial_count']}**",
        "- Backends: Open3D and PCL; Native is forbidden.",
        "- Conditions: IDEAL_MATCHED, INDEPENDENT_NOISE_FREE, FULL_NOISE.",
        "- `CONFIRMATORY_RUN_AUTHORIZED = false`",
        "",
        "INDEPENDENT_NOISE_FREE has one input per scene × geometry seed and is not represented as repeated stochastic evidence. FULL_NOISE has 3 measurement seeds × 5 repeats.",
        "",
        "## Frozen hypotheses",
        "",
    ]
    for name, contract in gates.items():
        lines.extend(
            [
                f"### {name}",
                "",
                "```json",
                json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Frozen models",
            "",
            "Four Development-trained Ridge models use unique-input/condition-balanced weights. Scalers, coefficients, intercepts, feature order, alpha, data SHA, and code SHA are locked. Confirmatory refitting and restandardization are forbidden.",
            "",
            "## Interpretation boundary",
            "",
            "Model B is a post-registration explanatory model. Association findings are not causal claims.",
            "",
        ]
    )
    return "\n".join(lines)


def materialize_synthetic_confirmatory_protocol(
    repository_root: str | Path,
    *,
    scientific_survival_decision: Mapping[str, Any],
    model_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the seven required protocol files after both hard gates pass."""

    repository = Path(repository_root).resolve()
    provenance = scan_confirmatory_seed_provenance(repository)
    bundle = build_synthetic_confirmatory_protocol(
        scientific_survival_decision=scientific_survival_decision,
        seed_provenance=provenance,
        model_lock=model_lock,
    )
    protocol_dir = repository / "protocols"
    frozen_dir = repository / "frozen_assets"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    frozen_dir.mkdir(parents=True, exist_ok=True)
    snapshot_fields = (
        "planned_snapshot_id",
        "scene_variant",
        "condition",
        "geometry_seed",
        "measurement_seed",
        "repeat_index",
        "planned_backend_count",
        "replicate_semantics",
    )
    trial_fields = (
        "planned_trial_id",
        "planned_snapshot_id",
        "scene_variant",
        "condition",
        "geometry_seed",
        "measurement_seed",
        "repeat_index",
        "backend",
    )
    model_bytes = _canonical_json_bytes(bundle["model_lock"])
    model_descriptor_core = {
        "feature_change_authorized": False,
        "model_asset_path": "frozen_assets/confirmatory_development_trained_models_v1.json",
        "model_asset_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "model_count": 4,
        "refit_authorized": False,
        "restandardization_authorized": False,
        "schema_version": "confirmatory_development_model_lock_v1",
        "weighting_scheme": FINAL_MODEL_WEIGHTING,
    }
    model_descriptor = {
        **model_descriptor_core,
        "model_lock_payload_sha256": canonical_json_sha256(model_descriptor_core),
    }
    files = {
        protocol_dir / "synthetic_confirmatory_protocol_v1.json": _canonical_json_bytes(bundle["protocol"]),
        protocol_dir / "synthetic_confirmatory_protocol_v1.md": (
            synthetic_confirmatory_protocol_markdown(bundle) + "\n"
        ).encode("utf-8"),
        protocol_dir / "synthetic_confirmatory_planned_snapshots.csv": _csv_bytes(bundle["planned_snapshots"], snapshot_fields),
        protocol_dir / "synthetic_confirmatory_planned_trials.csv": _csv_bytes(bundle["planned_trials"], trial_fields),
        protocol_dir / "synthetic_confirmatory_gate_contract.json": _canonical_json_bytes(bundle["gate_contract"]),
        protocol_dir / "confirmatory_seed_provenance_audit.json": _canonical_json_bytes(provenance),
        protocol_dir / "confirmatory_development_model_lock.json": _canonical_json_bytes(model_descriptor),
        frozen_dir / "confirmatory_development_trained_models_v1.json": model_bytes,
    }
    existing = [path for path in files if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace Confirmatory protocol output: {existing[0]}")
    for path, payload in files.items():
        with path.open("xb") as stream:
            stream.write(payload)
    return {
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_PROTOCOL_READY": True,
        "file_count": len(files),
        "files": {
            path.relative_to(repository).as_posix(): hashlib.sha256(payload).hexdigest()
            for path, payload in files.items()
        },
        "planned_snapshot_count": SNAPSHOT_COUNT,
        "planned_trial_count": TRIAL_COUNT,
        "registration_execution_count": 0,
        "snapshot_generation_count": 0,
    }


__all__ = [
    "BACKENDS",
    "CONDITIONS",
    "SCENES",
    "SNAPSHOT_COUNT",
    "TRIAL_COUNT",
    "build_synthetic_confirmatory_gate_contract",
    "build_synthetic_confirmatory_plan",
    "build_synthetic_confirmatory_protocol",
    "materialize_synthetic_confirmatory_protocol",
    "synthetic_confirmatory_protocol_markdown",
]
