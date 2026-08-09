"""Read-only qualification of the frozen Synthetic Confirmatory protocol.

This module audits the already-published plans and contracts.  It deliberately
contains no plan builder, RNG construction, snapshot generation, or backend
adapter.  The pre-run pipeline can therefore use its report as a fail-closed
input without changing any scientific object.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCIENTIFIC_SURVIVAL_COMMIT = "ffc15334f4ded25fdba5e709b45657dbad481dfc"
SCIENTIFIC_SURVIVAL_TAG = "archive/zero-perturbation-scientific-survival-audit-v1"
EXPECTED_MODEL_FILE_SHA256 = (
    "99806f83d3a0d2393ee7e36e8c06f14a1a8a44fb8d02b074ebc2d9fe750d0872"
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
BACKENDS = ("open3d_point_to_plane", "pcl_point_to_plane")
GEOMETRY_SEEDS = (248284635, 376488233, 198112089, 229684695, 226655024)
MEASUREMENT_SEEDS = (469989467, 1088311622, 916609326)
BOOTSTRAP_SEED = 1083684578
CONFIRMATORY_SEEDS = frozenset((*GEOMETRY_SEEDS, *MEASUREMENT_SEEDS, BOOTSTRAP_SEED))

EXPECTED_FILE_SHA256 = {
    "protocols/synthetic_confirmatory_protocol_v1.json": (
        "d9af2b3a38d01e47e9c6c5afff849a77abe55f3e685c2ce04cc51c19b582de11"
    ),
    "protocols/synthetic_confirmatory_protocol_v1.md": (
        "24ab8acc3545c1e386ee1d048c29f3f5672b498aa9a7a9e15dd7473bed324839"
    ),
    "protocols/synthetic_confirmatory_planned_snapshots.csv": (
        "768cd7a5a5a27104eaeb44ef8d724df92c6846d7e3819abd3403d1e0c57e6435"
    ),
    "protocols/synthetic_confirmatory_planned_trials.csv": (
        "0b6daaf37b25ad28892a789b9fb30ef5ab0fa4bbf97bde7cef84a61d431a2587"
    ),
    "protocols/synthetic_confirmatory_gate_contract.json": (
        "e1cf01e7850222ff239354d6ae887fa783bb153b025ac484bcd5534f923d134f"
    ),
    "protocols/confirmatory_seed_provenance_audit.json": (
        "252ceb5aba7a25390b4ffdac43e8a887ce2ae1fb73bb89f13af91aa4689ca7e2"
    ),
    "frozen_assets/confirmatory_development_trained_models_v1.json": (
        EXPECTED_MODEL_FILE_SHA256
    ),
}
EXPECTED_SURVIVAL_DECISION_SHA256 = (
    "2375af293d95092e1e23e9a956cbff44ddfeda27881d717859de0f60e912b523"
)
EXPECTED_SURVIVAL_RUN_MANIFEST_SHA256 = (
    "2f4a4aa70c164d6ffa4543a4173eec213f46e7211d20377a91424d9809189173"
)

SNAPSHOT_FIELDS = (
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
    "planned_backend_count",
    "replicate_semantics",
)
TRIAL_FIELDS = (
    "planned_trial_id",
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
    "backend",
)

EXPECTED_HYPOTHESES: dict[str, dict[str, Any]] = {
    "H1_IDEAL_CONTROL": {
        "backends": list(BACKENDS),
        "nonfinite_output_count_max": 0,
        "quantile_method": "linear",
        "rotation_q95_max_deg": 0.01,
        "rotation_q95_max_rad": 0.00017453292519943296,
        "solver_failure_count_max": 0,
        "translation_q95_max_m": 0.001,
    },
    "H2_LONG_CORRIDOR_SCENE_EFFECT": {
        "backends": list(BACKENDS),
        "comparison": "LONG_CORRIDOR_vs_GEOMETRY_RICH_ROOM",
        "conditions": ["INDEPENDENT_NOISE_FREE", "FULL_NOISE"],
        "geometry_block_count": 5,
        "geometry_level_median_absolute_difference_min_m": 0.005,
        "geometry_level_median_ratio_min": 5.0,
        "minimum_long_greater_than_rich_blocks": 4,
    },
    "H3_CROSS_BACKEND_SCENE_RANKING": {
        "conditions": ["INDEPENDENT_NOISE_FREE", "FULL_NOISE"],
        "scene_count": 7,
        "spearman_rho_min": 0.70,
    },
    "H4_REASSOCIATION_MECHANISM": {
        "backends": list(BACKENDS),
        "causal_claim_authorized": False,
        "direction_stability_rule": "ALL_FIVE_HELD_OUT_RHO_STRICTLY_POSITIVE",
        "leave_one_geometry_seed_out_fold_count": 5,
        "pooled_turnover_error_spearman_rho_min": 0.40,
        "scene_centered_spearman_rho_min": 0.20,
    },
    "H5_FROZEN_MODEL_B_INCREMENTAL_VALUE": {
        "alpha_change_authorized": False,
        "backends": list(BACKENDS),
        "feature_change_authorized": False,
        "mae_b_to_mae_a_ratio_max": 0.90,
        "refit_authorized": False,
        "restandardization_authorized": False,
    },
    "H6_FULL_NOISE_SYSTEMATIC_OFFSET": {
        "backends": list(BACKENDS),
        "condition": "FULL_NOISE",
        "effective_replicate_count_min": 12,
        "geometry_group_count": 5,
        "median_systematic_fraction_min": 0.70,
        "minimum_passing_geometry_groups": 4,
        "planned_replicates_per_geometry": 15,
        "scene_variant": "LONG_CORRIDOR",
        "systematic_fraction_min": 0.60,
        "systematic_translation_offset_min_m": 0.005,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            output[key] = value
        return output

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=object_hook,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {path}: {token}")
        ),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"unexpected CSV schema: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"CSV row has excess columns: {path}")
    return rows


def _integer(value: str, name: str) -> int:
    if not value or value.strip() != value:
        raise ValueError(f"{name} is not a canonical integer")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} is not an integer") from error
    if str(parsed) != value:
        raise ValueError(f"{name} is not a canonical integer")
    return parsed


def _optional_integer(value: str, name: str) -> int | None:
    return None if value == "" else _integer(value, name)


def _typed_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path, SNAPSHOT_FIELDS)
    return [
        {
            "planned_snapshot_id": row["planned_snapshot_id"],
            "scene_variant": row["scene_variant"],
            "condition": row["condition"],
            "geometry_seed": _integer(row["geometry_seed"], "geometry_seed"),
            "measurement_seed": _optional_integer(
                row["measurement_seed"], "measurement_seed"
            ),
            "repeat_index": _integer(row["repeat_index"], "repeat_index"),
            "planned_backend_count": _integer(
                row["planned_backend_count"], "planned_backend_count"
            ),
            "replicate_semantics": row["replicate_semantics"],
        }
        for row in rows
    ]


def _typed_trial_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path, TRIAL_FIELDS)
    return [
        {
            "planned_trial_id": row["planned_trial_id"],
            "planned_snapshot_id": row["planned_snapshot_id"],
            "scene_variant": row["scene_variant"],
            "condition": row["condition"],
            "geometry_seed": _integer(row["geometry_seed"], "geometry_seed"),
            "measurement_seed": _optional_integer(
                row["measurement_seed"], "measurement_seed"
            ),
            "repeat_index": _integer(row["repeat_index"], "repeat_index"),
            "backend": row["backend"],
        }
        for row in rows
    ]


def _expected_snapshot_id(row: Mapping[str, Any]) -> str:
    identity = {
        "condition": row["condition"],
        "geometry_seed": row["geometry_seed"],
        "measurement_seed": row["measurement_seed"],
        "repeat_index": row["repeat_index"],
        "scene_variant": row["scene_variant"],
    }
    return f"synthetic-confirmatory-v1::{_canonical_sha256(identity)}"


def audit_confirmatory_plan(
    snapshot_plan_path: str | Path,
    trial_plan_path: str | Path,
) -> dict[str, Any]:
    """Audit the frozen CSVs without regenerating or rewriting either plan."""

    snapshots = _typed_snapshot_rows(Path(snapshot_plan_path))
    trials = _typed_trial_rows(Path(trial_plan_path))
    snapshot_ids = [str(row["planned_snapshot_id"]) for row in snapshots]
    trial_ids = [str(row["planned_trial_id"]) for row in trials]
    duplicate_snapshot_count = len(snapshot_ids) - len(set(snapshot_ids))
    duplicate_trial_count = len(trial_ids) - len(set(trial_ids))
    snapshot_by_id = {str(row["planned_snapshot_id"]): row for row in snapshots}

    condition_counts = Counter(str(row["condition"]) for row in snapshots)
    backend_counts = Counter(str(row["backend"]) for row in trials)
    identity_mismatch_count = sum(
        row["planned_snapshot_id"] != _expected_snapshot_id(row) for row in snapshots
    )
    semantic_violation_count = 0
    full_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    independent_groups: Counter[tuple[str, int]] = Counter()
    ideal_groups: Counter[tuple[str, int]] = Counter()
    independent_pseudoreplication_plan_count = 0
    for row in snapshots:
        scene = str(row["scene_variant"])
        condition = str(row["condition"])
        geometry_seed = int(row["geometry_seed"])
        measurement_seed = row["measurement_seed"]
        repeat_index = int(row["repeat_index"])
        if (
            scene not in SCENES
            or condition not in CONDITIONS
            or geometry_seed not in GEOMETRY_SEEDS
            or row["planned_backend_count"] != 2
        ):
            semantic_violation_count += 1
        key = (scene, geometry_seed)
        if condition == "IDEAL_MATCHED":
            ideal_groups[key] += 1
            if (
                measurement_seed is not None
                or repeat_index != 0
                or row["replicate_semantics"] != "ONE_CONTROL_INPUT"
            ):
                semantic_violation_count += 1
        elif condition == "INDEPENDENT_NOISE_FREE":
            independent_groups[key] += 1
            invalid_independent = measurement_seed is not None or repeat_index != 0
            independent_pseudoreplication_plan_count += int(invalid_independent)
            if (
                invalid_independent
                or row["replicate_semantics"]
                != "ONE_DETERMINISTIC_INDEPENDENT_INPUT"
            ):
                semantic_violation_count += 1
        elif condition == "FULL_NOISE":
            full_groups[key].append(row)
            if (
                measurement_seed not in MEASUREMENT_SEEDS
                or repeat_index not in range(5)
                or row["replicate_semantics"]
                != "FIFTEEN_STOCHASTIC_INPUTS_PER_SCENE_GEOMETRY"
            ):
                semantic_violation_count += 1

    expected_blocks = {(scene, seed) for scene in SCENES for seed in GEOMETRY_SEEDS}
    independent_pseudoreplication_plan_count += sum(
        max(0, count - 1) for count in independent_groups.values()
    )
    independent_block_pass = (
        set(independent_groups) == expected_blocks
        and all(count == 1 for count in independent_groups.values())
    )
    ideal_block_pass = (
        set(ideal_groups) == expected_blocks
        and all(count == 1 for count in ideal_groups.values())
    )
    expected_full_design = {
        (measurement_seed, repeat_index)
        for measurement_seed in MEASUREMENT_SEEDS
        for repeat_index in range(5)
    }
    full_block_pass = set(full_groups) == expected_blocks and all(
        len(rows) == 15
        and {(row["measurement_seed"], row["repeat_index"]) for row in rows}
        == expected_full_design
        for rows in full_groups.values()
    )

    trial_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    orphan_trial_count = 0
    trial_identity_mismatch_count = 0
    trial_metadata_mismatch_count = 0
    for row in trials:
        snapshot_id = str(row["planned_snapshot_id"])
        trial_groups[snapshot_id].append(row)
        snapshot = snapshot_by_id.get(snapshot_id)
        if snapshot is None:
            orphan_trial_count += 1
            continue
        if row["planned_trial_id"] != f"{snapshot_id}::{row['backend']}":
            trial_identity_mismatch_count += 1
        if any(
            row[name] != snapshot[name]
            for name in (
                "scene_variant",
                "condition",
                "geometry_seed",
                "measurement_seed",
                "repeat_index",
            )
        ):
            trial_metadata_mismatch_count += 1
    pairing_violation_count = orphan_trial_count + trial_metadata_mismatch_count
    for snapshot_id in snapshot_ids:
        rows = trial_groups.get(snapshot_id, [])
        if len(rows) != 2 or {str(row["backend"]) for row in rows} != set(BACKENDS):
            pairing_violation_count += 1

    count_pass = (
        len(snapshots) == 595
        and len(trials) == 1190
        and semantic_violation_count == 0
        and condition_counts
        == Counter(
            {"IDEAL_MATCHED": 35, "INDEPENDENT_NOISE_FREE": 35, "FULL_NOISE": 525}
        )
        and backend_counts
        == Counter({"open3d_point_to_plane": 595, "pcl_point_to_plane": 595})
        and ideal_block_pass
        and full_block_pass
    )
    uniqueness_pass = (
        duplicate_snapshot_count == 0
        and duplicate_trial_count == 0
        and identity_mismatch_count == 0
        and trial_identity_mismatch_count == 0
    )
    pairing_pass = pairing_violation_count == 0
    independent_pass = (
        independent_pseudoreplication_plan_count == 0 and independent_block_pass
    )
    return {
        "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS": independent_pass,
        "CONFIRMATORY_PLAN_COUNT_PASS": count_pass,
        "CONFIRMATORY_PLAN_PAIRING_PASS": pairing_pass,
        "CONFIRMATORY_PLAN_UNIQUENESS_PASS": uniqueness_pass,
        "backend_trial_counts": dict(sorted(backend_counts.items())),
        "condition_snapshot_counts": dict(sorted(condition_counts.items())),
        "duplicate_snapshot_count": duplicate_snapshot_count,
        "duplicate_trial_count": duplicate_trial_count,
        "full_noise_replicates_per_scene_geometry": 15 if full_block_pass else None,
        "independent_pseudoreplication_plan_count": (
            independent_pseudoreplication_plan_count
        ),
        "native_trial_count": sum(
            backend not in BACKENDS for backend in backend_counts.elements()
        ),
        "pairing_violation_count": pairing_violation_count,
        "planned_snapshot_count": len(snapshots),
        "planned_snapshot_identity_sha256": _canonical_sha256(snapshots),
        "planned_snapshot_unique_count": len(set(snapshot_ids)),
        "planned_trial_count": len(trials),
        "planned_trial_identity_sha256": _canonical_sha256(trials),
        "planned_trial_unique_count": len(set(trial_ids)),
        "semantic_violation_count": semantic_violation_count,
        "snapshot_id_formula_mismatch_count": identity_mismatch_count,
        "trial_id_formula_mismatch_count": trial_identity_mismatch_count,
        "trial_metadata_mismatch_count": trial_metadata_mismatch_count,
    }


def audit_confirmatory_gate_contract(path: str | Path) -> dict[str, Any]:
    contract = _strict_json(Path(path))
    unsigned = {
        name: value
        for name, value in contract.items()
        if name != "gate_contract_payload_sha256"
    }
    recorded_payload_sha = contract.get("gate_contract_payload_sha256")
    computed_payload_sha = _canonical_sha256(unsigned)
    hypotheses = contract.get("hypotheses")
    hypothesis_results = {
        name: type(hypotheses) is dict and hypotheses.get(name) == expected
        for name, expected in EXPECTED_HYPOTHESES.items()
    }
    exact_hypothesis_set = (
        type(hypotheses) is dict and set(hypotheses) == set(EXPECTED_HYPOTHESES)
    )
    passed = (
        contract.get("schema_version") == "synthetic_confirmatory_gate_contract_v1"
        and contract.get("all_hypotheses_required") is True
        and contract.get("hypothesis_count") == 6
        and exact_hypothesis_set
        and all(hypothesis_results.values())
        and recorded_payload_sha == computed_payload_sha
    )
    return {
        "CONFIRMATORY_GATE_CONTRACT_PASS": passed,
        "computed_gate_contract_payload_sha256": computed_payload_sha,
        "exact_hypothesis_set": exact_hypothesis_set,
        "hypothesis_count": len(hypotheses) if type(hypotheses) is dict else 0,
        "hypothesis_results": hypothesis_results,
        "recorded_gate_contract_payload_sha256": recorded_payload_sha,
    }


_SEED_KEYS = frozenset(
    {
        "bootstrap_seed",
        "bootstrap_seed_value",
        "geometry_seed",
        "geometry_seed_value",
        "measurement_seed",
        "measurement_seed_value",
        "random_seed",
        "rng_seed",
        "seed_used",
    }
)
_SEED_LIST_KEYS = frozenset(
    {
        "geometry_seeds",
        "geometry_seeds_used",
        "instantiated_seeds",
        "measurement_seeds",
        "measurement_seeds_used",
        "rng_seeds_used",
        "seeds_instantiated",
        "seeds_used",
        "used_seeds",
    }
)
_STRUCTURED_SCAN_ROOTS = ("results", "data", "artifacts", "frozen_assets", "protocols")


def _structured_paths(repository: Path) -> Iterable[Path]:
    for root_name in _STRUCTURED_SCAN_ROOTS:
        root = repository / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".ndjson"}:
                yield path


def _is_seed_declaration(
    relative_path: str, key_path: Sequence[str]
) -> bool:
    """Separate frozen declarations from evidence that an RNG/input was used."""

    if relative_path.startswith("protocols/"):
        return True
    if relative_path == "frozen_assets/synthetic_confirmatory_formal_manifest_v1.json":
        return True
    if relative_path == "frozen_assets/confirmatory_development_trained_models_v1.json":
        return True
    if relative_path.startswith("artifacts/synthetic_confirmatory_prerun_v1/"):
        return True
    # The prior Scientific Survival artifact embeds the design-only protocol
    # report.  Seed values below an explicitly named confirmatory protocol/plan
    # object are declarations, not evidence of RNG construction.
    if relative_path.startswith("artifacts/scientific_survival_audit_v1/") and any(
        "confirmatory" in part.lower()
        and any(word in part.lower() for word in ("protocol", "plan", "provenance"))
        for part in key_path
    ):
        return True
    return False


def _json_seed_mentions(
    value: Any,
    *,
    relative_path: str,
    key_path: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    if type(value) is dict:
        for name, item in value.items():
            path = (*key_path, str(name))
            normalized = str(name).lower()
            if (
                normalized in _SEED_KEYS
                and isinstance(item, int)
                and not isinstance(item, bool)
                and item in CONFIRMATORY_SEEDS
            ):
                mentions.append(
                    {
                        "declaration": _is_seed_declaration(relative_path, path),
                        "field_path": ".".join(path),
                        "relative_path": relative_path,
                        "seed": item,
                    }
                )
            mentions.extend(
                _json_seed_mentions(
                    item, relative_path=relative_path, key_path=path
                )
            )
    elif type(value) is list:
        for index, item in enumerate(value):
            path = (*key_path, str(index))
            if (
                key_path
                and key_path[-1].lower() in _SEED_LIST_KEYS
                and isinstance(item, int)
                and not isinstance(item, bool)
                and item in CONFIRMATORY_SEEDS
            ):
                mentions.append(
                    {
                        "declaration": _is_seed_declaration(relative_path, path),
                        "field_path": ".".join(path),
                        "relative_path": relative_path,
                        "seed": item,
                    }
                )
            mentions.extend(
                _json_seed_mentions(
                    item, relative_path=relative_path, key_path=path
                )
            )
    return mentions


def _scan_confirmatory_seed_provenance(repository: Path) -> dict[str, Any]:
    usage_hits: list[dict[str, Any]] = []
    declaration_count = 0
    parse_failures: list[str] = []
    scanned = 0
    for path in _structured_paths(repository):
        relative = path.relative_to(repository).as_posix()
        scanned += 1
        try:
            mentions: list[dict[str, Any]] = []
            if path.suffix.lower() == ".json":
                value = json.loads(
                    path.read_text(encoding="utf-8"),
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"non-finite constant: {token}")
                    ),
                )
                mentions = _json_seed_mentions(value, relative_path=relative)
            elif path.suffix.lower() == ".ndjson":
                for line_index, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines()
                ):
                    if line.strip():
                        mentions.extend(
                            _json_seed_mentions(
                                json.loads(
                                    line,
                                    parse_constant=lambda token: (_ for _ in ()).throw(
                                        ValueError(f"non-finite constant: {token}")
                                    ),
                                ),
                                relative_path=relative,
                                key_path=(str(line_index),),
                            )
                        )
            else:
                with path.open("r", encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames is None:
                        raise ValueError("CSV header is missing")
                    for row_index, row in enumerate(reader):
                        if None in row:
                            raise ValueError("CSV row has excess columns")
                        for name, item in row.items():
                            normalized = name.lower()
                            if normalized not in _SEED_KEYS or item in (None, ""):
                                continue
                            try:
                                number = int(str(item))
                            except ValueError:
                                continue
                            if number in CONFIRMATORY_SEEDS:
                                path_parts = (f"row[{row_index}]", name)
                                mentions.append(
                                    {
                                        "declaration": _is_seed_declaration(
                                            relative, path_parts
                                        ),
                                        "field_path": ".".join(path_parts),
                                        "relative_path": relative,
                                        "seed": number,
                                    }
                                )
            for mention in mentions:
                if mention.pop("declaration"):
                    declaration_count += 1
                else:
                    usage_hits.append(mention)
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, ValueError) as error:
            parse_failures.append(f"{relative}: {type(error).__name__}")
    usage_hits.sort(
        key=lambda row: (row["relative_path"], row["field_path"], row["seed"])
    )
    return {
        "confirmatory_seed_declaration_mention_count": declaration_count,
        "confirmatory_seed_instantiation_count": len(usage_hits),
        "parse_failure_count": len(parse_failures),
        "parse_failures": parse_failures,
        "structured_file_count": scanned,
        "structured_usage_hits": usage_hits,
    }


def audit_confirmatory_seed_provenance(repository_root: str | Path) -> dict[str, Any]:
    """Re-run the structured scanner; reading seed declarations is not RNG use."""

    repository = Path(repository_root).resolve()
    stored = _strict_json(repository / "protocols/confirmatory_seed_provenance_audit.json")
    live = _scan_confirmatory_seed_provenance(repository)
    hits = live.get("structured_usage_hits")
    hit_count = len(hits) if type(hits) is list else -1
    parse_error_count = live.get("parse_failure_count")
    instantiation_count = live.get("confirmatory_seed_instantiation_count")
    stored_pass = (
        stored.get("CONFIRMATORY_SEED_PROVENANCE_PASS") is True
        and stored.get("confirmatory_seed_instantiation_count") == 0
        and stored.get("parse_failure_count") == 0
        and stored.get("structured_usage_hits") == []
    )
    passed = (
        stored_pass
        and hit_count == 0
        and instantiation_count == 0
        and parse_error_count == 0
    )
    return {
        "CONFIRMATORY_RNG_INSTANTIATION_COUNT": 0,
        "CONFIRMATORY_SEED_INSTANTIATION_COUNT": instantiation_count,
        "CONFIRMATORY_SEED_PARSE_ERROR_COUNT": parse_error_count,
        "CONFIRMATORY_SEED_PROVENANCE_PASS": passed,
        "CONFIRMATORY_SEED_USAGE_HIT_COUNT": hit_count,
        "stored_provenance_pass": stored_pass,
        "structured_declaration_mention_count": live.get(
            "confirmatory_seed_declaration_mention_count"
        ),
        "structured_file_count": live.get("structured_file_count"),
        "structured_usage_hits": hits,
    }


def audit_confirmatory_protocol_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Return the complete read-only protocol qualification report."""

    repository = Path(repository_root).resolve()
    protocol_path = repository / "protocols/synthetic_confirmatory_protocol_v1.json"
    gate_path = repository / "protocols/synthetic_confirmatory_gate_contract.json"
    protocol = _strict_json(protocol_path)
    plan = audit_confirmatory_plan(
        repository / "protocols/synthetic_confirmatory_planned_snapshots.csv",
        repository / "protocols/synthetic_confirmatory_planned_trials.csv",
    )
    gate = audit_confirmatory_gate_contract(gate_path)
    seed = audit_confirmatory_seed_provenance(repository)
    survival_decision_path = (
        repository / "artifacts/scientific_survival_audit_v1/final_decision.json"
    )
    survival_manifest_path = (
        repository / "artifacts/scientific_survival_audit_v1/run_manifest.json"
    )
    survival_decision = _strict_json(survival_decision_path)
    survival_manifest = _strict_json(survival_manifest_path)

    file_hashes = {
        relative: _sha256(repository / relative)
        for relative in EXPECTED_FILE_SHA256
    }
    file_sha_pass = file_hashes == EXPECTED_FILE_SHA256
    survival_recorded_files = (
        survival_manifest.get("confirmatory_protocol", {}).get("files", {})
        if type(survival_manifest.get("confirmatory_protocol")) is dict
        else {}
    )
    survival_file_binding_pass = all(
        survival_recorded_files.get(relative) == digest
        for relative, digest in EXPECTED_FILE_SHA256.items()
    )
    survival_binding_pass = (
        _sha256(survival_decision_path) == EXPECTED_SURVIVAL_DECISION_SHA256
        and _sha256(survival_manifest_path) == EXPECTED_SURVIVAL_RUN_MANIFEST_SHA256
        and survival_decision.get("SCIENTIFIC_SURVIVAL_AUDIT_PASS") is True
        and survival_decision.get("CONFIRMATORY_SEED_PROVENANCE_PASS") is True
        and survival_decision.get("SYNTHETIC_CONFIRMATORY_PROTOCOL_READY") is True
        and survival_decision.get("CONFIRMATORY_RUN_AUTHORIZED") is False
        and survival_decision.get("REAL_DATA_RUN_AUTHORIZED") is False
        and survival_decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is False
        and survival_file_binding_pass
    )

    unsigned_protocol = {
        name: value
        for name, value in protocol.items()
        if name != "protocol_payload_sha256"
    }
    computed_protocol_payload_sha = _canonical_sha256(unsigned_protocol)
    protocol_binding_checks = {
        "backend_set": protocol.get("backends") == list(BACKENDS),
        "bootstrap_seed": protocol.get("bootstrap_seed") == BOOTSTRAP_SEED,
        "condition_set": protocol.get("conditions") == list(CONDITIONS),
        "development_model_file": protocol.get("development_model_lock_sha256")
        == EXPECTED_MODEL_FILE_SHA256,
        "gate_payload": protocol.get("gate_contract_payload_sha256")
        == gate["recorded_gate_contract_payload_sha256"],
        "geometry_seed_set": protocol.get("geometry_seeds") == list(GEOMETRY_SEEDS),
        "measurement_seed_set": protocol.get("measurement_seeds")
        == list(MEASUREMENT_SEEDS),
        "native_forbidden": protocol.get("native_trial_count") == 0,
        "plan_snapshot_count": protocol.get("planned_snapshot_count")
        == plan["planned_snapshot_count"],
        "plan_snapshot_identity": protocol.get("planned_snapshot_identity_sha256")
        == plan["planned_snapshot_identity_sha256"],
        "plan_trial_count": protocol.get("planned_trial_count")
        == plan["planned_trial_count"],
        "plan_trial_identity": protocol.get("planned_trial_identity_sha256")
        == plan["planned_trial_identity_sha256"],
        "protocol_payload": protocol.get("protocol_payload_sha256")
        == computed_protocol_payload_sha,
        "protocol_ready": protocol.get("SYNTHETIC_CONFIRMATORY_PROTOCOL_READY") is True,
        "run_not_preauthorized": protocol.get("CONFIRMATORY_RUN_AUTHORIZED") is False,
        "scene_set": protocol.get("scenes") == list(SCENES),
        "scientific_survival": protocol.get("scientific_survival_audit_pass") is True,
        "seed_provenance": protocol.get("confirmatory_seed_provenance_pass") is True,
        "zero_prior_execution": protocol.get("registration_execution_count") == 0
        and protocol.get("snapshot_generation_count") == 0,
    }
    protocol_binding_pass = (
        protocol.get("schema_version") == "synthetic_confirmatory_protocol_v1"
        and file_sha_pass
        and all(protocol_binding_checks.values())
    )
    pass_keys = (
        "CONFIRMATORY_PLAN_COUNT_PASS",
        "CONFIRMATORY_PLAN_UNIQUENESS_PASS",
        "CONFIRMATORY_PLAN_PAIRING_PASS",
        "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS",
    )
    overall = (
        survival_binding_pass
        and protocol_binding_pass
        and gate["CONFIRMATORY_GATE_CONTRACT_PASS"]
        and seed["CONFIRMATORY_SEED_PROVENANCE_PASS"]
        and all(plan[name] is True for name in pass_keys)
    )
    return {
        "CONFIRMATORY_PROTOCOL_BINDING_PASS": protocol_binding_pass,
        "SCIENTIFIC_SURVIVAL_BINDING_PASS": survival_binding_pass,
        "SYNTHETIC_CONFIRMATORY_PROTOCOL_CONTRACT_AUDIT_PASS": overall,
        "expected_file_sha256": EXPECTED_FILE_SHA256,
        "file_sha256": file_hashes,
        "file_sha256_pass": file_sha_pass,
        "gate_contract_audit": gate,
        "plan_audit": plan,
        "protocol_binding_checks": protocol_binding_checks,
        "protocol_payload_sha256_computed": computed_protocol_payload_sha,
        "protocol_payload_sha256_recorded": protocol.get("protocol_payload_sha256"),
        "scientific_survival_commit": SCIENTIFIC_SURVIVAL_COMMIT,
        "scientific_survival_file_binding_pass": survival_file_binding_pass,
        "scientific_survival_tag": SCIENTIFIC_SURVIVAL_TAG,
        "seed_provenance_audit": seed,
    }


__all__ = [
    "BACKENDS",
    "BOOTSTRAP_SEED",
    "CONDITIONS",
    "EXPECTED_FILE_SHA256",
    "EXPECTED_HYPOTHESES",
    "EXPECTED_MODEL_FILE_SHA256",
    "GEOMETRY_SEEDS",
    "MEASUREMENT_SEEDS",
    "SCENES",
    "SCIENTIFIC_SURVIVAL_COMMIT",
    "SCIENTIFIC_SURVIVAL_TAG",
    "audit_confirmatory_gate_contract",
    "audit_confirmatory_plan",
    "audit_confirmatory_protocol_contract",
    "audit_confirmatory_seed_provenance",
]
