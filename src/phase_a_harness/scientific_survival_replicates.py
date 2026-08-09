"""Read-only replicate and robustness calculations for the survival audit.

The functions in this module are deliberately side-effect free.  They accept
already frozen snapshot/trial/analysis rows and return JSON- and CSV-ready
records.  A replicate is a unique ``(source_checksum, target_checksum)`` pair;
the two registration backends never create additional replicates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import product
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from scipy.spatial import cKDTree


SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
GEOMETRY_SEEDS = (1850310744, 1957656152, 1334931069)
MEASUREMENT_SEEDS = (217775206, 1664898153)
REPEAT_INDICES = (0, 1, 2, 3, 4)
NONIDEAL_CONDITIONS = (
    "INDEPENDENT_NOISE_FREE",
    "SCAN_NOISE_ONLY",
    "MAP_NOISE_ONLY",
    "DROPOUT_ONLY",
    "FULL_NOISE",
)
PRIMARY_CONDITIONS = ("INDEPENDENT_NOISE_FREE", "FULL_NOISE")
BACKENDS = ("Open3D", "PCL")

REPEATABLE_TERM = "REPEATABLE_SYSTEMATIC_OFFSET"
DETERMINISTIC_TERM = "DETERMINISTIC_ZERO_INITIALIZATION_DISPLACEMENT"
LIMITED_TERM = "CONSISTENT_DISPLACEMENT_WITH_LIMITED_REPLICATION"

_NOISE_SIDES = {
    "INDEPENDENT_NOISE_FREE": (),
    "SCAN_NOISE_ONLY": ("source",),
    "MAP_NOISE_ONLY": ("target",),
    "DROPOUT_ONLY": (),
    "FULL_NOISE": ("source", "target"),
}
_DROPOUT_SIDES = {
    "INDEPENDENT_NOISE_FREE": (),
    "SCAN_NOISE_ONLY": (),
    "MAP_NOISE_ONLY": (),
    "DROPOUT_ONLY": ("source",),
    "FULL_NOISE": ("source",),
}


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def classify_replicate_count(unique_pair_count: int) -> str:
    """Apply the prospectively fixed 1 / 2--7 / 8--10 classification."""

    count = int(unique_pair_count)
    if count < 1 or count > 10:
        raise ValueError("unique source-target pair count must be in [1, 10]")
    if count == 1:
        return "DETERMINISTIC_SINGLE_INPUT"
    if count <= 7:
        return "PARTIAL_REPLICATION"
    return "FULL_REPLICATION"


def _authorized_term(
    effective_count: int, unique_by_measurement_seed: Mapping[int, int]
) -> tuple[bool, str]:
    if effective_count == 1:
        return False, DETERMINISTIC_TERM
    balanced = bool(
        len(unique_by_measurement_seed) == 2
        and min(unique_by_measurement_seed.values()) >= 4
    )
    if effective_count >= 8 and balanced:
        return True, REPEATABLE_TERM
    return False, LIMITED_TERM


def _component_signature(
    row: Mapping[str, Any], kind: str, sides: Sequence[str], absent: str
) -> tuple[str, str, str | None]:
    declared = row.get(f"{kind}_checksum")
    if declared is not None:
        return (
            str(declared),
            str(
                row.get(
                    f"{kind}_signature_basis",
                    "PERSISTED_OR_OBSERVED_COMPONENT_SIGNATURE",
                )
            ),
            row.get(f"{kind}_signature_limitation"),
        )
    if not sides:
        return absent, "CONDITION_HAS_NO_ACTIVE_COMPONENT", None
    components: dict[str, Any] = {}
    for side in sides:
        components[f"{side}_checksum"] = str(row[f"{side}_checksum"])
        point_count = row.get(f"{side}_point_count")
        if point_count is not None:
            components[f"{side}_point_count"] = int(point_count)
    return (
        _digest(components),
        "AFFECTED_SIDE_FINAL_INPUT_CHECKSUM_PROXY",
        "component-level generator checksum was not persisted; proxy proves final input variation but not the latent noise field or dropout mask",
    )


def _exact_subsequence_indices(
    baseline: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    indices: list[int] = []
    cursor = 0
    for index, row in enumerate(baseline):
        if cursor < observed.shape[0] and np.array_equal(row, observed[cursor]):
            indices.append(index)
            cursor += 1
    if cursor != observed.shape[0]:
        raise ValueError("dropout-only source is not an exact baseline subsequence")
    return np.asarray(indices, dtype=np.int64)


def derive_observed_component_signatures(
    *,
    condition: str,
    source_points: np.ndarray,
    target_points: np.ndarray,
    independent_source_points: np.ndarray,
    independent_target_points: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct observable noise/dropout signatures from frozen arrays.

    These hashes are not claimed to equal the generator's unpersisted hashes.
    They bind observable float32 deltas and masks relative to the corresponding
    ``INDEPENDENT_NOISE_FREE`` input.  FULL_NOISE scan indices are accepted only
    when nearest-baseline matches are unique and strictly order preserving.
    """

    if condition not in NONIDEAL_CONDITIONS:
        raise ValueError(f"unknown non-IDEAL condition: {condition}")
    source = np.asarray(source_points)
    target = np.asarray(target_points)
    base_source = np.asarray(independent_source_points)
    base_target = np.asarray(independent_target_points)
    for label, array in (
        ("source", source),
        ("target", target),
        ("independent source", base_source),
        ("independent target", base_target),
    ):
        if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} array must be finite N x 3")
    if target.shape != base_target.shape:
        raise ValueError("map dropout is forbidden but target shape changed")

    if condition in ("INDEPENDENT_NOISE_FREE", "SCAN_NOISE_ONLY", "MAP_NOISE_ONLY"):
        if source.shape != base_source.shape:
            raise ValueError("non-dropout source shape changed")
        source_indices = np.arange(base_source.shape[0], dtype=np.int64)
        dropout_basis = "EXACT_ALL_TRUE_MASK_FROM_EQUAL_SHAPE_FROZEN_ARRAYS"
        dropout_limitation = None
        nearest_max = 0.0
        nearest_margin = None
    elif condition == "DROPOUT_ONLY":
        source_indices = _exact_subsequence_indices(base_source, source)
        dropout_basis = "EXACT_ORDERED_SUBSEQUENCE_MASK_FROM_FROZEN_ARRAYS"
        dropout_limitation = None
        nearest_max = 0.0
        nearest_margin = None
    else:
        distances, indices = cKDTree(np.asarray(base_source, dtype=np.float64)).query(
            np.asarray(source, dtype=np.float64), k=2
        )
        source_indices = np.asarray(indices[:, 0], dtype=np.int64)
        nearest_max = float(np.max(distances[:, 0], initial=0.0))
        nearest_margin = float(np.min(distances[:, 1] - distances[:, 0], initial=np.inf))
        reconstruction_pass = not (
            source_indices.size != np.unique(source_indices).size
            or np.any(np.diff(source_indices) <= 0)
            or not np.all(np.isfinite(distances))
            or np.any(distances[:, 1] <= distances[:, 0])
        )
        if not reconstruction_pass:
            observed_target_delta = np.asarray(target - base_target, dtype="<f4")
            return {
                "dropout_checksum": _digest(
                    {
                        "final_source_array_proxy": _array_digest(source),
                        "retained_source_count": int(source.shape[0]),
                    }
                ),
                "dropout_kept_source_count": int(source.shape[0]),
                "dropout_mask_reconstruction_pass": False,
                "dropout_signature_basis": (
                    "FINAL_SOURCE_ARRAY_CHECKSUM_PROXY_MASK_UNRESOLVED"
                ),
                "dropout_signature_limitation": (
                    "FULL_NOISE noise prevented a unique strictly ordered nearest-baseline mask reconstruction; signature proves final source variation, not mask identity"
                ),
                "full_noise_nearest_index_max_distance_m": nearest_max,
                "full_noise_nearest_index_min_margin_m": nearest_margin,
                "noise_checksum": _digest(
                    {
                        "final_source_array_proxy": _array_digest(source),
                        "target_delta": _array_digest(observed_target_delta),
                    }
                ),
                "noise_signature_basis": (
                    "FULL_NOISE_FINAL_SOURCE_ARRAY_AND_TARGET_DELTA_PROXY"
                ),
                "noise_signature_limitation": (
                    "scan noise and dropout cannot be separated for this frozen array; target delta is observable but source signature is combined"
                ),
                "observed_component_signature_pass": True,
            }
        dropout_basis = (
            "UNIQUE_STRICTLY_ORDERED_NEAREST_BASELINE_INDEX_MASK_FROM_FROZEN_ARRAYS"
        )
        dropout_limitation = (
            "FULL_NOISE scan coordinates include noise; mask is an observed-array nearest-index reconstruction, not the unpersisted generator mask checksum"
        )

    source_mask = np.zeros(base_source.shape[0], dtype=np.bool_)
    source_mask[source_indices] = True
    target_mask = np.ones(base_target.shape[0], dtype=np.bool_)
    matched_base_source = base_source[source_indices]
    observed_source_delta = np.asarray(source - matched_base_source, dtype="<f4")
    observed_target_delta = np.asarray(target - base_target, dtype="<f4")

    if condition in ("INDEPENDENT_NOISE_FREE", "DROPOUT_ONLY"):
        if not np.array_equal(source, matched_base_source) or not np.array_equal(
            target, base_target
        ):
            raise ValueError("noise-free condition differs from matched frozen baseline")
        # The generator formed its zero noise arrays before dropout.
        signature_source_delta = np.zeros_like(base_source, dtype="<f4")
        signature_target_delta = np.zeros_like(base_target, dtype="<f4")
        noise_basis = "EXACT_ZERO_PRE_DROPOUT_NOISE_FROM_FROZEN_CONDITION"
        noise_limitation = None
    else:
        if condition == "SCAN_NOISE_ONLY" and not np.array_equal(target, base_target):
            raise ValueError("SCAN_NOISE_ONLY unexpectedly changed target")
        if condition == "MAP_NOISE_ONLY" and not np.array_equal(source, base_source):
            raise ValueError("MAP_NOISE_ONLY unexpectedly changed source")
        signature_source_delta = observed_source_delta
        signature_target_delta = observed_target_delta
        noise_basis = "OBSERVED_FLOAT32_DELTA_TO_MATCHED_INDEPENDENT_BASELINE"
        noise_limitation = (
            "hash covers persisted float32 observable deltas; it is not the unpersisted pre-cast generator noise checksum"
        )
        if condition == "FULL_NOISE":
            noise_limitation += (
                "; scan delta is available only at points retained by the reconstructed dropout mask"
            )

    noise_checksum = _digest(
        {
            "source_delta": _array_digest(signature_source_delta),
            "target_delta": _array_digest(signature_target_delta),
        }
    )
    dropout_checksum = _digest(
        {
            "source_mask": _array_digest(source_mask),
            "target_mask": _array_digest(target_mask),
        }
    )
    return {
        "dropout_checksum": dropout_checksum,
        "dropout_kept_source_count": int(source_mask.sum()),
        "dropout_mask_reconstruction_pass": True,
        "dropout_signature_basis": dropout_basis,
        "dropout_signature_limitation": dropout_limitation,
        "full_noise_nearest_index_max_distance_m": nearest_max,
        "full_noise_nearest_index_min_margin_m": nearest_margin,
        "noise_checksum": noise_checksum,
        "noise_signature_basis": noise_basis,
        "noise_signature_limitation": noise_limitation,
        "observed_component_signature_pass": True,
    }


def attach_observed_component_signatures(
    snapshot_rows: Iterable[Mapping[str, Any]],
    array_loader: Callable[[Mapping[str, Any]], Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Attach reconstructed signatures using a caller-supplied read-only loader."""

    rows = [dict(row) for row in snapshot_rows]
    groups: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["scene_variant"]),
            int(row["geometry_seed"]),
            int(row["measurement_seed"]),
            int(row["repeat_index"]),
        )
        condition = str(row["condition"])
        if condition in groups[key]:
            raise ValueError("duplicate condition in snapshot design group")
        groups[key][condition] = row
    enriched: list[dict[str, Any]] = []
    for key, conditions in sorted(groups.items()):
        if set(conditions) != set(NONIDEAL_CONDITIONS):
            raise ValueError(f"component-signature design group is incomplete: {key}")
        baseline_arrays = array_loader(conditions["INDEPENDENT_NOISE_FREE"])
        for condition in NONIDEAL_CONDITIONS:
            row = conditions[condition]
            arrays = baseline_arrays if condition == "INDEPENDENT_NOISE_FREE" else array_loader(row)
            signatures = derive_observed_component_signatures(
                condition=condition,
                source_points=arrays["source"],
                target_points=arrays["target"],
                independent_source_points=baseline_arrays["source"],
                independent_target_points=baseline_arrays["target"],
            )
            enriched.append({**row, **signatures})
    return enriched


def audit_replicate_uniqueness(
    snapshot_rows: Iterable[Mapping[str, Any]], *, strict_contract: bool = True
) -> dict[str, Any]:
    """Audit non-IDEAL snapshot replication at scene/geometry/condition grain."""

    rows = [dict(row) for row in snapshot_rows]
    required = {
        "snapshot_id",
        "scene_variant",
        "geometry_seed",
        "measurement_seed",
        "repeat_index",
        "condition",
        "source_checksum",
        "target_checksum",
    }
    seen_ids: set[str] = set()
    cells: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"snapshot row is missing fields: {sorted(missing)}")
        snapshot_id = str(row["snapshot_id"])
        if snapshot_id in seen_ids:
            raise ValueError(f"duplicate snapshot identity: {snapshot_id}")
        seen_ids.add(snapshot_id)
        condition = str(row["condition"])
        if condition not in NONIDEAL_CONDITIONS:
            raise ValueError(f"unexpected non-IDEAL condition: {condition}")
        scene = str(row["scene_variant"])
        geometry_seed = int(row["geometry_seed"])
        measurement_seed = int(row["measurement_seed"])
        repeat_index = int(row["repeat_index"])
        row.update(
            scene_variant=scene,
            geometry_seed=geometry_seed,
            measurement_seed=measurement_seed,
            repeat_index=repeat_index,
            condition=condition,
        )
        cells[(scene, geometry_seed, condition)].append(row)

    expected_cells = set(product(SCENES, GEOMETRY_SEEDS, NONIDEAL_CONDITIONS))
    if strict_contract and set(cells) != expected_cells:
        raise ValueError("replicate audit cell inventory differs from frozen 105-cell plan")

    cell_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    structural_mismatch_count = 0
    for (scene, geometry_seed, condition), observations in sorted(cells.items()):
        pairs = [
            (str(row["source_checksum"]), str(row["target_checksum"]))
            for row in observations
        ]
        source_checksums = {pair[0] for pair in pairs}
        target_checksums = {pair[1] for pair in pairs}
        unique_pairs = set(pairs)
        by_measurement: dict[int, set[tuple[str, str]]] = defaultdict(set)
        by_repeat: dict[int, set[tuple[str, str]]] = defaultdict(set)
        noise_signatures: set[str] = set()
        dropout_signatures: set[str] = set()
        noise_bases: set[str] = set()
        dropout_bases: set[str] = set()
        signature_limitations: set[str] = set()
        design_keys: list[tuple[int, int]] = []
        for row, pair in zip(observations, pairs, strict=True):
            by_measurement[int(row["measurement_seed"])].add(pair)
            by_repeat[int(row["repeat_index"])].add(pair)
            design_keys.append((int(row["measurement_seed"]), int(row["repeat_index"])))
            noise_signature, noise_basis, noise_limitation = _component_signature(
                row, "noise", _NOISE_SIDES[condition], "NO_NOISE"
            )
            dropout_signature, dropout_basis, dropout_limitation = _component_signature(
                row, "dropout", _DROPOUT_SIDES[condition], "NO_DROPOUT"
            )
            noise_signatures.add(noise_signature)
            dropout_signatures.add(dropout_signature)
            noise_bases.add(noise_basis)
            dropout_bases.add(dropout_basis)
            signature_limitations.update(
                str(value)
                for value in (noise_limitation, dropout_limitation)
                if value is not None
            )
        actual_count = len(observations)
        expected_design = set(product(MEASUREMENT_SEEDS, REPEAT_INDICES))
        design_pass = bool(
            actual_count == 10
            and len(design_keys) == len(set(design_keys))
            and set(design_keys) == expected_design
        )
        structural_mismatch_count += int(not design_pass)
        measurement_counts = {
            seed: len(by_measurement.get(seed, set())) for seed in MEASUREMENT_SEEDS
        }
        repeat_counts = {
            repeat: len(by_repeat.get(repeat, set())) for repeat in REPEAT_INDICES
        }
        measurement_changes_input = len(
            {frozenset(by_measurement.get(seed, set())) for seed in MEASUREMENT_SEEDS}
        ) > 1
        repeat_changes_input = len(
            {frozenset(by_repeat.get(repeat, set())) for repeat in REPEAT_INDICES}
        ) > 1
        effective = len(unique_pairs)
        authorized, term = _authorized_term(effective, measurement_counts)
        dropout_reconstruction_values = [
            row.get("dropout_mask_reconstruction_pass") for row in observations
        ]
        cell_rows.append(
            {
                "actual_observation_count": actual_count,
                "authorized_term": term,
                "condition": condition,
                "duplicate_pair_count": actual_count - effective,
                "duplicate_source_count": actual_count - len(source_checksums),
                "duplicate_target_count": actual_count - len(target_checksums),
                "dropout_signature_basis": json.dumps(
                    sorted(dropout_bases), separators=(",", ":")
                ),
                "dropout_mask_reconstruction_failure_count": sum(
                    value is False for value in dropout_reconstruction_values
                ),
                "dropout_mask_reconstruction_pass_count": sum(
                    value is True for value in dropout_reconstruction_values
                ),
                "dropout_mask_reconstruction_unknown_count": sum(
                    value is None for value in dropout_reconstruction_values
                ),
                "effective_replicate_count": effective,
                "geometry_seed": geometry_seed,
                "measurement_seed_changes_input": measurement_changes_input,
                "noise_signature_basis": json.dumps(
                    sorted(noise_bases), separators=(",", ":")
                ),
                "planned_observation_count": 10,
                "repeat_index_changes_input": repeat_changes_input,
                "replicate_class": classify_replicate_count(effective),
                "scene_variant": scene,
                "signature_limitation": json.dumps(
                    sorted(signature_limitations), separators=(",", ":")
                )
                if signature_limitations
                else None,
                "snapshot_design_complete": design_pass,
                "systematic_offset_claim_authorized": authorized,
                "unique_dropout_checksum_count": len(dropout_signatures),
                "unique_noise_checksum_count": len(noise_signatures),
                "unique_pair_count_by_measurement_seed": json.dumps(
                    measurement_counts, sort_keys=True, separators=(",", ":")
                ),
                "unique_pair_count_by_repeat": json.dumps(
                    repeat_counts, sort_keys=True, separators=(",", ":")
                ),
                "unique_source_checksum_count": len(source_checksums),
                "unique_source_target_pair_count": effective,
                "unique_target_checksum_count": len(target_checksums),
            }
        )
        for seed in MEASUREMENT_SEEDS:
            measurement_rows.append(
                {
                    "condition": condition,
                    "geometry_seed": geometry_seed,
                    "measurement_seed": seed,
                    "measurement_seed_changes_input": measurement_changes_input,
                    "scene_variant": scene,
                    "unique_pair_count": measurement_counts[seed],
                    "wording_minimum_contribution_pass": measurement_counts[seed] >= 4,
                }
            )
        for repeat in REPEAT_INDICES:
            repeat_rows.append(
                {
                    "condition": condition,
                    "geometry_seed": geometry_seed,
                    "repeat_index": repeat,
                    "repeat_index_changes_input": repeat_changes_input,
                    "scene_variant": scene,
                    "unique_pair_count": repeat_counts[repeat],
                }
            )

    condition_rows: list[dict[str, Any]] = []
    for condition in NONIDEAL_CONDITIONS:
        selected = [row for row in cell_rows if row["condition"] == condition]
        classes = Counter(row["replicate_class"] for row in selected)
        counts = [int(row["effective_replicate_count"]) for row in selected]
        authorizations = [bool(row["systematic_offset_claim_authorized"]) for row in selected]
        if selected and all(row["authorized_term"] == DETERMINISTIC_TERM for row in selected):
            condition_term = DETERMINISTIC_TERM
        elif selected and all(authorizations):
            condition_term = REPEATABLE_TERM
        else:
            condition_term = LIMITED_TERM
        condition_rows.append(
            {
                "AUTHORIZED_TERM": condition_term,
                "SYSTEMATIC_OFFSET_CLAIM_AUTHORIZED_BY_CONDITION": bool(
                    selected and all(authorizations)
                ),
                "actual_observation_count": sum(
                    int(row["actual_observation_count"]) for row in selected
                ),
                "cell_count": len(selected),
                "condition": condition,
                "deterministic_single_input_cell_count": classes[
                    "DETERMINISTIC_SINGLE_INPUT"
                ],
                "effective_replicate_count_max": max(counts) if counts else None,
                "effective_replicate_count_median": float(median(counts))
                if counts
                else None,
                "effective_replicate_count_min": min(counts) if counts else None,
                "effective_replicate_count_sum": sum(counts),
                "full_replication_cell_count": classes["FULL_REPLICATION"],
                "partial_replication_cell_count": classes["PARTIAL_REPLICATION"],
                "planned_observation_count": 210,
            }
        )

    class_totals = Counter(row["replicate_class"] for row in cell_rows)
    exact_inventory = bool(
        len(rows) == 1050
        and len(cell_rows) == 105
        and structural_mismatch_count == 0
        and set(cells) == expected_cells
    )
    active_dropout_cells = [
        row for row in cell_rows if row["condition"] in ("DROPOUT_ONLY", "FULL_NOISE")
    ]
    reconstruction_failures = sum(
        row["dropout_mask_reconstruction_failure_count"] for row in cell_rows
    )
    reconstruction_unknown = sum(
        row["dropout_mask_reconstruction_unknown_count"] for row in active_dropout_cells
    )
    return {
        "DROPOUT_INPUT_VARIATION_OBSERVED": bool(
            active_dropout_cells
            and all(row["unique_dropout_checksum_count"] == 10 for row in active_dropout_cells)
        ),
        "DROPOUT_MASK_IDENTITY_FULLY_RECONSTRUCTED": bool(
            active_dropout_cells
            and reconstruction_failures == 0
            and reconstruction_unknown == 0
        ),
        "PSEUDOREPLICATION_RISK_IDENTIFIED": any(
            int(row["effective_replicate_count"])
            < int(row["planned_observation_count"])
            for row in cell_rows
        ),
        "REPLICATE_UNIQUENESS_AUDIT_PASS": exact_inventory
        if strict_contract
        else structural_mismatch_count == 0,
        "cell_rows": cell_rows,
        "component_signature_limitation_cell_count": sum(
            row["signature_limitation"] is not None for row in cell_rows
        ),
        "component_signature_proxy_cell_count": sum(
            "PROXY" in row["noise_signature_basis"] + row["dropout_signature_basis"]
            for row in cell_rows
        ),
        "condition_rows": condition_rows,
        "deterministic_single_input_cell_count": class_totals[
            "DETERMINISTIC_SINGLE_INPUT"
        ],
        "measurement_seed_rows": measurement_rows,
        "partial_replication_cell_count": class_totals["PARTIAL_REPLICATION"],
        "full_replication_cell_count": class_totals["FULL_REPLICATION"],
        "dropout_mask_reconstruction_failure_count": reconstruction_failures,
        "dropout_mask_reconstruction_unknown_count": reconstruction_unknown,
        "repeat_index_rows": repeat_rows,
        "snapshot_count": len(rows),
        "structural_mismatch_count": structural_mismatch_count,
    }


def evaluate_long_corridor_systematic_offset(
    cell_rows: Iterable[Mapping[str, Any]],
    systematic_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the stricter FULL_NOISE wording gate without rewriting history."""

    replicate = {
        (str(row["scene_variant"]), int(row["geometry_seed"]), str(row["condition"])): row
        for row in cell_rows
    }
    output: list[dict[str, Any]] = []
    for row in systematic_rows:
        if (
            row.get("scene_variant") != "LONG_CORRIDOR"
            or row.get("condition") not in PRIMARY_CONDITIONS
            or row.get("backend") not in BACKENDS
        ):
            continue
        key = (
            "LONG_CORRIDOR",
            int(row["geometry_seed"]),
            str(row["condition"]),
        )
        if key not in replicate:
            raise ValueError(f"missing replicate audit cell for {key}")
        effective = int(replicate[key]["effective_replicate_count"])
        offset = _finite_float(
            row["systematic_translation_offset_m"], "systematic translation offset"
        )
        repeatability = _finite_float(
            row["translation_repeatability_rms_m"], "repeatability RMS"
        )
        fraction = _finite_float(
            row["systematic_fraction_translation"], "systematic fraction"
        )
        concentration = _finite_float(
            row["translation_direction_concentration"], "direction concentration"
        )
        condition = str(row["condition"])
        qualifies = bool(
            condition == "FULL_NOISE"
            and effective >= 8
            and offset >= 0.005
            and fraction >= 0.60
        )
        authorized, term = (
            bool(replicate[key]["systematic_offset_claim_authorized"]),
            str(replicate[key]["authorized_term"]),
        )
        output.append(
            {
                "authorized_term": term,
                "backend": str(row["backend"]),
                "condition": condition,
                "direction_concentration": concentration,
                "effective_replicate_count": effective,
                "full_noise_geometry_gate_pass": qualifies,
                "geometry_seed": int(row["geometry_seed"]),
                "repeatability_rms_m": repeatability,
                "systematic_fraction_translation": fraction,
                "systematic_offset_claim_authorized_by_replication": authorized,
                "systematic_translation_offset_m": offset,
            }
        )
    expected = set(product(BACKENDS, PRIMARY_CONDITIONS, GEOMETRY_SEEDS))
    observed = {
        (row["backend"], row["condition"], row["geometry_seed"]) for row in output
    }
    if observed != expected or len(output) != len(expected):
        raise ValueError("LONG_CORRIDOR systematic audit inventory is incomplete")
    backend_rows: list[dict[str, Any]] = []
    for backend in BACKENDS:
        selected = [
            row
            for row in output
            if row["backend"] == backend and row["condition"] == "FULL_NOISE"
        ]
        qualifying = sum(bool(row["full_noise_geometry_gate_pass"]) for row in selected)
        fraction_median = float(
            median(float(row["systematic_fraction_translation"]) for row in selected)
        )
        backend_rows.append(
            {
                "backend": backend,
                "geometry_group_count": 3,
                "qualifying_geometry_group_count": qualifying,
                "systematic_fraction_median": fraction_median,
                "systematic_offset_full_noise_backend_pass": bool(
                    qualifying >= 2 and fraction_median >= 0.70
                ),
            }
        )
    independent_deterministic = all(
        row["authorized_term"] == DETERMINISTIC_TERM
        for row in output
        if row["condition"] == "INDEPENDENT_NOISE_FREE"
    )
    full_pass = all(
        row["systematic_offset_full_noise_backend_pass"] for row in backend_rows
    )
    return {
        "INDEPENDENT_NOISE_FREE_AUTHORIZED_TERM": DETERMINISTIC_TERM
        if independent_deterministic
        else LIMITED_TERM,
        "SYSTEMATIC_OFFSET_CLAIM_SCOPE": "LONG_CORRIDOR_FULL_NOISE"
        if full_pass
        else "NOT_AUTHORIZED",
        "SYSTEMATIC_OFFSET_FULL_NOISE_PASS": full_pass,
        "backend_rows": backend_rows,
        "geometry_rows": sorted(
            output,
            key=lambda row: (
                row["backend"], row["condition"], row["geometry_seed"]
            ),
        ),
    }


def collapse_trials_to_unique_inputs(
    trial_rows: Iterable[Mapping[str, Any]],
    snapshot_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Give each unique input pair one weight within its scientific cell."""

    snapshots: dict[str, tuple[str, str]] = {}
    for row in snapshot_rows:
        snapshot_id = str(row["snapshot_id"])
        if snapshot_id in snapshots:
            raise ValueError(f"duplicate snapshot identity: {snapshot_id}")
        snapshots[snapshot_id] = (
            str(row["source_checksum"]),
            str(row["target_checksum"]),
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    trial_ids: set[str] = set()
    for source in trial_rows:
        row = dict(source)
        trial_id = str(row.get("planned_trial_id", ""))
        if trial_id:
            if trial_id in trial_ids:
                raise ValueError(f"duplicate trial identity: {trial_id}")
            trial_ids.add(trial_id)
        if row.get("solver_failure") is True or row.get("finite_output") is False:
            continue
        if row.get("failure_classification") not in (None, "NONE"):
            continue
        snapshot_id = str(row["snapshot_id"])
        if snapshot_id not in snapshots:
            raise ValueError(f"trial references unknown snapshot: {snapshot_id}")
        error = _finite_float(row["translation_error_m"], "translation error")
        if error < 0.0:
            raise ValueError("translation error must be nonnegative")
        pair = snapshots[snapshot_id]
        key = (
            str(row["backend"]),
            str(row["condition"]),
            str(row["scene_variant"]),
            int(row["geometry_seed"]),
            pair[0],
            pair[1],
        )
        grouped[key].append({**row, "translation_error_m": error})
    collapsed: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        backend, condition, scene, geometry_seed, source_sha, target_sha = key
        turnovers = [
            _finite_float(row["correspondence_turnover"], "correspondence turnover")
            for row in members
            if row.get("correspondence_turnover") is not None
            and row.get("common_association_valid", True) is True
        ]
        collapsed.append(
            {
                "backend": backend,
                "condition": condition,
                "correspondence_turnover": float(median(turnovers))
                if turnovers
                else None,
                "geometry_seed": geometry_seed,
                "scene_variant": scene,
                "source_checksum": source_sha,
                "target_checksum": target_sha,
                "translation_error_m": float(
                    median(row["translation_error_m"] for row in members)
                ),
                "trial_count_collapsed": len(members),
                "unique_input_id": _digest(
                    {"source_checksum": source_sha, "target_checksum": target_sha}
                ),
            }
        )
    return collapsed


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    ranked_x = rankdata(x, method="average")
    ranked_y = rankdata(y, method="average")
    if float(np.ptp(ranked_x)) == 0.0 or float(np.ptp(ranked_y)) == 0.0:
        return None
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def evaluate_unique_unit_primary_scene_effect(
    trial_rows: Iterable[Mapping[str, Any]],
    snapshot_rows: Iterable[Mapping[str, Any]],
    *,
    original_trial_weighted_rows: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Recompute the four preregistered scene comparisons after de-duplication."""

    collapsed = collapse_trials_to_unique_inputs(trial_rows, snapshot_rows)
    cell_values: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in collapsed:
        if row["condition"] in PRIMARY_CONDITIONS and row["scene_variant"] in (
            "LONG_CORRIDOR",
            "GEOMETRY_RICH_ROOM",
        ):
            cell_values[
                (
                    row["backend"],
                    row["condition"],
                    row["scene_variant"],
                    row["geometry_seed"],
                )
            ].append(float(row["translation_error_m"]))
    original = {
        (str(row["backend"]), str(row["condition"])): row
        for row in original_trial_weighted_rows
    }
    rows: list[dict[str, Any]] = []
    for backend, condition in product(BACKENDS, PRIMARY_CONDITIONS):
        geometry_rows: list[dict[str, Any]] = []
        for seed in GEOMETRY_SEEDS:
            corridor = cell_values.get((backend, condition, "LONG_CORRIDOR", seed), [])
            rich = cell_values.get((backend, condition, "GEOMETRY_RICH_ROOM", seed), [])
            if not corridor or not rich:
                raise ValueError("unique-unit primary scene cell is incomplete")
            corridor_median = float(median(corridor))
            rich_median = float(median(rich))
            geometry_rows.append(
                {
                    "corridor_median_m": corridor_median,
                    "geometry_seed": seed,
                    "paired_win": corridor_median > rich_median,
                    "rich_median_m": rich_median,
                }
            )
        corridor_unique_values = [
            value
            for seed in GEOMETRY_SEEDS
            for value in cell_values[(backend, condition, "LONG_CORRIDOR", seed)]
        ]
        rich_unique_values = [
            value
            for seed in GEOMETRY_SEEDS
            for value in cell_values[(backend, condition, "GEOMETRY_RICH_ROOM", seed)]
        ]
        corridor_median = float(median(corridor_unique_values))
        rich_median = float(median(rich_unique_values))
        ratio = corridor_median / rich_median if rich_median > 0.0 else None
        difference = corridor_median - rich_median
        win_count = sum(bool(row["paired_win"]) for row in geometry_rows)
        gate = bool(
            ratio is not None
            and ratio >= 5.0
            and difference >= 0.005
            and win_count >= 2
        )
        previous = original.get((backend, condition))
        rows.append(
            {
                "absolute_median_difference_m": difference,
                "backend": backend,
                "condition": condition,
                "corridor_median_m": corridor_median,
                "geometry_level_paired_win_count": win_count,
                "geometry_level_paired_win_rate": win_count / 3.0,
                "geometry_rows_json": json.dumps(
                    geometry_rows, sort_keys=True, separators=(",", ":")
                ),
                "original_absolute_median_difference_m": (
                    float(previous["absolute_median_difference_m"])
                    if previous is not None
                    else None
                ),
                "original_paired_win_rate": (
                    float(previous["paired_win_rate"])
                    if previous is not None
                    else None
                ),
                "original_weak_rich_median_ratio": (
                    float(previous["weak_rich_median_ratio"])
                    if previous is not None
                    else None
                ),
                "primary_scene_effect_unique_unit_cell_pass": gate,
                "rich_median_m": rich_median,
                "unique_input_corridor_count": len(corridor_unique_values),
                "unique_input_rich_count": len(rich_unique_values),
                "weak_rich_median_ratio": ratio,
            }
        )
    return {
        "PRIMARY_SCENE_EFFECT_UNIQUE_UNIT_PASS": all(
            row["primary_scene_effect_unique_unit_cell_pass"] for row in rows
        ),
        "rows": rows,
    }


def evaluate_cross_backend_unique_unit(
    trial_rows: Iterable[Mapping[str, Any]],
    snapshot_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute scene rankings with one weight per unique input and backend."""

    collapsed = collapse_trials_to_unique_inputs(trial_rows, snapshot_rows)
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in collapsed:
        if row["condition"] in NONIDEAL_CONDITIONS:
            values[(row["condition"], row["scene_variant"], row["backend"])].append(
                float(row["translation_error_m"])
            )
    scene_rows: list[dict[str, Any]] = []
    for condition, scene in product(NONIDEAL_CONDITIONS, SCENES):
        medians: dict[str, float] = {}
        for backend in BACKENDS:
            selected = values.get((condition, scene, backend), [])
            if not selected:
                raise ValueError("cross-backend unique-unit scene inventory is incomplete")
            medians[backend] = float(median(selected))
        scene_rows.append(
            {
                "condition": condition,
                "open3d_translation_median_m": medians["Open3D"],
                "pcl_translation_median_m": medians["PCL"],
                "scene_variant": scene,
            }
        )
    condition_rows: list[dict[str, Any]] = []
    for condition in NONIDEAL_CONDITIONS:
        selected = [row for row in scene_rows if row["condition"] == condition]
        rho = _spearman(
            [row["open3d_translation_median_m"] for row in selected],
            [row["pcl_translation_median_m"] for row in selected],
        )
        threshold = 0.70 if condition in PRIMARY_CONDITIONS else 0.50
        condition_rows.append(
            {
                "condition": condition,
                "scene_count": len(selected),
                "spearman_gate_threshold": threshold,
                "spearman_rho": rho,
                "threshold_pass": bool(rho is not None and rho >= threshold),
            }
        )
    pooled = _spearman(
        [row["open3d_translation_median_m"] for row in scene_rows],
        [row["pcl_translation_median_m"] for row in scene_rows],
    )
    rhos = [row["spearman_rho"] for row in condition_rows]
    finite_rhos = [float(value) for value in rhos if value is not None]
    at_least_half = sum(value >= 0.50 for value in finite_rhos)
    rho_median = float(median(finite_rhos)) if len(finite_rhos) == 5 else None
    by_condition = {row["condition"]: row for row in condition_rows}
    passed = bool(
        len(scene_rows) == 35
        and by_condition["INDEPENDENT_NOISE_FREE"]["spearman_rho"] is not None
        and by_condition["INDEPENDENT_NOISE_FREE"]["spearman_rho"] >= 0.70
        and by_condition["FULL_NOISE"]["spearman_rho"] is not None
        and by_condition["FULL_NOISE"]["spearman_rho"] >= 0.70
        and at_least_half >= 4
        and rho_median is not None
        and rho_median >= 0.70
        and pooled is not None
        and pooled >= 0.75
    )
    return {
        "CROSS_BACKEND_UNIQUE_UNIT_PASS": passed,
        "condition_rho_at_least_0_50_count": at_least_half,
        "condition_rho_median": rho_median,
        "condition_rows": condition_rows,
        "pooled_scene_condition_spearman_rho": pooled,
        "scene_rows": scene_rows,
    }


def _centered_turnover_rows(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    cells: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        cells[(str(row["scene_variant"]), str(row["condition"]))].append(
            (float(row["log_translation_error"]), float(row["correspondence_turnover"]))
        )
    centered: list[tuple[float, float]] = []
    for values in cells.values():
        error_center = float(median(value[0] for value in values))
        turnover_center = float(median(value[1] for value in values))
        centered.extend(
            (error - error_center, turnover - turnover_center)
            for error, turnover in values
        )
    return centered


def evaluate_turnover_robustness(
    trial_rows: Iterable[Mapping[str, Any]],
    snapshot_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stress-test turnover/error association without making a causal claim."""

    trials = [dict(row) for row in trial_rows]
    valid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        if row.get("backend") not in BACKENDS or row.get("condition") not in NONIDEAL_CONDITIONS:
            continue
        if row.get("common_association_valid", True) is not True:
            continue
        if row.get("correspondence_turnover") is None:
            continue
        error = _finite_float(row["translation_error_m"], "translation error")
        turnover = _finite_float(
            row["correspondence_turnover"], "correspondence turnover"
        )
        if error < 0.0 or not 0.0 <= turnover <= 1.0:
            raise ValueError("turnover robustness input is outside its valid domain")
        valid[str(row["backend"])].append(
            {
                **row,
                "correspondence_turnover": turnover,
                "log_translation_error": math.log10(error + 1e-9),
                "translation_error_m": error,
            }
        )
    collapsed = collapse_trials_to_unique_inputs(trials, snapshot_rows)
    backend_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for backend in BACKENDS:
        rows = valid.get(backend, [])
        pooled = _spearman(
            [row["correspondence_turnover"] for row in rows],
            [row["log_translation_error"] for row in rows],
        )
        centered_pairs = _centered_turnover_rows(rows)
        centered = _spearman(
            [pair[1] for pair in centered_pairs],
            [pair[0] for pair in centered_pairs],
        )
        scene_rhos: list[float | None] = []
        for omitted in SCENES:
            selected = [row for row in rows if row["scene_variant"] != omitted]
            pairs = _centered_turnover_rows(selected)
            rho = _spearman([pair[1] for pair in pairs], [pair[0] for pair in pairs])
            scene_rhos.append(rho)
            sensitivity_rows.append(
                {
                    "backend": backend,
                    "omitted_level": omitted,
                    "robustness_type": "LEAVE_ONE_SCENE_OUT_CENTERED",
                    "spearman_rho": rho,
                }
            )
        condition_rhos: list[float | None] = []
        for omitted in NONIDEAL_CONDITIONS:
            selected = [row for row in rows if row["condition"] != omitted]
            pairs = _centered_turnover_rows(selected)
            rho = _spearman([pair[1] for pair in pairs], [pair[0] for pair in pairs])
            condition_rhos.append(rho)
            sensitivity_rows.append(
                {
                    "backend": backend,
                    "omitted_level": omitted,
                    "robustness_type": "LEAVE_ONE_CONDITION_OUT_CENTERED",
                    "spearman_rho": rho,
                }
            )
        unique = [
            row
            for row in collapsed
            if row["backend"] == backend
            and row["condition"] in NONIDEAL_CONDITIONS
            and row["correspondence_turnover"] is not None
        ]
        unique_rho = _spearman(
            [float(row["correspondence_turnover"]) for row in unique],
            [math.log10(float(row["translation_error_m"]) + 1e-9) for row in unique],
        )
        positive_scene = sum(rho is not None and rho > 0.0 for rho in scene_rhos)
        positive_condition = sum(
            rho is not None and rho > 0.0 for rho in condition_rhos
        )
        backend_pass = bool(
            pooled is not None
            and pooled >= 0.40
            and centered is not None
            and centered >= 0.20
            and positive_scene >= 6
            and positive_condition >= 4
            and unique_rho is not None
            and unique_rho >= 0.20
        )
        backend_rows.append(
            {
                "backend": backend,
                "centered_spearman_rho": centered,
                "leave_one_condition_out_positive_count": positive_condition,
                "leave_one_scene_out_positive_count": positive_scene,
                "pooled_spearman_rho": pooled,
                "reassociation_robustness_backend_pass": backend_pass,
                "unique_input_spearman_rho": unique_rho,
                "unique_input_trial_count": len(unique),
                "valid_trial_count": len(rows),
            }
        )
    return {
        "CAUSAL_REASSOCIATION_CLAIM_AUTHORIZED": False,
        "REASSOCIATION_ROBUSTNESS_PASS": all(
            row["reassociation_robustness_backend_pass"] for row in backend_rows
        ),
        "allowed_wording": "association turnover is statistically associated with displacement",
        "backend_rows": backend_rows,
        "forbidden_wording": "causal effect of reassociation",
        "sensitivity_rows": sensitivity_rows,
    }


__all__ = [
    "BACKENDS",
    "DETERMINISTIC_TERM",
    "GEOMETRY_SEEDS",
    "LIMITED_TERM",
    "MEASUREMENT_SEEDS",
    "NONIDEAL_CONDITIONS",
    "PRIMARY_CONDITIONS",
    "REPEATABLE_TERM",
    "REPEAT_INDICES",
    "SCENES",
    "attach_observed_component_signatures",
    "audit_replicate_uniqueness",
    "classify_replicate_count",
    "collapse_trials_to_unique_inputs",
    "derive_observed_component_signatures",
    "evaluate_cross_backend_unique_unit",
    "evaluate_long_corridor_systematic_offset",
    "evaluate_turnover_robustness",
    "evaluate_unique_unit_primary_scene_effect",
]
