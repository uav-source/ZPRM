"""Registration-blind geometry-only analysis at the identity transform."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from phase_a_harness.common_association_analysis import (
    MAX_ASSOCIATION_DISTANCE_M,
    PCA_MIN_NEIGHBOR_COUNT,
    PCA_NEIGHBOR_COUNT,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_selection import (
    GEOMETRY_ONLY_FIELDS,
    TargetGeometryContext,
    compute_geometry_only_initial_metrics,
)

from . import PILOT_FLAGS
from .bag_reader import PilotBagError


FORBIDDEN_FIELD_FRAGMENTS = (
    "final",
    "turnover",
    "fitness",
    "solver",
    "displacement",
    "registration",
    "residual",
    "gradient",
)


def _classify(medians: Mapping[str, float], config: Mapping[str, Any]) -> str:
    rules = config["classification"]
    normalized = float(medians["normalized_lambda_min_trans"])
    condition = float(medians["condition_number_trans"])
    entropy = float(medians["spectral_entropy_trans"])
    if (
        normalized <= float(rules["weak_normalized_lambda_min_max"])
        or condition >= float(rules["weak_condition_number_min"])
    ):
        return "GEOMETRY_APPEARS_WEAK"
    if (
        normalized >= float(rules["rich_normalized_lambda_min_min"])
        and condition <= float(rules["rich_condition_number_max"])
        and entropy >= float(rules["rich_spectral_entropy_min"])
    ):
        return "GEOMETRY_APPEARS_RICH"
    return "GEOMETRY_INTERMEDIATE"


def compute_pilot_geometry(
    query_points: Sequence[np.ndarray],
    query_metadata: Sequence[Mapping[str, Any]],
    target_map: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(query_points) != 10 or len(query_metadata) != 10:
        raise PilotBagError("geometry-only analysis requires exactly 10 query scans")
    target = np.ascontiguousarray(target_map, dtype="<f8")
    context = TargetGeometryContext.prepare(target)
    identity = np.eye(4, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for selection_index, (source, metadata) in enumerate(zip(query_points, query_metadata)):
        finite = np.asarray(source, dtype=np.float64)
        finite = finite[np.all(np.isfinite(finite), axis=1)]
        if finite.shape[0] == 0:
            raise PilotBagError(f"selected query {selection_index} has no finite points")
        metrics = compute_geometry_only_initial_metrics(finite, identity, context=context)
        if tuple(metrics) != GEOMETRY_ONLY_FIELDS:
            raise PilotBagError("geometry-only wrapper schema changed")
        if any(fragment in key for key in metrics for fragment in FORBIDDEN_FIELD_FRAGMENTS):
            raise PilotBagError("registration-derived field escaped geometry-only firewall")
        rows.append(
            {
                "selection_index": selection_index,
                "frame_index": int(metadata["frame_index"]),
                "timestamp": float(metadata["timestamp"]),
                "query_finite_point_count": int(finite.shape[0]),
                **metrics,
            }
        )
    spectral_fields = GEOMETRY_ONLY_FIELDS[2:]
    if any(row[field] is None for row in rows for field in spectral_fields):
        raise PilotBagError("geometry-only spectrum is noncomputable for a selected query")
    medians = {
        field: float(np.median([float(row[field]) for row in rows]))
        for field in GEOMETRY_ONLY_FIELDS
    }
    minimums = {
        field: float(np.min([float(row[field]) for row in rows]))
        for field in GEOMETRY_ONLY_FIELDS
    }
    maximums = {
        field: float(np.max([float(row[field]) for row in rows]))
        for field in GEOMETRY_ONLY_FIELDS
    }
    classification = _classify(medians, config)
    summary = {
        "schema": "mid360_pilot_geometry_summary_v1",
        **PILOT_FLAGS,
        "status": "PASS",
        "query_count": len(rows),
        "T0": "IDENTITY_4X4",
        "transform_convention": "SOURCE_QUERY_TO_TARGET_MAP",
        "geometry_only": True,
        "registration_executed": False,
        "registration_derived_fields_forbidden": True,
        "metric_fields": list(GEOMETRY_ONLY_FIELDS),
        "association_distance_limit_m": MAX_ASSOCIATION_DISTANCE_M,
        "target_normal_pca_k": PCA_NEIGHBOR_COUNT,
        "target_normal_pca_min_neighbors": PCA_MIN_NEIGHBOR_COUNT,
        "target_map_point_count": int(target.shape[0]),
        "median": medians,
        "minimum": minimums,
        "maximum": maximums,
        "GEOMETRY_DESCRIPTION": classification,
        "classification_contract": dict(config["classification"]),
        "classification_is_paper_threshold": False,
    }
    return rows, summary


__all__ = ["FORBIDDEN_FIELD_FRAGMENTS", "compute_pilot_geometry"]
