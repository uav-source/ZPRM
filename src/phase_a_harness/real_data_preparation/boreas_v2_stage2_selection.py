"""Blind Boreas v2 Stage-2 geometry screening and snapshot selection.

This module is deliberately registration-free.  It computes only the initial
normal-geometry spectrum at the frozen reference transform, constructs the
already anchored five-second candidate intervals, and applies the frozen
weak-first greedy and five-quantile rules.  It never forms a point-to-plane
residual, a gradient, an estimated transform, or a backend result.

The public functions consume and return JSON-native rows.  Every persisted row
has a canonical SHA-256 so a later, independently implemented verifier can
recompute the full selection without trusting CSV order or producer summaries.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from phase_a_harness.common_association_analysis import (
    MAX_ASSOCIATION_DISTANCE_M,
    PCA_CHUNK_SIZE,
    PCA_MIN_NEIGHBOR_COUNT,
    PCA_NEIGHBOR_COUNT,
    _estimate_target_normals_with_tree,
    associate_source_points,
)

from .io import canonical_json_bytes


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BOREAS_STAGE2_SELECTION_SCHEMA = "boreas_v2_stage2_blind_selection_v1"
PRODUCTION_AUTHORITY = "BOREAS_V2_STAGE2_PREREGISTRATION"
SYNTHETIC_AUTHORITY = "SYNTHETIC_FIXTURE_ONLY"

WEAK_LABEL = "CORRIDOR_OR_WEAK_GEOMETRY"
RICH_LABEL = "GEOMETRY_RICH"
LABEL_ORDER = (WEAK_LABEL, RICH_LABEL)

GEOMETRY_ONLY_FIELDS = (
    "initial_correspondence_count",
    "initial_valid_normal_correspondence_count",
    "lambda_min_trans",
    "lambda_mid_trans",
    "lambda_max_trans",
    "normalized_lambda_min_trans",
    "normalized_lambda_mid_trans",
    "normalized_lambda_max_trans",
    "condition_number_trans",
    "spectral_entropy_trans",
)
SPECTRAL_FIELDS = GEOMETRY_ONLY_FIELDS[2:]
INTERVAL_SCORE_FIELDS = (
    "normalized_lambda_min_trans",
    "condition_number_trans",
    "spectral_entropy_trans",
)

FIRST_PASS_SCAN_FIELDS = frozenset(
    {
        "query_ordinal",
        "sequence_id",
        "object_key",
        "timestamp_us",
        "remote_size_bytes",
        "last_modified",
        "etag",
        "payload_sha256",
        "finite_source_point_count",
        "target_map_point_count",
        "reference_interpolation_valid",
        "reference_gap_s",
        "gt_overlap_within_5m",
        "target_map_frozen_complete",
        "deskew_processing_contract_valid",
        "gt_sha256",
        "calibration_sha256",
        "preprocessing_contract_sha256",
        "target_map_sha256",
        *GEOMETRY_ONLY_FIELDS,
    }
)

CANDIDATE_SCAN_CSV_FIELDS = (
    "query_ordinal",
    "sequence_id",
    "object_key",
    "timestamp_us",
    "remote_size_bytes",
    "last_modified",
    "etag",
    "payload_sha256",
    "interval_id",
    "window_index",
    "finite_source_point_count",
    "target_map_point_count",
    *GEOMETRY_ONLY_FIELDS,
    "reference_interpolation_valid",
    "reference_gap_s",
    "gt_gap_within_limit",
    "gt_overlap_within_5m",
    "target_map_frozen_complete",
    "target_map_coverage_valid",
    "deskew_processing_contract_valid",
    "geometry_valid",
    "exclusion_reason",
    "gt_sha256",
    "calibration_sha256",
    "preprocessing_contract_sha256",
    "selection_contract_sha256",
    "target_map_sha256",
    "candidate_scan_row_sha256",
)

GEOMETRY_METRIC_CSV_FIELDS = (
    "query_ordinal",
    "object_key",
    "timestamp_us",
    *GEOMETRY_ONLY_FIELDS,
    "geometry_valid",
    "exclusion_reason",
    "candidate_scan_row_sha256",
)

CANDIDATE_INTERVAL_CSV_FIELDS = (
    "interval_id",
    "window_index",
    "interval_index",
    "start_time_us",
    "end_time_us",
    "duration_us",
    "center_time_us",
    "center_world_x_m",
    "center_world_y_m",
    "center_world_z_m",
    "center_quaternion_x",
    "center_quaternion_y",
    "center_quaternion_z",
    "center_quaternion_w",
    "center_reference_lower_timestamp_us",
    "center_reference_upper_timestamp_us",
    "center_interpolation_method",
    "candidate_scan_count",
    "geometry_valid_scan_count",
    "geometry_invalid_scan_count",
    "geometry_valid_fraction",
    "minimum_geometry_valid_fraction",
    "interval_valid",
    "exclusion_reason",
    *INTERVAL_SCORE_FIELDS,
    "candidate_scan_rows_sha256",
    "selection_contract_sha256",
    "candidate_interval_row_sha256",
)

SELECTED_INTERVAL_CSV_FIELDS = (
    "scene_label",
    "selection_rank_within_label",
    "interval_id",
    "window_index",
    "start_time_us",
    "end_time_us",
    "center_world_x_m",
    "center_world_y_m",
    "center_world_z_m",
    *INTERVAL_SCORE_FIELDS,
    "candidate_interval_row_sha256",
    "selected_interval_row_sha256",
)

SELECTED_SNAPSHOT_CSV_FIELDS = (
    "selection_index",
    "snapshot_id",
    "scene_label",
    "interval_id",
    "interval_selection_rank",
    "selected_interval_row_sha256",
    "quantile_index",
    "quantile_probability",
    "quantile_method",
    "quantile_target_timestamp_us",
    "selected_timestamp_us",
    "absolute_quantile_delta_us",
    "object_key",
    "query_ordinal",
    "first_pass_geometry_row_sha256",
    "selection_contract_sha256",
    "selected_snapshot_row_sha256",
)


class BoreasStage2SelectionError(RuntimeError):
    """A blind-selection input or frozen rule was violated."""


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise BoreasStage2SelectionError(f"{label} must be a lowercase SHA-256")
    return text


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise BoreasStage2SelectionError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise BoreasStage2SelectionError(f"{label} must be >= {minimum}")
    return result


def _finite_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise BoreasStage2SelectionError(f"{label} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise BoreasStage2SelectionError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise BoreasStage2SelectionError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise BoreasStage2SelectionError(f"{label} must be >= {minimum}")
    return result


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise BoreasStage2SelectionError(f"{label} must be boolean")
    return value


def _finite_xyz(value: Any, label: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise BoreasStage2SelectionError(f"{label} must be three finite coordinates")
    return tuple(float(child) for child in array)


@dataclass(frozen=True)
class Stage2SelectionContract:
    """Frozen scientific rules for candidate validity and deterministic selection."""

    parameter_authority: str = PRODUCTION_AUTHORITY
    expected_candidate_scan_count: int = 11_859
    expected_candidate_interval_count: int = 246
    interval_duration_us: int = 5_000_000
    minimum_finite_source_point_count: int = 1_000
    minimum_target_map_point_count: int = 10_000
    minimum_valid_normal_correspondence_count: int = 100
    minimum_valid_normal_correspondence_fraction: float = 0.05
    minimum_interval_valid_numerator: int = 4
    minimum_interval_valid_denominator: int = 5
    maximum_reference_gap_s: float = 0.2
    gt_overlap_radius_m: float = 5.0
    interval_count_per_label: int = 10
    snapshots_per_interval: int = 5
    minimum_center_time_separation_s: float = 10.0
    minimum_center_position_separation_m: float = 1.0
    quantile_probabilities: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
    quantile_method: str = "linear"
    weak_first: bool = True
    proximity_rejection_logic: str = "TIME_LT_THRESHOLD_AND_3D_DISTANCE_LT_THRESHOLD"

    def __post_init__(self) -> None:
        if self.parameter_authority not in {PRODUCTION_AUTHORITY, SYNTHETIC_AUTHORITY}:
            raise BoreasStage2SelectionError("unsupported Stage-2 selection authority")
        for field in (
            "expected_candidate_scan_count",
            "expected_candidate_interval_count",
            "interval_duration_us",
            "minimum_finite_source_point_count",
            "minimum_target_map_point_count",
            "minimum_valid_normal_correspondence_count",
            "minimum_interval_valid_numerator",
            "minimum_interval_valid_denominator",
            "interval_count_per_label",
            "snapshots_per_interval",
        ):
            if _strict_int(getattr(self, field), field, minimum=1) != getattr(self, field):
                raise AssertionError("unreachable integer normalization mismatch")
        for field in (
            "minimum_valid_normal_correspondence_fraction",
            "maximum_reference_gap_s",
            "gt_overlap_radius_m",
            "minimum_center_time_separation_s",
            "minimum_center_position_separation_m",
        ):
            _finite_float(getattr(self, field), field, minimum=0.0)
        if self.minimum_interval_valid_numerator > self.minimum_interval_valid_denominator:
            raise BoreasStage2SelectionError("interval validity fraction cannot exceed one")
        if self.quantile_method != "linear":
            raise BoreasStage2SelectionError("only the frozen linear quantile method is allowed")
        if tuple(self.quantile_probabilities) != (0.10, 0.30, 0.50, 0.70, 0.90):
            raise BoreasStage2SelectionError("snapshot quantiles differ from frozen 10/30/50/70/90")
        if self.snapshots_per_interval != len(self.quantile_probabilities):
            raise BoreasStage2SelectionError("snapshot count and quantile count disagree")
        if self.interval_count_per_label != 10 or self.snapshots_per_interval != 5:
            raise BoreasStage2SelectionError("the 10 intervals x 5 snapshots design is immutable")
        if self.weak_first is not True:
            raise BoreasStage2SelectionError("the frozen greedy selector must process weak first")
        if self.proximity_rejection_logic != (
            "TIME_LT_THRESHOLD_AND_3D_DISTANCE_LT_THRESHOLD"
        ):
            raise BoreasStage2SelectionError("independence logic differs from frozen selector")
        if self.parameter_authority == PRODUCTION_AUTHORITY:
            expected = {
                "expected_candidate_scan_count": 11_859,
                "expected_candidate_interval_count": 246,
                "interval_duration_us": 5_000_000,
                "minimum_finite_source_point_count": 1_000,
                "minimum_target_map_point_count": 10_000,
                "minimum_valid_normal_correspondence_count": 100,
                "minimum_valid_normal_correspondence_fraction": 0.05,
                "minimum_interval_valid_numerator": 4,
                "minimum_interval_valid_denominator": 5,
                "maximum_reference_gap_s": 0.2,
                "gt_overlap_radius_m": 5.0,
                "minimum_center_time_separation_s": 10.0,
                "minimum_center_position_separation_m": 1.0,
            }
            for field, expected_value in expected.items():
                if getattr(self, field) != expected_value:
                    raise BoreasStage2SelectionError(
                        f"production selection parameter changed: {field}"
                    )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["quantile_probabilities"] = list(self.quantile_probabilities)
        value.update(
            {
                "geometry_metric_parameter_binding": {
                    "association_distance_m": MAX_ASSOCIATION_DISTANCE_M,
                    "target_normal_pca_k": PCA_NEIGHBOR_COUNT,
                    "target_normal_pca_min_neighbors": PCA_MIN_NEIGHBOR_COUNT,
                },
                "interval_aggregation": "MEDIAN_OF_GEOMETRY_VALID_SCANS_ONLY",
                "interval_boundary": "FROZEN_HALF_OPEN_START_INCLUSIVE_END_EXCLUSIVE",
                "schema": BOREAS_STAGE2_SELECTION_SCHEMA,
            }
        )
        return value

    @property
    def sha256(self) -> str:
        return _hash_json(self.as_dict())


@dataclass(frozen=True)
class SelectionBindings:
    """Immutable identities shared by every first-pass scientific row."""

    primary_query_sequence_id: str
    gt_sha256: str
    calibration_sha256: str
    preprocessing_contract_sha256: str
    target_map_sha256: str

    def validated(self) -> "SelectionBindings":
        if not self.primary_query_sequence_id:
            raise BoreasStage2SelectionError("primary query sequence ID is empty")
        return SelectionBindings(
            primary_query_sequence_id=str(self.primary_query_sequence_id),
            gt_sha256=_require_sha256(self.gt_sha256, "gt_sha256"),
            calibration_sha256=_require_sha256(
                self.calibration_sha256, "calibration_sha256"
            ),
            preprocessing_contract_sha256=_require_sha256(
                self.preprocessing_contract_sha256, "preprocessing_contract_sha256"
            ),
            target_map_sha256=_require_sha256(
                self.target_map_sha256, "target_map_sha256"
            ),
        )


@dataclass(frozen=True)
class ReferencePoseSeries:
    """Frozen LiDAR poses used only to locate an interval's exact midpoint."""

    timestamps_us: np.ndarray
    translations_xyz_m: np.ndarray
    quaternions_xyzw: np.ndarray

    def validated(self) -> "ReferencePoseSeries":
        timestamps = np.asarray(self.timestamps_us)
        if (
            timestamps.ndim != 1
            or timestamps.size < 1
            or timestamps.dtype.kind not in "iu"
        ):
            raise BoreasStage2SelectionError("reference timestamps must be integer microseconds")
        timestamps = np.ascontiguousarray(timestamps, dtype=np.int64)
        if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0):
            raise BoreasStage2SelectionError("reference timestamps must be strictly increasing")
        translations = np.asarray(self.translations_xyz_m, dtype=np.float64)
        quaternions = np.asarray(self.quaternions_xyzw, dtype=np.float64)
        if translations.shape != (timestamps.size, 3) or not np.all(np.isfinite(translations)):
            raise BoreasStage2SelectionError("reference translations are invalid")
        if quaternions.shape != (timestamps.size, 4) or not np.all(np.isfinite(quaternions)):
            raise BoreasStage2SelectionError("reference quaternions are invalid")
        norms = np.linalg.norm(quaternions, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-10):
            raise BoreasStage2SelectionError("reference quaternions must be unit xyzw")
        return ReferencePoseSeries(
            timestamps_us=timestamps,
            translations_xyz_m=np.ascontiguousarray(translations, dtype="<f8"),
            quaternions_xyzw=np.ascontiguousarray(quaternions, dtype="<f8"),
        )

    def interpolate_midpoint(
        self, timestamp_us: int, *, maximum_gap_s: float
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        series = self.validated()
        query = _strict_int(timestamp_us, "midpoint timestamp_us")
        times = series.timestamps_us
        exact = int(np.searchsorted(times, query, side="left"))
        if exact < times.size and int(times[exact]) == query:
            return (
                series.translations_xyz_m[exact].copy(),
                series.quaternions_xyzw[exact].copy(),
                query,
                query,
            )
        upper = int(np.searchsorted(times, query, side="right"))
        if upper == 0 or upper >= times.size:
            raise BoreasStage2SelectionError("interval midpoint is outside reference coverage")
        lower = upper - 1
        lower_time = int(times[lower])
        upper_time = int(times[upper])
        gap_s = (upper_time - lower_time) / 1_000_000.0
        if gap_s > maximum_gap_s:
            raise BoreasStage2SelectionError("midpoint reference gap exceeds frozen 0.2 s")
        weight = (query - lower_time) / float(upper_time - lower_time)
        translation = (
            (1.0 - weight) * series.translations_xyz_m[lower]
            + weight * series.translations_xyz_m[upper]
        )
        rotations = Rotation.from_quat(series.quaternions_xyzw[[lower, upper]])
        quaternion = Slerp(
            np.asarray([lower_time, upper_time], dtype=np.float64), rotations
        )([float(query)]).as_quat()[0]
        return (
            np.ascontiguousarray(translation, dtype="<f8"),
            np.ascontiguousarray(quaternion, dtype="<f8"),
            lower_time,
            upper_time,
        )


@dataclass(frozen=True)
class TargetGeometryContext:
    """One reusable normal field and search tree for a frozen target map.

    ``prepare`` accepts an ordinary array or a read-only ``numpy.memmap``.  A
    C-contiguous little-endian float64 memmap is retained without copying, so a
    production first pass can prepare the multi-gigabyte target exactly once
    and reuse it for every query scan.  The backing file must remain immutable
    for the lifetime of this context.
    """

    target_points: np.ndarray
    target_normals: np.ndarray
    target_normal_valid: np.ndarray
    target_tree: cKDTree

    @classmethod
    def prepare(
        cls,
        target_points: Any,
        *,
        normal_chunk_size: int = PCA_CHUNK_SIZE,
    ) -> "TargetGeometryContext":
        target = np.asarray(target_points, dtype="<f8")
        if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] == 0:
            raise BoreasStage2SelectionError(
                "target points must be a non-empty Nx3 array"
            )
        if not np.all(np.isfinite(target)):
            raise BoreasStage2SelectionError("target points must be finite")
        if (
            not isinstance(normal_chunk_size, int)
            or isinstance(normal_chunk_size, bool)
            or normal_chunk_size <= 0
        ):
            raise BoreasStage2SelectionError(
                "normal_chunk_size must be a positive integer"
            )
        target = np.ascontiguousarray(target, dtype="<f8")
        tree = cKDTree(target, copy_data=False)
        normals, normal_valid = _estimate_target_normals_with_tree(
            target,
            tree,
            chunk_size=normal_chunk_size,
        )
        target.setflags(write=False)
        return cls(
            target_points=target,
            target_normals=normals,
            target_normal_valid=normal_valid,
            target_tree=tree,
        )

    def compute(self, source_points: Any, t_reference: Any) -> dict[str, Any]:
        """Compute the ten allowed fields while reusing this target context."""

        source = np.asarray(source_points, dtype=np.float64)
        transform = np.asarray(t_reference, dtype=np.float64)
        if (
            source.ndim != 2
            or source.shape[1] != 3
            or not np.all(np.isfinite(source))
        ):
            raise BoreasStage2SelectionError(
                "source points must be a finite Nx3 array"
            )
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise BoreasStage2SelectionError(
                "T_reference must be a finite 4x4 matrix"
            )
        if not np.allclose(
            transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-12
        ):
            raise BoreasStage2SelectionError(
                "T_reference homogeneous row is invalid"
            )
        source = np.ascontiguousarray(source, dtype="<f8")
        transform = np.ascontiguousarray(transform, dtype="<f8")

        association = associate_source_points(
            source,
            transform,
            target_tree=self.target_tree,
            target_point_count=self.target_points.shape[0],
        )
        valid_matches = self.target_normal_valid[association.target_indices]
        selected_normals = self.target_normals[
            association.target_indices[valid_matches]
        ]
        valid_count = int(selected_normals.shape[0])
        result: dict[str, Any] = {
            "initial_correspondence_count": int(association.count),
            "initial_valid_normal_correspondence_count": valid_count,
        }
        if valid_count == 0:
            result.update({field: None for field in SPECTRAL_FIELDS})
            return result

        hessian = (selected_normals.T @ selected_normals) / float(valid_count)
        eigenvalues = np.maximum(np.linalg.eigvalsh(hessian), 0.0)
        total = max(float(np.sum(eigenvalues)), 1.0e-12)
        normalized = eigenvalues / total
        positive = normalized > 0.0
        entropy = -float(
            np.sum(normalized[positive] * np.log(normalized[positive]))
        ) / math.log(3.0)
        result.update(
            {
                "lambda_min_trans": float(eigenvalues[0]),
                "lambda_mid_trans": float(eigenvalues[1]),
                "lambda_max_trans": float(eigenvalues[2]),
                "normalized_lambda_min_trans": float(normalized[0]),
                "normalized_lambda_mid_trans": float(normalized[1]),
                "normalized_lambda_max_trans": float(normalized[2]),
                "condition_number_trans": float(
                    eigenvalues[2] / max(float(eigenvalues[0]), 1.0e-12)
                ),
                "spectral_entropy_trans": entropy,
            }
        )
        if set(result) != set(GEOMETRY_ONLY_FIELDS):
            raise AssertionError("geometry-only output schema drift")
        if not all(
            math.isfinite(float(result[field])) for field in SPECTRAL_FIELDS
        ):
            raise BoreasStage2SelectionError("geometry-only spectrum is nonfinite")
        return result


def compute_geometry_only_initial_metrics(
    source_points: Any,
    target_points: Any = None,
    t_reference: Any = None,
    *,
    context: TargetGeometryContext | None = None,
) -> dict[str, Any]:
    """Return the ten allowed initial geometry fields without a solver outcome.

    Passing a prepared ``context`` is the production path and avoids rebuilding
    target normals or the target cKDTree per scan.  It supports the compact call
    ``compute_geometry_only_initial_metrics(source, T_reference, context=ctx)``.
    A separately supplied target must share the context backing array, which
    prevents an accidentally mismatched target from being silently ignored.
    Omitting the context is a convenience for small tests and prepares one for
    the traditional ``(source, target, T_reference)`` call.

    No signed point-to-plane values are formed, so neither initial residual
    summaries nor a translation gradient exist even transiently.
    """

    if context is not None and t_reference is None:
        t_reference = target_points
        target_points = None
    if t_reference is None:
        raise BoreasStage2SelectionError("T_reference is required")
    if context is None:
        if target_points is None:
            raise BoreasStage2SelectionError(
                "target_points is required when context is absent"
            )
        context = TargetGeometryContext.prepare(target_points)
    elif target_points is not None:
        candidate = np.asarray(target_points, dtype="<f8")
        if not (
            candidate is context.target_points
            or (
                candidate.shape == context.target_points.shape
                and np.shares_memory(candidate, context.target_points)
            )
        ):
            raise BoreasStage2SelectionError(
                "target_points does not share the prepared target context"
            )
    return context.compute(source_points, t_reference)


def _validated_windows(
    frozen_windows: Sequence[Mapping[str, Any]], contract: Stage2SelectionContract
) -> list[dict[str, Any]]:
    if len(frozen_windows) != contract.expected_candidate_interval_count:
        raise BoreasStage2SelectionError(
            "frozen five-second candidate interval count differs from contract"
        )
    output: list[dict[str, Any]] = []
    previous_end: int | None = None
    for expected_index, source in enumerate(frozen_windows):
        allowed = {
            "window_index",
            "interval_index",
            "start_time_us",
            "end_time_us",
            "duration_us",
        }
        if set(source) != allowed:
            raise BoreasStage2SelectionError("frozen window schema differs")
        window_index = _strict_int(source["window_index"], "window_index")
        if window_index != expected_index:
            raise BoreasStage2SelectionError("window indices are not exactly 0..N-1")
        parent_index = _strict_int(source["interval_index"], "interval_index")
        start = _strict_int(source["start_time_us"], "start_time_us")
        end = _strict_int(source["end_time_us"], "end_time_us")
        duration = _strict_int(source["duration_us"], "duration_us", minimum=1)
        if duration != contract.interval_duration_us or end - start != duration:
            raise BoreasStage2SelectionError("candidate window is not exactly frozen 5 s")
        if previous_end is not None and start < previous_end:
            raise BoreasStage2SelectionError("candidate windows overlap or are unordered")
        previous_end = end
        output.append(
            {
                "duration_us": duration,
                "end_time_us": end,
                "interval_id": f"boreas-v2-window-{window_index:03d}",
                "interval_index": parent_index,
                "start_time_us": start,
                "window_index": window_index,
            }
        )
    return output


def _validate_first_pass_row(
    source: Mapping[str, Any],
    *,
    bindings: SelectionBindings,
    contract: Stage2SelectionContract,
) -> dict[str, Any]:
    if set(source) != FIRST_PASS_SCAN_FIELDS:
        missing = sorted(FIRST_PASS_SCAN_FIELDS - set(source))
        extra = sorted(set(source) - FIRST_PASS_SCAN_FIELDS)
        raise BoreasStage2SelectionError(
            f"first-pass scan field set differs; missing={missing}, extra={extra}"
        )
    query_ordinal = _strict_int(source["query_ordinal"], "query_ordinal")
    timestamp_us = _strict_int(source["timestamp_us"], "timestamp_us")
    sequence_id = str(source["sequence_id"])
    if sequence_id != bindings.primary_query_sequence_id:
        raise BoreasStage2SelectionError("first-pass scan is not from PRIMARY query")
    object_key = str(source["object_key"])
    expected_suffix = f"/{timestamp_us}.bin"
    if not object_key.startswith(f"{sequence_id}/lidar/") or not object_key.endswith(
        expected_suffix
    ):
        raise BoreasStage2SelectionError("object key/timestamp/sequence identity disagrees")
    if not str(source["last_modified"]) or not str(source["etag"]):
        raise BoreasStage2SelectionError("remote LastModified/ETag identity is empty")
    source_count = _strict_int(
        source["finite_source_point_count"], "finite_source_point_count"
    )
    target_count = _strict_int(
        source["target_map_point_count"], "target_map_point_count"
    )
    initial_count = _strict_int(
        source["initial_correspondence_count"], "initial_correspondence_count"
    )
    valid_count = _strict_int(
        source["initial_valid_normal_correspondence_count"],
        "initial_valid_normal_correspondence_count",
    )
    if valid_count > initial_count or initial_count > source_count:
        raise BoreasStage2SelectionError("correspondence counts are inconsistent")
    spectral_values = [source[field] for field in SPECTRAL_FIELDS]
    all_missing = all(value is None for value in spectral_values)
    all_finite = all(
        value is not None
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in spectral_values
    )
    if not (all_missing or all_finite):
        raise BoreasStage2SelectionError(
            "spectral fields must be either all finite or all null for an invalid scan"
        )
    if valid_count > 0 and not all_finite:
        raise BoreasStage2SelectionError("positive valid-normal count requires finite spectrum")
    reference_gap = _finite_float(source["reference_gap_s"], "reference_gap_s", minimum=0.0)
    bool_fields = {
        field: _strict_bool(source[field], field)
        for field in (
            "reference_interpolation_valid",
            "gt_overlap_within_5m",
            "target_map_frozen_complete",
            "deskew_processing_contract_valid",
        )
    }
    for field, expected in (
        ("gt_sha256", bindings.gt_sha256),
        ("calibration_sha256", bindings.calibration_sha256),
        ("preprocessing_contract_sha256", bindings.preprocessing_contract_sha256),
        ("target_map_sha256", bindings.target_map_sha256),
    ):
        if _require_sha256(source[field], field) != expected:
            raise BoreasStage2SelectionError(f"first-pass {field} binding changed")
    row = {
        "query_ordinal": query_ordinal,
        "sequence_id": sequence_id,
        "object_key": object_key,
        "timestamp_us": timestamp_us,
        "remote_size_bytes": _strict_int(
            source["remote_size_bytes"], "remote_size_bytes", minimum=1
        ),
        "last_modified": str(source["last_modified"]),
        "etag": str(source["etag"]),
        "payload_sha256": _require_sha256(source["payload_sha256"], "payload_sha256"),
        "finite_source_point_count": source_count,
        "target_map_point_count": target_count,
        "initial_correspondence_count": initial_count,
        "initial_valid_normal_correspondence_count": valid_count,
        **{
            field: (None if source[field] is None else float(source[field]))
            for field in SPECTRAL_FIELDS
        },
        **bool_fields,
        "reference_gap_s": reference_gap,
        "gt_gap_within_limit": reference_gap <= contract.maximum_reference_gap_s,
        "target_map_coverage_valid": (
            bool_fields["gt_overlap_within_5m"]
            and bool_fields["target_map_frozen_complete"]
        ),
        "gt_sha256": bindings.gt_sha256,
        "calibration_sha256": bindings.calibration_sha256,
        "preprocessing_contract_sha256": bindings.preprocessing_contract_sha256,
        "selection_contract_sha256": contract.sha256,
        "target_map_sha256": bindings.target_map_sha256,
    }
    return row


def _candidate_validity(
    row: Mapping[str, Any], contract: Stage2SelectionContract
) -> tuple[bool, str]:
    source_count = int(row["finite_source_point_count"])
    if source_count < contract.minimum_finite_source_point_count:
        return False, "FINITE_SOURCE_POINT_COUNT_LT_1000"
    if int(row["target_map_point_count"]) < contract.minimum_target_map_point_count:
        return False, "TARGET_MAP_POINT_COUNT_LT_10000"
    required = max(
        contract.minimum_valid_normal_correspondence_count,
        math.ceil(contract.minimum_valid_normal_correspondence_fraction * source_count),
    )
    if int(row["initial_valid_normal_correspondence_count"]) < required:
        return False, "INSUFFICIENT_VALID_NORMAL_CORRESPONDENCES"
    for field, reason in (
        ("reference_interpolation_valid", "REFERENCE_INTERPOLATION_INVALID"),
        ("gt_gap_within_limit", "GT_GAP_EXCEEDS_0_2_S"),
        ("target_map_coverage_valid", "TARGET_MAP_COVERAGE_INVALID"),
        ("deskew_processing_contract_valid", "DESKEW_PROCESSING_CONTRACT_INVALID"),
    ):
        if row[field] is not True:
            return False, reason
    if not all(row[field] is not None for field in SPECTRAL_FIELDS):
        return False, "GEOMETRY_SPECTRUM_NOT_COMPUTABLE"
    return True, ""


def build_candidate_scan_inventory(
    first_pass_rows: Sequence[Mapping[str, Any]],
    frozen_windows: Sequence[Mapping[str, Any]],
    *,
    bindings: SelectionBindings,
    contract: Stage2SelectionContract = Stage2SelectionContract(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign every PRIMARY query allowlist scan to one immutable 5 s window."""

    bindings = bindings.validated()
    windows = _validated_windows(frozen_windows, contract)
    if len(first_pass_rows) != contract.expected_candidate_scan_count:
        raise BoreasStage2SelectionError("candidate query scan count differs from contract")
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    timestamps: set[int] = set()
    for expected_ordinal, source in enumerate(first_pass_rows):
        row = _validate_first_pass_row(source, bindings=bindings, contract=contract)
        if row["query_ordinal"] != expected_ordinal:
            raise BoreasStage2SelectionError("query ordinals are not exactly 0..N-1")
        if row["object_key"] in keys or row["timestamp_us"] in timestamps:
            raise BoreasStage2SelectionError("duplicate query object key or timestamp")
        keys.add(row["object_key"])
        timestamps.add(row["timestamp_us"])
        matches = [
            window
            for window in windows
            if window["start_time_us"] <= row["timestamp_us"] < window["end_time_us"]
        ]
        if len(matches) != 1:
            raise BoreasStage2SelectionError(
                "candidate scan must belong to exactly one frozen half-open window"
            )
        window = matches[0]
        valid, reason = _candidate_validity(row, contract)
        core = {
            **row,
            "interval_id": window["interval_id"],
            "window_index": window["window_index"],
            "geometry_valid": valid,
            "exclusion_reason": reason,
        }
        rows.append({**core, "candidate_scan_row_sha256": _hash_json(core)})
    excluded = [dict(row) for row in rows if row["geometry_valid"] is False]
    return rows, excluded


def build_candidate_intervals(
    candidate_scans: Sequence[Mapping[str, Any]],
    frozen_windows: Sequence[Mapping[str, Any]],
    reference_poses: ReferencePoseSeries,
    *,
    contract: Stage2SelectionContract = Stage2SelectionContract(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate valid-only medians on all fixed windows and retain exclusions."""

    windows = _validated_windows(frozen_windows, contract)
    if len(candidate_scans) != contract.expected_candidate_scan_count:
        raise BoreasStage2SelectionError("candidate scan inventory is incomplete")
    poses = reference_poses.validated()
    grouped: dict[str, list[Mapping[str, Any]]] = {
        window["interval_id"]: [] for window in windows
    }
    for row in candidate_scans:
        interval_id = str(row.get("interval_id", ""))
        if interval_id not in grouped:
            raise BoreasStage2SelectionError("scan names an unknown candidate interval")
        core = {key: value for key, value in row.items() if key != "candidate_scan_row_sha256"}
        if row.get("candidate_scan_row_sha256") != _hash_json(core):
            raise BoreasStage2SelectionError("candidate scan row SHA mismatch")
        grouped[interval_id].append(row)

    output: list[dict[str, Any]] = []
    for window in windows:
        rows = grouped[window["interval_id"]]
        if not rows:
            raise BoreasStage2SelectionError("frozen candidate interval has no scans")
        if rows != sorted(rows, key=lambda row: int(row["timestamp_us"])):
            raise BoreasStage2SelectionError("interval scan rows are not timestamp ordered")
        valid_rows = [row for row in rows if row.get("geometry_valid") is True]
        valid_count = len(valid_rows)
        total_count = len(rows)
        interval_valid = (
            valid_count * contract.minimum_interval_valid_denominator
            >= total_count * contract.minimum_interval_valid_numerator
        )
        midpoint = (window["start_time_us"] + window["end_time_us"]) // 2
        translation, quaternion, lower, upper = poses.interpolate_midpoint(
            midpoint, maximum_gap_s=contract.maximum_reference_gap_s
        )
        score_values: dict[str, float | None]
        if interval_valid:
            score_values = {
                field: float(np.median([float(row[field]) for row in valid_rows]))
                for field in INTERVAL_SCORE_FIELDS
            }
        else:
            score_values = {field: None for field in INTERVAL_SCORE_FIELDS}
        source_rows_sha256 = _hash_json(
            [row["candidate_scan_row_sha256"] for row in rows]
        )
        core = {
            **window,
            "center_time_us": midpoint,
            "center_world_position": [float(value) for value in translation],
            "center_world_quaternion_xyzw": [float(value) for value in quaternion],
            "center_reference_lower_timestamp_us": lower,
            "center_reference_upper_timestamp_us": upper,
            "center_interpolation_method": "LINEAR_TRANSLATION_SLERP_XYZW",
            "candidate_scan_count": total_count,
            "geometry_valid_scan_count": valid_count,
            "geometry_invalid_scan_count": total_count - valid_count,
            "geometry_valid_fraction": valid_count / total_count,
            "minimum_geometry_valid_fraction": (
                contract.minimum_interval_valid_numerator
                / contract.minimum_interval_valid_denominator
            ),
            "interval_valid": interval_valid,
            "exclusion_reason": "" if interval_valid else "GEOMETRY_VALID_FRACTION_LT_0_80",
            **score_values,
            "candidate_scan_rows_sha256": source_rows_sha256,
            "selection_contract_sha256": contract.sha256,
        }
        output.append({**core, "candidate_interval_row_sha256": _hash_json(core)})
    excluded = [dict(row) for row in output if row["interval_valid"] is False]
    return output, excluded


def _rank_key(row: Mapping[str, Any], label: str) -> tuple[Any, ...]:
    primary = _finite_float(row["normalized_lambda_min_trans"], "interval primary")
    condition = _finite_float(row["condition_number_trans"], "interval condition")
    entropy = _finite_float(row["spectral_entropy_trans"], "interval entropy")
    start = _strict_int(row["start_time_us"], "interval start_time_us")
    interval_id = str(row["interval_id"])
    if label == WEAK_LABEL:
        return (primary, -condition, entropy, start, interval_id)
    if label == RICH_LABEL:
        return (-primary, condition, -entropy, start, interval_id)
    raise BoreasStage2SelectionError(f"unsupported scene label: {label}")


def _compatible(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    contract: Stage2SelectionContract,
) -> bool:
    start = _strict_int(candidate["start_time_us"], "candidate start")
    end = _strict_int(candidate["end_time_us"], "candidate end")
    center = (start + end) / 2_000_000.0
    position = np.asarray(_finite_xyz(candidate["center_world_position"], "candidate center"))
    for prior in selected:
        prior_start = _strict_int(prior["start_time_us"], "prior start")
        prior_end = _strict_int(prior["end_time_us"], "prior end")
        if max(start, prior_start) < min(end, prior_end):
            return False
        prior_center = (prior_start + prior_end) / 2_000_000.0
        prior_position = np.asarray(
            _finite_xyz(prior["center_world_position"], "prior center")
        )
        if (
            abs(center - prior_center) < contract.minimum_center_time_separation_s
            and float(np.linalg.norm(position - prior_position))
            < contract.minimum_center_position_separation_m
        ):
            return False
    return True


def select_frozen_intervals(
    candidate_intervals: Sequence[Mapping[str, Any]],
    *,
    contract: Stage2SelectionContract = Stage2SelectionContract(),
) -> list[dict[str, Any]]:
    """Select 10 weak then 10 rich intervals with the frozen global exclusion."""

    if len(candidate_intervals) != contract.expected_candidate_interval_count:
        raise BoreasStage2SelectionError("candidate interval inventory is incomplete")
    eligible: list[dict[str, Any]] = []
    interval_ids: set[str] = set()
    for source in candidate_intervals:
        row = dict(source)
        core = {key: value for key, value in row.items() if key != "candidate_interval_row_sha256"}
        if row.get("candidate_interval_row_sha256") != _hash_json(core):
            raise BoreasStage2SelectionError("candidate interval row SHA mismatch")
        interval_id = str(row.get("interval_id", ""))
        if not interval_id or interval_id in interval_ids:
            raise BoreasStage2SelectionError("candidate interval IDs must be unique")
        interval_ids.add(interval_id)
        if row.get("interval_valid") is True:
            for field in INTERVAL_SCORE_FIELDS:
                _finite_float(row.get(field), field)
            eligible.append(row)

    selected: list[dict[str, Any]] = []
    occupied: list[dict[str, Any]] = []
    for label in LABEL_ORDER:
        ranked = sorted(eligible, key=lambda row: _rank_key(row, label))
        selected_for_label: list[dict[str, Any]] = []
        for row in ranked:
            if _compatible(row, occupied + selected_for_label, contract):
                core = {
                    **row,
                    "scene_label": label,
                    "selection_rank_within_label": len(selected_for_label) + 1,
                }
                selected_for_label.append(
                    {**core, "selected_interval_row_sha256": _hash_json(core)}
                )
            if len(selected_for_label) == contract.interval_count_per_label:
                break
        if len(selected_for_label) != contract.interval_count_per_label:
            raise BoreasStage2SelectionError(
                f"STAGE2_SELECTION_FAIL: {label} interval count "
                f"{len(selected_for_label)} != {contract.interval_count_per_label}"
            )
        selected.extend(selected_for_label)
        occupied.extend(selected_for_label)
    if len({row["interval_id"] for row in selected}) != len(selected):
        raise BoreasStage2SelectionError("weak/rich selected intervals are not disjoint")
    return selected


def _snapshot_prefix(label: str) -> str:
    if label == WEAK_LABEL:
        return "weak"
    if label == RICH_LABEL:
        return "rich"
    raise BoreasStage2SelectionError("snapshot label is not weak/rich")


def select_interval_quantile_snapshots(
    candidate_scans: Sequence[Mapping[str, Any]],
    selected_intervals: Sequence[Mapping[str, Any]],
    *,
    contract: Stage2SelectionContract = Stage2SelectionContract(),
) -> list[dict[str, Any]]:
    """Choose 10/30/50/70/90% nearest unused valid frames per interval."""

    expected_intervals = 2 * contract.interval_count_per_label
    if len(selected_intervals) != expected_intervals:
        raise BoreasStage2SelectionError("selected interval count differs from 10+10")
    by_interval: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidate_scans:
        if row.get("geometry_valid") is True:
            by_interval.setdefault(str(row["interval_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for interval in selected_intervals:
        interval_core = {
            key: value for key, value in interval.items() if key != "selected_interval_row_sha256"
        }
        if interval.get("selected_interval_row_sha256") != _hash_json(interval_core):
            raise BoreasStage2SelectionError("selected interval row SHA mismatch")
        interval_id = str(interval["interval_id"])
        rows = sorted(by_interval.get(interval_id, []), key=lambda row: int(row["timestamp_us"]))
        if len(rows) < contract.snapshots_per_interval:
            raise BoreasStage2SelectionError("selected interval has fewer than five valid scans")
        timestamps = [int(row["timestamp_us"]) for row in rows]
        if len(timestamps) != len(set(timestamps)):
            raise BoreasStage2SelectionError("valid scan timestamps are not unique")
        array = np.asarray(timestamps, dtype=np.float64)
        chosen: set[int] = set()
        label = str(interval["scene_label"])
        prefix = _snapshot_prefix(label)
        interval_rank = _strict_int(
            interval["selection_rank_within_label"], "selection_rank_within_label", minimum=1
        )
        by_timestamp = {int(row["timestamp_us"]): row for row in rows}
        for quantile_index, probability in enumerate(contract.quantile_probabilities):
            target = float(np.quantile(array, probability, method=contract.quantile_method))
            available = [timestamp for timestamp in timestamps if timestamp not in chosen]
            selected_timestamp = min(
                available, key=lambda timestamp: (abs(float(timestamp) - target), timestamp)
            )
            chosen.add(selected_timestamp)
            source = by_timestamp[selected_timestamp]
            percent = int(round(probability * 100))
            core = {
                "selection_index": len(output),
                "snapshot_id": f"boreas-v2-{prefix}-{interval_rank:02d}-q{percent:02d}",
                "scene_label": label,
                "interval_id": interval_id,
                "interval_selection_rank": interval_rank,
                "selected_interval_row_sha256": interval["selected_interval_row_sha256"],
                "quantile_index": quantile_index,
                "quantile_probability": probability,
                "quantile_method": contract.quantile_method,
                "quantile_target_timestamp_us": target,
                "selected_timestamp_us": selected_timestamp,
                "absolute_quantile_delta_us": abs(float(selected_timestamp) - target),
                "object_key": source["object_key"],
                "query_ordinal": source["query_ordinal"],
                "first_pass_geometry_row_sha256": source["candidate_scan_row_sha256"],
                "selection_contract_sha256": contract.sha256,
            }
            output.append({**core, "selected_snapshot_row_sha256": _hash_json(core)})
    expected_snapshots = expected_intervals * contract.snapshots_per_interval
    if len(output) != expected_snapshots or len(
        {row["object_key"] for row in output}
    ) != expected_snapshots:
        raise BoreasStage2SelectionError("snapshot selection is not exactly 100 unique objects")
    label_counts = {
        label: sum(row["scene_label"] == label for row in output) for label in LABEL_ORDER
    }
    if label_counts != {WEAK_LABEL: 50, RICH_LABEL: 50}:
        raise BoreasStage2SelectionError("snapshot selection is not 50 weak + 50 rich")
    return output


def build_blind_selection_manifest(
    *,
    candidate_scans: Sequence[Mapping[str, Any]],
    candidate_intervals: Sequence[Mapping[str, Any]],
    selected_intervals: Sequence[Mapping[str, Any]],
    selected_snapshots: Sequence[Mapping[str, Any]],
    contract: Stage2SelectionContract,
    bindings: SelectionBindings,
) -> dict[str, Any]:
    """Bind the complete geometry-only selection prior to canonicalization."""

    bindings = bindings.validated()
    core = {
        "schema": BOREAS_STAGE2_SELECTION_SCHEMA,
        "selection_contract": contract.as_dict(),
        "selection_contract_sha256": contract.sha256,
        "bindings": asdict(bindings),
        "candidate_scan_count": len(candidate_scans),
        "geometry_valid_scan_count": sum(
            row.get("geometry_valid") is True for row in candidate_scans
        ),
        "candidate_scan_rows_sha256": _hash_json(
            [row["candidate_scan_row_sha256"] for row in candidate_scans]
        ),
        "candidate_interval_count": len(candidate_intervals),
        "eligible_interval_count": sum(
            row.get("interval_valid") is True for row in candidate_intervals
        ),
        "candidate_interval_rows_sha256": _hash_json(
            [row["candidate_interval_row_sha256"] for row in candidate_intervals]
        ),
        "selected_interval_count": len(selected_intervals),
        "selected_interval_rows_sha256": _hash_json(
            [row["selected_interval_row_sha256"] for row in selected_intervals]
        ),
        "weak_interval_count": sum(
            row["scene_label"] == WEAK_LABEL for row in selected_intervals
        ),
        "rich_interval_count": sum(
            row["scene_label"] == RICH_LABEL for row in selected_intervals
        ),
        "snapshot_count": len(selected_snapshots),
        "weak_snapshot_count": sum(
            row["scene_label"] == WEAK_LABEL for row in selected_snapshots
        ),
        "rich_snapshot_count": sum(
            row["scene_label"] == RICH_LABEL for row in selected_snapshots
        ),
        "selected_snapshot_rows_sha256": _hash_json(
            [row["selected_snapshot_row_sha256"] for row in selected_snapshots]
        ),
        "selector_visibility": "GEOMETRY_ONLY_NO_REGISTRATION_FIELDS",
        "registration_execution_count": 0,
        "r14_selection_frozen": True,
    }
    return {**core, "blind_selection_manifest_sha256": _hash_json(core)}


def candidate_scan_csv_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the exact scalar CSV projection of candidate scan rows."""

    return [
        {field: row[field] for field in CANDIDATE_SCAN_CSV_FIELDS}
        for row in rows
    ]


def geometry_metric_csv_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the selector-visible geometry-only CSV projection."""

    return [
        {field: row[field] for field in GEOMETRY_METRIC_CSV_FIELDS}
        for row in rows
    ]


def candidate_interval_csv_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten midpoint pose vectors into an exact scalar interval CSV schema."""

    output: list[dict[str, Any]] = []
    for row in rows:
        position = _finite_xyz(row["center_world_position"], "interval center")
        quaternion = np.asarray(row["center_world_quaternion_xyzw"], dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise BoreasStage2SelectionError("interval center quaternion is invalid")
        value = {
            field: row[field]
            for field in CANDIDATE_INTERVAL_CSV_FIELDS
            if field
            not in {
                "center_world_x_m",
                "center_world_y_m",
                "center_world_z_m",
                "center_quaternion_x",
                "center_quaternion_y",
                "center_quaternion_z",
                "center_quaternion_w",
            }
        }
        value.update(
            {
                "center_world_x_m": position[0],
                "center_world_y_m": position[1],
                "center_world_z_m": position[2],
                "center_quaternion_x": float(quaternion[0]),
                "center_quaternion_y": float(quaternion[1]),
                "center_quaternion_z": float(quaternion[2]),
                "center_quaternion_w": float(quaternion[3]),
            }
        )
        output.append({field: value[field] for field in CANDIDATE_INTERVAL_CSV_FIELDS})
    return output


def selected_interval_csv_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the exact selected-interval CSV projection."""

    output: list[dict[str, Any]] = []
    for row in rows:
        position = _finite_xyz(row["center_world_position"], "selected interval center")
        value = {
            field: row[field]
            for field in SELECTED_INTERVAL_CSV_FIELDS
            if field not in {"center_world_x_m", "center_world_y_m", "center_world_z_m"}
        }
        value.update(
            {
                "center_world_x_m": position[0],
                "center_world_y_m": position[1],
                "center_world_z_m": position[2],
            }
        )
        output.append({field: value[field] for field in SELECTED_INTERVAL_CSV_FIELDS})
    return output


def selected_snapshot_csv_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the exact selected-snapshot CSV projection."""

    return [
        {field: row[field] for field in SELECTED_SNAPSHOT_CSV_FIELDS}
        for row in rows
    ]


__all__ = [
    "BOREAS_STAGE2_SELECTION_SCHEMA",
    "BoreasStage2SelectionError",
    "CANDIDATE_INTERVAL_CSV_FIELDS",
    "CANDIDATE_SCAN_CSV_FIELDS",
    "FIRST_PASS_SCAN_FIELDS",
    "GEOMETRY_ONLY_FIELDS",
    "GEOMETRY_METRIC_CSV_FIELDS",
    "INTERVAL_SCORE_FIELDS",
    "LABEL_ORDER",
    "PRODUCTION_AUTHORITY",
    "RICH_LABEL",
    "ReferencePoseSeries",
    "SPECTRAL_FIELDS",
    "SYNTHETIC_AUTHORITY",
    "SelectionBindings",
    "SELECTED_INTERVAL_CSV_FIELDS",
    "SELECTED_SNAPSHOT_CSV_FIELDS",
    "Stage2SelectionContract",
    "TargetGeometryContext",
    "WEAK_LABEL",
    "build_blind_selection_manifest",
    "build_candidate_intervals",
    "build_candidate_scan_inventory",
    "candidate_interval_csv_rows",
    "candidate_scan_csv_rows",
    "compute_geometry_only_initial_metrics",
    "geometry_metric_csv_rows",
    "select_frozen_intervals",
    "select_interval_quantile_snapshots",
    "selected_interval_csv_rows",
    "selected_snapshot_csv_rows",
]
