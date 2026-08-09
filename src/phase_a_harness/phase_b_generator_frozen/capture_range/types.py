"""Immutable records for directional capture-range measurements."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


PERTURBATION_TYPES = frozenset({"translation", "rotation"})


class FrozenDict(dict):
    """Small recursively immutable dict compatible with JSON/mapping callers."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("frozen mapping cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo):
        return self


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly_float_array(value, name="mapping_array", finite=False)
    return copy.deepcopy(value)


def _readonly_float_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    columns: int | None = None,
    finite: bool = True,
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if columns is not None and (array.ndim != 2 or array.shape[1] != columns):
        raise ValueError(f"{name} must have shape [N,{columns}], got {array.shape}")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _validated_unit_direction(value: Any, *, canonical_sign: bool = False) -> np.ndarray:
    direction = _readonly_float_array(value, name="direction", shape=(3,))
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("direction cannot be zero")
    normalized = np.array(direction / norm, dtype=np.float64, copy=True)
    if canonical_sign:
        nonzero = np.flatnonzero(np.abs(normalized) > 1.0e-12)
        if nonzero.size and float(normalized[int(nonzero[0])]) < 0.0:
            normalized = -normalized
    normalized.setflags(write=False)
    return normalized


def _validate_pose(value: Any, *, name: str, finite: bool) -> np.ndarray:
    pose = _readonly_float_array(value, name=name, finite=finite)
    if pose.shape == (8,):
        if finite and float(np.linalg.norm(pose[4:8])) <= 1.0e-12:
            raise ValueError(f"{name} quaternion cannot be zero")
        return pose
    if pose.shape == (4, 4):
        if finite:
            if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12):
                raise ValueError(f"{name} has an invalid homogeneous bottom row")
            rotation = pose[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=1.0e-8
            ):
                raise ValueError(f"{name} rotation block must be in SO(3)")
        return pose
    raise ValueError(f"{name} must have shape [8] or [4,4], got {pose.shape}")


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1


def _derived_direction_id(perturbation_type: str, direction: np.ndarray, side: int) -> str:
    component = int(np.argmax(np.abs(direction)))
    axis = ("x", "y", "z")[component]
    axis_sign = "positive" if float(direction[component]) >= 0.0 else "negative"
    side_name = "positive" if side > 0 else "negative"
    return f"{perturbation_type}_{axis}_{axis_sign}_{side_name}"


@dataclass(frozen=True)
class RegistrationSnapshot:
    """One fixed scan/map/reference/configuration measurement object.

    Poses may use the repository-native TUM row convention
    ``[timestamp, tx, ty, tz, qx, qy, qz, qw]`` or an equivalent 4x4 matrix.
    """

    snapshot_id: str
    scan_points: np.ndarray
    local_map_points: np.ndarray
    reference_pose: np.ndarray
    registration_config: dict[str, Any]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not str(self.snapshot_id):
            raise ValueError("snapshot_id must be non-empty")
        scan = _readonly_float_array(
            self.scan_points, name="scan_points", columns=3
        )
        local_map = _readonly_float_array(
            self.local_map_points, name="local_map_points", columns=3
        )
        if scan.shape[0] == 0 or local_map.shape[0] == 0:
            raise ValueError("scan_points and local_map_points must be non-empty")
        object.__setattr__(self, "snapshot_id", str(self.snapshot_id))
        object.__setattr__(self, "scan_points", scan)
        object.__setattr__(self, "local_map_points", local_map)
        object.__setattr__(
            self,
            "reference_pose",
            _validate_pose(self.reference_pose, name="reference_pose", finite=True),
        )
        object.__setattr__(self, "registration_config", _deep_freeze(self.registration_config))
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


@dataclass(frozen=True)
class PerturbationSpec:
    """A signed, one-family perturbation with an explicit curve identity.

    ``direction`` is a unit basis direction and ``signed_amplitude`` carries
    the side.  At zero amplitude the side cannot be inferred, so callers must
    supply both ``direction_id`` and ``signed_side`` to keep the two zero rows
    separate.
    """

    perturbation_type: str
    direction: np.ndarray
    signed_amplitude: float
    repeat_index: int
    seed: int
    direction_id: str = ""
    signed_side: int | None = None

    def __post_init__(self) -> None:
        perturbation_type = str(self.perturbation_type)
        if perturbation_type not in PERTURBATION_TYPES:
            raise ValueError(
                f"perturbation_type must be one of {sorted(PERTURBATION_TYPES)}"
            )
        amplitude = float(self.signed_amplitude)
        if not math.isfinite(amplitude):
            raise ValueError("signed_amplitude must be finite")
        if int(self.repeat_index) < 0:
            raise ValueError("repeat_index must be non-negative")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        direction = _validated_unit_direction(self.direction, canonical_sign=True)
        if self.signed_side is None:
            if amplitude == 0.0:
                raise ValueError("zero amplitude requires explicit signed_side")
            side = _sign(amplitude)
        else:
            side = int(self.signed_side)
        if side not in {-1, 1}:
            raise ValueError("signed_side must be -1 or +1")
        if amplitude != 0.0 and _sign(amplitude) != side:
            raise ValueError("signed_side must match signed_amplitude")
        direction_id = str(self.direction_id)
        if amplitude == 0.0 and not direction_id:
            raise ValueError("zero amplitude requires explicit direction_id")
        if not direction_id:
            direction_id = _derived_direction_id(perturbation_type, direction, side)
        object.__setattr__(self, "perturbation_type", perturbation_type)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "signed_amplitude", amplitude)
        object.__setattr__(self, "repeat_index", int(self.repeat_index))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "direction_id", direction_id)
        object.__setattr__(self, "signed_side", side)

    @property
    def amplitude(self) -> float:
        return abs(self.signed_amplitude)

    @property
    def perturbation_vector(self) -> np.ndarray:
        value = np.array(
            self.direction * self.signed_amplitude, dtype=np.float64, copy=True
        )
        value.setflags(write=False)
        return value

    @property
    def direction_group_key(self) -> tuple[str, str, int]:
        return self.perturbation_type, self.direction_id, int(self.signed_side)


@dataclass(frozen=True)
class RecoveryTrialResult:
    snapshot_id: str
    perturbation_type: str
    direction_id: str
    signed_amplitude: float
    repeat_index: int
    initial_pose: np.ndarray
    final_pose: np.ndarray
    translation_error_m: float
    rotation_error_rad: float
    final_cost: float
    correspondence_count: int
    iteration_count: int
    solver_converged: bool
    finite_result: bool
    success: bool
    runtime_ms: float
    full_reassociation: bool
    failure_reason: str
    initial_correspondence_count: int = 0
    final_correspondence_count: int | None = None
    correspondence_checksum: str = ""
    initial_correspondence_checksum: str = ""
    initial_cost: float = float("nan")
    termination_reason: str = ""
    iteration_limit_not_failed: bool = True
    full_reassociation_count: int = 0
    baseline_only: bool = False
    plane_fit_count: int = 0
    transform_count: int = 0
    nearest_neighbor_search_count: int = 0
    correspondence_build_count: int = 0
    jacobian_recompute_count: int = 0
    jacobian_recomputation_count: int | None = None
    signed_side: int | None = None
    correspondence_checksum_trace: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if str(self.perturbation_type) not in PERTURBATION_TYPES:
            raise ValueError("invalid perturbation_type")
        amplitude = float(self.signed_amplitude)
        if not math.isfinite(amplitude):
            raise ValueError("signed_amplitude must be finite")
        if self.signed_side is None:
            if amplitude == 0.0:
                raise ValueError("zero-amplitude result requires explicit signed_side")
            side = _sign(amplitude)
        else:
            side = int(self.signed_side)
        if side not in {-1, 1}:
            raise ValueError("signed_side must be -1 or +1")
        if amplitude != 0.0 and _sign(amplitude) != side:
            raise ValueError("signed_side must match signed_amplitude")
        final_count = (
            int(self.correspondence_count)
            if self.final_correspondence_count is None
            else int(self.final_correspondence_count)
        )
        jacobian_count = (
            int(self.jacobian_recompute_count)
            if self.jacobian_recomputation_count is None
            else int(self.jacobian_recomputation_count)
        )
        if int(self.jacobian_recompute_count) not in {0, jacobian_count}:
            raise ValueError("Jacobian recomputation count aliases disagree")
        integer_counts = (
            self.repeat_index,
            self.correspondence_count,
            self.iteration_count,
            self.initial_correspondence_count,
            final_count,
            self.full_reassociation_count,
            self.plane_fit_count,
            self.transform_count,
            self.nearest_neighbor_search_count,
            self.correspondence_build_count,
            jacobian_count,
        )
        if any(int(value) < 0 for value in integer_counts):
            raise ValueError("trial indices and counts must be non-negative")
        if bool(self.full_reassociation) and bool(self.baseline_only):
            raise ValueError("formal and baseline path flags are mutually exclusive")
        checksum_trace = tuple(str(value) for value in self.correspondence_checksum_trace)
        if bool(self.full_reassociation) and len(checksum_trace) != int(
            self.full_reassociation_count
        ):
            raise ValueError("formal correspondence trace must cover every reassociation pass")
        if bool(self.baseline_only) and checksum_trace:
            raise ValueError("frozen baseline trial cannot contain a reassociation trace")
        object.__setattr__(self, "snapshot_id", str(self.snapshot_id))
        object.__setattr__(self, "perturbation_type", str(self.perturbation_type))
        object.__setattr__(self, "direction_id", str(self.direction_id))
        object.__setattr__(self, "signed_amplitude", amplitude)
        object.__setattr__(self, "repeat_index", int(self.repeat_index))
        object.__setattr__(
            self,
            "initial_pose",
            _validate_pose(self.initial_pose, name="initial_pose", finite=True),
        )
        object.__setattr__(
            self,
            "final_pose",
            _validate_pose(self.final_pose, name="final_pose", finite=False),
        )
        object.__setattr__(self, "correspondence_count", int(self.correspondence_count))
        object.__setattr__(self, "iteration_count", int(self.iteration_count))
        object.__setattr__(
            self, "initial_correspondence_count", int(self.initial_correspondence_count)
        )
        object.__setattr__(self, "final_correspondence_count", final_count)
        object.__setattr__(self, "full_reassociation_count", int(self.full_reassociation_count))
        object.__setattr__(self, "plane_fit_count", int(self.plane_fit_count))
        object.__setattr__(self, "transform_count", int(self.transform_count))
        object.__setattr__(
            self,
            "nearest_neighbor_search_count",
            int(self.nearest_neighbor_search_count),
        )
        object.__setattr__(
            self, "correspondence_build_count", int(self.correspondence_build_count)
        )
        object.__setattr__(
            self, "jacobian_recompute_count", jacobian_count
        )
        object.__setattr__(self, "jacobian_recomputation_count", jacobian_count)
        object.__setattr__(self, "signed_side", side)
        object.__setattr__(self, "correspondence_checksum_trace", checksum_trace)


@dataclass(frozen=True)
class DirectionRecoveryCurve:
    direction_id: str
    direction: np.ndarray
    signed_side: int
    amplitudes: np.ndarray
    raw_probabilities: np.ndarray
    fitted_probabilities: np.ndarray
    wilson_lower: np.ndarray
    wilson_upper: np.ndarray
    d50: float | None
    d90: float | None
    d50_right_censored: bool
    d90_right_censored: bool
    snapshot_id: str = ""
    perturbation_type: str = ""
    registration_path: str = "full_reassociation"
    full_reassociation: bool = True
    baseline_only: bool = False

    def __post_init__(self) -> None:
        side = int(self.signed_side)
        if side not in {-1, 1}:
            raise ValueError("signed_side must be -1 or +1")
        direction = _validated_unit_direction(self.direction)
        arrays = {
            "amplitudes": _readonly_float_array(
                self.amplitudes, name="amplitudes", finite=True
            ),
            "raw_probabilities": _readonly_float_array(
                self.raw_probabilities, name="raw_probabilities", finite=True
            ),
            "fitted_probabilities": _readonly_float_array(
                self.fitted_probabilities, name="fitted_probabilities", finite=True
            ),
            "wilson_lower": _readonly_float_array(
                self.wilson_lower, name="wilson_lower", finite=True
            ),
            "wilson_upper": _readonly_float_array(
                self.wilson_upper, name="wilson_upper", finite=True
            ),
        }
        if any(array.ndim != 1 for array in arrays.values()):
            raise ValueError("curve arrays must be one-dimensional")
        sizes = {array.size for array in arrays.values()}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError("curve arrays must be equal-length and non-empty")
        amplitudes = arrays["amplitudes"]
        if np.any(amplitudes < 0.0) or np.any(np.diff(amplitudes) < 0.0):
            raise ValueError("amplitudes must be non-negative and sorted")
        for name in (
            "raw_probabilities",
            "fitted_probabilities",
            "wilson_lower",
            "wilson_upper",
        ):
            if np.any((arrays[name] < 0.0) | (arrays[name] > 1.0)):
                raise ValueError(f"{name} must lie in [0,1]")
        if np.any(np.diff(arrays["fitted_probabilities"]) > 1.0e-12):
            raise ValueError("fitted_probabilities must be non-increasing")
        if np.any(arrays["wilson_lower"] > arrays["wilson_upper"]):
            raise ValueError("Wilson lower bound exceeds upper bound")
        for name, value, censored in (
            ("d50", self.d50, self.d50_right_censored),
            ("d90", self.d90, self.d90_right_censored),
        ):
            if bool(censored) != (value is None):
                raise ValueError(f"{name} must be None exactly when right-censored")
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be a finite non-negative amplitude")
        if bool(self.full_reassociation) and bool(self.baseline_only):
            raise ValueError("formal and baseline curve flags are mutually exclusive")
        object.__setattr__(self, "direction_id", str(self.direction_id))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "signed_side", side)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        if self.d50 is not None:
            object.__setattr__(self, "d50", float(self.d50))
        if self.d90 is not None:
            object.__setattr__(self, "d90", float(self.d90))

    @property
    def direction_group_key(self) -> tuple[str, str, str, int, str]:
        return (
            self.snapshot_id,
            self.perturbation_type,
            self.direction_id,
            self.signed_side,
            self.registration_path,
        )
