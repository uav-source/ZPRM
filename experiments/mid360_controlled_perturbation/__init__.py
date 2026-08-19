"""Frozen, nonformal Mid-360 controlled-perturbation pilot."""

from .benchmark import (
    BACKEND_CONTRACT_SHA256,
    MAGNITUDES_M,
    RECOVERY_ROTATION_DEG,
    RECOVERY_TRANSLATION_M,
    build_run_id,
    construct_translation_perturbation,
    deterministic_eigendirections,
    pose_recovered,
)

__all__ = [
    "BACKEND_CONTRACT_SHA256",
    "MAGNITUDES_M",
    "RECOVERY_ROTATION_DEG",
    "RECOVERY_TRANSLATION_M",
    "build_run_id",
    "construct_translation_perturbation",
    "deterministic_eigendirections",
    "pose_recovered",
]
