"""Fail-closed helpers for real-data preregistration preparation.

Nothing in this package executes a registration backend.  It only handles
source qualification, independent-reference geometry, deterministic scene
selection, canonical inputs, and immutable audit manifests.
"""

from .guard import NoRegistrationGuard, assert_preparation_sources_are_safe
from .overlap import OverlapContract, compute_gt_only_overlap
from .selection import select_scene_intervals

__all__ = [
    "NoRegistrationGuard",
    "OverlapContract",
    "assert_preparation_sources_are_safe",
    "compute_gt_only_overlap",
    "select_scene_intervals",
]
