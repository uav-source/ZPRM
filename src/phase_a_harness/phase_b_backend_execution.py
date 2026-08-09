"""Condition-only Phase B bridge to the frozen Phase A backend executors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .phase_a_execution_chain_audit import (
    execute_open3d_fixture,
    execute_pcl_fixture,
)
from .phase_a_trial_result_schema import OPEN3D_BACKEND, PCL_BACKEND
from .phase_b_trial_result import (
    PHASE_B_CONDITIONS,
    validate_phase_b_trial_result_strict,
)


_PHASE_A_VALIDATION_CONDITION = "IDEAL_MATCHED"


def _normalized_common(
    fixture: Any, common: Mapping[str, Any], *, expected_backend: str
) -> tuple[dict[str, Any], str]:
    condition = common.get("condition")
    if condition not in PHASE_B_CONDITIONS:
        raise PermissionError("backend execution requires an authorized Phase B condition")
    if common.get("backend") != expected_backend:
        raise PermissionError("backend identity does not match the selected executor")
    if getattr(fixture, "condition", None) != condition:
        raise ValueError("fixture/common condition pairing mismatch")
    normalized = dict(common)
    normalized["condition"] = _PHASE_A_VALIDATION_CONDITION
    return normalized, str(condition)


def _restore_condition_only(
    frozen_result: Mapping[str, Any], condition: str
) -> dict[str, Any]:
    # The frozen executor must first prove its complete native result contract.
    if frozen_result.get("condition") != _PHASE_A_VALIDATION_CONDITION:
        raise RuntimeError("frozen backend executor returned an unexpected condition")
    restored = dict(frozen_result)
    restored["condition"] = condition
    return validate_phase_b_trial_result_strict(restored)


def execute_phase_b_open3d_fixture(
    *,
    fixture: Any,
    common: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the frozen Open3D executor with only condition normalization."""

    normalized, condition = _normalized_common(
        fixture, common, expected_backend=OPEN3D_BACKEND
    )
    frozen_result = execute_open3d_fixture(
        fixture=fixture,
        common=normalized,
        parameters=parameters,
    )
    return _restore_condition_only(frozen_result, condition)


def execute_phase_b_pcl_fixture(
    *,
    fixture: Any,
    common: Mapping[str, Any],
    parameters: Mapping[str, Any],
    pcl_cli: Path,
) -> dict[str, Any]:
    """Run the frozen PCL executor with only condition normalization."""

    normalized, condition = _normalized_common(
        fixture, common, expected_backend=PCL_BACKEND
    )
    frozen_result = execute_pcl_fixture(
        fixture=fixture,
        common=normalized,
        parameters=parameters,
        pcl_cli=pcl_cli,
    )
    return _restore_condition_only(frozen_result, condition)


__all__ = [
    "execute_phase_b_open3d_fixture",
    "execute_phase_b_pcl_fixture",
]
