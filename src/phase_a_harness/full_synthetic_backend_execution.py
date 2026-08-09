"""Condition-only bridge to the already frozen Phase A backend executors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .full_synthetic_trial_result import (
    FULL_SYNTHETIC_CONDITIONS,
    validate_full_synthetic_trial_result_strict,
)
from .phase_a_execution_chain_audit import execute_open3d_fixture, execute_pcl_fixture
from .phase_a_trial_result_schema import OPEN3D_BACKEND, PCL_BACKEND


def _normalized_common(
    fixture: Any, common: Mapping[str, Any], expected_backend: str
) -> tuple[dict[str, Any], str]:
    condition = common.get("condition")
    if condition not in FULL_SYNTHETIC_CONDITIONS:
        raise PermissionError("backend execution condition is unauthorized")
    if common.get("backend") != expected_backend:
        raise PermissionError("backend identity mismatch")
    if getattr(fixture, "condition", None) != condition:
        raise ValueError("fixture/common condition mismatch")
    normalized = dict(common)
    normalized["condition"] = "IDEAL_MATCHED"
    return normalized, str(condition)


def _restore(value: Mapping[str, Any], condition: str) -> dict[str, Any]:
    if value.get("condition") != "IDEAL_MATCHED":
        raise RuntimeError("frozen executor condition normalization failed")
    restored = dict(value)
    restored["condition"] = condition
    return validate_full_synthetic_trial_result_strict(restored)


def execute_full_synthetic_open3d_fixture(
    *, fixture: Any, common: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    normalized, condition = _normalized_common(fixture, common, OPEN3D_BACKEND)
    return _restore(
        execute_open3d_fixture(
            fixture=fixture, common=normalized, parameters=parameters
        ),
        condition,
    )


def execute_full_synthetic_pcl_fixture(
    *,
    fixture: Any,
    common: Mapping[str, Any],
    parameters: Mapping[str, Any],
    pcl_cli: Path,
) -> dict[str, Any]:
    normalized, condition = _normalized_common(fixture, common, PCL_BACKEND)
    return _restore(
        execute_pcl_fixture(
            fixture=fixture,
            common=normalized,
            parameters=parameters,
            pcl_cli=pcl_cli,
        ),
        condition,
    )


__all__ = [
    "execute_full_synthetic_open3d_fixture",
    "execute_full_synthetic_pcl_fixture",
]
