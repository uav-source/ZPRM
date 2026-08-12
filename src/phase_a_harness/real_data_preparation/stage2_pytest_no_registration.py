"""Fixed pytest bootstrap that keeps Stage-2's NO-ICP guard in the child.

This module is loaded only through the formal finalization command's explicit
``-p`` argument.  It writes one canonical attestation before pytest exits;
the parent binds those bytes to the JUnit/result/status crash-recovery chain.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .guard import NoRegistrationGuard
from .io import atomic_write_json, sha256_file
from .stage2_safe_fixture_replay import (
    authenticate_formal_stage2_environment,
    initialize_safe_replay_event_log,
    remove_safe_replay_event_lock,
    summarize_safe_replay_event_log,
)


CHILD_ATTESTATION_SCHEMA = (
    "zprm.boreas.v2.stage2.full_test_child_no_registration.v2"
)
CHILD_ATTESTATION_FIELDS = frozenset(
    {
        "actual_open3d_trials",
        "actual_pcl_trials",
        "actual_trials",
        "estimated_transform_count",
        "estimated_transform_evidence",
        "estimated_transform_file_count",
        "estimated_transform_files",
        "guard_implementation_path",
        "guard_implementation_sha256",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pass",
        "pcl_cli_invocation_count",
        "plugin_path",
        "plugin_sha256",
        "pytest_exit_status_at_attestation",
        "real_trial_result_count",
        "registration_execution_count",
        "safe_fixture_replay_catalog_sha256",
        "safe_fixture_replay_event_chain_head_sha256",
        "safe_fixture_replay_event_count",
        "safe_fixture_replay_event_log_path",
        "safe_fixture_replay_event_log_sha256",
        "safe_fixture_replay_event_schema",
        "safe_fixture_replay_marker_path",
        "safe_fixture_replay_marker_sha256",
        "safe_simulated_materialization_count",
        "safe_simulated_open3d_materialization_count",
        "safe_simulated_pcl_materialization_count",
        "schema",
        "status",
        "structured_result_scan_error_count",
        "structured_result_scan_error_files",
    }
)
ATTESTATION_ENV = "ZPRM_STAGE2_PYTEST_GUARD_ATTESTATION"
SCAN_ROOT_ENV = "ZPRM_STAGE2_PYTEST_GUARD_SCAN_ROOT"
_guard: NoRegistrationGuard | None = None
_scan_root: Path | None = None
_attestation_path: Path | None = None


def _canonical_empty_directory(raw: str | None, *, label: str) -> Path:
    if raw is None:
        raise RuntimeError(f"{label} environment binding is absent")
    path = Path(raw)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
        or path.resolve(strict=True) != path
        or any(path.iterdir())
    ):
        raise RuntimeError(f"{label} must be an empty canonical directory")
    return path


def pytest_sessionstart(session: Any) -> None:  # pragma: no cover - child hook
    del session
    global _guard, _scan_root, _attestation_path
    _scan_root = _canonical_empty_directory(
        os.environ.get(SCAN_ROOT_ENV), label="child NO-ICP scan root"
    )
    raw_attestation = os.environ.get(ATTESTATION_ENV)
    if raw_attestation is None:
        raise RuntimeError("child NO-ICP attestation path is absent")
    attestation = Path(raw_attestation)
    if (
        not attestation.is_absolute()
        or attestation.is_symlink()
        or attestation.parent != _scan_root.parent
        or attestation.exists()
    ):
        raise RuntimeError("child NO-ICP attestation path is unsafe")
    binding = authenticate_formal_stage2_environment(required=True)
    if binding is None or binding["event_log_path"].parent != _scan_root.parent:
        raise RuntimeError("formal Stage-2 safe replay marker is not co-located")
    initialize_safe_replay_event_log()
    try:
        import open3d as open3d_module  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - formal environment includes Open3D
        open3d_module = None
    _attestation_path = attestation
    _guard = NoRegistrationGuard(open3d_module=open3d_module)
    _guard.__enter__()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:  # pragma: no cover
    del session
    if _guard is None or _scan_root is None or _attestation_path is None:
        raise RuntimeError("child NO-ICP guard was not initialized")
    if not _guard.active:
        raise RuntimeError("child NO-ICP guard became inactive")
    value = _guard.attestation(_scan_root)
    replay = summarize_safe_replay_event_log()
    plugin_path = Path(__file__).resolve(strict=True)
    guard_path = Path(__file__).with_name("guard.py").resolve(strict=True)
    complete = {
        **value,
        **replay,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_trials": 0,
        "guard_implementation_path": str(guard_path),
        "guard_implementation_sha256": sha256_file(guard_path),
        "plugin_path": str(plugin_path),
        "plugin_sha256": sha256_file(plugin_path),
        "pytest_exit_status_at_attestation": int(exitstatus),
        "schema": CHILD_ATTESTATION_SCHEMA,
        "status": "PASS" if value.get("pass") is True else "FAIL",
    }
    if set(complete) != CHILD_ATTESTATION_FIELDS:
        raise RuntimeError("child NO-registration attestation schema changed")
    atomic_write_json(_attestation_path, complete, overwrite=False)
    remove_safe_replay_event_lock()


def pytest_unconfigure(config: Any) -> None:  # pragma: no cover - child hook
    del config
    global _guard
    if _guard is not None and _guard.active:
        _guard.__exit__(None, None, None)
