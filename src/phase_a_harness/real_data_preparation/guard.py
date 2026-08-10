"""Three-layer prohibition on registration during data preparation."""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping


class RegistrationForbiddenError(PermissionError):
    """A registration operation was requested during preparation."""


DENIED_PROCESS_TOKENS = (
    "pcl_point_to_plane_cli",
    "registration_icp",
    "kiss_icp",
    "kiss-icp",
    "lio_sam",
    "lio-sam",
    "fast_lio",
    "fast-lio",
    "dlio",
    "gicp",
    "ndt",
    "icp",
)

DENIED_IMPORT_FRAGMENTS = (
    "phase_a_execution_chain_audit",
    "full_synthetic_backend_execution",
    "synthetic_confirmatory_v3_runner",
    "open3d.pipelines.registration",
)

OPEN3D_REGISTRATION_ENTRY_PREFIXES = (
    "registration_",
    "get_information_matrix_",
)


def _command_text(command: Any) -> str:
    if isinstance(command, (tuple, list)):
        return " ".join(os.fspath(value) for value in command).lower()
    return os.fspath(command).lower()


def _is_denied_process(command: Any) -> bool:
    text = _command_text(command)
    normalized = text.replace("-", "_")
    # Match command/path tokens on identifier boundaries.  A naive substring
    # test would, for example, mistake "grandtour" for the NDT executable.
    return any(
        re.search(
            rf"(?<![a-z0-9])(?:[a-z0-9]+_)*{re.escape(token.replace('-', '_'))}(?:_[a-z0-9]+)*(?![a-z0-9])",
            normalized,
        )
        is not None
        for token in DENIED_PROCESS_TOKENS
    )


def _forbidden_callable(*args: Any, **kwargs: Any) -> Any:
    raise RegistrationForbiddenError("registration is forbidden during real-data preparation")


def assert_preparation_sources_are_safe(source_root: str | Path) -> dict[str, Any]:
    """AST-audit preparation sources without naive token false positives."""

    root = Path(source_root).resolve(strict=True)
    violations: list[dict[str, Any]] = []
    scanned = 0
    for path in sorted(root.rglob("*.py")):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    if any(fragment in imported for fragment in DENIED_IMPORT_FRAGMENTS):
                        violations.append({"path": str(path), "line": node.lineno, "import": imported})
            elif isinstance(node, ast.ImportFrom):
                imported = node.module or ""
                if any(fragment in imported for fragment in DENIED_IMPORT_FRAGMENTS):
                    violations.append({"path": str(path), "line": node.lineno, "import": imported})
    if violations:
        raise RegistrationForbiddenError(f"unsafe preparation imports: {violations}")
    return {"pass": True, "python_file_count": scanned, "violations": []}


class NoRegistrationGuard:
    """Context manager patching subprocess and optional Open3D registration entrypoints."""

    def __init__(self, *, open3d_module: ModuleType | None = None) -> None:
        self.open3d_module = open3d_module
        self._original_subprocess: dict[str, Callable[..., Any]] = {}
        self._original_open3d: dict[str, Any] = {}
        self.denied_process_attempt_count = 0
        self.denied_open3d_attempt_count = 0
        self.active = False

    def _guard_process(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(command: Any, *args: Any, **kwargs: Any) -> Any:
            if _is_denied_process(command):
                self.denied_process_attempt_count += 1
                raise RegistrationForbiddenError(
                    f"forbidden registration process: {_command_text(command)}"
                )
            return original(command, *args, **kwargs)

        return guarded

    def _guard_open3d(self) -> None:
        module = self.open3d_module
        if module is None:
            return
        registration = getattr(getattr(module, "pipelines", None), "registration", None)
        if registration is None:
            return
        for name in dir(registration):
            if not name.startswith(OPEN3D_REGISTRATION_ENTRY_PREFIXES):
                continue
            value = getattr(registration, name)
            if not callable(value):
                continue
            self._original_open3d[name] = value

            def denied(*args: Any, _name: str = name, **kwargs: Any) -> Any:
                self.denied_open3d_attempt_count += 1
                raise RegistrationForbiddenError(f"Open3D registration entry blocked: {_name}")

            setattr(registration, name, denied)

    def __enter__(self) -> "NoRegistrationGuard":
        if os.environ.get("ZPRM_REAL_DATA_PREP_NO_REGISTRATION") != "1":
            raise RegistrationForbiddenError(
                "ZPRM_REAL_DATA_PREP_NO_REGISTRATION=1 is required"
            )
        if self.active:
            raise RegistrationForbiddenError("guard is already active")
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            original = getattr(subprocess, name)
            self._original_subprocess[name] = original
            setattr(subprocess, name, self._guard_process(original))
        self._guard_open3d()
        self.active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        module = self.open3d_module
        if module is not None:
            registration = getattr(getattr(module, "pipelines", None), "registration", None)
            if registration is not None:
                for name, value in self._original_open3d.items():
                    setattr(registration, name, value)
        for name, value in self._original_subprocess.items():
            setattr(subprocess, name, value)
        self.active = False

    def attestation(self, runtime_root: str | Path) -> dict[str, Any]:
        root = Path(runtime_root)
        estimated = []
        result_files = []
        if root.exists():
            estimated = sorted(
                str(path) for path in root.rglob("*")
                if path.is_file() and any(
                    token in path.name.lower()
                    for token in ("t_estimated", "final_transform", "translation_displacement", "rotation_displacement", "correspondence_turnover")
                )
            )
            result_files = sorted(str(path) for path in root.rglob("raw_results/*.json"))
        value = {
            "estimated_transform_file_count": len(estimated),
            "estimated_transform_files": estimated,
            "open3d_registration_call_count": self.denied_open3d_attempt_count,
            "other_registration_process_count": self.denied_process_attempt_count,
            "pcl_cli_invocation_count": self.denied_process_attempt_count,
            "real_trial_result_count": len(result_files),
            "registration_execution_count": 0,
        }
        value["pass"] = all(
            value[key] == 0
            for key in (
                "estimated_transform_file_count",
                "open3d_registration_call_count",
                "other_registration_process_count",
                "pcl_cli_invocation_count",
                "real_trial_result_count",
                "registration_execution_count",
            )
        )
        return value
