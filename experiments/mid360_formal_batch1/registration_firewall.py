"""Fail-closed firewall for the FMB1 real-batch pre-backend boundary.

This module is deliberately narrower than a formal registration runner.  It
can authenticate preregistration evidence, inspect the production preflight
scope, block backend entry points, and detect result/lock artifacts.  It does
not construct a trial matrix, issue a lock, authorize registration, or invoke
either backend.

Synthetic fixtures are a separate authority.  Their paths and counters can be
validated with :func:`assert_fixture_only_isolated`, but fixture evidence is
never accepted by :func:`preflight_real_batch` and never contributes to the
real-batch execution counters.
"""

from __future__ import annotations

import ast
import builtins
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence


EXPECTED_BACKEND_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
REAL_RESULTS_RELATIVE = Path("results/mid360_formal_batch1")
REAL_RUNTIME_RELATIVE = Path(
    "zero_perturbation_runtime/mid360_formal_batch1_ingest_v1"
)
QUALIFICATION_RUNTIME_RELATIVE = Path(
    "zero_perturbation_runtime/mid360_formal_batch1_prebackend_qualification_v1"
)
REAL_SCOPE = "REAL_PRE_BACKEND"
FIXTURE_SCOPE = "SYNTHETIC_FIXTURE_ONLY"

_PRODUCTION_PREFLIGHT_FILES = (
    Path("experiments/mid360_formal_batch1/registration_firewall.py"),
    Path("tools/mid360_formal_batch1/preflight_formal_registration.py"),
)
_GUARD_IMPLEMENTATION = _PRODUCTION_PREFLIGHT_FILES[0]
_PRIMARY_NO_REGISTRATION_ENV = "ZPRM_FMB1_NO_FORMAL_REGISTRATION"
_LEGACY_NO_REGISTRATION_ENV = "NO_FORMAL_REGISTRATION"
_EXECUTION_SCOPE_ENV = "FMB1_EXECUTION_SCOPE"
_DENIED_TRUE_ENV = (
    "FMB1_FORMAL_REGISTRATION_AUTHORIZED",
    "FORMAL_REGISTRATION_AUTHORIZED",
    "FMB1_FORMAL_ICP_UNLOCKED",
    "FORMAL_ICP_UNLOCKED",
    "FMB1_BACKEND_AUTHORIZED",
)
_FIXTURE_ENV = (
    "FMB1_FIXTURE_ONLY",
    "FMB1_SYNTHETIC_FIXTURE_ONLY",
)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

_BACKEND_IMPORT_ROOTS = frozenset(
    {
        "open3d",
        "pcl",
        "pclpy",
        "open3d_backend",
        "pcl_backend",
        "debug_registration",
    }
)
_BACKEND_CALL_NAMES = frozenset(
    {
        "registration_icp",
        "registration_generalized_icp",
        "registration_ransac_based_on_feature_matching",
        "registration_fgr_based_on_feature_matching",
        "generalized_icp",
        "gicp",
        "ndt",
        "scan_matching",
        "scan_match",
        "run_open3d",
        "run_pcl",
        "run_backend",
        "execute_backend",
    }
)
_AUTHORITY_CALL_NAMES = frozenset(
    {
        "freeze_batch",
        "formal_icp_authorized",
        "require_formal_icp_authorization",
    }
)
_PROCESS_CALLS = frozenset(
    {
        "subprocess.Popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "os.system",
    }
)
_OPEN3D_CALL_PREFIXES = (
    "registration_",
    "get_information_matrix_",
)

_FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "t_est",
        "t_estimated",
        "estimated_transform",
        "estimated_pose",
        "translation_error",
        "translation_error_m",
        "rotation_error",
        "rotation_error_deg",
        "final_residual",
        "final_residual_rmse",
        "correspondence_turnover",
        "accepted_source_turnover",
        "fitness",
        "fitness_score",
        "solver_result",
        "solver_success",
        "registration_result",
        "registration_output",
        "inlier_rmse",
    }
)
_ZERO_ONLY_ARTIFACT_KEYS = frozenset(
    {
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "formal_trial_count",
        "registration_execution_count",
        "actual_registration_trials",
        "actual_trials",
        "formal_icp_executions",
        "backend_invocation_count",
        "actual_formal_trials",
        "real_t_est_file_count",
    }
)
_FALSE_ONLY_ARTIFACT_KEYS = frozenset(
    {
        "formal_registration_authorized",
        "formal_icp_unlocked",
        "formal_measurement_result",
        "measurement_final_result",
        "backend_authorized",
        "formal_authority",
        "actual_registration_execution",
        "formal_lock_issued",
        "formal_run_matrix_issued",
        "formal_batch_lock_issued",
        "trial_matrix_issued",
    }
)
_PROTECTED_ISSUANCE_FILENAMES = frozenset(
    {
        "formal_batch1_lock.json",
        "formal_batch1_fingerprint.json",
        "formal_registration_authorization.json",
        "fmb1_formal_registration_authorization.json",
        "formal_registration_trial_matrix.csv",
        "formal_registration_trial_matrix.json",
        "fmb1_formal_registration_trial_matrix.csv",
        "fmb1_formal_registration_trial_matrix.json",
    }
)
_ALLOWED_PRELOCK_STATUS_FILENAMES = frozenset({"formal_icp_status.json"})
_RESULT_NAME_TERMS = frozenset(
    {"result", "results", "trial", "trials", "transform", "pose", "metrics", "output"}
)
_BACKEND_NAME_TERMS = frozenset({"open3d", "pcl", "icp", "gicp", "ndt"})


class RegistrationFirewallError(RuntimeError):
    """A pre-backend firewall contract was violated."""


class BackendExecutionBlocked(PermissionError):
    """A backend entry point was blocked before the real call occurred."""


def _truthy(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _TRUE_VALUES


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_process_command(node: ast.Call) -> Any | None:
    callee = _dotted_name(node.func)
    if callee not in _PROCESS_CALLS or not node.args:
        return None
    try:
        return ast.literal_eval(node.args[0])
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _command_parts(command: Any) -> list[str]:
    if isinstance(command, (list, tuple)):
        raw = [os.fsdecode(os.fspath(value)) for value in command]
    else:
        raw = [os.fsdecode(os.fspath(command))]
    parts: list[str] = []
    for value in raw:
        parts.extend(candidate for candidate in re.split(r"[\s/\\]+", value) if candidate)
    return parts


def classify_backend_command(command: Any) -> str | None:
    """Classify an obvious backend process command without executing it."""

    normalized = [
        re.sub(r"\.(?:py|sh|bin|exe)$", "", part.lower().replace("-", "_"))
        for part in _command_parts(command)
    ]
    if any("preflight_formal_registration" in part for part in normalized):
        return None
    fixture_marker = any("fixture" in part for part in normalized)
    if any(
        part == "pcl_point_to_plane_cli"
        or part == "pcl_icp"
        or part == "pcl_gicp"
        or part == "pcl_ndt"
        or part.startswith("pcl_registration_")
        for part in normalized
    ):
        return "fixture" if fixture_marker else "pcl"
    if any(
        part in {
            "open3d",
            "open3d_backend",
            "registration_icp",
            "registration_generalized_icp",
            "run_open3d",
            "run_open3d_full",
        }
        or part.startswith("open3d_registration_")
        for part in normalized
    ):
        return "fixture" if fixture_marker else "open3d"
    if any(
        part in {
            "icp",
            "gicp",
            "ndt",
            "run_icp",
            "run_gicp",
            "run_ndt",
            "scan_matching",
            "debug_registration",
            "formal_registration_runner",
        }
        for part in normalized
    ):
        return "fixture" if fixture_marker else "other"
    for index, part in enumerate(normalized[:-1]):
        if part in {"--backend", "backend"} and normalized[index + 1] in {
            "open3d",
            "pcl",
            "icp",
            "gicp",
            "ndt",
        }:
            return "fixture" if fixture_marker else normalized[index + 1]
    return None


def _scan_python_source(
    path: Path, repository: Path, *, guard_implementation: bool
) -> list[dict[str, Any]]:
    relative = path.relative_to(repository).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [
            {
                "path": relative,
                "line": int(getattr(exc, "lineno", 0) or 0),
                "kind": "UNREADABLE_OR_INVALID_PYTHON",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        ]
    findings: list[dict[str, Any]] = []

    def add(node: ast.AST, kind: str, detail: str) -> None:
        findings.append(
            {
                "path": relative,
                "line": int(getattr(node, "lineno", 0) or 0),
                "kind": kind,
                "detail": detail,
            }
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0].lower() in _BACKEND_IMPORT_ROOTS:
                    add(node, "BACKEND_IMPORT", alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".", 1)[0].lower() in _BACKEND_IMPORT_ROOTS:
                add(node, "BACKEND_IMPORT", module)
            for alias in node.names:
                if alias.name in _AUTHORITY_CALL_NAMES:
                    add(node, "AUTHORITY_SYMBOL_IMPORT", alias.name)
        elif isinstance(node, ast.Call) and not guard_implementation:
            callee = _dotted_name(node.func)
            tail = callee.rsplit(".", 1)[-1]
            if tail in _BACKEND_CALL_NAMES:
                add(node, "BACKEND_CALL", callee)
            if tail in _AUTHORITY_CALL_NAMES:
                add(node, "AUTHORITY_CALL", callee)
            if callee in {"eval", "exec", "compile", "__import__", "importlib.import_module"}:
                add(node, "DYNAMIC_EXECUTION", callee)
            command = _literal_process_command(node)
            if command is not None and classify_backend_command(command) is not None:
                add(node, "BACKEND_PROCESS_LITERAL", repr(command))
        elif isinstance(node, ast.Dict) and not guard_implementation:
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    name = _normalized(key.value)
                    if _is_forbidden_artifact_key(name):
                        add(key, "BACKEND_RESULT_FIELD", key.value)
    unique = {
        (row["path"], row["line"], row["kind"], row["detail"]): row
        for row in findings
    }
    return [unique[key] for key in sorted(unique)]


def _declares_fixture_only(tree: ast.AST) -> bool:
    values: dict[str, Any] = {}
    for node in getattr(tree, "body", ()):  # top-level constants only
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    strict_marker = bool(
        values.get("FIXTURE_ONLY") is True
        and values.get("EXECUTION_SCOPE") == FIXTURE_SCOPE
        and values.get("ELIGIBLE_FOR_REAL_AUTHORIZATION") is False
    )
    lifecycle_marker = bool(
        values.get("FIXTURE_CLASSIFICATION") == "FIXTURE_ONLY_DO_NOT_CITE"
        and values.get("ACTUAL_REGISTRATION_EXECUTION") is False
    )
    return strict_marker or lifecycle_marker


def _discover_additional_prebackend_entries(repository: Path) -> list[Path]:
    patterns = (
        "*formal_registration*.py",
        "*prebackend*.py",
        "*backend_qualification*.py",
    )
    found: set[Path] = set()
    for directory in (
        repository / "experiments/mid360_formal_batch1",
        repository / "tools/mid360_formal_batch1",
    ):
        if not directory.is_dir():
            continue
        for pattern in patterns:
            found.update(path.resolve(strict=True) for path in directory.glob(pattern))
    return sorted(found)


def assert_prebackend_static_safe(
    repository: str | Path,
    *,
    extra_production_files: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """AST-scan only the real pre-backend production scope.

    Test/fixture runners are not implicitly scanned or trusted.  Callers may
    add real production files explicitly; every supplied file must remain
    inside the repository and outside ``tests`` and ``frozen_assets/fixtures``.
    """

    root = Path(repository).resolve(strict=True)
    if not root.is_dir():
        raise RegistrationFirewallError(f"repository is not a directory: {root}")
    candidates: list[Path] = []
    discovered = _discover_additional_prebackend_entries(root)
    for relative in (*_PRODUCTION_PREFLIGHT_FILES, *discovered, *extra_production_files):
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_symlink():
            raise RegistrationFirewallError(f"production source may not be a symlink: {candidate}")
        resolved = candidate.resolve(strict=True)
        if not _is_within(resolved, root):
            raise RegistrationFirewallError(f"production source escapes repository: {resolved}")
        relative_path = resolved.relative_to(root)
        if relative_path.parts and relative_path.parts[0] == "tests":
            raise RegistrationFirewallError("fixture/test source cannot enter real static scope")
        if relative_path.parts[:2] == ("frozen_assets", "fixtures"):
            raise RegistrationFirewallError("fixture assets cannot enter real static scope")
        if resolved not in candidates:
            candidates.append(resolved)
    findings: list[dict[str, Any]] = []
    fixture_sources: list[str] = []
    for path in candidates:
        try:
            source_tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            source_tree = ast.Module(body=[], type_ignores=[])
        fixture_only = _declares_fixture_only(source_tree)
        if fixture_only:
            fixture_sources.append(path.relative_to(root).as_posix())
        rows = _scan_python_source(
            path,
            root,
            guard_implementation=path.relative_to(root) == _GUARD_IMPLEMENTATION,
        )
        if fixture_only:
            # Explicit synthetic runners may name/import their fixture backend,
            # but may never import or call real lock/authorization authority.
            rows = [
                row
                for row in rows
                if row["kind"]
                not in {"BACKEND_IMPORT", "BACKEND_CALL", "BACKEND_PROCESS_LITERAL"}
            ]
        findings.extend(rows)
    report = {
        "schema": "mid360_fmb1_real_prebackend_static_firewall_v1",
        "status": "PASS" if not findings else "FAIL",
        "pass": not findings,
        "execution_scope": REAL_SCOPE,
        "fixture_only_sources_accepted": False,
        "repository": str(root),
        "scanned_file_count": len(candidates),
        "scanned_files": [path.relative_to(root).as_posix() for path in candidates],
        "fixture_only_scanned_files": fixture_sources,
        "fixture_only_sources_contribute_to_real_authority": False,
        "guard_implementation_excluded_from_executable_call_rules": str(
            _GUARD_IMPLEMENTATION
        ),
        "violation_count": len(findings),
        "violations": findings,
    }
    if findings:
        raise RegistrationFirewallError(f"unsafe real pre-backend source: {findings}")
    return report


def assert_fixture_only_isolated(
    repository: str | Path,
    fixture_root: str | Path,
) -> dict[str, Any]:
    """Validate a synthetic fixture root without granting real authority."""

    root = Path(repository).resolve(strict=True)
    supplied = Path(fixture_root)
    if supplied.is_symlink():
        raise RegistrationFirewallError("fixture root may not be a symlink")
    fixture = supplied.resolve(strict=True)
    if not fixture.is_dir():
        raise RegistrationFirewallError("fixture root must be a directory")
    real_roots = [
        (root / REAL_RESULTS_RELATIVE).resolve(),
        (root / REAL_RUNTIME_RELATIVE).resolve(),
        (root / QUALIFICATION_RUNTIME_RELATIVE).resolve(),
        (root / "bags").resolve(),
    ]
    if any(_is_within(fixture, real) or _is_within(real, fixture) for real in real_roots):
        raise RegistrationFirewallError("fixture root overlaps real FMB1 data/results")
    in_test_tree = _is_within(fixture, (root / "tests").resolve())
    in_tmp = _is_within(fixture, Path("/tmp").resolve())
    if not (in_test_tree or in_tmp):
        raise RegistrationFirewallError(
            "fixture root must be under repository tests/ or /tmp"
        )
    return {
        "schema": "mid360_fmb1_fixture_only_isolation_v1",
        "status": "PASS",
        "execution_scope": FIXTURE_SCOPE,
        "fixture_only": True,
        "fixture_root": str(fixture),
        "eligible_for_real_preflight": False,
        "eligible_for_real_authorization": False,
        "fixture_counters_contribute_to_real_batch": False,
        "real_open3d_registration_call_count": 0,
        "real_pcl_cli_invocation_count": 0,
        "real_other_registration_process_count": 0,
        "real_formal_trial_count": 0,
    }


def _is_forbidden_artifact_key(name: str) -> bool:
    if name in _FORBIDDEN_ARTIFACT_KEYS:
        return True
    return (
        name.startswith("t_est_")
        or "translation_error" in name
        or "rotation_error" in name
        or "final_residual" in name
        or "turnover" in name
        or name.startswith("fitness_")
        or name.endswith("_fitness")
        or "solver_result" in name
        or "registration_result" in name
        or "registration_output" in name
    )


def _is_result_artifact_name(name: str) -> bool:
    lower = name.lower()
    if lower in _ALLOWED_PRELOCK_STATUS_FILENAMES:
        return False
    if lower in _PROTECTED_ISSUANCE_FILENAMES:
        return True
    stem_terms = set(filter(None, re.split(r"[^a-z0-9]+", Path(lower).stem)))
    return bool(stem_terms & _BACKEND_NAME_TERMS) and bool(
        stem_terms & _RESULT_NAME_TERMS
    )


def _scan_artifact_value(
    value: Any,
    *,
    relative: str,
    json_path: str,
    findings: list[dict[str, Any]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = _normalized(key)
            child_path = f"{json_path}.{key}"
            if _is_forbidden_artifact_key(name):
                findings.append(
                    {
                        "path": relative,
                        "kind": "BACKEND_RESULT_FIELD",
                        "json_path": child_path,
                    }
                )
            if name in _ZERO_ONLY_ARTIFACT_KEYS and (
                isinstance(child, bool) or not isinstance(child, int) or child != 0
            ):
                findings.append(
                    {
                        "path": relative,
                        "kind": "NONZERO_REAL_EXECUTION_COUNTER",
                        "json_path": child_path,
                        "value": child,
                    }
                )
            inactive_value = child is False or child is None or (
                isinstance(child, str)
                and child.strip().lower() in {"", "0", "false", "no", "off"}
            )
            if name in _FALSE_ONLY_ARTIFACT_KEYS and not inactive_value:
                findings.append(
                    {
                        "path": relative,
                        "kind": "AUTHORIZATION_OR_LOCK_ACTIVE",
                        "json_path": child_path,
                        "value": child,
                    }
                )
            if name == "fixture_only" and child is True:
                findings.append(
                    {
                        "path": relative,
                        "kind": "FIXTURE_ARTIFACT_IN_REAL_SCOPE",
                        "json_path": child_path,
                    }
                )
            _scan_artifact_value(
                child,
                relative=relative,
                json_path=child_path,
                findings=findings,
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_artifact_value(
                child,
                relative=relative,
                json_path=f"{json_path}[{index}]",
                findings=findings,
            )


def _zero_real_counters(value: Mapping[str, Any]) -> bool:
    normalized = {_normalized(key): child for key, child in value.items()}
    for key in _ZERO_ONLY_ARTIFACT_KEYS:
        if key in normalized and normalized[key] != 0:
            return False
    return True


def _explicit_nonreal_json_scope(
    payload: Any,
    *,
    path: Path,
) -> str | None:
    """Recognize narrowly marked schema/template/fixture evidence.

    These artifacts may describe future result fields, but cannot carry real
    authority or contribute a real execution counter.  A mere
    ``fixture_only=true`` is intentionally insufficient to prevent a fake real
    result from escaping the scanner.
    """

    if not isinstance(payload, Mapping):
        return None
    normalized_payload = {_normalized(key): value for key, value in payload.items()}
    no_authority = all(
        normalized_payload.get(key, False) is False
        for key in _FALSE_ONLY_ARTIFACT_KEYS
    )
    if not no_authority or not _zero_real_counters(payload):
        return None
    if (
        payload.get("fixture_only") is True
        and payload.get("execution_scope") == FIXTURE_SCOPE
        and payload.get("real_batch", False) is False
        and payload.get("eligible_for_real_authorization") is False
        and payload.get("fixture_counters_contribute_to_real_batch") is False
    ):
        return "EXPLICIT_FIXTURE_ONLY"
    fixture_lifecycle_marker = bool(
        normalized_payload.get("fixture_only") is True
        and normalized_payload.get("fixture_only_do_not_cite") is True
        and normalized_payload.get("not_real_fmb1") is True
        and normalized_payload.get("not_formal_measurement") is True
        and normalized_payload.get("actual_registration_execution") is False
    )
    if fixture_lifecycle_marker:
        return "EXPLICIT_FIXTURE_ONLY"
    name = path.name.lower()
    schema_marker = bool(
        payload.get("schema_only") is True
        or payload.get("artifact_scope") == "SCHEMA_ONLY"
        or payload.get("execution_scope") == "SCHEMA_ONLY"
        or ("schema" in Path(name).stem and "$schema" in payload)
    )
    if schema_marker and payload.get("real_batch", False) is False:
        return "SCHEMA_ONLY"
    template_marker = bool(
        payload.get("template_unissued") is True
        or payload.get("artifact_state") == "UNISSUED_TEMPLATE"
        or payload.get("execution_scope") == "UNISSUED_TEMPLATE"
        or (
            normalized_payload.get("template_only") is True
            and normalized_payload.get("lock_status") == "UNISSUED"
            and normalized_payload.get("formal_lock_issued") is False
        )
    )
    explicitly_unissued = bool(
        normalized_payload.get("issued", False) is False
        and normalized_payload.get("eligible_for_real_authorization", False)
        is False
        and normalized_payload.get("real_batch", False) is False
    )
    if template_marker and explicitly_unissued:
        return "UNISSUED_TEMPLATE"
    return None


def scan_registration_artifacts(
    roots: Iterable[str | Path],
) -> dict[str, Any]:
    """Scan real result/runtime roots for backend outputs or issued authority."""

    findings: list[dict[str, Any]] = []
    scanned_files: list[str] = []
    excluded_nonreal: list[dict[str, str]] = []
    resolved_roots: list[Path] = []
    for supplied in roots:
        candidate = Path(supplied)
        if candidate.is_symlink():
            findings.append(
                {"path": str(candidate), "kind": "SYMLINK_REAL_ARTIFACT_ROOT"}
            )
            continue
        try:
            root = candidate.resolve(strict=True)
        except FileNotFoundError:
            findings.append({"path": str(candidate), "kind": "MISSING_ARTIFACT_ROOT"})
            continue
        if not root.is_dir():
            findings.append({"path": str(root), "kind": "ARTIFACT_ROOT_NOT_DIRECTORY"})
            continue
        resolved_roots.append(root)
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = f"{root.name}/{path.relative_to(root).as_posix()}"
            scanned_files.append(relative)
            if path.is_symlink():
                findings.append({"path": relative, "kind": "SYMLINK_REAL_ARTIFACT"})
                continue
            suffix = path.suffix.lower()
            if suffix == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    findings.append(
                        {
                            "path": relative,
                            "kind": "UNREADABLE_JSON_ARTIFACT",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                nonreal_scope = _explicit_nonreal_json_scope(payload, path=path)
                if nonreal_scope is not None:
                    # Exact real lock/matrix names are never exempt, even when
                    # a payload tries to label itself as a fixture/template.
                    if path.name.lower() not in _PROTECTED_ISSUANCE_FILENAMES:
                        excluded_nonreal.append(
                            {"path": relative, "scope": nonreal_scope}
                        )
                        continue
                _scan_artifact_value(
                    payload,
                    relative=relative,
                    json_path="$",
                    findings=findings,
                )
            elif suffix == ".csv":
                try:
                    with path.open("r", encoding="utf-8", newline="") as stream:
                        header = next(csv.reader(stream), [])
                except (OSError, UnicodeError, csv.Error) as exc:
                    findings.append(
                        {
                            "path": relative,
                            "kind": "UNREADABLE_CSV_ARTIFACT",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                for field in header:
                    if _is_forbidden_artifact_key(_normalized(field)):
                        findings.append(
                            {
                                "path": relative,
                                "kind": "BACKEND_RESULT_FIELD",
                                "csv_field": field,
                            }
                        )
            if _is_result_artifact_name(path.name):
                kind = (
                    "AUTHORIZATION_MATRIX_OR_LOCK_ARTIFACT"
                    if path.name.lower() in _PROTECTED_ISSUANCE_FILENAMES
                    else "BACKEND_RESULT_ARTIFACT"
                )
                findings.append({"path": relative, "kind": kind})
    unique = {
        json.dumps(row, sort_keys=True, ensure_ascii=True): row for row in findings
    }
    ordered = [unique[key] for key in sorted(unique)]
    return {
        "schema": "mid360_fmb1_real_registration_artifact_scan_v1",
        "status": "PASS" if not ordered else "FAIL",
        "pass": not ordered,
        "execution_scope": REAL_SCOPE,
        "fixture_only_artifacts_accepted": False,
        "roots": [str(path) for path in resolved_roots],
        "scanned_file_count": len(scanned_files),
        "excluded_nonreal_artifact_count": len(excluded_nonreal),
        "excluded_nonreal_artifacts": excluded_nonreal,
        "real_registration_artifact_count": len(ordered),
        "artifacts": ordered,
    }


def snapshot_backend_processes(
    proc_root: str | Path = "/proc",
) -> list[dict[str, Any]]:
    """Return obvious live backend processes from a Linux ``/proc`` snapshot."""

    root = Path(proc_root)
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        if not directory.name.isdigit() or not directory.is_dir():
            continue
        try:
            command = [
                os.fsdecode(part)
                for part in (directory / "cmdline").read_bytes().split(b"\0")
                if part
            ]
            kind = classify_backend_command(command)
            if kind is None:
                continue
            comm_path = directory / "comm"
            comm = (
                comm_path.read_text(encoding="utf-8").strip()
                if comm_path.is_file()
                else ""
            )
            rows.append(
                {
                    "pid": int(directory.name),
                    "comm": comm,
                    "executable": Path(command[0]).name if command else "",
                    "backend_kind": kind,
                }
            )
        except (OSError, UnicodeError, ValueError):
            continue
    return rows


class RealPreBackendGuard:
    """Block subprocess, import, and loaded Open3D backend entry points.

    Blocked attempts have separate forensic counters.  Because the original
    callable/process is never reached, all real execution counters remain zero.
    """

    _SUBPROCESS_NAMES = ("Popen", "run", "call", "check_call", "check_output")

    def __init__(
        self,
        *,
        proc_root: str | Path = "/proc",
        open3d_module: ModuleType | None = None,
    ) -> None:
        self.proc_root = Path(proc_root)
        self.open3d_module = open3d_module
        self._original_subprocess: dict[str, Callable[..., Any]] = {}
        self._original_os_system: Callable[..., Any] | None = None
        self._original_import: Callable[..., Any] | None = None
        self._original_open3d: list[tuple[Any, str, Any]] = []
        self.active = False
        self.ever_activated = False
        self.blocked_open3d_attempt_count = 0
        self.blocked_pcl_attempt_count = 0
        self.blocked_other_backend_attempt_count = 0
        self.blocked_fixture_attempt_count = 0
        self.processes_at_entry: list[dict[str, Any]] = []
        self.processes_at_exit: list[dict[str, Any]] = []
        self.environment_at_entry: dict[str, str | None] = {}

    def _record_block(self, kind: str) -> None:
        if kind == "open3d":
            self.blocked_open3d_attempt_count += 1
        elif kind == "pcl":
            self.blocked_pcl_attempt_count += 1
        elif kind == "fixture":
            self.blocked_fixture_attempt_count += 1
        else:
            self.blocked_other_backend_attempt_count += 1

    def _guard_process(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(*args: Any, **kwargs: Any) -> Any:
            command = args[0] if args else kwargs.get("args")
            kind = classify_backend_command(command)
            if kind is not None:
                self._record_block(kind)
                raise BackendExecutionBlocked(
                    f"backend process blocked in {REAL_SCOPE}: {command!r}"
                )
            return original(*args, **kwargs)

        return guarded

    def _guard_os_system(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(command: Any) -> Any:
            kind = classify_backend_command(command)
            if kind is not None:
                self._record_block(kind)
                raise BackendExecutionBlocked(
                    f"backend shell command blocked in {REAL_SCOPE}: {command!r}"
                )
            return original(command)

        return guarded

    def _guard_import(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
            root = str(name).split(".", 1)[0].lower()
            if root in _BACKEND_IMPORT_ROOTS:
                self._record_block("pcl" if root in {"pcl", "pclpy"} else "open3d")
                raise BackendExecutionBlocked(
                    f"backend import blocked in {REAL_SCOPE}: {name}"
                )
            return original(name, *args, **kwargs)

        return guarded

    def _guard_loaded_open3d(self) -> None:
        module = self.open3d_module or sys.modules.get("open3d")
        if module is None:
            return
        containers = [
            getattr(getattr(module, "pipelines", None), "registration", None),
            getattr(module, "registration", None),
        ]
        for container in containers:
            if container is None:
                continue
            for name in dir(container):
                if not name.startswith(_OPEN3D_CALL_PREFIXES):
                    continue
                original = getattr(container, name)
                if not callable(original):
                    continue

                def denied(*args: Any, _name: str = name, **kwargs: Any) -> Any:
                    del args, kwargs
                    self._record_block("open3d")
                    raise BackendExecutionBlocked(
                        f"Open3D backend call blocked in {REAL_SCOPE}: {_name}"
                    )

                self._original_open3d.append((container, name, original))
                setattr(container, name, denied)
        self.open3d_module = module

    def _restore(self) -> None:
        for container, name, original in reversed(self._original_open3d):
            setattr(container, name, original)
        self._original_open3d.clear()
        for name, original in self._original_subprocess.items():
            setattr(subprocess, name, original)
        self._original_subprocess.clear()
        if self._original_os_system is not None:
            os.system = self._original_os_system
            self._original_os_system = None
        if self._original_import is not None:
            builtins.__import__ = self._original_import
            self._original_import = None

    def __enter__(self) -> "RealPreBackendGuard":
        if self.active:
            raise RegistrationFirewallError("real pre-backend guard is already active")
        observed_names = (
            _PRIMARY_NO_REGISTRATION_ENV,
            _LEGACY_NO_REGISTRATION_ENV,
            _EXECUTION_SCOPE_ENV,
            *_DENIED_TRUE_ENV,
            *_FIXTURE_ENV,
        )
        self.environment_at_entry = {name: os.environ.get(name) for name in observed_names}
        if not _truthy(os.environ.get(_PRIMARY_NO_REGISTRATION_ENV)):
            raise RegistrationFirewallError(
                f"{_PRIMARY_NO_REGISTRATION_ENV}=1 is required for real preflight"
            )
        if not _truthy(os.environ.get(_LEGACY_NO_REGISTRATION_ENV)):
            raise RegistrationFirewallError(
                f"{_LEGACY_NO_REGISTRATION_ENV}=true is also required"
            )
        if os.environ.get(_EXECUTION_SCOPE_ENV) != REAL_SCOPE:
            raise RegistrationFirewallError(
                f"{_EXECUTION_SCOPE_ENV}={REAL_SCOPE} is required"
            )
        active_authority = [name for name in _DENIED_TRUE_ENV if _truthy(os.environ.get(name))]
        if active_authority:
            raise RegistrationFirewallError(
                f"authorization environment is forbidden before preflight: {active_authority}"
            )
        active_fixture = [name for name in _FIXTURE_ENV if _truthy(os.environ.get(name))]
        if active_fixture:
            raise RegistrationFirewallError(
                f"fixture environment cannot enter real preflight: {active_fixture}"
            )
        self.processes_at_entry = snapshot_backend_processes(self.proc_root)
        if self.processes_at_entry:
            raise RegistrationFirewallError(
                f"backend process already active before preflight: {self.processes_at_entry}"
            )
        try:
            for name in self._SUBPROCESS_NAMES:
                original = getattr(subprocess, name)
                self._original_subprocess[name] = original
                setattr(subprocess, name, self._guard_process(original))
            self._original_os_system = os.system
            os.system = self._guard_os_system(os.system)
            self._original_import = builtins.__import__
            builtins.__import__ = self._guard_import(builtins.__import__)
            self._guard_loaded_open3d()
        except Exception:
            self._restore()
            raise
        self.active = True
        self.ever_activated = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.processes_at_exit = snapshot_backend_processes(self.proc_root)
        self._restore()
        self.active = False

    def report(self) -> dict[str, Any]:
        blocked_total = (
            self.blocked_open3d_attempt_count
            + self.blocked_pcl_attempt_count
            + self.blocked_other_backend_attempt_count
            + self.blocked_fixture_attempt_count
        )
        real_processes = [
            *self.processes_at_entry,
            *self.processes_at_exit,
        ]
        passed = bool(self.ever_activated and blocked_total == 0 and not real_processes)
        return {
            "schema": "mid360_fmb1_real_prebackend_runtime_firewall_v1",
            "status": "PASS" if passed else "FAIL",
            "pass": passed,
            "execution_scope": REAL_SCOPE,
            "fixture_only": False,
            "fixture_execution_allowed": False,
            "guard_was_activated": self.ever_activated,
            "guard_active_at_report": self.active,
            "environment_at_entry": dict(self.environment_at_entry),
            "blocked_open3d_attempt_count": self.blocked_open3d_attempt_count,
            "blocked_pcl_attempt_count": self.blocked_pcl_attempt_count,
            "blocked_other_backend_attempt_count": self.blocked_other_backend_attempt_count,
            "blocked_fixture_attempt_count": self.blocked_fixture_attempt_count,
            "blocked_backend_attempt_count": blocked_total,
            "real_open3d_registration_call_count": 0,
            "real_pcl_cli_invocation_count": 0,
            "real_other_registration_process_count": len(real_processes),
            "real_formal_trial_count": 0,
            "open3d_registration_call_count": 0,
            "pcl_cli_invocation_count": 0,
            "other_registration_process_count": len(real_processes),
            "formal_trial_count": 0,
            "registration_processes_at_entry": list(self.processes_at_entry),
            "registration_processes_at_exit": list(self.processes_at_exit),
        }


def _load_json_file(path: Path, label: str, blockers: list[dict[str, Any]]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        blockers.append({"code": "MISSING_OR_UNSAFE_REQUIRED_FILE", "file": label})
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "INVALID_REQUIRED_JSON",
                "file": label,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        return {}
    if not isinstance(payload, dict):
        blockers.append({"code": "REQUIRED_JSON_NOT_OBJECT", "file": label})
        return {}
    return payload


def _append_once(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    encoded = json.dumps(row, sort_keys=True, ensure_ascii=True)
    if all(json.dumps(value, sort_keys=True, ensure_ascii=True) != encoded for value in rows):
        rows.append(row)


def _verify_sha256sums(
    results: Path,
    blockers: list[dict[str, Any]],
) -> dict[str, str]:
    path = results / "SHA256SUMS"
    if path.is_symlink() or not path.is_file():
        blockers.append({"code": "MISSING_SHA256SUMS"})
        return {}
    rows: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        blockers.append({"code": "UNREADABLE_SHA256SUMS", "detail": str(exc)})
        return {}
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if match is None or match.group(2) in rows:
            blockers.append(
                {"code": "MALFORMED_SHA256SUMS", "line": line_number}
            )
            continue
        digest, name = match.groups()
        target = results / name
        if target.is_symlink() or not target.is_file():
            blockers.append({"code": "SHA256SUM_TARGET_MISSING", "file": name})
            continue
        actual = _sha256_file(target)
        if actual != digest:
            blockers.append(
                {
                    "code": "SHA256SUM_MISMATCH",
                    "file": name,
                    "expected": digest,
                    "actual": actual,
                }
            )
        rows[name] = digest
    required = {
        "NO_ICP_ATTESTATION.json",
        "fmb1_canonical_input_manifest.csv",
        "fmb1_deep_verification_report.json",
        "fmb1_frozen_manifest.json",
        "fmb1_frozen_manifest.sha256",
        "fmb1_geometry_admission_manifest.json",
        "fmb1_pre_registration_readiness.json",
    }
    for name in sorted(required - set(rows)):
        blockers.append({"code": "CRITICAL_FILE_NOT_HASH_LISTED", "file": name})
    return rows


def _validate_preacquisition_bindings(
    repository: Path,
    results: Path,
    blockers: list[dict[str, Any]],
) -> None:
    preacquisition = _load_json_file(
        results / "preacquisition_manifest.json",
        "preacquisition_manifest.json",
        blockers,
    )
    bindings = preacquisition.get("files")
    if not isinstance(bindings, Mapping):
        blockers.append({"code": "PREACQUISITION_BINDINGS_MISSING"})
        return
    for relative, expected in bindings.items():
        candidate = repository / str(relative)
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            blockers.append(
                {"code": "PREACQUISITION_BINDING_INVALID", "file": str(relative)}
            )
            continue
        actual = _sha256_file(candidate)
        if actual != expected:
            blockers.append(
                {
                    "code": "PREACQUISITION_SOURCE_DRIFT",
                    "file": str(relative),
                    "expected": expected,
                    "actual": actual,
                }
            )
    contract = repository / "frozen_assets/backend_parameter_contract.json"
    if contract.is_symlink() or not contract.is_file():
        blockers.append({"code": "BACKEND_PARAMETER_CONTRACT_MISSING"})
    else:
        actual_contract = _sha256_file(contract)
        declared = preacquisition.get("backend_parameter_contract_sha256")
        if (
            actual_contract != EXPECTED_BACKEND_CONTRACT_SHA256
            or declared != EXPECTED_BACKEND_CONTRACT_SHA256
        ):
            blockers.append(
                {
                    "code": "BACKEND_PARAMETER_CONTRACT_DRIFT",
                    "expected": EXPECTED_BACKEND_CONTRACT_SHA256,
                    "actual": actual_contract,
                    "declared": declared,
                }
            )


def _real_zero_counter_payload() -> dict[str, int]:
    return {
        "real_open3d_registration_call_count": 0,
        "real_pcl_cli_invocation_count": 0,
        "real_other_registration_process_count": 0,
        "real_formal_trial_count": 0,
        "open3d_registration_call_count": 0,
        "pcl_cli_invocation_count": 0,
        "other_registration_process_count": 0,
        "formal_trial_count": 0,
        "actual_registration_trials": 0,
        "real_T_est_file_count": 0,
    }


def _current_block_reason(report: Mapping[str, Any]) -> str | None:
    if report.get("pass") is True:
        return None
    for group in ("integrity_blockers", "security_blockers", "scientific_blockers"):
        rows = report.get(group)
        if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
            return str(rows[0].get("code", "UNKNOWN_BLOCKER"))
    return "FAIL_CLOSED_UNSPECIFIED_BLOCKER"


def _apply_nonissuance_aliases(report: dict[str, Any]) -> None:
    report["CURRENT_REAL_BATCH_BLOCKED"] = report.get("pass") is not True
    report["CURRENT_BLOCK_REASON"] = _current_block_reason(report)
    report["FORMAL_RUN_MATRIX_ISSUED"] = False
    report["FORMAL_LOCK_ISSUED"] = False
    report["TRIAL_MATRIX_ISSUED"] = False
    report["FORMAL_BATCH_LOCK_ISSUED"] = False
    report["FORMAL_REGISTRATION_AUTHORIZED"] = False
    report["FORMAL_ICP_UNLOCKED"] = False
    report["BACKEND_INVOCATION_ALLOWED"] = False
    report["backend_invoked"] = False


def evaluate_real_batch_preflight(
    repository: str | Path,
    *,
    results_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    qualification_runtime_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate the current real FMB1 state without issuing authority."""

    root = Path(repository).resolve(strict=True)
    expected_results = (root / REAL_RESULTS_RELATIVE).resolve()
    expected_runtime = (root / REAL_RUNTIME_RELATIVE).resolve()
    expected_qualification_runtime = (
        root / QUALIFICATION_RUNTIME_RELATIVE
    ).resolve()
    results = Path(results_dir).resolve() if results_dir is not None else expected_results
    runtime = Path(runtime_dir).resolve() if runtime_dir is not None else expected_runtime
    qualification_runtime = (
        Path(qualification_runtime_dir).resolve()
        if qualification_runtime_dir is not None
        else expected_qualification_runtime
    )
    integrity_blockers: list[dict[str, Any]] = []
    security_blockers: list[dict[str, Any]] = []
    scientific_blockers: list[dict[str, Any]] = []

    if results != expected_results:
        security_blockers.append(
            {
                "code": "NONCANONICAL_REAL_RESULTS_DIRECTORY",
                "expected": str(expected_results),
                "actual": str(results),
            }
        )
    if runtime != expected_runtime:
        security_blockers.append(
            {
                "code": "NONCANONICAL_REAL_RUNTIME_DIRECTORY",
                "expected": str(expected_runtime),
                "actual": str(runtime),
            }
        )
    if qualification_runtime != expected_qualification_runtime:
        security_blockers.append(
            {
                "code": "NONCANONICAL_QUALIFICATION_RUNTIME_DIRECTORY",
                "expected": str(expected_qualification_runtime),
                "actual": str(qualification_runtime),
            }
        )
    if any(_truthy(os.environ.get(name)) for name in _FIXTURE_ENV):
        security_blockers.append({"code": "FIXTURE_ENVIRONMENT_IN_REAL_PREFLIGHT"})

    try:
        static_report = assert_prebackend_static_safe(root)
    except (RegistrationFirewallError, OSError) as exc:
        static_report = {
            "schema": "mid360_fmb1_real_prebackend_static_firewall_v1",
            "status": "FAIL",
            "pass": False,
            "violations": [{"detail": f"{type(exc).__name__}: {exc}"}],
        }
        security_blockers.append({"code": "STATIC_PREFLIGHT_SCOPE_UNSAFE"})

    artifact_report = scan_registration_artifacts((results, runtime))
    if not artifact_report["pass"]:
        security_blockers.append(
            {
                "code": "REGISTRATION_OR_AUTHORITY_ARTIFACT_PRESENT",
                "artifact_count": artifact_report["real_registration_artifact_count"],
            }
        )
    qualification_runtime_artifact_report = scan_registration_artifacts(
        (qualification_runtime,)
    )
    if not qualification_runtime_artifact_report["pass"]:
        security_blockers.append(
            {
                "code": "QUALIFICATION_RUNTIME_REGISTRATION_ARTIFACT_PRESENT",
                "artifact_count": qualification_runtime_artifact_report[
                    "real_registration_artifact_count"
                ],
            }
        )

    sha_rows = _verify_sha256sums(results, integrity_blockers)
    _validate_preacquisition_bindings(root, results, integrity_blockers)

    manifest = _load_json_file(
        results / "fmb1_frozen_manifest.json",
        "fmb1_frozen_manifest.json",
        integrity_blockers,
    )
    manifest_sha_path = results / "fmb1_frozen_manifest.sha256"
    if manifest and manifest_sha_path.is_file() and not manifest_sha_path.is_symlink():
        actual_manifest_sha = _sha256_file(results / "fmb1_frozen_manifest.json")
        try:
            line = manifest_sha_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            line = ""
        expected_line = f"{actual_manifest_sha}  fmb1_frozen_manifest.json"
        if line != expected_line or sha_rows.get("fmb1_frozen_manifest.json") != actual_manifest_sha:
            integrity_blockers.append({"code": "FROZEN_MANIFEST_ANCHOR_MISMATCH"})

    readiness = _load_json_file(
        results / "fmb1_pre_registration_readiness.json",
        "fmb1_pre_registration_readiness.json",
        integrity_blockers,
    )
    geometry = _load_json_file(
        results / "fmb1_geometry_admission_manifest.json",
        "fmb1_geometry_admission_manifest.json",
        integrity_blockers,
    )
    no_icp = _load_json_file(
        results / "NO_ICP_ATTESTATION.json",
        "NO_ICP_ATTESTATION.json",
        integrity_blockers,
    )
    deep = _load_json_file(
        results / "fmb1_deep_verification_report.json",
        "fmb1_deep_verification_report.json",
        integrity_blockers,
    )
    independent = _load_json_file(
        results / "fmb1_verification_report.json",
        "fmb1_verification_report.json",
        integrity_blockers,
    )
    status_file = _load_json_file(
        results / "formal_icp_status.json",
        "formal_icp_status.json",
        integrity_blockers,
    )

    if manifest.get("readiness") != readiness:
        integrity_blockers.append({"code": "READINESS_NOT_BOUND_IN_FROZEN_MANIFEST"})
    if manifest.get("geometry_scenes") != geometry.get("scene_summaries"):
        integrity_blockers.append({"code": "GEOMETRY_NOT_BOUND_IN_FROZEN_MANIFEST"})
    if manifest.get("no_icp_attestation") != no_icp:
        integrity_blockers.append({"code": "NO_ICP_ATTESTATION_NOT_BOUND"})
    if manifest.get("deep_verification") != deep:
        integrity_blockers.append({"code": "DEEP_VERIFICATION_NOT_BOUND"})
    if manifest.get("backend_parameter_contract_sha256") != EXPECTED_BACKEND_CONTRACT_SHA256:
        integrity_blockers.append({"code": "MANIFEST_BACKEND_CONTRACT_BINDING_INVALID"})

    for payload_name, payload in (
        ("readiness", readiness),
        ("no_icp_attestation", no_icp),
        ("deep_verification", deep),
        ("independent_verification", independent),
        ("formal_icp_status", status_file),
    ):
        for key in (
            "FORMAL_REGISTRATION_AUTHORIZED",
            "FORMAL_ICP_UNLOCKED",
            "FORMAL_MEASUREMENT_RESULT",
            "MEASUREMENT_FINAL_RESULT",
        ):
            if payload.get(key, False) is not False:
                security_blockers.append(
                    {"code": "AUTHORIZATION_OR_LOCK_ACTIVE", "source": payload_name, "field": key}
                )
        for key in _ZERO_ONLY_ARTIFACT_KEYS:
            if key in payload and payload[key] != 0:
                security_blockers.append(
                    {"code": "REAL_EXECUTION_COUNTER_NONZERO", "source": payload_name, "field": key}
                )
    if no_icp.get("status") != "PASS" or no_icp.get("NO_ICP_ATTESTATION_PASS") is not True:
        integrity_blockers.append({"code": "NO_ICP_ATTESTATION_NOT_PASS"})
    if deep.get("status") != "PASS":
        integrity_blockers.append({"code": "DEEP_VERIFICATION_NOT_PASS"})

    scene_rows = geometry.get("scene_summaries")
    if not isinstance(scene_rows, list):
        scene_rows = []
        integrity_blockers.append({"code": "GEOMETRY_SCENE_ROWS_MISSING"})
    admitted_rich = sorted(
        str(row.get("scene_id"))
        for row in scene_rows
        if isinstance(row, Mapping)
        and row.get("final_geometry_class") == "RICH"
        and row.get("geometry_admission_status") == "GEOMETRY_ADMITTED"
        and row.get("semantic_candidate_label") == "RICH_CANDIDATE"
    )
    admitted_weak = sorted(
        str(row.get("scene_id"))
        for row in scene_rows
        if isinstance(row, Mapping)
        and row.get("final_geometry_class") == "WEAK"
        and row.get("geometry_admission_status") == "GEOMETRY_ADMITTED"
        and row.get("semantic_candidate_label") == "WEAK_CANDIDATE"
    )
    w02 = next(
        (
            row
            for row in scene_rows
            if isinstance(row, Mapping) and row.get("scene_id") == "FMB1_W02"
        ),
        None,
    )
    w04 = next(
        (
            row
            for row in scene_rows
            if isinstance(row, Mapping) and row.get("scene_id") == "FMB1_W04"
        ),
        None,
    )
    if len(admitted_rich) != 3:
        scientific_blockers.append(
            {
                "code": "ADMITTED_RICH_SCENE_COUNT_MISMATCH",
                "required": 3,
                "actual": len(admitted_rich),
                "scene_ids": admitted_rich,
            }
        )
    missing_w04 = not (
        isinstance(w04, Mapping)
        and w04.get("semantic_candidate_label") == "WEAK_CANDIDATE"
        and w04.get("final_geometry_class") == "WEAK"
        and w04.get("geometry_admission_status") == "GEOMETRY_ADMITTED"
    )
    expected_current_rejection = bool(
        isinstance(w02, Mapping)
        and w02.get("semantic_candidate_label") == "WEAK_CANDIDATE"
        and w02.get("final_geometry_class") == "RICH"
        and w02.get("geometry_admission_status") == "GEOMETRY_REJECTED"
        and w02.get("replacement_allowed_under_preregistration") is True
    )
    if len(admitted_weak) != 3:
        if len(admitted_weak) == 2 and expected_current_rejection and missing_w04:
            scientific_blockers.append(
                {
                    "code": "MISSING_ADMITTED_WEAK_REPLACEMENT_W04",
                    "required_weak_scene_count": 3,
                    "admitted_weak_scene_count": 2,
                    "admitted_weak_scene_ids": admitted_weak,
                    "rejected_candidate_scene_id": "FMB1_W02",
                    "required_replacement_scene_id": "FMB1_W04",
                }
            )
        else:
            scientific_blockers.append(
                {
                    "code": "ADMITTED_WEAK_SCENE_COUNT_MISMATCH",
                    "required": 3,
                    "actual": len(admitted_weak),
                    "scene_ids": admitted_weak,
                }
            )

    if readiness.get("rich_scene_count") != len(admitted_rich):
        integrity_blockers.append({"code": "READINESS_RICH_COUNT_MISMATCH"})
    if readiness.get("weak_scene_count") != len(admitted_weak):
        integrity_blockers.append({"code": "READINESS_WEAK_COUNT_MISMATCH"})
    current_missing_only = [row.get("code") for row in scientific_blockers] == [
        "MISSING_ADMITTED_WEAK_REPLACEMENT_W04"
    ]
    if current_missing_only:
        if readiness.get("FMB1_PRE_REGISTRATION_DATA_READY") is not False:
            integrity_blockers.append({"code": "READINESS_MUST_BE_FALSE_WHILE_W04_MISSING"})
        independent_error = independent.get("error")
        error_text = (
            str(independent_error.get("message", ""))
            if isinstance(independent_error, Mapping)
            else ""
        )
        if independent.get("status") != "FAIL" or "FMB1_W02" not in error_text:
            integrity_blockers.append(
                {"code": "INDEPENDENT_VERIFIER_FAILURE_NOT_EXPLAINED_BY_W02"}
            )
    elif independent.get("status") != "PASS":
        integrity_blockers.append({"code": "INDEPENDENT_VERIFIER_NOT_PASS"})

    all_blockers = [*integrity_blockers, *security_blockers, *scientific_blockers]
    passed = not all_blockers
    report = {
        "schema": "mid360_fmb1_formal_registration_preflight_v1",
        "status": "PASS" if passed else "BLOCKED",
        "pass": passed,
        "execution_scope": REAL_SCOPE,
        "real_batch": True,
        "fixture_only": False,
        "fixture_evidence_accepted": False,
        "fixture_counters_included": False,
        "repository": str(root),
        "results_dir": str(results),
        "runtime_dir": str(runtime),
        "qualification_runtime_dir": str(qualification_runtime),
        "scientific_blockers": scientific_blockers,
        "scientific_blocker_codes": [row["code"] for row in scientific_blockers],
        "integrity_blockers": integrity_blockers,
        "security_blockers": security_blockers,
        "admitted_rich_scene_count": len(admitted_rich),
        "admitted_rich_scene_ids": admitted_rich,
        "required_rich_scene_count": 3,
        "admitted_weak_scene_count": len(admitted_weak),
        "admitted_weak_scene_ids": admitted_weak,
        "required_weak_scene_count": 3,
        "required_replacement_scene_id": "FMB1_W04" if missing_w04 else None,
        "preregistration_ready": readiness.get("FMB1_PRE_REGISTRATION_DATA_READY") is True,
        "static_firewall": static_report,
        "artifact_scan": artifact_report,
        "qualification_runtime_artifact_scan": qualification_runtime_artifact_report,
        "TRIAL_MATRIX_ISSUED": False,
        "FORMAL_BATCH_LOCK_ISSUED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "BACKEND_INVOCATION_ALLOWED": False,
        "backend_invoked": False,
        **_real_zero_counter_payload(),
    }
    _apply_nonissuance_aliases(report)
    return report


def preflight_real_batch(
    repository: str | Path,
    *,
    results_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    qualification_runtime_dir: str | Path | None = None,
    proc_root: str | Path = "/proc",
    open3d_module: ModuleType | None = None,
) -> dict[str, Any]:
    """Run the complete real preflight under the runtime backend guard."""

    guard = RealPreBackendGuard(
        proc_root=proc_root,
        open3d_module=open3d_module,
    )
    try:
        with guard:
            report = evaluate_real_batch_preflight(
                repository,
                results_dir=results_dir,
                runtime_dir=runtime_dir,
                qualification_runtime_dir=qualification_runtime_dir,
            )
        runtime_report = guard.report()
    except RegistrationFirewallError as exc:
        report = {
            "schema": "mid360_fmb1_formal_registration_preflight_v1",
            "status": "BLOCKED",
            "pass": False,
            "execution_scope": REAL_SCOPE,
            "real_batch": True,
            "fixture_only": False,
            "scientific_blockers": [],
            "scientific_blocker_codes": [],
            "integrity_blockers": [],
            "security_blockers": [
                {"code": "RUNTIME_FIREWALL_PRECONDITION_FAILED", "detail": str(exc)}
            ],
            "TRIAL_MATRIX_ISSUED": False,
            "FORMAL_BATCH_LOCK_ISSUED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "FORMAL_ICP_UNLOCKED": False,
            "BACKEND_INVOCATION_ALLOWED": False,
            "backend_invoked": False,
            **_real_zero_counter_payload(),
        }
        runtime_report = guard.report()
    report["runtime_firewall"] = runtime_report
    if not runtime_report.get("pass"):
        report["status"] = "BLOCKED"
        report["pass"] = False
        _append_once(
            report["security_blockers"],
            {"code": "RUNTIME_FIREWALL_NOT_PASS"},
        )
    for field in (
        "real_open3d_registration_call_count",
        "real_pcl_cli_invocation_count",
        "real_other_registration_process_count",
        "real_formal_trial_count",
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "formal_trial_count",
    ):
        if field in runtime_report:
            report[field] = runtime_report[field]
    # This function is evidence-only even when every prerequisite eventually
    # passes; authorization and issuance belong to a separate explicit task.
    _apply_nonissuance_aliases(report)
    return report


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise RegistrationFirewallError(f"refusing symlink output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_preflight_reports(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write only the current-real-batch JSON/Markdown qualification report."""

    directory = Path(output_dir)
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise RegistrationFirewallError(f"unsafe preflight output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "current_real_batch_preflight_report.json"
    markdown_path = directory / "current_real_batch_preflight_report.md"
    encoded = (
        json.dumps(dict(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    blockers = report.get("scientific_blocker_codes", [])
    markdown = "\n".join(
        [
            "# FMB1 current real batch preflight",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Current batch blocked: `{str(report.get('CURRENT_REAL_BATCH_BLOCKED')).lower()}`",
            f"- Current block reason: `{report.get('CURRENT_BLOCK_REASON')}`",
            f"- Admitted Rich scenes: `{report.get('admitted_rich_scene_count')}/3`",
            f"- Admitted Weak scenes: `{report.get('admitted_weak_scene_count')}/3`",
            f"- Scientific blockers: `{', '.join(map(str, blockers)) or 'NONE'}`",
            "- Formal run matrix issued: `false`",
            "- Formal lock issued: `false`",
            "- Formal registration authorized: `false`",
            "- Backend invoked: `false`",
            "",
        ]
    ).encode("utf-8")
    _atomic_write(json_path, encoded)
    _atomic_write(markdown_path, markdown)
    return {
        "json_path": str(json_path.resolve(strict=True)),
        "json_sha256": _sha256_file(json_path),
        "markdown_path": str(markdown_path.resolve(strict=True)),
        "markdown_sha256": _sha256_file(markdown_path),
    }


def write_no_icp_attestation_tonight(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Emit a zero-execution attestation, never an authorization artifact.

    A scientific preflight block (the current missing W04 replacement) is
    compatible with this attestation.  Any integrity/security/firewall failure
    is not.
    """

    runtime = report.get("runtime_firewall")
    artifact = report.get("artifact_scan")
    qualification_artifact = report.get("qualification_runtime_artifact_scan")
    if not isinstance(runtime, Mapping) or runtime.get("pass") is not True:
        raise RegistrationFirewallError("cannot attest: runtime firewall is not PASS")
    if not isinstance(artifact, Mapping) or artifact.get("pass") is not True:
        raise RegistrationFirewallError("cannot attest: artifact scan is not PASS")
    if (
        not isinstance(qualification_artifact, Mapping)
        or qualification_artifact.get("pass") is not True
        or qualification_artifact.get("real_registration_artifact_count") != 0
    ):
        raise RegistrationFirewallError(
            "cannot attest: qualification runtime artifact scan is not clean"
        )
    if report.get("integrity_blockers") or report.get("security_blockers"):
        raise RegistrationFirewallError(
            "cannot attest: integrity/security blocker is present"
        )
    for key in _real_zero_counter_payload():
        if report.get(key) != 0:
            raise RegistrationFirewallError(f"cannot attest: {key} is nonzero")
    if any(
        report.get(key) is not False
        for key in (
            "FORMAL_REGISTRATION_AUTHORIZED",
            "FORMAL_ICP_UNLOCKED",
            "FORMAL_RUN_MATRIX_ISSUED",
            "FORMAL_LOCK_ISSUED",
            "backend_invoked",
        )
    ):
        raise RegistrationFirewallError("cannot attest: authority/issuance flag is active")
    report_bytes = json.dumps(
        dict(report), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    written = report.get("written_reports")
    written_report_sha = (
        written.get("json_sha256") if isinstance(written, Mapping) else None
    )
    if not isinstance(written_report_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", written_report_sha
    ):
        written_report_sha = hashlib.sha256(report_bytes).hexdigest()
    qualification_scan_bytes = json.dumps(
        dict(qualification_artifact),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    qualification_fixture_exclusions = sum(
        1
        for row in qualification_artifact.get("excluded_nonreal_artifacts", [])
        if isinstance(row, Mapping) and row.get("scope") == "EXPLICIT_FIXTURE_ONLY"
    )
    attestation = {
        "schema": "mid360_fmb1_no_icp_attestation_tonight_v1",
        "status": "PASS",
        "pass": True,
        "NO_ICP_ATTESTATION_TONIGHT_PASS": True,
        "execution_scope": REAL_SCOPE,
        "real_batch": True,
        "fixture_only": False,
        "fixture_counters_included": False,
        "current_real_batch_preflight_sha256": written_report_sha,
        "CURRENT_REAL_BATCH_BLOCKED": report.get("CURRENT_REAL_BATCH_BLOCKED"),
        "CURRENT_BLOCK_REASON": report.get("CURRENT_BLOCK_REASON"),
        "scientific_blocker_codes": list(report.get("scientific_blocker_codes", [])),
        "excluded_schema_template_fixture_artifact_count": artifact.get(
            "excluded_nonreal_artifact_count", 0
        ),
        "qualification_runtime_artifact_scan_sha256": hashlib.sha256(
            qualification_scan_bytes
        ).hexdigest(),
        "qualification_runtime_artifact_scan_status": "PASS",
        "qualification_runtime_scanned_file_count": qualification_artifact.get(
            "scanned_file_count", 0
        ),
        "qualification_runtime_excluded_nonreal_artifact_count": (
            qualification_artifact.get("excluded_nonreal_artifact_count", 0)
        ),
        "qualification_runtime_fixture_exclusion_count": (
            qualification_fixture_exclusions
        ),
        "qualification_runtime_real_registration_artifact_count": 0,
        "FORMAL_RUN_MATRIX_ISSUED": False,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "BACKEND_INVOCATION_ALLOWED": False,
        "backend_invoked": False,
        **_real_zero_counter_payload(),
    }
    directory = Path(output_dir)
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise RegistrationFirewallError(f"unsafe attestation output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "NO_ICP_ATTESTATION_TONIGHT.json"
    encoded = (
        json.dumps(attestation, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write(path, encoded)
    return {
        **attestation,
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256_file(path),
    }


__all__ = [
    "BackendExecutionBlocked",
    "EXPECTED_BACKEND_CONTRACT_SHA256",
    "FIXTURE_SCOPE",
    "QUALIFICATION_RUNTIME_RELATIVE",
    "REAL_SCOPE",
    "RealPreBackendGuard",
    "RegistrationFirewallError",
    "assert_fixture_only_isolated",
    "assert_prebackend_static_safe",
    "classify_backend_command",
    "evaluate_real_batch_preflight",
    "preflight_real_batch",
    "scan_registration_artifacts",
    "snapshot_backend_processes",
    "write_no_icp_attestation_tonight",
    "write_preflight_reports",
]
