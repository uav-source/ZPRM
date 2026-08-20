"""Fail-closed guards for registration-free FMB1 data preparation.

This module is intentionally independent from the FMB1 authorization lock in
``protocol.py``.  It protects only ingestion, acquisition audit, target-map
construction, snapshot freezing, and geometry-only analysis.  It does not
authorize a formal backend run.
"""

from __future__ import annotations

import ast
import hashlib
import io
import os
import re
import subprocess
import sys
import tokenize
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence


NO_FORMAL_REGISTRATION_ENV = "NO_FORMAL_REGISTRATION"
EXPECTED_BACKEND_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)

# ``protocol.py`` contains the already-frozen future authorization mechanism,
# including explanatory registration text.  It is authenticated elsewhere and
# is deliberately not part of the preparation execution scope.  The legacy
# ``freeze_batch.py`` is excluded for the same reason: this preparation task
# must neither import nor execute it.
_EXCLUDED_SCOPE_PATHS = frozenset(
    {
        "experiments/mid360_formal_batch1/__init__.py",
        "experiments/mid360_formal_batch1/preregistration_firewall.py",
        # Versioned future-execution/result-schema authorities are outside the
        # registration-free preparation scope.  They are individually AST- and
        # hash-audited by zero_perturbation_v1_1_r1_verify.py; no directory-wide
        # exclusion is permitted.
        "experiments/mid360_formal_batch1/zero_perturbation_r1_trial_assets.py",
        "experiments/mid360_formal_batch1/zero_perturbation_v1_1_r1_runner.py",
        "tools/mid360_formal_batch1/freeze_batch.py",
    }
)

_FORBIDDEN_IMPORT_PATTERNS = (
    re.compile(r"(^|\.)open3d($|\.)"),
    re.compile(r"(^|\.)(?:pcl|pclpy)($|\.)"),
    re.compile(r"(^|\.)(?:open3d_backend|pcl_backend|debug_registration)($|\.)"),
    re.compile(r"(^|\.)(?:backend_execution|phase_b_backend_execution)($|\.)"),
    re.compile(r"(^|\.)mid360_controlled_perturbation($|\.)"),
    re.compile(r"(^|\.)mid360_capture_basin($|\.)"),
    re.compile(r"(^|\.)mid360_two_scene_pilot\.registration($|\.)"),
)

_FORBIDDEN_PROTOCOL_SYMBOLS = frozenset(
    {
        "freeze_batch",
        "formal_icp_authorized",
        "require_formal_icp_authorization",
    }
)

_FORBIDDEN_EXECUTABLE_IDENTIFIERS = frozenset(
    {
        "registration_icp",
        "registration_generalized_icp",
        "generalized_icp",
        "gicp",
        "ndt",
        "scan_matching",
        "scan_match",
        "run_open3d",
        "run_open3d_full",
        "run_pcl",
        "run_pcl_point_to_plane",
        "analyze_estimated_transform",
        "safe_analyze_estimated_transform",
        "freeze_batch",
        "formal_icp_authorized",
        "require_formal_icp_authorization",
    }
)

_FORBIDDEN_RESULT_IDENTIFIERS = frozenset(
    {
        "t_est",
        "t_estimated",
        "estimated_transform",
        "estimated_pose",
        "final_residual",
        "final_residual_rmse",
        "correspondence_turnover",
        "accepted_source_turnover",
        "fitness",
        "fitness_score",
        "solver_result",
        "registration_result",
        "registration_output",
    }
)

_FORBIDDEN_DYNAMIC_CALLS = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "compile",
        "import_module",
    }
)

_LEXICAL_EXECUTABLE_PATTERNS = tuple(
    re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", re.IGNORECASE)
    for name in sorted(_FORBIDDEN_EXECUTABLE_IDENTIFIERS | _FORBIDDEN_RESULT_IDENTIFIERS)
)

_REGISTRATION_PROCESS_TOKENS = frozenset(
    {
        "pcl_point_to_plane_cli",
        "registration_icp",
        "registration_generalized_icp",
        "generalized_icp",
        "open3d_backend",
        "pcl_backend",
        "kiss_icp",
        "kiss-icp",
        "fast_lio",
        "fast-lio",
        "lio_sam",
        "lio-sam",
        "kiss_icp",
        "gicp",
        "ndt",
        "icp",
        "scan_matching",
        "scan-matching",
    }
)


class PreregistrationFirewallError(RuntimeError):
    """A static preparation-scope rule or attestation precondition failed."""


class RegistrationForbiddenError(PermissionError):
    """A process or callable attempted registration while the guard was active."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _normalized_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _forbidden_import(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    return any(pattern.search(normalized) is not None for pattern in _FORBIDDEN_IMPORT_PATTERNS)


def _forbidden_identifier(value: str) -> bool:
    normalized = _normalized_identifier(value)
    return (
        normalized in _FORBIDDEN_EXECUTABLE_IDENTIFIERS
        or normalized in _FORBIDDEN_RESULT_IDENTIFIERS
    )


def _string_literal(value: ast.AST) -> str | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _executable_source_without_text(source: str) -> str:
    """Return tokenized Python with comments and string literals blanked out."""

    output: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type in {tokenize.COMMENT, tokenize.STRING}:
                output.append(" " * len(token.string))
            else:
                output.append(token.string)
    except (IndentationError, tokenize.TokenError) as exc:
        raise PreregistrationFirewallError(f"cannot tokenize preparation source: {exc}") from exc
    return " ".join(output)


def _process_literal_text(node: ast.Call) -> str | None:
    callee = _dotted_name(node.func).lower()
    process_calls = {
        "subprocess.popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "os.system",
    }
    if callee not in process_calls or not node.args:
        return None
    try:
        literal = ast.literal_eval(node.args[0])
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    if isinstance(literal, (list, tuple)):
        return " ".join(os.fspath(item) for item in literal)
    if isinstance(literal, (str, bytes, os.PathLike)):
        return os.fsdecode(os.fspath(literal))
    return None


def _classify_registration_command(command: Any) -> str | None:
    if isinstance(command, (list, tuple)):
        parts = [os.fsdecode(os.fspath(item)) for item in command]
    else:
        parts = [os.fsdecode(os.fspath(command))]
    normalized_parts: list[str] = []
    for part in parts:
        # Shell strings and script paths are both inspected, while boundary
        # matching avoids false positives such as ``preregistration``.
        for candidate in re.split(r"[\s/\\]+", part.lower()):
            normalized = candidate.replace("-", "_")
            normalized = re.sub(r"\.(?:py|sh|bin|exe)$", "", normalized)
            if normalized:
                normalized_parts.append(normalized)
    matches: set[str] = set()
    for candidate in normalized_parts:
        for token in _REGISTRATION_PROCESS_TOKENS:
            normalized_token = token.replace("-", "_")
            # Short algorithm names must be executable-like, not merely part
            # of evidence names such as ``NO_ICP_ATTESTATION.json``.
            if normalized_token in {"icp", "gicp", "ndt"}:
                matched = candidate in {
                    normalized_token,
                    f"{normalized_token}_cli",
                    f"run_{normalized_token}",
                    f"run_{normalized_token}_cli",
                } or candidate.startswith(f"run_{normalized_token}_")
            else:
                matched = candidate == normalized_token or re.search(
                    rf"(?<![a-z0-9]){re.escape(normalized_token)}(?![a-z0-9])",
                    candidate,
                ) is not None
            if matched:
                matches.add(normalized_token)
    if not matches:
        return None
    if "pcl_point_to_plane_cli" in matches or "pcl_backend" in matches:
        return "pcl"
    if any(value.startswith("open3d") or value.startswith("registration_") for value in matches):
        return "open3d"
    return "other"


def _discover_static_scope(repository: Path) -> tuple[list[Path], list[str]]:
    candidates: set[Path] = set()
    for directory in (
        repository / "experiments/mid360_formal_batch1",
        repository / "tools/mid360_formal_batch1",
    ):
        if directory.is_dir():
            candidates.update(path for path in directory.rglob("*.py") if path.is_file())
    tools = repository / "tools"
    if tools.is_dir():
        candidates.update(
            path
            for path in tools.glob("*mid360*formal*batch1*.py")
            if path.is_file()
        )
    included: list[Path] = []
    excluded: list[str] = []
    for candidate in sorted(candidates):
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(repository).as_posix()
        except ValueError as exc:
            raise PreregistrationFirewallError(
                f"preparation source escapes repository: {candidate}"
            ) from exc
        if candidate.is_symlink():
            raise PreregistrationFirewallError(
                f"symbolic-link preparation source is forbidden: {candidate}"
            )
        if relative in _EXCLUDED_SCOPE_PATHS:
            excluded.append(relative)
        else:
            included.append(resolved)
    return included, excluded


def _scan_source(path: Path, repository: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    relative = path.relative_to(repository).as_posix()
    authority_protocol = relative == "experiments/mid360_formal_batch1/protocol.py"
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [
            {
                "path": relative,
                "line": exc.lineno,
                "kind": "SYNTAX_ERROR",
                "detail": str(exc),
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
                if _forbidden_import(alias.name):
                    add(node, "FORBIDDEN_IMPORT", alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _forbidden_import(module):
                add(node, "FORBIDDEN_IMPORT", module)
            protocol_import = module.endswith("mid360_formal_batch1.protocol") or (
                node.level > 0 and module == "protocol"
            )
            if protocol_import:
                for alias in node.names:
                    if alias.name in _FORBIDDEN_PROTOCOL_SYMBOLS or alias.name == "*":
                        add(node, "FORBIDDEN_PROTOCOL_SYMBOL", alias.name)
            for alias in node.names:
                qualified = f"{module}.{alias.name}" if module else alias.name
                if _forbidden_import(qualified):
                    add(node, "FORBIDDEN_IMPORT", qualified)
        elif isinstance(node, ast.Call):
            callee = _dotted_name(node.func)
            tail = callee.rsplit(".", 1)[-1]
            if _forbidden_identifier(tail) and not (
                authority_protocol and tail in _FORBIDDEN_PROTOCOL_SYMBOLS
            ):
                add(node, "FORBIDDEN_CALL", callee)
            # Only the builtin ``compile`` is dynamic execution; ordinary
            # helpers such as ``re.compile`` are safe and common in ingest
            # filename/schema validation.
            dynamic_call = (
                callee in {"__import__", "eval", "exec", "compile"}
                or callee == "importlib.import_module"
                or (callee == "import_module" and tail == "import_module")
            )
            if dynamic_call:
                add(node, "DYNAMIC_EXECUTION_FORBIDDEN", callee)
            if tail in {"getattr", "__import__", "import_module"}:
                for argument in node.args:
                    literal = _string_literal(argument)
                    if literal is not None and (
                        _forbidden_import(literal) or _forbidden_identifier(literal)
                    ):
                        add(node, "DYNAMIC_FORBIDDEN_SYMBOL", literal)
            process_text = _process_literal_text(node)
            if process_text is not None and _classify_registration_command(process_text):
                add(node, "FORBIDDEN_PROCESS_LITERAL", process_text)
        elif isinstance(node, (ast.Name, ast.Attribute)):
            identifier = node.id if isinstance(node, ast.Name) else node.attr
            if _normalized_identifier(identifier) in _FORBIDDEN_RESULT_IDENTIFIERS:
                add(node, "FORBIDDEN_RESULT_IDENTIFIER", identifier)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if key is None:
                    continue
                literal = _string_literal(key)
                if literal is not None and _forbidden_identifier(literal):
                    add(key, "FORBIDDEN_RESULT_FIELD", literal)

    executable = _executable_source_without_text(source)
    lexical_patterns = _LEXICAL_EXECUTABLE_PATTERNS
    if authority_protocol:
        lexical_patterns = tuple(
            pattern
            for pattern, name in zip(
                _LEXICAL_EXECUTABLE_PATTERNS,
                sorted(
                    _FORBIDDEN_EXECUTABLE_IDENTIFIERS
                    | _FORBIDDEN_RESULT_IDENTIFIERS
                ),
            )
            if name not in _FORBIDDEN_PROTOCOL_SYMBOLS
        )
    for pattern in lexical_patterns:
        match = pattern.search(executable)
        if match is not None:
            line = executable.count("\n", 0, match.start()) + 1
            findings.append(
                {
                    "path": relative,
                    "line": line,
                    "kind": "FORBIDDEN_EXECUTABLE_TOKEN",
                    "detail": match.group(0),
                }
            )
    unique = {
        (row["path"], row["line"], row["kind"], row["detail"]): row
        for row in findings
    }
    return [unique[key] for key in sorted(unique)]


def assert_static_scope_safe(repository: str | Path) -> dict[str, Any]:
    """AST-audit the FMB1 preparation execution scope and fail on any violation.

    Frozen ``protocol.py`` is AST-audited too: its explanatory strings are
    ignored, while forbidden imports and backend calls still fail. Preparation
    code may only import its non-authorizing helpers.
    """

    root = Path(repository).resolve(strict=True)
    if not root.is_dir():
        raise PreregistrationFirewallError(f"repository is not a directory: {root}")
    files, excluded = _discover_static_scope(root)
    if not files:
        raise PreregistrationFirewallError("no FMB1 preparation Python sources found")
    findings = [row for path in files for row in _scan_source(path, root)]
    report = {
        "schema": "mid360_fmb1_preregistration_static_scope_v1",
        "pass": not findings,
        "repository": str(root),
        "scanned_file_count": len(files),
        "scanned_files": [path.relative_to(root).as_posix() for path in files],
        "authority_text_allowed_files": [
            "experiments/mid360_formal_batch1/protocol.py"
        ],
        "excluded_authority_files": excluded,
        "violation_count": len(findings),
        "violations": findings,
    }
    if findings:
        raise PreregistrationFirewallError(
            f"unsafe FMB1 preparation scope: {findings}"
        )
    return report


def snapshot_registration_processes(
    proc_root: str | Path = "/proc",
) -> list[dict[str, Any]]:
    """Return obvious live registration processes from a Linux ``/proc`` view."""

    root = Path(proc_root)
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name):
        if not directory.name.isdigit() or not directory.is_dir():
            continue
        try:
            pid = int(directory.name)
            raw = (directory / "cmdline").read_bytes()
            command_parts = [
                os.fsdecode(part) for part in raw.split(b"\0") if part
            ]
            comm_path = directory / "comm"
            comm = comm_path.read_text(encoding="utf-8").strip() if comm_path.exists() else ""
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, UnicodeError):
            continue
        command: Sequence[str] = command_parts or ([comm] if comm else [])
        if not command:
            continue
        kind = _classify_registration_command(command)
        if kind is None:
            continue
        rows.append(
            {
                "pid": pid,
                "comm": comm,
                "executable": Path(command[0]).name,
                "registration_process_kind": kind,
            }
        )
    return rows


class NoRegistrationGuard:
    """Block obvious registration process/callable entry points during FMB1 prep."""

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
        self._original_open3d: dict[str, Any] = {}
        self.active = False
        self.ever_activated = False
        self.blocked_process_attempt_count = 0
        self.pcl_cli_invocation_count = 0
        self.open3d_registration_call_count = 0
        self.processes_at_entry: list[dict[str, Any]] = []
        self.processes_at_exit: list[dict[str, Any]] = []

    def _record_blocked_process(self, command: Any, kind: str) -> None:
        self.blocked_process_attempt_count += 1
        if kind == "pcl":
            self.pcl_cli_invocation_count += 1
        elif kind == "open3d":
            self.open3d_registration_call_count += 1

    def _guard_process(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(command: Any, *args: Any, **kwargs: Any) -> Any:
            kind = _classify_registration_command(command)
            if kind is not None:
                self._record_blocked_process(command, kind)
                raise RegistrationForbiddenError(
                    f"registration process is forbidden during FMB1 preparation: {command!r}"
                )
            return original(command, *args, **kwargs)

        return guarded

    def _guard_os_system(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(command: Any) -> Any:
            kind = _classify_registration_command(command)
            if kind is not None:
                self._record_blocked_process(command, kind)
                raise RegistrationForbiddenError(
                    f"registration process is forbidden during FMB1 preparation: {command!r}"
                )
            return original(command)

        return guarded

    def _guard_open3d(self) -> None:
        module = self.open3d_module or sys.modules.get("open3d")
        if module is None:
            return
        registration = getattr(getattr(module, "pipelines", None), "registration", None)
        if registration is None:
            return
        for name in dir(registration):
            if not (
                name.startswith("registration_")
                or name.startswith("get_information_matrix_")
            ):
                continue
            value = getattr(registration, name)
            if not callable(value):
                continue
            self._original_open3d[name] = value

            def denied(*args: Any, _name: str = name, **kwargs: Any) -> Any:
                self.open3d_registration_call_count += 1
                raise RegistrationForbiddenError(
                    f"Open3D registration entry is forbidden during FMB1 preparation: {_name}"
                )

            setattr(registration, name, denied)
        self.open3d_module = module

    def _restore(self) -> None:
        module = self.open3d_module
        if module is not None:
            registration = getattr(getattr(module, "pipelines", None), "registration", None)
            if registration is not None:
                for name, value in self._original_open3d.items():
                    setattr(registration, name, value)
        self._original_open3d.clear()
        for name, value in self._original_subprocess.items():
            setattr(subprocess, name, value)
        self._original_subprocess.clear()
        if self._original_os_system is not None:
            os.system = self._original_os_system
            self._original_os_system = None

    def __enter__(self) -> "NoRegistrationGuard":
        value = os.environ.get(NO_FORMAL_REGISTRATION_ENV, "").strip().lower()
        if value not in {"1", "true"}:
            raise RegistrationForbiddenError(
                f"{NO_FORMAL_REGISTRATION_ENV}=true is required"
            )
        if self.active:
            raise RegistrationForbiddenError("NoRegistrationGuard is already active")
        self.processes_at_entry = snapshot_registration_processes(self.proc_root)
        try:
            for name in self._SUBPROCESS_NAMES:
                original = getattr(subprocess, name)
                self._original_subprocess[name] = original
                setattr(subprocess, name, self._guard_process(original))
            self._original_os_system = os.system
            os.system = self._guard_os_system(os.system)
            self._guard_open3d()
        except Exception:
            self._restore()
            raise
        self.active = True
        self.ever_activated = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.processes_at_exit = snapshot_registration_processes(self.proc_root)
        self._restore()
        self.active = False

    def current_registration_processes(self) -> list[dict[str, Any]]:
        if self.active:
            return snapshot_registration_processes(self.proc_root)
        return list(self.processes_at_exit)

    def new_registration_processes(self) -> list[dict[str, Any]]:
        baseline = {int(row["pid"]) for row in self.processes_at_entry}
        return [
            row
            for row in self.current_registration_processes()
            if int(row["pid"]) not in baseline
        ]

    def report(self) -> dict[str, Any]:
        """Return one JSON-safe preparation-stage guard report.

        A report may be taken while the guard is active or after it exits. The
        latter includes the exit ``/proc`` snapshot and is preferred for
        multi-stage orchestration.
        """

        current = self.current_registration_processes()
        observed_by_pid = {
            int(row["pid"]): dict(row)
            for row in [*self.processes_at_entry, *current]
        }
        observed = [observed_by_pid[pid] for pid in sorted(observed_by_pid)]
        new = self.new_registration_processes()
        observed_pcl = sum(
            row.get("registration_process_kind") == "pcl" for row in observed
        )
        observed_open3d = sum(
            row.get("registration_process_kind") == "open3d" for row in observed
        )
        open3d_count = self.open3d_registration_call_count + observed_open3d
        pcl_count = self.pcl_cli_invocation_count + observed_pcl
        other_count = self.blocked_process_attempt_count + len(observed)
        passed = bool(
            self.ever_activated
            and open3d_count == 0
            and pcl_count == 0
            and other_count == 0
        )
        return {
            "schema": "mid360_fmb1_no_registration_guard_stage_v1",
            "status": "PASS" if passed else "FAIL",
            "pass": passed,
            "guard_was_activated": bool(self.ever_activated),
            "guard_active_at_report": bool(self.active),
            "open3d_registration_call_count": int(open3d_count),
            "pcl_cli_invocation_count": int(pcl_count),
            "other_registration_process_count": int(other_count),
            "formal_trial_count": 0,
            "blocked_process_attempt_count": int(
                self.blocked_process_attempt_count
            ),
            "registration_processes_at_guard_entry": [
                dict(row) for row in self.processes_at_entry
            ],
            "registration_processes_at_report": [dict(row) for row in current],
            "registration_processes_observed": observed,
            "new_registration_processes": [dict(row) for row in new],
        }


_STAGE_COUNT_FIELDS = (
    "open3d_registration_call_count",
    "pcl_cli_invocation_count",
    "other_registration_process_count",
    "formal_trial_count",
)


def _validated_stage_report(value: Mapping[str, Any], index: int) -> dict[str, Any]:
    report = dict(value)
    for field in _STAGE_COUNT_FIELDS:
        count = report.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PreregistrationFirewallError(
                f"stage_reports[{index}].{field} must be a nonnegative integer"
            )
    if report.get("guard_was_activated") is not True:
        raise PreregistrationFirewallError(
            f"stage_reports[{index}] was not produced by an activated guard"
        )
    return report


def build_no_icp_attestation(
    repository: str | Path,
    guard: NoRegistrationGuard | None = None,
    *,
    static_scope_report: Mapping[str, Any] | None = None,
    static_report: Mapping[str, Any] | None = None,
    stage_reports: Sequence[Mapping[str, Any]] | None = None,
    formal_trial_count: int = 0,
) -> dict[str, Any]:
    """Build the FMB1 zero-registration attestation without authorizing ICP."""

    root = Path(repository).resolve(strict=True)
    if isinstance(formal_trial_count, bool) or not isinstance(formal_trial_count, int):
        raise PreregistrationFirewallError("formal_trial_count must be an integer")
    if formal_trial_count < 0:
        raise PreregistrationFirewallError("formal_trial_count cannot be negative")
    if static_scope_report is not None and static_report is not None:
        raise PreregistrationFirewallError(
            "pass only one of static_scope_report or static_report"
        )
    supplied_static_report = (
        static_scope_report if static_scope_report is not None else static_report
    )
    reports = [
        _validated_stage_report(value, index)
        for index, value in enumerate(stage_reports or ())
    ]
    if guard is not None:
        if not guard.ever_activated:
            raise PreregistrationFirewallError(
                "NoRegistrationGuard must be activated before attestation"
            )
        reports.append(_validated_stage_report(guard.report(), len(reports)))
    if not reports:
        raise PreregistrationFirewallError(
            "at least one activated guard or stage report is required"
        )
    # Never trust a caller-supplied PASS in a scientific attestation. Re-scan
    # the current bytes, then require any supplied report to describe the same
    # file set and violation count.
    verified_static_report = assert_static_scope_safe(root)
    supplied_static_matches = True
    if supplied_static_report is not None:
        supplied_static_matches = (
            supplied_static_report.get("pass") is True
            and supplied_static_report.get("scanned_files")
            == verified_static_report.get("scanned_files")
            and supplied_static_report.get("violation_count") == 0
        )
    static_report = verified_static_report
    static_safe = static_report.get("pass") is True and supplied_static_matches
    backend_path = root / "frozen_assets/backend_parameter_contract.json"
    try:
        backend_sha = _sha256_file(backend_path.resolve(strict=True))
    except (FileNotFoundError, OSError):
        backend_sha = None
    backend_unchanged = backend_sha == EXPECTED_BACKEND_CONTRACT_SHA256
    open3d_count = sum(
        int(report["open3d_registration_call_count"]) for report in reports
    )
    pcl_count = sum(int(report["pcl_cli_invocation_count"]) for report in reports)
    other_count = sum(
        int(report["other_registration_process_count"]) for report in reports
    )
    formal_count = formal_trial_count + sum(
        int(report["formal_trial_count"]) for report in reports
    )
    observed_processes = [
        dict(row)
        for report in reports
        for row in report.get("registration_processes_observed", [])
    ]
    new_processes = [
        dict(row)
        for report in reports
        for row in report.get("new_registration_processes", [])
    ]
    passed = bool(
        static_safe
        and backend_unchanged
        and open3d_count == 0
        and pcl_count == 0
        and other_count == 0
        and formal_count == 0
        and all(report.get("pass") is True for report in reports)
    )
    return {
        "schema": "mid360_fmb1_no_icp_attestation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "NO_FORMAL_REGISTRATION": True,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "FORMAL_MEASUREMENT_RESULT": False,
        "open3d_registration_call_count": int(open3d_count),
        "pcl_cli_invocation_count": int(pcl_count),
        "other_registration_process_count": int(other_count),
        "formal_trial_count": int(formal_count),
        "registration_execution_count": 0,
        "actual_trials": 0,
        "actual_registration_trials": 0,
        "backend_parameter_contract_path": str(backend_path),
        "backend_parameter_contract_sha256": backend_sha,
        "backend_contract_sha256": backend_sha,
        "backend_parameter_contract_expected_sha256": EXPECTED_BACKEND_CONTRACT_SHA256,
        "backend_parameter_contract_unchanged": backend_unchanged,
        "static_scope_safe": static_safe,
        "static_scope_scanned_file_count": int(
            static_report.get("scanned_file_count", 0)
        ),
        "guard_environment_flag": f"{NO_FORMAL_REGISTRATION_ENV}=true",
        "guard_was_activated": all(
            report.get("guard_was_activated") is True for report in reports
        ),
        "guard_stage_count": len(reports),
        "guard_stage_reports": reports,
        "registration_processes_observed": observed_processes,
        "new_registration_processes": new_processes,
    }


__all__ = [
    "EXPECTED_BACKEND_CONTRACT_SHA256",
    "NO_FORMAL_REGISTRATION_ENV",
    "NoRegistrationGuard",
    "PreregistrationFirewallError",
    "RegistrationForbiddenError",
    "assert_static_scope_safe",
    "build_no_icp_attestation",
    "snapshot_registration_processes",
]
