from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERIC_CORE_FILES = (
    REPOSITORY_ROOT / "src/phase_a_harness/formal_lifecycle.py",
    REPOSITORY_ROOT / "src/phase_a_harness/formal_lifecycle_contract.py",
    REPOSITORY_ROOT / "src/phase_a_harness/formal_lifecycle_components.py",
    REPOSITORY_ROOT / "src/phase_a_harness/formal_lifecycle_paths.py",
)
FORBIDDEN_V3_REFERENCES = (
    "synthetic-confirmatory-v3",
    "synthetic_confirmatory_v3",
    "V3_FORMAL_RUNTIME_ROOT",
    "V3_MANIFEST",
    "V3_CONTRACT",
    "V3_RUN_ID",
    "V3_TAG",
    "V3_BRANCH",
    "load_v3_",
)
_RUN_ID_TARGET = re.compile(r"(?:^|_)run_id$", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    category: str
    value: str

    def render(self) -> str:
        relative = self.path.relative_to(REPOSITORY_ROOT)
        return f"{relative}:{self.line}: {self.category}: {self.value!r}"


def _generic_sources() -> Iterable[tuple[Path, str, ast.Module]]:
    missing = [path for path in GENERIC_CORE_FILES if not path.is_file()]
    assert not missing, "generic lifecycle core is incomplete: " + ", ".join(
        str(path.relative_to(REPOSITORY_ROOT)) for path in missing
    )
    for path in GENERIC_CORE_FILES:
        source = path.read_text(encoding="utf-8")
        yield path, source, ast.parse(source, filename=str(path))


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _target_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Attribute):
        return (target.attr,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            name for child in target.elts for name in _target_names(child)
        )
    return ()


def _fixed_run_id_findings(path: Path, tree: ast.Module) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _constant_string(node.value)
            names = tuple(
                name for target in node.targets for name in _target_names(target)
            )
            if value is not None and any(_RUN_ID_TARGET.search(name) for name in names):
                findings.append(
                    Finding(path, node.lineno, "fixed run ID assignment", value)
                )
        elif isinstance(node, ast.AnnAssign):
            value = _constant_string(node.value)
            names = _target_names(node.target)
            if value is not None and any(_RUN_ID_TARGET.search(name) for name in names):
                findings.append(
                    Finding(path, node.lineno, "fixed run ID assignment", value)
                )
        elif isinstance(node, ast.keyword):
            value = _constant_string(node.value)
            if (
                node.arg is not None
                and _RUN_ID_TARGET.search(node.arg)
                and value is not None
            ):
                findings.append(
                    Finding(path, node.value.lineno, "fixed run ID argument", value)
                )
        elif isinstance(node, ast.Dict):
            for key_node, value_node in zip(node.keys, node.values):
                key = _constant_string(key_node)
                value = _constant_string(value_node)
                if (
                    key is not None
                    and _RUN_ID_TARGET.search(key)
                    and value is not None
                ):
                    findings.append(
                        Finding(
                            path,
                            value_node.lineno,
                            "fixed run ID mapping value",
                            value,
                        )
                    )
    return findings


def _path_literal_findings(path: Path, tree: ast.Module) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        value = _constant_string(node)
        if value is None:
            continue
        lowered = value.lower()
        is_absolute_runtime = value.startswith("/") and any(
            marker in lowered
            for marker in ("runtime", "confirmatory", "qualification")
        )
        is_manifest_path = "manifest" in lowered and (
            "/" in value
            or "\\" in value
            or lowered.endswith((".json", ".yaml", ".yml"))
        )
        if is_absolute_runtime:
            findings.append(
                Finding(path, node.lineno, "fixed absolute runtime path", value)
            )
        if is_manifest_path:
            findings.append(
                Finding(path, node.lineno, "fixed manifest path", value)
            )
    return findings


def _is_formal_execution_context_call(node: ast.Call) -> bool:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id == "formal_execution_context"
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "formal_execution_context"
    )


def test_generic_core_has_zero_forbidden_v3_references() -> None:
    findings: list[Finding] = []
    for path, source, _tree in _generic_sources():
        for token in FORBIDDEN_V3_REFERENCES:
            for match in re.finditer(re.escape(token), source):
                findings.append(
                    Finding(
                        path,
                        source.count("\n", 0, match.start()) + 1,
                        "forbidden v3 reference",
                        token,
                    )
                )

    assert not findings, "GENERIC_CORE_V3_REFERENCE_COUNT != 0\n" + "\n".join(
        finding.render() for finding in findings
    )


def test_generic_core_has_zero_fixed_runtime_manifest_or_run_id() -> None:
    findings: list[Finding] = []
    for path, _source, tree in _generic_sources():
        findings.extend(_path_literal_findings(path, tree))
        findings.extend(_fixed_run_id_findings(path, tree))

    assert not findings, (
        "generic core contains a fixed runtime, manifest, or run identity\n"
        + "\n".join(finding.render() for finding in findings)
    )


def test_formal_execution_context_requires_explicit_arguments() -> None:
    findings: list[Finding] = []
    for path, _source, tree in _generic_sources():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _is_formal_execution_context_call(node)
                and not node.args
                and not node.keywords
            ):
                findings.append(
                    Finding(
                        path,
                        node.lineno,
                        "implicit context reload",
                        "formal_execution_context()",
                    )
                )

    assert not findings, (
        "generic core calls formal_execution_context() without explicit arguments\n"
        + "\n".join(finding.render() for finding in findings)
    )
