from __future__ import annotations

import ast
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V3_RUNNER = (
    REPOSITORY_ROOT
    / "src/phase_a_harness/synthetic_confirmatory_v3_runner.py"
)
PUBLIC_WRAPPER = "execute_synthetic_confirmatory_v3"
_LOCK_IDENTIFIER = re.compile(r"(?:^|_)lock(?:_|$)", re.IGNORECASE)


def _wrapper_node() -> ast.FunctionDef:
    source = V3_RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(V3_RUNNER))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == PUBLIC_WRAPPER
    ]
    assert len(matches) == 1, (
        f"{V3_RUNNER.relative_to(REPOSITORY_ROOT)} must expose exactly one "
        f"{PUBLIC_WRAPPER}"
    )
    return matches[0]


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _identifiers(function: ast.FunctionDef) -> set[str]:
    names = {
        node.id for node in ast.walk(function) if isinstance(node, ast.Name)
    }
    names.update(
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
    )
    return names


def test_v3_public_execute_wrapper_has_no_execution_loop_or_pool() -> None:
    wrapper = _wrapper_node()
    loop_nodes = (
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    assert not any(isinstance(node, loop_nodes) for node in ast.walk(wrapper))
    identifiers = _identifiers(wrapper)
    forbidden_concurrency = {
        "ThreadPoolExecutor",
        "ProcessPoolExecutor",
        "as_completed",
        "executor",
        "futures",
        "submit",
    }
    assert not (identifiers & forbidden_concurrency), (
        "v3 compatibility wrapper contains execution-pool logic: "
        f"{sorted(identifiers & forbidden_concurrency)}"
    )


def test_v3_public_execute_wrapper_has_no_lock_manifest_or_backend_logic() -> None:
    wrapper = _wrapper_node()
    identifiers = _identifiers(wrapper)

    lock_names = sorted(
        name for name in identifiers if _LOCK_IDENTIFIER.search(name)
    )
    raw_manifest_names = sorted(
        name for name in identifiers if "raw_manifest" in name.lower()
    )
    forbidden_backend_names = {
        "_execute_one",
        "_fixture",
        "backend_registry",
        "execute_open3d_fixture",
        "execute_pcl_fixture",
        "execute_full_synthetic_open3d_fixture",
        "execute_full_synthetic_pcl_fixture",
    }
    backend_names = sorted(identifiers & forbidden_backend_names)

    assert lock_names == [], f"v3 wrapper contains lock logic: {lock_names}"
    assert raw_manifest_names == [], (
        f"v3 wrapper contains raw-manifest logic: {raw_manifest_names}"
    )
    assert backend_names == [], (
        f"v3 wrapper contains backend dispatch logic: {backend_names}"
    )


def test_v3_public_execute_wrapper_delegates_only_to_generic_lifecycle() -> None:
    wrapper = _wrapper_node()
    calls = [
        _call_name(node)
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
    ]

    assert calls.count("execute_formal_lifecycle") == 1
    execution_calls = sorted(
        name for name in calls if name.startswith("execute_")
    )
    assert execution_calls == ["execute_formal_lifecycle"]

    imports = {
        (node.module, alias.name)
        for node in ast.walk(wrapper)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("formal_lifecycle", "execute_formal_lifecycle") in imports
