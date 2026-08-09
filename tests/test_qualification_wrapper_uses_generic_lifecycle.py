from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_DRIVER = (
    REPOSITORY_ROOT / "scripts/qualify_version_agnostic_formal_lifecycle.py"
)
CONTEXT_RUNNER = (
    REPOSITORY_ROOT
    / "scripts/run_version_agnostic_formal_lifecycle_fixture.py"
)


def _tree(path: Path) -> ast.Module:
    assert path.is_file(), (
        f"qualification wrapper is absent: "
        f"{path.relative_to(REPOSITORY_ROOT)}"
    )
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(path: Path, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in _tree(path).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, (
        f"{path.relative_to(REPOSITORY_ROOT)} must define exactly one {name}"
    )
    return matches[0]


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _calls(function: ast.FunctionDef) -> list[str]:
    return [
        _call_name(node)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]


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


def _loop_iterators(function: ast.FunctionDef) -> list[ast.AST]:
    iterators: list[ast.AST] = []
    for node in ast.walk(function):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iterators.append(node.iter)
        elif isinstance(
            node,
            (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            iterators.extend(generator.iter for generator in node.generators)
    return iterators


def _contains_snapshot_or_trial_reference(node: ast.AST) -> bool:
    values = {
        child.id.lower()
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    }
    values.update(
        child.attr.lower()
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
    )
    values.update(
        child.value.lower()
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    )
    return any(
        "snapshot" in value or "trial" in value for value in values
    )


def test_qualification_wrappers_have_no_snapshot_or_trial_execution_loop() -> None:
    wrappers = (
        _function(QUALIFICATION_DRIVER, "_run_context"),
        _function(CONTEXT_RUNNER, "main"),
    )

    for wrapper in wrappers:
        assert not any(
            isinstance(node, (ast.For, ast.AsyncFor, ast.While))
            for node in ast.walk(wrapper)
        ), f"{wrapper.name} contains a direct statement execution loop"
        forbidden_iterators = [
            iterator
            for iterator in _loop_iterators(wrapper)
            if _contains_snapshot_or_trial_reference(iterator)
        ]
        assert forbidden_iterators == [], (
            f"{wrapper.name} iterates a snapshot/trial collection directly"
        )


def test_qualification_wrappers_never_call_private_fixture_or_dispatch() -> None:
    wrappers = (
        _function(QUALIFICATION_DRIVER, "_run_context"),
        _function(CONTEXT_RUNNER, "main"),
        _function(QUALIFICATION_DRIVER, "main"),
    )
    forbidden = {
        "_execute_one",
        "_fixture",
        "snapshot_materializer",
        "snapshot_reader",
        "snapshot_validator",
        "trial_validator",
        "backend_registry",
        "execute_open3d_fixture",
        "execute_pcl_fixture",
        "monkeypatch",
        "patch",
    }

    for wrapper in wrappers:
        present = sorted(_identifiers(wrapper) & forbidden)
        assert present == [], (
            f"{wrapper.name} bypasses the generic lifecycle: {present}"
        )


def test_qualification_execution_delegates_only_to_generic_entries() -> None:
    context_driver = _function(QUALIFICATION_DRIVER, "_run_context")
    context_runner = _function(CONTEXT_RUNNER, "main")

    # The qualification driver must exercise the real CLI boundary twice
    # (fresh then resume).  The CLI is the sole in-process delegate to the two
    # generic production entries; the driver must not call either entry
    # directly or recreate their loops.
    driver_calls = _calls(context_driver)
    assert driver_calls.count("_run_fixture_process") == 2
    assert not {
        "execute_formal_lifecycle",
        "execute_postrun_pipeline",
    } & set(driver_calls)

    execute_calls = sorted(
        name
        for name in _calls(context_runner)
        if name.startswith("execute_")
    )
    assert execute_calls == [
        "execute_formal_lifecycle",
        "execute_postrun_pipeline",
    ]

    qualification_main_calls = _calls(
        _function(QUALIFICATION_DRIVER, "main")
    )
    assert qualification_main_calls.count("_run_context") == 2
