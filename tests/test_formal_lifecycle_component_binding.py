from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from phase_a_harness.formal_lifecycle_components import (
    BACKEND_IDS,
    ComponentBinding,
    FormalLifecycleComponentError,
    FormalLifecycleComponents,
)
from phase_a_harness.execution_context import ExecutionMode


_COMPONENT_FIELDS = (
    "snapshot_materializer",
    "snapshot_reader",
    "snapshot_validator",
    "trial_validator",
    "snapshot_lock_builder",
    "snapshot_lock_validator",
    "backend_input_builder",
    "common_record_builder",
    "result_validator",
    "resume_result_validator",
    "primary_analyzer",
    "independent_verifier",
    "difference_auditor",
    "publisher",
    "artifact_verifier",
    "git_gate",
    "run_contract_builder",
)


def _load_component_module(implementation_path: Path) -> Any:
    implementation_path.write_text(
        "def component(**kwargs):\n"
        "    return dict(kwargs)\n"
        "\n"
        "def bad_module(**kwargs):\n"
        "    return dict(kwargs)\n"
        "\n"
        "def bad_qualname(**kwargs):\n"
        "    return dict(kwargs)\n",
        encoding="utf-8",
    )
    module_name = (
        "_component_binding_fixture_"
        + hashlib.sha256(str(implementation_path).encode("utf-8")).hexdigest()[:16]
    )
    module_spec = importlib.util.spec_from_file_location(
        module_name, implementation_path
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def _valid_binding(tmp_path: Path) -> tuple[ComponentBinding, Any, Path, str]:
    implementation_path = (tmp_path / "components.py").resolve()
    module = _load_component_module(implementation_path)
    implementation_sha256 = hashlib.sha256(
        implementation_path.read_bytes()
    ).hexdigest()
    binding = ComponentBinding(
        component_id="fixture_component",
        callable=module.component,
        implementation_path=implementation_path,
        implementation_sha256=implementation_sha256,
    )
    return binding, module, implementation_path, implementation_sha256


def _bundle(binding: ComponentBinding, **overrides: Any) -> FormalLifecycleComponents:
    values: dict[str, Any] = {
        "bundle_id": "fixture-component-bundle-v1",
        "bound_mode": ExecutionMode.QUALIFICATION,
        "bound_contract_id": "fixture-contract",
        "bound_plan_id": "fixture-plan",
        "bound_manifest_sha256": "a" * 64,
        "bound_run_id": "fixture-run",
        "bound_runtime_root": Path("/tmp/formal-lifecycle-fixture-runtime"),
        "bound_publisher_inventory_sha256": "b" * 64,
        "backend_registry": {backend: binding for backend in BACKEND_IDS},
        **{field: binding for field in _COMPONENT_FIELDS},
    }
    values.update(overrides)
    return FormalLifecycleComponents(**values)


def test_component_binding_reports_callable_and_file_identity(
    tmp_path: Path,
) -> None:
    binding, module, implementation_path, implementation_sha256 = _valid_binding(
        tmp_path
    )

    assert binding.report() == {
        "component_id": "fixture_component",
        "callable_module": module.__name__,
        "callable_qualname": "component",
        "implementation_path": str(implementation_path),
        "implementation_sha256": implementation_sha256,
    }


def test_component_binding_rejects_file_sha_mismatch_before_entry(
    tmp_path: Path,
) -> None:
    _, module, implementation_path, implementation_sha256 = _valid_binding(
        tmp_path
    )
    wrong_sha256 = (
        "0" * 64 if implementation_sha256 != "0" * 64 else "1" * 64
    )

    with pytest.raises(FormalLifecycleComponentError, match="SHA mismatch"):
        ComponentBinding(
            component_id="fixture_component",
            callable=module.component,
            implementation_path=implementation_path,
            implementation_sha256=wrong_sha256,
        )


def test_component_binding_rejects_callable_file_mismatch_before_entry(
    tmp_path: Path,
) -> None:
    _, module, _, _ = _valid_binding(tmp_path)
    different_path = (tmp_path / "different_components.py").resolve()
    different_path.write_text(
        "def component(**kwargs):\n"
        "    return dict(kwargs)\n",
        encoding="utf-8",
    )
    different_sha256 = hashlib.sha256(different_path.read_bytes()).hexdigest()

    with pytest.raises(
        FormalLifecycleComponentError,
        match="source differs from binding",
    ):
        ComponentBinding(
            component_id="fixture_component",
            callable=module.component,
            implementation_path=different_path,
            implementation_sha256=different_sha256,
        )


@pytest.mark.parametrize("component_id", ("", " component", "component "))
def test_component_binding_rejects_noncanonical_component_id(
    tmp_path: Path,
    component_id: str,
) -> None:
    _, module, implementation_path, implementation_sha256 = _valid_binding(
        tmp_path
    )

    with pytest.raises(FormalLifecycleComponentError, match="component_id"):
        ComponentBinding(
            component_id=component_id,
            callable=module.component,
            implementation_path=implementation_path,
            implementation_sha256=implementation_sha256,
        )


@pytest.mark.parametrize(
    ("callable_name", "identity_attribute"),
    (
        ("bad_module", "__module__"),
        ("bad_qualname", "__qualname__"),
    ),
)
def test_component_binding_rejects_invalid_callable_identity(
    tmp_path: Path,
    callable_name: str,
    identity_attribute: str,
) -> None:
    _, module, implementation_path, implementation_sha256 = _valid_binding(
        tmp_path
    )
    callable_object = getattr(module, callable_name)
    setattr(callable_object, identity_attribute, "")

    with pytest.raises(FormalLifecycleComponentError, match="component callable"):
        ComponentBinding(
            component_id="fixture_component",
            callable=callable_object,
            implementation_path=implementation_path,
            implementation_sha256=implementation_sha256,
        )


def test_component_bundle_is_frozen_and_backend_registry_is_exact(
    tmp_path: Path,
) -> None:
    binding, _, _, _ = _valid_binding(tmp_path)
    bundle = _bundle(binding)

    assert isinstance(bundle.backend_registry, MappingProxyType)
    assert tuple(bundle.backend_registry) == BACKEND_IDS
    with pytest.raises(TypeError):
        bundle.backend_registry["native"] = binding  # type: ignore[index]
    with pytest.raises(
        FormalLifecycleComponentError,
        match="exactly Open3D and PCL",
    ):
        _bundle(
            binding,
            backend_registry={
                **{backend: binding for backend in BACKEND_IDS},
                "native": binding,
            },
        )
