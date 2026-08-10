from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_cli_exposes_required_fail_closed_options() -> None:
    module = _load("real_data_prepare_cli", "scripts/prepare_real_data_validation_v1.py")
    destinations = {action.dest for action in module.build_parser()._actions}
    assert {
        "repository_root",
        "data_root",
        "runtime_root",
        "synthetic_run_root",
        "mode",
        "workers",
        "metadata_only",
        "verify_only",
        "no_registration",
    }.issubset(destinations)


def test_independent_verifier_requires_only_runtime_root() -> None:
    module = _load("real_data_verify_cli", "scripts/verify_real_data_preregistration_v1.py")
    destinations = {action.dest for action in module.build_parser()._actions}
    assert destinations == {"help", "runtime_root"}
