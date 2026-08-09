from __future__ import annotations

import ast
from pathlib import Path

from phase_a_harness.synthetic_confirmatory_v3_adapter import (
    ADAPTER_SCHEMA,
    QUALIFICATION_RUN_ID,
    qualification_layout,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v3_adapter_is_seed_free_before_schedule_derivation() -> None:
    path = ROOT / "src/phase_a_harness/synthetic_confirmatory_v3_adapter.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "numpy" not in imported_roots
    assert "random" not in imported_roots
    assert "zero_perturbation_synthetic_confirmatory_v3_" not in source
    assert ADAPTER_SCHEMA == "synthetic_confirmatory_v3_execution_adapter_fixture_v1"


def test_v3_adapter_qualification_layout_is_exact_and_external() -> None:
    layout = qualification_layout(repository=ROOT, resume=True)
    assert QUALIFICATION_RUN_ID == "synthetic_confirmatory_v3_prerun"
    assert layout.run_root == Path(
        "/home/lj/zero_perturbation_runtime/qualification/"
        "synthetic_confirmatory_v3_prerun"
    )
    assert ROOT not in layout.run_root.parents
    assert Path("/home/lj/Degen-LIO") not in layout.run_root.parents
    assert layout.snapshot_cache.parent == layout.run_root
    assert layout.snapshot_lock.parent == layout.run_root
    assert layout.raw_results.parent == layout.run_root
    assert layout.artifact_staging.parent == layout.run_root


def test_v3_adapter_delegates_instead_of_copying_scientific_algorithms() -> None:
    source = (
        ROOT / "src/phase_a_harness/synthetic_confirmatory_v3_adapter.py"
    ).read_text(encoding="utf-8")
    forbidden_definitions = {
        "analyze_synthetic_confirmatory_records",
        "independently_recompute_synthetic_confirmatory",
        "build_snapshot",
        "execute_open3d_fixture",
        "execute_pcl_fixture",
    }
    defined = {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not (defined & forbidden_definitions)
    assert "run_fixture_lifecycle" in source
    assert "analyze_v2_fixture_results" in source
    assert "independently_analyze_v2_fixture_results" in source
    assert "publish_fixture_runtime_artifact" in source
