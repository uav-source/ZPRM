from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_a_harness.synthetic_confirmatory_v3_requalified import (
    DEFAULT_MANIFEST_RELATIVE,
    DEFAULT_RUNTIME_ROOT,
    EXECUTION_CLASSIFICATION,
    EXPECTED_BRANCH,
    EXPECTED_TAG,
    RUN_ID,
    RequalifiedV3Error,
    build_requalified_formal_lifecycle_spec,
    build_requalified_manifest,
    load_requalified_plans,
    requalified_resume_result_validator,
    scientific_asset_hashes,
    strict_json_object,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_requalified_manifest_binds_unchanged_595_1190_contract() -> None:
    generated = build_requalified_manifest(repository_root=REPOSITORY)
    committed = strict_json_object(REPOSITORY / DEFAULT_MANIFEST_RELATIVE)
    snapshots, trials = load_requalified_plans(REPOSITORY)

    assert generated == committed
    assert committed["execution_classification"] == EXECUTION_CLASSIFICATION
    assert committed["historical_formal_run_impersonated"] is False
    assert committed["scientific_contract_mutation_allowed"] is False
    assert len(snapshots) == 595
    assert len(trials) == 1190
    assert committed["backend_trial_counts"] == {
        "open3d_point_to_plane": 595,
        "pcl_point_to_plane": 595,
    }
    assert committed["native_trial_count"] == 0


def test_requalified_spec_is_generic_lifecycle_bound_without_runtime_write() -> None:
    before = DEFAULT_RUNTIME_ROOT.exists()
    spec = build_requalified_formal_lifecycle_spec(
        repository=REPOSITORY,
        manifest_path=DEFAULT_MANIFEST_RELATIVE,
        run_id=RUN_ID,
        runtime_root=DEFAULT_RUNTIME_ROOT,
        workers=2,
        expected_commit="0" * 40,
        expected_branch=EXPECTED_BRANCH,
        expected_tag=EXPECTED_TAG,
    )

    assert spec.manifest["execution_classification"] == EXECUTION_CLASSIFICATION
    assert len(spec.snapshot_plan) == 595
    assert len(spec.trial_plan) == 1190
    assert spec.workers == 2
    contract = spec.component_bundle.run_contract_builder.callable(spec=spec)
    assert contract["confirmatory_seed_access_count"] == 3290
    assert contract["confirmatory_rng_instantiation_count"] == 1575
    assert set(spec.component_bundle.backend_registry) == {
        "open3d_point_to_plane",
        "pcl_point_to_plane",
    }
    assert DEFAULT_RUNTIME_ROOT.exists() is before


def test_scientific_asset_hashes_are_exact() -> None:
    report = scientific_asset_hashes(REPOSITORY)

    assert report["all_assets_match"] is True
    assert report["mismatch_count"] == 0
    assert report["asset_count"] == 8


def test_requalified_resume_rejects_uncommitted_orphan(
    tmp_path: Path,
) -> None:
    from phase_a_harness.runtime_lifecycle_io import (
        atomic_create_canonical_json,
    )

    manifest_path = tmp_path / "raw_result_manifest.json"
    atomic_create_canonical_json(manifest_path, {"results": {}})
    spec = SimpleNamespace(paths=SimpleNamespace(raw_manifest=manifest_path))
    with pytest.raises(RequalifiedV3Error, match="uncommitted orphan"):
        requalified_resume_result_validator(
            spec=spec,
            trial_row={
                "condition": "IDEAL_MATCHED",
                "planned_trial_id": "orphan-trial",
            },
            common={},
            path=tmp_path / "orphan.json",
            entry={
                "path": "orphan.json",
                "planned_trial_id": "orphan-trial",
                "sha256": "0" * 64,
            },
        )


def test_committed_manifest_rejects_another_runtime_root(tmp_path: Path) -> None:
    with pytest.raises(RequalifiedV3Error, match="constructed contract"):
        build_requalified_formal_lifecycle_spec(
            repository=REPOSITORY,
            manifest_path=DEFAULT_MANIFEST_RELATIVE,
            run_id=RUN_ID,
            runtime_root=(tmp_path / "different-runtime").resolve(),
            workers=2,
            expected_commit="0" * 40,
            expected_branch=EXPECTED_BRANCH,
            expected_tag=EXPECTED_TAG,
        )


def test_cli_exposes_required_non_scientific_lifecycle_options() -> None:
    path = REPOSITORY / "scripts/run_synthetic_confirmatory_v3_requalified.py"
    module_spec = importlib.util.spec_from_file_location("v3_requalified_cli", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    parser = module.build_parser()
    destinations = {action.dest for action in parser._actions}

    assert {
        "preflight",
        "dry_run",
        "mode",
        "runtime_root",
        "workers",
        "run_postrun",
    }.issubset(destinations)
