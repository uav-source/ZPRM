from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phase_a_harness.contracts import file_sha256
from phase_a_harness import synthetic_confirmatory_v2_contract as contract
from phase_a_harness import synthetic_confirmatory_v2_snapshot_builder as v2_builder
from phase_a_harness.synthetic_confirmatory_runner import (
    load_synthetic_confirmatory_stack as load_v1_stack,
)
from phase_a_harness.synthetic_confirmatory_v2_independent_verifier import (
    compare_v2_fixture_primary_and_independent,
    independently_analyze_v2_fixture_results,
    independently_recompute_v2_lineage,
)
from phase_a_harness.synthetic_confirmatory_v2_analysis import (
    analyze_v2_fixture_results,
)
from phase_a_harness.synthetic_confirmatory_v2_artifact_verifier import (
    verify_synthetic_confirmatory_v2_fixture_artifact,
)
from phase_a_harness.synthetic_confirmatory_artifact_verifier import (
    verify_synthetic_confirmatory_prerun_artifact as verify_v1_prerun_artifact,
)
from phase_a_harness.synthetic_confirmatory_v2_publisher import (
    publish_synthetic_confirmatory_v2_fixture,
)
from phase_a_harness.synthetic_confirmatory_v2_runner import (
    dry_run_synthetic_confirmatory,
)
from phase_a_harness.synthetic_confirmatory_v2_prerun import (
    PRERUN_ROOT_FILES,
    REQUIRED_TRUE_GATES,
    REQUIRED_ZERO_COUNTERS,
    build_v2_prerun_decision,
)
from phase_a_harness.synthetic_confirmatory_v2_seed_audit import (
    audit_v2_seed_provenance,
)
from phase_a_harness.synthetic_confirmatory_v2_snapshot_builder import (
    DEVELOPMENT_GEOMETRY_SEEDS,
    ConfirmatoryV2SeedFirewall,
    build_development_nonideal_regression_snapshot,
    build_development_ideal_qualification_snapshot,
    build_development_independent_qualification_snapshot,
)
from phase_a_harness.full_synthetic_development_protocol import (
    SNAPSHOT_CACHE_RELATIVE as DEVELOPMENT_CACHE_RELATIVE,
    SNAPSHOT_LOCK_RELATIVE as DEVELOPMENT_LOCK_RELATIVE,
    read_full_synthetic_plans,
)
from phase_a_harness.full_synthetic_snapshot_builder import (
    read_full_synthetic_snapshot,
    validate_full_synthetic_snapshot_lock,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "1c78372ef6f2c62e441f69c028b58f7bc48f6c35"
V1_BUNDLE = Path(
    "/tmp/zero-perturbation-synthetic-confirmatory-v1-ideal-lineage-fail.bundle"
)
V1_BUNDLE_SHA = "21e803756da78c3bf06a93d957c0395850a36d6cf119b23cd00c88ab927dcb72"


def _source_only_package() -> bool:
    return "formal_runtime_artifacts_included=false" in (
        ROOT / "SOURCE_STATE.txt"
    ).read_text(encoding="utf-8")


def _require_historical_commit() -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{BASE_COMMIT}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0 and _source_only_package():
        pytest.skip("source-only ZIP excludes the historical v2 Git object")
    assert result.returncode == 0


@pytest.fixture(scope="session")
def ideal_snapshot() -> dict[str, object]:
    return build_development_ideal_qualification_snapshot(
        ROOT,
        scene="GEOMETRY_RICH_ROOM",
        geometry_seed=DEVELOPMENT_GEOMETRY_SEEDS[0],
    )


@pytest.fixture(scope="session")
def independent_snapshot() -> dict[str, object]:
    return build_development_independent_qualification_snapshot(
        ROOT,
        scene="GEOMETRY_RICH_ROOM",
        geometry_seed=DEVELOPMENT_GEOMETRY_SEEDS[0],
    )


@pytest.fixture(scope="session")
def full_noise_regression_pair() -> tuple[dict[str, object], dict[str, object]]:
    plans = read_full_synthetic_plans(ROOT)[0]
    plan = next(row for row in plans if row["condition"] == "FULL_NOISE")
    lock = validate_full_synthetic_snapshot_lock(
        ROOT / DEVELOPMENT_LOCK_RELATIVE,
        ROOT / DEVELOPMENT_CACHE_RELATIVE,
        plans,
    )
    lock_by_id = {row["snapshot_id"]: row for row in lock["snapshots"]}
    baseline = read_full_synthetic_snapshot(
        ROOT / DEVELOPMENT_CACHE_RELATIVE,
        plan,
        expected_lock_entry=lock_by_id[plan["snapshot_id"]],
        arrays=True,
    )
    candidate = build_development_nonideal_regression_snapshot(ROOT, plan)
    return baseline, candidate


def _ast_sha(path: Path, function: str) -> str:
    nodes = [
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function
    ]
    assert len(nodes) == 1
    return hashlib.sha256(
        ast.dump(nodes[0], include_attributes=False).encode("utf-8")
    ).hexdigest()


def test_v1_failure_archive_is_preserved() -> None:
    if not V1_BUNDLE.is_file() and _source_only_package():
        pytest.skip("source-only ZIP excludes the historical v1 failure bundle")
    _require_historical_commit()
    assert V1_BUNDLE.is_file()
    assert file_sha256(V1_BUNDLE) == V1_BUNDLE_SHA
    assert subprocess.run(
        ["git", "bundle", "verify", str(V1_BUNDLE)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    assert (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "archive/zero-perturbation-synthetic-confirmatory-v1-ideal-lineage-fail^{commit}",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
        == BASE_COMMIT
    )


def test_v1_historical_prerun_artifact_remains_readable() -> None:
    report = verify_v1_prerun_artifact(
        ROOT / "artifacts/synthetic_confirmatory_prerun_v1",
        write_report=False,
    )
    assert report["CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"] is True
    assert report["evidence_semantic_failure_count"] == 0
    assert report["sha256_mismatch_files"] == []


@pytest.mark.parametrize(
    ("domain", "index", "expected"),
    [
        *(('geometry', index, value) for index, value in enumerate(contract.GEOMETRY_SEEDS)),
        *(('measurement', index, value) for index, value in enumerate(contract.MEASUREMENT_SEEDS)),
        ("bootstrap", 0, contract.BOOTSTRAP_SEED),
    ],
)
def test_v2_seed_derivation_is_fixed(domain: str, index: int, expected: int) -> None:
    assert contract.derive_seed(domain, index) == expected


def test_v2_namespace_and_seed_sets_are_collision_free() -> None:
    development = {1850310744, 1957656152, 1334931069, 217775206, 1664898153}
    v1 = {
        *contract.OLD_V1_GEOMETRY_SEEDS,
        *contract.OLD_V1_MEASUREMENT_SEEDS,
        contract.OLD_V1_BOOTSTRAP_SEED,
    }
    v2 = {*contract.GEOMETRY_SEEDS, *contract.MEASUREMENT_SEEDS, contract.BOOTSTRAP_SEED}
    assert contract.NAMESPACE == (
        "zero_perturbation_synthetic_confirmatory_v2_20260729_ideal_lineage_fix"
    )
    assert len(v2) == 9
    assert not (v2 & v1)
    assert not (v2 & development)


def test_v2_static_seed_provenance_audit_is_exact_and_constructor_free() -> None:
    _require_historical_commit()
    report = audit_v2_seed_provenance(ROOT)
    assert report["STATIC_V2_SEED_PROVENANCE_AUDIT_PASS"] is True
    assert report["NEW_V2_NAMESPACE_COLLISION"] is False
    assert report["NEW_V2_SEED_PROVENANCE_COLLISION_COUNT"] == 0
    assert report["declaration_count"] == 9
    assert report["reference_counts_by_domain"] == {
        "geometry": 1785,
        "measurement": 1575,
        "total": 3360,
    }
    assert report["current_seed_literal_unexpected_path_count"] == 0
    assert report["namespace_unexpected_path_count"] == 0
    assert report["test_formal_seed_constructor_hit_count"] == 0


@pytest.mark.parametrize(
    "seed",
    [
        *contract.OLD_V1_GEOMETRY_SEEDS,
        *contract.OLD_V1_MEASUREMENT_SEEDS,
        contract.OLD_V1_BOOTSTRAP_SEED,
    ],
)
def test_old_v1_seed_is_rejected_by_v2_firewall(seed: int) -> None:
    firewall = ConfirmatoryV2SeedFirewall(
        SimpleNamespace(section=lambda _name: {"global_scene_seed": 0})
    )
    with pytest.raises(PermissionError, match="retired v1"):
        firewall.assert_access(seed, seed, 0)
    assert firewall.report()["NEW_V2_RNG_INSTANTIATION_COUNT"] == 0


def test_v2_plan_is_exact_595_1190() -> None:
    report = contract.audit_v2_plan(
        ROOT / contract.SNAPSHOT_PLAN_RELATIVE,
        ROOT / contract.TRIAL_PLAN_RELATIVE,
    )
    assert report["CONFIRMATORY_PLAN_COUNT_PASS"] is True
    assert report["CONFIRMATORY_PLAN_UNIQUENESS_PASS"] is True
    assert report["CONFIRMATORY_PLAN_PAIRING_PASS"] is True
    assert report["CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS"] is True
    assert report["planned_snapshot_count"] == report["planned_snapshot_unique_count"] == 595
    assert report["planned_trial_count"] == report["planned_trial_unique_count"] == 1190
    assert report["native_trial_count"] == 0


def test_v2_plan_condition_and_backend_counts_are_exact() -> None:
    report = contract.audit_v2_plan(
        ROOT / contract.SNAPSHOT_PLAN_RELATIVE,
        ROOT / contract.TRIAL_PLAN_RELATIVE,
    )
    assert report["condition_snapshot_counts"] == {
        "FULL_NOISE": 525,
        "IDEAL_MATCHED": 35,
        "INDEPENDENT_NOISE_FREE": 35,
    }
    assert report["backend_trial_counts"] == {
        "open3d_point_to_plane": 595,
        "pcl_point_to_plane": 595,
    }


def test_gate_h1_h6_payload_is_byte_semantically_unchanged() -> None:
    v1 = json.loads(
        (ROOT / "protocols/synthetic_confirmatory_gate_contract.json").read_text()
    )
    v2 = json.loads((ROOT / contract.GATE_RELATIVE).read_text())
    assert v2["hypotheses"] == v1["hypotheses"]
    assert v2["hypothesis_count"] == v1["hypothesis_count"] == 6


def test_frozen_model_and_backend_implementations_are_unchanged() -> None:
    assert file_sha256(ROOT / contract.FROZEN_MODEL_RELATIVE) == (
        "99806f83d3a0d2393ee7e36e8c06f14a1a8a44fb8d02b074ebc2d9fe750d0872"
    )
    assert file_sha256(ROOT / contract.PCL_CLI_RELATIVE) == (
        "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff"
    )
    _require_historical_commit()
    baseline = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:src/phase_a_harness/open3d_backend.py"],
        cwd=ROOT,
    )
    assert file_sha256(ROOT / "src/phase_a_harness/open3d_backend.py") == (
        hashlib.sha256(baseline).hexdigest()
    )


def test_primary_and_independent_h1_h6_core_ast_is_unchanged() -> None:
    assert _ast_sha(
        ROOT / "src/phase_a_harness/synthetic_confirmatory_analysis.py",
        "analyze_synthetic_confirmatory_records",
    ) == "cff961f392b35694cb7d29c7310960d89ed0fed8636d0cc52cea8ad753167485"
    assert _ast_sha(
        ROOT / "src/phase_a_harness/synthetic_confirmatory_independent_verifier.py",
        "independently_recompute_synthetic_confirmatory",
    ) == "2c91d28d22ea812e544bcf82cb9421a62e0b758a1d25ea8f0e805800ab657d54"


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        ("canonical_target", "5684f5bbcbd9001be0b29fcdd670470d0e19d7758d698fc39aaba4b73be3a5c8"),
        ("eligible_parent_indices", "f8fd5aed7876e30cacfac41c1b699518d04e7c134f23d189c8f217f9af5e275a"),
        ("quantization_closure", "2ec95cab5329881772ccd56518b73705efded616254710d013db95324594ebac"),
        ("reference_pose_from_development", "c9f8faa358bc09ee6e226547ec936f9bb2e95e7f98cac662b3a10da8b608caa1"),
        ("source_from_parent_indices", "787fc81f8a0a00dab2e0fa9fe761e7a7b61d65f7b14305cc32627bb84265fa94"),
    ],
)
def test_phase_a_lineage_and_closure_core_ast_is_unchanged(
    function: str, expected: str
) -> None:
    assert _ast_sha(
        ROOT
        / "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_v1_2.py",
        function,
    ) == expected


def test_development_ideal_uses_parent_lineage(ideal_snapshot: dict[str, object]) -> None:
    metadata = ideal_snapshot["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["source_has_target_parent_lineage"] is True
    assert metadata["source_is_target_subset"] is True
    assert metadata["lineage_validation_method"] == (
        "PARENT_INDEX_ROW_CORRESPONDENCE_PLUS_PHASE_A_QUANTIZATION_CLOSURE"
    )


def test_development_ideal_never_reads_measurement_or_repeat_rng(
    ideal_snapshot: dict[str, object],
) -> None:
    assert ideal_snapshot["firewall_audit"] == {
        "geometry_access_count": 1,
        "measurement_seed_access_count": 0,
        "repeat_randomness_count": 0,
        "rng_instantiation_count": 0,
    }


def test_parent_index_array_contract(ideal_snapshot: dict[str, object]) -> None:
    source = ideal_snapshot["source"]
    target = ideal_snapshot["target"]
    indices = ideal_snapshot["parent_indices"]
    assert isinstance(source, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert source.dtype == target.dtype == np.dtype("<f4")
    assert indices.dtype == np.dtype("<i8")
    assert source.flags.c_contiguous and target.flags.c_contiguous and indices.flags.c_contiguous
    assert indices.ndim == 1 and len(indices) == len(source)
    assert len(np.unique(indices)) == len(indices)
    assert np.count_nonzero((indices < 0) | (indices >= len(target))) == 0
    assert hashlib.sha256(indices.tobytes(order="C")).hexdigest() == (
        ideal_snapshot["metadata"]["source_parent_target_indices_sha256"]
    )


def test_parent_index_lineage_has_exact_44214_rows(
    ideal_snapshot: dict[str, object],
) -> None:
    assert len(ideal_snapshot["parent_indices"]) == 44214


def test_independent_recomputation_matches_builder_closure(
    ideal_snapshot: dict[str, object],
) -> None:
    result = independently_recompute_v2_lineage(
        ideal_snapshot["source"],
        ideal_snapshot["target"],
        ideal_snapshot["reference"],
        ideal_snapshot["parent_indices"],
    )
    metadata = ideal_snapshot["metadata"]
    for name in (
        "reconstruction_error_median_m",
        "reconstruction_error_q95_m",
        "reconstruction_error_max_m",
        "predicted_quantization_median_m",
        "predicted_quantization_q95_m",
        "predicted_quantization_max_m",
        "closure_residual_max_m",
        "float64_guard_max_m",
        "max_normalized_closure_ratio",
        "closure_residual_violation_count",
        "actual_error_bound_violation_count",
        "quantization_closure_pass",
    ):
        assert result[name] == metadata[name]
    assert result["row_correspondence_pass"] is True
    assert result["lineage_closure_violation_count"] == 0


def test_independent_verifier_does_not_trust_builder_lineage_booleans(
    ideal_snapshot: dict[str, object],
) -> None:
    claimed = dict(ideal_snapshot["metadata"])
    claimed["source_has_target_parent_lineage"] = False
    claimed["source_is_target_subset"] = False
    recomputed = independently_recompute_v2_lineage(
        ideal_snapshot["source"],
        ideal_snapshot["target"],
        ideal_snapshot["reference"],
        ideal_snapshot["parent_indices"],
    )
    expected_lineage = bool(
        recomputed["row_correspondence_pass"]
        and recomputed["quantization_closure_pass"]
        and recomputed["lineage_closure_violation_count"] == 0
    )
    assert expected_lineage is True
    assert claimed["source_has_target_parent_lineage"] is not expected_lineage
    assert claimed["source_is_target_subset"] is not expected_lineage


def test_float32_roundtrip_difference_does_not_invalidate_lineage(
    ideal_snapshot: dict[str, object],
) -> None:
    source = ideal_snapshot["source"]
    target = ideal_snapshot["target"]
    reference = ideal_snapshot["reference"]
    indices = ideal_snapshot["parent_indices"]
    transformed = np.ascontiguousarray(
        source.astype(np.float64) @ reference[:3, :3].T + reference[:3, 3],
        dtype="<f4",
    )
    assert np.count_nonzero(np.any(transformed != target[indices], axis=1)) > 0
    assert ideal_snapshot["metadata"]["lineage_closure_violation_count"] == 0
    assert ideal_snapshot["metadata"]["source_is_target_subset"] is True


def test_v2_builder_contains_no_coordinate_byte_membership() -> None:
    source = (
        ROOT / "src/phase_a_harness/synthetic_confirmatory_v2_snapshot_builder.py"
    ).read_text(encoding="utf-8")
    metadata_source = inspect.getsource(v2_builder._metadata)
    assert "target_rows" not in source
    assert "point.tobytes()" not in source
    assert " in target_rows" not in source
    assert '"source_has_target_parent_lineage": True' not in metadata_source
    assert '"source_is_target_subset": True' not in metadata_source


def test_metadata_cannot_hardcode_a_tampered_lineage_true(
    ideal_snapshot: dict[str, object],
) -> None:
    source = ideal_snapshot["source"]
    target = ideal_snapshot["target"]
    reference = ideal_snapshot["reference"]
    indices = np.array(ideal_snapshot["parent_indices"], copy=True, order="C")
    indices[1] = indices[0]
    with pytest.raises(ValueError, match="cannot claim invalid parent lineage"):
        v2_builder._metadata(
            snapshot_id="development-only/tampered-lineage",
            scene="GEOMETRY_RICH_ROOM",
            condition="IDEAL_MATCHED",
            geometry_seed=DEVELOPMENT_GEOMETRY_SEEDS[0],
            measurement_seed=None,
            repeat_index=0,
            source=source,
            target=target,
            reference=reference,
            rng_count=0,
            lineage={
                "closure": {"quantization_closure_pass": True},
                "indices": indices,
                "lineage_closure_violation_count": 0,
                "parent_points": target[indices].astype("<f8"),
            },
        )


@pytest.mark.parametrize("mutation", ["duplicate", "out_of_range", "reorder_source", "reference"])
def test_independent_verifier_detects_lineage_tampering(
    ideal_snapshot: dict[str, object], mutation: str
) -> None:
    source = np.array(ideal_snapshot["source"], copy=True, order="C")
    target = np.array(ideal_snapshot["target"], copy=True, order="C")
    reference = np.array(ideal_snapshot["reference"], copy=True, order="C")
    indices = np.array(ideal_snapshot["parent_indices"], copy=True, order="C")
    if mutation == "duplicate":
        indices[1] = indices[0]
    elif mutation == "out_of_range":
        indices[0] = len(target)
    elif mutation == "reorder_source":
        source[[0, 1]] = source[[1, 0]]
    else:
        reference[0, 3] += 1.0e-3
    result = independently_recompute_v2_lineage(source, target, reference, indices)
    assert result["quantization_closure_pass"] is False
    assert result["row_correspondence_pass"] is False


def test_independent_negative_control_has_no_lineage(
    independent_snapshot: dict[str, object],
) -> None:
    metadata = independent_snapshot["metadata"]
    assert independent_snapshot["parent_indices"] is None
    assert metadata["source_has_target_parent_lineage"] is False
    assert metadata["source_is_target_subset"] is False
    assert metadata["source_parent_target_indices_path"] is None
    assert metadata["source_parent_target_indices_sha256"] is None
    assert metadata["quantization_closure_pass"] is False
    assert independent_snapshot["firewall_audit"] == {
        "geometry_access_count": 3,
        "measurement_seed_access_count": 0,
        "repeat_randomness_count": 0,
        "rng_instantiation_count": 0,
    }


def test_full_noise_cannot_claim_ideal_lineage(
    full_noise_regression_pair: tuple[dict[str, object], dict[str, object]],
) -> None:
    _baseline, candidate = full_noise_regression_pair
    metadata = candidate["metadata"]
    assert metadata["condition"] == "FULL_NOISE"
    assert candidate["parent_indices"] is None
    assert metadata["source_has_target_parent_lineage"] is False
    assert metadata["source_is_target_subset"] is False
    assert metadata["source_parent_target_indices_sha256"] is None


def test_development_nonideal_scientific_payload_is_byte_exact(
    full_noise_regression_pair: tuple[dict[str, object], dict[str, object]],
) -> None:
    baseline, candidate = full_noise_regression_pair
    for name in ("source", "target", "reference"):
        assert np.array_equal(candidate[name], baseline[name])
    for name in (
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
    ):
        assert candidate["metadata"][name] == baseline["metadata"][name]


def test_v1_and_v2_manifests_are_explicitly_version_selected() -> None:
    _require_historical_commit()
    v1_path = ROOT / "frozen_assets/synthetic_confirmatory_formal_manifest_v1.json"
    baseline = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:{v1_path.relative_to(ROOT).as_posix()}"],
        cwd=ROOT,
    )
    assert v1_path.read_bytes() == baseline
    with pytest.raises(ValueError, match="exact v2 manifest path"):
        contract.verify_manifest(
            v1_path,
            require_authorized=False,
        )
    with pytest.raises(ValueError, match="one v1 manifest"):
        load_v1_stack(ROOT / contract.MANIFEST_RELATIVE, require_authorized=False)


def test_v2_manifest_is_exact_and_initially_or_finally_authorized() -> None:
    _root, manifest = contract.verify_manifest(
        ROOT / contract.MANIFEST_RELATIVE, require_authorized=False
    )
    assert manifest["manifest_schema"] == contract.MANIFEST_SCHEMA
    assert type(manifest["formal_execution_authorized"]) is bool
    assert manifest["planned_snapshot_count"] == 595
    assert manifest["planned_trial_count"] == 1190


def test_v2_manifest_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = json.loads((ROOT / contract.MANIFEST_RELATIVE).read_text())
    repository = tmp_path / "repository"
    path = repository / contract.MANIFEST_RELATIVE
    path.parent.mkdir(parents=True)
    tampered = dict(live)
    tampered["formal_workers"] = 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(contract, "signed_manifest", lambda *_args, **_kwargs: live)
    with pytest.raises(ValueError, match="exact live bindings"):
        contract.verify_manifest(path, require_authorized=False)


def test_v2_dry_run_has_zero_execution_and_no_runtime_state() -> None:
    manifest = json.loads((ROOT / contract.MANIFEST_RELATIVE).read_text())
    if manifest["formal_execution_authorized"] is True:
        # The one-way pre-run transition has already occurred and the partial
        # v2 execution is now permanently retired.  Re-running even its old
        # pre-authorization dry-run entry point must fail closed without
        # recreating runtime state.
        with pytest.raises(
            PermissionError, match="dry-run requires authorization=false"
        ):
            dry_run_synthetic_confirmatory(
                manifest_path=ROOT / contract.MANIFEST_RELATIVE,
                run_id=contract.FORMAL_RUN_ID,
                output_dir=ROOT / contract.FORMAL_OUTPUT_DIR,
                workers=contract.FORMAL_WORKERS,
            )
        assert not (ROOT / contract.FORMAL_OUTPUT_DIR).exists()
        assert not (ROOT / contract.SNAPSHOT_CACHE_ROOT).exists()
        assert not (ROOT / contract.SNAPSHOT_LOCK_RELATIVE).exists()
        return
    report = dry_run_synthetic_confirmatory(
        manifest_path=ROOT / contract.MANIFEST_RELATIVE,
        run_id=contract.FORMAL_RUN_ID,
        output_dir=ROOT / contract.FORMAL_OUTPUT_DIR,
        workers=contract.FORMAL_WORKERS,
    )
    assert report["CONFIRMATORY_DRY_RUN_PASS"] is True
    assert report["planned_snapshot_count"] == report["planned_snapshot_unique_count"] == 595
    assert report["planned_trial_count"] == report["planned_trial_unique_count"] == 1190
    for name in (
        "NEW_V2_RNG_INSTANTIATION_COUNT",
        "NEW_V2_SNAPSHOT_CONSTRUCTION_COUNT",
        "NEW_V2_BACKEND_EXECUTION_COUNT",
        "NEW_V2_TRIAL_RESULT_COUNT",
        "NEW_V2_STARTED_EVENT_COUNT",
        "NATIVE_EXECUTION_COUNT",
    ):
        assert report[name] == 0
    assert report["output_dir_created"] is False
    assert report["snapshot_cache_dir_created"] is False
    assert report["snapshot_lock_created"] is False


@pytest.mark.parametrize(
    "script_name",
    ["analyze_synthetic_confirmatory_v2.py", "verify_synthetic_confirmatory_v2.py"],
)
def test_v2_read_only_clis_enforce_the_full_runtime_isolation_contract(
    script_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv(
        "MAMBA_ROOT_PREFIX", "/home/lj/.local/share/degen-lio-micromamba"
    )
    monkeypatch.delenv("PYTHONPATH", raising=False)
    module._assert_preimport_isolation()

    monkeypatch.setenv("MAMBA_ROOT_PREFIX", "/tmp/not-the-frozen-prefix")
    with pytest.raises(PermissionError, match="frozen MAMBA_ROOT_PREFIX"):
        module._assert_preimport_isolation()

    monkeypatch.setenv(
        "MAMBA_ROOT_PREFIX", "/home/lj/.local/share/degen-lio-micromamba"
    )
    monkeypatch.setenv("PYTHONPATH", "/home/lj/Degen-LIO")
    with pytest.raises(PermissionError, match="source repository"):
        module._assert_preimport_isolation()


def test_v2_qualification_cli_enforces_the_frozen_micromamba_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = ROOT / "scripts/qualify_synthetic_confirmatory_v2.py"
    spec = importlib.util.spec_from_file_location("test_v2_qualification_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv(
        "MAMBA_ROOT_PREFIX", "/home/lj/.local/share/degen-lio-micromamba"
    )
    monkeypatch.delenv("PYTHONPATH", raising=False)
    module._assert_runtime_isolation(ROOT)

    monkeypatch.setenv("MAMBA_ROOT_PREFIX", "/tmp/not-the-frozen-prefix")
    with pytest.raises(PermissionError, match="frozen MAMBA_ROOT_PREFIX"):
        module._assert_runtime_isolation(ROOT)


def test_v2_prerun_freezer_cli_exposes_the_required_dry_run_evidence() -> None:
    path = ROOT / "scripts/freeze_synthetic_confirmatory_v2_prerun.py"
    spec = importlib.util.spec_from_file_location("test_v2_prerun_freezer_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destinations = {action.dest for action in module.build_parser()._actions}
    assert "dry_run_report" in destinations
    freezer_source = path.read_text(encoding="utf-8")
    verifier_source = (
        ROOT
        / "src/phase_a_harness/synthetic_confirmatory_v2_artifact_verifier.py"
    ).read_text(encoding="utf-8")
    assert '"formal_run_id": FORMAL_RUN_ID' in freezer_source
    assert 'run.get("formal_run_id") == FORMAL_RUN_ID' in verifier_source


def test_direct_true_manifest_write_is_forbidden() -> None:
    with pytest.raises(PermissionError):
        contract.write_manifest(ROOT, authorized=True, replace=True)


def test_prerun_gate_fails_closed() -> None:
    gates = {name: True for name in REQUIRED_TRUE_GATES}
    gates.update(
        ROOT_CAUSE="FLOAT_QUANTIZATION_BYTEWISE_COMPARISON",
        OLD_V1_SEED_SET_REUSE_AUTHORIZED=False,
        NEW_V2_NAMESPACE_COLLISION=False,
        IDEAL_DEVELOPMENT_SNAPSHOT_COUNT=21,
        INDEPENDENT_NEGATIVE_CONTROL_COUNT=21,
        INDEPENDENT_FALSE_LINEAGE_COUNT=21,
    )
    counters = {name: 0 for name in REQUIRED_ZERO_COUNTERS}
    bad = dict(counters)
    bad["PRIMARY_VERIFIER_DIFFERENCE_COUNT"] = 1
    decision = build_v2_prerun_decision(gates=gates, counters=bad, authorize=False)
    assert decision["SYNTHETIC_CONFIRMATORY_V2_PRE_RUN_QUALIFICATION_PASS"] is False
    with pytest.raises(PermissionError):
        build_v2_prerun_decision(gates=gates, counters=bad, authorize=True)


def test_prerun_gate_keeps_science_not_evaluated() -> None:
    gates = {name: True for name in REQUIRED_TRUE_GATES}
    gates.update(
        ROOT_CAUSE="FLOAT_QUANTIZATION_BYTEWISE_COMPARISON",
        OLD_V1_SEED_SET_REUSE_AUTHORIZED=False,
        NEW_V2_NAMESPACE_COLLISION=False,
        IDEAL_DEVELOPMENT_SNAPSHOT_COUNT=21,
        INDEPENDENT_NEGATIVE_CONTROL_COUNT=21,
        INDEPENDENT_FALSE_LINEAGE_COUNT=21,
    )
    decision = build_v2_prerun_decision(
        gates=gates,
        counters={name: 0 for name in REQUIRED_ZERO_COUNTERS},
        authorize=True,
    )
    assert decision["SYNTHETIC_CONFIRMATORY_V2_PRE_RUN_QUALIFICATION_PASS"] is True
    assert decision["CONFIRMATORY_V2_RUN_AUTHORIZED"] is True
    assert decision["SYNTHETIC_CONFIRMATORY_V2_EXECUTED"] is False
    assert decision["SYNTHETIC_CONFIRMATORY_V2_PASS"] == "NOT_EVALUATED"
    assert decision["REAL_DATA_RUN_AUTHORIZED"] is False
    assert decision["MEASUREMENT_PAPER_MAINLINE_AUTHORIZED"] is False


def test_prerun_root_inventory_is_exact_26() -> None:
    assert len(PRERUN_ROOT_FILES) == 26
    assert len(set(PRERUN_ROOT_FILES)) == 26
    assert set(PRERUN_ROOT_FILES) >= {
        "final_decision.json",
        "run_manifest.json",
        "pre_run_report.md",
        "MANIFEST.csv",
        "SHA256SUMS",
        "artifact_verification.json",
    }


def test_formal_v2_runtime_paths_remain_absent_during_qualification() -> None:
    assert not (ROOT / contract.SNAPSHOT_CACHE_ROOT).exists()
    assert not (ROOT / contract.SNAPSHOT_LOCK_RELATIVE).exists()
    assert not (ROOT / contract.FORMAL_OUTPUT_DIR).exists()


def test_snapshot_metadata_schema_is_explicit_v2(
    ideal_snapshot: dict[str, object], independent_snapshot: dict[str, object]
) -> None:
    for value in (ideal_snapshot, independent_snapshot):
        metadata = value["metadata"]
        assert metadata["schema_version"] == contract.METADATA_SCHEMA
        assert metadata["snapshot_schema_version"] == contract.SNAPSHOT_SCHEMA
        assert metadata["lineage_schema_version"] == contract.LINEAGE_SCHEMA


def _fixture_rows() -> list[dict[str, object]]:
    rows = []
    for condition, classification, failure in (
        ("FIXTURE_IDENTITY", "NONE", False),
        ("FIXTURE_NONIDENTITY_REFERENCE", "NONE", False),
        ("FIXTURE_NO_CORRESPONDENCE", "NO_CORRESPONDENCES", True),
    ):
        snapshot_id = f"seed-free/{condition}"
        checksums = {
            name: hashlib.sha256(f"{condition}/{name}".encode()).hexdigest()
            for name in (
                "snapshot_checksum",
                "source_checksum",
                "target_checksum",
                "reference_pose_checksum",
            )
        }
        for backend in contract.BACKENDS:
            rows.append(
                {
                    "backend": backend,
                    "condition": condition,
                    "failure_classification": classification,
                    "finite_output": not failure,
                    "planned_trial_id": f"{snapshot_id}/{backend}",
                    "runtime_ms": 1.0,
                    "snapshot_id": snapshot_id,
                    "solver_failure": failure,
                    "translation_update_m": None if failure else 0.0,
                    **checksums,
                }
            )
    return rows


def test_fixture_primary_independent_publisher_and_artifact_chain(tmp_path: Path) -> None:
    rows = _fixture_rows()
    primary = analyze_v2_fixture_results(rows)
    independent = independently_analyze_v2_fixture_results(rows)
    difference = compare_v2_fixture_primary_and_independent(primary, independent)
    assert difference["exact_match_pass"] is True
    run = {
        "backend_execution_count": 6,
        "fixture_snapshot_count": 3,
        "fixture_trial_count": 6,
        "formal_confirmatory_science_evaluated": False,
        "formal_v2_seed_reference_count": 0,
        "fresh_resume_scientific_equivalence": True,
        "resume_backend_execution_count": 0,
        "schema_version": "synthetic_confirmatory_v2_fixture_run_v1",
    }
    artifact = tmp_path / "fixture"
    result = publish_synthetic_confirmatory_v2_fixture(
        primary=primary,
        independent=independent,
        run_manifest=run,
        artifact_dir=artifact,
    )
    assert result["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is True
    assert result["published_file_count"] == 17
    live = verify_synthetic_confirmatory_v2_fixture_artifact(
        artifact, write_report=False
    )
    assert live["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is True
    assert live["actual_file_count"] == 17


@pytest.mark.parametrize("damage", ["missing", "extra"])
def test_fixture_artifact_verifier_rejects_missing_or_extra(
    tmp_path: Path, damage: str
) -> None:
    rows = _fixture_rows()
    primary = analyze_v2_fixture_results(rows)
    independent = independently_analyze_v2_fixture_results(rows)
    artifact = tmp_path / "fixture"
    publish_synthetic_confirmatory_v2_fixture(
        primary=primary,
        independent=independent,
        run_manifest={
            "backend_execution_count": 6,
            "fixture_snapshot_count": 3,
            "fixture_trial_count": 6,
            "formal_confirmatory_science_evaluated": False,
            "formal_v2_seed_reference_count": 0,
            "fresh_resume_scientific_equivalence": True,
            "resume_backend_execution_count": 0,
            "schema_version": "synthetic_confirmatory_v2_fixture_run_v1",
        },
        artifact_dir=artifact,
    )
    if damage == "missing":
        (artifact / "tables/fixture_trial_inventory.csv").unlink()
    else:
        (artifact / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    report = verify_synthetic_confirmatory_v2_fixture_artifact(
        artifact, write_report=False
    )
    assert report["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is False


def test_v1_fixture_artifact_cannot_masquerade_as_v2() -> None:
    report = verify_synthetic_confirmatory_v2_fixture_artifact(
        ROOT / "artifacts/synthetic_confirmatory_prerun_v1/fixture_publication",
        write_report=False,
    )
    assert report["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is False
