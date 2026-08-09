from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from phase_a_harness.contracts import canonical_json_sha256, file_sha256
from phase_a_harness.fixture_publication import (
    audit_and_publish_existing_fixture_results,
)
from phase_a_harness.fixture_publication_artifact_verifier import (
    verify_fixture_publication_artifact,
)
from phase_a_harness.phase_a_trial_result_schema import canonical_json_bytes
from phase_a_harness.phase_a_trial_result_writer import result_filename
from phase_a_harness.synthetic_confirmatory_artifact_verifier import (
    PRERUN_ROOT_FILES,
    PRERUN_REQUIRED_GATE_NAMES,
    PRERUN_ZERO_EXECUTION_COUNTER_NAMES,
)
from phase_a_harness.synthetic_confirmatory_manifest import (
    FILE_BINDINGS,
    MANIFEST_RELATIVE,
    authorize_synthetic_confirmatory_manifest_once,
    signed_synthetic_confirmatory_manifest,
    verify_synthetic_confirmatory_manifest,
)
from phase_a_harness.synthetic_confirmatory_prerun import (
    REQUIRED_GATE_NAMES,
    ZERO_EXECUTION_COUNTER_NAMES,
    build_prerun_qualification_decision,
)
from phase_a_harness.synthetic_confirmatory_protocol import audit_confirmatory_plan
from phase_a_harness import synthetic_confirmatory_runner as runner
from phase_a_harness import synthetic_confirmatory_snapshot_builder as snapshots


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_binds_all_formal_transitive_local_dependencies() -> None:
    required = {
        "asset_verifier",
        "backend_metrics",
        "backend_phase_a_metrics",
        "backend_types",
        "confirmatory_protocol_builder",
        "core_contracts",
        "full_synthetic_backend_execution",
        "full_synthetic_development_protocol",
        "full_synthetic_snapshot_builder",
        "full_synthetic_trial_result",
        "fixture_backend_parameter_lock",
        "fixture_independent_verifier",
        "fixture_plan",
        "fixture_primary_analysis",
        "fixture_publication",
        "fixture_publication_artifact_verifier",
        "fixture_qualification",
        "fixture_snapshot_lock",
        "generator_development_protocol",
        "generator_frozen_capture_protocol",
        "generator_frozen_capture_types",
        "generator_frozen_phase_a_protocol",
        "generator_frozen_phase_a_v1_2",
        "generator_frozen_zero_protocol",
        "generator_frozen_zero_snapshot_builder",
        "generator_frozen_zero_types",
        "gitignore_runtime_outputs",
        "local_metric_models",
        "phase_a_attempt_events",
        "phase_a_execution_chain",
        "phase_a_execution_fixture",
        "phase_a_trial_result_schema",
        "rotation_metrics",
        "scientific_survival_models",
    }
    assert required <= set(FILE_BINDINGS)
    assert all((ROOT / relative).is_file() for relative in FILE_BINDINGS.values())
    assert not any(
        relative.startswith(("results/", "data/", "artifacts/"))
        for relative in FILE_BINDINGS.values()
    )


def test_independent_manifest_binding_literal_exactly_mirrors_primary() -> None:
    independent_path = (
        ROOT
        / "src/phase_a_harness/synthetic_confirmatory_independent_verifier.py"
    )
    tree = ast.parse(independent_path.read_text(encoding="utf-8"))
    literal = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "_INDEPENDENT_MANIFEST_BINDINGS"
            for target in node.targets
        ):
            literal = ast.literal_eval(node.value)
            break
    assert literal is not None
    assert literal == dict(FILE_BINDINGS)


def _fake_stack_root(tmp_path: Path, *, authorized: bool = False) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "harness"
    (root / "frozen_assets").mkdir(parents=True)
    (root / "protocols").mkdir()
    (root / "bin").mkdir()
    shutil.copyfile(
        ROOT / "protocols/synthetic_confirmatory_planned_snapshots.csv",
        root / "protocols/synthetic_confirmatory_planned_snapshots.csv",
    )
    shutil.copyfile(
        ROOT / "protocols/synthetic_confirmatory_planned_trials.csv",
        root / "protocols/synthetic_confirmatory_planned_trials.csv",
    )
    shutil.copyfile(
        ROOT / "frozen_assets/backend_parameter_contract.json",
        root / "frozen_assets/backend_parameter_contract.json",
    )
    (root / "bin/pcl_point_to_plane_cli").write_bytes(b"fixture-only-placeholder")
    parameters = json.loads(
        (root / "frozen_assets/backend_parameter_contract.json").read_text(
            encoding="utf-8"
        )
    )
    manifest: dict[str, Any] = {
        "bound_files": {
            "backend_parameter_contract": {
                "path": "frozen_assets/backend_parameter_contract.json"
            },
            "pcl_cli": {"path": "bin/pcl_point_to_plane_cli"},
        },
        "formal_execution_authorized": authorized,
        "formal_output_dir": "results/synthetic_confirmatory_v1",
        "formal_run_id": "synthetic-confirmatory-v1",
        "formal_workers": 2,
        "manifest_payload_sha256": "a" * 64,
        "open3d_parameter_sha256": canonical_json_sha256(
            parameters["open3d"]["parameters"]
        ),
        "pcl_parameter_sha256": canonical_json_sha256(
            parameters["pcl"]["parameters"]
        ),
        "planned_snapshots_path": (
            "protocols/synthetic_confirmatory_planned_snapshots.csv"
        ),
        "planned_trials_path": "protocols/synthetic_confirmatory_planned_trials.csv",
        "scientific_protocol_sha256": "b" * 64,
        "snapshot_cache_root": "data/synthetic_confirmatory_v1_snapshots",
    }
    (root / MANIFEST_RELATIVE).write_text("{}\n", encoding="utf-8")
    return root, manifest


def _install_stack_mocks(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    manifest: dict[str, Any],
) -> None:
    from phase_a_harness import synthetic_confirmatory_protocol as protocol

    plan = audit_confirmatory_plan(
        root / "protocols/synthetic_confirmatory_planned_snapshots.csv",
        root / "protocols/synthetic_confirmatory_planned_trials.csv",
    )

    def verify(fake_root: Path, *, require_authorized: bool) -> dict[str, Any]:
        assert Path(fake_root).resolve() == root.resolve()
        if require_authorized and manifest["formal_execution_authorized"] is not True:
            raise PermissionError("formal execution is not authorized")
        return manifest

    def protocol_audit(fake_root: Path) -> dict[str, Any]:
        assert Path(fake_root).resolve() == root.resolve()
        return {
            "CONFIRMATORY_PROTOCOL_BINDING_PASS": True,
            "SCIENTIFIC_SURVIVAL_BINDING_PASS": True,
            "SYNTHETIC_CONFIRMATORY_PROTOCOL_CONTRACT_AUDIT_PASS": True,
            "gate_contract_audit": {"CONFIRMATORY_GATE_CONTRACT_PASS": True},
            "plan_audit": plan,
            "seed_provenance_audit": {
                "CONFIRMATORY_RNG_INSTANTIATION_COUNT": 0,
                "CONFIRMATORY_SEED_INSTANTIATION_COUNT": 0,
                "CONFIRMATORY_SEED_PARSE_ERROR_COUNT": 0,
                "CONFIRMATORY_SEED_PROVENANCE_PASS": True,
                "CONFIRMATORY_SEED_USAGE_HIT_COUNT": 0,
            },
        }

    monkeypatch.setattr(runner, "verify_synthetic_confirmatory_manifest", verify)
    monkeypatch.setattr(protocol, "audit_confirmatory_protocol_contract", protocol_audit)


@pytest.fixture
def dry_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    root, manifest = _fake_stack_root(tmp_path)
    _install_stack_mocks(monkeypatch, root, manifest)
    return root, manifest


def test_loader_reads_exact_595_1190_design(
    dry_stack: tuple[Path, dict[str, Any]],
) -> None:
    root, _ = dry_stack
    stack = runner.load_synthetic_confirmatory_stack(
        root / MANIFEST_RELATIVE, require_authorized=False
    )
    assert len(stack["snapshots"]) == 595
    assert len(stack["trials"]) == 1190
    assert stack["plan_audit"]["CONFIRMATORY_PLAN_COUNT_PASS"] is True
    assert stack["plan_audit"]["CONFIRMATORY_PLAN_PAIRING_PASS"] is True


def test_dry_run_has_exact_counts_and_zero_execution(
    dry_stack: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = dry_stack

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("dry-run crossed the execution boundary")

    monkeypatch.setattr(runner, "_execute_one", forbidden)
    monkeypatch.setattr(
        snapshots, "build_synthetic_confirmatory_snapshot", forbidden
    )
    report = runner.dry_run_synthetic_confirmatory(
        manifest_path=root / MANIFEST_RELATIVE,
        run_id="synthetic-confirmatory-v1",
        output_dir=root / "results/synthetic_confirmatory_v1",
        workers=2,
    )
    assert report["CONFIRMATORY_DRY_RUN_PASS"] is True
    assert report["planned_snapshot_count"] == 595
    assert report["planned_trial_count"] == 1190
    assert report["condition_snapshot_counts"] == {
        "IDEAL_MATCHED": 35,
        "INDEPENDENT_NOISE_FREE": 35,
        "FULL_NOISE": 525,
    }
    assert report["open3d_trial_count"] == report["pcl_trial_count"] == 595
    assert report["native_trial_count"] == 0
    assert report["CONFIRMATORY_RNG_INSTANTIATION_COUNT"] == 0
    assert report["CONFIRMATORY_SNAPSHOT_GENERATION_COUNT"] == 0
    assert report["CONFIRMATORY_BACKEND_EXECUTION_COUNT"] == 0
    assert report["CONFIRMATORY_TRIAL_RESULT_COUNT"] == 0
    assert report["NATIVE_EXECUTION_COUNT"] == 0
    assert report["attempt_started_event_count"] == 0


def test_dry_run_does_not_create_output_directory(
    dry_stack: tuple[Path, dict[str, Any]],
) -> None:
    root, _ = dry_stack
    output = root / "results/synthetic_confirmatory_v1"
    report = runner.dry_run_synthetic_confirmatory(
        manifest_path=root / MANIFEST_RELATIVE,
        run_id="synthetic-confirmatory-v1",
        output_dir=output,
        workers=2,
    )
    assert report["output_dir_created"] is False
    assert not output.exists()


def test_dry_run_rejects_already_authorized_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = _fake_stack_root(tmp_path, authorized=True)
    _install_stack_mocks(monkeypatch, root, manifest)
    with pytest.raises(PermissionError, match="authorization=false"):
        runner.dry_run_synthetic_confirmatory(
            manifest_path=root / MANIFEST_RELATIVE,
            run_id="synthetic-confirmatory-v1",
            output_dir=root / "results/synthetic_confirmatory_v1",
            workers=2,
        )


@pytest.mark.parametrize(
    ("run_id", "output_relative", "workers"),
    [
        ("changed", "results/synthetic_confirmatory_v1", 2),
        ("synthetic-confirmatory-v1", "results/changed", 2),
        ("synthetic-confirmatory-v1", "results/synthetic_confirmatory_v1", 1),
    ],
)
def test_dry_run_rejects_invocation_drift(
    dry_stack: tuple[Path, dict[str, Any]],
    run_id: str,
    output_relative: str,
    workers: int,
) -> None:
    root, _ = dry_stack
    with pytest.raises(ValueError, match="invocation differs"):
        runner.dry_run_synthetic_confirmatory(
            manifest_path=root / MANIFEST_RELATIVE,
            run_id=run_id,
            output_dir=root / output_relative,
            workers=workers,
        )


def test_dry_run_rejects_preexisting_output_directory(
    dry_stack: tuple[Path, dict[str, Any]],
) -> None:
    root, _ = dry_stack
    output = root / "results/synthetic_confirmatory_v1"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="absent output directory"):
        runner.dry_run_synthetic_confirmatory(
            manifest_path=root / MANIFEST_RELATIVE,
            run_id="synthetic-confirmatory-v1",
            output_dir=output,
            workers=2,
        )


def _manifest_fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "manifest-root"
    real_names = {
        "scientific_protocol",
        "scientific_protocol_document",
        "gate_contract",
        "planned_snapshots",
        "planned_trials",
        "seed_provenance_audit",
        "frozen_model",
        "backend_parameter_contract",
    }
    for name, relative in FILE_BINDINGS.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name in real_names:
            shutil.copyfile(ROOT / relative, destination)
        else:
            destination.write_text(f"fixture binding: {name}\n", encoding="utf-8")
    manifest = signed_synthetic_confirmatory_manifest(root, authorized=False)
    destination = root / MANIFEST_RELATIVE
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_manifest_verifier_accepts_exact_false_manifest(tmp_path: Path) -> None:
    root = _manifest_fixture_root(tmp_path)
    manifest = verify_synthetic_confirmatory_manifest(
        root, require_authorized=False
    )
    assert manifest["formal_execution_authorized"] is False
    with pytest.raises(PermissionError, match="not authorized"):
        verify_synthetic_confirmatory_manifest(root, require_authorized=True)


def test_manifest_payload_tamper_is_rejected(tmp_path: Path) -> None:
    root = _manifest_fixture_root(tmp_path)
    path = root / MANIFEST_RELATIVE
    value = json.loads(path.read_text(encoding="utf-8"))
    value["formal_run_id"] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from exact bindings"):
        verify_synthetic_confirmatory_manifest(root, require_authorized=False)


def test_manifest_bound_file_tamper_is_rejected(tmp_path: Path) -> None:
    root = _manifest_fixture_root(tmp_path)
    (root / FILE_BINDINGS["runner"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from exact bindings"):
        verify_synthetic_confirmatory_manifest(root, require_authorized=False)


def _valid_qualification_decision() -> dict[str, Any]:
    return build_prerun_qualification_decision(
        gates={name: True for name in REQUIRED_GATE_NAMES},
        counters={name: 0 for name in ZERO_EXECUTION_COUNTER_NAMES},
        confirmatory_run_authorized=True,
    )


def test_manifest_authorization_requires_live_prerun_artifact(tmp_path: Path) -> None:
    root = _manifest_fixture_root(tmp_path)
    decision = _valid_qualification_decision()
    with pytest.raises(PermissionError, match="artifact"):
        authorize_synthetic_confirmatory_manifest_once(
            root, qualification_decision=decision
        )
    assert verify_synthetic_confirmatory_manifest(
        root, require_authorized=False
    )["formal_execution_authorized"] is False


def test_manifest_authorization_rejects_gate_name_subset(tmp_path: Path) -> None:
    root = _manifest_fixture_root(tmp_path)
    decision = _valid_qualification_decision()
    decision["required_gate_names"] = ["CONFIRMATORY_DRY_RUN_PASS"]
    with pytest.raises(PermissionError, match="qualification is incomplete"):
        authorize_synthetic_confirmatory_manifest_once(
            root, qualification_decision=decision
        )
    assert verify_synthetic_confirmatory_manifest(
        root, require_authorized=False
    )["formal_execution_authorized"] is False


def test_noiseless_seed_sentinel_cannot_construct_rng() -> None:
    class Protocol:
        def section(self, name: str) -> dict[str, int]:
            assert name == "scene_generation"
            return {"global_scene_seed": 314159}

    firewall = snapshots.ConfirmatorySeedFirewall(Protocol())
    firewall.assert_access(
        snapshots.GEOMETRY_SEEDS[0], snapshots.NON_RNG_MEASUREMENT_SENTINEL, 0
    )
    assert firewall.rng_instantiation_count == 0
    assert snapshots.NON_RNG_MEASUREMENT_SENTINEL not in snapshots.MEASUREMENT_SEEDS
    assert snapshots.FULL_NOISE_RNG_STREAM_ROLES == (
        "scan_dropout",
        "scan_noise",
        "map_noise",
    )


def test_snapshot_builder_has_no_module_level_numpy_or_rng_import() -> None:
    tree = ast.parse(
        (ROOT / "src/phase_a_harness/synthetic_confirmatory_snapshot_builder.py").read_text(
            encoding="utf-8"
        )
    )
    top_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    names = {
        alias.name
        for node in top_imports
        for alias in node.names
    }
    assert "numpy" not in names
    assert not any(name.startswith("open3d") or name.startswith("pcl") for name in names)


def test_formal_frozen_protocol_audit_does_not_call_live_seed_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phase_a_harness import synthetic_confirmatory_protocol as protocol

    monkeypatch.setattr(
        protocol,
        "audit_confirmatory_protocol_contract",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("formal resume called the live seed-use scanner")
        ),
    )
    report = runner._audit_frozen_protocol_for_formal(
        ROOT,
        {
            "gate_contract_path": "protocols/synthetic_confirmatory_gate_contract.json",
            "planned_snapshots_path": (
                "protocols/synthetic_confirmatory_planned_snapshots.csv"
            ),
            "planned_trials_path": "protocols/synthetic_confirmatory_planned_trials.csv",
            "scientific_protocol_path": "protocols/synthetic_confirmatory_protocol_v1.json",
            "scientific_survival_commit": (
                "ffc15334f4ded25fdba5e709b45657dbad481dfc"
            ),
            "scientific_survival_tag": (
                "archive/zero-perturbation-scientific-survival-audit-v1"
            ),
            "seed_provenance_audit_path": (
                "protocols/confirmatory_seed_provenance_audit.json"
            ),
        },
    )
    assert report["SYNTHETIC_CONFIRMATORY_PROTOCOL_CONTRACT_AUDIT_PASS"] is True
    assert report["seed_provenance_audit"]["evidence_scope"] == (
        "FROZEN_PRE_RUN_PROVENANCE"
    )


def _seedless_resume_case(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    row = {
        "backend": "open3d_point_to_plane",
        "condition": "IDEAL_MATCHED",
        "planned_snapshot_id": "fixture/snapshot",
        "planned_trial_id": "fixture/snapshot::open3d_point_to_plane",
        "scene_variant": "FIXTURE_SCENE",
    }
    fixture = SimpleNamespace(
        checksums={
            "reference_pose_checksum": "1" * 64,
            "snapshot_checksum": "2" * 64,
            "source_checksum": "3" * 64,
            "target_checksum": "4" * 64,
        },
        condition="IDEAL_MATCHED",
        scene_variant="FIXTURE_SCENE",
        snapshot_id="fixture/snapshot",
    )
    stack = {
        "manifest": {
            "manifest_payload_sha256": "5" * 64,
            "scientific_protocol_sha256": "6" * 64,
        },
        "snapshot_lock_sha256": "7" * 64,
    }
    monkeypatch.setattr(runner, "_fixture", lambda unused_stack, unused_row: fixture)
    expected = runner._common(stack, fixture, row)
    payload = {
        **expected,
        "backend_diagnostics": {
            "correspondence_set_size": 10,
            "fitness": 1.0,
            "inlier_rmse": 0.0,
        },
        "failure_classification": "NONE",
        "failure_detail": None,
        "final_transform_4x4": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "finite_output": True,
        "orthogonality_defect_fro": 0.0,
        "projection_correction_fro": 0.0,
        "raw_rotation_determinant": 1.0,
        "raw_rotation_finite": True,
        "rotation_update_rad": 0.0,
        "runtime_ms": 0.0,
        "solver_failure": False,
        "translation_update_m": 0.0,
    }
    return stack, row, payload


def test_resume_fresh_inventory_has_no_orphan_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack, row, _ = _seedless_resume_case(monkeypatch)
    raw = runner._empty_raw_manifest("synthetic-confirmatory-v1")
    raw_path = tmp_path / "raw_result_manifest.json"
    report = runner._recover_atomic_result_orphans(
        stack,
        tmp_path / "raw_results",
        raw_path,
        raw,
        {row["planned_trial_id"]: row},
    )
    assert report["recovered_orphan_result_count"] == 0
    assert raw["results"] == {}
    assert not raw_path.exists()


def test_resume_partial_manifest_keeps_valid_entry_without_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack, row, payload = _seedless_resume_case(monkeypatch)
    results = tmp_path / "raw_results"
    results.mkdir()
    path = results / result_filename(row["planned_trial_id"])
    path.write_bytes(canonical_json_bytes(payload))
    entry = {
        "path": path.name,
        "planned_trial_id": row["planned_trial_id"],
        "sha256": file_sha256(path),
    }
    raw = runner._empty_raw_manifest("synthetic-confirmatory-v1")
    raw["results"][row["planned_trial_id"]] = entry
    raw_path = tmp_path / "raw_result_manifest.json"
    raw_path.write_bytes(canonical_json_bytes(raw))
    report = runner._recover_atomic_result_orphans(
        stack,
        results,
        raw_path,
        raw,
        {row["planned_trial_id"]: row},
    )
    assert report["recovered_orphan_result_count"] == 0
    audit = runner._audit_raw_inventory(
        stack, results, raw, {row["planned_trial_id"]: row}
    )
    assert len(audit["payloads"]) == 1
    assert audit["corrupt_trial_count"] == 0


def test_resume_adopts_canonical_result_from_atomic_crash_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack, row, payload = _seedless_resume_case(monkeypatch)
    results = tmp_path / "raw_results"
    results.mkdir()
    path = results / result_filename(row["planned_trial_id"])
    path.write_bytes(canonical_json_bytes(payload))
    raw = runner._empty_raw_manifest("synthetic-confirmatory-v1")
    raw_path = tmp_path / "raw_result_manifest.json"
    report = runner._recover_atomic_result_orphans(
        stack,
        results,
        raw_path,
        raw,
        {row["planned_trial_id"]: row},
    )
    assert report == {
        "recovered_orphan_result_count": 1,
        "recovered_orphan_trial_ids": [row["planned_trial_id"]],
    }
    entry = raw["results"][row["planned_trial_id"]]
    assert entry["path"] == path.name
    assert entry["sha256"] == file_sha256(path)
    assert json.loads(raw_path.read_text(encoding="utf-8")) == raw
    assert runner._audit_raw_inventory(
        stack, results, raw, {row["planned_trial_id"]: row}
    )["missing_trial_count"] == 0


def test_completed_raw_manifest_sha_is_bound_to_canonical_file(
    tmp_path: Path,
) -> None:
    raw = runner._empty_raw_manifest("synthetic-confirmatory-v1")
    raw_path = tmp_path / "raw_result_manifest.json"
    raw_path.write_bytes(canonical_json_bytes(raw))
    assert runner._completed_raw_manifest_sha256(raw_path, raw) == file_sha256(
        raw_path
    )

    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical in-memory inventory"):
        runner._completed_raw_manifest_sha256(raw_path, raw)


def _current_manifest_fixture_run(tmp_path: Path) -> Path:
    """Rebind tracked seed-free results to the live test manifest identity."""

    source = ROOT / "artifacts/fixture_qualification"
    run = tmp_path / "fixture-run"
    shutil.copytree(source / "raw_results", run / "raw_results")
    raw = json.loads(
        (source / "raw_result_manifest.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "frozen_assets/frozen_experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for trial_id, entry in raw["results"].items():
        path = run / "raw_results" / entry["path"]
        value = json.loads(path.read_text(encoding="utf-8"))
        value["implementation_sha256"] = manifest["manifest_payload_sha256"]
        path.write_bytes(canonical_json_bytes(value))
        entry["sha256"] = file_sha256(path)
        assert entry["planned_trial_id"] == trial_id
    (run / "raw_result_manifest.json").write_bytes(canonical_json_bytes(raw))
    return run


def test_existing_fixture_publication_runs_real_seed_free_verification_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phase_a_harness import phase_a_execution_chain_audit as execution

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("fixture publication crossed an execution boundary")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(execution, "execute_open3d_fixture", forbidden)
    monkeypatch.setattr(execution, "execute_pcl_fixture", forbidden)
    artifact = tmp_path / "fixture-publication"
    fixture_run = _current_manifest_fixture_run(tmp_path)
    report = audit_and_publish_existing_fixture_results(
        manifest_path=ROOT / "frozen_assets/frozen_experiment_manifest.json",
        fixture_run_dir=fixture_run,
        artifact_dir=artifact,
    )
    assert report["FIXTURE_PUBLICATION_PASS"] is True
    assert report["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is True
    assert report["fixture_snapshot_count"] == 3
    assert report["fixture_trial_count"] == 6
    assert report["open3d_trial_count"] == report["pcl_trial_count"] == 3
    assert report["analysis_verifier_difference_count"] == 0
    assert report["input_pairing_violation_count"] == 0
    assert report["fresh_resume_scientific_equivalence"] is True
    assert report["resume_backend_execution_count"] == 0
    assert report["backend_execution_count"] == 0
    assert report["rng_instantiation_count"] == 0
    live = verify_fixture_publication_artifact(artifact, write_report=False)
    recorded = json.loads(
        (artifact / "artifact_verification.json").read_text(encoding="utf-8")
    )
    assert live == recorded
    assert live["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is True
    assert live["artifact_inventory_pass"] is True
    assert live["sha256_validation_pass"] is True
    assert live["decision_semantics_pass"] is True
    recorded["sha256_entry_count"] = -1
    (artifact / "artifact_verification.json").write_text(
        json.dumps(recorded, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_record = verify_fixture_publication_artifact(
        artifact, write_report=False
    )
    assert tampered_record["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is False
    assert tampered_record["recorded_verification_match_pass"] is False


def test_fixture_publication_artifact_verifier_rejects_result_tamper(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "fixture-publication"
    fixture_run = _current_manifest_fixture_run(tmp_path)
    audit_and_publish_existing_fixture_results(
        manifest_path=ROOT / "frozen_assets/frozen_experiment_manifest.json",
        fixture_run_dir=fixture_run,
        artifact_dir=artifact,
    )
    result_path = next((artifact / "evidence/raw_results").glob("*.json"))
    result_path.write_text("{}\n", encoding="utf-8")
    report = verify_fixture_publication_artifact(artifact, write_report=False)
    assert report["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is False
    assert report["sha256_mismatch_count"] == 1


def test_fixture_publisher_rejects_primary_independent_difference_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phase_a_harness import fixture_publication as publication

    original = publication.independently_verify_phase_a_stage1_fixture

    def disagree(**kwargs: Any) -> dict[str, Any]:
        result = original(**kwargs)
        result["difference_count"] = 1
        result["differences"] = ["backend_inventory"]
        return result

    monkeypatch.setattr(
        publication,
        "independently_verify_phase_a_stage1_fixture",
        disagree,
    )
    artifact = tmp_path / "fixture-publication"
    fixture_run = _current_manifest_fixture_run(tmp_path)
    with pytest.raises(ValueError, match="primary/independent exact"):
        audit_and_publish_existing_fixture_results(
            manifest_path=ROOT
            / "frozen_assets/frozen_experiment_manifest.json",
            fixture_run_dir=fixture_run,
            artifact_dir=artifact,
        )
    assert not artifact.exists()


@pytest.mark.parametrize("tamper_kind", ["identity", "extra_file", "noncanonical"])
def test_resume_rejects_tampered_or_unrecognized_orphan_without_manifest_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    stack, row, payload = _seedless_resume_case(monkeypatch)
    results = tmp_path / "raw_results"
    results.mkdir()
    if tamper_kind == "extra_file":
        path = results / "unrecognized.json"
        path.write_bytes(canonical_json_bytes(payload))
    else:
        path = results / result_filename(row["planned_trial_id"])
        if tamper_kind == "identity":
            payload["source_checksum"] = "8" * 64
            path.write_bytes(canonical_json_bytes(payload))
        else:
            path.write_text(
                json.dumps(payload, sort_keys=False, allow_nan=False),
                encoding="utf-8",
            )
    raw = runner._empty_raw_manifest("synthetic-confirmatory-v1")
    raw_path = tmp_path / "raw_result_manifest.json"
    with pytest.raises(ValueError, match="orphan"):
        runner._recover_atomic_result_orphans(
            stack,
            results,
            raw_path,
            raw,
            {row["planned_trial_id"]: row},
        )
    assert raw["results"] == {}
    assert not raw_path.exists()


def _prerun_decision() -> dict[str, Any]:
    return {
        **{name: True for name in PRERUN_REQUIRED_GATE_NAMES},
        **{name: 0 for name in PRERUN_ZERO_EXECUTION_COUNTER_NAMES},
        "CONFIRMATORY_RUN_AUTHORIZED": True,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_PASS": "NOT_EVALUATED",
        "SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS": True,
        "required_gate_names": list(PRERUN_REQUIRED_GATE_NAMES),
        "zero_execution_counter_names": list(
            PRERUN_ZERO_EXECUTION_COUNTER_NAMES
        ),
    }


def _formal_prerun_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Any], list[Path]]:
    root = tmp_path / "formal-root"
    manifest_path = root / MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "bound_files": {"runner": {"path": "runner.py", "sha256": "a" * 64}},
        "formal_execution_authorized": True,
        "manifest_payload_sha256": "b" * 64,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact = root / runner.PRERUN_ARTIFACT_RELATIVE
    artifact.mkdir(parents=True)
    decision = _prerun_decision()
    implementation = {
        "bound_files": manifest["bound_files"],
        "formal_branch": runner.FORMAL_BRANCH,
        "formal_execution_authorized": True,
        "formal_manifest_file_sha256": file_sha256(manifest_path),
        "formal_manifest_path": MANIFEST_RELATIVE.as_posix(),
        "formal_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "formal_output_dir": "results/synthetic_confirmatory_v1",
        "formal_pre_run_tag": runner.FORMAL_PRERUN_TAG,
        "formal_run_id": "synthetic-confirmatory-v1",
        "formal_workers": 2,
        "schema_version": "synthetic_confirmatory_prerun_implementation_manifest_v1",
    }
    run = {
        "final_decision": decision,
        "formal_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "formal_run_id": "synthetic-confirmatory-v1",
    }
    for name in PRERUN_ROOT_FILES:
        if not name.endswith(".json") or name == "artifact_verification.json":
            continue
        value: dict[str, Any] = {}
        if name == "final_decision.json":
            value = decision
        elif name == "implementation_manifest.json":
            value = implementation
        elif name == "run_manifest.json":
            value = run
        (artifact / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (artifact / "pre_run_report.md").write_text(
        "SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS\n"
        "CONFIRMATORY_RUN_AUTHORIZED\n"
        "SYNTHETIC_CONFIRMATORY_EXECUTED\n"
        "SYNTHETIC_CONFIRMATORY_PASS\n"
        "no Confirmatory snapshot\n",
        encoding="utf-8",
    )
    live = {
        "CONFIRMATORY_ARTIFACT_VERIFICATION_PASS": True,
        "schema_version": "synthetic_confirmatory_prerun_artifact_verification_v1",
    }
    (artifact / "artifact_verification.json").write_text(
        json.dumps(live, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    calls: list[Path] = []

    def live_verifier(path: Path, *, write_report: bool = False) -> dict[str, Any]:
        assert write_report is False
        calls.append(Path(path).resolve())
        return dict(live)

    from phase_a_harness import synthetic_confirmatory_artifact_verifier as verifier

    monkeypatch.setattr(
        verifier, "verify_synthetic_confirmatory_prerun_artifact", live_verifier
    )
    return root, manifest_path, manifest, calls


def test_formal_prerun_artifact_is_live_verified_and_manifest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_path, manifest, calls = _formal_prerun_fixture(
        tmp_path, monkeypatch
    )
    report = runner.verify_formal_prerun_artifact_binding(
        root, manifest_path, manifest
    )
    assert report["FORMAL_PRERUN_ARTIFACT_BINDING_PASS"] is True
    assert report["implementation_manifest_binding_pass"] is True
    assert report["live_artifact_verification_match_pass"] is True
    assert calls == [(root / runner.PRERUN_ARTIFACT_RELATIVE).resolve()]


def test_formal_prerun_binding_rejects_self_consistent_implementation_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_path, manifest, _ = _formal_prerun_fixture(
        tmp_path, monkeypatch
    )
    artifact = root / runner.PRERUN_ARTIFACT_RELATIVE
    path = artifact / "implementation_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["formal_manifest_payload_sha256"] = "c" * 64
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(PermissionError, match="artifact/manifest"):
        runner.verify_formal_prerun_artifact_binding(
            root, manifest_path, manifest
        )


def test_formal_prerun_binding_requires_exact_artifact_path(tmp_path: Path) -> None:
    root = tmp_path / "formal-root"
    manifest_path = root / MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="exact.*artifact"):
        runner.verify_formal_prerun_artifact_binding(
            root,
            manifest_path,
            {"formal_execution_authorized": True},
        )


def _git(*arguments: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_formal_git_gate_requires_tagged_clean_head_and_committed_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "git-gate"
    root.mkdir()
    _git("init", cwd=root)
    _git("checkout", "-b", runner.FORMAL_BRANCH, cwd=root)
    _git("config", "user.email", "fixture@example.invalid", cwd=root)
    _git("config", "user.name", "Fixture", cwd=root)
    manifest_path = root / MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"formal_execution_authorized":true}\n', encoding="utf-8")
    _git("add", MANIFEST_RELATIVE.as_posix(), cwd=root)
    _git("commit", "-m", "fixture pre-run freeze", cwd=root)
    _git("tag", runner.FORMAL_PRERUN_TAG, cwd=root)
    report = runner.verify_formal_git_gate(root, manifest_path)
    assert report["FORMAL_GIT_GATE_PASS"] is True
    assert report["head_commit"] == report["pre_run_tag_commit"]
    manifest_path.write_text('{"formal_execution_authorized":false}\n', encoding="utf-8")
    with pytest.raises(PermissionError, match="tag/HEAD/clean"):
        runner.verify_formal_git_gate(root, manifest_path)


def _load_cli_module() -> Any:
    path = ROOT / "scripts/run_synthetic_confirmatory.py"
    specification = importlib.util.spec_from_file_location(
        "test_run_synthetic_confirmatory_cli", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_cli_exposes_only_six_scientific_arguments() -> None:
    parser = _load_cli_module().build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert options == {
        "--manifest",
        "--run-id",
        "--output-dir",
        "--workers",
        "--resume",
        "--dry-run",
    }


@pytest.mark.parametrize(
    "forbidden",
    [
        "--override",
        "--replace-seed",
        "--change-gate",
        "--change-model",
        "--fit-model",
        "--backend-subset",
        "--exclude-scene",
        "--native",
        "--ignore-manifest",
    ],
)
def test_cli_rejects_forbidden_override_arguments(forbidden: str) -> None:
    parser = _load_cli_module().build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--manifest",
                "manifest.json",
                "--run-id",
                "synthetic-confirmatory-v1",
                "--output-dir",
                "results/synthetic_confirmatory_v1",
                "--workers",
                "2",
                "--dry-run",
                forbidden,
            ]
        )


def test_unauthorized_formal_call_stops_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "results/synthetic_confirmatory_v1"
    from phase_a_harness import full_synthetic_development_protocol as isolation

    monkeypatch.setattr(isolation, "assert_isolated_python_runtime", lambda: {})
    monkeypatch.setattr(
        runner,
        "load_synthetic_confirmatory_stack",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("formal execution is not authorized")
        ),
    )
    with pytest.raises(PermissionError, match="not authorized"):
        runner.execute_synthetic_confirmatory(
            manifest_path=tmp_path / MANIFEST_RELATIVE,
            run_id="synthetic-confirmatory-v1",
            output_dir=output,
            workers=2,
            resume=True,
        )
    assert not output.exists()
