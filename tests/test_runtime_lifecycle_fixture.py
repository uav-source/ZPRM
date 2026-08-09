from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from phase_a_harness.contracts import file_sha256
from phase_a_harness.runtime_lifecycle_fixture import (
    RuntimeLifecycleCorruption,
    audit_runtime_results,
    build_fixture_run_contract,
    execute_runtime_trials,
    load_completed_fixture_results,
    prepare_runtime_snapshots,
    publish_fixture_runtime_artifact,
    recover_canonical_result_orphans,
    run_fixture_lifecycle,
    summarize_fixture_outcomes,
    validate_all_runtime_snapshots,
    validate_runtime_snapshot,
)
from phase_a_harness.runtime_lifecycle_io import (
    atomic_replace_canonical_json,
    canonical_json_sha256,
    read_canonical_json,
)
from phase_a_harness.runtime_lifecycle_science_audit import (
    fixture_scientific_regression,
    scientific_core_binding,
)
from phase_a_harness.runtime_path_policy import qualify_runtime_paths
from phase_a_harness.phase_a_execution_chain_fixture import build_fixture_snapshots
from phase_a_harness.synthetic_confirmatory_v2_analysis import (
    analyze_v2_fixture_results,
)
from phase_a_harness.synthetic_confirmatory_v2_artifact_verifier import (
    verify_synthetic_confirmatory_v2_fixture_artifact,
)
from phase_a_harness.synthetic_confirmatory_v2_independent_verifier import (
    compare_v2_fixture_primary_and_independent,
    independently_analyze_v2_fixture_results,
)


ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = ROOT / "scripts/run_runtime_lifecycle_fixture.py"
BASELINE_ARTIFACT = ROOT / "artifacts/synthetic_confirmatory_v2_prerun/fixture_publication"
IMPLEMENTATION_SHA = json.loads(
    (ROOT / "frozen_assets/frozen_experiment_manifest.json").read_text()
)["manifest_payload_sha256"]


def _load_worker_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "runtime_lifecycle_worker_under_test", WORKER_SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _monitor_probe(
    *, log_path: Path, mode: str, external_path: Path | None = None
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    code = r"""
import importlib.util
import json
import sys
from pathlib import Path

script, log_path, repository, mode, external = sys.argv[1:]
spec = importlib.util.spec_from_file_location("runtime_lifecycle_worker_probe", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
prepared = module._prepare_source_access_log(Path(log_path), repository=Path(repository))
monitor = module._EarlySourceAccessMonitor(prepared)
monitor.install()
if mode == "external":
    Path(external).read_bytes()
elif mode == "source-audit-event":
    try:
        sys.audit("open", "/home/lj/Degen-LIO/__runtime_audit_probe__", "r", 0)
    except PermissionError:
        pass
else:
    raise AssertionError(mode)
secondary = type("Secondary", (), {"count": 0, "paths": []})()
print(json.dumps(module._source_access_report(
    early_monitor=monitor,
    source_monitor=secondary,
    import_paths=[],
), sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(WORKER_SCRIPT),
            str(log_path),
            str(ROOT),
            mode,
            "" if external_path is None else str(external_path),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout) if completed.returncode == 0 else {}
    return value, completed


def _layout(tmp_path: Path, run_id: str = "fixture-test"):
    return qualify_runtime_paths(
        run_id,
        runtime_root=tmp_path / "runtime",
        repository_root=ROOT,
        run_kind="fixture",
    ).layout


def _baseline_rows() -> list[dict[str, Any]]:
    return json.loads((BASELINE_ARTIFACT / "primary_analysis.json").read_text())[
        "results"
    ]


def _file_inventory_for_test(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _install_fake_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = {
        (row["condition"], row["backend"]): row for row in _baseline_rows()
    }

    def fake(*, fixture, common, **_kwargs):
        row = dict(rows[(fixture.condition, common["backend"])])
        assert all(row[key] == value for key, value in common.items())
        row["runtime_ms"] = 1.0
        return row

    monkeypatch.setattr(
        "phase_a_harness.runtime_lifecycle_fixture.execute_open3d_fixture", fake
    )
    monkeypatch.setattr(
        "phase_a_harness.runtime_lifecycle_fixture.execute_pcl_fixture", fake
    )


def _contract(layout, delay: float = 0.0) -> dict[str, Any]:
    return build_fixture_run_contract(
        repository=ROOT,
        layout=layout,
        run_id=layout.run_id,
        workers=2,
        expected_commit="0" * 40,
        expected_branch="fixture-branch",
        expected_tag="fixture-tag",
        runtime_path_policy_sha256="1" * 64,
        qualification_delay_seconds=delay,
    )


def _complete_fake_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _install_fake_backends(monkeypatch)
    layout = _layout(tmp_path)
    prepare_runtime_snapshots(layout=layout)
    contract = _contract(layout)
    result = execute_runtime_trials(
        repository=ROOT,
        layout=layout,
        run_id=layout.run_id,
        invocation_id="fresh",
        contract_sha256=canonical_json_sha256(contract),
        implementation_sha256=IMPLEMENTATION_SHA,
        resume=False,
    )
    return layout, contract, result


def test_contract_is_exact_seed_free_three_by_six(tmp_path: Path) -> None:
    contract = _contract(_layout(tmp_path))
    assert contract["fixture_only"] is True
    assert contract["formal_seed_values_included"] is False
    assert contract["fixture_generation_rng_count"] == 0
    assert len(contract["planned_snapshots"]) == 3
    assert len(contract["planned_trials"]) == 6
    assert {row["backend"] for row in contract["planned_trials"]} == {
        "open3d_point_to_plane",
        "pcl_point_to_plane",
    }


def test_contract_binds_every_external_mutable_path(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    contract = _contract(layout)
    assert contract["runtime_paths"] == layout.as_dict()
    assert all(ROOT not in path.parents for path in layout.mutable_paths().values())


def test_snapshot_fresh_real_write_is_three_and_valid(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    report = prepare_runtime_snapshots(layout=layout)
    assert report["generated_snapshot_count"] == 3
    assert report["resume_skipped_valid_snapshot_count"] == 0
    assert len(validate_all_runtime_snapshots(layout)) == 3


def test_snapshot_resume_does_not_rebuild_valid_objects(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    prepare_runtime_snapshots(layout=layout)
    before = {
        path.relative_to(layout.snapshot_cache).as_posix(): file_sha256(path)
        for path in layout.snapshot_cache.rglob("*")
        if path.is_file()
    }
    report = prepare_runtime_snapshots(layout=layout)
    after = {
        path.relative_to(layout.snapshot_cache).as_posix(): file_sha256(path)
        for path in layout.snapshot_cache.rglob("*")
        if path.is_file()
    }
    assert report["generated_snapshot_count"] == 0
    assert report["resume_skipped_valid_snapshot_count"] == 3
    assert before == after


def test_snapshot_unknown_file_is_rejected(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    prepare_runtime_snapshots(layout=layout)
    target = layout.snapshot_cache / "identity"
    (target / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(RuntimeLifecycleCorruption, match="inventory mismatch"):
        validate_runtime_snapshot(target, build_fixture_snapshots()[0])


def test_snapshot_byte_corruption_is_rejected(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    prepare_runtime_snapshots(layout=layout)
    target = layout.snapshot_cache / "identity/source_points.npy"
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(RuntimeLifecycleCorruption, match="file SHA mismatch"):
        validate_runtime_snapshot(
            layout.snapshot_cache / "identity", build_fixture_snapshots()[0]
        )


def test_lineage_corruption_is_rejected(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    prepare_runtime_snapshots(layout=layout)
    target = layout.snapshot_cache / "identity/source_parent_target_indices.npy"
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(RuntimeLifecycleCorruption, match="file SHA mismatch"):
        validate_runtime_snapshot(
            layout.snapshot_cache / "identity", build_fixture_snapshots()[0]
        )


def test_no_correspondence_has_no_false_lineage_sidecar(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    prepare_runtime_snapshots(layout=layout)
    assert not (
        layout.snapshot_cache
        / "no-correspondence/source_parent_target_indices.npy"
    ).exists()


def test_fake_backend_fresh_run_writes_six_strict_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _layout_value, _contract_value, result = _complete_fake_run(
        tmp_path, monkeypatch
    )
    assert result["backend_execution_count_this_invocation"] == 6
    assert result["resume_skipped_valid_result_count"] == 0
    assert len(result["rows"]) == 6
    assert summarize_fixture_outcomes(result["rows"])[
        "FIXTURE_EXECUTION_CHAIN_PASS"
    ] is True


def test_completed_trial_resume_executes_no_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, contract, _result = _complete_fake_run(tmp_path, monkeypatch)

    def forbidden(**_kwargs):
        raise AssertionError("valid trial was reexecuted")

    monkeypatch.setattr(
        "phase_a_harness.runtime_lifecycle_fixture.execute_open3d_fixture", forbidden
    )
    monkeypatch.setattr(
        "phase_a_harness.runtime_lifecycle_fixture.execute_pcl_fixture", forbidden
    )
    resumed = execute_runtime_trials(
        repository=ROOT,
        layout=layout,
        run_id=layout.run_id,
        invocation_id="resume",
        contract_sha256=canonical_json_sha256(contract),
        implementation_sha256=IMPLEMENTATION_SHA,
        resume=True,
    )
    assert resumed["backend_execution_count_this_invocation"] == 0
    assert resumed["resume_skipped_valid_result_count"] == 6


def test_trial_checksum_is_unchanged_by_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, contract, _result = _complete_fake_run(tmp_path, monkeypatch)
    result_dir = layout.raw_results / "results"
    before = {path.name: file_sha256(path) for path in result_dir.iterdir()}
    execute_runtime_trials(
        repository=ROOT,
        layout=layout,
        run_id=layout.run_id,
        invocation_id="resume",
        contract_sha256=canonical_json_sha256(contract),
        implementation_sha256=IMPLEMENTATION_SHA,
        resume=True,
    )
    after = {path.name: file_sha256(path) for path in result_dir.iterdir()}
    assert before == after


def test_canonical_orphan_is_adopted_without_backend_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, contract, result = _complete_fake_run(tmp_path, monkeypatch)
    manifest_path = layout.raw_results / "raw_result_manifest.json"
    manifest = read_canonical_json(manifest_path)
    trial_id = sorted(manifest["results"])[0]
    manifest["results"].pop(trial_id)
    atomic_replace_canonical_json(manifest_path, manifest)
    recovered = recover_canonical_result_orphans(
        repository=ROOT,
        layout=layout,
        run_id=layout.run_id,
        contract_sha256=canonical_json_sha256(contract),
        implementation_sha256=IMPLEMENTATION_SHA,
    )
    assert recovered == {
        "recovered_orphan_result_count": 1,
        "recovered_orphan_trial_ids": [trial_id],
    }
    assert len(result["rows"]) == 6


def test_unknown_orphan_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, contract, _result = _complete_fake_run(tmp_path, monkeypatch)
    (layout.raw_results / "results/unknown.json").write_text("{}\n")
    with pytest.raises(RuntimeLifecycleCorruption, match="unrecognized result orphan"):
        recover_canonical_result_orphans(
            repository=ROOT,
            layout=layout,
            run_id=layout.run_id,
            contract_sha256=canonical_json_sha256(contract),
            implementation_sha256=IMPLEMENTATION_SHA,
        )


def test_corrupt_trial_is_rejected_and_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, contract, _result = _complete_fake_run(tmp_path, monkeypatch)
    path = next((layout.raw_results / "results").iterdir())
    path.write_bytes(path.read_bytes() + b"tamper")
    corrupted = path.read_bytes()
    with pytest.raises(RuntimeLifecycleCorruption, match="raw result SHA mismatch"):
        audit_runtime_results(
            repository=ROOT,
            layout=layout,
            run_id=layout.run_id,
            contract_sha256=canonical_json_sha256(contract),
            implementation_sha256=IMPLEMENTATION_SHA,
        )
    assert path.read_bytes() == corrupted


def test_primary_and_independent_fixture_analysis_match_exactly() -> None:
    rows = _baseline_rows()
    primary = analyze_v2_fixture_results(rows)
    independent = independently_analyze_v2_fixture_results(rows)
    comparison = compare_v2_fixture_primary_and_independent(primary, independent)
    assert comparison["exact_match_pass"] is True
    assert comparison["leaf_difference_count"] == 0


def test_external_publisher_uses_separate_staging_directory(tmp_path: Path) -> None:
    primary = json.loads((BASELINE_ARTIFACT / "primary_analysis.json").read_text())
    independent = json.loads(
        (BASELINE_ARTIFACT / "independent_verification.json").read_text()
    )
    run = json.loads((BASELINE_ARTIFACT / "run_manifest.json").read_text())
    layout = _layout(tmp_path, "publisher-test")
    result = publish_fixture_runtime_artifact(
        layout=layout,
        primary=primary,
        independent=independent,
        run_manifest=run,
    )
    assert result["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is True
    publisher = Path(result["publisher_staging_path"])
    artifact = Path(result["artifact_staging_path"])
    assert publisher.is_dir()
    assert artifact.is_dir()
    assert _file_inventory_for_test(publisher) == _file_inventory_for_test(artifact)


def test_artifact_corruption_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(BASELINE_ARTIFACT, artifact)
    table = artifact / "tables/fixture_trial_inventory.csv"
    table.write_bytes(table.read_bytes() + b"tamper")
    report = verify_synthetic_confirmatory_v2_fixture_artifact(
        artifact, write_report=False
    )
    assert report["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is False
    assert "tables/fixture_trial_inventory.csv" in report["sha256_mismatch_files"]


def test_scientific_core_binding_is_unchanged() -> None:
    report = scientific_core_binding(ROOT)
    assert report["SCIENTIFIC_CORE_FILE_CHANGE_COUNT"] == 0
    assert report["H1_H6_SEMANTICS_CHANGE_COUNT"] == 0
    assert report["FROZEN_MODEL_CHANGE_COUNT"] == 0
    assert report["BACKEND_BINDING_CHANGE_COUNT"] == 0


def test_historical_fixture_science_regression_excludes_only_runtime() -> None:
    report = fixture_scientific_regression(ROOT, _baseline_rows())
    assert report["excluded_fields"] == ["runtime_ms"]
    assert report["FIXTURE_SCIENTIFIC_PAYLOAD_CHANGE_COUNT"] == 0
    assert report["FIXTURE_TRIAL_SCIENTIFIC_FIELD_CHANGE_COUNT"] == 0


def test_full_lifecycle_binds_lock_and_git_checkpoints_with_fake_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_backends(monkeypatch)
    layout = _layout(tmp_path, "full-lifecycle")
    checkpoints: list[str] = []

    def gate(checkpoint: str):
        checkpoints.append(checkpoint)
        return {"RUNTIME_GIT_GATE_PASS": True, "checkpoint": checkpoint}

    result = run_fixture_lifecycle(
        repository=ROOT,
        layout=layout,
        run_id=layout.run_id,
        invocation_id="fresh",
        workers=2,
        resume=False,
        expected_commit="0" * 40,
        expected_branch="fixture-branch",
        expected_tag="fixture-tag",
        runtime_path_policy_sha256="1" * 64,
        git_gate=gate,
    )
    assert result["FIXTURE_EXECUTION_CHAIN_PASS"] is True
    assert layout.snapshot_lock.is_file()
    assert "PRE_RUN_GIT_GATE" in checkpoints
    assert "MID_SNAPSHOT_GIT_GATE" in checkpoints
    assert "MID_TRIAL_GIT_GATE" in checkpoints
    assert "POST_TRIAL_GIT_GATE" in checkpoints


def test_load_completed_results_requires_exact_six(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, contract, _result = _complete_fake_run(tmp_path, monkeypatch)
    rows = load_completed_fixture_results(
        repository=ROOT,
        layout=layout,
        run_id=layout.run_id,
        contract_sha256=canonical_json_sha256(contract),
        implementation_sha256=IMPLEMENTATION_SHA,
    )
    assert len(rows) == 6
    assert [row["planned_trial_id"] for row in rows] == sorted(
        row["planned_trial_id"] for row in rows
    )


@pytest.mark.parametrize(
    "search_path",
    [
        "/home/lj/Degen-LIO",
        "/home/lj/Degen-LIO/src",
        "/home/lj",
        "/home",
        "/",
    ],
)
def test_worker_rejects_source_repository_descendants_and_ancestors_in_pythonpath(
    search_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _load_worker_script()
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv(
        "MAMBA_ROOT_PREFIX", worker.FROZEN_MAMBA_ROOT_PREFIX
    )
    monkeypatch.setenv("PYTHONPATH", search_path)
    monkeypatch.setattr(worker.sys, "path", [str(ROOT / "src")])
    with pytest.raises(PermissionError, match="ancestor containing it"):
        worker._assert_isolation()


@pytest.mark.parametrize(
    "search_path", ["/home/lj/Degen-LIO", "/home/lj/Degen-LIO/tests", "/home/lj"]
)
def test_worker_rejects_source_overlap_in_sys_path(
    search_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _load_worker_script()
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv(
        "MAMBA_ROOT_PREFIX", worker.FROZEN_MAMBA_ROOT_PREFIX
    )
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(worker.sys, "path", [search_path, str(ROOT / "src")])
    with pytest.raises(PermissionError, match="ancestor containing it"):
        worker._assert_isolation()


def test_worker_requires_absolute_external_source_access_log(tmp_path: Path) -> None:
    worker = _load_worker_script()
    with pytest.raises(PermissionError, match="absolute external"):
        worker._prepare_source_access_log(
            Path("relative-source-access.ndjson"), repository=ROOT
        )
    with pytest.raises(PermissionError, match="outside both repositories"):
        worker._prepare_source_access_log(
            ROOT / "source-access.ndjson", repository=ROOT
        )


@pytest.mark.parametrize("existing_kind", ["regular", "symlink", "fifo"])
def test_worker_rejects_existing_source_access_path_without_open(
    existing_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_worker_script()
    log_path = tmp_path / "source-access.ndjson"
    if existing_kind == "regular":
        log_path.write_text("existing evidence\n", encoding="utf-8")
    elif existing_kind == "symlink":
        target = tmp_path / "symlink-target"
        target.write_text("target\n", encoding="utf-8")
        log_path.symlink_to(target)
    elif existing_kind == "fifo":
        os.mkfifo(log_path)
    else:  # pragma: no cover - parameter contract guard
        raise AssertionError(existing_kind)

    open_calls: list[tuple[object, ...]] = []

    def forbidden_open(*args: object, **_kwargs: object) -> int:
        open_calls.append(args)
        raise AssertionError("existing source-access path must not be opened")

    monkeypatch.setattr(worker.os, "open", forbidden_open)
    with pytest.raises(FileExistsError, match="must not already exist"):
        worker._prepare_source_access_log(log_path, repository=ROOT)
    assert open_calls == []


def test_worker_installs_both_monitors_before_other_runtime_imports() -> None:
    source = WORKER_SCRIPT.read_text(encoding="utf-8")
    early_install = source.index("early_source_monitor.install()")
    established_import = source.index(
        "from phase_a_harness.runner import SourceAccessMonitor"
    )
    established_install = source.index("source_monitor.install()", established_import)
    first_other_runtime_import = source.index(
        "from phase_a_harness.runtime_git_gate import verify_runtime_git_gate"
    )
    assert early_install < established_import
    assert established_import < established_install < first_other_runtime_import


def test_worker_external_source_access_log_reports_zero_on_normal_probe(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external-input.txt"
    external.write_text("external only", encoding="utf-8")
    log_path = tmp_path / "source-access.ndjson"
    report, completed = _monitor_probe(
        log_path=log_path, mode="external", external_path=external
    )
    assert completed.returncode == 0, completed.stderr
    assert log_path.is_absolute()
    assert log_path.read_bytes() == b""
    assert report["source_access_log_path"] == str(log_path)
    assert report["source_repository_early_file_read_count"] == 0
    assert report["source_repository_early_file_read_paths"] == []
    assert report["source_repository_runtime_file_read_count"] == 0
    assert report["source_repository_runtime_file_read_paths"] == []
    assert report["source_repository_runtime_import_count"] == 0
    assert report["source_repository_runtime_import_paths"] == []


def test_worker_early_hook_fsyncs_source_access_evidence_before_rejection(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "source-access.ndjson"
    report, completed = _monitor_probe(
        log_path=log_path, mode="source-audit-event"
    )
    assert completed.returncode == 0, completed.stderr
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    pid = event.pop("pid")
    assert event == {
        "event": "open",
        "event_schema": "runtime_lifecycle_source_access_event_v1",
        "path": "/home/lj/Degen-LIO/__runtime_audit_probe__",
        "sequence": 1,
        "source_repository": "/home/lj/Degen-LIO",
    }
    assert isinstance(pid, int) and pid > 0
    assert report["source_repository_early_file_read_count"] == 1
    assert report["source_repository_runtime_file_read_count"] == 1
    assert report["source_repository_runtime_file_read_paths"] == [
        "/home/lj/Degen-LIO/__runtime_audit_probe__"
    ]
