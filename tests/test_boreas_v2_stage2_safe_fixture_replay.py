from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import phase_a_harness.runtime_lifecycle_fixture as lifecycle_fixture
import phase_a_harness.phase_a_execution_chain_audit as execution_audit
from phase_a_harness.phase_a_execution_chain_fixture import (
    build_fixture_snapshots,
)
from phase_a_harness.phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
)
from phase_a_harness.real_data_preparation import stage2_safe_fixture_replay as replay
from phase_a_harness.runtime_lifecycle_io import canonical_json_sha256
from phase_a_harness.runtime_path_policy import qualify_runtime_paths


ROOT = Path(__file__).resolve().parents[1]
FORMAL_PYTHON = Path(
    "/home/lj/.local/share/degen-lio-micromamba/envs/"
    "degen-lio-zprm-py311/bin/python3.11"
).resolve(strict=True)
IMPLEMENTATION_SHA = json.loads(
    (ROOT / "frozen_assets/frozen_experiment_manifest.json").read_text()
)["manifest_payload_sha256"]


def _write_marker(
    directory: Path,
    *,
    repository: Path = ROOT,
    python: Path = FORMAL_PYTHON,
) -> tuple[Path, Path, dict[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    marker = (directory / "formal_stage2_marker.json").resolve()
    events = (directory / "safe_fixture_replay_events.jsonl").resolve()
    value = replay.build_formal_stage2_marker(
        repository_root=repository,
        git_head="a" * 40,
        formal_python_executable=python,
        event_log_path=events,
    )
    marker.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    environment = replay.formal_stage2_environment(
        marker_path=marker,
        marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
    )
    return marker, events, environment


@pytest.fixture
def formal_replay_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    marker, events, environment = _write_marker(tmp_path / "evidence")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    replay.initialize_safe_replay_event_log()
    return marker, events


def _common(fixture, backend: str) -> dict[str, object]:
    return {
        "backend": backend,
        "condition": fixture.condition,
        "implementation_sha256": "1" * 64,
        "planned_trial_id": f"{fixture.snapshot_id}/{backend}",
        "protocol_sha256": "2" * 64,
        "reference_pose_checksum": fixture.checksums["reference_pose_checksum"],
        "scene_variant": fixture.scene_variant,
        "schema_version": "phase_a_trial_result_v1",
        "snapshot_checksum": fixture.checksums["snapshot_checksum"],
        "snapshot_id": fixture.snapshot_id,
        "snapshot_lock_sha256": "3" * 64,
        "source_checksum": fixture.checksums["source_checksum"],
        "target_checksum": fixture.checksums["target_checksum"],
    }


def _parameters(backend: str) -> dict[str, object]:
    value = json.loads(
        (
            ROOT
            / "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
        ).read_text(encoding="utf-8")
    )
    section = (
        "open3d_parameter_contract"
        if backend == OPEN3D_BACKEND
        else "pcl_parameter_contract"
    )
    return value[section]["parameters"]


def test_triple_gate_rejects_partial_marker_but_no_registration_alone_is_normal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        replay.FORMAL_FULL_PYTEST_ENV,
        replay.MARKER_PATH_ENV,
        replay.MARKER_SHA256_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(replay.NO_REGISTRATION_ENV, "1")
    assert replay.safe_fixture_replay_active() is False
    monkeypatch.setenv(replay.FORMAL_FULL_PYTEST_ENV, "1")
    with pytest.raises(replay.Stage2SafeFixtureReplayError, match="partial"):
        replay.safe_fixture_replay_active()


def test_normal_environment_calls_the_existing_backend_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        replay.NO_REGISTRATION_ENV,
        replay.FORMAL_FULL_PYTEST_ENV,
        replay.MARKER_PATH_ENV,
        replay.MARKER_SHA256_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    observed = {"calls": 0}

    def sentinel(**kwargs):
        observed["calls"] += 1
        return {"sentinel": kwargs}

    monkeypatch.setattr(lifecycle_fixture, "execute_open3d_fixture", sentinel)
    fixture = build_fixture_snapshots()[0]
    result = lifecycle_fixture.execute_open3d_fixture(
        fixture=fixture,
        common=_common(fixture, OPEN3D_BACKEND),
        parameters={"normal": True},
    )
    assert observed["calls"] == 1
    assert "sentinel" in result


def test_direct_replay_never_calls_backends_and_records_exact_chain(
    formal_replay_environment: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(**_kwargs):
        raise AssertionError("a real registration backend was invoked")

    monkeypatch.setattr(lifecycle_fixture, "execute_open3d_fixture", forbidden)
    monkeypatch.setattr(lifecycle_fixture, "execute_pcl_fixture", forbidden)
    for fixture in build_fixture_snapshots():
        for backend in (OPEN3D_BACKEND, PCL_BACKEND):
            result = replay.replay_fixture_result(
                fixture=fixture,
                common=_common(fixture, backend),
                parameters=_parameters(backend),
                pcl_cli=(ROOT / "bin/pcl_point_to_plane_cli")
                if backend == PCL_BACKEND
                else None,
            )
            expected = (
                "NO_CORRESPONDENCES"
                if fixture.condition == "FIXTURE_NO_CORRESPONDENCE"
                else "NONE"
            )
            assert result["failure_classification"] == expected
            assert result["translation_update_m"] == 0.0
            assert result["rotation_update_rad"] == 0.0
    summary = replay.summarize_safe_replay_event_log()
    assert summary["safe_fixture_replay_event_count"] == 6
    assert summary["safe_simulated_materialization_count"] == 6
    assert summary["safe_simulated_open3d_materialization_count"] == 3
    assert summary["safe_simulated_pcl_materialization_count"] == 3
    assert summary["safe_fixture_replay_catalog_sha256"] == (
        "5402064dcfa87d489541685f57723bca5fd2b8128ec13762590af00637ba5fb9"
    )


def test_pcl_replay_rejects_noncanonical_or_changed_executable(
    formal_replay_environment: tuple[Path, Path], tmp_path: Path
) -> None:
    fixture = build_fixture_snapshots()[0]
    changed = tmp_path / "pcl_point_to_plane_cli"
    changed.write_bytes(b"not the frozen CLI")
    with pytest.raises(replay.Stage2SafeFixtureReplayError, match="SHA differs"):
        replay.replay_fixture_result(
            fixture=fixture,
            common=_common(fixture, PCL_BACKEND),
            parameters=_parameters(PCL_BACKEND),
            pcl_cli=changed,
        )


def test_runtime_lifecycle_direct_fresh_resume_replays_then_executes_zero(
    formal_replay_environment: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = qualify_runtime_paths(
        "safe-replay-runtime-lifecycle",
        runtime_root=tmp_path / "runtime",
        repository_root=ROOT,
        run_kind="fixture",
    ).layout
    lifecycle_fixture.prepare_runtime_snapshots(layout=layout)
    contract = lifecycle_fixture.build_fixture_run_contract(
        repository=ROOT,
        layout=layout,
        run_id=layout.run_id,
        workers=1,
        expected_commit="0" * 40,
        expected_branch="fixture-branch",
        expected_tag="fixture-tag",
        runtime_path_policy_sha256="4" * 64,
        qualification_delay_seconds=0.0,
    )

    def forbidden(**_kwargs):
        raise AssertionError("a real registration backend was invoked")

    monkeypatch.setattr(execution_audit, "run_open3d_full", forbidden)
    monkeypatch.setattr(execution_audit, "run_pcl_point_to_plane", forbidden)
    fresh = lifecycle_fixture.execute_runtime_trials(
        repository=ROOT,
        layout=layout,
        run_id=layout.run_id,
        invocation_id="fresh",
        contract_sha256=canonical_json_sha256(contract),
        implementation_sha256=IMPLEMENTATION_SHA,
        resume=False,
    )
    resumed = lifecycle_fixture.execute_runtime_trials(
        repository=ROOT,
        layout=layout,
        run_id=layout.run_id,
        invocation_id="resume",
        contract_sha256=canonical_json_sha256(contract),
        implementation_sha256=IMPLEMENTATION_SHA,
        resume=True,
    )
    assert fresh["backend_execution_count_this_invocation"] == 6
    assert resumed["backend_execution_count_this_invocation"] == 0
    assert resumed["resume_skipped_valid_result_count"] == 6
    assert replay.summarize_safe_replay_event_log()[
        "safe_fixture_replay_event_count"
    ] == 6


def test_qualification_adapters_replay_without_low_level_backend_calls(
    formal_replay_environment: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a low-level registration backend was invoked")

    monkeypatch.setattr(execution_audit, "run_open3d_full", forbidden)
    monkeypatch.setattr(execution_audit, "run_pcl_point_to_plane", forbidden)
    fixture = build_fixture_snapshots()[0]
    for backend in (OPEN3D_BACKEND, PCL_BACKEND):
        backend_input = lifecycle_fixture.QualificationBackendInput(
            backend=backend,
            fixture=fixture,
            parameters=_parameters(backend),
            pcl_cli=(ROOT / "bin/pcl_point_to_plane_cli")
            if backend == PCL_BACKEND
            else None,
        )
        if backend == OPEN3D_BACKEND:
            result = lifecycle_fixture.execute_qualification_open3d(
                backend_input=backend_input,
                common=_common(fixture, backend),
            )
        else:
            result = lifecycle_fixture.execute_qualification_pcl(
                backend_input=backend_input,
                common=_common(fixture, backend),
            )
        assert result["failure_classification"] == "NONE"
    summary = replay.summarize_safe_replay_event_log()
    assert summary["safe_fixture_replay_event_count"] == 2
    assert summary["safe_simulated_open3d_materialization_count"] == 1
    assert summary["safe_simulated_pcl_materialization_count"] == 1


def test_frozen_pytest_double_is_called_and_attested_under_formal_replay(
    formal_replay_environment: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_fixture_snapshots()[0]
    common = _common(fixture, OPEN3D_BACKEND)
    rows = json.loads(
        (
            ROOT
            / "artifacts/synthetic_confirmatory_v2_prerun/"
            "fixture_publication/primary_analysis.json"
        ).read_text(encoding="utf-8")
    )["results"]
    baseline = next(
        row
        for row in rows
        if row["condition"] == fixture.condition
        and row["backend"] == OPEN3D_BACKEND
    )
    calls = {"count": 0}

    def frozen_double(**_kwargs):
        calls["count"] += 1
        value = dict(baseline)
        value.update(common)
        value["runtime_ms"] = 1.0
        return value

    frozen_double.__module__ = "test_runtime_lifecycle_fixture"
    frozen_double.__qualname__ = "_install_fake_backends.<locals>.fake"
    monkeypatch.setattr(
        lifecycle_fixture, "execute_open3d_fixture", frozen_double
    )
    result = lifecycle_fixture._execute_or_replay_stage2_fixture(
        executor=lifecycle_fixture.execute_open3d_fixture,
        original=lifecycle_fixture._ORIGINAL_EXECUTE_OPEN3D_FIXTURE,
        fixture=fixture,
        common=common,
        parameters=_parameters(OPEN3D_BACKEND),
    )
    assert calls["count"] == 1
    assert result["runtime_ms"] == 1.0
    assert replay.summarize_safe_replay_event_log()[
        "safe_simulated_open3d_materialization_count"
    ] == 1


def test_sigterm_after_adapter_replay_leaves_one_durable_verified_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _marker, _events, controlled = _write_marker(tmp_path / "evidence")
    for name, value in controlled.items():
        monkeypatch.setenv(name, value)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")
    code = r"""
import json, os, signal
from pathlib import Path
from phase_a_harness.phase_a_execution_chain_fixture import build_fixture_snapshots
from phase_a_harness.phase_a_trial_result_schema import OPEN3D_BACKEND
from phase_a_harness.real_data_preparation.stage2_safe_fixture_replay import initialize_safe_replay_event_log
from phase_a_harness.runtime_lifecycle_fixture import QualificationBackendInput, execute_qualification_open3d
root=Path.cwd()
parameters=json.loads((root/'frozen_assets/fixtures/fixture_backend_parameter_lock.json').read_text())['open3d_parameter_contract']['parameters']
fixture=build_fixture_snapshots()[0]
common={'backend':OPEN3D_BACKEND,'condition':fixture.condition,'implementation_sha256':'1'*64,'planned_trial_id':fixture.snapshot_id+'/'+OPEN3D_BACKEND,'protocol_sha256':'2'*64,'reference_pose_checksum':fixture.checksums['reference_pose_checksum'],'scene_variant':fixture.scene_variant,'schema_version':'phase_a_trial_result_v1','snapshot_checksum':fixture.checksums['snapshot_checksum'],'snapshot_id':fixture.snapshot_id,'snapshot_lock_sha256':'3'*64,'source_checksum':fixture.checksums['source_checksum'],'target_checksum':fixture.checksums['target_checksum']}
initialize_safe_replay_event_log()
execute_qualification_open3d(backend_input=QualificationBackendInput(OPEN3D_BACKEND,fixture,parameters,None),common=common)
os.kill(os.getpid(), signal.SIGTERM)
"""
    completed = subprocess.run(
        [str(FORMAL_PYTHON), "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == -signal.SIGTERM
    summary = replay.summarize_safe_replay_event_log()
    assert summary["safe_fixture_replay_event_count"] == 1
    assert summary["safe_simulated_open3d_materialization_count"] == 1


def test_python_grandchild_in_copied_repository_inherits_and_replays(
    tmp_path: Path,
) -> None:
    copied = (tmp_path / "copied-repository").resolve()
    shutil.copytree(ROOT / "src", copied / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (copied / "bin").mkdir()
    shutil.copy2(ROOT / "bin/pcl_point_to_plane_cli", copied / "bin/pcl_point_to_plane_cli")
    parameter_relative = Path(
        "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
    )
    (copied / parameter_relative.parent).mkdir(parents=True)
    shutil.copy2(ROOT / parameter_relative, copied / parameter_relative)
    marker, events, controlled = _write_marker(tmp_path / "grandchild-evidence")
    environment = dict(os.environ)
    environment.update(controlled)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(copied / "src")
    replay.initialize_safe_replay_event_log.__module__  # import-path sanity
    # The parent does not import the copied provider.  A child creates a Python
    # grandchild, and only the grandchild imports and executes copied bytes.
    parent_code = r"""
import os, subprocess, sys
code = r'''\
import json
from pathlib import Path
from phase_a_harness.phase_a_execution_chain_fixture import build_fixture_snapshots
from phase_a_harness.phase_a_trial_result_schema import OPEN3D_BACKEND, PCL_BACKEND
from phase_a_harness.real_data_preparation.stage2_safe_fixture_replay import initialize_safe_replay_event_log, summarize_safe_replay_event_log
from phase_a_harness.runtime_lifecycle_fixture import QualificationBackendInput, execute_qualification_open3d, execute_qualification_pcl
initialize_safe_replay_event_log()
parameter_lock=json.loads(Path('frozen_assets/fixtures/fixture_backend_parameter_lock.json').read_text())
for fixture in build_fixture_snapshots():
    for backend in (OPEN3D_BACKEND, PCL_BACKEND):
        common={
          'backend':backend,'condition':fixture.condition,'implementation_sha256':'1'*64,
          'planned_trial_id':fixture.snapshot_id+'/'+backend,'protocol_sha256':'2'*64,
          'reference_pose_checksum':fixture.checksums['reference_pose_checksum'],
          'scene_variant':fixture.scene_variant,'schema_version':'phase_a_trial_result_v1',
          'snapshot_checksum':fixture.checksums['snapshot_checksum'],'snapshot_id':fixture.snapshot_id,
          'snapshot_lock_sha256':'3'*64,'source_checksum':fixture.checksums['source_checksum'],
          'target_checksum':fixture.checksums['target_checksum']}
        section='open3d_parameter_contract' if backend==OPEN3D_BACKEND else 'pcl_parameter_contract'
        backend_input=QualificationBackendInput(backend,fixture,parameter_lock[section]['parameters'],Path('bin/pcl_point_to_plane_cli').resolve() if backend==PCL_BACKEND else None)
        (execute_qualification_open3d if backend==OPEN3D_BACKEND else execute_qualification_pcl)(backend_input=backend_input,common=common)
print(json.dumps(summarize_safe_replay_event_log(), sort_keys=True))
'''
completed=subprocess.run([sys.executable,'-c',code],cwd=os.getcwd(),env=dict(os.environ),capture_output=True,text=True,check=False)
print(completed.stdout,end='')
print(completed.stderr,end='',file=sys.stderr)
raise SystemExit(completed.returncode)
"""
    parent_path = tmp_path / "safe_replay_parent.py"
    parent_path.write_text(parent_code, encoding="utf-8")
    completed = subprocess.run(
        [str(FORMAL_PYTHON), str(parent_path)],
        cwd=copied,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    assert value["safe_fixture_replay_event_count"] == 6
    assert value["safe_simulated_open3d_materialization_count"] == 3
    assert value["safe_simulated_pcl_materialization_count"] == 3
    assert value["safe_fixture_replay_marker_path"] == str(marker)
    assert events.is_file()


def test_copied_repository_replay_rejects_changed_bound_implementation(
    tmp_path: Path,
) -> None:
    copied = (tmp_path / "copied-repository").resolve()
    shutil.copytree(
        ROOT / "src",
        copied / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    marker, _events, controlled = _write_marker(tmp_path / "evidence")
    candidate = (
        copied
        / "src/phase_a_harness/real_data_preparation/"
        "stage2_safe_fixture_replay.py"
    )
    candidate.write_bytes(candidate.read_bytes() + b"\n# tamper\n")
    environment = dict(os.environ)
    environment.update(controlled)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(copied / "src")
    completed = subprocess.run(
        [
            str(FORMAL_PYTHON),
            "-c",
            (
                "from phase_a_harness.real_data_preparation."
                "stage2_safe_fixture_replay import safe_fixture_replay_active;"
                "safe_fixture_replay_active()"
            ),
        ],
        cwd=copied,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "implementation SHA differs" in completed.stderr
    assert marker.is_file()
