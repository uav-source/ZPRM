from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from phase_a_harness.real_data_preparation.io import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)
from phase_a_harness.real_data_preparation.stage2_safe_fixture_replay import (
    formal_stage2_environment,
)
from phase_a_harness.real_data_preparation.stage2_pytest_no_registration import (
    CHILD_ATTESTATION_FIELDS,
    CHILD_ATTESTATION_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "boreas_v2_stage2_finalizer_for_test",
    ROOT / "scripts/finalize_boreas_v2_stage2_preparation.py",
)
assert SPEC is not None and SPEC.loader is not None
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


def _junit(path: Path) -> None:
    suite = ET.Element("testsuite")
    for node_id, reason in finalizer.EXPECTED_SKIPPED_TESTS:
        classname, name = node_id.split("::", 1)
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        ET.SubElement(case, "skipped", message=reason)
    for index in range(924):
        ET.SubElement(
            suite,
            "testcase",
            classname="tests.synthetic_release_gate",
            name=f"test_{index:04d}",
        )
    path.write_bytes(ET.tostring(suite, encoding="utf-8", xml_declaration=True))


def _child_attestation(
    path: Path,
    plugin: Path,
    guard: Path,
    marker: Path,
    event_log: Path,
) -> None:
    atomic_write_json(
        path,
        {
            "actual_open3d_trials": 0,
            "actual_pcl_trials": 0,
            "actual_trials": 0,
            "estimated_transform_count": 0,
            "estimated_transform_evidence": [],
            "estimated_transform_file_count": 0,
            "estimated_transform_files": [],
            "guard_implementation_path": str(guard),
            "guard_implementation_sha256": sha256_file(guard),
            "open3d_registration_call_count": 0,
            "other_registration_process_count": 0,
            "pass": True,
            "pcl_cli_invocation_count": 0,
            "plugin_path": str(plugin),
            "plugin_sha256": sha256_file(plugin),
            "pytest_exit_status_at_attestation": 0,
            "real_trial_result_count": 0,
            "registration_execution_count": 0,
            "safe_fixture_replay_catalog_sha256": (
                finalizer.SAFE_REPLAY_CATALOG_SHA256
            ),
            "safe_fixture_replay_event_chain_head_sha256": "0" * 64,
            "safe_fixture_replay_event_count": 0,
            "safe_fixture_replay_event_log_path": str(event_log),
            "safe_fixture_replay_event_log_sha256": hashlib.sha256(b"").hexdigest(),
            "safe_fixture_replay_event_schema": finalizer.SAFE_REPLAY_EVENT_SCHEMA,
            "safe_fixture_replay_marker_path": str(marker),
            "safe_fixture_replay_marker_sha256": sha256_file(marker),
            "safe_simulated_materialization_count": 0,
            "safe_simulated_open3d_materialization_count": 0,
            "safe_simulated_pcl_materialization_count": 0,
            "schema": finalizer.CHILD_GUARD_SCHEMA,
            "status": "PASS",
            "structured_result_scan_error_count": 0,
            "structured_result_scan_error_files": [],
        },
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, int]]:
    repository = tmp_path / "repository"
    runtime = tmp_path / "runtime"
    for path in (
        repository / "src/phase_a_harness/real_data_preparation",
        runtime / "evidence",
        runtime / "checkpoints",
    ):
        path.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    plugin = (
        repository
        / "src/phase_a_harness/real_data_preparation/stage2_pytest_no_registration.py"
    )
    guard = repository / "src/phase_a_harness/real_data_preparation/guard.py"
    plugin.write_text("# fixed child plugin\n")
    guard.write_text("# fixed guard\n")
    (repository / "src/phase_a_harness/runtime_lifecycle_fixture.py").write_text(
        "# fixed runtime lifecycle fixture\n"
    )
    (
        repository
        / "src/phase_a_harness/real_data_preparation/stage2_safe_fixture_replay.py"
    ).write_text("# fixed safe fixture replay\n")
    python = tmp_path / "python3.11"
    python.write_bytes(b"synthetic interpreter")
    python.chmod(0o700)
    monkeypatch.setattr(finalizer, "FORMAL_PYTHON", python)
    monkeypatch.setattr(finalizer, "_source_worktree_status", lambda _: b"")

    def git(_repository: Path, *arguments: str) -> bytes:
        if arguments == ("rev-parse", "HEAD"):
            return b"a" * 40 + b"\n"
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return b"b" * 40 + b"\n"
        if arguments[:4] == ("ls-tree", "-r", "--name-only", "-z"):
            return b"pyproject.toml\0src/fixed.py\0"
        raise AssertionError(arguments)

    monkeypatch.setattr(finalizer, "_git_value", git)
    calls = {"pytest": 0}

    def run(
        argv: list[str], *, cwd: Path, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        if argv == [str(python), "--version"]:
            return subprocess.CompletedProcess(argv, 0, b"Python 3.11.15\n", b"")
        assert cwd == repository
        assert env is not None
        calls["pytest"] += 1
        assert argv[-1] == "tests"
        assert "-c" in argv and argv[argv.index("-c") + 1] == "pyproject.toml"
        assert finalizer.CHILD_GUARD_PLUGIN in argv
        assert env["PYTEST_ADDOPTS"] == ""
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["MAMBA_ROOT_PREFIX"] == (
            "/home/lj/.local/share/degen-lio-micromamba"
        )
        assert env["ZPRM_REAL_DATA_PREP_NO_REGISTRATION"] == "1"
        junit = Path(next(value.split("=", 1)[1] for value in argv if value.startswith("--junitxml=")))
        _junit(junit)
        marker = Path(env["ZPRM_STAGE2_FORMAL_FULL_PYTEST_MARKER"])
        marker_value = json.loads(marker.read_text())
        event_log = Path(marker_value["event_log_path"])
        event_log.write_bytes(b"")
        _child_attestation(
            Path(env["ZPRM_STAGE2_PYTEST_GUARD_ATTESTATION"]),
            plugin,
            guard,
            marker,
            event_log,
        )
        return subprocess.CompletedProcess(argv, 0, b"934 passed, 10 skipped\n", b"")

    monkeypatch.setattr(finalizer, "_run_checked", run)
    return repository, runtime, calls


@pytest.mark.parametrize(
    "crash_label",
    (
        "AFTER_FULL_TEST_RESULT",
        "AFTER_FULL_TEST_JUNIT_RENAME",
        "AFTER_FULL_TEST_ATTESTATION_RENAME",
        "AFTER_FULL_TEST_STATUS",
    ),
)
def test_fixed_full_test_gate_recovers_each_published_pair_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_label: str
) -> None:
    repository, runtime, calls = _fixture(tmp_path, monkeypatch)

    def crash(label: str) -> None:
        if label == crash_label:
            raise RuntimeError(label)

    with pytest.raises(RuntimeError, match=crash_label):
        finalizer.run_fixed_full_test_gate(
            repository, runtime, _fault_hook=crash
        )
    assert calls["pytest"] == 1
    status = finalizer.run_fixed_full_test_gate(repository, runtime)
    assert status == runtime / "evidence/full_test_status.json"
    assert calls["pytest"] == 1
    finalizer.run_fixed_full_test_gate(repository, runtime)
    assert calls["pytest"] == 1
    assert not (runtime / "checkpoints/full_test_gate_intent.json").exists()


def test_fixed_full_test_gate_rejects_forged_immutable_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, runtime, _ = _fixture(tmp_path, monkeypatch)
    status = finalizer.run_fixed_full_test_gate(repository, runtime)
    forged = json.loads(status.read_text())
    forged["passed"] -= 1
    status.write_bytes(canonical_json_bytes(forged))
    with pytest.raises(finalizer.BoreasV2Stage2ClosureError, match="status projection"):
        finalizer.run_fixed_full_test_gate(repository, runtime)


@pytest.mark.parametrize(
    ("crash_label", "calls_after_crash", "calls_after_resume"),
    (
        ("AFTER_FULL_TEST_INTENT", 0, 1),
        ("AFTER_FULL_TEST_CHILD", 1, 2),
    ),
)
def test_fixed_full_test_gate_recovers_marker_and_event_log_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_label: str,
    calls_after_crash: int,
    calls_after_resume: int,
) -> None:
    repository, runtime, calls = _fixture(tmp_path, monkeypatch)

    def crash(label: str) -> None:
        if label == crash_label:
            raise RuntimeError(label)

    with pytest.raises(RuntimeError, match=crash_label):
        finalizer.run_fixed_full_test_gate(repository, runtime, _fault_hook=crash)
    assert calls["pytest"] == calls_after_crash
    marker = runtime / "evidence/full_pytest_safe_fixture_replay_marker.json"
    assert marker.is_file()
    finalizer.run_fixed_full_test_gate(repository, runtime)
    assert calls["pytest"] == calls_after_resume
    event_log = runtime / "evidence/full_pytest_safe_fixture_replay_events.jsonl"
    assert event_log.is_file()
    assert hashlib.sha256(event_log.read_bytes()).hexdigest() == hashlib.sha256(
        b""
    ).hexdigest()


def test_fixed_child_pytest_plugin_blocks_registration_and_attests_attempt(
    tmp_path: Path,
) -> None:
    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    attestation = tmp_path / "child-attestation.json"
    event_log = tmp_path / "safe-fixture-replay-events.jsonl"
    marker = tmp_path / "formal-stage2-marker.json"
    formal_python = Path(
        "/home/lj/.local/share/degen-lio-micromamba/envs/"
        "degen-lio-zprm-py311/bin/python3.11"
    ).resolve(strict=True)
    marker_value = finalizer._formal_safe_replay_marker(
        repository=ROOT,
        python=formal_python,
        git_head="a" * 40,
        event_log=event_log,
    )
    atomic_write_json(marker, marker_value, overwrite=False)
    test_file = tmp_path / "test_child_guard.py"
    test_file.write_text(
        "import subprocess\n"
        "import pytest\n"
        "from phase_a_harness.real_data_preparation.guard import "
        "RegistrationForbiddenError\n\n"
        "def test_child_guard_is_live():\n"
        "    with pytest.raises(RegistrationForbiddenError):\n"
        "        subprocess.run(['pcl_point_to_plane_cli'], check=False)\n"
    )
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", ""),
        "PYTEST_ADDOPTS": "",
        "PYTHONHASHSEED": "0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONSTARTUP": "",
        "PYTHONWARNINGS": "",
        "MAMBA_ROOT_PREFIX": "/home/lj/.local/share/degen-lio-micromamba",
        "TZ": "UTC",
        "ZPRM_REAL_DATA_PREP_NO_REGISTRATION": "1",
        "ZPRM_STAGE2_PYTEST_GUARD_ATTESTATION": str(attestation),
        "ZPRM_STAGE2_PYTEST_GUARD_SCAN_ROOT": str(scan_root),
    }
    environment.update(
        formal_stage2_environment(
            marker_path=marker,
            marker_sha256=sha256_file(marker),
        )
    )
    result = subprocess.run(
        [
            str(formal_python),
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            finalizer.CHILD_GUARD_PLUGIN,
            "-c",
            str(ROOT / "pyproject.toml"),
            str(test_file),
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    value = json.loads(attestation.read_text())
    assert set(value) == CHILD_ATTESTATION_FIELDS
    assert value["schema"] == CHILD_ATTESTATION_SCHEMA
    assert value["safe_fixture_replay_event_count"] == 0
    assert value["safe_simulated_materialization_count"] == 0
    assert value["safe_simulated_open3d_materialization_count"] == 0
    assert value["safe_simulated_pcl_materialization_count"] == 0
    assert value["other_registration_process_count"] == 1
    assert value["pcl_cli_invocation_count"] == 1
    assert value["pass"] is False
    with pytest.raises(
        finalizer.BoreasV2Stage2ClosureError,
        match="child NO-ICP attestation differs",
    ):
        finalizer._validate_child_guard_attestation(
            attestation,
            plugin_path=(
                ROOT
                / "src/phase_a_harness/real_data_preparation/"
                "stage2_pytest_no_registration.py"
            ),
            guard_path=ROOT / "src/phase_a_harness/real_data_preparation/guard.py",
            marker_path=marker,
            event_log_path=event_log,
        )


def test_independent_verifier_recomputes_marker_implementations_and_event_chain(
    tmp_path: Path,
) -> None:
    from phase_a_harness.real_data_preparation import (
        boreas_v2_stage2_preparation_verifier as verifier,
    )

    runtime = tmp_path / "runtime"
    evidence = runtime / "evidence"
    evidence.mkdir(parents=True)
    marker = evidence / "full_pytest_safe_fixture_replay_marker.json"
    event_log = evidence / "full_pytest_safe_fixture_replay_events.jsonl"
    formal_python = Path(
        "/home/lj/.local/share/degen-lio-micromamba/envs/"
        "degen-lio-zprm-py311/bin/python3.11"
    ).resolve(strict=True)
    marker_value = finalizer._formal_safe_replay_marker(
        repository=ROOT,
        python=formal_python,
        git_head="a" * 40,
        event_log=event_log,
    )
    marker.write_bytes(canonical_json_bytes(marker_value))
    event_log.write_bytes(b"")
    status = {
        "git_head": "a" * 40,
        "python_executable": str(formal_python),
        "safe_fixture_replay_event_log_relative_path": (
            "evidence/full_pytest_safe_fixture_replay_events.jsonl"
        ),
        "safe_fixture_replay_event_log_sha256": sha256_file(event_log),
        "safe_fixture_replay_marker_relative_path": (
            "evidence/full_pytest_safe_fixture_replay_marker.json"
        ),
        "safe_fixture_replay_marker_sha256": sha256_file(marker),
    }
    _, _, projection = verifier._verify_safe_fixture_replay_evidence(
        runtime=runtime,
        repository=ROOT,
        status=status,
    )
    assert projection["safe_fixture_replay_event_count"] == 0

    forged_marker = dict(marker_value)
    forged_marker["safe_fixture_replay_sha256"] = "f" * 64
    unsigned = dict(forged_marker)
    unsigned.pop("marker_payload_sha256")
    forged_marker["marker_payload_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    marker.write_bytes(canonical_json_bytes(forged_marker))
    status["safe_fixture_replay_marker_sha256"] = sha256_file(marker)
    with pytest.raises(
        verifier.BoreasStage2PreparationVerificationError,
        match="implementation",
    ):
        verifier._verify_safe_fixture_replay_evidence(
            runtime=runtime,
            repository=ROOT,
            status=status,
        )

    marker.write_bytes(canonical_json_bytes(marker_value))
    status["safe_fixture_replay_marker_sha256"] = sha256_file(marker)
    event_log.write_bytes(b"{}\n")
    status["safe_fixture_replay_event_log_sha256"] = sha256_file(event_log)
    with pytest.raises(
        verifier.BoreasStage2PreparationVerificationError,
        match="event chain",
    ):
        verifier._verify_safe_fixture_replay_evidence(
            runtime=runtime,
            repository=ROOT,
            status=status,
        )


def test_publication_only_descendant_git_proof_accepts_exact_commit_and_rejects_source_edit(
    tmp_path: Path,
) -> None:
    from phase_a_harness.real_data_preparation import (
        boreas_v2_stage2_preparation_verifier as verifier,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Release Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "release@example.invalid"],
        cwd=repository,
        check=True,
    )
    (repository / "source.txt").write_text("tested source\n")
    subprocess.run(["git", "add", "source.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "tested"], cwd=repository, check=True)
    source_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    source_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repository, text=True
    ).strip()
    inventory = hashlib.sha256(
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "-z", source_head],
            cwd=repository,
        )
    ).hexdigest()
    closure_root = repository / "frozen_assets/boreas_v2_stage2_preparation"
    closure_root.mkdir(parents=True)
    for name in verifier.REQUIRED_FILES:
        (closure_root / name).write_bytes(f"closure:{name}\n".encode())
    subprocess.run(["git", "add", "frozen_assets"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "publish"], cwd=repository, check=True)
    verifier._verify_publication_git_state(
        repository=repository,
        closure_root=closure_root,
        source_head=source_head,
        source_tree=source_tree,
        tracked_file_inventory_sha256=inventory,
    )
    (repository / "source.txt").write_text("changed source\n")
    subprocess.run(["git", "add", "source.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "source edit"], cwd=repository, check=True)
    with pytest.raises(
        verifier.BoreasStage2PreparationVerificationError,
        match="pre-existing source path",
    ):
        verifier._verify_publication_git_state(
            repository=repository,
            closure_root=closure_root,
            source_head=source_head,
            source_tree=source_tree,
            tracked_file_inventory_sha256=inventory,
        )


def test_authorization_git_provenance_requires_exact_branch_commit_and_tree(
    tmp_path: Path,
) -> None:
    from phase_a_harness.real_data_preparation import (
        boreas_v2_stage2_preparation_verifier as verifier,
    )

    repository = tmp_path / "repository"
    runtime = tmp_path / "runtime"
    (runtime / "evidence").mkdir(parents=True)
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Auth Test"], cwd=repository, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "auth@example.invalid"],
        cwd=repository,
        check=True,
    )
    (repository / "source.txt").write_text("source\n")
    subprocess.run(["git", "add", "source.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "source"], cwd=repository, check=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repository, text=True
    ).strip()
    atomic_write_json(
        runtime / "evidence/full_test_status.json",
        {"git_head": head, "git_tree": tree},
    )
    authorization = {
        "branch": "run/boreas-v2-stage2-data-preparation",
        "commit": head,
    }
    verifier._verify_authorization_git_provenance(
        repository=repository, runtime=runtime, authorization=authorization
    )
    with pytest.raises(
        verifier.BoreasStage2PreparationVerificationError, match="branch"
    ):
        verifier._verify_authorization_git_provenance(
            repository=repository,
            runtime=runtime,
            authorization={**authorization, "branch": "forged"},
        )
    with pytest.raises(
        verifier.BoreasStage2PreparationVerificationError,
        match="authorization/formal-test Git commit",
    ):
        verifier._verify_authorization_git_provenance(
            repository=repository,
            runtime=runtime,
            authorization={**authorization, "commit": "f" * 40},
        )
    atomic_write_json(
        runtime / "evidence/full_test_status.json",
        {"git_head": head, "git_tree": "f" * 40},
        overwrite=True,
    )
    with pytest.raises(
        verifier.BoreasStage2PreparationVerificationError, match="Git tree"
    ):
        verifier._verify_authorization_git_provenance(
            repository=repository, runtime=runtime, authorization=authorization
        )
