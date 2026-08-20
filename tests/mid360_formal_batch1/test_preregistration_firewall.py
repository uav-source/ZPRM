from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mid360_formal_batch1.preregistration_firewall import (
    EXPECTED_BACKEND_CONTRACT_SHA256,
    NO_FORMAL_REGISTRATION_ENV,
    NoRegistrationGuard,
    PreregistrationFirewallError,
    RegistrationForbiddenError,
    assert_static_scope_safe,
    build_no_icp_attestation,
    snapshot_registration_processes,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def _scope_repository(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "repository"
    experiment = root / "experiments/mid360_formal_batch1"
    tools = root / "tools/mid360_formal_batch1"
    experiment.mkdir(parents=True)
    tools.mkdir(parents=True)
    (experiment / "protocol.py").write_text(
        '"""Text may mention registration_icp, T_est, fitness, and final residual."""\n',
        encoding="utf-8",
    )
    (experiment / "ingest.py").write_text(source, encoding="utf-8")
    (tools / "audit.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def _attestation_repository(tmp_path: Path) -> Path:
    root = _scope_repository(tmp_path, "VALUE = 1\n")
    contract = root / "frozen_assets/backend_parameter_contract.json"
    contract.parent.mkdir(parents=True)
    shutil.copy2(
        REPOSITORY / "frozen_assets/backend_parameter_contract.json", contract
    )
    return root


def _fake_open3d() -> SimpleNamespace:
    registration = SimpleNamespace(
        registration_icp=lambda *args, **kwargs: "forbidden",
        registration_generalized_icp=lambda *args, **kwargs: "forbidden",
        harmless_constant=1,
    )
    return SimpleNamespace(pipelines=SimpleNamespace(registration=registration))


def test_real_fmb1_preparation_scope_passes_static_firewall() -> None:
    report = assert_static_scope_safe(REPOSITORY)
    assert report["pass"] is True
    assert "experiments/mid360_formal_batch1/protocol.py" in report["scanned_files"]
    assert "tools/mid360_formal_batch1/freeze_batch.py" in report[
        "excluded_authority_files"
    ]


def test_protocol_text_is_ignored_but_authorization_import_is_rejected(
    tmp_path: Path,
) -> None:
    safe = _scope_repository(
        tmp_path,
        "from experiments.mid360_formal_batch1.protocol import geometry_class\n"
        'NOTE = "registration_icp and T_est are forbidden here"\n',
    )
    assert assert_static_scope_safe(safe)["pass"] is True
    unsafe_path = safe / "experiments/mid360_formal_batch1/ingest.py"
    unsafe_path.write_text(
        "from experiments.mid360_formal_batch1.protocol import formal_icp_authorized\n",
        encoding="utf-8",
    )
    with pytest.raises(PreregistrationFirewallError, match="FORBIDDEN_PROTOCOL_SYMBOL"):
        assert_static_scope_safe(safe)


def test_protocol_explanatory_text_is_allowed_but_backend_import_is_not(
    tmp_path: Path,
) -> None:
    root = _scope_repository(tmp_path, "VALUE = 1\n")
    protocol = root / "experiments/mid360_formal_batch1/protocol.py"
    protocol.write_text(
        '"""registration_icp T_est final residual turnover fitness solver result"""\n'
        "import open3d\n",
        encoding="utf-8",
    )
    with pytest.raises(PreregistrationFirewallError, match="FORBIDDEN_IMPORT"):
        assert_static_scope_safe(root)


@pytest.mark.parametrize(
    "source,marker",
    [
        ("import open3d\n", "FORBIDDEN_IMPORT"),
        ("from phase_a_harness.pcl_backend import run_pcl_point_to_plane\n", "FORBIDDEN_IMPORT"),
        ("def run(x):\n    return x.registration_icp()\n", "FORBIDDEN_CALL"),
        ("RESULT = {'T_est': None}\n", "FORBIDDEN_RESULT_FIELD"),
        ("import subprocess\nsubprocess.run(['pcl_point_to_plane_cli'])\n", "FORBIDDEN_PROCESS_LITERAL"),
        ("def run(x):\n    return getattr(x, 'fitness')\n", "DYNAMIC_FORBIDDEN_SYMBOL"),
    ],
)
def test_static_firewall_rejects_backend_and_result_tamper(
    tmp_path: Path, source: str, marker: str
) -> None:
    root = _scope_repository(tmp_path, source)
    with pytest.raises(PreregistrationFirewallError, match=marker):
        assert_static_scope_safe(root)


def test_guard_requires_explicit_environment_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(NO_FORMAL_REGISTRATION_ENV, raising=False)
    with pytest.raises(RegistrationForbiddenError, match=NO_FORMAL_REGISTRATION_ENV):
        NoRegistrationGuard().__enter__()


def test_guard_allows_read_only_nonregistration_process_and_restores_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(NO_FORMAL_REGISTRATION_ENV, "true")
    original_run = subprocess.run
    original_popen = subprocess.Popen
    original_system = os.system
    with NoRegistrationGuard() as guard:
        completed = subprocess.run(
            ["/bin/true", "rosbag", "info"], check=True, capture_output=True
        )
        assert completed.returncode == 0
        attestation_name = subprocess.run(
            ["/bin/true", "NO_ICP_ATTESTATION.json"],
            check=True,
            capture_output=True,
        )
        assert attestation_name.returncode == 0
        assert guard.blocked_process_attempt_count == 0
    assert subprocess.run is original_run
    assert subprocess.Popen is original_popen
    assert os.system is original_system


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: subprocess.Popen(["pcl_point_to_plane_cli"]),
        lambda: subprocess.run(["pcl_point_to_plane_cli"]),
        lambda: subprocess.check_call(["pcl_point_to_plane_cli"]),
        lambda: subprocess.check_output(["pcl_point_to_plane_cli"]),
        lambda: os.system("pcl_point_to_plane_cli"),
    ],
)
def test_guard_blocks_every_required_process_entrypoint(
    monkeypatch: pytest.MonkeyPatch, invoke
) -> None:
    monkeypatch.setenv(NO_FORMAL_REGISTRATION_ENV, "1")
    with NoRegistrationGuard() as guard:
        with pytest.raises(RegistrationForbiddenError):
            invoke()
        assert guard.pcl_cli_invocation_count == 1
        assert guard.blocked_process_attempt_count == 1


def test_guard_blocks_loaded_open3d_registration_without_importing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(NO_FORMAL_REGISTRATION_ENV, "true")
    module = _fake_open3d()
    original = module.pipelines.registration.registration_icp
    with NoRegistrationGuard(open3d_module=module) as guard:
        with pytest.raises(RegistrationForbiddenError):
            module.pipelines.registration.registration_icp(None, None)
        assert guard.open3d_registration_call_count == 1
    assert module.pipelines.registration.registration_icp is original


def test_exited_guard_report_is_json_safe_and_has_four_zero_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(NO_FORMAL_REGISTRATION_ENV, "true")
    with NoRegistrationGuard(proc_root=tmp_path / "empty-proc") as guard:
        pass
    report = guard.report()
    json.dumps(report, allow_nan=False)
    assert report["guard_was_activated"] is True
    assert report["guard_active_at_report"] is False
    for field in (
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "formal_trial_count",
    ):
        assert report[field] == 0


def test_proc_snapshot_detects_only_obvious_registration_processes(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    normal = proc / "100"
    forbidden = proc / "101"
    misleading = proc / "102"
    for directory in (normal, forbidden, misleading):
        directory.mkdir(parents=True)
    (normal / "cmdline").write_bytes(b"python\x00worker.py\x00")
    (normal / "comm").write_text("python\n", encoding="utf-8")
    (forbidden / "cmdline").write_bytes(b"/opt/bin/pcl_point_to_plane_cli\x00")
    (forbidden / "comm").write_text("pcl_point_to_plane_cli\n", encoding="utf-8")
    (misleading / "cmdline").write_bytes(b"pytest\x00test_preregistration_firewall.py\x00")
    (misleading / "comm").write_text("pytest\n", encoding="utf-8")
    rows = snapshot_registration_processes(proc)
    assert [row["pid"] for row in rows] == [101]
    assert rows[0]["registration_process_kind"] == "pcl"


def test_clean_attestation_has_four_required_zero_counts_and_no_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _attestation_repository(tmp_path)
    static = assert_static_scope_safe(root)
    monkeypatch.setenv(NO_FORMAL_REGISTRATION_ENV, "true")
    with NoRegistrationGuard(proc_root=tmp_path / "empty-proc") as guard:
        attestation = build_no_icp_attestation(
            root, guard, static_scope_report=static
        )
    assert attestation["pass"] is True
    assert attestation["status"] == "PASS"
    assert attestation["backend_parameter_contract_sha256"] == (
        EXPECTED_BACKEND_CONTRACT_SHA256
    )
    assert attestation["FORMAL_REGISTRATION_AUTHORIZED"] is False
    for field in (
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "formal_trial_count",
    ):
        assert attestation[field] == 0


def test_attestation_fails_on_blocked_attempt_or_backend_contract_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _attestation_repository(tmp_path)
    static = assert_static_scope_safe(root)
    contract = root / "frozen_assets/backend_parameter_contract.json"
    value = json.loads(contract.read_text(encoding="utf-8"))
    value["tampered"] = True
    contract.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setenv(NO_FORMAL_REGISTRATION_ENV, "true")
    with NoRegistrationGuard(proc_root=tmp_path / "empty-proc") as guard:
        with pytest.raises(RegistrationForbiddenError):
            subprocess.run(["pcl_point_to_plane_cli"])
        attestation = build_no_icp_attestation(
            root, guard, static_scope_report=static
        )
    assert attestation["pass"] is False
    assert attestation["backend_parameter_contract_unchanged"] is False
    assert attestation["pcl_cli_invocation_count"] == 1
    assert attestation["other_registration_process_count"] == 1


def test_attestation_fails_when_proc_snapshot_already_contains_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _attestation_repository(tmp_path)
    static = assert_static_scope_safe(root)
    proc = tmp_path / "proc"
    process = proc / "123"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"pcl_point_to_plane_cli\x00")
    (process / "comm").write_text("pcl_point_to_plane_cli\n", encoding="utf-8")
    monkeypatch.setenv(NO_FORMAL_REGISTRATION_ENV, "true")
    with NoRegistrationGuard(proc_root=proc) as guard:
        attestation = build_no_icp_attestation(
            root, guard, static_scope_report=static
        )
    assert attestation["pass"] is False
    assert attestation["pcl_cli_invocation_count"] == 1
    assert attestation["other_registration_process_count"] == 1


def test_attestation_requires_an_activated_guard(tmp_path: Path) -> None:
    root = _attestation_repository(tmp_path)
    with pytest.raises(PreregistrationFirewallError, match="must be activated"):
        build_no_icp_attestation(root, NoRegistrationGuard())


def test_attestation_aggregates_multiple_exited_stage_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _attestation_repository(tmp_path)
    static = assert_static_scope_safe(root)
    monkeypatch.setenv(NO_FORMAL_REGISTRATION_ENV, "true")
    reports = []
    for index in range(2):
        with NoRegistrationGuard(proc_root=tmp_path / f"empty-proc-{index}") as guard:
            pass
        reports.append(guard.report())
    attestation = build_no_icp_attestation(
        root,
        stage_reports=reports,
        static_report=static,
    )
    assert attestation["pass"] is True
    assert attestation["guard_stage_count"] == 2
    assert attestation["formal_trial_count"] == 0


def test_attestation_rejects_fabricated_stage_report(tmp_path: Path) -> None:
    root = _attestation_repository(tmp_path)
    fabricated = {
        "guard_was_activated": False,
        **{field: 0 for field in (
            "open3d_registration_call_count",
            "pcl_cli_invocation_count",
            "other_registration_process_count",
            "formal_trial_count",
        )},
    }
    with pytest.raises(PreregistrationFirewallError, match="not produced"):
        build_no_icp_attestation(root, stage_reports=[fabricated])
