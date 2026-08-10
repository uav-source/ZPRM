from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from phase_a_harness.real_data_preparation.guard import (
    NoRegistrationGuard,
    RegistrationForbiddenError,
    assert_preparation_sources_are_safe,
)


@pytest.mark.parametrize(
    "command",
    (
        ["genz_icp", "--help"],
        ["GENZ-ICP", "--help"],
        ["rtabmap", "--version"],
        ["ros2", "run", "rtabmap_ros", "rtabmap"],
    ),
)
def test_cavers_runtime_guard_blocks_explicit_registration_processes(
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        with pytest.raises(RegistrationForbiddenError, match="forbidden registration process"):
            subprocess.run(command, check=True)
        report = guard.attestation(Path("/definitely/absent"))
    assert report["other_registration_process_count"] == 1
    assert report["estimated_transform_count"] == 0
    assert report["pass"] is False


@pytest.mark.parametrize(
    "source",
    (
        "import genz_icp\n",
        "from genz_icp import register\n",
        "import rtabmap_ros\n",
        "from safe_wrapper import rtabmap\n",
    ),
)
def test_cavers_static_source_audit_rejects_registration_imports(
    tmp_path: Path,
    source: str,
) -> None:
    (tmp_path / "candidate.py").write_text(source, encoding="utf-8")
    with pytest.raises(RegistrationForbiddenError, match="unsafe preparation imports"):
        assert_preparation_sources_are_safe(tmp_path)


def test_attestation_detects_suspicious_result_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    (tmp_path / "estimated_pose.json").write_text('{"status": "unexpected"}\n', encoding="utf-8")
    with NoRegistrationGuard() as guard:
        report = guard.attestation(tmp_path)
    assert report["estimated_transform_file_count"] == 1
    assert report["estimated_transform_count"] == 1
    assert report["estimated_transform_evidence"] == [str(tmp_path / "estimated_pose.json")]
    assert report["pass"] is False


def test_attestation_detects_nested_json_and_csv_result_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    (tmp_path / "audit.json").write_text(
        json.dumps({"rows": [{"final_transform_4x4": [[1, 0], [0, 1]]}]}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.csv").write_text(
        "snapshot_id,registration_error_m\nsnapshot-1,0.1\n",
        encoding="utf-8",
    )
    with NoRegistrationGuard() as guard:
        report = guard.attestation(tmp_path)
    assert report["estimated_transform_file_count"] == 0
    assert report["estimated_transform_count"] == 2
    assert any("final_transform_4x4" in row for row in report["estimated_transform_evidence"])
    assert any("registration_error_m" in row for row in report["estimated_transform_evidence"])
    assert report["pass"] is False


def test_attestation_rejects_nonzero_declared_estimated_transform_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    (tmp_path / "declared_counts.json").write_text(
        json.dumps({"estimated_transform_count": 1}) + "\n",
        encoding="utf-8",
    )
    with NoRegistrationGuard() as guard:
        report = guard.attestation(tmp_path)
    assert report["estimated_transform_count"] == 1
    assert report["estimated_transform_evidence"] == [
        f"{tmp_path / 'declared_counts.json'}#$.estimated_transform_count"
    ]
    assert report["pass"] is False


def test_attestation_allows_nominal_transform_and_zero_count_audit_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    (tmp_path / "cavers_transform_chain_manifest.json").write_text(
        json.dumps(
            {
                "T_rig_lidar": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "registration_output_consulted": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "NO_ICP_ATTESTATION.json").write_text(
        json.dumps(
            {
                "estimated_transform_count": 0,
                "estimated_transform_file_count": 0,
                "registration_execution_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with NoRegistrationGuard() as guard:
        report = guard.attestation(tmp_path)
    assert report["estimated_transform_count"] == 0
    assert report["structured_result_scan_error_count"] == 0
    assert report["pass"] is True


def test_attestation_fails_closed_on_unreadable_structured_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    (tmp_path / "broken.json").write_text("{not-json}\n", encoding="utf-8")
    with NoRegistrationGuard() as guard:
        report = guard.attestation(tmp_path)
    assert report["estimated_transform_count"] == 0
    assert report["structured_result_scan_error_count"] == 1
    assert report["pass"] is False
