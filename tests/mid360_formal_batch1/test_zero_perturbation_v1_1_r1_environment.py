from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_environment import (
    EXPECTED_VERSIONS,
    R1EnvironmentError,
    collect_environment_manifest,
    verify_environment_manifest,
)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    contract = root / "frozen_assets/backend_parameter_contract.json"
    contract.parent.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "frozen_assets/backend_parameter_contract.json",
        contract,
    )
    pcl = root / "bin/pcl_point_to_plane_cli"
    pcl.parent.mkdir(parents=True)
    pcl.write_bytes(b"pcl-test")
    pcl.chmod(0o755)
    return root


def _collect(root: Path) -> dict[str, object]:
    return collect_environment_manifest(
        root,
        observed_versions={key: value for key, value in EXPECTED_VERSIONS.items() if key != "pcl"},
        pcl_version_text="1.15.1",
        ldd_text="libpcl_common.so => /test/libpcl_common.so",
    )


def test_exact_environment_qualifies_without_backend_execution(repository: Path) -> None:
    payload = _collect(repository)
    assert payload["qualification_pass"] is True
    assert payload["open3d_import_count"] == 0
    assert payload["pcl_registration_call_count"] == 0
    report = verify_environment_manifest(payload, repository, remeasure_versions=False)
    assert report["pass"] is True
    assert report["backend_calls"] == 0


@pytest.mark.parametrize("key", ["python", "numpy", "scipy", "open3d", "pcl"])
def test_version_drift_fails_qualification(repository: Path, key: str) -> None:
    versions = {name: value for name, value in EXPECTED_VERSIONS.items() if name != "pcl"}
    if key == "pcl":
        pcl_version = "1.15.0"
    else:
        versions[key] = "WRONG"
        pcl_version = "1.15.1"
    payload = collect_environment_manifest(
        repository, observed_versions=versions, pcl_version_text=pcl_version,
        ldd_text="libpcl_common.so => /test/libpcl_common.so",
    )
    assert payload["qualification_pass"] is False
    with pytest.raises(R1EnvironmentError):
        verify_environment_manifest(payload, repository, remeasure_versions=False)


def test_missing_ldd_dependency_fails(repository: Path) -> None:
    payload = collect_environment_manifest(
        repository,
        observed_versions={key: value for key, value in EXPECTED_VERSIONS.items() if key != "pcl"},
        pcl_version_text="1.15.1", ldd_text="libpcl_common.so => not found",
    )
    assert payload["qualification_pass"] is False


def test_pcl_byte_tamper_fails(repository: Path) -> None:
    payload = _collect(repository)
    Path(payload["pcl_cli"]["path"]).write_bytes(b"changed")
    with pytest.raises(R1EnvironmentError):
        verify_environment_manifest(payload, repository, remeasure_versions=False)

