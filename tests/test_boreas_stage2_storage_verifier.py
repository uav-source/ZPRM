from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from phase_a_harness.real_data_preparation import (
    boreas_stage2_storage_optimization as producer,
)
from phase_a_harness.real_data_preparation.io import canonical_json_bytes
from phase_a_harness.real_data_preparation import (
    boreas_stage2_storage_optimization_verifier as verifier,
)


REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def valid_storage_closure(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("boreas-stage2-storage-verifier")
    data_root = root / "boreas_stage1_v1"
    data_root.mkdir()
    closure = root / "closure"
    junit = root / "source-only.xml"
    testcases = "".join(
        f'<testcase classname="fixture" name="test_{index}">'
        + ("<skipped />" if index < 10 else "")
        + "</testcase>"
        for index in range(811)
    )
    junit.write_text(
        '<testsuites><testsuite name="pytest" tests="811" failures="0" '
        f'errors="0" skipped="10">{testcases}</testsuite></testsuites>',
        encoding="utf-8",
    )
    names = (
        "ZPRM_REAL_DATA_PREP_NO_REGISTRATION",
        "ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD",
    )
    before = {name: os.environ.get(name) for name in names}
    original_repository_gate = producer.inspect_repository_gate
    try:
        for name in names:
            os.environ[name] = "1"
        recorded_gate = original_repository_gate(REPOSITORY, require_clean=False)
        recorded_gate["worktree_clean"] = True
        producer.inspect_repository_gate = lambda *_args, **_kwargs: recorded_gate
        producer.build_boreas_stage2_storage_optimization(
            repository=REPOSITORY,
            data_root=data_root,
            runtime_root=closure,
            pytest_junit_xml=junit,
            require_clean_worktree=False,
        )
    finally:
        producer.inspect_repository_gate = original_repository_gate
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return closure, data_root


def test_independent_storage_verifier_accepts_valid_closure(
    valid_storage_closure: tuple[Path, Path],
) -> None:
    closure, data_root = valid_storage_closure
    report = verifier.verify_boreas_stage2_storage_optimization(
        repository=REPOSITORY,
        runtime_root=closure,
        data_root=data_root,
    )
    assert report["BOREAS_V2_STAGE2_STORAGE_VERIFICATION_PASS"] is True
    assert report["closure_file_count"] == 24
    assert report["allowlist_object_count"] == 20_061
    assert report["allowlist_remote_bytes"] == 104_158_637_472
    assert report["physical_target_map_copy_count"] == 1
    assert report["planned_snapshot_count"] == 100
    assert report["registration_execution_count"] == 0
    assert report["lidar_payload_download_count"] == 0


def test_verifier_has_no_producer_or_planner_import() -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("stage2_storage_planner" in name for name in imported)
    assert not any("boreas_stage2_storage_optimization" in name for name in imported)


def test_eleven_resigned_semantic_tampers_all_fail(
    valid_storage_closure: tuple[Path, Path],
) -> None:
    closure, data_root = valid_storage_closure
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/test_boreas_v2_stage2_storage_tamper.py"),
            "--repository-root",
            str(REPOSITORY),
            "--data-root",
            str(data_root),
            "--frozen-root",
            str(closure),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(completed.stdout)
    assert report["tamper_case_count"] == 11
    assert report["all_tamper_cases_rejected"] is True
    assert report["lidar_bin_test_file_created"] is False
    assert report["registration_execution_count"] == 0
    assert all(row["nonzero_exit"] for row in report["results"])
    assert {row["case"] for row in report["results"]} == {
        "primary_pair_changed",
        "object_count_changed",
        "remote_bytes_changed",
        "target_copy_count_greater_than_one",
        "open3d_pcl_source_sha_diverged",
        "snapshot_count_changed",
        "voxel_size_configured_by_storage_planner",
        "no_lidar_attestation_changed",
        "disk_budget_arithmetic_changed",
        "low_disk_watermark_changed",
        "stage1_manifest_sha_binding_changed",
    }


def test_no_icp_nested_static_audit_schema_is_closed(
    valid_storage_closure: tuple[Path, Path], tmp_path: Path
) -> None:
    closure, _data_root = valid_storage_closure
    candidate = tmp_path / "static-audit-tamper"
    shutil.copytree(closure, candidate)
    path = candidate / "NO_ICP_ATTESTATION.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["static_source_audit"]["unexpected_authority"] = True
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(
        verifier.BoreasStage2StorageVerificationError,
        match="static source audit field set",
    ):
        verifier._verify_attestations(candidate)


def test_no_icp_zero_counts_require_empty_evidence_lists(
    valid_storage_closure: tuple[Path, Path], tmp_path: Path
) -> None:
    closure, _data_root = valid_storage_closure
    candidate = tmp_path / "nonempty-no-icp-evidence"
    shutil.copytree(closure, candidate)
    path = candidate / "NO_ICP_ATTESTATION.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["estimated_transform_evidence"] = ["contradicts-zero-count"]
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(
        verifier.BoreasStage2StorageVerificationError,
        match="empty evidence list",
    ):
        verifier._verify_attestations(candidate)
